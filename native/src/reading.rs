use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, PoisonError, RwLock};

use crate::lists::Block;
use crate::postings::{Shape, score};
use crate::ranges::Ranges;
use crate::segment::lengths_read;

const OUTSIDE: &str = "the keyword index holds chunks outside its segments";
const LENGTHLESS: &str = "the keyword index holds words in passages of no length";
const MISREAD: &str = "the reads of a search gave other than it asked for";
const UNPLACED: &str = "a block holds fewer postings than its ranges place in it";

pub type LengthBlock = Vec<u32>;
pub type Failure = Box<dyn Error + Send + Sync>;
pub type Scored = (Vec<i64>, Vec<f64>);

#[derive(Default)]
pub struct LengthsTable {
    held: RwLock<HashMap<i64, Arc<LengthBlock>>>,
}

pub struct Segment {
    pub first: i64,
    pub last: i64,
    pub dead: Vec<i64>,
    pub lengths: Arc<LengthsTable>,
}

pub struct Part {
    pub segment: usize,
    pub ranges: Arc<Ranges>,
    pub slack: f64,
}

pub struct Reached {
    pub threshold: f64,
    pub held: usize,
}

pub trait Reads {
    fn blocks(&mut self, wanted: &[(usize, usize)]) -> Result<Vec<Arc<Block>>, Failure>;
    fn lengths(&mut self, wanted: &[(usize, i64)]) -> Result<Vec<Arc<LengthBlock>>, Failure>;
    fn offered(&mut self, places: &[i64], scores: &[f64]) -> Result<Reached, Failure>;
    fn expired(&mut self) -> Result<bool, Failure>;
}

pub struct Store {
    pub segments: Vec<Segment>,
    pub parts: Vec<Part>,
    pub shape: Shape,
    pub average: f64,
}

pub struct Reader<'a> {
    pub store: &'a Store,
    pub reads: &'a mut dyn Reads,
    blocks: Vec<Vec<Option<Arc<Block>>>>,
    lengths: Vec<HashMap<i64, Arc<LengthBlock>>>,
}

pub struct Picked {
    pub places: Vec<i64>,
    pub counts: Vec<i64>,
}

impl LengthsTable {
    fn got(&self, number: i64) -> Option<Arc<LengthBlock>> {
        let held = self.held.read().unwrap_or_else(PoisonError::into_inner);
        held.get(&number).cloned()
    }

    fn kept(&self, number: i64, block: &Arc<LengthBlock>) {
        let mut held = self.held.write().unwrap_or_else(PoisonError::into_inner);
        held.insert(number, Arc::clone(block));
    }

    pub fn held_bytes(&self) -> usize {
        let held = self.held.read().unwrap_or_else(PoisonError::into_inner);
        held.values().map(|block| size_of_val(block.as_slice())).sum()
    }
}

pub fn length_blocks<R: AsRef<[u8]> + Sync>(rows: &[R], shift: u32) -> Result<Vec<Arc<LengthBlock>>, String> {
    let found = lengths_read(rows, shift)?;
    let blocks = found.chunks_exact(size_of::<u32>() << shift).map(|block| {
        let lengths = block.as_chunks::<4>().0.iter().map(|each| u32::from_le_bytes(*each));
        Arc::new(lengths.collect())
    });
    Ok(blocks.collect())
}

pub fn chosen(ids: &[i64], wanted: &[i64]) -> Vec<usize> {
    let (mut found, mut at) = (Vec::new(), 0usize);
    for &id in wanted {
        at += ids[at..].partition_point(|&each| each < id);
        if ids.get(at) == Some(&id) {
            found.push(at);
        }
    }
    found
}

pub fn live(dead: &[i64], places: Vec<i64>, counts: Vec<i64>) -> (Vec<i64>, Vec<i64>) {
    if dead.is_empty() {
        return (places, counts);
    }
    let (mut kept, mut times, mut at) = (
        Vec::with_capacity(places.len()),
        Vec::with_capacity(counts.len()),
        0usize,
    );
    for (place, count) in places.into_iter().zip(counts) {
        at += dead[at..].partition_point(|&each| each < place);
        if dead.get(at) != Some(&place) {
            kept.push(place);
            times.push(count);
        }
    }
    (kept, times)
}

impl Store {
    fn segment_of(&self, place: i64, near: usize) -> Result<usize, String> {
        let inside = |at: usize| {
            self.segments
                .get(at)
                .is_some_and(|held| held.first <= place && place <= held.last)
        };
        if inside(near) {
            return Ok(near);
        }
        let at = self.segments.partition_point(|held| held.first <= place).checked_sub(1);
        at.filter(|&at| inside(at)).ok_or_else(|| OUTSIDE.to_owned())
    }

    fn numbers(&self, part: &Part, ranges: &[usize]) -> Vec<usize> {
        let block = self.shape.block();
        let mut found: Vec<usize> = Vec::new();
        for &at in ranges {
            let (start, end) = (part.ranges.start(at), part.ranges.ends()[at]);
            for number in start / block..=(end - 1) / block {
                if found.last() != Some(&number) {
                    found.push(number);
                }
            }
        }
        found
    }
}

impl<'a> Reader<'a> {
    pub fn new(store: &'a Store, reads: &'a mut dyn Reads) -> Self {
        let block = store.shape.block();
        let blocks = store
            .parts
            .iter()
            .map(|part| vec![None; part.ranges.total().div_ceil(block)]);
        Self {
            store,
            reads,
            blocks: blocks.collect(),
            lengths: store.segments.iter().map(|_| HashMap::new()).collect(),
        }
    }

    fn read_blocks(&mut self, wanted: &[(usize, Vec<usize>)]) -> Result<(), Failure> {
        let mut missing: Vec<(usize, usize)> = Vec::new();
        for (part, ranges) in wanted {
            let numbers = self.store.numbers(&self.store.parts[*part], ranges);
            let held = &self.blocks[*part];
            missing.extend(
                numbers
                    .into_iter()
                    .filter(|&number| held[number].is_none())
                    .map(|number| (*part, number)),
            );
        }
        if missing.is_empty() {
            return Ok(());
        }
        let found = self.reads.blocks(&missing)?;
        if found.len() != missing.len() {
            return Err(MISREAD.into());
        }
        for ((part, number), block) in missing.into_iter().zip(found) {
            self.blocks[part][number] = Some(block);
        }
        Ok(())
    }

    fn gathered(&self, part: usize, ranges: &[usize]) -> Result<Picked, String> {
        let (held, block) = (&self.store.parts[part], self.store.shape.block());
        let (mut places, mut counts) = (Vec::new(), Vec::new());
        for &at in ranges {
            for index in held.ranges.start(at)..held.ranges.ends()[at] {
                let (found, offset) = (self.blocks[part][index / block].as_ref(), index % block);
                let posting = found.and_then(|found| found.ids.get(offset).zip(found.counts.get(offset)));
                let (place, count) = posting.ok_or_else(|| UNPLACED.to_owned())?;
                places.push(*place);
                counts.push(*count);
            }
        }
        let (places, counts) = live(&self.store.segments[held.segment].dead, places, counts);
        Ok(Picked { places, counts })
    }

    pub fn postings(&mut self, wanted: &[(usize, Vec<usize>)]) -> Result<Vec<Picked>, Failure> {
        self.read_blocks(wanted)?;
        let found: Result<Vec<Picked>, String> = wanted
            .iter()
            .map(|(part, ranges)| self.gathered(*part, ranges))
            .collect();
        Ok(found?)
    }

    fn placed(&self, places: &[i64]) -> Result<Vec<(usize, i64)>, String> {
        let shift = self.store.shape.lengths_shift();
        let (mut found, mut near): (Vec<(usize, i64)>, usize) = (Vec::new(), 0);
        for &place in places {
            near = self.store.segment_of(place, near)?;
            let key = (near, place >> shift);
            if found.last() != Some(&key) {
                found.push(key);
            }
        }
        Ok(found)
    }

    fn read_lengths(&mut self, each: &[Picked]) -> Result<(), Failure> {
        let mut missing: Vec<(usize, i64)> = Vec::new();
        for gathered in each {
            for (segment, number) in self.placed(&gathered.places)? {
                if self.lengths[segment].contains_key(&number) {
                    continue;
                }
                match self.store.segments[segment].lengths.got(number) {
                    Some(block) => {
                        self.lengths[segment].insert(number, block);
                    }
                    None => missing.push((segment, number)),
                }
            }
        }
        missing.sort_unstable();
        missing.dedup();
        if missing.is_empty() {
            return Ok(());
        }
        let found = self.reads.lengths(&missing)?;
        let size = 1usize << self.store.shape.lengths_shift();
        if found.len() != missing.len() || found.iter().any(|block| block.len() != size) {
            return Err(MISREAD.into());
        }
        for ((segment, number), block) in missing.into_iter().zip(found) {
            self.store.segments[segment].lengths.kept(number, &block);
            self.lengths[segment].insert(number, block);
        }
        Ok(())
    }

    fn scored(&self, gathered: Picked) -> Result<Scored, String> {
        if gathered.places.is_empty() {
            return Ok((gathered.places, Vec::new()));
        }
        let average = self.store.average;
        if average <= 0.0 {
            return Err(LENGTHLESS.to_owned());
        }
        let (shift, mut near) = (self.store.shape.lengths_shift(), 0usize);
        let mask = (1i64 << shift) - 1;
        let mut scores = Vec::with_capacity(gathered.places.len());
        for (&place, &count) in gathered.places.iter().zip(&gathered.counts) {
            near = self.store.segment_of(place, near)?;
            let block = &self.lengths[near][&(place >> shift)];
            scores.push(score(count, block[usize::try_from(place & mask).unwrap_or(0)], average));
        }
        Ok((gathered.places, scores))
    }

    pub fn scores(&mut self, each: Vec<Picked>) -> Result<Vec<Scored>, Failure> {
        self.read_lengths(&each)?;
        let found: Result<Vec<Scored>, String> = each.into_iter().map(|gathered| self.scored(gathered)).collect();
        Ok(found?)
    }
}
