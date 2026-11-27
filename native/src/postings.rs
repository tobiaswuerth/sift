use std::sync::Arc;

use rayon::prelude::*;

use crate::lists::{Coded, Lists, list};
use crate::numbers::narrow;
use crate::packed::deflated;

const K1: f64 = 1.2;
const B: f64 = 0.75;
const TOP: f64 = K1 + 1.0;
const STEPS: f64 = 255.0;
const RANGE: usize = 6;
const MOST_RANGE_SHIFT: u32 = 8;
const MOST_BLOCK: usize = 1 << 12;
pub const MOST_LENGTHS_SHIFT: u32 = 20;
const FAR_APART: &str = "ranges not ascending, or more than 2**32 apart: more than ranges hold";

#[derive(Clone, Copy)]
pub struct Shape {
    range_shift: u32,
    block: usize,
    long: usize,
    lengths_shift: u32,
}

pub struct Range {
    id: i64,
    size: usize,
    best: f64,
}

pub enum Kept {
    Short(Coded),
    Long { ranges: Vec<u8>, blocks: Vec<Coded> },
}

pub struct Row {
    pub term: Box<str>,
    pub chunks: usize,
    pub held: Kept,
}

pub type Piece = (Arc<str>, usize, Coded);

pub struct Postings {
    places: Vec<i64>,
    counts: Vec<i64>,
    positions: Vec<i64>,
    offsets: Vec<usize>,
    sizes: Vec<u32>,
}

impl Shape {
    pub fn new(range_shift: u32, block: usize, long: usize, lengths_shift: u32) -> Result<Self, String> {
        let sized = (1..=MOST_BLOCK).contains(&block) && (1..=MOST_BLOCK).contains(&long);
        if sized && range_shift <= MOST_RANGE_SHIFT && (1..=MOST_LENGTHS_SHIFT).contains(&lengths_shift) {
            return Ok(Self {
                range_shift,
                block,
                long,
                lengths_shift,
            });
        }
        Err(
            "a shape's blocks and long lists hold 1 to 4,096 entries, its ranges at most 2**8 places \
             and its blocks of lengths 2**1 to 2**20"
                .to_owned(),
        )
    }

    pub fn range_shift(self) -> u32 {
        self.range_shift
    }

    pub fn block(self) -> usize {
        self.block
    }

    pub fn long(self) -> usize {
        self.long
    }

    pub fn lengths_shift(self) -> u32 {
        self.lengths_shift
    }
}

fn score(count: i64, length: u32, average: f64) -> f64 {
    let times = f64::from(u32::try_from(count).unwrap_or(u32::MAX));
    times * TOP / (times + K1 * (1.0 - B + B * f64::from(length) / average))
}

fn quantized(best: f64) -> u8 {
    let scaled = (best * (STEPS / TOP)).ceil();
    let (mut low, mut high) = (0u8, u8::MAX);
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        if f64::from(middle) <= scaled {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    low
}

fn one_fewer(size: usize) -> Result<u8, String> {
    let fewer = size.checked_sub(1).and_then(|fewer| u8::try_from(fewer).ok());
    fewer.ok_or_else(|| "a range holds more postings than its count keeps, or none".to_owned())
}

pub fn ranged(held: &[Range]) -> Result<Vec<u8>, String> {
    let first = held.first().ok_or_else(|| "a long list holds no range".to_owned())?;
    let mut body = Vec::with_capacity(size_of::<i64>() + held.len() * RANGE);
    body.extend_from_slice(&first.id.to_le_bytes());
    body.extend_from_slice(&narrow(held.len())?.to_le_bytes());
    for pair in held.windows(2) {
        let gap = pair[1].id.checked_sub(pair[0].id).filter(|&gap| gap >= 1);
        let gap = gap.and_then(|gap| u32::try_from(gap).ok());
        body.extend_from_slice(&gap.ok_or_else(|| FAR_APART.to_owned())?.to_le_bytes());
    }
    for each in held {
        body.push(one_fewer(each.size)?);
    }
    body.extend(held.iter().map(|each| quantized(each.best)));
    deflated(&body, None)
}

impl Default for Postings {
    fn default() -> Self {
        Self {
            places: Vec::new(),
            counts: Vec::new(),
            positions: Vec::new(),
            offsets: vec![0],
            sizes: Vec::new(),
        }
    }
}

impl Postings {
    pub fn push<P: Iterator<Item = i64>>(&mut self, place: i64, size: u32, positions: P) -> Result<(), String> {
        let before = self.positions.len();
        self.positions.extend(positions);
        self.places.push(place);
        self.sizes.push(size);
        self.counts.push(i64::from(narrow(self.positions.len() - before)?));
        self.offsets.push(self.positions.len());
        Ok(())
    }

    pub fn chunks(&self) -> usize {
        self.places.len()
    }

    fn extend_from(&mut self, more: &Self, from: usize, to: usize) {
        let (cut, before) = (more.offsets[from], self.positions.len());
        self.places.extend_from_slice(&more.places[from..to]);
        self.counts.extend_from_slice(&more.counts[from..to]);
        self.sizes.extend_from_slice(&more.sizes[from..to]);
        self.positions.extend_from_slice(&more.positions[cut..more.offsets[to]]);
        self.offsets
            .extend(more.offsets[from + 1..=to].iter().map(|offset| offset - cut + before));
    }

    pub fn ranged(&self, found: &mut Vec<Range>, average: f64, shift: u32) {
        for at in 0..self.chunks() {
            let (within, best) = (
                self.places[at] >> shift,
                score(self.counts[at], self.sizes[at], average),
            );
            match found.last_mut() {
                Some(last) if last.id == within => {
                    last.size += 1;
                    last.best = last.best.max(best);
                }
                _ => found.push(Range {
                    id: within,
                    size: 1,
                    best,
                }),
            }
        }
    }

    pub fn coded(&self) -> Result<Coded, String> {
        let lists = Lists {
            ids: &self.places,
            counts: &self.counts,
            positions: &self.positions,
            offsets: &self.offsets,
        };
        list(&lists, 0, self.chunks())
    }
}

pub fn gathered(parts: &[&Postings], start: usize, end: usize) -> Postings {
    let (mut found, mut before) = (Postings::default(), 0);
    for part in parts {
        let held = part.chunks();
        let (from, to) = (
            start.saturating_sub(before).min(held),
            end.saturating_sub(before).min(held),
        );
        if from < to {
            found.extend_from(part, from, to);
        }
        before += held;
    }
    found
}

pub fn blocks_of(parts: &[&Postings], upto: usize, block: usize) -> Result<Vec<Coded>, String> {
    let starts: Vec<usize> = (0..upto).step_by(block).collect();
    starts
        .par_iter()
        .map(|&start| gathered(parts, start, (start + block).min(upto)).coded())
        .collect()
}

pub fn joined(term: &str, parts: &[&Postings], average: f64, shape: Shape) -> Result<Row, String> {
    let chunks = parts.iter().map(|part| part.chunks()).sum();
    if chunks < shape.long {
        let coded = match parts {
            [one] => one.coded(),
            _ => gathered(parts, 0, chunks).coded(),
        };
        return Ok(Row {
            term: term.into(),
            chunks,
            held: Kept::Short(coded?),
        });
    }
    let mut found = Vec::new();
    for part in parts {
        part.ranged(&mut found, average, shape.range_shift);
    }
    Ok(Row {
        term: term.into(),
        chunks,
        held: Kept::Long {
            ranges: ranged(&found)?,
            blocks: blocks_of(parts, chunks, shape.block)?,
        },
    })
}
