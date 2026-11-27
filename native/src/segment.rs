use rayon::prelude::*;

use crate::lists::{Coded, Lists, list};
use crate::packed::deflated;
use crate::words::{Table, Terms};

const MOST_TIMES: usize = 1024;
const TOGETHER: usize = 64;
const K1: f64 = 1.2;
const B: f64 = 0.75;
const TOP: f64 = K1 + 1.0;
const STEPS: f64 = 255.0;
const LENGTH: usize = 4;
const FAR_APART: &str = "ranges not ascending, or more than 2**32 apart: more than ranges hold";

#[derive(Clone, Copy)]
pub struct Shape {
    pub range_shift: u32,
    pub block: usize,
    pub long: usize,
    pub lengths_shift: u32,
}

pub struct Span<'a> {
    pub place: i64,
    pub title: &'a [u32],
    pub text: &'a str,
    pub meta: &'a [u32],
}

struct Token {
    term: u32,
    owner: u32,
    place: u32,
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

pub struct Written {
    pub rows: Vec<Row>,
    pub lengths: Vec<(i64, Vec<u8>)>,
}

pub struct Held {
    most: usize,
    terms: Terms,
    tokens: Vec<Token>,
    ids: Vec<i64>,
    lengths: Vec<u32>,
    streamed: usize,
}

fn at(number: u32) -> usize {
    usize::try_from(number).unwrap_or(usize::MAX)
}

fn narrow(number: usize) -> Result<u32, String> {
    u32::try_from(number).map_err(|_| "a segment holds more than it numbers".to_owned())
}

fn score(count: i64, length: u32, average: f64) -> Result<f64, String> {
    let times = u32::try_from(count).map_err(|_| "a passage holds a term more often than is counted".to_owned())?;
    let times = f64::from(times);
    Ok(times * TOP / (times + K1 * (1.0 - B + B * f64::from(length) / average)))
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

pub fn ranges_packed(ids: &[i64], fewer: &[u8], best: &[u8]) -> Result<Vec<u8>, String> {
    let first = ids.first().ok_or_else(|| "a long list holds no range".to_owned())?;
    let mut body = Vec::with_capacity(12 + ids.len() * 6);
    body.extend_from_slice(&first.to_le_bytes());
    body.extend_from_slice(&narrow(ids.len())?.to_le_bytes());
    for pair in ids.windows(2) {
        let gap = pair[1]
            .checked_sub(pair[0])
            .filter(|&gap| gap >= 1)
            .and_then(|gap| u32::try_from(gap).ok());
        body.extend_from_slice(&gap.ok_or_else(|| FAR_APART.to_owned())?.to_le_bytes());
    }
    body.extend_from_slice(fewer);
    body.extend_from_slice(best);
    deflated(&body, None)
}

pub fn one_fewer<S: TryInto<usize>>(size: S) -> Result<u8, String> {
    let fewer = size
        .try_into()
        .ok()
        .and_then(|size: usize| size.checked_sub(1))
        .and_then(|fewer| u8::try_from(fewer).ok());
    fewer.ok_or_else(|| "a range holds more postings than its count keeps, or none".to_owned())
}

pub fn ranged(held: &[Range]) -> Result<Vec<u8>, String> {
    let ids: Vec<i64> = held.iter().map(|each| each.id).collect();
    let fewer: Result<Vec<u8>, String> = held.iter().map(|each| one_fewer(each.size)).collect();
    let best: Vec<u8> = held.iter().map(|each| quantized(each.best)).collect();
    ranges_packed(&ids, &fewer?, &best)
}

pub fn lengths_written(ids: &[i64], lengths: &[u32], shift: u32) -> Result<Vec<(i64, Vec<u8>)>, String> {
    let (size, mask) = (1usize << shift, (1i64 << shift) - 1);
    let paired: Vec<(i64, u32)> = ids.iter().copied().zip(lengths.iter().copied()).collect();
    let mut found = Vec::new();
    for run in paired.chunk_by(|one, next| one.0 >> shift == next.0 >> shift) {
        let mut dense = vec![0u8; size * LENGTH];
        for &(id, length) in run {
            let start = usize::try_from(id & mask).map_err(|_| "a chunk lies below zero".to_owned())? * LENGTH;
            dense[start..start + LENGTH].copy_from_slice(&length.to_le_bytes());
        }
        found.push((run[0].0 >> shift, deflated(&dense, None)?));
    }
    Ok(found)
}

pub struct Postings {
    places: Vec<i64>,
    counts: Vec<i64>,
    positions: Vec<i64>,
    offsets: Vec<usize>,
    sizes: Vec<u32>,
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

    pub fn ranged(&self, found: &mut Vec<Range>, average: f64, shift: u32) -> Result<(), String> {
        for at in 0..self.chunks() {
            let (within, best) = (
                self.places[at] >> shift,
                score(self.counts[at], self.sizes[at], average)?,
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
        Ok(())
    }

    pub fn coded(&self) -> Result<Coded, String> {
        list(
            &Lists {
                ids: &self.places,
                counts: &self.counts,
                positions: &self.positions,
                offsets: &self.offsets,
            },
            0,
            self.chunks(),
        )
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
        part.ranged(&mut found, average, shape.range_shift)?;
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

fn row(
    term: &str,
    held: &[(u32, u32)],
    ids: &[i64],
    lengths: &[u32],
    average: f64,
    shape: Shape,
) -> Result<Row, String> {
    let mut gathered = Postings::default();
    for run in held.chunk_by(|one, next| one.0 == next.0) {
        let (owner, most) = (at(run[0].0), run.len().min(MOST_TIMES));
        gathered.push(
            ids[owner],
            lengths[owner],
            run[..most].iter().map(|each| i64::from(each.1)),
        )?;
    }
    joined(term, &[&gathered], average, shape)
}

impl Held {
    pub fn new(most: usize) -> Self {
        Self {
            most,
            terms: Terms::default(),
            tokens: Vec::new(),
            ids: Vec::new(),
            lengths: Vec::new(),
            streamed: 0,
        }
    }

    pub fn full(&self) -> bool {
        self.streamed >= self.most
    }

    pub fn chunks(&self) -> usize {
        self.ids.len()
    }

    pub fn first(&self) -> Option<i64> {
        self.ids.first().copied()
    }

    pub fn last(&self) -> Option<i64> {
        self.ids.last().copied()
    }

    pub fn tokens(&self) -> u64 {
        self.lengths.iter().map(|&length| u64::from(length)).sum()
    }

    pub fn coded(&mut self, table: &Table, text: &str, found: &mut Vec<u32>) -> Result<(), String> {
        let mut refused = None;
        table.words(text, |word| match self.terms.code(word) {
            Ok(code) => found.push(code),
            Err(problem) => refused = Some(problem),
        });
        refused.map_or(Ok(()), Err)
    }

    fn placed(&mut self, codes: &[u32], owner: u32, first: u32) -> Result<u32, String> {
        let mut place = first;
        for &term in codes {
            self.tokens.push(Token { term, owner, place });
            place = place
                .checked_add(1)
                .ok_or_else(|| "a passage holds more words than are placed".to_owned())?;
        }
        Ok(place)
    }

    pub fn add(&mut self, table: &Table, span: &Span, scratch: &mut Vec<u32>) -> Result<(), String> {
        let owner = narrow(self.ids.len())?;
        scratch.clear();
        self.coded(table, span.text, scratch)?;
        let text = self.placed(span.title, owner, 0)? + 1;
        let meta = self.placed(scratch, owner, text)? + 1;
        let length = self.placed(span.meta, owner, meta)? - 2;
        self.ids.push(span.place);
        self.lengths.push(length);
        self.streamed += at(length) + 2;
        Ok(())
    }

    pub fn written(&self, average: f64, shape: Shape) -> Result<Written, String> {
        if self.ids.is_empty() {
            return Err("a segment holds a chunk at least".to_owned());
        }
        let names = self.terms.names();
        let mut order: Vec<usize> = (0..names.len()).collect();
        order.sort_unstable_by_key(|&code| names[code].as_bytes());
        let mut rank = vec![0usize; names.len()];
        for (ranked, &code) in order.iter().enumerate() {
            rank[code] = ranked;
        }
        let mut starts = vec![0usize; names.len() + 1];
        for token in &self.tokens {
            starts[rank[at(token.term)] + 1] += 1;
        }
        for ranked in 0..names.len() {
            starts[ranked + 1] += starts[ranked];
        }
        let mut next = starts.clone();
        let mut sorted = vec![(0u32, 0u32); self.tokens.len()];
        for token in &self.tokens {
            let ranked = rank[at(token.term)];
            sorted[next[ranked]] = (token.owner, token.place);
            next[ranked] += 1;
        }
        let rows: Result<Vec<Option<Row>>, String> = (0..names.len())
            .into_par_iter()
            .with_min_len(TOGETHER)
            .map(|ranked| {
                let held = &sorted[starts[ranked]..starts[ranked + 1]];
                if held.is_empty() {
                    return Ok(None);
                }
                row(&names[order[ranked]], held, &self.ids, &self.lengths, average, shape).map(Some)
            })
            .collect();
        let lengths = lengths_written(&self.ids, &self.lengths, shape.lengths_shift)?;
        Ok(Written {
            rows: rows?.into_iter().flatten().collect(),
            lengths,
        })
    }
}
