use std::collections::VecDeque;
use std::sync::Arc;

use rayon::prelude::*;

use crate::lists::{Block, block, gaps, placed};
use crate::merge::sources::Fed;
use crate::postings::{Kept, Piece, Postings, Range, Row, Shape, blocks_of, gathered, joined, ranged};

const HEAVY: usize = 64 << 10;
const ASTRAY: &str = "the keyword index holds chunks the store does not";

pub struct Chunks {
    pub live: Vec<i64>,
    pub lengths: Vec<u32>,
    pub dead: Vec<i64>,
    pub average: f64,
}

pub struct Joining {
    chunks: Chunks,
    shape: Shape,
    most_tokens: usize,
}

pub struct Long {
    pub term: Arc<str>,
    pub sources: VecDeque<usize>,
    held: Postings,
    chunks: usize,
    blocks: usize,
    ranges: Vec<Range>,
}

pub enum Merged {
    Gone,
    Large,
    Kept(Row),
}

#[derive(Default)]
pub struct Made {
    pub rows: Vec<Row>,
    pub pieces: Vec<Piece>,
}

fn each<T, R, F>(rows: &[R], heavy: bool, one: F) -> Result<Vec<T>, String>
where
    T: Send,
    R: Sync,
    F: Fn(&R) -> Result<T, String> + Send + Sync,
{
    if heavy {
        rows.par_iter().map(one).collect()
    } else {
        rows.iter().map(one).collect()
    }
}

fn heavy(rows: &[Fed]) -> bool {
    rows.iter().map(|row| row.size().bytes).sum::<usize>() >= HEAVY
}

fn tokens(found: &Block) -> usize {
    let held = found.counts.iter().try_fold(0i64, |sum, &count| sum.checked_add(count));
    held.and_then(|held| usize::try_from(held).ok()).unwrap_or(usize::MAX)
}

impl Long {
    pub fn new(term: Arc<str>, sources: VecDeque<usize>) -> Self {
        Self {
            term,
            sources,
            held: Postings::default(),
            chunks: 0,
            blocks: 0,
            ranges: Vec::new(),
        }
    }
}

impl Joining {
    pub fn new(chunks: Chunks, shape: Shape, most_tokens: usize) -> Result<Self, String> {
        if chunks.live.len() != chunks.lengths.len() {
            return Err("each chunk of a merge has a length".to_owned());
        }
        Ok(Self {
            chunks,
            shape,
            most_tokens,
        })
    }

    fn part(&self, found: &Block, coded: &[u8]) -> Result<Postings, String> {
        let mut places = gaps(coded)?;
        placed(&mut places, &found.counts)?;
        let (mut held, mut from) = (Postings::default(), 0usize);
        for (&id, &count) in found.ids.iter().zip(&found.counts) {
            let to = from + usize::try_from(count).unwrap_or(0);
            if self.chunks.dead.binary_search(&id).is_err() {
                let which = self.chunks.live.binary_search(&id).map_err(|_| ASTRAY.to_owned())?;
                held.push(id, self.chunks.lengths[which], places[from..to].iter().copied())?;
            }
            from = to;
        }
        Ok(held)
    }

    pub fn term(&self, term: &str, rows: &[Fed]) -> Result<Merged, String> {
        let heavy = heavy(rows);
        let blocks = each(rows, heavy, |row| block(&row.postings))?;
        if blocks.iter().map(tokens).fold(0, usize::saturating_add) > self.most_tokens {
            return Ok(Merged::Large);
        }
        let paired: Vec<(&Block, &Fed)> = blocks.iter().zip(rows).collect();
        let parts = each(&paired, heavy, |(found, row)| self.part(found, &row.positions))?;
        if parts.iter().all(|part| part.chunks() == 0) {
            return Ok(Merged::Gone);
        }
        let parts: Vec<&Postings> = parts.iter().collect();
        joined(term, &parts, self.chunks.average, self.shape).map(Merged::Kept)
    }

    fn within(&self, blocks: &[Block], start: usize) -> usize {
        let (mut end, mut held) = (start + 1, tokens(&blocks[start]));
        while end < blocks.len() && held.saturating_add(tokens(&blocks[end])) <= self.most_tokens {
            held = held.saturating_add(tokens(&blocks[end]));
            end += 1;
        }
        end
    }

    pub fn piece(&self, long: &mut Long, rows: &[Fed], made: &mut Made) -> Result<(), String> {
        let heavy = heavy(rows);
        let blocks = each(rows, heavy, |row| block(&row.postings))?;
        let paired: Vec<(&Block, &Fed)> = blocks.iter().zip(rows).collect();
        let mut start = 0;
        while start < rows.len() {
            let end = self.within(&blocks, start);
            let parts = each(&paired[start..end], heavy, |(found, row)| {
                self.part(found, &row.positions)
            })?;
            for part in &parts {
                long.chunks += part.chunks();
                part.ranged(&mut long.ranges, self.chunks.average, self.shape.range_shift());
            }
            let rest = std::mem::take(&mut long.held);
            let all: Vec<&Postings> = std::iter::once(&rest).chain(&parts).collect();
            let held: usize = all.iter().map(|part| part.chunks()).sum();
            let full = if long.chunks >= self.shape.long() {
                held / self.shape.block() * self.shape.block()
            } else {
                0
            };
            for coded in blocks_of(&all, full, self.shape.block())? {
                made.pieces.push((Arc::clone(&long.term), long.blocks, coded));
                long.blocks += 1;
            }
            long.held = gathered(&all, full, held);
            start = end;
        }
        Ok(())
    }

    pub fn finished(&self, long: &Long, made: &mut Made) -> Result<(), String> {
        if long.chunks == 0 {
            return Ok(());
        }
        if long.chunks < self.shape.long() {
            let row = joined(&long.term, &[&long.held], self.chunks.average, self.shape)?;
            made.rows.push(row);
            return Ok(());
        }
        if long.held.chunks() > 0 {
            made.pieces
                .push((Arc::clone(&long.term), long.blocks, long.held.coded()?));
        }
        made.rows.push(Row {
            term: (*long.term).into(),
            chunks: long.chunks,
            held: Kept::Long {
                ranges: ranged(&long.ranges)?,
                blocks: Vec::new(),
            },
        });
        Ok(())
    }
}
