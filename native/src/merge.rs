mod joining;
mod sources;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;

use rayon::prelude::*;

use crate::packs::{Packer, entries};
use crate::postings::{Filed, Shape};

pub use joining::Chunks;
use joining::{Joining, Long, Made, Merged};
pub use sources::Fed;
use sources::{Size, Source};

const TOGETHER: usize = 16;
const PIECE: usize = 32;

type Head = Reverse<(Arc<str>, usize)>;

pub struct Limits {
    pub postings: usize,
    pub bytes: usize,
    pub tokens: usize,
}

pub struct Step {
    pub filed: Filed,
    pub wanted: Vec<usize>,
    pub done: bool,
}

pub struct Merger {
    joining: Joining,
    sources: Vec<Source>,
    most: Size,
    ahead: Size,
    heads: BinaryHeap<Head>,
    long: Option<Long>,
    packer: Packer,
}

impl Merger {
    pub fn new(chunks: Chunks, shape: Shape, limits: &Limits, sources: usize) -> Result<Self, String> {
        let most = Size {
            postings: limits.postings,
            bytes: limits.bytes,
        };
        Ok(Self {
            joining: Joining::new(chunks, shape, limits.tokens)?,
            sources: (0..sources).map(|_| Source::default()).collect(),
            most,
            ahead: most.shared(sources),
            heads: BinaryHeap::new(),
            long: None,
            packer: Packer::default(),
        })
    }

    fn headed(&mut self, source: usize) {
        if let Some(term) = self.sources[source].first() {
            self.heads.push(Reverse((Arc::clone(term), source)));
        }
    }

    pub fn fed(&mut self, source: usize, rows: Vec<Fed>) -> Option<bool> {
        let owned = self.long.as_ref().is_some_and(|long| long.sources.contains(&source));
        let held = self.sources.get_mut(source)?;
        let first = held.first().is_none();
        held.fed(rows);
        let low = held.low(self.ahead);
        if first && !owned {
            self.headed(source);
        }
        Some(low)
    }

    pub fn fed_packs<R: AsRef<[u8]>>(&mut self, source: usize, packs: &[(R, R)]) -> Result<Option<bool>, String> {
        if source >= self.sources.len() {
            return Ok(None);
        }
        let mut rows = Vec::new();
        for (lists, places) in packs {
            let found = entries(lists.as_ref(), Some(places.as_ref()))?;
            rows.extend(found.iter().filter(|each| !each.postings.is_empty()).map(|each| Fed {
                term: Arc::from(each.term),
                chunks: each.chunks,
                postings: each.postings.to_vec(),
                positions: each.positions.to_vec(),
            }));
        }
        Ok(self.fed(source, rows))
    }

    pub fn ended(&mut self, source: usize) -> Option<bool> {
        self.sources.get_mut(source)?.end();
        Some(false)
    }

    fn streamed(&mut self, mut long: Long, made: &mut Made) -> Result<Vec<usize>, String> {
        while let Some(&source) = long.sources.front() {
            let rows: Vec<Fed> = (0..PIECE)
                .map_while(|_| self.sources[source].next(&long.term))
                .collect();
            self.joining.piece(&mut long, &rows, made)?;
            if self.sources[source].first().is_some() {
                if rows.len() < PIECE {
                    self.headed(source);
                    long.sources.pop_front();
                }
            } else if self.sources[source].ended() {
                long.sources.pop_front();
            } else {
                self.long = Some(long);
                return Ok(vec![source]);
            }
        }
        self.joining.finished(&long, made)?;
        Ok(Vec::new())
    }

    fn batched(&self, batch: &[(Arc<str>, Vec<Fed>)], made: &mut Made) -> Result<(), String> {
        let joining = &self.joining;
        let found: Result<Vec<Merged>, String> = batch
            .par_iter()
            .with_min_len(TOGETHER)
            .map(|(term, rows)| joining.term(term, rows))
            .collect();
        for (each, (term, rows)) in found?.into_iter().zip(batch) {
            match each {
                Merged::Gone => {}
                Merged::Kept(row) => made.rows.push(row),
                Merged::Large => {
                    let mut long = Long::new(Arc::clone(term), VecDeque::new());
                    for piece in rows.chunks(PIECE) {
                        joining.piece(&mut long, piece, made)?;
                    }
                    joining.finished(&long, made)?;
                }
            }
        }
        Ok(())
    }

    fn having(&mut self, term: &str) -> Vec<usize> {
        let mut found = Vec::new();
        while self.heads.peek().is_some_and(|head| &*head.0.0 == term) {
            found.extend(self.heads.pop().map(|head| head.0.1));
        }
        found
    }

    fn back(&mut self, term: &Arc<str>, having: &[usize]) {
        for &source in having {
            self.heads.push(Reverse((Arc::clone(term), source)));
        }
    }

    fn taken(&mut self, term: &str, having: &[usize]) -> Vec<Fed> {
        let mut rows = Vec::new();
        for &source in having {
            while let Some(row) = self.sources[source].next(term) {
                rows.push(row);
            }
            self.headed(source);
        }
        rows
    }

    fn wanted(&self, short: &[usize]) -> Vec<usize> {
        (0..self.sources.len())
            .filter(|at| self.sources[*at].low(self.ahead) || short.contains(at))
            .collect()
    }

    fn gathered(&mut self, made: &mut Made) -> Result<(Vec<usize>, bool), String> {
        if let Some(long) = self.long.take() {
            return Ok((self.streamed(long, made)?, false));
        }
        if self.sources.iter().any(Source::dry) {
            return Ok((self.wanted(&[]), false));
        }
        let mut batch: Vec<(Arc<str>, Vec<Fed>)> = Vec::new();
        let (mut held, mut short) = (Size::default(), Vec::new());
        while let Some(term) = self.heads.peek().map(|head| Arc::clone(&head.0.0)) {
            let having = self.having(&term);
            let leading: Vec<(bool, Size)> = having.iter().map(|&at| self.sources[at].leading(&term)).collect();
            let size = leading.iter().fold(Size::default(), |sum, each| sum.with(each.1));
            let unfinished: Vec<usize> = having
                .iter()
                .zip(&leading)
                .filter(|each| !each.1.0)
                .map(|each| *each.0)
                .collect();
            let large = size.reaches(self.most);
            if large && batch.is_empty() {
                let long = Long::new(term, having.into_iter().collect());
                return Ok((self.streamed(long, made)?, false));
            }
            if large || !unfinished.is_empty() {
                self.back(&term, &having);
                short = if large { Vec::new() } else { unfinished };
                break;
            }
            let rows = self.taken(&term, &having);
            batch.push((term, rows));
            held = held.with(size);
            if held.reaches(self.most) {
                break;
            }
        }
        self.batched(&batch, made)?;
        let wanted = self.wanted(&short);
        let done = wanted.is_empty() && self.heads.is_empty();
        Ok((wanted, done))
    }

    pub fn stepped(&mut self) -> Result<Step, String> {
        let (mut made, mut filed) = (Made::default(), Filed::default());
        let (wanted, done) = self.gathered(&mut made)?;
        for row in made.rows {
            filed.take(&mut self.packer, row)?;
        }
        filed.blocks.extend(made.pieces);
        if done {
            filed.close(&mut self.packer)?;
        }
        Ok(Step { filed, wanted, done })
    }
}
