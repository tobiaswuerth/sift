use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use rayon::prelude::*;

use crate::building::{Listed, Piece, listed};
use crate::lists::{Block, block, gaps, numbers_of, placed};
use crate::segment::{Kept, Postings, Range, Row, Shape, blocks_of, gathered, joined, ranged};

const TOGETHER: usize = 16;
const HEAVY: usize = 64 << 10;
const PIECE: usize = 32;
const ASTRAY: &str = "the keyword index holds chunks the store does not";

type Given = (PyBackedStr, usize, PyBackedBytes, PyBackedBytes);
type Head = Reverse<(Arc<str>, usize)>;
type Stepped<'py> = (Listed<'py>, Vec<usize>, bool);

struct Fed {
    term: Arc<str>,
    chunks: usize,
    postings: Vec<u8>,
    positions: Vec<u8>,
}

#[derive(Clone, Copy, Default)]
struct Size {
    postings: usize,
    bytes: usize,
}

#[derive(Default)]
struct Source {
    rows: VecDeque<Fed>,
    held: Size,
    ended: bool,
}

struct Long {
    term: Arc<str>,
    sources: VecDeque<usize>,
    held: Postings,
    chunks: usize,
    blocks: usize,
    ranges: Vec<Range>,
}

enum Merged {
    Gone,
    Large,
    Kept(Row),
}

#[derive(Default)]
struct Out {
    rows: Vec<Row>,
    pieces: Vec<Piece>,
}

struct Limits {
    postings: usize,
    bytes: usize,
    tokens: usize,
}

struct Core {
    live: Vec<i64>,
    lengths: Vec<u32>,
    dead: Vec<i64>,
    average: f64,
    shape: Shape,
    limits: Limits,
}

#[pyclass(module = "sift._native")]
pub struct Merger {
    core: Core,
    sources: Vec<Source>,
    ahead: Size,
    heads: BinaryHeap<Head>,
    long: Option<Long>,
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
    usize::try_from(found.counts.iter().sum::<i64>()).unwrap_or(usize::MAX)
}

impl Fed {
    fn size(&self) -> Size {
        Size {
            postings: self.chunks,
            bytes: self.postings.len() + self.positions.len(),
        }
    }
}

impl Size {
    fn with(self, more: Self) -> Self {
        Self {
            postings: self.postings + more.postings,
            bytes: self.bytes + more.bytes,
        }
    }

    fn without(self, less: Self) -> Self {
        Self {
            postings: self.postings - less.postings,
            bytes: self.bytes - less.bytes,
        }
    }
}

impl Source {
    fn low(&self, ahead: Size) -> bool {
        let few = self.held.postings < ahead.postings && self.held.bytes < ahead.bytes;
        !self.ended && (few || self.rows.is_empty())
    }

    fn next(&mut self, term: &str) -> Option<Fed> {
        let row = self.rows.pop_front_if(|row| &*row.term == term)?;
        self.held = self.held.without(row.size());
        Some(row)
    }

    fn leading(&self, term: &str) -> (usize, Size) {
        let (mut rows, mut size) = (0, Size::default());
        for row in self.rows.iter().take_while(|row| &*row.term == term) {
            rows += 1;
            size = size.with(row.size());
        }
        (rows, size)
    }
}

impl Long {
    fn new(term: Arc<str>, sources: VecDeque<usize>) -> Self {
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

impl Limits {
    fn reached(&self, size: Size) -> bool {
        size.postings >= self.postings || size.bytes >= self.bytes
    }
}

impl Core {
    fn part(&self, found: &Block, coded: &[u8]) -> Result<Postings, String> {
        let mut places = gaps(coded)?;
        placed(&mut places, &found.counts)?;
        let (mut held, mut from) = (Postings::default(), 0usize);
        for (&id, &count) in found.ids.iter().zip(&found.counts) {
            let to = from + usize::try_from(count).unwrap_or(0);
            if self.dead.binary_search(&id).is_err() {
                let which = self.live.binary_search(&id).map_err(|_| ASTRAY.to_owned())?;
                held.push(id, self.lengths[which], places[from..to].iter().copied())?;
            }
            from = to;
        }
        Ok(held)
    }

    fn term(&self, term: &str, rows: &[Fed]) -> Result<Merged, String> {
        let heavy = heavy(rows);
        let blocks = each(rows, heavy, |row| block(&row.postings))?;
        if blocks.iter().map(tokens).fold(0, usize::saturating_add) > self.limits.tokens {
            return Ok(Merged::Large);
        }
        let paired: Vec<(&Block, &Fed)> = blocks.iter().zip(rows).collect();
        let parts = each(&paired, heavy, |(found, row)| self.part(found, &row.positions))?;
        if parts.iter().all(|part| part.chunks() == 0) {
            return Ok(Merged::Gone);
        }
        let parts: Vec<&Postings> = parts.iter().collect();
        joined(term, &parts, self.average, self.shape).map(Merged::Kept)
    }

    fn within(&self, blocks: &[Block], start: usize) -> usize {
        let (mut end, mut held) = (start + 1, tokens(&blocks[start]));
        while end < blocks.len() && held.saturating_add(tokens(&blocks[end])) <= self.limits.tokens {
            held = held.saturating_add(tokens(&blocks[end]));
            end += 1;
        }
        end
    }

    fn piece(&self, long: &mut Long, rows: &[Fed], out: &mut Out) -> Result<(), String> {
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
                part.ranged(&mut long.ranges, self.average, self.shape.range_shift)?;
            }
            let rest = std::mem::take(&mut long.held);
            let all: Vec<&Postings> = std::iter::once(&rest).chain(&parts).collect();
            let held: usize = all.iter().map(|part| part.chunks()).sum();
            let full = if long.chunks >= self.shape.long {
                held / self.shape.block * self.shape.block
            } else {
                0
            };
            for coded in blocks_of(&all, full, self.shape.block)? {
                out.pieces.push((Arc::clone(&long.term), long.blocks, coded));
                long.blocks += 1;
            }
            long.held = gathered(&all, full, held);
            start = end;
        }
        Ok(())
    }

    fn finished(&self, long: &Long, out: &mut Out) -> Result<(), String> {
        if long.chunks == 0 {
            return Ok(());
        }
        if long.chunks < self.shape.long {
            out.rows
                .push(joined(&long.term, &[&long.held], self.average, self.shape)?);
            return Ok(());
        }
        if long.held.chunks() > 0 {
            out.pieces
                .push((Arc::clone(&long.term), long.blocks, long.held.coded()?));
        }
        out.rows.push(Row {
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

impl Merger {
    fn headed(&mut self, source: usize) {
        if let Some(next) = self.sources[source].rows.front() {
            self.heads.push(Reverse((Arc::clone(&next.term), source)));
        }
    }

    fn fed(&mut self, source: usize, rows: Vec<Fed>) {
        let owned = self.long.as_ref().is_some_and(|long| long.sources.contains(&source));
        let held = &mut self.sources[source];
        if rows.is_empty() {
            held.ended = true;
            return;
        }
        let first = held.rows.is_empty();
        held.held = rows.iter().fold(held.held, |sum, row| sum.with(row.size()));
        held.rows.extend(rows);
        if first && !owned {
            self.headed(source);
        }
    }

    fn streamed(&mut self, out: &mut Out) -> Result<Option<usize>, String> {
        let Some(mut long) = self.long.take() else {
            return Ok(None);
        };
        while let Some(&source) = long.sources.front() {
            let rows: Vec<Fed> = (0..PIECE)
                .map_while(|_| self.sources[source].next(&long.term))
                .collect();
            self.core.piece(&mut long, &rows, out)?;
            let held = &self.sources[source];
            if !held.rows.is_empty() {
                if rows.len() < PIECE {
                    self.headed(source);
                    long.sources.pop_front();
                }
            } else if held.ended {
                long.sources.pop_front();
            } else {
                self.long = Some(long);
                return Ok(Some(source));
            }
        }
        self.core.finished(&long, out)?;
        Ok(None)
    }

    fn batched(&self, batch: &[(Arc<str>, Vec<Fed>)], out: &mut Out) -> Result<(), String> {
        let core = &self.core;
        let found: Result<Vec<Merged>, String> = batch
            .par_iter()
            .with_min_len(TOGETHER)
            .map(|(term, rows)| core.term(term, rows))
            .collect();
        for (each, (term, rows)) in found?.into_iter().zip(batch) {
            match each {
                Merged::Gone => {}
                Merged::Kept(row) => out.rows.push(row),
                Merged::Large => {
                    let mut long = Long::new(Arc::clone(term), VecDeque::new());
                    for piece in rows.chunks(PIECE) {
                        core.piece(&mut long, piece, out)?;
                    }
                    core.finished(&long, out)?;
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

    fn stepped(&mut self, out: &mut Out) -> Result<(Vec<usize>, bool), String> {
        if self.long.is_some() {
            return Ok((self.streamed(out)?.into_iter().collect(), false));
        }
        if self.sources.iter().any(|held| held.rows.is_empty() && !held.ended) {
            return Ok((self.wanted(&[]), false));
        }
        let mut batch: Vec<(Arc<str>, Vec<Fed>)> = Vec::new();
        let (mut held, mut short) = (Size::default(), Vec::new());
        while let Some(term) = self.heads.peek().map(|head| Arc::clone(&head.0.0)) {
            let having = self.having(&term);
            let leading: Vec<(usize, Size)> = having.iter().map(|&at| self.sources[at].leading(&term)).collect();
            let size = leading.iter().fold(Size::default(), |sum, each| sum.with(each.1));
            let unfinished: Vec<usize> = having
                .iter()
                .zip(&leading)
                .filter(|&(&at, each)| each.0 == self.sources[at].rows.len() && !self.sources[at].ended)
                .map(|(&at, _)| at)
                .collect();
            let large = self.core.limits.reached(size);
            if large && batch.is_empty() {
                self.long = Some(Long::new(term, having.into_iter().collect()));
                return Ok((self.streamed(out)?.into_iter().collect(), false));
            }
            if large || !unfinished.is_empty() {
                self.back(&term, &having);
                short = if large { Vec::new() } else { unfinished };
                break;
            }
            let rows = self.taken(&term, &having);
            batch.push((term, rows));
            held = held.with(size);
            if self.core.limits.reached(held) {
                break;
            }
        }
        self.batched(&batch, out)?;
        let wanted = self.wanted(&short);
        let done = wanted.is_empty() && self.heads.is_empty();
        Ok((wanted, done))
    }
}

#[pymethods]
impl Merger {
    #[new]
    fn new(
        live: &Bound<'_, PyAny>,
        lengths: &Bound<'_, PyAny>,
        dead: &Bound<'_, PyAny>,
        average: f64,
        shape: (u32, usize, usize, u32),
        limits: (usize, usize, usize),
        sources: usize,
    ) -> PyResult<Self> {
        let lengths: Result<Vec<u32>, _> = numbers_of(lengths)?.into_iter().map(u32::try_from).collect();
        let lengths = lengths.map_err(|_| PyValueError::new_err("a passage is longer than its length keeps"))?;
        let live = numbers_of(live)?;
        if live.len() != lengths.len() {
            return Err(PyValueError::new_err("each chunk of a merge has a length"));
        }
        let core = Core {
            live,
            lengths,
            dead: numbers_of(dead)?,
            average,
            shape: Shape {
                range_shift: shape.0,
                block: shape.1,
                long: shape.2,
                lengths_shift: shape.3,
            },
            limits: Limits {
                postings: limits.0,
                bytes: limits.1,
                tokens: limits.2,
            },
        };
        Ok(Self {
            core,
            sources: (0..sources).map(|_| Source::default()).collect(),
            ahead: Size {
                postings: limits.0 / sources.max(1),
                bytes: limits.1 / sources.max(1),
            },
            heads: BinaryHeap::new(),
            long: None,
        })
    }

    fn feed(&mut self, source: usize, rows: &Bound<'_, PyAny>) -> PyResult<bool> {
        if source >= self.sources.len() {
            return Err(PyIndexError::new_err("a merge was fed rows of a source it has not"));
        }
        let given: Vec<Given> = rows.extract()?;
        let rows = given
            .into_iter()
            .map(|(term, chunks, postings, positions)| Fed {
                term: Arc::from(&*term),
                chunks,
                postings: postings.to_vec(),
                positions: positions.to_vec(),
            })
            .collect();
        self.fed(source, rows);
        Ok(self.sources[source].low(self.ahead))
    }

    fn step<'py>(&mut self, py: Python<'py>, segment: i64) -> PyResult<Stepped<'py>> {
        let mut out = Out::default();
        let (wanted, done) = py.detach(|| self.stepped(&mut out)).map_err(PyValueError::new_err)?;
        Ok((listed(py, segment, &out.rows, &out.pieces)?, wanted, done))
    }
}
