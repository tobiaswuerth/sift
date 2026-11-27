use std::collections::VecDeque;
use std::sync::Arc;

pub struct Fed {
    pub term: Arc<str>,
    pub chunks: usize,
    pub postings: Vec<u8>,
    pub positions: Vec<u8>,
}

#[derive(Clone, Copy, Default)]
pub struct Size {
    pub postings: usize,
    pub bytes: usize,
}

#[derive(Default)]
pub struct Source {
    rows: VecDeque<Fed>,
    held: Size,
    ended: bool,
}

impl Fed {
    pub fn size(&self) -> Size {
        Size {
            postings: self.chunks,
            bytes: self.postings.len() + self.positions.len(),
        }
    }
}

impl Size {
    pub fn with(self, more: Self) -> Self {
        Self {
            postings: self.postings.saturating_add(more.postings),
            bytes: self.bytes.saturating_add(more.bytes),
        }
    }

    fn without(self, less: Self) -> Self {
        Self {
            postings: self.postings.saturating_sub(less.postings),
            bytes: self.bytes.saturating_sub(less.bytes),
        }
    }

    pub fn reaches(self, most: Self) -> bool {
        self.postings >= most.postings || self.bytes >= most.bytes
    }

    pub fn shared(self, among: usize) -> Self {
        Self {
            postings: self.postings / among.max(1),
            bytes: self.bytes / among.max(1),
        }
    }
}

impl Source {
    pub fn low(&self, ahead: Size) -> bool {
        !self.ended && (self.rows.is_empty() || !self.held.reaches(ahead))
    }

    pub fn dry(&self) -> bool {
        !self.ended && self.rows.is_empty()
    }

    pub fn ended(&self) -> bool {
        self.ended
    }

    pub fn first(&self) -> Option<&Arc<str>> {
        self.rows.front().map(|row| &row.term)
    }

    pub fn fed(&mut self, rows: Vec<Fed>) {
        if rows.is_empty() {
            self.ended = true;
        }
        self.held = rows.iter().fold(self.held, |sum, row| sum.with(row.size()));
        self.rows.extend(rows);
    }

    pub fn next(&mut self, term: &str) -> Option<Fed> {
        let row = self.rows.pop_front_if(|row| &*row.term == term)?;
        self.held = self.held.without(row.size());
        Some(row)
    }

    pub fn leading(&self, term: &str) -> (bool, Size) {
        let (mut rows, mut size) = (0, Size::default());
        for row in self.rows.iter().take_while(|row| &*row.term == term) {
            rows += 1;
            size = size.with(row.size());
        }
        (rows < self.rows.len() || self.ended, size)
    }
}
