use std::collections::HashMap;
use std::hash::{BuildHasher, RandomState};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{PoisonError, RwLock};

use crate::numbers::narrow;

const POINTS: usize = 0x11_0000;
const WIDTH: usize = 4;
const APART: u32 = 0x20;
const DROPPED: u32 = u32::MAX;
const SHARDS: usize = 64;
const SHARDS_WIDE: u64 = 64;

pub struct Table {
    held: Vec<u32>,
}

impl Table {
    pub fn new(stored: &[u8]) -> Result<Self, String> {
        let (points, rest) = stored.as_chunks::<WIDTH>();
        if points.len() != POINTS || !rest.is_empty() {
            return Err("a table of words holds four bytes for each of Unicode's points".to_owned());
        }
        Ok(Self {
            held: points.iter().map(|each| u32::from_le_bytes(*each)).collect(),
        })
    }

    fn kept(&self, point: char) -> u32 {
        usize::try_from(u32::from(point))
            .ok()
            .and_then(|at| self.held.get(at))
            .copied()
            .unwrap_or(APART)
    }

    pub fn words<F: FnMut(&str) -> Result<(), String>>(&self, text: &str, mut found: F) -> Result<(), String> {
        let mut word = String::new();
        for point in text.chars() {
            match self.kept(point) {
                APART if word.is_empty() => {}
                APART => {
                    found(&word)?;
                    word.clear();
                }
                DROPPED => {}
                kept => word.extend(char::from_u32(kept)),
            }
        }
        if word.is_empty() { Ok(()) } else { found(&word) }
    }
}

type Codes<C> = HashMap<Box<str>, C, RandomState>;

pub struct Terms {
    shards: Vec<RwLock<Codes<usize>>>,
    apart: RandomState,
    held: AtomicUsize,
}

#[derive(Default)]
pub struct Seen {
    codes: Codes<u32>,
}

impl Default for Terms {
    fn default() -> Self {
        Self {
            shards: (0..SHARDS).map(|_| RwLock::default()).collect(),
            apart: RandomState::new(),
            held: AtomicUsize::new(0),
        }
    }
}

impl Terms {
    fn shard(&self, word: &str) -> &RwLock<Codes<usize>> {
        &self.shards[usize::try_from(self.apart.hash_one(word) % SHARDS_WIDE).unwrap_or(0)]
    }

    pub fn code(&self, word: &str) -> Result<u32, String> {
        let shard = self.shard(word);
        let known = shard.read().unwrap_or_else(PoisonError::into_inner).get(word).copied();
        let Some(code) = known else {
            let mut held = shard.write().unwrap_or_else(PoisonError::into_inner);
            let next = || self.held.fetch_add(1, Ordering::Relaxed);
            return narrow(*held.entry(word.into()).or_insert_with(next));
        };
        narrow(code)
    }

    pub fn names(&self) -> Vec<Box<str>> {
        let mut names = vec![Box::<str>::default(); self.held.load(Ordering::Relaxed)];
        for shard in &self.shards {
            for (name, &code) in shard.read().unwrap_or_else(PoisonError::into_inner).iter() {
                names[code] = name.clone();
            }
        }
        names
    }
}

impl Seen {
    pub fn code(&mut self, word: &str) -> Result<u32, String> {
        if let Some(&code) = self.codes.get(word) {
            return Ok(code);
        }
        let code = narrow(self.codes.len())?;
        self.codes.insert(word.into(), code);
        Ok(code)
    }

    pub fn known(&self, terms: &Terms) -> Result<Vec<u32>, String> {
        let mut found = vec![0; self.codes.len()];
        for (word, &code) in &self.codes {
            found[usize::try_from(code).unwrap_or(usize::MAX)] = terms.code(word)?;
        }
        Ok(found)
    }
}
