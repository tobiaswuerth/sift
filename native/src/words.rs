use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const POINTS: usize = 0x11_0000;
const WIDTH: usize = 4;
const APART: u32 = 0x20;
const DROPPED: u32 = u32::MAX;
const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0100_0000_01b3;

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

    pub fn words<F: FnMut(&str)>(&self, text: &str, mut found: F) {
        let mut word = String::new();
        for point in text.chars() {
            match self.kept(point) {
                APART => {
                    if !word.is_empty() {
                        found(&word);
                        word.clear();
                    }
                }
                DROPPED => {}
                kept => word.extend(char::from_u32(kept)),
            }
        }
        if !word.is_empty() {
            found(&word);
        }
    }
}

pub struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Self {
        Self(BASIS)
    }
}

impl Hasher for Fnv {
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u64::from(*byte)).wrapping_mul(PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Default)]
pub struct Terms {
    codes: HashMap<Box<str>, u32, BuildHasherDefault<Fnv>>,
    names: Vec<Box<str>>,
}

impl Terms {
    pub fn code(&mut self, word: &str) -> Result<u32, String> {
        if let Some(&code) = self.codes.get(word) {
            return Ok(code);
        }
        let code =
            u32::try_from(self.names.len()).map_err(|_| "a segment holds more terms than it numbers".to_owned())?;
        self.codes.insert(word.into(), code);
        self.names.push(word.into());
        Ok(code)
    }

    pub fn names(&self) -> &[Box<str>] {
        &self.names
    }
}
