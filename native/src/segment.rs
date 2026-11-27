use rayon::prelude::*;

use crate::numbers::narrow;
use crate::packed::deflated;
use crate::packs::Packer;
use crate::postings::{Filed, MOST_LENGTHS_SHIFT, Postings, Row, Shape, joined};
use crate::words::{Table, Terms};

const MOST_TIMES: usize = 1024;
const TOGETHER: usize = 64;
const LENGTH: usize = 4;
const TOO_MANY_WORDS: &str = "a passage holds more words than are placed";

pub struct Passages<'a> {
    pub places: &'a [i64],
    pub of: &'a [usize],
    pub fields: &'a [(&'a str, &'a str)],
    pub texts: &'a [&'a str],
}

pub struct Written {
    pub filed: Filed,
    pub terms: Vec<(Box<str>, usize)>,
    pub lengths: Vec<(i64, Vec<u8>)>,
}

pub struct Held {
    most: usize,
    most_field: usize,
    terms: Terms,
    tokens: Vec<Token>,
    ids: Vec<i64>,
    lengths: Vec<u32>,
    streamed: usize,
}

struct Span<'a> {
    place: i64,
    title: &'a [u32],
    text: &'a str,
    meta: &'a [u32],
}

struct Token {
    term: u32,
    owner: u32,
    place: u32,
}

fn at(number: u32) -> usize {
    usize::try_from(number).unwrap_or(usize::MAX)
}

fn cut(field: &str, most: usize) -> (&str, bool) {
    match field.char_indices().nth(most) {
        Some((end, _)) => (&field[..end], true),
        None => (field, false),
    }
}

pub fn lengths_written(ids: &[i64], lengths: &[u32], shift: u32) -> Result<Vec<(i64, Vec<u8>)>, String> {
    if ids.len() != lengths.len() || shift > MOST_LENGTHS_SHIFT {
        return Err("each chunk has a length, in blocks of at most 2**20".to_owned());
    }
    let (size, mask) = (1usize << shift, (1i64 << shift) - 1);
    let paired: Vec<(i64, u32)> = ids.iter().copied().zip(lengths.iter().copied()).collect();
    let mut found = Vec::new();
    for run in paired.chunk_by(|one, next| one.0 >> shift == next.0 >> shift) {
        let mut dense = vec![0u8; size * LENGTH];
        for &(id, length) in run {
            let start = usize::try_from(id & mask).unwrap_or(0) * LENGTH;
            dense[start..start + LENGTH].copy_from_slice(&length.to_le_bytes());
        }
        found.push((run[0].0 >> shift, deflated(&dense, None)?));
    }
    Ok(found)
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
        let places = run[..most].iter().map(|each| i64::from(each.1));
        gathered.push(ids[owner], lengths[owner], places)?;
    }
    joined(term, &[&gathered], average, shape)
}

impl Held {
    pub fn new(most: usize, most_field: usize) -> Self {
        Self {
            most,
            most_field,
            terms: Terms::default(),
            tokens: Vec::new(),
            ids: Vec::new(),
            lengths: Vec::new(),
            streamed: 0,
        }
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

    fn coded(&mut self, table: &Table, text: &str, found: &mut Vec<u32>) -> Result<(), String> {
        table.words(text, |word| {
            found.push(self.terms.code(word)?);
            Ok(())
        })
    }

    fn field(&mut self, table: &Table, text: &str) -> Result<Vec<u32>, String> {
        let (whole, shortened) = cut(text, self.most_field);
        let mut found = Vec::new();
        self.coded(table, whole, &mut found)?;
        if shortened {
            found.pop();
        }
        Ok(found)
    }

    fn placed(&mut self, codes: &[u32], owner: u32, first: u32) -> Result<u32, String> {
        let mut place = first;
        for &term in codes {
            self.tokens.push(Token { term, owner, place });
            place = place.checked_add(1).ok_or(TOO_MANY_WORDS)?;
        }
        Ok(place)
    }

    fn add(&mut self, table: &Table, span: &Span, scratch: &mut Vec<u32>) -> Result<(), String> {
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

    pub fn taken(&mut self, table: &Table, passages: &Passages) -> Result<usize, String> {
        if passages.places.len() != passages.texts.len() {
            return Err("each passage has a place and a text".to_owned());
        }
        let mut named: Vec<Option<(Vec<u32>, Vec<u32>)>> = passages.fields.iter().map(|_| None).collect();
        let mut scratch = Vec::new();
        for (number, (&place, text)) in passages.places.iter().zip(passages.texts).enumerate() {
            if self.streamed >= self.most {
                return Ok(number);
            }
            let which = passages.of.get(number).copied().filter(|&which| which < named.len());
            let which = which.ok_or_else(|| "a passage names a document that was not given".to_owned())?;
            let (title, meta) = match &mut named[which] {
                Some(read) => read,
                unread => {
                    let (title, meta) = passages.fields[which];
                    unread.insert((self.field(table, title)?, self.field(table, meta)?))
                }
            };
            let span = Span {
                place,
                title,
                text,
                meta,
            };
            self.add(table, &span, &mut scratch)?;
        }
        Ok(passages.places.len())
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
        let lengths = lengths_written(&self.ids, &self.lengths, shape.lengths_shift())?;
        let (mut filed, mut packer, mut terms) = (Filed::default(), Packer::default(), Vec::new());
        for row in rows?.into_iter().flatten() {
            terms.push((row.term.clone(), row.chunks));
            filed.take(&mut packer, row)?;
        }
        filed.close(&mut packer)?;
        Ok(Written { filed, terms, lengths })
    }
}
