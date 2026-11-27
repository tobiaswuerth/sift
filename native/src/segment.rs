use rayon::prelude::*;

use crate::numbers::narrow;
use crate::packed::{deflated, inflated_into};
use crate::packs::Packer;
use crate::postings::{Filed, MOST_LENGTHS_SHIFT, Postings, Row, Shape, joined};
use crate::words::{Seen, Table, Terms};

const MOST_TIMES: usize = 1024;
const TOGETHER: usize = 64;
const LENGTH: usize = 4;
const PASSAGES: usize = 64;
const GAPS: usize = 2;
const BLOCKS: usize = 16;
const UNEVEN: &str = "each chunk has a length, in blocks of at most 2**20";

pub struct Passages<'a> {
    pub places: &'a [i64],
    pub of: &'a [usize],
    pub fields: &'a [(&'a str, &'a str)],
    pub texts: &'a [&'a str],
}

pub struct Written {
    pub filed: Filed,
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

struct Run {
    codes: Vec<u32>,
    ends: Vec<usize>,
}

struct Token {
    term: u32,
    owner: u32,
    place: u32,
}

impl Run {
    fn of(&self, number: usize) -> &[u32] {
        &self.codes[self.ends[number]..self.ends[number + 1]]
    }
}

fn at(number: u32) -> usize {
    usize::try_from(number).unwrap_or(usize::MAX)
}

fn coded(table: &Table, seen: &mut Seen, text: &str, codes: &mut Vec<u32>) -> Result<(), String> {
    table.words(text, |word| {
        codes.push(seen.code(word)?);
        Ok(())
    })
}

fn run<'a, T>(table: &Table, terms: &Terms, texts: T) -> Result<Run, String>
where
    T: Iterator<Item = (&'a str, bool)>,
{
    let (mut seen, mut codes, mut ends) = (Seen::default(), Vec::new(), vec![0]);
    for (text, shortened) in texts {
        let before = codes.len();
        coded(table, &mut seen, text, &mut codes)?;
        if shortened && codes.len() > before {
            codes.pop();
        }
        ends.push(codes.len());
    }
    let known = seen.known(terms)?;
    for code in &mut codes {
        *code = known[at(*code)];
    }
    Ok(Run { codes, ends })
}

fn cut(field: &str, most: usize) -> (&str, bool) {
    match field.char_indices().nth(most) {
        Some((end, _)) => (&field[..end], true),
        None => (field, false),
    }
}

pub fn lengths_written(ids: &[i64], lengths: &[u32], shift: u32) -> Result<Vec<(i64, Vec<u8>)>, String> {
    if ids.len() != lengths.len() || shift > MOST_LENGTHS_SHIFT {
        return Err(UNEVEN.to_owned());
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

pub fn lengths_read<R: AsRef<[u8]> + Sync>(rows: &[R], shift: u32) -> Result<Vec<u8>, String> {
    if shift > MOST_LENGTHS_SHIFT {
        return Err(UNEVEN.to_owned());
    }
    let size = LENGTH << shift;
    let mut found = vec![0u8; rows.len() * size];
    if rows.len() < 2 * BLOCKS {
        for (row, block) in rows.iter().zip(found.chunks_exact_mut(size)) {
            inflated_into(row.as_ref(), block)?;
        }
    } else {
        let paired = rows.par_iter().zip(found.par_chunks_exact_mut(size));
        let unpacked = |(row, block): (&R, &mut [u8])| inflated_into(row.as_ref(), block);
        paired.with_min_len(BLOCKS).try_for_each(unpacked)?;
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

    fn placed(&mut self, codes: &[u32], owner: u32, first: u32) -> u32 {
        let mut place = first;
        for &term in codes {
            self.tokens.push(Token { term, owner, place });
            place += 1;
        }
        place
    }

    fn add(&mut self, place: i64, title: &[u32], text: &[u32], meta: &[u32]) -> Result<(), String> {
        let (owner, length) = (narrow(self.ids.len())?, narrow(title.len() + text.len() + meta.len())?);
        narrow(title.len() + text.len() + meta.len() + GAPS)?;
        let after_title = self.placed(title, owner, 0) + 1;
        let after_text = self.placed(text, owner, after_title) + 1;
        self.placed(meta, owner, after_text);
        self.ids.push(place);
        self.lengths.push(length);
        self.streamed += at(length) + GAPS;
        Ok(())
    }

    fn fitting(&self, passages: &Passages) -> Result<(), String> {
        if passages.places.len() != passages.texts.len() {
            return Err("each passage has a place and a text".to_owned());
        }
        let named = passages.of.get(..passages.places.len());
        if !named.is_some_and(|named| named.iter().all(|&which| which < passages.fields.len())) {
            return Err("a passage names a document that was not given".to_owned());
        }
        let mut before = self.ids.last().copied().unwrap_or(-1);
        for &place in passages.places {
            if place <= before {
                return Err("a segment takes passages at rising places, from zero on".to_owned());
            }
            before = place;
        }
        Ok(())
    }

    pub fn taken(&mut self, table: &Table, passages: &Passages) -> Result<usize, String> {
        if self.streamed >= self.most {
            return Ok(0);
        }
        self.fitting(passages)?;
        let (terms, most) = (&self.terms, self.most_field);
        let named: Result<Vec<Run>, String> = passages
            .fields
            .par_chunks(PASSAGES)
            .map(|part| {
                run(
                    table,
                    terms,
                    part.iter().flat_map(|held| [cut(held.0, most), cut(held.1, most)]),
                )
            })
            .collect();
        let read: Result<Vec<Run>, String> = passages
            .texts
            .par_chunks(PASSAGES)
            .map(|part| run(table, terms, part.iter().map(|text| (*text, false))))
            .collect();
        let (named, read) = (named?, read?);
        for (number, &place) in passages.places.iter().enumerate() {
            if self.streamed >= self.most {
                return Ok(number);
            }
            let which = passages.of[number];
            let fields = &named[which / PASSAGES];
            let (title, meta) = (fields.of(2 * (which % PASSAGES)), fields.of(2 * (which % PASSAGES) + 1));
            self.add(place, title, read[number / PASSAGES].of(number % PASSAGES), meta)?;
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
        let (mut filed, mut packer) = (Filed::default(), Packer::default());
        for row in rows?.into_iter().flatten() {
            filed.take(&mut packer, row)?;
        }
        filed.close(&mut packer)?;
        Ok(Written { filed, lengths })
    }
}
