use crate::numbers::narrow;

const PACK_BYTES: usize = 4000;
const COUNT: usize = 4;
const MORE: u8 = 0x80;
const LOW: u8 = 0x7F;
const BITS: usize = 7;
const MOST_NUMBER_BYTES: usize = 9;
const DAMAGED: &str = "a pack of lists holds other than its head says";
const DISORDERED: &str = "lists are packed in the order of their terms, each term once";
const MISFILED: &str = "a pack of lists is filed under another term than its first";

pub struct Pack {
    pub first: Box<str>,
    pub lists: Vec<u8>,
    pub places: Vec<u8>,
}

pub struct Entry<'a> {
    pub term: &'a str,
    pub chunks: usize,
    pub postings: &'a [u8],
    pub positions: &'a [u8],
}

struct Heading<'a> {
    term: &'a str,
    chunks: usize,
    listed: usize,
    placed: usize,
}

#[derive(Default)]
pub struct Packer {
    last: Option<Box<str>>,
    first: Option<Box<str>>,
    count: usize,
    heads: Vec<u8>,
    lists: Vec<u8>,
    places: Vec<u8>,
}

fn sized(number: usize) -> usize {
    usize::try_from(number.bit_width())
        .unwrap_or(usize::MAX)
        .div_ceil(BITS)
        .max(1)
}

fn put(out: &mut Vec<u8>, mut number: usize) {
    while number > usize::from(LOW) {
        out.push(number.to_le_bytes()[0] & LOW | MORE);
        number >>= BITS;
    }
    out.push(number.to_le_bytes()[0]);
}

fn number(data: &[u8], at: &mut usize) -> Result<usize, String> {
    let mut found = 0;
    for (place, &byte) in data[*at..].iter().take(MOST_NUMBER_BYTES).enumerate() {
        found |= usize::from(byte & LOW) << (BITS * place);
        if byte < MORE && (byte > 0 || place == 0) {
            *at += place + 1;
            return Ok(found);
        }
        if byte < MORE {
            break;
        }
    }
    Err(DAMAGED.to_owned())
}

fn taken<'a>(data: &'a [u8], at: &mut usize, held: usize) -> Result<&'a [u8], String> {
    let end = at.checked_add(held).filter(|&end| end <= data.len());
    let found = &data[*at..end.ok_or(DAMAGED)?];
    *at += held;
    Ok(found)
}

impl Packer {
    fn held(&self) -> usize {
        let first = self.first.as_ref().map_or(0, |first| first.len());
        COUNT + first + self.heads.len() + self.lists.len() + self.places.len()
    }

    pub fn push(&mut self, term: &str, chunks: usize, coded: Option<(&[u8], &[u8])>) -> Result<Option<Pack>, String> {
        if self.last.as_deref().is_some_and(|last| term <= last) {
            return Err(DISORDERED.to_owned());
        }
        self.last = Some(term.into());
        let (postings, positions) = coded.unwrap_or_default();
        let numbers = [term.len(), chunks, postings.len(), positions.len()];
        let more =
            numbers.iter().map(|&each| sized(each)).sum::<usize>() + term.len() + postings.len() + positions.len();
        let closed = if self.held() + more > PACK_BYTES {
            self.finished()?
        } else {
            None
        };
        self.first.get_or_insert_with(|| term.into());
        put(&mut self.heads, term.len());
        self.heads.extend_from_slice(term.as_bytes());
        for held in &numbers[1..] {
            put(&mut self.heads, *held);
        }
        self.lists.extend_from_slice(postings);
        self.places.extend_from_slice(positions);
        self.count += 1;
        Ok(closed)
    }

    pub fn finished(&mut self) -> Result<Option<Pack>, String> {
        let Some(first) = self.first.take() else {
            return Ok(None);
        };
        let mut lists = Vec::with_capacity(COUNT + self.heads.len() + self.lists.len());
        lists.extend_from_slice(&narrow(self.count)?.to_le_bytes());
        lists.append(&mut self.heads);
        lists.append(&mut self.lists);
        self.count = 0;
        let places = std::mem::take(&mut self.places);
        Ok(Some(Pack { first, lists, places }))
    }
}

fn heads<'a>(lists: &'a [u8], first: Option<&str>) -> Result<(Vec<Heading<'a>>, usize), String> {
    let Some((count, _)) = lists.split_first_chunk::<COUNT>() else {
        return Err(DAMAGED.to_owned());
    };
    let (mut found, mut at, mut bodies) = (Vec::<Heading>::new(), COUNT, Some(0usize));
    for _ in 0..u32::from_le_bytes(*count) {
        let named = number(lists, &mut at)?;
        let term = str::from_utf8(taken(lists, &mut at, named)?).map_err(|_| DAMAGED)?;
        let (chunks, listed) = (number(lists, &mut at)?, number(lists, &mut at)?);
        let placed = number(lists, &mut at)?;
        if found.last().is_some_and(|before| term <= before.term) {
            return Err(DAMAGED.to_owned());
        }
        bodies = bodies.and_then(|sum| sum.checked_add(listed));
        found.push(Heading {
            term,
            chunks,
            listed,
            placed,
        });
    }
    let whole = bodies.and_then(|bodies| at.checked_add(bodies)) == Some(lists.len());
    if found.is_empty() || !whole {
        return Err(DAMAGED.to_owned());
    }
    if first.is_some_and(|first| found[0].term != first) {
        return Err(MISFILED.to_owned());
    }
    Ok((found, at))
}

pub fn counted<'a>(
    lists: &'a [u8],
    first: Option<&str>,
    low: &str,
    high: Option<&str>,
) -> Result<Vec<(&'a str, usize)>, String> {
    let (heads, _) = heads(lists, first)?;
    let within = heads
        .iter()
        .filter(|head| head.term >= low && high.is_none_or(|high| head.term < high));
    Ok(within.map(|head| (head.term, head.chunks)).collect())
}

pub fn entries<'a>(lists: &'a [u8], places: Option<&'a [u8]>, first: Option<&str>) -> Result<Vec<Entry<'a>>, String> {
    let (heads, mut at) = heads(lists, first)?;
    let (mut found, mut past) = (Vec::with_capacity(heads.len()), 0);
    for head in heads {
        let postings = taken(lists, &mut at, head.listed)?;
        let positions = places.map_or(Ok(&[][..]), |places| taken(places, &mut past, head.placed))?;
        found.push(Entry {
            term: head.term,
            chunks: head.chunks,
            postings,
            positions,
        });
    }
    if places.is_some_and(|places| past != places.len()) {
        return Err(DAMAGED.to_owned());
    }
    Ok(found)
}
