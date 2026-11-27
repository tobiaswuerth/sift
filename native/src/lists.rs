use rayon::prelude::*;

use crate::numbers::{narrow, read, written};
use crate::packed::{unwrapped, wrapped};

const HEAD: usize = 12;
const FIRST: usize = 8;
const GAP: usize = 4;
const MOST_POSITION: i64 = 1 << 32;
const TOGETHER: usize = 256;
const FAR_APART: &str = "ids not ascending, or more than 2**32 apart: more than a block holds";
const PAST: &str = "a list holds a position past any passage";

pub struct Block {
    pub ids: Vec<i64>,
    pub counts: Vec<i64>,
}

pub struct Blocks {
    pub ids: Vec<i64>,
    pub counts: Vec<i64>,
    pub bounds: Vec<usize>,
}

pub struct Lists<'a> {
    pub ids: &'a [i64],
    pub counts: &'a [i64],
    pub positions: &'a [i64],
    pub offsets: &'a [usize],
}

pub type Coded = (Vec<u8>, Vec<u8>);

fn each_of<T, R, F>(rows: &[R], one: F) -> Result<Vec<T>, String>
where
    T: Send,
    R: AsRef<[u8]> + Sync,
    F: Fn(&[u8]) -> Result<T, String> + Send + Sync,
{
    if rows.len() < TOGETHER {
        rows.iter().map(|row| one(row.as_ref())).collect()
    } else {
        rows.par_iter()
            .with_min_len(TOGETHER)
            .map(|row| one(row.as_ref()))
            .collect()
    }
}

pub fn block(data: &[u8]) -> Result<Block, String> {
    let raw = unwrapped(data)?;
    let opening = raw
        .split_first_chunk::<FIRST>()
        .and_then(|(first, rest)| Some((first, rest.split_first_chunk::<GAP>()?)));
    let Some((first, (held, rest))) = opening else {
        return Err("a block is shorter than its head".to_owned());
    };
    let held = usize::try_from(u32::from_le_bytes(*held)).unwrap_or(usize::MAX);
    let split = held
        .checked_sub(1)
        .and_then(|gaps| gaps.checked_mul(GAP))
        .filter(|&split| split <= rest.len());
    let Some((steps, counted)) = split.map(|split| rest.split_at(split)) else {
        return Err("a block holds fewer ids than its head says".to_owned());
    };
    let mut ids = Vec::with_capacity(held);
    let mut at = i64::from_le_bytes(*first);
    ids.push(at);
    for gap in steps.as_chunks::<GAP>().0 {
        let step = i64::from(u32::from_le_bytes(*gap));
        if step < 1 {
            return Err("a block holds ids out of order".to_owned());
        }
        at = at.wrapping_add(step);
        ids.push(at);
    }
    let mut counts = Vec::with_capacity(held);
    read(counted, &mut counts)?;
    if counts.len() != held {
        return Err("a block holds counts other than its ids".to_owned());
    }
    if counts.iter().any(|&count| count < 1) {
        return Err("a block holds a count below one".to_owned());
    }
    Ok(Block { ids, counts })
}

pub fn blocks<R: AsRef<[u8]> + Sync>(rows: &[R]) -> Result<Blocks, String> {
    let found = each_of(rows, block)?;
    let held = found.iter().map(|one| one.ids.len()).sum();
    let (mut ids, mut counts) = (Vec::with_capacity(held), Vec::with_capacity(held));
    let mut bounds = Vec::with_capacity(found.len() + 1);
    bounds.push(0);
    for one in &found {
        ids.extend_from_slice(&one.ids);
        counts.extend_from_slice(&one.counts);
        bounds.push(ids.len());
    }
    Ok(Blocks { ids, counts, bounds })
}

pub fn gaps(data: &[u8]) -> Result<Vec<i64>, String> {
    let raw = unwrapped(data)?;
    let mut found = Vec::with_capacity(raw.len());
    read(&raw, &mut found)?;
    Ok(found)
}

pub fn placed(found: &mut [i64], counts: &[i64]) -> Result<(), String> {
    let mut at = 0usize;
    for &count in counts {
        let end = usize::try_from(count)
            .ok()
            .and_then(|held| at.checked_add(held))
            .filter(|&end| end <= found.len());
        let Some(end) = end else {
            return Err("a list holds positions other than its counts".to_owned());
        };
        let mut place = 0i64;
        for gap in &mut found[at..end] {
            if *gap >= MOST_POSITION {
                return Err(PAST.to_owned());
            }
            place += *gap;
            *gap = place;
        }
        if place >= MOST_POSITION {
            return Err(PAST.to_owned());
        }
        at = end;
    }
    if at == found.len() {
        Ok(())
    } else {
        Err("a list holds positions other than its counts".to_owned())
    }
}

pub fn places<R: AsRef<[u8]> + Sync>(rows: &[R], counts: &[i64]) -> Result<Vec<i64>, String> {
    let parts = each_of(rows, gaps)?;
    let mut found = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for part in &parts {
        found.extend_from_slice(part);
    }
    placed(&mut found, counts)?;
    Ok(found)
}

pub fn list(lists: &Lists, start: usize, end: usize) -> Result<Coded, String> {
    if start >= end || end > lists.ids.len() {
        return Err("a list holds no id, or more than there are".to_owned());
    }
    let held = narrow(end - start)?;
    let mut body = Vec::with_capacity(HEAD + (end - start) * (GAP + 1));
    body.extend_from_slice(&lists.ids[start].to_le_bytes());
    body.extend_from_slice(&held.to_le_bytes());
    for pair in lists.ids[start..end].windows(2) {
        let gap = pair[1]
            .checked_sub(pair[0])
            .filter(|&gap| gap >= 1)
            .and_then(|gap| u32::try_from(gap).ok());
        body.extend_from_slice(&gap.ok_or_else(|| FAR_APART.to_owned())?.to_le_bytes());
    }
    for &count in &lists.counts[start..end] {
        written(&mut body, count)?;
    }
    let mut placed = Vec::with_capacity(lists.offsets[end] - lists.offsets[start]);
    for at in start..end {
        let mut before = 0i64;
        for &place in &lists.positions[lists.offsets[at]..lists.offsets[at + 1]] {
            written(&mut placed, place.wrapping_sub(before))?;
            before = place;
        }
    }
    Ok((wrapped(&body)?, wrapped(&placed)?))
}
