use std::ops::Range;

use rayon::prelude::*;

use crate::numbers::{narrow, read, written};
use crate::packed::{unwrapped, wrapped};

const HEAD: usize = 12;
const FIRST: usize = 8;
const GAP: usize = 4;
const MOST_POSITION: i64 = 1 << 32;
const COUNT: usize = 2;
const POSITION: usize = 5;
pub const MOST_BLOCK: usize = 1 << 12;
pub const MOST_TIMES: usize = 1024;
const BLOCK_BYTES: usize = HEAD + GAP * (MOST_BLOCK - 1) + COUNT * MOST_BLOCK;
const TOGETHER: usize = 256;
const TASK: usize = 1 << 14;
const WORTH: usize = 4;
const FAR_APART: &str = "ids not ascending, or more than 2**32 apart: more than a block holds";
const PAST: &str = "a list holds a position past any passage";
const TWICE: &str = "a list holds a position twice, or positions out of their order";
const UNEVEN: &str = "the passages of some blocks are told other than block by block";

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

pub struct Sought<'a> {
    pub bounds: &'a [usize],
    pub ids: &'a [i64],
    pub counts: &'a [i64],
    pub wanted: &'a [i64],
}

#[derive(Default)]
pub struct Occurring {
    pub ids: Vec<i64>,
    pub counts: Vec<i64>,
    pub positions: Vec<i64>,
}

pub type Coded = (Vec<u8>, Vec<u8>);

pub fn each_of<T, R, F>(rows: &[R], one: F) -> Result<Vec<T>, String>
where
    T: Send,
    R: AsRef<[u8]> + Sync,
    F: Fn(usize, &[u8]) -> Result<T, String> + Send + Sync,
{
    let bytes: usize = rows.iter().map(|row| row.as_ref().len()).sum();
    if rows.len() < TOGETHER && bytes < WORTH * TASK {
        let numbered = rows.iter().enumerate();
        numbered.map(|(at, row)| one(at, row.as_ref())).collect()
    } else {
        let least = (rows.len() * TASK / bytes.max(1)).clamp(1, TOGETHER);
        let numbered = rows.par_iter().enumerate().with_min_len(least);
        numbered.map(|(at, row)| one(at, row.as_ref())).collect()
    }
}

pub fn block(data: &[u8]) -> Result<Block, String> {
    let raw = unwrapped(data, BLOCK_BYTES)?;
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
    if at < 0 {
        return Err("a block holds an id below zero".to_owned());
    }
    ids.push(at);
    for gap in steps.as_chunks::<GAP>().0 {
        let step = i64::from(u32::from_le_bytes(*gap));
        if step < 1 {
            return Err("a block holds ids out of order".to_owned());
        }
        let next = at.checked_add(step);
        at = next.ok_or_else(|| "a block holds ids past the last place".to_owned())?;
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
    if counts
        .iter()
        .any(|&count| usize::try_from(count).is_ok_and(|count| count > MOST_TIMES))
    {
        return Err("a block holds a count above 1,024".to_owned());
    }
    Ok(Block { ids, counts })
}

pub fn blocks<R: AsRef<[u8]> + Sync>(rows: &[R]) -> Result<Blocks, String> {
    let found = each_of(rows, |_, row| block(row))?;
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

pub fn positioned(data: &[u8], counts: &[i64]) -> Result<Vec<i64>, String> {
    let each = counts.iter().map(|&count| usize::try_from(count).unwrap_or(0));
    let held = each.fold(0usize, usize::saturating_add);
    let raw = unwrapped(data, held.min(MOST_BLOCK * MOST_TIMES) * POSITION)?;
    let mut found = Vec::with_capacity(raw.len());
    read(&raw, &mut found)?;
    placed(&mut found, counts)?;
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
        for (number, gap) in found[at..end].iter_mut().enumerate() {
            if *gap >= MOST_POSITION {
                return Err(PAST.to_owned());
            }
            if number > 0 && *gap < 1 {
                return Err(TWICE.to_owned());
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

fn chosen(row: &[u8], sought: &Sought, held: Range<usize>) -> Result<Occurring, String> {
    let (ids, counts) = (&sought.ids[held.clone()], &sought.counts[held]);
    let positions = positioned(row, counts)?;
    let from = ids
        .first()
        .map_or(0, |&first| sought.wanted.partition_point(|&each| each < first));
    let mut wanted = sought.wanted[from..].iter().copied().peekable();
    let (mut found, mut at) = (Occurring::default(), 0usize);
    for (&id, &count) in ids.iter().zip(counts) {
        let end = at + usize::try_from(count).unwrap_or(0);
        while wanted.next_if(|&each| each < id).is_some() {}
        if wanted.peek() == Some(&id) {
            found.ids.push(id);
            found.counts.push(count);
            found.positions.extend_from_slice(&positions[at..end]);
        }
        at = end;
    }
    Ok(found)
}

pub fn occurring<R: AsRef<[u8]> + Sync>(rows: &[R], sought: &Sought) -> Result<Occurring, String> {
    let bounds = sought.bounds;
    let even = bounds.len() == rows.len() + 1
        && bounds.first() == Some(&0)
        && bounds.is_sorted()
        && bounds.last() == Some(&sought.ids.len())
        && sought.counts.len() == sought.ids.len();
    if !even {
        return Err(UNEVEN.to_owned());
    }
    let parts = each_of(rows, |at, row| chosen(row, sought, bounds[at]..bounds[at + 1]))?;
    let mut found = Occurring::default();
    for part in &parts {
        found.ids.extend_from_slice(&part.ids);
        found.counts.extend_from_slice(&part.counts);
        found.positions.extend_from_slice(&part.positions);
    }
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
        let mut before = None;
        for &place in &lists.positions[lists.offsets[at]..lists.offsets[at + 1]] {
            let gap = before.map_or(Some(place), |before: i64| {
                place.checked_sub(before).filter(|&gap| gap >= 1)
            });
            written(&mut placed, gap.ok_or_else(|| TWICE.to_owned())?)?;
            before = Some(place);
        }
    }
    Ok((wrapped(&body)?, wrapped(&placed)?))
}
