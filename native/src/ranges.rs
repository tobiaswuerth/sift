use rayon::prelude::*;

use crate::packed::inflated;

const HEAD: usize = 12;
const FIRST: usize = 8;
const GAP: usize = 4;
const EACH: usize = 2;
const TOGETHER: usize = 16;
const MISTOLD: &str = "ranges hold other than their head says";
const DISORDERED: &str = "ranges hold ids out of order";
const OUTSIDE: &str = "the ranges of a list lie outside its segment";
const UNEVEN: &str = "a list is read with a segment and a count for each";

pub struct Ranges {
    ids: Vec<i64>,
    ends: Vec<usize>,
    best: Vec<u8>,
}

pub struct Bounds {
    pub low: i64,
    pub high: i64,
    pub chunks: usize,
}

impl Ranges {
    pub fn ids(&self) -> &[i64] {
        &self.ids
    }

    pub fn ends(&self) -> &[usize] {
        &self.ends
    }

    pub fn best(&self) -> &[u8] {
        &self.best
    }

    pub fn start(&self, at: usize) -> usize {
        at.checked_sub(1)
            .and_then(|before| self.ends.get(before))
            .copied()
            .unwrap_or(0)
    }

    pub fn total(&self) -> usize {
        self.ends.last().copied().unwrap_or(0)
    }
}

fn most_bytes(bounds: &Bounds) -> Result<usize, String> {
    let most = bounds
        .high
        .checked_sub(bounds.low)
        .and_then(|gap| usize::try_from(gap).ok());
    let most = most.ok_or_else(|| OUTSIDE.to_owned())?;
    let sized = most
        .checked_mul(GAP + EACH)
        .and_then(|size| size.checked_add(HEAD + EACH));
    sized.ok_or_else(|| OUTSIDE.to_owned())
}

fn ids_of(first: i64, gaps: &[u8]) -> Result<Vec<i64>, String> {
    let mut ids = Vec::with_capacity(gaps.len() / GAP + 1);
    let mut at = first;
    ids.push(at);
    for gap in gaps.as_chunks::<GAP>().0 {
        let step = i64::from(u32::from_le_bytes(*gap));
        if step < 1 {
            return Err(DISORDERED.to_owned());
        }
        at = at.checked_add(step).ok_or_else(|| OUTSIDE.to_owned())?;
        ids.push(at);
    }
    Ok(ids)
}

pub fn ranges_read(data: &[u8], bounds: &Bounds) -> Result<Ranges, String> {
    let raw = inflated(data, most_bytes(bounds)?)?;
    let opening = raw
        .split_first_chunk::<FIRST>()
        .and_then(|(first, rest)| Some((first, rest.first_chunk::<GAP>()?)));
    let (first, held) = opening.map_or((0, 0), |(first, told)| {
        let told = usize::try_from(u32::from_le_bytes(*told)).unwrap_or(usize::MAX);
        (i64::from_le_bytes(*first), told)
    });
    let size = held.checked_sub(1).and_then(|gaps| gaps.checked_mul(GAP + EACH));
    let size = size.and_then(|size| size.checked_add(HEAD + EACH));
    if size != Some(raw.len()) {
        return Err(MISTOLD.to_owned());
    }
    let (gaps, rest) = raw[HEAD..].split_at(GAP * (held - 1));
    let (counts, best) = rest.split_at(held);
    let ids = ids_of(first, gaps)?;
    if first < bounds.low || ids.last().is_some_and(|&last| last > bounds.high) {
        return Err(OUTSIDE.to_owned());
    }
    let ends: Vec<usize> = counts
        .iter()
        .scan(0usize, |total, &count| {
            *total += usize::from(count) + 1;
            Some(*total)
        })
        .collect();
    let found = Ranges {
        ids,
        ends,
        best: best.to_vec(),
    };
    if found.total() != bounds.chunks {
        let (total, chunks) = (found.total(), bounds.chunks);
        return Err(format!("the ranges of a list hold {total} passages, not its {chunks}"));
    }
    Ok(found)
}

pub fn ranges_all<R: AsRef<[u8]> + Sync>(rows: &[R], bounds: &[Bounds]) -> Result<Vec<Ranges>, String> {
    if rows.len() != bounds.len() {
        return Err(UNEVEN.to_owned());
    }
    let one = |(row, held): (&R, &Bounds)| ranges_read(row.as_ref(), held);
    if rows.len() < TOGETHER {
        rows.iter().zip(bounds).map(one).collect()
    } else {
        rows.par_iter().zip(bounds).map(one).collect()
    }
}
