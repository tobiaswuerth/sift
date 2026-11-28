use rayon::prelude::*;

const WORD_BYTES: usize = 8;
const DISTANCE_BYTES: usize = 2;
const ROWS_A_TASK: usize = 1 << 14;
const UNEVEN: &str = "a scan takes a query of whole words and codes of whole rows as wide as the query";
const UNFIT: &str = "a scan writes two bytes of distance for each code it scans";
const OUTSIDE: &str = "rows were asked for that the codes do not hold";

pub fn apart(code: impl IntoIterator<Item = u64>, other: &[u64]) -> u32 {
    code.into_iter()
        .zip(other)
        .map(|(one, more)| (one ^ more).count_ones())
        .sum()
}

fn words(row: &[u8]) -> impl Iterator<Item = u64> + '_ {
    row.as_chunks::<WORD_BYTES>()
        .0
        .iter()
        .map(|word| u64::from_le_bytes(*word))
}

fn counted(codes: &[u8], out: &mut [u8], query: &[u64], size: usize) -> Vec<u64> {
    let mut counts = vec![0u64; size];
    let rows = codes.chunks_exact(query.len() * WORD_BYTES);
    for (row, slot) in rows.zip(out.as_chunks_mut::<DISTANCE_BYTES>().0) {
        let distance = u16::try_from(apart(words(row), query)).unwrap_or(u16::MAX);
        *slot = distance.to_le_bytes();
        counts[usize::from(distance)] += 1;
    }
    counts
}

fn added(mut counts: Vec<u64>, more: Vec<u64>) -> Vec<u64> {
    for (count, other) in counts.iter_mut().zip(more) {
        *count += other;
    }
    counts
}

pub fn scanned(held: &[(&[u8], usize)], query: &[u8], out: &mut [u8]) -> Result<Vec<u64>, String> {
    let width = query.len();
    let (asked, rest) = query.as_chunks::<WORD_BYTES>();
    let fits =
        |&(codes, each): &(&[u8], usize)| codes.is_empty() || (each == width && codes.len().is_multiple_of(width));
    let whole = width > 0 && rest.is_empty() && held.iter().all(fits);
    if !whole {
        return Err(UNEVEN.to_owned());
    }
    let rows: usize = held.iter().map(|(codes, _)| codes.len() / width).sum();
    if out.len() != DISTANCE_BYTES * rows {
        return Err(UNFIT.to_owned());
    }
    let asked: Vec<u64> = asked.iter().map(|word| u64::from_le_bytes(*word)).collect();
    let mut parts = Vec::new();
    let mut left = out;
    for &(codes, _) in held {
        let (mine, after) = left.split_at_mut(DISTANCE_BYTES * (codes.len() / width));
        left = after;
        let tasks = codes.chunks(ROWS_A_TASK * width);
        parts.extend(tasks.zip(mine.chunks_mut(ROWS_A_TASK * DISTANCE_BYTES)));
    }
    let size = 8 * width + 1;
    let each = parts
        .into_par_iter()
        .map(|(codes, out)| counted(codes, out, &asked, size));
    Ok(each.reduce(|| vec![0u64; size], added))
}

pub fn gathered(codes: &[u8], width: usize, at: &[i64]) -> Result<Vec<u8>, String> {
    let rows = codes.len().checked_div(width).unwrap_or(0);
    let mut found = Vec::with_capacity(at.len() * width);
    for &row in at {
        let row = usize::try_from(row)
            .ok()
            .filter(|&row| row < rows)
            .ok_or_else(|| OUTSIDE.to_owned())?;
        found.extend_from_slice(&codes[row * width..(row + 1) * width]);
    }
    Ok(found)
}
