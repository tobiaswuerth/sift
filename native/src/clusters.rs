use rayon::prelude::*;

use crate::hamming::apart;

const BITS: usize = 64;
const ROWS_A_TASK: usize = 1 << 12;
const UNEVEN: &str = "a clustering takes codes and centroids of whole words, and at least one centroid";
const UNOWNED: &str = "a clustering takes a centroid for each code, among those it holds";

fn even(held: &[u64], centroids: &[u64], width: usize) -> Result<(), String> {
    let whole = width > 0 && held.len().is_multiple_of(width) && centroids.len().is_multiple_of(width);
    if whole && !centroids.is_empty() {
        Ok(())
    } else {
        Err(UNEVEN.to_owned())
    }
}

fn nearest_of(code: &[u64], centroids: &[u64], width: usize) -> i64 {
    let mut best = (u32::MAX, 0i64);
    for (centroid, at) in centroids.chunks_exact(width).zip(0i64..) {
        let found = apart(code.iter().copied(), centroid);
        if found < best.0 {
            best = (found, at);
        }
    }
    best.1
}

pub fn nearest(held: &[u64], centroids: &[u64], width: usize) -> Result<Vec<i64>, String> {
    even(held, centroids, width)?;
    let each = held.par_chunks_exact(width).with_min_len(ROWS_A_TASK);
    Ok(each.map(|code| nearest_of(code, centroids, width)).collect())
}

fn counted(mut ones: Vec<u32>, (code, owner): (&[u64], usize), width: usize) -> Vec<u32> {
    let row = &mut ones[owner * (width * BITS + 1)..(owner + 1) * (width * BITS + 1)];
    row[width * BITS] += 1;
    for (word, &value) in code.iter().enumerate() {
        let mut left = value;
        while left != 0 {
            let bit = usize::try_from(left.trailing_zeros()).unwrap_or(0);
            row[word * BITS + bit] += 1;
            left &= left - 1;
        }
    }
    ones
}

fn added(mut ones: Vec<u32>, other: Vec<u32>) -> Vec<u32> {
    for (one, more) in ones.iter_mut().zip(other) {
        *one += more;
    }
    ones
}

pub fn majority(held: &[u64], assigned: &[i64], centroids: &[u64], width: usize) -> Result<Vec<u64>, String> {
    even(held, centroids, width)?;
    let clusters = centroids.len() / width;
    let owners: Option<Vec<usize>> = assigned
        .iter()
        .map(|&owner| usize::try_from(owner).ok().filter(|&owner| owner < clusters))
        .collect();
    let owners = owners.filter(|owners| owners.len() * width == held.len());
    let Some(owners) = owners else {
        return Err(UNOWNED.to_owned());
    };
    let size = clusters * (width * BITS + 1);
    let rows = held
        .par_chunks_exact(width)
        .zip(owners.par_iter().copied())
        .with_min_len(ROWS_A_TASK);
    let ones = rows
        .fold(|| vec![0u32; size], |ones, row| counted(ones, row, width))
        .reduce(|| vec![0u32; size], added);
    let mut made = centroids.to_vec();
    for (cluster, row) in ones.chunks_exact(width * BITS + 1).enumerate() {
        let codes = row[width * BITS];
        if codes == 0 {
            continue;
        }
        for (word, bits) in row[..width * BITS].as_chunks::<BITS>().0.iter().enumerate() {
            let set = bits.iter().zip(0u32..).filter(|&(count, _)| count * 2 > codes);
            made[cluster * width + word] = set.fold(0u64, |value, (_, bit)| value | (1 << bit));
        }
    }
    Ok(made)
}
