use rayon::prelude::*;

const TOGETHER: usize = 1 << 15;
const UNEVEN: &str = "the passages of lists are told with a count for each";

pub type Summed = (Vec<i64>, Vec<i64>);

pub fn summed(ids: &[i64], counts: &[i64]) -> Result<Summed, String> {
    if ids.len() != counts.len() {
        return Err(UNEVEN.to_owned());
    }
    let mut paired: Vec<(i64, i64)> = ids.iter().copied().zip(counts.iter().copied()).collect();
    if paired.len() < TOGETHER {
        paired.sort_unstable_by_key(|each| each.0);
    } else {
        paired.par_sort_unstable_by_key(|each| each.0);
    }
    let (mut found, mut sums) = (Vec::new(), Vec::new());
    for run in paired.chunk_by(|one, next| one.0 == next.0) {
        found.push(run[0].0);
        sums.push(run.iter().fold(0i64, |sum, each| sum.saturating_add(each.1)));
    }
    Ok((found, sums))
}
