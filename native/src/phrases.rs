use rayon::prelude::*;

const TOGETHER: usize = 1 << 10;
const UNEVEN: &str = "the passages of a word are told with other positions than their counts";
const WORDLESS: &str = "a phrase of no word is counted nowhere";
const TWICE: &str = "a passage of a word is told twice";

pub struct Word<'a> {
    pub ids: &'a [i64],
    pub counts: &'a [i64],
    pub positions: &'a [i64],
}

fn at_or_after(candidates: &[i64], from: usize, id: i64) -> usize {
    if candidates.get(from).is_none_or(|&held| held > id) {
        return candidates.partition_point(|&held| held < id);
    }
    let mut step = 1;
    while candidates.get(from + step).is_some_and(|&held| held < id) {
        step *= 2;
    }
    let (low, high) = (from + step / 2, candidates.len().min(from + step + 1));
    low + candidates[low..high].partition_point(|&held| held < id)
}

fn placed<'a>(candidates: &[i64], word: &Word<'a>) -> Result<Vec<&'a [i64]>, String> {
    if word.ids.len() != word.counts.len() {
        return Err(UNEVEN.to_owned());
    }
    let mut found = vec![&word.positions[..0]; candidates.len()];
    let mut told = vec![false; candidates.len()];
    let (mut at, mut near) = (0usize, 0usize);
    for (&id, &count) in word.ids.iter().zip(word.counts) {
        let held = usize::try_from(count).ok();
        let end = held.and_then(|held| at.checked_add(held));
        let Some(end) = end.filter(|&end| end <= word.positions.len()) else {
            return Err(UNEVEN.to_owned());
        };
        near = at_or_after(candidates, near, id);
        if candidates.get(near) == Some(&id) {
            if std::mem::replace(&mut told[near], true) {
                return Err(TWICE.to_owned());
            }
            found[near] = &word.positions[at..end];
        }
        at = end;
    }
    if at == word.positions.len() {
        Ok(found)
    } else {
        Err(UNEVEN.to_owned())
    }
}

fn counted(first: &[i64], rest: &[Vec<&[i64]>], candidate: usize) -> i64 {
    let follows = |start: &&i64| {
        rest.iter().zip(1i64..).all(|(word, shift)| {
            let sought = start.checked_add(shift);
            sought.is_some_and(|sought| word[candidate].binary_search(&sought).is_ok())
        })
    };
    i64::try_from(first.iter().filter(follows).count()).unwrap_or(i64::MAX)
}

pub fn phrased(candidates: &[i64], words: &[Word]) -> Result<Vec<i64>, String> {
    let each = words.par_iter().map(|word| placed(candidates, word));
    let held = each.collect::<Result<Vec<_>, String>>()?;
    let Some((first, rest)) = held.split_first() else {
        return Err(WORDLESS.to_owned());
    };
    let one = |candidate: usize| counted(first[candidate], rest, candidate);
    if candidates.len() < TOGETHER {
        Ok((0..candidates.len()).map(one).collect())
    } else {
        let all = (0..candidates.len()).into_par_iter().with_min_len(TOGETHER);
        Ok(all.map(one).collect())
    }
}
