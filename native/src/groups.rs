use crate::postings::TOP;
use crate::reading::{Failure, Picked, Reached, Reader, Scored, chosen};
use crate::scoring::{Expression, Mask, may_reach, unreached_alone};

pub struct Leaf {
    pub places: Vec<i64>,
    pub scores: Vec<f64>,
    pub ranges: Vec<i64>,
    pub parts: Vec<usize>,
    pub bounding: f64,
    pub weight: f64,
    pub size: usize,
}

pub struct Grouping<'a> {
    pub reader: Reader<'a>,
    pub leaves: Vec<Leaf>,
    pub expression: Expression,
    pub reached: Reached,
}

type HeldScores = Vec<Option<Vec<f64>>>;

impl Leaf {
    fn most(&self) -> f64 {
        self.bounding * TOP
    }

    fn exact_within(&self, ranges: &[i64]) -> Scored {
        let (mut places, mut scores) = (Vec::new(), Vec::new());
        for &range in ranges {
            let low = self.ranges.partition_point(|&each| each < range);
            let high = low + self.ranges[low..].partition_point(|&each| each <= range);
            places.extend_from_slice(&self.places[low..high]);
            scores.extend_from_slice(&self.scores[low..high]);
        }
        (places, scores)
    }
}

fn distinct(places: &[i64], shift: u32) -> Vec<i64> {
    let mut found: Vec<i64> = Vec::new();
    for &place in places {
        if found.last() != Some(&(place >> shift)) {
            found.push(place >> shift);
        }
    }
    found
}

fn joined(parts: Vec<Scored>) -> Scored {
    let mut paired: Vec<(i64, f64)> = parts
        .into_iter()
        .flat_map(|(places, scores)| places.into_iter().zip(scores))
        .collect();
    paired.sort_by_key(|each| each.0);
    paired.into_iter().unzip()
}

fn summed(leaves: &[Leaf], held: &HeldScores, size: usize) -> Vec<f64> {
    let mut total = vec![0.0; size];
    for (leaf, scores) in leaves.iter().zip(held) {
        if let Some(scores) = scores {
            for (sum, score) in total.iter_mut().zip(scores) {
                *sum += leaf.bounding * score;
            }
        }
    }
    total
}

fn matched(places: &[i64], here: &Scored) -> Vec<f64> {
    let (found, scores) = here;
    let at = |place: &i64| found.binary_search(place).ok().map_or(0.0, |at| scores[at]);
    places.iter().map(at).collect()
}

fn kept(keep: &[bool], values: &[f64]) -> Vec<f64> {
    keep.iter()
        .zip(values)
        .filter(|each| *each.0)
        .map(|each| *each.1)
        .collect()
}

fn kept_places(keep: &[bool], places: &[i64]) -> Vec<i64> {
    keep.iter()
        .zip(places)
        .filter(|each| *each.0)
        .map(|each| *each.1)
        .collect()
}

fn held_at(eager: &[usize], found: &[Scored], leaves: usize) -> (Vec<i64>, HeldScores) {
    let mut places: Vec<i64> = found.iter().flat_map(|(here, _)| here.iter().copied()).collect();
    places.sort_unstable();
    places.dedup();
    let mut held: HeldScores = vec![None; leaves];
    for (&leaf, here) in eager.iter().zip(found) {
        held[leaf] = Some(matched(&places, here));
    }
    (places, held)
}

impl Grouping<'_> {
    fn within(&mut self, wanted: &[(usize, Vec<i64>)]) -> Result<Vec<Scored>, Failure> {
        let mut asked: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut owners: Vec<usize> = Vec::new();
        for (at, (leaf, ranges)) in wanted.iter().enumerate() {
            for &part in &self.leaves[*leaf].parts {
                let picked = chosen(self.reader.store.parts[part].ranges.ids(), ranges);
                if !picked.is_empty() {
                    asked.push((part, picked));
                    owners.push(at);
                }
            }
        }
        let gathered: Vec<Picked> = self.reader.postings(&asked)?;
        let read = self.reader.scores(gathered)?;
        let mut each: Vec<Vec<Scored>> = wanted
            .iter()
            .map(|(leaf, ranges)| vec![self.leaves[*leaf].exact_within(ranges)])
            .collect();
        for (owner, scored) in owners.into_iter().zip(read) {
            each[owner].push(scored);
        }
        Ok(each.into_iter().map(joined).collect())
    }

    fn offer(&mut self, keep: &[bool], places: &[i64], scores: &[f64]) -> Result<(), Failure> {
        let places = kept_places(keep, places);
        if places.is_empty() {
            return Ok(());
        }
        self.reached = self.reader.reads.offered(&places, &kept(keep, scores))?;
        Ok(())
    }

    fn lazy(&self) -> Vec<usize> {
        if !self.expression.flat() {
            return Vec::new();
        }
        let mut ascending: Vec<usize> = (0..self.leaves.len()).collect();
        ascending.sort_by(|&one, &other| self.leaves[one].most().total_cmp(&self.leaves[other].most()));
        let mosts: Vec<f64> = ascending.iter().map(|&leaf| self.leaves[leaf].most()).collect();
        ascending.truncate(unreached_alone(&mosts, self.reached.threshold));
        ascending
    }

    fn with_lazy(
        &mut self,
        mut places: Vec<i64>,
        mut held: HeldScores,
        mut lazy: Vec<usize>,
    ) -> Result<(Vec<i64>, HeldScores), Failure> {
        let shift = self.reader.store.shape.range_shift();
        lazy.sort_by(|&one, &other| self.leaves[other].most().total_cmp(&self.leaves[one].most()));
        let mut acc = summed(&self.leaves, &held, places.len());
        for (n, &leaf) in lazy.iter().enumerate() {
            let rest = lazy[n..]
                .iter()
                .fold(0.0, |total, &other| total + self.leaves[other].most());
            let near: Mask = acc
                .iter()
                .map(|&sum| may_reach(sum + rest, self.reached.threshold))
                .collect();
            places = kept_places(&near, &places);
            acc = kept(&near, &acc);
            for scores in held.iter_mut().flatten() {
                *scores = kept(&near, scores);
            }
            let wanted = [(leaf, distinct(&places, shift))];
            let here = self.within(&wanted)?.pop().unwrap_or_default();
            let found = matched(&places, &here);
            for (sum, score) in acc.iter_mut().zip(&found) {
                *sum += self.leaves[leaf].bounding * score;
            }
            held[leaf] = Some(found);
        }
        Ok((places, held))
    }

    fn all_of(&mut self, group: &[i64]) -> Result<(), Failure> {
        let shift = self.reader.store.shape.range_shift();
        let mut order: Vec<usize> = (0..self.leaves.len()).collect();
        order.sort_by_key(|&leaf| self.leaves[leaf].size);
        let (mut places, mut held): (Vec<i64>, HeldScores) = (Vec::new(), vec![None; self.leaves.len()]);
        for (n, &leaf) in order.iter().enumerate() {
            let ranges = if n == 0 {
                group.to_vec()
            } else {
                distinct(&places, shift)
            };
            let here = self.within(&[(leaf, ranges)])?.pop().unwrap_or_default();
            if n == 0 {
                (places, held[leaf]) = (here.0, Some(here.1));
                continue;
            }
            let hit: Mask = places.iter().map(|place| here.0.binary_search(place).is_ok()).collect();
            let found = matched(&places, &here);
            places = kept_places(&hit, &places);
            for scores in held.iter_mut().flatten() {
                *scores = kept(&hit, scores);
            }
            held[leaf] = Some(kept(&hit, &found));
        }
        let acc = summed(&self.leaves, &held, places.len());
        let keep: Mask = acc.iter().map(|&sum| sum >= self.reached.threshold).collect();
        self.offer(&keep, &places, &acc)
    }

    pub fn scored_group(&mut self, group: &[i64]) -> Result<(), Failure> {
        if self.expression.conjunction() {
            return self.all_of(group);
        }
        let lazy = self.lazy();
        let eager: Vec<usize> = (0..self.leaves.len()).filter(|leaf| !lazy.contains(leaf)).collect();
        let wanted: Vec<(usize, Vec<i64>)> = eager.iter().map(|&leaf| (leaf, group.to_vec())).collect();
        let found = self.within(&wanted)?;
        if found.iter().all(|(places, _)| places.is_empty()) {
            return Ok(());
        }
        let (places, held) = held_at(&eager, &found, self.leaves.len());
        let threshold = self.reached.threshold;
        if self.expression.flat() {
            let (places, held) = self.with_lazy(places, held, lazy)?;
            let acc = summed(&self.leaves, &held, places.len());
            let keep: Mask = acc.iter().map(|&sum| sum >= threshold && sum > 0.0).collect();
            return self.offer(&keep, &places, &acc);
        }
        let present: Vec<Mask> = held
            .iter()
            .map(|scores| scores.iter().flatten().map(|&score| score > 0.0).collect())
            .collect();
        let (holds, counting) = self.expression.counted(&present);
        let mut acc = vec![0.0; places.len()];
        for (leaf, mask) in counting {
            let (weight, scores) = (self.leaves[leaf].weight, held[leaf].as_deref().unwrap_or_default());
            for ((sum, &score), here) in acc.iter_mut().zip(scores).zip(mask) {
                *sum += if here { weight * score } else { 0.0 };
            }
        }
        let keep: Vec<bool> = holds
            .iter()
            .zip(&acc)
            .map(|(&holds, &sum)| holds && sum >= threshold)
            .collect();
        self.offer(&keep, &places, &acc)
    }
}
