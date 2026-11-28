use crate::groups::{Grouping, Leaf};
use crate::reading::{Failure, Part, Picked, Reached, Reader, Reads, Store, live};
use crate::scoring::{Expression, Mask, may_reach, unquantized};

const UNPLANNED: &str = "a search is planned with parts, segments or batches it does not hold";

pub struct ExactList {
    pub places: Vec<i64>,
    pub counts: Vec<i64>,
    pub segment: Option<usize>,
}

pub struct Asked {
    pub exact: Vec<ExactList>,
    pub parts: Vec<usize>,
    pub bounding: f64,
    pub weight: f64,
}

pub struct Plan {
    pub store: Store,
    pub leaves: Vec<Asked>,
    pub expression: Expression,
    pub batches: (usize, usize),
}

struct Bounded {
    start: i64,
    bound: Vec<f64>,
    present: Vec<Mask>,
}

fn sound(plan: &Plan) -> bool {
    let (segments, parts) = (plan.store.segments.len(), plan.store.parts.len());
    let listed = |each: &ExactList| each.segment.is_none_or(|segment| segment < segments);
    let asked = |leaf: &Asked| leaf.exact.iter().all(listed) && leaf.parts.iter().all(|&part| part < parts);
    let placed = plan.store.parts.iter().all(|part| part.segment < segments);
    plan.batches.0 > 0 && plan.batches.1 > 0 && placed && plan.leaves.iter().all(asked)
}

fn gathered(store: &Store, exact: Vec<ExactList>) -> Picked {
    let mut paired: Vec<(i64, i64)> = Vec::new();
    for listed in exact {
        let (places, counts) = match listed.segment {
            Some(segment) => live(&store.segments[segment].dead, listed.places, listed.counts),
            None => (listed.places, listed.counts),
        };
        paired.extend(places.into_iter().zip(counts));
    }
    paired.sort_by_key(|each| each.0);
    let (places, counts) = paired.into_iter().unzip();
    Picked { places, counts }
}

fn leaves(reader: &mut Reader<'_>, asked: Vec<Asked>) -> Result<Vec<Leaf>, Failure> {
    let (store, shift) = (reader.store, reader.store.shape.range_shift());
    let (mut each, mut rest) = (Vec::new(), Vec::new());
    for leaf in asked {
        each.push(gathered(store, leaf.exact));
        rest.push((leaf.parts, leaf.bounding, leaf.weight));
    }
    let scored = reader.scores(each)?;
    let made = scored
        .into_iter()
        .zip(rest)
        .map(|((places, scores), (parts, bounding, weight))| {
            let size = places.len()
                + parts
                    .iter()
                    .map(|&part| store.parts[part].ranges.total())
                    .sum::<usize>();
            Leaf {
                ranges: places.iter().map(|place| place >> shift).collect(),
                places,
                scores,
                parts,
                bounding,
                weight,
                size,
            }
        });
    Ok(made.collect())
}

fn span(leaves: &[Leaf], parts: &[Part]) -> Option<(i64, i64)> {
    let exact = leaves
        .iter()
        .filter_map(|leaf| leaf.ranges.first().zip(leaf.ranges.last()));
    let ranged = parts
        .iter()
        .filter_map(|part| part.ranges.ids().first().zip(part.ranges.ids().last()));
    exact.chain(ranged).fold(None, |held, (&low, &high)| {
        Some(held.map_or((low, high), |(start, end): (i64, i64)| (start.min(low), end.max(high))))
    })
}

fn at(id: i64, start: i64) -> usize {
    usize::try_from(id - start).unwrap_or(0)
}

fn bounded(leaves: &[Leaf], parts: &[Part]) -> Bounded {
    let Some((start, end)) = span(leaves, parts) else {
        return Bounded {
            start: 0,
            bound: Vec::new(),
            present: leaves.iter().map(|_| Vec::new()).collect(),
        };
    };
    let size = at(end, start) + 1;
    let (mut bound, mut present) = (vec![0.0; size], Vec::with_capacity(leaves.len()));
    for leaf in leaves {
        let mut here = vec![0.0f64; size];
        for (&range, &score) in leaf.ranges.iter().zip(&leaf.scores) {
            let held = &mut here[at(range, start)];
            *held = held.max(score);
        }
        for part in leaf.parts.iter().map(|&part| &parts[part]) {
            for (&id, &best) in part.ranges.ids().iter().zip(part.ranges.best()) {
                let held = &mut here[at(id, start)];
                *held = held.max(unquantized(best) * part.slack);
            }
        }
        for (sum, &most) in bound.iter_mut().zip(&here) {
            *sum += leaf.bounding * most;
        }
        present.push(here.iter().map(|&most| most > 0.0).collect());
    }
    Bounded { start, bound, present }
}

fn group_of(remaining: &mut Vec<(i64, f64)>, batch: usize) -> Vec<i64> {
    let taken = if remaining.len() > batch {
        remaining.select_nth_unstable_by(batch - 1, |one, other| other.1.total_cmp(&one.1));
        let rest = remaining.split_off(batch);
        std::mem::replace(remaining, rest)
    } else {
        std::mem::take(remaining)
    };
    let mut group: Vec<i64> = taken.into_iter().map(|each| each.0).collect();
    group.sort_unstable();
    group
}

pub fn searched(plan: Plan, reached: Reached, reads: &mut dyn Reads) -> Result<bool, Failure> {
    if !sound(&plan) {
        return Err(UNPLANNED.into());
    }
    let Plan {
        store,
        leaves: asked,
        expression,
        batches,
    } = plan;
    let mut reader = Reader::new(&store, reads);
    let leaves = leaves(&mut reader, asked)?;
    let held = bounded(&leaves, &store.parts);
    let holds = expression.may_hold(&held.present);
    let mut remaining: Vec<(i64, f64)> = (held.start..)
        .zip(held.bound.iter().zip(holds))
        .filter(|(_, (bound, holds))| *holds && **bound > 0.0)
        .map(|(range, (&bound, _))| (range, bound))
        .collect();
    let mut grouping = Grouping {
        reader,
        leaves,
        expression,
        reached,
    };
    let (mut batch, mut began) = (batches.0, false);
    while !remaining.is_empty() {
        let threshold = grouping.reached.threshold;
        remaining.retain(|each| may_reach(each.1, threshold));
        if began && !remaining.is_empty() && grouping.reached.held > 0 && grouping.reader.reads.expired()? {
            return Ok(true);
        }
        let group = group_of(&mut remaining, batch);
        if !group.is_empty() {
            grouping.scored_group(&group)?;
            began = true;
        }
        batch = batch.saturating_mul(2).min(batches.1);
    }
    Ok(false)
}
