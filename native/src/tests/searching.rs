use std::collections::HashMap;
use std::sync::Arc;

use crate::lists::Block;
use crate::postings::{Shape, score};
use crate::reading::{Failure, LengthBlock, Part, Reached, Reads, Segment, Store};
use crate::scoring::{Expression, Node};
use crate::search::{Asked, ExactList, Plan, searched};
use crate::tests::support::held_ranges;

pub const LENGTH: u32 = 4;
pub const AVERAGE: f64 = 5.0;

pub type Posted = (i64, i64);

#[derive(Default)]
pub struct Fake {
    pub blocks: Vec<Vec<Arc<Block>>>,
    pub lengths: HashMap<(usize, i64), Arc<LengthBlock>>,
    pub k: usize,
    pub best: Vec<(f64, i64)>,
    pub asked_blocks: Vec<(usize, usize)>,
    pub asked_lengths: Vec<(usize, i64)>,
    pub offers: usize,
    pub expiring: bool,
    pub failing: &'static str,
    pub fewer: &'static str,
}

pub struct Built {
    pub plan: Plan,
    pub fake: Fake,
}

pub struct GivenLeaf<'a> {
    pub whole: Vec<(&'a [Posted], Option<usize>)>,
    pub ranged: Vec<(usize, &'a [Posted])>,
    pub bounding: f64,
}

impl Fake {
    fn failed(&self, name: &str) -> Result<(), String> {
        if self.failing == name {
            return Err(format!("{name} failed"));
        }
        Ok(())
    }

    fn fewer<T>(&self, name: &str, mut found: Vec<T>) -> Vec<T> {
        if self.fewer == name {
            found.pop();
        }
        found
    }

    pub fn threshold(&self) -> f64 {
        if self.best.len() >= self.k {
            self.best.last().map_or(0.0, |each| each.0)
        } else {
            0.0
        }
    }
}

impl Reads for Fake {
    fn blocks(&mut self, wanted: &[(usize, usize)]) -> Result<Vec<Arc<Block>>, Failure> {
        self.failed("blocks")?;
        self.asked_blocks.extend_from_slice(wanted);
        let found = wanted
            .iter()
            .map(|&(part, number)| Arc::clone(&self.blocks[part][number]));
        Ok(self.fewer("blocks", found.collect()))
    }

    fn lengths(&mut self, wanted: &[(usize, i64)]) -> Result<Vec<Arc<LengthBlock>>, Failure> {
        self.failed("lengths")?;
        self.asked_lengths.extend_from_slice(wanted);
        let found = wanted.iter().map(|key| Arc::clone(&self.lengths[key]));
        Ok(self.fewer("lengths", found.collect()))
    }

    fn offered(&mut self, places: &[i64], scores: &[f64]) -> Result<Reached, Failure> {
        self.failed("offered")?;
        self.offers += 1;
        self.best.extend(scores.iter().copied().zip(places.iter().copied()));
        self.best
            .sort_by(|one, other| other.0.total_cmp(&one.0).then(one.1.cmp(&other.1)));
        self.best.truncate(self.k);
        Ok(Reached {
            threshold: self.threshold(),
            held: self.best.len(),
        })
    }

    fn expired(&mut self) -> Result<bool, Failure> {
        self.failed("expired")?;
        Ok(self.expiring)
    }
}

pub fn shape() -> Result<Shape, String> {
    Shape::new(2, 2, 3, 3)
}

fn ranged(postings: &[Posted], shape: Shape) -> Result<(Arc<crate::ranges::Ranges>, Vec<Arc<Block>>), String> {
    let (mut ids, mut sizes): (Vec<i64>, Vec<u8>) = (Vec::new(), Vec::new());
    for &(place, _) in postings {
        let range = place >> shape.range_shift();
        match (ids.last(), sizes.last_mut()) {
            (Some(&last), Some(size)) if last == range => *size += 1,
            _ => {
                ids.push(range);
                sizes.push(1);
            }
        }
    }
    let ranges = held_ranges(&ids, &sizes, &vec![255; ids.len()])?;
    let blocks = postings.chunks(shape.block()).map(|chunk| {
        let (ids, counts) = chunk.iter().copied().unzip();
        Arc::new(Block { ids, counts })
    });
    Ok((Arc::new(ranges), blocks.collect()))
}

pub fn segments(bounds: &[(i64, i64, &[i64])]) -> Vec<Segment> {
    let each = bounds.iter().map(|&(first, last, dead)| Segment {
        first,
        last,
        dead: dead.to_vec(),
        lengths: Arc::default(),
    });
    each.collect()
}

pub fn built(segments: Vec<Segment>, given: &[GivenLeaf], nodes: Vec<Node>, k: usize) -> Result<Built, String> {
    let shape = shape()?;
    let mut fake = Fake { k, ..Fake::default() };
    for (at, held) in segments.iter().enumerate() {
        for number in held.first >> shape.lengths_shift()..=held.last >> shape.lengths_shift() {
            fake.lengths
                .insert((at, number), Arc::new(vec![LENGTH; 1 << shape.lengths_shift()]));
        }
    }
    let (mut parts, mut leaves) = (Vec::new(), Vec::new());
    for leaf in given {
        let mut owned = Vec::new();
        for &(segment, postings) in &leaf.ranged {
            let (ranges, blocks) = ranged(postings, shape)?;
            owned.push(parts.len());
            parts.push(Part {
                segment,
                ranges,
                slack: 1.0,
            });
            fake.blocks.push(blocks);
        }
        let exact = leaf.whole.iter().map(|&(postings, segment)| {
            let (places, counts) = postings.iter().copied().unzip();
            ExactList {
                places,
                counts,
                segment,
            }
        });
        leaves.push(Asked {
            exact: exact.collect(),
            parts: owned,
            bounding: leaf.bounding,
            weight: leaf.bounding,
        });
    }
    let expression = Expression::new(nodes, given.len())?;
    let store = Store {
        segments,
        parts,
        shape,
        average: AVERAGE,
    };
    let plan = Plan {
        store,
        leaves,
        expression,
        batches: (1, 4),
    };
    Ok(Built { plan, fake })
}

pub fn run(made: Built) -> Result<(bool, Fake), String> {
    let Built { plan, mut fake } = made;
    let start = Reached {
        threshold: 0.0,
        held: 0,
    };
    let cut = searched(plan, start, &mut fake).map_err(|failure| failure.to_string())?;
    Ok((cut, fake))
}

pub fn scored(count: i64, bounding: f64) -> f64 {
    bounding * score(count, LENGTH, AVERAGE)
}
