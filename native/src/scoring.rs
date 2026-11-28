use std::mem;

use crate::postings::{STEPS, TOP};

const ROUNDING: f64 = 1e-9;
const UNTOLD: &str = "a query is told with parts that are not each under one other, or with leaves it does not hold";

pub type Mask = Vec<bool>;
pub type Counting = Vec<(usize, Mask)>;

pub enum Node {
    Leaf(usize),
    Not([usize; 2]),
    Or(Vec<usize>),
    And(Vec<usize>),
}

pub struct Expression {
    nodes: Vec<Node>,
}

pub fn may_reach(bound: f64, threshold: f64) -> bool {
    bound * (1.0 + ROUNDING) >= threshold
}

pub fn unreached_alone(ascending: &[f64], threshold: f64) -> usize {
    let mut total = 0.0;
    for (at, &most) in ascending.iter().enumerate() {
        total += most;
        if may_reach(total, threshold) {
            return at;
        }
    }
    ascending.len()
}

pub fn unquantized(step: u8) -> f64 {
    f64::from(step) * (TOP / STEPS)
}

fn joined(parts: Vec<Mask>, both: fn(bool, bool) -> bool) -> Mask {
    let mut parts = parts.into_iter();
    let first = parts.next().unwrap_or_default();
    parts.fold(first, |held, part| {
        held.into_iter()
            .zip(part)
            .map(|(one, other)| both(one, other))
            .collect()
    })
}

fn narrowed(counting: Counting, held: &[bool]) -> Counting {
    let each = |(leaf, mask): (usize, Mask)| {
        let kept = mask.into_iter().zip(held).map(|(one, &other)| one && other);
        (leaf, kept.collect())
    };
    counting.into_iter().map(each).collect()
}

impl Node {
    fn parts(&self) -> &[usize] {
        match self {
            Self::Leaf(_) => &[],
            Self::Not(both) => both,
            Self::Or(parts) | Self::And(parts) => parts,
        }
    }

    fn sound(&self, at: usize, leaves: usize) -> bool {
        let parts = self.parts();
        let inside = parts.iter().all(|&part| part < at);
        match self {
            Self::Leaf(leaf) => *leaf < leaves,
            Self::Not(_) => inside,
            Self::Or(_) | Self::And(_) => inside && !parts.is_empty(),
        }
    }

    fn of_leaves(&self, nodes: &[Self]) -> bool {
        let leaf = |&part: &usize| nodes.get(part).is_some_and(|node| matches!(node, Self::Leaf(_)));
        self.parts().iter().all(leaf)
    }
}

impl Expression {
    pub fn new(nodes: Vec<Node>, leaves: usize) -> Result<Self, String> {
        let mut under = vec![0usize; nodes.len()];
        for (at, node) in nodes.iter().enumerate() {
            if !node.sound(at, leaves) {
                return Err(UNTOLD.to_owned());
            }
            for &part in node.parts() {
                under[part] += 1;
            }
        }
        let Some((root, rest)) = under.split_last() else {
            return Err(UNTOLD.to_owned());
        };
        if *root != 0 || rest.iter().any(|&times| times != 1) {
            return Err(UNTOLD.to_owned());
        }
        Ok(Self { nodes })
    }

    fn root(&self) -> Option<&Node> {
        self.nodes.last()
    }

    pub fn flat(&self) -> bool {
        self.root()
            .is_some_and(|root| matches!(root, Node::Or(_)) && root.of_leaves(&self.nodes))
    }

    pub fn conjunction(&self) -> bool {
        self.root()
            .is_some_and(|root| matches!(root, Node::And(_)) && root.of_leaves(&self.nodes))
    }

    pub fn may_hold(&self, present: &[Mask]) -> Mask {
        let mut found: Vec<Mask> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let mut taken =
                |parts: &[usize]| -> Vec<Mask> { parts.iter().map(|&part| mem::take(&mut found[part])).collect() };
            let held = match node {
                Node::Leaf(leaf) => present.get(*leaf).cloned().unwrap_or_default(),
                Node::Not([kept, _]) => mem::take(&mut found[*kept]),
                Node::Or(parts) => joined(taken(parts), |one, other| one || other),
                Node::And(parts) => joined(taken(parts), |one, other| one && other),
            };
            found.push(held);
        }
        found.pop().unwrap_or_default()
    }

    pub fn counted(&self, present: &[Mask]) -> (Mask, Counting) {
        let mut found: Vec<(Mask, Counting)> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let held = match node {
                Node::Leaf(leaf) => {
                    let mask = present.get(*leaf).cloned().unwrap_or_default();
                    (mask.clone(), vec![(*leaf, mask)])
                }
                Node::Not([kept, dropped]) => {
                    let ((mask, counting), (gone, _)) = (mem::take(&mut found[*kept]), mem::take(&mut found[*dropped]));
                    let held: Mask = mask.into_iter().zip(gone).map(|(one, other)| one && !other).collect();
                    let counting = narrowed(counting, &held);
                    (held, counting)
                }
                Node::Or(parts) | Node::And(parts) => {
                    let parts: Vec<(Mask, Counting)> = parts.iter().map(|&part| mem::take(&mut found[part])).collect();
                    let (masks, countings): (Vec<Mask>, Vec<Counting>) = parts.into_iter().unzip();
                    let counting: Counting = countings.into_iter().flatten().collect();
                    if matches!(node, Node::Or(_)) {
                        (joined(masks, |one, other| one || other), counting)
                    } else {
                        let held = joined(masks, |one, other| one && other);
                        let counting = narrowed(counting, &held);
                        (held, counting)
                    }
                }
            };
            found.push(held);
        }
        found.pop().unwrap_or_default()
    }
}
