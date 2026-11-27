use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use crate::lists::{Coded, block, gaps, placed};
use crate::merge::{Fed, Merger};
use crate::postings::{Kept, Piece, Row, Shape};
use crate::segment::{Held, Passages, Written};
use crate::words::Table;

const POINTS: u32 = 0x11_0000;
const APART: u32 = 0x20;
const DROPPED: u32 = u32::MAX;
pub const AVERAGE: f64 = 4.0;

pub type Decoded = (Vec<i64>, Vec<i64>, Vec<i64>);
pub type Shown = (String, usize, Option<Coded>, Option<Vec<u8>>);
pub type ShownBlock = (String, usize, Coded);

pub struct Outcome {
    pub rows: Vec<Row>,
    pub pieces: Vec<Piece>,
    pub steps: Vec<Vec<String>>,
}

static TABLE: LazyLock<Result<Table, String>> = LazyLock::new(|| {
    let mut stored = Vec::new();
    for point in 0..POINTS {
        let kept = match char::from_u32(point) {
            Some('\'') => DROPPED,
            Some(each) if each.is_alphanumeric() => each.to_lowercase().next().map_or(APART, u32::from),
            _ => APART,
        };
        stored.extend_from_slice(&kept.to_le_bytes());
    }
    Table::new(&stored)
});

pub fn refused<T>(found: Result<T, String>, saying: &str) -> bool {
    found.is_err_and(|problem| problem.contains(saying))
}

pub fn table() -> Result<&'static Table, String> {
    TABLE.as_ref().map_err(Clone::clone)
}

pub fn small() -> Result<Shape, String> {
    Shape::new(2, 2, 3, 3)
}

pub fn wide() -> Result<Shape, String> {
    Shape::new(7, 4096, 4096, 10)
}

pub fn words(text: &str) -> Result<Vec<String>, String> {
    let mut found = Vec::new();
    table()?.words(text, |word| {
        found.push(word.to_owned());
        Ok(())
    })?;
    Ok(found)
}

pub fn built(texts: &[(i64, &str)], shape: Shape) -> Result<Written, String> {
    let mut held = Held::new(usize::MAX, usize::MAX);
    let places: Vec<i64> = texts.iter().map(|each| each.0).collect();
    let bodies: Vec<&str> = texts.iter().map(|each| each.1).collect();
    let passages = Passages {
        places: &places,
        of: &vec![0; texts.len()],
        fields: &[("", "")],
        texts: &bodies,
    };
    if held.taken(table()?, &passages)? != texts.len() {
        return Err("a segment of every token took fewer passages than it was given".to_owned());
    }
    held.written(AVERAGE, shape)
}

fn listed(coded: &Coded, found: &mut Decoded) -> Result<(), String> {
    let held = block(&coded.0)?;
    let mut places = gaps(&coded.1)?;
    placed(&mut places, &held.counts)?;
    found.0.extend(held.ids);
    found.1.extend(held.counts);
    found.2.extend(places);
    Ok(())
}

pub fn decoded(row: &Row) -> Result<Decoded, String> {
    let mut found = Decoded::default();
    match &row.held {
        Kept::Short(coded) => listed(coded, &mut found)?,
        Kept::Long { blocks, .. } => {
            for coded in blocks {
                listed(coded, &mut found)?;
            }
        }
    }
    Ok(found)
}

pub fn row<'a>(made: &'a Written, term: &str) -> Result<&'a Row, String> {
    let found = made.rows.iter().find(|row| &*row.term == term);
    found.ok_or_else(|| format!("the segment holds no list of {term}"))
}

pub fn shown(rows: &[Row], pieces: &[Piece]) -> (Vec<Shown>, Vec<ShownBlock>) {
    let (mut lists, mut blocks) = (Vec::new(), Vec::new());
    for row in rows {
        match &row.held {
            Kept::Short(coded) => lists.push((row.term.to_string(), row.chunks, Some(coded.clone()), None)),
            Kept::Long { ranges, blocks: held } => {
                lists.push((row.term.to_string(), row.chunks, None, Some(ranges.clone())));
                for (number, coded) in held.iter().enumerate() {
                    blocks.push((row.term.to_string(), number, coded.clone()));
                }
            }
        }
    }
    for (term, number, coded) in pieces {
        blocks.push((term.to_string(), *number, coded.clone()));
    }
    blocks.sort();
    (lists, blocks)
}

pub fn sources(made: &Written, shape: Shape) -> Vec<Vec<Fed>> {
    let (mut short, mut blocks) = (Vec::new(), Vec::new());
    for row in &made.rows {
        let term: Arc<str> = Arc::from(&*row.term);
        match &row.held {
            Kept::Short(coded) => short.push(Fed {
                term,
                chunks: row.chunks,
                postings: coded.0.clone(),
                positions: coded.1.clone(),
            }),
            Kept::Long { blocks: held, .. } => {
                for coded in held {
                    blocks.push(Fed {
                        term: Arc::clone(&term),
                        chunks: shape.block(),
                        postings: coded.0.clone(),
                        positions: coded.1.clone(),
                    });
                }
            }
        }
    }
    vec![short, blocks]
}

pub fn merged(merger: &mut Merger, given: Vec<Vec<Fed>>, at_once: usize) -> Result<Outcome, String> {
    let mut left: Vec<VecDeque<Fed>> = given.into_iter().map(VecDeque::from).collect();
    let mut made = Outcome {
        rows: Vec::new(),
        pieces: Vec::new(),
        steps: Vec::new(),
    };
    loop {
        let step = merger.stepped()?;
        made.steps
            .push(step.rows.iter().map(|row| row.term.to_string()).collect());
        made.rows.extend(step.rows);
        made.pieces.extend(step.pieces);
        if step.done {
            return Ok(made);
        }
        for source in step.wanted {
            let rows = left
                .get_mut(source)
                .ok_or_else(|| "a merge asked for a source it has not".to_owned())?;
            while merger
                .fed(source, (0..at_once).map_while(|_| rows.pop_front()).collect())
                .ok_or_else(|| "a merge refused a source it asked for".to_owned())?
            {}
        }
    }
}
