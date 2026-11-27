use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};

use crate::lists::{Coded, block, gaps, placed};
use crate::merge::{Fed, Merger};
use crate::packs::{counted, entries};
use crate::postings::{Filed, Shape};
use crate::segment::{Held, Passages, Written};
use crate::words::Table;

const POINTS: u32 = 0x11_0000;
const APART: u32 = 0x20;
const DROPPED: u32 = u32::MAX;
pub const AVERAGE: f64 = 4.0;

pub type Decoded = (Vec<i64>, Vec<i64>, Vec<i64>);
pub type Shown = (String, usize, Option<Coded>, Option<Vec<u8>>);
pub type ShownBlock = (String, usize, Coded);
pub type Told = (Vec<String>, Vec<Shown>, Vec<ShownBlock>);

type Left = (VecDeque<(Vec<u8>, Vec<u8>)>, VecDeque<Fed>);

pub struct Feed {
    pub packs: Vec<(Vec<u8>, Vec<u8>)>,
    pub blocks: Vec<Fed>,
}

pub struct Outcome {
    pub filed: Filed,
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

pub fn taken(held: &mut Held, texts: &[(i64, &str)]) -> Result<usize, String> {
    let places: Vec<i64> = texts.iter().map(|each| each.0).collect();
    let bodies: Vec<&str> = texts.iter().map(|each| each.1).collect();
    let passages = Passages {
        places: &places,
        of: &vec![0; texts.len()],
        fields: &[("", "")],
        texts: &bodies,
    };
    held.taken(table()?, &passages)
}

pub fn built(texts: &[(i64, &str)], shape: Shape) -> Result<Written, String> {
    let mut held = Held::new(usize::MAX, usize::MAX);
    if taken(&mut held, texts)? != texts.len() {
        return Err("a segment of every token took fewer passages than it was given".to_owned());
    }
    held.written(AVERAGE, shape)
}

pub fn shown(filed: &Filed) -> Result<Told, String> {
    let (mut firsts, mut lists) = (Vec::new(), Vec::new());
    for pack in &filed.packs {
        firsts.push(pack.first.to_string());
        for each in entries(&pack.lists, Some(&pack.places), Some(&pack.first))? {
            let coded = (each.postings.to_vec(), each.positions.to_vec());
            let ranges = filed.ranged.iter().find(|held| &*held.term == each.term);
            let short = (!each.postings.is_empty()).then_some(coded);
            lists.push((
                each.term.to_owned(),
                each.chunks,
                short,
                ranges.map(|held| held.ranges.clone()),
            ));
        }
    }
    let mut blocks: Vec<ShownBlock> = filed
        .blocks
        .iter()
        .map(|each| (each.0.to_string(), each.1, each.2.clone()))
        .collect();
    blocks.sort();
    Ok((firsts, lists, blocks))
}

pub fn terms(filed: &Filed) -> Result<Vec<String>, String> {
    Ok(shown(filed)?.1.into_iter().map(|each| each.0).collect())
}

pub fn counts(filed: &Filed) -> Result<Vec<(String, usize)>, String> {
    let mut found = Vec::new();
    for pack in &filed.packs {
        let held = counted(&pack.lists, None, "", None)?;
        found.extend(held.into_iter().map(|(term, chunks)| (term.to_owned(), chunks)));
    }
    Ok(found)
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

pub fn decoded(filed: &Filed, term: &str) -> Result<(usize, Decoded), String> {
    let (_, lists, blocks) = shown(filed)?;
    let held = lists.iter().find(|each| each.0 == term);
    let held = held.ok_or_else(|| format!("the segment holds no list of {term}"))?;
    let mut found = Decoded::default();
    match &held.2 {
        Some(coded) => listed(coded, &mut found)?,
        None => {
            for each in blocks.iter().filter(|each| each.0 == term) {
                listed(&each.2, &mut found)?;
            }
        }
    }
    Ok((held.1, found))
}

pub fn long(filed: &Filed, term: &str) -> usize {
    filed.blocks.iter().filter(|each| &*each.0 == term).count()
}

pub fn sources(made: &Written, shape: Shape) -> Feed {
    let packs = made
        .filed
        .packs
        .iter()
        .map(|pack| (pack.lists.clone(), pack.places.clone()));
    let blocks = made.filed.blocks.iter().map(|(term, _, coded)| Fed {
        term: Arc::clone(term),
        chunks: shape.block(),
        postings: coded.0.clone(),
        positions: coded.1.clone(),
    });
    Feed {
        packs: packs.collect(),
        blocks: blocks.collect(),
    }
}

pub fn fed_whole(merger: &mut Merger, given: Vec<Feed>) -> Result<(), String> {
    for (number, each) in given.into_iter().enumerate() {
        merger.fed_packs(2 * number, &each.packs)?;
        merger.fed(2 * number + 1, each.blocks)?;
        merger.ended(2 * number);
        merger.ended(2 * number + 1);
    }
    Ok(())
}

pub fn merged(merger: &mut Merger, given: Vec<Feed>, at_once: usize) -> Result<Outcome, String> {
    let mut left: Vec<Left> = given
        .into_iter()
        .map(|each| (each.packs.into(), each.blocks.into()))
        .collect();
    let mut made = Outcome {
        filed: Filed::default(),
        steps: Vec::new(),
    };
    loop {
        let step = merger.stepped()?;
        made.steps.push(terms(&step.filed)?);
        made.filed.packs.extend(step.filed.packs);
        made.filed.ranged.extend(step.filed.ranged);
        made.filed.blocks.extend(step.filed.blocks);
        if step.done {
            return Ok(made);
        }
        for source in step.wanted {
            let Some(held) = left.get_mut(source / 2) else {
                return Err("a merge asked for a source it has not".to_owned());
            };
            let mut more = Some(true);
            while more == Some(true) {
                more = if source % 2 == 0 {
                    let packs: Vec<(Vec<u8>, Vec<u8>)> = (0..at_once).map_while(|_| held.0.pop_front()).collect();
                    if packs.is_empty() {
                        merger.ended(source)
                    } else {
                        merger.fed_packs(source, &packs)?
                    }
                } else {
                    let rows: Vec<Fed> = (0..at_once).map_while(|_| held.1.pop_front()).collect();
                    if rows.is_empty() {
                        merger.ended(source)
                    } else {
                        merger.fed(source, rows)?
                    }
                };
            }
        }
    }
}
