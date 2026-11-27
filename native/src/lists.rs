use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyList};
use rayon::prelude::*;

use crate::numbers::{little_endian, read, written};
use crate::packed::{unwrapped, wrapped};
use crate::segment::{lengths_written, one_fewer, ranges_packed};

const HEAD: usize = 12;
const FIRST: usize = 8;
const GAP: usize = 4;
const MOST_POSITION: i64 = 1 << 32;
const TOGETHER: usize = 256;
const FAR_APART: &str = "ids not ascending, or more than 2**32 apart: more than a block holds";
const PAST: &str = "a list holds a position past any passage";

pub struct Block {
    pub ids: Vec<i64>,
    pub counts: Vec<i64>,
}

struct Blocks {
    ids: Vec<i64>,
    counts: Vec<i64>,
    bounds: Vec<usize>,
}

pub type Coded = (Vec<u8>, Vec<u8>);

fn each_of<T, F>(rows: &[PyBackedBytes], one: F) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(&[u8]) -> Result<T, String> + Sync,
{
    if rows.len() < TOGETHER {
        rows.iter().map(|row| one(row)).collect()
    } else {
        rows.par_iter().with_min_len(TOGETHER).map(|row| one(row)).collect()
    }
}

pub fn block(data: &[u8]) -> Result<Block, String> {
    let raw = unwrapped(data)?;
    let opening = raw
        .split_first_chunk::<FIRST>()
        .and_then(|(first, rest)| Some((first, rest.split_first_chunk::<GAP>()?)));
    let Some((first, (held, rest))) = opening else {
        return Err("a block is shorter than its head".to_owned());
    };
    let held = usize::try_from(u32::from_le_bytes(*held)).unwrap_or(usize::MAX);
    let split = held
        .checked_sub(1)
        .and_then(|gaps| gaps.checked_mul(GAP))
        .filter(|&split| split <= rest.len());
    let Some((steps, counted)) = split.map(|split| rest.split_at(split)) else {
        return Err("a block holds fewer ids than its head says".to_owned());
    };
    let mut ids = Vec::with_capacity(held);
    let mut at = i64::from_le_bytes(*first);
    ids.push(at);
    for gap in steps.as_chunks::<GAP>().0 {
        let step = i64::from(u32::from_le_bytes(*gap));
        if step < 1 {
            return Err("a block holds ids out of order".to_owned());
        }
        at = at.wrapping_add(step);
        ids.push(at);
    }
    let mut counts = Vec::with_capacity(held);
    read(counted, &mut counts)?;
    if counts.len() != held {
        return Err("a block holds counts other than its ids".to_owned());
    }
    if counts.iter().any(|&count| count < 1) {
        return Err("a block holds a count below one".to_owned());
    }
    Ok(Block { ids, counts })
}

fn blocks(rows: &[PyBackedBytes]) -> Result<Blocks, String> {
    let found = each_of(rows, block)?;
    let held = found.iter().map(|one| one.ids.len()).sum();
    let (mut ids, mut counts) = (Vec::with_capacity(held), Vec::with_capacity(held));
    let mut bounds = Vec::with_capacity(found.len() + 1);
    bounds.push(0);
    for one in &found {
        ids.extend_from_slice(&one.ids);
        counts.extend_from_slice(&one.counts);
        bounds.push(ids.len());
    }
    Ok(Blocks { ids, counts, bounds })
}

pub fn gaps(data: &[u8]) -> Result<Vec<i64>, String> {
    let raw = unwrapped(data)?;
    let mut found = Vec::with_capacity(raw.len());
    read(&raw, &mut found)?;
    Ok(found)
}

pub fn placed(found: &mut [i64], counts: &[i64]) -> Result<(), String> {
    let mut at = 0usize;
    for &count in counts {
        let end = usize::try_from(count)
            .ok()
            .and_then(|held| at.checked_add(held))
            .filter(|&end| end <= found.len());
        let Some(end) = end else {
            return Err("a list holds positions other than its counts".to_owned());
        };
        let mut place = 0i64;
        for gap in &mut found[at..end] {
            if *gap >= MOST_POSITION {
                return Err(PAST.to_owned());
            }
            place += *gap;
            *gap = place;
        }
        if place >= MOST_POSITION {
            return Err(PAST.to_owned());
        }
        at = end;
    }
    if at == found.len() {
        Ok(())
    } else {
        Err("a list holds positions other than its counts".to_owned())
    }
}

fn places(rows: &[PyBackedBytes], counts: &[i64]) -> Result<Vec<i64>, String> {
    let parts = each_of(rows, gaps)?;
    let mut found = Vec::with_capacity(parts.iter().map(Vec::len).sum());
    for part in &parts {
        found.extend_from_slice(part);
    }
    placed(&mut found, counts)?;
    Ok(found)
}

pub struct Lists<'a> {
    pub ids: &'a [i64],
    pub counts: &'a [i64],
    pub positions: &'a [i64],
    pub offsets: Vec<usize>,
}

fn offsets(counts: &[i64], positions: usize) -> Result<Vec<usize>, String> {
    let mut found = Vec::with_capacity(counts.len() + 1);
    let mut held = 0usize;
    found.push(held);
    for &count in counts {
        let more = usize::try_from(count).ok().and_then(|size| held.checked_add(size));
        held = more.ok_or_else(|| "a count is no number of positions".to_owned())?;
        found.push(held);
    }
    if held == positions {
        Ok(found)
    } else {
        Err("lists hold positions other than their counts".to_owned())
    }
}

pub fn list(lists: &Lists, start: usize, end: usize) -> Result<Coded, String> {
    if start >= end || end > lists.ids.len() {
        return Err("a list holds no id, or more than there are".to_owned());
    }
    let held = u32::try_from(end - start).map_err(|_| FAR_APART.to_owned())?;
    let mut body = Vec::with_capacity(HEAD + (end - start) * (GAP + 1));
    body.extend_from_slice(&lists.ids[start].to_le_bytes());
    body.extend_from_slice(&held.to_le_bytes());
    for pair in lists.ids[start..end].windows(2) {
        let gap = pair[1]
            .checked_sub(pair[0])
            .filter(|&gap| gap >= 1)
            .and_then(|gap| u32::try_from(gap).ok());
        body.extend_from_slice(&gap.ok_or_else(|| FAR_APART.to_owned())?.to_le_bytes());
    }
    for &count in &lists.counts[start..end] {
        written(&mut body, count)?;
    }
    let mut placed = Vec::with_capacity(lists.offsets[end] - lists.offsets[start]);
    for at in start..end {
        let mut before = 0i64;
        for &place in &lists.positions[lists.offsets[at]..lists.offsets[at + 1]] {
            written(&mut placed, place.wrapping_sub(before))?;
            before = place;
        }
    }
    Ok((wrapped(&body)?, wrapped(&placed)?))
}

fn coded(lists: &Lists, bounds: &[usize]) -> Result<Vec<Coded>, String> {
    if bounds.len() <= TOGETHER {
        bounds.windows(2).map(|pair| list(lists, pair[0], pair[1])).collect()
    } else {
        bounds
            .par_windows(2)
            .with_min_len(TOGETHER)
            .map(|pair| list(lists, pair[0], pair[1]))
            .collect()
    }
}

pub fn bytes_of(each: &Bound<'_, PyAny>) -> PyResult<Vec<PyBackedBytes>> {
    each.extract()
}

pub fn numbers_of(held: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    PyBuffer::<i64>::get(held)?.to_vec(held.py())
}

#[pyfunction]
pub fn blocked_all<'py>(each: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>)> {
    let (py, rows) = (each.py(), bytes_of(each)?);
    let found = py.detach(|| blocks(&rows)).map_err(PyValueError::new_err)?;
    let (ids, counts) = (little_endian(&found.ids), little_endian(&found.counts));
    Ok((PyBytes::new(py, &ids), PyBytes::new(py, &counts), found.bounds))
}

#[pyfunction]
pub fn positions_all<'py>(each: &Bound<'py, PyAny>, counts: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let (py, rows, counts) = (each.py(), bytes_of(each)?, numbers_of(counts)?);
    let found = py.detach(|| places(&rows, &counts).map(|found| little_endian(&found)));
    Ok(PyBytes::new(py, &found.map_err(PyValueError::new_err)?))
}

#[pyfunction]
pub fn encoded<'py>(
    ids: &Bound<'py, PyAny>,
    counts: &Bound<'py, PyAny>,
    positions: &Bound<'py, PyAny>,
    bounds: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let (py, bounds) = (ids.py(), bounds.extract::<Vec<usize>>()?);
    let (ids, counts, positions) = (numbers_of(ids)?, numbers_of(counts)?, numbers_of(positions)?);
    let made = py.detach(|| {
        let offsets = offsets(&counts, positions.len())?;
        coded(
            &Lists {
                ids: &ids,
                counts: &counts,
                positions: &positions,
                offsets,
            },
            &bounds,
        )
    });
    let made = made.map_err(PyValueError::new_err)?;
    let postings = PyList::new(py, made.iter().map(|each| PyBytes::new(py, &each.0)))?;
    let placed = PyList::new(py, made.iter().map(|each| PyBytes::new(py, &each.1)))?;
    Ok((postings, placed))
}

#[pyfunction]
pub fn ranged<'py>(
    ids: &Bound<'py, PyAny>,
    sizes: &Bound<'py, PyAny>,
    best: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (py, ids, best) = (ids.py(), numbers_of(ids)?, best.extract::<PyBackedBytes>()?);
    let fewer: Result<Vec<u8>, String> = numbers_of(sizes)?.into_iter().map(one_fewer).collect();
    let fewer = fewer.map_err(PyValueError::new_err)?;
    if ids.len() != fewer.len() || ids.len() != best.len() {
        return Err(PyValueError::new_err("each range has an id, a size and a best score"));
    }
    let packed = py
        .detach(|| ranges_packed(&ids, &fewer, &best))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new(py, &packed))
}

#[pyfunction]
pub fn with_lengths<'py>(
    ids: &Bound<'py, PyAny>,
    lengths: &Bound<'py, PyAny>,
    shift: u32,
) -> PyResult<Bound<'py, PyList>> {
    let (py, ids) = (ids.py(), numbers_of(ids)?);
    let lengths: Result<Vec<u32>, _> = numbers_of(lengths)?.into_iter().map(u32::try_from).collect();
    let lengths = lengths.map_err(|_| PyValueError::new_err("a passage is longer than its length keeps"))?;
    if ids.len() != lengths.len() {
        return Err(PyValueError::new_err("each chunk has a length"));
    }
    let found = py
        .detach(|| lengths_written(&ids, &lengths, shift))
        .map_err(PyValueError::new_err)?;
    PyList::new(
        py,
        found.iter().map(|(block, packed)| (*block, PyBytes::new(py, packed))),
    )
}
