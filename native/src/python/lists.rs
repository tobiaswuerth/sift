use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyList, PyString};

use crate::lists::{Sought, blocks, occurring};
use crate::numbers::little_endian;
use crate::packs::{self, Entry, entries};
use crate::phrases::{Word, phrased};
use crate::python::{lengths_of, numbers_of};
use crate::segment::{lengths_read, lengths_written};
use crate::unions::summed;

type Blocked<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>);
type Counted<'py> = Vec<(Bound<'py, PyString>, usize)>;
type Pair<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>);
type Worded<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);
type Occurrences<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>);
type Unpacked<'py> = (
    Bound<'py, PyString>,
    usize,
    Option<Bound<'py, PyBytes>>,
    Option<Bound<'py, PyBytes>>,
);

#[pyfunction]
pub fn blocked_all<'py>(each: &Bound<'py, PyAny>) -> PyResult<Blocked<'py>> {
    let (py, rows) = (each.py(), each.extract::<Vec<PyBackedBytes>>()?);
    let found = py.detach(|| blocks(&rows)).map_err(PyValueError::new_err)?;
    let (ids, counts) = (little_endian(&found.ids), little_endian(&found.counts));
    Ok((PyBytes::new(py, &ids), PyBytes::new(py, &counts), found.bounds))
}

#[pyfunction]
pub fn occurrences<'py>(
    each: &Bound<'py, PyAny>,
    bounds: &Bound<'py, PyAny>,
    ids: &Bound<'py, PyAny>,
    counts: &Bound<'py, PyAny>,
    wanted: &Bound<'py, PyAny>,
) -> PyResult<Occurrences<'py>> {
    let (py, rows, bounds) = (
        each.py(),
        each.extract::<Vec<PyBackedBytes>>()?,
        bounds.extract::<Vec<usize>>()?,
    );
    let (ids, counts, wanted) = (numbers_of(ids)?, numbers_of(counts)?, numbers_of(wanted)?);
    let sought = Sought {
        bounds: &bounds,
        ids: &ids,
        counts: &counts,
        wanted: &wanted,
    };
    let found = py.detach(|| occurring(&rows, &sought)).map_err(PyValueError::new_err)?;
    let stored = |numbers: &[i64]| PyBytes::new(py, &little_endian(numbers));
    Ok((stored(&found.ids), stored(&found.counts), stored(&found.positions)))
}

#[pyfunction]
pub fn summed_by_id<'py>(ids: &Bound<'py, PyAny>, counts: &Bound<'py, PyAny>) -> PyResult<Pair<'py>> {
    let (py, ids, counts) = (ids.py(), numbers_of(ids)?, numbers_of(counts)?);
    let found = py.detach(|| summed(&ids, &counts)).map_err(PyValueError::new_err)?;
    Ok((
        PyBytes::new(py, &little_endian(&found.0)),
        PyBytes::new(py, &little_endian(&found.1)),
    ))
}

#[pyfunction]
pub fn phrase_counts<'py>(candidates: &Bound<'py, PyAny>, words: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let (py, candidates) = (candidates.py(), numbers_of(candidates)?);
    let mut held = Vec::new();
    for (ids, counts, positions) in words.extract::<Vec<Worded<'py>>>()? {
        held.push((numbers_of(&ids)?, numbers_of(&counts)?, numbers_of(&positions)?));
    }
    let each = held
        .iter()
        .map(|(ids, counts, positions)| Word { ids, counts, positions });
    let words: Vec<Word> = each.collect();
    let found = py.detach(|| phrased(&candidates, &words));
    Ok(PyBytes::new(py, &little_endian(&found.map_err(PyValueError::new_err)?)))
}

fn told<'py>(py: Python<'py>, each: &Entry, placed: bool) -> Unpacked<'py> {
    let short = (!each.postings.is_empty()).then(|| PyBytes::new(py, each.postings));
    let positions = short.as_ref().filter(|_| placed);
    let positions = positions.map(|_| PyBytes::new(py, each.positions));
    (PyString::new(py, each.term), each.chunks, short, positions)
}

#[pyfunction]
pub fn unpacked<'py>(
    first: &str,
    lists: &Bound<'py, PyAny>,
    places: &Bound<'py, PyAny>,
    terms: &Bound<'py, PyAny>,
) -> PyResult<Vec<Unpacked<'py>>> {
    let (py, lists, terms) = (
        lists.py(),
        lists.extract::<PyBackedBytes>()?,
        terms.extract::<Vec<PyBackedStr>>()?,
    );
    let places: Option<PyBackedBytes> = places.extract()?;
    let found = entries(&lists, places.as_deref(), Some(first)).map_err(PyValueError::new_err)?;
    let wanted = found
        .iter()
        .filter(|each| terms.iter().any(|term| &**term == each.term));
    Ok(wanted.map(|each| told(py, each, places.is_some())).collect())
}

#[pyfunction]
#[pyo3(signature = (first, lists, low, high))]
pub fn within<'py>(
    first: &str,
    lists: &Bound<'py, PyAny>,
    low: &str,
    high: Option<&str>,
) -> PyResult<Vec<Unpacked<'py>>> {
    let (py, lists) = (lists.py(), lists.extract::<PyBackedBytes>()?);
    let found = entries(&lists, None, Some(first)).map_err(PyValueError::new_err)?;
    let wanted = found
        .iter()
        .filter(|each| each.term >= low && high.is_none_or(|high| each.term < high));
    Ok(wanted.map(|each| told(py, each, false)).collect())
}

#[pyfunction]
#[pyo3(signature = (rows, low, high))]
pub fn counted<'py>(rows: &Bound<'py, PyAny>, low: &str, high: Option<&str>) -> PyResult<Counted<'py>> {
    let (py, rows) = (rows.py(), rows.extract::<Vec<(PyBackedStr, PyBackedBytes)>>()?);
    let mut found = Vec::new();
    for (first, lists) in &rows {
        let held = packs::counted(lists, Some(first), low, high).map_err(PyValueError::new_err)?;
        found.extend(held.into_iter().map(|(term, chunks)| (PyString::new(py, term), chunks)));
    }
    Ok(found)
}

#[pyfunction]
pub fn lengths_all<'py>(each: &Bound<'py, PyAny>, shift: u32) -> PyResult<Bound<'py, PyBytes>> {
    let (py, rows) = (each.py(), each.extract::<Vec<PyBackedBytes>>()?);
    let found = py.detach(|| lengths_read(&rows, shift));
    Ok(PyBytes::new(py, &found.map_err(PyValueError::new_err)?))
}

#[pyfunction]
pub fn with_lengths<'py>(
    ids: &Bound<'py, PyAny>,
    lengths: &Bound<'py, PyAny>,
    shift: u32,
) -> PyResult<Bound<'py, PyList>> {
    let (py, ids, lengths) = (ids.py(), numbers_of(ids)?, lengths_of(lengths)?);
    let found = py.detach(|| lengths_written(&ids, &lengths, shift));
    let found = found.map_err(PyValueError::new_err)?;
    PyList::new(
        py,
        found.iter().map(|(block, packed)| (*block, PyBytes::new(py, packed))),
    )
}
