use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyList, PyString};

use crate::lists::{blocks, places};
use crate::numbers::little_endian;
use crate::packs::{self, entries};
use crate::python::{lengths_of, numbers_of};
use crate::segment::lengths_written;

type Blocked<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>);
type Counted<'py> = Vec<(Bound<'py, PyString>, usize)>;
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
pub fn positions_all<'py>(each: &Bound<'py, PyAny>, counts: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
    let (py, rows, counts) = (each.py(), each.extract::<Vec<PyBackedBytes>>()?, numbers_of(counts)?);
    let found = py.detach(|| places(&rows, &counts).map(|found| little_endian(&found)));
    Ok(PyBytes::new(py, &found.map_err(PyValueError::new_err)?))
}

#[pyfunction]
pub fn unpacked<'py>(
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
    let found = entries(&lists, places.as_deref()).map_err(PyValueError::new_err)?;
    let wanted = found
        .iter()
        .filter(|each| terms.iter().any(|term| &**term == each.term));
    let told = wanted.map(|each| {
        let short = (!each.postings.is_empty()).then(|| PyBytes::new(py, each.postings));
        let positions = short.as_ref().and(places.as_ref());
        let positions = positions.map(|_| PyBytes::new(py, each.positions));
        (PyString::new(py, each.term), each.chunks, short, positions)
    });
    Ok(told.collect())
}

#[pyfunction]
#[pyo3(signature = (rows, low, high))]
pub fn counted<'py>(rows: &Bound<'py, PyAny>, low: &str, high: Option<&str>) -> PyResult<Counted<'py>> {
    let (py, rows) = (rows.py(), rows.extract::<Vec<PyBackedBytes>>()?);
    let mut found = Vec::new();
    for lists in &rows {
        let held = packs::counted(lists, low, high).map_err(PyValueError::new_err)?;
        found.extend(held.into_iter().map(|(term, chunks)| (PyString::new(py, term), chunks)));
    }
    Ok(found)
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
