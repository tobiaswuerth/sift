use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyList};

use crate::lists::{blocks, places};
use crate::numbers::little_endian;
use crate::python::{lengths_of, numbers_of};
use crate::segment::lengths_written;

type Blocked<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>);

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
