use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::clusters::{majority, nearest};
use crate::numbers::little_endian;
use crate::python::numbers_of;

fn words_of(held: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    PyBuffer::<u64>::get(held)?.to_vec(held.py())
}

#[pyfunction]
pub fn nearest_centroids<'py>(
    held: &Bound<'py, PyAny>,
    centroids: &Bound<'py, PyAny>,
    width: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let (py, held, centroids) = (held.py(), words_of(held)?, words_of(centroids)?);
    let found = py.detach(|| nearest(&held, &centroids, width));
    Ok(PyBytes::new(py, &little_endian(&found.map_err(PyValueError::new_err)?)))
}

#[pyfunction]
pub fn majority_centroids<'py>(
    held: &Bound<'py, PyAny>,
    assigned: &Bound<'py, PyAny>,
    centroids: &Bound<'py, PyAny>,
    width: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let (py, held, assigned, centroids) = (held.py(), words_of(held)?, numbers_of(assigned)?, words_of(centroids)?);
    let found = py.detach(|| majority(&held, &assigned, &centroids, width));
    let made = found.map_err(PyValueError::new_err)?;
    let bytes: Vec<u8> = made.iter().flat_map(|word| word.to_le_bytes()).collect();
    Ok(PyBytes::new(py, &bytes))
}
