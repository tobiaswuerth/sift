use std::mem;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::hamming::{gathered, scanned};
use crate::python::numbers_of;

const OVERFILLED: &str = "codes were filled past the rows they were made for, or with part of a row";
const UNFILLED: &str = "codes were finished before all the rows they were made for were filled";
const TOO_MANY: &str = "codes were made for more rows than memory can hold";

#[pyclass(module = "sift._native")]
pub struct Filling {
    held: Vec<u8>,
    rows: usize,
    width: usize,
}

#[pymethods]
impl Filling {
    #[new]
    fn new(rows: usize, width: usize) -> PyResult<Self> {
        let size = rows.checked_mul(width).filter(|&size| isize::try_from(size).is_ok());
        let size = size.ok_or_else(|| PyValueError::new_err(TOO_MANY))?;
        Ok(Self {
            held: Vec::with_capacity(size),
            rows,
            width,
        })
    }

    fn extend(&mut self, more: &Bound<'_, PyAny>) -> PyResult<()> {
        let more: PyBackedBytes = more.extract()?;
        let room = self.rows * self.width - self.held.len();
        if more.len() > room || !more.len().is_multiple_of(self.width.max(1)) {
            return Err(PyValueError::new_err(OVERFILLED));
        }
        self.held.extend_from_slice(&more);
        Ok(())
    }
}

#[pyclass(frozen, module = "sift._native", name = "Codes")]
pub struct HeldCodes {
    held: Vec<u8>,
    width: usize,
}

#[pymethods]
impl HeldCodes {
    #[new]
    fn new(filled: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut filling: PyRefMut<'_, Filling> = filled.extract()?;
        if filling.held.len() != filling.rows * filling.width {
            return Err(PyValueError::new_err(UNFILLED));
        }
        Ok(Self {
            held: mem::take(&mut filling.held),
            width: filling.width,
        })
    }

    fn picked<'py>(&self, at: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBytes>> {
        let found = gathered(&self.held, self.width, &numbers_of(at)?);
        Ok(PyBytes::new(at.py(), &found.map_err(PyValueError::new_err)?))
    }
}

#[pyfunction]
pub fn scanned_codes<'py>(
    py: Python<'py>,
    held: &Bound<'py, PyAny>,
    query: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, Vec<u64>)> {
    let held: Vec<Bound<'py, HeldCodes>> = held.extract()?;
    let codes: Vec<(&[u8], usize)> = held
        .iter()
        .map(|each| (each.get().held.as_slice(), each.get().width))
        .collect();
    let rows: usize = codes.iter().map(|(each, _)| each.len() / query.len().max(1)).sum();
    let mut counted = Ok(Vec::new());
    let fill = |out: &mut [u8]| {
        counted = py.detach(|| scanned(&codes, query, out));
        Ok(())
    };
    let found = PyBytes::new_with(py, 2 * rows, fill)?;
    Ok((found, counted.map_err(PyValueError::new_err)?))
}
