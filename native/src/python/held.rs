use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::lists::Block;
use crate::numbers::little_endian;
use crate::postings::owned_blocks;
use crate::python::shaped;
use crate::ranges::{Bounds, Ranges, ranges_all};
use crate::reading::LengthsTable;

const UNKEPT: &str = "a value kept of the keyword index is neither a block of postings nor a table of lengths";

type Arrays<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<usize>);
type Spans<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
);

#[pyclass(frozen, module = "sift._native", name = "Ranges")]
pub struct HeldRanges(pub Arc<Ranges>);

#[pyclass(frozen, module = "sift._native", name = "Block")]
pub struct HeldBlock(pub Arc<Block>);

#[pyclass(frozen, module = "sift._native", name = "LengthsTable")]
pub struct HeldTable(pub Arc<LengthsTable>);

fn stored(py: Python<'_>, numbers: impl Iterator<Item = usize>) -> Bound<'_, PyBytes> {
    let numbers: Vec<i64> = numbers
        .map(|number| i64::try_from(number).unwrap_or(i64::MAX))
        .collect();
    PyBytes::new(py, &little_endian(&numbers))
}

#[pyfunction]
pub fn ranges_kept(py: Python<'_>, each: &Bound<'_, PyAny>, bounds: &Bound<'_, PyAny>) -> PyResult<Vec<HeldRanges>> {
    let rows = each.extract::<Vec<PyBackedBytes>>()?;
    let told = bounds.extract::<Vec<(i64, i64, usize)>>()?;
    let bounds: Vec<Bounds> = told
        .into_iter()
        .map(|(low, high, chunks)| Bounds { low, high, chunks })
        .collect();
    let found = py
        .detach(|| ranges_all(&rows, &bounds))
        .map_err(PyValueError::new_err)?;
    Ok(found.into_iter().map(|ranges| HeldRanges(Arc::new(ranges))).collect())
}

#[pyfunction]
pub fn ranges_arrays<'py>(held: &Bound<'py, PyAny>) -> PyResult<Spans<'py>> {
    let (py, held) = (held.py(), held.cast::<HeldRanges>()?.get());
    let ranges = &held.0;
    let ids = PyBytes::new(py, &little_endian(ranges.ids()));
    let starts = stored(py, (0..ranges.ids().len()).map(|at| ranges.start(at)));
    let ends = stored(py, ranges.ends().iter().copied());
    Ok((ids, starts, ends, PyBytes::new(py, ranges.best())))
}

#[pyfunction]
pub fn blocked_within(
    py: Python<'_>,
    each: &Bound<'_, PyAny>,
    numbers: &Bound<'_, PyAny>,
    ranges: &Bound<'_, PyAny>,
    shape: (u32, usize, usize, u32),
) -> PyResult<Vec<HeldBlock>> {
    let (rows, numbers) = (each.extract::<Vec<PyBackedBytes>>()?, numbers.extract::<Vec<usize>>()?);
    let (ranges, shape) = (Arc::clone(&ranges.cast::<HeldRanges>()?.get().0), shaped(shape)?);
    let found = py.detach(|| owned_blocks(&rows, &numbers, &ranges, shape));
    let found = found.map_err(PyValueError::new_err)?;
    Ok(found.into_iter().map(|block| HeldBlock(Arc::new(block))).collect())
}

#[pyfunction]
pub fn postings_of<'py>(py: Python<'py>, blocks: &Bound<'py, PyAny>) -> PyResult<Arrays<'py>> {
    let held = blocks.extract::<Vec<Bound<'py, HeldBlock>>>()?;
    let (mut ids, mut counts, mut bounds) = (Vec::new(), Vec::new(), vec![0]);
    for each in &held {
        let block = &each.get().0;
        ids.extend_from_slice(&block.ids);
        counts.extend_from_slice(&block.counts);
        bounds.push(ids.len());
    }
    let (ids, counts) = (
        PyBytes::new(py, &little_endian(&ids)),
        PyBytes::new(py, &little_endian(&counts)),
    );
    Ok((ids, counts, bounds))
}

#[pyfunction]
pub fn kept_bytes(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(block) = value.cast::<HeldBlock>() {
        let held = &block.get().0;
        return Ok(size_of_val(held.ids.as_slice()) + size_of_val(held.counts.as_slice()));
    }
    let table = value.cast::<HeldTable>().map_err(|_| PyTypeError::new_err(UNKEPT))?;
    Ok(table.get().0.held_bytes())
}
