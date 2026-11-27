mod building;
mod lists;
mod merging;

use mimalloc::MiMalloc;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};

use crate::postings::{Filed, Shape};

type Listed<'py> = (Bound<'py, PyList>, Bound<'py, PyList>, Bound<'py, PyList>);

#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;

fn numbers_of(held: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    PyBuffer::<i64>::get(held)?.to_vec(held.py())
}

fn lengths_of(held: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    let lengths: Result<Vec<u32>, _> = numbers_of(held)?.into_iter().map(u32::try_from).collect();
    lengths.map_err(|_| PyValueError::new_err("a passage is longer than its length keeps"))
}

fn shaped(shape: (u32, usize, usize, u32)) -> PyResult<Shape> {
    Shape::new(shape.0, shape.1, shape.2, shape.3).map_err(PyValueError::new_err)
}

fn listed<'py>(py: Python<'py>, segment: i64, filed: &Filed) -> PyResult<Listed<'py>> {
    let packs = filed.packs.iter().map(|pack| {
        let (lists, places) = (PyBytes::new(py, &pack.lists), PyBytes::new(py, &pack.places));
        (segment, PyString::new(py, &pack.first), lists, places)
    });
    let ranged = filed.ranged.iter().map(|held| {
        let ranges = PyBytes::new(py, &held.ranges);
        (segment, PyString::new(py, &held.term), held.chunks, ranges)
    });
    let blocks = filed.blocks.iter().map(|(term, number, coded)| {
        let (postings, positions) = (PyBytes::new(py, &coded.0), PyBytes::new(py, &coded.1));
        (segment, PyString::new(py, term), *number, postings, positions)
    });
    Ok((
        PyList::new(py, packs)?,
        PyList::new(py, ranged)?,
        PyList::new(py, blocks)?,
    ))
}

#[pymodule(name = "_native")]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lists::blocked_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::positions_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::with_lengths, module)?)?;
    module.add_function(wrap_pyfunction!(lists::unpacked, module)?)?;
    module.add_function(wrap_pyfunction!(building::gathered, module)?)?;
    module.add_class::<building::Words>()?;
    module.add_class::<building::Building>()?;
    module.add_class::<merging::Merging>()?;
    Ok(())
}
