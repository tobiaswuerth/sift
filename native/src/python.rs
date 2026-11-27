mod building;
mod lists;
mod merging;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyString};

use crate::postings::{Kept, Piece, Row, Shape};

type Listed<'py> = (Bound<'py, PyList>, Bound<'py, PyList>);

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

fn listed<'py>(py: Python<'py>, segment: i64, rows: &[Row], pieces: &[Piece]) -> PyResult<Listed<'py>> {
    let (lists, blocks) = (PyList::empty(py), PyList::empty(py));
    let none = py.None().into_bound(py);
    for row in rows {
        let term = PyString::new(py, &row.term);
        match &row.held {
            Kept::Short(coded) => {
                let (postings, positions) = (PyBytes::new(py, &coded.0), PyBytes::new(py, &coded.1));
                lists.append((segment, term, row.chunks, postings, positions, &none))?;
            }
            Kept::Long { ranges, blocks: held } => {
                lists.append((segment, &term, row.chunks, &none, &none, PyBytes::new(py, ranges)))?;
                for (number, coded) in held.iter().enumerate() {
                    let (postings, positions) = (PyBytes::new(py, &coded.0), PyBytes::new(py, &coded.1));
                    blocks.append((segment, &term, number, postings, positions))?;
                }
            }
        }
    }
    for (term, number, coded) in pieces {
        let (postings, positions) = (PyBytes::new(py, &coded.0), PyBytes::new(py, &coded.1));
        blocks.append((segment, PyString::new(py, term), *number, postings, positions))?;
    }
    Ok((lists, blocks))
}

#[pymodule(name = "_native")]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lists::blocked_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::positions_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::with_lengths, module)?)?;
    module.add_function(wrap_pyfunction!(building::gathered, module)?)?;
    module.add_class::<building::Words>()?;
    module.add_class::<building::Building>()?;
    module.add_class::<merging::Merging>()?;
    Ok(())
}
