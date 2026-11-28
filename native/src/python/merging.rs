use std::sync::Arc;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};

use crate::merge::{Chunks, Fed, Limits, Merger};
use crate::python::{Listed, lengths_of, listed, numbers_of, shaped};

const UNKNOWN: &str = "a merge was fed rows of a source it has not";

type Given = (PyBackedStr, usize, PyBackedBytes, PyBackedBytes);
type Stepped<'py> = (Listed<'py>, Vec<usize>, bool);

#[pyclass(name = "Merger", module = "sift._native")]
pub struct Merging {
    held: Merger,
}

#[pymethods]
impl Merging {
    #[new]
    fn new(
        live: &Bound<'_, PyAny>,
        lengths: &Bound<'_, PyAny>,
        dead: &Bound<'_, PyAny>,
        average: f64,
        shape: (u32, usize, usize, u32),
        limits: (usize, usize, usize),
        sources: usize,
    ) -> PyResult<Self> {
        let chunks = Chunks {
            live: numbers_of(live)?,
            lengths: lengths_of(lengths)?,
            dead: numbers_of(dead)?,
            average,
        };
        let limits = Limits {
            postings: limits.0,
            bytes: limits.1,
            tokens: limits.2,
        };
        let held = Merger::new(chunks, shaped(shape)?, &limits, sources);
        Ok(Self {
            held: held.map_err(PyValueError::new_err)?,
        })
    }

    fn feed(&mut self, source: usize, rows: &Bound<'_, PyAny>) -> PyResult<bool> {
        let given: Vec<Given> = rows.extract()?;
        let rows: Vec<Fed> = given
            .into_iter()
            .map(|(term, chunks, postings, positions)| Fed {
                term: Arc::from(&*term),
                chunks,
                postings: postings.to_vec(),
                positions: positions.to_vec(),
            })
            .collect();
        let more = if rows.is_empty() {
            self.held.ended(source)
        } else {
            self.held.fed(source, rows).map_err(PyValueError::new_err)?
        };
        more.ok_or_else(|| PyIndexError::new_err(UNKNOWN))
    }

    fn feed_packs(&mut self, source: usize, rows: &Bound<'_, PyAny>) -> PyResult<bool> {
        let packs: Vec<(PyBackedBytes, PyBackedBytes)> = rows.extract()?;
        let more = if packs.is_empty() {
            self.held.ended(source)
        } else {
            self.held.fed_packs(source, &packs).map_err(PyValueError::new_err)?
        };
        more.ok_or_else(|| PyIndexError::new_err(UNKNOWN))
    }

    fn step<'py>(&mut self, py: Python<'py>, segment: i64) -> PyResult<Stepped<'py>> {
        let step = py.detach(|| self.held.stepped()).map_err(PyValueError::new_err)?;
        Ok((listed(py, segment, &step.filed)?, step.wanted, step.done))
    }
}
