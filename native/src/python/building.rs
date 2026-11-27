use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyList};

use crate::python::{Listed, listed, numbers_of, shaped};
use crate::segment::{Held, Passages};
use crate::words::Table;

const USED: &str = "a segment is written once, and takes no more after";

type Handed<'py> = (Listed<'py>, Bound<'py, PyList>);
type Gathered = (Option<i64>, Option<i64>, usize, u64);

#[pyclass(frozen, module = "sift._native")]
pub struct Words {
    table: Arc<Table>,
}

#[pyclass(module = "sift._native")]
pub struct Building {
    table: Arc<Table>,
    held: Option<Held>,
}

#[pyfunction]
pub fn gathered(building: &Bound<'_, Building>) -> PyResult<Gathered> {
    let building = building.try_borrow()?;
    let held = building.held.as_ref();
    let ends = (held.and_then(Held::first), held.and_then(Held::last));
    Ok((
        ends.0,
        ends.1,
        held.map_or(0, Held::chunks),
        held.map_or(0, Held::tokens),
    ))
}

#[pymethods]
impl Words {
    #[new]
    fn new(stored: &Bound<'_, PyAny>) -> PyResult<Self> {
        let stored: PyBackedBytes = stored.extract()?;
        let table = Table::new(&stored).map_err(PyValueError::new_err)?;
        Ok(Self { table: Arc::new(table) })
    }

    fn building(&self, most: usize, most_field: usize) -> Building {
        Building {
            table: Arc::clone(&self.table),
            held: Some(Held::new(most, most_field)),
        }
    }
}

#[pymethods]
impl Building {
    fn add(
        &mut self,
        places: &Bound<'_, PyAny>,
        of: &Bound<'_, PyAny>,
        fields: &Bound<'_, PyAny>,
        texts: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let (places, py) = (numbers_of(places)?, places.py());
        let (of, fields): (Vec<usize>, Vec<(PyBackedStr, PyBackedStr)>) = (of.extract()?, fields.extract()?);
        let texts: Vec<PyBackedStr> = texts.extract()?;
        let fields: Vec<(&str, &str)> = fields.iter().map(|(title, meta)| (&**title, &**meta)).collect();
        let texts: Vec<&str> = texts.iter().map(|text| &**text).collect();
        let passages = Passages {
            places: &places,
            of: &of,
            fields: &fields,
            texts: &texts,
        };
        let table = Arc::clone(&self.table);
        let held = self.held.as_mut().ok_or_else(|| PyValueError::new_err(USED))?;
        py.detach(|| held.taken(&table, &passages))
            .map_err(PyValueError::new_err)
    }

    fn written<'py>(
        &mut self,
        py: Python<'py>,
        segment: i64,
        average: f64,
        shape: (u32, usize, usize, u32),
    ) -> PyResult<Handed<'py>> {
        let shape = shaped(shape)?;
        let held = self.held.take().ok_or_else(|| PyValueError::new_err(USED))?;
        let made = py
            .detach(|| held.written(average, shape))
            .map_err(PyValueError::new_err)?;
        let lengths = made.lengths.iter();
        let lengths = lengths.map(|(block, packed)| (segment, *block, PyBytes::new(py, packed)));
        Ok((listed(py, segment, &made.filed)?, PyList::new(py, lengths)?))
    }
}
