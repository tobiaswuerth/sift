use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::{PyBytes, PyList, PyString};

use crate::lists::{Coded, numbers_of};
use crate::segment::{Held, Kept, Row, Shape, Span, Written};
use crate::words::Table;

const USED: &str = "a segment is written once, and takes no more after";

type Field = (PyBackedStr, PyBackedStr);

struct Batch {
    places: Vec<i64>,
    of: Vec<usize>,
    fields: Vec<Field>,
    texts: Vec<PyBackedStr>,
}

#[pyclass(frozen, module = "sift._native")]
pub struct Words {
    table: Arc<Table>,
}

#[pyclass(module = "sift._native")]
pub struct Building {
    table: Arc<Table>,
    most_field: usize,
    held: Option<Held>,
}

fn cut(field: &str, most: usize) -> (&str, bool) {
    match field.char_indices().nth(most) {
        Some((end, _)) => (&field[..end], true),
        None => (field, false),
    }
}

fn field(held: &mut Held, table: &Table, text: &str, most: usize) -> Result<Vec<u32>, String> {
    let (whole, shortened) = cut(text, most);
    let mut found = Vec::new();
    held.coded(table, whole, &mut found)?;
    if shortened {
        found.pop();
    }
    Ok(found)
}

fn taken(held: &mut Held, table: &Table, most_field: usize, batch: &Batch) -> Result<usize, String> {
    let mut named: Vec<Option<(Vec<u32>, Vec<u32>)>> = batch.fields.iter().map(|_| None).collect();
    let mut scratch = Vec::new();
    for (at, (&place, text)) in batch.places.iter().zip(&batch.texts).enumerate() {
        if held.full() {
            return Ok(at);
        }
        let which = batch.of.get(at).copied().filter(|&which| which < named.len());
        let which = which.ok_or_else(|| "a passage names a document that was not given".to_owned())?;
        if named[which].is_none() {
            let (title, meta) = &batch.fields[which];
            named[which] = Some((
                field(held, table, title, most_field)?,
                field(held, table, meta, most_field)?,
            ));
        }
        let Some((title, meta)) = &named[which] else {
            return Err("a document's fields were not read".to_owned());
        };
        held.add(
            table,
            &Span {
                place,
                title,
                text,
                meta,
            },
            &mut scratch,
        )?;
    }
    Ok(batch.places.len())
}

fn pair<'py>(py: Python<'py>, coded: &Coded) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
    (PyBytes::new(py, &coded.0), PyBytes::new(py, &coded.1))
}

fn shown<'py>(py: Python<'py>, row: &Row) -> PyResult<Bound<'py, PyAny>> {
    let term = PyString::new(py, &row.term);
    let none = py.None().into_bound(py);
    let listed = match &row.held {
        Kept::Short(coded) => {
            let (postings, positions) = pair(py, coded);
            (
                term,
                row.chunks,
                postings.into_any(),
                positions.into_any(),
                none,
                PyList::empty(py),
            )
        }
        Kept::Long { ranges, blocks } => {
            let blocks = PyList::new(py, blocks.iter().map(|coded| pair(py, coded)))?;
            (
                term,
                row.chunks,
                none.clone(),
                none,
                PyBytes::new(py, ranges).into_any(),
                blocks,
            )
        }
    };
    Ok(listed.into_pyobject(py)?.into_any())
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
            most_field,
            held: Some(Held::new(most)),
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
        let batch = Batch {
            places: numbers_of(places)?,
            of: of.extract()?,
            fields: fields.extract()?,
            texts: texts.extract()?,
        };
        if batch.places.len() != batch.texts.len() {
            return Err(PyValueError::new_err("each passage has a place and a text"));
        }
        let (table, most_field) = (Arc::clone(&self.table), self.most_field);
        let held = self.held.as_mut().ok_or_else(|| PyValueError::new_err(USED))?;
        places
            .py()
            .detach(|| taken(held, &table, most_field, &batch))
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn chunks(&self) -> usize {
        self.held.as_ref().map_or(0, Held::chunks)
    }

    #[getter]
    fn tokens(&self) -> u64 {
        self.held.as_ref().map_or(0, Held::tokens)
    }

    fn written<'py>(
        &mut self,
        py: Python<'py>,
        average: f64,
        range_shift: u32,
        block: usize,
        long: usize,
        lengths_shift: u32,
    ) -> PyResult<(i64, i64, Bound<'py, PyList>, Bound<'py, PyList>)> {
        let held = self.held.take().ok_or_else(|| PyValueError::new_err(USED))?;
        let shape = Shape {
            range_shift,
            block,
            long,
            lengths_shift,
        };
        let made: Written = py
            .detach(|| held.written(average, shape))
            .map_err(PyValueError::new_err)?;
        let rows = PyList::empty(py);
        for row in &made.rows {
            rows.append(shown(py, row)?)?;
        }
        let lengths = PyList::new(
            py,
            made.lengths
                .iter()
                .map(|(block, packed)| (*block, PyBytes::new(py, packed))),
        )?;
        Ok((made.first, made.last, rows, lengths))
    }
}
