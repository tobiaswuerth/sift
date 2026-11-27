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

pub type Listed<'py> = (Bound<'py, PyList>, Bound<'py, PyList>);

pub type Piece = (Arc<str>, usize, Coded);

pub fn listed<'py>(py: Python<'py>, segment: i64, rows: &[Row], pieces: &[Piece]) -> PyResult<Listed<'py>> {
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

    #[getter]
    fn first(&self) -> Option<i64> {
        self.held.as_ref().and_then(Held::first)
    }

    #[getter]
    fn last(&self) -> Option<i64> {
        self.held.as_ref().and_then(Held::last)
    }

    fn written<'py>(
        &mut self,
        py: Python<'py>,
        segment: i64,
        average: f64,
        shape: (u32, usize, usize, u32),
    ) -> PyResult<(Listed<'py>, Bound<'py, PyList>, Bound<'py, PyList>)> {
        let held = self.held.take().ok_or_else(|| PyValueError::new_err(USED))?;
        let shape = Shape {
            range_shift: shape.0,
            block: shape.1,
            long: shape.2,
            lengths_shift: shape.3,
        };
        let made: Written = py
            .detach(|| held.written(average, shape))
            .map_err(PyValueError::new_err)?;
        let lengths = PyList::new(
            py,
            made.lengths
                .iter()
                .map(|(block, packed)| (segment, *block, PyBytes::new(py, packed))),
        )?;
        let terms = PyList::new(
            py,
            made.rows.iter().map(|row| (PyString::new(py, &row.term), row.chunks)),
        )?;
        Ok((listed(py, segment, &made.rows, &[])?, lengths, terms))
    }
}
