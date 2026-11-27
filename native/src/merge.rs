use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
use pyo3::types::PyList;
use rayon::prelude::*;

use crate::building::shown;
use crate::lists::{block, bytes_of, gaps, numbers_of, placed};
use crate::segment::{Postings, Row, Shape, kept};

const TOGETHER: usize = 16;
const ASTRAY: &str = "the keyword index holds chunks the store does not";

type Spans = Vec<(usize, usize)>;

enum Merged {
    Gone,
    Large,
    Kept(Row),
}

struct Rows {
    terms: Vec<PyBackedStr>,
    postings: Vec<PyBackedBytes>,
    positions: Vec<PyBackedBytes>,
}

#[pyclass(frozen, module = "sift._native")]
pub struct Merging {
    live: Vec<i64>,
    lengths: Vec<u32>,
    dead: Vec<i64>,
    average: f64,
    shape: Shape,
}

fn bounds(terms: &[PyBackedStr]) -> Vec<usize> {
    let mut found: Vec<usize> = (0..terms.len())
        .filter(|&at| at == 0 || terms[at][..] != terms[at - 1][..])
        .collect();
    found.push(terms.len());
    found
}

impl Merging {
    fn term(&self, rows: &Rows, start: usize, end: usize, most: usize) -> Result<Merged, String> {
        let blocks = rows.postings[start..end]
            .iter()
            .map(|coded| block(coded))
            .collect::<Result<Vec<_>, _>>()?;
        let tokens: i64 = blocks.iter().flat_map(|found| &found.counts).sum();
        if usize::try_from(tokens).map_or(true, |tokens| tokens > most) {
            return Ok(Merged::Large);
        }
        let mut held = Postings::default();
        for (found, coded) in blocks.iter().zip(&rows.positions[start..end]) {
            let mut places = gaps(coded)?;
            placed(&mut places, &found.counts)?;
            let mut from = 0usize;
            for (&id, &count) in found.ids.iter().zip(&found.counts) {
                let to = from + usize::try_from(count).unwrap_or(0);
                if self.dead.binary_search(&id).is_err() {
                    let which = self.live.binary_search(&id).map_err(|_| ASTRAY.to_owned())?;
                    held.push(id, self.lengths[which], places[from..to].iter().copied())?;
                }
                from = to;
            }
        }
        if held.chunks() == 0 {
            return Ok(Merged::Gone);
        }
        kept(&rows.terms[start], &held, self.average, self.shape).map(Merged::Kept)
    }

    fn merged(&self, rows: &Rows, most: usize) -> Result<(Vec<Row>, Spans), String> {
        let cuts = bounds(&rows.terms);
        let found: Result<Vec<Merged>, String> = cuts
            .par_windows(2)
            .with_min_len(TOGETHER)
            .map(|pair| self.term(rows, pair[0], pair[1], most))
            .collect();
        let (mut made, mut large) = (Vec::new(), Vec::new());
        for (each, pair) in found?.into_iter().zip(cuts.windows(2)) {
            match each {
                Merged::Kept(row) => made.push(row),
                Merged::Large => large.push((pair[0], pair[1])),
                Merged::Gone => {}
            }
        }
        Ok((made, large))
    }
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
    ) -> PyResult<Self> {
        let lengths: Result<Vec<u32>, _> = numbers_of(lengths)?.into_iter().map(u32::try_from).collect();
        let lengths = lengths.map_err(|_| PyValueError::new_err("a passage is longer than its length keeps"))?;
        let live = numbers_of(live)?;
        if live.len() != lengths.len() {
            return Err(PyValueError::new_err("each chunk of a merge has a length"));
        }
        let shape = Shape {
            range_shift: shape.0,
            block: shape.1,
            long: shape.2,
            lengths_shift: shape.3,
        };
        Ok(Self {
            live,
            lengths,
            dead: numbers_of(dead)?,
            average,
            shape,
        })
    }

    fn lists<'py>(
        &self,
        terms: &Bound<'py, PyAny>,
        postings: &Bound<'py, PyAny>,
        positions: &Bound<'py, PyAny>,
        most: usize,
    ) -> PyResult<(Bound<'py, PyList>, Spans)> {
        let py = terms.py();
        let rows = Rows {
            terms: terms.extract()?,
            postings: bytes_of(postings)?,
            positions: bytes_of(positions)?,
        };
        if rows.terms.len() != rows.postings.len() || rows.terms.len() != rows.positions.len() {
            return Err(PyValueError::new_err(
                "each row of a merge has a term, its postings and their positions",
            ));
        }
        let (made, large) = py.detach(|| self.merged(&rows, most)).map_err(PyValueError::new_err)?;
        let shown_rows = PyList::empty(py);
        for row in &made {
            shown_rows.append(shown(py, row)?)?;
        }
        Ok((shown_rows, large))
    }
}
