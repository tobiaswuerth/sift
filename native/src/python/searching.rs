use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::PyBytes;

use crate::lists::Block;
use crate::numbers::little_endian;
use crate::python::held::{HeldBlock, HeldRanges, HeldTable};
use crate::python::{numbers_of, shaped};
use crate::reading::{Failure, LengthBlock, LengthsTable, Part, Reached, Reads, Segment, Store, length_blocks};
use crate::scoring::{Expression, Node};
use crate::search::{Asked, ExactList, Plan, searched as scored};

const KINDLESS: &str = "a query is told with a part of no kind a search knows";

type SegmentTold<'py> = (i64, i64, Bound<'py, PyAny>, Option<Bound<'py, PyAny>>);
type PartTold<'py> = (usize, Bound<'py, PyAny>, f64);
type ListedTold<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Option<usize>);
type LeafTold<'py> = (Vec<ListedTold<'py>>, Vec<usize>, f64, f64);
type NodeTold = (u8, usize, Vec<usize>);
type Numbers = (f64, (u32, usize, usize, u32), usize, usize);
type PlanTold<'py> = (
    Vec<SegmentTold<'py>>,
    Vec<PartTold<'py>>,
    Vec<LeafTold<'py>>,
    Vec<NodeTold>,
    Numbers,
    (f64, usize),
);

struct Called {
    reads: Py<PyAny>,
    shift: u32,
}

fn floats(numbers: &[f64]) -> Vec<u8> {
    numbers.iter().flat_map(|number| number.to_le_bytes()).collect()
}

fn blocks_of(found: &Bound<'_, PyAny>) -> PyResult<Vec<Arc<Block>>> {
    let held = found.extract::<Vec<Bound<'_, HeldBlock>>>()?;
    Ok(held.iter().map(|each| Arc::clone(&each.get().0)).collect())
}

impl Reads for Called {
    fn blocks(&mut self, wanted: &[(usize, usize)]) -> Result<Vec<Arc<Block>>, Failure> {
        let found = Python::attach(|py| {
            let told = self.reads.bind(py).call_method1("read_blocks", (wanted.to_vec(),))?;
            blocks_of(&told)
        });
        Ok(found?)
    }

    fn lengths(&mut self, wanted: &[(usize, i64)]) -> Result<Vec<Arc<LengthBlock>>, Failure> {
        let rows: PyResult<Vec<PyBackedBytes>> = Python::attach(|py| {
            self.reads
                .bind(py)
                .call_method1("read_lengths", (wanted.to_vec(),))?
                .extract()
        });
        Ok(length_blocks(&rows?, self.shift)?)
    }

    fn offered(&mut self, places: &[i64], scores: &[f64]) -> Result<Reached, Failure> {
        let found: PyResult<(f64, usize)> = Python::attach(|py| {
            let (places, scores) = (
                PyBytes::new(py, &little_endian(places)),
                PyBytes::new(py, &floats(scores)),
            );
            self.reads.bind(py).call_method1("offered", (places, scores))?.extract()
        });
        let (threshold, held) = found?;
        Ok(Reached { threshold, held })
    }

    fn expired(&mut self) -> Result<bool, Failure> {
        let found: PyResult<bool> = Python::attach(|py| self.reads.bind(py).call_method0("out_of_time")?.extract());
        Ok(found?)
    }
}

fn raised(failure: Failure) -> PyErr {
    failure
        .downcast::<PyErr>()
        .map_or_else(|other| PyValueError::new_err(other.to_string()), |error| *error)
}

fn node((kind, leaf, parts): NodeTold) -> PyResult<Node> {
    match (kind, parts.as_slice()) {
        (0, _) => Ok(Node::Leaf(leaf)),
        (1, &[kept, dropped]) => Ok(Node::Not([kept, dropped])),
        (2, _) => Ok(Node::Or(parts)),
        (3, _) => Ok(Node::And(parts)),
        _ => Err(PyValueError::new_err(KINDLESS)),
    }
}

fn listed((places, counts, segment): &ListedTold<'_>) -> PyResult<ExactList> {
    Ok(ExactList {
        places: numbers_of(places)?,
        counts: numbers_of(counts)?,
        segment: *segment,
    })
}

fn asked((exact, parts, bounding, weight): LeafTold<'_>) -> PyResult<Asked> {
    let exact: PyResult<Vec<ExactList>> = exact.iter().map(listed).collect();
    Ok(Asked {
        exact: exact?,
        parts,
        bounding,
        weight,
    })
}

fn table(told: Option<Bound<'_, PyAny>>) -> PyResult<Arc<LengthsTable>> {
    told.map_or_else(
        || Ok(Arc::default()),
        |held| Ok(Arc::clone(&held.cast::<HeldTable>()?.get().0)),
    )
}

fn segment((first, last, dead, lengths): SegmentTold<'_>) -> PyResult<Segment> {
    let (dead, lengths) = (numbers_of(&dead)?, table(lengths)?);
    Ok(Segment {
        first,
        last,
        dead,
        lengths,
    })
}

fn planned(told: PlanTold<'_>) -> PyResult<(Plan, Reached)> {
    let (segments, parts, leaves, nodes, (average, shape, first, most), (threshold, held)) = told;
    let segments: PyResult<Vec<Segment>> = segments.into_iter().map(segment).collect();
    let parts: PyResult<Vec<Part>> = parts
        .into_iter()
        .map(|(segment, ranges, slack)| {
            let ranges = Arc::clone(&ranges.cast::<HeldRanges>()?.get().0);
            Ok(Part { segment, ranges, slack })
        })
        .collect();
    let (count, leaves) = (
        leaves.len(),
        leaves.into_iter().map(asked).collect::<PyResult<Vec<Asked>>>()?,
    );
    let nodes: PyResult<Vec<Node>> = nodes.into_iter().map(node).collect();
    let expression = Expression::new(nodes?, count).map_err(PyValueError::new_err)?;
    let store = Store {
        segments: segments?,
        parts: parts?,
        shape: shaped(shape)?,
        average,
    };
    let plan = Plan {
        store,
        leaves,
        expression,
        batches: (first, most),
    };
    Ok((plan, Reached { threshold, held }))
}

#[pyfunction]
pub fn searched(py: Python<'_>, plan: &Bound<'_, PyAny>, reads: &Bound<'_, PyAny>) -> PyResult<(bool, Vec<HeldTable>)> {
    let (plan, reached) = planned(plan.extract::<PlanTold<'_>>()?)?;
    let segments = plan.store.segments.iter();
    let tables: Vec<HeldTable> = segments
        .map(|segment| HeldTable(Arc::clone(&segment.lengths)))
        .collect();
    let mut called = Called {
        reads: reads.clone().unbind(),
        shift: plan.store.shape.lengths_shift(),
    };
    let cut = py.detach(|| scored(plan, reached, &mut called)).map_err(raised)?;
    Ok((cut, tables))
}
