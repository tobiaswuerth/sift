mod building;
mod lists;
mod merge;
mod numbers;
mod packed;
mod segment;
mod words;

use pyo3::prelude::*;

#[pymodule(name = "_native")]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lists::blocked_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::positions_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::encoded, module)?)?;
    module.add_function(wrap_pyfunction!(lists::ranged, module)?)?;
    module.add_function(wrap_pyfunction!(lists::with_lengths, module)?)?;
    module.add_class::<building::Words>()?;
    module.add_class::<building::Building>()?;
    module.add_class::<merge::Merging>()?;
    Ok(())
}
