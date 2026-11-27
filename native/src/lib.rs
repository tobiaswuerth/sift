mod lists;
mod numbers;
mod packed;

use pyo3::prelude::*;

#[pymodule(name = "_native")]
fn native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(lists::blocked_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::positions_all, module)?)?;
    module.add_function(wrap_pyfunction!(lists::encoded, module)?)?;
    Ok(())
}
