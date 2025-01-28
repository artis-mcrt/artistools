use crate::estimators::estimparse;
use crate::transitions::read_transitiondata;
use pyo3::prelude::*;

mod estimators;
mod transitions;

/// This is an artistools submodule consisting of compiled rust functions to improve performance.
#[pymodule]
fn rustext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimparse, m)?)?;
    m.add_function(wrap_pyfunction!(read_transitiondata, m)?)?;
    Ok(())
}
