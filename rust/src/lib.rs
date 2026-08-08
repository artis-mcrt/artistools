use crate::estimators::estimparse;
use crate::transitions::read_transitiondata;
use crate::wsv::read_wsv_normalized;
use pyo3::prelude::*;

mod estimators;
mod parse;
mod transitions;
mod wsv;

/// This is an artistools submodule consisting of compiled rust functions to improve performance.
#[pymodule(gil_used = false)]
fn rustext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimparse, m)?)?;
    m.add_function(wrap_pyfunction!(read_transitiondata, m)?)?;
    m.add_function(wrap_pyfunction!(read_wsv_normalized, m)?)?;
    Ok(())
}
