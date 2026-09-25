use crate::estimators::estimparse;
use crate::opacities::sum_binned_line_opacities;
use crate::transitions::read_transitiondata;
use pyo3::prelude::*;

mod estimators;
mod opacities;
mod parse;
mod transitions;

/// This is an artistools submodule consisting of compiled rust functions to improve performance.
#[pymodule(gil_used = false)]
fn rustext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimparse, m)?)?;
    m.add_function(wrap_pyfunction!(read_transitiondata, m)?)?;
    m.add_function(wrap_pyfunction!(sum_binned_line_opacities, m)?)?;
    Ok(())
}
