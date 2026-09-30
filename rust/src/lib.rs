use crate::estimators::estimparse;
use crate::estimators::estimparse_allranks;
use crate::estimators::estimtimesteps;
use crate::opacities::sum_binned_line_opacities;
use crate::packetbins::get_bin_indices;
use crate::packetbins::sum_weights_in_bins;
use crate::transitions::read_transitiondata;
use pyo3::prelude::*;

mod estimators;
mod opacities;
mod packetbins;
mod parse;
mod transitions;

/// This is an artistools submodule consisting of compiled rust functions to improve performance.
#[pymodule(gil_used = false)]
fn rustext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimparse, m)?)?;
    m.add_function(wrap_pyfunction!(estimparse_allranks, m)?)?;
    m.add_function(wrap_pyfunction!(estimtimesteps, m)?)?;
    m.add_function(wrap_pyfunction!(read_transitiondata, m)?)?;
    m.add_function(wrap_pyfunction!(sum_binned_line_opacities, m)?)?;
    m.add_function(wrap_pyfunction!(get_bin_indices, m)?)?;
    m.add_function(wrap_pyfunction!(sum_weights_in_bins, m)?)?;
    Ok(())
}
