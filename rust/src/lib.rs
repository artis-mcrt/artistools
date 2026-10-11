use crate::estimators::estimparse;
use crate::estimators::estimparse_allranks;
use crate::estimators::estimparse_allranks_textbatches;
use crate::estimators::estimparse_textbatches;
use crate::estimators::estimtimesteps;
use crate::opacities::sum_binned_line_opacities;
use crate::packetbins::get_bin_indices;
use crate::packetbins::sum_weights_in_bins;
use pyo3::prelude::*;

mod estimators;
mod opacities;
mod packetbins;
mod parse;

/// This is an artistools submodule consisting of compiled rust functions to improve performance.
#[pymodule(gil_used = false)]
fn rustext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // artistools sets POLARS_MAX_THREADS before it loads this module, thus the global pool of this module and the pool
    // of polars have the same size. An error says that the pool exists already, e.g. after a second load
    if let Some(threadcount) = std::env::var("POLARS_MAX_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threadcount)
            .build_global();
    }
    m.add_function(wrap_pyfunction!(estimparse, m)?)?;
    m.add_function(wrap_pyfunction!(estimparse_allranks, m)?)?;
    m.add_function(wrap_pyfunction!(estimparse_allranks_textbatches, m)?)?;
    m.add_function(wrap_pyfunction!(estimparse_textbatches, m)?)?;
    m.add_function(wrap_pyfunction!(estimtimesteps, m)?)?;
    m.add_function(wrap_pyfunction!(sum_binned_line_opacities, m)?)?;
    m.add_function(wrap_pyfunction!(get_bin_indices, m)?)?;
    m.add_function(wrap_pyfunction!(sum_weights_in_bins, m)?)?;
    Ok(())
}
