use pyo3::prelude::*;
extern crate core;
extern crate polars;
extern crate rayon;
extern crate zstd;
use crate::estimators::estimparse;
use crate::transitions::read_transitiondata;

mod estimators;
mod transitions;

/// A Python module implemented in Rust.
#[pymodule]
fn rustext(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimparse, m)?)?;
    m.add_function(wrap_pyfunction!(read_transitiondata, m)?)?;
    Ok(())
}
