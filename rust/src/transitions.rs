use crate::parse::next_field;
use crate::parse::open_decompressed;
use crate::parse::parse_field;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict as _, PyDict};
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;
use std::collections::HashSet;
use std::io::Read as _;
use std::path::PathBuf;

/// ARTIS numbers levels from one, but artistools uses zero-based level indices
const FIRSTLEVELNUMBER: i32 = 1;

/// Parse the header line of an ion block into (`atomic_number`, `ion_stage`, `transitioncount`)
///
/// Returns `None` for the blank lines that separate the ion blocks.
fn parse_ion_header(line: &str) -> PolarsResult<Option<(i32, i32, usize)>> {
    let mut tokens = line.split_whitespace();
    let Some(atomic_number) = tokens.next() else {
        return Ok(None);
    };

    Ok(Some((
        parse_field(atomic_number, "an atomic number")?,
        next_field(&mut tokens, "an ion stage")?,
        next_field(&mut tokens, "a transition count")?,
    )))
}

/// Parse the transition lines of a single ion into a `DataFrame`
fn parse_ion_transitions<'a>(
    lines: impl Iterator<Item = &'a str>,
    transitioncount: usize,
) -> PolarsResult<DataFrame> {
    let mut vec_lower: Vec<i32> = Vec::with_capacity(transitioncount);
    let mut vec_upper: Vec<i32> = Vec::with_capacity(transitioncount);
    let mut vec_avalue: Vec<f32> = Vec::with_capacity(transitioncount);
    let mut vec_collstr: Vec<f32> = Vec::with_capacity(transitioncount);
    let mut vec_forbidden: Vec<i32> = Vec::with_capacity(transitioncount);

    for line in lines {
        let mut tokens = line.split_whitespace();
        vec_lower.push(next_field::<i32>(&mut tokens, "a lower level")? - FIRSTLEVELNUMBER);
        vec_upper.push(next_field::<i32>(&mut tokens, "an upper level")? - FIRSTLEVELNUMBER);
        vec_avalue.push(next_field(&mut tokens, "an A value")?);
        vec_collstr.push(next_field(&mut tokens, "a collision strength")?);
        // the forbidden flag is absent from files written by older ARTIS versions
        vec_forbidden.push(match tokens.next() {
            Some(token) => parse_field(token, "a forbidden flag")?,
            None => 0,
        });
    }

    df!(
        "lower" => vec_lower,
        "upper" => vec_upper,
        "A" => vec_avalue,
        "collstr" => vec_collstr,
        "forbidden" => vec_forbidden,
    )
}

/// Read an ARTIS transitiondata.txt file and return a dictionary of `DataFrames`, keyed by (`atomic_number`, `ion_stage`).
#[pyfunction]
#[pyo3(signature = (transitions_filename, ionlist=None))]
#[expect(clippy::needless_pass_by_value)]
pub fn read_transitiondata(
    py: Python<'_>,
    transitions_filename: PathBuf,
    ionlist: Option<HashSet<(i32, i32)>>,
) -> PyResult<Py<PyDict>> {
    let mut filecontent = String::new();
    open_decompressed(&transitions_filename)?.read_to_string(&mut filecontent)?;

    let mut transitiondata = Vec::new();
    let mut lines = filecontent.lines();
    while let Some(headerline) = lines.next() {
        let Some((atomic_number, ion_stage, transitioncount)) =
            parse_ion_header(headerline).map_err(PyPolarsErr::from)?
        else {
            continue;
        };
        let ionlines = lines.by_ref().take(transitioncount);

        // ionlist=None means keep every ion in the file
        if ionlist
            .as_ref()
            .is_none_or(|ions| ions.contains(&(atomic_number, ion_stage)))
        {
            let df = parse_ion_transitions(ionlines, transitioncount).map_err(PyPolarsErr::from)?;
            transitiondata.push(((atomic_number, ion_stage), PyDataFrame(df)));
        } else {
            ionlines.for_each(drop); // skip past this ion's table
        }
    }

    Ok(transitiondata.into_py_dict(py)?.into())
}
