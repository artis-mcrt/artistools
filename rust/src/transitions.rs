use autocompress::autodetect_open;
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
fn parse_ion_header(line: &str) -> Option<(i32, i32, usize)> {
    let mut tokens = line.split_whitespace();
    let atomic_number = tokens.next()?.parse().unwrap();
    let ion_stage = tokens.next().unwrap().parse().unwrap();
    let transitioncount = tokens.next().unwrap().parse().unwrap();

    Some((atomic_number, ion_stage, transitioncount))
}

/// Parse the transition lines of a single ion into a `DataFrame`
fn parse_ion_transitions<'a>(
    lines: impl Iterator<Item = &'a str>,
    transitioncount: usize,
) -> PolarsResult<DataFrame> {
    let mut vec_lower = Vec::with_capacity(transitioncount);
    let mut vec_upper = Vec::with_capacity(transitioncount);
    let mut vec_avalue = Vec::with_capacity(transitioncount);
    let mut vec_collstr = Vec::with_capacity(transitioncount);
    let mut vec_forbidden = Vec::with_capacity(transitioncount);

    for line in lines {
        let mut tokens = line.split_whitespace();
        vec_lower.push(tokens.next().unwrap().parse::<i32>().unwrap() - FIRSTLEVELNUMBER);
        vec_upper.push(tokens.next().unwrap().parse::<i32>().unwrap() - FIRSTLEVELNUMBER);
        vec_avalue.push(tokens.next().unwrap().parse::<f32>().unwrap());
        vec_collstr.push(tokens.next().unwrap().parse::<f32>().unwrap());
        // the forbidden flag is absent from files written by older ARTIS versions
        vec_forbidden.push(tokens.next().map_or(0, |token| token.parse().unwrap()));
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
    autodetect_open(transitions_filename)?.read_to_string(&mut filecontent)?;

    let mut transitiondata = Vec::new();
    let mut lines = filecontent.lines();
    while let Some(headerline) = lines.next() {
        let Some((atomic_number, ion_stage, transitioncount)) = parse_ion_header(headerline) else {
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
