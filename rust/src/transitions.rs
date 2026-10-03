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
use std::path::{Path, PathBuf};

/// ARTIS numbers levels from one, but artistools uses zero-based level indices
const FIRSTLEVELNUMBER: i32 = 1;

/// Parse the header line of an ion block into (`atomic_number`, `ion_stage`, `transitioncount`)
///
/// Returns `None` for the blank lines that separate the ion blocks, and for the comment lines
/// that artisatomic writes before the header line of each ion.
fn parse_ion_header(line: &str) -> PolarsResult<Option<(i32, i32, usize)>> {
    let mut tokens = line.split_whitespace();
    let Some(atomic_number) = tokens.next() else {
        return Ok(None);
    };
    if atomic_number.starts_with('#') {
        return Ok(None);
    }

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

/// Read the transition tables of an ARTIS transitiondata.txt file, keyed by (`atomic_number`, `ion_stage`)
///
/// `ionlist` of `None` keeps every ion in the file. A table with fewer lines than its header gives is an error.
/// A cut file decodes with no error, thus a short table would lose the lines in silence. ARTIS ends each line with
/// a newline, thus a table whose last line ends the file with no newline is also short.
fn read_transition_tables(
    filepath: &Path,
    ionlist: Option<&HashSet<(i32, i32)>>,
) -> PolarsResult<Vec<((i32, i32), DataFrame)>> {
    let mut filecontent = String::new();
    open_decompressed(filepath)?.read_to_string(&mut filecontent)?;

    let fileendsinsideline = !filecontent.is_empty() && !filecontent.ends_with('\n');
    let mut transitiondata = Vec::new();
    let mut lines = filecontent.lines();
    while let Some(headerline) = lines.next() {
        let Some((atomic_number, ion_stage, transitioncount)) = parse_ion_header(headerline)?
        else {
            continue;
        };
        let ionlines = lines.by_ref().take(transitioncount);

        let linecount = if ionlist.is_none_or(|ions| ions.contains(&(atomic_number, ion_stage))) {
            let df = parse_ion_transitions(ionlines, transitioncount)?;
            let linecount = df.height();
            transitiondata.push(((atomic_number, ion_stage), df));
            linecount
        } else {
            ionlines.count() // skip past this ion's table
        };
        // a cut inside the last line can leave a line that parses, e.g. "7.6" of "7.65e-01"
        let lastlineiscut = linecount > 0 && fileendsinsideline && lines.clone().next().is_none();
        let linecount = linecount - usize::from(lastlineiscut);
        if linecount < transitioncount {
            polars_bail!(
                ComputeError:
                "{}: the file ends after {linecount} of the {transitioncount} transitions of Z={atomic_number} \
                 ion_stage={ion_stage}",
                filepath.display()
            );
        }
    }

    Ok(transitiondata)
}

/// Read an ARTIS transitiondata.txt file, and return a dictionary of `DataFrames` keyed by
/// (`atomic_number`, `ion_stage`).
///
/// The parse runs without the GIL, thus other Python threads can run at the same time.
#[pyfunction]
#[pyo3(signature = (transitions_filename, ionlist=None))]
#[expect(clippy::needless_pass_by_value)]
pub fn read_transitiondata(
    py: Python<'_>,
    transitions_filename: PathBuf,
    ionlist: Option<HashSet<(i32, i32)>>,
) -> PyResult<Py<PyDict>> {
    let transitiondata = py
        .detach(|| read_transition_tables(&transitions_filename, ionlist.as_ref()))
        .map_err(PyPolarsErr::from)?;

    let pyframes: Vec<((i32, i32), PyDataFrame)> = transitiondata
        .into_iter()
        .map(|(ion, df)| (ion, PyDataFrame(df)))
        .collect();
    Ok(pyframes.into_py_dict(py)?.into())
}
