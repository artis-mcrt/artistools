use crate::parse::error_at_line;
use crate::parse::io_error_at;
use crate::parse::malformed;
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

/// Return the count of the numbers at the start of a line, as ARTIS counts the columns of a table
fn count_leading_numbers(line: &str) -> usize {
    line.split_whitespace()
        .take_while(|token| token.parse::<f64>().is_ok_and(f64::is_finite))
        .count()
}

/// The zero-based lower and upper level, the A value, the collision strength, and the forbidden flag of a transition
type Transition = (i32, i32, f32, f32, i32);

/// Parse a line of a transition table. The level numbers of the file start at `firstlevelnumber`.
fn parse_transition(
    line: &str,
    islegacyformat: bool,
    firstlevelnumber: i32,
) -> PolarsResult<Transition> {
    let mut tokens = line.split_whitespace();
    if islegacyformat {
        // ARTIS reads the transition index and does not use it
        next_field::<i32>(&mut tokens, "a transition index")?;
    }
    let lower = next_field::<i32>(&mut tokens, "a lower level")? - firstlevelnumber;
    let upper = next_field::<i32>(&mut tokens, "an upper level")? - firstlevelnumber;
    let avalue = next_field(&mut tokens, "an A value")?;
    if islegacyformat {
        // the values that ARTIS gives to a transition of the legacy format
        return Ok((lower, upper, avalue, -1.0, 0));
    }

    Ok((
        lower,
        upper,
        avalue,
        next_field(&mut tokens, "a collision strength")?,
        next_field(&mut tokens, "a forbidden flag")?,
    ))
}

/// Parse the transition lines of a single ion into a `DataFrame`
///
/// ARTIS takes the format of a table from the count of the numbers in its first line:
/// - 5 numbers: `lower upper A collstr forbidden`;
/// - 4 numbers: `index lower upper A`, which is a legacy format with no collision strength and no forbidden flag.
///
/// Each item of `lines` holds the line number in the file and the text of the line. An error names the file and the
/// line.
fn parse_ion_transitions<'a>(
    lines: impl Iterator<Item = (usize, &'a str)>,
    transitioncount: usize,
    firstlevelnumber: i32,
    filepath: &Path,
) -> PolarsResult<DataFrame> {
    let mut vec_lower: Vec<i32> = Vec::with_capacity(transitioncount);
    let mut vec_upper: Vec<i32> = Vec::with_capacity(transitioncount);
    let mut vec_avalue: Vec<f32> = Vec::with_capacity(transitioncount);
    let mut vec_collstr: Vec<f32> = Vec::with_capacity(transitioncount);
    let mut vec_forbidden: Vec<i32> = Vec::with_capacity(transitioncount);

    let mut lines = lines.peekable();
    let islegacyformat = match lines.peek() {
        None => false,
        Some(&(linenum, line)) => match count_leading_numbers(line) {
            4 => true,
            5 => false,
            count => {
                return Err(error_at_line(
                    &malformed(format!(
                        "the first line of a table has {count} numbers, but ARTIS reads 5 numbers (lower upper A \
                         collstr forbidden) or 4 numbers (index lower upper A)"
                    )),
                    filepath,
                    linenum,
                ));
            }
        },
    };

    for (linenum, line) in lines {
        let (lower, upper, avalue, collstr, forbidden) =
            parse_transition(line, islegacyformat, firstlevelnumber)
                .map_err(|err| error_at_line(&err, filepath, linenum))?;
        vec_lower.push(lower);
        vec_upper.push(upper);
        vec_avalue.push(avalue);
        vec_collstr.push(collstr);
        vec_forbidden.push(forbidden);
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
/// `ionlist` of `None` keeps every ion in the file. The level numbers of the file start at `firstlevelnumber`, which
/// ARTIS takes from the first level of adata.txt.
///
/// A table with fewer lines than its header gives is an error. A cut file decodes with no error, thus a short table
/// would lose the lines in silence. A cut inside the last line of a kept table leaves too few numbers for its format,
/// and the parse then gives an error. ARTIS reads a last line with no newline, thus such a line is not an error.
fn read_transition_tables(
    filepath: &Path,
    ionlist: Option<&HashSet<(i32, i32)>>,
    firstlevelnumber: i32,
) -> PolarsResult<Vec<((i32, i32), DataFrame)>> {
    if !(0..=1).contains(&firstlevelnumber) {
        polars_bail!(ComputeError: "ARTIS numbers the levels from 0 or from 1, not from {firstlevelnumber}");
    }
    let mut filecontent = String::new();
    open_decompressed(filepath)?
        .read_to_string(&mut filecontent)
        .map_err(|err| io_error_at(&err, &filepath.display().to_string()))?;

    let mut transitiondata = Vec::new();
    // an editor gives the first line the number 1
    let mut lines = (1..).zip(filecontent.lines());
    while let Some((headerlinenum, headerline)) = lines.next() {
        let Some((atomic_number, ion_stage, transitioncount)) = parse_ion_header(headerline)
            .map_err(|err| error_at_line(&err, filepath, headerlinenum))?
        else {
            continue;
        };
        let ionlines = lines.by_ref().take(transitioncount);

        let linecount = if ionlist.is_none_or(|ions| ions.contains(&(atomic_number, ion_stage))) {
            let df = parse_ion_transitions(ionlines, transitioncount, firstlevelnumber, filepath)?;
            let linecount = df.height();
            transitiondata.push(((atomic_number, ion_stage), df));
            linecount
        } else {
            ionlines.count() // skip past this ion's table
        };
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
/// The level numbers of the file start at `firstlevelnumber`, and the frames give zero-based level indices. The parse
/// runs without the GIL, thus other Python threads can run at the same time.
#[pyfunction]
#[pyo3(signature = (transitions_filename, ionlist=None, firstlevelnumber=1))]
#[expect(clippy::needless_pass_by_value)]
pub fn read_transitiondata(
    py: Python<'_>,
    transitions_filename: PathBuf,
    ionlist: Option<HashSet<(i32, i32)>>,
    firstlevelnumber: i32,
) -> PyResult<Py<PyDict>> {
    let transitiondata = py
        .detach(|| {
            read_transition_tables(&transitions_filename, ionlist.as_ref(), firstlevelnumber)
        })
        .map_err(PyPolarsErr::from)?;

    let pyframes: Vec<((i32, i32), PyDataFrame)> = transitiondata
        .into_iter()
        .map(|(ion, df)| (ion, PyDataFrame(df)))
        .collect();
    Ok(pyframes.into_py_dict(py)?.into())
}
