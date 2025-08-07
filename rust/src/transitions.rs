use autocompress::autodetect_open;
use polars::prelude::*;
use pyo3::types::{IntoPyDict as _, PyDict};
use pyo3::{IntoPyObject as _, prelude::*};
use pyo3_polars::PyDataFrame;
use std::collections::HashSet;
use std::io::Read as _;

/// Read an ARTIS transitiondata.txt file and return a dictionary of `DataFrames`, keyed by (`atomic_number`, `ion_stage`).
#[pyfunction]
#[pyo3(signature = (transitions_filename, ionlist=None))]
#[expect(clippy::needless_pass_by_value)]
pub fn read_transitiondata(
    py: Python<'_>,
    transitions_filename: String,
    ionlist: Option<HashSet<(i32, i32)>>,
) -> Py<PyDict> {
    let firstlevelnumber = 1;
    let mut transitiondata = Vec::new();
    let mut filecontent = String::new();
    autodetect_open(transitions_filename)
        .unwrap()
        .read_to_string(&mut filecontent)
        .unwrap();
    let mut lines = filecontent.lines();

    while let Some(l) = lines.next() {
        let mut linesplit = l.split_whitespace();
        let atomic_number = match linesplit.next() {
            Some(token) => token.parse::<i32>().unwrap(),
            _ => continue,
        };

        let ion_stage = linesplit.next().unwrap().parse::<i32>().unwrap();

        let transitioncount = linesplit.next().unwrap().parse::<usize>().unwrap();

        // keep the ion if it is in the ionlist or if ionlist is None (i.e., all ions are kept)
        let keep_ion = match ionlist {
            Some(ref someionlist) => someionlist.contains(&(atomic_number, ion_stage)),
            None => true,
        };

        if keep_ion {
            let mut vec_lower = Vec::with_capacity(transitioncount);
            let mut vec_upper = Vec::with_capacity(transitioncount);
            let mut vec_avalue = Vec::with_capacity(transitioncount);
            let mut vec_collstr = Vec::with_capacity(transitioncount);
            let mut vec_forbidden = Vec::with_capacity(transitioncount);
            for _ in 0..transitioncount {
                let tableline = match lines.next() {
                    Some(l) => l.to_owned(),
                    None => break,
                };

                // println!("{:?}", line);
                let mut linesplit = tableline.split_whitespace();
                vec_lower
                    .push(linesplit.next().unwrap().parse::<i32>().unwrap() - firstlevelnumber);
                vec_upper
                    .push(linesplit.next().unwrap().parse::<i32>().unwrap() - firstlevelnumber);
                vec_avalue.push(linesplit.next().unwrap().parse::<f32>().unwrap());
                vec_collstr.push(linesplit.next().unwrap().parse::<f32>().unwrap());
                match linesplit.next() {
                    Some(f) => vec_forbidden.push(f.parse::<i32>().unwrap()),
                    _ => vec_forbidden.push(0),
                }
            }
            let pydf = PyDataFrame(
                df!(
            "lower" => vec_lower,
            "upper" => vec_upper,
            "A" => vec_avalue,
            "collstr" => vec_collstr,
            "forbidden" => vec_forbidden)
                .unwrap(),
            );

            transitiondata.push(((atomic_number, ion_stage), pydf.into_pyobject(py).unwrap()));
        } else {
            for _ in 0..transitioncount {
                match lines.next() {
                    Some(_) => (),
                    None => break,
                }
            }
        }
    }

    transitiondata.into_py_dict(py).unwrap().into()
}
