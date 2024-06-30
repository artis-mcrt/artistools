use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
extern crate core;
extern crate polars;
extern crate rayon;
use autocompress::autodetect_open;
use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use std::io::Read;

/// Read an ARTIS transitiondata.txt file and return a dictionary of DataFrames, keyed by (atomic_number, ion_stage).
#[pyfunction]
pub fn read_transitiondata(
    py: Python<'_>,
    transitions_filename: String,
    ionlist: Option<Vec<(i32, i32)>>,
) -> Py<PyDict> {
    let firstlevelnumber = 1;
    let mut transitiondata = Vec::new();
    let mut filecontent = String::new();
    autodetect_open(transitions_filename)
        .unwrap()
        .read_to_string(&mut filecontent)
        .unwrap();
    let mut lines = filecontent.lines();

    loop {
        let line;
        match lines.next() {
            Some(l) => line = l.to_owned(),
            None => break,
        }

        let mut linesplit = line.split_whitespace();
        let atomic_number;
        match linesplit.next() {
            Some(token) => atomic_number = token.parse::<i32>().unwrap(),
            _ => continue,
        }

        let ion_stage = linesplit.next().unwrap().parse::<i32>().unwrap();

        let transitioncount = linesplit.next().unwrap().parse::<usize>().unwrap();
        let mut keep_ion = true;
        if ionlist.is_some() {
            keep_ion = false;
            for (a, b) in ionlist.as_ref().unwrap() {
                if atomic_number == *a && ion_stage == *b {
                    keep_ion = true;
                    break;
                }
            }
        }
        if keep_ion {
            let mut vec_lower = vec![0; transitioncount];
            let mut vec_upper = vec![0; transitioncount];
            let mut vec_avalue = vec![0.; transitioncount];
            let mut vec_collstr = vec![0.; transitioncount];
            let mut vec_forbidden = vec![0; transitioncount];
            for i in 0..transitioncount {
                let tableline;
                match lines.next() {
                    Some(l) => tableline = l.to_owned(),
                    None => break,
                }

                // println!("{:?}", line);
                let mut linesplit = tableline.split_whitespace();
                vec_lower[i] = linesplit.next().unwrap().parse::<i32>().unwrap() - firstlevelnumber;
                vec_upper[i] = linesplit.next().unwrap().parse::<i32>().unwrap() - firstlevelnumber;
                vec_avalue[i] = linesplit.next().unwrap().parse::<f32>().unwrap();
                vec_collstr[i] = linesplit.next().unwrap().parse::<f32>().unwrap();
                match linesplit.next() {
                    Some(f) => vec_forbidden[i] = f.parse::<i32>().unwrap(),
                    _ => vec_forbidden[i] = 0,
                }
            }
            let df = df!(
            "lower" => vec_lower.to_owned(),
            "upper" => vec_upper.to_owned(),
            "A" => vec_avalue.to_owned(),
            "collstr" => vec_collstr.to_owned(),
            "forbidden" => vec_forbidden.to_owned())
            .unwrap();

            transitiondata.push(((atomic_number, ion_stage), PyDataFrame(df).into_py(py)));
        } else {
            for _ in 0..transitioncount {
                match lines.next() {
                    Some(_) => (),
                    None => break,
                }
            }
        }
    }

    let dict = transitiondata.into_py_dict_bound(py);
    dict.unbind()
}
