use pyo3::types::{IntoPyDict, PyDict};
use pyo3::{prelude::*, IntoPyObject};
extern crate core;
extern crate polars;
extern crate rayon;
use autocompress::autodetect_open;
use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashSet;
use std::io::Read;

/// Read an ARTIS transitiondata.txt file and return a dictionary of DataFrames, keyed by (atomic_number, ion_stage).
#[pyfunction]
#[pyo3(signature = (transitions_filename, ionlist=None))]
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
        let keep_ion;
        if ionlist.is_none() {
            keep_ion = true;
        } else {
            keep_ion = ionlist
                .as_ref()
                .unwrap()
                .contains(&(atomic_number, ion_stage));
        }

        if keep_ion {
            let mut vec_lower = Vec::with_capacity(transitioncount);
            let mut vec_upper = Vec::with_capacity(transitioncount);
            let mut vec_avalue = Vec::with_capacity(transitioncount);
            let mut vec_collstr = Vec::with_capacity(transitioncount);
            let mut vec_forbidden = Vec::with_capacity(transitioncount);
            for _ in 0..transitioncount {
                let tableline;
                match lines.next() {
                    Some(l) => tableline = l.to_owned(),
                    None => break,
                }

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
            "lower" => vec_lower.to_owned(),
            "upper" => vec_upper.to_owned(),
            "A" => vec_avalue.to_owned(),
            "collstr" => vec_collstr.to_owned(),
            "forbidden" => vec_forbidden.to_owned())
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
