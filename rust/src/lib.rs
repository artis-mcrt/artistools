use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
extern crate core;
extern crate polars;
extern crate rayon;
extern crate zstd;
use autocompress::autodetect_open;
use core::f32;
use polars::chunked_array::ChunkedArray;
use polars::datatypes::Float32Type;
use polars::prelude::*;
use polars::series::IntoSeries;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Error as IoError, Lines, Read};
use std::path::Path;
use zstd::stream::read::Decoder;

const ELSYMBOLS: [&str; 119] = [
    "n", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Uut", "Fl", "Uup", "Lv", "Uus", "Uuo",
];

const ROMAN: [&str; 10] = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];

fn read_lines<P>(filename: P) -> Result<Lines<BufReader<File>>, IoError>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(BufReader::new(file).lines())
}

fn read_lines_zst<P>(
    filename: P,
) -> Result<Lines<BufReader<Decoder<'static, BufReader<File>>>>, IoError>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    let decoder = Decoder::new(file)?;
    Ok(BufReader::new(decoder).lines())
}

fn match_colsizes(coldata: &mut HashMap<String, Vec<f32>>, outputrownum: usize) {
    for singlecoldata in coldata.values_mut() {
        if singlecoldata.len() < outputrownum {
            assert_eq!(singlecoldata.len(), outputrownum - 1);
            singlecoldata.push(0.);
        }
    }
}

fn append_or_create(
    coldata: &mut HashMap<String, Vec<f32>>,
    colname: &String,
    colvalue: f32,
    outputrownum: &usize,
) {
    if !coldata.contains_key(colname) {
        coldata.insert(colname.clone(), vec![0.; *outputrownum - 1]);
    }

    let singlecoldata = coldata.get_mut(colname).unwrap();
    singlecoldata.push(colvalue);
    assert_eq!(singlecoldata.len(), *outputrownum, "colname: {:?}", colname);
}

fn parse_estimator_line(
    line: &str,
    mut coldata: &mut HashMap<String, Vec<f32>>,
    outputrownum: &mut usize,
) {
    let linesplit: Vec<&str> = line.split_whitespace().collect();
    if linesplit.len() == 0 {
        return;
    }

    if linesplit[0] == "timestep" {
        match_colsizes(&mut coldata, *outputrownum);
        if linesplit[4] != "EMPTYCELL" {
            //println!("{:?}", line);
            // println!("{:?}", linesplit);

            *outputrownum += 1;
            for i in (0..linesplit.len()).step_by(2) {
                let colname = linesplit[i].to_string();
                let colvalue = linesplit[i + 1].parse::<f32>().unwrap();

                append_or_create(&mut coldata, &colname, colvalue, outputrownum);
            }
        }
    } else if linesplit[1].starts_with("Z=") {
        let atomic_number;
        let startindex;
        if linesplit[1].ends_with("=") {
            atomic_number = linesplit[2].parse::<i32>().unwrap();
            startindex = 3;
        } else {
            // there was no space between Z= and the atomic number
            atomic_number = linesplit[1].replace("Z=", "").parse::<i32>().unwrap();
            startindex = 2;
        }
        let elsym = ELSYMBOLS[atomic_number as usize];

        let variablename = linesplit[0];
        let mut nnelement = 0.0;
        for i in (startindex..linesplit.len()).step_by(2) {
            let ionstagestr = linesplit[i].strip_suffix(":").unwrap();
            let colvalue = linesplit[i + 1].parse::<f32>().unwrap();

            let outcolname: String;
            if variablename == "populations" && ionstagestr == "SUM" {
                nnelement = colvalue;
            } else {
                if variablename == "populations" {
                    if ionstagestr.chars().next().unwrap().is_numeric() {
                        let ionstageroman = ROMAN[ionstagestr.parse::<usize>().unwrap()];
                        outcolname = format!("nnion_{elsym}_{ionstageroman}");
                        nnelement += colvalue;
                    } else {
                        outcolname = format!("nniso_{ionstagestr}");
                    }
                } else {
                    let ionstageroman = ROMAN[ionstagestr.parse::<usize>().unwrap()];
                    outcolname = format!("{variablename}_{elsym}_{ionstageroman}");

                    if variablename.ends_with("*nne") {
                        let colname_nonne = format!(
                            "{}_{}_{}",
                            variablename.strip_suffix("*nne").unwrap(),
                            elsym,
                            ionstageroman
                        );
                        let colvalue_nonne = colvalue / coldata["nne"].last().unwrap();
                        append_or_create(
                            &mut coldata,
                            &colname_nonne,
                            colvalue_nonne,
                            outputrownum,
                        );
                    }
                }
                append_or_create(&mut coldata, &outcolname, colvalue, outputrownum);
            }
        }

        if variablename == "populations" {
            append_or_create(
                &mut coldata,
                &format!("nnelement_{elsym}"),
                nnelement,
                outputrownum,
            );
        }
    } else if linesplit[0].ends_with(":") {
        // deposition, heating, cooling
        for i in (1..linesplit.len()).step_by(2) {
            let firsttoken = linesplit[0];
            let colname: String =
                format!("{}_{}", firsttoken.strip_suffix(":").unwrap(), linesplit[i]);
            let colvalue = linesplit[i + 1].parse::<f32>().unwrap();

            append_or_create(&mut coldata, &colname, colvalue, outputrownum);
        }
    }
}

pub fn read_estimator_file(folderpath: String, rank: i32) -> DataFrame {
    let mut coldata: HashMap<String, Vec<f32>> = HashMap::new();
    let mut outputrownum = 0;

    let filename = format!("{}/estimators_{:04}.out", folderpath, rank);
    if Path::new(&filename).is_file() {
        // println!("Reading file: {:?}", filename);
        for line in read_lines(filename).unwrap() {
            parse_estimator_line(&line.unwrap(), &mut coldata, &mut outputrownum);
        }
    } else {
        let filename_zst = filename + ".zst";
        // println!("Reading file: {:?}", filename_zst);
        for line in read_lines_zst(filename_zst).unwrap() {
            parse_estimator_line(&line.unwrap(), &mut coldata, &mut outputrownum);
        }
    }

    match_colsizes(&mut coldata, outputrownum);
    for singlecolumn in coldata.values() {
        assert_eq!(singlecolumn.len(), outputrownum);
    }

    let df = DataFrame::new(
        coldata
            .iter()
            .map(|(colname, value)| {
                let x =
                    ChunkedArray::<Float32Type>::from_vec(colname, value.to_owned()).into_series();
                x
            })
            .collect(),
    )
    .unwrap();
    df
}

#[pyfunction]
fn estimparse(folderpath: String, rankmin: i32, rankmax: i32) -> PyResult<PyDataFrame> {
    let ranks: Vec<i32> = (rankmin..rankmax + 1).collect();
    let mut vecdfs: Vec<DataFrame> = Vec::new();
    ranks
        .par_iter() // Convert the iterator to a parallel iterator
        .map(|&rank| read_estimator_file(folderpath.clone(), rank))
        .collect_into_vec(&mut vecdfs);

    let dfbatch = polars::functions::concat_df_diagonal(&vecdfs).unwrap();
    Ok(PyDataFrame(dfbatch))
}

#[pyfunction]
fn read_transitiondata(
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

/// A Python module implemented in Rust.
#[pymodule]
fn rustext(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimparse, m)?)?;
    m.add_function(wrap_pyfunction!(read_transitiondata, m)?)?;
    Ok(())
}
