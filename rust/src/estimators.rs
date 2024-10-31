use pyo3::prelude::*;
extern crate autocompress;
extern crate core;
extern crate polars;
extern crate rayon;
use polars::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;

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

/// Ensure that all columns have the same length matching the outputrownum
///
/// If a column is shorter than outputrownum, append zeros to it
/// This is necessary because the estimator files may define different quantities for different
/// cells (e.g. because zero-abundances ions were skipped)
fn match_colsizes(coldata: &mut HashMap<String, Vec<f32>>, outputrownum: usize) {
    for singlecoldata in coldata.values_mut() {
        if singlecoldata.len() < outputrownum {
            assert_eq!(singlecoldata.len(), outputrownum - 1);
            singlecoldata.push(0.);
        }
    }
}

/// Append a value to a column, or create the column if it doesn't exist (filling with zeros)
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

/// Parse a single line from an estimator file and update the column data
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

/// Read a single ARTIS estimators*.out[.zst] file and return a DataFrame
fn read_estimator_file(folderpath: String, rank: i32) -> Result<DataFrame, PolarsError> {
    let mut coldata: HashMap<String, Vec<f32>> = HashMap::new();
    let mut outputrownum = 0;

    // let mut filename = format!("{}/estimators_{:04}.out", folderpath, rank);
    let extensions = vec!["", ".zst", ".gz", ".xz"];
    let mut filename = None;
    for ext in extensions {
        let filenameplusext = format!("{}/estimators_{:04}.out{}", folderpath, rank, ext);
        if Path::new(&filenameplusext).is_file() {
            filename = Some(filenameplusext);
            break;
        }
    }
    if filename.is_none() {
        panic!("No estimator file found for rank {}", rank);
    }

    // println!("Reading file: {:?}", filename.unwrap());
    let file = autocompress::autodetect_open(filename.unwrap());

    BufReader::new(file.unwrap()).lines().for_each(|line| {
        parse_estimator_line(&line.unwrap(), &mut coldata, &mut outputrownum);
    });

    match_colsizes(&mut coldata, outputrownum);
    for singlecolumn in coldata.values() {
        assert_eq!(singlecolumn.len(), outputrownum);
    }

    DataFrame::new(
        coldata
            .iter()
            .map(|(colname, values)| {
                let x = Column::new(colname.into(), values.to_owned());
                x
            })
            .collect(),
    )
}

/// Read the estimator files from rankmin to rankmax and concatenate them into a single DataFrame
#[pyfunction]
pub fn estimparse(folderpath: String, rankmin: i32, rankmax: i32) -> PyResult<PyDataFrame> {
    let ranks: Vec<i32> = (rankmin..rankmax + 1).collect();
    let mut vecdfs: Vec<DataFrame> = Vec::new();
    ranks
        .par_iter() // Convert the iterator to a parallel iterator
        .map(|&rank| read_estimator_file(folderpath.clone(), rank).unwrap())
        .collect_into_vec(&mut vecdfs);

    let dfbatch = polars::functions::concat_df_diagonal(&vecdfs).unwrap();
    Ok(PyDataFrame(dfbatch))
}
