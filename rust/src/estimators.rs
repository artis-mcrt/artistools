use crate::parse::malformed;
use crate::parse::open_decompressed;
use crate::parse::parse_field;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::{BufRead as _, BufReader};
use std::path::{Path, PathBuf};

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

/// Split a line into (name, value) token pairs, ignoring an unpaired trailing token
fn token_pairs<'a>(tokens: &'a [&'a str]) -> impl Iterator<Item = (&'a str, &'a str)> {
    tokens.chunks_exact(2).map(|pair| (pair[0], pair[1]))
}

/// Look up the element symbol for an atomic number given as a decimal string
fn elsymbol(atomic_number: &str) -> PolarsResult<&'static str> {
    let atomic_number: usize = parse_field(atomic_number, "an atomic number")?;

    ELSYMBOLS.get(atomic_number).copied().ok_or_else(|| {
        malformed(format!(
            "no element symbol for atomic number {atomic_number}"
        ))
    })
}

/// Convert an ion stage given as a decimal string into a roman numeral, e.g. "2" -> "II"
fn ionstage_roman(ionstage: &str) -> PolarsResult<&'static str> {
    let ionstage: usize = parse_field(ionstage, "an ion stage")?;

    ROMAN
        .get(ionstage)
        .copied()
        .ok_or_else(|| malformed(format!("no roman numeral for ion stage {ionstage}")))
}

/// Column-oriented store of the values parsed out of one estimator file.
///
/// Every column is kept at the same length, so a column that first appears part-way through the
/// file is back-filled with zeros, and columns that a cell does not mention are padded with a zero
/// once that cell ends. This is necessary because the estimator files may define different
/// quantities for different cells (e.g. because zero-abundance ions were skipped).
#[derive(Default)]
struct EstimatorColumns {
    coldata: HashMap<String, Vec<f32>>,
    /// index columns, kept as integers because f32 cannot hold every cell number of a large 3D grid exactly
    intcoldata: HashMap<String, Vec<i32>>,
    /// number of cells seen so far, including the one currently being filled
    rownum: usize,
}

/// Columns that identify a row rather than measure a quantity, and so must stay exact integers.
/// Only names verified to be written as integers belong here: parsing is strict, so listing a column that
/// some ARTIS version writes as a float would turn a readable file into a hard error.
const INDEX_COLUMNS: [&str; 2] = ["timestep", "modelgridindex"];

/// Pad every column of a store out to `rownum`, for the columns the current cell did not define
fn resize_columns<T: Copy + Default>(coldata: &mut HashMap<String, Vec<T>>, rownum: usize) {
    for values in coldata.values_mut() {
        values.resize(rownum, T::default());
    }
}

/// Set a column value for the current cell, creating the column if it doesn't exist yet
fn push_value<T: Copy + Default>(
    coldata: &mut HashMap<String, Vec<T>>,
    rownum: usize,
    colname: String,
    colvalue: T,
) -> PolarsResult<()> {
    let values = coldata.entry(colname).or_default();
    if values.len() >= rownum {
        return Err(malformed(
            "a column was given two values for one cell".into(),
        ));
    }
    // back-fill with zeros if the column first appeared part-way through the file
    values.resize(rownum - 1, T::default());
    values.push(colvalue);

    Ok(())
}

impl EstimatorColumns {
    /// Finish the current cell by padding every column that it did not define with a zero
    fn end_cell(&mut self) {
        resize_columns(&mut self.coldata, self.rownum);
        resize_columns(&mut self.intcoldata, self.rownum);
    }

    /// Set a measured column value for the current cell
    fn push(&mut self, colname: String, colvalue: f32) -> PolarsResult<()> {
        push_value(&mut self.coldata, self.rownum, colname, colvalue)
    }

    /// Set an index column value for the current cell
    fn push_int(&mut self, colname: String, colvalue: i32) -> PolarsResult<()> {
        push_value(&mut self.intcoldata, self.rownum, colname, colvalue)
    }

    /// Parse a single line from an estimator file and update the column data
    fn parse_line(&mut self, line: &str) -> PolarsResult<()> {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&firsttoken, rest)) = tokens.split_first() else {
            return Ok(());
        };

        if firsttoken == "timestep" {
            self.parse_cell_header(&tokens)?;
        } else if rest.first().is_some_and(|token| token.starts_with("Z=")) {
            self.parse_ion_line(firsttoken, rest)?;
        } else if let Some(prefix) = firsttoken.strip_suffix(':') {
            // deposition, heating, cooling
            for (name, value) in token_pairs(rest) {
                self.push(format!("{prefix}_{name}"), parse_field(value, "a number")?)?;
            }
        }

        Ok(())
    }

    /// Finish the previous cell and start a new one from a line like
    /// `timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 71393.3`
    fn parse_cell_header(&mut self, tokens: &[&str]) -> PolarsResult<()> {
        self.end_cell();

        if tokens.get(4) == Some(&"EMPTYCELL") {
            return Ok(());
        }

        self.rownum += 1;
        for (colname, value) in token_pairs(tokens) {
            if INDEX_COLUMNS.contains(&colname) {
                self.push_int(colname.to_owned(), parse_field(value, "an integer")?)?;
            } else {
                self.push(colname.to_owned(), parse_field(value, "a number")?)?;
            }
        }

        Ok(())
    }

    /// Parse a per-ion line, e.g. `populations  Z=26  1: 6.226e+05  2: 8.059e+01  3: 3.940e-24`
    fn parse_ion_line(&mut self, variablename: &str, tokens: &[&str]) -> PolarsResult<()> {
        let ztoken = tokens[0]
            .strip_prefix("Z=")
            .ok_or_else(|| malformed(format!("{:?} does not start with Z=", tokens[0])))?;

        let (atomic_number, stagetokens) = if ztoken.is_empty() {
            // the atomic number is a separate token: "Z= 26"
            tokens[1..]
                .split_first()
                .ok_or_else(|| malformed("ion line ends after Z=".into()))?
        } else {
            // no space after the equals sign: "Z=26"
            (&ztoken, &tokens[1..])
        };
        let elsym = elsymbol(atomic_number)?;

        let mut nnelement = 0.0;
        for (ionstage, value) in token_pairs(stagetokens) {
            let ionstage = ionstage.strip_suffix(':').ok_or_else(|| {
                malformed(format!("ion stage {ionstage:?} has no trailing colon"))
            })?;
            let colvalue: f32 = parse_field(value, "a number")?;

            if variablename == "populations" {
                if ionstage == "SUM" {
                    nnelement = colvalue;
                } else if ionstage.starts_with(|c: char| c.is_ascii_digit()) {
                    nnelement += colvalue;
                    self.push(
                        format!("nnion_{elsym}_{}", ionstage_roman(ionstage)?),
                        colvalue,
                    )?;
                } else {
                    // an isotopic population, where the ion stage field holds e.g. "Ni56"
                    self.push(format!("nniso_{ionstage}"), colvalue)?;
                }
                continue;
            }

            let ionstageroman = ionstage_roman(ionstage)?;
            if let Some(varname_nonne) = variablename.strip_suffix("*nne") {
                // also store the quantity divided by the electron density of this cell. The column is only as long
                // as the current row once this cell has set it, so a shorter column means nne belongs to an
                // earlier cell and must not be used here
                let nne = self
                    .coldata
                    .get("nne")
                    .filter(|values| values.len() == self.rownum)
                    .and_then(|values| values.last())
                    .copied()
                    .ok_or_else(|| malformed("nne is not set for this cell".into()))?;

                self.push(
                    format!("{varname_nonne}_{elsym}_{ionstageroman}"),
                    colvalue / nne,
                )?;
            }
            self.push(format!("{variablename}_{elsym}_{ionstageroman}"), colvalue)?;
        }

        if variablename == "populations" {
            self.push(format!("nnelement_{elsym}"), nnelement)?;
        }

        Ok(())
    }

    /// Finish the last cell and convert the columns into a `DataFrame`
    fn into_dataframe(mut self) -> PolarsResult<DataFrame> {
        self.end_cell();

        let columns: Vec<Column> = self
            .intcoldata
            .into_iter()
            .map(|(colname, values)| Column::new(colname.into(), values))
            .chain(
                self.coldata
                    .into_iter()
                    .map(|(colname, values)| Column::new(colname.into(), values)),
            )
            .collect();

        DataFrame::new(self.rownum, columns)
    }
}

/// Find the estimator file of an MPI rank, which may or may not be compressed
fn find_estimator_file(folderpath: &Path, rank: i32) -> Option<PathBuf> {
    ["", ".zst", ".gz", ".xz"]
        .iter()
        .map(|ext| folderpath.join(format!("estimators_{rank:04}.out{ext}")))
        .find(|filepath| filepath.is_file())
}

/// Read a single ARTIS estimators*.out[.zst] file and return a `DataFrame`
fn read_estimator_file(folderpath: &Path, rank: i32) -> PolarsResult<DataFrame> {
    let filepath = find_estimator_file(folderpath, rank).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no estimator file found for rank {rank} in {}",
                folderpath.display()
            ),
        )
    })?;

    let mut columns = EstimatorColumns::default();
    for (linenum, line) in BufReader::new(open_decompressed(&filepath)?)
        .lines()
        .enumerate()
    {
        columns.parse_line(&line?).map_err(|err| {
            err.wrap_msg(|msg| format!("{}:{}: {msg}", filepath.display(), linenum + 1))
        })?;
    }

    columns.into_dataframe()
}

/// Read the estimator files from rankmin to rankmax and concatenate them into a single `DataFrame`
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
pub fn estimparse(folderpath: PathBuf, rankmin: i32, rankmax: i32) -> PyResult<PyDataFrame> {
    let vecdfs: Vec<DataFrame> = (rankmin..=rankmax)
        .into_par_iter()
        .map(|rank| read_estimator_file(&folderpath, rank))
        .collect::<PolarsResult<_>>()
        .map_err(PyPolarsErr::from)?;

    let dfbatch = polars::functions::concat_df_diagonal(&vecdfs).map_err(PyPolarsErr::from)?;

    Ok(PyDataFrame(dfbatch))
}
