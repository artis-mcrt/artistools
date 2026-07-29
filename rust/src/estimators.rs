use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
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

/// Suffixes tried (in order) when looking for the estimator file of an MPI rank
const FILE_EXTENSIONS: [&str; 4] = ["", ".zst", ".gz", ".xz"];

/// Split a line into (name, value) token pairs, ignoring an unpaired trailing token
fn token_pairs<'a>(tokens: &'a [&'a str]) -> impl Iterator<Item = (&'a str, &'a str)> {
    tokens.chunks_exact(2).map(|pair| (pair[0], pair[1]))
}

fn parse_f32(token: &str) -> f32 {
    token
        .parse()
        .unwrap_or_else(|_| panic!("{token:?} is not a number"))
}

/// Convert an ion stage given as a decimal string into a roman numeral, e.g. "2" -> "II"
fn ionstage_roman(ionstage: &str) -> &'static str {
    let ionstage: usize = ionstage
        .parse()
        .unwrap_or_else(|_| panic!("{ionstage:?} is not an ion stage"));

    ROMAN[ionstage]
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
    /// number of cells seen so far, including the one currently being filled
    rownum: usize,
}

impl EstimatorColumns {
    /// Finish the current cell by padding every column that it did not define with a zero
    fn end_cell(&mut self) {
        for values in self.coldata.values_mut() {
            if values.len() < self.rownum {
                assert_eq!(values.len(), self.rownum - 1);
                values.push(0.);
            }
        }
    }

    /// Set a column value for the current cell, creating the column if it doesn't exist yet
    fn push(&mut self, colname: String, colvalue: f32) {
        let rownum = self.rownum;
        match self.coldata.entry(colname) {
            Entry::Occupied(entry) => {
                assert_eq!(
                    entry.get().len(),
                    rownum - 1,
                    "column {:?} was given two values for one cell",
                    entry.key()
                );
                entry.into_mut().push(colvalue);
            }
            Entry::Vacant(entry) => {
                let mut values = vec![0.; rownum - 1];
                values.push(colvalue);
                entry.insert(values);
            }
        }
    }

    /// Parse a single line from an estimator file and update the column data
    fn parse_line(&mut self, line: &str) {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some((&firsttoken, rest)) = tokens.split_first() else {
            return;
        };

        if firsttoken == "timestep" {
            self.parse_cell_header(&tokens);
        } else if rest.first().is_some_and(|token| token.starts_with("Z=")) {
            self.parse_ion_line(firsttoken, rest);
        } else if let Some(prefix) = firsttoken.strip_suffix(':') {
            // deposition, heating, cooling
            for (name, value) in token_pairs(rest) {
                self.push(format!("{prefix}_{name}"), parse_f32(value));
            }
        }
    }

    /// Start a new cell, e.g. `timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 71393.3`
    fn parse_cell_header(&mut self, tokens: &[&str]) {
        self.end_cell();

        if tokens.get(4) == Some(&"EMPTYCELL") {
            return;
        }

        self.rownum += 1;
        for (colname, value) in token_pairs(tokens) {
            self.push(colname.to_owned(), parse_f32(value));
        }
    }

    /// Parse a per-ion line, e.g. `populations  Z=26  1: 6.226e+05  2: 8.059e+01  3: 3.940e-24`
    fn parse_ion_line(&mut self, variablename: &str, tokens: &[&str]) {
        // the atomic number either follows "Z=" directly or is given as a separate token
        let ztoken = tokens[0].strip_prefix("Z=").expect("checked by caller");
        let (atomic_number, tokens) = if ztoken.is_empty() {
            (tokens[1], &tokens[2..])
        } else {
            (ztoken, &tokens[1..])
        };
        let elsym = ELSYMBOLS[atomic_number
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{atomic_number:?} is not an atomic number"))];

        let mut nnelement = 0.0;
        for (ionstage, value) in token_pairs(tokens) {
            let ionstage = ionstage
                .strip_suffix(':')
                .unwrap_or_else(|| panic!("{ionstage:?} should end with a colon"));
            let colvalue = parse_f32(value);

            if variablename == "populations" {
                if ionstage == "SUM" {
                    nnelement = colvalue;
                } else if ionstage.starts_with(|c: char| c.is_ascii_digit()) {
                    nnelement += colvalue;
                    self.push(
                        format!("nnion_{elsym}_{}", ionstage_roman(ionstage)),
                        colvalue,
                    );
                } else {
                    // an isotopic population, where the ion stage field holds e.g. "Ni56"
                    self.push(format!("nniso_{ionstage}"), colvalue);
                }
                continue;
            }

            let ionstageroman = ionstage_roman(ionstage);
            if let Some(varname_nonne) = variablename.strip_suffix("*nne") {
                // also store the quantity divided by the electron density of this cell
                let nne = self.coldata["nne"]
                    .last()
                    .copied()
                    .expect("nne should be set by the timestep line of this cell");
                self.push(
                    format!("{varname_nonne}_{elsym}_{ionstageroman}"),
                    colvalue / nne,
                );
            }
            self.push(format!("{variablename}_{elsym}_{ionstageroman}"), colvalue);
        }

        if variablename == "populations" {
            self.push(format!("nnelement_{elsym}"), nnelement);
        }
    }

    /// Finish the last cell and convert the columns into a `DataFrame`
    fn into_dataframe(mut self) -> PolarsResult<DataFrame> {
        self.end_cell();

        DataFrame::new(
            self.rownum,
            self.coldata
                .into_iter()
                .map(|(colname, values)| Column::new(colname.into(), values))
                .collect(),
        )
    }
}

/// Find the estimator file of an MPI rank, which may or may not be compressed
fn find_estimator_file(folderpath: &Path, rank: i32) -> Option<PathBuf> {
    FILE_EXTENSIONS
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
    for line in BufReader::new(autocompress::autodetect_open(filepath)?).lines() {
        columns.parse_line(&line?);
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
