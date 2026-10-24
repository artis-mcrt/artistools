use crate::parse::TextBlocks;
use crate::parse::error_at_line;
use crate::parse::io_error_at;
use crate::parse::malformed;
use crate::parse::open_decompressed;
use crate::parse::open_text_blocks;
use crate::parse::parse_field;
use polars::prelude::*;
use pyo3::exceptions::PyRuntimeWarning;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use std::ffi::CString;
use std::io::{BufRead, BufReader};
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

/// The roman numeral of each ion stage, up to 119 for the bare nucleus of the heaviest element. The table must agree
/// with `roman_numerals` in artistools/atomic/core.py
const ROMAN: [&str; 120] = [
    "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV",
    "XV", "XVI", "XVII", "XVIII", "XIX", "XX", "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI",
    "XXVII", "XXVIII", "XXIX", "XXX", "XXXI", "XXXII", "XXXIII", "XXXIV", "XXXV", "XXXVI",
    "XXXVII", "XXXVIII", "XXXIX", "XL", "XLI", "XLII", "XLIII", "XLIV", "XLV", "XLVI", "XLVII",
    "XLVIII", "XLIX", "L", "LI", "LII", "LIII", "LIV", "LV", "LVI", "LVII", "LVIII", "LIX", "LX",
    "LXI", "LXII", "LXIII", "LXIV", "LXV", "LXVI", "LXVII", "LXVIII", "LXIX", "LXX", "LXXI",
    "LXXII", "LXXIII", "LXXIV", "LXXV", "LXXVI", "LXXVII", "LXXVIII", "LXXIX", "LXXX", "LXXXI",
    "LXXXII", "LXXXIII", "LXXXIV", "LXXXV", "LXXXVI", "LXXXVII", "LXXXVIII", "LXXXIX", "XC", "XCI",
    "XCII", "XCIII", "XCIV", "XCV", "XCVI", "XCVII", "XCVIII", "XCIX", "C", "CI", "CII", "CIII",
    "CIV", "CV", "CVI", "CVII", "CVIII", "CIX", "CX", "CXI", "CXII", "CXIII", "CXIV", "CXV",
    "CXVI", "CXVII", "CXVIII", "CXIX",
];

/// Split the tokens of a line into (name, value) pairs. A name at the end of the line with no value is an error,
/// because the column of that name would get a zero that the line does not give
fn token_pairs<'a>(
    mut tokens: impl Iterator<Item = &'a str>,
) -> impl Iterator<Item = PolarsResult<(&'a str, &'a str)>> {
    std::iter::from_fn(move || {
        let name = tokens.next()?;
        Some(
            tokens.next().map(|value| (name, value)).ok_or_else(|| {
                malformed(format!("the line ends after {name:?}, which has no value"))
            }),
        )
    })
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
    /// the position of each measured column in `colnames` and `coldata`
    colindex: PlHashMap<String, usize>,
    colnames: Vec<String>,
    coldata: Vec<Vec<f32>>,
    /// index columns, kept as integers because f32 cannot hold every cell number of a large 3D grid exactly
    intcoldata: HashMap<String, Vec<i32>>,
    /// the name of the next column value. A name that the store already holds needs no new String
    namebuf: String,
    /// the columns of the values of the cell before, in the order of the text. Each cell usually gives its values in
    /// the same order, thus a comparison with the expected name replaces most lookups of a name
    prevcellcolumns: Vec<usize>,
    /// the columns of the values of the current cell so far
    cellcolumns: Vec<usize>,
    /// the position of each column in `prevcellcolumns`, or `usize::MAX` for a column that the cell before did not give
    prevcellposition: Vec<usize>,
    /// the position in `prevcellcolumns` of the expected column of the next value
    expectedposition: usize,
    /// number of cells seen so far, including the one currently being filled
    rownum: usize,
    /// the count of the values above the range of f32, which the columns hold as infinity
    overflowcount: usize,
}

/// Columns that identify a row rather than measure a quantity, and so must stay exact integers.
/// Only names verified to be written as integers belong here: parsing is strict, so listing a column that
/// some ARTIS version writes as a float would turn a readable file into a hard error.
const INDEX_COLUMNS: [&str; 2] = ["timestep", "modelgridindex"];

/// Set the value of a column for the current cell, after the zeros of the cells before it
fn push_value<T: Copy + Default>(
    values: &mut Vec<T>,
    rownum: usize,
    colvalue: T,
) -> PolarsResult<()> {
    if rownum == 0 {
        return Err(malformed(
            "the line gives a value, but no cell header with values comes before it".into(),
        ));
    }
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
        for values in &mut self.coldata {
            values.resize(self.rownum, 0.0);
        }
        for values in self.intcoldata.values_mut() {
            values.resize(self.rownum, 0);
        }
        if !self.cellcolumns.is_empty() {
            std::mem::swap(&mut self.prevcellcolumns, &mut self.cellcolumns);
            self.cellcolumns.clear();
            self.prevcellposition.clear();
            self.prevcellposition.resize(self.coldata.len(), usize::MAX);
            for (position, &colindex) in self.prevcellcolumns.iter().enumerate() {
                self.prevcellposition[colindex] = position;
            }
        }
        self.expectedposition = 0;
    }

    /// Set `namebuf` to the words of a column name, with an underscore between each pair of words
    fn set_namebuf(&mut self, namewords: &[&str]) {
        self.namebuf.clear();
        for (wordindex, nameword) in namewords.iter().enumerate() {
            if wordindex > 0 {
                self.namebuf.push('_');
            }
            self.namebuf.push_str(nameword);
        }
    }

    /// Return the column that `namewords` names, and add the column if the store does not hold it
    fn find_or_add_column(&mut self, namewords: &[&str]) -> usize {
        self.set_namebuf(namewords);
        if let Some(&colindex) = self
            .prevcellcolumns
            .get(self.expectedposition)
            .filter(|&&colindex| self.colnames[colindex] == self.namebuf)
        {
            self.expectedposition += 1;
            return colindex;
        }
        let colindex = if let Some(&colindex) = self.colindex.get(self.namebuf.as_str()) {
            colindex
        } else {
            let colindex = self.coldata.len();
            self.colindex.insert(self.namebuf.clone(), colindex);
            self.colnames.push(self.namebuf.clone());
            self.coldata.push(Vec::new());
            colindex
        };
        // the cell skipped some columns of the cell before or gave a new column. The expected order continues after
        // this column
        if let Some(&position) = self.prevcellposition.get(colindex)
            && position != usize::MAX
        {
            self.expectedposition = position + 1;
        }
        colindex
    }

    /// Set the value of the measured column that `namewords` names for the current cell
    fn push(&mut self, namewords: &[&str], colvalue: f32) -> PolarsResult<()> {
        let colindex = self.find_or_add_column(namewords);
        self.cellcolumns.push(colindex);
        push_value(&mut self.coldata[colindex], self.rownum, colvalue)
    }

    /// Set an index column value for the current cell
    fn push_int(&mut self, colname: &str, colvalue: i32) -> PolarsResult<()> {
        let values = self.intcoldata.entry(colname.to_owned()).or_default();
        push_value(values, self.rownum, colvalue)
    }

    /// Parse a measured value into f32
    ///
    /// ARTIS writes a rate far below the smallest f32, e.g. "5.313e-95" in the `gamma_R` row of the test model. Rust
    /// parses such a token to 0.0 with no error, which is the right value for a rate that small. ARTIS can also write a
    /// value above the largest f32, e.g. the photoionisation correction ratio of a cell whose analytic rate is
    /// subnormal. Such a value becomes infinity, and the reader counts it for a warning.
    fn parse_value(&mut self, token: &str) -> PolarsResult<f32> {
        let value: f64 = parse_field(token, "a number")?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "a value above the range of f32 becomes infinity, and the reader counts it for a warning"
        )]
        let value_f32 = value as f32;
        if value.is_finite() && value_f32.is_infinite() {
            self.overflowcount += 1;
        }

        Ok(value_f32)
    }

    /// Parse a single line from an estimator file and update the column data
    fn parse_line(&mut self, line: &str) -> PolarsResult<()> {
        let mut tokens = line.split_ascii_whitespace();
        let Some(firsttoken) = tokens.next() else {
            return Ok(());
        };

        if firsttoken == "timestep" {
            self.parse_cell_header(line)?;
        } else if tokens
            .clone()
            .next()
            .is_some_and(|token| token.starts_with("Z="))
        {
            self.parse_ion_line(firsttoken, tokens)?;
        } else if let Some(prefix) = firsttoken.strip_suffix(':') {
            // emission_ana, deposition, heating, cooling
            for pair in token_pairs(tokens) {
                let (name, value) = pair?;
                let value = self.parse_value(value)?;
                self.push(&[prefix, name], value)?;
            }
        } else {
            // each version of ARTIS since 2017 writes only the forms above. A line of a different form would lose
            // its values
            return Err(malformed(format!(
                "the line starts with {firsttoken:?} and does not have the form of an estimator line. A cell header \
                 starts with \"timestep\". An ion line gives Z= after its name. A line of totals ends its first \
                 word with a colon. A file of the classic ARTIS code needs --classicartis"
            )));
        }

        Ok(())
    }

    /// Finish the previous cell and start a new one from a line like
    /// `timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 71393.3`
    fn parse_cell_header(&mut self, line: &str) -> PolarsResult<()> {
        self.end_cell();

        if line.split_ascii_whitespace().nth(4) == Some("EMPTYCELL") {
            return Ok(());
        }

        self.rownum += 1;
        for pair in token_pairs(line.split_ascii_whitespace()) {
            let (colname, value) = pair?;
            if INDEX_COLUMNS.contains(&colname) {
                self.push_int(colname, parse_field(value, "an integer")?)?;
            } else {
                let value = self.parse_value(value)?;
                self.push(&[colname], value)?;
            }
        }
        // the row of a cell with no index value would get a zero, which names a different cell
        for colname in INDEX_COLUMNS {
            if self
                .intcoldata
                .get(colname)
                .is_none_or(|values| values.len() < self.rownum)
            {
                return Err(malformed(format!("the cell header gives no {colname}")));
            }
        }

        Ok(())
    }

    /// Parse a per-ion line, e.g. `populations  Z=26  1: 6.226e+05  2: 8.059e+01  3: 3.940e-24`
    fn parse_ion_line<'a>(
        &mut self,
        variablename: &str,
        mut tokens: impl Iterator<Item = &'a str>,
    ) -> PolarsResult<()> {
        let firsttoken = tokens.next().unwrap_or_default();
        let ztoken = firsttoken
            .strip_prefix("Z=")
            .ok_or_else(|| malformed(format!("{firsttoken:?} does not start with Z=")))?;

        let atomic_number = if ztoken.is_empty() {
            // the atomic number is a separate token: "Z= 26"
            tokens
                .next()
                .ok_or_else(|| malformed("ion line ends after Z=".into()))?
        } else {
            // no space after the equals sign: "Z=26"
            ztoken
        };
        let elsym = elsymbol(atomic_number)?;

        // the quantity divided by the electron density (nne) of this cell is also a column. After this cell sets nne,
        // its column has one value for each row. A shorter column holds the nne of an earlier cell, thus the code
        // cannot use it
        let nonne = variablename
            .strip_suffix("*nne")
            .map(|varname_nonne| {
                self.colindex
                    .get("nne")
                    .map(|&colindex| &self.coldata[colindex])
                    .filter(|values| values.len() == self.rownum)
                    .and_then(|values| values.last())
                    .map(|&nne| (varname_nonne, nne))
                    .ok_or_else(|| malformed("nne is not set for this cell".into()))
            })
            .transpose()?;

        let mut nnelement = 0.0;
        for pair in token_pairs(tokens) {
            let (ionstage, value) = pair?;
            let ionstage = ionstage.strip_suffix(':').ok_or_else(|| {
                malformed(format!("ion stage {ionstage:?} has no trailing colon"))
            })?;
            let colvalue = self.parse_value(value)?;

            if variablename == "populations" {
                if ionstage == "SUM" {
                    nnelement = colvalue;
                } else if ionstage.starts_with(|c: char| c.is_ascii_digit()) {
                    nnelement += colvalue;
                    self.push(&["nnion", elsym, ionstage_roman(ionstage)?], colvalue)?;
                } else {
                    // an isotopic population, where the ion stage field holds e.g. "Ni56"
                    self.push(&["nniso", ionstage], colvalue)?;
                }
                continue;
            }

            let ionstageroman = ionstage_roman(ionstage)?;
            if let Some((varname_nonne, nne)) = nonne {
                self.push(&[varname_nonne, elsym, ionstageroman], colvalue / nne)?;
            }
            self.push(&[variablename, elsym, ionstageroman], colvalue)?;
        }

        if variablename == "populations" {
            self.push(&["nnelement", elsym], nnelement)?;
        }

        Ok(())
    }

    /// Finish the last cell and convert the columns into a `DataFrame`
    ///
    /// The frame always holds the index columns. Thus a text whose cells are all empty gives the index columns and no
    /// row. Only a text with no complete cell gives a frame with no column.
    fn into_dataframe(mut self) -> PolarsResult<DataFrame> {
        self.end_cell();
        for colname in INDEX_COLUMNS {
            self.intcoldata.entry(colname.to_owned()).or_default();
        }

        // a column keeps its buffer until the end of the conversion, and the growth of a Vec can leave half of it unused
        let columns: Vec<Column> = self
            .intcoldata
            .into_iter()
            .map(|(colname, mut values)| {
                values.shrink_to_fit();
                Column::new(colname.into(), values)
            })
            .chain(
                self.colnames
                    .into_iter()
                    .zip(self.coldata)
                    .map(|(colname, mut values)| {
                        values.shrink_to_fit();
                        Column::new(colname.into(), values)
                    }),
            )
            .collect();

        DataFrame::new(self.rownum, columns)
    }
}

/// The count of the values above the range of f32 in a text, and the line of the first such value
#[derive(Clone, Copy, Default)]
struct Overflows {
    count: usize,
    firstline: usize,
}

impl Overflows {
    /// Add the overflows of a later part of the same text, whose first line has the number `firstlinenum`
    fn add(&mut self, later: Self, firstlinenum: usize) {
        if self.count == 0 && later.count > 0 {
            self.firstline = firstlinenum + later.firstline - 1;
        }
        self.count += later.count;
    }

    /// Return the text of a warning about the values of a file above the range of f32, or None for no such value
    fn warning(self, filepath: &Path) -> Option<String> {
        (self.count > 0).then(|| {
            format!(
                "{}:{}: the file holds {} values above the range of f32, and this line holds the first of them. The \
                 columns hold infinity for each of these values",
                filepath.display(),
                self.firstline,
                self.count
            )
        })
    }
}

/// An error in a part of an estimator text, at a line number that counts from the first line of the part
///
/// The threads parse the parts in parallel, thus the line number of the first line of a part is known only after the
/// parse of the parts before it.
enum PartError {
    /// a byte that is not UTF-8
    NotUtf8 { linenum: usize },
    /// a line that the parser cannot read
    Malformed { linenum: usize, err: PolarsError },
}

impl PartError {
    /// Return the error with the file and the line in the file, for a part whose first line has the number
    /// `firstlinenum`
    fn in_file(self, filepath: &Path, firstlinenum: usize) -> PolarsError {
        match self {
            // the same message as a read of each line. pyo3-polars gives Python only the text of the io::Error, thus
            // the place goes into that text
            Self::NotUtf8 { linenum } => io_error_at(
                &std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "stream did not contain valid UTF-8",
                ),
                &format!("{}:{}", filepath.display(), firstlinenum + linenum - 1),
            )
            .into(),
            Self::Malformed { linenum, err } => {
                error_at_line(&err, filepath, firstlinenum + linenum - 1)
            }
        }
    }
}

/// The table of one part of an estimator text, its overflows, and the count of its lines
struct ParsedPart {
    dfpart: DataFrame,
    overflows: Overflows,
    linecount: usize,
}

/// Parse one part of an estimator text. The line numbers of an error and of the overflows count from the first line
/// of the part.
///
/// The thread of each part checks its UTF-8 and counts its lines. A check by the thread that splits the text would
/// make each part wait for the parts before it.
fn parse_part(text: &[u8]) -> Result<ParsedPart, PartError> {
    let text = std::str::from_utf8(text).map_err(|err| PartError::NotUtf8 {
        linenum: count_newlines(&text[..err.valid_up_to()]) + 1,
    })?;
    let mut columns = EstimatorColumns::default();
    let mut firstoverflowline = 0;
    let mut linecount = 0;
    for (linenum, line) in (1..).zip(text.lines()) {
        columns
            .parse_line(line)
            .map_err(|err| PartError::Malformed { linenum, err })?;
        if firstoverflowline == 0 && columns.overflowcount > 0 {
            firstoverflowline = linenum;
        }
        linecount = linenum;
    }
    let overflows = Overflows {
        count: columns.overflowcount,
        firstline: firstoverflowline,
    };
    let dfpart = columns
        .into_dataframe()
        .map_err(|err| PartError::Malformed {
            linenum: linecount,
            err,
        })?;

    Ok(ParsedPart {
        dfpart,
        overflows,
        linecount,
    })
}

/// Find the estimator file of an MPI rank, which may or may not be compressed
fn find_estimator_file(folderpath: &Path, rank: i32) -> PolarsResult<PathBuf> {
    ["", ".zst", ".gz", ".xz"]
        .iter()
        .map(|ext| folderpath.join(format!("estimators_{rank:04}.out{ext}")))
        .find(|filepath| filepath.is_file())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "no estimator file found for rank {rank} in {}",
                    folderpath.display()
                ),
            )
            .into()
        })
}

/// Join the `DataFrame`s of the files of the ranks, or of the parts of one file, into one `DataFrame`
///
/// Within one file, `EstimatorColumns` gives a zero to a cell that does not write a quantity, e.g. the ion of an
/// element that the cell does not hold. A diagonal join gives a null to the rows of a file or a part that does not
/// write the quantity at all. The boundaries of the ranks and of the parts are arbitrary, thus a zero replaces each
/// such null, and both forms of the text give the same values. No frame gives a frame with no column.
fn concat_estimator_frames(vecdfs: &[DataFrame]) -> PolarsResult<DataFrame> {
    match vecdfs {
        // concat_df_diagonal panics on no frame. A text with no complete cell gives no part
        [] => Ok(DataFrame::empty()),
        [dfsingle] => Ok(dfsingle.clone()),
        _ => polars::functions::concat_df_diagonal(vecdfs)?.fill_null(FillNullStrategy::Zero),
    }
}

/// Read the files of the ranks from rankmin to rankmax in text batches, and give the `DataFrame` of each text batch to
/// `on_batch` in the order of the ranks
///
/// The threads read one file of a rank each. A text batch ends after the file of a rank that brings its text to
/// `textbatchbytes` or more, thus a run with few ranks and many cells also keeps its memory small. The result is a
/// warning for each file with values above the range of f32.
fn read_rankfile_textbatches(
    folderpath: &Path,
    rankmin: i32,
    rankmax: i32,
    textbatchbytes: usize,
    mut on_batch: impl FnMut(DataFrame) -> PolarsResult<()>,
) -> PolarsResult<Vec<String>> {
    if rankmin > rankmax {
        polars_bail!(ComputeError: "the rank range {rankmin} to {rankmax} holds no rank");
    }
    let ranks: Vec<i32> = (rankmin..=rankmax).collect();
    let mut warnings = Vec::new();
    let mut textbatchframes = Vec::new();
    let mut textbytes = 0;
    for rankgroup in ranks.chunks(rayon::current_num_threads()) {
        let rankresults: Vec<(DataFrame, Option<String>, usize)> = rankgroup
            .par_iter()
            .map(|&rank| read_estimator_text(&find_estimator_file(folderpath, rank)?, false))
            .collect::<PolarsResult<_>>()?;
        for (dfrank, warning, ranktextbytes) in rankresults {
            textbatchframes.push(dfrank);
            warnings.extend(warning);
            textbytes += ranktextbytes;
            if textbytes >= textbatchbytes {
                on_batch(concat_estimator_frames(&textbatchframes)?)?;
                textbatchframes.clear();
                textbytes = 0;
            }
        }
    }
    if !textbatchframes.is_empty() {
        on_batch(concat_estimator_frames(&textbatchframes)?)?;
    }

    Ok(warnings)
}

/// Call a Python function for each text batch of a reader, and return the warnings of the reader
///
/// The read runs without the GIL, and each call of `on_batch` holds it. An exception in `on_batch` stops the read.
fn read_with_batch_callback<T>(
    py: Python<'_>,
    on_batch: &Py<PyAny>,
    read: impl Send + FnOnce(&mut dyn FnMut(DataFrame) -> PolarsResult<()>) -> PolarsResult<T>,
) -> PyResult<T>
where
    T: Send,
{
    let mut callbackerror: Option<PyErr> = None;
    let result = py.detach(|| {
        read(&mut |dftextbatch| {
            Python::attach(|py| on_batch.call1(py, (PyDataFrame(dftextbatch),)))
                .map(|_| ())
                .map_err(|err| {
                    callbackerror = Some(err);
                    polars_err!(ComputeError: "the batch callback raised an exception")
                })
        })
    });
    if let Some(err) = callbackerror {
        return Err(err);
    }
    Ok(result.map_err(PyPolarsErr::from)?)
}

/// Read the estimator files from rankmin to rankmax and concatenate them into a single `DataFrame`
///
/// The parse runs without the GIL, thus other Python threads can run at the same time.
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
pub fn estimparse(
    py: Python<'_>,
    folderpath: PathBuf,
    rankmin: i32,
    rankmax: i32,
) -> PyResult<PyDataFrame> {
    let (dfranks, warnings) = py
        .detach(|| -> PolarsResult<(DataFrame, Vec<String>)> {
            let mut dfranks = DataFrame::empty();
            // a text batch of unlimited size holds the text of all the ranks
            let warnings =
                read_rankfile_textbatches(&folderpath, rankmin, rankmax, usize::MAX, |dfbatch| {
                    dfranks = dfbatch;
                    Ok(())
                })?;
            Ok((dfranks, warnings))
        })
        .map_err(PyPolarsErr::from)?;
    give_warnings(py, warnings.iter())?;

    Ok(PyDataFrame(dfranks))
}

/// Read the estimator files from rankmin to rankmax in text batches, and call `on_batch` with the `DataFrame` of each
/// text batch in the order of the ranks
///
/// A text batch ends after the file of a rank that brings its text to `textbatchbytes` or more, thus a caller can
/// convert files whose table is larger than the memory.
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
pub fn estimparse_textbatches(
    py: Python<'_>,
    folderpath: PathBuf,
    rankmin: i32,
    rankmax: i32,
    textbatchbytes: usize,
    on_batch: Py<PyAny>,
) -> PyResult<()> {
    let warnings = read_with_batch_callback(py, &on_batch, |on_textbatch| {
        read_rankfile_textbatches(&folderpath, rankmin, rankmax, textbatchbytes, on_textbatch)
    })?;
    give_warnings(py, warnings.iter())
}

/// Give a Python warning for each text, e.g. a warning about the values of a file above the range of f32
///
/// The warning needs the GIL, thus the caller gives it after the parse.
fn give_warnings<'a>(py: Python<'_>, warnings: impl Iterator<Item = &'a String>) -> PyResult<()> {
    for warning in warnings {
        PyErr::warn(
            py,
            py.get_type::<PyRuntimeWarning>().as_any(),
            &CString::new(warning.as_str())?,
            1,
        )?;
    }

    Ok(())
}

/// The minimum size of the text of one part of an estimator file. One thread parses each part.
const TEXTPART_BYTES: usize = 16 * 1024 * 1024;

/// Split the blocks of a text into parts that end at the end of a cell. An empty line ends each cell, thus no
/// cell spans two parts. The last part drops a cell that the end of the text cuts.
///
/// One thread splits the text for all the threads that parse it, thus it does as little as possible with the text. A
/// block of ARTIS frames ends at the end of a cell, and it then becomes a part with no copy.
struct TextParts<'a> {
    blocks: TextBlocks,
    /// the file of the text, which an I/O error names
    filepath: &'a Path,
    /// the text after the last complete cell of the part before
    carry: Vec<u8>,
    finished: bool,
}

/// Return the end of the last empty line of a text, which is the end of the last complete cell, or None
fn last_cell_end(text: &[u8]) -> Option<usize> {
    // the newline that ends the last complete line. A text after it is an incomplete line
    let mut lineend = text.iter().rposition(|&byte| byte == b'\n')?;
    loop {
        let linestart = text[..lineend]
            .iter()
            .rposition(|&byte| byte == b'\n')
            .map_or(0, |newline| newline + 1);
        if text[linestart..lineend].iter().all(u8::is_ascii_whitespace) {
            return Some(lineend + 1);
        }
        if linestart == 0 {
            return None;
        }
        lineend = linestart - 1;
    }
}

/// Return the number of lines that end in a text
#[expect(
    clippy::naive_bytecount,
    reason = "the compiler vectorises this loop, and the bytecount crate adds a dependency"
)]
fn count_newlines(text: &[u8]) -> usize {
    text.iter().filter(|&&byte| byte == b'\n').count()
}

impl Iterator for TextParts<'_> {
    type Item = std::io::Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let mut text = std::mem::take(&mut self.carry);
        loop {
            match self.blocks.next() {
                None => {
                    self.finished = true;
                    // A text that does not end with an empty line ends inside a cell. The decoder gives the complete
                    // zstd blocks of a frame that a stopped job cut, and it gives no error. Thus the part removes the
                    // cut cell, and the conversion then drops its timestep as incomplete.
                    text.truncate(last_cell_end(&text).unwrap_or(0));
                    break;
                }
                Some(Ok(block)) => {
                    if text.is_empty() {
                        text = block;
                    } else {
                        text.extend_from_slice(&block);
                    }
                    // a cell that is longer than the text so far needs another block. The search starts at the end
                    // of the text, thus it reads only the last cell
                    if text.len() >= TEXTPART_BYTES
                        && let Some(cellsend) = last_cell_end(&text)
                    {
                        self.carry = text.split_off(cellsend);
                        break;
                    }
                }
                Some(Err(err)) => {
                    self.finished = true;
                    return Some(Err(io_error_at(&err, &self.filepath.display().to_string())));
                }
            }
        }

        (!text.is_empty()).then_some(Ok(text))
    }
}

/// Read an estimator text in text batches, and give the `DataFrame` of each text batch to `on_batch` in the order of
/// the file
///
/// A text batch holds the text of whole parts, and it ends after the part that brings its text to `textbatchbytes` or
/// more. The reader holds one text batch at a time. The threads parse the parts of a text batch in parallel, and the
/// rows keep the order of the file. A text that does not end with an empty line ends inside a cell, e.g. because a job
/// stopped during the write. The reader drops that cell in both forms of the text, thus each value in the table comes
/// from a complete cell of the text. The result is a warning about the values above the range of f32, or None for no
/// such value, and the size of the text.
fn read_estimator_textbatches(
    filepath: &Path,
    parallelzstd: bool,
    textbatchbytes: usize,
    mut on_batch: impl FnMut(DataFrame) -> PolarsResult<()>,
) -> PolarsResult<(Option<String>, usize)> {
    let mut parts = TextParts {
        blocks: open_text_blocks(filepath, parallelzstd)?,
        filepath,
        carry: Vec::new(),
        finished: false,
    };
    let mut overflows = Overflows::default();
    // the line number of the first line of the next part
    let mut nextlinenum = 1;
    let mut totaltextbytes = 0;
    while let Some(firstpart) = parts.next() {
        let mut textbytes = firstpart.as_ref().map_or(0, Vec::len);
        let parsedparts: Vec<(usize, std::io::Result<Result<ParsedPart, PartError>>)> =
            if parts.finished {
                // the last part of the text needs no thread and no concatenation, e.g. the whole text of a small rank file
                vec![(0, firstpart.map(|text| parse_part(&text)))]
            } else {
                let laterparts = std::iter::from_fn(|| {
                    if textbytes >= textbatchbytes {
                        return None;
                    }
                    let part = parts.next()?;
                    textbytes += part.as_ref().map_or(0, Vec::len);
                    Some(part)
                });
                // the threads parse each part while the splitter reads the next, and the result of each part stays,
                // thus an error gets its line number from the line counts of the parts before it
                let mut parsedparts: Vec<(usize, std::io::Result<Result<ParsedPart, PartError>>)> =
                    std::iter::once(firstpart)
                        .chain(laterparts)
                        .enumerate()
                        .par_bridge()
                        .map(|(partindex, part)| (partindex, part.map(|text| parse_part(&text))))
                        .collect();
                parsedparts.sort_unstable_by_key(|(partindex, _)| *partindex);
                parsedparts
            };
        totaltextbytes += textbytes;

        let mut vecdfs = Vec::with_capacity(parsedparts.len());
        for (_, parsedpart) in parsedparts {
            let parsedpart = parsedpart?.map_err(|err| err.in_file(filepath, nextlinenum))?;
            overflows.add(parsedpart.overflows, nextlinenum);
            nextlinenum += parsedpart.linecount;
            vecdfs.push(parsedpart.dfpart);
        }
        on_batch(concat_estimator_frames(&vecdfs)?)?;
    }

    Ok((overflows.warning(filepath), totaltextbytes))
}

/// Read an estimator text, which is the file of one rank or the file of all ranks, and return a `DataFrame`
///
/// The second item is a warning about the values above the range of f32, or None if the text holds none. The third item
/// is the size of the text.
fn read_estimator_text(
    filepath: &Path,
    parallelzstd: bool,
) -> PolarsResult<(DataFrame, Option<String>, usize)> {
    let mut dfestimators = DataFrame::empty();
    // a text batch of unlimited size holds the whole text
    let (warning, textbytes) =
        read_estimator_textbatches(filepath, parallelzstd, usize::MAX, |dfbatch| {
            dfestimators = dfbatch;
            Ok(())
        })?;

    Ok((dfestimators, warning, textbytes))
}

/// Read the estimator file of all ranks, e.g. `estimators_allranks.out.zst`, and return a `DataFrame`
///
/// ARTIS writes this file in place of one file for each rank. The parse runs without the GIL.
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
pub fn estimparse_allranks(py: Python<'_>, filepath: PathBuf) -> PyResult<PyDataFrame> {
    let (dfallranks, warning, _) = py
        .detach(|| read_estimator_text(&filepath, true))
        .map_err(PyPolarsErr::from)?;
    give_warnings(py, warning.iter())?;

    Ok(PyDataFrame(dfallranks))
}

/// Read the estimator file of all ranks in text batches, and call `on_batch` with the `DataFrame` of each text batch
///
/// A text batch ends after the part of the text that brings its text to `textbatchbytes` or more. The reader holds one
/// text batch at a time, thus a caller can convert a file whose table is larger than the memory.
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
pub fn estimparse_allranks_textbatches(
    py: Python<'_>,
    filepath: PathBuf,
    textbatchbytes: usize,
    on_batch: Py<PyAny>,
) -> PyResult<()> {
    let (warning, _) = read_with_batch_callback(py, &on_batch, |on_textbatch| {
        read_estimator_textbatches(&filepath, true, textbatchbytes, on_textbatch)
    })?;
    give_warnings(py, warning.iter())
}

/// Return the timesteps of the cells of an estimator file in ascending order, without repeats
///
/// The scan parses only the timestep of each cell header, thus it is much faster than a full parse of the file. A
/// header counts when its line is complete, i.e. a newline ends it. The end of a cut file can cut a header line,
/// e.g. "timestep " or "timestep 1" of "timestep 11". Such a line gave an error, or it named a timestep that the file
/// does not hold, thus the scan ignores it.
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
pub fn estimtimesteps(py: Python<'_>, filepath: PathBuf) -> PyResult<Vec<i32>> {
    let timesteps = py
        .detach(|| -> PolarsResult<Vec<i32>> {
            let mut reader = BufReader::new(open_decompressed(&filepath)?);
            let mut timesteps = BTreeSet::new();
            let mut line = String::new();
            let mut linenum: usize = 0;
            loop {
                line.clear();
                let bytecount = reader.read_line(&mut line).map_err(|err| {
                    io_error_at(&err, &format!("{}:{}", filepath.display(), linenum + 1))
                })?;
                if bytecount == 0 {
                    break;
                }
                linenum += 1;
                let mut tokens = line.split_whitespace();
                if tokens.next() != Some("timestep") || !line.ends_with('\n') {
                    continue;
                }
                let timestep: i32 = tokens
                    .next()
                    .ok_or_else(|| malformed("the cell header gives no timestep".into()))
                    .and_then(|token| parse_field(token, "an integer"))
                    .map_err(|err| error_at_line(&err, &filepath, linenum))?;
                timesteps.insert(timestep);
            }
            Ok(timesteps.into_iter().collect())
        })
        .map_err(PyPolarsErr::from)?;

    Ok(timesteps)
}
