use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;
use rayon::prelude::*;
use std::ops::Range;

/// The number of cells in one group, which is one parallel task. The code reads the line table from memory
/// one time for each group. For each line, the two level populations of all these cells are in two
/// contiguous rows.
const CELLSPERTASK: usize = 32;

/// Below this absolute optical depth, `tau - tau^2 / 2` gives `1 - exp(-tau)` to double precision, and it needs
/// no `exp`. The next term of the series is below `tau^3 / 6`, which is 1.7e-17 of `tau`.
const SERIESTAU: f64 = 1e-8;

fn f64_column<'a>(df: &'a DataFrame, name: &str) -> PolarsResult<&'a [f64]> {
    df.column(name)?
        .as_materialized_series()
        .f64()?
        .cont_slice()
}

fn u32_column<'a>(df: &'a DataFrame, name: &str) -> PolarsResult<&'a [u32]> {
    df.column(name)?
        .as_materialized_series()
        .u32()?
        .cont_slice()
}

/// Return the range of rows of each ion in the levels table, which holds the levels of each ion together
fn ion_level_ranges(level_ion: &[u32], ioncount: usize) -> PolarsResult<Vec<Range<usize>>> {
    let mut ranges = vec![0..0; ioncount];
    let mut start = 0;
    for (row, &ion) in level_ion.iter().enumerate() {
        if level_ion.get(row + 1) != Some(&ion) {
            let range = ranges.get_mut(ion as usize).ok_or_else(
                || polars_err!(ComputeError: "the levels name ion {ion}, but the cells give {ioncount} ions"),
            )?;
            if range.end > 0 {
                polars_bail!(ComputeError: "the levels of ion {ion} are not together in one block");
            }
            *range = start..row + 1;
            start = row + 1;
        }
    }
    Ok(ranges)
}

struct Levels<'a> {
    ranges: Vec<Range<usize>>,
    g: &'a [f64],
    energy_ev: &'a [f64],
}

struct Lines<'a> {
    binindex: &'a [u32],
    lower: &'a [u32],
    upper: &'a [u32],
    sobolev_lower: &'a [f64],
    sobolev_upper: &'a [f64],
    lambda_angstroms: &'a [f64],
}

/// One value for each cell of a group. A fixed length lets the compiler vectorise the loops over the cells.
type CellRow = [f64; CELLSPERTASK];

/// Return the LTE population of each level for each cell, in the order [level][cell]
///
/// A group with fewer cells than `CELLSPERTASK` has a population of zero in the other cells, thus they add
/// nothing to a sum.
fn get_level_pops(
    levels: &Levels,
    t_exc: &[f64],
    nnion: &[&[f64]],
    k_b_ev_per_k: f64,
) -> Vec<CellRow> {
    let mut pops = vec![[0.0; CELLSPERTASK]; levels.g.len()];
    for (cell, &cell_t_exc) in t_exc.iter().enumerate() {
        // a temperature of zero gives 0 / 0 for the ground level, and a negative temperature inverts the
        // populations. Such a cell has no LTE populations, thus each level stays at zero
        if cell_t_exc <= 0.0 {
            continue;
        }
        for (ionrange, ionpops) in levels.ranges.iter().zip(nnion) {
            let cellnnion = ionpops[cell];
            // an ion with no population gives zero in each level
            if cellnnion == 0.0 {
                continue;
            }
            let mut partitionfunction = 0.0;
            for level in ionrange.clone() {
                let boltzmannfactor =
                    levels.g[level] * (-levels.energy_ev[level] / k_b_ev_per_k / cell_t_exc).exp();
                pops[level][cell] = boltzmannfactor;
                partitionfunction += boltzmannfactor;
            }
            for levelpops in &mut pops[ionrange.clone()] {
                levelpops[cell] = levelpops[cell] / partitionfunction * cellnnion;
            }
        }
    }
    pops
}

/// Return the sums over the lines of each bin for a group of cells, each in the order [cell][bin]
///
/// The sums are the expansion opacity, the line-binned opacity, and then one line-binned opacity for each value of
/// `taucaps`, with each `tau_sobolev` capped at that value.
fn sum_cell_group(
    levels: &Levels,
    lines: &Lines,
    t_exc: &[f64],
    nnion: &[&[f64]],
    taucaps: &[f64],
    numbins: usize,
    k_b_ev_per_k: f64,
) -> Vec<Vec<f64>> {
    let pops = get_level_pops(levels, t_exc, nnion, k_b_ev_per_k);

    // in the order [bin][cell], each line adds to one contiguous row of the sums
    let mut exopac = vec![[0.0; CELLSPERTASK]; numbins];
    let mut linebinned = vec![[0.0; CELLSPERTASK]; numbins];
    let mut linebinned_capped = vec![vec![[0.0; CELLSPERTASK]; numbins]; taucaps.len()];

    for line in 0..lines.binindex.len() {
        let lowerpops = &pops[lines.lower[line] as usize];
        let upperpops = &pops[lines.upper[line] as usize];
        let (sobolev_lower, sobolev_upper) = (lines.sobolev_lower[line], lines.sobolev_upper[line]);
        let tau: CellRow = std::array::from_fn(|cell| {
            lowerpops[cell] * sobolev_lower - upperpops[cell] * sobolev_upper
        });

        let lambda = lines.lambda_angstroms[line];
        let bin = lines.binindex[line] as usize;
        // the linear sums have no branch, thus the compiler can vectorise these loops
        for (&celltau, lb) in tau.iter().zip(&mut linebinned[bin]) {
            *lb += celltau * lambda;
        }
        for (cappedsums, &taucap) in linebinned_capped.iter_mut().zip(taucaps) {
            for (&celltau, lbcapped) in tau.iter().zip(&mut cappedsums[bin]) {
                // f64::min returns the cap for a NaN optical depth. NaN must reach each of the sums
                *lbcapped += (if celltau > taucap { taucap } else { celltau }) * lambda;
            }
        }
        for (&celltau, ex) in tau.iter().zip(&mut exopac[bin]) {
            // 1 - exp(-tau) loses most of its digits for a small tau, and it is zero below 5.6e-17. exp_m1
            // keeps them. A NaN optical depth takes the exp_m1 branch, thus it reaches the sum
            let absorbedfraction = if celltau.abs() < SERIESTAU {
                celltau * (1.0 - 0.5 * celltau)
            } else {
                -(-celltau).exp_m1()
            };
            *ex += absorbedfraction * lambda;
        }
    }

    [exopac, linebinned]
        .into_iter()
        .chain(linebinned_capped)
        .map(|binsums| {
            (0..t_exc.len())
                .flat_map(|cell| binsums.iter().map(move |cellsums| cellsums[cell]))
                .collect()
        })
        .collect()
}

/// Return the sums of the Sobolev line opacities in each wavelength bin of each cell, times the wavelength.
///
/// The caller divides each sum by these values:
/// - the bin width;
/// - the speed of light;
/// - the time;
/// - the density.
///
/// The columns are `exopac`, `linebinned`, and then one column for each pair of a name and a cap in `taucaps`. Such
/// a column caps each `tau_sobolev` at the cap of its pair.
///
/// The rows are in the order [cell][bin]. The level populations are the LTE populations at the excitation temperature
/// `T_exc` of each cell. A cell with a `T_exc` of zero or below has no populations, thus each of its sums is zero.
/// `lower` and `upper` give the row of each level in `dflevels`.
///
/// The sum runs without the global interpreter lock (GIL), thus other Python threads can run at the same time.
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
#[expect(
    clippy::too_many_arguments,
    reason = "Python calls the function with each argument by position"
)]
pub fn sum_binned_line_opacities(
    py: Python<'_>,
    dflevels: PyDataFrame,
    dflines: PyDataFrame,
    dfcells: PyDataFrame,
    nnioncolumns: Vec<String>,
    taucaps: Vec<(String, f64)>,
    numbins: usize,
    k_b_ev_per_k: f64,
) -> PyResult<PyDataFrame> {
    let (cappedcolumns, taucaps): (Vec<String>, Vec<f64>) = taucaps.into_iter().unzip();
    let dfsums = py
        .detach(|| {
            let (mut dflevels, mut dflines, mut dfcells) = (dflevels.0, dflines.0, dfcells.0);
            for df in [&mut dflevels, &mut dflines, &mut dfcells] {
                df.rechunk_mut_par();
            }

            let levels = Levels {
                ranges: ion_level_ranges(u32_column(&dflevels, "ionindex")?, nnioncolumns.len())?,
                g: f64_column(&dflevels, "g")?,
                energy_ev: f64_column(&dflevels, "energy_ev")?,
            };
            let lines = Lines {
                binindex: u32_column(&dflines, "lambda_angstroms_binindex")?,
                lower: u32_column(&dflines, "lower")?,
                upper: u32_column(&dflines, "upper")?,
                sobolev_lower: f64_column(&dflines, "sobolev_lower")?,
                sobolev_upper: f64_column(&dflines, "sobolev_upper")?,
                lambda_angstroms: f64_column(&dflines, "lambda_angstroms")?,
            };
            let levelcount = levels.g.len();
            if lines.binindex.iter().any(|&bin| bin as usize >= numbins)
                || lines
                    .lower
                    .iter()
                    .chain(lines.upper)
                    .any(|&level| level as usize >= levelcount)
            {
                polars_bail!(ComputeError: "a line names a bin or a level that does not exist");
            }

            let t_exc = f64_column(&dfcells, "T_exc")?;
            let nnion: Vec<&[f64]> = nnioncolumns
                .iter()
                .map(|name| f64_column(&dfcells, name))
                .collect::<PolarsResult<_>>()?;

            let groupsums: Vec<Vec<Vec<f64>>> = (0..t_exc.len())
                .step_by(CELLSPERTASK)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|firstcell| {
                    let cells = firstcell..(firstcell + CELLSPERTASK).min(t_exc.len());
                    let groupnnion: Vec<&[f64]> = nnion
                        .iter()
                        .map(|ionpops| &ionpops[cells.clone()])
                        .collect();
                    sum_cell_group(
                        &levels,
                        &lines,
                        &t_exc[cells],
                        &groupnnion,
                        &taucaps,
                        numbins,
                        k_b_ev_per_k,
                    )
                })
                .collect();

            let columnnames = ["exopac".to_string(), "linebinned".to_string()]
                .into_iter()
                .chain(cappedcolumns);
            DataFrame::new(
                t_exc.len() * numbins,
                columnnames
                    .enumerate()
                    .map(|(quantity, name)| {
                        let sums: Vec<f64> = groupsums
                            .iter()
                            .flat_map(|sums| sums[quantity].iter().copied())
                            .collect();
                        Column::new(name.into(), sums)
                    })
                    .collect(),
            )
        })
        .map_err(PyPolarsErr::from)?;

    Ok(PyDataFrame(dfsums))
}
