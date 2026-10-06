use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use pyo3_polars::error::PyPolarsErr;
use rayon::prelude::*;

/// The count of parts of the packets. Each part has its own sums, and the parts add in a fixed order. Thus the
/// result does not depend on the count of threads.
const PARTCOUNT: usize = 64;

/// The smallest part, because a part with few packets costs more to make and to add than it saves.
const MINPARTSIZE: usize = 1 << 16;

/// Return the count of values in each part, which does not depend on the count of threads
///
/// A part makes, fills, and adds one sum for each bin of each group, thus a part holds at least one value for each
/// such sum. With 16500 bins of 100 groups and 4 million values, this rule gives 3 parts and not 62, and the
/// sums of the parts need 120 MB of memory and not 2.5 GB.
fn get_partsize(valuecount: usize, sumcount: usize) -> usize {
    valuecount
        .div_ceil(PARTCOUNT)
        .max(MINPARTSIZE)
        .max(sumcount)
}

/// The edges of the bins. A bin is [lower, upper), and the last bin also holds its upper edge.
struct BinEdges<'a> {
    edges: &'a [f64],
    /// The first edge and the inverse of the width, if all the bins have one width. The guess of the bin from these
    /// values then takes one step, and the exact edges correct the guess.
    uniform: Option<(f64, f64)>,
}

impl<'a> BinEdges<'a> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of bins is far below 2^52"
    )]
    fn new(edges: &'a [f64]) -> Self {
        let nbins = edges.len() - 1;
        let width = (edges[nbins] - edges[0]) / nbins as f64;
        let isuniform = edges
            .windows(2)
            .all(|pair| ((pair[1] - pair[0]) - width).abs() <= 1e-9 * width.abs());
        let uniform = (isuniform && width > 0.0).then(|| (edges[0], 1.0 / width));
        Self { edges, uniform }
    }

    fn nbins(&self) -> usize {
        self.edges.len() - 1
    }

    /// Return the bin of a value, or None if the value is outside the edges or is NaN.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is inside the edges, thus the guess is a small positive number"
    )]
    fn index(&self, value: f64) -> Option<usize> {
        let nbins = self.nbins();
        if !(value >= self.edges[0] && value <= self.edges[nbins]) {
            return None;
        }
        let mut bin = match self.uniform {
            Some((low, inversewidth)) => (((value - low) * inversewidth) as usize).min(nbins - 1),
            None => self
                .edges
                .partition_point(|&edge| edge <= value)
                .saturating_sub(1)
                .min(nbins - 1),
        };
        // the guess of a uniform grid can miss by one bin through rounding, and the exact edges decide
        while bin > 0 && value < self.edges[bin] {
            bin -= 1;
        }
        while bin + 1 < nbins && value >= self.edges[bin + 1] {
            bin += 1;
        }
        Some(bin)
    }
}

/// The sums of the weights, the sums of the squares of the weights, and the counts of the values in each bin.
type BinSums = (Vec<f64>, Vec<f64>, Vec<u64>);

/// Return the weight sums, the squared-weight sums, and the value counts of each bin of each group.
///
/// Each vector has the order [group][bin]. If `squares` is false, the vector of the squared-weight sums is empty.
/// Then no part of the parallel sum allocates a third array. The threads sum one wave of parts at a time, thus the
/// memory holds the sums of one part for each thread and not for each part. The totals add the parts in the order of
/// the values.
fn sum_bins<T: Copy + Into<f64> + Sync>(
    values: &[T],
    weights: &[f64],
    groups: Option<&[i32]>,
    edges: &BinEdges,
    ngroups: usize,
    squares: bool,
) -> BinSums {
    let nbins = edges.nbins();
    let size = ngroups * nbins;
    let partsize = get_partsize(values.len(), size);
    let sum_part = |start: usize, chunk: &[T]| -> BinSums {
        let mut sums = vec![0.0; size];
        let mut sumsquares = vec![0.0; if squares { size } else { 0 }];
        let mut counts = vec![0_u64; size];
        for (offset, &value) in chunk.iter().enumerate() {
            let row = start + offset;
            if let Some(bin) = edges.index(value.into()) {
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "the caller checked that each group is in range"
                )]
                let group = groups.map_or(0, |groups| groups[row] as usize);
                let index = group * nbins + bin;
                sums[index] += weights[row];
                if squares {
                    sumsquares[index] += weights[row] * weights[row];
                }
                counts[index] += 1;
            }
        }
        (sums, sumsquares, counts)
    };

    let mut sums = vec![0.0; size];
    let mut sumsquares = vec![0.0; if squares { size } else { 0 }];
    let mut counts = vec![0_u64; size];
    let wavesize = partsize * rayon::current_num_threads();
    for (waveindex, wave) in values.chunks(wavesize).enumerate() {
        let parts: Vec<BinSums> = wave
            .par_chunks(partsize)
            .enumerate()
            .map(|(part, chunk)| sum_part(waveindex * wavesize + part * partsize, chunk))
            .collect();
        for (partsums, partsumsquares, partcounts) in &parts {
            for (total, value) in sums.iter_mut().zip(partsums) {
                *total += value;
            }
            for (total, value) in sumsquares.iter_mut().zip(partsumsquares) {
                *total += value;
            }
            for (total, value) in counts.iter_mut().zip(partcounts) {
                *total += value;
            }
        }
    }
    (sums, sumsquares, counts)
}

fn check_edges(edges: &[f64]) -> PolarsResult<()> {
    if edges.len() < 2 || edges.windows(2).any(|pair| pair[0] >= pair[1]) {
        polars_bail!(ComputeError: "the bin edges must be two or more values in increasing order");
    }
    Ok(())
}

/// Return the bin of each value, or -1 for a value outside the edges or NaN.
fn get_bins<T: Copy + Into<f64> + Sync>(values: &[T], edges: &BinEdges) -> Vec<i32> {
    values
        .par_iter()
        .map(|&value| {
            edges
                .index(value.into())
                .and_then(|bin| i32::try_from(bin).ok())
                .unwrap_or(-1)
        })
        .collect()
}

/// Return the index of the bin of each value in the column "binindex", or -1 for a value outside the edges, NaN, or null.
///
/// The bins are the bins of `sum_weights_in_bins`. A caller can then group the rows by the bin and by other columns,
/// e.g. by the emission type of each packet. The count of emission types is too large for the groups of
/// `sum_weights_in_bins`.
#[pyfunction]
#[expect(clippy::needless_pass_by_value)]
pub fn get_bin_indices(
    py: Python<'_>,
    df: PyDataFrame,
    valuecolumn: &str,
    edges: Vec<f64>,
) -> PyResult<PyDataFrame> {
    let dfbins = py
        .detach(|| {
            let mut df = df.0;
            df.rechunk_mut_par();
            check_edges(&edges)?;
            let binedges = BinEdges::new(&edges);
            let values = df.column(valuecolumn)?.as_materialized_series();
            // a NaN value is in no bin, thus a null value takes NaN. A column with no null needs no copy
            let bins = match values.dtype() {
                DataType::Float32 => get_bins(
                    values
                        .f32()?
                        .fill_null_with_values(f32::NAN)?
                        .cont_slice()?,
                    &binedges,
                ),
                _ => get_bins(
                    values
                        .f64()?
                        .fill_null_with_values(f64::NAN)?
                        .cont_slice()?,
                    &binedges,
                ),
            };
            df!("binindex" => bins)
        })
        .map_err(PyPolarsErr::from)?;

    Ok(PyDataFrame(dfbins))
}

/// Return the weight sum, the squared-weight sum, and the value count of each bin.
///
/// The columns are "sum", "count", and, if `sumsquares` is true, "sumsquares". For a spectrum, the weight is the packet
/// energy and each bin is a wavelength bin. sqrt(sumsquares) / sum is the relative Monte Carlo noise of a bin. The
/// squares need more time and memory, and most callers do not read them. Thus only `sumsquares=True` adds the column.
///
/// `edges` gives the lower edge of each bin and the upper edge of the last bin, in increasing order. A bin is
/// [lower, upper), and the last bin also holds its upper edge. A value outside the edges, or a NaN value, is in no
/// bin. A row with a null value, a null weight, or a null group is also in no bin, e.g. the last row of a cut file.
/// The value column can be Float32 or Float64, and the weight column is Float64.
///
/// With `groupcolumn`, each group from 0 to `ngroups` - 1 has its own bins, e.g. the direction bins. The rows are then
/// in the order [group][bin]. The sums have the same value for each count of threads.
///
/// The sum runs without the global interpreter lock (GIL), thus other Python threads can run at the same time.
#[pyfunction]
#[pyo3(signature = (df, valuecolumn, weightcolumn, edges, groupcolumn=None, ngroups=1, sumsquares=false))]
#[expect(clippy::needless_pass_by_value)]
#[expect(
    clippy::too_many_arguments,
    reason = "Python gives each keyword argument as one parameter of the function"
)]
pub fn sum_weights_in_bins(
    py: Python<'_>,
    df: PyDataFrame,
    valuecolumn: &str,
    weightcolumn: &str,
    edges: Vec<f64>,
    groupcolumn: Option<&str>,
    ngroups: usize,
    sumsquares: bool,
) -> PyResult<PyDataFrame> {
    let dfsums = py
        .detach(|| {
            let usedcolumns: Vec<&str> = [valuecolumn, weightcolumn]
                .into_iter()
                .chain(groupcolumn)
                .collect();
            // a frame with no null needs no copy
            let mut df = df.0.drop_nulls(Some(&usedcolumns))?;
            df.rechunk_mut_par();
            check_edges(&edges)?;
            if ngroups == 0 {
                polars_bail!(ComputeError: "the count of groups must be one or more");
            }
            let binedges = BinEdges::new(&edges);
            let weights = df
                .column(weightcolumn)?
                .as_materialized_series()
                .f64()?
                .cont_slice()?;
            let groups = match groupcolumn {
                Some(name) => {
                    let groups = df
                        .column(name)?
                        .as_materialized_series()
                        .i32()?
                        .cont_slice()?;
                    if groups
                        .iter()
                        .any(|&group| usize::try_from(group).map_or(true, |group| group >= ngroups))
                    {
                        polars_bail!(ComputeError: "a group is outside 0 to {} - 1", ngroups);
                    }
                    Some(groups)
                }
                None => None,
            };
            let values = df.column(valuecolumn)?.as_materialized_series();
            let (sums, squaresums, counts) = match values.dtype() {
                DataType::Float32 => sum_bins(
                    values.f32()?.cont_slice()?,
                    weights,
                    groups,
                    &binedges,
                    ngroups,
                    sumsquares,
                ),
                _ => sum_bins(
                    values.f64()?.cont_slice()?,
                    weights,
                    groups,
                    &binedges,
                    ngroups,
                    sumsquares,
                ),
            };
            if sumsquares {
                df!("sum" => sums, "sumsquares" => squaresums, "count" => counts)
            } else {
                df!("sum" => sums, "count" => counts)
            }
        })
        .map_err(PyPolarsErr::from)?;

    Ok(PyDataFrame(dfsums))
}
