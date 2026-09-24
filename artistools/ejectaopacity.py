"""Script for computing binned expansion opacities and Planck-mean opacities in postprocessing."""

import argparse
import math
import time
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import polars.selectors as cs

from artistools.atomic import get_ionstring
from artistools.atomic import get_levels
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import h_erg_s
from artistools.constants import K_B_erg_per_K
from artistools.constants import K_B_ev_per_K
from artistools.estimators import scan_estimators
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import exit_with_error
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_single_timestep
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_time
from artistools.misc import parse_cli_args

HCLIGHTOVERFOURPI = h_erg_s * C_cm_per_s / 4 / math.pi

# exopac is the expansion opacity. linebinned is the sum of tau_sobolev, and linebinned_maxone
# limits each tau_sobolev to 1
OPACITYCOLUMNS = ("exopac", "linebinned", "linebinned_maxone")

# for one block and one cell, a temporary column uses 0.8 MB, thus it stays in the processor cache
LINESPERBLOCK = 100_000

# one query calculates each opacity of each cell of a batch as a separate column, thus a large batch gives
# each thread work. For 40.7 million lines, a cell took 0.16 s in a batch of 32 cells and 0.08 s in 256 or 512
CELLSPERBATCH = 256


class OpacityLines(t.NamedTuple):
    """The levels and the binned lines of the ions that the estimators hold, for all the cells of a run."""

    ionstrs: list[str]
    """The ion of each ionindex in dflevels."""

    dflevels: pl.DataFrame
    """The levels of all the ions, with the ionindex, the statistical weight, and the energy of each level."""

    dflines: pl.DataFrame
    """The lines in the wavelength range, in the order of the bins.

    lower and upper give the row of each level in dflevels. The Sobolev optical depth of a line is
    pop_lower * sobolev_lower - pop_upper * sobolev_upper.
    """


def get_lambda_bin_edges(lambdamin: float, lambdamax: float, deltalambda: float) -> list[float]:
    """Return the edges of the wavelength bins in Angstroms."""
    numbins = int((lambdamax - lambdamin) / deltalambda)
    return [lambdamin + i * deltalambda for i in range(numbins + 1)]


def get_opacity_lines(
    adata: pl.DataFrame, estimatorcolumns: Sequence[str], lambda_bin_edges: list[float], time_days: float
) -> OpacityLines:
    """Return the levels and the lines of each ion that the estimators give a population for.

    The lines do not depend on a cell, thus a run prepares them one time for all the cells.
    """
    sobolevfactor = HCLIGHTOVERFOURPI * time_days * day_to_s
    ionstrs: list[str] = []
    levelframes: list[pl.DataFrame] = []
    lineframes: list[pl.LazyFrame] = []
    levelcount = 0
    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        ionstr = get_ionstring(Z, ion_stage, sep="_")
        if f"nnion_{ionstr}" not in estimatorcolumns:
            continue

        # a line reads its level populations by the position of the level
        if not dflevels.select((pl.col("levelindex") == pl.int_range(pl.len())).all()).item():
            msg = f"The levels of {ionstr} do not have the level indices 0 to {dflevels.height - 1} in order"
            raise ValueError(msg)

        levelframes.append(dflevels.select(pl.lit(len(ionstrs), dtype=pl.UInt32).alias("ionindex"), "g", "energy_ev"))
        lineframes.append(
            dftransitions
            .lazy()
            .filter(
                pl.col("lambda_angstroms").is_between(lambda_bin_edges[0], lambda_bin_edges[-1]),
                pl.col("lower").is_between(0, dflevels.height - 1),
                pl.col("upper").is_between(0, dflevels.height - 1),
            )
            .with_columns(nu_trans=1e8 * C_cm_per_s / (pl.col("lambda_angstroms")))
            .with_columns(B_ul=C_cm_per_s**2 / 2 / h_erg_s / pl.col("nu_trans").pow(3) * pl.col("A"))
            .with_columns(B_lu=pl.col("upper_g") / pl.col("lower_g") * pl.col("B_ul"))
            .select(
                # give cut only the interior edges. A line at an outer edge then falls into the first or
                # the last bin, not into an out-of-range category with the index -1
                pl
                .col("lambda_angstroms")
                .cut(breaks=lambda_bin_edges[1:-1])
                .to_physical()
                .cast(pl.Int32)
                .alias("lambda_angstroms_binindex"),
                "lambda_angstroms",
                lower=(pl.col("lower") + levelcount).cast(pl.UInt32),
                upper=(pl.col("upper") + levelcount).cast(pl.UInt32),
                sobolev_lower=pl.col("B_lu") * sobolevfactor,
                sobolev_upper=pl.col("B_ul") * sobolevfactor,
            )
        )
        ionstrs.append(ionstr)
        levelcount += dflevels.height

    if not ionstrs:
        msg = "The estimators hold no ion population column for any ion of the atomic data"
        raise ValueError(msg)

    return OpacityLines(
        ionstrs=ionstrs,
        dflevels=pl.concat(levelframes),
        dflines=pl.concat(lineframes).sort("lambda_angstroms_binindex").collect(),
    )


def get_level_pops(opacitylines: OpacityLines, dfcells: pl.DataFrame) -> pl.DataFrame:
    """Return the population of each level in local thermodynamic equilibrium (LTE), with one column for each cell."""
    dfionpops = (
        dfcells
        .select(f"nnion_{ionstr}" for ionstr in opacitylines.ionstrs)
        .transpose(column_names=[str(cellpos) for cellpos in range(dfcells.height)])
        .with_columns(ionindex=pl.int_range(pl.len(), dtype=pl.UInt32))
    )

    def boltzmannfactor(te: float | None) -> pl.Expr:
        return pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / te).exp()

    return opacitylines.dflevels.join(dfionpops, on="ionindex", how="left", maintain_order="left").select(
        (boltzmannfactor(te) / boltzmannfactor(te).sum().over("ionindex") * pl.col(str(cellpos))).alias(str(cellpos))
        for cellpos, te in enumerate(dfcells["Te"])
    )


def get_expansion_opacities(
    opacitylines: OpacityLines, dfcells: pl.DataFrame, lambda_bin_edges: list[float], time_days: float
) -> pl.DataFrame:
    """Return the binned expansion opacity and the line-binned opacities of each cell.

    Each cell has one column of level populations, and each line reads them with a gather by position.
    Thus the query needs no join of the lines with the cells. The sum runs over blocks of lines, because
    the temporary columns of a small block stay in the processor cache.
    """
    numbins = len(lambda_bin_edges) - 1
    deltalambda = lambda_bin_edges[1] - lambda_bin_edges[0]
    time_s = time_days * day_to_s
    lambdaexpr = pl.col("lambda_angstroms")

    sumexprs: list[pl.Expr] = []
    for cellpos, levelpops in enumerate(get_level_pops(opacitylines, dfcells).iter_columns()):
        tau_sobolev = pl.lit(levelpops).gather(pl.col("lower")) * pl.col("sobolev_lower") - pl.lit(levelpops).gather(
            pl.col("upper")
        ) * pl.col("sobolev_upper")
        sumexprs += [
            ((1 - (-tau_sobolev).exp()) * lambdaexpr).alias(f"exopac_{cellpos}"),
            (tau_sobolev * lambdaexpr).alias(f"linebinned_{cellpos}"),
            (tau_sobolev.clip(upper_bound=1.0) * lambdaexpr).alias(f"linebinned_maxone_{cellpos}"),
        ]

    blocksums = [
        opacitylines.dflines
        .slice(firstline, LINESPERBLOCK)
        .lazy()
        .select("lambda_angstroms_binindex", *sumexprs)
        .group_by("lambda_angstroms_binindex")
        .agg(pl.all().sum())
        .collect()
        for firstline in range(0, opacitylines.dflines.height, LINESPERBLOCK)
    ]

    # a bin with no lines has a sum of zero
    dfbinsums = (
        pl
        .DataFrame({"lambda_angstroms_binindex": range(numbins)}, schema={"lambda_angstroms_binindex": pl.Int32})
        .join(
            pl.concat(blocksums).group_by("lambda_angstroms_binindex").agg(pl.all().sum()),
            on="lambda_angstroms_binindex",
            how="left",
            maintain_order="left",
        )
        .with_columns(pl.exclude("lambda_angstroms_binindex").fill_null(0.0))
    )

    # the columns of one quantity are [bin][cell], thus the flat order of the transpose is [cell][bin]
    opacitysums = {
        column: dfbinsums.select(f"{column}_{cellpos}" for cellpos in range(dfcells.height)).to_numpy().ravel(order="F")
        for column in OPACITYCOLUMNS
    }

    return (
        dfcells
        .select("modelgridindex", "Te", "mass_g", "rho")
        .join(dfbinsums.select("lambda_angstroms_binindex"), how="cross", maintain_order="left_right")
        .with_columns(
            lambda_angstroms_bin_mid=lambda_bin_edges[0]
            + pl.col("lambda_angstroms_binindex") * deltalambda
            + deltalambda / 2,
            **{
                column: pl.Series(values) / deltalambda / (C_cm_per_s * time_s * pl.col("rho"))
                for column, values in opacitysums.items()
            },
        )
        .select(
            "modelgridindex",
            pl.col("lambda_angstroms_binindex").cast(pl.Int64),
            "lambda_angstroms_bin_mid",
            "Te",
            "mass_g",
            *OPACITYCOLUMNS,
        )
    )


def get_selected_timestep(modelpath: Path | str, timestep: str | int | None, timedays: str | None) -> int:
    """Return the timestep that -timestep or -timedays gives, or exit with an error."""
    if timedays is not None:
        if timestep is not None:
            exit_with_error("specify only one of -timestep and -timedays")
        return get_timestep_of_timedays(modelpath, timedays)

    selectedtimestep = get_single_timestep(timestep, modelpath)
    if selectedtimestep is None:
        exit_with_error("no time was given", "Give a time or a timestep, e.g. -timedays 250 or -timestep 30")

    return selectedtimestep


def get_cell_estimators(modelpath: Path | str, timestep: int, modelgridindex: int | None) -> pl.DataFrame:
    """Return the estimators and the mass of each cell at the timestep."""
    dfestimators = (
        scan_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex, join_modeldata=True)
        .select("modelgridindex", "timestep", "Te", "rho", "mass_g", cs.starts_with("nnion_"))
        .collect()
    )
    # ARTIS writes no estimators for a cell that holds no matter
    if dfestimators.is_empty():
        cellstr = "any cell" if modelgridindex is None else f"cell {modelgridindex}"
        msg = f"The estimators hold no values for {cellstr} at timestep {timestep}. An empty cell has no estimators"
        raise ValueError(msg)

    return dfestimators


def get_opacity_atomic_data(modelpath: Path | str) -> pl.DataFrame:
    """Return the levels and the transitions of each ion, with the columns that the opacities need."""
    # get_opacity_lines() needs the statistical weights as well as the wavelength, and
    # add_transition_columns() drops each derived column that this call does not request
    return get_levels(
        modelpath, get_transitions=True, derived_transitions_columns=["lambda_angstroms", "lower_g", "upper_g"]
    )


def get_cell_batches(dfestimators: pl.DataFrame) -> list[pl.DataFrame]:
    """Split the cells into batches of CELLSPERBATCH cells for get_expansion_opacities()."""
    return [dfestimators.slice(firstcell, CELLSPERBATCH) for firstcell in range(0, dfestimators.height, CELLSPERBATCH)]


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_timestep(parser, helptext="Timestep number to select")
    addarg_timedays(parser, kind="str", helptext="Time in days to select")
    addarg_modelpath(parser, default=Path(), helptext="Path of ARTIS model")
    parser.add_argument(
        "--show_binned_opacities",
        action="store_true",
        help="Show the binned opacities for each cell (can be very large)",
    )
    addarg_modelgridindex(parser, helptext="Model grid cell to select. If not specified, all cells are processed")

    # every command that reads a range of wavelengths takes both spellings, thus -xmin reaches this
    # command as well. This one bins the opacities over that range, and it draws no plot
    parser.add_argument(
        "-xmin", "-lambdamin", type=float, default=20.0, help="Minimum wavelength in Angstroms for binned opacities"
    )
    parser.add_argument(
        "-xmax", "-lambdamax", type=float, default=50000.0, help="Maximum wavelength in Angstroms for binned opacities"
    )
    parser.add_argument(
        "-deltalambda", type=float, default=10.0, help="Wavelength bin width in Angstroms for binned opacities"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Compute binned expansion opacities and Planck-mean opacities in postprocessing."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    dfestimators = get_cell_estimators(args.modelpath, timestep, get_single_modelgridindex(args.modelgridindex))

    time_days = get_timestep_time(args.modelpath, timestep)

    print()
    print(f"timestep {timestep} T_days = {time_days:.2f}")

    lambda_bin_edges = get_lambda_bin_edges(args.xmin, args.xmax, args.deltalambda)
    opacitylines = get_opacity_lines(
        get_opacity_atomic_data(args.modelpath), dfestimators.columns, lambda_bin_edges, time_days
    )

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(5000)
    cellcount = dfestimators.select(pl.len()).item()
    cells_processed = 0
    time_start = time.perf_counter()
    planckmeanopacity_times_mass = 0.0
    mass_g_sum = 0.0
    for dfcellbatch in get_cell_batches(dfestimators):
        dfbinnedopacities = get_expansion_opacities(opacitylines, dfcellbatch, lambda_bin_edges, time_days)
        if args.show_binned_opacities:
            print(dfbinnedopacities)

        dfplanckmean = (
            (
                dfbinnedopacities
                .lazy()
                .with_columns(lambda_cm_bin_mid=pl.col("lambda_angstroms_bin_mid") * 1e-8)
                .with_columns(
                    planckfactor=(
                        (pl.col("lambda_cm_bin_mid").pow(-5))
                        / (
                            (h_erg_s * C_cm_per_s / pl.col("lambda_cm_bin_mid") / pl.col("Te") / K_B_erg_per_K).exp()
                            - 1
                        )
                    )
                )
                .group_by("modelgridindex", "mass_g")
                .agg(
                    planckmean_opacity=(
                        (pl.col("planckfactor") * pl.col("exopac")).sum() / pl.col("planckfactor").sum()
                    )
                )
            )
            .sort("modelgridindex")
            .collect(engine="streaming")
        )

        print(dfplanckmean)
        planckmeanopacity_times_mass += (dfplanckmean.select(pl.col("planckmean_opacity").dot(pl.col("mass_g")))).item()
        mass_g_sum += dfplanckmean.select(pl.col("mass_g").sum()).item()

        cells_processed += dfcellbatch.select(pl.len()).item()
        elapsed = time.perf_counter() - time_start
        timepercell = elapsed / cells_processed
        print(
            f" average seconds per cell: {timepercell:.3f}. cells remaining: {cellcount - cells_processed}. time remaining: {timepercell * (cellcount - cells_processed):.1f}s"
        )

    print()
    globalplanckmeanopacity = planckmeanopacity_times_mass / mass_g_sum
    print(f"Global Planck mean opacity: {globalplanckmeanopacity:.2f} cm^2/g")
