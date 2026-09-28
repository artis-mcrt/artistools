"""Script for computing binned expansion opacities and Planck-mean opacities in postprocessing."""

import argparse
import math
import re
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
from artistools.misc import get_artis_source_text
from artistools.misc import get_model_logname
from artistools.misc import get_npts_model
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_single_timestep
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_time
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import print_warning
from artistools.rustext import sum_binned_line_opacities

HCLIGHTOVERFOURPI = h_erg_s * C_cm_per_s / 4 / math.pi

# exopac is the expansion opacity. linebinned is the sum of tau_sobolev, and linebinned_maxone
# limits each tau_sobolev to 1
OPACITYCOLUMNS = ("exopac", "linebinned", "linebinned_maxone")

# the estimator columns that can give the excitation temperature T_exc of the LTE level populations
EXCITATIONTEMPERATURE_NAMES = {"TJ": "the temperature of the mean intensity J", "Te": "the electron temperature"}

# sum_binned_line_opacities() gives each group of 32 cells to a thread, thus a batch of 4096 cells gives each
# core work. A batch has one row for each cell and bin, and 4096 cells of 1200 bins took 0.8 GB. 4096 cells of
# the 4998 bins of ejectaopacity took 3.6 GB, thus a batch of more bins holds fewer cells
CELLSPERBATCH = 4096
ROWSPERBATCH = CELLSPERBATCH * 1200


class OpacityLines(t.NamedTuple):
    """The levels and the binned lines of the ions that the estimators hold, for all the cells of a run."""

    ionstrs: list[str]
    """The ion of each ionindex in dflevels."""

    dflevels: pl.DataFrame
    """The levels of all the ions, with the ionindex, the statistical weight, and the energy of each level."""

    dflines: pl.DataFrame
    """The lines in the wavelength range, in the order of the upper level and then the lower level.

    lower and upper give the row of each level in dflevels. The Sobolev optical depth of a line is
    pop_lower * sobolev_lower - pop_upper * sobolev_upper.
    """


def get_expopac_grid(modelpath: Path | str) -> tuple[float, float, float] | None:
    """Return the minimum, the maximum, and the width of the expansion opacity bins of ARTIS in Angstroms.

    The values come from rpkt.h in the artis folder of the run, which holds the source code of the run. A run with
    no such file, or a file with no expopac constants, gives None.
    """
    rpkttext = get_artis_source_text(modelpath, "rpkt.h")
    if rpkttext is None:
        return None
    values = dict(re.findall(r"expopac_(lambdamin|lambdamax|deltalambda)\s*=\s*([-+0-9.eE]+)", rpkttext))
    if set(values) != {"lambdamin", "lambdamax", "deltalambda"}:
        return None
    return float(values["lambdamin"]), float(values["lambdamax"]), float(values["deltalambda"])


def get_lambda_bin_edges(lambdamin: float, lambdamax: float, deltalambda: float) -> list[float]:
    """Return the edges of the wavelength bins in Angstroms.

    The last bin ends at lambdamax. If the range does not hold a whole number of bins, the last bin ends
    above lambdamax, thus the bins cover the full range.
    """
    if lambdamax <= lambdamin or deltalambda <= 0.0:
        msg = (
            f"The wavelength range {lambdamin:g} to {lambdamax:g} Angstroms holds no bin of width {deltalambda:g}"
            " Angstroms"
        )
        raise ValueError(msg)

    # int() of 999.9999999999999 is 999, thus the tolerance keeps a range of whole bins whole
    numbins = math.ceil((lambdamax - lambdamin) / deltalambda - 1e-9)
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

        levelframes.append(
            dflevels.select(
                pl.lit(len(ionstrs), dtype=pl.UInt32).alias("ionindex"), pl.col("g").cast(pl.Float64), "energy_ev"
            )
        )
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
                .cast(pl.UInt32)
                .alias("lambda_angstroms_binindex"),
                "lambda_angstroms",
                lower=(pl.col("lower") + levelcount).cast(pl.UInt32),
                upper=(pl.col("upper") + levelcount).cast(pl.UInt32),
                sobolev_lower=pl.col("B_lu") * sobolevfactor,
                sobolev_upper=pl.col("B_ul") * sobolevfactor,
            )
            # a null A value or a null statistical weight gives no optical depth, and the kernel reads no nulls
            .drop_nulls(["sobolev_lower", "sobolev_upper"])
        )
        ionstrs.append(ionstr)
        levelcount += dflevels.height

    if not ionstrs:
        msg = "The estimators hold no ion population column for any ion of the atomic data"
        raise ValueError(msg)

    return OpacityLines(
        ionstrs=ionstrs,
        dflevels=pl.concat(levelframes),
        # in this order, the kernel reads nearby rows of the level populations. For 40.7 million lines, this took
        # 20 % less time than the order of the bins
        dflines=pl.concat(lineframes).sort("upper", "lower").collect(),
    )


def get_expansion_opacities(
    opacitylines: OpacityLines, dfcells: pl.DataFrame, lambda_bin_edges: list[float], time_days: float
) -> pl.DataFrame:
    """Return the binned expansion opacity and the line-binned opacities of each cell.

    The Rust function sum_binned_line_opacities() calculates the LTE level populations and sums the lines of
    each bin. A query in polars took 12 times longer, because each operation writes a full column.
    """
    numbins = len(lambda_bin_edges) - 1
    deltalambda = lambda_bin_edges[1] - lambda_bin_edges[0]
    time_s = time_days * day_to_s
    nnioncolumns = [f"nnion_{ionstr}" for ionstr in opacitylines.ionstrs]

    dfsums = sum_binned_line_opacities(
        opacitylines.dflevels,
        opacitylines.dflines,
        dfcells.select(
            # the kernel skips an ion with no population, thus it does not use the temperature 0 of such a cell
            pl.col("T_exc").cast(pl.Float64).fill_null(0.0),
            # a null population or a null temperature gives no opacity
            *(
                pl
                .when(pl.col("T_exc").is_not_null())
                .then(pl.col(column).cast(pl.Float64))
                .fill_null(0.0)
                .alias(column)
                for column in nnioncolumns
            ),
        ),
        nnioncolumns,
        numbins,
        K_B_ev_per_K,
    )

    return (
        dfcells
        .select("modelgridindex", "T_exc", "mass_g", "rho")
        .join(pl.DataFrame({"lambda_angstroms_binindex": range(numbins)}), how="cross", maintain_order="left_right")
        .with_columns(
            lambda_angstroms_bin_mid=lambda_bin_edges[0]
            + pl.col("lambda_angstroms_binindex") * deltalambda
            + deltalambda / 2,
            **{
                column: dfsums[column] / deltalambda / (C_cm_per_s * time_s * pl.col("rho"))
                for column in OPACITYCOLUMNS
            },
        )
        .select(
            "modelgridindex",
            "lambda_angstroms_binindex",
            "lambda_angstroms_bin_mid",
            "T_exc",
            "mass_g",
            *OPACITYCOLUMNS,
        )
    )


def get_planck_mean_opacities(dfbinnedopacities: pl.DataFrame) -> pl.DataFrame:
    """Return the Planck mean of the expansion opacity of each cell, with the mass of the cell.

    The Planck function at the excitation temperature T_exc of the cell gives the weight of each bin. A cell with no
    T_exc has no Planck function, thus it has no row.
    """
    return (
        dfbinnedopacities
        .lazy()
        .filter(pl.col("T_exc") > 0.0)
        .with_columns(lambda_cm_bin_mid=pl.col("lambda_angstroms_bin_mid") * 1e-8)
        .with_columns(
            planckfactor=(
                (pl.col("lambda_cm_bin_mid").pow(-5))
                / ((h_erg_s * C_cm_per_s / pl.col("lambda_cm_bin_mid") / pl.col("T_exc") / K_B_erg_per_K).exp() - 1)
            )
        )
        .group_by("modelgridindex", "mass_g")
        .agg(planckmean_opacity=((pl.col("planckfactor") * pl.col("exopac")).sum() / pl.col("planckfactor").sum()))
        .sort("modelgridindex")
        .collect(engine="streaming")
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


def get_artis_excitation_uses_tj(modelpath: Path | str) -> bool | None:
    """Return LTEPOP_EXCITATION_USE_TJ of artis/artisoptions.h in the folder of the run.

    A run with no such file, or a file with no such option, gives None.
    """
    optionstext = get_artis_source_text(modelpath, "artisoptions.h")
    if optionstext is None:
        return None
    match = re.search(
        r"^\s*constexpr\s+bool\s+LTEPOP_EXCITATION_USE_TJ\s*=\s*(true|false)\s*;", optionstext, flags=re.MULTILINE
    )
    return None if match is None else match.group(1) == "true"


def get_excitation_temperature_column(modelpath: Path | str, selection: str) -> str:
    """Return the estimator column that gives T_exc, and log the column and the reason for it.

    A selection of "auto" takes the temperature that ARTIS used for the LTE level populations of the run.
    """
    if selection != "auto":
        column, reason = selection, f"-exctemperature {selection}"
    elif (usestj := get_artis_excitation_uses_tj(modelpath)) is None:
        column, reason = "Te", "the default"
        print_warning(
            "artis/artisoptions.h of the run gives no LTEPOP_EXCITATION_USE_TJ, thus the level populations take Te."
            " Give -exctemperature TJ or -exctemperature Te to select the temperature"
        )
    else:
        column = "TJ" if usestj else "Te"
        reason = f"ARTIS used it, because artis/artisoptions.h sets LTEPOP_EXCITATION_USE_TJ = {str(usestj).lower()}"

    print_detail(
        f"The level populations are LTE populations at T_exc = {column}, {EXCITATIONTEMPERATURE_NAMES[column]} of"
        f" each cell ({reason})"
    )
    return column


def print_planck_mean_method(xmin: float, xmax: float) -> None:
    """Log the wavelength range and the weights of the Planck mean of each cell and of the cells."""
    print_detail(
        f"The Planck mean of each cell weights the bins from {xmin:g} to {xmax:g} Angstroms with the Planck function"
        " at T_exc. The mean of the cells takes the mass of each cell as its weight"
    )


def get_cell_estimators(
    modelpath: Path | str, timestep: int, modelgridindex: int | None, temperaturecolumn: str
) -> pl.DataFrame:
    """Return the estimators, the mass, and the mid-point velocity of each cell at the timestep.

    The column T_exc holds the values of temperaturecolumn.
    """
    dfestimators = (
        scan_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex, join_modeldata=True)
        .select(
            "modelgridindex",
            "timestep",
            pl.col(temperaturecolumn).alias("T_exc"),
            "rho",
            "mass_g",
            "vel_r_mid_on_c",
            cs.starts_with("nnion_"),
        )
        .collect()
    )
    # ARTIS writes no estimators for a cell that holds no matter
    if dfestimators.is_empty():
        cellstr = "any cell" if modelgridindex is None else f"cell {modelgridindex}"
        msg = f"The estimators hold no values for {cellstr} at timestep {timestep}. An empty cell has no estimators"
        if modelgridindex is not None:
            msg += f". {get_next_cell_with_estimators_text(modelpath, timestep, modelgridindex)}"
        raise ValueError(msg)

    return dfestimators


def get_next_cell_with_estimators_text(modelpath: Path | str, timestep: int, modelgridindex: int) -> str:
    """Return a sentence that names the first cell after modelgridindex that has estimators at the timestep.

    A scan of a range of cells reads only the batches of the ranks that update them. Thus the search starts with a
    small range, and it doubles the range until it finds a cell. A first use of a large run then converts few batches.
    """
    npts_model = get_npts_model(modelpath)
    firstcell, cellcount = modelgridindex + 1, 64
    while firstcell < npts_model:
        lastcell = min(firstcell + cellcount, npts_model) - 1
        nextcell = (
            scan_estimators(modelpath, timestep=timestep, modelgridindex=range(firstcell, lastcell + 1))
            .select(pl.col("modelgridindex").min())
            .collect()
            .item()
        )
        if nextcell is not None:
            return f"The next cell with estimators is cell {nextcell}"
        firstcell, cellcount = lastcell + 1, 2 * cellcount

    return f"No cell after cell {modelgridindex} has estimators"


def get_opacity_atomic_data(modelpath: Path | str) -> pl.DataFrame:
    """Return the levels and the transitions of each ion, with the columns that the opacities need."""
    # get_opacity_lines() needs the statistical weights as well as the wavelength, and
    # add_transition_columns() drops each derived column that this call does not request
    return get_levels(
        modelpath, get_transitions=True, derived_transitions_columns=["lambda_angstroms", "lower_g", "upper_g"]
    )


def get_cell_batches(dfestimators: pl.DataFrame, numbins: int) -> list[pl.DataFrame]:
    """Split the cells into batches for get_expansion_opacities().

    A batch holds a maximum of CELLSPERBATCH cells and ROWSPERBATCH pairs of a cell and a bin. It holds whole groups of
    32 cells.
    """
    cellsperbatch = max(32, min(CELLSPERBATCH, ROWSPERBATCH // numbins) // 32 * 32)
    return [dfestimators.slice(firstcell, cellsperbatch) for firstcell in range(0, dfestimators.height, cellsperbatch)]


def addarg_excitationtemperature(parser: argparse.ArgumentParser) -> None:
    """Add the -exctemperature argument, which selects the temperature of the LTE level populations."""
    parser.add_argument(
        "-exctemperature",
        choices=("auto", *EXCITATIONTEMPERATURE_NAMES),
        default="auto",
        help=(
            "Estimator temperature of the LTE level populations: "
            + " or ".join(f"{column}, {name}" for column, name in EXCITATIONTEMPERATURE_NAMES.items())
            + ". auto takes the temperature that ARTIS used, from LTEPOP_EXCITATION_USE_TJ in artis/artisoptions.h of"
            " the run, or Te if the run has no such option"
        ),
    )


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
    addarg_excitationtemperature(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Compute binned expansion opacities and Planck-mean opacities in postprocessing."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    time_days = get_timestep_time(args.modelpath, timestep)
    print(f"Opacities of {get_model_logname(args.modelpath)} at {time_days:.2f}d (timestep {timestep})")
    print_modelpath(args.modelpath)
    temperaturecolumn = get_excitation_temperature_column(args.modelpath, args.exctemperature)
    print_planck_mean_method(args.xmin, args.xmax)
    dfestimators = get_cell_estimators(
        args.modelpath, timestep, get_single_modelgridindex(args.modelgridindex), temperaturecolumn
    )

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
    for dfcellbatch in get_cell_batches(dfestimators, len(lambda_bin_edges) - 1):
        dfbinnedopacities = get_expansion_opacities(opacitylines, dfcellbatch, lambda_bin_edges, time_days)
        if args.show_binned_opacities:
            print(dfbinnedopacities)

        dfplanckmean = get_planck_mean_opacities(dfbinnedopacities)

        print(dfplanckmean)
        planckmeanopacity_times_mass += (dfplanckmean.select(pl.col("planckmean_opacity").dot(pl.col("mass_g")))).item()
        mass_g_sum += dfplanckmean.select(pl.col("mass_g").sum()).item()

        cells_processed += dfcellbatch.select(pl.len()).item()
        elapsed = time.perf_counter() - time_start
        timepercell = elapsed / cells_processed
        print(
            f" average seconds per cell: {timepercell:.3f}. cells remaining: {cellcount - cells_processed}."
            f" time remaining: {timepercell * (cellcount - cells_processed):.1f}s"
        )

    print()
    globalplanckmeanopacity = planckmeanopacity_times_mass / mass_g_sum
    print(f"Global Planck mean opacity: {globalplanckmeanopacity:.2f} cm^2/g")
