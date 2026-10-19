"""Give the Sobolev optical depth and the emission of each line in a wavelength range, in each cell of a model."""

import argparse
import math
import typing as t
from collections.abc import Collection
from collections.abc import Iterator
from collections.abc import Sequence
from itertools import starmap
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from artistools.atomic import get_ion_tuple
from artistools.atomic import get_ionstring
from artistools.constants import angstroms_per_cm
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import h_erg_s
from artistools.constants import K_B_ev_per_K
from artistools.constants import km_to_cm
from artistools.ejectaopacity import addarg_excitationtemperature
from artistools.ejectaopacity import get_excitation_temperature_column
from artistools.ejectaopacity import get_opacity_atomic_data
from artistools.ejectaopacity import get_selected_timestep
from artistools.estimators import scan_estimators
from artistools.inputmodel import get_cell_selection
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import get_model_logname
from artistools.misc import get_timestep_time
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_weighted_quantile_expr
from artistools.misc import parse_cli_args
from artistools.misc import path_is_file
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import print_product
from artistools.misc import print_saved
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.misc.cliutils import CommaJoinAction
from artistools.misc.cliutils import get_provenance_lines
from artistools.misc.cliutils import write_data_table
from artistools.misc.columns import get_column_description_lines
from artistools.misc.dirbins import get_costheta_bins
from artistools.nltepops import has_nlte_populations
from artistools.nltepops import read_nltepops
from artistools.plottools import get_velocity_bin_index_expr

DEFAULTOUTPUTNAME = "linecontributions.txt"

# the columns that name one line, i.e. one transition of one ion
LINEKEYS: t.Final = ("ion", "lambda_angstroms", "lower", "upper", "A_per_s")

# a batch of cells makes one row for each pair of a cell and a line, and one value for each pair of a cell and a
# level. This count keeps a batch near 1 GB for the columns of the cell table
ROWSPERBATCH = 4_000_000

# for a small tau, (1 - exp(-tau)) / tau loses accuracy, thus a small tau takes the series 1 - tau / 2
SMALLTAU = 1e-6

# an estimator file gives each nnion_ value with about 5 digits. Thus a population of the nlte_*.out files can be above
# the nnion_ value of its ion by about 1e-4 of that value
NNION_RTOL: t.Final = 1e-3

NLTEPOPS_SCHEMA = pl.Schema({
    "modelgridindex": pl.Int64,
    "Z": pl.Int64,
    "ion_stage": pl.Int64,
    "level": pl.Int64,
    "n_NLTE": pl.Float64,
})

ESCAPE_PROBABILITY_REFERENCE = "Castor J. I., 1970, MNRAS, 149, 111-127, doi:10.1093/mnras/149.2.111"


def get_ion_levels_and_lines(
    modelpath: Path | str, lambdamin: float, lambdamax: float, ions: Collection[tuple[int, int]]
) -> tuple[dict[tuple[int, int], pl.DataFrame], pl.DataFrame]:
    """Return the kept levels of each ion, and the lines of the ions in the wavelength range.

    The lines take only the kept levels of ARTIS, and a line with no A value has no optical depth and no emission.
    """
    levelsofion: dict[tuple[int, int], pl.DataFrame] = {}
    lineframes: list[pl.DataFrame] = []
    adata = get_opacity_atomic_data(modelpath)
    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        if (Z, ion_stage) not in ions:
            continue
        levels = dflevels.select(
            level=pl.col("levelindex").cast(pl.Int64),
            g=pl.col("g").cast(pl.Float64),
            energy_ev=pl.col("energy_ev").cast(pl.Float64),
            levelname=pl.col("levelname"),
        )
        if not levels.select((pl.col("level") == pl.int_range(pl.len())).all()).item():
            msg = f"The levels of {get_ionstring(Z, ion_stage)} do not have the level indices 0 to {levels.height - 1}"
            raise ValueError(msg)
        levelsofion[Z, ion_stage] = levels
        lineframes.append(
            dftransitions
            .lazy()
            .filter(
                pl.col("lambda_angstroms").is_between(lambdamin, lambdamax),
                pl.col("lower").is_between(0, levels.height - 1),
                pl.col("upper").is_between(0, levels.height - 1),
                pl.col("A") > 0.0,
            )
            .select(
                pl.lit(Z, dtype=pl.Int64).alias("Z"),
                pl.lit(ion_stage, dtype=pl.Int64).alias("ion_stage"),
                pl.lit(get_ionstring(Z, ion_stage)).alias("ion"),
                pl.col("lambda_angstroms").cast(pl.Float64),
                pl.col("lower").cast(pl.Int64),
                pl.col("upper").cast(pl.Int64),
                pl.col("A").cast(pl.Float64).alias("A_per_s"),
                pl.col("lower_g").cast(pl.Float64),
                pl.col("upper_g").cast(pl.Float64),
            )
            .collect()
        )

    dflines = pl.concat(lineframes) if lineframes else pl.DataFrame()
    return levelsofion, dflines


def get_boltzmann_factors(
    levels: pl.DataFrame, temperatures: "np.ndarray[t.Any, np.dtype[np.float64]]", referenceenergy_ev: float
) -> "np.ndarray[t.Any, np.dtype[np.float64]]":
    """Return g exp(-(E - E_ref) / k T) of each cell (row) and each level (column).

    The energies start at referenceenergy_ev, thus a high level at a low temperature does not give zero.
    """
    energies = levels["energy_ev"].to_numpy() - referenceenergy_ev
    with np.errstate(divide="ignore", invalid="ignore"):
        return levels["g"].to_numpy()[np.newaxis, :] * np.exp(
            -energies[np.newaxis, :] / K_B_ev_per_K / temperatures[:, np.newaxis]
        )


def get_ion_populations(
    dfcells: pl.DataFrame, levels: pl.DataFrame, neededlevels: Sequence[int], dfnltepops: pl.DataFrame, nnioncolumn: str
) -> pl.DataFrame:
    """Return the population of each needed level of one ion in each cell, as calculate_levelpop() of ARTIS gives it.

    A cell with no NLTE row of the ion takes the LTE populations of the nnion_ estimator at T_exc. ARTIS writes no
    NLTE row of an element that has no NLTE solution in a cell, e.g. an element that fell back to LTE. A cell with
    NLTE rows takes:
    - the NLTE population of the ground level and of each NLTE level;
    - for a level in the superlevel, the superlevel population times g exp(-E / k T_exc) of the level, divided by
      the sum of these factors over the superlevel;
    - for a level above the NLTE levels of an ion with no superlevel, the ground-level population times
      g / g_0 exp(-(E - E_0) / k T_exc).
    """
    neededframe = levels.filter(pl.col("level").is_in(neededlevels))
    isnltecell = pl.col("modelgridindex").is_in(dfnltepops["modelgridindex"].unique().implode())
    framesofcells = []
    if not (dfltecells := dfcells.filter(~isnltecell)).is_empty():
        framesofcells.append(get_lte_ion_populations(dfltecells, levels, neededframe, nnioncolumn))
    if not (dfnltecells := dfcells.filter(isnltecell)).is_empty():
        framesofcells.append(get_nlte_ion_populations(dfnltecells, levels, neededframe, dfnltepops))
    return pl.concat(framesofcells)


def get_lte_ion_populations(
    dfcells: pl.DataFrame, levels: pl.DataFrame, neededframe: pl.DataFrame, nnioncolumn: str
) -> pl.DataFrame:
    """Return the LTE population of each level of neededframe in each cell, from the nnion_ estimator at T_exc.

    A model with no nnion_ estimator of the ion gives a null population.
    """
    temperatures = dfcells["T_exc"].cast(pl.Float64).to_numpy()
    nnion = (
        dfcells[nnioncolumn].cast(pl.Float64).fill_null(0.0).to_numpy()
        if nnioncolumn in dfcells.columns
        else np.full(dfcells.height, np.nan)
    )
    partitionfunctions = get_boltzmann_factors(levels, temperatures, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pops = (
            nnion[:, np.newaxis]
            * get_boltzmann_factors(neededframe, temperatures, 0.0)
            / partitionfunctions[:, np.newaxis]
        )
    return (
        dfcells
        .select("modelgridindex")
        .join(neededframe.select("level"), how="cross", maintain_order="left_right")
        .with_columns(
            population=pl.Series(pops.reshape(-1), dtype=pl.Float64).fill_nan(None), population_source=pl.lit("LTE")
        )
    )


def get_nlte_ion_populations(
    dfcells: pl.DataFrame, levels: pl.DataFrame, neededframe: pl.DataFrame, dfnltepops: pl.DataFrame
) -> pl.DataFrame:
    """Return the population of each level of neededframe in each cell from the NLTE rows, see get_ion_populations."""
    temperatures = dfcells["T_exc"].cast(pl.Float64).to_numpy()
    cellcolumn = dfcells.select("modelgridindex")
    lastnltelevel = dfnltepops.select(pl.col("level").max()).item()
    assert isinstance(lastnltelevel, int)
    dfresolved = (
        dfnltepops
        .filter(pl.col("level") >= 0)
        .select("modelgridindex", "level", population="n_NLTE")
        .with_columns(population_source=pl.lit("NLTE"))
    )
    framesoflevels = [
        cellcolumn.join(
            neededframe.filter(pl.col("level") <= lastnltelevel).select("level"),
            how="cross",
            maintain_order="left_right",
        ).join(dfresolved, on=["modelgridindex", "level"], how="left", maintain_order="left")
    ]

    upperneeded = neededframe.filter(pl.col("level") > lastnltelevel)
    if not upperneeded.is_empty():
        dfsuperlevel = dfnltepops.filter(pl.col("level") < 0).select("modelgridindex", superlevelpop="n_NLTE")
        if not dfsuperlevel.is_empty():
            superlevels = levels.filter(pl.col("level") > lastnltelevel)
            referenceenergy = float(superlevels["energy_ev"][0])
            slpartition = get_boltzmann_factors(superlevels, temperatures, referenceenergy).sum(axis=1)
            superlevelpop = (
                cellcolumn
                .join(dfsuperlevel, on="modelgridindex", how="left", maintain_order="left")["superlevelpop"]
                .cast(pl.Float64)
                .to_numpy()
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                pops = (
                    superlevelpop[:, np.newaxis]
                    * get_boltzmann_factors(upperneeded, temperatures, referenceenergy)
                    / slpartition[:, np.newaxis]
                )
            source = "superlevel"
        else:
            groundpop = (
                cellcolumn
                .join(
                    dfresolved.filter(pl.col("level") == 0).select("modelgridindex", groundpop="population"),
                    on="modelgridindex",
                    how="left",
                    maintain_order="left",
                )["groundpop"]
                .cast(pl.Float64)
                .to_numpy()
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                pops = (
                    groundpop[:, np.newaxis]
                    * get_boltzmann_factors(upperneeded, temperatures, float(levels["energy_ev"][0]))
                    / float(levels["g"][0])
                )
            source = "Boltzmann from the ground level"
        framesoflevels.append(
            cellcolumn.join(upperneeded.select("level"), how="cross", maintain_order="left_right").with_columns(
                population=pl.Series(pops.reshape(-1), dtype=pl.Float64), population_source=pl.lit(source)
            )
        )

    return pl.concat(framesoflevels)


def get_cell_line_table(
    dfcells: pl.DataFrame,
    dflines: pl.DataFrame,
    levelsofion: dict[tuple[int, int], pl.DataFrame],
    dfnltepops: pl.DataFrame,
    time_days: float,
) -> pl.DataFrame:
    """Return one row for each cell and line, with the level populations, tau_sobolev, and the emission.

    tau_sobolev = (A lambda^3 t / 8 pi) (g_u n_l / g_l - n_u). The emission is n_u A h nu beta, with the escape
    probability beta = (1 - exp(-tau)) / tau.
    """
    time_s = time_days * day_to_s
    popframes = []
    for (Z, ion_stage), dfionlines in dflines.group_by("Z", "ion_stage", maintain_order=True):
        assert isinstance(Z, int)
        assert isinstance(ion_stage, int)
        neededlevels = sorted({*dfionlines["lower"].to_list(), *dfionlines["upper"].to_list()})
        dfionpops = dfnltepops.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))
        popframes.append(
            get_ion_populations(
                dfcells,
                levelsofion[Z, ion_stage],
                neededlevels,
                dfionpops.filter(pl.col("modelgridindex").is_in(dfcells["modelgridindex"].implode())),
                f"nnion_{get_ionstring(Z, ion_stage, sep='_')}",
            ).with_columns(pl.lit(Z, dtype=pl.Int64).alias("Z"), pl.lit(ion_stage, dtype=pl.Int64).alias("ion_stage"))
        )
    dfpops = pl.concat(popframes)

    tau = pl.col("tau_sobolev")
    return (
        dfcells
        .select("modelgridindex", "velocity_kmps", "volume", "Te", "nne", "T_exc")
        .join(dflines, how="cross", maintain_order="left_right")
        .join(
            dfpops.rename({"level": "lower", "population": "n_lower_per_cm3", "population_source": "n_lower_source"}),
            on=["modelgridindex", "Z", "ion_stage", "lower"],
            how="left",
            maintain_order="left",
        )
        .join(
            dfpops.rename({"level": "upper", "population": "n_upper_per_cm3", "population_source": "n_upper_source"}),
            on=["modelgridindex", "Z", "ion_stage", "upper"],
            how="left",
            maintain_order="left",
        )
        .with_columns(
            tau_sobolev=pl.col("A_per_s")
            * (pl.col("lambda_angstroms") / angstroms_per_cm).pow(3)
            * time_s
            / (8 * math.pi)
            * (pl.col("upper_g") / pl.col("lower_g") * pl.col("n_lower_per_cm3") - pl.col("n_upper_per_cm3"))
        )
        .with_columns(
            escape_probability=pl.when(tau.abs() < SMALLTAU).then(1.0 - tau / 2.0).otherwise((1.0 - (-tau).exp()) / tau)
        )
        .with_columns(
            line_emissivity_erg_per_s_per_cm3=pl.col("n_upper_per_cm3")
            * pl.col("A_per_s")
            * h_erg_s
            * C_cm_per_s
            * angstroms_per_cm
            / pl.col("lambda_angstroms")
            * pl.col("escape_probability")
        )
        .with_columns(emission_erg_per_s=pl.col("line_emissivity_erg_per_s_per_cm3") * pl.col("volume"))
        .select(
            "modelgridindex",
            "velocity_kmps",
            "volume",
            "Te",
            "nne",
            "T_exc",
            "ion",
            "lambda_angstroms",
            "lower",
            "upper",
            "A_per_s",
            "n_lower_per_cm3",
            "n_lower_source",
            "n_upper_per_cm3",
            "n_upper_source",
            "tau_sobolev",
            "escape_probability",
            "line_emissivity_erg_per_s_per_cm3",
            "emission_erg_per_s",
        )
    )


def get_line_summary(dfcelllines: pl.DataFrame) -> pl.DataFrame:
    """Return one row for each line, with the sums and the maxima over the cells of one batch.

    get_line_summary_of_batches() combines the batches, then it divides the sums by the emission to give the means.
    """
    emission = pl.col("emission_erg_per_s").fill_nan(None).fill_null(0.0)
    return dfcelllines.group_by(LINEKEYS, maintain_order=True).agg(
        # an ion can take NLTE populations in one cell and LTE populations in a different cell
        pl.col("n_lower_source").unique().sort(),
        pl.col("n_upper_source").unique().sort(),
        emission_erg_per_s=emission.sum(),
        emission_times_velocity=(emission * pl.col("velocity_kmps")).sum(),
        emission_times_Te=(emission * pl.col("Te")).sum(),
        tau_sobolev_max=pl.col("tau_sobolev").max(),
        velocity_kmps_at_tau_sobolev_max=pl.col("velocity_kmps").sort_by("tau_sobolev").last(),
        cellcount_tau_sobolev_above_1=(pl.col("tau_sobolev") > 1.0).sum(),
    )


def get_line_summary_of_batches(dfsummaries: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Combine the line summaries of the batches of cells, and give the fraction of the emission of each line.

    A line with more than one method of its level populations, e.g. NLTE in one cell and LTE in a different cell,
    gives the methods with a "/" between them.
    """
    emission = pl.col("emission_erg_per_s")
    return (
        pl
        .concat(dfsummaries)
        .group_by(LINEKEYS, maintain_order=True)
        .agg(
            *(
                pl.col(column).explode().unique().sort().str.join("/")
                for column in ("n_lower_source", "n_upper_source")
            ),
            emission.sum(),
            pl.col("emission_times_velocity").sum(),
            pl.col("emission_times_Te").sum(),
            pl.col("velocity_kmps_at_tau_sobolev_max").sort_by("tau_sobolev_max").last(),
            pl.col("tau_sobolev_max").max(),
            pl.col("cellcount_tau_sobolev_above_1").sum(),
        )
        .with_columns(
            emission_fraction=emission / emission.sum(),
            velocity_kmps_emission_mean=pl.when(emission > 0).then(pl.col("emission_times_velocity") / emission),
            Te_emission_mean=pl.when(emission > 0).then(pl.col("emission_times_Te") / emission),
        )
        .sort("emission_erg_per_s", "lambda_angstroms", descending=[True, False])
        .select(
            *LINEKEYS,
            "emission_erg_per_s",
            "emission_fraction",
            "velocity_kmps_emission_mean",
            "Te_emission_mean",
            "tau_sobolev_max",
            "velocity_kmps_at_tau_sobolev_max",
            "cellcount_tau_sobolev_above_1",
            "n_lower_source",
            "n_upper_source",
        )
    )


def parse_ions(iontexts: Sequence[str], optionname: str) -> list[tuple[int, int]]:
    """Return the (Z, ion_stage) of each ion name, e.g. "Fe II", or exit with an error for an element."""
    ions: list[tuple[int, int]] = []
    for iontext in iontexts:
        iontuple = get_ion_tuple(iontext.strip())
        if not isinstance(iontuple, tuple):
            exit_with_error(
                f"{optionname} takes ions, and '{iontext}' names an element", f'Give an ion, e.g. {optionname} "Fe II"'
            )
        ions.append(iontuple)
    return ions


class CellLineInputs(t.NamedTuple):
    """The data that get_cell_line_batches() reads for each batch of cells."""

    dfcells: pl.DataFrame
    dflines: pl.DataFrame
    levelsofion: dict[tuple[int, int], pl.DataFrame]
    dfnltepops: pl.DataFrame
    time_days: float


def get_cell_line_inputs(
    modelpath: Path | str,
    timestep: int,
    lambdamin: float,
    lambdamax: float,
    ions: Collection[tuple[int, int]] | None = None,
    exctemperature: str = "auto",
    costhetarange: tuple[float, float] | None = None,
    taumin: float | None = None,
) -> CellLineInputs:
    """Read the data of the cells and of the lines in the wavelength range for get_cell_line_batches().

    costhetarange selects the cells with cos θ = v_z / |v| of the cell centre in the range, edges included. The z axis
    is the polar axis of the direction bins. taumin leaves out each line whose tau_sobolev cannot reach it in a cell,
    see get_lines_that_can_reach_tau(). The function logs the method and the assumptions.
    """
    time_days = get_timestep_time(modelpath, timestep)
    temperaturecolumn = get_excitation_temperature_column(
        modelpath, exctemperature, usetext="The LTE populations and the Boltzmann factors of the superlevels take"
    )
    lfestimators = scan_estimators(modelpath, timestep=timestep, join_modeldata=True)
    estimatorcolumns = lfestimators.collect_schema().names()
    selectedions = (
        list(ions)
        if ions is not None
        else parse_ions(
            [
                column.removeprefix("nnion_").replace("_", " ")
                for column in estimatorcolumns
                if column.startswith("nnion_")
            ],
            "nnion_",
        )
    )

    # a model with no NLTE files has no NLTE row for any ion, thus each ion takes its LTE populations
    dfnltepops = pl.DataFrame(schema=NLTEPOPS_SCHEMA)
    if has_nlte_populations(modelpath):
        dfnltepops = read_nltepops(modelpath, timestep=timestep)
        if dfnltepops.is_empty():
            exit_with_error(
                f"the model has NLTE populations, but the NLTE files hold no row for timestep {timestep}",
                "ARTIS writes the NLTE populations only for some timesteps. Give a different time",
            )
        dfnltepops = (
            dfnltepops
            .filter(pl.col("timestep") == timestep)
            .select(NLTEPOPS_SCHEMA.names())
            .unique(subset=["modelgridindex", "Z", "ion_stage", "level"], keep="first", maintain_order=True)
        )
    if path_is_file(Path(modelpath, "autoion.txt")):
        print_warning(
            "The model has autoion.txt. ARTIS does not include the autoionising levels in the Boltzmann sum of the"
            " superlevel, but this command includes them. Thus the population of a level in the superlevel can be too"
            " small"
        )

    nlteions = set(dfnltepops.select("Z", "ion_stage").unique().iter_rows())
    usableions = [
        ion
        for ion in selectedions
        if ion in nlteions or f"nnion_{get_ionstring(ion[0], ion[1], sep='_')}" in estimatorcolumns
    ]
    if missingions := [ion for ion in selectedions if ion not in usableions]:
        print_warning(
            "The model gives no population for "
            + ", ".join(starmap(get_ionstring, missingions))
            + ", thus the table leaves out the lines of these ions"
        )

    levelsofion, dflines = get_ion_levels_and_lines(modelpath, lambdamin, lambdamax, usableions)
    if dflines.is_empty():
        exit_with_error(
            f"the atomic data of the ions holds no line with an A value from {lambdamin:g} to {lambdamax:g} Å",
            "Give a different wavelength range, or different ions with -ions",
        )

    if costhetarange is not None:
        if "vel_z_mid" not in estimatorcolumns:
            exit_with_error(
                "a 1D model has no direction of a cell, thus -costhetamin and -costhetamax cannot select cells",
                "Remove -costhetamin and -costhetamax",
            )
        # a cell on a bound stays in the selection, although the Float32 error of the velocities can move it across
        # the bound. The cell at the centre of the grid has no direction, thus the selection leaves it out
        lfestimators = lfestimators.filter(
            get_cell_selection(
                thetamin=math.degrees(math.acos(costhetarange[1])), thetamax=math.degrees(math.acos(costhetarange[0]))
            )
        )
        print_detail(
            f"The cells have cos θ = v_z / |v| of the cell centre from {costhetarange[0]:g} to {costhetarange[1]:g}."
            " The z axis is the polar axis of the direction bins"
        )

    dfcells = (
        lfestimators
        .select(
            "modelgridindex",
            "Te",
            "nne",
            "volume",
            pl.col(temperaturecolumn).alias("T_exc"),
            pl.col("^nnion_.*$"),
            # the trough estimate needs the velocity vector of the cell centre. Only a 3D model has its three parts
            cs.by_name("vel_x_mid", "vel_y_mid", "vel_z_mid", require_all=False),
            velocity_kmps=pl.col("vel_r_mid_on_c") * C_cm_per_s / km_to_cm,
        )
        .sort("modelgridindex")
        .collect()
    )
    # the bound of -taumin is zero with no cell, thus this check comes first
    if dfcells.is_empty():
        if costhetarange is not None:
            exit_with_error(
                "no cell with matter has its centre in the range of cos θ",
                "Give a wider range of -costhetamin and -costhetamax",
            )
        exit_with_error(f"the estimators hold no cell for timestep {timestep}", "Give a different time")

    print_detail(
        f"{dflines.height} lines of {dflines.select('Z', 'ion_stage').n_unique()} ions from {lambdamin:g} to"
        f" {lambdamax:g} Å. The wavelengths are vacuum wavelengths from the level energies"
    )
    if taumin is not None:
        dflines = get_lines_that_can_reach_tau(dflines, dfcells, levelsofion, nlteions, taumin, time_days)
        if dflines.is_empty():
            exit_with_error(
                f"no line can reach a tau_sobolev of {taumin:g} in a cell, thus -taumin leaves out each line",
                "Give a smaller -taumin, or a different wavelength range",
            )
    print_detail(f"{dfcells.height} cells at the mid time {time_days:.2f} d of timestep {timestep}")
    print_detail(
        "tau_sobolev = (A lambda^3 t / 8 pi) (g_u n_l / g_l - n_u) at the mid time t of the timestep. A is the Einstein"
        " coefficient of the spontaneous emission, g is the statistical weight, and n is the population of the lower"
        " (l) or upper (u) level"
    )
    print_detail(
        "The emission of a line in a cell is n_u A h nu beta times the volume of the cell. The escape probability is"
        f" beta = (1 - exp(-tau)) / tau ({ESCAPE_PROBABILITY_REFERENCE})"
    )
    print_detail(
        "The emission is the energy that leaves the cell in the line each second. It is not the flux that escapes the"
        " ejecta, and it does not subtract the absorption of the radiation field by the line"
    )
    print_detail(
        "The level populations follow calculate_levelpop() of ARTIS. The columns n_lower_source and n_upper_source"
        " name the method of each level:\n"
        "    - NLTE: the population from the nlte_*.out files;\n"
        "    - superlevel: the superlevel population times g exp(-E / k T_exc), divided by the sum of these"
        " factors over the superlevel;\n"
        "    - Boltzmann from the ground level: the ground-level population times g / g_0 exp(-(E - E_0) / k T_exc);\n"
        "    - LTE: the population of the nnion_ estimator at T_exc."
    )

    return CellLineInputs(dfcells, dflines, levelsofion, dfnltepops, time_days)


def get_lte_population_bounds(
    nnion: "np.ndarray[t.Any, np.dtype[np.float64]]",
    temperatures: "np.ndarray[t.Any, np.dtype[np.float64]]",
    energies_ev: "np.ndarray[t.Any, np.dtype[np.float64]]",
) -> "np.ndarray[t.Any, np.dtype[np.float64]]":
    """Return max over the cells of n_ion exp(-E / k T) for each energy E above the ground level.

    A cell that another cell beats in both n_ion and T cannot give the maximum for any E of zero or more. Thus only the
    cells of the Pareto set go into the matrix, and that set is small.
    """
    isusable = (nnion > 0.0) & (temperatures > 0.0)
    lognnion = np.log(nnion[isusable])
    inversekt = 1.0 / (K_B_ev_per_K * temperatures[isusable])
    if lognnion.size == 0:
        return np.zeros_like(energies_ev)
    # the hottest cell first: a cell stays only if it is denser than each hotter cell
    order = np.argsort(inversekt, kind="stable")
    sortedlog = lognnion[order]
    previousmax = np.concatenate(([-np.inf], np.maximum.accumulate(sortedlog)[:-1]))
    ispareto = sortedlog > previousmax
    paretolog, paretoinversekt = sortedlog[ispareto], inversekt[order][ispareto]
    return np.exp((paretolog[:, np.newaxis] - paretoinversekt[:, np.newaxis] * energies_ev[np.newaxis, :]).max(axis=0))


def get_lines_that_can_reach_tau(
    dflines: pl.DataFrame,
    dfcells: pl.DataFrame,
    levelsofion: dict[tuple[int, int], pl.DataFrame],
    nlteions: Collection[tuple[int, int]],
    taumin: float,
    time_days: float,
) -> pl.DataFrame:
    """Return the lines whose tau_sobolev can reach taumin in a cell, and log the number of the other lines.

    tau_sobolev <= (A lambda^3 t / 8 pi) (g_u / g_l) n_l. For an ion with LTE populations in each cell,
    n_l = n_ion g_l exp(-(E_l - E_0) / k T_exc) / Z <= n_ion (g_l / g_0) exp(-(E_l - E_0) / k T_exc), because the
    partition function Z is at least g_0.

    For an ion with NLTE rows, an NLTE level and a level of the superlevel have n_l <= n_ion. A level above the NLTE
    levels of an ion with no superlevel takes n_0 (g_l / g_0) exp(-(E_l - E_0) / k T_exc), which can be above n_ion.
    Thus the bound takes n_l <= n_ion max(1, g_l / g_0), with a margin of NNION_RTOL. An ion with no nnion_ estimator
    has no bound, thus all its lines stay.
    """
    temperatures = dfcells["T_exc"].cast(pl.Float64).fill_null(0.0).to_numpy()
    lineframes = []
    for (Z, ion_stage), dfionlines in dflines.group_by("Z", "ion_stage", maintain_order=True):
        assert isinstance(Z, int)
        assert isinstance(ion_stage, int)
        nnioncolumn = f"nnion_{get_ionstring(Z, ion_stage, sep='_')}"
        if nnioncolumn not in dfcells.columns:
            lineframes.append(dfionlines)
            continue
        nnion = dfcells[nnioncolumn].cast(pl.Float64).fill_null(0.0).to_numpy()
        levels = levelsofion[Z, ion_stage]
        energies_ev = levels["energy_ev"].to_numpy()
        lowerlevels = dfionlines["lower"].to_numpy()
        if (Z, ion_stage) in nlteions:
            populationbound = (
                nnion.max(initial=0.0)
                * np.maximum(1.0, dfionlines["lower_g"].to_numpy() / float(levels["g"][0]))
                * (1.0 + NNION_RTOL)
            )
        else:
            excitation_ev = energies_ev[lowerlevels] - energies_ev[0]
            populationbound = (
                get_lte_population_bounds(nnion, temperatures, excitation_ev)
                * dfionlines["lower_g"].to_numpy()
                / float(levels["g"][0])
            )
        taubound = (
            dfionlines["A_per_s"].to_numpy()
            * (dfionlines["lambda_angstroms"].to_numpy() / angstroms_per_cm) ** 3
            * time_days
            * day_to_s
            / (8 * math.pi)
            * dfionlines["upper_g"].to_numpy()
            / dfionlines["lower_g"].to_numpy()
            * populationbound
        )
        lineframes.append(dfionlines.filter(pl.Series(taubound >= taumin)))
    dfkept = pl.concat(lineframes) if lineframes else dflines.clear()
    print_detail(
        f"-taumin {taumin:g} leaves out {dflines.height - dfkept.height} of {dflines.height} lines. A left-out line has"
        " an upper bound of tau_sobolev below -taumin in each cell. For an ion with LTE populations, the bound is"
        " (A lambda^3 t / 8 pi) (g_u / g_0) times the largest n_ion exp(-(E_l - E_0) / k T_exc) of the cells. For an"
        " ion with NLTE rows, it is (A lambda^3 t / 8 pi) (g_u / g_l) max(1, g_l / g_0) times the largest n_ion. An"
        " optically thin line can emit much, but the sums of the emission and the ranking by emission leave it out"
    )
    return dfkept


def get_cell_line_batches(inputs: CellLineInputs) -> Iterator[pl.DataFrame]:
    """Yield the table of get_cell_line_table() for each batch of cells.

    The ions with a line in the range set the size of a batch, because only these ions get populations.
    """
    ionswithlines = set(inputs.dflines.select("Z", "ion_stage").unique().iter_rows())
    maxlevelcount = max(levels.height for ion, levels in inputs.levelsofion.items() if ion in ionswithlines)
    cellsperbatch = max(1, ROWSPERBATCH // max(inputs.dflines.height, maxlevelcount))
    for firstcell in range(0, inputs.dfcells.height, cellsperbatch):
        yield get_cell_line_table(
            inputs.dfcells.slice(firstcell, cellsperbatch),
            inputs.dflines,
            inputs.levelsofion,
            inputs.dfnltepops,
            inputs.time_days,
        )


def warn_about_line_summary(dfsummary: pl.DataFrame) -> None:
    """Warn about the lines with a population inversion in each cell, and the lines with no level population."""
    if (inverted := dfsummary.filter(pl.col("tau_sobolev_max") < 0.0).height) > 0:
        print_warning(
            f"{inverted} lines have a negative tau_sobolev in each cell, i.e. a population inversion. The escape"
            " probability of such a line is above 1"
        )
    if (nopopulation := dfsummary.filter(pl.col("tau_sobolev_max").is_null()).height) > 0:
        print_warning(f"{nopopulation} lines have no level population in any cell, thus they have no tau_sobolev")


def get_line_contributions(
    modelpath: Path | str,
    timestep: int,
    lambdamin: float,
    lambdamax: float,
    ions: Collection[tuple[int, int]] | None = None,
    exctemperature: str = "auto",
    *,
    percell: bool = False,
    taumin: float | None = None,
) -> pl.DataFrame:
    """Return the emission of each line in the wavelength range, or with percell, one row for each cell and line.

    The method and the assumptions go to the log.
    """
    celltables = []
    summaries = []
    inputs = get_cell_line_inputs(modelpath, timestep, lambdamin, lambdamax, ions, exctemperature, taumin=taumin)
    for dfcelllines in get_cell_line_batches(inputs):
        summaries.append(get_line_summary(dfcelllines))
        if percell:
            celltables.append(dfcelllines)

    dfsummary = get_line_summary_of_batches(summaries)
    warn_about_line_summary(dfsummary)
    return pl.concat(celltables) if percell else dfsummary


def get_ranked_lines(dfsummary: pl.DataFrame, rankby: t.Literal["emission", "tau"], maxlines: int) -> pl.DataFrame:
    """Return the summary rows of the maxlines lines with the most emission or the largest tau_sobolev in a cell."""
    sortcolumn = "emission_erg_per_s" if rankby == "emission" else "tau_sobolev_max"
    return dfsummary.sort(sortcolumn, "lambda_angstroms", descending=[True, False], nulls_last=True).head(maxlines)


class LineVelocityProfiles(t.NamedTuple):
    """The summary of all the lines, the ranked lines, and the velocity profiles. See get_line_velocity_profiles()."""

    dfsummary: pl.DataFrame
    dfranked: pl.DataFrame
    dfalllines: pl.DataFrame
    dfprofiles: pl.DataFrame
    dfrankedcells: pl.DataFrame


def get_line_velocity_profiles(
    inputs: CellLineInputs, velocityedges_kmps: Sequence[float], rankby: t.Literal["emission", "tau"], maxlines: int
) -> LineVelocityProfiles:
    """Return the line summary, the ranked lines, and the velocity profiles of all the lines and of each ranked line.

    The bins have equal widths. The first pass over the cells gives the summary of all the lines and the emission of
    all the lines in each bin. The second pass gives one row for each cell and ranked line (dfrankedcells), because a
    median needs the values of all the cells. The profile of a ranked line gives in each bin:
    - the emission and the number of cells;
    - the mean and the median of tau_sobolev, with the volume of each cell as the weight;
    - the volume fraction of the cells with a tau_sobolev above 1.
    A cell with no tau_sobolev is transparent, as in the trough estimate, thus each value of the bin takes it as a
    tau_sobolev of 0. A cell outside the bins is in the summary, but not in a bin.
    """
    velocitybin = get_velocity_bin_index_expr(pl.col("velocity_kmps"), velocityedges_kmps)
    summaries = []
    alllinesbins = []
    for dfcelllines in get_cell_line_batches(inputs):
        summaries.append(get_line_summary(dfcelllines))
        alllinesbins.append(
            dfcelllines
            .lazy()
            .select("emission_erg_per_s", velocitybin=velocitybin)
            .filter(pl.col("velocitybin").is_not_null())
            .group_by("velocitybin")
            .agg(pl.col("emission_erg_per_s").fill_nan(None).fill_null(0.0).sum())
            .collect()
        )
    dfsummary = get_line_summary_of_batches(summaries)
    warn_about_line_summary(dfsummary)
    dfalllines = pl.concat(alllinesbins).group_by("velocitybin").agg(pl.col("emission_erg_per_s").sum())

    dfranked = get_ranked_lines(dfsummary, rankby, maxlines)
    rankedinputs = inputs._replace(
        dflines=inputs.dflines.join(dfranked.select(LINEKEYS), on=list(LINEKEYS), how="semi", maintain_order="left")
    )
    dfrankedcells = pl.concat(get_cell_line_batches(rankedinputs)).with_columns(velocitybin=velocitybin)

    binindices = list(range(len(velocityedges_kmps) - 1))
    binwidth_kmps = (velocityedges_kmps[-1] - velocityedges_kmps[0]) / len(binindices)
    tau = pl.col("tau_sobolev")
    dfprofiles = (
        dfrankedcells
        .filter(pl.col("velocitybin").is_not_null())
        .with_columns(tau.fill_null(0.0))
        .group_by(*LINEKEYS, "velocitybin")
        .agg(
            emission_erg_per_s=pl.col("emission_erg_per_s").fill_nan(None).fill_null(0.0).sum(),
            tau_sobolev_volume_mean=(tau * pl.col("volume")).sum() / pl.col("volume").sum(),
            tau_sobolev_volume_median=get_weighted_quantile_expr("tau_sobolev", "volume", 0.5),
            volume_fraction_tau_sobolev_above_1=pl.col("volume").filter(tau > 1.0).sum() / pl.col("volume").sum(),
            cellcount=pl.len(),
        )
        .with_columns(
            x_lower_kmps=pl.col("velocitybin").replace_strict(
                binindices, velocityedges_kmps[:-1], return_dtype=pl.Float64
            ),
            x_upper_kmps=pl.col("velocitybin").replace_strict(
                binindices, velocityedges_kmps[1:], return_dtype=pl.Float64
            ),
            emission_per_velocity_erg_per_s_per_kmps=pl.col("emission_erg_per_s") / binwidth_kmps,
        )
        .sort(*LINEKEYS, "velocitybin")
    )
    return LineVelocityProfiles(dfsummary, dfranked, dfalllines, dfprofiles, dfrankedcells)


def get_ranked_line_summary(profiles: LineVelocityProfiles) -> pl.DataFrame:
    """Return one row for each ranked line, with the velocities of its emission and of a median tau_sobolev above 1.

    The quartiles of the velocity take the emission of each cell as the weight. The range where the median
    tau_sobolev is above 1 starts at the lower edge of the first such bin. It ends at the upper edge of the last such
    bin. Thus the range can hold a bin with a median of 1 or less.
    """
    dfquartiles = profiles.dfrankedcells.group_by(LINEKEYS).agg(**{
        f"velocity_emission_q{round(quantile * 100)}_kmps": get_weighted_quantile_expr(
            "velocity_kmps", "emission_erg_per_s", quantile
        )
        for quantile in (0.25, 0.5, 0.75)
    })
    dftauranges = (
        profiles.dfprofiles
        .filter(pl.col("tau_sobolev_volume_median") > 1.0)
        .group_by(LINEKEYS)
        .agg(
            velocity_median_tau_above_1_min_kmps=pl.col("x_lower_kmps").min(),
            velocity_median_tau_above_1_max_kmps=pl.col("x_upper_kmps").max(),
        )
    )
    return (
        profiles.dfranked
        .select(
            *LINEKEYS,
            "emission_erg_per_s",
            "emission_fraction",
            "tau_sobolev_max",
            "velocity_kmps_at_tau_sobolev_max",
            "cellcount_tau_sobolev_above_1",
        )
        .join(dfquartiles, on=list(LINEKEYS), how="left", maintain_order="left")
        .join(dftauranges, on=list(LINEKEYS), how="left", maintain_order="left")
    )


# each cell of the trough estimate has this number of points along each axis. Thus a cell puts its volume in each bin
# of the line-of-sight velocity that it crosses. With bins as wide as a cell, three points gave the analytic depth of a
# shell within 0.01
TROUGHSUBPOINTS: t.Final = 3


def get_sobolev_trough(
    profiles: LineVelocityProfiles,
    inputs: CellLineInputs,
    dirbin: int,
    photospherevelocity_kmps: float,
    velocityedges_kmps: Sequence[float],
    cellwidth_kmps: float,
    *,
    relativisticdoppler: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the Sobolev absorption depth of each ranked line against the line-of-sight velocity, and of their blend.

    The observer is at the centre cos θ of the direction bin, and the depth is the mean over the centres of the phi
    bins. The photosphere is a sphere at v_ph = photospherevelocity_kmps. A part of a cell absorbs if it is on the near
    side of the photosphere, inside the cylinder of the photospheric disk. The depth at a line-of-sight velocity u is
    sum((1 - exp(-tau)) V_v) / (du pi v_ph^2). V_v is the volume in velocity space, and du is the width of the bin.
    Thus the depth is the fraction of the flux of a uniform disk that the line removes.

    The bins of u have the width of the bins of velocityedges_kmps. They continue past its last edge up to the
    fastest part of a cell, thus a low -velocitymax keeps all the absorption. Each cell is a cube of cellwidth_kmps,
    with TROUGHSUBPOINTS^3 points at the centres of its parts. Each point takes its own line-of-sight velocity and its
    own part of the volume, thus a cell puts its volume in each bin that it crosses.

    The estimate has no emission of the line and no limb darkening. The blend is 1 - prod(1 - depth) of the lines at
    each wavelength. The depth of a line goes linearly between the wavelengths of the bin centres. The first bin and
    the last bin keep their depth from their centre to their outer edge.

    The wavelength of a velocity u comes from the Doppler factor of ARTIS: 1 - u / c, or with relativisticdoppler,
    sqrt((1 - u / c) / (1 + u / c)). The relativistic factor of ARTIS also has the transverse velocity v_t, which a
    bin does not hold. Thus the blend is too blue by a fraction of about v_t^2 / (2 c^2). A bin with its centre at or
    above c gives no wavelength.
    """
    nphibins = get_viewingdirection_phibincount()
    costhetalower, costhetaupper, _ = get_costheta_bins(usedegrees=False)
    costhetabin = dirbin // nphibins
    costheta = (costhetalower[costhetabin] + costhetaupper[costhetabin]) / 2.0
    sintheta = math.sqrt(1.0 - costheta**2)

    vph_kmps = photospherevelocity_kmps
    time_s = inputs.time_days * day_to_s
    dfcells = inputs.dfcells.select("modelgridindex", "vel_x_mid", "vel_y_mid", "vel_z_mid")
    centres_kmps = dfcells.select(pl.exclude("modelgridindex")).to_numpy() / km_to_cm
    # the offsets of the points of a cell from its centre, in units of the cell width
    subpointfractions = (np.arange(TROUGHSUBPOINTS) + 0.5) / TROUGHSUBPOINTS - 0.5
    offsets_kmps = cellwidth_kmps * np.array(np.meshgrid(*[subpointfractions] * 3, indexing="ij")).reshape(3, -1).T

    binwidth_kmps = (velocityedges_kmps[-1] - velocityedges_kmps[0]) / (len(velocityedges_kmps) - 1)
    maxspeed_kmps = float(np.linalg.norm(centres_kmps, axis=1).max() + np.linalg.norm(offsets_kmps, axis=1).max())
    extrabincount = max(0, math.ceil((maxspeed_kmps - velocityedges_kmps[-1]) / binwidth_kmps))
    edges_kmps = np.append(velocityedges_kmps, velocityedges_kmps[-1] + binwidth_kmps * np.arange(1, extrabincount + 1))
    nbins = len(edges_kmps) - 1

    # a negative tau_sobolev is a population inversion, and a null one has no population. The estimate takes such a
    # line as transparent
    dflinecells = profiles.dfrankedcells.select(
        *LINEKEYS,
        "modelgridindex",
        "volume",
        absorbed=1.0 - (-pl.col("tau_sobolev").fill_null(0.0).clip(lower_bound=0.0)).exp(),
    )
    # the volume in velocity space [(km/s)^3] times the absorbed fraction, of each point of each cell and line
    lineweights = []
    for linekeys in profiles.dfranked.select(LINEKEYS).iter_rows():
        islinekey = pl.all_horizontal(pl.col(key) == value for key, value in zip(LINEKEYS, linekeys, strict=True))
        dfline = dfcells.select("modelgridindex").join(
            dflinecells.filter(islinekey), on="modelgridindex", how="left", maintain_order="left"
        )
        cellweights = (dfline["absorbed"].fill_null(0.0) * dfline["volume"].fill_null(0.0)).to_numpy()
        lineweights.append(cellweights / (time_s * km_to_cm) ** 3 / offsets_kmps.shape[0])

    depths = np.zeros((profiles.dfranked.height, nbins))
    for phiindex in range(nphibins):
        phi = 2.0 * math.pi * (phiindex + 0.5) / nphibins
        direction = np.array([sintheta * math.cos(phi), sintheta * math.sin(phi), costheta])
        # one point of each cell at a time keeps each array at one value for each cell
        for offset_kmps in offsets_kmps:
            points_kmps = centres_kmps + offset_kmps
            vlos_kmps = points_kmps @ direction
            impactsquared = (points_kmps**2).sum(axis=1) - vlos_kmps**2
            isinfront = (impactsquared < vph_kmps**2) & (
                vlos_kmps > np.sqrt(np.clip(vph_kmps**2 - impactsquared, 0.0, None))
            )
            # np.histogram sorted the velocities again for each line. These bins are the same: a bin holds its lower
            # edge, and the last bin also holds its upper edge
            binindices = np.searchsorted(edges_kmps, vlos_kmps, side="right") - 1
            binindices[vlos_kmps == edges_kmps[-1]] = nbins - 1
            iscounted = isinfront & (binindices >= 0) & (binindices < nbins)
            for lineindex, weights in enumerate(lineweights):
                depths[lineindex] += np.bincount(binindices[iscounted], weights=weights[iscounted], minlength=nbins)
    depths /= binwidth_kmps * math.pi * vph_kmps**2 * nphibins

    dfdepths = (
        profiles.dfranked
        .select(LINEKEYS)
        .join(
            pl.DataFrame({"x_lower_kmps": edges_kmps[:-1], "x_upper_kmps": edges_kmps[1:]}),
            how="cross",
            maintain_order="left_right",
        )
        .with_columns(trough_depth=pl.Series(depths.reshape(-1)))
        .select(*LINEKEYS, "trough_depth", "x_lower_kmps", "x_upper_kmps")
    )

    edges_on_c = edges_kmps * km_to_cm / C_cm_per_s
    bincentres_on_c = (edges_on_c[:-1] + edges_on_c[1:]) / 2.0
    # the corner of a fast grid can be above c, and a cell with matter cannot be there
    belowcbincount = int(np.count_nonzero(bincentres_on_c < 1.0))
    outeredge_on_c = edges_on_c[belowcbincount : belowcbincount + 1]
    outeredge_on_c = outeredge_on_c[outeredge_on_c < 1.0]
    betas = np.concatenate((edges_on_c[:1], bincentres_on_c[:belowcbincount], outeredge_on_c))
    dopplerfactors = np.sqrt((1.0 - betas) / (1.0 + betas)) if relativisticdoppler else 1.0 - betas
    lambdarests_angstroms = profiles.dfranked["lambda_angstroms"].to_numpy()
    lambdagrid_angstroms = np.linspace(
        lambdarests_angstroms.min() * dopplerfactors.min(), lambdarests_angstroms.max(), 400
    )
    transmission = np.ones_like(lambdagrid_angstroms)
    for lambdarest_angstroms, linedepths in zip(lambdarests_angstroms, depths, strict=True):
        belowcdepths = linedepths[:belowcbincount]
        sampledepths = np.concatenate((belowcdepths[:1], belowcdepths, belowcdepths[-1:][: len(outeredge_on_c)]))
        # np.interp needs increasing x, and a larger velocity gives a shorter wavelength
        transmission *= 1.0 - np.clip(
            np.interp(
                lambdagrid_angstroms,
                lambdarest_angstroms * dopplerfactors[::-1],
                sampledepths[::-1],
                left=0.0,
                right=0.0,
            ),
            0.0,
            1.0,
        )
    dfblend = pl.DataFrame({"lambda_angstroms": lambdagrid_angstroms, "trough_depth": 1.0 - transmission})
    return dfdepths, dfblend


def addarg_taumin(parser: argparse.ArgumentParser) -> None:
    """Add -taumin, which leaves out the lines that cannot reach a given tau_sobolev in any cell."""
    parser.add_argument(
        "-taumin",
        type=float,
        help=(
            "Leave out each line whose upper bound of tau_sobolev in the cells is below this value. Use -taumin for a"
            " wavelength range with very many lines. An optically thin line can emit much, but the sums of the"
            " emission and the ranking by emission then leave it out. Default: no limit"
        ),
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path(), helptext="Path of ARTIS model")
    addarg_timestep(parser, helptext="Timestep number to select")
    addarg_timedays(parser, kind="str", helptext="Time in days to select")
    parser.add_argument("-xmin", "-lambdamin", type=float, help="Minimum wavelength of the lines in Angstroms")
    parser.add_argument("-xmax", "-lambdamax", type=float, help="Maximum wavelength of the lines in Angstroms")
    parser.add_argument(
        "-ions",
        action=CommaJoinAction,
        help='Ions of the lines, e.g. "Fe II,Co II". Default: each ion with a population in the model',
    )
    parser.add_argument(
        "-maxlines", type=int, default=20, help="Number of lines in the table, with the most emission first"
    )
    addarg_taumin(parser)
    parser.add_argument(
        "--percell",
        action="store_true",
        help="Give one row for each cell and line, with the populations, tau_sobolev, and the emission",
    )
    addarg_excitationtemperature(parser)
    addarg_output(parser, kind="file", helptext="Path/filename for the output text file")
    addarg_force(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Give the Sobolev optical depth and the emission of each line in a wavelength range."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    if args.xmin is None or args.xmax is None:
        exit_with_error("give the wavelength range of the lines", "e.g. -lambdamin 8400 -lambdamax 8700")

    outputfile = resolve_outputfile(args.outputfile, DEFAULTOUTPUTNAME) if args.outputfile else None
    if outputfile is not None:
        confirm_overwrite([outputfile.parent], [outputfile.name], force=args.force)

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    print(f"Line contributions of {get_model_logname(args.modelpath)} at timestep {timestep}")
    print_modelpath(args.modelpath)

    ions = parse_ions(str(args.ions).split(","), "-ions") if args.ions else None
    dftable = get_line_contributions(
        args.modelpath,
        timestep,
        args.xmin,
        args.xmax,
        ions,
        args.exctemperature,
        percell=args.percell,
        taumin=args.taumin,
    )

    if not args.percell:
        shown = dftable.head(args.maxlines)
        lineset = "all the lines" if args.taumin is None else "the lines that -taumin keeps"
        print_detail(
            f"The table gives {shown.height} of {dftable.height} lines, which emit"
            f" {shown['emission_fraction'].sum():.1%} of the emission of {lineset}. The means take the emission"
            " of each cell as the weight"
        )
        dftable = shown

    print_product(args, dftable)
    if outputfile is not None:
        commentlines = [*get_provenance_lines(args), *get_column_description_lines(dftable.columns)]
        write_data_table(dftable, outputfile, commentlines=commentlines)
        print_saved(outputfile)
