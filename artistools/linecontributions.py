"""Give the Sobolev optical depth and the emission of each line in a wavelength range, in each cell of a model."""

import argparse
import math
import typing as t
from collections.abc import Collection
from collections.abc import Sequence
from itertools import starmap
from pathlib import Path

import numpy as np
import polars as pl

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
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import get_model_logname
from artistools.misc import get_timestep_time
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
from artistools.nltepops import has_nlte_populations
from artistools.nltepops import read_nltepops

DEFAULTOUTPUTNAME = "linecontributions.txt"

# the columns that name one line, i.e. one transition of one ion
LINEKEYS: t.Final = ("ion", "lambda_angstroms", "lower", "upper", "A_per_s")

# a batch of cells makes one row for each pair of a cell and a line, and one value for each pair of a cell and a
# level. This count keeps a batch near 1 GB for the columns of the cell table
ROWSPERBATCH = 4_000_000

# for a small tau, (1 - exp(-tau)) / tau loses accuracy, thus a small tau takes the series 1 - tau / 2
SMALLTAU = 1e-6

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


def get_line_contributions(
    modelpath: Path | str,
    timestep: int,
    lambdamin: float,
    lambdamax: float,
    ions: Collection[tuple[int, int]] | None = None,
    exctemperature: str = "auto",
    *,
    percell: bool = False,
) -> pl.DataFrame:
    """Return the emission of each line in the wavelength range, or with percell, one row for each cell and line.

    The method and the assumptions go to the log.
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

    dfcells = (
        lfestimators
        .select(
            "modelgridindex",
            "Te",
            "nne",
            "volume",
            pl.col(temperaturecolumn).alias("T_exc"),
            pl.col("^nnion_.*$"),
            velocity_kmps=pl.col("vel_r_mid_on_c") * C_cm_per_s / km_to_cm,
        )
        .sort("modelgridindex")
        .collect()
    )

    print_detail(
        f"{dflines.height} lines of {len(levelsofion)} ions from {lambdamin:g} to {lambdamax:g} Å. The wavelengths are"
        " vacuum wavelengths from the level energies"
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

    maxlevelcount = max(levels.height for levels in levelsofion.values())
    cellsperbatch = max(1, ROWSPERBATCH // max(dflines.height, maxlevelcount))
    celltables = []
    summaries = []
    for firstcell in range(0, dfcells.height, cellsperbatch):
        dfcelllines = get_cell_line_table(
            dfcells.slice(firstcell, cellsperbatch), dflines, levelsofion, dfnltepops, time_days
        )
        summaries.append(get_line_summary(dfcelllines))
        if percell:
            celltables.append(dfcelllines)

    dfsummary = get_line_summary_of_batches(summaries)
    if (inverted := dfsummary.filter(pl.col("tau_sobolev_max") < 0.0).height) > 0:
        print_warning(
            f"{inverted} lines have a negative tau_sobolev in each cell, i.e. a population inversion. The escape"
            " probability of such a line is above 1"
        )
    if (nopopulation := dfsummary.filter(pl.col("tau_sobolev_max").is_null()).height) > 0:
        print_warning(f"{nopopulation} lines have no level population in any cell, thus they have no tau_sobolev")

    return pl.concat(celltables) if percell else dfsummary


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
        args.modelpath, timestep, args.xmin, args.xmax, ions, args.exctemperature, percell=args.percell
    )

    if not args.percell:
        shown = dftable.head(args.maxlines)
        print_detail(
            f"The table gives {shown.height} of {dftable.height} lines, which emit"
            f" {shown['emission_fraction'].sum():.1%} of the emission of all the lines. The means take the emission"
            " of each cell as the weight"
        )
        dftable = shown

    print_product(args, dftable)
    if outputfile is not None:
        commentlines = [*get_provenance_lines(args), *get_column_description_lines(dftable.columns)]
        write_data_table(dftable, outputfile, commentlines=commentlines)
        print_saved(outputfile)
