"""Artistools - NLTE population related functions."""

import argparse
import contextlib
import itertools
import math
import typing as t
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import ticker

from artistools import misc
from artistools import plottools
from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.atomic import get_elsymbolslist
from artistools.atomic import get_ionstring
from artistools.atomic import get_kept_level_counts
from artistools.atomic import get_levels
from artistools.constants import km_to_cm
from artistools.estimators import read_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_mgi_of_velocity_kms
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_figscale
from artistools.misc import addarg_labelfontsize
from artistools.misc import addarg_legend
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_positional_items
from artistools.misc import addarg_show
from artistools.misc import addarg_verbose
from artistools.misc import exit_with_error
from artistools.misc import format_frame_path
from artistools.misc import get_model_name
from artistools.misc import get_npts_model
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_time_range
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_time
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import read_wsv
from artistools.misc import resolve_outputfile
from artistools.misc import resolve_positional_modelpath
from artistools.misc.cliutils import CellListAction
from artistools.misc.cliutils import get_cell_list
from artistools.nltepops.core import add_lte_pops
from artistools.nltepops.core import read_nltepops
from artistools.nltepops.core import texifyconfiguration
from artistools.nltepops.core import texifyterm
from artistools.plottools import get_next_color
from artistools.plottools import iter_axes
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_axis_labels
from artistools.plottools import set_axis_properties
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title

defaultoutputfile = "plotnltepops_{elsymbol}_cell{cell:05d}_ts{timestep:03d}_{timedays:.2f}d.pdf"
# a plot against time covers a range of timesteps, and one against velocity a range of cells, so
# neither can be named after the single cell and timestep that the default filename describes
defaultoutputfile_timeorvelocity = "plotnltepops_{elsymbol}.pdf"


def annotate_emission_line(ax: mplax.Axes, y: float, upperlevel: int, lowerlevel: int, label: str) -> None:
    """Draw a labelled arrow between the upper and lower level of an emission line."""
    ax.annotate(
        "",
        xy=(lowerlevel, y),
        xycoords=("data", "axes fraction"),
        xytext=(upperlevel, y),
        textcoords=("data", "axes fraction"),
        arrowprops={"facecolor": "black", "width": 0.1, "headwidth": 6},
    )

    ax.annotate(
        label,
        xy=((upperlevel + lowerlevel) / 2, y),
        xycoords=("data", "axes fraction"),
        size=10,
        va="bottom",
        ha="center",
    )


def plot_reference_data(
    ax: mplax.Axes,
    atomic_number: int,
    ion_stage: int,
    estimators_celltimestep: dict[str, t.Any],
    dfpopthision: pl.DataFrame,
    annotatelines: bool,
) -> None:
    """Overplot the CHIANTI level populations for the same conditions, when a level map file is available."""
    nne, Te, TR, W = (estimators_celltimestep[s] for s in ("nne", "Te", "TR", "W"))
    # comparison to Chianti file
    elsym = get_elsymbol(atomic_number)
    elsymlower = elsym.lower()
    if Path("data", f"{elsymlower}_{ion_stage}-levelmap.txt").exists():
        with Path("data", f"{elsymlower}_{ion_stage}-levelmap.txt").open("r", encoding="utf-8") as levelmapfile:
            levelnumofconfigterm = {}
            for line in levelmapfile:
                row = line.split()
                levelnumofconfigterm[row[0], row[1]] = int(row[2]) - 1

        for depfilepath in sorted(Path("data").rglob(f"chianti_{elsym}_{ion_stage}_*.txt")):
            with depfilepath.open("r", encoding="utf-8") as depfile:
                firstline = depfile.readline()
                file_nne = float(firstline[firstline.find("ne = ") + 5 :].split(",")[0])
                file_Te = float(firstline[firstline.find("Te = ") + 5 :].split(",")[0])
                file_TR = float(firstline[firstline.find("TR = ") + 5 :].split(",")[0])
                file_W = float(firstline[firstline.find("W = ") + 5 :].split(",")[0])
                if math.isclose(file_nne, nne, rel_tol=0.01) and math.isclose(file_Te, Te, abs_tol=10):
                    if file_W > 0:
                        bbstr = " with dilute blackbody"
                        color = "C2"
                        marker = "+"
                    else:
                        bbstr = ""
                        color = "C1"
                        marker = "^"

                    print(
                        f"Plotting reference data from {depfilepath}: "
                        f"nne = {file_nne} (ARTIS {nne}) cm^-3, Te = {file_Te} (ARTIS {Te}) K, "
                        f"TR = {file_TR} (ARTIS {TR}) K, W = {file_W} (ARTIS {W})"
                    )
                    levelnums = []
                    depcoeffs = []
                    firstdep = -1.0
                    for line in depfile:
                        row = line.split()
                        with contextlib.suppress(KeyError, IndexError, ValueError):
                            levelnum = levelnumofconfigterm[row[1], row[2]]
                            if levelnum in dfpopthision["level"].to_numpy():
                                levelnums.append(levelnum)
                                if firstdep < 0:
                                    firstdep = float(row[0])
                                depcoeffs.append(float(row[0]) / firstdep)
                    ax.plot(
                        levelnums,
                        depcoeffs,
                        linewidth=1.5,
                        color=color,
                        label=f"CHIANTI NLTE{bbstr}",
                        linestyle="None",
                        marker=marker,
                        zorder=-1,
                    )

        if annotatelines and atomic_number == 28 and ion_stage == 2:
            annotate_emission_line(ax=ax, y=0.04, upperlevel=6, lowerlevel=0, label=r"7378$~\mathrm{{\AA}}$")
            annotate_emission_line(ax=ax, y=0.15, upperlevel=6, lowerlevel=2, label=r"1.939 $\mu$m")
            annotate_emission_line(ax=ax, y=0.26, upperlevel=7, lowerlevel=1, label=r"7412$~\mathrm{{\AA}}$")

    if annotatelines and atomic_number == 26 and ion_stage == 2:
        annotate_emission_line(ax=ax, y=0.66, upperlevel=9, lowerlevel=0, label=r"12570$~\mathrm{{\AA}}$")
        annotate_emission_line(ax=ax, y=0.53, upperlevel=16, lowerlevel=5, label=r"7155$~\mathrm{{\AA}}$")


def get_floers_data(
    dfpopthision: pl.DataFrame,
    ionpopulation: float,
    atomic_number: int,
    ion_stage: int,
    modelpath: Path,
    T_e: float,
    modelgridindex: int,
) -> tuple[list[int] | None, npt.NDArray[np.floating] | None]:
    """Return Andreas Floers's Fe II/III level populations for Shingles et al. (2022), or None if unavailable.

    ionpopulation is the population of all the levels of the ion, also of the levels that -maxlevel hides. The
    single-zone file gives the fraction of the ion population in each level, thus each fraction takes this
    population.
    """
    floers_levelnums, floers_levelpop_values = None, None

    # comparison to Andeas Floers's NLTE pops for Shingles et al. (2022)
    if atomic_number == 26 and ion_stage in {2, 3}:
        floersfilename = "andreas_level_populations_fe2.txt" if ion_stage == 2 else "andreas_level_populations_fe3.txt"
        if Path(modelpath / floersfilename).is_file():
            print(f"reading {floersfilename}")
            dffloers_levelpops = read_wsv(modelpath / floersfilename, comment_prefix="#").sort("energypercm")
            floers_levelnums = list(range(dffloers_levelpops.height))
            floers_levelpop_values = dffloers_levelpops["frac_ionpop"].to_numpy() * ionpopulation

        floersmultizonefilename = None
        if modelpath.stem.startswith("w7_"):
            if "workfn" in modelpath.parts[-1]:
                floersmultizonefilename = "level_pops_w7_workfn-247d.csv"
            elif "lossboost" not in modelpath.parts[-1]:
                floersmultizonefilename = "level_pops_w7-247d.csv"

        elif modelpath.stem.startswith("subchdet_shen2018_"):
            if "workfn" in modelpath.parts[-1]:
                floersmultizonefilename = "level_pops_subch_shen2018_workfn-247d.csv"
            elif "lossboost4x" in modelpath.parts[-1]:
                floersmultizonefilename = "level_pops_subch_shen2018_electronlossboost4x-247d.csv"
            elif "lossboost8x" in modelpath.parts[-1]:
                print("Shen2018 SubMch lossboost8x detected")
                floersmultizonefilename = "level_pops_subch_shen2018_electronlossboost8x-247d.csv"
            elif "lossboost" not in modelpath.parts[-1]:
                print("Shen2018 SubMch detected")
                floersmultizonefilename = "level_pops_subch_shen2018-247d.csv"

        # the model folder holds the file of one model, and the working folder holds a file that several models share
        multizonepaths: list[Path] = (
            [modelpath / floersmultizonefilename, Path(floersmultizonefilename)] if floersmultizonefilename else []
        )
        floersmultizonepath = next((path for path in multizonepaths if path.is_file()), None)
        if floersmultizonepath is not None:
            # the reference file names the outer velocity of each shell. vel_r_max is in cm/s, and
            # add_derived_cols_to_modeldata gives it for a model of any dimension
            dfmodel, modelmeta = get_modeldata(modelpath)
            modeldata = add_derived_cols_to_modeldata(dfmodel, modelmeta).select("vel_r_max").collect()
            vel_outer = modeldata["vel_r_max"].item(modelgridindex) / km_to_cm
            print(f"  Reading {floersmultizonepath} for vel_outer {vel_outer} and Te {T_e}")
            dffloers = pl.read_csv(floersmultizonepath).filter((pl.col("vel_outer") - vel_outer).abs() < 0.5)
            # the reference file has no superlevel, thus it takes the resolved levels that the plot shows. The two
            # populations then have the same sum over these levels
            dfresolved = dfpopthision.filter(pl.col("config") != "superlevel")
            for row in dffloers.iter_rows(named=True):
                print(f"  ARTIS cell vel_outer: {vel_outer}, Floersfile: {row['vel_outer']}")
                print(f"  ARTIS cell Te: {T_e}, Floersfile: {row['Te']}")
                floers_levelpops = np.array(list(row.values())[4:], dtype=float)[: dfresolved.height]
                floers_levelnums = list(range(len(floers_levelpops)))
                floers_levelpop_values = floers_levelpops * (
                    dfresolved["n_NLTE"].head(len(floers_levelpops)).sum() / sum(floers_levelpops)
                )

    return floers_levelnums, floers_levelpop_values


def get_config_labels(configlist: Sequence[str]) -> list[str]:
    """Return one LaTeX label for the configuration of each level.

    A level that keeps the configuration of the level before it shows a mark and its term only. The
    axis labels then stay short.
    """
    configtexlist: list[str] = []
    prevconfiguration = ""
    for config in configlist:
        # a level name is the configuration, an underscore, and the term. A blank name or a name with no
        # underscore gives an empty configuration, thus such a name never shows a mark
        configuration, _, term = config.rpartition("_")
        if configuration and configuration == prevconfiguration:
            configtexlist.append('" ' + texifyterm(term))
        else:
            configtexlist.append(texifyconfiguration(config))
        prevconfiguration = configuration

    return configtexlist


def set_level_xticks(
    ax: mplax.Axes, levelindices: Sequence[int] | range, configtexlist: Sequence[str], xmode: str, *, lastsubplot: bool
) -> None:
    """Put one tick at each level. The lowest subplot alone shows the names of the configurations."""
    if xmode == "config":
        ax.set_xticks(levelindices)
        if lastsubplot:
            ax.set_xticklabels(configtexlist, rotation=60, horizontalalignment="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels("" for _ in configtexlist)
    elif xmode == "none":
        # set_xticklabels needs fixed ticks, thus this hides the labels of the automatic ticks instead
        ax.tick_params(axis="x", labelbottom=False)


def get_radiative_decays(dftransitions: pl.LazyFrame, dfpopthision: pl.DataFrame, maxlevel_ion: int) -> pl.DataFrame:
    """Return the transitions from the levels that the plot shows, with the emission strength of each one.

    The superlevel row takes a position above the resolved levels, and the level of the atomic data at that
    position is not the superlevel. Thus a transition takes the population of a resolved level alone.
    """
    dfresolvedpops = dfpopthision.filter(pl.col("config") != "superlevel").select(
        pl.col("level").cast(pl.Int32), "n_NLTE"
    )
    return (
        dftransitions
        .filter(pl.col("upper") <= maxlevel_ion)
        .collect()
        .join(dfresolvedpops, how="left", left_on="upper", right_on="level", maintain_order="left")
        .with_columns(
            emissionstrength=pl
            .when(pl.col("n_NLTE").is_not_null())
            .then(pl.col("n_NLTE") * pl.col("A") * pl.col("epsilon_trans_ev"))
            .otherwise(0)
        )
    )


def print_top_radiative_decays(ion_data: dict[str, t.Any], dfpopthision: pl.DataFrame, maxlevel_ion: int) -> None:
    """Print the transitions that emit most strongly from the levels that the plot shows."""
    if "upper" not in ion_data["transitions"].collect_schema().names():
        return

    dftrans = get_radiative_decays(ion_data["transitions"], dfpopthision, maxlevel_ion)
    if dftrans.is_empty():
        return

    print("\nTop radiative decays")
    print(dftrans.sort(by="emissionstrength", descending=True).head(20))


def plot_reference_populations(
    ax: mplax.Axes,
    dfpopthision: pl.DataFrame,
    floers_levelnums: list[int] | None,
    floers_levelpop_values: npt.NDArray[np.floating] | None,
    T_e: float,
    T_R: float,
    ionpopulation: float,
    args: argparse.Namespace,
) -> str:
    """Draw the LTE curves and the reference data, and return the column that holds the ARTIS series."""
    if args.departuremode:
        ax.axhline(y=1.0, color="0.7", linestyle="dashed", linewidth=1.5)
        ax.set_ylabel("Departure coefficient")

        # this mode does not draw T_e, thus skip its colour to keep the colour of every other label
        get_next_color(ax)
        if floers_levelpop_values is not None:
            assert floers_levelnums is not None
            # the reference file can hold a different count of levels, and it has no superlevel. Thus only a resolved
            # level of both gets a coefficient
            dfresolved = dfpopthision.filter(pl.col("config") != "superlevel")
            lte_of_level = dict(
                zip(dfresolved["level"].to_list(), dfresolved["n_LTE_T_e_normed"].to_list(), strict=True)
            )
            departurecoeff_pairs = [
                (level, levelpop / lte_of_level[level])
                for level, levelpop in zip(floers_levelnums, floers_levelpop_values, strict=True)
                if level in lte_of_level
            ]
            ax.plot(
                [level for level, _ in departurecoeff_pairs],
                [departurecoeff for _, departurecoeff in departurecoeff_pairs],
                linewidth=1.5,
                label="Flörs NLTE",
                linestyle="None",
                marker="*",
            )

        return "departure_coeff"

    ax.set_ylabel(r"Level population [cm$^{-3}$]")
    ax.plot(
        dfpopthision["level"],
        dfpopthision["n_LTE_T_e_normed"],
        linewidth=1.5,
        label=f"LTE T$_e$ = {T_e:.0f} K",
        linestyle="None",
        marker="*",
    )

    if floers_levelnums is not None:
        assert floers_levelpop_values is not None
        ax.plot(
            floers_levelnums, floers_levelpop_values, linewidth=1.5, label="Flörs NLTE", linestyle="None", marker="*"
        )

    if not args.hide_lte_tr:
        # the T_R curve also matches the ion population, thus the two LTE curves differ in shape alone
        n_LTE_T_R_normed = dfpopthision["n_LTE_T_R"] * (ionpopulation / float(dfpopthision["n_LTE_T_R"].sum()))
        ax.plot(
            dfpopthision["level"],
            n_LTE_T_R_normed,
            linewidth=1.5,
            label=f"LTE T$_R$ = {T_R:.0f} K",
            linestyle="None",
            marker="*",
        )

    return "n_NLTE"


def get_keptlevelcount_of_element(modelpath: Path) -> dict[int, int | None]:
    """Return the count of levels that ARTIS keeps for each element.

    The dict holds no element if the model has no compositiondata.txt. The superlevel holds no level above the kept
    levels.
    """
    try:
        return get_kept_level_counts(modelpath)
    except FileNotFoundError:
        print_warning(f"{modelpath} holds no compositiondata.txt, thus each superlevel holds all the levels above it")
        return {}


def make_ionsubplot(
    ax: mplax.Axes,
    modelpath: Path,
    atomic_number: int,
    ion_stage: int,
    dfpop: pl.DataFrame,
    adata: pl.DataFrame,
    estimators: dict[tuple[int, int], dict[str, t.Any]],
    T_e: float,
    T_R: float,
    modelgridindex: int,
    timestep: int,
    args: argparse.Namespace,
    lastsubplot: bool | np.bool,
) -> None:
    """Plot the level populations of one ion, in one cell, at one timestep."""
    ion_data = adata.filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage)).row(0, named=True)

    dfpopthision = dfpop.filter(
        (pl.col("modelgridindex") == modelgridindex)
        & (pl.col("timestep") == timestep)
        & (pl.col("Z") == atomic_number)
        & (pl.col("ion_stage") == ion_stage)
    )

    if dfpopthision.is_empty():
        # a cell holds no row for an ion that it has no population of, thus the subplot of that ion stays empty
        print_warning(
            f"cell {modelgridindex} at timestep {timestep} holds no population of"
            f" {get_ionstring(atomic_number, ion_stage, style='spectral')}"
        )
        return

    lte_columns: list[tuple[str, float]] = [("n_LTE_T_e", T_e)]
    if not args.hide_lte_tr:
        lte_columns.append(("n_LTE_T_R", T_R))

    # add_lte_pops moves the superlevel (level -1) above every resolved level. Thus a level number above the
    # largest resolved level in the file identifies a superlevel row
    maxresolvedlevel = dfpopthision["level"].max()
    assert isinstance(maxresolvedlevel, int)
    dfpopthision = add_lte_pops(
        dfpopthision,
        adata,
        lte_columns,
        noprint=False,
        maxlevel=args.maxlevel,
        keptlevelcount_of_element=get_keptlevelcount_of_element(modelpath),
    )
    # the population of the whole ion, before -maxlevel hides levels
    ionpopulation_alllevels = float(dfpopthision["n_NLTE"].sum())

    if args.maxlevel >= 0:
        dfpopthision = dfpopthision.filter(pl.col("level") <= args.maxlevel)

    ionpopulation = float(dfpopthision["n_NLTE"].sum())
    ionkey = get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
    ionpopulation_fromest = estimators.get((timestep, modelgridindex), {}).get(f"nnion_{ionkey}", 0.0)

    maxlevel_ion = dfpopthision["level"].max()
    assert isinstance(maxlevel_ion, int)
    levelnames = ion_data["levels"]["levelname"].to_list()
    maxresolvedlevel_shown = min(maxresolvedlevel, maxlevel_ion)
    configlist = levelnames[: maxresolvedlevel_shown + 1]
    configtexlist = get_config_labels(configlist)

    # the superlevel takes the highest position, with one blank position below it. It has no entry in
    # the atomic data, thus its tick must not take the name of a resolved level
    blankpositions = maxlevel_ion - maxresolvedlevel_shown - 1
    if blankpositions >= 0:
        configlist = [*configlist, *[""] * blankpositions, "superlevel"]
        configtexlist = [*configtexlist, *[""] * blankpositions, "superlevel"]

    levels: list[int] = dfpopthision["level"].to_list()
    dfpopthision = dfpopthision.with_columns(
        # a level name that ends in "o" in front of the term is a level of odd parity
        parity=pl.Series([
            1 if (level <= maxresolvedlevel and levelnames[level].split("[")[0].endswith("o")) else 0
            for level in levels
        ]),
        config=pl.Series(["superlevel" if level > maxresolvedlevel else configlist[level] for level in levels]),
        texname=pl.Series(["superlevel" if level > maxresolvedlevel else configtexlist[level] for level in levels]),
    )

    set_level_xticks(ax, range(maxlevel_ion + 1), configtexlist, args.x, lastsubplot=bool(lastsubplot))

    print(
        f"{get_ionstring(atomic_number, ion_stage, style='spectral')} has a summed "
        f"level population of {ionpopulation:.1f} (from estimator file ion pop = {ionpopulation_fromest})"
    )

    lte_scalefactor = (
        # scale to match the ground state populations
        float(dfpopthision["n_NLTE"].item(0) / dfpopthision["n_LTE_T_e"].item(0))
        if args.departuremode
        # else scale to match the ion population
        else ionpopulation / float(dfpopthision["n_LTE_T_e"].sum())
    )

    dfpopthision = dfpopthision.with_columns(n_LTE_T_e_normed=pl.col("n_LTE_T_e") * lte_scalefactor).with_columns(
        departure_coeff=pl.col("n_NLTE") / pl.col("n_LTE_T_e_normed")
    )

    if dfpopthision.height < 30:
        with pl.Config(tbl_cols=150, tbl_rows=30):
            print(dfpopthision.drop("timestep", "modelgridindex", "Z", "parity", "texname"))

    print_top_radiative_decays(ion_data, dfpopthision, maxlevel_ion)

    ax.set_yscale("log")

    floers_levelnums, floers_levelpop_values = get_floers_data(
        dfpopthision, ionpopulation_alllevels, atomic_number, ion_stage, modelpath, T_e, modelgridindex
    )
    ycolumnname = plot_reference_populations(
        ax, dfpopthision, floers_levelnums, floers_levelpop_values, T_e, T_R, ionpopulation, args
    )

    ax.plot(
        dfpopthision["level"],
        dfpopthision[ycolumnname],
        linewidth=1.5,
        linestyle="None",
        marker="x",
        label="ARTIS NLTE",
        color="black",
    )

    dfpopthisionoddlevels = dfpopthision.filter(pl.col("parity") == 1)
    if not dfpopthisionoddlevels.is_empty():
        ax.plot(
            dfpopthisionoddlevels["level"],
            dfpopthisionoddlevels[ycolumnname],
            linewidth=2,
            label="Odd parity",
            linestyle="None",
            marker="s",
            markersize=10,
            markerfacecolor=(0, 0, 0, 0),
            markeredgecolor="black",
        )

    # the ion names the subplot rather than every legend entry, thus the legend stays short
    ax.annotate(
        get_ionstring(atomic_number, ion_stage, style="chargelatex"),
        xy=(1.0, 1.0),
        xycoords="axes fraction",
        xytext=(-10, -10),
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize="large",
    )

    # a comparison with reference data needs the estimator values of the cell, thus skip it if they are absent
    if args.plotrefdata and (timestep, modelgridindex) in estimators:
        plot_reference_data(
            ax, atomic_number, ion_stage, estimators[timestep, modelgridindex], dfpopthision, annotatelines=True
        )


def get_levelnames_of_ion(modelpath: Path | str, ions: Sequence[tuple[int, int]]) -> dict[tuple[int, int], list[str]]:
    """Return the names of the levels of each ion.

    The function stops with a message for an ion that the atomic data does not hold. The plot shows the names of the
    levels alone, thus it reads no transitions.
    """
    adata = get_levels(modelpath, ionlist=ions)
    levelnames_of_ion: dict[tuple[int, int], list[str]] = {
        (Z, ion_stage): dflevels["levelname"].to_list()
        for Z, ion_stage, dflevels in adata.select("Z", "ion_stage", "levels").iter_rows()
    }
    if missing := [ion for ion in ions if ion not in levelnames_of_ion]:
        exit_with_error(
            f"the atomic data holds no {', '.join(itertools.starmap(get_ionstring, missing))}",
            "Give an ion stage of the model with -ion_stages",
        )

    return levelnames_of_ion


def make_plot_populations_with_time_or_velocity(
    modelpaths: Sequence[Path | str], atomic_number: int, args: argparse.Namespace
) -> None:
    """Plot how the chosen level populations of the ion stages of one element vary with time or velocity.

    Each ion stage of -ion_stages gives one series for each level of -levels, and the figure goes to its own file.
    """
    ions = [(atomic_number, int(ion_stage)) for ion_stage in args.ion_stages]
    levelnames_of_ion = get_levelnames_of_ion(modelpaths[0], ions)

    if args.timedayslist:
        rows = len(args.timedayslist)
        timedayslist = args.timedayslist
        args.subplots = True
    else:
        rows = 1
        timedayslist = [get_timestep_time(modelpaths[0], ts) for ts in range(args.timestepmin, args.timestepmax + 1)]
        args.subplots = False

    if args.x == "time" and not args.subplots:
        # the time branch of plot_populations_with_time_or_velocity draws the full time series in
        # one call, thus one call fills the single axes
        timedayslist = timedayslist[:1]

    # a range of times puts the series of each time on one axes, thus a series also names its time
    showtime = args.x == "velocity" and not args.subplots and len(timedayslist) > 1

    cols = 1
    fig, axesgrid = make_frame_figure(args, rows=rows, cols=cols, aspect=0.847, sharey=True)
    ax = axesgrid.flatten() if args.subplots else axesgrid[0][0]

    for plotnumber, timedays in enumerate(timedayslist):
        axis = ax[plotnumber] if args.subplots else ax
        assert isinstance(axis, mplax.Axes)
        plot_populations_with_time_or_velocity(
            axis, modelpaths, timedays, args.levels, levelnames_of_ion, args=args, showtime=showtime
        )

    # the axis label size comes from the artistools matplotlibrc
    labelfontsize = None
    if args.x == "time":
        xlabel = "Time Since Explosion [days]"
    elif args.x == "velocity":
        xlabel = r"Cell mid-point velocity [km s$^{-1}$]"
    ylabel = r"Level population [cm$^{-3}$]"

    set_axis_labels(fig, ax, xlabel, ylabel, labelfontsize, args)
    if args.subplots:
        for plotnumber, axis in enumerate(ax):
            axis.set_yscale("log")
            if args.timedayslist:
                ymin, _ = axis.get_ylim()
                _, xmax = axis.get_xlim()
                axis.text(xmax * 0.85, ymin * 50, f"{args.timedayslist[plotnumber]} days")
        plottools.set_legend(ax[0], args, loc="best", frameon=True, fontsize="x-small")
    else:
        assert isinstance(ax, mplax.Axes)
        plottools.set_legend(ax, args, loc="best", frameon=True, fontsize="x-small")
        ax.set_yscale("log")

    title = ", ".join(itertools.starmap(get_ionstring, ions))
    if args.x == "time":
        title += f", mgi = {get_single_modelgridindex(args.modelgridindex)}"
    elif args.x == "velocity":
        title += f", {timedayslist} days"
    set_plot_title(iter_axes(ax)[-1], title, args)

    set_axis_properties(ax, args)

    outputfilename = str(args.outputfile).format(elsymbol=get_elsymbol(atomic_number))
    save_figure(fig, outputfilename, format="pdf", args=args)


def get_level_label(levelnames: Sequence[str], ionlevel: int) -> str:
    """Return the name of a level for the legend, or its index if the atomic data gives no name."""
    levelname = levelnames[ionlevel] if ionlevel < len(levelnames) else ""
    return levelname or f"level {ionlevel}"


def plot_populations_with_time_or_velocity(
    ax: mplax.Axes,
    modelpaths: Sequence[Path | str],
    timedays: float,
    ionlevels: Sequence[int],
    levelnames_of_ion: Mapping[tuple[int, int], Sequence[str]],
    args: argparse.Namespace,
    showtime: bool = False,
) -> None:
    """Plot one series for each level of each ion, against time or velocity as selected by args.x.

    The label of a series names its ion if the plot shows more than one ion, and its time if showtime is true.
    """
    velocity: pl.Series | None = None
    if args.x == "time":
        timesteps = list(range(args.timestepmin, args.timestepmax + 1))

        if not args.modelgridindex:
            exit_with_error("-x time needs one cell. Give it with -modelgridindex")

        modelgridindex = get_single_modelgridindex(args.modelgridindex)
        assert modelgridindex is not None, "the branch above stops when no cell is given"
        modelgridindex_list = [modelgridindex] * len(timesteps)

    if args.x == "velocity":
        # vel_r_mid is the mid-point radial velocity of a cell in a model of any dimension, in cm/s
        dfmodel, modelmeta = get_modeldata(modelpaths[0])
        modeldata = add_derived_cols_to_modeldata(dfmodel, modelmeta).select("vel_r_mid").collect()
        velocity = modeldata["vel_r_mid"] / km_to_cm
        modelgridindex_list = [mgi for mgi, _ in enumerate(velocity)]

        timesteps = [get_timestep_of_timedays(modelpaths[0], timedays)] * len(modelgridindex_list)

    markers = ["o", "x", "^", "s", "8"]
    for modelnumber, modelpath in enumerate(modelpaths):
        # the loop changes only the cell or only the timestep, thus one read with that filter supplies the whole loop
        dfpop_all = (
            read_nltepops(modelpath, modelgridindex=modelgridindex_list[0])
            if args.x == "time"
            else read_nltepops(modelpath, timestep=timesteps[0])
        )
        for (Z, ion_stage), levelnames in levelnames_of_ion.items():
            populations = get_level_populations(dfpop_all, Z, ion_stage, timesteps, modelgridindex_list, ionlevels)
            ionlabel = f"{get_ionstring(Z, ion_stage)} " if len(levelnames_of_ion) > 1 else ""
            timelabel = f" at {timedays:.0f}d" if showtime else ""
            for ionlevel in ionlevels:
                plot_level_population(
                    ax,
                    modelpath,
                    populations,
                    ionlevel,
                    velocity,
                    marker=markers[modelnumber],
                    label=f"{ionlabel}{get_level_label(levelnames, ionlevel)}{timelabel}",
                )


def get_level_populations(
    dfpop_all: pl.DataFrame,
    atomic_number: int,
    ion_stage: int,
    timesteps: Sequence[int],
    modelgridindex_list: Sequence[int],
    ionlevels: Sequence[int],
) -> dict[tuple[int, int, int], float]:
    """Return the NLTE population of each level of one ion, keyed by (timestep, level, modelgridindex).

    timesteps and modelgridindex_list give the pairs of timestep and cell. A pair with no row gives no entry.
    """
    populations: dict[tuple[int, int, int], float] = {}
    # a 3D model holds thousands of cells, thus one partition costs much less than a filter for each cell
    dfpop_of_cell = dfpop_all.filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage)).partition_by(
        "modelgridindex", as_dict=True
    )
    for timestep, mgi in zip(timesteps, modelgridindex_list, strict=False):
        dfpop = dfpop_of_cell.get((mgi,))
        if dfpop is None:
            continue
        timesteppops = dfpop.filter(pl.col("timestep") == timestep)
        if timesteppops.is_empty():
            continue
        # setdefault keeps the first row for a duplicated level, matching the .item(0) this replaces
        pop_of_level: dict[int, float] = {}
        for level, n_nlte in zip(timesteppops["level"], timesteppops["n_NLTE"], strict=True):
            pop_of_level.setdefault(level, n_nlte)
        for ionlevel in ionlevels:
            # a 3D model holds cells of low density, and such a cell can hold fewer levels. The plot
            # leaves out that cell in place of stopping the command
            if ionlevel not in pop_of_level:
                print_warning(
                    f"cell {mgi} at timestep {timestep} holds no level {ionlevel} of"
                    f" {get_ionstring(atomic_number, ion_stage, style='spectral')}."
                    f" The cell holds the levels {min(pop_of_level)} to {max(pop_of_level)}"
                )
                continue
            populations[timestep, ionlevel, mgi] = pop_of_level[ionlevel]

    return populations


def plot_level_population(
    ax: mplax.Axes,
    modelpath: Path | str,
    populations: Mapping[tuple[int, int, int], float],
    ionlevel: int,
    velocity: pl.Series | None,
    marker: str,
    label: str,
) -> None:
    """Plot the population of one level against time, or against the velocity of each cell if velocity is given.

    populations is the result of get_level_populations. velocity gives the velocity of each cell in km/s.
    """
    keys = [(ts, mgi) for ts, level, mgi in populations if level == ionlevel]
    plotpopulations = np.array([populations[ts, ionlevel, mgi] for ts, mgi in keys])

    if velocity is None:
        timedayslist = [get_timestep_time(modelpath, ts) for ts, _mgi in keys]
        ax.plot(timedayslist, plotpopulations, marker=marker, label=label)
        return

    plotvelocities = np.array([float(velocity[mgi]) for _ts, mgi in keys])
    # a 2D or a 3D model numbers its cells by position and not by speed, thus the line joins the
    # cells in the order of their speed
    velocityorder = np.argsort(plotvelocities, kind="stable")
    ax.plot(plotvelocities[velocityorder], plotpopulations[velocityorder], marker=marker, label=label)


def get_subplot_block(mgilistindex: int, nionstages: int) -> tuple[int, int]:
    """Return the first and the last subplot index of one cell.

    Each cell owns one subplot for each ion stage, thus its block starts after the blocks of the cells
    in front of it. A block that started at the index of the cell drew over the block before it.
    """
    firstindex = mgilistindex * nionstages

    return firstindex, firstindex + nionstages - 1


def make_singletimestep_plot(
    modelpath: Path,
    atomic_number: int,
    ion_stages_displayed: list[int] | None,
    mgilist: Sequence[int],
    timestep: int,
    args: argparse.Namespace,
) -> None:
    """Plot level populations for chosens ions of an element in a cell and timestep of an ARTIS model."""
    modelname = get_model_name(modelpath)
    adata = get_levels(
        modelpath,
        get_transitions=args.gettransitions,
        derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms"],
    )

    time_days = get_timestep_time(modelpath, timestep)

    # one read of the ranks that own the cells in mgilist supplies the data for every cell
    dfpop_allcells = read_nltepops(modelpath, timestep=timestep, modelgridindex=list(mgilist))

    # every cell gets the same rows of subplots, thus the ion stages come from the cells together. The
    # first cell alone can hold no population, e.g. a cell of low density
    dfpop_element = dfpop_allcells.filter(pl.col("Z") == atomic_number)
    if dfpop_element.is_empty():
        print(f"No NLTE population data for Z={atomic_number} at timestep {timestep}")
        return

    max_ion_stage = dfpop_element["ion_stage"].max()

    assert isinstance(max_ion_stage, int)
    if dfpop_element.filter(pl.col("ion_stage") == max_ion_stage)["level"].n_unique() == 1:
        # a single-level ion shows nothing, thus the plot leaves it out
        max_ion_stage -= 1

    ion_stage_list = sorted([
        i
        for i in dfpop_element["ion_stage"].unique()
        if i <= max_ion_stage and (ion_stages_displayed is None or i in ion_stages_displayed)
    ])
    if not ion_stage_list:
        availablestages = sorted(i for i in dfpop_element["ion_stage"].unique() if i <= max_ion_stage)
        exit_with_error(
            f"no ion stage of {get_elsymbol(atomic_number)} to plot at timestep {timestep}",
            f"The NLTE file holds the ion stages {', '.join(str(i) for i in availablestages)} of more than one level."
            " Give one of them with -ion_stages",
        )

    # the height of one frame in inches, as a part of the width of a frame
    subplotheight = (7.0 * (2.4 / 6 if args.x == "config" else 1.8 / 6) - 0.47) / 6.47

    nrows = len(ion_stage_list) * len(mgilist)
    fig, axesgrid = make_frame_figure(args, rows=nrows, aspect=subplotheight, sharex=False)
    axes = axesgrid[:, 0]

    assert mgilist

    # invariant to the cell loop, so read the estimators and the model once instead of once per cell
    estimators = read_estimators(modelpath, timestep=timestep, modelgridindex=list(mgilist))
    lzmodeldata, modelmeta = get_modeldata(modelpath)
    lzmodeldata = add_derived_cols_to_modeldata(lzmodeldata, modelmeta=modelmeta)
    velocity_kmps_of_mgi = {
        mgi: vel_r_mid / km_to_cm
        for mgi, vel_r_mid in lzmodeldata
        .filter(pl.col("modelgridindex").is_in(mgilist))
        .select(["modelgridindex", "vel_r_mid"])
        .collect()
        .iter_rows()
    }

    elsymbol = get_elsymbol(atomic_number)
    # the lowest panel that the loop draws shows the configuration names, and a cell without data draws no panel
    mgis_withdata = set(dfpop_allcells["modelgridindex"].to_list())
    lastmgi_withdata = next(mgi for mgi in reversed(mgilist) if mgi in mgis_withdata)

    for mgilistindex, modelgridindex in enumerate(mgilist):
        mgifirstaxindex, mgilastaxindex = get_subplot_block(mgilistindex, len(ion_stage_list))

        print(
            f"Plotting NLTE pops for {modelname} modelgridindex {modelgridindex}, timestep {timestep} (t={time_days}d)"
        )
        print(f"Z={atomic_number} {elsymbol}")

        if (timestep, modelgridindex) in estimators:
            T_e = estimators[timestep, modelgridindex]["Te"]
            T_R = estimators[timestep, modelgridindex]["TR"]
            W = estimators[timestep, modelgridindex]["W"]
            nne = estimators[timestep, modelgridindex]["nne"]
            print(f"nne = {nne} cm^-3, T_e = {T_e} K, T_R = {T_R} K, W = {W}")
        else:
            print_warning(f"No estimator data. Setting T_e = T_R = {args.exc_temperature} K, nne and W unknown")
            T_e = args.exc_temperature
            T_R = args.exc_temperature
            # only used for display in the subplot title, so report them as unknown rather than inventing a value
            W = math.nan
            nne = math.nan

        dfpop = dfpop_allcells.filter(pl.col("modelgridindex") == modelgridindex)

        if dfpop.is_empty():
            print(f"No NLTE population data for modelgrid cell {modelgridindex} timestep {timestep}")
            # skip this cell alone. A return would discard the panels that the earlier cells filled
            continue

        dfpop = dfpop.filter(pl.col("Z") == atomic_number)

        subplot_title = modelname
        if len(subplot_title) > 10:
            subplot_title += "\n"
        subplot_title += f" {velocity_kmps_of_mgi[modelgridindex]:.0f} km/s at {time_days:.0f}d"
        subplot_title += rf" (Te={T_e:.0f} K, nne={nne:.1e} cm$^{{-3}}$, T$_R$={T_R:.0f} K, W={W:.1e})"

        set_plot_title(axes[mgifirstaxindex], subplot_title, args)

        for ax, ion_stage in zip(axes[mgifirstaxindex : mgilastaxindex + 1], ion_stage_list, strict=False):
            lastsubplot = modelgridindex == lastmgi_withdata and ion_stage == ion_stage_list[-1]
            make_ionsubplot(
                ax,
                modelpath,
                atomic_number,
                int(ion_stage),
                dfpop,
                adata,
                estimators,
                T_e,
                T_R,
                modelgridindex,
                timestep,
                args,
                lastsubplot=lastsubplot,
            )

            ax.set_xlim(left=-1)

    # one legend for the figure, because the annotation names the ion and every subplot draws the same
    # series. An ion with no odd-parity level adds no entry for it, thus collect the entries of every
    # subplot. Reverse the order so that the entry of the first subplot wins for a repeated label
    handlesbylabel = {
        label: handle
        for ax in reversed(iter_axes(axes))
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True)
    }
    set_legend(
        axes[0],
        args,
        handles=list(handlesbylabel.values()),
        labels=list(handlesbylabel.keys()),
        loc="best",
        handlelength=1,
        frameon=True,
        numpoints=1,
        edgecolor="0.93",
        facecolor="0.93",
    )

    if args.x == "index":
        axes[-1].set_xlabel(r"Level index")

    set_axis_properties(axes, args)
    # after set_axis_properties, which turns the automatic minor ticks on: a level index axis wants one
    # minor tick for each level, thus it keeps its own locator
    for ax in axes:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

    outputfilename = format_frame_path(
        args.outputfile, elsymbol=get_elsymbol(atomic_number), cell=mgilist[0], timestep=timestep, timedays=time_days
    )
    save_figure(fig, outputfilename, format="pdf", args=args)


def get_atomic_numbers(elements: Sequence[str | int]) -> list[int]:
    """Return the atomic number of each element that the command names, e.g. Fe or 26.

    A name of no element gives a warning, and the command then plots the other elements.
    """
    atomic_numbers = []
    for element in elements:
        atomic_number = int(element) if str(element).isdigit() else get_atomic_number(str(element))
        if not 1 <= atomic_number < len(get_elsymbolslist()):
            print_warning(f"could not find the element '{element}'")
            continue
        atomic_numbers.append(atomic_number)

    return atomic_numbers


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_positional_items(
        parser,
        dest="elements",
        metavar="element",
        helptext="Elements to plot (default: Fe), then the ARTIS folder, e.g. Fe Co mymodel",
    )

    # the default is None, thus resolve_positional_modelpath sees a -modelpath that the user gave
    addarg_modelpath(parser)

    timegroup = parser.add_mutually_exclusive_group()
    timegroup.add_argument("-timedays", "-time", "-t", help="Time in days to plot")

    timegroup.add_argument("-timedayslist", nargs="+", help="List of times in days for time sequence subplots")

    # no type, thus this reads the same text as every other command: a number, "last", or a range
    timegroup.add_argument("-timestep", "-ts", help="Timestep number to plot, e.g. 40, last, or 40-45")

    cellgroup = parser.add_mutually_exclusive_group()
    # a mutually exclusive group, thus the flags are spelled out rather than taken from addarg_modelgridindex
    cellgroup.add_argument(
        "-modelgridindex",
        "-cell",
        "-mgi",
        action=CellListAction,
        help="Plotted model grid cell, a range e.g. 3-7, or a list e.g. 3,5",
    )

    cellgroup.add_argument("-velocity", "-v", default=[], type=float, nargs="*", help="Specify cell by velocity")

    parser.add_argument("-exc-temperature", type=float, default=6000.0, help="Default if no estimator data")

    parser.add_argument(
        "-x", choices=["index", "config", "time", "velocity", "none"], default="index", help="Horizontal axis variable"
    )

    parser.add_argument("-ion_stages", help="Ion stage range, 1 is neutral, 2 is 1+")

    parser.add_argument(
        "-levels", type=int, nargs="+", help="Choose levels to plot"
    )  # currently only for x axis = time

    parser.add_argument("-maxlevel", default=-1, type=int, help="Maximum level to plot")

    addarg_figscale(parser)

    parser.add_argument(
        "--departuremode", action="store_true", help="Show departure coefficients instead of populations"
    )

    parser.add_argument("--gettransitions", action="store_true", help="Show the most significant transitions")

    parser.add_argument("--plotrefdata", action="store_true", help="Show reference data")

    parser.add_argument("--hide-lte-tr", action="store_true", help="Hide LTE populations at T=T_R")

    addarg_notitle(parser)

    addarg_legend(parser)
    addarg_show(parser)
    addarg_darkmode(parser)
    addarg_verbose(parser)

    addarg_labelfontsize(parser)

    addarg_axislimits(parser)

    # no default here: which one applies depends on -x, so main chooses it when resolving the path
    addarg_output(parser, kind="file", helptext="Path/filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS non-LTE populations."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    # the parser gives text such as "1-3", and a keyword argument of main() can also give a number or a list
    args.ion_stages = get_cell_list(args.ion_stages) if args.ion_stages else None
    # the ARTIS folder is the last positional argument, thus "plotnltepops Fe mymodel" reads mymodel
    args.elements = resolve_positional_modelpath(args, "elements") or ["Fe"]

    modelpath = args.modelpath
    if args.x == "time" and args.timedayslist:
        exit_with_error(
            "-x time puts the time on the horizontal axis, thus -timedayslist gives no panel for each time",
            "Give -x velocity with -timedayslist, or give -x time with a range such as -timedays 270-275",
        )

    if args.x in {"time", "velocity"}:
        args.modelpath = normalize_path_list(args.modelpath)

        if not args.ion_stages:
            misc.exit_with_error("no ion stage was given", "Give an ion stage with -ion_stages, e.g. -ion_stages 2")

        if not args.levels:
            misc.exit_with_error("no levels were given", "Give the levels to plot with -levels, e.g. -levels 0 1 2")

    timesteps_selected: list[int] | None = None
    if args.timedayslist:
        print(f"Plotting the times {args.timedayslist}")
        # the -x time path reads args.timestepmin and args.timestepmax, thus set them from the
        # listed times. The level-index loop below iterates only the listed timesteps, because a
        # range would add the timesteps between two non-adjacent list entries.
        timesteps_selected = sorted({get_timestep_of_timedays(modelpath, timedays) for timedays in args.timedayslist})
        args.timestepmin, args.timestepmax = timesteps_selected[0], timesteps_selected[-1]
    elif args.timedays is not None or args.timestep is not None:
        # get_time_range reads one time, one timestep, or a range of either
        args.timestepmin, args.timestepmax, _, _ = get_time_range(
            modelpath, timestep_range_str=args.timestep, timedays_range_str=args.timedays
        )
    elif args.x == "velocity":
        # each time gives one series for each level, thus every timestep of a run gave a plot that nobody can read
        exit_with_error(
            "-x velocity needs a time",
            "Give a time with -timedays, e.g. -timedays 300, or a panel for each time with -timedayslist",
        )
    elif args.x == "time":
        args.timestepmin, args.timestepmax, _, _ = get_time_range(modelpath, timemin=0, timemax=math.inf)
    else:
        exit_with_error(
            "no time given. Use -timedays or -timestep. A model of more than one cell also needs -modelgridindex"
        )

    args.outputfile = resolve_outputfile(
        args.outputfile, defaultoutputfile_timeorvelocity if args.x in {"time", "velocity"} else defaultoutputfile
    )

    mgilist: list[int] = list(args.modelgridindex or [])
    mgilist.extend(mgi for mgi in [get_mgi_of_velocity_kms(modelpath, vel) for vel in args.velocity] if mgi is not None)
    # the branches below read args.modelgridindex, thus give them the cells of -velocity too
    args.modelgridindex = mgilist

    npts_model = get_npts_model(modelpath)
    # a velocity plot draws every cell of the model, thus it needs no cell of its own. A time plot and a
    # level index plot draw one cell, thus they do need one
    if not mgilist and args.x != "velocity":
        if npts_model > 1:
            exit_with_error(
                f"no model grid cell given, and this model has {npts_model} cells. "
                "Use -modelgridindex (or -velocity) to select one"
            )
        mgilist.append(0)

    if outofrange := [mgi for mgi in mgilist if not 0 <= mgi < npts_model]:
        exit_with_error(f"model grid cell {outofrange[0]} is outside the range 0 to {npts_model - 1}")

    if args.x in {"time", "velocity"}:
        # each element takes a figure of its own, as in a plot against the level index
        for atomic_number in get_atomic_numbers(args.elements):
            make_plot_populations_with_time_or_velocity(args.modelpath, atomic_number, args)
        return

    timesteps_included = (
        timesteps_selected if timesteps_selected is not None else list(range(args.timestepmin, args.timestepmax + 1))
    )
    for atomic_number in get_atomic_numbers(args.elements):
        for timestep in timesteps_included:
            make_singletimestep_plot(modelpath, atomic_number, args.ion_stages, mgilist, timestep, args)
