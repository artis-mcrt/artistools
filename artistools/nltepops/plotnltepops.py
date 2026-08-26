"""Artistools - NLTE population related functions."""

import argparse
import contextlib
import itertools
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import ticker

import artistools as at
from artistools.commands import run_subcommand
from artistools.constants import km_to_cm
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_nolegend
from artistools.misc import addarg_notitle
from artistools.misc import addarg_outputfile
from artistools.misc import addarg_quiet
from artistools.misc import addarg_show
from artistools.misc import exit_with_error
from artistools.misc import get_npts_model
from artistools.plottools import get_figsize
from artistools.plottools import save_figure
from artistools.plottools import set_legend

defaultoutputfile = "plotnlte_{elsymbol}_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf"
# a plot against time covers a range of timesteps, and one against velocity a range of cells, so
# neither can be named after the single cell and timestep that the default filename describes
defaultoutputfile_timeorvelocity = "plotnltelevelpops_{elsymbol}.pdf"


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
    elsym = at.get_elsymbol(atomic_number)
    elsymlower = elsym.lower()
    if Path("data", f"{elsymlower}_{ion_stage}-levelmap.txt").exists():
        # ax.set_ylim(bottom=2e-3)
        # ax.set_ylim(top=4)
        with Path("data", f"{elsymlower}_{ion_stage}-levelmap.txt").open("r", encoding="utf-8") as levelmapfile:
            levelnumofconfigterm = {}
            for line in levelmapfile:
                row = line.split()
                levelnumofconfigterm[row[0], row[1]] = int(row[2]) - 1

        # ax.set_ylim(bottom=5e-4)
        for depfilepath in sorted(Path("data").rglob(f"chianti_{elsym}_{ion_stage}_*.txt")):
            with depfilepath.open("r", encoding="utf-8") as depfile:
                firstline = depfile.readline()
                file_nne = float(firstline[firstline.find("ne = ") + 5 :].split(",")[0])
                file_Te = float(firstline[firstline.find("Te = ") + 5 :].split(",")[0])
                file_TR = float(firstline[firstline.find("TR = ") + 5 :].split(",")[0])
                file_W = float(firstline[firstline.find("W = ") + 5 :].split(",")[0])
                # print(depfilepath, file_nne, nne, file_Te, Te, file_TR, TR, file_W, W)
                if math.isclose(file_nne, nne, rel_tol=0.01) and math.isclose(file_Te, Te, abs_tol=10):
                    if file_W > 0:
                        bbstr = " with dilute blackbody"
                        color = "C2"
                        marker = "+"
                    else:
                        bbstr = ""
                        color = "C1"
                        marker = "^"

                    print(f"Plotting reference data from {depfilepath},")
                    print(
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
    dfpopthision: pl.DataFrame, atomic_number: int, ion_stage: int, modelpath: Path, T_e: float, modelgridindex: int
) -> tuple[list[int] | None, npt.NDArray[np.floating] | None]:
    """Return Andreas Floers's Fe II/III level populations for Shingles et al. (2022), or None if unavailable."""
    floers_levelnums, floers_levelpop_values = None, None

    # comparison to Andeas Floers's NLTE pops for Shingles et al. (2022)
    if atomic_number == 26 and ion_stage in {2, 3}:
        floersfilename = "andreas_level_populations_fe2.txt" if ion_stage == 2 else "andreas_level_populations_fe3.txt"
        if Path(modelpath / floersfilename).is_file():
            print(f"reading {floersfilename}")
            dffloers_levelpops = at.read_wsv(modelpath / floersfilename, comment_prefix="#").sort("energypercm")
            # floers_levelnums = floers_levelpops['index'].values - 1
            floers_levelnums = list(range(dffloers_levelpops.height))
            floers_levelpop_values = dffloers_levelpops["frac_ionpop"].to_numpy() * dfpopthision["n_NLTE"].sum()

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

        if floersmultizonefilename and Path(floersmultizonefilename).is_file():
            modeldata = at.inputmodel.get_modeldata(modelpath)[0].collect()
            vel_outer = modeldata["vel_r_max_kmps"].item(modelgridindex)
            print(f"  reading {floersmultizonefilename}", vel_outer, T_e)
            dffloers = pl.read_csv(floersmultizonefilename).filter((pl.col("vel_outer") - vel_outer).abs() < 0.5)
            for row in dffloers.iter_rows(named=True):
                print(f"  ARTIS cell vel_outer: {vel_outer}, Floersfile: {row['vel_outer']}")
                print(f"  ARTIS cell Te: {T_e}, Floersfile: {row['Te']}")
                floers_levelpops = np.array(list(row.values())[4:], dtype=float)
                if len(dfpopthision["level"]) < len(floers_levelpops):
                    floers_levelpops = floers_levelpops[: len(dfpopthision["level"])]
                floers_levelnums = list(range(len(floers_levelpops)))
                floers_levelpop_values = floers_levelpops * (dfpopthision["n_NLTE"].sum() / sum(floers_levelpops))

    return floers_levelnums, floers_levelpop_values


def get_config_labels(configlist: Sequence[str]) -> list[str]:
    """Return one LaTeX label for the configuration of each level.

    A level that keeps the configuration of the level before it shows a mark and its term only. The
    axis labels then stay short.
    """
    configtexlist = [at.nltepops.texifyconfiguration(configlist[0])]
    for prevconfig, config in itertools.pairwise(configlist):
        if config.rsplit("_", maxsplit=1)[0] == prevconfig.rsplit("_", maxsplit=1)[0]:
            configtexlist.append('" ' + at.nltepops.texifyterm(config.rsplit("_", maxsplit=1)[1]))
        else:
            configtexlist.append(at.nltepops.texifyconfiguration(config))

    return configtexlist


def set_level_xticks(
    ax: mplax.Axes, levelindices: "pl.Series", configtexlist: Sequence[str], xmode: str, *, lastsubplot: bool
) -> None:
    """Put one tick at each level. The lowest subplot alone shows the names of the configurations."""
    if xmode == "config":
        ax.set_xticks(levelindices)
        if lastsubplot:
            ax.set_xticklabels(configtexlist, rotation=60, horizontalalignment="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels("" for _ in configtexlist)
    elif xmode == "none":
        ax.set_xticklabels("" for _ in configtexlist)


def print_top_radiative_decays(ion_data: dict[str, t.Any], dfpopthision: pl.DataFrame, maxlevel_ion: int) -> None:
    """Print the transitions that emit most strongly from the levels that the plot shows."""
    if "upper" not in ion_data["transitions"].collect_schema().names():
        return

    dftrans = ion_data["transitions"].filter(pl.col("upper") <= maxlevel_ion).collect()
    if dftrans.is_empty():
        return

    dftrans = dftrans.join(
        dfpopthision.select("level", "n_NLTE").with_columns(pl.col("level").cast(pl.Int32)),
        how="left",
        left_on="upper",
        right_on="level",
        coalesce=True,
    ).with_columns(
        emissionstrength=pl
        .when(pl.col("n_NLTE").is_not_null())
        .then(pl.col("n_NLTE") * pl.col("A") * pl.col("epsilon_trans_ev"))
        .otherwise(0)
    )

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
        at.plottools.get_next_color(ax)
        if floers_levelpop_values is not None:
            assert floers_levelnums is not None
            ax.plot(
                floers_levelnums,
                floers_levelpop_values / dfpopthision["n_LTE_T_e_normed"].to_numpy(),
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

    lte_columns: list[tuple[str, float]] = [("n_LTE_T_e", T_e)]
    if not args.hide_lte_tr:
        lte_columns.append(("n_LTE_T_R", T_R))

    dfpopthision = at.nltepops.add_lte_pops(dfpopthision, adata, lte_columns, noprint=False, maxlevel=args.maxlevel)

    if args.maxlevel >= 0:
        dfpopthision = dfpopthision.filter(pl.col("level") <= args.maxlevel)

    ionpopulation = float(dfpopthision["n_NLTE"].sum())
    ionkey = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
    ionpopulation_fromest = estimators.get((timestep, modelgridindex), {}).get(f"nnion_{ionkey}", 0.0)

    maxlevel_ion = dfpopthision["level"].max()
    assert isinstance(maxlevel_ion, int)
    levelnames = ion_data["levels"]["levelname"].to_list()
    configlist = levelnames[: maxlevel_ion + 1]
    configtexlist = get_config_labels(configlist)

    dfpopthision = dfpopthision.with_columns(
        # a level name that ends in "o" in front of the term is a level of odd parity
        parity=pl.Series([
            1 if (level != -1 and levelnames[int(level)].split("[")[0][-1] == "o") else 0
            for level in dfpopthision["level"]
        ]),
        config=pl.Series([configlist[level] for level in dfpopthision["level"]]),
        texname=pl.Series([configtexlist[level] for level in dfpopthision["level"]]),
    )

    set_level_xticks(
        ax, ion_data["levels"]["levelindex"][: maxlevel_ion + 1], configtexlist, args.x, lastsubplot=bool(lastsubplot)
    )

    print(
        f"{at.get_elsymbol(atomic_number)} {at.roman_numerals[ion_stage]} has a summed "
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
        dfpopthision, atomic_number, ion_stage, modelpath, T_e, modelgridindex
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
        at.get_ionstring(atomic_number, ion_stage, style="chargelatex"),
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


def make_plot_populations_with_time_or_velocity(modelpaths: Sequence[Path | str], args: argparse.Namespace) -> None:
    """Plot how selected level populations vary with time or velocity, and save the figure."""
    font = {"size": 18}
    mpl.rc("font", **font)

    ionlevels = args.levels

    Z = at.get_atomic_number(args.elements[0])
    ion_stage = int(args.ion_stages[0])

    adata = at.atomic.get_levels(modelpaths[0], get_transitions=True)

    ion_data = adata.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage)).row(0, named=True)
    levelconfignames = ion_data["levels"]["levelname"].to_list()
    # levelconfignames = [at.nltepops.texifyconfiguration(name) for name in levelconfignames]

    if args.timedayslist:
        rows = len(args.timedayslist)
        timedayslist = args.timedayslist
        args.subplots = True
    else:
        rows = 1
        timedayslist = [at.get_timestep_time(modelpaths[0], ts) for ts in range(args.timestepmin, args.timestepmax + 1)]
        args.subplots = False

    cols = 1
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=True,
        figsize=get_figsize(args, rows=rows, cols=cols, aspect=0.85, offset=0.0),
        tight_layout={"pad": 2.0, "w_pad": 0.2, "h_pad": 0.2},
    )

    if args.subplots:
        ax = ax.flatten()

    for plotnumber, timedays in enumerate(timedayslist):
        axis = ax[plotnumber] if args.subplots else ax
        assert isinstance(axis, mplax.Axes)
        plot_populations_with_time_or_velocity(
            axis, modelpaths, timedays, ion_stage, ionlevels, Z, levelconfignames, args=args
        )

    # the axis label size comes from the artistools matplotlibrc
    labelfontsize = None
    if args.x == "time":
        xlabel = "Time Since Explosion [days]"
    elif args.x == "velocity":
        xlabel = r"Zone outer velocity [km s$^{-1}$]"
    ylabel = r"Level population [cm$^{-3}$]"

    at.plottools.set_axis_labels(fig, ax, xlabel, ylabel, labelfontsize, args)
    if args.subplots:
        for plotnumber, axis in enumerate(ax):
            axis.set_yscale("log")
            if args.timedayslist:
                ymin, _ = axis.get_ylim()
                _, xmax = axis.get_xlim()
                axis.text(xmax * 0.85, ymin * 50, f"{args.timedayslist[plotnumber]} days")
        at.plottools.set_legend(ax[0], args, loc="best", frameon=True, fontsize="x-small", ncol=1)
    else:
        assert isinstance(ax, mplax.Axes)
        at.plottools.set_legend(ax, args, loc="best", frameon=True, fontsize="x-small", ncol=1)
        ax.set_yscale("log")

    title = f"Z={Z}, ion_stage={ion_stage}"
    if args.x == "time":
        title += f", mgi = {args.modelgridindex[0]}"
    elif args.x == "velocity":
        title += f", {timedayslist} days"
    at.plottools.set_plot_title(at.plottools.iter_axes(ax)[-1], title, args)

    at.plottools.set_axis_properties(ax, args)

    outputfilename = str(args.outputfile).format(elsymbol=at.get_elsymbol(Z))
    save_figure(fig, outputfilename, format="pdf", show=args.show)


def plot_populations_with_time_or_velocity(
    ax: mplax.Axes,
    modelpaths: Sequence[Path | str],
    timedays: float,
    ion_stage: int,
    ionlevels: list[int],
    Z: int,
    levelconfignames: list[str | int],
    args: argparse.Namespace,
) -> None:
    """Plot one series per level, against time or velocity as selected by args.x."""
    if args.x == "time":
        timesteps = list(range(args.timestepmin, args.timestepmax + 1))

        if not args.modelgridindex:
            print("Please specify modelgridindex")
            sys.exit(1)

        modelgridindex_list = [int(args.modelgridindex[0])] * len(timesteps)

    if args.x == "velocity":
        modeldata = at.inputmodel.get_modeldata(modelpaths[0])[0].collect()
        velocity = modeldata["vel_r_max_kmps"]
        modelgridindex_list = [mgi for mgi, _ in enumerate(velocity)]

        timesteps = [at.get_timestep_of_timedays(modelpaths[0], timedays)] * len(modelgridindex_list)

    markers = ["o", "x", "^", "s", "8"]
    for modelnumber, modelpath in enumerate(modelpaths):
        # modelname = at.get_model_name(modelpath)

        populations = {}
        # populationsLTE = {}

        for timestep, mgi in zip(timesteps, modelgridindex_list, strict=False):
            dfpop = at.nltepops.read_files(modelpath, timestep=timestep, modelgridindex=mgi)
            if dfpop.is_empty():
                continue
            timesteppops = dfpop.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))
            if timesteppops.is_empty():
                continue
            # setdefault keeps the first row for a duplicated level, matching the .item(0) this replaces
            pop_of_level: dict[int, float] = {}
            for level, n_nlte in zip(timesteppops["level"], timesteppops["n_NLTE"], strict=True):
                pop_of_level.setdefault(level, n_nlte)
            for ionlevel in ionlevels:
                populations[timestep, ionlevel, mgi] = pop_of_level[ionlevel]
                # populationsLTE[(timestep, ionlevel)] = (timesteppops.loc[timesteppops['level']
                #                                                          == ionlevel]['n_LTE'].values[0])

        for ionlevel in ionlevels:
            plottimesteps = [ts for ts, level, _mgi in populations if level == ionlevel]
            timedayslist = [at.get_timestep_time(modelpath, ts) for ts in plottimesteps]
            plotpopulations = np.array([
                populations[ts, level, mgi] for ts, level, mgi in populations if level == ionlevel
            ])
            # plotpopulationsLTE = np.array([float(populationsLTE[ts, level]) for ts, level in populationsLTE.keys()
            #                             if level == ionlevel])
            linelabel = str(levelconfignames[ionlevel])
            # linelabel = f'level {ionlevel} {modelname}'

            if args.x == "time":
                ax.plot(timedayslist, plotpopulations, marker=markers[modelnumber], label=linelabel)
            elif args.x == "velocity":
                plotvelocities = [float(velocity[mgi]) for _ts, level, mgi in populations if level == ionlevel]
                ax.plot(plotvelocities, plotpopulations, marker=markers[modelnumber], label=linelabel)
            # plt.plot(timedayslist, plotpopulationsLTE, marker=markers[modelnumber+1],
            #          label=f'level {ionlevel} {modelname} LTE')


def make_singletimestep_plot(
    modelpath: Path,
    atomic_number: int,
    ion_stages_displayed: list[int] | None,
    mgilist: Sequence[int],
    timestep: int,
    args: argparse.Namespace,
) -> None:
    """Plot level populations for chosens ions of an element in a cell and timestep of an ARTIS model."""
    modelname = at.get_model_name(modelpath)
    adata = at.atomic.get_levels(
        modelpath,
        get_transitions=args.gettransitions,
        derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms"],
    )

    time_days = at.get_timestep_time(modelpath, timestep)
    modelname = at.get_model_name(modelpath)

    dfpop = at.nltepops.read_files(modelpath, timestep=timestep, modelgridindex=mgilist[0])

    if dfpop.is_empty():
        print(f"No NLTE population data for modelgrid cell {mgilist[0]} timestep {timestep}")
        return

    dfpop = dfpop.filter(pl.col("Z") == atomic_number)

    # top_ion = 9999
    max_ion_stage = dfpop["ion_stage"].max()

    assert isinstance(max_ion_stage, int)
    if dfpop.filter(pl.col("ion_stage") == max_ion_stage).height == 1:  # single-level ion, so skip it
        max_ion_stage -= 1

    ion_stage_list = sorted([
        i
        for i in dfpop["ion_stage"].unique()
        if i <= max_ion_stage and (ion_stages_displayed is None or i in ion_stages_displayed)
    ])

    subplotheight = 2.4 / 6 if args.x == "config" else 1.8 / 6

    nrows = len(ion_stage_list) * len(mgilist)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharex=False,
        figsize=get_figsize(args, rows=nrows, aspect=subplotheight, offset=0.0),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    if nrows == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

    assert mgilist

    # invariant to the cell loop, so read the estimators and the model once instead of once per cell
    estimators = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=list(mgilist))
    lzmodeldata, _ = at.inputmodel.get_modeldata(modelpath, derived_cols="vel_r_mid")
    velocity_kmps_of_mgi = {
        mgi: vel_r_mid / km_to_cm
        for mgi, vel_r_mid in lzmodeldata
        .filter(pl.col("modelgridindex").is_in(mgilist))
        .select(["modelgridindex", "vel_r_mid"])
        .collect()
        .iter_rows()
    }

    elsymbol = at.get_elsymbol(atomic_number)

    for mgilistindex, modelgridindex in enumerate(mgilist):
        mgifirstaxindex = mgilistindex
        mgilastaxindex = mgilistindex + len(ion_stage_list) - 1

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
            print(f"WARNING: No estimator data. Setting T_e = T_R = {args.exc_temperature} K, nne and W unknown")
            T_e = args.exc_temperature
            T_R = args.exc_temperature
            # only used for display in the subplot title, so report them as unknown rather than inventing a value
            W = math.nan
            nne = math.nan

        dfpop = at.nltepops.read_files(modelpath, timestep=timestep, modelgridindex=modelgridindex)

        if dfpop.is_empty():
            print(f"No NLTE population data for modelgrid cell {modelgridindex} timestep {timestep}")
            return

        dfpop = dfpop.filter(pl.col("Z") == atomic_number)

        # top_ion = 9999
        max_ion_stage = dfpop["ion_stage"].max()

        assert isinstance(max_ion_stage, int)
        if dfpop.filter(pl.col("ion_stage") == max_ion_stage).height == 1:  # single-level ion, so skip it
            max_ion_stage -= 1

        subplot_title = modelname
        if len(subplot_title) > 10:
            subplot_title += "\n"
        subplot_title += f" {velocity_kmps_of_mgi[modelgridindex]:.0f} km/s at"

        try:
            time_days = at.get_timestep_time(modelpath, timestep)
        except FileNotFoundError:
            time_days = 0
            subplot_title += f" timestep {timestep:d}"
        else:
            subplot_title += f" {time_days:.0f}d"
        subplot_title += rf" (Te={T_e:.0f} K, nne={nne:.1e} cm$^{{-3}}$, T$_R$={T_R:.0f} K, W={W:.1e})"

        at.plottools.set_plot_title(axes[mgifirstaxindex], subplot_title, args)

        for ax, ion_stage in zip(axes[mgifirstaxindex : mgilastaxindex + 1], ion_stage_list, strict=False):
            lastsubplot = modelgridindex == mgilist[-1] and ion_stage == ion_stage_list[-1]
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
        for ax in reversed(at.plottools.iter_axes(axes))
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

    at.plottools.set_axis_properties(axes, args)
    # after set_axis_properties, which turns the automatic minor ticks on: a level index axis wants one
    # minor tick for each level, thus it keeps its own locator
    for ax in axes:
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

    outputfilename = str(args.outputfile).format(
        elsymbol=at.get_elsymbol(atomic_number), cell=mgilist[0], timestep=timestep, time_days=time_days
    )
    save_figure(fig, outputfilename, format="pdf", show=args.show)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("elements", nargs="*", default=["Fe"], help="List of elements to plot")

    addarg_modelpath(parser, default=Path())

    # arg to give multiple model paths - can use for x axis = time but breaks other plots
    # parser.add_argument('-modelpath', default=[Path('.')], nargs='*', type=Path,
    #                     help='Paths to ARTIS folders')

    timegroup = parser.add_mutually_exclusive_group()
    timegroup.add_argument("-timedays", "-time", "-t", help="Time in days to plot")

    timegroup.add_argument("-timedayslist", nargs="+", help="List of times in days for time sequence subplots")

    timegroup.add_argument("-timestep", "-ts", type=int, help="Timestep number to plot")

    cellgroup = parser.add_mutually_exclusive_group()
    # a mutually exclusive group, thus the flags are spelled out rather than taken from addarg_modelgridindex
    cellgroup.add_argument(
        "-modelgridindex", "-cell", "-mgi", default=[], help="Plotted model grid cell, or a range e.g. 3-7"
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

    addarg_figscale(parser, figscaledefault=1.4)

    parser.add_argument(
        "--departuremode", action="store_true", help="Show departure coefficients instead of populations"
    )

    parser.add_argument("--gettransitions", action="store_true", help="Show the most significant transitions")

    parser.add_argument("--plotrefdata", action="store_true", help="Show reference data")

    parser.add_argument("--hide-lte-tr", action="store_true", help="Hide LTE populations at T=T_R")

    addarg_notitle(parser)

    addarg_nolegend(parser)
    addarg_show(parser)
    addarg_quiet(parser)

    parser.add_argument(
        "-labelfontsize",
        type=float,
        default=None,
        help="Font size of the tick labels. The default comes from the artistools matplotlibrc.",
    )

    addarg_axislimits(parser)

    # no default here: which one applies depends on -x, so main chooses it when resolving the path
    addarg_outputfile(parser, helptext="path/filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS non-LTE populations."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    at.set_mpl_style()
    modelpath = args.modelpath
    if args.x in {"time", "velocity"}:
        args.modelpath = at.normalize_path_list(args.modelpath)

        if not args.ion_stages:
            at.exit_with_error("specify an ion stage with -ion_stages, e.g. -ion_stages 2")

        if not args.levels:
            at.exit_with_error("specify the levels to plot with -levels, e.g. -levels 0 1 2")

    if args.timedays:
        # a command line gives a string, and a keyword argument of the API gives a number
        if "-" in str(args.timedays):
            args.timestepmin, args.timestepmax, _, _ = at.get_time_range(
                modelpath, timedays_range_str=str(args.timedays)
            )
        else:
            timestep = at.get_timestep_of_timedays(modelpath, args.timedays)
            args.timestep = timestep
            args.timestepmin, args.timestepmax = timestep, timestep
    elif args.timedayslist:
        print(args.timedayslist)
    elif args.timestep is not None:
        args.timestepmin, args.timestepmax, _, _ = at.get_time_range(modelpath, timestep_range_str=str(args.timestep))
    elif args.x in {"time", "velocity"}:
        args.timestepmin, args.timestepmax, _, _ = at.get_time_range(modelpath, timemin=0, timemax=math.inf)
    else:
        exit_with_error(
            "no time given. Use -timedays or -timestep. A model of more than one cell also needs -modelgridindex"
        )

    args.outputfile = at.resolve_outputfile(
        args.outputfile, defaultoutputfile_timeorvelocity if args.x in {"time", "velocity"} else defaultoutputfile
    )

    ion_stages_permitted = at.parse_range_list(args.ion_stages) if args.ion_stages else None

    if isinstance(args.modelgridindex, str | int):
        args.modelgridindex = [args.modelgridindex]

    if isinstance(args.elements, str):
        args.elements = [args.elements]

    if isinstance(args.velocity, float | int):
        args.velocity = [args.velocity]

    cellargs = args.modelgridindex if isinstance(args.modelgridindex, list) else [args.modelgridindex]
    mgilist = [mgi for cellarg in cellargs for mgi in at.parse_range_list(str(cellarg))]
    mgilist.extend(
        mgi
        for mgi in [at.inputmodel.get_mgi_of_velocity_kms(modelpath, vel) for vel in args.velocity]
        if mgi is not None
    )
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
        make_plot_populations_with_time_or_velocity(modelpaths=args.modelpath, args=args)
        return

    for el_in in args.elements:
        try:
            atomic_number = int(el_in)
            elsymbol = at.get_elsymbol(atomic_number)
        except ValueError:
            elsymbol = el_in
            atomic_number = at.get_atomic_number(el_in)
            if atomic_number < 1:
                print(f"Could not find element '{elsymbol}'")

        for timestep in range(args.timestepmin, args.timestepmax + 1):
            make_singletimestep_plot(modelpath, atomic_number, ion_stages_permitted, mgilist, timestep, args)


if __name__ == "__main__":
    run_subcommand("plotnltepops")
