#!/usr/bin/env python3
"""Artistools - NLTE population related functions."""

import argparse
import contextlib
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import ticker

import artistools as at

defaultoutputfile = "plotnlte_{elsymbol}_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf"


def annotate_emission_line(ax: mplax.Axes, y: float, upperlevel: int, lowerlevel: int, label: str) -> None:
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
    dfpopthision: pd.DataFrame,
    annotatelines: bool,
) -> None:
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
                    ionstr = at.get_ionstring(atomic_number, ion_stage, style="chargelatex")
                    ax.plot(
                        levelnums,
                        depcoeffs,
                        linewidth=1.5,
                        color=color,
                        label=f"{ionstr} CHIANTI NLTE{bbstr}",
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
    dfpopthision: pd.DataFrame, atomic_number: int, ion_stage: int, modelpath, T_e: float, modelgridindex: int
) -> tuple[list[int] | None, list[float] | None]:
    floers_levelnums, floers_levelpop_values = None, None

    # comparison to Andeas Floers's NLTE pops for Shingles et al. (2022)
    if atomic_number == 26 and ion_stage in {2, 3}:
        floersfilename = "andreas_level_populations_fe2.txt" if ion_stage == 2 else "andreas_level_populations_fe3.txt"
        if Path(modelpath / floersfilename).is_file():
            print(f"reading {floersfilename}")
            dffloers_levelpops = pd.read_csv(modelpath / floersfilename, comment="#", sep=r"\s+")
            # floers_levelnums = floers_levelpops['index'].values - 1
            dffloers_levelpops = dffloers_levelpops.sort_values(by="energypercm")
            floers_levelnums = list(range(len(dffloers_levelpops)))
            floers_levelpop_values = dffloers_levelpops["frac_ionpop"].to_numpy() * dfpopthision["n_NLTE"].sum()

        floersmultizonefilename = None
        if modelpath.stem.startswith("w7_"):
            if "workfn" not in modelpath.parts[-1]:
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
            modeldata, _ = at.inputmodel.get_modeldata_pandas(modelpath)  # TODO: move into modelpath loop
            vel_outer = modeldata.iloc[modelgridindex].vel_r_max_kmps
            print(f"  reading {floersmultizonefilename}", vel_outer, T_e)
            dffloers = pd.read_csv(floersmultizonefilename)
            for _, row in dffloers.iterrows():
                if abs(row["vel_outer"] - vel_outer) < 0.5:
                    print(f"  ARTIS cell vel_outer: {vel_outer}, Floersfile: {row['vel_outer']}")
                    print(f"  ARTIS cell Te: {T_e}, Floersfile: {row['Te']}")
                    floers_levelpops = row.to_numpy()[4:]
                    if len(dfpopthision["level"]) < len(floers_levelpops):
                        floers_levelpops = floers_levelpops[: len(dfpopthision["level"])]
                    floers_levelnums = list(range(len(floers_levelpops)))
                    floers_levelpop_values = floers_levelpops * (dfpopthision["n_NLTE"].sum() / sum(floers_levelpops))

    return floers_levelnums, floers_levelpop_values


def make_ionsubplot(
    ax,
    modelpath,
    atomic_number,
    ion_stage,
    dfpop,
    adata,
    estimators,
    T_e,
    T_R,
    modelgridindex,
    timestep,
    args,
    lastsubplot,
) -> None:
    """Plot the level populations the specified ion, cell, and timestep."""
    ionstr = at.get_ionstring(atomic_number, ion_stage, style="chargelatex")
    ion_data = adata.filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage)).row(0, named=True)

    dfpopthision = dfpop.query(
        "modelgridindex == @modelgridindex and timestep == @timestep "
        "and Z == @atomic_number and ion_stage == @ion_stage",
        inplace=False,
    ).copy()

    lte_columns: list[tuple[str, float]] = [("n_LTE_T_e", T_e)]
    if not args.hide_lte_tr:
        lte_columns.append(("n_LTE_T_R", T_R))

    dfpopthision = at.nltepops.add_lte_pops(dfpopthision, adata, lte_columns, noprint=False, maxlevel=args.maxlevel)

    if args.maxlevel >= 0:
        dfpopthision = dfpopthision.query("level <= @args.maxlevel")

    ionpopulation = dfpopthision["n_NLTE"].sum()
    ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
    ionpopulation_fromest = estimators[timestep, modelgridindex].get(f"nnion_{ionstr}", 0.0)

    dfpopthision["parity"] = [
        1 if (row.level != -1 and ion_data["levels"]["levelname"].item(int(row.level)).split("[")[0][-1] == "o") else 0
        for _, row in dfpopthision.iterrows()
    ]

    configlist = ion_data["levels"]["levelname"][: max(dfpopthision.level) + 1]

    configtexlist = [at.nltepops.texifyconfiguration(configlist[0])]
    for i in range(1, len(configlist)):
        prevconfignoterm = configlist[i - 1].rsplit("_", maxsplit=1)[0]
        confignoterm = configlist[i].rsplit("_", maxsplit=1)[0]
        if confignoterm == prevconfignoterm:
            configtexlist.append('" ' + at.nltepops.texifyterm(configlist[i].rsplit("_", maxsplit=1)[1]))
        else:
            configtexlist.append(at.nltepops.texifyconfiguration(configlist[i]))

    dfpopthision["config"] = [configlist[level] for level in dfpopthision.level]
    dfpopthision["texname"] = [configtexlist[level] for level in dfpopthision.level]

    if args.x == "config":
        # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=100))
        ax.set_xticks(ion_data["levels"]["levelindex"][: max(dfpopthision.level) + 1])

        if not lastsubplot:
            ax.set_xticklabels("" for _ in configtexlist)
        else:
            ax.set_xticklabels(
                configtexlist,
                # fontsize=8,
                rotation=60,
                horizontalalignment="right",
                rotation_mode="anchor",
            )
    elif args.x == "none":
        ax.set_xticklabels("" for _ in configtexlist)

    print(
        f"{at.get_elsymbol(atomic_number)} {at.roman_numerals[ion_stage]} has a summed "
        f"level population of {ionpopulation:.1f} (from estimator file ion pop = {ionpopulation_fromest})"
    )

    lte_scalefactor = (
        # scale to match the ground state populations
        float(dfpopthision["n_NLTE"].iloc[0] / dfpopthision["n_LTE_T_e"].iloc[0])
        if args.departuremode
        # else scale to match the ion population
        else float(ionpopulation / dfpopthision["n_LTE_T_e"].sum())
    )

    dfpopthision = dfpopthision.eval("n_LTE_T_e_normed = n_LTE_T_e * @x", local_dict={"x": lte_scalefactor}).eval(
        "departure_coeff = n_NLTE / n_LTE_T_e_normed"
    )

    pd.set_option("display.max_columns", 150)
    if len(dfpopthision) < 30:
        # print(dfpopthision[
        #     ['Z', 'ion_stage', 'level', 'config', 'departure_coeff', 'texname']].to_string(index=False))
        print(
            dfpopthision.loc[
                :, [c not in {"timestep", "modelgridindex", "Z", "parity", "texname"} for c in dfpopthision.columns]
            ].to_string(index=False)
        )

    maxlevel = max(dfpopthision["level"])
    dftrans: pl.DataFrame | None = None
    if "upper" in ion_data["transitions"].collect_schema().names():
        dftrans = ion_data["transitions"].filter(pl.col("upper") <= maxlevel).collect()
        if dftrans is not None and dftrans.is_empty():
            dftrans = None

    if dftrans is not None:
        dftrans = dftrans.join(
            pl.from_pandas(dfpopthision[["level", "n_NLTE"]]).with_columns(pl.col("level").cast(pl.Int32)),
            how="left",
            left_on="upper",
            right_on="level",
            coalesce=True,
        )
        dftrans = dftrans.with_columns(
            emissionstrength=pl.when(pl.col("n_NLTE").is_not_null())
            .then(pl.col("n_NLTE") * pl.col("A") * pl.col("epsilon_trans_ev"))
            .otherwise(0)
        )

        dftrans = dftrans.sort(by="emissionstrength", descending=True)

        print("\nTop radiative decays")
        print(dftrans.head(20))

    ax.set_yscale("log")

    floers_levelnums, floers_levelpop_values = get_floers_data(
        dfpopthision, atomic_number, ion_stage, modelpath, T_e, modelgridindex
    )

    if args.departuremode:
        ax.axhline(y=1.0, color="0.7", linestyle="dashed", linewidth=1.5)
        ax.set_ylabel("Departure coefficient")

        ycolumnname = "departure_coeff"

        ax._get_lines.get_next_color()  # noqa: SLF001  # skip one color, since T_e is not plotted in departure mode
        if floers_levelnums is not None:
            ax.plot(
                floers_levelnums,
                floers_levelpop_values / dfpopthision["n_LTE_T_e_normed"],
                linewidth=1.5,
                label=f"{ionstr} Flörs NLTE",
                linestyle="None",
                marker="*",
            )
    else:
        ax.set_ylabel(r"Population density (cm$^{-3}$)")

        ycolumnname = "n_NLTE"

        ax.plot(
            dfpopthision["level"],
            dfpopthision["n_LTE_T_e_normed"],
            linewidth=1.5,
            label=f"{ionstr} LTE T$_e$ = {T_e:.0f} K",
            linestyle="None",
            marker="*",
        )

        if floers_levelnums is not None:
            ax.plot(
                floers_levelnums,
                floers_levelpop_values,
                linewidth=1.5,
                label=f"{ionstr} Flörs NLTE",
                linestyle="None",
                marker="*",
            )

        if not args.hide_lte_tr:
            lte_scalefactor = float(ionpopulation / dfpopthision["n_LTE_T_R"].sum())
            dfpopthision = dfpopthision.eval("n_LTE_T_R_normed = n_LTE_T_R * @lte_scalefactor")
            ax.plot(
                dfpopthision["level"],
                dfpopthision["n_LTE_T_R_normed"],
                linewidth=1.5,
                label=f"{ionstr} LTE T$_R$ = {T_R:.0f} K",
                linestyle="None",
                marker="*",
            )

    ax.plot(
        dfpopthision["level"],
        dfpopthision[ycolumnname],
        linewidth=1.5,
        linestyle="None",
        marker="x",
        label=f"{ionstr} ARTIS NLTE",
        color="black",
    )

    dfpopthisionoddlevels = dfpopthision.query("parity==1")
    if not dfpopthisionoddlevels.level.empty:
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

    if args.plotrefdata:
        plot_reference_data(
            ax, atomic_number, ion_stage, estimators[timestep, modelgridindex], dfpopthision, annotatelines=True
        )


def make_plot_populations_with_time_or_velocity(modelpaths: list[Path | str], args: argparse.Namespace) -> None:
    font = {"size": 18}
    mpl.rc("font", **font)

    ionlevels = args.levels

    Z = at.get_atomic_number(args.elements[0])
    ion_stage = int(args.ion_stages[0])

    adata = at.atomic.get_levels(modelpaths[0], get_transitions=True)
    ion_data = adata.query("Z == @Z and ion_stage == @ion_stage").iloc[0]
    levelconfignames = ion_data["levels"]["levelname"]
    # levelconfignames = [at.nltepops.texifyconfiguration(name) for name in levelconfignames]

    if args.timedayslist:
        rows = len(args.timedayslist)
        timedayslist = args.timedayslist
        args.subplots = True
    else:
        rows = 1
        timedayslist = [at.get_timestep_time(modelpaths[0], ts) for ts in range(args.timestepmin, args.timestepmax)]
        args.subplots = False

    cols = 1
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=True,
        figsize=(at.get_config()["figwidth"] * 2 * cols, at.get_config()["figwidth"] * 0.85 * rows),
        tight_layout={"pad": 2.0, "w_pad": 0.2, "h_pad": 0.2},
    )
    assert isinstance(ax, np.ndarray)
    if args.subplots:
        ax = ax.flatten()

    for plotnumber, timedays in enumerate(timedayslist):
        axis = ax[plotnumber] if args.subplots else ax
        assert isinstance(axis, mplax.Axes)
        plot_populations_with_time_or_velocity(
            axis, modelpaths, timedays, ion_stage, ionlevels, Z, levelconfignames, args
        )

    labelfontsize = 20
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
        ax[0].legend(loc="best", frameon=True, fontsize="x-small", ncol=1)
    else:
        assert isinstance(ax, mplax.Axes)
        ax.legend(loc="best", frameon=True, fontsize="x-small", ncol=1)
        ax.set_yscale("log")

    if not args.notitle:
        title = f"Z={Z}, ion_stage={ion_stage}"
        if args.x == "time":
            title += f", mgi = {args.modelgridindex[0]}"
        elif args.x == "velocity":
            title += f", {args.timedays} days"
        plt.title(title)

    at.plottools.set_axis_properties(ax, args)

    figname = f"plotnltelevelpopsZ{Z}.pdf"
    plt.savefig(Path(modelpaths[0]) / figname, format="pdf")
    print(f"Saved {figname}")


def plot_populations_with_time_or_velocity(
    ax: mplax.Axes,
    modelpaths: list[Path | str],
    timedays: float,
    ion_stage: int,
    ionlevels,
    Z: int,
    levelconfignames,
    args: argparse.Namespace,
) -> None:
    if args.x == "time":
        timesteps = list(range(args.timestepmin, args.timestepmax))

        if not args.modelgridindex:
            print("Please specify modelgridindex")
            sys.exit(1)

        modelgridindex_list = [int(args.modelgridindex[0])] * len(timesteps)

    if args.x == "velocity":
        modeldata, _ = at.inputmodel.get_modeldata_pandas(modelpaths[0])  # TODO: move into modelpath loop
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
            try:
                timesteppops = dfpop.loc[(dfpop["Z"] == Z) & (dfpop["ion_stage"] == ion_stage)]
            except KeyError:
                continue
            for ionlevel in ionlevels:
                populations[timestep, ionlevel, mgi] = timesteppops.loc[timesteppops["level"] == ionlevel][
                    "n_NLTE"
                ].to_numpy()[0]
                # populationsLTE[(timestep, ionlevel)] = (timesteppops.loc[timesteppops['level']
                #                                                          == ionlevel]['n_LTE'].values[0])

        for ionlevel in ionlevels:
            plottimesteps = np.array([ts for ts, level, mgi in populations if level == ionlevel])
            timedayslist = [at.get_timestep_time(modelpath, ts) for ts in plottimesteps]
            plotpopulations = np.array([
                float(populations[ts, level, mgi]) for ts, level, mgi in populations if level == ionlevel
            ])
            # plotpopulationsLTE = np.array([float(populationsLTE[ts, level]) for ts, level in populationsLTE.keys()
            #                             if level == ionlevel])
            linelabel = str(levelconfignames[ionlevel])
            # linelabel = f'level {ionlevel} {modelname}'

            if args.x == "time":
                ax.plot(timedayslist, plotpopulations, marker=markers[modelnumber], label=linelabel)
            elif args.x == "velocity":
                ax.plot(velocity, plotpopulations, marker=markers[modelnumber], label=linelabel)
            # plt.plot(timedayslist, plotpopulationsLTE, marker=markers[modelnumber+1],
            #          label=f'level {ionlevel} {modelname} LTE')


def make_plot(modelpath, atomic_number, ion_stages_displayed, mgilist, timestep, args: argparse.Namespace) -> None:
    """Plot level populations for chosens ions of an element in a cell and timestep of an ARTIS model."""
    modelname = at.get_model_name(modelpath)
    adata = at.atomic.get_levels_polars(
        modelpath,
        get_transitions=args.gettransitions,
        derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms"],
    )

    time_days = at.get_timestep_time(modelpath, timestep)
    modelname = at.get_model_name(modelpath)

    dfpop = at.nltepops.read_files(modelpath, timestep=timestep, modelgridindex=mgilist[0])

    if dfpop.empty:
        print(f"No NLTE population data for modelgrid cell {mgilist[0]} timestep {timestep}")
        return

    dfpop = dfpop.query("Z == @atomic_number")

    # top_ion = 9999
    max_ion_stage = dfpop.ion_stage.max()

    if len(dfpop.query("ion_stage == @max_ion_stage")) == 1:  # single-level ion, so skip it
        max_ion_stage -= 1

    ion_stage_list = sorted([
        i
        for i in dfpop.ion_stage.unique()
        if i <= max_ion_stage and (ion_stages_displayed is None or i in ion_stages_displayed)
    ])

    subplotheight = 2.4 / 6 if args.x == "config" else 1.8 / 6

    nrows = len(ion_stage_list) * len(mgilist)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharex=False,
        figsize=(
            args.figscale * at.get_config()["figwidth"],
            args.figscale * at.get_config()["figwidth"] * subplotheight * nrows,
        ),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    if nrows == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

    prev_ion_stage = -1
    assert len(mgilist) > 0
    for mgilistindex, modelgridindex in enumerate(mgilist):
        mgifirstaxindex = mgilistindex
        mgilastaxindex = mgilistindex + len(ion_stage_list) - 1

        estimators = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)
        elsymbol = at.get_elsymbol(atomic_number)
        print(
            f"Plotting NLTE pops for {modelname} modelgridindex {modelgridindex}, timestep {timestep} (t={time_days}d)"
        )
        print(f"Z={atomic_number} {elsymbol}")

        if estimators:
            T_e = estimators[timestep, modelgridindex]["Te"]
            T_R = estimators[timestep, modelgridindex]["TR"]
            W = estimators[timestep, modelgridindex]["W"]
            nne = estimators[timestep, modelgridindex]["nne"]
            print(f"nne = {nne} cm^-3, T_e = {T_e} K, T_R = {T_R} K, W = {W}")
        else:
            print("WARNING: No estimator data. Setting T_e = T_R =  6000 K")
            T_e = args.exc_temperature
            T_R = args.exc_temperature

        dfpop = at.nltepops.read_files(modelpath, timestep=timestep, modelgridindex=modelgridindex).copy()

        if dfpop.empty:
            print(f"No NLTE population data for modelgrid cell {modelgridindex} timestep {timestep}")
            return

        dfpop = dfpop.query("Z == @atomic_number")

        # top_ion = 9999
        max_ion_stage = dfpop.ion_stage.max()

        if len(dfpop.query("ion_stage == @max_ion_stage")) == 1:  # single-level ion, so skip it
            max_ion_stage -= 1

        # timearray = at.get_timestep_times(modelpath)
        nne = estimators[timestep, modelgridindex]["nne"]
        W = estimators[timestep, modelgridindex]["W"]

        subplot_title = modelname
        if len(subplot_title) > 10:
            subplot_title += "\n"
        modeldata, _ = at.inputmodel.get_modeldata(modelpath, derived_cols="vel_r_mid")
        velocity_kmps = (
            modeldata.filter(pl.col("modelgridindex") == modelgridindex).select("vel_r_mid").collect().item() / 1e5
        )
        subplot_title += f" {velocity_kmps:.0f} km/s at"

        try:
            time_days = at.get_timestep_time(modelpath, timestep)
        except FileNotFoundError:
            time_days = 0
            subplot_title += f" timestep {timestep:d}"
        else:
            subplot_title += f" {time_days:.0f}d"
        subplot_title += rf" (Te={T_e:.0f} K, nne={nne:.1e} cm$^{{-3}}$, T$_R$={T_R:.0f} K, W={W:.1e})"

        if not args.notitle:
            axes[mgifirstaxindex].set_title(subplot_title, fontsize=10)

        for ax, ion_stage in zip(axes[mgifirstaxindex : mgilastaxindex + 1], ion_stage_list, strict=False):
            lastsubplot = modelgridindex == mgilist[-1] and ion_stage == ion_stage_list[-1]
            make_ionsubplot(
                ax,
                modelpath,
                atomic_number,
                ion_stage,
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

            # ax.annotate(ionstr, xy=(0.95, 0.96), xycoords='axes fraction',
            #             horizontalalignment='right', verticalalignment='top', fontsize=12)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

            ax.set_xlim(left=-1)
            if args.xmin is not None:
                ax.set_xlim(left=args.xmin)
            if args.xmax is not None:
                ax.set_xlim(right=args.xmax)
            if args.ymin is not None:
                ax.set_ylim(bottom=args.ymin)
            if args.ymax is not None:
                ax.set_ylim(top=args.ymax)

            if not args.nolegend and prev_ion_stage != ion_stage:
                ax.legend(loc="best", handlelength=1, frameon=True, numpoints=1, edgecolor="0.93", facecolor="0.93")

            prev_ion_stage = ion_stage

    if args.x == "index":
        axes[-1].set_xlabel(r"Level index")

    outputfilename = str(args.outputfile).format(
        elsymbol=at.get_elsymbol(atomic_number), cell=mgilist[0], timestep=timestep, time_days=time_days
    )
    fig.savefig(outputfilename, format="pdf")
    print(f"Saved {outputfilename}")
    plt.close()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("elements", nargs="*", default=["Fe"], help="List of elements to plot")

    parser.add_argument("-modelpath", default=Path(), type=Path, help="Path to ARTIS folder")

    # arg to give multiple model paths - can use for x axis = time but breaks other plots
    # parser.add_argument('-modelpath', default=[Path('.')], nargs='*', action=at.AppendPath,
    #                     help='Paths to ARTIS folders')

    timegroup = parser.add_mutually_exclusive_group()
    timegroup.add_argument("-timedays", "-time", "-t", help="Time in days to plot")

    timegroup.add_argument("-timedayslist", nargs="+", help="List of times in days for time sequence subplots")

    timegroup.add_argument("-timestep", "-ts", type=int, help="Timestep number to plot")

    cellgroup = parser.add_mutually_exclusive_group()
    cellgroup.add_argument("-modelgridindex", "-cell", nargs="?", default=[], help="Plotted modelgrid cell(s)")

    cellgroup.add_argument("-velocity", "-v", nargs="?", default=[], type=float, help="Specify cell by velocity")

    parser.add_argument("-exc-temperature", type=float, default=6000.0, help="Default if no estimator data")

    parser.add_argument(
        "-x", choices=["index", "config", "time", "velocity", "none"], default="index", help="Horizontal axis variable"
    )

    parser.add_argument("-ion_stages", help="Ion stage range, 1 is neutral, 2 is 1+")

    parser.add_argument(
        "-levels", type=int, nargs="+", help="Choose levels to plot"
    )  # currently only for x axis = time

    parser.add_argument("-maxlevel", default=-1, type=int, help="Maximum level to plot")

    parser.add_argument(
        "-figscale", type=float, default=1.6, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument(
        "--departuremode", action="store_true", help="Show departure coefficients instead of populations"
    )

    parser.add_argument("--gettransitions", action="store_true", help="Show the most significant transitions")

    parser.add_argument("--plotrefdata", action="store_true", help="Show reference data")

    parser.add_argument("--hide-lte-tr", action="store_true", help="Hide LTE populations at T=T_R")

    parser.add_argument("--notitle", action="store_true", help="Suppress the top title from the plot")

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument("-xmin", type=float, default=None, help="Plot range: x-axis")

    parser.add_argument("-xmax", type=float, default=None, help="Plot range: x-axis")

    parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-outputfile", "-o", type=Path, default=defaultoutputfile, help="path/filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS non-LTE populations."""
    if args is None:
        parser = argparse.ArgumentParser(description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    at.set_mpl_style()
    timestep = -1
    modelpath = args.modelpath
    if args.x in {"time", "velocity"}:
        args.modelpath = [args.modelpath]

        # if not args.timedays:
        #     print("Please specify time range with -timedays")
        #     sys.exit(1)
        if not args.ion_stages:
            print("Please specify ion_stage")
            sys.exit(1)
        if not args.levels:
            print("Please specify levels")
            sys.exit(1)

    if args.timedays:
        if "-" in args.timedays:
            args.timestepmin, args.timestepmax, _, _ = at.get_time_range(modelpath, timedays_range_str=args.timedays)
        else:
            timestep = at.get_timestep_of_timedays(modelpath, args.timedays)
            args.timestep = timestep
    elif args.timedayslist:
        print(args.timedayslist)
    elif args.timestep is not None:
        timestep = int(args.timestep)
    else:
        print("Please specify time with -timedays or -timestep")
        sys.exit(1)

    if Path(args.outputfile).is_dir():
        args.outputfile = Path(args.outputfile, defaultoutputfile)

    ion_stages_permitted = at.parse_range_list(args.ion_stages) if args.ion_stages else None

    if isinstance(args.modelgridindex, str):
        args.modelgridindex = [args.modelgridindex]

    if isinstance(args.elements, str):
        args.elements = [args.elements]

    if isinstance(args.velocity, float | int):
        args.velocity = [args.velocity]

    mgilist: list[int | None] = [int(mgi) for mgi in args.modelgridindex]
    mgilist.extend(at.inputmodel.get_mgi_of_velocity_kms(modelpath, vel) for vel in args.velocity)
    if not mgilist:
        mgilist.append(0)

    if args.x in {"time", "velocity"}:
        make_plot_populations_with_time_or_velocity(args.modelpath, args)
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

        make_plot(modelpath, atomic_number, ion_stages_permitted, mgilist, timestep, args)


if __name__ == "__main__":
    main()
