#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Functions for plotting artis estimators and internal structure.

Examples are temperatures, populations, heating/cooling rates.
"""

import argparse
import contextlib
import math
import typing as t
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at

colors_tab10: list[str] = list(plt.get_cmap("tab10")(np.linspace(0, 1.0, 10)))

# reserve colours for these elements
elementcolors = {
    "Fe": colors_tab10[0],
    "Ni": colors_tab10[1],
    "Co": colors_tab10[2],
}


def get_elemcolor(atomic_number: int | None = None, elsymbol: str | None = None) -> str | np.ndarray:
    """Get the colour of an element from the reserved color list (reserving a new one if needed)."""
    assert (atomic_number is None) != (elsymbol is None)
    if atomic_number is not None:
        elsymbol = at.get_elsymbol(atomic_number)
    assert elsymbol is not None
    # assign a new colour to this element if needed

    return elementcolors.setdefault(elsymbol, colors_tab10[len(elementcolors)])


def get_ylabel(variable: str) -> str:
    return at.estimators.get_variablelongunits(variable) or at.estimators.get_units_string(variable)


def plot_init_abundances(
    ax: plt.Axes,
    xlist: list[float],
    specieslist: list[str],
    mgilist: t.Sequence[float],
    modelpath: Path,
    seriestype: str,
    startfromzero: bool,
    args: argparse.Namespace,
    **plotkwargs,
) -> None:
    assert len(xlist) == len(mgilist)

    if seriestype == "initabundances":
        mergemodelabundata, _ = at.inputmodel.get_modeldata(modelpath, get_elemabundances=True)
    elif seriestype == "initmasses":
        mergemodelabundata = at.inputmodel.plotinitialcomposition.get_model_abundances_Msun_1D(modelpath)
    else:
        raise AssertionError

    if startfromzero:
        xlist = xlist.copy()
        xlist.insert(0, 0.0)

    for speciesstr in specieslist:
        splitvariablename = speciesstr.split("_")
        elsymbol = splitvariablename[0].strip("0123456789")
        atomic_number = at.get_atomic_number(elsymbol)
        if seriestype == "initabundances":
            ax.set_ylim(1e-20, 1.0)
            ax.set_ylabel("Initial mass fraction")
            valuetype = "X_"
        elif seriestype == "initmasses":
            ax.set_ylabel(r"Initial mass [M$_\odot$]")
            valuetype = "mass_X_"
        else:
            raise AssertionError

        ylist = []
        linelabel = speciesstr
        linestyle = "-"
        for modelgridindex in mgilist:
            if speciesstr.lower() in {"ni_56", "ni56", "56ni"}:
                yvalue = mergemodelabundata.loc[modelgridindex][f"{valuetype}Ni56"]
                linelabel = "$^{56}$Ni"
                linestyle = "--"
            elif speciesstr.lower() in {"ni_stb", "ni_stable"}:
                yvalue = (
                    mergemodelabundata.loc[modelgridindex][f"{valuetype}{elsymbol}"]
                    - mergemodelabundata.loc[modelgridindex]["X_Ni56"]
                )
                linelabel = "Stable Ni"
            elif speciesstr.lower() in {"co_56", "co56", "56co"}:
                yvalue = mergemodelabundata.loc[modelgridindex][f"{valuetype}Co56"]
                linelabel = "$^{56}$Co"
            elif speciesstr.lower() in {"fegrp", "ffegroup"}:
                yvalue = mergemodelabundata.loc[modelgridindex][f"{valuetype}Fegroup"]
            else:
                yvalue = mergemodelabundata.loc[modelgridindex][f"{valuetype}{elsymbol}"]
            ylist.append(yvalue)

        color = get_elemcolor(atomic_number=atomic_number)

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)

        if startfromzero:
            ylist.insert(0, ylist[0])

        ax.plot(xlist, ylist, linewidth=1.5, label=linelabel, linestyle=linestyle, color=color, **plotkwargs)

        # if args.yscale == 'log':
        #     ax.set_yscale('log')


def plot_average_ionisation_excitation(
    ax: plt.Axes,
    xlist: list[float],
    seriestype: str,
    params: t.Sequence[str],
    timestepslist: t.Sequence[t.Sequence[int]],
    mgilist: t.Sequence[int],
    estimators: pl.LazyFrame,
    modelpath: Path | str,
    startfromzero: bool,
    args=None,
    **plotkwargs,
) -> None:
    if seriestype == "averageexcitation":
        ax.set_ylabel("Average excitation [eV]")
    elif seriestype == "averageionisation":
        ax.set_ylabel("Average ion charge")
    else:
        raise ValueError

    if startfromzero:
        xlist = xlist.copy()
        xlist.insert(0, 0.0)

    arr_tdelta = at.get_timestep_times(modelpath, loc="delta")
    for paramvalue in params:
        print(f"Plotting {seriestype} {paramvalue}")
        if seriestype == "averageionisation":
            atomic_number = at.get_atomic_number(paramvalue)
        else:
            atomic_number = at.get_atomic_number(paramvalue.split(" ")[0])
            ion_stage = at.decode_roman_numeral(paramvalue.split(" ")[1])
        ylist = []
        if seriestype == "averageexcitation":
            print("  This will be slow!")
            for modelgridindex, timesteps in zip(mgilist, timestepslist):
                exc_ev_times_tdelta_sum = 0.0
                tdeltasum = 0.0
                for timestep in timesteps:
                    T_exc = (
                        estimators.filter(pl.col("timestep") == timestep)
                        .filter(pl.col("modelgridindex") == modelgridindex)
                        .select("Te")
                        .lazy()
                        .collect()
                        .item(0, 0)
                    )
                    exc_ev = at.estimators.get_averageexcitation(
                        modelpath, modelgridindex, timestep, atomic_number, ion_stage, T_exc
                    )
                    if exc_ev is not None:
                        exc_ev_times_tdelta_sum += exc_ev * arr_tdelta[timestep]
                        tdeltasum += arr_tdelta[timestep]
            if tdeltasum == 0.0:
                msg = f"ERROR: No excitation data found for {paramvalue}"
                raise ValueError(msg)
            ylist.append(exc_ev_times_tdelta_sum / tdeltasum if tdeltasum > 0 else math.nan)

        elif seriestype == "averageionisation":
            elsymb = at.get_elsymbol(atomic_number)
            if f"nnelement_{elsymb}" not in estimators.columns:
                msg = f"ERROR: No element data found for {paramvalue}"
                raise ValueError(msg)
            dfselected = (
                estimators.select(
                    cs.starts_with(f"nnion_{elsymb}_")
                    | cs.by_name(f"nnelement_{elsymb}")
                    | cs.by_name("modelgridindex")
                    | cs.by_name("timestep")
                    | cs.by_name("xvalue")
                    | cs.by_name("plotpointid")
                )
                .with_columns(pl.col(pl.Float32).fill_null(0.0))
                .collect()
                .join(
                    pl.DataFrame({"timestep": range(len(arr_tdelta)), "tdelta": arr_tdelta}).with_columns(
                        pl.col("timestep").cast(pl.Int32)
                    ),
                    on="timestep",
                    how="left",
                )
            )
            dfselected = dfselected.filter(pl.col(f"nnelement_{elsymb}") > 0.0)

            ioncols = [col for col in dfselected.columns if col.startswith(f"nnion_{elsymb}_")]
            ioncharges = [at.decode_roman_numeral(col.removeprefix(f"nnion_{elsymb}_")) - 1 for col in ioncols]
            ax.set_ylim(0.0, max(ioncharges) + 0.1)

            dfselected = dfselected.with_columns(
                (
                    pl.sum_horizontal([pl.col(ioncol) * ioncharge for ioncol, ioncharge in zip(ioncols, ioncharges)])
                    / pl.col(f"nnelement_{elsymb}")
                ).alias(f"averageionisation_{elsymb}")
            )

            series = (
                dfselected.group_by("plotpointid", maintain_order=True)
                .agg(pl.col(f"averageionisation_{elsymb}").mean(), pl.col("xvalue").mean())
                .lazy()
                .collect()
            )

            xlist = series["xvalue"].to_list()
            if startfromzero:
                xlist.insert(0, 0.0)

            ylist = series[f"averageionisation_{elsymb}"].to_list()

        color = get_elemcolor(atomic_number=atomic_number)

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)
        if startfromzero:
            ylist.insert(0, ylist[0])

        ax.plot(xlist, ylist, label=paramvalue, color=color, **plotkwargs)


def plot_levelpop(
    ax: plt.Axes,
    xlist: t.Sequence[int | float] | np.ndarray,
    seriestype: str,
    params: t.Sequence[str],
    timestepslist: t.Sequence[t.Sequence[int]],
    mgilist: t.Sequence[int | t.Sequence[int]],
    estimators: pl.LazyFrame | pl.DataFrame,
    modelpath: str | Path,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    if seriestype == "levelpopulation_dn_on_dvel":
        ax.set_ylabel("dN/dV [{}km$^{{-1}}$ s]")
        ax.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(ax.get_ylabel(), useMathText=True))
    elif seriestype == "levelpopulation":
        ax.set_ylabel("X$_{{i}}$ [{}/cm3]")
        ax.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(ax.get_ylabel(), useMathText=True))
    else:
        raise ValueError

    modeldata, _ = at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g", "volume"])

    adata = at.atomic.get_levels(modelpath)

    arr_tdelta = at.get_timestep_times(modelpath, loc="delta")
    for paramvalue in params:
        paramsplit = paramvalue.split(" ")
        atomic_number = at.get_atomic_number(paramsplit[0])
        ion_stage = at.decode_roman_numeral(paramsplit[1])
        levelindex = int(paramsplit[2])

        ionlevels = adata.query("Z == @atomic_number and ion_stage == @ion_stage").iloc[0].levels
        levelname = ionlevels.iloc[levelindex].levelname
        label = (
            f"{at.get_ionstring(atomic_number, ion_stage, style='chargelatex')} level {levelindex}:"
            f" {at.nltepops.texifyconfiguration(levelname)}"
        )

        print(f"plot_levelpop {label}")

        # level index query goes outside for caching granularity reasons
        dfnltepops = at.nltepops.read_files(
            modelpath, dfquery=f"Z=={atomic_number:.0f} and ion_stage=={ion_stage:.0f}"
        ).query("level==@levelindex")

        ylist = []
        for modelgridindex, timesteps in zip(mgilist, timestepslist):
            valuesum = 0
            tdeltasum = 0
            # print(f'modelgridindex {modelgridindex} timesteps {timesteps}')

            for timestep in timesteps:
                levelpop = (
                    dfnltepops.query(
                        "modelgridindex==@modelgridindex and timestep==@timestep and Z==@atomic_number"
                        " and ion_stage==@ion_stage and level==@levelindex"
                    )
                    .iloc[0]
                    .n_NLTE
                )

                valuesum += levelpop * arr_tdelta[timestep]
                tdeltasum += arr_tdelta[timestep]

            if seriestype == "levelpopulation_dn_on_dvel":
                deltav = modeldata.loc[modelgridindex].vel_r_max_kmps - modeldata.loc[modelgridindex].vel_r_min_kmps
                ylist.append(valuesum / tdeltasum * modeldata.loc[modelgridindex].volume / deltav)
            else:
                ylist.append(valuesum / tdeltasum)

        ylist.insert(0, ylist[0])

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, label=label, **plotkwargs)


def plot_multi_ion_series(
    ax: plt.Axes,
    startfromzero: bool,
    seriestype: str,
    ionlist: t.Sequence[str],
    estimators: pl.LazyFrame | pl.DataFrame,
    modelpath: str | Path,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    """Plot an ion-specific property, e.g., populations."""
    # if seriestype == 'populations':
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.10))

    plotted_something = False

    def get_iontuple(ionstr):
        if ionstr in at.get_elsymbolslist():
            return (at.get_atomic_number(ionstr), "ALL")
        if " " in ionstr:
            return (at.get_atomic_number(ionstr.split(" ")[0]), at.decode_roman_numeral(ionstr.split(" ")[1]))
        if ionstr.rstrip("-0123456789") in at.get_elsymbolslist():
            atomic_number = at.get_atomic_number(ionstr.rstrip("-0123456789"))
            return (atomic_number, ionstr)
        atomic_number = at.get_atomic_number(ionstr.split("_")[0])
        return (atomic_number, ionstr)

    # decoded into atomic number and parameter, e.g., [(26, 1), (26, 2), (26, 'ALL'), (26, 'Fe56')]
    iontuplelist = [get_iontuple(ionstr) for ionstr in ionlist]
    iontuplelist.sort()
    print(f"Subplot with ions: {iontuplelist}")

    missingions = set()
    try:
        if not args.classicartis:
            compositiondata = at.get_composition_data(modelpath)
            for atomic_number, ion_stage in iontuplelist:
                if (
                    not hasattr(ion_stage, "lower")
                    and not args.classicartis
                    and compositiondata.query(
                        "Z == @atomic_number & lowermost_ion_stage <= @ion_stage & uppermost_ion_stage >= @ion_stage"
                    ).empty
                ):
                    missingions.add((atomic_number, ion_stage))

    except FileNotFoundError:
        print("WARNING: Could not read an ARTIS compositiondata.txt file to check ion availability")
        for atomic_number, ion_stage in iontuplelist:
            ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
            if f"nnion_{ionstr}" not in estimators.columns:
                missingions.add((atomic_number, ion_stage))

    if missingions:
        print(f" Warning: Can't plot {seriestype} for {missingions} because these ions are not in compositiondata.txt")

    iontuplelist = [iontuple for iontuple in iontuplelist if iontuple not in missingions]
    prev_atomic_number = iontuplelist[0][0]
    colorindex = 0
    for atomic_number, ion_stage in iontuplelist:
        if atomic_number != prev_atomic_number:
            colorindex += 1

        elsymbol = at.get_elsymbol(atomic_number)

        ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
        if seriestype == "populations":
            if ion_stage == "ALL":
                key = f"nnelement_{elsymbol}"
            elif hasattr(ion_stage, "lower") and ion_stage.startswith(at.get_elsymbol(atomic_number)):
                # not really an ion_stage but an isotope name
                key = f"nniso_{ion_stage}"
            else:
                key = f"nnion_{ionstr}"
        else:
            key = f"{seriestype}_{ionstr}"

        print(f"Plotting {seriestype} {ionstr.replace('_', ' ')}")

        if seriestype != "populations" or args.ionpoptype == "absolute":
            scalefactor = pl.lit(1)
        elif args.ionpoptype == "elpop":
            scalefactor = pl.col(f"nnelement_{elsymbol}").mean()
        elif args.ionpoptype == "totalpop":
            scalefactor = pl.col("nntot").mean()
        else:
            raise AssertionError

        series = (
            estimators.group_by("plotpointid", maintain_order=True)
            .agg(pl.col(key).mean() / scalefactor, pl.col("xvalue").mean())
            .lazy()
            .collect()
            .sort("xvalue")
        )
        xlist = series["xvalue"].to_list()
        ylist = series[key].to_list()
        if startfromzero:
            # make a line segment from 0 velocity
            xlist.insert(0, 0.0)
            ylist.insert(0, ylist[0])

        plotlabel = (
            ion_stage
            if hasattr(ion_stage, "lower") and ion_stage != "ALL"
            else at.get_ionstring(atomic_number, ion_stage, style="chargelatex")
        )

        color = get_elemcolor(atomic_number=atomic_number)

        # linestyle = ['-.', '-', '--', (0, (4, 1, 1, 1)), ':'] + [(0, x) for x in dashes_list][ion_stage - 1]
        dashes: tuple[float, ...]
        if ion_stage == "ALL":
            dashes = ()
            linewidth = 1.0
        else:
            if hasattr(ion_stage, "lower") and ion_stage.endswith("stable"):
                index = 8
            elif hasattr(ion_stage, "lower"):
                # isotopic abundance, use the mass number
                index = int(ion_stage.lstrip(at.get_elsymbol(atomic_number)))
            else:
                index = ion_stage

            dashes_list = [(3, 1, 1, 1), (), (1.5, 1.5), (6, 3), (1, 3)]
            dashes = dashes_list[(index - 1) % len(dashes_list)]
            linewidth_list = [1.0, 1.0, 1.0, 0.7, 0.7]
            linewidth = linewidth_list[(index - 1) % len(linewidth_list)]
            # color = ['blue', 'green', 'red', 'cyan', 'purple', 'grey', 'brown', 'orange'][index - 1]

            if args.colorbyion:
                color = f"C{index - 1 % 10}"
                # plotlabel = f'{at.get_elsymbol(atomic_number)} {at.roman_numerals[ion_stage]}'
                dashes = ()

        # assert colorindex < 10
        # color = f'C{colorindex}'
        # or ax.step(where='pre', )

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)
        if plotkwargs.get("linestyle", "solid") != "None":
            plotkwargs["dashes"] = dashes
        ax.plot(xlist, ylist, linewidth=linewidth, label=plotlabel, color=color, **plotkwargs)
        prev_atomic_number = atomic_number
        plotted_something = True

    if seriestype == "populations":
        if args.ionpoptype == "absolute":
            ax.set_ylabel(r"Number density $\left[\rm{cm}^{-3}\right]$")
        elif args.ionpoptype == "elpop":
            # elsym = at.get_elsymbol(atomic_number)
            ax.set_ylabel(r"X$_{i}$/X$_{\rm element}$")
        elif args.ionpoptype == "totalpop":
            ax.set_ylabel(r"X$_{i}$/X$_{rm tot}$")
        else:
            raise AssertionError
    else:
        ax.set_ylabel(at.estimators.get_varname_formatted(seriestype))

    if plotted_something:
        ax.set_yscale(args.yscale)
        if args.yscale == "log":
            ymin, ymax = ax.get_ylim()
            ymin = max(ymin, ymax / 1e10)
            ax.set_ylim(bottom=ymin)
            # make space for the legend
            new_ymax = ymax * 10 ** (0.1 * math.log10(ymax / ymin))
            if ymin > 0 and new_ymax > ymin and np.isfinite(new_ymax):
                ax.set_ylim(top=new_ymax)


def plot_series(
    ax: plt.Axes,
    startfromzero: bool,
    variablename: str,
    showlegend: bool,
    modelpath: str | Path,
    estimators: pl.LazyFrame | pl.DataFrame,
    args: argparse.Namespace,
    nounits: bool = False,
    **plotkwargs: t.Any,
) -> None:
    """Plot something like Te or TR."""
    formattedvariablename = at.estimators.get_varname_formatted(variablename)
    serieslabel = f"{formattedvariablename}"
    units_string = at.estimators.get_units_string(variablename)

    if showlegend:
        linelabel = serieslabel
        if not nounits:
            linelabel += units_string
    else:
        ax.set_ylabel(serieslabel + units_string)
        linelabel = None
    print(f"Plotting {variablename}")

    series = (
        estimators.group_by("plotpointid", maintain_order=True)
        .agg(pl.col(variablename).mean(), pl.col("xvalue").mean())
        .lazy()
        .collect()
    )
    ylist = series[variablename].to_list()
    xlist = series["xvalue"].to_list()

    if min(ylist) == 0 or math.log10(max(ylist) / min(ylist)) > 2:
        ax.set_yscale("log")

    dictcolors = {
        "Te": "red",
        # 'heating_gamma': 'blue',
        # 'cooling_adiabatic': 'blue'
    }

    if startfromzero:
        # make a line segment from 0 velocity
        xlist.insert(0, 0.0)
        ylist.insert(0, ylist[0])

    xlist_filtered, ylist_filtered = at.estimators.apply_filters(xlist, ylist, args)

    ax.plot(
        xlist_filtered, ylist_filtered, linewidth=1.5, label=linelabel, color=dictcolors.get(variablename), **plotkwargs
    )


def get_xlist(
    xvariable: str,
    allnonemptymgilist: t.Sequence[int],
    estimators: pl.LazyFrame,
    timestepslist: t.Any,
    modelpath: str | Path,
    groupbyxvalue: bool,
    args: t.Any,
) -> tuple[list[float | int], list[int], list[list[int]], pl.LazyFrame]:
    estimators = estimators.filter(pl.col("timestep").is_in([ts for tssublist in timestepslist for ts in tssublist]))

    if xvariable in {"cellid", "modelgridindex"}:
        estimators = estimators.with_columns(xvalue=pl.col("modelgridindex"), plotpointid=pl.col("modelgridindex"))
    elif xvariable == "timestep":
        estimators = estimators.with_columns(xvalue=pl.col("timestep"), plotpointid=pl.col("timestep"))
    elif xvariable == "time":
        estimators = estimators.with_columns(xvalue=pl.col("time_mid"), plotpointid=pl.col("timestep"))
    elif xvariable in {"velocity", "beta"}:
        velcolumn = "vel_r_mid"
        scalefactor = 1e5 if xvariable == "velocity" else 29979245800
        estimators = estimators.with_columns(
            xvalue=(pl.col(velcolumn) / scalefactor), plotpointid=pl.col("modelgridindex")
        )
    else:
        assert xvariable in estimators.columns
        estimators = estimators.with_columns(xvalue=pl.col(xvariable), plotpointid=pl.col("modelgridindex"))

    # single valued line plot
    if groupbyxvalue:
        estimators = estimators.with_columns(plotpointid=pl.col("xvalue"))

    if args.xmax > 0:
        estimators = estimators.filter(pl.col("xvalue") <= args.xmax)

    estimators = estimators.sort("plotpointid")
    pointgroups = (
        (
            estimators.select(["plotpointid", "xvalue", "modelgridindex", "timestep"])
            .group_by("plotpointid", maintain_order=True)
            .agg(pl.col("xvalue").first(), pl.col("modelgridindex").first(), pl.col("timestep").unique())
        )
        .lazy()
        .collect()
    )

    return (
        pointgroups["xvalue"].to_list(),
        pointgroups["modelgridindex"].to_list(),
        pointgroups["timestep"].to_list(),
        estimators,
    )


def plot_subplot(
    ax: plt.Axes,
    timestepslist: list[list[int]],
    xlist: list[float | int],
    xvariable: str,
    startfromzero: bool,
    plotitems: list[t.Any],
    mgilist: list[int],
    modelpath: str | Path,
    estimators: pl.LazyFrame,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    """Make plot from ARTIS estimators."""
    # these three lists give the x value, modelgridex, and a list of timesteps (for averaging) for each plot of the plot
    showlegend = False
    seriescount = 0
    ylabel = None
    sameylabel = True
    for variablename in plotitems:
        if isinstance(variablename, str):
            seriescount += 1
            if ylabel is None:
                ylabel = get_ylabel(variablename)
            elif ylabel != get_ylabel(variablename):
                sameylabel = False
                break

    for plotitem in plotitems:
        if isinstance(plotitem, str):
            showlegend = seriescount > 1 or len(plotitem) > 20 or not sameylabel
            plot_series(
                ax=ax,
                startfromzero=startfromzero,
                variablename=plotitem,
                showlegend=showlegend,
                modelpath=modelpath,
                estimators=estimators,
                args=args,
                nounits=sameylabel,
                **plotkwargs,
            )
            if showlegend and sameylabel and ylabel is not None:
                ax.set_ylabel(ylabel)
        else:  # it's a sequence of values
            seriestype, params = plotitem

            if seriestype in {"initabundances", "initmasses"}:
                showlegend = True
                plot_init_abundances(
                    ax=ax,
                    xlist=xlist,
                    specieslist=params,
                    mgilist=mgilist,
                    modelpath=Path(modelpath),
                    seriestype=seriestype,
                    startfromzero=startfromzero,
                    args=args,
                )

            elif seriestype == "levelpopulation" or seriestype.startswith("levelpopulation_"):
                showlegend = True
                plot_levelpop(
                    ax,
                    xlist,
                    seriestype,
                    params,
                    timestepslist,
                    mgilist,
                    estimators,
                    modelpath,
                    args=args,
                )

            elif seriestype in {"averageionisation", "averageexcitation"}:
                showlegend = True
                plot_average_ionisation_excitation(
                    ax,
                    xlist,
                    seriestype,
                    params,
                    timestepslist,
                    mgilist,
                    estimators,
                    modelpath,
                    startfromzero=startfromzero,
                    args=args,
                    **plotkwargs,
                )

            elif seriestype == "_ymin":
                ax.set_ylim(bottom=params)

            elif seriestype == "_ymax":
                ax.set_ylim(top=params)

            elif seriestype == "_yscale":
                ax.set_yscale(params)

            else:
                showlegend = True
                seriestype, ionlist = plotitem
                plot_multi_ion_series(
                    ax=ax,
                    startfromzero=startfromzero,
                    seriestype=seriestype,
                    ionlist=ionlist,
                    estimators=estimators,
                    modelpath=modelpath,
                    args=args,
                    **plotkwargs,
                )

    ax.tick_params(right=True)
    if showlegend and not args.nolegend:
        ax.legend(
            loc="upper right",
            handlelength=2,
            frameon=False,
            numpoints=1,
            **(
                {"ncol": math.ceil(len(plotitems[0][1]) / 2.0)}
                if (plotitems[0][0] == "populations" and args.yscale == "log")
                else {}
            ),
            markerscale=3,
        )


def make_plot(
    modelpath: Path | str,
    timestepslist_unfiltered: list[list[int]],
    allnonemptymgilist: list[int],
    estimators: pl.LazyFrame,
    xvariable: str,
    plotlist,
    args: t.Any,
    **plotkwargs: t.Any,
) -> str:
    modelname = at.get_model_name(modelpath)

    fig, axes = plt.subplots(
        nrows=len(plotlist),
        ncols=1,
        sharex=True,
        figsize=(
            args.figscale * at.get_config()["figwidth"] * args.scalefigwidth,
            args.figscale * at.get_config()["figwidth"] * 0.5 * len(plotlist),
        ),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )
    if len(plotlist) == 1:
        axes = [axes]

    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    if not args.hidexlabel:
        axes[-1].set_xlabel(
            f"{at.estimators.get_varname_formatted(xvariable)}{at.estimators.get_units_string(xvariable)}"
        )

    xlist, mgilist, timestepslist, estimators = get_xlist(
        xvariable=xvariable,
        allnonemptymgilist=allnonemptymgilist,
        estimators=estimators,
        timestepslist=timestepslist_unfiltered,
        modelpath=modelpath,
        groupbyxvalue=not args.markersonly,
        args=args,
    )
    startfromzero = (xvariable.startswith("velocity") or xvariable == "beta") and not args.markersonly

    xmin = args.xmin if args.xmin >= 0 else min(xlist)
    xmax = args.xmax if args.xmax > 0 else max(xlist)

    if args.markersonly:
        plotkwargs["linestyle"] = "None"
        plotkwargs["marker"] = "."
        plotkwargs["markersize"] = 3
        plotkwargs["alpha"] = 0.5

        # with no lines, line styles cannot distringuish ions
        args.colorbyion = True

    for ax, plotitems in zip(axes, plotlist):
        ax.set_xlim(left=xmin, right=xmax)
        plot_subplot(
            ax=ax,
            timestepslist=timestepslist,
            xlist=xlist,
            xvariable=xvariable,
            plotitems=plotitems,
            mgilist=mgilist,
            modelpath=modelpath,
            estimators=estimators,
            startfromzero=startfromzero,
            args=args,
            **plotkwargs,
        )

    if len(set(mgilist)) == 1 and len(timestepslist[0]) > 1:  # single grid cell versus time plot
        figure_title = f"{modelname}\nCell {mgilist[0]}"

        defaultoutputfile = "plotestimators_cell{modelgridindex:03d}.{format}"
        if Path(args.outputfile).is_dir():
            args.outputfile = str(Path(args.outputfile) / defaultoutputfile)

        outfilename = str(args.outputfile).format(modelgridindex=mgilist[0], format=args.format)

    else:
        if args.multiplot:
            timestep = f"ts{timestepslist[0][0]:02d}"
            timedays = f"{at.get_timestep_time(modelpath, timestepslist[0][0]):.2f}d"
        else:
            timestepmin = min(timestepslist[0])
            timestepmax = max(timestepslist[0])
            timestep = f"ts{timestepmin:02d}-ts{timestepmax:02d}"
            timedays = f"{at.get_timestep_time(modelpath, timestepmin):.2f}d-{at.get_timestep_time(modelpath, timestepmax):.2f}d"

        figure_title = f"{modelname}\nTimestep {timestep} ({timedays})"
        print("Plotting ", figure_title.replace("\n", " "))

        defaultoutputfile = "plotestimators_{timestep}_{timedays}.{format}"
        if Path(args.outputfile).is_dir():
            args.outputfile = str(Path(args.outputfile) / defaultoutputfile)

        assert isinstance(timestepslist[0], list)
        outfilename = str(args.outputfile).format(timestep=timestep, timedays=timedays, format=args.format)

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=12)

    print(f"Saving {outfilename} ...")
    fig.savefig(outfilename, dpi=300)

    if args.show:
        plt.show()
    else:
        plt.close()

    return outfilename


def plot_recombrates(modelpath, estimators, atomic_number, ion_stage_list, **plotkwargs):
    fig, axes = plt.subplots(
        nrows=len(ion_stage_list),
        ncols=1,
        sharex=True,
        figsize=(5, 8),
        tight_layout={"pad": 0.5, "w_pad": 0.0, "h_pad": 0.0},
    )
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axes[-1].set_xlabel("T_e in kelvins")

    recombcalibrationdata = at.atomic.get_ionrecombratecalibration(modelpath)

    for ax, ion_stage in zip(axes, ion_stage_list):
        ionstr = (
            f"{at.get_elsymbol(atomic_number)} {at.roman_numerals[ion_stage]} to {at.roman_numerals[ion_stage - 1]}"
        )

        listT_e = []
        list_rrc = []
        list_rrc2 = []
        for dicttimestepmodelgrid in estimators.values():
            if (atomic_number, ion_stage) in dicttimestepmodelgrid["RRC_LTE_Nahar"]:
                listT_e.append(dicttimestepmodelgrid["Te"])
                list_rrc.append(dicttimestepmodelgrid["RRC_LTE_Nahar"][(atomic_number, ion_stage)])
                list_rrc2.append(dicttimestepmodelgrid["Alpha_R"][(atomic_number, ion_stage)])

        if not list_rrc:
            continue

        # sort the pairs by temperature ascending
        listT_e, list_rrc, list_rrc2 = zip(*sorted(zip(listT_e, list_rrc, list_rrc2), key=lambda x: x[0]))

        ax.plot(listT_e, list_rrc, linewidth=2, label=f"{ionstr} ARTIS RRC_LTE_Nahar", **plotkwargs)
        ax.plot(listT_e, list_rrc2, linewidth=2, label=f"{ionstr} ARTIS Alpha_R", **plotkwargs)

        with contextlib.suppress(KeyError):
            dfrates = recombcalibrationdata[(atomic_number, ion_stage)].query(
                "T_e > @T_e_min & T_e < @T_e_max", local_dict={"T_e_min": min(listT_e), "T_e_max": max(listT_e)}
            )

            ax.plot(
                dfrates.T_e,
                dfrates.rrc_total,
                linewidth=2,
                label=f"{ionstr} (calibration)",
                markersize=6,
                marker="s",
                **plotkwargs,
            )
        # rrcfiles = glob.glob(
        #     f'/Users/lshingles/Library/Mobile Documents/com~apple~CloudDocs/GitHub/'
        #     f'artis-atomic/atomic-data-nahar/{at.get_elsymbol(atomic_number).lower()}{ion_stage - 1}.rrc*.txt')
        # if rrcfiles:
        #     dfrecombrates = get_ionrecombrates_fromfile(rrcfiles[0])
        #
        #     dfrecombrates.query("logT > @logT_e_min & logT < @logT_e_max",
        #                         local_dict={'logT_e_min': math.log10(min(listT_e)),
        #                                     'logT_e_max': math.log10(max(listT_e))}, inplace=True)
        #
        #     listT_e_Nahar = [10 ** x for x in dfrecombrates['logT'].values]
        #     ax.plot(listT_e_Nahar, dfrecombrates['RRC_total'], linewidth=2,
        #             label=ionstr + " (Nahar)", markersize=6, marker='s', **plotkwargs)

        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1, prop={"size": 10})

    # modelname = at.get_model_name(".")
    # plotlabel = f'Timestep {timestep}'
    # time_days = float(at.get_timestep_time('spec.out', timestep))
    # if time_days >= 0:
    #     plotlabel += f' (t={time_days:.2f}d)'
    # fig.suptitle(plotlabel, fontsize=12)
    elsymbol = at.get_elsymbol(atomic_number)
    outfilename = f"plotestimators_recombrates_{elsymbol}.pdf"
    fig.savefig(outfilename)
    print(f"Saved {outfilename}")
    plt.close()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath", default=".", help="Paths to ARTIS folder (or virtual path e.g. codecomparison/ddc10/cmfgen)"
    )

    parser.add_argument("--recombrates", action="store_true", help="Make a recombination rate plot")

    parser.add_argument(
        "-modelgridindex", "-cell", "-mgi", type=int, default=None, help="Modelgridindex for time evolution plot"
    )

    parser.add_argument("-timestep", "-ts", nargs="?", help="Timestep number for internal structure plot")

    parser.add_argument("-timedays", "-time", "-t", nargs="?", help="Time in days to plot for internal structure plot")

    parser.add_argument("-timemin", type=float, help="Lower time in days")

    parser.add_argument("-timemax", type=float, help="Upper time in days")

    parser.add_argument("--multiplot", action="store_true", help="Make multiple plots for timesteps in range")

    parser.add_argument("-x", help="Horizontal axis variable, e.g. cellid, velocity, timestep, or time")

    parser.add_argument("-xmin", type=float, default=-1, help="Plot range: minimum x value")

    parser.add_argument("-xmax", type=float, default=-1, help="Plot range: maximum x value")

    parser.add_argument(
        "-yscale", default="log", choices=["log", "linear"], help="Set yscale to log or linear (default log)"
    )

    parser.add_argument("--hidexlabel", action="store_true", help="Hide the bottom horizontal axis label")

    parser.add_argument(
        "--markersonly", action="store_true", help="Plot markers instead of lines (always set for 2D and 3D)"
    )

    parser.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")

    parser.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and polyorder.e.g. -filtersavgol 5 3",
    )

    parser.add_argument("-format", "-f", default="pdf", choices=["pdf", "png"], help="Set format of output plot files")

    parser.add_argument("--makegif", action="store_true", help="Make a gif with time evolution (requires --multiplot)")

    parser.add_argument("--notitle", action="store_true", help="Suppress the top title from the plot")

    parser.add_argument("-plotlist", type=list, default=[], help="Plot list (when calling from Python only)")

    parser.add_argument(
        "-ionpoptype",
        default="elpop",
        choices=["absolute", "totalpop", "elpop"],
        help="Plot absolute ion populations, or ion populations as a fraction of total or element population",
    )

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument(
        "-figscale", type=float, default=1.0, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("-scalefigwidth", type=float, default=1.0, help="Scale factor for plot width.")

    parser.add_argument("--show", action="store_true", help="Show plot before quitting")

    parser.add_argument(
        "-o", action="store", dest="outputfile", type=Path, default=Path(), help="Filename for PDF file"
    )

    parser.add_argument(
        "--colorbyion", action="store_true", help="Populations plots colored by ion rather than element"
    )

    parser.add_argument(
        "--classicartis", action="store_true", help="Flag to show using output from classic ARTIS branch"
    )

    parser.add_argument(
        "-readonlymgi",
        default=False,
        choices=["alongaxis", "cone"],  # plan to extend this to e.g. 2D slice
        help="Option to read only selected mgi and choice of which mgi to select. Choose which axis with args.axis",
    )

    parser.add_argument(
        "-axis",
        default="+z",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        help="Choose an axis for use with args.readonlymgi. Hint: for negative use e.g. -axis=-z",
    )


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs) -> None:
    """Plot ARTIS estimators."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    modelpath = Path(args.modelpath)

    modelname = at.get_model_name(modelpath)

    if not args.timedays and not args.timestep and args.modelgridindex is not None:
        args.timestep = f"0-{len(at.get_timestep_times(modelpath)) - 1}"

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )

    if args.readonlymgi:
        args.sliceaxis = args.axis[1]
        assert args.axis[0] in {"+", "-"}
        args.positive_axis = args.axis[0] == "+"

        axes = ["x", "y", "z"]
        axes.remove(args.sliceaxis)
        args.other_axis1 = axes[0]
        args.other_axis2 = axes[1]

    print(
        f"Plotting estimators for '{modelname}' timesteps {timestepmin} to {timestepmax} "
        f"({args.timemin:.1f} to {args.timemax:.1f}d)"
    )

    plotlist = args.plotlist or [
        # [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
        # ['heating_dep', 'heating_coll', 'heating_bf', 'heating_ff',
        #  ['_yscale', 'linear']],
        # ['cooling_adiabatic', 'cooling_coll', 'cooling_fb', 'cooling_ff',
        #  ['_yscale', 'linear']],
        # [['initmasses', ['Ni_56', 'He', 'C', 'Mg']]],
        # ['heating_gamma/gamma_dep'],
        # ["nne", ["_ymin", 1e5], ["_ymax", 1e10]],
        ["rho", ["_yscale", "log"], ["_ymin", 1e-16]],
        ["TR", ["_yscale", "linear"]],  # , ["_ymin", 1000], ["_ymax", 15000]
        # ["Te"],
        # ["Te", "TR"],
        [["averageionisation", ["Sr"]]],
        # [["averageexcitation", ["Fe II", "Fe III"]]],
        # [["populations", ["Sr90", "Sr91", "Sr92", "Sr94"]]],
        [["populations", ["Sr I", "Sr II", "Sr III", "Sr IV"]]],
        # [['populations', ['He I', 'He II', 'He III']]],
        # [['populations', ['C I', 'C II', 'C III', 'C IV', 'C V']]],
        # [['populations', ['O I', 'O II', 'O III', 'O IV']]],
        # [['populations', ['Ne I', 'Ne II', 'Ne III', 'Ne IV', 'Ne V']]],
        # [['populations', ['Si I', 'Si II', 'Si III', 'Si IV', 'Si V']]],
        # [['populations', ['Cr I', 'Cr II', 'Cr III', 'Cr IV', 'Cr V']]],
        # [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Fe VI', 'Fe VII', 'Fe VIII']]],
        # [['populations', ['Co I', 'Co II', 'Co III', 'Co IV', 'Co V', 'Co VI', 'Co VII']]],
        # [['populations', ['Ni I', 'Ni II', 'Ni III', 'Ni IV', 'Ni V', 'Ni VI', 'Ni VII']]],
        # [['populations', ['Fe II', 'Fe III', 'Co II', 'Co III', 'Ni II', 'Ni III']]],
        # [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
        # [['RRC_LTE_Nahar', ['Fe II', 'Fe III', 'Fe IV', 'Fe V']]],
        # [['RRC_LTE_Nahar', ['Co II', 'Co III', 'Co IV', 'Co V']]],
        # [['RRC_LTE_Nahar', ['Ni I', 'Ni II', 'Ni III', 'Ni IV', 'Ni V', 'Ni VI', 'Ni VII']]],
        # [['Alpha_R / RRC_LTE_Nahar', ['Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni III']]],
        # [['gamma_NT', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
    ]

    if args.readonlymgi:
        if args.readonlymgi == "alongaxis":
            print(f"Getting mgi along {args.axis} axis")
            dfselectedcells = at.inputmodel.slice1dfromconein3dmodel.get_profile_along_axis(args=args)

        elif args.readonlymgi == "cone":
            print(f"Getting mgi lying within a cone around {args.axis} axis")
            dfselectedcells = at.inputmodel.slice1dfromconein3dmodel.make_cone(args)
        else:
            msg = f"Invalid args.readonlymgi: {args.readonlymgi}"
            raise ValueError(msg)
        dfselectedcells = dfselectedcells[dfselectedcells["rho"] > 0]
        args.modelgridindex = list(dfselectedcells["inputcellid"])

    timesteps_included = list(range(timestepmin, timestepmax + 1))
    if args.classicartis:
        import artistools.estimators.estimators_classic

        modeldata, _ = at.inputmodel.get_modeldata(modelpath)
        estimatorsdict = artistools.estimators.estimators_classic.read_classic_estimators(modelpath, modeldata)
        assert estimatorsdict is not None
        estimators = pl.DataFrame(
            [
                {
                    "timestep": ts,
                    "modelgridindex": mgi,
                    **estimvals,
                }
                for (ts, mgi), estimvals in estimatorsdict.items()
            ]
        ).lazy()
    else:
        estimators = at.estimators.scan_estimators(
            modelpath=modelpath,
            modelgridindex=args.modelgridindex,
            timestep=tuple(timesteps_included),
        )
    assert estimators is not None
    tmids = at.get_timestep_times(modelpath, loc="mid")
    estimators = estimators.join(
        pl.DataFrame({"timestep": range(len(tmids)), "time_mid": tmids})
        .with_columns(pl.col("timestep").cast(pl.Int32))
        .lazy(),
        on="timestep",
        how="left",
    )

    for ts in reversed(timesteps_included):
        tswithdata = estimators.select("timestep").unique().collect().to_series()
        for ts in timesteps_included:
            if ts not in tswithdata:
                timesteps_included.remove(ts)
                print(f"ts {ts} requested but no data found. Removing.")

    if not timesteps_included:
        print("No timesteps with data are included")
        return

    if args.recombrates:
        plot_recombrates(modelpath, estimators, 26, [2, 3, 4, 5])
        plot_recombrates(modelpath, estimators, 27, [3, 4])
        plot_recombrates(modelpath, estimators, 28, [3, 4, 5])

        return

    assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath)

    outdir = args.outputfile if (args.outputfile).is_dir() else Path()
    if not args.readonlymgi and (args.modelgridindex is not None or args.x in {"time", "timestep"}):
        # plot time evolution in specific cell
        if not args.x:
            args.x = "time"
        mgilist = [args.modelgridindex] * len(timesteps_included)
        timestepslist_unfiltered = [[ts] for ts in timesteps_included]
        if not assoc_cells.get(args.modelgridindex, []):
            msg = f"cell {args.modelgridindex} is empty. no estimators available"
            raise ValueError(msg)
        make_plot(
            modelpath=modelpath,
            timestepslist_unfiltered=timestepslist_unfiltered,
            allnonemptymgilist=mgilist,
            estimators=estimators,
            xvariable=args.x,
            plotlist=plotlist,
            args=args,
        )
    else:
        # plot a range of cells in a time snapshot showing internal structure

        if not args.x:
            args.x = "velocity"

        dfmodel, modelmeta = at.inputmodel.get_modeldata_polars(modelpath, derived_cols=["ALL"])
        if args.x == "velocity" and modelmeta["vmax_cmps"] > 0.3 * 29979245800:
            args.x = "beta"

        dfmodel = dfmodel.filter(pl.col("vel_r_mid") <= modelmeta["vmax_cmps"])
        estimators = estimators.join(dfmodel, on="modelgridindex")
        estimators = estimators.with_columns(
            rho_init=pl.col("rho"),
            rho=pl.col("rho") * (modelmeta["t_model_init_days"] / pl.col("time_mid")) ** 3,
        )

        if args.readonlymgi:
            estimators = estimators.filter(pl.col("modelgridindex").is_in(args.modelgridindex))

        if args.classicartis:
            modeldata, _ = at.inputmodel.get_modeldata(modelpath)
            allnonemptymgilist = [
                modelgridindex
                for modelgridindex in modeldata.index
                if not estimators.filter(pl.col("modelgridindex") == modelgridindex)
                .select("modelgridindex")
                .lazy()
                .collect()
                .is_empty()
            ]
        else:
            allnonemptymgilist = [mgi for mgi, assocpropcells in assoc_cells.items() if assocpropcells]

        estimators = estimators.filter(pl.col("modelgridindex").is_in(allnonemptymgilist)).filter(
            pl.col("timestep").is_in(timesteps_included)
        )

        frames_timesteps_included = (
            [[ts] for ts in range(timestepmin, timestepmax + 1)] if args.multiplot else [timesteps_included]
        )

        if args.makegif:
            args.multiplot = True
            args.format = "png"

        outputfiles = []
        for timesteps_included in frames_timesteps_included:
            timestepslist_unfiltered = [timesteps_included] * len(allnonemptymgilist)
            outfilename = make_plot(
                modelpath=modelpath,
                timestepslist_unfiltered=timestepslist_unfiltered,
                allnonemptymgilist=allnonemptymgilist,
                estimators=estimators,
                xvariable=args.x,
                plotlist=plotlist,
                args=args,
            )

            outputfiles.append(outfilename)

        if len(outputfiles) > 1:
            if args.makegif:
                assert args.multiplot
                assert args.format == "png"
                import imageio.v2 as iio

                gifname = outdir / f"plotestim_evolution_ts{timestepmin:03d}_ts{timestepmax:03d}.gif"
                with iio.get_writer(gifname, mode="I", duration=1000) as writer:
                    for filename in outputfiles:
                        image = iio.imread(filename)
                        writer.append_data(image)  # type: ignore[attr-defined]
                print(f"Created gif: {gifname}")
            elif args.format == "pdf":
                at.merge_pdf_files(outputfiles)


if __name__ == "__main__":
    main()
