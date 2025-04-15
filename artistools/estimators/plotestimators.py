#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Functions for plotting artis estimators and internal structure.

Examples are temperatures, populations, heating/cooling rates.
"""

import argparse
import contextlib
import math
import string
import typing as t
from collections.abc import Sequence
from itertools import chain
from pathlib import Path

import argcomplete
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib import ticker

import artistools as at

colors_tab10: list[str] = list(plt.get_cmap("tab10")(np.linspace(0, 1.0, 10)))

# reserve colours for these elements
elementcolors = {"Fe": colors_tab10[0], "Ni": colors_tab10[1], "Co": colors_tab10[2]}


def get_elemcolor(atomic_number: int | None = None, elsymbol: str | None = None) -> str | npt.NDArray[t.Any]:
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
    ax: mplax.Axes,
    specieslist: list[str],
    estimators: pl.LazyFrame,
    seriestype: str,
    startfromzero: bool,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    if seriestype == "initmasses":
        estimators = estimators.with_columns(
            (pl.col(massfraccol) * pl.col("mass_g") / 1.989e33).alias(
                f"init_mass_{massfraccol.removeprefix('init_X_')}"
            )
            for massfraccol in estimators.collect_schema().names()
            if massfraccol.startswith("init_X_")
        )
        ax.set_ylabel(r"Initial mass per x point [M$_\odot$]")
        valuetype = "init_mass_"
    else:
        assert seriestype == "initabundances"
        ax.set_ylim(1e-20, 1.0)
        ax.set_ylabel("Initial mass fraction")
        valuetype = "init_X_"

    for speciesstr in specieslist:
        splitvariablename = speciesstr.split("_")
        elsymbol = splitvariablename[0].strip(string.digits)
        atomic_number = at.get_atomic_number(elsymbol)

        ylist = []
        linestyle = "-"
        if speciesstr.lower() in {"ni_56", "ni56", "56ni"}:
            yvalue = pl.col(f"{valuetype}Ni56")
            linelabel = "$^{56}$Ni"
            linestyle = "--"
        elif speciesstr.lower() in {"ni_stb", "ni_stable"}:
            yvalue = pl.col(f"{valuetype}{elsymbol}") - pl.col(f"{valuetype}Ni56")
            linelabel = "Stable Ni"
        elif speciesstr.lower() in {"co_56", "co56", "56co"}:
            yvalue = pl.col(f"{valuetype}Co56")
            linelabel = "$^{56}$Co"
        elif speciesstr.lower() in {"fegrp", "ffegroup"}:
            yvalue = pl.col(f"{valuetype}Fegroup")
            linelabel = "Fe group"
        else:
            linelabel = speciesstr
            yvalue = pl.col(f"{valuetype}{elsymbol}")

        color = get_elemcolor(atomic_number=atomic_number)

        series = (
            estimators.group_by("plotpointid", maintain_order=True)
            .agg(yvalue=(yvalue * pl.col("mass_g")).sum() / (pl.col("mass_g")).sum(), xvalue=pl.col("xvalue").mean())
            .sort("xvalue")
            .collect()
        )

        ylist = series["yvalue"].to_list()
        xlist = series["xvalue"].to_list()

        if startfromzero:
            # make a line segment from 0 velocity
            xlist = [0.0, *xlist]
            ylist = [ylist[0], *ylist]

        xlist_filtered, ylist_filtered = at.estimators.apply_filters(xlist, ylist, args)

        ax.plot(
            xlist_filtered,
            ylist_filtered,
            linewidth=1.5,
            label=linelabel,
            linestyle=linestyle,
            color=color,
            **plotkwargs,
        )

        # if args.yscale == 'log':
        #     ax.set_yscale('log')


def plot_average_ionisation_excitation(
    ax: mplax.Axes,
    xlist: list[float],
    seriestype: str,
    params: Sequence[str],
    timestepslist: Sequence[Sequence[int]],
    mgilist: Sequence[int],
    estimators: pl.LazyFrame,
    modelpath: Path | str,
    startfromzero: bool,
    args: argparse.Namespace | None = None,
    **plotkwargs: t.Any,
) -> None:
    if args is None:
        args = argparse.Namespace()

    if seriestype == "averageexcitation":
        ax.set_ylabel("Average excitation [eV]")
    elif seriestype == "averageionisation":
        ax.set_ylabel("Average ion charge")
    else:
        raise ValueError

    if startfromzero:
        xlist = [0.0, *xlist]

    arr_tdelta = at.get_timestep_times(modelpath, loc="delta")
    for paramvalue in params:
        print(f"Plotting {seriestype} {paramvalue}")
        if seriestype == "averageionisation":
            atomic_number = at.get_atomic_number(paramvalue)
            ion_stage = None
        else:
            atomic_number = at.get_atomic_number(paramvalue.split(" ")[0])
            ion_stage = at.decode_roman_numeral(paramvalue.split(" ")[1])
        ylist = []
        if seriestype == "averageexcitation":
            print("  This will be slow! TODO: reimplement with polars.")
            assert ion_stage is not None
            for modelgridindex, timesteps in zip(mgilist, timestepslist, strict=False):
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
            if f"nnelement_{elsymb}" not in estimators.collect_schema().names():
                msg = f"ERROR: No element data found for {paramvalue}"
                raise ValueError(msg)

            ioncols = [col for col in estimators.collect_schema().names() if col.startswith(f"nnion_{elsymb}_")]
            ioncharges = [at.decode_roman_numeral(col.removeprefix(f"nnion_{elsymb}_")) - 1 for col in ioncols]
            ax.set_ylim(0.0, max(ioncharges) + 0.1)

            series = (
                estimators.filter(pl.col(f"nnelement_{elsymb}") > 0.0)
                .group_by("plotpointid", maintain_order=True)
                .agg(
                    (
                        (
                            pl.sum_horizontal([
                                ioncharge * pl.col(ioncol)
                                for ioncol, ioncharge in zip(ioncols, ioncharges, strict=True)
                            ])
                            * pl.col("volume")
                            * pl.col("tdelta")
                        ).sum()
                        / (pl.col(f"nnelement_{elsymb}") * pl.col("volume") * pl.col("tdelta")).sum()
                    ).alias(f"averageionisation_{elsymb}"),
                    pl.col("xvalue").mean(),
                )
                .sort("xvalue")
                .lazy()
                .collect()
            )

            xlist = series["xvalue"].to_list()
            if startfromzero:
                xlist = [0.0, *xlist]

            ylist = series[f"averageionisation_{elsymb}"].to_list()

        color = get_elemcolor(atomic_number=atomic_number)

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)
        if startfromzero:
            ylist = [ylist[0], *ylist]

        ax.plot(xlist, ylist, label=paramvalue, color=color, **plotkwargs)


def plot_levelpop(
    ax: mplax.Axes,
    xlist: Sequence[int | float] | npt.NDArray[np.floating],
    seriestype: str,
    params: Sequence[str],
    timestepslist: Sequence[Sequence[int]],
    mgilist: Sequence[int | Sequence[int]],
    modelpath: str | Path,
    startfromzero: bool,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    if seriestype == "levelpopulation_dn_on_dvel":
        ax.set_ylabel("dN/dV [{}km$^{{-1}}$ s]")
        ax.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(ax.get_ylabel()))
    elif seriestype == "levelpopulation":
        ax.set_ylabel("X$_{{i}}$ [{}/cm³]")
        ax.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(ax.get_ylabel()))
    else:
        raise ValueError

    modeldata, _ = at.inputmodel.get_modeldata_pandas(modelpath, derived_cols=["mass_g", "volume"])

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
        for modelgridindex, timesteps in zip(mgilist, timestepslist, strict=False):
            valuesum = 0.0
            tdeltasum = 0.0
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
                assert isinstance(modelgridindex, int)
                deltav = modeldata.loc[modelgridindex].vel_r_max_kmps - modeldata.loc[modelgridindex].vel_r_min_kmps
                ylist.append(valuesum / tdeltasum * modeldata.loc[modelgridindex].volume / deltav)
            else:
                ylist.append(valuesum / tdeltasum)

        if startfromzero:
            # make a line segment from 0 velocity
            xlist = [0.0, *xlist]
            ylist = [ylist[0], *ylist]

        xlist, ylist = at.estimators.apply_filters(xlist, np.array(ylist), args)

        ax.plot(xlist, ylist, label=label, **plotkwargs)


def plot_multi_ion_series(
    ax: mplax.Axes,
    startfromzero: bool,
    seriestype: str,
    ionlist: Sequence[str],
    estimators: pl.LazyFrame,
    modelpath: str | Path,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    """Plot an ion-specific property, e.g., populations."""
    # if seriestype == 'populations':
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.10))

    plotted_something = False

    def get_iontuple(ionstr: str) -> tuple[int, str | int]:
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

    missingions: set[tuple[int, str | int]] = set()
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
            if f"nnion_{ionstr}" not in estimators.collect_schema().names():
                missingions.add((atomic_number, ion_stage))

    if missingions:
        print(f" Warning: Can't plot {seriestype} for {missingions} because these ions are not in compositiondata.txt")

    iontuplelist = [iontuple for iontuple in iontuplelist if iontuple not in missingions]
    lazyframes = []
    for atomic_number, ion_stage in iontuplelist:
        elsymbol = at.get_elsymbol(atomic_number)

        ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
        if seriestype == "populations":
            if ion_stage == "ALL":
                expr_yvals = pl.col(f"nnelement_{elsymbol}")
            elif isinstance(ion_stage, str) and ion_stage.startswith(at.get_elsymbol(atomic_number)):
                # not really an ion_stage but an isotope name
                expr_yvals = pl.col(f"nniso_{ion_stage}")
            else:
                expr_yvals = pl.col(f"nnion_{ionstr}")
        else:
            expr_yvals = pl.col(f"{seriestype}_{ionstr}")

        print(f"Plotting {seriestype} {ionstr.replace('_', ' ')}")

        if seriestype != "populations" or args.poptype == "absolute":
            expr_normfactor = pl.lit(1)
        elif args.poptype == "elpop":
            expr_normfactor = pl.col(f"nnelement_{elsymbol}")
        elif args.poptype == "totalpop":
            expr_normfactor = pl.col("nntot")
        elif args.poptype in {"radialdensity", "cylradialdensity"}:
            # get the volumetric number density to later be multiplied by the surface area of a sphere or cylinder
            expr_normfactor = pl.lit(1)
        elif args.poptype == "cumulative":
            # multiply by volume to get number of particles
            expr_normfactor = pl.lit(1) / pl.col("volume")
        else:
            raise AssertionError

        expr_yvals = (expr_yvals * pl.col("volume") * pl.col("tdelta")).sum() / (
            expr_normfactor * pl.col("volume") * pl.col("tdelta")
        ).sum()

        # convert volumetric number density to radial density
        if args.poptype == "radialdensity":
            expr_yvals *= 4 * math.pi * pl.col("vel_r_mid").mean().pow(2)
        elif args.poptype == "cylradialdensity":
            expr_yvals *= 2 * math.pi * pl.col("vel_rcyl_mid").mean()

        series_lazy = (
            estimators.group_by("plotpointid", maintain_order=True)
            .agg(yvalue=expr_yvals, xvalue=pl.col("xvalue").mean())
            .sort("xvalue")
        )
        lazyframes.append(series_lazy)

    for seriesindex, (iontuple, series) in enumerate(zip(iontuplelist, pl.collect_all(lazyframes), strict=True)):
        atomic_number, ion_stage = iontuple
        xlist = series.get_column("xvalue").to_list()
        ylist = (
            series.get_column("yvalue").cum_sum() if args.poptype == "cumulative" else series.get_column("yvalue")
        ).to_list()
        if startfromzero:
            # make a line segment from 0 velocity
            xlist = [0.0, *xlist]
            ylist = [ylist[0], *ylist]

        plotlabel = (
            ion_stage
            if hasattr(ion_stage, "lower") and ion_stage != "ALL"
            else at.get_ionstring(atomic_number, ion_stage, style="chargelatex")
        )

        color = get_elemcolor(atomic_number=atomic_number)

        # linestyle = ['-.', '-', '--', (0, (4, 1, 1, 1)), ':'] + [(0, x) for x in dashes_list][ion_stage - 1]
        dashes: tuple[float, ...] = ()
        styleindex = 0
        if isinstance(ion_stage, str):
            if ion_stage != "ALL":
                # isotopic abundance
                if args.colorbyion:
                    color = f"C{seriesindex % 10}"
                else:
                    styleindex = seriesindex
        else:
            assert isinstance(ion_stage, int)
            if args.colorbyion:
                color = f"C{(ion_stage - 1) % 10}"
            else:
                styleindex = ion_stage - 1

        dashes_list = [(3, 1, 1, 1), (), (1.5, 1.5), (6, 3), (1, 3)]
        dashes = dashes_list[styleindex % len(dashes_list)]

        linewidth_list = [1.0, 1.0, 1.0, 0.7, 0.7]
        linewidth = linewidth_list[styleindex % len(linewidth_list)]

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)
        if plotkwargs.get("linestyle", "solid") != "None":
            plotkwargs["dashes"] = dashes

        ax.plot(xlist, ylist, linewidth=linewidth, label=plotlabel, color=color, **plotkwargs)
        plotted_something = True

    if seriestype == "populations":
        if args.poptype == "absolute":
            ax.set_ylabel(r"Number density $\left[\rm{cm}^{-3}\right]$")
        elif args.poptype == "elpop":
            # elsym = at.get_elsymbol(atomic_number)
            ax.set_ylabel(r"X$_{i}$/X$_{\rm element}$")
        elif args.poptype == "totalpop":
            ax.set_ylabel(r"X$_{i}$/X$_{rm tot}$")
        elif args.poptype == "radialdensity":
            ax.set_ylabel(r"Radial density dN/dr $\left[\rm{cm}^{-1}\right]$")
        elif args.poptype == "cylradialdensity":
            ax.set_ylabel(r"Cylindrical radial density dN/drcyl $\left[\rm{cm}^{-1}\right]$")
        elif args.poptype == "cumulative":
            ax.set_ylabel(r"Cumulative particle count")
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
    ax: mplax.Axes,
    startfromzero: bool,
    variable: str | pl.Expr,
    showlegend: bool,
    estimators: pl.LazyFrame,
    args: argparse.Namespace,
    nounits: bool = False,
    **plotkwargs: t.Any,
) -> None:
    """Plot something like Te or TR."""
    if isinstance(variable, pl.Expr):
        colexpr = variable
    else:
        assert variable in estimators.collect_schema().names(), f"Variable {variable} not found in estimators"
        colexpr = pl.col(variable)

    variablename = colexpr.meta.output_name()

    serieslabel = at.estimators.get_varname_formatted(variablename)
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
        .agg(
            yvalue=(colexpr * pl.col("volume") * pl.col("tdelta")).sum() / (pl.col("volume") * pl.col("tdelta")).sum(),
            xvalue=pl.col("xvalue").mean(),
        )
        .sort("xvalue")
        .collect()
    )

    ylist = series["yvalue"].to_list()
    xlist = series["xvalue"].to_list()

    with contextlib.suppress(ValueError):
        if min(ylist) == 0 or math.log10(max(ylist) / min(ylist)) > 2:
            ax.set_yscale("log")

    dictcolors = {
        "Te": "red"
        # 'heating_gamma': 'blue',
        # 'cooling_adiabatic': 'blue'
    }

    if startfromzero:
        # make a line segment from 0 velocity
        xlist = [0.0, *xlist]
        ylist = [ylist[0], *ylist]

    xlist_filtered, ylist_filtered = at.estimators.apply_filters(xlist, ylist, args)

    ax.plot(
        xlist_filtered, ylist_filtered, linewidth=1.5, label=linelabel, color=dictcolors.get(variablename), **plotkwargs
    )


def get_xlist(
    xvariable: str, estimators: pl.LazyFrame, timestepslist: t.Any, groupbyxvalue: bool, args: t.Any
) -> tuple[list[float | int], list[int], list[list[int]], pl.LazyFrame]:
    estimators = estimators.filter(pl.col("timestep").is_in(set(chain.from_iterable(timestepslist))))

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
        assert xvariable in estimators.collect_schema().names()
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
    assert len(pointgroups) > 0, "No data found for x-axis variable"

    return (
        pointgroups["xvalue"].to_list(),
        pointgroups["modelgridindex"].to_list(),
        pointgroups["timestep"].to_list(),
        estimators,
    )


def plot_subplot(
    ax: mplax.Axes,
    timestepslist: list[list[int]],
    xlist: list[float | int],
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
    legend_kwargs = {}
    seriescount = 0
    ylabel = None
    sameylabel = True
    seriesvars = [var for var in plotitems if isinstance(var, str | pl.Expr)]
    seriescount = len(seriesvars)

    for variable in seriesvars:
        variablename = variable.meta.output_name() if isinstance(variable, pl.Expr) else variable
        if ylabel is None:
            ylabel = get_ylabel(variablename)
        elif ylabel != get_ylabel(variablename):
            sameylabel = False
            break

    for plotitem in plotitems:
        if isinstance(plotitem, str | pl.Expr):
            variablename = plotitem.meta.output_name() if isinstance(plotitem, pl.Expr) else plotitem
            assert isinstance(variablename, str)
            showlegend = seriescount > 1 or len(variablename) > 35 or not sameylabel
            plot_series(
                ax=ax,
                startfromzero=startfromzero,
                variable=plotitem,
                showlegend=showlegend,
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
                    specieslist=params,
                    estimators=estimators,
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
                    modelpath,
                    startfromzero=startfromzero,
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
                if seriestype == "populations" and len(ionlist) > 2 and args.yscale == "log":
                    legend_kwargs["ncol"] = 2

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
        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1, **legend_kwargs, markerscale=3)


def make_plot(
    modelpath: Path | str,
    timestepslist_unfiltered: list[list[int]],
    estimators: pl.LazyFrame,
    xvariable: str,
    plotlist: list[list[t.Any]],
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
        layout="constrained",
        # tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )
    if len(plotlist) == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

    for ax in axes:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    if not args.hidexlabel:
        axes[-1].set_xlabel(
            f"{at.estimators.get_varname_formatted(xvariable)}{at.estimators.get_units_string(xvariable)}"
        )

    xlist, mgilist, timestepslist, estimators = get_xlist(
        xvariable=xvariable,
        estimators=estimators,
        timestepslist=timestepslist_unfiltered,
        groupbyxvalue=not args.markersonly,
        args=args,
    )
    startfromzero = (xvariable.startswith("velocity") or xvariable == "beta") and not args.markersonly
    xmin = args.xmin if args.xmin >= 0 else min(xlist)
    xmax = args.xmax if args.xmax > 0 else max(xlist)

    if args.markersonly:
        plotkwargs |= {"linestyle": "None", "marker": ".", "markersize": 4, "alpha": 0.7, "markeredgewidth": 0}

        # with no lines, line styles cannot distinguish ions
        args.colorbyion = True

    for ax, plotitems in zip(axes, plotlist, strict=False):
        if xmin != xmax:
            ax.set_xlim(left=xmin, right=xmax)

        plot_subplot(
            ax=ax,
            timestepslist=timestepslist,
            xlist=xlist,
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
        print("Plotting " + figure_title.replace("\n", " "))

        defaultoutputfile = "plotestimators_{timestep}_{timedays}.{format}"
        if Path(args.outputfile).is_dir():
            args.outputfile = str(Path(args.outputfile) / defaultoutputfile)

        assert isinstance(timestepslist[0], list)
        outfilename = str(args.outputfile).format(timestep=timestep, timedays=timedays, format=args.format)

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=10)

    print(f"Saving {outfilename}")
    fig.savefig(outfilename, dpi=300)

    if args.show:
        plt.show()
    else:
        plt.close()

    return outfilename


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath", default=".", help="Paths to ARTIS folder (or virtual path e.g. codecomparison/ddc10/cmfgen)"
    )

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
        "-poptype",
        dest="poptype",
        default="elpop",
        choices=["absolute", "totalpop", "elpop", "radialdensity", "cylradialdensity", "cumulative"],
        help="Plot absolute ion populations, or ion populations as a fraction of total or element population",
    )

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument(
        "-figscale", type=float, default=1.0, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("-scalefigwidth", type=float, default=1.0, help="Scale factor for plot width.")

    parser.add_argument("--show", action="store_true", help="Show plot before quitting")

    parser.add_argument(
        "-outputfile",
        "-outputpath",
        "-o",
        action="store",
        dest="outputfile",
        type=Path,
        default=Path(),
        help="Filename for PDF file",
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


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
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
        # [
        #     (pl.col("heating_coll") - pl.col("cooling_coll")).alias("collisional heating - cooling"),
        #     ["_yscale", "linear"],
        # ],
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

        modeldata, _ = at.inputmodel.get_modeldata_pandas(modelpath)
        estimatorsdict = artistools.estimators.estimators_classic.read_classic_estimators(modelpath, modeldata)
        assert estimatorsdict is not None
        estimators = pl.DataFrame([
            {"timestep": ts, "modelgridindex": mgi, **estimvals} for (ts, mgi), estimvals in estimatorsdict.items()
        ]).lazy()
    else:
        estimators = at.estimators.scan_estimators(
            modelpath=modelpath, modelgridindex=args.modelgridindex, timestep=tuple(timesteps_included)
        )

    estimators, modelmeta = at.estimators.join_cell_modeldata(estimators=estimators, modelpath=modelpath, verbose=False)
    estimators = estimators.filter(pl.col("vel_r_mid") <= modelmeta["vmax_cmps"])
    tswithdata = estimators.select("timestep").unique().collect().to_series()
    for ts in reversed(timesteps_included):
        if ts not in tswithdata:
            timesteps_included.remove(ts)
            print(f"ts {ts} requested but no data found. Removing.")

    if not timesteps_included:
        print("No timesteps with data are included")
        return

    assoc_cells, _ = at.get_grid_mapping(modelpath)

    outdir = Path(args.outputfile) if Path(args.outputfile).is_dir() else Path()

    if not args.readonlymgi and (args.modelgridindex is not None or args.x in {"time", "timestep"}):
        # plot time evolution in specific cell
        if not args.x:
            args.x = "time"
        assert isinstance(args.modelgridindex, int)
        timestepslist_unfiltered = [[ts] for ts in timesteps_included]
        if not assoc_cells.get(args.modelgridindex):
            msg = f"cell {args.modelgridindex} is empty. no estimators available"
            raise ValueError(msg)
        make_plot(
            modelpath=modelpath,
            timestepslist_unfiltered=timestepslist_unfiltered,
            estimators=estimators,
            xvariable=args.x,
            plotlist=plotlist,
            args=args,
        )
    else:
        # plot a range of cells in a time snapshot showing internal structure

        if not args.x:
            args.x = "velocity"

        if args.x == "velocity" and modelmeta["vmax_cmps"] > 0.3 * 29979245800:
            args.x = "beta"

        if args.readonlymgi:
            estimators = estimators.filter(pl.col("modelgridindex").is_in(args.modelgridindex))

        if args.classicartis:
            modeldata, _ = at.inputmodel.get_modeldata_pandas(modelpath)
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
                        writer.append_data(image)  # type: ignore[attr-defined] # pyright: ignore[reportAttributeAccessIssue]
                print(f"Created gif: {gifname}")
            elif args.format == "pdf":
                at.merge_pdf_files(outputfiles)


if __name__ == "__main__":
    main()
