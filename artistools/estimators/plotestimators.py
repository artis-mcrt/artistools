#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Functions for plotting artis estimators and internal structure.

Examples are temperatures, populations, heating/cooling rates.
"""

import argparse
import math
import string
import typing as t
from collections.abc import Collection
from collections.abc import Sequence
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

VARIABLE_ALIASES = {"T_e": "Te", "n_e": "nne", "T_R": "TR", "T_J": "TJ"}


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


def adjust_lightness(color: t.Any, amount: float = 0.5) -> tuple[float, float, float]:
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except (SyntaxWarning, KeyError, TypeError):
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot_data(
    dfplotdata: pl.DataFrame | pl.LazyFrame,
    ax: mplax.Axes,
    label: str | None,
    args: argparse.Namespace,
    startfromzero: bool = False,
    **plotkwargs: t.Any,
) -> None:
    dfplotdata = dfplotdata.lazy()

    # Calculate the average line and optionally, the min-max bounding area
    dflinepoints = (
        dfplotdata.group_by("xvalue_binned", maintain_order=True)
        .agg(
            yvalue_binned=(pl.col("yvalue") * pl.col("celltsweight")).sum() / pl.col("celltsweight").sum(),
            yvalue_binned_min=pl.col("yvalue").min(),
            yvalue_binned_max=pl.col("yvalue").max(),
        )
        .sort("xvalue_binned")
        .drop_nans()
    )

    filterfunc = at.get_filterfunc(args)
    if filterfunc is not None:
        dflinepoints = dflinepoints.with_columns(
            pl.col("yvalue_binned").map_batches(filterfunc, return_dtype=pl.self_dtype())
        )

    if startfromzero:
        dflinepoints = pl.concat([
            pl.LazyFrame(
                {
                    "xvalue_binned": [0.0],
                    **{
                        col: [dflinepoints.select(pl.col(col).head(1)).collect().item()]
                        for col in dflinepoints.collect_schema().names()
                        if col != "xvalue_binned"
                    },
                },
                schema=dflinepoints.collect_schema(),
            ),
            dflinepoints,
        ]).lazy()

    xvalues_binned = dflinepoints.select(["xvalue_binned"]).collect().get_column("xvalue_binned")
    yvalues_binned = dflinepoints.select(["yvalue_binned"]).collect().get_column("yvalue_binned")

    (plotobj,) = ax.plot(xvalues_binned, yvalues_binned, label=label, **plotkwargs)
    color = plotobj.get_color()

    if args.markers:
        plotkwargs_markers = plotkwargs | {
            "linestyle": "None",
            "marker": ".",
            "markersize": 5,
            "color": adjust_lightness(color, 1.5),
            # "alpha": 0.4,
            "markeredgewidth": 0,
            "zorder": -1,
        }
        plotkwargs_markers.pop("dashes", None)
        plotkwargs_markers.pop("label", None)
        if dfplotdata.select(pl.len() > 10000).collect().item():
            plotkwargs_markers["rasterized"] = True
        # plot the markers first
        ax.plot(
            dfplotdata.select("xvalue").collect().to_series(),
            dfplotdata.select("yvalue").collect().to_series(),
            **plotkwargs_markers,
        )

    else:
        yvalues_binned_min = dflinepoints.select("yvalue_binned_min").collect().get_column("yvalue_binned_min")
        yvalues_binned_max = dflinepoints.select("yvalue_binned_max").collect().get_column("yvalue_binned_max")
        plotobj = ax.fill_between(
            xvalues_binned, yvalues_binned_min, yvalues_binned_max, alpha=0.2, color=color, linewidth=0, zorder=-2
        )


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

        linestyle = "-"
        if speciesstr.lower() in {"ni_56", "ni56", "56ni"}:
            expr_yvalue = pl.col(f"{valuetype}Ni56")
            linelabel = "$^{56}$Ni"
            linestyle = "--"
        elif speciesstr.lower() in {"ni_stb", "ni_stable"}:
            expr_yvalue = pl.col(f"{valuetype}{elsymbol}") - pl.col(f"{valuetype}Ni56")
            linelabel = "Stable Ni"
        elif speciesstr.lower() in {"co_56", "co56", "56co"}:
            expr_yvalue = pl.col(f"{valuetype}Co56")
            linelabel = "$^{56}$Co"
        elif speciesstr.lower() in {"fegrp", "ffegroup"}:
            expr_yvalue = pl.col(f"{valuetype}Fegroup")
            linelabel = "Fe group"
        else:
            linelabel = speciesstr
            expr_yvalue = pl.col(f"{valuetype}{elsymbol}")

        plotkwargs["color"] = get_elemcolor(atomic_number=atomic_number)
        plotkwargs.setdefault("linewidth", 1.5)
        series = estimators.with_columns(celltsweight=pl.col("rho") * pl.col("deltavol_deltat"), yvalue=expr_yvalue)

        if "linestyle" not in plotkwargs:
            plotkwargs["linestyle"] = linestyle

        plot_data(series, ax=ax, args=args, startfromzero=startfromzero, label=linelabel, **plotkwargs)


def plot_average_ionisation(
    ax: mplax.Axes,
    params: Sequence[str],
    estimators: pl.LazyFrame,
    startfromzero: bool,
    args: argparse.Namespace | None = None,
    **plotkwargs: t.Any,
) -> None:
    if args is None:
        args = argparse.Namespace()

    ax.set_ylabel("Average ion charge")

    for paramvalue in params:
        print(f"  plotting averageionisation {paramvalue}")
        atomic_number = at.get_atomic_number(paramvalue)

        color = get_elemcolor(atomic_number=atomic_number)
        elsymb = at.get_elsymbol(atomic_number)
        if f"nnelement_{elsymb}" not in estimators.collect_schema().names():
            msg = f"ERROR: No element data found for {paramvalue}"
            raise ValueError(msg)

        ioncols = [col for col in estimators.collect_schema().names() if col.startswith(f"nnion_{elsymb}_")]
        ioncharges = [at.decode_roman_numeral(col.removeprefix(f"nnion_{elsymb}_")) - 1 for col in ioncols]
        ax.set_ylim(0.0, max(ioncharges) + 0.1)
        expr_charge_per_nuc = pl.sum_horizontal([
            ioncharge * pl.col(ioncol) for ioncol, ioncharge in zip(ioncols, ioncharges, strict=True)
        ]) / pl.col(f"nnelement_{elsymb}")

        dfplotdata = estimators.with_columns(
            celltsweight=pl.col(f"nnelement_{elsymb}") * pl.col("deltavol_deltat"), yvalue=expr_charge_per_nuc
        ).filter(pl.col(f"nnelement_{elsymb}") > 0.0)

        plot_data(
            dfplotdata=dfplotdata,
            ax=ax,
            args=args,
            startfromzero=startfromzero,
            label=paramvalue,
            color=color,
            **plotkwargs,
        )


def plot_levelpop(
    ax: mplax.Axes,
    xlist: Sequence[int | float] | npt.NDArray[np.floating],
    seriestype: str,
    params: Sequence[str],
    timestepslist: Sequence[int],
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

    modeldata = (
        at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g", "volume"])[0]
        .collect()
        .to_pandas(use_pyarrow_extension_array=True)
    )

    adata = (
        at.atomic.get_levels(modelpath)
        .with_columns(
            levels=pl.col("levels").map_elements(
                lambda x: x.to_pandas(use_pyarrow_extension_array=True), return_dtype=pl.Object
            )
        )
        .to_pandas(use_pyarrow_extension_array=True)
    )

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
        for modelgridindex in mgilist:
            valuesum = 0.0
            tdeltasum = 0.0
            # print(f'modelgridindex {modelgridindex} timesteps {timesteps}')

            for timestep in timestepslist:
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

        plot_data(
            pl.DataFrame({"xvalue": xlist, "yvalue": ylist}),
            ax=ax,
            args=args,
            startfromzero=startfromzero,
            label=label,
            **plotkwargs,
        )


def get_iontuple(ionstr: str) -> tuple[int, str | int]:
    """Decode into atomic number and parameter, e.g., [(26, 1), (26, 2), (26, 'ALL'), (26, 'Fe56')]."""
    if ionstr in at.get_elsymbolslist():
        return (at.get_atomic_number(ionstr), "ALL")

    # a space separates the element symbol from the ionstage, e.g. Fe II
    if " " in ionstr:
        return (at.get_atomic_number(ionstr.split(" ", maxsplit=1)[0]), at.decode_roman_numeral(ionstr.split(" ")[1]))

    # for element symbol with a mass number after it, e.g. Fe56
    if ionstr.rstrip("-0123456789") in at.get_elsymbolslist():
        atomic_number = at.get_atomic_number(ionstr.rstrip("-0123456789"))
        return (atomic_number, ionstr)

    # for element and ionstage without a space, e.g. FeII
    for elsymb in at.get_elsymbolslist():
        if ionstr.startswith(elsymb):
            possible_roman = at.decode_roman_numeral(ionstr.removeprefix(elsymb))
            if possible_roman > 0:
                return (at.get_atomic_number(elsymb), possible_roman)

    atomic_number = at.get_atomic_number(ionstr.split("_", maxsplit=1)[0])
    return (atomic_number, ionstr)


def get_column_name(seriestype: str, atomic_number: int, ion_stage: str | int) -> tuple[str, str]:
    ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
    if seriestype == "populations":
        if ion_stage == "ALL":
            elsymbol = at.get_elsymbol(atomic_number)
            return f"nnelement_{elsymbol}", ionstr
        if isinstance(ion_stage, str) and ion_stage.startswith(at.get_elsymbol(atomic_number)):
            # not really an ion_stage but an isotope name
            return f"nniso_{ion_stage}", ionstr
        return f"nnion_{ionstr}", ionstr
    return f"{seriestype}_{ionstr}", ionstr


def plot_multi_ion_series(
    ax: mplax.Axes,
    startfromzero: bool,
    seriestype: str,
    ionlist: Sequence[str],
    estimators: pl.LazyFrame,
    modelpath: str | Path,
    args: argparse.Namespace,
    ymin: float | None = None,
    ymax: float | None = None,
    **plotkwargs: t.Any,
) -> None:
    """Plot an ion-specific property, e.g., populations."""
    # if seriestype == 'populations':
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.10))

    plotted_something = False

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
                    and compositiondata.filter(
                        (pl.col("Z") == atomic_number)
                        & (pl.col("lowermost_ion_stage") <= ion_stage)
                        & (pl.col("uppermost_ion_stage") >= ion_stage)
                    ).is_empty()
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
        colname, ionstr = get_column_name(seriestype, atomic_number, ion_stage)
        expr_yvals = pl.col(colname)
        print(f"  plotting {seriestype} {ionstr.replace('_', ' ')}")

        if seriestype != "populations" or args.poptype == "absolute":
            expr_normfactor = pl.lit(1)
        elif args.poptype == "elpop":
            elsymbol = at.get_elsymbol(atomic_number)
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

        # convert volumetric number density to radial density
        if args.poptype == "radialdensity":
            expr_yvals *= 4 * math.pi * pl.col("vel_r_mid").mean().pow(2)
        elif args.poptype == "cylradialdensity":
            expr_yvals *= 2 * math.pi * pl.col("vel_rcyl_mid").mean()

        if args.poptype == "cumulative":
            expr_yvals = expr_yvals.cum_sum()

        lazyframes.append(
            estimators.with_columns(celltsweight=pl.col("deltavol_deltat"), yvalue=expr_yvals / expr_normfactor)
        )

    for seriesindex, (iontuple, dfseries) in enumerate(zip(iontuplelist, pl.collect_all(lazyframes), strict=True)):
        atomic_number, ion_stage = iontuple
        plotlabel = str(
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

        dashes_list = [(), (3, 1, 1, 1), (1.5, 1.5), (6, 3), (1, 3)]
        dashes = dashes_list[styleindex % len(dashes_list)]

        linewidth_list = [1.0, 1.0, 1.0, 0.7, 0.7]
        linewidth = linewidth_list[styleindex % len(linewidth_list)] * 1.5

        if plotkwargs.get("linestyle", "solid") != "None":
            plotkwargs["dashes"] = dashes

        plot_data(
            dfseries,
            linewidth=linewidth,
            label=plotlabel,
            ax=ax,
            args=args,
            startfromzero=startfromzero,
            color=color,
            **plotkwargs,
        )
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

    if plotted_something and ax.get_yscale() == "log":
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

    series = estimators.with_columns(celltsweight=pl.col("deltavol_deltat"), yvalue=colexpr)

    if variablename in (dictcolors := {"Te": "red", "heating_gamma": "blue", "cooling_adiabatic": "blue"}):
        plotkwargs.setdefault("color", dictcolors[variablename])
    plotkwargs.setdefault("linewidth", 1.5)

    print(f"  plotting {variablename}")
    plot_data(series, ax=ax, label=linelabel, args=args, startfromzero=startfromzero, **plotkwargs)


def get_xlist(
    xvariable: str, estimators: pl.LazyFrame, timestepslist: Collection[int] | None, args: t.Any
) -> tuple[list[float | int], list[int], list[int], pl.LazyFrame]:
    if timestepslist is not None:
        estimators = estimators.filter(pl.col("timestep").is_in(timestepslist))

    if xvariable in {"cellid", "modelgridindex"}:
        estimators = estimators.with_columns(xvalue=pl.col("modelgridindex"))
    elif xvariable == "timestep":
        estimators = estimators.with_columns(xvalue=pl.col("timestep"))
    elif xvariable == "time":
        estimators = estimators.with_columns(xvalue=pl.col("tmid_days"))
    elif xvariable in {"velocity", "beta"}:
        velcolumn = "vel_r_mid"
        scalefactor = 1e5 if xvariable == "velocity" else 29979245800
        estimators = estimators.with_columns(xvalue=(pl.col(velcolumn) / scalefactor))
    else:
        assert xvariable in estimators.collect_schema().names()
        estimators = estimators.with_columns(xvalue=pl.col(xvariable))

    xmin = estimators.select(pl.col("xvalue").min()).collect().item() if args.xmin is None else args.xmin
    xmax = estimators.select(pl.col("xvalue").max()).collect().item() if args.xmax is None else args.xmax

    if (
        args.xbins is None
        and estimators.select(pl.n_unique("xvalue") * pl.n_unique("timestep") < pl.len()).collect().item()
    ):
        print("There are multiple plot points per x value. Using automatic bins (use -xbins N to change this)")
        args.xbins = -1
        args.colorbyion = True

    if args.xbins is not None and args.xbins < 0:
        xdeltamax = estimators.select(pl.col("xvalue").sort().diff().max()).collect().item()
        args.xbins = int((xmax - xmin) / xdeltamax)
        print(
            f"Setting xbins to {args.xbins} based on data range [{xmin}, {xmax}] and largest x interval of {xdeltamax}"
        )
        if args.xbins <= 3:
            print(f"  would have only {args.xbins} bins. Replacing with 25")
            args.xbins = 25

    if args.xbins is not None:
        xbinedges = np.linspace(xmin, xmax, args.xbins)
        xlower = xbinedges[:-1]
        xupper = xbinedges[1:]
        xmids = (xlower + xupper) / 2
        estimators = (
            estimators.with_columns(
                pl.col("xvalue")
                .cut(breaks=list(xbinedges), labels=[str(x) for x in range(-1, len(xbinedges))])
                .cast(pl.Utf8)
                .cast(pl.Int32)
                .alias("xbinindex")
            )
            .filter(pl.col("xbinindex").is_between(0, len(xmids) - 1, closed="both"))
            .join(pl.LazyFrame({"xvalue_binned": xmids}).with_row_index("xbinindex"), on="xbinindex", how="left")
            .drop("xbinindex")
        )
    else:
        estimators = estimators.with_columns(xvalue_binned=pl.col("xvalue"))

    if args.xmin is not None:
        estimators = estimators.filter(pl.col("xvalue") >= args.xmin)

    if args.xmax is not None:
        estimators = estimators.filter(pl.col("xvalue") <= args.xmax)

    estimators = estimators.sort("xvalue")
    xvalues = estimators.select("xvalue").unique().collect().get_column("xvalue")
    assert len(xvalues) > 0, "No data found for x-axis variable"

    return (
        xvalues.to_list(),
        estimators.select(pl.col("modelgridindex").unique()).collect()["modelgridindex"].to_list(),
        estimators.select(pl.col("timestep").unique()).collect()["timestep"].to_list(),
        estimators,
    )


def plot_subplot(
    ax: mplax.Axes,
    timestepslist: list[int],
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
    print(f"Subplot: {plotitems}")
    for variable in seriesvars:
        variablename = variable.meta.output_name() if isinstance(variable, pl.Expr) else variable
        if ylabel is None:
            ylabel = get_ylabel(variablename)
        elif ylabel != get_ylabel(variablename):
            sameylabel = False
            break
    ymin, ymax = None, None
    for plotitem in plotitems:
        if isinstance(plotitem, str | pl.Expr):
            continue
        seriestype, params = plotitem
        if seriestype == "_ymin":
            ymin = float(params) if isinstance(params, str) else params
            ax.set_ylim(bottom=ymin)

        elif seriestype == "_ymax":
            ymax = float(params) if isinstance(params, str) else params
            ax.set_ylim(top=ymax)

        elif seriestype == "_yscale":
            ax.set_yscale(params)

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
            showlegend = True
            if isinstance(params, str) and seriestype.startswith("_"):
                continue

            if seriestype in {"initabundances", "initmasses"}:
                assert isinstance(params, list)
                plot_init_abundances(
                    ax=ax,
                    specieslist=params,
                    estimators=estimators,
                    seriestype=seriestype,
                    startfromzero=startfromzero,
                    args=args,
                    **plotkwargs,
                )

            elif seriestype == "levelpopulation" or seriestype.startswith("levelpopulation_"):
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

            elif seriestype == "averageionisation":
                plot_average_ionisation(ax, params, estimators, startfromzero=startfromzero, args=args, **plotkwargs)

            else:
                seriestype, ionlist = plotitem
                if seriestype.startswith("_"):
                    continue
                ax.set_yscale("log")
                if seriestype == "populations" and len(ionlist) > 2 and ax.get_yscale() == "log":
                    legend_kwargs["ncol"] = 2

                plot_multi_ion_series(
                    ax=ax,
                    startfromzero=startfromzero,
                    seriestype=seriestype,
                    ionlist=ionlist,
                    estimators=estimators,
                    modelpath=modelpath,
                    args=args,
                    ymin=ymin,
                    ymax=ymax,
                    **plotkwargs,
                )

    ax.tick_params(right=True)
    if showlegend and not args.nolegend:
        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1, **legend_kwargs, markerscale=3)


def make_figure(
    modelpath: Path | str,
    timestepslist: Collection[int] | None,
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
        xvariable=xvariable, estimators=estimators, timestepslist=timestepslist, args=args
    )

    startfromzero = xvariable.startswith("velocity") or xvariable == "beta"
    xmin = args.xmin if args.xmin is not None else min(xlist)
    xmax = args.xmax if args.xmax is not None else max(xlist)

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

    if len(set(mgilist)) == 1 and len(timestepslist) > 1:  # single grid cell versus time plot
        figure_title = f"{modelname}\nCell {mgilist[0]}"

        defaultoutputfile = "plotestimators_cell{modelgridindex:03d}.{format}"
        if Path(args.outputfile).is_dir():
            args.outputfile = str(Path(args.outputfile) / defaultoutputfile)

        outfilename = str(args.outputfile).format(modelgridindex=mgilist[0], format=args.format)

    else:
        if args.multiplot:
            strtimestep = f"ts{timestepslist[0]:02d}"
            strtimedays = f"{at.get_timestep_time(modelpath, timestepslist[0]):.2f}d"
        else:
            timesteps_flat = at.flatten_list(timestepslist)
            timestepmin = min(timesteps_flat)
            timestepmax = max(timesteps_flat)

            strtimestep = (
                f"ts{timestepmin:02d}-ts{timestepmax:02d}" if timestepmax != timestepmin else f"ts{timestepmin:02d}"
            )
            dftimesteps = at.get_timesteps(modelpath)
            timelow_days = (
                dftimesteps.filter(pl.col("timestep") == timestepmin).select(pl.col("tstart_days")).collect().item()
            )
            timehigh_days = (
                dftimesteps.filter(pl.col("timestep") == timestepmax).select(pl.col("tend_days")).collect().item()
            )
            strtimedays = f"{timelow_days:.2f}d-{timehigh_days:.2f}d"

        figure_title = f"{modelname}\nTimestep {strtimestep} ({strtimedays})"
        print("  plotting " + figure_title.replace("\n", " "))

        defaultoutputfile = "plotestimators_{timestep}_{timedays}.{format}"
        if Path(args.outputfile).is_dir():
            args.outputfile = str(Path(args.outputfile) / defaultoutputfile)

        assert isinstance(timestepslist, list)
        outfilename = str(args.outputfile).format(timestep=strtimestep, timedays=strtimedays, format=args.format)

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=10)

    print(f"open {outfilename}")
    fig.savefig(outfilename, dpi=600)

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

    parser.add_argument("-x", default=None, help="Horizontal axis variable, e.g. velocity, timestep, or time")

    parser.add_argument("-xmin", type=float, default=None, help="Plot range: minimum x value")

    parser.add_argument("-xmax", type=float, default=None, help="Plot range: maximum x value")

    parser.add_argument(
        "-xbins", type=int, default=None, help="Number of x bins between xmax and xmin (or -1 for automatic bin size)"
    )

    parser.add_argument("--hidexlabel", action="store_true", help="Hide the bottom horizontal axis label")

    parser.add_argument("--markers", action="store_true", help="Plot markers instead of shaded area")

    parser.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")

    parser.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and polyorder.e.g. -filtersavgol 5 3",
    )

    parser.add_argument("-format", "-f", default="pdf", choices=["pdf", "png"], help="Set format of output plot files")

    parser.add_argument("--makegif", action="store_true", help="Make a gif with time evolution (requires --multiplot)")

    parser.add_argument("--notitle", action="store_true", help="Suppress the top title from the plot")

    parser.add_argument(
        "-plotlist",
        "-plot",
        "-p",
        nargs="*",
        type=str,
        action="append",
        help="List of plots to generate. Specify estimator names or population types. Examples: -plot Te TR -plot nne -plot SrI 'Sr II'",
    )

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

    should_use_all_timesteps = (
        not args.timedays
        and not args.timestep
        and (args.modelgridindex is not None or args.x in {None, "time", "timestep"})
    )

    if should_use_all_timesteps:
        args.timestep = f"0-{len(at.get_timestep_times(modelpath)) - 1}"
        if args.x is None:
            args.x = "time"
            print(f"Setting x variable to {args.x}")
    elif args.x is None:
        args.x = "velocity"
        print(f"Setting x variable to {args.x}")

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

        estimatorsdict = artistools.estimators.estimators_classic.read_classic_estimators(modelpath)
        assert estimatorsdict is not None
        estimators = pl.DataFrame([
            {"timestep": ts, "modelgridindex": mgi, **estimvals} for (ts, mgi), estimvals in estimatorsdict.items()
        ]).lazy()
    else:
        estimators = at.estimators.scan_estimators(
            modelpath=modelpath, modelgridindex=args.modelgridindex, timestep=tuple(timesteps_included)
        )

    estimators, modelmeta = at.estimators.join_cell_modeldata(estimators=estimators, modelpath=modelpath, verbose=False)
    if estimators.select(pl.len()).collect().item() == 0:
        print("No data was found for the requested timesteps/cells.")
        estimators = at.estimators.scan_estimators(modelpath=modelpath)
        print("Cells with data: ")
        print(estimators.select(pl.col("modelgridindex").unique().sort()).collect().to_series().to_list())
        print("Timesteps with data: ")
        print(estimators.select(pl.col("timestep").unique().sort()).collect().to_series().to_list())
        return

    if args.modelgridindex is None:
        estimators = estimators.filter(pl.col("vel_r_mid") <= modelmeta["vmax_cmps"])

    estimators = estimators.with_columns(deltavol_deltat=pl.col("volume") * pl.col("twidth_days"))

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
        # [['gamma_NT', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
    ]

    estimatorcolumns = estimators.collect_schema().names()

    # detect a list of populations and convert to the correct form
    # e.g. ["Sr I", "Sr II"] is re-written to [['populations', ['Sr I', 'Sr II']]]
    for i in range(len(plotlist)):
        if isinstance(plotlist[i], str):
            plotlist[i] = [plotlist[i]]
        assert isinstance(plotlist[i], list)
        plot_directives = [
            plotvar.split("=", maxsplit=1)
            for plotvar in plotlist[i]
            if isinstance(plotvar, str) and plotvar.startswith("_") and "=" in plotvar
        ]
        plotlist[i] = [
            VARIABLE_ALIASES.get(plotvar, plotvar) if isinstance(plotvar, str) else plotvar
            for plotvar in plotlist[i]
            if not isinstance(plotvar, str) or not plotvar.startswith("_") or "=" not in plotvar
        ]
        if isinstance(plotlist[i][0], str) and plotlist[i][0] not in estimatorcolumns:
            # this is going to cause an error, so attempt to interpret it as populations
            rewrite_is_valid = False
            for plotvar in plotlist[i]:
                if isinstance(plotvar, list):
                    continue
                if not isinstance(plotvar, str):
                    break
                if plotvar.startswith("_"):
                    continue
                atomic_number, ionstage = get_iontuple(plotvar)
                if get_column_name("populations", atomic_number, ionstage)[0] not in estimatorcolumns:
                    break
            else:
                rewrite_is_valid = True
            if rewrite_is_valid:
                new_plotvars = [["populations", plotlist[i]]]
                print(f"Rewriting plotlist {plotlist[i]} to {new_plotvars}")
                plotlist[i] = new_plotvars
        if plot_directives:
            plotlist[i].extend(plot_directives)

    outdir = Path(args.outputfile) if Path(args.outputfile).is_dir() else Path()
    assert args.x is not None
    if args.x in {"time", "timestep"}:
        # plot time evolution
        make_figure(
            modelpath=modelpath,
            timestepslist=timesteps_included,
            estimators=estimators,
            xvariable=args.x,
            plotlist=plotlist,
            args=args,
        )
    else:
        # plot a range of cells in a time snapshot showing internal structure

        if args.x == "velocity" and modelmeta["vmax_cmps"] > 0.3 * 29979245800:
            args.x = "beta"

        if args.readonlymgi:
            if not isinstance(args.modelgridindex, list):
                args.modelgridindex = [args.modelgridindex] if args.modelgridindex is not None else []
            estimators = estimators.filter(pl.col("modelgridindex").is_in(args.modelgridindex))

        frames_timesteps_included = (
            [[ts] for ts in range(timestepmin, timestepmax + 1)] if args.multiplot else [timesteps_included]
        )

        if args.makegif:
            args.multiplot = True
            args.format = "png"

        outputfiles = []
        for timesteps_included in frames_timesteps_included:
            outfilename = make_figure(
                modelpath=modelpath,
                timestepslist=timesteps_included,
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
