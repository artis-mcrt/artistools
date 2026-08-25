# PYTHON_ARGCOMPLETE_OK
"""Functions for plotting artis estimators and internal structure.

Examples are temperatures, populations, heating/cooling rates.
"""

import argparse
import math
import string
import typing as t
from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import matplotlib.axes as mplax
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars import selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import km_to_cm
from artistools.constants import Msun_to_g
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_dpi
from artistools.misc import addarg_figscale
from artistools.misc import addarg_filter
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_nolegend
from artistools.misc import addarg_notitle
from artistools.misc import addarg_outputfile
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timeminmax
from artistools.misc import addarg_timestep
from artistools.plottools import save_figure
from artistools.plottools import set_axis_properties
from artistools.plottools import set_plot_title

colors_tab10 = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
]

# reserve colours for these elements. Immutable, because get_unreserved_elemcolors() derives the rest of the
# palette from it once and caches the result, so a later mutation would silently hand a reserved colour out twice
elementcolors: t.Final[Mapping[str, tuple[float, float, float, float]]] = MappingProxyType({
    "Fe": colors_tab10[0],
    "Ni": colors_tab10[1],
    "Co": colors_tab10[2],
})

VARIABLE_ALIASES = {"T_e": "Te", "n_e": "nne", "T_R": "TR", "T_J": "TJ"}

POPTYPE_YLABELS: t.Final[Mapping[str, str]] = MappingProxyType({
    "absolute": r"Number density $\left[\rm{cm}^{-3}\right]$",
    "elpop": r"X$_{i}$/X$_{\rm element}$",
    "totalpop": r"X$_{i}$/X$_{\rm tot}$",
    "radialdensity": r"Radial density dN/dr $\left[\rm{cm}^{-1}\right]$",
    "cylradialdensity": r"Cylindrical radial density dN/drcyl $\left[\rm{cm}^{-1}\right]$",
    "cumulative": r"Cumulative particle count",
})


def get_elemcolor(atomic_number: int | None = None, elsymbol: str | None = None) -> t.Any:
    """Return the plot colour of an element, keyed on the element itself so that it never varies between plots.

    The three reserved elements keep their colour. Every other element takes one from a long palette indexed by
    atomic number, so a given element gets the same colour in every figure of every run. Handing out colours in
    call order instead made an element's colour depend on which plots preceded it in the same process, and ran
    off the end of the ten-colour list once eleven elements had been seen.
    """
    assert (atomic_number is None) != (elsymbol is None)
    if atomic_number is None:
        assert elsymbol is not None
        atomic_number = at.get_atomic_number(elsymbol)
        if atomic_number < 0:
            msg = f"{elsymbol!r} is not an element symbol, so it has no colour"
            raise ValueError(msg)
    else:
        elsymbol = at.get_elsymbol(atomic_number)

    if elsymbol in elementcolors:
        return elementcolors[elsymbol]

    palette = get_unreserved_elemcolors()

    return palette[atomic_number % len(palette)]


@lru_cache(maxsize=1)
def get_unreserved_elemcolors() -> tuple[t.Any, ...]:
    """Return the colours available to elements with no reserved colour, in a stable order.

    get_unused_colors compares by value, thus a rounded copy of a reserved colour is also removed. A
    comparison of the tuples let the rounded glasbey copies of the tab10 colours through, which gave
    nitrogen the blue of iron and oxygen the orange of nickel in one figure.
    """
    from artistools.plottools import get_unused_colors
    from artistools.plottools import glasbey_category20_nogreys

    palette = [*colors_tab10, *glasbey_category20_nogreys]

    return tuple(get_unused_colors(palette, [mc.to_hex(rgba) for rgba in elementcolors.values()]))


def get_ylabel(variable: str) -> str:
    """Return the y-axis label for an estimator variable, preferring its long units over the short ones."""
    return at.estimators.get_variablelongunits(variable) or at.estimators.get_units_string(variable)


def adjust_lightness(color: t.Any, amount: float = 0.5) -> tuple[float, float, float]:
    """Return the colour with its lightness scaled by amount, so related series can share a hue."""
    import colorsys

    try:
        c = mc.cnames[color]
    except (SyntaxWarning, KeyError, TypeError):
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0.0, min(1.0, amount * c[1])), c[2])


def plot_data(
    dfplotdata: pl.DataFrame | pl.LazyFrame,
    ax: mplax.Axes,
    label: str | None,
    args: argparse.Namespace,
    startfromzero: bool = False,
    **plotkwargs: t.Any,
) -> None:
    """Plot a series with an average line and optionally, a min-max bounding area.

    These columns are required: xvalue, xvalue_binned, yvalue, celltsweight (the weight for averaging, e.g., cell volume times timestep duration).
    """
    dfplotdata = dfplotdata.lazy()

    # Calculate the average line and optionally, the min-max bounding area
    dflinepoints = (
        dfplotdata
        .group_by("xvalue_binned", maintain_order=True)
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

    # collect once and index the DataFrame, rather than re-running this aggregation for every column read below
    dflinepointsdf = dflinepoints.collect()

    if startfromzero:
        # repeat the first point at x=0, keeping the column's own dtype so the frames stack
        firstrow = dflinepointsdf.head(1).with_columns(
            xvalue_binned=pl.lit(0.0, dtype=dflinepointsdf.schema["xvalue_binned"])
        )
        dflinepointsdf = pl.concat([firstrow, dflinepointsdf])

    xvalues_binned = dflinepointsdf.get_column("xvalue_binned")
    yvalues_binned = dflinepointsdf.get_column("yvalue_binned")

    (plotobj,) = ax.plot(xvalues_binned, yvalues_binned, label=label, **plotkwargs)
    color = plotobj.get_color()

    if args.markers:
        plotkwargs_markers: dict[str, t.Any] = plotkwargs | {
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
        dfplotdatadf = dfplotdata.select("xvalue", "yvalue").collect()
        if dfplotdatadf.height > 10000:
            plotkwargs_markers["rasterized"] = True
        # plot the markers first
        ax.plot(dfplotdatadf.get_column("xvalue"), dfplotdatadf.get_column("yvalue"), **plotkwargs_markers)

    else:
        yvalues_binned_min = dflinepointsdf.get_column("yvalue_binned_min")
        yvalues_binned_max = dflinepointsdf.get_column("yvalue_binned_max")
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
    """Plot the initial abundance or mass of each species in specieslist."""
    if seriestype == "initmasses":
        estimators = estimators.with_columns(
            (pl.col(massfraccol) * pl.col("mass_g") / Msun_to_g).alias(
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
    """Plot the mean ion charge of each element in params."""
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


def plot_average_excitation(
    ax: mplax.Axes,
    params: Sequence[str],
    estimators: pl.LazyFrame,
    modelpath: str | Path,
    startfromzero: bool,
    args: argparse.Namespace | None = None,
    **plotkwargs: t.Any,
) -> None:
    """Plot the population-weighted mean level excitation energy of each requested ion."""
    if args is None:
        args = argparse.Namespace()

    ax.set_ylabel("Average excitation energy [eV]")

    estimatorcolumns = estimators.collect_schema().names()
    # the superlevel population is spread over the levels it stands in for at the electron temperature
    dftexc = estimators.select("timestep", "modelgridindex", T_exc=pl.col("Te"))

    for paramvalue in params:
        print(f"  plotting averageexcitation {paramvalue}")
        iontuple = at.get_ion_tuple(paramvalue)
        if isinstance(iontuple, int):
            msg = f"averageexcitation needs an ion such as 'Fe II', but got {paramvalue!r}"
            raise TypeError(msg)
        atomic_number, ion_stage = iontuple

        dfavgexc = at.estimators.get_averageexcitation(modelpath, atomic_number, ion_stage, dftexc)

        # weight the average by the ion population where it is available, as plot_average_ionisation
        # weights by the element population
        nnioncol = f"nnion_{at.get_elsymbol(atomic_number)}_{at.roman_numerals[ion_stage]}"
        weightcol = pl.col(nnioncol) if nnioncol in estimatorcolumns else pl.lit(1.0)

        dfplotdata = (
            estimators
            .join(dfavgexc, on=["timestep", "modelgridindex"], how="inner")
            .with_columns(celltsweight=weightcol * pl.col("deltavol_deltat"), yvalue=pl.col("averageexcitation"))
            .filter(pl.col("yvalue").is_not_nan() & pl.col("yvalue").is_not_null())
        )

        plot_data(
            dfplotdata=dfplotdata,
            ax=ax,
            args=args,
            startfromzero=startfromzero,
            label=paramvalue,
            color=get_elemcolor(atomic_number=atomic_number),
            **plotkwargs,
        )


def plot_levelpop(
    ax: mplax.Axes,
    xlist: Sequence[int | float],
    seriestype: str,
    params: Sequence[str],
    timestepslist: Sequence[int],
    mgilist: Sequence[int | Sequence[int]],
    modelpath: str | Path,
    startfromzero: bool,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> None:
    """Plot the population of each level in params, either directly or per unit velocity."""
    if seriestype == "levelpopulation_dn_on_dvel":
        ax.set_ylabel("dN/dV [{}km$^{{-1}}$ s]")
    elif seriestype == "levelpopulation":
        ax.set_ylabel("X$_{{i}}$ [{}/cm³]")
    else:
        raise ValueError

    at.plottools.set_exponent_label(ax)

    modeldata = at.inputmodel.get_modeldata(
        modelpath, derived_cols=["mass_g", "volume", "vel_r_min_kmps", "vel_r_max_kmps"]
    )[0].collect()

    adata = at.atomic.get_levels(modelpath)

    arr_tdelta = at.get_timestep_times(modelpath, loc="delta")

    # read_files is uncached, so read every rank's nlte output once rather than once per param
    dfnltepops_allions = at.nltepops.read_files(modelpath)

    for paramvalue in params:
        paramsplit = paramvalue.split(" ")
        atomic_number = at.get_atomic_number(paramsplit[0])
        ion_stage = at.decode_roman_numeral(paramsplit[1])
        levelindex = int(paramsplit[2])

        ionlevels = adata.filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage)).row(
            0, named=True
        )["levels"]
        levelname = ionlevels["levelname"].item(levelindex)
        label = (
            f"{at.get_ionstring(atomic_number, ion_stage, style='chargelatex')} level {levelindex}:"
            f" {at.nltepops.texifyconfiguration(levelname)}"
        )

        print(f"plot_levelpop {label}")

        dfnltepops = dfnltepops_allions.filter(
            (pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage) & (pl.col("level") == levelindex)
        )

        # one pass over the populations instead of re-filtering the frame for every cell and timestep below.
        # setdefault keeps the first row for a duplicated key, matching the .item(0) this replaces
        levelpop_of_mgi_ts: dict[tuple[int, int], float] = {}
        for mgi, ts, n_nlte in dfnltepops.select("modelgridindex", "timestep", "n_NLTE").iter_rows():
            levelpop_of_mgi_ts.setdefault((mgi, ts), n_nlte)

        ylist = []
        for modelgridindex in mgilist:
            assert isinstance(modelgridindex, int)
            valuesum = 0.0
            tdeltasum = 0.0
            # print(f'modelgridindex {modelgridindex} timesteps {timesteps}')

            for timestep in timestepslist:
                levelpop = levelpop_of_mgi_ts[modelgridindex, timestep]

                valuesum += levelpop * arr_tdelta[timestep]
                tdeltasum += arr_tdelta[timestep]

            if seriestype == "levelpopulation_dn_on_dvel":
                assert isinstance(modelgridindex, int)
                cell = modeldata.row(modelgridindex, named=True)
                deltav = cell["vel_r_max_kmps"] - cell["vel_r_min_kmps"]
                ylist.append(valuesum / tdeltasum * cell["volume"] / deltav)
            else:
                ylist.append(valuesum / tdeltasum)

        plot_data(
            pl.DataFrame({"xvalue": xlist, "yvalue": ylist}, orient="col").with_columns(
                xvalue_binned=pl.col("xvalue"), celltsweight=pl.lit(1.0)
            ),
            ax=ax,
            args=args,
            startfromzero=startfromzero,
            label=label,
            **plotkwargs,
        )


def get_iontuple(ionstr: str) -> tuple[int, str | int]:
    """Decode into atomic number and parameter, e.g., [(26, 1), (26, 2), (26, 'ALL'), (26, 'Fe56')]."""
    # interpret bare integers as atomic numbers
    if ionstr.isdigit():
        atomic_number = int(ionstr)
        return (atomic_number, "ALL")

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


def could_be_ion(plotvar: t.Any) -> bool:
    """Return True if plotvar could be part of an ion population plot, i.e. an ion/element/atomic number or a plot directive."""
    # lists are plot directives and bare integers are atomic numbers
    if isinstance(plotvar, (list, int)):
        return True

    if not isinstance(plotvar, str):
        return False

    # a string that is an integer is an atomic number
    return plotvar.isdigit() or "=" in plotvar or get_iontuple(plotvar)[0] >= 1


def default_plotitem_has_data(
    plotitems: t.Any, estimatorcolumns: Collection[str], modelpath: str | Path | None = None
) -> bool:
    """Return False if a plot item names an element that is missing from this model's estimators.

    The built-in plot list names particular elements (e.g. Sr), which most models do not contain. This is only
    applied to that default list: an explicitly requested plot item is never dropped, so a typo there still raises.
    """
    if isinstance(plotitems, str):
        # an estimator variable always wins over the element reading of its name, because several estimator names
        # are also element symbols (Te is tellurium, W is tungsten)
        if plotitems in estimatorcolumns:
            return True

        atomic_number = get_iontuple(plotitems)[0]
        if 1 <= atomic_number < len(at.get_elsymbolslist()):
            return f"nnelement_{at.get_elsymbol(atomic_number)}" in estimatorcolumns
        return True

    if isinstance(plotitems, (list, tuple)):
        # initabundances/initmasses series read the input model file, not the estimators, so the element names in
        # those items say nothing about which estimator columns exist
        if len(plotitems) == 2 and isinstance(plotitems[0], str) and plotitems[0] in {"initabundances", "initmasses"}:
            return True

        # averageexcitation reads the NLTE population files, which a model need not have written
        if len(plotitems) == 2 and plotitems[0] == "averageexcitation" and modelpath is not None:
            if at.firstexisting_or_none("nlte_0000.out", folder=modelpath, tryzipped=True) is None:
                return False
            return all(default_plotitem_has_data(item, estimatorcolumns, modelpath) for item in plotitems[1])

        return all(default_plotitem_has_data(item, estimatorcolumns, modelpath) for item in plotitems)

    return True


def normalise_plotitems(plotitems: t.Any, estimatorcolumns: Collection[str]) -> list[t.Any]:
    """Resolve variable aliases and move any 'key=value' plot directives to the end of the plot item list.

    A list of ions such as ["Sr I", "Sr II"] is rewritten as a populations plot [["populations", ["Sr I", "Sr II"]]].
    """
    if isinstance(plotitems, str):
        plotitems = [plotitems]
    assert isinstance(plotitems, list)

    plot_directives = [
        plotvar.split("=", maxsplit=1) for plotvar in plotitems if isinstance(plotvar, str) and ("=" in plotvar)
    ]
    plotvars = [
        VARIABLE_ALIASES.get(plotvar, plotvar) if isinstance(plotvar, str) else plotvar
        for plotvar in plotitems
        if not isinstance(plotvar, str) or "=" not in plotvar
    ]

    if not plotvars:
        msg = "Empty plot item list; provide at least one plot variable after -plot (e.g. -plot Te)."
        raise ValueError(msg)

    if isinstance(plotvars[0], str) and plotvars[0] not in estimatorcolumns and all(map(could_be_ion, plotvars)):
        # plotting this as a variable would cause an error, so interpret it as ion populations instead
        new_plotvars = [["populations", plotvars]]
        print(f"Rewriting plotlist {plotvars} to {new_plotvars}")
        plotvars = new_plotvars

    return plotvars + plot_directives


def get_column_name(seriestype: str, atomic_number: int, ion_stage: str | int) -> tuple[str, str]:
    """Return the estimator column name for one ion, element, or isotope, along with its plot label."""
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
            estimators.select(
                pl.col("deltavol_deltat").alias("celltsweight"),
                (expr_yvals / expr_normfactor).fill_nan(0.0).alias("yvalue"),
                cs.starts_with("xvalue"),
            )
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
        ylabel = POPTYPE_YLABELS.get(args.poptype)
        if ylabel is None:
            msg = f"Unknown poptype: {args.poptype}"
            raise ValueError(msg)
        ax.set_ylabel(ylabel)
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
    xvariable: str, estimators: pl.LazyFrame, timestepslist: Collection[int] | None, args: argparse.Namespace
) -> tuple[list[float | int], list[int], list[int], pl.LazyFrame]:
    """Return the x values, model grid indices, and timesteps to plot, along with the filtered estimators."""
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
        scalefactor = km_to_cm if xvariable == "velocity" else C_cm_per_s
        estimators = estimators.with_columns(xvalue=(pl.col(velcolumn) / scalefactor))
    else:
        assert xvariable in estimators.collect_schema().names()
        estimators = estimators.with_columns(xvalue=pl.col(xvariable))

    # one collect for these streaming aggregations, rather than re-running the whole scan once per column. Only
    # the ones the command line did not already pin down are requested, so supplying -xmin -xmax -xbins scans
    # nothing at all here. xdeltamax stays out: it needs a full sort, and is only read for automatic binning.
    statexprs: dict[str, pl.Expr] = {}
    if args.xmin is None:
        statexprs["xmin"] = pl.col("xvalue").min()
    if args.xmax is None:
        statexprs["xmax"] = pl.col("xvalue").max()
    if args.xbins is None:
        statexprs["multiple_points_per_xvalue"] = pl.n_unique("xvalue") * pl.n_unique("timestep") < pl.len()

    xstats: dict[str, t.Any] = estimators.select(**statexprs).collect().row(0, named=True) if statexprs else {}

    xmin = xstats["xmin"] if args.xmin is None else args.xmin
    xmax = xstats["xmax"] if args.xmax is None else args.xmax

    if args.xbins is None and xstats["multiple_points_per_xvalue"]:
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

    if args.xbins is not None and args.xbins == 0:
        estimators = estimators.with_columns(xvalue_binned=pl.lit(None).cast(pl.Float64))
    elif args.xbins is not None:
        xbinedges = np.linspace(xmin, xmax, args.xbins)
        xlower = xbinedges[:-1]
        xupper = xbinedges[1:]
        xmids = (xlower + xupper) / 2
        estimators = (
            estimators
            .with_columns(
                (pl.col("xvalue").cut(breaks=list(xbinedges)).to_physical().cast(pl.Int32) - 1).alias("xbinindex")
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

    # again one collect rather than three separate scans of the same query
    uniques = (
        estimators
        .select(
            # sort all three: mgilist[0] and timestepslist[0] name the output file and the figure title,
            # and polars' unique() does not maintain order, so an unsorted list makes those vary between runs
            xvalue=pl.col("xvalue").unique().sort().implode(),
            modelgridindex=pl.col("modelgridindex").unique().sort().implode(),
            timestep=pl.col("timestep").unique().sort().implode(),
        )
        .collect()
        .row(0, named=True)
    )

    assert len(uniques["xvalue"]) > 0, "No data found for x-axis variable"

    return (uniques["xvalue"], uniques["modelgridindex"], uniques["timestep"], estimators)


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
    legend_ncols = 1
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

    remaining_plotitems: list[t.Any] = []
    ymin, ymax = None, None
    for plotitem in plotitems:
        if isinstance(plotitem, str | pl.Expr):
            remaining_plotitems.append(plotitem)
            continue
        seriestype, params = plotitem
        seriestype = seriestype.removeprefix("_").lower()
        if seriestype == "ymin":
            ymin = float(params) if isinstance(params, str) else params
            ax.set_ylim(bottom=ymin)

        elif seriestype == "ymax":
            ymax = float(params) if isinstance(params, str) else params
            ax.set_ylim(top=ymax)

        elif seriestype == "yscale":
            ax.set_yscale(params)
        else:
            remaining_plotitems.append(plotitem)

    for plotitem in remaining_plotitems:
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

            elif seriestype == "averageexcitation":
                plot_average_excitation(
                    ax, params, estimators, modelpath, startfromzero=startfromzero, args=args, **plotkwargs
                )

            else:
                seriestype, ionlist = plotitem
                ax.set_yscale("log")
                if seriestype == "populations" and len(ionlist) > 2 and ax.get_yscale() == "log":
                    legend_ncols = 2

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

    if showlegend and not args.nolegend:
        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1, ncols=legend_ncols, markerscale=3)


def make_figure(
    modelpath: Path | str,
    timestepslist: Collection[int] | None,
    estimators: pl.LazyFrame,
    xvariable: str,
    plotlist: list[list[t.Any]],
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> str:
    """Plot one subplot per entry in plotlist, save the figure, and return the output filename."""
    modelname = at.get_model_name(modelpath)

    fig, axes = plt.subplots(
        nrows=len(plotlist),
        ncols=1,
        sharex=True,
        figsize=(args.figscale * 5.0 * args.figwidthscale, args.figscale * 5.0 * 0.5 * len(plotlist)),
        layout="constrained",
        # tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )
    if len(plotlist) == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)

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

    # the x range comes from the data when the user gives no -xmin/-xmax. A degenerate range goes to
    # matplotlib as no limit at all, so that it keeps its own padding around the single value.
    xlimits = (xmin, xmax, "-xmin") if xmin != xmax else (None, None, "-xmin")
    set_axis_properties(axes, args, xlimits=xlimits)

    for ax, plotitems in zip(axes, plotlist, strict=False):
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
        args.outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)

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
        args.outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)

        assert isinstance(timestepslist, list)
        outfilename = str(args.outputfile).format(timestep=strtimestep, timedays=strtimedays, format=args.format)

    set_plot_title(axes[0], figure_title, args)

    save_figure(fig, outfilename, show=args.show, dpi=args.dpi)

    return outfilename


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(
        parser, default=".", helptext="Path to ARTIS folder (or virtual path e.g. codecomparison/ddc10/cmfgen)"
    )

    addarg_modelgridindex(parser, helptext="Model grid cell for the time evolution plot")

    addarg_timestep(parser, helptext="Timestep number for internal structure plot")

    addarg_timedays(parser, helptext="Time in days to plot for internal structure plot")

    addarg_timeminmax(parser)

    parser.add_argument("--multiplot", action="store_true", help="Make multiple plots for timesteps in range")

    parser.add_argument("-x", default=None, help="Horizontal axis variable, e.g. velocity, timestep, or time")

    addarg_axislimits(parser, include_y=False)

    parser.add_argument(
        "-xbins", type=int, default=None, help="Number of x bins between xmax and xmin (or -1 for automatic bin size)"
    )

    parser.add_argument("--hidexlabel", action="store_true", help="Hide the bottom horizontal axis label")

    parser.add_argument("--markers", action="store_true", help="Plot markers instead of shaded area")

    addarg_filter(parser)

    parser.add_argument("-format", "-f", default="pdf", choices=["pdf", "png"], help="Set format of output plot files")

    parser.add_argument("--makegif", action="store_true", help="Make a gif with time evolution (requires --multiplot)")

    addarg_notitle(parser)

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
        choices=list(POPTYPE_YLABELS),
        help="Plot absolute ion populations, or ion populations as a fraction of total or element population",
    )

    addarg_nolegend(parser)

    parser.add_argument(
        "-labelfontsize",
        type=float,
        default=10,
        help="Font size of the tick labels. The default is smaller than for the other plot commands, "
        "because this command stacks one subplot per requested quantity",
    )

    addarg_figscale(parser, include_figwidthscale=True)
    # deprecated spelling of -figwidthscale kept as a hidden alias
    parser.add_argument("-scalefigwidth", dest="figwidthscale", type=float, help=argparse.SUPPRESS)

    addarg_show(parser)

    addarg_dpi(parser, default=600)

    addarg_outputfile(parser, extraflags=("-outputpath",), default=Path(), helptext="Filename for PDF file")

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
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpath = Path(args.modelpath)

    modelname = at.get_model_name(modelpath)

    should_use_all_timesteps = (
        not args.timedays
        and not args.timemin
        and not args.timemax
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
            dfselectedcells = at.inputmodel.slice1dfromconein3dmodel.make_cone(args, logprint=print)
        else:
            msg = f"Invalid args.readonlymgi: {args.readonlymgi}"
            raise ValueError(msg)
        dfselectedcells = dfselectedcells.filter(pl.col("rho") > 0)
        args.modelgridindex = list(dfselectedcells["inputcellid"])

    timesteps_included = list(range(timestepmin, timestepmax + 1))
    estimators = at.estimators.scan_estimators(
        modelpath=modelpath,
        modelgridindex=args.modelgridindex,
        timestep=tuple(timesteps_included),
        classicartis=args.classicartis,
    )

    estimators, modelmeta = at.estimators.join_cell_modeldata(estimators=estimators, modelpath=modelpath, verbose=False)
    # pl.len() lets projection pushdown read 2 columns; head(1) would force every column to materialise
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

    usingdefaultplotlist = not args.plotlist
    plotlist: list[t.Any] = args.plotlist or [
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
        [["averageexcitation", ["Fe II", "Fe III"]]],
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

    if usingdefaultplotlist:
        keptplotlist: list[t.Any] = []
        skippedplotlist: list[t.Any] = []
        for plotitems in plotlist:
            target = (
                keptplotlist if default_plotitem_has_data(plotitems, estimatorcolumns, modelpath) else skippedplotlist
            )
            target.append(plotitems)

        if skippedplotlist:
            print(f"Skipping default plots for elements that are not in this model: {skippedplotlist}")
        if not keptplotlist:
            msg = "No default plots apply to this model. Choose what to plot with -plot (e.g. -plot Te TR)"
            raise ValueError(msg)
        plotlist = keptplotlist

    plotlist = [normalise_plotitems(plotitems, estimatorcolumns) for plotitems in plotlist]

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

        if args.x == "velocity" and modelmeta["vmax_cmps"] > 0.3 * C_cm_per_s:
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

        outputfiles: list[str] = []
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
                gifname = outdir / f"plotestim_evolution_ts{timestepmin:03d}_ts{timestepmax:03d}.gif"
                at.write_gif(gifname, outputfiles, duration=1000)
            elif args.format == "pdf":
                at.merge_pdf_files(outputfiles)


if __name__ == "__main__":
    main()
