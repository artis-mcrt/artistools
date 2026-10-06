"""Functions for plotting artis estimators and internal structure.

Examples are temperatures, populations, heating/cooling rates.
"""

import argparse
import contextlib
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
import numpy as np
import polars as pl
from polars import selectors as cs

from artistools.atomic import decode_roman_numeral
from artistools.atomic import get_atomic_number
from artistools.atomic import get_composition_data
from artistools.atomic import get_elsymbol
from artistools.atomic import get_elsymbolset
from artistools.atomic import get_elsymbolslist
from artistools.atomic import get_ion_levels
from artistools.atomic import get_ion_tuple
from artistools.atomic import get_ionstring
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.constants import Msun_to_g
from artistools.estimators.core import get_averageexcitation
from artistools.estimators.core import get_levelpop_modeldata
from artistools.estimators.core import get_units
from artistools.estimators.core import get_units_string
from artistools.estimators.core import get_variablelongunits
from artistools.estimators.core import get_varname_formatted
from artistools.estimators.core import join_cell_modeldata
from artistools.estimators.core import scan_estimators
from artistools.estimators.core import summarise_columns
from artistools.estimators.core import VARIABLES
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.inputmodel.slice1dfromconein3dmodel import get_profile_along_axis
from artistools.inputmodel.slice1dfromconein3dmodel import make_cone
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_dpi
from artistools.misc import addarg_figscale
from artistools.misc import addarg_filter
from artistools.misc import addarg_labelfontsize
from artistools.misc import addarg_legend
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_positional_items
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timeminmax
from artistools.misc import addarg_timestep
from artistools.misc import addarg_verbose
from artistools.misc import artis_subfolders
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import flatten_list
from artistools.misc import folder_is_artis_run
from artistools.misc import format_frame_path
from artistools.misc import get_filterfunc
from artistools.misc import get_model_name
from artistools.misc import get_time_range
from artistools.misc import get_time_range_text
from artistools.misc import get_timestep_time
from artistools.misc import get_timestep_times
from artistools.misc import get_timesteps
from artistools.misc import item_names_a_folder
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import path_is_codecomparison
from artistools.misc import print_modelpath
from artistools.misc import print_product
from artistools.misc import print_warning
from artistools.misc import resolve_frameset_paths
from artistools.misc import resolve_outputfile
from artistools.misc import resolve_positional_modelpath
from artistools.misc import suggest_names
from artistools.misc.general import get_bin_index_expr
from artistools.misc.remote import on_model_host
from artistools.nltepops import read_nltepops
from artistools.nltepops import texifyconfiguration
from artistools.plottools import get_drawn_values
from artistools.plottools import get_next_color
from artistools.plottools import log_axis_limit
from artistools.plottools import make_frame_figure
from artistools.plottools import prune_log_ticks
from artistools.plottools import save_figure
from artistools.plottools import set_axis_properties
from artistools.plottools import set_exponent_label
from artistools.plottools import set_legend
from artistools.plottools import set_mpl_style
from artistools.plottools import set_plot_title
from artistools.plottools import wants_log_scale

if t.TYPE_CHECKING:
    import matplotlib.figure as mplfig
    import matplotlib.typing as mplt
    import numpy.typing as npt

    from artistools.estimators.core import EstimatorBatchCache
    from artistools.misc import FrameSet


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
    "cylradialdensity": r"Cylindrical radial density dN/drcyl/dz $\left[\rm{cm}^{-2}\right]$",
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
        atomic_number = get_atomic_number(elsymbol)
        if atomic_number < 0:
            msg = f"{elsymbol!r} is not an element symbol, so it has no colour"
            raise ValueError(msg)
    else:
        elsymbol = get_elsymbol(atomic_number)

    if elsymbol in elementcolors:
        return elementcolors[elsymbol]

    palette = get_unreserved_elemcolors()

    return palette[atomic_number % len(palette)]


@lru_cache(maxsize=1)
def get_unreserved_elemcolors() -> tuple["mplt.ColorType", ...]:
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
    return get_variablelongunits(variable) or get_units_string(variable)


def get_point_colour(color: "mplt.ColorType") -> tuple[float, float, float]:
    """Return a lighter shade of the colour of a series, for its points.

    The lightness stays below 0.85, because a lighter colour is almost white and a point of that colour does not
    show on the white background.
    """
    import colorsys

    hue, lightness, saturation = colorsys.rgb_to_hls(*mc.to_rgb(color))
    return colorsys.hls_to_rgb(hue, min(0.85, 1.5 * lightness), saturation)


def draw_points(
    ax: mplax.Axes,
    dfpoints: pl.DataFrame,
    seriescolor: "mplt.ColorType",
    plotkwargs: dict[str, t.Any],
    label: str | None,
) -> None:
    """Draw every point of a series in a lighter shade of the colour of the series.

    The ticks of the axes draw above the points. --markers and -xbins 0 both draw with this function, thus a plot
    with and without bins agrees.
    """
    pointkwargs: dict[str, t.Any] = plotkwargs | {
        "linestyle": "None",
        "marker": ".",
        "markersize": 5,
        "color": get_point_colour(seriescolor),
        "markeredgewidth": 0,
        "zorder": -1,
    }
    pointkwargs.pop("dashes", None)
    if dfpoints.height > 10000:
        pointkwargs["rasterized"] = True
    ax.plot(dfpoints.get_column("xvalue"), dfpoints.get_column("yvalue"), label=label, **pointkwargs)


def repeat_endpoint(dflinepoints: pl.DataFrame, xvalue: float, *, atstart: bool) -> pl.DataFrame:
    """Return the line with its first or its last point repeated at the given x value.

    The point keeps the dtype of the column, thus the two frames stack.
    """
    end = dflinepoints.head(1) if atstart else dflinepoints.tail(1)
    row = end.with_columns(xvalue_binned=pl.lit(xvalue, dtype=dflinepoints.schema["xvalue_binned"]))

    return pl.concat([row, dflinepoints] if atstart else [dflinepoints, row])


class SeriesPlan(t.NamedTuple):
    """One series of a subplot before the collection of its data.

    dfseries holds these columns: xvalue, xvalue_binned, yvalue, and celltsweight (the weight of the
    average, e.g. the cell volume times the timestep duration).
    """

    label: str | None
    dfseries: pl.LazyFrame
    plotkwargs: dict[str, t.Any]


class SeriesData(t.NamedTuple):
    """The collected data of one series of a subplot, which the plot draws.

    dflinepoints comes from get_line_points, and it is None for -xbins 0. dfpoints holds the xvalue and the yvalue
    of every point for --markers, and it is None without them.
    """

    label: str | None
    plotkwargs: dict[str, t.Any]
    dflinepoints: pl.DataFrame | None
    dfpoints: pl.DataFrame | None


class SubplotSettings(t.TypedDict, total=False):
    """The settings of the axes of a subplot, which the functions of its series give before draw_subplot draws it.

    The host of a remote model runs these functions, thus the settings are values and not calls on the axes.
    """

    ylabel: str
    yscale: str
    ylim: tuple[float, float]
    # move the power-of-ten offset of the y axis into the label, see set_exponent_label
    exponentlabel: bool
    # clip the bottom of a log axis to ten decades below the top, after draw_subplot draws the series
    cliplogbottom: bool
    # the directives ymin= and ymax=, which draw_subplot applies after it draws the series
    ymin: float
    ymax: float
    showlegend: bool
    legendncols: int


def get_line_points(dfseries: pl.LazyFrame, args: argparse.Namespace) -> pl.LazyFrame:
    """Return the average line of a series in each x bin, with the minimum and the maximum of the bin."""
    dflinepoints = (
        dfseries
        # a cell that reported no value must not pull the average to zero, and a NaN is not a null.
        # A mask of the cell keeps the bin, which .drop_nans() on the aggregate took away
        .filter(pl.col("yvalue").is_not_null() & pl.col("yvalue").cast(pl.Float64).is_not_nan())
        .group_by("xvalue_binned")
        .agg(
            # every weight of a bin can be zero, e.g. an element that is absent from the bin has no mass.
            # Equal weights make the weighted average the plain average, thus the bin keeps its point
            yvalue_binned=pl
            .when(pl.col("celltsweight").sum() != 0.0)
            .then((pl.col("yvalue") * pl.col("celltsweight")).sum() / pl.col("celltsweight").sum())
            .otherwise(pl.col("yvalue").mean()),
            yvalue_binned_min=pl.col("yvalue").min(),
            yvalue_binned_max=pl.col("yvalue").max(),
        )
        .sort("xvalue_binned")
    )

    filterfunc = get_filterfunc(args)
    if filterfunc is not None:
        dflinepoints = dflinepoints.with_columns(
            pl.col("yvalue_binned").map_batches(filterfunc, return_dtype=pl.self_dtype())
        )

    return dflinepoints


def draw_series(
    dflinepoints: pl.DataFrame | None,
    dfpoints: pl.DataFrame | None,
    ax: mplax.Axes,
    label: str | None,
    args: argparse.Namespace,
    startfromzero: bool = False,
    xbinwidth: float | None = None,
    **plotkwargs: t.Any,
) -> None:
    """Draw the average line of a series, with markers at the points or a min-max area.

    dflinepoints comes from get_line_points. dfpoints holds the xvalue and the yvalue of every point,
    and --markers draws them. -xbins 0 draws those points alone, with no average line. xbinwidth is the width of
    each x bin, or None for a plot with no bins.
    """
    if dflinepoints is None:
        assert dfpoints is not None
        # -xbins 0 draws no line, thus the colour comes from the caller or from the cycle of the axes
        draw_points(ax, dfpoints, plotkwargs.get("color") or get_next_color(ax), plotkwargs, label=label)
        return

    # a binned line runs through bin middles, thus it stops half a bin short. The value holds across
    # the bin, thus reach the outer edges and leave no gap. The bins of a line can have gaps, thus the
    # space between the last two points does not give the width of a bin
    xbinned = dflinepoints.get_column("xvalue_binned")
    if xbinwidth and not xbinned.is_empty():
        halfwidth = xbinwidth / 2.0
        dflinepoints = repeat_endpoint(dflinepoints, xbinned[-1] + halfwidth, atstart=False)
        # startfromzero takes the line further to the left, thus that end comes below
        if not startfromzero:
            dflinepoints = repeat_endpoint(dflinepoints, xbinned[0] - halfwidth, atstart=True)

    if startfromzero:
        dflinepoints = repeat_endpoint(dflinepoints, 0.0, atstart=True)

    xvalues_binned = dflinepoints.get_column("xvalue_binned")
    yvalues_binned = dflinepoints.get_column("yvalue_binned")

    (plotobj,) = ax.plot(xvalues_binned, yvalues_binned, label=label, **plotkwargs)
    color = plotobj.get_color()

    if args.markers:
        assert dfpoints is not None
        draw_points(ax, dfpoints, color, plotkwargs, label=None)

    else:
        yvalues_binned_min = dflinepoints.get_column("yvalue_binned_min")
        yvalues_binned_max = dflinepoints.get_column("yvalue_binned_max")
        ax.fill_between(
            xvalues_binned, yvalues_binned_min, yvalues_binned_max, alpha=0.2, color=color, linewidth=0, zorder=-2
        )


def get_subplot_queries(plans: Sequence[SeriesPlan], args: argparse.Namespace) -> list[pl.LazyFrame]:
    """Return the queries of the average lines and then of the points of every series of the subplot."""
    # -xbins 0 draws the points alone, thus it needs no average line
    linequeries = [get_line_points(plan.dfseries, args) for plan in plans] if args.xbins != 0 else []
    pointqueries = [plan.dfseries.select("xvalue", "yvalue") for plan in plans] if args.markers else []
    return [*linequeries, *pointqueries]


def get_series_data(
    plans: Sequence[SeriesPlan], frames: Sequence[pl.DataFrame], args: argparse.Namespace
) -> list[SeriesData]:
    """Return the data of each series of the subplot, from the frames of the queries of get_subplot_queries."""
    linecount = len(plans) if args.xbins != 0 else 0
    dflinepoints_of_plan: Sequence[pl.DataFrame | None] = frames[:linecount] or [None] * len(plans)
    dfpoints_of_plan: Sequence[pl.DataFrame | None] = frames[linecount:] or [None] * len(plans)
    return [
        SeriesData(plan.label, plan.plotkwargs, dflinepoints, dfpoints)
        for plan, dflinepoints, dfpoints in zip(plans, dflinepoints_of_plan, dfpoints_of_plan, strict=True)
    ]


def draw_subplot(
    ax: mplax.Axes,
    series: Sequence[SeriesData],
    settings: SubplotSettings,
    args: argparse.Namespace,
    startfromzero: bool,
    xbinwidth: float | None = None,
) -> None:
    """Apply the settings of the subplot to the axes, draw its series in order, and add the limits and the legend."""
    # set the scale and the limits before the data, so that the axis autoscales in the correct space
    if "yscale" in settings:
        ax.set_yscale(settings["yscale"])
    if "ylim" in settings:
        ax.set_ylim(*settings["ylim"])
    if "ylabel" in settings:
        ax.set_ylabel(settings["ylabel"])
    if settings.get("exponentlabel"):
        set_exponent_label(ax)

    for seriesdata in series:
        draw_series(
            seriesdata.dflinepoints,
            seriesdata.dfpoints,
            ax=ax,
            label=seriesdata.label,
            args=args,
            startfromzero=startfromzero,
            xbinwidth=xbinwidth,
            **seriesdata.plotkwargs,
        )

    if settings.get("cliplogbottom") and ax.get_yscale() == "log":
        # a log axis of populations can go down many decades. set_legend adds space for the legend later
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=max(ymin, ymax / 1e10))

    finish_subplot(ax, args, settings)


def plot_init_abundances(
    settings: SubplotSettings, specieslist: list[str], estimators: pl.LazyFrame, seriestype: str, **plotkwargs: t.Any
) -> list[SeriesPlan]:
    """Return the series of the initial abundance or mass of each species in specieslist."""
    if seriestype == "initmasses":
        estimators = estimators.with_columns(
            (pl.col(massfraccol) * pl.col("mass_g") / Msun_to_g).alias(
                f"init_mass_{massfraccol.removeprefix('init_X_')}"
            )
            for massfraccol in estimators.collect_schema().names()
            if massfraccol.startswith("init_X_")
        )
        settings["ylabel"] = r"Initial mass per x point [M$_\odot$]"
        valuetype = "init_mass_"
    else:
        assert seriestype == "initabundances"
        settings["ylim"] = (1e-20, 1.0)
        settings["ylabel"] = "Initial mass fraction"
        valuetype = "init_X_"

    columnnames = set(estimators.collect_schema().names())
    plans = []
    for speciesstr in specieslist:
        splitvariablename = speciesstr.split("_")
        elsymbol = splitvariablename[0].strip(string.digits)
        atomic_number = get_atomic_number(elsymbol)

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
            # an isotope, e.g. Fe52, has a column of its own, and an element takes the column of its symbol
            speciescolumn = f"{valuetype}{speciesstr}"
            expr_yvalue = pl.col(speciescolumn if speciescolumn in columnnames else f"{valuetype}{elsymbol}")

        series = estimators.with_columns(celltsweight=pl.col("rho") * pl.col("deltavol_deltat"), yvalue=expr_yvalue)

        # each species takes its own copy of the arguments. A shared copy gave the dashed style of
        # Ni56 to every species after it. The caller still sets the style of every species
        speciesplotkwargs: dict[str, t.Any] = {"linewidth": 1.5, "linestyle": linestyle} | plotkwargs
        speciesplotkwargs["color"] = get_elemcolor(atomic_number=atomic_number)

        plans.append(SeriesPlan(label=linelabel, dfseries=series, plotkwargs=speciesplotkwargs))

    return plans


def get_average_charge_expr(element: str, colnames: Collection[str]) -> tuple[pl.Expr, int]:
    """Return the mean charge of the ions of an element in a cell, and the highest charge of its ions.

    The mean takes the number density of each ion as the weight, thus it is the charge for each nucleus.
    """
    elsymb = get_elsymbol(get_atomic_number(element))
    if f"nnelement_{elsymb}" not in colnames:
        msg = f"ERROR: No element data found for {element}"
        raise ValueError(msg)

    ioncols = [col for col in colnames if col.startswith(f"nnion_{elsymb}_")]
    if not ioncols:
        msg = f"ERROR: No ion data found for {element}"
        raise ValueError(msg)

    ioncharges = [decode_roman_numeral(col.removeprefix(f"nnion_{elsymb}_")) - 1 for col in ioncols]
    expr_charge_per_nuc = pl.sum_horizontal([
        ioncharge * pl.col(ioncol) for ioncol, ioncharge in zip(ioncols, ioncharges, strict=True)
    ]) / pl.col(f"nnelement_{elsymb}")
    return expr_charge_per_nuc, max(ioncharges)


def plot_average_ionisation(
    settings: SubplotSettings, params: Sequence[str], estimators: pl.LazyFrame, **plotkwargs: t.Any
) -> list[SeriesPlan]:
    """Return the series of the mean ion charge of each element in params."""
    settings["ylabel"] = "Average ion charge"

    # a lazy plan resolves its schema on each call, thus read the names one time for the whole loop
    colnames = estimators.collect_schema().names()

    plans = []
    maxioncharge = 0
    for paramvalue in params:
        print(f"  plotting averageionisation {paramvalue}")
        atomic_number = get_atomic_number(paramvalue)

        color = get_elemcolor(atomic_number=atomic_number)
        elsymb = get_elsymbol(atomic_number)
        expr_charge_per_nuc, elementmaxcharge = get_average_charge_expr(paramvalue, colnames)
        maxioncharge = max(maxioncharge, elementmaxcharge)

        dfplotdata = estimators.with_columns(
            celltsweight=pl.col(f"nnelement_{elsymb}") * pl.col("deltavol_deltat"), yvalue=expr_charge_per_nuc
        ).filter(pl.col(f"nnelement_{elsymb}") > 0.0)

        plans.append(SeriesPlan(label=paramvalue, dfseries=dfplotdata, plotkwargs={"color": color} | plotkwargs))

    # the limit must cover every element, thus set it after the loop over the elements
    settings["ylim"] = (0.0, maxioncharge + 0.1)

    return plans


def read_nltepops_of_estimators(modelpath: str | Path, timesteps: Sequence[int], cells: Sequence[int]) -> pl.DataFrame:
    """Return the NLTE populations of the timesteps and the cells of the plot.

    A read of every rank, timestep, and cell needs about 5e9 rows for a 3D run of 1e5 cells and 100 timesteps,
    although a snapshot plot uses one timestep. read_nltepops reads one timestep or every timestep.
    """
    dfnltepops = read_nltepops(modelpath, timestep=timesteps[0] if len(timesteps) == 1 else None, modelgridindex=cells)
    return dfnltepops.filter(pl.col("timestep").is_in(timesteps))


def plot_average_excitation(
    settings: SubplotSettings,
    params: Sequence[str],
    timestepslist: Sequence[int],
    mgilist: Sequence[int],
    estimators: pl.LazyFrame,
    modelpath: str | Path,
    **plotkwargs: t.Any,
) -> list[SeriesPlan]:
    """Return the series of the population-weighted mean level excitation energy of each requested ion."""
    settings["ylabel"] = "Average excitation energy [eV]"

    estimatorcolumns = estimators.collect_schema().names()
    # the superlevel population is spread over the levels it stands in for at the electron temperature
    dftexc = estimators.select("timestep", "modelgridindex", T_exc=pl.col("Te"))

    # one read serves every series of the subplot
    dfnltepops_allions = read_nltepops_of_estimators(modelpath, timestepslist, mgilist)

    plans = []
    for paramvalue in params:
        print(f"  plotting averageexcitation {paramvalue}")
        iontuple = get_ion_tuple(paramvalue)
        if isinstance(iontuple, int):
            exit_with_error(
                f"averageexcitation needs an ion, and '{paramvalue}' names an element",
                'Give an ion, e.g. -plot averageexcitation "Fe II"',
            )
        atomic_number, ion_stage = iontuple

        dfavgexc = get_averageexcitation(
            modelpath, atomic_number, ion_stage, dftexc, dfnltepops=dfnltepops_allions.lazy()
        )

        # weight the average by the ion population where it is available, as plot_average_ionisation
        # weights by the element population
        nnioncol = f"nnion_{get_ionstring(atomic_number, ion_stage, sep='_', style='spectral')}"
        weightcol = pl.col(nnioncol) if nnioncol in estimatorcolumns else pl.lit(1.0)

        dfplotdata = (
            estimators
            .join(dfavgexc, on=["timestep", "modelgridindex"], how="inner", maintain_order="left")
            .with_columns(celltsweight=weightcol * pl.col("deltavol_deltat"), yvalue=pl.col("averageexcitation"))
            .filter(pl.col("yvalue").is_not_nan() & pl.col("yvalue").is_not_null())
        )

        plans.append(
            SeriesPlan(
                label=paramvalue,
                dfseries=dfplotdata,
                plotkwargs={"color": get_elemcolor(atomic_number=atomic_number)} | plotkwargs,
            )
        )

    return plans


def plot_levelpop(
    settings: SubplotSettings,
    seriestype: str,
    params: Sequence[str],
    timestepslist: Sequence[int],
    mgilist: Sequence[int],
    modelpath: str | Path,
    estimators: pl.LazyFrame,
    **plotkwargs: t.Any,
) -> list[SeriesPlan]:
    """Return the series of the population of each level in params, directly or per unit velocity."""
    if seriestype == "levelpopulation_dn_on_dvel":
        # the number for each unit of velocity, thus a lower-case v
        settings["ylabel"] = "dN/dv [{}km$^{{-1}}$ s]"
    elif seriestype == "levelpopulation":
        settings["ylabel"] = "X$_{{i}}$ [{}/cm³]"
    else:
        exit_with_error(
            f"'{seriestype}' is not a type of series",
            "The types of a level population are levelpopulation and levelpopulation_dn_on_dvel",
        )

    settings["exponentlabel"] = True

    # only the levelpopulation_dn_on_dvel series reads the shell velocities, which only a 1D model gives
    modeldata, t_model_init_days = get_levelpop_modeldata(Path(modelpath))
    if seriestype == "levelpopulation_dn_on_dvel" and not {"vel_r_min_kmps", "vel_r_max_kmps"} <= set(
        modeldata.columns
    ):
        exit_with_error(
            "levelpopulation_dn_on_dvel divides by the velocity width of a shell, and only a 1D model has shells",
            "Give levelpopulation for the number density of the level",
        )

    arr_tdelta = get_timestep_times(modelpath, loc="delta")

    # model.txt gives the volume at t_model_init, and the homologous flow expands the cell by the cube
    # of the time. dN/dv needs the number in the cell, thus the density takes the expanded volume
    arr_volumefactor = (
        (np.array(get_timestep_times(modelpath, loc="mid")) / t_model_init_days) ** 3
        if seriestype == "levelpopulation_dn_on_dvel"
        else np.ones(len(arr_tdelta))
    )

    # this series draws one point for each cell, thus the horizontal axis must give one value for
    # each cell. A time axis gives one value for each timestep instead
    dfxofmgi = (
        estimators.select(pl.col("modelgridindex").cast(pl.Int64), "xvalue").unique().sort("modelgridindex").collect()
    )
    if dfxofmgi.height != dfxofmgi["modelgridindex"].n_unique():
        exit_with_error(
            "a level population plot draws one point for each cell, thus the horizontal axis must"
            " give one value for each cell",
            "Give -x velocity or -x modelgridindex. A time axis gives one value for each timestep.",
        )

    # the mean over the time range weights the population of each timestep with its duration
    dftimesteps = pl.DataFrame({
        "timestep": [int(timestep) for timestep in timestepslist],
        "volumefactor": [float(arr_volumefactor[timestep]) for timestep in timestepslist],
        "tdelta": [float(arr_tdelta[timestep]) for timestep in timestepslist],
    })
    # the row of a cell in the model data has the index of the cell. dN/dv takes the number in the cell for each unit
    # of the width in velocity of the shell
    dfcellfactors = (
        modeldata.with_row_index("modelgridindex").select(
            pl.col("modelgridindex").cast(pl.Int64),
            "volume",
            deltav=pl.col("vel_r_max_kmps") - pl.col("vel_r_min_kmps"),
        )
        if seriestype == "levelpopulation_dn_on_dvel"
        else pl.DataFrame({"modelgridindex": dfxofmgi["modelgridindex"], "volume": 1.0, "deltav": 1.0})
    )

    # one read serves every series of the subplot
    dfnltepops_allions = read_nltepops_of_estimators(modelpath, timestepslist, mgilist)

    plans = []
    for paramvalue in params:
        paramsplit = paramvalue.split(" ")
        atomic_number = get_atomic_number(paramsplit[0])
        ion_stage = decode_roman_numeral(paramsplit[1])
        levelindex = int(paramsplit[2])

        ionlevels = get_ion_levels(Path(modelpath), atomic_number, ion_stage)
        assert ionlevels is not None
        levelname = ionlevels["levelname"].item(levelindex)
        label = (
            f"{get_ionstring(atomic_number, ion_stage, style='chargelatex')} level {levelindex}:"
            f" {texifyconfiguration(levelname)}"
        )

        print(f"plot_levelpop {label}")

        # a cell that holds no matter has no NLTE row at a timestep, thus the mean of a cell takes the durations of the
        # timesteps that hold a row of the cell. A cell with no row at all gives no point
        dfseries = (
            dfnltepops_allions
            .lazy()
            .filter(
                (pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage) & (pl.col("level") == levelindex)
            )
            .select(pl.col("modelgridindex").cast(pl.Int64), pl.col("timestep").cast(pl.Int64), "n_NLTE")
            # the first row of a repeated cell and timestep stays
            .unique(subset=["modelgridindex", "timestep"], keep="first", maintain_order=True)
            .join(dftimesteps.lazy(), on="timestep", how="inner")
            .group_by("modelgridindex")
            .agg(levelpop=(pl.col("n_NLTE") * pl.col("volumefactor") * pl.col("tdelta")).sum() / pl.col("tdelta").sum())
            .join(dfxofmgi.lazy(), on="modelgridindex", how="inner")
            .join(dfcellfactors.lazy(), on="modelgridindex", how="inner")
            .sort("modelgridindex")
            .select(
                "xvalue",
                yvalue=pl.col("levelpop") * pl.col("volume") / pl.col("deltav"),
                xvalue_binned=pl.col("xvalue"),
                celltsweight=pl.lit(1.0),
            )
        )
        plans.append(SeriesPlan(label=label, dfseries=dfseries, plotkwargs=plotkwargs.copy()))

    return plans


# The plot directives that a plot item can carry, e.g. "-p rho yscale=log". The underscore of a name
# is optional.
DIRECTIVES = ("ionpoptype", "ymin", "ymax", "yscale")

# a series type groups the names that follow it, e.g. -plot averageionisation Fe Ni. Each one has its
# own plot function in plot_subplot. An ion series needs no entry here, because the estimator columns
# name it, e.g. gamma_NT_Fe_II gives the series type gamma_NT
SERIESTYPES = (
    "averageexcitation",
    "averageionisation",
    "initabundances",
    "initmasses",
    "levelpopulation",
    "populations",
)


def is_seriestype(name: t.Any, estimatorcolumns: Collection[str]) -> bool:
    """Return True when a name has its own plot function, e.g. "populations" in "populations Fe II"."""
    if not isinstance(name, str) or name in estimatorcolumns:
        return False

    return name in SERIESTYPES or name.startswith("levelpopulation_")


def is_ionseriestype(name: t.Any, estimatorcolumns: Collection[str], params: Sequence[t.Any]) -> bool:
    """Return True when a name plus the ions after it give the columns of an ion series.

    An estimator that names an ion, e.g. gamma_NT_Fe_II, gives the series type gamma_NT. Every name
    after it must be an ion, because a name such as "heating" is also the prefix of heating_coll, and
    "heating coll" must keep the message that names the column.

    A type of series can also be a variable of its own. cooling_coll holds the cooling rate of the
    whole cell, and cooling_coll_Fe_II holds the part of one ion. A column of one of the named ions
    then proves the reading. A subplot of the total alone takes its own -plot, e.g.
    -plot cooling_coll -plot cooling_coll "Fe II".
    """
    if not isinstance(name, str) or not params:
        return False

    if not all(isinstance(param, str) and is_valid_ion(param) for param in params):
        return False

    # A column of one of the ions proves the reading, also for a name that is a variable of its own.
    # The test takes one column and not every column. An element can lose its top ion here, e.g. a
    # model that holds Fe I to Fe V has no gamma_NT_Fe_V.
    if any(get_column_name(name, *get_iontuple(param))[0] in estimatorcolumns for param in params):
        return True

    # A name that gives no variable of its own keeps the older and looser test, which asks only that
    # the model holds the family. Such a name has no other reading, thus it needs no column here.
    return name not in estimatorcolumns and any(col.startswith(f"{name}_") for col in estimatorcolumns)


# The subplots share one horizontal axis, thus no directive can set it for one subplot alone. The
# arguments -xmin and -xmax set it for the figure, and they also drop the data outside that range.
FIGURE_ARGUMENTS = ("xmin", "xmax")


def get_directive_name(seriestype: str) -> str | None:
    """Return the name of the directive that a plot item gives, e.g. "ymin", or None for a series.

    A figure argument and an unknown directive stop the command.
    """
    # normalise_plotitems adds the underscore, thus report the name as the user wrote it
    given = seriestype.removeprefix("_")
    # a figure argument stops the command whichever way the caller spells it, because a plot item
    # of that name reaches the ion branch and gives an error that names no argument
    if given.lower() in FIGURE_ARGUMENTS:
        exit_with_error(
            f"'{given}' belongs to the whole figure and not to one subplot, because the subplots "
            f"share one horizontal axis. Give -{given.lower()} instead, which also drops the data outside"
        )

    if seriestype.startswith("_") and given.lower() not in DIRECTIVES:
        suggestion = suggest_names(given, DIRECTIVES)
        exit_with_error(
            f"'{given}' is not a plot directive",
            f"{suggestion + ' ' if suggestion else ''}The directives are "
            f"{', '.join(f'{name}=' for name in DIRECTIVES)}",
        )

    return given.lower() if given.lower() in DIRECTIVES else None


def get_iontuple(ionstr: str) -> tuple[int, str | int]:
    """Decode into atomic number and parameter, e.g., [(26, 1), (26, 2), (26, 'ALL'), (26, 'Fe56')].

    An element gives "ALL". An isotope or a family of nuclides keeps its name, e.g. Fe56, Ni_56, or
    Ni_stable, because the column name of the estimators holds that name. Every other name goes to
    get_ion_tuple.
    """
    elsymbols = get_elsymbolset()
    if ionstr not in elsymbols and not ionstr.isdigit():
        stem = ionstr.rstrip("-0123456789")
        if stem in elsymbols:
            return (get_atomic_number(stem), ionstr)

        elsymbol, _, suffix = ionstr.partition("_")
        if suffix and elsymbol in elsymbols and decode_roman_numeral(suffix) < 0:
            return (get_atomic_number(elsymbol), ionstr)

    # get_ion_tuple strips an X_, nnelement_, or nnion_ prefix. Such a prefix names a column and not
    # an ion, thus a name that holds one goes to the fallback below and the caller then rejects it
    if not ionstr.startswith(("X_", "nnelement_", "nnion_")):
        with contextlib.suppress(ValueError):
            return get_element_or_ion_tuple(ionstr)

    # a name that is no ion at all, e.g. a mistyped variable. The caller tests the atomic number
    return (get_atomic_number(ionstr.split("_", maxsplit=1)[0]), ionstr)


def get_element_or_ion_tuple(ionstr: str) -> tuple[int, str | int]:
    """Return the atomic number and the ion stage of a name that get_ion_tuple reads, or "ALL" for an element."""
    iontuple = get_ion_tuple(ionstr)
    return (iontuple, "ALL") if isinstance(iontuple, int) else iontuple


def is_valid_ion(ionstr: str) -> bool:
    """Return True when the string names an element, an ion, or an isotope that get_iontuple can read.

    could_be_ion is deliberately permissive, thus it accepts a name such as "te" that get_atomic_number
    reads as tellurium. This test rejects that name, so that the caller can suggest the variable "Te".
    """
    atomic_number, param = get_iontuple(ionstr)
    if atomic_number < 1:
        return False

    if isinstance(param, int):
        return param >= 1

    elsymbols = get_elsymbolset()

    # get_column_name reads a suffix that joins the symbol, e.g. Fe_otherstable gives
    # nniso_Fe_otherstable, thus the first part of the name decides
    return param == "ALL" or param.rstrip("-0123456789") in elsymbols or param.split("_", maxsplit=1)[0] in elsymbols


def could_be_ion(plotvar: t.Any) -> bool:
    """Return True when plotvar can name part of an ion population plot: an ion, an element, or an atomic number."""
    # lists are plot directives and bare integers are atomic numbers
    if isinstance(plotvar, (list, int)):
        return True

    if not isinstance(plotvar, str):
        return False

    # get_iontuple reads a digit string as an atomic number, and normalise_plotitems has already taken
    # every "key=value" directive out of the list that reaches here
    return get_iontuple(plotvar)[0] >= 1


def get_default_plotitem_skip_reason(
    plotitems: t.Any, estimatorcolumns: Collection[str], modelpath: str | Path | None = None
) -> str | None:
    """Return the reason that the default plot list leaves out a plot item for this model, or None to keep it.

    The built-in plot list names particular elements (e.g. Sr), which most models do not contain. This is only
    applied to that default list: an explicitly requested plot item is never dropped, so a typo there still raises.
    """
    if isinstance(plotitems, str):
        # an estimator variable always wins over the element reading of its name, because several estimator names
        # are also element symbols (Te is tellurium, W is tungsten)
        if plotitems in estimatorcolumns:
            return None

        atomic_number = get_iontuple(plotitems)[0]
        if 1 <= atomic_number < len(get_elsymbolslist()):
            elsymbol = get_elsymbol(atomic_number)
            return None if f"nnelement_{elsymbol}" in estimatorcolumns else f"the estimators have no {elsymbol}"
        return None

    if isinstance(plotitems, (list, tuple)):
        # initabundances/initmasses series read the input model file, not the estimators, so the element names in
        # those items say nothing about which estimator columns exist
        if len(plotitems) == 2 and isinstance(plotitems[0], str) and plotitems[0] in {"initabundances", "initmasses"}:
            return None

        # averageexcitation reads the NLTE population files, which a model need not have written
        if len(plotitems) == 2 and plotitems[0] == "averageexcitation" and modelpath is not None:
            if firstexisting_or_none("nlte_0000.out", folder=modelpath, tryzipped=True) is None:
                return "the run has no NLTE population files (nlte_*.out)"
            plotitems = plotitems[1]

        return next(
            (
                reason
                for item in plotitems
                if (reason := get_default_plotitem_skip_reason(item, estimatorcolumns, modelpath)) is not None
            ),
            None,
        )

    return None


def get_model_default_plotlist(
    estimatorcolumns: Collection[str], modelpath: str | Path | None
) -> tuple[list[t.Any], list[tuple[t.Any, str]]]:
    """Return the default plot items that apply to this model, and each other default item with the reason."""
    plotlist: list[t.Any] = []
    skippedplotlist: list[tuple[t.Any, str]] = []
    for plotitems in get_default_plotlist():
        reason = get_default_plotitem_skip_reason(plotitems, estimatorcolumns, modelpath)
        if reason is None:
            plotlist.append(plotitems)
        else:
            skippedplotlist.append((plotitems, reason))
    return plotlist, skippedplotlist


def normalise_plotitems(plotitems: t.Any, estimatorcolumns: Collection[str]) -> list[t.Any]:
    """Resolve variable aliases and move any 'key=value' plot directives to the end of the plot item list.

    A list of ions such as ["Sr I", "Sr II"] is rewritten as a populations plot [["populations", ["Sr I", "Sr II"]]].
    """
    if isinstance(plotitems, str):
        plotitems = [plotitems]
    assert isinstance(plotitems, list)

    # the underscore of a directive is optional, thus "-p rho yscale=log" and "_yscale=log" are the same.
    # One shape here lets plot_subplot find an unknown directive whichever spelling the user gave
    plot_directives = [
        ["_" + key.removeprefix("_"), value]
        for key, value in (
            plotvar.split("=", maxsplit=1) for plotvar in plotitems if isinstance(plotvar, str) and "=" in plotvar
        )
    ]
    plotvars = [
        VARIABLE_ALIASES.get(plotvar, plotvar) if isinstance(plotvar, str) else plotvar
        for plotvar in plotitems
        if not isinstance(plotvar, str) or "=" not in plotvar
    ]

    # a subplot with no series draws an empty frame, e.g. in the viewer after the user removed its last series
    if not plotvars:
        return plot_directives

    # the grouped form names the type of series first, e.g. -plot populations "Fe II" "Fe III"
    if is_seriestype(plotvars[0], estimatorcolumns):
        if len(plotvars) == 1:
            exit_with_error(
                f"'{plotvars[0]}' names a type of series and takes at least one name after it",
                f'e.g. -plot {plotvars[0]} Fe. Quote an ion that holds a space, e.g. -plot {plotvars[0]} "Fe II"',
            )
        if all(isinstance(plotvar, str) for plotvar in plotvars[1:]):
            plotvars = [[plotvars[0], plotvars[1:]]]
    elif is_ionseriestype(plotvars[0], estimatorcolumns, plotvars[1:]):
        plotvars = [[plotvars[0], plotvars[1:]]]

    if isinstance(plotvars[0], str) and plotvars[0] not in estimatorcolumns and all(map(could_be_ion, plotvars)):
        # an ion population plot is the reading of last resort, thus reject a name that is no ion at all
        if notions := [var for var in plotvars if isinstance(var, str) and not is_valid_ion(var)]:
            exit_with_error(
                f"'{notions[0]}' is neither an estimator variable nor an ion",
                suggest_names(notions[0], estimatorcolumns) or "Run with --listvariables to see the variables",
            )

        # Te and W name an estimator variable and also an element. A run with no estimator files has no such column,
        # and the plot then read tellurium in place of the temperature and drew an empty subplot
        if absentvariables := [
            var for var in plotvars if isinstance(var, str) and var in VARIABLES and var not in estimatorcolumns
        ]:
            exit_with_error(
                f"'{absentvariables[0]}' is an estimator variable, and the estimators of this model hold no such"
                " column",
                "Run with --listvariables to see the variables of this model",
            )

        # plotting this as a variable would cause an error, so interpret it as ion populations instead
        new_plotvars = [["populations", plotvars]]
        print(f"Rewriting plotlist {plotvars} to {new_plotvars}")
        plotvars = new_plotvars

    return plotvars + plot_directives


def get_column_name(seriestype: str, atomic_number: int, ion_stage: str | int) -> tuple[str, str]:
    """Return the estimator column name for one ion, element, or isotope, along with its plot label."""
    ionstr = get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
    if seriestype == "populations":
        if ion_stage == "ALL":
            elsymbol = get_elsymbol(atomic_number)
            return f"nnelement_{elsymbol}", ionstr
        if isinstance(ion_stage, str) and ion_stage.startswith(get_elsymbol(atomic_number)):
            # not really an ion_stage but an isotope name
            return f"nniso_{ion_stage}", ionstr
        return f"nnion_{ionstr}", ionstr
    return f"{seriestype}_{ionstr}", ionstr


def get_iontuple_sortkey(iontuple: tuple[int, str | int]) -> tuple[int, int, int, str]:
    """Return a sort key that puts an element first, then its ion stages, then its isotopes.

    A plain sort of the tuples compares an ion stage with the name of an isotope, which raises a
    TypeError, e.g. for the pair of names "Fe" and "Fe II".
    """
    atomic_number, ion_stage = iontuple
    if ion_stage == "ALL":
        return (atomic_number, 0, 0, "")

    if isinstance(ion_stage, int):
        return (atomic_number, 1, ion_stage, "")

    return (atomic_number, 2, 0, ion_stage)


def get_population_normfactor(seriestype: str, poptype: str, atomic_number: int) -> pl.Expr:
    """Return the divisor that turns the number density of an ion into the population type, e.g. elpop."""
    if seriestype == "populations" and poptype == "elpop":
        return pl.col(f"nnelement_{get_elsymbol(atomic_number)}")
    if seriestype == "populations" and poptype == "totalpop":
        return pl.col("nntot")
    return pl.lit(1)


def get_ion_series_weight(
    seriestype: str, poptype: str, atomic_number: int, ion_stage: str | int, estimatorcolumns: Collection[str]
) -> pl.Expr:
    """Return the weight of a cell in the mean of an ion series, which multiplies the volume and the duration.

    A fraction of the element or of all the nuclei gives the ratio of the totals of a bin when the weight is its
    divisor. A rate for each ion, e.g. gamma_NT, takes the number of the ion as its weight, as averageexcitation does.
    The reader gives 0 for a rate in a cell that holds none of the element, and that weight leaves such a cell out.
    A population and a rate for each unit of volume, e.g. cooling_coll_Fe_II, take the volume alone.
    """
    if seriestype == "populations":
        return get_population_normfactor(seriestype, poptype, atomic_number)

    if get_units(get_column_name(seriestype, atomic_number, ion_stage)[0], latex=False) == "erg/s/cm^3":
        return pl.lit(1.0)

    popcolumn = get_column_name("populations", atomic_number, ion_stage)[0]
    if popcolumn not in estimatorcolumns:
        popcolumn = f"nnelement_{get_elsymbol(atomic_number)}"
    return pl.col(popcolumn) if popcolumn in estimatorcolumns else pl.lit(1.0)


def plot_multi_ion_series(
    settings: SubplotSettings,
    seriestype: str,
    ionlist: Sequence[str],
    estimators: pl.LazyFrame,
    modelpath: str | Path,
    poptype: str,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> list[SeriesPlan]:
    """Return the series of an ion-specific property, e.g. populations.

    The poptype parameter sets the normalisation of a population series. Each subplot carries its own
    value, thus one figure can show an absolute density in one subplot and an ion fraction in another.
    """
    iontuplelist = [get_iontuple(ionstr) for ionstr in ionlist]
    iontuplelist.sort(key=get_iontuple_sortkey)
    print(f"Subplot with ions: {iontuplelist}")

    missingions: set[tuple[int, str | int]] = set()
    if not args.classicartis:
        try:
            compositiondata = get_composition_data(modelpath)
        except FileNotFoundError:
            # the pass over the estimator columns below drops an ion that this run holds no column for,
            # thus the plot still gives the other ions
            print_warning("Could not read an ARTIS compositiondata.txt file to check ion availability")
        else:
            for atomic_number, ion_stage in iontuplelist:
                if (
                    isinstance(ion_stage, int)
                    and compositiondata.filter(
                        (pl.col("Z") == atomic_number)
                        & (pl.col("lowermost_ion_stage") <= ion_stage)
                        & (pl.col("uppermost_ion_stage") >= ion_stage)
                    ).is_empty()
                ):
                    missingions.add((atomic_number, ion_stage))

    if missingions:
        print_warning(f"Can't plot {seriestype} for {missingions} because these ions are not in compositiondata.txt")

    iontuplelist = [iontuple for iontuple in iontuplelist if iontuple not in missingions]

    # An ion of the model can still have no column of this series. The top ion of an element has no
    # gamma_NT and no bound-free cooling. Drop such an ion here, so that the plot gives the others.
    estimatorcolumns = estimators.collect_schema().names()
    nocolumnions = {
        iontuple for iontuple in iontuplelist if get_column_name(seriestype, *iontuple)[0] not in estimatorcolumns
    }
    if nocolumnions:
        print_warning(f"Can't plot {seriestype} for {nocolumnions} because the estimators hold no such column")

    iontuplelist = [iontuple for iontuple in iontuplelist if iontuple not in nocolumnions]

    lazyframes = []
    for atomic_number, ion_stage in iontuplelist:
        colname, ionstr = get_column_name(seriestype, atomic_number, ion_stage)
        expr_yvals = pl.col(colname)
        print(f"  plotting {seriestype} {ionstr.replace('_', ' ')}")

        assert poptype in POPTYPE_YLABELS
        # a radial density and a cumulative count take the number density, and the code below converts it
        expr_normfactor = get_population_normfactor(seriestype, poptype, atomic_number)

        # convert the volumetric number density [cm^-3] with the radius of each cell. A radial density
        # is dN/dr [cm^-1], and a cylindrical radial density is dN/drcyl/dz [cm^-2]. Only a population is a
        # number density. A rate such as gamma_NT keeps its value, as the image plots already do
        ispopulation = seriestype == "populations"
        expr_tmid_s = pl.col("tmid_days") * day_to_s
        if ispopulation and poptype == "radialdensity":
            expr_yvals *= 4 * math.pi * (pl.col("vel_r_mid") * expr_tmid_s).pow(2)
        elif ispopulation and poptype == "cylradialdensity":
            expr_yvals *= 2 * math.pi * pl.col("vel_rcyl_mid") * expr_tmid_s

        if ispopulation and poptype == "cumulative":
            # multiply each cell's number density by its volume before the sum, so the result is a particle count
            # the sum is over the cells of one timestep, thus it must restart at each timestep
            expr_yvals = (expr_yvals * pl.col("volume")).cum_sum().over("timestep")

        weight = get_ion_series_weight(seriestype, poptype, atomic_number, ion_stage, estimatorcolumns)
        lazyframes.append(
            estimators.select(
                (pl.col("deltavol_deltat") * weight).alias("celltsweight"),
                # 0/0 gives NaN for a cell that holds none of the element. Make it null. The weighted
                # mean in get_line_points then drops the cell and does not count it as zero.
                (expr_yvals / expr_normfactor).fill_nan(None).alias("yvalue"),
                cs.starts_with("xvalue"),
            )
        )

    plans = []
    for seriesindex, ((atomic_number, ion_stage), dfseries) in enumerate(zip(iontuplelist, lazyframes, strict=True)):
        plotlabel = str(
            ion_stage
            if hasattr(ion_stage, "lower") and ion_stage != "ALL"
            else get_ionstring(atomic_number, ion_stage, style="chargelatex")
        )

        # an element takes the first style. An isotope has no ion stage, thus its place in the list
        # separates it from the other isotopes of its element
        if ion_stage == "ALL":
            variantindex = 0
        elif isinstance(ion_stage, int):
            variantindex = ion_stage - 1
        else:
            variantindex = seriesindex

        color = get_elemcolor(atomic_number=atomic_number)
        styleindex = variantindex
        if args.colorbyion:
            # the colour separates the ions, thus every series keeps the first style
            styleindex = 0
            if ion_stage != "ALL":
                color = f"C{variantindex % 10}"

        dashes_list = [(), (3, 1, 1, 1), (1.5, 1.5), (6, 3), (1, 3)]
        dashes = dashes_list[styleindex % len(dashes_list)]

        linewidth_list = [1.0, 1.0, 1.0, 0.7, 0.7]
        linewidth = linewidth_list[styleindex % len(linewidth_list)] * 1.5

        if plotkwargs.get("linestyle", "solid") != "None":
            plotkwargs["dashes"] = dashes

        plans.append(
            SeriesPlan(
                label=plotlabel, dfseries=dfseries, plotkwargs={"linewidth": linewidth, "color": color} | plotkwargs
            )
        )

    if seriestype == "populations":
        ylabel = POPTYPE_YLABELS.get(poptype)
        if ylabel is None:
            msg = f"Unknown poptype: {poptype}"
            raise ValueError(msg)
        settings["ylabel"] = ylabel
    else:
        settings["ylabel"] = get_varname_formatted(seriestype)

    if plans:
        settings["cliplogbottom"] = True

    return plans


def plot_series(
    settings: SubplotSettings,
    variable: str | pl.Expr,
    showlegend: bool,
    estimators: pl.LazyFrame,
    nounits: bool = False,
    **plotkwargs: t.Any,
) -> list[SeriesPlan]:
    """Return the series of an estimator variable such as Te or TR."""
    if isinstance(variable, pl.Expr):
        colexpr = variable
    else:
        columns = estimators.collect_schema().names()
        if variable not in columns:
            exit_with_error(
                f"'{variable}' is not an estimator variable",
                suggest_names(variable, columns) or "Run with --listvariables to see the variables of this model",
            )
        colexpr = pl.col(variable)

    variablename = colexpr.meta.output_name()

    serieslabel = get_varname_formatted(variablename)
    units_string = get_units_string(variablename)

    if showlegend:
        linelabel = serieslabel
        if not nounits:
            linelabel += units_string
    else:
        settings["ylabel"] = serieslabel + units_string
        linelabel = None

    series = estimators.with_columns(celltsweight=pl.col("deltavol_deltat"), yvalue=colexpr)

    if variablename in (dictcolors := {"Te": "red", "heating_gamma": "blue", "cooling_adiabatic": "blue"}):
        plotkwargs.setdefault("color", dictcolors[variablename])
    plotkwargs.setdefault("linewidth", 1.5)

    print(f"  plotting {variablename}")
    return [SeriesPlan(label=linelabel, dfseries=series, plotkwargs=plotkwargs)]


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
        # -x takes any variable of the model as well as the four names above, thus no choices list can
        # hold them. A name that no model gives must still say what the choices are
        columns = estimators.collect_schema().names()
        if xvariable not in columns:
            suggestion = suggest_names(xvariable, columns)
            exit_with_error(
                f"'{xvariable}' is not a variable of this model, thus the horizontal axis cannot show it",
                f"{suggestion + ' ' if suggestion else ''}The other choices are time, timestep, velocity, and "
                "beta. Run with --listvariables to see every variable of this model",
            )

        estimators = estimators.with_columns(xvalue=pl.col(xvariable))

    # a row with no finite x value has no place on the axis, e.g. tmid_days_prevtimestep at the first timestep. The
    # minimum of such a list gave a TypeError, and set_xlim refuses a NaN or an infinite limit
    hasxvalue = pl.col("xvalue").is_not_null() & pl.col("xvalue").cast(pl.Float64).is_finite()
    xvalues = pl.col("xvalue").filter(hasxvalue)
    # one collect gives the statistics and the unique values. The command line can give the statistics, and then the
    # query leaves them out
    statexprs: dict[str, pl.Expr] = {"rowcount": pl.len(), "noxvaluecount": (~hasxvalue).sum()}
    if args.xmin is None:
        statexprs["xmin"] = xvalues.min()
    if args.xmax is None:
        statexprs["xmax"] = xvalues.max()
    # a time axis gives one x value to each timestep, and get_line_points averages the cells at each x value. Automatic
    # bins would merge the timesteps of a model that has more cells than timesteps, thus only a spatial axis takes them
    if args.xbins is None and xvariable not in TIME_XVARIABLES:
        statexprs["multiple_points_per_xvalue"] = (
            xvalues.n_unique() * pl.col("timestep").filter(hasxvalue).n_unique() < hasxvalue.sum()
        )
    if args.xbins is None or args.xbins < 0:
        # the automatic bins need this. The full sort is small beside a round trip
        statexprs["xdeltamax"] = xvalues.sort().diff().max()

    inxrange = hasxvalue
    if args.xmin is not None:
        inxrange &= pl.col("xvalue") >= args.xmin
    if args.xmax is not None:
        inxrange &= pl.col("xvalue") <= args.xmax
    # sort all three: mgilist[0] and timestepslist[0] name the output file and the figure title,
    # and polars' unique() does not maintain order, so an unsorted list makes those vary between runs
    uniqueexprs = {
        column: pl.col(column).filter(inxrange).unique().sort().implode()
        for column in ("xvalue", "modelgridindex", "timestep")
    }

    stats = estimators.select(**statexprs, **uniqueexprs).collect().row(0, named=True)
    xstats = {name: stats[name] for name in statexprs}

    # a selection with no rows has no minimum and no maximum, and the bins below need both
    if xstats["rowcount"] == 0:
        raise NoEstimatorRowsError(get_no_rows_message(timestepslist, args))
    if xstats["noxvaluecount"] == xstats["rowcount"]:
        msg = f"-x {xvariable} has no value in the timesteps and the cells of the plot"
        raise ValueError(msg)
    if xstats["noxvaluecount"] > 0:
        print_warning(
            f"{xstats['noxvaluecount']} of the {xstats['rowcount']} pairs of a cell and a timestep have no finite"
            f" value of {xvariable}, thus the plot leaves them out"
        )

    xmin = xstats["xmin"] if args.xmin is None else args.xmin
    xmax = xstats["xmax"] if args.xmax is None else args.xmax

    # -xbins 0 draws the points alone. The points reach the plot only with --markers, thus this turns it on
    if args.xbins == 0:
        args.markers = True

    if xstats.get("multiple_points_per_xvalue"):
        print("There are multiple plot points per x value. Using automatic bins (use -xbins N to change this)")
        args.xbins = -1

    if args.xbins is not None and args.xbins < 0:
        xdeltamax = xstats["xdeltamax"]
        if not xdeltamax:
            # a single row gives None, and a column that holds one x value gives 0.0
            print(f"The x values give no interval to bin by ({xdeltamax}). Setting xbins to 25")
            args.xbins = 25
        else:
            args.xbins = int((xmax - xmin) / xdeltamax)
            print(
                f"Setting xbins to {args.xbins} based on data range [{xmin}, {xmax}]"
                f" and largest x interval of {xdeltamax}"
            )
            if args.xbins <= 3:
                print(f"  would have only {args.xbins} bins. Replacing with 25")
                args.xbins = 25

    if args.xbins:
        # the ions of an element have one colour and a different dash, and a shaded area or a marker of a bin has no
        # dash. Thus each ion of a plot with bins takes a colour of its own
        args.colorbyion = True
        # -xbins gives the number of bins, thus the number of edges is one more than that. It gave
        # the number of edges before, thus "-xbins 30" drew 29 bins and the help said 30
        # a range of zero width gives equal edges, and equal edges give an error. Thus one bin holds all the x values
        xbinedges = np.linspace(xmin, xmax, args.xbins + 1 if xmax > xmin else 2)
        xlower = xbinedges[:-1]
        xupper = xbinedges[1:]
        xmids = (xlower + xupper) / 2
        # only the interior edges, thus a value at an outer edge falls into the first or the last bin
        xbinindex = get_bin_index_expr(pl.col("xvalue"), xbinedges[1:-1], right_closed=True) + 1
        estimators = (
            estimators
            .with_columns(xbinindex=xbinindex)
            .filter(pl.col("xbinindex").is_between(0, len(xmids) - 1, closed="both"))
            .with_columns(xvalue_binned=pl.lit(pl.Series(xmids)).gather(pl.col("xbinindex")))
            .drop("xbinindex")
        )
    else:
        estimators = estimators.with_columns(xvalue_binned=pl.col("xvalue"))

    estimators = estimators.filter(inxrange).sort("xvalue")

    xlist, mgilist, timesteps = (stats[column] for column in ("xvalue", "modelgridindex", "timestep"))
    if not xlist:
        raise NoEstimatorRowsError(get_no_rows_message(timestepslist, args))

    return xlist, mgilist, timesteps, estimators


class NoEstimatorRowsError(ValueError):
    """The estimators hold no row for the timesteps, the cells, and the x range of a plot.

    The command then names the cells and the timesteps that hold data. The server sends this error back to the client.
    """


def get_no_rows_message(timestepslist: Collection[int] | None, args: argparse.Namespace) -> str:
    """Return the message of a plot whose selection of timesteps, cells, and x range gives no estimator row.

    The parser expands a range of cells, and the code before the plot converts -xmin and -xmax. Thus the message
    gives the size of the selection and not those values. A status line shows one line of the message.
    """
    parts: list[str] = []
    if timestepslist:
        parts.append(f"the timesteps {min(timestepslist)} to {max(timestepslist)}")
    if args.modelgridindex is not None:
        cells: list[int] = args.modelgridindex
        parts.append(
            f"the cells {', '.join(map(str, cells))}"
            if len(cells) <= 3
            else f"{len(cells)} cells from {min(cells)} to {max(cells)}"
        )
    if args.xmin is not None or args.xmax is not None:
        parts.append("the x range of -xmin and -xmax")
    return f"The estimators hold no row for {', '.join(parts)}" if parts else "The estimators hold no row"


def get_data_range(ax: mplax.Axes) -> tuple[float, float] | None:
    """Return the lowest and the highest value that the axes draw, or None when they draw nothing.

    The vertical range of the axes carries a margin above and below the data, thus a test against that
    range accepts a limit that leaves every point out of view.
    """
    drawn, _ = get_drawn_values(ax)
    finite = drawn[np.isfinite(drawn)]

    return (float(finite.min()), float(finite.max())) if finite.size > 0 else None


def get_subplot_plans(
    timestepslist: list[int],
    plotitems: list[t.Any],
    mgilist: list[int],
    modelpath: str | Path,
    estimators: pl.LazyFrame,
    args: argparse.Namespace,
    **plotkwargs: t.Any,
) -> tuple[list[SeriesPlan], SubplotSettings]:
    """Return the series of a subplot of the ARTIS estimators before the collection, and the settings of its axes.

    The caller collects the series of all the subplots in one call.
    """
    settings: SubplotSettings = {"showlegend": False, "legendncols": 1}
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
    # the -ionpoptype argument gives the type for the whole figure. A directive of a subplot replaces it
    poptype = args.poptype
    for plotitem in plotitems:
        if isinstance(plotitem, str | pl.Expr):
            remaining_plotitems.append(plotitem)
            continue
        seriestype, params = plotitem
        seriestype = get_directive_name(seriestype) or seriestype
        if seriestype == "ymin":
            # only record it. set_ylim stops the autoscale of the full axis. A limit before draw_subplot draws the
            # series thus keeps the other side at its value before the data
            settings["ymin"] = float(params)

        elif seriestype == "ymax":
            settings["ymax"] = float(params)

        elif seriestype == "ionpoptype":
            poptype = str(params)
            if poptype not in POPTYPE_YLABELS:
                suggestion = suggest_names(poptype, POPTYPE_YLABELS)
                exit_with_error(
                    f"'{poptype}' is not an ion population type",
                    f"{suggestion + ' ' if suggestion else ''}The types are {', '.join(POPTYPE_YLABELS)}",
                )

        elif seriestype == "yscale":
            # "lin" is the alias that the -yscale argument of the light curve commands also accepts
            settings["yscale"] = "linear" if params == "lin" else str(params)
        else:
            remaining_plotitems.append(plotitem)

    # draw_subplot keeps the order of the items and of their series
    plans: list[SeriesPlan] = []
    for plotitem in remaining_plotitems:
        if isinstance(plotitem, str | pl.Expr):
            variablename = plotitem.meta.output_name() if isinstance(plotitem, pl.Expr) else plotitem
            assert isinstance(variablename, str)
            showlegend = seriescount > 1 or len(variablename) > 35 or not sameylabel
            settings["showlegend"] = showlegend
            plans += plot_series(
                settings=settings,
                variable=plotitem,
                showlegend=showlegend,
                estimators=estimators,
                nounits=sameylabel,
                **plotkwargs,
            )
            if showlegend and sameylabel and ylabel is not None:
                settings["ylabel"] = ylabel
        else:  # it's a sequence of values
            seriestype, params = plotitem
            settings["showlegend"] = True

            if seriestype in {"initabundances", "initmasses"}:
                assert isinstance(params, list)
                plans += plot_init_abundances(
                    settings=settings, specieslist=params, estimators=estimators, seriestype=seriestype, **plotkwargs
                )

            elif seriestype == "levelpopulation" or seriestype.startswith("levelpopulation_"):
                plans += plot_levelpop(settings, seriestype, params, timestepslist, mgilist, modelpath, estimators)

            elif seriestype == "averageionisation":
                plans += plot_average_ionisation(settings, params, estimators, **plotkwargs)

            elif seriestype == "averageexcitation":
                plans += plot_average_excitation(
                    settings, params, timestepslist, mgilist, estimators, modelpath, **plotkwargs
                )

            else:
                seriestype, ionlist = plotitem
                # an ion population plot reads best on a log scale, thus that is the default here. A
                # yscale directive of the plot item wins over it
                settings.setdefault("yscale", "log")
                if seriestype == "populations" and len(ionlist) > 2 and settings["yscale"] == "log":
                    settings["legendncols"] = 2

                plans += plot_multi_ion_series(
                    settings=settings,
                    seriestype=seriestype,
                    ionlist=ionlist,
                    estimators=estimators,
                    modelpath=modelpath,
                    poptype=poptype,
                    args=args,
                    **plotkwargs,
                )

    return plans, settings


def finish_subplot(ax: mplax.Axes, args: argparse.Namespace, settings: SubplotSettings) -> None:
    """Apply the limits of the vertical axis and add the legend, after the draw of the data."""
    ymin = settings.get("ymin")
    ymax = settings.get("ymax")
    # Apply the requested limits now that the data has set the range of the axis. A fixed limit of the
    # plot list, e.g. the rho floor of the default list, suits one range of models. A limit outside the
    # data of this model would give an empty panel, thus test each one against the data range first.
    # set_ylim also accepts a bottom above the top, which turns the axis upside down and stays that way
    # through a later autoscale, thus the test has to come before the call and not after it.
    if ymin is not None or ymax is not None:
        # the axis label carries the LaTeX marks of a plot, thus a message on the terminal drops them
        quantity = ax.get_ylabel().translate(str.maketrans("", "", "$\\{}")) or "data"
        datarange = get_data_range(ax)
        if ymin is not None:
            if datarange is None or ymin < datarange[1]:
                ax.set_ylim(bottom=ymin)
            else:
                print_warning(f"every {quantity} value is below the requested minimum of {ymin}. Using the data range")

        if ymax is not None:
            if datarange is None or ymax > datarange[0]:
                ax.set_ylim(top=ymax)
            else:
                print_warning(f"every {quantity} value is above the requested maximum of {ymax}. Using the data range")

    if settings.get("showlegend"):
        set_legend(
            ax,
            args,
            keeptop=ymax is not None,
            keepbottom=ymin is not None,
            loc="best",
            handlelength=2,
            frameon=False,
            numpoints=1,
            # one column was the default. An ncols of None makes set_legend fit the columns
            ncols=legendncols if (legendncols := settings.get("legendncols", 1)) > 1 else None,
            markerscale=3,
        )


def get_snapshot_timestrings(
    modelpath: Path | str, timestepslist: Sequence[t.Any], *, multiplot: bool
) -> tuple[str, str]:
    """Return the timesteps and the time range of a snapshot as text for a title and a file name."""
    if multiplot:
        return f"ts{timestepslist[0]:03d}", f"{get_timestep_time(modelpath, timestepslist[0]):.2f}d"

    timesteps_flat = flatten_list(list(timestepslist))
    timestepmin = min(timesteps_flat)
    timestepmax = max(timesteps_flat)

    strtimestep = f"ts{timestepmin:03d}-ts{timestepmax:03d}" if timestepmax != timestepmin else f"ts{timestepmin:03d}"
    timelow_days, timehigh_days = (
        get_timesteps(modelpath)
        .select(
            pl.col("tstart_days").filter(pl.col("timestep") == timestepmin).first(),
            pl.col("tend_days").filter(pl.col("timestep") == timestepmax).first(),
        )
        .collect()
        .row(0)
    )
    return strtimestep, f"{timelow_days:.2f}d-{timehigh_days:.2f}d"


def get_subplot_grid(nsubplots: int, subplotsperrow: int) -> tuple[int, int]:
    """Return the number of rows and of columns of a figure of subplots, which fills each row from the left."""
    if subplotsperrow < 1:
        exit_with_error(f"-subplotsperrow {subplotsperrow} gives no column", "Give a number of 1 or more")
    ncols = max(min(subplotsperrow, nsubplots), 1)
    return math.ceil(nsubplots / ncols), ncols


# a figure of estimators often has several columns of subplots, and a frame of the full text width then gives
# labels that are small beside the data. Each frame thus takes 0.7 of the size that -figscale gives
SUBPLOT_FRAMESCALE: t.Final = 0.7
IMAGE_HEIGHT_INCHES: t.Final = 3.5
# measured with matplotlib 3.10: the y label on the left and the colour bar on the right take 1.75 inches, and the x
# label below takes 0.4 inches. Each value has a gap to the next panel
IMAGE_LABEL_INCHES: t.Final = (1.95, 0.55)
# the title has two lines
IMAGE_TITLE_INCHES: t.Final = 0.55


class LineFigureData(t.NamedTuple):
    """The data of a figure of line subplots, which the host of the model gives and the client draws."""

    subplots: list[tuple[list[SeriesData], SubplotSettings]]
    # the limits of the x axis. A range of zero width gives no limits, so that matplotlib keeps its own padding
    xlimits: tuple[float | None, float | None]
    title: str
    # the fields of the filename, see make_figure
    framefields: dict[str, int | str]


def get_line_figure_data(
    modelpath: Path | str,
    timestepslist: Collection[int] | None,
    estimators: pl.LazyFrame,
    xvariable: str,
    plotlist: list[list[t.Any]],
    args: argparse.Namespace,
) -> LineFigureData:
    """Return the data of a figure with one subplot per entry in plotlist.

    A plot of one cell against time gives the field cell, and a snapshot gives the fields timestep and timedays.
    """
    xlist, mgilist, timestepslist, estimators = get_xlist(
        xvariable=xvariable, estimators=estimators, timestepslist=timestepslist, args=args
    )
    xmin = args.xmin if args.xmin is not None else min(xlist)
    xmax = args.xmax if args.xmax is not None else max(xlist)

    subplotplans = [
        get_subplot_plans(timestepslist, plotitems, mgilist, modelpath, estimators, args) for plotitems in plotlist
    ]
    queries = [get_subplot_queries(plans, args) for plans, _ in subplotplans]
    # one collect_all runs the series of all the subplots in parallel
    frames = pl.collect_all([query for subplotqueries in queries for query in subplotqueries])
    subplots: list[tuple[list[SeriesData], SubplotSettings]] = []
    firstframe = 0
    for (plans, settings), subplotqueries in zip(subplotplans, queries, strict=True):
        subplotframes = frames[firstframe : firstframe + len(subplotqueries)]
        subplots.append((get_series_data(plans, subplotframes, args), settings))
        firstframe += len(subplotqueries)

    modelname = get_model_name(modelpath)
    framefields: dict[str, int | str]
    # a snapshot of a model of one cell also holds one cell, thus only the horizontal axis tells the two plots apart
    if xvariable in TIME_XVARIABLES and len(set(mgilist)) == 1 and len(timestepslist) > 1:
        figure_title = f"{modelname}\nCell {mgilist[0]}"
        framefields = {"cell": mgilist[0]}
    else:
        strtimestep, strtimedays = get_snapshot_timestrings(modelpath, timestepslist, multiplot=args.multiplot)
        figure_title = f"{modelname}\nTimestep {strtimestep} ({strtimedays})"
        if args.slice is not None:
            figure_title += f", {args.slicelabel}"
        print("  plotting " + figure_title.replace("\n", " "))
        framefields = {"timestep": strtimestep, "timedays": strtimedays}

    return LineFigureData(
        subplots=subplots,
        xlimits=(xmin, xmax) if xmin != xmax else (None, None),
        title=figure_title,
        framefields=framefields,
    )


def draw_line_figure(
    figuredata: LineFigureData, args: argparse.Namespace, fig: "mplfig.Figure | None" = None
) -> "mplfig.Figure":
    """Draw the line subplots of the data, and return the figure.

    If the caller gives an empty figure as fig, the function draws on it, e.g. for a window that stays open.
    """
    xvariable = args.x
    # each frame holds a size in inches, thus a grid of panels in a paper takes one room for each
    nrows, ncols = get_subplot_grid(len(figuredata.subplots), args.subplotsperrow)
    fig, axesgrid = make_frame_figure(
        args, rows=nrows, cols=ncols, aspect=0.468, sharex=True, framescale=SUBPLOT_FRAMESCALE, fig=fig
    )
    axes = axesgrid.ravel()[: len(figuredata.subplots)]
    for emptyaxis in axesgrid.ravel()[len(figuredata.subplots) :]:
        emptyaxis.set_visible(False)

    assert isinstance(axes, np.ndarray)

    # the lowest subplot of each column carries the x labels, also above an empty place of the last row
    for index, ax in enumerate(axes):
        if index + ncols >= len(axes):
            ax.tick_params(axis="x", which="both", labelbottom=True)
            if not args.hidexlabel:
                ax.set_xlabel(f"{get_varname_formatted(xvariable)}{get_units_string(xvariable)}")

    set_axis_properties(axes, args, xlimits=(*figuredata.xlimits, "-xmin"))

    startfromzero = xvariable.startswith("velocity") or xvariable == "beta"
    # get_xlist divides the range of the x axis into the bins
    xlow, xhigh = figuredata.xlimits
    xbinwidth = (xhigh - xlow) / args.xbins if args.xbins and xlow is not None and xhigh is not None else None
    for ax, (series, settings) in zip(axes, figuredata.subplots, strict=True):
        draw_subplot(ax, series, settings, args, startfromzero, xbinwidth)
        # a stacked subplot puts its lowest label beside the highest label of the subplot below
        prune_log_ticks(ax.yaxis)

    set_plot_title(axes[0], figuredata.title, args)

    return fig


def make_figure(
    figuredata: "LineFigureData | ImageFigureData", args: argparse.Namespace, frameset: "FrameSet | None" = None
) -> str:
    """Draw the figure of the data, save it, and return the output filename.

    A frame of a gif or of a merged pdf is one part of the product and not the product, thus --show
    and --open leave it alone. The caller opens the file that holds every frame.
    """
    fig = draw_figure_data(figuredata, args)
    framefields = figuredata.framefields
    if isinstance(figuredata, ImageFigureData):
        assert frameset is not None
        outfilename = format_frame_path(frameset.frametemplate, **framefields, format=args.format)
    elif "cell" in framefields:
        # a plot of one cell against time is no frame of a set, thus it names itself
        outpath = resolve_outputfile(args.outputfile, CELLEVOLUTIONFRAMENAME)
        outfilename = format_frame_path(outpath, **framefields, format=args.format)
    else:
        # a line of -slice has the plot of a snapshot, thus its file name must hold the line
        slicefields: dict[str, str] = (
            {"kind": "slice", "plane": get_slice_filetag(args)} if args.slice is not None else {}
        )
        framename = IMAGEFRAMENAME if slicefields else SNAPSHOTFRAMENAME
        # the caller of a set of frames gives the frameset, thus every frame lands beside its product
        outpath = frameset.frametemplate if frameset is not None else resolve_outputfile(args.outputfile, framename)
        outfilename = format_frame_path(outpath, **framefields, format=args.format, **slicefields)

    save_figure(fig, outfilename, args=args, isframe=frameset is not None and frameset.combines, dpi=args.dpi)

    return outfilename


def draw_figure_data(
    figuredata: "LineFigureData | ImageFigureData", args: argparse.Namespace, fig: "mplfig.Figure | None" = None
) -> "mplfig.Figure":
    """Draw the line subplots or the colour image of the data, and return the figure."""
    if isinstance(figuredata, ImageFigureData):
        return draw_image_figure(figuredata, args, fig)
    return draw_line_figure(figuredata, args, fig)


class PanelStyle(t.NamedTuple):
    """The label and the colour scale of one panel of a colour image, which the directives of its subplot give."""

    label: str
    colourscale: str | None
    vmin: float | None
    vmax: float | None
    # the position of the subplot in the plot list, which get_panel_axes_label names
    subplotindex: int = 0


class ImagePanel(t.NamedTuple):
    """One variable of a colour image, and its style."""

    colexpr: pl.Expr
    style: PanelStyle
    # the weight of each cell in the mean of a pixel, with the volume and the duration. None gives the volume and the
    # duration alone, e.g. the number density of an element gives the mean charge of its nuclei
    weightexpr: pl.Expr | None = None


class ImageFigureData(t.NamedTuple):
    """The data of a colour image of a snapshot, which the host of the model gives and the client draws."""

    grids: "list[npt.NDArray[np.float64]]"
    styles: list[PanelStyle]
    # the axes of the image, e.g. ("rcyl", "z") or ("x", "y")
    plotaxes: tuple[str, str]
    vmax_cmps: float
    title: str
    # the fields of the filename, see make_figure
    framefields: dict[str, int | str]


def get_panel_axes_label(subplotindex: int) -> str:
    """Return the label of the axes of each panel of a subplot in a colour image, e.g. for the controls of a window."""
    return f"subplot {subplotindex}"


# a colour image shows a density or a fraction of it, and the other population types belong to a line
IMAGEPOPTYPES = ("absolute", "elpop", "totalpop")


def get_ion_panel_columns(
    seriestype: str, ionlist: Sequence[str], poptype: str, estimatorcolumns: Collection[str]
) -> list[tuple[pl.Expr, str, str, pl.Expr]]:
    """Return the expression, the column name, the label, and the weight of each ion of a series in the estimators.

    get_ion_series_weight gives the weight, thus a pixel and a bin of a line take the same mean.
    """
    if seriestype == "populations" and poptype not in IMAGEPOPTYPES:
        exit_with_error(
            f"a colour image cannot show the ion population type '{poptype}'",
            f"The types for an image are {', '.join(IMAGEPOPTYPES)}",
        )

    columns: list[tuple[pl.Expr, str, str, pl.Expr]] = []
    for ionstr in ionlist:
        atomic_number, ion_stage = get_iontuple(ionstr)
        colname, ionlabel = get_column_name(seriestype, atomic_number, ion_stage)
        normfactor = get_population_normfactor(seriestype, poptype, atomic_number)
        if not {colname, *normfactor.meta.root_names()} <= set(estimatorcolumns):
            # the line plot also leaves such an ion out, thus the other ions of the series stay
            print_warning(f"Can't plot {seriestype} for {ionstr} because the estimators hold no such column")
            continue

        ispopulation = seriestype == "populations"
        label = (
            f"{ionlabel.replace('_', ' ')} {POPTYPE_YLABELS[poptype]}"
            if ispopulation
            else f"{ionlabel.replace('_', ' ')} {seriestype}{get_units_string(colname)}"
        )
        # 0/0 gives NaN for a cell that holds none of the element, and the mean leaves such a cell out
        weight = get_ion_series_weight(seriestype, poptype, atomic_number, ion_stage, estimatorcolumns)
        columns.append(((pl.col(colname) / normfactor).alias(colname), colname, label, weight))

    return columns


def get_image_panels(plotlist: list[list[t.Any]], estimatorcolumns: Collection[str], poptype: str) -> list[ImagePanel]:
    """Return one panel for each variable and each ion of the plot list.

    The directives yscale=, ymin=, and ymax= of a subplot apply to the colour scale of its panels, and
    ionpoptype= replaces poptype. A series that an image cannot show gives a warning, because the
    default plot list holds such series.
    """
    panels: list[ImagePanel] = []
    for subplotindex, plotitems in enumerate(plotlist):
        directives: dict[str, t.Any] = {
            directive: plotitem[1]
            for plotitem in plotitems
            if not isinstance(plotitem, str | pl.Expr) and (directive := get_directive_name(plotitem[0])) is not None
        }
        columns: list[tuple[pl.Expr, str, str]] = []
        weights: dict[str, pl.Expr] = {}
        for plotitem in plotitems:
            if isinstance(plotitem, pl.Expr):
                colname = plotitem.meta.output_name()
                columns.append((plotitem, colname, f"{get_varname_formatted(colname)}{get_units_string(colname)}"))
            elif isinstance(plotitem, str):
                if plotitem not in estimatorcolumns:
                    exit_with_error(
                        f"'{plotitem}' is not an estimator variable",
                        suggest_names(plotitem, estimatorcolumns)
                        or "Run with --listvariables to see the variables of this model",
                    )
                label = f"{get_varname_formatted(plotitem)}{get_units_string(plotitem)}"
                columns.append((pl.col(plotitem), plotitem, label))
            elif get_directive_name(plotitem[0]) is not None:
                continue
            elif is_ionseriestype(plotitem[0], estimatorcolumns, plotitem[1]):
                subplotpoptype = str(directives.get("ionpoptype", poptype))
                for colexpr, colname, label, weight in get_ion_panel_columns(
                    plotitem[0], plotitem[1], subplotpoptype, estimatorcolumns
                ):
                    columns.append((colexpr, colname, label))
                    weights[colname] = weight
            elif plotitem[0] == "averageionisation":
                for element in plotitem[1]:
                    elsymb = get_elsymbol(get_atomic_number(element))
                    chargeexpr, _ = get_average_charge_expr(element, estimatorcolumns)
                    colname = f"averageionisation_{elsymb}"
                    columns.append((chargeexpr.alias(colname), colname, f"Average ion charge of {elsymb}"))
                    # the line of a subplot takes the same weight, thus the image gives the charge of each nucleus
                    weights[colname] = pl.col(f"nnelement_{elsymb}")
            else:
                print_warning(f"a colour image cannot show '{plotitem[0]}', thus the figure leaves it out")

        yscale = directives.get("yscale")
        if yscale not in {None, "log", "lin", "linear"}:
            exit_with_error(f"the colour scale of an image cannot be '{yscale}'", "Give yscale=log or yscale=linear")
        colourscale = "linear" if yscale == "lin" else yscale
        vmin = float(directives["ymin"]) if "ymin" in directives else None
        vmax = float(directives["ymax"]) if "ymax" in directives else None
        panels += [
            ImagePanel(colexpr, PanelStyle(label, colourscale, vmin, vmax, subplotindex), weights.get(colname))
            for colexpr, colname, label in columns
        ]

    if not panels:
        exit_with_error(
            "the plot list holds nothing that a colour image can show",
            "Give a variable, e.g. Te, or an ion, e.g. 'Fe II'",
        )
    return panels


def get_panel_means(panels: Sequence[ImagePanel]) -> list[pl.Expr]:
    """Return the mean of each panel over the cells and the timesteps of a group, with volume x time as the weight.

    A panel with a weightexpr takes the product of that weight with volume x time.
    """
    means = []
    for panelindex, panel in enumerate(panels):
        weight = pl.col("deltavol_deltat") * (panel.weightexpr if panel.weightexpr is not None else pl.lit(1.0))
        value = panel.colexpr.cast(pl.Float64)
        # a cell or a timestep with no value must not pull the mean to zero, and a NaN is not a null
        hasvalue = value.is_not_null() & value.is_not_nan()
        weightsum = weight.filter(hasvalue).sum()
        means.append(
            pl
            .when(weightsum != 0.0)
            .then((value * weight).filter(hasvalue).sum() / weightsum)
            # equal weights make the weighted mean the plain mean, as get_line_points does
            .otherwise(value.filter(hasvalue).mean())
            .alias(f"panel{panelindex}")
        )
    return means


def get_shell_values_on_rz_grid(
    estimators: pl.LazyFrame, panels: Sequence[ImagePanel], vmax_cmps: float, timesteps: Collection[int]
) -> "list[npt.NDArray[np.float64]]":
    """Return the grid of values of each panel for a 1D model, which gives each point the value of its shell."""
    shellquery = (
        estimators
        .filter(pl.col("timestep").is_in(list(timesteps)))
        .group_by("modelgridindex")
        .agg(pl.col("vel_r_min").first(), pl.col("vel_r_max").first(), *get_panel_means(panels))
        .sort("vel_r_min")
    )
    dfshells = shellquery.collect()
    # two points across the thinnest shell of a model with equal shells, and 200 for a smooth circle
    nradialpoints = max(200, 2 * dfshells.height)
    pointwidth = vmax_cmps / nradialpoints
    vel_rcyl = (np.arange(nradialpoints) + 0.5) * pointwidth
    vel_z = (np.arange(2 * nradialpoints) + 0.5) * pointwidth - vmax_cmps
    pointradius = np.hypot(vel_rcyl, vel_z[:, np.newaxis])
    if dfshells.is_empty():
        # a timestep of a set of frames can have no estimators, and its frame stays empty
        return [np.full(pointradius.shape, np.nan) for _ in panels]

    shellindex = np.searchsorted(dfshells["vel_r_max"].to_numpy(), pointradius, side="right")
    # an empty shell has no estimators, thus a point between two shells that have them lies in neither
    shellindex_clipped = np.minimum(shellindex, dfshells.height - 1)
    isinshell = (shellindex < dfshells.height) & (pointradius >= dfshells["vel_r_min"].to_numpy()[shellindex_clipped])
    return [
        np.where(
            isinshell,
            dfshells[f"panel{panelindex}"].cast(pl.Float64).fill_null(float("nan")).to_numpy()[shellindex_clipped],
            np.nan,
        )
        for panelindex in range(len(panels))
    ]


def get_image_values(
    estimators: pl.LazyFrame,
    panels: Sequence[ImagePanel],
    modelmeta: dict[str, t.Any],
    sliceaxis: str | None,
    timesteps: Collection[int],
) -> "tuple[list[npt.NDArray[np.float64]], tuple[str, str]]":
    """Return the grid of values of each panel, and the two plot axes.

    With a sliceaxis, a pixel takes the cells of the estimators with its indices on the two other axes. The estimators
    of -slice hold the cells of one plane of a 3D model, which is normal to that axis. The estimators of -projection
    hold each cell, thus a pixel is the mean of one line of cells.

    With no sliceaxis, the grid holds the average around the z axis. A pixel then takes the cells of one ring of
    cylindrical radius and one layer of z, as the reduction of a 3D model to 2D gives. A 2D model has this grid
    already, and a 1D model gives the value of its shell at each point.

    An empty cell has no estimators and gives NaN. Each value is the mean over the cells and the timesteps, and
    get_panel_means gives the weight.
    """
    vmax_cmps = float(modelmeta["vmax_cmps"])
    if modelmeta["dimensions"] == 1:
        return get_shell_values_on_rz_grid(estimators, panels, vmax_cmps, timesteps), ("rcyl", "z")

    def cellindex(axisname: str) -> pl.Expr:
        ncells = int(modelmeta[f"ncoordgrid{axisname}"])
        return ((pl.col(f"vel_{axisname}_mid") + vmax_cmps) / (2.0 * vmax_cmps / ncells)).floor().cast(pl.Int32)

    if sliceaxis is None:
        plotaxis1, plotaxis2 = "rcyl", "z"
        is3d = modelmeta["dimensions"] == 3
        ncells1 = int(modelmeta["ncoordgridx"]) // 2 if is3d else int(modelmeta["ncoordgridrcyl"])
        vel_rcyl_mid = (pl.col("vel_x_mid") ** 2 + pl.col("vel_y_mid") ** 2).sqrt() if is3d else pl.col("vel_rcyl_mid")
        cellindex1 = (vel_rcyl_mid / (vmax_cmps / ncells1)).floor().cast(pl.Int32)
    else:
        plotaxis1, plotaxis2 = (axisname for axisname in "xyz" if axisname != sliceaxis)
        ncells1 = int(modelmeta[f"ncoordgrid{plotaxis1}"])
        cellindex1 = cellindex(plotaxis1)

    cellquery = (
        estimators
        .filter(pl.col("timestep").is_in(list(timesteps)))
        .with_columns(cellindex1=cellindex1, cellindex2=cellindex(plotaxis2))
        # a cell in a corner of the cube lies outside the largest cylinder
        .filter(pl.col("cellindex1") < ncells1)
        .group_by("cellindex1", "cellindex2")
        .agg(get_panel_means(panels))
    )
    dfcells = cellquery.collect()

    grids = []
    for panelindex in range(len(panels)):
        grid = np.full((int(modelmeta[f"ncoordgrid{plotaxis2}"]), ncells1), np.nan)
        grid[dfcells["cellindex2"].to_numpy(), dfcells["cellindex1"].to_numpy()] = (
            dfcells[f"panel{panelindex}"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
        )
        grids.append(grid)

    return grids, (plotaxis1, plotaxis2)


def get_colour_norm(panel: PanelStyle, grid: "npt.NDArray[np.float64]") -> mc.Normalize:
    """Return the colour scale of a panel, which is log only when the panel holds a value above zero."""
    colourscale = panel.colourscale or ("log" if wants_log_scale(grid.ravel()) else "linear")
    with np.errstate(invalid="ignore"):
        haspositive = bool((grid > 0.0).any())
    if colourscale == "log" and not haspositive:
        print_warning(f"'{panel.label}' holds no value above zero, thus its colour scale is linear")
        colourscale = "linear"

    if colourscale == "log":
        return mc.LogNorm(
            vmin=log_axis_limit(panel.vmin, logscale=True, argname="ymin="),
            vmax=log_axis_limit(panel.vmax, logscale=True, argname="ymax="),
        )
    return mc.Normalize(vmin=panel.vmin, vmax=panel.vmax)


def get_image_figure_data(
    modelpath: Path | str,
    timestepslist: Sequence[int],
    estimators: pl.LazyFrame,
    panels: Sequence[ImagePanel],
    modelmeta: dict[str, t.Any],
    args: argparse.Namespace,
) -> ImageFigureData:
    """Return the data of a colour image of a snapshot, with one grid for each panel.

    The image shows a plane of a 3D model for -slice, and the model at each cylindrical radius and each z without it.
    """
    grids, (plotaxis1, plotaxis2) = get_image_values(estimators, panels, modelmeta, args.sliceaxis, timestepslist)
    isplane = plotaxis1 != "rcyl"

    strtimestep, strtimedays = get_snapshot_timestrings(modelpath, timestepslist, multiplot=args.multiplot)
    projection = args.projection
    strimage = f"plane {args.slicelabel}" if isplane else "cylindrical radius and z"
    if projection is not None:
        strimage = f"mean along the {projection} axis"
    elif not isplane and modelmeta["dimensions"] == 3:
        strimage = "average around the z axis"
    figure_title = f"{get_model_name(modelpath)}\nTimestep {strtimestep} ({strtimedays}), {strimage}"
    print("  plotting " + figure_title.replace("\n", " "))

    return ImageFigureData(
        grids=grids,
        styles=[panel.style for panel in panels],
        plotaxes=(plotaxis1, plotaxis2),
        vmax_cmps=float(modelmeta["vmax_cmps"]),
        title=figure_title,
        framefields={
            "kind": "projection" if projection is not None else "slice" if isplane else "cylindrical",
            "plane": projection if projection is not None else get_slice_filetag(args) if isplane else "rz",
            "timestep": strtimestep,
            "timedays": strtimedays,
        },
    )


def draw_image_figure(
    figuredata: ImageFigureData, args: argparse.Namespace, fig: "mplfig.Figure | None" = None
) -> "mplfig.Figure":
    """Draw each panel of the data as a colour image, and return the figure.

    If the caller gives an empty figure as fig, the function draws on it, e.g. for a window that stays open.
    """
    import matplotlib.pyplot as plt

    set_mpl_style()
    plotaxis1, plotaxis2 = figuredata.plotaxes
    isplane = plotaxis1 != "rcyl"

    nrows, ncols = get_subplot_grid(len(figuredata.styles), args.subplotsperrow)
    figsize = get_image_figsize(nrows, ncols, isplane=isplane, args=args)
    if fig is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig.set_size_inches(*figsize, forward=True)
    # an image has an equal aspect, thus it is smaller than its layout box. The compressed layout moves the images
    # together at the centre, and it gives each colour bar the height of its image
    fig.set_layout_engine("compressed")
    axesgrid = fig.subplots(nrows, ncols, squeeze=False)
    vmax_on_c = figuredata.vmax_cmps / C_cm_per_s
    # the axis of an image holds v/c. -x velocity takes km/s, and every other x variable takes v/c already
    xscale_to_c = km_to_cm / C_cm_per_s if args.x == "velocity" else 1.0
    xmin_on_c = None if args.xmin is None else args.xmin * xscale_to_c
    xmax_on_c = None if args.xmax is None else args.xmax * xscale_to_c
    for ax, style, grid in zip(axesgrid.flat, figuredata.styles, figuredata.grids, strict=False):
        norm = get_colour_norm(style, grid)
        values = np.ma.masked_invalid(grid)
        if isinstance(norm, mc.LogNorm):
            # a log colour scale cannot show a value of zero or below, thus such a cell stays empty
            values = np.ma.masked_less_equal(values, 0.0)
        edges1 = np.linspace(-vmax_on_c if isplane else 0.0, vmax_on_c, grid.shape[1] + 1)
        edges2 = np.linspace(-vmax_on_c, vmax_on_c, grid.shape[0] + 1)
        # the grid of a 1D model has 80 000 points, which are slow and large as vector shapes
        image = ax.pcolormesh(edges1, edges2, values, norm=norm, rasterized=True)
        ax.set_label(get_panel_axes_label(style.subplotindex))
        colourbar = fig.colorbar(image, ax=ax)
        colourbar.set_label(style.label, fontsize=args.labelfontsize)
        # an empty cell has no value, and black sets it apart from the lowest colour of the scale
        ax.set_facecolor("black")
        ax.tick_params(which="both", color="white")
        if args.labelfontsize is not None:
            colourbar.ax.tick_params(labelsize=args.labelfontsize)
        # the image has its own y range, thus only the x range and the font size come from the arguments
        set_axis_properties(ax, args, xlimits=(xmin_on_c, xmax_on_c, "-xmin"), setyaxis=False)
        ax.set_aspect("equal")
        ax.set_xlabel(r"v$_{r,xy}$ [$c$]" if plotaxis1 == "rcyl" else rf"v$_{plotaxis1}$ [$c$]")
        ax.set_ylabel(rf"v$_{plotaxis2}$ [$c$]")
    for ax in list(axesgrid.flat)[len(figuredata.styles) :]:
        ax.set_visible(False)

    if not args.notitle:
        title = fig.suptitle(figuredata.title)
        # a long title of a narrow figure, e.g. of one cylindrical image, goes past the edges of the figure
        titlewidth = title.get_window_extent().width / fig.dpi + 0.2
        if titlewidth > figsize[0]:
            fig.set_size_inches(titlewidth, figsize[1])

    return fig


def get_image_figsize(nrows: int, ncols: int, *, isplane: bool, args: argparse.Namespace) -> tuple[float, float]:
    """Return the size in inches of a figure of colour images, from the shape of an image and the space of its labels.

    An image has an equal aspect, and the image at each cylindrical radius has half the width of a plane. The labels
    have a font size in points, thus their space stays the same at each -figscale. -figwidthscale scales all the width.
    """
    imageheight = IMAGE_HEIGHT_INCHES * args.figscale * SUBPLOT_FRAMESCALE
    imagewidth = imageheight if isplane else imageheight / 2.0
    labelwidth, labelheight = IMAGE_LABEL_INCHES
    width = ncols * (imagewidth + labelwidth) * (getattr(args, "figwidthscale", None) or 1.0)
    height = nrows * (imageheight + labelheight) + (0.0 if args.notitle else IMAGE_TITLE_INCHES)
    return width, height


def get_slice_filetag(args: argparse.Namespace) -> str:
    """Return the plane or the line of -slice as text for a file name, e.g. "z=-0.2c" or "z=0,y=0"."""
    return str(args.slicelabel).replace(" ", "").replace("km/s", "kmps")


def complete_plotitem(prefix: str, **kwargs: t.Any) -> list[str]:
    """Return the names that the tab key offers for a positional argument.

    argcomplete calls this function. It gives these names:

    - the estimator variables;
    - the types of series;
    - the directives;
    - the folders.

    It reads no estimator file. Such a read prints progress, and the shell takes that text as a name.
    Such a read can also build the parquet cache of a full run.
    """
    from argcomplete.completers import DirectoriesCompleter

    from artistools.estimators.core import PREFIX_GROUPS

    names = [
        *(key for key, info in VARIABLES.items() if not info.group),
        *PREFIX_GROUPS,
        *VARIABLE_ALIASES,
        *SERIESTYPES,
        *(f"{directive}=" for directive in DIRECTIVES),
    ]
    folders = DirectoriesCompleter()(prefix=prefix, **kwargs)

    return sorted({name for name in names if name.startswith(prefix)} | set(folders))


def filter_listed_columns(columns: Sequence[str], searchterms: Sequence[str]) -> list[str]:
    """Return the columns that hold one of the search terms, or every column when there is no term.

    A model holds many variables, thus "--listvariables heating" searches the list. An empty term
    matches every column, thus this function ignores it.
    """
    lowerterms = [term.lower() for term in searchterms if term]
    if not lowerterms:
        return list(columns)

    return [column for column in columns if any(term in column.lower() for term in lowerterms)]


def print_listing(args: argparse.Namespace, estimatorcolumns: Sequence[str]) -> None:
    """Print the estimator variables of the model, or the variables that hold a search term.

    A search shows fewer variables than the full list. Thus the heading names the search terms.
    """
    searchterms = [term for term in args.plotitems if term]
    listedcolumns = filter_listed_columns(estimatorcolumns, searchterms)
    if searchterms and not listedcolumns:
        exit_with_error(
            f"no estimator variable of this model holds {' or '.join(searchterms)}",
            suggest_names(searchterms[0], estimatorcolumns) or "Give no name to list every variable",
        )

    # the heading and the folder are progress, thus --quiet leaves the listing alone
    print(
        f"Estimator variables of '{get_model_name(args.modelpath)}'"
        + (f" that hold {' or '.join(searchterms)}" if searchterms else "")
    )
    print_modelpath(args.modelpath)

    print_product(args, summarise_columns(listedcolumns, fullnuclides=args.listnuclides))
    print_product(args, 'Plot a variable with e.g. "artistools plotestimators Te rho -t 300"')


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    itemsarg = addarg_positional_items(
        parser,
        dest="plotitems",
        metavar="variable",
        helptext=(
            "The estimator variables of one subplot, e.g. 'artistools plotestimators Te TR'. The ARTIS "
            "folder comes after them, e.g. 'artistools plotestimators Te mymodel'. The working folder "
            "is the default. Each -plot adds a different subplot. With --listvariables these names "
            "search the listing"
        ),
    )
    # argcomplete reads this attribute, which argparse does not declare
    itemsarg.completer = complete_plotitem  # ty:ignore[unresolved-attribute]  # pyrefly: ignore[missing-attribute]

    # the default is None and not Path(), thus resolve_positional_modelpath sees an explicit value
    addarg_modelpath(parser, helptext="Path to ARTIS folder (or virtual path e.g. codecomparison/ddc10/cmfgen)")

    addarg_modelgridindex(parser, helptext="Model grid cell for the time evolution plot")

    addarg_timestep(parser, helptext="Timestep number for internal structure plot")

    addarg_timedays(parser, helptext="Time in days to plot for internal structure plot")

    addarg_timeminmax(parser)

    parser.add_argument(
        "--multiplot",
        action="store_true",
        help=(
            "Make one snapshot for each timestep of the time range, which is the whole run when no time is given."
            " The default PDF format puts all the snapshots in one file"
        ),
    )

    parser.add_argument("-x", default=None, help="Horizontal axis variable, e.g. velocity, timestep, or time")

    addarg_axislimits(
        parser,
        include_y=False,
        xminhelp="Plot range: minimum x value in the units of -x, e.g. km/s for velocity, c for beta, days for time",
        xmaxhelp="Plot range: maximum x value in the units of -x, e.g. km/s for velocity, c for beta, days for time",
    )

    parser.add_argument(
        "-xbins",
        type=int,
        default=None,
        help=(
            "Number of x bins between xmax and xmin"
            " (-1 for an automatic bin size, 0 for the points alone with no average line)"
        ),
    )

    parser.add_argument("--hidexlabel", action="store_true", help="Hide the bottom horizontal axis label")

    parser.add_argument("--markers", action="store_true", help="Plot markers instead of shaded area")

    addarg_filter(parser)

    parser.add_argument("-format", "-f", default="pdf", choices=["pdf", "png"], help="Set format of output plot files")

    parser.add_argument(
        "--makegif", action="store_true", help="Make a gif of the time evolution, one frame per timestep"
    )

    addarg_notitle(parser)

    parser.add_argument(
        "--listvariables",
        "--listvars",
        action="store_true",
        help=(
            "List the estimator variables of this model, then stop. A name before the folder searches "
            "the listing, e.g. artistools plotestimators heating --listvariables"
        ),
    )

    parser.add_argument(
        "--listnuclides",
        action="store_true",
        help="List the estimator variables as --listvariables does, and name every nuclide of a family",
    )

    parser.add_argument(
        "-plotlist",
        "-plot",
        "-p",
        nargs="*",
        type=str,
        action="append",
        help=(
            "List of plots to generate, one -plot for each subplot. The variables of one subplot need no "
            "-plot, e.g. 'artistools plotestimators Te TR'. Give estimator variables, ions, a type "
            "of series with the names that it covers, or a directive of the form key=value. Examples: "
            "-plot Te TR -plot nne -plot SrI 'Sr II'. A type of series comes first and groups the names "
            f"after it, e.g. -plot averageionisation Fe Ni. The types are {', '.join(SERIESTYPES)}, and "
            "an estimator that names an ion, e.g. -plot gamma_NT 'Fe II' or -plot cooling_coll 'Fe II'. "
            "A name that also gives a total of the cell, e.g. cooling_coll, gives the ions when the "
            "model holds a column for one of them. Give the total its own subplot, e.g. -plot "
            "cooling_coll -plot cooling_coll 'Fe II'. Quote an ion that holds "
            f"a space. The directives are {', '.join(f'{name}=' for name in DIRECTIVES)}, e.g. "
            "-plot Te TR yscale=lin -plot rho yscale=log ymin=1e-17 -plot 'Fe II' 'Fe III' ionpoptype=elpop. "
            "The directive ionpoptype= sets the normalisation of an ion population series to one of "
            f"{', '.join(POPTYPE_YLABELS)}. The subplots share one horizontal axis, thus -xmin and -xmax set "
            "that axis for the whole figure"
        ),
    )

    # the ionpoptype= directive of a plot item replaces this argument, because each subplot holds its
    # own type. The argument gives the type for the whole figure, thus an older script still runs
    parser.add_argument(
        "-ionpoptype",
        "-poptype",
        dest="poptype",
        default="absolute",
        choices=list(POPTYPE_YLABELS),
        help=argparse.SUPPRESS,
    )

    addarg_legend(parser)

    addarg_labelfontsize(parser)

    addarg_figscale(parser, include_figwidthscale=True)
    # deprecated spelling of -figwidthscale kept as a hidden alias
    parser.add_argument("-scalefigwidth", dest="figwidthscale", type=float, help=argparse.SUPPRESS)

    addarg_show(parser)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a window with controls for the time, the cell, and the subplots, and show the command",
    )
    addarg_verbose(parser)

    addarg_dpi(parser, default=600)

    addarg_output(parser, kind="file", default=Path(), helptext="Filename for PDF file")

    parser.add_argument(
        "--colorbyion",
        action="store_true",
        help=(
            "Give each ion a colour of its own, and not the colour of its element. A plot with x bins always does this"
        ),
    )

    parser.add_argument(
        "--classicartis",
        action="store_true",
        help=(
            "Read the estimator format of the classic ARTIS code. A modern run writes the modern"
            " format, even when it takes the classic options"
        ),
    )

    parser.add_argument(
        "-readonlymgi",
        default=False,
        choices=["alongaxis", "cone"],
        help="Option to read only selected mgi and choice of which mgi to select. Choose which axis with -axis",
    )

    parser.add_argument(
        "-axis",
        default="+z",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        help="Choose an axis for use with -readonlymgi. Hint: for negative use e.g. -axis=-z",
    )

    parser.add_argument(
        "-slice",
        default=None,
        metavar="PLANE",
        help=(
            "Plot each variable as a colour image of a plane of a 3D model. Give the two axes of a plane through"
            " the origin, e.g. -slice xy. As an alternative, give the normal axis and its velocity in km/s or as"
            " a fraction of c, e.g. -slice z=-0.2c. The plot shows the layer of cells that holds the plane,"
            " which is the layer above it for a plane between two layers. Two axes give a line of cells, e.g."
            " -slice z=0,y=0 along the x axis, and the plot then shows the variables against that velocity"
        ),
    )

    parser.add_argument(
        "-dimensionreduce",
        "-dim",
        type=int,
        default=None,
        choices=[1, 2],
        help=(
            "Show the model in this number of dimensions for the plot of a snapshot. 1 is the default, which"
            " plots the cells against -x. 2 shows each variable as a colour image at each cylindrical radius and"
            " each z. A 3D model gives the average around the z axis, as -dimensionreduce 2 of makeartismodel"
            " does. A 1D model gives the value of its shell at each point. With -slice, the image shows a plane"
            " of a 3D model"
        ),
    )

    parser.add_argument(
        "-coneangle",
        type=float,
        default=30.0,
        help="The full angle of the cone in degrees for -readonlymgi cone. The half angle is coneangle/2",
    )

    parser.add_argument(
        "-projection",
        default=None,
        choices=["x", "y", "z"],
        help=(
            "Show each variable as a colour image of the mean along this axis of a 3D model, e.g. -projection z gives"
            " an image in x and y. Each pixel is the mean over one line of cells along the axis, with the volume"
            " times the timestep duration as the weight"
        ),
    )

    parser.add_argument(
        "-subplotsperrow",
        type=int,
        default=1,
        help="The number of subplots in each row of the figure, or of the panels of a colour image",
    )


# -x with one of these variables gives a plot against time, and each other variable gives a snapshot
TIME_XVARIABLES: t.Final = frozenset({"time", "timestep"})


def time_is_given(args: argparse.Namespace) -> bool:
    """Return True if the arguments select a time or a timestep.

    A timestep of 0 and a time of 0 are real selections, and both are falsy. Thus this tests for absence and not for
    truth.
    """
    return any(value is not None for value in (args.timedays, args.timemin, args.timemax, args.timestep))


def get_default_x(*, timegiven: bool, makegif: bool) -> str:
    """Return the x variable of a command with no -x.

    A gif and the frames of --multiplot hold one snapshot for each timestep, thus their x axis shows a spatial
    variable. makegif is True for both of them. A time that the user gave also selects one snapshot.
    """
    return "time" if not timegiven and not makegif else "velocity"


def set_x_and_timesteps(args: argparse.Namespace, modelpath: Path) -> tuple[int, int]:
    """Apply the default x variable and the default time range, and return the first and last timestep.

    A plot against time takes every timestep, thus a user who gives no time gets the full evolution. A
    gif, the frames of --multiplot, a list of the variables, and a plot of one cell also take every timestep. A plot of
    a snapshot against a spatial variable needs a time, thus it keeps the default time range.
    """
    notimegiven = not time_is_given(args)
    framepertimestep = args.makegif or args.multiplot
    wantswholerun = framepertimestep or args.listvariables or args.listnuclides
    if notimegiven and (wantswholerun or args.modelgridindex is not None or args.x in {None, *TIME_XVARIABLES}):
        args.timestep = f"0-{len(get_timestep_times(modelpath)) - 1}"

    if args.x is None:
        args.x = get_default_x(timegiven=not notimegiven, makegif=framepertimestep)
        print(f"Setting x variable to {args.x}")

    # a frame of a gif or of --multiplot holds one timestep. A plot against time draws all the timesteps in one plot,
    # thus such a command wrote one plot with no gif, and its title named the first timestep alone
    if framepertimestep and args.x in TIME_XVARIABLES:
        flag = "--makegif" if args.makegif else "--multiplot"
        exit_with_error(
            f"{flag} gives one snapshot for each timestep, thus its horizontal axis cannot show {args.x}",
            f"Remove -x {args.x} to plot each snapshot against the velocity, or remove {flag} for one plot"
            f" against {args.x}",
        )

    # get_time_range returns these times, thus keep what the user gave for the message below
    given_timemin, given_timemax = args.timemin, args.timemax
    timestepmin, timestepmax, args.timemin, args.timemax = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )

    if timestepmin == timestepmax == -1:
        # the reader then found no cell and named the model, which hid the time that the user gave
        given = " and ".join(
            f"{name} {value}"
            for name, value in (("-timemin", given_timemin), ("-timemax", given_timemax))
            if value is not None
        )
        tstarts = get_timestep_times(modelpath, loc="start")
        tends = get_timestep_times(modelpath, loc="end")
        exit_with_error(
            f"{given} lies outside the run, which goes from {tstarts[0]:.1f} to {tends[-1]:.1f} days",
            "Give a time inside that range",
        )

    return timestepmin, timestepmax


def parse_slice_argument(slicetext: str) -> list[tuple[str, float, str]]:
    """Return the axis, its velocity [cm/s], and the label of each condition of -slice.

    "xy" and "z=-0.2c" give one condition, which is a plane. "z=0,y=0" gives two, which is a line.
    """
    from artistools.spectra import get_velocity_label
    from artistools.spectra import parse_velocity_argument

    text = slicetext.strip().lower()
    helptext = (
        "Give a plane through the origin, e.g. -slice xy. As an alternative, give an axis and its velocity,"
        " e.g. -slice z=-0.2c. Two axes give a line, e.g. -slice z=0,y=0"
    )
    if len(text) == 2 and text[0] != text[1] and set(text) <= set("xyz"):
        normalaxis = next(axisname for axisname in "xyz" if axisname not in text)
        return [(normalaxis, 0.0, f"{normalaxis} = 0")]

    conditions: list[tuple[str, float, str]] = []
    for conditiontext in text.split(","):
        axisname, separator, velocitytext = (part.strip() for part in conditiontext.partition("="))
        if axisname not in {"x", "y", "z"} or not separator:
            exit_with_error(f"'{slicetext}' is not a plane or a line of the model", helptext)
        try:
            velocity_kmps, unit = parse_velocity_argument(velocitytext)
        except argparse.ArgumentTypeError as err:
            exit_with_error(str(err), helptext)

        label = get_velocity_label(velocity_kmps, unit)
        if velocity_kmps == 0.0:
            label = "0"
        conditions.append((axisname, velocity_kmps * km_to_cm, f"{axisname} = {label}"))

    if len(conditions) > 2 or len({axisname for axisname, _, _ in conditions}) < len(conditions):
        exit_with_error(f"'{slicetext}' must give one axis for a plane, or two different axes for a line", helptext)
    return conditions


def get_layer_index(velocity_cmps: float, vmax_cmps: float, ncells: int) -> int:
    """Return the index of the layer of cells that holds a velocity on one axis of a 3D model.

    A velocity between two layers takes the layer above it. The quotient of such a velocity can lie one
    rounding step below the whole number, e.g. 24.999999999999996 for vmax = 6724085530.798534 cm/s and 50
    cells, thus a small part of a cell goes on before the floor.
    """
    return min(math.floor((velocity_cmps + vmax_cmps) / (2.0 * vmax_cmps / ncells) + 1e-6), ncells - 1)


def resolve_snapshot_arguments(args: argparse.Namespace) -> list[tuple[str, float, str]]:
    """Apply -slice and -dimensionreduce to the arguments, and return the conditions of -slice.

    A plane of -slice gives a colour image, and a line of -slice gives a plot against the velocity on
    its axis. An argument that disagrees with the selection stops the command.
    """
    conditions: list[tuple[str, float, str]] = parse_slice_argument(args.slice) if args.slice is not None else []
    projection = args.projection
    if projection is not None:
        if conditions:
            exit_with_error("-projection and -slice select different cells", "Remove -projection or -slice")
        if args.dimensionreduce == 1:
            exit_with_error("-projection gives a colour image, thus -dimensionreduce 1 does not apply", "Remove one")
        args.dimensionreduce = 2
    if conditions:
        # one condition is a plane, and two conditions are a line along the axis that stays
        slicedimensions = 2 if len(conditions) == 1 else 1
        if args.dimensionreduce not in {None, slicedimensions}:
            exit_with_error(
                f"-slice {args.slice} gives a plot with -dimensionreduce {slicedimensions}, and not"
                f" {args.dimensionreduce}",
                "Remove -dimensionreduce",
            )
        args.dimensionreduce = slicedimensions
    elif args.dimensionreduce is None:
        args.dimensionreduce = 1

    isimage = args.dimensionreduce == 2
    # a projection groups the cells as a plane normal to its axis does, and it reads every cell of each line
    args.sliceaxis = projection or (conditions[0][0] if isimage and conditions else None)
    args.slicelabel = ", ".join(label for _, _, label in conditions)
    if not isimage and not conditions:
        return conditions

    selection = "-projection" if projection else "-slice" if conditions else "-dimensionreduce 2"
    if args.readonlymgi or args.modelgridindex is not None:
        exit_with_error(
            f"{selection} selects the cells of the plot, thus -readonlymgi and -cell do not apply",
            f"Remove {selection}, or remove the other argument",
        )

    if isimage:
        if args.x is not None:
            exit_with_error(
                f"the two axes of a colour image are velocities, thus -x {args.x} does not apply",
                f"Remove -x, or remove {selection}",
            )
        ignored = [
            name
            for name, given in (
                ("-xbins", args.xbins is not None),
                ("-filtermovingavg", bool(args.filtermovingavg)),
                ("-filtersavgol", bool(args.filtersavgol)),
                ("--markers", bool(args.markers)),
            )
            if given
        ]
        if ignored:
            print_warning(f"{', '.join(ignored)} do not apply to a colour image")
        # a colour image is a snapshot, thus it takes the time range of a plot against the velocity
        args.x = "velocity"
    elif args.x is None:
        lineaxis = next(axisname for axisname in "xyz" if axisname not in {axisname for axisname, _, _ in conditions})
        args.x = f"vel_{lineaxis}_mid_on_c"

    return conditions


def select_cells_of_slice(
    args: argparse.Namespace, modelpath: Path, conditions: Sequence[tuple[str, float, str]]
) -> None:
    """Select the cells of a 3D model that hold the plane or the line of -slice."""
    args.modelgridindex = get_cells_of_slice(modelpath, list(conditions))
    print(f"Getting the {len(args.modelgridindex)} cells that hold {args.slicelabel}")


@on_model_host
def get_cells_of_slice(modelpath: Path, conditions: Sequence[tuple[str, float, str]]) -> list[int]:
    """Return the cells of a 3D model that hold the plane or the line of -slice. The host of a remote model reads it."""
    lzmodel, modelmeta = get_modeldata(modelpath)
    if modelmeta["dimensions"] != 3:
        exit_with_error(
            f"-slice needs a 3D model, and this model has {modelmeta['dimensions']} dimension(s)",
            "Remove -slice to plot the variables against the velocity",
        )
    vmax_cmps = float(modelmeta["vmax_cmps"])
    for axisname, velocity_cmps, label in conditions:
        if abs(velocity_cmps) >= vmax_cmps:
            exit_with_error(
                f"{label} lies outside the model, which ends at {vmax_cmps / C_cm_per_s:.3g}c",
                "Give a velocity inside the model",
            )
        layerindex = get_layer_index(velocity_cmps, vmax_cmps, int(modelmeta[f"ncoordgrid{axisname}"]))
        poscolumn = pl.col(f"pos_{axisname}_min")
        lzmodel = lzmodel.filter(poscolumn == poscolumn.unique().sort().get(layerindex))

    return lzmodel.select("modelgridindex").collect()["modelgridindex"].to_list()


def select_cells_along_axis(args: argparse.Namespace) -> None:
    """Select the cells on an axis or in a cone of a 3D model, and record the two axes that stay.

    The selection functions of slice1dfromconein3dmodel read these axis names from the arguments.
    """
    args.sliceaxis = args.axis[1]
    assert args.axis[0] in {"+", "-"}
    args.positive_axis = args.axis[0] == "+"
    otheraxes = [axisname for axisname in "xyz" if axisname != args.sliceaxis]
    args.other_axis1, args.other_axis2 = otheraxes[0], otheraxes[1]

    args.modelgridindex = get_cells_along_axis(normalize_path_list(args.modelpath)[0], args)


@on_model_host
def get_cells_along_axis(modelpath: Path, args: argparse.Namespace) -> list[int]:
    """Return the cells with matter on an axis or in a cone of a 3D model. The host of a remote model reads it."""
    lzmodel, modelmeta = get_modeldata(modelpath)
    # the selection reads the positions of the cells on three axes, which only a 3D model gives
    if modelmeta["dimensions"] != 3:
        exit_with_error(
            f"-readonlymgi {args.readonlymgi} needs a 3D model, and this model has {modelmeta['dimensions']}"
            " dimension(s)",
            "Remove -readonlymgi to plot every cell against the velocity",
        )
    if args.readonlymgi == "alongaxis":
        print(f"Getting mgi along {args.axis} axis")
        dfmodel = lzmodel.select("modelgridindex", "rho", "pos_x_min", "pos_y_min", "pos_z_min").collect()
        dfselectedcells = get_profile_along_axis(dfmodel, args)
    elif args.readonlymgi == "cone":
        print(f"Getting mgi lying within a cone around the {args.axis} half-axis")
        # the cone selection reads the mid-point positions, which are derived columns
        lzmodel = add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        dfselectedcells = make_cone(args, lzmodel, logprint=print)
    else:
        msg = f"Invalid args.readonlymgi: {args.readonlymgi}"
        raise ValueError(msg)

    # the estimators hold the zero-based modelgridindex, not the one-based inputcellid
    return list(dfselectedcells.filter(pl.col("rho") > 0)["modelgridindex"])


@on_model_host
def report_data_available(modelpath: Path, *, classicartis: bool) -> None:
    """Name the cells and the timesteps for which the model holds estimator data."""
    print("No data was found for the requested timesteps/cells.")
    cells, timesteps = (
        scan_estimators(modelpath=modelpath, classicartis=classicartis)
        .select(pl.col("modelgridindex").unique().sort().implode(), pl.col("timestep").unique().sort().implode())
        .collect()
        .row(0)
    )
    print(f"Cells with data: {cells}")
    print(f"Timesteps with data: {timesteps}")


SNAPSHOTFRAMENAME = "plotestimators_{timestep}_{timedays}.{format}"
IMAGEFRAMENAME = "plotestimators_{kind}_{plane}_{timestep}_{timedays}.{format}"
CELLEVOLUTIONFRAMENAME = "plotestimators_cell{cell:05d}.{format}"


def get_default_plotlist() -> list[t.Any]:
    """Return the plot items that a command with no -plot argument draws.

    The commented lines are examples that a user can copy. Each one gives a shape that -plot accepts.
    """
    return [
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
        ["rho", ["_yscale", "log"]],
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


def resolve_plotlist(args: argparse.Namespace, estimatorcolumns: Collection[str], modelpath: Path) -> list[list[t.Any]]:
    """Return the plot items of each subplot, with the aliases resolved and the directives at the end.

    The default list names particular elements and the NLTE populations, thus a model that holds no such data loses
    those items. A user who names an item always keeps it, thus an error in that name still stops the command.
    """
    plotlist: list[t.Any] = args.plotlist
    if not plotlist:
        plotlist, skippedplotlist = get_model_default_plotlist(estimatorcolumns, modelpath)
        for plotitems, reason in skippedplotlist:
            print(f"Skipping the default subplot {plotitems}, because {reason}")

        if not plotlist:
            msg = "No default plots apply to this model. Choose what to plot with -plot (e.g. -plot Te TR)"
            raise ValueError(msg)

    return [normalise_plotitems(plotitems, estimatorcolumns) for plotitems in plotlist]


def prepare_snapshot(
    args: argparse.Namespace, estimators: pl.LazyFrame, modelmeta: dict[str, t.Any], plotlist: list[list[t.Any]]
) -> tuple[pl.LazyFrame, list[ImagePanel]]:
    """Return the estimators of the selected cells of a snapshot, and the panels of a colour image.

    A model faster than 0.3c takes v/c in place of the velocity on the horizontal axis. A plot that is not an image
    has no panels.
    """
    if args.x == "velocity" and modelmeta["vmax_cmps"] > 0.3 * C_cm_per_s:
        args.x = "beta"
        # the user gave -xmin and -xmax in km/s for -x velocity, and the axis is now v/c
        if args.xmin is not None:
            args.xmin *= km_to_cm / C_cm_per_s
        if args.xmax is not None:
            args.xmax *= km_to_cm / C_cm_per_s

    if args.readonlymgi or args.slice is not None:
        estimators = estimators.filter(pl.col("modelgridindex").is_in(args.modelgridindex or []))

    panels: list[ImagePanel] = []
    if args.projection is not None and modelmeta["dimensions"] != 3:
        exit_with_error(
            f"-projection needs a 3D model, and this model has {modelmeta['dimensions']} dimension(s)",
            "Give -dimensionreduce 2 for the average around the z axis",
        )
    if args.dimensionreduce == 2:
        panels = get_image_panels(plotlist, estimators.collect_schema().names(), args.poptype)
        # an image reads a small number of the columns, and a set of frames writes a copy of the estimators
        panelcolumns = {
            name
            for panel in panels
            for expr in (panel.colexpr, panel.weightexpr)
            if expr is not None
            for name in expr.meta.root_names()
        }
        estimators = estimators.select(
            cs.by_name("timestep", "modelgridindex", "deltavol_deltat", *sorted(panelcolumns)) | cs.starts_with("vel_")
        )

    return estimators, panels


def write_snapshot_figures(
    args: argparse.Namespace, figures: "Sequence[LineFigureData | ImageFigureData]", timesteps_included: list[int]
) -> None:
    """Draw the figures of the internal structure at one time, and write one file for each frame.

    With --multiplot each timestep gives one frame. artistools then joins the frames into a gif or into
    one PDF file.
    """
    isimage = isinstance(figures[0], ImageFigureData)
    # a gif or a merged PDF file holds every frame, thus many figures make one product
    firstts, lastts = timesteps_included[0], timesteps_included[-1]
    frameset = resolve_frameset_paths(
        args.outputfile,
        framecount=len(figures),
        framename=IMAGEFRAMENAME if isimage or args.slice is not None else SNAPSHOTFRAMENAME,
        productname=f"plotestimators_evolution_ts{firstts:03d}-ts{lastts:03d}.gif" if args.makegif else None,
        combines=len(figures) > 1 and (args.makegif or args.format == "pdf"),
        gifduration=1000.0 if args.makegif else None,
    )
    outputfiles = [make_figure(figuredata, args, frameset) for figuredata in figures]
    frameset.finish(outputfiles, args)


def resolve_positional_args(args: argparse.Namespace) -> None:
    """Move the positional arguments to the model path and to the plot list.

    A user names the variables directly, e.g. "artistools plotestimators Te TR -t 300". The ARTIS folder
    comes after them. The positional variables share one subplot. The names after one -plot also share
    one subplot.
    """
    # -plot takes every name that follows it, thus the last -plot group can hold the folder of the user.
    # The code moves that folder to the end of the positional list, because one rule then sets the order
    # of both forms. The positional list can hold a variable of its own, e.g. "Te -p rho mymodel"
    endswithfolder = bool(args.plotitems) and item_names_a_folder(str(args.plotitems[-1]))
    lastgroup = args.plotlist[-1] if args.plotlist else None
    if not endswithfolder and isinstance(lastgroup, list) and lastgroup and item_names_a_folder(str(lastgroup[-1])):
        args.plotitems = [*args.plotitems, lastgroup.pop()]
        if not lastgroup:
            # -plot held the folder alone, thus that subplot has no variable of its own
            args.plotlist.pop()

    if plotvars := resolve_positional_modelpath(args, "plotitems"):
        args.plotlist = [plotvars, *(args.plotlist or [])]


def require_artis_folder(modelpath: Path) -> None:
    """Stop with an error message when the path does not name an ARTIS folder.

    The command reads the working folder when the user names no folder. A user can run the command in
    a folder that is not an ARTIS folder. The message then names that folder and not an absent file.
    """
    if path_is_codecomparison(modelpath) or folder_is_artis_run(modelpath):
        return

    # a user often runs the command one level above the runs, thus name the folders that are near
    nearby = artis_subfolders(modelpath)
    helptext = (
        f"Name one of these subfolders after the variables: {', '.join(nearby)}"
        if nearby
        else (
            'An ARTIS folder holds input.txt. Name one after the variables, e.g. "artistools plotestimators Te mymodel"'
        )
    )

    if modelpath == Path():
        exit_with_error("no ARTIS folder was given, and the working folder is not an ARTIS folder", helptext)

    exit_with_error(f"'{modelpath}' is not an ARTIS folder", helptext)


def resolve_plot_args(args: argparse.Namespace) -> tuple[Path, list[int]]:
    """Apply the defaults and the cell selection to the arguments, and return the model path and the timesteps."""
    resolve_positional_args(args)
    modelpath = Path(args.modelpath)
    require_artis_folder(modelpath)
    sliceconditions = resolve_snapshot_arguments(args)
    timestepmin, timestepmax = set_x_and_timesteps(args, modelpath)
    wantslisting = args.listvariables or args.listnuclides

    if not wantslisting:
        print(
            f"'{get_model_name(modelpath)}': "
            + get_time_range_text(timestepmin, timestepmax, args.timemin, args.timemax)
        )
        print_modelpath(modelpath)

    if sliceconditions and not wantslisting:
        select_cells_of_slice(args, modelpath, sliceconditions)
    elif args.readonlymgi:
        select_cells_along_axis(args)

    return modelpath, list(range(timestepmin, timestepmax + 1))


def get_plot_estimators(
    args: argparse.Namespace,
    modelpath: Path,
    timesteps_included: Sequence[int],
    batchcaches: "Sequence[EstimatorBatchCache] | None" = None,
) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
    """Return the estimators of the selected cells and timesteps with the model data of each cell, and the metadata.

    batchcaches gives the current parquet caches of all the batches of the run, e.g. for a window that draws many
    plots. The scan then checks and converts no file.
    """
    estimators = scan_estimators(
        modelpath=modelpath,
        modelgridindex=args.modelgridindex,
        timestep=tuple(timesteps_included),
        classicartis=args.classicartis,
        verbose=args.verbose,
        batchcaches=batchcaches,
    )
    return join_cell_modeldata(estimators=estimators, modelpath=modelpath, verbose=args.verbose)


def add_plot_columns(
    args: argparse.Namespace, estimators: pl.LazyFrame, modelmeta: dict[str, t.Any]
) -> tuple[pl.LazyFrame, list[str]]:
    """Return the estimators with the columns that the plot reads, and the names of all the columns."""
    # the average around the z axis reads all the cells, and it applies the limit of the cylindrical radius itself
    if args.modelgridindex is None and args.dimensionreduce == 1:
        estimators = estimators.filter(pl.col("vel_r_mid") <= modelmeta["vmax_cmps"])

    estimators = estimators.with_columns(deltavol_deltat=pl.col("volume") * pl.col("twidth_days"))
    return estimators, estimators.collect_schema().names()


@on_model_host
def get_plot_columns(modelpath: Path, args: argparse.Namespace, timesteps_included: list[int]) -> list[str]:
    """Return the names of the columns that a plot can read, for --listvariables and --listnuclides."""
    estimators, modelmeta = get_plot_estimators(args, modelpath, timesteps_included)
    return add_plot_columns(args, estimators, modelmeta)[1]


@on_model_host
def get_figures_data(
    modelpath: Path,
    args: argparse.Namespace,
    timesteps_included: list[int],
    batchcaches: "Sequence[EstimatorBatchCache] | None" = None,
) -> "tuple[list[LineFigureData | ImageFigureData], dict[str, t.Any]]":
    """Return the data of each figure of the arguments, and the arguments that the data code changed.

    The host of a remote model runs this function, thus it reads the estimators there and only the data to draw
    comes back. The data code changes some arguments, e.g. -xbins, and the client draws with the new values. A
    selection with no row gives NoEstimatorRowsError. batchcaches gives the current parquet caches of the run, e.g. for
    a window that draws many plots.
    """
    import copy

    from artistools.misc.remote import is_plain_value

    # a path of the arguments comes back with the name of the host, thus only the changed values come back. The copy
    # keeps each list as it was, e.g. for a change of one item
    plainargs = copy.deepcopy({key: value for key, value in vars(args).items() if is_plain_value(value)})

    def get_changed_args() -> dict[str, t.Any]:
        return {key: value for key, value in vars(args).items() if key in plainargs and plainargs[key] != value}

    estimators, modelmeta = get_plot_estimators(args, modelpath, timesteps_included, batchcaches)
    estimators, estimatorcolumns = add_plot_columns(args, estimators, modelmeta)
    plotlist = resolve_plotlist(args, estimatorcolumns, modelpath)

    assert args.x is not None
    if args.x in TIME_XVARIABLES:
        figuredata = get_line_figure_data(modelpath, timesteps_included, estimators, args.x, plotlist, args)
        return [figuredata], get_changed_args()

    estimators, panels = prepare_snapshot(args, estimators, modelmeta, plotlist)
    # get_xlist finds a line plot with no row in its statistics query, and an image has no such query. pl.len() lets
    # projection pushdown read 2 columns
    if args.dimensionreduce == 2 and estimators.select(pl.len()).collect().item() == 0:
        raise NoEstimatorRowsError(get_no_rows_message(timesteps_included, args))

    # a gif needs one PNG frame for each timestep, because imageio reads PNG. Thus --makegif sets --multiplot and
    # -format png
    if args.makegif:
        args.multiplot = True
        args.format = "png"

    # each frame reads its timestep from the parquet caches of the run. A copy of the selected timesteps held every
    # column, and for 10 frames of a 3D run it made the command 16 s and 5.5 GB in place of 5.4 s and 0.85 GB
    frames = [[timestep] for timestep in timesteps_included] if args.multiplot else [timesteps_included]
    figures: list[LineFigureData | ImageFigureData] = []
    skippedtimesteps: list[int] = []
    for frame in frames:
        try:
            figures.append(
                get_image_figure_data(modelpath, frame, estimators, panels, modelmeta, args)
                if args.dimensionreduce == 2
                else get_line_figure_data(modelpath, frame, estimators, args.x, plotlist, args)
            )
        except NoEstimatorRowsError:
            # a run that stopped early has no estimators for its last timesteps. The frames of the other timesteps
            # still give a product, thus a set of frames leaves out a frame that has no row
            if not args.multiplot:
                raise
            skippedtimesteps.extend(frame)

    if skippedtimesteps:
        print_warning(
            f"The estimators hold no row for {len(skippedtimesteps)} of the {len(frames)} timesteps, from"
            f" {min(skippedtimesteps)} to {max(skippedtimesteps)}, thus the plot leaves out their frames."
        )
    if not figures:
        raise NoEstimatorRowsError(get_no_rows_message(timesteps_included, args))

    return figures, get_changed_args()


def draw_plot(
    args: argparse.Namespace, fig: "mplfig.Figure", batchcaches: "Sequence[EstimatorBatchCache] | None" = None
) -> None:
    """Draw the plot of one frame of the arguments on an empty figure, e.g. for a window that stays open.

    The plot reads the estimators as the command does, thus the window draws the plot of the command. batchcaches
    gives the current parquet caches of the run, and a window keeps them between its plots. The arguments must select
    one plot. A list of the variables, a gif, and a set of frames each give a different action.
    """
    modelpath, timesteps_included = resolve_plot_args(args)
    figures, changedargs = get_figures_data(modelpath, args, timesteps_included, batchcaches)
    vars(args).update(changedargs)
    draw_figure_data(figures[0], args, fig)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS estimators."""
    # the dispatcher parses the command line and gives args alone, thus the viewer then reads sys.argv
    fromdispatcher = args is not None
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.interactive:
        from artistools.estimators.interactive import run_viewer
        from artistools.viewertools.core import get_command_tokens

        run_viewer(get_command_tokens(args, argsraw, kwargs, fromdispatcher=fromdispatcher))
        return

    modelpath, timesteps_included = resolve_plot_args(args)
    if args.listvariables or args.listnuclides:
        # a list of the variables needs only the schema, thus do not count the rows
        print_listing(args, get_plot_columns(modelpath, args, timesteps_included))
        return

    try:
        figures, changedargs = get_figures_data(modelpath, args, timesteps_included)
    except NoEstimatorRowsError as exc:
        report_data_available(modelpath, classicartis=args.classicartis)
        # the command makes no plot, thus a script that runs it must see a failure
        exit_with_error(str(exc), "The output above names the cells and the timesteps that hold data")

    vars(args).update(changedargs)

    if args.x in TIME_XVARIABLES:
        make_figure(figures[0], args)
    else:
        write_snapshot_figures(args, figures, timesteps_included)
