"""Plot the binned opacities of the ejecta against wavelength."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl

from artistools.constants import C_cm_per_s
from artistools.constants import km_to_cm
from artistools.ejectaopacity import addarg_excitationtemperature
from artistools.ejectaopacity import DEFAULT_TAUCAPS
from artistools.ejectaopacity import format_taucap
from artistools.ejectaopacity import get_capped_columns
from artistools.ejectaopacity import get_cell_batches
from artistools.ejectaopacity import get_cell_estimators
from artistools.ejectaopacity import get_excitation_temperature_column
from artistools.ejectaopacity import get_expansion_opacities
from artistools.ejectaopacity import get_expopac_grid
from artistools.ejectaopacity import get_lambda_bin_edges
from artistools.ejectaopacity import get_opacity_atomic_data
from artistools.ejectaopacity import get_opacity_columns
from artistools.ejectaopacity import get_opacity_lines
from artistools.ejectaopacity import get_planck_mean_opacities
from artistools.ejectaopacity import get_selected_timestep
from artistools.ejectaopacity import print_planck_mean_method
from artistools.inputmodel import get_cell_selection
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_figscale
from artistools.misc import addarg_legend
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import addarg_yscale
from artistools.misc import df_filter_minmax_bracketed
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_timestep_time
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import print_warning
from artistools.misc.general import get_progress_class
from artistools.plottools import make_frame_figure
from artistools.plottools import RESIDUALROWHEIGHT
from artistools.plottools import save_figure
from artistools.plottools import set_auto_yscale
from artistools.plottools import set_axis_properties
from artistools.plottools import set_legend
from artistools.plottools import set_log_ticks_every_decade
from artistools.plottools import set_plot_title
from artistools.spectra import get_velocity_label
from artistools.spectra import parse_velocity_argument

if t.TYPE_CHECKING:
    import matplotlib.typing as mplt
    import numpy.typing as npt

# the width of a bin in Angstroms for a run with no rpkt.h. ARTIS used this width in 2026
DEFAULT_DELTALAMBDA: t.Final = 20.0

# the line style of each capped opacity, in the order of the caps. The expansion opacity and an opacity capped at 1
# are often almost equal, thus the dashes show the difference. The Planck mean uses the dotted line, thus this list
# has none
CAPPEDLINESTYLES: t.Final = ("--", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (8, 2)))
# where two opacities are equal, their lines are at the same place. A thin line covers less of the line below it
OPACITYLINE_WIDTH: t.Final = 0.6


def get_opacity_series(taucaps: Sequence[float]) -> list[tuple[str, str, "mplt.LineStyleType"]]:
    """Return the column, the label, and the line style of each opacity.

    The capped opacities are between the expansion opacity and the line-binned opacity in the list, because their
    values are between these two opacities.
    """
    return [
        ("exopac", "Expansion opacity", "-"),
        *(
            (
                column,
                rf"Line-binned, $\tau_\mathrm{{S}}$ capped at {format_taucap(taucap)}",
                CAPPEDLINESTYLES[index % len(CAPPEDLINESTYLES)],
            )
            for index, (column, taucap) in enumerate(get_capped_columns(taucaps).items())
        ),
        ("linebinned", "Line-binned", "-"),
    ]


def get_massweighted_opacities(
    adata: pl.DataFrame,
    time_days: float,
    dfestimators: pl.DataFrame,
    lambda_bin_edges: Sequence[float],
    planckrange: tuple[float, float] | None = None,
    taucaps: Sequence[float] = DEFAULT_TAUCAPS,
) -> tuple[pl.DataFrame, float]:
    """Return the mean binned opacities over the cells, and the mean Planck mean of the expansion opacity.

    The mass of each cell is the weight of both means. For one cell, the result is the opacity of that cell. The bins
    have one width. The Planck mean takes the bins with a middle in planckrange. A planckrange of None, or a run with
    no cell temperature, gives a Planck mean of NaN.

    The column linecount gives the number of lines in each bin. It counts each line that the opacities sum, thus it
    is the same for each cell.
    """
    if dfestimators.height > 1:
        print_detail(f"The curves are the mass-weighted mean of the opacities of {dfestimators.height} cells")
    if planckrange is not None:
        print_planck_mean_method(*planckrange)
    lambda_bin_edges = list(lambda_bin_edges)
    deltalambda = lambda_bin_edges[1] - lambda_bin_edges[0]
    opacitylines = get_opacity_lines(adata, dfestimators.columns, lambda_bin_edges, time_days)
    opacitycolumns = get_opacity_columns(taucaps)

    batchsums: list[pl.DataFrame] = []
    planckmean_times_mass = 0.0
    planckmass = 0.0
    # the bar gives the rate and the time until the end, which a model of many cells needs
    for dfcellbatch in get_progress_class()(
        get_cell_batches(dfestimators, len(lambda_bin_edges) - 1), desc="Calculating the opacities", unit="batch"
    ):
        dfbinnedopacities = get_expansion_opacities(opacitylines, dfcellbatch, lambda_bin_edges, time_days, taucaps)
        batchsums.append(
            dfbinnedopacities.group_by("lambda_angstroms_binindex").agg(
                (pl.col(*opacitycolumns) * pl.col("mass_g")).sum(), pl.col("mass_g").sum()
            )
        )
        if planckrange is None:
            continue
        dfplanckmean = get_planck_mean_opacities(
            dfbinnedopacities.filter(pl.col("lambda_angstroms_bin_mid").is_between(*planckrange))
        )
        planckmean_times_mass += dfplanckmean.select(pl.col("planckmean_opacity").dot(pl.col("mass_g"))).item()
        planckmass += dfplanckmean.select(pl.col("mass_g").sum()).item()

    linecounts = opacitylines.dflines.group_by("lambda_angstroms_binindex").agg(linecount=pl.len())
    # the index of a bin gives its middle, thus the sums of the batches need no float key
    dfopacities = (
        pl
        .concat(batchsums)
        .group_by("lambda_angstroms_binindex")
        .agg(pl.all().sum())
        .join(linecounts, on="lambda_angstroms_binindex", how="left")
        .sort("lambda_angstroms_binindex")
        .select(
            pl.col(*opacitycolumns) / pl.col("mass_g"),
            pl.col("linecount").fill_null(0),
            lambda_angstroms_bin_mid=lambda_bin_edges[0] + (pl.col("lambda_angstroms_binindex") + 0.5) * deltalambda,
            lambda_angstroms_lower=lambda_bin_edges[0] + pl.col("lambda_angstroms_binindex") * deltalambda,
            lambda_angstroms_upper=lambda_bin_edges[0] + (pl.col("lambda_angstroms_binindex") + 1) * deltalambda,
        )
    )
    return dfopacities, planckmean_times_mass / planckmass if planckmass > 0.0 else math.nan


def get_computed_bin_edges(
    modelpath: Path | str, xmin: float, xmax: float, deltalambda: float | None, movingaveragewidth: float
) -> tuple[list[float], float]:
    """Return the edges of the bins that the plot needs, and the width of a bin.

    The bins lie on the grid of rpkt.h of the run. The command calculates only the bins of the plot range, and one bin
    more than half the window of the moving average at each end. Each moving average of the plot then takes the same
    bins as a calculation of the full grid. A run with no rpkt.h takes bins from xmin, with a width of 20 Angstroms.
    """
    grid = get_expopac_grid(modelpath)
    if deltalambda is None:
        deltalambda = DEFAULT_DELTALAMBDA if grid is None else grid[2]
        if grid is None:
            print_warning(f"{modelpath} has no artis/rpkt.h, thus each bin takes a width of {deltalambda:g} Angstroms")
    marginbins = get_window_bins(movingaveragewidth, deltalambda) // 2 + 1 if movingaveragewidth > 0.0 else 0
    lower, upper = xmin - marginbins * deltalambda, xmax + marginbins * deltalambda
    gridmin, gridmax = (lower, upper) if grid is None else grid[:2]
    gridlowers = get_lambda_bin_edges(gridmin, gridmax, deltalambda)[:-1]
    lowers = (
        df_filter_minmax_bracketed(pl.DataFrame({"lower": gridlowers}), "lower", lower, upper)
        .collect()
        .get_column("lower")
        .to_list()
    )
    if not lowers or lowers[0] >= xmax or lowers[-1] + deltalambda <= xmin:
        msg = (
            f"The grid of rpkt.h, {gridmin:g} to {gridmax:g} Angstroms, holds no bin from {xmin:g} to {xmax:g}"
            " Angstroms"
        )
        raise ValueError(msg)
    edges = [*lowers, lowers[-1] + deltalambda]
    gridtext = (
        "" if grid is None else f" of the {len(gridlowers)} bins of rpkt.h from {gridmin:g} to {gridmax:g} Angstroms"
    )
    print_detail(
        f"{len(lowers)} wavelength bins of {deltalambda:g} Angstroms from {edges[0]:g} to {edges[-1]:g}"
        f" Angstroms{gridtext}"
    )
    return edges, deltalambda


def get_window_bins(width: float, deltalambda: float) -> int:
    """Return the odd number of bins nearest to the width of the window of the moving average.

    An odd number of bins puts the centre of the window at the middle of a bin. If the width is an even number of
    bins, two odd numbers are equally near, and the function gives the larger one. The tolerance of 1e-9 makes a
    quotient such as 1.2 / 0.2 = 5.999999999999999 give the same result as 6.
    """
    return 2 * math.floor(width / deltalambda / 2 + 1e-9) + 1


def get_moving_averages(dfopacities: pl.DataFrame, windowbins: int, columns: Sequence[str]) -> pl.DataFrame:
    """Return the centred moving average of each column at the middle of each bin.

    Near each end of the range, the window holds fewer bins.
    """
    return dfopacities.select(
        "lambda_angstroms_bin_mid", pl.col(*columns).rolling_mean(window_size=windowbins, center=True, min_samples=1)
    )


def plot_opacities(
    dfopacities: pl.DataFrame,
    dfmovingaverages: pl.DataFrame | None,
    planckmean: float,
    title: str,
    args: argparse.Namespace,
) -> None:
    """Plot each type of binned opacity against wavelength, with the panels below it, and save the figure.

    Each bin is a short horizontal line from its lower edge to its upper edge. With a moving average, a line in the
    same colour gives the moving average of each opacity. The panel below gives the ratio of each line-binned
    opacity to the expansion opacity. A finite planckmean gives a dotted line at the Planck mean of the expansion
    opacity. With --showlinecount, a third panel gives the number of lines in each bin.

    The plot takes the bins of the x range, and the moving average keeps one point past each end, thus its line
    reaches the edge of the frame. A value outside the x range then does not change the y range.
    """
    dfopacities = dfopacities.filter(
        pl.col("lambda_angstroms_upper") > args.xmin, pl.col("lambda_angstroms_lower") < args.xmax
    )
    if dfmovingaverages is not None:
        dfmovingaverages = df_filter_minmax_bracketed(
            dfmovingaverages, "lambda_angstroms_bin_mid", args.xmin, args.xmax
        ).collect()
    rowheights = (1.0, RESIDUALROWHEIGHT, RESIDUALROWHEIGHT) if args.showlinecount else (1.0, RESIDUALROWHEIGHT)
    # the frame takes one column of the page, as the plots of the other ejecta properties do, e.g. plotdensity
    fig, axes = make_frame_figure(args, rows=len(rowheights), fullwidth=False, rowheights=rowheights)
    ax, ratioaxis = axes[0, 0], axes[1, 0]
    bottomaxis = axes[-1, 0]

    # a NaN after each bin breaks the line, thus each series stays one line for the legend and for the y scale
    binbreaks = np.full(dfopacities.height, np.nan)
    binx = np.column_stack([dfopacities["lambda_angstroms_lower"], dfopacities["lambda_angstroms_upper"], binbreaks])

    def get_bin_line(column: str) -> "tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]":
        opacities = dfopacities[column].to_numpy()
        return binx.ravel(), np.column_stack([opacities, opacities, binbreaks]).ravel()

    def get_line(column: str) -> "tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]":
        if dfmovingaverages is None:
            return get_bin_line(column)
        return dfmovingaverages["lambda_angstroms_bin_mid"].to_numpy(), dfmovingaverages[column].to_numpy()

    opacityseries = get_opacity_series(args.taucaps)
    colors: dict[str, mplt.ColorType] = {}
    for column, label, linestyle in opacityseries:
        # the moving average goes on top of the bins, and the legend shows the moving average when there is one
        (binlines,) = ax.plot(
            *get_bin_line(column),
            linewidth=OPACITYLINE_WIDTH,
            linestyle=linestyle if dfmovingaverages is None else "-",
            # a butt cap ends the line of a bin at the edge of the bin
            solid_capstyle="butt",
            label=label if dfmovingaverages is None else None,
        )
        colors[column] = binlines.get_color()
        if dfmovingaverages is not None:
            ax.plot(
                *get_line(column), linewidth=OPACITYLINE_WIDTH, linestyle=linestyle, color=colors[column], label=label
            )

    if math.isfinite(planckmean):
        ax.axhline(
            planckmean,
            color="0.3",
            linestyle=":",
            linewidth=1.0,
            label=rf"Planck mean of the expansion opacity, {planckmean:.3g} cm$^2$/g",
        )

    linex, expansion = get_line("exopac")
    for column, _label, linestyle in opacityseries[1:]:
        _, opacities = get_line(column)
        ratioaxis.plot(
            linex,
            np.divide(opacities, expansion, out=np.full_like(opacities, np.nan), where=expansion != 0.0),
            linewidth=OPACITYLINE_WIDTH,
            linestyle=linestyle,
            color=colors[column],
            solid_capstyle="butt",
        )

    if args.showlinecount:
        print_detail(
            "The number of lines in each bin counts each line of an ion with estimators, if ARTIS keeps the lower"
            " level and the upper level"
        )
        bottomaxis.plot(*get_bin_line("linecount"), linewidth=OPACITYLINE_WIDTH, color="0.3", solid_capstyle="butt")
        bottomaxis.set_ylabel("Lines\nper bin")
        # the number of lines in a bin is from zero to some thousands. A log scale hides a bin with no line
        bottomaxis.set_yscale("log")
        set_axis_properties(bottomaxis, args, setyaxis=False)
        set_log_ticks_every_decade(bottomaxis.yaxis)

    ax.set_ylabel(r"Opacity [cm$^2$/g]")
    bottomaxis.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    ratioaxis.set_ylabel("Ratio to\nexpansion")
    # the line-binned opacity is up to 1000 times the expansion opacity, and the opacity capped at 1 is 1 to 1.58
    # times it
    ratioaxis.set_yscale("log")
    set_auto_yscale(ax, args)
    set_axis_properties(ax, args)
    set_axis_properties(ratioaxis, args, setyaxis=False)
    set_log_ticks_every_decade(ax.yaxis)
    set_log_ticks_every_decade(ratioaxis.yaxis)
    set_plot_title(ax, title, args)
    set_legend(ax, args, title=get_smoothing_text(dfopacities, args.movingaveragewidth), alignment="left")

    save_figure(fig, args.outputfile, args=args)


def get_smoothing_text(dfopacities: pl.DataFrame, movingaveragewidth: float) -> str | None:
    """Return the title of the legend, which gives the moving average of the lines, or None."""
    if movingaveragewidth <= 0.0 or dfopacities.is_empty():
        return None
    deltalambda = dfopacities["lambda_angstroms_upper"][0] - dfopacities["lambda_angstroms_lower"][0]
    windowbins = get_window_bins(movingaveragewidth, deltalambda)
    return rf"Lines: moving average of {movingaveragewidth:g} $\mathrm{{\AA}}$ ({windowbins} bins)"


def get_velocity_bounds_text(
    vmin: tuple[float, t.Literal["kmps", "c"]] | None, vmax: tuple[float, t.Literal["kmps", "c"]] | None
) -> str:
    """Return the bounds of the velocity range in the unit of the user, e.g. "vmin = 0.1c and vmax = 60000 km/s"."""
    bounds = [f"{name} = {get_velocity_label(*v)}" for name, v in (("vmin", vmin), ("vmax", vmax)) if v is not None]
    return " and ".join(bounds)


def select_velocity_range(
    dfestimators: pl.DataFrame,
    vmin: tuple[float, t.Literal["kmps", "c"]] | None,
    vmax: tuple[float, t.Literal["kmps", "c"]] | None,
) -> pl.DataFrame:
    """Return the cells with a mid-point velocity in the range, and print the number of cells that match."""
    if vmin is None and vmax is None:
        return dfestimators

    speedoflight_kmps = C_cm_per_s / km_to_cm
    dfselected = dfestimators.filter(
        get_cell_selection(
            vmin=None if vmin is None else vmin[0] / speedoflight_kmps,
            vmax=None if vmax is None else vmax[0] / speedoflight_kmps,
        )
    )
    boundstext = get_velocity_bounds_text(vmin, vmax)
    print_detail(
        f"{dfselected.height} of {dfestimators.height} cells with estimators are in the velocity range with {boundstext}"
    )
    if dfselected.is_empty():
        msg = f"No cell with estimators is in the velocity range with {boundstext}"
        raise ValueError(msg)
    return dfselected


def get_average_cell(dfestimators: pl.DataFrame) -> pl.DataFrame:
    """Return one cell with the mass-weighted mean composition, temperature, and density of the cells, and log them.

    The opacity per gram depends on the ion densities per gram, thus the mean takes each n_ion / rho. The expansion
    opacity also depends on the density, thus the cell takes the mean density. A cell with no T_exc does not count in
    the mean T_exc.
    """
    mass = pl.col("mass_g")
    hastemperature = pl.col("T_exc") > 0.0
    meanrho = (mass * pl.col("rho")).sum() / mass.sum()
    dfcell = dfestimators.select(
        pl.col("modelgridindex").first(),
        pl.col("timestep").first(),
        (mass * pl.col("T_exc")).filter(hastemperature).sum().truediv(mass.filter(hastemperature).sum()).alias("T_exc"),
        meanrho.alias("rho"),
        mass.sum(),
        *(
            (meanrho * (mass * pl.col(column) / pl.col("rho")).sum() / mass.sum()).alias(column)
            for column in dfestimators.columns
            if column.startswith("nnion_")
        ),
    )
    temperature = dfcell["T_exc"].item()
    if temperature is None or not math.isfinite(temperature):
        msg = "No cell has a value of T_exc, thus the mean cell has no temperature"
        raise ValueError(msg)
    print_detail(
        f"--averagecell: one cell with the mass-weighted means of {dfestimators.height} cells (the ion densities per"
        f" gram, the density, and T_exc of the cells with T_exc > 0): T_exc = {temperature:.0f} K,"
        f" rho = {dfcell['rho'].item():.3g} g/cm^3"
    )
    return dfcell


def get_cells_text(
    modelgridindex: int | None,
    vmin: tuple[float, t.Literal["kmps", "c"]] | None,
    vmax: tuple[float, t.Literal["kmps", "c"]] | None,
    averagetemperaturetext: str | None = None,
) -> str:
    """Return the text of the title that names the cells of the plot, with each velocity in the unit of the user.

    averagetemperaturetext gives T_exc of the mean cell of --averagecell, e.g. "TJ = 5000 K".
    """
    if modelgridindex is not None:
        return f"cell {modelgridindex}"
    boundstext = get_velocity_bounds_text(vmin, vmax)
    cellstext = f"the cells with {boundstext}" if boundstext else "all cells"
    if averagetemperaturetext is None:
        return f"mass-weighted mean of {cellstext}"
    return f"mean composition of {cellstext} at {averagetemperaturetext}"


def parse_taucap(text: str) -> float:
    """Return the cap of tau_sobolev in the text, which must be above zero."""
    try:
        value = float(text)
    except ValueError as exc:
        msg = f"invalid float value: {text!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not value > 0.0:
        msg = f"{text} is not a positive cap of tau_sobolev. Give a value above 0"
        raise argparse.ArgumentTypeError(msg)
    return value


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path(), helptext="Path of the ARTIS model")
    addarg_timestep(parser, helptext="Timestep to plot, e.g. 40 or last")
    addarg_timedays(parser, kind="str")
    addarg_modelgridindex(
        parser,
        helptext="Cell to plot. If you do not give a cell, the plot shows the mass-weighted mean over all the cells",
    )
    for name, limit in (("vmin", "Minimum"), ("vmax", "Maximum")):
        parser.add_argument(
            f"-{name}",
            type=partial(parse_velocity_argument, requireunit=True),
            default=None,
            metavar="VELOCITY",
            help=(
                f"{limit} mid-point velocity of a cell for the mass-weighted mean. Give a number that ends in c or"
                " km/s, e.g. 0.1c or 5000km/s"
            ),
        )

    # the command bins the opacities over the range of the plot, thus a smaller range takes less time
    addarg_axislimits(
        parser,
        xmindefault=1000.0,
        xmaxdefault=25000.0,
        xminhelp="Minimum wavelength in Angstroms",
        xmaxhelp="Maximum wavelength in Angstroms",
        wavelength_aliases=True,
    )
    parser.add_argument(
        "-deltalambda",
        type=float,
        default=None,
        help=(
            "Wavelength bin width in Angstroms. The default is expopac_deltalambda of artis/rpkt.h in the folder of"
            f" the run, or {DEFAULT_DELTALAMBDA:g} if the run has no rpkt.h"
        ),
    )
    parser.add_argument(
        "-movingaveragewidth",
        type=float,
        default=200.0,
        help=(
            "Width in Angstroms of the window of the moving average. The window holds the nearest odd number of bins."
            " Give 0 for no moving average"
        ),
    )

    parser.add_argument("--logscalex", action="store_true", help="Use a log scale for the wavelength axis")
    parser.add_argument(
        "--averagecell",
        action="store_true",
        help=(
            "Replace the cells with one cell of their mass-weighted mean composition, T_exc, and density. The"
            " calculation then takes the time of one cell"
        ),
    )
    parser.add_argument(
        "-taucaps",
        type=parse_taucap,
        nargs="+",
        default=list(DEFAULT_TAUCAPS),
        metavar="TAUCAP",
        help=(
            "Caps of tau_sobolev. The plot shows one line-binned opacity for each cap, with each tau_sobolev capped"
            " at that value, e.g. -taucaps 0.1 1 10"
        ),
    )
    parser.add_argument(
        "--showlinecount", action="store_true", help="Add a panel with the number of lines in each wavelength bin"
    )
    parser.add_argument(
        "--showplanckmean",
        action="store_true",
        help="Draw a line at the mass-weighted Planck mean of the expansion opacity over the wavelength range",
    )
    addarg_excitationtemperature(parser)
    addarg_yscale(parser, default="log")
    addarg_notitle(parser)
    addarg_legend(parser)
    addarg_figscale(parser, helptext="Scale factor for plot area. 1.0 fills one column of a page")
    addarg_show(parser)
    addarg_output(parser, kind="file", defaultname="plotopacity.pdf", helptext="Path/filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the expansion opacity and the line-binned opacities against wavelength."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    time_days = get_timestep_time(args.modelpath, timestep)
    print(f"Plotting {get_model_logname(args.modelpath)} at {time_days:.1f}d (timestep {timestep})")
    print_modelpath(args.modelpath)
    modelgridindex = get_single_modelgridindex(args.modelgridindex)
    lambda_bin_edges, deltalambda = get_computed_bin_edges(
        args.modelpath, args.xmin, args.xmax, args.deltalambda, args.movingaveragewidth
    )

    temperaturecolumn = get_excitation_temperature_column(args.modelpath, args.exctemperature)
    dfestimators = select_velocity_range(
        get_cell_estimators(args.modelpath, timestep, modelgridindex, temperaturecolumn), args.vmin, args.vmax
    )
    averagetemperaturetext = None
    if args.averagecell and modelgridindex is None:
        dfestimators = get_average_cell(dfestimators)
        averagetemperaturetext = f"{temperaturecolumn} = {dfestimators['T_exc'].item():.0f} K"

    dfopacities, planckmean = get_massweighted_opacities(
        adata=get_opacity_atomic_data(args.modelpath),
        time_days=time_days,
        dfestimators=dfestimators,
        lambda_bin_edges=lambda_bin_edges,
        planckrange=(args.xmin, args.xmax) if args.showplanckmean else None,
        taucaps=args.taucaps,
    )

    # the frame takes one column of the page, thus the cells take a second line of the title
    title = (
        f"{get_model_name(args.modelpath)} at {time_days:.1f}d (timestep {timestep})\n"
        f"{get_cells_text(modelgridindex, args.vmin, args.vmax, averagetemperaturetext)}"
    )
    windowbins = get_window_bins(args.movingaveragewidth, deltalambda)
    dfmovingaverages = (
        get_moving_averages(dfopacities, windowbins, get_opacity_columns(args.taucaps))
        if args.movingaveragewidth > 0.0
        else None
    )
    plot_opacities(dfopacities, dfmovingaverages, planckmean, title, args)
