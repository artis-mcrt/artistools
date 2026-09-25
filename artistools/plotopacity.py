"""Plot the binned opacities of the ejecta against wavelength."""

import argparse
import time
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from artistools.ejectaopacity import get_cell_batches
from artistools.ejectaopacity import get_cell_estimators
from artistools.ejectaopacity import get_expansion_opacities
from artistools.ejectaopacity import get_lambda_bin_edges
from artistools.ejectaopacity import get_opacity_atomic_data
from artistools.ejectaopacity import get_opacity_lines
from artistools.ejectaopacity import get_selected_timestep
from artistools.ejectaopacity import OPACITYCOLUMNS
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_nolegend
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import addarg_yscale
from artistools.misc import get_model_name
from artistools.misc import get_single_modelgridindex
from artistools.misc import get_timestep_time
from artistools.misc import parse_cli_args
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_auto_yscale
from artistools.plottools import set_axis_properties
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title

# where two opacities are equal, their lines are at the same place. Each line is thinner than the
# line below it, thus each colour stays visible
OPACITYSERIES = (
    ("exopac", "Expansion opacity", 2.4),
    ("linebinned_maxone", r"Line-binned, $\tau_\mathrm{S}$ capped at 1", 1.4),
    ("linebinned", "Line-binned", 0.6),
)


def get_massweighted_opacities(
    adata: pl.DataFrame,
    time_days: float,
    dfestimators: pl.DataFrame,
    lambdamin: float,
    lambdamax: float,
    deltalambda: float,
) -> pl.DataFrame:
    """Return the mean binned opacities over the cells, with the mass of each cell as the weight.

    For one cell, the result is the opacity of that cell.
    """
    lambda_bin_edges = get_lambda_bin_edges(lambdamin, lambdamax, deltalambda)
    opacitylines = get_opacity_lines(adata, dfestimators.columns, lambda_bin_edges, time_days)

    batchsums = []
    cellsdone = 0
    time_start = time.perf_counter()
    for dfcellbatch in get_cell_batches(dfestimators):
        batchsums.append(
            get_expansion_opacities(opacitylines, dfcellbatch, lambda_bin_edges, time_days)
            .group_by("lambda_angstroms_binindex", "lambda_angstroms_bin_mid")
            .agg((pl.col(*OPACITYCOLUMNS) * pl.col("mass_g")).sum(), pl.col("mass_g").sum())
        )
        cellsdone += dfcellbatch.height
        secondspercell = (time.perf_counter() - time_start) / cellsdone
        print(
            f"  {cellsdone} of {dfestimators.height} cells,"
            f" approximately {secondspercell * (dfestimators.height - cellsdone):.0f} s until the end"
        )

    return (
        pl
        .concat(batchsums)
        .group_by("lambda_angstroms_binindex", "lambda_angstroms_bin_mid")
        .agg(pl.all().sum())
        .select(
            pl.col(*OPACITYCOLUMNS) / pl.col("mass_g"),
            lambda_angstroms_lower=pl.col("lambda_angstroms_bin_mid") - deltalambda / 2,
            lambda_angstroms_upper=pl.col("lambda_angstroms_bin_mid") + deltalambda / 2,
        )
        .sort("lambda_angstroms_lower")
    )


def get_moving_averages(dfopacities: pl.DataFrame, windowbins: int) -> pl.DataFrame:
    """Return the centred moving average of each opacity at the middle of each bin.

    Near each end of the range, the window holds fewer bins.
    """
    return dfopacities.select(
        pl.mean_horizontal("lambda_angstroms_lower", "lambda_angstroms_upper").alias("lambda_angstroms_bin_mid"),
        pl.col(*OPACITYCOLUMNS).rolling_mean(window_size=windowbins, center=True, min_samples=1),
    )


def plot_opacities(
    dfopacities: pl.DataFrame, dfmovingaverages: pl.DataFrame | None, title: str, args: argparse.Namespace
) -> None:
    """Plot each type of binned opacity against wavelength, and save the figure.

    Each bin is a horizontal line from its lower edge to its upper edge, with no vertical line to the next bin.
    If dfmovingaverages is not None, a line in the same colour gives the moving average of each opacity.
    """
    fig, axes = make_frame_figure(args)
    ax = axes[0][0]

    # a NaN after each bin breaks the line, thus each series stays one line for the legend and for the y scale
    binbreaks = np.full(dfopacities.height, np.nan)
    xvalues = np.column_stack([dfopacities["lambda_angstroms_lower"], dfopacities["lambda_angstroms_upper"], binbreaks])
    colors = []
    for column, label, linewidth in OPACITYSERIES:
        opacities = dfopacities[column].to_numpy()
        (binlines,) = ax.plot(
            xvalues.ravel(),
            np.column_stack([opacities, opacities, binbreaks]).ravel(),
            linewidth=linewidth,
            # a butt cap ends the line at the edge of the bin
            solid_capstyle="butt",
            # the legend shows the moving average when there is one, because a short bin is hard to see there
            label=label if dfmovingaverages is None else None,
        )
        colors.append(binlines.get_color())

    if dfmovingaverages is not None:
        for (column, label, linewidth), color in zip(OPACITYSERIES, colors, strict=True):
            ax.plot(
                dfmovingaverages["lambda_angstroms_bin_mid"],
                dfmovingaverages[column],
                linewidth=linewidth,
                color=color,
                label=label,
            )

    ax.set_xlabel(r"Wavelength ($\mathrm{\AA}$)")
    ax.set_ylabel(r"Opacity [cm$^2$/g]")
    set_auto_yscale(ax, args)
    set_axis_properties(ax, args)
    set_plot_title(ax, title, args)
    set_legend(ax, args)

    save_figure(fig, args.outputfile, args=args)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path(), helptext="Path of the ARTIS model")
    addarg_timestep(parser, helptext="Timestep to plot, e.g. 40 or last")
    addarg_timedays(parser, kind="str")
    addarg_modelgridindex(
        parser,
        helptext="Cell to plot. If you do not give a cell, the plot shows the mass-weighted mean over all the cells",
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
    parser.add_argument("-deltalambda", type=float, default=20.0, help="Wavelength bin width in Angstroms")
    parser.add_argument(
        "-movingaveragewidth",
        type=float,
        default=200.0,
        help=(
            "Width in Angstroms of the window of the moving average. The window holds the nearest odd number of bins."
            " Give 0 for no moving average"
        ),
    )

    addarg_yscale(parser, default="log")
    addarg_notitle(parser)
    addarg_nolegend(parser)
    addarg_figscale(parser)
    addarg_show(parser)
    addarg_output(parser, kind="file", defaultname="plotopacity.pdf", helptext="Path/filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the expansion opacity and the line-binned opacities against wavelength."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    time_days = get_timestep_time(args.modelpath, timestep)
    modelgridindex = get_single_modelgridindex(args.modelgridindex)

    dfopacities = get_massweighted_opacities(
        adata=get_opacity_atomic_data(args.modelpath),
        time_days=time_days,
        dfestimators=get_cell_estimators(args.modelpath, timestep, modelgridindex),
        lambdamin=args.xmin,
        lambdamax=args.xmax,
        deltalambda=args.deltalambda,
    )

    cellstr = "mass-weighted mean of all cells" if modelgridindex is None else f"cell {modelgridindex}"
    title = f"{get_model_name(args.modelpath)} at {time_days:.1f}d (timestep {timestep}), {cellstr}"
    # an odd number of bins puts the centre of the window at the middle of a bin
    windowbins = 2 * round(args.movingaveragewidth / args.deltalambda / 2) + 1
    dfmovingaverages = get_moving_averages(dfopacities, windowbins) if args.movingaveragewidth > 0.0 else None
    plot_opacities(dfopacities, dfmovingaverages, title, args)
