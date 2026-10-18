"""Plot the Sobolev emission and the Sobolev optical depth of the strongest lines in a range, against velocity."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from artistools.ejectaopacity import addarg_excitationtemperature
from artistools.ejectaopacity import get_selected_timestep
from artistools.linecontributions import get_line_velocity_profiles
from artistools.linecontributions import get_velocity_bin_edges_kmps
from artistools.linecontributions import LINEKEYS
from artistools.linecontributions import parse_ions
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import exit_with_error
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_timestep_time
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import resolve_outputfile
from artistools.misc.cliutils import CommaJoinAction
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_axis_properties
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title

if t.TYPE_CHECKING:
    import matplotlib.axes as mplax
    import numpy.typing as npt

DEFAULTOUTPUTNAME = "plotlinecontributions.pdf"

# a line with a tau_sobolev far below 1 has no effect on the spectrum. A log axis down to such values makes the
# range near 1 too small to read
TAUAXISMIN = 1e-3


def get_line_label() -> pl.Expr:
    """Return the label of a line: the ion, the vacuum wavelength, and the upper and the lower level."""
    return pl.format(
        "{} λ{} {}-{}",
        pl.col("ion"),
        pl.col("lambda_angstroms").round(0).cast(pl.Int64),
        pl.col("upper"),
        pl.col("lower"),
    )


def get_ranked_lines(dfsummary: pl.DataFrame, rankby: t.Literal["emission", "tau"], maxlines: int) -> pl.DataFrame:
    """Return the summary rows of the maxlines lines with the most emission or the largest tau_sobolev in a cell."""
    sortcolumn = "emission_erg_per_s" if rankby == "emission" else "tau_sobolev_max"
    return (
        dfsummary
        .sort(sortcolumn, "lambda_angstroms", descending=[True, False], nulls_last=True)
        .head(maxlines)
        .with_columns(label=get_line_label())
    )


def plot_line_profiles(
    axes: "npt.NDArray[np.object_] | Sequence[mplax.Axes]",
    dfsummary: pl.DataFrame,
    dfprofiles: pl.DataFrame,
    dfranked: pl.DataFrame,
    velocityedges_kmps: Sequence[float],
) -> pl.DataFrame:
    """Draw the emission profile and the mean tau_sobolev of each ranked line, and return the drawn data.

    The emission profile of a line is its emission per unit velocity, divided by the emission of the line in all the
    cells. Thus a weak line and a strong line have profiles of the same size, and the legend gives the strength. The
    emission panel also draws the profile of the sum of all the lines in the wavelength range.
    """
    edges = np.array(velocityedges_kmps)
    nbins = len(edges) - 1

    def get_bin_values(dfline: pl.DataFrame, column: str, emptyvalue: float) -> "npt.NDArray[np.float64]":
        values = np.full(nbins, emptyvalue)
        values[dfline["velocitybin"].to_numpy()] = dfline[column].fill_null(emptyvalue).to_numpy()
        return values

    profilecolumn = "emission_fraction_per_kmps"
    alllinesemission = float(dfsummary["emission_erg_per_s"].sum())
    dfalllines = dfprofiles.group_by("velocitybin").agg(
        (pl.col("emission_per_velocity_erg_per_s_per_kmps").sum() / alllinesemission).alias(profilecolumn)
    )
    axes[0].stairs(
        get_bin_values(dfalllines, profilecolumn, 0.0),
        edges,
        label="All lines in the range",
        color="black",
        linewidth=1.5,
    )

    dfdrawn = dfprofiles.join(
        dfranked.select(*LINEKEYS, "label", lineemission="emission_erg_per_s"), on=list(LINEKEYS), how="inner"
    ).with_columns(
        pl
        .when(pl.col("lineemission") > 0.0)
        .then(pl.col("emission_per_velocity_erg_per_s_per_kmps") / pl.col("lineemission"))
        .alias(profilecolumn)
    )
    drawntau = []
    for label, emissionfraction in dfranked.select("label", "emission_fraction").iter_rows():
        dfline = dfdrawn.filter(pl.col("label") == label)
        stairs = axes[0].stairs(
            get_bin_values(dfline, profilecolumn, 0.0), edges, label=f"{label} ({emissionfraction:.1%})"
        )
        tauvalues = get_bin_values(dfline, "tau_sobolev_volume_mean", np.nan)
        # a log axis cannot show a population inversion, i.e. a negative tau_sobolev
        tauvalues = np.where(tauvalues > 0.0, tauvalues, np.nan)
        axes[1].stairs(tauvalues, edges, baseline=None, color=stairs.get_edgecolor())
        drawntau.append(tauvalues)

    axes[1].axhline(1.0, color="grey", linestyle="dashed", linewidth=1.0)
    axes[1].set_yscale("log")
    if drawntau and np.nanmin(np.concatenate(drawntau), initial=np.inf) < TAUAXISMIN:
        axes[1].set_ylim(bottom=TAUAXISMIN)
        print_detail(f"The lower panel shows only a tau_sobolev of {TAUAXISMIN:g} or more")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xlim(edges[0], edges[-1])

    return dfdrawn.select(
        "label",
        *LINEKEYS,
        "x_lower_kmps",
        "x_upper_kmps",
        "emission_erg_per_s",
        "emission_per_velocity_erg_per_s_per_kmps",
        profilecolumn,
        "tau_sobolev_volume_mean",
        "cellcount",
    ).sort("label", "x_lower_kmps")


def print_ranked_lines(dfranked: pl.DataFrame, rankby: str) -> None:
    """Print the emission fraction, the mean velocity of the emission, and the maximum tau_sobolev of each line."""
    rankedby = "the most emission" if rankby == "emission" else "the largest tau_sobolev in a cell"
    print_detail(
        f"These are the {dfranked.height} lines with {rankedby}. The mean velocity takes the emission of each cell"
        " as the weight:"
    )
    for row in dfranked.iter_rows(named=True):
        meanvelocity = row["velocity_kmps_emission_mean"]
        emissiontext = (
            f"{row['emission_fraction']:.1%} of the emission of the lines, mean velocity {meanvelocity:.0f} km/s"
            if meanvelocity is not None
            else "no emission"
        )
        taumax = row["tau_sobolev_max"]
        tautext = (
            f"maximum tau_sobolev {taumax:.3g} at {row['velocity_kmps_at_tau_sobolev_max']:.0f} km/s,"
            f" {row['cellcount_tau_sobolev_above_1']} cells with tau_sobolev above 1"
            if taumax is not None
            else "no tau_sobolev"
        )
        print_detail(f"  {row['label']}: {emissiontext}, {tautext}")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path(), helptext="Path of ARTIS model")
    addarg_timestep(parser, helptext="Timestep number to select")
    addarg_timedays(parser, kind="str", helptext="Time in days to select")
    parser.add_argument("-xmin", "-lambdamin", type=float, help="Minimum wavelength of the lines in Angstroms")
    parser.add_argument("-xmax", "-lambdamax", type=float, help="Maximum wavelength of the lines in Angstroms")
    parser.add_argument(
        "-ions",
        action=CommaJoinAction,
        help='Ions of the lines, e.g. "Fe II,Co II". Default: each ion with a population in the model',
    )
    parser.add_argument("-maxlines", type=int, default=8, help="Number of lines to plot")
    parser.add_argument(
        "-rankby",
        choices=["emission", "tau"],
        default="emission",
        help=(
            "Plot the lines with the most emission, or the lines with the largest tau_sobolev in a cell. Use tau to"
            " find the lines of an absorption feature"
        ),
    )
    parser.add_argument("-nvelocitybins", type=int, default=25, help="Number of velocity bins of equal width")
    parser.add_argument(
        "-velocitymax", type=float, help="Upper edge of the velocity bins in km/s. Default: the vmax of the model"
    )
    addarg_excitationtemperature(parser)
    addarg_output(parser, kind="file", defaultname=DEFAULTOUTPUTNAME, helptext="Path/filename for the plot file")
    addarg_figscale(parser)
    addarg_data_output(parser)
    addarg_show(parser)
    addarg_darkmode(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the Sobolev emission and the Sobolev optical depth of the strongest lines against velocity."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    if args.xmin is None or args.xmax is None:
        exit_with_error("give the wavelength range of the lines", "e.g. -lambdamin 8400 -lambdamax 8700")
    if args.nvelocitybins < 1:
        exit_with_error(f"-nvelocitybins {args.nvelocitybins} gives no bins", "Give a number of 1 or more")
    if args.velocitymax is not None and args.velocitymax <= 0.0:
        exit_with_error(f"-velocitymax {args.velocitymax} gives no range", "Give a velocity above 0 km/s")

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    print(f"Line contributions of {get_model_logname(args.modelpath)} at timestep {timestep}")
    print_modelpath(args.modelpath)

    velocityedges_kmps = get_velocity_bin_edges_kmps(args.modelpath, args.nvelocitybins, args.velocitymax)
    ions = parse_ions(str(args.ions).split(","), "-ions") if args.ions else None
    dfsummary, dfprofiles = get_line_velocity_profiles(
        args.modelpath, timestep, args.xmin, args.xmax, velocityedges_kmps, ions, args.exctemperature
    )

    upperedgesource = "-velocitymax" if args.velocitymax is not None else "the vmax of the model"
    print_detail(
        f"{args.nvelocitybins} velocity bins of {velocityedges_kmps[1]:.0f} km/s from 0 to {velocityedges_kmps[-1]:.0f}"
        f" km/s ({upperedgesource}). Each cell goes in the bin of the radial velocity of its centre"
    )
    print_detail(
        "The upper panel gives the emission of the cells in each bin, divided by the width of the bin. The panel also"
        " divides it by the emission of the line in all the cells. The legend gives the fraction of the emission of"
        " all the lines. The lower panel gives the mean tau_sobolev of the cells in each bin, with the volume of each"
        " cell as the weight"
    )
    totalemission = float(dfsummary["emission_erg_per_s"].sum())
    if totalemission > 0.0:
        binnedfraction = float(dfprofiles["emission_erg_per_s"].sum()) / totalemission
        if binnedfraction < 1.0 - 1e-6:
            print_detail(
                f"The bins hold {binnedfraction:.1%} of the emission of the lines. The rest is outside the bins"
            )

    dfranked = get_ranked_lines(dfsummary, args.rankby, args.maxlines)
    print_ranked_lines(dfranked, args.rankby)

    fig, axesgrid = make_frame_figure(args, rows=2, aspect=0.45)
    axes = axesgrid[:, 0]
    dfdrawn = plot_line_profiles(axes, dfsummary, dfprofiles, dfranked, velocityedges_kmps)
    # -xmin and -xmax give the wavelength range of the lines, thus they do not set the limits of the velocity axis
    set_axis_properties(axes[0], args, xlimits=(None, None, "-xmin"), setyaxis=False)
    axes[1].set_xlabel("Radial velocity [km/s]")
    axes[0].set_ylabel(r"$\Delta L / \Delta v / L$ [(km/s)$^{-1}$]")
    axes[1].set_ylabel(r"$\langle\tau_\mathrm{Sobolev}\rangle_V$")
    set_legend(axes[0], args)
    set_plot_title(
        axes[0], f"{get_model_name(args.modelpath)} {get_timestep_time(args.modelpath, timestep):.2f} d", args
    )

    outputfile = resolve_outputfile(args.outputfile, DEFAULTOUTPUTNAME)
    output_data(args, {"": dfdrawn}, outputfile)
    save_figure(fig, outputfile, args=args)
