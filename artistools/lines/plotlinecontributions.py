"""Plot the Sobolev emission and the Sobolev optical depth of the strongest lines in a range, against velocity."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.colors as mplcolors
import numpy as np
import polars as pl

from artistools.inputmodel import get_modeldata
from artistools.lines.core import addarg_excitationtemperature
from artistools.lines.linecontributions import addarg_lineselection
from artistools.lines.linecontributions import addarg_taumin
from artistools.lines.linecontributions import get_cell_line_inputs
from artistools.lines.linecontributions import get_line_velocity_profiles
from artistools.lines.linecontributions import get_ranked_line_summary
from artistools.lines.linecontributions import get_sobolev_trough
from artistools.lines.linecontributions import LINEKEYS
from artistools.lines.linecontributions import parse_ions
from artistools.lines.linecontributions import TROUGHSAMPLESPERCELL
from artistools.misc import addarg_averageoverphi
from artistools.misc import addarg_costhetarange
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import check_dirbin
from artistools.misc import exit_with_error
from artistools.misc import format_costheta_range
from artistools.misc import get_artis_option
from artistools.misc import get_costheta_range
from artistools.misc import get_dirbin_labels
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_selected_timestep
from artistools.misc import get_timestep_time
from artistools.misc import get_timestep_times
from artistools.misc import get_weighted_quantile_expr
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import positive_int_arg
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import print_warning
from artistools.plottools import addarg_velocitybins
from artistools.plottools import convert_velocity_columns
from artistools.plottools import format_velocity
from artistools.plottools import get_cell_widths_kmps
from artistools.plottools import get_grid_corner_velocity_kmps
from artistools.plottools import get_velocity_bin_edges_kmps
from artistools.plottools import get_velocity_unit
from artistools.plottools import INVERSE_VELOCITYUNIT_AXISLABELS
from artistools.plottools import KMPS_PER_VELOCITYUNIT
from artistools.plottools import make_frame_figure
from artistools.plottools import positive_velocity_kmps_arg
from artistools.plottools import save_figure
from artistools.plottools import set_axis_properties
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title
from artistools.plottools import VELOCITYUNIT_AXISLABELS

if t.TYPE_CHECKING:
    import matplotlib.axes as mplax
    import numpy.typing as npt

    from artistools.lines.linecontributions import CellLineInputs
    from artistools.lines.linecontributions import LineVelocityProfiles

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


def plot_line_profiles(
    axes: "npt.NDArray[np.object_] | Sequence[mplax.Axes]",
    profiles: "LineVelocityProfiles",
    dfranked: pl.DataFrame,
    velocityedges_kmps: Sequence[float],
    unit: t.Literal["kmps", "c"],
    alllineslabel: str,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Draw the emission profile and the median tau_sobolev of each ranked line, and return the data and the colours.

    The emission profile of a line is its emission per unit velocity, divided by the emission of the line in all the
    cells. Thus a weak line and a strong line have profiles of the same size, and the legend gives the strength. The
    emission panel also draws the profile of the sum of the lines with the label alllineslabel. The median of
    tau_sobolev takes the volume of each cell as the weight. A few dense cells can make the mean much larger than the
    median, thus the table gives both. The axis and the drawn data take the velocity unit unit.
    """
    kmpsperunit = KMPS_PER_VELOCITYUNIT[unit]
    edges = np.array(velocityedges_kmps) / kmpsperunit
    nbins = len(edges) - 1

    def get_bin_values(dfline: pl.DataFrame, column: str, emptyvalue: float) -> "npt.NDArray[np.float64]":
        values = np.full(nbins, emptyvalue)
        values[dfline["velocitybin"].to_numpy()] = dfline[column].fill_null(emptyvalue).to_numpy()
        return values

    profilecolumn = "emission_fraction_per_kmps"
    binwidth_kmps = velocityedges_kmps[1] - velocityedges_kmps[0]
    alllinesemission = float(profiles.dfsummary["emission_erg_per_s"].sum())
    dfalllines = profiles.dfalllines.with_columns(
        emission_per_velocity_erg_per_s_per_kmps=pl.col("emission_erg_per_s") / binwidth_kmps,
        x_lower_kmps=pl.lit(pl.Series(velocityedges_kmps[:-1])).gather(pl.col("velocitybin")),
        x_upper_kmps=pl.lit(pl.Series(velocityedges_kmps[1:])).gather(pl.col("velocitybin")),
    ).with_columns(
        pl
        .when(pl.lit(alllinesemission > 0.0))
        .then(pl.col("emission_per_velocity_erg_per_s_per_kmps") / alllinesemission)
        .otherwise(0.0)
        .alias(profilecolumn)
    )
    axes[0].stairs(
        get_bin_values(dfalllines, profilecolumn, 0.0) * kmpsperunit,
        edges,
        label=alllineslabel,
        color="black",
        linewidth=1.5,
    )

    dfdrawn = profiles.dfprofiles.join(
        dfranked.select(*LINEKEYS, "label", lineemission="emission_erg_per_s"), on=list(LINEKEYS), how="inner"
    ).with_columns(
        pl
        .when(pl.col("lineemission") > 0.0)
        .then(pl.col("emission_per_velocity_erg_per_s_per_kmps") / pl.col("lineemission"))
        .alias(profilecolumn)
    )
    colours: dict[str, str] = {}
    drawntau = []
    for label, emissionfraction in dfranked.select("label", "emission_fraction").iter_rows():
        dfline = dfdrawn.filter(pl.col("label") == label)
        fractiontext = "no emission" if emissionfraction is None else f"{emissionfraction:.1%}"
        stairs = axes[0].stairs(
            get_bin_values(dfline, profilecolumn, 0.0) * kmpsperunit, edges, label=f"{label} ({fractiontext})"
        )
        colours[label] = mplcolors.to_hex(stairs.get_edgecolor())
        tauvalues = get_bin_values(dfline, "tau_sobolev_volume_median", np.nan)
        # a log axis cannot show a population inversion, i.e. a negative tau_sobolev
        tauvalues = np.where(tauvalues > 0.0, tauvalues, np.nan)
        axes[1].stairs(tauvalues, edges, baseline=None, color=colours[label])
        drawntau.append(tauvalues)

    axes[1].axhline(1.0, color="grey", linestyle="dashed", linewidth=1.0)
    axes[1].set_yscale("log")
    if drawntau and np.nanmin(np.concatenate(drawntau), initial=np.inf) < TAUAXISMIN:
        axes[1].set_ylim(bottom=TAUAXISMIN)
        print_detail(f"The tau_sobolev panel shows only a median tau_sobolev of {TAUAXISMIN:g} or more")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xlim(edges[0], edges[-1])

    linecolumns = [
        "label",
        *LINEKEYS,
        "x_lower_kmps",
        "x_upper_kmps",
        "emission_erg_per_s",
        "emission_per_velocity_erg_per_s_per_kmps",
        profilecolumn,
        "tau_sobolev_volume_median",
        "tau_sobolev_volume_mean",
        "volume_fraction_tau_sobolev_above_1",
        "cellcount",
    ]
    # the rows of the black curve have no line, and the profile of each line comes after them
    dftable = pl.concat(
        [
            dfalllines.select(pl.lit(alllineslabel).alias("label"), pl.exclude("velocitybin")).sort("x_lower_kmps"),
            dfdrawn.select(linecolumns).sort("label", "x_lower_kmps"),
        ],
        how="diagonal_relaxed",
    ).select(linecolumns)
    return convert_velocity_columns(dftable, unit), colours


def plot_trough(
    ax: "mplax.Axes",
    dfdepths: pl.DataFrame,
    dfranked: pl.DataFrame,
    colours: dict[str, str],
    unit: t.Literal["kmps", "c"],
    photospherevelocity_kmps: float,
) -> None:
    """Draw the Sobolev absorption depth of each ranked line against the line-of-sight velocity."""
    kmpsperunit = KMPS_PER_VELOCITYUNIT[unit]
    dflabelled = dfdepths.join(dfranked.select(*LINEKEYS, "label"), on=list(LINEKEYS), how="inner")
    for label in dfranked["label"]:
        dfline = dflabelled.filter(pl.col("label") == label).sort("x_lower_kmps")
        edges = np.append(dfline["x_lower_kmps"].to_numpy(), dfline["x_upper_kmps"][-1]) / kmpsperunit
        ax.stairs(dfline["trough_depth"].to_numpy(), edges, color=colours[label])
    ax.axvline(photospherevelocity_kmps / kmpsperunit, color="grey", linestyle="dotted", linewidth=1.0)
    ax.set_ylim(0.0, 1.0)


def plot_trough_estimate(
    ax: "mplax.Axes",
    profiles: "LineVelocityProfiles",
    inputs: "CellLineInputs",
    dfranked: pl.DataFrame,
    colours: dict[str, str],
    troughsettings: tuple[int, float],
    velocityedges_kmps: Sequence[float],
    unit: t.Literal["kmps", "c"],
    modelpath: Path,
    *,
    averageoverphi: bool,
) -> dict[str, pl.DataFrame]:
    """Draw the Sobolev estimate of the trough of the ranked lines, log its method, and return its tables."""
    dirbin, photospherevelocity_kmps = troughsettings
    dopplershift = get_artis_option(modelpath, "USE_RELATIVISTIC_DOPPLER_SHIFT")
    relativisticdoppler = dopplershift == "true"
    _, modelmeta = get_modeldata(modelpath, printwarningsonly=True)
    dfdepths, dfblend = get_sobolev_trough(
        profiles,
        inputs,
        dirbin,
        photospherevelocity_kmps,
        velocityedges_kmps,
        get_cell_widths_kmps(modelmeta),
        relativisticdoppler=relativisticdoppler,
        averageoverphi=averageoverphi,
    )
    plot_trough(ax, dfdepths, dfranked, colours, unit, photospherevelocity_kmps)
    print_trough_method(
        dirbin,
        photospherevelocity_kmps,
        unit,
        dfranked.height,
        dfblend,
        dopplershift=dopplershift,
        averageoverphi=averageoverphi,
    )
    # the gas in front of the photosphere moves towards the observer, thus its light arrives at about t (1 - v_ph / c)
    arrival_days = inputs.time_days * (1.0 - photospherevelocity_kmps / KMPS_PER_VELOCITYUNIT["c"])
    print_detail(
        "To test the estimate, compare it with the absorption of each line in the packet spectrum of the same"
        f" direction: artistools plotspectra {modelpath} --frompackets --dirbin {dirbin}"
        f"{' --average-over-phi-angle' if averageoverphi else ''}"
        f" --showabsorption --groupby line --timedays {arrival_days:.3g}. The light of the gas in front of the"
        f" photosphere arrives at about t (1 - v_ph / c) = {arrival_days:.3g} d"
    )
    ax.set_ylabel("Trough depth")
    return {
        "_trough": convert_velocity_columns(
            dfdepths.join(dfranked.select(*LINEKEYS, "label"), on=list(LINEKEYS), how="left").select(
                "label", pl.exclude("label")
            ),
            unit,
        ),
        "_troughblend": dfblend,
    }


def print_arrival_times(
    modelpath: Path, profiles: "LineVelocityProfiles", time_days: float, unit: t.Literal["kmps", "c"]
) -> None:
    """Print the arrival times of the light of the ranked lines, and the timesteps of the packet commands that hold it.

    The median takes the emission of each cell and ranked line as the weight. The cells at the radial velocity v make
    a shell. A distant observer sees the light of that shell from t (1 - v / c) to t (1 + v / c).
    """
    median_kmps = profiles.dfrankedcells.select(
        get_weighted_quantile_expr("velocity_kmps", "emission_erg_per_s", 0.5)
    ).item()
    if median_kmps is None:
        return
    beta = median_kmps / KMPS_PER_VELOCITYUNIT["c"]
    arrivalrange_days = (time_days * (1.0 - beta), time_days * (1.0 + beta))
    tstarts = get_timestep_times(modelpath, loc="start")
    tends = get_timestep_times(modelpath, loc="end")

    def get_timestep_text(arrival_days: float) -> str:
        # the packet commands keep only the arrival times inside the timesteps
        if arrival_days < tstarts[0]:
            return f"no timestep, because it is before the start of the first timestep at {tstarts[0]:.3g} d"
        if arrival_days >= tends[-1]:
            return f"no timestep, because it is after the end of the last timestep at {tends[-1]:.3g} d"
        return f"timestep {int(np.searchsorted(tstarts, arrival_days, side='right') - 1)}"

    print_detail(
        "Half of the emission of the ranked lines comes from cells below the radial velocity"
        f" {format_velocity(median_kmps, unit)}. A distant observer sees the light of the cells at that velocity from"
        f" {arrivalrange_days[0]:.3g} to {arrivalrange_days[1]:.3g} d, i.e. t (1 - v / c) to t (1 + v / c). The"
        f" packet commands put the first time in {get_timestep_text(arrivalrange_days[0])}. They put the last time in"
        f" {get_timestep_text(arrivalrange_days[1])}"
    )


def print_ranked_lines(dflinesummary: pl.DataFrame, rankby: str, unit: t.Literal["kmps", "c"], lineset: str) -> None:
    """Print the emission fraction, the velocity quartiles of the emission, and the range of a median tau above 1.

    lineset names the lines of the ranking and of the fractions, e.g. "all the lines".
    """
    rankedby = "the most emission" if rankby == "emission" else "the largest tau_sobolev in a cell"
    print_detail(
        f"These are the {dflinesummary.height} lines with {rankedby}, of {lineset}. The quartiles of the velocity"
        " take the emission of each cell as the weight:"
    )
    for row in dflinesummary.iter_rows(named=True):
        quartiles = [row[f"velocity_emission_q{quantile}_kmps"] for quantile in (25, 50, 75)]
        emissiontext = (
            f"{row['emission_fraction']:.1%} of the emission of {lineset}, velocity quartiles"
            f" {' '.join(format_velocity(value, unit) for value in quartiles)}"
            if None not in quartiles and row["emission_fraction"] is not None
            else "no emission"
        )
        tauabove1min_kmps = row["velocity_median_tau_above_1_min_kmps"]
        tauabove1max_kmps = row["velocity_median_tau_above_1_max_kmps"]
        taurangetext = (
            f"median tau_sobolev above 1 from {format_velocity(tauabove1min_kmps, unit)} to"
            f" {format_velocity(tauabove1max_kmps, unit)}"
            if tauabove1min_kmps is not None
            else "median tau_sobolev of 1 or less in each bin"
        )
        tautext = (
            f"maximum tau_sobolev {row['tau_sobolev_max']:.3g} at"
            f" {format_velocity(row['velocity_kmps_at_tau_sobolev_max'], unit)}"
            if row["tau_sobolev_max"] is not None
            else "no tau_sobolev"
        )
        print_detail(f"  {row['label']}: {emissiontext}, {taurangetext}, {tautext}")


def get_trough_settings(args: argparse.Namespace, unit: t.Literal["kmps", "c"]) -> tuple[int, float] | None:
    """Return the direction bin and the velocity of the photosphere [km/s] for the trough estimate, or None.

    The function checks the options before the command reads the estimators.
    """
    if args.troughdirbin is None:
        if args.photospherevelocity_kmps is not None:
            exit_with_error("--photospherevelocity applies only to the trough estimate", "Give --troughdirbin too")
        return None
    check_dirbin(args.troughdirbin, "-troughdirbin", averageoverphi=args.average_over_phi_angle, allowall=False)
    _, modelmeta = get_modeldata(args.modelpath, printwarningsonly=True)
    if modelmeta["dimensions"] != 3:
        # a cell of a 2D model is a ring, and a cell of a 1D model is a shell
        exit_with_error(
            f"The trough estimate needs the velocity vector of each cell, and a {modelmeta['dimensions']}D model has no"
            " such vector",
            "Remove --troughdirbin",
        )
    photospherevelocity_kmps = args.photospherevelocity_kmps
    if photospherevelocity_kmps is None:
        exit_with_error(
            "The trough estimate needs the velocity of the photosphere",
            "Give --photospherevelocity, e.g. the median of plotlastinteractionvelocity for a range with no strong"
            " line",
        )
    if photospherevelocity_kmps >= KMPS_PER_VELOCITYUNIT["c"]:
        exit_with_error(
            f"--photospherevelocity {format_velocity(photospherevelocity_kmps, unit)} is at or above the speed of"
            " light",
            "Give a velocity below 1c",
        )
    partwidth_kmps = min(get_cell_widths_kmps(modelmeta)) / TROUGHSAMPLESPERCELL
    if photospherevelocity_kmps < partwidth_kmps:
        exit_with_error(
            f"--photospherevelocity {format_velocity(photospherevelocity_kmps, unit)} is less than"
            f" {format_velocity(partwidth_kmps, unit)}, a third of the width of a cell, thus the trough estimate has no"
            " resolution",
            "Give a larger --photospherevelocity",
        )
    cornervelocity_kmps = get_grid_corner_velocity_kmps(modelmeta)
    if photospherevelocity_kmps >= cornervelocity_kmps:
        exit_with_error(
            f"--photospherevelocity {format_velocity(photospherevelocity_kmps, unit)} is at or above the corner of the"
            f" grid at {format_velocity(cornervelocity_kmps, unit)}, thus no cell is in front of the photosphere",
            "Give a lower --photospherevelocity",
        )
    if args.dirbin is not None or args.costhetamin is not None or args.costhetamax is not None:
        exit_with_error(
            "The trough estimate needs all the cells in front of the photosphere",
            "Remove --dirbin, --costhetamin, and --costhetamax, or remove --troughdirbin",
        )
    return args.troughdirbin, photospherevelocity_kmps


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_lineselection(parser)
    parser.add_argument("-maxlines", type=positive_int_arg, default=8, help="The number of lines to plot")
    parser.add_argument(
        "-rankby",
        choices=["emission", "tau"],
        default="emission",
        help=(
            "Plot the lines with the most emission, or the lines with the largest tau_sobolev in a cell. Use tau to"
            " find the lines of an absorption feature"
        ),
    )
    addarg_velocitybins(parser)
    addarg_costhetarange(parser)
    addarg_taumin(parser)
    parser.add_argument(
        "-troughdirbin",
        type=int,
        help=(
            "Add a Sobolev estimate of the absorption trough for an observer at the centre of this direction bin. The"
            " bins are the same as in plotspectra. With --average-over-phi-angle, give the first bin of a costheta"
            " bin, e.g. 90 for the pole at +z"
        ),
    )
    addarg_averageoverphi(parser, "-troughdirbin")
    parser.add_argument(
        "-photospherevelocity",
        dest="photospherevelocity_kmps",
        type=positive_velocity_kmps_arg,
        help=(
            "Velocity of the photosphere of --troughdirbin, with its unit, e.g. 0.157c or 47000km/s."
            " plotlastinteractionvelocity gives an estimate"
        ),
    )
    addarg_excitationtemperature(parser)
    addarg_output(parser, kind="file", defaultname=DEFAULTOUTPUTNAME, helptext="Path/filename for the plot file")
    addarg_figscale(parser)
    addarg_data_output(parser)
    addarg_show(parser)
    addarg_darkmode(parser)


def print_profile_method(
    profiles: "LineVelocityProfiles", velocityedges_kmps: Sequence[float], unit: t.Literal["kmps", "c"], lineset: str
) -> None:
    """Print the velocity bins, the quantities of the two panels, and the part of the emission in the bins.

    lineset names the lines of the fractions of the emission, e.g. "all the lines".
    """
    print_detail(
        f"{len(velocityedges_kmps) - 1} velocity bins of {format_velocity(velocityedges_kmps[1], unit)} from 0 to"
        f" {format_velocity(velocityedges_kmps[-1], unit)}. Each cell goes in the bin of the radial velocity of its"
        " centre"
    )
    print_detail(
        "The emission panel gives the emission of the cells in each bin, divided by the width of the bin. It shows"
        " where each line emits. It does not show where the part of the line in the observed luminosity comes from."
        " The panel"
        " also divides it by the emission of the line in all the cells. The legend gives the fraction of the emission"
        f" of {lineset}. The tau_sobolev panel gives the median tau_sobolev of the cells in each bin, with the"
        " volume of each cell as the weight"
    )
    totalemission = float(profiles.dfsummary["emission_erg_per_s"].sum())
    if totalemission > 0.0:
        binnedfraction = float(profiles.dfalllines["emission_erg_per_s"].sum()) / totalemission
        if binnedfraction < 1.0 - 1e-6:
            print_detail(
                f"The bins hold {binnedfraction:.1%} of the emission of the lines. The rest is outside the bins"
            )
    else:
        print_warning(
            "The lines have no emission in the selected cells, thus the emission panel is empty. The cells can have"
            " no population of the upper levels"
        )


def label_line_axes(
    axes: "npt.NDArray[np.object_]",
    args: argparse.Namespace,
    unit: t.Literal["kmps", "c"],
    title: str,
    *,
    hastrough: bool,
) -> None:
    """Set the axis properties, the labels, the legend, and the title of the panels."""
    # -xmin and -xmax give the wavelength range of the lines, thus they do not set the limits of the velocity axis
    set_axis_properties(axes[0], args, xlimits=(None, None, "-xmin"), setyaxis=False)
    unitlabel = VELOCITYUNIT_AXISLABELS[unit]
    axes[-1].set_xlabel(
        f"Velocity [{unitlabel}]: radial, and line of sight in the trough panel"
        if hastrough
        else f"Radial velocity [{unitlabel}]"
    )
    axes[0].set_ylabel(rf"$\Delta L / \Delta v / L$ [{INVERSE_VELOCITYUNIT_AXISLABELS[unit]}]")
    axes[1].set_ylabel(r"median $\tau_\mathrm{Sobolev}$")
    set_legend(axes[0], args)
    set_plot_title(axes[0], title, args)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the Sobolev emission and the Sobolev optical depth of the strongest lines against velocity."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    if args.xmin is None or args.xmax is None:
        exit_with_error("Give the wavelength range of the lines", "e.g. --lambdamin 8400 --lambdamax 8700")
    if args.maxlines < 1:
        exit_with_error(f"--maxlines {args.maxlines} gives no lines", "Give a number of 1 or more")

    timestep = get_selected_timestep(args.modelpath, args.timestep, args.timedays)
    print(f"Line contributions of {get_model_logname(args.modelpath)} at timestep {timestep}")
    print_modelpath(args.modelpath)

    unit = get_velocity_unit(args.modelpath, args.velocityunit)
    troughsettings = get_trough_settings(args, unit)
    velocityedges_kmps = get_velocity_bin_edges_kmps(args.modelpath, args.nvelocitybins, args.velocitymax_kmps, unit)
    ions = parse_ions(str(args.ions).split(","), "-ions") if args.ions else None
    costhetarange = get_costheta_range(args)
    inputs = get_cell_line_inputs(
        args.modelpath, timestep, args.xmin, args.xmax, ions, args.exctemperature, costhetarange, args.taumin
    )
    profiles = get_line_velocity_profiles(inputs, velocityedges_kmps, args.rankby, args.maxlines)

    # -taumin leaves out each line that cannot reach it, thus the sums of the emission take only the other lines
    lineset = "all the lines" if args.taumin is None else "the lines that --taumin keeps"
    if args.taumin is not None and args.rankby == "emission":
        print_warning(
            "--taumin can leave out an optically thin line, and such a line can emit much. Thus the ranking by"
            " emission and the fractions of the emission take only the lines that --taumin keeps. Give --rankby tau,"
            " or --ions in place of --taumin"
        )
    print_profile_method(profiles, velocityedges_kmps, unit, lineset)

    dfranked = profiles.dfranked.with_columns(label=get_line_label())
    dflinesummary = get_ranked_line_summary(profiles).with_columns(label=get_line_label())
    print_ranked_lines(dflinesummary, args.rankby, unit, lineset)
    print_arrival_times(args.modelpath, profiles, inputs.time_days, unit)

    fig, axesgrid = make_frame_figure(args, rows=3 if troughsettings else 2, aspect=0.45)
    axes = axesgrid[:, 0]
    alllineslabel = "All lines in the range" if args.taumin is None else f"Lines that --taumin {args.taumin:g} keeps"
    dfdrawn, colours = plot_line_profiles(axes, profiles, dfranked, velocityedges_kmps, unit, alllineslabel)
    # the empty tables replace the trough files of the same name from an earlier command
    tables = {
        "": dfdrawn,
        "_summary": convert_velocity_columns(dflinesummary.select("label", pl.exclude("label")), unit),
        "_trough": pl.DataFrame(),
        "_troughblend": pl.DataFrame(),
    }
    if troughsettings is not None:
        tables |= plot_trough_estimate(
            axes[2],
            profiles,
            inputs,
            dfranked,
            colours,
            troughsettings,
            velocityedges_kmps,
            unit,
            args.modelpath,
            averageoverphi=args.average_over_phi_angle,
        )

    title = f"{get_model_name(args.modelpath)} {get_timestep_time(args.modelpath, timestep):.2f} d"
    if costhetarange is not None:
        title += f", {format_costheta_range(costhetarange)}"
    label_line_axes(axes, args, unit, title, hastrough=troughsettings is not None)

    outputfile = Path(args.outputfile)
    output_data(args, tables, outputfile)
    save_figure(fig, outputfile, args=args)


def print_trough_method(
    dirbin: int,
    photospherevelocity_kmps: float,
    unit: t.Literal["kmps", "c"],
    linecount: int,
    dfblend: pl.DataFrame,
    *,
    dopplershift: str | None,
    averageoverphi: bool,
) -> None:
    """Print the method and the assumptions of the trough estimate, and the deepest point of the blend.

    dopplershift is the value of USE_RELATIVISTIC_DOPPLER_SHIFT in artisoptions.h, or None with no such option.
    """
    directionlabel = get_dirbin_labels([dirbin], average_over_phi=averageoverphi)[dirbin]
    observertext = (
        f"at the centre cos θ of direction bin {dirbin} ({directionlabel}), as the mean over the phi bins"
        if averageoverphi
        else f"at the centre of direction bin {dirbin} ({directionlabel})"
    )
    print_detail(
        f"The trough panel is a Sobolev estimate for an observer {observertext}. The photosphere is a sphere at"
        f" {format_velocity(photospherevelocity_kmps, unit)}"
    )
    print_detail(
        "Rays go from the photospheric disk towards the observer, with"
        f" {TROUGHSAMPLESPERCELL} rays across the narrowest width of a cell. A ray meets a line at the point where"
        " the line-of-sight velocity u gives its Doppler shift, and the cell there transmits exp(-tau) of its flux. A"
        " point behind the near side of the photosphere absorbs nothing. The depth in a bin of u is 1 - the mean"
        " transmission over the rays and over the bin. The bins of u have the width of the velocity bins, and they"
        " continue to the fastest corner of a cell"
    )
    print_detail(
        f"The estimate has no emission of the lines, no limb darkening, and only the {linecount} ranked lines. A cell"
        " with a negative tau_sobolev (a population inversion) or with no tau_sobolev absorbs nothing. At each"
        " wavelength, the blend is 1 - the mean over the rays of prod(exp(-tau)) of the lines. Thus two lines that"
        " absorb the same part of the disk do not remove its flux two times"
    )
    print_warning(
        "The estimate takes a sharp photosphere and pure absorption. Many optically thick lines in front of the"
        " photosphere scatter the light many times. Then the real trough can be much less deep. plotopacity with"
        " --vmin, --costhetamin, --costhetamax, and --cellaverage median gives the expansion opacity of each direction"
    )
    if dopplershift == "true":
        print_detail(
            "The wavelength of a bin is lambda_rest sqrt((1 - u / c) / (1 + u / c)) for the line-of-sight velocity u"
            " of its centre. artisoptions.h sets USE_RELATIVISTIC_DOPPLER_SHIFT, thus the Doppler factor of ARTIS"
            " also has the transverse velocity v_t. A bin has no v_t, thus the blend is too blue by a fraction of"
            " about v_t^2 / (2 c^2)"
        )
    elif dopplershift is not None:
        print_detail(
            "The wavelength of a bin is lambda_rest (1 - u / c) for the line-of-sight velocity u of its centre. This"
            " is the Doppler factor of ARTIS, because artisoptions.h sets USE_RELATIVISTIC_DOPPLER_SHIFT ="
            f" {dopplershift}"
        )
    else:
        print_warning(
            "The model folder has no artisoptions.h, or its artisoptions.h gives no USE_RELATIVISTIC_DOPPLER_SHIFT."
            " Thus the wavelength of a bin is lambda_rest (1 - u / c) for the line-of-sight velocity u of its centre."
            " If ARTIS used the relativistic factor, the blend is too blue by a fraction of about u^2 / (2 c^2)"
        )
    deepest = dfblend.top_k(1, by="trough_depth").row(0, named=True)
    print_detail(
        f"The blend has its largest depth {deepest['trough_depth']:.3f} at {deepest['lambda_angstroms']:.0f} Å"
        " (rest frame of the observer)"
    )
