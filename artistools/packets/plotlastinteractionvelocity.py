"""Plot the velocity of the last interaction of the escaped packets in a wavelength range against time."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import c_ang_per_s
from artistools.constants import km_to_cm
from artistools.misc import addarg_averageoverphi
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_plotinvalidpart
from artistools.misc import addarg_show
from artistools.misc import addarg_timeminmax
from artistools.misc import addarg_wavelengthrange
from artistools.misc import check_dirbin
from artistools.misc import exit_with_error
from artistools.misc import get_bin_index_expr
from artistools.misc import get_dirbin_labels
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_model_name
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_weighted_quantile_expr
from artistools.misc import normalize_path_list
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import print_warning
from artistools.packets.core import addarg_maxpacketfiles
from artistools.packets.core import addarg_velocitykind
from artistools.packets.core import get_last_interaction_velocity_expr
from artistools.packets.core import NOISY_RELATIVE_ERROR
from artistools.packets.core import print_arrival_time_method
from artistools.packets.core import print_last_interaction_convention
from artistools.packets.core import print_velocity_method
from artistools.packets.core import scan_packets
from artistools.plottools import addarg_velocityunit
from artistools.plottools import convert_velocity_columns
from artistools.plottools import format_velocity
from artistools.plottools import get_grid_corner_velocity_kmps
from artistools.plottools import get_velocity_unit
from artistools.plottools import KMPS_PER_VELOCITYUNIT
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_axis_properties
from artistools.plottools import set_legend
from artistools.plottools import VELOCITYUNIT_AXISLABELS

if t.TYPE_CHECKING:
    import matplotlib.figure as mplfig

DEFAULTOUTPUTNAME = "plotlastinteractionvelocity.pdf"

# the quartiles come from a histogram of the velocity, because a timestep can hold 4e8 packets and their sort needs
# much memory. This number of bins up to the grid corner gives a resolution far below the width of a cell
VELOCITYHISTOGRAMBINS = 2000


def get_last_interaction_velocity_quartiles(
    modelpath: Path,
    dirbins: Sequence[int],
    lambdamin: float,
    lambdamax: float,
    position: t.Literal["em", "trueem"],
    velocitykind: t.Literal["radial", "lineofsight"],
    histogrambinwidth_kmps: float,
    timerange_days: tuple[float, float],
    *,
    plotinvalidpart: bool,
    averageoverphi: bool = False,
    maxpacketfiles: int | None = None,
) -> dict[int, tuple[pl.DataFrame, int]]:
    """Return, for each direction bin, the velocity quartiles of the last interaction of each timestep.

    Each value of the dictionary also gives the number of packets with no position. The packets are the escaped
    r-packets with a rest-frame wavelength from lambdamin to lambdamax at the escape. A packet takes the timestep of its
    arrival time, and a timestep is in the result if its mid time is in timerange_days. A range with no mid time of
    the model gives a warning and empty tables, because a different model can have timesteps in the range. A timestep
    outside the valid range of the arrival times holds only a part of the light, thus only plotinvalidpart keeps it.

    The quartiles take the energy of each packet as the weight, and each velocity takes the centre of its bin of the
    histogram. One scan of the packets gives the quartiles of all the direction bins. A direction bin takes all the phi
    bins of its costheta bin, and -1 takes all the directions.
    """
    _, lzpackets = scan_packets(
        modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )
    tstarts = get_timestep_times(modelpath, loc="start")
    tends = get_timestep_times(modelpath, loc="end")
    timemids = get_timestep_times(modelpath, loc="mid")
    timesteps = [timestep for timestep, tmid in enumerate(timemids) if timerange_days[0] <= tmid <= timerange_days[1]]
    if not timesteps:
        print_warning(
            f"No timestep of {modelpath} has its mid time in the range of --timemin and --timemax. The mid times go"
            f" from {timemids[0]:.2f} to {timemids[-1]:.2f} d"
        )
        return {dirbin: (pl.DataFrame(), 0) for dirbin in dirbins}
    if not plotinvalidpart:
        timesteps = get_timesteps_in_valid_range(modelpath, timesteps)
        if not timesteps:
            return {dirbin: (pl.DataFrame(), 0) for dirbin in dirbins}
    print_arrival_time_method(modelpath, timesteps, "each timestep with a mid time from --timemin to --timemax")
    print_last_interaction_convention(modelpath)
    timeedges = [*tstarts, tends[-1]]
    try:
        velocity_kmps = (
            get_last_interaction_velocity_expr(position, velocitykind, lzpackets.collect_schema().names()) / km_to_cm
        )
    except ValueError as exc:
        exit_with_error(str(exc))
    # the filter of the arrival time comes first, thus the scan reads only the row groups of the selected timesteps
    dfhistogram = (
        lzpackets
        .filter(pl.col("t_arrive_d").is_between(tstarts[timesteps[0]], tends[timesteps[-1]], closed="left"))
        .filter((c_ang_per_s / pl.col("nu_rf")).is_between(lambdamin, lambdamax))
        .select(
            "e_rf",
            "costhetabin",
            "dirbin",
            timestep=get_bin_index_expr(pl.col("t_arrive_d"), timeedges),
            # a packet with no record has a velocity of NaN, and the cast gives it no bin
            velocitybin=(velocity_kmps / histogrambinwidth_kmps).floor().cast(pl.Int64, strict=False),
        )
        .filter(pl.col("timestep").is_in(timesteps))
        .group_by("costhetabin", "dirbin", "timestep", "velocitybin")
        .agg(
            pl.col("e_rf").cast(pl.Float64).sum(),
            energysquared=(pl.col("e_rf").cast(pl.Float64) ** 2).sum(),
            packetcount=pl.len(),
        )
        .collect()
    )

    nphibins = get_viewingdirection_phibincount()
    results = {}
    for dirbin in dirbins:
        if dirbin == -1:
            dfdirbin = dfhistogram
        elif averageoverphi:
            dfdirbin = dfhistogram.filter(pl.col("costhetabin") == dirbin // nphibins)
        else:
            dfdirbin = dfhistogram.filter(pl.col("dirbin") == dirbin)
        dfdirbin = dfdirbin.group_by("timestep", "velocitybin").agg(
            pl.col("e_rf", "energysquared", "packetcount").sum()
        )
        norecordcount = int(dfdirbin.filter(pl.col("velocitybin").is_null())["packetcount"].sum())
        dfquartiles = (
            dfdirbin
            .filter(pl.col("velocitybin").is_not_null())
            .with_columns(velocity_kmps=(pl.col("velocitybin").cast(pl.Float64) + 0.5) * histogrambinwidth_kmps)
            .group_by("timestep")
            .agg(
                packetcount=pl.col("packetcount").sum(),
                luminosity_relative_error=pl.col("energysquared").sum().sqrt() / pl.col("e_rf").sum(),
                **{
                    f"velocity_luminosity_q{round(quantile * 100)}_kmps": get_weighted_quantile_expr(
                        "velocity_kmps", "e_rf", quantile
                    )
                    for quantile in (0.25, 0.5, 0.75)
                },
            )
            .with_columns(
                time_days=pl.col("timestep").replace_strict(range(len(timemids)), timemids, return_dtype=pl.Float64)
            )
            .sort("timestep")
            .select(
                "timestep", "time_days", "packetcount", "luminosity_relative_error", pl.col("^velocity_luminosity_q.*$")
            )
        )
        results[dirbin] = (dfquartiles, norecordcount)
    return results


def get_timesteps_in_valid_range(modelpath: Path, timesteps: Sequence[int]) -> list[int]:
    """Return the timesteps inside the valid range of the arrival times, and log the timesteps that it leaves out."""
    tstarts = get_timestep_times(modelpath, loc="start")
    tends = get_timestep_times(modelpath, loc="end")
    try:
        _, validstart_days, validend_days = get_escaped_arrivalrange(modelpath)
    except FileNotFoundError:
        return list(timesteps)
    validtimesteps = [
        timestep
        for timestep in timesteps
        if validstart_days is not None
        and validend_days is not None
        and tstarts[timestep] >= validstart_days
        and tends[timestep] <= validend_days
    ]
    if len(validtimesteps) < len(timesteps):
        rangetext = (
            "The model has no valid range of the arrival times"
            if validstart_days is None or validend_days is None
            else f"The valid range of the arrival times is {validstart_days:.3g} to {validend_days:.3g} d"
        )
        print_warning(
            f"{rangetext}, thus the command leaves out {len(timesteps) - len(validtimesteps)} of {len(timesteps)}"
            f" timesteps of {modelpath}. The light of a part of the ejecta does not reach the observer in them."
            " --plotinvalidpart keeps them"
        )
    return validtimesteps


def print_method(
    position: t.Literal["em", "trueem"],
    velocitykind: t.Literal["radial", "lineofsight"],
    lambdamin: float,
    lambdamax: float,
    histogrambinwidth_kmps: float,
) -> None:
    """Print the selection of the packets and the method of the quartiles."""
    print_detail(
        f"The packets are the escaped r-packets with a rest-frame wavelength from {lambdamin:g} to {lambdamax:g} Å at"
        " the escape. Each packet reaches the observer. Thus the energy of a packet gives its part of the observed"
        " luminosity in that wavelength range"
    )
    print_velocity_method(position, velocitykind)
    if position == "em" and velocitykind == "radial":
        print_detail(
            "In a range with no strong line, the median radial velocity is an estimate of the velocity of the"
            " photosphere"
        )
    print_detail(
        "In the plot, each point of a series is one timestep, at its mid time. The curve gives the median. The shaded"
        " region goes from the 25% quantile to the 75% quantile. The energy of each packet is the weight."
        " Thus half of the observed luminosity comes from a last interaction below the median velocity. The velocities"
        f" come from a histogram with bins of {histogrambinwidth_kmps:.3g} km/s"
    )


def plot_velocity_against_time(
    args: argparse.Namespace, unit: t.Literal["kmps", "c"]
) -> "tuple[dict[str, pl.DataFrame], mplfig.Figure]":
    """Plot the median and the quartiles of each model and direction bin, and return the tables and the figure."""
    from artistools.inputmodel import scan_modeldata

    fig, axesgrid = make_frame_figure(args, aspect=0.6)
    ax = axesgrid[0, 0]
    kmpsperunit = KMPS_PER_VELOCITYUNIT[unit]
    position: t.Literal["em", "trueem"] = "trueem" if args.use_thermalemissiontype else "em"
    seriestables = []
    summaryrows = []
    for modelpath in args.modelpath:
        print_modelpath(modelpath)
        _, modelmeta = scan_modeldata(modelpath, printwarningsonly=True)
        histogrambinwidth_kmps = get_grid_corner_velocity_kmps(modelmeta) / VELOCITYHISTOGRAMBINS
        print_method(position, args.velocitykind, args.xmin, args.xmax, histogrambinwidth_kmps)
        dirbinlabels = get_dirbin_labels(args.dirbin, average_over_phi=args.average_over_phi_angle)
        timerange_days = (
            -math.inf if args.timemin is None else args.timemin,
            math.inf if args.timemax is None else args.timemax,
        )
        quartilesofdirbin = get_last_interaction_velocity_quartiles(
            modelpath,
            args.dirbin,
            args.xmin,
            args.xmax,
            position,
            args.velocitykind,
            histogrambinwidth_kmps,
            timerange_days,
            plotinvalidpart=args.plotinvalidpart,
            averageoverphi=args.average_over_phi_angle,
            maxpacketfiles=args.maxpacketfiles,
        )
        for dirbin in args.dirbin:
            dfquartiles, norecordcount = quartilesofdirbin[dirbin]
            label = f"{get_model_name(modelpath)}, {dirbinlabels[dirbin]}"
            if norecordcount > 0:
                print_detail(f"{label}: {norecordcount} packets have no record of the position, thus no velocity")
            if dfquartiles.is_empty():
                print_detail(f"{label}: no packet in the range of wavelength and time")
                continue
            times = dfquartiles["time_days"].to_numpy()
            # a line with no marker draws nothing for a series of one timestep, and the marker shows each timestep
            (line,) = ax.plot(
                times, dfquartiles["velocity_luminosity_q50_kmps"].to_numpy() / kmpsperunit, marker=".", label=label
            )
            ax.fill_between(
                times,
                dfquartiles["velocity_luminosity_q25_kmps"].to_numpy() / kmpsperunit,
                dfquartiles["velocity_luminosity_q75_kmps"].to_numpy() / kmpsperunit,
                color=line.get_color(),
                alpha=0.2,
                linewidth=0,
            )
            seriestables.append(
                dfquartiles.select(
                    pl.lit(label).alias("series"),
                    pl.lit(str(modelpath)).alias("modelpath"),
                    pl.lit(dirbin).alias("dirbin"),
                    pl.all(),
                )
            )
            medians = dfquartiles["velocity_luminosity_q50_kmps"].to_numpy()
            summaryrows.append({
                "series": label,
                "packetcount": int(dfquartiles["packetcount"].sum()),
                "time_days_min": float(times.min()),
                "time_days_max": float(times.max()),
                "velocity_luminosity_q50_min_kmps": float(medians.min()),
                "velocity_luminosity_q50_max_kmps": float(medians.max()),
            })
            # the packets can have different energies, thus the error comes from the energies and not from the counts
            noisytimes = dfquartiles.filter(pl.col("luminosity_relative_error") > NOISY_RELATIVE_ERROR)[
                "time_days"
            ].to_list()
            if noisytimes:
                print_warning(
                    f"{label}: {len(noisytimes)} of {dfquartiles.height} timesteps have a relative error"
                    f" sqrt(sum(e^2)) / sum(e) above {NOISY_RELATIVE_ERROR:.0%}, e.g. at {noisytimes[0]:.2f} d. Their"
                    " quartiles come from few packets"
                )
            print_detail(
                f"{label}: {int(dfquartiles['packetcount'].sum())} packets in {dfquartiles.height} timesteps, median"
                f" from {format_velocity(float(medians[0]), unit)} at {times[0]:.2f} d to"
                f" {format_velocity(float(medians[-1]), unit)} at {times[-1]:.2f} d"
            )

    if not seriestables:
        exit_with_error(
            "No model has a packet in the range of wavelength and time",
            "Give a different time range or a wider wavelength range",
        )
    ax.set_xlabel("Time [d]")
    interaction = "thermal emission" if position == "trueem" else "interaction"
    velocityname = (
        "Radial velocity" if args.velocitykind == "radial" else "Line-of-sight velocity (towards the observer)"
    )
    ax.set_ylabel(f"{velocityname} of the last {interaction} [{VELOCITYUNIT_AXISLABELS[unit]}]")
    set_axis_properties(ax, args, xlimits=(args.timemin, args.timemax, "-timemin"))
    if args.velocitykind == "radial":
        ax.set_ylim(bottom=0.0)
    set_legend(ax, args)
    tables = {
        "": convert_velocity_columns(pl.concat(seriestables), unit),
        "_summary": convert_velocity_columns(pl.DataFrame(summaryrows), unit),
    }
    return tables, fig


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[], helptext="Paths of the ARTIS models")
    addarg_wavelengthrange(parser, "the packets at the escape, in the rest frame,")
    parser.add_argument(
        "-dirbin",
        type=int,
        nargs="+",
        default=[-1],
        help=(
            "The direction bins of the escaped packets, as in plotspectra, e.g. 0 55 99. --average-over-phi-angle"
            " takes all the phi bins of the costheta bin of each. The value -1 takes all the directions"
        ),
    )
    addarg_timeminmax(parser)
    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Take the velocity of the last thermal emission and not of the last interaction",
    )
    addarg_velocityunit(parser)
    addarg_velocitykind(parser)
    addarg_output(parser, kind="file", defaultname=DEFAULTOUTPUTNAME, helptext="Path/filename for the plot file")
    addarg_figscale(parser)
    addarg_data_output(parser)
    addarg_plotinvalidpart(parser, "the timesteps")
    addarg_averageoverphi(parser)
    addarg_maxpacketfiles(parser)
    addarg_show(parser)
    addarg_darkmode(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the median velocity of the last interaction of the packets in a wavelength range against time."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    if args.xmin is None or args.xmax is None:
        exit_with_error(
            "The command needs the wavelength range of the packets",
            "Give --lambdamin and --lambdamax, e.g. --lambdamin 7200 --lambdamax 7800",
        )
    for dirbin in args.dirbin:
        check_dirbin(dirbin, "-dirbin", averageoverphi=args.average_over_phi_angle, allowall=True)
    args.modelpath = normalize_path_list(args.modelpath)
    unit = get_velocity_unit(args.modelpath[0], args.velocityunit)

    tables, fig = plot_velocity_against_time(args, unit)
    outputfile = Path(args.outputfile)
    output_data(args, tables, outputfile)
    save_figure(fig, outputfile, args=args)
