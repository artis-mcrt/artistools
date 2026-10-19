"""Plot the velocity of the last interaction of the escaped packets in a wavelength range against time."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import c_ang_per_s
from artistools.constants import km_to_cm
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import addarg_timeminmax
from artistools.misc import exit_with_error
from artistools.misc import get_model_name
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import get_weighted_quantile_expr
from artistools.misc import normalize_path_list
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_modelpath
from artistools.misc import resolve_outputfile
from artistools.misc.dirbins import get_dirbin_labels
from artistools.misc.general import get_bin_index_expr
from artistools.packets.core import filter_packets_dirbin
from artistools.packets.core import get_emission_velocity_of_columns_expr
from artistools.packets.core import get_packets
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
    dirbin: int,
    lambdamin: float,
    lambdamax: float,
    position: t.Literal["em", "trueem"],
    histogrambinwidth_kmps: float,
) -> tuple[pl.DataFrame, int]:
    """Return the velocity quartiles of the last interaction of each timestep, and the number with no position.

    The packets are the escaped r-packets of the direction bin with a rest-frame wavelength from lambdamin to
    lambdamax at the escape. A packet takes the timestep of its arrival time. The quartiles take the energy of each
    packet as the weight, and each velocity takes the centre of its bin of the histogram.
    """
    _, lzpackets = get_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    lzpackets, _ = filter_packets_dirbin(lzpackets, dirbin, average_over_phi=True)
    timeedges = [*get_timestep_times(modelpath, loc="start"), get_timestep_times(modelpath, loc="end")[-1]]
    velocity_kmps = get_emission_velocity_of_columns_expr(position, lzpackets.collect_schema().names()) / km_to_cm
    dfhistogram = (
        lzpackets
        .filter((c_ang_per_s / pl.col("nu_rf")).is_between(lambdamin, lambdamax))
        .select(
            "e_rf",
            timestep=get_bin_index_expr(pl.col("t_arrive_d"), timeedges),
            # a packet with no record has a velocity of NaN, and the cast gives it no bin
            velocitybin=(velocity_kmps / histogrambinwidth_kmps).floor().cast(pl.Int64, strict=False),
        )
        .filter(pl.col("timestep").is_between(0, len(timeedges) - 2))
        .group_by("timestep", "velocitybin")
        .agg(pl.col("e_rf").cast(pl.Float64).sum(), packetcount=pl.len())
        .collect()
    )
    norecordcount = int(dfhistogram.filter(pl.col("velocitybin").is_null())["packetcount"].sum())
    timemids = get_timestep_times(modelpath, loc="mid")
    dfquartiles = (
        dfhistogram
        .filter(pl.col("velocitybin").is_not_null())
        .with_columns(velocity_kmps=(pl.col("velocitybin").cast(pl.Float64) + 0.5) * histogrambinwidth_kmps)
        .group_by("timestep")
        .agg(
            packetcount=pl.col("packetcount").sum(),
            **{
                f"velocity_q{round(quantile * 100)}_kmps": get_weighted_quantile_expr("velocity_kmps", "e_rf", quantile)
                for quantile in (0.25, 0.5, 0.75)
            },
        )
        .with_columns(
            time_days=pl.col("timestep").replace_strict(range(len(timemids)), timemids, return_dtype=pl.Float64)
        )
        .sort("timestep")
        .select("timestep", "time_days", "packetcount", pl.col("^velocity_q.*$"))
    )
    return dfquartiles, norecordcount


def print_method(
    position: t.Literal["em", "trueem"], lambdamin: float, lambdamax: float, histogrambinwidth_kmps: float
) -> None:
    """Print the selection of the packets and the method of the quartiles."""
    print_detail(
        f"The packets are the escaped r-packets with a rest-frame wavelength from {lambdamin:g} to {lambdamax:g} Å at"
        " the escape. A packet takes the timestep of its arrival time"
    )
    if position == "em":
        print_detail(
            "The velocity is |em_pos| / em_time, the radial velocity of the last interaction. In a range with no strong"
            " line, its median is an estimate of the velocity of the photosphere. An electron scattering also sets"
            " em_pos"
        )
    else:
        print_detail(
            "The velocity is |trueem_pos| / trueem_time, the radial velocity of the last thermal emission. An old"
            " packets file has no trueem_pos, and its column true_emission_velocity then gives the velocity"
        )
    print_detail(
        "In the plot, the curve of each series gives the median. The shaded region goes from the 25% quantile to the"
        " 75% quantile, with the energy of each packet as the weight. The velocities come from a histogram with bins"
        f" of {histogrambinwidth_kmps:.3g} km/s"
    )


def plot_velocity_against_time(
    args: argparse.Namespace, unit: t.Literal["kmps", "c"]
) -> "tuple[dict[str, pl.DataFrame], mplfig.Figure]":
    """Plot the median and the quartiles of each model and direction bin, and return the tables and the figure."""
    from artistools.inputmodel import get_modeldata

    fig, axesgrid = make_frame_figure(args, aspect=0.6)
    ax = axesgrid[0, 0]
    kmpsperunit = KMPS_PER_VELOCITYUNIT[unit]
    position: t.Literal["em", "trueem"] = "trueem" if args.use_thermalemissiontype else "em"
    seriestables = []
    summaryrows = []
    for modelpath in args.modelpath:
        print_modelpath(modelpath)
        _, modelmeta = get_modeldata(modelpath, printwarningsonly=True)
        histogrambinwidth_kmps = get_grid_corner_velocity_kmps(modelmeta) / VELOCITYHISTOGRAMBINS
        print_method(position, args.xmin, args.xmax, histogrambinwidth_kmps)
        dirbinlabels = get_dirbin_labels(args.dirbin, average_over_phi=True, usedegrees=True)
        for dirbin in args.dirbin:
            dfquartiles, norecordcount = get_last_interaction_velocity_quartiles(
                modelpath, dirbin, args.xmin, args.xmax, position, histogrambinwidth_kmps
            )
            dfquartiles = dfquartiles.filter(
                pl.col("time_days").is_between(args.timemin or -math.inf, args.timemax or math.inf)
            )
            label = f"{get_model_name(modelpath)}, {dirbinlabels[dirbin]}"
            if norecordcount > 0:
                print_detail(f"{label}: {norecordcount} packets have no record of the position, thus no velocity")
            if dfquartiles.is_empty():
                print_detail(f"{label}: no packet in the range of wavelength and time")
                continue
            times = dfquartiles["time_days"].to_numpy()
            # a line with no marker draws nothing for a series of one timestep, and the marker shows each timestep
            (line,) = ax.plot(times, dfquartiles["velocity_q50_kmps"].to_numpy() / kmpsperunit, marker=".", label=label)
            ax.fill_between(
                times,
                dfquartiles["velocity_q25_kmps"].to_numpy() / kmpsperunit,
                dfquartiles["velocity_q75_kmps"].to_numpy() / kmpsperunit,
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
            medians = dfquartiles["velocity_q50_kmps"].to_numpy()
            summaryrows.append({
                "series": label,
                "packetcount": int(dfquartiles["packetcount"].sum()),
                "time_days_min": float(times.min()),
                "time_days_max": float(times.max()),
                "velocity_q50_min_kmps": float(medians.min()),
                "velocity_q50_max_kmps": float(medians.max()),
            })
            print_detail(
                f"{label}: {int(dfquartiles['packetcount'].sum())} packets in {dfquartiles.height} timesteps, median"
                f" from {format_velocity(float(medians[0]), unit)} at {times[0]:.2f} d to"
                f" {format_velocity(float(medians[-1]), unit)} at {times[-1]:.2f} d"
            )

    ax.set_xlabel("Time [d]")
    interaction = "thermal emission" if position == "trueem" else "interaction"
    ax.set_ylabel(f"Velocity of the last {interaction} [{VELOCITYUNIT_AXISLABELS[unit]}]")
    set_axis_properties(ax, args, xlimits=(args.timemin, args.timemax, "-timemin"))
    ax.set_ylim(bottom=0.0)
    set_legend(ax, args)
    tables = {
        "": convert_velocity_columns(pl.concat(seriestables), unit) if seriestables else pl.DataFrame(),
        "_summary": convert_velocity_columns(pl.DataFrame(summaryrows), unit) if summaryrows else pl.DataFrame(),
    }
    return tables, fig


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[], helptext="Paths of ARTIS models")
    parser.add_argument(
        "-xmin",
        "-lambdamin",
        type=float,
        help="Minimum rest-frame wavelength of the packets at the escape in Angstroms",
    )
    parser.add_argument(
        "-xmax",
        "-lambdamax",
        type=float,
        help="Maximum rest-frame wavelength of the packets at the escape in Angstroms",
    )
    parser.add_argument(
        "-dirbin",
        type=int,
        nargs="+",
        default=[-1],
        help=(
            "The first direction bin of each costheta bin, e.g. 0 50 90. Each series takes all the phi bins of its"
            " costheta bin. The value -1 takes all the directions"
        ),
    )
    addarg_timeminmax(parser)
    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Take the velocity of the last thermal emission and not of the last interaction",
    )
    addarg_velocityunit(parser)
    addarg_output(parser, kind="file", defaultname=DEFAULTOUTPUTNAME, helptext="Path/filename for the plot file")
    addarg_figscale(parser)
    addarg_data_output(parser)
    addarg_show(parser)
    addarg_darkmode(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the median velocity of the last interaction of the packets in a wavelength range against time."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    if args.xmin is None or args.xmax is None:
        exit_with_error(
            "the command needs the wavelength range of the packets",
            "Give -lambdamin and -lambdamax, e.g. -lambdamin 7200 -lambdamax 7800",
        )
    nphibins = get_viewingdirection_phibincount()
    ndirbins = get_viewingdirectionbincount()
    for dirbin in args.dirbin:
        if dirbin != -1 and not (0 <= dirbin < ndirbins and dirbin % nphibins == 0):
            exit_with_error(
                f"-dirbin {dirbin} is not the first direction bin of a costheta bin",
                f"Give -1 for all the directions, or a multiple of {nphibins} from 0 to {ndirbins - nphibins}",
            )
    args.modelpath = normalize_path_list(args.modelpath)
    unit = get_velocity_unit(args.modelpath[0], args.velocityunit)

    tables, fig = plot_velocity_against_time(args, unit)
    outputfile = resolve_outputfile(args.outputfile, DEFAULTOUTPUTNAME)
    output_data(args, tables, outputfile)
    save_figure(fig, outputfile, args=args)
