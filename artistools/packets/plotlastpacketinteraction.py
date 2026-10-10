"""Plot 2D histograms of where in the ejecta packets were last emitted or scattered."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from artistools.atomic import decode_roman_numeral
from artistools.atomic import get_atomic_number
from artistools.atomic import get_lineindices
from artistools.constants import c_ang_per_s
from artistools.constants import C_cm_per_s as CLIGHT
from artistools.constants import day_to_s
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_unsupported
from artistools.misc import exit_with_error
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import parse_cli_args
from artistools.misc import resolve_outputfile
from artistools.packets.core import filter_packets_dirbin
from artistools.packets.core import get_emission_time_expr
from artistools.packets.core import get_packets
from artistools.plottools import save_figure
from artistools.plottools import set_mpl_style
from artistools.rustext import get_bin_indices
from artistools.rustext import sum_weights_in_bins


def get_required_packets(
    modelpath: Path, Z_list: Sequence[int] | None, ion_stage_list: Sequence[int] | None, srII_triplet: bool = False
) -> tuple[int, pl.LazyFrame]:
    """Return the escaped packets that a line of the given elements and ion stages last absorbed.

    A None list selects every element or every ion stage. The Sr II triplet takes the place of both lists.
    """
    # careful: ion_stage is counted from 1 here, i.e. 1 <-> neutral, 2 <-> singly ionized
    nprocs_read, dfpackets = get_packets(
        modelpath=modelpath, maxpacketfiles=None, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if not srII_triplet and Z_list is ion_stage_list is None:
        # the plot keeps every line, thus the filter needs no list of line indices. A packet
        # that no line absorbed has a negative absorption_type
        return nprocs_read, dfpackets.filter(pl.col("absorption_type") >= 0)

    if srII_triplet:
        lineindices = get_lineindices(
            modelpath,
            [38],
            [2],
            linefilter=((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 3))
            | ((pl.col("lowerlevelindex") == 2) & (pl.col("upperlevelindex") == 4))
            | ((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 4)),
        )
    else:
        lineindices = get_lineindices(modelpath, Z_list, ion_stage_list)

    return nprocs_read, dfpackets.filter(pl.col("absorption_type").is_in(lineindices.implode()))


def get_reduced_packet_set(
    modelpath: Path,
    dirbin: int,
    Z: Sequence[int] | None,
    ion_stage: Sequence[int] | None,
    wavelen: float | None = None,
    binwidth: float | None = None,
    srII_triplet: bool = False,
) -> tuple[int, pl.LazyFrame, float]:
    """Get packets in specific escape angle bins for observer direction.

    The function returns the number of MPI ranks, the packets, and the solid-angle factor
    (4 pi / solidangle) of the direction bin. `get_required_packets()` selects the packets for the
    given element and ion filters. If `wavelen` and `binwidth` both have a value, the function keeps
    only the packets in that wavelength slice. It then keeps only the packets of the direction bin.
    """
    nprocs_read, dfpackets_selected = get_required_packets(modelpath, Z, ion_stage, srII_triplet=srII_triplet)
    dfpackets_selected = dfpackets_selected.with_columns((c_ang_per_s / pl.col("nu_rf")).alias("lambda_rf"))

    if wavelen is not None and binwidth is not None:
        lam_min = wavelen - binwidth / 2
        lam_max = wavelen + binwidth / 2

        dfpackets_selected = dfpackets_selected.filter(
            (pl.col("lambda_rf") > lam_min) & (pl.col("lambda_rf") < lam_max)
        )
    dfpackets_selected, solidangle_factor = filter_packets_dirbin(dfpackets_selected, dirbin, average_over_phi=True)

    return nprocs_read, dfpackets_selected, solidangle_factor


def packets_2d_hist_bin_and_ejecta_vel(
    modelpath: Path,
    tdays: float,
    srIItriplet: bool,
    colorlogscale: bool,
    dirbin: int,
    trueem: bool,
    Z: int | None = None,
    ion_stage_str: str | None = None,
    wavelen: float | None = None,
    binwidth: float | None = None,
    outputfile: Path | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Plot a 2D histogram of packet emission position against ejecta velocity, and save the figure.

    An outputfile of None or of a folder gives the file a name from the selection of the packets.
    """
    start_of_filename = "" if modelpath == Path() else f"{modelpath.name}_"
    if wavelen is not None:
        start_of_filename = f"{start_of_filename}{wavelen:.0f}A_"
    start_of_filename = f"{start_of_filename}Z={Z}_" if Z else f"{start_of_filename}allelements_"
    start_of_filename = f"{start_of_filename}I={ion_stage_str}_" if ion_stage_str else f"{start_of_filename}allions_"

    # Step 1) collect packets IDs and select according to arrival time. None selects every element or ion stage
    Z_list = [Z] if Z else None
    ion_stage_list = [decode_roman_numeral(ion_stage_str)] if ion_stage_str else None
    if ion_stage_list == [-1]:
        exit_with_error(f"-ionstage {ion_stage_str} is no Roman numeral", "Give the ion stage as e.g. II")

    nprocs_read, dfpackets, inverse_solidangle_fraction = get_reduced_packet_set(
        modelpath, dirbin, Z_list, ion_stage_list, wavelen=wavelen, binwidth=binwidth, srII_triplet=srIItriplet
    )

    start_of_filename += f"t_arrive_d_{tdays}_"
    timeminarray = get_timestep_times(modelpath=modelpath, loc="start")
    timemaxarray = get_timestep_times(modelpath=modelpath, loc="end")
    timestep = get_timestep_of_timedays(modelpath, tdays)
    t_min = timeminarray[timestep]
    t_max = timemaxarray[timestep]
    Delta_t_secs = (t_max - t_min) * day_to_s
    Delta_beta = 0.5 / 25

    position: t.Literal["em", "trueem"] = "em"
    if trueem:
        required_cols = {"trueem_posx", "trueem_posy", "trueem_posz", "trueem_time"}
        missing_cols = required_cols - set(dfpackets.collect_schema().names())
        if missing_cols:
            message = (
                "--use_thermalemissiontype requires packets with columns "
                f"{sorted(required_cols)} (missing {sorted(missing_cols)})"
            )
            raise ValueError(message)
        position = "trueem"
    print(f"t_min selected: {t_min} t_max_selected: {t_max}, is {Delta_t_secs} seconds")
    # a timestep holds the times from its start up to its end, and the end belongs to the next timestep
    dfpackets = dfpackets.filter(pl.col("t_arrive_d").is_between(t_min, t_max, closed="left"))
    # a packet with no record of the emission has a time of NaN, thus it is outside each bin of the histogram
    emtime = get_emission_time_expr(position)
    dfpackets = dfpackets.with_columns(
        ((pl.col(f"{position}_posx") ** 2 + pl.col(f"{position}_posy") ** 2).sqrt() / emtime / CLIGHT).alias(
            "beta_r_cyl_em"
        )
    ).with_columns((pl.col(f"{position}_posz") / emtime / CLIGHT).alias("beta_z_em"))

    dfpackets = dfpackets.with_columns(
        ((pl.col("beta_r_cyl_em") / Delta_beta).floor() * Delta_beta * CLIGHT * emtime).alias("R_cyl_inner_em")
    ).with_columns((pl.col("R_cyl_inner_em") + Delta_beta * CLIGHT * emtime).alias("R_cyl_outer_em"))
    dfpackets_selected = dfpackets.with_columns(
        (
            np.pi
            * (pl.col("R_cyl_outer_em").cast(pl.Float64) ** 2 - pl.col("R_cyl_inner_em").cast(pl.Float64) ** 2)
            * CLIGHT
            * Delta_beta
            * emtime
        ).alias("hollow_cyl_vol_em")
    ).collect()
    energy_sum = float(dfpackets_selected["e_rf"].sum())
    print(
        f"Directional 4pi-equivalent bol. luminosity of {energy_sum / nprocs_read / Delta_t_secs * inverse_solidangle_fraction}"
    )

    # Step 2) create the heatmap. Normalise packet energy to modelgrid cell volume at packet emission time (lab frame)
    # the kernel gives the bins of np.histogram2d. np.histogram2d took 0.13 s for 2.5 million packets at 2 days of
    # a 3D kilonova run. The kernel made the command 0.11 s faster
    xedges = np.linspace(0, 0.5, num=26)
    yedges = np.linspace(-0.5, 0.5, num=51)
    xbinindex = get_bin_indices(dfpackets_selected.select("beta_r_cyl_em"), "beta_r_cyl_em", xedges.tolist())[
        "binindex"
    ]
    dfinxrange = dfpackets_selected.select(
        pl.col("beta_z_em").cast(pl.Float64),
        weight=(pl.col("e_rf") / pl.col("hollow_cyl_vol_em")).cast(pl.Float64),
        xbinindex=xbinindex,
    ).filter(pl.col("xbinindex") >= 0)
    hist2D = (
        sum_weights_in_bins(dfinxrange, "beta_z_em", "weight", yedges.tolist(), "xbinindex", len(xedges) - 1)["sum"]
        .to_numpy()
        .reshape(len(xedges) - 1, len(yedges) - 1)
    )
    heatmap = hist2D / Delta_t_secs / nprocs_read * inverse_solidangle_fraction
    heatmap = np.ma.masked_less_equal(heatmap, 0.0)
    if colorlogscale:
        heatmap = np.ma.log(heatmap)

    # an image with a colorbar keeps plt.subplots: fig.colorbar takes space that the
    # Divider of make_frame_figure gives back at draw time, and the two then overlap
    set_mpl_style()
    fig, ax = plt.subplots(figsize=(3.5, 4.5), layout="constrained")
    z = heatmap.T

    im = ax.imshow(z, origin="lower", cmap="viridis", extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$v_r$ [$c$]")
    ax.set_ylabel(r"$v_z$ [$c$]")
    cbar = fig.colorbar(im, ax=ax)
    if colorlogscale:
        cbar.set_label(r"log volumetric emissivity [erg/(s cm$^3$)]")
    else:
        cbar.set_label(r"volumetric emissivity [erg/(s cm$^3$)]")

    ax.set_xticks(np.linspace(xedges[0], xedges[-1], 6))
    ax.set_yticks(np.linspace(yedges[0], yedges[-1], 6))

    outfilename = resolve_outputfile(outputfile, start_of_filename + f"ts{timestep}_into_dirbin{dirbin}.pdf")
    save_figure(fig, outfilename, dpi=300, args=args)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, required=True, helptext="Path to ARTIS simulation")
    addarg_darkmode(parser)

    addarg_timedays(
        parser,
        kind="float",
        helptext="Time in days. The plot takes the packets that arrive in the timestep of this time",
    )
    parser.add_argument("-tdays", dest="timedays", type=float, help=argparse.SUPPRESS)
    addarg_unsupported(parser, "-timestep", "-ts", instead="-timedays")

    parser.add_argument("-wavelen", type=float, default=None, help="Central wavelength in Angstrom")
    parser.add_argument("-binwidth", type=float, default=None, help="Wavelength bin width in Angstrom")

    parser.add_argument("-element", type=str, default=None, help="Element symbol")
    parser.add_argument("-ionstage", type=str, default=None, help="Ionisation stage (spectroscopic notation)")

    parser.add_argument(
        "-dirbin",
        type=int,
        default=-1,
        help=(
            "The first direction bin of a costheta bin, e.g. 0, 10, or 90. The plot takes all the phi bins of that"
            " costheta bin. The value -1 takes all directions"
        ),
    )
    parser.add_argument("--srIItriplet", action="store_true", help="Plot packets from SrII triplet only")
    parser.add_argument("--colorlogscale", action="store_true", help="Log scale for color bar in 2D plot")

    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Plot true thermal emission rather than last interaction location",
    )

    addarg_output(parser, kind="file", helptext="Path/filename for the PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot last packet interaction properties versus ejecta velocity for selected packets."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if (args.wavelen is None) != (args.binwidth is None):
        message = "Wavelength mode requires both -wavelen and -binwidth to be provided."
        raise ValueError(message)

    if args.timedays is None:
        exit_with_error("the time is missing", "Give the time in days, e.g. -timedays 300")

    nphibins = get_viewingdirection_phibincount()
    ndirbins = get_viewingdirectionbincount()
    if args.dirbin != -1 and not (0 <= args.dirbin < ndirbins and args.dirbin % nphibins == 0):
        exit_with_error(
            f"-dirbin {args.dirbin} is not the first direction bin of a costheta bin",
            f"Give -1 for all directions, or a multiple of {nphibins} from 0 to {ndirbins - nphibins}",
        )

    # get_atomic_number gives -1 for a text that is no element symbol, and 0 for the free neutron, which has no lines
    atomic_number = get_atomic_number(args.element) if args.element else None
    if atomic_number is not None and atomic_number < 1:
        exit_with_error(f"-element {args.element} is no element symbol", "Give the element as e.g. Sr")

    packets_2d_hist_bin_and_ejecta_vel(
        Path(args.modelpath),
        args.timedays,
        args.srIItriplet,
        args.colorlogscale,
        dirbin=args.dirbin,
        Z=atomic_number,
        trueem=args.use_thermalemissiontype,
        ion_stage_str=args.ionstage,
        wavelen=args.wavelen,
        binwidth=args.binwidth,
        outputfile=args.outputfile,
        args=args,
    )
