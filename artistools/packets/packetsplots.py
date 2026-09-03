# PYTHON_ARGCOMPLETE_OK
"""Plot 2D histograms of where in the ejecta packets were last emitted or scattered."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at
from artistools.constants import c_ang_per_s
from artistools.constants import C_cm_per_s as CLIGHT
from artistools.constants import day_to_s
from artistools.misc import addarg_modelpath
from artistools.plottools import save_figure
from artistools.plottools import set_mpl_style


def get_required_packets(
    modelpath: Path, Z_list: Sequence[int] | None, ion_stage_list: Sequence[int] | None, srII_triplet: bool = False
) -> tuple[int, pl.LazyFrame]:
    """Return the escaped packets that a line of the given elements and ion stages last absorbed.

    A None list selects every element or every ion stage. The Sr II triplet takes the place of both lists.
    """
    # careful: ion_stage is counted from 1 here, i.e. 1 <-> neutral, 2 <-> singly ionized

    linelist_lazyframe = at.atomic.get_linelist_pldf(modelpath)
    if srII_triplet:
        linelist_lazyframe = linelist_lazyframe.filter(
            (pl.col("atomic_number") == 38)
            & (pl.col("ion_stage") == 2)
            & (
                ((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 3))
                | ((pl.col("lowerlevelindex") == 2) & (pl.col("upperlevelindex") == 4))
                | ((pl.col("lowerlevelindex") == 1) & (pl.col("upperlevelindex") == 4))
            )
        )
    else:
        if Z_list is not None:
            linelist_lazyframe = linelist_lazyframe.filter(pl.col("atomic_number").is_in(Z_list))
        if ion_stage_list is not None:
            linelist_lazyframe = linelist_lazyframe.filter(pl.col("ion_stage").is_in(ion_stage_list))
    lineindices = linelist_lazyframe.select("lineindex").collect().get_column("lineindex")
    nprocs_read, dfpackets = at.packets.get_packets(
        modelpath=modelpath, maxpacketfiles=None, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )
    dfpackets_selected = dfpackets.filter(pl.col("absorption_type").is_in(lineindices))

    return nprocs_read, dfpackets_selected


def get_reduced_packet_set(
    modelpath: Path,
    dirbin: int,
    Z: Sequence[int] | None,
    ion_stage: Sequence[int] | None,
    wavelen: float | None = None,
    binwidth: float | None = None,
    srII_triplet: bool = False,
) -> tuple[int, pl.LazyFrame]:
    """Get packets in specific escape angle bins for observer direction.

    Selection is based on the packets returned by `get_required_packets()`
    for the requested element/ion filters. If both `wavelen` and `binwidth`
    are provided, the packets are additionally restricted to that wavelength
    slice before filtering to the requested escape-angle bins.
    """
    nprocs_read, dfpackets_selected = get_required_packets(modelpath, Z, ion_stage, srII_triplet=srII_triplet)
    dfpackets_selected = dfpackets_selected.with_columns((c_ang_per_s / pl.col("nu_rf")).alias("lambda_rf"))

    if wavelen is not None and binwidth is not None:
        lam_min = wavelen - binwidth / 2
        lam_max = wavelen + binwidth / 2

        dfpackets_selected = dfpackets_selected.filter(
            (pl.col("lambda_rf") > lam_min) & (pl.col("lambda_rf") < lam_max)
        )
    if dirbin >= 0:
        dfpackets_selected, _ = at.packets.filter_packets_dirbin(dfpackets_selected, dirbin, average_over_phi=True)

    return nprocs_read, dfpackets_selected


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
) -> None:
    """Plot a 2D histogram of packet emission position against ejecta velocity, and save the figure."""
    start_of_filename = "" if modelpath == Path() else f"{modelpath.name}_"
    if wavelen is not None:
        start_of_filename = f"{wavelen:.0f}A_"
    start_of_filename = f"{start_of_filename}Z={Z}_" if Z else f"{start_of_filename}allelements_"
    start_of_filename = f"{start_of_filename}I={ion_stage_str}_" if ion_stage_str else f"{start_of_filename}allions_"

    # Step 1) collect packets IDs and select according to arrival time. None selects every element or ion stage
    Z_list = [Z] if Z else None
    ion_stage_list = [at.decode_roman_numeral(ion_stage_str)] if ion_stage_str else None

    nprocs_read, dfpackets = get_reduced_packet_set(
        modelpath, dirbin, Z_list, ion_stage_list, wavelen=wavelen, binwidth=binwidth, srII_triplet=srIItriplet
    )

    start_of_filename += f"t_arrive_d_{tdays}_"
    timeminarray = at.misc.get_timestep_times(modelpath=modelpath, loc="start")
    timemaxarray = at.misc.get_timestep_times(modelpath=modelpath, loc="end")
    timestep = at.misc.get_timestep_of_timedays(modelpath, tdays)
    t_min = timeminarray[timestep]
    t_max = timemaxarray[timestep]
    Delta_t_secs = (t_max - t_min) * day_to_s
    Delta_beta = 0.5 / 25

    pos_type_str = ""
    if trueem:
        required_cols = {"trueem_posx", "trueem_posy", "trueem_posz", "trueem_time"}
        missing_cols = required_cols - set(dfpackets.collect_schema().names())
        if missing_cols:
            message = (
                "--use_thermalemissiontype requires packets with columns "
                f"{sorted(required_cols)} (missing {sorted(missing_cols)})"
            )
            raise ValueError(message)
        pos_type_str = "true"
    print(f"t_min selected: {t_min} t_max_selected: {t_max}, is {Delta_t_secs} seconds")
    dfpackets = dfpackets.filter(pl.col("t_arrive_d").is_between(t_min, t_max, closed="right"))
    dfpackets = dfpackets.with_columns(
        (
            (pl.col(f"{pos_type_str}em_posx") ** 2 + pl.col(f"{pos_type_str}em_posy") ** 2).sqrt()
            / pl.col(f"{pos_type_str}em_time")
            / CLIGHT
        ).alias("beta_r_cyl_em")
    ).with_columns((pl.col(f"{pos_type_str}em_posz") / pl.col(f"{pos_type_str}em_time") / CLIGHT).alias("beta_z_em"))

    dfpackets = dfpackets.with_columns(
        ((pl.col("beta_r_cyl_em") / Delta_beta).floor() * Delta_beta * CLIGHT * pl.col(f"{pos_type_str}em_time")).alias(
            "R_cyl_inner_em"
        )
    ).with_columns(
        (pl.col("R_cyl_inner_em") + Delta_beta * CLIGHT * pl.col(f"{pos_type_str}em_time")).alias("R_cyl_outer_em")
    )
    dfpackets_selected = dfpackets.with_columns(
        (
            np.pi
            * (pl.col("R_cyl_outer_em").cast(pl.Float64) ** 2 - pl.col("R_cyl_inner_em").cast(pl.Float64) ** 2)
            * CLIGHT
            * Delta_beta
            * pl.col(f"{pos_type_str}em_time")
        ).alias("hollow_cyl_vol_em")
    ).collect()
    inverse_solidangle_fraction = at.get_viewingdirection_costhetabincount() if dirbin >= 0 else 1.0
    energy_sum = float(dfpackets_selected["e_rf"].sum())
    print(
        f"Directional 4pi-equivalent bol. luminosity of {energy_sum / nprocs_read / Delta_t_secs * inverse_solidangle_fraction}"
    )

    # Step 2) create the heatmap. Normalise packet energy to modelgrid cell volume at packet emission time (lab frame)
    weights = dfpackets_selected["e_rf"] / dfpackets_selected["hollow_cyl_vol_em"]
    # derive the emission velocity for each packet from the emission position
    hist2D, xedges, yedges = np.histogram2d(
        dfpackets_selected["beta_r_cyl_em"],
        dfpackets_selected["beta_z_em"],
        bins=[np.linspace(0, 0.5, num=26), np.linspace(-0.5, 0.5, num=51)],
        weights=weights,
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

    outfilename = start_of_filename + f"ts{timestep}_into_dirbin{dirbin}.pdf"
    save_figure(fig, outfilename, dpi=300)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, required=True, helptext="Path to ARTIS simulation")

    parser.add_argument(
        "-tdays",
        type=float,
        required=True,
        help="Time in days, collects packets for the timestep in which the specified value lies in",
    )

    parser.add_argument("-wavelen", type=float, default=None, help="Central wavelength in Angstrom")
    parser.add_argument("-binwidth", type=float, default=None, help="Wavelength bin width in Angstrom")

    parser.add_argument("-element", type=str, default=None, help="Element symbol")
    parser.add_argument("-ionstage", type=str, default=None, help="Ionisation stage (spectroscopic notation)")

    parser.add_argument("-dirbin", type=int, default=-1, help="Viewing direction bin. Default is isotropic (-1)")
    parser.add_argument("--srIItriplet", action="store_true", help="Plot packets from SrII triplet only")
    parser.add_argument("--colorlogscale", action="store_true", help="Log scale for color bar in 2D plot")

    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Plot true thermal emission rather than last interaction location",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot last packet interaction properties versus ejecta velocity for selected packets."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if (args.wavelen is None) != (args.binwidth is None):
        message = "Wavelength mode requires both -wavelen and -binwidth to be provided."
        raise ValueError(message)

    assert args.dirbin == -1 or (args.dirbin % at.get_viewingdirection_phibincount()) == 0, (
        "dirbin needs to be -1 (isotropic) or a multiple of 10 (to be improved)"
    )

    packets_2d_hist_bin_and_ejecta_vel(
        Path(args.modelpath),
        args.tdays,
        args.srIItriplet,
        args.colorlogscale,
        dirbin=args.dirbin,
        Z=at.get_atomic_number(args.element) if args.element else None,
        trueem=args.use_thermalemissiontype,
        ion_stage_str=args.ionstage,
        wavelen=args.wavelen,
        binwidth=args.binwidth,
    )


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
