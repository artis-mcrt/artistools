import argparse
import typing as t
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import cast
from typing import TYPE_CHECKING

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import artistools as at
import artistools.linefluxes

CLIGHT = 2.99792458e10
MSUN = 1.989e33
DAY = 86400
colours = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def make_2d_packets_plot_imshow(modelpath: Path, timestep_min: int, timestep_max: int) -> None:
    plmodeldata, modelmeta = at.inputmodel.get_modeldata(modelpath)
    vmax_cms = modelmeta["vmax_cmps"]
    modeldata = plmodeldata.collect().to_pandas(use_pyarrow_extension_array=True)
    em_time = True  # False for arrive time

    hist = at.packets.make_3d_histogram_from_packets(
        modelpath, timestep_min=timestep_min, timestep_max=timestep_max, em_time=em_time
    )

    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    vmax_cms /= CLIGHT

    # # Don't plot empty cells
    # i = 0
    # for z in range(0, grid):
    #     for y in range(0, grid):
    #         for x in range(0, grid):
    #             if modeldata["rho"][i] == 0.0:
    #                 hist[x, y, z] = None
    #             i += 1

    timeminarray = at.get_timestep_times(modelpath=modelpath, loc="start")
    timemaxarray = at.get_timestep_times(modelpath=modelpath, loc="end")
    time_lower = timeminarray[timestep_min]
    time_upper = timemaxarray[timestep_max]
    title = f"{time_lower:.2f} - {time_upper:.2f} days"
    print(f"plotting packets between {title}")
    escapetitle = "pktemissiontime" if em_time else "pktarrivetime"
    title = title + "\n" + escapetitle

    plot_axes_list = ["xz", "xy"]
    for plot_axes in plot_axes_list:
        data, extent = at.plottools.imshow_init_for_artis_grid(grid, vmax_cms, hist / 1e41, plot_axes=plot_axes)

        plt.imshow(data, extent=extent)
        cbar = plt.colorbar()
        # cbar.set_label('n packets', rotation=90)
        cbar.set_label(r"energy emission rate ($10^{41}$ erg/s)", rotation=90)
        # cbar.set_label(r'npackets)', rotation=90)
        plt.xlabel(f"v{plot_axes[0]} / c")
        plt.ylabel(f"v{plot_axes[1]} / c")
        plt.xlim(-vmax_cms, vmax_cms)
        plt.ylim(-vmax_cms, vmax_cms)

        # plt.title(title)
        # plt.show()
        outfilename = f"packets_hist_{time_lower:.2f}d_{plot_axes}_{escapetitle}.pdf"
        plt.savefig(Path(modelpath) / outfilename, format="pdf")
        print(f"open {outfilename}")
        plt.clf()


def make_2d_packets_plot_pyvista(modelpath: Path, timestep: int) -> None:
    import pyvista as pv

    plmodeldata, modelmeta = at.inputmodel.get_modeldata(modelpath)
    vmax_cms = modelmeta["vmax_cmps"]
    modeldata = plmodeldata.collect().to_pandas(use_pyarrow_extension_array=True)
    _, x, y, z = at.packets.make_3d_grid(modeldata, vmax_cms)
    mesh: t.Any = pv.StructuredGrid(x, y, z)

    hist = at.packets.make_3d_histogram_from_packets(modelpath, timestep)

    mesh["energy [erg/s]"] = hist.ravel(order="F")
    # print(max(mesh['energy [erg/s]']))

    sargs = {
        "height": 0.75,
        "vertical": True,
        "position_x": 0.04,
        "position_y": 0.1,
        "title_font_size": 22,
        "label_font_size": 25,
    }

    pv.set_plot_theme("document")
    p: t.Any = pv.Plotter()

    p.set_scale(p, xscale=1.5, yscale=1.5, zscale=1.5)
    single_slice = mesh.slice(normal="y")
    # single_slice = mesh.slice(normal='z')
    p.add_mesh(single_slice, scalar_bar_args=sargs)
    p.show_bounds(
        p,
        grid=False,
        xlabel="vx / c",
        ylabel="vy / c",
        zlabel="vz / c",
        ticks="inside",
        minor_ticks=False,
        use_2d=True,
        font_size=26,
        bold=False,
    )
    # labels = dict(xlabel='vx / c', ylabel='vy / c', zlabel='vz / c')
    # p.show_grid(**labels)
    p.camera_position = "zx"
    timeminarray = at.get_timestep_times(modelpath=modelpath, loc="start")
    time = timeminarray[timestep]
    p.add_title(f"{time:.2f} - {timeminarray[timestep + 1]:.2f} days")
    print(pv.global_theme)

    p.show(screenshot=modelpath / f"3Dplot_pktsemitted{time:.1f}days_disk.png")


def plot_packet_mean_emission_velocity(modelpath: str | Path, write_emission_data: bool = True) -> None:
    emission_data = at.packets.get_mean_packet_emission_velocity_per_ts(modelpath)

    plt.plot(emission_data["t_arrive_d"], emission_data["mean_emission_velocity"])

    plt.xlim(0.02, 30)
    plt.ylim(0.15, 0.35)
    plt.xscale("log")
    plt.xlabel("Time (days)")
    plt.ylabel("Mean emission velocity / c")
    plt.legend()

    if write_emission_data:
        emission_data.to_csv(Path(modelpath) / "meanemissionvelocity.txt", sep=" ", index=False)

    outfilename = "meanemissionvelocity.pdf"
    plt.savefig(Path(modelpath) / outfilename, format="pdf")
    print(f"open {outfilename}")


def plot_last_emission_velocities_histogram(
    modelpath: Path,
    timestep_min: int,
    timestep_max: int,
    costhetabin: int | None = None,
    maxpacketfiles: int | None = None,
) -> None:
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(5, 4), tight_layout={"pad": 1.0, "w_pad": 0.0, "h_pad": 0.5}, sharex=True
    )

    dfmodel, modelmeta = at.get_modeldata(modelpath=modelpath, printwarningsonly=True)

    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles=maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )
    dfpackets = at.packets.bin_packet_directions_polars(dfpackets=dfpackets)
    dfpackets = at.packets.add_derived_columns_lazy(dfpackets, modelmeta=modelmeta, dfmodel=dfmodel)
    print("read packets data")

    timeminarray = at.misc.get_timestep_times(modelpath=modelpath, loc="start")
    timemaxarray = at.misc.get_timestep_times(modelpath=modelpath, loc="end")
    timelow = timeminarray[timestep_min]
    timehigh = timemaxarray[timestep_max]
    print(f"Using packets arriving at observer between {timelow:.2f} and {timehigh:.2f} days")

    dfpackets_selected = dfpackets.filter(pl.col("t_arrive_d").is_between(timelow, timehigh, closed="right"))

    if costhetabin is not None:
        dfpackets_selected = dfpackets.filter(pl.col("costhetabin") == costhetabin)

    weight_by_energy = True
    if weight_by_energy:
        e_rf = dfpackets_selected.select("e_rf").collect()
        weights = e_rf
    else:
        weights = None

    # bin packets by ejecta velocity the packet was emitted from
    hist, bin_edges = np.histogram(
        dfpackets_selected.select("emission_velocity").collect() / 2.99792458e10,
        range=(0.0, 0.7),
        bins=28,
        weights=weights,
    )
    hist = hist / nprocs_read / (timemaxarray[timestep_max] - timeminarray[timestep_min]) / 86400  # erg/s

    hist /= 1e40
    width = np.diff(bin_edges)
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.bar(center, hist, align="center", width=width, linewidth=2, fill=True)

    ax.set_xticks(bin_edges[::4])
    ax.set_xlabel("Velocity [c]")
    ax.set_ylabel(r"Energy rate ($10^{40}$ erg/s)")

    outfilename = f"hist_emission_vel_{timeminarray[timestep_min]:.2f}-{timemaxarray[timestep_max]:.2f}d.pdf"
    fig.savefig(Path(modelpath) / outfilename, format="pdf")
    print(f"open {outfilename}")


def get_required_packets(modelpath: Path, Z: int, ion_stage: int, srII_triplet: bool = False) -> pd.DataFrame:
    """Options for this function: Either the Sr II triplet, specific or all ion stages of an element."""
    # careful: ion_stage is counted from 1 here, i.e. 1 <-> neutral, 2 <-> singly ionized

    # Sr II triplet
    if srII_triplet:
        lineindices = list(
            chain(
                at.linefluxes.get_closelines(
                    modelpath=modelpath,
                    atomic_number=38,
                    ion_stage=2,
                    approxlambdalabel=10917,  # 10 914.874 AA
                    lowerlevelindex=1,  # 4p6.4d 2D,enpercm=14555.9,j=1.5
                    upperlevelindex=3,
                )[3],  # 4p6.5p 2P,enpercm=23715.19,j=0.5
                at.linefluxes.get_closelines(
                    modelpath=modelpath,
                    atomic_number=38,
                    ion_stage=2,
                    approxlambdalabel=10330,  # 10 327.309 AA
                    lowerlevelindex=2,  # 4p6.4d 2D,enpercm=14836.24,j=2.5
                    upperlevelindex=4,
                )[3],  # 4p6.5p 2P,enpercm=24516.65,j=1.5
                at.linefluxes.get_closelines(
                    modelpath=modelpath,
                    atomic_number=38,
                    ion_stage=2,
                    approxlambdalabel=10039,  # 10 036.654 AA
                    lowerlevelindex=1,  # 4p6.4d 2D,enpercm=14555.9,j=1.5
                    upperlevelindex=4,
                )[3],  # 4p6.5p 2P,enpercm=24516.65,j=1.5
            )
        )
    elif ion_stage:
        lineindices = at.linefluxes.get_ion_linelist(modelpath=modelpath, atomic_number=Z, ion_stage=ion_stage)
    else:
        # all ionisation stages
        lineindices = list(
            chain.from_iterable(
                at.linefluxes.get_ion_linelist(modelpath=modelpath, atomic_number=Z, ion_stage=i) for i in range(1, 5)
            )
        )

    dfpackets_selected, _ = at.linefluxes.get_packets_with_emtype(
        modelpath=modelpath, emtypecolumn="absorption_type", lineindices=lineindices, maxpacketfiles=None
    )

    return dfpackets_selected


def get_red_packet_set(
    modelpath: Path,
    escape_angles: list[int],
    Z: int,
    ion_stage: int,
    wavelen: float | None = None,
    binwidth: float | None = None,
    srII_triplet: bool = False,
) -> pd.DataFrame:
    """Get packets in specific escape angle bins.

    Options:
        - all_packets=True  -> use all escaping packets
        - Z only            -> all ion stages of element
        - Z + ion_stage     -> specific ion
        - wavelen+binwidth  -> wavelength slice
    """
    dfpackets_selected = get_required_packets(modelpath, Z, ion_stage, srII_triplet=srII_triplet)

    # wavelength cut (if requested)
    if wavelen is not None and binwidth is not None:
        lam_min = wavelen - binwidth / 2
        lam_max = wavelen + binwidth / 2
        dfpackets_selected = dfpackets_selected[
            (dfpackets_selected["lambda_rf"] > lam_min) & (dfpackets_selected["lambda_rf"] < lam_max)
        ]

    dfpackets_selected = at.packets.add_derived_columns(
        dfpackets_selected, modelpath, ["emission_velocity", "angle_bin"]
    )
    return dfpackets_selected[dfpackets_selected["angle_bin"].isin(escape_angles)]


def packets_2d_hist_bin_and_ejecta_vel(
    modelpath: Path,
    tdays: float,
    srIItriplet: bool,
    dirbin_range: list[int] | None = None,
    Z: int | None = None,
    ion_stage: str | None = None,
    wavelen: float | None = None,
    binwidth: float | None = None,
) -> None:
    all_angles = False
    if not dirbin_range:
        # bolometric case, i.e. in all directions
        all_angles = True
    dirbin_range = [50, 59]
    all_packets = False
    if all(x is None for x in (Z, ion_stage)):
        all_packets = True

    arrow_angles = {0: 2.82, 90: 0.32, 40: 1.67, 50: 1.47}  # dirbin angle: angle of arrow in rad

    usecols = ["t_arrive_d", "em_time", "emission_velocity", "e_rf", "em_ejecta_angle_bin"]

    if all_packets:
        filename = f"dfpackets_dirbins{dirbin_range[0]}-{dirbin_range[1]}TYPE_ESCAPETYPE_RPKT.txt"
        dfpackets = pd.read_csv(modelpath / filename, sep=r"\s+", usecols=usecols)
        print(f"read packets data bins {dirbin_range[0]}-{dirbin_range[1]}")
        startoflabel = ""
        if all_angles:
            startoflabel = "all_angles_"
            for i, bin_number in enumerate(dirbin_range):
                dirbin_range[i] = bin_number + 10
            while dirbin_range[0] < 100:
                print(f"reading packets bins {dirbin_range[0]}-{dirbin_range[1]}")
                dfpackets_next = pd.read_csv(modelpath / filename, sep=r"\s+", usecols=usecols)
                dfpackets = pd.concat([dfpackets, dfpackets_next], axis=0, ignore_index=True)
                print(dfpackets)
                for i, bin_number in enumerate(dirbin_range):
                    dirbin_range[i] = bin_number + 10

            dfpackets["phi_bin"] = dfpackets["em_ejecta_angle_bin"] % 10
            dfpackets = dfpackets[dfpackets["phi_bin"] == 0]
            print(dfpackets)

            # wavelength cut (if requested)
            if wavelen is not None and binwidth is not None:
                lam_min = wavelen - binwidth / 2
                lam_max = wavelen + binwidth / 2
                dfpackets = dfpackets[(dfpackets["lambda_rf"] > lam_min) & (dfpackets["lambda_rf"] < lam_max)]

    else:
        dirbin = dirbin_range[0]
        escape_angles = list(range(dirbin, dirbin + 10))
        assert Z is not None
        assert ion_stage is not None
        ion_stage_int = at.decode_roman_numeral(ion_stage)
        dfpackets = get_red_packet_set(
            modelpath, escape_angles, Z, ion_stage_int, wavelen=wavelen, binwidth=binwidth, srII_triplet=srIItriplet
        )
        # at.packets.get_ejecta_angle_bin_where_packet_emitted(modelpath, dfpackets)
        print(dfpackets)
        if all_packets:
            startoflabel = "allpackets_"
        elif wavelen is not None:
            startoflabel = f"{wavelen:.0f}A_"
        elif Z is not None:
            if ion_stage is not None:
                startoflabel = f"{at.get_elsymbol(Z)}_ion{ion_stage}_"
            else:
                startoflabel = f"{at.get_elsymbol(Z)}_allions_"
        else:
            startoflabel = ""

    logscale = False

    mask_empty_cells = True

    packetsfiles = at.packets.get_packets_text_paths(modelpath)
    nprocs_read = len(packetsfiles)
    assert nprocs_read > 0

    startoflabel += "t_arrive_d"
    timeminarray = at.misc.get_timestep_times(modelpath=modelpath, loc="start")
    timemaxarray = at.misc.get_timestep_times(modelpath=modelpath, loc="end")
    print(timeminarray, timemaxarray)
    timestep_min = at.misc.get_timestep_of_timedays(modelpath, tdays)
    timestep_max = timestep_min + 1
    t_min = timeminarray[timestep_min]
    t_max = timemaxarray[timestep_max]

    dfpackets_selected = dfpackets[(dfpackets["t_arrive_d"] > t_min) & (dfpackets["t_arrive_d"] < t_max)]

    if "emission_velocity" not in dfpackets_selected:
        at.packets.add_derived_columns(dfpackets_selected, modelpath, ["emission_velocity"])

    print(f" between {timeminarray[timestep_min]}-{timemaxarray[timestep_max]} days")
    print(f"npackets = {len(dfpackets_selected)}")
    # print(f"sum e_rf dfpackets_selected {dfpackets_selected['e_rf'].sum()}")
    # print(f"luminosity from packets {(dfpackets_selected['e_rf'].sum() / nprocs_read * 10 / (timemaxarray[timestep_max] - timeminarray[timestep_min])) * (u.erg / u.day).to('erg/s')}")
    weight_by_energy = True
    if weight_by_energy:
        e_rf = dfpackets_selected["e_rf"]
        # e_cmf = dfpackets_selected["e_cmf"]
        weights = e_rf
        # weights = e_cmf
    else:
        weights = None

    heatmap, xedges, _ = np.histogram2d(
        dfpackets_selected["emission_velocity"] / CLIGHT,
        dfpackets_selected["em_ejecta_angle_bin"],
        bins=[np.linspace(0, 0.7, num=29), np.linspace(0, 100, num=11)],
        weights=weights,
    )
    heatmap = heatmap / nprocs_read * 10 / (timemaxarray[timestep_max] - timeminarray[timestep_min])  # erg/day
    heatmap /= DAY  # conversion from per erg/day to erg/s
    heatmap = np.log(heatmap) if logscale else heatmap / 1e40

    if mask_empty_cells:
        heatmap = np.ma.masked_where(heatmap == 0.0, heatmap)

    if TYPE_CHECKING:
        from matplotlib.projections.polar import PolarAxesSubplot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(4, 3.5))
    if TYPE_CHECKING:
        ax: PolarAxesSubplot
    ax = cast("PolarAxesSubplot", ax)

    costhetas = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    thetas = np.arccos(costhetas)

    radii = xedges
    z = heatmap

    im = ax.pcolormesh(thetas, radii, z, shading="auto")

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_offset(np.pi / 2)

    ax.set_xticks(thetas)
    ax.set_xticklabels(np.round(costhetas, 2))
    ax.grid(True)

    cbar = fig.colorbar(im)
    if logscale:
        cbar.set_label(r"log energy [erg/s]", rotation=90)
    else:
        cbar.set_label(r"Energy rate [10$^{40}$ erg/s]", rotation=90)

    ax.set_xlabel(r"Polar angle [cos($\theta$)]")
    ax.set_ylabel("Radial velocity [c]")
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_label_coords(0.9, 0.5)

    ax.annotate(
        "",
        xy=(arrow_angles[dirbin_range[0]], 0.7),
        xytext=(arrow_angles[dirbin_range[0]], 0.52),
        arrowprops={"facecolor": "black", "edgecolor": "white"},
    )

    outfilename = (
        startoflabel
        + f"escape_from_bins_ts{timestep_min}-{timestep_max}_into_dirbins{dirbin_range[0]}-{dirbin_range[1]}.pdf"
    )
    print(f"Saving {outfilename}")
    plt.savefig(Path(modelpath) / outfilename, dpi=300)
    plt.clf()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", required=True, help="Path to ARTIS simulation")

    parser.add_argument(
        "-tdays",
        type=float,
        required=True,
        help="Time in days, collects packets for the timestep in which the specified value lies in",
    )

    parser.add_argument("-wavelen", type=float, default=None, help="Central wavelength in Angstrom")
    parser.add_argument("-binwidth", type=float, default=None, help="Wavelength bin width in Angstrom")

    parser.add_argument("-element", type=int, default=None, help="Element symbol")
    parser.add_argument("-ionstage", type=str, default=None, help="Ionisation stage (spectroscopic notation)")

    parser.add_argument("-dir", type=str, default="eq", help="Viewing direction bin start (Options: eq, npol, spol)")
    parser.add_argument("--srIItriplet", action="store_true", help="Analyse SrII triplet in particular")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Comparison to constant beta decay splitup factors."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    match args.dir:
        case "eq":
            dirbin = 50
        case "npol":
            dirbin = 90
        case "spol":
            dirbin = 0
        case _:
            dirbin = None

    packets_2d_hist_bin_and_ejecta_vel(
        Path(args.modelpath),
        args.tdays,
        args.srIItriplet,
        dirbin_range=[dirbin, dirbin + 9] if dirbin else None,
        Z=at.get_atomic_number(args.element) if args.element else None,
        ion_stage=args.ionstage,
        wavelen=args.wavelen,
        binwidth=args.binwidth,
    )


if __name__ == "__main__":
    main()
