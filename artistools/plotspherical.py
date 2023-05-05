#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
from pathlib import Path
from typing import Optional
from typing import Union

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at


def plot_spherical(
    modelpath: Union[str, Path],
    timemindays: Optional[float],
    timemaxdays: Optional[float],
    nphibins: int,
    ncosthetabins: int,
    outputfile: Union[Path, str],
    maxpacketfiles: Optional[int] = None,
    atomic_number: Optional[int] = None,
    ion_stage: Optional[int] = None,
    interpolate: bool = False,
    gaussian_sigma: Optional[int] = None,
    plotvars: Optional[list[str]] = None,
    cmap: Optional[str] = None,
) -> None:
    if plotvars is None:
        plotvars = ["luminosity", "emvelocityoverc", "emlosvelocityoverc"]

    _, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=True, printwarningsonly=True)

    dfpackets: Union[pl.LazyFrame, pl.DataFrame]
    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    _, tmin_d_valid, tmax_d_valid = at.get_escaped_arrivalrange(modelpath)
    if tmin_d_valid is None or tmax_d_valid is None:
        print("WARNING! The observer never gets light from the entire ejecta. Plotting all packets anyway")
        timemindays, timemaxdays = (
            dfpackets.select(pl.col("t_arrive_d").min().alias("tmin"), pl.col("t_arrive_d").max().alias("tmax"))
            .collect()
            .to_numpy()[0]
        )
    else:
        if timemindays is None:
            print(f"setting timemindays to start of valid observable range {tmin_d_valid:.2f} d")
            timemindays = tmin_d_valid
        elif timemindays < tmin_d_valid:
            print(
                f"WARNING! timemindays {timemindays} is too early for light to travel from the entire ejecta "
                f" ({tmin_d_valid:.2f} d)"
            )

        if timemaxdays is None:
            print(f"setting timemaxdays to end of valid observable range {tmax_d_valid:.2f} d")
            timemaxdays = tmax_d_valid
        elif timemaxdays > tmax_d_valid:
            print(
                f"WARNING! timemaxdays {timemaxdays} is too late to recieve light from the entire ejecta "
                f" ({tmax_d_valid:.2f} d)"
            )
        dfpackets = dfpackets.filter((pl.col("t_arrive_d") >= timemindays) & (pl.col("t_arrive_d") <= timemaxdays))

    assert timemindays is not None
    assert timemaxdays is not None

    fig, axes = plt.subplots(
        len(plotvars),
        1,
        figsize=(4, 3 * len(plotvars)),
        subplot_kw={"projection": "mollweide"},
        tight_layout={"pad": 0.1, "w_pad": 1.5, "h_pad": 0.0},
    )

    if len(plotvars) == 1:
        axes = [axes]

    # phi definition (with syn_dir=[0 0 1])
    # x=math.cos(-phi)
    # y=math.sin(-phi)

    dfpackets = at.packets.bin_packet_directions_lazypolars(
        dfpackets=dfpackets, nphibins=nphibins, ncosthetabins=ncosthetabins, phibintype="monotonic"
    )

    # for figuring out where the axes are on the plot, make a cut
    # dfpackets = dfpackets.filter(pl.col("dirz") < -0.9)

    solidanglefactor = nphibins * ncosthetabins
    aggs = []
    dfpackets = at.packets.add_derived_columns_lazy(dfpackets)
    if "emvelocityoverc" in plotvars:
        aggs.append(
            ((pl.col("emission_velocity") * pl.col("e_rf")).mean() / pl.col("e_rf").mean() / 29979245800).alias(
                "emvelocityoverc"
            )
        )

    if "emlosvelocityoverc" in plotvars:
        aggs.append(
            (
                (pl.col("emission_velocity_lineofsight") * pl.col("e_rf")).mean() / pl.col("e_rf").mean() / 29979245800
            ).alias("emlosvelocityoverc")
        )

    if "luminosity" in plotvars:
        aggs.append(
            (pl.col("e_rf").sum() / nprocs_read * solidanglefactor / (timemaxdays - timemindays) / 86400).alias(
                "luminosity"
            )
        )

    if atomic_number is not None or ion_stage is not None:
        dflinelist = at.get_linelist_pldf(modelpath)
        if atomic_number is not None:
            print(f"Including only packets emitted by Z={atomic_number} {at.get_elsymbol(atomic_number)}")
            dflinelist = dflinelist.filter(pl.col("atomic_number") == atomic_number)
        if ion_stage is not None:
            print(f"Including only packets emitted by ionisation stage {ion_stage}")
            dflinelist = dflinelist.filter(pl.col("ion_stage") == ion_stage)

        selected_emtypes = dflinelist.select("lineindex").collect().get_column("lineindex")
        dfpackets = dfpackets.filter(pl.col("emissiontype").is_in(selected_emtypes))

    aggs.append(pl.count())

    dfpackets = dfpackets.groupby(["costhetabin", "phibin"]).agg(aggs)
    dfpackets = dfpackets.select(["costhetabin", "phibin", "count", *plotvars])

    ndirbins = nphibins * ncosthetabins
    alldirbins = pl.DataFrame(
        {"phibin": (d % nphibins for d in range(ndirbins)), "costhetabin": (d // nphibins for d in range(ndirbins))}
    ).with_columns(pl.all().cast(pl.Int32))
    alldirbins = (
        alldirbins.join(
            dfpackets.collect(),
            how="left",
            on=["costhetabin", "phibin"],
        )
        .fill_null(0)
        .sort(["costhetabin", "phibin"])
    )

    print(f'total packets contributed: {alldirbins.select("count").sum().to_numpy()[0][0]:.1e}')

    # these phi and theta angle ranges are defined differently to artis
    phigrid = np.linspace(-np.pi, np.pi, nphibins + 1, endpoint=True)

    # costhetabin zero is (0,0,-1) so theta angle
    costhetagrid = np.linspace(-1, 1, ncosthetabins + 1, endpoint=True)
    # for Molleweide projection, theta range is [-pi/2, +pi/2]
    thetagrid = np.arccos(costhetagrid) - np.pi / 2

    meshgrid_phi, meshgrid_theta = np.meshgrid(phigrid, thetagrid)

    for ax, plotvar in zip(axes, plotvars):
        data = alldirbins.get_column(plotvar).to_numpy().reshape((ncosthetabins, nphibins))

        if gaussian_sigma is not None and gaussian_sigma > 0:
            import scipy.ndimage

            sigma_bins = gaussian_sigma / 360 * nphibins
            data = scipy.ndimage.gaussian_filter(data, sigma=sigma_bins, mode="wrap")

        if not interpolate:
            colormesh = ax.pcolormesh(meshgrid_phi, meshgrid_theta, data, rasterized=True, cmap=cmap)
        else:
            ngridhighres = 1024
            print(f"interpolating onto {ngridhighres}^2 grid")

            from scipy.interpolate import CloughTocher2DInterpolator

            meshgrid_phi_input, meshgrid_theta_input = np.meshgrid(
                np.linspace(-np.pi, np.pi, nphibins, endpoint=True),
                np.arccos(np.linspace(-1, 1, ncosthetabins, endpoint=True)) - np.pi / 2,
            )
            finterp = CloughTocher2DInterpolator(
                list(zip(meshgrid_phi_input.flatten(), meshgrid_theta_input.flatten())), data.flatten()
            )

            meshgrid_phi_highres, meshgrid_theta_highres = np.meshgrid(
                np.linspace(-np.pi, np.pi, ngridhighres + 1, endpoint=True),
                np.linspace(-np.pi / 2.0, np.pi / 2.0, ngridhighres + 1, endpoint=True),
            )
            data_interp = finterp(meshgrid_phi_highres, meshgrid_theta_highres)

            colormesh = ax.pcolormesh(
                meshgrid_phi_highres, meshgrid_theta_highres, data_interp, rasterized=True, cmap=cmap
            )

        if plotvar == "emvelocityoverc":
            colorbartitle = r"Last interaction ejecta velocity [c]"
        elif plotvar == "emlosvelocityoverc":
            colorbartitle = r"Mean line of sight velocity [c]"
        elif plotvar == "luminosity":
            colorbartitle = r"$I_{e,\Omega}\cdot4\pi/\Omega$ [erg/s]"
        else:
            raise AssertionError

        cbar = fig.colorbar(colormesh, location="bottom")
        cbar.ax.set_title(colorbartitle)
        cbar.outline.set_linewidth(0)

        # ax.set_xlabel("Azimuthal angle")
        # ax.set_ylabel("Polar angle")
        # ax.tick_params(colors="white", axis="x", which="both")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.grid(True)
        ax.axis("off")

    fig.savefig(outputfile)
    print(f"Saved {outputfile}")


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath",
        type=Path,
        default=Path(),
        help="Path to ARTIS folder",
    )
    parser.add_argument("-timemin", "-tmin", action="store", type=float, default=None, help="Time minimum [d]")
    parser.add_argument("-timemax", "-tmax", action="store", type=float, default=None, help="Time maximum [d]")
    parser.add_argument("-nphibins", action="store", type=int, default=64, help="Number of azimuthal bins")
    parser.add_argument("-ncosthetabins", action="store", type=int, default=32, help="Number of polar angle bins")
    parser.add_argument("-maxpacketfiles", type=int, default=None, help="Limit the number of packet files read")
    parser.add_argument("-gaussian_sigma", type=int, default=None, help="Apply Gaussian filter")
    parser.add_argument(
        "-plotvars",
        default=["luminosity", "emvelocityoverc", "emlosvelocityoverc"],
        choices=["luminosity", "emvelocityoverc", "emlosvelocityoverc"],
        nargs="+",
        help="Variable to plot: luminosity, emvelocityoverc, emlosvelocityoverc",
    )
    parser.add_argument("-elem", type=str, default=None, help="Filter emitted packets by element of last emission")
    parser.add_argument(
        "-atomic_number", type=int, default=None, help="Filter emitted packets by element of last emission"
    )
    parser.add_argument(
        "-ion_stage", type=int, default=None, help="Filter emitted packets by ionistion stage of last emission"
    )
    parser.add_argument("-cmap", default=None, type=str, help="Matplotlib color map name")

    parser.add_argument("--interpolate", action="store_true", help="Interpolate grid to higher resolution")

    parser.add_argument(
        "-o",
        action="store",
        dest="outputfile",
        type=Path,
        default=Path("plotspherical.pdf"),
        help="Filename for PDF file",
    )


def main(args=None, argsraw=None, **kwargs) -> None:
    """Plot direction maps based on escaped packets."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = ["."]

    if args.elem is not None:
        assert args.atomic_number is None
        args.atomic_number = at.get_atomic_number(args.elem)

    plot_spherical(
        modelpath=args.modelpath,
        timemindays=args.timemin,
        timemaxdays=args.timemax,
        nphibins=args.nphibins,
        ncosthetabins=args.ncosthetabins,
        maxpacketfiles=args.maxpacketfiles,
        outputfile=args.outputfile,
        interpolate=args.interpolate,
        gaussian_sigma=args.gaussian_sigma,
        atomic_number=args.atomic_number,
        ion_stage=args.ion_stage,
        plotvars=args.plotvars,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
