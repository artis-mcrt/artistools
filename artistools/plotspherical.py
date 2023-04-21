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
    interpolate: bool = False,
    gaussian_sigma: Optional[int] = None,
) -> None:
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
            print(f"setting timemin to start of valid observable range {tmin_d_valid:.2f} d")
            timemindays = tmin_d_valid
        elif timemindays < tmin_d_valid:
            print(
                f"WARNING! timemindays {timemindays} is too early for light to travel from the entire ejecta "
                f" ({tmin_d_valid} d)"
            )

        if timemaxdays is None:
            print(f"setting timemin to end of valid observable range {tmax_d_valid:.2f} d")
            timemaxdays = tmax_d_valid
        elif timemaxdays > tmax_d_valid:
            print(
                f"WARNING! timemaxdays {timemaxdays} is too late to recieve light from the entire ejecta "
                f" ({tmin_d_valid} d)"
            )
        dfpackets = dfpackets.filter((pl.col("t_arrive_d") >= timemindays) & (pl.col("t_arrive_d") <= timemaxdays))
    assert timemindays is not None
    assert timemaxdays is not None

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(8, 4),
        subplot_kw={"projection": "mollweide"},
        tight_layout={"pad": 0.1, "w_pad": 0.0, "h_pad": 0.0},
    )

    # phi definition (with syn_dir=[0 0 1])
    # x=math.cos(-phi)
    # y=math.sin(-phi)

    dfpackets = at.packets.bin_packet_directions_lazypolars(
        modelpath=modelpath, dfpackets=dfpackets, nphibins=nphibins, nthetabins=ncosthetabins, phibintype="monotonic"
    )
    dfpackets = dfpackets.groupby("dirbin").agg(pl.col("e_rf").sum())

    alldirbins = pl.DataFrame([pl.Series("dirbin", np.arange(0, nphibins * ncosthetabins), dtype=pl.Int32)])
    alldirbins = alldirbins.join(dfpackets.collect(), how="left", on="dirbin").fill_null(0)
    e_rf_sumgrid = alldirbins.get_column("e_rf").to_numpy().reshape((ncosthetabins, nphibins))

    solidanglefactor = nphibins * ncosthetabins
    data = e_rf_sumgrid / nprocs_read * solidanglefactor / (timemaxdays - timemindays) / 86400
    # these phi and theta angle ranges are defined differently to artis
    phigrid = np.linspace(-np.pi, np.pi, nphibins + 1, endpoint=True)

    # costhetabin zero is (0,0,-1) so theta angle
    costhetagrid = np.linspace(-1, 1, ncosthetabins + 1, endpoint=True)
    # for Molleweide projection, theta range is [-pi/2, +pi/2]
    thetagrid = np.arccos(costhetagrid) - np.pi / 2

    # cmap = "rainbow"
    # cmap = "viridis"
    # cmap = "hot"
    # cmap = "Blues_r"
    cmap = None

    if gaussian_sigma is not None and gaussian_sigma > 0:
        import scipy.ndimage

        data = scipy.ndimage.gaussian_filter(data, sigma=gaussian_sigma, mode="wrap")

    meshgrid_phi, meshgrid_theta = np.meshgrid(phigrid, thetagrid)
    if not interpolate:
        colormesh = ax.pcolormesh(meshgrid_phi, meshgrid_theta, data, rasterized=True, cmap=cmap)
    else:
        ngridhighres = 1024
        print(f"interpolating onto {ngridhighres}^2 grid")

        phigrid_highres = np.linspace(-np.pi, np.pi, ngridhighres + 1, endpoint=True)
        thetagrid_highres = np.linspace(-np.pi / 2.0, np.pi / 2.0, ngridhighres + 1, endpoint=True)

        meshgrid_phi_highres_noendpoint, meshgrid_theta_highres_noendpoint = np.meshgrid(
            phigrid_highres[:-1], thetagrid_highres[:-1]
        )

        from scipy.interpolate import CloughTocher2DInterpolator

        meshgrid_phi_noendpoint, meshgrid_theta_noendpoint = np.meshgrid(phigrid[:-1], thetagrid[:-1])
        finterp = CloughTocher2DInterpolator(
            list(zip(meshgrid_phi_noendpoint.flatten(), meshgrid_theta_noendpoint.flatten())), data.flatten()
        )

        meshgrid_phi_highres, meshgrid_theta_highres = np.meshgrid(phigrid_highres, thetagrid_highres)
        data_interp = finterp(meshgrid_phi_highres, meshgrid_theta_highres)

        colormesh = ax.pcolormesh(meshgrid_phi_highres, meshgrid_theta_highres, data_interp, rasterized=True, cmap=cmap)

    cbar = fig.colorbar(colormesh)
    cbar.ax.set_title(r"$I_{e,\Omega}\cdot4\pi/\Omega$ [erg/s]")

    # ax.set_xlabel("Azimuthal angle")
    # ax.set_ylabel("Polar angle")
    # ax.tick_params(colors="white", axis="x", which="both")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(True)

    fig.savefig(outputfile)
    print(f"Saved {outputfile}")
    # plt.show()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath",
        type=Path,
        default=Path(),
        help="Path to ARTIS folder",
    )
    parser.add_argument("-timemin", action="store", type=float, default=None, help="Time minimum [d]")
    parser.add_argument("-timemax", action="store", type=float, default=None, help="Time maximum [d]")
    parser.add_argument("-nphibins", action="store", type=int, default=32, help="Number of azimuthal bins")
    parser.add_argument("-ncosthetabins", action="store", type=int, default=32, help="Number of polar angle bins")
    parser.add_argument("-maxpacketfiles", type=int, default=None, help="Limit the number of packet files read")
    parser.add_argument("-gaussian_sigma", type=int, default=None, help="Apply Gaussian filter")

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
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter, description="Plot ARTIS input model composition"
        )
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = ["."]

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
    )


if __name__ == "__main__":
    main()
