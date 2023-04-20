#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import math
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
) -> None:
    _, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=True, printwarningsonly=True)

    dfpackets: Union[pl.LazyFrame, pl.DataFrame]
    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if timemindays is not None:
        dfpackets = dfpackets.filter(pl.col("t_arrive_d") >= timemindays)
    else:
        timemindays = float(dfpackets.select("t_arrive_d").collect().get_column("t_arrive_d").to_numpy().min())
        print(f"time min is {timemindays:.2f} d")

    if timemaxdays is not None:
        dfpackets = dfpackets.filter(pl.col("t_arrive_d") <= timemaxdays)
    else:
        timemaxdays = float(dfpackets.select("t_arrive_d").collect().get_column("t_arrive_d").to_numpy().max())
        print(f"time max is {timemaxdays:.2f} d")

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

    dfpackets = dfpackets.select(
        [
            "e_rf",
            "phi",
            "costheta",
        ]
    )  # .lazy()

    # print(dfpackets)
    e_rf_sumgrid = np.zeros((ncosthetabins, nphibins))

    for x in range(nphibins):
        phi_low = 2 * math.pi / nphibins * x
        phi_high = 2 * math.pi / nphibins * (x + 1)
        dfpackets_phibin = dfpackets.filter((pl.col("phi") >= phi_low) & (pl.col("phi") <= phi_high)).collect()
        for y in range(ncosthetabins):
            costheta_low = 1 - 2 / ncosthetabins * (y + 1)
            costheta_high = 1 - 2 / ncosthetabins * y
            e_rf_sum = (
                dfpackets_phibin.filter((pl.col("costheta") >= costheta_low) & (pl.col("costheta") <= costheta_high))
                .get_column("e_rf")
                .sum()
            )
            e_rf_sumgrid[y, x] = e_rf_sum

    solidanglefactor = nphibins * ncosthetabins
    data = e_rf_sumgrid / nprocs_read * solidanglefactor / (timemaxdays - timemindays) / 86400
    # these phi and theta angle ranges are defined differently to artis
    phigrid = np.linspace(-np.pi, np.pi, nphibins)

    costhetagrid = np.linspace(1, -1, ncosthetabins, endpoint=True)
    thetagrid = np.arccos(costhetagrid) - np.pi / 2
    # thetagrid = np.linspace(-np.pi / 2.0, np.pi / 2.0, ncosthetabins)
    # cmap = "rainbow"
    # cmap = "viridis"
    # cmap = "hot"
    # cmap = "Blues_r"
    cmap = None

    # import scipy.ndimage

    # data = scipy.ndimage.gaussian_filter(data, sigma=1)

    meshgrid_phi, meshgrid_theta = np.meshgrid(phigrid, thetagrid)
    if not interpolate:
        colormesh = ax.pcolormesh(meshgrid_phi, meshgrid_theta, data, rasterized=True, cmap=cmap)
    else:
        phigrid_highres = np.linspace(-np.pi, np.pi, 1024)
        thetagrid_highres = np.linspace(-np.pi / 2.0, np.pi / 2.0, 1024)
        from scipy.interpolate import CloughTocher2DInterpolator

        finterp = CloughTocher2DInterpolator(
            list(zip(meshgrid_phi.flatten(), meshgrid_theta.flatten())), data.flatten()
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
    )


if __name__ == "__main__":
    main()
