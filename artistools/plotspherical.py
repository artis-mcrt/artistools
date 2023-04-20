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
    outputfile: Union[Path, str],
    maxpacketfiles: Optional[int] = None,
) -> None:
    _, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=True, printwarningsonly=True)

    dfpackets: Union[pl.LazyFrame, pl.DataFrame]
    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if timemindays is not None:
        dfpackets = dfpackets.filter(pl.col("t_arrive_d") >= timemindays)
    if timemaxdays is not None:
        dfpackets = dfpackets.filter(pl.col("t_arrive_d") <= timemaxdays)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(8, 4),
        subplot_kw={"projection": "mollweide"},
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )

    # phi definition (with syn_dir=[0 0 1])
    # x=math.cos(-phi)
    # y=math.sin(-phi)

    ntheta = nphi = 64

    dfpackets = dfpackets.select(
        [
            "e_rf",
            "phi",
            "costheta",
        ]
    )  # .lazy()

    # print(dfpackets)
    data = np.zeros((ntheta, nphi))

    for x in range(nphi):
        phi_low = 2 * math.pi / nphi * x
        phi_high = 2 * math.pi / nphi * (x + 1)
        dfpackets_phibin = dfpackets.filter((pl.col("phi") >= phi_low) & (pl.col("phi") <= phi_high)).collect()
        for y in range(ntheta):
            costheta_low = 1 - 2 / ntheta * (y + 1)
            costheta_high = 1 - 2 / ntheta * y
            e_rf_sum = (
                dfpackets_phibin.filter((pl.col("costheta") >= costheta_low) & (pl.col("costheta") <= costheta_high))
                .get_column("e_rf")
                .sum()
            )
            data[y, x] = e_rf_sum

    # these phi and theta angle ranges are defined differently to artis
    phigrid = np.linspace(-np.pi, np.pi, nphi)

    costhetagrid = np.linspace(1, -1, ntheta)
    thetagrid = np.arccos(costhetagrid) - np.pi / 2
    # thetagrid = np.linspace(-np.pi / 2.0, np.pi / 2.0, ntheta)

    meshgrid_phi, meshgrid_theta = np.meshgrid(phigrid, thetagrid)
    ax.pcolormesh(meshgrid_phi, meshgrid_theta, data, rasterized=True)

    # finterp = interp2d(phigrid, thetagrid, data, kind="cubic")
    # phigrid_highres = np.linspace(-np.pi, np.pi, 256)
    # thetagrid_highres = np.linspace(-np.pi / 2.0, np.pi / 2.0, 128)
    # data1 = finterp(phigrid_highres, thetagrid_highres)
    # meshgrid_phi_highres, meshgrid_theta_highres = np.meshgrid(phigrid_highres, thetagrid_highres)
    # ax.pcolormesh(meshgrid_phi_highres, meshgrid_theta_highres, data1, rasterized=True)

    ax.xaxis.label.set_color("red")
    # ax.grid(True, color="white")

    fig.savefig(outputfile)
    print(f"Saved {outputfile}")
    # plt.show()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-tmin", action="store", type=float, default=None, help="Time minimum [d]")
    parser.add_argument("-tmax", action="store", type=float, default=None, help="Time maximum [d]")
    parser.add_argument(
        "-o",
        action="store",
        dest="outputfile",
        type=Path,
        default=Path("plotspherical.pdf"),
        help="Filename for PDF file",
    )
    parser.add_argument(
        "-modelpath",
        type=Path,
        default=Path(),
        help="Path to ARTIS folder",
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

    plot_spherical(modelpath=args.modelpath, timemindays=args.tmin, timemaxdays=args.tmax, outputfile=args.outputfile)


if __name__ == "__main__":
    main()
