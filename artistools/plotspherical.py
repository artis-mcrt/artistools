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

    dfpackets = dfpackets.select(
        [
            "e_rf",
            "theta",
            "phi",
        ]
    ).collect(
        streaming=True
    )  # .lazy()
    print(dfpackets)
    fig, ax = plt.subplots(
        1,
        1,
        # subplot_kw={"projection": "mollweide"},
    )

    # phi definition (with syn_dir=[0 0 1])
    # x=math.cos(-phi)
    # y=math.sin(-phi)

    long = dfpackets["phi"].to_numpy() - np.pi
    lat = dfpackets["theta"].to_numpy() - np.pi / 2

    ax.hist2d(
        long * 0.9,
        lat * 0.9,
        bins=300,
        weights=dfpackets["e_rf"].to_numpy(),
    )

    # test = dfpackets.groupby("dirbin").agg(pl.col("e_rf").sum().alias("e_rf_sum"))  # .lazy().collect()
    # nbins = 50
    # lon_edges = np.linspace(-np.pi, np.pi, nbins + 1)
    # lat_edges = np.linspace(-np.pi / 2.0, np.pi / 2.0, nbins + 1)

    # ax.grid(True)

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
        dest="modelpath",
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
