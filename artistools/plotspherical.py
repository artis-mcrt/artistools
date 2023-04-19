#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
from pathlib import Path
from typing import Optional
from typing import Union

import argcomplete
import matplotlib.pyplot as plt
import polars as pl

import artistools as at


def plot_spherical(
    modelpath: Union[str, Path],
    timemindays: Optional[float],
    timemaxdays: Optional[float],
    maxpacketfiles: Optional[int] = None,
) -> None:
    _, modelmeta = at.get_modeldata(modelpath=modelpath, getheadersonly=True, printwarningsonly=True)

    nprocs_read, dfpackets = at.packets.get_packets_pl(
        modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )

    if timemindays is not None:
        dfpackets = dfpackets.filter(pl.col("t_arrive_d") >= timemindays)
    if timemaxdays is not None:
        dfpackets = dfpackets.filter(pl.col("t_arrive_d") <= timemaxdays)
    dfpackets = dfpackets.select(["e_rf", "phibin", "costhetabin", "dirbin"]).collect(streaming=True).lazy()

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "mollweide"})

    test = dfpackets.groupby("dirbin").agg(pl.col("e_rf").sum().alias("e_rf_sum")).lazy().collect()
    print(test)

    ax.grid(True)

    plt.show()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-tmin", action="store", type=float, default=None, help="Time minimum [d]")
    parser.add_argument("-tmax", action="store", type=float, default=None, help="Time maximum [d]")
    parser.add_argument("-o", action="store", dest="outputfile", type=Path, default=None, help="Filename for PDF file")
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

    plot_spherical(modelpath=args.modelpath, timemindays=args.tmin, timemaxdays=args.tmax)


if __name__ == "__main__":
    main()
