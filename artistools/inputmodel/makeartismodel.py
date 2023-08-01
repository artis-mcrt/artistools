#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
from pathlib import Path

import argcomplete

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", default=[], nargs="*", action=at.AppendPath, help="Path to initial model file")

    parser.add_argument(
        "--downscale3dgrid", action="store_true", help="Downscale a 3D ARTIS model to smaller grid size"
    )

    parser.add_argument("-inputgridsize", default=200, type=int, help="Size of big model grid for downscale script")

    parser.add_argument("-outputgridsize", default=50, type=int, help="Size of small model grid for downscale script")

    parser.add_argument(
        "--makemodelfromgriddata", action="store_true", help="Make ARTIS model files from arepo grid.dat file"
    )

    parser.add_argument("-pathtogriddata", default=".", help="Path to arepo grid.dat file")

    parser.add_argument(
        "--fillcentralhole", action="store_true", help="Fill hole in middle of ejecta from arepo kilonova model"
    )

    parser.add_argument(
        "--getcellopacityfromYe",
        action="store_true",
        help="Make opacity.txt where opacity is set in each cell by Ye from arepo model",
    )

    parser.add_argument(
        "--makeenergyinputfiles", action="store_true", help="Downscale a 3D ARTIS model to smaller grid size"
    )

    parser.add_argument(
        "-modeldim", type=int, default=None, help="Choose how many dimensions input model has. 1, 2 or 3"
    )

    parser.add_argument("-outputpath", "-o", default=".", help="Folder for output")


def main(args=None, argsraw=None, **kwargs):
    """Tools to create an ARTIS input model."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = [Path()]
    elif isinstance(args.modelpath, str | Path):
        args.modelpath = [args.modelpath]

    args.modelpath = at.flatten_list(args.modelpath)

    if args.downscale3dgrid:
        at.inputmodel.downscale3dgrid.make_downscaled_3d_grid(
            modelpath=Path(args.modelpath[0]), outputgridsize=args.outputgridsize
        )
        return

    if args.makemodelfromgriddata:
        print(args)
        at.inputmodel.modelfromhydro.makemodelfromgriddata(
            gridfolderpath=args.pathtogriddata, outputpath=args.modelpath[0], args=args
        )

    if args.makeenergyinputfiles:
        model, _ = at.inputmodel.get_modeldata(args.modelpath[0])
        rho = 10 ** model["logrho"] if args.modeldim == 1 else model["rho"]
        Mtot_grams = model["cellmass_grams"].sum()

        print(f"total mass {Mtot_grams / 1.989e33} Msun")

        at.inputmodel.energyinputfiles.make_energy_files(rho, Mtot_grams, outputpath=args.outputpath)


if __name__ == "__main__":
    main()
