#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", default=[], nargs="*", action=at.AppendPath, help="Path to input model file")

    parser.add_argument(
        "--downscale3dgrid", action="store_true", help="Downscale a 3D ARTIS model to smaller grid size"
    )

    parser.add_argument("-inputgridsize", default=200, type=int, help="Size of big model grid for downscale script")

    parser.add_argument("-outputgridsize", default=50, type=int, help="Size of small model grid for downscale script")

    parser.add_argument(
        "-dimensionreduce",
        "-d",
        default=None,
        type=int,
        help="Number of dimensions: 0 for one-zone, 1 for spherically symmetric 1D, 2 for 2D Cylindrical",
    )

    parser.add_argument(
        "--makemodelfromgriddata", action="store_true", help="Make ARTIS model files from SPH grid.dat file"
    )

    parser.add_argument("-pathtogriddata", default=".", help="Path to SPH grid.dat file")

    parser.add_argument(
        "--fillcentralhole", action="store_true", help="Fill hole in middle of ejecta from SPH kilonova model"
    )

    parser.add_argument(
        "--getcellopacityfromYe",
        action="store_true",
        help="Make opacity.txt where opacity is set in each cell by Ye from SPH model",
    )

    parser.add_argument(
        "--makeenergyinputfiles", action="store_true", help="Downscale a 3D ARTIS model to smaller grid size"
    )

    parser.add_argument("-outputpath", "-o", default=".", help="Folder for output")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Tools to create an ARTIS input model."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

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

    if args.dimensionreduce is not None:
        ndim_out = args.dimensionreduce
        assert ndim_out in {0, 1, 2}
        for modelpath in args.modelpath:
            dfmodel, modelmeta = at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g"])
            ndim_in = modelmeta["dimensions"]
            if ndim_in <= ndim_out:
                msg = f"Cannot reduce {ndim_in}D model to {ndim_out}D"
                raise ValueError(msg)

            dfelabundances = at.inputmodel.get_initelemabundances_polars(modelpath)
            dfgridcontributions = (
                at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(modelpath)
                if Path(modelpath, "gridcontributions.txt").is_file()
                else None
            )

            (dfmodel_out, dfelabundances_out, _, modelmeta_out) = at.inputmodel.dimension_reduce_model(
                dfmodel=dfmodel,
                outputdimensions=ndim_out,
                dfelabundances=dfelabundances,
                dfgridcontributions=dfgridcontributions,
                modelmeta=modelmeta,
            )
            outdir = (
                Path(args.outputpath) if Path(args.outputpath).is_dir() else Path(args.outputpath).parent
            ) / f"dimreduce_{ndim_out}d"
            outdir.mkdir(exist_ok=True, parents=True)
            modelmeta_out["headercommentlines"] = [
                *modelmeta.get("headercommentlines", []),
                f"Dimension reduced from {ndim_in}-dimensional model",
            ]
            assert dfelabundances_out is not None
            at.inputmodel.save_initelemabundances(dfelabundances_out, outpath=outdir)
            at.inputmodel.save_modeldata(dfmodel=dfmodel_out, modelmeta=modelmeta_out, outpath=outdir)

    if args.makemodelfromgriddata:
        print(args)
        at.inputmodel.modelfromhydro.makemodelfromgriddata(
            gridfolderpath=args.pathtogriddata, outputpath=args.modelpath[0], args=args
        )

    if args.makeenergyinputfiles:
        model, modelmeta = at.inputmodel.get_modeldata_pandas(args.modelpath[0], derived_cols=["mass_g"])
        rho = (10 ** model["logrho"] if modelmeta["dimensions"] == 1 else model["rho"]).to_numpy()
        Mtot_grams = model["mass_g"].sum()

        print(f"total mass {Mtot_grams / 1.989e33} Msun")

        at.inputmodel.energyinputfiles.make_energy_files(rho, Mtot_grams, outputpath=args.outputpath)


if __name__ == "__main__":
    main()
