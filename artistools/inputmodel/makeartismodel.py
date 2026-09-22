# PYTHON_ARGCOMPLETE_OK
"""Build an ARTIS input model by downscaling, dimension-reducing, or rescaling an existing model."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import Msun_to_g
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import dimension_reduce_model
from artistools.inputmodel.core import get_initelemabundances
from artistools.inputmodel.core import get_modeldata
from artistools.inputmodel.core import save_initelemabundances
from artistools.inputmodel.core import save_modeldata
from artistools.inputmodel.downscale3dgrid import make_downscaled_3d_grid
from artistools.inputmodel.energyinputfiles import make_energy_files
from artistools.inputmodel.modelfromhydro import makemodelfromgriddata
from artistools.inputmodel.rprocess_from_trajectory import get_gridparticlecontributions_or_none
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, multiplepaths=True, default=[], helptext="Path to input model file")

    parser.add_argument(
        "--downscale3dgrid", action="store_true", help="Downscale a 3D ARTIS model to smaller grid size"
    )

    parser.add_argument(
        "--downscaleplot", action="store_true", help="Write a density-slice diagnostic plot when downscaling"
    )

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
        "--makeenergyinputfiles", action="store_true", help="Write energydistribution.txt and energyrate.txt files"
    )

    addarg_output(parser, kind="folder", helptext="Folder for output", default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Tools to create an ARTIS input model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpath_given = bool(args.modelpath)
    args.modelpath = normalize_path_list(args.modelpath)

    if args.downscale3dgrid:
        # -o holds the working folder when the user gave none, and the default output folder is a
        # subfolder of the model
        outputfolder = args.outputfile if Path(args.outputfile) != Path() else None
        make_downscaled_3d_grid(
            modelpath=Path(args.modelpath[0]),
            outputgridsize=args.outputgridsize,
            plot=args.downscaleplot,
            outputfolder=outputfolder,
        )
        return

    if args.dimensionreduce is not None:
        ndim_out = args.dimensionreduce
        assert ndim_out in {0, 1, 2}
        for modelpath in args.modelpath:
            dfmodel, modelmeta = get_modeldata(modelpath)
            ndim_in = modelmeta["dimensions"]
            if ndim_in <= ndim_out:
                msg = f"Cannot reduce {ndim_in}D model to {ndim_out}D"
                raise ValueError(msg)

            dfelabundances = get_initelemabundances(modelpath)
            dfgridcontributions = get_gridparticlecontributions_or_none(modelpath)

            (dfmodel_out, dfelabundances_out, _, modelmeta_out) = dimension_reduce_model(
                dfmodel=dfmodel,
                outputdimensions=ndim_out,
                dfelabundances=dfelabundances,
                dfgridcontributions=dfgridcontributions,
                modelmeta=modelmeta,
            )
            # the name of the model is part of the folder, thus each model path writes a different folder
            outdir = (
                resolve_outputfile(args.outputfile, "model.txt").parent
                / f"{Path(modelpath).resolve().name}_dimreduce_{ndim_out}d"
            )
            outdir.mkdir(exist_ok=True, parents=True)
            modelmeta_out["headercommentlines"] = [
                *modelmeta.get("headercommentlines", []),
                f"Dimension reduced from {ndim_in}-dimensional model",
            ]
            assert dfelabundances_out is not None
            save_initelemabundances(dfelabundances_out, outpath=outdir)
            save_modeldata(dfmodel=dfmodel_out, modelmeta=modelmeta_out, outpath=outdir)

    if args.makemodelfromgriddata:
        print(args)
        # before the -o argument existed, -modelpath gave the output folder. The command keeps that
        # behaviour when the command line holds no -o.
        outputpath = args.outputfile
        if Path(args.outputfile) == Path() and modelpath_given:
            outputpath = args.modelpath[0]
            print_warning(f"-modelpath sets the output folder to {outputpath}. Use -o for the output folder.")
        makemodelfromgriddata(
            gridfolderpath=args.pathtogriddata,
            outputpath=outputpath,
            fillcentralhole=args.fillcentralhole,
            getcellopacityfromYe=args.getcellopacityfromYe,
        )

    if args.makeenergyinputfiles:
        plmodel, modelmeta = get_modeldata(args.modelpath[0])
        model = add_derived_cols_to_modeldata(plmodel, modelmeta=modelmeta).select("rho", "mass_g").collect()
        rho = model["rho"].cast(pl.Float64).to_numpy()
        Mtot_grams = float(model["mass_g"].sum())

        print(f"total mass {Mtot_grams / Msun_to_g} Msun")

        make_energy_files(rho, Mtot_grams, outputpath=args.outputfile)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
