"""Build an ARTIS input model by downscaling, dimension-reducing, or rescaling an existing model."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import Msun_to_g
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import dimension_reduce_model
from artistools.inputmodel.core import get_model_text_folder
from artistools.inputmodel.core import MODEL_FILE_NAMES
from artistools.inputmodel.core import save_initelemabundances
from artistools.inputmodel.core import save_modeldata
from artistools.inputmodel.core import scan_initelemabundances
from artistools.inputmodel.core import scan_modeldata
from artistools.inputmodel.core import with_cell_ids_from_zero
from artistools.inputmodel.downscale3dgrid import get_downscale_outputfolder
from artistools.inputmodel.downscale3dgrid import make_downscaled_3d_grid
from artistools.inputmodel.energyinputfiles import ENERGY_FILE_NAMES
from artistools.inputmodel.energyinputfiles import make_energy_files
from artistools.inputmodel.modelfromhydro import get_griddata_output_file_names
from artistools.inputmodel.modelfromhydro import makemodelfromgriddata
from artistools.inputmodel.rprocess_from_trajectory import get_gridparticlecontributions_or_none
from artistools.inputmodel.rprocess_from_trajectory import save_gridparticlecontributions
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import get_model_coordinates
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.misc import SPHERICAL_EDGE_COORDINATES


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[], helptext="Paths of the input models")

    parser.add_argument(
        "--downscale3dgrid", action="store_true", help="Downscale a 3D ARTIS model to smaller grid size"
    )

    parser.add_argument(
        "--downscaleplot", action="store_true", help="Write a density-slice diagnostic plot when downscaling"
    )
    addarg_darkmode(parser)
    addarg_show(parser)

    parser.add_argument("-outputgridsize", default=50, type=int, help="Size of small model grid for downscale script")

    parser.add_argument(
        "-dimensionreduce",
        "-d",
        default=None,
        type=int,
        choices=[0, 1, 2],
        help="The number of dimensions: 0 for one-zone, 1 for spherically symmetric 1D, and 2 for 2D. A 3D Cartesian"
        " model gives a 2D cylindrical model, and a 3D spherical model gives a 2D spherical model",
    )

    parser.add_argument(
        "--format-version",
        type=int,
        choices=[1, 2],
        default=None,
        help="The format version of the model.txt that the command writes. Without a different action, the command"
        " converts the model.txt of the model path, in the model folder or in the -o folder. With --dimensionreduce,"
        " the reduced model gets this version. Version 2 starts with the line 'format_version 2' and has a header"
        " of keys with the cell edges. An sn3d that is older than version 2 does not read it. A 2D spherical or 3D"
        " spherical model has version 2 only",
    )

    parser.add_argument(
        "--makemodelfromgriddata", action="store_true", help="Make ARTIS model files from SPH grid.dat file"
    )

    parser.add_argument("-pathtogriddata", default=".", help="The folder that holds the SPH grid.dat file")

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

    addarg_output(parser, kind="folder", helptext="The folder for output")
    addarg_force(parser)


def get_griddata_outputfolder(outputfile: Path | None, modelpaths: Sequence[Path], modelpath_given: bool) -> Path:
    """Return the output folder of --makemodelfromgriddata.

    Before the -o argument existed, --modelpath gave the output folder. The command keeps that behaviour when the
    command line holds no -o.
    """
    if outputfile is not None:
        return outputfile

    if modelpath_given:
        print_warning(f"--modelpath sets the output folder to {modelpaths[0]}. Use -o for the output folder.")
        return Path(modelpaths[0])

    return Path()


def get_dimreduce_outputfolder(outputfile: Path | None, modelpath: Path, ndim_out: int) -> Path:
    """Return the output folder of a dimension reduction, which holds the name of the model.

    Thus each model path writes a different folder.
    """
    return (
        resolve_outputfile(outputfile, "model.txt").parent / f"{Path(modelpath).resolve().name}_dimreduce_{ndim_out}d"
    )


def convert_model_format(modelpath: Path, outputfolder: Path, formatversion: int) -> None:
    """Write the model.txt of a model again in a format version, into outputfolder.

    The cells keep their sequence and their values to the Float32 precision that artistools reads. Version 1 gives
    the position of each cell in columns, and version 2 gives the cell edges in the header.
    """
    lzmodel, modelmeta = scan_modeldata(modelpath)
    coordinates = get_model_coordinates(modelmeta)
    if formatversion == 1 and coordinates in SPHERICAL_EDGE_COORDINATES:
        exit_with_error(
            f"Format version 1 of model.txt has no {coordinates} coordinates",
            "A 2D spherical or 3D spherical model has format version 2 only",
        )
    print(
        f"The model.txt of {modelpath} has format version {modelmeta['formatversion']} and {coordinates} coordinates."
        f" The command writes it in format version {formatversion}."
    )
    if formatversion == 2:
        print("  Version 2 gives the cell edges as velocities [km/s] in its header, and each cell has rho [g/cm3].")
    else:
        print("  Version 1 gives the position of each cell in columns. A 1D cell has log10(rho [g/cm3]).")
    print("  Each value has the Float32 precision that artistools reads, which is 8 significant figures.")

    # save_modeldata can write the file that the query scans, thus the read is complete before the write
    dfmodel = lzmodel.collect()
    gridcolumns = {"inputcellid", "modelgridindex", "rho", "logrho"}
    customcols = [col for col in dfmodel.columns if col not in gridcolumns and not col.startswith(("pos_", "vel_"))]
    outputfolder.mkdir(exist_ok=True, parents=True)

    if formatversion == 2 and dfmodel["inputcellid"].min() != 0:
        # each file of a model has the same cell ids, and version 2 has cell ids from 0
        print("  Version 2 has cell ids from 0, thus abundances.txt and gridcontributions.txt get the same ids.")
        hasabundances = firstexisting_or_none("abundances.txt", folder=modelpath, tryzipped=True) is not None
        dfmodel, dfelabundances, dfgridcontributions = with_cell_ids_from_zero(
            dfmodel,
            scan_initelemabundances(modelpath).collect() if hasabundances else None,
            get_gridparticlecontributions_or_none(modelpath),
        )
        if dfelabundances is not None:
            save_initelemabundances(dfelabundances, outpath=outputfolder)
        if dfgridcontributions is not None:
            save_gridparticlecontributions(dfgridcontributions, outputfolder / "gridcontributions.txt")

    save_modeldata(
        dfmodel, outpath=outputfolder, modelmeta=modelmeta | {"formatversion": formatversion}, extracols=customcols
    )


def get_converted_file_names(modelpaths: Sequence[Path], formatversion: int | None) -> Sequence[str]:
    """Return the names of the files that convert_model_format writes for these models.

    A conversion to version 2 also writes abundances.txt and gridcontributions.txt if the cell ids of a model do
    not start at 0.
    """
    if formatversion != 2:
        return ("model.txt",)
    firstcellids = [
        scan_modeldata(modelpath, printwarningsonly=True)[0].select(pl.col("inputcellid").min()).collect().item()
        for modelpath in modelpaths
    ]
    return ("model.txt",) if all(firstcellid == 0 for firstcellid in firstcellids) else MODEL_FILE_NAMES


def get_model_output_folders(args: argparse.Namespace, modelpath_given: bool) -> dict[str, list[Path]]:
    """Return the folders that get the output files, for each action of args."""
    if args.downscale3dgrid:
        return {
            "downscale3dgrid": [get_downscale_outputfolder(args.modelpath[0], args.outputgridsize, args.outputfile)]
        }

    outputfolders: dict[str, list[Path]] = {}
    if args.dimensionreduce is not None:
        outputfolders["dimensionreduce"] = [
            get_dimreduce_outputfolder(args.outputfile, modelpath, args.dimensionreduce) for modelpath in args.modelpath
        ]
    if args.makemodelfromgriddata:
        outputfolders["makemodelfromgriddata"] = [
            get_griddata_outputfolder(args.outputfile, args.modelpath, modelpath_given)
        ]
    if args.makeenergyinputfiles:
        outputfolders["makeenergyinputfiles"] = [Path(args.outputfile or Path())]
    if args.format_version is not None and args.dimensionreduce is None:
        if args.outputfile is not None and len(args.modelpath) > 1:
            exit_with_error(
                "-o names one folder, and each model path writes a model.txt",
                "Give one model path with -o, or give no -o to convert each model.txt in its model folder",
            )
        outputfolders["convertformat"] = [
            Path(args.outputfile) if args.outputfile is not None else get_model_text_folder(modelpath)
            for modelpath in args.modelpath
        ]
    return outputfolders


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Tools to create an ARTIS input model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpath_given = bool(args.modelpath)
    args.modelpath = normalize_path_list(args.modelpath)

    actions = (
        args.downscale3dgrid,
        args.dimensionreduce is not None,
        args.makemodelfromgriddata,
        args.makeenergyinputfiles,
        args.format_version is not None,
    )
    if not any(actions):
        exit_with_error(
            "The command needs an action",
            "Give --downscale3dgrid, --dimensionreduce, --format-version, --makemodelfromgriddata, or"
            " --makeenergyinputfiles",
        )

    outputfolders = get_model_output_folders(args, modelpath_given)
    outputfilenames: dict[str, Sequence[str]] = {
        "makemodelfromgriddata": get_griddata_output_file_names(args.getcellopacityfromYe),
        "makeenergyinputfiles": ENERGY_FILE_NAMES,
    }
    if "convertformat" in outputfolders:
        outputfilenames["convertformat"] = get_converted_file_names(args.modelpath, args.format_version)
    for action, folders in outputfolders.items():
        confirm_overwrite(folders, outputfilenames.get(action, MODEL_FILE_NAMES), force=args.force)

    if args.downscale3dgrid:
        make_downscaled_3d_grid(
            modelpath=Path(args.modelpath[0]),
            outputgridsize=args.outputgridsize,
            plot=args.downscaleplot,
            outputfolder=outputfolders["downscale3dgrid"][0],
            args=args,
        )
        return

    if args.dimensionreduce is not None:
        ndim_out = args.dimensionreduce
        assert ndim_out in {0, 1, 2}
        for modelpath, outdir in zip(args.modelpath, outputfolders["dimensionreduce"], strict=True):
            dfmodel, modelmeta = scan_modeldata(modelpath)
            ndim_in = modelmeta["dimensions"]
            if ndim_in <= ndim_out:
                msg = f"Cannot reduce {ndim_in}D model to {ndim_out}D"
                raise ValueError(msg)

            dfelabundances = scan_initelemabundances(modelpath)
            dfgridcontributions = get_gridparticlecontributions_or_none(modelpath)

            (dfmodel_out, dfelabundances_out, dfgridcontributions_out, modelmeta_out) = dimension_reduce_model(
                dfmodel=dfmodel,
                outputdimensions=ndim_out,
                dfelabundances=dfelabundances,
                dfgridcontributions=dfgridcontributions,
                modelmeta=modelmeta,
            )
            outdir.mkdir(exist_ok=True, parents=True)
            modelmeta_out["headercommentlines"] = [
                *modelmeta.get("headercommentlines", []),
                f"Dimension reduced from {ndim_in}-dimensional model",
            ]
            if args.format_version is not None:
                modelmeta_out["formatversion"] = args.format_version
            assert dfelabundances_out is not None
            save_initelemabundances(dfelabundances_out, outpath=outdir)
            save_modeldata(dfmodel=dfmodel_out, modelmeta=modelmeta_out, outpath=outdir)
            # the function gives an empty frame for a model with no contributions, thus test the input
            if dfgridcontributions is not None:
                save_gridparticlecontributions(dfgridcontributions_out, outdir / "gridcontributions.txt")

    for modelpath, outdir in zip(args.modelpath, outputfolders.get("convertformat", []), strict=False):
        convert_model_format(modelpath, outdir, args.format_version)

    if args.makemodelfromgriddata:
        makemodelfromgriddata(
            gridfolderpath=args.pathtogriddata,
            outputpath=outputfolders["makemodelfromgriddata"][0],
            fillcentralhole=args.fillcentralhole,
            getcellopacityfromYe=args.getcellopacityfromYe,
        )

    if args.makeenergyinputfiles:
        plmodel, modelmeta = scan_modeldata(args.modelpath[0])
        cellmass_grams = (
            add_derived_cols_to_modeldata(plmodel, modelmeta=modelmeta)
            .select(pl.col("mass_g").cast(pl.Float64))
            .collect()
            .to_series()
            .to_numpy()
        )
        print(f"Total mass {cellmass_grams.sum() / Msun_to_g} Msun")

        make_energy_files(cellmass_grams, outputpath=outputfolders["makeenergyinputfiles"][0])
