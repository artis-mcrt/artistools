"""Build an ARTIS input model by downscaling, dimension-reducing, or rescaling an existing model."""

import argparse
import shutil
import tempfile
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import Msun_to_g
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import CELL_ID_FILE_NAMES
from artistools.inputmodel.core import dimension_reduce_model
from artistools.inputmodel.core import get_elemabundances_of_model_columns
from artistools.inputmodel.core import get_model_file_path
from artistools.inputmodel.core import get_model_input_folder
from artistools.inputmodel.core import get_modelmeta
from artistools.inputmodel.core import MODEL_FILE_NAMES
from artistools.inputmodel.core import remove_parquet_cache
from artistools.inputmodel.core import require_no_ye_file_of_earlier_model
from artistools.inputmodel.core import save_modeldata
from artistools.inputmodel.core import scan_abundance_file
from artistools.inputmodel.core import scan_initelemabundances
from artistools.inputmodel.core import scan_modeldata
from artistools.inputmodel.core import with_cell_ids_from_zero
from artistools.inputmodel.core import with_ye_of_yefile
from artistools.inputmodel.downscale3dgrid import get_downscale_outputfolder
from artistools.inputmodel.downscale3dgrid import make_downscaled_3d_grid
from artistools.inputmodel.energyinputfiles import ENERGY_FILE_NAMES
from artistools.inputmodel.energyinputfiles import make_energy_files
from artistools.inputmodel.modelfromhydro import get_griddata_output_file_names
from artistools.inputmodel.modelfromhydro import makemodelfromgriddata
from artistools.inputmodel.rprocess_from_trajectory import get_gridparticlecontributions
from artistools.inputmodel.rprocess_from_trajectory import save_gridparticlecontributions
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import find_sn3d_input_file
from artistools.misc import firstexisting_or_none
from artistools.misc import get_file_copies
from artistools.misc import get_model_coordinates
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.misc import SPHERICAL_EDGE_COORDINATES
from artistools.misc import without_compressed_extension


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
        help="The format version of the model.txt that the command writes. If the command has no other action,"
        " it converts the model.txt of each model, in its model folder or in the -o folder. With --dimensionreduce,"
        " the reduced model gets this version. Version 2 starts with the line 'format_version 2' and has a header"
        " of keys with the cell edges, and its model has no abundances.txt. An older sn3d, which does not know format"
        " version 2, cannot read such a file. A 2D spherical or a 3D spherical model has version 2 only",
    )

    parser.add_argument(
        "--makemodelfromgriddata",
        action="store_true",
        help="Make ARTIS model files from the grid.dat or the grid_spherical.dat of an SPH map",
    )

    parser.add_argument(
        "-pathtogriddata",
        default=".",
        help="The folder that holds the grid.dat or the grid_spherical.dat of an SPH map",
    )

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


def get_contributions_beside_model(sourcefolder: Path) -> pl.DataFrame | None:
    """Return the particle contributions of the gridcontributions.txt in the folder of model.txt, or None.

    An output folder can be a subfolder of the source folder, and its gridcontributions.txt is of a different model.
    Thus the function does not search the subfolders.
    """
    contributionsfile = firstexisting_or_none(
        "gridcontributions.txt", folder=sourcefolder, tryzipped=True, search_subfolders=False
    )
    return None if contributionsfile is None else get_gridparticlecontributions(sourcefolder)


def convert_model_format(modelpath: Path, outputfolder: Path, formatversion: int) -> None:
    """Convert the model.txt of a model to the format version formatversion, and write it into outputfolder.

    The cells keep their sequence and their values to the Float32 precision that artistools reads. Version 1 gives
    the position of each cell in columns, and version 2 gives the cell edges in the header. A version 2 model has no
    abundances.txt and no Ye.txt, because its model.txt holds their values. It also has no opacity.txt, which sn3d
    does not read.
    """
    lzmodel, modelmeta = scan_modeldata(modelpath)
    coordinates = get_model_coordinates(modelmeta)
    if formatversion == 1 and coordinates in SPHERICAL_EDGE_COORDINATES:
        exit_with_error(
            f"Format version 1 of model.txt has no {coordinates} coordinates",
            "A 2D spherical or a 3D spherical model has format version 2 only",
        )
    print(
        f"The model.txt of {modelpath} has format version {modelmeta['formatversion']} and {coordinates} coordinates."
        f" The command writes it in format version {formatversion}."
    )
    if formatversion == 2:
        print("  Version 2 gives the cell edges as velocities [km/s] in its header, and each cell has rho [g/cm3].")
    else:
        print("  Version 1 gives the position of each cell in columns. A 1D cell has log10(rho [g/cm3]).")
    print(
        "  Each value has the Float32 precision that artistools reads. Version 2 holds 9 significant figures, which"
        " give each Float32 value again with no change, and version 1 holds 8."
    )

    # save_modeldata can write the file that the query scans, thus the read is complete before the write
    dfmodel = lzmodel.collect()
    sourcefolder = get_model_input_folder(modelpath)
    hasotherfolder = outputfolder.resolve() != sourcefolder.resolve()
    modelcopies = [path for path in get_file_copies(sourcefolder / "model.txt") if path.is_file()]
    if len(modelcopies) > 1:
        # a reader of each copy can take the parquet cache of the other copy, and abundances.txt is of one of them
        exit_with_error(
            f"{sourcefolder} holds {' and '.join(path.name for path in modelcopies)}",
            "sn3d reads model.txt, and the other copy can hold different values. Delete one copy",
        )
    # sn3d reads Ye.txt or Ye.txt.zst, and no copy that gzip or xz compressed
    yefilepath = find_sn3d_input_file("Ye.txt", sourcefolder, datafolders=(".",))
    if yefilepath is not None and modelmeta["formatversion"] == 2:
        exit_with_error(
            f"{sourcefolder} holds Ye.txt, but its model.txt has format version 2",
            "sn3d stops for such a model. The column Ye of model.txt gives the electron fraction. Delete Ye.txt",
        )
    opacityfilepath = firstexisting_or_none("opacity.txt", folder=sourcefolder, tryzipped=True, search_subfolders=False)
    if formatversion == 2 and opacityfilepath is not None:
        # the values of opacity.txt have no column in model.txt, thus the command does not delete the file
        if outputfolder.resolve() == sourcefolder.resolve():
            exit_with_error(
                f"{sourcefolder} holds opacity.txt, and a model of format version 2 has no such file",
                "sn3d does not read opacity.txt. Delete the file, or give -o to write the model into a different"
                " folder",
            )
        print("  Version 2 has no opacity.txt, thus the output folder gets no copy of it. sn3d does not read the file.")
    if yefilepath is not None and formatversion == 2:
        print("  Version 2 has no Ye.txt.")
        dfmodel = with_ye_of_yefile(dfmodel, yefilepath)
    if hasotherfolder and formatversion == 1 and yefilepath is None:
        # a version 1 output in a different folder gets the Ye.txt of the source model, and here there is none
        require_no_ye_file_of_earlier_model(outputfolder, modelpath)
    gridcolumns = {"inputcellid", "modelgridindex", "rho", "logrho"}
    customcols = [col for col in dfmodel.columns if col not in gridcolumns and not col.startswith(("pos_", "vel_"))]
    outputfolder.mkdir(exist_ok=True, parents=True)

    # The element mass fractions come from the columns of the model.txt that the command read (version 2), or from
    # abundances.txt (version 1). Another copy of model.txt in the folder can give other values
    if modelmeta["formatversion"] == 2:
        dfelabundances = get_elemabundances_of_model_columns(dfmodel.lazy()).collect()
    else:
        # sn3d reads the abundances.txt beside model.txt, and a file in a subfolder is of a different model
        abundancefile = firstexisting_or_none(
            "abundances.txt", folder=sourcefolder, tryzipped=True, search_subfolders=False
        )
        if abundancefile is None:
            exit_with_error(
                f"{sourcefolder} holds no abundances.txt beside its model.txt of format version 1",
                "sn3d needs abundances.txt for such a model, thus the element mass fractions of the model are not"
                " known. Give the model its abundances.txt",
            )
        dfelabundances = scan_abundance_file(abundancefile, printwarningsonly=True).collect()

    dfgridcontributions = None
    if formatversion == 2 and dfmodel["inputcellid"].min() != 0:
        # Each file of a model has the same cell ids, and version 2 has cell ids from 0. The output folder can be a
        # subfolder of the source folder, thus the contributions come from the source folder alone
        print("  Version 2 has cell ids from 0, thus gridcontributions.txt gets the same ids.")
        dfmodel, dfelabundances, dfgridcontributions = with_cell_ids_from_zero(
            dfmodel, dfelabundances, get_contributions_beside_model(sourcefolder)
        )

    if formatversion == 2:
        print(
            "  Version 2 has no abundances.txt. The mass fraction of an element is the sum of its nuclide columns and"
            " of its column X_<element>_otherstable."
        )
    # A conversion in the model folder replaces the files of the source model. Thus each new file goes first into a
    # temporary folder in the output folder, and the files replace the old files only after each write is complete.
    # A rename in one file system needs no space, thus a full disk cannot stop it
    stagingfolder = Path(tempfile.mkdtemp(prefix=".formatconversion-", dir=outputfolder))
    try:
        # With dfelabundances, version 1 also gets abundances.txt, and version 2 gets the otherstable columns
        save_modeldata(
            dfmodel,
            outpath=stagingfolder,
            modelmeta=modelmeta | {"formatversion": formatversion},
            extracols=customcols,
            dfelabundances=dfelabundances,
        )
        if dfgridcontributions is not None:
            save_gridparticlecontributions(dfgridcontributions, stagingfolder / "gridcontributions.txt")
        # A different output folder also gets each file with cell ids that needs no change, thus the model is
        # complete. sn3d also reads a compressed copy of such a file
        unchangedfilenames = [*(CELL_ID_FILE_NAMES if formatversion == 1 else ())]
        if dfgridcontributions is None:
            unchangedfilenames.append("gridcontributions.txt")
        obsoletefilenames = ["abundances.txt", *CELL_ID_FILE_NAMES] if formatversion == 2 else []
        for filename in unchangedfilenames if hasotherfolder else []:
            sourcefile = firstexisting_or_none(filename, folder=sourcefolder, tryzipped=True, search_subfolders=False)
            if sourcefile is not None:
                shutil.copy2(sourcefile, stagingfolder / sourcefile.name)
            elif filename == "gridcontributions.txt":
                # the particle contributions of an earlier model do not go with the cells of the new model.txt. The
                # command stopped above for a Ye.txt of an earlier model, and sn3d does not read opacity.txt
                obsoletefilenames.append(filename)

        for stagedfile in sorted(stagingfolder.iterdir()):
            newfile = outputfolder / stagedfile.name
            stagedfile.replace(newfile)
            # the new file replaces each copy of the old file, e.g. a plain Ye.txt beside the new Ye.txt.zst
            for oldfile in get_file_copies(newfile):
                if oldfile != newfile and oldfile.is_file():
                    oldfile.unlink()
            remove_parquet_cache(newfile)
            print(f"  Saved {newfile}")
        for filename in obsoletefilenames:
            for oldfile in get_file_copies(outputfolder / filename):
                if oldfile.is_file():
                    oldfile.unlink()
                    remove_parquet_cache(oldfile)
                    print(f"  Deleted {oldfile}, which the converted model has no use for")
    finally:
        shutil.rmtree(stagingfolder, ignore_errors=True)


def get_converted_file_names(
    modelpaths: Sequence[Path], formatversion: int | None, hasoutputfolder: bool
) -> Sequence[str]:
    """Return the names of the files that convert_model_format writes for these models.

    A conversion to version 1 also writes abundances.txt. A conversion to version 2 deletes abundances.txt and
    Ye.txt. If the cell ids of a model do not start at 0, it also writes gridcontributions.txt. A conversion into a
    different folder (hasoutputfolder) always copies or writes the files that the output model has. For version 2,
    it also deletes a Ye.txt and an opacity.txt of an earlier model in that folder.
    """
    if formatversion != 2:
        return (*MODEL_FILE_NAMES, *CELL_ID_FILE_NAMES) if hasoutputfolder else ("model.txt", "abundances.txt")
    firstcellids = [
        scan_modeldata(modelpath, printwarningsonly=True)[0].select(pl.col("inputcellid").min()).collect().item()
        for modelpath in modelpaths
    ]
    if all(firstcellid == 0 for firstcellid in firstcellids) and not hasoutputfolder:
        return ("model.txt", "abundances.txt", "Ye.txt")
    return (*MODEL_FILE_NAMES, *CELL_ID_FILE_NAMES) if hasoutputfolder else (*MODEL_FILE_NAMES, "Ye.txt")


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
    if args.format_version is not None and (args.makemodelfromgriddata or args.makeenergyinputfiles):
        exit_with_error(
            "--format-version applies to a conversion, to --dimensionreduce, and to --downscale3dgrid, but not to"
            " --makemodelfromgriddata or --makeenergyinputfiles",
            "Run the conversion with --format-version as a separate command",
        )
    if args.format_version is not None and args.dimensionreduce is None:
        if args.outputfile is not None and len(args.modelpath) > 1:
            exit_with_error(
                "-o names one folder, but the command writes one model.txt for each model path",
                "Give one model path with -o, or give no -o to convert each model.txt in its model folder",
            )
        outputfolders["convertformat"] = [
            Path(args.outputfile) if args.outputfile is not None else get_model_input_folder(modelpath)
            for modelpath in args.modelpath
        ]
        for modelpath, outputfolder in zip(args.modelpath, outputfolders["convertformat"], strict=True):
            require_model_txt_in_source_folder(modelpath, outputfolder)
    return outputfolders


def require_model_txt_in_source_folder(modelpath: Path, outputfolder: Path) -> None:
    """Stop for a conversion of a model file with a different name into the folder of that file.

    The conversion writes model.txt, and the model.txt of that folder, with its abundances.txt and Ye.txt, is a
    different model.
    """
    modelfile = get_model_file_path(modelpath)
    if (
        modelfile is not None
        and without_compressed_extension(modelfile).name != "model.txt"
        and outputfolder.resolve() == modelfile.parent.resolve()
    ):
        exit_with_error(
            f"The conversion of {modelfile} writes model.txt in its folder, and that folder can hold a different"
            " model with the files model.txt, abundances.txt, and Ye.txt",
            "Give -o with a different folder",
        )


def get_output_format_version(args: argparse.Namespace, modelpath: Path) -> int:
    """Return the format version of the model.txt that a reduction or a downscale of a model path writes.

    --format-version gives the version. Without it, a downscale keeps the version of the model. A reduction gives
    version 2 to a 2D spherical model, which has no version 1, and version 1 to each other model.
    """
    if args.format_version is not None:
        return int(args.format_version)
    modelmeta = get_modelmeta(modelpath)
    if args.downscale3dgrid:
        return int(modelmeta["formatversion"])
    return 2 if get_model_coordinates(modelmeta) == "spherical3d" and args.dimensionreduce == 2 else 1


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
        "makemodelfromgriddata": get_griddata_output_file_names(args.getcellopacityfromYe, args.pathtogriddata),
        "makeenergyinputfiles": ENERGY_FILE_NAMES,
    }
    if "convertformat" in outputfolders:
        outputfilenames["convertformat"] = get_converted_file_names(
            args.modelpath, args.format_version, hasoutputfolder=args.outputfile is not None
        )
    for action, folders in outputfolders.items():
        filenames = outputfilenames.get(action)
        if filenames is None:
            # save_modeldata deletes a Ye.txt and an opacity.txt beside a model of format version 2
            writesversion2 = any(get_output_format_version(args, modelpath) == 2 for modelpath in args.modelpath)
            filenames = (*MODEL_FILE_NAMES, *CELL_ID_FILE_NAMES) if writesversion2 else MODEL_FILE_NAMES
        confirm_overwrite(folders, filenames, force=args.force)

    if args.downscale3dgrid:
        make_downscaled_3d_grid(
            modelpath=Path(args.modelpath[0]),
            outputgridsize=args.outputgridsize,
            plot=args.downscaleplot,
            outputfolder=outputfolders["downscale3dgrid"][0],
            args=args,
            formatversion=get_output_format_version(args, Path(args.modelpath[0])),
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

            sourcefolder = get_model_input_folder(modelpath)
            dfelabundances = scan_initelemabundances(modelpath)
            dfgridcontributions = get_contributions_beside_model(sourcefolder)
            outputformatversion = get_output_format_version(args, modelpath)
            yefilepath = find_sn3d_input_file("Ye.txt", sourcefolder, datafolders=(".",))
            if yefilepath is not None and outputformatversion == 2:
                # sn3d takes the electron fractions of Ye.txt for the source model, and the reduced model of version 2
                # gets them in its column Ye. A reduced model of version 1 gets the values of the column Ye of the
                # source model.txt, as before
                print("  Version 2 has no Ye.txt.")
                dfmodel = with_ye_of_yefile(dfmodel.collect(), yefilepath).lazy()

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
            modelmeta_out["formatversion"] = outputformatversion
            if outputformatversion == 1:
                # the reduced model has different cells, thus no Ye.txt of the output folder is of this model
                require_no_ye_file_of_earlier_model(outdir, modelpath)
            if outputformatversion == 2:
                # version 2 has cell ids from 0 in each file of the model
                dfmodel_out, dfelabundances_zero, dfgridcontributions_zero = with_cell_ids_from_zero(
                    dfmodel_out,
                    dfelabundances_out,
                    dfgridcontributions_out if "cellindex" in dfgridcontributions_out.columns else None,
                )
                dfelabundances_out = dfelabundances_zero if dfelabundances_zero is not None else dfelabundances_out
                if dfgridcontributions_zero is not None:
                    dfgridcontributions_out = dfgridcontributions_zero
            assert dfelabundances_out is not None
            # a version 1 model also gets abundances.txt, and a version 2 model gets otherstable columns
            # the overwrite prompt names the Ye.txt and the opacity.txt of an earlier model in the output folder
            save_modeldata(
                dfmodel=dfmodel_out,
                modelmeta=modelmeta_out,
                outpath=outdir,
                dfelabundances=dfelabundances_out,
                deletecellidfiles=True,
            )
            # the function gives an empty frame for a model with no contributions, thus test the input
            if dfgridcontributions is not None:
                save_gridparticlecontributions(dfgridcontributions_out, outdir / "gridcontributions.txt")
            else:
                # the particle contributions of an earlier model do not go with the cells of the new model.txt
                for oldfile in get_file_copies(outdir / "gridcontributions.txt"):
                    if oldfile.is_file():
                        oldfile.unlink()
                        print(f"  Deleted {oldfile}, which is a file of an earlier model in the output folder")

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
