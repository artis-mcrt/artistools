"""Build an ARTIS input model by downscaling, dimension-reducing, or rescaling an existing model."""

import argparse
import shutil
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import Msun_to_g
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import CELL_ID_FILE_NAMES
from artistools.inputmodel.core import dimension_reduce_model
from artistools.inputmodel.core import get_model_text_folder
from artistools.inputmodel.core import get_modelmeta
from artistools.inputmodel.core import MODEL_FILE_NAMES
from artistools.inputmodel.core import remove_parquet_cache
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
from artistools.misc import get_file_copies
from artistools.misc import get_model_coordinates
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.misc import SPHERICAL_EDGE_COORDINATES
from artistools.misc import zopen


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


def with_ye_of_yefile(dfmodel: pl.DataFrame, yefilepath: Path) -> pl.DataFrame:
    """Return the cells of a version 1 model with the electron fractions of its Ye.txt in the column Ye.

    sn3d reads Ye.txt after model.txt, thus a value of Ye.txt replaces the value of the column Ye. Ye.txt can hold
    only some of the cells, and each other cell keeps the value of the column. A cell with no value from the two
    sources gets 0. sn3d with TANAKA2020_ELECTRONFRAC stops for such a cell, as it does in format version 1.
    """
    with zopen(yefilepath) as yefile:
        yelines = [line.split() for line in yefile.read().splitlines()[1:] if line.strip()]
    if any(len(tokens) != 2 for tokens in yelines):
        msg = f"{yefilepath}: each line after the first line must hold a cell id and an electron fraction"
        raise ValueError(msg)
    dfye = pl.DataFrame(
        {"inputcellid": [int(cellid) for cellid, _ in yelines], "Ye_file": [float(ye) for _, ye in yelines]},
        schema={"inputcellid": pl.Int32, "Ye_file": pl.Float32},
    ).unique(subset="inputcellid", keep="last")
    unknownids = dfye.join(dfmodel.select("inputcellid"), on="inputcellid", how="anti")
    if not unknownids.is_empty():
        msg = (
            f"{yefilepath}: {unknownids.height} cell ids are not in model.txt, e.g. {unknownids['inputcellid'].min()}."
            " The cell ids of Ye.txt must be the ids of model.txt"
        )
        raise ValueError(msg)

    columnvalue = pl.col("Ye") if "Ye" in dfmodel.columns else pl.lit(None, dtype=pl.Float32)
    dfmodel = (
        dfmodel
        .join(dfye, on="inputcellid", how="left", maintain_order="left")
        .with_columns(Ye=pl.coalesce(pl.col("Ye_file"), columnvalue))
        .drop("Ye_file")
    )
    print(
        f"  Version 2 has no Ye.txt. The column Ye of model.txt gets the {dfye.height} electron fractions of Ye.txt,"
        " and each other cell keeps the value of that column."
    )
    novaluecount = dfmodel.filter(pl.col("Ye").is_null() & (pl.col("rho") > 0)).height
    if novaluecount > 0:
        print_warning(
            f"{novaluecount} cells with density have no electron fraction in Ye.txt or in model.txt. The column Ye"
            " gets 0 for these cells, and sn3d with TANAKA2020_ELECTRONFRAC stops for an electron fraction of 0"
        )
    return dfmodel.with_columns(pl.col("Ye").fill_null(0.0))


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
    sourcefolder = get_model_text_folder(modelpath)
    # sn3d also reads a compressed copy of Ye.txt
    yefilepath = firstexisting_or_none("Ye.txt", folder=sourcefolder, tryzipped=True, search_subfolders=False)
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
        dfmodel = with_ye_of_yefile(dfmodel, yefilepath)
    gridcolumns = {"inputcellid", "modelgridindex", "rho", "logrho"}
    customcols = [col for col in dfmodel.columns if col not in gridcolumns and not col.startswith(("pos_", "vel_"))]
    outputfolder.mkdir(exist_ok=True, parents=True)

    # the element mass fractions come from abundances.txt (version 1) or from the columns of model.txt (version 2)
    haselements = (
        modelmeta["formatversion"] == 2
        or firstexisting_or_none("abundances.txt", folder=get_model_text_folder(modelpath), tryzipped=True) is not None
    )
    dfelabundances = scan_initelemabundances(modelpath, printwarningsonly=True).collect() if haselements else None

    dfgridcontributions = None
    if formatversion == 2 and dfmodel["inputcellid"].min() != 0:
        # each file of a model has the same cell ids, and version 2 has cell ids from 0
        print("  Version 2 has cell ids from 0, thus gridcontributions.txt gets the same ids.")
        dfmodel, dfelabundances, dfgridcontributions = with_cell_ids_from_zero(
            dfmodel, dfelabundances, get_gridparticlecontributions_or_none(modelpath)
        )

    if formatversion == 2:
        print(
            "  Version 2 has no abundances.txt. The mass fraction of an element is the sum of its nuclide columns and"
            " of its column X_<element>_otherstable."
        )
    hasotherfolder = outputfolder.resolve() != sourcefolder.resolve()
    if hasotherfolder:
        # A different output folder holds only the files of the output model. A file of an earlier model in that
        # folder gives its values to the wrong cells, and save_modeldata stops for some of them. The (y/n) prompt of
        # the command named each of these files
        for filename in (*CELL_ID_FILE_NAMES, "abundances.txt"):
            for oldfile in get_file_copies(outputfolder / filename):
                if oldfile.is_file():
                    oldfile.unlink()
                    remove_parquet_cache(oldfile)
                    print(f"Deleted {oldfile}, which is a file of an earlier model in the output folder")

    # save_modeldata checks the cells and can stop, thus it comes before gridcontributions.txt gets new cell ids. With
    # dfelabundances, version 1 also gets abundances.txt, and version 2 gets the otherstable columns
    save_modeldata(
        dfmodel,
        outpath=outputfolder,
        modelmeta=modelmeta | {"formatversion": formatversion},
        extracols=customcols,
        dfelabundances=dfelabundances,
    )
    if dfgridcontributions is not None:
        save_gridparticlecontributions(dfgridcontributions, outputfolder / "gridcontributions.txt")

    if hasotherfolder:
        # A different output folder also gets each file with cell ids that needs no change, thus the model is
        # complete. sn3d also reads a compressed copy of such a file
        unchangedfilenames = [*(CELL_ID_FILE_NAMES if formatversion == 1 else ())]
        if dfgridcontributions is None:
            unchangedfilenames.append("gridcontributions.txt")
        unchangedfiles = [
            firstexisting_or_none(filename, folder=sourcefolder, tryzipped=True, search_subfolders=False)
            for filename in unchangedfilenames
        ]
        for sourcefile in unchangedfiles:
            if sourcefile is None:
                continue
            shutil.copy2(sourcefile, outputfolder / sourcefile.name)
            print(f"Copied {sourcefile.name} to {outputfolder}, because its cell ids need no change")


def get_converted_file_names(
    modelpaths: Sequence[Path], formatversion: int | None, hasoutputfolder: bool
) -> Sequence[str]:
    """Return the names of the files that convert_model_format writes for these models.

    A conversion to version 1 also writes abundances.txt. A conversion to version 2 deletes abundances.txt and
    Ye.txt. If the cell ids of a model do not start at 0, it also writes gridcontributions.txt. A conversion into a
    different folder (hasoutputfolder) always copies or writes the files that the output model has, and it deletes
    a Ye.txt and an opacity.txt of an earlier model in that folder.
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
    if args.format_version is not None and args.dimensionreduce is None:
        if args.outputfile is not None and len(args.modelpath) > 1:
            exit_with_error(
                "-o names one folder, but the command writes one model.txt for each model path",
                "Give one model path with -o, or give no -o to convert each model.txt in its model folder",
            )
        outputfolders["convertformat"] = [
            Path(args.outputfile) if args.outputfile is not None else get_model_text_folder(modelpath)
            for modelpath in args.modelpath
        ]
    return outputfolders


def writes_format_version_2(args: argparse.Namespace) -> bool:
    """Return True if a reduction or a downscale of the models of args writes a model.txt of format version 2.

    A model keeps its format version if the user gives no --format-version.
    """
    if args.format_version is not None:
        return bool(args.format_version == 2)
    return any(get_modelmeta(modelpath)["formatversion"] == 2 for modelpath in args.modelpath)


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
            filenames = (*MODEL_FILE_NAMES, *CELL_ID_FILE_NAMES) if writes_format_version_2(args) else MODEL_FILE_NAMES
        confirm_overwrite(folders, filenames, force=args.force)

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
            if args.format_version == 2:
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
            save_modeldata(
                dfmodel=dfmodel_out, modelmeta=modelmeta_out, outpath=outdir, dfelabundances=dfelabundances_out
            )
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
