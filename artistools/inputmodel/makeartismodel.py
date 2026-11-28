"""Build an ARTIS input model by downscaling, dimension-reducing, or rescaling an existing model."""

import argparse
import shutil
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import day_to_s
from artistools.constants import Msun_to_g
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import CELL_ID_FILE_NAMES
from artistools.inputmodel.core import dimension_reduce_model
from artistools.inputmodel.core import find_abundance_file
from artistools.inputmodel.core import find_ye_file
from artistools.inputmodel.core import get_elemabundances_of_model_columns
from artistools.inputmodel.core import get_gridcontributions_folder
from artistools.inputmodel.core import get_middle_layer_lower_edge
from artistools.inputmodel.core import get_model_input_folder
from artistools.inputmodel.core import get_modelmeta
from artistools.inputmodel.core import MODEL_FILE_NAMES
from artistools.inputmodel.core import require_no_ye_file_of_earlier_model
from artistools.inputmodel.core import require_no_ye_file_of_version_2_model
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
from artistools.inputmodel.rprocess_from_trajectory import delete_gridparticlecontributions
from artistools.inputmodel.rprocess_from_trajectory import get_gridparticlecontributions_or_none
from artistools.inputmodel.rprocess_from_trajectory import save_gridparticlecontributions
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import ALL_MODEL_EDGE_KEYS
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import get_model_coordinates
from artistools.misc import has_mirror_plane
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
        "--make-2d-model-from-slice-axis",
        dest="make2dmodelfromsliceaxis",
        choices=["x", "y", "z"],
        help="Select the axis normal to a central layer of a 3D Cartesian model for a 2D cylindrical model",
    )
    parser.add_argument(
        "--make2dmodelfromsliceaxis", dest="make2dmodelfromsliceaxis", choices=["x", "y", "z"], help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--model-r-coord-axis-2d",
        dest="modelrcoordaxis2d",
        choices=["+x", "+y", "+z", "-x", "-y", "-z"],
        help="Select the signed axis of the slice that gives the cylindrical radius, for example +y",
    )
    parser.add_argument(
        "--modelrcoordaxis2d",
        dest="modelrcoordaxis2d",
        choices=["+x", "+y", "+z", "-x", "-y", "-z"],
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--format-version",
        type=int,
        choices=[1, 2],
        default=None,
        help="The format version of the model.txt that the command writes. If the command has no other action,"
        " it converts the model.txt of one model into the -o folder, which holds no model file. With --dimensionreduce,"
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


def convert_model_format(modelpath: Path, outputfolder: Path, formatversion: int) -> None:
    """Convert the model.txt of a model to the format version formatversion, and write it into outputfolder.

    The cells keep their sequence and their values to the Float32 precision that artistools reads. Version 1 gives
    the position of each cell in columns, and version 2 gives the cell edges in the header. A version 2 model has no
    abundances.txt and no Ye.txt, because its model.txt holds their values. It also has no opacity.txt, which sn3d
    does not read. The output folder holds no model file before the conversion, thus the command changes no file
    of a model.
    """
    # the search of the readers, which also finds a compressed file and a file in a subfolder, e.g. data/model.txt
    oldfile = firstexisting_or_none([*MODEL_FILE_NAMES, *CELL_ID_FILE_NAMES], folder=outputfolder)
    if oldfile is not None:
        exit_with_error(
            f"The output folder holds {oldfile}, and a conversion writes into a folder with no model file",
            "Give -o with a new folder",
        )
    lzmodel, modelmeta = scan_modeldata(modelpath)
    coordinates = get_model_coordinates(modelmeta)
    if formatversion == 1 and coordinates in SPHERICAL_EDGE_COORDINATES:
        exit_with_error(
            f"Format version 1 of model.txt has no {coordinates} coordinates",
            "A 2D spherical or a 3D spherical model has format version 2 only",
        )
    if formatversion == 1 and has_mirror_plane(modelmeta):
        exit_with_error(
            "Format version 1 of model.txt has no mirror symmetry",
            "A model with the key mirror_plane z=0 has format version 2 only",
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
    if formatversion == 1 and coordinates == "spherical1d":
        # the writer of a 1D model of version 1 keeps the format of the files of earlier versions of artistools
        print_warning(
            "A 1D model of version 1 gives vel_r_max_kmps with 2 decimal places. Thus each shell edge can move by"
            " up to 0.005 km/s, which changes the volume and the mass of the shell"
        )

    dfmodel = lzmodel.collect()
    sourcefolder = get_model_input_folder(modelpath)
    yefilepath = find_ye_file(sourcefolder)
    require_no_ye_file_of_version_2_model(yefilepath, modelmeta, sourcefolder)
    if yefilepath is not None:
        print("  The column Ye of the new model.txt gets the electron fractions of Ye.txt, thus it gets no Ye.txt.")
        dfmodel = with_ye_of_yefile(dfmodel, yefilepath)
    gridcolumns = {"inputcellid", "modelgridindex", "rho", "logrho"}
    customcols = [col for col in dfmodel.columns if col not in gridcolumns and not col.startswith(("pos_", "vel_"))]

    # The element mass fractions come from the columns of the model.txt that the command read (version 2), or from
    # abundances.txt (version 1)
    if modelmeta["formatversion"] == 2:
        dfelabundances = get_elemabundances_of_model_columns(dfmodel.lazy()).collect()
    else:
        abundancefile = find_abundance_file(sourcefolder)
        if abundancefile is None:
            exit_with_error(
                f"sn3d finds no abundances.txt for the model.txt of format version 1 in {sourcefolder}",
                "sn3d needs abundances.txt for such a model, thus the element mass fractions of the model are not"
                " known. Give the model its abundances.txt",
            )
        dfelabundances = scan_abundance_file(abundancefile, printwarningsonly=True).collect()

    dfgridcontributions = get_gridparticlecontributions_or_none(get_gridcontributions_folder(modelpath))
    if formatversion == 2:
        print(
            "  Version 2 has no abundances.txt. The mass fraction of an element is the sum of its nuclide columns and"
            " of its column X_<element>_otherstable. It has cell ids from 0, and gridcontributions.txt gets the same"
            " ids."
        )
        dfmodel, dfelabundances, dfgridcontributions = with_cell_ids_from_zero(
            dfmodel, dfelabundances, dfgridcontributions
        )

    outputfolder.mkdir(exist_ok=True, parents=True)
    # With dfelabundances, version 1 also gets abundances.txt, and version 2 gets the otherstable columns
    save_modeldata(
        dfmodel,
        outpath=outputfolder,
        modelmeta=modelmeta | {"formatversion": formatversion},
        extracols=customcols,
        dfelabundances=dfelabundances,
    )
    if dfgridcontributions is not None:
        save_gridparticlecontributions(dfgridcontributions, outputfolder / "gridcontributions.txt")
    opacityfile = firstexisting_or_none("opacity.txt", folder=sourcefolder, tryzipped=True, search_subfolders=False)
    if opacityfile is not None and formatversion == 1:
        shutil.copy2(opacityfile, outputfolder / opacityfile.name)
    elif opacityfile is not None:
        print("  Version 2 has no opacity.txt, thus the output folder gets no copy of it. sn3d does not read the file.")


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
        # the name of a reduction folder holds only the last part of the model path. A file system of macOS ignores
        # the letter case of a name, thus the test does too
        foldernames = [str(folder.resolve()).casefold() for folder in outputfolders["dimensionreduce"]]
        if len(set(foldernames)) < len(foldernames):
            samefolderpaths = [
                str(modelpath)
                for modelpath, foldername in zip(args.modelpath, foldernames, strict=True)
                if foldernames.count(foldername) > 1
            ]
            exit_with_error(
                f"The model paths {', '.join(samefolderpaths)} give one output folder, thus the reduction of one"
                " model replaces the reduction of the other",
                "Run the command for one model at a time, with a different -o for each model",
            )
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
        if args.outputfile is None or len(args.modelpath) > 1:
            exit_with_error(
                "A conversion writes the model.txt of one model path into the folder of -o",
                "Give one model path, and give -o with a new folder",
            )
        outputfolders["convertformat"] = [Path(args.outputfile)]
    return outputfolders


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


def make_cylindrical_slice(
    dfmodel: pl.LazyFrame,
    modelmeta: dict[str, t.Any],
    dfelabundances: pl.LazyFrame,
    dfgridcontributions: pl.DataFrame | None,
    sliceaxis: str,
    radialaxis: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None, dict[str, t.Any]]:
    """Make a cylindrical model from one central half-plane of a 3D Cartesian model."""
    if get_model_coordinates(modelmeta) != "cartesian3d":
        msg = "A cylindrical slice needs a 3D Cartesian model"
        raise ValueError(msg)

    if sliceaxis not in {"x", "y", "z"} or radialaxis not in {"+x", "+y", "+z", "-x", "-y", "-z"}:
        msg = "A cylindrical slice needs a Cartesian slice axis and a signed Cartesian radius axis"
        raise ValueError(msg)
    if sliceaxis == radialaxis[1:]:
        msg = "The slice axis and the cylindrical radius axis must be different"
        raise ValueError(msg)

    ncoordgrid = int(modelmeta["ncoordgridx"])
    if ncoordgrid % 2 or any(modelmeta[f"ncoordgrid{axis}"] != ncoordgrid for axis in "yz"):
        msg = "A cylindrical slice needs the same even number of cells on each axis"
        raise ValueError(msg)

    radiusaxis = radialaxis[1]
    verticalaxis = next(axis for axis in ("x", "y", "z") if axis not in {sliceaxis, radiusaxis})
    slicedf = dfmodel.select(f"pos_{sliceaxis}_min").collect()
    sliceedge = get_middle_layer_lower_edge(slicedf, sliceaxis, positive=True)
    radialmid = pl.col(f"pos_{radiusaxis}_mid")
    signedradialmid = radialmid if radialaxis[0] == "+" else -radialmid
    selected = (
        add_derived_cols_to_modeldata(dfmodel, modelmeta)
        .filter((pl.col(f"pos_{sliceaxis}_min") == sliceedge) & (signedradialmid > 0))
        .with_columns(pos_rcyl_mid=signedradialmid, pos_z_mid=pl.col(f"pos_{verticalaxis}_mid"))
        .sort("pos_z_mid", "pos_rcyl_mid")
        .with_row_index("outputcellid", offset=1)
        .with_columns(pl.col("outputcellid").cast(pl.Int32))
        .collect()
    )
    if selected.height != ncoordgrid**2 // 2:
        msg = f"The slice has {selected.height} cells, but the cylindrical grid needs {ncoordgrid**2 // 2}"
        raise ValueError(msg)

    mapping = selected.select("inputcellid", "outputcellid").lazy()
    abundances = (
        dfelabundances
        .join(mapping, on="inputcellid", how="inner")
        .drop("inputcellid")
        .rename({"outputcellid": "inputcellid"})
        .sort("inputcellid")
    )
    contributions = (
        dfgridcontributions
        .lazy()
        .join(mapping, left_on="cellindex", right_on="inputcellid", how="inner")
        .drop("cellindex")
        .rename({"outputcellid": "cellindex"})
        .sort("cellindex", "particleid")
        if dfgridcontributions is not None
        else pl.LazyFrame()
    )
    dfabundances_out, dfcontributions_out = pl.collect_all((abundances, contributions))
    if dfabundances_out["inputcellid"].to_list() != list(range(1, selected.height + 1)):
        msg = "The slice needs one row of elemental abundances for each cell"
        raise ValueError(msg)

    inputcolumns = dfmodel.collect_schema().names()
    modelcolumns = [
        col
        for col in inputcolumns
        if col not in {"inputcellid", "modelgridindex"} and not col.startswith(("pos_", "vel_"))
    ]
    dfmodel_out = selected.select(
        pl.col("outputcellid").alias("inputcellid"), "pos_rcyl_mid", "pos_z_mid", *modelcolumns
    )
    xmax = modelmeta["vmax_cmps"] * modelmeta["t_model_init_days"] * day_to_s
    edgekeys = (*ALL_MODEL_EDGE_KEYS, "firstinputcellid", "formatversion", "mirror_plane")
    modelmeta_out = {
        key: value
        for key, value in modelmeta.items()
        if not key.startswith(("ncoord", "wid_init")) and key not in edgekeys
    }
    modelmeta_out.update({
        "dimensions": 2,
        "coordinates": "cylindrical2d",
        "npts_model": selected.height,
        "ncoordgridrcyl": ncoordgrid // 2,
        "ncoordgridz": ncoordgrid,
        "wid_init_rcyl": 2 * xmax / ncoordgrid,
        "wid_init_z": 2 * xmax / ncoordgrid,
    })
    print(
        f"The slice uses the central layer on the positive {sliceaxis} side. The {radialaxis} axis gives the"
        f" cylindrical radius, and the {verticalaxis} axis gives the z coordinate. Each 2D cell uses the density"
        " and composition of its source 3D cell."
    )
    print("The 2D model assumes axial symmetry about the z axis. Its mass comes from the cylindrical cell volumes.")
    return (
        dfmodel_out,
        dfabundances_out,
        dfcontributions_out if dfgridcontributions is not None else None,
        modelmeta_out,
    )


def reduce_model_dimensions(
    modelpath: Path,
    outdir: Path,
    ndim_out: int,
    outputformatversion: int,
    sliceaxis: str | None = None,
    radialaxis: str | None = None,
) -> None:
    """Write the model of a model path with ndim_out dimensions into outdir, with its abundances and contributions.

    Ye.txt has the cells of the source model, thus the reduced model gets the electron fractions in its column Ye.
    """
    dfmodel, modelmeta = scan_modeldata(modelpath)
    ndim_in = modelmeta["dimensions"]
    if ndim_in <= ndim_out:
        msg = f"Cannot reduce {ndim_in}D model to {ndim_out}D"
        raise ValueError(msg)

    sourcefolder = get_model_input_folder(modelpath)
    dfelabundances = scan_initelemabundances(modelpath)
    dfgridcontributions = get_gridparticlecontributions_or_none(get_gridcontributions_folder(modelpath))
    yefilepath = find_ye_file(sourcefolder)
    require_no_ye_file_of_version_2_model(yefilepath, modelmeta, sourcefolder)
    if yefilepath is not None:
        # sn3d takes the electron fractions of Ye.txt for the source model. Ye.txt has the cells of the source
        # model, thus the reduced model gets these values in its column Ye, also in version 1
        dfmodel = with_ye_of_yefile(dfmodel.collect(), yefilepath).lazy()

    if sliceaxis is not None and radialaxis is not None:
        dfmodel_out, dfelabundances_out, dfgridcontributions_out, modelmeta_out = make_cylindrical_slice(
            dfmodel, modelmeta, dfelabundances, dfgridcontributions, sliceaxis, radialaxis
        )
    else:
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
    if sliceaxis is not None and radialaxis is not None:
        modelmeta_out["headercommentlines"].append(
            f"Central layer on positive {sliceaxis}, with cylindrical radius from {radialaxis}"
        )
    modelmeta_out["formatversion"] = outputformatversion
    if outputformatversion == 1:
        # the reduced model has different cells, thus no Ye.txt of the output folder is of this model
        require_no_ye_file_of_earlier_model(outdir, modelpath)
    if outputformatversion == 2:
        # version 2 has cell ids from 0 in each file of the model
        dfmodel_out, dfelabundances_out, dfgridcontributions_out = with_cell_ids_from_zero(
            dfmodel_out, dfelabundances_out, dfgridcontributions_out
        )
    extracols = (
        [
            col
            for col in dfmodel_out.columns
            if col not in {"inputcellid", "rho", "logrho", "mass_g", "pos_rcyl_mid", "pos_z_mid"}
            and not col.startswith("X_")
        ]
        if sliceaxis is not None
        else []
    )
    # a version 1 model also gets abundances.txt, and a version 2 model gets otherstable columns
    # the overwrite prompt names the Ye.txt and the opacity.txt of an earlier model in the output folder
    save_modeldata(
        dfmodel=dfmodel_out,
        modelmeta=modelmeta_out,
        outpath=outdir,
        extracols=extracols,
        dfelabundances=dfelabundances_out,
        deletecellidfiles=True,
    )
    if dfgridcontributions_out is not None:
        save_gridparticlecontributions(dfgridcontributions_out, outdir / "gridcontributions.txt")
    else:
        delete_gridparticlecontributions(outdir)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Tools to create an ARTIS input model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.make2dmodelfromsliceaxis is not None or args.modelrcoordaxis2d is not None:
        if (
            args.dimensionreduce != 2
            or args.downscale3dgrid
            or args.make2dmodelfromsliceaxis is None
            or args.modelrcoordaxis2d is None
        ):
            exit_with_error(
                "A cylindrical slice needs --dimensionreduce 2 and both slice axis options",
                "Give --make-2d-model-from-slice-axis and --model-r-coord-axis-2d without --downscale3dgrid",
            )
        if args.make2dmodelfromsliceaxis == args.modelrcoordaxis2d[1:]:
            exit_with_error("The slice axis and the cylindrical radius axis must be different")

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
    for action, folders in outputfolders.items():
        if action == "convertformat":
            # a conversion stops for an output folder that holds a model file, thus it replaces no file
            continue
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
            reduce_model_dimensions(
                modelpath,
                outdir,
                ndim_out,
                get_output_format_version(args, modelpath),
                args.make2dmodelfromsliceaxis,
                args.modelrcoordaxis2d,
            )

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
