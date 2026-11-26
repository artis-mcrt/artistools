"""Plot density, Ye, and abundances of a multidimensional ARTIS model as 2D slices or 3D surfaces."""

import argparse
import math
import string
import typing as t
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
from matplotlib import gridspec
from matplotlib.image import AxesImage

from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import get_middle_layer_lower_edge
from artistools.inputmodel.core import get_spherical_image_cell_indices
from artistools.inputmodel.core import scan_modeldata
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_positional_items
from artistools.misc import addarg_show
from artistools.misc import addarg_sliceaxis
from artistools.misc import get_model_coordinates
from artistools.misc import has_mirror_plane
from artistools.misc import import_optional
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import read_wsv
from artistools.misc import require_cartesian3d_model
from artistools.misc import resolve_outputfile
from artistools.misc import resolve_positional_modelpath
from artistools.misc import resolve_slice_axis
from artistools.misc import SPHERICAL_EDGE_COORDINATES
from artistools.plottools import save_figure

type AxisType = t.Literal["x", "y", "z", "r", "rcyl"]


def get_2d_slice_through_3d_model(
    dfmodel: pl.DataFrame,
    sliceaxis: AxisType,
    modelmeta: dict[str, t.Any] | None = None,
    plotaxis1: AxisType | None = None,
    plotaxis2: AxisType | None = None,
    positive_axis: bool = True,
) -> pl.DataFrame:
    """Return the cells of a 3D model in the layer along sliceaxis that touches the origin on the given side.

    The centre layer of an odd grid holds the origin, thus both sides give that layer.
    """
    sliceposition = get_middle_layer_lower_edge(dfmodel, sliceaxis, positive=positive_axis)

    slicedf = dfmodel.filter(pl.col(f"pos_{sliceaxis}_min") == sliceposition)

    if modelmeta is not None and plotaxis1 is not None and plotaxis2 is not None:
        assert slicedf.height == modelmeta[f"ncoordgrid{plotaxis1}"] * modelmeta[f"ncoordgrid{plotaxis2}"]

    return slicedf


def get_point_slice_through_spherical_model(
    dfmodel: pl.DataFrame, modelmeta: dict[str, t.Any], sliceaxis: str | None, positive_axis: bool
) -> tuple[pl.DataFrame, dict[str, t.Any], tuple[AxisType, AxisType]]:
    """Return the points of an image of a spherical2d or spherical3d model, their metadata, and the two plot axes.

    Such a model has no cell with straight edges on the axes of an image. Thus the image is a grid of points with
    equal steps (see get_spherical_image_cell_indices), and each point has the values of the cell that holds it. A
    point outside the grid has null values. The metadata gives the counts of the points as the ncoordgrid values of
    the two plot axes. The pos_ columns give the edges of the points, as for the cells of a slice.

    A spherical2d model gives the plane of the cylindrical radius and z, and sliceaxis is then None. A spherical3d
    model gives the plane normal to sliceaxis through the centre. The plane lies at 1e-6 of the outer radius of the
    innermost shell, on the positive side or on the negative side of that axis. Thus a cell edge in the plane gives
    the cells of that side, and the plane cuts the innermost shell also if that shell is narrow.
    """
    vmax_cmps = float(modelmeta["vmax_cmps"])
    innershellouter_cmps = float(modelmeta["vel_r_edges_kmps"][1]) * km_to_cm
    slicevelocity_cmps = (1.0 if positive_axis else -1.0) * 1.0e-6 * innershellouter_cmps
    cellindices = get_spherical_image_cell_indices(modelmeta, sliceaxis, slicevelocity_cmps)
    pointcount2, pointcount1 = cellindices.shape
    plotaxes: tuple[AxisType, AxisType] = ("rcyl", "z")
    if sliceaxis is not None:
        axeschars: list[AxisType] = ["x", "y", "z"]
        first, second = (axisname for axisname in axeschars if axisname != sliceaxis)
        plotaxes = (first, second)
    print(
        f"The image has {pointcount1} x {pointcount2} points with equal steps in {plotaxes[0]} and {plotaxes[1]}. Each"
        " point takes the value of the cell that holds it. A cell that is narrower than the step of the points can have"
        " no point in the image."
    )

    xmax_tmodel = vmax_cmps * modelmeta["t_model_init_days"] * day_to_s
    lowest1 = 0.0 if sliceaxis is None else -xmax_tmodel
    pointwidth = xmax_tmodel / (pointcount2 // 2)
    index1 = np.tile(np.arange(pointcount1), pointcount2)
    index2 = np.repeat(np.arange(pointcount2), pointcount1)
    isingrid = pl.Series(cellindices.ravel() >= 0)
    dfpoints = (
        dfmodel
        .sort("modelgridindex")[np.maximum(cellindices.ravel(), 0)]
        .with_columns(pl.when(isingrid).then(pl.all()))
        .with_columns(
            pl.Series(f"pos_{plotaxes[0]}_min", lowest1 + index1 * pointwidth),
            pl.Series(f"pos_{plotaxes[0]}_max", lowest1 + (index1 + 1) * pointwidth),
            pl.Series(f"pos_{plotaxes[1]}_min", -xmax_tmodel + index2 * pointwidth),
            pl.Series(f"pos_{plotaxes[1]}_max", -xmax_tmodel + (index2 + 1) * pointwidth),
        )
    )
    pointmeta = modelmeta | {f"ncoordgrid{plotaxes[0]}": pointcount1, f"ncoordgrid{plotaxes[1]}": pointcount2}
    return dfpoints, pointmeta, plotaxes


def get_full_grid_of_cylindrical_model(
    dfmodel: pl.DataFrame, modelmeta: dict[str, t.Any]
) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return the cells and the metadata of the full grid of a 2D cylindrical model, from -vmax to vmax in z.

    A mirrored model holds the cells with z >= 0. Each such cell also fills its mirror cell, thus the rows of the half
    with z < 0 come first, in the sequence of the grid. The image then shows the full plane.
    """
    if not has_mirror_plane(modelmeta):
        return dfmodel, modelmeta
    ncoordgridrcyl = int(modelmeta["ncoordgridrcyl"])
    dfupper = dfmodel.sort("modelgridindex")
    mirrorcols = {"pos_z_min": "pos_z_max", "pos_z_max": "pos_z_min", "pos_z_mid": "pos_z_mid"}
    dflower = dfupper.with_columns(
        (-pl.col(othercol)).alias(col) for col, othercol in mirrorcols.items() if col in dfupper.columns
    ).sort(pl.col("modelgridindex") // ncoordgridrcyl, descending=True, maintain_order=True)
    return pl.concat([dflower, dfupper]), modelmeta | {"ncoordgridz": 2 * int(modelmeta["ncoordgridz"])}


def plot_slice_modelcolumn(
    ax: mplax.Axes,
    dfmodelslice: pl.DataFrame,
    modelmeta: dict[str, t.Any],
    colname: str,
    plotaxis1: str,
    plotaxis2: str,
    t_model_d: float,
    args: argparse.Namespace,
) -> AxesImage:
    """Draw one model column as a colour image on the axes, and return the image."""
    print(f"Plotting {colname}")
    colorscale = (
        (dfmodelslice[colname] * dfmodelslice["rho"]) if colname.startswith("X_") else dfmodelslice[colname]
    ).to_numpy()

    if dfmodelslice["rho"].null_count() > 0:
        # a point of an image of a spherical model that is outside the grid has no cell. The floor of a log scale
        # must not give it a colour
        colorscale = np.ma.masked_where(dfmodelslice["rho"].is_null().to_numpy(), colorscale)

    if args.hideemptycells:
        # Don't plot empty cells:
        colorscale = np.ma.masked_where(colorscale == 0.0, colorscale)

    if args.logcolorscale:
        # logscale for colormap
        if args.floorval is not None:
            if not args.floorval > 0:
                # the floor is a linear value that the clamp below applies before the logarithm
                msg = f"--floorval must be positive with --logcolorscale, got {args.floorval}"
                raise ValueError(msg)
            # np.ma.where keeps a masked array masked. np.where and a comprehension both give a
            # plain array, thus they put every cell that --hideemptycells masked back on the plot
            with np.errstate(invalid="ignore"):
                colorscale = np.ma.where(
                    np.isfinite(colorscale) & (colorscale >= args.floorval), colorscale, args.floorval
                )
        with np.errstate(divide="ignore"):
            colorscale = np.log10(colorscale)

    unitfactor = 1.0 / {"c": C_cm_per_s, "kmps": km_to_cm}[args.velocityunitplot]
    t_model_s = t_model_d * day_to_s

    # turn 1D flattened array back into 2D array
    valuegrid = colorscale.reshape((modelmeta[f"ncoordgrid{plotaxis2}"], modelmeta[f"ncoordgrid{plotaxis1}"]))

    posmin_ax1 = dfmodelslice[f"pos_{plotaxis1}_min"].min()
    posmax_ax1 = dfmodelslice[f"pos_{plotaxis1}_max"].max()
    posmin_ax2 = dfmodelslice[f"pos_{plotaxis2}_min"].min()
    posmax_ax2 = dfmodelslice[f"pos_{plotaxis2}_max"].max()
    assert isinstance(posmin_ax1, int | float)
    assert isinstance(posmax_ax1, int | float)
    assert isinstance(posmin_ax2, int | float)
    assert isinstance(posmax_ax2, int | float)
    vmin_ax1 = posmin_ax1 / t_model_s * unitfactor
    vmax_ax1 = posmax_ax1 / t_model_s * unitfactor
    vmin_ax2 = posmin_ax2 / t_model_s * unitfactor
    vmax_ax2 = posmax_ax2 / t_model_s * unitfactor
    if colname == "rho":
        if args.logcolorscale:
            # the colour scale holds log10 of the density, thus the linear floor moves to log units here
            vmin = -15 if args.floorval is None else math.log10(args.floorval)
            vmax = -7
        else:
            vmin = 1e-15 if args.floorval is None else args.floorval
            vmax = 1e-7
    elif colname == "Ye":
        assert not args.logcolorscale, "log colorscale not supported for Ye"
        vmin = 0 if args.floorval is None else args.floorval
        vmax = 0.6
    else:
        vmin = None
        vmax = None
    im = ax.imshow(
        valuegrid,
        cmap="viridis",
        interpolation="nearest",
        extent=(vmin_ax1, vmax_ax1, vmin_ax2, vmax_ax2),
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    if "_" in colname:
        ax.annotate(
            colname.split("_")[1],
            color="white",
            xy=(0.9, 0.9),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
        )

    return im


def add_ye_from_yefile(lzdfmodel: pl.LazyFrame, modelpath: Path | str, modelmeta: dict[str, t.Any]) -> pl.LazyFrame:
    """Add the Ye column of Ye.txt to a model whose model.txt holds no Ye column.

    Ye.txt gives the number of lines first, and then the cell id and Ye of each cell. The ids are the ids of
    model.txt, thus a join on the id is correct also for a file that holds only some of the cells. A model of
    format version 2 has no Ye.txt, thus the function stops for such a model.
    """
    if modelmeta.get("formatversion", 1) == 2:
        msg = (
            f"The model.txt of {modelpath} has format version 2 and no column Ye. sn3d reads no Ye.txt for format"
            " version 2"
        )
        raise ValueError(msg)
    dfye = read_wsv(Path(modelpath) / "Ye.txt", has_header=False, skip_rows=1, new_columns=["inputcellid", "Ye"])
    return lzdfmodel.join(
        dfye.lazy().select(pl.col("inputcellid").cast(pl.Int32), pl.col("Ye").cast(pl.Float32)),
        on="inputcellid",
        how="left",
        maintain_order="left",
    )


def get_colorbar_label(colname: str, logcolorscale: bool) -> str:
    """Return the label of the colour bar of one panel, which names the quantity of that panel."""
    if colname == "rho":
        quantity = r"$\rho$ [g/cm³]"
    elif colname.startswith("X_"):
        # the panel of a mass fraction shows the density of that species, which is the mass fraction times rho
        quantity = rf"$\rho_\mathrm{{{colname.removeprefix('X_')}}}$ [g/cm³]"
    else:
        quantity = colname
    return f"log10({quantity})" if logcolorscale else quantity


def get_table_column_name(colname: str, logcolorscale: bool) -> str:
    """Return the name of the column of one panel in the data table, with the unit of the panel."""
    if colname == "rho":
        name = "rho_g_per_cm3"
    elif colname.startswith("X_"):
        # the panel of a mass fraction shows the density of that species, as the label of its colour bar says
        name = f"rho_{colname.removeprefix('X_')}_g_per_cm3"
    else:
        name = colname
    return f"log10_{name}" if logcolorscale else name


def get_slice_table(
    images: Sequence[AxesImage],
    colnames: Sequence[str],
    plotaxis1: str,
    plotaxis2: str,
    logcolorscale: bool,
    velocityunitplot: str,
) -> pl.DataFrame:
    """Return one row for each pixel of the slice, with the velocities of the pixel centre and the value of each panel.

    The values come from the images, thus they have the floor and the log10 of the colour scale. A cell that
    --hideemptycells hides has a null value. The velocities use c or km/s, as -velocityunitplot selects.
    """
    xmin, xmax, ymin, ymax = images[0].get_extent()
    firstarray = images[0].get_array()
    assert firstarray is not None
    nrows, ncols = firstarray.shape[:2]
    xedges = np.linspace(xmin, xmax, ncols + 1)
    yedges = np.linspace(ymin, ymax, nrows + 1)
    unitsuffix = {"c": "on_c", "kmps": "kmps"}[velocityunitplot]
    # origin="lower" puts row 0 of the array at ymin, and each row of the array holds one value of y
    columns = [
        pl.Series(f"vel_{plotaxis1}_{unitsuffix}", np.tile((xedges[:-1] + xedges[1:]) / 2.0, nrows)),
        pl.Series(f"vel_{plotaxis2}_{unitsuffix}", np.repeat((yedges[:-1] + yedges[1:]) / 2.0, ncols)),
    ]
    for image, colname in zip(images, colnames, strict=True):
        valuegrid = image.get_array()
        assert valuegrid is not None
        # polars gives null for each masked value
        columns.append(pl.Series(get_table_column_name(colname, logcolorscale), valuegrid.ravel()))
    return pl.DataFrame(columns)


def plot_2d_initial_abundances(modelpath: Path | str, args: argparse.Namespace) -> None:
    """Plot each of args.plotvars as a 2D slice through the model and save the figure."""
    # if the species doesn't end in a number (isotope, e.g. Sr92) then we need to also get element abundances (e.g., Sr)
    get_elemabundances = any(plotvar[-1] not in string.digits for plotvar in args.plotvars)
    lzdfmodel, modelmeta = scan_modeldata(modelpath, get_elemabundances=get_elemabundances)
    if modelmeta["dimensions"] not in {2, 3}:
        msg = f"plotinitialcomposition plots a 2D or a 3D model, but the model in {modelpath} is 1D"
        raise ValueError(msg)

    if "Ye" in args.plotvars and "Ye" not in lzdfmodel.collect_schema().names():
        # the model file can hold no Ye column, and then Ye.txt gives it, as for the 3D plot
        lzdfmodel = add_ye_from_yefile(lzdfmodel, modelpath, modelmeta)

    modelcolumns = lzdfmodel.collect_schema().names()
    colnames = [plotvar if plotvar in modelcolumns else f"X_{plotvar.title()}" for plotvar in args.plotvars]
    if missingcolumns := [colname for colname in colnames if colname not in modelcolumns]:
        msg = f"The model in {modelpath} holds no column {', '.join(missingcolumns)} to plot"
        raise ValueError(msg)
    # the plot reads the cell edges, which are derived columns in 2D. The other derived columns stay out of memory
    dfmodel = (
        add_derived_cols_to_modeldata(lzdfmodel, modelmeta=modelmeta)
        .select(
            cs.by_name(lzdfmodel.collect_schema().names()) | (cs.starts_with("pos_") & cs.ends_with("_min", "_max"))
        )
        .collect()
    )

    coordinates = get_model_coordinates(modelmeta)
    if coordinates in SPHERICAL_EDGE_COORDINATES:
        if coordinates == "spherical3d":
            print(f"The image shows the plane {args.sliceaxis} = 0")
        # the image takes one row of the model for each point, thus it keeps only the plotted columns
        df2dslice, modelmeta, (plotaxis1, plotaxis2) = get_point_slice_through_spherical_model(
            dfmodel.select(*dict.fromkeys(["modelgridindex", "rho", *colnames])),
            modelmeta,
            args.sliceaxis if coordinates == "spherical3d" else None,
            args.positive_axis,
        )
    elif modelmeta["dimensions"] == 3:
        sliceaxis: AxisType = args.sliceaxis

        axeschars: list[AxisType] = ["x", "y", "z"]
        plotaxis1 = next(ax for ax in axeschars if ax != sliceaxis)
        plotaxis2 = next(ax for ax in axeschars if ax not in {sliceaxis, plotaxis1})
        print(f"Plotting slice through {sliceaxis}=0, plotting {plotaxis1} vs {plotaxis2}")

        df2dslice = get_2d_slice_through_3d_model(
            dfmodel=dfmodel,
            modelmeta=modelmeta,
            sliceaxis=sliceaxis,
            plotaxis1=plotaxis1,
            plotaxis2=plotaxis2,
            positive_axis=args.positive_axis,
        )
    elif modelmeta["dimensions"] == 2:
        df2dslice, modelmeta = get_full_grid_of_cylindrical_model(dfmodel, modelmeta)
        plotaxis1 = "rcyl"
        plotaxis2 = "z"
    else:
        msg = f"Model dimensions {modelmeta['dimensions']} not supported"
        raise ValueError(msg)

    nrows = 1
    ncols = len(args.plotvars)
    xfactor = 1 if modelmeta["dimensions"] == 3 else 0.5
    figwidth = 5.0
    fig = plt.figure(
        figsize=(figwidth * xfactor * ncols, figwidth * nrows), tight_layout={"pad": 1.0, "w_pad": 0.0, "h_pad": 0.0}
    )
    gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios=[0.05, 1], width_ratios=[1] * ncols)

    axes = [fig.add_subplot(gs[1, y]) for y in range(ncols)]

    # each panel has its own colour limits, e.g. a fixed range for rho and an automatic range for a mass fraction,
    # thus each panel gets its own colour bar
    images: list[AxesImage] = []
    for column, (colname, ax) in enumerate(zip(colnames, axes, strict=True)):
        im = plot_slice_modelcolumn(
            ax, df2dslice, modelmeta, colname, plotaxis1, plotaxis2, modelmeta["t_model_init_days"], args
        )
        images.append(im)
        cbar = fig.colorbar(im, cax=fig.add_subplot(gs[0, column]), location="top", use_gridspec=True)
        cbar.set_label(get_colorbar_label(colname, args.logcolorscale))

    if args.velocityunitplot == "kmps":
        for ax in axes:
            ax.xaxis.set_major_formatter(lambda value, _: f"{value / 1e3:g}")
            ax.yaxis.set_major_formatter(lambda value, _: f"{value / 1e3:g}")

    unitlabel = {"c": "$c$", "kmps": r"$10^3$ km/s"}[args.velocityunitplot]
    xlabel = r"v$_{" + str(plotaxis1) + r"}$ [" + unitlabel + "]"
    ylabel = r"v$_{" + str(plotaxis2) + r"}$ [" + unitlabel + "]"

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    defaultfilename = f"plotcomposition_{','.join(v.lower() for v in args.plotvars)}.pdf"
    outfilename = resolve_outputfile(args.outputfile, defaultfilename)

    if getattr(args, "print_data", False) or getattr(args, "write_data", False):
        dfslice = get_slice_table(images, colnames, plotaxis1, plotaxis2, args.logcolorscale, args.velocityunitplot)
        output_data(args, {"": dfslice}, outfilename)
    save_figure(fig, outfilename, args=args, format="pdf")


def get_3d_tables(
    xgrid: npt.NDArray[np.floating],
    values: npt.NDArray[np.floating],
    valuename: str,
    surfacelevels: npt.NDArray[np.floating],
    velocityunitplot: str,
) -> dict[str, pl.DataFrame]:
    """Return the values of the cells that the isosurfaces come from, and the value of each isosurface.

    The pyvista mesh has one point at the lower corner of each cell. The velocities use c or km/s, as -velocityunitplot
    selects.
    The axes of `values` have the order (x, y, z).
    """
    x, y, z = np.meshgrid(xgrid, xgrid, xgrid, indexing="ij")
    unitsuffix = {"c": "on_c", "kmps": "kmps"}[velocityunitplot]
    return {
        "": pl.DataFrame({
            f"vel_x_min_{unitsuffix}": x.ravel(order="F"),
            f"vel_y_min_{unitsuffix}": y.ravel(order="F"),
            f"vel_z_min_{unitsuffix}": z.ravel(order="F"),
            valuename: values.ravel(order="F"),
        }),
        "_isosurfaces": pl.DataFrame({f"{valuename}_of_isosurface": surfacelevels}),
    }


def get_levels_with_surface(surf: t.Any, scalarname: str, levels: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Return the levels of a contour of pyvista that give a surface, e.g. not a level outside the range of the data.

    Each point of the contour holds the value of its level, thus a level with no point gives no surface.
    """
    surfacevalues = np.unique(np.asarray(surf[scalarname], dtype=float)) if surf.n_points else np.empty(0)
    levelarray = np.asarray(levels, dtype=float)
    hassurface = [bool(np.any(np.isclose(surfacevalues, level, rtol=1e-6, atol=0.0))) for level in levelarray]
    return levelarray[np.asarray(hassurface, dtype=bool)]


def make_3d_plot(modelpath: Path, args: argparse.Namespace) -> None:
    """Render an isosurface of the 3D model with pyvista, coloured by the first of args.plotvars."""
    pv = import_optional("pyvista")

    # the dark theme of pyvista has a black background and white text, as --darkmode gives a matplotlib plot
    pv.set_plot_theme("dark" if args.darkmode else "document")

    # choose what surface will be coloured by
    plotvar = "rho" if "rho" in args.plotvars else "Ye" if "Ye" in args.plotvars else args.plotvars[0]
    # an element such as Sr needs the element abundances, and an isotope such as Sr92 is a column of the model
    plmodel, modelmeta = scan_modeldata(modelpath, get_elemabundances=plotvar[-1] not in string.digits)
    # the plot puts the cells on a cube of x, y, and z
    require_cartesian3d_model(modelmeta, "--plot3d needs", "Remove --plot3d to plot a slice of the model")
    # a column such as tracercount keeps its name, and an element or an isotope names its mass fraction, as in 2D
    coloursurfaceby = plotvar if plotvar in {*plmodel.collect_schema().names(), "Ye"} else f"X_{plotvar.title()}"
    print(f"Colours set by {coloursurfaceby}")
    vmax = modelmeta["vmax_cmps"]
    if "Ye" in args.plotvars and "Ye" not in plmodel.collect_schema().names():
        # the model file can hold no Ye column, and then Ye.txt gives it
        plmodel = add_ye_from_yefile(plmodel, modelpath, modelmeta)
    model = plmodel.select(cs.by_name({"rho", coloursurfaceby}, require_all=False)).collect()

    # generate grid from data
    grid = round(len(model["rho"]) ** (1.0 / 3.0))
    surfacearr = np.asarray(model[coloursurfaceby], dtype=float)
    vmax /= {"c": C_cm_per_s, "kmps": km_to_cm}[args.velocityunitplot]
    # cells are ordered with x varying fastest, i.e. Fortran order on (nx, ny, nz)
    surfacecolorscale = surfacearr.reshape((grid, grid, grid), order="F")
    xgrid = -vmax + 2 * np.arange(grid) * vmax / grid

    # the first axis of the data is x, thus the grid needs matrix indexing and not the default xy indexing
    x, y, z = np.meshgrid(xgrid, xgrid, xgrid, indexing="ij")

    mesh: t.Any = pv.StructuredGrid(x, y, z)
    print(mesh)  # tells you the properties of the mesh

    mesh[coloursurfaceby] = surfacecolorscale.ravel(order="F")  # add data to the mesh
    minval = np.min(mesh[coloursurfaceby][np.nonzero(mesh[coloursurfaceby])])  # minimum non zero value
    print(f"{coloursurfaceby} minumin {minval}, maximum {max(mesh[coloursurfaceby])}")

    if not args.surfaces3d:
        surfacepositions = np.linspace(min(mesh[coloursurfaceby]), max(mesh[coloursurfaceby]), num=10)
        print(f"Using default surfaces {surfacepositions} \n define these with --surfaces3d for better results")
    else:
        surfacepositions = args.surfaces3d  # expects an array of surface positions

    surf = mesh.contour(surfacepositions, scalars=coloursurfaceby)  # create isosurfaces
    if getattr(args, "print_data", False) or getattr(args, "write_data", False):
        output_data(
            args,
            get_3d_tables(
                xgrid,
                surfacecolorscale,
                coloursurfaceby,
                get_levels_with_surface(surf, coloursurfaceby, surfacepositions),
                args.velocityunitplot,
            ),
            modelpath / "3Dplot.png",
        )

    sargs = {
        "height": 0.25,
        "vertical": True,
        "position_x": 0.05,
        "position_y": 0.1,
        "title_font_size": 22,
        "label_font_size": 22,
    }

    plotter: t.Any = pv.Plotter()
    plotcoloropacity = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # some choices: 'linear' 'sigmoid'
    axistitles = (
        ("vx / c", "vy / c", "vz / c")
        if args.velocityunitplot == "c"
        else ("vx [10³ km/s]", "vy [10³ km/s]", "vz [10³ km/s]")
    )
    axes_ranges = tuple(bound / 1e3 for bound in mesh.bounds) if args.velocityunitplot == "kmps" else None
    plotter.show_bounds(
        mesh=mesh,
        axes_ranges=axes_ranges,
        grid=False,
        location="outer",
        xtitle=axistitles[0],
        ytitle=axistitles[1],
        ztitle=axistitles[2],
        ticks="inside",
        minor_ticks=False,
        font_size=28,
        bold=False,
    )
    plotter.add_mesh(surf, opacity=plotcoloropacity, scalar_bar_args=sargs, cmap="coolwarm_r")

    plotter.camera_position = "xz"
    assert plotter.camera is not None
    plotter.camera.azimuth = 45.0
    plotter.camera.elevation = 10.0
    plotter.show(screenshot=modelpath / "3Dplot.png", auto_close=False)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_positional_items(
        parser,
        dest="plotvars",
        metavar="plotvar",
        helptext=(
            "Element symbols (Fe, Ni, Sr) for mass fraction or other model columns (rho, tracercount) to plot"
            " (default: rho), then the model folder, e.g. rho Fe mymodel"
        ),
    )

    # the default is None, thus resolve_positional_modelpath sees a -modelpath that the user gave
    addarg_modelpath(parser)

    addarg_output(parser, kind="file", helptext="Filename for PDF file")

    parser.add_argument("--logcolorscale", action="store_true", help="Use log scale for colour map")

    parser.add_argument("--hideemptycells", action="store_true", help="Don't plot empty cells")

    parser.add_argument("--plot3d", action="store_true", help="Make 3D plot")

    parser.add_argument("-surfaces3d", type=float, nargs="+", help="Define positions of surfaces for 3D plots")

    parser.add_argument("-floorval", default=None, type=float, help="Set a floor value for colorscale. Expects float")

    parser.add_argument(
        "-velocityunitplot",
        choices=["c", "kmps"],
        default="c",
        help=("Set the velocity-axis unit to c or kmps. Default is c"),
    )

    addarg_sliceaxis(parser, default="+z", helptext="Slice axis for 2D plots")
    addarg_data_output(parser)
    addarg_show(parser)
    addarg_darkmode(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS input model composition."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    # the model folder is the last positional argument, thus "plotinitialcomposition Fe mymodel" reads mymodel
    args.plotvars = resolve_positional_modelpath(args, "plotvars") or ["rho"]

    resolve_slice_axis(args)

    if args.plot3d:
        make_3d_plot(Path(args.modelpath), args)
        return

    plot_2d_initial_abundances(args.modelpath, args)


# the names before the rule invalid-function-name. Each alias gives a DeprecationWarning that names the new
# function
@warnings.deprecated("get_2D_slice_through_3d_model is deprecated. Use get_2d_slice_through_3d_model.")
def get_2D_slice_through_3d_model(  # ruff:ignore[invalid-function-name]
    dfmodel: pl.DataFrame,
    sliceaxis: AxisType,
    modelmeta: dict[str, t.Any] | None = None,
    plotaxis1: AxisType | None = None,
    plotaxis2: AxisType | None = None,
    positive_axis: bool = True,
) -> pl.DataFrame:
    """Return the result of get_2d_slice_through_3d_model, the new name of this function."""
    return get_2d_slice_through_3d_model(dfmodel, sliceaxis, modelmeta, plotaxis1, plotaxis2, positive_axis)
