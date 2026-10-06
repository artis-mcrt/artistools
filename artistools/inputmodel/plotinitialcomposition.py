"""Plot density, Ye, and abundances of a multidimensional ARTIS model as 2D slices or 3D surfaces."""

import argparse
import math
import string
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
from matplotlib import gridspec
from matplotlib.image import AxesImage

from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import get_middle_layer_lower_edge
from artistools.inputmodel.core import get_modeldata
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_positional_items
from artistools.misc import addarg_show
from artistools.misc import import_optional
from artistools.misc import parse_cli_args
from artistools.misc import read_wsv
from artistools.misc import resolve_outputfile
from artistools.misc import resolve_positional_modelpath
from artistools.plottools import save_figure

type AxisType = t.Literal["x", "y", "z", "r", "rcyl"]


def get_2D_slice_through_3d_model(
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
    print(f"plotting {colname}")
    colorscale = (
        (dfmodelslice[colname] * dfmodelslice["rho"]) if colname.startswith("X_") else dfmodelslice[colname]
    ).to_numpy()

    if args.hideemptycells:
        # Don't plot empty cells:
        colorscale = np.ma.masked_where(colorscale == 0.0, colorscale)

    if args.logcolorscale:
        # logscale for colormap
        if args.floorval is not None:
            if not args.floorval > 0:
                # the floor is a linear value that the clamp below applies before the logarithm
                msg = f"-floorval must be positive with --logcolorscale, got {args.floorval}"
                raise ValueError(msg)
            # np.ma.where keeps a masked array masked. np.where and a comprehension both give a
            # plain array, thus they put every cell that --hideemptycells masked back on the plot
            with np.errstate(invalid="ignore"):
                colorscale = np.ma.where(
                    np.isfinite(colorscale) & (colorscale >= args.floorval), colorscale, args.floorval
                )
        with np.errstate(divide="ignore"):
            colorscale = np.log10(colorscale)

    cmps_to_beta = 1.0 / C_cm_per_s
    unitfactor = cmps_to_beta
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


def add_ye_from_yefile(lzdfmodel: pl.LazyFrame, modelpath: Path | str) -> pl.LazyFrame:
    """Add the Ye column of Ye.txt to a model whose model.txt holds no Ye column.

    Ye.txt gives the number of lines first, and then the cell id and Ye of each cell. The ids are the ids of
    model.txt, thus a join on the id is correct also for a file that holds only some of the cells.
    """
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


def plot_2d_initial_abundances(modelpath: Path | str, args: argparse.Namespace) -> None:
    """Plot each of args.plotvars as a 2D slice through the model and save the figure."""
    # if the species doesn't end in a number (isotope, e.g. Sr92) then we need to also get element abundances (e.g., Sr)
    get_elemabundances = any(plotvar[-1] not in string.digits for plotvar in args.plotvars)
    lzdfmodel, modelmeta = get_modeldata(modelpath, get_elemabundances=get_elemabundances)
    if modelmeta["dimensions"] not in {2, 3}:
        msg = f"plotinitialcomposition plots a 2D or a 3D model, but the model in {modelpath} is 1D"
        raise ValueError(msg)

    if "Ye" in args.plotvars and "Ye" not in lzdfmodel.collect_schema().names():
        # the model file can hold no Ye column, and then Ye.txt gives it, as for the 3D plot
        lzdfmodel = add_ye_from_yefile(lzdfmodel, modelpath)

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

    if modelmeta["dimensions"] == 3:
        sliceaxis: AxisType = args.sliceaxis

        axeschars: list[AxisType] = ["x", "y", "z"]
        plotaxis1: AxisType = next(ax for ax in axeschars if ax != sliceaxis)
        plotaxis2: AxisType = next(ax for ax in axeschars if ax not in {sliceaxis, plotaxis1})
        print(f"Plotting slice through {sliceaxis}=0, plotting {plotaxis1} vs {plotaxis2}")

        df2dslice = get_2D_slice_through_3d_model(
            dfmodel=dfmodel,
            modelmeta=modelmeta,
            sliceaxis=sliceaxis,
            plotaxis1=plotaxis1,
            plotaxis2=plotaxis2,
            positive_axis=args.positive_axis,
        )
    elif modelmeta["dimensions"] == 2:
        df2dslice = dfmodel
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
    for column, (colname, ax) in enumerate(zip(colnames, axes, strict=True)):
        im = plot_slice_modelcolumn(
            ax, df2dslice, modelmeta, colname, plotaxis1, plotaxis2, modelmeta["t_model_init_days"], args
        )
        cbar = fig.colorbar(im, cax=fig.add_subplot(gs[0, column]), location="top", use_gridspec=True)
        cbar.set_label(get_colorbar_label(colname, args.logcolorscale))

    xlabel = r"v$_{" + str(plotaxis1) + r"}$ [$c$]"
    ylabel = r"v$_{" + str(plotaxis2) + r"}$ [$c$]"

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    defaultfilename = f"plotcomposition_{','.join(v.lower() for v in args.plotvars)}.pdf"
    outfilename = resolve_outputfile(args.outputfile, defaultfilename)

    save_figure(fig, outfilename, args=args, format="pdf")


def make_3d_plot(modelpath: Path, args: argparse.Namespace) -> None:
    """Render an isosurface of the 3D model with pyvista, coloured by the first of args.plotvars."""
    pv = import_optional("pyvista")

    # set white background
    pv.set_plot_theme("document")

    # choose what surface will be coloured by
    plotvar = "rho" if "rho" in args.plotvars else "Ye" if "Ye" in args.plotvars else args.plotvars[0]
    # an element such as Sr needs the element abundances, and an isotope such as Sr92 is a column of the model
    plmodel, modelmeta = get_modeldata(modelpath, get_elemabundances=plotvar[-1] not in string.digits)
    # a column such as tracercount keeps its name, and an element or an isotope names its mass fraction, as in 2D
    coloursurfaceby = plotvar if plotvar in {*plmodel.collect_schema().names(), "Ye"} else f"X_{plotvar.title()}"
    print(f"Colours set by {coloursurfaceby}")
    vmax = modelmeta["vmax_cmps"]
    if "Ye" in args.plotvars and "Ye" not in plmodel.collect_schema().names():
        # the model file can hold no Ye column, and then Ye.txt gives it
        plmodel = add_ye_from_yefile(plmodel, modelpath)
    model = plmodel.select(cs.by_name({"rho", coloursurfaceby}, require_all=False)).collect()

    # generate grid from data
    grid = round(len(model["rho"]) ** (1.0 / 3.0))
    surfacearr = np.asarray(model[coloursurfaceby], dtype=float)
    vmax /= C_cm_per_s
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
        print(f"Using default surfaces {surfacepositions} \n define these with -surfaces3d for better results")
    else:
        surfacepositions = args.surfaces3d  # expects an array of surface positions

    surf = mesh.contour(surfacepositions, scalars=coloursurfaceby)  # create isosurfaces

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
    plotter.show_bounds(
        mesh,
        grid=False,
        location="outer",
        xlabel="vx / c",
        ylabel="vy / c",
        zlabel="vz / c",
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
        "-axis",
        default="+z",
        choices=["x", "y", "z", "+x", "-x", "+y", "-y", "+z", "-z"],
        help="Slice axis for 2D plots. Hint: for negative use e.g. -axis=-z",
    )
    addarg_show(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS input model composition."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    # the model folder is the last positional argument, thus "plotinitialcomposition Fe mymodel" reads mymodel
    args.plotvars = resolve_positional_modelpath(args, "plotvars") or ["rho"]

    # a bare axis name, e.g. "z", gives the positive side
    args.positive_axis = args.axis[0] != "-"
    args.sliceaxis = args.axis.lstrip("+-")

    if args.plot3d:
        make_3d_plot(Path(args.modelpath), args)
        return

    plot_2d_initial_abundances(args.modelpath, args)
