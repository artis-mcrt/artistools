#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import math
import typing as t
from pathlib import Path

import argcomplete
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

import artistools as at

if t.TYPE_CHECKING:
    import pandas as pd


def plot_2d_initial_abundances(modelpath: Path, args: argparse.Namespace) -> None:
    model = at.inputmodel.get_2d_modeldata(modelpath)
    abundances = at.inputmodel.get_initelemabundances(modelpath)

    abundances["inputcellid"] = abundances["inputcellid"].apply(float)

    merge_dfs = model.merge(abundances, how="inner", on="inputcellid")

    with Path(modelpath, "model.txt").open() as fmodelin:
        fmodelin.readline()  # npts r, npts z
        t_model = float(fmodelin.readline())  # days
        vmax = float(fmodelin.readline())  # v_max in [cm/s]

    r = merge_dfs["cellpos_mid[r]"] / t_model * (u.cm / u.day).to("km/s") / 10**3
    z = merge_dfs["cellpos_mid[z]"] / t_model * (u.cm / u.day).to("km/s") / 10**3

    colname = f"X_{args.plotvars}"
    font = {"weight": "bold", "size": 18}

    f = plt.figure(figsize=(4, 5))
    ax = f.add_subplot(111)
    im = ax.scatter(r, z, c=merge_dfs[colname], marker="8")

    f.colorbar(im)
    plt.xlabel(r"v$_x$ in 10$^3$ km/s", fontsize="x-large")  # , fontweight='bold')
    plt.ylabel(r"v$_z$ in 10$^3$ km/s", fontsize="x-large")  # , fontweight='bold')
    plt.text(20, 25, args.plotvars, color="white", fontweight="bold", fontsize="x-large")
    plt.tight_layout()
    # ax.labelsize: 'large'
    # plt.title(f'At {sliceaxis} = {sliceposition}')

    outfilename = f"plotcomposition{args.plotvars}.pdf"
    plt.savefig(Path(modelpath) / outfilename, format="pdf")
    print(f"Saved {outfilename}")


def get_2D_slice_through_3d_model(
    dfmodel: pd.DataFrame,
    sliceaxis: t.Literal["x", "y", "z"],
    modelmeta: dict[str, t.Any] | None = None,
    plotaxis1: t.Literal["x", "y", "z"] | None = None,
    plotaxis2: t.Literal["x", "y", "z"] | None = None,
    sliceindex: int | None = None,
) -> pd.DataFrame:
    if not sliceindex:
        # get midpoint
        sliceposition: float = dfmodel.iloc[(dfmodel["pos_x_min"]).abs().argsort()][:1]["pos_x_min"].item()
        # Choose position to slice. This gets minimum absolute value as the closest to 0
    else:
        cell_boundaries = []
        for x in dfmodel[f"pos_{sliceaxis}_min"]:
            if x not in cell_boundaries:
                cell_boundaries.append(x)
        sliceposition = cell_boundaries[sliceindex]

    slicedf = dfmodel.loc[dfmodel[f"pos_{sliceaxis}_min"] == sliceposition]

    if modelmeta is not None and plotaxis1 is not None and plotaxis2 is not None:
        assert slicedf.shape[0] == modelmeta[f"ncoordgrid{plotaxis1}"] * modelmeta[f"ncoordgrid{plotaxis2}"]

    return slicedf


def plot_slice_modelcol(ax, dfmodelslice, modelmeta, colname, plotaxis1, plotaxis2, t_model_d, args):
    print(f"plotting {colname}")
    colorscale = (
        dfmodelslice[colname] * dfmodelslice["rho"] if colname.startswith("X_") else dfmodelslice[colname]
    ).to_numpy()

    if args.hideemptycells:
        # Don't plot empty cells:
        colorscale = np.ma.masked_where(colorscale == 0.0, colorscale)

    if args.logcolorscale:
        # logscale for colormap
        with np.errstate(divide="ignore"):
            colorscale = np.log10(colorscale)

    normalise_between_0_and_1 = False
    if normalise_between_0_and_1:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scaledmap = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
        scaledmap.set_array([])
        colorscale = scaledmap.to_rgba(colorscale)  # colorscale fixed between 0 and 1
    else:
        scaledmap = None

    cmps_to_kmps = 1e-5
    cmps_to_beta = 1.0 / (2.99792458e10)
    unitfactor = cmps_to_beta
    t_model_s = t_model_d * 86400.0

    # take flat array and turn in into 2D array
    grid = colorscale.reshape((modelmeta[f"ncoordgrid{plotaxis1}"], modelmeta[f"ncoordgrid{plotaxis2}"]))

    im = ax.imshow(
        grid,
        cmap="viridis",
        interpolation="nearest",
        extent=(
            dfmodelslice[f"pos_{plotaxis1}_min"].min() / t_model_s * unitfactor,
            dfmodelslice[f"pos_{plotaxis1}_max"].max() / t_model_s * unitfactor,
            dfmodelslice[f"pos_{plotaxis2}_min"].min() / t_model_s * unitfactor,
            dfmodelslice[f"pos_{plotaxis2}_max"].max() / t_model_s * unitfactor,
        ),
        origin="lower",
        # vmin=0.0,
        # vmax=1.0,
    )

    plot_vmax = 0.2
    ax.set_ylim(bottom=-plot_vmax, top=plot_vmax)
    ax.set_xlim(left=-plot_vmax, right=plot_vmax)
    if "_" in colname:
        ax.annotate(
            colname.split("_")[1],
            color="white",
            xy=(0.9, 0.9),
            xycoords="axes fraction",
            horizontalalignment="right",
            verticalalignment="top",
            # fontsize=10,
        )

    return im, scaledmap


def plot_3d_initial_abundances(modelpath, args=None) -> None:
    font = {
        # 'weight': 'bold',
        "size": 18
    }
    mpl.rc("font", **font)

    dfmodel, modelmeta = at.get_modeldata(
        modelpath,
        skipnuclidemassfraccolumns=True,
        get_elemabundances=True,
        dtype_backend="pyarrow",
        derived_cols=["pos_max"],
        use_polars=False,
    )

    targetmodeltime_days = None
    if targetmodeltime_days is not None:
        print(
            f"Scaling modeldata to {targetmodeltime_days} days. \nWARNING: abundances not scaled for radioactive decays"
        )

        dfmodel, modelmeta = at.inputmodel.scale_model_to_time(
            targetmodeltime_days=targetmodeltime_days, modelmeta=modelmeta, dfmodel=dfmodel
        )

    sliceaxis: t.Literal["x", "y", "z"] = "z"

    axes: list[t.Literal["x", "y", "z"]] = ["x", "y", "z"]
    plotaxis1: t.Literal["x", "y", "z"] = next(ax for ax in axes if ax != sliceaxis)
    plotaxis2: t.Literal["x", "y", "z"] = next(ax for ax in axes if ax not in [sliceaxis, plotaxis1])

    df2dslice = get_2D_slice_through_3d_model(
        dfmodel=dfmodel, modelmeta=modelmeta, sliceaxis=sliceaxis, plotaxis1=plotaxis1, plotaxis2=plotaxis2
    )

    subplots = bool(len(args.plotvars) > 1)
    if not subplots:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7), tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0})
    else:
        rows = 1
        cols = len(args.plotvars)

        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            sharex=True,
            sharey=True,
            figsize=(at.get_config()["figwidth"] * cols, at.get_config()["figwidth"] * 1.4),
            # tight_layout={"pad": 5.0, "w_pad": 0.0, "h_pad": 0.0},
        )
        for ax in axes:
            ax.set(aspect="equal")

    for index, plotvar in enumerate(args.plotvars):
        colname = plotvar if plotvar in df2dslice.columns else f"X_{plotvar}"

        if subplots:
            ax = axes[index]
        im, scaledmap = plot_slice_modelcol(
            ax, df2dslice, modelmeta, colname, plotaxis1, plotaxis2, modelmeta["t_model_init_days"], args
        )

    xlabel = rf"v$_{plotaxis1}$ [$c$]"
    ylabel = rf"v$_{plotaxis2}$ [$c$]"

    if not subplots:
        cbar = fig.colorbar(im)
        plt.xlabel(xlabel, fontsize="x-large")  # , fontweight='bold')
        plt.ylabel(ylabel, fontsize="x-large")  # , fontweight='bold')
    else:
        cbar = fig.colorbar(scaledmap, ax=axes, shrink=cols * 0.08, location="top", pad=0.8, anchor=(0.5, 3.0))
        fig.text(0.5, 0.15, xlabel, ha="center", va="center")
        fig.text(0.05, 0.5, ylabel, ha="center", va="center", rotation="vertical")

    if "cellYe" not in args.plotvars and "tracercount" not in args.plotvars:
        if args.logcolorscale:
            cbar.ax.set_title(r"log10($\rho$) [g/cm3]", size="small")
        else:
            cbar.ax.set_title(r"$\rho$ [g/cm3]", size="small")
    # cbar.ax.tick_params(labelsize='x-large')

    # plt.tight_layout()
    # ax.labelsize: 'large'
    # plt.title(f'At {sliceaxis} = {sliceposition}')

    outfilename = args.outputfile or f"plotcomposition_{','.join(args.plotvars)}.pdf"
    plt.savefig(Path(modelpath) / outfilename, format="pdf")

    print(f"Saved {outfilename}")


def get_model_abundances_Msun_1D(modelpath: Path) -> pd.DataFrame:
    filename = modelpath / "model.txt"
    modeldata, t_model_init_days, _ = at.inputmodel.get_modeldata_tuple(filename)
    abundancedata = at.inputmodel.get_initelemabundances(modelpath)

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60

    modeldata["volume_shell"] = (
        4
        / 3
        * math.pi
        * (
            (modeldata["velocity_outer"] * 1e5 * t_model_init_seconds) ** 3
            - (modeldata["velocity_inner"] * 1e5 * t_model_init_seconds) ** 3
        )
    )

    modeldata["mass_shell"] = (10 ** modeldata["logrho"]) * modeldata["volume_shell"]

    merge_dfs = modeldata.merge(abundancedata, how="inner", on="inputcellid")

    print("Total mass (Msun):")
    for key in merge_dfs:
        if "X_" in key:
            merge_dfs[f"mass_{key}"] = merge_dfs[key] * merge_dfs["mass_shell"] * u.g.to("solMass")
            # get mass of element in each cell
            print(key, merge_dfs[f"mass_{key}"].sum())  # print total mass of element in solmass

    return merge_dfs


def plot_most_abundant(modelpath, args):
    model, _ = at.inputmodel.get_modeldata(modelpath[0])
    abundances = at.inputmodel.get_initelemabundances(modelpath[0])

    merge_dfs = model.merge(abundances, how="inner", on="inputcellid")
    elements = [x for x in merge_dfs if "X_" in x]

    merge_dfs["max"] = merge_dfs[elements].idxmax(axis=1)

    merge_dfs["max"] = merge_dfs["max"].apply(lambda x: at.get_atomic_number(x[2:]))
    return merge_dfs[merge_dfs["max"] != 1]


def make_3d_plot(modelpath, args):
    import pyvista as pv

    pv.set_plot_theme("document")  # set white background

    model, t_model, vmax = at.inputmodel.get_modeldata_tuple(modelpath, get_elemabundances=False)
    abundances = at.inputmodel.get_initelemabundances(modelpath)

    abundances["inputcellid"] = abundances["inputcellid"].apply(float)

    merge_dfs = model.merge(abundances, how="inner", on="inputcellid")
    model = merge_dfs

    # choose what surface will be coloured by
    if args.rho:
        coloursurfaceby = "rho"
    elif args.opacity:
        model["opacity"] = at.inputmodel.opacityinputfile.get_opacity_from_file(modelpath)
        coloursurfaceby = "opacity"
    else:
        print(f"Colours set by X_{args.plotvars}")
        coloursurfaceby = f"X_{args.plotvars}"

    # generate grid from data
    grid = round(len(model["rho"]) ** (1.0 / 3.0))
    surfacecolorscale = np.zeros((grid, grid, grid))  # needs 3D array
    xgrid = np.zeros(grid)

    surfacearr = np.array(model[coloursurfaceby])

    i = 0
    for z in range(grid):
        for y in range(grid):
            for x in range(grid):
                surfacecolorscale[x, y, z] = surfacearr[i]
                xgrid[x] = -vmax + 2 * x * vmax / grid
                i += 1

    x, y, z = np.meshgrid(xgrid, xgrid, xgrid)

    mesh = pv.StructuredGrid(x, y, z)
    print(mesh)  # tells you the properties of the mesh

    mesh[coloursurfaceby] = surfacecolorscale.ravel(order="F")  # add data to the mesh
    # mesh.plot()
    minval = np.min(mesh[coloursurfaceby][np.nonzero(mesh[coloursurfaceby])])  # minimum non zero value
    print(f"{coloursurfaceby} minumin {minval}, maximum {max(mesh[coloursurfaceby])}")

    if not args.surfaces3d:
        surfacepositions = np.linspace(min(mesh[coloursurfaceby]), max(mesh[coloursurfaceby]), num=10)
        print(f"Using default surfaces {surfacepositions} \n define these with -surfaces3d for better results")
    else:
        surfacepositions = args.surfaces3d
    # surfacepositions = [1, 50, 100, 300, 500, 800, 1000, 1100, 1200, 1300, 1400, 1450, 1500] # choose these

    surf = mesh.contour(surfacepositions, scalars=coloursurfaceby)  # create isosurfaces

    surf.plot(opacity="linear", screenshot=modelpath / "3Dplot.png")  # plot surfaces and save screenshot


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath",
        type=Path,
        default=Path(),
        help="Path to ARTIS folder",
    )

    parser.add_argument("-o", action="store", dest="outputfile", type=Path, default=None, help="Filename for PDF file")

    parser.add_argument(
        "plotvars",
        type=str,
        default=["rho"],
        nargs="+",
        help=(
            "Element symbols (Fe, Ni, Sr) for mass fraction or other model columns (rho, tracercount) to plot. Default"
            " is rho"
        ),
    )

    parser.add_argument("--logcolorscale", action="store_true", help="Use log scale for colour map")

    parser.add_argument("--hideemptycells", action="store_true", help="Don't plot empty cells")

    parser.add_argument("--opacity", action="store_true", help="Plot opacity from opacity.txt (if available for model)")

    parser.add_argument("--plot3d", action="store_true", help="Make 3D plot")

    parser.add_argument("-surfaces3d", type=float, nargs="+", help="define positions of surfaces for 3D plots")


def main(args=None, argsraw=None, **kwargs) -> None:
    """Plot ARTIS input model composition."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = ["."]

    if args.plot3d:
        make_3d_plot(Path(args.modelpath), args)
        return

    _, modelmeta = at.get_modeldata(modelpath=args.modelpath, getheadersonly=True, printwarningsonly=True)

    if modelmeta["dimensions"] == 2:
        plot_2d_initial_abundances(args.modelpath, args)

    elif modelmeta["dimensions"] == 3:
        plot_3d_initial_abundances(args.modelpath, args)


if __name__ == "__main__":
    main()
