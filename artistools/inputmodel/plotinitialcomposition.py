#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


import argparse
import math
import typing as t
from pathlib import Path

import argcomplete
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import gridspec

import artistools as at

AxisType: t.TypeAlias = t.Literal["x", "y", "z", "r", "rcyl", "z"]


def get_2D_slice_through_3d_model(
    dfmodel: pd.DataFrame,
    sliceaxis: AxisType,
    modelmeta: dict[str, t.Any] | None = None,
    plotaxis1: AxisType | None = None,
    plotaxis2: AxisType | None = None,
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


def plot_slice_modelcolumn(ax, dfmodelslice, modelmeta, colname, plotaxis1, plotaxis2, t_model_d, args):
    print(f"plotting {colname}")
    colorscale = (
        dfmodelslice[colname] * dfmodelslice["rho"] if colname.startswith("X_") else dfmodelslice[colname]
    ).to_numpy()

    if args.hideemptycells:
        # Don't plot empty cells:
        colorscale = np.ma.masked_where(colorscale == 0.0, colorscale)

    if args.logcolorscale:
        # logscale for colormap
        if args.floorval:
            colorscale = [args.floorval if x < args.floorval or not math.isfinite(x) else x for x in colorscale]
        with np.errstate(divide="ignore"):
            colorscale = np.log10(colorscale)
        # np.nan_to_num(colorscale, posinf=-99, neginf=-99)

    normalise_between_0_and_1 = False
    if normalise_between_0_and_1:
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        scaledmap = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
        scaledmap.set_array([])
        colorscale = scaledmap.to_rgba(colorscale)  # colorscale fixed between 0 and 1
    else:
        scaledmap = None

    cmps_to_beta = 1.0 / (2.99792458e10)
    unitfactor = cmps_to_beta
    t_model_s = t_model_d * 86400.0

    # turn 1D flattened array back into 2D array
    valuegrid = colorscale.reshape((modelmeta[f"ncoordgrid{plotaxis2}"], modelmeta[f"ncoordgrid{plotaxis1}"]))

    vmin_ax1 = dfmodelslice[f"pos_{plotaxis1}_min"].min() / t_model_s * unitfactor
    vmax_ax1 = dfmodelslice[f"pos_{plotaxis1}_max"].max() / t_model_s * unitfactor
    vmin_ax2 = dfmodelslice[f"pos_{plotaxis2}_min"].min() / t_model_s * unitfactor
    vmax_ax2 = dfmodelslice[f"pos_{plotaxis2}_max"].max() / t_model_s * unitfactor
    im = ax.imshow(
        valuegrid,
        cmap="viridis",
        interpolation="nearest",
        extent=(
            vmin_ax1,
            vmax_ax1,
            vmin_ax2,
            vmax_ax2,
        ),
        origin="lower",
        # vmin=0.0,
        # vmax=1.0,
    )

    # plot_vmax = 0.2
    # ax.set_ylim(bottom=-plot_vmax, top=plot_vmax)
    # ax.set_xlim(left=-plot_vmax, right=plot_vmax)

    # ax.set_xlim(left=vmin_ax1, right=vmax_ax1)
    # ax.set_ylim(bottom=vmin_ax2, top=vmax_ax2)

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


def plot_2d_initial_abundances(modelpath, args=None) -> None:
    # if the species ends in a number then we need to also get the nuclear mass fractions (not just element abundances)
    get_elemabundances = any(plotvar[-1] not in "0123456789" for plotvar in args.plotvars)
    dfmodel, modelmeta = at.get_modeldata(
        modelpath,
        get_elemabundances=get_elemabundances,
        derived_cols=["pos_min", "pos_max"],
    )
    assert modelmeta["dimensions"] > 1

    targetmodeltime_days = None
    if targetmodeltime_days is not None:
        print(
            f"Scaling modeldata to {targetmodeltime_days} days. \nWARNING: abundances not scaled for radioactive decays"
        )

        dfmodel, modelmeta = at.inputmodel.scale_model_to_time(
            targetmodeltime_days=targetmodeltime_days, modelmeta=modelmeta, dfmodel=dfmodel
        )

    if modelmeta["dimensions"] == 3:
        sliceaxis: AxisType = args.sliceaxis

        axeschars: list[AxisType] = ["x", "y", "z"]
        plotaxis1 = next(ax for ax in axeschars if ax != sliceaxis)
        plotaxis2 = next(ax for ax in axeschars if ax not in {sliceaxis, plotaxis1})
        print(f"Plotting slice through {sliceaxis}=0, plotting {plotaxis1} vs {plotaxis2}")

        df2dslice = get_2D_slice_through_3d_model(
            dfmodel=dfmodel, modelmeta=modelmeta, sliceaxis=sliceaxis, plotaxis1=plotaxis1, plotaxis2=plotaxis2
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
    figwidth = at.get_config()["figwidth"]
    fig = plt.figure(
        figsize=(figwidth * xfactor * ncols, figwidth * nrows),
        tight_layout={"pad": 1.0, "w_pad": 0.0, "h_pad": 0.0},
    )
    gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios=[0.05, 1], width_ratios=[1] * ncols)

    axcbar = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, y]) for y in range(ncols)]
    # fig, axes = plt.subplots(
    #     nrows=nrows,
    #     ncols=ncols,
    #     sharex=True,
    #     sharey=True,
    #     squeeze=False,
    #     figsize=(figwidth * xfactor * ncols, figwidth * 1.4 * nrows),
    #     tight_layout=None,
    # )

    for plotvar, ax in zip(args.plotvars, axes):
        colname = plotvar if plotvar in df2dslice.columns else f"X_{plotvar}"

        im, scaledmap = plot_slice_modelcolumn(
            ax, df2dslice, modelmeta, colname, plotaxis1, plotaxis2, modelmeta["t_model_init_days"], args
        )

    xlabel = r"v$_{" + f"{plotaxis1}" + r"}$ [$c$]"
    ylabel = r"v$_{" + f"{plotaxis2}" + r"}$ [$c$]"

    cbar = fig.colorbar(im, cax=axcbar, location="top", use_gridspec=True)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    if "cellYe" not in args.plotvars and "tracercount" not in args.plotvars:
        cbar.set_label(r"log10($\rho$ [g/cm3])" if args.logcolorscale else r"$\rho$ [g/cm3]")
    else:
        cbar.set_label("Ye" if "cellYe" in args.plotvars else "tracercount")

    defaultfilename = Path(modelpath) / f"plotcomposition_{','.join(args.plotvars)}.pdf"
    if args.outputfile and Path(args.outputfile).is_dir():
        outfilename = Path(modelpath) / defaultfilename
    elif args.outputfile:
        outfilename = args.outputfile
    else:
        outfilename = defaultfilename

    plt.savefig(outfilename, format="pdf")

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
            (modeldata["vel_r_max_kmps"] * 1e5 * t_model_init_seconds) ** 3
            - (modeldata["vel_r_min_kmps"] * 1e5 * t_model_init_seconds) ** 3
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

    get_elemabundances = False
    # choose what surface will be coloured by
    if "rho" in args.plotvars:
        coloursurfaceby = "rho"
    elif "cellYe" in args.plotvars:
        coloursurfaceby = "cellYe"
    else:
        print(f"Colours set by X_{args.plotvars}")
        coloursurfaceby = f"X_{args.plotvars}"
        get_elemabundances = True

    model, t_model, vmax = at.inputmodel.get_modeldata_tuple(modelpath, get_elemabundances=get_elemabundances)

    if "cellYe" in args.plotvars and "cellYe" not in model:
        file_contents = np.loadtxt(Path(modelpath) / "Ye.txt", unpack=True, skiprows=1)
        Ye = file_contents[1]
        model["cellYe"] = Ye

    mincellparticles = 0
    if mincellparticles > 0:
        if "tracercount" not in model:
            griddata = pd.read_csv(modelpath / "grid.dat", sep=r"\s+", comment="#", skiprows=3)
            model["tracercount"] = griddata["tracercount"]
        print(model["tracercount"], max(model["tracercount"]))
        model[coloursurfaceby][model["tracercount"] < mincellparticles] = 0

    # generate grid from data
    grid = round(len(model["rho"]) ** (1.0 / 3.0))
    surfacecolorscale = np.zeros((grid, grid, grid))  # needs 3D array
    xgrid = np.zeros(grid)

    surfacearr = np.array(model[coloursurfaceby])
    vmax = vmax / 2.99792458e10
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
        surfacepositions = args.surfaces3d  # expects an array of surface positions

    surf = mesh.contour(surfacepositions, scalars=coloursurfaceby)  # create isosurfaces

    # surf.plot(opacity="linear", screenshot=modelpath / "3Dplot.png")  # plot surfaces and save screenshot
    sargs = {
        "height": 0.25,
        "vertical": True,
        "position_x": 0.05,
        "position_y": 0.1,
        "title_font_size": 22,
        "label_font_size": 22,
    }

    plotter = pv.Plotter()
    # plotter.add_mesh(mesh.outline(), color="k")
    plotcoloropacity = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # some choices: 'linear' 'sigmoid'
    # plotter.set_scale(0.95, 0.95, 0.95) # adjusts fig resolution
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
    # plotter.add_mesh(surf, opacity=plotcoloropacity, use_transparency=True, cmap='coolwarm_r') #magma

    # plotter.remove_scalar_bar() # removes colorbar

    plotter.camera_position = "xz"
    plotter.camera.azimuth = 45.0
    plotter.camera.elevation = 10.0
    # plotter.camera.azimuth = 15
    plotter.show(screenshot=modelpath / "3Dplot.png", auto_close=False)

    # Make gif:
    # # viewup = [0.5, 0.5, 1]
    # path = plotter.generate_orbital_path(n_points=150, shift=mesh.length / 5)
    # plotter.open_gif("orbit.gif")
    # plotter.orbit_on_path(path, write_frames=True)
    # plotter.close()


def plot_phi_hist(modelpath):
    dfmodel, modelmeta = at.get_modeldata(modelpath, derived_cols=["pos_x_mid", "pos_y_mid", "pos_z_mid", "vel_r_mid"])
    # print(dfmodel.keys())
    # quit()
    at.inputmodel.inputmodel_misc.get_cell_angle(dfmodel, modelpath)
    CLIGHT = 2.99792458e10
    # MSUN = 1.989e33

    # dfmodel.query("cos_bin in [40, 50]", inplace=True)
    # mass = dfmodel["cellmass_grams"] / MSUN
    # weights = mass
    # weights = dfmodel['cellYe']
    # weights = dfmodel['q']
    weightby = "rho"
    weights = dfmodel[weightby]
    labeldict = {"cellYe": "Ye"}
    if weightby in labeldict:
        weightby = labeldict[weightby]

    nphibins = 25
    nvbins = 25
    vmin = 0.0  # c
    vmax = 0.7  # c
    heatmap, xedges, yedges = np.histogram2d(
        dfmodel["vel_r_mid"] / CLIGHT,
        dfmodel["phi"],
        bins=[np.linspace(vmin, vmax, num=nvbins), np.linspace(0, 2 * np.pi, num=nphibins)],
        weights=weights,
    )
    # heatmap = heatmap / (2 * np.pi) / (vmax - vmin) / nphibins / nvbins
    print("WARNING: histogram not normalised")
    plt.clf()

    heatmap = np.ma.masked_where(heatmap == 0.0, heatmap)
    # heatmap = np.log10(heatmap)

    fig = plt.figure(figsize=[5, 4])
    ax = fig.add_axes([0.15, 0.15, 0.75, 0.75], polar=True)

    radii = xedges
    z = heatmap
    phis = yedges

    cmap = "coolwarm_r" if weightby == "Ye" else "coolwarm"
    im = ax.pcolormesh(phis, radii, z, cmap=cmap)
    cbar = fig.colorbar(im)

    cbar.set_label(f"{weightby}", rotation=90)
    plt.xlabel(r"azimuthal angle")
    plt.ylabel("Radial velocity [c]")
    ax.yaxis.set_label_coords(-0.15, 0.5)

    outfilename = f"model{weightby}phi.pdf"
    plt.savefig(Path(modelpath) / outfilename, format="pdf")
    print(f"Saved {outfilename}")
    plt.close()


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

    parser.add_argument("-floorval", default=False, type=float, help="Set a floor value for colorscale. Expects float.")

    parser.add_argument(
        "-axis",
        default="+z",
        choices=["x", "y", "z", "+x", "-x", "+y", "-y", "+z", "-z"],
        help="Choose an axis for use with args.readonlymgi. Hint: for negative use e.g. -axis=-z",
    )


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs) -> None:
    """Plot ARTIS input model composition."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    if args.axis[0] in {"+", "-"}:
        args.positive_axis = args.axis[0] == "+"
        args.axis = args.axis[1]
    args.sliceaxis = args.axis

    args.plotvars = ["cellYe" if var == "Ye" else var for var in args.plotvars]

    if not args.modelpath:
        args.modelpath = ["."]

    if args.plot3d:
        make_3d_plot(Path(args.modelpath), args)
        return

    plot_2d_initial_abundances(args.modelpath, args)


if __name__ == "__main__":
    main()
