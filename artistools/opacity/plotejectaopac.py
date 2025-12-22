"""Script added by Gerrit for plotting the Planck mean opacity structure of any ARTIS run."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from cellopacitycalculator import calc_cell_opacs
from matplotlib import gridspec

import artistools as at

CLIGHT = 2.99792458e10
days_to_s = 86400


def make_2D_slice_opac_plot(
    modelpath: Path,
    odf: pl.DataFrame,
    t_slice_days: float,
    model_dim: int,
    n_x: int,
    n_y: int,
    plotaxis1: str,
    plotaxis2: str,
    outputpath: Path | None = None,
) -> None:
    # mostly taken from at.inputmodel.plotinitialcomposition
    nrows = 1
    ncols = 1
    xfactor = 1 if model_dim == 3 else 0.5
    figwidth = at.get_config()["figwidth"]
    fig = plt.figure(
        figsize=(figwidth * xfactor * ncols, figwidth * nrows), tight_layout={"pad": 1.0, "w_pad": 0.0, "h_pad": 0.0}
    )
    gs = gridspec.GridSpec(nrows + 1, ncols, height_ratios=[0.05, 1], width_ratios=[1] * ncols)

    axcbar = fig.add_subplot(gs[0, :])
    axes = [fig.add_subplot(gs[1, y]) for y in range(ncols)]
    # actual plot
    colorscale = (odf["kappa_Pl"]).to_numpy()

    colorscale = np.ma.masked_where(colorscale == 0.0, colorscale)  # type: ignore[no-untyped-call]

    normalise_between_0_and_1 = False
    if normalise_between_0_and_1:
        norm = mplcolors.Normalize(vmin=0, vmax=1)
        scaledmap = mplcm.ScalarMappable(cmap="viridis", norm=norm)
        scaledmap.set_array([])
        colorscale = scaledmap.to_rgba(colorscale)  # colorscale fixed between 0 and 1
    else:
        scaledmap = None

    cmps_to_beta = 1.0 / CLIGHT
    unitfactor = cmps_to_beta
    t_model_s = t_slice_days * days_to_s

    # turn 1D flattened array back into 2D array
    valuegrid = colorscale.reshape((n_x, n_y))

    vmin_ax1 = odf.select(pl.col(f"pos_{plotaxis1}_min").min()).item() / t_model_s * unitfactor
    vmax_ax1 = odf.select(pl.col(f"pos_{plotaxis1}_max").max()).item() / t_model_s * unitfactor
    vmin_ax2 = odf.select(pl.col(f"pos_{plotaxis2}_min").min()).item() / t_model_s * unitfactor
    vmax_ax2 = odf.select(pl.col(f"pos_{plotaxis2}_max").max()).item() / t_model_s * unitfactor
    im = axes[0].imshow(
        valuegrid,
        cmap="viridis",
        interpolation="nearest",
        extent=(vmin_ax1, vmax_ax1, vmin_ax2, vmax_ax2),
        origin="lower",
    )

    xlabel = r"v$_{" + str(plotaxis1) + r"}$ [$c$]"
    ylabel = r"v$_{" + str(plotaxis2) + r"}$ [$c$]"

    cbar = fig.colorbar(im, cax=axcbar, location="top", use_gridspec=True)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    cbar.set_label(r"$\kappa_{Pl}$ [cm$^2$ $g^{-1}$]")

    # save either in working directory or in the seperate output directory if specified
    defaultfilename = Path(modelpath) / "plotplanckopac.pdf"
    outfilename = Path(modelpath) / defaultfilename if outputpath and Path(outputpath).is_dir() else defaultfilename

    plt.savefig(outfilename, format="pdf")

    print(f"Saved {outfilename}.pdf")


def addargs(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("-modelpath", "-o", type=Path, default=Path(), help="Path of ARTIS model")

    parser.add_argument("-t", type=float, help="Time in days for the 2D opacity plot. Either in 2D or 3D mode.")

    parser.add_argument(
        "-timemin", type=float, help="Minimum time in days for the radial opacity plot. Either in 2D or 3D mode."
    )

    parser.add_argument(
        "-timemax", type=float, help="Maximum time in days for the radial opacity plot. Either in 2D or 3D mode."
    )

    parser.add_argument("-slice", default="xy", help="Plane of slice in case of a 3D model. Example: xy <-> z=0.")

    parser.add_argument("-outputpath", "-o", type=Path, default=Path(), help="Path to output PDF")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    # get inputmodel, tuple of LazyFrame and dictionary
    im = at.inputmodel.get_modeldata(modelpath=Path(args.modelpath), derived_cols=["mass_g", "velocity"])
    model_dim = im[1]["dimensions"]
    t_snap_days = im[1]["t_model_init_days"]

    # elf: estimators Lazy Frame
    elf = at.estimators.scan_estimators(modelpath=Path(args.modelpath))

    opac_data: pl.DataFrame

    if model_dim == 0:
        # 0D / one-zone: plot opacity as function of time
        opac_data = calc_cell_opacs(args.modelpath, elf)
    elif model_dim == 1:
        # 1D: ejecta opacity as function of radius for specified range of time
        pass
    elif model_dim in {2, 3}:
        # 2D: 2D-plot of ejecta opacity at one specified time
        assert args.t is not None, "No time specified. Abort."
        # get closest timestep
        t_d = float(args.t)
        t_s = t_d * days_to_s
        exp_factor = (t_snap_days / t_d) ** 3
        ts_lf = at.get_timesteps(Path(args.modelpath))

        plot_ts = (ts_lf.filter(pl.col("tstart_days") <= t_d).select(pl.col("timestep").max())).collect().item()

        if model_dim == 2:
            plotaxis1 = "rcyl"
            plotaxis2 = "z"
            n_ax1 = im[1]["ncoordgridrcyl"]
            n_ax2 = im[1]["ncoordgridz"]
            # reduce input estimator data to required cells
            elf = elf.filter(pl.col("timestep") == plot_ts).sort("modelgridindex")
            # add total mass density to cell data
            elf = elf.join(im[0].select(["modelgridindex", "rho"]), on="modelgridindex", how="left").with_columns(
                (pl.col("rho") * exp_factor).alias("rho")
            )
            opac_data = calc_cell_opacs(args.modelpath, elf, t_s)
            opac_data = opac_data.join(
                elf.select(["modelgridindex", "pos_rcyl_min", "pos_rcyl_max", "pos_z_min", "pos_z_max"]),
                on="modelgridindex",
                how="left",
            )
        elif model_dim == 3:
            # 3D: 2D-slice of ejecta opacity at one specified time
            plotaxis1: str
            plotaxis2: str
            if args.slice == "xy":
                # plane z = 0
                plotaxis1 = "x"
                plotaxis2 = "y"
                # SELECT cells, TODO
                opac_data = calc_cell_opacs(args.modelpath, elf.filter(pl.col("timestep") == plot_ts), t_s)
            elif args.slice == "xz":
                # plane y = 0
                plotaxis1 = "x"
                plotaxis2 = "z"
            opac_data = opac_data.join(
                elf.select(["modelgridindex", "pos_rcyl_min", "pos_rcyl_max", "pos_z_min", "pos_z_max"]),
                on="modelgridindex",
                how="left",
            )

        make_2D_slice_opac_plot(
            args.modelpath, opac_data, t_d, model_dim, n_ax1, n_ax2, plotaxis1, plotaxis2, args.outputpath
        )


if __name__ == "__main__":
    main()
