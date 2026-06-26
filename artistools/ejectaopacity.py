# PYTHON_ARGCOMPLETE_OK
"""Script for computing binned expansion opacities and Planck-mean opacities in postprocessing."""

import argparse
import math
import time
import typing as t
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import h_erg_s
from artistools.constants import K_B_erg_per_K
from artistools.constants import K_B_ev_per_K

HCLIGHTOVERFOURPI = h_erg_s * C_cm_per_s / 4 / math.pi
CLIGHTKMPERSECOND = 2.99792458e5


def get_binned_opacities_ion(
    dfcells: pl.LazyFrame,
    dflevels: pl.LazyFrame,
    dftransitions: pl.LazyFrame,
    ionstr: str,
    lambda_bin_edges: list[float],
    expopac_deltalambda: float,
    time_days: float,
) -> pl.LazyFrame:
    time_s = time_days * 86400.0
    dfcelllevelpops = dflevels.join(dfcells, how="cross").with_columns(
        nnlevel=pl.col("g")
        * (-pl.col("energy_ev") / K_B_ev_per_K / pl.col("Te")).exp()
        / ((pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / pl.col("Te")).exp()).sum().over("modelgridindex"))
        * pl.col(f"nnion_{ionstr}")
    )

    return (
        dftransitions
        .filter(pl.col("lambda_angstroms").is_between(lambda_bin_edges[0], lambda_bin_edges[-1]))
        .with_columns(nu_trans=1e8 * C_cm_per_s / (pl.col("lambda_angstroms")))
        .with_columns(B_ul=C_cm_per_s**2 / 2 / h_erg_s / pl.col("nu_trans").pow(3) * pl.col("A"))
        .with_columns(B_lu=pl.col("upper_g") / pl.col("lower_g") * pl.col("B_ul"))
        .with_columns(
            (
                pl.col("lambda_angstroms").cut(
                    breaks=lambda_bin_edges, labels=[str(x) for x in range(-1, len(lambda_bin_edges))]
                )
            )
            .cast(pl.String)
            .cast(pl.Int32)
            .alias("lambda_angstroms_binindex")
        )
        .join(dfcells.select("modelgridindex", "rho"), how="cross")
        .join(
            dfcelllevelpops.select("modelgridindex", lower=pl.col("levelindex"), nnlevel_lower=pl.col("nnlevel")),
            on=("modelgridindex", "lower"),
            how="left",
        )
        .join(
            dfcelllevelpops.select("modelgridindex", upper=pl.col("levelindex"), nnlevel_upper=pl.col("nnlevel")),
            on=("modelgridindex", "upper"),
            how="left",
        )
        .with_columns(
            tau_sobolev=(pl.col("nnlevel_lower") * pl.col("B_lu") - pl.col("nnlevel_upper") * pl.col("B_ul"))
            * HCLIGHTOVERFOURPI
            * time_s
        )
        .group_by("modelgridindex", "lambda_angstroms_binindex")
        .agg(
            (
                (
                    (1 - (-pl.col("tau_sobolev")).exp())
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (C_cm_per_s * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"exopac_contribution_{ionstr}"),
            (
                (
                    pl.min_horizontal(pl.col("tau_sobolev"), 1.0)
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (C_cm_per_s * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"linebinned_maxone_contribution_{ionstr}"),
            (
                (
                    pl.col("tau_sobolev")
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (C_cm_per_s * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"linebinned_contribution_{ionstr}"),
        )
    )


def get_expansion_opacities(
    adata: pl.DataFrame,
    time_days: float,
    dfestimators: pl.DataFrame,
    lambdamin: float,
    lambdamax: float,
    deltalambda: float,
) -> pl.LazyFrame:
    numbins = int((lambdamax - lambdamin) / deltalambda)

    print("Summing opacities...")

    dfbinnedopacities = (
        pl
        .LazyFrame({"lambda_angstroms_binindex": range(numbins)})
        .set_sorted("lambda_angstroms_binindex")
        .with_columns(lambda_angstroms_binlower=lambdamin + pl.col("lambda_angstroms_binindex") * deltalambda)
        .with_columns(lambda_angstroms_bin_mid=pl.col("lambda_angstroms_binlower") + (deltalambda / 2))
        .join(dfestimators.select("modelgridindex", "Te", "mass_g").lazy(), how="cross")
    )

    lambda_bin_edges = [lambdamin + i * deltalambda for i in range(numbins + 1)]

    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        ionstr = at.get_ionstring(Z, ion_stage, sep="_")

        if f"nnion_{ionstr}" not in dfestimators.collect_schema().names():
            continue

        dfbinnedopacities = dfbinnedopacities.join(
            get_binned_opacities_ion(
                dfestimators.lazy(), dflevels.lazy(), dftransitions, ionstr, lambda_bin_edges, deltalambda, time_days
            ),
            on=("modelgridindex", "lambda_angstroms_binindex"),
            how="left",
        )

    return dfbinnedopacities.select(
        "modelgridindex",
        "lambda_angstroms_binindex",
        "lambda_angstroms_bin_mid",
        "Te",
        "mass_g",
        *[
            pl.sum_horizontal(cs.starts_with(prefix)).alias(prefix.removesuffix("_contribution_"))
            for prefix in ("exopac_contribution_", "linebinned_contribution_", "linebinned_maxone_contribution_")
        ],
    ).sort("modelgridindex", "lambda_angstroms_binindex")


def plot_planck_mean_opacity(
    modelpath: Path,
    outputpath: Path,
    opacity_frame: pl.DataFrame,
    timestep: int,
    modelmeta: dict[str, t.Any],
    slice_of_3D_model: str | None = None,
) -> None:
    """Plot the Planck mean opacity for any given model dimension at a specified time."""
    model_dim = modelmeta["dimensions"]
    assert model_dim > 0, "Plotting function should not be called for 1-zone models."
    assert len(opacity_frame) == modelmeta["npts_model"], (
        f"len(opacity_frame): {len(opacity_frame)} npts_model: {modelmeta['npts_model']} but must be equal"
    )

    if model_dim == 1:
        # plot opacity as function of radius
        at.plottools.set_mpl_style()
        fig, axis = plt.subplots(nrows=1, ncols=1)
        min_val = cast("float", opacity_frame["vel_r_max_kmps"].min())

        opacity_frame = opacity_frame.with_columns((pl.col("vel_r_max_kmps") - 0.5 * min_val).alias("vel_r_max_kmps"))
        axis.plot(opacity_frame["vel_r_max_kmps"] / CLIGHTKMPERSECOND, opacity_frame["planckmean_opacity"])
        axis.set_xlabel("velocity (fraction of c)")
        axis.set_ylabel(r"Planck mean opacity (cm$^2$ g$^{-1}$)")
    else:
        numb_x_axis_pts = 0
        numb_y_axis_pts = 0
        x_axis_coord = "r"
        y_axis_coord = "z"
        if model_dim == 3:
            # reduce opacity frame to slice
            assert slice_of_3D_model is not None, "No model slice provided"
            if "x" in slice_of_3D_model:
                x_axis_coord = "x"
                y_axis_coord = slice_of_3D_model.replace("x", "")
            else:
                x_axis_coord = "y"
                y_axis_coord = "z"
            numb_x_axis_pts = modelmeta[f"ncoordgrid{x_axis_coord}"]
            numb_y_axis_pts = modelmeta[f"ncoordgrid{y_axis_coord}"]
        else:
            numb_x_axis_pts = modelmeta["ncoordgridrcyl"]
            numb_y_axis_pts = modelmeta["ncoordgridz"]

        opacity_array = opacity_frame["planckmean_opacity"].to_numpy()
        colorscale = np.ma.masked_where(opacity_array == 0.0, opacity_array)
        valuegrid = colorscale.reshape((numb_y_axis_pts, numb_x_axis_pts))

        cellsize = 0.5
        figwidth = numb_x_axis_pts * cellsize
        figheight = numb_y_axis_pts * cellsize
        fig, axis = plt.subplots(figsize=(figwidth, figheight))

        vmin_x_axis = opacity_frame.select(pl.col(f"vel_{x_axis_coord}_min_on_c").min()).item()
        vmax_x_axis = opacity_frame.select(pl.col(f"vel_{x_axis_coord}_max_on_c").max()).item()
        vmin_y_axis = opacity_frame.select(pl.col(f"vel_{y_axis_coord}_min_on_c").min()).item()
        vmax_y_axis = opacity_frame.select(pl.col(f"vel_{y_axis_coord}_max_on_c").max()).item()

        im = axis.imshow(
            valuegrid,
            cmap="viridis",
            interpolation="nearest",
            extent=(vmin_x_axis, vmax_x_axis, vmin_y_axis, vmax_y_axis),
            origin="lower",
            aspect="auto",
        )

        axis.set_xlabel(r"v$_{" + str(x_axis_coord) + r"}$ [$c$]", fontsize=16)
        axis.set_ylabel(r"v$_{" + str(y_axis_coord) + r"}$ [$c$]", fontsize=16)
        axis.tick_params(axis="both", which="major", labelsize=16)

        cbar = fig.colorbar(im, ax=axis, orientation="horizontal", pad=0.05)
        cbar.set_label(r"$\kappa_{Pl}$ [cm$^2$ $g^{-1}$]", fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        fig.tight_layout(pad=0.5)
    defaultfilename = modelpath / Path(f"plotplanckopac_ts{timestep}")
    outputfilepath = (
        outputpath / f"plotplanckopac_ts{timestep}.pdf" if outputpath and outputpath.is_dir() else defaultfilename
    )
    fig.savefig(outputfilepath, format="pdf", dpi=300)
    print(f"Saved {outputfilepath}")
    plt.close()


def addargs(parser: argparse.ArgumentParser) -> None:
    # mutex with time in days:
    timegroup = parser.add_argument_group("time selection (specify either timestep or time in days)")
    timegroup.add_argument("-timestep", "-ts", type=int, help="Timestep number to select")
    timegroup.add_argument("-timedays", "-time", "-t", type=float, help="Time in days to select.")

    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path of ARTIS model")
    parser.add_argument(
        "--show_binned_opacities",
        action="store_true",
        help="Show the binned opacities for each cell (can be very large).",
    )
    parser.add_argument(
        "-modelgridindex",
        "-mgi",
        "-cell",
        type=int,
        default=None,
        help="Model grid index (cell) to select. If not specified, all cells are processed.",
    )

    parser.add_argument(
        "-lambdamin", type=float, default=20.0, help="Minimum wavelength in Angstroms for binned opacities."
    )
    parser.add_argument(
        "-lambdamax", type=float, default=50000.0, help="Maximum wavelength in Angstroms for binned opacities."
    )
    parser.add_argument(
        "-deltalambda", type=float, default=10.0, help="Wavelength bin width in Angstroms for binned opacities."
    )
    parser.add_argument(
        "--plot_planck_opacities",
        "--plot_planck",
        action="store_true",
        help="Plot the resulting Planck mean opacities.",
    )
    parser.add_argument(
        "-slice",
        type=str,
        default=None,
        choices=["xy", "yx", "yz", "zy", "xz", "zx"],
        help="For 3D models, plot this slice only.",
    )
    parser.add_argument("-outputpath", type=Path, default=Path(), help="Path to output PDF")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    if args.timedays is not None:
        assert args.timestep is None, "Cannot specify both timestep and timedays. Please specify only one of them."
        timestep = at.misc.get_timestep_of_timedays(args.modelpath, args.timedays)
    else:
        timestep = args.timestep
        assert timestep is not None, "Please specify either -timestep or -timedays."

    # Step 1) get model dimension
    im = at.inputmodel.get_modeldata(modelpath=Path(args.modelpath), derived_cols=["mass_g", "velocity"])

    im_join_columns = [c for c in im[0].collect_schema().names() if "vel" in c]
    im_join_columns.append("modelgridindex")
    dfestimators = (
        at.estimators
        .scan_estimators(args.modelpath, timestep=timestep, modelgridindex=args.modelgridindex, join_modeldata=True)
        .select("modelgridindex", "timestep", "Te", "rho", "mass_g", cs.starts_with("nnion_"))
        .join(im[0].select(im_join_columns), on="modelgridindex", how="left")
        .collect()
    ).with_columns(batchindex=(pl.row_index() / 32).cast(pl.Int64))

    if args.slice:
        # reduce cells for 3D model
        if args.slice in {"xy", "yx"}:
            dfestimators = dfestimators.filter(pl.col("vel_z_min_on_c") == 0)
        elif args.slice in {"xz", "zx"}:
            dfestimators = dfestimators.filter(pl.col("vel_y_min_on_c") == 0)
        elif args.slice in {"yz", "zy"}:
            dfestimators = dfestimators.filter(pl.col("vel_x_min_on_c") == 0)

    time_days = at.misc.get_timestep_time(args.modelpath, timestep)

    print()
    print(f"timestep {timestep} time_days = {time_days:.2f}")

    adata = at.atomic.get_levels(args.modelpath, get_transitions=True, derived_transitions_columns=["lambda_angstroms"])

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(5000)
    # pl.Config.set_engine_affinity("streaming")
    cellcount = dfestimators.select(pl.len()).item()
    cells_processed = 0
    time_start = time.perf_counter()
    planckmeanopacity_times_mass = 0.0
    mass_g_sum = 0.0
    for dfcellbatch in dfestimators.partition_by("batchindex", maintain_order=True, include_key=False):
        dfbinnedopacities = get_expansion_opacities(
            adata=adata,
            time_days=time_days,
            dfestimators=dfcellbatch,
            lambdamin=args.lambdamin,
            lambdamax=args.lambdamax,
            deltalambda=args.deltalambda,
        )
        if args.show_binned_opacities:
            dfbinnedopacities = dfbinnedopacities.collect()
            print(dfbinnedopacities)

        dfplanckmean = (
            (
                dfbinnedopacities
                .lazy()
                .with_columns(lambda_cm_bin_mid=pl.col("lambda_angstroms_bin_mid") * 1e-8)
                .with_columns(
                    planckfactor=(
                        (pl.col("lambda_cm_bin_mid").pow(-5))
                        / (
                            (h_erg_s * C_cm_per_s / pl.col("lambda_cm_bin_mid") / pl.col("Te") / K_B_erg_per_K).exp()
                            - 1
                        )
                    )
                )
                .group_by("modelgridindex", "mass_g")
                .agg(
                    planckmean_opacity=(
                        (pl.col("planckfactor") * pl.col("exopac")).sum() / pl.col("planckfactor").sum()
                    )
                )
            )
            .sort("modelgridindex")
            .collect(engine="streaming")
        )

        print(dfplanckmean)
        planckmeanopacity_times_mass += (dfplanckmean.select(pl.col("planckmean_opacity").dot(pl.col("mass_g")))).item()
        mass_g_sum += dfplanckmean.select(pl.col("mass_g").sum()).item()

        cells_processed += dfcellbatch.select(pl.len()).item()
        elapsed = time.perf_counter() - time_start
        timepercell = elapsed / cells_processed
        print(
            f" average seconds per cell: {timepercell:.3f}. cells remaining: {cellcount - cells_processed}. time remaining: {timepercell * (cellcount - cells_processed):.1f}s"
        )

    if args.plot_planck_opacities:
        dfplanckmean = dfplanckmean.join(im[0].select(im_join_columns).collect(), on="modelgridindex", how="left")
        plot_planck_mean_opacity(
            args.modelpath, args.outputpath, dfplanckmean, timestep, im[1], slice_of_3D_model=args.slice
        )

    print()
    globalplanckmeanopacity = planckmeanopacity_times_mass / mass_g_sum
    print(f"Global Planck mean opacity: {globalplanckmeanopacity:.2f} cm^2/g")


if __name__ == "__main__":
    main()
