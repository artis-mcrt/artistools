"""Script for plotting the Planck mean opacity structure in postprocessing."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at

CLIGHT = 2.99792458e10
CLIGHT_KMperS = 2.99792458e5
DAYS_TO_S = 86400
A_TO_CM = 1e-8
CM_TO_A = 1e8
K_B_CGS = 1.380649e-16  # erg / K
K_B_AU = 8.617333262145e-5  # eV / K
H_CGS = 6.62607015e-27  # erg * s
HC_AU = 12398.4198
M_E_CGS = 9.1093837015e-28  # g
E_CGS = 4.803204712570263e-10  # Fr (esu)
HC_EV_ANG = 12398.4193

WAVELEN_STEPS = 1000  # 5000 K <-> 100 A bin

ion_stage_spectroscopic_notation = {1: "I", 2: "II", 3: "III", 4: "IV"}


def calc_abs_coeffs(estimators_lazyframe: pl.LazyFrame, modelpath: Path, timesecs_plot: float) -> pl.DataFrame:

    element_to_Z = {Z: at.get_elsymbol(Z) for Z in range(1, 119)}

    adata_df = at.atomic.get_levels(modelpath)

    master_levelframe = pl.concat(
        [
            lf.with_columns([pl.lit(Z).alias("Z"), pl.lit(ion).alias("ion_stage")])
            for Z, ion, lf in zip(adata_df["Z"], adata_df["ion_stage"], adata_df["levels"], strict=True)
        ],
        how="vertical",
    ).lazy()
    transdata_dict = at.atomic.get_transitiondata(modelpath)

    master_transitionframe = (
        pl
        .concat(
            [
                lf.with_columns([pl.lit(Z).alias("Z"), pl.lit(ion_stage).alias("ion_stage")])
                for (Z, ion_stage), lf in transdata_dict.items()
            ],
            how="vertical",
        )
        .filter(pl.col("forbidden") == 0.0)
        .join(
            master_levelframe.select(
                pl.col("levelindex").alias("lower"),
                pl.col("energy_ev").alias("lower_level_energy_eV"),
                pl.col("g").alias("lower_level_g"),
                "Z",
                "ion_stage",
            ),
            on=["lower", "Z", "ion_stage"],
            how="left",
        )
        .join(
            master_levelframe.select(
                pl.col("levelindex").alias("upper"),
                pl.col("energy_ev").alias("upper_level_energy_eV"),
                pl.col("g").alias("upper_level_g"),
                "Z",
                "ion_stage",
            ),
            on=["upper", "Z", "ion_stage"],
            how="left",
        )
        .with_columns([
            (HC_EV_ANG / (pl.col("upper_level_energy_eV") - pl.col("lower_level_energy_eV"))).alias(
                "wavelength_Angstrom"
            ),
            (
                (pl.col("upper_level_g") / pl.col("lower_level_g"))
                * (HC_EV_ANG / (pl.col("upper_level_energy_eV") - pl.col("lower_level_energy_eV"))) ** 2
                * pl.col("A")
                * 1.49919e-16
            ).alias("oscillator_strength"),
        ])
        .with_columns((pl.col("wavelength_Angstrom") * 1e-8).alias("wavelength_cm"))
        .drop([
            "collstr",
            "forbidden",
            "upper_level_energy_eV",
            "upper_level_g",
            "lower",
            "upper",
            "A",
            "wavelength_Angstrom",
        ])
        # Build the ion_density_key here, on the transition side
        .with_columns([
            pl.col("Z").replace_strict(element_to_Z, default="UNKNOWN").alias("element"),
            pl
            .col("ion_stage")
            .replace_strict(ion_stage_spectroscopic_notation, default="UNKNOWN")
            .alias("ion_stage_str"),
        ])
        .with_columns(
            pl.concat_str([pl.lit("nnion_"), pl.col("element"), pl.lit("_"), pl.col("ion_stage_str")]).alias(
                "ion_density_key"
            )
        )
        .drop(["element", "ion_stage_str"])
    )
    ion_densities_long = estimators_lazyframe.select(["modelgridindex", pl.col("^nnion_.*$")]).unpivot(
        index="modelgridindex", variable_name="ion_density_key", value_name="ion_density"
    )

    cell_frame = estimators_lazyframe.select(["modelgridindex", "rho", "TR", "wavelength_binwidth_cm"]).unique(
        "modelgridindex"
    )
    master = master_transitionframe.join(ion_densities_long, on="ion_density_key", how="inner").join(
        cell_frame, on="modelgridindex", how="left"
    )
    tr_frame = estimators_lazyframe.select(["modelgridindex", "TR"]).unique()

    partition_function = (
        master_levelframe
        .join(tr_frame, how="cross")
        .with_columns(
            (pl.col("g") * (-pl.col("energy_ev") / (pl.lit(K_B_AU) * pl.col("TR"))).exp()).alias("boltzmann_weight")
        )
        .group_by(["Z", "ion_stage", "modelgridindex"])
        .agg(pl.sum("boltzmann_weight").alias("partition_function"))
    )

    master = master.join(partition_function, on=["Z", "ion_stage", "modelgridindex"], how="left").drop([
        "Z",
        "ion_stage",
        "ion_density_key",
    ])
    master = master.with_columns(
        (
            pl.col("ion_density")
            * (-pl.col("lower_level_energy_eV") / (pl.lit(K_B_AU) * pl.col("TR"))).exp()
            * pl.col("lower_level_g")
            / pl.col("partition_function")
        ).alias("lower_level_number_density")
    ).drop(["ion_density", "lower_level_g", "lower_level_energy_eV", "partition_function"])
    master = (
        master
        .with_columns(
            (
                np.pi
                * E_CGS**2
                / (CLIGHT * M_E_CGS)
                * timesecs_plot
                * pl.col("wavelength_cm")
                * pl.col("oscillator_strength")
                * pl.col("lower_level_number_density")
            ).alias("optical_depth")
        )
        .with_columns((1 - (-pl.col("optical_depth")).exp()).alias("absorption_probability"))
        .with_columns(
            (
                pl.col("absorption_probability")
                * pl.col("wavelength_cm")
                / pl.col("wavelength_binwidth_cm")
                / (CLIGHT * timesecs_plot)
            ).alias("expansion_abs_coeff")
        )
        .drop(["optical_depth", "absorption_probability", "lower_level_number_density", "oscillator_strength"])
    )
    master = (
        master
        .with_columns((pl.col("wavelength_cm") / pl.col("wavelength_binwidth_cm")).cast(pl.Int64).alias("bin_index"))
        .group_by(["modelgridindex", "bin_index"])
        .agg([
            pl.sum("expansion_abs_coeff").alias("expansion_bin_abs_coeff"),
            pl.first("TR").alias("TR"),
            pl.first("wavelength_cm").alias("wavelength_cm"),
            pl.first("rho").alias("rho"),
            pl.first("wavelength_binwidth_cm").alias("wavelength_binwidth_cm"),
        ])
    )
    master = (
        master
        .with_columns(
            (
                2
                * H_CGS
                * CLIGHT**2
                / (
                    pl.col("wavelength_cm") ** 5
                    * ((H_CGS * CLIGHT / (pl.col("wavelength_cm") * K_B_CGS * pl.col("TR"))).exp() - 1.0)
                )
            ).alias("planck_weight")
        )
        .with_columns(
            (pl.col("expansion_bin_abs_coeff") * pl.col("planck_weight") * pl.col("wavelength_binwidth_cm")).alias(
                "num"
            ),
            (pl.col("planck_weight") * pl.col("wavelength_binwidth_cm")).alias("den"),
        )
        .group_by("modelgridindex")
        .agg([pl.sum("num").alias("num"), pl.sum("den").alias("den")])
        .with_columns((pl.col("num") / pl.col("den")).alias("Planck_mean_absorption_coeff"))
        .select(["modelgridindex", "Planck_mean_absorption_coeff"])
    )
    result = (
        estimators_lazyframe
        .join(master, on="modelgridindex", how="left")
        .with_columns((pl.col("Planck_mean_absorption_coeff") / pl.col("rho")).alias("Planck_mean_opacity"))
        .drop("Planck_mean_absorption_coeff")
    )

    print("Calculating and collecting cell opacities...")
    return result.collect()


def plot_opacity(
    dimension: int,
    modelpath: Path,
    outputpath: Path,
    estimators_frame: pl.DataFrame,
    x_coord: str | None,
    y_coord: str | None,
    numb_x_pts: int | None,
    numb_y_pts: int | None,
) -> None:
    plot_info = ""

    fig, axis = plt.subplots(nrows=1, ncols=1)
    at.plottools.set_mpl_style()

    if dimension == 0:
        # 1-zone model
        estimators_frame = estimators_frame.sort("timedays", descending=False)
        axis.plot(estimators_frame["timedays"], estimators_frame["Planck_mean_opacity"])
        axis.set_xlabel("time (days)")
        axis.set_ylabel(r"Planck mean opacity (cm$^2$ g$^{-1}$)")
        plot_info = "0D"
    elif dimension == 1:
        # 1D model, plot opacity as function of radius
        estimators_frame = estimators_frame.sort("velocity", descending=False)
        axis.plot(estimators_frame["timedays"] / CLIGHT_KMperS, estimators_frame["Planck_mean_opacity"])
        axis.set_xlabel("velocity (fraction of c)")
        axis.set_ylabel(r"Planck mean opacity (cm$^2$ g$^{-1}$)")
        plot_info = "1D"
    elif dimension == 2:
        # 2D model or 2D slice from 3D model, plot as 2D grid at a specific time
        assert x_coord is not None
        assert y_coord is not None
        assert numb_x_pts is not None
        assert numb_y_pts is not None

        colorscale = np.ma.masked_where(
            estimators_frame["Planck_mean_opacity"] == 0.0, estimators_frame["Planck_mean_opacity"]
        )
        valuegrid = colorscale.reshape((numb_x_pts, numb_y_pts))

        vmin_ax1 = estimators_frame.select(pl.col(f"vel_{x_coord}_min_on_c").min()).item()
        vmax_ax1 = estimators_frame.select(pl.col(f"vel_{x_coord}_max_on_c").max()).item()
        vmin_ax2 = estimators_frame.select(pl.col(f"vel_{y_coord}_min_on_c").min()).item()
        vmax_ax2 = estimators_frame.select(pl.col(f"vel_{y_coord}_max_on_c").max()).item()

        cellsize = 0.5
        figwidth = numb_x_pts * cellsize
        figheight = numb_y_pts * cellsize
        fig, axis = plt.subplots(figsize=(figwidth, figheight))

        im = axis.imshow(
            valuegrid,
            cmap="viridis",
            interpolation="nearest",
            extent=(vmin_ax1, vmax_ax1, vmin_ax2, vmax_ax2),
            origin="lower",
            aspect="auto",
        )

        axis.set_xlabel(r"v$_{" + str(x_coord) + r"}$ [$c$]", fontsize=16)
        axis.set_ylabel(r"v$_{" + str(y_coord) + r"}$ [$c$]", fontsize=16)
        axis.tick_params(axis="both", which="major", labelsize=16)

        cbar = fig.colorbar(im, ax=axis, orientation="horizontal", pad=0.05)
        cbar.set_label(r"$\kappa_{Pl}$ [cm$^2$ $g^{-1}$]", fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        fig.tight_layout(pad=0.5)
        plot_info = f"2D{x_coord}{y_coord}"

    defaultfilename = modelpath / Path(f"plotplanckopac_{plot_info}.pdf")
    outfilename = (
        outputpath / Path(f"plotplanckopac_{plot_info}.pdf") if outputpath and outputpath.is_dir() else defaultfilename
    )
    plt.savefig(outfilename, format="pdf", dpi=300)
    print(f"Saved {outfilename}")
    plt.close()


def select_2D_slice(estimators_lazyframe: pl.LazyFrame, sliceplane: str) -> tuple[pl.LazyFrame, str, str, int, int]:
    assert sliceplane in {"xy", "xz", "yz"}, "Slice must be either x=0, y=0 or z=0."
    estimators_dataframe = estimators_lazyframe.collect()

    if sliceplane == "xy":
        return (
            estimators_lazyframe.filter((pl.col("pos_z_min") <= 0.0) & (pl.col("pos_z_max") >= 0.0)),
            "x",
            "y",
            estimators_dataframe.select(pl.col("pos_x_min").n_unique()).item(),
            estimators_dataframe.select(pl.col("pos_y_min").n_unique()).item(),
        )

    if sliceplane == "xz":
        return (
            estimators_lazyframe.filter((pl.col("pos_y_min") <= 0.0) & (pl.col("pos_y_max") >= 0.0)),
            "x",
            "z",
            estimators_dataframe.select(pl.col("pos_x_min").n_unique()).item(),
            estimators_dataframe.select(pl.col("pos_z_min").n_unique()).item(),
        )

    if sliceplane == "yz":
        return (
            estimators_lazyframe.filter((pl.col("pos_x_min") <= 0.0) & (pl.col("pos_x_max") >= 0.0)),
            "y",
            "z",
            estimators_dataframe.select(pl.col("pos_y_min").n_unique()).item(),
            estimators_dataframe.select(pl.col("pos_z_min").n_unique()).item(),
        )

    msg = "Should not be reached"
    raise ValueError(msg)


def addargs(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path of ARTIS model")

    parser.add_argument(
        "-tdays", type=float, default=1.0, help="Time in days for the 2D opacity plot. Either in 2D or 3D mode."
    )

    parser.add_argument("-slice", default="xy", help="Plane of slice in case of a 3D model. Example: xy <-> z=0.")

    parser.add_argument("-outputpath", type=Path, default=Path(), help="Path to output PDF")


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
    snopshot_time_days = im[1]["t_model_init_days"]

    # add density to estimators
    timesteps_lazyframe = at.get_timesteps(args.modelpath)
    im_join_columns = ["modelgridindex", "rho"]
    if model_dim in {0, 1}:
        pass
    elif model_dim == 2:
        im_join_columns.extend(["vel_rcyl_mid", "vel_z_mid"])
    elif model_dim == 3:
        pass
    estimators_lazyframe = (
        at.estimators
        .scan_estimators(modelpath=Path(args.modelpath))
        .join(
            timesteps_lazyframe.select(pl.col("timestep"), pl.col("tstart_days").alias("tdays")),
            left_on="timestep",
            right_on="timestep",
            how="left",
        )
        .with_columns(((snopshot_time_days / pl.col("tdays")) ** 3).alias("exp_factor"))
        .join(im[0].select(im_join_columns), on="modelgridindex", how="left")
        .with_columns((pl.col("rho") * pl.col("exp_factor")).alias("rho"))
    )
    estimators_lazyframe = estimators_lazyframe.with_columns(
        (4 / pl.col("TR") / WAVELEN_STEPS).alias("wavelength_binwidth_cm")
    )

    plotaxis1 = None
    plotaxis2 = None
    n_ax1 = None
    n_ax2 = None
    plot_dimension = model_dim
    if model_dim > 0:
        assert args.tdays is not None, "No time specified. Abort."
        # select closest timestep for plotting
        res = (
            timesteps_lazyframe
            .filter(pl.col("tmid_days") <= args.tdays)
            .select([pl.col("timestep").max().alias("plot_ts"), pl.col("tmid_days").max().alias("timedays_plot")])
            .collect()
            .row(0)
        )

        plot_ts, timedays_plot = res
        estimators_lazyframe = estimators_lazyframe.filter(pl.col("timestep") == plot_ts).sort("modelgridindex")
        if model_dim == 2:
            # 2D: 2D-plot of ejecta opacity at one specified time
            # 3D: select 2D slice first and then plot
            plotaxis1 = "r"
            plotaxis2 = "z"
            n_ax1 = im[1]["ncoordgridrcyl"]
            n_ax2 = im[1]["ncoordgridz"]
        elif model_dim == 3:
            plot_dimension = 2
            estimators_lazyframe, plotaxis1, plotaxis2, n_ax1, n_ax2 = select_2D_slice(estimators_lazyframe, args.slice)
    """floers_test = {
        "modelgridindex": 0,
        "TR": 5000.0,
        "rho": 1e-13,
        "wavelength_binwidth_cm": 1e-6,
        "nnion_Yb_I": 0.0,
        "nnion_Yb_II": 0.53 * 1e-13 / (173 * 1.66e-24),
        "nnion_Yb_III": 0.47 * 1e-13 / (173 * 1.66e-24),
        "nnion_Yb_IV": 0.0,
    }
    estimators_lazyframe = pl.DataFrame(floers_test).lazy()
    timedays_plot = 1.0"""
    estimators_dataframe = calc_abs_coeffs(estimators_lazyframe, args.modelpath, timedays_plot * DAYS_TO_S)
    plot_opacity(
        plot_dimension, args.modelpath, args.outputpath, estimators_dataframe, plotaxis1, plotaxis2, n_ax1, n_ax2
    )


if __name__ == "__main__":
    main()
