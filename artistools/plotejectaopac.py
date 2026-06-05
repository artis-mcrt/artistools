"""Script for plotting the Planck mean opacity structure in postprocessing."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

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


def load_ARTIS_run_atomic_data(modelpath: Path) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Calculate opacities as a new column of estimators lazyframe."""
    # Step 1) load atomic level data from adata.txt
    adata_df = at.atomic.get_levels(modelpath)  # gives a pl.DataFrame (eager), convert to lazyframes
    master_levelframe = pl.concat(
        [
            lf.with_columns([pl.lit(Z).alias("Z"), pl.lit(ion).alias("ion_stage")])
            for Z, ion, lf in zip(adata_df["Z"], adata_df["ion_stage"], adata_df["levels"], strict=True)
        ],
        how="vertical",
    ).lazy()

    # Step 2) create master lazyframe from transitiondata.txt
    transdata_dict = at.atomic.get_transitiondata(modelpath)  # gives a dict with keys for every ion
    frames = []

    # merge the single ion transition lazyframes into a huge single one
    for (Z, ion_stage), lf in transdata_dict.items():
        frames.append(lf.with_columns([pl.lit(Z).alias("Z"), pl.lit(ion_stage).alias("ion_stage")]))
    master_transitionframe = pl.concat(frames, how="vertical")  # .top_k(k=3_000, by="A")
    master_transitionframe = master_transitionframe.filter(pl.col("forbidden") == 0.0).lazy()

    # Step 3) expand the master frame using the information from level dataframes
    master_transitionframe = master_transitionframe.join(
        master_levelframe.select(
            pl.col("levelindex").alias("lower"),
            pl.col("energy_ev").alias("lower_level_energy_eV"),
            pl.col("g").alias("lower_level_g"),
            pl.col("Z"),
            pl.col("ion_stage"),
        ),
        on=["lower", "Z", "ion_stage"],
        how="left",
    ).join(
        master_levelframe.select(
            pl.col("levelindex").alias("upper"),
            pl.col("energy_ev").alias("upper_level_energy_eV"),
            pl.col("g").alias("upper_level_g"),
            pl.col("Z"),
            pl.col("ion_stage"),
        ),
        on=["upper", "Z", "ion_stage"],
        how="left",
    )

    master_transitionframe = master_transitionframe.with_columns([
        ((HC_EV_ANG) / (pl.col("upper_level_energy_eV") - pl.col("lower_level_energy_eV"))).alias("wavelength_Angstrom")
    ])
    master_transitionframe = master_transitionframe.with_columns([
        (
            (pl.col("upper_level_g") / pl.col("lower_level_g"))
            * pl.col("wavelength_Angstrom") ** 2
            * pl.col("A")
            * 1.49919e-16
        ).alias("oscillator_strength")
    ])
    # apply gf-value selection criterion from Gillanders+26 to reduce the amount of data
    master_transitionframe = master_transitionframe.filter(
        (pl.col("lower_level_g") * pl.col("oscillator_strength")) > 1e-3
    )
    master_transitionframe = master_transitionframe.with_columns(
        (pl.col("wavelength_Angstrom") * 1e-8).alias("wavelength_cm")
    )
    master_transitionframe = master_transitionframe.drop([
        "collstr",
        "forbidden",
        "upper_level_energy_eV",
        "upper_level_g",
    ]).filter(pl.col("wavelength_Angstrom").is_finite())
    return master_transitionframe, master_levelframe


def calc_abs_coeffs(
    master_transitionframe: pl.LazyFrame,
    master_levelframe: pl.LazyFrame,
    estimators_lazyframe: pl.LazyFrame,
    timesecs_plot: float,
) -> pl.LazyFrame:
    element_to_Z = {Z: at.get_elsymbol(Z) for Z in range(1, 119)}
    # Step 1) merge with estimators data to obtain the master frame
    master_lazyframe = master_transitionframe.join(
        estimators_lazyframe.select(["modelgridindex", "rho", "TR", "wavelength_binwidth_cm"]), how="cross"
    )
    ion_densities_long = estimators_lazyframe.select(["modelgridindex", pl.col("^nnion_.*$")]).unpivot(
        index=["modelgridindex"], variable_name="ion_density_key", value_name="ion_density"
    )
    # partition functions
    partition_function_lazyframe = (
        master_levelframe
        .select(["Z", "ion_stage", pl.col("energy_ev"), pl.col("g")])
        .join(estimators_lazyframe.select(["modelgridindex", "TR"]).unique(), how="cross")
        .with_columns(
            (pl.col("g") * (-pl.col("energy_ev") / (pl.lit(K_B_AU) * pl.col("TR"))).exp()).alias("boltzmann_weight")
        )
        .group_by(["Z", "ion_stage", "modelgridindex"])
        .agg(pl.sum("boltzmann_weight").alias("partition_function"))
    )
    master_lazyframe = master_lazyframe.join(
        partition_function_lazyframe, on=["Z", "ion_stage", "modelgridindex"], how="left"
    )
    master_lazyframe = master_lazyframe.with_columns([
        pl.col("Z").replace_strict(element_to_Z, default="UNKNOWN").alias("element"),
        pl.col("ion_stage").replace_strict(ion_stage_spectroscopic_notation, default="UNKNOWN").alias("ion_stage_str"),
    ])
    """assert (
        not master_lazyframe.select(pl.col("ion_stage_str") == "UNKNOWN").any().item()
    ), "ERROR: Unmapped ion_stage values detected (UNKNOWN present)"
    assert not master_lazyframe.select(pl.col("element") == "UNKNOWN").any().item(), (
        "ERROR: Unmapped Z values detected (UNKNOWN present)"
    )"""
    master_lazyframe = (
        master_lazyframe
        .with_columns(
            pl.concat_str([pl.lit("nnion_"), pl.col("element"), pl.lit("_"), pl.col("ion_stage_str")]).alias(
                "ion_density_key"
            )
        )
        .join(ion_densities_long, on=["modelgridindex", "ion_density_key"], how="left")
        .filter(pl.col("ion_density").is_not_null())
        .drop(["lower", "upper", "A", "Z", "ion_stage", "element", "ion_stage_str", "ion_density_key"])
    )

    # Step 2) calculate contributions of individual transitions to the absorption coefficient
    master_lazyframe = master_lazyframe.with_columns(
        (
            pl.col("ion_density")
            * (-pl.col("lower_level_energy_eV") / (pl.lit(K_B_AU) * pl.col("TR"))).exp()
            * pl.col("lower_level_g")
            / pl.col("partition_function")
        ).alias("lower_level_number_density")
    ).drop(["lower_level_g", "partition_function", "lower_level_energy_eV", "ion_density"])

    master_lazyframe = master_lazyframe.with_columns(
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
    master_lazyframe = master_lazyframe.with_columns(
        (1 - pl.col("optical_depth").neg().exp()).alias("absorption_probability")
    )

    master_lazyframe = master_lazyframe.with_columns(
        (
            pl.col("absorption_probability")
            * pl.col("wavelength_cm")
            / pl.col("wavelength_binwidth_cm")
            / (CLIGHT * timesecs_plot)
        ).alias("expansion_absorption_coefficient_contribution")
    )
    # Step 3) Reduce master_frame again to obtain the total opacities
    master_lazyframe = master_lazyframe.with_columns(
        (pl.col("wavelength_cm") / pl.col("wavelength_binwidth_cm")).cast(pl.Int64).alias("bin_index")
    )
    master_lazyframe = master_lazyframe.group_by(["modelgridindex", "bin_index"]).agg(
        pl.sum("expansion_absorption_coefficient_contribution").alias("expansion_bin_absorption_coefficient"),
        pl.mean("TR").alias("TR"),
        pl.mean("wavelength_cm").alias("wavelength_cm"),
        pl.mean("rho").alias("rho"),
        pl.mean("wavelength_binwidth_cm").alias("wavelength_binwidth_cm"),
    )
    master_lazyframe = master_lazyframe.with_columns(
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
    master_lazyframe = master_lazyframe.with_columns(
        (
            pl.col("expansion_bin_absorption_coefficient") * pl.col("planck_weight") * pl.col("wavelength_binwidth_cm")
        ).alias("numerator_term"),
        (pl.col("planck_weight") * pl.col("wavelength_binwidth_cm")).alias("denominator_term"),
    )

    master_lazyframe = (
        master_lazyframe
        .group_by("modelgridindex")
        .agg(pl.sum("numerator_term").alias("num"), pl.sum("denominator_term").alias("den"))
        .with_columns((pl.col("num") / pl.col("den")).alias("Planck_mean_absorption_coeff"))
    )
    absorption_coefficient_lazyframe = master_lazyframe.select(["modelgridindex", "Planck_mean_absorption_coeff"])
    estimators_lazyframe = estimators_lazyframe.join(absorption_coefficient_lazyframe, on="modelgridindex", how="left")

    # Step 4) multiply by density to obtain the opacity and return the estimators again
    return estimators_lazyframe.with_columns(
        (pl.col("Planck_mean_absorption_coeff") / pl.col("rho")).alias("Planck_mean_opacity")
    ).select(["Planck_mean_opacity", "modelgridindex"])


def plot_opacity(
    dimension: int,
    modelpath: Path,
    outputpath: Path,
    opacity_frame: pl.DataFrame,
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
        opacity_frame = opacity_frame.sort("timedays", descending=False)
        axis.plot(opacity_frame["timedays"], opacity_frame["Planck_mean_opacity"])
        axis.set_xlabel("time (days)")
        axis.set_ylabel(r"Planck mean opacity (cm$^2$ g$^{-1}$)")
        plot_info = "0D"
    elif dimension == 1:
        # 1D model, plot opacity as function of radius
        opacity_frame = opacity_frame.sort("vel_r_max_kmps", descending=False)
        min_val = cast("float", opacity_frame["vel_r_max_kmps"].min())

        opacity_frame = opacity_frame.with_columns((pl.col("vel_r_max_kmps") - 0.5 * min_val).alias("vel_r_max_kmps"))
        axis.plot(opacity_frame["vel_r_max_kmps"] / CLIGHT_KMperS, opacity_frame["Planck_mean_opacity"])
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
            opacity_frame["Planck_mean_opacity"] == 0.0, opacity_frame["Planck_mean_opacity"]
        )
        valuegrid = colorscale.reshape((numb_x_pts, numb_y_pts))

        vmin_ax1 = opacity_frame.select(pl.col(f"vel_{x_coord}_min_on_c").min()).item()
        vmax_ax1 = opacity_frame.select(pl.col(f"vel_{x_coord}_max_on_c").max()).item()
        vmin_ax2 = opacity_frame.select(pl.col(f"vel_{y_coord}_min_on_c").min()).item()
        vmax_ax2 = opacity_frame.select(pl.col(f"vel_{y_coord}_max_on_c").max()).item()

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
    rho_prefix = "log" if model_dim == 1 else ""
    im_join_columns = ["modelgridindex", f"{rho_prefix}rho"]
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
    )
    velocity_cols = ["modelgridindex"]
    if model_dim in {0, 1}:
        velocity_cols.extend(["vel_r_max_kmps"])
        estimators_lazyframe = estimators_lazyframe.with_columns(
            (10 ** pl.col("logrho") * pl.col("exp_factor")).alias("rho")
        )
    elif model_dim == 2:
        velocity_cols.extend(["vel_rcyl_mid", "vel_z_mid"])
        estimators_lazyframe = estimators_lazyframe.with_columns((pl.col("rho") * pl.col("exp_factor")).alias("rho"))
    elif model_dim == 3:
        pass
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
    # loop over cells here
    master_transitionframe, master_levelframe = load_ARTIS_run_atomic_data(args.modelpath)
    opacity_lazyframes = []
    indices = estimators_lazyframe.select("modelgridindex").unique().collect().to_series()

    for idx in tqdm(indices, desc="Calculating cell opacities..."):
        subset = estimators_lazyframe.filter(pl.col("modelgridindex") == idx)

        df = calc_abs_coeffs(master_transitionframe, master_levelframe, subset, timedays_plot * DAYS_TO_S)

        opacity_lazyframes.append(df.collect())
    opacity_dataframe = pl.concat(opacity_lazyframes)
    # join with velocity data for the plot
    opacity_dataframe = opacity_dataframe.join(im[0].select(velocity_cols).collect(), on="modelgridindex", how="left")
    plot_opacity(plot_dimension, args.modelpath, args.outputpath, opacity_dataframe, plotaxis1, plotaxis2, n_ax1, n_ax2)


if __name__ == "__main__":
    main()
