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
C_AU = 137.035999
HC_AU = 12398.4198
M_E_CGS = 9.1093837015e-28  # g
E_CGS = 4.803204712570263e-10  # Fr (esu)
HC_EV_ANG = 12398.4193

WAVELEN_STEPS = 250  # 25000 A <-> 100 A bin

roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4}


def calc_abs_coeffs(estimators_lazyframe: pl.LazyFrame, modelpath: Path) -> pl.DataFrame:
    element_to_Z = {at.get_elsymbol(Z): Z for Z in range(1, 101)}
    """Calculate opacities as a new column of estimators lazyframe."""
    # Step 1) load atomic level data from adata.txt
    adata_df = at.atomic.get_levels(modelpath)  # gives a pl.DataFrame (eager), convert to lazyframes
    master_levelframe = pl.concat(
        [
            lf.with_columns([pl.lit(Z).alias("Z"), pl.lit(ion).alias("ion_stage")])
            for Z, ion, lf in zip(adata_df["Z"], adata_df["ion_stage"], adata_df["levels"], strict=True)
        ],
        how="vertical",
    )

    # Step 2) create master lazyframe from transitiondata.txt
    transdata_dict = at.atomic.get_transitiondata(modelpath)  # gives a dict with keys for every ion
    frames = []

    # merge the single ion transition lazyframes into a huge single one
    for (Z, ion_stage), lf in transdata_dict.items():
        frames.append(lf.with_columns([pl.lit(Z).alias("charge_number"), (pl.lit(ion_stage) - 1).alias("ion_charge")]))
    master_transitionframe = pl.concat(frames, how="vertical")

    # Step 3) expand the master frame using the information from level dataframes

    master_transitionframe = master_transitionframe.join(
        master_levelframe.select(
            pl.col("levelindex").alias("lower"),
            pl.col("energy_ev").alias("lower_level_energy_eV"),
            pl.col("g").alias("lower_level_g"),
        ).lazy(),
        on="lower",
        how="left",
    )
    master_transitionframe = master_transitionframe.join(
        master_levelframe.select(
            pl.col("levelindex").alias("upper"),
            pl.col("energy_ev").alias("upper_level_energy_eV"),
            pl.col("g").alias("upper_level_g"),
        ).lazy(),
        on="upper",
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
            * C_AU
            / (8 * np.pi**2)
        ).alias("f_lu")
    ])
    master_transitionframe = master_transitionframe.with_columns(
        (pl.col("wavelength_Angstrom") * 1e-8).alias("wavelength_cm")
    )

    # Step 4) merge with estimators data to obtain the master frame
    estimators_lazyframe_long = estimators_lazyframe.unpivot(index=[], variable_name="species_key", value_name="nnion")
    estimators_lazyframe_long = estimators_lazyframe_long.with_columns([
        pl.col("species_key").str.extract(r"n_density_([A-Za-z]+)_([IVXLCDM]+)", 1).alias("element"),
        pl.col("species_key").str.extract(r"n_density_([A-Za-z]+)_([IVXLCDM]+)", 2).alias("roman_stage"),
    ])
    estimators_lazyframe_long = estimators_lazyframe_long.with_columns(
        pl.col("element").replace_strict(element_to_Z).alias("Z")
    )
    master_lazyframe = master_transitionframe.join(
        estimators_lazyframe_long.select(["Z", "ion_stage", "n_density", "rho", "tdays", "Trad", "modelgridindex"]),
        on=["Z", "ion_stage"],  # BUGALERT
        how="left",
    )
    master_lazyframe = master_lazyframe.with_columns((pl.col("tdays") * DAYS_TO_S).alias("time_s"))

    # Step 5) calculate contributions of individual transitions to the absorption coefficient
    master_lazyframe = master_lazyframe.with_columns(
        (
            pl.col("ion_number_density")
            * np.exp(-pl.col("lower_level_energy") / (K_B_CGS * pl.col("temperature")))
            * pl.col("lower_level_g")
            / pl.col("ground_state_g")
        ).alias("lower_level_number_density")
    )

    master_lazyframe = master_lazyframe.with_columns(
        (
            np.pi
            * E_CGS**2
            / (CLIGHT * M_E_CGS)
            * pl.col("time_s")
            * pl.col("wavelength_cm")
            * pl.col("oscillator_strength")
            * pl.col("lower_level_number_density")
        ).alias("optical_depth")
    )
    master_lazyframe = master_lazyframe.with_columns(
        (1 - np.exp(-pl.col("optical_depth"))).alias("absorption_probability")
    )

    master_lazyframe = master_lazyframe.with_columns(
        (
            pl.col("absorption_probability")
            * pl.col("wavelength_cm")
            / pl.col("wavelength_binwidth_cm")
            / (CLIGHT * pl.col("time_s"))
        ).alias("Planck_absorption_coefficient_contribution")
    )
    # Step 6) Reduce master_frame again to obtain the total opacities
    master_lazyframe = master_lazyframe.group_by([
        "modelgridindex",
        "wavelength_cm",
        "wavelength_binwidth_cm",
        "temperature",
    ]).agg(pl.sum("Planck_absorption_coefficient_contribution").alias("bin_absorption_coefficient"))

    master_lazyframe = master_lazyframe.with_columns(
        planck_weight=(
            1.0
            / (
                pl.col("wavelength_cm") ** 5
                * ((CLIGHT**2 / (pl.col("wavelength_cm") * pl.col("temperature"))).exp() - 1.0)
            )
        )
    )

    master_lazyframe = master_lazyframe.with_columns(
        numerator_term=(
            pl.col("bin_absorption_coefficient") * pl.col("planck_weight") * pl.col("wavelength_binwidth_cm")
        ),
        denominator_term=(pl.col("planck_weight") * pl.col("wavelength_binwidth_cm")),
    )

    master_lazyframe = (
        master_lazyframe
        .group_by("modelgridindex")
        .agg([pl.sum("numerator_term").alias("num"), pl.sum("denominator_term").alias("den")])
        .with_columns((pl.col("num") / pl.col("den")).alias("Planck_absorption_coeff"))
    )
    opacity_lazyframe = master_lazyframe.select(["modelgridindex", "planck_mean_absorption_coefficient"])
    estimators_lazyframe = estimators_lazyframe.join(opacity_lazyframe, on="modelgridindex", how="left")

    # Step 7) multiply by density to obtain the opacity and return the estimators again
    return estimators_lazyframe.with_columns(
        (pl.col("Planck_absorption_coeff") * pl.col("rho")).alias("Planck_opacity")
    ).collect()


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
        axis.plot(estimators_frame["timedays"], estimators_frame["Planck_opacity"])
        axis.set_xlabel("time (days)")
        axis.set_ylabel(r"Planck mean opacity (cm$^2$ g$^{-1}$)")
        plot_info = "0D"
    elif dimension == 1:
        # 1D model, plot opacity as function of radius
        estimators_frame = estimators_frame.sort("velocity", descending=False)
        axis.plot(estimators_frame["timedays"] / CLIGHT_KMperS, estimators_frame["Planck_opacity"])
        axis.set_xlabel("velocity (fraction of c)")
        axis.set_ylabel(r"Planck mean opacity (cm$^2$ g$^{-1}$)")
        plot_info = "1D"
    elif dimension == 2:
        # 2D model or 2D slice from 3D model, plot as 2D grid at a specific time
        assert x_coord is not None
        assert y_coord is not None
        assert numb_x_pts is not None
        assert numb_y_pts is not None

        colorscale = np.ma.masked_where(estimators_frame["Planck_opacity"] == 0.0, estimators_frame["Planck_opacity"])
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
    estimators_lazyframe = (
        at.estimators
        .scan_estimators(modelpath=Path(args.modelpath))
        .join(
            timesteps_lazyframe.select(pl.col("timestep"), pl.col("tstart_days").alias("tdays")),
            left_on="lower",
            right_on="timestep",
            how="left",
        )
        .with_columns(((snopshot_time_days / pl.col("tdays")) ** 3).alias("exp_factor"))
        .join(im[0].select(["modelgridindex", "rho", "velocity"]), on="modelgridindex", how="left")
        .with_columns((pl.col("rho") * pl.col("exp_factor")).alias("rho"))
    )
    estimators_lazyframe = estimators_lazyframe.with_columns((1.25 / pl.col("Trad")).alias("wavelength_binwidth_cm"))

    plotaxis1 = None
    plotaxis2 = None
    n_ax1 = None
    n_ax2 = None
    plot_dimension = model_dim
    if model_dim > 0:
        assert args.tdays is not None, "No time specified. Abort."
        # select closest timestep for plotting
        plot_ts = (
            (timesteps_lazyframe.filter(pl.col("tstart_days") <= args.tdays).select(pl.col("timestep").max()))
            .collect()
            .item()
        )
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
    estimators_dataframe = calc_abs_coeffs(estimators_lazyframe, args.modelpath)
    plot_opacity(
        plot_dimension, args.modelpath, args.outputpath, estimators_dataframe, plotaxis1, plotaxis2, n_ax1, n_ax2
    )


if __name__ == "__main__":
    main()
