"""Script added by Gerrit for plotting the Planck mean opacity structure of any ARTIS run."""

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
DAYS_TO_S = 86400
A_TO_CM = 1e-8
M_U_CGS = 1.66054 * 1e-24
K_B_CGS = 1.380649e-16  # erg / K
K_B_AU = 8.617333262145e-5  # eV / K
H_CGS = 6.62607015e-27  # erg * s
C_CGS = 2.99792458e10  # cm / s
C_AU = 137.035999
HtimesC_AU = 12398.4198
M_E_CGS = 9.1093837015e-28  # g
E_CGS = 4.803204712570263e-10  # Fr (esu)
SIGMA_CGS = 5.6704e-9  # Stefan-Boltzmann constant, g s^-3 K^-4

wavelen_steps = 250  # 25000 A <-> 100 A bin


def planck_lambda(T: float, wavelen_A: np.ndarray) -> np.ndarray:
    wavelen_cm = wavelen_A * A_TO_CM
    x = H_CGS * C_CGS / (K_B_CGS * T * wavelen_cm)
    prefac = 2.0 * H_CGS * C_CGS**2 / wavelen_cm**5
    large = x > 100  # exp(700) ~ float limit
    small = ~large

    result = np.zeros_like(x)

    result[small] = prefac[small] / np.expm1(x[small])
    result[large] = prefac[large] * np.exp(-x[large])

    return result


def canonical_part_fct(T: float, ldf: pl.DataFrame) -> float:
    beta = 1 / (T * K_B_AU)
    return ldf.select((pl.col("g_values") * (-beta * pl.col("E_values")).exp()).sum()).item()


def integrated_exp_opac(
    T: float, texp_s: float, n_ion: float, rho: float, tddf: pl.DataFrame, ldf: pl.DataFrame
) -> float:

    if n_ion <= 0.0 or rho <= 0.0:
        return 0.0

    max_wl = 1e8 / T
    bin_width = max_wl / wavelen_steps
    wl_edges = np.array([i * bin_width for i in range(wavelen_steps)])
    wl_centers = wl_edges + 0.5 * bin_width
    B_lambda = planck_lambda(T, wl_centers)

    beta = 1.0 / (T * K_B_AU)
    lev_plDF = ldf["levels"][0]
    part_fct = (lev_plDF["g"].to_numpy() * np.exp(-beta * lev_plDF["energy_ev"].to_numpy())).sum()

    tr_plDF = (
        tddf
        .join(
            lev_plDF.select([pl.col("levelindex").alias("lower"), pl.col("energy_ev").alias("lower_energy_ev")]),
            on="lower",
            how="left",
        )
        .join(
            lev_plDF.select([pl.col("levelindex").alias("upper"), pl.col("energy_ev").alias("upper_energy_ev")]),
            on="upper",
            how="left",
        )
        .join(lev_plDF.select([pl.col("levelindex").alias("lower"), pl.col("g").alias("g_l")]), on="lower", how="left")
        .join(lev_plDF.select([pl.col("levelindex").alias("upper"), pl.col("g").alias("g_u")]), on="upper", how="left")
        .with_columns((pl.col("upper_energy_ev") - pl.col("lower_energy_ev")).alias("delta_energy_ev"))
    )
    tr_plDF = tr_plDF.with_columns((HtimesC_AU / pl.col("delta_energy_ev")).alias("wavelength_A"))
    E_l = tr_plDF["lower_energy_ev"].to_numpy()
    g_l = tr_plDF["g_l"].to_numpy()
    wl = tr_plDF["wavelength_A"].to_numpy()
    # TODO: check this
    f_lu = tr_plDF["g_u"] / tr_plDF["g_l"] * tr_plDF["wavelength_A"] ** 2 * tr_plDF["A"] * C_AU / (8 * np.pi**2)

    n_l = n_ion * g_l / part_fct * np.exp(-E_l * beta)

    prefactor = np.pi * E_CGS**2 / (C_CGS * M_E_CGS) * 1e-8 * texp_s
    tau = prefactor * wl * f_lu * n_l
    one_minus_exp_tau = -np.expm1(-tau)

    kappa_line = wl / bin_width * one_minus_exp_tau / (C_CGS * texp_s * rho)
    kappa_line = np.asarray(kappa_line)

    bin_index = np.digitize(wl, wl_edges) - 1
    valid = (bin_index >= 0) & (bin_index < wavelen_steps)

    kappa_binned = np.zeros(wavelen_steps)
    np.add.at(kappa_binned, bin_index[valid], kappa_line[valid])

    numerator = np.trapezoid(kappa_binned * B_lambda, wl_centers)
    denominator = np.trapezoid(B_lambda, wl_centers)

    return numerator / denominator


def calc_Planck_mean_opacity(cdlf: pl.LazyFrame, ldf: pl.DataFrame, tdd: dict, texp_s: float) -> pl.DataFrame:

    odf = cdlf.select(["modelgridindex", "TR", "rho"]).collect()
    nion_df = cdlf.collect()

    odf = odf.with_columns(pl.lit(0.0).alias("kappa_Pl"))

    T_values = odf["TR"].to_numpy()
    rho_values = odf["rho"].to_numpy()

    # for every ion
    for ion_tuple, tditemlf in tdd.items():
        tddf = tditemlf.collect()
        ion_ldf = ldf.filter((pl.col("Z") == ion_tuple[0]) & (pl.col("ion_stage") == ion_tuple[1]))

        n_ion_values = nion_df[
            f"nnion_{at.get_elsymbol(ion_tuple[0])}_{at.misc.roman_numerals[ion_tuple[1]]}"
        ].to_numpy()
        kappa_vals = np.array([
            integrated_exp_opac(T, texp_s, nion, rho, tddf, ion_ldf)
            for T, nion, rho in zip(T_values, n_ion_values, rho_values, strict=True)
        ])

        # new column for every ion
        odf = odf.with_columns(pl.Series(f"kappa_{ion_tuple}", kappa_vals))

        # add ion column to total Planck opacity
        odf = odf.with_columns((pl.col("kappa_Pl") + pl.col(f"kappa_{ion_tuple}")).alias("kappa_Pl"))

    return odf


def calc_cell_opacs(modelpath: Path, cell_data: pl.LazyFrame, texp_s: float) -> pl.DataFrame:
    # load atomic data first
    levels_plDF = at.atomic.get_levels(Path(modelpath))  # gives a pl.DataFrame (eager)
    # transition data dictionary
    trans_dict = at.atomic.get_transitiondata(Path(modelpath))  # gives a dict with keys for every ion

    # gets all required cell Data as polars LazyFrame
    odf = calc_Planck_mean_opacity(cell_data, levels_plDF, trans_dict, texp_s)

    # return data frame with cell positions, opacity, etc.
    return odf.sort("modelgridindex")


def make_2D_slice_opac_plot(
    modelpath: Path,
    odf: pl.DataFrame,
    n_x: int,
    n_y: int,
    plotaxis1: str,
    plotaxis2: str,
    outputpath: Path | None = None,
) -> None:
    """Plot a 2D slice of Planck opacity with quadratically scaled cells.

    Each cell in the plot will appear square. The overall figure aspect ratio
    is determined by n_y / n_x.

    Parameters
    ----------
    modelpath : Path
        Path to the model (used for default saving).
    odf : pl.DataFrame
        DataFrame containing the opacity data and velocity columns.
    n_x, n_y : int
        Number of cells along x- and y-axis.
    plotaxis1, plotaxis2 : str
        Names of the velocity axes to plot.
    outputpath : Path | None
        Optional directory for saving the plot.

    """
    colorscale = odf["kappa_Pl"].to_numpy()
    colorscale = np.ma.masked_where(colorscale == 0.0, colorscale)
    valuegrid = colorscale.reshape((n_y, n_x))

    vmin_ax1 = odf.select(pl.col(f"vel_{plotaxis1}_min_on_c").min()).item()
    vmax_ax1 = odf.select(pl.col(f"vel_{plotaxis1}_max_on_c").max()).item()
    vmin_ax2 = odf.select(pl.col(f"vel_{plotaxis2}_min_on_c").min()).item()
    vmax_ax2 = odf.select(pl.col(f"vel_{plotaxis2}_max_on_c").max()).item()

    cellsize = 0.5  # size of each cell in inches (an adjustable parameter)
    figwidth = n_x * cellsize
    figheight = n_y * cellsize
    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    im = ax.imshow(
        valuegrid,
        cmap="viridis",
        interpolation="nearest",
        extent=(vmin_ax1, vmax_ax1, vmin_ax2, vmax_ax2),
        origin="lower",
        aspect="auto",  # aspect is handled by figure sizing
    )

    ax.set_xlabel(r"v$_{" + str(plotaxis1) + r"}$ [$c$]", fontsize=16)
    ax.set_ylabel(r"v$_{" + str(plotaxis2) + r"}$ [$c$]", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05)
    cbar.set_label(r"$\kappa_{Pl}$ [cm$^2$ $g^{-1}$]", fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout(pad=0.5)

    defaultfilename = Path(modelpath) / "plotplanckopac.pdf"
    outfilename = (
        Path(outputpath) / "plotplanckopac.pdf" if outputpath and Path(outputpath).is_dir() else defaultfilename
    )

    plt.savefig(outfilename, format="pdf")
    plt.close(fig)  # free memory

    print(f"Saved {outfilename}")


def select_3D_slice(elf: pl.LazyFrame, sliceplane: str) -> tuple[pl.LazyFrame, str, str]:
    assert sliceplane in {"xy", "xz", "yz"}, "Slice must be either x=0, y=0 or z=0."

    if sliceplane == "xy":
        return elf.filter((pl.col("pos_z_min") <= 0.0) & (pl.col("pos_z_max") >= 0.0)), "x", "y"

    if sliceplane == "xz":
        return elf.filter((pl.col("pos_y_min") <= 0.0) & (pl.col("pos_y_max") >= 0.0)), "x", "z"

    if sliceplane == "yz":
        return elf.filter((pl.col("pos_x_min") <= 0.0) & (pl.col("pos_x_max") >= 0.0)), "y", "z"

    raise ValueError


def addargs(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path of ARTIS model")

    parser.add_argument("-tdays", type=float, help="Time in days for the 2D opacity plot. Either in 2D or 3D mode.")

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
    t_snap_days = im[1]["t_model_init_days"]

    # elf: estimators Lazy Frame
    elf = at.estimators.scan_estimators(modelpath=Path(args.modelpath))

    opac_data: pl.DataFrame

    if model_dim == 0:
        # 0D / one-zone: plot opacity as function of time
        # opac_data = calc_cell_opacs(args.modelpath, elf)
        pass
    elif model_dim == 1:
        # 1D: ejecta opacity as function of radius for specified range of time
        pass
    elif model_dim in {2, 3}:
        # 2D: 2D-plot of ejecta opacity at one specified time
        assert args.tdays is not None, "No time specified. Abort."
        # get closest timestep
        t_d = args.tdays
        t_s = t_d * DAYS_TO_S
        exp_factor = (t_snap_days / t_d) ** 3
        ts_lf = at.get_timesteps(Path(args.modelpath))

        plot_ts = (ts_lf.filter(pl.col("tstart_days") <= t_d).select(pl.col("timestep").max())).collect().item()

        if model_dim == 2:
            plotaxis1 = "r"
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
                im[0]
                .select(["modelgridindex", "vel_r_min_on_c", "vel_r_max_on_c", "vel_z_min_on_c", "vel_z_max_on_c"])
                .collect(),
                on="modelgridindex",
                how="left",
            )
            make_2D_slice_opac_plot(
                args.modelpath, opac_data, n_ax1, n_ax2, plotaxis1, plotaxis2, outputpath=args.outputpath
            )
        elif model_dim == 3:
            # ---- select time ----
            assert args.tdays is not None, "No time specified."
            t_d = float(args.tdays)
            t_s = t_d * DAYS_TO_S

            ts_lf = at.get_timesteps(Path(args.modelpath))
            plot_ts = ts_lf.filter(pl.col("tstart_days") <= t_d).select(pl.col("timestep").max()).collect().item()

            # ---- reduce to selected timestep ----
            elf_ts = elf.filter(pl.col("timestep") == plot_ts)

            # ---- select slice of 3D model ----
            elf_slice, plotaxis1, plotaxis2 = select_3D_slice(elf_ts, args.slice)

            elf_slice = elf_slice.sort("modelgridindex")

            # ---- append density to dataframe ----
            exp_factor = (t_snap_days / t_d) ** 3

            elf_slice = elf_slice.join(
                im[0].select(["modelgridindex", "rho"]), on="modelgridindex", how="left"
            ).with_columns((pl.col("rho") * exp_factor).alias("rho"))

            # ---- calculate opacities ----
            opac_data = calc_cell_opacs(args.modelpath, elf_slice, t_s)

            # ---- append position data ----
            pos_cols = [
                "modelgridindex",
                f"pos_{plotaxis1}_min",
                f"pos_{plotaxis1}_max",
                f"pos_{plotaxis2}_min",
                f"pos_{plotaxis2}_max",
            ]

            opac_data = opac_data.join(elf_slice.select(pos_cols), on="modelgridindex", how="left")

            # ---- determine grid dimensions ----
            n_ax1 = opac_data.select(pl.col(f"pos_{plotaxis1}_min")).n_unique()
            n_ax2 = opac_data.select(pl.col(f"pos_{plotaxis2}_min")).n_unique()

            make_2D_slice_opac_plot(
                args.modelpath, opac_data, n_ax1, n_ax2, plotaxis1, plotaxis2, outputpath=args.outputpath
            )


if __name__ == "__main__":
    main()
