"""Script for plotting the Planck mean opacity structure of 2D ARTIS models and slices of 3D runs."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path
import itertools
from tqdm import tqdm
import math

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at

CLIGHT = 2.99792458e10
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
LAZYFRAME_FILTER_FACTOR = 10

master_levels_DF = pl.DataFrame()  
master_trans_dict = {} 
global_level_DF = pl.DataFrame()
global_trans_DF = pl.LazyFrame()


def planck_lambda(T: float, wavelen_A: np.ndarray) -> np.ndarray:
    wavelen_cm = wavelen_A * A_TO_CM
    x = H_CGS * CLIGHT / (K_B_CGS * T * wavelen_cm)
    prefac = 2.0 * H_CGS * CLIGHT**2 / wavelen_cm**5
    large = x > 100  # exp(700) ~ float limit
    small = ~large

    result = np.zeros_like(x)

    result[small] = prefac[small] / np.expm1(x[small])
    result[large] = prefac[large] * np.exp(-x[large])

    return result


def calc_ionic_Planck_abs_coeff(T, t_days, Z, ion_stage, n_ij, bin_centres, bin_edges, bin_width):
    # calculate Planck opacity for a single ion using trapezoidal integration
    global master_levels_DF
    global global_trans_DF
    row = master_levels_DF.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))
    levels_df = row.select("levels").item()
    E_max_eV = 10 * T * K_B_AU
    trans_lf = master_trans_dict[(Z, ion_stage)]  # lf: lazy frame

    E_ion_eV = row.select("ion_pot").item()
    part_fct = levels_df.select((pl.col("g") * (-pl.col("energy_ev") / (K_B_AU * T)).exp()).sum()).item()

    beta_pereV = 1 / (T * K_B_AU)
    t_exp = t_days * DAYS_TO_S

    n_l_expr = pl.col("g_l") * n_ij / part_fct * (-pl.col("E_l") / (T * K_B_AU)).exp()
    trans_DF = global_trans_DF.with_columns(n_l_expr.alias("n_l"))

    opt_depth_prefactor = -np.pi * E_CGS**2 / (CLIGHT * M_E_CGS) * t_exp

    trans_DF = trans_DF.with_columns(
        (
            1 - (opt_depth_prefactor * pl.col("wavelength_A") * A_TO_CM * pl.col("f_lu") * pl.col("n_l")).exp()
        ).alias("abs_prob")
    )

    # here: reduced number of additional columns
    trans_DF = trans_DF.with_columns(
        (pl.col("wavelength_A") / bin_width * pl.col("abs_prob") / (CLIGHT * t_exp)).alias("exp_abs_coeff_contr")
    )
    # here is the absorption coefficient calculation
    trans_DF = trans_DF.with_columns(
        (
            pl.col("wavelength_A") / bin_width * pl.col("abs_prob") / (CLIGHT * t_exp)
        ).alias("exp_abs_coeff_contr")
    )
    trans_DF = trans_DF.with_columns(
        pl.col("wavelength_A").cut(bin_edges, labels=None).alias("wavelength_bin_idx").cast(pl.Int32)
    )
    all_bins = pl.DataFrame({"wavelength_bin_idx": list(range(WAVELEN_STEPS))}).lazy()
    all_bins = all_bins.with_columns(pl.col("wavelength_bin_idx").cast(pl.Int32))
    exp_abs_coeff_df = (
    trans_DF.group_by("wavelength_bin_idx")
        .agg(pl.sum("exp_abs_coeff_contr").alias("exp_abs_coeff"))
    )
    exp_abs_coeff_df = all_bins.join(exp_abs_coeff_df, on="wavelength_bin_idx", how="left").with_columns(
        pl.col("exp_abs_coeff").fill_null(0)
    )
    B_wavelength_table = planck_lambda(T, bin_centres)
    
    exp_abs_coeff_data = (
        exp_abs_coeff_df
        .sort("wavelength_bin_idx")
        .select("exp_abs_coeff") 
        .collect()
        .to_numpy()
    ).flatten()

    # trapezoidal integration
    Planck_abs_coeff_num = np.sum(
        0.5
        * (bin_centres[1:] - bin_centres[:-1])
        * (
            (exp_abs_coeff_data[:-1] * B_wavelength_table[:-1] / bin_centres[:-1] ** 2)
            + (exp_abs_coeff_data[1:] * B_wavelength_table[1:] / bin_centres[1:] ** 2)
        )
    )

    Planck_abs_coeff_den = np.sum(
        0.5
        * (bin_centres[1:] - bin_centres[:-1])
        * ((B_wavelength_table[:-1] / bin_centres[:-1] ** 2) + (B_wavelength_table[1:] / bin_centres[1:] ** 2))
    )

    return Planck_abs_coeff_num / Planck_abs_coeff_den


def g_e(T: float) -> float:
    lambda_therm_e = H_CGS / np.sqrt(2 * np.pi * M_E_CGS * K_B_CGS * T)
    return 2 / lambda_therm_e**3


def phi(T: float, Z_ijp1: float, Z_ij: float, boltzmann_factor: float) -> float:
    return g_e(T) * Z_ijp1 / Z_ij * boltzmann_factor


def alpha(T: float, part_fcts_list: Sequence[float], boltzmann_list: Sequence[float], n_e: float) -> float:
    # alpha function as defined in the TARDIS wiki, compare
    # https://tardis-sn.github.io/tardis/physics_walkthrough/setup/plasma/lte_plasma.html
    alpha = 1
    factor = 1
    for i in range(len(part_fcts_list) - 1):
        factor *= phi(T, part_fcts_list[i + 1], part_fcts_list[i], boltzmann_list[i]) / n_e
        alpha += factor
    return alpha


def calc_atomic_chi_Planck(Z: int, t_exp: float, T: float, n_e: float, n_i: float) -> float:
    global global_level_DF
    global global_trans_DF
    # Step 1) calculate ionisation balance
    ion_stages = [key[1] for key in master_trans_dict.keys() if key[0] == Z]
    numb_ions = len(ion_stages)
    n_ij_arr = np.zeros(numb_ions)
    part_fcts = np.zeros(numb_ions)
    boltzmann_factors = np.zeros(numb_ions)
    for idx, ion_stage in enumerate(ion_stages):
        row = master_levels_DF.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))
        levels_df = row.select("levels").item()
        part_fcts[idx] = levels_df.select((pl.col("g") * (-pl.col("energy_ev") / (K_B_AU * T)).exp()).sum()).item()
        boltzmann_factors[idx] = np.exp(-row.select("ion_pot").item() / (K_B_AU * T))
    n_ij_arr[0] = n_i / alpha(T, part_fcts, boltzmann_factors, n_e)
    for idx, ion_stage in enumerate(ion_stages):
        if idx == 0:
            continue
        n_ij_arr[idx] = (
            n_ij_arr[idx - 1] / n_e * g_e(T) * part_fcts[idx] / part_fcts[idx - 1] * boltzmann_factors[idx - 1]
        )

    # Step 2) prepare wavelength bins depending on temperature
    max_lambda = CM_TO_A / T  # maximum wavelength in Angstrom
    bin_width = math.ceil(max_lambda / WAVELEN_STEPS)
    bin_edges = np.arange(0, bin_width * (WAVELEN_STEPS + 1), bin_width)  
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2                

    # Step 3) perform the main computation of absorption coefficients
    atomic_Planck_abs_coeff = 0
    for idx, n_ij in enumerate(n_ij_arr):
        ion_stage = idx + 1  # ion_charge in ARTIS atomic data is 1-based, i.e. neutral <-> 1 !!
        # reset global dataframes and perform join on transition dataframe
        global_level_DF = (
            master_levels_DF.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage)).select("levels").item()
        )
        global_trans_DF = master_trans_dict[(Z, ion_stage)]
        global_trans_DF = global_trans_DF.join(
            global_level_DF
                .select([
                    pl.col("levelindex").alias("lower"),
                    pl.col("energy_ev").alias("E_l"),
                    pl.col("g").alias("g_l")
                ])
                .lazy(),
            on="lower",
            how="left",
        )
        global_trans_DF = global_trans_DF.join(
            global_level_DF
                .select([
                    pl.col("levelindex").alias("upper"),
                    pl.col("energy_ev").alias("E_u"),
                    pl.col("g").alias("g_u")
                ])
                .lazy(),
            on="upper",
            how="left",
        )
        global_trans_DF = global_trans_DF.with_columns([
            ((HC_EV_ANG) / (pl.col("E_u") - pl.col("E_l"))).alias("wavelength_A")
        ])
        global_trans_DF = global_trans_DF.with_columns([
            (
                (pl.col("g_u") / pl.col("g_l")) 
                * pl.col("wavelength_A")**2 
                * pl.col("A") 
                * C_AU 
                / (8 * np.pi**2)
            ).alias("f_lu")
        ])
        if n_ij > 1e-30:
            atomic_Planck_abs_coeff += calc_ionic_Planck_abs_coeff(
                T, t_exp, Z, ion_stage, n_ij, bin_centres, bin_edges, bin_width
            )
    return atomic_Planck_abs_coeff


def create_opacity_table(modelpath: Path):
    """
    create a Planck mean opacity table as a polars dataframe stored in a parquet file. Tabulation parameters:
    - Z
    - T
    - rho: ATTENTION rho can just be multiplied and is not tabulated therefore
    - t_exp
    - n_e
    - n_i
    """
    # Step 1) Load atomic data
    # load atomic data first
    global master_levels_DF
    global master_trans_dict
    master_levels_DF = at.atomic.get_levels(Path(modelpath))  # gives a pl.DataFrame (eager)
    # transition data dictionary
    master_trans_dict = at.atomic.get_transitiondata(Path(modelpath))  # gives a dict with keys for every ion
    Z_range = np.array(sorted({k[0] for k in master_trans_dict.keys()}))

    # Step 2) Get tabulation boundaries
    t_range = np.concatenate([[0.1, 0.5], np.arange(1.0, 4.5, 0.5), np.arange(5.0, 11.0, 1.0)])
    T_range = np.arange(500, 10500, 500)  # ionisation stages above not covered in atomic data
    n_i_range = 10 ** np.arange(4, 15)
    n_e_range = 10 ** np.arange(7, 17)

    # Step 3) calculate opacities
    ATOMNUMB, TIME, TEMP, NE, NI = np.meshgrid(Z_range, t_range, T_range, n_e_range, n_i_range, indexing="ij")
    ATOMNUMB_flat = ATOMNUMB.ravel()
    TIME_flat = TIME.ravel()
    TEMP_flat = TEMP.ravel()
    NE_flat = NE.ravel()
    NI_flat = NI.ravel()
    chi_column = np.array([
        calc_atomic_chi_Planck(Z, t, T, n_e, n_i)
        for Z, t, T, n_e, n_i in tqdm(
            zip(ATOMNUMB_flat, TIME_flat, TEMP_flat, NE_flat, NI_flat),
            total=len(ATOMNUMB_flat)
        )
    ])
    abs_coeff_df = pl.DataFrame({
        "Z": ATOMNUMB_flat,
        "t_exp": TIME_flat.astype(np.float32),
        "T": TEMP_flat.astype(np.float32),
        "n_e": NE_flat.astype(np.float32),
        "n_i": NI_flat.astype(np.float32),
        "chi_Planck": chi_column.astype(np.float32),
    })

    # Step 4) dump result to parquet file
    abs_coeff_df.write_parquet(modelpath / Path("planck_abs_coeff_table.parquet"))


def get_abs_coeff_from_table(abs_coeff_df,Z,t_exp,T,n_e,n_i):
    df_with_dist = (
        abs_coeff_df
        .with_columns(
            (
                (pl.col("Z") - Z)**2 +
                (pl.col("t_exp") - t_exp)**2 +
                (pl.col("T") - T)**2 +
                (pl.col("n_e") - n_e)**2 +
                (pl.col("n_i") - n_i)**2
            ).alias("param_dist")
        )
    )
    
    nearest_row = df_with_dist.sort("param_dist").limit(1).collect()
    
    return nearest_row["chi_Planck"][0]


def plot_opacity_vs_time_0dmodel(modelpath: Path, estimators_lazyframe: pl.LazyFrame, abs_coeff_df: pl.LazyFrame):
    timesteps_lazyframe = at.get_timesteps(modelpath)
    numb_ts = timesteps_lazyframe.select(pl.count()).collect().item()
    tstart_days_arr = timesteps_lazyframe.select("tstart_days").collect()["tstart_days"].to_numpy()
    kappa_Pl_arr = np.zeros(numb_ts)
    Z_arr = abs_coeff_df.select(pl.col("Z")).unique().collect().to_numpy().flatten()

    for ts_idx, tdays in enumerate(tstart_days_arr):
        t_exp = tdays * DAYS_TO_S
        row = estimators_lazyframe.filter(pl.col("timestep") == ts_idx).select(["TR", "nne", "rho"]).collect()
        T_R = row["TR"].item()
        n_e  = row["nne"].item()
        rho  = row["rho"].item()

        filtered_abs_coeff_df = abs_coeff_df.filter(
            (pl.col("rho") >= rho / LAZYFRAME_FILTER_FACTOR) & (pl.col("rho") <= rho * LAZYFRAME_FILTER_FACTOR) &
            (pl.col("TR")  >= T_R / LAZYFRAME_FILTER_FACTOR) & (pl.col("TR")  <= T_R * LAZYFRAME_FILTER_FACTOR) &
            (pl.col("nne") >= n_e / LAZYFRAME_FILTER_FACTOR) & (pl.col("nne") <= n_e * LAZYFRAME_FILTER_FACTOR)
        )
        for Z in Z_arr:
            n_i = estimators_lazyframe.select(f"nnelement_{at.get_elsymbol(Z)}").collect()[f"nnelement_{at.get_elsymbol(Z)}"].item()
            kappa_Pl_arr[ts_idx] += get_abs_coeff_from_table(filtered_abs_coeff_df,Z,t_exp,T_R,n_e,n_i) / rho

    plt.plot(tstart_days_arr,opac_values)
    plt.xscale('log')
    plt.xlabel('time (days)')
    plt.ylabel(r'Planck mean opacity (cm$^2$ g$^{-1}$)')

    outfilename = (
        Path(modelpath) / "plotplanckopac_vstime_0D.pdf"
    )

    plt.savefig(outfilename, format="pdf",dpi=300)
    plt.close(fig) 

    print(f"Saved {outfilename}")


def plot_opacity_vs_radius_1dmodel(estimators_lazyframe: pl.LazyFrame, abs_coeff_df: pl.LazyFrame, tdays: float):
    mgi_arr = estimators_lazyframe.select(pl.col("modelgridindex")).unique().collect().to_numpy().flatten().sort()
    numb_cells = len(mgi_arr)
    kappa_Pl_arr = np.zeros(numb_cells)
    Z_arr = abs_coeff_df.select(pl.col("Z")).unique().collect().to_numpy().flatten()

    # get closest timestep
    timesteps_lazyframe = at.get_timesteps(Path(args.modelpath))
    plot_ts = (timesteps_lazyframe.filter(pl.col("tstart_days") <= t_d).select(pl.col("timestep").max())).collect().item()
    reduced_estimators = estimators_lazyframe.filter(pl.col("timestep") == plot_ts)
    reduced_estimators_df = reduced_estimators.sort("modelgridindex").collect()
    T_R_arr = reduced_estimators_df["TR"].to_numpy()
    n_e_arr = reduced_estimators_df["nne"].to_numpy()
    rho_arr = reduced_estimators_df["rho"].to_numpy()
    t_exp = tdays * DAYS_TO_S

    for Z in Z_arr:
        n_i_arr = reduced_estimators_df[f"nnelement_{at.get_elsymbol(Z)}"].to_numpy() 
        for mgi_idx, mgi in enumerate(mgi_arr):
            kappa_Pl_arr[mgi_idx] += get_abs_coeff_from_table(filtered_abs_coeff_df,Z,t_exp,T_R_arr[mgi_idx],n_e_arr[mgi_idx],n_i_arr[mgi_idx]) / rho_arr[mgi_idx]

    plt.plot(tstart_days_arr,opac_values)
    plt.xlabel('velocity (fraction of c)')
    plt.ylabel(r'Planck mean opacity (cm$^2$ g$^{-1}$)')

    outfilename = (
        Path(modelpath) / f"plotplanckopac_vsradius_1D_{tdays}d.pdf"
    )

    plt.savefig(outfilename, format="pdf",dpi=300)
    plt.close(fig) 

    print(f"Saved {outfilename}")


def plot_opacity_2dslice(
    estimators_lazyframe: pl.LazyFrame, 
    abs_coeff_df: pl.LazyFrame, 
    tdays: float,
    plotaxis1: str,
    plotaxis2: str,
) -> None:
    """Plot a 2D slice of Planck opacity with quadratically scaled cells.

    Each cell in the plot will appear square. The overall figure aspect ratio
    is determined by n_plotaxis2 / n_plotaxis1.

    Parameters
    ----------
    estimators_lazyframe: pl.LazyFrame
        estimator lazyframe (reduced to slice and time)
    abs_coeff_df : pl.LazyFrame
        DataFrame containing the opacity data.
    tdays: float
        expansion time
    plotaxis1, plotaxis2 : int
        decides which axes to plot
    """
    # reduce estimators dataframe
    timesteps_lazyframe = at.get_timesteps(Path(args.modelpath))
    plot_ts = (timesteps_lazyframe.filter(pl.col("tstart_days") <= t_d).select(pl.col("timestep").max())).collect().item()
    reduced_estimators = estimators_lazyframe.filter(pl.col("timestep") == plot_ts)
    # no 3D version yet. Current 2D version sorts running in positive z-direction and radially outwards
    reduced_estimators_df = reduced_estimators.sort("modelgridindex").collect() 
    # default values, generalise later
    n_plotaxis1 = 25
    n_plotaxis2 = 50
    T_R_arr = reduced_estimators_df["TR"].to_numpy()
    n_e_arr = reduced_estimators_df["nne"].to_numpy()
    rho_arr = reduced_estimators_df["rho"].to_numpy()
    mgi_arr = reduced_estimators.select(pl.col("modelgridindex")).unique().collect().to_numpy().flatten().sort()
    numb_cells = len(mgi_arr)
    kappa_Pl_arr = np.zeros(numb_cells)
    t_exp = tdays * DAYS_TO_S
    Z_arr = abs_coeff_df.select(pl.col("Z")).unique().collect().to_numpy().flatten()

    for Z in Z_arr:
        n_i_arr = reduced_estimators_df[f"nnelement_{at.get_elsymbol(Z)}"].to_numpy() 
        
        for mgi_idx, mgi in enumerate(mgi_arr):
            kappa_Pl_arr[mgi_idx] += get_abs_coeff_from_table(filtered_abs_coeff_df,Z,t_exp,T_R_arr[mgi_idx],n_e_arr[mgi_idx],n_i_arr[mgi_idx]) / rho_arr[mgi_idx]

    colorscale = odf["kappa_Pl"].to_numpy()


    colorscale = np.ma.masked_where(colorscale == 0.0, colorscale)
    valuegrid = colorscale.reshape((n_y, n_x))

    vmin_ax1 = odf.select(pl.col(f"vel_{plotaxis1}_min_on_c").min()).item()
    vmax_ax1 = odf.select(pl.col(f"vel_{plotaxis1}_max_on_c").max()).item()
    vmin_ax2 = odf.select(pl.col(f"vel_{plotaxis2}_min_on_c").min()).item()
    vmax_ax2 = odf.select(pl.col(f"vel_{plotaxis2}_max_on_c").max()).item()

    cellsize = 0.5 
    figwidth = n_plotaxis1 * cellsize
    figheight = n_plotaxis2 * cellsize
    fig, ax = plt.subplots(figsize=(figwidth, figheight))

    im = ax.imshow(
        valuegrid,
        cmap="viridis",
        interpolation="nearest",
        extent=(vmin_ax1, vmax_ax1, vmin_ax2, vmax_ax2),
        origin="lower",
        aspect="auto", 
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

    plt.savefig(outfilename, format="pdf",dpi=300)
    plt.close(fig) 

    print(f"Saved {outfilename}")


def select_3D_slice(elf: pl.LazyFrame, sliceplane: str) -> tuple[pl.LazyFrame, str, str]:
    assert sliceplane in {"xy", "xz", "yz"}, "Slice must be either x=0, y=0 or z=0."

    if sliceplane == "xy":
        return elf.filter((pl.col("pos_z_min") <= 0.0) & (pl.col("pos_z_max") >= 0.0)), "x", "y"

    if sliceplane == "xz":
        return elf.filter((pl.col("pos_y_min") <= 0.0) & (pl.col("pos_y_max") >= 0.0)), "x", "z"

    if sliceplane == "yz":
        return elf.filter((pl.col("pos_x_min") <= 0.0) & (pl.col("pos_x_max") >= 0.0)), "y", "z"

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

    # search for opacity table first and if not found create it
    file_path = Path(args.modelpath / "planck_abs_coeff_table.parquet")
    if not file_path.exists():
        print("No opacity table found. Create Planck mean absorption coefficient table first...")
        create_opacity_table(args.modelpath)
    else:
        print(f"Found {str(file_path)}. Use it to create the plots")

    # get inputmodel, tuple of LazyFrame and dictionary
    im = at.inputmodel.get_modeldata(modelpath=Path(args.modelpath), derived_cols=["mass_g", "velocity"])
    model_dim = im[1]["dimensions"]
    t_snap_days = im[1]["t_model_init_days"]

    # add density to estimators
    estimators_lazyframe = (
        at.estimators.scan_estimators(modelpath=Path(args.modelpath))
        .join(
            ts_lazyframe.select(pl.col("timestep"), pl.col("tstart_days").alias("tdays")),
            left_on="lower",
            right_on="timestep",
            how="left"
        )
        .with_columns(((t_snap_days / pl.col("tdays")) ** 3).alias("exp_factor"))
        .join(
            im[0].select(["modelgridindex", "rho", "velocity"]),
            on="modelgridindex",
            how="left"
        )
        .with_columns((pl.col("rho") * pl.col("exp_factor")).alias("rho"))
    )

    # scan opacity dataframe first
    abs_coeff_df = pl.scan_parquet(Path(args.modelpath / "planck_abs_coeff_table.parquet"))

    if model_dim == 0:
        # 0D / one-zone: plot opacity as function of time
        plot_opacity_vs_time_0dmodel(args.modelpath, estimators_lazyframe, abs_coeff_df)
    elif model_dim == 1:
        # 1D: ejecta opacity as function of radius for specified range of time
        plot_opacity_vs_radius_1dmodel(args.modelpath, estimators_lazyframe, abs_coeff_df,args.tdays)
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
            plot_opacity_2dslice(
                args.modelpath, opac_data, n_ax1, n_ax2, plotaxis1, plotaxis2, outputpath=args.outputpath
            )
        elif model_dim == 3:
            elf_ts = elf.filter(pl.col("timestep") == plot_ts)
            elf_ts = elf_ts.join(
                im[0]
                .select(["modelgridindex", "vel_r_min_on_c", "vel_r_max_on_c", "vel_z_min_on_c", "vel_z_max_on_c"])
                .collect(),
                on="modelgridindex",
                how="left",
            )

            elf_slice, plotaxis1, plotaxis2 = select_3D_slice(elf_ts, args.slice)

            elf_slice = elf_slice.sort("modelgridindex")

            elf_slice = elf_slice.join(
                im[0].select(["modelgridindex", "rho"]), on="modelgridindex", how="left"
            ).with_columns((pl.col("rho") * exp_factor).alias("rho"))

            opac_data = calc_cell_opacs(args.modelpath, elf_slice, t_s)
            opac_data = opac_data.join(
                im[0]
                .select(["modelgridindex", "vel_r_min_on_c", "vel_r_max_on_c", "vel_z_min_on_c", "vel_z_max_on_c"])
                .collect(),
                on="modelgridindex",
                how="left",
            )

            pos_cols = [
                "modelgridindex",
                f"pos_{plotaxis1}_min",
                f"pos_{plotaxis1}_max",
                f"pos_{plotaxis2}_min",
                f"pos_{plotaxis2}_max",
            ]

            opac_data = opac_data.join(elf_slice.select(pos_cols), on="modelgridindex", how="left")

            n_ax1 = opac_data.select(pl.col(f"pos_{plotaxis1}_min")).n_unique()
            n_ax2 = opac_data.select(pl.col(f"pos_{plotaxis2}_min")).n_unique()

            plot_opacity_2dslice(
                args.modelpath, opac_data, n_ax1, n_ax2, plotaxis1, plotaxis2, outputpath=args.outputpath
            )


if __name__ == "__main__":
    main()
