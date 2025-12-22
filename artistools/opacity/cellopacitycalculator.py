"""Script for calculating the Planck mean opacity of any ARTIS model cell."""

import math
from pathlib import Path

import numpy as np
import polars as pl

import artistools as at

m_u_cgs = 1.66054 * 1e-24
k_b_cgs = 1.380649e-16  # erg / K
k_b_au = 8.617333262145e-5  # eV / K
h_cgs = 6.62607015e-27  # erg * s
c_cgs = 2.99792458e10  # cm / s
m_e_cgs = 9.1093837015e-28  # g
e_cgs = 4.803204712570263e-10  # Fr (esu)
sigma_cgs = 5.6704e-9  # g s^-3 K^-4

wl_steps = 250  # 25000 A <-> 100 A bin

# dict of atomic masses in units of m_u
atomic_masses = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.941,
    "Be": 9.012,
    "B": 10.811,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.086,
    "P": 30.974,
    "S": 32.066,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.631,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 84.798,
    "Rb": 84.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98.907,
    "Ru": 101.07,
    "Rh": 102.906,
    "Pd": 106.42,
    "Ag": 107.868,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.711,
    "Sb": 121.760,
    "Te": 126.7,
    "I": 126.904,
    "Xe": 131.294,
    "Cs": 132.905,
    "Ba": 137.328,
    "La": 138.905,
    "Ce": 140.116,
    "Pr": 140.908,
    "Nd": 144.243,
    "Pm": 144.913,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.925,
    "Dy": 162.500,
    "Ho": 164.930,
    "Er": 167.259,
    "Tm": 168.934,
    "Yb": 173.055,
    "Lu": 174.967,
    "Hf": 178.49,
    "Ta": 180.948,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.085,
    "Au": 196.967,
    "Hg": 200.592,
    "Tl": 204.383,
    "Pb": 207.2,
    "Bi": 208.980,
    "Po": 208.982,
    "At": 209.987,
    "Rn": 222.081,
    "Fr": 223.020,
    "Ra": 226.025,
    "Ac": 227.028,
    "Th": 232.038,
    "Pa": 231.036,
    "U": 238.029,
    "Np": 237,
    "Pu": 244,
    "AM": 243,
    "Cm": 247,
    "Bk": 247,
    "Ct": 251,
    "Es": 252,
    "FM": 257,
    "Md": 258,
    "No": 259,
    "Lr": 262,
    "Rf": 261,
    "DB": 262,
    "Sg": 266,
    "Bh": 264,
    "Hs": 269,
    "Mt": 268,
    "DS": 271,
    "Rg": 272,
    "Cn": 285,
    "Nh": 284,
    "Fl": 289,
    "MC": 288,
    "Lv": 292,
    "Ts": 294,
    "Og": 294,
}


def Planck_spectrum(T: float, wl: float) -> float:
    # wl: wavelength in Angström
    wl_cm = wl / 10**8  # Angström to cm
    return 2 * h_cgs * c_cgs**2 / wl_cm**5 * 1 / (np.exp(h_cgs * c_cgs / (k_b_cgs * T * wl_cm)) - 1)


def canonical_part_fct(T: float, ldf: pl.DataFrame) -> float:
    beta = 1 / (T * k_b_au)
    return ldf.select((pl.col("g_values") * (-beta * pl.col("E_values")).exp()).sum()).item()


def integrated_exp_opac(
    T: float, texp_s: float, n_ion: float, rho: float, tddf: pl.DataFrame, ldf: pl.DataFrame
) -> float:
    # set the integration wavelength grid
    max_wl = 10**8 / T
    bin_width = math.ceil(max_wl / wl_steps)
    wl_bincentres = np.linspace(bin_width / 2, bin_width * (wl_steps + 1 / 2), wl_steps)
    wlbinedges = [wl - bin_width / 2 for wl in wl_bincentres] + [wl_bincentres[-1] + bin_width / 2]
    Planck_spectrum_values = [Planck_spectrum(T, wl) for wl in wl_bincentres]
    expopac_times_Planck = np.zeros(len(wl_bincentres))

    # now calculate the core arrays for integration
    part_fct_value = canonical_part_fct(T, ldf)
    tddf = tddf.with_columns(
        (n_ion * pl.col("g_l") / part_fct_value * (-pl.col("E_l") / (T * k_b_au)).exp()).alias("n_l")
    )

    n_l = n_ion * tddf.select(pl.col("g_l") / part_fct_value * (-pl.col("E_l") / (T * k_b_au)).exp()).to_series()

    assert len(n_l) == len(tddf), "Error. Number of lower level densities does not match transition dataframe length!"

    prefactor = -np.pi * e_cgs**2 / (c_cgs * m_e_cgs) * 1e-8 * texp_s
    one_minus_Sob_opt_depth = 1 - (prefactor * tddf["wavelength_A"] * tddf["f_lu"] * n_l).exp()

    # here: reduced number of additional columns
    tddf = tddf.with_columns(
        (pl.col("wavelength_A") / bin_width * one_minus_Sob_opt_depth / (c_cgs * texp_s * rho)).alias("exp_opac_contr")
    )

    # pd.cut here instead of between
    """
    tddf = tddf.with_columns(
        pl.cut(
            pl.col("wavelength_A"),
            bins=wlbinedges,
            labels=list(range(len(wl_bincentres))),
            include_lowest=True,
        ).alias("wl_bin")
    )
    # .groupby here
    expopac_times_Planck = (
        tddf
        .groupby("wl_bin")
        .agg(pl.col("exp_opac_contr").sum())
        .sort("wl_bin")
        .select("exp_opac_contr")
        .to_series()
    )
    """
    wl_values = tddf["wavelength_A"].to_numpy()
    wl_bins = np.digitize(wl_values, wlbinedges) - 1  # np.digitize liefert 1-basiert

    exp_opac_contr = (tddf["wavelength_A"] / bin_width * one_minus_Sob_opt_depth / (c_cgs * texp_s * rho)).to_numpy()

    expopac_times_Planck = np.zeros(len(wl_bincentres))
    for i in range(len(wl_bincentres)):
        expopac_times_Planck[i] = exp_opac_contr[wl_bins == i].sum()

    assert len(expopac_times_Planck) == len(Planck_spectrum_values), (
        "Different number of opacity and Planck spectrum values!"
    )

    expopac_times_Planck *= Planck_spectrum_values

    integral_expopac_Planck = np.trapezoid(expopac_times_Planck, wl_bincentres)

    return float(integral_expopac_Planck)


def calc_Planck_mean_opacity(cdlf: pl.LazyFrame, ldf: pl.DataFrame, tdd: dict, texp_s: float) -> pl.DataFrame:
    # returns a LazyFrame with ionic, elemental and total Planck mean opacity for every cell
    odf = cdlf.select(["modelgridindex", "TR", "rho"]).collect()
    nion_df = cdlf.collect()
    T_values = odf["TR"].to_numpy()
    rho_values = odf["rho"].to_numpy()

    # loop over all atoms of ions
    for ion_tuple, tditemlf in tdd.items():
        el_symbol = at.get_elsymbol(ion_tuple[0])
        ion_str = at.get_ionstring(ion_tuple[0], ion_tuple[1])
        ion_stage_str = ion_str.split()[1]

        tddf = tditemlf.collect()
        ion_ldf = ldf.filter((pl.col("Z") == ion_tuple[0]) & (pl.col("ion_stage") == ion_tuple[1]))
        n_ion_values = nion_df[f"nnion_{el_symbol}_{ion_stage_str}"].to_numpy()

        kappa_Pl_numerators = np.array([
            integrated_exp_opac(T, texp_s, n_ion, rho, tddf, ion_ldf)
            for (T, n_ion, rho) in zip((T_values, n_ion_values, rho_values))
        ])

        ion_col_name = f"kappa_Pl_{ion_str}"
        odf = odf.with_columns(pl.Series(ion_col_name, kappa_Pl_numerators))

        # denominator from Planck mean opacity
        odf = odf.with_columns((pl.col(ion_col_name) * sigma_cgs * pl.col("Trad") ** 4 / np.pi).alias(ion_col_name))
        # create or add to elemental Planck mean opacity
        el_col_name = f"kappa_Pl_{el_symbol}"
        if el_col_name not in odf.columns:
            odf = odf.with_columns(pl.col(ion_col_name).alias(el_col_name))
        else:
            odf = odf.with_columns((pl.col(el_col_name) + pl.col(ion_col_name)).alias(el_col_name))
        # add values to total Planck mean opacity
        odf = odf.with_columns((pl.col("kappa_Pl") + pl.col(ion_col_name)).alias("kappa_Pl"))
    return odf


def calc_cell_opacs(modelpath: Path, cell_data: pl.LazyFrame, texp_s: float) -> pl.DataFrame:
    # load atomic data first
    llf = at.atomic.get_levels(Path(modelpath))  # gives a pl.DataFrame (eager)
    # transition data dictionary
    tdd = at.atomic.get_transitiondata(Path(modelpath))

    # gets all required cell Data as polars LazyFrame
    odf = calc_Planck_mean_opacity(cell_data, llf, tdd, texp_s)

    # return data frame with cell positions, opacity, etc.
    return odf.sort("modelgridindex")
