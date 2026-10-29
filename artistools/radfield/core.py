"""Read the radiation field estimators of the cells, and give the binned field and the fitted field."""

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.constants import c_ang_per_s
from artistools.constants import h_erg_s
from artistools.constants import K_B_erg_per_K
from artistools.misc import get_artis_option
from artistools.misc import read_rank_outputfiles


def read_radfield(modelpath: Path | str, modelgridindex: int | Sequence[int] | None = None) -> pl.DataFrame:
    """Read radiation field data from a model folder, possibly with a modelgridindex filter."""
    return read_rank_outputfiles(modelpath, "radfield_{mpirank:04d}.out", modelgridindex=modelgridindex)


def select_radfield_subset(
    radfielddata: pl.DataFrame, binfilter: pl.Expr, modelgridindex: int | None, timestep: int | None
) -> pl.DataFrame:
    """Filter radfield rows by a bin_num condition and optionally by modelgridindex and timestep."""
    subset = radfielddata.filter(binfilter)
    if modelgridindex is not None:
        subset = subset.filter(pl.col("modelgridindex") == modelgridindex)
    if timestep is not None:
        subset = subset.filter(pl.col("timestep") == timestep)
    return subset


def get_binaverage_field(
    radfielddata: pl.DataFrame, modelgridindex: int | None = None, timestep: int | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Get the dJ/dlambda constant average estimators of each bin.

    The estimator J of a bin does not depend on the fit of the bin, thus a bin with no fit (T_R < 0) still gives
    its J.
    """
    # exclude the global fit parameters and detailed lines with negative "bin_num"
    bindata = select_radfield_subset(radfielddata, pl.col("bin_num") >= 0, modelgridindex, timestep)

    arr_lambda = np.array(get_binedges(bindata))

    yvalues = (
        bindata
        .with_columns(dlambda=c_ang_per_s * (1 / pl.col("nu_lower") - 1 / pl.col("nu_upper")))
        .select(pl.col("J") / pl.col("dlambda"))
        .to_series()
        .to_numpy()
    )

    # the first bin edge is a starting point with no field
    return arr_lambda, np.insert(yvalues, 0, 0.0)


def j_nu_dbb(arr_nu_hz: Sequence[float] | npt.NDArray[np.floating], W: float, T: float) -> list[float]:
    """Calculate the spectral energy density of a dilute blackbody radiation field.

    Parameters
    ----------
    arr_nu_hz : list
        A list of frequencies (in Hz) at which to calculate the spectral energy density.
    W : float
        The dilution factor of the blackbody radiation field.
    T : float
        The temperature of the blackbody radiation field (in Kelvin).

    Returns
    -------
    list
        A list of spectral energy density values (in CGS units) corresponding to the input frequencies.

    """
    if W <= 0.0:
        return [0.0 for _ in arr_nu_hz]

    # hnu/kT above this overflows math.expm1, and the Wien tail there is far below any plotted value, so those
    # frequencies contribute zero. Catching OverflowError around the whole comprehension instead would discard
    # every frequency's value, not just the ones that overflowed
    max_exponent = math.log(sys.float_info.max)

    def j_nu(nu_hz: float) -> float:
        exponent = h_erg_s * nu_hz / T / K_B_erg_per_K
        if exponent >= max_exponent:
            return 0.0
        return W * 1.4745007e-47 * pow(nu_hz, 3) / math.expm1(exponent)

    # iterate Python floats, since math.expm1 on numpy scalars is much slower
    return [j_nu(nu_hz) for nu_hz in np.asarray(arr_nu_hz, dtype=float).tolist()]


def get_fullspecfittedfield(
    radfielddata: pl.DataFrame, xmin: float, xmax: float, modelgridindex: int | None = None, timestep: int | None = None
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the wavelengths and J_lambda of the full-spectrum dilute blackbody fit for one cell and timestep."""
    radfielddata = select_radfield_subset(radfielddata, pl.col("bin_num") == -1, modelgridindex, timestep)
    W = radfielddata.item(0, "W")
    assert isinstance(W, float)
    T_R = radfielddata.item(0, "T_R")
    assert isinstance(T_R, float)
    nu_lower = c_ang_per_s / xmin
    nu_upper = c_ang_per_s / xmax
    arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500, dtype=np.float64)
    arr_j_nu = j_nu_dbb(arr_nu_hz, W, T_R)

    arr_lambda = c_ang_per_s / arr_nu_hz
    arr_j_lambda = arr_j_nu * arr_nu_hz / arr_lambda

    return arr_lambda, arr_j_lambda


def get_fitted_field(
    radfielddata: pl.DataFrame, modelgridindex: int | None = None, timestep: int | None = None, usebinfits: bool = True
) -> tuple[list[float], list[float]]:
    """Return the radiation field model of ARTIS (list of lambda, list of J_lambda) made up of all bins.

    ARTIS radfield() takes the dilute blackbody fit of the bin. A bin with a negative W has no fit, and ARTIS
    then takes the full-spectrum dilute blackbody fit, which is bin -1 of the file. usebinfits=False gives every
    bin the full-spectrum fit, as ARTIS does before MULTIBIN_RADFIELD_FIRST_TIMESTEP.
    """
    arr_lambda: list[float] = []
    j_lambda_fitted: list[float] = []

    fullspecfit = select_radfield_subset(radfielddata, pl.col("bin_num") == -1, modelgridindex, timestep)
    W_fullspec, T_R_fullspec = float(fullspecfit.item(0, "W")), float(fullspecfit.item(0, "T_R"))

    radfielddata_subset = select_radfield_subset(radfielddata, pl.col("bin_num") >= 0, modelgridindex, timestep)

    for nu_lower, nu_upper, W_bin, T_R_bin in radfielddata_subset.select(
        "nu_lower", "nu_upper", "W", "T_R"
    ).iter_rows():
        W, T_R = (float(W_bin), float(T_R_bin)) if usebinfits and W_bin >= 0 else (W_fullspec, T_R_fullspec)
        arr_nu_hz_bin = np.linspace(nu_lower, nu_upper, num=200)
        arr_j_nu = j_nu_dbb(arr_nu_hz_bin, W, T_R)

        arr_lambda_bin = c_ang_per_s / arr_nu_hz_bin
        arr_j_lambda_bin = arr_j_nu * arr_nu_hz_bin / arr_lambda_bin

        arr_lambda += arr_lambda_bin.tolist()
        j_lambda_fitted += arr_j_lambda_bin.tolist()

    return arr_lambda, j_lambda_fitted


# ARTIS renamed this option two times on 2026-10-16 (commits d056232a and d135c5b3), thus an older model gives an older
# name. The newest name comes first
MULTIBIN_RADFIELD_FIRST_TIMESTEP_NAMES = (
    "MULTIBIN_RADFIELD_FIRST_TIMESTEP",
    "FIRST_BINNED_RADFIELD_TIMESTEP",
    "FIRST_NLTE_RADFIELD_TIMESTEP",
)


def get_multibin_radfield_first_timestep(modelpath: Path | str) -> tuple[int, str] | None:
    """Return the first timestep of the fits of the bins, and the name of its option in artisoptions.h, or None.

    ARTIS uses the fits of the bins from this timestep. Before it, ARTIS uses the full-spectrum fit for every bin.
    """
    for optionname in MULTIBIN_RADFIELD_FIRST_TIMESTEP_NAMES:
        value = get_artis_option(modelpath, optionname)
        if value is not None and value.isdigit():
            return int(value), optionname
    return None


def get_binedges(radfielddata: pl.DataFrame) -> list[float]:
    """Return the radiation field bin boundaries as wavelengths [Angstroms]."""
    radfielddata = radfielddata.filter(pl.col("bin_num") >= 0)
    return [c_ang_per_s / radfielddata["nu_lower"].item(0), *list(c_ang_per_s / radfielddata["nu_upper"])]
