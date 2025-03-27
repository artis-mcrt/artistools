"""Artistools - spectra related functions."""

import argparse
import contextlib
import math
import re
import typing as t
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import polars.selectors as cs

import artistools.packets as atpackets
from artistools.configuration import get_config
from artistools.misc import average_direction_bins
from artistools.misc import firstexisting
from artistools.misc import get_bflist
from artistools.misc import get_elsymbol
from artistools.misc import get_file_metadata
from artistools.misc import get_ionstring
from artistools.misc import get_linelist_pldf
from artistools.misc import get_nprocs
from artistools.misc import get_nu_grid
from artistools.misc import get_nuclides
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_costhetabincount
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import get_vpkt_config
from artistools.misc import split_multitable_dataframe
from artistools.misc import zopenpl


class FluxContributionTuple(t.NamedTuple):
    fluxcontrib: float
    linelabel: str
    array_flambda_emission: npt.NDArray[np.floating]
    array_flambda_absorption: npt.NDArray[np.floating]
    color: t.Any


megaparsec_to_cm = 3.0856e24


def timeshift_fluxscale_co56law(scaletoreftime: float | None, spectime: float) -> float:
    if scaletoreftime is not None:
        # Co56 decay flux scaling
        assert spectime > 150
        return math.exp(spectime / 113.7) / math.exp(scaletoreftime / 113.7)

    return 1.0


def get_dfspectrum_x_y_with_units(
    dfspectrum: pl.DataFrame, xunit: str, yvariable: str, fluxdistance_mpc: float
) -> pl.DataFrame:
    h = 4.1356677e-15  # Planck's constant [eV s]

    if "nu" not in dfspectrum.columns:
        dfspectrum = dfspectrum.with_columns((299792458.0 / (pl.col("lambda_angstroms") * 1e-10)).alias("nu"))
    if "f_nu" not in dfspectrum.columns:
        dfspectrum = dfspectrum.with_columns(f_nu=(pl.col("f_lambda") * pl.col("lambda_angstroms") / pl.col("nu")))

    match xunit.lower():
        case "angstroms":
            dfspectrum = dfspectrum.with_columns(x=pl.col("lambda_angstroms"), yflux=pl.col("f_lambda"))

        case "nm":
            dfspectrum = dfspectrum.with_columns(x=pl.col("lambda_angstroms") / 10, yflux=pl.col("f_lambda") * 10)

        case "micron":
            dfspectrum = dfspectrum.with_columns(x=pl.col("lambda_angstroms") / 10000, yflux=pl.col("f_lambda") * 10000)

        case "hz":
            dfspectrum = dfspectrum.with_columns(x=pl.col("nu"), yflux=pl.col("f_nu"))

        case "ev":
            dfspectrum = (
                dfspectrum.with_columns(en_ev=h * pl.col("nu"))
                .with_columns(f_en_kev=pl.col("f_nu") * pl.col("nu") / pl.col("en_ev"))
                .with_columns(x=pl.col("en_ev"), yflux=pl.col("f_en_kev"))
            )

        case "kev":
            dfspectrum = (
                dfspectrum.with_columns(en_kev=h * pl.col("nu") / 1000.0)
                .with_columns(f_en_kev=pl.col("f_nu") * pl.col("nu") / pl.col("en_kev"))
                .with_columns(x=pl.col("en_kev"), yflux=pl.col("f_en_kev"))
            )

        case "mev":
            dfspectrum = (
                dfspectrum.with_columns(en_mev=h * pl.col("nu") / 1e6)
                .with_columns(f_en_mev=pl.col("f_nu") * pl.col("nu") / pl.col("en_mev"))
                .with_columns(x=pl.col("en_mev"), yflux=pl.col("f_en_mev"))
            )

        case _:
            msg = f"Unit {xunit} not implemented"
            raise NotImplementedError(msg)

    match yvariable.lower():
        case "flux":
            assert yvariable == "flux"
            dfspectrum = dfspectrum.with_columns(y=pl.col("yflux") / fluxdistance_mpc**2)
        case "eflux":
            erg_to_angstrom = 1.986454e-8
            erg_to_xunit = convert_angstroms_to_unit(erg_to_angstrom, xunit.lower())
            dfspectrum = dfspectrum.with_columns(y=(pl.col("yflux") / fluxdistance_mpc**2 * erg_to_xunit) * pl.col("x"))
        case "photoncount":
            ev_to_erg = 1.60218e-12
            dfspectrum = dfspectrum.with_columns(
                y=pl.col("yflux") / fluxdistance_mpc**2 / (h * pl.col("nu") * ev_to_erg)
            )
        case "packetcount":
            dfspectrum = dfspectrum.with_columns(y=pl.col("packetcount"))
        case _:
            msg = f"Unit {yvariable} not implemented"
            raise NotImplementedError(msg)

    return dfspectrum.sort("x")


def get_exspec_bins(
    modelpath: str | Path | None = None,
    mnubins: int | None = None,
    nu_min_r: float | None = None,
    nu_max_r: float | None = None,
    gamma: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Get the wavelength bins for the emergent spectrum."""
    if modelpath is not None:
        dfspec = read_spec(modelpath, gamma=gamma)
        if mnubins is None:
            mnubins = dfspec.height

        nu_centre_min = dfspec.item(0, 0)
        nu_centre_max = dfspec.item(dfspec.height - 1, 0)

        # This is not an exact solution for dlognu since we're assuming the bin centre spacing matches the bin edge spacing
        # but it's close enough for our purposes and avoids the difficulty of finding the exact solution (lots more algebra)
        dlognu = math.log(dfspec.item(1, 0) / dfspec.item(0, 0))  # second nu value divided by the first nu value

        if nu_min_r is None:
            nu_min_r = nu_centre_min / (1 + 0.5 * dlognu)

        if nu_max_r is None:
            nu_max_r = nu_centre_max * (1 + 0.5 * dlognu)

    assert nu_min_r is not None
    assert nu_max_r is not None
    assert mnubins is not None

    c_ang_s = 2.99792458e18

    dlognu = (math.log(nu_max_r) - math.log(nu_min_r)) / mnubins

    bins_nu_lower = np.array([math.exp(math.log(nu_min_r) + (m * (dlognu))) for m in range(mnubins)])
    # bins_nu_upper = np.array([math.exp(math.log(nu_min_r) + ((m + 1) * (dlognu))) for m in range(mnubins)])
    bins_nu_upper = bins_nu_lower * math.exp(dlognu)
    bins_nu_centre = 0.5 * (bins_nu_lower + bins_nu_upper)

    # np.flip is used to get an ascending wavelength array from an ascending nu array
    lambda_bin_edges = np.append(c_ang_s / np.flip(bins_nu_upper), c_ang_s / bins_nu_lower[0])
    lambda_bin_centres = c_ang_s / np.flip(bins_nu_centre)
    delta_lambdas = np.flip(c_ang_s / bins_nu_lower - c_ang_s / bins_nu_upper)

    return lambda_bin_edges, lambda_bin_centres, delta_lambdas


def convert_xunit_aliases_to_canonical(xunit: str) -> str:
    match xunit.lower():
        case "ev" | "electronvolt":
            return "ev"
        case "kev" | "kiloelectronvolt":
            return "kev"
        case "mev" | "megaelectronvolt":
            return "mev"
        case "angstroms" | "angstrom" | "a" | "ang" | "å" | "ångström":
            return "angstroms"
        case "nm" | "nanometer" | "nanometers":
            return "nm"
        case "micron" | "microns" | "mu" | "μ" | "μm":
            return "micron"
        case "hz":
            return "hz"
        case _:
            msg = f"Unknown xunit {xunit}"
            raise ValueError(msg)


def convert_angstroms_to_unit(value_angstroms: float, new_units: str) -> float:
    """Convert a wavelength in angstroms to a different unit, either length, frequency, or energy."""
    c = 2.99792458e18  # speed of light [angstroms/s]
    h = 4.1356677e-15  # Planck's constant [eV s]
    hc_ev_angstroms = h * c  # [eV angstroms]
    match new_units.lower():
        case "ev":
            return hc_ev_angstroms / value_angstroms
        case "kev":
            return hc_ev_angstroms / value_angstroms / 1e3
        case "mev":
            return hc_ev_angstroms / value_angstroms / 1e6
        case "hz":
            return c / value_angstroms
        case "angstroms":
            return value_angstroms
        case "nm":
            return value_angstroms / 10
        case "micron":
            return value_angstroms / 10000
    msg = f"Unknown xunit {new_units}"
    raise ValueError(msg)


def convert_unit_to_angstroms(value: float, old_units: str) -> float:
    """Convert a wavelength, frequency, or energy to wavelength angstroms."""
    c = 2.99792458e18  # speed of light [angstroms/s]
    h = 4.1356677e-15  # Planck's constant [eV s]
    hc_ev_angstroms = h * c  # [eV angstroms]

    match old_units.lower():
        case "ev":
            return hc_ev_angstroms / value
        case "kev":
            return hc_ev_angstroms / value / 1e3
        case "mev":
            return hc_ev_angstroms / value / 1e6
        case "hz":
            return c / value
        case "angstroms":
            return value
        case "nm":
            return value * 10
    if old_units.lower() == "micron":
        return value * 10000

    msg = f"Unknown xunit {old_units}"
    raise ValueError(msg)


def stackspectra(
    spectra_and_factors: list[tuple[np.ndarray[t.Any, np.dtype[np.floating[t.Any]]], float]],
) -> np.ndarray[t.Any, np.dtype[np.floating[t.Any]]]:
    """Add spectra using weighting factors, i.e., specout[nu] = spec1[nu] * factor1 + spec2[nu] * factor2 + ...

    spectra_and_factors should be a list of tuples: spectra[], factor.
    """
    factor_sum = sum(factor for _, factor in spectra_and_factors)

    stackedspectrum = np.zeros_like(spectra_and_factors[0][0], dtype=float)
    for spectrum, factor in spectra_and_factors:
        stackedspectrum += spectrum * factor / factor_sum

    return stackedspectrum


def get_spectrum_at_time(
    modelpath: Path,
    timestep: int,
    time: float,
    args: argparse.Namespace | None,
    dirbin: int = -1,
    average_over_phi: bool | None = None,
    average_over_theta: bool | None = None,
) -> pd.DataFrame:
    if dirbin >= 0:
        if args is not None and args.plotvspecpol and (modelpath / "vpkt.txt").is_file():
            return get_vspecpol_spectrum(modelpath, time, dirbin, args).to_pandas(use_pyarrow_extension_array=True)
        assert average_over_phi is not None
        assert average_over_theta is not None
    else:
        average_over_phi = False
        average_over_theta = False

    return get_spectrum(
        modelpath=modelpath,
        directionbins=[dirbin],
        timestepmin=timestep,
        timestepmax=timestep,
        average_over_phi=average_over_phi,
        average_over_theta=average_over_theta,
    )[dirbin].to_pandas(use_pyarrow_extension_array=True)


def get_from_packets(
    modelpath: Path | str,
    timelowdays: float,
    timehighdays: float,
    lambda_min: float,
    lambda_max: float,
    delta_lambda: float | npt.NDArray[np.floating] | None = None,
    use_time: t.Literal["arrival", "emission", "escape"] = "arrival",
    maxpacketfiles: int | None = None,
    directionbins: Collection[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    nu_column: str = "nu_rf",
    fluxfilterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    nprocs_read_dfpackets: tuple[int, pl.DataFrame | pl.LazyFrame] | None = None,
    directionbins_are_vpkt_observers: bool = False,
    gamma: bool = False,
) -> dict[int, pl.DataFrame]:
    """Get a spectrum dataframe using the packets files as input."""
    if directionbins is None:
        directionbins = [-1]

    assert use_time in {"arrival", "emission", "escape"}

    if nu_column == "absorption_freq":
        nu_column = "nu_absorbed"

    lambda_bin_edges: npt.NDArray[np.floating]
    pl_delta_lambda: pl.Series | pl.Expr
    if delta_lambda is None:
        lambda_bin_edges, lambda_bin_centres, delta_lambda = get_exspec_bins(modelpath=modelpath, gamma=gamma)
        lambda_min = lambda_bin_centres[0]
        lambda_max = lambda_bin_centres[-1]
        pl_delta_lambda = pl.Series(delta_lambda)
    elif isinstance(delta_lambda, float | int):
        lambda_bin_edges = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
        lambda_bin_centres = 0.5 * (lambda_bin_edges[:-1] + lambda_bin_edges[1:])  # bin centres
        pl_delta_lambda = pl.lit(delta_lambda)
    elif isinstance(delta_lambda, np.ndarray):
        lambda_bin_edges = np.array([lambda_min + (delta_lambda[:i]).sum() for i in range(len(delta_lambda) + 1)])
        lambda_bin_centres = 0.5 * (lambda_bin_edges[:-1] + lambda_bin_edges[1:])
        pl_delta_lambda = pl.Series(delta_lambda)
    else:
        msg = f"Invalid delta_lambda type: {type(delta_lambda)}"
        raise ValueError(msg)

    delta_time_s = (timehighdays - timelowdays) * 86400.0

    nphibins = get_viewingdirection_phibincount()
    ncosthetabins = get_viewingdirection_costhetabincount()
    ndirbins = get_viewingdirectionbincount()

    if nprocs_read_dfpackets:
        nprocs_read, dfpackets = nprocs_read_dfpackets[0], nprocs_read_dfpackets[1].lazy()
    elif directionbins_are_vpkt_observers:
        assert not gamma
        nprocs_read, dfpackets = atpackets.get_virtual_packets_pl(modelpath, maxpacketfiles=maxpacketfiles)
    else:
        nprocs_read, dfpackets = atpackets.get_packets_pl(
            modelpath,
            maxpacketfiles=maxpacketfiles,
            packet_type="TYPE_ESCAPE",
            escape_type="TYPE_GAMMA" if gamma else "TYPE_RPKT",
        )

    dfpackets = dfpackets.with_columns([
        (2.99792458e18 / pl.col(colname)).alias(
            colname.replace("absorption_freq", "nu_absorbed").replace("nu_", "lambda_angstroms_")
        )
        for colname in dfpackets.collect_schema().names()
        if "nu_" in colname or colname == "absorption_freq"
    ])

    dfbinned_lazy = (
        pl.DataFrame(
            {"lambda_angstroms": lambda_bin_centres, "lambda_binindex": range(len(lambda_bin_centres))},
            schema_overrides={"lambda_binindex": pl.Int32},
        )
        .sort(["lambda_binindex", "lambda_angstroms"])
        .lazy()
    )
    escapesurfacegamma: float | None = None
    if directionbins_are_vpkt_observers:
        vpkt_config = get_vpkt_config(modelpath)
        for vspecindex in directionbins:
            obsdirindex = vspecindex // vpkt_config["nspectraperobs"]
            opacchoiceindex = vspecindex % vpkt_config["nspectraperobs"]
            lambda_column = (
                f"dir{obsdirindex}_lambda_angstroms_rf"
                if nu_column == "nu_rf"
                else nu_column.replace("absorption_freq", "nu_absorbed").replace("nu_", "lambda_angstroms_")
            )
            energy_column = f"dir{obsdirindex}_e_rf_{opacchoiceindex}"

            pldfpackets_dirbin_lazy = dfpackets.filter(pl.col(lambda_column).is_between(lambda_min, lambda_max)).filter(
                pl.col(f"dir{obsdirindex}_t_arrive_d").is_between(timelowdays, timehighdays)
            )

            dfbinned_dirbin = atpackets.bin_and_sum(
                pldfpackets_dirbin_lazy,
                bincol=lambda_column,
                bins=list(lambda_bin_edges),
                sumcols=[energy_column],
                getcounts=True,
            ).select([
                pl.col(f"{lambda_column}_bin").alias("lambda_binindex"),
                (
                    pl.col(f"{energy_column}_sum")
                    / pl_delta_lambda
                    / delta_time_s
                    / (megaparsec_to_cm**2)
                    / nprocs_read
                ).alias(f"f_lambda_dirbin{vspecindex}"),
                pl.col("count").alias(f"count_dirbin{vspecindex}"),
            ])

            dfbinned_lazy = dfbinned_lazy.join(dfbinned_dirbin, on="lambda_binindex", how="left", coalesce=True)

        assert use_time == "arrival"
    else:
        lambda_column = nu_column.replace("nu_", "lambda_angstroms_")
        energy_column = "e_cmf" if use_time == "escape" else "e_rf"

        if use_time == "arrival":
            dfpackets = dfpackets.filter(pl.col("t_arrive_d").is_between(timelowdays, timehighdays))

        elif use_time == "escape":
            from artistools.inputmodel import get_modeldata

            dfmodel, _ = get_modeldata(modelpath)
            vmax_beta = dfmodel.select(pl.col("vel_r_max_kmps").max() * 299792.458).collect().item()
            escapesurfacegamma = math.sqrt(1 - vmax_beta**2)

            dfpackets = dfpackets.filter(
                (pl.col("escape_time") * escapesurfacegamma / 86400.0).is_between(timelowdays, timehighdays)
            )

        elif use_time == "emission":
            # We bin packets according to the emission time, but we shift times so we're still centered around the observer arrival time.
            # This makes easier to directly compare specta between emission time (no relative light travel time effects) and the standard arrival time
            col_emit_time = "tdecay" if gamma else "em_time"
            mean_correction = (pl.col(col_emit_time) - pl.col("t_arrive_d") * 86400.0).mean()

            dfpackets = dfpackets.filter(
                pl.col(col_emit_time).is_between(
                    timelowdays * 86400.0 + mean_correction, timehighdays * 86400.0 + mean_correction
                )
            )

        dfpackets = dfpackets.filter(pl.col(lambda_column).is_between(lambda_min, lambda_max))

        for dirbin in directionbins:
            if dirbin == -1:
                solidanglefactor = 1.0
                pldfpackets_dirbin_lazy = dfpackets
            elif average_over_phi:
                assert not average_over_theta
                solidanglefactor = ncosthetabins
                pldfpackets_dirbin_lazy = dfpackets.filter(pl.col("costhetabin") * nphibins == dirbin)
            elif average_over_theta:
                solidanglefactor = nphibins
                pldfpackets_dirbin_lazy = dfpackets.filter(pl.col("phibin") == dirbin)
            else:
                solidanglefactor = ndirbins
                pldfpackets_dirbin_lazy = dfpackets.filter(pl.col("dirbin") == dirbin)

            dfbinned_dirbin = atpackets.bin_and_sum(
                pldfpackets_dirbin_lazy,
                bincol=lambda_column,
                bins=list(lambda_bin_edges),
                sumcols=[energy_column],
                getcounts=True,
            ).select([
                pl.col(f"{lambda_column}_bin").alias("lambda_binindex"),
                (
                    pl.col(f"{energy_column}_sum")
                    / pl_delta_lambda
                    / delta_time_s
                    / (4 * math.pi)
                    * solidanglefactor
                    / (megaparsec_to_cm**2)
                    / nprocs_read
                ).alias(f"f_lambda_dirbin{dirbin}"),
                pl.col("count").alias(f"count_dirbin{dirbin}"),
            ])

            if use_time == "escape":
                assert escapesurfacegamma is not None
                dfbinned_dirbin = dfbinned_dirbin.with_columns(
                    pl.col(f"f_lambda_dirbin{dirbin}").mul(1.0 / escapesurfacegamma)
                )

            dfbinned_lazy = dfbinned_lazy.join(dfbinned_dirbin, on="lambda_binindex", how="left", coalesce=True)

    if fluxfilterfunc:
        print("Applying filter to ARTIS spectrum")
        dfbinned_lazy = dfbinned_lazy.with_columns(cs.starts_with("f_lambda_").map_batches(fluxfilterfunc))

    dfbinned = dfbinned_lazy.collect()
    assert isinstance(dfbinned, pl.DataFrame)

    dfdict = {}
    for dirbin in directionbins:
        dfdict[dirbin] = dfbinned.select([
            "lambda_angstroms",
            pl.col(f"f_lambda_dirbin{dirbin}").alias("f_lambda"),
            pl.col(f"count_dirbin{dirbin}").alias("packetcount"),
            (299792458.0 / (pl.col("lambda_angstroms") * 1e-10)).alias("nu"),
        ]).with_columns(f_nu=(pl.col("f_lambda") * pl.col("lambda_angstroms") / pl.col("nu")))

        if nprocs_read_dfpackets is None:
            npkts_selected = dfdict[dirbin].get_column("packetcount").sum()
            print(f"    dirbin {dirbin:2d} plots {npkts_selected:.2e} packets")

    return dfdict


@lru_cache(maxsize=16)
def read_spec(modelpath: Path | str, gamma: bool = False) -> pl.DataFrame:
    specfilename = firstexisting("gamma_spec.out" if gamma else "spec.out", folder=modelpath, tryzipped=True)
    print(f"Reading {specfilename}")

    return (
        pl.read_csv(zopenpl(specfilename), separator=" ", infer_schema=False, truncate_ragged_lines=True)
        .with_columns(pl.all().cast(pl.Float64))
        .rename({"0": "nu"})
    )


@lru_cache(maxsize=16)
def read_spec_res(modelpath: Path | str) -> dict[int, pl.DataFrame]:
    """Return a dataframe of time-series spectra for every viewing direction."""
    specfilename = (
        modelpath
        if Path(modelpath).is_file()
        else firstexisting(["spec_res.out", "specpol_res.out"], folder=modelpath, tryzipped=True)
    )

    print(f"Reading {specfilename} (in read_spec_res)")
    res_specdata_in = pl.read_csv(zopenpl(specfilename), separator=" ", has_header=False, infer_schema=False)

    # drop last column of nulls (caused by trailing space on each line)
    if res_specdata_in[res_specdata_in.columns[-1]].is_null().all():
        res_specdata_in = res_specdata_in.drop(res_specdata_in.columns[-1])

    res_specdata = split_multitable_dataframe(res_specdata_in)

    prev_dfshape = None
    for dirbin in res_specdata:
        assert isinstance(res_specdata[dirbin], pl.DataFrame)
        newcolnames = [str(x) for x in res_specdata[dirbin][0, :].to_numpy()[0]]
        newcolnames[0] = "nu"

        newcolnames_unique = set(newcolnames)
        oldcolnames = res_specdata[dirbin].columns
        if len(newcolnames) > len(newcolnames_unique):
            # for POL_ON, the time columns repeat for Q, U, and V stokes params.
            # here, we keep the first set (I) and drop the rest of the columns
            assert len(newcolnames) % len(newcolnames_unique) == 0  # must be an exact multiple
            newcolnames = newcolnames[: len(newcolnames_unique)]
            oldcolnames = oldcolnames[: len(newcolnames_unique)]
            res_specdata[dirbin] = res_specdata[dirbin].select(oldcolnames)

        res_specdata[dirbin] = (
            res_specdata[dirbin][1:]  # drop the first row that contains time headers
            .with_columns(pl.all().cast(pl.Float64))
            .rename(dict(zip(oldcolnames, newcolnames, strict=False)))
        )

        # the number of timesteps and nu bins should match for all direction bins
        assert prev_dfshape is None or prev_dfshape == res_specdata[dirbin].shape
        prev_dfshape = res_specdata[dirbin].shape

    return res_specdata


@lru_cache(maxsize=200)
def read_emission_absorption_file(emabsfilename: str | Path) -> pl.DataFrame:
    """Read into a DataFrame one of: emission.out. emissionpol.out, emissiontrue.out, absorption.out."""
    try:
        emissionfilesize = Path(emabsfilename).stat().st_size / 1024 / 1024
        print(f" Reading {emabsfilename} ({emissionfilesize:.2f} MiB)")

    except AttributeError:
        print(f" Reading {emabsfilename}")

    dfemabs = pl.read_csv(zopenpl(emabsfilename), separator=" ", has_header=False, infer_schema_length=0).with_columns(
        pl.all().cast(pl.Float32, strict=False)
    )

    # drop last column of nulls (caused by trailing space on each line)
    if dfemabs[dfemabs.columns[-1]].is_null().all():
        dfemabs = dfemabs.drop(dfemabs.columns[-1])

    return dfemabs


@lru_cache(maxsize=4)
def get_spec_res(
    modelpath: Path, average_over_theta: bool = False, average_over_phi: bool = False
) -> dict[int, pl.DataFrame]:
    res_specdata = read_spec_res(modelpath)
    if average_over_theta:
        res_specdata = average_direction_bins(res_specdata, overangle="theta")
    if average_over_phi:
        res_specdata = average_direction_bins(res_specdata, overangle="phi")

    return res_specdata


def get_spectrum(
    modelpath: Path,
    timestepmin: int,
    timestepmax: int | None = None,
    directionbins: Sequence[int] | None = None,
    fluxfilterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    average_over_theta: bool = False,
    average_over_phi: bool = False,
    stokesparam: t.Literal["I", "Q", "U"] = "I",
    gamma: bool = False,
) -> dict[int, pl.DataFrame]:
    """Get a polars DataFrame containing an ARTIS emergent spectrum."""
    if timestepmax is None or timestepmax < 0:
        timestepmax = timestepmin

    if directionbins is None:
        directionbins = [-1]

    # keys are direction bins (or -1 for spherical average)
    specdata: dict[int, pl.DataFrame] = {}

    if any(dirbin != -1 for dirbin in directionbins):
        assert stokesparam == "I"
        try:
            specdata |= get_spec_res(
                modelpath=modelpath, average_over_theta=average_over_theta, average_over_phi=average_over_phi
            )
        except FileNotFoundError:
            msg = "WARNING: Direction-resolved spectra not found. Getting only spherically averaged spectra instead."
            print(msg)
            directionbins = [-1]

    if -1 in directionbins:
        # spherically averaged spectra
        if stokesparam == "I":
            try:
                specdata[-1] = read_spec(modelpath=modelpath, gamma=gamma)

            except FileNotFoundError:
                specdata[-1] = get_specpol_data(angle=-1, modelpath=modelpath)[stokesparam]

        else:
            specdata[-1] = get_specpol_data(angle=-1, modelpath=modelpath)[stokesparam]

    specdataout: dict[int, pl.DataFrame] = {}
    for dirbin in directionbins:
        if dirbin not in specdata:
            print(f"WARNING: Direction bin {dirbin} not found in specdata. Dirbins: {list(specdata.keys())}")
            continue
        arr_tdelta = get_timestep_times(modelpath, loc="delta")

        try:
            arr_f_nu = specdata[dirbin].select(
                pl.sum_horizontal(
                    pl.col(specdata[dirbin].columns[timestep + 1]) * arr_tdelta[timestep]
                    for timestep in range(timestepmin, timestepmax + 1)
                )
                / sum(arr_tdelta[timestepmin : timestepmax + 1])
            )
        except IndexError as e:
            msg = " ERROR: data not available for timestep range"
            raise ValueError(msg) from e

        dfspectrum = (
            pl.DataFrame({"nu": specdata[dirbin]["nu"], "f_nu": arr_f_nu})
            .lazy()
            .with_columns(lambda_angstroms=2.99792458e18 / pl.col("nu"))
        )

        if fluxfilterfunc:
            if dirbin == directionbins[0]:
                print("Applying filter to ARTIS spectrum")
            dfspectrum = dfspectrum.with_columns(cs.starts_with("f_nu").map_batches(fluxfilterfunc))

        dfspectrum = dfspectrum.with_columns(f_lambda=pl.col("f_nu") * pl.col("nu") / pl.col("lambda_angstroms")).sort(
            by="nu" if gamma else "lambda_angstroms"
        )

        specdataout[dirbin] = dfspectrum.collect()

    return specdataout


def make_virtual_spectra_summed_file(modelpath: Path | str) -> None:
    nprocs = get_nprocs(modelpath)
    print("nprocs", nprocs)
    # virtual packet spectra for each observer (all directions and opacity choices)
    vspecpol_data_allranks: dict[int, pl.DataFrame] = {}
    vpktconfig = get_vpkt_config(modelpath)
    nvirtual_spectra = vpktconfig["nobsdirections"] * vpktconfig["nspectraperobs"]
    print(
        f"nobsdirections {vpktconfig['nobsdirections']} nspectraperobs {vpktconfig['nspectraperobs']} (total observers:"
        f" {nvirtual_spectra})"
    )
    vspecpol_data = None
    for mpirank in range(nprocs):
        vspecpolpath = firstexisting(
            [f"vspecpol_{mpirank:04d}.out", f"vspecpol_{mpirank}-0.out"], folder=modelpath, tryzipped=True
        )
        print(f"Reading rank {mpirank} filename {vspecpolpath}")

        vspecpol_data_alldirs = pl.read_csv(vspecpolpath, separator=" ", has_header=False)

        if vspecpol_data_alldirs[vspecpol_data_alldirs.columns[-1]].is_null().all():
            vspecpol_data_alldirs = vspecpol_data_alldirs.drop(cs.last())

        vspecpol_data = split_multitable_dataframe(vspecpol_data_alldirs)
        assert len(vspecpol_data) == nvirtual_spectra

        for specindex in vspecpol_data:
            if specindex not in vspecpol_data_allranks:
                vspecpol_data_allranks[specindex] = vspecpol_data[specindex]
            else:
                vspecpol_data_allranks[specindex] = vspecpol_data_allranks[specindex].with_columns([
                    (pl.col(col) + vspecpol_data[specindex].get_column(col)).alias(col)
                    for col in vspecpol_data_allranks[specindex].columns[1:]
                ])

    assert vspecpol_data is not None
    for spec_index, vspecpol in vspecpol_data_allranks.items():
        # fix the header row, which got summed along with the data
        dfvspecpol = pl.concat([vspecpol_data[spec_index][0], vspecpol[1:]])

        outfile = Path(modelpath, f"vspecpol_total-{spec_index}.out")
        dfvspecpol.write_csv(outfile, separator=" ", include_header=False)
        print(f"Saved {outfile}")


def make_averaged_vspecfiles(args: argparse.Namespace) -> None:
    filenames = [
        vspecfile.name
        for vspecfile in Path(args.modelpath[0]).iterdir()
        if vspecfile.name.startswith("vspecpol_total-")
    ]

    def sorted_by_number(lst: list[str]) -> list[str]:
        def convert(text: str) -> int | str:
            return int(text) if text.isdigit() else text

        def alphanum_key(key: str) -> list[int | str]:
            return [convert(c) for c in re.split(r"([0-9]+)", key)]

        return sorted(lst, key=alphanum_key)

    filenames = sorted_by_number(filenames)

    for spec_index, filename in enumerate(filenames):  # vspecpol-total files
        vspecdata = [pd.read_csv(modelpath / filename, sep=r"\s+", header=None) for modelpath in args.modelpath]
        for i in range(1, len(vspecdata)):
            vspecdata[0].iloc[1:, 1:] += vspecdata[i].iloc[1:, 1:]

        vspecdata[0].iloc[1:, 1:] /= len(vspecdata)
        vspecdata[0].to_csv(
            args.modelpath[0] / f"vspecpol_averaged-{spec_index}.out", sep=" ", index=False, header=False
        )


@lru_cache(maxsize=4)
def get_specpol_data(
    angle: int = -1, modelpath: Path | None = None, specdata: pl.DataFrame | None = None
) -> dict[str, pl.DataFrame]:
    if specdata is None:
        assert modelpath is not None
        specfilename = (
            firstexisting("specpol.out", folder=modelpath, tryzipped=True)
            if angle == -1
            else firstexisting(f"specpol_res_{angle}.out", folder=modelpath, tryzipped=True)
        )

        print(f"Reading {specfilename}")
        specdata = pl.read_csv(zopenpl(specfilename), separator=" ", has_header=True, infer_schema=False).with_columns(
            pl.all().cast(pl.Float64)
        )

    return split_dataframe_stokesparams(specdata)


@lru_cache(maxsize=4)
def get_vspecpol_data(vspecindex: int, modelpath: Path | str) -> dict[str, pl.DataFrame]:
    assert modelpath is not None
    # alternatively use f'vspecpol_averaged-{angle}.out' ?

    try:
        specfilename = firstexisting(f"vspecpol_total-{vspecindex}.out", folder=modelpath, tryzipped=True)
    except FileNotFoundError:
        print(f"vspecpol_total-{vspecindex}.out does not exist. Generating all-rank summed vspec files..")
        make_virtual_spectra_summed_file(modelpath=modelpath)
        specfilename = firstexisting(f"vspecpol_total-{vspecindex}.out", folder=modelpath, tryzipped=True)

    print(f"Reading {specfilename}")
    specdata = pl.read_csv(specfilename, separator=" ", has_header=True)

    return split_dataframe_stokesparams(specdata)


def split_dataframe_stokesparams(specdata: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """DataFrames read from specpol*.out and vspecpol*.out are repeated over I, Q, U parameters. Split these into a dictionary of DataFrames."""
    specdata = specdata.rename({specdata.columns[0]: "nu"})
    stokes_params = {
        "I": specdata.select(cs.exclude(cs.contains("_duplicated_"))),
        "Q": specdata.select(
            pl.col("nu"), cs.ends_with("_duplicated_0").name.map(lambda x: x.removesuffix("_duplicated_0"))
        ),
        "U": specdata.select(
            pl.col("nu"), cs.ends_with("_duplicated_1").name.map(lambda x: x.removesuffix("_duplicated_1"))
        ),
    }

    for param in ("Q", "U"):
        stokes_params[param + "/I"] = stokes_params[param].select(
            pl.col("nu"),
            *(pl.col(colname) / stokes_params["I"].get_column(colname) for colname in stokes_params[param].columns[1:]),
        )

    return stokes_params


def get_vspecpol_spectrum(
    modelpath: Path | str,
    timeavg: float,
    angle: int,
    args: argparse.Namespace,
    fluxfilterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
) -> pl.DataFrame:
    stokes_params = get_vspecpol_data(vspecindex=angle, modelpath=Path(modelpath))
    if "stokesparam" not in args:
        args.stokesparam = "I"
    vspecdata = stokes_params[args.stokesparam]

    arr_tmid = [float(i) for i in vspecdata.columns[1:]]
    vspec_timesteps = range(len(arr_tmid))
    arr_tdelta = [l1 - l2 for l1, l2 in zip(arr_tmid[1:], arr_tmid[:-1], strict=False)] + [arr_tmid[-1] - arr_tmid[-2]]

    def match_closest_time(reftime: float) -> int:
        return min(vspec_timesteps, key=lambda ts: abs(arr_tmid[ts] - reftime))

    if "timemin" and "timemax" in args:
        timestepmin = match_closest_time(args.timemin)  # how timemin, timemax are used changed at some point
        timestepmax = match_closest_time(args.timemax)  # to average over multiple timesteps needs to fix this
    else:
        timestepmin = match_closest_time(timeavg)
        timestepmax = match_closest_time(timeavg)

    timelower = arr_tmid[timestepmin]
    timeupper = arr_tmid[timestepmax]
    print(f" vpacket spectrum timesteps {timestepmin} ({timelower}d) to {timestepmax} ({timeupper}d)")

    dfout = vspecdata.select(
        f_nu=(
            pl.sum_horizontal(
                pl.col(vspecdata.columns[timestep + 1]) * arr_tdelta[timestep]
                for timestep in range(timestepmin, timestepmax + 1)
            )
            / sum(arr_tdelta[timestepmin : timestepmax + 1])
        ),
        nu=vspecdata["nu"],
    ).with_columns(lambda_angstroms=2.99792458e18 / pl.col("nu"))

    if fluxfilterfunc:
        print("Applying filter to ARTIS spectrum")
        dfout = dfout.with_columns(cs.starts_with("f_nu").map_batches(fluxfilterfunc))

    return dfout.with_columns(f_lambda=pl.col("f_nu") * pl.col("nu") / pl.col("lambda_angstroms")).sort(
        by="lambda_angstroms"
    )


@lru_cache(maxsize=4)
def get_flux_contributions(
    modelpath: Path,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    timestepmin: int = -1,
    timestepmax: int = -1,
    getemission: bool = True,
    getabsorption: bool = True,
    use_lastemissiontype: bool = True,
    directionbin: int | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> tuple[list[FluxContributionTuple], npt.NDArray[np.floating[t.Any]]]:
    from scipy import integrate

    arr_tmid = get_timestep_times(modelpath, loc="mid")
    arr_tdelta = get_timestep_times(modelpath, loc="delta")
    arraynu = get_nu_grid(modelpath)
    arraylambda = 2.99792458e18 / arraynu
    if not Path(modelpath, "compositiondata.txt").is_file():
        print("WARNING: compositiondata.txt not found. Using output*.txt instead")
        from artistools.misc import get_composition_data_from_outputfile

        elementlist = get_composition_data_from_outputfile(modelpath)
    else:
        from artistools.misc import get_composition_data

        elementlist = get_composition_data(modelpath)
    nelements = len(elementlist)

    if directionbin is None:
        dbinlist = [-1]
    elif average_over_phi:
        assert not average_over_theta
        assert directionbin % get_viewingdirection_phibincount() == 0
        dbinlist = list(range(directionbin, directionbin + get_viewingdirection_phibincount()))
    elif average_over_theta:
        assert not average_over_phi
        assert directionbin < get_viewingdirection_phibincount()
        dbinlist = list(range(directionbin, get_viewingdirectionbincount(), get_viewingdirection_phibincount()))
    else:
        dbinlist = [directionbin]

    emissiondata = {}
    absorptiondata = {}
    maxion: int | None = None
    for dbin in dbinlist:
        if getemission:
            emissionfilenames = ["emission.out", "emissionpol.out"] if use_lastemissiontype else ["emissiontrue.out"]

            if dbin != -1:
                emissionfilenames = [x.replace(".out", f"_res_{dbin:02d}.out") for x in emissionfilenames]

            emissionfilename = firstexisting(emissionfilenames, folder=modelpath, tryzipped=True)

            if "pol" in str(emissionfilename):
                print("This artis run contains polarisation data")
                # File contains I, Q and U and so times are repeated 3 times
                arr_tmid = list(np.tile(np.array(arr_tmid), 3))

            emissiondata[dbin] = read_emission_absorption_file(emissionfilename)

            maxion_float = (emissiondata[dbin].shape[1] - 1) / 2 / nelements  # also known as MIONS in ARTIS sn3d.h
            assert maxion_float.is_integer()
            if maxion is None:
                maxion = int(maxion_float)
                print(
                    f" inferred MAXION = {maxion} from emission file using nlements = {nelements} from"
                    " compositiondata.txt"
                )
            else:
                assert maxion == int(maxion_float)

            # check that the row count is product of timesteps and frequency bins found in spec.out
            assert emissiondata[dbin].shape[0] == len(arraynu) * len(arr_tmid)

        if getabsorption:
            absorptionfilenames = ["absorption.out", "absorptionpol.out"]
            if directionbin is not None:
                absorptionfilenames = [x.replace(".out", f"_res_{dbin:02d}.out") for x in absorptionfilenames]

            absorptionfilename = firstexisting(absorptionfilenames, folder=modelpath, tryzipped=True)

            absorptiondata[dbin] = read_emission_absorption_file(absorptionfilename)
            absorption_maxion_float = absorptiondata[dbin].shape[1] / nelements
            assert absorption_maxion_float.is_integer()
            absorption_maxion = int(absorption_maxion_float)
            if maxion is None:
                maxion = absorption_maxion
                print(
                    f" inferred MAXION = {maxion} from absorption file using nlements = {nelements}from"
                    " compositiondata.txt"
                )
            else:
                assert absorption_maxion == maxion
            assert absorptiondata[dbin].shape[0] == len(arraynu) * len(arr_tmid)

    array_flambda_emission_total = np.zeros_like(arraylambda, dtype=float)
    contribution_list = []
    if filterfunc:
        print("Applying filter to ARTIS spectrum")

    assert maxion is not None
    for element in range(nelements):
        nions = elementlist.nions[element]
        # nions = elementlist.iloc[element].uppermost_ion_stage - elementlist.iloc[element].lowermost_ion_stage + 1
        for ion in range(nions):
            ion_stage = ion + elementlist.lowermost_ion_stage[element]
            ionserieslist: list[tuple[int, str]] = [
                (element * maxion + ion, "bound-bound"),
                (nelements * maxion + element * maxion + ion, "bound-free"),
            ]

            if element == ion == 0:
                ionserieslist.append((2 * nelements * maxion, "free-free"))

            for selectedcolumn, emissiontypeclass in ionserieslist:
                # if linelabel.startswith('Fe ') or linelabel.endswith("-free"):
                #     continue
                if getemission:
                    array_fnu_emission = stackspectra([
                        (
                            emissiondata[dbin][timestep :: len(arr_tmid), selectedcolumn].to_numpy(),
                            arr_tdelta[timestep] / len(dbinlist),
                        )
                        for timestep in range(timestepmin, timestepmax + 1)
                        for dbin in dbinlist
                    ])
                else:
                    array_fnu_emission = np.zeros_like(arraylambda, dtype=float)

                if absorptiondata and selectedcolumn < nelements * maxion:  # bound-bound process
                    array_fnu_absorption = stackspectra([
                        (
                            absorptiondata[dbin][timestep :: len(arr_tmid), selectedcolumn].to_numpy(),
                            arr_tdelta[timestep] / len(dbinlist),
                        )
                        for timestep in range(timestepmin, timestepmax + 1)
                        for dbin in dbinlist
                    ])
                else:
                    array_fnu_absorption = np.zeros_like(arraylambda, dtype=float)

                if filterfunc:
                    array_fnu_emission = filterfunc(array_fnu_emission)
                    if selectedcolumn <= nelements * maxion:
                        array_fnu_absorption = filterfunc(array_fnu_absorption)

                array_flambda_emission = array_fnu_emission * arraynu / arraylambda
                array_flambda_absorption = array_fnu_absorption * arraynu / arraylambda

                array_flambda_emission_total += array_flambda_emission
                fluxcontribthisseries = abs(integrate.trapezoid(array_fnu_emission, x=arraynu)) + abs(
                    integrate.trapezoid(array_fnu_absorption, x=arraynu)
                )

                if emissiontypeclass == "bound-bound":
                    linelabel = get_ionstring(elementlist.Z[element], ion_stage)
                elif emissiontypeclass == "free-free":
                    linelabel = "free-free"
                else:
                    linelabel = f"{get_ionstring(elementlist.Z[element], ion_stage)} {emissiontypeclass}"

                contribution_list.append(
                    FluxContributionTuple(
                        fluxcontrib=fluxcontribthisseries,
                        linelabel=linelabel,
                        array_flambda_emission=array_flambda_emission,
                        array_flambda_absorption=array_flambda_absorption,
                        color=None,
                    )
                )

    return contribution_list, array_flambda_emission_total


def get_flux_contributions_from_packets(
    modelpath: Path,
    timelowdays: float,
    timehighdays: float,
    lambda_min: float,
    lambda_max: float,
    delta_lambda: float | npt.NDArray[np.floating] | None = None,
    getemission: bool = True,
    getabsorption: bool = True,
    maxpacketfiles: int | None = None,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    groupby: str = "ion",
    maxseriescount: int | None = None,
    fixedionlist: list[str] | None = None,
    use_time: t.Literal["arrival", "emission", "escape"] = "arrival",
    emtypecolumn: str | None = None,
    emissionvelocitycut: float | None = None,
    directionbin: int | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    directionbins_are_vpkt_observers: bool = False,
    vpkt_match_emission_exclusion_to_opac: bool = False,
    gamma: bool = False,
) -> tuple[list[FluxContributionTuple], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    from scipy import integrate

    assert groupby in {"ion", "line", "nuc", "nucmass"}
    assert emtypecolumn in {"emissiontype", "trueemissiontype", "pellet_nucindex"}

    if gamma:
        assert groupby in {"nuc", "nucmass"}
        assert emtypecolumn == "pellet_nucindex"

    if directionbin is None:
        directionbin = -1

    linelistlazy, bflistlazy = (
        (get_linelist_pldf(modelpath=modelpath, get_ion_str=True), get_bflist(modelpath, get_ion_str=True))
        if groupby != "nuc"
        else (None, None)
    )

    cols = {"e_rf"}
    cols.add({"arrival": "t_arrive_d", "emission": "em_time", "escape": "escape_time"}[use_time])

    nu_min = 2.99792458e18 / lambda_max
    nu_max = 2.99792458e18 / lambda_min

    vpkt_config = None
    opacchoiceindex = None
    if directionbins_are_vpkt_observers:
        vpkt_config = get_vpkt_config(modelpath)
        obsdirindex = directionbin // vpkt_config["nspectraperobs"]
        opacchoiceindex = directionbin % vpkt_config["nspectraperobs"]
        nprocs_read, lzdfpackets = atpackets.get_virtual_packets_pl(modelpath, maxpacketfiles=maxpacketfiles)
        lzdfpackets = lzdfpackets.with_columns(e_rf=pl.col(f"dir{obsdirindex}_e_rf_{opacchoiceindex}"))
        dirbin_nu_column = f"dir{obsdirindex}_nu_rf"

        cols |= {dirbin_nu_column, f"dir{obsdirindex}_t_arrive_d", f"dir{obsdirindex}_e_rf_{opacchoiceindex}"}
        lzdfpackets = lzdfpackets.filter(pl.col(f"dir{obsdirindex}_t_arrive_d").is_between(timelowdays, timehighdays))

    else:
        nprocs_read, lzdfpackets = atpackets.get_packets_pl(
            modelpath,
            maxpacketfiles=maxpacketfiles,
            packet_type="TYPE_ESCAPE",
            escape_type="TYPE_GAMMA" if gamma else "TYPE_RPKT",
        )
        dirbin_nu_column = "nu_rf"

        lzdfpackets = lzdfpackets.filter(pl.col("t_arrive_d").is_between(timelowdays, timehighdays))

    condition_nu_emit = pl.col(dirbin_nu_column).is_between(nu_min, nu_max) if getemission else pl.lit(False)
    condition_nu_abs = pl.col("absorption_freq").is_between(nu_min, nu_max) if getabsorption else pl.lit(False)
    lzdfpackets = lzdfpackets.filter(condition_nu_emit | condition_nu_abs)

    if emissionvelocitycut is not None:
        lzdfpackets = atpackets.add_derived_columns_lazy(lzdfpackets)
        lzdfpackets = lzdfpackets.filter(pl.col("emission_velocity") > emissionvelocitycut)

    expr_linelist_to_str = (
        pl.col("ion_str")
        if groupby == "ion"
        else pl.format(
            "{} λ{} {}-{}",
            pl.col("ion_str"),
            pl.col("lambda_angstroms_air").sub(0.5).round(0).cast(pl.String).str.strip_suffix(".0"),
            pl.col("upperlevelindex"),
            pl.col("lowerlevelindex"),
        )
    )

    if getemission:
        cols |= {"emissiontype_str", dirbin_nu_column}
        if groupby == "nuc":
            emtypestrings = get_nuclides(modelpath=modelpath).rename({"nucname": "emissiontype_str"})
        elif groupby == "nucmass":
            emtypestrings = get_nuclides(modelpath=modelpath).with_columns(
                (
                    pl.when(pl.col("pellet_nucindex") == -1).then("nucname").otherwise(pl.format("A={}", pl.col("A")))
                ).alias("emissiontype_str")
            )
        else:
            assert linelistlazy is not None
            assert bflistlazy is not None
            bflistlazy = bflistlazy.with_columns((-1 - pl.col("bfindex").cast(pl.Int32)).alias(emtypecolumn))
            expr_bflist_to_str = (
                pl.col("ion_str") + " bound-free"
                if groupby == "ion"
                else pl.format("{} bound-free {}-{}", pl.col("ion_str"), pl.col("lowerlevel"), pl.col("upperionlevel"))
            )

            emtypestrings = pl.concat([
                linelistlazy.select([
                    pl.col("lineindex").alias(emtypecolumn),
                    expr_linelist_to_str.alias("emissiontype_str"),
                ]),
                pl.DataFrame(
                    {emtypecolumn: [-9999999], "emissiontype_str": ["free-free"]},
                    schema_overrides={emtypecolumn: pl.Int32, "emissiontype_str": pl.String},
                ).lazy(),
                bflistlazy.select([pl.col(emtypecolumn), expr_bflist_to_str.alias("emissiontype_str")]),
            ])

        lzdfpackets = lzdfpackets.join(emtypestrings, on=emtypecolumn, how="left")

        if vpkt_match_emission_exclusion_to_opac and directionbins_are_vpkt_observers:
            assert vpkt_config is not None
            assert opacchoiceindex is not None
            z_exclude = int(vpkt_config["z_excludelist"][opacchoiceindex])
            if z_exclude == -1:
                # no bound-bound
                lzdfpackets = lzdfpackets.filter(pl.col("emissiontype_str").str.contains("bound-free"))
            elif z_exclude == -2:
                # no bound-free
                lzdfpackets = lzdfpackets.filter(pl.col("emissiontype_str").str.contains("bound-free").not_())
            elif z_exclude > 0:
                elsymb = get_elsymbol(z_exclude)
                lzdfpackets = lzdfpackets.filter(pl.col("emissiontype_str").str.starts_with(f"{elsymb} ").not_())

    if getabsorption:
        cols |= {"absorptiontype_str", "absorption_freq"}
        assert linelistlazy is not None
        abstypestrings = pl.concat([
            linelistlazy.select([
                pl.col("lineindex").alias("absorption_type"),
                expr_linelist_to_str.alias("absorptiontype_str"),
            ]),
            pl.DataFrame(
                {"absorption_type": [-1, -2], "absorptiontype_str": ["free-free", "bound-free"]},
                schema_overrides={"absorption_type": pl.Int32, "absorptiontype_str": pl.String},
            ).lazy(),
        ]).with_columns(pl.col("absorptiontype_str"))

        lzdfpackets = lzdfpackets.join(abstypestrings, on="absorption_type", how="left")

    if directionbin != -1:
        cols |= {"costhetabin", "phibin", "dirbin"}

    dfpackets = lzdfpackets.select([col for col in cols if col in lzdfpackets.collect_schema().names()]).collect()

    emissiongroups = (
        {
            k: v
            for (k,), v in dfpackets.filter(pl.col(dirbin_nu_column).is_between(nu_min, nu_max)).group_by(
                "emissiontype_str"
            )
        }
        if getemission
        else {}
    )

    absorptiongroups = (
        {
            k: v
            for (k,), v in dfpackets.filter(pl.col("absorption_freq").is_between(nu_min, nu_max)).group_by(
                "absorptiontype_str"
            )
        }
        if getabsorption
        else {}
    )

    allgroupnames = set(emissiongroups.keys()) | set(absorptiongroups.keys())

    if maxseriescount is None:
        maxseriescount = len(allgroupnames)

    # group small contributions together to avoid the cost of binning individual spectra for them
    grouptotals: list[tuple[float, str]] = []
    for groupname in allgroupnames:
        groupemiss = emissiongroups[groupname]["e_rf"].sum() if groupname in emissiongroups else 0.0
        groupabs = absorptiongroups[groupname]["e_rf"].sum() if groupname in absorptiongroups else 0.0
        grouptotal = groupemiss + groupabs

        if groupname is not None:
            assert isinstance(groupname, str)
            grouptotals.append((grouptotal, groupname))
        else:
            with contextlib.suppress(KeyError):
                del emissiongroups[groupname]
                del absorptiongroups[groupname]

    allgroupnames = set(emissiongroups.keys()) | set(absorptiongroups.keys())

    if fixedionlist is not None and (unrecognised_items := [x for x in fixedionlist if x not in allgroupnames]):
        print(f"WARNING: (packets) did not find {len(unrecognised_items)} items in fixedionlist: {unrecognised_items}")

    def sortkey(x: tuple[float, str]) -> tuple[int, float]:
        (grouptotal, groupname) = x

        if fixedionlist is None:
            return (0, -grouptotal)

        return (fixedionlist.index(groupname), 0) if groupname in fixedionlist else (len(fixedionlist) + 1, -grouptotal)

    sorted_grouptotals = sorted(grouptotals, key=sortkey)
    other_groups = sorted_grouptotals[maxseriescount:]

    if other_groups:
        allgroupnames.add("Other")

        if emdfs := [emissiongroups[groupname] for _, groupname in other_groups if groupname in emissiongroups]:
            emissiongroups["Other"] = pl.concat(emdfs)

        if absdfs := [absorptiongroups[groupname] for _, groupname in other_groups if groupname in absorptiongroups]:
            absorptiongroups["Other"] = pl.concat(absdfs)

        for _grouptotal, groupname in other_groups:
            with contextlib.suppress(KeyError):
                del emissiongroups[groupname]
                del absorptiongroups[groupname]
            allgroupnames.remove(groupname)

    array_flambda_emission_total = None
    contribution_list = []
    array_lambda = None
    for groupname in allgroupnames:
        array_flambda_emission = None

        if groupname in emissiongroups:
            spec_group = get_from_packets(
                modelpath=modelpath,
                timelowdays=timelowdays,
                timehighdays=timehighdays,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                use_time=use_time,
                delta_lambda=delta_lambda,
                fluxfilterfunc=filterfunc,
                nprocs_read_dfpackets=(nprocs_read, emissiongroups[groupname]),
                directionbins=[directionbin],
                directionbins_are_vpkt_observers=directionbins_are_vpkt_observers,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                gamma=gamma,
            )[directionbin]

            if array_lambda is None:
                array_lambda = spec_group["lambda_angstroms"].to_numpy()

            array_flambda_emission = spec_group["f_lambda"].to_numpy()

            if array_flambda_emission_total is None:
                array_flambda_emission_total = np.zeros_like(array_flambda_emission, dtype=float)

            array_flambda_emission_total += array_flambda_emission

        if groupname in absorptiongroups:
            spec_group = get_from_packets(
                modelpath=modelpath,
                timelowdays=timelowdays,
                timehighdays=timehighdays,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                use_time=use_time,
                delta_lambda=delta_lambda,
                nu_column="absorption_freq",
                fluxfilterfunc=filterfunc,
                nprocs_read_dfpackets=(nprocs_read, absorptiongroups[groupname]),
                directionbins=[directionbin],
                directionbins_are_vpkt_observers=directionbins_are_vpkt_observers,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
            )[directionbin]

            if array_lambda is None:
                array_lambda = spec_group["lambda_angstroms"].to_numpy()

            array_flambda_absorption = spec_group["f_lambda"].to_numpy()
        else:
            array_flambda_absorption = np.zeros_like(array_flambda_emission, dtype=float)

        if array_flambda_emission is None:
            array_flambda_emission = np.zeros_like(array_flambda_absorption, dtype=float)

        fluxcontribthisseries = abs(integrate.trapezoid(array_flambda_emission, x=array_lambda)) + abs(
            integrate.trapezoid(array_flambda_absorption, x=array_lambda)
        )

        if fluxcontribthisseries > 0.0:
            contribution_list.append(
                FluxContributionTuple(
                    fluxcontrib=fluxcontribthisseries,
                    linelabel=str(groupname),
                    array_flambda_emission=array_flambda_emission,
                    array_flambda_absorption=array_flambda_absorption,
                    color=None,
                )
            )

    if array_flambda_emission_total is None:
        array_flambda_emission_total = np.zeros_like(array_lambda, dtype=float)

    assert array_lambda is not None

    return contribution_list, array_flambda_emission_total, array_lambda


def sort_and_reduce_flux_contribution_list(
    contribution_list_in: list[FluxContributionTuple],
    maxseriescount: int,
    arraylambda_angstroms: npt.NDArray[np.floating],
    fixedionlist: list[str] | None = None,
    hideother: bool = False,
    greyscale: bool = False,
) -> list[FluxContributionTuple]:
    from scipy import integrate

    if fixedionlist:
        if unrecognised_items := [x for x in fixedionlist if x not in [y.linelabel for y in contribution_list_in]]:
            print(f"WARNING: did not understand these items in fixedionlist: {unrecognised_items}")

        # sort in manual order
        def sortkey(x: FluxContributionTuple) -> tuple[int, float]:
            assert fixedionlist is not None
            return (
                fixedionlist.index(x.linelabel) if x.linelabel in fixedionlist else len(fixedionlist) + 1,
                -x.fluxcontrib,
            )

    else:
        # sort descending by flux contribution
        def sortkey(x: FluxContributionTuple) -> tuple[int, float]:
            return (0, -x.fluxcontrib)

    contribution_list = sorted(contribution_list_in, key=sortkey)

    color_list: list[t.Any]
    if greyscale:
        from artistools.spectra.plotspectra import hatchestypes

        seriescount = len(fixedionlist) if fixedionlist else maxseriescount
        colorcount = math.ceil(seriescount / 1.0 / len(hatchestypes))
        greylist = [str(x) for x in np.linspace(0.4, 0.9, colorcount, endpoint=True)]
        color_list = []
        for c in range(colorcount):
            for _h in hatchestypes:
                color_list.append(greylist[c])
        # color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))
        mpl.rcParams["hatch.linewidth"] = 0.1
        # TODO: remove???
        color_list = list(plt.get_cmap("tab20")(np.linspace(0, 1.0, 20)))
    else:
        color_list = list(plt.get_cmap("tab20")(np.linspace(0, 1.0, 20)))

    # combine the items past maxseriescount or not in manual list into a single item
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_fluxcontrib = 0.0

    contribution_list_out = []
    numotherprinted = 0
    maxnumotherprinted = 20
    entered_other = False
    plotted_ion_list = []
    index = 0

    for row in contribution_list:
        if row.linelabel != "Other" and fixedionlist and row.linelabel in fixedionlist:
            contribution_list_out.append(row._replace(color=color_list[fixedionlist.index(row.linelabel)]))
        elif row.linelabel != "Other" and not fixedionlist and index < maxseriescount:
            contribution_list_out.append(row._replace(color=color_list[index]))
            plotted_ion_list.append(row.linelabel)
        else:
            remainder_fluxcontrib += row.fluxcontrib
            remainder_flambda_emission += row.array_flambda_emission
            remainder_flambda_absorption += row.array_flambda_absorption
            if row.linelabel != "Other" and not entered_other:
                print(f"  Other (top {maxnumotherprinted}):")
                entered_other = True

        if row.linelabel != "Other":
            index += 1

        if numotherprinted < maxnumotherprinted and row.linelabel != "Other":
            integemiss = abs(integrate.trapezoid(row.array_flambda_emission, x=arraylambda_angstroms))
            integabsorp = abs(integrate.trapezoid(-row.array_flambda_absorption, x=arraylambda_angstroms))
            if integabsorp > 0.0 and integemiss > 0.0:
                print(
                    f"{row.fluxcontrib:.1e}, emission {integemiss:.1e}, "
                    f"absorption {integabsorp:.1e} [erg/s/cm^2]: '{row.linelabel}'"
                )
            elif integemiss > 0.0:
                print(f"  emission {integemiss:.1e} [erg/s/cm^2]: '{row.linelabel}'")
            else:
                print(f"absorption {integabsorp:.1e} [erg/s/cm^2]: '{row.linelabel}'")

            if entered_other:
                numotherprinted += 1

    if not fixedionlist:
        cmdarg = "'" + "' '".join(plotted_ion_list) + "'"
        print("To reuse this ion/process contribution list, pass the following command-line argument: ")
        print(f"     -fixedionlist {cmdarg}")
        print("Or in python: ")
        print(f"     fixedionlist={plotted_ion_list}")

    if remainder_fluxcontrib > 0.0 and not hideother:
        contribution_list_out.append(
            FluxContributionTuple(
                fluxcontrib=remainder_fluxcontrib,
                linelabel="Other",
                array_flambda_emission=remainder_flambda_emission,
                array_flambda_absorption=remainder_flambda_absorption,
                color="grey",
            )
        )

    return contribution_list_out


def print_integrated_flux(
    arr_df_on_dx: npt.NDArray[np.floating] | pl.Series, arr_x: npt.NDArray[np.floating] | pl.Series
) -> float:
    from scipy import integrate

    integrated_flux = abs(integrate.trapezoid(np.nan_to_num(arr_df_on_dx, nan=0.0), x=arr_x))
    x_min = arr_x.min()
    x_max = arr_x.max()
    assert isinstance(x_min, int | float)
    assert isinstance(x_max, int | float)

    print(f" integrated flux (x={x_min:.1f} to x={x_max:.1f}): {integrated_flux:.3e} erg/s/cm2 at 1 Mpc")
    assert isinstance(integrated_flux, float)
    return integrated_flux


def get_reference_spectrum(filename: Path | str) -> tuple[pl.DataFrame, dict[t.Any, t.Any]]:
    if Path(filename).is_file():
        filepath = Path(filename)
    else:
        filepath = Path(get_config()["path_artistools_dir"], "data", "refspectra", filename)

        if not filepath.is_file():
            filepathxz = filepath.with_suffix(f"{filepath.suffix}.xz")
            if filepathxz.is_file():
                filepath = filepathxz
            else:
                filepathgz = filepath.with_suffix(f"{filepath.suffix}.gz")
                if filepathgz.is_file():
                    filepath = filepathgz

    metadata = get_file_metadata(filepath)

    flambdaindex = metadata.get("f_lambda_columnindex", 1)

    specdata = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["lambda_angstroms", "f_lambda"],
        usecols=[0, flambdaindex],
    )

    # new_lambda_angstroms = []
    # binned_flux = []
    #
    # wavelengths = specdata['lambda_angstroms']
    # fluxes = specdata['f_lambda']
    # nbins = 10
    #
    # for i in np.arange(start=0, stop=len(wavelengths) - nbins, step=nbins):
    #     new_lambda_angstroms.append(wavelengths[i + int(nbins / 2)])
    #     sum_flux = 0
    #     for j in range(i, i + nbins):
    #
    #         if not math.isnan(fluxes[j]):
    #             print(fluxes[j])
    #             sum_flux += fluxes[j]
    #     binned_flux.append(sum_flux / nbins)
    #
    # filtered_specdata = pd.DataFrame(new_lambda_angstroms, columns=['lambda_angstroms'])
    # filtered_specdata['f_lamba'] = binned_flux
    # print(filtered_specdata)
    # plt.plot(specdata['lambda_angstroms'], specdata['f_lambda'])
    # plt.plot(new_lambda_angstroms, binned_flux)
    #
    # filtered_specdata.to_csv('/Users/ccollins/artis_nebular/artistools/artistools/data/refspectra/' + name,
    #                          index=False, header=False, sep=' ')

    if "a_v" in metadata or "e_bminusv" in metadata:
        print("Correcting for reddening")
        from extinction import apply
        from extinction import ccm89

        specdata["f_lambda"] = apply(
            ccm89(specdata["lambda_angstroms"].to_numpy(), a_v=-metadata["a_v"], r_v=metadata["r_v"], unit="aa"),
            specdata["f_lambda"].to_numpy(),
        )

    if "z" in metadata:
        print("Correcting for redshift")
        specdata["lambda_angstroms"] /= 1 + metadata["z"]

    return pl.from_pandas(specdata), metadata
