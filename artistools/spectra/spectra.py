"""Artistools - spectra related functions."""

import argparse
import math
import re
import typing as t
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import matplotlib.typing as mplt
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

import artistools.constants as const
import artistools.packets as atpackets
from artistools.atomic import add_ion_str_column
from artistools.atomic import get_bflist
from artistools.atomic import get_elsymbol
from artistools.atomic import get_ionstring
from artistools.atomic import get_linelist_pldf
from artistools.atomic import get_nuclides
from artistools.misc import average_direction_bins
from artistools.misc import check_averaging_angles
from artistools.misc import df_filter_minmax_bracketed
from artistools.misc import drop_trailing_null_column
from artistools.misc import firstexisting
from artistools.misc import get_dirbins
from artistools.misc import get_file_metadata
from artistools.misc import get_nprocs
from artistools.misc import get_nu_grid
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import get_vpkt_config
from artistools.misc import match_closest_time
from artistools.misc import polars_source
from artistools.misc import print_detail
from artistools.misc import print_saved
from artistools.misc import print_warning
from artistools.misc import read_wsv
from artistools.misc import split_multitable_dataframe


class FluxContributionTuple(t.NamedTuple):
    """One emission/absorption series (an ion, line, or nuclide) and its total contribution to the flux."""

    fluxcontrib: float
    linelabel: str
    array_flambda_emission: npt.NDArray[np.floating]
    array_flambda_absorption: npt.NDArray[np.floating]
    color: mplt.ColorType | None = None


def timeshift_fluxscale_co56law(scaletoreftime: float | None, spectime: float) -> float:
    """Return the factor that scales a spectrum to scaletoreftime assuming Co56 decay, or 1.0 when unset."""
    if scaletoreftime is not None:
        # Co56 decay flux scaling
        assert spectime > 150
        return math.exp(spectime / 113.7) / math.exp(scaletoreftime / 113.7)

    return 1.0


def get_dfspectrum_x_y_with_units(
    dfspectrum: pl.DataFrame | pl.LazyFrame, xunit: str, yvariable: str, fluxdistance_mpc: float
) -> pl.LazyFrame:
    """Add an x column in xunit and a y column for yvariable, scaled to an observer at fluxdistance_mpc."""
    from artistools.constants import c_ang_per_s
    from artistools.constants import h_erg_s
    from artistools.constants import h_ev_s
    from artistools.constants import megaparsec_to_cm

    dfspectrum = dfspectrum.lazy()

    if "nu" not in dfspectrum.collect_schema().names():
        dfspectrum = dfspectrum.with_columns((const.c_ang_per_s / pl.col("lambda_angstroms")).alias("nu"))
    if "f_nu" not in dfspectrum.collect_schema().names():
        dfspectrum = dfspectrum.with_columns(f_nu=(pl.col("f_lambda") * pl.col("lambda_angstroms") / pl.col("nu")))

    # set dflux_on_dx_onempc in units of [erg/s/cm^2/xunit] at 1 Mpc distance
    match xunit.lower():
        case "angstroms":
            dfspectrum = dfspectrum.with_columns(x=pl.col("lambda_angstroms"), dflux_on_dx_onempc=pl.col("f_lambda"))

        case "nm":
            dfspectrum = dfspectrum.with_columns(
                x=pl.col("lambda_angstroms") / 10, dflux_on_dx_onempc=pl.col("f_lambda") * 10
            )

        case "micron":
            dfspectrum = dfspectrum.with_columns(
                x=pl.col("lambda_angstroms") / 10000, dflux_on_dx_onempc=pl.col("f_lambda") * 10000
            )

        case "hz":
            dfspectrum = dfspectrum.with_columns(x=pl.col("nu"), dflux_on_dx_onempc=pl.col("f_nu"))

        case "erg":
            dfspectrum = (
                dfspectrum
                .with_columns(en_erg=h_erg_s * pl.col("nu"))
                .with_columns(f_en_erg=pl.col("f_nu") * pl.col("nu") / pl.col("en_erg"))
                .with_columns(x=pl.col("en_erg"), dflux_on_dx_onempc=pl.col("f_en_erg"))
            )

        case "ev":
            dfspectrum = (
                dfspectrum
                .with_columns(en_ev=h_ev_s * pl.col("nu"))
                .with_columns(f_en_ev=pl.col("f_nu") * pl.col("nu") / pl.col("en_ev"))
                .with_columns(x=pl.col("en_ev"), dflux_on_dx_onempc=pl.col("f_en_ev"))
            )

        case "kev":
            dfspectrum = (
                dfspectrum
                .with_columns(en_kev=h_ev_s * pl.col("nu") / 1000.0)
                .with_columns(f_en_kev=pl.col("f_nu") * pl.col("nu") / pl.col("en_kev"))
                .with_columns(x=pl.col("en_kev"), dflux_on_dx_onempc=pl.col("f_en_kev"))
            )

        case "mev":
            dfspectrum = (
                dfspectrum
                .with_columns(en_mev=h_ev_s * pl.col("nu") / 1e6)
                .with_columns(f_en_mev=pl.col("f_nu") * pl.col("nu") / pl.col("en_mev"))
                .with_columns(x=pl.col("en_mev"), dflux_on_dx_onempc=pl.col("f_en_mev"))
            )

        case _:
            msg = f"Unit {xunit} not implemented"
            raise NotImplementedError(msg)

    match yvariable.lower():
        case "luminosity":
            # multiply by 4pi dist^2 to cancel out the /cm^2 at 1 Mpc
            # [erg/s/xunit]
            dfspectrum = dfspectrum.with_columns(y=pl.col("dflux_on_dx_onempc") * 4 * math.pi * megaparsec_to_cm**2)

        case "flux":
            # adjust flux to required distance
            # [erg/s/cm^2/xunit]
            dfspectrum = dfspectrum.with_columns(y=pl.col("dflux_on_dx_onempc") / fluxdistance_mpc**2)

        case "eflux":
            # adjust for distance, convert erg to xunit and multiply by another factor of x
            # [xunit/s/cm^2]
            # the wavelength of a one erg photon, i.e. Planck's constant times the speed of light
            erg_to_angstrom = h_erg_s * c_ang_per_s
            xunit_per_erg = convert_angstroms_to_unit(erg_to_angstrom, xunit.lower())
            dfspectrum = dfspectrum.with_columns(
                y=(pl.col("dflux_on_dx_onempc") / fluxdistance_mpc**2 * xunit_per_erg) * pl.col("x")
            )

        case "photonflux":
            # divide by the photon energy to get a count rate and adjust for distance
            # [#/s/cm^2/xunit]
            dfspectrum = dfspectrum.with_columns(
                y=pl.col("dflux_on_dx_onempc") / fluxdistance_mpc**2 / (h_erg_s * pl.col("nu"))
            )

        case "photoncount":
            # divide by the photon energy and multiply by 4pi dist^2 to cancel out the /cm^2 at 1 Mpc
            # [#/s/xunit]
            dfspectrum = dfspectrum.with_columns(
                y=pl.col("dflux_on_dx_onempc") * 4 * math.pi * megaparsec_to_cm**2 / (h_erg_s * pl.col("nu"))
            )

        case "packetcount":
            # Monte Carlo packet count is stored separately
            dfspectrum = dfspectrum.with_columns(y=pl.col("packetcount"))

        case _:
            msg = f"Unit {yvariable} not implemented"
            raise NotImplementedError(msg)

    return dfspectrum.sort("x")


def get_exspec_lambda_bin_edges(
    modelpath: str | Path | None = None,
    mnubins: int | None = None,
    nu_min_r: float | None = None,
    nu_max_r: float | None = None,
    gamma: bool = False,
) -> npt.NDArray[np.floating]:
    """Get the wavelength bins for the emergent spectrum."""
    if modelpath is not None:
        try:
            dfspec = read_spec(modelpath, gamma=gamma).collect()
        except FileNotFoundError:
            mnubins = 1000
            if gamma:
                min_mev_on_h = 0.05
                nu_min_r = min_mev_on_h * const.MEV_to_erg / const.h_erg_s
                max_mev_on_h = 4.0
                nu_max_r = max_mev_on_h * const.MEV_to_erg / const.h_erg_s
                print(
                    f"No gamma_spec.out found. Using default gamma bins: mnubins {mnubins} nu_min_r {min_mev_on_h:.2f} MeV/H nu_max_r {max_mev_on_h:.2f} MeV/H"
                )
            else:
                nu_min_r = 1e13
                nu_max_r = 5e16
                print(
                    f"No spec.out found. Using default rpkt bins: mnubins {mnubins} nu_min_r {nu_min_r:.2e} nu_max_r {nu_max_r:.2e}"
                )
        else:
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

    dlognu = (math.log(nu_max_r) - math.log(nu_min_r)) / mnubins

    nu_bin_edges = np.array([math.exp(math.log(nu_min_r) + (m * (dlognu))) for m in range(mnubins + 1)])

    # np.flip is used to get an ascending wavelength array from an ascending nu array
    return const.c_ang_per_s / np.flip(nu_bin_edges)


def get_lambda_bin_edges(
    xmin_plot: float,
    xmax_plot: float,
    deltax: float | None,
    deltalogx: float | None,
    deltalambda: float | None,
    xunit: str,
    modelpath: Path | str,
    gamma: bool = False,
) -> npt.NDArray[np.floating]:
    """Get the minimum and maximum wavelength to collect data for, and the bin width to ensure coverage of the plotted range."""
    assert sum(param is not None for param in (deltax, deltalogx, deltalambda)) <= 1, (
        "Options deltax, deltalogx, and deltalambda are mutually exclusive, but more than one was provided."
    )
    if deltalogx is not None:
        if not deltalogx > 0:
            msg = f"deltalogx must be positive, got {deltalogx}"
            raise ValueError(msg)
        # xmin_plot is the centre of the first bin, so we need to subtract half a bin width to get the lower edge of the first bin
        xbin_lower = xmin_plot / (1 + deltalogx) ** 0.5
        xmax = xmax_plot * (1 + deltalogx) ** 0.5
        list_x_bin_edges = [xbin_lower]
        while xbin_lower <= xmax:
            xbin_lower *= 1 + deltalogx
            list_x_bin_edges.append(xbin_lower)
        x_bin_edges = np.array(list_x_bin_edges)
        lambda_bin_edges = np.sort(convert_unit_to_angstroms(x_bin_edges, xunit))
    elif deltax is not None:
        if not deltax > 0:
            msg = f"deltax must be positive, got {deltax}"
            raise ValueError(msg)
        x_bin_edges = np.arange(xmin_plot - deltax * 0.5, xmax_plot + deltax * 1.5, deltax)
        lambda_bin_edges = np.sort(convert_unit_to_angstroms(x_bin_edges, xunit))
    elif deltalambda is not None:
        if not deltalambda > 0:
            msg = f"deltalambda must be positive, got {deltalambda}"
            raise ValueError(msg)
        # the plotted x limits are bin centres, not bin edges, so shift them by half a bin width
        deltax = convert_angstroms_to_unit(deltalambda, xunit)
        xmin = xmin_plot - deltax * 0.5
        xmax = xmax_plot + deltax * 0.5
        lambda_min, lambda_max = convert_xlimits_to_lambda_range(xmin, xmax, xunit)
        lambda_bin_edges = np.arange(lambda_min, lambda_max + deltalambda, deltalambda)
    else:
        lambda_min_plot, lambda_max_plot = convert_xlimits_to_lambda_range(xmin_plot, xmax_plot, xunit)
        lambda_bin_edges_fullrange = get_exspec_lambda_bin_edges(modelpath=modelpath, gamma=gamma)
        lambda_bin_edges = (
            df_filter_minmax_bracketed(
                pl.LazyFrame({
                    "lambda_bin_lower": lambda_bin_edges_fullrange[:-1],
                    "lambda_bin_upper": lambda_bin_edges_fullrange[1:],
                }).with_columns(lambda_bin_centre=0.5 * (pl.col("lambda_bin_lower") + pl.col("lambda_bin_upper"))),
                "lambda_bin_centre",
                lambda_min_plot,
                lambda_max_plot,
            )
            .select(pl.col("lambda_bin_lower").append(pl.col("lambda_bin_upper").last()).alias("lambda_bin_edges"))
            .collect()
            .to_numpy()
            .flatten()
        )

    return lambda_bin_edges


# the spellings that a user can give for each unit of the horizontal axis of a spectrum. One table
# gives the conversion, the message that names the units, and the suggestion for a name that is close.
XUNITALIASES: t.Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "angstroms": ("angstrom", "a", "ang", "\u00e5", "\u00e5ngstr\u00f6m"),
    "nm": ("nanometer", "nanometers"),
    "micron": ("microns", "mu", "\u03bc", "\u03bcm"),
    "hz": (),
    "erg": ("ergs",),
    "ev": ("electronvolt",),
    "kev": ("kiloelectronvolt",),
    "mev": ("megaelectronvolt",),
})


def get_xunit_names() -> list[str]:
    """Return every spelling of a unit of the horizontal axis that a user can give."""
    return [name for canonical, aliases in XUNITALIASES.items() for name in (canonical, *aliases)]


def parse_xunit_argument(value: str) -> str:
    """Return the canonical unit of the horizontal axis, or refuse a name that no unit takes.

    argparse calls this while it reads the command line, thus a name with a mistake stops the
    command before it reads a file. convert_xunit_aliases_to_canonical guards the keyword of the API.
    """
    from artistools.misc import suggest_names

    try:
        return convert_xunit_aliases_to_canonical(value)
    except ValueError:
        suggestion = suggest_names(value, get_xunit_names()) or f"The units are {', '.join(XUNITALIASES)}"
        msg = f"'{value}' is not a unit of the horizontal axis. {suggestion}"
        raise argparse.ArgumentTypeError(msg) from None


def convert_xunit_aliases_to_canonical(xunit: str) -> str:
    """Return the canonical spelling of a spectrum x-axis unit name."""
    lowered = xunit.lower()
    for canonical, aliases in XUNITALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical

    msg = f"Unknown xunit {xunit}"
    raise ValueError(msg)


@t.overload
def convert_angstroms_to_unit(value_angstroms: float, new_units: str) -> float: ...


@t.overload
def convert_angstroms_to_unit(
    value_angstroms: npt.NDArray[np.floating], new_units: str
) -> npt.NDArray[np.floating]: ...


def convert_angstroms_to_unit(
    value_angstroms: float | npt.NDArray[np.floating], new_units: str
) -> float | npt.NDArray[np.floating]:
    """Convert a wavelength in angstroms to a different unit, either length, frequency, or energy."""
    hc_ev_angstroms = const.h_ev_s * const.c_ang_per_s  # [eV angstroms]
    hc_erg_angstroms = hc_ev_angstroms * const.EV_to_erg  # [erg angstroms]
    match new_units.lower():
        case "erg":
            return hc_erg_angstroms / value_angstroms
        case "ev":
            return hc_ev_angstroms / value_angstroms
        case "kev":
            return hc_ev_angstroms / value_angstroms / 1.0e3
        case "mev":
            return hc_ev_angstroms / value_angstroms / 1.0e6
        case "hz":
            return const.c_ang_per_s / value_angstroms
        case "angstroms":
            return value_angstroms
        case "nm":
            return value_angstroms / 10.0
        case "micron":
            return value_angstroms / 10000.0
        case _:
            msg = f"Unknown xunit {new_units}"
            raise ValueError(msg)


@t.overload
def convert_unit_to_angstroms(value: float, old_units: str) -> float: ...


@t.overload
def convert_unit_to_angstroms(value: npt.NDArray[np.floating], old_units: str) -> npt.NDArray[np.floating]: ...


def convert_unit_to_angstroms(
    value: float | npt.NDArray[np.floating], old_units: str
) -> float | npt.NDArray[np.floating]:
    """Convert a wavelength, frequency, or energy to wavelength angstroms."""
    c = const.c_ang_per_s
    h = const.h_ev_s
    hc_ev_angstroms = h * c  # [eV angstroms]
    match old_units.lower():
        case "erg":
            return hc_ev_angstroms * const.EV_to_erg / value
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
        case "micron":
            return value * 10000
        case _:
            msg = f"Unknown xunit {old_units}"
            raise ValueError(msg)


def convert_xlimits_to_lambda_range(xmin: float, xmax: float, xunit: str) -> tuple[float, float]:
    """Convert plot x-axis limits to an ascending wavelength range in angstroms.

    Frequency and energy units invert the ordering, so the converted limits are sorted.
    """
    lambda_min, lambda_max = sorted((convert_unit_to_angstroms(xmin, xunit), convert_unit_to_angstroms(xmax, xunit)))
    return lambda_min, lambda_max


def weighted_average_spectra(
    spectra_and_factors: list[tuple[npt.NDArray[np.floating], float]],
) -> npt.NDArray[np.floating]:
    """Average spectra using (normalised) weighting factors, i.e., specout[nu] = (spec1[nu] * factor1 + spec2[nu] * factor2 + ...) / (factor1 + factor2 + ...).

    spectra_and_factors should be a list of tuples: spectra[], factor.
    """
    spectra, factors = zip(*spectra_and_factors, strict=True)

    return np.average(spectra, axis=0, weights=factors)


def get_spectrum_at_time(
    modelpath: Path,
    timestep: int,
    time: float,
    args: argparse.Namespace | None,
    dirbin: int = -1,
    average_over_phi: bool | None = None,
    average_over_theta: bool | None = None,
) -> pl.DataFrame:
    """Return the spectrum of one direction bin at a single timestep."""
    if dirbin >= 0:
        if args is not None and args.plotvspecpol and (modelpath / "vpkt.txt").is_file():
            return get_vspecpol_spectrum(modelpath, time, dirbin, args).collect()
        assert average_over_phi is not None
        assert average_over_theta is not None
    else:
        average_over_phi = False
        average_over_theta = False

    return get_spectra(
        modelpath=modelpath,
        timestepmin=timestep,
        timestepmax=timestep,
        average_over_phi=average_over_phi,
        average_over_theta=average_over_theta,
    )[dirbin].collect()


@lru_cache(maxsize=4)
def get_binned_lambda_frame_cached(lambda_bin_edges_bytes: bytes, count: int) -> pl.LazyFrame:
    """Return the wavelength bin frame for the given packed bin edges.

    The bytes key makes the arguments hashable for the lru_cache.
    """
    lambda_bin_edges = np.frombuffer(lambda_bin_edges_bytes, dtype=np.float64, count=count)
    return (
        pl
        .DataFrame({
            "lambda_angstroms": 0.5 * (lambda_bin_edges[:-1] + lambda_bin_edges[1:]),
            "delta_lambda": lambda_bin_edges[1:] - lambda_bin_edges[:-1],
        })
        .with_row_index("lambda_binindex")
        .with_columns(nu=(const.c_ang_per_s / pl.col("lambda_angstroms")))
        .lazy()
    )


def get_binned_lambda_frame(lambda_bin_edges: npt.NDArray[np.floating]) -> pl.LazyFrame:
    """Return the centre, the width and the frequency of each wavelength bin.

    The result is in a cache, because the code bins each emission contribution and each absorption contribution
    separately. Without the cache, the code makes this frame again for each contribution.
    """
    edges = np.ascontiguousarray(lambda_bin_edges, dtype=np.float64)
    return get_binned_lambda_frame_cached(edges.tobytes(), edges.size)


def select_dirbins(alldirbins: list[int], requested: Sequence[int] | None) -> list[int]:
    """Return the requested direction bins. Return all the available direction bins if the caller requests none."""
    if requested is None:
        return alldirbins

    if unavailable := [dirbin for dirbin in requested if dirbin not in alldirbins]:
        msg = f"Direction bins {unavailable} are not available (have {alldirbins})"
        raise ValueError(msg)

    return list(requested)


@lru_cache(maxsize=16)
def get_escape_surface_gamma(modelpath: Path | str) -> float:
    """Return the Lorentz factor correction at the outer model boundary."""
    from artistools.inputmodel import get_modeldata

    _, modelmeta = get_modeldata(modelpath)
    vmax_beta = float(modelmeta["vmax_cmps"]) / const.C_cm_per_s
    return math.sqrt(1 - vmax_beta**2)


def filter_packets_by_time(
    dfpackets: pl.LazyFrame,
    modelpath: Path | str,
    timelowdays: float,
    timehighdays: float,
    use_time: t.Literal["arrival", "emission", "escape"],
    gamma: bool,
) -> tuple[pl.LazyFrame, float | None]:
    """Filter packets with the selected time and return the escape correction when applicable."""
    if use_time == "arrival":
        return dfpackets.filter(pl.col("t_arrive_d").is_between(timelowdays, timehighdays)), None

    if use_time == "escape":
        escapesurfacegamma = get_escape_surface_gamma(modelpath)
        return (
            dfpackets.filter(
                (pl.col("escape_time") * escapesurfacegamma / const.day_to_s).is_between(timelowdays, timehighdays)
            ),
            escapesurfacegamma,
        )

    col_emit_time = "tdecay" if gamma else "em_time"
    mean_correction = (pl.col(col_emit_time) - pl.col("t_arrive_d") * const.day_to_s).mean()
    return (
        dfpackets.filter(
            pl.col(col_emit_time).is_between(
                timelowdays * const.day_to_s + mean_correction, timehighdays * const.day_to_s + mean_correction
            )
        ),
        None,
    )


def get_from_packets(
    modelpath: Path | str,
    timelowdays: float,
    timehighdays: float,
    lambda_bin_edges: npt.NDArray[np.floating] | None = None,
    use_time: t.Literal["arrival", "emission", "escape"] = "arrival",
    maxpacketfiles: int | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    nu_column: str = "nu_rf",
    fluxfilterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    nprocs_read_dfpackets: tuple[int, pl.DataFrame | pl.LazyFrame] | None = None,
    directionbins_are_vpkt_observers: bool = False,
    directionbins: Sequence[int] | None = None,
    gamma: bool = False,
    packets_are_time_filtered: bool = False,
) -> dict[int, pl.LazyFrame]:
    """Return a spectrum dataframe. The packets files are the input.

    The directionbins parameter selects the viewing direction bins. The default is all the direction bins.
    A query for each direction bin has a cost. Thus a caller that needs one bin must request one bin.
    """
    assert use_time in {"arrival", "emission", "escape"}
    if directionbins_are_vpkt_observers and use_time != "arrival":
        msg = "Virtual packet spectra support only observer arrival time"
        raise ValueError(msg)

    if nu_column == "absorption_freq":
        nu_column = "nu_absorbed"
    if lambda_bin_edges is None:
        lambda_bin_edges = get_exspec_lambda_bin_edges(modelpath=modelpath, gamma=gamma)
    lambda_bin_edges = np.sort(lambda_bin_edges)
    delta_time_s = (timehighdays - timelowdays) * const.day_to_s

    if nprocs_read_dfpackets:
        nprocs_read, dfpackets = nprocs_read_dfpackets[0], nprocs_read_dfpackets[1].lazy()
    elif directionbins_are_vpkt_observers:
        assert not gamma
        nprocs_read, dfpackets = atpackets.get_virtual_packets(modelpath, maxpacketfiles=maxpacketfiles)
    else:
        nprocs_read, dfpackets = atpackets.get_packets(
            modelpath,
            maxpacketfiles=maxpacketfiles,
            packet_type="TYPE_ESCAPE",
            escape_type="TYPE_GAMMA" if gamma else "TYPE_RPKT",
        )

    dfpackets = dfpackets.with_columns([
        (const.c_ang_per_s / pl.col(colname)).alias(
            colname.replace("absorption_freq", "nu_absorbed").replace("nu_", "lambda_angstroms_")
        )
        for colname in dfpackets.collect_schema().names()
        if "nu_" in colname or colname == "absorption_freq"
    ])

    dfbinned_lazy = get_binned_lambda_frame(lambda_bin_edges)
    escapesurfacegamma: float | int | None = None
    dirbin_spectra: dict[int, pl.LazyFrame] = {}
    if directionbins_are_vpkt_observers:
        vpkt_config = get_vpkt_config(modelpath)
        alldirbins = list(range(vpkt_config["nobsdirections"] * vpkt_config["nspectraperobs"]))
        for vspecindex in select_dirbins(alldirbins, directionbins):
            obsdirindex, opacchoiceindex = divmod(vspecindex, vpkt_config["nspectraperobs"])
            lambda_column = (
                f"dir{obsdirindex}_lambda_angstroms_rf"
                if nu_column == "nu_rf"
                else nu_column.replace("absorption_freq", "nu_absorbed").replace("nu_", "lambda_angstroms_")
            )
            energy_column = f"dir{obsdirindex}_e_rf_{opacchoiceindex}"

            dirbin_spectra[vspecindex] = (
                atpackets
                .bin_and_sum(
                    dfpackets
                    if packets_are_time_filtered
                    else dfpackets.filter(pl.col(f"dir{obsdirindex}_t_arrive_d").is_between(timelowdays, timehighdays)),
                    bincol=lambda_column,
                    bins=lambda_bin_edges.tolist(),
                    sumcols=[energy_column],
                    getcounts=True,
                )
                .select(
                    lambda_binindex=pl.col(f"{lambda_column}_bin"),
                    flux=pl.col(f"{energy_column}_sum") / delta_time_s / (const.megaparsec_to_cm**2) / nprocs_read,
                    packetcount=pl.col("count"),
                )
                .join(dfbinned_lazy, on="lambda_binindex", how="left", coalesce=True, maintain_order="left")
                .with_columns(f_lambda=pl.col("flux") / pl.col("delta_lambda"))
                .drop("flux")
            )

            if fluxfilterfunc:
                dirbin_spectra[vspecindex] = (
                    dirbin_spectra[vspecindex]
                    .with_columns(pl.col("f_lambda").map_batches(fluxfilterfunc, return_dtype=pl.self_dtype()))
                    .with_columns(f_nu=(pl.col("f_lambda") * pl.col("lambda_angstroms") / pl.col("nu")))
                )

    else:
        alldirbins = [-1, *get_dirbins(average_over_phi=average_over_phi, average_over_theta=average_over_theta)]
        lambda_column = nu_column.replace("nu_", "lambda_angstroms_")
        energy_column = "e_cmf" if use_time == "escape" else "e_rf"

        if packets_are_time_filtered:
            if use_time == "escape":
                escapesurfacegamma = get_escape_surface_gamma(modelpath)
        else:
            dfpackets, escapesurfacegamma = filter_packets_by_time(
                dfpackets, modelpath, timelowdays, timehighdays, use_time, gamma
            )

        dfpackets = dfpackets.filter(pl.col(lambda_column).is_between(lambda_bin_edges[0], lambda_bin_edges[-1]))

        for dirbin in select_dirbins(alldirbins, directionbins):
            pldfpackets_dirbin_lazy, inverse_solidangle_fraction = atpackets.filter_packets_dirbin(
                dfpackets, dirbin, average_over_phi=average_over_phi, average_over_theta=average_over_theta
            )

            dirbin_spectra[dirbin] = atpackets.bin_and_sum(
                pldfpackets_dirbin_lazy,
                bincol=lambda_column,
                bins=lambda_bin_edges.tolist(),
                sumcols=[energy_column],
                getcounts=True,
            ).select(
                lambda_binindex=pl.col(f"{lambda_column}_bin"),
                flux=(
                    pl.col(f"{energy_column}_sum")
                    / delta_time_s
                    * inverse_solidangle_fraction
                    / (4 * math.pi * const.megaparsec_to_cm**2)
                    / nprocs_read
                ),
                packetcount=pl.col("count"),
            )

            if use_time == "escape":
                assert escapesurfacegamma is not None
                dirbin_spectra[dirbin] = dirbin_spectra[dirbin].with_columns(
                    pl.col("flux").mul(1.0 / escapesurfacegamma)
                )

            dirbin_spectra[dirbin] = (
                dirbin_spectra[dirbin]
                .join(dfbinned_lazy, on="lambda_binindex", how="left", coalesce=True, maintain_order="left")
                .with_columns(f_lambda=pl.col("flux") / pl.col("delta_lambda"))
                .drop("flux")
                .with_columns(f_nu=(pl.col("f_lambda") * pl.col("lambda_angstroms") / pl.col("nu")))
            )

            if fluxfilterfunc:
                dirbin_spectra[dirbin] = dirbin_spectra[dirbin].with_columns(
                    cs.by_name(("f_lambda", "f_nu")).map_batches(fluxfilterfunc, return_dtype=pl.self_dtype())
                )

    if fluxfilterfunc:
        print("Applying filter to ARTIS spectrum")

    return dirbin_spectra


# maxsize is small because this reads eagerly and every cached entry retains a whole spec file. A cached
# scan would hold only the query plan, thus each collect by a caller would parse the file again.
@lru_cache(maxsize=2)
def read_spec(modelpath: Path | str, gamma: bool = False) -> pl.LazyFrame:
    """Return the angle-averaged spectra from spec.out, or from gamma_spec.out when gamma is set.

    Callers must not mutate the returned frame, which is shared between calls.
    """
    specfilename = firstexisting("gamma_spec.out" if gamma else "spec.out", folder=modelpath, tryzipped=True)
    print(f"Reading {specfilename}")

    return (
        pl
        .read_csv(polars_source(specfilename), separator=" ", infer_schema=False, truncate_ragged_lines=True)
        .with_columns(pl.all().cast(pl.Float64))
        .rename({"0": "nu"})
        .lazy()
    )


# maxsize is small because, unlike read_spec above, this reads eagerly and every cached entry
# retains a whole spec_res file: the per-dirbin frames are all slices of one parsed frame
@lru_cache(maxsize=2)
def read_spec_res(modelpath: Path | str, gamma: bool = False) -> dict[int, pl.LazyFrame]:
    """Return a dict of LazyFrames of time-series spectra keyed to the viewing direction bin.

    Callers must not mutate the returned dict, which is shared between calls.
    """
    resfilenames = ["gamma_spec_res.out"] if gamma else ["spec_res.out", "specpol_res.out"]
    specfilename = (
        modelpath if Path(modelpath).is_file() else firstexisting(resfilenames, folder=modelpath, tryzipped=True)
    )

    print(f"Reading {specfilename} (in read_spec_res)")
    res_specdata_in = drop_trailing_null_column(
        pl.read_csv(polars_source(specfilename), separator=" ", has_header=False, infer_schema=False).lazy()
    )

    res_specdata = split_multitable_dataframe(res_specdata_in)

    for dirbin in res_specdata:
        # the column names are not stored as dataframe.columns yet, but exist in the first row of the DataFrame
        newcolnames = [str(x) for x in res_specdata[dirbin].select(pl.all().slice(0, 1)).collect().row(0)]
        newcolnames[0] = "nu"

        newcolnames_unique = set(newcolnames)
        oldcolnames = res_specdata[dirbin].collect_schema().names()
        if len(newcolnames) > len(newcolnames_unique):
            # for POL_ON, the time columns repeat for the Q and U stokes parameters.
            # here, we keep the first set (I) and drop the rest of the columns.
            # the layout is one nu column plus an exact multiple of the unique time columns
            assert (len(newcolnames) - 1) % (len(newcolnames_unique) - 1) == 0
            newcolnames = newcolnames[: len(newcolnames_unique)]
            oldcolnames = oldcolnames[: len(newcolnames_unique)]
            res_specdata[dirbin] = res_specdata[dirbin].select(oldcolnames)

        res_specdata[dirbin] = (
            res_specdata[dirbin]
            .select(pl.all().slice(offset=1))  # drop the first row that contains time headers
            .with_columns(pl.all().cast(pl.Float64))
            .rename(dict(zip(oldcolnames, newcolnames, strict=True)))
        )

    return res_specdata


def read_emission_absorption_file(emabsfilename: str | Path) -> pl.LazyFrame:
    """Read into a DataFrame one of: emission.out. emissionpol.out, emissiontrue.out, absorption.out."""
    try:
        emissionfilesize = Path(emabsfilename).stat().st_size / 1024 / 1024
        print(f" Reading {emabsfilename} ({emissionfilesize:.2f} MiB)")

    except AttributeError:
        print(f" Reading {emabsfilename}")

    dfemabs = pl.scan_csv(
        polars_source(emabsfilename), separator=" ", has_header=False, infer_schema_length=0
    ).with_columns(pl.all().cast(pl.Float32, strict=True))

    return drop_trailing_null_column(dfemabs)


def get_spectra(
    modelpath: Path,
    timestepmin: int,
    timestepmax: int | None = None,
    fluxfilterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    average_over_theta: bool = False,
    average_over_phi: bool = False,
    gamma: bool = False,
) -> dict[int, pl.LazyFrame]:
    """Get a mapping direction bins to polars LazyFrames containing ARTIS emergent UVOIR spectra."""
    if timestepmax is None or timestepmax < 0:
        timestepmax = timestepmin

    check_averaging_angles(average_over_phi, average_over_theta)

    specdata_alltimesteps: dict[int, pl.LazyFrame] = {}
    with suppress(FileNotFoundError):
        # the direction-resolved file must match the packet type of the spherically averaged one below,
        # otherwise the dirbins would silently hold UVOIR spectra while dirbin -1 holds gamma spectra
        res_specdata = read_spec_res(modelpath, gamma=gamma)
        if average_over_theta:
            res_specdata = average_direction_bins(res_specdata, overangle="theta")
        if average_over_phi:
            res_specdata = average_direction_bins(res_specdata, overangle="phi")
        specdata_alltimesteps |= res_specdata

    # spherically averaged spectra
    try:
        specdata_alltimesteps[-1] = read_spec(modelpath=modelpath, gamma=gamma)

    except FileNotFoundError as e:
        if gamma:
            msg = "ERROR: No spherically averaged gamma spectrum found."
            raise FileNotFoundError(msg) from e

    arr_tdelta = get_timestep_times(modelpath, loc="delta")
    specdataout: dict[int, pl.LazyFrame] = {}
    for dirbin in specdata_alltimesteps:
        dfspectrum = (
            specdata_alltimesteps[dirbin]
            .select(
                pl.col("nu"),
                (
                    pl.sum_horizontal(
                        cs.by_index(timestep + 1) * arr_tdelta[timestep]
                        for timestep in range(timestepmin, timestepmax + 1)
                    )
                    / sum(arr_tdelta[timestepmin : timestepmax + 1])
                ).alias("f_nu"),
            )
            .with_columns(lambda_angstroms=const.c_ang_per_s / pl.col("nu"))
        )

        if fluxfilterfunc:
            dfspectrum = dfspectrum.with_columns(
                cs.starts_with("f_nu").map_batches(fluxfilterfunc, return_dtype=pl.self_dtype())
            )

        specdataout[dirbin] = dfspectrum.with_columns(
            f_lambda=pl.col("f_nu") * pl.col("nu") / pl.col("lambda_angstroms")
        ).sort(by="nu" if gamma else "lambda_angstroms")

    if fluxfilterfunc:
        print("Applying filter to ARTIS spectrum")

    return specdataout


def make_virtual_spectra_summed_file(modelpath: Path | str) -> None:
    """Sum the per-rank virtual packet spectra into one vspecpol_total file per observer direction."""
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

        vspecpol_data_alldirs = drop_trailing_null_column(
            pl.read_csv(polars_source(vspecpolpath), separator=" ", has_header=False)
        )

        vspecpol_data = {k: v.collect() for k, v in split_multitable_dataframe(vspecpol_data_alldirs).items()}
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
        print_saved(outfile)


def make_averaged_vspecfiles(modelpaths: Sequence[Path]) -> None:
    """Average the vspecpol_total files of several models into one vspecpol_averaged file per observer direction."""
    filenames = [
        vspecfile.name for vspecfile in Path(modelpaths[0]).iterdir() if vspecfile.name.startswith("vspecpol_total-")
    ]

    def sorted_by_number(lst: list[str]) -> list[str]:
        def convert(text: str) -> int | str:
            return int(text) if text.isdigit() else text

        def alphanum_key(key: str) -> list[int | str]:
            return [convert(c) for c in re.split(r"([0-9]+)", key)]

        return sorted(lst, key=alphanum_key)

    filenames = sorted_by_number(filenames)

    for spec_index, filename in enumerate(filenames):  # vspecpol-total files
        vspecarrays = [read_wsv(modelpath / filename, has_header=False).to_numpy() for modelpath in modelpaths]
        averaged = vspecarrays[0].copy()
        # the first row (times) and first column (frequencies) are labels shared by all models, so average the rest
        averaged[1:, 1:] = np.mean([arr[1:, 1:] for arr in vspecarrays], axis=0)
        pl.DataFrame(averaged).write_csv(
            Path(modelpaths[0]) / f"vspecpol_averaged-{spec_index}.out", separator=" ", include_header=False
        )


def get_specpol_data(dirbin: int = -1, modelpath: Path | str | None = None) -> dict[str, pl.LazyFrame]:
    """Return the I, Q, and U spectra of one direction bin, read from specpol.out or specpol_res_<dirbin>.out."""
    assert modelpath is not None
    specfilename = (
        firstexisting("specpol.out", folder=modelpath, tryzipped=True)
        if dirbin == -1
        else firstexisting(f"specpol_res_{dirbin}.out", folder=modelpath, tryzipped=True)
    )

    print(f"Reading {specfilename}")
    specdata = drop_trailing_null_column(
        pl.scan_csv(polars_source(specfilename), separator=" ", has_header=True, infer_schema=False)
    ).with_columns(pl.all().cast(pl.Float64))

    return split_dataframe_stokesparams(specdata)


# maxsize is small because this reads eagerly and every cached entry retains a whole vspecpol_total file.
# Callers collect the frames once per timestep, so a cache miss on each call would parse the file again.
@lru_cache(maxsize=2)
def get_vspecpol_data(vspecindex: int, modelpath: Path | str) -> dict[str, pl.LazyFrame]:
    """Return the I, Q, and U virtual packet spectra of one observer, summing the per-rank files if needed.

    Callers must not mutate the returned dict, which is shared between calls.
    """
    assert modelpath is not None
    # alternatively use f'vspecpol_averaged-{angle}.out' ?

    try:
        specfilename = firstexisting(f"vspecpol_total-{vspecindex}.out", folder=modelpath, tryzipped=True)
    except FileNotFoundError:
        print(f"vspecpol_total-{vspecindex}.out does not exist. Generating all-rank summed vspec files..")
        make_virtual_spectra_summed_file(modelpath=modelpath)
        specfilename = firstexisting(f"vspecpol_total-{vspecindex}.out", folder=modelpath, tryzipped=True)

    print(f"Reading {specfilename}")
    specdata = pl.read_csv(polars_source(specfilename), separator=" ", has_header=True)

    return split_dataframe_stokesparams(specdata)


def split_dataframe_stokesparams(specdata: pl.DataFrame | pl.LazyFrame) -> dict[str, pl.LazyFrame]:
    """DataFrames read from specpol*.out and vspecpol*.out are repeated over I, Q, U parameters. Split these into a dictionary of DataFrames."""
    specdata = specdata.rename({specdata.collect_schema().names()[0]: "nu"}).lazy()
    stokes_params = {
        "I": specdata.select(cs.exclude(cs.contains("_duplicated_"))),
        "Q": specdata.select(
            pl.col("nu"), cs.ends_with("_duplicated_0").name.map(lambda x: x.removesuffix("_duplicated_0"))
        ),
        "U": specdata.select(
            pl.col("nu"), cs.ends_with("_duplicated_1").name.map(lambda x: x.removesuffix("_duplicated_1"))
        ),
    }

    stokes_params |= {
        f"{param}/I": stokes_params[param]
        .join(stokes_params["I"], on="nu", how="left", suffix="_I", maintain_order="left")
        .select(
            cs.by_name("nu"),
            *(
                pl.col(col) / pl.col(f"{col}_I")
                for col in stokes_params["I"].collect_schema().names()
                if col != "nu" and not col.endswith("_I")
            ),
        )
        for param in ("Q", "U")
    }

    return stokes_params


def get_vspecpol_spectrum(
    modelpath: Path | str,
    timeavg: float,
    angle: int,
    args: argparse.Namespace,
    fluxfilterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
) -> pl.LazyFrame:
    """Return the virtual packet spectrum of one observer, averaged over the timesteps around timeavg."""
    stokes_params = get_vspecpol_data(vspecindex=angle, modelpath=Path(modelpath))
    if "stokesparam" not in args:
        args.stokesparam = "I"
    vspecdata = stokes_params[args.stokesparam]

    arr_tmid = [float(i) for i in vspecdata.collect_schema().names()[1:]]
    arr_tdelta = [l1 - l2 for l1, l2 in zip(arr_tmid[1:], arr_tmid[:-1], strict=False)] + [arr_tmid[-1] - arr_tmid[-2]]

    if "timemin" in args and "timemax" in args and args.timemin is not None and args.timemax is not None:
        # how timemin, timemax are used changed at some point. to average over multiple timesteps needs to fix this
        timestepmin = arr_tmid.index(match_closest_time(args.timemin, arr_tmid))
        timestepmax = arr_tmid.index(match_closest_time(args.timemax, arr_tmid))
    else:
        timestepmin = arr_tmid.index(match_closest_time(timeavg, arr_tmid))
        timestepmax = timestepmin

    timelower = arr_tmid[timestepmin]
    timeupper = arr_tmid[timestepmax]
    print(f" vpacket spectrum timesteps {timestepmin} ({timelower}d) to {timestepmax} ({timeupper}d)")

    dfout = vspecdata.select(
        f_nu=(
            pl.sum_horizontal(
                pl.col(vspecdata.collect_schema().names()[timestep + 1]) * arr_tdelta[timestep]
                for timestep in range(timestepmin, timestepmax + 1)
            )
            / sum(arr_tdelta[timestepmin : timestepmax + 1])
        ),
        nu=pl.col("nu"),
    ).with_columns(lambda_angstroms=const.c_ang_per_s / pl.col("nu"))

    if fluxfilterfunc:
        print("Applying filter to ARTIS spectrum")
        dfout = dfout.with_columns(cs.starts_with("f_nu").map_batches(fluxfilterfunc, return_dtype=pl.self_dtype()))

    return dfout.with_columns(f_lambda=pl.col("f_nu") * pl.col("nu") / pl.col("lambda_angstroms")).sort(
        by="lambda_angstroms"
    )


def get_emabs_timeblock_count(dfemabs: pl.DataFrame, n_nu: int, n_timesteps: int, emabsfilename: str) -> int:
    """Get the number of time blocks per frequency bin in an emission or absorption file.

    These files store one row per (frequency, time) pair with time varying fastest, so this is the stride between
    consecutive rows of the same frequency bin. Polarisation files hold the I, Q, and U components, giving three time
    blocks per timestep instead of one.
    """
    nrows = dfemabs.height
    assert nrows % n_nu == 0, f"{emabsfilename}: row count {nrows} is not a multiple of the {n_nu} frequency bins"

    n_timeblocks = nrows // n_nu
    assert n_timeblocks in {n_timesteps, 3 * n_timesteps}, (
        f"{emabsfilename}: got {n_timeblocks} time blocks per frequency bin, expected {n_timesteps} timesteps"
        f" or {3 * n_timesteps} for a polarisation file"
    )

    return n_timeblocks


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
    lambda_min: float = 0.0,
    lambda_max: float = math.inf,
) -> tuple[list[FluxContributionTuple], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the per-ion emission and absorption contributions from emission.out, and the flux and wavelength arrays.

    The returned spectra are restricted to lambda_min to lambda_max [Å], so that the flux contributions used for
    ranking count only the plotted window, matching get_flux_contributions_from_packets.
    """
    arr_tmid = get_timestep_times(modelpath, loc="mid")
    arr_tdelta = get_timestep_times(modelpath, loc="delta")
    arraynu_full = get_nu_grid(modelpath)
    arraylambda_full = const.c_ang_per_s / arraynu_full
    nu_select = (arraylambda_full >= lambda_min) & (arraylambda_full <= lambda_max)
    arraynu = arraynu_full[nu_select]
    arraylambda = arraylambda_full[nu_select]
    if not Path(modelpath, "compositiondata.txt").is_file():
        print_warning("compositiondata.txt not found. Using output*.txt instead")
        from artistools.atomic import get_composition_data_from_outputfile

        elementlist = get_composition_data_from_outputfile(modelpath)
    else:
        from artistools.atomic import get_composition_data

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

    emissiondata: dict[int, pl.DataFrame] = {}
    absorptiondata: dict[int, pl.DataFrame] = {}
    # the row stride of each frame, derived from that frame alone so that emission and absorption (and separate
    # direction bins) can never pick up each other's value
    emission_timeblocks: dict[int, int] = {}
    absorption_timeblocks: dict[int, int] = {}
    maxion: int | None = None
    polarisation_notified = False

    for dbin in dbinlist:
        if getemission:
            emissionfilenames = ["emission.out", "emissionpol.out"] if use_lastemissiontype else ["emissiontrue.out"]

            if dbin != -1:
                emissionfilenames = [x.replace(".out", f"_res_{dbin:02d}.out") for x in emissionfilenames]

            emissionfilename = firstexisting(emissionfilenames, folder=modelpath, tryzipped=True)

            emissiondata[dbin] = read_emission_absorption_file(emissionfilename).collect()
            emission_timeblocks[dbin] = get_emabs_timeblock_count(
                emissiondata[dbin], len(arraynu_full), len(arr_tmid), str(emissionfilename)
            )

            if emission_timeblocks[dbin] > len(arr_tmid) and not polarisation_notified:
                print("This artis run contains polarisation data")
                polarisation_notified = True

            maxion_float = (
                (len(emissiondata[dbin].collect_schema().names()) - 1) / 2.0 / nelements
            )  # also known as MIONS in ARTIS sn3d.h
            assert maxion_float.is_integer()
            if maxion is None:
                maxion = int(maxion_float)
                print(
                    f" inferred MAXION = {maxion} from emission file using nlements = {nelements} from"
                    " compositiondata.txt"
                )
            else:
                assert maxion == int(maxion_float)

        if getabsorption:
            absorptionfilenames = ["absorption.out", "absorptionpol.out"]
            if dbin != -1:
                absorptionfilenames = [x.replace(".out", f"_res_{dbin:02d}.out") for x in absorptionfilenames]

            absorptionfilename = firstexisting(absorptionfilenames, folder=modelpath, tryzipped=True)

            absorptiondata[dbin] = read_emission_absorption_file(absorptionfilename).collect()
            absorption_timeblocks[dbin] = get_emabs_timeblock_count(
                absorptiondata[dbin], len(arraynu_full), len(arr_tmid), str(absorptionfilename)
            )

            if absorption_timeblocks[dbin] > len(arr_tmid) and not polarisation_notified:
                print("This artis run contains polarisation data")
                polarisation_notified = True

            absorption_maxion_float = len(absorptiondata[dbin].collect_schema().names()) / nelements
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

    array_flambda_emission_total = np.zeros_like(arraylambda, dtype=float)
    contribution_list = []
    if filterfunc:
        print("Applying filter to ARTIS spectrum")

    assert maxion is not None
    for elementindex in range(nelements):
        nions = elementlist["nions"][elementindex]
        for ion in range(nions):
            ion_stage = ion + elementlist["lowermost_ion_stage"][elementindex]
            ionserieslist: list[tuple[int, str]] = [
                (elementindex * maxion + ion, "bound-bound"),
                (nelements * maxion + elementindex * maxion + ion, "bound-free"),
            ]

            if elementindex == ion == 0:
                ionserieslist.append((2 * nelements * maxion, "free-free"))

            for selectedcolumn, emissiontypeclass in ionserieslist:
                if getemission:
                    array_fnu_emission = weighted_average_spectra([
                        (
                            emissiondata[dbin][timestep :: emission_timeblocks[dbin], selectedcolumn].to_numpy(),
                            arr_tdelta[timestep] / len(dbinlist),
                        )
                        for timestep in range(timestepmin, timestepmax + 1)
                        for dbin in dbinlist
                    ])[nu_select]
                else:
                    array_fnu_emission = np.zeros_like(arraylambda, dtype=float)

                if absorptiondata and selectedcolumn < nelements * maxion:  # bound-bound process
                    array_fnu_absorption = weighted_average_spectra([
                        (
                            absorptiondata[dbin][timestep :: absorption_timeblocks[dbin], selectedcolumn].to_numpy(),
                            arr_tdelta[timestep] / len(dbinlist),
                        )
                        for timestep in range(timestepmin, timestepmax + 1)
                        for dbin in dbinlist
                    ])[nu_select]
                else:
                    array_fnu_absorption = np.zeros_like(arraylambda, dtype=float)

                if filterfunc:
                    array_fnu_emission = filterfunc(array_fnu_emission)
                    if selectedcolumn < nelements * maxion:
                        array_fnu_absorption = filterfunc(array_fnu_absorption)

                array_flambda_emission = array_fnu_emission * arraynu / arraylambda
                array_flambda_absorption = array_fnu_absorption * arraynu / arraylambda

                array_flambda_emission_total += array_flambda_emission
                fluxcontribthisseries = abs(np.trapezoid(array_fnu_emission, x=arraynu)) + abs(
                    np.trapezoid(array_fnu_absorption, x=arraynu)
                )
                assert isinstance(fluxcontribthisseries, float)

                if emissiontypeclass == "bound-bound":
                    linelabel = get_ionstring(elementlist["Z"][elementindex], ion_stage)
                elif emissiontypeclass == "free-free":
                    linelabel = "free-free"
                else:
                    linelabel = f"{get_ionstring(elementlist['Z'][elementindex], ion_stage)} {emissiontypeclass}"

                contribution_list.append(
                    FluxContributionTuple(
                        fluxcontrib=fluxcontribthisseries,
                        linelabel=linelabel,
                        array_flambda_emission=array_flambda_emission,
                        array_flambda_absorption=array_flambda_absorption,
                        color=None,
                    )
                )

    return contribution_list, array_flambda_emission_total, arraylambda


def get_linelist_label_columns(modelpath: Path | str, groupby: str) -> pl.DataFrame:
    """Return the columns of each line that a group label needs. The rows are in lineindex order.

    The caller keeps this frame while it makes the labels. A linelist can have tens of millions of lines.
    Thus this frame can be hundreds of megabytes. The emission labels and the absorption labels both use it.
    """
    linecolumns = ["atomic_number", "ion_stage"]
    if groupby != "ion":
        linecolumns += ["lambda_angstroms_air", "upperlevelindex", "lowerlevelindex"]

    return get_linelist_pldf(modelpath=modelpath).select(linecolumns).collect()


def get_line_labels(dflines: pl.DataFrame, lineindices: pl.Series, groupby: str, labelcolumn: str) -> pl.LazyFrame:
    """Return a frame that gives the ion label or the line label of each supplied line index.

    A linelist can have tens of millions of lines. One spectrum uses only a small part of them.
    Thus the code gets the line data at the supplied indices. It does not join the packets to the full linelist.
    The dflines parameter comes from get_linelist_label_columns(). Its row position is the line index.
    """
    typecolumn = lineindices.name

    # A negative code is a free-free marker or a bound-free marker. For a negative index, gather() gets a row at the
    # end of the linelist. For an index after the last row, gather() makes an error. Remove both types of index here.
    # The join of the caller then finds no label for these codes.
    lineindices = lineindices.filter(lineindices.is_between(0, dflines.height - 1)).unique()

    return add_ion_str_column(
        pl.LazyFrame({typecolumn: lineindices.cast(pl.Int32)}).select(
            typecolumn, *[pl.lit(dflines[col]).gather(pl.col(typecolumn)).alias(col) for col in dflines.columns]
        )
    ).select(
        typecolumn,
        pl.col("ion_str").alias(labelcolumn)
        if groupby == "ion"
        else pl.format(
            "{} λ{} {}-{}",
            pl.col("ion_str"),
            pl.col("lambda_angstroms_air").sub(0.5).round(0).cast(pl.String).str.strip_suffix(".0"),
            pl.col("upperlevelindex"),
            pl.col("lowerlevelindex"),
        ).alias(labelcolumn),
    )


def get_flux_contributions_from_packets(
    modelpath: Path,
    timelowdays: float,
    timehighdays: float,
    lambda_bin_edges: npt.NDArray[np.floating],
    getemission: bool = True,
    getabsorption: bool = True,
    maxpacketfiles: int | None = None,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    groupby: str = "ion",
    maxseriescount: int | None = None,
    fixedionlist: list[str] | None = None,
    use_time: t.Literal["arrival", "emission", "escape"] = "arrival",
    emtypecolumn: str | None = None,
    directionbin: int | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    directionbins_are_vpkt_observers: bool = False,
    vpkt_match_emission_exclusion_to_opac: bool = False,
    gamma: bool = False,
) -> tuple[list[FluxContributionTuple], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the emission and absorption contributions binned from the packets, and the flux and wavelength arrays.

    groupby selects whether the packets are grouped by ion, line, nuclide, or nuclide mass.
    """
    assert groupby in {"ion", "line", "nuc", "nucmass"}
    assert use_time in {"arrival", "emission", "escape"}
    assert emtypecolumn in {"emissiontype", "trueemissiontype", "pellet_nucindex"}
    if getabsorption and groupby == "nuc":
        # A nuclide emits a packet, but a nuclide does not absorb a packet.
        # Thus a nuclide name cannot be a label for an absorption contribution.
        msg = (
            "Absorption contributions cannot be grouped by nuclide. Use -groupby ion or line, or drop --showabsorption"
        )
        raise ValueError(msg)

    if groupby == "line":
        print(
            "Grouping by line. Line labels are wavelengths in air between 2,000-20,000 Å, and vacuum wavelengths outside this range. This matches the NIST default options and many astrophysics papers."
        )

    if gamma:
        assert groupby in {"nuc", "nucmass"}
        assert emtypecolumn == "pellet_nucindex"

    if directionbins_are_vpkt_observers and use_time != "arrival":
        msg = "Virtual packet contributions support only observer arrival time"
        raise ValueError(msg)

    if directionbin is None:
        directionbin = -1

    energy_column = "e_cmf" if use_time == "escape" else "e_rf"
    cols = {energy_column}

    nu_min = const.c_ang_per_s / lambda_bin_edges[-1]
    nu_max = const.c_ang_per_s / lambda_bin_edges[0]

    vpkt_config = None
    opacchoiceindex = None
    if directionbins_are_vpkt_observers:
        vpkt_config = get_vpkt_config(modelpath)
        obsdirindex, opacchoiceindex = divmod(directionbin, vpkt_config["nspectraperobs"])
        nprocs_read, lzdfpackets = atpackets.get_virtual_packets(modelpath, maxpacketfiles=maxpacketfiles)
        lzdfpackets = lzdfpackets.with_columns(e_rf=pl.col(f"dir{obsdirindex}_e_rf_{opacchoiceindex}"))
        dirbin_nu_column = f"dir{obsdirindex}_nu_rf"

        cols |= {dirbin_nu_column, f"dir{obsdirindex}_t_arrive_d", f"dir{obsdirindex}_e_rf_{opacchoiceindex}"}
        lzdfpackets = lzdfpackets.filter(pl.col(f"dir{obsdirindex}_t_arrive_d").is_between(timelowdays, timehighdays))

    else:
        nprocs_read, lzdfpackets = atpackets.get_packets(
            modelpath,
            maxpacketfiles=maxpacketfiles,
            packet_type="TYPE_ESCAPE",
            escape_type="TYPE_GAMMA" if gamma else "TYPE_RPKT",
        )
        dirbin_nu_column = "nu_rf"

        lzdfpackets, _ = filter_packets_by_time(lzdfpackets, modelpath, timelowdays, timehighdays, use_time, gamma)

        lzdfpackets, _ = atpackets.filter_packets_dirbin(
            lzdfpackets, directionbin, average_over_phi=average_over_phi, average_over_theta=average_over_theta
        )

    condition_nu_emit = pl.col(dirbin_nu_column).is_between(nu_min, nu_max) if getemission else pl.lit(value=False)
    condition_nu_abs = pl.col("absorption_freq").is_between(nu_min, nu_max) if getabsorption else pl.lit(value=False)
    lzdfpackets = lzdfpackets.filter(condition_nu_emit | condition_nu_abs)

    if getemission:
        cols |= {emtypecolumn, dirbin_nu_column}

    if getabsorption:
        cols |= {"absorption_type", "absorption_freq"}

    if directionbin != -1:
        if average_over_phi:
            cols.add("costhetabin")
        elif average_over_theta:
            cols.add("phibin")
        else:
            cols.add("dirbin")

    dfpackets = lzdfpackets.select(cs.by_name(cols, require_all=False)).collect()

    # The code reads these columns one time. The emission labels and the absorption labels both use them.
    # The memory becomes free when this function returns. Each absorption label is a line label.
    # Thus only an emission-only plot with a nuclide group can omit the linelist.
    needs_linelist = getabsorption or (getemission and groupby in {"ion", "line"})
    dflines = get_linelist_label_columns(modelpath, groupby) if needs_linelist else pl.DataFrame()

    # The code adds the labels after it collects the packets. Thus it finds a label only for a code that a packet uses.
    if getemission:
        if groupby == "nuc":
            emtypelabels = get_nuclides(modelpath=modelpath).rename({"nucname": "emissiontype_str"})
        elif groupby == "nucmass":
            emtypelabels = get_nuclides(modelpath=modelpath).with_columns(
                (
                    pl.when(pl.col("pellet_nucindex") == -1).then("nucname").otherwise(pl.format("A={}", pl.col("A")))
                ).alias("emissiontype_str")
            )
        else:
            expr_bflist_to_str = (
                pl.col("ion_str") + " bound-free"
                if groupby == "ion"
                else pl.format("{} bound-free {}-{}", pl.col("ion_str"), pl.col("lowerlevel"), pl.col("upperionlevel"))
            )

            emtypelabels = pl.concat([
                get_line_labels(dflines, dfpackets[emtypecolumn], groupby, "emissiontype_str"),
                pl.LazyFrame(
                    {emtypecolumn: [-9999999, -9999000], "emissiontype_str": ["free-free", "NOT SET"]},
                    schema={emtypecolumn: pl.Int32, "emissiontype_str": pl.String},
                    orient="col",
                ),
                get_bflist(modelpath).select(
                    (-1 - pl.col("bfindex").cast(pl.Int32)).alias(emtypecolumn),
                    expr_bflist_to_str.alias("emissiontype_str"),
                ),
            ])

        # Select only the key column and the label column.
        # The nuclide table has more columns. Without this selection, each packet gets those columns.
        dfpackets = dfpackets.join(
            emtypelabels.select(emtypecolumn, "emissiontype_str").collect(),
            on=emtypecolumn,
            how="left",
            maintain_order="left",
        ).drop(emtypecolumn)

        if vpkt_match_emission_exclusion_to_opac and directionbins_are_vpkt_observers:
            assert vpkt_config is not None
            assert opacchoiceindex is not None
            z_exclude = int(vpkt_config["z_excludelist"][opacchoiceindex])
            if z_exclude == -1:
                # no bound-bound
                dfpackets = dfpackets.filter(pl.col("emissiontype_str").str.contains("bound-free"))
            elif z_exclude == -2:
                # no bound-free
                dfpackets = dfpackets.filter(pl.col("emissiontype_str").str.contains("bound-free").not_())
            elif z_exclude > 0:
                elsymb = get_elsymbol(z_exclude)
                dfpackets = dfpackets.filter(pl.col("emissiontype_str").str.starts_with(f"{elsymb} ").not_())

    if getabsorption:
        abstypelabels = pl.concat([
            get_line_labels(dflines, dfpackets["absorption_type"], groupby, "absorptiontype_str"),
            pl.LazyFrame(
                {"absorption_type": [-1, -2], "absorptiontype_str": ["free-free", "bound-free"]},
                schema={"absorption_type": pl.Int32, "absorptiontype_str": pl.String},
                orient="col",
            ),
        ])

        dfpackets = dfpackets.join(
            abstypelabels.collect(), on="absorption_type", how="left", maintain_order="left"
        ).drop("absorption_type")

    # The label column and the frequency column of each type of contribution.
    # When the code bins one type, it removes the columns of the other type.
    emission_columns = ("emissiontype_str", dirbin_nu_column)
    absorption_columns = ("absorptiontype_str", "absorption_freq")

    # The dfpackets frame is a parameter and not a captured variable. The code deletes that variable below.
    # The deletion makes the memory free before the code bins the groups.
    def group_by_label(dfpkts: pl.DataFrame, keep: tuple[str, str], drop: tuple[str, str]) -> dict[str, pl.DataFrame]:
        """Divide the packets into one frame for each label. Keep only the columns that the bin operation needs."""
        labelcolumn, nucolumn = keep
        # partition_by() copies each group into new memory. Thus the memory of the intermediate frame becomes free.
        return {
            groupname: dfgroup
            for (groupname,), dfgroup in (
                dfpkts
                .drop(drop, strict=False)
                .filter(pl.col(nucolumn).is_between(nu_min, nu_max) & pl.col(labelcolumn).is_not_null())
                .partition_by(labelcolumn, include_key=False, as_dict=True)
            ).items()
        }

    # These are two different dictionaries and not one shared empty dictionary.
    # The "Other" group operation below changes them.
    emissiongroups: dict[str, pl.DataFrame] = {}
    absorptiongroups: dict[str, pl.DataFrame] = {}
    if getemission:
        emissiongroups = group_by_label(dfpackets, emission_columns, absorption_columns)
    if getabsorption:
        absorptiongroups = group_by_label(dfpackets, absorption_columns, emission_columns)

    del dfpackets, dflines

    group_energy_sum: dict[str, float] = {}
    for groups in (emissiongroups, absorptiongroups):
        for groupname, dfgroup in groups.items():
            group_energy_sum[groupname] = group_energy_sum.get(groupname, 0.0) + float(dfgroup[energy_column].sum())

    allgroupnames = list(group_energy_sum)

    if fixedionlist is not None and (unrecognised_items := [x for x in fixedionlist if x not in allgroupnames]):
        print_warning(f"(packets) did not find {len(unrecognised_items)} items in fixedionlist: {unrecognised_items}")

    def sortkey(groupname: str) -> tuple[int, float | int]:
        grouptotal = group_energy_sum[groupname]

        if fixedionlist is None:
            return (0, -grouptotal)

        return (
            (fixedionlist.index(groupname), 0.0) if groupname in fixedionlist else (len(fixedionlist) + 1, -grouptotal)
        )

    # group small contributions together to avoid the cost of binning individual spectra for them

    allgroupnames.sort(key=sortkey)

    if maxseriescount is None:
        maxseriescount = len(allgroupnames)
    if len(allgroupnames) > maxseriescount:
        other_groupnames = allgroupnames[maxseriescount:]
        allgroupnames = [*allgroupnames[:maxseriescount], "Other"]

        # a group name can be present for only one of emission and absorption (e.g. "Fe II bound-free" is never an
        # absorption label), so each dict is combined independently and may get no contributions at all
        for groups, getthis in ((emissiongroups, getemission), (absorptiongroups, getabsorption)):
            if not getthis or not groups:
                continue
            other_subgroups = [groups[groupname] for groupname in other_groupnames if groupname in groups]
            groups["Other"] = (
                pl.concat(other_subgroups, rechunk=False)
                if other_subgroups
                else pl.DataFrame(schema=next(iter(groups.values())).schema)
            )

            for groupname in other_groupnames:
                groups.pop(groupname, None)

    array_flambda_emission_total = None
    contribution_list = []
    # These are the bin centres of each group spectrum. An empty selection thus also gives the correct axis.
    array_lambda = get_binned_lambda_frame(lambda_bin_edges).select("lambda_angstroms").collect().to_series().to_numpy()

    def group_spectra(groups: dict[str, pl.DataFrame], dirbin: int, **extraargs: t.Any) -> dict[str, pl.DataFrame]:
        """Return the binned spectrum of each group, collecting every group in one pass."""
        return dict(
            zip(
                groups.keys(),
                pl.collect_all([
                    get_from_packets(
                        modelpath=modelpath,
                        timelowdays=timelowdays,
                        timehighdays=timehighdays,
                        lambda_bin_edges=lambda_bin_edges,
                        use_time=use_time,
                        fluxfilterfunc=filterfunc,
                        nprocs_read_dfpackets=(nprocs_read, dfpkts),
                        directionbins_are_vpkt_observers=directionbins_are_vpkt_observers,
                        directionbins=[dirbin],
                        average_over_phi=average_over_phi,
                        average_over_theta=average_over_theta,
                        gamma=gamma,
                        packets_are_time_filtered=True,
                        **extraargs,
                    )[dirbin].select("lambda_angstroms", "f_lambda")
                    for dfpkts in groups.values()
                ]),
                strict=True,
            )
        )

    group_em_specs = group_spectra(emissiongroups, directionbin)
    group_abs_specs = group_spectra(absorptiongroups, directionbin, nu_column="absorption_freq")
    for groupname in allgroupnames:
        array_flambda_emission = (
            group_em_specs[groupname]["f_lambda"].to_numpy()
            if groupname in group_em_specs
            else np.zeros_like(array_lambda, dtype=float)
        )
        array_flambda_absorption = (
            group_abs_specs[groupname]["f_lambda"].to_numpy()
            if groupname in group_abs_specs
            else np.zeros_like(array_lambda, dtype=float)
        )

        if groupname in group_em_specs:
            if array_flambda_emission_total is None:
                array_flambda_emission_total = array_flambda_emission.copy()
            else:
                array_flambda_emission_total += array_flambda_emission

        fluxcontribthisseries = abs(float(np.trapezoid(array_flambda_emission, x=array_lambda))) + abs(
            float(np.trapezoid(array_flambda_absorption, x=array_lambda))
        )

        if fluxcontribthisseries > 0.0:
            contribution_list.append(
                FluxContributionTuple(
                    fluxcontrib=fluxcontribthisseries,
                    linelabel=groupname,
                    array_flambda_emission=array_flambda_emission,
                    array_flambda_absorption=array_flambda_absorption,
                    color=None,
                )
            )

    if array_flambda_emission_total is None:
        array_flambda_emission_total = np.zeros_like(array_lambda, dtype=float)

    return contribution_list, array_flambda_emission_total, array_lambda


def sort_and_reduce_flux_contribution_list(
    contribution_list_in: list[FluxContributionTuple],
    maxseriescount: int,
    arraylambda_angstroms: npt.NDArray[np.floating],
    fixedionlist: list[str] | None = None,
    hideother: bool = False,
) -> list[FluxContributionTuple]:
    """Return the contributions sorted by flux, keeping at most maxseriescount and merging the rest into 'Other'."""
    if fixedionlist:
        if unrecognised_items := [x for x in fixedionlist if x not in [y.linelabel for y in contribution_list_in]]:
            print_warning(f"did not understand these items in fixedionlist: {unrecognised_items}")

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

    import matplotlib.pyplot as plt

    from artistools.plottools import glasbey_category20_nogreys
    from artistools.plottools import remove_greys

    tab20_rgba = np.asarray(plt.get_cmap("tab20")(np.linspace(0, 1.0, 20)))
    rgb_candidates: list[mplt.ColorType] = [(float(r), float(g), float(b)) for r, g, b, _a in tab20_rgba]
    # the first ten glasbey colours repeat the tab10 colours that tab20 already supplies, and skipping
    # them keeps the order of every series that a published figure already used
    rgb_candidates.extend(glasbey_category20_nogreys[10:])

    color_list: list[mplt.ColorType] = remove_greys(rgb_candidates)

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
            integemiss = abs(np.trapezoid(row.array_flambda_emission, x=arraylambda_angstroms))
            integabsorp = abs(np.trapezoid(-row.array_flambda_absorption, x=arraylambda_angstroms))
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
    """Print and return the flux integrated over the given x range [erg/s/cm2 at 1 Mpc]."""
    integrated_flux = abs(np.trapezoid(np.nan_to_num(arr_df_on_dx, nan=0.0), x=arr_x))
    x_min = arr_x.min()
    x_max = arr_x.max()
    assert isinstance(x_min, int | float)
    assert isinstance(x_max, int | float)

    # x is the name of the axis in the code, and the line below it names the unit of that axis
    print_detail(f"integrated flux ({x_min:.1f} to {x_max:.1f}): {integrated_flux:.3e} erg/s/cm2 at 1 Mpc")
    assert isinstance(integrated_flux, float)
    return integrated_flux


def get_reference_spectrum(filepath: Path | str) -> pl.DataFrame:
    """Return an observed reference spectrum, applying any scaling and time shift from its metadata."""
    metadata = get_file_metadata(filepath)

    flambdaindex = metadata.get("f_lambda_columnindex", 1)

    specdata = read_wsv(filepath, has_header=False, comment_prefix="#").select(
        cs.by_index(0).alias("lambda_angstroms"), cs.by_index(flambdaindex).alias("f_lambda")
    )

    if "a_v" in metadata and "r_v" in metadata:
        from extinction import apply
        from extinction import ccm89

        specdata = specdata.with_columns(
            f_lambda=apply(
                ccm89(
                    specdata["lambda_angstroms"].to_numpy(writable=True),
                    a_v=-metadata["a_v"],
                    r_v=metadata["r_v"],
                    unit="aa",
                ),
                specdata["f_lambda"].to_numpy(),
            )
        )
        print(
            f"Correcting for reddening using CCM89 law with A_V = {metadata['a_v']} and R_V = {metadata.get('r_v', 3.1)}"
        )

    if "z" in metadata:
        specdata = specdata.with_columns(lambda_angstroms=pl.col("lambda_angstroms") / (1 + metadata["z"]))
        print(f"Correcting for redshift z = {metadata['z']}")

    return specdata
