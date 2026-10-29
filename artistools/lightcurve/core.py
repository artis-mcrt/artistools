"""Read ARTIS light curves and reference observational light curves, and derive band magnitudes from spectra."""

import io
import math
import typing as t
import warnings
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.atomic import scan_nuclides
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import H0_km_per_s_per_mpc
from artistools.constants import km_to_cm
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import Mbol_sun
from artistools.constants import megaparsec_to_cm
from artistools.constants import omega_matter
from artistools.misc import average_direction_bins
from artistools.misc import check_averaging_angles
from artistools.misc import df_filter_minmax_bracketed
from artistools.misc import exit_with_error
from artistools.misc import find_reference_data_file
from artistools.misc import firstexisting
from artistools.misc import firstexisting_or_none
from artistools.misc import get_file_metadata
from artistools.misc import get_metadata_label
from artistools.misc import get_metadata_path
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import get_vpkt_config
from artistools.misc import on_model_host
from artistools.misc import parse_metadata_redshift
from artistools.misc import path_is_reference_data
from artistools.misc import polars_error_note
from artistools.misc import print_warning
from artistools.misc import read_complete_lines
from artistools.misc import read_wsv
from artistools.misc import require_reference_data_file
from artistools.misc import scan_timesteps
from artistools.misc import split_multitable_dataframe
from artistools.misc import warn_direction_bin_noise
from artistools.misc import zopen
from artistools.packets import scan_packets
from artistools.packets import scan_virtual_packets
from artistools.packets import sum_packets_by_dirbin
from artistools.packets import sum_virtual_packets_by_observer
from artistools.paths import get_path
from artistools.spectra import get_escape_surface_gamma
from artistools.spectra import scan_spectra
from artistools.spectra import scan_spectrum_at_time
from artistools.spectra import scan_vspecpol_data

# ARTIS writes the Sloan filters with a trailing "s"; map them back to the conventional single-letter names
FILTERNAME_ALIASES: t.Final[Mapping[str, str]] = MappingProxyType({"rs": "r", "gs": "g", "is": "i", "zs": "z"})


def get_filename_part(name: str) -> str:
    """Return a band name or a model name in a form that an output file name can hold.

    A filter of an instrument has a name with a folder, e.g. NOT/B. A slash in a file name names a folder that does not
    exist, thus the slash becomes an underscore.
    """
    return name.replace("/", "_")


def derived_lum_unit_cols() -> list[pl.Expr]:
    """Return the expressions adding the bolometric magnitude and erg/s columns to a luminosity in Lsun.

    Every loader must produce the same set of unit columns, since the plotting code selects one of them by name.
    """
    return [
        (Mbol_sun - (2.5 * pl.col("luminosity_Lsun").log10())).alias("mag"),
        (cs.ends_with("_Lsun") * Lsun_to_erg_per_s).name.replace("_Lsun", "_erg/s"),
    ]


def lum_lsun_to_mag(lum_lsun: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Return the bolometric magnitude of a luminosity in solar luminosities.

    A zero or negative luminosity has no magnitude, so it becomes inf or nan rather than warning.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return Mbol_sun - (2.5 * np.log10(lum_lsun))


def get_katz_integral(time_days: pl.Series, luminosity_erg_per_s: pl.Series) -> tuple[float, float]:
    """Return the integral of t L(t) dt in erg s with the trapezoid rule, and the days that it leaves out.

    A null or NaN luminosity is not a zero, thus the integral leaves out each trapezoid with such an end. Katz B.,
    Kushnir D., Dong S. (2013), arXiv:1301.6766, compare this integral with the same integral of the deposition rate.
    """
    time = time_days.cast(pl.Float64).to_numpy()
    integrand = luminosity_erg_per_s.cast(pl.Float64).fill_null(np.nan).to_numpy() * time * day_to_s
    segments = 0.5 * (integrand[:-1] + integrand[1:]) * np.diff(time) * day_to_s
    isfinitesegment = np.isfinite(segments)
    return float(segments[isfinitesegment].sum()), float(np.abs(np.diff(time))[~isfinitesegment].sum())


@on_model_host
def scan_lightcurve(
    filepath: Path, *, directionresolved: bool = False, average_over_phi: bool = False, average_over_theta: bool = False
) -> dict[int, pl.LazyFrame]:
    """Return a LazyFrame of a light curve file for each direction bin, with an optional average over phi or theta.

    A direction-resolved file, e.g. light_curve_res.out, holds one table for each direction bin. A different file
    holds the angle average as bin -1. The caller knows which file it reads, thus the name of the file decides nothing.
    The angle average of bin -1 is the mean over every direction already, thus it takes no average over phi or theta.

    The averaging belongs here rather than to the caller because the magnitude column is not linear in the
    bin contributions. The magnitude comes after the averaging, thus no plot shows the mean of the magnitudes.
    In that mean, a single dark bin sends the whole averaged bin to inf.
    """
    check_averaging_angles(average_over_phi, average_over_theta)
    print(f"Reading {filepath}")
    # sn3d writes the whole file again at each timestep, thus the file can be empty or end with a cut line
    lcbytes = read_complete_lines(filepath, content="light curve")
    with polars_error_note(Path(filepath)):
        dflc = pl.read_csv(
            io.BytesIO(lcbytes),
            separator=" ",
            has_header=False,
            new_columns=["time_days", "luminosity_Lsun", "luminosity_cmf_Lsun"],
            # ARTIS writes 0.0 as 0, thus 100 rows of zero at the start inferred an integer column, and a later
            # value such as 1.5e+07 then stopped the read
            schema_overrides={
                "time_days": pl.Float64,
                "luminosity_Lsun": pl.Float64,
                "luminosity_cmf_Lsun": pl.Float64,
            },
        )
    try:
        lcdata = split_multitable_dataframe(dflc)
    except ValueError as exc:
        exc.add_note(f"while reading {filepath}")
        raise

    # a user can name the file, thus an angle-averaged light curve must not hold the tables of the direction bins.
    # An old light_curve.out holds a second table, which repeats the times. A build of ARTIS with one or two
    # direction bins writes as few tables, thus the count check below gives a warning and no error for that case
    if not directionresolved:
        if len(lcdata) > 2:
            msg = f"{filepath} holds {len(lcdata)} tables, thus it is a direction-resolved light curve"
            raise ValueError(msg)

        return {-1: lcdata[0].with_columns(derived_lum_unit_cols())}

    # ARTIS sets MABINS when it compiles, thus a different build can write a different count of direction bins
    if len(lcdata) != get_viewingdirectionbincount():
        print_warning(
            f"{filepath} holds {len(lcdata)} tables, and artistools gives the angles of"
            f" {get_viewingdirectionbincount()} direction bins. Thus the angle of a direction bin can be incorrect"
        )

    if average_over_phi:
        lcdata = average_direction_bins(lcdata, overangle="phi")

    if average_over_theta:
        lcdata = average_direction_bins(lcdata, overangle="theta")

    # after the averaging, never before it: a magnitude is not linear in the bin contributions, so the mean
    # of the magnitudes of the bins is not the magnitude of their mean luminosity
    unitcols = derived_lum_unit_cols()

    return {dirbin: lzdf.with_columns(unitcols) for dirbin, lzdf in lcdata.items()}


def select_timesteps_inside_vpkt_window(dftimesteps: pl.DataFrame, vpkt_config: Mapping[str, t.Any]) -> pl.DataFrame:
    """Return the timesteps that lie fully inside the time window of the virtual packets in vpkt.txt.

    ARTIS writes a virtual packet only for an arrival time inside the window. A timestep outside the window thus
    gave zero luminosity, and a timestep across an edge gave a low luminosity. With no time override, vpkt.txt takes
    the window of the constants of the ARTIS build. The file does not give that window, thus each timestep stays.
    """
    if not vpkt_config["time_limits_enabled"]:
        return dftimesteps

    windowstart, windowend = vpkt_config["initial_time"], vpkt_config["final_time"]
    # a window can start or stop at the edge of a timestep, thus a rounded edge must not drop that timestep
    tolerance = 1e-6 * windowend
    dfinside = dftimesteps.filter(
        (pl.col("tstart_days") >= windowstart - tolerance)
        & (pl.col("tstart_days") + pl.col("twidth_days") <= windowend + tolerance)
    )
    if dfinside.is_empty():
        msg = (
            f"No selected timestep lies fully inside the time window of the virtual packets, {windowstart} to"
            f" {windowend} days. ARTIS writes a virtual packet only inside this window of vpkt.txt"
        )
        raise ValueError(msg)
    if dfinside.height < dftimesteps.height:
        print_warning(
            f"The light curve of the virtual packets leaves out {dftimesteps.height - dfinside.height} timesteps,"
            f" because they are not fully inside the time window {windowstart} to {windowend} days of vpkt.txt"
        )

    return dfinside


@on_model_host
def scan_lightcurve_from_packets(
    modelpath: str | Path,
    escape_type: str = "TYPE_RPKT",
    maxpacketfiles: int | None = None,
    directionbins: Collection[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    directionbins_are_vpkt_observers: bool = False,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
    timedaysmin: float | None = None,
    timedaysmax: float | None = None,
) -> dict[int, pl.LazyFrame]:
    """Get ARTIS luminosity vs time from packets files."""
    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)
    if directionbins is None:
        directionbins = [-1]
    if directionbins_are_vpkt_observers and (pellet_nucname is not None or use_pellet_decay_time):
        msg = "The virtual packets hold no pellet, thus an observer has no light curve of a nuclide or of a decay time"
        raise ValueError(msg)

    dftimesteps_selected = df_filter_minmax_bracketed(
        scan_timesteps(Path(modelpath)), "tmid_days", timedaysmin, timedaysmax
    ).collect()

    vpkt_config = get_vpkt_config(Path(modelpath)) if directionbins_are_vpkt_observers else None
    if vpkt_config is not None:
        dftimesteps_selected = select_timesteps_inside_vpkt_window(dftimesteps_selected, vpkt_config)

    timebinstarts_plusend = [
        *dftimesteps_selected["tstart_days"],
        dftimesteps_selected.select(pl.col("tstart_days").last() + pl.col("twidth_days").last()).item(),
    ]

    if directionbins_are_vpkt_observers:
        nprocs_read, dfpackets = scan_virtual_packets(modelpath, maxpacketfiles=maxpacketfiles)
    else:
        nprocs_read, dfpackets = scan_packets(
            modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type=escape_type
        )
        # the comoving frame needs the velocity of the model surface. A copy of a model folder can leave out a large
        # model.txt, and the rest frame light curve needs no model.txt
        try:
            escapesurfacegamma = get_escape_surface_gamma(modelpath)
        except FileNotFoundError:
            print_warning(f"{modelpath} holds no model.txt, thus the light curve has no comoving frame luminosity")
        else:
            # the escape time multiplied by the Lorentz factor at the model surface gives the comoving-frame arrival
            # time
            dfpackets = dfpackets.with_columns(t_arrive_cmf_d=pl.col("escape_time") * escapesurfacegamma / day_to_s)

    if pellet_nucname is not None:
        atomic_number = get_atomic_number(pellet_nucname)
        if get_elsymbol(atomic_number) == pellet_nucname:
            expr = pl.col("atomic_number") == atomic_number
        else:
            expr = pl.col("nucname") == pellet_nucname
        dfpackets = dfpackets.filter(
            pl.col("pellet_nucindex").is_in(
                scan_nuclides(modelpath=Path(modelpath))
                .filter(expr)
                .select("pellet_nucindex")
                .collect()
                .to_series()
                .implode()
            )
        )

    if use_pellet_decay_time:
        dfpackets = dfpackets.with_columns([(pl.col("tdecay") / day_to_s).alias("tdecay_d")])

    timecol = "tdecay_d" if use_pellet_decay_time else "t_arrive_d"

    if directionbins_are_vpkt_observers:
        assert vpkt_config is not None
        rfsums = sum_virtual_packets_by_observer(
            dfpackets,
            list(directionbins),
            vpkt_config,
            lambda obsdirindex: pl.col(f"dir{obsdirindex}_t_arrive_d"),
            timebinstarts_plusend,
            sumsquares=True,
        )
        cmfsums = None
    else:
        rfsums = sum_packets_by_dirbin(
            dfpackets,
            list(directionbins),
            timecol,
            timebinstarts_plusend,
            "e_rf",
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
            sumsquares=True,
        )
        cmfsums = (
            sum_packets_by_dirbin(
                dfpackets,
                list(directionbins),
                "t_arrive_cmf_d",
                timebinstarts_plusend,
                "e_cmf",
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
            )
            if "t_arrive_cmf_d" in dfpackets.collect_schema().names()
            else None
        )

    # the luminosity in Lsun of each timestep is the packet energy times this factor and the solid-angle factor
    lumfactors = 1.0 / (nprocs_read * dftimesteps_selected["twidth_days"].to_numpy() * day_to_s * Lsun_to_erg_per_s)
    dftimes = dftimesteps_selected.select(pl.col("timestep").cast(pl.Int32), time_days=pl.col("tmid_days"))
    lcdata: dict[int, pl.LazyFrame] = {}
    for dirbin, sums in rfsums.items():
        assert sums.weightsquaresums is not None
        # relnoise = sqrt(sum(e^2)) / sum(e) is the relative Monte Carlo noise of a timestep with packets
        relnoise = np.sqrt(sums.weightsquaresums) / np.where(sums.weightsums > 0.0, sums.weightsums, np.nan)
        dflc = dftimes.select(
            pl.Series("packetcount", sums.packetcounts),
            pl.all(),
            pl.Series("luminosity_Lsun", sums.weightsums * sums.solidanglefactor * lumfactors),
            pl.Series("relnoise", relnoise).fill_nan(None),
        )
        if cmfsums is not None:
            dflc = dflc.with_columns(
                pl.Series("luminosity_cmf_Lsun", cmfsums[dirbin].weightsums * sums.solidanglefactor * lumfactors)
            )
        lcdata[dirbin] = dflc.lazy().with_columns(derived_lum_unit_cols())

    return lcdata


@on_model_host
def get_top_nuclides(
    modelpath: Path,
    escape_type: str,
    count: int,
    maxpacketfiles: int | None = None,
    timemindays: float | None = None,
    timemaxdays: float | None = None,
    *,
    use_pellet_decay_time: bool = False,
) -> pl.DataFrame:
    """Return the nuclides whose decays give the most energy to the escaped packets, with the energy of each one.

    The host of a remote model reads the packets, and only the table of the nuclides comes back.
    """
    dfnuclides = scan_nuclides(modelpath=modelpath)
    _, dfpackets = scan_packets(modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type=escape_type)
    return (
        df_filter_minmax_bracketed(
            dfpackets.with_columns(tdecay_d=pl.col("tdecay") / day_to_s),
            "tdecay_d" if use_pellet_decay_time else "t_arrive_d",
            timemindays,
            timemaxdays,
        )
        .group_by("pellet_nucindex")
        .agg(pl.sum("e_rf").alias("e_rf_sum"))
        .top_k(by="e_rf_sum", k=count)
        .sort("e_rf_sum", descending=True)
        .join(dfnuclides, on="pellet_nucindex", how="left", maintain_order="left")
        .select(["e_rf_sum", "nucname", "pellet_nucindex"])
        .collect()
    )


@on_model_host
def generate_band_lightcurve_data(
    modelpath: Path,
    filternames: Sequence[str],
    *,
    dirbin: int = -1,
    vpkt_observers: bool = False,
    timemin: float | None = None,
    timemax: float | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    """Integrate the spectra to get the band magnitudes against time.

    The method comes from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py. With vpkt_observers, a direction
    bin names a virtual packet observer of vpkt.txt. The name "bol" gives the bolometric magnitude.
    """
    # scan_spectrum_at_time reads the angle average of bin -1 from spec.out, thus its times come from there too.
    # A model with virtual packets can have no specpol.out
    isvpktobserver = vpkt_observers and dirbin >= 0 and Path(modelpath, "vpkt.txt").is_file()
    if isvpktobserver:
        print("Found vpkt.txt, using virtual packets")
        vspecdata = scan_vspecpol_data(vspecindex=dirbin, modelpath=modelpath)["I"]
        timearray = vspecdata.collect_schema().names()[1:]
    else:
        specfilename = (
            firstexisting_or_none(["specpol_res.out", "spec_res.out"], folder=modelpath, tryzipped=True)
            if dirbin >= 0
            else None
        )
        if specfilename is None:
            if dirbin >= 0:
                print_warning("No direction-resolved spectra available. Using angle-averaged spectra.")
            specfilename = firstexisting(["spec.out", "specpol.out"], folder=modelpath, tryzipped=True)

        with zopen(specfilename) as fspec:
            # pol and res files repeat the time columns for the Stokes Q and U blocks, so keep the first of each
            timearray = list(dict.fromkeys(fspec.readline().split()[1:]))

    selectedtimes = [
        (timestep, time)
        for timestep, time in enumerate(float(time) for time in timearray)
        if (timemin is None or timemin <= time) and (timemax is None or timemax >= time)
    ]

    # one collect_all call evaluates the spectra of all the times together, and every band reads them
    spectra = pl.collect_all([
        scan_spectrum_at_time(
            Path(modelpath),
            timestep=timestep,
            time=time,
            dirbin=dirbin,
            vpkt_observers=vpkt_observers,
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
        )
        for timestep, time in selectedtimes
    ])
    times_spectra = list(zip((time for _, time in selectedtimes), spectra, strict=True))

    filterdir = Path(get_path("artistools_dir"), "data/filters/")
    filters_dict: dict[str, list[tuple[float, float]]] = {}

    # the virtual packet spectra cover the wavelength range of vpkt.txt, which can be much less than the range of
    # spec.out. A band outside that range has no flux, thus the magnitude is too faint
    spectrumwavelengths = times_spectra[0][1]["lambda_angstroms"].to_numpy() if times_spectra else None
    spectrumrange = (
        (float(spectrumwavelengths.min()), float(spectrumwavelengths.max()))
        if spectrumwavelengths is not None and spectrumwavelengths.size
        else None
    )
    for filter_name in filternames:
        if filter_name == "bol":
            if isvpktobserver and spectrumrange is not None:
                print_warning(
                    f"The bol band of a virtual packet observer integrates only the range of its spectrum,"
                    f" {spectrumrange[0]:.0f} to {spectrumrange[1]:.0f} Å"
                )
            bol_magnitudes = (
                (time, float(lum_lsun_to_mag(np.asarray(spectrum_to_bolometric_lum(spectrum) / Lsun_to_erg_per_s))))
                for time, spectrum in times_spectra
            )
            # a time with no flux keeps a NaN magnitude, thus a reader knows that the band has no value there
            filters_dict["bol"] = [
                (time, magnitude if math.isfinite(magnitude) else math.nan) for time, magnitude in bol_magnitudes
            ]
            continue

        bandpoints = filters_dict.setdefault(filter_name, [])
        zeropointenergyflux, _, wavefilter, transmission, wavefilter_min, wavefilter_max = get_filter_data(
            filterdir, filter_name
        )
        transmittedwavelengths = wavefilter[transmission > 0.0]
        if spectrumrange is not None and (
            transmittedwavelengths.min() < spectrumrange[0] or transmittedwavelengths.max() > spectrumrange[1]
        ):
            print_warning(
                f"The spectrum covers {spectrumrange[0]:.0f} to {spectrumrange[1]:.0f} Å, and the {filter_name} filter"
                f" transmits from {transmittedwavelengths.min():.0f} to {transmittedwavelengths.max():.0f} Å. The"
                " magnitude has no flux from the part of the filter outside the spectrum"
            )

        for time, spectrum in times_spectra:
            wavelength_from_spectrum, flux = bracket_spectrum_to_band(spectrum, wavefilter_min, wavefilter_max)

            # interpolate the series with the coarser samples onto the finer wavelength grid before the
            # product. Outside the sampled range, the contribution is zero
            spectrum_in_filter = np.logical_and(
                wavelength_from_spectrum >= wavefilter_min, wavelength_from_spectrum <= wavefilter_max
            )
            wavelength_in_filter = wavelength_from_spectrum[spectrum_in_filter]
            flux_in_filter = flux[spectrum_in_filter]
            if len(wavelength_in_filter) > len(wavefilter):
                integrand = flux_in_filter * np.interp(
                    wavelength_in_filter, wavefilter, transmission, left=0.0, right=0.0
                )
                integration_grid = wavelength_in_filter
            else:
                integrand = np.interp(wavefilter, wavelength_from_spectrum, flux, left=0.0, right=0.0) * transmission
                integration_grid = wavefilter

            weighted_flux_obs = float(abs(np.trapezoid(integrand, integration_grid)))
            if weighted_flux_obs <= 0.0:
                # no flux in the band gives no magnitude. A zero magnitude would show an extremely bright source
                print(f"  No {filter_name} band flux at {time:.2f} d. The band has no magnitude at this time.")
                bandpoints.append((time, math.nan))
                continue

            # the spectrum is at 1 Mpc, and 25 = 5 log10(1 Mpc / 10 pc) gives the absolute magnitude at 10 pc
            phot_filtobs_mag = -2.5 * math.log10(weighted_flux_obs / zeropointenergyflux) - 25
            bandpoints.append((time, phot_filtobs_mag))

    return filters_dict


def spectrum_to_bolometric_lum(dfspectrum: pl.DataFrame) -> float:
    """Return the bolometric luminosity in erg/s of a spectrum, given as f_lambda at a distance of 1 Mpc."""
    Mpc_to_cm = megaparsec_to_cm
    return float(
        np.trapezoid(dfspectrum["f_lambda"], dfspectrum["lambda_angstroms"]) * 4 * np.pi * np.power(Mpc_to_cm, 2)
    )


def get_bolometric_luminosities(
    modelpath: Path, timesteps: Sequence[int], dirbins: Sequence[int] = (-1,)
) -> dict[int, list[float]]:
    """Return the bolometric luminosity in erg/s of each direction bin at each given timestep.

    The luminosity comes from an integral of the emergent spectrum at a distance of 1 Mpc. One
    scan_spectra call builds the queries of every timestep, and one collect_all call then evaluates
    the timesteps of one direction bin. Thus the frames of every bin do not stay in memory together.
    """
    lazyspectra = [
        scan_spectra(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep, directionbins=dirbins)
        for timestep in timesteps
    ]

    return {
        dirbin: [
            spectrum_to_bolometric_lum(spectrum)
            for spectrum in pl.collect_all([spectra_timestep[dirbin] for spectra_timestep in lazyspectra])
        ]
        for dirbin in dirbins
    }


@lru_cache(maxsize=32)
def get_filter_data(
    filterdir: Path | str, filter_name: str
) -> tuple[float, float, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float]:
    """Return the zero point, the reference wavelength, the transmission curve, and its wavelength range of a filter.

    The files in data/filters come from https://github.com/cinserra/S3/tree/master/src/s3/metadata. The first line
    holds the zero point in energy flux (erg/cm^2/s). The third line holds the reference wavelength in Angstroms.
    The lines after the fourth hold the wavelength and the transmission.
    """
    with Path(filterdir, f"{filter_name}.txt").open("r", encoding="utf-8") as filter_metadata:
        line_in_filter_metadata = filter_metadata.readlines()

    zeropointenergyflux = float(line_in_filter_metadata[0])
    lambda0 = float(line_in_filter_metadata[2])

    wavefilter: list[float] = []
    transmission: list[float] = []
    for row in line_in_filter_metadata[4:]:
        # lines where the wave and transmission are stored
        wavefilter.append(float(row.split()[0]))
        transmission.append(float(row.split()[1]))

    # the files under data/filters/NOT are not in ascending wavelength order, and callers both
    # interpolate on this grid and integrate over it with np.trapezoid, so sort it here
    sortidx = np.argsort(wavefilter)
    arr_wavefilter = np.array(wavefilter)[sortidx]
    arr_transmission = np.array(transmission)[sortidx]

    # the cache shares one instance with every caller, thus a write must not reach it
    arr_wavefilter.setflags(write=False)
    arr_transmission.setflags(write=False)

    return (
        zeropointenergyflux,
        lambda0,
        arr_wavefilter,
        arr_transmission,
        float(arr_wavefilter[0]),
        float(arr_wavefilter[-1]),
    )


def bracket_spectrum_to_band(
    spectrum: pl.DataFrame, wavefilter_min: float, wavefilter_max: float
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the wavelengths and the fluxes of the spectrum rows that bracket a filter wavelength range."""
    spectrum_bracketed = (
        df_filter_minmax_bracketed(spectrum, "lambda_angstroms", wavefilter_min, wavefilter_max)
        .select("lambda_angstroms", "f_lambda")
        .collect()
    )
    return spectrum_bracketed["lambda_angstroms"].to_numpy(), spectrum_bracketed["f_lambda"].to_numpy()


def get_band_lightcurve(
    band_lightcurve_data: Mapping[str, Sequence[tuple[float, float]]],
    band_name: str,
    *,
    timemin: float | None = None,
    timemax: float | None = None,
) -> tuple[Sequence[float], npt.NDArray[np.floating]]:
    """Return the times and magnitudes of one band's light curve, restricted to the range timemin to timemax.

    A time with no flux in the band has no magnitude, thus it gives no point. get_band_gap_times gives these times.
    """
    selected = [
        (time, brightness)
        for time, brightness in band_lightcurve_data[band_name]
        if math.isfinite(brightness) and (timemin is None or timemin <= time) and (timemax is None or timemax >= time)
    ]
    if not selected:
        msg = f"The {band_name} band has no light curve points between timemin {timemin} and timemax {timemax}"
        raise ValueError(msg)

    times, brightness_in_mag = zip(*selected, strict=True)

    return times, np.array(brightness_in_mag)


def get_band_gap_times(
    band_lightcurve_data: Mapping[str, Sequence[tuple[float, float]]], band_name: str
) -> list[float]:
    """Return the times at which a band has no flux and no magnitude, e.g. for a test of an interpolation."""
    return [time for time, brightness in band_lightcurve_data[band_name] if not math.isfinite(brightness)]


def get_colour_delta_mag(
    band_lightcurve_data: Mapping[str, Iterable[tuple[float, float]]], filter_names: Sequence[str]
) -> tuple[list[float], list[float]]:
    """Return the times and magnitude differences of two bands, using only the times sampled by both bands."""
    # make magnitude dictionaries where time is the key
    time_dict_1 = {time: mag for time, mag in band_lightcurve_data[filter_names[0]] if math.isfinite(mag)}
    time_dict_2 = {time: mag for time, mag in band_lightcurve_data[filter_names[1]] if math.isfinite(mag)}

    # a band with no flux at some time contributes no point there, so the two bands can be sampled at different
    # times. Only times present in both bands have a colour
    plot_times = sorted(time_dict_1.keys() & time_dict_2.keys())
    colour_delta_mag = [time_dict_1[time] - time_dict_2[time] for time in plot_times]

    return plot_times, colour_delta_mag


def read_hesma_lightcurve_file(hesma_modelpath: Path | str) -> pl.DataFrame:
    """Read a HESMA model light curve, taking the column names from a leading comment line if there is one."""
    return read_wsv(hesma_modelpath, comment_prefix="#", header_from_comment=True)


def read_hesma_lightcurve(hesmamodel: str) -> pl.DataFrame:
    """Return the HESMA model light curve of a path or of a file name in the data/hesma folder."""
    return read_hesma_lightcurve_file(require_reference_data_file(hesmamodel, "data/hesma", "HESMA model light curve"))


def luminosity_distance(H0: float, Om0: float, z: float) -> float:
    """Return the luminosity distance in Mpc to redshift z for a flat LambdaCDM cosmology.

    H0 is the Hubble constant in km/s/Mpc and Om0 the present-day matter density parameter. Radiation is
    neglected, so the dark energy density is 1 - Om0. The luminosity distance is D_L = (1 + z) D_C for a
    comoving distance

        D_C = (c / H0) * int_0^z dz' / sqrt(Om0 (1 + z')^3 + 1 - Om0).

    This splits the same three ways as astropy's FlatLambdaCDM for a cosmology without radiation. The de
    Sitter (Om0 = 0) and Einstein-de Sitter (Om0 = 1) cases have the closed forms astropy uses, and every
    other matter density reduces to the function of Baes, Camps & Van De Putte (2017),

        T(x) = 2 sqrt(x) 2F1(1/6, 1/2; 7/6; -x^3) = 2 * int_0^sqrt(x) dy / sqrt(1 + y^6),

    as D_C = (c / H0) [T(s) - T(s / (1 + z))] / sqrt(s Om0) with s = ((1 - Om0) / Om0)^(1/3). astropy
    evaluates T through scipy's hyp2f1. This evaluates the difference of the two T values as the single
    integral it stands for, substituting y = sqrt(s) u to absorb s and give

        D_C = (c / H0) * int_u(z)^1 2 du / sqrt(Om0 + (1 - Om0) u^6),   u = (1 + z')^(-1/2),

    which needs no scipy, and improves on the hypergeometric form in two places: subtracting two T values
    of nearly equal size loses precision as z -> 0, where integrating over the range between them does not,
    and s is a cube root of a negative number for Om0 > 1, which this form never has to take.

    No one variable suits the whole range, since 1 / E falls off as (1 + z')^(-3/2) only above the point
    where the two terms of the radicand balance, and a fixed node count cannot follow that corner once the
    sides differ by decades. So the range is split there and each side taken in the variable that flattens
    it: u above, and below it whichever of the two the sign of 1 - Om0 calls for. Dark energy (Om0 < 1)
    holds 1 / E within sqrt(2) of 1 / sqrt(1 - Om0) below its crossover at Om0 (1 + z')^3 = 1 - Om0, which
    is astropy's s, so z serves as it stands. A negative density (Om0 > 1) instead cancels the matter term
    at the turnaround rejected below, leaving an integrable pole that q = sqrt((1 + z')^3 - cturn) unfolds
    into 2 dq / (3 sqrt(Om0) (q^2 + cturn)^(2/3)), flat while q^2 stays under cturn = (Om0 - 1) / Om0.

    Measured against adaptive quadrature, that holds every matter density from 1e-8 to 1e6 accurate to a
    few ulp over the whole domain, from the turnaround or z = -1 + 1e-12 out to z = 1e8.

    The de Sitter and Einstein-de Sitter closed forms are exact and cheaper, so they stay, but only the
    first is now needed: Om0 = 0 has no crossover to divide by.
    """
    if z <= -1.0:
        msg = f"1 + z is a ratio of scale factors, so the redshift must be greater than -1, but got z={z}"
        raise ValueError(msg)

    # A negative dark energy density (Om0 > 1) halts the expansion where E(z)^2 falls to zero, and nothing
    # beyond that turnaround has a distance. E(z)^2 rises with z, so this endpoint covers the whole range.
    # Without the check the radicand goes negative: at the endpoint alone that gives a plausible finite
    # number from the interior nodes, and further out a NaN, which would reach the magnitudes as one.
    if Om0 * (1.0 + z) ** 3 + (1.0 - Om0) <= 0.0:
        zturnaround = ((Om0 - 1.0) / Om0) ** (1.0 / 3.0) - 1.0
        msg = f"A flat cosmology with Om0={Om0} stops expanding at z={zturnaround}, so z={z} has no distance"
        raise ValueError(msg)

    dist_hubble_mpc = (C_cm_per_s / km_to_cm) / H0  # c in km/s over H0 in km/s/Mpc

    # de Sitter: a matter-free universe expands with E(z) = 1, making the comoving distance exactly (c/H0) z
    if Om0 == 0.0:
        return (1.0 + z) * dist_hubble_mpc * z

    # Einstein-de Sitter: a matter-only universe integrates to D_C = 2 (c / H0) (1 - (1 + z)^(-1/2)). expm1
    # keeps the bracket exact as z -> 0, where 1 - 1 / sqrt(1 + z) would lose most of it to cancellation.
    if Om0 == 1.0:
        return (1.0 + z) * dist_hubble_mpc * -2.0 * math.expm1(-0.5 * math.log1p(z))

    nodes, weights = np.polynomial.legendre.leggauss(32)

    # where the two terms of the radicand balance: the dark energy crossover, or twice the turnaround
    cturn = (Om0 - 1.0) / Om0
    zbalance = ((1.0 - Om0) / Om0) ** (1.0 / 3.0) - 1.0 if Om0 < 1.0 else (2.0 * cturn) ** (1.0 / 3.0) - 1.0
    zlo, zhi = min(0.0, z), max(0.0, z)
    zsplit = min(max(zbalance, zlo), zhi)

    integral = 0.0

    if zsplit > zlo:
        if Om0 < 1.0:
            # dark energy holds 1 / E to within sqrt(2) below the crossover, so integrate it as it stands
            halfwidth = 0.5 * (zsplit - zlo)
            zvals = halfwidth * (nodes + 1.0) + zlo
            integral += float(weights @ (1.0 / np.sqrt(Om0 * (1.0 + zvals) ** 3 + (1.0 - Om0)))) * halfwidth
        else:
            # q unfolds the pole that a negative dark energy density leaves at the turnaround. Its width
            # comes from the difference of cubes expanded about z' rather than from qhi - qlo, which would
            # cancel as the range shrinks onto z' = 0. The clamp is only in case this difference rounds
            # below zero where the check above, on the algebraically equivalent E(z)^2, found it positive.
            qlo = math.sqrt(max(0.0, (1.0 + zlo) ** 3 - cturn))
            qhi = math.sqrt(max(0.0, (1.0 + zsplit) ** 3 - cturn))
            cubediff = (zsplit - zlo) * (3.0 + 3.0 * (zlo + zsplit) + (zlo * zlo + zlo * zsplit + zsplit**2))
            halfwidth = 0.5 * cubediff / (qhi + qlo)
            qvals = halfwidth * (nodes + 1.0) + qlo
            qintegrand = 2.0 / (3.0 * math.sqrt(Om0) * (qvals**2 + cturn) ** (2.0 / 3.0))
            integral += float(weights @ qintegrand) * halfwidth

    if zhi > zsplit:
        # above it, u = (1 + z')^(-1/2) flattens the (1 + z')^(-3/2) tail. Taking the width of the u range
        # from the difference of the roots keeps it exact as the range shrinks onto z' = 0, where
        # (1 + z')^(-1/2) - 1 would lose it to cancellation.
        rootlo, roothi = math.sqrt(1.0 + zsplit), math.sqrt(1.0 + zhi)
        halfwidth = 0.5 * (zhi - zsplit) / (rootlo * roothi * (rootlo + roothi))
        uvals = halfwidth * (nodes + 1.0) + 1.0 / roothi
        integral += float(weights @ (2.0 / np.sqrt(Om0 + (1.0 - Om0) * uvals**6))) * halfwidth

    return (1.0 + z) * dist_hubble_mpc * (integral if z >= 0.0 else -integral)


def read_reflightcurve_band_table(data_path: Path, lines: list[str], *, header_from_comment: bool) -> pl.DataFrame:
    """Return the table of a band reference light curve file, which separates its columns with commas or with spaces.

    With header_from_comment, a first line that starts with "#" gives the names of the columns.
    """
    if header_from_comment and lines and lines[0].lstrip().startswith("#"):
        lines = [lines[0].lstrip().removeprefix("#"), *lines[1:]]
    # a reference light curve file can put a comment after a value, thus cut each line at the first "#"
    csvtext = "\n".join(line.split("#", 1)[0].rstrip() for line in lines)
    lightcurve_data = pl.read_csv(csvtext.encode())
    if lightcurve_data.width == 1:
        lightcurve_data = read_wsv(data_path, comment_prefix="#", header_from_comment=header_from_comment)
    return lightcurve_data


def read_reflightcurve_band_data(lightcurvefilename: Path | str) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return an observed band light curve from a reference data file, along with its metadata.

    The package holds the metadata of each known reference light curve, but not the data of it. Thus the
    working folder gives the data file, and the package folder gives the metadata that has no file beside
    the data.
    """
    data_path = find_reference_data_file(lightcurvefilename, "data/lightcurves")
    if data_path is None:
        msg = f"Could not find the reference light curve {lightcurvefilename}"
        raise FileNotFoundError(msg)

    # a copy, because get_file_metadata is cached and this function adds a derived distance below. The metadata
    # comes from the folder of the data, or else from the bundled folder
    metadatasource = data_path
    metadata = dict(get_file_metadata(metadatasource))
    if not metadata:
        metadatasource = Path(get_path("artistools_dir"), "data/lightcurves", data_path.name)
        metadata = dict(get_file_metadata(metadatasource))

    # the time column below needs the timecorrection value. The command stops when the metadata
    # gives no such value
    if "timecorrection" not in metadata:
        exit_with_error(
            f"The reference light curve {data_path} has no timecorrection value",
            f"Write {get_metadata_path(data_path).name} in the folder of the data, with a timecorrection value and a"
            " label",
        )
    # a file with no label takes the name that the command gives, as the bolometric curves and the spectra do
    metadata["label"] = get_metadata_label(metadata, lightcurvefilename)

    with zopen(data_path, encoding="utf-8") as datafile:
        lines = datafile.read().splitlines()
    lightcurve_data = read_reflightcurve_band_table(data_path, lines, header_from_comment=False)
    if "magnitude" not in lightcurve_data.columns:
        # the header line can start with "#", as in a bolometric reference light curve
        lightcurve_data = read_reflightcurve_band_table(data_path, lines, header_from_comment=True)
    if not {"time", "magnitude", "band"}.issubset(lightcurve_data.columns):
        msg = (
            f"{data_path} gives the columns {lightcurve_data.columns}, and a band light curve needs time, magnitude,"
            " and band"
        )
        raise ValueError(msg)

    # m - M = 5log(d) - 5  Get absolute magnitude. A distance modulus from the metadata is a measured value,
    # thus the distance from the redshift applies only when the metadata gives neither
    if "dist_mpc" not in metadata and "dist_modulus" not in metadata and "z" in metadata:
        # the command reports the ValueError of an incorrect z as an error message, with no traceback
        metafile = get_metadata_path(metadatasource)
        # a negative z gives a negative luminosity distance
        z = parse_metadata_redshift(metadata["z"], metafile, minimum=0.0, allowminimum=True)
        if z == 0.0:
            # z: 0 marks the rest frame, as for a reference spectrum, thus there is no distance
            print_warning(
                f"{metafile} gives z = 0 and no dist_mpc or dist_modulus, thus the magnitudes of {metadata['label']}"
                " have no distance correction"
            )
        else:
            metadata["dist_mpc"] = luminosity_distance(H0=H0_km_per_s_per_mpc, Om0=omega_matter, z=z)
            print(
                f"Luminosity distance from redshift z = {z}: {metadata['dist_mpc']:.4g} Mpc for {metadata['label']},"
                f" with a flat cosmology of H0 = {H0_km_per_s_per_mpc} km/s/Mpc and Om0 = {omega_matter} (round"
                " values that many papers use, from no single measurement)"
            )

    magnitude = pl.col("magnitude").cast(pl.Float64)
    if "dist_mpc" in metadata:
        magnitude = magnitude - 5 * np.log10(metadata["dist_mpc"] * 10**6) + 5
    elif "dist_modulus" in metadata:
        magnitude -= metadata["dist_modulus"]

    lightcurve_data = lightcurve_data.with_columns(magnitude, pl.col("time") - metadata["timecorrection"])

    return lightcurve_data, metadata


def find_bol_reflightcurve_file(lightcurvefilename: str | Path) -> Path | None:
    """Return the path of a file with a reference light curve, or None if no such file exists.

    The file is either at the given path, or in the bundled data/lightcurves/bollightcurves folder.
    A compressed file with the same name is also accepted.
    """
    return find_reference_data_file(lightcurvefilename, "data/lightcurves/bollightcurves")


def find_lightcurve_file(modelpath: Path | str, *, directionresolved: bool = False, gamma: bool = False) -> Path:
    """Return the path of the ARTIS light curve output file, accepting a compressed file.

    One owner of the file name, so that every command reads the same file for the same request and
    reports the same message when it is absent.
    """
    if directionresolved and gamma:
        msg = (
            "ARTIS writes no direction-resolved gamma-ray light curve file."
            " Give --frompackets to make one from the packets"
        )
        raise FileNotFoundError(msg)

    if directionresolved:
        lcfilename = "light_curve_res.out"
    elif gamma:
        lcfilename = "gamma_light_curve.out"
    else:
        lcfilename = "light_curve.out"

    return firstexisting(lcfilename, folder=modelpath, tryzipped=True)


def path_is_reference_lightcurve(filepath: str | Path) -> bool:
    """Return whether the path is a bolometric reference light curve file and not an ARTIS model."""
    return path_is_reference_data(filepath, "data/lightcurves/bollightcurves")


def read_bol_reflightcurve_data(lightcurvefilename: str | Path) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return an observed bolometric light curve and its metadata, from a given path or the bundled reference data."""
    data_path = find_bol_reflightcurve_file(lightcurvefilename)
    if data_path is None:
        msg = f"Reference light curve file not found: {lightcurvefilename}"
        raise FileNotFoundError(msg)

    metadata = get_file_metadata(data_path)

    # the column names come from a leading comment line if there is one
    dflightcurve = read_wsv(data_path, has_header=False, comment_prefix="#", header_from_comment=True)

    if colrenames := {
        k: v
        for k, v in {dflightcurve.columns[0]: "time_days", dflightcurve.columns[1]: "luminosity_erg/s"}.items()
        if k != v
    }:
        print(f"{data_path}: renaming columns {colrenames}")
        dflightcurve = dflightcurve.rename(colrenames)

    return dflightcurve, metadata


def get_phillips_relation_data() -> tuple[pl.DataFrame, str]:
    """Return the observed dm15(B) against peak MB data of Hicken et al. (2009), and its plot label.

    The package holds no copy of the data, thus the working folder or the data folder gives the file.
    """
    datafilepath = require_reference_data_file(
        "CfA3_Phillips.dat", "data/lightcurves/SNsample", "Phillips relation data of Hicken et al. (2009)"
    )
    sn_data = read_wsv(datafilepath, comment_prefix="#").with_columns(
        pl.col("dm15(B)").cast(pl.Float64), pl.col("MB").cast(pl.Float64)
    )
    print(sn_data)

    return sn_data, "Observed (Hicken et al. 2009)"


def read_artis_lightcurve(
    modelpath: Path,
    dirbins: Sequence[int],
    *,
    lcfilename: str | None,
    escape_type: str,
    frompackets: bool,
    maxpacketfiles: int | None,
    average_over_phi: bool,
    average_over_theta: bool,
    vpkt_observers: bool,
    pellet_nucname: str | None,
    use_pellet_decay_time: bool,
    timemin: float | None,
    timemax: float | None,
) -> dict[int, pl.DataFrame] | None:
    """Return the light curve of each direction bin between timemin and timemax, or None if the file does not exist.

    Each frame keeps the nearest timestep beyond each end of the time range. With frompackets, the function reads the
    packets files. Otherwise it reads lcfilename, or light_curve.out for bin -1 and light_curve_res.out for the other
    bins. Only the packets give the nuclide and the decay time of a pellet.
    """
    # bin -1 is the angle average, which only light_curve.out holds
    directionresolved = list(dirbins) != [-1]

    if frompackets:
        lcdataframes = scan_lightcurve_from_packets(
            modelpath,
            escape_type=escape_type,
            maxpacketfiles=maxpacketfiles,
            directionbins=dirbins,
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
            directionbins_are_vpkt_observers=vpkt_observers,
            pellet_nucname=pellet_nucname,
            use_pellet_decay_time=use_pellet_decay_time,
            timedaysmin=timemin,
            timedaysmax=timemax,
        )
    else:
        if pellet_nucname is not None or use_pellet_decay_time:
            msg = "Only the packets give the nuclide and the decay time of a pellet, thus give frompackets=True"
            raise ValueError(msg)
        if vpkt_observers:
            exit_with_error(
                "--plotvspecpol names virtual packet observers, which light_curve_res.out does not hold",
                "Give --frompackets to make the light curve of each virtual observer from the packets.",
            )
        try:
            lcpath = (
                firstexisting(lcfilename, folder=modelpath, tryzipped=True)
                if lcfilename is not None
                else find_lightcurve_file(
                    modelpath, directionresolved=directionresolved, gamma=escape_type == "TYPE_GAMMA"
                )
            )
        except FileNotFoundError as exc:
            print_warning(f"Skipping {modelpath}: {exc}")
            return None

        try:
            lcdataframes = scan_lightcurve(
                lcpath,
                directionresolved=directionresolved,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
            )
        except ValueError as exc:
            exit_with_error(
                str(exc),
                "Give --plotviewingangle with light_curve_res.out, give no --plotviewingangle with light_curve.out, and"
                " give a file that ARTIS wrote in full",
            )
        if directionresolved:
            warn_direction_bin_noise(
                lcpath.name, average_over_phi=average_over_phi, average_over_theta=average_over_theta
            )
        # light_curve_res.out holds the bins 0 to 99, thus the angle average of bin -1 comes from light_curve.out
        if -1 in dirbins and -1 not in lcdataframes:
            lcdataframes[-1] = scan_lightcurve(
                find_lightcurve_file(modelpath, directionresolved=False, gamma=escape_type == "TYPE_GAMMA")
            )[-1]

    return dict(
        zip(
            dirbins,
            pl.collect_all(
                df_filter_minmax_bracketed(lcdataframes[dirbin], colname="time_days", minval=timemin, maxval=timemax)
                for dirbin in dirbins
            ),
            strict=True,
        )
    )


def split_at_valid_range(
    lcdata: pl.DataFrame, validrange_start_days: float | None, validrange_end_days: float | None
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Return the rows of a light curve before, inside, and after the valid range of the arrival times.

    The part before and the part after each also hold the nearest row of the valid range, thus their lines join the
    line of the valid range. A range of None, or a range that holds no row, gives all the rows to the part before.
    """
    emptylcdata = pl.DataFrame(schema=lcdata.schema)
    if validrange_start_days is None or validrange_end_days is None:
        return lcdata, emptylcdata, emptylcdata

    lcdata_valid = lcdata.filter(pl.col("time_days").is_between(validrange_start_days, validrange_end_days))
    if lcdata_valid.is_empty():
        return lcdata, lcdata_valid, emptylcdata

    return (
        lcdata.filter(pl.col("time_days") <= lcdata_valid["time_days"].min()),
        lcdata_valid,
        lcdata.filter(pl.col("time_days") >= lcdata_valid["time_days"].max()),
    )


# the names before the readers took the prefix scan_ or read_. Each alias keeps the old signature, thus a script
# of a user still runs, and it gives a DeprecationWarning that names the new function
@warnings.deprecated("get_from_packets is deprecated. Use scan_lightcurve_from_packets.")
def get_from_packets(
    modelpath: str | Path,
    escape_type: str = "TYPE_RPKT",
    maxpacketfiles: int | None = None,
    directionbins: Collection[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    directionbins_are_vpkt_observers: bool = False,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
    timedaysmin: float | None = None,
    timedaysmax: float | None = None,
) -> dict[int, pl.LazyFrame]:
    """Return the result of scan_lightcurve_from_packets, the new name of this function."""
    return scan_lightcurve_from_packets(
        modelpath,
        escape_type=escape_type,
        maxpacketfiles=maxpacketfiles,
        directionbins=directionbins,
        average_over_phi=average_over_phi,
        average_over_theta=average_over_theta,
        directionbins_are_vpkt_observers=directionbins_are_vpkt_observers,
        pellet_nucname=pellet_nucname,
        use_pellet_decay_time=use_pellet_decay_time,
        timedaysmin=timedaysmin,
        timedaysmax=timedaysmax,
    )
