"""Read ARTIS light curves and reference observational light curves, and derive band magnitudes from spectra."""

import argparse
import math
import typing as t
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import Lsun_to_erg_per_s

# ARTIS writes the Sloan filters with a trailing "s"; map them back to the conventional single-letter names
FILTERNAME_ALIASES: t.Final[Mapping[str, str]] = MappingProxyType({"rs": "r", "gs": "g", "is": "i", "zs": "z"})


def readfile(filepath: str | Path) -> dict[int, pl.LazyFrame]:
    """Read an ARTIS light curve file."""
    print(f"Reading {filepath}")
    lcdata: dict[int, pl.LazyFrame] = {}
    lzdf = pl.scan_csv(
        at.zopenpl(filepath),
        separator=" ",
        has_header=False,
        new_columns=["time_days", "luminosity_Lsun", "luminosity_cmf_Lsun"],
    ).with_columns(
        (4.74 - (2.5 * pl.col("luminosity_Lsun").log10())).alias("mag"),
        (cs.ends_with("_Lsun") * Lsun_to_erg_per_s).name.replace("_Lsun", "_erg/s"),
    )
    if "_res" in Path(filepath).stem:
        # get a dict of dfs with light curves at each viewing direction bin
        lcdata = at.split_multitable_dataframe(lzdf)
    else:
        lcdata[-1] = lzdf

        # if the light_curve.out file repeats x values, keep the first half only
        if lcdata[-1].select(pl.col("time_days").n_unique() < pl.len()).collect().item():
            lcdata[-1] = lcdata[-1].select(pl.all().slice(0, pl.len() // 2))

    return lcdata


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
    """Get ARTIS luminosity vs time from packets files."""
    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)
    if directionbins is None:
        directionbins = [-1]

    dftimesteps_selected = at.misc.df_filter_minmax_bounded(
        at.get_timesteps(modelpath), "tmid_days", timedaysmin, timedaysmax
    ).collect()

    _, modelmeta = at.inputmodel.get_modeldata(modelpath, printwarningsonly=True)

    timebinstarts_plusend = [
        *dftimesteps_selected["tstart_days"],
        dftimesteps_selected.select(pl.col("tstart_days").last() + pl.col("twidth_days").last()).item(),
    ]

    vpkt_config = at.get_vpkt_config(modelpath) if directionbins_are_vpkt_observers else None
    assert not directionbins_are_vpkt_observers or pellet_nucname is None  # we don't track which pellet led to vpkts
    # only set for real packets, where the escape times are measured at the model surface
    escapesurfacegamma: float | None = None
    if directionbins_are_vpkt_observers:
        nprocs_read, dfpackets = at.packets.get_virtual_packets(modelpath, maxpacketfiles=maxpacketfiles)
    else:
        nprocs_read, dfpackets = at.packets.get_packets(
            modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type=escape_type
        )
        escapesurfacegamma = math.sqrt(1 - (modelmeta["vmax_cmps"] / C_cm_per_s) ** 2)
        dfpackets = dfpackets.with_columns([
            (pl.col("escape_time") * escapesurfacegamma / day_to_s).alias("t_arrive_cmf_d")
        ])

    if pellet_nucname is not None:
        atomic_number = at.get_atomic_number(pellet_nucname)
        if at.get_elsymbol(atomic_number) == pellet_nucname:
            expr = pl.col("atomic_number") == atomic_number
        else:
            expr = pl.col("nucname") == pellet_nucname
        dfpackets = dfpackets.filter(
            pl.col("pellet_nucindex").is_in(
                at.get_nuclides(modelpath=modelpath).filter(expr).select("pellet_nucindex").collect().to_series()
            )
        )

    if use_pellet_decay_time:
        assert not directionbins_are_vpkt_observers
        dfpackets = dfpackets.with_columns([(pl.col("tdecay") / day_to_s).alias("tdecay_d")])

    timecol = "tdecay_d" if use_pellet_decay_time else "t_arrive_d"

    lcdata: dict[int, pl.LazyFrame] = {}
    for dirbin in directionbins:
        if directionbins_are_vpkt_observers:
            assert vpkt_config is not None
            obsdirindex, opacchoiceindex = divmod(dirbin, vpkt_config["nspectraperobs"])
            pldfpackets_dirbin = dfpackets.with_columns(
                e_rf=pl.col(f"dir{obsdirindex}_e_rf_{opacchoiceindex}"),
                t_arrive_d=pl.col(f"dir{obsdirindex}_t_arrive_d"),
            )
            inverse_solidangle_fraction = 4 * math.pi
        else:
            pldfpackets_dirbin, inverse_solidangle_fraction = at.packets.filter_packets_dirbin(
                dfpackets, dirbin, average_over_phi=average_over_phi, average_over_theta=average_over_theta
            )

        lcdata[dirbin] = (
            at.packets
            .bin_and_sum(
                pldfpackets_dirbin, bincol=timecol, bins=timebinstarts_plusend, sumcols=["e_rf"], getcounts=True
            )
            .with_columns(timestep=pl.col(f"{timecol}_bin").cast(pl.Int32) + dftimesteps_selected["timestep"].min())
            .rename({"count": "packetcount"})
            .join(dftimesteps_selected.select("timestep", "twidth_days", "tmid_days").lazy(), how="left", on="timestep")
            .with_columns(
                luminosity_Lsun=(
                    pl.col("e_rf_sum")
                    / nprocs_read
                    * inverse_solidangle_fraction
                    / (pl.col("twidth_days") * day_to_s)
                    / Lsun_to_erg_per_s
                )
            )
            .drop("e_rf_sum", f"{timecol}_bin")
        )

        if escapesurfacegamma is not None and "t_arrive_cmf_d" in pldfpackets_dirbin.collect_schema().names():
            lcdata[dirbin] = (
                lcdata[dirbin]
                .join(
                    at.packets
                    .bin_and_sum(
                        pldfpackets_dirbin, bincol="t_arrive_cmf_d", bins=timebinstarts_plusend, sumcols=["e_cmf"]
                    )
                    .with_columns(
                        timestep=pl.col("t_arrive_cmf_d_bin").cast(pl.Int32) + dftimesteps_selected["timestep"].min()
                    )
                    .drop("t_arrive_cmf_d_bin"),
                    how="left",
                    on="timestep",
                )
                .with_columns(
                    luminosity_cmf_Lsun=pl.col("e_cmf_sum")
                    / nprocs_read
                    * inverse_solidangle_fraction
                    / escapesurfacegamma
                    / (pl.col("twidth_days") * day_to_s)
                    / Lsun_to_erg_per_s
                )
                .drop("e_cmf_sum")
            )

        lcdata[dirbin] = (
            lcdata[dirbin]
            .rename({"tmid_days": "time_days"})
            .drop("twidth_days")
            .with_columns(
                (4.74 - (2.5 * pl.col("luminosity_Lsun").log10())).alias("mag"),
                (cs.ends_with("_Lsun") * Lsun_to_erg_per_s).name.replace("_Lsun", "_erg/s"),
            )
        )

    return lcdata


def args_from_kwargs(
    args: argparse.Namespace | None, kwargs: dict[str, t.Any], **defaults: t.Any
) -> argparse.Namespace:
    """Return args if it was given, otherwise build a Namespace from kwargs on top of the given defaults."""
    if args is None:
        return argparse.Namespace(**(defaults | kwargs))

    if kwargs:
        msg = "Specify either args or keyword options, not both"
        raise TypeError(msg)

    return args


def generate_band_lightcurve_data(
    modelpath: Path | str,
    args: argparse.Namespace | None = None,
    dirbin: int = -1,
    modelnumber: int | None = None,  # ruff:ignore[unused-function-argument]
    **kwargs: t.Any,
) -> dict[str, t.Any]:
    """Integrate spectra to get band magnitude vs time. Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py."""
    args = args_from_kwargs(
        args,
        kwargs,
        plotvspecpol=False,
        plotviewingangle=False,
        filter=None,
        timemin=None,
        timemax=None,
        average_over_phi_angle=False,
        average_over_theta_angle=False,
    )
    if args.plotvspecpol and Path(modelpath, "vpkt.txt").is_file():
        print("Found vpkt.txt, using virtual packets")
        stokes_params = (
            at.spectra.get_vspecpol_data(vspecindex=dirbin, modelpath=modelpath)
            if dirbin >= 0
            else at.spectra.get_specpol_data(dirbin=dirbin, modelpath=modelpath)
        )
        vspecdata = stokes_params["I"]
        timearray = vspecdata.collect_schema().names()[1:]
    else:
        specfilename = (
            at.firstexisting_or_none(["specpol_res.out", "spec_res.out"], folder=modelpath, tryzipped=True)
            if args.plotviewingangle
            else None
        )
        if specfilename is None:
            if args.plotviewingangle:
                print("WARNING: no direction-resolved spectra available. Using angle-averaged spectra.")
            specfilename = at.firstexisting(["spec.out", "specpol.out"], folder=modelpath, tryzipped=True)

        with at.zopen(specfilename) as fspec:
            # pol and res files repeat the time columns for the Stokes Q and U blocks, so keep the first of each
            timearray = list(dict.fromkeys(fspec.readline().split()[1:]))

    filters_dict = {}
    if not args.filter:
        args.filter = ["B"]

    filters_list = args.filter

    for filter_name in filters_list:
        if filter_name == "bol":
            times, bol_magnitudes = bolometric_magnitude(
                Path(modelpath),
                timearray,
                args,
                angle=dirbin,
                average_over_phi=args.average_over_phi_angle,
                average_over_theta=args.average_over_theta_angle,
            )
            filters_dict["bol"] = [
                (time, bol_magnitude)
                for time, bol_magnitude in zip(times, bol_magnitudes, strict=False)
                if math.isfinite(bol_magnitude)
            ]
        elif filter_name not in filters_dict:
            filters_dict[filter_name] = []

    filterdir = Path(at.get_path("artistools_dir"), "data/filters/")

    for filter_name in filters_list:
        if filter_name == "bol":
            continue
        zeropointenergyflux, wavefilter, transmission, wavefilter_min, wavefilter_max = get_filter_data(
            filterdir, filter_name
        )

        for timestep, time in enumerate(float(time) for time in timearray):
            if (args.timemin is None or args.timemin <= time) and (args.timemax is None or args.timemax >= time):
                wavelength_from_spectrum, flux = get_spectrum_in_filter_range(
                    modelpath=modelpath,
                    timestep=timestep,
                    time=time,
                    wavefilter_min=wavefilter_min,
                    wavefilter_max=wavefilter_max,
                    angle=dirbin,
                    args=args,
                    average_over_phi=args.average_over_phi_angle,
                    average_over_theta=args.average_over_theta_angle,
                )

                # interpolate the coarser-sampled series onto the finer wavelength grid before multiplying,
                # with zero contribution outside the sampled range
                if len(wavelength_from_spectrum) > len(wavefilter):
                    integrand = flux * np.interp(
                        wavelength_from_spectrum, wavefilter, transmission, left=0.0, right=0.0
                    )
                    integration_grid = wavelength_from_spectrum
                else:
                    integrand = (
                        np.interp(wavefilter, wavelength_from_spectrum, flux, left=0.0, right=0.0) * transmission
                    )
                    integration_grid = wavefilter

                weighted_flux_obs = float(abs(np.trapezoid(integrand, integration_grid)))
                if weighted_flux_obs <= 0.0:
                    # no flux in the band, so there is no magnitude to report. Appending zero here would look like
                    # an extremely bright source
                    print(f"  no {filter_name} band flux at {time:.2f} d. Skipping this time.")
                    continue

                # 25 converts the apparent magnitude at 10 pc to an absolute magnitude
                phot_filtobs_mag = -2.5 * math.log10(weighted_flux_obs / zeropointenergyflux) - 25
                filters_dict[filter_name].append((time, phot_filtobs_mag))

    return filters_dict


def bolometric_magnitude(
    modelpath: Path,
    timearray: Collection[float | str],
    args: argparse.Namespace,
    angle: int = -1,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> tuple[list[float], list[float]]:
    """Return the times and bolometric magnitudes obtained by integrating the spectra of one direction bin."""
    magnitudes = []
    times = []

    Mpc_to_cm = at.constants.megaparsec_to_cm
    for timestep, time in enumerate(float(time) for time in timearray):
        if (args.timemin is None or args.timemin <= time) and (args.timemax is None or args.timemax >= time):
            if angle == -1:
                spectrum = at.spectra.get_spectra(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep)[
                    -1
                ].collect()
            elif args.plotvspecpol:
                spectrum = at.spectra.get_vspecpol_spectrum(modelpath, time, angle, args).collect()
            else:
                spectrum = at.spectra.get_spectra(
                    modelpath=modelpath,
                    timestepmin=timestep,
                    timestepmax=timestep,
                    average_over_phi=average_over_phi,
                    average_over_theta=average_over_theta,
                )[angle].collect()
            integrated_flux = np.trapezoid(spectrum["f_lambda"], spectrum["lambda_angstroms"])
            integrated_luminosity = integrated_flux * 4 * np.pi * np.power(Mpc_to_cm, 2)
            Mbol_sun = 4.74
            with np.errstate(divide="ignore"):
                magnitude = Mbol_sun - (2.5 * np.log10(integrated_luminosity / Lsun_to_erg_per_s))
            magnitudes.append(magnitude)
            times.append(time)

    return times, magnitudes


def get_filter_data(
    filterdir: Path | str, filter_name: str
) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating], float, float]:
    """Filter data in 'data/filters' taken from https://github.com/cinserra/S3/tree/master/src/s3/metadata."""
    with Path(filterdir, f"{filter_name}.txt").open("r", encoding="utf-8") as filter_metadata:  # definition of the file
        line_in_filter_metadata = filter_metadata.readlines()  # list of lines

    zeropointenergyflux = float(line_in_filter_metadata[0])
    # zero point in energy flux (erg/cm^2/s)

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

    return zeropointenergyflux, arr_wavefilter, arr_transmission, float(arr_wavefilter[0]), float(arr_wavefilter[-1])


def get_spectrum_in_filter_range(
    modelpath: Path | str,
    timestep: int,
    time: float,
    wavefilter_min: float,
    wavefilter_max: float,
    angle: int = -1,
    args: argparse.Namespace | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the wavelengths and fluxes of the spectrum at one timestep, restricted to a filter's wavelength range."""
    spectrum = at.spectra.get_spectrum_at_time(
        Path(modelpath),
        timestep=timestep,
        time=time,
        args=args,
        dirbin=angle,
        average_over_phi=average_over_phi,
        average_over_theta=average_over_theta,
    )
    assert spectrum is not None

    wavelength_from_spectrum: list[float] = []
    flux: list[float] = []
    for wavelength, flambda in zip(spectrum["lambda_angstroms"], spectrum["f_lambda"], strict=True):
        if wavefilter_min <= wavelength <= wavefilter_max:  # to match the spectrum wavelengths to those of the filter
            wavelength_from_spectrum.append(wavelength)
            flux.append(flambda)

    return np.array(wavelength_from_spectrum), np.array(flux)


def get_band_lightcurve(
    band_lightcurve_data: dict[str, Sequence[tuple[float, float]]],
    band_name: str,
    args: argparse.Namespace | None = None,
    **kwargs: t.Any,
) -> tuple[Sequence[float], npt.NDArray[np.floating]]:
    """Return the times and magnitudes of one band's light curve, restricted to the args.timemin/timemax range."""
    args = args_from_kwargs(args, kwargs)

    times, brightness_in_mag = zip(
        *[
            (time, brightness)
            for time, brightness in band_lightcurve_data[band_name]
            if ((args.timemin is None or args.timemin <= time) and (args.timemax is None or args.timemax >= time))
        ],
        strict=False,
    )

    return times, np.array(brightness_in_mag)


def get_colour_delta_mag(
    band_lightcurve_data: Mapping[str, Iterable[tuple[float, float]]], filter_names: Sequence[str]
) -> tuple[list[float], list[float]]:
    """Return the times and magnitude differences of two bands, using only the times sampled by both bands."""
    # make magnitude dictionaries where time is the key
    time_dict_1 = dict(band_lightcurve_data[filter_names[0]])
    time_dict_2 = dict(band_lightcurve_data[filter_names[1]])

    # a band with no flux at some time contributes no point there, so the two bands can be sampled at different
    # times. Only times present in both bands have a colour
    plot_times = sorted(time_dict_1.keys() & time_dict_2.keys())
    colour_delta_mag = [time_dict_1[time] - time_dict_2[time] for time in plot_times]

    return plot_times, colour_delta_mag


def read_hesma_lightcurve_file(hesma_modelpath: Path | str) -> pl.DataFrame:
    """Read a HESMA model light curve, taking the column names from a leading comment line if there is one."""
    return at.read_wsv(hesma_modelpath, comment_prefix="#", header_from_comment=True)


def read_hesma_lightcurve(args: argparse.Namespace) -> pl.DataFrame:
    """Return the HESMA model light curve named by args.plot_hesma_model."""
    return read_hesma_lightcurve_file(Path(at.get_path("artistools_dir"), "data/hesma", args.plot_hesma_model))


def read_reflightcurve_band_data(lightcurvefilename: Path | str) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return an observed band light curve from the bundled reference data, along with its metadata."""
    filepath = Path(at.get_path("artistools_dir"), "data", "lightcurves", lightcurvefilename)
    metadata = at.get_file_metadata(filepath)

    data_path = Path(at.get_path("artistools_dir"), f"data/lightcurves/{lightcurvefilename}")
    # like the pandas comment="#" this replaces, cut each line at a "#" anywhere, not only at line starts
    csvtext = "\n".join(line.split("#", 1)[0].rstrip() for line in data_path.read_text(encoding="utf-8").splitlines())
    lightcurve_data = pl.read_csv(csvtext.encode())
    if lightcurve_data.width == 1:
        lightcurve_data = at.read_wsv(data_path, comment_prefix="#")

    # m - M = 5log(d) - 5  Get absolute magnitude
    if "dist_mpc" not in metadata and "z" in metadata:
        from astropy import cosmology

        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)  # pyright: ignore[reportCallIssue] # pyrefly: ignore[unexpected-keyword]  # ty:ignore[unknown-argument]
        metadata["dist_mpc"] = cosmo.luminosity_distance(metadata["z"]).value  # pyright: ignore[reportAttributeAccessIssue]  # ty:ignore[unresolved-attribute]
        print(f"luminosity distance from redshift = {metadata['dist_mpc']} for {metadata['label']}")

    magnitude = pl.col("magnitude").cast(pl.Float64)
    if "dist_mpc" in metadata:
        magnitude = magnitude - 5 * np.log10(metadata["dist_mpc"] * 10**6) + 5
    elif "dist_modulus" in metadata:
        magnitude -= metadata["dist_modulus"]

    lightcurve_data = lightcurve_data.with_columns(magnitude, pl.col("time") - metadata["timecorrection"])

    return lightcurve_data, metadata


def read_bol_reflightcurve_data(lightcurvefilename: str | Path) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return an observed bolometric light curve and its metadata, from a given path or the bundled reference data."""
    data_path = (
        Path(lightcurvefilename)
        if Path(lightcurvefilename).is_file()
        else Path(at.get_path("artistools_dir"), "data/lightcurves/bollightcurves", lightcurvefilename)
    )

    metadata = at.get_file_metadata(data_path)

    # the column names come from a leading comment line if there is one
    dflightcurve = at.read_wsv(data_path, has_header=False, comment_prefix="#", header_from_comment=True)

    if colrenames := {
        k: v
        for k, v in {dflightcurve.columns[0]: "time_days", dflightcurve.columns[1]: "luminosity_erg/s"}.items()
        if k != v
    }:
        print(f"{data_path}: renaming columns {colrenames}")
        dflightcurve = dflightcurve.rename(colrenames)

    return dflightcurve, metadata


def get_phillips_relation_data() -> tuple[pl.DataFrame, str]:
    """Return the observed dm15(B) against peak MB data of Hicken et al. (2009), and its plot label."""
    datafilepath = Path(at.get_path("artistools_dir"), "data", "lightcurves", "SNsample", "CfA3_Phillips.dat")
    sn_data = at.read_wsv(datafilepath, comment_prefix="#").with_columns(
        pl.col("dm15(B)").cast(pl.Float64), pl.col("MB").cast(pl.Float64)
    )
    print(sn_data)

    return sn_data, "Observed (Hicken et al. 2009)"
