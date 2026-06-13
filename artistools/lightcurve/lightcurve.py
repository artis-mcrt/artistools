import argparse
import math
import typing as t
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import matplotlib.container as mplcontainer
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import megaparsec_to_cm
from artistools.misc import directionspec_index
from artistools.misc import directionspec_kind
from artistools.misc import directionspec_legacy_key
from artistools.misc import directionspec_packet_filter


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


def get_vspecpol_lightcurve(modelpath: str | Path, vspecindex: int) -> pl.LazyFrame:
    """Get a bolometric light curve for a virtual packet observer by integrating the vspecpol fluxes over frequency."""
    vspecdata = at.spectra.get_vspecpol_data(vspecindex=vspecindex, modelpath=modelpath)["I"].collect()
    timecolumns = vspecdata.columns[1:]
    arr_nu = vspecdata["nu"].to_numpy()
    luminosity_erg_per_s = [
        float(abs(np.trapezoid(vspecdata[timecolumn].to_numpy(), x=arr_nu))) * 4 * math.pi * megaparsec_to_cm**2
        for timecolumn in timecolumns
    ]

    return pl.LazyFrame({
        "time_days": [float(timecolumn) for timecolumn in timecolumns],
        "luminosity_Lsun": np.array(luminosity_erg_per_s) / Lsun_to_erg_per_s,
    }).with_columns(
        (4.74 - (2.5 * pl.col("luminosity_Lsun").log10())).alias("mag"),
        (cs.ends_with("_Lsun") * Lsun_to_erg_per_s).name.replace("_Lsun", "_erg/s"),
    )


def get_lightcurves_by_directionspec(
    modelpath: str | Path,
    directionspecs: Sequence[str],
    *,
    frompackets: bool = False,
    escape_type: str = "TYPE_RPKT",
    maxpacketfiles: int | None = None,
    lcfilename: str | None = None,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
    timedaysmin: float | int | None = None,
    timedaysmax: float | int | None = None,
) -> dict[str, pl.LazyFrame]:
    """Get light curves for any mix of direction specifications (real packet direction bins with per-spec phi/theta averaging, and virtual packet observers).

    Specs are grouped by the loader mode required, with one call to the existing loaders per group.
    """
    lcdataout: dict[str, pl.LazyFrame] = {}
    vpktspecs = [spec for spec in directionspecs if directionspec_kind(spec) == "vpkt"]
    realspecs = [spec for spec in directionspecs if directionspec_kind(spec) != "vpkt"]

    if frompackets:
        # virtual and real packets come from different files, so they need one loader call each
        if vpktspecs:
            lcdataout |= get_from_packets(
                modelpath,
                escape_type=escape_type,
                maxpacketfiles=maxpacketfiles,
                directionbins=vpktspecs,
                timedaysmin=timedaysmin,
                timedaysmax=timedaysmax,
            )
        if realspecs:
            lcdataout |= get_from_packets(
                modelpath,
                escape_type=escape_type,
                maxpacketfiles=maxpacketfiles,
                directionbins=realspecs,
                pellet_nucname=pellet_nucname,
                use_pellet_decay_time=use_pellet_decay_time,
                timedaysmin=timedaysmin,
                timedaysmax=timedaysmax,
            )
    else:
        for spec in vpktspecs:
            lcdataout[spec] = get_vspecpol_lightcurve(modelpath, directionspec_index(spec))

        if realspecs:
            lcdata: dict[int, pl.LazyFrame] = {}
            if lcfilename is not None:
                lcdata = readfile(at.firstexisting(lcfilename, folder=modelpath, tryzipped=True))
            else:
                if any(directionspec_kind(spec) != "sphere" for spec in realspecs):
                    lcdata |= readfile(at.firstexisting("light_curve_res.out", folder=modelpath, tryzipped=True))
                if any(directionspec_kind(spec) == "sphere" for spec in realspecs):
                    spherefilename = "gamma_light_curve.out" if escape_type == "TYPE_GAMMA" else "light_curve.out"
                    lcdata |= readfile(at.firstexisting(spherefilename, folder=modelpath, tryzipped=True))

            plainspecs = [spec for spec in realspecs if directionspec_kind(spec) in {"sphere", "dirbin"}]
            lcdataout |= {
                spec: lcdata[directionspec_legacy_key(spec)]
                for spec in plainspecs
                if directionspec_legacy_key(spec) in lcdata
            }

            overanglegroups: list[tuple[t.Literal["phi", "theta"], list[str]]] = [
                ("phi", [spec for spec in realspecs if directionspec_kind(spec) == "costheta"]),
                ("theta", [spec for spec in realspecs if directionspec_kind(spec) == "phi"]),
            ]
            for overangle, specgroup in overanglegroups:
                if specgroup:
                    averaged = at.average_direction_bins(lcdata, overangle=overangle)
                    lcdataout |= {spec: averaged[directionspec_legacy_key(spec)] for spec in specgroup}

    # return in the same order as the requested directionspecs
    return {spec: lcdataout[spec] for spec in directionspecs if spec in lcdataout}


def get_from_packets(
    modelpath: str | Path,
    escape_type: str = "TYPE_RPKT",
    maxpacketfiles: int | None = None,
    directionbins: Sequence[str] | None = None,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
    timedaysmin: float | int | None = None,
    timedaysmax: float | int | None = None,
) -> dict[str, pl.LazyFrame]:
    """Get ARTIS luminosity vs time from packets files for each requested direction spec.

    directionbins is a list of direction spec tokens (e.g. "all", "23", "costheta2", "phi3", or "vpkt1").
    Virtual packet observer specs ("vpkt...") read from the virtual packets and cannot be mixed with real
    direction bins in a single call. The returned dict is keyed by the same tokens.
    """
    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)
    if directionbins is None:
        directionbins = ["all"]
    vpkt_observers = any(directionspec_kind(spec) == "vpkt" for spec in directionbins)
    assert all((directionspec_kind(spec) == "vpkt") == vpkt_observers for spec in directionbins), (
        "cannot mix virtual packet observers with real direction bins in a single call"
    )

    dftimesteps_selected = at.misc.df_filter_minmax_bounded(
        at.get_timesteps(modelpath), "tmid_days", timedaysmin, timedaysmax
    ).collect()

    _, modelmeta = at.inputmodel.get_modeldata(modelpath, printwarningsonly=True)

    timebinstarts_plusend = [
        *dftimesteps_selected["tstart_days"],
        dftimesteps_selected.select(pl.col("tstart_days").last() + pl.col("twidth_days").last()).item(),
    ]

    vpkt_config = at.get_vpkt_config(modelpath) if vpkt_observers else None
    assert not vpkt_observers or pellet_nucname is None  # we don't track which pellet led to vpkts
    if vpkt_observers:
        nprocs_read, dfpackets = at.packets.get_virtual_packets(modelpath, maxpacketfiles=maxpacketfiles)
    else:
        nprocs_read, dfpackets = at.packets.get_packets(
            modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type=escape_type
        )
        escapesurfacegamma = math.sqrt(1 - (modelmeta["vmax_cmps"] / 29979245800) ** 2)
        dfpackets = dfpackets.with_columns([
            (pl.col("escape_time") * escapesurfacegamma / 86400.0).alias("t_arrive_cmf_d")
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
        assert not vpkt_observers
        dfpackets = dfpackets.with_columns([(pl.col("tdecay") / 86400).alias("tdecay_d")])

    timecol = "tdecay_d" if use_pellet_decay_time else "t_arrive_d"

    lcdata: dict[str, pl.LazyFrame] = {}
    for spec in directionbins:
        if vpkt_observers:
            assert vpkt_config is not None
            vspecindex = directionspec_index(spec)
            obsdirindex = vspecindex // vpkt_config["nspectraperobs"]
            opacchoiceindex = vspecindex % vpkt_config["nspectraperobs"]
            pldfpackets_dirbin = dfpackets.with_columns(
                e_rf=pl.col(f"dir{obsdirindex}_e_rf_{opacchoiceindex}"),
                t_arrive_d=pl.col(f"dir{obsdirindex}_t_arrive_d"),
            )
            solidanglefactor = 4 * math.pi
        else:
            filterexpr, solidanglefactor = directionspec_packet_filter(spec)
            pldfpackets_dirbin = dfpackets.filter(filterexpr)

        lcdata[spec] = (
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
                    * solidanglefactor
                    / (pl.col("twidth_days") * 86400)
                    / Lsun_to_erg_per_s
                )
            )
            .drop("e_rf_sum", f"{timecol}_bin")
        )

        if "t_arrive_cmf_d" in pldfpackets_dirbin.collect_schema().names():
            lcdata[spec] = (
                lcdata[spec]
                .join(
                    at.packets.bin_and_sum(
                        pldfpackets_dirbin, bincol="t_arrive_cmf_d", bins=timebinstarts_plusend, sumcols=["e_cmf"]
                    ).rename({"t_arrive_cmf_d_bin": "timestep"}),
                    how="left",
                    on="timestep",
                )
                .with_columns(
                    luminosity_cmf_Lsun=pl.col("e_cmf_sum")
                    / nprocs_read
                    * solidanglefactor
                    / escapesurfacegamma
                    / (pl.col("twidth_days") * 86400)
                    / Lsun_to_erg_per_s
                )
                .drop("e_cmf_sum")
            )

        lcdata[spec] = (
            lcdata[spec]
            .rename({"tmid_days": "time_days"})
            .drop("twidth_days")
            .with_columns(
                (4.74 - (2.5 * pl.col("luminosity_Lsun").log10())).alias("mag"),
                (cs.ends_with("_Lsun") * Lsun_to_erg_per_s).name.replace("_Lsun", "_erg/s"),
            )
        )

    return lcdata


def generate_band_lightcurve_data(
    modelpath: Path | str,
    args: argparse.Namespace,
    dirbin: int = -1,
    modelnumber: int | None = None,  # noqa: ARG001
) -> dict[str, t.Any]:
    """Integrate spectra to get band magnitude vs time. Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py."""
    from scipy.interpolate import interp1d

    if args.plotvspecpol and Path(modelpath, "vpkt.txt").is_file():
        print("Found vpkt.txt, using virtual packets")
        stokes_params = (
            at.spectra.get_vspecpol_data(vspecindex=dirbin, modelpath=modelpath)
            if dirbin >= 0
            else at.spectra.get_specpol_data(dirbin=dirbin, modelpath=modelpath)
        )
        vspecdata = stokes_params["I"]
        timearray = vspecdata.columns[1:]
    elif args.plotviewingangle and at.anyexist(["specpol_res.out", "spec_res.out"], folder=modelpath, tryzipped=True):
        specfilename = at.firstexisting(["specpol_res.out", "spec_res.out"], folder=modelpath, tryzipped=True)
        specdataresdata = pd.read_csv(specfilename, sep=r"\s+")
        timearray = [i for i in specdataresdata.columns.to_numpy()[1:] if i[-2] != "."]
    # elif Path(modelpath, 'specpol.out').is_file():
    #     specfilename = os.path.join(modelpath, "specpol.out")
    #     specdata = pd.read_csv(specfilename, sep='\s+')
    #     timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        if args.plotviewingangle:
            print("WARNING: no direction-resolved spectra available. Using angle-averaged spectra.")

        specfilename = at.firstexisting(["spec.out", "specpol.out"], folder=modelpath, tryzipped=True)
        specdata = pd.read_csv(specfilename, sep=r"\s+")

        timearray = (
            # Ignore Q and U values in pol file
            [i for i in specdata.columns.to_numpy()[1:] if i[-2] != "."]
            if "specpol.out" in str(specfilename)
            else specdata.columns.to_list()[1:]
        )

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

                if len(wavelength_from_spectrum) > len(wavefilter):
                    interpolate_fn = interp1d(x=wavefilter, y=transmission, bounds_error=False, fill_value=0.0)  # type: ignore[arg-type] # pyright: ignore[reportArgumentType] # pyrefly: ignore [bad-argument-type]
                    wavefilter = np.linspace(
                        np.min(wavelength_from_spectrum),
                        int(np.max(wavelength_from_spectrum)),
                        len(wavelength_from_spectrum),
                    )
                    transmission = interpolate_fn(wavefilter)
                else:
                    interpolate_fn = interp1d(wavelength_from_spectrum, flux, bounds_error=False, fill_value=0.0)
                    wavelength_from_spectrum = np.linspace(wavefilter_min, wavefilter_max, len(wavefilter))
                    flux = interpolate_fn(wavelength_from_spectrum)

                weighted_flux_obs = abs(np.trapezoid(flux * transmission, wavelength_from_spectrum))  # pyright: ignore[reportArgumentType]
                assert isinstance(weighted_flux_obs, float)
                phot_filtobs_sn: float | int = (
                    0.0 if weighted_flux_obs == 0.0 else -2.5 * np.log10(weighted_flux_obs / zeropointenergyflux)
                )

                if phot_filtobs_sn != 0.0:
                    phot_filtobs_sn -= 25  # Absolute magnitude
                filters_dict[filter_name].append((time, phot_filtobs_sn))

    return filters_dict


def bolometric_magnitude(
    modelpath: Path,
    timearray: Collection[float | str],
    args: argparse.Namespace,
    angle: int = -1,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> tuple[list[float], list[float]]:

    magnitudes = []
    times = []

    Mpc_to_cm = 3.085677581491367e24
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
) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating], float, int]:
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

    wavefilter_min = min(wavefilter)
    wavefilter_max = int(max(wavefilter))

    return zeropointenergyflux, np.array(wavefilter), np.array(transmission), wavefilter_min, wavefilter_max


def get_spectrum_in_filter_range(
    modelpath: Path | str,
    timestep: int,
    time: float | int,
    wavefilter_min: float | int,
    wavefilter_max: float | int,
    angle: int = -1,
    args: argparse.Namespace | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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

    wavelength_from_spectrum, flux = [], []
    for wavelength, flambda in zip(spectrum["lambda_angstroms"], spectrum["f_lambda"], strict=True):
        if wavefilter_min <= wavelength <= wavefilter_max:  # to match the spectrum wavelengths to those of the filter
            wavelength_from_spectrum.append(wavelength)
            flux.append(flambda)

    return np.array(wavelength_from_spectrum), np.array(flux)


def get_band_lightcurve(
    band_lightcurve_data: dict[str, Sequence[tuple[float, float]]], band_name: str, args: argparse.Namespace
) -> tuple[Sequence[float], npt.NDArray[np.floating]]:
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
    band_lightcurve_data: dict[str, Iterable[t.Any]], filter_names: Sequence[str]
) -> tuple[list[float], list[float]]:
    time_dict_1 = {}
    time_dict_2 = {}

    plot_times = []
    colour_delta_mag = []

    for filter_1, filter_2 in zip(
        band_lightcurve_data[filter_names[0]], band_lightcurve_data[filter_names[1]], strict=True
    ):
        # Make magnitude dictionaries where time is the key
        time_dict_1[float(filter_1[0])] = filter_1[1]
        time_dict_2[float(filter_2[0])] = filter_2[1]

    for time in time_dict_1 | time_dict_2:
        plot_times.append(time)
        colour_delta_mag.append(time_dict_1[time] - time_dict_2[time])

    return plot_times, colour_delta_mag


def read_hesma_lightcurve(args: argparse.Namespace) -> pd.DataFrame:
    hesma_directory = Path(at.get_path("artistools_dir"), "data/hesma")
    filename = args.plot_hesma_model
    hesma_modelname = hesma_directory / filename

    column_names: list[str] = []
    with hesma_modelname.open(encoding="utf-8") as f:
        first_line = f.readline()
        if "#" in first_line:
            column_names.extend(i for i in first_line if i not in {"#", " ", "\n"})
            hesma_model = pd.read_csv(hesma_modelname, sep=r"\s+", header=None, comment="#", names=column_names)

        else:
            hesma_model = pd.read_csv(hesma_modelname, sep=r"\s+")
    return hesma_model


def read_reflightcurve_band_data(lightcurvefilename: Path | str) -> tuple[pd.DataFrame, dict[str, t.Any]]:
    filepath = Path(at.get_path("artistools_dir"), "data", "lightcurves", lightcurvefilename)
    metadata = at.get_file_metadata(filepath)

    data_path = Path(at.get_path("artistools_dir"), f"data/lightcurves/{lightcurvefilename}")
    lightcurve_data = pd.read_csv(data_path, comment="#")
    if len(lightcurve_data.keys()) == 1:
        lightcurve_data = pd.read_csv(data_path, comment="#", sep=r"\s+")

    lightcurve_data["magnitude"] = pd.to_numeric(lightcurve_data["magnitude"])  # force column to be float

    lightcurve_data["time"] = lightcurve_data["time"].apply(lambda x: x - (metadata["timecorrection"]))
    # m - M = 5log(d) - 5  Get absolute magnitude
    if "dist_mpc" not in metadata and "z" in metadata:
        from astropy import cosmology

        cosmo = (
            cosmology.FlatLambdaCDM(H0=70, Om0=0.3)  # ty: ignore[unknown-argument] # pyright: ignore[reportCallIssue] # pyrefly: ignore[unexpected-keyword] # zuban: ignore[call-arg]
        )
        metadata["dist_mpc"] = cosmo.luminosity_distance(metadata["z"]).value  # pyright: ignore[reportAttributeAccessIssue]
        print(f"luminosity distance from redshift = {metadata['dist_mpc']} for {metadata['label']}")

    if "dist_mpc" in metadata:
        lightcurve_data["magnitude"] = lightcurve_data["magnitude"].apply(
            lambda x: x - 5 * np.log10(metadata["dist_mpc"] * 10**6) + 5
        )
    elif "dist_modulus" in metadata:
        lightcurve_data["magnitude"] = lightcurve_data["magnitude"].apply(lambda x: x - metadata["dist_modulus"])

    return lightcurve_data, metadata


def read_bol_reflightcurve_data(lightcurvefilename: str | Path) -> tuple[pd.DataFrame, dict[str, t.Any]]:
    data_path = (
        Path(lightcurvefilename)
        if Path(lightcurvefilename).is_file()
        else Path(at.get_path("artistools_dir"), "data/lightcurves/bollightcurves", lightcurvefilename)
    )

    metadata = at.get_file_metadata(data_path)

    # check for possible header line and read table
    with data_path.open(encoding="utf-8") as flc:
        filepos = flc.tell()
        line = flc.readline()
        if line.startswith("#"):
            columns = line.lstrip("#").split()
        else:
            flc.seek(filepos)  # undo the readline() and go back
            columns = None

        dflightcurve = pd.read_csv(flc, sep=r"\s+", header=None, names=columns)

    if colrenames := {
        k: v
        for k, v in {dflightcurve.columns[0]: "time_days", dflightcurve.columns[1]: "luminosity_erg/s"}.items()
        if k != v
    }:
        print(f"{data_path}: renaming columns {colrenames}")
        dflightcurve = dflightcurve.rename(columns=colrenames)

    return dflightcurve, metadata


def get_sn_sample_bol() -> tuple[t.Any, str]:
    datafilepath = Path(at.get_path("artistools_dir"), "data", "lightcurves", "SNsample", "bololc.txt")
    sn_data = pd.read_csv(datafilepath, sep=r"\s+", comment="#")

    print(sn_data)
    bol_luminosity = sn_data["Lmax"].astype(float)
    bol_magnitude = 4.74 - (2.5 * np.log10((10**bol_luminosity) / Lsun_to_erg_per_s))  # Mbol,sun = 4.74

    bol_magnitude_error_upper = bol_magnitude - (
        4.74 - (2.5 * np.log10((10 ** (bol_luminosity + sn_data["+/-.2"].astype(float))) / Lsun_to_erg_per_s))
    )
    # bol_magnitude_error_lower = (4.74 - (2.5 * np.log10
    #     10**(bol_luminosity - sn_data['+/-.2'].astype(float))) / Lsun_in_erg_per_s))) - bol_magnitude
    # print(bol_magnitude_error_upper, "============")
    # print(bol_magnitude_error_lower, "============")
    # print(bol_magnitude_error_upper == bol_magnitude_error_lower)

    # a0 = plt.errorbar(x=sn_data['dm15'].astype(float), y=sn_data['Lmax'].astype(float),
    #                   yerr=sn_data['+/-.2'].astype(float), xerr=sn_data['+/-'].astype(float),
    #                   color='grey', marker='o', ls='None')
    #
    sn_data["bol_mag"] = bol_magnitude
    print(sn_data[["name", "bol_mag", "dm15", "dm40"]])
    sn_data[["name", "bol_mag", "dm15", "dm40"]].to_csv("boldata.txt", sep=" ", index=False)
    a0 = plt.errorbar(
        x=sn_data["dm15"].astype(float),
        y=bol_magnitude,
        yerr=bol_magnitude_error_upper,
        xerr=sn_data["+/-"].astype(float),
        color="k",
        marker="o",
        ls="None",
    )

    # a0 = plt.errorbar(x=sn_data['dm15'].astype(float), y=sn_data['dm40'].astype(float),
    #                   yerr=sn_data['+/-.1'].astype(float), xerr=sn_data['+/-'].astype(float),
    #                   color='k', marker='o', ls='None')

    # a0 = plt.scatter(sn_data['dm15'].astype(float), bol_magnitude, s=80, color='k', marker='o')
    # plt.gca().invert_yaxis()
    # plt.show()

    label = "Bolometric data (Scalzo et al. 2019)"
    return a0, label


def get_phillips_relation_data() -> tuple[pd.DataFrame, str]:
    datafilepath = Path(at.get_path("artistools_dir"), "data", "lightcurves", "SNsample", "CfA3_Phillips.dat")
    sn_data = pd.read_csv(datafilepath, sep=r"\s+", comment="#")
    print(sn_data)

    sn_data["dm15(B)"] = sn_data["dm15(B)"].astype(float)
    sn_data["MB"] = sn_data["MB"].astype(float)

    return sn_data, "Observed (Hicken et al. 2009)"


def plot_phillips_relation_data() -> tuple[mplcontainer.ErrorbarContainer, str]:
    sn_data, label = get_phillips_relation_data()

    # a0 = plt.scatter(deltam_15B, M_B, s=80, color='grey', marker='o', label=label)
    a0 = plt.errorbar(
        x=sn_data["dm15(B)"],
        y=sn_data["MB"],
        yerr=sn_data["err_MB"],
        xerr=sn_data["err_dm15(B)"],
        color="k",
        alpha=0.9,
        marker=".",
        capsize=2,
        label=label,
        ls="None",
        zorder=5,
    )
    # plt.gca().invert_yaxis()
    # plt.show()
    return a0, label
