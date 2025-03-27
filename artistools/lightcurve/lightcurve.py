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

import artistools as at


def readfile(filepath: str | Path) -> dict[int, pl.DataFrame]:
    """Read an ARTIS light curve file."""
    print(f"Reading {filepath}")
    lcdata: dict[int, pl.DataFrame] = {}
    if "_res" in str(Path(filepath).stem):
        # get a dict of dfs with light curves at each viewing direction bin
        lcdata_res = pl.read_csv(
            at.zopenpl(filepath), separator=" ", has_header=False, new_columns=["time", "lum", "lum_cmf"]
        )
        lcdata = at.split_multitable_dataframe(lcdata_res)
    else:
        dfsphericalaverage = pl.read_csv(
            at.zopenpl(filepath), separator=" ", has_header=False, new_columns=["time", "lum", "lum_cmf"]
        )

        if list(dfsphericalaverage["time"].to_numpy()) != sorted(dfsphericalaverage["time"].to_numpy()):
            # the light_curve.out file repeats x values, so keep the first half only
            dfsphericalaverage = dfsphericalaverage[: dfsphericalaverage.height // 2]
        lcdata[-1] = dfsphericalaverage

    return lcdata


def read_3d_gammalightcurve(filepath: str | Path) -> dict[int, pd.DataFrame]:
    columns = ["time"]
    columns.extend(np.arange(0, 100))
    lcdata = pd.read_csv(filepath, sep=r"\s+", header=None).set_axis(columns, axis=1)
    # lcdata = lcdata.rename(columns={0: 'time', 1: 'lum', 2: 'lum_cmf'})

    res_data = {}
    for angle in range(100):
        res_data[angle] = lcdata[["time", angle]]
        res_data[angle] = res_data[angle].rename(columns={angle: "lum"})

    return res_data


def get_from_packets(
    modelpath: str | Path,
    escape_type: str = "TYPE_RPKT",
    maxpacketfiles: int | None = None,
    directionbins: Collection[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    get_cmf_column: bool = True,
    directionbins_are_vpkt_observers: bool = False,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
) -> dict[int, pl.DataFrame]:
    """Get ARTIS luminosity vs time from packets files."""
    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)
    if directionbins is None:
        directionbins = [-1]
    tmidarray = at.get_timestep_times(modelpath=modelpath, loc="mid")
    timearray = at.get_timestep_times(modelpath=modelpath, loc="start")
    arr_timedelta = at.get_timestep_times(modelpath=modelpath, loc="delta")
    # timearray = np.arange(250, 350, 0.1)
    if get_cmf_column:
        _, modelmeta = at.inputmodel.get_modeldata(modelpath, getheadersonly=True, printwarningsonly=True)
        escapesurfacegamma = math.sqrt(1 - (modelmeta["vmax_cmps"] / 29979245800) ** 2)
    else:
        escapesurfacegamma = None

    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()
    ndirbins = at.get_viewingdirectionbincount()

    vpkt_config = None
    if directionbins_are_vpkt_observers:
        assert pellet_nucname is None  # we don't track which pellet led to vpkts
        vpkt_config = at.get_vpkt_config(modelpath)
        nprocs_read, dfpackets = at.packets.get_virtual_packets_pl(modelpath, maxpacketfiles=maxpacketfiles)
    else:
        nprocs_read, dfpackets = at.packets.get_packets_pl(
            modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type=escape_type
        )

    if get_cmf_column:
        dfpackets = dfpackets.with_columns([
            (pl.col("escape_time") * escapesurfacegamma / 86400.0).alias("t_arrive_cmf_d")
        ])

    try:
        dfnuclides = at.get_nuclides(modelpath=modelpath)
        if pellet_nucname is not None:
            atomic_number = at.get_atomic_number(pellet_nucname)
            if at.get_elsymbol(atomic_number) == pellet_nucname:
                expr = pl.col("atomic_number") == atomic_number
            else:
                expr = pl.col("nucname") == pellet_nucname
            dfpackets = dfpackets.filter(
                pl.col("pellet_nucindex").is_in(
                    dfnuclides.filter(expr).select(["pellet_nucindex"]).collect().get_column("pellet_nucindex")
                )
            )
    except FileNotFoundError:
        assert pellet_nucname is None

    if use_pellet_decay_time:
        assert not directionbins_are_vpkt_observers
        dfpackets = dfpackets.with_columns([(pl.col("tdecay") / 86400).alias("tdecay_d")])

    timecol = "tdecay_d" if use_pellet_decay_time else "t_arrive_d"
    erg_per_day_to_Lsun = 3.023530322380897e-39

    lazyframes = []
    for dirbin in directionbins:
        if directionbins_are_vpkt_observers:
            assert vpkt_config is not None
            obsdirindex = dirbin // vpkt_config["nspectraperobs"]
            opacchoiceindex = dirbin % vpkt_config["nspectraperobs"]
            pldfpackets_dirbin = dfpackets.with_columns(
                e_rf=pl.col(f"dir{obsdirindex}_e_rf_{opacchoiceindex}"),
                t_arrive_d=pl.col(f"dir{obsdirindex}_t_arrive_d"),
            )
            solidanglefactor = 4 * math.pi
        elif dirbin == -1:
            solidanglefactor = 1.0
            pldfpackets_dirbin = dfpackets
        elif average_over_phi:
            assert not average_over_theta
            solidanglefactor = ncosthetabins
            pldfpackets_dirbin = dfpackets.filter(pl.col("costhetabin") * nphibins == dirbin)
        elif average_over_theta:
            solidanglefactor = nphibins
            pldfpackets_dirbin = dfpackets.filter(pl.col("phibin") == dirbin)
        else:
            solidanglefactor = ndirbins
            pldfpackets_dirbin = dfpackets.filter(pl.col("dirbin") == dirbin)

        dftimebinned = at.packets.bin_and_sum(
            pldfpackets_dirbin, bincol=timecol, bins=list(timearrayplusend), sumcols=["e_rf"], getcounts=True
        )

        dftimebinned = dftimebinned.with_columns([
            pl.Series(name="time", values=tmidarray),
            (
                (pl.col("e_rf_sum") / nprocs_read * solidanglefactor * erg_per_day_to_Lsun) / pl.Series(arr_timedelta)
            ).alias("lum"),
        ]).drop(["e_rf_sum", f"{timecol}_bin"])

        if get_cmf_column:
            dftimebinned_cmf = at.packets.bin_and_sum(
                pldfpackets_dirbin, bincol="t_arrive_cmf_d", bins=list(timearrayplusend), sumcols=["e_cmf"]
            ).collect()

            assert escapesurfacegamma is not None
            dftimebinned = dftimebinned.with_columns(
                pl.Series(
                    name="lum_cmf",
                    values=(
                        dftimebinned_cmf.get_column("e_cmf_sum").to_numpy()
                        / nprocs_read
                        * solidanglefactor
                        / escapesurfacegamma
                        * erg_per_day_to_Lsun
                        / arr_timedelta
                    ),
                )
            )

        lazyframes.append(dftimebinned)

    lcdata = dict(zip(directionbins, pl.collect_all(lazyframes), strict=True))
    for dirbin, df in lcdata.items():
        npkts_selected = df.select(pl.col("count").sum()).item()
        print(f"    dirbin {dirbin} plotting {npkts_selected:.2e} packets")

    return lcdata


def generate_band_lightcurve_data(
    modelpath: Path,
    args: argparse.Namespace,
    angle: int = -1,
    modelnumber: int | None = None,  # noqa: ARG001
) -> dict[str, t.Any]:
    """Integrate spectra to get band magnitude vs time. Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py."""
    from scipy.interpolate import interp1d

    if args.plotvspecpol and (modelpath / "vpkt.txt").is_file():
        print("Found vpkt.txt, using virtual packets")
        stokes_params = (
            at.spectra.get_vspecpol_data(vspecindex=angle, modelpath=modelpath)
            if angle >= 0
            else at.spectra.get_specpol_data(angle=angle, modelpath=modelpath)
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
                modelpath,
                timearray,
                args,
                angle=angle,
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

    filterdir = Path(at.get_config()["path_artistools_dir"], "data/filters/")

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
                    angle=angle,
                    args=args,
                    average_over_phi=args.average_over_phi_angle,
                    average_over_theta=args.average_over_theta_angle,
                )

                if len(wavelength_from_spectrum) > len(wavefilter):
                    interpolate_fn = interp1d(wavefilter, transmission, bounds_error=False, fill_value=0.0)
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

                phot_filtobs_sn = evaluate_magnitudes(flux, transmission, wavelength_from_spectrum, zeropointenergyflux)

                # print(time, phot_filtobs_sn)
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
    from scipy import integrate

    magnitudes = []
    times = []

    Mpc_to_cm = 3.085677581491367e24
    Lsun_to_erg_per_s = 3.828e33
    for timestep, time in enumerate(float(time) for time in timearray):
        if (args.timemin is None or args.timemin <= time) and (args.timemax is None or args.timemax >= time):
            if angle == -1:
                spectrum = at.spectra.get_spectrum(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep)[-1]

            elif args.plotvspecpol:
                spectrum = at.spectra.get_vspecpol_spectrum(modelpath, time, angle, args)
            else:
                spectrum = at.spectra.get_spectrum(
                    modelpath=modelpath,
                    directionbins=[angle],
                    timestepmin=timestep,
                    timestepmax=timestep,
                    average_over_phi=average_over_phi,
                    average_over_theta=average_over_theta,
                )[angle]
            integrated_flux = integrate.trapezoid(spectrum["f_lambda"], spectrum["lambda_angstroms"])
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

    wavefilter, transmission = [], []
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
    time: float,
    wavefilter_min: float,
    wavefilter_max: float,
    angle: int = -1,
    spectrum: pd.DataFrame | None = None,
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


def evaluate_magnitudes(
    flux: npt.NDArray[np.floating],
    transmission: npt.NDArray[np.floating],
    wavelength_from_spectrum: npt.NDArray[np.floating],
    zeropointenergyflux: float,
) -> float:
    from scipy import integrate

    cf = flux * transmission
    flux_obs = abs(integrate.trapezoid(cf, wavelength_from_spectrum))  # using trapezoidal rule to integrate
    val = 0.0 if flux_obs == 0.0 else -2.5 * np.log10(flux_obs / zeropointenergyflux)
    assert isinstance(val, float)
    return val


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
    hesma_directory = Path(at.get_config()["path_artistools_dir"], "data/hesma")
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
    filepath = Path(at.get_config()["path_artistools_dir"], "data", "lightcurves", lightcurvefilename)
    metadata = at.get_file_metadata(filepath)

    data_path = Path(at.get_config()["path_artistools_dir"], f"data/lightcurves/{lightcurvefilename}")
    lightcurve_data = pd.read_csv(data_path, comment="#")
    if len(lightcurve_data.keys()) == 1:
        lightcurve_data = pd.read_csv(data_path, comment="#", sep=r"\s+")

    lightcurve_data["magnitude"] = pd.to_numeric(lightcurve_data["magnitude"])  # force column to be float

    lightcurve_data["time"] = lightcurve_data["time"].apply(lambda x: x - (metadata["timecorrection"]))
    # m - M = 5log(d) - 5  Get absolute magnitude
    if "dist_mpc" not in metadata and "z" in metadata:
        from astropy import cosmology

        cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
        metadata["dist_mpc"] = cosmo.luminosity_distance(metadata["z"]).value
        print(f"luminosity distance from redshift = {metadata['dist_mpc']} for {metadata['label']}")

    if "dist_mpc" in metadata:
        lightcurve_data["magnitude"] = lightcurve_data["magnitude"].apply(
            lambda x: (x - 5 * np.log10(metadata["dist_mpc"] * 10**6) + 5)
        )
    elif "dist_modulus" in metadata:
        lightcurve_data["magnitude"] = lightcurve_data["magnitude"].apply(lambda x: (x - metadata["dist_modulus"]))

    return lightcurve_data, metadata


def read_bol_reflightcurve_data(lightcurvefilename: str | Path) -> tuple[pd.DataFrame, dict[str, t.Any]]:
    data_path = (
        Path(lightcurvefilename)
        if Path(lightcurvefilename).is_file()
        else Path(at.get_config()["path_artistools_dir"], "data/lightcurves/bollightcurves", lightcurvefilename)
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
    datafilepath = Path(at.get_config()["path_artistools_dir"], "data", "lightcurves", "SNsample", "bololc.txt")
    sn_data = pd.read_csv(datafilepath, sep=r"\s+", comment="#")

    print(sn_data)
    bol_luminosity = sn_data["Lmax"].astype(float)
    Lsun_to_erg_per_s = 3.826e33
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
    datafilepath = Path(at.get_config()["path_artistools_dir"], "data", "lightcurves", "SNsample", "CfA3_Phillips.dat")
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
