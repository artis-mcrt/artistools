#!/usr/bin/env python3
# import glob
# import itertools
import argparse
import math
import multiprocessing
import os
from collections.abc import Collection
from collections.abc import Sequence
from collections.abc import Sized
from functools import partial
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at
import artistools.spectra


def readfile(
    filepath: Union[str, Path],
) -> dict[int, pd.DataFrame]:
    lcdata: dict[int, pd.DataFrame] = {}
    if "_res" in str(filepath):
        # get a list of dfs with light curves at each viewing angle
        lcdata_res = pd.read_csv(
            filepath, sep=" ", engine=at.get_config()["pandas_engine"], header=None, names=["time", "lum", "lum_cmf"]
        )
        lcdata = at.gather_res_data(lcdata_res, index_of_repeated_value=0)
    else:
        lcdata[-1] = pd.read_csv(
            filepath, sep=" ", engine=at.get_config()["pandas_engine"], header=None, names=["time", "lum", "lum_cmf"]
        )

        if list(lcdata[-1].time.values) != list(sorted(lcdata[-1].time.values)):
            # the light_curve.out file repeats x values, so keep the first half only
            lcdata[-1] = lcdata[-1].iloc[: len(lcdata[-1]) // 2]
            lcdata[-1].index.name = "timestep"

    return lcdata


def read_3d_gammalightcurve(
    filepath: Union[str, Path],
) -> dict[int, pd.DataFrame]:
    columns = ["time"]
    columns.extend(np.arange(0, 100))
    lcdata = pd.read_csv(filepath, delim_whitespace=True, header=None)
    lcdata.columns = columns
    # lcdata = lcdata.rename(columns={0: 'time', 1: 'lum', 2: 'lum_cmf'})

    res_data = {}
    for angle in np.arange(0, 100):
        res_data[angle] = lcdata[["time", angle]].copy()
        res_data[angle].rename(columns={angle: "lum"}, inplace=True)

    return res_data


def get_lightcurve_from_packets_worker(
    packetsfile,
    modelpath,
    tmidarray,
    timearrayplusend,
    escape_type,
    escapesurfacegamma,
    get_cmf_column,
    directionbins,
    contribbinlists,
) -> tuple[dict[int, np.ndarray], dict[int, Optional[np.ndarray]]]:
    dfpackets = at.packets.readfile(packetsfile, type="TYPE_ESCAPE", escape_type=escape_type)
    lum = {dirbin: np.zeros(len(tmidarray), dtype=float) for dirbin in directionbins}
    lum_cmf = {dirbin: np.zeros(len(tmidarray), dtype=float) if get_cmf_column else None for dirbin in directionbins}
    if not (dfpackets.empty):
        if get_cmf_column:
            dfpackets.eval("t_arrive_cmf_d = escape_time * @escapesurfacegamma / 86400", inplace=True)
        if directionbins != [-1]:
            dfpackets = at.packets.bin_packet_directions(modelpath, dfpackets)

        # print(f"sum of e_cmf {dfpackets['e_cmf'].sum()} e_rf {dfpackets['e_rf'].sum()}")

        for dirbin, contribbins in zip(directionbins, contribbinlists):
            dfpackets_dirbin = dfpackets.query("dirbin in @contribbins") if contribbins != [] else dfpackets

            timebins = pd.cut(
                dfpackets_dirbin["t_arrive_d"],
                timearrayplusend,
                labels=range(len(tmidarray)),
                include_lowest=True,
            )
            lum[dirbin] += dfpackets_dirbin.groupby(timebins).e_rf.sum().values

            if get_cmf_column:
                timebins_cmf = pd.cut(
                    dfpackets_dirbin["t_arrive_cmf_d"],
                    timearrayplusend,
                    labels=range(len(tmidarray)),
                    include_lowest=True,
                )
                lum_cmf[dirbin] += dfpackets_dirbin.groupby(timebins_cmf).e_cmf.sum().values

    return lum, lum_cmf


def get_from_packets(
    modelpath: Union[str, Path],
    escape_type: str = "TYPE_RPKT",
    maxpacketfiles: Optional[int] = None,
    directionbins: Sequence[int] = [-1],
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    get_cmf_column: bool = True,
) -> dict[int, pd.DataFrame]:
    import artistools.packets

    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles=maxpacketfiles)
    nprocs_read = len(packetsfiles)
    assert nprocs_read > 0

    tmidarray = at.get_timestep_times_float(modelpath=modelpath, loc="mid")
    timearray = at.get_timestep_times_float(modelpath=modelpath, loc="start")
    arr_timedelta = at.get_timestep_times_float(modelpath=modelpath, loc="delta")
    # timearray = np.arange(250, 350, 0.1)
    if get_cmf_column:
        _, modelmeta = at.inputmodel.get_modeldata(
            modelpath, getheadersonly=True, skipabundancecolumns=True, printwarningsonly=True
        )
        escapesurfacegamma = math.sqrt(1 - (modelmeta["vmax_cmps"] / 29979245800) ** 2)
    else:
        escapesurfacegamma = None

    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    lcdata = {}
    for dirbin in directionbins:
        lcdata[dirbin] = pd.DataFrame(
            {
                "time": tmidarray,
                "lum": np.zeros_like(tmidarray, dtype=float),
                "lum_cmf": np.zeros_like(tmidarray, dtype=float),
            }
        )

    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()
    ndirbins = at.get_viewingdirectionbincount()

    contribbinlists: list[Sized] = []
    for dirbin in directionbins:
        if dirbin == -1:
            contribbinlists.append([])
            continue

        if average_over_phi and average_over_theta:
            assert False
        elif average_over_phi:
            contribbinlists.append(range(dirbin, dirbin + nphibins))
        elif average_over_theta:
            contribbinlists.append(range(dirbin, ndirbins, nphibins))
        else:
            contribbinlists.append([dirbin])

    processfile = partial(
        get_lightcurve_from_packets_worker,
        modelpath=modelpath,
        tmidarray=tmidarray,
        timearrayplusend=timearrayplusend,
        escape_type=escape_type,
        escapesurfacegamma=escapesurfacegamma,
        get_cmf_column=get_cmf_column,
        directionbins=directionbins,
        contribbinlists=contribbinlists,
    )
    if at.get_config()["num_processes"] > 1:
        with multiprocessing.Pool(processes=at.get_config()["num_processes"]) as pool:
            results = pool.map(processfile, packetsfiles)
            pool.close()
            pool.join()
            pool.terminate()
    else:
        results = [processfile(p) for p in packetsfiles]

    for dirbin, contribbins in zip(directionbins, contribbinlists):
        solidanglefactor = float(ndirbins) / len(contribbins) if contribbins != [] else 1.0

        summed_e_rf = np.ufunc.reduce(np.add, [arr_e_rf[dirbin] for arr_e_rf, _ in results])

        lcdata[dirbin]["lum"] = np.divide(
            summed_e_rf / nprocs_read * solidanglefactor * (u.erg / u.day).to("solLum"), arr_timedelta
        )

        if get_cmf_column:
            summed_e_cmf = np.ufunc.reduce(np.add, [arr_e_cmf[dirbin] for _, arr_e_cmf in results])
            lcdata[dirbin]["lum_cmf"] = np.divide(
                summed_e_cmf / nprocs_read * solidanglefactor / escapesurfacegamma * (u.erg / u.day).to("solLum"),
                arr_timedelta,
            )

    return lcdata


def average_lightcurve_phi_bins(lcdataframes: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    dirbincount = at.get_viewingdirectionbincount()
    nphibins = at.get_viewingdirection_phibincount()
    for start_bin in range(0, dirbincount, nphibins):
        for bin_number in range(start_bin + 1, start_bin + nphibins):
            lcdataframes[bin_number] = lcdataframes[bin_number].copy()  # important to not affect the LRU cached copy
            lcdataframes[bin_number] = lcdataframes[bin_number].set_index(
                lcdataframes[start_bin].index
            )  # need indexes to match or else gives NaN
            lcdataframes[start_bin]["lum"] += lcdataframes[bin_number]["lum"]
            del lcdataframes[bin_number]

        lcdataframes[start_bin]["lum"] /= nphibins  # every nth bin is the average of n bins
        print(f"bin number {start_bin} = the average of bins {start_bin} to {start_bin + nphibins-1}")

    return lcdataframes


def average_lightcurve_theta_bins(lcdataframes: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    dirbincount = at.get_viewingdirectionbincount()
    nphibins = at.get_viewingdirection_phibincount()
    ncosthetabins = at.get_viewingdirection_costhetabincount()
    for start_bin in range(0, nphibins):
        contribbins = range(start_bin + ncosthetabins, dirbincount, ncosthetabins)
        for bin_number in contribbins:
            lcdataframes[bin_number] = lcdataframes[bin_number].set_index(
                lcdataframes[start_bin].index
            )  # need indexes to match or else gives NaN
            lcdataframes[start_bin]["lum"] += lcdataframes[bin_number]["lum"]
            del lcdataframes[bin_number]

        lcdataframes[start_bin]["lum"] /= ncosthetabins  # every nth bin is the average of n bins
        print(f"bin number {start_bin} = the average of bins {[start_bin] + list(contribbins)}")

    return lcdataframes


def generate_band_lightcurve_data(
    modelpath: Path,
    args: argparse.Namespace,
    angle: Optional[int] = None,
    modelnumber: Optional[int] = None,
) -> dict:
    """Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py"""
    from scipy.interpolate import interp1d

    if args and args.plotvspecpol and os.path.isfile(modelpath / "vpkt.txt"):
        print("Found vpkt.txt, using virtual packets")
        stokes_params = at.spectra.get_specpol_data(angle, modelpath)
        vspecdata = stokes_params["I"]
        timearray = vspecdata.keys()[1:]
    elif (
        args
        and args.plotviewingangle
        and at.anyexist(["specpol_res.out", "specpol_res.out.xz", "spec_res.out"], path=modelpath)
    ):
        specfilename = at.firstexisting(["specpol_res.out", "spec_res.out"], path=modelpath)
        specdataresdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdataresdata.columns.values[1:] if i[-2] != "."]
    # elif Path(modelpath, 'specpol.out').is_file():
    #     specfilename = os.path.join(modelpath, "specpol.out")
    #     specdata = pd.read_csv(specfilename, delim_whitespace=True)
    #     timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        if args.plotviewingangle:
            print("WARNING: no direction-resolved spectra available. Using angle-averaged spectra.")

        specfilename = at.firstexisting(["spec.out", "specpol.out"], path=modelpath, tryzipped=True)
        if "specpol.out" in str(specfilename):
            specdata = pd.read_csv(specfilename, delim_whitespace=True)
            timearray = [i for i in specdata.columns.values[1:] if i[-2] != "."]  # Ignore Q and U values in pol file
        else:
            specdata = pd.read_csv(specfilename, delim_whitespace=True)
            timearray = specdata.columns.values[1:]

    filters_dict = {}
    if not args.filter:
        args.filter = ["B"]

    filters_list = args.filter

    res_specdata = None
    if angle is not None and angle != -1:
        try:
            res_specdata = at.spectra.read_spec_res(modelpath).copy()
            if args and args.average_over_phi_angle:
                at.spectra.average_phi_bins(res_specdata, angle)

        except FileNotFoundError:
            pass

    for filter_name in filters_list:
        if filter_name == "bol":
            times, bol_magnitudes = bolometric_magnitude(
                modelpath, timearray, args, angle=angle, res_specdata=res_specdata
            )
            filters_dict["bol"] = [
                (time, bol_magnitude)
                for time, bol_magnitude in zip(times, bol_magnitudes)
                if math.isfinite(bol_magnitude)
            ]
        elif filter_name not in filters_dict:
            filters_dict[filter_name] = []

    filterdir = os.path.join(at.get_config()["path_artistools_dir"], "data/filters/")

    for filter_name in filters_list:
        if filter_name == "bol":
            continue
        zeropointenergyflux, wavefilter, transmission, wavefilter_min, wavefilter_max = get_filter_data(
            filterdir, filter_name
        )

        for timestep, time in enumerate(timearray):
            time = float(time)
            if (args.timemin is None or args.timemin <= time) and (args.timemax is None or args.timemax >= time):
                wavelength_from_spectrum, flux = get_spectrum_in_filter_range(
                    modelpath,
                    timestep,
                    time,
                    wavefilter_min,
                    wavefilter_max,
                    angle,
                    res_specdata=res_specdata,
                    modelnumber=modelnumber,
                    args=args,
                )

                if len(wavelength_from_spectrum) > len(wavefilter):
                    interpolate_fn = interp1d(wavefilter, transmission, bounds_error=False, fill_value=0.0)
                    wavefilter = np.linspace(
                        min(wavelength_from_spectrum), int(max(wavelength_from_spectrum)), len(wavelength_from_spectrum)
                    )
                    transmission = interpolate_fn(wavefilter)
                else:
                    interpolate_fn = interp1d(wavelength_from_spectrum, flux, bounds_error=False, fill_value=0.0)
                    wavelength_from_spectrum = np.linspace(wavefilter_min, wavefilter_max, len(wavefilter))
                    flux = interpolate_fn(wavelength_from_spectrum)

                phot_filtobs_sn = evaluate_magnitudes(flux, transmission, wavelength_from_spectrum, zeropointenergyflux)

                # print(time, phot_filtobs_sn)
                if phot_filtobs_sn != 0.0:
                    phot_filtobs_sn = phot_filtobs_sn - 25  # Absolute magnitude
                filters_dict[filter_name].append((time, phot_filtobs_sn))

    return filters_dict


def bolometric_magnitude(
    modelpath: Path,
    timearray: Collection[float],
    args: argparse.Namespace,
    angle: Optional[int] = None,
    res_specdata: Optional[dict[int, pd.DataFrame]] = None,
) -> tuple[list[float], list[float]]:
    magnitudes = []
    times = []

    for timestep, time in enumerate(timearray):
        time = float(time)
        if (args.timemin is None or args.timemin <= time) and (args.timemax is None or args.timemax >= time):
            if angle is not None and angle != -1:
                if args.plotvspecpol:
                    spectrum = at.spectra.get_vspecpol_spectrum(modelpath, time, angle, args)
                else:
                    if res_specdata is None:
                        res_specdata = at.spectra.read_spec_res(modelpath)
                        if args and args.average_over_phi_angle:
                            at.spectra.average_phi_bins(res_specdata, angle)
                    spectrum = at.spectra.get_res_spectrum(
                        modelpath, timestep, timestep, angle=angle, res_specdata=res_specdata
                    )
            else:
                spectrum = at.spectra.get_spectrum(modelpath, timestep, timestep)

            integrated_flux = np.trapz(spectrum["f_lambda"], spectrum["lambda_angstroms"])
            integrated_luminosity = integrated_flux * 4 * np.pi * np.power(u.Mpc.to("cm"), 2)
            Mbol_sun = 4.74
            magnitude = Mbol_sun - (2.5 * np.log10(integrated_luminosity / const.L_sun.to("erg/s").value))
            magnitudes.append(magnitude)
            times.append(time)
            # print(const.L_sun.to('erg/s').value)
            # quit()

    return times, magnitudes


def get_filter_data(
    filterdir: Union[Path, str], filter_name: str
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Filter data in 'data/filters' taken from https://github.com/cinserra/S3/tree/master/src/s3/metadata"""

    with Path(filterdir, filter_name + ".txt").open("r") as filter_metadata:  # defintion of the file
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
    modelpath: Union[Path, str],
    timestep: int,
    time: float,
    wavefilter_min: float,
    wavefilter_max: float,
    angle: Optional[int] = None,
    res_specdata: Optional[dict[int, pd.DataFrame]] = None,
    modelnumber: Optional[int] = None,
    spectrum: Optional[pd.DataFrame] = None,
    args: Optional[argparse.Namespace] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if spectrum is None:
        spectrum = at.spectra.get_spectrum_at_time(
            Path(modelpath),
            timestep=timestep,
            time=time,
            args=args,
            angle=angle,
            res_specdata=res_specdata,
            modelnumber=modelnumber,
        )

    wavelength_from_spectrum, flux = [], []
    for wavelength, flambda in zip(spectrum["lambda_angstroms"], spectrum["f_lambda"]):
        if wavefilter_min <= wavelength <= wavefilter_max:  # to match the spectrum wavelengths to those of the filter
            wavelength_from_spectrum.append(wavelength)
            flux.append(flambda)

    return np.array(wavelength_from_spectrum), np.array(flux)


def evaluate_magnitudes(flux, transmission, wavelength_from_spectrum, zeropointenergyflux: float) -> float:
    cf = flux * transmission
    flux_obs = abs(np.trapz(cf, wavelength_from_spectrum))  # using trapezoidal rule to integrate
    if flux_obs == 0.0:
        phot_filtobs_sn = 0.0
    else:
        phot_filtobs_sn = -2.5 * np.log10(flux_obs / zeropointenergyflux)

    return phot_filtobs_sn


def get_band_lightcurve(band_lightcurve_data, band_name, args: argparse.Namespace) -> tuple[list[float], np.ndarray]:
    time, brightness_in_mag = zip(
        *[
            (time, brightness)
            for time, brightness in band_lightcurve_data[band_name]
            if ((args.timemin is None or args.timemin <= time) and (args.timemax is None or args.timemax >= time))
        ]
    )

    return time, np.array(brightness_in_mag)


def get_colour_delta_mag(band_lightcurve_data, filter_names) -> tuple[list[float], list[float]]:
    time_dict_1 = {}
    time_dict_2 = {}

    plot_times = []
    colour_delta_mag = []

    for filter_1, filter_2 in zip(band_lightcurve_data[filter_names[0]], band_lightcurve_data[filter_names[1]]):
        # Make magnitude dictionaries where time is the key
        time_dict_1[float(filter_1[0])] = filter_1[1]
        time_dict_2[float(filter_2[0])] = filter_2[1]

    for time in time_dict_1.keys():
        if time in time_dict_2.keys():  # Test if time has a magnitude for both filters
            plot_times.append(time)
            colour_delta_mag.append(time_dict_1[time] - time_dict_2[time])

    return plot_times, colour_delta_mag


def read_hesma_lightcurve(args: argparse.Namespace) -> pd.DataFrame:
    hesma_directory = os.path.join(at.get_config()["path_artistools_dir"], "data/hesma")
    filename = args.plot_hesma_model
    hesma_modelname = hesma_directory / filename

    column_names = []
    with open(hesma_modelname) as f:
        first_line = f.readline()
        if "#" in first_line:
            for i in first_line:
                if i != "#" and i != " " and i != "\n":
                    column_names.append(i)

            hesma_model = pd.read_csv(
                hesma_modelname, delim_whitespace=True, header=None, comment="#", names=column_names
            )

        else:
            hesma_model = pd.read_csv(hesma_modelname, delim_whitespace=True)
    return hesma_model


def read_reflightcurve_band_data(lightcurvefilename: Union[Path, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    filepath = Path(at.get_config()["path_artistools_dir"], "data", "lightcurves", lightcurvefilename)
    metadata = at.misc.get_file_metadata(filepath)

    data_path = os.path.join(at.get_config()["path_artistools_dir"], f"data/lightcurves/{lightcurvefilename}")
    lightcurve_data = pd.read_csv(data_path, comment="#")
    if len(lightcurve_data.keys()) == 1:
        lightcurve_data = pd.read_csv(data_path, comment="#", delim_whitespace=True)

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


def read_bol_reflightcurve_data(lightcurvefilename):
    if Path(lightcurvefilename).is_file():
        data_path = Path(lightcurvefilename)
    else:
        data_path = Path(at.get_config()["path_artistools_dir"], "data/lightcurves/bollightcurves", lightcurvefilename)

    metadata = at.misc.get_file_metadata(data_path)

    # check for possible header line and read table
    with open(data_path, "r") as flc:
        filepos = flc.tell()
        line = flc.readline()
        if line.startswith("#"):
            columns = line.lstrip("#").split()
        else:
            flc.seek(filepos)  # undo the readline() and go back
            columns = None

        dflightcurve = pd.read_csv(flc, delim_whitespace=True, header=None, names=columns)

    colrenames = {
        k: v
        for k, v in {dflightcurve.columns[0]: "time_days", dflightcurve.columns[1]: "luminosity_erg/s"}.items()
        if k != v
    }
    if colrenames:
        print(f"{data_path}: renaming columns {colrenames}")
        dflightcurve.rename(columns=colrenames, inplace=True)

    return dflightcurve, metadata


def get_sn_sample_bol():
    datafilepath = Path(at.get_config()["path_artistools_dir"], "data", "lightcurves", "SNsample", "bololc.txt")
    sn_data = pd.read_csv(datafilepath, delim_whitespace=True, comment="#")

    print(sn_data)
    bol_luminosity = sn_data["Lmax"].astype(float)
    bol_magnitude = 4.74 - (2.5 * np.log10((10**bol_luminosity) / const.L_sun.to("erg/s").value))  # 𝑀𝑏𝑜𝑙,𝑠𝑢𝑛 = 4.74

    bol_magnitude_error_upper = bol_magnitude - (
        4.74
        - (2.5 * np.log10((10 ** (bol_luminosity + sn_data["+/-.2"].astype(float))) / const.L_sun.to("erg/s").value))
    )
    # bol_magnitude_error_lower = (4.74 - (2.5 * np.log10
    #     10**(bol_luminosity - sn_data['+/-.2'].astype(float))) / const.L_sun.to('erg/s').value))) - bol_magnitude
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


def get_phillips_relation_data():
    datafilepath = Path(at.get_config()["path_artistools_dir"], "data", "lightcurves", "SNsample", "CfA3_Phillips.dat")
    sn_data = pd.read_csv(datafilepath, delim_whitespace=True, comment="#")
    print(sn_data)

    sn_data["dm15(B)"] = sn_data["dm15(B)"].astype(float)
    sn_data["MB"] = sn_data["MB"].astype(float)

    label = "Observed (Hicken et al. 2009)"
    return sn_data, label


def plot_phillips_relation_data():
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
