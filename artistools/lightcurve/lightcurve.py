#!/usr/bin/env python3

# import glob
# import itertools
import math
import os
# import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
import matplotlib.pyplot as plt

import artistools as at
import artistools.spectra

from scipy.interpolate import interp1d


def readfile(filepath_or_buffer, args=None):
    lcdata = pd.read_csv(filepath_or_buffer, delim_whitespace=True, header=None, names=['time', 'lum', 'lum_cmf'])

    if args is not None and args.plotviewingangle is not None:
        # get a list of dfs with light curves at each viewing angle
        lcdata = at.gather_res_data(lcdata, index_of_repeated_value=0)

    else:
        # the light_curve.dat file repeats x values, so keep the first half only
        lcdata = lcdata.iloc[:len(lcdata) // 2]
        lcdata.index.name = 'timestep'
    return lcdata


def get_from_packets(modelpath, lcpath, packet_type='TYPE_ESCAPE', escape_type='TYPE_RPKT', maxpacketfiles=None):
    import artistools.packets

    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles=maxpacketfiles)
    nprocs_read = len(packetsfiles)
    assert nprocs_read > 0

    timearray = at.get_timestep_times_float(modelpath=modelpath, loc='mid')
    arr_timedelta = at.get_timestep_times_float(modelpath=modelpath, loc='delta')
    # timearray = np.arange(250, 350, 0.1)
    model, _, _ = at.inputmodel.get_modeldata(modelpath)
    vmax = model.iloc[-1].velocity_outer * u.km / u.s
    betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    lcdata = pd.DataFrame({'time': timearray,
                           'lum': np.zeros_like(timearray, dtype=float),
                           'lum_cmf': np.zeros_like(timearray, dtype=float)})

    for packetsfile in packetsfiles:
        dfpackets = at.packets.readfile(packetsfile, type=packet_type, escape_type=escape_type)

        if not (dfpackets.empty):
            print(f"sum of e_cmf {dfpackets['e_cmf'].sum()} e_rf {dfpackets['e_rf'].sum()}")

            binned = pd.cut(dfpackets['t_arrive_d'], timearrayplusend, labels=False, include_lowest=True)
            for binindex, e_rf_sum in dfpackets.groupby(binned)['e_rf'].sum().iteritems():
                lcdata['lum'][binindex] += e_rf_sum

            dfpackets['t_arrive_cmf_d'] = dfpackets['escape_time'] * betafactor * u.s.to('day')

            binned_cmf = pd.cut(dfpackets['t_arrive_cmf_d'], timearrayplusend, labels=False, include_lowest=True)
            for binindex, e_cmf_sum in dfpackets.groupby(binned_cmf)['e_cmf'].sum().iteritems():
                lcdata['lum_cmf'][binindex] += e_cmf_sum

    lcdata['lum'] = np.divide(lcdata['lum'] / nprocs_read * (u.erg / u.day).to('solLum'), arr_timedelta)
    lcdata['lum_cmf'] = np.divide(lcdata['lum_cmf'] / nprocs_read / betafactor * (u.erg / u.day).to('solLum'),
                                  arr_timedelta)
    return lcdata


def get_magnitudes(modelpath, args, angle=None, modelnumber=None):
    """Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py"""
    if args and args.plotvspecpol and os.path.isfile(modelpath / 'vpkt.txt'):
        print("Found vpkt.txt, using vitual packets")
        stokes_params = at.spectra.get_polarisation(angle, modelpath)
        vspecdata = stokes_params['I']
        timearray = vspecdata.keys()[1:]
    elif args and args.plotviewingangle and os.path.isfile(modelpath / 'specpol_res.out'):
        specfilename = os.path.join(modelpath, "specpol_res.out")
        specdataresdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdataresdata.columns.values[1:] if i[-2] != '.']
    elif Path(modelpath, 'specpol.out').is_file():
        specfilename = os.path.join(modelpath, "specpol.out")
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        specfilename = at.firstexisting(['spec.out.xz', 'spec.out.gz', 'spec.out'], path=modelpath)
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = specdata.columns.values[1:]

    filters_dict = {}
    if not args.filter:
        args.filter = ['B']

    filters_list = args.filter

    # filters_list = ['B']
    if angle is not None and os.path.isfile(modelpath / 'specpol_res.out'):
        res_specdata = at.spectra.read_specpol_res(modelpath, angle=angle, args=args)
    else:
        res_specdata = None

    for filter_name in filters_list:
        if filter_name == 'bol':
            times, bol_magnitudes = bolometric_magnitude(modelpath, timearray, args, angle=angle,
                                                         res_specdata=res_specdata)
            filters_dict['bol'] = [
                (time, bol_magnitude) for time, bol_magnitude in
                zip(times, bol_magnitudes)
                if math.isfinite(bol_magnitude)]
        elif filter_name not in filters_dict:
            filters_dict[filter_name] = []

    filterdir = os.path.join(at.PYDIR, 'data/filters/')

    for filter_name in filters_list:
        if filter_name == 'bol':
            continue
        zeropointenergyflux, wavefilter, transmission, wavefilter_min, wavefilter_max \
            = get_filter_data(filterdir, filter_name)

        for timestep, time in enumerate(timearray):
            time = float(time)
            if args.timemin < time < args.timemax:
                wavelength_from_spectrum, flux = \
                    get_spectrum_in_filter_range(modelpath, timestep, time, wavefilter_min, wavefilter_max, angle,
                                                 res_specdata=res_specdata, modelnumber=modelnumber, args=args)

                if len(wavelength_from_spectrum) > len(wavefilter):
                    interpolate_fn = interp1d(wavefilter, transmission, bounds_error=False, fill_value=0.)
                    wavefilter = np.linspace(min(wavelength_from_spectrum), int(max(wavelength_from_spectrum)),
                                             len(wavelength_from_spectrum))
                    transmission = interpolate_fn(wavefilter)
                else:
                    interpolate_fn = interp1d(wavelength_from_spectrum, flux, bounds_error=False, fill_value=0.)
                    wavelength_from_spectrum = np.linspace(wavefilter_min, wavefilter_max, len(wavefilter))
                    flux = interpolate_fn(wavelength_from_spectrum)

                phot_filtobs_sn = evaluate_magnitudes(flux, transmission, wavelength_from_spectrum, zeropointenergyflux)

                # print(time, phot_filtobs_sn)
                # if phot_filtobs_sn != 0.0:
                phot_filtobs_sn = phot_filtobs_sn - 25  # Absolute magnitude
                filters_dict[filter_name].append((time, phot_filtobs_sn))

    return filters_dict


def bolometric_magnitude(modelpath, timearray, args, angle=None, res_specdata=None):
    magnitudes = []
    times = []
    for timestep, time in enumerate(timearray):
        time = float(time)
        if args.timemin < time < args.timemax:
            if angle is not None:
                if args.plotvspecpol:
                    spectrum = at.spectra.get_vspecpol_spectrum(modelpath, time, angle, args)
                else:
                    if res_specdata is None:
                        res_specdata = at.spectra.read_specpol_res(modelpath, angle=angle, args=args)
                    spectrum = at.spectra.get_res_spectrum(modelpath, timestep, timestep, angle=angle,
                                                           res_specdata=res_specdata)
            else:
                spectrum = at.spectra.get_spectrum(modelpath, timestep, timestep)

            integrated_flux = np.trapz(spectrum['f_lambda'], spectrum['lambda_angstroms'])
            integrated_luminosity = integrated_flux * 4 * np.pi * np.power(u.Mpc.to('cm'), 2)
            Mbol_sun = 4.74
            magnitude = Mbol_sun - (2.5 * np.log10(integrated_luminosity / const.L_sun.to('erg/s').value))
            magnitudes.append(magnitude)
            times.append(time)
            # print(const.L_sun.to('erg/s').value)
            # quit()

    return times, magnitudes


def get_filter_data(filterdir, filter_name):
    """Filter data in 'data/filters' taken from https://github.com/cinserra/S3/tree/master/src/s3/metadata"""

    with open(filterdir / Path(filter_name + '.txt'), 'r') as filter_metadata:  # defintion of the file
        line_in_filter_metadata = filter_metadata.readlines()  # list of lines

    zeropointenergyflux = float(line_in_filter_metadata[0])
    # zero point in energy flux (erg/cm^2/s)

    wavefilter, transmission = [], []
    for row in line_in_filter_metadata[4:]:
        # lines where the wave and transmission are stored
        wavefilter.append(float(row.split()[0]))
        transmission.append(float(row.split()[1]))

    wavefilter_min = min(wavefilter)
    wavefilter_max = int(max(wavefilter))  # integer is needed for a sharper cut-off

    return zeropointenergyflux, np.array(wavefilter), np.array(transmission), wavefilter_min, wavefilter_max


def get_spectrum_in_filter_range(modelpath, timestep, time, wavefilter_min, wavefilter_max, angle=None,
                                 res_specdata=None, modelnumber=None, spectrum=None, args=None):
    if spectrum is None:
        spectrum = at.spectra.get_spectrum_at_time(
            modelpath, timestep=timestep, time=time, args=args,
            angle=angle, res_specdata=res_specdata, modelnumber=modelnumber)

    wavelength_from_spectrum, flux = [], []
    for wavelength, flambda in zip(spectrum['lambda_angstroms'], spectrum['f_lambda']):
        if wavefilter_min <= wavelength <= wavefilter_max:  # to match the spectrum wavelengths to those of the filter
            wavelength_from_spectrum.append(wavelength)
            flux.append(flambda)

    return np.array(wavelength_from_spectrum), np.array(flux)


def evaluate_magnitudes(flux, transmission, wavelength_from_spectrum, zeropointenergyflux):
    cf = flux * transmission
    flux_obs = abs(np.trapz(cf, wavelength_from_spectrum))  # using trapezoidal rule to integrate
    if flux_obs == 0.0:
        phot_filtobs_sn = 0.0
    else:
        phot_filtobs_sn = -2.5 * np.log10(flux_obs / zeropointenergyflux)

    return phot_filtobs_sn


def calculate_costheta_phi_for_viewing_angles(viewing_angles, modelpath):
    modelpath = Path(modelpath)
    if os.path.isfile(modelpath / 'absorptionpol_res_99.out') \
            and os.path.isfile(modelpath / 'absorptionpol_res_100.out'):
        print("Too many viewing angle bins (MABINS) for this method to work, it only works for MABINS = 100")
        exit()
    elif os.path.isfile(modelpath / 'light_curve_res.out'):
        angle_definition = {}

        costheta_viewing_angle_bins = ['-1.0 \u2264 cos(\u03B8) < -0.8', '-0.8 \u2264 cos(\u03B8) < -0.6',
                                       '-0.6 \u2264 cos(\u03B8) < -0.4', '-0.4 \u2264 cos(\u03B8) < -0.2',
                                       '-0.2 \u2264 cos(\u03B8) < 0', '0 \u2264 cos(\u03B8) < 0.2',
                                       '0.2 \u2264 cos(\u03B8) < 0.4', '0.4 \u2264 cos(\u03B8) < 0.6',
                                       '0.6 \u2264 cos(\u03B8) < 0.8', '0.8 \u2264 cos(\u03B8) < 1']
        phi_viewing_angle_bins = ['0 \u2264 \u03D5 < \u03c0/5', '\u03c0/5 \u2264 \u03D5 < 2\u03c0/5',
                                  '2\u03c0/5 \u2264 \u03D5 < 3\u03c0/5', '3\u03c0/5 \u2264 \u03D5 < 4\u03c0/5',
                                  '4\u03c0/5 \u2264 \u03D5 < \u03c0', '9\u03c0/5 < \u03D5 < 2\u03c0',
                                  '8\u03c0/5 < \u03D5 \u2264 9\u03c0/5', '7\u03c0/5 < \u03D5 \u2264 8\u03c0/5',
                                  '6\u03c0/5 < \u03D5 \u2264 7\u03c0/5', '\u03c0 < \u03D5 \u2264 6\u03c0/5']
        for angle in viewing_angles:
            MABINS = 100
            phibins = int(math.sqrt(MABINS))
            costheta_index = angle // phibins
            phi_index = angle % phibins

            angle_definition[angle] = f'{costheta_viewing_angle_bins[costheta_index]}, {phi_viewing_angle_bins[phi_index]}'
            print(f"{angle:4d}   {costheta_viewing_angle_bins[costheta_index]}   {phi_viewing_angle_bins[phi_index]}")

        return angle_definition
    else:
        print("Too few viewing angle bins (MABINS) for this method to work, it only works for MABINS = 100")
        exit()


def read_hesma_lightcurve(args):
    hesma_directory = os.path.join(at.PYDIR, 'data/hesma')
    filename = args.plot_hesma_model
    hesma_modelname = hesma_directory / filename

    column_names = []
    with open(hesma_modelname) as f:
        first_line = f.readline()
        if '#' in first_line:
            for i in first_line:
                if i != '#' and i != ' ' and i != '\n':
                    column_names.append(i)

            hesma_model = pd.read_csv(hesma_modelname, delim_whitespace=True, header=None,
                                      comment='#', names=column_names)

        else:
            hesma_model = pd.read_csv(hesma_modelname, delim_whitespace=True)
    return hesma_model


def read_reflightcurve_band_data(lightcurvefilename):
    filepath = Path(at.PYDIR, 'data', 'lightcurves', lightcurvefilename)
    metadata = at.get_file_metadata(filepath)

    data_path = os.path.join(at.PYDIR, f"data/lightcurves/{lightcurvefilename}")
    lightcurve_data = pd.read_csv(data_path, comment='#')
    lightcurve_data['time'] = lightcurve_data['time'].apply(lambda x: x - (metadata['timecorrection']))
    # m - M = 5log(d) - 5  Get absolute magnitude
    if 'dist_mpc' in metadata:
        lightcurve_data['magnitude'] = lightcurve_data['magnitude'].apply(lambda x: (
            x - 5 * np.log10(metadata['dist_mpc'] * 10 ** 6) + 5))
    elif 'dist_modulus' in metadata:
        lightcurve_data['magnitude'] = lightcurve_data['magnitude'].apply(lambda x: (
            x - metadata['dist_modulus']))

    return lightcurve_data, metadata


def read_bol_reflightcurve_data(lightcurvefilename):
    data_path = os.path.join(at.PYDIR, f"data/lightcurves/bollightcurves/{lightcurvefilename}")
    metadata = at.get_file_metadata(data_path)

    bollightcurve_data = pd.read_csv(data_path, comment='#', delim_whitespace=True, header=None)

    bollightcurve_data = bollightcurve_data.rename(columns={0: 'time_days'})  # days
    bollightcurve_data = bollightcurve_data.rename(columns={1: 'luminosity_erg/s'})  # erg/s

    return bollightcurve_data, metadata


def get_sn_sample_bol():
    datafilepath = Path(at.PYDIR, 'data', 'lightcurves', 'SNsample', 'bololc.txt')
    sn_data = pd.read_csv(datafilepath, delim_whitespace=True, comment='#')

    print(sn_data)
    bol_luminosity = sn_data['Lmax'].astype(float)
    bol_magnitude = 4.74 - (2.5 * np.log10((10**bol_luminosity) / const.L_sun.to('erg/s').value))  # 𝑀𝑏𝑜𝑙,𝑠𝑢𝑛 = 4.74

    bol_magnitude_error_upper = bol_magnitude - (4.74 - (2.5 * np.log10((10**(bol_luminosity + sn_data['+/-.2'].astype(float))) / const.L_sun.to('erg/s').value)))
    # bol_magnitude_error_lower = (4.74 - (2.5 * np.log10((10**(bol_luminosity - sn_data['+/-.2'].astype(float))) / const.L_sun.to('erg/s').value))) - bol_magnitude
    # print(bol_magnitude_error_upper, "============")
    # print(bol_magnitude_error_lower, "============")
    # print(bol_magnitude_error_upper == bol_magnitude_error_lower)

    # a0 = plt.errorbar(x=sn_data['dm15'].astype(float), y=sn_data['Lmax'].astype(float),
    #                   yerr=sn_data['+/-.2'].astype(float), xerr=sn_data['+/-'].astype(float),
    #                   color='grey', marker='o', ls='None')
    #
    sn_data['bol_mag'] = bol_magnitude
    print(sn_data[['name', 'bol_mag', 'dm15', 'dm40']])
    sn_data[['name', 'bol_mag', 'dm15', 'dm40']].to_csv('boldata.txt', sep=' ', index=False)
    a0 = plt.errorbar(x=sn_data['dm15'].astype(float), y=bol_magnitude,
                      yerr=bol_magnitude_error_upper, xerr=sn_data['+/-'].astype(float),
                      color='k', marker='o', ls='None')


    # a0 = plt.errorbar(x=sn_data['dm15'].astype(float), y=sn_data['dm40'].astype(float),
    #                   yerr=sn_data['+/-.1'].astype(float), xerr=sn_data['+/-'].astype(float),
    #                   color='k', marker='o', ls='None')

    # a0 = plt.scatter(sn_data['dm15'].astype(float), bol_magnitude, s=80, color='k', marker='o')
    # plt.gca().invert_yaxis()
    # plt.show()

    label = 'Bolometric data (Scalzo et al. 2019)'
    return a0, label


def get_phillips_relation_data():
    datafilepath = Path(at.PYDIR, 'data', 'lightcurves', 'SNsample', 'CfA3_Phillips.dat')
    sn_data = pd.read_csv(datafilepath, delim_whitespace=True, comment='#')
    print(sn_data)

    deltam_15B = sn_data['dm15(B)'].astype(float)
    M_B = sn_data['MB'].astype(float)


    label = 'Observed (Hicken et al. 2009)'
    # a0 = plt.scatter(deltam_15B, M_B, s=80, color='grey', marker='o', label=label)
    a0 = plt.errorbar(x=deltam_15B, y=M_B, yerr=sn_data['err_MB'], xerr=sn_data['err_dm15(B)'],
                      color='k', alpha=0.9, marker='.', capsize=2, label=label, ls='None', zorder=5)
    # plt.gca().invert_yaxis()
    # plt.show()
    return a0, label
