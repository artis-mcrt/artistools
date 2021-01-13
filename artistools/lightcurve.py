#!/usr/bin/env python3

import argparse
# import glob
# import itertools
import math
import multiprocessing
import os
# import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at
import artistools.spectra
import matplotlib.pyplot as plt
import matplotlib
from extinction import apply, ccm89

from scipy.interpolate import interp1d
from textwrap import wrap

from matplotlib.legend_handler import HandlerTuple

color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))


def readfile(filepath_or_buffer):
    lcdata = pd.read_csv(filepath_or_buffer, delim_whitespace=True, header=None, names=['time', 'lum', 'lum_cmf'])
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
    model, _ = at.get_modeldata(modelpath)
    vmax = model.iloc[-1].velocity_outer * u.km / u.s
    betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    lcdata = pd.DataFrame({'time': timearray,
                           'lum': np.zeros_like(timearray, dtype=np.float),
                           'lum_cmf': np.zeros_like(timearray, dtype=np.float)})

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


def make_lightcurve_plot(modelpaths, filenameout, frompackets=False, escape_type=False, maxpacketfiles=None, args=None):
    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if not frompackets and escape_type not in ['TYPE_RPKT', 'TYPE_GAMMA']:
        print(f'Escape_type of {escape_type} not one of TYPE_RPKT or TYPE_GAMMA, so frompackets must be enabled')
        assert False
    elif not frompackets and args.packet_type != 'TYPE_ESCAPE' and args.packet_type is not None:
        print(f'Looking for non-escaped packets, so frompackets must be enabled')
        assert False

    for seriesindex, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")
        lcname = 'gamma_light_curve.out' if (escape_type == 'TYPE_GAMMA' and not frompackets) else 'light_curve.out'
        try:
            lcpath = at.firstexisting([lcname + '.xz', lcname + '.gz', lcname], path=modelpath)
        except FileNotFoundError:
            print(f"Skipping {modelname} because {lcname} does not exist")
            continue
        if not os.path.exists(str(lcpath)):
            print(f"Skipping {modelname} because {lcpath} does not exist")
            continue
        elif frompackets:
            lcdata = at.lightcurve.get_from_packets(
                modelpath, lcpath, packet_type=args.packet_type, escape_type=escape_type, maxpacketfiles=maxpacketfiles)
        else:
            lcdata = at.lightcurve.readfile(lcpath)

        plotkwargs = {}
        if args.label[seriesindex] is None:
            plotkwargs['label'] = modelname
        else:
            plotkwargs['label'] = args.label[seriesindex]

        plotkwargs['linestyle'] = args.linestyle[seriesindex]
        plotkwargs['color'] = args.color[seriesindex]
        if args.dashes[seriesindex]:
            plotkwargs['dashes'] = args.dashes[seriesindex]
        if args.linewidth[seriesindex]:
            plotkwargs['linewidth'] = args.linewidth[seriesindex]
        axis.plot(lcdata['time'], lcdata['lum'], **plotkwargs)
        if args.print_data:
            print(lcdata[['time', 'lum', 'lum_cmf']].to_string(index=False))
        if args.plotcmf:
            plotkwargs['linewidth'] = 1
            plotkwargs['label'] += ' (cmf)'
            axis.plot(lcdata.time, lcdata['lum_cmf'], **plotkwargs)

    if args.xmin:
        axis.set_xlim(left=args.xmin)
    if args.xmax:
        axis.set_xlim(right=args.xmax)
    # axis.set_ylim(bottom=-0.1, top=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    if escape_type == 'TYPE_GAMMA':
        lum_suffix = r'_\gamma'
    elif escape_type == 'TYPE_RPKT':
        lum_suffix = r'_{\mathrm{OVOIR}}'
    else:
        lum_suffix = r'_{\mathrm{' + escape_type.replace("_", r"\_") + '}}'
    axis.set_ylabel(r'$\mathrm{L} ' + lum_suffix + r'/ \mathrm{L}_\odot$')

    fig.savefig(str(filenameout), format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


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
            filters_dict['bol'] = [
                (float(time), bol_magnitude) for time, bol_magnitude in
                zip(timearray, bolometric_magnitude(modelpath, timearray, args, angle=angle, res_specdata=res_specdata))
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
            wavelength_from_spectrum, flux = \
                get_spectrum_in_filter_range(modelpath, timestep, time, wavefilter_min, wavefilter_max, args, angle,
                                             res_specdata=res_specdata, modelnumber=modelnumber)

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
    for timestep, time in enumerate(timearray):
        if angle is not None:
            if args.plotvspecpol:
                spectrum = at.spectra.get_vspecpol_spectrum(modelpath, time, angle, args)
            else:
                if res_specdata is None:
                    res_specdata = at.spectra.read_specpol_res(modelpath, angle=angle)
                spectrum = at.spectra.get_res_spectrum(modelpath, timestep, timestep, angle=angle,
                                                       res_specdata=res_specdata)
        else:
            spectrum = at.spectra.get_spectrum(modelpath, timestep, timestep)

        integrated_flux = np.trapz(spectrum['f_lambda'], spectrum['lambda_angstroms'])
        integrated_luminosity = integrated_flux * 4 * np.pi * np.power(u.Mpc.to('cm'), 2)
        magnitude = 4.74 - (2.5 * np.log10(integrated_luminosity / const.L_sun.to('erg/s').value))
        magnitudes.append(magnitude)

    return magnitudes


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


def get_spectrum_in_filter_range(modelpath, timestep, time, wavefilter_min, wavefilter_max, args, angle=None,
                                 res_specdata=None, modelnumber=None, spectrum=None):
    if spectrum is None:
        spectrum = at.spectra.get_spectrum_at_time(modelpath, timestep, time, args, angle, res_specdata, modelnumber)

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


def make_magnitudes_plot(modelpaths, filternames_conversion_dict, outputfolder, args):
    labelfontsize = 22
    font = {'size': labelfontsize}
    matplotlib.rc('font', **font)
    if args.filter and len(args.filter) > 1:
        rows = 2
        cols = 3
    else:
        rows = 1
        cols = 1
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True,
                           figsize=(at.figwidth * 1.6 * cols, at.figwidth * 1.1 * rows*1.75),
                           tight_layout={"pad": 3.0, "w_pad": 0.6, "h_pad": 0.6})  # (6.2 * 3, 9.4 * 3)
    if args.filter and len(args.filter) > 1:
        ax = ax.flatten()

    plt.gca().invert_yaxis()
    if args.ymax is None:
        args.ymax = -20
    if args.ymin is None:
        args.ymin = -14
    if args.xmax is None:
        args.xmax = 100
    if args.xmin is None:
        args.xmin = 0
    if args.timemax is None:
        args.timemax = args.xmax
    if args.timemin is None:
        args.timemin = args.xmin

    band_risetime_polyfit = []
    band_peakmag_polyfit = []
    band_deltam15_polyfit = []
    plotvalues = []
    modelnames = []
    modelnumbers = []
    band_risetime_angle_averaged_polyfit = []
    band_peakmag_angle_averaged_polyfit = []
    band_delta_m15_angle_averaged_polyfit = []

    colours = ['k', 'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold',
               'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green', 'purple',
               'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k',
               'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan',
               'darkblue', 'bisque', 'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange',
               'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k', 'tab:blue', 'tab:red',
               'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue', 'bisque',
               'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray',
               'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green', 'purple',
               'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k',
               'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan',
               'darkblue', 'bisque', 'yellow']

    colours2 = ['gray', 'lightblue', 'pink', 'yellowgreen', 'mediumorchid', 'sandybrown', 'plum', 'lightgray',
                'wheat', 'paleturquoise']

    # angle_names = [0, 45, 90, 180]
    # plt.style.use('dark_background')

    for modelnumber, modelpath in enumerate(modelpaths):
        modelpath = Path(modelpath)
        if args.plotvspecpol and os.path.isfile(modelpath / 'vpkt.txt'):
            angles = args.plotvspecpol
        elif args.plotviewingangle and args.plotviewingangle[0] == -1 and os.path.isfile(modelpath / 'specpol_res.out'):
            angles = np.arange(0, 100, 1, dtype=int)
        elif args.plotviewingangle and os.path.isfile(modelpath / 'specpol_res.out'):
            angles = args.plotviewingangle
        elif args.calculate_costheta_phi_from_viewing_angle_numbers and \
                args.calculate_costheta_phi_from_viewing_angle_numbers[0] == -1:
            viewing_angles = np.arange(0, 100, 1, dtype=int)
            calculate_costheta_phi_for_viewing_angles(viewing_angles, modelpath)
        elif args.calculate_costheta_phi_from_viewing_angle_numbers:
            viewing_angles = args.calculate_costheta_phi_from_viewing_angle_numbers
            calculate_costheta_phi_for_viewing_angles(viewing_angles, modelpath)
        else:
            angles = [None]
        # angles.append(None)

        for index, angle in enumerate(angles):
            linenames = []

            modelname = at.get_model_name(modelpath)
            linenames.append(modelname)
            print(f'Reading spectra: {modelname}')
            filters_dict = get_magnitudes(modelpath, args, angle, modelnumber=modelnumber)

            if modelnumber == 0 and args.plot_hesma_model:
                hesma_model = read_hesma_lightcurve(args)
                linename = str(args.plot_hesma_model).split('_')[:3]
                linename = "_".join(linename)

                if linename not in linenames:
                    linenames.append(linename)

            for plotnumber, key in enumerate(filters_dict):
                time = []
                magnitude = []

                for t, mag in filters_dict[key]:
                    # adjusting plot range for artis simulation (x-axis)
                    if (args.timemin < t < args.timemax):
                        time.append(t)
                        magnitude.append(mag)

                if args.plotvspecpol and angle is not None and os.path.isfile(modelpath / 'vpkt.txt'):
                    vpkt_config = at.get_vpkt_config(modelpath)
                    viewing_angle = round(math.degrees(math.acos(vpkt_config['cos_theta'][angle])))
                    linelabel = fr"$\theta$ = {viewing_angle}"
                elif args.plotviewingangle and angle is not None and os.path.isfile(modelpath / 'specpol_res.out'):
                    linelabel = fr"bin number = {angle}"
                    # linelabel = None
                    # linelabel = fr"{modelname} $\theta$ = {angle_names[index]}$^\circ$"
                    # plt.plot(time, magnitude, label=linelabel, linewidth=3)
                elif args.label:
                    linelabel = fr'{args.label[modelnumber]}'
                else:
                    linelabel = f'{modelname}'
                    # linelabel = 'Angle averaged'

                if linelabel == 'None' or linelabel is None:
                    linelabel = f'{modelname}'

                # linelabel = '\n'.join(wrap(linelabel, 40))

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    magnitude = filterfunc(magnitude)

                # Calculating band peak time, peak magnitude and delta m15
                if (args.calculate_peakmag_risetime_delta_m15
                        or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file
                        or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
                        or args.make_viewing_angle_peakmag_risetime_scatter_plot
                        or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
                    band_risetime_polyfit, band_peakmag_polyfit, band_deltam15_polyfit = \
                        calculate_peak_time_mag_deltam15(time, magnitude, modelname, angle, key, band_risetime_polyfit,
                                                         band_peakmag_polyfit, band_deltam15_polyfit,
                                                         filternames_conversion_dict, args)

                if args.plotviewingangle and args.plotviewingangles_lightcurves:
                    plt.plot(time, magnitude, label=modelname, color=colours[angle], linewidth=3)

                # if 'redshifttoz' in args and args.redshifttoz[modelnumber] != 0:
                #     # print('time before', time)
                #     # print('z', args.redshifttoz[modelnumber])
                #     time = np.array(time) * (1 + args.redshifttoz[modelnumber])
                #     print(f'Correcting for time dilation at redshift {args.redshifttoz[modelnumber]}')
                #     # print('time after', time)
                #     linestyle = '--'
                #     color = 'darkmagenta'
                #     linelabel=args.label[1]
                # else:
                #     linestyle = '-'
                #     color='k'
                # plt.plot(time, magnitude, label=linelabel, linewidth=3)

                if modelnumber == 0 and args.plot_hesma_model and key in hesma_model.keys():
                    ax.plot(hesma_model.t, hesma_model[key], color='black')

                # axarr[plotnumber].axis([0, 60, -16, -19.5])
                if key in filternames_conversion_dict:
                    text_key = filternames_conversion_dict[key]
                else:
                    text_key = key
                if args.filter and len(args.filter) > 1:
                    ax[plotnumber].text(args.xmax * 0.1, args.ymax * 0.97, text_key)
                # else:
                #     ax.text(args.xmax * 0.75, args.ymax * 0.95, text_key)

                if (args.magnitude or args.plotviewingangles_lightcurves) and not (
                        args.calculate_peakmag_risetime_delta_m15
                        or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
                        or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file
                        or args.test_viewing_angle_fit
                        or args.make_viewing_angle_peakmag_risetime_scatter_plot
                        or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):

                    if args.reflightcurves and modelnumber == 0:
                        if len(angles) > 1 and index > 0:
                            print('already plotted reflightcurve')
                        else:
                            colours = args.refspeccolors
                            markers = args.refspecmarkers
                            for i, reflightcurve in enumerate(args.reflightcurves):
                                plot_lightcurve_from_data(filters_dict.keys(), reflightcurve, colours[i], markers[i],
                                                          filternames_conversion_dict, ax, plotnumber)

                if args.color:
                    color = args.color[modelnumber]
                else:
                    color = colours[modelnumber]
                if args.linestyle:
                    linestyle = args.linestyle[modelnumber]

                if args.magnitude and not (
                        args.calculate_peakmag_risetime_delta_m15 or
                        args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
                        or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file or args.test_viewing_angle_fit
                        or args.make_viewing_angle_peakmag_risetime_scatter_plot or
                        args.make_viewing_angle_peakmag_delta_m15_scatter_plot):

                    if args.filter and len(args.filter) > 1:
                        if linestyle == 'dashed':
                            alpha = 0.6
                        else:
                            alpha = 1

                        if len(angles) > 1 or (args.plotviewingangle and os.path.isfile(modelpath / 'specpol_res.out')):
                            ax[plotnumber].plot(time, magnitude, label=linelabel, linewidth=4, linestyle=linestyle,
                                                alpha=alpha)
                        else:
                            ax[plotnumber].plot(time, magnitude, label=linelabel, linewidth=4, color=color,
                                                linestyle=linestyle, alpha=alpha)
                            # if key is not 'bol':
                            #     ax[plotnumber].plot(
                            #         cmfgen_mags['time[d]'], cmfgen_mags[key], label='CMFGEN', color='k', linewidth=3)
                    else:
                        # if 'FM3' in str(modelpath):
                        #     ax.plot(time, magnitude, label=linelabel, linewidth=3, color='darkblue')
                        # elif 'M2a' in str(modelpath):
                        #     ax.plot(time, magnitude, label=linelabel, linewidth=3, color='k')
                        # else:
                        ax.plot(time, magnitude, label=linelabel, linewidth=3)  # color=color, linestyle=linestyle)

        # Saving viewing angle data so it can be read in and plotted later on without re-running the script
        #    as it is quite time consuming
        if args.save_viewing_angle_peakmag_risetime_delta_m15_to_file:
            np.savetxt(key + "band_" + f'{modelname}' + "_viewing_angle_data.txt",
                       np.c_[band_peakmag_polyfit, band_risetime_polyfit, band_deltam15_polyfit],
                       delimiter=' ', header='peak_mag_polyfit risetime_polyfit deltam15_polyfit', comments='')

        elif (args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
              or args.make_viewing_angle_peakmag_risetime_scatter_plot
              or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):

            band_risetime_angle_averaged_polyfit.append(band_risetime_polyfit)
            band_peakmag_angle_averaged_polyfit.append(band_peakmag_polyfit)
            band_delta_m15_angle_averaged_polyfit.append(band_deltam15_polyfit)

        band_risetime_polyfit = []
        band_peakmag_polyfit = []
        band_deltam15_polyfit = []

        # Saving the information required to make the viewing angle scatter plots to arrays
        if (args.make_viewing_angle_peakmag_risetime_scatter_plot
                or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
            modelnames.append(modelname)
            modelnumbers.append(modelnumber)

        # if args.magnitude and not (
        #         args.calculate_peakmag_risetime_delta_m15 or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
        #         or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file or args.test_viewing_angle_fit
        #         or args.make_viewing_angle_peakmag_risetime_scatter_plot or
        #         args.make_viewing_angle_peakmag_delta_m15_scatter_plot or args.plotviewingangle):
        #     plt.plot(time, magnitude, label=modelname, color=colours[modelnumber], linewidth=3)

    # Saving all this viewing angle info for each model to a file so that it is available to plot if required again
    # as it takes relatively long to run this for all viewing angles

    if (args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
            or args.make_viewing_angle_peakmag_risetime_scatter_plot
            or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
        np.savetxt(key + "band_" + f'{modelname}' + "_angle_averaged_all_models_data.txt",
                   np.c_[modelnames, band_risetime_angle_averaged_polyfit, band_peakmag_angle_averaged_polyfit,
                         band_delta_m15_angle_averaged_polyfit],
                   delimiter=' ', fmt='%s',
                   header="object " + str(key) + "_band_risetime " + str(key) + "_band_peakmag " + str(
                       key) + "_band_deltam15 ", comments='')

    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, band_risetime_angle_averaged_polyfit,
                                                         band_peakmag_angle_averaged_polyfit, colours, colours2,
                                                         plotvalues, key)
        return

    elif args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        make_viewing_angle_peakmag_delta_m15_scatter_plot(modelnames, key, colours, colours2,
                                                          band_delta_m15_angle_averaged_polyfit,
                                                          band_peakmag_angle_averaged_polyfit, plotvalues)
        return

    # if (args.magnitude or args.plotviewingangles_lightcurves) and not (
    #         args.calculate_peakmag_risetime_delta_m15 or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
    #         or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file or args.test_viewing_angle_fit
    #         or args.make_viewing_angle_peakmag_risetime_scatter_plot or
    #         args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
    #     if args.reflightcurves:
    #         colours = args.refspeccolors
    #         markers = args.refspecmarkers
    #         for i, reflightcurve in enumerate(args.reflightcurves):
    #             plot_lightcurve_from_data(filters_dict.keys(), reflightcurve, colours[i], markers[i],
    #                                       filternames_conversion_dict)

    if args.filter and len(args.filter) > 1:
        for axis in ax:
            # axis.set_xscale('log')
            axis.minorticks_on()
            axis.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2,
                             labelsize=labelfontsize, direction='in')
            axis.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2,
                             labelsize=labelfontsize, direction='in')

    else:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2, labelsize=labelfontsize,
                       direction='in')
        ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2, labelsize=labelfontsize,
                       direction='in')

    if args.filter and len(args.filter) > 1:
        fig.text(0.5, 0.025, 'Time Since Explosion [days]', ha='center', va='center')
        fig.text(0.02, 0.5, 'Absolute Magnitude', ha='center', va='center', rotation='vertical')
    elif key in filternames_conversion_dict:
        ax.set_ylabel(f'{filternames_conversion_dict[key]} Magnitude', fontsize=labelfontsize)
    else:
        ax.set_ylabel(f'{key} Magnitude', fontsize=labelfontsize)  # r'M$_{\mathrm{bol}}$'
        ax.set_xlabel('Time Since Explosion [days]', fontsize=labelfontsize)

    plt.ylim(args.ymin, args.ymax)
    plt.xlim(args.xmin, args.xmax)

    plt.minorticks_on()
    # f.suptitle(f'{modelname}')
    # f.legend(labels=linenames, frameon=True, fontsize='xx-small')
    # plt.tight_layout()
    # f.set_figheight(8)
    # f.set_figwidth(7)
    if not args.nolegend:
        if args.filter and len(args.filter) > 1:
            ax[0].legend(loc='best', frameon=True, fontsize='xx-small', ncol=1)
        else:
            ax.legend(loc='best', frameon=False, fontsize='small', ncol=1, handlelength=0.7)
    if args.filter and len(filters_dict) == 1:
        args.outputfile = os.path.join(outputfolder, f'plot{key}lightcurves.pdf')
    if args.show:
        plt.show()
    plt.savefig(args.outputfile, format='pdf')
    print(f'Saved figure: {args.outputfile}')


def calculate_peak_time_mag_deltam15(time, magnitude, modelname, angle, key, band_risetime_polyfit,
                                     band_peakmag_polyfit, band_deltam15_polyfit, filternames_conversion_dict, args):
    """Calculating band peak time, peak magnitude and delta m15"""
    zfit = np.polyfit(x=time, y=magnitude, deg=10)
    xfit = np.linspace(args.timemin + 1, args.timemax - 1, num=1000)

    # Taking line_min and line_max from the limits set for the lightcurve being plotted
    fxfit = []
    for j in range(len(xfit)):
        fxfit.append(zfit[0] * (xfit[j] ** 10) + zfit[1] * (xfit[j] ** 9) + zfit[2] * (xfit[j] ** 8) +
                     zfit[3] * (xfit[j] ** 7) + zfit[4] * (xfit[j] ** 6) + zfit[5] * (xfit[j] ** 5) +
                     zfit[6] * (xfit[j] ** 4) + zfit[7] * (xfit[j] ** 3) + zfit[8] * (xfit[j] ** 2) +
                     zfit[9] * (xfit[j]) + zfit[10])
        # polynomial with 10 degrees of freedom used here but change as required if it improves the fit

    def match_closest_time_polyfit(reftime_polyfit):
        return str("{}".format(min([float(x) for x in xfit], key=lambda x: abs(x - reftime_polyfit))))

    index_min = np.argmin(fxfit)
    tmax_polyfit = xfit[index_min]
    time_after15days_polyfit = match_closest_time_polyfit(tmax_polyfit + 15)
    for ii, xfits in enumerate(xfit):
        if float(xfits) == float(time_after15days_polyfit):
            index_after_15_days = ii

    mag_after15days_polyfit = fxfit[index_after_15_days]
    print(f'{key}_max polyfit = {min(fxfit)} at time = {tmax_polyfit}')
    print(f'deltam15 polyfit = {min(fxfit) - mag_after15days_polyfit}')

    band_risetime_polyfit.append(tmax_polyfit)
    band_peakmag_polyfit.append(min(fxfit))
    band_deltam15_polyfit.append((min(fxfit) - mag_after15days_polyfit) * -1)

    # Plotting the lightcurves for all viewing angles specified in the command line along with the
    # polynomial fit and peak mag, risetime to peak and delta m15 marked on the plots to check the
    # fit is working correctly
    if args.test_viewing_angle_fit:
        plt.plot(time, magnitude)
        plt.plot(xfit, fxfit)

        if key in filternames_conversion_dict:
            plt.ylabel(f'{filternames_conversion_dict[key]} Magnitude')
        else:
            plt.ylabel(f'{key} Magnitude')

        plt.xlabel('Time Since Explosion [d]')
        plt.gca().invert_yaxis()
        plt.xlim(0, 40)
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2, labelsize=12)
        plt.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2, labelsize=12)
        plt.axhline(y=min(fxfit), color="black", linestyle="--")
        plt.axhline(y=mag_after15days_polyfit, color="black", linestyle="--")
        plt.axvline(x=tmax_polyfit, color="black", linestyle="--")
        plt.axvline(x=float(time_after15days_polyfit), color="black", linestyle="--")
        print("time after 15 days polyfit = ", time_after15days_polyfit)
        plt.tight_layout()
        plt.savefig(f'{key}' + "_band_" + f'{modelname}' + "_viewing_angle" + str(angle) + ".png")
        plt.close()

    return band_risetime_polyfit, band_peakmag_polyfit, band_deltam15_polyfit


def make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, band_risetime_angle_averaged_polyfit,
                                                     band_peakmag_angle_averaged_polyfit, colours, colours2,
                                                     plotvalues, key):
    for ii, modelname in enumerate(modelnames):
        viewing_angle_plot_data = pd.read_csv(key + "band_" + f'{modelname}' + "_viewing_angle_data.txt",
                                              delimiter=" ")
        band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].values
        band_risetime_viewing_angles = viewing_angle_plot_data["risetime_polyfit"].values

        a0 = plt.scatter(band_risetime_viewing_angles, band_peak_mag_viewing_angles, marker='x', color=colours2[ii])
        p0 = plt.scatter(band_risetime_angle_averaged_polyfit[ii], band_peakmag_angle_averaged_polyfit[ii],
                         marker='o', color=colours[ii], s=40)
        plotvalues.append((a0, p0))
        plt.errorbar(band_risetime_angle_averaged_polyfit[ii], band_peakmag_angle_averaged_polyfit[ii],
                     xerr=np.std(band_risetime_viewing_angles),
                     yerr=np.std(band_peak_mag_viewing_angles), ecolor=colours[ii], capsize=2)

    plt.legend(plotvalues, modelnames, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='upper right', fontsize=8, ncol=2, columnspacing=1)
    plt.xlabel('Rise Time in Days', fontsize=14)
    plt.ylabel('Peak ' + key + ' Band Magnitude', fontsize=14)
    # plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', top=False, right=False, length=5, width=2, labelsize=12)
    plt.tick_params(axis='both', which='major', top=False, right=False, length=8, width=2, labelsize=12)
    plt.tight_layout()
    plt.savefig(key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_risetime_scatter_plot.pdf", format="pdf")
    print("saving " + key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_risetime_scatter_plot.pdf")
    plt.close()


def make_viewing_angle_peakmag_delta_m15_scatter_plot(modelnames, key, colours, colours2,
                                                      band_delta_m15_angle_averaged_polyfit,
                                                      band_peakmag_angle_averaged_polyfit, plotvalues):
    for ii, modelname in enumerate(modelnames):
        viewing_angle_plot_data = pd.read_csv(key + "band_" + f'{modelname}' + "_viewing_angle_data.txt",
                                              delimiter=" ")

        band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].values
        band_delta_m15_viewing_angles = viewing_angle_plot_data["deltam15_polyfit"].values

        a0 = plt.scatter(band_delta_m15_viewing_angles, band_peak_mag_viewing_angles, marker='x',
                         color=colours2[ii])
        p0 = plt.scatter(band_delta_m15_angle_averaged_polyfit[ii], band_peakmag_angle_averaged_polyfit[ii],
                         marker='o', color=colours[ii], s=40)
        plotvalues.append((a0, p0))
        plt.errorbar(band_delta_m15_angle_averaged_polyfit[ii], band_peakmag_angle_averaged_polyfit[ii],
                     xerr=np.std(band_delta_m15_viewing_angles),
                     yerr=np.std(band_peak_mag_viewing_angles), ecolor=colours[ii], capsize=2)

    plt.legend(plotvalues, modelnames, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='upper right', fontsize=8, ncol=2, columnspacing=1)
    plt.xlabel(r'Decline Rate ($\Delta$m$_{15}$)', fontsize=14)
    plt.ylabel('Peak ' + key + ' Band Magnitude', fontsize=14)
    # plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', top=False, right=False, length=5, width=2, labelsize=12)
    plt.tick_params(axis='both', which='major', top=False, right=False, length=8, width=2, labelsize=12)
    plt.tight_layout()
    plt.savefig(key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_delta_m15_scatter_plot.pdf", format="pdf")
    print("saving " + key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_delta_m15_scatter_plot.pdf")
    plt.close()


def calculate_costheta_phi_for_viewing_angles(viewing_angles, modelpath):
    if os.path.isfile(modelpath / 'absorptionpol_res_99.out') \
            and os.path.isfile(modelpath / 'absorptionpol_res_100.out'):
        print("Too many viewing angle bins (MABINS) for this method to work, it only works for MABINS = 100")
        exit()
    elif os.path.isfile(modelpath / 'absorptionpol_res_99.out'):
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
        for i in viewing_angles:
            MABINS = 100
            phibins = int(math.sqrt(MABINS))
            costheta_index = i // phibins
            phi_index = i % phibins

            print(f"{i:4d}   {costheta_viewing_angle_bins[costheta_index]}   {phi_viewing_angle_bins[phi_index]}")
    else:
        print("Too few viewing angle bins (MABINS) for this method to work, it only works for MABINS = 100")
        exit()

    exit()


def colour_evolution_plot(modelpaths, filternames_conversion_dict, outputfolder, args):
    font = {'size': 24}
    matplotlib.rc('font', **font)

    # colours = ['k', 'darkmagenta', 'darkred']

    if args.ymax is None:
        args.ymax = 1
    if args.ymin is None:
        args.ymin = -1
    if args.xmax is None:
        args.xmax = 80
    if args.xmin is None:
        args.xmin = 5

    if len(args.colour_evolution) > 1:
        rows = 1
        cols = 3
    else:
        rows = 1
        cols = 1
    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True,
                           figsize=(at.figwidth * 1.3 * cols, at.figwidth * 1.25 * rows),
                           tight_layout={"pad": 2.0, "w_pad": 0.2, "h_pad": 0.2})

    if len(args.colour_evolution) > 1:
        ax = ax.flatten()

    for modelnumber, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f'Reading spectra: {modelname}')

        if args.plotvspecpol:
            angles = args.plotvspecpol
        elif args.plotviewingangle and os.path.isfile(modelpath/'specpol_res.out'):
            angles = args.plotviewingangle
        else:
            angles = [None]

        for index, angle in enumerate(angles):

            for plotnumber, filters in enumerate(args.colour_evolution):

                filter_names = args.colour_evolution[plotnumber].split('-')
                args.filter = filter_names
                filters_dict = get_magnitudes(modelpath, args, angle=angle, modelnumber=modelnumber)

                time_dict_1 = {}
                time_dict_2 = {}

                plot_times = []
                diff = []

                for filter_1, filter_2 in zip(filters_dict[filter_names[0]], filters_dict[filter_names[1]]):
                    # Make magnitude dictionaries where time is the key
                    time_dict_1[float(filter_1[0])] = filter_1[1]
                    time_dict_2[float(filter_2[0])] = filter_2[1]

                for time in time_dict_1.keys():
                    if time in time_dict_2.keys():  # Test if time has a magnitude for both filters
                        plot_times.append(time)
                        diff.append(time_dict_1[time] - time_dict_2[time])

                if args.plotvspecpol and angle is not None:
                    vpkt_config = at.get_vpkt_config(modelpath)
                    viewing_angle = round(math.degrees(math.acos(vpkt_config['cos_theta'][angle])))
                    linelabel = fr"$\theta$ = {viewing_angle}"
                elif args.plotviewingangle and angle is not None:
                    linelabel = f"bin number = {angle}"
                    # linelabel = fr"$\theta$ = {angle_names[index]}$^\circ$"
                elif args.label:
                    linelabel = args.label[modelnumber]
                else:
                    linelabel = f'{modelname}'

                if linelabel is None:
                    linelabel = f'{modelname}'

                if args.color:
                    color = args.color[modelnumber]
                if args.linestyle:
                    linestyle = args.linestyle[modelnumber]

                if args.reflightcurves and modelnumber == 0:
                    colours = args.refspeccolors
                    markers = args.refspecmarkers
                    for i, reflightcurve in enumerate(args.reflightcurves):
                        plot_color_evolution_from_data(
                            filter_names, reflightcurve, colours[i], markers[i], filternames_conversion_dict,
                            ax, plotnumber)

                # if 'redshifttoz' in args and args.redshifttoz[modelnumber] != 0:
                #     plot_times = np.array(plot_times) * (1 + args.redshifttoz[modelnumber])
                #     print(f'Correcting for time dilation at redshift {args.redshifttoz[modelnumber]}')
                #     linestyle = '--'
                #     color='darkmagenta'
                #     linelabel = args.label[1]
                # else:
                #     linestyle = '-'
                #     color='k'
                #     color='k'

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    diff = filterfunc(diff)

                if len(args.colour_evolution) > 1:

                    ax[plotnumber].plot(plot_times, diff, label=linelabel, linewidth=4, linestyle=linestyle,
                                        color=color)
                else:
                    ax.plot(plot_times, diff, label=linelabel, linewidth=3, linestyle=linestyle,
                            color=color)

                if len(args.colour_evolution) > 1:
                    ax[plotnumber].text(10, args.ymax - 0.5, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')
                else:
                    ax.text(60, args.ymax * 0.8, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')
        # UNCOMMENT TO ESTIMATE COLOUR AT TIME B MAX
        # def match_closest_time(reftime):
        #     return ("{}".format(min([float(x) for x in plot_times], key=lambda x: abs(x - reftime))))
        #
        # tmax_B = 17.0  # CHANGE TO TIME OF B MAX
        # tmax_B = float(match_closest_time(tmax_B))
        # print(f'{filter_names[0]} - {filter_names[1]} at t_Bmax ({tmax_B}) = '
        #       f'{diff[plot_times.index(tmax_B)]}')

    if len(args.colour_evolution) > 1:
        fig.text(0.5, 0.025, 'Time Since Explosion [days]', ha='center', va='center')
        fig.text(0.02, 0.5, r'$\Delta$m', ha='center', va='center', rotation='vertical')
    else:
        plt.ylabel(r'$\Delta$m')
        plt.xlabel('Time Since Explosion [days]')
        plt.tight_layout()
    if not args.nolegend:
        if len(args.colour_evolution) > 1:
            ax[1].legend(loc='best', frameon=True, fontsize='x-small', ncol=1, handlelength=1)
        else:
            ax.legend(loc='best', frameon=True, fontsize='x-small', ncol=1, handlelength=1)

    if len(args.colour_evolution) > 1:
        for axis in ax:
            # axis.set_xscale('log')
            axis.minorticks_on()
            axis.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2, labelsize=18, direction='in')
            axis.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2, labelsize=18, direction='in')
    else:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2, labelsize=18, direction='in')
        ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2, labelsize=18, direction='in')

    plt.ylim(args.ymin, args.ymax)
    plt.xlim(args.xmin, args.xmax)

    args.outputfile = os.path.join(outputfolder, f'plotcolorevolution{filter_names[0]}-{filter_names[1]}.pdf')
    for i in range(2):
        if filter_names[i] in filternames_conversion_dict:
            filter_names[i] = filternames_conversion_dict[filter_names[i]]
    plt.text(10, args.ymax - 0.5, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')

    if args.show:
        plt.show()
    plt.savefig(args.outputfile, format='pdf')


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


def read_lightcurve_data(lightcurvefilename):
    filepath = Path(at.PYDIR, 'data', 'lightcurves', lightcurvefilename)
    metadata_all = at.spectra.load_yaml_path(filepath.parent.resolve())
    metadata = metadata_all.get(str(lightcurvefilename), {})

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


def plot_lightcurve_from_data(
        filter_names, lightcurvefilename, color, marker, filternames_conversion_dict, ax, plotnumber):

    lightcurve_data, metadata = read_lightcurve_data(lightcurvefilename)
    linename = metadata['label'] if plotnumber == 0 else None
    filterdir = os.path.join(at.PYDIR, 'data/filters/')

    filter_data = {}
    for plotnumber, filter_name in enumerate(filter_names):
        if filter_name == 'bol':
            continue
        f = open(filterdir / Path(f'{filter_name}.txt'))
        lines = f.readlines()
        lambda0 = float(lines[2])

        if filter_name == 'bol':
            continue
        elif filter_name in filternames_conversion_dict:
            filter_name = filternames_conversion_dict[filter_name]
        filter_data[filter_name] = lightcurve_data.loc[lightcurve_data['band'] == filter_name]
        # plt.plot(limits_x, limits_y, 'v', label=None, color=color)
        # else:

        if 'a_v' in metadata or 'e_bminusv' in metadata:
            print('Correcting for reddening')
            metadata.setdefault('r_v', metadata['a_v'] / metadata['e_bminusv'])
            metadata.setdefault('a_v', metadata['e_bminusv'] * metadata['r_v'])

            clightinangstroms = 3e+18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.array([lambda0] * len(filter_data[filter_name]['magnitude']), dtype=float)

            filter_data[filter_name]['flux'] = clightinangstroms / (lambda0 ** 2) * 10 ** -(
                (filter_data[filter_name]['magnitude'] + 48.6) / 2.5)  # gs

            filter_data[filter_name]['dered'] = apply(
                ccm89(filters[:], a_v=-metadata['a_v'], r_v=metadata['r_v']), filter_data[filter_name]['flux'])

            filter_data[filter_name]['magnitude'] = 2.5 * np.log10(
                clightinangstroms / (filter_data[filter_name]['dered'] * lambda0 ** 2)) - 48.6
        if len(filter_names) > 1:
            ax[plotnumber].plot(filter_data[filter_name]['time'], filter_data[filter_name]['magnitude'], marker,
                                label=linename, color=color)
        else:
            ax.plot(filter_data[filter_name]['time'], filter_data[filter_name]['magnitude'], marker,
                    label=linename, color=color, linewidth=4)

        # if linename == 'SN 2018byg':
        #     x_values = []
        #     y_values = []
        #     limits_x = []
        #     limits_y = []
        #     for index, row in filter_data[filter_name].iterrows():
        #         if row['date'] == 58252:
        #             plt.plot(row['time'], row['magnitude'], '*', label=linename, color=color)
        #         elif row['e_magnitude'] != -1:
        #             x_values.append(row['time'])
        #             y_values.append(row['magnitude'])
        #         else:
        #             limits_x.append(row['time'])
        #             limits_y.append(row['magnitude'])
        #     print(x_values, y_values)
        #     plt.plot(x_values, y_values, 'o', label=linename, color=color)
        #     plt.plot(limits_x, limits_y, 's', label=linename, color=color)
    return linename


def plot_color_evolution_from_data(filter_names, lightcurvefilename, color, marker, filternames_conversion_dict, ax, plotnumber):
    lightcurve_from_data, metadata = read_lightcurve_data(lightcurvefilename)
    filterdir = os.path.join(at.PYDIR, 'data/filters/')

    filter_data = []
    for i, filter_name in enumerate(filter_names):
        f = open(filterdir / Path(f'{filter_name}.txt'))
        lines = f.readlines()
        lambda0 = float(lines[2])

        if filter_name in filternames_conversion_dict:
            filter_name = filternames_conversion_dict[filter_name]
        filter_data.append(lightcurve_from_data.loc[lightcurve_from_data['band'] == filter_name])

        if 'a_v' in metadata or 'e_bminusv' in metadata:
            print('Correcting for reddening')
            if 'r_v' not in metadata:
                metadata['r_v'] = metadata['a_v'] / metadata['e_bminusv']
            elif 'a_v' not in metadata:
                metadata['a_v'] = metadata['e_bminusv'] * metadata['r_v']

            clightinangstroms = 3e+18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.array([lambda0] * filter_data[i].shape[0], dtype=float)

            filter_data[i]['flux'] = clightinangstroms / (lambda0 ** 2) * 10 ** -(
                (filter_data[i]['magnitude'] + 48.6) / 2.5)

            filter_data[i]['dered'] = apply(ccm89(filters[:], a_v=-metadata['a_v'], r_v=metadata['r_v']),
                                            filter_data[i]['flux'])

            filter_data[i]['magnitude'] = 2.5 * np.log10(
                clightinangstroms / (filter_data[i]['dered'] * lambda0 ** 2)) - 48.6

    # for i in range(2):
    #     # if metadata['label'] == 'SN 2018byg':
    #     #     filter_data[i] = filter_data[i][filter_data[i].e_magnitude != -99.00]
    #     if metadata['label'] in ['SN 2016jhr', 'SN 2018byg']:
    #         filter_data[i]['time'] = filter_data[i]['time'].apply(lambda x: round(float(x)))  # round to nearest day

    merge_dataframes = filter_data[0].merge(filter_data[1], how='inner', on=['time'])
    if len(filter_names) > 2:
        ax[plotnumber].plot(merge_dataframes['time'], merge_dataframes['magnitude_x'] - merge_dataframes['magnitude_y'], marker,
                            label=metadata['label'], color=color, linewidth=4)
    else:
        ax.plot(merge_dataframes['time'], merge_dataframes['magnitude_x'] - merge_dataframes['magnitude_y'], marker,
                label=metadata['label'], color=color)


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path(s) to ARTIS folders with light_curve.out or packets files'
                        ' (may include wildcards such as * and **)')

    parser.add_argument('-label', default=[], nargs='*',
                        help='List of series label overrides')

    parser.add_argument('--nolegend', action='store_true',
                        help='Suppress the legend from the plot')

    parser.add_argument('-color', default=[f'C{i}' for i in range(10)], nargs='*',
                        help='List of line colors')

    parser.add_argument('-linestyle', default=[], nargs='*',
                        help='List of line styles')

    parser.add_argument('-linewidth', default=[], nargs='*',
                        help='List of line widths')

    parser.add_argument('-dashes', default=[], nargs='*',
                        help='Dashes property of lines')

    parser.add_argument('--frompackets', action='store_true',
                        help='Read packets files instead of light_curve.out')

    parser.add_argument('-maxpacketfiles', type=int, default=None,
                        help='Limit the number of packet files read')

    parser.add_argument('--gamma', action='store_true',
                        help='Make light curve from gamma rays instead of R-packets')

    parser.add_argument('-packet_type', default='TYPE_ESCAPE',
                        help='Type of escaping packets')

    parser.add_argument('-escape_type', default='TYPE_RPKT',
                        help='Type of escaping packets')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path,
                        help='Filename for PDF file')

    parser.add_argument('--plotcmf', action='store_true',
                        help='Plot comoving frame light curve')

    parser.add_argument('--magnitude', action='store_true',
                        help='Plot synthetic magnitudes')

    parser.add_argument('--filter', type=str, nargs='+',
                        help='Choose filter eg. bol U B V R I. Default B. '
                        'WARNING: filter names are not case sensitive eg. sloan-r is not r, it is rs')

    parser.add_argument('--colour_evolution', nargs='*',
                        help='Plot of colour evolution. Give two filters eg. B-V')

    parser.add_argument('--print_data', action='store_true',
                        help='Print plotted data')

    parser.add_argument('--plot_hesma_model', action='store', type=Path, default=False,
                        help='Plot hesma model on top of lightcurve plot. '
                        'Enter model name saved in data/hesma directory')

    parser.add_argument('--plotvspecpol', type=int, nargs='+',
                        help='Plot vspecpol. Expects int for spec number in vspecpol files')

    parser.add_argument('--plotviewingangle', type=int, nargs='+',
                        help='Plot viewing angles. Expects int for angle number in specpol_res.out'
                        'use args = -1 to select all the viewing angles')

    parser.add_argument('-ymax', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-ymin', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-xmax', type=float, default=None,
                        help='Plot range: x-axis')

    parser.add_argument('-xmin', type=float, default=None,
                        help='Plot range: x-axis')

    parser.add_argument('-timemax', type=float, default=None,
                        help='Time max to plot')

    parser.add_argument('-timemin', type=float, default=None,
                        help='Time min to plot')

    parser.add_argument('-reflightcurves', type=str, nargs='+', dest='reflightcurves',
                        help='Also plot reference lightcurves from these files')

    parser.add_argument('-refspeccolors', default=['0.0', '0.3', '0.5'], nargs='*',
                        help='Set a list of color for reference spectra')

    parser.add_argument('-refspecmarkers', default=['o', 's', 'h'], nargs='*',
                        help='Set a list of markers for reference spectra')

    parser.add_argument('-filtersavgol', nargs=2,
                        help='Savitzky–Golay filter. Specify the window_length and poly_order.'
                             'e.g. -filtersavgol 5 3')

    parser.add_argument('-redshifttoz', type=float, nargs='+',
                        help='Redshift to z = x. Expects array length of number modelpaths.'
                        'If not to be redshifted then = 0.')

    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plot before saving')

    parser.add_argument('--calculate_peakmag_risetime_delta_m15', action='store_true',
                        help='Calculate band risetime, peak mag and delta m15 values for '
                        'the models specified using a polynomial fitting method and '
                        'print to screen')

    parser.add_argument('--save_angle_averaged_peakmag_risetime_delta_m15_to_file', action='store_true',
                        help='Save the band risetime, peak mag and delta m15 values for '
                        'the angle averaged model lightcurves to file')

    parser.add_argument('--save_viewing_angle_peakmag_risetime_delta_m15_to_file', action='store_true',
                        help='Save the band risetime, peak mag and delta m15 values for '
                        'all viewing angles specified for plotting at a later time '
                        'as these values take a long time to calculate for all '
                        'viewing angles. Need to run this command first alongside '
                        '--plotviewingangles in order to save the data for the '
                        'viewing angles you want to use before making the scatter'
                        'plots')

    parser.add_argument('--test_viewing_angle_fit', action='store_true',
                        help='Plots the lightcurves for each  viewing angle along with'
                        'the polynomial fit for each viewing angle specified'
                        'to check the fit is working properly: use alongside'
                        '--plotviewingangle ')

    parser.add_argument('--make_viewing_angle_peakmag_risetime_scatter_plot', action='store_true',
                        help='Makes scatter plot of band peak mag with risetime with the '
                        'angle averaged values being the solid dot and the errors bars'
                        'representing the standard deviation of the viewing angle'
                        'distribution')

    parser.add_argument('--make_viewing_angle_peakmag_delta_m15_scatter_plot', action='store_true',
                        help='Makes scatter plot of band peak with delta m15 with the angle'
                        'averaged values being the solid dot and the error bars representing '
                        'the standard deviation of the viewing angle distribution')

    parser.add_argument('--plotviewingangles_lightcurves', action='store_true',
                        help='Make lightcurve plots for the viewing angles and models specified')

    parser.add_argument('--average_every_tenth_viewing_angle', action='store_true',
                        help='average every tenth viewing angle to reduce noise')

    parser.add_argument('--calculate_costheta_phi_from_viewing_angle_numbers', type=int, nargs='+',
                        help='calculate costheta and phi for each viewing angle given the number of the viewing angle'
                             'Expects ints for angle number supplied from the argument of plot viewing angle'
                             'use args = -1 to select all viewing angles'
                             'Note: this method will only work if the number of angle bins (MABINS) = 100'
                             'if this is not the case an error will be printed')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS radiation field.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath and not args.magnitude and not args.colour_evolution:
        args.modelpath = ['.']
    elif not args.modelpath and (args.magnitude or args.colour_evolution):
        args.modelpath = ['.']
    elif not isinstance(args.modelpath, Iterable):
        args.modelpath = [args.modelpath]

    args.modelpath = at.flatten_list(args.modelpath)
    # flatten the list
    modelpaths = []
    for elem in args.modelpath:
        if isinstance(elem, list):
            modelpaths.extend(elem)
        else:
            modelpaths.append(elem)

    args.color, args.label, args.linestyle, args.dashes, args.linewidth = at.trim_or_pad(
        len(args.modelpath), args.color, args.label, args.linestyle, args.dashes, args.linewidth)

    if args.gamma:
        args.escape_type = 'TYPE_GAMMA'

    if args.magnitude:
        defaultoutputfile = f'plotlightcurves.pdf'
    elif args.colour_evolution:
        defaultoutputfile = 'plot_colour_evolution.pdf'
    elif args.escape_type == 'TYPE_GAMMA':
        defaultoutputfile = 'plotlightcurve_gamma.pdf'
    elif args.escape_type == 'TYPE_RPKT':
        defaultoutputfile = 'plotlightcurve.pdf'
    else:
        defaultoutputfile = f'plotlightcurve_{args.escape_type}.pdf'

    if not args.outputfile:
        outputfolder = Path()
        args.outputfile = defaultoutputfile
    elif os.path.isdir(args.outputfile):
        outputfolder = Path(args.outputfile)
        args.outputfile = os.path.join(outputfolder, defaultoutputfile)

    filternames_conversion_dict = {'rs': 'r', 'gs': 'g', 'is': 'i'}
    if args.magnitude:
        make_magnitudes_plot(modelpaths, filternames_conversion_dict, outputfolder, args)
        print(f'Saved figure: {args.outputfile}')

    elif args.colour_evolution:
        colour_evolution_plot(modelpaths, filternames_conversion_dict, outputfolder, args)
        print(f'Saved figure: {args.outputfile}')
    else:
        make_lightcurve_plot(args.modelpath, args.outputfile, args.frompackets,
                             args.escape_type, maxpacketfiles=args.maxpacketfiles, args=args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
