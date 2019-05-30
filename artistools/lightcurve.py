#!/usr/bin/env python3

import argparse
import glob
import itertools
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at
import artistools.spectra
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


def readfile(filepath_or_buffer):
    lcdata = pd.read_csv(filepath_or_buffer, delim_whitespace=True, header=None, names=['time', 'lum', 'lum_cmf'])
    # the light_curve.dat file repeats x values, so keep the first half only
    lcdata = lcdata.iloc[:len(lcdata) // 2]
    return lcdata


def get_from_packets(modelpath, lcpath, packet_type='TYPE_ESCAPE', escape_type='TYPE_RPKT', maxpacketfiles=None):
    import artistools.packets

    packetsfiles = at.packets.get_packetsfiles(modelpath, maxpacketfiles=maxpacketfiles)
    nprocs_read = len(packetsfiles)
    assert nprocs_read > 0

    timearray = at.lightcurve.readfile(lcpath)['time'].values
    # timearray = np.arange(250, 350, 0.1)
    model, _ = at.get_modeldata(modelpath)
    vmax = model.iloc[-1].velocity_outer * u.km / u.s
    betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    arr_timedelta = [at.get_timestep_time_delta(timestep, timearray) for timestep in range(len(timearray))]
    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    lcdata = pd.DataFrame({'time': timearray,
                           'lum': np.zeros_like(timearray, dtype=np.float),
                           'lum_cmf': np.zeros_like(timearray, dtype=np.float)})

    for packetsfile in packetsfiles:
        dfpackets = at.packets.readfile(packetsfile, usecols=[
            'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
            'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'],
            type=packet_type, escape_type=escape_type)

        if not (dfpackets.empty):
            print(f"sum of e_cmf {dfpackets['e_cmf'].sum()} e_rf {dfpackets['e_rf'].sum()}")

            dfpackets['t_arrive_d'] = dfpackets.apply(lambda packet: at.packets.t_arrive(packet) * u.s.to('day'), axis=1)

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
        plotkwargs['linewidth'] = 1
        plotkwargs['label']=f'{modelname} (cmf)'
        axis.plot(lcdata.time, lcdata['lum_cmf'], **plotkwargs)

    # axis.set_xlim(left=xminvalue, right=xmaxvalue)
    # axis.set_ylim(bottom=-0.1, top=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    if escape_type == 'TYPE_GAMMA':
        lum_suffix = r'_\gamma'
    elif escape_type == 'TYPE_RPKT':
        lum_suffix = r'_{\mathrm{OVOIR}}'
    else:
        lum_suffix = r'_{\mathrm{' + escape_type.replace("_", "\_") + '}}'
    axis.set_ylabel(r'$\mathrm{L} ' + lum_suffix + r'/ \mathrm{L}_\odot$')

    fig.savefig(str(filenameout), format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def get_magnitudes(modelpath, args, angle=None):
    """Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py"""
    if args and args.plotvspecpol:
        stokes_params = at.spectra.get_polarisation(angle, modelpath)
        vspecdata = stokes_params['I']
        timearray = vspecdata.keys()[1:]
    elif Path(modelpath, 'specpol.out').is_file():
        specfilename = os.path.join(modelpath, "specpol.out")
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        specfilename = at.firstexisting(['spec.out.xz', 'spec.out.gz', 'spec.out'], path=modelpath)
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = specdata.columns.values[1:]

    filters_dict = {}

    filters_dict['bol'] = [
        (time, bol_magnitude) for time, bol_magnitude in zip(timearray, bolometric_magnitude(modelpath, timearray, args, angle))
        if math.isfinite(bol_magnitude)]

    # filters_list = ['B']

    # for filter_name in filters_list:
    #     if filter_name not in filters_dict:
    #         filters_dict[filter_name] = []

    filterdir = os.path.join(at.PYDIR, 'data/filters/')

    # for filter_name in filters_list:
    #
    #     for timestep, time in enumerate(timearray):
    #
    #         zeropointenergyflux, wavefilter, transmission, wavefilter_min, wavefilter_max \
    #             = get_filter_data(filterdir, filter_name)
    #
    #         wave, flux = get_spectrum_in_filter_range(modelpath, timestep, wavefilter_min, wavefilter_max)
    #
    #         if len(wave) > len(wavefilter):
    #             interpolate_fn = interp1d(wavefilter, transmission, bounds_error=False, fill_value=0.)
    #             wavefilter = np.linspace(min(wave), int(max(wave)), len(wave))
    #             transmission = interpolate_fn(wavefilter)
    #         else:
    #             interpolate_fn = interp1d(wave, flux, bounds_error=False, fill_value=0.)
    #             wave = np.linspace(wavefilter_min, wavefilter_max, len(wavefilter))
    #             flux = interpolate_fn(wave)
    #
    #         phot_filtobs_sn = evaluate_magnitudes(flux, transmission, wave, zeropointenergyflux)
    #
    #         if phot_filtobs_sn != 0.0:
    #             phot_filtobs_sn = phot_filtobs_sn - 25  # Absolute magnitude
    #             filters_dict[filter_name].append((timearray[timestep], phot_filtobs_sn))

    return filters_dict


def bolometric_magnitude(modelpath, timearray, args, angle=None):
    magnitudes = []
    for timestep, time in enumerate(timearray):
        if angle != None:
            spectrum = at.spectra.get_vspecpol_spectrum(modelpath, time, angle, args)
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


def get_spectrum_in_filter_range(modelpath, timestep, wavefilter_min, wavefilter_max):
    spectrum = at.spectra.get_spectrum(modelpath, timestep, timestep)

    wave, flux = [], []
    for wavelength, flambda in zip(spectrum['lambda_angstroms'], spectrum['f_lambda']):
        if wavefilter_min <= wavelength <= wavefilter_max:  # to match the spectrum wavelengths to those of the filter
            wave.append(wavelength)
            flux.append(flambda)

    return np.array(wave), np.array(flux)


def evaluate_magnitudes(flux, transmission, wave, zeropointenergyflux):
    cf = flux * transmission
    flux_obs = abs(np.trapz(cf, wave))  # using trapezoidal rule to integrate
    if flux_obs == 0.0:
        phot_filtobs_sn = 0.0
    else:
        phot_filtobs_sn = -2.5 * np.log10(flux_obs / zeropointenergyflux)

    return phot_filtobs_sn


def make_magnitudes_plot(modelpaths, args):
    # rows = 1
    # cols = 1
    # f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex='all', sharey='all', squeeze=True)
    # axarr = axarr.flatten()

    if args.plotvspecpol:
        angles = args.plotvspecpol
    else:
        angles = [None]

    for angle in angles:
        linenames = []

        for modelnumber, modelpath in enumerate(modelpaths):

            modelname = at.get_model_name(modelpath)
            linenames.append(modelname)
            print(f'Reading spectra: {modelname}')
            filters_dict = get_magnitudes(modelpath, args, angle)

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
                    time.append(float(t))
                    magnitude.append(mag)
                if angle != None:
                    vpkt_data = at.get_vpkt_data(modelpath)
                    viewing_angle = round(math.degrees(math.acos(vpkt_data['cos_theta'][angle])))
                    linelabel = fr"$\theta$ = {viewing_angle}"
                else:
                    linelabel = f'{modelname}'
                plt.plot(time, magnitude, label=linelabel)

                if modelnumber == 0 and args.plot_hesma_model and key in hesma_model.keys():
                    plt.plot(hesma_model.t, hesma_model[key], color='black')

                # axarr[plotnumber].axis([0, 60, -16, -19.5])
                # plt.text(45, -19, key)
    plt.minorticks_on()
    plt.tick_params(axis='both', top=True, right=True)

        # linenames.append(plot_lightcurve_from_data(filters_dict.keys(), axarr))

    # for plot in range(rows):
    plt.ylabel('Magnitude', fontsize='x-large')
    # plt.ylabel('Magnitude')

    # plots = np.arange(0, rows*cols)
    # for plot in plots[-cols:]:
    plt.xlabel('Time in Days', fontsize='x-large')

    # axarr[0].axis([10, 50, -17.5, -19.5])
    # axarr[1].axis([10, 50, -17.5, -19.5])
    plt.axis([10, 30, -17, -19.5])
    # plt.gca().invert_yaxis()

    plt.minorticks_on()
    # axarr[0].tick_params(axis='both', top=True, right=True)
    # f.suptitle(f'{modelname}')
    # plt.subplots_adjust(hspace=.0, wspace=.0)
    # f.legend(labels=linenames, frameon=True, fontsize='xx-small')
    # plt.tight_layout()
    # f.set_figheight(8)
    # f.set_figwidth(7)
    # plt.show()
    plt.legend(loc='best')
    plt.savefig(args.outputfile, format='pdf')


def colour_evolution_plot(modelpaths, args):
    for modelpath in modelpaths:
        modelname = at.get_model_name(modelpath)
        print(f'Reading spectra: {modelname}')
        filters_dict = get_magnitudes(modelpath)

        time_dict_1 = {}
        time_dict_2 = {}

        plot_times = []
        diff = []

        filter_names = args.colour_evolution[0].split('-')

        for filter_1, filter_2 in zip(filters_dict[filter_names[0]], filters_dict[filter_names[1]]):
            # Make magnitude dictionaries where time is the key
            time_dict_1[filter_1[0]] = filter_1[1]
            time_dict_2[filter_2[0]] = filter_2[1]

        for time in time_dict_1.keys():
            if time in time_dict_2.keys():  # Test if time has a magnitude for both filters
                plot_times.append(float(time))
                diff.append(time_dict_1[time] - time_dict_2[time])

        plt.plot(plot_times, diff, label=modelname)

    # plot_color_evoloution_from_data(filter_names)

    plt.ylabel(f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')
    plt.xlabel('Time in Days Since Explosion', fontsize='x-large')
    plt.tight_layout()
    plt.legend(loc='best', frameon=True)
    plt.minorticks_on()
    plt.tick_params(axis='both', top=True, right=True)
    plt.xlim(0, 80)

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


def read_12fr_lightcurve():
    data_path = os.path.join(at.PYDIR, 'data/lightcurves/SN2012fr.dat')
    lightcurve_data = pd.read_csv(data_path)
    lightcurve_data['time'] = lightcurve_data['time'].apply(lambda x: x - 56223)
    lightcurve_data['magnitude'] = lightcurve_data['magnitude'].apply(lambda x: (x - 5 * np.log10(17.9e6/10))) ##24e6 is distance to 2012fr - needs changed
    return lightcurve_data


def read_11fe_lightcurve():
    data_path = os.path.join(at.PYDIR, "data/lightcurves/SN2011fe.dat")
    lightcurve_data = pd.read_csv(data_path)
    lightcurve_data['time'] = lightcurve_data['time'].apply(lambda x: x - (54975.239+822.6870999999956))
    lightcurve_data['magnitude'] = lightcurve_data['magnitude'].apply(lambda x: (x - 5 * np.log10(6.4e6 / 10)))
    return lightcurve_data


def plot_lightcurve_from_data(filter_names, axarr):
    lightcurve_data = read_11fe_lightcurve()

    filter_data = {}
    for plotnumber, filter_name in enumerate(filter_names):
        if filter_name is 'bol':
            continue
        filter_data[filter_name] = lightcurve_data.loc[lightcurve_data['band'] == filter_name]
        axarr[plotnumber].plot(filter_data[filter_name]['time'], filter_data[filter_name]['magnitude'], '.')
    linename = 'SN2011fe'
    return linename


def plot_color_evoloution_from_data(filter_names):
    lightcurve_from_data = read_11fe_lightcurve()

    band_data = []
    for i, band in enumerate(filter_names):
        band_data.append(lightcurve_from_data.loc[lightcurve_from_data['band'] == filter_names[i]])

    merge_dataframes = band_data[0].merge(band_data[1], how='inner', on=['time'])
    plt.plot(merge_dataframes['time'], merge_dataframes['magnitude_x'] - merge_dataframes['magnitude_y'], '.', label='SN2011fe')


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path(s) to ARTIS folders with light_curve.out or packets files'
                        ' (may include wildcards such as * and **)')

    parser.add_argument('-label', default=[], nargs='*',
                        help='List of series label overrides')

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

    parser.add_argument('--magnitude', action='store_true',
                        help='Plot synthetic magnitudes')

    parser.add_argument('--colour_evolution', action='append',
                        help='Plot of colour evolution. Give two filters eg. B-V')

    parser.add_argument('--print_data', action='store_true',
                        help='Print plotted data')

    parser.add_argument('--plot_hesma_model', action='store', type=Path, default=False,
                        help='Plot hesma model on top of lightcurve plot. '
                        'Enter model name saved in data/hesma directory')

    parser.add_argument('--plotvspecpol', type=int, nargs='+',
                        help='Plot vspecpol. Expects int for spec number in vspecpol files')


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

    at.trim_or_pad(len(args.modelpath), args.color, args.label, args.linestyle, args.dashes, args.linewidth)

    if args.gamma:
        args.escape_type = 'TYPE_GAMMA'

    if args.magnitude:
        defaultoutputfile = 'plot_colour_band_lightcurves.pdf'
    elif args.colour_evolution:
        defaultoutputfile = 'plot_colour_evolution.pdf'
    elif args.escape_type == 'TYPE_GAMMA':
        defaultoutputfile = 'plotlightcurve_gamma.pdf'
    elif args.escape_type == 'TYPE_RPKT':
        defaultoutputfile = 'plotlightcurve.pdf'
    else:
        defaultoutputfile = f'plotlightcurve_{args.escape_type}.pdf'

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    if args.magnitude:
        make_magnitudes_plot(modelpaths, args)
        print(f'Saved figure: {args.outputfile}')

    elif args.colour_evolution:
        colour_evolution_plot(modelpaths, args)
        print(f'Saved figure: {args.outputfile}')
    else:
        make_lightcurve_plot(args.modelpath, args.outputfile, args.frompackets,
                             args.escape_type, maxpacketfiles=args.maxpacketfiles, args=args)


if __name__ == "__main__":
    main()
