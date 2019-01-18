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
    vmax = model.iloc[-1].velocity * u.km / u.s
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
    lcdata['lum_cmf'] = np.divide(lcdata['lum_cmf'] / nprocs_read / betafactor * (u.erg / u.day).to('solLum'), arr_timedelta)
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
        axis.plot(lcdata.time, lcdata['lum_cmf'], label=f'{modelname} (cmf)')

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


def get_magnitudes(modelpath):
    """Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py"""
    if Path(modelpath, 'specpol.out').is_file():
        specfilename = os.path.join(modelpath, "specpol.out")
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        specfilename = at.firstexisting(['spec.out.xz', 'spec.out.gz', 'spec.out'], path=modelpath)
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = specdata.columns.values[1:]

    filters_dict = {}

    filters_dict['bol'] = [
        (time, bol_magnitude) for time, bol_magnitude in zip(timearray, bolometric_magnitude(modelpath, timearray))
        if math.isfinite(bol_magnitude)]

    filters_list = ['U', 'B', 'V', 'R', 'I']

    for filter_name in filters_list:
        if filter_name not in filters_dict:
            filters_dict[filter_name] = []

    filterdir = os.path.join(at.PYDIR, 'data/filters/')

    for filter_name in filters_list:

        for timestep, time in enumerate(timearray):

            zeropointenergyflux, wavefilter, transmission, wavefilter_min, wavefilter_max \
                = get_filter_data(filterdir, filter_name)

            wave, flux = get_spectrum_in_filter_range(modelpath, timestep, wavefilter_min, wavefilter_max)

            if len(wave) > len(wavefilter):
                interpolate_fn = interp1d(wavefilter, transmission, bounds_error=False, fill_value=0.)
                wavefilter = np.linspace(min(wave), int(max(wave)), len(wave))
                transmission = interpolate_fn(wavefilter)
            else:
                interpolate_fn = interp1d(wave, flux, bounds_error=False, fill_value=0.)
                wave = np.linspace(wavefilter_min, wavefilter_max, len(wavefilter))
                flux = interpolate_fn(wave)

            phot_filtobs_sn = evaluate_magnitudes(flux, transmission, wave, zeropointenergyflux)

            if phot_filtobs_sn != 0.0:
                phot_filtobs_sn = phot_filtobs_sn - 25  # Absolute magnitude
                filters_dict[filter_name].append((timearray[timestep], phot_filtobs_sn))

    return filters_dict


def bolometric_magnitude(modelpath, timearray):
    magnitudes = []
    for timestep, time in enumerate(timearray):
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


def make_magnitudes_plot(modelpath, args):
    modelname = at.get_model_name(modelpath)
    print(f'Reading spectra: {modelname}')
    filters_dict = get_magnitudes(modelpath)

    if args.plot_hesma_model:
        hesma_model = read_hesma_lightcurve(args)
        linename = str(args.plot_hesma_model).split('_')[:3]
        linename = "_".join(linename)

    f, axarr = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='all', squeeze=True)
    axarr = axarr.flatten()

    colours = ['brown', 'purple', 'blue', 'green', 'red', 'darkred']
    axarr[0].invert_yaxis()
    for plotnumber, (key, colour) in enumerate(zip(filters_dict, colours)):
        time = []
        magnitude = []

        for t, mag in filters_dict[key]:
            time.append(float(t))
            magnitude.append(mag)

        axarr[plotnumber].plot(time, magnitude, label=key, color=colour)
        if args.plot_hesma_model and key in hesma_model.keys():
            axarr[plotnumber].plot(hesma_model.t, hesma_model[key], color='black', label=linename)

        axarr[plotnumber].legend(loc='best', frameon=True, fontsize='small')
        axarr[plotnumber].set_ylabel('Absolute Magnitude')
        axarr[plotnumber].set_xlabel('Time in Days')
        axarr[plotnumber].axis([0, 100, -14, -20])

    plt.minorticks_on()
    f.suptitle(f'{modelname}')
    plt.savefig(args.outputfile, format='pdf')


def colour_evolution_plot(filter_name1, filter_name2, modelpath, args):
    modelname = at.get_model_name(modelpath)
    print(f'Reading spectra: {modelname}')
    filters_dict = get_magnitudes(modelpath)

    time_dict_1 = {}
    time_dict_2 = {}

    plot_times = []
    diff = []

    for filter_1, filter_2 in zip(filters_dict[filter_name1], filters_dict[filter_name2]):
        # Make magnitude dictionaries where time is the key
        time_dict_1[filter_1[0]] = filter_1[1]
        time_dict_2[filter_2[0]] = filter_2[1]

    for time in time_dict_1.keys():
        if time in time_dict_2.keys():  # Test if time has a magnitude for both filters
            plot_times.append(float(time))
            diff.append(time_dict_1[time] - time_dict_2[time])

    plt.plot(plot_times, diff, marker='.', linestyle='None', label=modelname)
    plt.ylabel(f'{filter_name1}-{filter_name2}')
    plt.xlabel('Time in Days')

    plt.legend(loc='best', frameon=True)
    plt.title('B-V Colour Evolution')
    # plt.ylim(-0.5, 3)

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

    parser.add_argument('--colour_evolution', action='store_true',
                        help='Plot of colour evolution')

    parser.add_argument('--print_data', action='store_true',
                        help='Print plotted data')

    parser.add_argument('--plot_hesma_model', action='store', type=Path, default=False,
                        help='Plot hesma model on top of lightcurve plot. '
                        'Enter model name saved in data/hesma directory')


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

    at.trim_or_pad(len(args.modelpath), args.color, args.label, args.linestyle, args.dashes, args.linewidth)

    if args.gamma:
        args.escape_type = 'TYPE_GAMMA'

    if args.magnitude:
        defaultoutputfile = 'magnitude_absolute.pdf'
    elif args.colour_evolution:
        defaultoutputfile = 'colour_evolution.pdf'
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
        for modelpath in args.modelpath:
            make_magnitudes_plot(modelpath, args)
        print(f'Saved figure: {args.outputfile}')

    elif args.colour_evolution:
        for modelpath in args.modelpath:
            colour_evolution_plot('B', 'V', modelpath, args)
        print(f'Saved figure: {args.outputfile}')
    else:
        make_lightcurve_plot(args.modelpath, args.outputfile, args.frompackets,
                             args.escape_type, maxpacketfiles=args.maxpacketfiles, args=args)


if __name__ == "__main__":
    main()
