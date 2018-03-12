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
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.interpolate import interp1d
import artistools.spectra as spectra
from collections import defaultdict, namedtuple

def readfile(filepath_or_buffer):
    lcdata = pd.read_csv(filepath_or_buffer, delim_whitespace=True, header=None, names=['time', 'lum', 'lum_cmf'])
    # the light_curve.dat file repeats x values, so keep the first half only
    lcdata = lcdata.iloc[:len(lcdata) // 2]
    return lcdata


def get_from_packets(modelpath, lcpath, escape_type='TYPE_RPKT'):
    packetsfiles = (glob.glob(os.path.join(modelpath, 'packets00_????.out')) +
                    glob.glob(os.path.join(modelpath, 'packets00_????.out.gz')))
    ranks = [int(os.path.basename(filename)[10:10 + 4]) for filename in packetsfiles]
    nprocs = max(ranks) + 1
    print(f'Reading packets for {nprocs} processes')
    assert len(packetsfiles) == nprocs

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
            'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'])

        dfpackets.query('type == "TYPE_ESCAPE" and escape_type == @escape_type', inplace=True)

        print(f"{len(dfpackets)} {escape_type} packets escaped")

        dfpackets['t_arrive_d'] = dfpackets.apply(lambda packet: at.packets.t_arrive(packet) * u.s.to('day'), axis=1)

        binned = pd.cut(dfpackets['t_arrive_d'], timearrayplusend, labels=False, include_lowest=True)
        for binindex, e_rf_sum in dfpackets.groupby(binned)['e_rf'].sum().iteritems():
            lcdata['lum'][binindex] += e_rf_sum

        dfpackets['t_arrive_cmf_d'] = dfpackets['escape_time'] * betafactor * u.s.to('day')

        binned_cmf = pd.cut(dfpackets['t_arrive_cmf_d'], timearrayplusend, labels=False, include_lowest=True)
        for binindex, e_cmf_sum in dfpackets.groupby(binned_cmf)['e_cmf'].sum().iteritems():
            lcdata['lum_cmf'][binindex] += e_cmf_sum

    lcdata['lum'] = np.divide(lcdata['lum'] / nprocs * (u.erg / u.day).to('solLum'), arr_timedelta)
    lcdata['lum_cmf'] = np.divide(lcdata['lum_cmf'] / nprocs / betafactor * (u.erg / u.day).to('solLum'), arr_timedelta)
    return lcdata


def make_lightcurve_plot(modelpaths, filenameout, frompackets=False, gammalc=False):
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
        "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for index, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")
        lcname = 'gamma_light_curve.out' if gammalc else 'light_curve.out'
        try:
            lcpath = at.firstexisting([lcname + '.gz', lcname], path=modelpath)
        except FileNotFoundError:
            print(f"Skipping {modelname} because {lcpath} does not exist")
            continue
        if not os.path.exists(str(lcpath)):
            print(f"Skipping {modelname} because {lcpath} does not exist")
            continue
        elif frompackets:
            lcdata = at.lightcurve.get_from_packets(
                modelpath, lcpath, escape_type='TYPE_GAMMA' if gammalc else 'TYPE_RPKT')
        else:
            lcdata = at.lightcurve.readfile(lcpath)

        print("Plotting...")

        linestyle = ['-', '--'][int(index / 7)]

        axis.plot(lcdata.time, lcdata['lum'], linewidth=2, linestyle=linestyle, label=f'{modelname}')
        # axis.plot(lcdata.time, lcdata['lum_cmf'], linewidth=2, linestyle=linestyle, label=f'{modelname} (cmf)')

    # axis.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # axis.set_ylim(ymin=-0.1,ymax=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    axis.set_ylabel(r'$\mathrm{L} ' + ('_\gamma' if gammalc else '') + r'/ \mathrm{L}_\odot$')

    fig.savefig(str(filenameout), format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def get_magnitudes(modelpath):
    """Method adapted from https://github.com/cinserra/S3/blob/master/src/s3/SMS.py"""
    if os.path.isfile('specpol.out'):
        # master_branch = True
        specfilename = os.path.join(modelpath, "specpol.out")
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        specfilename = at.firstexisting(['spec.out.gz', 'spec.out'], path=modelpath)
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = specdata.columns.values[1:]

    filters_list = ['U', 'B', 'V', 'R', 'I']

    filters_dict = {}
    for filter_name in filters_list:
        if filter_name not in filters_dict:
            filters_dict[filter_name] = []

    filterdir = at.get_artistools_path() / Path('artistools/data/filters/')

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
    spectrum = spectra.get_spectrum(modelpath, timestep, timestep)

    wave, flux = [], []
    for wavelength, flambda in zip(spectrum['lambda_angstroms'], spectrum['f_lambda']):
        if wavefilter_min <= wavelength <= wavefilter_max:  # to match the spectrum wavelengths to those of the filter
            wave.append(wavelength)
            flux.append(flambda)

    return np.array(wave), np.array(flux)


def evaluate_magnitudes(flux, transmission, wave, zeropointenergyflux):
    cf = flux * transmission
    flux_obs = max(integrate.cumtrapz(cf, wave))  # using trapezoidal rule to integrate
    if flux_obs == 0.0:
        phot_filtobs_sn = 0.0
    else:
        phot_filtobs_sn = -2.5 * np.log10(flux_obs / zeropointenergyflux)

    return phot_filtobs_sn


def make_magnitudes_plot(modelpath, args):

    filters_dict = get_magnitudes(modelpath)

    if args.plot_HESMA_model:
        HESMA_model = read_HESMA_lightcurve(args)

    f, axarr = plt.subplots(nrows=2, ncols=3, sharex='all', sharey='all', squeeze=True)
    axarr = axarr.flatten()

    colours = ['purple', 'blue', 'green', 'red', 'darkred']

    for plotnumber, (key, colour) in enumerate(zip(filters_dict, colours)):
        time = []
        magnitude = []

        for t, mag in filters_dict[key]:
            time.append(float(t))
            magnitude.append(mag)

        axarr[0].invert_yaxis()
        axarr[0].plot(time, magnitude, label=key, marker='.', linestyle='None', color=colour)
        axarr[0].legend(loc='best', frameon=True, ncol=3, fontsize='xx-small')
        axarr[0].set_ylabel('Absolute Magnitude')
        axarr[0].set_xlabel('Time in Days')

        axarr[plotnumber + 1].plot(time, magnitude, label=key, marker='.', linestyle='None', color=colour)
        axarr[plotnumber + 1].legend(loc='best', frameon=True)
        axarr[plotnumber + 1].set_ylabel('Absolute Magnitude')
        axarr[plotnumber + 1].set_xlabel('Time in Days')

        if args.plot_HESMA_model and key in HESMA_model.keys():
            axarr[plotnumber + 1].plot(HESMA_model.time, HESMA_model[key], color='black')

    plt.minorticks_on()
    directory = os.getcwd().split('/')[-1]
    f.suptitle(directory)
    plt.savefig(args.outputfile, format='pdf')

    print(f'Saved figure: {args.outputfile}')


def colour_evolution_plot(filter_name1, filter_name2, modelpath, args):

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

    plt.plot(plot_times, diff, marker='.', linestyle='None')
    plt.ylabel(f'{filter_name1}-{filter_name2}')
    plt.xlabel('Time in Days')

    directory = os.getcwd().split('/')[-2:]
    plt.title(directory)
    # plt.ylim(-0.5, 3)

    plt.savefig(args.outputfile, format='pdf')

    print(f'Saved figure: {args.outputfile}')


def read_HESMA_lightcurve(args):
    HESMA_directory = at.get_artistools_path() / Path('artistools/data/HESMA')
    filename = args.plot_HESMA_model
    HESMA_modelname = HESMA_directory / filename

    HESMA_model = pd.read_csv(HESMA_modelname, delim_whitespace=True, header=None, comment='#', names=['time', 'U', 'B', 'V', 'R'])
    # HESMA_model = np.loadtxt(HESMA_modelname)

    return HESMA_model


def addargs(parser):
    parser.add_argument('modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path(s) to ARTIS folders with light_curve.out or packets files'
                        ' (may include wildcards such as * and **)')
    parser.add_argument('--frompackets', action='store_true',
                        help='Read packets files instead of light_curve.out')
    parser.add_argument('--gamma', action='store_true',
                        help='Make light curve from gamma rays instead of R-packets')
    parser.add_argument('-o', action='store', dest='outputfile', type=Path,
                        help='Filename for PDF file')
    parser.add_argument('--magnitude', action='store_true',
                        help='Plot synthetic magnitudes')
    parser.add_argument('--colour_evolution', action='store_true',
                        help='Plot of colour evolution')
    parser.add_argument('--plot_HESMA_model', action='store', type=Path, default=False,
                        help='Plot HESMA model on top of lightcurve plot. '
                        'Enter model name saved in data/HESMA directory')


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
    elif not args.modelpath:
        args.modelpath = ['.', '*']
    elif not isinstance(args.modelpath, Iterable):
        args.modelpath = [args.modelpath]

    # flatten the list
    modelpaths = []
    for elem in args.modelpath:
        if isinstance(elem, list):
            modelpaths.extend(elem)
        else:
            modelpaths.append(elem)

    if args.magnitude:
        defaultoutputfile = 'magnitude_absolute.pdf'
    elif args.colour_evolution:
        defaultoutputfile = 'color_evolution.pdf'
    else:
        defaultoutputfile = 'plotlightcurve_gamma.pdf' if args.gamma else 'plotlightcurve.pdf'

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    if args.magnitude:
        for modelpath in modelpaths:
            make_magnitudes_plot(modelpath, args)

    elif args.colour_evolution:
        for modelpath in modelpaths:
            colour_evolution_plot('B', 'V', modelpath, args)

    else:
        # combined the results of applying wildcards on each input
        modelpaths = list(itertools.chain.from_iterable([Path().glob(str(x)) for x in args.modelpath]))
        make_lightcurve_plot(modelpaths, args.outputfile, args.frompackets, args.gamma)


if __name__ == "__main__":
    main()
