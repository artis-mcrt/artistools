#!/usr/bin/env python3

import argparse
import glob
import itertools
import math
import os.path

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at
import matplotlib.pyplot as plt


def readfile(filename):
    lcdata = pd.read_csv(filename, delim_whitespace=True, header=None, names=['time', 'lum', 'lum_cmf'])
    # the light_curve.dat file repeats x values, so keep the first half only
    lcdata = lcdata.iloc[:len(lcdata) // 2]
    return lcdata


def get_from_packets(packetsfiles, timearray, nprocs, vmax, escape_type='TYPE_RPKT'):
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
        lcpath = os.path.join(modelpath, 'gamma_light_curve.out' if gammalc else 'light_curve.out')
        if not os.path.exists(lcpath):
            print(f"Skipping {modelname} because {lcpath} does not exist")
            continue
        else:
            lcdata = at.lightcurve.readfile(lcpath)
            if frompackets:
                foundpacketsfiles = glob.glob(os.path.join(modelpath, 'packets00_????.out'))
                ranks = [int(os.path.basename(filename)[10:10 + 4]) for filename in foundpacketsfiles]
                nprocs = max(ranks) + 1
                print(f'Reading packets for {nprocs} processes')
                packetsfilepaths = [os.path.join(modelpath, f'packets00_{rank:04d}.out') for rank in range(nprocs)]

                timearray = lcdata['time'].values
                # timearray = np.arange(250, 350, 0.1)
                model, _ = at.get_modeldata(os.path.join(modelpath, 'model.txt'))
                vmax = model.iloc[-1].velocity * u.km / u.s
                lcdata = at.lightcurve.get_from_packets(packetsfilepaths, timearray, nprocs, vmax,
                                                        escape_type='TYPE_GAMMA' if gammalc else 'TYPE_RPKT')

        print("Plotting...")

        linestyle = ['-', '--'][int(index / 7)]

        axis.plot(lcdata.time, lcdata['lum'], linewidth=2, linestyle=linestyle, label=f'{modelname}')
        axis.plot(lcdata.time, lcdata['lum_cmf'], linewidth=2, linestyle=linestyle, label=f'{modelname} (cmf)')

    # axis.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # axis.set_ylim(ymin=-0.1,ymax=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    axis.set_ylabel(r'$\mathrm{L} ' + ('_\gamma' if gammalc else '') + r'/ \mathrm{L}_\odot$')

    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def addargs(parser):
    parser.add_argument('modelpath', default=[], nargs='*',
                        help='Paths to ARTIS folders with light_curve.out or packets files'
                        ' (may include wildcards such as * and **)')
    parser.add_argument('--frompackets', default=False, action='store_true',
                        help='Read packets files instead of light_curve.out')
    parser.add_argument('--gamma', default=False, action='store_true',
                        help='Make light curve from gamma rays instead of R-packets')
    parser.add_argument('-o', action='store', dest='outputfile',
                        help='Filename for PDF file')


def main(argsraw=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS radiation field.')
    addargs(parser)
    args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = ['.', '*']

    # combined the results of applying wildcards on each input
    modelpaths = list(itertools.chain.from_iterable([glob.glob(x) for x in args.modelpath if os.path.isdir(x)]))

    defaultoutputfile = 'plotlightcurve_gamma.pdf' if args.gamma else 'plotlightcurve.pdf'

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    make_lightcurve_plot(modelpaths, args.outputfile, args.frompackets, args.gamma)


if __name__ == "__main__":
    main()
