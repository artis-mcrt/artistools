#!/usr/bin/env python3
import argparse
import glob
import itertools
import os
import sys

from astropy import units as u
import matplotlib.pyplot as plt
import pandas as pd

import artistools as at


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS radiation field.')
    parser.add_argument('-modelpath', action='append', default=[],
                        help='Paths to artis folders with light_curve.out or packets files'
                        ' (may include wildcards such as * and **)')
    parser.add_argument('--frompackets', default=False, action='store_true',
                        help='Read packets files instead of light_curve.out')
    parser.add_argument('-o', action='store', dest='outputfile', default='plotlightcurve.pdf',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if len(args.modelpath) == 0:
        args.modelpath = ['.', '*']

    # combined the results of applying wildcards on each input
    modelpaths = list(itertools.chain.from_iterable([glob.glob(x) for x in args.modelpath if os.path.isdir(x)]))

    make_lightcurve_plot(modelpaths, args.outputfile, args.frompackets)


def make_lightcurve_plot(modelpaths, filenameout, frompackets):
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
        "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for index, modelpath in enumerate(modelpaths):
        lcfilename = os.path.join(modelpath, 'light_curve.out')
        if not os.path.exists(lcfilename):
            continue
        else:
            lcdata = pd.read_csv(lcfilename, delim_whitespace=True, header=None, names=['time', 'lum', 'lum_cmf'])
            # the light_curve.dat file repeats x values, so keep the first half only
            lcdata = lcdata.iloc[:len(lcdata) // 2]
            if frompackets:
                packetsfiles = glob.glob(os.path.join(modelpath, 'packets00_????.out'))
                print(packetsfiles)
                nprocs = len(packetsfiles)  # hopefully this is true
                dfpackets = at.packets.readfiles(packetsfiles, [
                    'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time', 'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'])
                timearray = lcdata['time'].values
                model, _ = at.get_modeldata(os.path.join(modelpath, 'model.txt'))
                vmax = model.iloc[-1].velocity * u.km / u.s
                lcdata = at.get_lightcurve_from_packets(dfpackets, timearray, nprocs, vmax)

        modelname = at.get_model_name(modelpath)
        print(f"Plotting {modelname}")

        linestyle = ['-', '--'][int(index / 7)]

        axis.plot(lcdata.time, lcdata['lum'], linewidth=2, linestyle=linestyle, label=modelname)
        axis.plot(lcdata.time, lcdata['lum_cmf'], linewidth=2, linestyle=linestyle, label=f'{modelname} (cmf)')

    # axis.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # axis.set_ylim(ymin=-0.1,ymax=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    axis.set_ylabel(r'$\mathrm{L} / \mathrm{L}_\odot$')

    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


if __name__ == "__main__":
    main()
