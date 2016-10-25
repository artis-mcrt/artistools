#!/usr/bin/env python3
import argparse
# import math
# import os
import glob

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from astropy import constants as const

import readartisfiles as af

C = const.c.to('m/s').value
DEFAULTSPECPATH = '../example_run/spec.out'


def main():
    """
        Plot the electron energy distribution
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS radiation field.')
    parser.add_argument('-path', action='store', default='./',
                        help='Path to nonthermalspec.out file')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=1,
                        help='Timestep number to plot')
    parser.add_argument('-timestepmax', type=int, default=-1,
                        help='Make plots for all timesteps up to this timestep')
    parser.add_argument('-modelgridindex', type=int, default=0,
                        help='Modelgridindex to plot')
    parser.add_argument('-xmin', type=int, default=40,
                        help='Plot range: minimum energy in eV')
    parser.add_argument('-xmax', type=int, default=10000,
                        help='Plot range: maximum energy in eV')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotnonthermal_cell{0:03d}_timestep{1:03d}.pdf',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        nonthermaldata = None
        nonthermal_files = glob.glob('nonthermalspec_????.out', recursive=True) + glob.glob('nonthermalspec-????.out', recursive=True) + glob.glob('nonthermalspec.out', recursive=True)
        for nonthermal_file in nonthermal_files:
            print('Loading {:}...'.format(nonthermal_file))

            nonthermaldata_thisfile = pd.read_csv(nonthermal_file, delim_whitespace=True)
            nonthermaldata_thisfile.query('modelgridindex==@args.modelgridindex', inplace=True)
            if len(nonthermaldata_thisfile) > 0:
                if nonthermaldata is None:
                    nonthermaldata = nonthermaldata_thisfile.copy()
                else:
                    nonthermaldata.append(nonthermaldata_thisfile, ignore_index=True)

        if args.timestep < 0:
            timestepmin = max(nonthermaldata['timestep'])
        else:
            timestepmin = args.timestep

        if not args.timestepmax or args.timestepmax < 0:
            timestepmax = timestepmin + 1
        else:
            timestepmax = args.timestepmax

        list_timesteps = range(timestepmin, timestepmax)

        for timestep in list_timesteps:
            nonthermaldata_currenttimestep = nonthermaldata.query('timestep==@timestep')

            if len(nonthermaldata_currenttimestep) > 0:
                print('Plotting timestep {0:d}'.format(timestep))
                outputfile = args.outputfile.format(args.modelgridindex, timestep)
                make_plot(nonthermaldata_currenttimestep, timestep, outputfile, args)
            else:
                print('No data for timestep {0:d}'.format(timestep))


def make_plot(nonthermaldata, timestep, outputfile, args):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(6, 4),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ymax = max(nonthermaldata['y'])

    # nonthermaldata.plot(x='energy_ev', y='y', lw=1.5, ax=axis, color='blue', legend=False)
    axis.plot(nonthermaldata['energy_ev'], nonthermaldata['y'], linewidth=1.5, color='blue')

    axis.annotate('Timestep {0:d}\nCell {1:d}'.format(timestep, args.modelgridindex),
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Energy (eV)')
    axis.set_ylabel(r'y (e$^-$ / cm$^2$ / s / eV)')
    axis.set_yscale("log", nonposy='clip')
    # axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    # axis.legend(loc='upper center', handlelength=2,
    #             frameon=False, numpoints=1, prop={'size': 13})

    print('Saving to {0:s}'.format(outputfile))
    fig.savefig(outputfile, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
