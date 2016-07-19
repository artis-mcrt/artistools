#!/usr/bin/env python3
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
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
                        help='Path to nonthermal.out file')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=1,
                        help='Timestep number to plot')
    parser.add_argument('-timestepmax', type=int, default=-1,
                        help='Make plots for all timesteps up to this timestep')
    parser.add_argument('-modelgridindex', type=int, default=0,
                        help='Modelgridindex to plot')
    parser.add_argument('-xmin', type=int, default=40,
                        help='Plot range: minimum energy in ev')
    parser.add_argument('-xmax', type=int, default=10000,
                        help='Plot range: maximum energy in ev')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotnonthermal_{0:03d}.pdf',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        input_file = 'nonthermal.out'
        print('Loading {:}...'.format(input_file))
        nonthermaldata = pd.read_csv(input_file, delim_whitespace=True)
        nonthermaldata.query('modelgridindex==@args.modelgridindex', inplace=True)

        if not args.timestep or args.timestep < 0:
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
                outputfile = args.outputfile.format(timestep)
                make_plot(nonthermaldata_currenttimestep, timestep, outputfile, args)
            else:
                print('No data for timestep {0:d}'.format(timestep))


def make_plot(nonthermaldata, timestep, outputfile, args):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                             tight_layout={
                                 "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ymax = max(nonthermaldata['y'])
    axis.plot(nonthermaldata['energy'], nonthermaldata['y'], linewidth=1, color='blue')

    axis.annotate('Timestep {0:d}\nCell {1:d}'.format(timestep, args.modelgridindex),
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Energy (eV)')
    axis.set_ylabel(r'distribution y(x)')
    # axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    axis.legend(loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 13})

    print('Saving to {0:s}'.format(outputfile))
    fig.savefig(outputfile, format='pdf')
    plt.close()



if __name__ == "__main__":
    main()
