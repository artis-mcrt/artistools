#!/usr/bin/env python3
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const

import readartisfiles as af

parser = argparse.ArgumentParser(description='Plot ARTIS radiation field.')
parser.add_argument('-path', action='store', default='./',
                    help='Path to radfield.out file')
parser.add_argument('-listtimesteps', action='store_true', default=False,
                    help='Show the times at each timestep')
parser.add_argument('-timestep', type=int, default=10,
                    help='Timestep number to plot')
parser.add_argument('-xmin', type=int, default=2000,
                    help='Plot range: minimum wavelength in Angstroms')
parser.add_argument('-xmax', type=int, default=10000,
                    help='Plot range: maximum wavelength in Angstroms')
parser.add_argument('-o', action='store', dest='outputfile',
                    default='plotartisradfield.pdf',
                    help='Filename for PDF file')
args = parser.parse_args()


def main():
    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        make_plot()


def make_plot():
    C = const.c.to('m/s').value
    K_B = const.k_B.to('eV/K').value
    H = const.h.to('eV s').value
    radfield_file = 'radfield.out'
    print('Loading {:}...'.format(radfield_file))
    radfielddata = pd.read_csv(radfield_file, delim_whitespace=True)

    if not args.timestep or args.timestep < 0:
        selected_timestep = max(radfielddata['timestep'])
    else:
        selected_timestep = args.timestep

    # filter the list
    radfielddata = radfielddata[
        ((radfielddata[:]['modelgridindex'] == 0) &
         (radfielddata[:]['timestep'] == selected_timestep))
    ]

    print('Timestep {0:d}'.format(selected_timestep))

    xvalues = []
    yvalues = []
    for _, row in radfielddata.iterrows():
        xvalues.append(1e10 * C / row['nu_lower'])
        xvalues.append(1e10 * C / row['nu_upper'])
        dlambda = (C / row['nu_lower']) - \
            (C / row['nu_upper'])
        yvalues.append(row['J'] / dlambda)
        yvalues.append(row['J'] / dlambda)

    fittedxvalues = []
    fittedyvalues = []
    for _, row in radfielddata.iterrows():
        delta_nu = (row['nu_upper'] - row['nu_lower']) / 100
        for nu in np.arange(row['nu_lower'], row['nu_upper'], delta_nu):
            j_nu = (row['W'] * 1.4745007e-47 * pow(nu, 3) *
                    1.0 / (math.expm1(H * nu / row['T_R'] / K_B)))  # CGS units
            j_lambda = j_nu * (nu ** 2) / C

            fittedxvalues.append(C / nu * 1e10)
            fittedyvalues.append(j_lambda)

    specoutxvalues, specoutyvalues = af.get_spectrum('spec.out',
                                                     40, 70,
                                                     normalised=True)
    specoutyvalues *= max(yvalues)

    binedges = [C / radfielddata['nu_lower'].iloc[0] * 1e10] + \
        list(C / radfielddata[:]['nu_upper'] * 1e10)
    # print(binedges)
    print('Plotting...')
    draw_plot(xvalues, yvalues, fittedxvalues, fittedyvalues, binedges,
              specoutxvalues, specoutyvalues)


def draw_plot(xvalues, yvalues, fittedxvalues, fittedyvalues, binedges,
              specoutxvalues, specoutyvalues):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                           tight_layout={
                               "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ax.plot(xvalues, yvalues, linewidth=1, label='Field estimators',
            color='blue')
    ax.plot(fittedxvalues, fittedyvalues, linewidth=1, color='green',
            label='Fitted field')
    ax.vlines(binedges, ymin=0.0, ymax=max(yvalues), linewidth=0.2,
              color='0.5', label='')
    ax.plot(specoutxvalues, specoutyvalues, linewidth=1, color='black',
            label='Emergent spectrum')

    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel(r'J$_\lambda$ [erg/cm$^2$/m]')
    ax.set_xlim(xmin=args.xmin, xmax=args.xmax)

    # ax.set_xlabel(r'Energy (eV)')
    # ax.set_ylabel(r'dJ / dE')
    # ax.set_xlim(xmin=0.0, xmax=5)
    ax.legend(loc='best', handlelength=2,
              frameon=False, numpoints=1, prop={'size': 13})

    # ax.set_ylim(ymin=-0.05,ymax=1.1)

    fig.savefig('plotradfield.pdf', format='pdf')
    plt.close()


main()
