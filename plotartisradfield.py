#!/usr/bin/env python3
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const

import readartisfiles as af

C = const.c.to('m/s').value
K_B = const.k_B.to('eV/K').value
H = const.h.to('eV s').value


def main():
    """
        Plot the radiation field estimators and the fitted radiation field
        based on the fitted field parameters (temperature and scale factor W
        for a diluted blackbody)
    """
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

    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        radfield_file = 'radfield.out'
        print('Loading {:}...'.format(radfield_file))
        radfielddata = pd.read_csv(radfield_file, delim_whitespace=True)

        if not args.timestep or args.timestep < 0:
            selected_timestep = max(radfielddata['timestep'])
        else:
            selected_timestep = args.timestep

        radfielddata.query(
            'modelgridindex==0 and timestep==@selected_timestep',
            inplace=True)

        print('Timestep {0:d}'.format(selected_timestep))

        print('Plotting...')
        draw_plot(radfielddata, args)


def draw_plot(radfielddata, args):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                           tight_layout={
                               "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    binedges = [C / radfielddata['nu_lower'].iloc[0] * 1e10] + \
        list(C / radfielddata['nu_upper'] * 1e10)

    ymax = plot_field_estimators(ax, radfielddata)
    ax.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.2,
              color='red', label='')
    plot_fitted_field(ax, radfielddata)
    plot_specout(ax, ymax)

    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel(r'J$_\lambda$ [erg/cm$^2$/m]')
    ax.set_xlim(xmin=args.xmin, xmax=args.xmax)

    ax.legend(loc='best', handlelength=2,
              frameon=False, numpoints=1, prop={'size': 13})

    fig.savefig('plotradfield.pdf', format='pdf')
    plt.close()


def plot_field_estimators(ax, radfielddata):
    """
        Plot the dJ/dlambda estimators for each bin
    """
    xvalues = []
    yvalues = []
    for _, row in radfielddata.iterrows():
        xvalues.append(1e10 * C / row['nu_lower'])
        xvalues.append(1e10 * C / row['nu_upper'])
        dlambda = (C / row['nu_lower']) - \
            (C / row['nu_upper'])
        yvalues.append(row['J'] / dlambda)
        yvalues.append(row['J'] / dlambda)
    ax.plot(xvalues, yvalues, linewidth=1, label='Field estimators',
            color='blue')
    return max(yvalues)


def plot_fitted_field(ax, radfielddata):
    """
        Plot the fitted diluted blackbody of each bins
    """
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

    ax.plot(fittedxvalues, fittedyvalues, linewidth=1, color='green',
            label='Fitted field')


def plot_specout(ax, peak_value):
    spectrum = af.get_spectrum('../example_run_testing/spec.out',
                               10, 10, normalised=True)
    spectrum['f_lambda'] = spectrum['f_lambda'] * peak_value

    spectrum.plot(x='lambda_angstroms',
                  y='f_lambda', ax=ax,
                  linewidth=1, color='black',
                  label='Emergent spectrum')


if __name__ == "__main__":
    main()
