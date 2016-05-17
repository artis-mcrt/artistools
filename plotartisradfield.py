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


def main():
    """
        Plot the radiation field estimators and the fitted radiation field
        based on the fitted field parameters (temperature and scale factor W
        for a diluted blackbody)
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS radiation field.')
    parser.add_argument('-path', action='store', default='./',
                        help='Path to radfield.out file')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=11,
                        help='Timestep number to plot')
    parser.add_argument('-xmin', type=int, default=100,
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
    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                             tight_layout={
                                 "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ymax1 = plot_field_estimators(axis, radfielddata)

    if len(radfielddata) < 1000:
        ymax2 = plot_fitted_field(axis, radfielddata, args)
        ymax = max(ymax1, ymax2)
        binedges = [C / radfielddata['nu_lower'].iloc[1] * 1e10] + \
            list(C / radfielddata['nu_upper'][1:] * 1e10)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=1.0,
                    color='red', label='', zorder=-1, alpha=0.3)
    else:
        ymax = ymax1
    plot_specout(axis, ymax)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_ylabel(r'J$_\lambda$ [erg/cm$^2$/m]')
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    axis.legend(loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 13})

    fig.savefig('plotradfield.pdf', format='pdf')
    plt.close()


def plot_field_estimators(axis, radfielddata):
    """
        Plot the dJ/dlambda estimators for each bin
    """
    xvalues = []
    yvalues = []
    for _, row in radfielddata.iterrows():
        if row['bin_num'] >= 0:
            xvalues.append(1e10 * C / row['nu_lower'])
            xvalues.append(1e10 * C / row['nu_upper'])
            dlambda = (C / row['nu_lower']) - \
                (C / row['nu_upper'])
            j_lambda = row['J'] / dlambda
            yvalues.append(j_lambda)
            yvalues.append(j_lambda)

    axis.plot(xvalues, yvalues, linewidth=1, label='Field estimators',
              color='blue')

    return max(yvalues)


def plot_fitted_field(axis, radfielddata, args):
    """
        Plot the fitted diluted blackbody of each bins
    """
    fittedxvalues = []
    fittedyvalues = []

    fullspecfitxvalues = []
    fullspecfityvalues = []

    for _, row in radfielddata.iterrows():
        if row['bin_num'] == -1:
            nu_lower = C / (args.xmin * 1e-10)
            nu_upper = C / (args.xmax * 1e-10)
            arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500)
            arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])
            arr_j_lambda = [j_nu * (nu_hz ** 2) / C for (nu_hz, j_nu) in zip(arr_nu_hz, arr_j_nu)]

            fullspecfitxvalues += list(C / arr_nu_hz * 1e10)
            fullspecfityvalues += arr_j_lambda
            axis.plot(fullspecfitxvalues, fullspecfityvalues, linewidth=1, color='purple',
                      label='Full-spectrum fitted field')
        else:
            arr_nu_hz = np.linspace(row['nu_lower'], row['nu_upper'], num=500)
            arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])
            arr_j_lambda = [j_nu * (nu_hz ** 2) / C for (nu_hz, j_nu) in zip(arr_nu_hz, arr_j_nu)]

            fittedxvalues += list(C / arr_nu_hz * 1e10)
            fittedyvalues += arr_j_lambda

    if fittedxvalues:
        axis.plot(fittedxvalues, fittedyvalues, linewidth=1, color='green',
                  label='Fitted field')

    return max(fittedyvalues + fullspecfityvalues)


# CGS units
def j_nu_dbb(arr_nu_hz, W, T):
    k_b = const.k_B.to('eV/K').value
    h = const.h.to('eV s').value

    if W > 0.:
        for nu_hz in arr_nu_hz:
            yield W * 1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(h * nu_hz / T / k_b))
    else:
        for nu_hz in arr_nu_hz:
            yield 0.


def plot_specout(axis, peak_value):
    """
        Plot the ARTIS spectrum
    """
    specfilename = 'spec.out'
    if not os.path.isfile(specfilename):
        specfilename = '../example_run_testing/spec.out'

    spectrum = af.get_spectrum(specfilename,
                               10, 10, normalised=True)
    spectrum['f_lambda'] = spectrum['f_lambda'] * peak_value

    spectrum.plot(x='lambda_angstroms',
                  y='f_lambda', ax=axis,
                  linewidth=1, color='black',
                  label='Emergent spectrum')


if __name__ == "__main__":
    main()
