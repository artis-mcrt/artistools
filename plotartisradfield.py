#!/usr/bin/env python3
import argparse
import math
import os
import glob

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const

import readartisfiles as af

DEFAULTSPECPATH = '../example_run/spec.out'
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
    parser.add_argument('-timestepmax', type=int, default=-1,
                        help='Make plots for all timesteps up to this timestep')
    parser.add_argument('-modelgridindex', type=int, default=0,
                        help='Modelgridindex to plot')
    parser.add_argument('-xmin', type=int, default=1000,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=10000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotartisradfield_{0:03d}.pdf',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        radfielddata = None
        radfield_files = glob.glob('radfield_????.out', recursive=True) + \
            glob.glob('radfield-????.out', recursive=True) + glob.glob('radfield.out', recursive=True)
        for radfield_file in radfield_files:
            print(f'Loading {radfield_file}...')

            radfielddata_thisfile = pd.read_csv(radfield_file, delim_whitespace=True)
            radfielddata_thisfile.query('modelgridindex==@args.modelgridindex', inplace=True)
            if len(radfielddata_thisfile) > 0:
                if radfielddata is None:
                    radfielddata = radfielddata_thisfile.copy()
                else:
                    radfielddata.append(radfielddata_thisfile, ignore_index=True)

        if not args.timestep or args.timestep < 0:
            timestepmin = max(radfielddata['timestep'])
        else:
            timestepmin = args.timestep

        if not args.timestepmax or args.timestepmax < 0:
            timestepmax = timestepmin + 1
        else:
            timestepmax = args.timestepmax

        list_timesteps = range(timestepmin, timestepmax)

        for timestep in list_timesteps:
            radfielddata_currenttimestep = radfielddata.query('timestep==@timestep')

            if len(radfielddata_currenttimestep) > 0:
                print(f'Plotting timestep {timestep:d}')
                outputfile = args.outputfile.format(timestep)
                make_plot(radfielddata_currenttimestep, timestep, outputfile, args)
            else:
                print(f'No data for timestep {timestep:d}')


def make_plot(radfielddata, timestep, outputfile, args):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ymax1 = plot_field_estimators(axis, radfielddata)
    ymax2 = plot_fitted_field(axis, radfielddata, args)

    ymax = max(ymax1, ymax2)

    if len(radfielddata) < 400:
        binedges = [C / radfielddata['nu_lower'].iloc[1] * 1e10] + \
            list(C / radfielddata['nu_upper'][1:] * 1e10)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5,
                    color='red', label='', zorder=-1, alpha=0.4)

    plot_specout(axis, ymax, timestep)

    axis.annotate(f'Timestep {timestep:d}\nCell {args.modelgridindex:d}',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_ylabel(r'J$_\lambda$ [erg/cm$^2$/m]')
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    axis.legend(loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 13})

    print('Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


def plot_field_estimators(axis, radfielddata):
    """
        Plot the dJ/dlambda estimators for each bin
    """
    xvalues = []
    yvalues = []
    for _, row in radfielddata.iterrows():
        if row['bin_num'] >= 0:
            xvalues.append(1e10 * C / row['nu_lower'])  # in future, avoid this and use drawstyle='steps-pre'
            xvalues.append(1e10 * C / row['nu_upper'])
            if row['T_R'] >= 0.:
                dlambda = (C / row['nu_lower']) - \
                    (C / row['nu_upper'])
                j_lambda = row['J'] / dlambda
                if not math.isnan(j_lambda):
                    yvalues.append(j_lambda)
                    yvalues.append(j_lambda)
                else:
                    yvalues.append(0.)
                    yvalues.append(0.)
            else:
                yvalues.append(0.)
                yvalues.append(0.)

    axis.plot(xvalues, yvalues, linewidth=1.5, label='Field estimators',
              color='blue')
    return max(yvalues)


def plot_fitted_field(axis, radfielddata, args):
    """
        Plot the fitted diluted blackbody for each bin as well as the global fit
    """
    fittedxvalues = []
    fittedyvalues = []
    ymaxglobalfit = -1

    for _, row in radfielddata.iterrows():
        if row['bin_num'] == -1:
            # Full-spectrum fit
            nu_lower = C / (args.xmin * 1e-10)
            nu_upper = C / (args.xmax * 1e-10)
            arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500)
            arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])
            arr_j_lambda = [j_nu * (nu_hz ** 2) / C for (nu_hz, j_nu) in zip(arr_nu_hz, arr_j_nu)]

            arr_lambda_angstroms = C / arr_nu_hz * 1e10
            ymaxglobalfit = max(arr_j_lambda)
            axis.plot(arr_lambda_angstroms, arr_j_lambda, linewidth=1.5, color='purple',
                      label='Full-spectrum fitted field')
        elif row['W'] >= 0:
            arr_nu_hz = np.linspace(row['nu_lower'], row['nu_upper'], num=500)
            arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])
            arr_j_lambda = [j_nu * (nu_hz ** 2) / C for (nu_hz, j_nu) in zip(arr_nu_hz, arr_j_nu)]

            fittedxvalues += list(C / arr_nu_hz * 1e10)
            fittedyvalues += arr_j_lambda
        else:
            arr_nu_hz = (row['nu_lower'], row['nu_upper'])
            arr_j_lambda = [0., 0.]

            fittedxvalues += [C / nu * 1e10 for nu in arr_nu_hz]
            fittedyvalues += arr_j_lambda

    if fittedxvalues:
        axis.plot(fittedxvalues, fittedyvalues, linewidth=1.5, color='green',
                  label='Fitted field', alpha=0.8)

    return max(max(fittedyvalues), ymaxglobalfit)


def j_nu_dbb(arr_nu_hz, W, T):
    """# CGS units J_nu for diluted blackbody"""

    k_b = const.k_B.to('eV/K').value
    h = const.h.to('eV s').value

    if W > 0.:
        for nu_hz in arr_nu_hz:
            yield W * 1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(h * nu_hz / T / k_b))
    else:
        for nu_hz in arr_nu_hz:
            yield 0.


def plot_specout(axis, peak_value, timestep):
    """
        Plot the ARTIS spectrum
    """
    specfilename = 'spec.out'
    if not os.path.isfile(specfilename):
        specfilename = DEFAULTSPECPATH

    if not os.path.isfile(specfilename):
        print('Could not find ' + specfilename)
        return

    spectrum = af.get_spectrum(specfilename, timestep, normalised=True)
    spectrum['f_lambda'] = spectrum['f_lambda'] * peak_value

    spectrum.plot(x='lambda_angstroms',
                  y='f_lambda', ax=axis,
                  linewidth=1.5, color='black',
                  alpha=0.7,
                  label='Emergent spectrum (normalised)')


if __name__ == "__main__":
    main()
