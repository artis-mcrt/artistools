#!/usr/bin/env python3
import argparse
import glob
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at

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
                        help='Path to radfield_nnnn.out files')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=-1,
                        help='Timestep number to plot, or -1 for last')
    parser.add_argument('-timestepmax', type=int, default=-1,
                        help='Make plots for all timesteps up to this timestep')
    parser.add_argument('-modelgridindex', type=int, default=0,
                        help='Modelgridindex to plot')
    parser.add_argument('--nospec', action='store_true', default=False,
                        help='Don\'t plot the emergent specrum')
    parser.add_argument('-xmin', type=int, default=1000,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=20000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotartisradfield_{0:03d}.pdf',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if args.listtimesteps:
        at.showtimesteptimes('spec.out')
    else:
        radfielddata = None
        radfield_files = glob.glob('radfield_????.out', recursive=True) + \
            glob.glob('radfield-????.out', recursive=True) + glob.glob('radfield.out', recursive=True)

        if not radfield_files:
            print("No radfield files found")
            return

        for radfield_file in radfield_files:
            print(f'Loading {radfield_file}...')

            radfielddata_thisfile = pd.read_csv(radfield_file, delim_whitespace=True)
            # radfielddata_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)
            radfielddata_thisfile.query('modelgridindex==@args.modelgridindex', inplace=True)
            if len(radfielddata_thisfile) > 0:
                if radfielddata is None:
                    radfielddata = radfielddata_thisfile.copy()
                else:
                    radfielddata.append(radfielddata_thisfile, ignore_index=True)

        if radfielddata is None or len(radfielddata) == 0:
            print("No radfield data found")
            return

        if not args.timestep or args.timestep < 0:
            timestepmin = max(radfielddata['timestep'])
        else:
            timestepmin = args.timestep

        if not args.timestepmax or args.timestepmax < 0:
            timestepmax = timestepmin + 1
        else:
            timestepmax = args.timestepmax

        specfilename = 'spec.out'

        if not os.path.isfile(specfilename):
            specfilename = DEFAULTSPECPATH

        if not os.path.isfile(specfilename):
            print(f'Could not find {specfilename}')
            return

        for timestep in range(timestepmin, timestepmax):
            radfielddata_currenttimestep = radfielddata.query('timestep==@timestep')

            if len(radfielddata_currenttimestep) > 0:
                time_days = at.get_timestep_time(specfilename, timestep)
                print(f'Plotting timestep {timestep:d} (t={time_days})')
                outputfile = args.outputfile.format(timestep)
                make_plot(radfielddata_currenttimestep, specfilename, timestep, outputfile, args)
            else:
                print(f'No data for timestep {timestep:d}')


def make_plot(radfielddata, specfilename, timestep, outputfile, args):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    time_days = at.get_timestep_time(specfilename, timestep)

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

    if not args.nospec:
        modeldata, t_model_init = at.get_modeldata('model.txt')
        v_surface = modeldata.loc[int(radfielddata.modelgridindex.max())].velocity * u.km / u.s  # outer velocity
        r_surface = (327.773 * u.day * v_surface).to('km')
        r_observer = u.megaparsec.to('km')
        scale_factor = (r_observer / r_surface) ** 2 / (2 * math.pi)
        print(f'Scaling emergent spectrum flux at 1 Mpc to specific intensity at surface (v={v_surface:.3e}, r={r_surface:.3e})')
        plot_specout(axis, specfilename, timestep, scale_factor=scale_factor)  # peak_value=ymax)

    axis.annotate(f'Timestep {timestep:d} (t={time_days})\nCell {args.modelgridindex:d}',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_ylabel(r'J$_\lambda$ [erg/s/cm$^2$/$\AA$]')
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    axis.legend(loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
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
                dlambda = (C / row['nu_lower']) - (C / row['nu_upper'])
                j_lambda = row['J'] / dlambda / 1e10
                if not math.isnan(j_lambda):
                    yvalues.append(j_lambda)
                    yvalues.append(j_lambda)
                else:
                    yvalues.append(0.)
                    yvalues.append(0.)
            else:
                yvalues.append(0.)
                yvalues.append(0.)

    axis.plot(xvalues, yvalues, linewidth=1.5, label='Field estimators', color='blue')
    return max(yvalues)


def plot_fitted_field(axis, radfielddata, args):
    """
        Plot the fitted diluted blackbody for each bin as well as the global fit
    """
    fittedxvalues = []
    fittedyvalues = []
    ymaxglobalfit = -1

    for _, row in radfielddata.iterrows():
        if row['bin_num'] == -1 or row['W'] >= 0:
            if row['bin_num'] == -1:
                # Full-spectrum fit
                nu_lower = const.c.to('angstrom/s').value / args.xmin
                nu_upper = const.c.to('angstrom/s').value / args.xmax
            else:
                nu_lower = row['nu_lower']
                nu_upper = row['nu_upper']

            arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500)
            arr_lambda = const.c.to('angstrom/s').value / arr_nu_hz
            arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])
            arr_j_lambda = arr_j_nu * arr_nu_hz / arr_lambda

            if row['bin_num'] == -1:
                ymaxglobalfit = max(arr_j_lambda)
                axis.plot(arr_lambda, arr_j_lambda, linewidth=1.5, color='purple', label='Full-spectrum fitted field')
            else:
                fittedxvalues += list(arr_lambda)
                fittedyvalues += list(arr_j_lambda)
        else:
            arr_nu_hz = (row['nu_lower'], row['nu_upper'])
            arr_j_lambda = [0., 0.]

            fittedxvalues += [const.c.to('angstrom/s').value / nu for nu in arr_nu_hz]
            fittedyvalues += arr_j_lambda

    if fittedxvalues:
        axis.plot(fittedxvalues, fittedyvalues, linewidth=1.5, color='green', label='Fitted field', alpha=0.8)

    return max(max(fittedyvalues), ymaxglobalfit)


def j_nu_dbb(arr_nu_hz, W, T):
    """# CGS units J_nu for diluted blackbody"""

    k_b = const.k_B.to('eV/K').value
    h = const.h.to('eV s').value

    if W > 0.:
        return [W * 1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(h * nu_hz / T / k_b)) for nu_hz in arr_nu_hz]
    else:
        return [0. for _ in arr_nu_hz]


def plot_specout(axis, specfilename, timestep, peak_value=None, scale_factor=None):
    """
        Plot the ARTIS spectrum
    """

    print(f"Plotting {specfilename}")

    dfspectrum = at.spectra.get_spectrum(specfilename, timestep)
    if scale_factor:
        dfspectrum['f_lambda'] = dfspectrum['f_lambda'] * scale_factor
    if peak_value:
        dfspectrum['f_lambda'] = dfspectrum['f_lambda'] / dfspectrum['f_lambda'].max() * peak_value

    dfspectrum.plot(x='lambda_angstroms', y='f_lambda', ax=axis, linewidth=1.5, color='black', alpha=0.7,
                    label='Emergent spectrum (normalised)')


if __name__ == "__main__":
    main()
