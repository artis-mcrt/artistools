#!/usr/bin/env python3

import argparse
import glob
import math
import os.path
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at
import artistools.spectra

defaultoutputfile = 'plotradfield_cell{0:03d}_{1:03d}.pdf'


def read_files(radfield_files, modelgridindex=-1):
    radfielddata = None
    if not radfield_files:
        print("No radfield files")
    else:
        for _, radfield_file in enumerate(radfield_files):
            filerank = int(re.search('[0-9]+', os.path.basename(radfield_file)).group(0))

            if filerank > modelgridindex and modelgridindex >= 0:
                continue

            print(f'Loading {radfield_file}...')

            radfielddata_thisfile = pd.read_csv(radfield_file, delim_whitespace=True)
            # radfielddata_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)

            if modelgridindex >= 0:
                radfielddata_thisfile.query('modelgridindex==@modelgridindex', inplace=True)

            if radfielddata_thisfile is not None:
                if len(radfielddata_thisfile) > 0:
                    if radfielddata is None:
                        radfielddata = radfielddata_thisfile.copy()
                    else:
                        radfielddata = radfielddata.append(radfielddata_thisfile.copy(), ignore_index=True)

        if radfielddata is None or len(radfielddata) == 0:
            print("No radfield data found")

    return radfielddata


def plot_field_estimators(axis, radfielddata):
    """
        Plot the dJ/dlambda estimators for each bin
    """
    bindata = radfielddata.copy().query('bin_num >= 0')  # exclude the global fit parameters

    arr_lambda = const.c.to('angstrom/s').value / bindata['nu_upper'].values

    bindata['dlambda'] = bindata.apply(
        lambda row: const.c.to('angstrom/s').value * (1 / row['nu_lower'] - 1 / row['nu_upper']), axis=1)

    yvalues = bindata.apply(
        lambda row: row['J'] / row['dlambda'] if (
            not math.isnan(row['J'] / row['dlambda']) and row['T_R'] >= 0) else 0.0, axis=1).values

    # add the starting point
    arr_lambda = np.insert(arr_lambda, 0, const.c.to('angstrom/s').value / bindata['nu_lower'].iloc[0])
    yvalues = np.insert(yvalues, 0, 0.)

    axis.step(arr_lambda, yvalues, where='pre', linewidth=1.5, label='Field estimators', color='blue')
    return max(yvalues)


def plot_fitted_field(axis, radfielddata, xmin, xmax):
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
                nu_lower = const.c.to('angstrom/s').value / xmin
                nu_upper = const.c.to('angstrom/s').value / xmax
            else:
                nu_lower = row['nu_lower']
                nu_upper = row['nu_upper']

            arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500)
            arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])

            arr_lambda = const.c.to('angstrom/s').value / arr_nu_hz
            arr_j_lambda = arr_j_nu * arr_nu_hz / arr_lambda

            if row['bin_num'] == -1:
                ymaxglobalfit = max(arr_j_lambda)
                axis.plot(arr_lambda, arr_j_lambda, linewidth=1.5, color='purple', label='Full-spectrum fitted field')
                print(row)
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


def make_plot(radfielddata, modelpath, specfilename, timestep, outputfile, xmin, xmax, modelgridindex, nospec=False,
              normalised=False):
    """
    Draw the bin edges, fitted field, and emergent spectrum
    """
    time_days = at.get_timestep_time(specfilename, timestep)

    print(f'Plotting timestep {timestep:d} (t={time_days})')

    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ymax1 = at.radfield.plot_field_estimators(axis, radfielddata)
    ymax2 = at.radfield.plot_fitted_field(axis, radfielddata, xmin, xmax)

    ymax = max(ymax1, ymax2)

    if len(radfielddata) < 400:
        binedges = [const.c.to('angstrom/s').value / radfielddata['nu_lower'].iloc[1]] + \
            list(const.c.to('angstrom/s').value / radfielddata['nu_upper'][1:])
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5,
                    color='red', label='', zorder=-1, alpha=0.4)
    if not nospec:
        if not normalised:
            modeldata, t_model_init = at.get_modeldata(os.path.join(modelpath, 'model.txt'))
            v_surface = modeldata.loc[int(radfielddata.modelgridindex.max())].velocity * u.km / u.s  # outer velocity
            r_surface = (327.773 * u.day * v_surface).to('km')
            r_observer = u.megaparsec.to('km')
            scale_factor = (r_observer / r_surface) ** 2 / (2 * math.pi)
            print(f'Scaling emergent spectrum flux at 1 Mpc to specific intensity '
                  f'at surface (v={v_surface:.3e}, r={r_surface:.3e})')
            plot_specout(axis, specfilename, timestep, scale_factor=scale_factor)  # peak_value=ymax)
        else:
            plot_specout(axis, specfilename, timestep, peak_value=ymax)

    T_R = radfielddata.query('bin_num == -1').iloc[0].T_R

    axis.annotate(f'Timestep {timestep:d} (t={time_days})\nCell {modelgridindex:d}\nT_R = {T_R:.0f} K',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_ylabel(r'J$_\lambda$ [erg/s/cm$^2$/$\AA$]')
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmin=xmin, xmax=xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    axis.legend(loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


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


def addargs(parser):
    parser.add_argument('-modelpath', default='',
                        help='Path to ARTIS folder')

    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', action='append',
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('--nospec', action='store_true', default=False,
                        help='Don\'t plot the emergent specrum')

    parser.add_argument('-xmin', type=int, default=1000,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=20000,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('--normalised', default=False, action='store_true',
                        help='Normalise the spectra to their peak values')

    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='Filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    """
    Plot the radiation field estimators and the fitted radiation field
    based on the fitted field parameters (temperature and scale factor W
    for a diluted blackbody)
    """

    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS internal radiation field.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    specfilename = os.path.join(args.modelpath, 'spec.out')

    if args.listtimesteps:
        at.showtimesteptimes(args.modelpath)
    else:
        radfield_files = (
            glob.glob(os.path.join(args.modelpath, 'radfield_????.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, 'radfield_????.out.gz'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, '*/radfield_????.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, '*/radfield_????.out.gz'), recursive=True))

        if not radfield_files:
            print("No radfield files found")
            return 1
        else:
            radfielddata = at.radfield.read_files(radfield_files, args.modelgridindex)


        if not os.path.isfile(specfilename):
            print(f'Could not find {specfilename}')
            args.nospec = True

        timesteplast = max(radfielddata['timestep'])
        if args.timedays:
            timesteplist = [at.get_closest_timestep(specfilename, args.timedays)]
        elif args.timestep:
            timesteplist = at.parse_range_list(args.timestep, dictvars={'last': timesteplast})
        else:
            print("Using last timestep.")
            timesteplist = [timesteplast]

        for timestep in timesteplist:
            radfielddata_currenttimestep = radfielddata.query('timestep==@timestep')

            if len(radfielddata_currenttimestep) > 0:
                outputfile = args.outputfile.format(args.modelgridindex, timestep)
                make_plot(radfielddata_currenttimestep, args.modelpath, specfilename, timestep, outputfile,
                          xmin=args.xmin, xmax=args.xmax, modelgridindex=args.modelgridindex, nospec=args.nospec,
                          normalised=args.normalised)
            else:
                print(f'No data for timestep {timestep:d}')

    return 0


if __name__ == "__main__":
    sys.exit(main())
