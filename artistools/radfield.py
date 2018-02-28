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
from pathlib import Path

import artistools as at
import artistools.spectra


def read_files(radfield_files, modelgridindex=-1):
    """Read radiation field data from a list of file paths into a pandas DataFrame."""
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
                if not radfielddata_thisfile.empty:
                    if radfielddata is None:
                        radfielddata = radfielddata_thisfile.copy()
                    else:
                        radfielddata = radfielddata.append(radfielddata_thisfile.copy(), ignore_index=True)

        if radfielddata is None or len(radfielddata) == 0:
            print("No radfield data found")

    return radfielddata


def select_bin(radfielddata, nu=None, lambda_angstroms=None, modelgridindex=None, timestep=None):
    assert nu is None or lambda_angstroms is None

    if lambda_angstroms is not None:
        nu = const.c.to('angstrom/s').value / lambda_angstroms
    else:
        lambda_angstroms = const.c.to('angstrom/s').value / nu

    dfselected = radfielddata.query(
        ('modelgridindex == @modelgridindex and ' if modelgridindex else '') +
        ('timestep == @timestep and ' if timestep else '') +
        'nu_lower <= @nu and nu_upper >= @nu and bin_num > -1')

    assert not dfselected.empty
    return dfselected.iloc[0].bin_num, dfselected.iloc[0].nu_lower, dfselected.iloc[0].nu_upper


def plot_field_estimators(axis, radfielddata, modelgridindex=None, timestep=None, **plotkwargs):
    """Plot the dJ/dlambda constant average estimators for each bin."""
    # exclude the global fit parameters and detailed lines with negative "bin_num"
    bindata = radfielddata.copy().query(
        'bin_num >= 0' +
        (' & modelgridindex==@modelgridindex' if modelgridindex else '') +
        (' & timestep==@timestep' if timestep else ''))

    arr_lambda = const.c.to('angstrom/s').value / bindata['nu_upper'].values

    bindata['dlambda'] = bindata.apply(
        lambda row: const.c.to('angstrom/s').value * (1 / row['nu_lower'] - 1 / row['nu_upper']), axis=1)

    yvalues = bindata.apply(
        lambda row: row['J'] / row['dlambda'] if (
            not math.isnan(row['J'] / row['dlambda']) and row['T_R'] >= 0) else 0.0, axis=1).values

    # add the starting point
    arr_lambda = np.insert(arr_lambda, 0, const.c.to('angstrom/s').value / bindata['nu_lower'].iloc[0])
    yvalues = np.insert(yvalues, 0, 0.)

    axis.step(arr_lambda, yvalues, where='pre', label='Field estimators', **plotkwargs)

    return max(yvalues)


def j_nu_dbb(arr_nu_hz, W, T):
    """# CGS units J_nu for diluted blackbody."""
    k_b = const.k_B.to('eV/K').value
    h = const.h.to('eV s').value

    if W > 0.:
        return [W * 1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(h * nu_hz / T / k_b)) for nu_hz in arr_nu_hz]

    return [0. for _ in arr_nu_hz]


def plot_fitted_field(axis, radfielddata, xmin, xmax, modelgridindex=None, timestep=None, **plotkwargs):
    """Plot the fitted diluted blackbody for each bin as well as the global fit."""
    fittedxvalues = []
    fittedyvalues = []
    ymaxglobalfit = -1

    radfielddata_subset = radfielddata.copy().query(
        'bin_num >= -1' +
        (' & modelgridindex==@modelgridindex' if modelgridindex else '') +
        (' & timestep==@timestep' if timestep else ''))

    for _, row in radfielddata_subset.iterrows():
        if row['bin_num'] == -1 or row['W'] >= 0:
            if row['bin_num'] == -1:
                # Full-spectrum fitted field should be plotted for the full x range
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
        axis.plot(fittedxvalues, fittedyvalues, label='Fitted field', **plotkwargs)

    return max(max(fittedyvalues), ymaxglobalfit)


def plot_line_estimators(axis, radfielddata, xmin, xmax, modelgridindex=None, timestep=None, **plotkwargs):
    """Plot the Jblue_lu values from the detailed line estimators on a spectrum."""
    ymax = -1

    radfielddataselected = radfielddata.query(
        'bin_num < -1' +
        (' & modelgridindex==@modelgridindex' if modelgridindex else '') +
        (' & timestep==@timestep' if timestep else ''))[['nu_upper', 'J_nu_avg']]

    const_c = const.c.to('angstrom/s').value
    radfielddataselected.eval('lambda_angstroms = @const_c / nu_upper', inplace=True)
    radfielddataselected.eval('Jb_lambda = J_nu_avg * (nu_upper ** 2) / @const_c', inplace=True)

    ymax = radfielddataselected['Jb_lambda'].max()

    axis.scatter(radfielddataselected['lambda_angstroms'].values, radfielddataselected['Jb_lambda'].values,
                 label='Line estimators', s=0.2, **plotkwargs)

    return ymax


def plot_specout(axis, specfilename, timestep, peak_value=None, scale_factor=None, **plotkwargs):
    """Plot the ARTIS spectrum."""
    print(f"Plotting {specfilename}")

    dfspectrum = at.spectra.get_spectrum(specfilename, timestep)
    label = 'Emergent spectrum'
    if scale_factor is not None:
        label += ' (scaled)'
        dfspectrum['f_lambda'] = dfspectrum['f_lambda'] * scale_factor

    if peak_value is not None:
        label += ' (normalised)'
        dfspectrum['f_lambda'] = dfspectrum['f_lambda'] / dfspectrum['f_lambda'].max() * peak_value

    dfspectrum.plot(x='lambda_angstroms', y='f_lambda', ax=axis, label=label, **plotkwargs)


def plot_celltimestep(
        radfielddata, modelpath, specfilename, timestep, outputfile,
        xmin, xmax, modelgridindex, args, normalised=False):
    """Plot a cell at a timestep things like the bin edges, fitted field, and emergent spectrum (from all cells)."""
    time_days = at.get_timestep_time(specfilename, timestep)

    print(f'Plotting timestep {timestep:d} (t={time_days})')

    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(8, 4), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ymax1 = plot_field_estimators(
        axis, radfielddata, modelgridindex=modelgridindex, timestep=timestep, color='blue', linewidth=1.5)

    ymax2 = plot_fitted_field(
        axis, radfielddata, xmin, xmax, modelgridindex=modelgridindex, timestep=timestep,
        alpha=0.8, color='green', linewidth=1.5)

    ymax3 = plot_line_estimators(
        axis, radfielddata, xmin, xmax, modelgridindex=modelgridindex, timestep=timestep, zorder=-2, color='red')

    ymax = max(ymax1, ymax2, ymax3)

    if not args.nospec:
        plotkwargs = {}
        if not normalised:
            modeldata, _t_model_init = at.get_modeldata(modelpath)
            # outer velocity
            v_surface = modeldata.loc[int(radfielddata.modelgridindex.max())].velocity * u.km / u.s
            r_surface = (time_days * u.day * v_surface).to('km')
            r_observer = u.megaparsec.to('km')
            scale_factor = (r_observer / r_surface) ** 2 / (2 * math.pi)
            print(f'Scaling emergent spectrum flux at 1 Mpc to specific intensity '
                  f'at surface (v={v_surface:.3e}, r={r_surface:.3e})')
            plotkwargs['scale_factor'] = scale_factor
        else:
            plotkwargs['peak_value'] = ymax

        plot_specout(axis, specfilename, timestep, zorder=-1,
                     color='black', alpha=0.6, linewidth=1.0, **plotkwargs)

    if args.showbinedges:
        binedges = [const.c.to('angstrom/s').value / radfielddata['nu_lower'].iloc[1]] + \
            list(const.c.to('angstrom/s').value / radfielddata['nu_upper'][1:])
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5,
                    color='red', label='', zorder=-1, alpha=0.4)

    T_R = radfielddata.query('bin_num == -1').iloc[0].T_R

    axis.annotate(f'Timestep {timestep:d} (t={time_days})\nCell {modelgridindex:d}\nT_R = {T_R:.0f} K',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_ylabel(r'J$_\lambda$ [erg/s/cm$^2$/$\AA$]')
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmin=xmin, xmax=xmax)
    axis.set_ylim(ymin=0.0, ymax=ymax)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile}')
    fig.savefig(str(outputfile), format='pdf')
    plt.close()


def plot_bin_fitted_field_evolution(axis, radfielddata, nu_line, modelgridindex, **plotkwargs):
    bin_num, _nu_lower, _nu_upper = select_bin(radfielddata, nu=nu_line, modelgridindex=modelgridindex)
    # print(f"Selected bin_num {bin_num} to get a binned radiation field estimator")

    radfielddataselected = radfielddata.query(
        'bin_num == @bin_num and modelgridindex == @modelgridindex'
        ' and nu_lower <= @nu_line and nu_upper >= @nu_line').copy()

    radfielddataselected['Jb_nu_at_line'] = radfielddataselected.apply(
        lambda x: j_nu_dbb([nu_line], x.W, x.T_R)[0], axis=1)

    const_c = const.c.to('angstrom/s').value
    radfielddataselected.eval('Jb_lambda_at_line = Jb_nu_at_line * (@nu_line ** 2) / @const_c', inplace=True)
    lambda_angstroms = const_c / nu_line

    radfielddataselected.plot(x='timestep', y='Jb_lambda_at_line', ax=axis,
                              label=f'Fitted field from bin at {lambda_angstroms:.1f} Å', **plotkwargs)


def plot_global_fitted_field_evolution(axis, radfielddata, nu_line, modelgridindex, **plotkwargs):
    radfielddataselected = radfielddata.query('bin_num == -1 and modelgridindex == @modelgridindex').copy()

    radfielddataselected['J_nu_fullspec_at_line'] = radfielddataselected.apply(
        lambda x: j_nu_dbb([nu_line], x.W, x.T_R)[0], axis=1)

    const_c = const.c.to('angstrom/s').value
    radfielddataselected.eval('J_lambda_fullspec_at_line = J_nu_fullspec_at_line * (@nu_line ** 2) / @const_c', inplace=True)
    lambda_angstroms = const_c / nu_line

    radfielddataselected.plot(x='timestep', y='J_lambda_fullspec_at_line', ax=axis,
                              label=f'Full-spec fitted field at {lambda_angstroms:.1f} Å', **plotkwargs)


def plot_line_estimator_evolution(axis, radfielddata, bin_num, modelgridindex=None,
                                  timestep_min=None, timestep_max=None, **plotkwargs):
    """Plot the Jblue_lu values over time for a detailed line estimators."""

    radfielddataselected = radfielddata.query(
        'bin_num == @bin_num' +
        (' & modelgridindex == @modelgridindex' if modelgridindex else '') +
        (' & timestep >= @timestep_min' if timestep_min else '') +
        (' & timestep <= @timestep_max' if timestep_max else ''))[['timestep', 'nu_upper', 'J_nu_avg']]

    const_c = const.c.to('angstrom/s').value
    radfielddataselected.eval('lambda_angstroms = @const_c / nu_upper', inplace=True)
    radfielddataselected.eval('Jb_lambda = J_nu_avg * (nu_upper ** 2) / @const_c', inplace=True)

    axis.plot(radfielddataselected['timestep'].values, radfielddataselected['Jb_lambda'].values,
              label=f'Jb_lu bin_num {bin_num}', **plotkwargs)


def plot_timeevolution(
        radfielddata, modelpath, outputfile, modelgridindex, args):
    """Plot a estimator evolution over time for a cell."""
    print(f'Plotting time evolution of cell {modelgridindex:d}')

    radfielddataselected = radfielddata.query('modelgridindex == @modelgridindex')

    const_c = const.c.to('angstrom/s').value

    nlinesplotted = 200
    fig, axes = plt.subplots(nlinesplotted, 1, sharex=True, figsize=(8, 1 + 3 * nlinesplotted),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    time_days = 330
    timestep = at.get_closest_timestep(modelpath, time_days)
    time_days = float(at.get_timestep_time(modelpath, timestep))

    dftopestimators = radfielddataselected.query('timestep==@timestep and bin_num < -1').copy()
    dftopestimators.eval('lambda_angstroms = @const_c / nu_upper', inplace=True)
    dftopestimators.eval('Jb_lambda = J_nu_avg * (nu_upper ** 2) / @const_c', inplace=True)
    dftopestimators.sort_values(by='Jb_lambda', ascending=False, inplace=True)
    dftopestimators = dftopestimators.iloc[0:nlinesplotted]

    print(f'Top estimators at timestep {timestep} t={time_days:.1f}')
    print(dftopestimators)

    for ax, bin_num_estimator, nu_line in zip(axes, dftopestimators.bin_num.values, dftopestimators.nu_upper.values):
        lambda_angstroms = const_c / nu_line
        print(f"Selected line estimator with bin_num {bin_num_estimator}, lambda={lambda_angstroms:.1f}")
        plot_line_estimator_evolution(ax, radfielddataselected, bin_num_estimator, modelgridindex=modelgridindex)

        plot_bin_fitted_field_evolution(ax, radfielddata, nu_line, modelgridindex=modelgridindex)

        plot_global_fitted_field_evolution(ax, radfielddata, nu_line, modelgridindex=modelgridindex)
        ax.annotate(
            f'$\lambda$={lambda_angstroms:.1f} Å in cell {modelgridindex:d}\n',
            xy=(0.02, 0.96), xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', fontsize=10)

        ax.set_ylabel(r'J$_\lambda$ [erg/s/cm$^2$/$\AA$]')
        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})

    axes[-1].set_xlabel(r'Timestep')
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    # axis.set_xlim(xmin=xmin, xmax=xmax)
    # axis.set_ylim(ymin=0.0, ymax=ymax)

    print(f'Saving to {outputfile}')
    fig.savefig(str(outputfile), format='pdf')
    plt.close()


def addargs(parser):
    """Add arguments to an argparse parser object."""
    parser.add_argument('-modelpath', default='.', type=Path,
                        help='Path to ARTIS folder')

    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')

    parser.add_argument('-xaxis', '-x', default='lambda', choices=['lambda', 'timestep'],
                        help='Horizontal axis variable.')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', action='append',
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('--nospec', action='store_true', default=False,
                        help='Don\'t plot the emergent specrum')

    parser.add_argument('--showbinedges', action='store_true', default=False,
                        help='Plot vertical lines at the bin edges')

    parser.add_argument('-xmin', type=int, default=1000,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=20000,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('--normalised', default=False, action='store_true',
                        help='Normalise the spectra to their peak values')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path,
                        help='Filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    """Plot the radiation field estimators."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS internal radiation field estimators.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if args.xaxis == 'lambda':
        defaultoutputfile = Path('plotradfield_cell{modelgridindex:03d}_ts{timestep:03d}.pdf')
    else:
        defaultoutputfile = Path('plotradfield_cell{modelgridindex:03d}_evolution.pdf')

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif args.outputfile.is_dir():
        args.outputfile = args.outputfile / defaultoutputfile

    specfilename = at.firstexisting(['spec.out', 'spec.out.gz'], path=args.modelpath)

    if args.listtimesteps:
        at.showtimesteptimes(modelpath=args.modelpath)
    else:
        filenames = [Path('radfield_????.out'),
                     Path('radfield_????.out.gz'),
                     Path('*/radfield_????.out'),
                     Path('*/radfield_????.out.gz')]

        radfield_files = []
        for filename in filenames:
            radfield_files.extend(glob.glob(str(args.modelpath / filename), recursive=True))

        if not radfield_files:
            print("No radfield files found")
            return 1

        radfielddata = read_files(radfield_files, args.modelgridindex)

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

        if args.xaxis == 'lambda':
            for timestep in timesteplist:
                radfielddata_currenttimestep = radfielddata.query('timestep==@timestep')

                if not radfielddata_currenttimestep.empty:
                    outputfile = str(args.outputfile).format(modelgridindex=args.modelgridindex, timestep=timestep)
                    plot_celltimestep(
                        radfielddata_currenttimestep, args.modelpath, specfilename, timestep, outputfile,
                        xmin=args.xmin, xmax=args.xmax, modelgridindex=args.modelgridindex,
                        args=args, normalised=args.normalised)
                else:
                    print(f'No data for timestep {timestep:d}')
        elif args.xaxis == 'timestep':
            outputfile = args.outputfile.format(modelgridindex=args.modelgridindex)
            plot_timeevolution(radfielddata, args.modelpath, outputfile, args.modelgridindex, args)
        else:
            print('Unknown plot type {args.plot}')
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
