#!/usr/bin/env python3

import argparse
import math
# import os
# import re
import sys

from astropy import constants as const
from astropy import units as u
from functools import lru_cache
from pathlib import Path
# from itertools import chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import matplotlib.patches as mpatches

import artistools as at
import artistools.spectra


@lru_cache(maxsize=4)
def read_files(modelpath, timestep=-1, modelgridindex=-1):
    """Read radiation field data from a list of file paths into a pandas DataFrame."""
    radfielddata = pd.DataFrame()

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=modelgridindex)
    for folderpath in at.get_runfolders(modelpath, timestep=timestep):
        for mpirank in mpiranklist:
            radfieldfilename = f'radfield_{mpirank:04d}.out'
            radfieldfilepath = Path(folderpath, radfieldfilename)
            radfieldfilepath = at.firstexisting(
                [radfieldfilename + '.xz', radfieldfilename + '.gz', radfieldfilename], path=folderpath)

            if modelgridindex > -1:
                filesize = Path(radfieldfilepath).stat().st_size / 1024 / 1024
                print(f'Reading {Path(radfieldfilepath).relative_to(modelpath.parent)} ({filesize:.2f} MiB)')

            radfielddata_thisfile = pd.read_csv(radfieldfilepath, delim_whitespace=True)
            # radfielddata_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)

            if timestep >= 0:
                radfielddata_thisfile.query('timestep==@timestep', inplace=True)

            if modelgridindex >= 0:
                radfielddata_thisfile.query('modelgridindex==@modelgridindex', inplace=True)

            if not radfielddata_thisfile.empty:
                if timestep >= 0 and modelgridindex >= 0:
                    return radfielddata_thisfile
                else:
                    radfielddata = radfielddata.append(radfielddata_thisfile.copy(), ignore_index=True)

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


def get_binaverage_field(axis, radfielddata, modelgridindex=None, timestep=None):
    """Get the dJ/dlambda constant average estimators of each bin."""
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

    return arr_lambda, yvalues


def j_nu_dbb(arr_nu_hz, W, T):
    """# CGS units J_nu for dilute blackbody."""
    k_b = const.k_B.to('eV/K').value
    h = const.h.to('eV s').value

    if W > 0.:
        return [W * 1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(h * nu_hz / T / k_b)) for nu_hz in arr_nu_hz]

    return [0. for _ in arr_nu_hz]


def get_fullspecfittedfield(radfielddata, xmin, xmax, modelgridindex=None, timestep=None):
    row = radfielddata.query(
        'bin_num == -1' +
        (' & modelgridindex==@modelgridindex' if modelgridindex else '') +
        (' & timestep==@timestep' if timestep else '')).copy().iloc[0]
    nu_lower = const.c.to('angstrom/s').value / xmin
    nu_upper = const.c.to('angstrom/s').value / xmax
    arr_nu_hz = np.linspace(nu_lower, nu_upper, num=500)
    arr_j_nu = j_nu_dbb(arr_nu_hz, row['W'], row['T_R'])

    arr_lambda = const.c.to('angstrom/s').value / arr_nu_hz
    arr_j_lambda = arr_j_nu * arr_nu_hz / arr_lambda

    return arr_lambda, arr_j_lambda


def get_fitted_field(radfielddata, modelgridindex=None, timestep=None, print_bins=False, xmin=None, xmax=None):
    """Return the fitted dilute blackbody made up of all bins."""
    arr_lambda = []
    j_lambda_fitted = []

    radfielddata_subset = radfielddata.copy().query(
        'bin_num >= 0' +
        (' & modelgridindex==@modelgridindex' if modelgridindex else '') +
        (' & timestep==@timestep' if timestep else ''))

    for _, row in radfielddata_subset.iterrows():
        nu_lower = row['nu_lower']
        nu_upper = row['nu_upper']

        if row['W'] >= 0:
            arr_nu_hz_bin = np.linspace(nu_lower, nu_upper, num=500)
            arr_j_nu = j_nu_dbb(arr_nu_hz_bin, row['W'], row['T_R'])

            arr_lambda_bin = const.c.to('angstrom/s').value / arr_nu_hz_bin
            arr_j_lambda_bin = arr_j_nu * arr_nu_hz_bin / arr_lambda_bin

            arr_lambda += list(arr_lambda_bin)
            j_lambda_fitted += list(arr_j_lambda_bin)
        else:
            arr_nu_hz_bin = [nu_lower, nu_upper]
            arr_j_lambda_bin = [0., 0.]

            arr_lambda += [const.c.to('angstrom/s').value / nu for nu in arr_nu_hz_bin]
            j_lambda_fitted += arr_j_lambda_bin

        lambda_lower = const.c.to('angstrom/s').value / row['nu_upper']
        lambda_upper = const.c.to('angstrom/s').value / row['nu_lower']
        if print_bins and (xmax is None or lambda_lower < xmax) and (xmin is None or lambda_upper > xmin):
            print(f"Bin lambda_lower {lambda_lower:.1f} W {row['W']:.1e} "
                  f"contribs {row['ncontrib']} J_nu_avg {row['J_nu_avg']:.1e}")

    return arr_lambda, j_lambda_fitted


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

    if not radfielddataselected.empty:
        axis.scatter(radfielddataselected['lambda_angstroms'], radfielddataselected['Jb_lambda'],
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


def calculate_photoionrates(axes, modelpath, radfielddata, modelgridindex, timestep, xmin, xmax, args):
    axes[0].set_ylabel(r'$\sigma$ [cm$^2$]')

    arr_lambda_fitted, j_lambda_fitted = get_fitted_field(
        radfielddata, modelgridindex=modelgridindex, timestep=timestep,
        print_bins=True, xmin=xmin, xmax=xmax)

    arr_lambda_fitted, j_lambda_fitted = zip(*[
        pt for pt in zip(arr_lambda_fitted, j_lambda_fitted) if xmin <= pt[0] <= xmax])

    (contribution_list, array_jlambda_emission_total,
     arraylambda_angstrom_em) = at.spectra.get_flux_contributions_from_packets(
        modelpath, timelowerdays=-1, timeupperdays=2000, lambda_min=args.xmin, lambda_max=args.xmax,
        getemission=True, getabsorption=False, modelgridindex=modelgridindex,
        maxpacketfiles=args.maxpacketfiles,
        groupby='ion', delta_lambda=20,
        useinternalpackets=True)

    from scipy.interpolate import interp1d
    # H = 6.6260755e-27  # Planck constant [erg s]
    KB = 1.38064852e-16
    EV = 1.6021772e-12  # eV to ergs [eV/erg]
    ONEOVERH = 1.509188961e+26
    HOVERKB = 4.799243681748932e-11

    T_R = radfielddata.query('bin_num == -1').iloc[0].T_R

    ionlist = ((26, 2), (28, 2))
    ionlist = ((26, 2), )
    adata = at.get_levels(modelpath, ionlist=ionlist, get_photoionisations=True)

    # arr_nu_hz = const.c.to('angstrom/s').value / np.array(arr_lambda_fitted)
    # j_nu_fitted = np.array(j_lambda_fitted) * arr_lambda_fitted / arr_nu_hz

    for atomic_number, ion_stage in ionlist:
        ionstr = at.get_ionstring(atomic_number, ion_stage)
        print(f'{ionstr}')
        ion_data = adata.query('Z == @atomic_number and ion_stage == @ion_stage').iloc[0]
        max_levels = 9

        ion_pop = 0
        for _, level in ion_data.levels[:max_levels].iterrows():
            ion_pop += level.g * math.exp(-level.energy_ev * EV / KB / T_R)

        x = []
        x += list([(arraylambda_angstrom_em, contribrow.array_flambda_emission, contribrow.linelabel)
                   for contribrow in contribution_list])
        x += [(arraylambda_angstrom_em, array_jlambda_emission_total, 'Total emission')]
        x += [(arr_lambda_fitted, j_lambda_fitted, 'Binned field')]
        for arraylambda_angstrom, j_lambda_arr, linelabel in x:
            arr_nu_hz = const.c.to('angstrom/s').value / np.array(arraylambda_angstrom)
            print(f'Radiation field due to {linelabel}')
            j_nu_arr = np.array(j_lambda_arr) * arraylambda_angstrom / arr_nu_hz
            gamma_r_ion2 = 0.
            arr_gamma_dnu = np.zeros_like(arr_nu_hz)
            for level_num, level in ion_data.levels[:max_levels].iterrows():
                nu_threshold = ONEOVERH * (ion_data.ion_pot - level.energy_ev) * EV

                interp_sigma_bf = interp1d(
                    level.phixstable[:, 0] * nu_threshold, level.phixstable[:, 1], kind='linear', bounds_error=True,
                    fill_value=0., assume_sorted=True)

                def sigma_bf(nu):
                    nu_factor = nu / nu_threshold
                    if nu_factor < level.phixstable[0, 0]:
                        return 0.
                    elif nu_factor > level.phixstable[-1, 0]:
                        return level.phixstable[-1, 1] * math.pow(level.phixstable[-1, 0] / nu_factor, 3)

                    # return np.interp(nu_factor, level.phixstable[:, 0], level.phixstable[:, 1], left=0.)
                    return interp_sigma_bf(nu)

                levelpopfrac = level.g * math.exp(-level.energy_ev * EV / KB / T_R) / ion_pop

                arr_sigma_bf = np.array([sigma_bf(nu) for nu in arr_nu_hz])

                arr_gamma_level_dnu = np.abs(
                    ONEOVERH * arr_sigma_bf / arr_nu_hz * j_nu_arr *
                    (1 - np.exp(-HOVERKB * arr_nu_hz / T_R)) * levelpopfrac)

                arr_gamma_dnu += arr_gamma_level_dnu

                gamma_r_level = np.abs(np.trapz(arr_gamma_level_dnu, x=arr_nu_hz))
                gamma_r_ion2 += gamma_r_level
                lambda_threshold = const.c.to('angstrom/s').value / nu_threshold
                print(f'  level {level_num} pop_frac {levelpopfrac:.2f} threshold {lambda_threshold:.1f} Å '
                      f' gamma_R({ionstr}) {gamma_r_level:.2e} {level.levelname}')
                # axes[0].plot(xlist, [sigma_bf(const.c.to('angstrom/s').value / lambda_angstroms)
                #                      for lambda_angstroms in xlist],
                #              label=f'Sigma_bf({ionstr} {level.levelname})')

            # xlist = arr_lambda_fitted
            arr_gamma_dlambda = arr_gamma_dnu * arr_nu_hz / arraylambda_angstrom

            axes[1].plot(arraylambda_angstrom, arr_gamma_dlambda,
                         label=r'd$\Gamma_R$(' + ionstr + f' due to ' + linelabel + r')/d$\lambda$')

            gamma_r_ion = abs(np.trapz(arr_gamma_dlambda, x=arraylambda_angstrom))
            print(f'Gamma_R({ionstr} due to {linelabel}): {gamma_r_ion:.2e}')
            print(f'Gamma_R({ionstr} due to {linelabel}): {gamma_r_ion2:.2e}')
    axes[2].plot(arraylambda_angstrom_em, array_jlambda_emission_total, label=f'J$\lambda$ from packets')


def get_binedges(radfielddata):
    return [const.c.to('angstrom/s').value / radfielddata['nu_lower'].iloc[1]] + \
               list(const.c.to('angstrom/s').value / radfielddata['nu_upper'][1:])


def plot_celltimestep(
        modelpath, timestep, outputfile, xmin, xmax, modelgridindex, args, normalised=False):
    """Plot a cell at a timestep things like the bin edges, fitted field, and emergent spectrum (from all cells)."""
    radfielddata = read_files(modelpath, timestep=timestep, modelgridindex=modelgridindex)
    if radfielddata.empty:
        print(f'No data for timestep {timestep:d} modelgridindex {modelgridindex:d}')
        return False

    modelname = at.get_model_name(modelpath)
    time_days = at.get_timestep_times_float(modelpath)[timestep]
    print(f'Plotting {modelname} timestep {timestep:d} (t={time_days:.3f}d)')
    T_R = radfielddata.query('bin_num == -1').iloc[0].T_R
    print(f'T_R = {T_R}')

    nrows = 1 if not args.photoionrates else 3
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True,
                             figsize=(args.figscale * at.figwidth, args.figscale * at.figwidth * (0.25 + nrows * 0.4)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    axis = axes if nrows == 1 else axes[-1]

    xlist, yvalues = get_fullspecfittedfield(
        radfielddata, xmin, xmax, modelgridindex=modelgridindex, timestep=timestep)

    label = r'Dilute blackbody model '
    # label += r'(T$_{\mathrm{R}}$'
    # label += f'= {row["T_R"]} K)')
    axis.plot(xlist, yvalues, label=label, color='purple', linewidth=1.5)
    ymax = max(yvalues)

    if not args.nobandaverage:
        arr_lambda, yvalues = get_binaverage_field(
            axis, radfielddata, modelgridindex=modelgridindex, timestep=timestep)
        axis.step(arr_lambda, yvalues, where='pre', label='Band-average field', color='green', linewidth=1.5)
        ymax = max([ymax] + [point[1] for point in zip(arr_lambda, yvalues) if xmin <= point[0] <= xmax])

    arr_lambda_fitted, j_lambda_fitted = get_fitted_field(
        radfielddata, modelgridindex=modelgridindex, timestep=timestep)
    ymax = max([ymax] + [point[1] for point in zip(arr_lambda_fitted, j_lambda_fitted) if xmin <= point[0] <= xmax])

    axis.plot(arr_lambda_fitted, j_lambda_fitted, label='Radiation field model', alpha=0.8, color='blue', linewidth=1.5)

    ymax3 = plot_line_estimators(
        axis, radfielddata, xmin, xmax, modelgridindex=modelgridindex, timestep=timestep, zorder=-2, color='red')

    ymax = max(ymax, ymax3)
    if args.ymax >= 0:
        ymax = args.ymax

    try:
        specfilename = at.firstexisting(['spec.out', 'spec.out.gz'], path=modelpath)
    except FileNotFoundError:
        print(f'Could not find spec.out')
        args.nospec = True

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
        binedges = get_binedges(radfielddata)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5,
                    color='red', label='', zorder=-1, alpha=0.4)

    velocity = at.get_modeldata(modelpath)[0]['velocity'][modelgridindex]

    figure_title = f'{modelname} {velocity:.0f} km/s at {time_days:.0f}d'
    # figure_title += '\ncell {modelgridindex} timestep {timestep}'

    if not args.notitle:
        axis.set_title(figure_title, fontsize=11)

    if args.photoionrates:
        calculate_photoionrates(
            axes, modelpath, radfielddata, modelgridindex=modelgridindex, timestep=timestep, xmin=xmin, xmax=xmax, args=args)
        axes[0].legend(loc='best', handlelength=2, frameon=False, numpoints=1)
        axes[1].legend(loc='best', handlelength=2, frameon=False, numpoints=1, fontsize=4)

    # axis.annotate(figure_title,
    #               xy=(0.02, 0.96), xycoords='axes fraction',
    #               horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_ylabel(r'J$_\lambda$ [{}erg/s/cm$^2$/$\AA$]')
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=500))
    axis.set_xlim(left=xmin, right=xmax)
    axis.set_ylim(bottom=0.0, top=ymax)
    axis.yaxis.set_major_formatter(at.ExponentLabelFormatter(axis.get_ylabel(), useMathText=True))

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1)

    print(f'Saving to {outputfile}')
    fig.savefig(str(outputfile), format='pdf')
    plt.close()
    return True


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
    radfielddataselected.eval('J_lambda_fullspec_at_line = J_nu_fullspec_at_line * (@nu_line ** 2) / @const_c',
                              inplace=True)
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

    axis.plot(radfielddataselected['timestep'], radfielddataselected['Jb_lambda'],
              label=f'Jb_lu bin_num {bin_num}', **plotkwargs)


def plot_timeevolution(modelpath, outputfile, modelgridindex, args):
    """Plot a estimator evolution over time for a cell. This is not well tested and should be checked."""
    print(f'Plotting time evolution of cell {modelgridindex:d}')

    radfielddata = read_files(modelpath, modelgridindex=modelgridindex)
    radfielddataselected = radfielddata.query('modelgridindex == @modelgridindex')

    const_c = const.c.to('angstrom/s').value

    nlinesplotted = 200
    fig, axes = plt.subplots(nlinesplotted, 1, sharex=True,
                             figsize=(args.figscale * at.figwidth,
                                      args.figscale * at.figwidth * (0.25 + nlinesplotted * 0.35)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    timestep = at.get_closest_timestep(modelpath, 330)
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
            r'$\lambda$='
            f'{lambda_angstroms:.1f} Å in cell {modelgridindex:d}\n',
            xy=(0.02, 0.96), xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', fontsize=10)

        ax.set_ylabel(r'J$_\lambda$ [erg/s/cm$^2$/$\AA$]')
        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1)

    axes[-1].set_xlabel(r'Timestep')
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    # axis.set_xlim(left=xmin, right=xmax)
    # axis.set_ylim(bottom=0.0, top=ymax)

    print(f'Saving to {outputfile}')
    fig.savefig(str(outputfile), format='pdf')
    plt.close()


def addargs(parser):
    """Add arguments to an argparse parser object."""
    parser.add_argument('-modelpath', default='.', type=Path,
                        help='Path to ARTIS folder')

    parser.add_argument('-listtimesteps', action='store_true',
                        help='Show the times at each timestep')

    parser.add_argument('-maxpacketfiles', type=int, default=None,
                        help='Limit the number of packet files read')

    parser.add_argument('-xaxis', '-x', default='lambda', choices=['lambda', 'timestep'],
                        help='Horizontal axis variable.')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', action='append',
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', action='append',
                        help='Modelgridindex to plot')

    parser.add_argument('-velocity', '-v', type=float, default=-1,
                        help='Specify cell by velocity')

    parser.add_argument('--nospec', action='store_true',
                        help='Don\'t plot the emergent specrum')

    parser.add_argument('--showbinedges', action='store_true',
                        help='Plot vertical lines at the bin edges')

    parser.add_argument('-xmin', type=int, default=1000,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=20000,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('-ymax', type=int, default=-1,
                        help='Plot range: maximum J_nu')

    parser.add_argument('--normalised', action='store_true',
                        help='Normalise the spectra to their peak values')

    parser.add_argument('--notitle', action='store_true',
                        help='Suppress the top title from the plot')

    parser.add_argument('--nobandaverage', action='store_true',
                        help='Suppress the band-average line')

    parser.add_argument('--photoionrates', action='store_true',
                        help='Suppress the band-average line')

    parser.add_argument('-figscale', type=float, default=1.,
                        help='Scale factor for plot area. 1.0 is for single-column')

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

    modelpath = args.modelpath

    pdf_list = []
    modelpath_list = []
    modelgridindexlist = []

    if args.listtimesteps:
        at.showtimesteptimes(modelpath=args.modelpath)
    else:
        if args.velocity >= 0.:
            modelgridindexlist[0] = at.get_closest_cell(modelpath, args.velocity)
        else:
            modelgridindexlist = at.parse_range_list(args.modelgridindex)

        timesteplast = len(at.get_timestep_times_float(modelpath))
        if args.timedays:
            timesteplist = [at.get_closest_timestep(modelpath, args.timedays)]
        elif args.timestep:
            timesteplist = at.parse_range_list(args.timestep, dictvars={'last': timesteplast})
        else:
            print("Using last timestep.")
            timesteplist = [timesteplast]

        for modelgridindex in modelgridindexlist:
            if args.xaxis == 'lambda':
                for timestep in timesteplist:
                    outputfile = str(args.outputfile).format(modelgridindex=modelgridindex, timestep=timestep)
                    make_plot = plot_celltimestep(
                        modelpath, timestep, outputfile,
                        xmin=args.xmin, xmax=args.xmax, modelgridindex=modelgridindex,
                        args=args, normalised=args.normalised)
                    if make_plot:
                        pdf_list.append(outputfile)
                        modelpath_list.append(args.modelpath)
            elif args.xaxis == 'timestep':
                outputfile = args.outputfile.format(modelgridindex=modelgridindex)
                plot_timeevolution(modelpath, outputfile, modelgridindex, args)
            else:
                print('Unknown plot type {args.plot}')
                return 1

    if len(pdf_list) > 1:
        print(pdf_list, modelpath_list)
        at.join_pdf_files(pdf_list, modelpath_list)

    return 0


if __name__ == "__main__":
    sys.exit(main())
