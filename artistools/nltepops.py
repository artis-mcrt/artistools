#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
import sys
from collections import namedtuple
from pathlib import Path
from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from astropy import constants as const

import artistools as at
import artistools.estimators

defaultoutputfile = 'plotnlte_{elsymbol}_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf'


def get_nltepops(modelpath, timestep, modelgridindex):
    """Read in NLTE populations from a model for a particular timestep and grid cell"""
    mpirank = at.get_mpirankofcell(modelgridindex, modelpath=modelpath)

    nlte_files = list(chain(
        Path(modelpath).rglob(f'nlte_{mpirank:04d}.out'),
        Path(modelpath).rglob(f'nlte_{mpirank:04d}.out.gz')))

    if not nlte_files:
        print("No NLTE files found.")
        return
    else:
        print(f'Loading {len(nlte_files)} NLTE files')
        for nltefilepath in nlte_files:
            # print(f'Reading {nltefilepath}')
            dfpop = pd.read_csv(nltefilepath, delim_whitespace=True)

            dfpop.query('(modelgridindex==@modelgridindex) & (timestep==@timestep)', inplace=True)
            if not dfpop.empty:
                return dfpop

    return pd.DataFrame()


def read_file(all_levels, nltefilename, modelgridindex, timestep, atomic_number, T_e, T_R, noprint=False):
    """Read NLTE populations from one file, adding in the LTE at T_E and T_R populations."""
    # print(f'Reading {nltefilename}...')
    try:
        dfpop = pd.read_csv(nltefilename, delim_whitespace=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    dfpop.query('(timestep==@timestep) & (Z==@atomic_number)', inplace=True)
    if modelgridindex >= 0:
        dfpop.query('modelgridindex==@modelgridindex', inplace=True)

    k_b = const.k_B.to('eV / K').value
    list_indicies = []
    list_ltepop_T_e = []
    list_ltepop_T_R = []
    list_parity = []
    gspop = {}
    ionlevels = {}
    for index, row in dfpop.iterrows():
        list_indicies.append(index)

        atomic_number = int(row.Z)
        ion_stage = int(row.ion_stage)
        if (atomic_number, ion_stage) not in gspop:
            gspop[(row.Z, row.ion_stage)] = dfpop.query(
                'modelgridindex==@row.modelgridindex and timestep==@row.timestep '
                'and Z==@atomic_number and ion_stage==@ion_stage and level==0').iloc[0]['n_NLTE']

        if (atomic_number, ion_stage) not in ionlevels:
            for _, ion_data in all_levels.iterrows():
                if ion_data.Z == atomic_number and ion_data.ion_stage == ion_stage:
                    ionlevels[(atomic_number, ion_stage)] = ion_data.levels
                    break

        ltepop_T_e = 0.0
        ltepop_T_R = 0.0
        levelnumber = int(row.level)
        gspopthision = gspop[(row.Z, row.ion_stage)]
        if levelnumber == -1:  # superlevel
            levelnumbersl = dfpop.query(
                'modelgridindex==@row.modelgridindex and timestep==@row.timestep '
                'and Z==@atomic_number and ion_stage==@ion_stage').level.max()
            dfpop.loc[index, 'level'] = levelnumbersl + 2
            parity = 0
            if not noprint:
                print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} '
                      f'has a superlevel at level {levelnumbersl}')

            gslevel = ionlevels[(atomic_number, ion_stage)].iloc[0]
            sl_levels = ionlevels[(atomic_number, ion_stage)].iloc[levelnumbersl:]
            ltepop_T_e = gspopthision * sl_levels.eval(
                'g / @gslevel.g * exp(- (energy_ev - @gslevel.energy_ev) / @k_b / @T_e)').sum()
            ltepop_T_R = gspopthision * sl_levels.eval(
                'g / @gslevel.g * exp(- (energy_ev - @gslevel.energy_ev) / @k_b / @T_R)').sum()
        else:
            level = ionlevels[(atomic_number, ion_stage)].iloc[levelnumber]
            gslevel = ionlevels[(atomic_number, ion_stage)].iloc[0]

            exc_energy = level.energy_ev - gslevel.energy_ev

            ltepop_T_e = gspopthision * level.g / gslevel.g * math.exp(- exc_energy / k_b / T_e)
            ltepop_T_R = gspopthision * level.g / gslevel.g * math.exp(- exc_energy / k_b / T_R)

            levelname = level.levelname.split('[')[0]
            parity = 1 if levelname[-1] == 'o' else 0

        list_ltepop_T_e.append(ltepop_T_e)
        list_ltepop_T_R.append(ltepop_T_R)
        list_parity.append(parity)

    dfpop['n_LTE_T_e'] = pd.Series(list_ltepop_T_e, index=list_indicies)
    dfpop['n_LTE_T_R'] = pd.Series(list_ltepop_T_R, index=list_indicies)
    dfpop['parity'] = pd.Series(list_parity, index=list_indicies)

    return dfpop


def read_files(modelpath, adata, atomic_number, T_e, T_R, timestep, modelgridindex=-1, noprint=False):
    """Read in NLTE populations from a model for a particular timestep and grid cell"""
    if modelgridindex > -1:
        mpirank = at.get_mpirankofcell(modelgridindex, modelpath=modelpath)

        nlte_files = list(chain(
            Path(modelpath).rglob(f'nlte_{mpirank:04d}.out'),
            Path(modelpath).rglob(f'nlte_{mpirank:04d}.out.gz')))
    else:
        nlte_files_all = chain(
            Path(modelpath).rglob('nlte_????.out'),
            Path(modelpath).rglob('nlte_????.out.gz'))

        def filerank(estfile):
            return int(re.findall('[0-9]+', os.path.basename(estfile))[-1])

        npts_model = at.get_npts_model(modelpath)
        nlte_files = [x for x in nlte_files_all if filerank(x) < npts_model]
        print(f'Reading {len(nlte_files)} NLTE population files...')

    dfpop = pd.DataFrame()

    if not nlte_files:
        print("No NLTE files found.")
        return dfpop

    for nltefilepath in sorted(nlte_files):
        if modelgridindex > -1:
            print(f'Reading {nltefilepath}...')
        dfpop_thisfile = read_file(
            adata, nltefilepath, modelgridindex, timestep, atomic_number, T_e, T_R, noprint=noprint)

        # found our data!
        if not dfpop_thisfile.empty:
            if modelgridindex >= 0:
                return dfpop_thisfile
            else:
                if dfpop.empty:
                    dfpop = dfpop_thisfile.copy()
                else:
                    dfpop = dfpop.append(dfpop_thisfile.copy(), ignore_index=True)

    return dfpop


def make_plot(modelpath, modeldata, estimators, dfpop, atomic_number, ionstages_permitted, T_e, T_R,
              modelgridindex, timestep, args):
    # top_ion = 9999
    max_ion_stage = dfpop.ion_stage.max()

    if len(dfpop.query('ion_stage == @max_ion_stage')) == 1:  # single-level ion, so skip it
        max_ion_stage -= 1

    # timearray = at.get_timestep_times_float(modelpath)
    Te = estimators[(timestep, modelgridindex)]['Te']
    nne = estimators[(timestep, modelgridindex)]['nne']

    ion_stage_list = sorted(
        [i for i in dfpop.ion_stage.unique()
         if i <= max_ion_stage and (ionstages_permitted is None or i in ionstages_permitted)])

    fig, axes = plt.subplots(nrows=len(ion_stage_list), ncols=1, sharex=False, figsize=(9, 2.7 * len(ion_stage_list)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if len(ion_stage_list) == 1:
        axes = [axes]

    if dfpop.empty:
        print('Error: No data for selected timestep and element')
        sys.exit()

    for ion, axis in enumerate(axes):
        ion_stage = ion_stage_list[ion]
        dfpopthision = dfpop.query('ion_stage==@ion_stage').copy()
        ionpopulation = dfpopthision['n_NLTE'].sum()
        ionpopulation_fromest = estimators[(timestep, modelgridindex)]['populations'].get((atomic_number, ion_stage), 0.)
        if args.maxlevel >= 0:
            dfpopthision.query('level <= @args.maxlevel', inplace=True)
        print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} has an summed level population of {ionpopulation:.1f}'
              f' (from estimator file ion pop = {ionpopulation_fromest})')

        if args.departuremode:
            # scale to match the ground state populations
            lte_scalefactor = float(dfpopthision['n_NLTE'].iloc[0] / dfpopthision['n_LTE_T_e'].iloc[0])
        else:
            # scale to match the ion population
            lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE_T_e'].sum())

        dfpopthision['n_LTE_T_e_normed'] = dfpopthision['n_LTE_T_e'] * lte_scalefactor

        dfpopthision.eval('departure_coeff = n_NLTE / n_LTE_T_e_normed', inplace=True)

        if not args.departuremode:
            axis.plot(dfpopthision.level.values, dfpopthision['n_LTE_T_e_normed'].values, linewidth=1.5,
                      label=f'LTE T$_e$ = {T_e:.0f} K', linestyle='None', marker='*')

            if not args.hide_lte_tr:
                lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE_T_R'].sum())
                dfpopthision['n_LTE_T_R_normed'] = dfpopthision['n_LTE_T_R'] * lte_scalefactor
                axis.plot(dfpopthision.level.values, dfpopthision['n_LTE_T_R_normed'].values, linewidth=1.5,
                          label=f'LTE T$_R$ = {T_R:.0f} K', linestyle='None', marker='*')

        # comparison to Andeas Floers
        # if atomic_number == 26 and ion_stage in [2, 3]:
        #     floersfilename = 'andreas_level_populations_fe2.txt' if ion_stage == 2 else 'andreas_level_populations_fe3.txt'
        #     floers_levelpops = pd.read_csv(floersfilename, comment='#', delim_whitespace = True)
        #     floers_levelpops.sort_values(by='energypercm', inplace=True)
        #     levelnums = list(range(len(floers_levelpops)))
        #     floers_levelpop_values = floers_levelpops['frac_ionpop'].values * dfpopthision['n_NLTE'].sum()
        #     axis.plot(levelnums, floers_levelpop_values, linewidth=1.5,
        #               label=f'Floers NLTE', linestyle='None', marker='*')

        dfpopthisionoddlevels = dfpopthision.query('parity==1')
        velocity = modeldata['velocity'][modelgridindex]
        if args.departuremode:
            print(dfpopthision[['level', 'departure_coeff']])
            axis.plot(dfpopthision['level'], dfpopthision['departure_coeff'], linewidth=1.5,
                      linestyle='None', marker='x', label=f'ARTIS NLTE', color='C0')
            axis.set_ylabel('Departure coefficient')

            axis.plot(dfpopthisionoddlevels.level.values, dfpopthisionoddlevels.departure_coeff.values, linewidth=2,
                      label='Odd parity', linestyle='None',
                      marker='s', markersize=10, markerfacecolor=(0, 0, 0, 0), markeredgecolor='black')
        else:
            axis.plot(dfpopthision.level, dfpopthision.n_NLTE, linewidth=1.5,
                      label='ARTIS NLTE', linestyle='None', marker='x')

            axis.plot(dfpopthisionoddlevels.level, dfpopthisionoddlevels.n_NLTE, linewidth=2,
                      label='Odd parity', linestyle='None',
                      marker='s', markersize=10, markerfacecolor=(0, 0, 0, 0), markeredgecolor='black')

        subplotlabel = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]}'

        axis.annotate(subplotlabel, xy=(0.75, 0.96), xycoords='axes fraction',
                      horizontalalignment='center', verticalalignment='top', fontsize=12)
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

    for axis in axes:
        axis.set_xlim(xmin=-1)
        # axis.set_xlim(xmin=270,xmax=300)
        # axis.set_ylim(ymin=-0.1,ymax=1.3)
        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
        axis.set_yscale('log')
    axes[-1].set_xlabel(r'Level index')

    modelname = at.get_model_name(modelpath)
    figure_title = (
        f'{modelname} {velocity:.0f} km/s at')

    try:
        time_days = float(at.get_timestep_time(modelpath, timestep))
    except FileNotFoundError:
        time_days = 0
        figure_title += f' timestep {timestep:d}'
    else:
        figure_title += f' {time_days:.0f}d'
    figure_title += f' (Te = {Te:.0f} K, nne = {nne:.1e} ' + r'cm$^{-3}$)'

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=11)

    outputfilename = str(args.outputfile).format(
        elsymbol=at.elsymbols[atomic_number], cell=modelgridindex,
        timestep=timestep, time_days=time_days)
    fig.savefig(str(outputfilename), format='pdf')
    print(f"Saved {outputfilename}")
    plt.close()


def addargs(parser):
    parser.add_argument('elements', nargs='*', default=['Fe'],
                        help='List of elements to plot')

    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', type=int,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Plotted modelgrid cell')

    parser.add_argument('-velocity', '-v', type=float, default=-1,
                        help='Specify cell by velocity')

    parser.add_argument('-exc-temperature', type=float, default=6000.,
                        help='Default if no estimator data')

    parser.add_argument('-ionstages',
                        help='Ion stage range, 1 is neutral, 2 is 1+')

    parser.add_argument('-maxlevel', default=-1,
                        help='Maximum level to plot')

    parser.add_argument('--departuremode', action='store_true',
                        help='Show departure coefficients instead of populations')

    parser.add_argument('--hide-lte-tr', action='store_true',
                        help='Hide LTE populations at T=T_R')

    parser.add_argument('--notitle', action='store_true',
                        help='Suppress the top title from the plot')

    parser.add_argument('-outputfile', '-o', type=Path,
                        default=defaultoutputfile,
                        help='path/filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            description='Plot ARTIS non-LTE corrections.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if args.timedays:
        timestep = at.get_closest_timestep(args.modelpath, args.timedays)
    else:
        timestep = int(args.timestep)

    modelpath = args.modelpath
    time_days = float(at.get_timestep_time(modelpath, timestep))

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    ionstages_permitted = at.parse_range_list(args.ionstages) if args.ionstages else None
    adata = at.get_levels(modelpath)

    modeldata, _ = at.get_modeldata(os.path.join(modelpath, 'model.txt'))

    if args.velocity >= 0.:
        modelgridindex = at.get_closest_cell(modelpath, args.velocity)
    else:
        modelgridindex = args.modelgridindex

    estimators = at.estimators.read_estimators(modelpath, modeldata=modeldata,
                                               timestep=timestep, modelgridindex=modelgridindex)
    print(f'modelgridindex {args.modelgridindex}, timestep {timestep} (t={time_days}d)')
    if estimators:
        if not estimators[(timestep, modelgridindex)]['emptycell']:
            T_e = estimators[(timestep, modelgridindex)]['Te']
            T_R = estimators[(timestep, modelgridindex)]['TR']
            W = estimators[(timestep, modelgridindex)]['W']
            nne = estimators[(timestep, modelgridindex)]['nne']
            print(f'nne = {nne} cm^-3, T_e = {T_e} K, T_R = {T_R} K, W = {W}')
        else:
            print(f'ERROR: cell {args.modelgridindex} is empty. Setting T_e = T_R = {args.exc_temperature} K')
            T_e = args.exc_temperature
            T_R = args.exc_temperature
    else:
        print('WARNING: No estimator data. Setting T_e = T_R =  6000 K')
        T_e = args.exc_temperature
        T_R = args.exc_temperature

    if isinstance(args.elements, str):
        args.elements = [args.elements]

    for el_in in args.elements:
        try:
            atomic_number = int(el_in)
            elsymbol = at.elsymbols[atomic_number]
        except ValueError:
            try:
                elsymbol = el_in
                atomic_number = next(
                    Z for Z, elsymb in enumerate(at.elsymbols) if elsymb.lower() == elsymbol.lower())
            except StopIteration:
                print(f"Could not find element '{elsymbol}'")
                continue

        print(f'Z={atomic_number} {elsymbol}')

        dfpop = read_files(modelpath, adata, atomic_number, T_e, T_R,
                           timestep=timestep, modelgridindex=modelgridindex)

        if dfpop.empty:
            print(f'No NLTE population data for modelgrid cell {args.modelgridindex} timestep {timestep}')
        else:
            make_plot(modelpath, modeldata, estimators, dfpop, atomic_number, ionstages_permitted, T_e, T_R,
                      modelgridindex, timestep, args)


if __name__ == "__main__":
    main()
