#!/usr/bin/env python3
"""Artistools - NLTE population related functions."""
import argparse
import math
import os
import re
# import sys
from functools import lru_cache
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


def texifyterm(strterm):
    """Replace a term string with TeX notation equivalent."""
    strtermtex = ''
    passed_term_Lchar = False

    for termpiece in re.split('([_A-Za-z])', strterm):
        if re.match('[0-9]', termpiece) is not None and not passed_term_Lchar:
            # 2S + 1 number
            strtermtex += r'$^{' + termpiece + r'}$'
        elif re.match('[A-Z]', termpiece) is not None:
            # L character - SPDFGH...
            strtermtex += termpiece
            passed_term_Lchar = True
        elif re.match('[eo]', termpiece) is not None and passed_term_Lchar:
            # odd flag, but don't want to confuse it with the energy index (e.g. o4Fo[2])
            strtermtex += r'$^{\rm ' + termpiece + r'}$'
        elif re.match(r'[0-9]?.*\]', termpiece) is not None:
            # J value
            strtermtex += termpiece.split('[')[0] + r'$_{' + termpiece.lstrip('0123456789').strip('[]') + r'}$'
        elif re.match('[0-9]', termpiece) is not None and passed_term_Lchar:
            # extra number after S char
            strtermtex += termpiece

    strtermtex = strtermtex.replace('$$', '')
    return strtermtex


def texifyconfiguration(levelname):
    """Replace a level configuration with the formatted LaTeX equivalent."""
    # the underscore gets confused with LaTeX subscript operator, so switch it to the hash symbol
    strout = '#'.join(levelname.split('_')[:-1]) + '#'
    for strorbitalocc in re.findall(r'[0-9][a-z][0-9]?[#(]', strout):
        n, lchar, occ = re.split('([a-z])', strorbitalocc)
        lastchar = '(' if occ.endswith('(') else '#'
        occ = occ.rstrip('#(')
        strorbitalocctex = n + lchar + (r'$^{' + occ + r'}$' if occ else '') + lastchar
        strout = strout.replace(strorbitalocc, strorbitalocctex)

    for parentterm in re.findall(r'\([0-9][A-Z][^)]?\)', strout):
        parentermtex = f'({texifyterm(parentterm.strip("()"))})'
        strout = strout.replace(parentterm, parentermtex)
    strterm = levelname.split('_')[-1]
    strout += ' ' + texifyterm(strterm)

    strout = strout.replace('#', '')
    strout = strout.replace('$$', '')
    # print(f"Replacing levelname '{levelname}' with '{strout}'")
    return strout


@lru_cache(maxsize=32)
def get_nltepops(modelpath, timestep, modelgridindex):
    """Read in NLTE populations from a model for a particular timestep and grid cell."""
    mpirank = at.get_mpirankofcell(modelgridindex, modelpath=modelpath)

    nlte_files = list(chain(
        Path(modelpath).rglob(f'nlte_{mpirank:04d}.out'),
        Path(modelpath).rglob(f'nlte_{mpirank:04d}.out.gz')))

    if not nlte_files:
        print("No NLTE files found.")
        return False
    else:
        print(f'Loading {len(nlte_files)} NLTE files')
        for nltefilepath in nlte_files:
            # print(f'Reading {nltefilepath}')
            dfpop = pd.read_csv(nltefilepath, delim_whitespace=True)

            dfpop.query('(modelgridindex==@modelgridindex) & (timestep==@timestep)', inplace=True)
            if not dfpop.empty:
                return dfpop

    return pd.DataFrame()


def add_lte_pops(modelpath, dfpop, columntemperature_tuples, noprint=False, maxlevel=-1):
    """Add columns to dfpop with LTE populations.

    columntemperature_tuples is a sequence of tuples of column name and temperature, e.g., ('mycolumn', 3000)
    """
    k_b = const.k_B.to('eV / K').value

    for _, row in dfpop.drop_duplicates(['modelgridindex', 'timestep', 'Z', 'ion_stage']).iterrows():
        modelgridindex = int(row.modelgridindex)
        timestep = int(row.timestep)
        Z = int(row.Z)
        ion_stage = int(row.ion_stage)

        ionlevels = at.get_levels(modelpath).query('Z == @Z and ion_stage == @ion_stage').iloc[0].levels

        gs_g = ionlevels.iloc[0].g
        gs_energy = ionlevels.iloc[0].energy_ev

        gs_pop = dfpop.query(
            'modelgridindex == @modelgridindex and timestep == @timestep '
            'and Z == @Z and ion_stage == @ion_stage and level == 0').iloc[0]['n_NLTE']

        masksuperlevel = (
            (dfpop['modelgridindex'] == modelgridindex) & (dfpop['timestep'] == timestep)
            & (dfpop['Z'] == Z) & (dfpop['ion_stage'] == ion_stage) & (dfpop['level'] == -1))

        if not dfpop[masksuperlevel].empty:
            levelnumber_sl = dfpop.query(
                'modelgridindex == @modelgridindex and timestep == @timestep '
                'and Z == @Z and ion_stage == @ion_stage').level.max()

            if maxlevel >= 0 and levelnumber_sl <= maxlevel:
                if not noprint:
                    print(f'{at.elsymbols[Z]} {at.roman_numerals[ion_stage]} '
                          f'has a superlevel at level {levelnumber_sl}')

                for columnname, T_exc in columntemperature_tuples:
                    dfpop.loc[masksuperlevel, columnname] = gs_pop * ionlevels.iloc[levelnumber_sl:].eval(
                        'g / @gs_g * exp(- (energy_ev - @gs_energy) / @k_b / @T_exc)').sum()

            dfpop.loc[masksuperlevel, 'level'] = levelnumber_sl + 2

        masknotsuperlevel = (
            (dfpop['modelgridindex'] == modelgridindex) & (dfpop['timestep'] == timestep)
            & (dfpop['Z'] == Z) & (dfpop['ion_stage'] == ion_stage) & (dfpop['level'] != -1))

        def f_ltepop(x, T_exc, gs_pop, gsg, gse, ionlevels):
            return (ionlevels.iloc[int(x.level)].g / gsg
                    * math.exp(- (ionlevels.iloc[int(x.level)].energy_ev - gse) / k_b / T_exc))

        for columnname, T_exc in columntemperature_tuples:
            dfpop.loc[masknotsuperlevel, columnname] = dfpop.loc[masknotsuperlevel].apply(
                f_ltepop, args=(T_exc, gs_pop, gs_g, gs_energy, ionlevels), axis=1)


@lru_cache(maxsize=8)
def read_file(nltefilename, modelpath, modelgridindex, timestep):
    """Read NLTE populations from one file."""
    if modelgridindex > -1:
        filesize = Path(nltefilename).stat().st_size / 1024 / 1024
        print(f'Reading {nltefilename} ({filesize:.3f} MiB)')

    try:
        dfpop = pd.read_csv(nltefilename, delim_whitespace=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    dfpop.query('timestep==@timestep', inplace=True)
    if modelgridindex >= 0:
        dfpop.query('modelgridindex==@modelgridindex', inplace=True)

    return dfpop


def read_files(modelpath, timestep, modelgridindex=-1, noprint=False):
    """Read in NLTE populations from a model for a particular timestep and grid cell."""
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
            """Get the MPI process rank of an estimator file name."""
            return int(re.findall('[0-9]+', os.path.basename(estfile))[-1])

        npts_model = at.get_npts_model(modelpath)
        nlte_files = [x for x in nlte_files_all if filerank(x) < npts_model]
        print(f'Reading {len(nlte_files)} NLTE population files...')

    dfpop = pd.DataFrame()

    if not nlte_files:
        print("No NLTE files found.")
        return dfpop

    for nltefilepath in sorted(nlte_files):

        dfpop_thisfile = read_file(
            nltefilepath, modelpath, modelgridindex, timestep).copy()

        # found our data!
        if not dfpop_thisfile.empty:
            if modelgridindex >= 0:
                return dfpop_thisfile
            else:
                if dfpop.empty:
                    dfpop = dfpop_thisfile
                else:
                    dfpop = dfpop.append(dfpop_thisfile, ignore_index=True)

    return dfpop


def plot_reference_data(ax, atomic_number, ion_stage, T_e, nne, dfpopthision):
    ionstr = at.get_ionstring(atomic_number, ion_stage)

    # comparison to Chianti file from Stuart Sim
    if atomic_number == 28 and ion_stage == 2 and Path('data', 'ni_2-levelmap.txt').exists():
        levelmapfile = Path('data', 'ni_2-levelmap.txt').open('r')
        levelnumofconfigterm = {}
        for line in levelmapfile:
            row = line.split()
            levelnumofconfigterm[(row[0], row[1])] = int(row[2]) - 1

        # ax.set_ylim(ymin=5e-4)
        for depfilepath in Path('data', 'chianti-tests-Stuart').rglob('ni_2*.txt'):
            with depfilepath.open('r') as depfile:
                firstline = depfile.readline()
                file_nne = float(firstline[firstline.find('ne = ') + 5:].split(',')[0])
                file_Te = float(firstline[firstline.find('Te = ') + 5:].split()[0])
                if (firstline.startswith(f'{ionstr} departure coefficients for ') and
                        math.isclose(file_nne, nne, rel_tol=0.1) and
                        math.isclose(file_Te, T_e, abs_tol=1000)):
                    print(f'Plotting reference data from {depfilepath},')
                    print(f'nne = {file_nne} (ARTIS {nne}) cm^-3, Te = {file_Te} (ARTIS {T_e}) K')
                    levelnums = []
                    depcoeffs = []
                    firstdep = -1
                    for line in depfile:
                        row = line.split()
                        try:
                            levelnum = levelnumofconfigterm[(row[1], row[2])]
                            if levelnum in dfpopthision.level.values:
                                levelnums.append(levelnum)
                                if firstdep < 0:
                                    firstdep = float(row[0])
                                depcoeffs.append(float(row[0]) / firstdep)
                        except (IndexError, ValueError):
                            pass

                    ax.plot(levelnums, depcoeffs, linewidth=1.5, color='C1',
                            label=f'{ionstr} Chianti NLTE', linestyle='None', marker='*', zorder=-1)

    elif atomic_number == 26 and ion_stage == 2 and Path('data', 'chianti-tests-Stuart', 'fe_2-test.txt').exists():
        levelnums = []
        depcoeffs = []
        firstdep = -1
        levelnumofconfigterm = {}
        compfile = Path('data', 'chianti-tests-Stuart', 'fe_2-test.txt').open('r')
        for line in compfile:
            row = line.split()
            try:
                levelnum = int(row[1]) - 1
                if levelnum in dfpopthision.level.values:
                    levelnums.append(levelnum)
                    if firstdep < 0:
                        firstdep = float(row[0])
                    depcoeffs.append(float(row[0]) / firstdep)
                    levelnumofconfigterm[(row[2], row[3])] = levelnum
            except (IndexError, ValueError):
                pass
        ax.plot(levelnums, depcoeffs, linewidth=1.5, color='C1',
                label=f'{ionstr} Chianti NLTE', linestyle='None', marker='*', zorder=-1)

        levelnums = []
        depcoeffs = []
        firstdep = -1
        compfile = Path('data', 'chianti-tests-Stuart', 'fe_2-test-withDiluteRad.txt').open('r')
        for line in compfile:
            row = line.split()
            try:
                levelnum = levelnumofconfigterm[(row[1], row[2])]
                if levelnum in dfpopthision.level.values:
                    levelnums.append(levelnum)
                    if firstdep < 0:
                        firstdep = float(row[0])
                    depcoeffs.append(float(row[0]) / firstdep)
            except (KeyError, IndexError, ValueError):
                pass
        ax.plot(levelnums, depcoeffs, linewidth=1.5, color='C2',
                label=f'{ionstr} Chianti NLTE DiluteBB', linestyle='None', marker='*', zorder=-1)

        levelnums = []
        depcoeffs = []
        firstdep = -1
        compfile = Path('data', 'chianti-tests-Stuart', 'fe_2-test-reducedDensity.txt').open('r')
        for line in compfile:
            row = line.split()
            try:
                levelnum = levelnumofconfigterm[(row[1], row[2])]
                if levelnum in dfpopthision.level.values:
                    levelnums.append(levelnum)
                    if firstdep < 0:
                        firstdep = float(row[0])
                    depcoeffs.append(float(row[0]) / firstdep)
            except (KeyError, IndexError, ValueError):
                pass
        ax.plot(levelnums, depcoeffs, linewidth=1.5, color='C3',
                label=f'{ionstr} Chianti NLTE DiluteBB lowD', linestyle='None', marker='*', zorder=-1)

        ax.set_ylim(ymin=1e-2)
        # axis.set_xlim(xmax=60)


def make_ionsubplot(ax, modelpath, atomic_number, ion_stage, dfpop, ion_data, estimators,
                    T_e, T_R, modelgridindex, timestep, args):
    """Plot the level populations the specified ion, cell, and timestep."""
    ionstr = at.get_ionstring(atomic_number, ion_stage)

    dfpopthision = dfpop.query(
        'modelgridindex == @modelgridindex and timestep == @timestep '
        'and Z == @atomic_number and ion_stage == @ion_stage', inplace=False).copy()

    lte_columns = [('n_LTE_T_e', T_e)]
    if not args.hide_lte_tr:
        lte_columns.append(('n_LTE_T_R', T_R))

    add_lte_pops(modelpath, dfpopthision, lte_columns, args.maxlevel)

    if args.maxlevel >= 0:
        dfpopthision.query('level <= @args.maxlevel', inplace=True)

    ionpopulation = dfpopthision['n_NLTE'].sum()
    ionpopulation_fromest = estimators[(timestep, modelgridindex)][
        'populations'].get((atomic_number, ion_stage), 0.)

    dfpopthision['parity'] = [
        1 if (row.level != -1 and
              ion_data.levels.iloc[
                  int(row.level)].levelname.split('[')[0][-1] == "o")
        else 0 for _, row in dfpopthision.iterrows()]

    configlist = ion_data.levels.iloc[:max(dfpopthision.level) + 1].levelname

    configtexlist = [texifyconfiguration(configlist[0])]
    for i in range(1, len(configlist)):
        prevconfignoterm = configlist[i - 1].rsplit('_', maxsplit=1)[0]
        confignoterm = configlist[i].rsplit('_', maxsplit=1)[0]
        if confignoterm == prevconfignoterm:
            configtexlist.append('" ' + texifyterm(configlist[i].rsplit('_', maxsplit=1)[1]))
        else:
            configtexlist.append(texifyconfiguration(configlist[i]))

    dfpopthision['config'] = [configlist[level] for level in dfpopthision.level]
    dfpopthision['texname'] = [configtexlist[level] for level in dfpopthision.level]

    if args.x == 'config':
        # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=100))
        ax.set_xticks(ion_data.levels.iloc[:max(dfpopthision.level) + 1].index)

        ax.set_xticklabels(
            configtexlist,
            # fontsize=8,
            rotation=60,
            horizontalalignment='right',
            rotation_mode='anchor')

    print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} has a summed '
          f'level population of {ionpopulation:.1f} (from estimator file ion pop = {ionpopulation_fromest})')

    if args.departuremode:
        # scale to match the ground state populations
        lte_scalefactor = float(dfpopthision['n_NLTE'].iloc[0] / dfpopthision['n_LTE_T_e'].iloc[0])
    else:
        # scale to match the ion population
        lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE_T_e'].sum())

    dfpopthision.eval('n_LTE_T_e_normed = n_LTE_T_e * @x',
                      local_dict={'x': lte_scalefactor}, inplace=True)

    dfpopthision.eval('departure_coeff = n_NLTE / n_LTE_T_e_normed', inplace=True)

    if not args.departuremode:
        ax.plot(dfpopthision.level.values, dfpopthision['n_LTE_T_e_normed'].values, linewidth=1.5,
                label=f'{ionstr} LTE T$_e$ = {T_e:.0f} K', linestyle='None', marker='*')

        if not args.hide_lte_tr:
            lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE_T_R'].sum())
            dfpopthision.eval('n_LTE_T_R_normed = n_LTE_T_R * @lte_scalefactor', inplace=True)
            ax.plot(dfpopthision.level.values, dfpopthision['n_LTE_T_R_normed'].values, linewidth=1.5,
                    label=f'{ionstr} LTE T$_R$ = {T_R:.0f} K', linestyle='None', marker='*')

    # comparison to Andeas Floers
    # if atomic_number == 26 and ion_stage in [2, 3]:
    #     floersfilename = (
    #         'andreas_level_populations_fe2.txt' if ion_stage == 2 else 'andreas_level_populations_fe3.txt')
    #     floers_levelpops = pd.read_csv(floersfilename, comment='#', delim_whitespace = True)
    #     floers_levelpops.sort_values(by='energypercm', inplace=True)
    #     levelnums = list(range(len(floers_levelpops)))
    #     floers_levelpop_values = floers_levelpops['frac_ionpop'].values * dfpopthision['n_NLTE'].sum()
    #     axis.plot(levelnums, floers_levelpop_values, linewidth=1.5,
    #               label=f'Floers NLTE', linestyle='None', marker='*')

    pd.set_option('display.max_columns', 150)
    if len(dfpopthision) < 30:
        print(dfpopthision[
            ['Z', 'ion_stage', 'level', 'config', 'departure_coeff', 'texname']].to_string(index=False))

    if not ion_data.transitions.empty:
        maxlevel = max(dfpopthision.level)
        dftrans = ion_data.transitions.query('upper <= @maxlevel').copy()
        dftrans['energy_trans'] = [(
            ion_data.levels.iloc[int(trans.upper)].energy_ev - ion_data.levels.iloc[int(trans.lower)].energy_ev)
            for _, trans in dftrans.iterrows()]

        dftrans['emissionstrength'] = [
            dfpopthision.query('level == @trans.upper').iloc[0].n_NLTE * trans.A * trans.energy_trans
            for _, trans in dftrans.iterrows()]

        dftrans['wavelength'] = [
            round((const.h * const.c).to('eV angstrom').value / trans.energy_trans)
            for _, trans in dftrans.iterrows()]

        dftrans.sort_values("emissionstrength", ascending=False, inplace=True)
        print("\nTop radiative decays")
        print(dftrans[:10].to_string(index=False))

    dfpopthisionoddlevels = dfpopthision.query('parity==1')
    if args.departuremode:
        ax.plot(dfpopthision['level'], dfpopthision['departure_coeff'], linewidth=1.5,
                linestyle='None', marker='x', label=f'{ionstr} ARTIS NLTE', color='C0')
        ax.set_ylabel('Departure coefficient')

        if not dfpopthisionoddlevels.level.empty:
            ax.plot(dfpopthisionoddlevels.level.values, dfpopthisionoddlevels.departure_coeff.values, linewidth=2,
                    label='Odd parity', linestyle='None',
                    marker='s', markersize=10, markerfacecolor=(0, 0, 0, 0), markeredgecolor='black')
    else:
        ax.plot(dfpopthision.level, dfpopthision.n_NLTE, linewidth=1.5,
                label=f'{ionstr} ARTIS NLTE', linestyle='None', marker='x')

        if not dfpopthisionoddlevels.level.empty:
            ax.plot(dfpopthisionoddlevels.level, dfpopthisionoddlevels.n_NLTE, linewidth=2,
                    label='Odd parity', linestyle='None',
                    marker='s', markersize=10, markerfacecolor=(0, 0, 0, 0), markeredgecolor='black')

    if args.plotrefdata:
        nne = estimators[(timestep, modelgridindex)]['nne']
        plot_reference_data(ax, atomic_number, ion_stage, T_e, nne, dfpopthision)


def make_plot(modelpath, atomic_number, ionstages_permitted, modelgridindex, timestep, args):
    """Plot level populations for chosens ions of an element in a cell and timestep of an ARTIS model."""
    adata = at.get_levels(modelpath, get_transitions=args.gettransitions)

    estimators = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)
    time_days = float(at.get_timestep_time(modelpath, timestep))
    modelname = at.get_model_name(modelpath)

    elsymbol = at.elsymbols[atomic_number]
    print(f'Plotting NLTE pops for {modelname} modelgridindex {args.modelgridindex}, '
          f'timestep {timestep} (t={time_days}d)')
    print(f'Z={atomic_number} {elsymbol}')

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

    dfpop = read_files(modelpath, timestep=timestep, modelgridindex=modelgridindex).query('Z == @atomic_number')

    if dfpop.empty:
        print(f'No NLTE population data for modelgrid cell {args.modelgridindex} timestep {timestep}')
        return

    # top_ion = 9999
    max_ion_stage = dfpop.ion_stage.max()

    if len(dfpop.query('ion_stage == @max_ion_stage')) == 1:  # single-level ion, so skip it
        max_ion_stage -= 1

    # timearray = at.get_timestep_times_float(modelpath)
    nne = estimators[(timestep, modelgridindex)]['nne']
    W = estimators[(timestep, modelgridindex)]['W']

    ion_stage_list = sorted(
        [i for i in dfpop.ion_stage.unique()
         if i <= max_ion_stage and (ionstages_permitted is None or i in ionstages_permitted)])

    subplotheight = 2.4 / 6 if args.x == 'config' else 1.8 / 6

    fig, axes = plt.subplots(nrows=len(ion_stage_list), ncols=1, sharex=False,
                             figsize=(args.figscale * at.figwidth,
                                      args.figscale * at.figwidth * subplotheight * len(ion_stage_list)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if len(ion_stage_list) == 1:
        axes = [axes]

    for index, ax in enumerate(axes):
        ion_stage = ion_stage_list[index]
        ion_data = adata.query('Z == @atomic_number and ion_stage == @ion_stage').iloc[0]
        make_ionsubplot(ax, modelpath, atomic_number, ion_stage, dfpop, ion_data, estimators,
                        T_e, T_R, modelgridindex, timestep, args)

        # ax.annotate(ionstr, xy=(0.95, 0.96), xycoords='axes fraction',
        #             horizontalalignment='right', verticalalignment='top', fontsize=12)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

    for ax in axes:
        ax.set_xlim(xmin=-1)
        # ax.set_xlim(xmin=270,xmax=300)
        # ax.set_ylim(ymin=-0.1,ymax=1.3)
        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
        ax.set_yscale('log')

    if args.x == 'index':
        axes[-1].set_xlabel(r'Level index')

    modelname = at.get_model_name(modelpath)
    figure_title = f'{modelname}'
    if len(modelname) > 10:
        figure_title += '\n'
    velocity = at.get_modeldata(modelpath)[0]['velocity'][modelgridindex]
    figure_title += f' {velocity:.0f} km/s at'

    try:
        time_days = float(at.get_timestep_time(modelpath, timestep))
    except FileNotFoundError:
        time_days = 0
        figure_title += f' timestep {timestep:d}'
    else:
        figure_title += f' {time_days:.0f}d'
    figure_title += f' (Te={T_e:.0f} K, nne={nne:.1e} ' + r'cm$^{-3}$, T$_R$=' + f'{T_R:.0f} K, W={W:.1e})'

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=10)

    outputfilename = str(args.outputfile).format(
        elsymbol=at.elsymbols[atomic_number], cell=modelgridindex,
        timestep=timestep, time_days=time_days)
    fig.savefig(str(outputfilename), format='pdf')
    print(f"Saved {outputfilename}")
    plt.close()


def addargs(parser):
    parser.add_argument(
        'elements', nargs='*', default=['Fe'],
        help='List of elements to plot')

    parser.add_argument(
        '-modelpath', default='.',
        help='Path to ARTIS folder')

    timegroup = parser.add_mutually_exclusive_group()
    timegroup.add_argument(
        '-timedays', '-time', '-t',
        help='Time in days to plot')

    timegroup.add_argument(
        '-timestep', '-ts', type=int,
        help='Timestep number to plot')

    cellgroup = parser.add_mutually_exclusive_group()
    cellgroup.add_argument(
        '-modelgridindex', '-cell', type=int, default=0,
        help='Plotted modelgrid cell')

    cellgroup.add_argument(
        '-velocity', '-v', type=float, default=-1,
        help='Specify cell by velocity')

    parser.add_argument(
        '-exc-temperature', type=float, default=6000.,
        help='Default if no estimator data')

    parser.add_argument(
        '-x', choices=['index', 'config'], default='index',
        help='Horizontal axis variable')

    parser.add_argument(
        '-ionstages',
        help='Ion stage range, 1 is neutral, 2 is 1+')

    parser.add_argument(
        '-maxlevel', default=-1,
        help='Maximum level to plot')

    parser.add_argument(
        '-figscale', type=float, default=1.6,
        help='Scale factor for plot area. 1.0 is for single-column')

    parser.add_argument(
        '--departuremode', action='store_true',
        help='Show departure coefficients instead of populations')

    parser.add_argument(
        '--gettransitions', action='store_true',
        help='Show the most significant transitions')

    parser.add_argument(
        '--plotrefdata', action='store_true',
        help='Show reference data')

    parser.add_argument(
        '--hide-lte-tr', action='store_true',
        help='Hide LTE populations at T=T_R')

    parser.add_argument(
        '--notitle', action='store_true',
        help='Suppress the top title from the plot')

    parser.add_argument(
        '-outputfile', '-o', type=Path, default=defaultoutputfile,
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

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    ionstages_permitted = at.parse_range_list(args.ionstages) if args.ionstages else None

    if args.velocity >= 0.:
        modelgridindex = at.get_closest_cell(modelpath, args.velocity)
    else:
        modelgridindex = args.modelgridindex

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

        make_plot(modelpath, atomic_number, ionstages_permitted,
                  modelgridindex, timestep, args)


if __name__ == "__main__":
    main()
