#!/usr/bin/env python3
"""Artistools - NLTE population related functions."""
import argparse
import math
import multiprocessing
import os
import re
# import sys
from functools import lru_cache
from functools import partial
from pathlib import Path
from itertools import chain

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import numpy as np
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

        masknotsuperlevel = (
            (dfpop['modelgridindex'] == modelgridindex) & (dfpop['timestep'] == timestep)
            & (dfpop['Z'] == Z) & (dfpop['ion_stage'] == ion_stage) & (dfpop['level'] != -1))

        def f_ltepop(x, T_exc, gsg, gse, ionlevels):
            return (ionlevels.iloc[int(x.level)].g / gsg
                    * math.exp(- (ionlevels.iloc[int(x.level)].energy_ev - gse) / k_b / T_exc))

        for columnname, T_exc in columntemperature_tuples:
            dfpop.loc[masknotsuperlevel, columnname] = dfpop.loc[masknotsuperlevel].apply(
                f_ltepop, args=(T_exc, gs_g, gs_energy, ionlevels), axis=1)

        if not dfpop[masksuperlevel].empty:
            levelnumber_sl = dfpop.query(
                'modelgridindex == @modelgridindex and timestep == @timestep '
                'and Z == @Z and ion_stage == @ion_stage').level.max() + 1

            if maxlevel < 0 or levelnumber_sl <= maxlevel:
                if not noprint:
                    print(f'{at.elsymbols[Z]} {at.roman_numerals[ion_stage]} '
                          f'has a superlevel at level {levelnumber_sl}')

                for columnname, T_exc in columntemperature_tuples:
                    dfpop.loc[masksuperlevel, columnname] = ionlevels.iloc[levelnumber_sl:].eval(
                        'g / @gs_g * exp(- (energy_ev - @gs_energy) / @k_b / @T_exc)').sum()

            dfpop.loc[masksuperlevel, 'level'] = levelnumber_sl + 2

    return dfpop


@at.diskcache(savegzipped=True)
def read_file(nltefilepath):
    """Read NLTE populations from one file."""

    if not nltefilepath.is_file():
        nltefilepath = Path(str(nltefilepath) + '.gz')
        print(nltefilepath)
        if not nltefilepath.is_file():
            # if the first file is not found in the folder, then skip the folder
            print(f'Warning: Could not find {nltefilepath}')
            return pd.DataFrame()

    filesize = Path(nltefilepath).stat().st_size / 1024 / 1024
    print(f'Reading {nltefilepath} ({filesize:.2f} MiB)')

    try:
        dfpop = pd.read_csv(nltefilepath, delim_whitespace=True)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    return dfpop


def read_file_filtered(nltefilepath, strquery=None, dfqueryvars=None):
    dfpopfile = read_file(nltefilepath)

    if strquery:
        dfpopfile.query(strquery, local_dict=dfqueryvars, inplace=True)

    return dfpopfile


@lru_cache(maxsize=2)
@at.diskcache(savegzipped=True, funcversion="2020-07-03.1327", saveonly=False)
def read_files(modelpath, timestep=-1, modelgridindex=-1, dfquery=None, dfqueryvars={}):
    """Read in NLTE populations from a model for a particular timestep and grid cell."""

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=modelgridindex)

    dfpop = pd.DataFrame()

    nltefilepaths = [Path(folderpath, f'nlte_{mpirank:04d}.out')
                     for folderpath in at.get_runfolders(modelpath, timestep=timestep) for mpirank in mpiranklist]

    dfqueryvars['modelgridindex'] = modelgridindex
    dfqueryvars['timestep'] = timestep

    dfquery_full = ''
    if timestep >= 0:
        dfquery_full = 'timestep==@timestep'

    if modelgridindex >= 0:
        if dfquery_full:
            dfquery_full += ' and '
        dfquery_full += 'modelgridindex==@modelgridindex'

    if dfquery:
        if dfquery_full:
            dfquery_full = f'({dfquery_full}) and '
        dfquery_full += f'({dfquery})'

    if at.num_processes > 1:
        with multiprocessing.Pool(processes=at.num_processes) as pool:
            arr_dfnltepop = pool.map(partial(read_file_filtered, strquery=dfquery_full, dfqueryvars=dfqueryvars),
                                     nltefilepaths)
            pool.close()
            pool.join()
            pool.terminate()
    else:
        arr_dfnltepop = [read_file_filtered(f, strquery=dfquery_full, dfqueryvars=dfqueryvars) for f in nltefilepaths]

    dfpop = pd.concat(arr_dfnltepop).copy()

    return dfpop


def annotate_emission_line(ax, y, upperlevel, lowerlevel, label):
    ax.annotate('', xy=(lowerlevel, y), xycoords=('data', 'axes fraction'),
                xytext=(upperlevel, y), textcoords=('data', 'axes fraction'),
                arrowprops=dict(
                    facecolor='black', width=0.1, headwidth=6))

    ax.annotate(label, xy=((upperlevel + lowerlevel) / 2, y), xycoords=('data', 'axes fraction'),
                size=10, va="bottom", ha="center",)


def plot_reference_data(ax, atomic_number, ion_stage, estimators_celltimestep, dfpopthision, args, annotatelines):
    nne, Te, TR, W = [estimators_celltimestep[s] for s in ['nne', 'Te', 'TR', 'W']]
    # comparison to Chianti file
    elsym = at.elsymbols[atomic_number]
    elsymlower = elsym.lower()
    if Path('data', f'{elsymlower}_{ion_stage}-levelmap.txt').exists():
        # ax.set_ylim(bottom=2e-3)
        # ax.set_ylim(top=4)
        levelmapfile = Path('data', f'{elsymlower}_{ion_stage}-levelmap.txt').open('r')
        levelnumofconfigterm = {}
        for line in levelmapfile:
            row = line.split()
            levelnumofconfigterm[(row[0], row[1])] = int(row[2]) - 1

        # ax.set_ylim(bottom=5e-4)
        for depfilepath in sorted(Path('data').rglob(f'chianti_{elsym}_{ion_stage}_*.txt')):
            with depfilepath.open('r') as depfile:
                firstline = depfile.readline()
                file_nne = float(firstline[firstline.find('ne = ') + 5:].split(',')[0])
                file_Te = float(firstline[firstline.find('Te = ') + 5:].split(',')[0])
                file_TR = float(firstline[firstline.find('TR = ') + 5:].split(',')[0])
                file_W = float(firstline[firstline.find('W = ') + 5:].split(',')[0])
                # print(depfilepath, file_nne, nne, file_Te, Te, file_TR, TR, file_W, W)
                if (math.isclose(file_nne, nne, rel_tol=0.01) and
                        math.isclose(file_Te, Te, abs_tol=10)):
                    if file_W > 0:
                        continue
                        bbstr = ' with dilute blackbody'
                        color = 'C2'
                        marker = '+'
                    else:
                        bbstr = ''
                        color = 'C1'
                        marker = '^'

                    print(f'Plotting reference data from {depfilepath},')
                    print(f'nne = {file_nne} (ARTIS {nne}) cm^-3, Te = {file_Te} (ARTIS {Te}) K, '
                          f'TR = {file_TR} (ARTIS {TR}) K, W = {file_W} (ARTIS {W})')
                    levelnums = []
                    depcoeffs = []
                    firstdep = -1
                    for line in depfile:
                        row = line.split()
                        try:
                            levelnum = levelnumofconfigterm[(row[1], row[2])]
                            if levelnum in dfpopthision['level'].values:
                                levelnums.append(levelnum)
                                if firstdep < 0:
                                    firstdep = float(row[0])
                                depcoeffs.append(float(row[0]) / firstdep)
                        except (KeyError, IndexError, ValueError):
                            pass
                    ionstr = at.get_ionstring(atomic_number, ion_stage, spectral=False)
                    ax.plot(levelnums, depcoeffs, linewidth=1.5, color=color,
                            label=f'{ionstr} CHIANTI NLTE{bbstr}', linestyle='None', marker=marker, zorder=-1)

        if annotatelines and atomic_number == 28 and ion_stage == 2:
            annotate_emission_line(ax=ax, y=0.04, upperlevel=6, lowerlevel=0, label=r'7378$~\mathrm{{\AA}}$')
            annotate_emission_line(ax=ax, y=0.15, upperlevel=6, lowerlevel=2, label=r'1.939 $\mu$m')
            annotate_emission_line(ax=ax, y=0.26, upperlevel=7, lowerlevel=1, label=r'7412$~\mathrm{{\AA}}$')


def make_ionsubplot(ax, modelpath, atomic_number, ion_stage, dfpop, ion_data, estimators,
                    T_e, T_R, modelgridindex, timestep, args, lastsubplot):
    """Plot the level populations the specified ion, cell, and timestep."""
    ionstr = at.get_ionstring(atomic_number, ion_stage, spectral=False)

    dfpopthision = dfpop.query(
        'modelgridindex == @modelgridindex and timestep == @timestep '
        'and Z == @atomic_number and ion_stage == @ion_stage', inplace=False).copy()

    lte_columns = [('n_LTE_T_e', T_e)]
    if not args.hide_lte_tr:
        lte_columns.append(('n_LTE_T_R', T_R))

    dfpopthision = add_lte_pops(modelpath, dfpopthision, lte_columns, noprint=False, maxlevel=args.maxlevel)

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

        if not lastsubplot:
            ax.set_xticklabels('' for _ in configtexlist)
        else:
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

    pd.set_option('display.max_columns', 150)
    if len(dfpopthision) < 30:
        print(dfpopthision[
            ['Z', 'ion_stage', 'level', 'config', 'departure_coeff', 'texname']].to_string(index=False))

    if not ion_data.transitions.empty:
        dftrans = ion_data.transitions.query('upper <= @maxlevel',
                                             local_dict={'maxlevel': max(dfpopthision.level)}).copy()

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

    ax.set_yscale('log')

    if args.departuremode:
        ax.set_ylabel('Departure coefficient')

        ycolumnname = 'departure_coeff'
    else:
        ax.set_ylabel(r'Population density (cm$^{-3}$)')

        ycolumnname = 'n_NLTE'

        ax.plot(dfpopthision['level'], dfpopthision['n_LTE_T_e_normed'], linewidth=1.5,
                label=f'{ionstr} LTE T$_e$ = {T_e:.0f} K', linestyle='None', marker='*')

        if not args.hide_lte_tr:
            lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE_T_R'].sum())
            dfpopthision.eval('n_LTE_T_R_normed = n_LTE_T_R * @lte_scalefactor', inplace=True)
            ax.plot(dfpopthision['level'], dfpopthision['n_LTE_T_R_normed'], linewidth=1.5,
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

    ax.plot(dfpopthision['level'], dfpopthision[ycolumnname], linewidth=1.5,
            linestyle='None', marker='x', label=f'{ionstr} ARTIS NLTE', color='black')

    dfpopthisionoddlevels = dfpopthision.query('parity==1')
    if not dfpopthisionoddlevels.level.empty:
        ax.plot(dfpopthisionoddlevels['level'], dfpopthisionoddlevels[ycolumnname], linewidth=2,
                label='Odd parity', linestyle='None',
                marker='s', markersize=10, markerfacecolor=(0, 0, 0, 0), markeredgecolor='black')

    if args.plotrefdata:
        plot_reference_data(
            ax, atomic_number, ion_stage, estimators[(timestep, modelgridindex)],
            dfpopthision, args, annotatelines=lastsubplot)


def make_plot(modelpath, atomic_number, ionstages_displayed, mgilist, timestep, args):
    """Plot level populations for chosens ions of an element in a cell and timestep of an ARTIS model."""
    modelname = at.get_model_name(modelpath)
    adata = at.get_levels(modelpath, get_transitions=args.gettransitions)

    time_days = float(at.get_timestep_time(modelpath, timestep))
    modelname = at.get_model_name(modelpath)

    dfpop = read_files(modelpath, timestep=timestep, modelgridindex=mgilist[0]).copy()

    if dfpop.empty:
        print(f'No NLTE population data for modelgrid cell {mgilist[0]} timestep {timestep}')
        return

    dfpop.query('Z == @atomic_number', inplace=True)

    # top_ion = 9999
    max_ion_stage = dfpop.ion_stage.max()

    if len(dfpop.query('ion_stage == @max_ion_stage')) == 1:  # single-level ion, so skip it
        max_ion_stage -= 1

    ion_stage_list = sorted(
        [i for i in dfpop.ion_stage.unique()
         if i <= max_ion_stage and (ionstages_displayed is None or i in ionstages_displayed)])

    subplotheight = 2.4 / 6 if args.x == 'config' else 1.8 / 6

    nrows = len(ion_stage_list) * len(mgilist)
    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=False,
                             figsize=(args.figscale * at.figwidth,
                                      args.figscale * at.figwidth * subplotheight * nrows),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if nrows == 1:
        axes = [axes]

    prev_ion_stage = -1
    for mgilistindex, modelgridindex in enumerate(mgilist):
        mgifirstaxindex = mgilistindex
        mgilastaxindex = mgilistindex + len(ion_stage_list) - 1

        estimators = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)
        elsymbol = at.elsymbols[atomic_number]
        print(f'Plotting NLTE pops for {modelname} modelgridindex {modelgridindex}, '
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
                print(f'ERROR: cell {modelgridindex} is empty. Setting T_e = T_R = {args.exc_temperature} K')
                T_e = args.exc_temperature
                T_R = args.exc_temperature
        else:
            print('WARNING: No estimator data. Setting T_e = T_R =  6000 K')
            T_e = args.exc_temperature
            T_R = args.exc_temperature

        dfpop = read_files(modelpath, timestep=timestep, modelgridindex=modelgridindex).copy()

        if dfpop.empty:
            print(f'No NLTE population data for modelgrid cell {modelgridindex} timestep {timestep}')
            return

        dfpop.query('Z == @atomic_number', inplace=True)

        # top_ion = 9999
        max_ion_stage = dfpop.ion_stage.max()

        if len(dfpop.query('ion_stage == @max_ion_stage')) == 1:  # single-level ion, so skip it
            max_ion_stage -= 1

        # timearray = at.get_timestep_times_float(modelpath)
        nne = estimators[(timestep, modelgridindex)]['nne']
        W = estimators[(timestep, modelgridindex)]['W']

        subplot_title = f'{modelname}'
        if len(modelname) > 10:
            subplot_title += '\n'
        velocity = at.get_modeldata(modelpath)[0]['velocity_outer'][modelgridindex]
        subplot_title += f' {velocity:.0f} km/s at'

        try:
            time_days = float(at.get_timestep_time(modelpath, timestep))
        except FileNotFoundError:
            time_days = 0
            subplot_title += f' timestep {timestep:d}'
        else:
            subplot_title += f' {time_days:.0f}d'
        subplot_title += f' (Te={T_e:.0f} K, nne={nne:.1e} ' + r'cm$^{-3}$, T$_R$=' + f'{T_R:.0f} K, W={W:.1e})'

        if not args.notitle:
            axes[mgifirstaxindex].set_title(subplot_title, fontsize=10)

        for ax, ion_stage in zip(axes[mgifirstaxindex:mgilastaxindex + 1], ion_stage_list):
            ion_data = adata.query('Z == @atomic_number and ion_stage == @ion_stage').iloc[0]
            lastsubplot = modelgridindex == mgilist[-1] and ion_stage == ion_stage_list[-1]
            make_ionsubplot(ax, modelpath, atomic_number, ion_stage, dfpop, ion_data, estimators,
                            T_e, T_R, modelgridindex, timestep, args, lastsubplot=lastsubplot)

            # ax.annotate(ionstr, xy=(0.95, 0.96), xycoords='axes fraction',
            #             horizontalalignment='right', verticalalignment='top', fontsize=12)
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

            ax.set_xlim(left=-1)

            if not args.nolegend and prev_ion_stage != ion_stage:
                ax.legend(
                    loc='best', handlelength=1, frameon=True, numpoints=1, edgecolor='0.93', facecolor='0.93')

            prev_ion_stage = ion_stage

    if args.x == 'index':
        axes[-1].set_xlabel(r'Level index')

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
        '-modelpath', default=Path(),  type=Path,
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
        '-modelgridindex', '-cell', nargs='?', default=[],
        help='Plotted modelgrid cell(s)')

    cellgroup.add_argument(
        '-velocity', '-v', nargs='?', default=[],
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
        '-maxlevel', default=-1, type=int,
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
        '--nolegend', action='store_true',
        help='Suppress the legend from the plot')

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

    modelpath = args.modelpath

    if args.timedays:
        timestep = at.get_timestep_of_timedays(modelpath, args.timedays)
    else:
        timestep = int(args.timestep)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    ionstages_permitted = at.parse_range_list(args.ionstages) if args.ionstages else None

    if isinstance(args.modelgridindex, str):
        args.modelgridindex = [args.modelgridindex]

    if isinstance(args.elements, str):
        args.elements = [args.elements]

    if isinstance(args.velocity, float):
        args.velocity = [args.velocity]

    mgilist = []
    for mgi in args.modelgridindex:
        mgilist.append(int(mgi))

    for vel in args.velocity:
        mgilist.append(at.get_mgi_of_velocity_kms(modelpath, vel))

    if not mgilist:
        mgilist.append(0)

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
                  mgilist, timestep, args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
