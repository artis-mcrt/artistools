#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from astropy import constants as const

import artistools as at
import artistools.estimators

defaultoutputfile = 'plotnlte_{elsymbol}_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf'


def get_nlte_populations(all_levels, nltefilename, modelgridindex, timestep, atomic_number, T_e, T_R):
    dfpop = pd.read_csv(nltefilename, delim_whitespace=True)
    dfpop.query('(modelgridindex==@modelgridindex) & (timestep==@timestep) & (Z==@atomic_number)', inplace=True)

    k_b = const.k_B.to('eV / K').value
    list_indicies = []
    list_ltepop_T_e = []
    list_ltepop_T_R = []
    list_parity = []
    gspop = {}
    for index, row in dfpop.iterrows():
        list_indicies.append(index)

        atomic_number = int(row.Z)
        ion_stage = int(row.ion_stage)
        if (row.Z, row.ion_stage) not in gspop:
            gspop[(row.Z, row.ion_stage)] = dfpop.query(
                'timestep==@timestep and Z==@atomic_number and ion_stage==@ion_stage and level==0').iloc[0].n_NLTE

        ltepop_T_e = 0.0
        ltepop_T_R = 0.0
        levelnumber = int(row.level)
        if levelnumber == -1:  # superlevel
            levelnumber = dfpop.query(
                'timestep==@timestep and Z==@atomic_number and ion_stage==@ion_stage').level.max()
            dfpop.loc[index, 'level'] = levelnumber + 2
            parity = 0
            print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} '
                  f'has a superlevel at level {levelnumber}')
        else:
            for _, ion_data in all_levels.iterrows():
                if ion_data.Z == atomic_number and ion_data.ion_stage == ion_stage:
                    level = ion_data.levels.iloc[levelnumber]
                    gslevel = ion_data.levels.iloc[0]

            gspopthision = gspop[(row.Z, row.ion_stage)]
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


def get_nlte_populations_oldformat(all_levels, nltefilename, modelgridindex, timestep, atomic_number, temperature_exc):
    compositiondata = at.get_composition_data('compositiondata.txt')
    elementdata = compositiondata.query('Z==@atomic_number')

    if len(elementdata) < 1:
        print(f'Error: element Z={atomic_number} not in composition file')
        return None

    skip_block = False
    dfpop = pd.DataFrame().to_sparse()
    with open(nltefilename, 'r') as nltefile:
        for line in nltefile:
            row = line.split()

            if row and row[0] == 'timestep':
                skip_block = int(row[1]) != timestep
                if row[2] == 'modelgridindex' and int(row[3]) != modelgridindex:
                    skip_block = True

            if skip_block:
                continue
            elif len(row) > 2 and row[0] == 'nlte_index' and row[1] != '-':  # level row
                matchedgroundstateline = False
            elif len(row) > 1 and row[1] == '-':  # ground state
                matchedgroundstateline = True
            else:
                continue

            dfrow = parse_nlte_row(row, dfpop, elementdata, all_levels, timestep,
                                   temperature_exc, matchedgroundstateline)

            if dfrow is not None:
                dfpop = dfpop.append(dfrow, ignore_index=True)

    return dfpop


def parse_nlte_row(row, dfpop, elementdata, all_levels, timestep, temperature_exc, matchedgroundstateline):
    """
        Read a line from the NLTE output file and return a Pandas DataFrame
    """
    levelpoptuple = namedtuple(
        'ionpoptuple', 'timestep Z ion_stage level energy_ev parity n_LTE n_NLTE n_LTE_custom')

    elementindex = elementdata.index[0]
    atomic_number = int(elementdata.iloc[0].Z)
    element = int(row[row.index('element') + 1])
    if element != elementindex:
        return None
    ion = int(row[row.index('ion') + 1])
    ion_stage = int(elementdata.iloc[0].lowermost_ionstage) + ion

    if row[row.index('level') + 1] != 'SL':
        levelnumber = int(row[row.index('level') + 1])
        superlevel = False
    else:
        levelnumber = dfpop.query('timestep==@timestep and ion_stage==@ion_stage').level.max() + 3
        print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} has a superlevel at level {levelnumber}')
        superlevel = True

    for _, ion_data in enumerate(all_levels):
        if ion_data.Z == atomic_number and ion_data.ion_stage == ion_stage:
            level = ion_data.levels.iloc[levelnumber]
            gslevel = ion_data.levels.iloc[0]

    ltepop = float(row[row.index('nnlevel_LTE') + 1])

    if matchedgroundstateline:
        nltepop = ltepop_custom = ltepop

        levelname = gslevel.levelname.split('[')[0]
        energy_ev = gslevel.energy_ev
    else:
        nltepop = float(row[row.index('nnlevel_NLTE') + 1])

        k_b = const.k_B.to('eV / K').value
        gspop = dfpop.query('timestep==@timestep and ion_stage==@ion_stage and level==0').iloc[0].n_NLTE
        levelname = level.levelname.split('[')[0]
        energy_ev = (level.energy_ev - gslevel.energy_ev)

        ltepop_custom = gspop * level.g / gslevel.g * math.exp(
            -energy_ev / k_b / temperature_exc)

    parity = 1 if levelname[-1] == 'o' else 0
    if superlevel:
        parity = 0

    newrow = levelpoptuple(timestep=timestep, Z=int(elementdata.iloc[0].Z), ion_stage=ion_stage,
                           level=levelnumber, energy_ev=energy_ev, parity=parity,
                           n_LTE=ltepop, n_NLTE=nltepop, n_LTE_custom=ltepop_custom)

    return pd.DataFrame(data=[newrow], columns=levelpoptuple._fields)


def read_files(modelpath, adata, atomic_number, T_e, T_R, timestep, modelgridindex, oldformat=False):
    nlte_files = (
        glob.glob(os.path.join(modelpath, 'nlte_????.out'), recursive=True) +
        glob.glob(os.path.join(modelpath, '*/nlte_????.out'), recursive=True))

    if not nlte_files:
        print("No NLTE files found.")
        return -1

    print(f'Reading {len(nlte_files)} NLTE population files...')
    for nltefilepath in nlte_files:
        filerank = int(re.search('[0-9]+', os.path.basename(nltefilepath)).group(0))

        if filerank > modelgridindex:
            continue

        if not oldformat:
            dfpop_thisfile = get_nlte_populations(
                adata, nltefilepath, modelgridindex,
                timestep, atomic_number, T_e, T_R)
        else:
            dfpop_thisfile = get_nlte_populations_oldformat(
                adata, nltefilepath, modelgridindex,
                timestep, atomic_number, T_e, T_R)

        # found our data!
        if not dfpop_thisfile.empty:
            return dfpop_thisfile

    return pd.DataFrame()


def addargs(parser):
    parser.add_argument('elements', nargs='*', default=['Fe'],
                        help='List of elements to plot')

    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', default=26,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Plotted modelgrid cell')

    parser.add_argument('-exc_temperature', type=float, default=6000.,
                        help='Comparison plot')

    parser.add_argument('-ionstages',
                        help='Ion stage range, 1 is neutral, 2 is 1+')

    parser.add_argument('--oldformat', default=False, action='store_true',
                        help='Use the old file format')

    parser.add_argument('-outputfile', '-o',
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
        timestep = at.get_closest_timestep(os.path.join(args.modelpath, 'spec.out'), args.timedays)
    else:
        timestep = int(args.timestep)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    if args.listtimesteps:
        at.showtimesteptimes('spec.out')
    else:
        ionstages_permitted = at.parse_range_list(args.ionstages) if args.ionstages else None
        adata = at.get_levels(args.modelpath)

        modeldata, _ = at.get_modeldata(os.path.join(args.modelpath, 'model.txt'))
        estimators = at.estimators.read_estimators(args.modelpath, modeldata)
        if estimators:
            if not estimators[(timestep, args.modelgridindex)]['emptycell']:
                T_e = estimators[(timestep, args.modelgridindex)]['Te']
                T_R = estimators[(timestep, args.modelgridindex)]['TR']
            else:
                print(f'ERROR: cell {args.modelgridindex} is empty. Setting T_e = T_R =  6000 K')
                T_e = 6000
                T_R = 6000
        else:
            print('Setting T_e = T_R =  6000 K')
            T_e = 6000
            T_R = 6000

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

            print(elsymbol, atomic_number)

            print(f'Getting level populations for modelgrid cell {args.modelgridindex} '
                  f'timestep {timestep} element {elsymbol}')
            dfpop = read_files(args.modelpath, adata, atomic_number, T_e, T_R,
                               timestep, args.modelgridindex, args.oldformat)

            if dfpop.empty:
                print(f'No NLTE population data for modelgrid cell {args.modelgridindex} timestep {timestep}')
            else:
                make_plot(modeldata, estimators, dfpop, atomic_number, ionstages_permitted, T_e, T_R, timestep, args)


def make_plot(modeldata, estimators, dfpop, atomic_number, ionstages_permitted, T_e, T_R, timestep, args):
    # top_ion = 9999
    max_ion_stage = dfpop.ion_stage.max()

    if len(dfpop.query('ion_stage == @max_ion_stage')) == 1:  # single-level ion, so skip it
        max_ion_stage -= 1

    ion_stage_list = sorted(
        [i for i in dfpop.ion_stage.unique()
         if i <= max_ion_stage and (ionstages_permitted is None or i in ionstages_permitted)])

    fig, axes = plt.subplots(len(ion_stage_list), 1, sharex=False, figsize=(9, 3 * len(ion_stage_list)),
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
        print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} has a population of {ionpopulation:.1f}')

        lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE_T_e'].sum())
        dfpopthision['n_LTE_T_e_normed'] = dfpopthision['n_LTE_T_e'] * lte_scalefactor

        axis.plot(dfpopthision.level.values[:-1], dfpopthision['n_LTE_T_e_normed'].values[:-1], linewidth=1.5,
                  label=f'LTE T_e = {T_e:.0f} K', linestyle='None', marker='*')

        lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE_T_R'].sum())
        dfpopthision['n_LTE_T_R_normed'] = dfpopthision['n_LTE_T_R'] * lte_scalefactor

        axis.plot(dfpopthision.level.values[:-1], dfpopthision['n_LTE_T_R_normed'].values[:-1], linewidth=1.5,
                  label=f'LTE T_R = {T_R:.0f} K', linestyle='None', marker='*')

        axis.plot(dfpopthision.level.values, dfpopthision.n_NLTE.values, linewidth=1.5,
                  label='NLTE', linestyle='None', marker='x')

        dfpopthisionoddlevels = dfpopthision.query('parity==1')

        axis.plot(dfpopthisionoddlevels.level.values, dfpopthisionoddlevels.n_NLTE.values, linewidth=2,
                  label='Odd parity', linestyle='None',
                  marker='s', markersize=10, markerfacecolor=(0, 0, 0, 0), markeredgecolor='black')

        # list_departure_ratio = [
        #     nlte / lte for (nlte, lte) in zip(list_nltepop[ion],
        #                                       list_ltepop[ion])]
        # axis.plot(list_levels[ion], list_departure_ratio, linewidth=1.5,
        #         label='NLTE/LTE', linestyle='None', marker='x')
        # axis.set_ylabel(r'')
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

    Te = estimators[(timestep, args.modelgridindex)]['Te']
    modelname = at.get_model_name(args.modelpath)
    velocity = modeldata['velocity'][args.modelgridindex]
    figure_title = (
        f'{modelname}\n'
        f'Cell {args.modelgridindex} (v={velocity} km/s) with Te = {Te:.1f} K at timestep {timestep:d}')
    time_days = float(at.get_timestep_time(args.modelpath, timestep))
    if time_days != -1:
        figure_title += f' ({time_days:.1f}d)'

    axes[0].set_title(figure_title, fontsize=11)

    outputfilename = args.outputfile.format(elsymbol=at.elsymbols[atomic_number], cell=args.modelgridindex,
                                            timestep=timestep, time_days=time_days)
    print(f"Saving {outputfilename}")
    fig.savefig(outputfilename, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
