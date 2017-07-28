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


def get_nlte_populations(modelpath, nltefilename, modelgridindex, timestep, atomic_number, temperature_exc):
    all_levels = at.get_levels(os.path.join(modelpath, 'adata.txt'))

    dfpop = pd.read_csv(nltefilename, delim_whitespace=True)
    dfpop.query('(modelgridindex==@modelgridindex) & (timestep==@timestep) & (Z==@atomic_number)', inplace=True)

    k_b = const.k_B.to('eV / K').value
    list_indicies = []
    list_ltepopcustom = []
    list_parity = []
    gspop = {}
    for index, row in dfpop.iterrows():
        list_indicies.append(index)

        atomic_number = int(row.Z)
        ion_stage = int(row.ion_stage)
        if (row.Z, row.ion_stage) not in gspop:
            gspop[(row.Z, row.ion_stage)] = dfpop.query(
                'timestep==@timestep and Z==@atomic_number and ion_stage==@ion_stage and level==0').iloc[0].n_NLTE

        levelnumber = int(row.level)
        if levelnumber == -1:  # superlevel
            levelnumber = dfpop.query(
                'timestep==@timestep and Z==@atomic_number and ion_stage==@ion_stage').level.max()
            dfpop.loc[index, 'level'] = levelnumber + 2
            ltepopcustom = 0.0
            parity = 0
            print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} has a superlevel at level {levelnumber}')
        else:
            for _, ion_data in enumerate(all_levels):
                if ion_data.Z == atomic_number and ion_data.ion_stage == ion_stage:
                    level = ion_data.level_list[levelnumber]
                    gslevel = ion_data.level_list[0]

            ltepopcustom = gspop[(row.Z, row.ion_stage)] * level.g / gslevel.g * math.exp(
                - (level.energy_ev - gslevel.energy_ev) / k_b / temperature_exc)

            levelname = level.levelname.split('[')[0]
            parity = 1 if levelname[-1] == 'o' else 0

        list_ltepopcustom.append(ltepopcustom)
        list_parity.append(parity)

    dfpop['n_LTE_custom'] = pd.Series(list_ltepopcustom, index=list_indicies)
    dfpop['parity'] = pd.Series(list_parity, index=list_indicies)

    return dfpop


def get_nlte_populations_oldformat(modelpath, nltefilename, modelgridindex, timestep, atomic_number, temperature_exc):
    compositiondata = at.get_composition_data('compositiondata.txt')
    elementdata = compositiondata.query('Z==@atomic_number')

    if len(elementdata) < 1:
        print(f'Error: element Z={atomic_number} not in composition file')
        return None

    all_levels = at.get_levels(os.path.join(modelpath, 'adata.txt'))

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
            level = ion_data.level_list[levelnumber]
            gslevel = ion_data.level_list[0]

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


def read_files(nlte_files, atomic_number, args):
    for nltefilepath in nlte_files:
        filerank = int(re.search('[0-9]+', os.path.basename(nltefilepath)).group(0))

        if filerank > args.modelgridindex:
            continue

        print(f'Loading {nltefilepath}')

        if not args.oldformat:
            dfpop_thisfile = get_nlte_populations(
                args.modelpath, nltefilepath, args.modelgridindex,
                args.timestep, atomic_number, args.exc_temperature)
        else:
            dfpop_thisfile = get_nlte_populations_oldformat(
                args.modelpath, nltefilepath, args.modelgridindex,
                args.timestep, atomic_number, args.exc_temperature)

        # found our data!
        if not dfpop_thisfile.empty:
            return dfpop_thisfile

    return pd.DataFrame()


def addargs(parser, defaultoutputfile):
    parser.add_argument('modelpath', nargs='?', default='.',
                        help='Path to ARTIS folder')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', '-ts', type=int, default=70,
                        help='Plotted timestep')
    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Plotted modelgrid cell')
    parser.add_argument('element', nargs='?', default='Fe',
                        help='Plotted element')
    parser.add_argument('-exc_temperature', type=float, default=6000.,
                        help='Comparison plot')
    parser.add_argument('--oldformat', default=False, action='store_true',
                        help='Use the old file format')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='path/filename for PDF file')


def main(argsraw=None):
    defaultoutputfile = 'plotnlte_{elsymbol}_cell{cell:03d}_{timestep:03d}.pdf'

    parser = argparse.ArgumentParser(
        description='Plot ARTIS non-LTE corrections.')
    addargs(parser, defaultoutputfile)
    args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    if args.listtimesteps:
        at.showtimesteptimes('spec.out')
    else:
        if args.modelpath.title() in at.elsymbols:
            args.element = args.modelpath
            args.modelpath = '.'

        try:
            atomic_number = next(Z for Z, elsymb in enumerate(at.elsymbols) if elsymb.lower() == args.element.lower())
        except StopIteration:
            print(f"Could not find element '{args.element}'")
            return

        nlte_files = (
            glob.glob(os.path.join(args.modelpath, 'nlte_????.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, '*/nlte_????.out'), recursive=True))

        if not nlte_files:
            print("No NLTE files found.")
            return
        else:
            print(f'Getting level populations for modelgrid cell {args.modelgridindex} '
                  f'timestep {args.timestep} element {args.element}')
            dfpop = read_files(nlte_files, atomic_number, args)

            if dfpop.empty:
                print(f'No data for modelgrid cell {args.modelgridindex} timestep {args.timestep}')
            else:
                make_plot(dfpop, atomic_number, args.exc_temperature, args)


def make_plot(dfpop, atomic_number, exc_temperature, args):
    top_ion = -1 if args.oldformat else -2  # skip top ion, which is probably ground state only
    top_ion = 9999
    ion_stage_list = dfpop.ion_stage.unique()[:top_ion]
    fig, axes = plt.subplots(len(ion_stage_list), 1, sharex=False, figsize=(8, 7),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if len(dfpop) == 0:
        print('Error, no data for selected timestep and element')
        sys.exit()

    for ion, axis in enumerate(axes):
        ion_stage = ion_stage_list[ion]
        dfpopthision = dfpop.query('ion_stage==@ion_stage').copy()
        ionpopulation = dfpopthision['n_NLTE'].sum()
        print(f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} has a population of {ionpopulation:.1f}')

        lte_scalefactor = float(ionpopulation / dfpopthision['n_LTE'].sum())
        dfpopthision['n_LTE_normed'] = dfpopthision['n_LTE'].apply(lambda pop: pop * lte_scalefactor)
        lte_custom_scalefactor = float(ionpopulation / dfpopthision['n_LTE_custom'].sum())
        dfpopthision['n_LTE_custom_normed'] = dfpopthision['n_LTE_custom'].apply(
            lambda pop: pop * lte_custom_scalefactor)

        axis.plot(dfpopthision.level.values, dfpopthision.n_LTE_normed.values, linewidth=1.5,
                  label='LTE', linestyle='None', marker='+')

        axis.plot(dfpopthision.level.values[:-1], dfpopthision.n_LTE_custom_normed.values[:-1], linewidth=1.5,
                  label=f'LTE {exc_temperature:.0f} K', linestyle='None', marker='*')

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
        plotlabel = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]}'
        time_days = at.get_timestep_time('spec.out', args.timestep)
        if time_days != -1:
            plotlabel += f' at t={time_days} days'
        else:
            plotlabel += f' at timestep {args.timestep:d}'

        plotlabel += f' in cell {args.modelgridindex}'

        axis.annotate(plotlabel, xy=(0.5, 0.96), xycoords='axes fraction',
                      horizontalalignment='center', verticalalignment='top', fontsize=12)
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

    for axis in axes:
        # axis.set_xlim(xmin=270,xmax=300)
        # axis.set_ylim(ymin=-0.1,ymax=1.3)
        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
        axis.set_yscale('log')
    axes[-1].set_xlabel(r'Level index')

    outputfilename = args.outputfile.format(elsymbol=at.elsymbols[atomic_number], cell=args.modelgridindex, timestep=args.timestep)
    print(f"Saving {outputfilename}")
    fig.savefig(outputfilename, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
