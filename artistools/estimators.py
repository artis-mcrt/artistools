#!/usr/bin/env python3
"""Functions for reading and plotting estimator files.

Examples are temperatures, populations, heating/cooling rates.
"""
# import math
import argparse
import glob
import gzip
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

import artistools as at

# from astropy import constants as const

defaultoutputfile = Path('plotestimators_ts{timestep:02d}_{time_days:.0f}d.pdf')

colors_tab10 = list(plt.get_cmap('tab10')(np.linspace(0, 1.0, 10)))
elementcolors = {
    'Fe': colors_tab10[0],
    'Ni': colors_tab10[1],
    'Co': colors_tab10[2],
}

variableunits = {
    'TR': 'K',
    'Te': 'K',
    'TJ': 'K',
    'nne': 'e-/cm3',
    'heating': 'erg/s/cm3',
    'cooling': 'erg/s/cm3',
    'velocity': 'km/s',
}

variablelongunits = {
    'TR': 'Temperature in kelvin',
    'Te': 'Temperature in kelvin',
    'TJ': 'Temperature in kelvin',
}


def get_elemcolor(atomic_number=None, elsymbol=None):
    assert (atomic_number is None) != (elsymbol is None)
    if atomic_number is not None:
        elsymbol = at.elsymbols[atomic_number]

    if elsymbol not in elementcolors:
        elementcolors[elsymbol] = colors_tab10[len(elementcolors)]

    return elementcolors[elsymbol]


def get_ionrecombrates_fromfile(filename):
    """
        WARNING: copy pasted from artis-atomic! replace with a package import soon
        ionstage is the lower ion stage!
    """
    print(f'Reading {filename}')

    header_row = []
    with open(filename, 'r') as filein:
        while True:
            line = filein.readline()
            if line.strip().startswith('TOTAL RECOMBINATION RATE'):
                line = filein.readline()
                line = filein.readline()
                header_row = filein.readline().strip().replace(' n)', '-n)').split()
                break

        if not header_row:
            print("ERROR: no header found")
            sys.exit()

        index_logt = header_row.index('log(T)')
        index_low_n = header_row.index('RRC(low-n)')
        index_tot = header_row.index('RRC(total)')

        recomb_tuple = namedtuple("recomb_tuple", ['logT', 'RRC_low_n', 'RRC_total'])
        records = []
        for line in filein:
            row = line.split()
            if row:
                if len(row) != len(header_row):
                    print('Row contains wrong number of items for header:')
                    print(header_row)
                    print(row)
                    sys.exit()
                records.append(recomb_tuple(
                    *[float(row[index]) for index in [index_logt, index_low_n, index_tot]]))

    dfrecombrates = pd.DataFrame.from_records(records, columns=recomb_tuple._fields)
    return dfrecombrates


def get_units_string(variable):
    if variable in variableunits:
        return f' [{variableunits[variable]}]'
    elif variable.split('_')[0] in variableunits:
        return f' [{variableunits[variable.split("_")[0]]}]'
    return ''


def get_ylabel(variable):
    if variable in variablelongunits:
        return variablelongunits[variable]
    elif variable in variableunits:
        return f'[{variableunits[variable]}]'
    elif variable.split('_')[0] in variableunits:
        return f'[{variableunits[variable.split("_")[0]]}]'
    return ''


def parse_ion_row(row, outdict):
    variablename = row[0]
    if row[1].endswith('='):
        atomic_number = int(row[2])
        startindex = 3
    else:
        atomic_number = int(row[1].split('=')[1])
        startindex = 2

    if variablename not in outdict:
        outdict[variablename] = {}

    for index, token in list(enumerate(row))[startindex::2]:
        try:
            ion_stage = int(token.rstrip(':'))
        except ValueError:
            print(f'Cannot parse row: {row}')
            return

        value_thision = float(row[index + 1].rstrip(','))

        outdict[variablename][(atomic_number, ion_stage)] = value_thision

        if variablename == 'populations':
            elpop = outdict[variablename].get(atomic_number, 0)
            outdict[variablename][atomic_number] = elpop + value_thision

            totalpop = outdict[variablename].get('total', 0)
            outdict[variablename]['total'] = totalpop + value_thision

        elif variablename == 'Alpha_R*nne':
            if 'Alpha_R' not in outdict:
                outdict['Alpha_R'] = {}
            outdict['Alpha_R'][(atomic_number, ion_stage)] = value_thision / outdict['nne']


def read_estimators(modelpath, modeldata=None, modelgridindex=-1, timestep=-1):
    """Read estimator files into a nested dictionary structure.

    keymatch should be a tuple (timestep, modelgridindex).
    """
    match_timestep = timestep
    match_modelgridindex = modelgridindex
    if modeldata is None:
        modeldata, _ = at.get_modeldata(modelpath)

    if match_modelgridindex >= 0:
        mpirank = at.get_mpirankofcell(match_modelgridindex, modelpath=modelpath)
        strmpirank = f'{mpirank:04d}'
    else:
        strmpirank = '????'

    estimfiles = chain(
        Path(modelpath).rglob(f'**/estimators_{strmpirank}.out'),
        Path(modelpath).rglob(f'**/estimators_{strmpirank}.out.gz'))

    if match_modelgridindex < 0:
        npts_model = at.get_npts_model(modelpath)
        estimfiles = [x for x in estimfiles if
                      int(re.findall('[0-9]+', os.path.basename(x))[-1]) < npts_model]
        print(f'Reading {len(list(estimfiles))} estimator files from {modelpath}...')

    if not estimfiles:
        print("No estimator files found")
        return False

    estimators = {}
    for estfile in sorted(estimfiles):
        if match_modelgridindex >= 0:
            print(f'Reading {estfile}...')

        opener = gzip.open if str(estfile).endswith('.gz') else open
        with opener(estfile, 'rt') as estfile:
            timestep = 0
            modelgridindex = 0
            skip_block = False
            for line in estfile:
                row = line.split()
                if not row:
                    continue

                if row[0] == 'timestep':
                    if (
                            match_timestep >= 0 and match_modelgridindex >= 0 and
                            (match_timestep, match_modelgridindex)) in estimators:
                        # found our key, so exit now!
                        return estimators

                    timestep = int(row[1])
                    modelgridindex = int(row[3])
                    # print(f'Timestep {timestep} cell {modelgridindex}')
                    if ((timestep, modelgridindex) in estimators and
                            not estimators[(timestep, modelgridindex)]['emptycell']):
                        # print(f'WARNING: duplicate estimator data for timestep {timestep} cell {modelgridindex}. '
                        #       f'Kept old (T_e {estimators[(timestep, modelgridindex)]["Te"]}), '
                        #       f'instead of new (T_e {float(row[7])})')
                        skip_block = True
                    else:
                        skip_block = False
                        estimators[(timestep, modelgridindex)] = {}
                        estimators[(timestep, modelgridindex)]['velocity'] = modeldata['velocity'][modelgridindex]
                        emptycell = (row[4] == 'EMPTYCELL')
                        estimators[(timestep, modelgridindex)]['emptycell'] = emptycell
                        if not emptycell:
                            estimators[(timestep, modelgridindex)]['TR'] = float(row[5])
                            estimators[(timestep, modelgridindex)]['Te'] = float(row[7])
                            estimators[(timestep, modelgridindex)]['W'] = float(row[9])
                            estimators[(timestep, modelgridindex)]['TJ'] = float(row[11])
                            estimators[(timestep, modelgridindex)]['nne'] = float(row[15])

                elif row[1].startswith('Z=') and not skip_block:
                    parse_ion_row(row, estimators[(timestep, modelgridindex)])

                elif row[0] == 'heating:' and not skip_block:
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'heating_{token}'] = float(row[index + 1])
                    if estimators[(timestep, modelgridindex)]['heating_gamma/gamma_dep'] > 0:
                        estimators[(timestep, modelgridindex)]['gamma_dep'] = (
                            estimators[(timestep, modelgridindex)]['heating_gamma'] /
                            estimators[(timestep, modelgridindex)]['heating_gamma/gamma_dep'])

                elif row[0] == 'cooling:' and not skip_block:
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'cooling_{token}'] = float(row[index + 1])

    return estimators


def plot_init_abundances(ax, xlist, specieslist, mgilist, modeldata, abundancedata, **plotkwargs):
    ax.set_ylim(ymin=0.)
    ax.set_ylim(ymax=1.0)
    for speciesstr in specieslist:
        splitvariablename = speciesstr.split('_')
        atomic_number = at.get_atomic_number(splitvariablename[0].strip('0123456789'))
        ax.set_ylabel('Initial mass fraction')

        ylist = []
        linelabel = speciesstr
        linestyle = '-'
        for modelgridindex in mgilist:
            if speciesstr.lower() in ['ni_56', 'ni56', '56ni']:
                yvalue = modeldata.loc[modelgridindex]['X_Ni56']
                linelabel = '$^{56}$Ni'
                linestyle = '--'
            elif speciesstr.lower() in ['ni_stb', 'ni_stable']:
                yvalue = abundancedata.loc[modelgridindex][atomic_number] - modeldata.loc[modelgridindex]['X_Ni56']
                linelabel = 'Stable Ni'
            elif speciesstr.lower() in ['co_56', 'co56', '56co']:
                yvalue = modeldata.loc[modelgridindex]['X_Co56']
                linelabel = '$^{56}$Co'
            elif speciesstr.lower() in ['fegrp', 'ffegroup']:
                yvalue = modeldata.loc[modelgridindex]['X_Fegroup']
            else:
                yvalue = abundancedata.loc[modelgridindex][atomic_number]
            ylist.append(yvalue)

        ylist.insert(0, ylist[0])
        # or ax.step(where='pre', )
        color = get_elemcolor(atomic_number=atomic_number)
        ax.plot(xlist, ylist, linewidth=1.5, label=linelabel,
                  linestyle=linestyle, color=color, **plotkwargs)


def plot_multi_ion_series(
        ax, xlist, seriestype, ionlist, timesteplist, mgilist, estimators, compositiondata, args, **plotkwargs):
    ax.set_yscale('log')
    # if seriestype == 'populations':
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.10))

    # decoded into numeric form, e.g., [(26, 1), (26, 2)]
    iontuplelist = [
        (at.get_atomic_number(ionstr.split(' ')[0]), at.decode_roman_numeral(ionstr.split(' ')[1]))
        for ionstr in ionlist]
    iontuplelist.sort()
    print(f'Subplot with ions: {iontuplelist}')

    prev_atomic_number = iontuplelist[0][0]
    colorindex = 0
    for atomic_number, ion_stage in iontuplelist:
        if atomic_number != prev_atomic_number:
            colorindex += 1

        if compositiondata.query('Z == @atomic_number '
                                 '& lowermost_ionstage <= @ion_stage '
                                 '& uppermost_ionstage >= @ion_stage').empty:
            print(f"WARNING: Can't plot '{seriestype}' for Z={atomic_number} ion_stage {ion_stage} "
                  f"because this ion is not in compositiondata.txt")
            continue

        if seriestype == 'populations':
            if args.ionpoptype == 'absolute':
                ax.set_ylabel('X$_{ion}$ [/cm3]')
            elif args.ionpoptype == 'elpop':
                # elcode = at.elsymbols[atomic_number]
                ax.set_ylabel('X$_{ion}$/X$_{element}$')
            elif args.ionpoptype == 'totalpop':
                ax.set_ylabel('X$_{ion}$/X$_{tot}$')
            else:
                assert False
        else:
            ax.set_ylabel(seriestype)

        ylist = []
        for modelgridindex, timestep in zip(mgilist, timesteplist):
            estim = estimators[(timestep, modelgridindex)]

            if estim['emptycell']:
                continue

            if seriestype == 'populations':
                if (atomic_number, ion_stage) not in estim['populations']:
                    print(f'Note: population for {(atomic_number, ion_stage)} not in estimators for '
                          f'cell {modelgridindex} timestep {timestep}')
                    # print(f'Keys: {estim["populations"].keys()}')
                    # raise KeyError

                nionpop = estim['populations'].get((atomic_number, ion_stage), 0.)

                try:
                    if args.ionpoptype == 'absolute':
                        yvalue = nionpop  # Plot as fraction of element population
                    elif args.ionpoptype == 'elpop':
                        elpop = estim['populations'].get(atomic_number, 0.)
                        yvalue = nionpop / elpop  # Plot as fraction of element population
                    elif args.ionpoptype == 'totalpop':
                        totalpop = estim['populations']['total']
                        yvalue = nionpop / totalpop  # Plot as fraction of total population
                    else:
                        assert False
                except ZeroDivisionError:
                    yvalue = 0.

                ylist.append(yvalue)

            # elif seriestype == 'Alpha_R':
            #     ylist.append(estim['Alpha_R*nne'].get((atomic_number, ion_stage), 0.) / estim['nne'])
            # else:
            #     ylist.append(estim[seriestype].get((atomic_number, ion_stage), 0.))
            else:
                dictvars = {}
                for k, v in estim.items():
                    if 'items' in dir(v):
                        dictvars[k] = v.get((atomic_number, ion_stage), 0.)
                    else:
                        dictvars[k] = v
                try:
                    yvalue = eval(seriestype, {"__builtins__": math}, dictvars)
                except ZeroDivisionError:
                    yvalue = float('NaN')
                ylist.append(yvalue)

        plotlabel = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]}'

        ylist.insert(0, ylist[0])
        dashes_list = [(4.5, 2, 3, 2, 4.5, 2), (5, 1), (2, 1), (6, 2), (6, 1)]
        linestyle_list = ['-.', '-', '--', ':'] + [(0, x) for x in dashes_list]
        linestyle = linestyle_list[ion_stage - 1]
        linewidth = [1.5, 1.5, 1.0, 1.0, 1.0][ion_stage - 1]
        # color = ['blue', 'green', 'red', 'cyan', 'purple', 'grey', 'brown', 'orange'][ion_stage - 1]
        # assert colorindex < 10
        # color = f'C{colorindex}'
        color = get_elemcolor(atomic_number=atomic_number)
        # or ax.step(where='pre', )
        ax.plot(xlist, ylist, linewidth=linewidth, label=plotlabel, color=color, linestyle=linestyle, **plotkwargs)
        prev_atomic_number = atomic_number

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin=ymin, ymax=3 * ymax)


def plot_series(ax, xlist, variablename, showlegend, timesteplist, mgilist, estimators, nounits=False, **plotkwargs):
    dictlabelreplacements = {
        'Te': 'T$_e$',
        'TR': 'T$_R$'
    }
    formattedvariablename = dictlabelreplacements.get(variablename, variablename)
    serieslabel = f'{formattedvariablename}'
    if not nounits:
        serieslabel += get_units_string(variablename)

    if showlegend:
        linelabel = serieslabel
    else:
        ax.set_ylabel(serieslabel)
        linelabel = None

    ylist = []
    for modelgridindex, timestep in zip(mgilist, timesteplist):
        try:
            ylist.append(eval(variablename, {"__builtins__": math}, estimators[(timestep, modelgridindex)]))
        except KeyError:
            if (timestep, modelgridindex) in estimators:
                print(f"Undefined variable: {variablename} for timestep {timestep} in cell {modelgridindex}")
            else:
                print(f'No data for cell {modelgridindex} at timestep {timestep}')
            # print(estimators[(timestep, modelgridindex)])
            sys.exit()

    ylist.insert(0, ylist[0])

    try:
        if math.log10(max(ylist) / min(ylist)) > 2:
            ax.set_yscale('log')
    except ZeroDivisionError:
        ax.set_yscale('log')

    dictcolors = {
        'Te': 'red',
        # 'heating_gamma': 'blue',
        # 'cooling_adiabatic': 'blue'
    }

    # print out the data to stdout. Maybe want to add a CSV export option at some point?
    # print(f'#cellidorvelocity {variablename}\n' + '\n'.join([f'{x}  {y}' for x, y in zip(xlist, ylist)]))

    ax.plot(xlist, ylist, linewidth=1.5, label=linelabel, color=dictcolors.get(variablename, None), **plotkwargs)


def get_xlist(xvariable, allnonemptymgilist, estimators, timesteplist, args):
    if xvariable in ['cellid', 'modelgridindex']:
        if args.xmax >= 0:
            mgilist_out = [mgi for mgi in allnonemptymgilist if mgi <= args.xmax]
        else:
            mgilist_out = allnonemptymgilist
        xlist = mgilist_out
        timesteplist_out = timesteplist
    else:
        try:
            xlist = []
            mgilist_out = []
            timesteplist_out = []
            for modelgridindex, timestep in zip(allnonemptymgilist, timesteplist):
                xvalue = estimators[(timestep, modelgridindex)][xvariable]
                if args.xmax < 0 or xvalue <= args.xmax:
                    xlist.append(xvalue)
                    mgilist_out.append(modelgridindex)
                    timesteplist_out.append(timestep)
        except KeyError:
            if (timestep, modelgridindex) in estimators:
                print(f'Unknown x variable: {xvariable} for timestep {timestep} in cell {modelgridindex}')
            else:
                print(f'No data for cell {modelgridindex} at timestep {timestep}')
            print(estimators[(timestep, modelgridindex)])
            sys.exit()

    xlist, mgilist_out, timesteplist_out = zip(
        *[(xlist, mgi, timestep) for x, mgi, timestep in sorted(zip(xlist, mgilist_out, timesteplist_out))])

    return xlist, mgilist_out, timesteplist_out


def plot_subplot(ax, timesteplist, xlist, yvariables, mgilist, modeldata, abundancedata, compositiondata,
                 estimators, args, **plotkwargs):
    showlegend = False

    ylabel = 'UNDEFINED'
    sameylabel = True
    for variablename in yvariables:
        if not hasattr(variablename, 'lower'):
            pass
        elif ylabel == 'UNDEFINED':
            ylabel = get_ylabel(variablename)
        elif ylabel != get_ylabel(variablename):
            sameylabel = False
            break

    for variablename in yvariables:
        if not hasattr(variablename, 'lower'):  # if it's not a string, it's a list
            showlegend = True
            if variablename[0] == 'initabundances':
                plot_init_abundances(ax, xlist, variablename[1], mgilist, modeldata, abundancedata)
            else:
                seriestype, ionlist = variablename
                plot_multi_ion_series(ax, xlist, seriestype, ionlist, timesteplist, mgilist, estimators,
                                      compositiondata, args, **plotkwargs)
        else:
            showlegend = len(yvariables) > 1 or len(variablename) > 20
            plot_series(ax, xlist, variablename, showlegend, timesteplist, mgilist, estimators,
                        nounits=sameylabel, **plotkwargs)
            if showlegend and sameylabel:
                ax.set_ylabel(ylabel)

    if showlegend:
        if yvariables[0][0] == 'populations':
            ax.legend(loc='upper left', handlelength=2, ncol=3,
                      frameon=False, numpoints=1, prop={'size': 8})
        else:
            ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})


def plot_timestep(modelpath, timestep, allnonemptymgilist, estimators, xvariable, plotlist, modeldata, abundancedata,
                  compositiondata, args, **plotkwargs):

    modelname = at.get_model_name(modelpath)
    fig, axes = plt.subplots(nrows=len(plotlist), ncols=1, sharex=True, figsize=(6.4, 2.5 * len(plotlist)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    if len(plotlist) == 1:
        axes = [axes]

    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    timesteplist_unfiltered = [timestep] * len(allnonemptymgilist)  # constant timestep

    axes[-1].set_xlabel(f'{xvariable}{get_units_string(xvariable)}')
    xlist, mgilist, timesteplist = get_xlist(xvariable, allnonemptymgilist, estimators, timesteplist_unfiltered, args)
    xlist = np.insert(xlist, 0, 0.)

    xmin = args.xmin if args.xmin > 0 else min(xlist)
    xmax = args.xmax if args.xmax > 0 else max(xlist)

    for ax, yvariables in zip(axes, plotlist):
        ax.set_xlim(xmin=xmin, xmax=xmax)
        plot_subplot(ax, timesteplist, xlist, yvariables, mgilist, modeldata, abundancedata,
                     compositiondata, estimators, args, **plotkwargs)

    figure_title = f'{modelname}\nTimestep {timestep}'
    try:
        time_days = float(at.get_timestep_time(modelpath, timestep))
    except FileNotFoundError:
        time_days = 0
    else:
        figure_title += f' ({time_days:.2f}d)'

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=11)
    # plt.suptitle(figure_title, fontsize=11, verticalalignment='top')

    outfilename = str(args.outputfile).format(timestep=timestep, time_days=time_days)
    fig.savefig(outfilename, format='pdf')
    print(f'Saved {outfilename}')
    if args.show:
        plt.show()
    else:
        plt.close()


def plot_recombrates(estimators, outfilename, **plotkwargs):
    atomic_number = 28
    ion_stage_list = [2, 3, 4, 5]
    fig, axes = plt.subplots(
        nrows=len(ion_stage_list), ncols=1, sharex=True, figsize=(5, 8),
        tight_layout={"pad": 0.5, "w_pad": 0.0, "h_pad": 0.0})
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    for ax, ion_stage in zip(axes, ion_stage_list):

        ionstr = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} to {at.roman_numerals[ion_stage - 1]}'

        listT_e = []
        list_rrc = []
        for _, dicttimestepmodelgrid in estimators.items():
            if (atomic_number, ion_stage) in dicttimestepmodelgrid['RRC_LTE_Nahar']:
                listT_e.append(dicttimestepmodelgrid['Te'])
                list_rrc.append(dicttimestepmodelgrid['RRC_LTE_Nahar'][(atomic_number, ion_stage)])

        if not list_rrc:
            continue

        listT_e, list_rrc = zip(*sorted(zip(listT_e, list_rrc), key=lambda x: x[0]))

        rrcfiles = glob.glob(
            f'/Users/lshingles/Library/Mobile Documents/com~apple~CloudDocs/GitHub/artis-atomic/atomic-data-nahar/{at.elsymbols[atomic_number].lower()}{ion_stage - 1}.rrc*.txt')
        if rrcfiles:
            dfrecombrates = get_ionrecombrates_fromfile(rrcfiles[0])
            logT_e_min = math.log10(min(listT_e))
            logT_e_max = math.log10(max(listT_e))
            dfrecombrates.query("logT > @logT_e_min & logT < @logT_e_max", inplace=True)

            listT_e_Nahar = [10 ** x for x in dfrecombrates['logT'].values]
            ax.plot(listT_e_Nahar, dfrecombrates['RRC_total'], linewidth=2,
                      label=ionstr + " (Nahar)", markersize=6, marker='s', **plotkwargs)

        ax.plot(listT_e, list_rrc, linewidth=2, label=ionstr, markersize=6, marker='s', **plotkwargs)

        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})

    # modelname = at.get_model_name(".")
    # plotlabel = f'Timestep {timestep}'
    # time_days = float(at.get_timestep_time('spec.out', timestep))
    # if time_days >= 0:
    #     plotlabel += f' (t={time_days:.2f}d)'
    # fig.suptitle(plotlabel, fontsize=12)

    fig.savefig(outfilename, format='pdf')
    print(f'Saved {outfilename}')
    plt.close()


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('--recombrates', action='store_true',
                        help='Make a recombination rate plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=-1,
                        help='Modelgridindex for time evolution plot')

    parser.add_argument('-timestep', '-ts',
                        help='Timestep number for internal structure plot')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot for internal structure plot')

    parser.add_argument('-x', default='velocity',
                        help='Horizontal axis variable, e.g. cellid, velocity, timestep, or time')

    parser.add_argument('-xmin', type=int, default=-1,
                        help='Plot range: minimum x value')

    parser.add_argument('-xmax', type=int, default=-1,
                        help='Plot range: maximum x value')

    parser.add_argument('--notitle', action='store_true',
                        help='Suppress the top title from the plot')

    parser.add_argument('-plotlist', type=list, default=[],
                        help='Plot list (when calling from Python only)')

    parser.add_argument('-ionpoptype', default='elpop', choices=['absolute', 'totalpop', 'elpop'],
                        help=(
                            'Plot absolutely ion populations, or ion populations as a'
                            ' fraction of total or element population'))

    parser.add_argument('-show', action='store_true',
                        help='Show plot before quitting')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path,
                        default=defaultoutputfile,
                        help='Filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS estimators.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    modelpath = args.modelpath

    modeldata, _ = at.get_modeldata(modelpath)
    abundancedata = at.get_initialabundances(modelpath)
    compositiondata = at.get_composition_data(modelpath)

    estimators = read_estimators(modelpath, modeldata)

    if not estimators:
        return -1

    if args.plotlist:
        plotlist = args.plotlist
    else:
        plotlist = [
            [['initabundances', ['Fe', 'Ni_stable', 'Ni_56']]],
            # ['heating_gamma', 'heating_coll', 'heating_bf', 'heating_ff'],
            # ['cooling_adiabatic', 'cooling_coll', 'cooling_fb', 'cooling_ff'],
            # ['heating_gamma/gamma_dep'],
            # ['nne'],
            ['Te', 'TR'],
            # [['populations', ['He I', 'He II', 'He III']]],
            # [['populations', ['C I', 'C II', 'C III', 'C IV', 'C V']]],
            # [['populations', ['O I', 'O II', 'O III', 'O IV']]],
            # [['populations', ['Ne I', 'Ne II', 'Ne III', 'Ne IV', 'Ne V']]],
            # [['populations', ['Si I', 'Si II', 'Si III', 'Si IV', 'Si V']]],
            # [['populations', ['Cr I', 'Cr II', 'Cr III', 'Cr IV', 'Cr V']]],
            # [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Fe VI', 'Fe VII', 'Fe VIII']]],
            # [['populations', ['Co I', 'Co II', 'Co III', 'Co IV', 'Co V', 'Co VI', 'Co VII']]],
            # [['populations', ['Ni I', 'Ni II', 'Ni III', 'Ni IV', 'Ni V', 'Ni VI', 'Ni VII']]],
            [['populations', ['Fe II', 'Fe III', 'Co II', 'Co III', 'Ni II', 'Ni III']]],
            # [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
            # [['Alpha_R / RRC_LTE_Nahar', ['Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni III']]],
            # [['gamma_NT', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
        ]

    if args.recombrates:
        plot_recombrates(estimators, "plotestimators_recombrates.pdf")
    else:
        if args.timedays:
            if isinstance(args.timedays, str) and '-' in args.timedays:
                timestepmin, timestepmax = [
                    at.get_closest_timestep(modelpath, float(timedays))
                    for timedays in args.timedays.split('-')]
            else:
                timestep = at.get_closest_timestep(modelpath, args.timedays)
                timestepmin, timestepmax = timestep, timestep
        else:
            if not args.timestep:
                print('ERROR: A time or timestep must be specified')
                return -1

            if '-' in args.timestep:
                timestepmin, timestepmax = [int(nts) for nts in args.timestep.split('-')]
            else:
                timestepmin = int(args.timestep)
                timestepmax = timestepmin

        allnonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                              if not estimators[(timestepmin, modelgridindex)]['emptycell']]

        if args.modelgridindex > -1:
            for timestep in range(timestepmin, timestepmax + 1):
                plot_timestep(modelpath, timestep, allnonemptymgilist, estimators, args.x, plotlist,
                              modeldata, abundancedata, compositiondata, args)
        else:
            print("Time evolution plot not implemented yet")



if __name__ == "__main__":
    main()
