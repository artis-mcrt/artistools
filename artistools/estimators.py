#!/usr/bin/env python3
# import math
import argparse
import glob
import gzip
import math
import os
import re
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import artistools as at

# from astropy import constants as const

defaultoutputfile = 'plotestimators_ts{timestep:02d}_{time_days:.0f}d.pdf'


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
    units = {
        'TR': 'K',
        'Te': 'K',
        'TJ': 'K',
        'nne': 'e-/cm3',
        'heating': 'erg/s/cm3',
        'cooling': 'erg/s/cm3',
        'velocity': 'km/s',
    }

    if variable in units:
        return f' [{units[variable]}]'
    elif variable.split('_')[0] in units:
        return f' [{units[variable.split("_")[0]]}]'
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


def read_estimators(modelpath, modeldata, keymatch=None):
    """keymatch should be a tuple (timestep, modelgridindex)."""
    estimfiles = (glob.glob(os.path.join(modelpath, 'estimators_????.out'), recursive=True) +
                  glob.glob(os.path.join(modelpath, 'estimators_????.out.gz'), recursive=True) +
                  glob.glob(os.path.join(modelpath, '*/estimators_????.out'), recursive=True) +
                  glob.glob(os.path.join(modelpath, '*/estimators_????.out.gz'), recursive=True))

    if not estimfiles:
        print("No estimator files found")
        return False

    print(f'Reading {len(estimfiles)} estimator files...')

    estimators = {}
    for estfile in estimfiles:
        filerank = int(re.findall('[0-9]+', os.path.basename(estfile))[-1])

        if keymatch is not None and filerank > keymatch[1]:
            continue

        opener = gzip.open if estfile.endswith('.gz') else open
        with opener(estfile, 'rt') as estfile:
            timestep = 0
            modelgridindex = 0
            skip_block = False
            for line in estfile:
                row = line.split()
                if not row:
                    continue

                if row[0] == 'timestep':
                    if keymatch is not None and keymatch in estimators:
                        # found our key, so exit now!
                        return estimators

                    timestep = int(row[1])
                    modelgridindex = int(row[3])
                    # print(f'Timestep {timestep} cell {modelgridindex}')
                    if (timestep, modelgridindex) in estimators and not estimators[(timestep, modelgridindex)]['emptycell']:
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


def plot_init_abundances(axis, xlist, specieslist, mgilist, modeldata, abundancedata, **plotkwargs):
    for speciesstr in specieslist:
        splitvariablename = speciesstr.split('_')
        atomic_number = at.get_atomic_number(splitvariablename[0].strip('0123456789'))
        axis.set_ylabel('Initial mass fraction')

        ylist = []
        for modelgridindex in mgilist:
            if speciesstr.lower() in ['ni_56', 'ni56', '56ni']:
                yvalue = modeldata.loc[modelgridindex]['X_Ni56']
            elif speciesstr.lower() in ['ni_stb', 'ni_stable']:
                yvalue = abundancedata.loc[modelgridindex][atomic_number] - modeldata.loc[modelgridindex]['X_Ni56']
            elif speciesstr.lower() in ['co_56', 'co56', '56co']:
                yvalue = modeldata.loc[modelgridindex]['X_Co56']
            elif speciesstr.lower() in ['fegrp', 'ffegroup']:
                yvalue = modeldata.loc[modelgridindex]['X_Fegroup']
            else:
                yvalue = abundancedata.loc[modelgridindex][atomic_number]
            ylist.append(yvalue)

        ylist.insert(0, ylist[0])
        # or axis.step(where='pre', )
        axis.plot(xlist, ylist, linewidth=1.5, label=f'{speciesstr}', **plotkwargs)


def plot_multi_ion_series(axis, xlist, seriestype, ionlist, timestep, mgilist, estimators, **plotkwargs):
    if seriestype == 'populations':
        axis.yaxis.set_major_locator(ticker.MultipleLocator(base=0.10))

    linecount = 0
    for ionstr in ionlist:
        splitvariablename = ionstr.split(' ')
        atomic_number = at.get_atomic_number(splitvariablename[0])
        ion_stage = at.decode_roman_numeral(splitvariablename[1])
        if seriestype == 'populations':
            axis.set_ylabel('X$_{ion}$/X$_{tot}$')
        else:
            axis.set_ylabel(seriestype)

        ylist = []
        for modelgridindex in mgilist:
            estim = estimators[(timestep, modelgridindex)]

            if estim['emptycell']:
                continue

            if seriestype == 'populations':
                totalpop = estim['populations']['total']
                elpop = estim['populations'][atomic_number]
                nionpop = estim['populations'].get((atomic_number, ion_stage), 0.)
                # ylist.append(nionpop / totalpop)  # Plot as fraction of total population
                ylist.append(nionpop / elpop)  # Plot as fraction of element population
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
        color = ['blue', 'green', 'red', 'cyan', 'purple', 'grey', 'brown', 'orange'][linecount]
        # or axis.step(where='pre', )
        axis.plot(xlist, ylist, linewidth=1.5, label=plotlabel, color=color, **plotkwargs)
        linecount += 1


def plot_series(axis, xlist, variablename, showlegend, timestep, mgilist, estimators, **plotkwargs):
    serieslabel = f'{variablename}{get_units_string(variablename)}'
    if showlegend:
        plotlabel = serieslabel
    else:
        axis.set_ylabel(serieslabel)
        plotlabel = None

    ylist = []
    for modelgridindex in mgilist:
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
            axis.set_yscale('log')
    except ZeroDivisionError:
        axis.set_yscale('log')

    dictcolors = {
        'Te': 'red',
        # 'heating_gamma': 'blue',
        # 'cooling_adiabatic': 'blue'
    }
    axis.plot(xlist, ylist, linewidth=1.5, label=plotlabel, color=dictcolors.get(variablename, None), **plotkwargs)


def get_xlist(xvariable, mgilist, estimators, timestep, args):
    mgilist_out = []
    if xvariable in ['cellid', 'modelgridindex']:
        xlist = mgilist
        if args.xmax >= 0:
            xlist, mgilist_out = zip(*[(x, mgi) for x, mgi in zip(xlist, mgi) if x <= args.xmax])
    else:
        try:
            xlist = []
            for modelgridindex in mgilist:
                xvalue = estimators[(timestep, modelgridindex)][xvariable]
                if args.xmax < 0 or xvalue <= args.xmax:
                    xlist.append(xvalue)
                    mgilist_out.append(modelgridindex)
        except KeyError:
            if (timestep, modelgridindex) in estimators:
                print(f'Unknown x variable: {xvariable} for timestep {timestep} in cell {modelgridindex}')
            else:
                print(f'No data for cell {modelgridindex} at timestep {timestep}')
            print(estimators[(timestep, modelgridindex)])
            sys.exit()

    return xlist, mgilist_out


def plot_timestep_subplot(axis, timestep, xlist, yvariables, mgilist, modeldata, abundancedata, estimators,
                          **plotkwargs):
    showlegend = False

    for variablename in yvariables:
        if not hasattr(variablename, 'lower'):  # if it's not a string, it's a list
            showlegend = True
            if variablename[0] == 'initabundances':
                plot_init_abundances(axis, xlist, variablename[1], mgilist, modeldata, abundancedata)
            else:
                seriestype, ionlist = variablename
                plot_multi_ion_series(axis, xlist, seriestype, ionlist, timestep, mgilist, estimators, **plotkwargs)
        else:
            showlegend = len(yvariables) > 1 or len(variablename) > 20
            plot_series(axis, xlist, variablename, showlegend, timestep, mgilist, estimators, **plotkwargs)

    if showlegend:
        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})


def plot_timestep(modelname, timestep, mgilist, estimators, xvariable, series, modeldata, abundancedata,
                  args, **plotkwargs):

    fig, axes = plt.subplots(len(series), 1, sharex=True, figsize=(8, 2.3 * len(series)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    if len(series) == 1:
        axes = [axes]
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    axes[-1].set_xlabel(f'{xvariable}{get_units_string(xvariable)}')
    xlist, mgilist = get_xlist(xvariable, mgilist, estimators, timestep, args)
    xlist = np.insert(xlist, 0, 0.)

    xmin = args.xmin if args.xmin > 0 else min(xlist)
    xmax = args.xmax if args.xmax > 0 else max(xlist)

    # xlist, mgilist = zip(*[(x, y) for (x, y) in zip(xlist, mgilist) if x >= xmin and x <= xmax])

    for axis, yvariables in zip(axes, series):
        axis.set_xlim(xmin=xmin, xmax=xmax)
        plot_timestep_subplot(axis, timestep, xlist, yvariables, mgilist, modeldata, abundancedata,
                              estimators, **plotkwargs)

    figure_title = f'{modelname}\nTimestep {timestep}'
    time_days = float(at.get_timestep_time('.', timestep))
    if time_days >= 0:
        figure_title += f' ({time_days:.2f}d)'
    axes[0].set_title(figure_title, fontsize=11)
    # plt.suptitle(figure_title, fontsize=11, verticalalignment='top')

    outfilename = args.outputfile.format(timestep=timestep, time_days=time_days)
    fig.savefig(outfilename, format='pdf')
    print(f'Saved {outfilename}')
    plt.close()


def plot_recombrates(estimators, outfilename, **plotkwargs):
    atomic_number = 28
    ion_stage_list = [2, 3, 4, 5]
    fig, axes = plt.subplots(len(ion_stage_list), 1, sharex=True, figsize=(5, 8),
                             tight_layout={"pad": 0.5, "w_pad": 0.0, "h_pad": 0.0})
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    for axis, ion_stage in zip(axes, ion_stage_list):

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
            axis.plot(listT_e_Nahar, dfrecombrates['RRC_total'], linewidth=2,
                      label=ionstr + " (Nahar)", markersize=6, marker='s', **plotkwargs)

        axis.plot(listT_e, list_rrc, linewidth=2, label=ionstr, markersize=6, marker='s', **plotkwargs)

        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})

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

    parser.add_argument('--recombrates', default=False, action='store_true',
                        help='Make a recombination rate plot')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts',
                        help='Timestep number to plot')

    parser.add_argument('-xmin', type=int, default=-1,
                        help='Plot range: minimum x value')

    parser.add_argument('-xmax', type=int, default=-1,
                        help='Plot range: maximum x value')

    parser.add_argument('-x', default='velocity',
                        help='Horizontal axis variable (cellid or velocity)')

    parser.add_argument('-o', action='store', dest='outputfile',
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
    modelname = at.get_model_name(modelpath)

    estimators = read_estimators(modelpath, modeldata)

    if not estimators:
        return -1

    serieslist = [
        ['heating_gamma', 'heating_coll', 'heating_bf', 'heating_ff'],
        ['cooling_adiabatic', 'cooling_coll', 'cooling_fb', 'cooling_ff'],
        ['heating_gamma/gamma_dep'],
        ['Te', 'TR'],
        ['nne'],
        [['initabundances', ['Fe', 'Ni', 'Ni_56', 'Ni_stable', 'Ar']]],
        [['populations', ['Fe I', 'Fe II', 'Fe III', 'Ni II', 'Ni III', 'Ar I']]],
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

        for timestep in range(timestepmin, timestepmax + 1):

            nonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                               if not estimators[(timestep, modelgridindex)]['emptycell']]

            plot_timestep(modelname, timestep, nonemptymgilist, estimators, args.x, serieslist,
                          modeldata, abundancedata, args)


if __name__ == "__main__":
    main()
