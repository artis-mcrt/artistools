#!/usr/bin/env python3
"""Functions for reading and plotting estimator files.

Examples are temperatures, populations, heating/cooling rates.
"""
# import math
import argparse
import math
import os
# import re
import sys
from collections import namedtuple
from functools import lru_cache
from functools import partial
from functools import reduce
# from itertools import chain
from pathlib import Path
import multiprocessing

import matplotlib.pyplot as plt
import scipy.signal
# import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import artistools as at

# from astropy import constants as const

colors_tab10 = list(plt.get_cmap('tab10')(np.linspace(0, 1.0, 10)))

# reserve colours for these elements
elementcolors = {
    'Fe': colors_tab10[0],
    'Ni': colors_tab10[1],
    'Co': colors_tab10[2],
}

variableunits = {
    'time': 'days',
    'TR': 'K',
    'Te': 'K',
    'TJ': 'K',
    'nne': 'e-/cm3',
    'heating': 'erg/s/cm3',
    'cooling': 'erg/s/cm3',
    'velocity': 'km/s',
    'velocity_outer': 'km/s',
}

variablelongunits = {
    'TR': 'Temperature [K]',
    'Te': 'Temperature [K]',
    'TJ': 'Temperature [K]',
}

dictlabelreplacements = {
    'lognne': 'Log nne',
    'Te': 'T$_e$',
    'TR': 'T$_R$'
}


def get_elemcolor(atomic_number=None, elsymbol=None):
    """Get the colour of an element from the reserved color list (reserving a new one if needed)."""
    assert (atomic_number is None) != (elsymbol is None)
    if atomic_number is not None:
        elsymbol = at.elsymbols[atomic_number]

    # assign a new colour to this element
    if elsymbol not in elementcolors:
        elementcolors[elsymbol] = colors_tab10[len(elementcolors)]

    return elementcolors[elsymbol]


def moving_average(arr, n):
    arr_padded = np.pad(arr, (n // 2, n - 1 - n // 2), mode='edge')
    return np.convolve(arr_padded, np.ones((n,)) / n, mode='valid')


def apply_filters(xlist, ylist, args):
    if args.filtermovingavg > 0:
        ylist = moving_average(ylist, args.filtermovingavg)

    if args.filtersavgol:
        window_length, polyorder = [int(x) for x in args.filtersavgol]

        ylist = scipy.signal.savgol_filter(ylist, window_length=window_length, polyorder=polyorder,
                                           mode='nearest')

    return xlist, ylist


def get_ionrecombrates_fromfile(filename):
    """WARNING: copy pasted from artis-atomic! replace with a package import soon ionstage is the lower ion stage."""
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


def parse_estimfile(estfilepath, modelpath, get_ion_values=True, get_heatingcooling=True):
    """Generate timestep, modelgridindex, dict from estimator file."""
    itstep = at.get_inputparams(modelpath)['itstep']

    with at.zopen(estfilepath, 'rt') as estimfile:
        timestep = -1
        modelgridindex = -1
        estimblock = {}
        for line in estimfile:
            row = line.split()
            if not row:
                continue

            if row[0] == 'timestep':
                # yield the previous block before starting a new one
                if timestep >= 0 and modelgridindex >= 0:
                    yield timestep, modelgridindex, estimblock

                timestep = int(row[1])
                if timestep > itstep:
                    print(f"Dropping estimator data from timestep {timestep} and later (> itstep {itstep})")
                    # itstep in input.txt is updated by ARTIS at every timestep, so the data beyond here
                    # could be half-written to disk and cause parsing errors
                    return

                modelgridindex = int(row[3])
                # print(f'Timestep {timestep} cell {modelgridindex}')

                estimblock = {}
                emptycell = (row[4] == 'EMPTYCELL')
                estimblock['emptycell'] = emptycell
                if not emptycell:
                    # will be TR, Te, W, TJ, nne
                    for variablename, value in zip(row[4::2], row[5::2]):
                        estimblock[variablename] = float(value)
                    estimblock['lognne'] = math.log10(estimblock['nne'])

            elif row[1].startswith('Z=') and get_ion_values:
                variablename = row[0]
                if row[1].endswith('='):
                    atomic_number = int(row[2])
                    startindex = 3
                else:
                    atomic_number = int(row[1].split('=')[1])
                    startindex = 2

                if variablename not in estimblock:
                    estimblock[variablename] = {}

                for ion_stage_str, value in zip(row[startindex::2], row[startindex + 1::2]):
                    if ion_stage_str.strip() in ['SUM:', '(or']:
                        continue

                    try:
                        ion_stage = int(ion_stage_str.rstrip(':'))
                    except ValueError:
                        print(f'Cannot parse row: {row}')
                        continue

                    value_thision = float(value.rstrip(','))

                    estimblock[variablename][(atomic_number, ion_stage)] = value_thision

                    if variablename in ['Alpha_R*nne', 'AlphaR*nne']:
                        if 'Alpha_R' not in estimblock:
                            estimblock['Alpha_R'] = {}
                        estimblock['Alpha_R'][(atomic_number, ion_stage)] = value_thision / estimblock['nne']
                    else:  # variablename == 'populations':
                        # contribute the ion population to the element population
                        if atomic_number not in estimblock[variablename]:
                            estimblock[variablename][atomic_number] = 0.
                        estimblock[variablename][atomic_number] += value_thision

                if variablename == 'populations':
                    # contribute the element population to the total population
                    if 'total' not in estimblock['populations']:
                        estimblock['populations']['total'] = 0.
                    estimblock['populations']['total'] += estimblock['populations'][atomic_number]

            elif row[0] == 'heating:' and get_heatingcooling:
                for heatingtype, value in zip(row[1::2], row[2::2]):
                    key = 'heating_' + heatingtype if not heatingtype.startswith('heating_') else heatingtype
                    estimblock[key] = float(value)

                if 'heating_gamma/gamma_dep' in estimblock and estimblock['heating_gamma/gamma_dep'] > 0:
                    estimblock['gamma_dep'] = (
                        estimblock['heating_gamma'] /
                        estimblock['heating_gamma/gamma_dep'])
                elif 'heating_dep/total_dep' in estimblock and estimblock['heating_dep/total_dep'] > 0:
                    estimblock['total_dep'] = (
                        estimblock['heating_dep'] /
                        estimblock['heating_dep/total_dep'])

            elif row[0] == 'cooling:' and get_heatingcooling:
                for coolingtype, value in zip(row[1::2], row[2::2]):
                    estimblock['cooling_' + coolingtype] = float(value)

    # reached the end of file
    if timestep >= 0 and modelgridindex >= 0:
        yield timestep, modelgridindex, estimblock


@at.diskcache(ignorekwargs=['printfilename'])
def read_estimators_from_file(modelpath, folderpath, arr_velocity_outer, mpirank, printfilename=False,
                              get_ion_values=True, get_heatingcooling=True):
    estimators_thisfile = {}
    estimfilename = f'estimators_{mpirank:04d}.out'
    estfilepath = Path(folderpath, estimfilename)
    if not estfilepath.is_file():
        estfilepath = Path(folderpath, estimfilename + '.gz')
        if not estfilepath.is_file():
            print(f'Warning: Could not find {estfilepath.relative_to(modelpath.parent)}')
            return {}

    if printfilename:
        filesize = Path(estfilepath).stat().st_size / 1024 / 1024
        print(f'Reading {estfilepath.relative_to(modelpath.parent)} ({filesize:.2f} MiB)')

    for fileblock_timestep, fileblock_modelgridindex, file_estimblock in parse_estimfile(
            estfilepath, modelpath, get_ion_values=get_ion_values, get_heatingcooling=get_heatingcooling):

        file_estimblock['velocity_outer'] = arr_velocity_outer[fileblock_modelgridindex]
        file_estimblock['velocity'] = file_estimblock['velocity_outer']

        estimators_thisfile[(fileblock_timestep, fileblock_modelgridindex)] = file_estimblock

    return estimators_thisfile


@lru_cache(maxsize=16)
@at.diskcache(savegzipped=True)
def read_estimators(modelpath, modelgridindex=None, timestep=None, get_ion_values=True, get_heatingcooling=True):
    """Read estimator files into a nested dictionary structure.

    Speed it up by only retrieving estimators for a particular timestep(s) or modelgrid cells.
    """
    if modelgridindex is None:
        match_modelgridindex = []
    elif hasattr(modelgridindex, '__iter__'):
        match_modelgridindex = tuple(modelgridindex)
    else:
        match_modelgridindex = (modelgridindex,)

    if -1 in match_modelgridindex:
        match_modelgridindex = []

    if timestep is None:
        match_timestep = []
    else:
        match_timestep = tuple(timestep) if hasattr(timestep, '__iter__') else (timestep,)

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")

    modeldata, _ = at.get_modeldata(modelpath)
    arr_velocity_outer = tuple(list([float(v) for v in modeldata['velocity_outer'].values]))

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=match_modelgridindex)

    printfilename = len(mpiranklist) < 10

    estimators = {}
    for folderpath in at.get_runfolders(modelpath, timesteps=match_timestep):
        print(f'Reading {len(list(mpiranklist))} estimator files in {folderpath.relative_to(modelpath.parent)}')

        processfile = partial(read_estimators_from_file, modelpath, folderpath, arr_velocity_outer,
                              get_ion_values=get_ion_values, get_heatingcooling=get_heatingcooling,
                              printfilename=printfilename)

        if at.enable_multiprocessing:
            with multiprocessing.get_context("spawn").Pool() as pool:
                arr_rankestimators = pool.map(processfile, mpiranklist)
                pool.close()
                pool.join()
        else:
            arr_rankestimators = [processfile(rank) for rank in mpiranklist]

        for estimators_thisfile in arr_rankestimators:
            estimators.update(estimators_thisfile)

    return estimators


def get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, keys, avgadjcells=0):
    """Get the average of estimators[(timestep, modelgridindex)][keys[0]]...[keys[-1]] across timesteps."""
    if isinstance(keys, str):
        keys = [keys]

    # reduce(lambda d, k: d[k], keys, dictionary) returns dictionary[keys[0]][keys[1]]...[keys[-1]]
    # applying all keys in the keys list

    # if single timestep, no averaging needed
    if not hasattr(timesteps, '__iter__'):
        return reduce(lambda d, k: d[k], [(timesteps, modelgridindex)] + keys, estimators)

    firsttimestepvalue = reduce(lambda d, k: d[k], [(timesteps[0], modelgridindex)] + keys, estimators)
    if isinstance(firsttimestepvalue, dict):
        dictout = {k: get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, keys + [k])
                   for k in firsttimestepvalue.keys()}

        return dictout
    else:
        tdeltas = at.get_timestep_times_float(modelpath, loc='delta')
        valuesum = 0
        tdeltasum = 0
        for timestep, tdelta in zip(timesteps, tdeltas):
            for mgi in range(modelgridindex - avgadjcells, modelgridindex + avgadjcells + 1):
                try:
                    valuesum += reduce(lambda d, k: d[k], [(timestep, mgi)] + keys, estimators) * tdelta
                    tdeltasum += tdelta
                except KeyError:
                    pass
        return valuesum / tdeltasum

    # except KeyError:
    #     if (timestep, modelgridindex) in estimators:
    #         print(f'Unknown x variable: {xvariable} for timestep {timestep} in cell {modelgridindex}')
    #     else:
    #         print(f'No data for cell {modelgridindex} at timestep {timestep}')
    #     print(estimators[(timestep, modelgridindex)])
    #     sys.exit()


def plot_init_abundances(ax, xlist, specieslist, mgilist, modelpath, dfalldata=None, args=None, **plotkwargs):
    assert len(xlist) - 1 == len(mgilist)
    modeldata, _ = at.get_modeldata(modelpath)
    abundancedata = at.get_initialabundances(modelpath)

    ax.set_ylim(0., 1.0)
    for speciesstr in specieslist:
        splitvariablename = speciesstr.split('_')
        elsymbol = splitvariablename[0].strip('0123456789')
        atomic_number = at.get_atomic_number(elsymbol)
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
                yvalue = abundancedata.loc[modelgridindex][f'X_{elsymbol}'] - modeldata.loc[modelgridindex]['X_Ni56']
                linelabel = 'Stable Ni'
            elif speciesstr.lower() in ['co_56', 'co56', '56co']:
                yvalue = modeldata.loc[modelgridindex]['X_Co56']
                linelabel = '$^{56}$Co'
            elif speciesstr.lower() in ['fegrp', 'ffegroup']:
                yvalue = modeldata.loc[modelgridindex]['X_Fegroup']
            else:
                yvalue = abundancedata.loc[modelgridindex][f'X_{elsymbol}']
            ylist.append(yvalue)

        if dfalldata is not None:
            dfalldata['initabundances.' + speciesstr] = ylist

        ylist.insert(0, ylist[0])
        # or ax.step(where='pre', )
        color = get_elemcolor(atomic_number=atomic_number)

        xlist, ylist = apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, linewidth=1.5, label=linelabel, linestyle=linestyle, color=color, **plotkwargs)


def get_averageionisation(populations, atomic_number):
    free_electron_weighted_pop_sum = 0.
    found = False
    popsum = 0
    for key in populations.keys():
        if isinstance(key, tuple) and key[0] == atomic_number:
            found = True
            ion_stage = key[1]
            free_electron_weighted_pop_sum += populations[key] * (ion_stage - 1)
            popsum += populations[key]

    assert(found)
    return free_electron_weighted_pop_sum / populations[atomic_number]


def plot_averageionisation(
        ax, xlist, elementlist, timestepslist, mgilist, estimators, modelpath, dfalldata=None, args=None, **plotkwargs):
    ax.set_ylabel('Average ionisation')
    for elsymb in elementlist:
        atomic_number = at.get_atomic_number(elsymb)
        ylist = []
        for modelgridindex, timesteps in zip(mgilist, timestepslist):
            valuesum = 0
            tdeltasum = 0
            for timestep in timesteps:
                tdelta = at.get_timestep_time_delta(timestep, modelpath=modelpath)
                valuesum += (
                    get_averageionisation(estimators[(timestep, modelgridindex)]['populations'],
                                          atomic_number) * tdelta)
                tdeltasum += tdelta

            ylist.append(valuesum / tdeltasum)

        color = get_elemcolor(atomic_number=atomic_number)

        if dfalldata is not None:
            dfalldata['averageionisation.' + elsymb] = ylist

        ylist.insert(0, ylist[0])

        xlist, ylist = apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, label=elsymb, color=color, **plotkwargs)


def plot_multi_ion_series(
        ax, xlist, seriestype, ionlist, timestepslist, mgilist, estimators,
        modelpath, dfalldata=None, args=None, **plotkwargs):
    """Plot an ion-specific property, e.g., populations."""
    assert len(xlist) - 1 == len(mgilist) == len(timestepslist)
    # if seriestype == 'populations':
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.10))

    plotted_something = False

    compositiondata = at.get_composition_data(modelpath)

    # decoded into numeric form, e.g., [(26, 1), (26, 2)]
    iontuplelist = [
        (at.get_atomic_number(ionstr.split(' ')[0]), at.decode_roman_numeral(ionstr.split(' ')[1]))
        if ' ' in ionstr else (at.get_atomic_number(ionstr), 'ALL')
        for ionstr in ionlist]
    iontuplelist.sort()
    print(f'Subplot with ions: {iontuplelist}')

    missingions = set()
    for atomic_number, ion_stage in iontuplelist:
        if ion_stage != 'ALL' and compositiondata.query(
                'Z == @atomic_number '
                '& lowermost_ionstage <= @ion_stage '
                '& uppermost_ionstage >= @ion_stage').empty:
            missingions.add((atomic_number, ion_stage))

    if missingions:
        print(f" Warning: Can't plot {seriestype} for {missingions} "
              f"because these ions are not in compositiondata.txt")

    prev_atomic_number = iontuplelist[0][0]
    colorindex = 0
    for atomic_number, ion_stage in iontuplelist:
        if (atomic_number, ion_stage) in missingions:
            continue

        if atomic_number != prev_atomic_number:
            colorindex += 1

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
        for modelgridindex, timesteps in zip(mgilist, timestepslist):
            if seriestype == 'populations':
                # if (atomic_number, ion_stage) not in estim['populations']:
                #     print(f'Note: population for {(atomic_number, ion_stage)} not in estimators for '
                #           f'cell {modelgridindex} timesteps {timesteps}')

                try:
                    estimpop = get_averaged_estimators(
                        modelpath, estimators, timesteps, modelgridindex, ['populations'])
                except KeyError:
                    ylist.append(float('nan'))
                    continue
                if ion_stage == 'ALL':
                    nionpop = estimpop.get((atomic_number), 0.)
                else:
                    nionpop = estimpop.get((atomic_number, ion_stage), 0.)

                try:
                    if args.ionpoptype == 'absolute':
                        yvalue = nionpop  # Plot as fraction of element population
                    elif args.ionpoptype == 'elpop':
                        elpop = estimpop.get(atomic_number, 0.)
                        yvalue = nionpop / elpop  # Plot as fraction of element population
                    elif args.ionpoptype == 'totalpop':
                        totalpop = estimpop['total']
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
                # this is very slow!
                try:
                    estim = get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, [])
                except KeyError:
                    ylist.append(float('nan'))
                    continue

                dictvars = {}
                for k, value in estim.items():
                    if isinstance(value, dict):
                        dictvars[k] = value.get((atomic_number, ion_stage), 0.)
                    else:
                        dictvars[k] = value

                # dictvars will now define things like 'Te', 'TR',
                # as well as 'populations' which applies to the current ion

                try:
                    yvalue = eval(seriestype, {"__builtins__": math}, dictvars)
                except ZeroDivisionError:
                    yvalue = float('NaN')
                ylist.append(yvalue)

        plotlabel = at.get_ionstring(atomic_number, ion_stage, spectral=False)

        # linestyle = ['-.', '-', '--', (0, (4, 1, 1, 1)), ':'] + [(0, x) for x in dashes_list][ion_stage - 1]
        if ion_stage == 'ALL':
            dashes = ()
            linewidth = 1.0
        else:
            dashes = [(3, 1, 1, 1), (), (1.5, 1.5), (6, 3), (1, 3)][ion_stage - 1]
            linewidth = [1.0, 1.0, 1.0, 0.7, 0.7][ion_stage - 1]
            # color = ['blue', 'green', 'red', 'cyan', 'purple', 'grey', 'brown', 'orange'][ion_stage - 1]

        # assert colorindex < 10
        # color = f'C{colorindex}'
        color = get_elemcolor(atomic_number=atomic_number)
        # or ax.step(where='pre', )

        if args.colorbyion:
            color = f'C{ion_stage - 1}'
            plotlabel = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]}'
            dashes = ()

        if dfalldata is not None:
            dfalldata[f'{seriestype}.{args.ionpoptype}.{atomic_number}.{ion_stage}'] = ylist

        ylist.insert(0, ylist[0])

        xlist, ylist = apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, linewidth=linewidth, label=plotlabel, color=color, dashes=dashes, **plotkwargs)
        prev_atomic_number = atomic_number
        plotted_something = True

    if plotted_something:
        if args.yscale == 'log':
            ax.set_yscale('log')
            ymin, ymax = ax.get_ylim()
            new_ymax = ymax * 10 ** (0.3 * math.log10(ymax / ymin))
            if ymin > 0 and new_ymax > ymin and np.isfinite(new_ymax):
                ax.set_ylim(ymin, new_ymax)


def plot_series(ax, xlist, variablename, showlegend, timestepslist, mgilist,
                modelpath, estimators, args, nounits=False, dfalldata=None, **plotkwargs):
    """Plot something like Te or TR."""
    assert len(xlist) - 1 == len(mgilist) == len(timestepslist)
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
    for modelgridindex, timesteps in zip(mgilist, timestepslist):
        estimavg = get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, [])
        try:
            ylist.append(eval(variablename, {"__builtins__": math}, estimavg))
        except KeyError:
            if (timesteps[0], modelgridindex) in estimators:
                print(f"Undefined variable: {variablename} in cell {modelgridindex}")
            else:
                print(f'No data for cell {modelgridindex}')
            sys.exit()

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

    if dfalldata is not None:
        dfalldata[variablename] = ylist

    ylist.insert(0, ylist[0])

    xlist, ylist = apply_filters(xlist, ylist, args)

    ax.plot(xlist, ylist, linewidth=1.5, label=linelabel, color=dictcolors.get(variablename, None), **plotkwargs)


def get_xlist(xvariable, allnonemptymgilist, estimators, timestepslist, modelpath, args):
    if xvariable in ['cellid', 'modelgridindex']:
        if args.xmax >= 0:
            mgilist_out = [mgi for mgi in allnonemptymgilist if mgi <= args.xmax]
        else:
            mgilist_out = allnonemptymgilist
        xlist = mgilist_out
        timestepslist_out = timestepslist
    elif xvariable == 'timestep':
        mgilist_out = allnonemptymgilist
        xlist = timestepslist
        timestepslist_out = timestepslist
    elif xvariable == 'time':
        mgilist_out = allnonemptymgilist
        timearray = at.get_timestep_times_float(modelpath)
        xlist = [np.mean([timearray[ts] for ts in tslist]) for tslist in timestepslist]
        timestepslist_out = timestepslist
    else:
        xlist = []
        mgilist_out = []
        timestepslist_out = []
        for modelgridindex, timesteps in zip(allnonemptymgilist, timestepslist):
            xvalue = get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, xvariable)
            xlist.append(xvalue)
            mgilist_out.append(modelgridindex)
            timestepslist_out.append(timesteps)
            if args.xmax > 0 and xvalue > args.xmax:
                break

    xlist, mgilist_out, timestepslist_out = zip(
        *[xmt for xmt in sorted(zip(xlist, mgilist_out, timestepslist_out))])

    assert len(xlist) == len(mgilist_out) == len(timestepslist_out)

    return list(xlist), list(mgilist_out), list(timestepslist_out)


def plot_subplot(ax, timestepslist, xlist, plotitems, mgilist, modelpath,
                 estimators, dfalldata=None, args=None, **plotkwargs):
    """Make plot from ARTIS estimators."""
    # these three lists give the x value, modelgridex, and a list of timesteps (for averaging) for each plot of the plot
    assert len(xlist) - 1 == len(mgilist) == len(timestepslist)
    showlegend = False

    ylabel = 'UNDEFINED'
    sameylabel = True
    for variablename in plotitems:
        if not isinstance(variablename, str):
            pass
        elif ylabel == 'UNDEFINED':
            ylabel = get_ylabel(variablename)
        elif ylabel != get_ylabel(variablename):
            sameylabel = False
            break

    for plotitem in plotitems:
        if isinstance(plotitem, str):
            showlegend = len(plotitems) > 1 or len(variablename) > 20
            plot_series(ax, xlist, plotitem, showlegend, timestepslist, mgilist, modelpath,
                        estimators, args, nounits=sameylabel, dfalldata=dfalldata, **plotkwargs)
            if showlegend and sameylabel:
                ax.set_ylabel(ylabel)
        else:  # it's a sequence of values
            showlegend = True
            seriestype, params = plotitem
            if seriestype == 'initabundances':
                plot_init_abundances(ax, xlist, params, mgilist, modelpath, dfalldata=dfalldata, args=args)
            elif seriestype == 'averageionisation':
                plot_averageionisation(ax, xlist, params, timestepslist, mgilist, estimators,
                                       modelpath, dfalldata=dfalldata, args=args)
            elif seriestype == '_ymin':
                ax.set_ylim(bottom=params)
            elif seriestype == '_ymax':
                ax.set_ylim(top=params)
            elif seriestype == '_yscale':
                ax.set_yscale(params)
            else:
                seriestype, ionlist = plotitem
                plot_multi_ion_series(ax, xlist, seriestype, ionlist, timestepslist, mgilist, estimators,
                                      modelpath, dfalldata, args, **plotkwargs)

    ax.tick_params(right=True)
    if showlegend:
        if plotitems[0][0] == 'populations' and args.yscale == 'log':
            ax.legend(loc='upper right', handlelength=2, ncol=math.ceil(len(plotitems[0][1]) / 2.),
                      frameon=False, numpoints=1)
        else:
            ax.legend(loc='upper right', handlelength=2,
                      frameon=False, numpoints=1,)  # prop={'size': 9})


def make_plot(modelpath, timestepslist_unfiltered, allnonemptymgilist, estimators, xvariable, plotlist,
              args, **plotkwargs):
    modelname = at.get_model_name(modelpath)
    fig, axes = plt.subplots(nrows=len(plotlist), ncols=1, sharex=True,
                             figsize=(args.figscale * at.figwidth * args.scalefigwidth,
                                      args.figscale * at.figwidth * 0.5 * len(plotlist)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    if len(plotlist) == 1:
        axes = [axes]

    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    axes[-1].set_xlabel(f'{xvariable}{get_units_string(xvariable)}')
    xlist, mgilist, timestepslist = get_xlist(
        xvariable, allnonemptymgilist, estimators, timestepslist_unfiltered, modelpath, args)

    dfalldata = pd.DataFrame(index=mgilist)
    dfalldata.index.name = "modelgridindex"
    dfalldata[xvariable] = xlist

    xlist = np.insert(xlist, 0, 0.)

    xmin = args.xmin if args.xmin > 0 else min(xlist)
    xmax = args.xmax if args.xmax > 0 else max(xlist)

    for ax, plotitems in zip(axes, plotlist):
        ax.set_xlim(left=xmin, right=xmax)
        ax.tick_params(which='both', direction='in')
        plot_subplot(ax, timestepslist, xlist, plotitems, mgilist,
                     modelpath, estimators, dfalldata=dfalldata, args=args, **plotkwargs)

    if len(set(mgilist)) == 1:  # single grid cell plot
        figure_title = f'{modelname}\nCell {mgilist[0]}'

        defaultoutputfile = Path('plotestimators_cell{modelgridindex:03d}.pdf')
        if os.path.isdir(args.outputfile):
            args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

        outfilename = str(args.outputfile).format(modelgridindex=mgilist[0])

    else:
        timeavg = (args.timemin + args.timemax) / 2.
        if args.multiplot:
            tdays = estimators[(timestepslist[0][0], mgilist[0])]['tdays']
            figure_title = f'{modelname}\nTimestep {timestepslist[0]} ({tdays:.2f}d)'
        else:
            figure_title = f'{modelname}\nTimestep {timestepslist[0]} ({timeavg:.2f}d)'

        defaultoutputfile = Path('plotestimators_ts{timestep:02d}_{timeavg:.0f}d.pdf')
        if os.path.isdir(args.outputfile):
            args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

        outfilename = str(args.outputfile).format(timestep=timestepslist[0][0], timeavg=timeavg)

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=11)
    # plt.suptitle(figure_title, fontsize=11, verticalalignment='top')

    if args.write_data:
        dfalldata.sort_index(inplace=True)
        dataoutfilename = Path(outfilename).with_suffix('.txt')
        dfalldata.to_csv(dataoutfilename)
        print(f'Saved {dataoutfilename}')

    fig.savefig(outfilename, format='pdf')
    print(f'Saved {outfilename}')

    if args.show:
        plt.show()
    else:
        plt.close()

    return outfilename


def plot_recombrates(modelpath, estimators, atomic_number, ion_stage_list, **plotkwargs):
    fig, axes = plt.subplots(
        nrows=len(ion_stage_list), ncols=1, sharex=True, figsize=(5, 8),
        tight_layout={"pad": 0.5, "w_pad": 0.0, "h_pad": 0.0})
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axes[-1].set_xlabel(f'T_e in kelvins')

    recombcalibrationdata = at.get_ionrecombratecalibration(modelpath)

    for ax, ion_stage in zip(axes, ion_stage_list):

        ionstr = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]} to {at.roman_numerals[ion_stage - 1]}'

        listT_e = []
        list_rrc = []
        list_rrc2 = []
        for _, dicttimestepmodelgrid in estimators.items():
            if (not dicttimestepmodelgrid['emptycell']
                    and (atomic_number, ion_stage) in dicttimestepmodelgrid['RRC_LTE_Nahar']):
                listT_e.append(dicttimestepmodelgrid['Te'])
                list_rrc.append(dicttimestepmodelgrid['RRC_LTE_Nahar'][(atomic_number, ion_stage)])
                list_rrc2.append(dicttimestepmodelgrid['Alpha_R'][(atomic_number, ion_stage)])

        if not list_rrc:
            continue

        # sort the pairs by temperature ascending
        listT_e, list_rrc, list_rrc2 = zip(*sorted(zip(listT_e, list_rrc, list_rrc2), key=lambda x: x[0]))

        # markersize=4, marker='s',
        ax.plot(listT_e, list_rrc, linewidth=2, label=f'{ionstr} ARTIS RRC_LTE_Nahar', **plotkwargs)
        ax.plot(listT_e, list_rrc2, linewidth=2, label=f'{ionstr} ARTIS Alpha_R', **plotkwargs)

        try:
            dfrates = recombcalibrationdata[(atomic_number, ion_stage)].query(
                "T_e > @T_e_min & T_e < @T_e_max",
                local_dict={'T_e_min': min(listT_e), 'T_e_max': max(listT_e)})

            ax.plot(dfrates.T_e, dfrates.rrc_total, linewidth=2,
                    label=ionstr + " (calibration)", markersize=6, marker='s', **plotkwargs)
        except KeyError:
            pass

        # rrcfiles = glob.glob(
        #     f'/Users/lshingles/Library/Mobile Documents/com~apple~CloudDocs/GitHub/'
        #     f'artis-atomic/atomic-data-nahar/{at.elsymbols[atomic_number].lower()}{ion_stage - 1}.rrc*.txt')
        # if rrcfiles:
        #     dfrecombrates = get_ionrecombrates_fromfile(rrcfiles[0])
        #
        #     dfrecombrates.query("logT > @logT_e_min & logT < @logT_e_max",
        #                         local_dict={'logT_e_min': math.log10(min(listT_e)),
        #                                     'logT_e_max': math.log10(max(listT_e))}, inplace=True)
        #
        #     listT_e_Nahar = [10 ** x for x in dfrecombrates['logT'].values]
        #     ax.plot(listT_e_Nahar, dfrecombrates['RRC_total'], linewidth=2,
        #             label=ionstr + " (Nahar)", markersize=6, marker='s', **plotkwargs)

        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})

    # modelname = at.get_model_name(".")
    # plotlabel = f'Timestep {timestep}'
    # time_days = float(at.get_timestep_time('spec.out', timestep))
    # if time_days >= 0:
    #     plotlabel += f' (t={time_days:.2f}d)'
    # fig.suptitle(plotlabel, fontsize=12)
    elsymbol = at.elsymbols[atomic_number]
    outfilename = f"plotestimators_recombrates_{elsymbol}.pdf"
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

    parser.add_argument('-timestep', '-ts', nargs='?',
                        help='Timestep number for internal structure plot')

    parser.add_argument('-timedays', '-time', '-t', nargs='?',
                        help='Time in days to plot for internal structure plot')

    parser.add_argument('-timemin', type=float,
                        help='Lower time in days')

    parser.add_argument('-timemax', type=float,
                        help='Upper time in days')

    parser.add_argument('-multiplot', action='store_true',
                        help='Make multiple plots for timesteps in range')

    parser.add_argument('-x',
                        help='Horizontal axis variable, e.g. cellid, velocity, timestep, or time')

    parser.add_argument('-xmin', type=int, default=-1,
                        help='Plot range: minimum x value')

    parser.add_argument('-xmax', type=int, default=-1,
                        help='Plot range: maximum x value')

    parser.add_argument('-yscale', default='log', choices=['log', 'linear'],
                        help='Set yscale to log or linear (default log)')

    parser.add_argument('-filtermovingavg', type=int, default=0,
                        help='Smoothing length (1 is same as none)')

    parser.add_argument('-filtersavgol', nargs=2,
                        help='Savitzky–Golay filter. Specify the window_length and polyorder.'
                        'e.g. -filtersavgol 5 3')

    parser.add_argument('--notitle', action='store_true',
                        help='Suppress the top title from the plot')

    parser.add_argument('-plotlist', type=list, default=[],
                        help='Plot list (when calling from Python only)')

    parser.add_argument('-ionpoptype', default='elpop', choices=['absolute', 'totalpop', 'elpop'],
                        help=(
                            'Plot absolutely ion populations, or ion populations as a'
                            ' fraction of total or element population'))

    parser.add_argument('-figscale', type=float, default=1.,
                        help='Scale factor for plot area. 1.0 is for single-column')

    parser.add_argument('-scalefigwidth', type=float, default=1.,
                        help='Scale factor for plot width.')

    parser.add_argument('-show', action='store_true',
                        help='Show plot before quitting')

    parser.add_argument('--write_data', action='store_true',
                        help='Save data used to generate the plot in a CSV file')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path, default=Path(),
                        help='Filename for PDF file')

    parser.add_argument('-colorbyion', action='store_true',
                        help='Populations plots colored by ion rather than element')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS estimators.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    modelpath = Path(args.modelpath)

    modelname = at.get_model_name(modelpath)

    if not args.timedays and not args.timestep and args.modelgridindex > -1:
        args.timestep = f'0-{len(at.get_timestep_times(modelpath)) - 1}'

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
         modelpath, args.timestep, args.timemin, args.timemax, args.timedays)

    print(f"Plotting estimators for '{modelname}' timesteps {timestepmin} to {timestepmax} "
          f"({args.timemin:.1f} to {args.timemax:.1f}d)")

    timesteps_included = list(range(timestepmin, timestepmax + 1))
    estimators = read_estimators(modelpath, modelgridindex=args.modelgridindex, timestep=tuple(timesteps_included))

    for ts in reversed(timesteps_included):
        if (ts, 0) not in estimators:
            timesteps_included.remove(ts)

    if not timesteps_included:
        print("No timesteps with data are included")
        return

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
            [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Fe VI', 'Fe VII', 'Fe VIII']]],
            [['populations', ['Co I', 'Co II', 'Co III', 'Co IV', 'Co V', 'Co VI', 'Co VII']]],
            [['populations', ['Ni I', 'Ni II', 'Ni III', 'Ni IV', 'Ni V', 'Ni VI', 'Ni VII']]],
            # [['populations', ['Fe II', 'Fe III', 'Co II', 'Co III', 'Ni II', 'Ni III']]],
            # [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
            [['RRC_LTE_Nahar', ['Fe II', 'Fe III', 'Fe IV', 'Fe V']]],
            [['RRC_LTE_Nahar', ['Co II', 'Co III', 'Co IV', 'Co V']]],
            [['RRC_LTE_Nahar', ['Ni I', 'Ni II', 'Ni III', 'Ni IV', 'Ni V', 'Ni VI', 'Ni VII']]],
            # [['Alpha_R / RRC_LTE_Nahar', ['Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni III']]],
            # [['gamma_NT', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
        ]

    if args.recombrates:
        plot_recombrates(modelpath, estimators, 26, [2, 3, 4, 5])
        plot_recombrates(modelpath, estimators, 27, [3, 4])
        plot_recombrates(modelpath, estimators, 28, [3, 4, 5])
    else:
        modeldata, _ = at.get_modeldata(modelpath)
        allnonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                              if not estimators[(timesteps_included[0], modelgridindex)]['emptycell']]

        if args.modelgridindex > -1 or args.x == 'time':
            # plot time evolution in specific cell
            if not args.x:
                args.x = 'time'
            mgilist = [args.modelgridindex] * len(timesteps_included)
            timesteplist_unfiltered = [(ts,) for ts in timesteps_included]
            make_plot(modelpath, timesteplist_unfiltered, mgilist, estimators, args.x, plotlist, args)
        else:
            # plot a range of cells in a time snapshot showing internal structure

            if not args.x:
                args.x = 'velocity_outer'

            if args.multiplot:
                pdf_list = []
                modelpath_list = []
                for timestep in range(timestepmin, timestepmax + 1):
                    timesteplist_unfiltered = [[timestep]] * len(allnonemptymgilist)
                    outfilename = make_plot(modelpath, timesteplist_unfiltered, allnonemptymgilist, estimators, args.x,
                                            plotlist, args)

                    if '/' in outfilename:
                        outfilename = outfilename.split('/')[1]

                    pdf_list.append(outfilename)
                    modelpath_list.append(modelpath)

                if len(pdf_list) > 1:
                    at.join_pdf_files(pdf_list, modelpath_list)

            else:
                timesteplist_unfiltered = [timesteps_included] * len(allnonemptymgilist)
                make_plot(modelpath, timesteplist_unfiltered, allnonemptymgilist, estimators, args.x, plotlist, args)


if __name__ == "__main__":
    main()
