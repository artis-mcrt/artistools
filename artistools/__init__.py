#!/usr/bin/env python3
import math
import os.path

# import scipy.signal
# import numpy as np
import pandas as pd
from astropy import constants as const
# from astropy import units as u
from collections import namedtuple

import artistools.lightcurves
import artistools.packets
import artistools.plot
import artistools.radfield
import artistools.spectra

PYDIR = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII',
                  'XVIII', 'XIX', 'XX')


def showtimesteptimes(specfilename, numberofcolumns=5):
    """
        Print a table showing the timeteps and their corresponding times
    """
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    print('Time steps and corresponding times in days:\n')

    times = specdata.columns
    indexendofcolumnone = math.ceil((len(times) - 1) / numberofcolumns)
    for rownum in range(0, indexendofcolumnone):
        strline = ""
        for colnum in range(numberofcolumns):
            if colnum > 0:
                strline += '\t'
            newindex = rownum + colnum * indexendofcolumnone
            if newindex < len(times):
                strline += f'{newindex:4d}: {float(times[newindex + 1]):.3f}'
        print(strline)


def get_composition_data(filename):
    """
        Return a pandas DataFrame containing details of included
        elements and ions
    """

    columns = ('Z,nions,lowermost_ionstage,uppermost_ionstage,nlevelsmax_readin,'
               'abundance,mass,startindex').split(',')

    compdf = pd.DataFrame()

    with open(filename, 'r') as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline()  # T_preset
        fcompdata.readline()  # homogeneous_abundances
        startindex = 0
        for _ in range(nelements):
            line = fcompdata.readline()
            linesplit = line.split()
            row_list = list(map(int, linesplit[:5])) + list(map(float, linesplit[5:])) + [startindex]

            rowdf = pd.DataFrame([row_list], columns=columns)
            compdf = compdf.append(rowdf, ignore_index=True)

            startindex += int(rowdf['nions'])

    return compdf


def get_modeldata(filename):
    """
        Return a list containing named tuples for all model grid cells
    """
    modeldata = pd.DataFrame()
    gridcelltuple = namedtuple('gridcell', 'cellid velocity logrho ffe fni fco f52fe f48cr')

    with open(filename, 'r') as fmodel:
        gridcellcount = int(fmodel.readline())
        t_model_init = float(fmodel.readline())
        for line in fmodel:
            row = line.split()
            rowdf = pd.DataFrame([gridcelltuple._make([int(row[0]) - 1] + list(map(float, row[1:])))],
                                 columns=gridcelltuple._fields)
            modeldata = modeldata.append(rowdf)
    assert(len(modeldata) == gridcellcount)
    modeldata = modeldata.set_index(['cellid'])
    return modeldata, t_model_init


def get_initialabundances1d(filename):
    """
        Returns a list of mass fractions
    """
    abundancedata = []
    abundancedata.append([])
    with open(filename, 'r') as fabund:
        for line in fabund:
            row = line.split()
            abundancedata.append([int(row[0])] + list(map(float, row[1:])))

    return abundancedata


def get_timestep_times(specfilename):
    """
        Return a list of the time in days of each timestep using a spec.out file
    """
    time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)

    return time_columns.columns[1:]


def get_timestep_time(specfilename, timestep):
    """
        Return the time in days of a timestep number using a spec.out file
    """
    if os.path.isfile(specfilename):
        return get_timestep_times(specfilename)[timestep]
    else:
        return -1


def get_timestep_time_delta(timestep, timearray):
    """
        Return the time in days between timestep and timestep + 1
    """

    if timestep < len(timearray) - 1:
        delta_t = (float(timearray[timestep + 1]) - float(timearray[timestep]))
    else:
        delta_t = (float(timearray[timestep]) - float(timearray[timestep - 1]))

    return delta_t


def get_levels(adatafilename):
    """
        Return a list of lists of levels
    """
    level_lists = []
    iontuple = namedtuple('ion', 'Z ion_stage level_count ion_pot level_list')
    leveltuple = namedtuple('level', 'number energy_ev g transition_count levelname')

    with open(adatafilename, 'r') as fadata:
        for line in fadata:
            if len(line.strip()) > 0:
                ionheader = line.split()
                level_count = int(ionheader[2])

                level_list = []
                for _ in range(level_count):
                    line = fadata.readline()
                    row = line.split()
                    levelname = row[4].strip('\'')
                    level_list.append(leveltuple(int(row[0]), float(row[1]), float(row[2]), int(row[3]), levelname))

                level_lists.append(iontuple(int(ionheader[0]), int(ionheader[1]), level_count,
                                            float(ionheader[3]), list(level_list)))

    return level_lists


def get_nlte_populations(nltefile, modelgridindex, timestep, atomic_number, temperature_exc):
    all_levels = get_levels('adata.txt')

    dfpop = pd.read_csv(nltefile, delim_whitespace=True)
    dfpop.query('(modelgridindex==@modelgridindex) & (timestep==@timestep) & (Z==@atomic_number)',
                inplace=True)

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
            print(f'{elsymbols[atomic_number]} {roman_numerals[ion_stage]} has a superlevel at level {levelnumber}')
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


def get_nlte_populations_oldformat(nltefile, modelgridindex, timestep, atomic_number, temperature_exc):
    compositiondata = get_composition_data('compositiondata.txt')
    elementdata = compositiondata.query('Z==@atomic_number')

    if len(elementdata) < 1:
        print(f'Error: element Z={atomic_number} not in composition file')
        return None

    all_levels = get_levels('adata.txt')

    skip_block = False
    dfpop = pd.DataFrame().to_sparse()
    with open(nltefile, 'r') as nltefile:
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
        print(f'{elsymbols[atomic_number]} {roman_numerals[ion_stage]} has a superlevel at level {levelnumber}')
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


def get_model_name(path):
    """
        Get the name of an ARTIS model from the path to any file inside it
        either from a special plotlabel.txt file (if it exists)
        or the enclosing directory name
    """
    abspath = os.path.abspath(path)

    folderpath = abspath if os.path.isdir(abspath) else os.path.basename(os.path.dirname(os.path.abspath(path)))

    try:
        plotlabelfile = os.path.join(folderpath, 'plotlabel.txt')
        return (open(plotlabelfile, mode='r').readline().strip())
    except FileNotFoundError:
        return os.path.basename(folderpath)


def get_model_name_times(filename, timearray, timestep_range_str, timemin, timemax):
    if timestep_range_str:
        if '-' in timestep_range_str:
            timestepmin, timestepmax = [int(nts) for nts in timestep_range_str.split('-')]
        else:
            timestepmin = int(timestep_range_str)
            timestepmax = timestepmin
    else:
        if not timemin:
            timemin = 0.
        for timestep, time in enumerate(timearray):
            timefloat = float(time.strip('d'))
            if (timefloat >= timemin):
                timestepmin = timestep
                break

        if not timemax:
            timemax = float(timearray[-1].strip('d'))
        for timestep, time in enumerate(timearray):
            timefloat = float(time.strip('d'))
            if (timefloat + get_timestep_time_delta(timestep, timearray) <= timemax):
                timestepmax = timestep

    modelname = get_model_name(filename)

    time_days_lower = float(timearray[timestepmin])
    time_days_upper = float(timearray[timestepmax]) + get_timestep_time_delta(timestepmax, timearray)

    print(f'Plotting {modelname} timesteps {timestepmin} to {timestepmax} '
          f'(t={time_days_lower:.3f}d to {time_days_upper:.3f}d)')

    return modelname, timestepmin, timestepmax, time_days_lower, time_days_upper
