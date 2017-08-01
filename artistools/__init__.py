#!/usr/bin/env python3
import math
import os.path
import sys
# from astropy import units as u
from collections import namedtuple

# import scipy.signal
import numpy as np
import pandas as pd
from astropy import constants as const

from artistools import spectra

PYDIR = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'data', 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

console_scripts = [
    'artistools = artistools:show_help',
    'getartismodeldeposition = artistools.deposition:main',
    'makeartismodel1dslicefrom3d = artistools.slice3dmodel:main',
    'makeartismodelbotyanski = artistools.makemodelbotyanski:main',
    'plotartisestimators = artistools.estimators:main',
    'plotartislightcurve = artistools.lightcurve:main',
    'plotartisnltepops = artistools.nltepops:main',
    'plotartismacroatom = artistools.macroatom:main',
    'plotartisnonthermal = artistools.nonthermalspec:main',
    'plotartisradfield = artistools.radfield:main',
    'plotartisspectrum = artistools.spectra:main',
    'plotartistransitions = artistools.transitions:main',
]


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
    gridcelltuple = namedtuple('gridcell', 'cellid velocity logrho X_Fegroup X_Ni56 X_Co56 X_Fe52 X_Cr48')

    with open(filename, 'r') as fmodel:
        gridcellcount = int(fmodel.readline())
        t_model_init = float(fmodel.readline())
        for line in fmodel:
            row = line.split()
            rowdf = pd.DataFrame([gridcelltuple._make([int(row[0]) - 1] + list(map(float, row[1:])))],
                                 columns=gridcelltuple._fields)
            modeldata = modeldata.append(rowdf, ignore_index=True)

    assert(len(modeldata) == gridcellcount)
    modeldata = modeldata.set_index(['cellid'])
    return modeldata, t_model_init


def get_initialabundances1d(abundancefilename):
    """
        Returns a list of mass fractions
    """
    columns = ['inputcellid']
    columns.extend(['X_' + elsymbols[x] for x in range(1, 31)])
    abundancedata = pd.read_csv(abundancefilename, delim_whitespace=True, header=None, names=columns)
    abundancedata.index.name = 'modelgridindex'
    return abundancedata


def get_timestep_times(specfilename):
    """
        Return a list of the time in days of each timestep using a spec.out file
    """
    time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)

    return time_columns.columns[1:]


def get_timestep_times_float(specfilename):
    """
        Return a list of the time in days of each timestep using a spec.out file
    """
    return np.array([float(t.rstrip('d')) for t in get_timestep_times(specfilename)])


def get_closest_timestep(specfilename, timedays):
    return np.abs(get_timestep_times_float(specfilename) - float(timedays)).argmin()


def get_timestep_time(specfilename, timestep):
    """
        Return the time in days of a timestep number using a spec.out file
    """
    if os.path.isfile(specfilename):
        return get_timestep_times(specfilename)[timestep]
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


def get_model_name(path):
    """
        Get the name of an ARTIS model from the path to any file inside it
        either from a special plotlabel.txt file (if it exists)
        or the enclosing directory name
    """
    abspath = os.path.abspath(path)

    folderpath = abspath if os.path.isdir(abspath) else os.path.dirname(os.path.abspath(path))

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
        timestepmin = None
        if not timemin:
            timemin = 0.
        for timestep, time in enumerate(timearray):
            timefloat = float(time.strip('d'))
            if (timemin <= timefloat):
                timestepmin = timestep
                break

        if not timestepmin:
            print(f"Time min {timemin} is greater than all timesteps ({timearray[0]} to {timearray[-1]})")
            sys.exit()

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


def get_atomic_number(elsymbol):
    if elsymbol.title() in elsymbols:
        return elsymbols.index(elsymbol.title())
    return -1


def decode_roman_numeral(strin):
    if strin.upper() in roman_numerals:
        return roman_numerals.index(strin.upper())
    return -1


def show_help():
    print("artistools commands:")
    for script in sorted(console_scripts):
        command = script.split('=')[0].strip()
        print(f'  {command}')
