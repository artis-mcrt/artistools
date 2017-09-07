#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import math
import os.path
import sys
# from astropy import units as u
from collections import namedtuple

# import scipy.signal
import numpy as np
import pandas as pd
# from astropy import constants as const

PYDIR = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'data', 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

console_scripts = [
    'at = artistools:main',
    'artistools = artistools:main',
    'getartismodeldeposition = artistools.deposition:main',
    'makeartismodel1dslicefrom3d = artistools.slice3dmodel:main',
    'makeartismodelbotyanski = artistools.makemodelbotyanski:main',
    'plotartisestimators = artistools.estimators:main',
    'plotartislightcurve = artistools.lightcurve:main',
    'plotartisnltepops = artistools.nltepops:main',
    'plotartismacroatom = artistools.macroatom:main',
    'plotartisnonthermal = artistools.nonthermal:main',
    'plotartisradfield = artistools.radfield:main',
    'plotartisspectrum = artistools.spectra:main',
    'plotartistransitions = artistools.transitions:main',
]


def showtimesteptimes(specfilename, modelpath=None, numberofcolumns=5):
    """
        Print a table showing the timeteps and their corresponding times
    """
    if modelpath is not None:
        specfilename = os.path.join(modelpath, 'spec.out')
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
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'model.txt')
    modeldata = pd.DataFrame()
    gridcelltuple = namedtuple('gridcell', 'inputcellid velocity logrho X_Fegroup X_Ni56 X_Co56 X_Fe52 X_Cr48')

    with open(filename, 'r') as fmodel:
        gridcellcount = int(fmodel.readline())
        t_model_init_days = float(fmodel.readline())
        for line in fmodel:
            row = line.split()
            rowdf = pd.DataFrame([gridcelltuple._make([int(row[0])] + list(map(float, row[1:])))],
                                 columns=gridcelltuple._fields)
            modeldata = modeldata.append(rowdf, ignore_index=True)

    assert(len(modeldata) == gridcellcount)
    modeldata.index.name = 'cellid'
    return modeldata, t_model_init_days


def save_modeldata(dfmodeldata, t_model_init_days, filename) -> None:
    with open(filename, 'w') as fmodel:
        fmodel.write(f'{len(dfmodeldata)}\n{t_model_init_days:f}\n')
        for _, cell in dfmodeldata.iterrows():
            fmodel.write(f'{cell.inputcellid:6.0f}   {cell.velocity:9.2f}   {cell.logrho:10.8f} '
                         f'{cell.X_Fegroup:5.2f} {cell.X_Ni56:5.2f} {cell.X_Co56:5.2f} '
                         f'{cell.X_Fe52:5.2f} {cell.X_Cr48:5.2f}\n')


def get_initialabundances(abundancefilename):
    """Return a list of mass fractions."""
    columns = ['inputcellid', *['X_' + elsymbols[x] for x in range(1, 31)]]
    abundancedata = pd.read_csv(abundancefilename, delim_whitespace=True, header=None, names=columns)
    abundancedata.index.name = 'modelgridindex'
    return abundancedata


def save_initialabundances(dfabundances, abundancefilename) -> None:
    dfabundances['inputcellid'] = dfabundances['inputcellid'].astype(np.int)
    dfabundances.to_csv(abundancefilename, header=False, sep=' ', index=False)


def get_timestep_times(specfilename):
    """Return a list of the time in days of each timestep using a spec.out file."""
    if os.path.isdir(specfilename):
        specfilename = os.path.join(specfilename, 'spec.out')

    time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)

    return time_columns.columns[1:]


def get_timestep_times_float(specfilename):
    """Return a list of the time in days of each timestep using a spec.out file."""
    return np.array([float(t.rstrip('d')) for t in get_timestep_times(specfilename)])


def get_closest_timestep(specfilename, timedays):
    try:
        timedays_float = float(timedays.rstrip('d'))
    except AttributeError:
        timedays_float = float(timedays)
    return np.abs(get_timestep_times_float(specfilename) - timedays_float).argmin()


def get_timestep_time(specfilename, timestep):
    """Return the time in days of a timestep number using a spec.out file."""
    if os.path.isdir(specfilename):
        specfilename = os.path.join(specfilename, 'spec.out')

    if os.path.isfile(specfilename):
        return get_timestep_times(specfilename)[timestep]
    return -1


def get_timestep_time_delta(timestep, timearray):
    """Return the time in days between timestep and timestep + 1."""

    if timestep < len(timearray) - 1:
        delta_t = (float(timearray[timestep + 1]) - float(timearray[timestep]))
    else:
        delta_t = (float(timearray[timestep]) - float(timearray[timestep - 1]))

    return delta_t


def get_levels(modelpath, ionlist=None, get_transitions=False):
    """Return a list of lists of levels."""
    adatafilename = os.path.join(modelpath, 'adata.txt')

    transitiontuple = namedtuple('transition', 'lower upper A collstr forbidden')

    firstlevelnumber = 1

    transitionsdict = {}
    if get_transitions:
        transition_filename = os.path.join(modelpath, 'transitiondata.txt')

        print(f'Reading {transition_filename}')
        with open(transition_filename, 'r') as ftransitions:
            for line in ftransitions:
                if not line.strip():
                    continue

                ionheader = line.split()
                Z = int(ionheader[0])
                ionstage = int(ionheader[1])
                transition_count = int(ionheader[2])

                if not ionlist or (Z, ionstage) in ionlist:
                    translist = []
                    for _ in range(transition_count):
                        row = ftransitions.readline().split()
                        translist.append(
                            transitiontuple(int(row[0]) - firstlevelnumber, int(row[1]) - firstlevelnumber,
                                            float(row[2]), float(row[3]), int(row[4]) == 1))
                    transitionsdict[(Z, ionstage)] = pd.DataFrame(translist, columns=transitiontuple._fields)
                else:
                    for _ in range(transition_count):
                        ftransitions.readline()

    level_lists = []
    iontuple = namedtuple('ion', 'Z ion_stage level_count ion_pot levels transitions')
    leveltuple = namedtuple('level', 'energy_ev g transition_count levelname')
    with open(adatafilename, 'r') as fadata:
        print(f'Reading {adatafilename}')
        for line in fadata:
            if not line.strip():
                continue

            ionheader = line.split()
            Z = int(ionheader[0])
            ionstage = int(ionheader[1])
            level_count = int(ionheader[2])

            if not ionlist or (Z, ionstage) in ionlist:
                level_list = []
                for levelindex in range(level_count):
                    line = fadata.readline()
                    row = line.split()
                    levelname = row[4].strip('\'')
                    numberin = int(row[0])
                    assert(levelindex == numberin - firstlevelnumber)
                    level_list.append(leveltuple(float(row[1]), float(row[2]), int(row[3]), levelname))
                dflevels = pd.DataFrame(level_list)

                translist = transitionsdict.get((Z, ionstage), pd.DataFrame(columns=transitiontuple._fields))
                level_lists.append(iontuple(Z, ionstage, level_count, float(ionheader[3]), dflevels, translist))
            else:
                for _ in range(level_count):
                    fadata.readline()

    dfadata = pd.DataFrame(level_lists)
    return dfadata


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


def main(argsraw=None):
    import argcomplete
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()
    commandlist = [
        ('getmodeldeposition', 'deposition'),
        ('makemodel1dslicefrom3d', 'slice3dmodel'),
        ('makemodelbotyanski', 'makemodelbotyanski'),
        ('plotestimators', 'estimators'),
        ('plotlightcurve', 'lightcurve'),
        ('plotnltepops', 'nltepops'),
        ('plotmacroatom', 'macroatom'),
        ('plotnonthermal', 'nonthermal'),
        ('plotradfield', 'radfield'),
        ('plotspectrum', 'spectra'),
        ('plottransitions', 'transitions'),
        ('spencerfano', 'spencerfano'),
    ]

    for command, submodulename in commandlist:
        submodule = importlib.import_module('artistools.' + submodulename)
        subparser = subparsers.add_parser(command)
        submodule.addargs(subparser)
        subparser.set_defaults(func=submodule.main)

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if args.func is not None:
        args.func(args=args)
    else:
        # parser.print_help()
        print('usage: artistools <command>, where <command> is one of:\n')
        for command, _ in commandlist:
            print(f'  {command}')


def get_ionstring(atomic_number, ionstage):
    return f'{elsymbols[atomic_number]} {roman_numerals[ionstage]}'


def list_commands():
    print("artistools commands:")
    for script in sorted(console_scripts):
        command = script.split('=')[0].strip()
        print(f'  {command}')
