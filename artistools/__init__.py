#!/usr/bin/env python3
"""Artistools.

A collection of plotting, analysis, and file format conversion tools for the ARTIS radiative transfer code.
"""

import gzip
import math
import os.path
import sys
# from astropy import units as u
from collections import namedtuple
from itertools import chain

# import scipy.signal
import numpy as np
import pandas as pd
# from astropy import constants as const

PYDIR = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'data', 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

commandlist = [
    ('getartismodeldeposition', 'deposition'),
    ('getartisspencerfano', 'spencerfano'),
    ('makeartismodel1dslicefrom3d', 'makemodel.1dslicefrom3d'),
    ('makeartismodelbotyanski2017', 'makemodel.botyanski2017'),
    ('plotartisestimators', 'estimators'),
    ('plotartislightcurve', 'lightcurve'),
    ('plotartisnltepops', 'nltepops'),
    ('plotartismacroatom', 'macroatom'),
    ('plotartisnonthermal', 'nonthermal'),
    ('plotartisradfield', 'radfield'),
    ('plotartisspectrum', 'spectra'),
    ('plotartistransitions', 'transitions'),
]

console_scripts = [f'{command} = artistools.{submodulename}:main' for command, submodulename in commandlist]
console_scripts.append('at = artistools:main')
console_scripts.append('artistools = artistools:main')


def showtimesteptimes(specfilename, modelpath=None, numberofcolumns=5):
    """Print a table showing the timesteps and their corresponding times."""
    if modelpath is not None:
        specfilename = firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
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
            if newindex + 1 < len(times):
                strline += f'{newindex:4d}: {float(times[newindex + 1]):.3f}d'
        print(strline)


def get_composition_data(filename):
    """Return a pandas DataFrame containing details of included elements and ions."""
    if os.path.isdir(filename):
        filename = os.path.join(filename, 'compositiondata.txt')

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
    """Return a list containing named tuples for all model grid cells."""
    if os.path.isdir(filename):
        filename = firstexisting(['model.txt.gz', 'model.txt'], path=filename)

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

            # the model.txt file may contain more shells, but we should ignore them
            # if we have already read in the specified number of shells
            if len(modeldata) == gridcellcount:
                break

    assert len(modeldata) <= gridcellcount
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
    if os.path.isdir(abundancefilename):
        abundancefilename = firstexisting(['abundances.txt.gz', 'abundances.txt'], path=abundancefilename)

    columns = ['inputcellid', *['X_' + elsymbols[x] for x in range(1, 31)]]
    abundancedata = pd.read_csv(abundancefilename, delim_whitespace=True, header=None, names=columns)
    abundancedata.index.name = 'modelgridindex'
    return abundancedata


def save_initialabundances(dfabundances, abundancefilename):
    """Save a DataFrame (same format as get_initialabundances) to model.txt."""
    dfabundances['inputcellid'] = dfabundances['inputcellid'].astype(np.int)
    dfabundances.to_csv(abundancefilename, header=False, sep=' ', index=False)


def get_timestep_times(specfilename):
    """Return a list of the time in days of each timestep using a spec.out file."""
    if os.path.isdir(specfilename):
        specfilename = firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=specfilename)

    time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)

    return time_columns.columns[1:]


def get_timestep_times_float(specfilename):
    """Return a list of the time in days of each timestep using a spec.out file."""
    return np.array([float(t.rstrip('d')) for t in get_timestep_times(specfilename)])


def get_closest_timestep(specfilename, timedays):
    """Return the timestep number whose time is closest to timedays."""
    try:
        # could be a string like '330d'
        timedays_float = float(timedays.rstrip('d'))
    except AttributeError:
        timedays_float = float(timedays)
    return np.abs(get_timestep_times_float(specfilename) - timedays_float).argmin()


def get_timestep_time(specfilename, timestep):
    """Return the time in days of a timestep number using a spec.out file."""
    if os.path.isdir(specfilename):
        specfilename = firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=specfilename)

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


def get_levels(modelpath, ionlist=None, get_transitions=False, get_photoionisations=False):
    """Return a list of lists of levels."""
    adatafilename = os.path.join(modelpath, 'adata.txt')

    transitiontuple = namedtuple('transition', 'lower upper A collstr forbidden')

    firstlevelnumber = 1

    transitionsdict = {}
    if get_transitions:
        transition_filename = os.path.join(modelpath, 'transitiondata.txt')

        print(f'Reading {transition_filename}')
        with opengzip(transition_filename, 'r') as ftransitions:
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

    phixsdict = {}
    if get_photoionisations:
        phixs_filename = os.path.join(modelpath, 'phixsdata_v2.txt')

        print(f'Reading {phixs_filename}')
        with opengzip(phixs_filename, 'r') as fphixs:
            nphixspoints = int(fphixs.readline())
            phixsnuincrement = float(fphixs.readline())

            xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1), num=nphixspoints + 1, endpoint=False)

            for line in fphixs:
                if not line.strip():
                    continue

                ionheader = line.split()
                Z = int(ionheader[0])
                upperionstage = int(ionheader[1])
                upperionlevel = int(ionheader[2])
                lowerionstage = int(ionheader[3])
                lowerionlevel = int(ionheader[4])
                # threshold_ev = float(ionheader[5])

                assert upperionstage == lowerionstage + 1

                if upperionlevel >= 0:
                    targetlist = [(upperionlevel, 1.0)]
                else:
                    targetlist = []
                    ntargets = int(fphixs.readline())
                    for _ in range(ntargets):
                        level, fraction = fphixs.readline().split()
                        targetlist.append((int(level), float(fraction)))

                if not ionlist or (Z, lowerionstage) in ionlist:
                    phixslist = []
                    for _ in range(nphixspoints):
                        phixslist.append(float(fphixs.readline()))
                    phixsdict[(Z, lowerionstage, lowerionlevel)] = np.array(list(zip(xgrid, phixslist)))
                else:
                    for _ in range(nphixspoints):
                        fphixs.readline()

    level_lists = []
    iontuple = namedtuple('ion', 'Z ion_stage level_count ion_pot levels transitions')
    leveltuple = namedtuple('level', 'energy_ev g transition_count levelname phixstable')
    with opengzip(adatafilename, 'rt') as fadata:
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
                    assert levelindex == numberin - firstlevelnumber
                    phixstable = phixsdict.get((Z, ionstage, numberin), [])
                    level_list.append(leveltuple(float(row[1]), float(row[2]), int(row[3]), levelname, phixstable))
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
        return open(plotlabelfile, mode='r').readline().strip()
    except FileNotFoundError:
        return os.path.basename(folderpath)


def get_time_range(timearray, timestep_range_str, timemin, timemax, timedays_range_str):
    """Handle a time range specified in either days or timesteps"""

    # assertions make sure time is specified either by timesteps or times in days, but not both!
    if timestep_range_str is not None:
        assert timemin is None and timemax is None and timedays_range_str is None

        if '-' in timestep_range_str:
            timestepmin, timestepmax = [int(nts) for nts in timestep_range_str.split('-')]
        else:
            timestepmin = int(timestep_range_str)
            timestepmax = timestepmin
    else:
        assert (timemin is None and timemax is None and timedays_range_str is not None or
                timemin is not None and timemax is not None and timedays_range_str is None)
        if timedays_range_str is not None and '-' in timedays_range_str:
            timemin, timemax = [float(timedays) for timedays in timedays_range_str.split('-')]

        timestepmin = None
        for timestep, time in enumerate(timearray):
            timefloat = float(time.strip('d'))
            if float(timemin) <= timefloat:
                timestepmin = timestep
                break

        if timestepmin is None:
            print(f"Time min {timemin} is greater than all timesteps ({timearray[0]} to {timearray[-1]})")
            sys.exit()

        if not timemax:
            timemax = float(timearray[-1].strip('d'))
        for timestep, time in enumerate(timearray):
            timefloat = float(time.strip('d'))
            if timefloat + get_timestep_time_delta(timestep, timearray) <= timemax:
                timestepmax = timestep

    time_days_lower = float(timearray[timestepmin])
    time_days_upper = float(timearray[timestepmax]) + get_timestep_time_delta(timestepmax, timearray)

    return timestepmin, timestepmax, time_days_lower, time_days_upper


def get_atomic_number(elsymbol):
    if elsymbol.title() in elsymbols:
        return elsymbols.index(elsymbol.title())
    return -1


def decode_roman_numeral(strin):
    if strin.upper() in roman_numerals:
        return roman_numerals.index(strin.upper())
    return -1


def get_ionstring(atomic_number, ionstage):
    return f'{elsymbols[atomic_number]} {roman_numerals[ionstage]}'


# based on code from https://gist.github.com/kgaughan/2491663/b35e9a117b02a3567c8107940ac9b2023ba34ced
def parse_range(rng, dictvars={}):
    """Take a string with an integer range like 23-26 and return [23, 24, 25, 26], also replacing special variables in dictvars"""
    parts = rng.split('-')

    if len(parts) not in [1, 2]:
        raise ValueError("Bad range: '%s'" % (rng,))

    parts = [int(i) if i not in dictvars else dictvars[i] for i in parts]
    start = parts[0]
    end = start if len(parts) == 1 else parts[1]

    if start > end:
        end, start = start, end

    return range(start, end + 1)


def parse_range_list(rngs, dictvars={}):
    """Parse a string with comma-separated ranges or a list of range strings.

    Return a sorted list of integers in any of the ranges."""
    if isinstance(rngs, list):
        rngs = ','.join(rngs)

    return sorted(set(chain.from_iterable([parse_range(rng, dictvars) for rng in rngs.split(',')])))


def opengzip(filename, mode):
    """Open filename.gz or filename."""
    filenamegz = filename + '.gz'
    return gzip.open(filenamegz, mode) if os.path.exists(filenamegz) else open(filename, mode)


def firstexisting(filelist, path='.'):
    """Return the first existing file in filelist"""
    for filename in filelist:
        if os.path.exists(os.path.join(path, filename)):
            return os.path.join(path, filename)
    return os.path.join(path, filelist[-1])


def main(argsraw=None):
    """Show a list of available artistools commands."""
    import argcomplete
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers()

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
        print('artistools provides the following commands:\n')

        # for script in sorted(console_scripts):
        #     command = script.split('=')[0].strip()
        #     print(f'  {command}')

        for command, _ in commandlist:
            print(f'  {command}')
