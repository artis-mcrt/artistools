#!/usr/bin/env python3
import collections
import math
import os

import pandas as pd
from astropy import constants as const

PYDIR = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + list(pd.read_csv(
    os.path.join(PYDIR, 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII',
                  'XVIII', 'XIX', 'XX')


def showtimesteptimes(specfilename, numberofcolumns=5):
    """
        Print a table showing the timeteps and their corresponding times
    """
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    print('Time steps and times in days:\n')

    times = specdata.columns
    indexendofcolumnone = math.ceil((len(times) - 1) / numberofcolumns)
    for rownum in range(0, indexendofcolumnone):
        strline = ""
        for colnum in range(numberofcolumns):
            if colnum > 0:
                strline += '\t'
            newindex = rownum + colnum * indexendofcolumnone
            if newindex < len(times):
                strline += '{0:4d}: {1:.3f}'.format(
                    newindex, float(times[newindex + 1]))
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

    print(compdf)
    return compdf


def getmodeldata(filename):
    """
        Return a list containing named tuples for all model grid cells
    """
    modeldata = []
    gridcelltuple = collections.namedtuple(
        'gridcell', 'cellid velocity logrho ffe fni fco f52fe f48cr')
    modeldata.append(gridcelltuple._make([-1, 0., 0., 0., 0., 0., 0., 0.]))
    with open(filename, 'r') as fmodel:
        gridcellcount = int(fmodel.readline())
        t_model_init = float(fmodel.readline())
        for line in fmodel:
            row = line.split()
            modeldata.append(gridcelltuple._make(
                [int(row[0])] + list(map(float, row[1:]))))

    return modeldata


def getinitialabundances1d(filename):
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


def get_spectrum(specfilename, timesteplow, timestephigh=-1, normalised=False,
                 fnufilterfunc=None):
    """
        Return a pandas DataFrame containing an ARTIS emergent spectrum
    """
    if timestephigh < 0:
        timestephigh = timesteplow

    specdata = pd.read_csv(specfilename, delim_whitespace=True)

    arraynu = specdata['0']

    array_fnu = specdata[specdata.columns[timesteplow + 1]]

    for timestep in range(timesteplow + 1, timestephigh + 1):
        array_fnu += specdata[specdata.columns[timestep + 1]]

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        array_fnu = fnufilterfunc(array_fnu)

    array_fnu = array_fnu / (timestephigh - timesteplow + 1)

    dfspectrum = pd.DataFrame({'nu': arraynu,
                               'f_nu': array_fnu})

    dfspectrum['lambda_angstroms'] = const.c.value / dfspectrum['nu']  * 1e10
    dfspectrum['f_lambda'] = dfspectrum['f_nu'] * (dfspectrum['nu'] ** 2) / const.c.value

    if normalised:
        dfspectrum['f_nu'] /= dfspectrum['f_nu'].max()
        dfspectrum['f_lambda'] /= dfspectrum['f_lambda'].max()

    return dfspectrum


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


def get_levels(adatafilename):
    """
        Return a list of lists of levels
    """
    level_lists = []
    iontuple = collections.namedtuple(
        'ion', 'Z ion_stage level_count ion_pot level_list')
    leveltuple = collections.namedtuple(
        'level', 'number energyev g transition_count hillier_name')

    with open(adatafilename, 'r') as fadata:
        EOFfound = False
        while not EOFfound:
            line = fadata.readline()
            if not line:
                EOFfound = True
            elif len(line.strip()) > 0:
                ionheader = line.split()
                level_count = int(ionheader[2])

                level_list = []
                for n in range(level_count):
                    line = fadata.readline()
                    row = line.split()
                    hillier_name = row[5].strip('\'')
                    level_list.append(leveltuple(int(row[0]), float(row[1]), float(row[2]), int(row[3]), hillier_name))

                level_lists.append(iontuple(int(ionheader[0]), int(ionheader[1]), level_count, float(ionheader[3]), list(level_list)))

    return level_lists


if __name__ == "__main__":
    print("this script is for inclusion only")
