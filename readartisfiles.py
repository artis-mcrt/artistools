#!/usr/bin/env python3
import collections
import math
import os

import numpy as np
import pandas as pd

pydir = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + [line.split(',')[1]
                     for line in open(os.path.join(pydir, 'elements.csv'))]

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII',
                  'XVIII', 'XIX', 'XX')

C = 299792458  # [m / s]


def showtimesteptimes(specfilename, numberofcolumns=5):
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


def getartiselementlist(filename):
    elementtuple = collections.namedtuple(
        'elementtuple', 'Z,nions,lowermost_ionstage,uppermost_ionstage,nlevelsmax_readin,abundance,mass,startindex')

    elementlist = []
    with open(filename, 'r') as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline()  # T_preset
        fcompdata.readline()  # homogeneous_abundances
        startindex = 0
        for _ in nelements:
            line = fcompdata.readline()
            linesplit = line.split()
            elementlist.append(elementtuple._make(
                list(map(int, linesplit[:5])) + list(map(float, linesplit[5:])) + [startindex]))
            startindex += elementlist[-1].nions
            print(elementlist[-1])

    return elementlist


def getmodeldata(filename):
    modeldata = []
    gridcelltuple = collections.namedtuple(
        'gridcell', 'cellid velocity logrho ffe fni fco f52fe f48cr')
    modeldata.append(gridcelltuple._make([-1, 0., 0., 0., 0., 0., 0., 0.]))
    with open(filename, 'r') as fmodel:
        # gridcellcount = int(fmodel.readline())
        # t_model_init = float(fmodel.readline())
        for line in fmodel:
            row = line.split()
            modeldata.append(gridcelltuple._make(
                [int(row[0])] + list(map(float, row[1:]))))

    return modeldata


def getinitialabundances1d(filename):
    abundancedata = []
    abundancedata.append([])
    with open(filename, 'r') as fabund:
        for line in fabund:
            row = line.split()
            abundancedata.append([int(row[0])] + list(map(float, row[1:])))

    return abundancedata


def get_spectrum(specfilename, timesteplow, timestephigh=-1, normalised=False,
                 filter=False, filter_kwargs={}):
    """
        returns a tuple of (wavelgnth in Angstroms, flux over d lambda [lambda in meters])
    """
    specdata = pd.read_csv(specfilename, delim_whitespace=True)

    arraynu = specdata['0']
    arraylambda = C / arraynu

    array_fnu = specdata[specdata.columns[timesteplow + 1]]

    for timestep in range(timesteplow + 1, timestephigh + 1):
        array_fnu += specdata[specdata.columns[timestep + 1]]

    # best to use the filter on this list (because
    # it hopefully has regular sampling)
    if filter:
        import scipy.signal
        array_fnu = scipy.signal.savgol_filter(array_fnu, **filter_kwargs)

    array_fnu = array_fnu / (timestephigh - timesteplow + 1)

    array_flambda = array_fnu * (arraynu ** 2) / C

    if normalised:
        array_flambda /= max(array_flambda)

    return arraylambda * 1e10, array_flambda


def get_timestep_time(specfilename, timestep):
    time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)

    return time_columns.columns[timestep + 1]


if __name__ == "__main__":
    print("this script is for inclusion only")
