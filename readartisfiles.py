#!/usr/bin/env python3
import collections
import os

pydir = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n']+[line.split(',')[1] for line in open(os.path.join(pydir,'elements.csv'))]

roman_numerals = ('','I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII',
                  'XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX')

def showtimesteptimes(filename):
    specdata = pd.read_csv(filename, delim_whitespace=True)
    print('Time steps and times in days:\n')

    times = specdata.columns
    indexendofcolumnone = math.ceil((len(times)-1) / numberofcolumns)
    for rownum in range(0,indexendofcolumnone):
        strline = ""
        for colnum in range(numberofcolumns):
            if colnum > 0:
                strline += '\t'
            newindex = rownum + colnum*indexendofcolumnone
            if newindex < len(times):
                strline += '{0:4d}: {1:.3f}'.format(newindex,float(times[newindex+1]))
        print(strline)

def getartiselementlist(filename):
    elementtuple = collections.namedtuple('elementtuple','Z,nions,lowermost_ionstage,uppermost_ionstage,nlevelsmax_readin,abundance,mass,startindex')

    elementlist = []
    with open(filename,'r') as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline() #T_preset
        fcompdata.readline() #homogeneous_abundances
        startindex = 0
        for element in range(nelements):
            line = fcompdata.readline()
            linesplit = line.split()
            elementlist.append( elementtuple._make(list(map(int,linesplit[:5])) + list(map(float,linesplit[5:])) + [startindex]) )
            startindex += elementlist[-1].nions
            print(elementlist[-1])
    return elementlist

def getmodeldata(filename):
    modeldata = []
    gridcelltuple = collections.namedtuple('gridcell','cellid velocity logrho ffe fni fco f52fe f48cr')
    modeldata.append( gridcelltuple._make([-1,0.,0.,0.,0.,0.,0.,0.]) )
    with open(filename,'r') as fmodel:
        gridcellcount = int(fmodel.readline())
        t_model_init = float(fmodel.readline())
        for line in fmodel:
            row = line.split()
            modeldata.append( gridcelltuple._make([int(row[0])] + list(map(float,row[1:]))) )

    return modeldata

def getinitialabundances1d(filename):
    abundancedata = []
    abundancedata.append( [] )
    with open(filename,'r') as fabund:
        for line in fabund:
            row = line.split()
            abundancedata.append( [int(row[0])] + list(map(float,row[1:])) )

    return abundancedata
