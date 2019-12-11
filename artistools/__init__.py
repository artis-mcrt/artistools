#!/usr/bin/env python3
"""Artistools.

A collection of plotting, analysis, and file format conversion tools for the ARTIS radiative transfer code.
"""
import argparse
from functools import lru_cache
import gzip
import lzma
import math
import os.path
import sys
from collections import namedtuple
from itertools import chain
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Iterable

# import scipy.signal
import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants as const
from PyPDF2 import PdfFileMerger

if sys.version_info < (3,):
    print("Python 2 not supported")

PYDIR = os.path.dirname(os.path.abspath(__file__))

elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'data', 'elements.csv'))['symbol'].values)

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

commandlist = {
    'getartismodeldeposition': ('artistools.deposition', 'main_analytical'),
    'getartisspencerfano': ('artistools.spencerfano', 'main'),
    'listartistimesteps': ('artistools', 'showtimesteptimes'),
    'makeartismodel1dslicefrom3d': ('artistools.makemodel.1dslicefrom3d', 'main'),
    'makeartismodelbotyanski2017': ('artistools.makemodel.botyanski2017', 'main'),
    'makeartismodelfromshen2018': ('artistools.makemodel.shen2018', 'main'),
    'makeartismodelscalevelocity': ('artistools.makemodel.scalevelocity', 'main'),
    'makeartismodelfullymixed': ('artistools.makemodel.fullymixed', 'main'),
    'plotartisdeposition': ('artistools.deposition', 'main'),
    'plotartisestimators': ('artistools.estimators', 'main'),
    'plotartislightcurve': ('artistools.lightcurve', 'main'),
    'plotartisnltepops': ('artistools.nltepops', 'main'),
    'plotartismacroatom': ('artistools.macroatom', 'main'),
    'plotartisnonthermal': ('artistools.nonthermal', 'main'),
    'plotartisradfield': ('artistools.radfield', 'main'),
    'plotartisspectrum': ('artistools.spectra', 'main'),
    'plotartistransitions': ('artistools.transitions', 'main'),
    'plotartisinitialcomposition': ('artistools.initial_composition', 'main'),
}

console_scripts = [f'{command} = {submodulename}:{funcname}'
                   for command, (submodulename, funcname) in commandlist.items()]
console_scripts.append('at = artistools:main')
console_scripts.append('artistools = artistools:main')

figwidth = 5


class AppendPath(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # if getattr(args, self.dest) is None:
        #     setattr(args, self.dest, [])
        if isinstance(values, Iterable):
            for pathstr in values:
                getattr(args, self.dest).append(Path(pathstr))
        else:
            setattr(args, self.dest, Path(values))


class ExponentLabelFormatter(ticker.ScalarFormatter):
    """Formatter to move the 'x10^x' offset text into the axis label."""

    def __init__(self, labeltemplate, useMathText=True, decimalplaces=None):
        self.set_labeltemplate(labeltemplate)
        self.decimalplaces = decimalplaces
        super().__init__(useOffset=True, useMathText=useMathText)
        # ticker.ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)

    def _set_formatted_label_text(self):
        # or use self.orderOfMagnitude
        stroffset = self.get_offset().replace(r'$\times', '$') + ' '
        strnewlabel = self.labeltemplate.format(stroffset)
        self.axis.set_label_text(strnewlabel)
        assert(self.offset == 0)
        self.axis.offsetText.set_visible(False)

    def set_labeltemplate(self, labeltemplate):
        assert '{' in labeltemplate
        self.labeltemplate = labeltemplate

    def set_locs(self, locs):
        if self.decimalplaces is not None:
            self.format = '%1.' + str(self.decimalplaces) + 'f'
            if self._usetex:
                self.format = '$%s$' % self.format
            elif self._useMathText:
                self.format = '$%s$' % ('\\mathdefault{%s}' % self.format)
        super().set_locs(locs)

        if self.decimalplaces is not None:
            # rounding the tick labels will make the locations incorrect unless we round these too
            newlocs = [float(('%1.' + str(self.decimalplaces) + 'f') % (x / (10 ** self.orderOfMagnitude)))
                       * (10 ** self.orderOfMagnitude) for x in self.locs]
            super().set_locs(newlocs)

        self._set_formatted_label_text()

    def set_axis(self, axis):
        super().set_axis(axis)
        self._set_formatted_label_text()


def showtimesteptimes(modelpath=None, numberofcolumns=5, args=None):
    """Print a table showing the timesteps and their corresponding times."""
    if modelpath is None:
        modelpath = Path()

    print('Time steps and corresponding times in days:\n')

    times = get_timestep_times(modelpath)
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


@lru_cache(maxsize=8)
def get_composition_data(filename):
    """Return a pandas DataFrame containing details of included elements and ions."""
    if os.path.isdir(Path(filename)):
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


def get_composition_data_from_outputfile(modelpath):
    """Read ion list from output file"""
    atomic_composition = {}

    output = open(modelpath / "output_0-0.txt", 'r').read().splitlines()
    ioncount = 0
    for row in output:
        if row.split()[0] == '[input.c]':
            split_row = row.split()
            if split_row[1] == 'element':
                Z = int(split_row[4])
                ioncount = 0
            elif split_row[1] == 'ion':
                ioncount += 1
                atomic_composition[Z] = ioncount

    composition_df = pd.DataFrame([(Z, atomic_composition[Z]) for Z in atomic_composition.keys()], columns=['Z', 'nions'])
    composition_df['lowermost_ionstage'] = [1] * composition_df.shape[0]
    composition_df['uppermost_ionstage'] = composition_df['nions']
    return composition_df


@lru_cache(maxsize=8)
def get_modeldata(filename):
    """Return a list containing named tuples for all model grid cells."""
    if os.path.isdir(filename):
        filename = firstexisting(['model.txt.xz', 'model.txt.gz', 'model.txt'], path=filename)

    modeldata = pd.DataFrame()

    gridcelltuple = None
    velocity_inner = 0.
    with open(filename, 'r') as fmodel:
        gridcellcount = int(fmodel.readline())
        t_model_init_days = float(fmodel.readline())
        for line in fmodel:
            row = line.split()

            if gridcelltuple is None:
                gridcelltuple = namedtuple('gridcell', [
                    'inputcellid', 'velocity_inner', 'velocity_outer', 'logrho',
                    'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57'][:len(row) + 1])

            celltuple = gridcelltuple(int(row[0]), velocity_inner, *(map(float, row[1:])))
            modeldata = modeldata.append([celltuple], ignore_index=True)

            # next inner is the current outer
            velocity_inner = celltuple.velocity_outer

            # the model.txt file may contain more shells, but we should ignore them
            # if we have already read in the specified number of shells
            if len(modeldata) == gridcellcount:
                break

    assert len(modeldata) <= gridcellcount
    modeldata.index.name = 'cellid'
    return modeldata, t_model_init_days


def get_2d_modeldata(modelpath):
    filepath = os.path.join(modelpath, 'model.txt')
    num_lines = sum(1 for line in open(filepath))
    skiprowlist = [0, 1, 2]
    skiprowlistodds = skiprowlist + [i for i in range(3, num_lines) if i % 2 == 1]
    skiprowlistevens = skiprowlist + [i for i in range(3, num_lines) if i % 2 == 0]

    model1stlines = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=skiprowlistevens)
    model2ndlines = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=skiprowlistodds)

    model = pd.concat([model1stlines, model2ndlines], axis=1)
    column_names = ['inputcellid', 'cellpos_mid[r]', 'cellpos_mid[z]', 'rho_model',
                    'ffe', 'fni', 'fco', 'ffe52', 'fcr48']
    model.columns = column_names
    return model


def get_3d_modeldata(modelpath):
    model = pd.read_csv(os.path.join(modelpath[0], 'model.txt'), delim_whitespace=True, header=None, skiprows=3)
    columns = ['inputcellid', 'cellpos_in[z]', 'cellpos_in[y]', 'cellpos_in[x]', 'rho_model',
               'ffe', 'fni', 'fco', 'ffe52', 'fcr48']
    model = pd.DataFrame(model.values.reshape(-1, 10))
    model.columns = columns
    return model


def get_vpkt_data(modelpath):
    filename = Path(modelpath, 'vpkt.txt')
    vpkt_data = {}
    with open(filename, 'r') as vpkt:
        vpkt_data['nobservations'] = int(vpkt.readline())
        vpkt_data['cos_theta'] = [float(x) for x in vpkt.readline().split()]
        vpkt_data['phi'] = [int(x) for x in vpkt.readline().split()]
        line4 = vpkt.readline()
        _, vpkt_data['initial_time'], vpkt_data['final_time'] = [int(x) for x in vpkt.readline().split()]
        return vpkt_data


@lru_cache(maxsize=8)
def get_grid_mapping(modelpath):
    """Return dict with the associated propagation cells for each model grid cell and
    a dict with the associated model grid cell of each propagration cell."""

    if os.path.isdir(modelpath):
        filename = firstexisting(['grid.out.xz', 'grid.out.gz', 'grid.out'], path=modelpath)
    else:
        filename = modelpath

    assoc_cells = {}
    mgi_of_propcells = {}
    with open(filename, 'r') as fgrid:
        for line in fgrid:
            row = line.split()
            propcellid, mgi = int(row[0]), int(row[1])
            if mgi not in assoc_cells:
                assoc_cells[mgi] = []
            assoc_cells[mgi].append(propcellid)
            mgi_of_propcells[propcellid] = mgi

    return assoc_cells, mgi_of_propcells


def get_wid_init(modelpath):
    tmin = get_timestep_times_float(modelpath, loc='start')[0] * u.day.to('s')
    vmax = get_modeldata(modelpath)[0]['velocity_outer'].iloc[-1] * 1e5

    rmax = vmax * tmin

    coordmax0 = rmax
    ncoordgrid0 = 50

    wid_init = 2 * coordmax0 / ncoordgrid0
    return wid_init


def get_mgi_of_velocity(modelpath, velocity, mgilist=None):
    """Return the modelgridindex of the cell whose outer velocity is closest to velocity.
    If mgilist is given, then chose from these cells only"""
    modeldata, _ = get_modeldata(modelpath)

    if not mgilist:
        mgilist = [mgi for mgi in modeldata.index]
        arr_vouter = modeldata['velocity_outer'].values
    else:
        arr_vouter = np.array([modeldata['velocity_outer'][mgi] for mgi in mgilist])

    index_closestvouter = np.abs(arr_vouter - velocity).argmin()

    if velocity < arr_vouter[index_closestvouter] or index_closestvouter + 1 >= len(mgilist):
        return mgilist[index_closestvouter]
    elif velocity < arr_vouter[index_closestvouter + 1]:
        return mgilist[index_closestvouter + 1]
    else:
        assert(False)


def save_modeldata(dfmodeldata, t_model_init_days, filename):
    """Save a pandas DataFrame into ARTIS model.txt"""
    with open(filename, 'w') as fmodel:
        fmodel.write(f'{len(dfmodeldata)}\n{t_model_init_days:f}\n')
        for _, cell in dfmodeldata.iterrows():
            fmodel.write(f'{cell.inputcellid:6.0f}   {cell.velocity_outer:9.2f}   {cell.logrho:10.8f} '
                         f'{cell.X_Fegroup:10.4e} {cell.X_Ni56:10.4e} {cell.X_Co56:10.4e} '
                         f'{cell.X_Fe52:10.4e} {cell.X_Cr48:10.4e}')
            if 'X_Ni57' in dfmodeldata.columns:
                fmodel.write(f' {cell.X_Ni57:10.4e}')
                if 'X_Co57' in dfmodeldata.columns:
                    fmodel.write(f' {cell.X_Co57:10.4e}')
            fmodel.write('\n')


@lru_cache(maxsize=8)
def get_initialabundances(modelpath):
    """Return a list of mass fractions."""
    abundancefilepath = firstexisting(['abundances.txt.xz', 'abundances.txt.gz', 'abundances.txt'], path=modelpath)

    columns = ['inputcellid', *['X_' + elsymbols[x] for x in range(1, 31)]]
    abundancedata = pd.read_csv(abundancefilepath, delim_whitespace=True, header=None, names=columns)
    abundancedata.index.name = 'modelgridindex'
    return abundancedata


def save_initialabundances(dfabundances, abundancefilename):
    """Save a DataFrame (same format as get_initialabundances) to model.txt."""
    dfabundances['inputcellid'] = dfabundances['inputcellid'].astype(np.int)
    dfabundances.to_csv(abundancefilename, header=False, sep=' ', index=False)


@lru_cache(maxsize=16)
def get_nu_grid(modelpath):
    """Get an array of frequencies at which the ARTIS spectra are binned by exspec."""
    specfilename = firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    return specdata.loc[:, '0'].values


@lru_cache(maxsize=16)
def get_timestep_times(modelpath):
    """Return a list of the time in days of each timestep using a spec.out file."""
    try:
        specfilename = firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
        time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)
        return time_columns.columns[1:]
    except FileNotFoundError:
        return [f'{tdays:.3f}' for tdays in get_timestep_times_float(modelpath)]


@lru_cache(maxsize=16)
def get_timestep_times_float(modelpath, loc='mid'):
    """Return a list of the time in days of each timestep."""
    inputparams = get_inputparams(modelpath)
    tmin = inputparams['tmin']
    dlogt = (math.log(inputparams['tmax']) - math.log(tmin)) / inputparams['ntstep']
    timesteps = range(inputparams['ntstep'])
    if loc == 'mid':
        tmids = np.array([tmin * math.exp((ts + 0.5) * dlogt) for ts in timesteps])
        return tmids
    elif loc == 'start':
        tstarts = np.array([tmin * math.exp(ts * dlogt) for ts in timesteps])
        return tstarts
    elif loc == 'end':
        tends = np.array([tmin * math.exp((ts + 1) * dlogt) for ts in timesteps])
        return tends
    elif loc == 'delta':
        tdeltas = np.array([tmin * (math.exp((ts + 1) * dlogt) - math.exp(ts * dlogt)) for ts in timesteps])
        return tdeltas
    else:
        raise ValueError("loc must be one of 'mid', 'start', 'end', or 'delta'")


def get_timestep_of_timedays(modelpath, timedays):
    """Return the timestep containing the given time in days."""
    try:
        # could be a string like '330d'
        timedays_float = float(timedays.rstrip('d'))
    except AttributeError:
        timedays_float = float(timedays)

    arr_tstart = get_timestep_times_float(modelpath, loc='start')

    for ts, tstart in enumerate(arr_tstart):
        if tstart > timedays_float:
            return ts

    assert(False)
    return


def get_time_range(modelpath, timestep_range_str, timemin, timemax, timedays_range_str):
    """Handle a time range specified in either days or timesteps."""
    # assertions make sure time is specified either by timesteps or times in days, but not both!
    tstarts = get_timestep_times_float(modelpath, loc='start')
    tmids = get_timestep_times_float(modelpath, loc='mid')
    tends = get_timestep_times_float(modelpath, loc='end')

    timedays_is_specified = (timemin is not None and timemax is not None) or timedays_range_str is not None

    if timemin and timemin > tends[-1]:
        raise ValueError(f"timemin {timemin} is after the last timestep at {tends[-1]}")
    elif timemax and timemax < tstarts[0]:
        raise ValueError(f"timemax {timemax} is before the first timestep at {tstarts[0]}")

    if timestep_range_str is not None:
        if timedays_is_specified:
            raise ValueError("Cannot specify both time in days and timestep numbers.")

        if isinstance(timestep_range_str, str) and '-' in timestep_range_str:
            timestepmin, timestepmax = [int(nts) for nts in timestep_range_str.split('-')]
        else:
            timestepmin = int(timestep_range_str)
            timestepmax = timestepmin
    elif timedays_is_specified:
        timestepmin = None
        timestepmax = None
        if timedays_range_str is not None:
            if isinstance(timedays_range_str, str) and '-' in timedays_range_str:
                timemin, timemax = [float(timedays) for timedays in timedays_range_str.split('-')]
            else:
                timeavg = float(timedays_range_str)
                timestepmin = get_timestep_of_timedays(modelpath, timeavg)
                timestepmax = timestepmin
                timemin = tstarts[timestepmin]
                timemax = tends[timestepmax]
                # timedelta = 10
                # timemin, timemax = timeavg - timedelta, timeavg + timedelta

        for timestep, tmid in enumerate(tmids):
            if tmid >= float(timemin):
                timestepmin = timestep
                break

        if timestepmin is None:
            print(f"Time min {timemin} is greater than all timesteps ({tstarts[0]} to {tends[-1]})")
            raise ValueError

        if not timemax:
            timemax = tends[-1]
        for timestep, tmid in enumerate(tmids):
            if tmid <= float(timemax):
                timestepmax = timestep

        if timestepmax < timestepmin:
            raise ValueError("Specified time range does not include any full timesteps.")
    else:
        raise ValueError("Either time or timesteps must be specified.")

    timesteplast = len(tmids) - 1
    if timestepmax > timesteplast:
        print(f"Warning timestepmax {timestepmax} > timesteplast {timesteplast}")
        timestepmax = timesteplast
    time_days_lower = tstarts[timestepmin]
    time_days_upper = tends[timestepmax]

    return timestepmin, timestepmax, time_days_lower, time_days_upper


def get_timestep_time(modelpath, timestep):
    """Return the time in days of a timestep number using a spec.out file."""
    timearray = get_timestep_times(modelpath)
    if timearray is not None:
        return timearray[timestep]

    return -1


def get_timestep_time_delta(timestep, timearray=None, inputparams=None, modelpath=None):
    """Return the time in days between timestep and timestep + 1."""
    if inputparams is not None or modelpath is not None:
        if modelpath is not None:
            inputparams = get_inputparams(modelpath)
        tmin = inputparams['tmin']
        dlogt = (math.log(inputparams['tmax']) - math.log(tmin)) / inputparams['ntstep']
        timesteps = range(inputparams['ntstep'])
        tstarts = [tmin * math.exp(ts * dlogt) for ts in timesteps]
        delta_t = tmin * math.exp((timestep + 1) * dlogt) - tstarts[timestep]
    elif timearray is not None:
        if timestep < len(timearray) - 1:
            delta_t = (float(timearray[timestep + 1]) - float(timearray[timestep]))
        else:
            delta_t = (float(timearray[timestep]) - float(timearray[timestep - 1]))
    else:
        assert timearray or inputparams

    # assert(delta_t == get_timestep_times_float(modelpath, loc='delta'))
    return delta_t


def parse_adata(fadata, phixsdict, ionlist):
    """Generate ions and their level lists from adata.txt."""
    firstlevelnumber = 1

    for line in fadata:
        if not line.strip():
            continue

        ionheader = line.split()
        Z = int(ionheader[0])
        ionstage = int(ionheader[1])
        level_count = int(ionheader[2])
        ionisation_energy_ev = float(ionheader[3])

        if not ionlist or (Z, ionstage) in ionlist:
            level_list = []
            for levelindex in range(level_count):
                row = fadata.readline().split()

                levelname = row[4].strip('\'')
                numberin = int(row[0])
                assert levelindex == numberin - firstlevelnumber
                phixstargetlist, phixstable = phixsdict.get((Z, ionstage, numberin), ([], []))

                level_list.append((float(row[1]), float(row[2]), int(row[3]), levelname, phixstargetlist, phixstable))

            dflevels = pd.DataFrame(level_list,
                                    columns=['energy_ev', 'g', 'transition_count',
                                             'levelname', 'phixstargetlist', 'phixstable'])

            yield Z, ionstage, level_count, ionisation_energy_ev, dflevels

        else:
            for _ in range(level_count):
                fadata.readline()


def parse_transitiondata(ftransitions, ionlist):
    firstlevelnumber = 1

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
                    (int(row[0]) - firstlevelnumber, int(row[1]) - firstlevelnumber,
                     float(row[2]), float(row[3]), int(row[4]) == 1))

            yield Z, ionstage, pd.DataFrame(translist, columns=['lower', 'upper', 'A', 'collstr', 'forbidden'])
        else:
            for _ in range(transition_count):
                ftransitions.readline()


def parse_phixsdata(fphixs, ionlist):
    firstlevelnumber = 1
    nphixspoints = int(fphixs.readline())
    phixsnuincrement = float(fphixs.readline())

    xgrid = np.linspace(1.0, 1.0 + phixsnuincrement * (nphixspoints + 1), num=nphixspoints + 1, endpoint=False)

    for line in fphixs:
        if not line.strip():
            continue

        ionheader = line.split()
        Z = int(ionheader[0])
        upperionstage = int(ionheader[1])
        upperionlevel = int(ionheader[2]) - firstlevelnumber
        lowerionstage = int(ionheader[3])
        lowerionlevel = int(ionheader[4]) - firstlevelnumber
        # threshold_ev = float(ionheader[5])

        assert upperionstage == lowerionstage + 1

        if upperionlevel >= 0:
            targetlist = [(upperionlevel, 1.0)]
        else:
            targetlist = []
            ntargets = int(fphixs.readline())
            for _ in range(ntargets):
                level, fraction = fphixs.readline().split()
                targetlist.append((int(level) - firstlevelnumber, float(fraction)))

        if not ionlist or (Z, lowerionstage) in ionlist:
            phixslist = []
            for _ in range(nphixspoints):
                phixslist.append(float(fphixs.readline()) * 1e-18)
            phixstable = np.array(list(zip(xgrid, phixslist)))

            yield Z, upperionstage, upperionlevel, lowerionstage, lowerionlevel, targetlist, phixstable

        else:
            for _ in range(nphixspoints):
                fphixs.readline()


@lru_cache(maxsize=8)
def get_levels(modelpath, ionlist=None, get_transitions=False, get_photoionisations=False):
    """Return a list of lists of levels."""
    adatafilename = Path(modelpath, 'adata.txt')

    transitionsdict = {}
    if get_transitions:
        transition_filename = Path(modelpath, 'transitiondata.txt')

        print(f'Reading {transition_filename.relative_to(modelpath.parent)}')
        with zopen(transition_filename, 'rt') as ftransitions:
            transitionsdict = {
                (Z, ionstage): dftransitions
                for Z, ionstage, dftransitions in parse_transitiondata(ftransitions, ionlist)}

    phixsdict = {}
    if get_photoionisations:
        phixs_filename = Path(modelpath, 'phixsdata_v2.txt')

        print(f'Reading {phixs_filename.relative_to(modelpath.parent)}')
        with zopen(phixs_filename, 'rt') as fphixs:
            for (Z, upperionstage, upperionlevel, lowerionstage,
                 lowerionlevel, phixstargetlist, phixstable) in parse_phixsdata(fphixs, ionlist):
                phixsdict[(Z, lowerionstage, lowerionlevel)] = (phixstargetlist, phixstable)

    level_lists = []
    iontuple = namedtuple('ion', 'Z ion_stage level_count ion_pot levels transitions')

    with zopen(adatafilename, 'rt') as fadata:
        print(f'Reading {adatafilename.relative_to(modelpath.parent)}')

        for Z, ionstage, level_count, ionisation_energy_ev, dflevels in parse_adata(fadata, phixsdict, ionlist):
            translist = transitionsdict.get((Z, ionstage), pd.DataFrame())
            level_lists.append(iontuple(Z, ionstage, level_count, ionisation_energy_ev, dflevels, translist))

    dfadata = pd.DataFrame(level_lists)
    return dfadata


def parse_recombratefile(frecomb):
    for line in frecomb:
        Z, upper_ionstage, t_count = [int(x) for x in line.split()]
        arr_log10t = []
        arr_rrc_low_n = []
        arr_rrc_total = []
        for _ in range(int(t_count)):
            log10t, rrc_low_n, rrc_total = [float(x) for x in frecomb.readline().split()]

            arr_log10t.append(log10t)
            arr_rrc_low_n.append(rrc_low_n)
            arr_rrc_total.append(rrc_total)

        recombdata_thision = pd.DataFrame({
            'log10T_e': arr_log10t, 'rrc_low_n': arr_rrc_low_n, 'rrc_total': arr_rrc_total})

        recombdata_thision.eval('T_e = 10 ** log10T_e', inplace=True)

        yield Z, upper_ionstage, recombdata_thision


@lru_cache(maxsize=4)
def get_ionrecombratecalibration(modelpath):
    """Read recombrates file."""
    recombdata = {}
    with open(Path(modelpath, 'recombrates.txt'), 'r') as frecomb:
        for Z, upper_ionstage, dfrrc in parse_recombratefile(frecomb):
            recombdata[(Z, upper_ionstage)] = dfrrc

    return recombdata


@lru_cache(maxsize=8)
def get_model_name(path):
    """Get the name of an ARTIS model from the path to any file inside it.

    Name will be either from a special plotlabel.txt file if it exists or the enclosing directory name
    """
    abspath = os.path.abspath(path)

    modelpath = abspath if os.path.isdir(abspath) else os.path.dirname(abspath)

    try:
        plotlabelfile = os.path.join(modelpath, 'plotlabel.txt')
        return open(plotlabelfile, mode='r').readline().strip()
    except FileNotFoundError:
        return os.path.basename(modelpath)


def get_atomic_number(elsymbol):
    if elsymbol.title() in elsymbols:
        return elsymbols.index(elsymbol.title())
    return -1


def decode_roman_numeral(strin):
    if strin.upper() in roman_numerals:
        return roman_numerals.index(strin.upper())
    return -1


@lru_cache(maxsize=16)
def get_ionstring(atomic_number, ionstage, spectral=True):
    if ionstage == 'ALL' or ionstage is None:
        return f'{elsymbols[atomic_number]}'
    elif spectral:
        return f'{elsymbols[atomic_number]} {roman_numerals[ionstage]}'
    else:
        # ion notion e.g. Co+, Fe2+
        if ionstage > 2:
            strcharge = r'$^{' + str(ionstage - 1) + r'{+}}$'
        elif ionstage == 2:
            strcharge = r'$^{+}$'
        else:
            strcharge = ''
        return f'{elsymbols[atomic_number]}{strcharge}'


# based on code from https://gist.github.com/kgaughan/2491663/b35e9a117b02a3567c8107940ac9b2023ba34ced
def parse_range(rng, dictvars={}):
    """Parse a string with an integer range and return a list of numbers, replacing special variables in dictvars."""
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

    Return a sorted list of integers in any of the ranges.
    """
    if isinstance(rngs, list):
        rngs = ','.join(rngs)
    elif not hasattr(rngs, 'split'):
        return [rngs]

    return sorted(set(chain.from_iterable([parse_range(rng, dictvars) for rng in rngs.split(',')])))


def trim_or_pad(requiredlength, *listoflistin):
    for listin in listoflistin:
        if listin is None:
            listin = []
        if len(listin) < requiredlength:
            listin.extend([None for _ in range(requiredlength - len(listin))])
        if len(listin) > requiredlength:
            del listin[requiredlength:]


def flatten_list(listin):
    listout = []
    for elem in listin:
        if isinstance(elem, list):
            listout.extend(elem)
        else:
            listout.append(elem)
    return listout


def zopen(filename, mode):
    """Open filename.xz, filename.gz or filename."""
    filenamexz = str(filename) if str(filename).endswith(".xz") else str(filename) + '.xz'
    filenamegz = str(filename) if str(filename).endswith(".gz") else str(filename) + '.gz'
    if os.path.exists(filenamexz):
        return lzma.open(filenamexz, mode)
    elif os.path.exists(filenamegz):
        return gzip.open(filenamegz, mode)
    else:
        return open(filename, mode)


def firstexisting(filelist, path=Path('.')):
    """Return the first existing file in file list."""
    fullpaths = [Path(path) / filename for filename in filelist]
    for fullpath in fullpaths:
        if fullpath.exists():
            return fullpath

    raise FileNotFoundError(f'None of these files exist: {", ".join([str(x) for x in fullpaths])}')


def join_pdf_files(pdf_list, modelpath_list):

    merger = PdfFileMerger()

    for pdf, modelpath in zip(pdf_list, modelpath_list):
        fullpath = firstexisting([pdf], path=modelpath)
        merger.append(open(fullpath, 'rb'))
        os.remove(fullpath)

    resultfilename = f'{pdf_list[0].split(".")[0]}-{pdf_list[-1].split(".")[0]}'
    with open(f'{resultfilename}.pdf', 'wb') as resultfile:
        merger.write(resultfile)

    print(f'Files merged and saved to {resultfilename}.pdf')


@lru_cache(maxsize=2)
def get_bflist(modelpath, returntype='dict'):
    compositiondata = get_composition_data(modelpath)
    bflist = {}
    with zopen(Path(modelpath, 'bflist.dat'), 'rt') as filein:
        bflistcount = int(filein.readline())

        for k in range(bflistcount):
            rowints = [int(x) for x in filein.readline().split()]
            i, elementindex, ionindex, level = rowints[:4]
            if len(rowints) > 4:
                upperionlevel = rowints[4]
            else:
                upperionlevel = -1
            atomic_number = compositiondata.Z[elementindex]
            ion_stage = ionindex + compositiondata.lowermost_ionstage[elementindex]
            bflist[i] = (atomic_number, ion_stage, level, upperionlevel)

    return bflist


@lru_cache(maxsize=2)
def get_linelist(modelpath, returntype='dict'):
    """Load linestat.out containing transitions wavelength, element, ion, upper and lower levels."""
    with zopen(Path(modelpath, 'linestat.out'), 'rt') as linestatfile:
        lambda_angstroms = [float(wl) * 1e+8 for wl in linestatfile.readline().split()]
        nlines = len(lambda_angstroms)

        atomic_numbers = [int(z) for z in linestatfile.readline().split()]
        assert len(atomic_numbers) == nlines
        ion_stages = [int(ion_stage) for ion_stage in linestatfile.readline().split()]
        assert len(ion_stages) == nlines

        # the file adds one to the levelindex, i.e. lowest level is 1
        upper_levels = [int(levelplusone) - 1 for levelplusone in linestatfile.readline().split()]
        assert len(upper_levels) == nlines
        lower_levels = [int(levelplusone) - 1 for levelplusone in linestatfile.readline().split()]
        assert len(lower_levels) == nlines

    if returntype == 'dict':
        linetuple = namedtuple('line', 'lambda_angstroms atomic_number ionstage upperlevelindex lowerlevelindex')
        linelistdict = {
            index: linetuple(lambda_a, Z, ionstage, upper, lower) for index, lambda_a, Z, ionstage, upper, lower
            in zip(range(nlines), lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels)}
        return linelistdict
    elif returntype == 'dataframe':
        # considering our standard lineline is about 1.5 million lines,
        # using a dataframe make the lookup process very slow
        dflinelist = pd.DataFrame({
            'lambda_angstroms': lambda_angstroms,
            'atomic_number': atomic_numbers,
            'ionstage': ion_stages,
            'upperlevelindex': upper_levels,
            'lowerlevelindex': lower_levels,
        })
        dflinelist.index.name = 'linelistindex'

        return dflinelist


@lru_cache(maxsize=8)
def get_npts_model(modelpath):
    """Return the number of cell in the model.txt."""
    with Path(modelpath, 'model.txt').open('r') as modelfile:
        npts_model = int(modelfile.readline())
    return npts_model


@lru_cache(maxsize=8)
def get_nprocs(modelpath):
    """Return the number of MPI processes specified in input.txt."""
    return int(Path(modelpath, 'input.txt').read_text().split('\n')[21])


@lru_cache(maxsize=8)
def get_inputparams(modelpath):
    """Return parameters specified in input.txt."""
    params = {}
    with Path(modelpath, 'input.txt').open('r') as inputfile:
        params['pre_zseed'] = int(inputfile.readline())

        # number of time steps
        params['ntstep'] = int(inputfile.readline())

        # number of start and end time step
        params['itstep'], params['ftstep'] = [int(x) for x in inputfile.readline().split()]

        params['tmin'], params['tmax'] = [float(x) for x in inputfile.readline().split()]

        params['nusyn_min'], params['nusyn_max'] = [
            (float(x) * u.MeV / const.h).to('Hz') for x in inputfile.readline().split()]

        # number of times for synthesis
        params['nsyn_time'] = int(inputfile.readline())

        # start and end times for synthesis
        params['nsyn_time_start'], params['nsyn_time_end'] = [float(x) for x in inputfile.readline().split()]

        params['n_dimensions'] = int(inputfile.readline())

        # there are more parameters in the file that are not read yet...

    return params


@lru_cache(maxsize=16)
def get_runfolder_timesteps(folderpath):
    """Get the set of timesteps covered by the output files in an ARTIS run folder."""
    folder_timesteps = set()
    try:
        with zopen(Path(folderpath, 'estimators_0000.out'), 'rt') as estfile:
            restart_timestep = -1
            for line in estfile:
                if line.startswith('timestep '):
                    timestep = int(line.split()[1])

                    if (restart_timestep < 0 and timestep != 0):
                        # the first timestep of a restarted run is duplicate and should be ignored
                        restart_timestep = timestep

                    if timestep != restart_timestep:
                        folder_timesteps.add(timestep)

    except FileNotFoundError:
        pass

    return tuple(folder_timesteps)


def get_runfolders(modelpath, timestep=None, timesteps=None):
    """Get a list of folders containing ARTIS output files from a modelpath, optionally with a timestep restriction.

    The folder list may include non-ARTIS folders if a timestep is not specified."""
    folderlist_all = tuple(sorted([child for child in modelpath.iterdir() if child.is_dir()]) + [modelpath])
    folder_list_matching = []
    if (timestep is not None and timestep > -1) or (timesteps is not None and len(timesteps) > 0):
        for folderpath in folderlist_all:
            folder_timesteps = get_runfolder_timesteps(folderpath)
            if timesteps is None and timestep is not None and timestep in folder_timesteps:
                return (folderpath,)
            elif timesteps is not None and any([ts in folder_timesteps for ts in timesteps]):
                folder_list_matching.append(folderpath)

        return tuple(folder_list_matching)

    return [folderpath for folderpath in folderlist_all if get_runfolder_timesteps(folderpath)]


def get_mpiranklist(modelpath, modelgridindex=None):
    if modelgridindex is None or modelgridindex == []:
        return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))
    else:
        try:
            mpiranklist = set()
            for mgi in modelgridindex:
                if mgi < 0:
                    return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))
                else:
                    mpiranklist.add(get_mpirankofcell(mgi, modelpath=modelpath))

            return sorted(list(mpiranklist))
        except TypeError:
            if modelgridindex < 0:
                return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))
            else:
                return [get_mpirankofcell(modelgridindex, modelpath=modelpath)]


def get_cellsofmpirank(mpirank, modelpath):
    """Return an iterable of the cell numbers processed by a given MPI rank."""
    npts_model = get_npts_model(modelpath)
    nprocs = get_nprocs(modelpath)

    assert mpirank < nprocs

    nblock = npts_model // nprocs
    n_leftover = npts_model % nprocs

    if mpirank < n_leftover:
        ndo = nblock + 1
        nstart = mpirank * (nblock + 1)
    else:
        ndo = nblock
        nstart = n_leftover + mpirank * nblock

    return list(range(nstart, nstart + ndo))


def get_mpirankofcell(modelgridindex, modelpath):
    """Return the rank number of the MPI process responsible for handling a specified cell's updating and output."""
    npts_model = get_npts_model(modelpath)
    assert modelgridindex < npts_model

    nprocs = get_nprocs(modelpath)

    if nprocs > npts_model:
        mpirank = modelgridindex
    else:
        nblock = npts_model // nprocs
        n_leftover = npts_model % nprocs

        if modelgridindex <= n_leftover * (nblock + 1):
            mpirank = modelgridindex // (nblock + 1)
        else:
            mpirank = n_leftover + (modelgridindex - n_leftover * (nblock + 1)) // nblock

    assert modelgridindex in get_cellsofmpirank(mpirank, modelpath)

    return mpirank


def addargs(parser):
    pass


def main(argsraw=None):
    """Show a list of available artistools commands."""
    import argcomplete
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=None)

    subparsers = parser.add_subparsers(dest='subcommand')
    subparsers.required = False

    for command, (submodulename, funcname) in sorted(commandlist.items()):
        submodule = importlib.import_module(submodulename, package='artistools')
        subparser = subparsers.add_parser(command)
        submodule.addargs(subparser)
        subparser.set_defaults(func=getattr(submodule, funcname))

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

        for command in commandlist:
            print(f'  {command}')


if __name__ == '__main__':
    main()
