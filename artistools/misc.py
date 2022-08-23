#!/usr/bin/env python3

import argparse
from functools import lru_cache
import gzip
# import inspect
import lzma
import math
import os.path
import sys
import time
from collections import namedtuple
from itertools import chain
from functools import wraps
# from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable

# import scipy.signal
import numpy as np
import pandas as pd

import artistools as at

plt.style.use('file://' + str(at.config['path_artistools_dir'] / 'matplotlibrc'))

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')


def diskcache(ignoreargs=[], ignorekwargs=[], saveonly=False, quiet=False, savezipped=False,
              funcdepends=None, funcversion=None):
    import pickle
    import hashlib

    def printopt(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    @wraps(diskcache)
    def diskcacheinner(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # save cached files in the folder of the first file/folder specified in the arguments
            modelpath = None
            for arg in [*args, *kwargs.values()]:
                if modelpath is None:
                    try:
                        if os.path.isfile(arg):
                            modelpath = Path(arg).parent

                    except TypeError:
                        pass

            for arg in [*args, *kwargs.values()]:
                if modelpath is None:
                    try:
                        if os.path.isdir(arg):
                            modelpath = arg
                    except TypeError:
                        pass

            if modelpath is None:
                modelpath = Path()  # use current folder

            cachefolder = Path(modelpath, '__artistoolscache__.nosync')

            if cachefolder.is_dir():
                try:
                    import xattr
                    xattr.setxattr(cachefolder, "com.dropbox.ignored", b'1')
                except OSError:
                    pass
                except ModuleNotFoundError:
                    pass

            namearghash = hashlib.sha1()
            namearghash.update(func.__module__.encode('utf-8'))
            namearghash.update(func.__qualname__.encode('utf-8'))

            namearghash.update(
                str(tuple(arg for argindex, arg in enumerate(args) if argindex not in ignoreargs)).encode('utf-8'))

            namearghash.update(str({k: v for k, v in kwargs.items() if k not in ignorekwargs}).encode('utf-8'))

            namearghash_strhex = namearghash.hexdigest()

            # make sure modifications to any file paths in the arguments will trigger an update
            argfilesmodifiedhash = hashlib.sha1()
            for arg in args:
                try:
                    if os.path.isfile(arg):
                        argfilesmodifiedhash.update(str(os.path.getmtime(arg)).encode('utf-8'))
                except TypeError:
                    pass
            argfilesmodifiedhash_strhex = '_filesmodifiedhash_' + argfilesmodifiedhash.hexdigest()

            filename_nogz = Path(cachefolder, f'cached-{func.__module__}.{func.__qualname__}-{namearghash_strhex}.tmp')
            filename_xz = filename_nogz.with_suffix('.tmp.xz')
            filename_gz = filename_nogz.with_suffix('.tmp.gz')

            execfunc = True
            saveresult = False
            functime = -1

            if (filename_nogz.exists() or filename_xz.exists() or filename_gz.exists()) and not saveonly:
                # found a candidate file, so load it
                filename = (
                    filename_nogz if filename_nogz.exists() else filename_gz if filename_gz.exists() else filename_xz)

                filesize = Path(filename).stat().st_size / 1024 / 1024

                try:
                    printopt(f"diskcache: Loading '{filename}' ({filesize:.1f} MiB)...")

                    with zopen(filename, 'rb') as f:
                        result, version_filein = pickle.load(f)

                    if version_filein == str_funcversion + argfilesmodifiedhash_strhex:
                        execfunc = False
                    elif (not funcversion) and (not version_filein.startswith('funcversion_')):
                        execfunc = False
                    # elif version_filein == sourcehash_strhex:
                    #     execfunc = False
                    else:
                        printopt(f"diskcache: Overwriting '{filename}' (function version mismatch or file modified)")

                except Exception as ex:
                    # ex = sys.exc_info()[0]
                    printopt(f"diskcache: Overwriting '{filename}' (Error: {ex})")
                    pass

            if execfunc:
                timestart = time.perf_counter()
                result = func(*args, **kwargs)
                functime = time.perf_counter() - timestart

            if functime > 1:
                # slow functions are worth saving to disk
                saveresult = True
            else:
                # check if we need to replace the gzipped or non-gzipped file with the correct one
                # if we so, need to save the new file even though functime is unknown since we read
                # from disk version instead of executing the function
                if savezipped and filename_nogz.exists():
                    saveresult = True
                elif not savezipped and filename_xz.exists():
                    saveresult = True

            if saveresult:
                # if the cache folder doesn't exist, create it
                if not cachefolder.is_dir():
                    cachefolder.mkdir(parents=True, exist_ok=True)
                try:
                    import xattr
                    xattr.setxattr(cachefolder, "com.dropbox.ignored", b'1')
                except OSError:
                    pass
                except ModuleNotFoundError:
                    pass

                if filename_nogz.exists():
                    filename_nogz.unlink()
                if filename_gz.exists():
                    filename_gz.unlink()
                if filename_xz.exists():
                    filename_xz.unlink()

                fopen, filename = (lzma.open, filename_xz) if savezipped else (open, filename_nogz)
                with fopen(filename, 'wb') as f:
                    pickle.dump((result, str_funcversion + argfilesmodifiedhash_strhex), f,
                                protocol=pickle.HIGHEST_PROTOCOL)

                filesize = Path(filename).stat().st_size / 1024 / 1024
                printopt(f"diskcache: Saved '{filename}' ({filesize:.1f} MiB, functime {functime:.1f}s)")

            return result

        # sourcehash = hashlib.sha1()
        # sourcehash.update(inspect.getsource(func).encode('utf-8'))
        # if funcdepends:
        #     try:
        #         for f in funcdepends:
        #             sourcehash.update(inspect.getsource(f).encode('utf-8'))
        #     except TypeError:
        #         sourcehash.update(inspect.getsource(funcdepends).encode('utf-8'))
        #
        # sourcehash_strhex = sourcehash.hexdigest()
        str_funcversion = f'funcversion_{funcversion}' if funcversion else 'funcversion_none'

        return wrapper if at.config['enable_diskcache'] else func

    return diskcacheinner


class CustomArgHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def add_arguments(self, actions):
        def my_sort(arg):
            opstr = arg.option_strings[0] if len(arg.option_strings) > 0 else ''
            # chars = 'abcdefghijklmnopqrstuvwxyz-'
            opstr = opstr.upper().replace('-', 'z')  # push dash chars below alphabet

            return opstr

        actions = sorted(actions, key=my_sort)
        super(CustomArgHelpFormatter, self).add_arguments(actions)


class AppendPath(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        # if getattr(args, self.dest) is None:
        #     setattr(args, self.dest, [])
        if isinstance(values, Iterable):
            pathlist = getattr(args, self.dest)
            # not pathlist avoids repeated appending of the same items when called from Python
            # instead of from the command line
            if not pathlist:
                for pathstr in values:
                    # print(f"pathstr {pathstr}")
                    # if Path(pathstr) not in pathlist:
                    pathlist.append(Path(pathstr))
        else:
            setattr(args, self.dest, Path(values))


def make_namedtuple(typename, **fields):
    """Make a namedtuple from a dictionary of attributes and values.
    Example: make_namedtuple('mytuple', x=2, y=3)"""
    return namedtuple(typename, fields)(*fields.values())


def showtimesteptimes(modelpath=None, numberofcolumns=5, args=None):
    """Print a table showing the timesteps and their corresponding times."""
    if modelpath is None:
        modelpath = Path()

    print('Timesteps and midpoint times in days:\n')

    times = get_timestep_times_float(modelpath, loc='mid')
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

    rowdfs = []
    with open(filename, 'r') as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline()  # T_preset
        fcompdata.readline()  # homogeneous_abundances
        startindex = 0
        for _ in range(nelements):
            line = fcompdata.readline()
            linesplit = line.split()
            row_list = list(map(int, linesplit[:5])) + list(map(float, linesplit[5:])) + [startindex]

            rowdfs.append(pd.DataFrame([row_list], columns=columns))

            startindex += int(rowdfs[-1]['nions'])

    compdf = pd.concat(rowdfs, ignore_index=True)

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

    composition_df = pd.DataFrame(
        [(Z, atomic_composition[Z]) for Z in atomic_composition.keys()], columns=['Z', 'nions'])
    composition_df['lowermost_ionstage'] = [1] * composition_df.shape[0]
    composition_df['uppermost_ionstage'] = composition_df['nions']
    return composition_df


def gather_res_data(res_df, index_of_repeated_value=1):
    """res files repeat output for each angle.
    index_of_repeated_value is the value to look for repeating eg. time of ts 0.
    In spec_res files it's 1, but in lc_res file it's 0"""
    index_to_split = res_df.index[res_df.iloc[:, index_of_repeated_value]
                                  == res_df.iloc[0, index_of_repeated_value]]
    res_data = []
    for i, index_value in enumerate(index_to_split):
        if index_value != index_to_split[-1]:
            chunk = res_df.iloc[index_to_split[i]:index_to_split[i + 1], :]
        else:
            chunk = res_df.iloc[index_to_split[i]:, :]
        res_data.append(chunk)
    return res_data


def match_closest_time(reftime, searchtimes):
    """Get time closest to reftime in list of times (searchtimes)"""
    return str("{}".format(min([float(x) for x in searchtimes], key=lambda x: abs(x - reftime))))


def get_vpkt_config(modelpath):
    filename = Path(modelpath, 'vpkt.txt')
    vpkt_config = {}
    with open(filename, 'r') as vpkt_txt:
        vpkt_config['nobsdirections'] = int(vpkt_txt.readline())
        vpkt_config['cos_theta'] = [float(x) for x in vpkt_txt.readline().split()]
        vpkt_config['phi'] = [float(x) for x in vpkt_txt.readline().split()]
        nspecflag = int(vpkt_txt.readline())

        if nspecflag == 1:
            vpkt_config['nspectraperobs'] = int(vpkt_txt.readline())
            for i in range(vpkt_config['nspectraperobs']):
                vpkt_txt.readline()
        else:
            vpkt_config['nspectraperobs'] = 1

        vpkt_config['time_limits_enabled'], vpkt_config['initial_time'], vpkt_config['final_time'] = [
            int(x) for x in vpkt_txt.readline().split()]

    return vpkt_config


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


def get_wid_init_at_tmin(modelpath):
    # cell width in cm at time tmin
    day_to_sec = 86400
    tmin = get_timestep_times_float(modelpath, loc='start')[0] * day_to_sec
    _, _, vmax = at.get_modeldata(modelpath)

    rmax = vmax * tmin

    coordmax0 = rmax
    ncoordgrid0 = 50

    wid_init = 2 * coordmax0 / ncoordgrid0
    return wid_init


def get_wid_init_at_tmodel(modelpath=None, ngridpoints=None, t_model_days=None, xmax=None):
    if ngridpoints is None or t_model_days is None or xmax is None:
        # Luke: ngridpoint only equals the number of model cells if the model is 3D
        dfmodel, t_model_days, vmax = at.get_modeldata(modelpath)
        assert at.inputmodel.get_dfmodel_dimensions(dfmodel) == 3
        ngridpoints = len(dfmodel)
        xmax = vmax * t_model_days * (24 * 60 * 60)

    ncoordgridx = round(ngridpoints ** (1. / 3.))

    wid_init = 2 * xmax / ncoordgridx

    return wid_init


def get_syn_dir(modelpath):
    with open(modelpath / 'syn_dir.txt', 'r') as syn_dir_file:
        syn_dir = [int(x) for x in syn_dir_file.readline().split()]

    return syn_dir


def dot(x, y):
    return (x[0] * y[0]) + (x[1] * y[1]) + (x[2] * y[2])


def cross_prod(v1, v2, v3):
    v3[0] = (v1[1] * v2[2]) - (v2[1] * v1[2])
    v3[1] = (v1[2] * v2[0]) - (v2[2] * v1[0])
    v3[2] = (v1[0] * v2[1]) - (v2[0] * v1[1])
    return v3


def vec_len(vec):
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


@lru_cache(maxsize=16)
def get_nu_grid(modelpath):
    """Get an array of frequencies at which the ARTIS spectra are binned by exspec."""
    specfilename = firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    return specdata.loc[:, '0'].values


def get_deposition(modelpath):
    if Path(modelpath).is_file():
        depfilepath = modelpath
        modelpath = Path(modelpath).parent
    else:
        depfilepath = Path(modelpath, 'deposition.out')

    ts_mids = get_timestep_times_float(modelpath, loc='mid')

    with open(depfilepath, 'r') as fdep:
        filepos = fdep.tell()
        line = fdep.readline()
        if line.startswith('#'):
            columns = line.lstrip('#').split()
        else:
            fdep.seek(filepos)  # undo the readline() and go back
            columns = ['tmid_days', 'gammadep_Lsun', 'positrondep_Lsun', 'total_dep_Lsun']

        depdata = pd.read_csv(fdep, delim_whitespace=True, header=None, names=columns)

    depdata.index.name = 'timestep'

    # no timesteps are given in the old format of deposition.out, so ensure that
    # the times in days match up with the times of our assumed timesteps
    for timestep, row in depdata.iterrows():
        assert(abs(ts_mids[timestep] / row['tmid_days'] - 1) < 0.01)  # deposition times don't match input.txt

    return depdata


@lru_cache(maxsize=16)
def get_timestep_times(modelpath):
    """Return a list of the mid time in days of each timestep from a spec.out file."""
    try:
        specfilename = firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
        time_columns = pd.read_csv(specfilename, delim_whitespace=True, nrows=0)
        return time_columns.columns[1:]
    except FileNotFoundError:
        return [f'{tdays:.3f}' for tdays in get_timestep_times_float(modelpath, loc='mid')]


@lru_cache(maxsize=16)
def get_timestep_times_float(modelpath, loc='mid'):
    """Return a list of the time in days of each timestep."""

    modelpath = Path(modelpath)

    # virtual path to code comparison workshop models
    if not modelpath.exists() and modelpath.parts[0] == 'codecomparison':
        import artistools.codecomparison
        return artistools.codecomparison.get_timestep_times_float(modelpath=modelpath, loc=loc)

    # custom timestep file
    tsfilepath = Path(modelpath, 'timesteps.out')
    if tsfilepath.exists():
        dftimesteps = pd.read_csv(tsfilepath, delim_whitespace=True, escapechar='#', index_col='timestep')
        if loc == 'mid':
            return dftimesteps.tmid_days.values
        elif loc == 'start':
            return dftimesteps.tstart_days.values
        elif loc == 'end':
            return dftimesteps.tstart_days.values + dftimesteps.twidth_days.values
        elif loc == 'delta':
            return dftimesteps.twidth_days.values
        else:
            raise ValueError("loc must be one of 'mid', 'start', 'end', or 'delta'")

    # older versions of Artis always used logarithmic timesteps and didn't produce a timesteps.out file
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
    arr_tend = get_timestep_times_float(modelpath, loc='end')
    # to avoid roundoff errors, use the next timestep's tstart at each timestep's tend (t_width is not exact)
    arr_tend[:-1] = arr_tstart[1:]

    for ts, (tstart, tend) in enumerate(zip(arr_tstart, arr_tend)):
        if timedays_float >= tstart and timedays_float < tend:
            return ts

    raise ValueError(f"Could not find timestep bracketing time {timedays_float}")
    assert(False)
    return


def get_time_range(modelpath, timestep_range_str=None, timemin=None, timemax=None, timedays_range_str=None):
    """Handle a time range specified in either days or timesteps."""
    # assertions make sure time is specified either by timesteps or times in days, but not both!
    tstarts = get_timestep_times_float(modelpath, loc='start')
    tmids = get_timestep_times_float(modelpath, loc='mid')
    tends = get_timestep_times_float(modelpath, loc='end')

    timedays_is_specified = (timemin is not None and timemax is not None) or timedays_range_str is not None

    if timemin and timemin > tends[-1]:
        print(f"{get_model_name(modelpath)}: WARNING timemin {timemin} is after the last timestep at {tends[-1]:.1f}")
        return -1, -1, timemin, timemax
    elif timemax and timemax < tstarts[0]:
        print(f"{get_model_name(modelpath)}: WARNING timemax {timemax} is "
              f"before the first timestep at {tstarts[0]:.1f}")
        return -1, -1, timemin, timemax

    if timestep_range_str is not None:
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
    """Return the time in days of the midpoint of a timestep number """
    timearray = get_timestep_times_float(modelpath, loc='mid')
    if timearray is not None:
        return timearray[timestep]

    return -1


def get_escaped_arrivalrange(modelpath):
    dfmodel, t_model_init_days, vmax = at.inputmodel.get_modeldata(modelpath, printwarningsonly=True)
    cornervmax = math.sqrt(3 * vmax ** 2)

    # find the earliest possible escape time and add the largest possible travel time

    # for 3D models, the box corners can have non-zero density (allowing packet escape from tmin)
    # for 1D and 2D, the largest escape radius at tmin is the box side radius
    vmax_tmin = cornervmax if at.inputmodel.get_dfmodel_dimensions(dfmodel) == 3 else vmax

    # earliest completely valid time is tmin plus maximum possible travel time
    validrange_start_days = at.get_timestep_times_float(modelpath, loc='start')[0] * (1 + vmax_tmin / 29979245800)

    # find the last possible escape time and subtract the largest possible travel time
    depdata = at.get_deposition(modelpath=modelpath)  # use this file to find the last computed timestep
    nts_last = depdata.ts.max() if 'ts' in depdata.columns else len(depdata) - 1
    nts_last_tend = at.get_timestep_times_float(modelpath, loc='end')[nts_last]

    # latest possible valid range is the end of the latest computed timestep plus the longest travel time
    validrange_end_days = nts_last_tend * (1 - cornervmax / 29979245800)

    if validrange_start_days > validrange_end_days:
        validrange_start_days, validrange_end_days = None, None

    return nts_last, validrange_start_days, validrange_end_days


@lru_cache(maxsize=8)
def get_model_name(path):
    """Get the name of an ARTIS model from the path to any file inside it.

    Name will be either from a special plotlabel.txt file if it exists or the enclosing directory name
    """

    if not Path(path).exists() and Path(path).parts[0] == 'codecomparison':
        return str(path)

    abspath = os.path.abspath(path)

    modelpath = abspath if os.path.isdir(abspath) else os.path.dirname(abspath)

    try:
        plotlabelfile = os.path.join(modelpath, 'plotlabel.txt')
        return open(plotlabelfile, mode='r').readline().strip()
    except FileNotFoundError:
        return os.path.basename(modelpath)


def get_z_a_nucname(nucname):
    """ return atomic number and mass number from a string like 'Pb208' (returns 92, 208) """
    if nucname.startswith('X_'):
        nucname = nucname[2:]
    z = get_atomic_number(nucname.rstrip('0123456789'))
    assert z > 0
    a = int(nucname.lower().lstrip('abcdefghijklmnopqrstuvwxyz'))
    return z, a


@lru_cache(maxsize=1)
def get_elsymbolslist():
    elsymbols = ['n'] + list(pd.read_csv(
        at.config['path_datadir'] / 'elements.csv', usecols=['symbol'])['symbol'].values)

    return elsymbols


def get_atomic_number(elsymbol):
    assert elsymbol is not None
    if elsymbol.startswith('X_'):
        elsymbol = elsymbol[2:]

    elsymbol = elsymbol.split('_')[0].split('-')[0].rstrip('0123456789')

    if elsymbol.title() in get_elsymbolslist():
        return get_elsymbolslist().index(elsymbol.title())

    return -1


def decode_roman_numeral(strin):
    if strin.upper() in roman_numerals:
        return roman_numerals.index(strin.upper())
    return -1


def get_elsymbol(atomic_number):
    return get_elsymbolslist()[atomic_number]


@lru_cache(maxsize=16)
def get_ionstring(atomic_number, ionstage, spectral=True, nospace=False):
    if ionstage == 'ALL' or ionstage is None:
        return f'{get_elsymbol(atomic_number)}'
    elif spectral:
        return f"{get_elsymbol(atomic_number)}{' ' if not nospace else ''}{roman_numerals[ionstage]}"
    else:
        # ion notion e.g. Co+, Fe2+
        if ionstage > 2:
            strcharge = r'$^{' + str(ionstage - 1) + r'{+}}$'
        elif ionstage == 2:
            strcharge = r'$^{+}$'
        else:
            strcharge = ''
        return f'{get_elsymbol(atomic_number)}{strcharge}'


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


def makelist(x):
    """If x is not a list (or is a string), make a list containing x."""
    if x is None:
        return []
    elif isinstance(x, (str, Path)):
        return [x, ]
    else:
        return x


def trim_or_pad(requiredlength, *listoflistin):
    """Make lists equal in length to requiedlength either by padding with None or truncating"""
    for listin in listoflistin:
        listin = makelist(listin)

        if len(listin) < requiredlength:
            listout = listin.copy()
            listout.extend([None for _ in range(requiredlength - len(listin))])
        elif len(listin) > requiredlength:
            listout = listin[:requiredlength]
        else:
            listout = listin

        assert(len(listout) == requiredlength)
        yield listout


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

    raise FileNotFoundError(f'None of these files exist in {path}: {", ".join([str(x) for x in fullpaths])}')


def stripallsuffixes(f):
    """Take a file path (e.g. packets00_0000.out.gz) and return the Path with no suffixes (e.g. packets)"""
    f_nosuffixes = Path(f)
    for _ in f.suffixes:
        f_nosuffixes = f_nosuffixes.with_suffix('')  # each call removes only one suffix

    return f_nosuffixes


def readnoncommentline(file):
    """Read a line from the text file, skipping blank and comment lines that begin with #"""

    line = ''

    while not line.strip() or line.strip().lstrip().startswith('#'):
        line = file.readline()

    return line


@lru_cache(maxsize=24)
def get_file_metadata(filepath):
    def add_derived_metadata(metadata):

        if 'a_v' in metadata and 'e_bminusv' in metadata and 'r_v' not in metadata:
            metadata['r_v'] = metadata['a_v'] / metadata['e_bminusv']
        elif 'e_bminusv' in metadata and 'r_v' in metadata and 'a_v' not in metadata:
            metadata['a_v'] = metadata['e_bminusv'] * metadata['r_v']
        elif 'a_v' in metadata and 'r_v' in metadata and 'e_bminusv' not in metadata:
            metadata['e_bminusv'] = metadata['a_v'] / metadata['r_v']

        return metadata

    import yaml
    filepath = Path(filepath)

    # check if the reference file (e.g. spectrum.txt) has an metadata file (spectrum.txt.meta.yml)
    individualmetafile = filepath.with_suffix(filepath.suffix + '.meta.yml')
    if individualmetafile.exists():
        with individualmetafile.open('r') as yamlfile:
            metadata = yaml.load(yamlfile, Loader=yaml.FullLoader)

        return add_derived_metadata(metadata)

    # check if the metadata is in the big combined metadata file (todo: eliminate this file)
    combinedmetafile = Path(filepath.parent.resolve(), 'metadata.yml')
    if combinedmetafile.exists():
        with combinedmetafile.open('r') as yamlfile:
            combined_metadata = yaml.load(yamlfile, Loader=yaml.FullLoader)
        metadata = combined_metadata.get(str(filepath), {})

        return add_derived_metadata(metadata)

    return {}


def get_filterfunc(args, mode='interp'):
    """Using command line arguments to determine the appropriate filter function."""

    if hasattr(args, "filtermovingavg") and args.filtermovingavg > 0:
        def filterfunc(ylist):
            n = args.filtermovingavg
            arr_padded = np.pad(ylist, (n // 2, n - 1 - n // 2), mode='edge')
            return np.convolve(arr_padded, np.ones((n,)) / n, mode='valid')

    elif hasattr(args, "filtersavgol") and args.filtersavgol:
        import scipy.signal
        window_length, poly_order = [int(x) for x in args.filtersavgol]

        def filterfunc(ylist):
            return scipy.signal.savgol_filter(ylist, window_length, poly_order, mode=mode)
        print("Applying Savitzky–Golay filter")
    else:
        filterfunc = None

    return filterfunc


def join_pdf_files(pdf_list, modelpath_list):
    from PyPDF2 import PdfFileMerger

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


@lru_cache(maxsize=16)
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
        npts_model = int(readnoncommentline(modelfile))
    return npts_model


@lru_cache(maxsize=8)
def get_nprocs(modelpath):
    """Return the number of MPI processes specified in input.txt."""
    return int(Path(modelpath, 'input.txt').read_text().split('\n')[21].split('#')[0])


@lru_cache(maxsize=8)
def get_inputparams(modelpath):
    """Return parameters specified in input.txt."""
    from astropy import units as u
    from astropy import constants as const

    params = {}
    with Path(modelpath, 'input.txt').open('r') as inputfile:
        params['pre_zseed'] = int(readnoncommentline(inputfile).split('#')[0])

        # number of time steps
        params['ntstep'] = int(readnoncommentline(inputfile).split('#')[0])

        # number of start and end time step
        params['itstep'], params['ftstep'] = [int(x) for x in readnoncommentline(inputfile).split('#')[0].split()]

        params['tmin'], params['tmax'] = [float(x) for x in readnoncommentline(inputfile).split('#')[0].split()]

        params['nusyn_min'], params['nusyn_max'] = [
            (float(x) * u.MeV / const.h).to('Hz') for x in readnoncommentline(inputfile).split('#')[0].split()]

        # number of times for synthesis
        params['nsyn_time'] = int(readnoncommentline(inputfile).split('#')[0])

        # start and end times for synthesis
        params['nsyn_time_start'], params['nsyn_time_end'] = [
            float(x) for x in readnoncommentline(inputfile).split('#')[0].split()]

        params['n_dimensions'] = int(readnoncommentline(inputfile).split('#')[0])

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

                    if (restart_timestep < 0 and timestep != 0 and 0 not in folder_timesteps):
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
    folderlist_all = tuple(sorted([child for child in Path(modelpath).iterdir() if child.is_dir()]) + [Path(modelpath)])
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


def get_mpiranklist(modelpath, modelgridindex=None, only_ranks_withgridcells=False):
    """
        Get a list of rank ids. Parameters:
        - modelpath:
            pathlib.Path() to ARTIS model folder
        - modelgridindex:
            give a cell number to only return the rank number that updates this cell (and outputs its estimators)
        - only_ranks_withgridcells:
            set True to skip ranks that only update packets (i.e. that don't update any grid cells/output estimators)
    """
    if modelgridindex is None or modelgridindex == []:
        if only_ranks_withgridcells:
            return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))
        return range(get_nprocs(modelpath))
    else:
        try:
            mpiranklist = set()
            for mgi in modelgridindex:
                if mgi < 0:
                    if only_ranks_withgridcells:
                        return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))
                    return range(get_nprocs(modelpath))
                else:
                    mpiranklist.add(get_mpirankofcell(mgi, modelpath=modelpath))

            return sorted(list(mpiranklist))

        except TypeError:
            # in case modelgridindex is a single number rather than an iterable
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


@lru_cache(maxsize=16)
def get_dfrankassignments(modelpath):
    filerankassignments = Path(modelpath, 'modelgridrankassignments.out')
    if filerankassignments.is_file():
        df = pd.read_csv(filerankassignments, delim_whitespace=True)
        df.rename(columns={df.columns[0]: df.columns[0].lstrip('#')}, inplace=True)
        return df
    return None


def get_mpirankofcell(modelgridindex, modelpath):
    """Return the rank number of the MPI process responsible for handling a specified cell's updating and output."""
    npts_model = get_npts_model(modelpath)
    assert modelgridindex < npts_model

    dfrankassignments = get_dfrankassignments(modelpath)
    if dfrankassignments is not None:
        dfselected = dfrankassignments.query(
            'ndo > 0 and nstart <= @modelgridindex and (nstart + ndo - 1) >= @modelgridindex')
        assert len(dfselected) == 1
        return int(dfselected.iloc[0]['rank'])

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


def get_artis_constants(modelpath=None, srcpath=None, printdefs=False):
    # get artis options specified as preprocessor macro definitions in artisoptions.h and other header files
    if not srcpath:
        srcpath = Path(modelpath, 'artis')
        if not modelpath:
            raise ValueError('Either modelpath or srcpath must be specified in call to get_defines()')

    cfiles = [
        # Path(srcpath, 'constants.h'),
        # Path(srcpath, 'decay.h'),
        Path(srcpath, 'artisoptions.h'),
        # Path(srcpath, 'sn3d.h'),
    ]
    definedict = {
        'true': True,
        'false':  False,
    }
    for filepath in cfiles:
        definedict.update(parse_cdefines(srcfilepath=filepath))

    # evaluate booleans, numbers, and references to other constants
    for k, strvalue in definedict.copy().items():
        try:
            # definedict[k] = eval(strvalue, definedict)
            # print(f"{k} = '{strvalue}' = {definedict[k]}")
            pass
        except SyntaxError:
            pass
            # print(f"{k} = '{strvalue}' = (COULD NOT EVALUATE)")
        except TypeError:
            pass
            # print(f"{k} = '{strvalue}' = (COULD NOT EVALUATE)")

    # if printdefs:
    #     for k in definedict:
    #         print(f"{k} = '{definedict[k]}'")

    return definedict


def parse_cdefines(srcfilepath=None, printdefs=False):
    # adapted from h2py.py in Python source
    import re

    # p_define = re.compile('^[\t ]*#[\t ]*define[\t ]+([a-zA-Z0-9_]+)[\t ]+')
    p_define = re.compile(r'^[\t ]*#[\t ]*define[\t ]+([a-zA-Z0-9_]+)+')

    p_const = re.compile(r'(?:\w+\s+)([a-zA-Z_=][a-zA-Z0-9_=]*)*(?<!=)=(?!=)')

    p_comment = re.compile(r'/\*([^*]+|\*+[^/])*(\*+/)?')
    p_cpp_comment = re.compile('//.*')

    ignores = [p_comment, p_cpp_comment]

    p_char = re.compile(r"'(\\.[^\\]*|[^\\])'")

    p_hex = re.compile(r"0x([0-9a-fA-F]+)L?")

    def pytify(body):
        # replace ignored patterns by spaces
        for p in ignores:
            body = p.sub(' ', body)
        # replace char literals by ord(...)
        body = p_char.sub('ord(\\0)', body)
        # Compute negative hexadecimal constants
        start = 0
        UMAX = 2*(sys.maxsize+1)
        while 1:
            m = p_hex.search(body, start)
            if not m:
                break
            s, e = m.span()
            val = int(body[slice(*m.span(1))], 16)
            if val > sys.maxsize:
                val -= UMAX
                body = body[:s] + "(" + str(val) + ")" + body[e:]
            start = s + 1
        return body

    definedict = {}
    lineno = 0
    with open(srcfilepath, 'r') as optfile:
        while 1:
            line = optfile.readline()
            if not line:
                break
            lineno = lineno + 1
            match = p_define.match(line)
            if match:
                # gobble up continuation lines
                while line[-2:] == '\\\n':
                    nextline = optfile.readline()
                    if not nextline:
                        break
                    lineno = lineno + 1
                    line = line + nextline
                name = match.group(1)
                body = line[match.end():]
                body = pytify(body)
                definedict[name] = body.strip()
            match = p_const.match(line)
            if match:
                print('CONST', tuple(p_const.findall(line)))
            # if '=' in line and ';' in line:
            #     tokens = line.replace('==', 'IGNORE').replace('=', ' = ').split()
            #     varname = tokens.indexof('=')[-1]

    if printdefs:
        for k in definedict:
            print(f"{k} = '{definedict[k]}'")

    return definedict
