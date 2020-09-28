#!/usr/bin/env python3

# import math
import glob
import multiprocessing
from pathlib import Path

# import matplotlib.patches as mpatches
# import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

# from collections import namedtuple
from functools import lru_cache

import artistools as at


columns = (
    'number', 'where', 'type_id', 'posx', 'posy', 'posz', 'dirx', 'diry',
    'dirz', 'last_cross', 'tdecay', 'e_cmf', 'e_rf', 'nu_cmf', 'nu_rf',
    'escape_type_id', 'escape_time', 'scat_count', 'next_trans',
    'interactions', 'last_event', 'emissiontype', 'trueemissiontype',
    'em_posx', 'em_posy', 'em_posz', 'absorption_type', 'absorption_freq',
    'nscatterings', 'em_time', 'absorptiondirx', 'absorptiondiry',
    'absorptiondirz', 'stokes1', 'stokes2', 'stokes3', 'pol_dirx', 'pol_diry',
    'pol_dirz', 'originated_from_positron', 'true_emission_velocity'
)

types = {
    10: 'TYPE_GAMMA',
    11: 'TYPE_RPKT',
    20: 'TYPE_NTLEPTON',
    32: 'TYPE_ESCAPE',
}

type_ids = dict((v, k) for k, v in types.items())


def add_derived_columns(dfpackets, modelpath, colnames, allnonemptymgilist=None):
    if dfpackets.empty:
        return dfpackets

    colnames = at.makelist(colnames)

    def em_modelgridindex(packet):
        return at.get_mgi_of_velocity_kms(modelpath, packet.emission_velocity * u.cm.to('km'),
                                          mgilist=allnonemptymgilist)

    def emtrue_modelgridindex(packet):
        return at.get_mgi_of_velocity_kms(modelpath, packet.true_emission_velocity * u.cm.to('km'), mgilist=allnonemptymgilist)

    def em_timestep(packet):
        return at.get_timestep_of_timedays(modelpath, packet.em_time * u.s.to('day'))

    if 'emission_velocity' in colnames:
        dfpackets.eval(
            "emission_velocity = sqrt(em_posx ** 2 + em_posy ** 2 + em_posz ** 2) / em_time",
            inplace=True)

    if 'em_modelgridindex' in colnames:
        if 'emission_velocity' not in dfpackets.columns:
            dfpackets = add_derived_columns(dfpackets, modelpath, ['emission_velocity'],
                                            allnonemptymgilist=allnonemptymgilist)
        dfpackets['em_modelgridindex'] = dfpackets.apply(em_modelgridindex, axis=1)

    if 'emtrue_modelgridindex' in colnames:
        dfpackets['emtrue_modelgridindex'] = dfpackets.apply(emtrue_modelgridindex, axis=1)

    if 'em_timestep' in colnames:
        dfpackets['em_timestep'] = dfpackets.apply(em_timestep, axis=1)

    return dfpackets


@at.diskcache(savegzipped=True)
def readfile(packetsfile, type=None, escape_type=None):
    """Read a packet file into a pandas DataFrame."""
    filesize = Path(packetsfile).stat().st_size / 1024 / 1024

    print(f'Reading {packetsfile} ({filesize:.1f} MiB)', end='')
    inputcolumncount = len(pd.read_csv(packetsfile, nrows=1, delim_whitespace=True, header=None).columns)
    if inputcolumncount < 3:
        print("\nWARNING: packets file has no columns!")
        print(open(packetsfile, "r").readlines())

    # the packets file may have a truncated set of columns, but we assume that they
    # are only truncated, i.e. the columns with the same index have the same meaning
    usecols_nodata = [n for n in columns if columns.index(n) >= inputcolumncount]
    # usecols_actual = [n for n in columns if columns.index(n) < inputcolumncount]

    dfpackets = pd.read_csv(
        packetsfile, delim_whitespace=True,
        names=columns[:inputcolumncount], header=None)

    print(f' ({len(dfpackets):.1e} packets', end='')

    if escape_type is not None and escape_type != '' and escape_type != 'ALL':
        assert type is None or type == 'TYPE_ESCAPE'
        dfpackets.query(f'type_id == {type_ids["TYPE_ESCAPE"]} and escape_type_id == {type_ids[escape_type]}',
                        inplace=True)
        print(f', {len(dfpackets)} escaped as {escape_type})')
    elif type is not None and type != 'ALL' and type != '':
        dfpackets.query(f'type_id == {type_ids[type]}', inplace=True)
        print(f', {len(dfpackets)} with type {type})')
    else:
        print(')')

    # dfpackets['type'] = dfpackets['type_id'].map(lambda x: types.get(x, x))
    # dfpackets['escape_type'] = dfpackets['escape_type_id'].map(lambda x: types.get(x, x))

    if usecols_nodata:
        print(f'WARNING: no data in packets file for columns: {usecols_nodata}')
        for col in usecols_nodata:
            dfpackets[col] = float('NaN')

    # # neglect light travel time correction
    # dfpackets.eval("t_arrive_d = escape_time * @u.s.to('day')", inplace=True)

    dfpackets.eval(
        "t_arrive_d = (escape_time - "
        "(posx * dirx + posy * diry + posz * dirz) / @const.c.to('cm/s').value) * @u.s.to('day')", inplace=True)

    return dfpackets


@lru_cache(maxsize=16)
def get_packetsfilepaths(modelpath, maxpacketfiles=None):
    packetsfiles = sorted(
        glob.glob(str(Path(modelpath, 'packets00_*.out*'))) +
        glob.glob(str(Path(modelpath, 'packets', 'packets00_*.out*'))))
    if maxpacketfiles is not None and maxpacketfiles > 0 and len(packetsfiles) > maxpacketfiles:
        print(f'Using only the first {maxpacketfiles} of {len(packetsfiles)} packets files')
        packetsfiles = packetsfiles[:maxpacketfiles]

    return packetsfiles
