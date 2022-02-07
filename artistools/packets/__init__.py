#!/usr/bin/env python3

import math
import gzip
# import multiprocessing
import sys
from pathlib import Path

# import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# from collections import namedtuple
from functools import lru_cache

import artistools as at


columns = (
    'number',
    'where',
    'type_id',
    'posx', 'posy', 'posz',
    'dirx', 'diry', 'dirz',
    'last_cross',
    'tdecay',
    'e_cmf',
    'e_rf',
    'nu_cmf',
    'nu_rf',
    'escape_type_id',
    'escape_time',
    'scat_count',
    'next_trans',
    'interactions',
    'last_event',
    'emissiontype',
    'trueemissiontype',
    'em_posx', 'em_posy', 'em_posz',
    'absorption_type',
    'absorption_freq',
    'nscatterings',
    'em_time',
    'absorptiondirx',
    'absorptiondiry',
    'absorptiondirz', 'stokes1', 'stokes2', 'stokes3', 'pol_dirx', 'pol_diry',
    'pol_dirz',
    'originated_from_positron',
    'true_emission_velocity',
    'trueem_time',
    'pellet_nucindex',
)


types = {
    10: 'TYPE_GAMMA',
    11: 'TYPE_RPKT',
    20: 'TYPE_NTLEPTON',
    32: 'TYPE_ESCAPE',
}

type_ids = dict((v, k) for k, v in types.items())


def add_derived_columns(dfpackets, modelpath, colnames, allnonemptymgilist=None):
    cm_to_km = 1e-5
    day_in_s = 86400
    if dfpackets.empty:
        return dfpackets

    colnames = at.makelist(colnames)

    def em_modelgridindex(packet):
        return at.get_mgi_of_velocity_kms(modelpath, packet.emission_velocity * cm_to_km,
                                          mgilist=allnonemptymgilist)

    def emtrue_modelgridindex(packet):
        return at.get_mgi_of_velocity_kms(modelpath, packet.true_emission_velocity * cm_to_km,
                                          mgilist=allnonemptymgilist)

    def em_timestep(packet):
        return at.get_timestep_of_timedays(modelpath, packet.em_time / day_in_s)

    def emtrue_timestep(packet):
        return at.get_timestep_of_timedays(modelpath, packet.trueem_time / day_in_s)

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


def readfile_text(packetsfile):
    try:
        inputcolumncount = len(pd.read_csv(packetsfile, nrows=1, delim_whitespace=True, header=None).columns)
        if inputcolumncount < 3:
            print("\nWARNING: packets file has no columns!")
            print(open(packetsfile, "r").readlines())

    except gzip.BadGzipFile:
        print(f"\nBad Gzip File: {packetsfile}")
        raise gzip.BadGzipFile

    # the packets file may have a truncated set of columns, but we assume that they
    # are only truncated, i.e. the columns with the same index have the same meaning
    usecols_nodata = [n for n in columns if columns.index(n) >= inputcolumncount]
    # usecols_actual = [n for n in columns if columns.index(n) < inputcolumncount]

    try:
        dfpackets = pd.read_csv(
            packetsfile, delim_whitespace=True,
            names=columns[:inputcolumncount], header=None)
    except Exception as ex:
        print(f'Problem with file {packetsfile}')
        print(f'ERROR: {ex}')
        sys.exit(1)

    if usecols_nodata:
        print(f'WARNING: no data in packets file for columns: {usecols_nodata}')
        for col in usecols_nodata:
            dfpackets[col] = float('NaN')

    return dfpackets


@at.diskcache(savezipped=True)
def readfile(packetsfile, type=None, escape_type=None):
    """Read a packet file into a pandas DataFrame."""
    packetsfile = Path(packetsfile)

    if packetsfile.suffixes == ['.out', '.parquet']:
        dfpackets = pd.read_parquet(packetsfile)
    elif packetsfile.suffixes == ['.out', '.feather']:
        dfpackets = pd.read_feather(packetsfile)
    elif packetsfile.suffixes in [['.out'], ['.out', '.gz'], ['.out', '.xz']]:
        dfpackets = readfile_text(packetsfile)
        # dfpackets.to_parquet(at.stripallsuffixes(packetsfile).with_suffix('.out.parquet'),
        #                      compression='brotli', compression_level=99)
    else:
        print('ERROR')
        sys.exit(1)
    filesize = Path(packetsfile).stat().st_size / 1024 / 1024
    print(f'Reading {packetsfile} ({filesize:.1f} MiB)', end='')

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

    # # neglect light travel time correction
    # dfpackets.eval("t_arrive_d = escape_time / 86400", inplace=True)

    dfpackets.eval(
        "t_arrive_d = (escape_time - (posx * dirx + posy * diry + posz * dirz) / 29979245800) / 86400", inplace=True)

    return dfpackets


@lru_cache(maxsize=16)
def get_packetsfilepaths(modelpath, maxpacketfiles=None):

    def preferred_alternative(f, files):
        f_nosuffixes = at.stripallsuffixes(f)

        suffix_priority = [['.out', '.gz'], ['.out', '.xz'], ['.out', '.feather'], ['.out', '.parquet']]
        if f.suffixes in suffix_priority:
            startindex = suffix_priority.index(f.suffixes) + 1
        else:
            startindex = 0

        if any(f_nosuffixes.with_suffix(''.join(s)).is_file() for s in suffix_priority[startindex:]):
            return True
        return False

    packetsfiles = sorted(
        list(Path(modelpath).glob('packets00_*.out*')) +
        list(Path(modelpath, 'packets').glob('packets00_*.out*')))

    # strip out duplicates in the case that some are stored as binary and some are text files
    packetsfiles = [f for f in packetsfiles if not preferred_alternative(f, packetsfiles)]

    if maxpacketfiles is not None and maxpacketfiles > 0 and len(packetsfiles) > maxpacketfiles:
        print(f'Using only the first {maxpacketfiles} of {len(packetsfiles)} packets files')
        packetsfiles = packetsfiles[:maxpacketfiles]

    return packetsfiles


def get_escaping_packet_angle_bin(modelpath, dfpackets):
    MABINS = 100

    with open(modelpath / 'syn_dir.txt', 'r') as syn_dir_file:
        syn_dir = [int(x) for x in syn_dir_file.readline().split()]

    def dot(x, y):
        return (x[0] * y[0]) + (x[1] * y[1]) + (x[2] * y[2])

    def cross_prod(v1, v2, v3):
        v3[0] = (v1[1] * v2[2]) - (v2[1] * v1[2])
        v3[1] = (v1[2] * v2[0]) - (v2[2] * v1[0])
        v3[2] = (v1[0] * v2[1]) - (v2[0] * v1[1])
        return v3

    def vec_len(vec):
        return np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

    angle_number = np.zeros(len(dfpackets))
    for pkt_index, _ in dfpackets.iterrows():
        pkt_dir = [dfpackets['dirx'][pkt_index], dfpackets['diry'][pkt_index], dfpackets['dirz'][pkt_index]]
        costheta = dot(pkt_dir, syn_dir)
        thetabin = ((costheta + 1.0) * np.sqrt(MABINS) / 2.0)
        vec1 = vec2 = vec3 = [0, 0, 0]
        xhat = [1, 0, 0]
        vec1 = cross_prod(pkt_dir, syn_dir, vec1)
        vec2 = cross_prod(xhat, syn_dir, vec2)
        cosphi = dot(vec1, vec2) / vec_len(vec1) / vec_len(vec2)

        vec3 = cross_prod(vec2, syn_dir, vec3)
        testphi = dot(vec1, vec3)

        if testphi > 0:
            phibin = (math.acos(cosphi) / 2. / np.pi * np.sqrt(MABINS))
        else:
            phibin = ((math.acos(cosphi) + np.pi) / 2. / np.pi * np.sqrt(MABINS))
        na = (thetabin * np.sqrt(MABINS)) + phibin  # think na is angle number???
        angle_number[pkt_index] = int(na)

    dfpackets['angle_bin'] = angle_number
    return dfpackets
