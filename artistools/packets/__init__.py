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

CLIGHT = 2.99792458e10
DAY = 86400

types = {
    10: 'TYPE_GAMMA',
    11: 'TYPE_RPKT',
    20: 'TYPE_NTLEPTON',
    32: 'TYPE_ESCAPE',
}

type_ids = dict((v, k) for k, v in types.items())


@at.diskcache(savezipped=True)
def get_column_names(modelpath):
    modelpath = Path(modelpath)
    if Path('artis').is_dir():
        print('detected artis code directory')
        packet_properties = []
        inputfilename = at.firstexisting(
            ['packet_init.cc', 'packet_init.c'], path=(modelpath / 'artis'))
        print(f'found {inputfilename}: getting packet column names from artis code')
        with open(inputfilename) as inputfile:
            packet_print_lines = [line.split(',') for line in inputfile if 'fprintf(packets_file,' in line]
            for line in packet_print_lines:
                for element in line:
                    if 'pkt[i].' in element:
                        packet_properties.append(element)

        for i, element in enumerate(packet_properties):
            packet_properties[i] = element.split('.')[1].split(')')[0]

        columns = packet_properties
        replacements_dict = {'type': 'type_id',
                             'pos[0]': 'posx', 'pos[1]': 'posy', 'pos[2]': 'posz',
                             'dir[0]': 'dirx', 'dir[1]': 'diry', 'dir[2]': 'dirz',
                             'escape_type': 'escape_type_id',
                             'em_pos[0]': 'em_posx', 'em_pos[1]': 'em_posy', 'em_pos[2]': 'em_posz',
                             'absorptiontype': 'absorption_type', 'absorptionfreq': 'absorption_freq',
                             'absorptiondir[0]': 'absorptiondirx', 'absorptiondir[1]': 'absorptiondiry', 'absorptiondir[2]': 'absorptiondirz',
                             'stokes[0]': 'stokes1', 'stokes[1]': 'stokes2', 'stokes[2]': 'stokes3',
                             'pol_dir[0]': 'pol_dirx', 'pol_dir[1]': 'pol_diry', 'pol_dir[2]': 'pol_dirz'}
        for i, column_name in enumerate(columns):
            if column_name in replacements_dict:
                columns[i] = replacements_dict[column_name]

    else:
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
    return columns


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

        dfpackets.eval(
            "em_velx = em_posx / em_time",
            inplace=True)
        dfpackets.eval(
            "em_vely = em_posy / em_time",
            inplace=True)
        dfpackets.eval(
            "em_velz = em_posz / em_time",
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


def readfile_text(packetsfile, modelpath=Path('.')):
    columns = get_column_names(modelpath)
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

    angle_number = np.zeros(len(dfpackets))
    for pkt_index, _ in dfpackets.iterrows():
        pkt_dir = [dfpackets['dirx'][pkt_index], dfpackets['diry'][pkt_index], dfpackets['dirz'][pkt_index]]
        costheta = at.dot(pkt_dir, syn_dir)
        thetabin = ((costheta + 1.0) * np.sqrt(MABINS) / 2.0)
        vec1 = vec2 = vec3 = [0, 0, 0]
        xhat = [1, 0, 0]
        vec1 = at.cross_prod(pkt_dir, syn_dir, vec1)
        vec2 = at.cross_prod(xhat, syn_dir, vec2)
        cosphi = at.dot(vec1, vec2) / at.vec_len(vec1) / at.vec_len(vec2)

        vec3 = at.cross_prod(vec2, syn_dir, vec3)
        testphi = at.dot(vec1, vec3)

        if testphi > 0:
            phibin = (math.acos(cosphi) / 2. / np.pi * np.sqrt(MABINS))
        else:
            phibin = ((math.acos(cosphi) + np.pi) / 2. / np.pi * np.sqrt(MABINS))
        na = (thetabin * np.sqrt(MABINS)) + phibin  ## think na is angle number???
        if na >= 100:
            print(f'error bin number too high {na}')
        angle_number[pkt_index] = int(na)

    dfpackets['angle_bin'] = angle_number
    return dfpackets


def make_3d_histogram_from_packets(modelpath, timestep):
    modeldata, _, vmax_cms = at.inputmodel.get_modeldata(modelpath)

    timeminarray = at.get_timestep_times_float(modelpath=modelpath, loc='start')
    timedeltaarray = at.get_timestep_times_float(modelpath=modelpath, loc='delta')
    timemaxarray = at.get_timestep_times_float(modelpath=modelpath, loc='end')

    # timestep = 63 # 82 73 #63 #54 46 #27
    print([(ts, time) for ts, time in enumerate(timeminarray)])

    packetsfiles = at.packets.get_packetsfilepaths(modelpath)

    emission_position3d = [[], [], []]
    e_rf = []

    for npacketfile in range(0, len(packetsfiles)):
        # for npacketfile in range(0, 1):
        dfpackets = at.packets.readfile(packetsfiles[npacketfile])
        at.packets.add_derived_columns(dfpackets, modelpath, ['emission_velocity'])
        dfpackets = dfpackets.dropna(subset=['emission_velocity'])  # drop rows where emission_vel is NaN

        # print(dfpackets[['emission_velocity', 'em_velx', 'em_vely', 'em_velz']])
        # select only type escape and type r-pkt (don't include gamma-rays)
        dfpackets.query(f'type_id == {type_ids["TYPE_ESCAPE"]} and escape_type_id == {type_ids["TYPE_RPKT"]}',
                        inplace=True)
        dfpackets.query('@timeminarray[@timestep] < escape_time/@DAY < @timemaxarray[@timestep]', inplace=True)

        emission_position3d[0].extend([em_velx / CLIGHT for em_velx in dfpackets['em_velx']])
        emission_position3d[1].extend([em_vely / CLIGHT for em_vely in dfpackets['em_vely']])
        emission_position3d[2].extend([em_velz / CLIGHT for em_velz in dfpackets['em_velz']])

        e_rf.extend([e_rf for e_rf in dfpackets['e_rf']])

    emission_position3d = np.array(emission_position3d)
    e_rf = np.array(e_rf)
    print(emission_position3d.shape)
    print(emission_position3d[0].shape)

    # print(emission_position3d)
    grid_3d, _, _, _ = make_3d_grid(modeldata, vmax_cms)
    print(grid_3d)
    # https://stackoverflow.com/questions/49861468/binning-random-data-to-regular-3d-grid-with-unequal-axis-lengths
    hist, _ = np.histogramdd(emission_position3d.T, [np.append(ax, np.inf) for ax in grid_3d], weights=e_rf)

    # print(hist.shape)
    hist = hist / len(packetsfiles) / timedeltaarray[timestep]  # histogram weighted by energy
    # - need to divide by number of processes
    # and length of timestep
    return hist


def make_3d_grid(modeldata, vmax_cms):
    # modeldata, _, vmax_cms = at.inputmodel.get_modeldata(modelpath)
    grid = round(len(modeldata['inputcellid']) ** (1. / 3.))
    xgrid = np.zeros(grid)
    vmax = vmax_cms / CLIGHT
    i = 0
    for z in range(0, grid):
        for y in range(0, grid):
            for x in range(0, grid):
                xgrid[x] = -vmax + 2 * x * vmax / grid
                i += 1

    x, y, z = np.meshgrid(xgrid, xgrid, xgrid)
    grid_3d = np.array([xgrid, xgrid, xgrid])
    # grid_Te = np.zeros((grid, grid, grid))
    # print(grid_Te.shape)
    return grid_3d, x, y, z


def make_packets_plot(modelpath, timestep):
    import pyvista as pv
    modeldata, _, vmax_cms = at.inputmodel.get_modeldata(modelpath)
    _, x, y, z = make_3d_grid(modeldata, vmax_cms)
    mesh = pv.StructuredGrid(x, y, z)

    hist = make_3d_histogram_from_packets(modelpath, timestep)

    mesh['energy [erg/s]'] = hist.ravel(order='F')
    # print(max(mesh['energy [erg/s]']))

    sargs = dict(height=0.75, vertical=True, position_x=0.04, position_y=0.1,
                 title_font_size=22, label_font_size=25)

    pv.set_plot_theme("document")  # set white background
    p = pv.Plotter()
    p.set_scale(1.5, 1.5, 1.5)
    # single_slice = mesh.slice(normal='y')
    single_slice = mesh.slice(normal='z')
    p.add_mesh(single_slice, scalar_bar_args=sargs)
    p.show_bounds(grid=False, xlabel='vx / c', ylabel='vy / c', zlabel='vz / c',
                  ticks='inside', minor_ticks=False, use_2d=True, font_size=26, bold=False)
    # labels = dict(xlabel='vx / c', ylabel='vy / c', zlabel='vz / c')
    # p.show_grid(**labels)
    p.camera_position = 'xy'
    timeminarray = at.get_timestep_times_float(modelpath=modelpath, loc='start')
    time = timeminarray[timestep]
    p.add_title(f'{time:.2f} - {timeminarray[timestep + 1]:.2f} days')
    print(pv.global_theme)

    p.show(screenshot=modelpath / f'3Dplot_pktsemitted{time:.1f}days_disk.png')
