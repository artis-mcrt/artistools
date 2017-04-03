#!/usr/bin/env python3

# import glob
import math

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at


# import os


def readfile(filename):
    lcdata = pd.read_csv(filename, delim_whitespace=True, header=None, names=['time', 'lum', 'lum_cmf'])
    # the light_curve.dat file repeats x values, so keep the first half only
    lcdata = lcdata.iloc[:len(lcdata) // 2]
    return lcdata


def get_from_packets(packetsfiles, timearray, nprocs, vmax, escape_type='TYPE_RPKT'):
    arr_lum_raw = np.zeros_like(timearray)
    arr_lum_cmf_raw = np.zeros_like(timearray)
    betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)
    timearrayplusend = np.append(timearray, 2 * timearray[-1] - timearray[-2])
    arr_timedelta = [at.get_timestep_time_delta(timestep, timearray) for timestep in range(len(timearray))]

    for packetsfile in packetsfiles:
        dfpackets = at.packets.readfile(packetsfile, usecols=[
            'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
            'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'])

        dfpackets.query('type == "TYPE_ESCAPE" and escape_type == @escape_type', inplace=True)
        num_packets = len(dfpackets)
        print(f"{num_packets} {escape_type} packets escaped")

        # the bin is usually a timestep, but could also be -1 or timestep + 1
        dfpackets['t_arrive_bin'] = np.subtract(
            np.digitize([at.packets.t_arrive(packet) * u.s.to('day') for _, packet in dfpackets.iterrows()],
                        timearrayplusend),
            np.ones(num_packets, dtype=np.int))

        dfpackets['t_arrive_cmf_bin'] = np.subtract(
            np.digitize(dfpackets['escape_time'].values * betafactor * u.s.to('day'), timearrayplusend),
            np.ones(num_packets, dtype=np.int))

        arr_lum_raw += np.fromiter(
            (dfpackets.query('t_arrive_bin == @timestep')['e_rf'].sum() for timestep in range(len(timearray))),
            dtype=np.float)
        arr_lum_cmf_raw += np.fromiter(
            (dfpackets.query('t_arrive_cmf_bin == @timestep')['e_cmf'].sum() for timestep in range(len(timearray))),
            dtype=np.float)

    arr_lum = np.divide(arr_lum_raw / nprocs * (u.erg / u.day).to('solLum'), arr_timedelta)
    arr_lum_cmf = np.divide(arr_lum_cmf_raw / nprocs / betafactor * (u.erg / u.day).to('solLum'), arr_timedelta)
    lcdata = pd.DataFrame({'time': timearray, 'lum': arr_lum, 'lum_cmf': arr_lum_cmf})
    return lcdata
