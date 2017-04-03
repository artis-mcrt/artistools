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
    betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    arr_timedelta = [at.get_timestep_time_delta(timestep, timearray) for timestep in range(len(timearray))]
    timearrayplusend = np.concatenate([timearray, [timearray[-1] + arr_timedelta[-1]]])

    arr_lum_raw = np.zeros_like(timearray, dtype=np.float)
    arr_lum_cmf_raw = np.zeros_like(timearray, dtype=np.float)

    for packetsfile in packetsfiles:
        dfpackets = at.packets.readfile(packetsfile, usecols=[
            'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
            'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'])

        dfpackets.query('type == "TYPE_ESCAPE" and escape_type == @escape_type', inplace=True)
        print(f"{len(dfpackets)} {escape_type} packets escaped")

        dfpackets['t_arrive_d'] = dfpackets.apply(lambda packet: at.packets.t_arrive(packet) * u.s.to('day'), axis=1)

        binned = pd.cut(dfpackets['t_arrive_d'], timearrayplusend, labels=False, include_lowest=True)
        for binindex, e_rf_sum in dfpackets.groupby(binned)['e_rf'].sum().iteritems():
            arr_lum_raw[int(binindex)] += e_rf_sum

        dfpackets['t_arrive_cmf_d'] = dfpackets['escape_time'] * betafactor * u.s.to('day')

        binned_cmf = pd.cut(dfpackets['t_arrive_cmf_d'], timearrayplusend, labels=False, include_lowest=True)
        for binindex, e_cmf_sum in dfpackets.groupby(binned_cmf)['e_cmf'].sum().iteritems():
            arr_lum_cmf_raw[int(binindex)] += e_cmf_sum

    arr_lum = np.divide(arr_lum_raw / nprocs * (u.erg / u.day).to('solLum'), arr_timedelta)
    arr_lum_cmf = np.divide(arr_lum_cmf_raw / nprocs / betafactor * (u.erg / u.day).to('solLum'), arr_timedelta)
    lcdata = pd.DataFrame({'time': timearray, 'lum': arr_lum, 'lum_cmf': arr_lum_cmf})
    return lcdata
