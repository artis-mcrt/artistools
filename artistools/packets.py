#!/usr/bin/env python3

# import math
import os

# import matplotlib.patches as mpatches
# import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at

# from collections import namedtuple



columns = [
    'number', 'where', 'type_id', 'posx', 'posy', 'posz', 'dirx', 'diry',
    'dirz', 'last_cross', 'tdecay', 'e_cmf', 'e_rf', 'nu_cmf', 'nu_rf',
    'escape_type_id', 'escape_time', 'scat_count', 'next_trans',
    'interactions', 'last_event', 'emission_type', 'true_emission_type',
    'em_posx', 'em_posy', 'em_poz', 'absorption_type', 'absorption_freq',
    'nscatterings', 'em_time', 'absorptiondirx', 'absorptiondiry',
    'absorptiondirz', 'stokes1', 'stokes2', 'stokes3', 'pol_dirx', 'pol_diry',
    'pol_dirz'
]

types = {
    32: 'TYPE_ESCAPE',
    11: 'TYPE_RPKT',
}


def readfiles(packetsfiles, usecols):
    for index, packetsfile in enumerate(packetsfiles):
        dfpackets = readfile(packetsfile, usecols)
        if index == 0:
            dfpackets_all = dfpackets
        else:
            dfpackets_all = dfpackets_all.append(dfpackets, ignore_index=True)

    return dfpackets_all


def readfile(packetsfile, usecols):
    print(f'Reading from {packetsfile} ({os.path.getsize(packetsfile) / 1024 / 1024:.3f} MiB)')
    dfpackets = pd.read_csv(
        packetsfile,
        delim_whitespace=True,
        names=columns,
        header=None,
        usecols=usecols)
    dfpackets['type'] = dfpackets['type_id'].map(lambda x: types.get(x, x))
    dfpackets['escape_type'] = dfpackets['escape_type_id'].map(
        lambda x: types.get(x, x))

    return dfpackets


def t_arrive(packet):
    return (packet['escape_time'] -
            (packet['posx'] * packet['dirx'] + packet['posy'] * packet['diry']
             + packet['posz'] * packet['dirz']) / const.c.to('cm/s').value
            ) * u.s.to('day')
