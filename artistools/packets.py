#!/usr/bin/env python3

# import math
import os.path

# import matplotlib.patches as mpatches
# import numpy as np
import pandas as pd
from astropy import constants as const
# from astropy import units as u

# from collections import namedtuple


columns = (
    'number', 'where', 'type_id', 'posx', 'posy', 'posz', 'dirx', 'diry',
    'dirz', 'last_cross', 'tdecay', 'e_cmf', 'e_rf', 'nu_cmf', 'nu_rf',
    'escape_type_id', 'escape_time', 'scat_count', 'next_trans',
    'interactions', 'last_event', 'emission_type', 'true_emission_type',
    'em_posx', 'em_posy', 'em_posz', 'absorption_type', 'absorption_freq',
    'nscatterings', 'em_time', 'absorptiondirx', 'absorptiondiry',
    'absorptiondirz', 'stokes1', 'stokes2', 'stokes3', 'pol_dirx', 'pol_diry',
    'pol_dirz', 'originated_from_positron'
)

types = {
    32: 'TYPE_ESCAPE',
    11: 'TYPE_RPKT',
    10: 'TYPE_GAMMA',
}


def readfile(packetsfile, usecols):
    print(f'Reading {packetsfile} ({os.path.getsize(packetsfile) / 1024 / 1024:.3f} MiB)', end='')
    dfpackets = pd.read_csv(
        packetsfile,
        delim_whitespace=True,
        names=columns,
        header=None,
        usecols=usecols)
    dfpackets['type'] = dfpackets['type_id'].map(lambda x: types.get(x, x))
    dfpackets['escape_type'] = dfpackets['escape_type_id'].map(lambda x: types.get(x, x))
    print(f' ({len(dfpackets):.1e} packets)')

    return dfpackets


def t_arrive(packet):
    """
        time in seconds
    """
    return (packet['escape_time'] -
            (packet['posx'] * packet['dirx'] + packet['posy'] * packet['diry']
             + packet['posz'] * packet['dirz']) / const.c.to('cm/s').value)
