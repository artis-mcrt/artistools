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
    arr_lum = np.zeros(len(timearray))
    arr_lum_cmf = np.zeros(len(timearray))
    beta = (vmax / const.c).decompose().value

    for packetsfile in packetsfiles:
        dfpackets = at.packets.readfile(packetsfile, usecols=[
            'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
            'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'])

        dfpackets.query('type == "TYPE_ESCAPE" and escape_type == @escape_type', inplace=True)
        num_packets = len(dfpackets)
        print(f"{num_packets} escaped {escape_type} packets")

        for index, packet in dfpackets.iterrows():
            # lambda_rf = const.c.to('angstrom/s').value / packet.nu_rf
            t_arrive = at.packets.t_arrive(packet)
            t_arrive_cmf = packet['escape_time'] * math.sqrt(1 - beta ** 2) * u.s.to('day')
            # print(f"Packet escaped at {t_arrive:.1f} days with nu={packet.nu_rf:.2e}, lambda={lambda_rf:.1f}")

            for timestep, time in enumerate(timearray[:-1]):

                if time < t_arrive < timearray[timestep + 1]:
                    arr_lum[timestep] += (packet.e_rf * u.erg / (
                        at.get_timestep_time_delta(timestep, timearray) * u.day) /
                                          nprocs).to('solLum').value

                if time < t_arrive_cmf < timearray[timestep + 1]:
                    arr_lum_cmf[timestep] += (packet.e_cmf * u.erg / (
                            at.get_timestep_time_delta(timestep, timearray) * u.day) /
                                              nprocs / math.sqrt(1 - beta ** 2)).to('solLum').value

    lcdata = pd.DataFrame({'time': timearray, 'lum': arr_lum, 'lum_cmf': arr_lum_cmf})
    return lcdata
