#!/usr/bin/env python3

import math
import os

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from astropy import constants as const

import artistools as af

refspectralabels = {
    '2010lp_20110928_fors2.txt':
        'SN2010lp +264d (Taubenberger et al. 2013)',

    'dop_dered_SN2013aa_20140208_fc_final.txt':
        'SN2013aa +360d (Maguire et al. in prep)',

    '2003du_20031213_3219_8822_00.txt':
        'SN2003du +221.3d (Stanishev et al. 2007)',

    'FranssonJerkstrand2015_W7_330d_1Mpc':
        'Fransson & Jerkstrand (2015) W7 Iwamoto+1999 1Mpc +330d',

    'nero-nebspec.txt':
        'NERO +300d one-zone',

    'maurer2011_RTJ_W7_338d_1Mpc.txt':
        'RTJ W7 Nomoto+1984 1Mpc +338d (Maurer et al. 2011)'
}


def stackspectra(spectra_and_factors):
    factor_sum = sum([factor for _, factor in spectra_and_factors])

    for index, (spectrum, factor) in enumerate(spectra_and_factors):
        if index == 0:
            stackedspectrum = spectrum * factor / factor_sum
        else:
            stackedspectrum = stackedspectrum + (spectrum * factor / factor_sum)

    return stackedspectrum


def get_spectrum(specfilename, timesteplow, timestephigh=-1, fnufilterfunc=None):
    """
        Return a pandas DataFrame containing an ARTIS emergent spectrum
    """
    if timestephigh < 0:
        timestephigh = timesteplow

    specdata = pd.read_csv(specfilename, delim_whitespace=True)

    arraynu = specdata.loc[:, '0'].values
    timearray = specdata.columns.values[1:]

    array_fnu = stackspectra([
        (specdata[specdata.columns[timestep + 1]], af.get_timestep_time_delta(timestep, timearray))
        for timestep in range(timesteplow, timestephigh + 1)])

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        print("Applying filter")
        array_fnu = fnufilterfunc(array_fnu)

    dfspectrum = pd.DataFrame({'nu': arraynu, 'f_nu': array_fnu})

    dfspectrum['lambda_angstroms'] = const.c.to('angstrom/s').value / dfspectrum['nu']
    dfspectrum['f_lambda'] = dfspectrum['f_nu'] * dfspectrum['nu'] / dfspectrum['lambda_angstroms']

    return dfspectrum


def get_spectrum_from_packets(packetsfiles, timelowdays, timehighdays, lambda_min, lambda_max):
    # delta_lambda = (lambda_max - lambda_min) / 500
    delta_lambda = 20
    array_lambda = np.arange(lambda_min, lambda_max, delta_lambda)
    array_energysum = np.zeros(len(array_lambda))  # total packet energy sum of each bin

    columns = ['number', 'where', 'type', 'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz', 'last_cross', 'tdecay',
               'e_cmf', 'e_rf', 'nu_cmf', 'nu_rf', 'escape_type', 'escape_time', 'scat_count', 'next_trans',
               'interactions', 'last_event', 'emission_type', 'true_emission_type', 'em_posx', 'em_posy', 'em_poz',
               'absorption_type', 'absorption_freq', 'nscatterings', 'em_time', 'absorptiondirx', 'absorptiondiry',
               'absorptiondirz', 'stokes1', 'stokes2', 'stokes3', 'pol_dirx', 'pol_diry', 'pol_dirz']

    PARSEC = 3.0857e+18  # pc to cm [pc/cm]
    timelow = timelowdays * 86400
    timehigh = timehighdays * 86400
    nprocs = len(packetsfiles)  # hopefully this is true
    TYPE_ESCAPE = 32,
    TYPE_RPKT = 11,
    c_cgs = const.c.to('cm/s')
    c_ang_s = const.c.to('angstrom/s').value
    nu_min = c_ang_s / lambda_max
    nu_max = c_ang_s / lambda_min
    for packetsfile in packetsfiles:
        print(f"Loading {packetsfile}")
        dfpackets = pd.read_csv(packetsfile, delim_whitespace=True, names=columns, header=None, usecols=[
            'type', 'e_rf', 'nu_rf', 'escape_type', 'escape_time', 'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'])
        # pos_dot_dir = packet.posx * packet.dirx + packet.posy * packet.diry + packet.posz * packet.dirz
        # dfpackets['t_arrive'] = sfpackets['escape_time'] - (pos_dot_dir / 2.99792458e+10)
        dfpackets.query('type == @TYPE_ESCAPE and escape_type == @TYPE_RPKT and'
                        '@nu_min <= nu_rf < @nu_max and'
                        '@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh',
                        inplace=True)
        num_packets = len(dfpackets)
        print(f"{num_packets} escaped r-packets with matching nu and arrival time")
        for index, packet in dfpackets.iterrows():
            lambda_rf = c_ang_s / packet.nu_rf
            # print(f"Packet escaped at {t_arrive / 86400:.1f} days with nu={packet.nu_rf:.2e}, lambda={lambda_rf:.1f}")
            xindex = math.floor((lambda_rf - lambda_min) / delta_lambda)
            assert(xindex >= 0)
            array_energysum[xindex] += packet.e_rf

    array_flambda = array_energysum / delta_lambda / (timehigh - timelow) / 4.e12 / math.pi / PARSEC / PARSEC / nprocs

    return pd.DataFrame({'lambda_angstroms': array_lambda, 'f_lambda': array_flambda})


def plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None, scale_to_peak=None,
                           **plotkwargs):
    """
        Plot reference spectra listed in args.refspecfiles
    """
    if args.refspecfiles is not None:
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        colorlist = ['black', '0.4']
        refspectra = [(fn, refspectralabels.get(fn, fn), c) for fn, c in zip(args.refspecfiles, colorlist)]
        for (filename, serieslabel, linecolor) in refspectra:
            filepath = os.path.join(scriptdir, 'refspectra', filename)
            specdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                                   names=['lambda_angstroms', 'f_lambda'], usecols=[0, 1])

            specdata.query('lambda_angstroms > @args.xmin and lambda_angstroms < @args.xmax', inplace=True)

            print(f"'{serieslabel}' has {len(specdata)} points initially in the plot range")

            if len(specdata) > 5000:
                # specdata = scipy.signal.resample(specdata, 10000)
                # specdata = specdata.iloc[::3, :].copy()
                specdata.query('index % 3 == 0', inplace=True)
                print(f"  downsamping to {len(specdata)} points")

            # clamp negative values to zero
            specdata['f_lambda'] = specdata['f_lambda'].apply(lambda x: max(0, x))

            if flambdafilterfunc:
                specdata['f_lambda'] = specdata['f_lambda'].apply(flambdafilterfunc)

            if args.normalised:
                specdata['f_lambda_scaled'] = (specdata['f_lambda'] / specdata['f_lambda'].max() *
                                               (scale_to_peak if scale_to_peak else 1.0))
                ycolumnname = 'f_lambda_scaled'
            else:
                ycolumnname = 'f_lambda'

            if 'linewidth' not in plotkwargs and 'lw' not in plotkwargs:
                plotkwargs['linewidth'] = 1.5

            specdata.plot(x='lambda_angstroms', y=ycolumnname, ax=axis,
                          label=serieslabel, zorder=-1, color=linecolor, **plotkwargs)
            plotobjects.append(mpatches.Patch(color=linecolor))
            plotobjectlabels.append(serieslabel)
