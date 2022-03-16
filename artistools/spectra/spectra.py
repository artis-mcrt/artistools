#!/usr/bin/env python3
"""Artistools - spectra related functions."""
import math
import multiprocessing
from collections import namedtuple
from functools import lru_cache
from functools import partial
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.pyplot as plt  # needed to get the color map
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
import re

import artistools as at
import artistools.radfield
import artistools.packets

fluxcontributiontuple = namedtuple(
    'fluxcontribution', 'fluxcontrib linelabel array_flambda_emission array_flambda_absorption color')


def timeshift_fluxscale_co56law(scaletoreftime, spectime):
    if scaletoreftime is not None:
        # Co56 decay flux scaling
        assert spectime > 150
        return math.exp(float(spectime) / 113.7) / math.exp(scaletoreftime / 113.7)
    else:
        return 1.


def get_exspec_bins():
    MNUBINS = 1000
    NU_MIN_R = 1e13
    NU_MAX_R = 5e15

    c_ang_s = const.c.to('angstrom/s').value

    dlognu = (math.log(NU_MAX_R) - math.log(NU_MIN_R)) / MNUBINS

    bins_nu_lower = np.array(
        [math.exp(math.log(NU_MIN_R) + (m * (dlognu))) for m in range(MNUBINS)])
    # bins_nu_upper = np.array(
    #     [math.exp(math.log(NU_MIN_R) + ((m + 1) * (dlognu))) for m in range(MNUBINS)])
    bins_nu_upper = bins_nu_lower * math.exp(dlognu)
    bins_nu_centre = 0.5 * (bins_nu_lower + bins_nu_upper)

    array_lambdabinedges = np.append(c_ang_s / np.flip(bins_nu_upper), c_ang_s / bins_nu_lower[0])
    array_lambda = c_ang_s / np.flip(bins_nu_centre)
    delta_lambda = np.flip(c_ang_s / bins_nu_lower - c_ang_s / bins_nu_upper)

    return array_lambdabinedges, array_lambda, delta_lambda


def stackspectra(spectra_and_factors):
    """Add spectra using weighting factors, i.e., specout[nu] = spec1[nu] * factor1 + spec2[nu] * factor2 + ...
    spectra_and_factors should be a list of tuples: spectra[], factor"""

    factor_sum = sum([factor for _, factor in spectra_and_factors])

    stackedspectrum = np.zeros_like(spectra_and_factors[0][0], dtype=float)
    for spectrum, factor in spectra_and_factors:
        stackedspectrum += spectrum * factor / factor_sum

    return stackedspectrum


@lru_cache(maxsize=16)
def get_specdata(modelpath, stokesparam=None):
    polarisationdata = False
    if Path(modelpath, 'specpol.out').is_file():
        specfilename = Path(modelpath) / "specpol.out"
        polarisationdata = True
    elif Path(modelpath).is_dir():
        specfilename = at.firstexisting(['spec.out.xz', 'spec.out.gz', 'spec.out'], path=modelpath)
    else:
        specfilename = modelpath

    if polarisationdata:
        # angle = args.plotviewingangle[0]
        stokes_params = get_specpol_data(angle=None, modelpath=modelpath)
        if stokesparam is not None:
            specdata = stokes_params[stokesparam]
        else:
            specdata = stokes_params['I']
    else:
        assert stokesparam is None
        print(f"Reading {specfilename}")
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        specdata = specdata.rename(columns={'0': 'nu'})

    return specdata


def get_spectrum(
        modelpath, timestepmin: int, timestepmax=-1, fnufilterfunc=None,
        modelnumber=None):
    """Return a pandas DataFrame containing an ARTIS emergent spectrum."""
    if timestepmax < 0:
        timestepmax = timestepmin

    specdata = get_specdata(modelpath)

    nu = specdata.loc[:, 'nu'].values
    arr_tdelta = at.get_timestep_times_float(modelpath, loc='delta')

    f_nu = stackspectra([
        (specdata[specdata.columns[timestep + 1]],
         arr_tdelta[timestep])
        for timestep in range(timestepmin, timestepmax + 1)])

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        print("Applying filter to ARTIS spectrum")
        f_nu = fnufilterfunc(f_nu)

    dfspectrum = pd.DataFrame({'nu': nu, 'f_nu': f_nu})
    dfspectrum.sort_values(by='nu', ascending=False, inplace=True)

    dfspectrum.eval('lambda_angstroms = @c / nu', local_dict={'c': const.c.to('angstrom/s').value}, inplace=True)
    dfspectrum.eval('f_lambda = f_nu * nu / lambda_angstroms', inplace=True)

    # if 'redshifttoz' in args and args.redshifttoz[modelnumber] != 0:
    # #     plt.plot(dfspectrum['lambda_angstroms'], dfspectrum['f_lambda'], color='k')
    #     z = args.redshifttoz[modelnumber]
    #     dfspectrum['lambda_angstroms'] *= (1 + z)
    # #     plt.plot(dfspectrum['lambda_angstroms'], dfspectrum['f_lambda'], color='r')
    # #     plt.show()
    # #     quit()

    return dfspectrum


def get_spectrum_at_time(modelpath, timestep, time, args, angle=None, res_specdata=None, modelnumber=None):
    if angle is not None:
        if args.plotvspecpol and os.path.isfile(modelpath/'vpkt.txt'):
            spectrum = get_vspecpol_spectrum(modelpath, time, angle, args)
        elif os.path.isfile(modelpath/'specpol_res.out'):
            spectrum = get_res_spectrum(modelpath, timestep, timestep, angle=angle, res_specdata=res_specdata)
        else:
            spectrum = get_spectrum(modelpath, timestep, timestep, modelnumber=modelnumber)
    else:
        spectrum = get_spectrum(modelpath, timestep, timestep, modelnumber=modelnumber)

    return spectrum


def get_spectrum_from_packets_worker(querystr, qlocals, array_lambda, array_lambdabinedges, packetsfile,
                                     use_comovingframe=False, getpacketcount=False, betafactor=None):
    dfpackets = at.packets.readfile(packetsfile, type='TYPE_ESCAPE', escape_type='TYPE_RPKT').query(
        querystr, inplace=False, local_dict=qlocals)

    print(f"  {packetsfile}: {len(dfpackets)} escaped r-packets matching frequency and arrival time ranges ")

    dfpackets.eval('lambda_rf = @c_ang_s / nu_rf', inplace=True, local_dict=qlocals)
    wl_bins = pd.cut(
        x=dfpackets['lambda_rf'], bins=array_lambdabinedges, right=True,
        labels=range(len(array_lambda)), include_lowest=True)

    if use_comovingframe:
        array_energysum_onefile = dfpackets.e_cmf.groupby(wl_bins).sum().values / betafactor
    else:
        array_energysum_onefile = dfpackets.e_rf.groupby(wl_bins).sum().values

    if getpacketcount:
        array_pktcount_onefile = dfpackets.lambda_rf.groupby(wl_bins).count().values
    else:
        array_pktcount_onefile = None

    return array_energysum_onefile, array_pktcount_onefile


def get_spectrum_from_packets(
        modelpath, timelowdays, timehighdays, lambda_min, lambda_max,
        delta_lambda=None, use_comovingframe=None, maxpacketfiles=None, useinternalpackets=False,
        getpacketcount=False):
    """Get a spectrum dataframe using the packets files as input."""
    assert(not useinternalpackets)
    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles)

    if use_comovingframe:
        modeldata, _, _ = at.inputmodel.get_modeldata(Path(packetsfiles[0]).parent)
        vmax = modeldata.iloc[-1].velocity_outer * u.km / u.s
        betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)
    else:
        betafactor = None

    c_cgs = const.c.to('cm/s').value
    c_ang_s = const.c.to('angstrom/s').value
    nu_min = c_ang_s / lambda_max
    nu_max = c_ang_s / lambda_min

    if delta_lambda:
        array_lambdabinedges = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
        array_lambda = 0.5 * (array_lambdabinedges[:-1] + array_lambdabinedges[1:])  # bin centres
    else:
        array_lambdabinedges, array_lambda, delta_lambda = get_exspec_bins()

    array_energysum = np.zeros_like(array_lambda, dtype=float)  # total packet energy sum of each bin
    if getpacketcount:
        array_pktcount = np.zeros_like(array_lambda, dtype=int)  # number of packets in each bin

    timelow = timelowdays * u.day.to('s')
    timehigh = timehighdays * u.day.to('s')

    nprocs_read = len(packetsfiles)
    querystr = '@nu_min <= nu_rf < @nu_max and trueemissiontype >= 0 and '
    if not use_comovingframe:
        querystr += '@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh'
    else:
        querystr += '@timelow < (escape_time * @betafactor) < @timehigh'

    processfile = partial(
        get_spectrum_from_packets_worker, querystr,
        dict(nu_min=nu_min, nu_max=nu_max, timelow=timelow, timehigh=timehigh,
             betafactor=betafactor, c_cgs=c_cgs, c_ang_s=c_ang_s),
        array_lambda, array_lambdabinedges, use_comovingframe=use_comovingframe, getpacketcount=getpacketcount,
        betafactor=betafactor)
    if at.num_processes > 1:
        with multiprocessing.Pool(processes=at.num_processes) as pool:
            results = pool.map(processfile, packetsfiles)
            pool.close()
            pool.join()
            pool.terminate()
    else:
        results = [processfile(p) for p in packetsfiles]

    array_energysum = np.ufunc.reduce(np.add, [r[0] for r in results])
    if getpacketcount:
        array_pktcount += np.ufunc.reduce(np.add, [r[1] for r in results])

    array_flambda = (array_energysum / delta_lambda / (timehigh - timelow) /
                     4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs_read)

    dfdict = {
        'lambda_angstroms': array_lambda,
        'f_lambda': array_flambda,
        'energy_sum': array_energysum,
    }

    if getpacketcount:
        dfdict['packetcount'] = array_pktcount

    return pd.DataFrame(dfdict)


@at.diskcache(savezipped=True)
def read_specpol_res(modelpath):
    """Return specpol_res data for a given angle"""
    if Path(modelpath, 'specpol_res.out').is_file():
        specfilename = Path(modelpath) / "specpol_res.out"
    else:
        specfilename = modelpath

    print(f"Reading {specfilename} (in read_specpol_res)")
    specdata = pd.read_csv(specfilename, delim_whitespace=True, header=None, dtype=str)

    res_specdata = at.gather_res_data(specdata)

    # index_to_split = specdata.index[specdata.iloc[:, 1] == specdata.iloc[0, 1]]
    # # print(len(index_to_split))
    # res_specdata = []
    # for i, index_value in enumerate(index_to_split):
    #     if index_value != index_to_split[-1]:
    #         chunk = specdata.iloc[index_to_split[i]:index_to_split[i + 1], :]
    #     else:
    #         chunk = specdata.iloc[index_to_split[i]:, :]
    #     res_specdata.append(chunk)
    # print(res_specdata[0])

    columns = res_specdata[0].iloc[0]
    # print(columns)
    for i, res_spec in enumerate(res_specdata):
        res_specdata[i] = res_specdata[i].rename(columns=columns).drop(res_specdata[i].index[0])
        # These lines remove the Q and U values from the dataframe (I think)
        numberofIvalues = len(res_specdata[i].columns.drop_duplicates())
        res_specdata[i] = res_specdata[i].iloc[:, : numberofIvalues]
        res_specdata[i] = res_specdata[i].astype(float)
        res_specdata[i] = res_specdata[i].to_numpy()

    for i, res_spec in enumerate(res_specdata):
        res_specdata[i] = pd.DataFrame(data=res_specdata[i], columns=columns[:numberofIvalues])
        res_specdata[i] = res_specdata[i].rename(columns={'0': 'nu'})

    return res_specdata


def average_angle_bins(res_specdata, angle, args):
    # Averages over 10 bins to reduce noise

    if args and args.average_every_tenth_viewing_angle:
        for start_bin in np.arange(start=0, stop=100, step=10):
            # print(start_bin)
            for bin_number in range(start_bin+1, start_bin+10):
                # print(bin_number)
                res_specdata[start_bin] += res_specdata[bin_number]
            res_specdata[start_bin] /= 10  # every 10th bin is the average of 10 bins
            print(f'bin number {start_bin} = the average of bins {start_bin} to {start_bin + 9}')

        if angle and angle % 10 == 0:
            print(f"Bin number {angle} is the average of 10 angle bins")
    return res_specdata


def get_res_spectrum(
        modelpath, timestepmin: int, timestepmax=-1, angle=None, res_specdata=None, fnufilterfunc=None,
        args=None):

    """Return a pandas DataFrame containing an ARTIS emergent spectrum."""
    if timestepmax < 0:
        timestepmax = timestepmin

    # print(f"Reading spectrum at timestep {timestepmin}")

    if angle is None:
        angle = args.plotviewingangle[0]

    if res_specdata is None:
        print("Reading specpol_res.out")
        res_specdata = read_specpol_res(modelpath)
        if args and 'average_every_tenth_viewing_angle' in args:
            at.spectra.average_angle_bins(res_specdata, angle, args)

    nu = res_specdata[angle].loc[:, 'nu'].values
    arr_tmid = at.get_timestep_times_float(modelpath, loc='mid')
    arr_tdelta = at.get_timestep_times_float(modelpath, loc='delta')

    # for angle in args.plotviewingangle:
    f_nu = stackspectra([(res_specdata[angle][res_specdata[angle].columns[timestep + 1]],
                          arr_tdelta[timestep])
                         for timestep in range(timestepmin, timestepmax + 1)])

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        print("Applying filter to ARTIS spectrum")
        f_nu = fnufilterfunc(f_nu)

    dfspectrum = pd.DataFrame({'nu': nu, 'f_nu': f_nu})
    dfspectrum.sort_values(by='nu', ascending=False, inplace=True)

    dfspectrum.eval('lambda_angstroms = @c / nu', local_dict={'c': const.c.to('angstrom/s').value}, inplace=True)
    dfspectrum.eval('f_lambda = f_nu * nu / lambda_angstroms', inplace=True)
    return dfspectrum


def make_virtual_spectra_summed_file(modelpath):
    nprocs = at.get_nprocs(modelpath)
    print("nprocs", nprocs)
    vspecpol_data_old = []  # virtual packet spectra for each observer (all directions and opacity choices)
    vpktconfig = at.get_vpkt_config(modelpath)
    nvirtual_spectra = vpktconfig['nobsdirections'] * vpktconfig['nspectraperobs']
    print(f"nobsdirections {vpktconfig['nobsdirections']} nspectraperobs {vpktconfig['nspectraperobs']} (total observers: {nvirtual_spectra})")
    for mpirank in range(nprocs):
        vspecpolfilename = f'vspecpol_{mpirank}-0.out'
        print(f"Reading rank {mpirank} filename {vspecpolfilename}")
        vspecpolpath = Path(modelpath, vspecpolfilename)
        if not vspecpolpath.is_file():
            vspecpolpath = Path(modelpath, vspecpolfilename + '.gz')
            if not vspecpolpath.is_file():
                print(f'Warning: Could not find {vspecpolpath.relative_to(modelpath.parent)}')
                continue

        vspecpolfile = pd.read_csv(vspecpolpath, delim_whitespace=True, header=None)
        # Where times of timesteps are written out a new virtual spectrum starts
        # Find where the time in row 0, column 1 repeats in any column 1
        index_of_new_spectrum = vspecpolfile.index[vspecpolfile.iloc[:, 1] == vspecpolfile.iloc[0, 1]]
        vspecpol_data = []  # list of all predefined vspectra
        for i, index_spectrum_starts in enumerate(index_of_new_spectrum[:nvirtual_spectra]):
            # todo: this is different to at.gather_res_data() -- could be made to be same format to not repeat code
            if index_spectrum_starts != index_of_new_spectrum[-1]:
                chunk = vspecpolfile.iloc[index_spectrum_starts:index_of_new_spectrum[i+1], :]
            else:
                chunk = vspecpolfile.iloc[index_spectrum_starts:, :]
            vspecpol_data.append(chunk)

        if len(vspecpol_data_old) > 0:
            for i, _ in enumerate(vspecpol_data):
                dftmp = vspecpol_data[i].copy()  # copy of vspectrum number i in a file
                # add copy to the same spectrum number from previous file
                # (don't need to copy row 1 = time or column 1 = freq)
                dftmp.iloc[1:, 1:] += vspecpol_data_old[i].iloc[1:, 1:]
                # spectrum i then equals the sum of all previous files spectrum number i
                vspecpol_data[i] = dftmp
        # update array containing sum of previous files
        vspecpol_data_old = vspecpol_data

    for spec_index, vspecpol in enumerate(vspecpol_data):
        outfile = modelpath / f'vspecpol_total-{spec_index}.out'
        print(f'Saved {outfile}')
        vspecpol.to_csv(outfile, sep=' ', index=False, header=False)


def make_averaged_vspecfiles(args):
    filenames = []
    for vspecfile in os.listdir(args.modelpath[0]):
        if vspecfile.startswith('vspecpol_total-'):
            filenames.append(vspecfile)

    def sorted_by_number(l):

        def convert(text):
            return int(text) if text.isdigit() else text

        def alphanum_key(key):
            return [convert(c) for c in re.split('([0-9]+)', key)]

        return sorted(l, key=alphanum_key)

    filenames = sorted_by_number(filenames)

    for spec_index, filename in enumerate(filenames):  # vspecpol-total files
        vspecdata = []
        for modelpath in args.modelpath:
            vspecdata.append(pd.read_csv(modelpath / filename, delim_whitespace=True, header=None))
        for i in range(1, len(vspecdata)):
            vspecdata[0].iloc[1:, 1:] += vspecdata[i].iloc[1:, 1:]

        vspecdata[0].iloc[1:, 1:] = vspecdata[0].iloc[1:, 1:]/len(vspecdata)
        vspecdata[0].to_csv(args.modelpath[0] / f'vspecpol_averaged-{spec_index}.out',
                            sep=' ', index=False, header=False)


def get_specpol_data(angle=None, modelpath=None, specdata=None):
    if specdata is None:
        if angle is None:
            specfilename = Path(modelpath, 'specpol.out')
        else:
            # alternatively use f'vspecpol_averaged-{angle}.out' ?
            specfilename = Path(modelpath, f'vspecpol_total-{angle}.out')
            if not specfilename.exists():
                print(f"{specfilename} does not exist. Generating all-rank summed vspec files..")
                make_virtual_spectra_summed_file(modelpath=modelpath)

        print(f"Reading {specfilename}")
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        specdata = specdata.rename(columns={specdata.keys()[0]: 'nu'})

    cols_to_split = []
    stokes_params = {}
    for i, key in enumerate(specdata.keys()):
        if specdata.keys()[1] in key:
            cols_to_split.append(i)

    stokes_params['I'] = pd.concat([specdata['nu'], specdata.iloc[:, cols_to_split[0]: cols_to_split[1]]], axis=1)
    stokes_params['Q'] = pd.concat([specdata['nu'], specdata.iloc[:, cols_to_split[1]: cols_to_split[2]]], axis=1)
    stokes_params['U'] = pd.concat([specdata['nu'], specdata.iloc[:, cols_to_split[2]:]], axis=1)

    for param in ['Q', 'U']:
        stokes_params[param].columns = stokes_params['I'].keys()
        stokes_params[param + '/I'] = pd.concat([specdata['nu'],
                                                 stokes_params[param].iloc[:, 1:]
                                                 / stokes_params['I'].iloc[:, 1:]], axis=1)

    return stokes_params


def get_vspecpol_spectrum(modelpath, timeavg, angle, args, fnufilterfunc=None):
    stokes_params = get_specpol_data(angle, modelpath=modelpath)
    if 'stokesparam' not in args:
        args.stokesparam = 'I'
    vspecdata = stokes_params[args.stokesparam]

    nu = vspecdata.loc[:, 'nu'].values

    arr_tmid = [float(i) for i in vspecdata.columns.values[1:] if i[-2] != '.']
    arr_tdelta = [l1 - l2 for l1, l2 in zip(arr_tmid[1:], arr_tmid[:-1])] + [arr_tmid[-1] - arr_tmid[-2]]

    def match_closest_time(reftime):
        return str("{}".format(min([float(x) for x in arr_tmid], key=lambda x: abs(x - reftime))))

    # if 'timemin' and 'timemax' in args:
    #     timelower = match_closest_time(args.timemin)  # how timemin, timemax are used changed at some point
    #     timeupper = match_closest_time(args.timemax)  # to average over multiple timesteps needs to fix this
    # else:
    timelower = match_closest_time(timeavg)
    timeupper = match_closest_time(timeavg)
    timestepmin = vspecdata.columns.get_loc(timelower)
    timestepmax = vspecdata.columns.get_loc(timeupper)

    f_nu = stackspectra([
        (vspecdata[vspecdata.columns[timestep + 1]],
         arr_tdelta[timestep])
        for timestep in range(timestepmin-1, timestepmax)])

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        print("Applying filter to ARTIS spectrum")
        f_nu = fnufilterfunc(f_nu)

    dfspectrum = pd.DataFrame({'nu': nu, 'f_nu': f_nu})
    dfspectrum.sort_values(by='nu', ascending=False, inplace=True)

    dfspectrum.eval('lambda_angstroms = @c / nu', local_dict={'c': const.c.to('angstrom/s').value}, inplace=True)
    dfspectrum.eval('f_lambda = f_nu * nu / lambda_angstroms', inplace=True)

    return dfspectrum


@lru_cache(maxsize=4)
def get_flux_contributions(
        modelpath, filterfunc=None, timestepmin=0, timestepmax=None, getemission=True, getabsorption=True,
        use_lastemissiontype=False):

    arr_tmid = at.get_timestep_times_float(modelpath, loc='mid')
    arr_tdelta = at.get_timestep_times_float(modelpath, loc='delta')
    arraynu = at.get_nu_grid(modelpath)
    arraylambda = const.c.to('angstrom/s').value / arraynu
    if not Path(modelpath, 'compositiondata.txt').is_file():
        print('WARNING: compositiondata.txt not found. Using output*.txt instead')
        elementlist = at.get_composition_data_from_outputfile(modelpath)
    else:
        elementlist = at.get_composition_data(modelpath)
    nelements = len(elementlist)

    if getemission:
        if use_lastemissiontype:
            emissionfilenames = ['emission.out.xz', 'emission.out.gz', 'emission.out', 'emissionpol.out']
        else:
            emissionfilenames = ['emissiontrue.out.xz', 'emissiontrue.out.gz', 'emissiontrue.out']

        emissionfilename = at.firstexisting(emissionfilenames, path=modelpath)

        if 'pol' in str(emissionfilename):
            print("This artis run contains polarisation data")
            # File contains I, Q and U and so times are repeated 3 times
            arr_tmid = np.array(arr_tmid.tolist() * 3)

        try:
            emissionfilesize = Path(emissionfilename).stat().st_siplze / 1024 / 1024
            print(f' Reading {emissionfilename} ({emissionfilesize:.2f} MiB)')
        except AttributeError:
            print(f' Reading {emissionfilename}')
        emissiondata = pd.read_csv(emissionfilename, delim_whitespace=True, header=None)
        maxion_float = (emissiondata.shape[1] - 1) / 2 / nelements  # also known as MIONS in ARTIS sn3d.h
        assert maxion_float.is_integer()
        maxion = int(maxion_float)
        print(f' inferred MAXION = {maxion} from emission file using nlements = {nelements} from compositiondata.txt')

        # check that the row count is product of timesteps and frequency bins found in spec.out
        assert emissiondata.shape[0] == len(arraynu) * len(arr_tmid)

    if getabsorption:
        absorptionfilename = at.firstexisting(['absorption.out.xz', 'absorption.out.gz', 'absorption.out',
                                               'absorptionpol.out'], path=modelpath)
        try:
            absorptionfilesize = Path(absorptionfilename).stat().st_size / 1024 / 1024
            print(f' Reading {absorptionfilename} ({absorptionfilesize:.2f} MiB)')
        except AttributeError:
            print(f' Reading {emissionfilename}')
        absorptiondata = pd.read_csv(absorptionfilename, delim_whitespace=True, header=None)
        absorption_maxion_float = absorptiondata.shape[1] / nelements
        assert absorption_maxion_float.is_integer()
        absorption_maxion = int(absorption_maxion_float)
        if not getemission:
            maxion = absorption_maxion
            print(f' inferred MAXION = {maxion} from absorption file using nlements = {nelements}'
                  'from compositiondata.txt')
        else:
            assert absorption_maxion == maxion
        assert absorptiondata.shape[0] == len(arraynu) * len(arr_tmid)
    else:
        absorptiondata = None

    array_flambda_emission_total = np.zeros_like(arraylambda, dtype=float)
    contribution_list = []
    if filterfunc:
        print("Applying filter to ARTIS spectrum")

    for element in range(nelements):
        nions = elementlist.nions[element]
        # nions = elementlist.iloc[element].uppermost_ionstage - elementlist.iloc[element].lowermost_ionstage + 1
        for ion in range(nions):
            ion_stage = ion + elementlist.lowermost_ionstage[element]
            ionserieslist = [(element * maxion + ion, 'bound-bound'),
                             (nelements * maxion + element * maxion + ion, 'bound-free')]

            if element == ion == 0:
                ionserieslist.append((2 * nelements * maxion, 'free-free'))

            for (selectedcolumn, emissiontype) in ionserieslist:
                # if linelabel.startswith('Fe ') or linelabel.endswith("-free"):
                #     continue
                if getemission:
                    array_fnu_emission = stackspectra(
                        [(emissiondata.iloc[timestep::len(arr_tmid), selectedcolumn].values,
                          arr_tdelta[timestep])
                         for timestep in range(timestepmin, timestepmax + 1)])
                else:
                    array_fnu_emission = np.zeros_like(arraylambda, dtype=float)

                if absorptiondata is not None and selectedcolumn < nelements * maxion:  # bound-bound process
                    array_fnu_absorption = stackspectra(
                        [(absorptiondata.iloc[timestep::len(arr_tmid), selectedcolumn].values,
                          arr_tdelta[timestep])
                         for timestep in range(timestepmin, timestepmax + 1)])
                else:
                    array_fnu_absorption = np.zeros_like(arraylambda, dtype=float)

                # best to use the filter on fnu (because it hopefully has regular sampling)
                if filterfunc:
                    array_fnu_emission = filterfunc(array_fnu_emission)
                    if selectedcolumn <= nelements * maxion:
                        array_fnu_absorption = filterfunc(array_fnu_absorption)

                array_flambda_emission = array_fnu_emission * arraynu / arraylambda
                array_flambda_absorption = array_fnu_absorption * arraynu / arraylambda

                array_flambda_emission_total += array_flambda_emission
                fluxcontribthisseries = (
                    abs(np.trapz(array_fnu_emission, x=arraynu)) + abs(np.trapz(array_fnu_absorption, x=arraynu)))

                if emissiontype == 'bound-bound':
                    linelabel = at.get_ionstring(elementlist.Z[element], ion_stage)
                elif emissiontype != 'free-free':
                    linelabel = f'{at.get_ionstring(elementlist.Z[element], ion_stage)} {emissiontype}'
                else:
                    linelabel = f'{emissiontype}'

                contribution_list.append(
                    fluxcontributiontuple(fluxcontrib=fluxcontribthisseries, linelabel=linelabel,
                                          array_flambda_emission=array_flambda_emission,
                                          array_flambda_absorption=array_flambda_absorption,
                                          color=None))

    return contribution_list, array_flambda_emission_total


@lru_cache(maxsize=4)
def get_flux_contributions_from_packets(
        modelpath, timelowerdays, timeupperdays, lambda_min, lambda_max, delta_lambda=None,
        getemission=True, getabsorption=True, maxpacketfiles=None, filterfunc=None, groupby='ion', modelgridindex=None,
        use_comovingframe=False, use_lastemissiontype=False, useinternalpackets=False, emissionvelocitycut=None):

    assert groupby in [None, 'ion', 'line', 'upperterm', 'terms']

    if groupby in ['terms', 'upperterm']:
        adata = at.atomic.get_levels(modelpath)

    def get_emprocesslabel(emtype):
        if emtype >= 0:
            line = linelist[emtype]
            if groupby == 'line':
                # if line.atomic_number != 26 or line.ionstage != 2:
                #     return 'non-Fe II ions'
                return (f'{at.get_ionstring(line.atomic_number, line.ionstage)} '
                        f'λ{line.lambda_angstroms:.0f} '
                        f'({line.upperlevelindex}-{line.lowerlevelindex})')
            elif groupby == 'terms':
                upper_config = adata.query(
                    'Z == @line.atomic_number and ion_stage == @line.ionstage', inplace=False
                    ).iloc[0].levels.iloc[line.upperlevelindex].levelname
                upper_term_noj = upper_config.split('_')[-1].split('[')[0]
                lower_config = adata.query(
                    'Z == @line.atomic_number and ion_stage == @line.ionstage', inplace=False
                    ).iloc[0].levels.iloc[line.lowerlevelindex].levelname
                lower_term_noj = lower_config.split('_')[-1].split('[')[0]
                return f'{at.get_ionstring(line.atomic_number, line.ionstage)} {upper_term_noj}->{lower_term_noj}'
            elif groupby == 'upperterm':
                upper_config = adata.query(
                    'Z == @line.atomic_number and ion_stage == @line.ionstage', inplace=False
                    ).iloc[0].levels.iloc[line.upperlevelindex].levelname
                upper_term_noj = upper_config.split('_')[-1].split('[')[0]
                return f'{at.get_ionstring(line.atomic_number, line.ionstage)} {upper_term_noj}'
            return f'{at.get_ionstring(line.atomic_number, line.ionstage)} bound-bound'
        elif emtype == -9999999:
            return 'free-free'

        bflist = at.get_bflist(modelpath)
        bfindex = -emtype - 1
        if bfindex in bflist:
            (atomic_number, ionstage, level) = bflist[bfindex][:3]
            if groupby == 'line':
                return f'{at.get_ionstring(atomic_number, ionstage)} bound-free {level}'
            return f'{at.get_ionstring(atomic_number, ionstage)} bound-free'
        return f'? bound-free (bfindex={bfindex})'

    def get_absprocesslabel(abstype):
        if abstype >= 0:
            line = linelist[abstype]
            if groupby == 'line':
                return (f'{at.get_ionstring(line.atomic_number, line.ionstage)} '
                        f'λ{line.lambda_angstroms:.0f} '
                        f'({line.upperlevelindex}-{line.lowerlevelindex})')
            return f'{at.get_ionstring(line.atomic_number, line.ionstage)} bound-bound'
        if abstype == -1:
            return 'free-free'
        if abstype == -2:
            return 'bound-free'
        return '? other absorp.'

    if delta_lambda:
        array_lambdabinedges = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
        array_lambda = 0.5 * (array_lambdabinedges[:-1] + array_lambdabinedges[1:])  # bin centres
    else:
        array_lambdabinedges, array_lambda, delta_lambda = get_exspec_bins()

    if use_comovingframe:
        modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
        vmax = modeldata.iloc[-1].velocity_outer * u.km / u.s
        betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    import artistools.packets
    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles)

    linelist = at.get_linelist(modelpath)

    energysum_spectrum_emission_total = np.zeros_like(array_lambda, dtype=float)
    array_energysum_spectra = {}

    timelow = timelowerdays * u.day.to('s')
    timehigh = timeupperdays * u.day.to('s')

    nprocs_read = len(packetsfiles)
    c_cgs = const.c.to('cm/s').value
    c_ang_s = const.c.to('angstrom/s').value
    nu_min = c_ang_s / lambda_max
    nu_max = c_ang_s / lambda_min

    if useinternalpackets:
        emtypecolumn = 'emissiontype'
    else:
        emtypecolumn = 'emissiontype' if use_lastemissiontype else 'trueemissiontype'

    for index, packetsfile in enumerate(packetsfiles):
        if useinternalpackets:
            # if we're using packets*.out files, these packets are from the last timestep
            t_seconds = at.get_timestep_times_float(modelpath, loc='start')[-1] * u.day.to('s')

            if modelgridindex is not None:
                v_inner = at.inputmodel.get_modeldata(modelpath)[0]['velocity_inner'].iloc[modelgridindex] * 1e5
                v_outer = at.inputmodel.get_modeldata(modelpath)[0]['velocity_outer'].iloc[modelgridindex] * 1e5
            else:
                v_inner = 0.
                v_outer = at.inputmodel.get_modeldata(modelpath)[0]['velocity_outer'].iloc[-1] * 1e5

            r_inner = t_seconds * v_inner
            r_outer = t_seconds * v_outer

            dfpackets = at.packets.readfile(packetsfile, type='TYPE_RPKT')
            print("Using non-escaped internal r-packets")
            dfpackets.query(f'type_id == {at.packets.type_ids["TYPE_RPKT"]} and @nu_min <= nu_rf < @nu_max',
                            inplace=True)
            if modelgridindex is not None:
                assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
                # dfpackets.eval(f'velocity = sqrt(posx ** 2 + posy ** 2 + posz ** 2) / @t_seconds', inplace=True)
                # dfpackets.query(f'@v_inner <= velocity <= @v_outer',
                #                 inplace=True)
                dfpackets.query('where in @assoc_cells[@modelgridindex]', inplace=True)
            print(f"  {len(dfpackets)} internal r-packets matching frequency range")
        else:
            dfpackets = at.packets.readfile(packetsfile, type='TYPE_ESCAPE', escape_type='TYPE_RPKT')
            dfpackets.query(
                '@nu_min <= nu_rf < @nu_max and ' +
                ('@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh'
                 if not use_comovingframe else
                 '@timelow < escape_time * @betafactor < @timehigh'),
                inplace=True)
            print(f"  {len(dfpackets)} escaped r-packets matching frequency and arrival time ranges")

            if emissionvelocitycut:
                dfpackets = at.packets.add_derived_columns(
                    dfpackets, modelpath, ['emission_velocity'])

                dfpackets.query('(emission_velocity / 1e5) > @emissionvelocitycut', inplace=True)

        if np.isscalar(delta_lambda):
            dfpackets.eval('xindex = floor((@c_ang_s / nu_rf - @lambda_min) / @delta_lambda)', inplace=True)
            if getabsorption:
                dfpackets.eval('xindexabsorbed = floor((@c_ang_s / absorption_freq - @lambda_min) / @delta_lambda)',
                               inplace=True)
        else:
            dfpackets['xindex'] = np.digitize(c_ang_s / dfpackets.nu_rf, bins=array_lambdabinedges, right=True) - 1
            if getabsorption:
                dfpackets['xindexabsorbed'] = np.digitize(
                    c_ang_s / dfpackets.absorption_freq, bins=array_lambdabinedges, right=True) - 1

        for _, packet in dfpackets.iterrows():
            lambda_rf = c_ang_s / packet.nu_rf
            xindex = int(packet.xindex)
            assert xindex >= 0

            pkt_en = packet.e_cmf / betafactor if use_comovingframe else packet.e_rf

            energysum_spectrum_emission_total[xindex] += pkt_en

            if getemission:
                # if emtype >= 0 and linelist[emtype].upperlevelindex <= 80:
                #     continue
                # emprocesskey = get_emprocesslabel(packet.emissiontype)
                emprocesskey = get_emprocesslabel(packet[emtypecolumn])
                # print('packet lambda_cmf: {c_ang_s / packet.nu_cmf}.1f}, lambda_rf {lambda_rf:.1f}, {emprocesskey}')

                if emprocesskey not in array_energysum_spectra:
                    array_energysum_spectra[emprocesskey] = (
                        np.zeros_like(array_lambda, dtype=float), np.zeros_like(array_lambda, dtype=float))

                array_energysum_spectra[emprocesskey][0][xindex] += pkt_en

            if getabsorption:
                abstype = packet.absorption_type
                if abstype > 0:
                    absprocesskey = get_absprocesslabel(abstype)

                    xindexabsorbed = int(packet.xindexabsorbed)  # bin by absorption wavelength
                    # xindexabsorbed = xindex  # bin by final escaped wavelength

                    if absprocesskey not in array_energysum_spectra:
                        array_energysum_spectra[absprocesskey] = (
                            np.zeros_like(array_lambda, dtype=float), np.zeros_like(array_lambda, dtype=float))

                    array_energysum_spectra[absprocesskey][1][xindexabsorbed] += pkt_en

    if useinternalpackets:
        volume = 4 / 3. * math.pi * (r_outer ** 3 - r_inner ** 3)
        if modelgridindex:
            volume_shells = volume
            assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
            volume = (at.get_wid_init_at_tmin(modelpath) * t_seconds / (
                at.get_inputparams(modelpath)['tmin'] * u.day.to('s'))) ** 3 * len(assoc_cells[modelgridindex])
            print('volume', volume, 'shell volume', volume_shells, '-------------------------------------------------')
        normfactor = c_cgs / 4 / math.pi / delta_lambda / volume / nprocs_read
    else:
        normfactor = (1. / delta_lambda / (timehigh - timelow) / 4 / math.pi
                      / (u.megaparsec.to('cm') ** 2) / nprocs_read)

    array_flambda_emission_total = energysum_spectrum_emission_total * normfactor

    contribution_list = []
    for (groupname,
         (energysum_spec_emission, energysum_spec_absorption)) in array_energysum_spectra.items():
        array_flambda_emission = energysum_spec_emission * normfactor

        array_flambda_absorption = energysum_spec_absorption * normfactor

        fluxcontribthisseries = (
            abs(np.trapz(array_flambda_emission, x=array_lambda)) +
            abs(np.trapz(array_flambda_absorption, x=array_lambda)))

        linelabel = groupname.replace(' bound-bound', '')

        contribution_list.append(
            fluxcontributiontuple(fluxcontrib=fluxcontribthisseries, linelabel=linelabel,
                                  array_flambda_emission=array_flambda_emission,
                                  array_flambda_absorption=array_flambda_absorption,
                                  color=None))

    return contribution_list, array_flambda_emission_total, array_lambda


def sort_and_reduce_flux_contribution_list(
        contribution_list_in, maxseriescount, arraylambda_angstroms,
        fixedionlist=None, hideother=False, greyscale=False):

    if fixedionlist:
        # sort in manual order
        def sortkey(x):
            return (fixedionlist.index(x.linelabel) if x.linelabel in fixedionlist
                    else len(fixedionlist) + 1, -x.fluxcontrib)
    else:
        # sort descending by flux contribution
        def sortkey(x): return -x.fluxcontrib

    contribution_list = sorted(contribution_list_in, key=sortkey)

    # combine the items past maxseriescount or not in manual list into a single item
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms, dtype=float)
    remainder_fluxcontrib = 0

    if greyscale:
        hatches = artistools.spectra.plot.hatches
        seriescount = len(fixedionlist) if fixedionlist else maxseriescount
        colorcount = math.ceil(seriescount / 1. / len(hatches))
        greylist = [str(x) for x in np.linspace(0.4, 0.9, colorcount, endpoint=True)]
        color_list = []
        for c in range(colorcount):
            for h in hatches:
                color_list.append(greylist[c])
        # color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))
        mpl.rcParams['hatch.linewidth'] = 0.1
        # TODO: remove???
        color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))
    else:
        color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))

    contribution_list_out = []
    numotherprinted = 0
    maxnumotherprinted = 20
    entered_other = False
    plotted_ion_list = []
    for index, row in enumerate(contribution_list):
        if fixedionlist and row.linelabel in fixedionlist:
            contribution_list_out.append(row._replace(color=color_list[fixedionlist.index(row.linelabel)]))
        elif not fixedionlist and index < maxseriescount:
            contribution_list_out.append(row._replace(color=color_list[index]))
            plotted_ion_list.append(row.linelabel)
        else:
            remainder_fluxcontrib += row.fluxcontrib
            remainder_flambda_emission += row.array_flambda_emission
            remainder_flambda_absorption += row.array_flambda_absorption
            if not entered_other:
                print(f"  Other (top {maxnumotherprinted}):")
                entered_other = True

        if numotherprinted < maxnumotherprinted:
            integemiss = abs(np.trapz(row.array_flambda_emission, x=arraylambda_angstroms))
            integabsorp = abs(np.trapz(-row.array_flambda_absorption, x=arraylambda_angstroms))
            if integabsorp > 0. and integemiss > 0.:
                print(f'{row.fluxcontrib:.1e}, emission {integemiss:.1e}, '
                      f"absorption {integabsorp:.1e} [erg/s/cm^2]: '{row.linelabel}'")
            elif integemiss > 0.:
                print(f"  emission {integemiss:.1e} [erg/s/cm^2]: '{row.linelabel}'")
            else:
                print(f"absorption {integabsorp:.1e} [erg/s/cm^2]: '{row.linelabel}'")

            if entered_other:
                numotherprinted += 1

    if not fixedionlist:
        cmdarg = "'" + "' '".join(plotted_ion_list) + "'"
        print('To reuse this ion/process contribution list, pass the following command-line argument: ')
        print(f'     -fixedionlist {cmdarg}')

    if remainder_fluxcontrib > 0. and not hideother:
        contribution_list_out.append(fluxcontributiontuple(
            fluxcontrib=remainder_fluxcontrib, linelabel='Other',
            array_flambda_emission=remainder_flambda_emission, array_flambda_absorption=remainder_flambda_absorption,
            color='grey'))

    return contribution_list_out


def print_integrated_flux(arr_f_lambda, arr_lambda_angstroms, distance_megaparsec=1.):
    integrated_flux = abs(np.trapz(arr_f_lambda, x=arr_lambda_angstroms)) * u.erg / u.s / (u.cm ** 2)
    print(f' integrated flux ({arr_lambda_angstroms.min():.1f} to '
          f'{arr_lambda_angstroms.max():.1f} A): {integrated_flux:.3e}')
    # luminosity = integrated_flux * 4 * math.pi * (distance_megaparsec * u.megaparsec ** 2)
    # print(f'(L={luminosity.to("Lsun"):.3e})')


def get_line_flux(lambda_low, lambda_high, arr_f_lambda, arr_lambda_angstroms):
    index_low, index_high = [np.searchsorted(arr_lambda_angstroms, wl, side="left") for wl in (lambda_low, lambda_high)]
    flux_integral = abs(np.trapz(arr_f_lambda[index_low:index_high], x=arr_lambda_angstroms[index_low:index_high]))
    return flux_integral


def print_floers_line_ratio(modelpath, timedays, arr_f_lambda, arr_lambda_angstroms):
    f_12570 = get_line_flux(12570 - 200, 12570 + 200, arr_f_lambda, arr_lambda_angstroms)
    f_7155 = get_line_flux(7000, 7350, arr_f_lambda, arr_lambda_angstroms)
    print(f'f_12570 {f_12570:.2e} f_7155 {f_7155:.2e}')
    if f_7155 > 0 and f_12570 > 0:
        fratio = f_12570 / f_7155
        print(f'f_12570/f_7122 = {fratio:.2e} (log10 is {math.log10(fratio):.2e})')
        outfilename = f"fe2_nir_vis_ratio_{os.path.basename(modelpath)}.txt"
        print(f' saved to {outfilename}')
        with open(outfilename, "a+") as f:
            f.write(f'{timedays:.1f} {fratio:.3e}\n')


def get_reference_spectrum(filename):

    if Path(filename).is_file():
        filepath = Path(filename)
    else:
        filepath = Path(at.PYDIR, 'data', 'refspectra', filename)

    metadata = at.get_file_metadata(filepath)

    flambdaindex = metadata.get('f_lambda_columnindex', 1)

    specdata = pd.read_csv(filepath, delim_whitespace=True, header=None, comment='#',
                           names=['lambda_angstroms', 'f_lambda'], usecols=[0, flambdaindex])

    # new_lambda_angstroms = []
    # binned_flux = []
    #
    # wavelengths = specdata['lambda_angstroms']
    # fluxes = specdata['f_lambda']
    # nbins = 10
    #
    # for i in np.arange(start=0, stop=len(wavelengths) - nbins, step=nbins):
    #     new_lambda_angstroms.append(wavelengths[i + int(nbins / 2)])
    #     sum_flux = 0
    #     for j in range(i, i + nbins):
    #
    #         if not math.isnan(fluxes[j]):
    #             print(fluxes[j])
    #             sum_flux += fluxes[j]
    #     binned_flux.append(sum_flux / nbins)
    #
    # filtered_specdata = pd.DataFrame(new_lambda_angstroms, columns=['lambda_angstroms'])
    # filtered_specdata['f_lamba'] = binned_flux
    # print(filtered_specdata)
    # plt.plot(specdata['lambda_angstroms'], specdata['f_lambda'])
    # plt.plot(new_lambda_angstroms, binned_flux)
    #
    # filtered_specdata.to_csv('/Users/ccollins/artis_nebular/artistools/artistools/data/refspectra/' + name,
    #                          index=False, header=False, sep=' ')

    if 'a_v' in metadata or 'e_bminusv' in metadata:
        print('Correcting for reddening')
        from extinction import apply, ccm89

        specdata['f_lambda'] = apply(
            ccm89(specdata['lambda_angstroms'].values, a_v=-metadata['a_v'], r_v=metadata['r_v'], unit='aa'),
            specdata['f_lambda'].values)

    if 'z' in metadata:
        print('Correcting for redshift')
        specdata['lambda_angstroms'] /= (1 + metadata['z'])

    return specdata, metadata


def write_flambda_spectra(modelpath, args):
    """Write out spectra to text files.

    Writes lambda_angstroms and f_lambda to .txt files for all timesteps and create
    a text file containing the time in days for each timestep.
    """
    outdirectory = Path(modelpath, 'spectra')

    outdirectory.mkdir(parents=True, exist_ok=True)

    if Path(modelpath, 'specpol.out').is_file():
        specfilename = modelpath / 'specpol.out'
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        specfilename = at.firstexisting(['spec.out.xz', 'spec.out.gz', 'spec.out'], path=modelpath)
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = specdata.columns.values[1:]

    number_of_timesteps = len(timearray)

    if not args.timestep:
        args.timestep = f'0-{number_of_timesteps - 1}'

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays)

    with open(outdirectory / 'spectra_list.txt', 'w+') as spectra_list:

        arr_tmid = at.get_timestep_times_float(modelpath, loc='mid')

        for timestep in range(timestepmin, timestepmax + 1):

            dfspectrum = get_spectrum(modelpath, timestep, timestep)
            tmid = arr_tmid[timestep]

            outfilepath = outdirectory / f'spectrum_ts{timestep:02.0f}_{tmid:.0f}d.txt'

            with open(outfilepath, 'w') as spec_file:

                spec_file.write('#lambda f_lambda_1Mpc\n')
                spec_file.write('#[A] [erg/s/cm2/A]\n')

                dfspectrum.to_csv(spec_file, header=False, sep=' ', index=False,
                                  columns=["lambda_angstroms", "f_lambda"])

            spectra_list.write(str(outfilepath.absolute()) + '\n')

    with open(outdirectory / 'time_list.txt', 'w+') as time_list:
        for time in arr_tmid:
            time_list.write(f'{str(time)} \n')

    print(f'Saved in {outdirectory}')
