#!/usr/bin/env python3
"""Artistools - spectra related functions."""
import argparse
import math
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy import constants as const
from astropy import units as u

import artistools as at

fluxcontributiontuple = namedtuple(
    'fluxcontribution', 'fluxcontrib linelabel array_flambda_emission array_flambda_absorption color')

color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))


def stackspectra(spectra_and_factors):
    factor_sum = sum([factor for _, factor in spectra_and_factors])

    stackedspectrum = np.zeros_like(spectra_and_factors[0][0], dtype=np.float)
    for spectrum, factor in spectra_and_factors:
        stackedspectrum += spectrum * factor / factor_sum

    return stackedspectrum


def get_spectrum(modelpath, timestepmin: int, timestepmax=-1, fnufilterfunc=None):
    """Return a pandas DataFrame containing an ARTIS emergent spectrum."""
    if timestepmax < 0:
        timestepmax = timestepmin

    master_branch = False
    if Path(modelpath, 'specpol.out').is_file():
        specfilename = Path(modelpath) / "specpol.out"
        master_branch = True
    elif Path(modelpath).is_dir():
        specfilename = at.firstexisting(['spec.out.gz', 'spec.out'], path=modelpath)
    else:
        specfilename = modelpath

    specdata = pd.read_csv(specfilename, delim_whitespace=True)

    nu = specdata.loc[:, '0'].values
    if master_branch:
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
    else:
        timearray = specdata.columns.values[1:]

    f_nu = stackspectra([
        (specdata[specdata.columns[timestep + 1]], at.get_timestep_time_delta(timestep, timearray))
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


def get_spectrum_from_packets(
        modelpath, timelowdays, timehighdays, lambda_min, lambda_max,
        delta_lambda=30, use_comovingframe=None, maxpacketfiles=-1):
    """Get a spectrum dataframe using the packets files as input."""
    import artistools.packets
    packetsfiles = at.packets.get_packetsfiles(modelpath, maxpacketfiles)
    if use_comovingframe:
        modeldata, _ = at.get_modeldata(Path(packetsfiles[0]).parent)
        vmax = modeldata.iloc[-1].velocity * u.km / u.s
        betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    def update_min_sum_max(array, xindex, e_rf, value):
        if value < array[0][xindex] or array[0][xindex] == 0:
            array[0][xindex] = value

        array[1][xindex] += e_rf * value

        if value > array[2][xindex] or array[2][xindex] == 0:
            array[2][xindex] = value

    # linelist = at.get_linelist(modelpath)
    array_lambda = np.arange(lambda_min, lambda_max, delta_lambda)
    array_energysum = np.zeros_like(array_lambda, dtype=np.float)  # total packet energy sum of each bin
    array_energysum_positron = np.zeros_like(array_lambda, dtype=np.float)  # total packet energy sum of each bin
    array_pktcount = np.zeros_like(array_lambda, dtype=np.int)  # number of packets in each bin
    array_emvelocity = np.zeros((3, len(array_lambda)), dtype=np.float)
    array_trueemvelocity = np.zeros((3, len(array_lambda)), dtype=np.float)

    timelow = timelowdays * u.day.to('s')
    timehigh = timehighdays * u.day.to('s')

    nprocs_read = len(packetsfiles)
    c_cgs = const.c.to('cm/s').value
    c_ang_s = const.c.to('angstrom/s').value
    nu_min = c_ang_s / lambda_max
    nu_max = c_ang_s / lambda_min
    for index, packetsfile in enumerate(packetsfiles):
        dfpackets = at.packets.readfile(
            packetsfile,
            usecols=[
                'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
                'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz',
                'em_posx', 'em_posy', 'em_posz', 'em_time',
                'true_emission_velocity', 'originated_from_positron', 'true_emission_type'])

        querystr = 'type == "TYPE_ESCAPE" and escape_type == "TYPE_RPKT" and @nu_min <= nu_rf < @nu_max and'
        if not use_comovingframe:
            querystr += '@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh'
        else:
            querystr += '@timelow < escape_time * @betafactor < @timehigh'

        dfpackets.query(querystr, inplace=True)

        print(f"  {len(dfpackets)} escaped r-packets with matching nu and arrival time")
        for _, packet in dfpackets.iterrows():
            # transition = linelist[packet.true_emission_type]
            # if transition.atomic_number != 26 or transition.ionstage != 2:
            #     continue
            # if transition.upperlevelindex <= 80:
            #     continue

            lambda_rf = c_ang_s / packet.nu_rf
            # pos_dot_dir = packet.posx * packet.dirx + packet.posy * packet.diry + packet.posz * packet.dirz
            # t_arrive = packet['escape_time'] - (pos_dot_dir / c_cgs)
            # print(f"Packet escaped at {t_arrive / u.day.to('s'):.1f} days with "
            #       f"nu={packet.nu_rf:.2e}, lambda={lambda_rf:.1f}")
            xindex = math.floor((lambda_rf - lambda_min) / delta_lambda)
            assert xindex >= 0

            pkt_en = packet.e_cmf / betafactor if use_comovingframe else packet.e_rf

            array_energysum[xindex] += pkt_en
            if packet.originated_from_positron:
                array_energysum_positron[xindex] += pkt_en
            array_pktcount[xindex] += 1

            # convert cm/s to km/s
            emission_velocity = (
                math.sqrt(packet.em_posx ** 2 + packet.em_posy ** 2 + packet.em_posz ** 2) / packet.em_time) / 1e5
            update_min_sum_max(array_emvelocity, xindex, pkt_en, emission_velocity)

            true_emission_velocity = packet.true_emission_velocity / 1e5
            update_min_sum_max(array_trueemvelocity, xindex, pkt_en, true_emission_velocity)

    array_flambda = (array_energysum / delta_lambda / (timehigh - timelow) /
                     4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs_read)

    array_flambda_positron = (array_energysum_positron / delta_lambda / (timehigh - timelow) /
                              4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs_read)

    with np.errstate(divide='ignore', invalid='ignore'):
        array_emvelocity[1] = np.divide(array_emvelocity[1], array_energysum)
        array_trueemvelocity[1] = np.divide(array_trueemvelocity[1], array_energysum)

    dfspectrum = pd.DataFrame({
        'lambda_angstroms': array_lambda,
        'f_lambda': array_flambda,
        'f_lambda_originated_from_positron': array_flambda_positron,
        'packetcount': array_pktcount,
        'energy_sum': array_energysum,
        'emission_velocity_min': array_emvelocity[0],
        'emission_velocity_avg': array_emvelocity[1],
        'emission_velocity_max': array_emvelocity[2],
        'trueemission_velocity_min': array_trueemvelocity[0],
        'trueemission_velocity_avg': array_trueemvelocity[1],
        'trueemission_velocity_max': array_trueemvelocity[2],
    })

    return dfspectrum


@lru_cache(maxsize=4)
def get_flux_contributions(
        modelpath, filterfunc=None, timestepmin=0, timestepmax=None, getabsorption=True):
    # emissionfilenames = ['emissiontrue.out.gz', 'emissiontrue.out', 'emission.out.gz', 'emission.out']
    emissionfilenames = ['emissiontrue.out.gz', 'emissiontrue.out']
    emissionfilename = at.firstexisting(emissionfilenames, path=modelpath)
    absorptionfilename = (at.firstexisting(['absorption.out.gz', 'absorption.out'], path=modelpath)
                          if getabsorption else None)

    timearray = at.get_timestep_times(modelpath)
    arraynu = at.get_nu_grid(modelpath)
    arraylambda = const.c.to('angstrom/s').value / arraynu
    elementlist = at.get_composition_data(modelpath)
    nelements = len(elementlist)

    emissionfilesize = Path(emissionfilename).stat().st_size / 1024 / 1024
    print(f' Reading {emissionfilename} ({emissionfilesize:.2f} MiB)')
    emissiondata = pd.read_csv(emissionfilename, delim_whitespace=True, header=None)
    maxion_float = (emissiondata.shape[1] - 1) / 2 / nelements  # also known as MIONS in ARTIS sn3d.h
    assert maxion_float.is_integer()
    maxion = int(maxion_float)
    print(f' inferred MAXION = {maxion} from emission file using nlements = {nelements} from compositiondata.txt')

    # check that the row count is product of timesteps and frequency bins found in spec.out
    assert emissiondata.shape[0] == len(arraynu) * len(timearray)

    if absorptionfilename:
        absorptionfilesize = Path(absorptionfilename).stat().st_size / 1024 / 1024
        print(f' Reading {absorptionfilename} ({absorptionfilesize:.2f} MiB)')
        absorptiondata = pd.read_csv(absorptionfilename, delim_whitespace=True, header=None)
        absorption_maxion_float = absorptiondata.shape[1] / nelements
        assert absorption_maxion_float.is_integer()
        absorption_maxion = int(absorption_maxion_float)
        assert absorption_maxion == maxion
        assert absorptiondata.shape[0] == len(arraynu) * len(timearray)
    else:
        absorptiondata = None

    array_flambda_emission_total = np.zeros_like(arraylambda)
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
                array_fnu_emission = stackspectra(
                    [(emissiondata.iloc[timestep::len(timearray), selectedcolumn].values,
                      at.get_timestep_time_delta(timestep, timearray))
                     for timestep in range(timestepmin, timestepmax + 1)])

                if absorptiondata is not None and selectedcolumn < nelements * maxion:  # bound-bound process
                    array_fnu_absorption = stackspectra(
                        [(absorptiondata.iloc[timestep::len(timearray), selectedcolumn].values,
                          at.get_timestep_time_delta(timestep, timearray))
                         for timestep in range(timestepmin, timestepmax + 1)])
                else:
                    array_fnu_absorption = np.zeros_like(array_fnu_emission)

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
        modelpath, timelowdays, timehighdays, lambda_min, lambda_max, delta_lambda=30,
        getabsorption=True, maxpacketfiles=-1, filterfunc=None):
    array_lambda = np.arange(lambda_min, lambda_max, delta_lambda)

    import artistools.packets
    packetsfiles = at.packets.get_packetsfiles(modelpath, maxpacketfiles)

    linelist = at.get_linelist(modelpath)

    energysum_spectrum_emission_total = np.zeros_like(array_lambda, dtype=np.float)
    array_energysum_spectra = {}

    timelow = timelowdays * u.day.to('s')
    timehigh = timehighdays * u.day.to('s')

    nprocs_read = len(packetsfiles)
    c_cgs = const.c.to('cm/s').value
    c_ang_s = const.c.to('angstrom/s').value
    nu_min = c_ang_s / lambda_max
    nu_max = c_ang_s / lambda_min
    for index, packetsfile in enumerate(packetsfiles):
        dfpackets = at.packets.readfile(
            packetsfile,
            usecols=[
                'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
                'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz',
                # 'em_posx', 'em_posy', 'em_posz', 'em_time',
                'true_emission_velocity',
                # 'originated_from_positron',
                'true_emission_type',
                'absorption_type',
            ])

        querystr = 'type == "TYPE_ESCAPE" and escape_type == "TYPE_RPKT" and @nu_min <= nu_rf < @nu_max and'
        querystr += '@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh'

        dfpackets.query(querystr, inplace=True)

        print(f"  {len(dfpackets)} escaped r-packets with matching nu and arrival time")
        for _, packet in dfpackets.iterrows():
            lambda_rf = c_ang_s / packet.nu_rf

            xindex = math.floor((lambda_rf - lambda_min) / delta_lambda)
            assert xindex >= 0

            energysum_spectrum_emission_total[xindex] += packet.e_rf

            emtype = packet.true_emission_type
            if emtype >= 0:
                processtuple = (
                    int(linelist[emtype].atomic_number), int(linelist[emtype].ionstage), 'bound-bound')
                # print(linelist[emtype].lambda_angstroms, lambda_rf)

                if processtuple not in array_energysum_spectra:
                    array_energysum_spectra[processtuple] = (
                        np.zeros_like(array_lambda, dtype=np.float), np.zeros_like(array_lambda, dtype=np.float))

                array_energysum_spectra[processtuple][0][xindex] += packet.e_rf

            if getabsorption:
                abstype = packet.absorption_type
                if at >= 0:
                    processtuple = (
                        int(linelist[abstype].atomic_number), int(linelist[abstype].ionstage), 'bound-bound')

                    if processtuple not in array_energysum_spectra:
                        array_energysum_spectra[processtuple] = (
                            np.zeros_like(array_lambda, dtype=np.float), np.zeros_like(array_lambda, dtype=np.float))

                    array_energysum_spectra[processtuple][1][xindex] += packet.e_rf

    array_flambda_emission_total = (
        energysum_spectrum_emission_total / delta_lambda / (timehigh - timelow) /
        4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs_read)

    contribution_list = []
    for ((atomic_number, ionstage, emissiontype),
         (energysum_spec_emission, energysum_spec_absorption)) in array_energysum_spectra.items():
        array_flambda_emission = (
            energysum_spec_emission / delta_lambda / (timehigh - timelow) /
            4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs_read)

        array_flambda_absorption = (
            energysum_spec_absorption / delta_lambda / (timehigh - timelow) /
            4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs_read)

        if emissiontype == 'bound-bound':
            linelabel = at.get_ionstring(atomic_number, ionstage)
        elif emissiontype != 'free-free':
            linelabel = f'{at.get_ionstring(atomic_number, ionstage)} {emissiontype}'
        else:
            linelabel = f'{emissiontype}'

        fluxcontribthisseries = (
            abs(np.trapz(array_flambda_emission, x=array_lambda)) +
            abs(np.trapz(array_flambda_absorption, x=array_lambda)))

        contribution_list.append(
            fluxcontributiontuple(fluxcontrib=fluxcontribthisseries, linelabel=linelabel,
                                  array_flambda_emission=array_flambda_emission,
                                  array_flambda_absorption=array_flambda_absorption,
                                  color=None))

    return contribution_list, array_flambda_emission_total, array_lambda


def sort_and_reduce_flux_contribution_list(
        contribution_list_in, maxseriescount, arraylambda_angstroms, fixedionlist=[]):

    if fixedionlist:
        # sort in manual order
        contribution_list = sorted(contribution_list_in,
                                   key=lambda x: fixedionlist.index(x.linelabel)
                                   if x.linelabel in fixedionlist else len(fixedionlist) + 1)
    else:
        # sort descending by flux contribution
        contribution_list = sorted(contribution_list_in, key=lambda x: -x.fluxcontrib)

    # combine the items past maxseriescount or not in manual list into a single item
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms, dtype=np.float)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms, dtype=np.float)
    remainder_fluxcontrib = 0

    contribution_list_out = []
    for index, row in enumerate(contribution_list):
        if fixedionlist and row.linelabel in fixedionlist:
            contribution_list_out.append(row._replace(color=color_list[fixedionlist.index(row.linelabel)]))
        elif not fixedionlist and index < maxseriescount:
            contribution_list_out.append(row._replace(color=color_list[index]))
        else:
            remainder_fluxcontrib += row.fluxcontrib
            remainder_flambda_emission += row.array_flambda_emission
            remainder_flambda_absorption += row.array_flambda_absorption

    if remainder_fluxcontrib > 0.:
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


def plot_reference_spectra(axes, plotobjects, plotobjectlabels, args, flambdafilterfunc=None, scale_to_peak=None,
                           **plotkwargs):
    """Plot reference spectra listed in args.refspecfiles."""
    if args.refspecfiles is not None:
        if isinstance(args.refspecfiles, str):
            args.refspecfiles = [args.refspecfiles]
        for index, filename in enumerate(args.refspecfiles):
            if index < len(args.refspeccolors):
                plotkwargs['color'] = args.refspeccolors[index]

            for axindex, axis in enumerate(axes):
                supxmin, supxmax = axis.get_xlim()
                plotobj, serieslabel = plot_reference_spectrum(
                    filename, axis, supxmin, supxmax,
                    flambdafilterfunc, scale_to_peak, scaletoreftime=args.scaletoreftime, **plotkwargs)

                if axindex == 0:
                    plotobjects.append(plotobj)
                    plotobjectlabels.append(serieslabel)


@lru_cache(maxsize=4)
def load_yaml_path(folderpath):
    yamlpath = Path(folderpath, 'metadata.yml')
    if yamlpath.exists():
        with yamlpath.open('r') as yamlfile:
            metadata = yaml.load(yamlfile)
        return metadata
    else:
        return {}


def plot_reference_spectrum(
        filename, axis, xmin, xmax, flambdafilterfunc=None, scale_to_peak=None, scale_to_dist_mpc=1,
        scaletoreftime=None, **plotkwargs):
    """Plot a single reference spectrum.

    The filename must be in space separated text formated with the first two
    columns being wavelength in Angstroms, and F_lambda"""
    if Path(filename).is_file():
        filepath = filename
    else:
        filepath = Path(at.PYDIR, 'data', 'refspectra', filename)

    metadata_all = load_yaml_path(filepath.parent.resolve())
    metadata = metadata_all.get(filename, {})

    flambdaindex = metadata.get('f_lambda_columnindex', 1)

    specdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                           names=['lambda_angstroms', 'f_lambda'], usecols=[0, flambdaindex])

    if 'e_bminusv' in metadata:
        from extinction import apply, ccm89
        if 'r_v' not in metadata:
            metadata['r_v'] = metadata['a_v'] / metadata['e_bminusv']
        elif 'a_v' not in metadata:
            metadata['a_v'] = metadata['e_bminusv'] * metadata['r_v']

        specdata['f_lambda'] = apply(
            ccm89(specdata['lambda_angstroms'].values, a_v=-metadata['a_v'], r_v=metadata['r_v'], unit='aa'),
            specdata['f_lambda'].values)

    if 'z' in metadata:
        specdata['lambda_angstroms'] /= (1 + metadata['z'])

    # scale to flux at required distance
    if scale_to_dist_mpc:
        assert metadata['dist_mpc'] > 0  # we must know the true distance in order to scale to some other distance
        specdata['f_lambda'] = specdata['f_lambda'] * (metadata['dist_mpc'] / scale_to_dist_mpc) ** 2

    if 'label' not in plotkwargs:
        plotkwargs['label'] = metadata['label'] if 'label' in metadata else filename

    if scaletoreftime is not None:
        assert scaletoreftime > 100
        timefactor = math.exp(metadata['t'] / 133.) / math.exp(scaletoreftime / 133.)
        print(f" Scale from time {metadata['t']} to {scaletoreftime}, factor {timefactor}")
        specdata['f_lambda'] *= timefactor
        # plotkwargs['label'] += f' * {timefactor:.2f}'
    if 'scale_factor' in metadata:
        specdata['f_lambda'] *= metadata['scale_factor']

    print(f"Reference spectrum \'{plotkwargs['label']}\' has {len(specdata)} points in the plot range")
    print(f" file: {filename}")

    print(' Metadata: ' + ', '.join([f"{k}='{v}'" if hasattr(v, 'lower') else f'{k}={v}'
                                     for k, v in metadata.items()]))

    specdata.query('lambda_angstroms > @xmin and lambda_angstroms < @xmax', inplace=True)

    print_integrated_flux(
        specdata['f_lambda'], specdata['lambda_angstroms'], distance_megaparsec=metadata['dist_mpc'])

    if len(specdata) > 5000:
        # specdata = scipy.signal.resample(specdata, 10000)
        # specdata = specdata.iloc[::3, :].copy()
        print(f" downsampling to {len(specdata)} points")
        specdata.query('index % 3 == 0', inplace=True)

    # clamp negative values to zero
    # specdata['f_lambda'] = specdata['f_lambda'].apply(lambda x: max(0, x))

    if flambdafilterfunc:
        specdata['f_lambda'] = flambdafilterfunc(specdata['f_lambda'])

    if scale_to_peak:
        specdata['f_lambda_scaled'] = specdata['f_lambda'] / specdata['f_lambda'].max() * scale_to_peak
        ycolumnname = 'f_lambda_scaled'
    else:
        ycolumnname = 'f_lambda'

    if 'linewidth' not in plotkwargs and 'lw' not in plotkwargs:
        plotkwargs['linewidth'] = 0.7

    lineplot = specdata.plot(x='lambda_angstroms', y=ycolumnname, ax=axis, legend=None, **plotkwargs)

    return mpatches.Patch(color=lineplot.get_lines()[0].get_color()), plotkwargs['label']


def make_spectrum_stat_plot(spectrum, figure_title, outputpath, args):
    """Plot the min, max, and average velocity of emission vs wavelength."""
    nsubplots = 2
    fig, axes = plt.subplots(nrows=nsubplots, ncols=1, sharex=True,
                             figsize=(args.figscale * 8, args.figscale * 4 * nsubplots),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    spectrum.query('@args.xmin < lambda_angstroms and lambda_angstroms < @args.xmax', inplace=True)

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=11)

    axis = axes[0]
    axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\AA$]')
    spectrum.eval('f_lambda_not_from_positron = f_lambda - f_lambda_originated_from_positron', inplace=True)
    plotobjects = axis.stackplot(
        spectrum['lambda_angstroms'],
        [spectrum['f_lambda_originated_from_positron'], spectrum['f_lambda_not_from_positron']], linewidth=0)

    plotobjectlabels = ['f_lambda_originated_from_positron', 'f_lambda_not_from_positron']

    # axis.plot(spectrum['lambda_angstroms'], spectrum['f_lambda'], color='black', linewidth=0.5)

    axis.legend(plotobjects, plotobjectlabels, loc='best', handlelength=1,
                frameon=False, numpoints=1)

    axis = axes[1]
    # axis.plot(spectrum['lambda_angstroms'], spectrum['trueemission_velocity_min'], color='#089FFF')
    # axis.plot(spectrum['lambda_angstroms'], spectrum['trueemission_velocity_max'], color='#089FFF')
    axis.fill_between(spectrum['lambda_angstroms'],
                      spectrum['trueemission_velocity_min'],
                      spectrum['trueemission_velocity_max'],
                      alpha=0.5, facecolor='#089FFF')

    # axis.plot(spectrum['lambda_angstroms'], spectrum['emission_velocity_min'], color='#FF9848')
    # axis.plot(spectrum['lambda_angstroms'], spectrum['emission_velocity_max'], color='#FF9848')
    axis.fill_between(spectrum['lambda_angstroms'],
                      spectrum['emission_velocity_min'],
                      spectrum['emission_velocity_max'],
                      alpha=0.5, facecolor='#FF9848')

    axis.plot(spectrum['lambda_angstroms'], spectrum['trueemission_velocity_avg'], color='#1B2ACC',
              label='Average true emission velocity [km/s]')

    axis.plot(spectrum['lambda_angstroms'], spectrum['emission_velocity_avg'], color='#CC4F1B',
              label='Average emission velocity [km/s]')

    axis.set_ylabel('Velocity [km/s]')

    # axis = axes[2]
    # axis.set_ylabel('Number of packets per bin')
    # spectrum.plot(x='lambda_angstroms', y='packetcount', ax=axis)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    if (args.xmax - args.xmin) < 11000:
        axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    elif (args.xmax - args.xmin) < 14000:
        axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=250))

    filenameout = str(Path(outputpath, 'plotspecstats.pdf'))
    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def plot_artis_spectrum(
        axes, modelpath, args, scale_to_peak=None, from_packets=False, filterfunc=None, linelabel=None, **plotkwargs):
    """Plot an ARTIS output spectrum."""

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays)

    modelname = at.get_model_name(modelpath)
    timeavg = (args.timemin + args.timemax) / 2.
    timedelta = (args.timemax - args.timemin) / 2
    if linelabel is None:
        if len(modelname) < 70:
            linelabel = f'{modelname}'
        else:
            linelabel = f'...{modelname[-67:]}'

        linelabel += f' +{timeavg:.0f}d'
        if not args.hidemodeltimerange:
            linelabel += r' ($\pm$ ' + f'{timedelta:.0f}d)'
    else:
        linelabel = linelabel.format(**locals())

    if from_packets:
        spectrum = get_spectrum_from_packets(
            modelpath, args.timemin, args.timemax, lambda_min=args.xmin, lambda_max=args.xmax,
            use_comovingframe=args.use_comovingframe, maxpacketfiles=args.maxpacketfiles)
        make_spectrum_stat_plot(spectrum, linelabel, Path(args.outputfile).parent, args)
    else:
        spectrum = get_spectrum(modelpath, timestepmin, timestepmax, fnufilterfunc=filterfunc)

    spectrum.query('@args.xmin <= lambda_angstroms and lambda_angstroms <= @args.xmax', inplace=True)

    print(f'Plotting {modelname} timesteps {timestepmin} to {timestepmax} '
          f'(t={args.timemin:.3f} to {args.timemax:.3f}d)')
    print_integrated_flux(spectrum['f_lambda'], spectrum['lambda_angstroms'])

    if scale_to_peak:
        spectrum['f_lambda_scaled'] = spectrum['f_lambda'] / spectrum['f_lambda'].max() * scale_to_peak

        ycolumnname = 'f_lambda_scaled'
    else:
        ycolumnname = 'f_lambda'

    for index, axis in enumerate(axes):
        supxmin, supxmax = axis.get_xlim()
        spectrum.query('@supxmin <= lambda_angstroms and lambda_angstroms <= @supxmax').plot(
            x='lambda_angstroms', y=ycolumnname, ax=axis, legend=None,
            label=linelabel if index == 0 else None, **plotkwargs)


def make_spectrum_plot(modelpaths, axes, filterfunc, args, scale_to_peak=None):
    """Plot reference spectra and ARTIS spectra."""

    if not args.refspecafterartis:
        plot_reference_spectra(axes, [], [], args, scale_to_peak=scale_to_peak, flambdafilterfunc=filterfunc)

    for index, modelpath in enumerate(modelpaths):
        plotkwargs = {}
        # plotkwargs['dashes'] = dashesList[index]
        # plotkwargs['dash_capstyle'] = dash_capstyleList[index]
        plotkwargs['linestyle'] = '--' if (int(index / 7) % 2) else '-'
        plotkwargs['linewidth'] = 1.3  # - (0.1 * index)
        plotkwargs['color'] = args.modelcolors[index]
        plotkwargs['alpha'] = 0.9

        linelabel = args.modellabels[index] if index < len(args.modellabels) else None

        plot_artis_spectrum(axes, modelpath, linelabel=linelabel, args=args,
                            scale_to_peak=scale_to_peak, from_packets=args.frompackets,
                            filterfunc=filterfunc, **plotkwargs)

    if args.refspecafterartis:
        plot_reference_spectra(axes, [], [], args, scale_to_peak=scale_to_peak, flambdafilterfunc=filterfunc)

    for axis in axes:
        axis.set_ylim(ymin=0.)
        if args.normalised:
            axis.set_ylim(ymax=1.25)
            axis.set_ylabel(r'Scaled F$_\lambda$')


def make_emissionabsorption_plot(modelpath, axis, filterfunc, args, scale_to_peak=None):
    """Plot the emission and absorption by ion for an ARTIS model."""

    arraynu = at.get_nu_grid(modelpath)

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays)

    modelname = at.get_model_name(modelpath)
    print(f'Plotting {modelname} timesteps {timestepmin} to {timestepmax} '
          f'(t={args.timemin:.3f}d to {args.timemax:.3f}d)')

    if args.frompackets:
        (contribution_list, array_flambda_emission_total,
         arraylambda_angstroms) = at.spectra.get_flux_contributions_from_packets(
            modelpath, args.timemin, args.timemax, args.xmin, args.xmax,
            getabsorption=args.showabsorption, maxpacketfiles=args.maxpacketfiles, filterfunc=filterfunc)
    else:
        arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu
        contribution_list, array_flambda_emission_total = at.spectra.get_flux_contributions(
            modelpath, filterfunc, timestepmin, timestepmax, getabsorption=args.showabsorption)

    at.spectra.print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    contributions_sorted_reduced = at.spectra.sort_and_reduce_flux_contribution_list(
        contribution_list, args.maxseriescount, arraylambda_angstroms, fixedionlist=args.fixedionlist)

    plotobjectlabels = []
    plotobjects = []

    max_flambda_emission_total = max(
        [flambda if (args.xmin < lambda_ang < args.xmax) else -99.0
         for lambda_ang, flambda in zip(arraylambda_angstroms, array_flambda_emission_total)])

    scalefactor = (scale_to_peak / max_flambda_emission_total if scale_to_peak else 1.)

    if args.refspecfiles is None or args.refspecfiles == []:
        plotobjectlabels.append('Net spectrum')
        line = axis.plot(arraylambda_angstroms, array_flambda_emission_total * scalefactor,
                         linewidth=1.5, color='black', zorder=100)
        linecolor = line[0].get_color()
        plotobjects.append(mpatches.Patch(color=linecolor))

    if args.nostack:
        for x in contributions_sorted_reduced:
            emissioncomponentplot = axis.plot(
                arraylambda_angstroms, x.array_flambda_emission * scalefactor, linewidth=1, color=x.color)

            linecolor = emissioncomponentplot[0].get_color()
            plotobjects.append(mpatches.Patch(color=linecolor))

            if args.showabsorption:
                axis.plot(arraylambda_angstroms, -x.array_flambda_absorption * scalefactor,
                          color=linecolor, linewidth=1, alpha=0.6)
    else:
        stackplot = axis.stackplot(
            arraylambda_angstroms,
            [x.array_flambda_emission * scalefactor for x in contributions_sorted_reduced],
            colors=[x.color for x in contributions_sorted_reduced], linewidth=0)
        plotobjects.extend(stackplot)

        if args.showabsorption:
            facecolors = [p.get_facecolor()[0] for p in stackplot]
            axis.stackplot(
                arraylambda_angstroms,
                [-x.array_flambda_absorption * scalefactor for x in contributions_sorted_reduced],
                colors=facecolors, linewidth=0)

    plotobjectlabels.extend(list([x.linelabel for x in contributions_sorted_reduced]))

    plot_reference_spectra([axis], plotobjects, plotobjectlabels, args, flambdafilterfunc=filterfunc,
                           scale_to_peak=scale_to_peak, linewidth=0.5)

    axis.axhline(color='white', linewidth=0.5)

    plotlabel = f'{modelname}\nt={args.timemin:.2f}d to {args.timemax:.2f}d'
    if not args.notitle:
        axis.set_title(plotlabel, fontsize=11)
    # axis.annotate(plotlabel, xy=(0.97, 0.03), xycoords='axes fraction',
    #               horizontalalignment='right', verticalalignment='bottom', fontsize=7)

    axis.set_ylim(ymax=scalefactor * max_flambda_emission_total * 1.2)
    if scale_to_peak:
        axis.set_ylabel(r'Scaled F$_\lambda$')

    return plotobjects, plotobjectlabels


def make_plot(modelpaths, args):
    nrows = len(args.xsplit) + 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, sharey=False,
        figsize=(args.figscale * at.figwidth, args.figscale * at.figwidth * (0.25 + nrows * 0.4)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if nrows == 1:
        axes = [axes]

    if args.filtersavgol:
        import scipy.signal
        window_length, poly_order = [int(x) for x in args.filtersavgol]

        def filterfunc(y):
            return scipy.signal.savgol_filter(y, window_length, poly_order)
    else:
        filterfunc = None

    scale_to_peak = 1.0 if args.normalised else None

    xboundaries = [args.xmin] + args.xsplit + [args.xmax]
    for index, axis in enumerate(axes):
        axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [{}erg/s/cm$^2$/$\AA$]')
        supxmin = xboundaries[index]
        supxmax = xboundaries[index + 1]
        axis.set_xlim(xmin=supxmin, xmax=supxmax)

        if (args.xmax - args.xmin) < 11000:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
        elif (args.xmax - args.xmin) < 14000:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(base=2000))
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=500))

    if args.showemission or args.showabsorption:
        if len(modelpaths) > 1:
            raise ValueError("ERROR: emission/absorption plot can only take one input model", modelpaths)
        legendncol = 2
        defaultoutputfile = Path("plotspecemission_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf")

        plotobjects, plotobjectlabels = make_emissionabsorption_plot(
            modelpaths[0], axes[0], filterfunc, args, scale_to_peak=scale_to_peak)
    else:
        legendncol = 1
        defaultoutputfile = Path("plotspec_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf")

        make_spectrum_plot(modelpaths, axes, filterfunc, args, scale_to_peak=scale_to_peak)
        plotobjects, plotobjectlabels = axes[0].get_legend_handles_labels()

    if not args.nolegend:
        if args.reverselegendorder:
            plotobjects, plotobjectlabels = plotobjects[::-1], plotobjectlabels[::-1]
        leg = axes[0].legend(
            plotobjects, plotobjectlabels, loc='upper right', frameon=False,
            handlelength=1, ncol=legendncol, numpoints=1)

        for artist, text in zip(leg.legendHandles, leg.get_texts()):
            if hasattr(artist, 'get_color'):
                col = artist.get_color()
                artist.set_linewidth(2.0)
                # artist.set_visible(False)  # hide line next to text
            elif hasattr(artist, 'get_facecolor'):
                col = artist.get_facecolor()
            else:
                continue

            if isinstance(col, np.ndarray):
                col = col[0]
            text.set_color(col)

    # plt.setp(plt.getp(axis, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(axis, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #    axis.spines[axis].set_linewidth(framewidth)

    for ax in axes:
        ax.set_xlabel('')
        # ax.xaxis.set_major_formatter(plt.NullFormatter())
        if args.ymin is not None:
            ax.set_ylim(ymin=args.ymin)
        if args.ymax is not None:
            ax.set_ylim(ymax=args.ymax)

    axes[-1].set_xlabel(r'Wavelength [$\AA$]')
    for ax in axes:
        if '{' in ax.get_ylabel():
            ax.yaxis.set_major_formatter(at.ExponentLabelFormatter(axis.get_ylabel(), useMathText=True))

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile = args.outputfile / defaultoutputfile

    filenameout = str(args.outputfile).format(time_days_min=args.timemin, time_days_max=args.timemax)
    fig.savefig(Path(filenameout).open('wb'), format='pdf')
    # plt.show()
    print(f'Saved {filenameout}')
    plt.close()


def write_flambda_spectra(modelpath, args):
    """
    Write lambda_angstroms and f_lambda to .txt files for all timesteps. Also create a text file containing the time
    in days for each timestep.
    """

    outdirectory = Path(modelpath, 'spectrum_data')

    # if not outdirectory.is_dir():
    #     outdirectory.mkdir()

    outdirectory.mkdir(parents=True, exist_ok=True)

    if Path(modelpath, 'specpol.out').is_file():
        specfilename = modelpath / 'specpol.out'
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = [i for i in specdata.columns.values[1:] if i[-2] != '.']
        number_of_timesteps = len(timearray)

    else:
        specfilename = at.firstexisting(['spec.out.gz', 'spec.out'], path=modelpath)
        specdata = pd.read_csv(specfilename, delim_whitespace=True)
        timearray = specdata.columns.values[1:]
        number_of_timesteps = len(specdata.keys()) - 1

    if not args.timestep:
        args.timestep = f'0-{number_of_timesteps - 1}'

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays)

    with open(outdirectory / 'spectra_list.txt', 'w+') as spectra_list:

        for timestep in range(timestepmin, timestepmax + 1):

            spectrum = get_spectrum(modelpath, timestep, timestep)

            with open(outdirectory / f'spec_data_ts_{timestep}.txt', 'w+') as spec_file:

                for wavelength, flambda in zip(spectrum['lambda_angstroms'], spectrum['f_lambda']):
                    spec_file.write(f'{wavelength} {flambda}\n')

            spectra_list.write(str(Path(outdirectory, f'spec_data_ts_{timestep}.txt').absolute()) + '\n')

    with open(outdirectory / 'time_list.txt', 'w+') as time_list:
        for time in timearray:
            time_list.write(f'{str(time)} \n')

    print(f'Saved in {outdirectory}')


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Paths to ARTIS folders with spec.out or packets files')

    parser.add_argument('-modellabels', default=[], nargs='*',
                        help='Model name overrides')

    parser.add_argument('-modelcolors', default=[f'C{i}' for i in range(10)], nargs='*',
                        help='List of colors for ARTIS models')

    parser.add_argument('--frompackets', action='store_true',
                        help='Read packets files directly instead of exspec results')

    parser.add_argument('-maxpacketfiles', type=int, default=-1,
                        help='Limit the number of packet files read')

    parser.add_argument('--emissionabsorption', action='store_true',
                        help='Implies --showemission and --showabsorption')

    parser.add_argument('--showemission', action='store_true',
                        help='Plot the emission spectra by ion/process')

    parser.add_argument('--showabsorption', action='store_true',
                        help='Plot the absorption spectra by ion/process')

    parser.add_argument('--nostack', action='store_true',
                        help="Plot each emission/absorption contribution separately instead of a stackplot")

    parser.add_argument('-fixedionlist', type=list, nargs='+',
                        help='Maximum number of plot series (ions/processes) for emission/absorption plot')

    parser.add_argument('-maxseriescount', type=int, default=12,
                        help='Maximum number of plot series (ions/processes) for emission/absorption plot')

    parser.add_argument('--listtimesteps', action='store_true',
                        help='Show the times at each timestep')

    parser.add_argument('-filtersavgol', nargs=2,
                        help='Savitzky–Golay filter. Specify the window_length and poly_order.'
                        'e.g. -filtersavgol 5 3')

    parser.add_argument('-timestep', '-ts', nargs='?',
                        help='First timestep or a range e.g. 45-65')

    parser.add_argument('-timedays', '-time', '-t', nargs='?',
                        help='Range of times in days to plot (e.g. 50-100)')

    parser.add_argument('-timemin', type=float,
                        help='Lower time in days to integrate spectrum')

    parser.add_argument('-timemax', type=float,
                        help='Upper time in days to integrate spectrum')

    parser.add_argument('-xmin', type=int, default=2500,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=11000,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('-xsplit', nargs='*', default=[],
                        help='Split into subplots at xvalue(s)')

    parser.add_argument('-ymin', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-ymax', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('--hidemodeltimerange', action='store_true',
                        help='Hide the "at t=x to yd" from the line labels')

    parser.add_argument('--normalised', action='store_true',
                        help='Normalise all spectra to their peak values')

    parser.add_argument('--use_comovingframe', action='store_true',
                        help='Use the time of packet escape to the surface (instead of a plane toward the observer)')

    parser.add_argument('-obsspec', '-refspecfiles', action='append', dest='refspecfiles',
                        help='Also plot reference spectrum from this file')

    parser.add_argument('-refspeccolors', default=['0.0', '0.3', '0.5'], nargs='*',
                        help='Set a list of color for reference spectra')

    parser.add_argument('-fluxdistmpc', type=float,
                        help=('Plot flux at this distance in megaparsec. Default is the distance to '
                              'first reference spectrum if this is known, or otherwise 1 Mpc'))

    parser.add_argument('-scaletoreftime', type=float, default=None,
                        help=('Scale reference spectra flux using Co56 decay timescale'))

    parser.add_argument('-figscale', type=float, default=1.8,
                        help='Scale factor for plot area. 1.0 is for single-column')

    parser.add_argument('--notitle', action='store_true',
                        help='Suppress the top title from the plot')

    parser.add_argument('--nolegend', action='store_true',
                        help='Suppress the legend from the plot')

    parser.add_argument('--reverselegendorder', action='store_true',
                        help='Reverse the order of legend items')

    parser.add_argument('--refspecafterartis', action='store_true',
                        help='Plot reference spectra after artis spectra')

    parser.add_argument('-outputfile', '-o', action='store', dest='outputfile', type=Path,
                        help='path/filename for PDF file')

    parser.add_argument('--output_spectra', action='store_true',
                        help='Write out spectra to text files')


def main(args=None, argsraw=None, **kwargs):
    """Plot spectra from ARTIS and reference data."""

    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS model spectra by finding spec.out files '
                        'in the current directory or subdirectories.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = [Path('.')]
    elif isinstance(args.modelpath, (str, Path)):
        args.modelpath = [args.modelpath]

    # flatten the list
    modelpaths = []
    for elem in args.modelpath:
        if isinstance(elem, list):
            modelpaths.extend(elem)
        else:
            modelpaths.append(elem)

    if args.listtimesteps:
        at.showtimesteptimes(modelpath=modelpaths[0])

    elif args.output_spectra:
        for modelpath in modelpaths:
            write_flambda_spectra(modelpath, args)

    else:
        if args.emissionabsorption:
            args.showemission = True
            args.showabsorption = True

        make_plot(modelpaths, args)


if __name__ == "__main__":
    main()
