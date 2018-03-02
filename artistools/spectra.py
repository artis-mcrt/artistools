#!/usr/bin/env python3
"""Artistools - spectra related functions."""
import argparse
import glob
import itertools
import math
import os.path
import sys
import warnings
from collections import namedtuple
from itertools import chain
from pathlib import Path
from typing import Iterable

import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at

# filename : (label, distance / Mpc)
# use -1 for distance if unknown
refspectra = {
    'sn2011fe_PTF11kly_20120822_norm.txt':
        ('SN2011fe +364d (Mazzali et al. 2015)', -1),

    'sn2011fe_PTF11kly_20120822_norm_1Mpc.txt':
        ('SN2011fe (1 Mpc) +364d (Mazzali et al. 2015)', 1),

    '2010lp_20110928_fors2.txt':
        ('SN2010lp +264d (Taubenberger et al. 2013)', -1),

    'dop_dered_SN2013aa_20140208_fc_final.txt':
        ('SN2013aa +360d (Maguire et al. in prep)', -1),

    '2003du_20031213_3219_8822_00.txt':
        ('SN2003du +221.3d (Stanishev et al. 2007)', -1),

    '2003du_20031213_3219_8822_00_1Mpc.txt':
        ('SN2003du +221.3d (Stanishev et al. 2007)', 1),

    'FranssonJerkstrand2015_W7_330d_1Mpc':
        ('Fransson & Jerkstrand (2015) W7 Iwamoto+1999 1Mpc +330d', 1),

    'nero-nebspec.txt':
        ('NERO +300d one-zone', -1),

    'maurer2011_RTJ_W7_338d_1Mpc.txt':
        ('RTJ W7 Nomoto+1984 1Mpc +338d (Maurer et al. 2011)', 1),
}

fluxcontributiontuple = namedtuple(
    'fluxcontribution', 'fluxcontrib linelabel array_flambda_emission array_flambda_absorption')


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

    if os.path.isdir(modelpath):
        specfilename = at.firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
    else:
        specfilename = modelpath
    specdata = pd.read_csv(specfilename, delim_whitespace=True)

    nu = specdata.loc[:, '0'].values
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

    c = const.c.to('angstrom/s').value
    dfspectrum.eval('lambda_angstroms = @c / nu', inplace=True)
    dfspectrum.eval('f_lambda = f_nu * nu / lambda_angstroms', inplace=True)

    return dfspectrum


def get_spectrum_from_packets(packetsfiles, timelowdays, timehighdays, lambda_min, lambda_max, delta_lambda=30,
                              use_comovingframe=None):
    if use_comovingframe:
        modeldata, _ = at.get_modeldata(os.path.dirname(packetsfiles[0]))
        vmax = modeldata.iloc[-1].velocity * u.km / u.s
        betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    def update_min_sum_max(array, xindex, e_rf, value):
        if value < array[0][xindex] or array[0][xindex] == 0:
            array[0][xindex] = value

        array[1][xindex] += e_rf * value

        if value > array[2][xindex] or array[2][xindex] == 0:
            array[2][xindex] = value

    import artistools.packets
    array_lambda = np.arange(lambda_min, lambda_max, delta_lambda)
    array_energysum = np.zeros_like(array_lambda, dtype=np.float)  # total packet energy sum of each bin
    array_energysum_positron = np.zeros_like(array_lambda, dtype=np.float)  # total packet energy sum of each bin
    array_pktcount = np.zeros_like(array_lambda, dtype=np.int)  # number of packets in each bin
    array_emvelocity = np.zeros((3, len(array_lambda)), dtype=np.float)
    array_trueemvelocity = np.zeros((3, len(array_lambda)), dtype=np.float)

    timelow = timelowdays * u.day.to('s')
    timehigh = timehighdays * u.day.to('s')
    nprocs = len(packetsfiles)  # hopefully this is true
    c_cgs = const.c.to('cm/s').value
    c_ang_s = const.c.to('angstrom/s').value
    nu_min = c_ang_s / lambda_max
    nu_max = c_ang_s / lambda_min
    for packetsfile in packetsfiles:
        dfpackets = at.packets.readfile(packetsfile, usecols=[
            'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
            'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz',
            'em_posx', 'em_posy', 'em_posz', 'em_time',
            'true_emission_velocity', 'originated_from_positron'])

        querystr = 'type == "TYPE_ESCAPE" and escape_type == "TYPE_RPKT" and @nu_min <= nu_rf < @nu_max and'
        if not use_comovingframe:
            querystr += '@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh'
        else:
            querystr += '@timelow < escape_time * @betafactor < @timehigh'

        dfpackets.query(querystr, inplace=True)

        print(f"{len(dfpackets)} escaped r-packets with matching nu and arrival time")
        for _, packet in dfpackets.iterrows():
            lambda_rf = c_ang_s / packet.nu_rf
            # pos_dot_dir = packet.posx * packet.dirx + packet.posy * packet.diry + packet.posz * packet.dirz
            # t_arrive = packet['escape_time'] - (pos_dot_dir / c_cgs)
            # print(f"Packet escaped at {t_arrive / u.day.to('s'):.1f} days with "
            #       f"nu={packet.nu_rf:.2e}, lambda={lambda_rf:.1f}")
            xindex = math.floor((lambda_rf - lambda_min) / delta_lambda)
            assert(xindex >= 0)

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
                     4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs)

    array_flambda_positron = (array_energysum_positron / delta_lambda / (timehigh - timelow) /
                              4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs)

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


def get_flux_contributions(emissionfilename, absorptionfilename, timearray, arraynu,
                           filterfunc=None, xmin=-1, xmax=math.inf, timestepmin=0, timestepmax=None):
    arraylambda = const.c.to('angstrom/s').value / arraynu
    elementlist = at.get_composition_data(os.path.dirname(emissionfilename))
    nelements = len(elementlist)

    print(f'  Reading {emissionfilename}')
    emissiondata = pd.read_csv(emissionfilename, delim_whitespace=True, header=None)
    maxion_float = (emissiondata.shape[1] - 1) / 2 / nelements  # also known as MIONS in ARTIS sn3d.h
    assert maxion_float.is_integer()
    maxion = int(maxion_float)
    print(f'  inferred MAXION = {maxion} from emission file using nlements = {nelements} from compositiondata.txt')

    # check that the row count is product of timesteps and frequency bins found in spec.out
    assert emissiondata.shape[0] == len(arraynu) * len(timearray)

    if absorptionfilename:
        print(f'  Reading {absorptionfilename}')
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
                    integrate_flux(array_fnu_emission, arraynu) + integrate_flux(array_fnu_absorption, arraynu))

                if emissiontype != 'free-free':
                    linelabel = f'{at.elsymbols[elementlist.Z[element]]} {at.roman_numerals[ion_stage]} {emissiontype}'
                else:
                    linelabel = f'{emissiontype}'

                contribution_list.append(
                    fluxcontributiontuple(fluxcontrib=fluxcontribthisseries, linelabel=linelabel,
                                          array_flambda_emission=array_flambda_emission,
                                          array_flambda_absorption=array_flambda_absorption))

    return contribution_list, array_flambda_emission_total


def sort_and_reduce_flux_contribution_list(contribution_list_in, maxseriescount, arraylambda_angstroms):
    # sort descending by flux contribution
    contribution_list = sorted(contribution_list_in, key=lambda x: -x.fluxcontrib)

    # combine the items past maxseriescount into a single item
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms)
    remainder_fluxcontrib = 0
    for row in contribution_list[maxseriescount:]:
        remainder_fluxcontrib += row.fluxcontrib
        remainder_flambda_emission += row.array_flambda_emission
        remainder_flambda_absorption += row.array_flambda_absorption

    contribution_list_out = contribution_list[:maxseriescount]
    if remainder_fluxcontrib > 0.:
        contribution_list_out.append(fluxcontributiontuple(
            fluxcontrib=remainder_fluxcontrib, linelabel='other',
            array_flambda_emission=remainder_flambda_emission, array_flambda_absorption=remainder_flambda_absorption))
    return contribution_list_out


def integrate_flux(arr_dflux_by_dx, arr_x):
    #  use abs in case arr_x is decreasing
    arr_dx = np.abs(np.diff(arr_x))
    return np.dot(arr_dflux_by_dx[:-1], arr_dx) * u.erg / u.s / (u.cm ** 2)


def print_integrated_flux(arr_f_lambda, arr_lambda_angstroms):
    integrated_flux = integrate_flux(arr_f_lambda, arr_lambda_angstroms)
    luminosity = integrated_flux * 4 * math.pi * (u.megaparsec ** 2)
    print(f'  integrated flux ({arr_lambda_angstroms.min():.1f} A to '
          f'{arr_lambda_angstroms.max():.1f} A): {integrated_flux:.3e}, (L={luminosity.to("Lsun"):.3e})')


def plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None, scale_to_peak=None,
                           **plotkwargs):
    """Plot reference spectra listed in args.refspecfiles."""
    if args.refspecfiles is not None:
        colorlist = ['black', '0.4']
        for index, filename in enumerate(args.refspecfiles):
            if index < len(colorlist):
                plotkwargs['color'] = colorlist[index]

            plotobj, serieslabel = plot_reference_spectrum(
                filename, axis, args.xmin, args.xmax,
                flambdafilterfunc, scale_to_peak, zorder=1000, **plotkwargs)

            plotobjects.append(plotobj)
            plotobjectlabels.append(serieslabel)


def plot_reference_spectrum(filename, axis, xmin, xmax, flambdafilterfunc=None, scale_to_peak=None, **plotkwargs):
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(filename):
        filepath = filename
    else:
        filepath = os.path.join(scriptdir, 'data', 'refspectra', filename)

    specdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                           names=['lambda_angstroms', 'f_lambda'], usecols=[0, 1])

    if 'label' not in plotkwargs:
        plotkwargs['label'] = refspectra.get(filename, [filename, -1])[0]

    serieslabel = plotkwargs['label']

    print(f"Reference spectrum '{serieslabel}' has {len(specdata)} points in the plot range")

    specdata.query('lambda_angstroms > @xmin and lambda_angstroms < @xmax', inplace=True)

    print_integrated_flux(specdata.f_lambda, specdata.lambda_angstroms)

    if len(specdata) > 5000:
        # specdata = scipy.signal.resample(specdata, 10000)
        # specdata = specdata.iloc[::3, :].copy()
        print(f"  downsampling to {len(specdata)} points")
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
        plotkwargs['linewidth'] = 1.5

    lineplot = specdata.plot(x='lambda_angstroms', y=ycolumnname, ax=axis, **plotkwargs)
    # lineplot.get_lines()[0].get_color())
    return mpatches.Patch(color=plotkwargs['color']), plotkwargs['label']


def make_spectrum_stat_plot(spectrum, figure_title, outputpath, args):
    nsubplots = 2
    fig, axes = plt.subplots(nsubplots, 1, sharex=True, figsize=(8, 4 * nsubplots),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    spectrum.query('@args.xmin < lambda_angstroms and lambda_angstroms < @args.xmax', inplace=True)

    axes[0].set_title(figure_title, fontsize=11)

    axis = axes[0]
    axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\AA$]')
    spectrum.eval('f_lambda_not_from_positron = f_lambda - f_lambda_originated_from_positron', inplace=True)
    plotobjects = axis.stackplot(
        spectrum['lambda_angstroms'],
        [spectrum['f_lambda_originated_from_positron'], spectrum['f_lambda_not_from_positron']], linewidth=0)

    plotobjectlabels = ['f_lambda_originated_from_positron', 'f_lambda_not_from_positron']

    # axis.plot(spectrum['lambda_angstroms'], spectrum['f_lambda'], color='black', linewidth=0.5)

    axis.legend(plotobjects, plotobjectlabels, loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': args.legendfontsize})

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
    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': args.legendfontsize})

    # axis = axes[2]
    # axis.set_ylabel('Number of packets per bin')
    # spectrum.plot(x='lambda_angstroms', y='packetcount', ax=axis)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))

    filenameout = os.path.join(outputpath, 'plotspecstats.pdf')
    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def plot_artis_spectrum(axis, modelpath, args, scale_to_peak=None, from_packets=False, filterfunc=None, **plotkwargs):
    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        at.get_timestep_times(modelpath), args.timestep, args.timemin, args.timemax, args.timedays)

    modelname = at.get_model_name(modelpath)
    print(f'Plotting {modelname} timesteps {timestepmin} to {timestepmax} '
          f'(t={args.timemin:.3f}d to {args.timemax:.3f}d)')

    linelabel = f'{modelname} at t={args.timemin:.2f}d to {args.timemax:.2f}d'

    if from_packets:
        # find any other packets files in the same directory
        packetsfiles_thismodel = sorted(
            glob.glob(os.path.join(modelpath, 'packets00_*.out')) +
            glob.glob(os.path.join(modelpath, 'packets00_*.out.gz')))
        if args.maxpacketfiles >= 0 and len(packetsfiles_thismodel) > args.maxpacketfiles:
            print(f'Using on the first {args.maxpacketfiles} packet files out of {len(packetsfiles_thismodel)}')
            packetsfiles_thismodel = packetsfiles_thismodel[:args.maxpacketfiles]
        spectrum = get_spectrum_from_packets(
            packetsfiles_thismodel, args.timemin, args.timemax, lambda_min=args.xmin, lambda_max=args.xmax,
            use_comovingframe=args.use_comovingframe)
        make_spectrum_stat_plot(spectrum, linelabel, os.path.dirname(args.outputfile), args)
    else:
        spectrum = get_spectrum(modelpath, timestepmin, timestepmax, fnufilterfunc=filterfunc)

    spectrum.query('@args.xmin < lambda_angstroms and lambda_angstroms < @args.xmax', inplace=True)

    at.spectra.print_integrated_flux(spectrum['f_lambda'], spectrum['lambda_angstroms'])

    if scale_to_peak:
        spectrum['f_lambda_scaled'] = spectrum['f_lambda'] / spectrum['f_lambda'].max() * scale_to_peak

        ycolumnname = 'f_lambda_scaled'
    else:
        ycolumnname = 'f_lambda'

    spectrum.plot(x='lambda_angstroms', y=ycolumnname, ax=axis,
                  label=linelabel, alpha=0.95, **plotkwargs)


def make_spectrum_plot(modelpaths, axis, filterfunc, args, scale_to_peak=None):
    """Set up a matplotlib figure and plot observational and ARTIS spectra."""
    plot_reference_spectra(axis, [], [], args, scale_to_peak=scale_to_peak, flambdafilterfunc=filterfunc)

    for index, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")
        plotkwargs = {}
        # plotkwargs['dashes'] = dashesList[index]
        # plotkwargs['dash_capstyle'] = dash_capstyleList[index]
        plotkwargs['linestyle'] = '--' if (int(index / 7) % 2) else '-'
        plotkwargs['linewidth'] = 2.5 - (0.2 * index)
        plot_artis_spectrum(axis, modelpath, args=args, scale_to_peak=scale_to_peak, from_packets=args.frompackets,
                            filterfunc=filterfunc, **plotkwargs)

    if args.normalised:
        axis.set_ylim(ymin=-0.1, ymax=1.25)
        axis.set_ylabel(r'Scaled F$_\lambda$')


def make_emissionabsorption_plot(modelpath, axis, filterfunc, args, scale_to_peak=None):
    import scipy.interpolate as interpolate
    from cycler import cycler

    # emissionfilenames = ['emissiontrue.out.gz', 'emissiontrue.out', 'emission.out.gz', 'emission.out']
    emissionfilenames = ['emissiontrue.out.gz', 'emissiontrue.out']
    emissionfilename = at.firstexisting(emissionfilenames, path=modelpath)

    specfilename = at.firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    timearray = specdata.columns.values[1:]
    arraynu = specdata.loc[:, '0'].values
    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu

    (timestepmin, timestepmax,
     args.timemin, args.timemax) = at.get_time_range(
         timearray, args.timestep, args.timemin, args.timemax, args.timedays)

    modelname = at.get_model_name(modelpath)
    print(f'Plotting {modelname} timesteps {timestepmin} to {timestepmax} '
          f'(t={args.timemin:.3f}d to {args.timemax:.3f}d)')

    absorptionfilename = (at.firstexisting(['absorption.out.gz', 'absorption.out'], path=modelpath)
                          if args.showabsorption else None)
    contribution_list, array_flambda_emission_total = at.spectra.get_flux_contributions(
        emissionfilename, absorptionfilename, timearray, arraynu,
        filterfunc, args.xmin, args.xmax, timestepmin, timestepmax)

    at.spectra.print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    # axis.set_prop_cycle(cycler('color', ))
    # colors = [f'C{n}' for n in range(9)] + ['b' for n in range(25)]
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1.0, 20))

    contributions_sorted_reduced = at.spectra.sort_and_reduce_flux_contribution_list(
        contribution_list, args.maxseriescount, arraylambda_angstroms)

    plotobjectlabels = []
    plotobjects = []

    max_flambda_emission_total = max(
        [flambda if (args.xmin < lambda_ang < args.xmax) else -99.0
         for lambda_ang, flambda in zip(arraylambda_angstroms, array_flambda_emission_total)])

    scalefactor = (scale_to_peak / max_flambda_emission_total if scale_to_peak else 1.)

    if args.nostack:
        plotobjectlabels.append('Net spectrum')
        line = axis.plot(arraylambda_angstroms, array_flambda_emission_total * scalefactor, linewidth=1, color='black')
        linecolor = line[0].get_color()
        plotobjects.append(mpatches.Patch(color=linecolor))

        for x in contributions_sorted_reduced:
            emissioncomponentplot = axis.plot(
                arraylambda_angstroms, x.array_flambda_emission * scalefactor, linewidth=1)

            linecolor = emissioncomponentplot[0].get_color()
            plotobjects.append(mpatches.Patch(color=linecolor))

            if args.showabsorption:
                axis.plot(arraylambda_angstroms, -x.array_flambda_absorption * scalefactor,
                          color=linecolor, linewidth=1, alpha=0.6)
    else:
        stackplot = axis.stackplot(
            arraylambda_angstroms,
            [x.array_flambda_emission * scalefactor for x in contributions_sorted_reduced],
            colors=colors, linewidth=0)
        plotobjects.extend(stackplot)

        if args.showabsorption:
            facecolors = [p.get_facecolor()[0] for p in stackplot]
            axis.stackplot(
                arraylambda_angstroms,
                [-x.array_flambda_absorption * scalefactor for x in contributions_sorted_reduced],
                colors=facecolors, linewidth=0)

    plotobjectlabels.extend(list([x.linelabel for x in contributions_sorted_reduced]))

    plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=filterfunc,
                           scale_to_peak=scale_to_peak, linewidth=0.5)

    axis.axhline(color='white', linewidth=0.5)

    plotlabel = f'{modelname}\nt={args.timemin:.2f}d to {args.timemax:.2f}d'
    axis.annotate(plotlabel, xy=(0.97, 0.03), xycoords='axes fraction',
                  horizontalalignment='right', verticalalignment='bottom', fontsize=7)

    # axis.set_ylim(ymin=-0.05 * maxyvalueglobal, ymax=maxyvalueglobal * 1.3)
    if scale_to_peak:
        axis.set_ylabel(r'Scaled F$_\lambda$')

    return plotobjects, plotobjectlabels


def make_plot(modelpaths, args):
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\AA$]')

    import scipy.signal
    if args.filtersavgol:
        window_length, poly_order = args.filtersavgol

        def filterfunc(y):
            return scipy.signal.savgol_filter(y, window_length, poly_order)
    else:
        filterfunc = None

    scale_to_peak = 1.0 if args.normalised else None

    if args.showemission or args.showabsorption:
        if len(modelpaths) > 1:
            print("ERROR: emission/absorption plot can only take one input model", modelpaths)
            sys.exit()

        defaultoutputfile = Path("plotspecemission_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf")

        plotobjects, plotobjectlabels = make_emissionabsorption_plot(
            modelpaths[0], axis, filterfunc, args, scale_to_peak=scale_to_peak)
    else:
        defaultoutputfile = Path("plotspec_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf")

        make_spectrum_plot(modelpaths, axis, filterfunc, args, scale_to_peak=scale_to_peak)
        plotobjects, plotobjectlabels = axis.get_legend_handles_labels()

    axis.legend(plotobjects, plotobjectlabels, loc='best', handlelength=2,
                frameon=False, numpoints=1, prop={'size': args.legendfontsize})

    # plt.setp(plt.getp(axis, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(axis, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #    axis.spines[axis].set_linewidth(framewidth)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not args.outputfile.suffixes:
        args.outputfile = args.outputfile / defaultoutputfile

    filenameout = str(args.outputfile).format(time_days_min=args.timemin, time_days_max=args.timemax)
    if args.frompackets:
        filenameout = filenameout.replace('.pdf', '_frompackets.pdf')
    fig.savefig(Path(filenameout).open('wb'), format='pdf')
    # plt.show()
    print(f'Saved {filenameout}')
    plt.close()


def write_flambda_spectra(modelpath, args):
    """
    Write lambda_angstroms and f_lambda to .txt files for all timesteps. Also write text file with path to files and a
    file specifying filters. This can be used as input to https://github.com/cinserra/S3/blob/master/src/s3/SMS.py
    to plot synthetic magnitudes from spectra.
    """

    outdirectory = 'spectrum_data/'

    if not os.path.exists('spectrum_data'):
        os.makedirs('spectrum_data')

    open(outdirectory + 'spectra_list.txt', 'w+').close()  # clear files
    open(outdirectory + 'filter_list.txt', 'w+').close()

    specfilename = at.firstexisting(['spec.out.gz', 'spec.out', 'specpol.out'], path=modelpath)
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    timearray = specdata.columns.values[1:]
    number_of_timesteps = len(specdata.keys()) - 1

    if not args.timestep:
        args.timestep = f'0-{number_of_timesteps - 1}'

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        timearray, args.timestep, args.timemin, args.timemax, args.timedays)

    spectra_list = open(outdirectory + 'spectra_list.txt', 'a')
    filter_list = open(outdirectory + 'filter_list.txt', 'a')

    filter_name = ['B', 'V', 'R', 'U']

    for timestep in range(timestepmin, timestepmax + 1):

        spectrum = get_spectrum(modelpath, timestep, timestep)

        spec_file = open(outdirectory + 'spec_data_ts_' + str(timestep) + '.txt', 'w+')

        for wavelength, flambda in zip(spectrum['lambda_angstroms'], spectrum['f_lambda']):
            spec_file.write(f'{wavelength} {flambda}\n')
        spec_file.close()

        spectra_list.write(os.path.realpath(outdirectory + 'spec_data_ts_' + str(timestep) + '.txt') + '\n')

        for i in filter_name:
            filter_list.write(f'{i}\n')

    spectra_list.close()
    filter_list.close()

    print('Saved in ' + outdirectory)


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Paths to ARTIS folders with spec.out or packets files'
                        ' (may include wildcards such as * and **)')

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

    parser.add_argument('--normalised', action='store_true',
                        help='Normalise the spectra to their peak values')

    parser.add_argument('--use_comovingframe', action='store_true',
                        help='Use the time of packet escape to the surface (instead of a plane toward the observer)')

    parser.add_argument('-obsspec', action='append', dest='refspecfiles',
                        help='Also plot reference spectrum from this file')

    parser.add_argument('-legendfontsize', type=int, default=8,
                        help='Font size of legend text')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path,
                        help='path/filename for PDF file')

    parser.add_argument('--output_spectra', action='store_true',
                        help='Write out spectra to text files')


def main(args=None, argsraw=None, **kwargs):
    warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
    """
        Plot ARTIS spectra and (optionally) reference spectra
    """

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
    elif not isinstance(args.modelpath, Iterable):
        args.modelpath = [args.modelpath]

    # flatten the list
    modelpaths = []
    for elem in args.modelpath:
        if isinstance(elem, list):
            modelpaths.extend(elem)
        else:
            modelpaths.append(elem)

    # applying any wildcards to the modelpaths
    modelpaths = list(itertools.chain.from_iterable([
        list(Path().glob(pattern=str(x)))if not x.samefile(Path('.')) else [Path('.')] for x in modelpaths]))

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
