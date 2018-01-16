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

import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at

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
        specfilename = at.firstexisting(['spec.out.gz', 'spec.out'], path=modelpath)
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


def get_flux_contributions(emissionfilename, absorptionfilename, maxion,
                           timearray, arraynu, filterfunc=None, xmin=-1, xmax=math.inf, timestepmin=0, timestepmax=-1):
    print(f"  Reading {emissionfilename} and {absorptionfilename}")
    emissiondata = pd.read_csv(emissionfilename, sep=' ', header=None)
    absorptiondata = pd.read_csv(absorptionfilename, sep=' ', header=None)

    elementlist = at.get_composition_data(os.path.join(os.path.dirname(emissionfilename), 'compositiondata.txt'))

    arraylambda = const.c.to('angstrom/s').value / arraynu

    nelements = len(elementlist)
    maxyvalueglobal = 0.
    array_flambda_emission_total = np.zeros_like(arraylambda)
    contribution_list = []
    if filterfunc:
        print("Applying filter to reference spectrum")
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

                if selectedcolumn < nelements * maxion:  # bound-bound process
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
                fluxcontribthisseries = integrate_flux(array_fnu_emission, arraynu) + integrate_flux(array_fnu_absorption, arraynu)
                maxyvaluethisseries = max(
                    [array_flambda_emission[i] if (xmin < arraylambda[i] < xmax) else -99.0
                     for i in range(len(array_flambda_emission))])

                maxyvalueglobal = max(maxyvalueglobal, maxyvaluethisseries)

                if emissiontype != 'free-free':
                    linelabel = f'{at.elsymbols[elementlist.Z[element]]} {at.roman_numerals[ion_stage]} {emissiontype}'
                else:
                    linelabel = f'{emissiontype}'

                contribution_list.append(
                    fluxcontributiontuple(fluxcontrib=fluxcontribthisseries, linelabel=linelabel,
                                          array_flambda_emission=array_flambda_emission,
                                          array_flambda_absorption=array_flambda_absorption))

    return contribution_list, maxyvalueglobal, array_flambda_emission_total


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
                filename, axis, args.xmin, args.xmax, args.normalised,
                flambdafilterfunc, scale_to_peak, zorder=1000, **plotkwargs)

            plotobjects.append(plotobj)
            plotobjectlabels.append(serieslabel)


def plot_reference_spectrum(filename, axis, xmin, xmax, normalised,
                            flambdafilterfunc=None, scale_to_peak=None, **plotkwargs):
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(filename):
        filepath = filename
    else:
        filepath = os.path.join(scriptdir, 'data', 'refspectra', filename)

    specdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                           names=['lambda_angstroms', 'f_lambda'], usecols=[0, 1])

    if 'label' not in plotkwargs:
        plotkwargs['label'] = refspectralabels.get(filename, filename)

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

    if normalised:
        specdata['f_lambda_scaled'] = (specdata['f_lambda'] / specdata['f_lambda'].max() *
                                       (scale_to_peak if scale_to_peak else 1.0))
        ycolumnname = 'f_lambda_scaled'
    else:
        ycolumnname = 'f_lambda'

    if 'linewidth' not in plotkwargs and 'lw' not in plotkwargs:
        plotkwargs['linewidth'] = 1.5

    lineplot = specdata.plot(x='lambda_angstroms', y=ycolumnname, ax=axis, **plotkwargs)
    return mpatches.Patch(color=lineplot.get_lines()[0].get_color()), plotkwargs['label']


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


def plot_artis_spectrum(axis, modelpath, args, from_packets=False, filterfunc=None, **plotkwargs):
    (modelname, timestepmin, timestepmax,
     args.timemin, args.timemax) = at.get_model_name_times(
         modelpath, at.get_timestep_times(modelpath),
         args.timestep, args.timemin, args.timemax)

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

    spectrum['f_lambda_scaled'] = spectrum['f_lambda'] / spectrum['f_lambda'].max()
    ycolumnname = 'f_lambda_scaled' if args.normalised else 'f_lambda'
    spectrum.plot(x='lambda_angstroms', y=ycolumnname, ax=axis,
                  label=linelabel, alpha=0.95, **plotkwargs)


def make_spectrum_plot(modelpaths, axis, filterfunc, args):
    """Set up a matplotlib figure and plot observational and ARTIS spectra."""
    plot_reference_spectra(axis, [], [], args, flambdafilterfunc=filterfunc)

    for index, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")
        plotkwargs = {}
        # plotkwargs['dashes'] = dashesList[index]
        # plotkwargs['dash_capstyle'] = dash_capstyleList[index]
        plotkwargs['linestyle'] = '--' if (int(index / 7) % 2) else '-'
        plotkwargs['linewidth'] = 2.5 - (0.2 * index)
        plot_artis_spectrum(axis, modelpath, args=args, from_packets=args.frompackets,
                            filterfunc=filterfunc, **plotkwargs)

    if args.normalised:
        axis.set_ylim(ymin=-0.1, ymax=1.25)
        axis.set_ylabel(r'Scaled F$_\lambda$')


def make_emission_plot(modelpath, axis, filterfunc, args):
    import scipy.interpolate as interpolate
    from cycler import cycler
    maxion = 5  # must match sn3d.h value

    # emissionfilenames = ['emissiontrue.out.gz', 'emissiontrue.out', 'emission.out.gz', 'emission.out']
    emissionfilenames = ['emissiontrue.out.gz', 'emissiontrue.out']
    emissionfilename = at.firstexisting(emissionfilenames, path=modelpath)

    specfilename = at.firstexisting(['spec.out.gz', 'spec.out'], path=modelpath)
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    timearray = specdata.columns.values[1:]
    arraynu = specdata.loc[:, '0'].values
    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu

    (modelname, timestepmin, timestepmax,
     args.timemin, args.timemax) = at.get_model_name_times(
         specfilename, timearray, args.timestep, args.timemin, args.timemax)

    absorptionfilename = at.firstexisting(['absorption.out.gz', 'absorption.out'], path=modelpath)
    contribution_list, maxyvalueglobal, array_flambda_emission_total = at.spectra.get_flux_contributions(
        emissionfilename, absorptionfilename, maxion, timearray, arraynu,
        filterfunc, args.xmin, args.xmax, timestepmin, timestepmax)

    at.spectra.print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    # axis.set_prop_cycle(cycler('color', ))
    # colors = [f'C{n}' for n in range(9)] + ['b' for n in range(25)]
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.linspace(0, 1.0, 20))

    contributions_sorted_reduced = at.spectra.sort_and_reduce_flux_contribution_list(
        contribution_list, args.maxseriescount, arraylambda_angstroms)

    if args.nostack:
        plotobjects = []
        for x in contributions_sorted_reduced:
            line = axis.plot(arraylambda_angstroms, x.array_flambda_emission, linewidth=1)
            plotobjects.append(mpatches.Patch(color=line[0].get_color()))

        # facecolors = [p.get_facecolor()[0] for p in plotobjects]
        #
        # axis.plot(
        #     arraylambda_angstroms, [-x.array_flambda_absorption for x in contributions_sorted_reduced],
        #     colors=facecolors, linewidth=0)
    else:
        plotobjects = axis.stackplot(
            arraylambda_angstroms, [x.array_flambda_emission for x in contributions_sorted_reduced],
            colors=colors, linewidth=0)

        facecolors = [p.get_facecolor()[0] for p in plotobjects]

        axis.stackplot(
            arraylambda_angstroms, [-x.array_flambda_absorption for x in contributions_sorted_reduced],
            colors=facecolors, linewidth=0)

    plotobjectlabels = list([x.linelabel for x in contributions_sorted_reduced])

    plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None,
                           scale_to_peak=(maxyvalueglobal if args.normalised else None), linewidth=0.5)

    axis.axhline(color='white', linewidth=0.5)

    plotlabel = f't={args.timemin:.2f}d to {args.timemax:.2f}d\n{modelname}'
    axis.annotate(plotlabel, xy=(0.97, 0.03), xycoords='axes fraction',
                  horizontalalignment='right', verticalalignment='bottom', fontsize=9)

    # axis.set_ylim(ymin=-0.05 * maxyvalueglobal, ymax=maxyvalueglobal * 1.3)

    return plotobjects, plotobjectlabels


def make_plot(modelpaths, args):
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\AA$]')

    import scipy.signal

    def filterfunc(flambda):
        return scipy.signal.savgol_filter(flambda, 5, 3)

    # filterfunc = None
    if args.emissionabsorption:
        plotobjects, plotobjectlabels = make_emission_plot(modelpaths[0], axis, filterfunc, args)
    else:
        make_spectrum_plot(modelpaths, axis, filterfunc, args)
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

    filenameout = args.outputfile.format(time_days_min=args.timemin, time_days_max=args.timemax)
    if args.frompackets:
        filenameout = filenameout.replace('.pdf', '_frompackets.pdf')
    fig.savefig(filenameout, format='pdf')
    # plt.show()
    print(f'Saved {filenameout}')
    plt.close()


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action='append',
                        help='Paths to ARTIS folders with spec.out or packets files'
                        ' (may include wildcards such as * and **)')

    parser.add_argument('--frompackets', default=False, action='store_true',
                        help='Read packets files directly instead of exspec results')

    parser.add_argument('-maxpacketfiles', type=int, default=-1,
                        help='Limit the number of packet files read')

    parser.add_argument('--emissionabsorption', default=False, action='store_true',
                        help='Show an emission/absorption plot')

    parser.add_argument('--nostack', default=False, action='store_true',
                        help="Don't stack contributions")

    parser.add_argument('-maxseriescount', type=int, default=12,
                        help='Maximum number of plot series (ions/processes) for emission/absorption plot')

    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')

    parser.add_argument('-timestep', '-ts', nargs='?',
                        help='First timestep or a range e.g. 45-65')

    parser.add_argument('-timemin', type=float,
                        help='Lower time in days to integrate spectrum')

    parser.add_argument('-timemax', type=float,
                        help='Upper time in days to integrate spectrum')

    parser.add_argument('-xmin', type=int, default=2500,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=11000,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('--normalised', default=False, action='store_true',
                        help='Normalise the spectra to their peak values')

    parser.add_argument('--use_comovingframe', default=False, action='store_true',
                        help='Use the time of packet escape to the surface (instead of a plane toward the observer)')

    parser.add_argument('-obsspec', action='append', dest='refspecfiles',
                        help='Also plot reference spectrum from this file')

    parser.add_argument('-legendfontsize', type=int, default=8,
                        help='Font size of legend text')

    parser.add_argument('-o', action='store', dest='outputfile',
                        help='path/filename for PDF file')


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
        args.modelpath = ['.', '*']
    elif isinstance(args.modelpath, str):
        args.modelpath = [args.modelpath]
    else:
        args.modelpath = list(chain(*args.modelpath))

    # combined the results of applying wildcards on each input
    modelpaths = list(itertools.chain.from_iterable([glob.glob(x) for x in args.modelpath if os.path.isdir(x)]))

    if args.listtimesteps:
        specfilename = at.firstexisting(['spec.out.gz', 'spec.out'], path=modelpaths[0])
        at.showtimesteptimes(specfilename)
    else:
        if args.emissionabsorption:
            if len(modelpaths) > 1:
                print("ERROR: emission/absorption plot can only take one input model", modelpaths)
                sys.exit()
            defaultoutputfile = "plotspecemission_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf"
        else:
            defaultoutputfile = "plotspec_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf"

        if not args.outputfile:
            args.outputfile = defaultoutputfile
        elif os.path.isdir(args.outputfile):
            args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

        make_plot(modelpaths, args)


if __name__ == "__main__":
    main()
