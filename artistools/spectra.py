#!/usr/bin/env python3

import glob
import math
import os
from collections import namedtuple

import matplotlib.patches as mpatches
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
    'fluxcontribution', 'fluxemissioncontrib linelabel array_flambda_emission array_flambda_absorption')


def stackspectra(spectra_and_factors):
    factor_sum = sum([factor for _, factor in spectra_and_factors])

    for index, (spectrum, factor) in enumerate(spectra_and_factors):
        if index == 0:
            stackedspectrum = spectrum * factor / factor_sum
        else:
            stackedspectrum = stackedspectrum + (spectrum * factor / factor_sum)

    return stackedspectrum


def get_spectrum(specfilename, timestepmin, timestepmax=-1, fnufilterfunc=None):
    """
        Return a pandas DataFrame containing an ARTIS emergent spectrum
    """
    if timestepmax < 0:
        timestepmax = timestepmin

    specdata = pd.read_csv(specfilename, delim_whitespace=True)

    arraynu = specdata.loc[:, '0'].values
    timearray = specdata.columns.values[1:]

    array_fnu = stackspectra([
        (specdata[specdata.columns[timestep + 1]], at.get_timestep_time_delta(timestep, timearray))
        for timestep in range(timestepmin, timestepmax + 1)])

    # best to use the filter on this list because it
    # has regular sampling
    if fnufilterfunc:
        print("Applying filter")
        array_fnu = fnufilterfunc(array_fnu)

    dfspectrum = pd.DataFrame({'nu': arraynu, 'f_nu': array_fnu})
    dfspectrum.sort_values(by='nu', ascending=False, inplace=True)

    dfspectrum['lambda_angstroms'] = const.c.to('angstrom/s').value / dfspectrum['nu']
    dfspectrum['f_lambda'] = dfspectrum['f_nu'] * dfspectrum['nu'] / dfspectrum['lambda_angstroms']

    return dfspectrum


def get_spectrum_from_packets(packetsfiles, timelowdays, timehighdays, lambda_min, lambda_max, delta_lambda=30):
    array_lambda = np.arange(lambda_min, lambda_max, delta_lambda)
    array_energysum = np.zeros_like(array_lambda, dtype=np.float)  # total packet energy sum of each bin

    PARSEC = u.parsec.to('cm')
    timelow = timelowdays * u.day.to('s')
    timehigh = timehighdays * u.day.to('s')
    nprocs = len(packetsfiles)  # hopefully this is true
    c_cgs = const.c.to('cm/s').value
    c_ang_s = const.c.to('angstrom/s').value
    nu_min = c_ang_s / lambda_max
    nu_max = c_ang_s / lambda_min
    for packetsfile in packetsfiles:
        print(f"Loading {packetsfile}")
        dfpackets = at.packets.readfile(packetsfile, usecols=[
            'type_id', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
            'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz'])

        dfpackets.query('type == "TYPE_ESCAPE" and escape_type == "TYPE_RPKT" and'
                        '@nu_min <= nu_rf < @nu_max and'
                        '@timelow < (escape_time - (posx * dirx + posy * diry + posz * dirz) / @c_cgs) < @timehigh',
                        inplace=True)

        print(f"{len(dfpackets)} escaped r-packets with matching nu and arrival time")
        for index, packet in dfpackets.iterrows():
            lambda_rf = c_ang_s / packet.nu_rf
            # pos_dot_dir = packet.posx * packet.dirx + packet.posy * packet.diry + packet.posz * packet.dirz
            # t_arrive = packet['escape_time'] - (pos_dot_dir / c_cgs)
            # print(f"Packet escaped at {t_arrive / u.day.to('s'):.1f} days with nu={packet.nu_rf:.2e}, lambda={lambda_rf:.1f}")
            xindex = math.floor((lambda_rf - lambda_min) / delta_lambda)
            assert(xindex >= 0)
            array_energysum[xindex] += packet.e_rf

    array_flambda = array_energysum / delta_lambda / (timehigh - timelow) / 4.e12 / math.pi / PARSEC / PARSEC / nprocs

    return pd.DataFrame({'lambda_angstroms': array_lambda, 'f_lambda': array_flambda})


def get_flux_contributions(emissionfilename, absorptionfilename, maxion,
                           timearray, arraynu, filterfunc=None, xmin=-1, xmax=math.inf, timestepmin=0, timestepmax=-1):
    # this is much slower than it could be because of the order in which these data tables are accessed
    # TODO: change to use sequential access as much as possible
    print(f"  Reading {emissionfilename} and {absorptionfilename}")
    emissiondata = pd.read_csv(emissionfilename, sep=' ', header=None)
    absorptiondata = pd.read_csv(absorptionfilename, sep=' ', header=None)

    elementlist = at.get_composition_data(os.path.join(os.path.dirname(emissionfilename), 'compositiondata.txt'))

    arraylambda = const.c.to('angstrom/s').value / arraynu

    nelements = len(elementlist)
    maxyvalueglobal = 0.
    array_flambda_emission_total = np.zeros_like(arraylambda)
    contribution_list = []
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
                array_fnu_emission = at.spectra.stackspectra(
                    [(emissiondata.iloc[timestep::len(timearray), selectedcolumn].values,
                      at.get_timestep_time_delta(timestep, timearray))
                     for timestep in range(timestepmin, timestepmax + 1)])

                if selectedcolumn < nelements * maxion:  # bound-bound process
                    array_fnu_absorption = at.spectra.stackspectra(
                        [(absorptiondata.iloc[timestep::len(timearray), selectedcolumn].values,
                          at.get_timestep_time_delta(timestep, timearray))
                         for timestep in range(timestepmin, timestepmax + 1)])
                else:
                    array_fnu_absorption = np.zeros_like(array_fnu_emission)

                # best to use the filter on fnu (because it hopefully has regular sampling)
                if filterfunc:
                    print("Applying filter")
                    array_fnu_emission = filterfunc(array_fnu_emission)
                    if selectedcolumn <= nelements * maxion:
                        array_fnu_absorption = filterfunc(array_fnu_absorption)

                array_flambda_emission = array_fnu_emission * arraynu / arraylambda
                array_flambda_absorption = array_fnu_absorption * arraynu / arraylambda

                array_flambda_emission_total += array_flambda_emission
                fluxemissioncontribthisseries = at.spectra.integrated_flux(array_fnu_emission, arraynu)
                maxyvaluethisseries = max(
                    [array_flambda_emission[i] if (xmin < arraylambda[i] < xmax) else -99.0
                     for i in range(len(array_flambda_emission))])

                maxyvalueglobal = max(maxyvalueglobal, maxyvaluethisseries)

                if emissiontype != 'free-free':
                    linelabel = f'{at.elsymbols[elementlist.Z[element]]} {at.roman_numerals[ion_stage]} {emissiontype}'
                else:
                    linelabel = f'{emissiontype}'

                contribution_list.append(
                    fluxcontributiontuple(fluxemissioncontrib=fluxemissioncontribthisseries, linelabel=linelabel,
                                          array_flambda_emission=array_flambda_emission,
                                          array_flambda_absorption=array_flambda_absorption))

    return contribution_list, maxyvalueglobal, array_flambda_emission_total


def sort_and_reduce_flux_contribution_list(contribution_list_in, maxseriescount, arraylambda_angstroms):
    # sort descending by flux contribution
    contribution_list = sorted(contribution_list_in, key=lambda x: -x.fluxemissioncontrib)

    # combine the items past maxseriescount into a single item
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms)
    remainder_fluxcontrib = 0
    for row in contribution_list[maxseriescount:]:
        remainder_fluxcontrib += row.fluxemissioncontrib
        remainder_flambda_emission += row.array_flambda_emission
        remainder_flambda_absorption += row.array_flambda_absorption

    contribution_list_out = contribution_list[:maxseriescount]
    if remainder_fluxcontrib > 0.:
        contribution_list_out.append(at.spectra.fluxcontributiontuple(
            fluxemissioncontrib=remainder_fluxcontrib, linelabel='other',
            array_flambda_emission=remainder_flambda_emission, array_flambda_absorption=remainder_flambda_absorption))
    return contribution_list_out


def plot_artis_spectrum(axis, filename, xmin, xmax, args, filterfunc=None, **plotkwargs):
    from_packets = os.path.basename(filename).startswith('packets')

    if from_packets:
        specfilename = os.path.join(os.path.dirname(filename), 'spec.out')
    else:
        specfilename = filename

    (modelname, timestepmin, timestepmax,
     time_days_lower, time_days_upper) = at.get_model_name_times(
         filename, at.get_timestep_times(specfilename), args.timestepmin, args.timestepmax, args.timemin, args.timemax)

    linelabel = f'{modelname} at t={time_days_lower:.2f}d to {time_days_upper:.2f}d'

    if from_packets:
        # find any other packets files in the same directory
        packetsfiles_thismodel = glob.glob(os.path.join(os.path.dirname(filename), 'packets**.out'))
        print(packetsfiles_thismodel)
        spectrum = get_spectrum_from_packets(
            packetsfiles_thismodel, time_days_lower, time_days_upper, lambda_min=xmin, lambda_max=xmax)
    else:
        spectrum = get_spectrum(specfilename, timestepmin, timestepmax, fnufilterfunc=filterfunc)

    spectrum.query('@args.xmin < lambda_angstroms and lambda_angstroms < @args.xmax', inplace=True)

    print_integrated_flux(spectrum['f_lambda'], spectrum['lambda_angstroms'])

    spectrum['f_lambda_scaled'] = spectrum['f_lambda'] / spectrum['f_lambda'].max()
    ycolumnname = 'f_lambda_scaled' if args.normalised else 'f_lambda'
    spectrum.plot(x='lambda_angstroms', y=ycolumnname, ax=axis,
                  label=linelabel, alpha=0.95, color=None, **plotkwargs)


def plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None, scale_to_peak=None,
                           **plotkwargs):
    """
        Plot reference spectra listed in args.refspecfiles
    """
    if args.refspecfiles is not None:
        colorlist = ['black', '0.4']
        for index, filename in enumerate(args.refspecfiles):
            serieslabel = refspectralabels.get(filename, filename)

            if index < len(colorlist):
                plotkwargs['color'] = colorlist[index]

            plotobjects.append(
                plot_reference_spectrum(
                    filename, serieslabel, axis, args.xmin, args.xmax, args.normalised,
                    flambdafilterfunc, scale_to_peak, **plotkwargs))

            plotobjectlabels.append(serieslabel)


def plot_reference_spectrum(filename, serieslabel, axis, xmin, xmax, normalised,
                            flambdafilterfunc=None, scale_to_peak=None, **plotkwargs):
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(scriptdir, 'refspectra', filename)
    specdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                           names=['lambda_angstroms', 'f_lambda'], usecols=[0, 1])

    print(f"Reference spectrum '{serieslabel}' has {len(specdata)} points in the plot range")

    specdata.query('lambda_angstroms > @xmin and lambda_angstroms < @xmax', inplace=True)

    at.spectra.print_integrated_flux(specdata.f_lambda, specdata.lambda_angstroms)

    if len(specdata) > 5000:
        # specdata = scipy.signal.resample(specdata, 10000)
        # specdata = specdata.iloc[::3, :].copy()
        specdata.query('index % 3 == 0', inplace=True)
        print(f"  downsamping to {len(specdata)} points")

    # clamp negative values to zero
    specdata['f_lambda'] = specdata['f_lambda'].apply(lambda x: max(0, x))

    if flambdafilterfunc:
        specdata['f_lambda'] = specdata['f_lambda'].apply(flambdafilterfunc)

    if normalised:
        specdata['f_lambda_scaled'] = (specdata['f_lambda'] / specdata['f_lambda'].max() *
                                       (scale_to_peak if scale_to_peak else 1.0))
        ycolumnname = 'f_lambda_scaled'
    else:
        ycolumnname = 'f_lambda'

    if 'linewidth' not in plotkwargs and 'lw' not in plotkwargs:
        plotkwargs['linewidth'] = 1.5

    lineplot = specdata.plot(x='lambda_angstroms', y=ycolumnname, ax=axis, label=serieslabel, zorder=-1, **plotkwargs)
    return mpatches.Patch(color=lineplot.get_lines()[0].get_color())


def integrated_flux(arr_dflux_by_dx, arr_x):
    #  use abs in case arr_x is decreasing
    arr_dx = np.abs(np.diff(arr_x))
    return np.dot(arr_dflux_by_dx[:-1], arr_dx) * u.erg / u.s / (u.cm ** 2)


def print_integrated_flux(arr_f_lambda, arr_lambda_angstroms):
    integrated_flux = at.spectra.integrated_flux(arr_f_lambda, arr_lambda_angstroms)
    luminosity = integrated_flux * 4 * math.pi * (u.megaparsec ** 2)
    print(f'  integrated flux ({arr_lambda_angstroms.min():.1f} A to '
          f'{arr_lambda_angstroms.max():.1f} A): {integrated_flux:.3e}, (L={luminosity.to("Lsun"):.3e})')
