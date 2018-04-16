#!/usr/bin/env python3
"""Artistools - spectra related functions."""
import argparse
import glob
import math
from collections import namedtuple
from pathlib import Path

import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u

import artistools as at

# dict consistint of {filename : (legend label, distance in Mpc)}
# use -1 for distance if unknown
refspectra = {
    '2010lp_20110928_fors2.txt':
        ('SN2010lp +264d (Taubenberger et al. 2013)', -1),

    'sn2011fe_PTF11kly_20120822_norm.txt':
        ('SN2011fe +364d (Mazzali et al. 2015)', 6.40),

    'dop_dered_SN2013aa_20140208_fc_final.txt':
        ('SN2013aa +360d (Maguire et al. in prep)', 13.95),

    '2003du_20031213_3219_8822_00.txt':
        ('SN2003du +221.3d (Stanishev et al. 2007)', 30.47),

    'FranssonJerkstrand2015_W7_330d_10Mpc.txt':
        ('W7 (Fransson & Jerkstrand 2015) +330d', 10),  # Iwamoto+1999

    'maurer2011_RTJ_W7_338d_1Mpc.txt':
        ('RTJ W7 Nomoto+1984 +338d (Maurer et al. 2011)', 1),

    'nero-nebspec.txt':
        ('NERO +300d one-zone', 1),
}

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

    c = const.c.to('angstrom/s').value
    dfspectrum.eval('lambda_angstroms = @c / nu', inplace=True)
    dfspectrum.eval('f_lambda = f_nu * nu / lambda_angstroms', inplace=True)

    return dfspectrum


def get_spectrum_from_packets(packetsfiles, timelowdays, timehighdays, lambda_min, lambda_max, delta_lambda=30,
                              use_comovingframe=None, filehandles=None):
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
    assert not filehandles or len(packetsfiles) == len(filehandles)
    for index, packetsfile in enumerate(packetsfiles):
        dfpackets = at.packets.readfile(
            packetsfile if not filehandles else filehandles[index],
            usecols=[
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
    modelpath = Path(emissionfilename).parent
    elementlist = at.get_composition_data(modelpath)
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
                    abs(np.trapz(array_fnu_emission, x=arraynu)) + abs(np.trapz(array_fnu_absorption, x=arraynu)))

                if emissiontype == 'bound-bound':
                    linelabel = f'{at.elsymbols[elementlist.Z[element]]} {at.roman_numerals[ion_stage]}'
                elif emissiontype != 'free-free':
                    linelabel = f'{at.elsymbols[elementlist.Z[element]]} {at.roman_numerals[ion_stage]} {emissiontype}'
                else:
                    linelabel = f'{emissiontype}'

                contribution_list.append(
                    fluxcontributiontuple(fluxcontrib=fluxcontribthisseries, linelabel=linelabel,
                                          array_flambda_emission=array_flambda_emission,
                                          array_flambda_absorption=array_flambda_absorption,
                                          color=None))

    return contribution_list, array_flambda_emission_total


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
    remainder_flambda_emission = np.zeros_like(arraylambda_angstroms)
    remainder_flambda_absorption = np.zeros_like(arraylambda_angstroms)
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
    luminosity = integrated_flux * 4 * math.pi * (distance_megaparsec * u.megaparsec ** 2)
    print(f'  integrated flux ({arr_lambda_angstroms.min():.1f} A to '
          f'{arr_lambda_angstroms.max():.1f} A): {integrated_flux:.3e}, (L={luminosity.to("Lsun"):.3e})')


def plot_reference_spectra(axes, plotobjects, plotobjectlabels, args, flambdafilterfunc=None, scale_to_peak=None,
                           **plotkwargs):
    """Plot reference spectra listed in args.refspecfiles."""
    if args.refspecfiles is not None:
        if isinstance(args.refspecfiles, str):
            args.refspecfiles = [args.refspecfiles]
        colorlist = ['black', '0.4']
        for index, filename in enumerate(args.refspecfiles):
            print(filename)
            if index < len(colorlist):
                plotkwargs['color'] = colorlist[index]

            for index, axis in enumerate(axes):
                supxmin, supxmax = axis.get_xlim()
                plotobj, serieslabel = plot_reference_spectrum(
                    filename, axis, supxmin, supxmax,
                    flambdafilterfunc, scale_to_peak, zorder=1000, **plotkwargs)

                if index == 0:
                    plotobjects.append(plotobj)
                    plotobjectlabels.append(serieslabel)


def plot_reference_spectrum(
    filename, axis, xmin, xmax, flambdafilterfunc=None, scale_to_peak=None, scale_to_dist_mpc=1, **plotkwargs):
    """Plot a single reference spectrum.

    The filename must be in space separated text formated with the first two
    columns being wavelength in Angstroms, and F_lambda"""
    if Path(filename).is_file():
        filepath = filename
    else:
        filepath = Path(at.PYDIR, 'data', 'refspectra', filename)

    objectlabel, objectdist_megaparsec = refspectra.get(filename, [filename, -1])

    specdata = pd.read_csv(filepath, delim_whitespace=True, header=None,
                           names=['lambda_angstroms', 'f_lambda'], usecols=[0, 1])

    # scale to flux at required distance
    if scale_to_dist_mpc:
        assert objectdist_megaparsec > 0  # we must know the true distance in order to scale to some other distance
        specdata['f_lambda'] = specdata['f_lambda'] * (objectdist_megaparsec / scale_to_dist_mpc) ** 2

    if 'label' not in plotkwargs:
        plotkwargs['label'] = objectlabel

    serieslabel = plotkwargs['label']
    print(f"Reference spectrum '{serieslabel}' has {len(specdata)} points in the plot range")

    specdata.query('lambda_angstroms > @xmin and lambda_angstroms < @xmax', inplace=True)

    print_integrated_flux(specdata.f_lambda, specdata.lambda_angstroms, distance_megaparsec=objectdist_megaparsec)

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
        plotkwargs['linewidth'] = 0.5

    lineplot = specdata.plot(x='lambda_angstroms', y=ycolumnname, ax=axis, legend=None, **plotkwargs)
    # lineplot.get_lines()[0].get_color())
    return mpatches.Patch(color=plotkwargs['color']), plotkwargs['label']


def make_spectrum_stat_plot(spectrum, figure_title, outputpath, args):
    """Plot the min, max, and average velocity of emission vs wavelength."""
    nsubplots = 2
    fig, axes = plt.subplots(nrows=nsubplots, ncols=1, sharex=True, figsize=(8, 4 * nsubplots),
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

    axis.legend(plotobjects, plotobjectlabels, loc='best', handlelength=2,
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
    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': args.legendfontsize})

    # axis = axes[2]
    # axis.set_ylabel('Number of packets per bin')
    # spectrum.plot(x='lambda_angstroms', y='packetcount', ax=axis)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    if args.xmax - args.xmin < 11000:
        axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))

    filenameout = str(Path(outputpath, 'plotspecstats.pdf'))
    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def plot_artis_spectrum(axes, modelpath, args, scale_to_peak=None, from_packets=False, filterfunc=None, **plotkwargs):
    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        at.get_timestep_times(modelpath), args.timestep, args.timemin, args.timemax, args.timedays)

    modelname = at.get_model_name(modelpath)
    print(f'  Plotting timesteps {timestepmin} to {timestepmax} '
          f'(t={args.timemin:.3f}d to {args.timemax:.3f}d)')

    linelabel = f'{modelname}'
    if not args.hidemodeltimerange:
        linelabel += f' at t={args.timemin:.2f}d to {args.timemax:.2f}d'

    if from_packets:
        # find any other packets files in the same directory
        packetsfiles_thismodel = sorted(
            glob.glob(str(Path(modelpath, 'packets00_*.out'))) +
            glob.glob(str(Path(modelpath, 'packets00_*.out.gz'))))
        if args.maxpacketfiles >= 0 and len(packetsfiles_thismodel) > args.maxpacketfiles:
            print(f'Using on the first {args.maxpacketfiles} packet files out of {len(packetsfiles_thismodel)}')
            packetsfiles_thismodel = packetsfiles_thismodel[:args.maxpacketfiles]
        spectrum = get_spectrum_from_packets(
            packetsfiles_thismodel, args.timemin, args.timemax, lambda_min=args.xmin, lambda_max=args.xmax,
            use_comovingframe=args.use_comovingframe)
        make_spectrum_stat_plot(spectrum, linelabel, Path(args.outputfile).parent, args)
    else:
        spectrum = get_spectrum(modelpath, timestepmin, timestepmax, fnufilterfunc=filterfunc)

    spectrum.query('@args.xmin <= lambda_angstroms and lambda_angstroms <= @args.xmax', inplace=True)

    at.spectra.print_integrated_flux(spectrum['f_lambda'], spectrum['lambda_angstroms'])

    if scale_to_peak:
        spectrum['f_lambda_scaled'] = spectrum['f_lambda'] / spectrum['f_lambda'].max() * scale_to_peak

        ycolumnname = 'f_lambda_scaled'
    else:
        ycolumnname = 'f_lambda'

    for index, axis in enumerate(axes):
        supxmin, supxmax = axis.get_xlim()
        spectrum.query(
        '@supxmin <= lambda_angstroms and lambda_angstroms <= @supxmax').plot(
            x='lambda_angstroms', y=ycolumnname, ax=axis, legend=None,
            label=linelabel if index == 0 else None, alpha=0.95, **plotkwargs)


def make_spectrum_plot(modelpaths, axes, filterfunc, args, scale_to_peak=None):
    """Plot reference spectra and ARTIS spectra."""
    plot_reference_spectra(axes, [], [], args, scale_to_peak=scale_to_peak, flambdafilterfunc=filterfunc)

    for index, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"ARTIS model '{modelname}' at path '{modelpath}'")
        plotkwargs = {}
        # plotkwargs['dashes'] = dashesList[index]
        # plotkwargs['dash_capstyle'] = dash_capstyleList[index]
        plotkwargs['linestyle'] = '--' if (int(index / 7) % 2) else '-'
        plotkwargs['linewidth'] = 1.5 - (0.2 * index)
        if index < 3:
            plotkwargs['color'] = ['orange', 'red', 'blue'][index]
        plot_artis_spectrum(axes, modelpath, args=args, scale_to_peak=scale_to_peak, from_packets=args.frompackets,
                            filterfunc=filterfunc, **plotkwargs)

    for axis in axes:
        axis.set_ylim(ymin=0.)
        if args.normalised:
            axis.set_ylim(ymax=1.25)
            axis.set_ylabel(r'Scaled F$_\lambda$')


def make_emissionabsorption_plot(modelpath, axis, filterfunc, args, scale_to_peak=None):
    """Plot the emission and absorption by ion for an ARTIS model."""
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

    contributions_sorted_reduced = at.spectra.sort_and_reduce_flux_contribution_list(
        contribution_list, args.maxseriescount, arraylambda_angstroms, fixedionlist=args.fixedionlist)

    plotobjectlabels = []
    plotobjects = []

    max_flambda_emission_total = max(
        [flambda if (args.xmin < lambda_ang < args.xmax) else -99.0
         for lambda_ang, flambda in zip(arraylambda_angstroms, array_flambda_emission_total)])

    scalefactor = (scale_to_peak / max_flambda_emission_total if scale_to_peak else 1.)

    plotobjectlabels.append('Net spectrum')
    line = axis.plot(arraylambda_angstroms, array_flambda_emission_total * scalefactor, linewidth=1.5, color='black', zorder=100)
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
        figsize=(8, 2 + nrows * 3), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if nrows == 1:
        axes = [axes]

    import scipy.signal
    if args.filtersavgol:
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

        if supxmax - supxmin < 11000:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))

    if args.showemission or args.showabsorption:
        if len(modelpaths) > 1:
            raise ValueError("ERROR: emission/absorption plot can only take one input model", modelpaths)

        defaultoutputfile = Path("plotspecemission_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf")

        plotobjects, plotobjectlabels = make_emissionabsorption_plot(
            modelpaths[0], axes[0], filterfunc, args, scale_to_peak=scale_to_peak)
    else:
        defaultoutputfile = Path("plotspec_{time_days_min:.0f}d_{time_days_max:.0f}d.pdf")

        make_spectrum_plot(modelpaths, axes, filterfunc, args, scale_to_peak=scale_to_peak)
        plotobjects, plotobjectlabels = axes[0].get_legend_handles_labels()

    axes[0].legend(plotobjects, plotobjectlabels, loc='upper right', handlelength=2,
                   frameon=False, numpoints=1)  # , prop={'size': args.legendfontsize}

    # plt.setp(plt.getp(axis, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(axis, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #    axis.spines[axis].set_linewidth(framewidth)

    for axis in axes:
        axis.set_xlabel('')
        # axis.xaxis.set_major_formatter(plt.NullFormatter())

    axes[-1].set_xlabel(r'Wavelength ($\AA$)')
    for axis in axes:
        axis.yaxis.set_major_formatter(at.ExponentLabelFormatter(axis.get_ylabel(), useMathText=True))

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
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
        timearray, args.timestep, args.timemin, args.timemax, args.timedays)

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

    parser.add_argument('--hidemodeltimerange', action='store_true',
                        help='Hide the "at t=x to yd" from the line labels')

    parser.add_argument('--normalised', action='store_true',
                        help='Normalise all spectra to their peak values')

    parser.add_argument('--use_comovingframe', action='store_true',
                        help='Use the time of packet escape to the surface (instead of a plane toward the observer)')

    parser.add_argument('-obsspec', action='append', dest='refspecfiles',
                        help='Also plot reference spectrum from this file')

    parser.add_argument('-fluxdistmpc', type=float,
                        help=('Plot flux at this distance in megaparsec. Default is the distance to '
                              'first reference spectrum if this is known, or otherwise 1 Mpc'))

    parser.add_argument('--notitle', action='store_true',
                        help='Suppress the top title from the plot')

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
