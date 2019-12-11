#!/usr/bin/env python3
"""Artistools - spectra related functions."""
import argparse
import json
import math
import multiprocessing
from collections import namedtuple
from functools import lru_cache
from functools import partial
from pathlib import Path
import os

import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml
from astropy import constants as const
from astropy import units as u
import re

import artistools as at
import artistools.packets

EMTYPECOLUMN = 'emissiontype'
# EMTYPECOLUMN = 'trueemissiontype'


def get_packets_with_emtype_onefile(lineindices, packetsfile):
    dfpackets = at.packets.readfile(packetsfile, usecols=[
        'type_id', 'e_cmf', 'e_rf', 'nu_rf', 'escape_type_id', 'escape_time',
        'em_posx', 'em_posy', 'em_posz', 'em_time',
        'posx', 'posy', 'posz', 'dirx', 'diry', 'dirz', 'emissiontype',
        'trueemissiontype', 'true_emission_velocity'],
        type='TYPE_ESCAPE', escape_type='TYPE_RPKT')

    dfpackets_selected = dfpackets.query(f'{EMTYPECOLUMN} in @lineindices', inplace=False)

    return dfpackets_selected


# @lru_cache(maxsize=8)
def get_packets_with_emtype(modelpath, lineindices, maxpacketfiles=None):
    packetsfiles = at.packets.get_packetsfilepaths(modelpath, maxpacketfiles=maxpacketfiles)
    nprocs_read = len(packetsfiles)
    assert nprocs_read > 0

    model, _ = at.get_modeldata(modelpath)
    # vmax = model.iloc[-1].velocity_outer * u.km / u.s

    processfile = partial(get_packets_with_emtype_onefile, lineindices)
    if at.enable_multiprocessing:
        pool = multiprocessing.Pool()
        arr_dfmatchingpackets = pool.imap_unordered(processfile, packetsfiles)
        pool.close()
        pool.join()
    else:
        arr_dfmatchingpackets = [processfile(f) for f in packetsfiles]

    dfmatchingpackets = pd.concat(arr_dfmatchingpackets)

    return dfmatchingpackets, nprocs_read


def calculate_timebinned_packet_sum(dfpackets, timearrayplusend):
    binned = pd.cut(dfpackets['t_arrive_d'], timearrayplusend, labels=False, include_lowest=True)

    binnedenergysums = np.zeros_like(timearrayplusend[:-1], dtype=np.float)
    for binindex, e_rf_sum in dfpackets.groupby(binned)['e_rf'].sum().iteritems():
        binnedenergysums[int(binindex)] = e_rf_sum

    return binnedenergysums


def get_line_fluxes_from_packets(modelpath, labelandlineindices, maxpacketfiles=None, arr_tstart=None, arr_tend=None):
    if arr_tstart is None:
        arr_tstart = at.get_timestep_times_float(modelpath, loc='start')
    if arr_tend is None:
        arr_tend = at.get_timestep_times_float(modelpath, loc='end')

    arr_timedelta = np.array(arr_tend) - np.array(arr_tstart)
    arr_tmid = (np.array(arr_tstart) + np.array(arr_tend)) / 2.

    model, _ = at.get_modeldata(modelpath)
    # vmax = model.iloc[-1].velocity_outer * u.km / u.s
    # betafactor = math.sqrt(1 - (vmax / const.c).decompose().value ** 2)

    timearrayplusend = np.concatenate([arr_tstart, [arr_tend[-1]]])

    dictlcdata = {'time': arr_tmid}
    dictlcdata.update({
       f'flux_{label}': np.zeros_like(arr_tstart, dtype=np.float) for label, _ in labelandlineindices})

    alllineindices = tuple(np.concatenate([l for _, l in labelandlineindices]))
    dfpackets, nprocs_read = get_packets_with_emtype(modelpath, alllineindices, maxpacketfiles=maxpacketfiles)

    normfactor = (1. / 4 / math.pi / (u.megaparsec.to('cm') ** 2) / nprocs_read / u.s.to('day'))
    for label, lineindices in labelandlineindices:
        dfpackets_selected = dfpackets.query(f'{EMTYPECOLUMN} in @lineindices', inplace=False)
        energysumsreduced = calculate_timebinned_packet_sum(dfpackets_selected, timearrayplusend)
        fluxdata = np.divide(energysumsreduced * normfactor, arr_timedelta)
        dictlcdata[f'flux_{label}'] = fluxdata

    lcdata = pd.DataFrame(dictlcdata)
    return lcdata


def get_labelandlineindices(modelpath):
    dflinelist = at.get_linelist(modelpath, returntype='dataframe')
    dflinelist.query('atomic_number == 26 and ionstage == 2', inplace=True)

    labelandlineindices = []
    for lambdamin, lambdamax, approxlambda in [(7150, 7160, 7155), (12470, 12670, 12570)]:
        dflinelistclosematches = dflinelist.query('@lambdamin < lambda_angstroms < @lambdamax')
        linelistindicies = dflinelistclosematches.index.values
        labelandlineindices.append([approxlambda, linelistindicies])
        # for index, line in lineclosematches.iterrows():
        #     print(line)

        print(f'{approxlambda} Å feature including {len(linelistindicies)} lines '
              f'[{dflinelistclosematches.lambda_angstroms.min():.1f} Å, '
              f'{dflinelistclosematches.lambda_angstroms.max():.1f} Å]')
    return labelandlineindices


def make_flux_ratio_plot(args):
    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, sharey=False,
        figsize=(args.figscale * at.figwidth, args.figscale * at.figwidth * (0.25 + nrows * 0.4)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if nrows == 1:
        axes = [axes]

    axis = axes[0]
    axis.set_yscale('log')
    # axis.set_ylabel(r'log$_1$$_0$ F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\mathrm{{\AA}}$]')
    axis.set_ylabel(r'F$_{12570\,\mathrm{\AA}}$ / F$_{7155\,\mathrm{\AA}}$')

    # axis.set_xlim(left=supxmin, right=supxmax)
    pd.set_option('display.max_rows', 3500)
    pd.set_option('display.width', 150)
    pd.options.display.max_rows = 500
    for seriesindex, (modelpath, modellabel, modelcolor) in enumerate(zip(args.modelpath, args.label, args.color)):
        if not modellabel:
            modellabel = at.get_model_name(modelpath)

        labelandlineindices = get_labelandlineindices(modelpath)

        dflcdata = get_line_fluxes_from_packets(modelpath, labelandlineindices, maxpacketfiles=args.maxpacketfiles,
                                                arr_tstart=args.timebins_tstart,
                                                arr_tend=args.timebins_tend)
        dflcdata.eval('fratio_12570_7155 = flux_12570 / flux_7155', inplace=True)
        # for row in dflcdata
        # print(dflcdata.time)
        print(dflcdata)

        axis.plot(dflcdata.time, dflcdata['fratio_12570_7155'], label=modellabel, marker='s', color=modelcolor)

        axis.set_ylim(ymin=0.05)
        axis.set_ylim(ymax=4.2)
        tmin = dflcdata.time.min()
        tmax = dflcdata.time.max()

    for ax in axes:
        arr_tdays = np.linspace(tmin, tmax, 3)
        arr_floersfit = [10 ** (0.0043 * timedays - 1.65) for timedays in arr_tdays]
        ax.plot(arr_tdays, arr_floersfit, color='black', label='Floers et al. (2019) best fit', lw=2.)
        # ax.xaxis.set_major_formatter(plt.NullFormatter())
        # if args.ymin is not None:
        #     ax.set_ylim(bottom=args.ymin)
        # if args.ymax is not None:
        #     ax.set_ylim(top=args.ymax)

        # if '{' in ax.get_ylabel():
        #     ax.yaxis.set_major_formatter(at.ExponentLabelFormatter(ax.get_ylabel(), useMathText=True, decimalplaces=1))

        # if args.hidexticklabels:
        #     ax.tick_params(axis='x', which='both',
        #                    # bottom=True, top=True,
        #                    labelbottom=False)
        ax.set_xlabel(r'Time [days]')
        ax.tick_params(which='both', direction='in')
        ax.legend(loc='upper right', frameon=False, handlelength=1, ncol=2, numpoints=1)

    defaultoutputfile = 'linefluxes.pdf'
    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile = args.outputfile / defaultoutputfile

    fig.savefig(args.outputfile, format='pdf')
    # plt.show()
    print(f'Saved {args.outputfile}')
    plt.close()


def make_emitting_regions_plot(args):
    # font = {'size': 16}
    # matplotlib.rc('font', **font)

    with open('floers_te_nne.json', encoding='utf-8') as data_file:
        floers_te_nne = json.loads(data_file.read())

    floers_times = np.array([float(t) for t in floers_te_nne.keys()])
    print(f'Floers data available for times: {floers_times}')
    # times_days = np.array(sorted([float(t) for t in floers_te_nne.keys()]))
    times_days = [199, 233, 284, 321, 357]
    print(f'Selected times: {times_days}')
    timebins_tstart = np.array([t - 20 for t in times_days])
    timebins_tend = np.array([t + 20 for t in times_days])

    nrows = len(timebins_tstart)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, sharey=False, sharex=False,
        figsize=(args.figscale * at.figwidth, args.figscale * at.figwidth * (0.25 + nrows * 0.7)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.2})

    if nrows == 1:
        axes = [axes]

    axes[-1].set_xlabel(r'log10(nne [cm$^{-3}$])')

    floers_keys = [t for t in floers_te_nne.keys()]  # strings, not floats
    floers_data = [floers_te_nne[t] for t in floers_keys]
    for axis, t_days in zip(axes, times_days):
        floersindex = np.abs(floers_times - t_days).argmin()
        axis.plot(floers_data[floersindex]['ne'], floers_data[floersindex]['temp'],
                  color='black', lw=2, label=f'Floers et al. (2019) {floers_keys[floersindex]}d')

    # axis.set_xlim(left=supxmin, right=supxmax)
    # pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 250)
    pd.options.display.max_rows = 500
    for seriesindex, (modelpath, modellabel, modelcolor) in enumerate(zip(args.modelpath, args.label, args.color)):
        if not modellabel:
            modellabel = at.get_model_name(modelpath)

        estimators = at.estimators.read_estimators(modelpath)
        modeldata, _ = at.get_modeldata(modelpath)
        allnonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                              if not estimators[(0, modelgridindex)]['emptycell']]

        labelandlineindices = get_labelandlineindices(modelpath)
        lineindices = np.concatenate([l for _, l in labelandlineindices])

        dfpackets, nprocs_read = get_packets_with_emtype(modelpath, lineindices, maxpacketfiles=args.maxpacketfiles)

        print(f'{len(dfpackets)} packets with matching {EMTYPECOLUMN}')

        if dfpackets.empty:
            continue

        for axis, tmid, tstart, tend in zip(axes, times_days, timebins_tstart, timebins_tend):
            dfpackets_intimerange = dfpackets.copy()
            dfpackets_intimerange.query('t_arrive_d >= @tstart and t_arrive_d <= @tend', inplace=True)

            if dfpackets_intimerange.empty:
                continue

            at.packets.calculate_emtimestep(dfpackets_intimerange, modelpath)
            at.packets.calculate_emvelocity(dfpackets_intimerange)

            # if EMTYPECOLUMN == 'trueemissiontype':
            if True:
                at.packets.calculate_emtrue_modelgridindex(dfpackets_intimerange, modelpath, allnonemptymgilist)
                emmgicol = 'emtrue_modelgridindex'
            else:
                at.packets.calculate_emmodelgridindex(dfpackets_intimerange, modelpath, allnonemptymgilist)
                emmgicol = 'em_modelgridindex'

            def em_lognne(packet):
                try:
                    return math.log10(estimators[(packet.em_timestep, packet[emmgicol])]['nne'])
                except KeyError:
                    return -9999

            dfpackets_intimerange['em_log10nne'] = dfpackets_intimerange.apply(em_lognne, axis=1)

            def em_Te(packet):
                try:
                    return estimators[(packet.em_timestep, packet[emmgicol])]['Te']
                except KeyError:
                    return -9999

            dfpackets_intimerange['em_Te'] = dfpackets_intimerange.apply(em_Te, axis=1)

            markers = ['+', 'x']
            for index, (featurelabel, lineindices) in enumerate(labelandlineindices):
                dfpackets_selected = dfpackets_intimerange.query(f'{EMTYPECOLUMN} in @lineindices', inplace=False)
                print(f'{len(dfpackets_selected)} matching packets in time range')
                linelabel = f'{modellabel} {featurelabel}' + r' $\mathrm{\AA}$'
                axis.plot(dfpackets_selected.em_log10nne, dfpackets_selected.em_Te, label=linelabel, ls='None',
                          marker=markers[index], alpha=0.9, color=modelcolor)

    axes[0].legend(loc='upper left', frameon=False, handlelength=1, ncol=1, numpoints=1, fontsize='small')

    for axis, t_days in zip(axes, floers_te_nne.keys()):
        axis.set_ylim(ymin=3000)
        axis.set_ylim(ymax=10000)
        axis.set_xlim(xmin=5., xmax=7.)

        axis.tick_params(which='both', direction='in')
        axis.set_ylabel(r'Electron Temperature [K]')

        axis.annotate(f'{float(t_days):.0f}d', xy=(0.98, 0.96), xycoords='axes fraction',
                      horizontalalignment='right', verticalalignment='top', fontsize=16)

    defaultoutputfile = 'emittingregions.pdf'
    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile = args.outputfile / defaultoutputfile

    fig.savefig(args.outputfile, format='pdf')
    # plt.show()
    print(f'Saved {args.outputfile}')
    plt.close()


def addargs(parser):

    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Paths to ARTIS folders with spec.out or packets files')

    parser.add_argument('-label', default=[], nargs='*',
                        help='List of series label overrides')

    parser.add_argument('-color', default=[f'C{i}' for i in range(10)], nargs='*',
                        help='List of line colors')

    parser.add_argument('-linestyle', default=[], nargs='*',
                        help='List of line styles')

    parser.add_argument('-linewidth', default=[], nargs='*',
                        help='List of line widths')

    parser.add_argument('-dashes', default=[], nargs='*',
                        help='Dashes property of lines')

    parser.add_argument('-maxpacketfiles', type=int, default=None,
                        help='Limit the number of packet files read')

    # parser.add_argument('-timemin', type=float,
    #                     help='Lower time in days to integrate spectrum')
    #
    # parser.add_argument('-timemax', type=float,
    #                     help='Upper time in days to integrate spectrum')
    #
    parser.add_argument('-xmin', type=int, default=50,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=450,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('-ymin', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-ymax', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-timebins_tstart', default=[], nargs='*', action='append',
                        help='Time bin start values in days')

    parser.add_argument('-timebins_tend', default=[], nargs='*', action='append',
                        help='Time bin end values in days')

    parser.add_argument('-figscale', type=float, default=1.8,
                        help='Scale factor for plot area. 1.0 is for single-column')

    parser.add_argument('--write_data', action='store_true',
                        help='Save data used to generate the plot in a CSV file')

    parser.add_argument('--plotemittingregions', action='store_true',
                        help='Plot conditions where flux line is emitted')

    parser.add_argument('-outputfile', '-o', action='store', dest='outputfile', type=Path,
                        help='path/filename for PDF file')


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

    args.modelpath = at.flatten_list(args.modelpath)

    at.trim_or_pad(len(args.modelpath), args.label, args.color)

    if args.plotemittingregions:
        make_emitting_regions_plot(args)
    else:
        make_flux_ratio_plot(args)


if __name__ == "__main__":
    main()
