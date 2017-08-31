#!/usr/bin/env python3
import argparse
import glob
# import math
import re
import os
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.ticker as ticker
import pandas as pd

import artistools as at

DEFAULTSPECPATH = '../example_run/spec.out'
defaultoutputfile = 'plotnonthermal_cell{0:03d}_timestep{1:03d}.pdf'


def ar_xs(energy_ev, ionpot_ev, A, B, C, D):
    u = energy_ev / ionpot_ev
    if u <= 1:
        return 0
    return 1e-14 * (A * (1 - 1 / u) + B * pow((1 - 1 / u), 2) + C * np.log(u) + D * np.log(u) / u) / (u * pow(ionpot_ev, 2))


def xs_fe2_old(energy):
    # AR1985
    shell_a = ar_xs(energy, 16.2, 90.0, -60.0, 0.2, -86)
    shell_b = ar_xs(energy, 17.5, 18.6, -5.9, 0.6, -9)
    shell_c = ar_xs(energy, 81, 69.9, -23.7, 9.5, -51.7)
    return shell_a + shell_b + shell_c


def get_arxs_array_shell(arr_enev, row):
    ar_xs_array = np.zeros(len(arr_enev))

    ionpot_ev, A, B, C, D = row.ionpot_ev, row.A, row.B, row.C, row.D
    for index, energy_ev in enumerate(arr_enev):
        u = energy_ev / ionpot_ev
        if u <= 1:
            ar_xs_array[index] = 0.
        else:
            ar_xs_array[index] = 1e-14 * (A * (1 - 1 / u) + B * pow((1 - 1 / u), 2) + C * np.log(u) + D * np.log(u) / u) / (u * pow(ionpot_ev, 2))

    return ar_xs_array


def get_arxs_array(arr_enev, dfcollion, Z, ionstage):
    ar_xs_array = np.zeros(len(arr_enev))
    dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage')
    print(dfcollion_thision)
    for index, row in dfcollion_thision.iterrows():
        ar_xs_array = np.add(ar_xs_array, np.array([ar_xs(energy_ev, row.ionpot_ev, row.A, row.B, row.C, row.D) for energy_ev in arr_enev]))
    return ar_xs_array


def read_colliondata():
    collionrow = namedtuple('collionrow', ['Z', 'nelec', 'n', 'l', 'ionpot_ev', 'A', 'B', 'C', 'D'])
    with open(os.path.join(at.PYDIR, 'data', 'collion.txt'), 'r') as collionfile:
        print(f'Collionfile: expecting {collionfile.readline().strip()} rows')
        dfcollion = pd.read_csv(
            collionfile, delim_whitespace=True, header=None, names=collionrow._fields)
    dfcollion.eval('ionstage = Z - nelec + 1', inplace=True)

    return dfcollion


def make_xs_plot(axis, nonthermaldata, timestep, outputfile, args):
    dfcollion = read_colliondata()

    arr_en = nonthermaldata['energy_ev'].unique()

    # arr_xs_old = [xs_fe2_old(en) for en in arr_en]
    # arr_xs_times_y = [xs_fe1(en) * y for en, y in zip(nonthermaldata['energy_ev'], nonthermaldata['y'])]

    axis.plot(arr_en, get_arxs_array(arr_en, dfcollion, 26, 2), linewidth=2.0, label='Fe II')
    axis.plot(arr_en, get_arxs_array(arr_en, dfcollion, 28, 2), linewidth=2.0, label='Ni II')

    axis.set_ylabel(r'cross section (cm2)')

    axis.legend(loc='upper center', handlelength=2, frameon=False, numpoints=1, prop={'size': 13})


def make_espec_plot(axis, nonthermaldata, timestep, outputfile, args):
    # ymax = max(nonthermaldata['y'])

    # nonthermaldata.plot(x='energy_ev', y='y', linewidth=1.5, ax=axis, color='blue', legend=False)
    axis.plot(nonthermaldata['energy_ev'], np.log10(nonthermaldata['y']), linewidth=2.0, color='blue')
    axis.set_ylabel(r'log [y (e$^-$ / cm$^2$ / s / eV)]')


def make_plot(nonthermaldata, timestep, outputfile, args):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    nplots = 1 if not args.xsplot else 2
    fig, axes = plt.subplots(nplots, 1, sharex=True, figsize=(6, 4 * nplots),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if nplots == 1:
        axes = [axes]

    if args.xsplot:
        make_xs_plot(axes[0], nonthermaldata, timestep, outputfile, args)

    make_espec_plot(axes[-1], nonthermaldata, timestep, outputfile, args)

    figure_title = f'Cell {args.modelgridindex} at Timestep {timestep}'
    time_days = float(at.get_timestep_time('spec.out', timestep))
    if time_days >= 0:
        figure_title += f' ({time_days:.2f}d)'
    axes[0].set_title(figure_title, fontsize=13)

    axes[-1].set_xlabel(r'Energy (eV)')
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))
    # axis.set_yscale("log", nonposy='clip')
    # axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    # axis.set_ylim(ymin=0.0, ymax=ymax)

    # axis.legend(loc='upper center', handlelength=2,
    #             frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')

    parser.add_argument('-xsplot', action='store_true', default=False,
                        help='Show the cross section plot')

    parser.add_argument('-timestep', '-ts', type=int, default=-1,
                        help='Timestep number to plot')

    parser.add_argument('-timestepmax', type=int, default=-1,
                        help='Make plots for all timesteps up to this timestep')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('-xmin', type=int, default=40,
                        help='Plot range: minimum energy in eV')

    parser.add_argument('-xmax', type=int, default=10000,
                        help='Plot range: maximum energy in eV')

    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='Filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    """Plot the electron energy distribution."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS non-thermal electron energy spectrum.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    if args.listtimesteps:
        at.showtimesteptimes('spec.out')
    else:
        nonthermaldata = None
        nonthermal_files = (
            glob.glob(os.path.join(args.modelpath, 'nonthermalspec_????.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, '*/nonthermalspec_????.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, 'nonthermalspec-*.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, 'nonthermalspec.out'), recursive=True))

        for nonthermal_file in nonthermal_files:
            filerank = int(re.findall('[0-9]+', os.path.basename(nonthermal_file))[-1])

            if filerank > args.modelgridindex:
                continue
            print(f'Loading {nonthermal_file}...')

            nonthermaldata_thisfile = pd.read_csv(nonthermal_file, delim_whitespace=True, error_bad_lines=False)
            nonthermaldata_thisfile.query('modelgridindex==@args.modelgridindex', inplace=True)
            if len(nonthermaldata_thisfile) > 0:
                if nonthermaldata is None:
                    nonthermaldata = nonthermaldata_thisfile.copy()
                else:
                    nonthermaldata.append(nonthermaldata_thisfile, ignore_index=True)

        if args.timestep < 0:
            timestepmin = max(nonthermaldata['timestep'])
        else:
            timestepmin = args.timestep

        if not args.timestepmax or args.timestepmax < 0:
            timestepmax = timestepmin + 1
        else:
            timestepmax = args.timestepmax

        list_timesteps = range(timestepmin, timestepmax)

        for timestep in list_timesteps:
            nonthermaldata_currenttimestep = nonthermaldata.query('timestep==@timestep')

            if len(nonthermaldata_currenttimestep) > 0:
                outputfile = args.outputfile.format(args.modelgridindex, timestep)
                print(f'Plotting timestep {timestep:d}')
                make_plot(nonthermaldata_currenttimestep, timestep, outputfile, args)
            else:
                print(f'No data for timestep {timestep:d}')


if __name__ == "__main__":
    main()
