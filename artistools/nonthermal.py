#!/usr/bin/env python3
import argparse
# import glob
# import math
# import re
import os
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.ticker as ticker
import pandas as pd

import artistools as at

DEFAULTSPECPATH = '../example_run/spec.out'
defaultoutputfile = 'plotnonthermal_cell{0:03d}_timestep{1:03d}.pdf'


@lru_cache(maxsize=4)
def read_files(modelpath, timestep=-1, modelgridindex=-1):
    """Read ARTIS -thermal spectrum data into a pandas DataFrame."""
    nonthermaldata = pd.DataFrame()

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=modelgridindex)
    for folderpath in at.get_runfolders(modelpath, timestep=timestep):
        for mpirank in mpiranklist:
            nonthermalfile = f'nonthermalspec_{mpirank:04d}.out'
            filepath = Path(folderpath, nonthermalfile)
            if not filepath.is_file():
                filepath = Path(folderpath, nonthermalfile + '.gz')
                if not filepath.is_file():
                    print(f'Warning: Could not find {filepath.relative_to(modelpath.parent)}')
                    continue

            if modelgridindex > -1:
                filesize = Path(filepath).stat().st_size / 1024 / 1024
                print(f'Reading {Path(filepath).relative_to(modelpath.parent)} ({filesize:.2f} MiB)')

            nonthermaldata_thisfile = pd.read_csv(filepath, delim_whitespace=True, error_bad_lines=False)
            # radfielddata_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)

            if timestep >= 0:
                nonthermaldata_thisfile.query('timestep==@timestep', inplace=True)

            if modelgridindex >= 0:
                nonthermaldata_thisfile.query('modelgridindex==@modelgridindex', inplace=True)

            if not nonthermaldata_thisfile.empty:
                if timestep >= 0 and modelgridindex >= 0:
                    return nonthermaldata_thisfile
                else:
                    nonthermaldata = nonthermaldata.append(nonthermaldata_thisfile.copy(), ignore_index=True)

    return nonthermaldata


def ar_xs(energy_ev, ionpot_ev, A, B, C, D):
    u = energy_ev / ionpot_ev
    if u <= 1:
        return 0
    return 1e-14 * (
        A * (1 - 1 / u) + B * pow((1 - 1 / u), 2) + C * np.log(u) + D * np.log(u) / u) / (u * pow(ionpot_ev, 2))


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
            ar_xs_array[index] = (1e-14 * (A * (1 - 1 / u) + B * pow((1 - 1 / u), 2) +
                                           C * np.log(u) + D * np.log(u) / u) / (u * pow(ionpot_ev, 2)))

    return ar_xs_array


def get_arxs_array(arr_enev, dfcollion, Z, ionstage):
    ar_xs_array = np.zeros(len(arr_enev))
    dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage')
    print(dfcollion_thision)
    for index, row in dfcollion_thision.iterrows():
        ar_xs_array = np.add(ar_xs_array, np.array(
            [ar_xs(energy_ev, row.ionpot_ev, row.A, row.B, row.C, row.D) for energy_ev in arr_enev]))
    return ar_xs_array


def read_colliondata(collionfilename='collion.txt'):
    collionrow = namedtuple('collionrow', ['Z', 'nelec', 'n', 'l', 'ionpot_ev', 'A', 'B', 'C', 'D'])
    with open(os.path.join(at.PYDIR, 'data', collionfilename), 'r') as collionfile:
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

    if not args.nolegend:
        axis.legend(loc='upper center', handlelength=2, frameon=False, numpoints=1, prop={'size': 13})


def make_plot(modelpaths, args):
              # nonthermaldata, modelpath, modelgridindex, timestep, outputfile, args):
    """Draw the bin edges, fitted field, and emergent spectrum."""
    nplots = 1 if not args.xsplot else 2
    fig, axes = plt.subplots(nrows=nplots, ncols=1, sharex=True,
                             figsize=(args.figscale * at.figwidth, args.figscale * at.figwidth * 0.7 * nplots),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if nplots == 1:
        axes = [axes]

    if args.kf1992spec:
        kf92spec = pd.read_csv(
            Path(modelpath, 'KF1992spec-fig1.txt'),
            header=None, names=['e_kev', 'log10_y'])
        kf92spec['energy_ev'] = kf92spec['e_kev'] * 1000.
        kf92spec.eval('y = 10 ** log10_y', inplace=True)
        axes[0].plot(kf92spec['energy_ev'], kf92spec['log10_y'],
                     linewidth=2.0, color='red', label='Kozma & Fransson (1992)')

    for index, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        if args.velocity >= 0.:
            modelgridindex = at.get_closest_cell(modelpath, args.velocity)
        else:
            modelgridindex = args.modelgridindex

        if args.timedays:
            timestep = at.get_closest_timestep(modelpath, args.timedays)
        else:
            timestep = args.timestep

        nonthermaldata = read_files(
            modelpath=Path(modelpath),
            modelgridindex=modelgridindex, timestep=timestep)

        if nonthermaldata.empty:
            print(f'No data for timestep {timestep:d}')
            continue

        if index < len(args.modellabels):
            model_label = args.modellabels[index]
        else:
            model_label = f'{modelname} cell {modelgridindex} at timestep {timestep}'
            try:
                time_days = float(at.get_timestep_time('.', timestep))
            except FileNotFoundError:
                time_days = 0
            else:
                model_label += f' ({time_days:.2f}d)'

        outputfile = str(args.outputfile).format(modelgridindex, timestep)
        print(f'Plotting timestep {timestep:d}')
        # ymax = max(nonthermaldata['y'])

        # nonthermaldata.plot(x='energy_ev', y='y', linewidth=1.5, ax=axis, color='blue', legend=False)
        axes[0].plot((nonthermaldata['energy_ev']), np.log10(nonthermaldata['y']), label=model_label,
                     linewidth=0.0, marker='x', color='black' if index == 0 else None, alpha=0.95)
        axes[0].set_ylabel(r'log [y (e$^-$ / cm$^2$ / s / eV)]')

        if args.xsplot:
            make_xs_plot(axes[1], nonthermaldata, timestep, outputfile, args)

    if not args.nolegend:
        axes[0].legend(loc='best', handlelength=2, frameon=False, numpoints=1)

    axes[-1].set_xlabel(r'Energy (eV)')
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))
    # axis.set_yscale("log", nonposy='clip')
    for ax in axes:
        if args.xmin is not None:
            ax.set_xlim(xmin=args.xmin)
        if args.xmax:
            ax.set_xlim(xmax=args.xmax)
    # axis.set_ylim(ymin=0.0, ymax=ymax)

    # axis.legend(loc='upper center', handlelength=2,
    #             frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Paths to ARTIS folders with spec.out or packets files')

    parser.add_argument('-modellabels', default=[], nargs='*',
                        help='Model name overrides')

    parser.add_argument('-listtimesteps', action='store_true',
                        help='Show the times at each timestep')

    parser.add_argument('-xsplot', action='store_true',
                        help='Show the cross section plot')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', type=int, default=-1,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('-velocity', '-v', type=float, default=-1,
                        help='Specify cell by velocity')

    parser.add_argument('-xmin', type=float, default=0.,
                        help='Plot range: minimum energy in eV')

    parser.add_argument('-xmax', type=float,
                        help='Plot range: maximum energy in eV')

    parser.add_argument('--nolegend', action='store_true',
                        help='Suppress the legend from the plot')

    parser.add_argument('--kf1992spec', action='store_true',
                        help='Show the pure-oxygen result form Figure 1 of Kozma & Fransson 1992')

    parser.add_argument('-figscale', type=float, default=1.,
                        help='Scale factor for plot area. 1.0 is for single-column')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path,
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

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    if args.listtimesteps:
        at.showtimesteptimes()
    else:
        make_plot(modelpaths, args)


if __name__ == "__main__":
    main()
