#!/usr/bin/env python3
import argparse
# import math
import os
import glob

import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import artistools as at

DEFAULTSPECPATH = '../example_run/spec.out'


def ar_xs(energy_ev, ionpot_ev, A, B, C, D):
    u = energy_ev / ionpot_ev
    if u <= 1:
        return 0
    return 1e-14 * (A * (1 - 1/u) + B * pow((1 - 1/u), 2) + C * np.log(u) + D * np.log(u) / u) / (u * pow(ionpot_ev, 2))


def xs_fe1(energy):
    shell_a = ar_xs(energy, 16.18, 17.4, -3.27, 0.16, -10.2)
    shell_b = ar_xs(energy, 24.83, 30.1, -38.8, 18.6, -45.7)
    shell_c = ar_xs(energy, 83.37, 115, -72.4, 9.57, -107)
    return shell_a + shell_b + shell_c


def xs_fe1_old(energy):
    shell_a = ar_xs(energy, 16.2, 90.0, -60.0, 0.2, -86)
    shell_b = ar_xs(energy, 17.5, 18.6, -5.9, 0.6, -9)
    shell_c = ar_xs(energy, 81, 69.9, -23.7, 9.5, -51.7)
    return shell_a + shell_b + shell_c


def main(argsraw=None):
    """
        Plot the electron energy distribution
    """
    defaultoutputfile = 'plotnonthermal_cell{0:03d}_timestep{1:03d}.pdf'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS non-thermal electron energy spectrum.')
    parser.add_argument('modelpath', nargs='?', default='',
                        help='Path to ARTIS folder')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
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
    parser.add_argument('-o', action='store', dest='outputfile', default=defaultoutputfile,
                        help='Filename for PDF file')
    args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    if args.listtimesteps:
        at.showtimesteptimes('spec.out')
    else:
        nonthermaldata = None
        nonthermal_files = (
            glob.glob(os.path.join(args.modelpath, 'nonthermalspec_????.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, 'nonthermalspec-*.out'), recursive=True) +
            glob.glob(os.path.join(args.modelpath, 'nonthermalspec.out'), recursive=True))

        for nonthermal_file in nonthermal_files:
            print(f'Loading {nonthermal_file}...')

            nonthermaldata_thisfile = pd.read_csv(nonthermal_file, delim_whitespace=True)
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
                print(f'Plotting timestep {timestep:d}')
                outputfile = args.outputfile.format(args.modelgridindex, timestep)
                make_plot(nonthermaldata_currenttimestep, timestep, outputfile, args)
            else:
                print(f'No data for timestep {timestep:d}')


def make_plot(nonthermaldata, timestep, outputfile, args):
    """
        Draw the bin edges, fitted field, and emergent spectrum
    """
    import numpy as np
    fig, axis = plt.subplots(1, 1, sharex=True, figsize=(6, 4),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    # ymax = max(nonthermaldata['y'])

    # nonthermaldata.plot(x='energy_ev', y='y', linewidth=1.5, ax=axis, color='blue', legend=False)
    axis.plot(nonthermaldata['energy_ev'], np.log10(nonthermaldata['y']), linewidth=2.0, color='blue')

    arr_xs = [xs_fe1(en) for en in nonthermaldata['energy_ev']]
    # arr_xs_old = [xs_fe1_old(en) for en in nonthermaldata['energy_ev']]
    arr_xs_times_y = [xs_fe1(en) * y for en, y in zip(nonthermaldata['energy_ev'], nonthermaldata['y'])]

    # axis.plot(nonthermaldata['energy_ev'], arr_xs_times_y, linewidth=2.0, color='blue')

    axis.annotate(f'Timestep {timestep:d}\nCell {args.modelgridindex:d}',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlabel(r'Energy (eV)')
    axis.set_ylabel(r'log [y (e$^-$ / cm$^2$ / s / eV)]')
    # axis.yaxis.set_minor_locator(ticker.MultipleLocator(base=0.1))
    # axis.set_yscale("log", nonposy='clip')
    # axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    # axis.set_ylim(ymin=0.0, ymax=ymax)

    # axis.legend(loc='upper center', handlelength=2,
    #             frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
