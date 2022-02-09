#!/usr/bin/env python3
import argparse
import glob
import multiprocessing
import os.path
import sys

from astropy import constants as const
# from astropy import units as u

import artistools as at
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import pandas as pd

defaultoutputfile = 'plotmacroatom_cell{0:03d}_{1:03d}-{2:03d}.pdf'


def addargs(parser):
    parser.add_argument('--modelpath', nargs='?', default='',
                        help='Path to ARTIS folder')
    parser.add_argument('-timestep', type=int, default=10,
                        help='Timestep number to plot, or -1 for last')
    parser.add_argument('-timestepmax', type=int, default=-1,
                        help='Make plots for all timesteps up to this timestep')
    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')
    parser.add_argument('element', nargs='?', default='Fe',
                        help='Plotted element')
    parser.add_argument('-xmin', type=int, default=1000,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=15000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='Filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    """Plot the macroatom transitions."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Plot ARTIS macroatom transitions.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    try:
        atomic_number = next(Z for Z, elsymb in enumerate(at.elsymbols) if elsymb.lower() == args.element.lower())
    except StopIteration:
        print(f"Could not find element '{args.element}'")
        return

    timestepmin = args.timestep

    if not args.timestepmax or args.timestepmax < 0:
        timestepmax = timestepmin
    else:
        timestepmax = args.timestepmax

    input_files = (
        glob.glob(os.path.join(args.modelpath, 'macroatom_????.out*'), recursive=True) +
        glob.glob(os.path.join(args.modelpath, '*/macroatom_????.out*'), recursive=True))

    if not input_files:
        print("No macroatom files found")
        return 1
    else:
        dfall = read_files(input_files, args.modelgridindex, timestepmin, timestepmax, atomic_number)

    specfilename = os.path.join(args.modelpath, 'spec.out')

    if not os.path.isfile(specfilename):
        specfilename = os.path.join(args.modelpath, '../example_run/spec.out')

    if not os.path.isfile(specfilename):
        print(f'Could not find {specfilename}')
        return 1

    outputfile = args.outputfile.format(args.modelgridindex, timestepmin, timestepmax)
    make_plot(dfall, args.modelpath, specfilename, timestepmin, timestepmax, outputfile,
              xmin=args.xmin, xmax=args.xmax, modelgridindex=args.modelgridindex)

    return 0


def make_plot(dfmacroatom, modelpath, specfilename, timestepmin, timestepmax, outputfile,
              xmin, xmax, modelgridindex, nospec=False, normalised=False):

    time_days_min = at.get_timestep_time(modelpath, timestepmin)
    time_days_max = at.get_timestep_time(modelpath, timestepmax)

    print(f'Plotting {len(dfmacroatom)} transitions')

    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=(6, 6), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    axis.annotate(f'Timestep {timestepmin:d} to {timestepmax:d} (t={time_days_min} to '
                  f'{time_days_max})\nCell {modelgridindex:d}',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    lambda_cmf_in = const.c.to('angstrom/s').value / dfmacroatom['nu_cmf_in'].values
    lambda_cmf_out = const.c.to('angstrom/s').value / dfmacroatom['nu_cmf_out'].values
    # axis.scatter(lambda_cmf_in, lambda_cmf_out, s=1, alpha=0.5, edgecolor='none')
    axis.plot(lambda_cmf_in, lambda_cmf_out, linestyle='none', marker='o',  # alpha=0.5,
              markersize=2, markerfacecolor='red', markeredgewidth=0)
    axis.set_xlabel(r'Wavelength in ($\AA$)')
    axis.set_ylabel(r'Wavelength out ($\AA$)')
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(xmin, xmax)

    # axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 13})

    print(f'Saving to {outputfile:s}')
    fig.savefig(outputfile, format='pdf')
    plt.close()


def read_files(files, modelgridindex=None, timestepmin=None, timestepmax=None, atomic_number=None):
    dfall = None
    if not files:
        print("No files")
    else:
        for _, filepath in enumerate(files):
            print(f'Loading {filepath}...')

            df_thisfile = pd.read_csv(filepath, delim_whitespace=True)
            # df_thisfile[['modelgridindex', 'timestep']].apply(pd.to_numeric)
            if modelgridindex:
                df_thisfile.query('modelgridindex==@modelgridindex', inplace=True)
            if timestepmin is not None:
                df_thisfile.query('timestep>=@timestepmin', inplace=True)
            if timestepmax:
                df_thisfile.query('timestep<=@timestepmax', inplace=True)
            if atomic_number:
                df_thisfile.query('Z==@atomic_number', inplace=True)

            if df_thisfile is not None:
                if len(df_thisfile) > 0:
                    if dfall is None:
                        dfall = df_thisfile.copy()
                    else:
                        dfall = dfall.append(df_thisfile.copy(), ignore_index=True)

        if dfall is None or len(dfall) == 0:
            print("No data found")

    return dfall


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
