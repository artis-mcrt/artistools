#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import warnings

import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import pandas as pd
from astropy import constants as const

import artistools as at

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# TODO: input one or more ARTIS folders instead of file names


def main():
    """
        Plot ARTIS spectra and (optionally) reference spectra
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS model spectra by finding spec.out files '
                    'in the current directory or subdirectories.')
    parser.add_argument('-i', action='append', default=[], dest='filepaths',
                        help='Path to spec.out or packets*.out file (may include wildcards such as * and **)')
    parser.add_argument('--emissionabsorption', default=False, action='store_true',
                        help='Show an emission/absorption plot')
    parser.add_argument('-maxseriescount', type=int, default=9,
                        help='Maximum number of plot series (ions/processes) for emission/absorption plot')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestepmin', type=int, default=20,
                        help='First or only included timestep')
    parser.add_argument('-timestepmax', type=int,
                        help='Last included timestep')
    parser.add_argument('-timemin', type=float,
                        help='Time in days')
    parser.add_argument('-timemax', type=float,
                        help='Last included timestep time in days')
    parser.add_argument('-xmin', type=int, default=2500,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=11000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('--normalised', default=False, action='store_true',
                        help='Normalise the spectra to their peak values')
    parser.add_argument('-obsspec', action='append', dest='refspecfiles',
                        help='Also plot reference spectrum from this file')
    parser.add_argument('-legendfontsize', type=int, default=8,
                        help='Font size of legend text')
    parser.add_argument('-o', action='store', dest='outputfile',
                        help='path/filename for PDF file')
    args = parser.parse_args()

    if not args.filepaths:
        if args.emissionabsorption:
            args.filepaths = ['emission*.out', '*/emission*.out']
        else:
            args.filepaths = ['spec.out', '*/spec.out']  # '**/spec.out'

    inputfiles = []
    for filepath in args.filepaths:
        inputfiles.extend(glob.glob(filepath, recursive=True))

    if not inputfiles:
        print('no input files found')
        sys.exit()

    if args.emissionabsorption:
        if len(inputfiles) > 1:
            print("ERROR: emission/absorption plot can only take one input model")
            sys.exit()
        else:
            if not args.outputfile:
                args.outputfile = "plotspecemission.pdf"
            make_plot(inputfiles, args)
    elif args.listtimesteps:
        at.showtimesteptimes(inputfiles[0])
    else:
        if not args.outputfile:
            args.outputfile = "plotspec.pdf"
        make_plot(inputfiles, args)


def plot_artis_spectra(axis, inputfiles, args, filterfunc=None):
    """
        Plot ARTIS emergent spectra
    """

    # dashesList = [(), (1.5, 2, 9, 2), (5, 1), (0.5, 2), (4, 2)]
    # dash_capstyleList = ['butt', 'butt', 'butt', 'round', 'butt']
    # colorlist = [(0, .8*158./255, 0.6*115./255), (204./255, 121./255, 167./255), (213./255, 94./255, 0.0)]
    # inputfiles.sort(key=lambda x: os.path.dirname(x))
    for index, filename in enumerate(inputfiles):
        plotkwargs = {}
        # plotkwargs['dashes'] = dashesList[index]
        # plotkwargs['dash_capstyle'] = dash_capstyleList[index]
        plotkwargs['linestyle'] = ['-', '--'][int(index / 7) % 2]
        plotkwargs['linewidth'] = 2.5 - (0.2 * index)
        at.spectra.plot_artis_spectrum(axis, filename, xmin=args.xmin, xmax=args.xmax, args=args, **plotkwargs)


def make_spectrum_plot(inputfiles, axis, filterfunc, args):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """
    at.spectra.plot_reference_spectra(axis, [], [], args, flambdafilterfunc=filterfunc)
    plot_artis_spectra(axis, inputfiles, args, filterfunc)

    if args.normalised:
        axis.set_ylim(ymin=-0.1, ymax=1.25)
        axis.set_ylabel(r'Scaled F$_\lambda$')


def make_emission_plot(emissionfilename, axis, filterfunc, args):
    maxion = 5  # must match sn3d.h value

    specfilename = os.path.join(os.path.dirname(emissionfilename), 'spec.out')
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    timearray = specdata.columns.values[1:]
    arraynu = specdata.loc[:, '0'].values
    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu

    (modelname, timestepmin, timestepmax,
     time_days_lower, time_days_upper) = at.get_model_name_times(
         specfilename, timearray, args.timestepmin, args.timestepmax, args.timemin, args.timemax)

    absorptionfilename = os.path.join(os.path.dirname(emissionfilename), 'absorption.out')
    contribution_list, maxyvalueglobal, array_flambda_emission_total = at.spectra.get_flux_contributions(
        emissionfilename, absorptionfilename, maxion, timearray, arraynu,
        filterfunc, args.xmin, args.xmax, timestepmin, timestepmax)

    at.spectra.print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    contributions_sorted_reduced = at.spectra.sort_and_reduce_flux_contribution_list(
        contribution_list, args.maxseriescount, arraylambda_angstroms)

    plotobjects = axis.stackplot(
        arraylambda_angstroms, [x.array_flambda_emission for x in contributions_sorted_reduced], linewidth=0)

    facecolors = [p.get_facecolor()[0] for p in plotobjects]

    plotobjects.extend(axis.stackplot(
        arraylambda_angstroms, [-x.array_flambda_absorption for x in contributions_sorted_reduced],
        colors=facecolors, linewidth=0))

    plotobjectlabels = list([x.linelabel for x in contributions_sorted_reduced])

    at.spectra.plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None,
                                      scale_to_peak=(maxyvalueglobal if args.normalised else None), linewidth=0.5)

    axis.axhline(color='white', linewidth=0.5)

    plotlabel = f't={time_days_lower:.2f}d to {time_days_upper:.2f}d\n{modelname}'
    axis.annotate(plotlabel, xy=(0.97, 0.03), xycoords='axes fraction',
                  horizontalalignment='right', verticalalignment='bottom', fontsize=9)

    # axis.set_ylim(ymin=-0.05 * maxyvalueglobal, ymax=maxyvalueglobal * 1.3)

    return plotobjects, plotobjectlabels


def make_plot(inputfiles, args):
    import matplotlib.ticker as ticker

    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\AA$]')

    # import scipy.signal
    #
    # def filterfunc(flambda):
    #     return scipy.signal.savgol_filter(flambda, 5, 3)
    filterfunc = None
    if args.emissionabsorption:
        plotobjects, plotobjectlabels = make_emission_plot(inputfiles[0], axis, filterfunc, args)
    else:
        make_spectrum_plot(inputfiles, axis, filterfunc, args)
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

    filenameout = args.outputfile
    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


if __name__ == "__main__":
    main()
