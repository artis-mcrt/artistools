#!/usr/bin/env python3
import argparse
import glob
import itertools
import os.path
import sys
import warnings

import matplotlib.pyplot as plt

import artistools as at

# import matplotlib.ticker as ticker


warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


def main(argsraw=None):
    """
        Plot ARTIS spectra and (optionally) reference spectra
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS model spectra by finding spec.out files '
                    'in the current directory or subdirectories.')
    parser.add_argument('modelpath', default=[], nargs='*',
                        help='Paths to ARTIS folders with spec.out or packets files'
                        ' (may include wildcards such as * and **)')
    parser.add_argument('--frompackets', default=False, action='store_true',
                        help='Read packets files directly instead of exspec results')
    parser.add_argument('--emissionabsorption', default=False, action='store_true',
                        help='Show an emission/absorption plot')
    parser.add_argument('-maxseriescount', type=int, default=9,
                        help='Maximum number of plot series (ions/processes) for emission/absorption plot')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', nargs='?',
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
    parser.add_argument('-obsspec', action='append', dest='refspecfiles',
                        help='Also plot reference spectrum from this file')
    parser.add_argument('-legendfontsize', type=int, default=8,
                        help='Font size of legend text')
    parser.add_argument('-o', action='store', dest='outputfile',
                        help='path/filename for PDF file')
    args = parser.parse_args(argsraw)

    if len(args.modelpath) == 0:
        args.modelpath = ['.', '*']

    # combined the results of applying wildcards on each input
    modelpaths = list(itertools.chain.from_iterable([glob.glob(x) for x in args.modelpath if os.path.isdir(x)]))

    if args.listtimesteps:
        at.showtimesteptimes(modelpaths[0])
    else:
        if args.emissionabsorption:
            if len(modelpaths) > 1:
                print("ERROR: emission/absorption plot can only take one input model")
                sys.exit()
            defaultoutputfile = "plotspecemission.pdf"
        else:
            defaultoutputfile = "plotspec.pdf"

        if not args.outputfile:
            args.outputfile = defaultoutputfile
        elif os.path.isdir(args.outputfile):
            args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

        make_plot(modelpaths, args)


def make_spectrum_plot(modelpaths, axis, filterfunc, args):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """
    at.spectra.plot_reference_spectra(axis, [], [], args, flambdafilterfunc=filterfunc)

    for index, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")
        plotkwargs = {}
        # plotkwargs['dashes'] = dashesList[index]
        # plotkwargs['dash_capstyle'] = dash_capstyleList[index]
        plotkwargs['linestyle'] = '--' if (int(index / 7) % 2) else '-'
        plotkwargs['linewidth'] = 2.5 - (0.2 * index)
        at.spectra.plot_artis_spectrum(axis, modelpath, args=args, from_packets=args.frompackets,
                                       filterfunc=filterfunc, **plotkwargs)

    if args.normalised:
        axis.set_ylim(ymin=-0.1, ymax=1.25)
        axis.set_ylabel(r'Scaled F$_\lambda$')


def make_emission_plot(modelpath, axis, filterfunc, args):
    from astropy import constants as const
    import pandas as pd
    maxion = 5  # must match sn3d.h value

    emissionfilename = os.path.join(modelpath, 'emissiontrue.out')
    if not os.path.exists(emissionfilename):
        emissionfilename = os.path.join(modelpath, 'emission.out')

    specfilename = os.path.join(modelpath, 'spec.out')
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    timearray = specdata.columns.values[1:]
    arraynu = specdata.loc[:, '0'].values
    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu

    (modelname, timestepmin, timestepmax,
     time_days_lower, time_days_upper) = at.get_model_name_times(
         specfilename, timearray, args.timestep, args.timemin, args.timemax)

    absorptionfilename = os.path.join(modelpath, 'absorption.out')
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

    axis.stackplot(
        arraylambda_angstroms, [-x.array_flambda_absorption for x in contributions_sorted_reduced],
        colors=facecolors, linewidth=0)

    plotobjectlabels = list([x.linelabel for x in contributions_sorted_reduced])

    at.spectra.plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None,
                                      scale_to_peak=(maxyvalueglobal if args.normalised else None), linewidth=0.5)

    axis.axhline(color='white', linewidth=0.5)

    plotlabel = f't={time_days_lower:.2f}d to {time_days_upper:.2f}d\n{modelname}'
    axis.annotate(plotlabel, xy=(0.97, 0.03), xycoords='axes fraction',
                  horizontalalignment='right', verticalalignment='bottom', fontsize=9)

    # axis.set_ylim(ymin=-0.05 * maxyvalueglobal, ymax=maxyvalueglobal * 1.3)

    return plotobjects, plotobjectlabels


def make_plot(modelpaths, args):
    import matplotlib.ticker as ticker

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

    filenameout = args.outputfile
    fig.savefig(filenameout, format='pdf')
    # plt.show()
    print(f'Saved {filenameout}')
    plt.close()


if __name__ == "__main__":
    main()
