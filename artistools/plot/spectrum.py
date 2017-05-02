#!/usr/bin/env python3
import argparse
import glob
import itertools
import os.path
import sys
import warnings

import artistools as at
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

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


def plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None, scale_to_peak=None,
                           **plotkwargs):
    """
        Plot reference spectra listed in args.refspecfiles
    """
    if args.refspecfiles is not None:
        colorlist = ['black', '0.4']
        for index, filename in enumerate(args.refspecfiles):
            serieslabel = at.spectra.refspectralabels.get(filename, filename)

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

    print_integrated_flux(specdata.f_lambda, specdata.lambda_angstroms)

    if len(specdata) > 5000:
        # specdata = scipy.signal.resample(specdata, 10000)
        # specdata = specdata.iloc[::3, :].copy()
        specdata.query('index % 3 == 0', inplace=True)
        print(f"  downsamping to {len(specdata)} points")

    # clamp negative values to zero
    specdata['f_lambda'] = specdata['f_lambda'].apply(lambda x: max(0, x))

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

    lineplot = specdata.plot(x='lambda_angstroms', y=ycolumnname, ax=axis, label=serieslabel, zorder=-1, **plotkwargs)
    return mpatches.Patch(color=lineplot.get_lines()[0].get_color())


def plot_artis_spectrum(axis, modelpath, args, from_packets=False, filterfunc=None, **plotkwargs):
    specfilename = os.path.join(modelpath, 'spec.out')

    (modelname, timestepmin, timestepmax,
     time_days_lower, time_days_upper) = at.get_model_name_times(
         specfilename, at.get_timestep_times(specfilename),
         args.timestep, args.timemin, args.timemax)

    linelabel = f'{modelname} at t={time_days_lower:.2f}d to {time_days_upper:.2f}d'

    if from_packets:
        # find any other packets files in the same directory
        packetsfiles_thismodel = glob.glob(os.path.join(modelpath, 'packets**.out'))
        print(packetsfiles_thismodel)
        spectrum = at.spectra.get_spectrum_from_packets(
            packetsfiles_thismodel, time_days_lower, time_days_upper, lambda_min=args.xmin, lambda_max=args.xmax)
    else:
        spectrum = at.spectra.get_spectrum(specfilename, timestepmin, timestepmax, fnufilterfunc=filterfunc)

    spectrum.query('@args.xmin < lambda_angstroms and lambda_angstroms < @args.xmax', inplace=True)

    at.spectra.print_integrated_flux(spectrum['f_lambda'], spectrum['lambda_angstroms'])

    spectrum['f_lambda_scaled'] = spectrum['f_lambda'] / spectrum['f_lambda'].max()
    ycolumnname = 'f_lambda_scaled' if args.normalised else 'f_lambda'
    spectrum.plot(x='lambda_angstroms', y=ycolumnname, ax=axis,
                  label=linelabel, alpha=0.95, **plotkwargs)


def make_spectrum_plot(modelpaths, axis, filterfunc, args):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """
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
    from astropy import constants as const
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

    plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None,
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
