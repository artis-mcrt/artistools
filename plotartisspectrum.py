#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import warnings
from collections import namedtuple

import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from astropy import constants as const

import readartisfiles as af

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

fluxcontributiontuple = namedtuple(
    'fluxcontribution', 'maxyvalue linelabel array_flambda_emission array_flambda_absorption')

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
    af.addargs_timesteps(parser)
    af.addargs_spectrum(parser)
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
        af.showtimesteptimes(inputfiles[0])
    else:
        if not args.outputfile:
            args.outputfile = "plotspec.pdf"
        make_plot(inputfiles, args)


def get_flux_contributions(emissionfilename, absorptionfilename, elementlist, maxion,
                           timearray, arraynu, filterfunc, xmin, xmax, timestepmin, timestepmax):
    # this is much slower than it could be because of the order in which these data tables are accessed
    # TODO: change to use sequential access as much as possible
    print(f"Reading {emissionfilename}")
    emissiondata = np.loadtxt(emissionfilename)
    print(f"Reading {absorptionfilename}")
    absorptiondata = np.loadtxt(absorptionfilename)
    arraylambda = const.c.to('angstrom/s').value / arraynu

    nelements = len(elementlist)
    maxyvalueglobal = 0.0
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
                array_fnu_emission = af.stackspectra(
                    [(emissiondata[timestep::len(timearray), selectedcolumn],
                      af.get_timestep_time_delta(timestep, timearray))
                     for timestep in range(timestepmin, timestepmax + 1)])

                if selectedcolumn < nelements * maxion:  # bound-bound process
                    array_fnu_absorption = af.stackspectra(
                        [(absorptiondata[timestep::len(timearray), selectedcolumn],
                          af.get_timestep_time_delta(timestep, timearray))
                         for timestep in range(timestepmin, timestepmax + 1)])
                else:
                    array_fnu_absorption = np.zeros(len(array_fnu_emission))

                # best to use the filter on fnu (because it hopefully has regular sampling)
                if filterfunc:
                    print("Applying filter")
                    array_fnu_emission = filterfunc(array_fnu_emission)
                    if selectedcolumn <= nelements * maxion:
                        array_fnu_absorption = filterfunc(array_fnu_absorption)

                array_flambda_emission = array_fnu_emission * arraynu / arraylambda
                array_flambda_absorption = array_fnu_absorption * arraynu / arraylambda

                maxyvaluethisseries = max(
                    [array_flambda_emission[i] if (xmin < arraylambda[i] < xmax) else -99.0
                     for i in range(len(array_flambda_emission))])

                maxyvalueglobal = max(maxyvalueglobal, maxyvaluethisseries)

                if emissiontype != 'free-free':
                    linelabel = f'{af.elsymbols[elementlist.Z[element]]} {af.roman_numerals[ion_stage]} {emissiontype}'
                else:
                    linelabel = f'{emissiontype}'

                contribution_list.append(
                    fluxcontributiontuple(maxyvalue=maxyvaluethisseries, linelabel=linelabel,
                                          array_flambda_emission=array_flambda_emission,
                                          array_flambda_absorption=array_flambda_absorption))

    return contribution_list, maxyvalueglobal


def get_model_name_times(filename, timearray, args):
    timestepmin, timestepmax = af.get_minmax_timesteps(timearray, args)
    modelname = af.get_model_name(filename)

    time_days_lower = float(timearray[timestepmin])
    time_days_upper = float(timearray[timestepmax])

    print(f'Plotting {modelname} ({filename}) timesteps {timestepmin} to {timestepmax} '
          f'(t={time_days_lower}d to {time_days_upper}d)')

    return modelname, timestepmin, timestepmax, time_days_lower, time_days_upper


def plot_artis_spectra(axis, inputfiles, args, filterfunc=None):
    """
        Plot ARTIS emergent spectra
    """

    # dashesList = [(), (1.5, 2, 9, 2), (5, 1), (0.5, 2), (4, 2)]
    # dash_capstyleList = ['butt', 'butt', 'butt', 'round', 'butt']
    # colorlist = [(0, .8*158./255, 0.6*115./255), (204./255, 121./255, 167./255), (213./255, 94./255, 0.0)]
    # inputfiles.sort(key=lambda x: os.path.dirname(x))
    for index, filename in enumerate(inputfiles):
        from_packets = os.path.basename(filename).startswith('packets')

        if from_packets:
            specfilename = os.path.join(os.path.dirname(filename), 'spec.out')
        else:
            specfilename = filename

        (modelname, timestepmin, timestepmax,
         time_days_lower, time_days_upper) = get_model_name_times(filename, af.get_timestep_times(filename), args)

        linelabel = f'{modelname} at t={time_days_lower:.2f}d to {time_days_upper:.2f}d'

        if from_packets:
            # find any other packets files in the same directory
            packetsfiles_thismodel = glob.glob(os.path.join(os.path.dirname(filename), 'packets**.out'))
            print(packetsfiles_thismodel)
            spectrum = af.get_spectrum_from_packets(packetsfiles_thismodel, time_days_lower, time_days_upper,
                                                    lambda_min=args.xmin, lambda_max=args.xmax)
        else:
            spectrum = af.get_spectrum(specfilename, timestepmin, timestepmax, fnufilterfunc=filterfunc)

        maxyvaluethisseries = spectrum.query(
            '@args.xmin < lambda_angstroms and '
            'lambda_angstroms < @args.xmax')['f_lambda'].max()

        linestyle = ['-', '--'][int(index / 7) % 2]
        spectrum['f_lambda_scaled'] = (spectrum['f_lambda'] / maxyvaluethisseries)
        ycolumnname = 'f_lambda_scaled' if args.normalised else 'f_lambda'
        spectrum.plot(x='lambda_angstroms', y=ycolumnname, ax=axis,
                      linestyle=linestyle, linewidth=2.5 - (0.2 * index),
                      label=linelabel, alpha=0.95, color=None)  # colorlist[index % len(colorlist)]

        # dashes=dashesList[index], dash_capstyle=dash_capstyleList[index])


def make_emission_plot(emissionfilename, axis, filterfunc, args):
    elementlist = af.get_composition_data(os.path.join(os.path.dirname(emissionfilename), 'compositiondata.txt'))

    # print(f'nelements {len(elementlist)}')
    maxion = 5  # must match sn3d.h value

    specfilename = os.path.join(os.path.dirname(emissionfilename), 'spec.out')
    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    timearray = specdata.columns.values[1:]
    arraynu = specdata.iloc[:, 0].values

    (modelname, timestepmin, timestepmax,
     time_days_lower, time_days_upper) = get_model_name_times(specfilename, timearray, args)

    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu
    absorptionfilename = os.path.join(os.path.dirname(emissionfilename), 'absorption.out')
    contribution_list, maxyvalueglobal = get_flux_contributions(
        emissionfilename, absorptionfilename, elementlist, maxion, timearray, arraynu,
        filterfunc, args.xmin, args.xmax, timestepmin, timestepmax)
    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    contribution_list = sorted(contribution_list, key=lambda x: x.maxyvalue)
    remainder_sum = np.zeros(len(arraylambda_angstroms))
    remainder_sum_absorption = np.zeros(len(arraylambda_angstroms))
    for row in contribution_list[:- args.maxseriescount]:
        remainder_sum = np.add(remainder_sum, row.array_flambda_emission)
        remainder_sum_absorption = np.add(remainder_sum_absorption, row.array_flambda_absorption)

    contribution_list = list(reversed(contribution_list[- args.maxseriescount:]))
    contribution_list.append(fluxcontributiontuple(maxyvalue=0.0, linelabel='other',
                                                   array_flambda_emission=remainder_sum,
                                                   array_flambda_absorption=remainder_sum_absorption))

    plotobjects = axis.stackplot(arraylambda_angstroms, *[x.array_flambda_emission for x in contribution_list],
                                 linewidth=0)
    axis.stackplot(arraylambda_angstroms, *[-x.array_flambda_absorption for x in contribution_list], linewidth=0)
    plotobjectlabels = list([x.linelabel for x in contribution_list])

    af.plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None,
                              scale_to_peak=(maxyvalueglobal if args.normalised else None), linewidth=0.5)

    axis.axhline(color='white', linewidth=1.0)

    plotlabel = f'{modelname}\nt={time_days_lower:.2f}d to {time_days_upper:.2f}d'
    axis.annotate(plotlabel, xy=(0.05, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    # axis.set_ylim(ymin=-0.05 * maxyvalueglobal, ymax=maxyvalueglobal * 1.3)

    return plotobjects, plotobjectlabels


def make_spectrum_plot(inputfiles, axis, filterfunc, args):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """
    af.plot_reference_spectra(axis, [], [], args, flambdafilterfunc=filterfunc)
    plot_artis_spectra(axis, inputfiles, args, filterfunc)

    if args.normalised:
        axis.set_ylim(ymin=-0.1, ymax=1.25)
        axis.set_ylabel(r'Scaled F$_\lambda$')


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
