#!/usr/bin/env python3
import argparse
import glob
import math
import numpy as np
import os
import sys
import warnings

import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy import constants as const
from collections import namedtuple

import readartisfiles as af

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

fluxcontributiontuple = namedtuple(
    'fluxcontribution', 'maxyvalue linelabel array_flambda_emission array_flambda_absorption')


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
                args.outputfile = "plotemission.pdf"
            make_plot(inputfiles, args)
    elif args.listtimesteps:
        af.showtimesteptimes(inputfiles[0])
    else:
        if not args.outputfile:
            args.outputfile = "plotspec.pdf"
        make_plot(inputfiles, args)


def get_flux_contributions(emissionfilename, absorptionfilename, elementlist, maxion,
                           timearray, arraynu, args, timestepmin, timestepmax):
    # this is much slower than it could be because of the order in which these data tables are accessed
    # TODO: change to use sequential access as much as possible
    print(f"Reading {emissionfilename}")
    emissiondata = np.loadtxt(emissionfilename)
    print(f"Reading {absorptionfilename}")
    absorptiondata = np.loadtxt(absorptionfilename)
    c = const.c.to('m/s').value
    arraylambda = c / arraynu

    nelements = len(elementlist)
    maxyvalueglobal = 0.0
    contribution_list = []
    for element in range(nelements):
        nions = elementlist.nions[element]

        # nions = elementlist.iloc[element].uppermost_ionstage - elementlist.iloc[element].lowermost_ionstage + 1
        for ion in range(nions):
            ion_stage = ion + elementlist.lowermost_ionstage[element]
            ionserieslist = []

            ionserieslist.append((2 * nelements * maxion, 'free-free'))

            ionserieslist.append((element * maxion + ion, 'bound-bound'))

            ionserieslist.append((nelements * maxion + element * maxion + ion, 'bound-free'))

            for (selectedcolumn, emissiontype) in ionserieslist:
                array_fnu_emission = emissiondata[timestepmin::len(timearray), selectedcolumn]

                for timeindex in range(timestepmin + 1, timestepmax + 1):
                    array_fnu_emission += emissiondata[timeindex::len(timearray), selectedcolumn]

                if selectedcolumn < nelements * maxion:
                    array_fnu_absorption = absorptiondata[timestepmin::len(timearray), selectedcolumn]

                    for timeindex in range(timestepmin + 1, timestepmax + 1):
                        array_fnu_absorption += absorptiondata[timeindex::len(timearray), selectedcolumn]

                # rough normalisation for stacked timesteps. replace with dividing by time
                array_fnu_emission = array_fnu_emission / (timestepmax - timestepmin + 1)

                # best to use the filter on this list (because it hopefully has
                # regular sampling)
                array_fnu_emission = scipy.signal.savgol_filter(array_fnu_emission, 5, 2)
                array_flambda_emission = array_fnu_emission * (arraynu ** 2) / c

                if selectedcolumn <= nelements * maxion:
                    array_fnu_absorption = array_fnu_absorption / (timestepmax - timestepmin + 1)
                    array_fnu_absorption = scipy.signal.savgol_filter(array_fnu_absorption, 5, 2)
                    array_flambda_absorption = array_fnu_absorption * (arraynu ** 2) / c
                else:
                    array_flambda_absorption = np.zeros(len(array_fnu_emission))

                maxyvaluethisseries = max(
                    [array_flambda_emission[i] if (args.xmin < (1e10 * arraylambda[i]) < args.xmax) else -99.0
                     for i in range(len(array_flambda_emission))])

                maxyvalueglobal = max(maxyvalueglobal, maxyvaluethisseries)

                linelabel = ''
                if emissiontype != 'free-free':
                    linelabel += f'{af.elsymbols[elementlist.Z[element]]} {af.roman_numerals[ion_stage]} '
                linelabel += f'{emissiontype}'

                # if linelabel.startswith('Fe ') or linelabel.endswith("-free"):
                #     continue
                # contribution_list.append([maxyvaluethisseries, linelabel, array_flambda, array_flambda_absorption])
                contribution_list.append(fluxcontributiontuple(maxyvalue=maxyvaluethisseries, linelabel=linelabel,
                                                               array_flambda_emission=array_flambda_emission,
                                                               array_flambda_absorption=array_flambda_absorption))

    return contribution_list, maxyvalueglobal


def make_emission_plot(emissionfilename, axis, args):
    elementlist = af.get_composition_data(os.path.join(os.path.dirname(emissionfilename), 'compositiondata.txt'))

    # print(f'nelements {len(elementlist)}')
    maxion = 5  # must match sn3d.h value

    specfilename = os.path.join(os.path.dirname(emissionfilename), 'spec.out')
    specdata = np.loadtxt(specfilename)

    modelname = af.get_model_name(emissionfilename)

    timestepmin, timestepmax = af.get_minmax_timesteps(specfilename, args)

    time_in_days_lower = math.floor(float(af.get_timestep_time(specfilename, timestepmin)))
    plotlabel = f'{modelname}\nt={time_in_days_lower}d'

    if timestepmax > timestepmin:
        time_in_days_upper = math.floor(float(af.get_timestep_time(specfilename, timestepmax)))
        plotlabel += f' to {time_in_days_upper}d'
        print(f'Plotting {modelname} timesteps {timestepmin} to {timestepmax} (t={time_in_days_lower}d'
              f' to {time_in_days_upper}d)')
    else:
        print(f'Plotting {modelname} timestep {timestepmin} (t={time_in_days_lower}d)')

    timearray = specdata[0, 1:]
    arraynu = specdata[1:, 0]
    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu
    absorptionfilename = os.path.join(os.path.dirname(emissionfilename), 'absorption.out')
    contribution_list, maxyvalueglobal = get_flux_contributions(
        emissionfilename, absorptionfilename, elementlist, maxion, timearray, arraynu, args, timestepmin, timestepmax)
    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    contribution_list = sorted(contribution_list, key=lambda x: x.maxyvalue)
    remainder_sum = np.zeros(len(arraylambda_angstroms))
    remainder_sum_absorption = np.zeros(len(arraylambda_angstroms))
    for row in contribution_list[:- args.maxseriescount]:
        remainder_sum = np.add(remainder_sum, row[3])
        remainder_sum_absorption = np.add(remainder_sum_absorption, row[3])

    contribution_list = list(reversed(contribution_list[- args.maxseriescount:]))
    contribution_list.append(fluxcontributiontuple(maxyvalue=0.0, linelabel='other',
                                                   array_flambda_emission=remainder_sum,
                                                   array_flambda_absorption=remainder_sum_absorption))

    plotobjects = axis.stackplot(arraylambda_angstroms, *[x.array_flambda_emission for x in contribution_list],
                                 linewidth=0)
    axis.stackplot(arraylambda_angstroms, *[-x.array_flambda_absorption for x in contribution_list], linewidth=0)
    plotobjectlabels = list([x.linelabel for x in contribution_list])

    af.plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, flambdafilterfunc=None,
                              scale_to_peak=(maxyvalueglobal if args.normalised else None), lw=0.5)

    axis.axhline(color='white', lw=1.0)

    axis.annotate(plotlabel, xy=(0.05, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    # axis.set_ylim(ymin=-0.05 * maxyvalueglobal, ymax=maxyvalueglobal * 1.3)
    # axis.set_ylim(ymin=-0.1, ymax=1.1)

    axis.legend(plotobjects, plotobjectlabels, loc='upper right', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 9})

    axis.set_ylabel(r'F$_\lambda$')


def make_spectrum_plot(inputfiles, axis, args):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """

    # import scipy.signal
    #
    # def filterfunc(flambda):
    #     return scipy.signal.savgol_filter(flambda, 5, 3)
    filterfunc = None
    af.plot_reference_spectra(axis, [], [], args, flambdafilterfunc=filterfunc)
    plot_artis_spectra(axis, inputfiles, args)

    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    if args.normalised:
        axis.set_ylim(ymin=-0.1, ymax=1.25)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': args.legendfontsize})

    if args.normalised:
        axis.set_ylabel(r'Scaled F$_\lambda$')
    else:
        axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\AA$]')



def make_plot(inputfiles, args):
    import matplotlib.ticker as ticker

    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if args.emissionabsorption:
        make_emission_plot(inputfiles[0], axis, args)
    else:
        make_spectrum_plot(inputfiles, axis, args)

    # plt.setp(plt.getp(axis, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(axis, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #    axis.spines[axis].set_linewidth(framewidth)

    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))

    filenameout = args.outputfile
    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def plot_artis_spectra(axis, inputfiles, args):
    """
        Plot ARTIS emergent spectra
    """

    # dashesList = [(), (1.5, 2, 9, 2), (5, 1), (0.5, 2), (4, 2)]
    # dash_capstyleList = ['butt', 'butt', 'butt', 'round', 'butt']
    # colorlist = [(0, .8*158./255, 0.6*115./255), (204./255, 121./255, 167./255), (213./255, 94./255, 0.0)]
    # inputfiles.sort(key=lambda x: os.path.dirname(x))
    for index, filename in enumerate(inputfiles):
        modelname = af.get_model_name(filename)

        from_packets = os.path.basename(filename).startswith('packets')

        if from_packets:
            specfilename = os.path.join(os.path.dirname(filename), 'spec.out')
        else:
            specfilename = filename

        timestepmin, timestepmax = af.get_minmax_timesteps(specfilename, args)

        time_days_lower = float(af.get_timestep_time(specfilename, timestepmin))
        linelabel = f'{modelname} at t={time_days_lower:.2f}d'

        if timestepmax > timestepmin:
            time_days_upper = float(af.get_timestep_time(specfilename, timestepmax))
            linelabel += f' to {time_days_upper:.2f}d'
            print(f'Plotting {modelname} ({filename}) timesteps {timestepmin} to {timestepmax} (t={time_days_lower}d'
                  f' to {time_days_upper}d)')
        else:
            print(f'Plotting {modelname} timestep {timestepmin} (t={time_days_lower}d)')

        if from_packets:
            # find any other packets files in the same directory
            packetsfiles_thismodel = glob.glob(os.path.join(os.path.dirname(filename), 'packets**.out'))
            print(packetsfiles_thismodel)
            spectrum = af.get_spectrum_from_packets(packetsfiles_thismodel, time_days_lower, time_days_upper,
                                                    lambda_min=args.xmin, lambda_max=args.xmax)
        else:
            # def filterfunc(arrayfnu):
            #     from scipy.signal import savgol_filter
            #     return savgol_filter(arrayfnu, 5, 2)

            spectrum = af.get_spectrum(specfilename, timestepmin, timestepmax,)
            #                          fnufilterfunc=filterfunc)

        maxyvaluethisseries = spectrum.query(
            '@args.xmin < lambda_angstroms and '
            'lambda_angstroms < @args.xmax')['f_lambda'].max()

        linestyle = ['-', '--'][int(index / 7) % 2]
        spectrum['f_lambda_scaled'] = (spectrum['f_lambda'] / maxyvaluethisseries)
        ycolumnname = 'f_lambda_scaled' if args.normalised else 'f_lambda'
        spectrum.plot(x='lambda_angstroms', y=ycolumnname, ax=axis,
                      linestyle=linestyle, lw=2.5 - (0.2 * index),
                      label=linelabel, alpha=0.95, color=None)  # colorlist[index % len(colorlist)]
        # dashes=dashesList[index], dash_capstyle=dash_capstyleList[index])


if __name__ == "__main__":
    main()
