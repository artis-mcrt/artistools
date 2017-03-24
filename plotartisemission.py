#!/usr/bin/env python3
import argparse
import glob
import math
import os
import sys
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.signal
from astropy import constants as const
from collections import namedtuple

import readartisfiles as af

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# colorlist = ['black',(0.0,0.5,0.7),(0.35,0.7,1.0),(0.9,0.2,0.0),
#             (0.9,0.6,0.0),(0.0,0.6,0.5),(0.8,0.5,1.0),(0.95,0.9,0.25)]
colorlist = [(0.0, 0.5, 0.7), (0.9, 0.2, 0.0), (0.9, 0.6, 0.0),
             (0.0, 0.6, 0.5), (0.8, 0.5, 1.0), (0.95, 0.9, 0.25)]

fluxcontributiontuple = namedtuple(
    'fluxcontribution', 'maxyvalue linelabel array_flambda_emission array_flambda_absorption')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS emission spectrum')
    parser.add_argument('-filepath', action='store', default='**/emission*.out',
                        help='Path to emission.out file (may include wildcards such as * and **)')
    af.addargs_timesteps(parser)
    af.addargs_spectrum(parser)
    parser.add_argument('-maxseriescount', type=int, default=9,
                        help='Maximum number of plot series (ions/processes)')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotemission.pdf',
                        help='path/filename for PDF file')
    args = parser.parse_args()

    emissionfiles = glob.glob(args.filepath, recursive=True)

    if not emissionfiles:
        print('no emission.out files found')
        sys.exit()
    emissionfilename = emissionfiles[0]
    if args.listtimesteps:
        af.showtimesteptimes(os.path.join(os.path.dirname(emissionfilename), 'spec.out'))
    else:
        make_plot(emissionfilename, args)


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


def plot_reference_spectra(axis, plotobjects, plotobjectlabels, args, scale_to_peak=None):
    if args.refspecfiles is not None:
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        refspeccolorlist = ['0.0', '0.8']
        refspectra = [(fn, af.refspectralabels.get(fn, fn), c) for fn, c in zip(args.refspecfiles, refspeccolorlist)]

        for (filename, serieslabel, linecolor) in refspectra:
            specdata = np.loadtxt(os.path.join(scriptdir, 'spectra', filename))

            if len(specdata) > 5000:
                # specdata = scipy.signal.resample(specdata, 10000)
                specdata = specdata[::3]

            specdata[specdata[:, 1] < 0] = 0

            specdata = specdata[(specdata[:, 0] > args.xmin) & (specdata[:, 0] < args.xmax)]
            print(f"'{serieslabel}' has {len(specdata)} points")
            obsxvalues = specdata[:, 0]
            obsyvalues = specdata[:, 1]
            if scale_to_peak:
                obsyvalues *= scale_to_peak / max(obsyvalues)

            # obsyvalues = scipy.signal.savgol_filter(obsyvalues, 5, 3)
            axis.plot(obsxvalues, obsyvalues, lw=0.5, zorder=-1, color=linecolor)
            plotobjects.append(mpatches.Patch(color=linecolor))
            plotobjectlabels.append(serieslabel)


def make_plot(emissionfilename, args):
    elementlist = af.get_composition_data(os.path.join(os.path.dirname(emissionfilename), 'compositiondata.txt'))

    print(f'nelements {len(elementlist)}')
    maxion = 5  # must match sn3d.h value

    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    specfilename = os.path.join(os.path.dirname(emissionfilename), 'spec.out')
    specdata = np.loadtxt(specfilename)

    try:
        plotlabelfile = os.path.join(os.path.dirname(emissionfilename), 'plotlabel.txt')
        modelname = open(plotlabelfile, mode='r').readline().strip()
    except FileNotFoundError:
        modelname = os.path.dirname(emissionfilename)
        if not modelname:
            # use the current directory name
            modelname = os.path.split(os.path.dirname(os.path.abspath(emissionfilename)))[1]

    timestepmin, timestepmax = af.get_minmax_timesteps(specfilename, args)

    time_in_days_lower = math.floor(float(af.get_timestep_time(specfilename, timestepmin)))
    plotlabel = f'{modelname}\nt={time_in_days_lower}d'

    if timestepmax > timestepmin:
        time_in_days_upper = math.floor(float(af.get_timestep_time(specfilename, timestepmax)))
        plotlabel += f' to {time_in_days_upper}d'
        print(f'Plotting {specfilename} timesteps {timestepmin} to {timestepmax} (t={time_in_days_lower}d'
              f' to {time_in_days_upper}d)')
    else:
        print(f'Plotting {specfilename} timestep {timestepmin} (t={time_in_days_lower}d)')


    timearray = specdata[0, 1:]
    arraynu = specdata[1:, 0]
    arraylambda = const.c.to('m/s').value / arraynu
    absorptionfilename = os.path.join(os.path.dirname(emissionfilename), 'absorption.out')
    contribution_list, maxyvalueglobal = get_flux_contributions(
        emissionfilename, absorptionfilename, elementlist, maxion, timearray, arraynu, args, timestepmin, timestepmax)
    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    contribution_list = sorted(contribution_list, key=lambda x: x.maxyvalue)
    remainder_sum = np.zeros(len(arraylambda))
    remainder_sum_absorption = np.zeros(len(arraylambda))
    for row in contribution_list[:- args.maxseriescount]:
        remainder_sum = np.add(remainder_sum, row[3])
        remainder_sum_absorption = np.add(remainder_sum_absorption, row[3])

    contribution_list = list(reversed(contribution_list[- args.maxseriescount:]))
    contribution_list.append(fluxcontributiontuple(maxyvalue=0.0, linelabel='other',
                                                   array_flambda_emission=remainder_sum,
                                                   array_flambda_absorption=remainder_sum_absorption))

    plotobjects = axis.stackplot(1e10 * arraylambda, *[x.array_flambda_emission for x in contribution_list],
                                 linewidth=0)
    plotobjects = axis.stackplot(1e10 * arraylambda, *[-x.array_flambda_absorption for x in contribution_list],
                                 linewidth=0)
    plotobjectlabels = list([x.linelabel for x in contribution_list])

    plot_reference_spectra(axis, plotobjects, plotobjectlabels, args,
                           scale_to_peak=(maxyvalueglobal if args.normalised else None))

    axis.axhline(color='white', lw=1.0)

    axis.annotate(plotlabel, xy=(0.05, 0.96), xycoords='axes fraction',
                  horizontalalignment='left', verticalalignment='top', fontsize=8)

    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    # axis.set_ylim(ymin=-0.05 * maxyvalueglobal, ymax=maxyvalueglobal * 1.3)
    # axis.set_ylim(ymin=-0.1, ymax=1.1)

    axis.legend(plotobjects, plotobjectlabels, loc='upper right', handlelength=2,
                frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_ylabel(r'F$_\lambda$')

    fig.savefig(args.outputfile, format='pdf')
    print(f'Saving {args.outputfile}')
    plt.close()

    # plt.setp(plt.getp(axis, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(axis, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #    axis.spines[axis].set_linewidth(framewidth)


if __name__ == "__main__":
    main()
