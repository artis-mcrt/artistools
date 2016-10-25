#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np
import scipy.signal
from astropy import constants as const
import readartisfiles as af

# colorlist = ['black',(0.0,0.5,0.7),(0.35,0.7,1.0),(0.9,0.2,0.0),
#             (0.9,0.6,0.0),(0.0,0.6,0.5),(0.8,0.5,1.0),(0.95,0.9,0.25)]
colorlist = [(0.0, 0.5, 0.7), (0.9, 0.2, 0.0), (0.9, 0.6, 0.0),
             (0.0, 0.6, 0.5), (0.8, 0.5, 1.0), (0.95, 0.9, 0.25)]


def main():
    parser = argparse.ArgumentParser(
        description='Plot artis model spectra by finding spec.out files in '
                    'the current directory or subdirectories.')
    parser.add_argument('-specpath', action='store', default='**/spec.out',
                        help='Path to spec.out file (may include wildcards '
                        'such as * and **)')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestepmin', type=int, default=70,
                        help='First or only included timestep')
    parser.add_argument('-timestepmax', type=int, default=80,
                        help='Last included timestep')
    parser.add_argument('-xmin', type=int, default=3500,
                        help='Plot range: minimum wavelength')
    parser.add_argument('-xmax', type=int, default=7000,
                        help='Plot range: maximum wavelength')
    parser.add_argument('-maxseriescount', type=int, default=6,
                        help='Maximum number of plot series (ions/processes)')
    parser.add_argument('-obsspec', action='append', dest='obsspecfiles',
                        help='Include observational spectrum with this '
                        'file name')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotemission.pdf',
                        help='path/filename for PDF file')
    args = parser.parse_args()

    specfiles = glob.glob(args.specpath, recursive=True)

    if not specfiles:
        print('no spec.out files found')
        sys.exit()
    if args.listtimesteps:
        af.showtimesteptimes(specfiles[0])
    else:
        make_plot(specfiles, args)


def get_flux_contributions(emissionfilename, elementlist, maxion, timearray, arraynu, args, timeindexhigh):
    emissiondata = np.loadtxt(emissionfilename)
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
            if element == ion == 0:
                ionserieslist.append((2 * nelements * maxion, 'free-free'))
            ionserieslist.append((element * maxion + ion, 'bound-bound'))
            ionserieslist.append((nelements * maxion + element * maxion + ion, 'bound-free'))
            for (selectedcolumn, emissiontype) in ionserieslist:
                array_fnu = emissiondata[args.timestepmin::len(timearray), selectedcolumn]

                for timeindex in range(args.timestepmin + 1, timeindexhigh + 1):
                    array_fnu += emissiondata[timeindex::len(timearray), selectedcolumn]

                array_fnu = array_fnu / (timeindexhigh - args.timestepmin + 1)

                # best to use the filter on this list (because it hopefully has
                # regular sampling)
                array_fnu = scipy.signal.savgol_filter(array_fnu, 5, 2)

                array_flambda = array_fnu * (arraynu ** 2) / c

                maxyvaluethisseries = max(
                    [array_flambda[i] if (args.xmin < (1e10 * arraylambda[i]) < args.xmax) else -99.0
                     for i in range(len(array_flambda))])
                maxyvalueglobal = max(maxyvalueglobal, maxyvaluethisseries)

                linelabel = ''
                if emissiontype != 'free-free':
                    linelabel += '{0} {1} '.format(af.elsymbols[elementlist.Z[element]], af.roman_numerals[ion_stage])
                linelabel += '{:}'.format(emissiontype)

                # if not linelabel.startswith('Fe I '):
                contribution_list.append([maxyvaluethisseries, linelabel, array_flambda])
    return contribution_list, maxyvalueglobal


def plot_reference_spectra(axis, plotobjects, plotobjectlabels, maxyvalueglobal, args):
    if args.obsspecfiles is not None:
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        obscolorlist = ['0.4', '0.8']
        obsspectra = [(fn, af.obsspectralabels.get(fn, fn), c) for fn, c in zip(args.obsspecfiles, obscolorlist)]

        for (filename, serieslabel, linecolor) in obsspectra:
            obsfile = os.path.join(scriptdir, 'spectra', filename)
            obsdata = np.loadtxt(obsfile)
            if len(obsdata[:, 1]) > 5000:
                # obsdata = scipy.signal.resample(obsdata, 10000)
                obsdata = obsdata[::3]
            obsdata = obsdata[(obsdata[:, 0] > args.xmin) &
                              (obsdata[:, 0] < args.xmax)]
            print("'{0}' has {1} points".format(serieslabel, len(obsdata)))
            obsxvalues = obsdata[:, 0]
            obsyvalues = obsdata[:, 1] * (1.0 / max(obsdata[:, 1])) * maxyvalueglobal

            # obsyvalues = scipy.signal.savgol_filter(obsyvalues, 5, 3)
            axis.plot(obsxvalues, obsyvalues, lw=0.5, zorder=-1, color=linecolor)
            plotobjects.append(mpatches.Patch(color=linecolor))
            plotobjectlabels.append(serieslabel)


def make_plot(specfiles, args):
    elementlist = af.get_composition_data(specfiles[0].replace('spec.out', 'compositiondata.txt'))
    specfilename = specfiles[0]

    print('nelements {0}'.format(len(elementlist)))
    maxion = 5  # must match sn3d.h value

    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    # in the spec.out file, the column index is one more than the timestep
    # (because column 0 is wavelength row headers, not flux at a timestep)
    if args.timestepmax:
        timeindexhigh = args.timestepmax
        print('Ploting timesteps {0} to {1}'.format(
            args.timestepmin, args.timestepmax))
    else:
        print('Ploting timestep {0}'.format(args.timestepmin))
        timeindexhigh = args.timestepmin

    specdata = np.loadtxt(specfilename)

    try:
        plotlabelfile = os.path.join(os.path.dirname(specfilename), 'plotlabel.txt')
        modelname = open(plotlabelfile, mode='r').readline().strip()
    except FileNotFoundError:
        modelname = os.path.dirname(specfilename)
        if not modelname:
            # use the current directory name
            modelname = os.path.split(os.path.dirname(os.path.abspath(specfilename)))[1]

    plotlabel = '{0} at t={1:d}d'.format(modelname, math.floor(specdata[0, args.timestepmin + 1]))
    if timeindexhigh > args.timestepmin:
        plotlabel += ' to {0:d}d'.format(math.floor(specdata[0, timeindexhigh + 1]))

    timearray = specdata[0, 1:]
    arraynu = specdata[1:, 0]
    arraylambda = const.c.to('m/s').value / arraynu
    contribution_list, maxyvalueglobal = get_flux_contributions(
        specfilename.replace('spec.out', 'emission.out'), elementlist, maxion, timearray, arraynu, args, timeindexhigh)

    maxseriescount = args.maxseriescount
    contribution_list = sorted(contribution_list, key=lambda x: x[0])
    remainder_sum = np.zeros(len(arraylambda))
    for row in contribution_list[:-maxseriescount]:
        remainder_sum = np.add(remainder_sum, row[2])

    contribution_list = contribution_list[-maxseriescount:]
    contribution_list.insert(0, [0.0, 'other', remainder_sum])

    stackplot_emission_obj = axis.stackplot(1e10 * arraylambda, *[x[2] for x in contribution_list], linewidth=0)
    plotobjects = list(reversed(stackplot_emission_obj))
    plotobjectlabels = list(reversed([x[1] for x in contribution_list]))

    plot_reference_spectra(axis, plotobjects, plotobjectlabels, maxyvalueglobal, args)

    axis.annotate(plotlabel, xy=(0.1, 0.96), xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top', fontsize=12)
    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    #        axis.set_xlim(xmin=12000,xmax=19000)
    # axis.set_ylim(ymin=-0.05*maxyvalueglobal,ymax=maxyvalueglobal*1.3)
    # axis.set_ylim(ymin=-0.1, ymax=1.1)

    axis.legend(plotobjects, plotobjectlabels, loc='upper right', handlelength=2,
              frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Wavelength ($\AA$)')
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_ylabel(r'F$_\lambda$')

    fig.savefig(args.outputfile, format='pdf')
    print('Saving {0}'.format(args.outputfile))
    plt.close()

    # plt.setp(plt.getp(axis, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(axis, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #    axis.spines[axis].set_linewidth(framewidth)

    # for (x,y,symbol) in zip(highlightedatomicnumbers,
    #                         highlightedelementyposition,highlightedelements):
    #    axis.annotate(symbol, xy=(x, y - 0.0 * (x % 2)), xycoords='data',
    #                textcoords='offset points', xytext=(0,10),
    #                horizontalalignment='center',
    #                verticalalignment='center', weight='bold',
    #                fontsize=fs-1.5)

if __name__ == "__main__":
    main()
