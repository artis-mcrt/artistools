#!/usr/bin/env python3
import os
import sys
import scipy.signal
import numpy as np
import glob
import argparse
import readartisfiles as af

parser = argparse.ArgumentParser(
    description='Plot artis model spectra by finding spec.out files in the current directory or subdirectories.')
parser.add_argument('-specpath', action='store', default='**/spec.out',
                    help='Path to spec.out file (may include wildcards such as * and **)')
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
parser.add_argument('-obsspec', action='append', dest='obsspecfiles',
                    help='Include observational spectrum with this file name')
parser.add_argument('-o', action='store', dest='outputfile', default='plotartisspec.pdf',
                    help='path/filename for PDF file')
args = parser.parse_args()

xminvalue, xmaxvalue = args.xmin, args.xmax

# colorlist = ['black',(0.0,0.5,0.7),(0.35,0.7,1.0),(0.9,0.2,0.0),(0.9,0.6,0.0),(0.0,0.6,0.5),(0.8,0.5,1.0),(0.95,0.9,0.25)]
colorlist = [(0.0, 0.5, 0.7), (0.9, 0.2, 0.0), (0.9, 0.6, 0.0),
             (0.0, 0.6, 0.5), (0.8, 0.5, 1.0), (0.95, 0.9, 0.25)]

elementlist = []

numberofcolumns = 5
h = 6.62607004e-34  # m^2 kg / s
c = 299792458  # m / s

specfiles = glob.glob(args.specpath, recursive=True)


def main():
    if not specfiles:
        print('no spec.out files found')
        sys.exit()
    if args.listtimesteps:
        af.showtimesteptimes(specfiles[0])
    else:
        makeplot()


def makeplot():
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt

    import collections
    elementtuple = collections.namedtuple(
        'elementtuple', 'Z,nions,lowermost_ionstage,uppermost_ionstage,nlevelsmax_readin,abundance,mass')
    with open(specfiles[0].replace('spec.out', 'compositiondata.txt'), 'r') as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline()  # T_preset
        fcompdata.readline()  # homogeneous_abundances
        for element in range(nelements):
            line = fcompdata.readline()
            linesplit = line.split()
            elementlist.append(elementtuple._make(
                list(map(int, linesplit[:5])) + list(map(float, linesplit[5:]))))
            print(elementlist[-1])

    print('nelements {0}'.format(nelements))
    maxion = 5  # must match sn3d.h value

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
                           "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if args.obsspecfiles is not None:
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        obsspectralabels = \
            {
                '2010lp_20110928_fors2.txt': 'SN2010lp +264d (Taubenberger et al. 2013)',
                'dop_dered_SN2013aa_20140208_fc_final.txt': 'SN2013aa +360d (Maguire et al. in prep)',
                '2003du_20031213_3219_8822_00.txt': 'SN2003du +221.3d (Stanishev et al. 2007)'
            }
        obscolorlist = ['black', '0.4']
        obsspectra = [(fn, obsspectralabels[fn], c)
                      for fn, c in zip(args.obsspecfiles, obscolorlist)]
        for (filename, serieslabel, linecolor) in obsspectra:
            obsfile = os.path.join(scriptdir, 'spectra', filename)
            obsdata = np.loadtxt(obsfile)
            if len(obsdata[:, 1]) > 5000:
                # obsdata = scipy.signal.resample(obsdata, 10000)
                obsdata = obsdata[::3]
            obsdata = obsdata[(obsdata[:, 0] > xminvalue) & (obsdata[:, 0] < xmaxvalue)]
            print("'{0}' has {1} points".format(serieslabel, len(obsdata)))
            obsxvalues = obsdata[:, 0]
            obsyvalues = obsdata[:, 1] * (1.0 / max(obsdata[:, 1]))

            # obsyvalues = scipy.signal.savgol_filter(obsyvalues, 5, 3)
            ax.plot(obsxvalues, obsyvalues / max(obsyvalues), lw=1.5,
                    label=serieslabel, zorder=-1, color=linecolor)

    # in the spec.out file, the column index is one more than the timestep
    # (because column 0 is wavelength row headers, not flux at a timestep)
    timeindexlow = args.timestepmin
    if args.timestepmax:
        timeindexhigh = args.timestepmax
        print('Ploting timesteps {0} to {1}'.format(args.timestepmin, args.timestepmax))
    else:
        print('Ploting timestep {0}'.format(args.timestepmin))
        timeindexhigh = timeindexlow

    specdata = np.loadtxt(specfiles[0])
    emissiondata = np.loadtxt(specfiles[0].replace('spec.out', 'emission.out'))

    timearray = specdata[0, 1:]
    arraynu = specdata[1:, 0]
    arraylambda = c / specdata[1:, 0]

    maxyvalueglobal = 0.0
    linenumber = 0
    for element in reversed(range(nelements)):
        # nions = elementlist[element].nions
        nions = elementlist[element].uppermost_ionstage - \
            elementlist[element].lowermost_ionstage + 1
        for ion in range(nions):
            ion_stage = ion + elementlist[element].lowermost_ionstage
            ionserieslist = []
            if linenumber == 0:
                ionserieslist.append((2 * nelements * maxion, 'free-free'))
                linenumber += 1  # so the linestyle resets
            ionserieslist.append((element * maxion + ion, 'bound-bound'))
            ionserieslist.append((nelements * maxion + element * maxion + ion, 'bound-free'))
            for (selectedcolumn, emissiontype) in ionserieslist:
                array_fnu = emissiondata[timeindexlow::len(timearray), selectedcolumn]

                for timeindex in range(timeindexlow + 1, timeindexhigh + 1):
                    array_fnu += emissiondata[timeindex::len(timearray), selectedcolumn]

                array_fnu = array_fnu / (timeindexhigh - timeindexlow + 1)

                # best to use the filter on this list (because it hopefully has regular sampling)
                array_fnu = scipy.signal.savgol_filter(array_fnu, 5, 2)

                array_flambda = array_fnu * (arraynu ** 2) / c

                maxyvaluethisseries = max([array_flambda[i] if (xminvalue < (
                    1e10 * arraylambda[i]) < xmaxvalue) else -99.0 for i in range(len(array_flambda))])
                maxyvalueglobal = max(maxyvalueglobal, maxyvaluethisseries)

                linelabel = ''
                if emissiontype != 'free-free':
                    linelabel += '{0} {1}'.format(af.elsymbols[elementlist[element].Z],
                                                  af.roman_numerals[ion_stage])
                # linelabel += ' {:}'.format(emissiontype)
                plotlabel = 't={0}d'.format(specdata[0, timeindexlow])
                if timeindexhigh > timeindexlow:
                    plotlabel += ' to {0}d'.format(specdata[0, timeindexhigh])
                linewidth = [1.8, 0.8][emissiontype == 'bound-free']
                if emissiontype == 'bound-bound' and linelabel in ['Fe II', 'Fe III', 'O I', 'O II']:
                    ax.plot(1e10 * arraylambda, array_flambda / maxyvalueglobal,
                            color=colorlist[int(linenumber) % len(colorlist)], lw=linewidth, label=linelabel)
                    linenumber += 1

    ax.annotate(plotlabel, xy=(0.1, 0.96), xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top', fontsize=12)
    ax.set_xlim(xmin=xminvalue, xmax=xmaxvalue)
    #        ax.set_xlim(xmin=12000,xmax=19000)
    # ax.set_ylim(ymin=-0.05*maxyvalueglobal,ymax=maxyvalueglobal*1.3)
    ax.set_ylim(ymin=-0.1, ymax=1.1)

    ax.legend(loc='upper right', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    ax.set_xlabel(r'Wavelength ($\AA$)')
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    ax.set_ylabel(r'F$_\lambda$')

    # filenameout = 'plotartisspec_{:}_to_{:}.pdf'.format(*timesteparray)
    filenameout = 'plotartisemission.pdf'
    fig.savefig(filenameout, format='pdf')
    print('Saving {0}'.format(filenameout))
    plt.close()

    # plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(framewidth)

    # for (x,y,symbol) in zip(highlightedatomicnumbers,highlightedelementyposition,highlightedelements):
    #    ax.annotate(symbol, xy=(x, y - 0.0 * (x % 2)), xycoords='data', textcoords='offset points', xytext=(0,10), horizontalalignment='center', verticalalignment='center', weight='bold', fontsize=fs-1.5)

if __name__ == "__main__":
    main()
