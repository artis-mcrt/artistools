#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import pandas as pd

import readartisfiles as af

parser = argparse.ArgumentParser(
    description='Plot artis model spectra by finding spec.out files '
                'in the current directory or subdirectories.')
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
parser.add_argument('-obsspec', action='append', dest='obsspecfiles',
                    help='Include observational spectrum with this file name')
parser.add_argument('-o', action='store', dest='outputfile',
                    default='plotspec.pdf',
                    help='path/filename for PDF file')
args = parser.parse_args()

xminvalue, xmaxvalue = args.xmin, args.xmax
# xminvalue, xmaxvalue = 10000, 20000

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
    # import matplotlib.ticker as ticker
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
        "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if args.obsspecfiles is not None:
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        obsspectralabels = \
            {
                '2010lp_20110928_fors2.txt':
                    'SN2010lp +264d (Taubenberger et al. 2013)',
                'dop_dered_SN2013aa_20140208_fc_final.txt':
                    'SN2013aa +360d (Maguire et al. in prep)',
                '2003du_20031213_3219_8822_00.txt':
                    'SN2003du +221.3d (Stanishev et al. 2007)'
            }
        colorlist = ['black', '0.4']
        obsspectra = [(fn, obsspectralabels[fn], c)
                      for fn, c in zip(args.obsspecfiles, colorlist)]
        for (filename, serieslabel, linecolor) in obsspectra:
            obsfile = os.path.join(scriptdir, 'spectra', filename)
            obsdata = pd.read_csv(obsfile, delim_whitespace=True, header=None)

            if len(obsdata) > 5000:
                # obsdata = scipy.signal.resample(obsdata, 10000)
                obsdata = obsdata[::3]

            obsdata = obsdata[(obsdata[:][0] > xminvalue) &
                              (obsdata[:][0] < xmaxvalue)]
            print("'{0}' has {1} points".format(serieslabel, len(obsdata)))
            obsxvalues = obsdata[0]
            obsyvalues = obsdata[1]

            # obsyvalues = scipy.signal.savgol_filter(obsyvalues, 5, 3)
            ax.plot(obsxvalues, obsyvalues / max(obsyvalues), lw=1.5,
                    label=serieslabel, zorder=-1, color=linecolor)

    # in the spec.out file, the column index is one more than the timestep
    # (because column 0 is wavelength row headers, not flux at a timestep)
    if args.timestepmax:
        print('Ploting timesteps {0} to {1}'.format(
            args.timestepmin, args.timestepmax))
    else:
        print('Ploting timestep {0}'.format(args.timestepmin))

    for s, specfilename in enumerate(specfiles):
        linelabel = '{0} at t={1}d'.format(specfilename.split(
            '/spec.out')[0], af.get_timestep_time(specfilename,
                                                  args.timestepmin))
        if args.timestepmax > args.timestepmin:
            linelabel += ' to {0}d'.format(af.get_timestep_time(specfilename,
                                                                args.timestepmax))

        from scipy.signal import savgol_filter
        filterfunc = lambda x: savgol_filter(x, 5, 2)

        arraylambda, array_flambda = af.get_spectrum(specfilename,
                                                     args.timestepmin,
                                                     args.timestepmax,
                                                     normalised=False,
                                                     fnufilterfunc=filterfunc)

        maxyvaluethisseries = max(
            [flambda if (xminvalue < arraylambda[i] < xmaxvalue)
             else -99.0
             for i, flambda in enumerate(array_flambda)])

        linestyle = ['-', '--'][int(s / 7)]
        ax.plot(arraylambda, array_flambda / maxyvaluethisseries,
                linestyle=linestyle, lw=2.5 - (0.1 * s), label=linelabel)

    ax.set_xlim(xmin=xminvalue, xmax=xmaxvalue)
    #        ax.set_xlim(xmin=12000,xmax=19000)
    # ax.set_ylim(ymin=-0.1*maxyvalueglbal,ymax=maxyvalueglobal*1.1)
    ax.set_ylim(ymin=-0.1, ymax=1.1)

    ax.legend(loc='best', handlelength=2, frameon=False,
              numpoints=1, prop={'size': 11})
    ax.set_xlabel(r'Wavelength ($\AA$)')
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    ax.set_ylabel(r'F$_\lambda$')

    # filenameout = 'plotartisspec_{:}_to_{:}.pdf'.format(*timesteparray)
    filenameout = args.outputfile
    fig.savefig(filenameout, format='pdf')
    print('Saving {0}'.format(filenameout))
    plt.close()

    # plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(framewidth)

    # ax.annotate(symbol, xy=(x, y - 0.0 * (x % 2)), xycoords='data',
    #             textcoords='offset points', xytext=(0, 10),
    #             horizontalalignment='center', verticalalignment='center',
    #             weight='bold', fontsize=15)


if __name__ == "__main__":
    main()
