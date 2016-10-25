#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import math
import warnings

import matplotlib.pyplot as plt
import readartisfiles as af

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def main():
    """
        Plot ARTIS spectra
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS model spectra by finding spec.out files '
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
    parser.add_argument('-legendfontsize', type=int, default=8,
                        help='Font size of legend text')
    parser.add_argument('-obsspec', action='append', dest='obsspecfiles',
                        help='Include observational spectrum with this'
                             ' file name')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotspec.pdf',
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


def make_plot(specfiles, args):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """
    import matplotlib.ticker as ticker
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
        "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    af.plot_reference_spectra(axis, args)
    plot_artis_spectra(axis, args, specfiles)

    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.set_ylim(ymin=-0.1, ymax=1.25)

    axis.legend(loc='best', handlelength=2, frameon=False,
                numpoints=1, prop={'size': args.legendfontsize})
    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    axis.set_ylabel(r'Scaled F$_\lambda$')

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


def plot_artis_spectra(axis, args, specfiles):
    """
        Plot ARTIS emergent spectra
    """
    if args.timestepmax:
        print('Plotting timesteps {0} to {1}'.format(args.timestepmin, args.timestepmax))
    else:
        print('Plotting timestep {0}'.format(args.timestepmin))

    # dashesList = [(), (1.5, 2, 9, 2), (5, 1), (0.5, 2), (4, 2)]
    # dash_capstyleList = ['butt', 'butt', 'butt', 'round', 'butt']
    # colorlist = [(0, .8*158./255, 0.6*115./255), (204./255, 121./255, 167./255), (213./255, 94./255, 0.0)]

    for index, specfilename in enumerate(specfiles):
        print(specfilename)
        try:
            plotlabelfile = os.path.join(os.path.dirname(specfilename), 'plotlabel.txt')
            modelname = open(plotlabelfile, mode='r').readline().strip()
        except FileNotFoundError:
            modelname = os.path.dirname(specfilename)
            if not modelname:
                # use the current directory name
                modelname = os.path.split(os.path.dirname(os.path.abspath(specfilename)))[1]

        linelabel = '{0} at t={1:d}d'.format(
            modelname, math.floor(float(af.get_timestep_time(specfilename, args.timestepmin))))

        if args.timestepmax > args.timestepmin:
            linelabel += ' to {0:d}d'.format(math.floor(float(af.get_timestep_time(specfilename, args.timestepmax))))

        def filterfunc(arrayfnu):
            from scipy.signal import savgol_filter
            return savgol_filter(arrayfnu, 5, 2)

        spectrum = af.get_spectrum(specfilename,
                                   args.timestepmin,
                                   args.timestepmax,
                                   normalised=False,
                                   fnufilterfunc=filterfunc)

        maxyvaluethisseries = spectrum.query(
            '@args.xmin < lambda_angstroms and '
            'lambda_angstroms < @args.xmax')['f_lambda'].max()

        linestyle = ['-', '--'][int(index / 7) % 2]
        spectrum['f_lambda_scaled'] = (spectrum['f_lambda'] /
                                       maxyvaluethisseries)

        spectrum.plot(x='lambda_angstroms', y='f_lambda_scaled', ax=axis,
                      linestyle=linestyle, lw=2.5 - (0.2 * index),
                      label=linelabel, alpha=0.95, color=None)  # colorlist[index % len(colorlist)]
        # dashes=dashesList[index], dash_capstyle=dash_capstyleList[index])

if __name__ == "__main__":
    main()
