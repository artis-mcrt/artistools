#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

import readartisfiles as af


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
        make_plot(args, specfiles)


def make_plot(args, specfiles):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """
    # import matplotlib.ticker as ticker
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
        "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    plot_obs_spectra(axis, args)
    plot_artis_spectra(axis, args, specfiles)

    axis.set_xlim(xmin=args.xmin, xmax=args.xmax)
    axis.set_ylim(ymin=-0.1, ymax=1.1)

    axis.legend(loc='best', handlelength=2, frameon=False,
                numpoints=1, prop={'size': 10})
    axis.set_xlabel(r'Wavelength ($\AA$)')
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axis.set_ylabel(r'F$_\lambda$')

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


def plot_obs_spectra(axis, args):
    """
        Plot observational spectra listed in args.obsspecfiles
    """
    import scipy.signal
    if args.obsspecfiles is not None:
        scriptdir = os.path.dirname(os.path.abspath(__file__))
        obsspectralabels = {
            '2010lp_20110928_fors2.txt':
                'SN2010lp +264d (Taubenberger et al. 2013)',
            'dop_dered_SN2013aa_20140208_fc_final.txt':
                'SN2013aa +360d (Maguire et al. in prep)',
            '2003du_20031213_3219_8822_00.txt':
                'SN2003du +221.3d (Stanishev et al. 2007)',
            'nero-nebspec.txt':
                'NERO +300d'
        }
        colorlist = ['black', '0.4']
        obsspectra = [(fn, obsspectralabels.get(fn, fn), c)
                      for fn, c in zip(args.obsspecfiles, colorlist)]
        for (filename, serieslabel, linecolor) in obsspectra:
            obsfile = os.path.join(scriptdir, 'spectra', filename)
            obsdata = pd.read_csv(obsfile, delim_whitespace=True, header=None,
                                  names=['lambda_angstroms', 'f_lambda'])

            if len(obsdata) > 5000:
                # obsdata = scipy.signal.resample(obsdata, 10000)
                obsdata = obsdata[::3]

            obsdata.query('lambda_angstroms > @args.xmin and '
                          'lambda_angstroms < @args.xmax',
                          inplace=True)

            print("'{0}' has {1} points".format(serieslabel, len(obsdata)))

            obsdata['f_lambda'] = (obsdata['f_lambda'] /
                                   obsdata['f_lambda'].max())

            obsdata['f_lambda'] = scipy.signal.savgol_filter(
                obsdata['f_lambda'], 5, 3)

            obsdata.plot(x='lambda_angstroms',
                         y='f_lambda', lw=1.5, ax=axis,
                         label=serieslabel, zorder=-1, color=linecolor)


def plot_artis_spectra(axis, args, specfiles):
    """
        Plot ARTIS emergent spectra
    """
    if args.timestepmax:
        print('Plotting timesteps {0} to {1}'.format(
            args.timestepmin, args.timestepmax))
    else:
        print('Plotting timestep {0}'.format(args.timestepmin))

    for index, specfilename in enumerate(specfiles):
        print(specfilename)
        linelabel = '{0} at t={1}d'.format(specfilename.split(
            '/spec.out')[0], af.get_timestep_time(specfilename,
                                                  args.timestepmin))
        if args.timestepmax > args.timestepmin:
            linelabel += ' to {0}d'.format(
                af.get_timestep_time(specfilename, args.timestepmax))

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
                      label=linelabel, alpha=0.9)

if __name__ == "__main__":
    main()
