#!/usr/bin/env python3
import argparse
import glob
import math
import os
import sys
import warnings

import matplotlib.pyplot as plt

import readartisfiles as af

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


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
    af.addargs_timesteps(parser)
    af.addargs_spectrum(parser)
    parser.add_argument('-legendfontsize', type=int, default=8,
                        help='Font size of legend text')
    parser.add_argument('-o', action='store', dest='outputfile', default='plotspec.pdf',
                        help='path/filename for PDF file')
    args = parser.parse_args()

    if len(args.filepaths) == 0:
        args.filepaths = ['spec.out', '*/spec.out']  # '**/spec.out'

    inputfiles = []
    for filepath in args.filepaths:
        inputfiles.extend(glob.glob(filepath, recursive=True))

    if not inputfiles:
        print('no spec.out files found')
        sys.exit()
    if args.listtimesteps:
        af.showtimesteptimes(inputfiles[0])
    else:
        make_plot(inputfiles, args)


def make_plot(inputfiles, args):
    """
        Set up a matplotlib figure and plot observational and ARTIS spectra
    """
    import matplotlib.ticker as ticker
    fig, axis = plt.subplots(
        1, 1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

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
    axis.set_xlabel(r'Wavelength ($\AA$)')
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
    axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
    if args.normalised:
        axis.set_ylabel(r'Scaled F$_\lambda$')
    else:
        axis.set_ylabel(r'F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\AA$]')

    filenameout = args.outputfile
    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()

    # plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(framewidth)


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

            spectrum = af.get_spectrum(specfilename, timestepmin, timestepmax, normalised=False,)
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
