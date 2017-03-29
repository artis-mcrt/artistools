#!/usr/bin/env python3
import argparse
import glob
import itertools
import sys

import matplotlib.pyplot as plt
import pandas as pd

import artistools as af


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS radiation field.')
    parser.add_argument('-lcpath', action='append', default=[],
                        help='Paths to light_curve.out files (may include wildcards such as * and **)')
    parser.add_argument('-o', action='store', dest='outputfile', default='plotlightcurve.pdf',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if len(args.lcpath) == 0:
        args.lcpath = ['light_curve.out', '*/light_curve.out']

    # combined the results of applying wildcards on each input
    lcfiles = list(itertools.chain.from_iterable([glob.glob(x) for x in args.lcpath]))

    if not lcfiles:
        print('No light_curve.out files found.')
        sys.exit()
    else:
        make_lightcurve_plot(lcfiles, args.outputfile)


def make_lightcurve_plot(lcfiles, filenameout):
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
        "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for index, lcfilename in enumerate(lcfiles):
        lightcurvedata = pd.read_csv(lcfilename, delim_whitespace=True, header=None, names=['time', 'flux', 'flux_cmf'])

        # the light_curve.dat file repeats x values, so keep the first half only
        lightcurvedata = lightcurvedata.iloc[:len(lightcurvedata) // 2]

        modelname = af.get_model_name(lcfilename)
        print(f"Plotting {modelname}")

        linestyle = ['-', '--'][int(index / 7)]

        lightcurvedata.plot(x='time', y='flux', linewidth=1.5, ax=axis, linestyle=linestyle, label=modelname)

    # axis.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # axis.set_ylim(ymin=-0.1,ymax=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    axis.set_ylabel(r'$\propto$ Flux')

    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


if __name__ == "__main__":
    main()
