#!/usr/bin/env python3
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd


def main():
    lcfiles = glob.glob('**/light_curve.out', recursive=True)

    if not lcfiles:
        print('No light_curve.out files found.')
        sys.exit()
    else:
        makeplot(lcfiles)


def makeplot(lcfiles):
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
                           "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for index, lcfilename in enumerate(lcfiles):
        print(lcfilename)
        lightcurvedata = pd.read_csv(lcfilename, delim_whitespace=True,
                                     header=None, names=['time', 'flux', 'flux_cmf'])

        linelabel = f"{lcfilename.split('/light_curve.out')[0]}"

        linestyle = ['-', '--'][int(index / 7)]

        lightcurvedata.plot(x='time', y='flux', lw=1.5, ax=axis,
                            linestyle=linestyle, label=linelabel)

    # axis.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # axis.set_ylim(ymin=-0.1,ymax=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    axis.set_ylabel(r'$\propto$ Flux')

    fig.savefig('plotlightcurve.pdf', format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
