#!/usr/bin/env python3
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd


def main():
    lcfiles = glob.glob('light_curve.out') + glob.glob(
        '**/light_curve.out', recursive=True)

    if not lcfiles:
        print('No light_curve.out files found.')
        sys.exit()
    else:
        makeplot(lcfiles)


def makeplot(lcfiles):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
                           "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for n, lcfilename in enumerate(lcfiles):
        lightcurvedata = pd.read_csv(lcfilename, delim_whitespace=True,
                                     header=None)

        linelabel = '{0}'.format(lcfilename.split('/light_curve.out')[0])

        linestyle = ['-', '--'][int(n / 7)]

        arraytime, arrayflux = zip(*lightcurvedata.iloc[:, :2].values)

        ax.plot(arraytime, arrayflux, lw=1.5, linestyle=linestyle,
                label=linelabel)

    # ax.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # ax.set_ylim(ymin=-0.1,ymax=1.3)

    ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1,
              prop={'size': 9})
    ax.set_xlabel(r'Time (days)')
    ax.set_ylabel(r'$\propto$ Flux')

    fig.savefig('plotlightcurve.pdf', format='pdf')
    plt.close()

if __name__ == "__main__":
    main()
