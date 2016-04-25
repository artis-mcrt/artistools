#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
# import pandas as pd
import glob


def main():
    lcfiles = glob.glob('light_curve.out') + glob.glob('*/light_curve.out') + \
        glob.glob('*/*/light_curve.out')
    # could alternatively use
    # specfiles = glob.glob('**/light_curve.out',recursive=True)

    if not lcfiles:
        print('no light_curve.out files found')
        sys.exit()
    else:
        makeplot(lcfiles)


def makeplot(lcfiles):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
                           "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for n, lcfilename in enumerate(lcfiles):
        lightcurvedata = np.loadtxt(lcfilename)
        # lightcurvedata = pd.read_csv(lcfiles[n], delim_whitespace=True)  #maybe
        # switch to Pandas at some point

        linelabel = '{0}'.format(lcfilename.split('/light_curve.out')[0])

        arraytime = lightcurvedata[:, 0]

        arrayflux = lightcurvedata[:, 1]

        linestyle = ['-', '--'][int(n / 7)]

        ax.plot(arraytime, arrayflux, lw=1.5, linestyle=linestyle, label=linelabel)

    # ax.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # ax.set_ylim(ymin=-0.1,ymax=1.3)

    ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    ax.set_xlabel(r'Time (days)')
    ax.set_ylabel(r'$\propto$ Flux')

    fig.savefig('plotartislightcurve.pdf', format='pdf')
    plt.close()

main()
