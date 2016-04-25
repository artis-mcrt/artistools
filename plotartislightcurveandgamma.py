#!/usr/bin/env python3
import os
import sys
import math
import numpy as np
import glob


def main():
    glcfiles = glob.glob('gamma_light_curve.out') + \
        glob.glob('*/gamma_light_curve.out') + glob.glob('*/*/gamma_light_curve.out')
    lcfiles = glob.glob('light_curve.out') + glob.glob('*/light_curve.out') + \
        glob.glob('*/*/light_curve.out')

    if not lcfiles:
        print('no gamma_light_curve.out files found')
        sys.exit()
    else:
        makeplot(lcfiles, glcfiles)


def makeplot(lcfiles, glcfiles):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
                           "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for n, lcfilename in enumerate(lcfiles):
        lightcurvedata = np.loadtxt(lcfilename)
        gammalightcurvedata = np.loadtxt(glcfiles[n])

        linelabel = '{0}'.format(lcfilename.split('/gamma_light_curve.out')[0])

        arraytime = lightcurvedata[:, 0]
        arrayflux = lightcurvedata[:, 1]

        linestyle = ['-', '--'][int(n / 7)]
        ax.plot(arraytime, arrayflux, lw=1.5, linestyle=linestyle, label="Optical")
        arrayflux = gammalightcurvedata[:, 1]
        ax.plot(arraytime, arrayflux, lw=1.5, linestyle=linestyle, label="Gamma")
        ax.plot(arraytime, lightcurvedata[
                :, 1] + gammalightcurvedata[:, 1], lw=1.5, linestyle=linestyle, label="Sum")

    # ax.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # ax.set_ylim(ymin=-0.1,ymax=1.3)

    ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    ax.set_xlabel(r'Time (days)')
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    ax.set_ylabel(r'$\propto$ Flux (Optical + Gamma)')

    fig.savefig('plotartisrgammalightcurve.pdf', format='pdf')
    plt.close()

    # plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    # plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    # for axis in ['top','bottom','left','right']:
    #    ax.spines[axis].set_linewidth(framewidth)

main()
