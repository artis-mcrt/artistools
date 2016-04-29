#!/usr/bin/env python3
# import os
# import math
# import matplotlib.ticker as ticker
# import numpy as np
import pandas as pd

K_B = 8.617332478e-5   # eV / K
c = 299792458          # m / s
h = 4.13566766225e-15  # eV / s

plot_xmin = 3500        # plot range in angstroms
plot_xmax = 7000


def main():
    radfield_file = 'radfield.out'
    print('Loading {:}...'.format(radfield_file))
    radfielddata = pd.read_csv(radfield_file, delim_whitespace=True)

    # filter the line list
    radfielddata = radfielddata[
        ((radfielddata[:]['modelgridindex'] == 0) &
         (radfielddata[:]['timestep'] == 10))
    ]

    listlambda = 1e10 * c / \
        ((radfielddata[:]['nu_upper'] + radfielddata[:]['nu_lower']) / 2.0)
    listdlambda = (c / radfielddata[:]['nu_lower']) - (c / radfielddata[:]['nu_upper'])

    listenergy = h * \
        ((radfielddata[:]['nu_upper'] + radfielddata[:]['nu_lower']) / 2.0)
    listdenergy = h * (radfielddata[:]['nu_upper'] - radfielddata[:]['nu_lower'])
    xvalues = listlambda
    # xvalues = listenergy

    yvalues = radfielddata[:]['J'] / listdlambda
    # yvalues = radfielddata[:]['J'] / listdenergy

    print('Plotting...')
    make_plot(xvalues, yvalues)
    # print(xvalues)


def make_plot(xvalues, yvalues):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                           tight_layout={
                               "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ax.plot(xvalues, yvalues, linewidth=1.5)

    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel(r'dJ / d$\lambda$')
    ax.set_xlim(xmin=3500, xmax=7000)

    # ax.set_xlabel(r'Energy (eV)')
    # ax.set_ylabel(r'dJ / dE')
    # ax.set_xlim(xmin=0.0, xmax=5)
    # ax[ion_index].legend(loc='best', handlelength=2,
    #                          frameon=False, numpoints=1, prop={'size': 13})

    # ax.set_ylim(ymin=-0.05,ymax=1.1)

    fig.savefig('plotradfield.pdf', format='pdf')
    plt.close()

main()
