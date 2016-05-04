#!/usr/bin/env python3
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import matplotlib.ticker as ticker

K_B = 8.617332478e-5   # Boltzmann constant [eV / K]
C = 299792458          # [m / s]
H = 4.13566766225e-15  # Planck constant [eV / s]

plot_xmin = 200        # plot range in Angstroms
plot_xmax = 10000
selected_timestep = 10


def main():
    radfield_file = 'radfield.out'
    print('Loading {:}...'.format(radfield_file))
    radfielddata = pd.read_csv(radfield_file, delim_whitespace=True)

    if selected_timestep < 0:
        selected_timestep = max(radfielddata['timestep'])

    # filter the list
    radfielddata = radfielddata[
        ((radfielddata[:]['modelgridindex'] == 0) &
         (radfielddata[:]['timestep'] == selected_timestep))
    ]

    print('Timestep {0:d}'.format(selected_timestep))

    xvalues = []
    yvalues = []
    for _, row in radfielddata.iterrows():
        xvalues.append(1e10 * C / row['nu_lower'])
        xvalues.append(1e10 * C / row['nu_upper'])
        dlambda = (C / row['nu_lower']) - \
            (C / row['nu_upper'])
        yvalues.append(row['J'] / dlambda)
        yvalues.append(row['J'] / dlambda)

    fittedxvalues = []
    fittedyvalues = []
    for _, row in radfielddata.iterrows():
        delta_nu = (row['nu_upper'] - row['nu_lower']) / 100
        for nu in np.arange(row['nu_lower'], row['nu_upper'], delta_nu):
            j_nu = (row['W'] * 1.4745007e-47 * pow(nu, 3) *
                    1.0 / (math.expm1(H * nu / row['T_R'] / K_B)))  # CGS units
            j_lambda = j_nu * (nu ** 2) / C

            fittedxvalues.append(C / nu * 1e10)
            fittedyvalues.append(j_lambda)

    binedges = [C / radfielddata['nu_lower'].iloc[0] * 1e10] + list(C / radfielddata[:]['nu_upper'] * 1e10)
    # print(binedges)
    print('Plotting...')
    make_plot(xvalues, yvalues, fittedxvalues, fittedyvalues, binedges)
    # print(xvalues)


def make_plot(xvalues, yvalues, fittedxvalues, fittedyvalues, binedges):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 4),
                           tight_layout={
                               "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    ax.plot(xvalues, yvalues, linewidth=1, label='Field estimators')
    ax.plot(fittedxvalues, fittedyvalues, linewidth=1, color='green',
            label='Fitted field')
    ax.vlines(binedges, ymin=0.0, ymax=max(yvalues) * 2.0, linewidth=1.0,
              color='red', label='')

    ax.set_xlabel(r'Wavelength ($\AA$)')
    ax.set_ylabel(r'J$_\lambda$ [erg/cm$^2$/m]')
    ax.set_xlim(xmin=plot_xmin, xmax=plot_xmax)

    # ax.set_xlabel(r'Energy (eV)')
    # ax.set_ylabel(r'dJ / dE')
    # ax.set_xlim(xmin=0.0, xmax=5)
    ax.legend(loc='best', handlelength=2,
              frameon=False, numpoints=1, prop={'size': 13})

    # ax.set_ylim(ymin=-0.05,ymax=1.1)

    fig.savefig('plotradfield.pdf', format='pdf')
    plt.close()


main()
