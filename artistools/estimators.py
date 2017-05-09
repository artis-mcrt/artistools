#!/usr/bin/env python3
import math

import matplotlib.pyplot as plt

# import numpy as np
import artistools as at

# from astropy import constants as const


def get_estimators(estimfiles):
    estimators = {}
    for estfile in estimfiles:
        with open(estfile, 'r') as estfile:
            timestep = 0
            modelgridindex = 0
            for line in estfile:
                row = line.split()
                if not row:
                    continue

                if row[0] == 'timestep':
                    timestep = int(row[1])
                    modelgridindex = int(row[3])
                    estimators[(timestep, modelgridindex)] = {}
                    estimators[(timestep, modelgridindex)]['TR'] = float(row[5])
                    estimators[(timestep, modelgridindex)]['Te'] = float(row[7])
                    estimators[(timestep, modelgridindex)]['W'] = float(row[9])
                    estimators[(timestep, modelgridindex)]['TJ'] = float(row[11])
                    estimators[(timestep, modelgridindex)]['nne'] = float(row[15])
                    estimators[(timestep, modelgridindex)]['populations'] = {}
                elif row[0] == 'populations':
                    atomic_number = int(row[1].split('=')[1])
                    for index, token in list(enumerate(row))[2::2]:
                        ion_stage = int(token.rstrip(':'))
                        nionpopulation = float(row[index + 1])
                        estimators[(timestep, modelgridindex)]['populations'][(atomic_number, ion_stage)] = nionpopulation
                        elpop = estimators[(timestep, modelgridindex)]['populations'].get(atomic_number, 0)
                        estimators[(timestep, modelgridindex)]['populations'][atomic_number] = elpop + nionpopulation
                        totalpop = estimators[(timestep, modelgridindex)]['populations'].get('total', 0)
                        estimators[(timestep, modelgridindex)]['populations']['total'] = totalpop + nionpopulation
                elif row[0] == 'heating:':
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'heating_{token}'] = float(row[index + 1])
                elif row[0] == 'cooling:':
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'cooling_{token}'] = float(row[index + 1])

    return estimators


def main():
    units = {
        'TR': 'K',
        'Te': 'K',
        'TJ': 'K',
        'nne': 'e-/cm3',
        'heating_gamma': 'erg/s/cm3',
    }
    # timesteptimes = []
    # with open('light_curve.out', 'r') as lcfile:
    #     for line in lcfile:
    #         timesteptimes.append(line.split()[0])

    # elementlist = at.get_composition_data('compositiondata.txt')
    modeldata, _ = at.get_modeldata('model.txt')
    # initalabundances = at.get_initialabundances1d('abundances.txt')

    estimators = get_estimators(['estimators_0000.out'])

    series = [['heating_gamma'], ['TR'], ['Te'], ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V']]
    fig, axes = plt.subplots(len(series), 1, sharex=True, figsize=(6, 8),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    timestep = 40
    xlist = modeldata['velocity']
    axes[-1].set_xlabel(r'Velocity [km/s]')
    for axis, subplotseries in zip(axes, series):
        if subplotseries[0].startswith('Fe'):
            axis.set_ylabel('X$_{ion}$/X$_{tot}$')
            atomic_number = 26
            for ion_stage in [1, 2, 3, 4, 5]:
                ylist = []
                for modelgridindex in modeldata.index:
                    totalpop = estimators[(timestep, modelgridindex)]['populations']['total']
                    nionpop = estimators[(timestep, modelgridindex)]['populations'][(atomic_number, ion_stage)]
                    ylist.append(nionpop / totalpop)

                plotlabel = f'Fe {at.roman_numerals[ion_stage]}'
                axis.plot(xlist, ylist, linewidth=1.5, label=plotlabel)
        else:
            if subplotseries[0].startswith('heating'):
                axis.set_yscale('log')
            for variablename in subplotseries:
                serieslabel = f'{variablename} [{units[variablename]}]'
                if len(subplotseries) == 1:
                    axis.set_ylabel(serieslabel)
                    plotlabel = None
                else:
                    plotlabel = serieslabel

                ylist = []
                for modelgridindex in modeldata.index:
                    ylist.append(estimators[(timestep, modelgridindex)][variablename])

                axis.plot(xlist, ylist, linewidth=1.5, label=plotlabel)

        if len(subplotseries) != 1:
            axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})

    plotlabel = f'timestep {timestep}'
    axes[0].annotate(plotlabel, xy=(0.5, 0.04), xycoords='axes fraction',
                     horizontalalignment='center', verticalalignment='bottom', fontsize=12)

    # for axis in axes:
    #     # axis.set_xlim(xmin=270,xmax=300)
    #     # axis.set_ylim(ymin=-0.1,ymax=1.3)

    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    fig.savefig('plotestimators.pdf', format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
