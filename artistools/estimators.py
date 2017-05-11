#!/usr/bin/env python3
# import math

import argparse

import matplotlib.pyplot as plt

# import numpy as np
import artistools as at


# from astropy import constants as const


def parse_pop_row(row, popdict):
    atomic_number = int(row[1].split('=')[1])
    for index, token in list(enumerate(row))[2::2]:
        ion_stage = int(token.rstrip(':'))
        nionpopulation = float(row[index + 1])

        popdict[(atomic_number, ion_stage)] = nionpopulation

        elpop = popdict.get(atomic_number, 0)
        popdict[atomic_number] = elpop + nionpopulation

        totalpop = popdict.get('total', 0)
        popdict['total'] = totalpop + nionpopulation


def read_estimators(estimfiles):
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
                    parse_pop_row(row, estimators[(timestep, modelgridindex)]['populations'])

                elif row[0] == 'heating:':
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'heating_{token}'] = float(row[index + 1])

                elif row[0] == 'cooling:':
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'cooling_{token}'] = float(row[index + 1])

    return estimators


def plotion(atomic_number, ion_stage, timestep, axis, modeldata, estimators):
    ylist = []
    for modelgridindex in modeldata.index:
        totalpop = estimators[(timestep, modelgridindex)]['populations']['total']
        nionpop = estimators[(timestep, modelgridindex)]['populations'][(atomic_number, ion_stage)]
        ylist.append(nionpop / totalpop)

    plotlabel = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]}'
    axis.plot(modeldata['velocity'], ylist, linewidth=1.5, label=plotlabel)


def plot_timestep(timestep, modeldata, estimators, units, series, outfilename):
    fig, axes = plt.subplots(len(series), 1, sharex=True, figsize=(6, 8),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    axes[-1].set_xlabel(r'Velocity [km/s]')
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    for axis, subplotseries in zip(axes, series):
        if subplotseries[0].startswith('heating'):
            axis.set_yscale('log')
        for variablename in subplotseries:
            splitvariablename = variablename.split(' ')
            if len(splitvariablename) == 2:
                atomic_number = at.get_atomic_number(splitvariablename[0])
                ionstage = at.decode_roman_numeral(splitvariablename[1])
                if atomic_number > 0 and ionstage > 0:
                    axis.set_ylabel('X$_{ion}$/X$_{tot}$')
                    plotion(atomic_number, ionstage, timestep, axis, modeldata, estimators)
                    continue

            serieslabel = f'{variablename} [{units[variablename]}]'
            if len(subplotseries) == 1:
                axis.set_ylabel(serieslabel)
                plotlabel = None
            else:
                plotlabel = serieslabel

            ylist = []
            for modelgridindex in modeldata.index:
                ylist.append(estimators[(timestep, modelgridindex)][variablename])

            axis.plot(modeldata['velocity'], ylist, linewidth=1.5, label=plotlabel)

        if len(subplotseries) != 1:
            axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})

    plotlabel = f'Timestep {timestep}'
    time_days = float(at.get_timestep_time('spec.out', timestep))
    if time_days >= 0:
        plotlabel += f' (t={time_days:.2f}d)'
    fig.suptitle(plotlabel, fontsize=12)

    fig.savefig(outfilename, format='pdf')
    print(f'Saved {outfilename}')
    plt.close()


def main(argsraw=None):
    defaultoutputfile = 'plotestimators_{0:02d}.pdf'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS estimators.')
    # parser.add_argument('modelpath', nargs='?', default='',
    #                     help='Path to ARTIS folder')
    parser.add_argument('-timestep', default='56',
                        help='Timestep number to plot')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='Filename for PDF file')
    args = parser.parse_args(argsraw)

    units = {
        'TR': 'K',
        'Te': 'K',
        'TJ': 'K',
        'nne': 'e-/cm3',
        'heating_gamma': 'erg/s/cm3',
    }

    if '-' in args.timestep:
        timestepmin, timestepmax = [int(nts) for nts in args.timestep.split('-')]
    else:
        timestepmin = int(args.timestep)
        timestepmax = timestepmin

    modelname = at.get_model_name(".")

    # elementlist = at.get_composition_data('compositiondata.txt')
    modeldata, _ = at.get_modeldata('model.txt')
    # initalabundances = at.get_initialabundances1d('abundances.txt')

    estimators = read_estimators(['estimators_0000.out'])

    series = [['heating_gamma'], ['TR'], ['Te'], ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Co II', 'Co III']]

    for timestep in range(timestepmin, timestepmax + 1):
        plot_timestep(timestep, modeldata, estimators, units, series, args.outputfile.format(timestep))


if __name__ == "__main__":
    main()
