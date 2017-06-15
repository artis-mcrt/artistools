#!/usr/bin/env python3
# import math
import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import artistools as at

# from astropy import constants as const


def get_units(variable):
    units = {
        'TR': 'K',
        'Te': 'K',
        'TJ': 'K',
        'nne': 'e-/cm3',
        'heating_gamma': 'erg/s/cm3',
        'velocity': 'km/s',
    }

    return units.get(variable, "?")


def parse_ion_row(row, outdict):
    variablename = row[0]
    atomic_number = int(row[1].split('=')[1])

    if variablename not in outdict:
        outdict[variablename] = {}

    for index, token in list(enumerate(row))[2::2]:
        ion_stage = int(token.rstrip(':'))
        value_thision = float(row[index + 1])

        outdict[variablename][(atomic_number, ion_stage)] = value_thision

        if variablename == 'populations':
            elpop = outdict.get(atomic_number, 0)
            outdict[variablename][atomic_number] = elpop + value_thision

            totalpop = outdict[variablename].get('total', 0)
            outdict[variablename]['total'] = totalpop + value_thision


def read_estimators(estimfiles, modeldata):
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
                    estimators[(timestep, modelgridindex)]['velocity'] = modeldata['velocity'][modelgridindex]
                    estimators[(timestep, modelgridindex)]['TR'] = float(row[5])
                    estimators[(timestep, modelgridindex)]['Te'] = float(row[7])
                    estimators[(timestep, modelgridindex)]['W'] = float(row[9])
                    estimators[(timestep, modelgridindex)]['TJ'] = float(row[11])
                    estimators[(timestep, modelgridindex)]['nne'] = float(row[15])

                elif row[1].startswith('Z='):
                    parse_ion_row(row, estimators[(timestep, modelgridindex)])

                elif row[0] == 'heating:':
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'heating_{token}'] = float(row[index + 1])

                elif row[0] == 'cooling:':
                    for index, token in list(enumerate(row))[1::2]:
                        estimators[(timestep, modelgridindex)][f'cooling_{token}'] = float(row[index + 1])

    return estimators


def plotionseries(seriestype, atomic_number, ion_stage, timestep, axis, mgilist, estimators, xlist, **plotkwargs):
    if seriestype == 'populations':
        axis.yaxis.set_major_locator(ticker.MultipleLocator(base=0.05))

    ylist = []
    for modelgridindex in mgilist:
        if seriestype == 'populations':
            totalpop = estimators[(timestep, modelgridindex)]['populations']['total']
            nionpop = estimators[(timestep, modelgridindex)]['populations'].get((atomic_number, ion_stage), 0.)
            ylist.append(nionpop / totalpop)
        else:
            ylist.append(estimators[(timestep, modelgridindex)][seriestype].get((atomic_number, ion_stage), 0.))

    plotlabel = f'{at.elsymbols[atomic_number]} {at.roman_numerals[ion_stage]}'

    ylist.insert(0, ylist[0])
    color = ['blue', 'green', 'red', 'cyan', 'purple'][ion_stage - 1]
    # or axis.step(where='pre', )
    axis.plot(xlist, ylist, linewidth=1.5, label=plotlabel, color=color, **plotkwargs)


def plot_multiseries(axis, xlist, serieslist, timestep, mgilist, estimators, **plotkwargs):
    seriestype, ionlist = serieslist
    for ionstr in ionlist:
        splitvariablename = ionstr.split(' ')
        atomic_number = at.get_atomic_number(splitvariablename[0])
        ionstage = at.decode_roman_numeral(splitvariablename[1])
        if seriestype == 'populations':
            axis.set_ylabel('X$_{ion}$/X$_{tot}$')
        else:
            axis.set_ylabel(seriestype)

        plotionseries(seriestype, atomic_number, ionstage, timestep, axis, mgilist, estimators, xlist, **plotkwargs)


def plot_singleseries(axis, xlist, variablename, singlevariableplot, timestep, mgilist, estimators, **plotkwargs):
    serieslabel = f'{variablename} [{get_units(variablename)}]'
    if singlevariableplot:
        axis.set_ylabel(serieslabel)
        plotlabel = None
        showlegend = False
    else:
        plotlabel = serieslabel
        showlegend = True

    ylist = []
    for modelgridindex in mgilist:
        ylist.append(estimators[(timestep, modelgridindex)][variablename])

    ylist.insert(0, ylist[0])
    dictcolors = {'Te': 'red', 'heating_gamma': 'blue'}
    axis.plot(xlist, ylist, linewidth=1.5, label=plotlabel, color=dictcolors.get(variablename, None), **plotkwargs)

    return showlegend


def plot_timestep(timestep, mgilist, estimators, series, outfilename, **plotkwargs):

    fig, axes = plt.subplots(len(series), 1, sharex=True, figsize=(5, 8),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    lastxvariable = ""
    for index, (axis, (xvariable, yvariables)) in enumerate(zip(axes, series)):
        showlegend = False

        if (lastxvariable != xvariable and lastxvariable != "") or index == len(axes) - 1:
            axis.set_xlabel(f'{xvariable} [{get_units(xvariable)}]')

        try:
            xlist = []
            for modelgridindex in mgilist:
                xlist.append(estimators[(timestep, modelgridindex)][xvariable])
        except KeyError:
            print("Unknown x variable")
            sys.exit()

        xlist = np.insert(xlist, 0, 0.)
        axis.set_xlim(xmin=0., xmax=xlist.max())

        try:
            if yvariables[0].startswith('heating'):
                axis.set_yscale('log')
        except AttributeError:
            pass

        for variablename in yvariables:
            if not hasattr(variablename, 'lower'):  # if it's a list, not a string
                showlegend = True
                plot_multiseries(axis, xlist, variablename, timestep, mgilist, estimators, **plotkwargs)
            else:
                showlegend = plot_singleseries(
                    axis, xlist, variablename, len(yvariables) == 1, timestep, mgilist, estimators, **plotkwargs)

        if showlegend:
            axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
        lastxvariable = xvariable

    # modelname = at.get_model_name(".")
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

    if '-' in args.timestep:
        timestepmin, timestepmax = [int(nts) for nts in args.timestep.split('-')]
    else:
        timestepmin = int(args.timestep)
        timestepmax = timestepmin

    # elementlist = at.get_composition_data('compositiondata.txt')
    modeldata, _ = at.get_modeldata('model.txt')
    # initalabundances = at.get_initialabundances1d('abundances.txt')

    estimators = read_estimators(['estimators_0000.out'], modeldata)

    series = [['velocity', ['heating_gamma']],
              ['velocity', ['Te']],
              ['velocity', [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V']]]],
            #   ['Te', [['recomb_coeff_R', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V']]]],
              ['velocity', ['TR']]]

    for timestep in range(timestepmin, timestepmax + 1):
        plot_timestep(timestep, modeldata.index, estimators, series, args.outputfile.format(timestep))


if __name__ == "__main__":
    main()
