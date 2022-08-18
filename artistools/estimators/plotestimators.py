#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Functions for plotting artis estimators and internal structure.

Examples are temperatures, populations, heating/cooling rates.
"""
# import math
import argparse
import math
import multiprocessing
import os
# import re
import sys
# from collections import namedtuple
# from functools import lru_cache, partial, reduce
# from itertools import chain
from pathlib import Path
import argcomplete

import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
# import scipy.signal
# from astropy import constants as const

import artistools as at
import artistools.initial_composition
import artistools.nltepops

colors_tab10 = list(plt.get_cmap('tab10')(np.linspace(0, 1.0, 10)))

# reserve colours for these elements
elementcolors = {
    'Fe': colors_tab10[0],
    'Ni': colors_tab10[1],
    'Co': colors_tab10[2],
}


def get_elemcolor(atomic_number=None, elsymbol=None):
    """Get the colour of an element from the reserved color list (reserving a new one if needed)."""
    assert (atomic_number is None) != (elsymbol is None)
    if atomic_number is not None:
        elsymbol = at.get_elsymbol(atomic_number)

    # assign a new colour to this element if needed

    return elementcolors.setdefault(elsymbol, colors_tab10[len(elementcolors)])


def get_ylabel(variable):
    if variable in at.estimators.variablelongunits:
        return at.estimators.variablelongunits[variable]
    if variable in at.estimators.variableunits:
        return f'[{at.estimators.variableunits[variable]}]'
    if variable.split('_')[0] in at.estimators.variableunits:
        return f'[{at.estimators.variableunits[variable.split("_")[0]]}]'
    return ''


def plot_init_abundances(ax, xlist, specieslist, mgilist, modelpath, seriestype, dfalldata=None, args=None,
                         **plotkwargs):
    assert len(xlist) - 1 == len(mgilist)

    if seriestype == 'initabundances':
        mergemodelabundata, _, _ = at.inputmodel.get_modeldata(modelpath, get_abundances=True)
    elif seriestype == 'initmasses':
        mergemodelabundata = at.initial_composition.get_model_abundances_Msun_1D(modelpath)

    for speciesstr in specieslist:
        splitvariablename = speciesstr.split('_')
        elsymbol = splitvariablename[0].strip('0123456789')
        atomic_number = at.get_atomic_number(elsymbol)
        if seriestype == 'initabundances':
            ax.set_ylim(1e-20, 1.0)
            ax.set_ylabel('Initial mass fraction')
            valuetype = 'X_'
        elif seriestype == 'initmasses':
            ax.set_ylabel(r'Initial mass [M$_\odot$]')
            valuetype = 'mass_X_'

        ylist = []
        linelabel = speciesstr
        linestyle = '-'
        for modelgridindex in mgilist:
            if speciesstr.lower() in ['ni_56', 'ni56', '56ni']:
                yvalue = mergemodelabundata.loc[modelgridindex][f'{valuetype}Ni56']
                linelabel = '$^{56}$Ni'
                linestyle = '--'
            elif speciesstr.lower() in ['ni_stb', 'ni_stable']:
                yvalue = (mergemodelabundata.loc[modelgridindex][f'{valuetype}{elsymbol}'] -
                          mergemodelabundata.loc[modelgridindex]['X_Ni56'])
                linelabel = 'Stable Ni'
            elif speciesstr.lower() in ['co_56', 'co56', '56co']:
                yvalue = mergemodelabundata.loc[modelgridindex][f'{valuetype}Co56']
                linelabel = '$^{56}$Co'
            elif speciesstr.lower() in ['fegrp', 'ffegroup']:
                yvalue = mergemodelabundata.loc[modelgridindex][f'{valuetype}Fegroup']
            else:
                yvalue = mergemodelabundata.loc[modelgridindex][f'{valuetype}{elsymbol}']
            ylist.append(yvalue)

        if dfalldata is not None:
            dfalldata['initabundances.' + speciesstr] = ylist

        ylist.insert(0, ylist[0])
        # or ax.step(where='pre', )
        color = get_elemcolor(atomic_number=atomic_number)

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, linewidth=1.5, label=linelabel, linestyle=linestyle, color=color, **plotkwargs)

        # if args.yscale == 'log':
        #     ax.set_yscale('log')


def plot_average_ionisation_excitation(
        ax, xlist, seriestype, params, timestepslist, mgilist,
        estimators, modelpath, dfalldata=None, args=None, **plotkwargs):
    if seriestype == 'averageionisation':
        ax.set_ylabel('Average ion charge')
    elif seriestype == 'averageexcitation':
        ax.set_ylabel('Average excitation [eV]')
    else:
        raise ValueError()

    arr_tdelta = at.get_timestep_times_float(modelpath, loc='delta')
    for paramvalue in params:
        if seriestype == 'averageionisation':
            atomic_number = at.get_atomic_number(paramvalue)
        else:
            atomic_number = at.get_atomic_number(paramvalue.split(' ')[0])
            ion_stage = at.decode_roman_numeral(paramvalue.split(' ')[1])
        ylist = []
        for modelgridindex, timesteps in zip(mgilist, timestepslist):
            valuesum = 0
            tdeltasum = 0
            for timestep in timesteps:

                if seriestype == 'averageionisation':
                    valuesum += (at.estimators.get_averageionisation(
                        estimators[(timestep, modelgridindex)]['populations'], atomic_number) * arr_tdelta[timestep])
                elif seriestype == 'averageexcitation':
                    T_exc = estimators[(timestep, modelgridindex)]['Te']
                    valuesum += (at.estimators.get_averageexcitation(
                        modelpath, modelgridindex, timestep, atomic_number, ion_stage, T_exc) * arr_tdelta[timestep])
                tdeltasum += arr_tdelta[timestep]

            ylist.append(valuesum / tdeltasum)

        color = get_elemcolor(atomic_number=atomic_number)

        if dfalldata is not None:
            dfalldata[seriestype + '.' + paramvalue] = ylist

        ylist.insert(0, ylist[0])

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, label=paramvalue, color=color, **plotkwargs)


def plot_levelpop(
        ax, xlist, seriestype, params, timestepslist, mgilist,
        estimators, modelpath, dfalldata=None, args=None, **plotkwargs):
    import artistools.plottools
    if seriestype == 'levelpopulation_dn_on_dvel':
        ax.set_ylabel('dN/dV [{}km$^{{-1}}$ s]')
        ax.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(ax.get_ylabel(), useMathText=True))
    elif seriestype == 'levelpopulation':
        ax.set_ylabel('X$_{{i}}$ [{}/cm3]')
        ax.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(ax.get_ylabel(), useMathText=True))
    else:
        raise ValueError()

    modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
    modeldata.eval('modelcellvolume = cellmass_grams / (10 ** logrho)', inplace=True)

    adata = at.atomic.get_levels(modelpath)

    arr_tdelta = at.get_timestep_times_float(modelpath, loc='delta')
    for paramvalue in params:
        paramsplit = paramvalue.split(' ')
        atomic_number = at.get_atomic_number(paramsplit[0])
        ion_stage = at.decode_roman_numeral(paramsplit[1])
        levelindex = int(paramsplit[2])

        ionlevels = adata.query('Z == @atomic_number and ion_stage == @ion_stage').iloc[0].levels
        levelname = ionlevels.iloc[levelindex].levelname
        label = (
            f'{at.get_ionstring(atomic_number, ion_stage, spectral=False)} level {levelindex}:'
            f' {at.nltepops.texifyconfiguration(levelname)}')

        print(f'plot_levelpop {label}')

        # level index query goes outside for caching granularity reasons
        dfnltepops = at.nltepops.read_files(
            modelpath,
            dfquery=f'Z=={atomic_number:.0f} and ion_stage=={ion_stage:.0f}').query('level==@levelindex')

        ylist = []
        for modelgridindex, timesteps in zip(mgilist, timestepslist):
            valuesum = 0
            tdeltasum = 0
            # print(f'modelgridindex {modelgridindex} timesteps {timesteps}')

            for timestep in timesteps:
                levelpop = dfnltepops.query(
                    'modelgridindex==@modelgridindex and timestep==@timestep and Z==@atomic_number'
                    ' and ion_stage==@ion_stage and level==@levelindex').iloc[0].n_NLTE

                valuesum += levelpop * arr_tdelta[timestep]
                tdeltasum += arr_tdelta[timestep]

            if seriestype == 'levelpopulation_dn_on_dvel':
                deltav = modeldata.loc[modelgridindex].velocity_outer - modeldata.loc[modelgridindex].velocity_inner
                ylist.append(valuesum / tdeltasum * modeldata.loc[modelgridindex].modelcellvolume / deltav)
            else:
                ylist.append(valuesum / tdeltasum)

        if dfalldata is not None:
            elsym = at.get_elsymbol(atomic_number).lower()
            if seriestype == 'levelpopulation_dn_on_dvel':
                colname = f'nlevel_on_dv_{elsym}_ionstage{ion_stage}_level{levelindex}'
            else:
                colname = f'nnlevel_{elsym}_ionstage{ion_stage}_level{levelindex}'
            dfalldata[colname] = ylist

        ylist.insert(0, ylist[0])

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, label=label, **plotkwargs)


def plot_multi_ion_series(
        ax, xlist, seriestype, ionlist, timestepslist, mgilist, estimators,
        modelpath, dfalldata=None, args=None, **plotkwargs):
    """Plot an ion-specific property, e.g., populations."""
    assert len(xlist) - 1 == len(mgilist) == len(timestepslist)
    # if seriestype == 'populations':
    #     ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.10))

    plotted_something = False

    def get_iontuple(ionstr):
        if ionstr in at.get_elsymbolslist():
            return (at.get_atomic_number(ionstr), 'ALL')
        elif ' ' in ionstr:
            return (at.get_atomic_number(ionstr.split(' ')[0]), at.decode_roman_numeral(ionstr.split(' ')[1]))
        elif ionstr.rstrip('-0123456789') in at.get_elsymbolslist():
            atomic_number = at.get_atomic_number(ionstr.rstrip('-0123456789'))
            return (atomic_number, ionstr)
        else:
            atomic_number = at.get_atomic_number(ionstr.split('_')[0])
            return (atomic_number, ionstr)

    # decoded into atomic number and parameter, e.g., [(26, 1), (26, 2), (26, 'ALL'), (26, 'Fe56')]
    iontuplelist = [get_iontuple(ionstr) for ionstr in ionlist]
    iontuplelist.sort()
    print(f'Subplot with ions: {iontuplelist}')

    missingions = set()
    try:
        if args.classicartis:
            import artistools.estimators.estimators_classic
            compositiondata = artistools.estimators.estimators_classic.get_atomic_composition(modelpath)
        else:
            compositiondata = at.get_composition_data(modelpath)
        for atomic_number, ion_stage in iontuplelist:
            if not hasattr(ion_stage, 'lower') and not args.classicartis and compositiondata.query(
                    'Z == @atomic_number '
                    '& lowermost_ionstage <= @ion_stage '
                    '& uppermost_ionstage >= @ion_stage').empty:
                missingions.add((atomic_number, ion_stage))

    except FileNotFoundError:
        print("WARNING: Could not read an ARTIS compositiondata.txt file")
        for atomic_number, ion_stage in iontuplelist:
            mgits = (timestepslist[0][0], mgilist[0])
            if (atomic_number, ion_stage) not in estimators[mgits]['populations']:
                missingions.add((atomic_number, ion_stage))

    if missingions:
        print(f" Warning: Can't plot {seriestype} for {missingions} "
              f"because these ions are not in compositiondata.txt")

    prev_atomic_number = iontuplelist[0][0]
    colorindex = 0
    for atomic_number, ion_stage in iontuplelist:
        if (atomic_number, ion_stage) in missingions:
            continue

        if atomic_number != prev_atomic_number:
            colorindex += 1

        if seriestype == 'populations':
            if args.ionpoptype == 'absolute':
                ax.set_ylabel('X$_{i}$ [/cm3]')
            elif args.ionpoptype == 'elpop':
                # elsym = at.get_elsymbol(atomic_number)
                ax.set_ylabel(r'X$_{i}$/X$_{\rm element}$')
            elif args.ionpoptype == 'totalpop':
                ax.set_ylabel(r'X$_{i}$/X$_{rm tot}$')
            else:
                assert False
        else:
            ax.set_ylabel(at.estimators.dictlabelreplacements.get(seriestype, seriestype))

        ylist = []
        for modelgridindex, timesteps in zip(mgilist, timestepslist):
            if seriestype == 'populations':
                # if (atomic_number, ion_stage) not in estim['populations']:
                #     print(f'Note: population for {(atomic_number, ion_stage)} not in estimators for '
                #           f'cell {modelgridindex} timesteps {timesteps}')

                try:
                    estimpop = at.estimators.get_averaged_estimators(
                        modelpath, estimators, timesteps, modelgridindex, ['populations'])
                except KeyError:
                    ylist.append(float('nan'))
                    continue

                if ion_stage == 'ALL':
                    nionpop = estimpop.get((atomic_number), 0.)
                elif hasattr(ion_stage, 'lower') and ion_stage.startswith(at.get_elsymbol(atomic_number)):
                    nionpop = estimpop.get(ion_stage, 0.)
                else:
                    nionpop = estimpop.get((atomic_number, ion_stage), 0.)

                try:
                    if args.ionpoptype == 'absolute':
                        yvalue = nionpop  # Plot as fraction of element population
                    elif args.ionpoptype == 'elpop':
                        elpop = estimpop.get(atomic_number, 0.)
                        yvalue = nionpop / elpop  # Plot as fraction of element population
                    elif args.ionpoptype == 'totalpop':
                        totalpop = estimpop['total']
                        yvalue = nionpop / totalpop  # Plot as fraction of total population
                    else:
                        assert False
                except ZeroDivisionError:
                    yvalue = 0.

                ylist.append(yvalue)

            # elif seriestype == 'Alpha_R':
            #     ylist.append(estim['Alpha_R*nne'].get((atomic_number, ion_stage), 0.) / estim['nne'])
            # else:
            #     ylist.append(estim[seriestype].get((atomic_number, ion_stage), 0.))
            else:
                # this is very slow!
                try:
                    estim = at.estimators.get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, [])
                except KeyError:
                    ylist.append(float('nan'))
                    continue

                dictvars = {}
                for k, value in estim.items():
                    if isinstance(value, dict):
                        dictvars[k] = value.get((atomic_number, ion_stage), 0.)
                    else:
                        dictvars[k] = value

                # dictvars will now define things like 'Te', 'TR',
                # as well as 'populations' which applies to the current ion

                try:
                    yvalue = eval(seriestype, {"__builtins__": math}, dictvars)
                except ZeroDivisionError:
                    yvalue = float('NaN')
                ylist.append(yvalue)

        if hasattr(ion_stage, 'lower') and ion_stage != 'ALL':
            plotlabel = ion_stage
        else:
            plotlabel = at.get_ionstring(atomic_number, ion_stage, spectral=False)

        color = get_elemcolor(atomic_number=atomic_number)

        # linestyle = ['-.', '-', '--', (0, (4, 1, 1, 1)), ':'] + [(0, x) for x in dashes_list][ion_stage - 1]
        if ion_stage == 'ALL':
            dashes = ()
            linewidth = 1.0
        else:
            if hasattr(ion_stage, 'lower') and ion_stage.endswith('stable'):
                index = 8
            elif hasattr(ion_stage, 'lower'):
                # isotopic abundance, use the mass number
                index = int(ion_stage.lstrip(at.get_elsymbol(atomic_number)))
            else:
                index = ion_stage

            dashes_list = [(3, 1, 1, 1), (), (1.5, 1.5), (6, 3), (1, 3)]
            dashes = dashes_list[(index - 1) % len(dashes_list)]
            linewidth_list = [1.0, 1.0, 1.0, 0.7, 0.7]
            linewidth = linewidth_list[(index - 1) % len(linewidth_list)]
            # color = ['blue', 'green', 'red', 'cyan', 'purple', 'grey', 'brown', 'orange'][index - 1]

            if args.colorbyion:
                color = f'C{index - 1 % 10}'
                # plotlabel = f'{at.get_elsymbol(atomic_number)} {at.roman_numerals[ion_stage]}'
                dashes = ()

        # assert colorindex < 10
        # color = f'C{colorindex}'
        # or ax.step(where='pre', )

        if dfalldata is not None:
            elsym = at.get_elsymbol(atomic_number).lower()
            if args.ionpoptype == 'absolute':
                colname = f'nnion_{elsym}_ionstage{ion_stage}'
            elif args.ionpoptype == 'elpop':
                colname = f'nnion_over_nnelem_{elsym}_ionstage{ion_stage}'
            elif args.ionpoptype == 'totalpop':
                colname = f'nnion_over_nntot_{elsym}_ionstage{ion_stage}'
            dfalldata[colname] = ylist

        ylist.insert(0, ylist[0])

        xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)

        ax.plot(xlist, ylist, linewidth=linewidth, label=plotlabel, color=color, dashes=dashes, **plotkwargs)
        prev_atomic_number = atomic_number
        plotted_something = True

    if plotted_something:
        ax.set_yscale(args.yscale)
        if args.yscale == 'log':
            ymin, ymax = ax.get_ylim()
            new_ymax = ymax * 10 ** (0.3 * math.log10(ymax / ymin))
            if ymin > 0 and new_ymax > ymin and np.isfinite(new_ymax):
                ax.set_ylim(ymin, new_ymax)


def plot_series(ax, xlist, variablename, showlegend, timestepslist, mgilist,
                modelpath, estimators, args, nounits=False, dfalldata=None, **plotkwargs):
    """Plot something like Te or TR."""
    assert len(xlist) - 1 == len(mgilist) == len(timestepslist)
    formattedvariablename = at.estimators.dictlabelreplacements.get(variablename, variablename)
    serieslabel = f'{formattedvariablename}'
    if not nounits:
        serieslabel += at.estimators.get_units_string(variablename)

    if showlegend:
        linelabel = serieslabel
    else:
        ax.set_ylabel(serieslabel)
        linelabel = None

    ylist = []
    for modelgridindex, timesteps in zip(mgilist, timestepslist):
        estimavg = at.estimators.get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, [])
        try:
            ylist.append(eval(variablename, {"__builtins__": math}, estimavg))
        except KeyError:
            if (timesteps[0], modelgridindex) in estimators:
                print(f"Undefined variable: {variablename} in cell {modelgridindex}")
            else:
                print(f'No data for cell {modelgridindex}')
            sys.exit()

    try:
        if math.log10(max(ylist) / min(ylist)) > 2:
            ax.set_yscale('log')
    except ZeroDivisionError:
        ax.set_yscale('log')

    dictcolors = {
        'Te': 'red',
        # 'heating_gamma': 'blue',
        # 'cooling_adiabatic': 'blue'
    }

    # print out the data to stdout. Maybe want to add a CSV export option at some point?
    # print(f'#cellidorvelocity {variablename}\n' + '\n'.join([f'{x}  {y}' for x, y in zip(xlist, ylist)]))

    if dfalldata is not None:
        dfalldata[variablename] = ylist

    ylist.insert(0, ylist[0])

    xlist, ylist = at.estimators.apply_filters(xlist, ylist, args)

    ax.plot(xlist, ylist, linewidth=1.5, label=linelabel, color=dictcolors.get(variablename, None), **plotkwargs)


def get_xlist(xvariable, allnonemptymgilist, estimators, timestepslist, modelpath, args):
    if xvariable in ['cellid', 'modelgridindex']:
        if args.xmax >= 0:
            mgilist_out = [mgi for mgi in allnonemptymgilist if mgi <= args.xmax]
        else:
            mgilist_out = allnonemptymgilist
        xlist = mgilist_out
        timestepslist_out = timestepslist
    elif xvariable == 'timestep':
        mgilist_out = allnonemptymgilist
        xlist = timestepslist
        timestepslist_out = timestepslist
    elif xvariable == 'time':
        mgilist_out = allnonemptymgilist
        timearray = at.get_timestep_times_float(modelpath)
        xlist = [np.mean([timearray[ts] for ts in tslist]) for tslist in timestepslist]
        timestepslist_out = timestepslist
    else:
        xlist = []
        mgilist_out = []
        timestepslist_out = []
        for modelgridindex, timesteps in zip(allnonemptymgilist, timestepslist):
            xvalue = at.estimators.get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, xvariable)
            xlist.append(xvalue)
            mgilist_out.append(modelgridindex)
            timestepslist_out.append(timesteps)
            if args.xmax > 0 and xvalue > args.xmax:
                break

    xlist, mgilist_out, timestepslist_out = zip(
        *list(sorted(zip(xlist, mgilist_out, timestepslist_out))))

    assert len(xlist) == len(mgilist_out) == len(timestepslist_out)

    return list(xlist), list(mgilist_out), list(timestepslist_out)


def plot_subplot(ax, timestepslist, xlist, plotitems, mgilist, modelpath,
                 estimators, dfalldata=None, args=None, **plotkwargs):
    """Make plot from ARTIS estimators."""
    # these three lists give the x value, modelgridex, and a list of timesteps (for averaging) for each plot of the plot
    assert len(xlist) - 1 == len(mgilist) == len(timestepslist)
    showlegend = False

    ylabel = 'UNDEFINED'
    sameylabel = True
    for variablename in plotitems:
        if not isinstance(variablename, str):
            pass
        elif ylabel == 'UNDEFINED':
            ylabel = get_ylabel(variablename)
        elif ylabel != get_ylabel(variablename):
            sameylabel = False
            break

    for plotitem in plotitems:
        if isinstance(plotitem, str):
            showlegend = len(plotitems) > 1 or len(variablename) > 20
            plot_series(ax, xlist, plotitem, showlegend, timestepslist, mgilist, modelpath,
                        estimators, args, nounits=sameylabel, dfalldata=dfalldata, **plotkwargs)
            if showlegend and sameylabel:
                ax.set_ylabel(ylabel)
        else:  # it's a sequence of values
            showlegend = True
            seriestype, params = plotitem

            if seriestype in ['initabundances', 'initmasses']:
                plot_init_abundances(ax, xlist, params, mgilist, modelpath, seriestype, dfalldata=dfalldata, args=args)

            elif seriestype == 'levelpopulation' or seriestype.startswith('levelpopulation_'):
                plot_levelpop(
                    ax, xlist, seriestype, params, timestepslist, mgilist, estimators, modelpath,
                    dfalldata=dfalldata, args=args)

            elif seriestype in ['averageionisation', 'averageexcitation']:
                plot_average_ionisation_excitation(
                    ax, xlist, seriestype, params, timestepslist, mgilist, estimators, modelpath,
                    dfalldata=dfalldata, args=args)

            elif seriestype == '_ymin':
                ax.set_ylim(bottom=params)

            elif seriestype == '_ymax':
                ax.set_ylim(top=params)

            elif seriestype == '_yscale':
                ax.set_yscale(params)

            else:
                seriestype, ionlist = plotitem
                plot_multi_ion_series(ax, xlist, seriestype, ionlist, timestepslist, mgilist, estimators,
                                      modelpath, dfalldata, args, **plotkwargs)

    ax.tick_params(right=True)
    if showlegend and not args.nolegend:
        if plotitems[0][0] == 'populations' and args.yscale == 'log':
            ax.legend(loc='best', handlelength=2, ncol=math.ceil(len(plotitems[0][1]) / 2.),
                      frameon=False, numpoints=1)
        else:
            ax.legend(loc='best', handlelength=2,
                      frameon=False, numpoints=1,)  # prop={'size': 9})


def make_plot(modelpath, timestepslist_unfiltered, allnonemptymgilist, estimators, xvariable, plotlist,
              args, **plotkwargs):
    modelname = at.get_model_name(modelpath)
    fig, axes = plt.subplots(nrows=len(plotlist), ncols=1, sharex=True,
                             figsize=(args.figscale * at.config['figwidth'] * args.scalefigwidth,
                                      args.figscale * at.config['figwidth'] * 0.5 * len(plotlist)),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    if len(plotlist) == 1:
        axes = [axes]

    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    if not args.hidexlabel:
        axes[-1].set_xlabel(f'{xvariable}{at.estimators.get_units_string(xvariable)}')
    xlist, mgilist, timestepslist = get_xlist(
        xvariable, allnonemptymgilist, estimators, timestepslist_unfiltered, modelpath, args)

    dfalldata = pd.DataFrame(index=mgilist)
    dfalldata.index.name = "modelgridindex"
    dfalldata[xvariable] = xlist

    if xvariable.startswith('velocity'):
        xlist = np.insert(xlist, 0, 0.)
    else:
        xlist = np.insert(xlist, 0, xlist[0])

    xmin = args.xmin if args.xmin >= 0 else min(xlist)
    xmax = args.xmax if args.xmax > 0 else max(xlist)

    for ax, plotitems in zip(axes, plotlist):
        ax.set_xlim(left=xmin, right=xmax)
        plot_subplot(ax, timestepslist, xlist, plotitems, mgilist,
                     modelpath, estimators, dfalldata=dfalldata, args=args, **plotkwargs)

    if len(set(mgilist)) == 1 and len(timestepslist[0]) > 1:  # single grid cell versus time plot
        figure_title = f'{modelname}\nCell {mgilist[0]}'

        defaultoutputfile = Path('plotestimators_cell{modelgridindex:03d}.pdf')
        if os.path.isdir(args.outputfile):
            args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

        outfilename = str(args.outputfile).format(modelgridindex=mgilist[0])

    else:
        timeavg = (args.timemin + args.timemax) / 2.
        if args.multiplot and not args.classicartis:
            tdays = estimators[(timestepslist[0][0], mgilist[0])]['tdays']
            figure_title = f'{modelname}\nTimestep {timestepslist[0]} ({tdays:.2f}d)'
        elif args.multiplot and args.classicartis:
            timedays = float(at.get_timestep_time(modelpath, timestepslist[0])[0])
            figure_title = f'{modelname}\nTimestep {timestepslist[0]} ({timedays:.2f}d)'
        else:
            figure_title = f'{modelname}\nTimestep {timestepslist[0]} ({timeavg:.2f}d)'

        defaultoutputfile = Path('plotestimators_ts{timestep:02d}_{timeavg:.0f}d.pdf')
        if os.path.isdir(args.outputfile):
            args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

        outfilename = str(args.outputfile).format(timestep=timestepslist[0][0], timeavg=timeavg)

    if not args.notitle:
        axes[0].set_title(figure_title, fontsize=11)
    # plt.suptitle(figure_title, fontsize=11, verticalalignment='top')

    if args.write_data:
        dfalldata.sort_index(inplace=True)
        dataoutfilename = Path(outfilename).with_suffix('.txt')
        dfalldata.to_csv(dataoutfilename)
        print(f'Saved {dataoutfilename}')

    fig.savefig(outfilename)
    print(f'Saved {outfilename}')

    if args.show:
        plt.show()
    else:
        plt.close()

    return outfilename


def plot_recombrates(modelpath, estimators, atomic_number, ion_stage_list, **plotkwargs):
    fig, axes = plt.subplots(
        nrows=len(ion_stage_list), ncols=1, sharex=True, figsize=(5, 8),
        tight_layout={"pad": 0.5, "w_pad": 0.0, "h_pad": 0.0})
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axes[-1].set_xlabel('T_e in kelvins')

    recombcalibrationdata = at.atomic.get_ionrecombratecalibration(modelpath)

    for ax, ion_stage in zip(axes, ion_stage_list):

        ionstr = (f'{at.get_elsymbol(atomic_number)} {at.roman_numerals[ion_stage]} to '
                  f'{at.roman_numerals[ion_stage - 1]}')

        listT_e = []
        list_rrc = []
        list_rrc2 = []
        for _, dicttimestepmodelgrid in estimators.items():
            if (not dicttimestepmodelgrid['emptycell']
                    and (atomic_number, ion_stage) in dicttimestepmodelgrid['RRC_LTE_Nahar']):
                listT_e.append(dicttimestepmodelgrid['Te'])
                list_rrc.append(dicttimestepmodelgrid['RRC_LTE_Nahar'][(atomic_number, ion_stage)])
                list_rrc2.append(dicttimestepmodelgrid['Alpha_R'][(atomic_number, ion_stage)])

        if not list_rrc:
            continue

        # sort the pairs by temperature ascending
        listT_e, list_rrc, list_rrc2 = zip(*sorted(zip(listT_e, list_rrc, list_rrc2), key=lambda x: x[0]))

        # markersize=4, marker='s',
        ax.plot(listT_e, list_rrc, linewidth=2, label=f'{ionstr} ARTIS RRC_LTE_Nahar', **plotkwargs)
        ax.plot(listT_e, list_rrc2, linewidth=2, label=f'{ionstr} ARTIS Alpha_R', **plotkwargs)

        try:
            dfrates = recombcalibrationdata[(atomic_number, ion_stage)].query(
                "T_e > @T_e_min & T_e < @T_e_max",
                local_dict={'T_e_min': min(listT_e), 'T_e_max': max(listT_e)})

            ax.plot(dfrates.T_e, dfrates.rrc_total, linewidth=2,
                    label=ionstr + " (calibration)", markersize=6, marker='s', **plotkwargs)
        except KeyError:
            pass

        # rrcfiles = glob.glob(
        #     f'/Users/lshingles/Library/Mobile Documents/com~apple~CloudDocs/GitHub/'
        #     f'artis-atomic/atomic-data-nahar/{at.get_elsymbol(atomic_number).lower()}{ion_stage - 1}.rrc*.txt')
        # if rrcfiles:
        #     dfrecombrates = get_ionrecombrates_fromfile(rrcfiles[0])
        #
        #     dfrecombrates.query("logT > @logT_e_min & logT < @logT_e_max",
        #                         local_dict={'logT_e_min': math.log10(min(listT_e)),
        #                                     'logT_e_max': math.log10(max(listT_e))}, inplace=True)
        #
        #     listT_e_Nahar = [10 ** x for x in dfrecombrates['logT'].values]
        #     ax.plot(listT_e_Nahar, dfrecombrates['RRC_total'], linewidth=2,
        #             label=ionstr + " (Nahar)", markersize=6, marker='s', **plotkwargs)

        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})

    # modelname = at.get_model_name(".")
    # plotlabel = f'Timestep {timestep}'
    # time_days = float(at.get_timestep_time('spec.out', timestep))
    # if time_days >= 0:
    #     plotlabel += f' (t={time_days:.2f}d)'
    # fig.suptitle(plotlabel, fontsize=12)
    elsymbol = at.get_elsymbol(atomic_number)
    outfilename = f"plotestimators_recombrates_{elsymbol}.pdf"
    fig.savefig(outfilename)
    print(f'Saved {outfilename}')
    plt.close()


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Paths to ARTIS folder (or virtual path e.g. codecomparison/ddc10/cmfgen)')

    parser.add_argument('--recombrates', action='store_true',
                        help='Make a recombination rate plot')

    parser.add_argument('-modelgridindex', '-cell', '-mgi', type=int, default=-1,
                        help='Modelgridindex for time evolution plot')

    parser.add_argument('-timestep', '-ts', nargs='?',
                        help='Timestep number for internal structure plot')

    parser.add_argument('-timedays', '-time', '-t', nargs='?',
                        help='Time in days to plot for internal structure plot')

    parser.add_argument('-timemin', type=float,
                        help='Lower time in days')

    parser.add_argument('-timemax', type=float,
                        help='Upper time in days')

    parser.add_argument('--multiplot', action='store_true',
                        help='Make multiple plots for timesteps in range')

    parser.add_argument('-x',
                        help='Horizontal axis variable, e.g. cellid, velocity, timestep, or time')

    parser.add_argument('-xmin', type=float, default=-1,
                        help='Plot range: minimum x value')

    parser.add_argument('-xmax', type=float, default=-1,
                        help='Plot range: maximum x value')

    parser.add_argument('-yscale', default='log', choices=['log', 'linear'],
                        help='Set yscale to log or linear (default log)')

    parser.add_argument('--hidexlabel', action='store_true',
                        help='Hide the bottom horizontal axis label')

    parser.add_argument('-filtermovingavg', type=int, default=0,
                        help='Smoothing length (1 is same as none)')

    parser.add_argument('-filtersavgol', nargs=2,
                        help='Savitzky–Golay filter. Specify the window_length and polyorder.'
                        'e.g. -filtersavgol 5 3')

    parser.add_argument('--notitle', action='store_true',
                        help='Suppress the top title from the plot')

    parser.add_argument('-plotlist', type=list, default=[],
                        help='Plot list (when calling from Python only)')

    parser.add_argument('-ionpoptype', default='elpop', choices=['absolute', 'totalpop', 'elpop'],
                        help=(
                            'Plot absolute ion populations, or ion populations as a'
                            ' fraction of total or element population'))

    parser.add_argument('--nolegend', action='store_true',
                        help='Suppress the legend from the plot')

    parser.add_argument('-figscale', type=float, default=1.,
                        help='Scale factor for plot area. 1.0 is for single-column')

    parser.add_argument('-scalefigwidth', type=float, default=1.,
                        help='Scale factor for plot width.')

    parser.add_argument('--show', action='store_true',
                        help='Show plot before quitting')

    parser.add_argument('--write_data', action='store_true',
                        help='Save data used to generate the plot in a CSV file')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path, default=Path(),
                        help='Filename for PDF file')

    parser.add_argument('--colorbyion', action='store_true',
                        help='Populations plots colored by ion rather than element')

    parser.add_argument('--classicartis', action='store_true',
                        help='Flag to show using output from classic ARTIS branch')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Plot ARTIS estimators.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    modelpath = Path(args.modelpath)

    modelname = at.get_model_name(modelpath)

    if not args.timedays and not args.timestep and args.modelgridindex > -1:
        args.timestep = f'0-{len(at.get_timestep_times_float(modelpath)) - 1}'

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays)

    print(f"Plotting estimators for '{modelname}' timesteps {timestepmin} to {timestepmax} "
          f"({args.timemin:.1f} to {args.timemax:.1f}d)")

    timesteps_included = list(range(timestepmin, timestepmax + 1))

    if args.classicartis:
        import artistools.estimators.estimators_classic
        modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
        estimators = artistools.estimators.estimators_classic.read_classic_estimators(modelpath, modeldata)
    else:
        estimators = at.estimators.read_estimators(
            modelpath=modelpath, modelgridindex=args.modelgridindex, timestep=tuple(timesteps_included))

    for ts in reversed(timesteps_included):
        tswithdata = [ts for (ts, mgi) in estimators.keys()]
        for ts in timesteps_included:
            if ts not in tswithdata:
                timesteps_included.remove(ts)
                print(f"ts {ts} requested but no data found. Removing.")

    if not timesteps_included:
        print("No timesteps with data are included")
        return

    if args.plotlist:
        plotlist = args.plotlist
    else:
        plotlist = [
            # [['initabundances', ['Fe', 'Ni_stable', 'Ni_56']]],
            # ['heating_dep', 'heating_coll', 'heating_bf', 'heating_ff',
            #  ['_yscale', 'linear']],
            # ['cooling_adiabatic', 'cooling_coll', 'cooling_fb', 'cooling_ff',
            #  ['_yscale', 'linear']],
            # [['initmasses', ['Ni_56', 'He', 'C', 'Mg']]],
            # ['heating_gamma/gamma_dep'],
            # ['nne'],
            ['TR', 'Te', 'TJ',
             ['_yscale', 'linear']],
            # ['Te'],
            # [['averageionisation', ['Fe', 'Ni']]],
            # [['averageexcitation', ['Fe II', 'Fe III']]],
            # [['populations', ['Sr89', 'Sr90', 'Sr91', 'Sr92', 'Sr93', 'Sr94', 'Sr95']],
            #  ['_ymin', 1e-3], ['_ymax', 5]],
            [['populations', ['Fe', 'Co', 'Ni', 'Sr', 'Nd', 'U']]],
            # [['populations', ['He I', 'He II', 'He III']]],
            # [['populations', ['C I', 'C II', 'C III', 'C IV', 'C V']]],
            # [['populations', ['O I', 'O II', 'O III', 'O IV']]],
            # [['populations', ['Ne I', 'Ne II', 'Ne III', 'Ne IV', 'Ne V']]],
            # [['populations', ['Si I', 'Si II', 'Si III', 'Si IV', 'Si V']]],
            # [['populations', ['Cr I', 'Cr II', 'Cr III', 'Cr IV', 'Cr V']]],
            # [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Fe VI', 'Fe VII', 'Fe VIII']]],
            # [['populations', ['Co I', 'Co II', 'Co III', 'Co IV', 'Co V', 'Co VI', 'Co VII']]],
            # [['populations', ['Ni I', 'Ni II', 'Ni III', 'Ni IV', 'Ni V', 'Ni VI', 'Ni VII']]],
            # [['populations', ['Fe II', 'Fe III', 'Co II', 'Co III', 'Ni II', 'Ni III']]],
            # [['populations', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
            # [['RRC_LTE_Nahar', ['Fe II', 'Fe III', 'Fe IV', 'Fe V']]],
            # [['RRC_LTE_Nahar', ['Co II', 'Co III', 'Co IV', 'Co V']]],
            # [['RRC_LTE_Nahar', ['Ni I', 'Ni II', 'Ni III', 'Ni IV', 'Ni V', 'Ni VI', 'Ni VII']]],
            # [['Alpha_R / RRC_LTE_Nahar', ['Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni III']]],
            # [['gamma_NT', ['Fe I', 'Fe II', 'Fe III', 'Fe IV', 'Fe V', 'Ni II']]],
        ]

    if args.recombrates:
        plot_recombrates(modelpath, estimators, 26, [2, 3, 4, 5])
        plot_recombrates(modelpath, estimators, 27, [3, 4])
        plot_recombrates(modelpath, estimators, 28, [3, 4, 5])
    else:
        modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)

        if args.modelgridindex > -1 or args.x in ['time', 'timestep']:
            # plot time evolution in specific cell
            if not args.x:
                args.x = 'time'
            mgilist = [args.modelgridindex] * len(timesteps_included)
            timesteplist_unfiltered = [(ts,) for ts in timesteps_included]
            if estimators[(args.modelgridindex, timesteps_included[0])]['emptycell']:
                raise ValueError(f'cell {args.modelgridindex} is empty. no estimators available')
            make_plot(modelpath, timesteplist_unfiltered, mgilist, estimators, args.x, plotlist, args)
        else:
            # plot a range of cells in a time snapshot showing internal structure

            if not args.x:
                args.x = 'velocity_outer'

            if args.classicartis:
                allnonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                                      if (timesteps_included[0], modelgridindex) in estimators]
            else:
                allnonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                                      if not estimators[(timesteps_included[0], modelgridindex)]['emptycell']]

            if args.multiplot:
                pdf_list = []
                modelpath_list = []
                for timestep in range(timestepmin, timestepmax + 1):
                    timesteplist_unfiltered = [[timestep]] * len(allnonemptymgilist)
                    outfilename = make_plot(modelpath, timesteplist_unfiltered, allnonemptymgilist, estimators, args.x,
                                            plotlist, args)

                    if '/' in outfilename:
                        outfilename = outfilename.split('/')[1]

                    pdf_list.append(outfilename)
                    modelpath_list.append(modelpath)

                if len(pdf_list) > 1:
                    at.join_pdf_files(pdf_list, modelpath_list)

            else:
                timesteplist_unfiltered = [timesteps_included] * len(allnonemptymgilist)
                make_plot(modelpath, timesteplist_unfiltered, allnonemptymgilist, estimators, args.x, plotlist, args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
