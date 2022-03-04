#!/usr/bin/env python3
import argparse
# import glob
import math
import multiprocessing
# import re
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
# import numexpr as ne
import numpy as np
import pandas as pd
from astropy import constants as const
# from astropy import units as u

import artistools as at
import artistools.estimators
import artistools.nltepops
import artistools.spectra

defaultoutputfile = 'plottransitions_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf'

iontuple = namedtuple('iontuple', 'Z ion_stage')


def get_kurucz_transitions():
    hc_evcm = (const.h * const.c).to('eV cm').value
    transitiontuple = namedtuple('transition',
                                 'Z ionstage lambda_angstroms A lower_energy_ev upper_energy_ev lower_g upper_g')
    translist = []
    ionlist = []
    with open('gfall.dat', 'r') as fnist:
        for line in fnist:
            row = line.split()
            if len(row) >= 24:
                Z, ionstage = int(row[2].split('.')[0]), int(row[2].split('.')[1]) + 1
                if Z < 44 or ionstage >= 2:  # and Z not in [26, 27]
                    continue
                lambda_angstroms = float(line[:12]) * 10
                loggf = float(line[11:18])
                lower_energy_ev, upper_energy_ev = hc_evcm * float(line[24:36]), hc_evcm * float(line[52:64])
                lower_g, upper_g = 2 * float(line[36:42]) + 1, 2 * float(line[64:70]) + 1
                fij = (10 ** loggf) / lower_g
                A = fij / (1.49919e-16 * upper_g / lower_g * lambda_angstroms ** 2)
                translist.append(transitiontuple(
                    Z, ionstage, lambda_angstroms, A, lower_energy_ev, upper_energy_ev, lower_g, upper_g))

                if iontuple(Z, ionstage) not in ionlist:
                    ionlist.append(iontuple(Z, ionstage))

    dftransitions = pd.DataFrame(translist, columns=transitiontuple._fields)
    return dftransitions, ionlist


def get_nist_transitions(filename):
    transitiontuple = namedtuple('transition', 'lambda_angstroms A lower_energy_ev upper_energy_ev lower_g upper_g')
    translist = []
    with open(filename, 'r') as fnist:
        for line in fnist:
            row = line.split('|')
            if len(row) == 17 and '-' in row[5]:
                if len(row[0].strip()) > 0:
                    lambda_angstroms = float(row[0])
                elif len(row[1].strip()) > 0:
                    lambda_angstroms = float(row[1])
                else:
                    continue
                if len(row[3].strip()) > 0:
                    A = float(row[3])
                else:
                    # continue
                    A = 1e8
                lower_energy_ev, upper_energy_ev = [float(x.strip(' []')) for x in row[5].split('-')]
                lower_g, upper_g = [float(x.strip()) for x in row[12].split('-')]
                translist.append(transitiontuple(
                    lambda_angstroms, A, lower_energy_ev, upper_energy_ev, lower_g, upper_g))

    dftransitions = pd.DataFrame(translist, columns=transitiontuple._fields)
    return dftransitions


def generate_ion_spectrum(transitions, xvalues, popcolumn, plot_resolution, args):
    yvalues = np.zeros(len(xvalues))

    # iterate over lines
    for _, line in transitions.iterrows():
        flux = line['flux_factor'] * line[popcolumn]

        # contribute the Gaussian line profile to the discrete flux bins

        centre_index = int(round((line['lambda_angstroms'] - args.xmin) / plot_resolution))
        sigma_angstroms = line['lambda_angstroms'] * args.sigma_v / const.c.to('km / s').value
        sigma_gridpoints = int(math.ceil(sigma_angstroms / plot_resolution))
        window_left_index = max(int(centre_index - args.gaussian_window * sigma_gridpoints), 0)
        window_right_index = min(int(centre_index + args.gaussian_window * sigma_gridpoints), len(xvalues))

        for x in range(max(0, window_left_index), min(len(xvalues), window_right_index)):
            yvalues[x] += flux * math.exp(
                -((x - centre_index) * plot_resolution / sigma_angstroms) ** 2) / sigma_angstroms

    return yvalues


def make_plot(
        xvalues, yvalues, temperature_list, vardict, ionlist, ionpopdict, xmin, xmax, figure_title, outputfilename):
    # npanels = len(ionlist) + 1
    npanels = len(ionlist)

    fig, axes = plt.subplots(
        nrows=npanels, ncols=1, sharex=True, sharey=False, figsize=(6, 2 * (len(ionlist) + 1)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if len(ionlist) == 1:
        axes = [axes]

    if figure_title:
        print(figure_title)
        axes[0].set_title(figure_title, fontsize=10)

    peak_y_value = -1
    yvalues_combined = np.zeros((len(temperature_list), len(xvalues)))
    for seriesindex, temperature in enumerate(temperature_list):
        T_exc = eval(temperature, vardict)
        serieslabel = 'NLTE' if T_exc < 0 else f'LTE {temperature} = {T_exc:.0f} K'
        for ion_index, axis in enumerate(axes[:len(ionlist)]):
            # an ion subplot
            yvalues_combined[seriesindex] += yvalues[seriesindex][ion_index]

            axis.plot(xvalues, yvalues[seriesindex][ion_index], linewidth=1.5, label=serieslabel)

            peak_y_value = max(yvalues[seriesindex][ion_index])

            axis.legend(loc='upper left', handlelength=1, frameon=False, numpoints=1, prop={'size': 8})

        if len(axes) > len(ionlist):
            axes[len(ionlist)].plot(xvalues, yvalues_combined[seriesindex], linewidth=1.5, label=serieslabel)
            peak_y_value = max(peak_y_value, max(yvalues_combined[seriesindex]))

    axislabels = [
        f'{at.get_elsymbol(Z)} {at.roman_numerals[ion_stage]}\n(pop={ionpopdict[(Z, ion_stage)]:.1e}/cm3)'
        for (Z, ion_stage) in ionlist]
    axislabels += ['Total']

    for axis, axislabel in zip(axes, axislabels):
        axis.annotate(
            axislabel, xy=(0.99, 0.96), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top', fontsize=10)

    # at.spectra.plot_reference_spectrum(
    #     'dop_dered_SN2013aa_20140208_fc_final.txt', axes[-1], xmin, xmax, True,
    #     scale_to_peak=peak_y_value, zorder=-1, linewidth=1, color='black')
    #
    # at.spectra.plot_reference_spectrum(
    #     '2003du_20031213_3219_8822_00.txt', axes[-1], xmin, xmax,
    #     scale_to_peak=peak_y_value, zorder=-1, linewidth=1, color='black')

    axes[-1].set_xlabel(r'Wavelength ($\AA$)')

    for axis in axes:
        axis.set_xlim(xmin, xmax)
        axis.set_ylabel(r'$\propto$ F$_\lambda$')

    print(f"Saving '{outputfilename}'")
    fig.savefig(outputfilename, format='pdf')
    plt.close()


def add_upper_lte_pop(dftransitions, T_exc, ionpop, ltepartfunc, columnname=None):
    K_B = const.k_B.to('eV / K').value
    scalefactor = ionpop / ltepartfunc
    if columnname is None:
        columnname = f'upper_pop_lte_{T_exc:.0f}K'
    dftransitions.eval(
        f'{columnname} = @scalefactor * upper_g * exp(-upper_energy_ev / @K_B / @T_exc)',
        inplace=True)


def addargs(parser):
    parser.add_argument('-modelpath', default=None, type=Path,
                        help='Path to ARTIS folder')

    parser.add_argument('-xmin', type=int, default=3500,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=8000,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('-T', type=float, dest='T', default=[], nargs='*',
                        help='Temperature in Kelvin')

    parser.add_argument('-sigma_v', type=float, default=5500.,
                        help='Gaussian width in km/s')

    parser.add_argument('-gaussian_window', type=float, default=3,
                        help='Truncate Gaussian line profiles n sigmas from the centre')

    parser.add_argument('--include-permitted', action='store_true',
                        help='Also consider permitted lines')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', type=int, default=70,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('--normalised', action='store_true',
                        help='Normalise all spectra to their peak values')

    parser.add_argument('--print-lines', action='store_true',
                        help='Output details of matching line details to standard out')

    parser.add_argument('--atomicdatabase', default='artis', choices=['artis', 'kurucz', 'nist'],
                        help='Source of atomic data for excitation transitions')

    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='path/filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Plot estimated spectra from bound-bound transitions.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if Path(args.outputfile).is_dir():
        args.outputfile = Path(args.outputfile, defaultoutputfile)

    if args.modelpath:
        from_model = True
    else:
        from_model = False
        args.modelpath = Path()

    modelpath = args.modelpath
    if from_model:
        modelgridindex = args.modelgridindex

        if args.timedays:
            timestep = at.get_timestep_of_timedays(modelpath, args.timedays)
        else:
            timestep = args.timestep

        modeldata, _, _ = at.inputmodel.get_modeldata(Path(modelpath, 'model.txt'))
        estimators_all = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)
        if not estimators_all:
            return -1

        estimators = estimators_all[(timestep, modelgridindex)]
        if estimators['emptycell']:
            print(f'ERROR: cell {modelgridindex} is marked as empty')
            return -1

    # also calculate wavelengths outside the plot range to include lines whose
    # edges pass through the plot range
    plot_xmin_wide = args.xmin * (1 - args.gaussian_window * args.sigma_v / const.c.to('km / s').value)
    plot_xmax_wide = args.xmax * (1 + args.gaussian_window * args.sigma_v / const.c.to('km / s').value)

    ionlist = [
        (26, 1), (26, 2), (26, 3),
        (27, 2), (27, 3),
        (28, 2), (28, 3),
        # (28, 2),
        # iontuple(45, 1),
        # iontuple(54, 1),
        # iontuple(54, 2),
        # iontuple(55, 1),
        # iontuple(55, 2),
        # iontuple(58, 1),
        # iontuple(79, 1),
        # iontuple(83, 1),
        # iontuple(26, 2),
        # iontuple(26, 3),
    ]

    if args.atomicdatabase == 'kurucz':
        dftransgfall, ionlist = get_kurucz_transitions()

    ionlist.sort()

    # resolution of the plot in Angstroms
    plot_resolution = max(1, int((args.xmax - args.xmin) / 1000))

    if args.atomicdatabase == 'artis':
        adata = at.atomic.get_levels(modelpath, tuple(ionlist), get_transitions=True)

    if from_model:
        dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)

        if dfnltepops is None or dfnltepops.empty:
            print(f'ERROR: no NLTE populations for cell {modelgridindex} at timestep {timestep}')
            return -1

        ionpopdict = {(Z, ion_stage): dfnltepops.query(
            'Z==@Z and ion_stage==@ion_stage')['n_NLTE'].sum() for Z, ion_stage in ionlist}

        modelname = at.get_model_name(modelpath)
        velocity = modeldata['velocity_outer'][modelgridindex]

        Te = estimators['Te']
        TR = estimators['TR']
        figure_title = f'{modelname}\n'
        figure_title += (f'Cell {modelgridindex} ({velocity} km/s) with '
                         f'Te = {Te:.1f} K, TR = {TR:.1f} K at timestep {timestep}')
        time_days = float(at.get_timestep_time(modelpath, timestep))
        if time_days != -1:
            figure_title += f' ({time_days:.1f}d)'

        # -1 means use NLTE populations
        temperature_list = ['Te', 'TR', '-1']
        temperature_list = ['-1']
        vardict = {'Te': Te, 'TR': TR}
    else:
        if not args.T:
            args.T = [2000]
        if len(args.T) == 1:
            figure_title = f'Te = {args.T[0]:.1f}'
        else:
            figure_title = None

        temperature_list = []
        vardict = {}
        for index, temperature in enumerate(args.T):
            tlabel = 'Te'
            if index > 0:
                tlabel += f'_{index + 1}'
            vardict[tlabel] = temperature
            temperature_list.append(tlabel)

        # Fe3overFe2 = 8  # number ratio
        # ionpopdict = {
        #     (26, 2): 1 / (1 + Fe3overFe2),
        #     (26, 3): Fe3overFe2 / (1 + Fe3overFe2),
        #     (28, 2): 1.0e-2,
        # }
        ionpopdict = {ion: 1 for ion in ionlist}

    hc = (const.h * const.c).to('eV Angstrom').value

    xvalues = np.arange(args.xmin, args.xmax, step=plot_resolution)
    yvalues = np.zeros((len(temperature_list) + 1, len(ionlist), len(xvalues)))

    for _, ion in adata.iterrows() if args.atomicdatabase == 'artis' else enumerate(ionlist):
        ionid = (ion.Z, ion.ion_stage)
        if ionid not in ionlist:
            continue
        else:
            ionindex = ionlist.index(ionid)

        if args.atomicdatabase == 'kurucz':
            dftransitions = dftransgfall.query('Z == @ion.Z and ionstage == @ion.ion_stage', inplace=False).copy()
        elif args.atomicdatabase == 'nist':
            dftransitions = get_nist_transitions(f'nist/nist-{ion.Z:02d}-{ion.ion_stage:02d}.txt')
        else:
            dftransitions = ion.transitions

        print(f'\n======> {at.get_elsymbol(ion.Z)} {at.roman_numerals[ion.ion_stage]:3s} '
              f'(pop={ionpopdict[ionid]:.2e} / cm3, {len(dftransitions):6d} transitions)')

        if not args.include_permitted and not dftransitions.empty:
            dftransitions.query('forbidden == True', inplace=True)
            print(f'  ({len(ion.transitions):6d} forbidden)')

        if not dftransitions.empty:
            if args.atomicdatabase == 'artis':
                dftransitions.eval('upper_energy_ev = @ion.levels.loc[upper].energy_ev.values', inplace=True)
                dftransitions.eval('lower_energy_ev = @ion.levels.loc[lower].energy_ev.values', inplace=True)
                dftransitions.eval('lambda_angstroms = @hc / (upper_energy_ev - lower_energy_ev)', inplace=True)

            dftransitions.query('lambda_angstroms >= @plot_xmin_wide & lambda_angstroms <= @plot_xmax_wide',
                                inplace=True)

            dftransitions.sort_values(by='lambda_angstroms', inplace=True)

            print(f'  {len(dftransitions)} plottable transitions')

            if args.atomicdatabase == 'artis':
                dftransitions.eval('upper_g = @ion.levels.loc[upper].g.values', inplace=True)
                K_B = const.k_B.to('eV / K').value
                T_exc = vardict['Te']
                ltepartfunc = ion.levels.eval('g * exp(-energy_ev / @K_B / @T_exc)').sum()
            else:
                ltepartfunc = 1.0

            dftransitions.eval('flux_factor = (upper_energy_ev - lower_energy_ev) * A', inplace=True)
            add_upper_lte_pop(dftransitions, vardict['Te'], ionpopdict[ionid], ltepartfunc, columnname='upper_pop_Te')

            for seriesindex, temperature in enumerate(temperature_list):
                T_exc = eval(temperature, vardict)
                if T_exc < 0:
                    dfnltepops_thision = dfnltepops.query('Z==@ion.Z & ion_stage==@ion.ion_stage')

                    nltepopdict = {x.level: x['n_NLTE'] for _, x in dfnltepops_thision.iterrows()}

                    dftransitions['upper_pop_nlte'] = dftransitions.apply(
                        lambda x: nltepopdict.get(x.upper, 0.), axis=1)

                    # dftransitions['lower_pop_nlte'] = dftransitions.apply(
                    #     lambda x: nltepopdict.get(x.lower, 0.), axis=1)

                    popcolumnname = 'upper_pop_nlte'
                    dftransitions.eval(f'flux_factor_nlte = flux_factor * {popcolumnname}', inplace=True)
                    dftransitions.eval('upper_departure = upper_pop_nlte / upper_pop_Te', inplace=True)
                    if ionid == (26, 2):
                        fe2depcoeff = dftransitions.query('upper == 16 and lower == 5').iloc[0].upper_departure
                    elif ionid == (28, 2):
                        ni2depcoeff = dftransitions.query('upper == 6 and lower == 0').iloc[0].upper_departure

                    with pd.option_context('display.width', 200):
                        print(dftransitions.nlargest(1, 'flux_factor_nlte'))
                else:
                    popcolumnname = f'upper_pop_lte_{T_exc:.0f}K'
                    if args.atomicdatabase == 'artis':
                        dftransitions.eval('upper_g = @ion.levels.loc[upper].g.values', inplace=True)
                        K_B = const.k_B.to('eV / K').value
                        ltepartfunc = ion.levels.eval('g * exp(-energy_ev / @K_B / @T_exc)').sum()
                    else:
                        ltepartfunc = 1.0
                    add_upper_lte_pop(dftransitions, T_exc, ionpopdict[ionid], ltepartfunc, columnname=popcolumnname)

                if args.print_lines:
                    dftransitions.eval(f'flux_factor_{popcolumnname} = flux_factor * {popcolumnname}', inplace=True)

                yvalues[seriesindex][ionindex] = generate_ion_spectrum(dftransitions, xvalues,
                                                                       popcolumnname, plot_resolution, args)
                if args.normalised:
                    yvalues[seriesindex][ionindex] /= max(yvalues[seriesindex][ionindex])  # todo: move to ax.plot line

        if args.print_lines:
            print(dftransitions.columns)
            print(dftransitions[[
                'lower', 'upper', 'forbidden', 'A', 'lambda_angstroms', 'flux_factor_upper_pop_lte_3000K']].to_string(
                    index=False))
    print()

    if from_model:
        feions = [2, 3]

        def get_strionfracs(atomic_number, ionstages):
            est_ionfracs = [
                estimators['populations'][(atomic_number, ionstage)] / estimators['populations'][atomic_number]
                for ionstage in ionstages]
            ionfracs_str = ' '.join([f'{pop:6.0e}' if pop < 0.01 else f'{pop:6.2f}' for pop in est_ionfracs])
            strions = ' '.join(
                [f'{at.get_elsymbol(atomic_number)}{at.roman_numerals[ionstage]}'.rjust(6) for ionstage in feions])
            return strions, ionfracs_str

        strfeions, est_fe_ionfracs_str = get_strionfracs(26, [2, 3])

        strniions, est_ni_ionfracs_str = get_strionfracs(28, [2, 3])

        print(f'                     Fe II 7155             Ni II 7378  {strfeions}   /  {strniions}'
              '      T_e    Fe III/II       Ni III/II')

        print(f'{velocity:5.0f} km/s({modelgridindex})      {fe2depcoeff:5.2f}                   '
              f'{ni2depcoeff:.2f}        '
              f'{est_fe_ionfracs_str}   /  {est_ni_ionfracs_str}      {Te:.0f}    '
              f"{estimators['populations'][(26, 3)] / estimators['populations'][(26, 2)]:.2f}          "
              f"{estimators['populations'][(28, 3)] / estimators['populations'][(28, 2)]:5.2f}")

    if from_model:
        outputfilename = str(args.outputfile).format(cell=modelgridindex, timestep=timestep, time_days=time_days)
    else:
        outputfilename = 'plottransitions.pdf'

    make_plot(xvalues, yvalues, temperature_list, vardict, ionlist,
              ionpopdict, args.xmin, args.xmax, figure_title, outputfilename)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
