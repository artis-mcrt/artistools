#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
# import numexpr as ne
import numpy as np
import pandas as pd
from astropy import constants as const
# from astropy import units as u

import artistools as at
from artistools import estimators, spectra

defaultoutputfile = 'plottransitions_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf'


def get_nltepops(modelpath, timestep, modelgridindex):
    nlte_files = (
        glob.glob(os.path.join(modelpath, 'nlte_????.out'), recursive=True) +
        glob.glob(os.path.join(modelpath, '*/nlte_????.out'), recursive=True))

    if not nlte_files:
        print("No NLTE files found.")
        return
    else:
        print(f'Loading {len(nlte_files)} NLTE files')
        for nltefilepath in nlte_files:
            filerank = int(re.search('[0-9]+', os.path.basename(nltefilepath)).group(0))

            if filerank > modelgridindex:
                continue

            dfpop = pd.read_csv(nltefilepath, delim_whitespace=True)

            dfpop.query('(modelgridindex==@modelgridindex) & (timestep==@timestep)', inplace=True)
            if not dfpop.empty:
                return dfpop

    return pd.DataFrame()


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


def make_plot(xvalues, yvalues, axes, temperature_list, vardict, ions, ionpopdict, xmin, xmax):
    peak_y_value = -1
    yvalues_combined = np.zeros((len(temperature_list), len(xvalues)))
    for seriesindex, temperature in enumerate(temperature_list):
        T_exc = eval(temperature, vardict)
        serieslabel = 'NLTE' if T_exc < 0 else f'LTE {temperature} = {T_exc:.0f} K'
        for ion_index, axis in enumerate(axes[:-1]):
            # an ion subplot
            yvalues_combined[seriesindex] += yvalues[seriesindex][ion_index]

            axis.plot(xvalues, yvalues[seriesindex][ion_index], linewidth=1.5, label=serieslabel)

        axes[-1].plot(xvalues, yvalues_combined[seriesindex], linewidth=1.5, label=serieslabel)
        peak_y_value = max(peak_y_value, max(yvalues_combined[seriesindex]))

    axislabels = [
        f'{at.elsymbols[ion.Z]} {at.roman_numerals[ion.ion_stage]}\n(pop={ionpopdict[(ion.Z, ion.ion_stage)]:.1e}/cm3)'
        for ion in ions] + ['Total']

    for axis, axislabel in zip(axes, axislabels):
        axis.annotate(
            axislabel, xy=(0.99, 0.96), xycoords='axes fraction',
            horizontalalignment='right', verticalalignment='top', fontsize=10)

    # at.spectra.plot_reference_spectrum(
    #     'dop_dered_SN2013aa_20140208_fc_final.txt', axes[-1], xmin, xmax, True,
    #     scale_to_peak=peak_y_value, zorder=-1, linewidth=1, color='black')

    at.spectra.plot_reference_spectrum(
        '2003du_20031213_3219_8822_00.txt', axes[-1], xmin, xmax, True,
        scale_to_peak=peak_y_value, zorder=-1, linewidth=1, color='black')

    axes[-1].set_xlabel(r'Wavelength ($\AA$)')

    for axis in axes:
        axis.set_xlim(xmin=xmin, xmax=xmax)
        axis.set_ylabel(r'$\propto$ F$_\lambda$')

    axes[-1].legend(loc='upper right', handlelength=1, frameon=False, numpoints=1, prop={'size': 8})


def add_upper_lte_pop(dftransitions, T_exc, ion, ionpop, columnname=None):
    K_B = const.k_B.to('eV / K').value
    ltepartfunc = ion.levels.eval('g * exp(-energy_ev / @K_B / @T_exc)').sum()
    scalefactor = ionpop / ltepartfunc
    if columnname is None:
        columnname = f'upper_pop_lte_{T_exc:.0f}K'
    dftransitions.eval(
        f'{columnname} = @scalefactor * upper_g * exp(-upper_energy_ev / @K_B / @T_exc)',
        inplace=True)


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-xmin', type=int, default=3500,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=8000,
                        help='Plot range: maximum wavelength in Angstroms')

    # parser.add_argument('-T', type=float, dest='T', default=2000,
    #                     help='Temperature in Kelvin')

    parser.add_argument('-sigma_v', type=float, default=5500.,
                        help='Gaussian width in km/s')

    parser.add_argument('-gaussian_window', type=float, default=3,
                        help='Truncate Gaussian line profiles n sigmas from the centre')

    parser.add_argument('--include-permitted', action='store_true', default=False,
                        help='Also consider permitted lines')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', type=int, default=70,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('--print-lines', action='store_true', default=False,
                        help='Output details of matching line details to standard out')

    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='path/filename for PDF file')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot estimated spectra from bound-bound transitions.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    modelpath = args.modelpath
    modelgridindex = args.modelgridindex

    if args.timedays:
        timestep = at.get_closest_timestep(os.path.join(modelpath, "spec.out"), args.timedays)
    else:
        timestep = args.timestep

    modeldata, _ = at.get_modeldata(os.path.join(modelpath, 'model.txt'))
    estimators_all = at.estimators.read_estimators(modelpath, modeldata, keymatch=(timestep, modelgridindex))

    estimators = estimators_all[(timestep, modelgridindex)]
    if estimators['emptycell']:
        print(f'ERROR: cell {modelgridindex} is marked as empty')
        return -1

    # also calculate wavelengths outside the plot range to include lines whose
    # edges pass through the plot range
    plot_xmin_wide = args.xmin * (1 - args.gaussian_window * args.sigma_v / const.c.to('km / s').value)
    plot_xmax_wide = args.xmax * (1 + args.gaussian_window * args.sigma_v / const.c.to('km / s').value)

    iontuple = namedtuple('ion', 'Z ion_stage ion_pop')

    Fe3overFe2 = 8  # number ratio
    ionlist = [
        iontuple(26, 2, 1 / (1 + Fe3overFe2)),
        iontuple(26, 3, Fe3overFe2 / (1 + Fe3overFe2)),
        # iontuple(27, 2, 1.0),
        # iontuple(27, 3, 1.0),
        iontuple(28, 2, 1.0e-2),
    ]

    fig, axes = plt.subplots(
        len(ionlist) + 1, 1, sharex=True, sharey=True, figsize=(6, 2 * (len(ionlist) + 1)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    # resolution of the plot in Angstroms
    plot_resolution = max(1, int((args.xmax - args.xmin) / 1000))

    iontuples = [(x.Z, x.ion_stage) for x in ionlist]

    adata = at.get_levels(modelpath, iontuples, get_transitions=True)

    dfnltepops = get_nltepops(modelpath, modelgridindex=modelgridindex, timestep=timestep)

    if dfnltepops.empty:
        print(f'ERROR: no NLTE populations for cell {modelgridindex} at timestep {timestep}')
        return -1

    # ionpopdict = {(ion.Z, ion.ion_stage): ion.ion_pop for ion in ionlist}

    ionpopdict = {(ion.Z, ion.ion_stage): dfnltepops.query(
        'Z==@ion.Z and ion_stage==@ion.ion_stage')['n_NLTE'].sum() for ion in ionlist}

    # ionpopdict[(28, 2)] = 0.5 * ionpopdict[(26, 2)]

    modelname = at.get_model_name(modelpath)
    velocity = modeldata['velocity'][modelgridindex]

    Te = estimators['Te']
    TR = estimators['TR']
    figure_title = f'{modelname}\n'
    figure_title += f'Cell {modelgridindex} (v={velocity} km/s) with Te = {Te:.1f} K, TR = {TR:.1f} K at timestep {timestep}'
    time_days = float(at.get_timestep_time(modelpath, timestep))
    if time_days != -1:
        figure_title += f' ({time_days:.1f}d)'
    print(figure_title)
    axes[0].set_title(figure_title, fontsize=10)

    hc = (const.h * const.c).to('eV Angstrom').value

    # -1 means use NLTE populations
    temperature_list = ['Te', 'TR', '-1']
    temperature_list = ['-1']
    vardict = {'Te': Te, 'TR': TR}

    xvalues = np.arange(args.xmin, args.xmax, step=plot_resolution)
    yvalues = np.zeros((len(temperature_list) + 1, len(ionlist), len(xvalues)))

    for _, ion in adata.iterrows():
        ionid = (ion.Z, ion.ion_stage)
        if ionid not in iontuples:
            continue
        else:
            ionindex = iontuples.index(ionid)

        print(f'\n======> {at.elsymbols[ion.Z]} {at.roman_numerals[ion.ion_stage]:3s} (pop={ionpopdict[ionid]:.2e} / cm3,'
              f'{ion.level_count:5d} levels, {len(ion.transitions):6d} transitions)', end='')

        dftransitions = ion.transitions
        if not args.include_permitted and not dftransitions.empty:
            dftransitions.query('forbidden == True', inplace=True)
            print(f' ({len(ion.transitions):6d} forbidden)')
        else:
            print()

        if not dftransitions.empty:
            dftransitions.eval('upper_energy_ev = @ion.levels.loc[upper].energy_ev.values', inplace=True)
            dftransitions.eval('lower_energy_ev = @ion.levels.loc[lower].energy_ev.values', inplace=True)
            dftransitions.eval('lambda_angstroms = @hc / (upper_energy_ev - lower_energy_ev)', inplace=True)

            dftransitions.query('lambda_angstroms >= @plot_xmin_wide & lambda_angstroms <= @plot_xmax_wide', inplace=True)

            # dftransitions.sort_values(by='lambda_angstroms', inplace=True)

            print(f'  {len(dftransitions)} plottable transitions')

            dftransitions.eval('upper_g = @ion.levels.loc[upper].g.values', inplace=True)

            dfnltepops_thision = dfnltepops.query('Z==@ion.Z & ion_stage==@ion.ion_stage')

            nltepopdict = {x.level: x['n_NLTE'] for _, x in dfnltepops_thision.iterrows()}

            dftransitions['upper_pop_nlte'] = dftransitions.apply(
                lambda x: nltepopdict.get(x.upper, 0.), axis=1)

            # dftransitions['lower_pop_nlte'] = dftransitions.apply(
            #     lambda x: nltepopdict.get(x.lower, 0.), axis=1)

            dftransitions.eval(f'flux_factor = (upper_energy_ev - lower_energy_ev) * A', inplace=True)

            add_upper_lte_pop(dftransitions, vardict['Te'], ion, ionpopdict[ionid], columnname='upper_pop_Te')

            for seriesindex, temperature in enumerate(temperature_list):
                T_exc = eval(temperature, vardict)
                if T_exc < 0:
                    popcolumnname = 'upper_pop_nlte'
                    dftransitions.eval(f'flux_factor_nlte = flux_factor * {popcolumnname}', inplace=True)
                    dftransitions.eval(f'upper_departure = upper_pop_nlte / upper_pop_Te', inplace=True)
                    if ionid == (26, 2):
                        fe2depcoeff = dftransitions.query('upper == 16 and lower == 5').iloc[0].upper_departure
                    elif ionid == (28, 2):
                        ni2depcoeff = dftransitions.query('upper == 6 and lower == 0').iloc[0].upper_departure

                    with pd.option_context('display.width', 200):
                        print(dftransitions.nlargest(1, 'flux_factor_nlte'))
                else:
                    add_upper_lte_pop(dftransitions, T_exc, ion, ionpopdict)

                yvalues[seriesindex][ionindex] = generate_ion_spectrum(dftransitions, xvalues,
                                                                       popcolumnname, plot_resolution, args)

    print()

    est_fe_ionfracs = [estimators['populations'][(26, ionstage)] / estimators['populations'][26] for ionstage in [1, 2, 3]]
    est_fe_ionfracs_str = ['{:5.2f}'.format(pop) for pop in est_fe_ionfracs]

    est_ni_ionfracs = [estimators['populations'][(28, ionstage)] / estimators['populations'][28] for ionstage in [2, 3]]
    est_ni_ionfracs_str = ['{:5.2f}'.format(pop) for pop in est_ni_ionfracs]

    print('                     Fe II 7155             Ni II 7378       FeI   FeII  FeIII  /    NiII  NiIII      T_e    Fe III/II       Ni III/II')
    print(f'{velocity:5.0f} km/s({modelgridindex})       {fe2depcoeff:.2f}                   {ni2depcoeff:.2f}            ', end='')

    print(f'{" ".join(est_fe_ionfracs_str)}   /   {" ".join(est_ni_ionfracs_str)}       {Te:.0f}   ', end='')

    print(f"{estimators['populations'][(26, 3)] / estimators['populations'][(26, 2)]:.2f}            {estimators['populations'][(28, 3)] / estimators['populations'][(28, 2)]:.2f}")

    make_plot(xvalues, yvalues, axes, temperature_list, vardict, ionlist, ionpopdict, args.xmin, args.xmax)

    outputfilename = args.outputfile.format(cell=modelgridindex, timestep=timestep, time_days=time_days)
    print(f"Saving '{outputfilename}'")
    fig.savefig(outputfilename, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
