#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt

# import numexpr as ne
import numpy as np
import pandas as pd
from astropy import constants as const

from astropy import units as u
import artistools as at

K_B = const.k_B.to('eV / K').value
c = const.c.to('km / s').value

PYDIR = os.path.dirname(os.path.abspath(__file__))

SPECTRA_DIR = os.path.join(PYDIR, 'data', 'refspectra')


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


def generate_ion_spectrum(transitions, xvalues, plot_resolution, popcolumn, args):
    yvalues = np.zeros(len(xvalues))

    print(transitions.loc[transitions['flux_factor'] == transitions['flux_factor'].max()])

    # iterate over lines
    for _, line in transitions.iterrows():
        flux_factor = line['flux_factor']

        if args.print_lines and flux_factor > 0.0:  # some lines have zero A value, so ignore these
            print_line_details(line, args.T)

        # contribute the Gaussian line profile to the discrete flux bins

        centre_index = int(round((line['lambda_angstroms'] - args.xmin) / plot_resolution))
        sigma_angstroms = line['lambda_angstroms'] * args.sigma_v / c
        sigma_gridpoints = int(math.ceil(sigma_angstroms / plot_resolution))
        window_left_index = max(int(centre_index - args.gaussian_window * sigma_gridpoints), 0)
        window_right_index = min(int(centre_index + args.gaussian_window * sigma_gridpoints), len(xvalues))

        for x in range(max(0, window_left_index), min(len(xvalues), window_right_index)):
            yvalues[x] += flux_factor * math.exp(
                -((x - centre_index) * plot_resolution / sigma_angstroms) ** 2) / sigma_angstroms

    return yvalues


def print_line_details(line, T_K):
    forbidden_status = 'forbidden' if line['forbidden'] else 'permitted'
    metastable_status = 'upper not metastable' if line['upper_has_permitted'] else ' upper is metastable'
    ion_name = f"{at.elsymbols[line['Z']]} {at.roman_numerals[line['ion_stage']]}"
    print(f"{line['lambda_angstroms']:7.1f} Å flux: {line['flux_factor']:9.3E} "
          f"{ion_name:6} {forbidden_status}, {metastable_status}, "
          f"lower: {line['lower_level']:29s} upper: {line['upper_level']}")


def make_plot(xvalues, yvalues, ax, ions, ionpopdict, args):
    yvalues_combined = np.zeros_like(xvalues, dtype=np.float)
    for ion_index in range(len(ions) + 1):
        if ion_index < len(ions):
            ion = ions[ion_index]
            # an ion subplot
            yvalues_combined += yvalues[ion_index]

            ax[ion_index].plot(xvalues, yvalues[ion_index], linewidth=1.5,
                               label=f'{at.elsymbols[ion.Z]} {at.roman_numerals[ion.ion_stage]}'
                               f' (pop={ionpopdict[(ion.Z, ion.ion_stage)]:.1e})')

        else:
            # the subplot showing combined spectrum of multiple ions
            # and observational data
            obsspectra = [
                # ('dop_dered_SN2013aa_20140208_fc_final.txt',
                #  'SN2013aa +360d (Maguire)','0.3'),
                # ('2010lp_20110928_fors2.txt',
                #  'SN2010lp +264d (Taubenberger et al. 2013)','0.1'),
                ('2003du_20031213_3219_8822_00.txt',
                 'SN2003du +221.3d (Stanishev et al. 2007)', '0.0'),
            ]

            for (filename, serieslabel, linecolor) in obsspectra:
                obsfile = os.path.join(SPECTRA_DIR, filename)
                obsdata = pd.read_csv(obsfile, delim_whitespace=True, header=None, names=['lambda_angstroms', 'flux'])
                obsdata = obsdata[
                    (obsdata[:]['lambda_angstroms'] > args.xmin) &
                    (obsdata[:]['lambda_angstroms'] < args.xmax)]
                obsdata['flux_scaled'] = obsdata['flux'] * max(yvalues_combined) / max(obsdata['flux'])
                obsdata.plot(x='lambda_angstroms', y='flux_scaled', ax=ax[-1], linewidth=1,
                             color='black', label=serieslabel, zorder=-1)

            combined_label = 'All ions'
            ax[-1].plot(xvalues, yvalues_combined, linewidth=1.5, label=combined_label)
            ax[-1].set_xlabel(r'Wavelength ($\AA$)')

        ax[ion_index].set_xlim(xmin=args.xmin, xmax=args.xmax)
        ax[ion_index].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
        ax[ion_index].set_ylabel(r'$\propto$ F$_\lambda$')


def addargs(parser, defaultoutputfile):
    parser.add_argument('-xmin', type=int, default=3500,
                        help='Plot range: minimum wavelength in Angstroms')

    parser.add_argument('-xmax', type=int, default=8000,
                        help='Plot range: maximum wavelength in Angstroms')

    parser.add_argument('-T', type=float, dest='T', default=5795.82,
                        help='Temperature in Kelvin')

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


def main(argsraw=None):
    defaultoutputfile = 'plottransitions_cell{cell:03d}_{timestep:03d}.pdf'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot estimated spectra from bound-bound transitions.')
    addargs(parser, defaultoutputfile)
    args = parser.parse_args(argsraw)

    if os.path.isdir(args.outputfile):
        args.outputfile = os.path.join(args.outputfile, defaultoutputfile)

    modelpath = '.'

    if args.timedays:
        timestep = at.get_closest_timestep(os.path.join(modelpath, "spec.out"), args.timedays)
    else:
        timestep = args.timestep

    # also calculate wavelengths outside the plot range to include lines whose
    # edges pass through the plot range
    plot_xmin_wide = args.xmin * (1 - args.gaussian_window * args.sigma_v / c)
    plot_xmax_wide = args.xmax * (1 + args.gaussian_window * args.sigma_v / c)

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
        len(ionlist) + 1, 1, sharex=True, figsize=(6, 2 * (len(ionlist) + 1)),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    # resolution of the plot in Angstroms
    plot_resolution = int((args.xmax - args.xmin) / 1000)

    xvalues = np.arange(args.xmin, args.xmax, step=plot_resolution)
    yvalues = np.zeros((len(ionlist), len(xvalues)))

    iontuples = [(x.Z, x.ion_stage) for x in ionlist]

    adata = at.get_levels(
        os.path.join(modelpath, 'adata.txt'), os.path.join(modelpath, 'transitiondata.txt'),
        iontuples)

    dfnltepops = get_nltepops(modelpath, modelgridindex=args.modelgridindex, timestep=timestep)

    popcolumn = 'upper_lte_pop'
    ionpopdict = {(ion.Z, ion.ion_stage): ion.ion_pop for ion in ionlist}

    # popcolumn = 'upper_nlte_pop'
    # ionpopdict = {(ion.Z, ion.ion_stage): dfnltepops.query(
    #     'Z==@ion.Z and ion_stage==@ion.ion_stage')['n_NLTE'].sum() for ion in ionlist}

    modelname = at.get_model_name(modelpath)
    modeldata, _ = at.get_modeldata(os.path.join(modelpath, 'model.txt'))
    velocity = modeldata['velocity'][args.modelgridindex]

    figure_title = f'{modelname}\nTimestep {timestep}'
    time_days = float(at.get_timestep_time('spec.out', timestep))
    if time_days >= 0:
        figure_title += f' (t={time_days:.2f}d)'
    figure_title += f' cell {args.modelgridindex} ({velocity} km/s) {popcolumn}'
    print(figure_title)
    axes[0].set_title(figure_title, fontsize=9)

    hc = (const.h * const.c).to('eV Angstrom').value

    transitions_dict = {}
    for _, ion in adata.iterrows():
        ionid = (ion.Z, ion.ion_stage)
        if ionid not in iontuples:
            continue
        else:
            ionindex = iontuples.index(ionid)

        print(f'{at.elsymbols[ion.Z]} {at.roman_numerals[ion.ion_stage]:3s} '
              f'{ion.level_count:5d} levels, {len(ion.transitions):6d} transitions', end='')

        dftransitions = ion.transitions
        if not args.include_permitted and not dftransitions.empty:
            dftransitions.query('forbidden == True', inplace=True)
            print(f' ({len(ion.transitions):6d} forbidden)')
        else:
            print()

        if not dftransitions.empty:
            print('Calculating wavelengths and filtering')

            dftransitions.eval('upper_energy_ev = @ion.levels.loc[upper_levelindex].energy_ev.values', inplace=True)

            dftransitions.eval('lower_energy_ev = @ion.levels.loc[lower_levelindex].energy_ev.values', inplace=True)

            dftransitions.eval('lambda_angstroms = @hc / (upper_energy_ev - lower_energy_ev)', inplace=True)

            dftransitions.query('lambda_angstroms >= @args.xmin & lambda_angstroms <= @args.xmax', inplace=True)

            # dftransitions.sort_values(by='lambda_angstroms', inplace=True)

            print(f'  {len(dftransitions):d} plottable transitions')

            dftransitions.eval('upper_statweight = @ion.levels.loc[upper_levelindex].g.values', inplace=True)

            print(f'  Getting NLTE populations')
            dfnltepops_thision = dfnltepops.query('Z==@ion.Z & ion_stage==@ion.ion_stage')

            nltepopdict = {x.level: x['n_NLTE'] for _, x in dfnltepops_thision.iterrows()}

            dftransitions['upper_nlte_pop'] = dftransitions.apply(
                lambda x: nltepopdict.get(x.upper_levelindex, 0.), axis=1)

            print(f'  Getting LTE populations')

            ltepartfunc = ion.levels.eval('g * exp(-energy_ev / @K_B / @args.T)').sum()

            scalefactor = ionpopdict[ionid] / ltepartfunc
            dftransitions.eval(
                'upper_lte_pop = @scalefactor * upper_statweight * exp(-upper_energy_ev / @K_B / @args.T)', inplace=True)

            print(f'  Generating ion spectrum')

            dftransitions.eval(f'flux_factor = (upper_energy_ev - lower_energy_ev) * A * {popcolumn}', inplace=True)

            yvalues[ionindex] = generate_ion_spectrum(
                dftransitions, xvalues, plot_resolution, popcolumn, args)

    make_plot(xvalues, yvalues, axes, ionlist, ionpopdict, args)

    outputfilename = args.outputfile.format(cell=args.modelgridindex, timestep=timestep)
    print(f"Saving '{outputfilename}'")
    fig.savefig(outputfilename, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
