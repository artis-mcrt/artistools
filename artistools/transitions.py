#!/usr/bin/env python3
import argparse
import math
import os
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
elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, 'data', 'elements.csv'))['symbol'].values)


iontuple = namedtuple('ion', 'ion_stage number_fraction')

default_ions = [
    iontuple(2, 0.2),
    iontuple(2, 0.2),
    iontuple(3, 0.2),
    iontuple(4, 0.2)]

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                  'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

SPECTRA_DIR = os.path.join(PYDIR, 'data', 'refspectra')


def generate_spectra(transitions, atomic_number, ions, plot_xmin_wide, plot_xmax_wide, args):
    # resolution of the plot in Angstroms
    plot_resolution = int((args.xmax - args.xmin) / 1000)

    xvalues = np.arange(args.xmin, args.xmax, step=plot_resolution)
    yvalues = np.zeros((len(ions), len(xvalues)))

    transitions['flux_factor'] = transitions.apply(f_flux_factor, axis=1, args=(args.T,))

    # iterate over lines
    for _, line in transitions.iterrows():

        ion_index = -1
        for tmpion_index, ion in enumerate(ions):
            if (atomic_number == line['Z'] and
                    ion.ion_stage == line['ion_stage']):
                ion_index = tmpion_index
                break

        if ion_index != -1:
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
                yvalues[ion_index][x] += flux_factor * math.exp(
                    -((x - centre_index) * plot_resolution / sigma_angstroms) ** 2) / sigma_angstroms

    return xvalues, yvalues


def f_flux_factor(line, T_K):
    return (line['upper_energy_Ev'] - line['lower_energy_Ev']) * (
        line['A'] * line['upper_statweight'] * math.exp(-line['upper_energy_Ev'] / K_B / T_K))


def print_line_details(line, T_K):
    forbidden_status = 'forbidden' if line['forbidden'] else 'permitted'
    metastable_status = 'upper not metastable' if line['upper_has_permitted'] else ' upper is metastable'
    ion_name = f"{elsymbols[line['Z']]} {roman_numerals[line['ion_stage']]}"
    print(f"{line['lambda_angstroms']:7.1f} Å flux: {line['flux_factor']:9.3E} "
          f"{ion_name:6} {forbidden_status}, {metastable_status}, "
          f"lower: {line['lower_level']:29s} upper: {line['upper_level']}")


def make_plot(xvalues, yvalues, elsymbol, ions, args):
    fig, ax = plt.subplots(
        len(ions) + 1, 1, sharex=True, figsize=(6, 6),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    yvalues_combined = np.zeros_like(xvalues, dtype=np.float)
    for ion_index in range(len(ions) + 1):
        if ion_index < len(ions):
            # an ion subplot
            if max(yvalues[ion_index]) > 0.0:
                yvalues_normalised = yvalues[ion_index] / max(yvalues[ion_index])
                yvalues_combined += yvalues_normalised * ions[ion_index].number_fraction
            else:
                yvalues_normalised = yvalues[ion_index]
            ax[ion_index].plot(xvalues, yvalues_normalised, linewidth=1.5,
                               label=f'{elsymbol} {roman_numerals[ions[ion_index].ion_stage]}')

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

            combined_label = ' + '.join([
                f'({ion.number_fraction:.1f} * {elsymbol} {roman_numerals[ion.ion_stage]})' for ion in ions])
            ax[-1].plot(xvalues, yvalues_combined, linewidth=1.5, label=combined_label)
            ax[-1].set_xlabel(r'Wavelength ($\AA$)')

        ax[ion_index].set_xlim(xmin=args.xmin, xmax=args.xmax)
        ax[ion_index].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
        ax[ion_index].set_ylabel(r'$\propto$ F$_\lambda$')

    # ax.set_ylim(ymin=-0.05,ymax=1.1)
    outfilename = f'transitions_{elsymbol}.pdf'
    print(f"Saving '{outfilename}'")
    fig.savefig(outfilename, format='pdf')
    plt.close()


def get_artisatomic_transitions(transition_file):
    if os.path.isfile(transition_file + '.tmp'):
        print(f"Loading '{transition_file}.tmp'...")
        # read the sorted binary file (fast)
        transitions = pd.read_pickle(transition_file + '.tmp')

    elif os.path.isfile(transition_file):
        print(f"Loading '{transition_file}'...")

        # read the text file (slower)
        transitions = pd.read_csv(transition_file, delim_whitespace=True)

        # save the dataframe in binary format for next time
        transitions.to_pickle(transition_file + '.tmp')

    else:
        transitions = None

    return transitions


def get_artis_transitions(modelpath, lambdamin, lambdamax, include_permitted, atomic_numbers=None):
    adata = at.get_levels(
        os.path.join(modelpath, 'adata.txt'), os.path.join(modelpath, 'transitiondata.txt'),
        atomic_numbers)

    fulltransitiontuple = namedtuple(
        'fulltransition',
        'lambda_angstroms A Z ion_stage lower_energy_Ev lower_statweight '
        'forbidden upper_statweight upper_energy_Ev upper_has_permitted')
        # 'lower_level upper_level '

    hc = (const.h * const.c).to('eV Angstrom').value

    fulltranslist_all = []
    for ion in adata:
        if ion.Z not in atomic_numbers:
            continue
        print(f'{ion.Z} {ion.ion_stage} levels: {ion.level_count} transitions: {len(ion.transitions)}')
        dftransitions = ion.transitions
        if not include_permitted and not ion.transitions.empty:
            dftransitions.query('forbidden == True', inplace=True)

        for index, transition in dftransitions.iterrows():
            upperlevel = ion.levels[ion.levels.number == transition.upper].iloc[0]
            lowerlevel = ion.levels[ion.levels.number == transition.lower].iloc[0]
            epsilon_trans_ev = upperlevel.energy_ev - lowerlevel.energy_ev
            if epsilon_trans_ev > 0:
                lambda_angstroms = hc / (epsilon_trans_ev)
            else:
                continue
            if lambda_angstroms < lambdamin or lambda_angstroms > lambdamax:
                continue

            fulltranslist_all.append(fulltransitiontuple(
                lambda_angstroms=lambda_angstroms,
                A=transition.A,
                Z=ion.Z,
                ion_stage=ion.ion_stage,
                lower_energy_Ev=lowerlevel.energy_ev,
                lower_statweight=lowerlevel.g,
                forbidden=1 if transition.forbidden else 0,
                # lower_level=lowerlevel.levelname,
                # upper_level=upperlevel.levelname,
                upper_statweight=upperlevel.g,
                upper_energy_Ev=upperlevel.energy_ev,
                upper_has_permitted='?'))

    return pd.DataFrame(fulltranslist_all)


def addargs(parser):
    parser.add_argument('--fromartisatomic', default=False, action='store_true',
                        help='Read transitions from the artisatomic output instead of an ARTIS model folder.')
    parser.add_argument('-xmin', type=int, default=2000,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=30000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('-T', type=float, dest='T', default=6000.,
                        help='Temperature in Kelvin')
    parser.add_argument('-sigma_v', type=float, default=5500.,
                        help='Gaussian width in km/s')
    parser.add_argument('-gaussian_window', type=float, default=4,
                        help='Truncate Gaussian line profiles n sigmas from the centre')
    parser.add_argument('--include-permitted', action='store_true', default=False,
                        help='Also consider permitted lines')
    parser.add_argument('--print-lines', action='store_true', default=False,
                        help='Output details of matching line details to standard out')
    parser.add_argument('--no-plot', action='store_true', default=False,
                        help="Don't save a plot file")
    parser.add_argument('-elements', '--item', action='store', dest='elements',
                        type=str, nargs='*', default=['Fe'],
                        help="Examples: -elements Fe Co")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot estimated spectra from bound-bound transitions.')
    addargs(parser)
    args = parser.parse_args()

    # also calculate wavelengths outside the plot range to include lines whose
    # edges pass through the plot range
    plot_xmin_wide = args.xmin * (1 - args.gaussian_window * args.sigma_v / c)
    plot_xmax_wide = args.xmax * (1 + args.gaussian_window * args.sigma_v / c)

    elementslist = []
    for elcode in args.elements:
        atomic_number = elsymbols.index(elcode.title())
        if atomic_number == 26:
            Fe3overFe2 = 2.7  # number ratio
            ionlist = [
                iontuple(1, 0.2),
                iontuple(2, 1 / (1 + Fe3overFe2)),
                iontuple(3, Fe3overFe2 / (1 + Fe3overFe2)),
                # iontuple(4, 0.1)
            ]
        elif atomic_number == 27:
            ionlist = [iontuple(2, 0.5), iontuple(3, 0.5)]
        else:
            ionlist = default_ions
        elementslist.append((atomic_number, ionlist))

    if not args.fromartisatomic:
        artistransitions_allelements = get_artis_transitions(
            '.', plot_xmin_wide, plot_xmax_wide, args.include_permitted, [x[0] for x in elementslist])

    for (atomic_number, ions) in elementslist:
        elsymbol = elsymbols[atomic_number]
        ion_stage_list = [ion.ion_stage for ion in ions]

        if args.fromartisatomic:
            transition_filepath = os.path.join(
                PYDIR, '..', '..', 'artis-atomic', 'transition_guide', f'transitions_{elsymbol}.txt')
            transitions = get_artisatomic_transitions(transition_filepath)

            if transitions is None:
                print(f"ERROR: could not find transitions file for {elsymbol} at {transition_filepath}")
                return

            transitions = transitions[
                (transitions[:]['lambda_angstroms'] >= plot_xmin_wide) &
                (transitions[:]['lambda_angstroms'] <= plot_xmax_wide) &
                (transitions['ion_stage'].isin(ion_stage_list))
                # (transitions[:]['upper_has_permitted'] == 0)
            ]
            if not args.include_permitted:
                transitions = transitions[transitions[:]['forbidden'] == 1]
        else:
            transitions = artistransitions_allelements.copy().query('Z==@atomic_number and ion_stage in @ion_stage_list')

        transitions.sort_values(by='lambda_angstroms', inplace=True)

        print(f'{len(transitions):d} matching lines of {elsymbol}')

        if len(transitions) > 0:
            print('Generating spectra...')
            xvalues, yvalues = generate_spectra(
                transitions.copy(), atomic_number, ions, plot_xmin_wide, plot_xmax_wide, args)
            if not args.no_plot:
                make_plot(xvalues, yvalues, elsymbol, ions, args)


if __name__ == "__main__":
    main()
