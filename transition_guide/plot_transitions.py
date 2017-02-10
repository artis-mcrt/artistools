#!/usr/bin/env python3
import argparse
import math
import os
from collections import namedtuple

import numpy as np
import pandas as pd
from astropy import constants as const
# from astropy import units as u

K_B = const.k_B.to('eV / K').value
c = const.c.to('km / s').value

PYDIR = os.path.dirname(os.path.abspath(__file__))
elsymbols = ['n'] + list(pd.read_csv(os.path.join(PYDIR, '..', 'elements.csv'))['symbol'].values)

Fe3overFe2 = 2.3        # number ratio of these ions

iontuple = namedtuple('ion', 'ion_stage number_fraction')

o_ions = [iontuple(1, 0.5),
          iontuple(2, 0.5)]

ca_ions = [iontuple(2, 1.0), ]

fe_ions = [
    # iontuple(1, 0.2),
    iontuple(2, 0.3),
    iontuple(3, 0.7),
    # iontuple(4, 0.1)
]

co_ions = [
    iontuple(2, 0.5),
    iontuple(3, 0.5)
]

elementslist = [(8, o_ions), (26, fe_ions), (27, co_ions)]
# elementslist = [(26, fe_ions),]
# elementslist = [(20, ca_ions)]

roman_numerals = ('', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
                  'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX')

TRANSITION_FILES_DIR = os.path.join('..', '..', 'artis-atomic', 'transition_guide')
SPECTRA_DIR = os.path.join(PYDIR, '..', 'spectra')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot estimated spectra from bound-bound transitions.')
    parser.add_argument('-xmin', type=int, default=3500,
                        help='Plot range: minimum wavelength in Angstroms')
    parser.add_argument('-xmax', type=int, default=7000,
                        help='Plot range: maximum wavelength in Angstroms')
    parser.add_argument('-T', type=float, dest='T', default=3000.0,
                        help='Temperature in Kelvin')
    parser.add_argument('-sigma_v', type=float, default=4000.0,  # 4000 matches the data
                        help='Gaussian width in km/s')
    parser.add_argument('-gaussian_window', type=float, default=4,
                        help='Truncate Gaussian line profiles n sigmas from the centre')
    parser.add_argument('--forbidden-only', action='store_true',
                        help='Only consider forbidden lines')
    parser.add_argument('--print-lines', action='store_true',
                        help='Output details of matching line details to standard out')
    args = parser.parse_args()

    # also calculate wavelengths outside the plot range to include lines whose
    # edges pass through the plot range
    plot_xmin_wide = args.xmin * (1 - args.gaussian_window * args.sigma_v / c)
    plot_xmax_wide = args.xmax * (1 + args.gaussian_window * args.sigma_v / c)

    for (atomic_number, ions) in elementslist:
        elsymbol = elsymbols[atomic_number]
        transition_filename = f'transitions_{elsymbol}.txt'
        if os.path.isfile(transition_filename):  # try the current directory first
            transitions = load_transitions(transition_filename)
        else:
            transition_file = os.path.join(TRANSITION_FILES_DIR, transition_filename)
            transitions = load_transitions(transition_file)

        ion_stage_list = [ion.ion_stage for ion in ions]
        # filter the line list
        transitions = transitions[
            (transitions[:]['lambda_angstroms'] >= plot_xmin_wide) &
            (transitions[:]['lambda_angstroms'] <= plot_xmax_wide) &
            (transitions['ion_stage'].isin(ion_stage_list))
            # (transitions[:]['upper_has_permitted'] == 0)
        ]
        if args.forbidden_only:
            transitions = transitions[transitions[:]['forbidden'] == 1]

        print(f'{len(transitions):d} matching lines of {elsymbol}')

        if len(transitions) > 0:
            print('Generating spectra...')
            xvalues, yvalues = generate_spectra(
                transitions, atomic_number, ions, plot_xmin_wide, plot_xmax_wide, args)

            make_plot(xvalues, yvalues, elsymbol, ions, args)


def load_transitions(transition_file):
    print(f"Loading '{transition_file}'...")
    if os.path.isfile(transition_file + '.tmp'):
        # read the sorted binary file (fast)
        transitions = pd.read_pickle(transition_file + '.tmp')
    else:
        # read the text file (slower)
        transitions = pd.read_csv(transition_file, delim_whitespace=True)
        transitions.sort_values(by='lambda_angstroms', inplace=True)

        # save the dataframe in binary format for next time
        # transitions.to_pickle(transition_file + '.tmp')

    return transitions


def generate_spectra(transitions, atomic_number, ions, plot_xmin_wide,
                     plot_xmax_wide, args):
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

            centre_index = int(round((line['lambda_angstroms'] - args.xmin) /
                                     plot_resolution))
            sigma_angstroms = line['lambda_angstroms'] * args.sigma_v / c
            sigma_gridpoints = int(math.ceil(sigma_angstroms / plot_resolution))
            window_left_index = max(int(centre_index - args.gaussian_window * sigma_gridpoints), 0)
            window_right_index = min(
                int(centre_index + args.gaussian_window * sigma_gridpoints),
                len(xvalues))

            for x in range(window_left_index, window_right_index):
                if 0 < x < len(xvalues):
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

    return


def make_plot(xvalues, yvalues, elsymbol, ions, args):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        len(ions) + 1, 1, sharex=True, figsize=(6, 6),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    yvalues_combined = np.zeros(len(xvalues))
    for ion_index in range(len(ions) + 1):
        if ion_index < len(ions):  # an ion subplot
            if max(yvalues[ion_index]) > 0.0:
                yvalues_normalised = yvalues[
                    ion_index] / max(yvalues[ion_index])
                yvalues_combined += yvalues_normalised * \
                    ions[ion_index].number_fraction
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
                obsdata = pd.read_csv(obsfile, delim_whitespace=True,
                                      header=None, names=[
                                          'lambda_angstroms', 'flux'])
                obsdata = obsdata[
                    (obsdata[:]['lambda_angstroms'] > args.xmin) &
                    (obsdata[:]['lambda_angstroms'] < args.xmax)]
                obsyvalues = obsdata[:]['flux'] * max(yvalues_combined) / max(obsdata[:]['flux'])
                ax[-1].plot(obsdata[:]['lambda_angstroms'], obsyvalues,
                            lw=1, color='black', label=serieslabel, zorder=-1)

            combined_label = ' + '.join([
                f'({ion.number_fraction:.1f} * {elsymbol} {roman_numerals[ion.ion_stage]})' for ion in ions])
            ax[-1].plot(xvalues, yvalues_combined, lw=1.5, label=combined_label)
            ax[-1].set_xlabel(r'Wavelength ($\AA$)')

        ax[ion_index].set_xlim(xmin=args.xmin, xmax=args.xmax)
        ax[ion_index].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
        ax[ion_index].set_ylabel(r'$\propto$ F$_\lambda$')

    # ax.set_ylim(ymin=-0.05,ymax=1.1)
    outfilename = f'transitions_{elsymbol}.pdf'
    print(f"Saving '{outfilename}'")
    fig.savefig(outfilename, format='pdf')
    plt.close()


main()
