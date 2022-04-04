#!/usr/bin/env python3

import argparse
import math
# import os.path

import numpy as np
# import pandas as pd

from artistools import CustomArgHelpFormatter, get_atomic_number
import artistools.inputmodel


def addargs(parser):

    parser.add_argument('-inputfile', '-i', default='model.txt',
                        help='Path of input file or folder containing model.txt')

    parser.add_argument('-cell', '-mgi', default=None,
                        help='Focus on particular cell number (0-indexed)')

    parser.add_argument('--noabund', action='store_true',
                        help='Give total masses only, no nuclear or elemental abundances')

    parser.add_argument('--getabundances', action='store_true',
                        help='Get elemental abundance masses')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=CustomArgHelpFormatter,
            description='Scale the velocity of an ARTIS model, keeping mass constant and saving back to ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    print(f'Reading {args.inputfile}')
    dfmodel, t_model_init_days, vmax = artistools.inputmodel.get_modeldata(
        args.inputfile, get_abundances=args.getabundances)

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60
    print(f'Model is defined at {t_model_init_days} days ({t_model_init_seconds:.4f} seconds)')

    if 'pos_x_min' in dfmodel.columns:
        nonemptycells = sum(dfmodel['rho'] > 0.)
        print(f'Model contains {len(dfmodel)} Cartesian grid cells ({nonemptycells} nonempty) with '
              f'vmax = {vmax} km/s ({vmax / 299792.458:.2f} * c)')
        corner_vmax = math.sqrt(3 * vmax ** 2)
        print(f'  corner vmax: {corner_vmax:.2e} cm/s ({corner_vmax / 299792.458:.2f} * c)')
    else:
        vmax = dfmodel['velocity_outer'].max()
        print(f'Model contains {len(dfmodel)} 1D spherical shells with vmax = {vmax} km/s '
              f'({vmax / 299792.458:.2f} * c)')

    if args.cell is not None:
        mgi = int(args.cell)
        if mgi >= 0:
            print(f'Selected single cell mgi {mgi}:')
            dfmodel.query('inputcellid == (@mgi + 1)', inplace=True)
            print(dfmodel.iloc[0])

    mass_msun_rho = dfmodel['cellmass_grams'].sum() / 1.989e33

    mass_msun_isotopes = 0.
    mass_msun_elem = 0.
    speciesmasses = {}
    for column in dfmodel.columns:
        if column.startswith('X_'):
            species = column.replace('X_', '')
            speciesabund_g = np.dot(dfmodel[column], dfmodel['cellmass_grams'])

            species_mass_msun = speciesabund_g / 1.989e33
            atomic_number = get_atomic_number(species)
            if species[-1].isdigit():
                strtotiso = species.rstrip('0123456789') + '-isosum'
                speciesmasses[strtotiso] = speciesmasses.get(strtotiso, 0.) + speciesabund_g
                mass_msun_isotopes += species_mass_msun
            elif species.lower() != 'fegroup':
                mass_msun_elem += species_mass_msun

            if speciesabund_g > 0.:
                speciesmasses[species] = speciesabund_g

    print(f'M_{"tot_rho":9s} {mass_msun_rho:8.5f} MSun (density * volume)')
    if mass_msun_elem > 0.:
        print(f'M_{"tot_elem":9s} {mass_msun_elem:8.5f} MSun ({mass_msun_elem / mass_msun_rho * 100:6.2f}% of M_tot_rho)')

    print(f'M_{"tot_iso":9s} {mass_msun_isotopes:8.5f} MSun ({mass_msun_isotopes / mass_msun_rho * 100:6.2f}% of M_tot_rho, but can be < 100% if stable isotopes not tracked)')

    if not args.noabund:
        for species, mass_g in sorted(speciesmasses.items(), key=lambda x: (get_atomic_number(x[0]), x[0])):
            species_mass_msun = mass_g / 1.989e33
            massfrac = species_mass_msun / mass_msun_rho
            strcomment = ''
            if species.endswith('-isosum') and args.getabundances:
                elsymb = species.replace('-isosum', '')
                elem_mass = speciesmasses.get(elsymb, 0.)
                if elem_mass > 0.:
                    strcomment += f' ({mass_g / elem_mass * 100:6.2f}% of {elsymb} element mass)'
                if mass_g > elem_mass:
                    strcomment += ' ERROR! isotope sum is greater than element abundance'

            print(f'{species:9s} {species_mass_msun:.3e} Msun    massfrac {massfrac:.3e}{strcomment}')


if __name__ == "__main__":
    main()
