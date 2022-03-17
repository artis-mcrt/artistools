#!/usr/bin/env python3

import argparse
# import math
# import os.path

import numpy as np
# import pandas as pd

from artistools import CustomArgHelpFormatter
import artistools.inputmodel


def addargs(parser):

    parser.add_argument('-inputfile', '-i', default='model.txt',
                        help='Path of input file or folder containing model.txt')

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

    if 'pos_x' in dfmodel.columns:
        nonemptycells = sum(dfmodel['rho'] > 0.)
        print(f'Model contains {len(dfmodel)} Cartesian grid cells ({nonemptycells} nonempty) with vmax = {vmax} km/s')
    else:
        vmax = dfmodel['velocity_outer'].max()
        print(f'Model contains {len(dfmodel)} 1D spherical shells with vmax = {vmax} km/s')

    mass_msun_rho = dfmodel['cellmass_grams'].sum() / 1.989e33

    mass_msun_isotopes = 0.
    mass_msun_elem = 0.
    speciesmasses = {}
    for column in dfmodel.columns:
        if column.startswith('X_'):
            species = column.replace('X_', '')
            speciesabund_g = np.dot(dfmodel[column], dfmodel['cellmass_grams'])

            species_mass_msun = speciesabund_g / 1.989e33
            if species[-1].isdigit():
                mass_msun_isotopes += species_mass_msun
            elif species.lower() != 'fegroup':
                mass_msun_elem += species_mass_msun

            if speciesabund_g > 0.:
                speciesmasses[species] = speciesabund_g

    print(f'M_{"tot_rho":8s} {mass_msun_rho:7.4f} MSun (density * volume)')
    if mass_msun_elem > 0.:
        print(f'M_{"tot_elem":8s} {mass_msun_elem:7.4f} MSun ({mass_msun_elem / mass_msun_rho * 100:6.2f}% of M_tot_rho)')

    print(f'M_{"tot_iso":8s} {mass_msun_isotopes:7.4f} MSun ({mass_msun_isotopes / mass_msun_rho * 100:6.2f}% of M_tot_rho, but can be small if stable isotopes not tracked)')

    for species, mass_g in speciesmasses.items():
        species_mass_msun = mass_g / 1.989e33
        massfrac = species_mass_msun / mass_msun_rho
        print(f'{species:8s} {species_mass_msun:.3e} Msun    massfrac {massfrac:.3e}')


if __name__ == "__main__":
    main()
