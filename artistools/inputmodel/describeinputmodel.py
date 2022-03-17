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
        print(f'Model contains {len(dfmodel)} Cartesian grid cells with vmax = {vmax} km/s')
    else:
        vmax = dfmodel['velocity_outer'].max()
        print(f'Model contains {len(dfmodel)} 1D spherical shells with vmax = {vmax} km/s')

    mass_msun = dfmodel['cellmass_grams'].sum() / 1.989e33

    print(f'M_{"tot":8s} {mass_msun:7.4f} MSun')
    speciesmasses = {}
    for column in dfmodel.columns:
        if column.startswith('X_'):
            species = column.replace('X_', '')
            speciesabund = np.dot(dfmodel[column], dfmodel['cellmass_grams'])
            if speciesabund > 0.:
                speciesmasses[species] = speciesabund

    for species, mass_g in speciesmasses.items():
        mass_msun = mass_g / 1.989e33
        print(f'M_{species:8s} {mass_msun:.3e} MSun')


if __name__ == "__main__":
    main()
