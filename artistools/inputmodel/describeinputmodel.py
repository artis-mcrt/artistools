#!/usr/bin/env python3

import argparse
import math
# import os.path

import numpy as np
# import pandas as pd

import artistools.inputmodel


def addargs(parser):

    parser.add_argument('-inputfile', '-i', default='model.txt',
                        help='Path of input file or folder containing model.txt')

    parser.add_argument('--getabundances', action='store_true',
                        help='Get elemental abundance masses')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Scale the velocity of an ARTIS model, keeping mass constant and saving back to ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    dfmodeldata, t_model_init_days, _ = artistools.inputmodel.get_modeldata(args.inputfile, get_abundances=args.getabundances)
    print(f'Read {args.inputfile}')

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60
    print(f'Model is defined at {t_model_init_days} days ({t_model_init_seconds:.4f} seconds)')

    if 'velocity_outer' in dfmodeldata.columns:
        vmax = dfmodeldata['velocity_outer'].max()
        print(f'Model contains {len(dfmodeldata)} 1D spherical shells with vmax = {vmax} km/s')
    else:
        assert False   # 3D mode not implemented yet

    mass_msun = dfmodeldata['shellmass_grams'].sum() / 1.989e33
    print(f'M_{"tot":8s} {mass_msun:7.4f} Msun')
    speciesmasses = {}
    for column in dfmodeldata.columns:
        if column.startswith('X_'):
            species = column.replace('X_', '')
            speciesmasses[species] = np.dot(dfmodeldata[column], dfmodeldata['shellmass_grams'])

    for species, mass_g in speciesmasses.items():
        mass_msun = mass_g / 1.989e33
        print(f'M_{species:8s} {mass_msun:7.4f} Msun')


if __name__ == "__main__":
    main()
