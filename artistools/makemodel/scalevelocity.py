#!/usr/bin/env python3

import argparse
import math
# import os.path

# import numpy as np
# import pandas as pd

import artistools as at


def addargs(parser):
    parser.add_argument('-kescalefactor', '-s',
                        default=0.5,
                        help='Kinetic energy scale factor')
    parser.add_argument('-inputfile', '-i',
                        default='model.txt',
                        help='Path of input file')
    parser.add_argument('-outputfile', '-o',
                        default='model_velscale.txt',
                        help='Path of output file')


def eval_mshell(dfmodeldata, t_model_init_seconds):
    dfmodeldata.eval('shellmass_grams = 10 ** logrho * 4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                     '* (1e5 * @t_model_init_seconds) ** 3', inplace=True)


def main(args=None, argsraw=None, **kwargs) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Scale the velocity of an ARTIS model, keeping mass constant and saving back to ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    dfmodeldata, t_model_init_days = at.get_modeldata(args.inputfile)
    print(f'Read {args.inputfile}')

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60

    eval_mshell(dfmodeldata, t_model_init_seconds)

    print(dfmodeldata)

    velocityscalefactor = math.sqrt(args.kescalefactor)

    dfmodeldata.velocity_inner *= velocityscalefactor
    dfmodeldata.velocity_outer *= velocityscalefactor

    dfmodeldata.eval('logrho = log10(shellmass_grams / ('
                     '4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                     ' * (1e5 * @t_model_init_seconds) ** 3))', inplace=True)

    eval_mshell(dfmodeldata, t_model_init_seconds)

    print(dfmodeldata)

    at.save_modeldata(dfmodeldata, t_model_init_days, args.outputfile)
    print(f'Saved {args.outputfile}')


if __name__ == "__main__":
    main()
