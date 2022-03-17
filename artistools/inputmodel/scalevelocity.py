#!/usr/bin/env python3

import argparse
import math
# import os.path

# import numpy as np
# import pandas as pd

import artistools as at


def addargs(parser):

    parser.add_argument('-kescale', '-k', default=None,
                        help='Kinetic energy scale factor')

    parser.add_argument('-velscale', '-v', default=None,
                        help='Velocity scale factor')

    parser.add_argument('-inputfile', '-i', default='model.txt',
                        help='Path of input file')

    parser.add_argument('-outputfile', '-o', default='model_velscale{velscale:.2f}.txt',
                        help='Path of output model file')


def eval_mshell(dfmodel, t_model_init_seconds):
    dfmodel.eval('cellmass_grams = 10 ** logrho * 4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                     '* (1e5 * @t_model_init_seconds) ** 3', inplace=True)


def main(args=None, argsraw=None, **kwargs) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Scale the velocity of an ARTIS model, keeping mass constant and saving back to ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    dfmodel, t_model_init_days, _ = at.inputmodel.get_modeldata(args.inputfile)
    print(f'Read {args.inputfile}')

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60

    eval_mshell(dfmodel, t_model_init_seconds)

    print(dfmodel)

    assert (args.kescale is None) != (args.velscale is None)  # kescale or velscale must be specfied

    if args.kescale is not None:
        kescale = float(args.kescale)
        velscale = math.sqrt(kescale)
    elif args.velscale is not None:
        velscale = float(args.velscale)
        kescale = velscale ** 2

    print(f"Applying velocity factor of {velscale} (kinetic energy factor {kescale}) and conserving shell masses")

    dfmodel.velocity_inner *= velscale
    dfmodel.velocity_outer *= velscale

    dfmodel.eval('logrho = log10(cellmass_grams / ('
                     '4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                     ' * (1e5 * @t_model_init_seconds) ** 3))', inplace=True)

    eval_mshell(dfmodel, t_model_init_seconds)

    print(dfmodel)

    outputfile = args.outputfile.format(velscale=velscale, kescale=kescale)

    at.inputmodel.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, filename=outputfile)
    print(f'Saved {outputfile}')


if __name__ == "__main__":
    main()
