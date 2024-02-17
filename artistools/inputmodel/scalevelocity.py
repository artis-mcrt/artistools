#!/usr/bin/env python3
import argparse
import math
import typing as t

import pandas as pd

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-kescale", "-k", default=None, help="Kinetic energy scale factor")

    parser.add_argument("-velscale", "-v", default=None, help="Velocity scale factor")

    parser.add_argument("-inputfile", "-i", default="model.txt", help="Path of input file")

    parser.add_argument(
        "-outputfile", "-o", default="model_velscale{velscale:.2f}.txt", help="Path of output model file"
    )


def eval_mshell(dfmodel: pd.DataFrame, t_model_init_seconds: float) -> None:
    dfmodel = dfmodel.eval(
        "mass_g = 10 ** logrho * 4. / 3. * @math.pi * (vel_r_max_kmps ** 3 - vel_r_min_kmps ** 3)"
        "* (1e5 * @t_model_init_seconds) ** 3",
    )


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs) -> None:
    """Scale the velocity of an ARTIS model, keeping mass constant and saving back to ARTIS format."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=__doc__,
        )

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    dfmodel, t_model_init_days, _ = at.inputmodel.get_modeldata_tuple(args.inputfile)
    print(f"Read {args.inputfile}")

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60

    eval_mshell(dfmodel, t_model_init_seconds)

    print(dfmodel)

    assert (args.kescale is None) != (args.velscale is None)  # kescale or velscale must be specfied

    if args.kescale is not None:
        kescale = float(args.kescale)
        velscale = math.sqrt(kescale)
    elif args.velscale is not None:
        velscale = float(args.velscale)
        kescale = velscale**2

    print(f"Applying velocity factor of {velscale} (kinetic energy factor {kescale}) and conserving shell masses")

    dfmodel.vel_r_min_kmps *= velscale
    dfmodel.vel_r_max_kmps *= velscale

    dfmodel = dfmodel.eval(
        "logrho = log10(mass_g / ("
        "4. / 3. * @math.pi * (vel_r_max_kmps ** 3 - vel_r_min_kmps ** 3)"
        " * (1e5 * @t_model_init_seconds) ** 3))",
    )

    eval_mshell(dfmodel, t_model_init_seconds)

    print(dfmodel)

    outputfile = args.outputfile.format(velscale=velscale, kescale=kescale)

    at.inputmodel.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, outpath=outputfile)
    print(f"Saved {outputfile}")


if __name__ == "__main__":
    main()
