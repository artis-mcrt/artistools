#!/usr/bin/env python3
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", default="massfracs.txt", help="Path to output file of mass fractions")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter, description="Create solar r-process pattern in ARTIS format."
        )

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    modelpath: Path = Path()
    timestep = 14
    elmass = {el.Z: el.mass for _, el in at.get_composition_data(modelpath).iterrows()}
    outfilename = args.outputpath
    with Path(outfilename).open("w", encoding="utf-8") as fout:
        modelgridindexlist = range(10)
        estimators = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindexlist)
        for modelgridindex in modelgridindexlist:
            tdays = estimators[timestep, modelgridindex]["tdays"]

            numberdens = {}
            totaldens = 0.0  # number density times atomic mass summed over all elements
            for key, val in estimators[timestep, modelgridindex].items():
                if key.startswith("nnelement_"):
                    atomic_number = at.get_atomic_number(key.removeprefix("nnelement_"))
                    numberdens[atomic_number] = val
                    totaldens += numberdens[atomic_number] * elmass[atomic_number]
            massfracs = {
                atomic_number: numberdens[atomic_number] * elmass[atomic_number] / totaldens
                for atomic_number in numberdens
            }

            fout.write(f"{tdays}d shell {modelgridindex}\n")
            massfracsum = 0.0
            for atomic_number, value in massfracs.items():
                massfracsum += value
                fout.write(f"{atomic_number} {at.get_elsymbol(atomic_number)} {value}\n")

            assert np.isclose(massfracsum, 1.0)

    print(f"Saved {outfilename}")


if __name__ == "__main__":
    main()
