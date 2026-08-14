"""Export elemental mass fractions from the ARTIS estimators to a text file."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    at.add_modelpath_arg(parser, default=Path())
    at.add_timestep_arg(parser, kind="int", default=14, helptext="Timestep number to export")
    parser.add_argument("-modelgridindex", "-cell", default="0-9", help="Range of cell numbers to export")
    at.add_outputpath_arg(parser, default="massfracs.txt", helptext="Path to output file of mass fractions")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Export elemental mass fractions from the estimators to a text file."""
    args = at.parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    modelpath = Path(args.modelpath)
    timestep = args.timestep
    elmass: dict[int, float] = dict(at.get_composition_data(modelpath).select("Z", "mass").iter_rows())
    tdays = at.get_timestep_time(modelpath, timestep)
    outfilename = args.outputpath
    with Path(outfilename).open("w", encoding="utf-8") as fout:
        modelgridindexlist = at.parse_range_list(args.modelgridindex)
        estimators = at.estimators.read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindexlist)
        for modelgridindex in modelgridindexlist:
            numberdens = {}
            totaldens = 0.0  # number density times atomic mass summed over all elements
            for key, val in estimators[timestep, modelgridindex].items():
                if key.startswith("nnelement_"):
                    atomic_number = at.get_atomic_number(key.removeprefix("nnelement_"))
                    if atomic_number not in elmass:
                        # the estimators can carry an element that compositiondata.txt gives no mass for, and
                        # there is no mass to weight it by. Report the fractions among the rest, which still sum
                        # to one, rather than failing on the lookup
                        print(f"WARNING: no mass in compositiondata.txt for Z={atomic_number}, excluding it")
                        continue
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

    at.print_saved(outfilename)


if __name__ == "__main__":
    main()
