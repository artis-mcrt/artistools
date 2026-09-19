"""Export elemental mass fractions from the ARTIS estimators to a text file."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from artistools.atomic import get_atomic_masses
from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.estimators.core import read_estimators
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timestep
from artistools.misc import get_single_timestep
from artistools.misc import get_timestep_time
from artistools.misc import parse_cli_args
from artistools.misc import parse_range_list
from artistools.misc import print_saved

DEFAULTOUTPUTNAME = "massfracs.txt"


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())
    addarg_timestep(parser, default=14, helptext="Timestep number to export")
    addarg_modelgridindex(parser, default="0", helptext="Range of cell numbers to export")
    addarg_output(parser, kind="file", defaultname=DEFAULTOUTPUTNAME, helptext="Path to output file of mass fractions")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Export elemental mass fractions from the estimators to a text file."""
    args = parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    modelpath = Path(args.modelpath)
    timestep = get_single_timestep(args.timestep, modelpath)
    assert timestep is not None, "-timestep holds a default, thus it names a timestep"
    # the standard atomic weights, not the masses in compositiondata.txt, which only cover the elements that
    # ARTIS treated in detail. Weighting over a subset would renormalise the fractions to a partial total
    elmass = get_atomic_masses()
    tdays = get_timestep_time(modelpath, timestep)
    outfilename = args.outputfile
    with Path(outfilename).open("w", encoding="utf-8") as fout:
        modelgridindexlist = parse_range_list(args.modelgridindex)
        estimators = read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindexlist)
        for modelgridindex in modelgridindexlist:
            numberdens = {}
            totaldens = 0.0  # number density times atomic mass summed over all elements
            for key, val in estimators[timestep, modelgridindex].items():
                if key.startswith("nnelement_"):
                    elsymbol = key.removeprefix("nnelement_")
                    atomic_number = get_atomic_number(elsymbol)
                    assert atomic_number in elmass, f"Unrecognised element in estimator column {key}: {elsymbol}"
                    numberdens[atomic_number] = val
                    totaldens += val * elmass[atomic_number]
            massfracs = {
                atomic_number: numberdens[atomic_number] * elmass[atomic_number] / totaldens
                for atomic_number in numberdens
            }

            fout.write(f"{tdays}d shell {modelgridindex}\n")
            massfracsum = 0.0
            for atomic_number, value in massfracs.items():
                massfracsum += value
                fout.write(f"{atomic_number} {get_elsymbol(atomic_number)} {value}\n")

            assert np.isclose(massfracsum, 1.0)

    print_saved(outfilename)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
