"""Export elemental mass fractions from the ARTIS estimators to a text file."""

import argparse
import typing as t
from collections.abc import Mapping
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
from artistools.misc import print_saved
from artistools.misc import print_warning

DEFAULTOUTPUTNAME = "massfracs.txt"


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())
    addarg_timestep(parser, default=14, helptext="Timestep number to export")
    addarg_modelgridindex(parser, default=[0], helptext="Range of cell numbers to export")
    addarg_output(parser, kind="file", defaultname=DEFAULTOUTPUTNAME, helptext="Path to output file of mass fractions")


def get_massfraction_lines(
    estimcell: dict[str, t.Any], elmass: Mapping[int, float], tdays: float, modelgridindex: int, timestep: int
) -> list[str]:
    """Return the lines of the output file for one cell: a header line, then the mass fraction of each element."""
    numberdens = {}
    totaldens = 0.0  # number density times atomic mass summed over all elements
    for key, val in estimcell.items():
        if key.startswith("nnelement_"):
            elsymbol = key.removeprefix("nnelement_")
            atomic_number = get_atomic_number(elsymbol)
            assert atomic_number in elmass, f"Unrecognised element in estimator column {key}: {elsymbol}"
            numberdens[atomic_number] = val
            totaldens += val * elmass[atomic_number]

    if totaldens <= 0.0:
        msg = (
            f"The estimators of cell {modelgridindex} at timestep {timestep} give no element number density"
            " (nnelement_*). A run of the classic ARTIS code, or a model with no estimator files, has none"
        )
        raise ValueError(msg)

    lines = [f"{tdays}d shell {modelgridindex}\n"]
    massfracsum = 0.0
    for atomic_number, numberdensity in numberdens.items():
        massfrac = numberdensity * elmass[atomic_number] / totaldens
        massfracsum += massfrac
        lines.append(f"{atomic_number} {get_elsymbol(atomic_number)} {massfrac}\n")

    assert np.isclose(massfracsum, 1.0)
    return lines


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
    modelgridindexlist: list[int] = args.modelgridindex
    estimators = read_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindexlist)

    # a cell with no matter has no estimators. The file opens after the read of every cell, thus an error leaves no
    # empty file
    lines: list[str] = []
    emptycells = [mgi for mgi in modelgridindexlist if (timestep, mgi) not in estimators]
    for modelgridindex in modelgridindexlist:
        if modelgridindex not in emptycells:
            lines += get_massfraction_lines(
                estimators[timestep, modelgridindex], elmass, tdays, modelgridindex, timestep
            )

    if emptycells:
        celltext = ", ".join(map(str, emptycells))
        cellword = "cells" if len(emptycells) > 1 else "cell"
        message = f"The estimators hold no row for the {cellword} {celltext} at timestep {timestep}"
        if not lines:
            msg = f"{message}. A cell with no matter has no estimators"
            raise ValueError(msg)
        print_warning(f"{message}, thus the file leaves them out. A cell with no matter has no estimators")

    outfilename = args.outputfile
    with Path(outfilename).open("w", encoding="utf-8") as fout:
        fout.writelines(lines)

    print_saved(outfilename)
