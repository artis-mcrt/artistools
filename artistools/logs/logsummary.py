"""Count the warnings and the errors in the logs of an ARTIS model, by kind of message.

Each MPI rank writes its own log, e.g. output_12-0.txt, and it logs the cells that it updates. The command reads the log
of rank 0 of each run folder. --ranks or --modelgridindex selects the logs of other ranks. A kind of message is its text
with each number replaced by N, thus the same warning for different cells and timesteps is one kind.
"""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

from artistools.logs.core import get_log_summary
from artistools.logs.core import get_logfiles
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import exit_with_error
from artistools.misc import get_model_folder
from artistools.misc import parse_cli_args
from artistools.misc import parse_range_list
from artistools.misc import print_detail
from artistools.misc import print_product


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())
    parser.add_argument("-ranks", default="0", help='The MPI ranks of the logs to read, e.g. 0-3,8 or "all"')
    addarg_modelgridindex(parser, helptext="Read the logs of the ranks that update these cells, in place of --ranks")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Count the warnings and the errors in the logs of an ARTIS model, by kind of message."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelpath = get_model_folder(args.modelpath)
    ranks: list[int] | None = None
    cells: list[int] | None = None
    if args.modelgridindex:
        cells = list(args.modelgridindex)
        rankstext = f"the ranks that update the cells {', '.join(str(cell) for cell in cells)} in each run folder"
        missingtext = f"no log of {rankstext}"
        hint = "Give --modelgridindex with the cells of the model"
    elif args.ranks == "all":
        rankstext = "each rank that has a log in each run folder"
        missingtext = "no log of a rank in a run folder"
        hint = "Make sure that a run folder holds the output_*-*.txt files of ARTIS"
    else:
        ranks = parse_range_list(args.ranks)
        rankstext = f"the ranks {', '.join(str(rank) for rank in ranks)}"
        missingtext = f"no log of {rankstext}"
        hint = "Give --ranks with ranks that ran"

    logfiles = get_logfiles(modelpath, ranks, cells)
    if not logfiles:
        exit_with_error(f"{modelpath} has {missingtext}", hint)
    print_detail(f"Reading {len(logfiles)} logs of {rankstext}")

    print_product(args, get_log_summary(logfiles))
