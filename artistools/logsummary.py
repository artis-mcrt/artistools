"""Count the warnings and the errors in the logs of an ARTIS model, by kind of message.

Each MPI rank writes its own log, e.g. output_12-0.txt, and it logs the cells that it updates. The command reads the log
of rank 0 of each run folder. -ranks or -modelgridindex selects the logs of other ranks. A kind of message is its text
with each number replaced by N, thus the same warning for different cells and timesteps is one kind.
"""

import argparse
import re
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import get_model_folder
from artistools.misc import get_run_subfolders
from artistools.misc import parallel_map
from artistools.misc import parse_cli_args
from artistools.misc import parse_range_list
from artistools.misc import print_detail
from artistools.misc import print_product
from artistools.misc import zopen
from artistools.misc.modelinfo import get_dfrankassignments
from artistools.misc.modelinfo import get_npts_model
from artistools.misc.modelinfo import get_ranks_of_cells_in_layout

# the current ARTIS tags a message with [warning] or [error], and an older ARTIS writes WARNING or ERROR
RE_LEVEL = re.compile(r"\[(warning|error)\]|\b(WARNING|ERROR)\b")
RE_TIMESTEP = re.compile(r"timestep (\d+): time before update grid")
RE_CELL = re.compile(r"\b(?:modelcell|modelgridindex|cell) (\d+)")
RE_LOGFILENAME = re.compile(r"output_(\d+)-0\.txt(?:\.gz|\.xz|\.zst)?")
RE_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

MESSAGESCHEMA = {
    "level": pl.String,
    "kind": pl.String,
    "timestep": pl.Int32,
    "modelgridindex": pl.Int32,
    "message": pl.String,
}


def read_log_messages(logpath: Path) -> pl.DataFrame:
    """Return one row for each warning and error of a log, with its kind, its timestep, its cell, and its text.

    A progress text can come before the tag on the same line, e.g. "Finding T_e in cell 4...[warning] ...", thus the
    message starts at the tag. A message before the first timestep has no timestep.
    """
    rows: list[tuple[str, str, int | None, int | None, str]] = []
    timestep: int | None = None
    with zopen(logpath) as flog:
        for line in flog:
            if (timestepmatch := RE_TIMESTEP.search(line)) is not None:
                timestep = int(timestepmatch.group(1))
                continue
            if (levelmatch := RE_LEVEL.search(line)) is None:
                continue
            message = line[levelmatch.start() :].strip()
            level = (levelmatch.group(1) or levelmatch.group(2)).lower()
            cellmatch = RE_CELL.search(message)
            rows.append((
                level,
                RE_NUMBER.sub("N", message),
                timestep,
                int(cellmatch.group(1)) if cellmatch is not None else None,
                message,
            ))
    return pl.DataFrame(rows, schema=MESSAGESCHEMA, orient="row")


def get_logfiles(modelpath: Path, ranks: Sequence[int] | None, cells: Sequence[int] | None = None) -> list[Path]:
    """Return the log of each selected rank in each run folder, or of each rank of the folder if ranks is None.

    A run that stops before it writes an estimator file has no run folder in get_runfolders, but its log can hold the
    error. Thus the search reads each subfolder of the model folder. A restart can use a different number of ranks,
    thus None finds the ranks of each folder from the names of its logs, and cells finds the ranks of these cells in
    the layout of each folder.
    """
    return [
        logpath
        for folder in get_run_subfolders(modelpath)
        for rank in get_folder_ranks(modelpath, folder, ranks, cells)
        if (logpath := firstexisting_or_none(f"output_{rank}-0.txt", folder=folder, search_subfolders=False))
        is not None
    ]


def get_folder_ranks(
    modelpath: Path, folder: Path, ranks: Sequence[int] | None, cells: Sequence[int] | None
) -> Sequence[int]:
    """Return the ranks to read in a run folder: the given ranks, the ranks of the cells, or each rank with a log.

    The cells take the modelgridrankassignments.out of the folder. A folder with no such file takes the blocks of
    ARTIS for the number of rank logs in the folder, because the file of the model folder can come from a later run.
    """
    if cells is None:
        return get_folder_log_ranks(folder) if ranks is None else ranks
    lzrankassignments = get_dfrankassignments(folder)
    folderlogranks = get_folder_log_ranks(folder)
    cellranks = get_ranks_of_cells_in_layout(
        np.asarray(cells, dtype=np.int64),
        get_npts_model(modelpath),
        lzrankassignments.collect() if lzrankassignments is not None else None,
        lambda: max(folderlogranks, default=0) + 1,
    )
    return sorted({int(rank) for rank in cellranks})


def get_folder_log_ranks(folder: Path) -> list[int]:
    """Return the MPI rank of each log in a folder, from the file names, e.g. 12 for output_12-0.txt.gz."""
    return sorted({
        int(namematch.group(1))
        for path in folder.glob("output_*-0.txt*")
        if (namematch := RE_LOGFILENAME.fullmatch(path.name)) is not None
    })


def summarise_log_messages(dfmessages: pl.DataFrame) -> pl.DataFrame:
    """Return one row for each kind of message, with its count, its timesteps, its number of cells, and an example."""
    return (
        dfmessages
        .group_by("level", "kind")
        .agg(
            count=pl.len(),
            first_timestep=pl.col("timestep").min(),
            last_timestep=pl.col("timestep").max(),
            cells=pl.col("modelgridindex").drop_nulls().n_unique(),
            example=pl.col("message").first(),
        )
        .sort(["count", "kind"], descending=[True, False])
        .select("level", "count", "first_timestep", "last_timestep", "cells", "kind", "example")
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())
    parser.add_argument(
        "-ranks", default="0", help='The MPI ranks of the logs to read, e.g. 0-3,8 or "all". The default is 0'
    )
    addarg_modelgridindex(parser, helptext="Read the logs of the ranks that update these cells, in place of -ranks")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Count the warnings and the errors in the logs of an ARTIS model, by kind of message."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelpath = get_model_folder(args.modelpath)
    ranks: list[int] | None
    cells: list[int] | None = None
    if args.modelgridindex:
        ranks, cells = None, list(args.modelgridindex)
    elif args.ranks == "all":
        ranks = None
    else:
        ranks = parse_range_list(args.ranks)

    logfiles = get_logfiles(modelpath, ranks, cells)
    if not logfiles:
        exit_with_error(f"{modelpath} has no log of the ranks {args.ranks}", "Give -ranks with ranks that ran")
    if cells is not None:
        rankstext = f"the ranks that update the cells {', '.join(str(cell) for cell in cells)} in each run folder"
    elif ranks is None:
        rankstext = "each rank that has a log in each run folder"
    else:
        rankstext = f"the ranks {', '.join(str(rank) for rank in ranks)}"
    print_detail(f"reading {len(logfiles)} logs of {rankstext}")

    dfmessages = pl.concat(parallel_map(read_log_messages, logfiles, desc="logs", unit="log"))
    print_product(args, summarise_log_messages(dfmessages))
