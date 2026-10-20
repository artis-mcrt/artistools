"""Count the warnings and the errors in the logs of an ARTIS model, by kind of message.

Each MPI rank writes its own log, e.g. output_12-0.txt, and it logs the cells that it updates. The command reads the log
of rank 0 of each run folder. -ranks or -modelgridindex selects the logs of other ranks. A kind of message is its text
with each number replaced by N, thus the same warning for different cells and timesteps is one kind.
"""

import argparse
import filecmp
import itertools
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
from artistools.misc import print_warning
from artistools.misc import zopen
from artistools.misc.cliutils import format_range_list
from artistools.misc.fileio import get_decompress_errors
from artistools.misc.fileio import without_compressed_extension
from artistools.misc.modelinfo import get_dfrankassignments
from artistools.misc.modelinfo import get_npts_model
from artistools.misc.modelinfo import get_ranks_of_cells_in_layout
from artistools.misc.remote import on_model_host

# the current ARTIS tags a message with [warning] or [error], and an older ARTIS writes WARNING or ERROR
RE_LEVEL = re.compile(r"\[(warning|error)\]|\b(WARNING|ERROR)\b")
RE_TIMESTEP = re.compile(r"timestep (\d+): time before update grid")
RE_CELL = re.compile(r"\b(?:modelcell|modelgridindex|cell) (\d+)")
RE_MPIRANKS = re.compile(r"rank \d+ of \[0\.\.(\d+)\] in MPI_COMM_WORLD")
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
            # a tag can follow the progress text on the line of a timestep, thus the line can also hold a message
            if (timestepmatch := RE_TIMESTEP.search(line)) is not None:
                timestep = int(timestepmatch.group(1))
            if (levelmatch := RE_LEVEL.search(line)) is None:
                continue
            message = line[levelmatch.start() :].strip()
            level = (levelmatch.group(1) or levelmatch.group(2)).lower()
            # the progress text before the tag can name the cell, e.g. "Finding T_e in cell 4...[warning] ...", thus
            # the search reads the message first and then the whole line
            cellmatch = RE_CELL.search(message) or RE_CELL.search(line)
            rows.append((
                level,
                RE_NUMBER.sub("N", message),
                timestep,
                int(cellmatch.group(1)) if cellmatch is not None else None,
                message,
            ))
    return pl.DataFrame(rows, schema=MESSAGESCHEMA, orient="row")


@on_model_host
def get_logfiles(modelpath: Path, ranks: Sequence[int] | None, cells: Sequence[int] | None = None) -> list[Path]:
    """Return the log of each selected rank in each run folder, or of each rank of the folder if ranks is None.

    A run that stops before it writes an estimator file has no run folder in get_runfolders, but its log can hold the
    error. Thus the search reads each subfolder of the model folder. A restart can use a different number of ranks,
    thus None finds the ranks of each folder from the names of its logs, and cells finds the ranks of these cells in
    the layout of each folder. The host of a remote model finds the logs. A folder with logs that has no log of a
    selected rank gives a warning, because the summary then has no messages of that rank of that run.
    """
    logfiles: list[Path] = []
    for folder in get_run_subfolders(modelpath):
        folderlogranks = get_folder_log_ranks(folder)
        missingranks: list[int] = []
        for rank in get_folder_ranks(modelpath, folder, ranks, cells):
            logpath = firstexisting_or_none(f"output_{rank}-0.txt", folder=folder, search_subfolders=False)
            if logpath is not None:
                logfiles.append(logpath)
            else:
                missingranks.append(rank)
        # a folder with no log is not a run folder, e.g. a folder of plots, thus only a folder with logs gives a warning
        if missingranks and folderlogranks:
            print_warning(
                f"{folder} has no log of the ranks {format_range_list(missingranks)}, thus the summary leaves out their"
                " messages of this run"
            )
    return get_unique_logfiles(modelpath, logfiles)


def get_unique_logfiles(modelpath: Path, logfiles: Sequence[Path]) -> list[Path]:
    """Return the logs with no copy: a link to a different log, or a log of the model folder that repeats a run log.

    A model folder can hold a copy or a link of the log of its last run, e.g. output_0-0.txt, and its messages then
    counted two times. The run subfolders come first in get_run_subfolders, thus the log of the run stays. Two retries
    in two run folders can stop with the same error and write the same text, thus only a log of the model folder can
    be a copy by its text. A copy has the name of its log, thus only logs of one name need a comparison, and a run of
    2000 ranks stays fast. The copy can have a different compression, thus the name leaves out that extension.
    """
    uniquelogs: list[Path] = []
    uniquelogsofname: dict[str, list[Path]] = {}
    # a local run writes its estimator files into the model folder, thus its log there is a run and not a copy
    modelfolderisrun = any(modelpath.glob("estimators_*.out*"))
    for logfile in logfiles:
        samename = uniquelogsofname.setdefault(without_compressed_extension(logfile).name, [])
        comparetext = logfile.parent == modelpath and not modelfolderisrun
        if all(not are_copies(logfile, uniquelog, comparetext=comparetext) for uniquelog in samename):
            samename.append(logfile)
            uniquelogs.append(logfile)
    return uniquelogs


def are_copies(logfile: Path, otherlogfile: Path, comparetext: bool) -> bool:
    """Return True if two logs are one file, or, with comparetext, if they hold the same text after decompression."""
    if logfile.resolve() == otherlogfile.resolve():
        return True
    if not comparetext:
        return False
    try:
        if logfile == without_compressed_extension(logfile) and otherlogfile == without_compressed_extension(
            otherlogfile
        ):
            # two text files with the same text have the same bytes, thus a different size is a different log
            return logfile.stat().st_size == otherlogfile.stat().st_size and filecmp.cmp(
                logfile, otherlogfile, shallow=False
            )
        # two compressed files of one text can have different bytes, e.g. a different header of gzip
        with zopen(logfile) as flog, zopen(otherlogfile) as fotherlog:
            return all(line == otherline for line, otherline in itertools.zip_longest(flog, fotherlog))
    except (OSError, EOFError, UnicodeDecodeError, *get_decompress_errors()):
        # an unreadable log is not a copy, and the reader of that log then reports the problem
        return False


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
    nprocs = get_logged_rank_count(folder)
    if lzrankassignments is None and nprocs is None:
        # the logs of the highest ranks can be missing, thus the names of the logs do not give the number of ranks
        print_warning(
            f"{folder} has no modelgridrankassignments.out, and its rank-0 log gives no number of ranks, thus the"
            " ranks of the cells are not known, and the summary leaves out the logs of this folder"
        )
        return []

    def get_nprocs_of_folder() -> int:
        assert nprocs is not None, "the branch above returns when the folder gives no number of ranks"
        return nprocs

    cellranks = get_ranks_of_cells_in_layout(
        np.asarray(cells, dtype=np.int64),
        get_npts_model(modelpath),
        lzrankassignments.collect() if lzrankassignments is not None else None,
        get_nprocs_of_folder,
    )
    return sorted({int(rank) for rank in cellranks})


def get_logged_rank_count(folder: Path) -> int | None:
    """Return the number of MPI ranks that the rank-0 log of a run folder gives, e.g. "rank 0 of [0..3]", or None."""
    logpath = firstexisting_or_none("output_0-0.txt", folder=folder, search_subfolders=False)
    if logpath is None:
        return None
    with zopen(logpath) as flog:
        # ARTIS writes the MPI layout near the start of the log, thus the search reads only the first lines
        for _, line in zip(range(200), flog, strict=False):
            if (rankmatch := RE_MPIRANKS.search(line)) is not None:
                return int(rankmatch.group(1)) + 1
    return None


def get_folder_log_ranks(folder: Path) -> list[int]:
    """Return the MPI rank of each log in a folder, from the file names, e.g. 12 for output_12-0.txt.gz."""
    return sorted({
        int(namematch.group(1))
        for path in folder.glob("output_*-0.txt*")
        if (namematch := RE_LOGFILENAME.fullmatch(path.name)) is not None
    })


def summarise_log_messages(dfmessages: pl.DataFrame) -> pl.DataFrame:
    """Return one row for each kind of message, with its count, its timesteps, its number of cells, and an example."""
    return combine_log_summaries([get_partial_log_summary(dfmessages)])


def get_partial_log_summary(dfmessages: pl.DataFrame) -> pl.DataFrame:
    """Return one row for each kind of message of one log, with its count, its timesteps, its cells, and an example.

    The cells stay a list, thus combine_log_summaries can count each cell of all the logs one time.
    """
    return dfmessages.group_by("level", "kind", maintain_order=True).agg(
        count=pl.len(),
        first_timestep=pl.col("timestep").min(),
        last_timestep=pl.col("timestep").max(),
        cells=pl.col("modelgridindex").drop_nulls().unique(),
        example=pl.col("message").first(),
    )


def combine_log_summaries(partialsummaries: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Return the summary of all the logs from the partial summary of each log, with the example of the first log."""
    return (
        pl
        .concat(partialsummaries)
        .group_by("level", "kind", maintain_order=True)
        .agg(
            pl.col("count").sum(),
            pl.col("first_timestep").min(),
            pl.col("last_timestep").max(),
            pl.col("cells").list.explode(keep_nulls=False, empty_as_null=False).n_unique(),
            pl.col("example").first(),
        )
        .sort(["count", "kind"], descending=[True, False])
        .select("level", "count", "first_timestep", "last_timestep", "cells", "kind", "example")
    )


def read_log_summary(logfile: Path) -> pl.DataFrame:
    """Return the partial summary of one log."""
    return get_partial_log_summary(read_log_messages(logfile))


@on_model_host
def get_log_summary(logfiles: list[Path]) -> pl.DataFrame:
    """Return the summary of the messages of the logs.

    A run of 2000 ranks can repeat a warning of each cell at each timestep, thus the messages of all the logs did not
    fit in the memory. Each log gives its own summary first, and the summaries then combine. The host of remote logs
    reads them, because only the summary goes through ssh.
    """
    return combine_log_summaries(parallel_map(read_log_summary, logfiles, desc="logs", unit="log"))


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
    print_detail(f"Reading {len(logfiles)} logs of {rankstext}")

    print_product(args, get_log_summary(logfiles))
