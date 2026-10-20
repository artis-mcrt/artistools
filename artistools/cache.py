"""Make, list, or delete the parquet caches that artistools reads in place of the text output of an ARTIS model.

A reader of artistools converts a text file to a parquet cache when it first reads the file. Later commands read the
cache. "make" does all the conversions at one time, e.g. in the job script after a run of ARTIS. Each later command
then reads its data quickly. "list" gives the caches and their sizes. "delete" deletes the caches and the leftover
files, which are the temporary files and folders of a conversion that stopped.
"""

import argparse
import re
import shutil
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.atomic import get_linelist_pldf
from artistools.estimators import scan_estimators
from artistools.inputmodel import get_initelemabundances
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_action
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_table_output
from artistools.misc import confirm_deletion
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import get_model_folder
from artistools.misc import get_runfolders
from artistools.misc import parse_cli_args
from artistools.misc import path_is_dir
from artistools.misc import print_detail
from artistools.misc import print_or_write_table
from artistools.misc import print_warning
from artistools.misc import require_action
from artistools.misc import resolve_table_outputfile
from artistools.misc.remote import check_local_path
from artistools.packets import get_packets_batch_parquet_paths

if t.TYPE_CHECKING:
    from collections.abc import Callable

# the names of the caches that the readers write: the input model, the line list, the estimators, and the batches of
# the packets and the virtual packets. A file of a different name, e.g. science.out.parquet, belongs to the user
CACHE_BASENAME: t.Final = (
    r"(?:model\.txt|abundances\.txt|linelist\.out|estimators_allranks\.out|estimbatch\d+_\d+_\d+\.out"
    r"|v?packetsbatch\d+_\d+_\d+\.out)\.parquet"
)

# a cache of an artistools version before 2026-10 also has the suffix .tmp
CACHE_NAME_PATTERN: t.Final = re.compile(rf"{CACHE_BASENAME}(?:\.tmp)?")

# write_parquet_atomic writes a cache to a temporary file of the name of mkstemp, and a conversion of estimators
# writes its parts into a temporary folder. A conversion that stops, e.g. at the time limit of a job, can leave them
LEFTOVER_NAME_PATTERN: t.Final = re.compile(rf"\.{CACHE_BASENAME}\.(?:partial\w+\.partial|parts\w+)")


def get_cache_folders(modelfolder: Path) -> list[Path]:
    """Return the model folder and its subfolders, e.g. packets and the run folders, where the readers put caches."""
    return [modelfolder, *sorted(child for child in modelfolder.iterdir() if child.is_dir())]


def find_cache_files(modelfolder: Path) -> tuple[list[Path], list[Path]]:
    """Return the parquet caches of a model, and the leftover files of the conversions that stopped.

    The lock files of the caches stay out of both lists. A process that waits for a lock keeps the lock file open,
    thus a deletion can give the lock to two writers at one time.
    """
    caches: list[Path] = []
    leftovers: list[Path] = []
    for folder in get_cache_folders(modelfolder):
        for path in sorted(folder.iterdir()):
            if CACHE_NAME_PATTERN.fullmatch(path.name) and path.is_file():
                caches.append(path)
            elif LEFTOVER_NAME_PATTERN.fullmatch(path.name):
                leftovers.append(path)
    return caches, leftovers


def get_size_bytes(path: Path) -> int:
    """Return the size of a file, or the total size of the files in a folder."""
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return path.stat().st_size


def get_cache_table(modelfolder: Path) -> pl.DataFrame:
    """Return one row for each cache and each leftover file, with the columns path, kind, and size_mb."""
    caches, leftovers = find_cache_files(modelfolder)
    rows = [
        (str(path.relative_to(modelfolder)), kind, get_size_bytes(path) / 1e6)
        for paths, kind in ((caches, "cache"), (leftovers, "leftover"))
        for path in paths
    ]
    return pl.DataFrame(rows, schema={"path": pl.String, "kind": pl.String, "size_mb": pl.Float64}, orient="row")


def make_inputmodel_caches(modelfolder: Path) -> str | None:
    """Read model.txt and abundances.txt. The readers then write a cache of each large file.

    Return None for a model folder with no model.txt.
    """
    if firstexisting_or_none("model.txt", folder=modelfolder, search_subfolders=False) is None:
        return None
    get_modeldata(modelfolder, printwarningsonly=True)
    if firstexisting_or_none("abundances.txt", folder=modelfolder, search_subfolders=False) is not None:
        get_initelemabundances(modelfolder, printwarningsonly=True)
    return "the reader keeps a cache of each input file larger than 2 MiB"


def make_linelist_cache(modelfolder: Path) -> str | None:
    """Read the line list. The reader then writes its cache. Return None for a model with no line list."""
    if firstexisting_or_none(["linelist.out", "linestat.out"], folder=modelfolder, search_subfolders=False) is None:
        return None
    get_linelist_pldf(modelfolder)
    return "linelist.out.parquet"


def make_estimator_caches(modelfolder: Path) -> str | None:
    """Scan the estimators. The reader then converts the files of each run folder.

    Return None for a model with no estimator files.
    """
    if not get_runfolders(modelfolder):
        return None
    scan_estimators(modelfolder)
    return "one cache for each batch of ranks in each run folder"


def has_packet_files(modelfolder: Path, virtual: bool) -> bool:
    """Return whether the model folder or one of its subfolders holds a packet text file or a batch cache.

    The readers search the same folders. A model with only some of the files is not a model with no data, thus the
    conversion then gives an error that names the missing file.
    """
    # the text files are vpackets_<rank>.out and packets00_<rank>.out, plain or compressed
    textpattern, strpacket = ("vpackets_*.out*", "vpackets") if virtual else ("packets00_*.out*", "packets")
    return any(
        any(folder.glob(pattern))
        for folder in (modelfolder, *(subfolder for subfolder in modelfolder.iterdir() if subfolder.is_dir()))
        for pattern in (textpattern, f"{strpacket}batch*.parquet*")
    )


def make_packet_caches(modelfolder: Path, virtual: bool) -> str | None:
    """Convert the packet files of each batch of ranks. Return None for a model with no such packet files.

    ARTIS writes the virtual packet files only with VPKT_WRITE_CONTRIBS. A model with vpkt.txt can thus have none.
    """
    # the conversion runs on this host, thus a remote model gives an error and not "no data"
    check_local_path(modelfolder)
    if not has_packet_files(modelfolder, virtual=virtual):
        return None
    nprocs_read, parquetfiles = get_packets_batch_parquet_paths(modelfolder, virtual=virtual)
    return f"{len(parquetfiles)} caches of the packets of {nprocs_read} ranks"


def make_caches(modelfolder: Path) -> pl.DataFrame:
    """Make each cache of the model, and return one row for each kind of data, with its status and its detail.

    Each kind of data has a separate conversion, thus an error in one kind gives the status "error", and the others
    still convert. The status is "ok", "no data", or "error".
    """
    # each step gives None when the model has no text files of its kind
    steps: list[tuple[str, Callable[[], str | None], str]] = [
        ("input model", lambda: make_inputmodel_caches(modelfolder), "the model folder holds no model.txt"),
        ("line list", lambda: make_linelist_cache(modelfolder), "the model folder holds no linelist.out"),
        ("estimators", lambda: make_estimator_caches(modelfolder), "the model holds no estimator files"),
        ("packets", lambda: make_packet_caches(modelfolder, virtual=False), "the model holds no packet files"),
        (
            "virtual packets",
            lambda: make_packet_caches(modelfolder, virtual=True),
            "the model holds no virtual packet files",
        ),
    ]
    rows: list[tuple[str, str, str]] = []
    for kind, step, nodatadetail in steps:
        try:
            detail = step()
        except (OSError, ValueError, AssertionError, pl.exceptions.PolarsError) as exc:
            rows.append((kind, "error", f"{type(exc).__name__}: {exc}"))
            continue
        rows.append((kind, "no data", nodatadetail) if detail is None else (kind, "ok", detail))
    return pl.DataFrame(rows, schema={"data": pl.String, "status": pl.String, "detail": pl.String}, orient="row")


def delete_caches(modelfolder: Path, force: bool) -> None:
    """Delete the caches and the leftover files of the model after the user agrees."""
    caches, leftovers = find_cache_files(modelfolder)
    paths = [*caches, *leftovers]
    if not paths:
        print(f"{modelfolder} holds no cache")
        return
    confirm_deletion(paths, force)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    print(f"Deleted {len(paths)} files. The next command that reads each kind of data converts the text files again")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_action(
        parser,
        choices=["make", "list", "delete"],
        helptext=(
            "make: convert the text output to the parquet caches. list: give the caches and their sizes. delete:"
            " delete the caches and the leftover files of the conversions that stopped"
        ),
    )
    addarg_modelpath(parser, positional=True, default=Path())
    addarg_table_output(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Make, list, or delete the parquet caches that artistools reads in place of the text output of a model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    require_action(args)
    modelfolder = get_model_folder(args.modelpath)
    if not path_is_dir(modelfolder):
        exit_with_error(f"The model folder {modelfolder} does not exist", "Give a model folder")

    outputfile = resolve_table_outputfile(args, f"cache_{args.action}.txt") if args.action != "delete" else None
    match args.action:
        case "make":
            dfresults = make_caches(modelfolder)
            print_or_write_table(args, dfresults, outputfile)
            if (dfresults["status"] == "error").any():
                exit_with_error("artistools could not make all the caches", "The table above gives each error")
        case "list":
            dftable = get_cache_table(modelfolder)
            print_or_write_table(args, dftable, outputfile)
            print_detail(f"total: {dftable['size_mb'].sum():.1f} MB in {dftable.height} files")
            if (dftable["kind"] == "leftover").any():
                print_warning("a conversion stopped and left files. 'artistools cache delete' deletes them")
        case "delete":
            delete_caches(modelfolder, args.force)
