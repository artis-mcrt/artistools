"""Make, list, or delete the parquet caches that artistools reads in place of the text output of an ARTIS model.

A reader of artistools converts a text file to a parquet cache at the first read, and later reads take the cache.
"make" does all the conversions at one time, e.g. in the job script after a run of ARTIS. Each later command then
reads its data quickly. "list" gives the caches and their sizes. "delete" removes the caches and the files that a
stopped conversion leaves.
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
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import confirm_deletion
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import get_model_folder
from artistools.misc import get_runfolders
from artistools.misc import parse_cli_args
from artistools.misc import path_is_dir
from artistools.misc import print_detail
from artistools.misc import print_product
from artistools.misc import print_warning
from artistools.misc import require_action
from artistools.packets import has_packets_files
from artistools.packets.core import get_packets_batch_parquet_paths

if t.TYPE_CHECKING:
    from collections.abc import Callable

# each reader names its cache after the text file and adds .parquet. A cache of an artistools version before 2026-10
# also has the suffix .tmp. A file of another name, e.g. the output of --write_data, is not a cache
CACHE_NAME_PATTERN: t.Final = re.compile(r"[^.].*\.(?:txt|out)\.parquet(?:\.tmp)?")

# write_parquet_atomic writes a cache to a temporary file of this name, and a conversion of estimators writes its
# parts into a temporary folder. A conversion that stops, e.g. at the time limit of a job, can leave them
LEFTOVER_NAME_PATTERN: t.Final = re.compile(r"\..*\.(?:partial|parts).*")


def get_cache_folders(modelfolder: Path) -> list[Path]:
    """Return the model folder and its subfolders, e.g. packets and the run folders, where the readers put caches."""
    return [modelfolder, *sorted(child for child in modelfolder.iterdir() if child.is_dir())]


def find_cache_files(modelfolder: Path) -> tuple[list[Path], list[Path]]:
    """Return the parquet caches of a model, and the temporary files and folders of conversions that stopped.

    The lock files of the caches stay out of both lists. A process that waits for a lock keeps the lock file open,
    thus a deletion can give the lock to two writers at one time.
    """
    caches: list[Path] = []
    leftovers: list[Path] = []
    for folder in get_cache_folders(modelfolder):
        for path in sorted(folder.iterdir()):
            if CACHE_NAME_PATTERN.fullmatch(path.name) and path.is_file():
                caches.append(path)
            elif LEFTOVER_NAME_PATTERN.fullmatch(path.name) and not path.name.endswith(".replace-lock"):
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
    """Read model.txt and abundances.txt, thus the reader writes the cache of each large file.

    Return None for a model folder with no model.txt.
    """
    if firstexisting_or_none("model.txt", folder=modelfolder, search_subfolders=False) is None:
        return None
    get_modeldata(modelfolder, printwarningsonly=True)
    if firstexisting_or_none("abundances.txt", folder=modelfolder, search_subfolders=False) is not None:
        get_initelemabundances(modelfolder, printwarningsonly=True)
    return "the reader keeps a cache of each input file larger than 2 MiB"


def make_linelist_cache(modelfolder: Path) -> str | None:
    """Read the line list, thus the reader writes its cache. Return None for a model with no line list."""
    if firstexisting_or_none(["linelist.out", "linestat.out"], folder=modelfolder, search_subfolders=False) is None:
        return None
    get_linelist_pldf(modelfolder)
    return "linelist.out.parquet"


def make_estimator_caches(modelfolder: Path) -> str | None:
    """Scan the estimators, thus the reader converts the files of each run folder. Return None with no estimators."""
    if not get_runfolders(modelfolder):
        return None
    scan_estimators(modelfolder)
    return "one cache for each batch of ranks in each run folder"


def make_packet_caches(modelfolder: Path, virtual: bool) -> str | None:
    """Convert the packet files of each batch of ranks. Return None for a model with no such packet files."""
    if virtual:
        # ARTIS writes virtual packets only when vpkt.txt exists
        if not (modelfolder / "vpkt.txt").is_file():
            return None
    elif not has_packets_files(modelfolder):
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
            "the model folder holds no vpkt.txt",
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
    print(f"Deleted {len(paths)} files. The next read of each kind of data converts the text files again")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_action(
        parser,
        choices=["make", "list", "delete"],
        helptext=(
            "make: convert the text output to the parquet caches. list: give the caches and their sizes. delete:"
            " delete the caches and the files of conversions that stopped"
        ),
    )
    addarg_modelpath(parser, positional=True, default=Path())
    addarg_force(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Make, list, or delete the parquet caches that artistools reads in place of the text output of a model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    require_action(args)
    modelfolder = get_model_folder(args.modelpath)
    if not path_is_dir(modelfolder):
        exit_with_error(f"The model folder {modelfolder} does not exist", "Give a model folder")

    match args.action:
        case "make":
            dfresults = make_caches(modelfolder)
            print_product(args, dfresults)
            if (dfresults["status"] == "error").any():
                exit_with_error("artistools could not make all the caches", "The table above gives each error")
        case "list":
            dftable = get_cache_table(modelfolder)
            print_product(args, dftable)
            print_detail(f"total: {dftable['size_mb'].sum():.1f} MB in {dftable.height} files")
            if (dftable["kind"] == "leftover").any():
                print_warning("a conversion stopped and left files. 'artistools cache delete' deletes them")
        case "delete":
            delete_caches(modelfolder, args.force)
