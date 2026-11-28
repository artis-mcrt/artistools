"""Make, list, or delete the parquet caches that artistools reads in place of the text output of an ARTIS model.

A reader of artistools converts a text file to a parquet cache when it first reads the file. Later commands read the
cache. "make" does all the conversions at one time, e.g. in the job script after a run of ARTIS. Each later command
then reads its data quickly. "list" gives the caches and their sizes. "delete" deletes the caches and the leftover
files, which are the temporary files and folders of a conversion that stopped. A cache whose text files are absent is
the only copy of its data, and "delete" keeps it unless the user gives --include-only-copies.
"""

import argparse
import os
import re
import shutil
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.atomic import scan_linelist
from artistools.estimators import get_estimator_cache_missing_ranks
from artistools.estimators import scan_estimators
from artistools.inputmodel import find_abundance_file
from artistools.inputmodel import get_model_file_path
from artistools.inputmodel import scan_initelemabundances
from artistools.inputmodel import scan_modeldata
from artistools.misc import addarg_action
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_table_output
from artistools.misc import check_local_path
from artistools.misc import confirm_deletion
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import format_range_list
from artistools.misc import get_model_folder
from artistools.misc import get_runfolders
from artistools.misc import parquet_is_readable
from artistools.misc import parse_cli_args
from artistools.misc import path_is_dir
from artistools.misc import print_detail
from artistools.misc import print_or_write_table
from artistools.misc import print_warning
from artistools.misc import require_action
from artistools.misc import resolve_table_outputfile
from artistools.misc import SN3D_INPUT_FOLDERS
from artistools.nltepops import get_nltepops_cache_missing_ranks
from artistools.nltepops import NLTE_CACHE_FILENAME
from artistools.packets import get_packets_batch_parquet_paths
from artistools.packets import get_packets_cache_missing_ranks

if t.TYPE_CHECKING:
    from collections.abc import Callable

# the names of the caches that the readers write: the input model, the line list, the estimators, and the batches of
# the packets and the virtual packets. A file of a different name, e.g. science.out.parquet, belongs to the user
CACHE_BASENAME: t.Final = (
    r"(?:model\.txt|abundances\.txt|linelist\.out|estimators_allranks\.out|nltepops_allranks\.out|estimbatch\d+_\d+_\d+\.out"
    r"|v?packetsbatch\d+_\d+_\d+\.out)\.parquet"
)

# a cache of an artistools version before 2026-10 also has the suffix .tmp
CACHE_NAME_PATTERN: t.Final = re.compile(rf"{CACHE_BASENAME}(?:\.tmp)?")

# write_parquet_atomic writes a cache to a temporary file of the name of mkstemp, and a conversion of estimators
# writes its parts into a temporary folder. A conversion that stops, e.g. at the time limit of a job, can leave them
LEFTOVER_NAME_PATTERN: t.Final = re.compile(rf"\.{CACHE_BASENAME}\.(?:partial\w+\.partial|parts\w+)")


def get_cache_folders(modelfolder: Path) -> list[Path]:
    """Return the model folder and its subfolders, e.g. packets and the run folders, where the readers put caches.

    The reader of abundances.txt takes the file that sn3d reads, which can be in artis/data/. The reader puts the
    cache beside that file.
    """
    subfolders = [child for child in modelfolder.iterdir() if child.is_dir()]
    sn3dfolders = [modelfolder / folder for folder in SN3D_INPUT_FOLDERS[1:] if (modelfolder / folder).is_dir()]
    return [modelfolder, *sorted(dict.fromkeys([*subfolders, *sn3dfolders]))]


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
    # the readers also open a model.txt in data/, and an abundances.txt that sn3d reads, e.g. in artis/data/
    modelfile = get_model_file_path(modelfolder)
    if modelfile is None:
        return None
    scan_modeldata(modelfolder, printwarningsonly=True)
    if find_abundance_file(modelfile.parent) is not None:
        scan_initelemabundances(modelfolder, printwarningsonly=True)
    return "the reader keeps a cache of each input file larger than 2 MiB"


def make_linelist_cache(modelfolder: Path) -> str | None:
    """Read the line list. The reader then writes its cache. Return None for a model with no line list."""
    if firstexisting_or_none(["linelist.out", "linestat.out"], folder=modelfolder, search_subfolders=False) is None:
        return None
    scan_linelist(modelfolder)
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
        except (
            OSError,
            EOFError,
            ValueError,
            AssertionError,
            pl.exceptions.PolarsError,
            pl.exceptions.PanicException,
        ) as exc:
            rows.append((kind, "error", f"{type(exc).__name__}: {exc}"))
            continue
        rows.append((kind, "no data", nodatadetail) if detail is None else (kind, "ok", detail))
    return pl.DataFrame(rows, schema={"data": pl.String, "status": pl.String, "detail": pl.String}, orient="row")


def get_missing_text_ranks(
    modelfolder: Path, cachepath: Path, folderlistings: dict[Path, dict[str, os.DirEntry[str]]]
) -> list[int]:
    """Return each rank whose text file is absent, thus no conversion can make the cache again.

    The readers of the packets, the estimators, and the NLTE populations read their caches with no text files, e.g. in
    an archived model folder. The readers of the other caches need the text file, thus such a cache gives no rank.
    """
    if cachepath.name.startswith(("packetsbatch", "vpacketsbatch")):
        return get_packets_cache_missing_ranks(modelfolder, cachepath, folderlistings)
    if cachepath.name.startswith(("estimbatch", "estimators_allranks")):
        return get_estimator_cache_missing_ranks(modelfolder, cachepath)
    if cachepath.name == NLTE_CACHE_FILENAME:
        return get_nltepops_cache_missing_ranks(modelfolder, cachepath.parent)
    return []


def get_only_copies(modelfolder: Path, caches: Sequence[Path]) -> dict[Path, str]:
    """Return each cache that is the only copy of its data, with the text files that are absent.

    A cache that polars cannot read holds no data. A cache whose text files artistools cannot find, e.g. because
    input.txt is absent, is also an only copy, because the deletion must not lose data.
    """
    folderlistings: dict[Path, dict[str, os.DirEntry[str]]] = {}
    onlycopies: dict[Path, str] = {}
    for cachepath in caches:
        if not parquet_is_readable(cachepath):
            continue
        try:
            missingranks = get_missing_text_ranks(modelfolder, cachepath, folderlistings)
        except (OSError, ValueError) as exc:
            onlycopies[cachepath] = f"artistools cannot find its text files ({type(exc).__name__}: {exc})"
            continue
        if missingranks:
            ranktext = (
                f"rank {missingranks[0]}" if len(missingranks) == 1 else f"ranks {format_range_list(missingranks)}"
            )
            onlycopies[cachepath] = f"the text files of {ranktext} are absent"
    return onlycopies


def delete_caches(modelfolder: Path, force: bool, include_only_copies: bool) -> None:
    """Delete the caches and the leftover files of the model after the user agrees.

    A cache that is the only copy of its data stays, unless include_only_copies is True, as git clean keeps an ignored
    file without -x. force skips only the question, as rm -f does.
    """
    caches, leftovers = find_cache_files(modelfolder)
    onlycopies = get_only_copies(modelfolder, caches)
    for cachepath, reason in onlycopies.items():
        print_warning(f"{cachepath} is the only copy of its data, because {reason}")
    if onlycopies and not include_only_copies:
        print_warning(
            "artistools does not delete an only copy. Give --include-only-copies to delete the only copies too. Their"
            " data is then lost"
        )
    paths = [*(path for path in caches if include_only_copies or path not in onlycopies), *leftovers]
    if not paths:
        print(f"{modelfolder} holds no other cache" if onlycopies else f"{modelfolder} holds no cache")
        return
    confirm_deletion(paths, force)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    print(f"Deleted {len(paths)} files. The next command that reads each kind of data converts the text files again")
    if include_only_copies and onlycopies:
        print_warning(f"{len(onlycopies)} of the deleted files were the only copies of their data")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_action(
        parser,
        choices=["make", "list", "delete"],
        helptext=(
            "make: convert the text output to the parquet caches. list: give the caches and their sizes. delete:"
            " delete the caches and the leftover files of the conversions that stopped. delete keeps each cache that"
            " is the only copy of its data"
        ),
    )
    addarg_modelpath(parser, positional=True, default=Path())
    parser.add_argument(
        "--include-only-copies",
        action="store_true",
        help=(
            "With delete: also delete each cache that is the only copy of its data, because its text files are absent."
            " The data is then lost"
        ),
    )
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
            print_detail(f"Total: {dftable['size_mb'].sum():.1f} MB in {dftable.height} files")
            if (dftable["kind"] == "leftover").any():
                print_warning("A conversion stopped and left files. Run `artistools cache delete` to delete them")
        case "delete":
            delete_caches(modelfolder, args.force, args.include_only_copies)
