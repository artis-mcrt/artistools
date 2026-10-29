"""Remove each packet except the escaped r-packets from the packets files of a model."""

import argparse
import stat
import typing as t
from pathlib import Path

from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import confirm_overwrite
from artistools.misc import get_file_copies
from artistools.misc import get_model_folder
from artistools.misc import get_nprocs
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import with_compressed_extension
from artistools.misc import without_compressed_extension
from artistools.misc import write_file_atomic
from artistools.misc import write_zstd_lines
from artistools.misc import zopen
from artistools.packets.core import find_packets_textsources
from artistools.packets.core import get_packets_mpirank_groups
from artistools.packets.core import get_packets_rankbatch_parquetpath
from artistools.packets.core import get_packets_text_columns
from artistools.packets.core import get_packets_textfilename
from artistools.packets.core import type_ids

if t.TYPE_CHECKING:
    from collections.abc import Sequence

# the zstd level of the filtered files. A higher level gives a smaller file, but the compression takes more time
ZSTD_LEVEL = 12

# the number of kept lines in one zstd frame. A frame of this size holds approximately 30 MB of text, thus the
# memory of the filter does not increase with the size of a packets file
LINES_PER_FRAME = 100_000

ESCAPED_RPKT_TYPES = (str(type_ids["TYPE_ESCAPE"]), str(type_ids["TYPE_RPKT"]))


class PacketCounts(t.NamedTuple):
    """The counts of the packets and the characters of one packets file, and of the packets that the filter keeps."""

    packets_in: int
    packets_kept: int
    chars_in: int
    chars_kept: int


def format_size(size_bytes: float) -> str:
    """Give a size in bytes as a short string, e.g. '1.2 GB'."""
    for unit in ("B", "kB", "MB", "GB"):
        if abs(size_bytes) < 1000:
            return f"{size_bytes:.0f} {unit}" if unit == "B" else f"{size_bytes:.1f} {unit}"
        size_bytes /= 1000
    return f"{size_bytes:.1f} TB"


def get_packets_files(modelfolder: Path) -> dict[int, Path]:
    """Return the packets file of each rank that the packets reader reads."""
    filenames = {get_packets_textfilename(rank, virtual=False): rank for rank in range(get_nprocs(modelfolder))}
    return {
        filenames[filename]: Path(entry.path)
        for filename, entry in find_packets_textsources(modelfolder, list(filenames)).items()
    }


def filter_packets_file(packetsfile: Path, modelfolder: Path, fout: t.IO[bytes] | None = None) -> PacketCounts:
    """Count the packets and the escaped r-packets of a packets file.

    With fout, the function also writes the comment lines and the lines of the escaped r-packets to fout as zstd.
    """
    columns = get_packets_text_columns(packetsfile, modelfolder)
    type_id_index = columns.index("type_id")
    escape_type_id_index = columns.index("escape_type_id")
    # a packet line has more than 30 columns, thus the split stops after the two columns that the filter reads
    maxsplit = max(type_id_index, escape_type_id_index) + 1
    keptlines: list[str] = []
    packets_in = packets_kept = chars_in = chars_kept = 0
    with zopen(packetsfile, mode="rt", encoding="utf-8") as fpackets:
        for line in fpackets:
            linesplit = line.split(maxsplit=maxsplit)
            if not linesplit or linesplit[0].startswith("#"):
                if fout is not None:
                    keptlines.append(line.removesuffix("\n"))
                continue
            packets_in += 1
            chars_in += len(line)
            if (linesplit[type_id_index], linesplit[escape_type_id_index]) == ESCAPED_RPKT_TYPES:
                packets_kept += 1
                chars_kept += len(line)
                if fout is not None:
                    keptlines.append(line.removesuffix("\n"))
            if fout is not None and len(keptlines) >= LINES_PER_FRAME:
                write_zstd_lines(fout, keptlines, compression_level=ZSTD_LEVEL)
                keptlines.clear()

    if fout is not None and keptlines:
        write_zstd_lines(fout, keptlines, compression_level=ZSTD_LEVEL)

    return PacketCounts(packets_in, packets_kept, chars_in, chars_kept)


def write_filtered_packets_file(packetsfile: Path, modelfolder: Path) -> Path:
    """Replace a packets file with a zstd file of its comment lines and its escaped r-packets.

    The function returns the path of the new file. The new file gets a temporary name first. A stop during the write
    thus leaves the original file.
    """
    outpath = with_compressed_extension(without_compressed_extension(packetsfile), ".zst")
    # the new file takes the mode of the packets file, also when it replaces a file of a different name
    filemode = stat.S_IMODE(packetsfile.stat().st_mode)
    write_file_atomic(outpath, lambda fout: filter_packets_file(packetsfile, modelfolder, fout), mode=filemode)
    # another copy of the rank, e.g. a .gz file, still holds the removed packets
    for filecopy in get_file_copies(outpath):
        if filecopy != outpath:
            filecopy.unlink(missing_ok=True)
    return outpath


def get_rebuildable_packets_caches(modelfolder: Path, packetsfiles: dict[int, Path]) -> dict[int, Path]:
    """Return the path of the parquet cache of the batch of each rank, for the batches that the text files rebuild.

    The readers compare the times of the text files with a tolerance. Thus a cache from a time just before the filter
    can stay in use, and the filter must remove it. A batch that has no text file for a rank is not in the result,
    because only its cache holds the packets of that rank.
    """
    return {
        rank: get_packets_rankbatch_parquetpath(modelfolder, batch_mpiranks, batchindex, virtual=False)
        for batchindex, batch_mpiranks in get_packets_mpirank_groups(modelfolder, maxpacketfiles=None)
        if all(rank in packetsfiles for rank in batch_mpiranks)
        for rank in batch_mpiranks
    }


def remove_packets_cache(parquetcache: Path | None) -> None:
    """Remove the parquet cache of a batch of packets files, if it exists."""
    if parquetcache is not None and parquetcache.is_file():
        parquetcache.unlink(missing_ok=True)
        print_detail(f"Removed the cache {parquetcache}")


class FilteredFileSizes(t.NamedTuple):
    """The packet counts and the sizes of one packets file that the filter changes."""

    packets_in: int
    packets_removed: int
    size_in: int
    size_out: float


def filter_and_report_packets_file(
    packetsfile: Path, modelfolder: Path, dry_run: bool, parquetcache: Path | None
) -> FilteredFileSizes | None:
    """Filter one packets file, or only count its packets with dry_run, and print the result.

    The function returns None for a file that the filter does not change. With dry_run, size_out is an estimate
    from the fraction of the text that the filter keeps. The function removes parquetcache before it replaces the
    file, thus a stop after the replacement leaves no cache of the removed packets. The function removes parquetcache
    again after the replacement, because a reader in another process can make the cache again during the write.
    """
    counts = filter_packets_file(packetsfile, modelfolder)
    if counts.packets_kept == counts.packets_in:
        print(f"{packetsfile}: only escaped r-packets")
        return None
    if counts.packets_kept == 0:
        # the readers of the packets need a packet after the header line
        print(f"{packetsfile}: no escaped r-packets, thus the filter keeps the file")
        return None

    size_in = packetsfile.stat().st_size
    removed = counts.packets_in - counts.packets_kept
    removed_fraction = 1 - counts.chars_kept / counts.chars_in
    print(f"{packetsfile}: the filter removes {removed} of {counts.packets_in} packets")
    print_detail(
        f"The removed packets are {removed_fraction * 100:.1f}% of the text, or about"
        f" {format_size(removed_fraction * size_in)} of {format_size(size_in)}"
    )
    if dry_run:
        return FilteredFileSizes(counts.packets_in, removed, size_in, (1 - removed_fraction) * size_in)

    remove_packets_cache(parquetcache)
    outpath = write_filtered_packets_file(packetsfile, modelfolder)
    remove_packets_cache(parquetcache)
    size_out = outpath.stat().st_size
    print_detail(f"Wrote {outpath}: {format_size(size_out)} ({size_out / size_in * 100:.1f}% of the input)")
    return FilteredFileSizes(counts.packets_in, removed, size_in, size_out)


def print_filter_summary(results: "Sequence[FilteredFileSizes]", nfiles: int, dry_run: bool) -> None:
    """Print the sums of the packet counts and of the sizes of the files that the filter changes."""
    if not results:
        print(f"The filter changes none of the {nfiles} packets files")
        return

    packets_total = sum(result.packets_in for result in results)
    packets_removed = sum(result.packets_removed for result in results)
    removes, changes, about = ("removes", "changes", "about ") if dry_run else ("removed", "changed", "")
    print(
        f"In {len(results)} of {nfiles} packets files, the filter {removes} {packets_removed} of"
        f" {packets_total} packets ({packets_removed / packets_total * 100:.1f}%). The size {changes} from"
        f" {format_size(sum(result.size_in for result in results))} to"
        f" {about}{format_size(sum(result.size_out for result in results))}."
    )


def filter_packets_of_models(modelfolders: "Sequence[Path]", dry_run: bool, force: bool) -> None:
    """Filter the packets files of each model, or only count their packets with dry_run."""
    packetsfiles_of_model = {modelfolder: get_packets_files(modelfolder) for modelfolder in modelfolders}
    packetsfiles = [path for packetsfiles in packetsfiles_of_model.values() for path in packetsfiles.values()]
    if not packetsfiles:
        print(f"No packets files in {', '.join(str(folder) for folder in modelfolders)}")
        return

    if not dry_run:
        # the filter replaces each packets file with a zstd file of the same name
        confirm_overwrite(
            sorted({path.parent for path in packetsfiles}),
            sorted({without_compressed_extension(path).name for path in packetsfiles}),
            force=force,
        )

    results: list[FilteredFileSizes] = []
    for modelfolder, packetsfiles_of_rank in packetsfiles_of_model.items():
        parquetcaches = get_rebuildable_packets_caches(modelfolder, packetsfiles_of_rank)
        for rank, packetsfile in packetsfiles_of_rank.items():
            result = filter_and_report_packets_file(packetsfile, modelfolder, dry_run, parquetcaches.get(rank))
            if result is not None:
                results.append(result)

    print_filter_summary(results, len(packetsfiles), dry_run)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(
        parser, positional=True, multiplepaths=True, helptext="Paths of the ARTIS models (default: the working folder)"
    )
    parser.add_argument("-n", "--dry-run", action="store_true", help="Show what the filter removes, and change no file")
    addarg_force(parser)


def main(args: argparse.Namespace | None = None, argsraw: "Sequence[str] | None" = None, **kwargs: t.Any) -> None:
    """Remove each packet except the escaped r-packets from the packets files of a model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelfolders = [get_model_folder(modelpath) for modelpath in normalize_path_list(args.modelpath)]
    filter_packets_of_models(modelfolders, dry_run=args.dry_run, force=args.force)
