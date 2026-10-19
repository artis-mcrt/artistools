"""Remove each packet except the escaped r-packets from the packets files of a model."""

import argparse
import typing as t
from pathlib import Path

from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import confirm_overwrite
from artistools.misc import get_model_folder
from artistools.misc import get_nprocs
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import zopen
from artistools.misc.fileio import with_compressed_extension
from artistools.misc.fileio import without_compressed_extension
from artistools.misc.fileio import write_file_atomic
from artistools.misc.fileio import write_zstd_lines
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
    write_file_atomic(outpath, lambda fout: filter_packets_file(packetsfile, modelfolder, fout))
    if packetsfile != outpath:
        packetsfile.unlink()
    return outpath


def remove_packets_caches(modelfolder: Path, packetsfiles: dict[int, Path], filteredranks: set[int]) -> None:
    """Remove the parquet cache of each batch that holds a filtered rank.

    The readers compare the times of the text files with a tolerance. Thus a cache from a time just before the filter
    can stay in use. The function keeps the cache of a batch that has no text file for a rank, because only the cache
    holds the packets of that rank.
    """
    for batchindex, batch_mpiranks in get_packets_mpirank_groups(modelfolder, maxpacketfiles=None):
        if filteredranks.isdisjoint(batch_mpiranks) or not all(rank in packetsfiles for rank in batch_mpiranks):
            continue
        parquetpath = get_packets_rankbatch_parquetpath(modelfolder, batch_mpiranks, batchindex, virtual=False)
        if parquetpath.is_file():
            parquetpath.unlink()
            print_detail(f"removed the cache {parquetpath}")


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
    packetsfiles_of_model = {modelfolder: get_packets_files(modelfolder) for modelfolder in modelfolders}
    packetsfiles = [path for packetsfiles in packetsfiles_of_model.values() for path in packetsfiles.values()]
    if not packetsfiles:
        print(f"No packets files in {', '.join(str(folder) for folder in modelfolders)}")
        return

    if not args.dry_run:
        # the filter replaces each packets file with a zstd file of the same name
        confirm_overwrite(
            sorted({path.parent for path in packetsfiles}),
            sorted({without_compressed_extension(path).name for path in packetsfiles}),
            force=args.force,
        )

    files_changed = packets_total = packets_removed = size_in_total = 0
    size_out_total = 0.0
    for modelfolder, packetsfiles_of_rank in packetsfiles_of_model.items():
        filteredranks: set[int] = set()
        for rank, packetsfile in packetsfiles_of_rank.items():
            counts = filter_packets_file(packetsfile, modelfolder)
            if counts.packets_kept == counts.packets_in:
                print(f"{packetsfile}: only escaped r-packets")
                continue
            if counts.packets_kept == 0:
                # the readers of the packets need a packet after the header line
                print(f"{packetsfile}: no escaped r-packets, thus the filter keeps the file")
                continue

            size_in = packetsfile.stat().st_size
            removed = counts.packets_in - counts.packets_kept
            removed_fraction = 1 - counts.chars_kept / counts.chars_in
            files_changed += 1
            packets_total += counts.packets_in
            packets_removed += removed
            size_in_total += size_in
            print(f"{packetsfile}: the filter removes {removed} of {counts.packets_in} packets")
            print_detail(
                f"the removed packets are {removed_fraction * 100:.1f}% of the text, or about"
                f" {format_size(removed_fraction * size_in)} of {format_size(size_in)}"
            )
            if args.dry_run:
                size_out_total += (1 - removed_fraction) * size_in
                continue

            outpath = write_filtered_packets_file(packetsfile, modelfolder)
            filteredranks.add(rank)
            size_out = outpath.stat().st_size
            size_out_total += size_out
            print_detail(f"wrote {outpath}: {format_size(size_out)} ({size_out / size_in * 100:.1f}% of the input)")

        if filteredranks:
            remove_packets_caches(modelfolder, packetsfiles_of_rank, filteredranks)

    if files_changed == 0:
        print(f"The filter changes none of the {len(packetsfiles)} packets files")
        return

    removes, changes, about = ("removes", "changes", "about ") if args.dry_run else ("removed", "changed", "")
    print(
        f"In {files_changed} of {len(packetsfiles)} packets files, the filter {removes} {packets_removed} of"
        f" {packets_total} packets ({packets_removed / packets_total * 100:.1f}%). The size {changes} from"
        f" {format_size(size_in_total)} to {about}{format_size(size_out_total)}."
    )
