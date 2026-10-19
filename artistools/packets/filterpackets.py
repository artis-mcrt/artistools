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
from artistools.misc.fileio import write_zstd_lines
from artistools.packets.core import find_packets_textsources
from artistools.packets.core import get_packets_text_columns
from artistools.packets.core import get_packets_textfilename
from artistools.packets.core import type_ids

if t.TYPE_CHECKING:
    from collections.abc import Sequence

# the zstd level of the filtered files. A higher level gives a smaller file, but the compression takes more time
ZSTD_LEVEL = 12

ESCAPED_RPKT_TYPES = (str(type_ids["TYPE_ESCAPE"]), str(type_ids["TYPE_RPKT"]))


class FilteredPacketsFile(t.NamedTuple):
    """The lines that the filter keeps from one packets file, and the counts of the packets and the characters."""

    keptlines: list[str]
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


def get_packets_files(modelfolder: Path) -> list[Path]:
    """Return the packets file of each rank that the packets reader reads."""
    filenames = [get_packets_textfilename(rank, virtual=False) for rank in range(get_nprocs(modelfolder))]
    return [Path(entry.path) for entry in find_packets_textsources(modelfolder, filenames).values()]


def filter_packets_file(packetsfile: Path, modelfolder: Path) -> FilteredPacketsFile:
    """Keep the comment lines and the lines of the escaped r-packets of a packets file."""
    columns = get_packets_text_columns(packetsfile, modelfolder)
    type_id_index = columns.index("type_id")
    escape_type_id_index = columns.index("escape_type_id")
    # a packet line has more than 30 columns, thus the split stops after the two columns that the filter reads
    maxsplit = max(type_id_index, escape_type_id_index) + 1
    keptlines: list[str] = []
    packets_in = packets_kept = chars_in = chars_kept = 0
    with zopen(packetsfile, mode="rt", encoding="utf-8") as fpackets:
        for line in fpackets:
            if line.startswith("#"):
                keptlines.append(line)
                continue
            packets_in += 1
            chars_in += len(line)
            linesplit = line.split(maxsplit=maxsplit)
            if (linesplit[type_id_index], linesplit[escape_type_id_index]) == ESCAPED_RPKT_TYPES:
                keptlines.append(line)
                packets_kept += 1
                chars_kept += len(line)

    return FilteredPacketsFile(keptlines, packets_in, packets_kept, chars_in, chars_kept)


def write_filtered_packets_file(packetsfile: Path, keptlines: "Sequence[str]") -> Path:
    """Replace a packets file with a zstd file that holds the kept lines, and return the path of the new file.

    The new file gets a temporary name first. A stop during the write thus leaves the original file.
    """
    outpath = with_compressed_extension(without_compressed_extension(packetsfile), ".zst")
    temppath = outpath.with_name(f"{outpath.name}.partial.tmp")
    try:
        with temppath.open("wb") as fout:
            write_zstd_lines(fout, [line.removesuffix("\n") for line in keptlines], compression_level=ZSTD_LEVEL)
        temppath.replace(outpath)
    finally:
        temppath.unlink(missing_ok=True)
    if packetsfile != outpath:
        packetsfile.unlink()
    return outpath


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
    packetsfiles = [(modelfolder, path) for modelfolder in modelfolders for path in get_packets_files(modelfolder)]
    if not packetsfiles:
        print(f"No packets files in {', '.join(str(folder) for folder in modelfolders)}")
        return

    if not args.dry_run:
        # the filter replaces each packets file with a zstd file of the same name
        confirm_overwrite(
            sorted({path.parent for _, path in packetsfiles}),
            sorted({without_compressed_extension(path).name for _, path in packetsfiles}),
            force=args.force,
        )

    files_changed = packets_total = packets_removed = size_in_total = 0
    size_out_total = 0.0
    for modelfolder, packetsfile in packetsfiles:
        result = filter_packets_file(packetsfile, modelfolder)
        if result.packets_kept == result.packets_in:
            print(f"{packetsfile}: only escaped r-packets")
            continue

        size_in = packetsfile.stat().st_size
        removed = result.packets_in - result.packets_kept
        removed_fraction = 1 - result.chars_kept / result.chars_in
        files_changed += 1
        packets_total += result.packets_in
        packets_removed += removed
        size_in_total += size_in
        print(f"{packetsfile}: the filter removes {removed} of {result.packets_in} packets")
        print_detail(
            f"the removed packets are {removed_fraction * 100:.1f}% of the text, or about"
            f" {format_size(removed_fraction * size_in)} of {format_size(size_in)}"
        )
        if args.dry_run:
            size_out_total += (1 - removed_fraction) * size_in
            continue

        outpath = write_filtered_packets_file(packetsfile, result.keptlines)
        size_out = outpath.stat().st_size
        size_out_total += size_out
        print_detail(f"wrote {outpath}: {format_size(size_out)} ({size_out / size_in * 100:.1f}% of the input)")

    if files_changed == 0:
        print(f"Each of the {len(packetsfiles)} packets files holds only escaped r-packets")
        return

    removes, changes, about = ("removes", "changes", "about ") if args.dry_run else ("removed", "changed", "")
    print(
        f"In {files_changed} of {len(packetsfiles)} packets files, the filter {removes} {packets_removed} of"
        f" {packets_total} packets ({packets_removed / packets_total * 100:.1f}%). The size {changes} from"
        f" {format_size(size_in_total)} to {about}{format_size(size_out_total)}."
    )
