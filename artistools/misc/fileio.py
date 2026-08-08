"""File helpers: compressed text files, file searching, metadata, and atomic parquet writes."""

import contextlib
import io
import shlex
import sys
import typing as t
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import polars as pl
import polars.selectors as cs

COMPRESSED_EXTENSIONS = (".zst", ".gz", ".xz")

# polars can read these compressed formats directly from a path
POLARS_READABLE_EXTENSIONS = (".zst", ".gz")


@t.overload
def drop_trailing_null_column(df: pl.DataFrame) -> pl.DataFrame: ...


@t.overload
def drop_trailing_null_column(df: pl.LazyFrame) -> pl.LazyFrame: ...


def drop_trailing_null_column(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Drop the all-null last column that a trailing space on every line of an ARTIS text file produces.

    Call this on the frame as it comes back from read_csv/scan_csv, before adding any column of your own: the
    check is positional, so a column appended first would be tested instead and the null column would survive.
    """
    # require at least one row: is_null().all() is vacuously true over an empty column, which would drop a real
    # column from a file that has no data rows (a rank that emitted no packets, say)
    isnullcol = df.select(cs.by_index(-1).is_null().all() & (pl.len() > 0))
    if isinstance(isnullcol, pl.LazyFrame):
        isnullcol = isnullcol.collect()

    return df.drop(cs.by_index(-1)) if isnullcol.item() else df


def print_saved(filepath: Path | str) -> None:
    """Report a saved output file as an 'open <relativepath>' command that can be run on macOS to view the file."""
    filepath = Path(filepath).resolve()
    with contextlib.suppress(ValueError):
        filepath = filepath.relative_to(Path.cwd(), walk_up=True)
    print(f"open {shlex.quote(str(filepath))}")


def find_compressed(filename: Path | str) -> tuple[str, Path] | None:
    """Return the extension and path of filename.zst, filename.gz or filename.xz, or None if no compressed file exists."""
    for ext in COMPRESSED_EXTENSIONS:
        path_withext = Path(str(filename) if str(filename).endswith(ext) else str(filename) + ext)
        if path_withext.exists():
            return ext, path_withext

    return None


def get_decompress_open(ext: str) -> t.Any:
    """Return the open() function of the compression module that handles the given file extension."""
    if sys.version_info >= (3, 14):
        # only available in Python 3.14+
        from compression import gzip
        from compression import lzma
        from compression import zstd

    else:
        import gzip
        import lzma

        import zstandard as zstd

    return {".zst": zstd.open, ".gz": gzip.open, ".xz": lzma.open}[ext]


def zopen(filename: Path | str, mode: str = "rt", encoding: str | None = None) -> t.Any:
    """Open filename, filename.zst, filename.gz or filename.xz."""
    if found := find_compressed(filename):
        ext, filepath = found
        return get_decompress_open(ext)(filepath, mode=mode, encoding=encoding)

    # open() can raise file not found if this file doesn't exist
    return Path(filename).open(mode=mode, encoding=encoding)


def zopenpl(filename: Path | str, mode: str = "r", encoding: str | None = None) -> t.Any | Path:
    """Open filename, filename.zst, filename.gz or filename.xz. If polars.read_csv can read the file directly, return a Path object instead of a file object."""
    if found := find_compressed(filename):
        ext, filepath = found
        if ext in POLARS_READABLE_EXTENSIONS:
            return filepath
        return get_decompress_open(ext)(filepath, mode=mode, encoding=encoding)

    return Path(filename)


def firstexisting(
    filelist: Sequence[str | Path] | str | Path,
    folder: Path | str = ".",
    tryzipped: bool = True,
    search_subfolders: bool = True,
) -> Path:
    """Return the first existing file in file list. If none exist, raise exception."""
    if isinstance(filelist, str | Path):
        filelist = [Path(filelist)]
    else:
        assert isinstance(filelist, Iterable)
        filelist = [Path(x) for x in filelist]

    folder = Path(folder)
    thispath = Path(folder, filelist[0])

    if thispath.exists():
        return thispath

    fullpaths = []

    def search_folders(filelist: list[str | Path] | list[Path]) -> Generator[Path]:
        yield Path(folder)
        if search_subfolders:
            for filename in filelist:
                for p in Path(folder).glob(f"*/{filename}*"):
                    yield p.parent

    for searchfolder in search_folders(filelist):
        for filename in filelist:
            thispath = Path(searchfolder, filename)
            if thispath.exists():
                return thispath

            fullpaths.append(thispath)

            if tryzipped:
                for ext in COMPRESSED_EXTENSIONS:
                    filename_withext = Path(str(filename) if str(filename).endswith(ext) else str(filename) + ext)
                    if filename_withext not in filelist:
                        thispath = Path(searchfolder, filename_withext)
                        if thispath.exists():
                            return thispath
                        fullpaths.append(thispath)

    strfilelist = "\n  ".join([str(x.relative_to(folder)) for x in fullpaths])
    orsub = " or subfolders" if search_subfolders else ""
    msg = f"None of these files exist in {folder}{orsub}: \n  {strfilelist}"
    raise FileNotFoundError(msg)


def firstexisting_or_none(
    filelist: Sequence[str | Path] | str | Path,
    folder: Path | str = ".",
    tryzipped: bool = True,
    search_subfolders: bool = True,
) -> Path | None:
    """Return the first existing file in file list, or None if none exist."""
    try:
        filepath = firstexisting(
            filelist=filelist, folder=folder, tryzipped=tryzipped, search_subfolders=search_subfolders
        )
    except FileNotFoundError:
        return None

    return filepath


def stripallsuffixes(f: Path) -> Path:
    """Take a file path (e.g. packets00_0000.out.gz) and return the Path with no suffixes (e.g. packets00_0000)."""
    f_nosuffixes = Path(f)
    for _ in f.suffixes:
        f_nosuffixes = f_nosuffixes.with_suffix("")  # each call removes only one suffix

    return f_nosuffixes


def readnoncommentline(file: io.TextIOBase) -> str:
    """Read a line from the text file, skipping blank and comment lines that begin with #.

    Raise EOFError if the end of the file is reached before any non-blank, non-comment line is found.
    """
    while line := file.readline():
        if line.strip() and not line.lstrip().startswith("#"):
            return line

    msg = "Reached end of file without finding a non-comment, non-blank line"
    raise EOFError(msg)


@lru_cache(maxsize=24)
def get_file_metadata(filepath: Path | str) -> dict[str, t.Any]:
    """Return a dict of metadata for a file, either from a metadata file or from the big combined metadata file."""
    filepath = Path(filepath)

    def add_derived_metadata(metadata: dict[str, t.Any]) -> dict[str, t.Any]:
        if "a_v" in metadata and "e_bminusv" in metadata and "r_v" not in metadata:
            metadata["r_v"] = metadata["a_v"] / metadata["e_bminusv"]
        elif "e_bminusv" in metadata and "r_v" in metadata and "a_v" not in metadata:
            metadata["a_v"] = metadata["e_bminusv"] * metadata["r_v"]
        elif "a_v" in metadata and "r_v" in metadata and "e_bminusv" not in metadata:
            metadata["e_bminusv"] = metadata["a_v"] / metadata["r_v"]

        return metadata

    import yaml

    if filepath.suffix in COMPRESSED_EXTENSIONS:
        filepath = filepath.with_suffix("")

    # check if the reference file (e.g. spectrum.txt) has an metadata file (spectrum.txt.meta.yml)
    individualmetafile = filepath.with_suffix(f"{filepath.suffix}.meta.yml")
    if individualmetafile.exists():
        with individualmetafile.open("r", encoding="utf-8") as yamlfile:
            metadata = yaml.safe_load(yamlfile)

        return add_derived_metadata(metadata)

    # check if the metadata is in the big combined metadata file (todo: eliminate this file)
    combinedmetafile = Path(filepath.parent.resolve(), "metadata.yml")
    if combinedmetafile.exists():
        with combinedmetafile.open("r", encoding="utf-8") as yamlfile:
            combined_metadata = yaml.safe_load(yamlfile)
        metadata = combined_metadata.get(str(filepath), {})

        return add_derived_metadata(metadata)

    print(f"No metadata found for: {filepath}")

    return {}


def merge_pdf_files(pdf_files: list[str]) -> None:
    """Merge a list of PDF files into a single PDF file, deleting the inputs once the merged file is written."""
    from pypdf import PdfWriter

    merger = PdfWriter()

    for pdfpath in pdf_files:
        with Path(pdfpath).open("rb") as pdffile:
            merger.append(pdffile)

    resultfilename = f"{Path(pdf_files[0]).with_suffix('')}-{Path(pdf_files[-1]).with_suffix('').name}.pdf"
    with Path(resultfilename).open("wb") as resultfile:
        merger.write(resultfile)

    # only remove the inputs once the merged file exists, so a failed write cannot destroy them
    for pdfpath in pdf_files:
        Path(pdfpath).unlink()

    print_saved(resultfilename)


def write_gif(giffile: Path | str, imagefiles: Sequence[Path | str], duration: float) -> None:
    """Combine image files into an animated gif, showing each frame for duration milliseconds."""
    import imageio.v2 as iio

    # bind the writer outside the with, because __enter__ is typed as returning the reader/writer base class
    writer = iio.get_writer(giffile, mode="I", duration=duration)
    with writer:
        for imagefile in imagefiles:
            writer.append_data(iio.imread(imagefile))

    print(f"Created gif: {giffile}")


def write_parquet_atomic(
    pldf: pl.DataFrame | pl.LazyFrame,
    parquetfilepath: Path,
    metadata: dict[str, str] | None = None,
    compression_level: int = 10,
) -> None:
    """Write a zstd-compressed parquet file via a temporary file and an atomic replace, so a partial write is never mistaken for a complete file.

    If a concurrent process wrote the destination first, it is atomically overwritten by this equally-valid replacement.
    """
    import os
    import tempfile

    fd, partialfilename = tempfile.mkstemp(
        dir=parquetfilepath.parent, prefix=f"{parquetfilepath.name}.partial", suffix=".partial"
    )
    os.close(fd)
    partialfilepath = Path(partialfilename)
    try:
        pldf.lazy().sink_parquet(
            partialfilepath, compression="zstd", compression_level=compression_level, metadata=metadata
        )
        partialfilepath.replace(parquetfilepath)
    finally:
        partialfilepath.unlink(missing_ok=True)
