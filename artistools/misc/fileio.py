"""File helpers: compressed text files, file searching, metadata, and atomic parquet writes."""

import io
import sys
import typing as t
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import polars as pl


def zopen(filename: Path | str, mode: str = "rt", encoding: str | None = None) -> t.Any:
    """Open filename, filename.zst, filename.gz or filename.xz."""
    if sys.version_info >= (3, 14):
        # only available in Python 3.14+
        from compression import gzip
        from compression import lzma
        from compression import zstd

    else:
        import gzip
        import lzma

        import zstandard as zstd  # pyright: ignore[reportMissingImports]

    ext_fopen: dict[str, t.Any] = {".zst": zstd.open, ".gz": gzip.open, ".xz": lzma.open}

    for ext, fopen in ext_fopen.items():
        file_withext = str(filename) if str(filename).endswith(ext) else str(filename) + ext
        if Path(file_withext).exists():
            return fopen(file_withext, mode=mode, encoding=encoding)

    # open() can raise file not found if this file doesn't exist
    return Path(filename).open(mode=mode, encoding=encoding)


def zopenpl(filename: Path | str, mode: str = "r", encoding: str | None = None) -> t.Any | Path:
    """Open filename, filename.zst, filename.gz or filename.xz. If polars.read_csv can read the file directly, return a Path object instead of a file object."""
    if sys.version_info >= (3, 14):
        from compression import lzma
    else:
        import lzma

    ext_fopen: dict[str, t.Any | None] = {".zst": None, ".gz": None, ".xz": lzma.open}

    for ext, fopen in ext_fopen.items():
        file_withext = str(filename) if str(filename).endswith(ext) else str(filename) + ext
        if Path(file_withext).exists():
            return Path(file_withext) if fopen is None else fopen(file_withext, mode=mode, encoding=encoding)

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
                for ext in (".zst", ".gz", ".xz"):
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
    filelist: Sequence[str | Path], folder: Path | str = ".", tryzipped: bool = True, search_subfolders: bool = True
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
    """Take a file path (e.g. packets00_0000.out.gz) and return the Path with no suffixes (e.g. packets)."""
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

    filepath = Path(str(filepath).replace(".xz", "").replace(".gz", "").replace(".zst", ""))

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
    """Merge a list of PDF files into a single PDF file."""
    from pypdf import PdfWriter

    merger = PdfWriter()

    for pdfpath in pdf_files:
        with Path(pdfpath).open("rb") as pdffile:
            merger.append(pdffile)
        Path(pdfpath).unlink()

    resultfilename = f"{pdf_files[0].replace('.pdf', '')}-{pdf_files[-1].replace('.pdf', '')}"
    with Path(f"{resultfilename}.pdf").open("wb") as resultfile:
        merger.write(resultfile)

    print(f"Files merged and saved to {resultfilename}.pdf")


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
