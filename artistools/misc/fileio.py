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


def zopen_unshadowed(filename: Path | str, encoding: str | None = None) -> t.Any:
    """Open filename, falling back to a compressed sibling only when the named file does not exist.

    Unlike zopen, a stale compressed copy never shadows a freshly written uncompressed file. This is the
    same precedence read_wsv uses, and is what a plain Path.open call is replaced by.
    """
    filepath = Path(filename)
    if not filepath.is_file():
        found = find_compressed(filename)
        if found is None:
            # let open() raise the FileNotFoundError naming the file the caller actually asked for
            return filepath.open(encoding=encoding)
        ext, foundpath = found
        return get_decompress_open(ext)(foundpath, mode="rt", encoding=encoding)

    # the named file exists, so open it directly. Going through zopen here would re-run find_compressed
    # and hand back a compressed sibling instead, which is the shadowing this function exists to avoid.
    if filepath.suffix in COMPRESSED_EXTENSIONS:
        return get_decompress_open(filepath.suffix)(filepath, mode="rt", encoding=encoding)

    return filepath.open(encoding=encoding)


def zopenpl(filename: Path | str, mode: str = "r", encoding: str | None = None) -> t.Any | Path:
    """Open filename, filename.zst, filename.gz or filename.xz. If polars.read_csv can read the file directly, return a Path object instead of a file object."""
    if found := find_compressed(filename):
        ext, filepath = found
        if ext in POLARS_READABLE_EXTENSIONS:
            return filepath
        # get_decompress_open() erases the three backends' differing signatures to Any, so the file
        # object is Any by design and the annotation says so rather than leaving it implicit
        fileobj: t.Any = get_decompress_open(ext)(filepath, mode=mode, encoding=encoding)
        return fileobj

    return Path(filename)


def read_wsv(
    filename: Path | str,
    *,
    has_header: bool = True,
    new_columns: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
    comment_prefix: str | None = None,
    header_from_comment: bool = False,
    skip_rows: int = 0,
    schema_overrides: t.Mapping[str, pl.DataType | type[pl.DataType]] | None = None,
) -> pl.DataFrame:
    """Read a whitespace-separated text file into a DataFrame, treating any run of spaces or tabs as one separator.

    Use this instead of pl.read_csv(separator=" ") for files whose columns are aligned with variable amounts of
    whitespace. Text from comment_prefix to the end of a line is removed, blank lines are dropped, and the file may
    be compressed (.zst/.gz/.xz; a compressed sibling of the given name is read only when the named file itself
    does not exist). A column that starts with integers but later contains floats is read as floats, at the cost
    of a second parse with full-file schema inference. Not-a-number tokens such as "nan" and "NA" become nulls
    rather than making the column a string column.

    When columns is given, only those columns are parsed and returned (in the given order), which saves memory
    on wide files. When header_from_comment is set and the first line is a comment, its words (after
    comment_prefix) become the column names, covering files whose header line is written as a comment.
    """
    from artistools import rustext

    filepath = Path(filename)
    if not filepath.is_file():
        # fall back to a compressed sibling only when the named file itself does not exist, so a freshly
        # written uncompressed file is never shadowed by a stale compressed copy
        found = find_compressed(filename)
        if found is not None:
            filepath = found[1]

    if header_from_comment:
        assert comment_prefix is not None
        # filepath was already resolved with unshadowed precedence above, so zopen would undo that by
        # re-running find_compressed and reading the header out of a stale compressed sibling
        with zopen_unshadowed(filepath) as fin:
            first_line = fin.readline()
        if first_line.lstrip().startswith(comment_prefix):
            new_columns = first_line.lstrip().removeprefix(comment_prefix).split()
            has_header = False

    normalized = rustext.read_wsv_normalized(filepath, skip_rows=skip_rows, comment_prefix=comment_prefix)

    # polars projects by ascending column index and names the projected columns positionally, so
    # translate a name-based projection of a headerless read into that form
    projected_indices: list[int] | None = None
    projected_new_columns = list(new_columns) if new_columns is not None else None
    if columns is not None and new_columns is not None:
        projected_indices = sorted(new_columns.index(col) for col in columns)
        projected_new_columns = [new_columns[i] for i in projected_indices]

    def parse(infer_schema_length: int | None) -> pl.DataFrame:
        return pl.read_csv(
            normalized,
            separator=" ",
            has_header=has_header,
            new_columns=projected_new_columns,
            columns=projected_indices if projected_indices is not None else (list(columns) if columns else None),
            schema_overrides=schema_overrides,
            infer_schema_length=infer_schema_length,
            null_values=["nan", "NaN", "-nan", "-NaN", "NA", "N/A", "null", "NULL"],
        )

    def sample_missed_numeric_column(dfout: pl.DataFrame) -> bool:
        """Report a String column whose non-null values all parse as numbers: the sample saw only null tokens."""
        overridden = set(schema_overrides) if schema_overrides is not None else set()
        return any(
            dtype == pl.String
            and name not in overridden
            and not dfout[name].drop_nulls().is_empty()
            and not dfout[name].drop_nulls().cast(pl.Float64, strict=False).has_nulls()
            for name, dtype in dfout.schema.items()
        )

    def parse_with_inference_fallback() -> pl.DataFrame:
        """Parse with sampled schema inference, repeating with full-file inference when the sample misled it."""
        try:
            # inferring the schema from a sample is much faster than scanning the whole file
            dfout = parse(infer_schema_length=10000)
        except pl.exceptions.ComputeError:
            # a column changed type after the sampled rows (e.g. integers followed by floats),
            # so pay for a full-file schema inference pass
            return parse(infer_schema_length=None)

        return parse(infer_schema_length=None) if sample_missed_numeric_column(dfout) else dfout

    try:
        dfout = parse_with_inference_fallback()
    except pl.exceptions.PolarsError as exc:
        # the parser sees only in-memory bytes, so name the file for it
        exc.add_note(f"while reading {filepath}")
        raise

    # restore the caller's requested column order, which index-based projection may have changed
    return dfout.select(list(columns)) if columns is not None else dfout


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

    The temporary name is prefixed with a dot so that it does not start with the destination's name. A glob
    keyed on that name (get_runfolder_timesteps() looks for "estimbatch*.out.parquet*") would otherwise match
    the in-flight temporary and read a file that is empty, half-written, or already renamed away.
    """
    import os
    import tempfile

    fd, partialfilename = tempfile.mkstemp(
        dir=parquetfilepath.parent, prefix=f".{parquetfilepath.name}.partial", suffix=".partial"
    )
    os.close(fd)
    partialfilepath = Path(partialfilename)
    # mkstemp creates the file 0600, and replace() carries that mode onto the destination, so a cache written
    # into a group-shared model directory would be unreadable to everyone but its author. Reuse the mode the
    # destination already has, or else the read/write bits of the directory holding it — reading the process
    # umask would need a get-and-restore that is not safe against other threads.
    destmode = (
        parquetfilepath.stat().st_mode if parquetfilepath.is_file() else parquetfilepath.parent.stat().st_mode & 0o666
    )
    partialfilepath.chmod(destmode & 0o777)
    try:
        pldf.lazy().sink_parquet(
            partialfilepath, compression="zstd", compression_level=compression_level, metadata=metadata
        )
        partialfilepath.replace(parquetfilepath)
    finally:
        partialfilepath.unlink(missing_ok=True)
