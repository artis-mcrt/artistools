"""File helpers: compressed text files, file searching, metadata, and atomic parquet writes."""

import contextlib
import io
import os
import re
import shlex
import sys
import typing as t
from collections.abc import Callable
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


def get_open_command() -> str:
    """Return the command that opens a file with its default application on this platform."""
    return "open" if sys.platform == "darwin" else "xdg-open"


def print_saved(filepath: Path | str) -> None:
    """Report a saved output file as an 'open <relativepath>' command that can be run on macOS to view the file."""
    from rich.console import Console
    from rich.text import Text

    fullpath = Path(filepath).resolve()
    filepath = fullpath
    with contextlib.suppress(ValueError):
        relativepath = fullpath.relative_to(Path.cwd(), walk_up=True)
        # a file outside the working folder gives a chain of "..", which is longer than the full path
        if len(str(relativepath)) < len(str(fullpath)):
            filepath = relativepath

    # the verb of the platform, thus the line runs as a command there. A terminal that shows links also
    # makes the path one, and a pipe gets the plain text
    opencommand = get_open_command()
    line = Text(f"{opencommand} ") + Text(shlex.quote(str(filepath)), style=f"link {fullpath.as_uri()}")
    Console(highlight=False, soft_wrap=True).print(line)


def find_compressed(filename: Path | str) -> tuple[str, Path] | None:
    """Return the extension and path of filename.zst, filename.gz or filename.xz, or None if no compressed file exists."""
    for ext in COMPRESSED_EXTENSIONS:
        path_withext = Path(str(filename) if str(filename).endswith(ext) else str(filename) + ext)
        if path_withext.exists():
            return ext, path_withext

    return None


def get_decompress_open(ext: str) -> Callable[..., t.IO[t.Any]]:
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


def zopen(filename: Path | str, mode: str = "rt", encoding: str | None = None, errors: str | None = None) -> t.IO[str]:
    """Open filename, falling back to filename.zst, filename.gz or filename.xz.

    The named file wins, thus a stale compressed copy never shadows a file that a run has just written.
    This is the same precedence that read_wsv and firstexisting use. The errors argument takes the value
    that the open functions take, e.g. "replace" for a file that holds a bad byte.
    """
    filepath = Path(filename)
    if not filepath.is_file():
        found = find_compressed(filename)
        if found is None:
            # let open() raise the FileNotFoundError naming the file the caller actually asked for
            return filepath.open(mode=mode, encoding=encoding, errors=errors)
        ext, foundpath = found
        return get_decompress_open(ext)(foundpath, mode=mode, encoding=encoding, errors=errors)

    if filepath.suffix in COMPRESSED_EXTENSIONS:
        return get_decompress_open(filepath.suffix)(filepath, mode=mode, encoding=encoding, errors=errors)

    return filepath.open(mode=mode, encoding=encoding, errors=errors)


def polars_source(filename: Path | str, mode: str = "r", encoding: str | None = None) -> t.IO[bytes] | Path:
    """Return the path of a file that polars reads itself, or a file object that decompresses it.

    polars reads a plain file, a zstd file, and a gzip file from the path. It cannot read an xz file.
    The caller gives the name of a file that exists. Use zopenpl to also find a compressed sibling.
    """
    filepath = Path(filename)
    if filepath.suffix not in COMPRESSED_EXTENSIONS or filepath.suffix in POLARS_READABLE_EXTENSIONS:
        return filepath

    # the default mode "r" opens a binary stream in all three backends, which is what polars reads
    return get_decompress_open(filepath.suffix)(filepath, mode=mode, encoding=encoding)


def zopenpl(filename: Path | str, mode: str = "r", encoding: str | None = None) -> t.IO[bytes] | Path:
    """Return a polars source for filename, or for a compressed sibling when the named file does not exist.

    The named file wins, for the same reason as in zopen. If polars can read the file directly, this
    returns a Path rather than a file object.
    """
    filepath = Path(filename)
    if not filepath.is_file() and (found := find_compressed(filename)):
        return polars_source(found[1], mode=mode, encoding=encoding)

    return polars_source(filepath, mode=mode, encoding=encoding)


@contextlib.contextmanager
def polars_error_note(filepath: Path) -> Generator[None]:
    """Name the file in a polars error, because the parser sees a path or in-memory bytes only."""
    try:
        yield
    except pl.exceptions.PolarsError as exc:
        exc.add_note(f"while reading {filepath}")
        raise


def scan_lines(filepath: Path, skip_rows: int = 0, encoding: t.Literal["utf8", "utf8-lossy"] = "utf8") -> pl.LazyFrame:
    """Return a lazy frame with one string column named "line" that holds each line of a text file.

    The function drops the first skip_rows lines. polars_source selects the compression format. The caller
    gives a path that it has resolved, thus this function must not search for a compressed sibling.
    """
    with contextlib.ExitStack() as stack:
        source = polars_source(filepath, mode="rb")
        if not isinstance(source, Path):
            # polars reads a file object when it makes the plan, thus the file can close after that
            stack.enter_context(source)

        return pl.scan_csv(
            source,
            # the ASCII unit separator, which an ARTIS text file never holds, so that each line stays
            # one field. artisatomic reads a line in the same way, with the same separator
            separator="\x1f",
            has_header=False,
            quote_char=None,
            encoding=encoding,
            new_columns=["line"],
            # a line is text, thus polars must not spend a pass on the type of the column
            infer_schema_length=0,
            skip_rows=skip_rows,
        )


def normalised_lines(
    filepath: Path,
    skip_rows: int = 0,
    comment_prefix: str | None = None,
    encoding: t.Literal["utf8", "utf8-lossy"] = "utf8",
) -> pl.LazyFrame:
    """Return the lines of a text file with each run of whitespace collapsed to a single space.

    The function drops the first skip_rows lines, removes the text from comment_prefix to the end of a line,
    and drops a blank line.
    """
    line = pl.col("line")
    if comment_prefix:
        # a comment can start anywhere on a line, thus keep the text before the first prefix only. split
        # takes the prefix as text, thus no escape of a regular expression character is necessary
        line = line.str.split(comment_prefix).list.first()

    return (
        scan_lines(filepath, skip_rows=skip_rows, encoding=encoding)
        # the class holds each ASCII whitespace character that a line can contain. The Unicode class \s
        # would also split a field that holds a no-break space, which an ARTIS column must keep
        .select(line.str.replace_all(r"[ \t\r\x0b\x0c]+", " ").str.strip_chars(" "))
        .filter(pl.col("line").str.len_bytes() > 0)
    )


def bytes_outside_comments_are_utf8(filepath: Path, skip_rows: int = 0, comment_prefix: str | None = None) -> bool:
    """Return True if each byte that no comment holds is valid UTF-8.

    The function reads the bytes of the file and answers at the byte level, as the reader of an earlier
    version did. A decoder cannot answer it, because a decoder replaces every bad byte, and a file can
    also hold the replacement character as data. This costs a read of the whole file, thus call it only
    after a strict read has failed.
    """
    if filepath.suffix in COMPRESSED_EXTENSIONS:
        with get_decompress_open(filepath.suffix)(filepath, mode="rb") as fin:
            data: bytes = fin.read()
    else:
        data = filepath.read_bytes()

    start = 0
    for _ in range(skip_rows):
        endofline = data.find(b"\n", start)
        if endofline < 0:
            # the file holds no line after the skipped ones
            return True

        start = endofline + 1

    data = data[start:]
    if comment_prefix:
        data = re.sub(re.escape(comment_prefix.encode()) + rb"[^\n]*", b"", data)

    try:
        data.decode()
    except UnicodeDecodeError:
        return False

    return True


def normalise_whitespace(filepath: Path, skip_rows: int = 0, comment_prefix: str | None = None) -> io.BytesIO:
    """Return a buffer of the normalised lines of a text file, which a CSV parser can read.

    The separator of the buffer is a single space, and the buffer is at position zero. A byte that is not
    valid UTF-8 raises an error, unless the comment step removes the text that holds it.
    """

    def sink(lflines: pl.LazyFrame) -> io.BytesIO:
        normalised = io.BytesIO()
        lflines.sink_csv(normalised, include_header=False, quote_style="never")
        # sink_csv leaves the buffer at the end, thus rewind it for the reader
        normalised.seek(0)

        return normalised

    try:
        return sink(normalised_lines(filepath, skip_rows, comment_prefix))
    except pl.exceptions.ComputeError:
        # polars decodes a line before the comment step removes it, thus a comment that holds a byte
        # which is not valid UTF-8 stops the read. Such a comment is common in a file from a different
        # source, e.g. a degree sign in Latin-1. Give the caller the first error if a column holds such
        # a byte, because the values of that column would be text that no reader can trust
        if not bytes_outside_comments_are_utf8(filepath, skip_rows, comment_prefix):
            raise

        # only a comment holds a bad byte, thus read the file again and replace each one
        return sink(normalised_lines(filepath, skip_rows, comment_prefix, encoding="utf8-lossy"))


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
    """Read a whitespace-separated text file into a DataFrame, where any run of whitespace is one separator.

    Use this instead of pl.read_csv(separator=" ") for a file whose columns are aligned with variable amounts
    of whitespace. A run of spaces, tabs, or carriage returns between two fields acts as a single separator.
    The function removes the text from comment_prefix to the end of a line, and it drops a blank line. The
    file can be compressed (.zst/.gz/.xz). The function reads a compressed sibling of the given name only
    when the named file does not exist. A column that starts with integers and later holds a float becomes a
    float column, at the cost of a second parse with a full-file schema inference. A not-a-number token, e.g.
    "nan" or "NA", becomes null, and it does not make the column a string column.

    When columns is given, the function parses and returns only those columns, in the given order, which
    saves memory on a wide file. When header_from_comment is set and the first line is a comment, its words
    (after comment_prefix) become the column names. This covers a file whose header line is a comment.
    """
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
        # the comment holds the column names, and it can hold a byte that is not valid UTF-8, thus
        # replace such a byte here as the read of the lines below also does
        with zopen(filepath, errors="replace") as fin:
            first_line = fin.readline()
        if first_line.lstrip().startswith(comment_prefix):
            new_columns = first_line.lstrip().removeprefix(comment_prefix).split()
            has_header = False

    with polars_error_note(filepath):
        normalised = normalise_whitespace(filepath, skip_rows=skip_rows, comment_prefix=comment_prefix)

    # polars projects by ascending column index and names the projected columns positionally, so
    # translate a name-based projection of a headerless read into that form
    projected_indices: list[int] | None = None
    projected_new_columns = list(new_columns) if new_columns is not None else None
    if columns is not None and new_columns is not None:
        projected_indices = sorted(new_columns.index(col) for col in columns)
        projected_new_columns = [new_columns[i] for i in projected_indices]

    def parse(infer_schema_length: int | None) -> pl.DataFrame:
        # this function runs again when the first schema turns out to be wrong, thus rewind the buffer
        normalised.seek(0)

        return pl.read_csv(
            normalised,
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

        if not sample_missed_numeric_column(dfout):
            return dfout

        # a frame of a wide file is large, thus drop this one before the next parse makes another
        del dfout

        return parse(infer_schema_length=None)

    with polars_error_note(filepath):
        dfout = parse_with_inference_fallback()

    # restore the caller's requested column order, which index-based projection may have changed
    return dfout.select(list(columns)) if columns is not None else dfout


def firstexisting(
    filelist: Sequence[str | Path] | str | Path,
    folder: Path | str = ".",
    tryzipped: bool = True,
    search_subfolders: bool = True,
    purpose: str = "",
) -> Path:
    """Return the first existing file in file list. If none exist, raise exception.

    A caller gives purpose to say what the file holds and which commands read it, because a list of
    names alone does not tell a user what to do next.
    """
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

    # an absolute path is not below the folder, thus the message shows the full path for it
    strfilelist = "\n  ".join([str(x.relative_to(folder)) if x.is_relative_to(folder) else str(x) for x in fullpaths])
    orsub = " or subfolders" if search_subfolders else ""
    msg = f"None of these files exist in {folder}{orsub}: \n  {strfilelist}"
    if purpose:
        msg += f"\n{purpose}"

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


def path_is_artis_model(filepath: Path | str) -> bool:
    """Return whether the path is an ARTIS model and not a reference data file.

    An ARTIS model is a folder, or an output file of ARTIS that is possibly compressed.
    """
    filepath = Path(filepath)

    return filepath.is_dir() or filepath.name.endswith((".out", *(f".out{ext}" for ext in COMPRESSED_EXTENSIONS)))


def path_is_codecomparison(filepath: Path | str) -> bool:
    """Return whether the path is a virtual codecomparison path and not a real folder on disk.

    A codecomparison path has the form "codecomparison/<model>/<code>". It names a data set of the
    radiative transfer code comparison workshop, thus no such folder exists.
    """
    filepath = Path(filepath)

    return not filepath.exists() and filepath.parts[0] == "codecomparison"


def readnoncommentline(file: t.IO[str]) -> str:
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


def open_file(filepath: Path | str) -> None:
    """Open a file in the application that the platform gives it."""
    import subprocess  # ruff:ignore[suspicious-subprocess-import]

    # the command is our own platform opener and a path that the caller has written
    subprocess.run([get_open_command(), str(filepath)], check=False)  # ruff:ignore[subprocess-without-shell-equals-true]


def merge_pdf_files(pdf_files: list[str]) -> str:
    """Merge a list of PDF files into one, and return its path. The inputs go once the merged file exists."""
    from artistools.misc.general import import_optional

    PdfWriter = import_optional("pypdf").PdfWriter

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

    return resultfilename


def write_gif(giffile: Path | str, imagefiles: Sequence[Path | str], duration: float) -> None:
    """Combine image files into an animated gif, showing each frame for duration milliseconds."""
    from artistools.misc.general import import_optional

    iio = import_optional("imageio.v2")

    # bind the writer outside the with, because __enter__ is typed as returning the reader/writer base class
    writer = iio.get_writer(giffile, mode="I", duration=duration)
    with writer:
        for imagefile in imagefiles:
            writer.append_data(iio.imread(imagefile))

    print(f"Created gif: {giffile}")


def get_file_identity(file: Path | os.stat_result) -> tuple[int, int] | None:
    """Return the device and inode numbers of a file, or None if it does not exist.

    Two stats of one path give the same pair only if it is still the same file. A rename onto the path
    gives a different pair, which no comparison of names or modification times can detect reliably. A
    caller that already holds a stat result passes it, so the identity describes that same snapshot.
    """
    if isinstance(file, os.stat_result):
        return (file.st_dev, file.st_ino)

    try:
        filestat = file.stat()
    except FileNotFoundError:
        return None

    return (filestat.st_dev, filestat.st_ino)


def replace_outdated_file(newfilepath: Path, destpath: Path, outdatedfile: tuple[int, int] | None) -> None:
    """Install newfilepath at destpath, unless a file other than the given out-of-date one is there.

    An empty destination takes the new file, the file whose identity is outdatedfile is replaced, and any
    other file is kept: it is a rival's fresh replacement, built from the same inputs. The identity check
    and the rename happen under an exclusive flock. Without it, two writers that both found the same
    out-of-date file could both pass the check, and the second rename would replace the first writer's
    fresh file while a reader scans it. A writer that finds the lock taken waits for the holder, and the
    operating system releases the lock of a holder that dies, so the lock needs no age heuristic that
    could steal it from a paused but live writer.

    The lock file stays in place once created. Removing it while a rival waits on it would hand out a
    second lock on a new inode, and two writers would hold the lock at once. The dot prefix keeps it out
    of the globs that find the parquet files, like the .partial file.
    """
    try:
        import fcntl
    except ImportError:
        # a platform without flock gets the unlocked check. Two simultaneous replacements of one
        # out-of-date file can then race, which the identity check narrows but cannot close
        if get_file_identity(destpath) in {None, outdatedfile}:
            newfilepath.replace(destpath)
        return

    lockpath = destpath.with_name(f".{destpath.name}.replace-lock")
    # flock locks a read-only descriptor, so a different user regenerating a cache in a shared model
    # directory needs only read access to the lock file. The chmod grants that under a restrictive umask,
    # and fails harmlessly for a user who does not own the lock
    lockfd = os.open(lockpath, os.O_CREAT | os.O_RDONLY, 0o666)
    with contextlib.suppress(OSError):
        lockpath.chmod(0o666)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX)
        identity = get_file_identity(destpath)
        if identity is None or identity == outdatedfile:
            newfilepath.replace(destpath)
    finally:
        os.close(lockfd)


def write_parquet_atomic(
    pldf: pl.DataFrame | pl.LazyFrame,
    parquetfilepath: Path,
    metadata: dict[str, str] | None = None,
    compression_level: int = 10,
    replaces: tuple[int, int] | None = None,
) -> None:
    """Write a zstd-compressed parquet file through a temporary file, so a partial write is never mistaken for a complete file.

    A parquet file that another process wrote while this one worked is kept, and this copy of the same data
    is discarded. polars opens a parquet file again by its path between reading the metadata and reading the
    row groups, so a rename onto a path that already holds a complete file corrupts every scan in progress:
    the second file gets read with the offsets of the first.

    replaces is the get_file_identity() of the file this write replaces, taken from the same stat snapshot
    that showed the caller its data was out of date. Only that exact file is replaced. The moment the write
    started is too late to take it: a rival that finished its own replacement first would be snapshotted as
    the file to replace, and its fresh file would be renamed over while a reader scans it.

    The temporary name is prefixed with a dot so that it does not start with the destination's name. A glob
    keyed on that name (get_runfolder_timesteps() looks for "estimbatch*.out.parquet*") would otherwise match
    the in-flight temporary and read a file that is empty, half-written, or already renamed away.
    """
    import tempfile

    try:
        deststat: os.stat_result | None = parquetfilepath.stat()
    except FileNotFoundError:
        deststat = None

    fd, partialfilename = tempfile.mkstemp(
        dir=parquetfilepath.parent, prefix=f".{parquetfilepath.name}.partial", suffix=".partial"
    )
    os.close(fd)
    partialfilepath = Path(partialfilename)
    # mkstemp creates the file 0600, and the destination takes the mode of the file that lands on it, so a
    # cache written into a group-shared model directory would be unreadable to everyone but its author.
    # Reuse the mode the destination already has, or else the read/write bits of the directory holding it —
    # reading the process umask would need a get-and-restore that is not safe against other threads.
    destmode = deststat.st_mode if deststat else parquetfilepath.parent.stat().st_mode & 0o666
    partialfilepath.chmod(destmode & 0o777)
    try:
        pldf.lazy().sink_parquet(
            partialfilepath, compression="zstd", compression_level=compression_level, metadata=metadata
        )
        try:
            # gives the file a second name, and fails if that name is taken, thus the destination appears
            # complete in one step and no reader of it ever sees a different file at the same path
            os.link(partialfilepath, parquetfilepath)
        except FileExistsError:
            replace_outdated_file(partialfilepath, parquetfilepath, replaces)
        except OSError:
            # a file system without hard links cannot make the destination appear in one step, but the
            # locked identity rule still applies: install, replace the out-of-date file, or keep a rival's
            replace_outdated_file(partialfilepath, parquetfilepath, replaces)
    finally:
        partialfilepath.unlink(missing_ok=True)
