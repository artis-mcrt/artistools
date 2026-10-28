"""File helpers: compressed text files, file searching, metadata, and atomic parquet writes."""

import contextlib
import io
import math
import os
import re
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from functools import cache
from functools import lru_cache
from pathlib import Path

import polars as pl
import polars.selectors as cs

from artistools.misc.console import print_warning
from artistools.misc.remote import check_local_path
from artistools.misc.remote import on_model_host

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


def with_compressed_extension(filename: Path | str, ext: str) -> Path:
    """Return the path of the compressed file, e.g. spec.out.zst, or the path itself if it has the extension."""
    return Path(str(filename) if str(filename).endswith(ext) else str(filename) + ext)


def without_compressed_extension(filepath: Path) -> Path:
    """Return the path without its compression extension, e.g. model.txt for model.txt.zst."""
    return filepath.with_suffix("") if filepath.suffix in COMPRESSED_EXTENSIONS else filepath


def write_file_atomic(filepath: Path, write: Callable[[t.IO[bytes]], object], mode: int | None = None) -> None:
    """Write a binary file through a temporary file in the same folder, then put the file at its path in one step.

    os.replace puts the new file in the place of a symbolic link, and it does not write through the link. A model
    folder of a different user can hold a link with the name of an output file, e.g. a link to ~/.bashrc. A new file
    takes the mode 0666 less the umask, as a file of open() does. A file that replaces a regular file takes its mode.
    A mode that is not None gives the mode of the new file in each case.
    """
    import secrets
    import stat

    try:
        deststat: os.stat_result | None = filepath.lstat()
    except FileNotFoundError:
        deststat = None
    partialfilepath = filepath.with_name(f".{filepath.name}.partial{secrets.token_hex(8)}")
    # O_EXCL fails for a name that exists, e.g. a link, and the kernel takes the umask away from the mode 0666
    fd = os.open(partialfilepath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
        if mode is None and deststat is not None and stat.S_ISREG(deststat.st_mode):
            mode = stat.S_IMODE(deststat.st_mode)
        if mode is not None:
            partialfilepath.chmod(mode)
        partialfilepath.replace(filepath)
    finally:
        partialfilepath.unlink(missing_ok=True)


def write_text_file_atomic(filepath: Path, write: Callable[[t.TextIO], object]) -> None:
    """Write a UTF-8 text file with write_file_atomic."""

    def write_text(stream: t.IO[bytes]) -> None:
        textstream = io.TextIOWrapper(stream, encoding="utf-8")
        write(textstream)
        textstream.flush()
        # the binary stream stays open, because write_file_atomic closes it
        textstream.detach()

    write_file_atomic(filepath, write_text)


def get_file_copies(filepath: Path) -> list[Path]:
    """Return the paths of the uncompressed copy and of each compressed copy of a file, e.g. model.txt.zst."""
    uncompressedpath = without_compressed_extension(filepath)
    return [uncompressedpath, *(with_compressed_extension(uncompressedpath, ext) for ext in COMPRESSED_EXTENSIONS)]


def find_compressed(filename: Path | str) -> tuple[str, Path] | None:
    """Return the extension and the path of filename.zst, .gz, or .xz, or None if no compressed file exists."""
    for ext in COMPRESSED_EXTENSIONS:
        path_withext = with_compressed_extension(filename, ext)
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

    return {".zst": zstd.open, ".gz": gzip.open, ".xz": lzma.open}[ext]  # ruff:ignore[banned-api]


def write_zstd_lines(fileobj: t.IO[bytes], lines: Sequence[str], compression_level: int | None = None) -> None:
    """Append the lines to an open binary file as one zstd frame.

    Each write_csv call with compression="zstd" adds one frame. ARTIS and zopen read a file of several frames.
    A compression_level of None gives the default level of polars.
    """
    pl.DataFrame({"line": lines}, schema={"line": pl.String}).write_csv(
        fileobj, include_header=False, quote_style="never", compression="zstd", compression_level=compression_level
    )


@t.overload
def zopen(
    filename: Path | str, mode: t.Literal["rb"], encoding: str | None = None, errors: str | None = None
) -> t.IO[bytes]: ...


@t.overload
def zopen(
    filename: Path | str, mode: str = "rt", encoding: str | None = None, errors: str | None = None
) -> t.IO[str]: ...


def zopen(
    filename: Path | str, mode: str = "rt", encoding: str | None = None, errors: str | None = None
) -> t.IO[str] | t.IO[bytes]:
    """Open filename, falling back to filename.zst, filename.gz or filename.xz.

    The named file wins, thus a stale compressed copy never shadows a file that a run has just written.
    This is the same precedence that read_wsv and firstexisting use. The errors argument takes the value
    that the open functions take, e.g. "replace" for a file that holds a bad byte.
    """
    check_local_path(filename)
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


def polars_source(filename: Path | str, mode: str = "r") -> t.IO[bytes] | Path:
    """Return the path of a file that polars reads itself, or a file object that decompresses it.

    polars reads a plain file, a zstd file, and a gzip file from the path. It cannot read an xz file.
    The caller gives the name of a file that exists. Use zopenpl to also find a compressed sibling.
    """
    check_local_path(filename)
    filepath = Path(filename)
    if filepath.suffix not in COMPRESSED_EXTENSIONS or filepath.suffix in POLARS_READABLE_EXTENSIONS:
        return filepath

    # the default mode "r" opens a binary stream in all three backends, which is what polars reads
    return get_decompress_open(filepath.suffix)(filepath, mode=mode)


@contextlib.contextmanager
def polars_source_open(filename: Path | str, mode: str = "r") -> Generator[t.IO[bytes] | Path]:
    """Yield a polars source and close it after the caller has read it.

    polars reads a file object when it builds the query plan, thus the object can close as soon as
    the scan_csv or read_csv call returns. A caller that drops the object instead leaks one file
    descriptor for each xz file that it reads.
    """
    source = polars_source(filename, mode=mode)
    if isinstance(source, Path):
        yield source
    else:
        with source:
            yield source


def zopenpl(filename: Path | str) -> t.IO[bytes] | Path:
    """Return a polars source for filename, or for a compressed sibling when the named file does not exist.

    The named file wins, for the same reason as in zopen. If polars can read the file directly, this
    returns a Path rather than a file object.
    """
    filepath = Path(filename)
    if not filepath.is_file() and (found := find_compressed(filename)):
        return polars_source(found[1])

    return polars_source(filepath)


@cache
def get_decompress_errors() -> tuple[type[Exception], ...]:
    """Return the error types that the .xz and .zst modules raise for damaged data. They are not an OSError.

    A Python build can lack a module. The caller evaluates this in an except clause, thus an ImportError here would
    replace the original error. The function leaves out the type of a module that is not available.
    """
    errors: list[type[Exception]] = []
    try:
        import lzma

        errors.append(lzma.LZMAError)
    except ImportError:
        pass
    try:
        if sys.version_info >= (3, 14):
            from compression.zstd import ZstdError
        else:
            from zstandard import ZstdError

        errors.append(ZstdError)
    except ImportError:
        pass
    return tuple(errors)


@contextlib.contextmanager
def polars_error_note(filepath: Path) -> Generator[None]:
    """Name the file in a read error, because the parser sees a path or in-memory bytes only.

    A damaged .xz or .zst file gives the error type of its module. A cut compressed file gives one of three errors.
    polars gives an OSError for a .gz file that it reads itself. A Python reader gives an EOFError, and polars gives a
    panic when it reads from that reader. The dispatcher reports an OSError with no traceback, thus an EOFError
    becomes an OSError.
    """
    try:
        yield
    except (pl.exceptions.PolarsError, pl.exceptions.PanicException, OSError, *get_decompress_errors()) as exc:
        exc.add_note(f"while reading {filepath}")
        raise
    except EOFError as exc:
        msg = (
            f"{filepath} ends before the end of its compressed data."
            " A partial copy or a run that has not finished can leave such a file"
        )
        raise OSError(msg) from exc


def scan_lines(filepath: Path, skip_rows: int = 0, encoding: t.Literal["utf8", "utf8-lossy"] = "utf8") -> pl.LazyFrame:
    """Return a lazy frame with one string column named "line" that holds each line of a text file.

    The function drops the first skip_rows lines. polars_source selects the compression format. The caller
    gives a path that it has resolved, thus this function must not search for a compressed sibling.
    """
    with polars_source_open(filepath, mode="rb") as source:
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
    line_filter: pl.Expr | None = None,
) -> pl.LazyFrame:
    """Return the lines of a text file with each run of whitespace collapsed to a single space.

    The function drops the first skip_rows lines, and it keeps only the lines for which line_filter is true. It then
    removes the text from comment_prefix to the end of a line, and it drops a blank line.
    """
    line = pl.col("line")
    if comment_prefix:
        # a comment can start anywhere on a line, thus keep the text before the first prefix only. split
        # takes the prefix as text, thus no escape of a regular expression character is necessary
        line = line.str.split(comment_prefix).list.first()

    lines = scan_lines(filepath, skip_rows=skip_rows, encoding=encoding)
    return (
        (lines.filter(line_filter) if line_filter is not None else lines)
        # the class holds each ASCII whitespace character that a line can contain. The Unicode class \s
        # would also split a field that holds a no-break space, which an ARTIS column must keep
        .select(line.str.replace_all(r"[ \t\r\x0b\x0c]+", " ").str.strip_chars(" "))
        .filter(pl.col("line").str.len_bytes() > 0)
    )


def read_decompressed_bytes(filepath: Path) -> bytes:
    """Return the bytes of a file. The function decompresses a .zst, .gz, or .xz file."""
    if filepath.suffix in COMPRESSED_EXTENSIONS:
        with get_decompress_open(filepath.suffix)(filepath, mode="rb") as fin:
            data: bytes = fin.read()
        return data

    return filepath.read_bytes()


def read_complete_lines(filename: Path | str, content: str) -> bytes:
    """Return the bytes of the complete lines of an ARTIS output file, which can be compressed.

    sn3d writes some files again at each timestep, thus a run that stops during that write leaves an empty file or
    a cut last line. The cut number of such a line is a valid number, thus only the missing line end shows the cut.
    The function drops a cut last line with a warning, and it raises a ValueError if no line remains. The content
    names what the file holds in that error, e.g. "light curve". The named file wins over a compressed sibling, as
    in zopen. The caller gives the bytes to polars, thus the function decompresses the file one time only.
    """
    check_local_path(filename)
    filepath = Path(filename)
    if not filepath.is_file() and (found := find_compressed(filename)):
        filepath = found[1]

    with polars_error_note(filepath):
        data = read_decompressed_bytes(filepath)

    if data.strip() and not data.endswith(b"\n"):
        print_warning(f"{filename} ends with a cut line, thus the command reads the file without its last line")
        # a file of one cut line holds no line end, thus rfind gives -1 and no byte remains
        data = data[: data.rfind(b"\n") + 1]

    if not data.strip():
        msg = f"{filename} holds no {content}. A run that stops while sn3d writes this file leaves it empty"
        raise ValueError(msg)

    return data


def raise_if_cut_line(df: pl.DataFrame, textfilepath: Path | str) -> None:
    """Stop if a line of an ARTIS text file has fewer values than the header.

    ARTIS writes each value of each line, thus a missing value shows a file that is not complete. A cache from such
    a file keeps the partial data, also after the write ends.
    """
    if df.null_count().sum_horizontal().item() > 0:
        msg = (
            f"The file {textfilepath} has a line with fewer values than the header. Possibly ARTIS or a copy"
            " still writes the file, or the end of the file is missing"
        )
        raise ValueError(msg)


def raise_if_no_final_line_end(filepath: Path) -> None:
    """Stop if an ARTIS text file is empty or does not end with a line end.

    ARTIS ends each row with a line end. A cut inside the last value gives a valid number, thus only the missing line
    end shows the cut. A compressed file of more than one frame that is cut at the end of a frame decompresses with no
    error, thus the function decompresses all of a compressed file. It reads the file in chunks and keeps only the
    last byte, because a line list can have tens of millions of lines. The reader then decompresses the file again.
    """
    lastbyte = b""
    # zstandard reads an empty .zst file as no data, but compression.zstd of Python 3.14 raises an EOFError. Thus the
    # function does not open an empty file, and all versions give the same error
    if filepath.stat().st_size > 0:
        if filepath.suffix in COMPRESSED_EXTENSIONS:
            with polars_error_note(filepath), get_decompress_open(filepath.suffix)(filepath, mode="rb") as fin:
                while chunk := fin.read(1 << 24):
                    lastbyte = chunk[-1:]
        else:
            with filepath.open("rb") as fin:
                fin.seek(-1, os.SEEK_END)
                lastbyte = fin.read(1)

    if lastbyte != b"\n":
        msg = (
            f"The file {filepath} is empty or does not end with a line end. Possibly ARTIS or a copy still writes the"
            " file, or the end of the file is missing"
        )
        raise ValueError(msg)


def bytes_outside_comments_are_utf8(filepath: Path, skip_rows: int = 0, comment_prefix: str | None = None) -> bool:
    """Return True if each byte that no comment holds is valid UTF-8.

    The function reads the bytes of the file and answers at the byte level, as the reader of an earlier
    version did. A decoder cannot answer it, because a decoder replaces every bad byte, and a file can
    also hold the replacement character as data. This costs a read of the whole file, thus call it only
    after a strict read has failed.
    """
    data = read_decompressed_bytes(filepath)

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


def normalise_whitespace(
    filepath: Path, skip_rows: int = 0, comment_prefix: str | None = None, line_filter: pl.Expr | None = None
) -> io.BytesIO:
    """Return a buffer of the normalised lines of a text file, which a CSV parser can read.

    The separator of the buffer is a single space, and the buffer is at position zero. A byte that is not
    valid UTF-8 raises an error, unless the comment step removes the text that holds it. The buffer holds only
    the lines for which line_filter is true.
    """

    def sink(lflines: pl.LazyFrame) -> io.BytesIO:
        normalised = io.BytesIO()
        lflines.sink_csv(normalised, include_header=False, quote_style="never")
        # sink_csv leaves the buffer at the end, thus rewind it for the reader
        normalised.seek(0)

        return normalised

    try:
        return sink(normalised_lines(filepath, skip_rows, comment_prefix, line_filter=line_filter))
    except pl.exceptions.ComputeError:
        # polars decodes a line before the comment step removes it, thus a comment that holds a byte
        # which is not valid UTF-8 stops the read. Such a comment is common in a file from a different
        # source, e.g. a degree sign in Latin-1. Give the caller the first error if a column holds such
        # a byte, because the values of that column would be text that no reader can trust
        if not bytes_outside_comments_are_utf8(filepath, skip_rows, comment_prefix):
            raise

        # only a comment holds a bad byte, thus read the file again and replace each one
        return sink(
            normalised_lines(filepath, skip_rows, comment_prefix, encoding="utf8-lossy", line_filter=line_filter)
        )


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
    line_filter: pl.Expr | None = None,
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
    line_filter is an expression on the column "line" of the raw text, and the function keeps only the lines for
    which it is true. It must keep the header line. A filter that drops most lines before the parse saves the
    memory of a large file.
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
        normalised = normalise_whitespace(
            filepath, skip_rows=skip_rows, comment_prefix=comment_prefix, line_filter=line_filter
        )

    def parse(infer_schema_length: int | None) -> pl.DataFrame:
        # this function runs again when the first schema turns out to be wrong, thus rewind the buffer
        normalised.seek(0)

        # a lazy scan gets the name of every column, and the projection pushdown then parses only the
        # requested columns. read_csv rejects a list of names that holds only some of the columns
        lzscan = pl.scan_csv(
            normalised,
            separator=" ",
            has_header=has_header,
            new_columns=list(new_columns) if new_columns is not None else None,
            schema_overrides=schema_overrides,
            infer_schema_length=infer_schema_length,
            null_values=["nan", "NaN", "-nan", "-NaN", "NA", "N/A", "null", "NULL"],
            # the buffer comes from a sink that quotes nothing, thus a quotation mark in a value is
            # data. A reader that takes it as a quote joins the fields that follow it into one column
            quote_char=None,
        )
        return (lzscan.select(list(columns)) if columns is not None else lzscan).collect()

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
        return parse_with_inference_fallback()


@on_model_host
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
    fullpaths = []

    def search_folders(filelist: list[str | Path] | list[Path]) -> Generator[Path]:
        yield Path(folder)
        if search_subfolders:
            for filename in filelist:
                # a glob gives the names in the order of the file system, thus the result would
                # differ between two machines. The natural order of the folders gives the order of the
                # runs, and get_run_subfolders and the cache checks take the same order
                yield from sorted({p.parent for p in Path(folder).glob(f"*/{filename}*")}, key=natural_sort_key)

    for searchfolder in search_folders(filelist):
        for filename in filelist:
            thispath = Path(searchfolder, filename)
            if thispath.exists():
                return thispath

            fullpaths.append(thispath)

            if tryzipped:
                for ext in COMPRESSED_EXTENSIONS:
                    filename_withext = with_compressed_extension(filename, ext)
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


def natural_sort_key(path: Path | str) -> tuple[tuple[int, int, str], ...]:
    """Return a sort key that orders a path by the value of each number that it holds.

    A run folder carries the number of the job, e.g. "9876543.slurm". A lexical order puts
    "10000001.slurm" in front of that folder, thus the later run would come first. The key of a
    number holds the text as well, because "job01" and "job1" give the same number.
    """
    return tuple(
        (0, int(token), token) if token.isdigit() else (1, 0, token) for token in re.split(r"(\d+)", str(path)) if token
    )


def readnoncommentline(file: t.IO[str]) -> str:
    """Read a line from the text file, skipping blank and comment lines that begin with #.

    Raise EOFError if the end of the file is reached before any non-blank, non-comment line is found.
    """
    while line := file.readline():
        if line.strip() and not line.lstrip().startswith("#"):
            return line

    msg = "Reached end of file without finding a non-comment, non-blank line"
    raise EOFError(msg)


def check_metadata_mapping(metadata: object, metafile: Path) -> dict[str, t.Any]:
    """Return the metadata of a file, or raise ValueError when the metadata file gives a value that is not a mapping."""
    if isinstance(metadata, dict):
        return t.cast("dict[str, t.Any]", metadata)

    # the dispatcher reports a ValueError as a fault of the input, without a traceback
    msg = f"The metadata in {metafile} must be a mapping of names to values, not {metadata!r}"
    raise ValueError(msg)


def get_file_metadata(filepath: Path | str) -> dict[str, t.Any]:
    """Return a dict of metadata for a file from its metadata file, e.g. spectrum.txt.meta.yml for spectrum.txt.

    The key of the cache holds the absolute path of the metadata file and its text. The absolute path keeps a symbolic
    link, because the metadata file of a link is beside the link. The text makes a process that runs for a long time,
    e.g. a viewer, read each edit. A metadata file holds only a few lines.
    """
    metafile = get_metadata_path(Path(filepath).absolute())
    # Path.exists() gives False for each error of the operating system, e.g. a symbolic link in a loop. An editor can
    # delete the file and write it again after exists(), thus a missing file at the read also gives no metadata
    metatext = None
    if metafile.exists():
        with contextlib.suppress(FileNotFoundError):
            metatext = metafile.read_text(encoding="utf-8")
    return get_file_metadata_cached(metafile, metatext)


def parse_metadata_redshift(value: object, metafile: Path, minimum: float, allowminimum: bool) -> float:
    """Return the redshift z from a metadata file, or raise a ValueError that names the file.

    z must be above minimum, or equal to it if allowminimum, e.g. above -1 for a de-redshift and 0 or above for a
    luminosity distance. get_file_metadata keeps z as YAML gives it, because a caller that reads only the label must
    not fail on z.
    """
    # str() makes a bool fail, and float() accepts True
    try:
        z = float(str(value))
    except ValueError:
        z = math.nan
    if not (math.isfinite(z) and (z >= minimum if allowminimum else z > minimum)):
        limit = f"of {minimum:g} or above" if allowminimum else f"above {minimum:g}"
        msg = f"The redshift z in {metafile} must be a number {limit}, not {value!r}"
        raise ValueError(msg)
    return z


def get_metadata_path(filepath: Path) -> Path:
    """Return the path of the metadata file of a file, e.g. spectrum.txt.meta.yml for spectrum.txt.xz."""
    filepath = without_compressed_extension(filepath)
    return filepath.with_suffix(f"{filepath.suffix}.meta.yml")


@lru_cache(maxsize=24)
def get_file_metadata_cached(metafile: Path, metatext: str | None) -> dict[str, t.Any]:
    """Return the metadata from the text of a metadata file, and keep it for the next caller."""

    def add_derived_metadata(metadata: dict[str, t.Any]) -> dict[str, t.Any]:
        if "a_v" in metadata and "e_bminusv" in metadata and "r_v" not in metadata:
            metadata["r_v"] = metadata["a_v"] / metadata["e_bminusv"]
        elif "e_bminusv" in metadata and "r_v" in metadata and "a_v" not in metadata:
            metadata["a_v"] = metadata["e_bminusv"] * metadata["r_v"]
        elif "a_v" in metadata and "r_v" in metadata and "e_bminusv" not in metadata:
            metadata["e_bminusv"] = metadata["a_v"] / metadata["r_v"]

        return metadata

    import yaml

    class MetadataLoader(yaml.SafeLoader):
        """Read a number with an exponent and no dot as a float, e.g. 1e-3, as YAML 1.2 does and YAML 1.1 does not."""

    # add_implicit_resolver copies the resolvers of SafeLoader before it adds one, thus SafeLoader stays the same
    MetadataLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float", re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"), list("-+0123456789")
    )

    if metatext is None:
        print(f"No metadata file at {metafile}")
        return {}

    # a file of comments alone gives None. MetadataLoader is a SafeLoader, but ruff does not know the subclass
    metadata = yaml.load(metatext, Loader=MetadataLoader) or {}  # ruff:ignore[unsafe-yaml-load]
    return add_derived_metadata(check_metadata_mapping(metadata, metafile))
