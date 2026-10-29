"""Keep the parquet cache of a text file up to date. A run that stops early leaves no corrupt cache file."""

import contextlib
import datetime as dt
import errno
import os
import sys
import time
import typing as t
from collections.abc import Generator
from pathlib import Path

import polars as pl


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


def get_file_state(path: Path) -> tuple[int, int, int, int]:
    """Return the device, the inode, the modification time, and the size of a file, for the key of a cache.

    A rewrite in place changes the time or the size, and a rename onto the path changes the device or the inode.
    """
    filestat = path.stat()
    return (filestat.st_dev, filestat.st_ino, filestat.st_mtime_ns, filestat.st_size)


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
    # NFS gives flock as a POSIX lock, and an exclusive POSIX lock needs a descriptor that can write. A user without
    # write access to the lock of a different user reads it, which flock accepts on a local file system
    try:
        lockfd = os.open(lockpath, os.O_CREAT | os.O_RDWR, 0o666)
    except PermissionError:
        lockfd = os.open(lockpath, os.O_CREAT | os.O_RDONLY, 0o666)
    # the chmod lets each user write the lock under a restrictive umask, and fails harmlessly for a user who does
    # not own the lock
    with contextlib.suppress(OSError):
        lockpath.chmod(0o666)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX)
        identity = get_file_identity(destpath)
        if identity is None or identity == outdatedfile:
            newfilepath.replace(destpath)
    finally:
        os.close(lockfd)


def format_mtime(mtime: float | str | None) -> str:
    """Return a file modification time as a local time string with the raw number.

    A cache that holds no stamp gives "absent". A stamp that no writer of this repository can
    produce, e.g. a hand-edited one, comes back unchanged.
    """
    if mtime is None:
        return "absent"
    try:
        localtime = dt.datetime.fromtimestamp(float(mtime)).astimezone()
    except (ValueError, OSError, OverflowError):
        return str(mtime)
    return f"{localtime.isoformat(sep=' ', timespec='seconds')} ({mtime})"


# the largest difference between a stamped modification time and the time of the text source that
# still counts as the same time. A Google Drive mount moved the times of the 960 packets files of one
# run by up to 6 seconds with no write, and every batch cache of that run became stale. The value
# keeps a margin above that, because the drift grows with the time that the ranks take to write
MTIME_TOLERANCE_S = 30.0


def mtime_matches_stamp(foundmtime: str | None, textsource_mtime: float) -> bool:
    """Return True when a stamped modification time counts as the time of the text source.

    A cache that holds no stamp dates its text source in no way, thus it matches no time. The caller
    decides what to do with such a cache, see the accept_unstamped argument of
    read_parquet_cache_metadata. A stamp that no writer of this repository can produce, e.g. a
    hand-edited one, matches only the same text. The size of the text source cannot decide this
    question: zstd keeps the time of a packets file and changes the size.
    """
    if foundmtime is None:
        return False

    try:
        return abs(float(foundmtime) - textsource_mtime) <= MTIME_TOLERANCE_S
    except ValueError:
        return foundmtime == str(textsource_mtime)


def rankbatch_parquet_staleness(
    parquetfilepath: Path, cacheversion: int, textsource_mtime: float | None, *, textsource_complete: bool
) -> str | None:
    """Return the reason why the parquet cache is stale, or None when the cache is current.

    The reader of all the batches and the writer of one batch both ask this. Thus one rule decides whether
    the code converts the text files again. read_parquet_cache_metadata gives every other reason,
    e.g. a cache that is absent, a cache that is damaged, or a cache of a different format version.

    A complete batch compares the newest text file with the stamp of the cache. An incomplete batch
    compares in one direction only. A text file that is newer than the stamp proves a rewrite. An
    absent text file proves nothing. The cache format version applies to a batch of either kind.

    A cache from before the stamps holds no stamp of its own. Such a cache stays in use for an
    incomplete batch, because the text files that rebuild it are gone. A complete batch holds every
    text file, thus an unstamped cache there is stale and the code converts the text files again.
    """
    # See the accept_unstamped argument of read_parquet_cache_metadata
    pqmetadata, stalereason = read_parquet_cache_metadata(
        parquetfilepath,
        cacheversion,
        textsource_mtime if textsource_complete else None,
        accept_unstamped=not textsource_complete,
    )
    if stalereason is not None or textsource_complete or textsource_mtime is None:
        return stalereason

    stampedmtime = (pqmetadata or {}).get("textsource_mtime")
    if stampedmtime is None:
        return None

    try:
        stampedvalue = float(stampedmtime)
    except ValueError:
        # a stamp that no writer of this repository can produce cannot date the text files
        return None

    if textsource_mtime > stampedvalue + MTIME_TOLERANCE_S:
        return (
            f"a text file of the batch changed at {format_mtime(textsource_mtime)},"
            f" after the cache stamp of {format_mtime(stampedmtime)}"
        )

    return None


def parquet_is_readable(parquetfilepath: Path) -> bool:
    """Return True when polars can read the metadata of a parquet file.

    A damaged cache is no source of data at all. A reader that keeps a stale cache, because no text
    file remains to rebuild it, still needs this test.
    """
    try:
        pl.read_parquet_metadata(parquetfilepath)
    except (FileNotFoundError, pl.exceptions.PolarsError, OSError):
        return False

    return True


def read_parquet_cache_metadata(
    parquetfilepath: Path,
    cacheversion: int,
    textsource_mtime: float | None,
    *,
    accept_unstamped: bool = False,
    textsource: str | None = None,
    textsource_size: int | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    """Return the metadata of a parquet cache, and the reason why the cache is stale.

    The writer of a cache stamps the cache format version and the modification time of its text
    source into the parquet metadata. A cache from a different artistools version, or from different
    text files, fails the comparison. The comparison of the times has the tolerance
    MTIME_TOLERANCE_S, because a file system can move the time of a file that no write changed. A new
    modification time of the cache does not make it current. read_parquet_metadata is eager, thus a
    damaged file gives a reason here and not an error at a distant collect().

    A current cache gives its metadata and no reason. A stale cache gives no metadata and the reason
    for the rejection, because a regeneration of a large cache costs minutes and the user must see
    what caused it. Each reason reads as a lower-case clause after the word "because".

    An artistools version before the stamps wrote a cache that holds neither stamp.
    accept_unstamped keeps such a cache: the version counts as the version of the reader, and the
    absent time matches. Only a reader whose text files cost hours to convert takes that option, e.g.
    the estimators and the packets of an archived model folder. A cache that a few seconds rebuild, e.g. the
    model or the line list, keeps the strict rule, because a stale one there gives wrong numbers for
    no gain.

    A writer that can read more than one text file stamps the path of the text file as "textsource". The path is
    relative to the model folder and has no compression extension, e.g. "linelist.out", because a compression keeps
    the modification time. Give the same path to such a reader, because two files can have the same modification
    time. A cache without the stamp is then stale.

    A textsource_mtime of None shows that the text source is absent. The function then compares no
    modification times, but the cache format version and the state of the file still apply.
    """
    try:
        pqmetadata = pl.read_parquet_metadata(parquetfilepath)
    except FileNotFoundError:
        return None, "the file does not exist"
    except (pl.exceptions.PolarsError, OSError) as exc:
        return None, f"the file is not a readable parquet file ({type(exc).__name__}: {exc})"

    foundversion = pqmetadata.get("cacheversion", str(cacheversion) if accept_unstamped else None)
    if foundversion != str(cacheversion):
        return None, (
            f"the cache format version is {foundversion}, but this artistools version writes {cacheversion}"
            if foundversion is not None
            else "the file has no cacheversion stamp, thus an artistools version before the stamp wrote it"
        )

    # accept_unstamped does not apply to the text source, thus this check comes before the return for an unstamped
    # cache
    if textsource is not None and (foundtextsource := pqmetadata.get("textsource")) != textsource:
        return None, (
            f"the cache is from {foundtextsource}, but the text source is now {textsource}"
            if foundtextsource is not None
            else "the cache has no record of its text source"
        )

    # a copy that keeps the modification time, e.g. cp -p, can replace the text source, thus the size counts too
    foundsize = pqmetadata.get("textsource_size")
    if textsource_size is not None and foundsize is not None and foundsize != str(textsource_size):
        return (
            None,
            f"the text source had {foundsize} bytes at the write of the cache, but it now has {textsource_size} bytes",
        )

    foundmtime = pqmetadata.get("textsource_mtime")
    if foundmtime is None and accept_unstamped:
        return pqmetadata, None

    if textsource_mtime is not None and not mtime_matches_stamp(foundmtime, textsource_mtime):
        return None, (
            f"the text source changed: the cache stamp is {format_mtime(foundmtime)},"
            f" but the text file now has {format_mtime(textsource_mtime)}"
        )

    return pqmetadata, None


def is_readonly_error(exc: OSError) -> bool:
    """Return True when the file system refuses a write, for a folder with no write permission or a read-only mount.

    A read-only mount gives errno EROFS, which Python raises as a plain OSError and not as a PermissionError.
    """
    return isinstance(exc, PermissionError) or exc.errno == errno.EROFS


def make_cache_folder(folder: Path, parentfolder: Path) -> None:
    """Make a cache folder with the access mode of its parent folder, if it does not exist.

    The folder takes the group and the access mode of parentfolder, with its setgid and sticky bits, thus a
    group that can write the parent folder can add a cache. A mkdir alone applies the umask of the process,
    which removes the group write bit. A different process can make the folder at the same time, and only its
    owner can change the mode. That process applies the mode after its mkdir, thus a process of a different
    user waits until a new folder is writable.

    A folder that exists and stays read-only passes, because a read of the caches in it needs no write, and a
    write then gives the error. A read-only parent folder gives a PermissionError with a readable message.
    """
    try:
        folder.mkdir()
    except FileExistsError:
        # the process that made the folder applies the mode a few microseconds after its mkdir, thus a folder that
        # is not writable gets a short wait when it is new. A folder that stays read-only passes at once, e.g. a
        # shared model whose caches are current, and a write then gives the error
        for _ in range(20):
            if os.access(folder, os.W_OK) or time.time() - folder.stat().st_mtime > 2.0:
                break
            time.sleep(0.1)
        return
    except OSError as exc:
        # a read-only mount gives EROFS and not PermissionError
        if not is_readonly_error(exc):
            raise
        msg = f"artistools cannot make its cache folder {folder}, because the folder {parentfolder} is read-only"
        raise PermissionError(msg) from exc

    if sys.platform == "win32":
        # Windows has no POSIX group or mode, and os.open rejects a folder there
        return

    parentstat = parentfolder.stat()
    # the descriptor of the new folder, and not its path, gets the group and the mode, thus a different user who
    # can write the parent folder cannot replace the folder by a symbolic link to a file between the calls
    folderfd = os.open(folder, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
    try:
        # a mkdir gives the folder the primary group of the user, where the parent folder can belong to a project
        # group. Only a member of the group can give it, and some file systems reject a chmod, thus a failure keeps
        # the group and the mode of the mkdir
        with contextlib.suppress(OSError):
            os.fchown(folderfd, -1, parentstat.st_gid)
        with contextlib.suppress(OSError):
            # the setgid bit and the sticky bit of the parent folder apply to the caches too: a sticky folder keeps
            # the cache of one user safe from a different user
            os.fchmod(folderfd, parentstat.st_mode & 0o3777)
    finally:
        os.close(folderfd)


def get_readonly_cache_error(parquetfilepath: Path) -> PermissionError:
    """Return the error for a cache that artistools cannot write, because its folder is read-only."""
    msg = (
        f"artistools cannot write its cache {parquetfilepath.name}, because the folder"
        f" {parquetfilepath.parent} is read-only"
    )
    return PermissionError(msg)


@contextlib.contextmanager
def temporary_parts_folder(parquetfilepath: Path) -> Generator[Path]:
    """Give a temporary folder beside a cache for the parts of the cache, and remove the folder after use.

    The folder is on the file system of the cache, because the parts of a large model need as much space as the cache.
    The name of the folder starts with a dot, thus no glob of the caches finds it.
    """
    import tempfile

    try:
        partsfolder = tempfile.TemporaryDirectory(
            dir=parquetfilepath.parent, prefix=f".{parquetfilepath.name}.parts", ignore_cleanup_errors=True
        )
    except OSError as exc:
        if not is_readonly_error(exc):
            raise
        raise get_readonly_cache_error(parquetfilepath) from exc
    with partsfolder as partsfoldername:
        yield Path(partsfoldername)


# up to 2026-10, the parquet caches had this suffix after .parquet. The suffix did not show a temporary file,
# because write_parquet_atomic gives a temporary file a name that starts with a dot
LEGACY_CACHE_SUFFIX: t.Final = ".tmp"


def get_parquet_cache_name(cachepath: Path) -> Path:
    """Return cachepath, or the cache of the earlier name cachepath + ".tmp" when only that file exists.

    A model folder can hold a cache of the earlier name. A reader then uses that cache, thus artistools does not convert
    the data again. A new cache takes the name cachepath.
    """
    legacypath = cachepath.with_name(cachepath.name + LEGACY_CACHE_SUFFIX)
    return legacypath if not cachepath.exists() and legacypath.is_file() else cachepath


def write_parquet_atomic(
    pldf: pl.DataFrame | pl.LazyFrame,
    parquetfilepath: Path,
    metadata: dict[str, str] | None = None,
    replaces: tuple[int, int] | None = None,
    compression_level: int = 13,
    row_group_size: int | None = None,
) -> None:
    """Write a parquet file with zstd compression through a temporary file, thus no reader sees a partial file.

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

    try:
        fd, partialfilename = tempfile.mkstemp(
            dir=parquetfilepath.parent, prefix=f".{parquetfilepath.name}.partial", suffix=".partial"
        )
    except OSError as exc:
        if not is_readonly_error(exc):
            raise
        raise get_readonly_cache_error(parquetfilepath) from exc
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
            partialfilepath,
            compression="zstd",
            compression_level=compression_level,
            metadata=metadata,
            row_group_size=row_group_size,
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
