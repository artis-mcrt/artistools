"""Build ARTIS model abundances from r-process nucleosynthesis trajectory calculations."""

import argparse
import gc
import math
import posixpath
import string
import tarfile
import time
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from functools import partial
from itertools import batched
from itertools import chain
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.inputmodel.core import MODEL_FILE_NAMES
from artistools.inputmodel.core import prepare_zstd_output
from artistools.inputmodel.core import save_initelemabundances
from artistools.inputmodel.core import save_modeldata
from artistools.misc import addarg_force
from artistools.misc import addarg_output
from artistools.misc import confirm_overwrite
from artistools.misc import firstexisting
from artistools.misc import firstexisting_or_none
from artistools.misc import get_file_identity
from artistools.misc import get_file_state
from artistools.misc import make_cache_folder
from artistools.misc import parallel_map
from artistools.misc import parse_cli_args
from artistools.misc import polars_source
from artistools.misc import print_warning
from artistools.misc import read_parquet_cache_metadata
from artistools.misc import read_wsv
from artistools.misc import write_parquet_atomic
from artistools.misc import zopen

TRAJECTORY_CACHEVERSION = 2
# the columns of energy_thermo.dat and heating.dat that the readers use. The cache holds these alone
READER_COLUMNS = ("nstep", "time_s", "Qdot", "hbeta", "halpha", "hspof", "htot")


def get_elemabund_from_nucabund(dfnucabund: pl.DataFrame) -> dict[str, float]:
    """Return a dictionary of elemental abundances from nuclear abundance DataFrame."""
    ZMAX = dfnucabund["Z"].max()
    assert isinstance(ZMAX, int)
    massfrac_of_z = dict(dfnucabund.group_by("Z").agg(pl.col("massfrac").sum()).iter_rows())
    return {
        f"X_{get_elsymbol(atomic_number)}": float(massfrac_of_z.get(atomic_number, 0.0))
        for atomic_number in range(1, ZMAX + 1)
    }


def get_dfelemabund_from_dfmodel(dfmodel: pl.DataFrame) -> pl.DataFrame:
    """Return per-element mass fractions by summing the isotopic X_ columns of the model."""
    timestart = time.perf_counter()
    print("Adding up isotopes for elemental abundances and creating dfelabundances...", end="", flush=True)

    isoabundcolnames = [
        colname for colname in dfmodel.collect_schema().names() if colname.startswith("X_") and colname[-1].isdigit()
    ]
    atomic_numbers = [get_atomic_number(colname[2:].rstrip(string.digits)) for colname in isoabundcolnames]

    elemisotopes: dict[int, list[str]] = {k: [] for k in set(atomic_numbers)}

    for atomic_number, colname in zip(atomic_numbers, isoabundcolnames, strict=True):
        elemisotopes[atomic_number].append(colname)

    dfelabundances = dfmodel.select(
        "inputcellid",
        *[
            pl.sum_horizontal(elemisotopes.get(atomic_number, pl.lit(0.0))).alias(f"X_{get_elsymbol(atomic_number)}")
            for atomic_number in range(1, max(atomic_numbers) + 1)
        ],
    )

    print(f" Took {time.perf_counter() - timestart:.1f} seconds")
    print(f" There are {len(isoabundcolnames)} nuclides from {len(elemisotopes)} elements included")

    return dfelabundances


def fix_fortran_exponents(dtype: pl.DataType | type[pl.DataType]) -> pl.Expr:
    """Return an expression that repairs Fortran triple-digit exponents, then casts to dtype.

    Fortran writes a value like 1.735904-244 without the "e", thus the column parses as strings. The cast is
    strict, thus a value that is corrupt for a different reason still raises. Only the sign before the three
    exponent digits gets the "e", thus a negative mantissa keeps its sign.
    """
    return cs.by_dtype(pl.String).str.replace(r"(\d)([+-])(\d{3})$", "${1}e${2}${3}").cast(dtype)


def parse_network_steps_table(text: str, sourcename: str) -> pl.DataFrame:
    """Parse a table of the network timesteps, e.g. energy_thermo.dat, whose first line names the columns.

    The timestep column "#count" becomes "nstep", and "time/s" becomes "time_s". The result holds the columns
    of READER_COLUMNS that the file has, with the names of the file, e.g. "Qdot" and "hbeta". A later line
    that starts with "#" is a comment or a second header, e.g. after a restart of the network, and the parse
    skips it. A restart can also write a timestep again, and the later row then replaces the earlier row. A
    corrupt value, e.g. a Fortran overflow token of asterisks, or a line with a different number of fields
    than the header, makes the table corrupt, because a reader must not get a null in place of a value. An
    incomplete last line alone, from a network run that stopped during a write, is left out with a warning
    that names sourcename.
    """
    headerline, _, body = text.partition("\n")
    colnames = [{"#count": "nstep", "time/s": "time_s"}.get(colname, colname) for colname in headerline.split()]
    fields = pl.col("line").str.extract_all(r"\S+")
    dflines = (
        pl
        .DataFrame({"line": body.splitlines()}, schema={"line": pl.String})
        .with_columns(pl.col("line").str.strip_chars())
        .filter((pl.col("line").str.len_chars() > 0) & ~pl.col("line").str.starts_with("#"))
        .select(
            fields.list.len().alias("fieldcount"),
            *(
                fields.list.get(index, null_on_oob=True).alias(colname)
                for index, colname in enumerate(colnames)
                if colname in READER_COLUMNS
            ),
        )
    )
    hasbadfieldcount = (dflines["fieldcount"] != len(colnames)).to_list()
    if any(hasbadfieldcount[:-1]):
        msg = f"A line has a different number of fields than the header, which has {len(colnames)} fields"
        raise ValueError(msg)
    if hasbadfieldcount and hasbadfieldcount[-1]:
        print_warning(f"The last line of {sourcename} is incomplete. The cache leaves out that line.")
        dflines = dflines.head(-1)

    return (
        dflines
        .drop("fieldcount")
        .with_columns(fix_fortran_exponents(pl.Float64))
        .with_columns(pl.col("nstep").cast(pl.Int32))
        .unique(subset="nstep", keep="last", maintain_order=True)
    )


def parse_abundance_files(abundancetexts: dict[str, str], sourcename: str) -> pl.DataFrame:
    """Return one row for each abundance file, e.g. nz-plane00200, with its header time and its nuclide lists.

    The first line of a file gives the timestep and the time. The other lines have a fixed width, thus each line
    goes to str.slice and not to a parser. One query reads all the files, because the fixed cost of a query is
    larger than the time to parse one file of 400 lines. The file nz-plane00200 is timestep 200. A named file,
    e.g. tday_nz-plane, has no timestep, because its time differs from the time of the timestep in its header.

    The function leaves out a file with a bad header line, a corrupt nuclide line, or no nuclide line, e.g.
    from a network run that stopped. A warning names the file and sourcename. The other files of the trajectory
    stay readable.
    """
    filenames: list[str] = []
    lines: list[str] = []
    for filename, text in abundancetexts.items():
        filelines = text.splitlines()
        filenames.extend([filename] * len(filelines))
        lines.extend(filelines)

    lzlines = (
        pl
        .DataFrame({"abundancefile": filenames, "line": lines}, schema={"abundancefile": pl.String, "line": pl.String})
        .lazy()
        .with_columns(linenumber=pl.int_range(pl.len()).over("abundancefile"))
    )
    headerfields = pl.col("line").str.extract_all(r"\S+")
    dfheaders, dfnuclides = pl.collect_all([
        lzlines.filter(pl.col("linenumber") == 0).select(
            "abundancefile",
            fieldcount=headerfields.list.len(),
            abundancetime_s=headerfields.list.get(1, null_on_oob=True).cast(pl.Float64, strict=False),
        ),
        lzlines
        .filter((pl.col("linenumber") > 0) & (pl.col("line").str.strip_chars().str.len_chars() > 0))
        .with_columns(
            pl.col("line").str.slice(0, 4).str.strip_chars().cast(pl.Int32, strict=False).alias("N"),
            pl.col("line").str.slice(4, 4).str.strip_chars().cast(pl.Int32, strict=False).alias("Z"),
            pl.col("line").str.slice(8, 13).str.strip_chars().cast(pl.Float64, strict=False).alias("log10abund"),
        )
        .group_by("abundancefile", maintain_order=True)
        .agg(
            pl.col("N"),
            pl.col("Z"),
            pl.col("log10abund"),
            hascorruptline=pl.any_horizontal(pl.col("N", "Z", "log10abund").is_null()).any(),
        ),
    ])

    # a blank header line gives no fields, a corrupt time gives null, and a file without lines gives no header row
    dfheaders = (
        dfheaders
        .join(dfnuclides.select("abundancefile", "hascorruptline"), on="abundancefile", how="left")
        .filter(
            (pl.col("fieldcount") == 6)
            & pl.col("abundancetime_s").is_not_null()
            # a file with no nuclide line, e.g. from a network run that stopped after the header, is corrupt too
            & ~pl.col("hascorruptline").fill_null(value=True)
        )
        .drop("fieldcount", "hascorruptline")
    )
    goodfiles = set(dfheaders["abundancefile"].to_list())
    for filename in abundancetexts:
        if filename not in goodfiles:
            print_warning(f"Cannot read {filename} in {sourcename}. The cache leaves out this file.")

    # each file of dfheaders has a nuclide group, thus the join leaves no list null
    return (
        dfheaders.join(dfnuclides.drop("hascorruptline"), on="abundancefile", how="left", maintain_order="left")
    ).with_columns(nstep=pl.col("abundancefile").str.extract(r"^nz-plane(\d+)$").cast(pl.Int32))


def is_network_data_file(filename: str) -> bool:
    """Return True for a file of the Run_rprocess folder that artistools reads.

    A macOS resource fork ._name gives False.
    """
    iswanted = filename in {"energy_thermo.dat", "heating.dat"} or "nz-plane" in filename
    return iswanted and not filename.startswith(".")


def read_network_texts(archivepath: Path) -> dict[str, str]:
    """Return the text of each file that artistools reads from a trajectory archive.

    The archive holds approximately 1000 files, and artistools reads the energy_thermo.dat, the heating.dat,
    and the nz-plane files of the Run_rprocess folder. A .tar.xz has one xz block, thus a read of one member
    also decompresses the full archive. A stream read decompresses the archive one time only.
    """
    texts: dict[str, str] = {}
    with (
        # zopen decompresses a .tar.zst or a .tar.xz on each Python version, where tarfile reads a .zst from 3.14
        zopen(archivepath, mode="rb") as archivestream,
        tarfile.open(fileobj=archivestream, mode="r|") as tarfilehandle,
    ):
        for member in tarfilehandle:
            folder, filename = posixpath.split(member.name)
            # the files are in the Run_rprocess folder at the top of the archive, and not in a nested copy of it
            if member.isfile() and posixpath.normpath(folder) == "Run_rprocess" and is_network_data_file(filename):
                memberfile = tarfilehandle.extractfile(member)
                assert memberfile is not None
                # a corrupt byte gives a corrupt token, which the parse reports, and not an unreadable trajectory
                texts[filename] = memberfile.read().decode("utf-8", errors="replace")

    return texts


def read_trajectory_network_data(archivepath: Path) -> tuple[pl.DataFrame, list[str]]:
    """Read the network data of one trajectory from its archive, and return one row for each timestep.

    A row holds the columns of READER_COLUMNS from energy_thermo.dat and heating.dat for its timestep, and the
    lists N, Z, and log10abund of its abundance file, if the timestep has one. The time of a timestep comes
    from energy_thermo.dat, or from heating.dat, or from the header of the abundance file. A named abundance
    file, e.g. tday_nz-plane, is a row with no timestep. The columns nstep and time_s always exist. A column of
    a file that the archive does not hold is absent, e.g. a test archive with the abundance files alone has no
    "Qdot" column.

    The function leaves out a table or an abundance file with a corrupt value and gives a warning, thus a
    reader of a different file of the trajectory can continue. The second result names the files that the
    function left out.
    """
    texts = read_network_texts(archivepath)
    if not texts:
        msg = f"{archivepath} holds no energy_thermo.dat, heating.dat, or nz-plane file in a Run_rprocess folder"
        raise FileNotFoundError(msg)

    frames = []
    leftoutfiles = []
    for filename in ("energy_thermo.dat", "heating.dat"):
        if filename not in texts:
            continue
        try:
            frames.append(parse_network_steps_table(texts[filename], f"{filename} in {archivepath.name}"))
        except (pl.exceptions.PolarsError, ValueError) as exc:
            # a reader of the abundance files must not stop for a corrupt value in a table that it does not read
            print_warning(f"Cannot read {filename} in {archivepath.name} ({exc}). The cache leaves out this file.")
            leftoutfiles.append(filename)
    abundancetexts = {name: text for name, text in texts.items() if "nz-plane" in name}
    dfabundances = parse_abundance_files(abundancetexts, archivepath.name)
    leftoutfiles.extend(name for name in abundancetexts if name not in set(dfabundances["abundancefile"].to_list()))
    frames.append(dfabundances)

    dftrajectory = frames[0]
    for dfnext in frames[1:]:
        sharedcols = [col for col in dfnext.columns if col in dftrajectory.columns and col != "nstep"]
        dftrajectory = dftrajectory.join(dfnext, on="nstep", how="full", coalesce=True, suffix="_next").with_columns(
            pl.coalesce(col, f"{col}_next").alias(col) for col in sharedcols
        )
        dftrajectory = dftrajectory.drop(f"{col}_next" for col in sharedcols)

    timecols = ["time_s", "abundancetime_s"] if "time_s" in dftrajectory.columns else ["abundancetime_s"]
    dftrajectory = dftrajectory.with_columns(time_s=pl.coalesce(timecols)).sort(
        "nstep", "abundancefile", nulls_last=True
    )
    return dftrajectory, leftoutfiles


def get_trajectory_archive_path(traj_root: Path, particleid: int) -> Path | None:
    """Return the archive of a trajectory, e.g. {particleid}.tar.xz, or None if the folder holds none.

    A .tar comes first, because its read needs no decompression, then a compressed .tar in the order of
    COMPRESSED_EXTENSIONS, which puts the .tar.zst before the .tar.xz. A .tar.zst decompresses five times
    faster than a .tar.xz.
    """
    plainnames = [f"{particleid}.tar", f"{particleid:05d}.tar"]
    plainpath = firstexisting_or_none(plainnames, folder=traj_root, tryzipped=False, search_subfolders=False)
    if plainpath is not None:
        return plainpath

    return firstexisting_or_none(plainnames, folder=traj_root, tryzipped=True, search_subfolders=False)


def get_trajectory_cache_path(traj_root: Path, particleid: int) -> Path:
    """Return the parquet cache of a trajectory, parquetcaches/{particleid}.parquet in the trajectory folder."""
    return traj_root / "parquetcaches" / f"{particleid}.parquet"


def scan_trajectory_network_data(traj_root: Path, particleid: int) -> pl.LazyFrame:
    """Scan the network data of a trajectory from its parquet cache, see read_trajectory_network_data.

    The first read of a trajectory reads its archive, and writes the cache
    parquetcaches/{particleid}.parquet in the trajectory folder. A later read scans the parquet
    file. The row groups of the file hold 64 rows, thus a filter on one abundance file reads a small part of the
    nuclide lists.

    A cache from a different artistools version or from a different archive is stale, and a new read of the
    archive replaces it. A cache whose archive is absent stays in use. The function checks the cache on each
    call, thus a long process reads a replaced archive again.

    Several processes can read the same trajectory at the same time, thus the write is atomic. If a different
    process wrote a complete cache first, the function keeps that cache.
    """
    parquetpath = get_trajectory_cache_path(traj_root, particleid)
    archivepath = get_trajectory_archive_path(traj_root, particleid)
    archivestat = archivepath.stat() if archivepath is not None else None
    textsource_mtime = archivestat.st_mtime if archivestat is not None else None
    # the identity of the cache that a rewrite replaces, from the same moment as the staleness check
    outdatedparquet = get_file_identity(parquetpath)
    pqmetadata, reason = read_parquet_cache_metadata(
        parquetpath,
        TRAJECTORY_CACHEVERSION,
        textsource_mtime,
        textsource=archivepath.name if archivepath is not None else None,
        textsource_size=archivestat.st_size if archivestat is not None else None,
    )
    if pqmetadata is not None:
        return pl.scan_parquet(parquetpath)

    if archivepath is None:
        msg = f"{traj_root} holds no network data for particle {particleid}"
        if outdatedparquet is not None:
            msg += f" ({parquetpath.name} is stale, because {reason}, and there is no archive to replace it)"
        raise FileNotFoundError(msg)
    assert archivestat is not None

    if outdatedparquet is not None:
        print(f"{parquetpath} is stale, because {reason}. artistools reads {archivepath.name} again.")

    make_cache_folder(parquetpath.parent, traj_root)
    dftrajectory, leftoutfiles = read_trajectory_network_data(archivepath)
    write_parquet_atomic(
        dftrajectory,
        parquetpath,
        replaces=outdatedparquet,
        metadata={
            "cacheversion": str(TRAJECTORY_CACHEVERSION),
            "textsource": archivepath.name,
            "textsource_mtime": str(textsource_mtime),
            "textsource_size": str(archivestat.st_size),
            # a later run must name the files that the cache left out as corrupt, see describe_leftout_files
            "leftoutfiles": ", ".join(leftoutfiles),
        },
        row_group_size=64,
    )
    # a rival process can install a cache of an older archive first, and the write then keeps that file. The
    # table in memory is the current one, thus a cache that fails the check serves from memory for this call
    pqmetadata, reason = read_parquet_cache_metadata(
        parquetpath,
        TRAJECTORY_CACHEVERSION,
        textsource_mtime,
        textsource=archivepath.name,
        textsource_size=archivestat.st_size,
    )
    if pqmetadata is None:
        print(f"{parquetpath} is stale after the write, because {reason}. This call reads the table in memory.")
        return dftrajectory.lazy()

    return pl.scan_parquet(parquetpath)


def get_trajectory_timesteps(traj_root: Path, particleid: int) -> pl.DataFrame:
    """Return the timesteps of a trajectory from its parquet cache, without the nuclide lists.

    A row holds nstep, time_s, abundancefile, abundancetime_s, and the columns of READER_COLUMNS that the
    archive has, in the order of nstep. The readers of one trajectory call this function several times, thus
    an lru_cache holds the result for each state of the archive. A call then costs a stat, and a changed
    archive gives a new read, as the check of the parquet cache on each call needs. A parquet cache of the
    same archive has the same content, thus its state is not part of the key, unless the archive is absent.
    Do not change the dataframe that the function returns.
    """
    archivepath = get_trajectory_archive_path(traj_root, particleid)
    statepath = archivepath if archivepath is not None else get_trajectory_cache_path(traj_root, particleid)
    filestate = get_file_state(statepath) if statepath.is_file() else None
    return collect_trajectory_timesteps(traj_root, particleid, filestate)


@lru_cache(maxsize=64)
def collect_trajectory_timesteps(
    traj_root: Path, particleid: int, filestate: tuple[int, int, int, int] | None
) -> pl.DataFrame:
    """Collect the timesteps for get_trajectory_timesteps. filestate is the key of the cache, see there."""
    del filestate
    return scan_trajectory_network_data(traj_root, particleid).select(cs.exclude("N", "Z", "log10abund")).collect()


def describe_leftout_files(traj_root: Path, particleid: int) -> str:
    """Return a sentence that names the files that the cache of a trajectory left out as corrupt, for an error."""
    try:
        leftoutfiles = pl.read_parquet_metadata(get_trajectory_cache_path(traj_root, particleid)).get("leftoutfiles")
    except (FileNotFoundError, pl.exceptions.PolarsError, OSError):
        leftoutfiles = None
    if leftoutfiles:
        return f"The cache left out these files of the archive as corrupt: {leftoutfiles}."
    return "The archive has no file with it."


def get_network_timesteps_table(traj_root: Path, particleid: int, columns: tuple[str, ...]) -> pl.DataFrame:
    """Return the network timesteps of a trajectory that hold a value in each of the given columns.

    The table holds nstep, time_s, and the given columns, in the order of nstep. A column of heating.dat, e.g.
    "hbeta", keeps the rows of that file, as a read of the file alone did. A table with no row, or with no
    column of the given ones, raises FileNotFoundError, because the callers leave out a trajectory with no
    network data.
    """
    dftimesteps = get_trajectory_timesteps(traj_root, particleid)
    missingcol = next((col for col in columns if col not in dftimesteps.columns), None)
    if missingcol is not None:
        # the callers leave out a trajectory with no network data, and a table that is absent, or that the cache
        # left out as corrupt, is the same case for a reader of its columns
        msg = (
            f"The network data of particle {particleid} in {traj_root} has no column {missingcol}."
            f" {describe_leftout_files(traj_root, particleid)}"
        )
        raise FileNotFoundError(msg)

    dftable = dftimesteps.filter(pl.all_horizontal(pl.col("nstep", "time_s", *columns).is_not_null())).select(
        "nstep", "time_s", *columns
    )
    if dftable.is_empty():
        # a reader of these columns has no network data, e.g. a heating.dat with a header alone
        msg = f"The network data of particle {particleid} in {traj_root} has no timestep with a value in {columns}"
        raise FileNotFoundError(msg)

    return dftable


def get_traj_network_timesteps(traj_root: Path | str, particleid: int) -> pl.DataFrame:
    """Return the numbers and the times [s] of the network timesteps that have an abundance file.

    The search for the closest timestep reads the abundance file of its result, thus a timestep with no file,
    e.g. a file that the cache left out as corrupt, must not enter the search.
    """
    return (
        get_trajectory_timesteps(Path(traj_root), particleid)
        .filter(pl.col("nstep").is_not_null() & pl.col("abundancefile").is_not_null())
        .select(pl.col("nstep"), pl.col("time_s").cast(pl.Float32).alias("timesec"))
    )


def get_closest_network_timesteps(
    traj_root: Path,
    particleid: int,
    timesec: float | Sequence[float] | npt.NDArray[np.floating],
    cond: t.Literal["lessorequal", "greaterorequal", "nearest"] = "nearest",
) -> list[int | None]:
    """Find the closest network timestep to a given time in seconds.

    cond:
      - 'lessorequal': find the highest timestep at or before the time
      - 'greaterorequal': find the lowest timestep at or after the time.

    A time that equals the time of a timestep gives that timestep for both conditions. For 'lessorequal' and
    'greaterorequal', a time with no network timestep on that side gives None.
    """
    dfevol = get_traj_network_timesteps(traj_root, particleid).unique(subset=["timesec"], keep="first").sort("timesec")
    # the times of the file are Float32, thus a time compares in Float32 as the polars filters did before
    arrtimesec = dfevol["timesec"].to_numpy()
    arrnstep: list[int] = dfevol["nstep"].to_list()
    arrtime_wanted = np.atleast_1d(np.asarray(timesec, dtype=np.float32))
    # one binary search for all the times replaces one polars query for each time
    if cond == "lessorequal":
        indices = np.searchsorted(arrtimesec, arrtime_wanted, side="right") - 1
        return [arrnstep[index] if index >= 0 else None for index in indices]

    if cond == "greaterorequal":
        indices = np.searchsorted(arrtimesec, arrtime_wanted, side="left")
        return [arrnstep[index] if index < len(arrnstep) else None for index in indices]

    if cond == "nearest":
        if not arrnstep:
            msg = f"The trajectory of particle {particleid} has no network timestep with an abundance file"
            raise FileNotFoundError(msg)
        upper = np.clip(np.searchsorted(arrtimesec, arrtime_wanted, side="left"), 0, len(arrnstep) - 1)
        lower = np.clip(upper - 1, 0, len(arrnstep) - 1)
        islowercloser = np.abs(arrtime_wanted - arrtimesec[lower]) <= np.abs(arrtimesec[upper] - arrtime_wanted)
        return [arrnstep[index] for index in np.where(islowercloser, lower, upper)]

    msg = f"Unknown cond value {cond}"
    raise AssertionError(msg)


def check_traj_time_matches(
    particleid: int, traj_time_s: float, target_time_s: float, rel_tol: float = 0.2, abs_tol: float = 1.0
) -> None:
    """Raise when a network step time is not the time that the caller asked for.

    The default tolerance is wide, because a network step lands near a target time rather than on it. A
    caller that already knows the exact step index passes a tight tolerance instead.
    """
    if not math.isclose(traj_time_s, target_time_s, rel_tol=rel_tol, abs_tol=abs_tol):
        msg = (
            f"ERROR: particle {particleid} step time of {traj_time_s} is not similar to target {target_time_s} seconds"
        )
        raise AssertionError(msg)


def get_trajectory_timestepfiles_nuc_abund(
    traj_root: Path | str, particleid: int, abundancefiles: Sequence[str]
) -> tuple[pl.DataFrame, list[float]]:
    """Get the nuclear abundances and the time of several abundance files of one trajectory.

    abundancefiles names the files, e.g. "nz-plane00200" or "tday_nz-plane". The column "fileindex" gives the
    position of the file of each row in abundancefiles.
    """
    abundancefiles = list(abundancefiles)
    dfrows = pl.DataFrame(
        {"abundancefile": abundancefiles, "fileindex": range(len(abundancefiles))},
        schema={"abundancefile": pl.String, "fileindex": pl.UInt32},
    ).join(
        scan_trajectory_network_data(Path(traj_root), particleid)
        .filter(pl.col("abundancefile").is_in(abundancefiles))
        .select("abundancefile", "abundancetime_s", "N", "Z", "log10abund")
        .collect(),
        on="abundancefile",
        how="left",
        maintain_order="left",
    )
    missingfiles = dfrows.filter(pl.col("abundancetime_s").is_null())["abundancefile"].to_list()
    if missingfiles:
        msg = (
            f"The trajectory of particle {particleid} has no abundance file {missingfiles[0]}."
            f" {describe_leftout_files(Path(traj_root), particleid)}"
        )
        # the other readers of a missing trajectory file raise FileNotFoundError, and the callers catch that type
        raise FileNotFoundError(msg)

    dfnucabund = (
        dfrows
        .select("fileindex", "N", "Z", "log10abund")
        .explode("N", "Z", "log10abund")
        .with_columns(massfrac=(pl.col("N") + pl.col("Z")) * (10 ** pl.col("log10abund")))
        .drop("log10abund")
    )

    return dfnucabund, dfrows["abundancetime_s"].to_list()


def get_trajectory_timestepfile_nuc_abund(
    traj_root: Path | str, particleid: int, abundancefile: str
) -> tuple[pl.DataFrame, float]:
    """Return the nuclear abundances of a trajectory at one time, and that time in seconds.

    abundancefile names the file of that time in the trajectory, e.g. "tday_nz-plane".
    """
    dfnucabund, timesecs = get_trajectory_timestepfiles_nuc_abund(traj_root, particleid, [abundancefile])
    return dfnucabund.drop("fileindex"), timesecs[0]


def get_trajectory_qdotintegral(particleid: int, traj_root: Path, nts_max: int, t_model_s: float) -> float:
    """Calculate initial cell energy [erg/g] from reactions t < t_model_s (reduced by work done).

    nts_max is the one-based #count of the last network timestep. The integration starts at one second, thus
    a nts_max before one second gives zero. An energy_thermo.dat that ends before nts_max gives
    FileNotFoundError, because the trajectory has no network data at nts_max. Each timestep from the first
    one at or after one second up to nts_max must have a Qdot value, because a trapezoid across a missing
    timestep would give an integral with no sign of the problem.
    """
    dftimesteps = get_trajectory_timesteps(Path(traj_root), particleid).filter(pl.col("nstep").is_not_null())
    if "Qdot" not in dftimesteps.columns:
        msg = (
            f"The network data of particle {particleid} in {traj_root} has no column Qdot."
            f" {describe_leftout_files(Path(traj_root), particleid)}"
        )
        raise FileNotFoundError(msg)

    dfrange = dftimesteps.filter((pl.col("time_s") >= 1.0) & (pl.col("nstep") <= nts_max))
    if dfrange.is_empty():
        return 0.0

    qdotsteps = dftimesteps.filter(pl.col("Qdot").is_not_null())["nstep"].to_list()
    lastqdotstep = qdotsteps[-1] if qdotsteps else None
    if lastqdotstep is None or lastqdotstep < nts_max:
        msg = (
            f"The energy_thermo.dat of particle {particleid} ends at timestep {lastqdotstep} before timestep {nts_max}"
        )
        raise FileNotFoundError(msg)

    dfthermo = dfrange.filter(pl.col("Qdot").is_not_null())
    # the range starts at the first timestep at or after one second, which can have no Qdot value
    if dfthermo["nstep"].to_list() != list(range(dfrange["nstep"].to_list()[0], nts_max + 1)):
        msg = f"The energy_thermo.dat of particle {particleid} has no Qdot value at each timestep up to {nts_max}"
        raise ValueError(msg)
    assert all(dfthermo["Qdot"] >= 0.0)

    dfthermo = dfthermo.with_columns(Qdot_expansionadjusted=pl.col("Qdot") * pl.col("time_s") / t_model_s)

    qdotintegral = float(np.trapezoid(y=dfthermo["Qdot_expansionadjusted"], x=dfthermo["time_s"]))
    assert qdotintegral >= 0.0

    return qdotintegral


def get_trajectory_abund_q(particleid: int, traj_root: Path, t_model_s: float) -> dict[tuple[int, int] | str, float]:
    """Return the nuclear mass fractions of a particle at a given time, and its integral of Qdot under the key "q"."""
    # find the closest timestep to the required time
    try:
        nts = get_closest_network_timesteps(traj_root, particleid, [t_model_s])[0]
    except FileNotFoundError:
        return {}
    # the nearest timestep always exists, thus only "lessorequal" and "greaterorequal" can give None
    assert nts is not None
    abundancefile = f"nz-plane{nts:05d}"

    try:
        dftrajnucabund, traj_time_s = get_trajectory_timestepfile_nuc_abund(traj_root, particleid, abundancefile)

    except FileNotFoundError:
        return {}

    massfractotal = dftrajnucabund["massfrac"].sum()
    dftrajnucabund = dftrajnucabund.filter(pl.col("Z") >= 1)

    assert math.isclose(massfractotal, 1.0, rel_tol=0.02)
    check_traj_time_matches(particleid, traj_time_s, t_model_s)

    dict_traj_nuc_abund: dict[tuple[int, int] | str, float] = {
        (Z, N): massfrac / massfractotal for Z, N, massfrac in dftrajnucabund[["Z", "N", "massfrac"]].iter_rows()
    }

    # set the cell energy at model time [erg/g]. A particle with no usable energy_thermo.dat has no network data
    try:
        dict_traj_nuc_abund["q"] = get_trajectory_qdotintegral(
            particleid=particleid, traj_root=traj_root, nts_max=nts, t_model_s=t_model_s
        )
    except FileNotFoundError:
        return {}

    return dict_traj_nuc_abund


def get_gridparticlecontributions_or_none(gridcontribpath: Path | str) -> pl.DataFrame | None:
    """Return the particle contributions, or None when the grid folder holds no contributions file.

    A caller that tested for the plain file itself could not see a compressed one, and so skipped the
    loader that would have found it.
    """
    contribfile = firstexisting_or_none("gridcontributions.txt", folder=gridcontribpath, tryzipped=True)

    return None if contribfile is None else get_gridparticlecontributions(gridcontribpath)


def get_gridparticlecontributions(gridcontribpath: Path | str) -> pl.DataFrame:
    """Return the mass fraction that each trajectory particle contributes to each grid cell."""
    return pl.read_csv(
        polars_source(firstexisting("gridcontributions.txt", folder=gridcontribpath, tryzipped=True)),
        has_header=True,
        separator=" ",
        schema_overrides={
            "particleid": pl.Int32,
            "cellindex": pl.Int32,
            "frac_of_cellmass": pl.Float64,
            "frac_of_cellmass_includemissing": pl.Float64,
        },
    )


def filtermissinggridparticlecontributions(dfcontribs: pl.DataFrame, missing_particleids: list[int]) -> pl.DataFrame:
    """Zero the contributions of particles that have no abundance data and renormalise the rest per cell."""
    print(
        f"Adding gridcontributions column that excludes {len(missing_particleids)} "
        "particles without abundance data and renormalising...",
        end="",
    )
    # after filtering, frac_of_cellmass_includemissing will still include particles with rho but no abundance data
    # frac_of_cellmass will exclude particles with no abundances
    dfcontribs = dfcontribs.with_columns(frac_of_cellmass_includemissing=pl.col("frac_of_cellmass")).with_columns(
        frac_of_cellmass=pl
        .when(pl.col("particleid").is_in(missing_particleids))
        .then(0.0)
        .otherwise(pl.col("frac_of_cellmass"))
    )

    dfcontribs = (
        dfcontribs
        .lazy()
        .with_columns([
            pl
            .when(pl.col(fraccol).sum().over("cellindex") > 0.0)
            .then(pl.col(fraccol) / pl.col(fraccol).sum().over("cellindex"))
            .otherwise(0.0)
            .alias(fraccol)
            for fraccol in ("frac_of_cellmass", "frac_of_cellmass_includemissing")
        ])
        .collect()
    )

    # the fractions of each cell sum to one, or to zero for a cell without particles
    fraccols = ["frac_of_cellmass", "frac_of_cellmass_includemissing"]
    cellsums_ok = (
        dfcontribs
        .group_by("cellindex")
        .agg(pl.col(fraccols).sum())
        .select(
            (
                (pl.col(col) == 0.0) | ((pl.col(col) - 1.0).abs() <= 0.02 * pl.max_horizontal(pl.col(col).abs(), 1.0))
            ).all()
            for col in fraccols
        )
        .row(0)
    )
    assert all(cellsums_ok), "the particle contributions of a cell do not sum to one"

    print("Done")

    return dfcontribs


def save_gridparticlecontributions(
    dfcontribs: pl.DataFrame, gridcontribpath: Path | str, float_precision: int | None = 7
) -> Path:
    """Write gridcontributions.txt.zst, delete each other copy of that file, and return its path.

    float_precision=None writes the shortest text that gives each float again, e.g. 1.0.
    """
    gridcontribpath = Path(gridcontribpath)
    if gridcontribpath.is_dir():
        gridcontribpath /= "gridcontributions.txt"
    gridcontribpath = prepare_zstd_output(gridcontribpath)

    dfcontribs.write_csv(
        gridcontribpath,
        separator=" ",
        float_scientific=True if float_precision is not None else None,
        float_precision=float_precision,
        compression="zstd",
    )
    return gridcontribpath


def get_dfnucabundances(
    dfcontribs: pl.DataFrame,
    particleids: Iterable[int],
    list_traj_nuc_abund: Sequence[dict[tuple[int, int] | str, float]],
) -> pl.DataFrame:
    """Return the mass fractions of each cell, which are the particle mass fractions weighted by frac_of_cellmass.

    A join of all the contributions with all the nuclide columns has 5e5 rows and 2473 columns (10 GB) for a 50^3
    grid. Thus each query joins a batch of 64 columns. The queries read tables in memory, thus none repeats a scan.
    """
    colname_of_key = {
        key: key if isinstance(key, str) else f"X_{get_elsymbol(key[0])}{key[0] + key[1]}"
        for key in dict.fromkeys(chain.from_iterable(list_traj_nuc_abund))
    }
    # one column for each nuclide and for the "q" energy, thus each batch query can select its columns
    dfparticlenucabund = (
        pl
        .DataFrame(
            {
                "particleid": [
                    particleid
                    for particleid, traj_nuc_abund in zip(particleids, list_traj_nuc_abund, strict=True)
                    for _ in traj_nuc_abund
                ],
                "colname": [colname_of_key[key] for traj_nuc_abund in list_traj_nuc_abund for key in traj_nuc_abund],
                "massfrac": [value for traj_nuc_abund in list_traj_nuc_abund for value in traj_nuc_abund.values()],
            },
            schema={"particleid": dfcontribs.schema["particleid"], "colname": pl.String, "massfrac": pl.Float64},
        )
        .pivot(on="colname", index="particleid", values="massfrac")
        .with_columns(cs.float().fill_null(0.0))
    )

    dfbatches = [
        dfcontribs
        .lazy()
        .select("particleid", "cellindex", "frac_of_cellmass")
        .join(dfparticlenucabund.lazy().select("particleid", *batchcolnames), on="particleid", how="left")
        .group_by("cellindex")
        .agg((pl.col(batchcolnames) * pl.col("frac_of_cellmass")).sum())
        .sort("cellindex")
        .collect()
        for batchcolnames in batched(colname_of_key.values(), 64, strict=False)
    ]
    return pl.DataFrame([
        dfbatches[0]["cellindex"].cast(pl.Int32).alias("inputcellid"),
        *(column for dfbatch in dfbatches for column in dfbatch.drop("cellindex").iter_columns()),
    ])


def add_abundancecontributions(
    dfgridcontributions: pl.DataFrame,
    dfmodel: pl.LazyFrame | pl.DataFrame,
    t_model_days_incpremerger: float,
    traj_root: Path | str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Add the abundances of the network calculations of the trajectories to the abundances of the model cells.

    Return dfmodel, dfelabundances, and dfcontribs.
    """
    t_model_s = t_model_days_incpremerger * day_to_s
    dfcontribs = dfgridcontributions

    dfmodel = dfmodel.lazy().collect()
    if "X_Fegroup" not in dfmodel.columns:
        dfmodel = dfmodel.with_columns(pl.lit(1.0).alias("X_Fegroup"))

    particleids = dfcontribs.select(pl.col("particleid").unique()).to_series()

    print("Reading trajectory abundances...")
    timestart = time.perf_counter()
    trajworker = partial(get_trajectory_abund_q, t_model_s=t_model_s, traj_root=Path(traj_root))

    list_traj_nuc_abund = parallel_map(trajworker, particleids, chunksize=16)

    missing_particle_ids = [
        particleid for particleid, df in zip(particleids, list_traj_nuc_abund, strict=True) if not df
    ]
    dfcontribs = filtermissinggridparticlecontributions(dfcontribs, missing_particle_ids).sort("particleid")
    active_inputcellcount = dfcontribs["cellindex"].n_unique()

    print(
        f"{active_inputcellcount} of {len(dfmodel)} model cells have >0 particles contributing "
        f"({len(dfcontribs)} cell contributions from {len(particleids)} particles)"
    )
    n_missing_particles = len(missing_particle_ids)
    print(f"  {n_missing_particles} particles are missing network abundance data out of {len(particleids)}")

    assert len(particleids) > n_missing_particles

    print(f"Reading trajectory abundances took {time.perf_counter() - timestart:.1f} seconds")

    timestart = time.perf_counter()
    print("Creating dfnucabundances...", end="", flush=True)
    dfnucabundances = get_dfnucabundances(dfcontribs, particleids, list_traj_nuc_abund)
    del list_traj_nuc_abund
    gc.collect()
    print(f" Took {time.perf_counter() - timestart:.1f} seconds")

    timestart = time.perf_counter()
    print("Merging isotopic abundances into dfmodel...", end="", flush=True)
    # The trajectories give q at t_model, and an old grid.dat can also give q, at the end of the hydrodynamics run.
    # The value of the trajectories replaces each column of the model that they also give. A join that keeps both
    # columns names the second one q_right, which no writer and no dimension reduction reads
    replacedcols = [col for col in dfnucabundances.columns if col != "inputcellid" and col in dfmodel.columns]
    if replacedcols:
        print(f" The trajectory values replace the model columns {replacedcols}...", end="", flush=True)
    dfmodel = (
        dfmodel
        .drop(replacedcols)
        .join(dfnucabundances, how="left", on="inputcellid", maintain_order="left")
        .fill_null(0)
    )
    print(f" Took {time.perf_counter() - timestart:.1f} seconds")

    return dfmodel, get_dfelemabund_from_dfmodel(dfmodel), dfcontribs


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    # the defaults are the trajectory of the first use of this command
    parser.add_argument(
        "-trajectoryroot",
        "-trajroot",
        type=Path,
        default=Path.home()
        / "Google Drive/Shared Drives/GSI NSM/Mergers/SFHo_long/Trajectory_SFHo_long-radius-entropy",
        help="The folder of the nuclear network trajectories",
    )
    parser.add_argument("-particleid", type=int, default=133371, help="Particle id of the trajectory")
    addarg_output(parser, kind="folder", default=Path())
    addarg_force(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create ARTIS model from single trajectory abundances."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    confirm_overwrite([args.outputfile], MODEL_FILE_NAMES, force=args.force)

    traj_root = Path(args.trajectoryroot)
    particleid = args.particleid
    print(f"Trajectory particle id {particleid}")
    dfnucabund, t_model_init_seconds = get_trajectory_timestepfile_nuc_abund(traj_root, particleid, "tday_nz-plane")
    dfnucabund = dfnucabund.filter(pl.col("Z") >= 1).with_columns(radioactive=pl.lit(value=True))

    t_model_init_days = t_model_init_seconds / day_to_s

    wollaeger_profilename = "wollaeger_ejectaprofile_10bins.txt"
    if Path(wollaeger_profilename).exists():
        dfdensities = get_wollaeger_density_profile(wollaeger_profilename, t_model_init_seconds)
    else:
        rho = 1e-11
        print(f"{wollaeger_profilename} not found. Using rho {rho} g/cm³")
        dfdensities = pl.DataFrame({"mgi": [0], "rho": [rho], "vel_r_max_kmps": [6.0e4]})

    dfdensities = dfdensities.with_columns(inputcellid=pl.col("mgi") + 1)

    # write abundances.txt
    dictelemabund = get_elemabund_from_nucabund(dfnucabund)

    dfelabundances = pl.DataFrame([dictelemabund | {"inputcellid": mgi + 1} for mgi in range(len(dfdensities))])
    save_initelemabundances(dfelabundances=dfelabundances, outpath=args.outputfile)

    # write model.txt

    rowdict = {
        "X_Fegroup": 1.0,
        "X_Ni56": 0.0,
        "X_Co56": 0.0,
        "X_Fe52": 0.0,
        "X_Cr48": 0.0,
        "X_Ni57": 0.0,
        "X_Co57": 0.0,
    }

    for row in dfnucabund.filter(pl.col("radioactive")).iter_rows(named=True):
        A = row["N"] + row["Z"]
        rowdict[f"X_{get_elsymbol(row['Z'])}{A}"] = row["massfrac"]

    dfmodel = pl.DataFrame(
        [
            {
                "inputcellid": densityrow["inputcellid"],
                "vel_r_max_kmps": densityrow["vel_r_max_kmps"],
                "logrho": math.log10(densityrow["rho"]),
            }
            | rowdict
            for densityrow in dfdensities.iter_rows(named=True)
        ],
        orient="row",
    )
    save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, outpath=Path(args.outputfile))

    save_gridparticlecontributions(
        dfmodel.select(
            pl.lit(particleid).alias("particleid"),
            pl.col("inputcellid").alias("cellindex"),
            frac_of_cellmass=pl.lit(1.0),
        ),
        Path(args.outputfile, "gridcontributions.txt"),
        float_precision=None,
    )


def get_wollaeger_density_profile(wollaeger_profilename: Path | str, t_model_init_seconds: float) -> pl.DataFrame:
    """Read a Wollaeger density profile and expand it homologously to t_model_init_seconds."""
    wollaeger_profilename = Path(wollaeger_profilename)
    print(f"{wollaeger_profilename} found")
    with Path(wollaeger_profilename).open("rt", encoding="utf-8") as f:
        t_model_init_days_in = float(f.readline().strip().removesuffix(" day"))
    t_model_init_seconds_in = t_model_init_days_in * day_to_s

    return (
        read_wsv(wollaeger_profilename, has_header=False, skip_rows=1, new_columns=["mgi", "vel_r_max_kmps", "rho"])
        .with_columns(pl.col("mgi").cast(pl.Int32))
        .with_columns(vel_r_min_kmps=pl.col("vel_r_max_kmps").shift(n=1, fill_value=0.0))
        .with_columns(
            mass_g=pl.col("rho")
            * 4.0
            / 3.0
            * math.pi
            * (pl.col("vel_r_max_kmps") ** 3 - pl.col("vel_r_min_kmps") ** 3)
            * (km_to_cm * t_model_init_seconds_in) ** 3
        )
        .with_columns(
            # now replace the density at the input time with the density at required time
            rho=pl.col("mass_g")
            / (
                4.0
                / 3.0
                * math.pi
                * (pl.col("vel_r_max_kmps") ** 3 - pl.col("vel_r_min_kmps") ** 3)
                * (km_to_cm * t_model_init_seconds) ** 3
            )
        )
    )
