"""Read the ARTIS atomic data files, and convert between the names of elements, ions, and nuclides."""

import math
import operator
import re
import string
import time
import typing as t
from collections.abc import Collection
from collections.abc import Generator
from collections.abc import Mapping
from collections.abc import Sequence
from functools import lru_cache
from functools import partial
from pathlib import Path
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import polars as pl
from polars import selectors as cs

from artistools import misc
from artistools.commands import get_path
from artistools.constants import hc_in_ev_angstrom
from artistools.constants import K_B_ev_per_K
from artistools.misc import firstexisting_or_none
from artistools.misc import polars_source
from artistools.misc.fileio import firstexisting
from artistools.misc.fileio import get_file_identity
from artistools.misc.fileio import modelpath_cache
from artistools.misc.fileio import polars_source_open
from artistools.misc.fileio import read_parquet_cache_metadata
from artistools.misc.fileio import readnoncommentline
from artistools.misc.fileio import resolve_modelpath
from artistools.misc.fileio import write_parquet_atomic
from artistools.misc.fileio import zopen
from artistools.misc.remote import on_model_host

if t.TYPE_CHECKING:
    from collections.abc import Callable

# The version of the line list parquet cache format. Increase it for a change that makes an older
# cache file incorrect, e.g. a new column, a removed column, or a different data type.
LINELIST_CACHEVERSION = 1


def parse_adata(
    fadata: t.IO[str],
    phixsdict: dict[tuple[int, int, int], tuple[npt.NDArray[np.void], npt.NDArray[np.void]]],
    ionlist: Collection[tuple[int, int]] | None,
    firstlevelnumber: int = 1,
) -> Generator[tuple[int, int, int, float, pl.DataFrame]]:
    """Generate ions and their level lists from adata.txt.

    ARTIS takes the number of the first level, 0 or 1, from the first level line of adata.txt. The same numbering
    applies to transitiondata.txt and phixsdata_v2.txt. firstlevelnumber is the numbering that the caller used for
    phixsdict and for the transitions. A file with a different numbering raises ValueError, because the level indices
    of the other files would then be wrong.
    """
    for line in fadata:
        # artisatomic writes a block of comment lines before the header line of each ion
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        ionheader = line.split()
        Z = int(ionheader[0])
        ion_stage = int(ionheader[1])
        level_count = int(ionheader[2])

        if not ionlist or (Z, ion_stage) in ionlist:
            level_list: list[
                tuple[float, float, int, str | None, npt.NDArray[np.void] | None, npt.NDArray[np.void] | None]
            ] = []
            for levelindex in range(level_count):
                strlevelnumber, strenergy_ev, strg, tail = read_level_line(fadata, Z, ion_stage).split(maxsplit=3)
                strtransition_count, _, namefield = tail.partition(" ")

                inputlevelnumber = int(strlevelnumber)
                if inputlevelnumber != levelindex + firstlevelnumber:
                    levelstr = "the first level" if levelindex == 0 else f"level {levelindex + 1}"
                    msg = (
                        f"adata.txt: Z={Z} ion_stage={ion_stage}: {levelstr} of the ion has the number"
                        f" {inputlevelnumber}, but the numbering from {firstlevelnumber} gives it the number"
                        f" {levelindex + firstlevelnumber}"
                    )
                    raise ValueError(msg)
                # parse_phixsdata() keys on the zero-based level index, so look up levelindex and not the
                # one-based inputlevelnumber, which would attach each level the next level's cross-sections
                phixstargetlist, phixstable = phixsdict.get((Z, ion_stage, levelindex), (None, None))

                # artisatomic writes the name of the level in a field of a fixed width, then a comment.
                # A level name holds no space, thus the first word of the field is the name. A level
                # with no name leaves the field blank. Such a field starts with a space
                levelname = "" if (not namefield or namefield[0].isspace()) else namefield.split(maxsplit=1)[0]

                level_list.append((
                    float(strenergy_ev),
                    float(strg),
                    int(strtransition_count),
                    levelname,
                    phixstargetlist,
                    phixstable,
                ))

            colnames = ["energy_ev", "g", "transition_count", "levelname", "phixstargetlist", "phixstable"]
            # transpose to columns, because a row-oriented constructor unpacks each structured numpy array into a
            # list of numpy scalars instead of storing the array itself in the object column
            dflevels = (
                pl
                .DataFrame(
                    dict(zip(colnames, zip(*level_list, strict=True), strict=True)) if level_list else {},
                    schema={
                        "energy_ev": pl.Float64,
                        "g": pl.Float32,
                        "transition_count": pl.Int32,
                        "levelname": pl.String,
                        "phixstargetlist": pl.Object,
                        "phixstable": pl.Object,
                    },
                )
                .with_row_index("levelindex")
                .with_columns(pl.col("levelindex").cast(pl.Int32))
            )

            ionisation_energy_ev = float(ionheader[3])
            yield Z, ion_stage, level_count, ionisation_energy_ev, dflevels

        else:
            for _ in range(level_count):
                read_level_line(fadata, Z, ion_stage)


def read_level_line(fadata: t.IO[str], atomic_number: int, ion_stage: int) -> str:
    """Return the next level line of adata.txt, or raise ValueError at the end of the file.

    A file that a copy cut short otherwise lost the ions after the cut with no message.
    """
    if line := fadata.readline():
        return line

    msg = f"adata.txt ends inside the levels of Z={atomic_number} ion_stage={ion_stage}. The file is not complete"
    raise ValueError(msg)


def get_first_level_number(modelpath: Path | str) -> int:
    """Return the number of the first level of adata.txt, 0 or 1, or 1 if the model has no adata.txt.

    ARTIS takes this number from the first level line of adata.txt, and it applies the same numbering to
    transitiondata.txt and phixsdata_v2.txt.
    """
    adatafilename = Path(modelpath, "adata.txt")
    try:
        with zopen(adatafilename, encoding="utf-8") as fadata:
            # the first line that is not blank and not a comment is the header line of the first ion
            readnoncommentline(fadata)
            firstlevelline = readnoncommentline(fadata)
    except (FileNotFoundError, EOFError):
        return 1

    firstlevelnumber = int(firstlevelline.split()[0])
    if firstlevelnumber not in {0, 1}:
        msg = f"{adatafilename}: the first level has the number {firstlevelnumber}, but ARTIS numbers the levels from 0 or 1"
        raise ValueError(msg)

    return firstlevelnumber


def parse_phixsdata(
    phixs_filename: Path | str, ionlist: Collection[tuple[int, int]] | None = None, firstlevelnumber: int = 1
) -> dict[tuple[int, int, int], tuple[npt.NDArray[np.void], npt.NDArray[np.void]]]:
    """Return the photoionisation cross section tables of phixsdata_v2.txt, keyed by (Z, ion stage, level).

    The level numbers of the file start at firstlevelnumber, which ARTIS takes from adata.txt. The keys and the
    target levels are zero-based level indices.
    """
    phixsdict: dict[tuple[int, int, int], tuple[npt.NDArray[np.void], npt.NDArray[np.void]]] = {}
    with misc.zopen(phixs_filename) as fphixs:
        nphixspoints = int(fphixs.readline())
        phixsnuincrement = float(fphixs.readline())
        xgrid = np.linspace(
            1.0, 1.0 + phixsnuincrement * nphixspoints, num=nphixspoints, endpoint=False, dtype=np.float64
        )
        for line in fphixs:
            # artisatomic writes a block of comment lines before the first table of each ion
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            ionheader = line.split()
            Z = int(ionheader[0])
            upperion_stage = int(ionheader[1])
            upperionlevelnumber = int(ionheader[2])
            lowerion_stage = int(ionheader[3])
            lowerionlevel = int(ionheader[4]) - firstlevelnumber

            assert upperion_stage == lowerion_stage + 1

            nptargetlist: npt.NDArray[np.void]
            # as in ARTIS, a level number of zero or more names the one target level. A negative number, e.g. -1, says
            # that a list of the target levels follows
            if upperionlevelnumber >= 0:
                upperionlevel = upperionlevelnumber - firstlevelnumber
                nptargetlist = np.array([(upperionlevel, 1.0)], dtype=[("level", np.int32), ("fraction", np.float32)])
            else:
                ntargets = int(fphixs.readline())
                # one structured entry per target, matching the shape of the single-target case above
                nptargetlist = np.empty(ntargets, dtype=[("level", np.int32), ("fraction", np.float32)])
                for phixstargetindex in range(ntargets):
                    level, fraction = fphixs.readline().split()
                    nptargetlist[phixstargetindex] = (int(level) - firstlevelnumber, float(fraction))

            if not ionlist or (Z, lowerion_stage) in ionlist:
                phixslist = [float(fphixs.readline()) * 1e-18 for _ in range(nphixspoints)]
                phixstable = np.array(
                    list(zip(xgrid, phixslist, strict=True)), dtype=[("x", np.float64), ("sigma_cm2", np.float32)]
                )

                phixsdict[Z, lowerion_stage, lowerionlevel] = (nptargetlist, phixstable)

            else:
                for _ in range(nphixspoints):
                    fphixs.readline()

    return phixsdict


def add_transition_columns(
    dftransitions: pl.LazyFrame | pl.DataFrame, dflevels: pl.DataFrame | pl.LazyFrame, columns: Sequence[str]
) -> pl.LazyFrame:
    """Add columns to a polars DataFrame of transitions."""
    dftransitions = dftransitions.lazy()
    columns_before = dftransitions.collect_schema().names()

    dflevels = dflevels.select(["g", "energy_ev", "levelname", "levelindex"]).lazy()

    dftransitions = (
        dftransitions
        .join(
            dflevels.select(
                lower="levelindex",
                lower_g=pl.col("g"),
                lower_energy_ev=pl.col("energy_ev"),
                lower_level=pl.col("levelname"),
            ),
            how="left",
            on="lower",
            maintain_order="left",
        )
        .join(
            dflevels.select(
                upper="levelindex",
                upper_g=pl.col("g"),
                upper_energy_ev=pl.col("energy_ev"),
                upper_level=pl.col("levelname"),
            ),
            how="left",
            on="upper",
            maintain_order="left",
        )
        .with_columns(epsilon_trans_ev=(pl.col("upper_energy_ev") - pl.col("lower_energy_ev")))
    )

    dftransitions = dftransitions.with_columns(lambda_angstroms=hc_in_ev_angstrom / pl.col("epsilon_trans_ev"))

    # clean up any columns used for intermediate calculations
    dftransitions = dftransitions.drop(
        col
        for col in dftransitions.collect_schema().names()
        if col not in columns_before and col not in columns and col != "levelindex"
    )

    columns_after = dftransitions.collect_schema().names()
    for col in columns:
        assert col in columns_after, f"Invalid column name {col}"

    return dftransitions


# the fields of a line of a transition table. The key is the count of leading numbers in the first line of the table.
# ARTIS reads 5 numbers, or 4 numbers in the legacy format, which has a transition index and no collision strength
TRANSITION_FIELDS: t.Final = MappingProxyType({
    5: (
        ("int", "a lower level"),
        ("int", "an upper level"),
        ("float", "an A value"),
        ("float", "a collision strength"),
        ("int", "a forbidden flag"),
    ),
    4: (("int", "a transition index"), ("int", "a lower level"), ("int", "an upper level"), ("float", "an A value")),
})
ION_HEADER_FIELDS: t.Final = (("int", "an atomic number"), ("int", "an ion stage"), ("count", "a transition count"))
TRANSITION_SCHEMA: t.Final = MappingProxyType({
    "lower": pl.Int32,
    "upper": pl.Int32,
    "A": pl.Float32,
    "collstr": pl.Float32,
    "forbidden": pl.Int32,
})
INTEGER_PATTERN: t.Final = re.compile(r"[+-]?\d+")


def count_leading_numbers(line: str) -> int:
    """Return the count of the finite numbers at the start of a line, as ARTIS counts them in a table."""
    count = 0
    for token in line.split():
        try:
            if not math.isfinite(float(token)):
                break
        except ValueError:
            break
        count += 1
    return count


def get_field_error(line: str, fields: Sequence[tuple[str, str]]) -> str | None:
    """Return the message for the first field of a line that does not parse, or None if each field parses."""
    tokens = line.split()
    for index, (kind, description) in enumerate(fields):
        if index >= len(tokens):
            return f"line ended where {description} was expected"
        token = tokens[index]
        if kind == "float":
            try:
                float(token)
            except ValueError:
                return f'could not parse "{token}" as {description}'
        elif not INTEGER_PATTERN.fullmatch(token) or (kind == "count" and token.startswith("-")):
            return f'could not parse "{token}" as {description}'
    return None


def get_format_error(line: str) -> str:
    """Return the message for the first line of a table that has neither format of ARTIS."""
    return (
        f"the first line of a table has {count_leading_numbers(line)} numbers, but ARTIS reads 5 numbers (lower upper"
        " A collstr forbidden) or 4 numbers (index lower upper A)"
    )


def read_transitiondata(
    filepath: Path, ionlist: Collection[tuple[int, int]] | None = None, firstlevelnumber: int = 1
) -> dict[tuple[int, int], pl.DataFrame]:
    """Read the transition table of each (atomic_number, ion_stage) from an ARTIS transitiondata.txt file.

    The header line of an ion gives the atomic number, the ion stage, and the count of the lines of its table. Blank
    lines and comment lines can stand between two tables. ARTIS takes the format of a table from the count of the
    numbers in its first line, see TRANSITION_FIELDS. The level numbers of the file start at firstlevelnumber, and
    the frames give zero-based level indices. An error names the file and the first line that does not parse. A
    table that the file ends inside is an error, because a cut compressed file decodes with no error.
    """
    if firstlevelnumber not in {0, 1}:
        msg = f"ARTIS numbers the levels from 0 or from 1, not from {firstlevelnumber}"
        raise ValueError(msg)

    def token(index: int) -> pl.Expr:
        return pl.col("tokens").list.get(index, null_on_oob=True)

    # the query gives only numbers, thus the streaming engine drops the text of each line after the casts. Row i holds
    # line i + 1 of the file
    with polars_source_open(filepath) as source:
        dflines = (
            pl
            .scan_lines(source)
            .select(tokens=pl.col("line").str.extract_all(r"\S+"))
            .select(
                iscontent=token(0).is_not_null() & token(0).str.starts_with("#").not_(),
                tokencount=pl.col("tokens").list.len(),
                int0=token(0).cast(pl.Int32, strict=False),
                int1=token(1).cast(pl.Int32, strict=False),
                int2=token(2).cast(pl.Int32, strict=False),
                float2=token(2).cast(pl.Float32, strict=False),
                float3=token(3).cast(pl.Float32, strict=False),
                int4=token(4).cast(pl.Int32, strict=False),
            )
            .collect()
        )

    def read_lines(rows: Collection[int]) -> dict[int, str]:
        """Return the text of each row. Only an error and an unusual first line of a table need the text."""
        if not rows:
            return {}
        with polars_source_open(filepath) as source:
            dfrows = pl.scan_lines(source, row_index_name="row").filter(pl.col("row").is_in(list(rows))).collect()
        return dict(zip(dfrows["row"].to_list(), dfrows["line"].to_list(), strict=True))

    # the row of each error, and the function that gives the message from the text of the row
    errors: list[tuple[int, Callable[[str], str | None]]] = []

    # ARTIS reads the next line that is not blank and not a comment as a header, then the lines of its table. Thus
    # only the few headers need a loop
    # numpy casts a UInt32 array at each search with a Python int. Thus cast the array to Int64 one time
    contentrows = dflines.with_row_index("row").filter("iscontent")["row"].cast(pl.Int64).to_numpy()
    tables: list[tuple[int, int, int, int]] = []
    position = 0
    while position < len(contentrows):
        headerrow = int(contentrows[position])
        atomic_number, ion_stage, count = (dflines[column][headerrow] for column in ("int0", "int1", "int2"))
        if atomic_number is None or ion_stage is None or count is None or count < 0:
            errors.append((headerrow, partial(get_field_error, fields=ION_HEADER_FIELDS)))
            break
        tables.append((headerrow + 1, atomic_number, ion_stage, count))
        position = int(np.searchsorted(contentrows, headerrow + count, side="right"))

    keptions = [table for table in tables if ionlist is None or (table[1], table[2]) in ionlist]
    parsedcolumns = {5: ("int0", "int1", "float2", "float3", "int4"), 4: ("int0", "int1", "int2", "float3")}

    # a first line of 4 or 5 numbers gives the format of its table. Another first line needs its text, e.g. a line
    # with a comment after the numbers
    formatofrow: dict[int, int] = {}
    for firstrow, *_, count in keptions:
        if count > 0 and firstrow < dflines.height:
            values = dflines.row(firstrow, named=True)
            if values["tokencount"] in parsedcolumns and all(
                values[column] is not None and math.isfinite(values[column])
                for column in parsedcolumns[values["tokencount"]]
            ):
                formatofrow[firstrow] = values["tokencount"]
    unusualrows = [firstrow for firstrow, *_, count in keptions if count > 0 and firstrow not in formatofrow]
    for row, line in read_lines(unusualrows).items():
        if (numbercount := count_leading_numbers(line)) in parsedcolumns:
            formatofrow[row] = numbercount
        else:
            errors.append((row, get_format_error))

    outputcolumns = {
        5: {
            "lower": pl.col("int0") - firstlevelnumber,
            "upper": pl.col("int1") - firstlevelnumber,
            "A": pl.col("float2"),
            "collstr": pl.col("float3"),
            "forbidden": pl.col("int4"),
        },
        4: {
            "lower": pl.col("int1") - firstlevelnumber,
            "upper": pl.col("int2") - firstlevelnumber,
            "A": pl.col("float3"),
            "collstr": pl.lit(-1.0, dtype=pl.Float32),
            "forbidden": pl.lit(0, dtype=pl.Int32),
        },
    }
    transitiondata: dict[tuple[int, int], pl.DataFrame] = {}
    for firstrow, atomic_number, ion_stage, count in keptions:
        dftable = dflines.slice(firstrow, count)
        tableformat = formatofrow.get(firstrow)
        if dftable.is_empty():
            transitiondata[atomic_number, ion_stage] = pl.DataFrame(schema=dict(TRANSITION_SCHEMA))
        elif tableformat is not None:
            badrows = dftable.with_row_index("row").filter(
                pl.any_horizontal(pl.col(parsedcolumns[tableformat]).is_null())
            )["row"]
            if badrows.is_empty():
                transitiondata[atomic_number, ion_stage] = dftable.select(**outputcolumns[tableformat])
            else:
                errors.append((firstrow + badrows[0], partial(get_field_error, fields=TRANSITION_FIELDS[tableformat])))

    # ARTIS reads the lines in order, thus the first line that does not parse gives the error
    if errors:
        row, get_message = min(errors, key=operator.itemgetter(0))
        msg = f"{filepath}:{row + 1}: {get_message(read_lines([row])[row]) or 'the line does not parse'}"
        raise ValueError(msg)

    if tables and tables[-1][0] + tables[-1][3] > dflines.height:
        firstrow, atomic_number, ion_stage, count = tables[-1]
        msg = (
            f"{filepath}: the file ends after {dflines.height - firstrow} of the {count} transitions of"
            f" Z={atomic_number} ion_stage={ion_stage}"
        )
        raise ValueError(msg)

    return transitiondata


def get_transitiondata(
    modelpath: str | Path,
    ionlist: Collection[tuple[int, int]] | None = None,
    quiet: bool = False,
    *,
    firstlevelnumber: int,
) -> dict[tuple[int, int], pl.DataFrame]:
    """Return a dictionary of transitions from (Z, ion_stage) to a polars DataFrame.

    firstlevelnumber is the number of the first level of adata.txt, see get_first_level_number. A caller gives a
    list or a tuple of ions, thus this makes the arguments hashable for the cache. The copy of the dictionary and of
    each frame keeps a caller that changes one of them from changing what the next caller reads. A clone is cheap,
    because polars shares the data of a frame. The cache key holds the absolute path, see ModelpathCache.
    """
    transitionsdict = get_transitiondata_cached(
        resolve_modelpath(modelpath),
        tuple(ionlist) if ionlist is not None else None,
        firstlevelnumber=firstlevelnumber,
        quiet=quiet,
    )

    return {ion: dftransitions.clone() for ion, dftransitions in transitionsdict.items()}


@lru_cache(maxsize=8)
def get_transitiondata_cached(
    modelpath: Path, ionlist: tuple[tuple[int, int], ...] | None = None, *, firstlevelnumber: int, quiet: bool = False
) -> dict[tuple[int, int], pl.DataFrame]:
    """Return the transitions of each ion, and keep them for the next caller.

    Do not change the dictionary that this function returns.
    """
    ionset = set(ionlist) if ionlist else None
    transition_filename = misc.firstexisting("transitiondata.txt", folder=modelpath)

    time_start = time.perf_counter()
    if not quiet:
        print(f"Reading {transition_filename.relative_to(Path(modelpath).parent)}...")

    transitionsdict = read_transitiondata(transition_filename, ionlist=ionset, firstlevelnumber=firstlevelnumber)

    if not quiet:
        print(f"  took {time.perf_counter() - time_start:.2f} seconds")

    return transitionsdict


def get_lte_partfunc(pldflevels: pl.DataFrame, T_exc: float) -> float:
    """Return the LTE partition function of the ion at the excitation temperature."""
    return float(pldflevels.select(pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / T_exc).exp()).sum().item())


@on_model_host
def get_ion_levels(modelpath: Path, atomic_number: int, ion_stage: int) -> pl.DataFrame | None:
    """Return the energy levels of one ion as a plain frame, or None if the atomic data has no such ion.

    get_levels holds the frame of each ion inside an object column, and Arrow IPC cannot send such a column. The
    host of a remote model thus gives one ion at a time, with no object column.
    """
    # a parse of one ion takes 0.04 s on the test model and a parse of all the ions takes 0.07 s. A caller asks for
    # one to three ions, and the frame of one ion holds less memory
    dfion = get_levels(modelpath, ionlist=[(atomic_number, ion_stage)])
    if dfion.is_empty():
        return None
    # an object column of a level frame, e.g. the transitions of each level, has no Arrow form
    dflevels: pl.DataFrame = dfion["levels"].item()
    return dflevels.select(pl.exclude(pl.Object))


def get_levels(
    modelpath: str | Path,
    ionlist: Collection[tuple[int, int]] | None = None,
    get_transitions: bool = False,
    get_photoionisations: bool = False,
    quiet: bool = False,
    derived_transitions_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Return a polars DataFrame of energy levels.

    A caller gives a list or a tuple of ions, thus this makes the arguments hashable for the cache. The
    clone is cheap, because polars shares the data of the frame, and it keeps a caller that changes the
    columns in place from changing what the next caller reads. The levels and the transitions of each
    ion are frames of their own inside an object column, thus each one needs a clone as well. The cache
    key holds the absolute path, see ModelpathCache.
    """
    dflevels = get_levels_cached(
        resolve_modelpath(modelpath),
        tuple(ionlist) if ionlist is not None else None,
        get_transitions=get_transitions,
        get_photoionisations=get_photoionisations,
        quiet=quiet,
        derived_transitions_columns=(
            tuple(derived_transitions_columns) if derived_transitions_columns is not None else None
        ),
    ).clone()

    return dflevels.with_columns([
        pl.Series(colname, [nested.clone() for nested in dflevels[colname]], dtype=pl.Object)
        for colname in ("levels", "transitions")
    ])


@lru_cache(maxsize=8)
def get_levels_cached(
    modelpath: Path,
    ionlist: tuple[tuple[int, int], ...] | None = None,
    *,
    get_transitions: bool = False,
    get_photoionisations: bool = False,
    quiet: bool = False,
    derived_transitions_columns: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Return a polars DataFrame of energy levels, and keep it for the next caller.

    adata.txt is large, and a plot of one frame reads the levels of each ion. Thus a run without the
    cache parsed the same file many times. Do not change the frame that this function returns.
    """
    adatafilename = Path(modelpath, "adata.txt")
    # the numbering of adata.txt applies to the transitions and the photoionisation tables
    firstlevelnumber = get_first_level_number(modelpath)

    transitionsdict: dict[tuple[int, int], pl.DataFrame] = (
        get_transitiondata(modelpath, ionlist=ionlist, quiet=quiet, firstlevelnumber=firstlevelnumber)
        if get_transitions
        else {}
    )

    phixsdict: dict[tuple[int, int, int], tuple[npt.NDArray[np.void], npt.NDArray[np.void]]] = {}
    if get_photoionisations:
        phixs_filename = Path(modelpath, "phixsdata_v2.txt")

        if not quiet:
            print(f"Reading {phixs_filename.relative_to(Path(modelpath).parent)}")

        phixsdict = parse_phixsdata(phixs_filename, ionlist, firstlevelnumber=firstlevelnumber)

    level_lists: list[tuple[int, int, int, float, pl.DataFrame, pl.LazyFrame]] = []

    with misc.zopen(adatafilename) as fadata:
        if not quiet:
            print(f"Reading {adatafilename.relative_to(Path(modelpath).parent)}")

        for Z, ion_stage, level_count, ionisation_energy_ev, dflevels in parse_adata(
            fadata, phixsdict, ionlist, firstlevelnumber=firstlevelnumber
        ):
            if (Z, ion_stage) in transitionsdict:
                dftransitions = transitionsdict[Z, ion_stage].lazy()
                if derived_transitions_columns is not None:
                    dftransitions = add_transition_columns(dftransitions, dflevels, derived_transitions_columns)
            else:
                dftransitions = pl.LazyFrame()

            level_lists.append((Z, ion_stage, level_count, ionisation_energy_ev, dflevels, dftransitions))

    # the schema gives a model that holds none of the ions the same columns, thus a caller needs no special case
    dfallions = pl.DataFrame(
        level_lists,
        schema={
            "Z": pl.Int64,
            "ion_stage": pl.Int64,
            "level_count": pl.Int64,
            "ion_pot": pl.Float64,
            "levels": pl.Object,
            "transitions": pl.Object,
        },
        orient="row",
    )
    if get_photoionisations:
        # the arrays hold the cross sections of this read alone, thus a run without them needs no walk
        freeze_photoionisation_arrays(dfallions)

    return dfallions


def freeze_photoionisation_arrays(dfallions: pl.DataFrame) -> None:
    """Refuse a write to the photoionisation arrays that the cache holds.

    A clone of a frame shares the numpy array of an object column, thus a caller could change the
    cross sections that the next caller reads. A copy of those arrays takes 10 ms for each call of a
    small model, against 0.05 ms for the call itself, and a model of a full run holds far more of them.
    The arrays take this mark one time instead, thus such a write raises in place of passing.
    """
    for dflevels in dfallions["levels"]:
        for colname in ("phixstargetlist", "phixstable"):
            if colname not in dflevels.columns:
                continue

            for value in dflevels[colname]:
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)


roman_numerals = (
    "",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
)


@modelpath_cache(maxsize=8)
@on_model_host
def get_composition_data(filename: Path | str) -> pl.DataFrame:
    """Return a DataFrame containing details of included elements and ions.

    filename is the model folder or the path of compositiondata.txt. ARTIS ignores the text to the right of a #
    character, and it reads the numbers in sequence and not line by line. This reader does the same.
    """
    filename = Path(filename, "compositiondata.txt") if Path(filename).is_dir() else Path(filename)

    with zopen(filename, encoding="utf-8") as fcompdata:
        numbers = [number for line in fcompdata for number in line.partition("#")[0].split()]

    # the count of elements, T_preset, and homogeneous_abundances come first, then seven numbers for each element
    nelements = int(numbers[0]) if numbers else 0
    elementnumbers = numbers[3:]
    if not numbers or len(elementnumbers) < 7 * nelements:
        msg = f"{filename} gives {nelements} elements, but it does not hold seven numbers for each element"
        raise ValueError(msg)

    rows = [
        [int(x) for x in elementnumbers[index : index + 5]] + [float(x) for x in elementnumbers[index + 5 : index + 7]]
        for index in range(0, 7 * nelements, 7)
    ]

    return pl.DataFrame(
        rows,
        schema=[
            ("Z", pl.Int32),
            ("nions", pl.Int32),
            ("lowermost_ion_stage", pl.Int32),
            ("uppermost_ion_stage", pl.Int32),
            ("nlevelsmax_readin", pl.Int32),
            ("abundance", pl.Float64),
            ("mass", pl.Float64),
        ],
        orient="row",
    )


def get_kept_level_counts(modelpath: Path | str) -> dict[int, int | None]:
    """Return the count of levels that ARTIS keeps for each ion of each element of compositiondata.txt.

    ARTIS keeps the first nlevelsmax_readin levels of each ion. A negative value keeps all the levels, and gives None.
    """
    dfcomposition = get_composition_data(modelpath)
    return {
        Z: None if nlevelsmax < 0 else nlevelsmax
        for Z, nlevelsmax in zip(
            dfcomposition["Z"].to_list(), dfcomposition["nlevelsmax_readin"].to_list(), strict=True
        )
    }


def get_kept_levels(dflevels: pl.DataFrame, keptlevelcount: int | None) -> pl.DataFrame:
    """Return the levels of an ion that ARTIS keeps. A count of None keeps all the levels."""
    return dflevels if keptlevelcount is None else dflevels.head(keptlevelcount)


def get_ionstages_from_outputfile(modelpath: Path | str) -> dict[int, list[int]]:
    """Return the ion stages of each element from the log of a run, in the sequence of the log.

    After the read of the atomic data, the log lists each element and its ion stages in one block of lines:
    - a classic run writes "[input.c]   element Z = 26" and "[input.c]     ion 2 with ...";
    - a later run writes "[input]  element 0 (Z=26 Fe)" and "[input]    ionstage 2: ...";
    - a current run writes the same lines with an [info] tag.

    An element with no ion line gives an empty list.
    """
    elementpattern = re.compile(r"\[(?:input\.c|input|info)\]\s+element (?:Z = |\d+ \(Z=\s*)(\d+)")
    ionpattern = re.compile(r"\[(?:input\.c|input|info)\]\s+(?:ionstage|ion) (\d+)\b")
    logpath = Path(modelpath, "output_0-0.txt")

    ionstages_of_element: dict[int, list[int]] = {}
    with zopen(logpath, encoding="utf-8") as foutput:
        for line in foutput:
            # a log can hold an empty line, e.g. where a scheduler cut it or joined two logs
            if not line.strip():
                continue
            if (elementmatch := elementpattern.search(line)) is not None:
                ionstages_of_element.setdefault(int(elementmatch.group(1)), [])
            elif ionstages_of_element and (ionmatch := ionpattern.search(line)) is not None:
                ionstages_of_element[next(reversed(ionstages_of_element))].append(int(ionmatch.group(1)))
            elif ionstages_of_element:
                # the block ends here, thus the read stops before the rest of a log that can hold gigabytes
                break

    if not ionstages_of_element:
        msg = f"{logpath} holds no list of the elements and the ion stages of the run"
        raise ValueError(msg)

    return ionstages_of_element


def get_composition_data_from_outputfile(modelpath: Path | str) -> pl.DataFrame:
    """Read the ion list from the log of a run, in case compositiondata.txt is not available."""
    return pl.DataFrame(
        [
            (Z, min(ionstages, default=None), max(ionstages, default=None))
            for Z, ionstages in get_ionstages_from_outputfile(modelpath).items()
        ],
        schema=[("Z", pl.Int32), ("lowermost_ion_stage", pl.Int32), ("uppermost_ion_stage", pl.Int32)],
        orient="row",
    ).with_columns(nions=pl.col("uppermost_ion_stage") - pl.col("lowermost_ion_stage") + 1)


def get_z_a_nucname(nucname: str) -> tuple[int, int]:
    """Return the atomic number and the mass number of a nuclide name.

    For example, 'Pb208', 'X_Pb208', and 'nniso_Pb208' each give (82, 208).
    """
    if "_" in nucname:
        nucname = nucname.split("_")[1]

    # the mass number stays on the name, because it tells the free neutron "n1" from nitrogen
    z = get_atomic_number(nucname)
    assert z >= 0, f"{nucname} does not start with an element symbol"

    a = int(nucname.lower().lstrip(string.ascii_lowercase))

    return z, a


@lru_cache(maxsize=1)
def get_elements_df() -> pl.DataFrame:
    """Return the whole element table (Z, symbol, name, mass) from data/elements.csv.

    The single place that knows the file's schema, so adding a column does not have to be mirrored into
    every helper below. Cached and shared, so callers must derive a new frame rather than modify this one.
    """
    return pl.read_csv(
        get_path("datadir") / "elements.csv", has_header=True, separator=",", schema_overrides={"Z": pl.Int32}
    )


@lru_cache(maxsize=1)
def get_elsymbolset() -> frozenset[str]:
    """Return the element symbols as a set, for a test of membership.

    get_elsymbolslist gives a tuple, thus "in" reads all 119 symbols. The listing of the estimator
    variables makes that test one time for each candidate split of every column name.
    """
    return frozenset(get_elsymbolslist())


@lru_cache(maxsize=1)
def get_elsymbolslist() -> tuple[str, ...]:
    """Return the element symbols indexed by atomic number.

    A tuple, not a list, because the single cached instance is shared by every caller.

    Example:
    -------
    get_elsymbolslist()[26] = 'Fe'.

    """
    return ("n", *get_elements_df()["symbol"].to_list())


@lru_cache(maxsize=1)
def get_atomic_masses() -> Mapping[int, float]:
    """Return the atomic mass in atomic mass units of every element, keyed by atomic number.

    These are the IUPAC standard atomic weights, except for the elements with no stable isotope, where the
    convention is the mass number of the longest-lived isotope. Every element from H to Og has an entry, so a
    lookup never needs a fallback.

    A read-only mapping, because the single cached instance is shared by every caller.
    """
    dfelements = get_elements_df()
    return MappingProxyType(dict(zip(dfelements["Z"].to_list(), dfelements["mass"].to_list(), strict=True)))


@lru_cache(maxsize=1)
def get_atomic_number_of_elsymbol() -> dict[str, int]:
    """Return a mapping of element symbol to atomic number, for lookups that would otherwise scan the whole list."""
    return {elsymbol: atomic_number for atomic_number, elsymbol in enumerate(get_elsymbolslist())}


@lru_cache(maxsize=1)
def get_elsymbols_longestfirst() -> tuple[str, ...]:
    """Return the element symbols ordered longest first, for matching a symbol at the start of a string.

    'Fe' must be tried before 'F', otherwise 'FeII' would match the fluorine prefix.
    """
    return tuple(sorted(get_elsymbolslist(), key=len, reverse=True))


def get_elsymbols_df() -> pl.LazyFrame:
    """Return a polars LazyFrame of atomic number and element symbols.

    Only the two columns, so that a join against this frame never picks up the rest of the element table.
    """
    return get_elements_df().lazy().select(pl.col("Z").alias("atomic_number"), pl.col("symbol").alias("elsymbol"))


def get_atomic_number(elsymbol: str) -> int:
    """Return the atomic number of an element symbol, or -1 if it is not an element symbol."""
    assert elsymbol is not None
    name = elsymbol.removeprefix("X_").split("_")[0].split("-")[0]

    # only "n1" is the free neutron. A symbol is not case sensitive, thus "n", "n14", and "nII" are
    # nitrogen. No isotope of nitrogen has a mass number of 1
    if name == "n1":
        return 0

    # a dict lookup, because this is called once per column name in some loops
    return get_atomic_number_of_elsymbol().get(name.rstrip(string.digits).title(), -1)


ROMANNUMERALCHARS = frozenset("IVXLCDM")


def split_compact_ion_name(ionstr: str) -> tuple[int, int] | None:
    """Return the atomic number and the ion stage of a name such as SiII, or None for another name.

    The roman numeral of an ion stage is always in upper case, thus the lower case "i" of "SiII"
    belongs to the symbol of the element and that name gives Si II. The symbol itself is not case
    sensitive, thus "siII" gives Si II as well.

    The longest run of roman numerals at the end comes first. Thus "SIII" gives S III, and not the
    Si II that the symbol "SI" and a shorter run would give.
    """
    for splitpos in range(1, len(ionstr)):
        elsymbol, ionstagestr = ionstr[:splitpos], ionstr[splitpos:]
        if any(char not in ROMANNUMERALCHARS for char in ionstagestr):
            continue

        atomic_number = get_atomic_number(elsymbol)
        ion_stage = decode_roman_numeral(ionstagestr)
        if atomic_number > 0 and ion_stage > 0:
            return atomic_number, ion_stage

    return None


def decode_roman_numeral(strin: str) -> int:
    """Return the integer corresponding to a Roman numeral."""
    if strin.upper() in roman_numerals:
        return roman_numerals.index(strin.upper())
    return -1


def get_ion_stage_roman_numeral_df() -> pl.DataFrame:
    """Return a polars DataFrame of ionisation stage and roman numerals."""
    return pl.DataFrame({"ion_stage_roman": roman_numerals[1:]}, schema={"ion_stage_roman": pl.String}).with_row_index(
        "ion_stage", offset=1
    )


def add_ion_str_column(lz: pl.LazyFrame) -> pl.LazyFrame:
    """Add an ion_str column, for example 'Fe II'.

    The frame must have an atomic_number column and an ion_stage column.
    The function adds one column only. It removes the two columns that the joins supply.
    """
    return (
        lz
        .join(get_ion_stage_roman_numeral_df().lazy(), on="ion_stage", how="left", maintain_order="left")
        .join(get_elsymbols_df().lazy(), on="atomic_number", how="left", maintain_order="left")
        .with_columns(ion_str=pl.col("elsymbol") + " " + pl.col("ion_stage_roman"))
        .drop("elsymbol", "ion_stage_roman")
    )


def get_elsymbol(atomic_number: int | np.int64) -> str:
    """Return the element symbol of an atomic number."""
    return get_elsymbolslist()[atomic_number]


def get_ion_tuple(ionstr: str) -> tuple[int, int] | int:
    """Return the atomic number and the ionisation stage of an ion string, e.g. (26, 2) for 'FeII', 'Fe II', or '26_2'.

    Return the atomic number for a string like 'Fe' or '26'.
    """
    ionstr = ionstr.removeprefix("X_").removeprefix("nnelement_").removeprefix("nnion_")

    if ionstr.isdigit():
        return int(ionstr)

    if ionstr in get_elsymbolslist():
        return get_atomic_number(ionstr)

    elem = "?"
    strion_stage = "?"
    if " " in ionstr:
        elem, strion_stage = ionstr.split(" ", maxsplit=1)
    elif "_" in ionstr:
        elem, strion_stage = ionstr.split("_", maxsplit=1)
    elif (compaction := split_compact_ion_name(ionstr)) is not None:
        # no separator, e.g. 'FeII'
        return compaction
    else:
        # a mass number in place of an ion stage, e.g. 'Fe56'
        for elsym in get_elsymbols_longestfirst():
            if ionstr.startswith(elsym) and ionstr.removeprefix(elsym).isdigit():
                elem = elsym
                strion_stage = ionstr.removeprefix(elsym)
                break

    if elem in {"?", ""} or strion_stage in {"?", ""}:
        msg = f"Could not parse ionstr {ionstr}"
        raise ValueError(msg)

    atomic_number = int(elem) if elem.isdigit() else get_atomic_number(elem)
    ion_stage = int(strion_stage) if strion_stage.isdigit() else decode_roman_numeral(strion_stage)
    if ion_stage < 0:
        msg = f"Could not parse ionstr {ionstr}"
        raise ValueError(msg)

    return (atomic_number, ion_stage)


def get_ionstring(
    atomic_number: int | np.int64,
    ion_stage: int | np.int64 | str | None,
    style: t.Literal["spectral", "chargelatex", "charge"] = "spectral",
    sep: str = " ",
) -> str:
    """Return a string with the element symbol and ionisation stage."""
    if ion_stage is None or ion_stage == "ALL":
        return get_elsymbol(atomic_number)

    if isinstance(ion_stage, str) and ion_stage.startswith(get_elsymbol(atomic_number)):
        # nuclides like Sr89 get passed in as atomic_number=38, ion_stage='Sr89'
        return ion_stage

    assert not isinstance(ion_stage, str)

    if style == "spectral":
        return f"{get_elsymbol(atomic_number)}{sep}{roman_numerals[ion_stage]}"

    strcharge = ""
    if style == "chargelatex":
        # ion notion e.g. Co+, Fe2+
        if ion_stage > 2:
            strcharge = r"$^{" + str(ion_stage - 1) + r"{+}}$"
        elif ion_stage == 2:
            strcharge = r"$^{+}$"
        elif ion_stage == 1:
            strcharge = r"$^{0}$"
    elif ion_stage > 2:
        strcharge = f"{ion_stage - 1}+"
    elif ion_stage == 2:
        strcharge = "+"
    elif ion_stage == 1:
        strcharge = "0"

    return f"{get_elsymbol(atomic_number)}{strcharge}"


@on_model_host
def get_nuclides(modelpath: Path | str) -> pl.LazyFrame:
    """Return the nuclides of nuclides.out, and a row with the pellet_nucindex -1 for the initial energy.

    The LazyFrame has the columns pellet_nucindex, atomic_number, A, elsymbol, and nucname.
    """
    filepath = firstexisting_or_none("nuclides.out", folder=modelpath, tryzipped=True, search_subfolders=False)
    if filepath is None:
        msg = f"File nuclides.out not found in {modelpath}"
        raise FileNotFoundError(msg)

    dfnuclides = (
        pl
        .scan_csv(polars_source(filepath), separator=" ", has_header=True)
        .rename({"#nucindex": "pellet_nucindex", "Z": "atomic_number"})
        .join(get_elsymbols_df().lazy(), on="atomic_number", how="left", maintain_order="left")
        .with_columns(nucname=pl.col("elsymbol") + pl.col("A").cast(pl.String))
    ).with_columns(pl.col(pl.Int64).cast(pl.Int32))

    return pl.concat(
        [
            pl.LazyFrame(
                {
                    "pellet_nucindex": [-1],
                    "atomic_number": [-1],
                    "A": [-1],
                    "elsymbol": ["initial energy"],
                    "nucname": ["initial energy"],
                },
                schema=dfnuclides.collect_schema(),
            ),
            dfnuclides,
        ],
        how="vertical",
    ).lazy()


def get_bflist(modelpath: Path | str) -> pl.LazyFrame:
    """Return a LazyFrame of the bound-free transitions in bflist.out. The frame includes an ion_str column."""
    compositiondata = get_composition_data(modelpath)
    bflistpath = firstexisting(["bflist.out", "bflist.dat"], folder=modelpath, tryzipped=True)
    print(f"Reading {bflistpath}")
    schema = {
        "bfindex": pl.Int32,
        "elementindex": pl.Int32,
        "ionindex": pl.Int32,
        "lowerlevel": pl.Int32,
        "upperionlevel": pl.Int32,
    }
    # a run with no bound-free transitions writes only the count line, and scan_csv would raise
    # NoDataError at whatever collect() eventually consumes this frame, far from the cause here
    with zopen(bflistpath) as fbflist:
        fbflist.readline()  # the number of transitions
        hastransitions = bool(fbflist.readline().strip())

    dfboundfree = (
        pl.scan_csv(
            polars_source(bflistpath),
            skip_rows=1,
            has_header=False,
            separator=" ",
            new_columns=["bfindex", "elementindex", "ionindex", "lowerlevel", "upperionlevel"],
            schema_overrides=schema,
        )
        if hastransitions
        else pl.LazyFrame(schema=schema)
    )

    # elementindex is the row position in compositiondata; replace_strict keeps the original behaviour of
    # failing loudly on an index that compositiondata does not cover
    z_of_elementindex = dict(enumerate(compositiondata["Z"]))
    lowermost_ion_stage_of_elementindex = dict(enumerate(compositiondata["lowermost_ion_stage"]))

    dfboundfree = dfboundfree.with_columns(
        atomic_number=pl.col("elementindex").replace_strict(z_of_elementindex, return_dtype=pl.Int32),
        ion_stage=(
            pl.col("ionindex")
            + pl.col("elementindex").replace_strict(lowermost_ion_stage_of_elementindex, return_dtype=pl.Int32)
        ),
    )

    return add_ion_str_column(dfboundfree.drop(["elementindex", "ionindex"]))


def read_linestatfile(
    filepath: Path | str,
) -> tuple[
    npt.NDArray[np.floating], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]
]:
    """Load linestat.out containing transitions wavelength, element, ion, upper and lower levels."""
    if Path(filepath).is_dir():
        filepath = firstexisting(
            "linestat.out",
            folder=filepath,
            tryzipped=True,
            purpose=(
                "linestat.out gives the wavelength, the element, the ion, and the levels of each line. "
                "The commands that identify a line read it, e.g. plotlinefluxes and "
                "plotspectra --emissionabsorption."
            ),
        )

    print(f"Reading {filepath}")

    # each of the five rows holds one value for each line. A file of one line gives a one-dimensional array without
    # ndmin, and a file of no line gives no row
    with zopen(filepath) as linestatfile:
        data = np.loadtxt(linestatfile, ndmin=2)
    if data.shape[0] < 5:
        data = np.zeros((5, 0))
    lambda_angstroms = data[0] * 1e8
    nlines = len(lambda_angstroms)

    atomic_numbers = data[1].astype(np.int32)
    assert len(atomic_numbers) == nlines

    ion_stages = data[2].astype(np.int32)
    assert len(ion_stages) == nlines

    # the file adds one to the levelindex, i.e. lowest level is 1
    upper_levels = data[3].astype(np.int32)
    assert len(upper_levels) == nlines

    lower_levels = data[4].astype(np.int32)
    assert len(lower_levels) == nlines

    return lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels


def get_linelist_pldf(modelpath: Path | str) -> pl.LazyFrame:
    """Return the transition list. Each row is one line. The rows are in lineindex order.

    Keep the rows in lineindex order. A caller finds a line by its row position in this frame.
    The emission and the absorption type codes of a packet are lineindex values.
    Do not add a sort operation, a filter operation, or a unique operation to this function.
    """
    textfile = firstexisting(
        "linestat.out",
        folder=modelpath,
        purpose=(
            "linestat.out gives the wavelength, the element, the ion, and the levels of each line. "
            "The commands that identify a line read it, e.g. plotlinefluxes and "
            "plotspectra --emissionabsorption."
        ),
    )
    # the .tmp suffix marks this as a regenerable cache, matching every other parquet file artistools writes
    parquetfile = Path(modelpath, "linelist.out.parquet.tmp")
    textsource_mtime = textfile.stat().st_mtime
    # leave a stale file in place: write_parquet_atomic() puts the new one at the path in one step. The
    # identity comes from before the check below, thus a fresh file that a rival process installs after
    # the check keeps its place and only the file that the check saw is replaced
    outdatedparquet = get_file_identity(parquetfile)
    _, stalereason = read_parquet_cache_metadata(parquetfile, LINELIST_CACHEVERSION, textsource_mtime)
    if stalereason is not None:
        if outdatedparquet is not None:
            print(f"{parquetfile} is not a current cache of {textfile.name}, because {stalereason}")

        lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels = read_linestatfile(textfile)

        pldf = (
            pl
            .DataFrame({
                "lambda_angstroms": lambda_angstroms,
                "atomic_number": atomic_numbers,
                "ion_stage": ion_stages,
                "upper_level": upper_levels,
                "lower_level": lower_levels,
            })
            .with_row_index(name="lineindex")
            .with_columns(cs.integer().cast(pl.Int32), cs.float().cast(pl.Float32))
        )
        write_parquet_atomic(
            pldf,
            parquetfile,
            metadata={"cacheversion": str(LINELIST_CACHEVERSION), "textsource_mtime": str(textsource_mtime)},
            replaces=outdatedparquet,
        )
        print(f"Wrote {parquetfile}")
    else:
        print(f"Reading {parquetfile}")

    return (
        pl
        .scan_parquet(parquetfile)
        .with_columns(
            pl
            .when(pl.col("lambda_angstroms").is_between(2000, 20000))
            .then(pl.col("lambda_angstroms") / 1.0003)
            .otherwise(pl.col("lambda_angstroms"))
            .alias("lambda_angstroms_air"),
            pl.col(pl.UInt32).cast(pl.Int32),
            pl.col(pl.Int64).cast(pl.Int32),
            pl.col(pl.Float64).cast(pl.Float32),
        )
        .with_columns(upperlevelindex=pl.col("upper_level") - 1, lowerlevelindex=pl.col("lower_level") - 1)
        .drop(["upper_level", "lower_level"])
    )


def get_lineindices(
    modelpath: Path | str,
    atomic_numbers: Collection[int] | None,
    ion_stages: Collection[int] | None,
    linefilter: pl.Expr | None = None,
) -> pl.Series:
    """Return the lineindex of each line of the given elements and ion stages. A None collection selects all.

    linefilter selects a part of those lines, e.g. by the levels of each line.
    """
    dflinelist = get_linelist_pldf(modelpath)
    if atomic_numbers is not None:
        dflinelist = dflinelist.filter(pl.col("atomic_number").is_in(atomic_numbers))
    if ion_stages is not None:
        dflinelist = dflinelist.filter(pl.col("ion_stage").is_in(ion_stages))
    if linefilter is not None:
        dflinelist = dflinelist.filter(linefilter)
    return dflinelist.select("lineindex").collect().get_column("lineindex")
