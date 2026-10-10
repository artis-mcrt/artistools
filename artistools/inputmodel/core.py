"""Read, write, and derive columns for ARTIS model.txt and abundance input files."""

import datetime
import errno
import gc
import json
import math
import os
import time
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.atomic import get_z_a_nucname
from artistools.commands import get_path
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.misc import firstexisting
from artistools.misc import get_file_identity
from artistools.misc import path_is_codecomparison
from artistools.misc import polars_source
from artistools.misc import read_parquet_cache_metadata
from artistools.misc import read_wsv
from artistools.misc import resolve_outputfile
from artistools.misc import write_parquet_atomic
from artistools.misc import zopen
from artistools.misc.fileio import COMPRESSED_EXTENSIONS
from artistools.misc.fileio import find_compressed
from artistools.misc.fileio import get_file_copies
from artistools.misc.fileio import get_plain_path
from artistools.misc.fileio import modelpath_cache
from artistools.misc.fileio import MTIME_TOLERANCE_S
from artistools.misc.fileio import with_compressed_extension
from artistools.misc.fileio import write_zstd_lines
from artistools.misc.general import get_bin_index_expr
from artistools.misc.modelinfo import parse_npts_line
from artistools.misc.remote import check_local_path
from artistools.misc.remote import on_model_host

CREATED_COMMENT_PREFIX = "created:"
CREATED_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"
UNITS_COMMENT_PREFIX = "column units:"
UNITS_COMMENT_END = "Each X_ column is a mass fraction"


def get_created_comment() -> str:
    """Return the comment line that gives the creation time of an input file in UTC."""
    return f"# {CREATED_COMMENT_PREFIX} {datetime.datetime.now(tz=datetime.UTC).strftime(CREATED_TIME_FORMAT)}"


def is_writer_comment(commentline: str) -> bool:
    """Return True for a header comment that save_modeldata writes again.

    A comment of a user can start with the same words, e.g. "created: by hand". Thus the units line
    must also have the end that save_modeldata writes, and the creation line must have the time format
    of get_created_comment.
    """
    if commentline.startswith(UNITS_COMMENT_PREFIX) and commentline.endswith(UNITS_COMMENT_END):
        return True
    try:
        time.strptime(commentline, f"{CREATED_COMMENT_PREFIX} {CREATED_TIME_FORMAT}")
    except ValueError:
        return False
    return True


def has_single_space_separators(line: str) -> bool:
    """Return True if one space separates each field of the line, and no space starts it.

    A leading space, a tab, or a double space gives polars an empty field. The cell id then becomes
    null, and each value moves one column to the right. The reader drops an empty field after the last value.
    An empty line has no field, e.g. the second data line of a model with one cell.
    """
    stripped = line.rstrip()
    return not stripped or stripped.split(" ") == stripped.split()


def read_noncomment_line(fmodel: t.IO[str], headercommentlines: list[str]) -> tuple[str, int]:
    """Return the next line that is not a comment or empty, and the number of lines that the function read.

    sn3d skips such lines before the cell count and before the time of the model. The function keeps each comment
    in headercommentlines, except a comment that save_modeldata writes again.
    """
    linecount = 1
    line = fmodel.readline()
    # an empty string is the end of the file, and a comment can have spaces before it
    while line.lstrip().startswith("#") or (line and not line.strip()):
        if line.strip():
            commentline = line.lstrip().removeprefix("#").removeprefix(" ").removesuffix("\n")
            # save_modeldata writes these lines again, thus a kept copy gives each line two times
            if not is_writer_comment(commentline):
                headercommentlines.append(commentline)
        linecount += 1
        line = fmodel.readline()

    return line, linecount


def read_modelfile_text(
    filename: Path | str, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides."""
    if not printwarningsonly:
        print(f"Reading {filename}")

    with zopen(filename) as fmodel:
        onelinepercellformat: bool | None = None

        modelmeta: dict[str, t.Any] = {"headercommentlines": []}
        xmax_tmodel: float | int = 0.0
        ncoordgridx: int = 0
        ncoordgridy: int = 0
        ncoordgridz: int = 0

        line, numheaderrows = read_noncomment_line(fmodel, modelmeta["headercommentlines"])
        cellcounts = parse_npts_line(line, filename)
        if len(cellcounts) == 2:
            modelmeta["dimensions"] = 2
            ncoordgridr, ncoordgridz = cellcounts
            modelmeta["ncoordgridrcyl"] = ncoordgridr
            modelmeta["ncoordgridz"] = ncoordgridz
            npts_model = ncoordgridr * ncoordgridz
            if not printwarningsonly:
                print(f"  detected 2D model file with n_r * n_z = {ncoordgridr} x {ncoordgridz} = {npts_model} cells")
        else:
            npts_model = cellcounts[0]

        modelmeta["npts_model"] = npts_model
        line, linecount = read_noncomment_line(fmodel, modelmeta["headercommentlines"])
        numheaderrows += linecount
        try:
            modelmeta["t_model_init_days"] = float(line.split("#", 1)[0])
        except ValueError:
            msg = f"In {filename}, the line after the cell count must give the time of the model in days, not {line!r}"
            raise ValueError(msg) from None
        t_model_init_seconds = modelmeta["t_model_init_days"] * 24 * 60 * 60

        line = fmodel.readline()
        # if the next line is a single float then the model is 2D or 3D (vmax)
        try:
            modelmeta["vmax_cmps"] = float(line.split("#", 1)[0])
        except ValueError:
            assert modelmeta.get("dimensions", -1) != 2, "2D model should have a vmax line here"
            if "dimensions" not in modelmeta:
                if not printwarningsonly:
                    print(f"  detected 1D model file with {npts_model} radial zones")
                modelmeta["dimensions"] = 1
        else:
            xmax_tmodel = modelmeta["vmax_cmps"] * t_model_init_seconds  # xmax = ymax = zmax
            numheaderrows += 1
            if "dimensions" not in modelmeta:  # not already detected as 2D
                modelmeta["dimensions"] = 3
                # number of grid cell steps along an axis (currently the same for xyz)
                ncoordgridx = ncoordgridy = ncoordgridz = round(npts_model ** (1.0 / 3.0))
                assert (ncoordgridx * ncoordgridy * ncoordgridz) == npts_model
                modelmeta["ncoordgridx"] = ncoordgridx
                modelmeta["ncoordgridy"] = ncoordgridy
                modelmeta["ncoordgridz"] = ncoordgridz
                modelmeta["ncoordgrid"] = ncoordgridx

                if not printwarningsonly:
                    print(
                        f"  detected 3D model file with {ncoordgridx} x {ncoordgridy} x {ncoordgridz} = {npts_model} cells"
                    )

            line = fmodel.readline()

        columns = None
        if line.startswith("#"):
            numheaderrows += 1
            columns = line.lstrip("#").split()
            line = fmodel.readline()

        data_line_even = line
        ncols_line_even = len(data_line_even.split())
        data_line_odd = fmodel.readline()
        ncols_line_odd = len(data_line_odd.split())

    if ncols_line_even == 0:
        msg = f"{filename}: found only 0 cells instead of {npts_model} expected."
        raise ValueError(msg)

    if columns is None:
        columns = get_standard_columns(modelmeta["dimensions"], includenico57=True, pos_unknown=True)
        # last two abundances are optional
        assert columns is not None
        if ncols_line_even == ncols_line_odd and (ncols_line_even + ncols_line_odd) > len(columns):
            # one line per cell format
            ncols_line_odd = 0

        assert len(columns) in {ncols_line_even + ncols_line_odd, ncols_line_even + ncols_line_odd + 2}
        columns = columns[: ncols_line_even + ncols_line_odd]

    assert columns is not None
    if ncols_line_even == len(columns):
        if not printwarningsonly:
            print("  model file is one line per cell")
        ncols_line_odd = 0
        onelinepercellformat = True
    else:
        if not printwarningsonly:
            print("  model file format is two lines per cell")
        # columns split over two lines
        assert (ncols_line_even + ncols_line_odd) == len(columns)
        onelinepercellformat = False

    if onelinepercellformat and all(has_single_space_separators(line) for line in (data_line_even, data_line_odd)):
        if not printwarningsonly:
            print("  using fast method polars.read_csv (requires one line per cell and single space delimiters)")

        dfmodel = pl.read_csv(
            polars_source(filename),
            separator=" ",
            new_columns=columns,
            n_rows=npts_model,
            has_header=False,
            skip_rows=numheaderrows,
            schema={col: pl.Int32 if col == "inputcellid" else pl.Float32 for col in columns},
            truncate_ragged_lines=True,
            # a trailing space gives an empty last field. polars 2 raises if a line has more fields than column names
            extra_columns="ignore",
            # a header comment can hold one quotation mark, e.g. 5" model. A reader that takes it as a quote
            # skips the rows to the next quotation mark, and the data lines then go with the header
            quote_char=None,
        ).lazy()

    else:
        # a cell of dfmodelraw can span two lines. One read of the full file and a slice after it
        # prevent a second read
        dfmodelraw = read_wsv(
            filename,
            has_header=False,
            skip_rows=numheaderrows,
            new_columns=[str(i) for i in range(max(ncols_line_even, ncols_line_odd))],
        )

        dfmodel = (
            (dfmodelraw if onelinepercellformat else dfmodelraw[: npts_model * 2 : 2])
            .select([pl.col(str(i)).alias(colname) for i, colname in enumerate(columns[:ncols_line_even])])
            .with_columns(pl.col("inputcellid").cast(pl.Int32))
        )

        if ncols_line_odd > 0 and not onelinepercellformat:
            # the second line of a cell holds no cell id. It follows the first line, thus the two go side by side.
            # An id from a count would put the second line of each cell with the next cell when the ids start at 0
            dfmodeloddlines = dfmodelraw[1 : npts_model * 2 : 2].select([
                pl.col(str(i)).alias(colname) for i, colname in enumerate(columns[ncols_line_even:])
            ])
            if dfmodeloddlines.height != dfmodel.height:
                msg = f"{filename}: found only {dfmodeloddlines.height} cells instead of {npts_model} expected."
                raise ValueError(msg)
            dfmodel = pl.concat([dfmodel, dfmodeloddlines], how="horizontal")

        dfmodel = dfmodel.head(npts_model).with_columns(pl.exclude("inputcellid").cast(pl.Float32)).lazy()

    # an old model.txt names the electron fraction cellYe, and Ye is the name everywhere after this point
    dfmodel = dfmodel.sort("inputcellid").rename({"velocity_outer": "vel_r_max_kmps", "cellYe": "Ye"}, strict=False)

    # sn3d stops for a file with too few cells. It takes the id of the first cell (0 or 1) as the start of the
    # cell index, and each id after it must be one more than the id before it
    cellcount, firstinputcellid, idsareconsecutive = (
        dfmodel
        .select(
            cellcount=pl.len(),
            firstinputcellid=pl.col("inputcellid").first(),
            idsareconsecutive=(pl.col("inputcellid") == pl.col("inputcellid").first() + pl.int_range(pl.len())).all(),
        )
        .collect()
        .row(0)
    )
    if cellcount != npts_model:
        msg = f"{filename}: found only {cellcount} cells instead of {npts_model} expected."
        raise ValueError(msg)
    if firstinputcellid not in {0, 1} or not idsareconsecutive:
        msg = f"{filename}: the inputcellid values must start at 0 or 1 and increase by one from each cell to the next"
        raise ValueError(msg)

    if modelmeta["dimensions"] == 1:
        vmax_kmps = dfmodel.select(pl.col("vel_r_max_kmps").max()).collect().item()
        assert isinstance(vmax_kmps, float)
        modelmeta["vmax_cmps"] = vmax_kmps * km_to_cm

    elif modelmeta["dimensions"] == 2:
        wid_init_rcyl = modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridrcyl"]
        wid_init_z = 2 * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridz"]
        modelmeta["wid_init_rcyl"] = wid_init_rcyl
        modelmeta["wid_init_z"] = wid_init_z

        # check pos_rcyl_mid and pos_z_mid are correct. One expression over the whole column instead of a Python
        # loop, which cost a round trip through the interpreter for every cell of the grid
        n_r = (pl.col("inputcellid") - firstinputcellid) % modelmeta["ncoordgridrcyl"]
        n_z = (pl.col("inputcellid") - firstinputcellid) // modelmeta["ncoordgridrcyl"]
        pos_z_min_grid = -modelmeta["vmax_cmps"] * t_model_init_seconds

        maxoffby = (
            dfmodel
            .select(
                rcyl_offby=(pl.col("pos_rcyl_mid") - wid_init_rcyl * (n_r + 0.5)).abs().max(),
                z_offby=(pl.col("pos_z_mid") - (pos_z_min_grid + wid_init_z * (n_z + 0.5))).abs().max(),
                rcyl_expected=(wid_init_rcyl * (n_r + 0.5)).abs().max(),
                z_expected=(pos_z_min_grid + wid_init_z * (n_z + 0.5)).abs().max(),
            )
            .collect()
            .row(0, named=True)
        )

        # half a cell width, plus the relative term np.isclose() used to contribute, so that a model which
        # loaded before this check was vectorised is not now rejected over float32 rounding of a ~1e15 cm position
        rtol = 1.0e-5
        # raise an error and do not assert, because the interpreter option -O removes an assert.
        # The model file comes from the user
        if maxoffby["rcyl_offby"] > wid_init_rcyl / 2.0 + rtol * maxoffby["rcyl_expected"]:
            msg = f"pos_rcyl_mid is up to {maxoffby['rcyl_offby']:.3e} cm from the expected cell centre"
            raise AssertionError(msg)
        if maxoffby["z_offby"] > wid_init_z / 2.0 + rtol * maxoffby["z_expected"]:
            msg = f"pos_z_mid is up to {maxoffby['z_offby']:.3e} cm from the expected cell centre"
            raise AssertionError(msg)

    elif modelmeta["dimensions"] == 3:
        wid_init_x = 2 * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridx"]
        wid_init_y = 2 * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridy"]
        wid_init_z = 2 * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridz"]
        modelmeta["wid_init_x"] = wid_init_x
        modelmeta["wid_init_y"] = wid_init_y
        modelmeta["wid_init_z"] = wid_init_z
        modelmeta["wid_init"] = wid_init_x
        if "pos_x_min" in dfmodel.collect_schema().names():
            if not printwarningsonly:
                print("  model cell positions are defined in the header")
            firstrow = dfmodel.select(cs.starts_with("pos_")).first().collect().row(index=0, named=True)
            expected_positions = (
                ("pos_x_min", -xmax_tmodel),
                ("pos_y_min", -xmax_tmodel),
                ("pos_z_min", -xmax_tmodel),
                ("pos_x_mid", -xmax_tmodel + wid_init_x / 2.0),
                ("pos_y_mid", -xmax_tmodel + wid_init_y / 2.0),
                ("pos_z_mid", -xmax_tmodel + wid_init_z / 2.0),
            )
            # a wrong vmax gives a wrong cell width, and thus a wrong volume and a wrong mass for each cell
            for col, pos in expected_positions:
                if col in firstrow and not math.isclose(firstrow[col], pos, rel_tol=0.01):
                    msg = (
                        f"{filename}: {col} of the first cell is {firstrow[col]:.6e} cm, but vmax and "
                        f"t_model_init_days give {pos:.6e} cm. Make vmax consistent with the cell positions."
                    )
                    raise ValueError(msg)

        else:
            # sn3d reads the positions from the three columns after inputcellid, whatever names the header gives
            # them, e.g. pos_x_mid. Thus the values below give the order of the axes and the place in the cell
            dfmodel = dfmodel.rename(
                dict(zip(columns[1:4], ("inputpos_a", "inputpos_b", "inputpos_c"), strict=True)), strict=False
            )

            def vectormatch(vec1: Sequence[float], vec2: Sequence[float]) -> bool:
                xclose = np.isclose(vec1[0], vec2[0], atol=wid_init_x * 0.05)
                yclose = np.isclose(vec1[1], vec2[1], atol=wid_init_y * 0.05)
                zclose = np.isclose(vec1[2], vec2[2], atol=wid_init_z * 0.05)

                return all([xclose, yclose, zclose])

            # candidate coordinate column orderings: key -> (message, column renames)
            posordercandidates = {
                "xyz_min": (
                    "  model cell positions are consistent with x-y-z min corner columns",
                    {"inputpos_a": "pos_x_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_z_min"},
                ),
                "zyx_min": (
                    "  cell positions are consistent with z-y-x min corner columns",
                    {"inputpos_a": "pos_z_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_x_min"},
                ),
                "xyz_mid": (
                    "  model cell positions are consistent with x-y-z midpoint columns",
                    {"inputpos_a": "pos_x_mid", "inputpos_b": "pos_y_mid", "inputpos_c": "pos_z_mid"},
                ),
                "zyx_mid": (
                    "  cell positions are consistent with z-y-x midpoint columns",
                    {"inputpos_a": "pos_z_mid", "inputpos_b": "pos_y_mid", "inputpos_c": "pos_x_mid"},
                ),
            }
            matched = dict.fromkeys(posordercandidates, True)
            # important cell numbers to check for coordinate column order
            indexlist = [
                0,
                ncoordgridx - 1,
                ncoordgridx,
                (ncoordgridx - 1) * (ncoordgridy - 1),
                (ncoordgridx - 1) * ncoordgridy,
                (ncoordgridx - 1) * (ncoordgridy - 1) * (ncoordgridz - 1),
            ]

            pos3_in_list = (
                dfmodel
                .select(
                    cs.by_name("inputpos_a", "inputpos_b", "inputpos_c").gather(indexlist).explode(empty_as_null=False)
                )
                .collect()
                .iter_rows()
            )
            for modelgridindex, pos3_in in zip(indexlist, pos3_in_list, strict=True):
                xindex = modelgridindex % ncoordgridx
                yindex = (modelgridindex // ncoordgridx) % ncoordgridy
                zindex = (modelgridindex // (ncoordgridx * ncoordgridy)) % ncoordgridz
                pos_x_min = -xmax_tmodel + xindex * wid_init_x
                pos_y_min = -xmax_tmodel + yindex * wid_init_y
                pos_z_min = -xmax_tmodel + zindex * wid_init_z
                pos_x_mid = -xmax_tmodel + (xindex + 0.5) * wid_init_x
                pos_y_mid = -xmax_tmodel + (yindex + 0.5) * wid_init_y
                pos_z_mid = -xmax_tmodel + (zindex + 0.5) * wid_init_z

                targets = {
                    "xyz_min": (pos_x_min, pos_y_min, pos_z_min),
                    "zyx_min": (pos_z_min, pos_y_min, pos_x_min),
                    "xyz_mid": (pos_x_mid, pos_y_mid, pos_z_mid),
                    "zyx_mid": (pos_z_mid, pos_y_mid, pos_x_mid),
                }
                for key, target in targets.items():
                    if not vectormatch(pos3_in, target):
                        matched[key] = False

            if sum(matched.values()) != 1:
                msg = (
                    f"{filename}: the cell positions agree with no order of the position columns. "
                    "Make vmax consistent with the cell positions."
                )
                raise ValueError(msg)

            matchedkey = next(key for key, ismatch in matched.items() if ismatch)
            message, colrenames = posordercandidates[matchedkey]
            print(message)

            dfmodel = dfmodel.rename(colrenames, strict=False)

            if matchedkey in {"xyz_mid", "zyx_mid"}:
                dfmodel = dfmodel.with_columns(
                    pos_x_min=(pl.col("pos_x_mid") - modelmeta["wid_init_x"] / 2.0),
                    pos_y_min=(pl.col("pos_y_mid") - modelmeta["wid_init_y"] / 2.0),
                    pos_z_min=(pl.col("pos_z_mid") - modelmeta["wid_init_z"] / 2.0),
                )
    return dfmodel, modelmeta


# The version of the parquet cache format of every text source that get_text_source_cached() reads,
# which is model.txt and abundances.txt. Increase it for a change that makes an older cache file
# incorrect, e.g. a new column or a different data type in either one. Version 2 checks the cell count and the
# cell ids of model.txt, and it puts the second line of a cell beside the first line in the order of the file.
CACHEVERSION = 2


def read_parquet_cache(
    parquetfilepath: Path, textsource_mtime: float, metadatakeys: Sequence[str] = (), printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[str, str]] | None:
    """Return the cached table and its metadata, or None if the cache is absent, stale, or unreadable.

    A cache without one of metadatakeys is stale. A rejected cache stays in place: the caller rewrites
    it through write_parquet_atomic(), which replaces only the exact file that this check saw. A
    deletion here could remove the fresh cache that a rival process installed after this check.
    """
    if not parquetfilepath.is_file():
        return None

    pqmetadata, stalereason = read_parquet_cache_metadata(parquetfilepath, CACHEVERSION, textsource_mtime)
    if pqmetadata is not None and (missingkey := next((key for key in metadatakeys if key not in pqmetadata), None)):
        pqmetadata, stalereason = None, f"the file has no {missingkey} stamp"

    if pqmetadata is None:
        print(f"{parquetfilepath} is not a current cache of the text source, because {stalereason}. Will regenerate.")
        return None

    # scan_parquet resolves its schema from the same footer that gave the metadata. Thus the check
    # above already rejects a damaged file, and this code needs no further check
    try:
        df = pl.scan_parquet(parquetfilepath)
    except (pl.exceptions.PolarsError, OSError) as exc:
        print(f"Could not read {parquetfilepath} ({type(exc).__name__}: {exc}). Will regenerate.")
        return None

    if not printwarningsonly:
        print(f"Reading table from {parquetfilepath}")

    return df, pqmetadata


def get_parquet_cache_path(textfilepath: Path) -> Path:
    """Return the path of the parquet cache of a text file that get_text_source_cached reads."""
    # model_a.1.txt and model_a.2.txt must not share a cache, thus remove only a compression suffix
    textname = (
        textfilepath.name.removesuffix(textfilepath.suffix)
        if textfilepath.suffix in COMPRESSED_EXTENSIONS
        else textfilepath.name
    )
    return textfilepath.with_name(f"{textname}.parquet.tmp")


def remove_parquet_cache(textfilepath: Path) -> None:
    """Delete the parquet cache of a text file that the caller wrote again.

    The cache check accepts a modification time within MTIME_TOLERANCE_S of its stamp. Thus the time of
    a text file that the caller wrote again soon after a read cannot show that the cache is stale.
    """
    parquetfilepath = get_parquet_cache_path(textfilepath)
    if parquetfilepath.is_file():
        print(f"Deleting {parquetfilepath}, because it is the cache of the old {textfilepath.name}")
        parquetfilepath.unlink(missing_ok=True)


def get_text_source_cached(
    textfilepath: Path,
    read_text_source: Callable[[], tuple[pl.LazyFrame, dict[str, str]]],
    metadatakeys: Sequence[str] = (),
    printwarningsonly: bool = False,
    validate_metadata: Callable[[dict[str, str]], None] | None = None,
) -> tuple[pl.LazyFrame, dict[str, str]]:
    """Return the table of a text file from its parquet cache, or read the text file and write the cache.

    read_text_source returns the table and the extra metadata strings that the cache stores under
    metadatakeys. The cache is written for a text file above 2 MiB, and also for a smaller text file
    when a cache already existed, so a rejected cache is replaced and not read and rejected on each run.

    validate_metadata reads the stored metadata strings of a cache. It raises ValueError for a value
    that it cannot read, e.g. a malformed json string, and the cache is then stale.
    """
    textsource_stat = textfilepath.stat()
    textsource_mtime = textsource_stat.st_mtime
    parquetfilepath = get_parquet_cache_path(textfilepath)
    # the identity of the cache that a rewrite replaces, from the same moment as the existence check
    outdatedparquet = get_file_identity(parquetfilepath)
    hadcachefile = outdatedparquet is not None

    cached = read_parquet_cache(
        parquetfilepath, textsource_mtime, metadatakeys=metadatakeys, printwarningsonly=printwarningsonly
    )
    if cached is not None:
        if validate_metadata is None:
            return cached

        try:
            validate_metadata(cached[1])
        except ValueError as exc:
            print(f"{parquetfilepath} holds metadata that this version cannot read ({exc}). Will regenerate.")
        else:
            return cached

    df, extrametadata = read_text_source()

    mebibyte = 1024 * 1024
    if not (hadcachefile or textsource_stat.st_size > 2 * mebibyte):
        return df, extrametadata

    # a writer can replace the text file during the read. A cache of the old text would then hold a time
    # within MTIME_TOLERANCE_S of the new file. A file system can move the time with no write, thus
    # the time has that tolerance here also
    try:
        textsource_stat_after = textfilepath.stat()
    except FileNotFoundError:
        textsource_stat_after = None
    if (
        textsource_stat_after is None
        or (textsource_stat_after.st_ino, textsource_stat_after.st_size)
        != (textsource_stat.st_ino, textsource_stat.st_size)
        or abs(textsource_stat_after.st_mtime - textsource_mtime) > MTIME_TOLERANCE_S
    ):
        # the reader opens the text file more than one time, thus the header and the cells can be from
        # two versions of the file
        print(f"{textfilepath} changed during the read. Reading it again, with no write of {parquetfilepath.name}.")
        return read_text_source()

    print(f"Saving {parquetfilepath}")
    try:
        write_parquet_atomic(
            df,
            parquetfilepath,
            replaces=outdatedparquet,
            metadata={
                "creationtimeutc": str(datetime.datetime.now(datetime.UTC)),
                "cacheversion": str(CACHEVERSION),
                "textsource_mtime": str(textsource_mtime),
            }
            | extrametadata,
        )
    except PermissionError as exc:
        # a command that only reads a model must also operate on a folder that the user cannot write
        print(f"  Could not write {parquetfilepath} ({exc}). The next read parses the text file again.")
        return df, extrametadata

    print("  Done.")
    del df
    gc.collect()
    return pl.scan_parquet(parquetfilepath), extrametadata


def get_model_text_folder(modelpath: Path | str) -> Path:
    """Return the folder that holds model.txt and abundances.txt of a model path.

    A code comparison path is virtual. The ARTIS input files of such a model are in a folder of the
    data set of that project.
    """
    inputpath = Path(modelpath)
    if path_is_codecomparison(inputpath):
        _, inputmodel, _ = inputpath.parts
        return Path(get_path("codecomparisonmodelartismodelpath"), inputmodel)

    return inputpath


@on_model_host
def get_modelmeta(modelpath: Path) -> dict[str, t.Any]:
    """Return the metadata of the model, e.g. the dimensions and the time of the model.

    The host of a remote model reads the model and sends back the metadata alone.
    """
    return get_modeldata(modelpath, printwarningsonly=True)[1]


# the viewer resolves -deltalogx smallestscale at each change, and the grid of a model does not change
@modelpath_cache(maxsize=16)
@on_model_host
def get_spatial_scales(modelpath: Path) -> tuple[float, float, str]:
    """Return the smallest and the largest spatial scale of the model grid in velocity [cm/s], and a description.

    For a 1D model, the smallest and the largest spatial scale are the widths of the narrowest and the widest shell. For
    a 2D or 3D model, the smallest spatial scale is the smallest cell width along an axis. The largest spatial scale is
    the diagonal of a cell.
    """
    dfmodel, modelmeta = get_modeldata(modelpath, printwarningsonly=True)
    vmax_cmps = float(modelmeta["vmax_cmps"])
    # wid_init_* is the cell width at t_model. A width divided by t_model gives the width in velocity
    t_model_init_s = float(modelmeta["t_model_init_days"]) * day_to_s
    vmaxtext = f"the maximum velocity vmax = {vmax_cmps / km_to_cm:.0f} km/s ({vmax_cmps / C_cm_per_s:.3g}c)"
    match modelmeta["dimensions"]:
        case 1:
            shellwidth = pl.col("vel_r_max_kmps") - pl.col("vel_r_min_kmps")
            shellwidths_kmps = (
                add_derived_cols_to_modeldata(dfmodel, modelmeta)
                .select(min=shellwidth.min(), max=shellwidth.max(), shellcount=pl.len())
                .collect()
                .row(0, named=True)
            )
            smallest, largest = shellwidths_kmps["min"] * km_to_cm, shellwidths_kmps["max"] * km_to_cm
            shellcount = shellwidths_kmps["shellcount"]
            gridtext = f"The 1D model has {shellcount} shell{'' if shellcount == 1 else 's'} from 0 to {vmaxtext}"
            # the widths come from float32 velocities, thus a uniform grid has widths that differ in the last digits
            if math.isclose(smallest, largest, rel_tol=1e-3):
                scaletext = (
                    f"Each shell is {smallest / km_to_cm:.0f} km/s wide, thus this width is the smallest and the"
                    " largest spatial scale"
                )
            else:
                scaletext = (
                    f"The smallest spatial scale is the width of the narrowest shell, {smallest / km_to_cm:.0f} km/s."
                    f" The largest spatial scale is the width of the widest shell, {largest / km_to_cm:.0f} km/s"
                )
        case 2:
            ncoordgridrcyl, ncoordgridz = int(modelmeta["ncoordgridrcyl"]), int(modelmeta["ncoordgridz"])
            rcylwidth = float(modelmeta["wid_init_rcyl"]) / t_model_init_s
            zwidth = float(modelmeta["wid_init_z"]) / t_model_init_s
            smallest, largest = min(rcylwidth, zwidth), math.hypot(rcylwidth, zwidth)
            gridtext = (
                f"The 2D model has a grid of n_rcyl x n_z = {ncoordgridrcyl} x {ncoordgridz} cells, with {vmaxtext}."
                " The cylindrical radius rcyl is from 0 to vmax, and z is from -vmax to vmax. The cell widths are"
                f" width_rcyl = vmax / n_rcyl = {rcylwidth / km_to_cm:.0f} km/s and"
                f" width_z = 2 vmax / n_z = {zwidth / km_to_cm:.0f} km/s"
            )
            scaletext = (
                f"The smallest spatial scale is the smaller cell width, {smallest / km_to_cm:.0f} km/s. The largest"
                " spatial scale is the diagonal of a cell in the plane of rcyl and z,"
                f" sqrt(width_rcyl^2 + width_z^2) = {largest / km_to_cm:.0f} km/s"
            )
        case _:
            cellwidths = [float(modelmeta[f"wid_init_{axis}"]) / t_model_init_s for axis in "xyz"]
            smallest, largest = min(cellwidths), math.hypot(*cellwidths)
            gridshape = " x ".join(str(modelmeta[f"ncoordgrid{axis}"]) for axis in "xyz")
            gridtext = (
                f"The 3D model has a grid of {gridshape} cells from -vmax to vmax, with {vmaxtext}. The cell widths"
                " width_x, width_y, and width_z are 2 vmax / n for n cells on that axis"
            )
            scaletext = (
                f"The smallest spatial scale is the smallest cell width, {smallest / km_to_cm:.0f} km/s. The largest"
                " spatial scale is the diagonal of a cell, sqrt(width_x^2 + width_y^2 + width_z^2) ="
                f" {largest / km_to_cm:.0f} km/s. This is the largest range of line-of-sight velocity in one cell"
            )
    return smallest, largest, f"{gridtext}. {scaletext}"


def get_modeldata(
    modelpath: Path | str = ".", get_elemabundances: bool = False, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Read the velocities, the densities, and the mass fractions of the radioactive nuclides of the cells in model.txt.

    Returns dfmodel, modelmeta
        - dfmodel: a polars LazyFrame with a row for each cell, and the columns of the model file.
          add_derived_cols_to_modeldata adds the other columns, e.g. the volume and the mass of each cell.
        - modelmeta: a dictionary of the model parameters, e.g. the keys t_model_init_days, vmax_cmps, and dimensions.

    Parameters
    ----------
    modelpath : Path | str
        either a path to model.txt file, or a folder containing model.txt
    get_elemabundances : bool
        also read elemental abundances (from abundances.txt) and merge with the output DataFrame
    printwarningsonly : bool
        if True, print warnings but skip informational progress messages

    """
    check_local_path(modelpath)
    inputpath = Path(modelpath)

    if inputpath.is_dir():
        modelpath = inputpath
        textfilepath = firstexisting("model.txt", folder=inputpath, tryzipped=True)
    elif inputpath.is_file() or find_compressed(inputpath) is not None:
        # a file name, e.g. model_1d.txt, also names its compressed copy, e.g. model_1d.txt.zst
        textfilepath = firstexisting(inputpath.name, folder=inputpath.parent, tryzipped=True)
        modelpath = inputpath.parent
    elif path_is_codecomparison(inputpath):
        modelpath = inputpath
        textfilepath = Path(get_model_text_folder(inputpath), "model.txt")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputpath)

    def read_text() -> tuple[pl.LazyFrame, dict[str, str]]:
        dfmodel, modelmeta = read_modelfile_text(filename=textfilepath, printwarningsonly=printwarningsonly)
        return dfmodel, {"modelmeta_json": json.dumps(modelmeta)}

    def check_modelmeta(metadata: dict[str, str]) -> None:
        json.loads(metadata["modelmeta_json"])

    dfmodel, metadata = get_text_source_cached(
        Path(textfilepath),
        read_text,
        metadatakeys=["modelmeta_json"],
        printwarningsonly=printwarningsonly,
        validate_metadata=check_modelmeta,
    )
    modelmeta: dict[str, t.Any] = json.loads(metadata["modelmeta_json"])

    if not printwarningsonly:
        print(f"  model is {modelmeta['dimensions']}D with {modelmeta['npts_model']} cells")

    # the reader checks that the ids start at 0 or 1 and increase by one, and it sorts the cells by id. sn3d
    # takes the id of the first cell as the start of the cell index
    firstinputcellid = dfmodel.select(pl.col("inputcellid").first()).collect().item()

    if get_elemabundances:
        abundancedata = get_initelemabundances(modelpath, printwarningsonly=printwarningsonly)
        dfmodel = dfmodel.join(abundancedata, how="inner", on="inputcellid", maintain_order="left")

    dfmodel = dfmodel.with_columns(pl.col("inputcellid").sub(firstinputcellid).alias("modelgridindex"))

    return dfmodel, modelmeta


def get_middle_layer_lower_edge(dfmodel: pl.DataFrame, axis: str, positive: bool) -> float:
    """Return the lower edge of the layer of cells that touches the origin on the positive or negative side of axis.

    The centre layer of an odd grid holds the origin, thus both sides give that layer.
    """
    # select the layer by index, because an edge at the origin can have a rounding error of either sign
    loweredges = dfmodel[f"pos_{axis}_min"].unique().sort()
    return float(loweredges.item(loweredges.len() // 2 if positive else (loweredges.len() - 1) // 2))


def min_abs_coordinate(ax: str) -> pl.Expr:
    """Get the smallest |coordinate| reached anywhere inside a cell along axis ax.

    A cell that straddles the axis (pos_min < 0 < pos_max) contains the origin plane, so its closest approach to that
    plane is zero rather than min(|pos_min|, |pos_max|).
    """
    pos_min = pl.col(f"pos_{ax}_min")
    pos_max = pl.col(f"pos_{ax}_max")

    return pl.when(pos_min * pos_max < 0.0).then(pl.lit(0.0)).otherwise(pl.min_horizontal(pos_min.abs(), pos_max.abs()))


# the clamp at -99 matches save_modeldata, which treats -99 as the empty-cell marker. A plain log10() gives -inf for
# rho == 0, which cannot go into model.txt
LOGRHO_FROM_RHO = pl.when(pl.col("rho") > 0).then(pl.max_horizontal(-99, pl.col("rho").log10())).otherwise(-99.0)
RHO_FROM_LOGRHO = pl.when(pl.col("logrho") > -98).then(10 ** pl.col("logrho")).otherwise(0.0)


def add_derived_cols_to_modeldata(dfmodel: pl.DataFrame | pl.LazyFrame, modelmeta: dict[str, t.Any]) -> pl.LazyFrame:
    """Add each column that follows from the columns of the model file, e.g. the volume and the mass of each cell.

    The function calculates each derived column again, also one that the dataframe already holds. Thus a call after
    a change to a column of the model file gives current values. Call this function last in the chain that selects
    the columns. The dataframe is lazy, thus the query then calculates only the columns that it selects. In 1D, the
    inner velocity of a shell comes from the row before it, thus call this function before a filter of the rows.
    """
    dfmodel = dfmodel.lazy()
    original_cols = dfmodel.collect_schema().names()

    t_model_init_seconds = modelmeta["t_model_init_days"] * day_to_s
    dimensions = modelmeta["dimensions"]

    # rho is the source of logrho, because a caller changes the density as rho. A 1D model file gives logrho only,
    # thus a dataframe without rho gets it from logrho first
    if "rho" not in original_cols:
        dfmodel = dfmodel.with_columns(rho=RHO_FROM_LOGRHO)
    dfmodel = dfmodel.with_columns(logrho=LOGRHO_FROM_RHO)

    axes: list[str] = []
    match dimensions:
        case 1:
            axes = ["r"]

            dfmodel = (
                dfmodel
                .with_columns(vel_r_min_kmps=pl.col("vel_r_max_kmps").shift(n=1, fill_value=0.0))
                .with_columns(
                    vel_r_min=(pl.col("vel_r_min_kmps") * km_to_cm), vel_r_max=(pl.col("vel_r_max_kmps") * km_to_cm)
                )
                .with_columns(vel_r_mid=((pl.col("vel_r_max") + pl.col("vel_r_min")) / 2))
                .with_columns(
                    volume=(
                        (4.0 / 3.0)
                        * math.pi
                        * (
                            pl.col("vel_r_max_kmps").cast(pl.Float64).pow(3)
                            - pl.col("vel_r_min_kmps").cast(pl.Float64).pow(3)
                        )
                        * (km_to_cm * t_model_init_seconds) ** 3
                    )
                )
                .with_columns(  # 1/2 m v^2 integrated across each spherical shell's vmin to vmax
                    kinetic_en_erg_r=2.0
                    / 5.0
                    * math.pi
                    * pl.col("rho")
                    * t_model_init_seconds**3
                    * (pl.col("vel_r_max").cast(pl.Float64).pow(5) - pl.col("vel_r_min").cast(pl.Float64).pow(5))
                )
            )

        case 2:
            axes = ["rcyl", "z"]

            assert t_model_init_seconds is not None
            # pos_mid is defined in the input file
            dfmodel = dfmodel.with_columns([
                (pl.col(f"pos_{ax}_mid") - modelmeta[f"wid_init_{ax}"] / 2.0).alias(f"pos_{ax}_min") for ax in axes
            ]).with_columns([
                (pl.col(f"pos_{ax}_mid") + modelmeta[f"wid_init_{ax}"] / 2.0).alias(f"pos_{ax}_max") for ax in axes
            ])

            # add a 3D radius column
            axes.append("r")
            dfmodel = dfmodel.with_columns(
                pos_r_min=(pl.col("pos_rcyl_min").pow(2) + min_abs_coordinate("z").pow(2)).sqrt(),
                pos_r_mid=(pl.col("pos_rcyl_mid").pow(2) + pl.col("pos_z_mid").pow(2)).sqrt(),
                pos_r_max=(
                    pl.col("pos_rcyl_max").pow(2)
                    + pl.max_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
                ).sqrt(),
                volume=(
                    math.pi
                    * (pl.col("pos_rcyl_max").cast(pl.Float64).pow(2) - pl.col("pos_rcyl_min").cast(pl.Float64).pow(2))
                    * modelmeta["wid_init_z"]
                ),
            ).with_columns(
                # two components of kinetic energy: 1/2 m v^2 in cylindrical and z directions
                kinetic_en_erg_rcyl=(
                    1
                    / 4
                    * math.pi
                    * pl.col("rho")
                    * t_model_init_seconds**-2
                    * modelmeta["wid_init_z"]
                    * (pl.col("pos_rcyl_max").cast(pl.Float64).pow(4) - pl.col("pos_rcyl_min").cast(pl.Float64).pow(4))
                ),
                kinetic_en_erg_z=(
                    1
                    / 6
                    * pl.col("rho")
                    * math.pi
                    * (pl.col("pos_rcyl_max").cast(pl.Float64).pow(2) - pl.col("pos_rcyl_min").cast(pl.Float64).pow(2))
                    * t_model_init_seconds**-2
                    * (pl.col("pos_z_max").cast(pl.Float64).pow(3) - pl.col("pos_z_min").cast(pl.Float64).pow(3))
                ),
            )

        case 3:
            axes = ["x", "y", "z"]
            for ax in axes:
                if f"wid_init_{ax}" not in modelmeta:
                    modelmeta[f"wid_init_{ax}"] = modelmeta["wid_init"]

            dfmodel = (
                dfmodel
                .with_columns(
                    volume=pl.lit(modelmeta["wid_init_x"] * modelmeta["wid_init_y"] * modelmeta["wid_init_z"])
                )
                .with_columns([
                    (pl.col(f"pos_{ax}_min") + 0.5 * modelmeta[f"wid_init_{ax}"]).alias(f"pos_{ax}_mid") for ax in axes
                ])
                .with_columns([
                    (pl.col(f"pos_{ax}_min") + modelmeta[f"wid_init_{ax}"]).alias(f"pos_{ax}_max") for ax in axes
                ])
            )

            # add a 3D radius column
            axes.append("r")

            # xyz positions can be negative, so the min xyz side of the cube can have a larger radius than the max side
            dfmodel = dfmodel.with_columns(
                pos_r_min=(
                    min_abs_coordinate("x").pow(2) + min_abs_coordinate("y").pow(2) + min_abs_coordinate("z").pow(2)
                ).sqrt(),
                pos_r_mid=(pl.col("pos_x_mid").pow(2) + pl.col("pos_y_mid").pow(2) + pl.col("pos_z_mid").pow(2)).sqrt(),
                pos_r_max=(
                    pl.max_horizontal(pl.col("pos_x_min").abs(), pl.col("pos_x_max").abs()).pow(2)
                    + pl.max_horizontal(pl.col("pos_y_min").abs(), pl.col("pos_y_max").abs()).pow(2)
                    + pl.max_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
                ).sqrt(),
            ).with_columns(
                (
                    1.0
                    / 6.0
                    * pl.col("rho")
                    * modelmeta[f"wid_init_{ax1}"]
                    * modelmeta[f"wid_init_{ax2}"]
                    * t_model_init_seconds**-2
                    * (
                        pl.col(f"pos_{ax3}_max").cast(pl.Float64).pow(3)
                        - pl.col(f"pos_{ax3}_min").cast(pl.Float64).pow(3)
                    )
                ).alias(f"kinetic_en_erg_{ax3}")
                for ax1, ax2, ax3 in (("x", "y", "z"), ("y", "z", "x"), ("z", "x", "y"))
            )

        case _:
            msg = f"Unhandled model dimensions: {dimensions}"
            raise ValueError(msg)

    # get total kinetic energy from orthogonal components. Every coordinate system also gets a radial component, which
    # would double-count the orthogonal ones, so only use "r" for 1D models where it is the sole axis
    orthogonal_axes = ["r"] if dimensions == 1 else [ax for ax in axes if ax != "r"]
    dfmodel = dfmodel.with_columns(
        kinetic_en_erg=(pl.sum_horizontal(pl.col(f"kinetic_en_erg_{ax}") for ax in orthogonal_axes))
    )

    poscols = [col for col in dfmodel.collect_schema().names() if col.startswith("pos_")]
    return (
        dfmodel
        .with_columns((pl.col(col) / t_model_init_seconds).alias(col.replace("pos_", "vel_")) for col in poscols)
        .with_columns(mass_g=(pl.col("rho") * pl.col("volume")))
        # the vel_*_kmps columns are in km/s and not in cm/s. The _on_c columns of an earlier call also start
        # with vel_. Thus the scale to c leaves out both groups
        .with_columns(((cs.starts_with("vel_") - cs.ends_with("_kmps", "_on_c")) / C_cm_per_s).name.suffix("_on_c"))
    )


def get_standard_columns(dimensions: int, includenico57: bool = False, pos_unknown: bool = False) -> list[str]:
    """Get standard (artis classic) columns for modeldata DataFrame."""
    cols: list[str] = []
    match dimensions:
        case 1:
            cols = ["inputcellid", "vel_r_max_kmps", "logrho"]
        case 2:
            cols = ["inputcellid", "pos_rcyl_mid", "pos_z_mid", "rho"]
        case 3:
            cols = (
                ["inputcellid", "inputpos_a", "inputpos_b", "inputpos_c", "rho"]
                if pos_unknown
                else ["inputcellid", "pos_x_min", "pos_y_min", "pos_z_min", "rho"]
            )
        case _:
            msg = f"Unhandled model dimensions: {dimensions}"
            raise ValueError(msg)

    cols += ["X_Fegroup", "X_Ni56", "X_Co56", "X_Fe52", "X_Cr48"]

    if includenico57:
        cols += ["X_Ni57", "X_Co57"]

    return cols


def customcolsortkey(col: str) -> tuple[float, int]:
    """Sort nuclide mass fraction columns by atomic number then mass number, and other columns last."""
    return get_z_a_nucname(col) if col.startswith("X_") else (math.inf, 0)


def write_artis_csv(df: pl.DataFrame, fileobj: t.IO[bytes]) -> None:
    """Append the dataframe as one zstd frame in the ARTIS text file format: space separated and no header.

    Eight significant figures round-trip the Float32 that the reader produces, whose relative spacing
    is 6e-8. Five figures lost more precision than the reader does, which showed up as a mass that
    changed by 2e-5 when a model was written and read again.
    """
    df.write_csv(
        fileobj,
        include_header=False,
        separator=" ",
        line_terminator="\n",
        float_scientific=True,
        float_precision=7,
        null_value="0.0",
        compression="zstd",
    )


# the files of an ARTIS input model that a command writes. confirm_overwrite also finds the compressed copies
MODEL_FILE_NAMES = ("model.txt", "abundances.txt", "gridcontributions.txt")


def remove_other_copies(filepath: Path) -> None:
    """Delete each plain or compressed copy of filepath other than filepath itself.

    If model.txt and model.txt.zst both exist, a reader uses model.txt. Thus an old plain copy must not stay beside a
    new compressed file.
    """
    for oldpath in get_file_copies(filepath):
        if oldpath != filepath and oldpath.is_file():
            oldpath.unlink()
            print(f"Deleted {oldpath}, because {filepath.name} replaces it")


def save_modeldata(
    dfmodel: pl.LazyFrame | pl.DataFrame,
    outpath: Path | str | None = None,
    modelmeta: dict[str, t.Any] | None = None,
    extracols: Sequence[str] = (),
    **kwargs: t.Any,
) -> None:
    """Write model.txt, a snapshot of the density and the composition, from the cell properties and the metadata.

    The metadata gives values such as the time after the explosion. The file gets zstd compression, and the extension
    .zst, e.g. model.txt.zst for model.txt or for model.txt.gz. ARTIS reads such a file directly.

    1D
    -------
    dfmodel must contain columns inputcellid, vel_r_max_kmps, logrho, X_Fegroup, X_Ni56, X_Co56, X_Fe52, X_Cr48
    modelmeta is not required

    2D
    -------
    dfmodel must contain columns inputcellid, pos_rcyl_mid, pos_z_mid, rho, X_Fegroup, X_Ni56, X_Co56, X_Fe52, X_Cr48
    modelmeta must define: vmax_cmps, ncoordgridrcyl and ncoordgridz

    3D
    -------
    dfmodel must contain columns inputcellid, pos_x_min, pos_y_min, pos_z_min, rho, X_Fegroup, X_Ni56, X_Co56,
    X_Fe52, X_Cr48
    modelmeta must define: vmax_cmps

    model.txt holds these comments:

    - a comment line that gives the creation time;
    - a comment line that gives the column units;
    - an inline comment after each header value;
    - a comment line that gives the column names.

    For a 1D model or a 2D model, sn3d reads these column names from v2024.04.

    The function deletes the parquet cache of the file that it writes. A LazyFrame from get_modeldata of the same
    file can scan that cache, thus collect such a LazyFrame before this call.

    model.txt gets the standard columns, each X_ column, and the custom columns that ARTIS reads (Ye, q, and
    tracercount) if dfmodel holds them. A caller names each other custom column in extracols. model.txt gets no
    other column, e.g. no derived column.
    """
    assert isinstance(dfmodel, (pl.LazyFrame, pl.DataFrame))
    colnames_in = dfmodel.collect_schema().names()
    if "inputcellid" not in colnames_in and "modelgridindex" in colnames_in:
        dfmodel = dfmodel.with_columns(inputcellid=pl.col("modelgridindex") + 1)

    if modelmeta is None:
        modelmeta = {}

    assert all(
        key not in modelmeta or modelmeta[key] == kwargs[key] for key in kwargs
    )  # can't define the same thing twice unless the values are the same

    modelmeta |= kwargs  # add any extra keyword arguments to modelmeta

    headercommentlines = modelmeta.get("headercommentlines")
    vmax = modelmeta.get("vmax_cmps")

    if modelmeta.get("dimensions") is None:
        modelmeta["dimensions"] = get_dfmodel_dimensions(dfmodel)

    if modelmeta["dimensions"] not in {1, 2, 3}:
        msg = f"dimensions must be 1, 2, or 3, not {modelmeta['dimensions']}"
        raise ValueError(msg)

    colnames = dfmodel.collect_schema().names()

    # the Ni57 and Co57 columns are optional, but their position counts. They must come before
    # every other custom column
    standardcols = get_standard_columns(
        modelmeta["dimensions"], includenico57=("X_Ni57" in colnames or "X_Co57" in colnames)
    )
    writtencustomcols = {"Ye", "q", "tracercount", *extracols}
    customcols = sorted(
        (col for col in colnames if col not in standardcols and (col.startswith("X_") or col in writtencustomcols)),
        key=customcolsortkey,
    )

    # the select comes before the collect, thus a lazy query does not calculate the columns that model.txt does not get
    dfmodel = (
        dfmodel
        .lazy()
        .with_columns(pl.lit(0.0).alias(col) for col in standardcols if col.startswith("X_") and col not in colnames)
        .select(*standardcols, *customcols)
        .with_columns(pl.col("inputcellid").cast(pl.Int32))
        .collect()
    )

    dfmodel_npts_model = dfmodel.height
    if "npts_model" in modelmeta:
        assert modelmeta["npts_model"] == dfmodel_npts_model
    else:
        modelmeta["npts_model"] = dfmodel_npts_model

    timestart = time.perf_counter()
    tmodelline = (str(modelmeta["t_model_init_days"]), "t_model_init_days: time of the snapshot [day]")
    if modelmeta["dimensions"] == 1:
        print(f" 1D grid radial bins: {dfmodel_npts_model}")
        strunits = "vel_r_max_kmps [km/s], logrho = log10(rho [g/cm^3]) at t_model_init_days"
        headerlines = [(str(dfmodel_npts_model), "npts_model: number of radial cells"), tmodelline]

    elif modelmeta["dimensions"] == 2:
        print(f" 2D grid size: {dfmodel_npts_model} ({modelmeta['ncoordgridrcyl']} x {modelmeta['ncoordgridz']})")
        assert modelmeta["ncoordgridrcyl"] * modelmeta["ncoordgridz"] == dfmodel_npts_model
        strunits = "pos_rcyl_mid and pos_z_mid [cm], rho [g/cm^3], all at t_model_init_days"
        headerlines = [
            (
                f"{modelmeta['ncoordgridrcyl']} {modelmeta['ncoordgridz']}",
                "ncoordgridrcyl ncoordgridz: number of cells along the cylindrical radius and along the z axis",
            ),
            tmodelline,
            (f"{vmax:.8e}", "vmax_cmps: maximum velocity along the radius and the z axis [cm/s]"),
        ]

    else:
        griddimension = round(dfmodel_npts_model ** (1.0 / 3.0))
        print(f" 3D grid size: {dfmodel_npts_model} ({griddimension}^3)")
        assert griddimension**3 == dfmodel_npts_model
        strunits = "pos_x_min, pos_y_min, and pos_z_min [cm], rho [g/cm^3], all at t_model_init_days"
        headerlines = [
            (str(dfmodel_npts_model), f"npts_model: number of cells ({griddimension}^3 Cartesian grid)"),
            tmodelline,
            (f"{vmax:.8e}", "vmax_cmps: maximum velocity along each axis [cm/s]"),
        ]

    modelfilepath = with_compressed_extension(get_plain_path(resolve_outputfile(outpath, "model.txt")), ".zst")

    remove_other_copies(modelfilepath)
    # a write that stops early must not leave the cache of the old file beside a part of the new file
    remove_parquet_cache(modelfilepath)

    with modelfilepath.open("wb") as fmodel:
        # sn3d reads the first comment line after the header values as the column names, thus each
        # other comment line comes before those values. sn3d reads the numbers at the start of a header
        # line, thus an inline comment can follow them
        headertextlines = [
            *[f"# {line}" for line in headercommentlines or []],
            get_created_comment(),
            f"# {UNITS_COMMENT_PREFIX} {strunits}. {UNITS_COMMENT_END}",
            *[f"{strvalue:<24} # {comment}" for strvalue, comment in headerlines],
            f"#{' '.join([*standardcols, *customcols])}",
        ]

        abundandcustomcols = [*[col for col in standardcols if col.startswith("X_")], *customcols]

        isintcol = [not dfmodel.schema[col].is_float() for col in abundandcustomcols]
        ismassfraccol = [col.startswith("X_") for col in abundandcustomcols]
        strzeroabund = " ".join(["0" if isint else "0.0" for isint in isintcol])
        if modelmeta["dimensions"] == 1:
            celllines = []
            for inputcellid, vel_r_max_kmps, logrho, *abundandcustomcolvals in dfmodel.select([
                "inputcellid",
                "vel_r_max_kmps",
                "logrho",
                *abundandcustomcols,
            ]).iter_rows():
                # write eight significant figures, because write_artis_csv gives the same precision to
                # the other dimensions. A null or NaN becomes zero, as write_artis_csv writes a null. A negative
                # custom value keeps its sign, but a negative mass fraction, e.g. from the noise of an
                # interpolation, becomes zero, because ARTIS needs a valid composition
                strabundandcustom = (
                    " ".join([
                        (
                            (f"{colvalue:d}" if isint else f"{colvalue:.7e}")
                            if colvalue is not None
                            and not math.isnan(colvalue)
                            and (colvalue > 0 if ismassfrac else colvalue != 0)
                            else ("0" if isint else "0.0")
                        )
                        for colvalue, isint, ismassfrac in zip(
                            abundandcustomcolvals, isintcol, ismassfraccol, strict=True
                        )
                    ])
                    if logrho > -99.0
                    else strzeroabund
                )
                celllines.append(f"{inputcellid:d} {vel_r_max_kmps:9.2f} {logrho:10.8f} {strabundandcustom}")

            write_zstd_lines(fmodel, [*headertextlines, *celllines])

        else:
            # startcols are the standard ones, but excluding any abundances
            startcols = [col for col in standardcols if not col.startswith("X_")]
            dfmodel = dfmodel.select([*startcols, *abundandcustomcols])
            # fast polars writer
            # set abundances to null for cells with zero density (so that shorter form "0.0" can be written)
            dfmodel = dfmodel.with_columns(
                pl.when(pl.col("rho") > 0).then(pl.col(col)).otherwise(pl.lit(None)).alias(col)
                for col in dfmodel.columns
                if not col.startswith("pos") and col != "inputcellid" and dfmodel.schema[col].is_float()
            )
            write_zstd_lines(fmodel, headertextlines)
            write_artis_csv(dfmodel, fmodel)

    remove_parquet_cache(modelfilepath)
    print(f"Wrote {modelfilepath} (took {time.perf_counter() - timestart:.1f} seconds)")


def get_mgi_of_velocity_kms(modelpath: Path, velocity: float) -> int | None:
    """Return the modelgridindex of the cell whose outer velocity brackets the given velocity."""
    if np.isnan(velocity):
        return None
    dfmodel, modelmeta = get_modeldata(modelpath)
    assert modelmeta["dimensions"] == 1, "get_mgi_of_velocity_kms only works for 1D models"
    arr_vouter = dfmodel.select("vel_r_max_kmps").collect().to_series().to_numpy()

    mgi_upper = int(np.searchsorted(arr_vouter, velocity))
    if mgi_upper >= len(arr_vouter):
        msg = f"Velocity {velocity} is larger than all cell outer velocities. Velocity list: {arr_vouter}"
        raise AssertionError(msg)
    assert arr_vouter[mgi_upper] >= velocity if mgi_upper < len(arr_vouter) else True
    assert arr_vouter[mgi_upper - 1] < velocity if mgi_upper > 0 else True
    return mgi_upper


def read_onespace_abundances(textfilepath: Path, ncolumns: int) -> pl.DataFrame | None:
    """Return the table of an abundances.txt that has one space between two values, or None.

    pl.read_csv is much faster than read_wsv, but it cannot read a run of spaces. A result of None shows
    that the caller must use read_wsv.
    """
    colnames = ["inputcellid", *[f"X_{get_elsymbol(z)}" for z in range(1, ncolumns)]]
    try:
        abundancedata = pl.read_csv(
            polars_source(textfilepath),
            separator=" ",
            has_header=False,
            comment_prefix="#",
            schema={col: pl.Int32 if col == "inputcellid" else pl.Float32 for col in colnames},
            # read_wsv gives null and not NaN for these tokens, thus this reader must do the same
            null_values=["nan", "NaN", "-nan", "-NaN", "NA", "N/A", "null", "NULL"],
        )
    except pl.exceptions.ComputeError:
        return None

    # an empty line gives a row of nulls, which read_wsv drops
    abundancedata = abundancedata.filter(~pl.all_horizontal(pl.all().is_null()))
    # a line with more spaces or with fewer values also gives a null, and read_wsv decides such a file
    return None if abundancedata.null_count().sum_horizontal().item() > 0 else abundancedata


def get_initelemabundances(modelpath: Path | str = ".", printwarningsonly: bool = False) -> pl.LazyFrame:
    """Return a table of elemental mass fractions by cell from abundances."""
    textfilepath = firstexisting("abundances.txt", folder=get_model_text_folder(modelpath), tryzipped=True)

    def read_text() -> tuple[pl.LazyFrame, dict[str, str]]:
        if not printwarningsonly:
            print(f"Reading {textfilepath}")

        with zopen(textfilepath) as fabund:
            firstdataline = next((line for line in fabund if line.strip() and not line.startswith("#")), "")

        # save_initelemabundances writes one space between two values
        abundancedata = (
            read_onespace_abundances(Path(textfilepath), len(firstdataline.split()))
            if firstdataline == " ".join(firstdataline.split()) + "\n"
            else None
        )
        if abundancedata is None:
            if not printwarningsonly:
                print("  read_wsv reads this file, because pl.read_csv cannot read its format")
            abundancedata = read_wsv(textfilepath, has_header=False, comment_prefix="#")
            colnames = ["inputcellid", *[f"X_{get_elsymbol(z)}" for z in range(1, len(abundancedata.columns))]]
            abundancedata = abundancedata.rename({
                col: colnames[idx] for idx, col in enumerate(abundancedata.columns)
            }).with_columns(cs.starts_with("X_").cast(pl.Float32), (~cs.starts_with("X_")).cast(pl.Int32))

        return abundancedata.lazy(), {}

    return get_text_source_cached(Path(textfilepath), read_text, printwarningsonly=printwarningsonly)[0]


def save_initelemabundances(
    dfelabundances: pl.DataFrame | pl.LazyFrame,
    outpath: Path | str | None = None,
    headercommentlines: Sequence[str] | None = None,
) -> None:
    """Save a DataFrame in the format of get_initelemabundances to abundances.txt.zst.

    The file name gets the extension .zst, as in save_modeldata.

    columns must be:
        - inputcellid: integer index to match model.txt (starting from 1)
        - X_i: mass fraction of element with two-letter code 'i' (e.g., X_H, X_He, H_Li, ...).
    """
    timestart = time.perf_counter()

    abundancefilename = with_compressed_extension(get_plain_path(resolve_outputfile(outpath, "abundances.txt")), ".zst")

    dfelabundances = (
        dfelabundances.lazy().with_columns([pl.col("inputcellid").cast(pl.Int32)]).sort("inputcellid").collect()
    )
    assert isinstance(dfelabundances, pl.DataFrame)

    assert dfelabundances["inputcellid"].min() == 1
    assert dfelabundances["inputcellid"].max() == len(dfelabundances)

    atomic_numbers = {
        get_atomic_number(colname.removeprefix("X_")) for colname in dfelabundances.select(cs.starts_with("X_")).columns
    }
    max_atomic_number = max([30, *atomic_numbers])
    elcolnames = [f"X_{get_elsymbol(Z)}" for Z in range(1, 1 + max_atomic_number)]
    for colname in elcolnames:
        if colname not in dfelabundances.columns:
            dfelabundances = dfelabundances.with_columns(pl.lit(0.0).alias(colname))

    dfelabundances = dfelabundances.select(["inputcellid", *elcolnames])

    remove_other_copies(abundancefilename)
    remove_parquet_cache(abundancefilename)

    with abundancefilename.open("wb") as fabund:
        # sn3d and get_initelemabundances skip each comment line, and both read the columns by position
        write_zstd_lines(
            fabund,
            [
                *[f"# {line}" for line in headercommentlines or []],
                get_created_comment(),
                f"# {UNITS_COMMENT_PREFIX} each X_ column is the mass fraction of an element",
                f"#{' '.join(dfelabundances.columns)}",
            ],
        )
        write_artis_csv(dfelabundances, fabund)

    remove_parquet_cache(abundancefilename)
    print(f"wrote {abundancefilename} (took {time.perf_counter() - timestart:.1f} seconds)")


def save_empty_abundance_file(npts_model: int, outputfilepath: str | Path = ".") -> None:
    """Save dummy abundance file with only zeros."""
    save_initelemabundances(pl.DataFrame({"inputcellid": range(1, npts_model + 1)}), outpath=outputfilepath)


def get_dfmodel_dimensions(dfmodel: pl.DataFrame | pl.LazyFrame) -> int:
    """Guess whether the model is 1D, 2D, or 3D based on which columns are present."""
    columns = dfmodel.collect_schema().names()
    if "pos_x_min" in columns:
        return 3

    return 2 if "pos_z_mid" in columns else 1


def remap_gridcontributions(
    dfgridcontributions: pl.DataFrame | pl.LazyFrame, dfoutcell_inputcells_masses: pl.DataFrame | pl.LazyFrame
) -> pl.LazyFrame:
    """Return the particle contributions on a new grid, with a weight from the mass of each cell.

    dfoutcell_inputcells_masses maps the inputcellid of each cell of the old grid to the
    out_inputcellid of its cell on the new grid. It also gives the mass_g of the old cell and the
    out_mass_g of the new one. Thus frac_of_cellmass counts against the mass of the new cell.
    """
    return (
        dfgridcontributions
        .lazy()
        .with_columns(pl.col("cellindex").cast(pl.Int32))
        .rename({"cellindex": "inputcellid"})
        .join(dfoutcell_inputcells_masses.lazy(), on="inputcellid", how="left")
        .drop("inputcellid")
        .group_by("out_inputcellid", "particleid")
        .agg((cs.starts_with("frac_").dot(pl.col("mass_g")) / pl.col("out_mass_g").first()).fill_nan(0.0))
        .rename({"out_inputcellid": "cellindex"})
        .drop_nulls("cellindex")
        .sort("cellindex", "particleid")
        .select(
            "particleid",
            "cellindex",
            "frac_of_cellmass",
            cs.by_name("frac_of_cellmass_includemissing", require_all=False),
        )
    )


def dimension_reduce_model(
    dfmodel: pl.DataFrame | pl.LazyFrame,
    outputdimensions: int,
    dfelabundances: pl.DataFrame | pl.LazyFrame | None = None,
    dfgridcontributions: pl.DataFrame | None = None,
    modelmeta: dict[str, t.Any] | None = None,
    **kwargs: t.Any,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, t.Any]]:
    """Convert a 3D Cartesian grid model to a 1D spherical model or a 2D cylindrical model.

    The function can also change the particle gridcontributions and the table of the elemental abundances to agree
    with the new model.
    """
    assert outputdimensions in {0, 1, 2}

    dfmodel = dfmodel.lazy()

    if modelmeta is None:
        modelmeta = {}

    modelmeta_out = {k: v for k, v in modelmeta.items() if not k.startswith(("ncoord", "wid_init"))}

    assert all(
        key not in modelmeta_out or modelmeta_out[key] == kwargs[key] for key in kwargs
    )  # can't define the same thing twice unless the values are the same

    modelmeta_out |= kwargs  # add any extra keyword arguments to modelmeta

    t_model_init_seconds = modelmeta["t_model_init_days"] * 24 * 60 * 60
    vmax = modelmeta["vmax_cmps"]
    xmax = vmax * t_model_init_seconds

    ndim_in = modelmeta["dimensions"]
    assert ndim_in > outputdimensions
    modelmeta_out["dimensions"] = max(outputdimensions, 1)

    in_ngridpoints = modelmeta.get("npts_model", dfmodel.select(pl.len()).collect().item())
    assert isinstance(in_ngridpoints, int)
    assert in_ngridpoints > 0

    print(f"Resampling {ndim_in:d}D model with {in_ngridpoints} cells to {outputdimensions}D...")
    timestart = time.perf_counter()

    # the aggregation below makes a list from each column of an unknown kind. Thus, only the velocities and the mass
    # go into the output with the columns of the input model file
    inputcols = dfmodel.collect_schema().names()
    dfmodel_out = add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta).select(
        cs.by_name(inputcols) | cs.starts_with("vel_") | cs.by_name("mass_g")
    )

    if outputdimensions == 0:
        ncoordgridr = 1
        ncoordgridz = 1
    elif outputdimensions == 1:
        # make 1D model
        if ndim_in == 2:
            ncoordgridr = int(modelmeta.get("ncoordgridrcyl", round(math.sqrt(in_ngridpoints / 2.0))))
        elif ndim_in == 3:
            ncoordgridx = int(modelmeta.get("ncoordgridx", round(math.cbrt(in_ngridpoints))))
            ncoordgridr = int(ncoordgridx / 2.0)
        else:
            ncoordgridr = 1
        modelmeta_out["ncoordgridr"] = ncoordgridr
        ncoordgridz = 1
    elif outputdimensions == 2:
        dfmodel_out = dfmodel_out.with_columns([
            (pl.col("vel_x_mid") ** 2 + pl.col("vel_y_mid") ** 2).sqrt().alias("vel_rcyl_mid")
        ])
        ncoordgridz = int(modelmeta.get("ncoordgridx", round(math.cbrt(in_ngridpoints))))
        assert ncoordgridz % 2 == 0
        ncoordgridr = ncoordgridz // 2
        modelmeta_out["ncoordgridz"] = ncoordgridz
        modelmeta_out["ncoordgridrcyl"] = ncoordgridr
        modelmeta_out["wid_init_z"] = 2 * xmax / ncoordgridz
        modelmeta_out["wid_init_rcyl"] = xmax / ncoordgridr
    else:
        msg = f"Invalid outputdimensions: {outputdimensions}"
        raise ValueError(msg)

    # velocities in cm/s
    vel_z_bins = [-vmax + 2 * vmax * n / ncoordgridz for n in range(ncoordgridz + 1)]

    # "r" is the cylindrical radius in 2D, or the spherical radius in 1D
    vel_r_bins = [vmax * n / ncoordgridr for n in range(ncoordgridr + 1)]

    col_vel_r = pl.col("vel_rcyl_mid") if outputdimensions == 2 else pl.col("vel_r_mid")
    # the bins are closed on the left, because the centre cell of an odd grid has a mid-point velocity of zero.
    # A bin that is closed on the right puts that cell below the first bin, and the filter then drops its mass
    dfmodel_out = dfmodel_out.with_columns(get_bin_index_expr(col_vel_r, vel_r_bins).alias("out_n_r")).filter(
        pl.col("out_n_r").is_between(0, ncoordgridr - 1)
    )

    if outputdimensions == 2:
        dfmodel_out = (
            dfmodel_out
            .with_columns(get_bin_index_expr(pl.col("vel_z_mid"), vel_z_bins).alias("out_n_z"))
            .filter(
                pl.col("out_n_r").is_between(0, ncoordgridr - 1) & (pl.col("out_n_z").is_between(0, ncoordgridz - 1))
            )
            .with_columns(mgiout=pl.col("out_n_z") * ncoordgridr + pl.col("out_n_r"))
        )
    else:
        assert outputdimensions in {0, 1}
        dfmodel_out = dfmodel_out.with_columns(mgiout=pl.col("out_n_r"))

    dfmodel_out = (
        dfmodel_out
        .group_by("mgiout", cs.starts_with("out_n_"))
        .agg(
            pl
            .when(pl.col("mass_g").sum() > 0)
            .then(
                (cs.starts_with("X_") | cs.by_name("Ye", require_all=False)).dot(pl.col("mass_g"))
                / pl.col("mass_g").sum()
            )
            .otherwise(0.0),
            cs.by_name("tracercount", require_all=False).sum(),
            pl
            .when(pl.col("mass_g").sum() > 0)
            .then((cs.by_name(["q"], require_all=False)).dot(pl.col("mass_g")) / pl.col("mass_g").sum())
            .otherwise(0.0),
            pl.col("mass_g").sum().alias("out_mass_g"),
            pl.col("inputcellid").implode().alias("inputcellid_list"),
            pl.col("mass_g").implode().alias("mass_g_list"),
            (
                ~(
                    cs.by_name(["mass_g", "inputcellid", "modelgridindex", "Ye", "q", "tracercount"], require_all=False)
                    | cs.starts_with("X_")
                    | cs.starts_with("pos_")
                    | cs.starts_with("vel_")
                )
            ).implode(),
        )
        .select((pl.col("mgiout") + 1).cast(pl.Int64).alias("inputcellid"), cs.all().exclude("mgiout"))
        .join(
            pl.LazyFrame({"inputcellid": range(1, ncoordgridr * ncoordgridz + 1)}, schema={"inputcellid": pl.Int64}),
            on="inputcellid",
            how="right",
        )
        .with_columns(
            rho=pl.lit(None).cast(pl.Float32),
            out_mass_g=pl.col("out_mass_g").fill_null(0.0),
            # recompute the grid indices so that output cells with no contributing input cells are filled in
            out_n_r=((pl.col("inputcellid") - 1) % ncoordgridr).cast(pl.Int32),
            out_n_z=((pl.col("inputcellid") - 1) // ncoordgridr).cast(pl.Int32),
        )
        .with_columns(
            cs.starts_with("X_").fill_null(0.0),
            cs.by_name("Ye", "q", require_all=False).fill_null(0.0),
            # tracercount counts the trajectories of a cell, thus a float fill would make the writer give 0.0
            cs.by_name("tracercount", require_all=False).fill_null(0),
        )
        .sort("inputcellid")
    )

    if outputdimensions == 2:
        dfmodel_out = dfmodel_out.with_columns(
            pos_rcyl_mid=(pl.col("out_n_r") + 0.5) * (xmax / ncoordgridr),
            pos_z_mid=(pl.col("out_n_z") + 0.5) * (2 * xmax / ncoordgridz) - xmax,
        )
    else:
        dfmodel_out = dfmodel_out.with_columns(vel_r_max_kmps=(pl.col("out_n_r") + 1) * (vmax / ncoordgridr) / km_to_cm)

    dfmodel_out = (
        add_derived_cols_to_modeldata(dfmodel_out, modelmeta=modelmeta_out)
        .select(cs.by_name(dfmodel_out.collect_schema().names()) | cs.by_name("volume"))
        .with_columns(rho=pl.col("out_mass_g") / pl.col("volume"))
        .drop("volume", cs.starts_with("out_n_"))
        .rename({"out_mass_g": "mass_g"})
    )
    if outputdimensions < 2:
        dfmodel_out = dfmodel_out.with_columns(logrho=LOGRHO_FROM_RHO).drop("rho")

    modelmeta_out["npts_model"] = dfmodel_out.select(pl.len()).collect().item()
    assert modelmeta_out["npts_model"] == ncoordgridr * ncoordgridz

    dfoutcell_inputcells_masses = dfmodel_out.select(
        out_inputcellid=pl.col("inputcellid"),
        inputcellid=pl.col("inputcellid_list"),
        mass_g=pl.col("mass_g_list"),
        out_mass_g=pl.col("mass_g_list").list.sum(),
    ).explode("inputcellid", "mass_g", empty_as_null=False)

    dfmodel_out = dfmodel_out.drop(["inputcellid_list", "mass_g_list"], strict=False)
    if other_cols := dfmodel_out.select(cs.by_dtype(pl.List)).collect_schema().names():
        assert not other_cols, f"Not sure how to combine column values: {other_cols}"

    dfelabundances_out = (
        (
            dfelabundances
            .lazy()
            .with_columns(pl.col("inputcellid").cast(pl.Int32))
            .join(dfoutcell_inputcells_masses, on="inputcellid", how="left")
            .drop("inputcellid")
            .group_by("out_inputcellid")
            .agg(
                (cs.starts_with("X_").dot(pl.col("mass_g")) / pl.col("mass_g").sum()).fill_nan(0.0),
                cs.by_name("mass_g").sum(),
            )
            .rename({"out_inputcellid": "inputcellid"})
            .drop_nulls("inputcellid")
            .sort("inputcellid")
        )
        if dfelabundances is not None
        else pl.LazyFrame()
    )

    dfgridcontributions_out = (
        remap_gridcontributions(dfgridcontributions, dfoutcell_inputcells_masses)
        if dfgridcontributions is not None
        else pl.LazyFrame()
    )

    dfmodel_out, dfelabundances_out, dfgridcontributions_out = pl.collect_all((
        dfmodel_out,
        dfelabundances_out,
        dfgridcontributions_out,
    ))

    if dfelabundances is not None:
        assert modelmeta_out["npts_model"] == dfelabundances_out.select(pl.len()).item()

    print(f"  took {time.perf_counter() - timestart:.1f} seconds")

    return (dfmodel_out, dfelabundances_out, dfgridcontributions_out, modelmeta_out)


def scale_model_to_time(
    dfmodel: pl.DataFrame,
    targetmodeltime_days: float,
    t_model_days: float | None = None,
    modelmeta: dict[str, t.Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Homologously expand model to targetmodeltime_days by reducing densities and adjusting cell positions."""
    if t_model_days is None:
        assert modelmeta is not None
        t_model_days = modelmeta["t_model_init_days"]

    assert t_model_days is not None

    timefactor = targetmodeltime_days / t_model_days

    print(
        f"Adjusting t_model to {targetmodeltime_days} days (factor {timefactor}) "
        "using homologous expansion of positions and densities"
    )

    scale_exprs: list[pl.Expr] = [cs.starts_with("pos_") * timefactor]
    if "rho" in dfmodel.columns:
        scale_exprs.append(pl.col("rho") * timefactor**-3)
    if "logrho" in dfmodel.columns:
        scale_exprs.append(pl.col("logrho") + math.log10(timefactor**-3))
    dfmodel = dfmodel.with_columns(scale_exprs)

    if modelmeta is None:
        modelmeta = {}

    modelmeta["t_model_init_days"] = targetmodeltime_days
    # the cell widths hold positions at t_model, thus they expand with the positions
    for key in [key for key in modelmeta if key == "wid_init" or key.startswith("wid_init_")]:
        modelmeta[key] *= timefactor
    modelmeta.setdefault("headercommentlines", []).append(
        f"scaled from {t_model_days} to {targetmodeltime_days} (no abund change from decays)"
    )

    return dfmodel, modelmeta


def savetologfile(outputfolderpath: Path, logfilename: str = "modellog.txt") -> Callable[..., None]:
    """Return a print-alike that also appends to a log file, truncating any previous log."""
    outputfolderpath.mkdir(parents=True, exist_ok=True)
    logfilepath = outputfolderpath / logfilename
    logfilepath.unlink(missing_ok=True)

    def logprint(*args: t.Any, **kwargs: t.Any) -> None:
        print(*args, **kwargs)
        with logfilepath.open("a", encoding="utf-8") as logfile:
            logfile.write(" ".join([str(x) for x in args]) + "\n")

    return logprint


# The model columns are Float32. Float32 rounds a value, thus a cell on a bound can move to the other side of
# the bound. The measured errors are 6e-8 in velocity and 8e-6 degrees in angle, thus these tolerances give a margin.
VELOCITY_RTOL = 1e-6
POLAR_ANGLE_ATOL_DEG = 1e-4


def get_selection_labels(
    vmin: float | None = None, vmax: float | None = None, thetamin: float | None = None, thetamax: float | None = None
) -> list[str]:
    """Return a label for each given bound, e.g. vmin=0.02.

    The label holds the shortest text that gives the same float again. Two different bounds then give two
    different file names.
    """
    bounds = {"vmin": vmin, "vmax": vmax, "thetamin": thetamin, "thetamax": thetamax}
    return [f"{name}={value!r}" for name, value in bounds.items() if value is not None]


def get_cell_selection(
    vmin: float | None = None, vmax: float | None = None, thetamin: float | None = None, thetamax: float | None = None
) -> pl.Expr:
    """Return a selection that is true for a cell in the velocity range [c] and the polar angle range [degrees].

    The selection reads the columns vel_r_mid_on_c and vel_z_mid_on_c. Call add_derived_cols_to_modeldata first.
    The positive z axis gives a polar angle of zero. A cell at the origin has no polar angle, thus a polar angle
    range excludes it. Both ends of a range are inside the range. A cell on a bound stays inside, because the
    tolerances are larger than the Float32 error.
    """
    if vmin is not None and vmax is not None and vmin > vmax:
        msg = f"vmin must be less than or equal to vmax, but vmin={vmin:g} and vmax={vmax:g}"
        raise ValueError(msg)
    if thetamin is not None and thetamax is not None and thetamin > thetamax:
        msg = f"thetamin must be less than or equal to thetamax, but thetamin={thetamin:g} and thetamax={thetamax:g}"
        raise ValueError(msg)
    for name, theta in (("thetamin", thetamin), ("thetamax", thetamax)):
        if theta is not None and not 0.0 <= theta <= 180.0:
            msg = f"{name} must be between 0 and 180 degrees, but {name}={theta:g}"
            raise ValueError(msg)

    vel_r = pl.col("vel_r_mid_on_c")
    conditions: list[pl.Expr] = []
    if vmin is not None:
        conditions.append(vel_r >= vmin * (1.0 - VELOCITY_RTOL))
    if vmax is not None:
        conditions.append(vel_r <= vmax * (1.0 + VELOCITY_RTOL))

    if thetamin is not None or thetamax is not None:
        # a cell at the origin has a null angle, and each comparison with a null gives false
        theta_deg = pl.when(vel_r > 0.0).then(pl.col("vel_z_mid_on_c") / vel_r).arccos().degrees()
        if thetamin is not None:
            conditions.append(theta_deg >= thetamin - POLAR_ANGLE_ATOL_DEG)
        if thetamax is not None:
            conditions.append(theta_deg <= thetamax + POLAR_ANGLE_ATOL_DEG)

    if not conditions:
        return pl.repeat(value=True, n=pl.len())

    return pl.all_horizontal(conditions).fill_null(value=False)
