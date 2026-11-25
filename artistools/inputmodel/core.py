"""Read, write, and derive columns for ARTIS model.txt and abundance input files."""

import argparse
import datetime as dt
import errno
import gc
import json
import math
import os
import re
import time
import typing as t
import warnings
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from artistools.atomic import get_atomic_number
from artistools.atomic import get_elsymbol
from artistools.atomic import get_z_a_nucname
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.constants import Msun_to_g
from artistools.misc import add_help_note
from artistools.misc import ALL_MODEL_EDGE_KEYS
from artistools.misc import CellDirectionRange
from artistools.misc import exit_with_error
from artistools.misc import find_sn3d_input_file
from artistools.misc import firstexisting
from artistools.misc import firstexisting_or_none
from artistools.misc import folder_is_artis_model
from artistools.misc import get_bin_index_expr
from artistools.misc import get_cell_counts_of_edges
from artistools.misc import get_cell_direction_range
from artistools.misc import get_costheta_phi_exprs
from artistools.misc import get_costhetabin_expr
from artistools.misc import get_file_copies
from artistools.misc import get_file_identity
from artistools.misc import get_model_coordinates
from artistools.misc import get_model_folder
from artistools.misc import get_model_format_version
from artistools.misc import get_parquet_cache_name
from artistools.misc import get_phibin_expr
from artistools.misc import has_mirror_plane
from artistools.misc import is_remote_path
from artistools.misc import MODEL_EDGE_KEYS
from artistools.misc import modelpath_cache
from artistools.misc import MTIME_TOLERANCE_S
from artistools.misc import on_model_host
from artistools.misc import parse_npts_line
from artistools.misc import path_is_codecomparison
from artistools.misc import polars_source
from artistools.misc import print_detail
from artistools.misc import print_saved
from artistools.misc import print_warning
from artistools.misc import raise_if_no_final_line_end
from artistools.misc import read_keyword_model_header
from artistools.misc import read_parquet_cache_metadata
from artistools.misc import read_wsv
from artistools.misc import resolve_modelpath
from artistools.misc import resolve_outputfile
from artistools.misc import SN3D_INPUT_FOLDERS
from artistools.misc import SPHERICAL_EDGE_COORDINATES
from artistools.misc import UNIFORM_EDGE_COORDINATES
from artistools.misc import validate_model_edges
from artistools.misc import with_compressed_extension
from artistools.misc import with_compressed_sibling
from artistools.misc import without_compressed_extension
from artistools.misc import write_file_atomic
from artistools.misc import write_parquet_atomic
from artistools.misc import write_zstd_lines
from artistools.misc import zopen
from artistools.paths import get_path

if t.TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    import numpy.typing as npt

CREATED_COMMENT_PREFIX = "created:"
CREATED_TIME_FORMAT = "%Y-%m-%d %H:%M:%S UTC"
UNITS_COMMENT_PREFIX = "column units:"
UNITS_COMMENT_END = "Each X_ column is a mass fraction"


def get_created_comment() -> str:
    """Return the comment line that gives the creation time of an input file in UTC."""
    return f"# {CREATED_COMMENT_PREFIX} {dt.datetime.now(tz=dt.UTC).strftime(CREATED_TIME_FORMAT)}"


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


def add_position_columns_of_edges(
    dfmodel: pl.LazyFrame, modelmeta: dict[str, t.Any], firstinputcellid: int
) -> pl.LazyFrame:
    """Return the cells of a version 2 file with the position columns of version 1 for its coordinates.

    A version 2 file gives the cell edges in its header and no position column. The code after the reader needs the
    columns of version 1: vel_r_max_kmps and logrho (1D), pos_rcyl_mid and pos_z_mid (2D cylindrical), and pos_x_min,
    pos_y_min, and pos_z_min (3D Cartesian). A 1D model also keeps rho. A spherical2d or spherical3d model has no
    such column, because add_spherical_derived_cols reads the edges.
    """
    coordinates = modelmeta["coordinates"]
    if coordinates in SPHERICAL_EDGE_COORDINATES:
        return dfmodel

    cellindex = pl.col("inputcellid") - firstinputcellid
    kmps_to_cm_tmodel = km_to_cm * modelmeta["t_model_init_days"] * day_to_s

    def edge_values(edgekey: str, axisindex: pl.Expr, offset: int = 0) -> pl.Expr:
        return pl.lit(pl.Series(modelmeta[edgekey], dtype=pl.Float64)).gather(axisindex + offset)

    match coordinates:
        case "spherical1d":
            positioncols = {"vel_r_max_kmps": edge_values("vel_r_edges_kmps", cellindex, 1), "logrho": LOGRHO_FROM_RHO}
        case "cylindrical2d":
            n_rcyl = modelmeta["ncoordgridrcyl"]
            index_rcyl, index_z = cellindex % n_rcyl, cellindex // n_rcyl
            positioncols = {
                "pos_rcyl_mid": (
                    edge_values("vel_rcyl_edges_kmps", index_rcyl) + edge_values("vel_rcyl_edges_kmps", index_rcyl, 1)
                )
                / 2.0
                * kmps_to_cm_tmodel,
                "pos_z_mid": (edge_values("vel_z_edges_kmps", index_z) + edge_values("vel_z_edges_kmps", index_z, 1))
                / 2.0
                * kmps_to_cm_tmodel,
            }
        case _:
            n_x, n_y = modelmeta["ncoordgridx"], modelmeta["ncoordgridy"]
            positioncols = {
                "pos_x_min": edge_values("vel_x_edges_kmps", cellindex % n_x) * kmps_to_cm_tmodel,
                "pos_y_min": edge_values("vel_y_edges_kmps", (cellindex // n_x) % n_y) * kmps_to_cm_tmodel,
                "pos_z_min": edge_values("vel_z_edges_kmps", cellindex // (n_x * n_y)) * kmps_to_cm_tmodel,
            }

    othercols = [col for col in dfmodel.collect_schema().names() if col not in {"inputcellid", "rho", *positioncols}]
    rho_and_logrho = ["logrho", "rho"] if coordinates == "spherical1d" else ["rho"]
    return dfmodel.with_columns(expr.cast(pl.Float32).alias(col) for col, expr in positioncols.items()).select(
        "inputcellid", *[col for col in positioncols if col != "logrho"], *rho_and_logrho, *othercols
    )


def scan_modelfile_text(
    filename: Path | str, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides."""
    if not printwarningsonly:
        print(f"Reading {filename}")

    with zopen(filename) as fmodel:
        formatversion = get_model_format_version(fmodel.readline(), filename)
    if formatversion == 2:
        # a cut inside the last value gives a valid number, and the key cell_line_count does not find it
        raise_if_no_final_line_end(with_compressed_sibling(filename))

    with zopen(filename) as fmodel:
        onelinepercellformat: bool | None = None

        modelmeta: dict[str, t.Any] = {"headercommentlines": [], "formatversion": formatversion}
        xmax_tmodel: float | int = 0.0
        ncoordgridx: int = 0
        ncoordgridy: int = 0
        ncoordgridz: int = 0

        columns: list[str] | None = None
        celllinecount: int | None = None
        if formatversion == 2:
            fmodel.readline()
            modelmeta, columns, linecount, celllinecount = read_keyword_model_header(fmodel, filename)
            numheaderrows = 1 + linecount
            # save_modeldata writes these lines again, thus a kept copy gives each line two times
            modelmeta["headercommentlines"] = [
                commentline for commentline in modelmeta["headercommentlines"] if not is_writer_comment(commentline)
            ]
            npts_model = modelmeta["npts_model"]
            if not printwarningsonly:
                gridshape = " x ".join(
                    str(len(modelmeta[key]) - 1) for key in MODEL_EDGE_KEYS[modelmeta["coordinates"]]
                )
                print(
                    f"  Detected {modelmeta['coordinates']} model file of format version 2 with {gridshape} ="
                    f" {npts_model} cells"
                )
            if modelmeta["coordinates"] in UNIFORM_EDGE_COORDINATES:
                xmax_tmodel = modelmeta["vmax_cmps"] * (modelmeta["t_model_init_days"] * day_to_s)
            line = fmodel.readline()
        else:
            line, numheaderrows = read_noncomment_line(fmodel, modelmeta["headercommentlines"])
            cellcounts = parse_npts_line(line, filename)
            if len(cellcounts) == 2:
                modelmeta["dimensions"] = 2
                ncoordgridr, ncoordgridz = cellcounts
                modelmeta["ncoordgridrcyl"] = ncoordgridr
                modelmeta["ncoordgridz"] = ncoordgridz
                npts_model = ncoordgridr * ncoordgridz
                if not printwarningsonly:
                    print(
                        f"  Detected 2D model file with n_r * n_z = {ncoordgridr} x {ncoordgridz} = {npts_model} cells"
                    )
            else:
                npts_model = cellcounts[0]

            modelmeta["npts_model"] = npts_model
            line, linecount = read_noncomment_line(fmodel, modelmeta["headercommentlines"])
            numheaderrows += linecount
            try:
                modelmeta["t_model_init_days"] = float(line.split("#", 1)[0])
            except ValueError:
                msg = (
                    f"In {filename}, the line after the cell count must give the time of the model in days,"
                    f" not {line!r}"
                )
                raise ValueError(msg) from None

            line = fmodel.readline()
            # if the next line is a single float then the model is 2D or 3D (vmax)
            try:
                modelmeta["vmax_cmps"] = float(line.split("#", 1)[0])
            except ValueError:
                assert modelmeta.get("dimensions", -1) != 2, "2D model should have a vmax line here"
                if "dimensions" not in modelmeta:
                    if not printwarningsonly:
                        print(f"  Detected 1D model file with {npts_model} radial zones")
                    modelmeta["dimensions"] = 1
            else:
                # xmax = ymax = zmax
                xmax_tmodel = modelmeta["vmax_cmps"] * (modelmeta["t_model_init_days"] * day_to_s)
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
                            f"  Detected 3D model file with {ncoordgridx} x {ncoordgridy} x {ncoordgridz}"
                            f" = {npts_model} cells"
                        )

                line = fmodel.readline()

            if line.startswith("#"):
                numheaderrows += 1
                columns = line.lstrip("#").split()
                line = fmodel.readline()

        # a comment line or an empty line can come before each cell line of format version 2, as in sn3d
        while formatversion == 2 and line and (not line.strip() or line.lstrip().startswith("#")):
            # the readers below start at the first cell line
            numheaderrows += 1
            line = fmodel.readline()
        data_line_even = line
        ncols_line_even = len(data_line_even.split())
        data_line_odd = fmodel.readline()
        while (
            formatversion == 2
            and data_line_odd
            and (not data_line_odd.strip() or data_line_odd.lstrip().startswith("#"))
        ):
            data_line_odd = fmodel.readline()
        ncols_line_odd = len(data_line_odd.split())

    if ncols_line_even == 0:
        msg = (
            f"{filename}: found no cell line. A model.txt of format version 2 needs one cell line or more"
            if formatversion == 2
            else f"{filename}: found only 0 cells instead of {npts_model} expected."
        )
        raise ValueError(msg)

    modelmeta["coordinates"] = get_model_coordinates(modelmeta)
    t_model_init_seconds = modelmeta["t_model_init_days"] * day_to_s
    if formatversion == 2 and ncols_line_even != len(columns or []):
        msg = (
            f"{filename}: the first cell line holds {ncols_line_even} values, but the key columns gives"
            f" {len(columns or [])} names. A cell has one line in format version 2"
        )
        raise ValueError(msg)
    # the header of a version 2 file gives no cell count. Thus the reader takes one row more than the edges give,
    # and the check below finds an extra row
    maxrowcount = npts_model + 1 if formatversion == 2 else npts_model

    if columns is None:
        columns = get_standard_columns(modelmeta["coordinates"], includenico57=True, pos_unknown=True)
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
            print("  Model file is one line per cell")
        ncols_line_odd = 0
        onelinepercellformat = True
    else:
        if not printwarningsonly:
            print("  Model file format is two lines per cell")
        # columns split over two lines
        assert (ncols_line_even + ncols_line_odd) == len(columns)
        onelinepercellformat = False

    dfmodelfast: pl.DataFrame | None = None
    if onelinepercellformat and all(has_single_space_separators(line) for line in (data_line_even, data_line_odd)):
        try:
            dfmodelfast = pl.read_csv(
                polars_source(filename),
                separator=" ",
                new_columns=columns,
                n_rows=maxrowcount,
                has_header=False,
                # skip_rows does not count a comment line if comment_prefix is set, and skip_lines counts each line
                skip_rows=0 if formatversion == 2 else numheaderrows,
                skip_lines=numheaderrows if formatversion == 2 else 0,
                comment_prefix="#" if formatversion == 2 else None,
                schema={col: get_reader_column_type(col) for col in columns},
                # A trailing space gives an empty last field. Version 1 ignores each field after the last column.
                # sn3d stops for such a field in version 2, thus the reader raises and the slow reader takes the file
                truncate_ragged_lines=formatversion == 1,
                extra_columns="ignore" if formatversion == 1 else "raise",
                # a header comment can hold one quotation mark, e.g. 5" model. A reader that takes it as a quote
                # skips the rows to the next quotation mark, and the data lines then go with the header
                quote_char=None,
                glob=False,
            )
        except pl.exceptions.PolarsError:
            if formatversion == 1:
                raise
            dfmodelfast = None
        # the test of the separators reads two lines. An extra space in a later line gives a null and moves the values
        if dfmodelfast is not None and dfmodelfast.null_count().sum_horizontal().item() > 0:
            dfmodelfast = None

    if dfmodelfast is not None:
        if not printwarningsonly:
            print("  Using fast method polars.read_csv (requires one line per cell and single space delimiters)")
        dfmodel = dfmodelfast.lazy()

    else:
        # a cell of dfmodelraw can span two lines. One read of the full file and a slice after it
        # prevent a second read
        try:
            dfmodelraw = read_wsv(
                filename,
                has_header=False,
                skip_rows=numheaderrows,
                # sn3d skips a comment line of version 2, but it stops for text after the values of a cell line
                line_filter=~pl.col("line").str.strip_chars_start().str.starts_with("#")
                if formatversion == 2
                else None,
                new_columns=[str(i) for i in range(max(ncols_line_even, ncols_line_odd))],
            )
        except pl.exceptions.SchemaError:
            if formatversion == 1:
                raise
            msg = f"{filename}: a cell line has more values than the key columns gives names, or text after its values"
            raise ValueError(msg) from None

        dfmodel = (dfmodelraw if onelinepercellformat else dfmodelraw[: npts_model * 2 : 2]).select([
            pl.col(str(i)).alias(colname) for i, colname in enumerate(columns[:ncols_line_even])
        ])
        if formatversion == 2 and dfmodel.schema["inputcellid"].is_float():
            # the cast to Int32 truncates an id such as 1.5, and sn3d stops for such an id
            notwholeids = dfmodel.filter(~IS_WHOLE_INPUTCELLID)["inputcellid"]
            if notwholeids.len():
                msg = f"{filename}: the inputcellid {notwholeids.item(0)!r} is not a whole number"
                raise ValueError(msg)
        dfmodel = dfmodel.with_columns(pl.col("inputcellid").cast(pl.Int32))

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

        dfmodel = (
            dfmodel
            .head(maxrowcount)
            .with_columns(
                pl.col(col).cast(get_reader_column_type(col)) for col in dfmodel.columns if col != "inputcellid"
            )
            .lazy()
        )

    if (
        formatversion == 2
        and not dfmodel.select((pl.col("inputcellid").diff().drop_nulls() > 0).all()).collect().item()
    ):
        # sn3d reads the cells in the sequence of the file and stops for an id that does not increase
        msg = f"{filename}: the inputcellid values of format version 2 must increase from each cell line to the next"
        raise ValueError(msg)

    # an old model.txt names the electron fraction cellYe, and Ye is the name everywhere after this point
    dfmodel = dfmodel.sort("inputcellid").rename({"velocity_outer": "vel_r_max_kmps", "cellYe": "Ye"}, strict=False)
    if (
        "tracercount" in dfmodel.collect_schema().names()
        and dfmodel
        .select((pl.col("tracercount").is_finite() & (pl.col("tracercount") == pl.col("tracercount").round())).all())
        .collect()
        .item()
    ):
        # a count of trajectories is a whole number, and the writer gives an integer column with all its digits
        dfmodel = dfmodel.with_columns(pl.col("tracercount").cast(pl.Int64))

    # In version 1, sn3d stops for a file with too few cells. It takes the id of the first cell (0 or 1) as the start
    # of the cell index, and each id after it must be one more than the id before it
    cellcount, firstinputcellid, lastinputcellid, idsareconsecutive, hasnullvalue = (
        dfmodel
        .select(
            cellcount=pl.len(),
            firstinputcellid=pl.col("inputcellid").first(),
            lastinputcellid=pl.col("inputcellid").last(),
            idsareconsecutive=(pl.col("inputcellid") == pl.col("inputcellid").first() + pl.int_range(pl.len())).all(),
            hasnullvalue=pl.any_horizontal(pl.all().is_null()).any(),
        )
        .collect()
        .row(0)
    )
    if formatversion == 2 and hasnullvalue:
        # sn3d stops for such a line
        msg = f"{filename}: a cell line has fewer values than the key columns gives names"
        raise ValueError(msg)
    if formatversion == 2:
        raise_for_values_that_sn3d_refuses(dfmodel, str(filename))
    if cellcount > npts_model:
        msg = (
            f"{filename}: found more than {npts_model} cells, but the cell edges of the header give {npts_model} cells"
        )
        raise ValueError(msg)
    if formatversion == 2:
        # the id of a cell is its index in the grid of the edges, from 0. A cell with no line is empty
        if firstinputcellid < 0 or lastinputcellid >= npts_model:
            msg = (
                f"{filename}: the cell edges of the header give {npts_model} cells, thus each inputcellid must be"
                f" from 0 to {npts_model - 1}"
            )
            raise ValueError(msg)
        if celllinecount is not None and cellcount != celllinecount:
            # a cell with no line is empty, thus only this count finds a file that lost lines at its end
            msg = (
                f"{filename}: the key cell_line_count gives {celllinecount} cell lines, but the file has {cellcount}."
                " The file is possibly not complete"
            )
            raise ValueError(msg)
        filecolumns = dfmodel.collect_schema().names()
        absentcolumns = [col for col in STANDARD_COMPOSITION_COLUMNS if col not in filecolumns]
        if absentcolumns:
            # the code after the reader takes the columns of version 1, and sn3d uses zero for an absent column
            if not printwarningsonly:
                print(f"  The file has no column {', '.join(absentcolumns)}. An absent column holds 0 in each cell")
            dfmodel = dfmodel.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(col) for col in absentcolumns)
        dfmodel = dfmodel.select(
            "inputcellid",
            "rho",
            *STANDARD_COMPOSITION_COLUMNS,
            *[col for col in filecolumns if col not in {"inputcellid", "rho", *STANDARD_COMPOSITION_COLUMNS}],
        )
        if cellcount < npts_model:
            if not printwarningsonly:
                print(f"  {npts_model - cellcount} cells have no line, and they are empty (rho = 0)")
            dfmodel = (
                pl
                .LazyFrame({"inputcellid": pl.int_range(npts_model, dtype=pl.Int32, eager=True)})
                .join(dfmodel, on="inputcellid", how="left", maintain_order="left")
                .with_columns(pl.exclude("inputcellid").fill_null(0.0))
            )
        firstinputcellid = 0
        dfmodel = add_position_columns_of_edges(dfmodel, modelmeta, firstinputcellid)
    elif cellcount != npts_model:
        msg = f"{filename}: found only {cellcount} cells instead of {npts_model} expected."
        raise ValueError(msg)
    elif firstinputcellid not in {0, 1} or not idsareconsecutive:
        msg = f"{filename}: the inputcellid values must start at 0 or 1 and increase by one from each cell to the next"
        raise ValueError(msg)

    if modelmeta["coordinates"] in SPHERICAL_EDGE_COORDINATES:
        # the file gives no position of a cell, thus the index of a cell in the grid of the edges comes from its id
        modelmeta["firstinputcellid"] = firstinputcellid

    elif modelmeta["dimensions"] == 1:
        if formatversion == 1:
            vmax_kmps = dfmodel.select(pl.col("vel_r_max_kmps").max()).collect().item()
            assert isinstance(vmax_kmps, float)
            modelmeta["vmax_cmps"] = vmax_kmps * km_to_cm

    elif modelmeta["dimensions"] == 2:
        wid_init_rcyl = modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridrcyl"]
        # the z axis of a mirrored model is the half from 0 to vmax
        zaxiswidth = 1 if has_mirror_plane(modelmeta) else 2
        wid_init_z = zaxiswidth * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridz"]
        modelmeta["wid_init_rcyl"] = wid_init_rcyl
        modelmeta["wid_init_z"] = wid_init_z

        # check pos_rcyl_mid and pos_z_mid are correct. One expression over the whole column instead of a Python
        # loop, which cost a round trip through the interpreter for every cell of the grid
        n_r = (pl.col("inputcellid") - firstinputcellid) % modelmeta["ncoordgridrcyl"]
        n_z = (pl.col("inputcellid") - firstinputcellid) // modelmeta["ncoordgridrcyl"]
        pos_z_min_grid = 0.0 if has_mirror_plane(modelmeta) else -modelmeta["vmax_cmps"] * t_model_init_seconds

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
                print("  Model cell positions are defined in the header")
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

            # the cells at these indices give each order of the axes and each place in the cell a different position
            gridindices = np.array([
                0,
                ncoordgridx - 1,
                ncoordgridx,
                (ncoordgridx - 1) * (ncoordgridy - 1),
                (ncoordgridx - 1) * ncoordgridy,
                (ncoordgridx - 1) * (ncoordgridy - 1) * (ncoordgridz - 1),
            ])
            axisindices = {
                "x": gridindices % ncoordgridx,
                "y": (gridindices // ncoordgridx) % ncoordgridy,
                "z": (gridindices // (ncoordgridx * ncoordgridy)) % ncoordgridz,
            }
            cellwidths = {"x": wid_init_x, "y": wid_init_y, "z": wid_init_z}
            filepositions = (
                dfmodel
                .select(
                    cs
                    .by_name("inputpos_a", "inputpos_b", "inputpos_c")
                    .gather(gridindices.tolist())
                    .explode(empty_as_null=False)
                )
                .collect()
                .to_numpy()
            )
            # each match is an order of the axes, and min for the lower corner or mid for the centre of a cell
            matchedorders = [
                (axesorder, place)
                for axesorder in (("x", "y", "z"), ("z", "y", "x"))
                for place, celloffset in (("min", 0.0), ("mid", 0.5))
                if np.isclose(
                    filepositions,
                    np.column_stack([
                        -xmax_tmodel + (axisindices[axis] + celloffset) * cellwidths[axis] for axis in axesorder
                    ]),
                    atol=np.array([cellwidths[axis] * 0.05 for axis in axesorder]),
                ).all()
            ]
            if len(matchedorders) != 1:
                msg = (
                    f"{filename}: the cell positions agree with no order of the position columns. "
                    "Make vmax consistent with the cell positions."
                )
                raise ValueError(msg)

            ((axesorder, place),) = matchedorders
            if not printwarningsonly:
                placetext = "min corner" if place == "min" else "midpoint"
                print(f"  model cell positions are consistent with {'-'.join(axesorder)} {placetext} columns")

            dfmodel = dfmodel.rename(
                {
                    filecol: f"pos_{axis}_{place}"
                    for filecol, axis in zip(("inputpos_a", "inputpos_b", "inputpos_c"), axesorder, strict=True)
                },
                strict=False,
            )

            if place == "mid":
                dfmodel = dfmodel.with_columns(
                    pos_x_min=(pl.col("pos_x_mid") - modelmeta["wid_init_x"] / 2.0),
                    pos_y_min=(pl.col("pos_y_mid") - modelmeta["wid_init_y"] / 2.0),
                    pos_z_min=(pl.col("pos_z_mid") - modelmeta["wid_init_z"] / 2.0),
                )
    return dfmodel, modelmeta


# the electron fraction is the number of electrons for each nucleon, thus it is from 0 to 1
# a null Ye is a value that is not known, and the writer gives a null as 0
IS_VALID_YE = (pl.col("Ye").is_finite() & pl.col("Ye").is_between(0.0, 1.0)).fill_null(value=False)
# sn3d reads the id of a cell as an integer, and a cast to an integer truncates a value such as 1.5
IS_WHOLE_INPUTCELLID = (
    pl.col("inputcellid").is_finite() & (pl.col("inputcellid") == pl.col("inputcellid").round())
).fill_null(value=False)

# The version of the parquet cache format of every text source that get_text_source_cached() reads,
# which is model.txt and abundances.txt. Increase it for a change that makes an older cache file
# incorrect, e.g. a new column or a different data type in either one. Version 2 checks the cell count and the
# cell ids of model.txt, and it puts the second line of a cell beside the first line in the order of the file.
# Version 3 gives the key coordinates in the metadata of each model. Version 4 gives the key formatversion.
# Version 5 checks more values of format version 2, as sn3d does. Version 6 gives tracercount as Int64.
CACHEVERSION = 8


def scan_parquet_cache(
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
        df = pl.scan_parquet(parquetfilepath, glob=False)
    except (pl.exceptions.PolarsError, OSError) as exc:
        print(f"Could not read {parquetfilepath} ({type(exc).__name__}: {exc}). Will regenerate.")
        return None

    if not printwarningsonly:
        print(f"Reading table from {parquetfilepath}")

    return df, pqmetadata


def get_parquet_cache_path(textfilepath: Path) -> Path:
    """Return the path of the parquet cache of a text file that get_text_source_cached reads."""
    # model_a.1.txt and model_a.2.txt must not share a cache, thus remove only a compression suffix
    return get_parquet_cache_name(textfilepath.with_name(f"{without_compressed_extension(textfilepath).name}.parquet"))


def remove_parquet_cache(textfilepath: Path) -> None:
    """Delete the parquet cache of a text file that the caller wrote again.

    The cache check accepts a modification time within MTIME_TOLERANCE_S of its stamp. Thus the time of
    a text file that the caller wrote again soon after a read cannot show that the cache is stale.
    """
    # a cache of the earlier name and a cache of the current name can both exist
    while (parquetfilepath := get_parquet_cache_path(textfilepath)).is_file():
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

    cached = scan_parquet_cache(
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
    write_parquet_atomic(
        df,
        parquetfilepath,
        replaces=outdatedparquet,
        metadata={
            "creationtimeutc": str(dt.datetime.now(dt.UTC)),
            "cacheversion": str(CACHEVERSION),
            "textsource_mtime": str(textsource_mtime),
        }
        | extrametadata,
    )

    print("  Done.")
    del df
    gc.collect()
    return pl.scan_parquet(parquetfilepath, glob=False), extrametadata


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


def get_model_input_folder(modelpath: Path | str) -> Path:
    """Return the folder of the input files of a local model path, e.g. abundances.txt and Ye.txt.

    This is the folder of the model.txt that the reader of model.txt opens for the path. A model path can name a
    folder, and the reader then also searches its subfolders. A model path can also name a file, e.g. model.txt for
    a folder that holds only model.txt.zst. A code comparison path is virtual, and get_model_text_folder gives the
    folder of its ARTIS input files.
    """
    modelfile = get_model_file_path(modelpath)
    return get_model_text_folder(modelpath) if modelfile is None else modelfile.parent


def get_gridcontributions_folder(modelpath: Path | str) -> Path:
    """Return the folder of the gridcontributions.txt of a local model path.

    The file is beside the model.txt that the reader opens, which can be in a subfolder such as data/. An earlier
    version of artistools wrote the file in the model folder, thus the model folder applies if the folder of model.txt
    has no such file.
    """
    inputfolder = get_model_input_folder(modelpath)
    if firstexisting_or_none("gridcontributions.txt", folder=inputfolder, tryzipped=True, search_subfolders=False):
        return inputfolder
    return get_model_folder(modelpath)


def get_model_file_path(modelpath: Path | str) -> Path | None:
    """Return the model.txt that the reader of model.txt opens for a local model path, or None.

    A model path can name a folder, and the reader then also searches its subfolders. A model path can also name a
    file, e.g. model_alt.txt, or model.txt for a folder that holds only model.txt.zst.
    """
    inputpath = Path(modelpath)
    if inputpath.is_dir():
        return firstexisting_or_none("model.txt", folder=inputpath, tryzipped=True)
    return firstexisting_or_none(inputpath.name, folder=inputpath.parent, search_subfolders=False)


def with_ye_of_yefile(dfmodel: pl.DataFrame, yefilepath: Path) -> pl.DataFrame:
    """Return the cells of a version 1 model with the electron fractions of its Ye.txt in the column Ye.

    sn3d reads Ye.txt after model.txt, thus a value of Ye.txt replaces the value of the column Ye. Ye.txt can hold
    only some of the cells, and each other cell keeps the value of the column. A cell with no value from the two
    sources gets 0. sn3d with TANAKA2020_ELECTRONFRAC stops for such a cell, as it does in format version 1.

    The first value of Ye.txt is the count of the cell lines. sn3d reads that count of pairs of a cell id and an
    electron fraction, and it stops for a file with fewer pairs. Thus the function also stops for such a file, and it
    gives a warning for a value after the last pair, which sn3d does not read.
    """
    with zopen(yefilepath) as yefile:
        tokens = yefile.read().split()
    try:
        linecount = int(tokens[0]) if tokens else -1
    except ValueError:
        linecount = -1
    if linecount < 0:
        msg = f"{yefilepath} must start with the count of its cell lines, but the count is unreadable or negative"
        raise ValueError(msg)
    pairtokens = tokens[1 : 1 + 2 * linecount]
    if len(pairtokens) < 2 * linecount:
        msg = (
            f"{yefilepath} ends after {len(pairtokens) // 2} of its {linecount} cell lines. sn3d stops for such a file"
        )
        raise ValueError(msg)
    if len(tokens) > 1 + 2 * linecount:
        print_warning(
            f"{yefilepath} has {len(tokens) - 1 - 2 * linecount} values after its {linecount} cell lines, which sn3d"
            " does not read. artistools also does not read them"
        )
    try:
        dfye = pl.DataFrame(
            {
                "inputcellid": [int(value) for value in pairtokens[0::2]],
                "Ye_file": [float(v) for v in pairtokens[1::2]],
            },
            schema={"inputcellid": pl.Int32, "Ye_file": pl.Float32},
        ).unique(subset="inputcellid", keep="last")
    except ValueError:
        msg = f"{yefilepath}: each cell line must hold a cell id and an electron fraction"
        raise ValueError(msg) from None
    dfinvalidye = dfye.filter(~(pl.col("Ye_file").is_finite() & pl.col("Ye_file").is_between(0.0, 1.0)))
    if not dfinvalidye.is_empty():
        msg = (
            f"{yefilepath}: the cell {dfinvalidye['inputcellid'].item(0)} has the electron fraction"
            f" {dfinvalidye['Ye_file'].item(0)!r}. Each electron fraction must be a finite number from 0 to 1"
        )
        raise ValueError(msg)
    unknownids = dfye.join(dfmodel.select("inputcellid"), on="inputcellid", how="anti")
    if not unknownids.is_empty():
        msg = (
            f"{yefilepath}: {unknownids.height} cell ids are not in model.txt, e.g. {unknownids['inputcellid'].min()}."
            " The cell ids of Ye.txt must be the ids of model.txt"
        )
        raise ValueError(msg)

    columnvalue = pl.col("Ye") if "Ye" in dfmodel.columns else pl.lit(None, dtype=pl.Float32)
    dfmodel = (
        dfmodel
        .join(dfye, on="inputcellid", how="left", maintain_order="left")
        .with_columns(Ye=pl.coalesce(pl.col("Ye_file"), columnvalue))
        .drop("Ye_file")
    )
    print(
        f"  The column Ye of model.txt gets the {dfye.height} electron fractions of Ye.txt, and each other cell keeps"
        " the value of that column."
    )
    novaluecount = dfmodel.filter(pl.col("Ye").is_null() & get_has_density_expr(dfmodel.columns)).height
    if novaluecount > 0:
        print_warning(
            f"{novaluecount} cells with density have no electron fraction in Ye.txt or in model.txt. The column Ye"
            " gets 0 for these cells, and sn3d with TANAKA2020_ELECTRONFRAC stops for an electron fraction of 0"
        )
    return dfmodel.with_columns(pl.col("Ye").fill_null(0.0))


def require_no_ye_file_of_version_2_model(
    yefilepath: Path | None, modelmeta: dict[str, t.Any], sourcefolder: Path
) -> None:
    """Stop for a Ye.txt that sn3d reads for a source model.txt of format version 2.

    sn3d stops for such a model, thus the file is not of the model. Its values can be of an earlier model, and a
    command must not give them to a new model.
    """
    if yefilepath is not None and modelmeta.get("formatversion", 1) == 2:
        exit_with_error(
            f"sn3d reads {yefilepath}, but the model.txt of {sourcefolder} has format version 2",
            "sn3d stops for such a model. The column Ye of model.txt gives the electron fraction. Delete Ye.txt",
        )


def require_no_ye_file_of_earlier_model(outputfolder: Path, modelpath: Path | str) -> None:
    """Stop before a write of a version 1 model.txt into a folder that holds the Ye.txt of an earlier model.

    sn3d reads Ye.txt after model.txt, thus the values of that file would replace the electron fractions of the new
    model. The command does not delete the file, because a version 1 model can have its own Ye.txt.
    """
    # sn3d searches the model folder, data/, and artis/data/, thus a Ye.txt in each of them applies
    oldyefiles = get_sn3d_file_copies(get_sn3d_model_folder(outputfolder), "Ye.txt")
    if oldyefiles:
        exit_with_error(
            f"{oldyefiles[0]} is a file of an earlier model, and the new model.txt of {modelpath} has format version 1",
            "sn3d reads Ye.txt for a model.txt of format version 1. Delete the file, or give a different output folder",
        )


def get_sn3d_model_folder(inputfolder: Path) -> Path:
    """Return the model folder of sn3d for the folder of a model.txt, e.g. the parent of a data folder.

    sn3d reads its input files from the model folder, data/, and artis/data/. A data folder is part of the model
    folder above it only if that folder holds input.txt. The function returns an absolute path. It keeps a symbolic
    link, e.g. a data/ that links to a shared folder, because sn3d finds the files through that link.
    """
    inputfolder = Path(os.path.normpath(inputfolder.absolute()))
    for datafolder in SN3D_INPUT_FOLDERS[1:]:
        dataparts = Path(datafolder).parts
        if inputfolder.parts[-len(dataparts) :] == dataparts:
            modelfolder = inputfolder.parents[len(dataparts) - 1]
            if folder_is_artis_model(modelfolder):
                return modelfolder
    return inputfolder


def get_sn3d_file_copies(modelfolder: Path, filename: str) -> list[Path]:
    """Return each copy of a file in the folders that sn3d searches, e.g. data/Ye.txt.zst and Ye.txt.gz."""
    return [
        path
        for datafolder in SN3D_INPUT_FOLDERS
        for path in get_file_copies(modelfolder / datafolder / filename)
        if path.is_file()
    ]


def find_ye_file(inputfolder: Path) -> Path | None:
    """Return the Ye.txt that sn3d reads for the model.txt in a folder, or None."""
    return find_sn3d_input_file("Ye.txt", get_sn3d_model_folder(inputfolder))


@on_model_host
def get_modelmeta(modelpath: Path) -> dict[str, t.Any]:
    """Return the metadata of the model, e.g. the dimensions and the time of the model.

    The host of a remote model reads the model and sends back the metadata alone.
    """
    return scan_modeldata(modelpath, printwarningsonly=True)[1]


# the viewer resolves -deltalogx smallestscale at each change, and the grid of a model does not change
@modelpath_cache(maxsize=16)
@on_model_host
def get_spatial_scales(modelpath: Path) -> tuple[float, float, str]:
    """Return the smallest and the largest spatial scale of the model grid in velocity [cm/s], and a description.

    For a 1D model, the smallest and the largest spatial scale are the widths of the narrowest and the widest shell. For
    a 2D or 3D model, the smallest spatial scale is the smallest cell width along an axis. The largest spatial scale is
    the diagonal of a cell.
    """
    dfmodel, modelmeta = scan_modeldata(modelpath, printwarningsonly=True)
    vmax_cmps = float(modelmeta["vmax_cmps"])
    # wid_init_* is the cell width at t_model. A width divided by t_model gives the width in velocity
    t_model_init_s = float(modelmeta["t_model_init_days"]) * day_to_s
    vmaxtext = f"the maximum velocity vmax = {vmax_cmps / km_to_cm:.0f} km/s ({vmax_cmps / C_cm_per_s:.3g}c)"
    coordinates = get_model_coordinates(modelmeta)
    match coordinates:
        case "spherical1d":
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
        case "cylindrical2d":
            ncoordgridrcyl, ncoordgridz = int(modelmeta["ncoordgridrcyl"]), int(modelmeta["ncoordgridz"])
            rcylwidth = float(modelmeta["wid_init_rcyl"]) / t_model_init_s
            zwidth = float(modelmeta["wid_init_z"]) / t_model_init_s
            smallest, largest = min(rcylwidth, zwidth), math.hypot(rcylwidth, zwidth)
            # a mirrored model holds the cells with z >= 0, and each cell also fills its mirror cell
            zrangetext, zwidthtext = (
                ("from 0 to vmax, with a mirror cell at -z", "vmax / n_z")
                if has_mirror_plane(modelmeta)
                else ("from -vmax to vmax", "2 vmax / n_z")
            )
            gridtext = (
                f"The 2D model has a grid of n_rcyl x n_z = {ncoordgridrcyl} x {ncoordgridz} cells, with {vmaxtext}."
                f" The cylindrical radius rcyl is from 0 to vmax, and z is {zrangetext}. The cell widths are"
                f" width_rcyl = vmax / n_rcyl = {rcylwidth / km_to_cm:.0f} km/s and"
                f" width_z = {zwidthtext} = {zwidth / km_to_cm:.0f} km/s"
            )
            scaletext = (
                f"The smallest spatial scale is the smaller cell width, {smallest / km_to_cm:.0f} km/s. The largest"
                " spatial scale is the diagonal of a cell in the plane of rcyl and z,"
                f" sqrt(width_rcyl^2 + width_z^2) = {largest / km_to_cm:.0f} km/s"
            )
        case "cartesian3d":
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
        case _:
            # the width of a cell along theta and phi increases with the radius, from zero at the centre
            msg = (
                f"The spatial scales of the grid have no definition for a model with {coordinates} coordinates."
                " Give a number to --deltalogx"
            )
            raise ValueError(msg)
    return smallest, largest, f"{gridtext}. {scaletext}"


def scan_modeldata(
    modelpath: Path | str = ".", get_elemabundances: bool = False, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Return the cells of model.txt as a LazyFrame, and the parameters of the model as a dict.

    The LazyFrame has one row for each cell. Each value is at the start time t_model_init_days. The columns are:

    - inputcellid, and modelgridindex, which is the index of the cell in the estimators;
    - for 1D, vel_r_max_kmps (the outer velocity, km/s) and logrho (log10 of the density in g/cm^3);
    - for 2D, pos_rcyl_mid and pos_z_mid (the centre of the cell, cm) and rho (g/cm^3);
    - for 3D, pos_x_min, pos_y_min, and pos_z_min (the corner of the cell, cm) and rho (g/cm^3);
    - X_Fegroup and X_<nuclide>, e.g. X_Ni56 (mass fractions);
    - with get_elemabundances, X_<El> (the mass fraction of each element from abundances.txt).

    add_derived_cols_to_modeldata adds more columns, e.g. volume and mass_g. The dict has these keys:

    - dimensions;
    - npts_model, the number of cells;
    - t_model_init_days;
    - vmax_cmps (cm/s).

    A 3D model also gives ncoordgridx and wid_init (cm).

    modelpath is a model folder, a model file such as model_1d.txt, or a remote model as host:path. The reader writes a
    parquet cache beside a model file of more than 2 MiB. A later call reads that cache while the text file stays the
    same. printwarningsonly removes the progress messages. docs/data.md gives each column.
    """
    path = resolve_modelpath(modelpath)
    if is_remote_path(path):
        return scan_remote_modeldata(path, get_elemabundances=get_elemabundances, printwarningsonly=printwarningsonly)
    return scan_local_modeldata(path, get_elemabundances=get_elemabundances, printwarningsonly=printwarningsonly)


@on_model_host
def get_remote_modeldata_schema(
    modelpath: Path, get_elemabundances: bool, printwarningsonly: bool
) -> tuple[pl.DataFrame, dict[t.Any, t.Any]]:
    """Return an empty frame with the columns of the model data of a remote model, and the metadata of the model."""
    dfmodel, modelmeta = scan_local_modeldata(
        modelpath, get_elemabundances=get_elemabundances, printwarningsonly=printwarningsonly
    )
    return dfmodel.clear().collect(), modelmeta


@on_model_host
def read_modeldata_rows_on_host(
    modelpath: Path,
    get_elemabundances: bool,
    with_columns: list[str] | None,
    predicate: bytes | None,
    n_rows: int | None,
) -> pl.DataFrame:
    """Return the rows and the columns that a query of the client reads from the model data of a remote model.

    The predicate is a serialised polars expression. The client and the server have the same polars, see start_server.
    """
    import io

    dfmodel, _ = scan_local_modeldata(modelpath, get_elemabundances=get_elemabundances, printwarningsonly=True)
    if predicate is not None:
        dfmodel = dfmodel.filter(pl.Expr.deserialize(io.BytesIO(predicate)))
    if with_columns is not None:
        dfmodel = dfmodel.select(with_columns)
    return (dfmodel if n_rows is None else dfmodel.head(n_rows)).collect()


def read_remote_modeldata_rows(
    modelpath: Path,
    get_elemabundances: bool,
    with_columns: list[str] | None,
    predicate: pl.Expr | None,
    n_rows: int | None,
    batch_size: int | None,
) -> "Iterator[pl.DataFrame]":
    """Yield the rows of a query of the model data of a remote model, for the polars IO source of the model.

    polars gives the projection, the filter, and the row limit of the query. The host applies them, and only the rows
    and the columns of the query come back through ssh. A kilonova model can have thousands of nuclide columns.
    """
    del batch_size
    predicatedata = None if predicate is None else predicate.meta.serialize()
    yield read_modeldata_rows_on_host(modelpath, get_elemabundances, with_columns, predicatedata, n_rows)


def scan_remote_modeldata(
    modelpath: Path, get_elemabundances: bool, printwarningsonly: bool
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Return a LazyFrame of the model data of a remote model, and the metadata of the model."""
    import functools

    from polars.io.plugins import register_io_source

    template, modelmeta = get_remote_modeldata_schema(modelpath, get_elemabundances, printwarningsonly)
    reader = functools.partial(read_remote_modeldata_rows, modelpath, get_elemabundances)
    return register_io_source(reader, schema=template.schema, is_pure=True), modelmeta


def scan_local_modeldata(
    modelpath: Path, get_elemabundances: bool = False, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Read model.txt, and abundances.txt with get_elemabundances, for a local path that scan_modeldata resolved."""
    if modelpath.is_dir():
        textfilepath = firstexisting("model.txt", folder=modelpath, tryzipped=True)
    elif (
        # a file name, e.g. model_1d.txt, also names its compressed copy, e.g. model_1d.txt.zst
        namedfile := firstexisting_or_none(modelpath.name, folder=modelpath.parent, search_subfolders=False)
    ) is not None:
        textfilepath = namedfile
    elif path_is_codecomparison(modelpath):
        textfilepath = Path(get_model_text_folder(modelpath), "model.txt")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), modelpath)

    def read_text() -> tuple[pl.LazyFrame, dict[str, str]]:
        dfmodel, modelmeta = scan_modelfile_text(filename=textfilepath, printwarningsonly=printwarningsonly)
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
        print(f"  Model is {modelmeta['dimensions']}D with {modelmeta['npts_model']} cells")

    # the reader checks that the ids start at 0 or 1 and increase by one, and it sorts the cells by id. sn3d
    # takes the id of the first cell as the start of the cell index
    firstinputcellid = dfmodel.select(pl.col("inputcellid").first()).collect().item()

    if get_elemabundances:
        # the element mass fractions are of the model file that the reader opened, also beside a different model.txt
        abundancedata = (
            get_elemabundances_of_model_columns(dfmodel)
            if modelmeta.get("formatversion", 1) == 2
            else scan_initelemabundances(textfilepath, printwarningsonly=printwarningsonly)
        )
        # an inner join drops a cell that abundances.txt does not give, and each mass then leaves that cell out
        missingcellcount = dfmodel.join(abundancedata, how="anti", on="inputcellid").select(pl.len()).collect().item()
        if missingcellcount:
            msg = (
                f"abundances.txt gives no abundances for {missingcellcount} of the {modelmeta['npts_model']} cells of"
                f" {textfilepath}. The two files must give the same cells"
            )
            raise ValueError(msg)
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


def get_has_density_expr(colnames: Sequence[str]) -> pl.Expr:
    """Return an expression that is true for each cell with density.

    A 1D model of format version 1 gives log10(rho) in the column logrho, and each other model gives rho. A value of
    -98 or less of logrho gives an empty cell, as RHO_FROM_LOGRHO does.
    """
    if "rho" in colnames:
        return pl.col("rho") > 0
    if "logrho" in colnames:
        return pl.col("logrho") > -98
    msg = f"The cells have no column rho and no column logrho, but the columns {', '.join(colnames)}"
    raise ValueError(msg)


def get_spherical_cell_indices(
    colnames: Sequence[str], modelmeta: dict[str, t.Any]
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """Return the radial, polar, and azimuthal index of each cell of a spherical2d or spherical3d model.

    The radial index changes fastest: the index of a cell is i_r + n_r * (i_costheta + n_costheta * i_phi). The cell
    index is modelgridindex if colnames holds it. If not, it is inputcellid minus the key firstinputcellid of the
    metadata, which is 0 without the key, because the cell ids of format version 2 start at 0. Thus the rows can be
    a selection of the cells in each order. The azimuthal index of a spherical2d model is 0.
    """
    ncoordgridr = len(modelmeta["vel_r_edges_kmps"]) - 1
    ncoordgridcostheta = len(modelmeta["costheta_edges"]) - 1
    cellindex = (
        pl.col("modelgridindex")
        if "modelgridindex" in colnames
        else pl.col("inputcellid") - modelmeta.get("firstinputcellid", 0)
    ).cast(pl.Int64)
    return (
        cellindex % ncoordgridr,
        (cellindex // ncoordgridr) % ncoordgridcostheta,
        cellindex // (ncoordgridr * ncoordgridcostheta),
    )


def add_spherical_derived_cols(dfmodel: pl.LazyFrame, modelmeta: dict[str, t.Any]) -> pl.LazyFrame:
    """Add the bounds, the mid-point, the volume, and the kinetic energy of each cell of a grid with spherical edges.

    The cell edges of the metadata give each value, because model.txt has no position column for the coordinates
    spherical2d and spherical3d. The mid-point of a cell is the mean of its edges in r, cos(theta), and phi. A
    spherical2d cell is a ring around the z axis, and its mid-point is in the plane of rcyl and z.
    """
    t_model_init_seconds = modelmeta["t_model_init_days"] * day_to_s
    i_r, i_costheta, i_phi = get_spherical_cell_indices(dfmodel.collect_schema().names(), modelmeta)

    def edge_at(key: str, index: pl.Expr, scale: float = 1.0) -> pl.Expr:
        return pl.lit(pl.Series(modelmeta[key], dtype=pl.Float64) * scale).gather(index)

    pos_scale = km_to_cm * t_model_init_seconds
    dfmodel = dfmodel.with_columns(
        pos_r_min=edge_at("vel_r_edges_kmps", i_r, pos_scale),
        pos_r_max=edge_at("vel_r_edges_kmps", i_r + 1, pos_scale),
        costheta_min=edge_at("costheta_edges", i_costheta),
        costheta_max=edge_at("costheta_edges", i_costheta + 1),
    ).with_columns(pos_r_mid=(pl.col("pos_r_min") + pl.col("pos_r_max")) / 2.0)

    costheta_mid = (pl.col("costheta_min") + pl.col("costheta_max")) / 2.0
    midpointcols = {
        "pos_rcyl_mid": pl.col("pos_r_mid") * (1.0 - costheta_mid.pow(2)).sqrt(),
        "pos_z_mid": pl.col("pos_r_mid") * costheta_mid,
    }
    azimuthal_width: pl.Expr | float = 2.0 * math.pi
    if "phi_edges_radians" in modelmeta:
        dfmodel = dfmodel.with_columns(
            phi_min_radians=edge_at("phi_edges_radians", i_phi), phi_max_radians=edge_at("phi_edges_radians", i_phi + 1)
        )
        phi_mid = (pl.col("phi_min_radians") + pl.col("phi_max_radians")) / 2.0
        midpointcols |= {
            "pos_x_mid": midpointcols["pos_rcyl_mid"] * phi_mid.cos(),
            "pos_y_mid": midpointcols["pos_rcyl_mid"] * phi_mid.sin(),
        }
        azimuthal_width = pl.col("phi_max_radians") - pl.col("phi_min_radians")

    # a cell of a mirrored model also fills its mirror cell
    solid_angle = (
        (pl.col("costheta_max") - pl.col("costheta_min"))
        * azimuthal_width
        * (2.0 if has_mirror_plane(modelmeta) else 1.0)
    )
    return dfmodel.with_columns(
        **midpointcols,
        volume=(pl.col("pos_r_max").pow(3) - pl.col("pos_r_min").pow(3)) / 3.0 * solid_angle,
        # 1/2 rho v^2 with v = r / t, integrated over the cell
        kinetic_en_erg_r=(
            pl.col("rho")
            * t_model_init_seconds**-2
            * (pl.col("pos_r_max").pow(5) - pl.col("pos_r_min").pow(5))
            / 10.0
            * solid_angle
        ),
    )


def get_spherical_cell_index_of_velocities(
    vel_x: "npt.NDArray[np.float64]",
    vel_y: "npt.NDArray[np.float64]",
    vel_z: "npt.NDArray[np.float64]",
    modelmeta: "Mapping[str, t.Any]",
) -> "npt.NDArray[np.int64]":
    """Return the index of the cell of a spherical2d or spherical3d model that holds each velocity vector [cm/s].

    A velocity at or above the outer edge of the grid gives -1. The rules are those of sn3d and of
    get_spherical_modelgridindex_expr of the packets. A cell holds its lower edge on each axis, and the last polar
    cell also holds cos(theta) = 1, which is the +z axis. An azimuthal angle that rounds to 2 pi is the angle 0. A
    velocity of zero has no direction, and it gets the first polar cell and the first azimuthal cell. The index of
    a cell is i_r + n_r * (i_costheta + n_costheta * i_phi), and a spherical2d model has one azimuthal cell.
    """
    vel_r_edges = np.array([edge * km_to_cm for edge in modelmeta["vel_r_edges_kmps"]], dtype=np.float64)
    costheta_edges = np.asarray(modelmeta["costheta_edges"], dtype=np.float64)
    ncoordgridr = len(vel_r_edges) - 1
    ncoordgridcostheta = len(costheta_edges) - 1

    # a velocity with z < 0 of a mirrored model is in the mirror cell of its mirror velocity
    if has_mirror_plane(modelmeta):
        vel_z = np.abs(vel_z)
    vel_r = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    index_r = np.searchsorted(vel_r_edges, vel_r, side="right") - 1
    ismoving = vel_r > 0.0
    costheta = np.where(ismoving, np.clip(vel_z / np.where(ismoving, vel_r, 1.0), -1.0, 1.0), costheta_edges[0])
    index_costheta = np.minimum(np.searchsorted(costheta_edges, costheta, side="right") - 1, ncoordgridcostheta - 1)

    cellindex = index_costheta * ncoordgridr + index_r
    if "phi_edges_radians" in modelmeta:
        phi = np.arctan2(vel_y, vel_x)
        phi = np.where(phi < 0.0, phi + 2.0 * math.pi, phi)
        phi = np.where(phi >= 2.0 * math.pi, 0.0, phi)
        index_phi = np.searchsorted(np.asarray(modelmeta["phi_edges_radians"], dtype=np.float64), phi, side="right") - 1
        cellindex += index_phi * (ncoordgridr * ncoordgridcostheta)
    return np.where(index_r < ncoordgridr, cellindex, -1).astype(np.int64)


def get_spherical_slice_velocity_cmps(slicevelocity_cmps: float, vmax_cmps: float) -> float:
    """Return the position of the plane of a slice of a spherical model, 1e-9 vmax above the given plane [cm/s].

    Each point of the image lies on the plane, and a cell holds its lower edge. Thus a plane on a cell edge took the
    cells of a different side in each half of the image. The plane above takes the cells on the positive side. The
    image takes the cell edges in float64, thus 1e-9 vmax is far above the rounding error. A plane less than 1e-9 vmax
    below vmax stays, because the plane above is outside the model.
    """
    planeabove_cmps = slicevelocity_cmps + 1.0e-9 * vmax_cmps
    return planeabove_cmps if planeabove_cmps < vmax_cmps else slicevelocity_cmps


# an image of 2000 x 2000 points takes approximately 130 MB for the cell search
MAX_IMAGE_POINTS = 1000


def get_spherical_image_cell_indices(
    modelmeta: "Mapping[str, t.Any]", sliceaxis: str | None, slicevelocity_cmps: float = 0.0
) -> "npt.NDArray[np.int64]":
    """Return the cell index of each point of an image of a spherical2d or spherical3d model, or -1 outside the grid.

    A plot of such a model has no cell with straight edges on the axes of an image. Thus the image is a grid of
    points with equal steps, as the image of a 1D model is. Each point takes the value of the cell that holds it.
    The count of the points from 0 to vmax is 200, or two for each radial shell to a maximum of MAX_IMAGE_POINTS. A
    cell that is narrower than the step of the points can have no point, and the image then does not show it.

    Without a sliceaxis, the image has the cylindrical radius from 0 to vmax and z from -vmax to vmax as its axes.
    The points are in the half-plane y = 0 with x > 0. With a sliceaxis, the points are in the plane normal to that
    axis at slicevelocity_cmps. The two other axes then go from -vmax to vmax, e.g. x and y for the sliceaxis z. The
    array has one row for each point of the second axis.
    """
    vmax_cmps = float(modelmeta["vmax_cmps"])
    pointcount = min(max(200, 2 * (len(modelmeta["vel_r_edges_kmps"]) - 1)), MAX_IMAGE_POINTS)
    pointwidth = vmax_cmps / pointcount
    positive_points = np.asarray((np.arange(pointcount) + 0.5) * pointwidth, dtype=np.float64)
    all_points = np.asarray((np.arange(2 * pointcount) + 0.5) * pointwidth - vmax_cmps, dtype=np.float64)
    velocities: dict[str, npt.NDArray[np.float64]]
    if sliceaxis is None:
        velocities = {"x": positive_points[np.newaxis, :], "y": np.zeros((1, 1)), "z": all_points[:, np.newaxis]}
    else:
        plotaxis1, plotaxis2 = (axisname for axisname in "xyz" if axisname != sliceaxis)
        velocities = {
            plotaxis1: all_points[np.newaxis, :],
            plotaxis2: all_points[:, np.newaxis],
            sliceaxis: np.full((1, 1), slicevelocity_cmps),
        }
    vel_x, vel_y, vel_z = np.broadcast_arrays(velocities["x"], velocities["y"], velocities["z"])
    return get_spherical_cell_index_of_velocities(vel_x, vel_y, vel_z, modelmeta)


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
    coordinates = get_model_coordinates(modelmeta)

    # rho is the source of logrho, because a caller changes the density as rho. A 1D model file gives logrho only,
    # thus a dataframe without rho gets it from logrho first
    if "rho" not in original_cols:
        dfmodel = dfmodel.with_columns(rho=RHO_FROM_LOGRHO)
    dfmodel = dfmodel.with_columns(logrho=LOGRHO_FROM_RHO)

    axes: list[str] = []
    match coordinates:
        case "spherical1d":
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

        case "cylindrical2d":
            axes = ["rcyl", "z"]
            # a cell of a mirrored model also fills its mirror cell
            mirrorfactor = 2.0 if has_mirror_plane(modelmeta) else 1.0

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
                    * mirrorfactor
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
                    * mirrorfactor
                ),
                kinetic_en_erg_z=(
                    1
                    / 6
                    * pl.col("rho")
                    * math.pi
                    * (pl.col("pos_rcyl_max").cast(pl.Float64).pow(2) - pl.col("pos_rcyl_min").cast(pl.Float64).pow(2))
                    * t_model_init_seconds**-2
                    * (pl.col("pos_z_max").cast(pl.Float64).pow(3) - pl.col("pos_z_min").cast(pl.Float64).pow(3))
                    * mirrorfactor
                ),
            )

        case "cartesian3d":
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

        case "spherical2d" | "spherical3d":
            # homologous expansion has only a radial velocity, thus the radial component is the full kinetic energy
            axes = ["r"]
            dfmodel = add_spherical_derived_cols(dfmodel, modelmeta)

        case _:
            msg = f"The coordinates {coordinates!r} are not known"
            raise ValueError(msg)

    # the total kinetic energy is the sum of the orthogonal components. A cylindrical model and a Cartesian model
    # also get a radial position, which is not a component. Thus "r" counts only where it is the sole axis
    orthogonal_axes = [ax for ax in axes if ax != "r"] or ["r"]
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


# Each version 1 file holds these columns after its position columns and its density. They are optional in a version
# 2 file, where an absent column holds zero in each cell
STANDARD_COMPOSITION_COLUMNS = ("X_Fegroup", "X_Ni56", "X_Co56", "X_Fe52", "X_Cr48")


def get_standard_columns(
    coordinates: str, includenico57: bool = False, pos_unknown: bool = False, formatversion: int = 1
) -> list[str]:
    """Return the columns that each model.txt holds for the coordinates of a model, e.g. cartesian3d.

    A version 2 file needs only inputcellid and rho. It has no position column, because the header gives the edges
    of the cells, and each composition column is optional. A model with spherical2d or spherical3d coordinates has
    version 2 only.
    """
    cols: list[str] = []
    if coordinates not in MODEL_EDGE_KEYS:
        msg = f"The coordinates {coordinates!r} are not known"
        raise ValueError(msg)
    match coordinates if formatversion == 1 else "version2":
        case "spherical1d":
            cols = ["inputcellid", "vel_r_max_kmps", "logrho"]
        case "cylindrical2d":
            cols = ["inputcellid", "pos_rcyl_mid", "pos_z_mid", "rho"]
        case "cartesian3d":
            cols = (
                ["inputcellid", "inputpos_a", "inputpos_b", "inputpos_c", "rho"]
                if pos_unknown
                else ["inputcellid", "pos_x_min", "pos_y_min", "pos_z_min", "rho"]
            )
        case _:
            cols = ["inputcellid", "rho"]

    if formatversion == 2:
        return cols

    cols += STANDARD_COMPOSITION_COLUMNS

    if includenico57:
        cols += ["X_Ni57", "X_Co57"]

    return cols


# A column X_<element>_otherstable of a version 2 model.txt holds the mass fraction of the stable isotopes of the
# element that have no nuclide column. Such a file has no abundances.txt
OTHERSTABLE_SUFFIX = "_otherstable"
# sn3d stops if the nuclide columns of an element hold this much more than the mass fraction of the element
OTHERSTABLE_NEGATIVE_LIMIT = 1e-2
# a remainder below this fraction of the mass fraction of the element is rounding of the Float32 values
OTHERSTABLE_ROUNDING_RTOL = 1e-5
# sn3d stops for a cell whose element mass fractions sum to more than this value, which permits rounding
ELEMENT_MASSFRAC_SUM_LIMIT = 1.0 + 1e-3
# sn3d stops for a mass fraction of this value or less, and it reads a value from this limit to 0 as 0
MIN_INPUT_MASSFRAC = -1e-6
# sn3d stops for X_Fegroup above this value, which permits rounding. A nuclide mass fraction has the maximum 1
MAX_FEGROUP_MASSFRAC = 1.0 + 1e-3


def get_reader_column_type(col: str) -> type[pl.DataType]:
    """Return the type that the readers of model.txt give a column.

    Float32 gives a tracer count of 16777217 as 16777216, and Float64 gives each whole number to 2^53 with no change.
    """
    return {"inputcellid": pl.Int32, "tracercount": pl.Float64}.get(col, pl.Float32)


def get_columns_above_max_massfrac(dfmodel: pl.LazyFrame) -> list[str]:
    """Return the mass fraction columns that hold a value above the maximum of sn3d in a cell with density.

    sn3d compares a nuclide mass fraction as Float32 with 1, and X_Fegroup with MAX_FEGROUP_MASSFRAC. An otherstable
    column has no maximum of its own, and the check of the element sum applies to it.
    """
    colnames = dfmodel.collect_schema().names()
    maxvalues = {
        col: 1.0
        for cols in get_element_columns_of_model(colnames).values()
        for col in cols
        if not col.endswith(OTHERSTABLE_SUFFIX)
    } | ({"X_Fegroup": MAX_FEGROUP_MASSFRAC} if "X_Fegroup" in colnames else {})
    if not maxvalues:
        return []
    # sn3d compares the Float32 value as a double, thus the cast to Float64 keeps a value just above 1.001 above it.
    # The writer gives a NaN as 0, and polars puts a NaN above each number
    isabove = dfmodel.select(
        ((pl.col(col).cast(pl.Float32).cast(pl.Float64).fill_nan(0.0) > maxvalue) & (pl.col("rho") > 0.0))
        .any()
        .alias(col)
        for col, maxvalue in maxvalues.items()
    ).collect()
    return [col for col in isabove.columns if isabove[col].item()]


def get_columns_below_min_massfrac(dfmodel: pl.LazyFrame) -> list[str]:
    """Return the mass fraction columns that hold MIN_INPUT_MASSFRAC or less in a cell with density.

    sn3d stops for such a value in a nuclide column, an otherstable column, and the column X_Fegroup.
    """
    isbelow = dfmodel.select(
        ((cs.starts_with("X_").as_expr() <= MIN_INPUT_MASSFRAC) & (pl.col("rho") > 0.0)).any()
    ).collect()
    return [col for col in isbelow.columns if isbelow[col].item()]


def raise_for_values_that_sn3d_refuses(dfcells: pl.LazyFrame, source: str) -> None:
    """Stop for a value of the cell lines of a version 2 model.txt that sn3d stops for.

    dfcells holds the cell lines, and source starts each message, e.g. the name of the file. sn3d stops for a value
    that is not finite in each line. It checks the range of a mass fraction, of q, and of Ye, and the sum of the
    element mass fractions, only in a cell with density. A null is a value that is not known, and .any() takes no
    account of a null, thus a null counts as a value that is not finite.
    """
    colnames = dfcells.collect_schema().names()
    hasdensity = pl.col("rho") > 0.0
    isfinite = {col: pl.col(col).cast(pl.Float64).is_finite().fill_null(value=False) for col in colnames}
    dfinvaliddensity = (
        dfcells.filter(~(isfinite["rho"] & (pl.col("rho") >= 0.0))).select("inputcellid", "rho").collect()
    )
    if dfinvaliddensity.height:
        msg = (
            f"{source}: {dfinvaliddensity.height} cells have a density that is not finite or is below 0, e.g. the cell"
            f" {dfinvaliddensity['inputcellid'].item(0)} with rho = {dfinvaliddensity['rho'].item(0)!r}. Each cell of"
            " format version 2 needs a finite density of 0 or more"
        )
        raise ValueError(msg)
    schema = dfcells.collect_schema()
    # the check of Ye below gives its own message
    valuecols = [col for col in colnames if col not in {"inputcellid", "rho", "Ye"} and schema[col].is_numeric()]
    isnotfinite = dfcells.select((~isfinite[col]).any().alias(col) for col in valuecols).collect()
    nonfinitecolumns = [col for col in isnotfinite.columns if isnotfinite[col].item()]
    if nonfinitecolumns:
        msg = (
            f"{source}: the columns {', '.join(nonfinitecolumns)} hold a value that is not finite, e.g. NaN. Each"
            " value of format version 2 must be a finite number"
        )
        raise ValueError(msg)
    if negativecolumns := get_columns_below_min_massfrac(dfcells):
        msg = (
            f"{source}: the columns {', '.join(negativecolumns)} hold a mass fraction of {MIN_INPUT_MASSFRAC:g} or"
            " less in a cell with density. sn3d stops for such a value"
        )
        raise ValueError(msg)
    if largecolumns := get_columns_above_max_massfrac(dfcells):
        msg = (
            f"{source}: the columns {', '.join(largecolumns)} hold a mass fraction above 1 in a cell with density, or"
            f" X_Fegroup above {MAX_FEGROUP_MASSFRAC:g}. sn3d stops for such a value"
        )
        raise ValueError(msg)
    if "q" in colnames and dfcells.filter(hasdensity & (pl.col("q") < 0.0)).select(pl.len()).collect().item():
        msg = f"{source}: a cell with density has a negative q. sn3d stops for it"
        raise ValueError(msg)
    if "Ye" in colnames:
        dfinvalidye = (
            dfcells.filter(~isfinite["Ye"] | (hasdensity & ~IS_VALID_YE)).select("inputcellid", "Ye").collect()
        )
        if dfinvalidye.height:
            msg = (
                f"{source}: the cell {dfinvalidye['inputcellid'].item(0)} has the electron fraction"
                f" {dfinvalidye['Ye'].item(0)!r}. Each value of the column Ye must be a finite number from 0 to 1"
            )
            raise ValueError(msg)
    # sn3d sums the nuclide columns and the otherstable columns of each element with Z of 1 or more
    elementcols = [col for cols in get_element_columns_of_model(colnames).values() for col in cols]
    raise_if_element_sum_above_limit(
        dfcells
        .filter(hasdensity)
        .select(
            "inputcellid",
            elementsum=pl.sum_horizontal([clamped_nuclide_massfrac(col) for col in elementcols] or [pl.lit(0.0)]),
        )
        .collect(),
        f"The cell lines of {source} give this sum.",
    )


def customcolsortkey(col: str) -> tuple[float, int]:
    """Sort nuclide mass fraction columns by atomic number then mass number, and other columns last.

    The otherstable column of an element comes after its nuclide columns.
    """
    if col.startswith("X_") and col.endswith(OTHERSTABLE_SUFFIX):
        return (get_atomic_number(col.removeprefix("X_").removesuffix(OTHERSTABLE_SUFFIX)), 10**6)
    return get_z_a_nucname(col) if col.startswith("X_") else (math.inf, 0)


def get_element_columns_of_model(colnames: Sequence[str]) -> dict[int, list[str]]:
    """Return the nuclide columns and the otherstable column of each element of a model, by atomic number.

    X_Fegroup, the free neutrons (X_n1), and an element column such as X_Fe are in no list.
    """
    elementcols: dict[int, list[str]] = {}
    for col in colnames:
        if col.startswith("X_") and col.endswith(OTHERSTABLE_SUFFIX):
            atomic_number = get_atomic_number(col.removeprefix("X_").removesuffix(OTHERSTABLE_SUFFIX))
        elif re.fullmatch(r"X_[A-Za-z]+\d+", col):
            atomic_number = get_z_a_nucname(col)[0]
        else:
            continue
        if atomic_number > 0:
            elementcols.setdefault(atomic_number, []).append(col)
    return elementcols


def get_elemabundances_of_model_columns(dfmodel: pl.LazyFrame) -> pl.LazyFrame:
    """Return the mass fraction of each element of each cell, as sn3d calculates it for a version 2 model.txt.

    The mass fraction of an element is the sum of its nuclide columns and of its column X_<element>_otherstable.
    sn3d reads a small negative value as 0, thus the sum takes such a value as 0. The result has the columns of
    scan_initelemabundances: inputcellid, and X_H to the last element (Z = 30 or more).
    """
    elementcols = get_element_columns_of_model(dfmodel.collect_schema().names())
    max_atomic_number = max([30, *elementcols])
    return dfmodel.select(
        "inputcellid",
        *[
            (
                pl.sum_horizontal(clamped_nuclide_massfrac(col) for col in elementcols[atomic_number])
                if atomic_number in elementcols
                else pl.lit(0.0)
            )
            .cast(pl.Float32)
            .alias(f"X_{get_elsymbol(atomic_number)}")
            for atomic_number in range(1, max_atomic_number + 1)
        ],
    )


def clamped_nuclide_massfrac(col: str) -> pl.Expr:
    """Return the mass fraction of a nuclide column, with 0 for a negative value or a NaN, as sn3d reads it.

    sn3d stops for a value below -1e-6 and reads a value from -1e-6 to 0 as 0. Thus the sum of the nuclide columns
    of an element must also take 0, or the otherstable value of the element gets the negative part. The writer gives
    a NaN as 0, and polars puts a NaN above each number in a comparison.
    """
    return pl.col(col).cast(pl.Float64).fill_nan(0.0).clip(lower_bound=0.0)


def raise_if_element_sum_above_limit(elementsums: pl.DataFrame, reason: str) -> None:
    """Stop if the element mass fractions of a cell sum to more than ELEMENT_MASSFRAC_SUM_LIMIT, as sn3d does.

    elementsums has the columns inputcellid and elementsum of the cells with density. The reason names the cause of
    such a sum, and the message gives it.
    """
    abovelimit = elementsums.filter(pl.col("elementsum") > ELEMENT_MASSFRAC_SUM_LIMIT)
    if not abovelimit.is_empty():
        msg = (
            f"The element mass fractions of {abovelimit.height} cells sum to more than {ELEMENT_MASSFRAC_SUM_LIMIT:g},"
            f" up to {abovelimit['elementsum'].max():.6g} in the cell with inputcellid"
            f" {abovelimit.sort('elementsum')['inputcellid'][-1]}. In format version 2, the mass fraction of an"
            f" element is the sum of its nuclide columns and its otherstable column. {reason} sn3d stops for such a"
            " cell"
        )
        raise ValueError(msg)


def with_otherstable_columns(dfmodel: pl.DataFrame, dfelabundances: pl.DataFrame) -> pl.DataFrame:
    """Return the cells of a model with the columns X_<element>_otherstable that give the element mass fractions.

    A version 2 model.txt has no abundances.txt. sn3d takes the mass fraction of an element as the sum of its nuclide
    columns and of its otherstable column. Thus the otherstable value of an element is its mass fraction in
    dfelabundances minus the sum of its nuclide columns in dfmodel. The function stops if the nuclide columns hold
    more than the element by OTHERSTABLE_NEGATIVE_LIMIT. A remainder below OTHERSTABLE_ROUNDING_RTOL of the element
    is rounding and becomes 0, and an element with no remainder in any cell gets no column.

    The function stops if a cell with density has no row in dfelabundances. Such a cell got no otherstable value,
    thus it lost the elements that its nuclide columns do not hold. It also stops if the element mass fractions of a
    cell sum to more than ELEMENT_MASSFRAC_SUM_LIMIT, which sn3d also stops for. Such a sum can come from nuclide
    columns that hold more than their element, because the otherstable value of that element is then 0.

    A column of dfelabundances that is not an element, e.g. X_n for the free neutrons, gets no otherstable column.
    sn3d reads X_n_otherstable as nitrogen. The writer of abundances.txt also leaves out such a column.
    """
    dfmodel = dfmodel.drop(cs.ends_with(OTHERSTABLE_SUFFIX))
    elementcols = get_element_columns_of_model(dfmodel.columns)
    hasdensity = get_has_density_expr(dfmodel.columns)
    cellswithnoelements = dfmodel.filter(hasdensity).join(
        dfelabundances.select("inputcellid"), on="inputcellid", how="anti"
    )
    if not cellswithnoelements.is_empty():
        msg = (
            f"The element abundances have no row for {cellswithnoelements.height} cells with density, e.g. the cell"
            f" with inputcellid {cellswithnoelements['inputcellid'].min()}. A model.txt of format version 2 gets its"
            " element mass fractions from these rows, thus each cell with density needs one"
        )
        raise ValueError(msg)
    # a null or a NaN is a value that is not known, and the otherstable column of such an element would get 0
    elementfraccols = [col for col in dfelabundances.columns if col != "inputcellid"]
    cellswithnullelement = (
        dfmodel
        .filter(hasdensity)
        .select("inputcellid")
        .join(dfelabundances, on="inputcellid", how="inner")
        .filter(pl.any_horizontal(~pl.col(elementfraccols).is_finite().fill_null(value=False)))
        if elementfraccols
        else pl.DataFrame()
    )
    if not cellswithnullelement.is_empty():
        msg = (
            f"The element abundances have no finite value for {cellswithnullelement.height} cells with density, e.g."
            f" the cell with inputcellid {cellswithnullelement['inputcellid'].min()}. Each element mass fraction of"
            " such a cell must be a finite number"
        )
        raise ValueError(msg)
    remainders = {}
    for elcol in dfelabundances.columns:
        elsymbol = elcol.removeprefix("X_")
        atomic_number = get_atomic_number(elsymbol)
        # get_atomic_number does not use the case of the symbol, thus n for the free neutrons gives nitrogen
        if elcol == "inputcellid" or atomic_number < 1 or get_elsymbol(atomic_number) != elsymbol:
            continue
        nuclidecols = elementcols.get(atomic_number, [])
        elmassfrac = pl.col(elcol).fill_null(0.0).cast(pl.Float64)
        remainder = elmassfrac - (
            pl.sum_horizontal([clamped_nuclide_massfrac(col) for col in nuclidecols]) if nuclidecols else pl.lit(0.0)
        )
        remainders[f"{elcol}{OTHERSTABLE_SUFFIX}"] = (
            pl
            .when(hasdensity & (remainder.abs() > OTHERSTABLE_ROUNDING_RTOL * elmassfrac))
            .then(remainder)
            .otherwise(0.0)
        )
    nuclidecolsofmodel = [col for cols in elementcols.values() for col in cols]
    densitycols = [col for col in ("rho", "logrho") if col in dfmodel.columns]
    dfremainders = (
        dfmodel
        .select("inputcellid", *densitycols, *nuclidecolsofmodel)
        .join(dfelabundances, on="inputcellid", how="left", maintain_order="left")
        .select(**remainders)
    )
    # sn3d takes the sum of the nuclide columns and the otherstable values of each element with Z of 1 or more
    elementsums = (
        dfmodel
        .select(
            "inputcellid",
            hasdensity.alias("hasdensity"),
            pl.sum_horizontal([clamped_nuclide_massfrac(col) for col in nuclidecolsofmodel] or [pl.lit(0.0)]).alias(
                "nuclidesum"
            ),
        )
        .with_columns(
            elementsum=pl.col("nuclidesum")
            + (
                dfremainders.select(pl.sum_horizontal(pl.all().clip(lower_bound=0.0))).to_series()
                if remainders
                else 0.0
            )
        )
        .filter(pl.col("hasdensity"))
    )
    raise_if_element_sum_above_limit(
        elementsums, "The nuclide columns of an element hold more than its mass fraction in the element abundances."
    )
    if not remainders:
        # the nuclide columns alone give each element
        print("  0 elements get an otherstable column, because the element abundances have no element column.")
        return dfmodel
    lowest: dict[str, float] = dfremainders.select(pl.all().min().fill_null(0.0)).row(0, named=True)
    highest: dict[str, float] = dfremainders.select(pl.all().max().fill_null(0.0)).row(0, named=True)
    for col, lowestremainder in lowest.items():
        if lowestremainder < -OTHERSTABLE_NEGATIVE_LIMIT:
            msg = (
                f"The nuclide columns of {col.removeprefix('X_').removesuffix(OTHERSTABLE_SUFFIX)} hold up to"
                f" {-lowestremainder:.3g} more mass fraction than the element has in the element abundances"
            )
            raise ValueError(msg)
    otherstablecols = [col for col, highestremainder in highest.items() if highestremainder > 0.0]
    negativecount = dfremainders.select(pl.sum_horizontal(pl.all() < 0.0).sum()).item()
    if negativecount > 0:
        # sn3d gives the element the sum of its nuclide columns in this case
        print_warning(
            f"{negativecount} values of a cell and an element have nuclide columns that hold more than the element,"
            f" by up to {-min(lowest.values()):.3g}. The element mass fraction becomes the sum of its nuclide columns"
        )
    print(
        f"  {len(otherstablecols)} elements get an otherstable column, which is the element mass fraction minus the sum"
        f" of the nuclide columns. A remainder below {OTHERSTABLE_ROUNDING_RTOL:g} of the element is rounding."
    )
    return dfmodel.with_columns(dfremainders.select(pl.col(otherstablecols).clip(lower_bound=0.0).cast(pl.Float32)))


def write_artis_csv(df: pl.DataFrame, fileobj: t.IO[bytes], significantfigures: int = 8) -> None:
    """Append the dataframe as one zstd frame in the ARTIS text file format: space separated and no header.

    The relative spacing of the Float32 values that the reader gives is 6e-8. Eight significant figures give a value
    to 5e-8, and a file of format version 1 has them. Nine figures give each Float32 value again with no change, and
    a model.txt of format version 2 has them. Five figures lost more precision than the reader does, which showed up
    as a mass that changed by 2e-5 when a model was written and read again.
    """
    df.write_csv(
        fileobj,
        include_header=False,
        separator=" ",
        line_terminator="\n",
        float_scientific=True,
        float_precision=significantfigures - 1,
        null_value="0.0",
        compression="zstd",
    )


# the files of an ARTIS input model that a command writes. confirm_overwrite also finds the compressed copies
MODEL_FILE_NAMES = ("model.txt", "abundances.txt", "gridcontributions.txt")

# The files of a model folder of format version 1 that give a value for a cell id. A model of format version 2 has
# none of them, and save_modeldata deletes them beside such a model
CELL_ID_FILE_NAMES = ("Ye.txt", "opacity.txt")


def prepare_zstd_output(filepath: Path) -> Path:
    """Return the path of the zstd file for filepath, e.g. model.txt.zst, and delete each old copy and the cache.

    If model.txt and model.txt.zst both exist, a reader uses model.txt. Thus an old uncompressed copy must not stay
    beside a new compressed file. A write that stops early must not leave the cache of the old file beside a part of
    the new file.
    """
    zstdpath = with_compressed_extension(without_compressed_extension(filepath), ".zst")
    for oldpath in get_file_copies(zstdpath):
        if oldpath != zstdpath and oldpath.is_file():
            oldpath.unlink()
            print(f"Deleted {oldpath}, because {zstdpath.name} replaces it")
    remove_parquet_cache(zstdpath)
    return zstdpath


def get_kmps_of_cmps(velocity_cmps: float) -> float:
    """Return the velocity [km/s] that gives velocity_cmps [cm/s] again when a reader multiplies it by km_to_cm.

    A plain division can give a value that is one bit different after the multiplication. sn3d then gets a different
    vmax from the version 2 file than from the version 1 file of the same model.
    """
    velocity_kmps = velocity_cmps / km_to_cm
    for candidate in (velocity_kmps, math.nextafter(velocity_kmps, math.inf), math.nextafter(velocity_kmps, -math.inf)):
        if candidate * km_to_cm == velocity_cmps:
            return candidate
    return velocity_kmps


def get_edges_of_position_columns(dfmodel: pl.DataFrame, modelmeta: dict[str, t.Any]) -> dict[str, list[float]]:
    """Return the cell edges of version 2 for a spherical1d, cylindrical2d, or cartesian3d model.

    dfmodel holds the cells in the sequence of inputcellid. The edges of a 1D model are the values of vel_r_max_kmps
    after 0. The edges of the other two have equal steps to vmax_cmps of the metadata. The function stops if a
    position column of dfmodel does not agree with these edges.
    """
    coordinates = modelmeta["coordinates"]
    if coordinates == "spherical1d":
        # str gives the shortest text of a Float32, e.g. 14315.06 and not 14315.0595703125
        vel_r_max_kmps = dfmodel["vel_r_max_kmps"].cast(pl.Float32).to_numpy()
        edges_kmps = [0.0, *[float(str(vel_kmps)) for vel_kmps in vel_r_max_kmps]]
        # the edges of the metadata have the full precision of the file that gave them. Use them if no caller
        # changed the column
        metaedges = modelmeta.get("vel_r_edges_kmps")
        if (
            metaedges is not None
            and len(metaedges) == len(edges_kmps)
            and np.array_equal(np.asarray(metaedges, dtype=np.float32)[1:], vel_r_max_kmps)
        ):
            edges_kmps = [float(edge) for edge in metaedges]
        return {"vel_r_edges_kmps": edges_kmps}

    vmax_kmps = get_kmps_of_cmps(modelmeta["vmax_cmps"])
    if coordinates == "cylindrical2d":
        cellcounts = {"rcyl": modelmeta["ncoordgridrcyl"], "z": modelmeta["ncoordgridz"]}
    else:
        griddimension = round(dfmodel.height ** (1.0 / 3.0))
        cellcounts = dict.fromkeys(("x", "y", "z"), griddimension)
    if math.prod(cellcounts.values()) != dfmodel.height:
        gridshape = " x ".join(str(count) for count in cellcounts.values())
        msg = f"A {coordinates} grid of {gridshape} cells cannot hold {dfmodel.height} cells"
        raise ValueError(msg)
    # the z axis of a mirrored model starts at the plane z = 0
    firstedges = {"rcyl": 0.0, "z": 0.0 if has_mirror_plane(modelmeta) else -vmax_kmps}
    edges = {
        f"vel_{axis}_edges_kmps": np.linspace(firstedges.get(axis, -vmax_kmps), vmax_kmps, count + 1).tolist()
        for axis, count in cellcounts.items()
    }

    # a version 2 file gives no position of a cell, thus a cell in the wrong place of the sequence has no other check
    modelmeta_edges = modelmeta | edges | {f"ncoordgrid{axis}": count for axis, count in cellcounts.items()}
    dfexpected = add_position_columns_of_edges(
        dfmodel.lazy().select("inputcellid", "rho"), modelmeta_edges, dfmodel["inputcellid"].item(0)
    ).collect()
    t_model_init_seconds = modelmeta["t_model_init_days"] * day_to_s
    for col in dfexpected.columns:
        if col.startswith("pos_") and col in dfmodel.columns:
            axis = col.split("_")[1]
            cellwidth = (edges[f"vel_{axis}_edges_kmps"][1] - edges[f"vel_{axis}_edges_kmps"][0]) * km_to_cm
            maxoffby = (dfmodel[col] - dfexpected[col]).abs().max()
            assert isinstance(maxoffby, float)
            if maxoffby > 0.5 * cellwidth * t_model_init_seconds:
                msg = (
                    f"{col} is up to {maxoffby:.3e} cm from the position that the cell ids give on a grid of equal"
                    f" cells to vmax_cmps = {modelmeta['vmax_cmps']:.8e} cm/s"
                )
                raise ValueError(msg)
    return edges


def get_version2_headerlines(inputcellids: pl.Series, modelmeta: dict[str, t.Any]) -> list[str]:
    """Return the key lines of a version 2 model.txt, from the key coordinates to the last list of edges.

    The function checks the model time and the cell edges of the metadata, and the sorted cell ids against the edges.
    The cell ids of a version 2 file are 0 to N - 1, where N is the count of the cells. The function sets the cell
    counts in the metadata, and for spherical2d and spherical3d also vmax_cmps and firstinputcellid. Each edge
    has the shortest text that gives its float again, e.g. 0.7. A line has a comment only for information that
    its key does not give.
    """
    if modelmeta.get("mirror_plane") not in {None, "z=0"}:
        msg = f"mirror_plane must be 'z=0' or absent, not {modelmeta['mirror_plane']!r}"
        raise ValueError(msg)
    # the reader of version 2 rejects a different model time, thus the writer must not write it
    tmodel_days = float(modelmeta["t_model_init_days"])
    if not math.isfinite(tmodel_days) or tmodel_days <= 0.0:
        msg = f"t_model_init_days must be one finite value above 0, not {modelmeta['t_model_init_days']!r}"
        raise ValueError(msg)
    coordinates = modelmeta["coordinates"]
    edgekeys = MODEL_EDGE_KEYS[coordinates]
    edges = validate_model_edges(
        coordinates,
        {key: modelmeta.get(key) for key in ALL_MODEL_EDGE_KEYS},
        "save_modeldata",
        mirrored=has_mirror_plane(modelmeta),
    )
    vmax_cmps = edges[edgekeys[0]][-1] * km_to_cm
    if modelmeta.get("vmax_cmps") is not None and not math.isclose(modelmeta["vmax_cmps"], vmax_cmps, rel_tol=1e-6):
        msg = (
            f"vmax_cmps = {modelmeta['vmax_cmps']:.8e} cm/s does not agree with the last value of {edgekeys[0]},"
            f" {edges[edgekeys[0]][-1]!r} km/s"
        )
        raise ValueError(msg)

    cellcounts = [len(edges[key]) - 1 for key in edgekeys]
    gridshape = " x ".join(str(count) for count in cellcounts)
    npts_model = math.prod(cellcounts)
    print(f" {coordinates} grid size: {inputcellids.len()} ({gridshape})")

    firstaxis = edgekeys[0].split("_")[1]
    # the writer gives a null cell id as 0.0, and .all() takes no account of a null. A comparison of two series of
    # different lengths raises, thus the length test comes first
    idsagree = (
        inputcellids.len() == npts_model
        and (inputcellids == pl.int_range(npts_model, eager=True)).fill_null(value=False).all()
    )
    if not idsagree:
        msg = (
            f"The cell edges give {gridshape} = {npts_model} cells, but the model has {inputcellids.len()} cells with"
            f" the first cell id {inputcellids.item(0) if inputcellids.len() else None}. In format version 2, the cell"
            f" ids are 0 to {npts_model - 1}, and the index of {firstaxis} changes fastest. Give the same cell ids to"
            " gridcontributions.txt, e.g. call with_cell_ids_from_zero"
        )
        raise ValueError(msg)

    modelmeta |= edges | get_cell_counts_of_edges(coordinates, edges)
    if coordinates in SPHERICAL_EDGE_COORDINATES:
        modelmeta |= {"vmax_cmps": vmax_cmps, "firstinputcellid": 0}
    edgecomments = {
        "costheta_edges": "  # the cosine of the polar angle from the +z axis",
        "phi_edges_radians": "  # the azimuthal angle phi = atan2(y, x)",
    }
    return [
        f"coordinates {coordinates}",
        f"cells_per_axis {' '.join(str(count) for count in cellcounts)}",
        *(["mirror_plane z=0"] if has_mirror_plane(modelmeta) else []),
        f"t_model_init_days {modelmeta['t_model_init_days']}",
        *[f"{key} {' '.join(repr(edge) for edge in edges[key])}{edgecomments.get(key, '')}" for key in edgekeys],
    ]


def with_rho_of_1d_model(dfmodel: pl.DataFrame) -> pl.DataFrame:
    """Return the cells of a 1D model with the column rho [g/cm3].

    A version 2 file holds rho in place of logrho.

    A version 1 file gives logrho only. A dataframe can hold both columns, e.g. from a version 2 file. The function
    then stops if they do not agree (see require_agreement_of_rho_and_logrho).
    """
    if "rho" not in dfmodel.columns:
        return dfmodel.with_columns(rho=RHO_FROM_LOGRHO.cast(pl.Float32))
    require_agreement_of_rho_and_logrho(dfmodel)
    return dfmodel


def require_agreement_of_rho_and_logrho(dfmodel: pl.DataFrame) -> None:
    """Stop if the dataframe of a 1D model holds rho and logrho, and the two columns do not agree.

    A dataframe can hold both columns, e.g. from a version 2 file. A writer reads one of them, and it cannot know
    which column a caller changed. An empty cell agrees if rho is 0 and logrho is -98 or less.
    """
    if "rho" not in dfmodel.columns or "logrho" not in dfmodel.columns:
        return
    isemptyinboth = (pl.col("rho") == 0.0) & (pl.col("logrho") <= -98.0)
    logdifference = pl.when(isemptyinboth).then(0.0).otherwise((LOGRHO_FROM_RHO - pl.col("logrho")).abs())
    maxlogdifference = dfmodel.select(logdifference.max()).item()
    if maxlogdifference is not None and maxlogdifference > 1e-4:
        msg = (
            f"The columns rho and logrho of the 1D model do not agree: log10(rho) and logrho have a difference of up"
            f" to {maxlogdifference:.3g}. Change both columns, or drop one of them"
        )
        raise ValueError(msg)


def with_cell_ids_from_zero(
    dfmodel: pl.DataFrame, dfelabundances: pl.DataFrame | None = None, dfgridcontributions: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None]:
    """Return the cells, the element abundances, and the particle contributions with cell ids that start at 0.

    The cell ids of model.txt are 0 to N - 1 in format version 2, where N is the count of the cells.
    gridcontributions.txt of the same model must hold the same ids, and dfelabundances must agree with dfmodel. Thus
    one call changes all three dataframes by the first id of dfmodel.
    """
    firstinputcellid = dfmodel["inputcellid"].min()
    assert isinstance(firstinputcellid, int)
    if firstinputcellid == 0:
        return dfmodel, dfelabundances, dfgridcontributions
    return (
        dfmodel.with_columns(pl.col("inputcellid") - firstinputcellid),
        None if dfelabundances is None else dfelabundances.with_columns(pl.col("inputcellid") - firstinputcellid),
        (
            None
            if dfgridcontributions is None
            else dfgridcontributions.with_columns(pl.col("cellindex") - firstinputcellid)
        ),
    )


def get_written_format_version(coordinates: str, modelmeta: dict[str, t.Any]) -> int:
    """Return the format version that save_modeldata writes: the key formatversion of the metadata, or its default.

    The default is 1, which each version of sn3d reads. A model with spherical2d or spherical3d coordinates or with
    mirror symmetry (the key mirror_plane) has version 2 only.
    """
    formatversion = modelmeta.get("formatversion")
    if coordinates in SPHERICAL_EDGE_COORDINATES or has_mirror_plane(modelmeta):
        if formatversion == 1:
            missing = f"{coordinates} coordinates" if coordinates in SPHERICAL_EDGE_COORDINATES else "mirror symmetry"
            msg = f"Format version 1 of model.txt has no {missing}. Use format version 2"
            raise ValueError(msg)
        return 2
    if formatversion not in {None, 1, 2}:
        msg = f"formatversion must be 1 or 2, not {formatversion!r}"
        raise ValueError(msg)
    return int(formatversion or 1)


def save_modeldata(
    dfmodel: pl.LazyFrame | pl.DataFrame,
    outpath: Path | str | None = None,
    modelmeta: dict[str, t.Any] | None = None,
    extracols: Sequence[str] = (),
    dfelabundances: pl.DataFrame | pl.LazyFrame | None = None,
    deletecellidfiles: bool = False,
    **kwargs: t.Any,
) -> None:
    """Write model.txt, a snapshot of the density and the composition, from the cell properties and the metadata.

    dfelabundances holds the element mass fractions of the cells, in the format of scan_initelemabundances. For
    format version 1, the function then also writes abundances.txt. Format version 2 has no abundances.txt: the
    function writes a column X_<element>_otherstable for each element whose nuclide columns do not hold its mass
    fraction (see with_otherstable_columns), and it deletes an abundances.txt of the output folder.

    The metadata gives values such as the time after the explosion. The file gets zstd compression, and the extension
    .zst, e.g. model.txt.zst for model.txt or for model.txt.gz. ARTIS reads such a file directly.

    1D
    -------
    dfmodel must contain columns inputcellid, vel_r_max_kmps, logrho, X_Fegroup, X_Ni56, X_Co56, X_Fe52, X_Cr48
    modelmeta is not required

    2D cylindrical
    -------
    dfmodel must contain columns inputcellid, pos_rcyl_mid, pos_z_mid, rho, X_Fegroup, X_Ni56, X_Co56, X_Fe52, X_Cr48
    modelmeta must define: vmax_cmps, ncoordgridrcyl and ncoordgridz

    3D Cartesian
    -------
    dfmodel must contain columns inputcellid, pos_x_min, pos_y_min, pos_z_min, rho, X_Fegroup, X_Ni56, X_Co56,
    X_Fe52, X_Cr48
    modelmeta must define: vmax_cmps

    2D spherical and 3D spherical
    -------
    dfmodel must contain columns inputcellid and rho
    modelmeta must define: coordinates (spherical2d or spherical3d), vel_r_edges_kmps, costheta_edges, and for
    spherical3d also phi_edges_radians

    The format version
    -------
    The key formatversion of modelmeta gives the version of the file: 1 or 2. Without the key, a 1D, a 2D
    cylindrical, or a 3D Cartesian model gets version 1, which each version of sn3d reads. scan_modeldata sets the
    key, thus a model that a command reads and writes again keeps its version. A 2D spherical or a 3D spherical model
    has version 2 only.

    A version 2 file starts with the line "format_version 2". Its header has one key on each line, in any sequence,
    and the key columns ends the header:

    - coordinates: spherical1d, cylindrical2d, cartesian3d, spherical2d, or spherical3d;
    - cells_per_axis: the count of the cells on each axis, in the sequence of the edge lists, e.g. 5 4 4 for
      n_r = 5, n_costheta = 4, and n_phi = 4. The key is optional, and a reader stops if a count is different from the
      count that the edges give;
    - mirror_plane z=0 (optional, from the key mirror_plane of modelmeta): mirror symmetry in the plane z = 0, for
      cylindrical2d, spherical2d, and spherical3d. The file then holds the cells with z >= 0: the z edges start at 0,
      and the list of cos(theta) starts at 0. Each cell also fills its mirror cell, thus its volume and its mass are two
      times the values of the half cell. Format version 1 has no mirror symmetry;
    - t_model_init_days;
    - the cell edges of each axis: vel_r_edges_kmps (and costheta_edges, phi_edges_radians), or
      vel_rcyl_edges_kmps and vel_z_edges_kmps, or vel_x_edges_kmps, vel_y_edges_kmps, and vel_z_edges_kmps;
    - cell_line_count: the count of the cell lines, thus a reader finds a file that lost lines at its end;
    - columns: inputcellid, rho, and the other column names.

    Each list of edges has one value more than the count of the cells on that axis, and it increases. A list of r,
    rcyl, or phi starts at 0, and the last azimuthal angle is 2 pi. costheta_edges holds the cosine of the polar
    angle theta from the +z axis, from -1 to 1, and the azimuthal angle is phi = atan2(y, x). A cylindrical or a
    Cartesian grid has cells of equal width from -vmax (0 for rcyl) to vmax. sn3d and artistools hold no other grid
    of these types. For these two, the function makes the edges from vmax_cmps and stops if a position column does
    not agree.

    The cells of a version 2 file have no position column. The cell id is the index of the cell in the grid of the
    edges, from 0. The index of the first axis changes fastest, e.g. i_r + n_r * (i_costheta + n_costheta * i_phi).
    Thus dfmodel must have the cell ids 0 to N - 1 for version 2, where N is the count of the cells.
    gridcontributions.txt of the model must hold the same ids (see with_cell_ids_from_zero). A cell with no line is
    empty, thus the function writes no line for an empty cell (rho = 0). A file needs one cell line, thus a model
    with no density keeps the line of its first cell. Each version 2 file holds rho, and no version 2 file holds
    logrho.

    A version 2 file needs only the columns inputcellid and rho. The columns X_Fegroup, X_Ni56, X_Co56, X_Fe52,
    X_Cr48, X_Ni57, and X_Co57 have a fixed place in version 1. In version 2, such a column that is absent holds zero
    in each cell. Thus the function writes one of these columns only if it is not zero in a cell with density.

    A version 2 model has no Ye.txt, because the column Ye of model.txt gives the electron fraction. It also has no
    opacity.txt, which sn3d does not read. When the function writes a version 2 file named model.txt, it stops for a
    Ye.txt or an opacity.txt of the output folder. Such a file can hold the only copy of its values.

    Give deletecellidfiles=True to delete these files. Give it after a merge of Ye.txt (see with_ye_of_yefile). Also
    give it when the user agreed to replace the files of an earlier model. The function does not change these two
    files when it writes a version 1 file. It also does not change them when it writes a file with a different name
    (e.g. model_alt.txt), which is not the model of the folder.

    A version 2 model has no abundances.txt. sn3d takes the mass fraction of an element as the sum of its nuclide
    columns and of its column X_<element>_otherstable, e.g. X_Ni56 + X_Ni58 + X_Ni_otherstable. Give dfelabundances
    to get the otherstable columns. Each mass fraction is a true mass fraction, because sn3d does not normalise the
    values of a cell.

    model.txt holds these comments:

    - a comment line that gives the creation time;
    - a comment line that gives the column units;
    - in version 1, an inline comment after each header value, and a comment line that gives the column names;
    - in version 2, an inline comment only for information that the key does not give, e.g. the axis of the polar angle.

    For a 1D model or a 2D model of version 1, sn3d reads these column names from v2024.04.

    The function deletes the parquet cache of the file that it writes. A LazyFrame from scan_modeldata of the same
    file can scan that cache, thus collect such a LazyFrame before this call.

    model.txt gets the standard columns, each X_ column, and the custom columns that ARTIS reads (Ye, q, and
    tracercount) if dfmodel holds them. A caller names each other custom column in extracols. model.txt gets no
    other column, e.g. no derived column.
    """
    assert isinstance(dfmodel, (pl.LazyFrame, pl.DataFrame))

    if modelmeta is None:
        modelmeta = {}

    assert all(
        key not in modelmeta or modelmeta[key] == kwargs[key] for key in kwargs
    )  # can't define the same thing twice unless the values are the same

    modelmeta |= kwargs  # add any extra keyword arguments to modelmeta

    headercommentlines = modelmeta.get("headercommentlines")
    vmax = modelmeta.get("vmax_cmps")

    # the columns give the dimensions of a spherical1d, cylindrical2d, or cartesian3d model. The columns of a
    # spherical2d model and a spherical3d model do not give the coordinates, thus the caller names them
    if modelmeta.get("dimensions") is None and modelmeta.get("coordinates") is None:
        modelmeta["dimensions"] = get_dfmodel_dimensions(dfmodel)

    if modelmeta.get("dimensions") not in {None, 1, 2, 3}:
        msg = f"dimensions must be 1, 2, or 3, not {modelmeta['dimensions']}"
        raise ValueError(msg)

    coordinates = get_model_coordinates(modelmeta)
    modelmeta["coordinates"] = coordinates
    modelmeta["dimensions"] = int(coordinates[-2])

    formatversion = get_written_format_version(coordinates, modelmeta)
    modelmeta["formatversion"] = formatversion
    colnames = dfmodel.collect_schema().names()
    if "inputcellid" not in colnames and "modelgridindex" in colnames:
        # the cell ids of version 2 start at 0. artistools gives the first cell of a version 1 file the id 1
        dfmodel = dfmodel.with_columns(inputcellid=pl.col("modelgridindex") + (0 if formatversion == 2 else 1))
        colnames = [*colnames, "inputcellid"]

    outputfolder = resolve_outputfile(outpath, "model.txt").parent
    # sn3d reads the abundances.txt and the Ye.txt beside model.txt. A file with a different name, e.g.
    # model_alt.txt, is a different model, thus its write leaves these files of the folder as they are
    writesmodeltxt = without_compressed_extension(resolve_outputfile(outpath, "model.txt")).name == "model.txt"
    oldabundancefiles = [
        path for path in get_file_copies(outputfolder / "abundances.txt") if writesmodeltxt and path.is_file()
    ]
    oldcellidfiles = [
        path
        for filename in CELL_ID_FILE_NAMES
        for path in get_file_copies(outputfolder / filename)
        if formatversion == 2 and writesmodeltxt and path.is_file()
    ]
    if oldcellidfiles and not deletecellidfiles:
        msg = (
            f"{oldcellidfiles[0]} gives values of the cells, and a model of format version 2 has no such file. Merge"
            " the values of Ye.txt with with_ye_of_yefile. Then give deletecellidfiles=True. As an alternative, delete"
            " the file"
        )
        raise ValueError(msg)
    if formatversion == 2:
        if dfelabundances is not None:
            dfmodel = with_otherstable_columns(dfmodel.lazy().collect(), dfelabundances.lazy().collect())
            colnames = dfmodel.columns
        elif oldabundancefiles:
            msg = (
                f"{outputfolder} holds abundances.txt, and a model of format version 2 has no such file. Give"
                " dfelabundances, or delete the file if the nuclide columns already give each element"
            )
            raise ValueError(msg)
    else:
        # only format version 2 has otherstable columns
        dfmodel = dfmodel.lazy().drop(cs.ends_with(OTHERSTABLE_SUFFIX))
        colnames = dfmodel.collect_schema().names()

    # the Ni57 and Co57 columns are optional, but their position counts. They must come before
    # every other custom column
    includenico57 = "X_Ni57" in colnames or "X_Co57" in colnames
    # version 1 gives the position columns of dfmodel. The edges of a version 2 header come from the same columns
    inputcols = get_standard_columns(coordinates, includenico57=includenico57)
    standardcols = get_standard_columns(coordinates, includenico57=includenico57, formatversion=formatversion)
    writtencustomcols = {"Ye", "q", "tracercount", *extracols}
    customcols = sorted(
        (col for col in colnames if col not in inputcols and (col.startswith("X_") or col in writtencustomcols)),
        key=customcolsortkey,
    )

    # the cast to Int32 below truncates a value such as 0.5, and the id check of version 2 then accepts it
    if (
        dfmodel.collect_schema()["inputcellid"].is_float()
        and not dfmodel.lazy().select(IS_WHOLE_INPUTCELLID.all()).collect().item()
    ):
        msg = "Each inputcellid must be a whole number, but the column inputcellid holds a different value"
        raise ValueError(msg)
    # the select comes before the collect, thus a lazy query does not calculate the columns that model.txt does not get
    dfmodel = (
        dfmodel
        .lazy()
        .with_columns(pl.lit(0.0).alias(col) for col in inputcols if col.startswith("X_") and col not in colnames)
        .select(*inputcols, *(["rho"] if "rho" in colnames and "rho" not in inputcols else []), *customcols)
        .with_columns(pl.col("inputcellid").cast(pl.Int32))
        .collect()
    )
    version2keylines: list[str] = []
    hasdensity = pl.col("rho") > 0
    if coordinates == "spherical1d" and formatversion == 1:
        # the writer of version 1 reads logrho alone, thus logrho must agree with a changed rho
        require_agreement_of_rho_and_logrho(dfmodel)
    if formatversion == 2:
        dfmodel = dfmodel.sort("inputcellid")
        if coordinates not in SPHERICAL_EDGE_COORDINATES:
            modelmeta |= get_edges_of_position_columns(dfmodel, modelmeta)
        if coordinates == "spherical1d":
            dfmodel = with_rho_of_1d_model(dfmodel)
        # the writer gives no line to a cell with no density. The reader then gives rho = 0 to a NaN density
        isvaliddensity = (pl.col("rho").is_finite() & (pl.col("rho") >= 0.0)).fill_null(value=False)
        dfinvaliddensity = dfmodel.filter(~isvaliddensity)
        if dfinvaliddensity.height:
            msg = (
                f"{dfinvaliddensity.height} cells have a density that is not finite or is below 0, e.g. the cell"
                f" {dfinvaliddensity['inputcellid'].item(0)} with rho = {dfinvaliddensity['rho'].item(0)!r}. Each"
                " cell of format version 2 needs a finite density of 0 or more"
            )
            raise ValueError(msg)
        # The writer gives a line to each cell with density, and it gives a null or a NaN mass fraction as 0. The
        # reader applies the same checks to the file
        writtencols = dict.fromkeys([
            "inputcellid",
            "rho",
            *(col for col in inputcols if col.startswith("X_")),
            *customcols,
        ])
        raise_for_values_that_sn3d_refuses(
            dfmodel
            .lazy()
            .filter(hasdensity)
            .select(list(writtencols))
            .with_columns(cs.starts_with("X_").fill_nan(0.0).fill_null(0.0)),
            "The model",
        )
        # an absent composition column holds zero in version 2. The writer gives a null and a NaN as zero
        compositioncols = [col for col in inputcols if col.startswith("X_")]
        isnonzero = dfmodel.select(
            ((pl.col(col).fill_nan(0.0).fill_null(0.0) != 0.0) & hasdensity).any() for col in compositioncols
        ).row(0, named=True)
        standardcols = [*standardcols, *[col for col in compositioncols if isnonzero[col]]]
        version2keylines = get_version2_headerlines(dfmodel["inputcellid"], modelmeta)

    dfmodel_npts_model = dfmodel.height
    if "npts_model" in modelmeta:
        assert modelmeta["npts_model"] == dfmodel_npts_model
    else:
        modelmeta["npts_model"] = dfmodel_npts_model

    timestart = time.perf_counter()
    tmodelline = (str(modelmeta["t_model_init_days"]), "t_model_init_days: time of the snapshot [day]")
    # the lines of numbers of a version 1 header. A version 2 header has key lines in their place
    headerlines: list[tuple[str, str]] = []
    match coordinates if formatversion == 1 else "version2":
        case "spherical1d":
            print(f" 1D grid radial bins: {dfmodel_npts_model}")
            strunits = "vel_r_max_kmps [km/s], logrho = log10(rho [g/cm^3]) at t_model_init_days"
            headerlines = [(str(dfmodel_npts_model), "npts_model: number of radial cells"), tmodelline]

        case "cylindrical2d":
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

        case "cartesian3d":
            griddimension = round(dfmodel_npts_model ** (1.0 / 3.0))
            print(f" 3D grid size: {dfmodel_npts_model} ({griddimension}^3)")
            assert griddimension**3 == dfmodel_npts_model
            strunits = "pos_x_min, pos_y_min, and pos_z_min [cm], rho [g/cm^3], all at t_model_init_days"
            headerlines = [
                (str(dfmodel_npts_model), f"npts_model: number of cells ({griddimension}^3 Cartesian grid)"),
                tmodelline,
                (f"{vmax:.8e}", "vmax_cmps: maximum velocity along each axis [cm/s]"),
            ]

        case _:
            strunits = "rho [g/cm^3] at t_model_init_days"

    if formatversion == 2:
        # a cell with no line is empty in version 2, thus the file needs no line for a cell with no density.
        # sn3d and scan_modeldata need one cell line, thus a model with no density keeps its first cell
        hasnodensity = not dfmodel.select(hasdensity.any()).item()
        dfmodel = dfmodel.filter(hasdensity | (~hasdensity.any() & (pl.int_range(pl.len()) == 0)))
        if hasnodensity:
            # sn3d reads no value of a cell with no density, and the checks above take no account of such a cell
            dfmodel = dfmodel.with_columns(
                pl.lit(0).cast(dtype).alias(col)
                for col, dtype in dfmodel.schema.items()
                if col not in {"inputcellid", "rho"} and dtype.is_numeric()
            )
        version2keylines.append(f"cell_line_count {dfmodel.height}")

    modelfilepath = with_compressed_extension(
        without_compressed_extension(resolve_outputfile(outpath, "model.txt")), ".zst"
    )

    def write_model_file(fmodel: t.IO[bytes]) -> None:
        # sn3d reads the first comment line after the header values as the column names, thus each
        # other comment line comes before those values. sn3d reads the numbers at the start of a header
        # line, thus an inline comment can follow them
        headertextlines = [
            # a reader finds the version on the first line. A version 1 file has no such line
            *(["format_version 2"] if formatversion == 2 else []),
            *[f"# {line}" for line in headercommentlines or []],
            get_created_comment(),
            f"# {UNITS_COMMENT_PREFIX} {strunits}. {UNITS_COMMENT_END}",
            *[f"{strvalue:<24} # {comment}" for strvalue, comment in headerlines],
            *version2keylines,
            # the key columns ends a version 2 header. A version 1 file gives the names in a comment line
            f"{'columns ' if formatversion == 2 else '#'}{' '.join([*standardcols, *customcols])}",
        ]

        abundandcustomcols = [*[col for col in standardcols if col.startswith("X_")], *customcols]

        isintcol = [not dfmodel.schema[col].is_float() for col in abundandcustomcols]
        ismassfraccol = [col.startswith("X_") for col in abundandcustomcols]
        strzeroabund = " ".join(["0" if isint else "0.0" for isint in isintcol])
        if coordinates == "spherical1d" and formatversion == 1:
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
            dfcells = dfmodel.select([*startcols, *abundandcustomcols])
            # fast polars writer
            # set abundances to null for cells with zero density (so that shorter form "0.0" can be written)
            dfcells = dfcells.with_columns(
                pl.when(pl.col("rho") > 0).then(pl.col(col)).otherwise(pl.lit(None)).alias(col)
                for col in dfcells.columns
                if not col.startswith("pos") and col != "inputcellid" and dfcells.schema[col].is_float()
            )
            if formatversion == 2:
                # a NaN mass fraction, e.g. from an interpolation, becomes zero, as in the 1D writer of version 1.
                # sn3d stops for a NaN
                dfcells = dfcells.with_columns(cs.starts_with("X_").fill_nan(0.0))
            write_zstd_lines(fmodel, headertextlines)
            write_artis_csv(dfcells, fmodel, significantfigures=9 if formatversion == 2 else 8)

    # The new file replaces the old file only after a complete write. Thus a write that stops early leaves the old
    # model. Each other copy of the old file goes after the write
    write_file_atomic(modelfilepath, write_model_file)
    prepare_zstd_output(modelfilepath)
    print_saved(modelfilepath)
    if dfelabundances is not None and formatversion == 1:
        save_initelemabundances(dfelabundances, outpath=modelfilepath.parent)
    elif formatversion == 2:
        for oldabundancefile in oldabundancefiles:
            oldabundancefile.unlink()
            remove_parquet_cache(oldabundancefile)
            print(f"Deleted {oldabundancefile}, because the model.txt of format version 2 gives the element masses")
    for oldcellfile in oldcellidfiles:
        oldcellfile.unlink()
        print(f"Deleted {oldcellfile}, because a model of format version 2 has no such file")
    print_detail(f"Took {time.perf_counter() - timestart:.1f} seconds")


def get_mgi_of_velocity_kms(modelpath: Path, velocity: float) -> int:
    """Return the modelgridindex of the first cell of a 1D model with an outer velocity [km/s] of velocity or above."""
    # a negative value was the default that gave no velocity, and searchsorted takes it as the first cell
    if velocity < 0.0:
        msg = f"--velocity {velocity:g} km/s is below zero"
        raise add_help_note(ValueError(msg), "Give a radial velocity of 0 or more")
    dfmodel, modelmeta = scan_modeldata(modelpath)
    if modelmeta["dimensions"] != 1:
        msg = f"--velocity selects a shell of a 1D model, and this model has {modelmeta['dimensions']} dimensions"
        raise add_help_note(ValueError(msg), "Give --modelgridindex")
    arr_vouter = dfmodel.select("vel_r_max_kmps").collect().to_series().to_numpy()

    mgi_upper = int(np.searchsorted(arr_vouter, velocity))
    if mgi_upper >= len(arr_vouter):
        msg = f"--velocity {velocity:g} km/s is above the outer velocity {arr_vouter[-1]:g} km/s of the last cell"
        raise add_help_note(ValueError(msg), "Give a lower velocity")
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
            glob=False,
        )
    except pl.exceptions.ComputeError:
        return None

    # an empty line gives a row of nulls, which read_wsv drops
    abundancedata = abundancedata.filter(~pl.all_horizontal(pl.all().is_null()))
    # a line with more spaces or with fewer values also gives a null, and read_wsv decides such a file
    return None if abundancedata.null_count().sum_horizontal().item() > 0 else abundancedata


def scan_initelemabundances(modelpath: Path | str = ".", printwarningsonly: bool = False) -> pl.LazyFrame:
    """Return a table of elemental mass fractions by cell.

    The table comes from abundances.txt. A model.txt of format version 2 has no abundances.txt, and the table is
    then the sum of the nuclide columns and the otherstable column of each element, as in sn3d.
    """
    modelfilepath = get_model_file_path(modelpath)
    inputfolder = get_model_input_folder(modelpath)
    if modelfilepath is not None:
        with zopen(modelfilepath) as fmodel:
            formatversion = get_model_format_version(fmodel.readline(), modelfilepath)
        if formatversion == 2:
            if (
                firstexisting_or_none("abundances.txt", folder=inputfolder, tryzipped=True, search_subfolders=False)
                is not None
            ):
                print_warning(
                    f"{inputfolder} holds abundances.txt, but its model.txt has format version 2, which gives the"
                    " element mass fractions. artistools does not read abundances.txt, and sn3d stops for it"
                )
            return get_elemabundances_of_model_columns(scan_modeldata(modelpath, printwarningsonly=True)[0])

    abundancefile = find_abundance_file(inputfolder)
    if abundancefile is None:
        msg = f"sn3d finds no abundances.txt for the model.txt in {inputfolder}"
        raise FileNotFoundError(msg)
    return scan_abundance_file(abundancefile, printwarningsonly)


def find_abundance_file(inputfolder: Path) -> Path | None:
    """Return the abundances.txt that sn3d reads for the model.txt in a folder, or None.

    sn3d searches the model folder, data/, and artis/data/, as for each input file. A different subfolder is of a
    different model. artistools also reads a copy beside model.txt that gzip or xz compressed, which sn3d does not.
    """
    sn3dfile = find_sn3d_input_file("abundances.txt", get_sn3d_model_folder(inputfolder))
    if sn3dfile is not None:
        return sn3dfile
    return firstexisting_or_none("abundances.txt", folder=inputfolder, tryzipped=True, search_subfolders=False)


def scan_abundance_file(textfilepath: Path, printwarningsonly: bool = False) -> pl.LazyFrame:
    """Return the table of the elemental mass fractions of each cell of an abundances.txt file."""

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
    """Save a DataFrame in the format of scan_initelemabundances to abundances.txt.zst.

    The file name gets the extension .zst, as in save_modeldata.

    columns must be:
        - inputcellid: integer index to match model.txt (starting from 0 or 1)
        - X_i: mass fraction of element with two-letter code 'i' (e.g., X_H, X_He, H_Li, ...).
    """
    timestart = time.perf_counter()

    abundancefilename = with_compressed_extension(
        without_compressed_extension(resolve_outputfile(outpath, "abundances.txt")), ".zst"
    )

    dfelabundances = (
        dfelabundances.lazy().with_columns([pl.col("inputcellid").cast(pl.Int32)]).sort("inputcellid").collect()
    )
    assert isinstance(dfelabundances, pl.DataFrame)

    # the cell ids are those of model.txt: from 1 or from 0 in format version 1, and from 0 in format version 2
    firstinputcellid = dfelabundances["inputcellid"].min()
    if (
        firstinputcellid not in {0, 1}
        or not (
            dfelabundances["inputcellid"] == firstinputcellid + pl.int_range(dfelabundances.height, eager=True)
        ).all()
    ):
        msg = (
            "The inputcellid values of abundances.txt must start at 0 or 1 and increase by one from each cell to the"
            " next"
        )
        raise ValueError(msg)

    atomic_numbers = {get_atomic_number(colname) for colname in dfelabundances.select(cs.starts_with("X_")).columns}
    max_atomic_number = max([30, *atomic_numbers])
    elcolnames = [f"X_{get_elsymbol(Z)}" for Z in range(1, 1 + max_atomic_number)]
    for colname in elcolnames:
        if colname not in dfelabundances.columns:
            dfelabundances = dfelabundances.with_columns(pl.lit(0.0).alias(colname))

    dfelabundances = dfelabundances.select(["inputcellid", *elcolnames])

    def write_abundances(fabund: t.IO[bytes]) -> None:
        # sn3d and scan_initelemabundances skip each comment line, and both read the columns by position
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

    # a process that stops after the replacement must not leave the cache of the old file beside the new file
    remove_parquet_cache(abundancefilename)
    write_file_atomic(abundancefilename, write_abundances)
    # a reader can write the cache of the old file again during the write, thus this call deletes the cache again
    prepare_zstd_output(abundancefilename)
    print_saved(abundancefilename)
    print_detail(f"Took {time.perf_counter() - timestart:.1f} seconds")


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
    *,
    modelmeta: dict[str, t.Any],
    **kwargs: t.Any,
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None, dict[str, t.Any]]:
    """Convert a model to fewer dimensions: to a 2D model, a 1D spherical model, or a model with one zone.

    A 3D Cartesian model gives a 2D cylindrical model. Each cell goes to the output cell that holds its mid-point.
    A model with spherical2d or spherical3d coordinates gives a spherical2d model or a 1D model with the same
    radial edges. Each output cell is then a sum of complete cells, thus the mass of each output cell is exact.

    The function can also change the particle gridcontributions and the table of the elemental abundances to agree
    with the new model. Each of these two results is None if the caller gives no such table.
    """
    assert outputdimensions in {0, 1, 2}

    dfmodel = dfmodel.lazy()

    hassphericaledges = get_model_coordinates(modelmeta) in SPHERICAL_EDGE_COORDINATES
    # the output has different coordinates, thus it gets the default format version of those coordinates
    edgekeys = (*ALL_MODEL_EDGE_KEYS, "firstinputcellid", "formatversion")
    modelmeta_out = {
        k: v for k, v in modelmeta.items() if not k.startswith(("ncoord", "wid_init")) and k not in edgekeys
    }

    assert all(
        key not in modelmeta_out or modelmeta_out[key] == kwargs[key] for key in kwargs
    )  # can't define the same thing twice unless the values are the same

    modelmeta_out |= kwargs  # add any extra keyword arguments to modelmeta

    t_model_init_seconds = modelmeta["t_model_init_days"] * day_to_s
    vmax = modelmeta["vmax_cmps"]
    xmax = vmax * t_model_init_seconds

    ndim_in = modelmeta["dimensions"]
    assert ndim_in > outputdimensions
    modelmeta_out["dimensions"] = max(outputdimensions, 1)
    if outputdimensions < 2:
        modelmeta_out["coordinates"] = "spherical1d"
        # the shells of a mirrored model hold the mass of both halves, and a 1D model has no mirror plane
        modelmeta_out.pop("mirror_plane", None)
    elif hassphericaledges:
        modelmeta_out |= {
            "coordinates": "spherical2d",
            "vel_r_edges_kmps": modelmeta["vel_r_edges_kmps"],
            "costheta_edges": modelmeta["costheta_edges"],
            "firstinputcellid": 1,
        }
    else:
        modelmeta_out["coordinates"] = "cylindrical2d"

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

    # the second axis of a 2D output is z for cylindrical coordinates and cos(theta) for spherical coordinates
    if hassphericaledges:
        in_n_r, in_n_costheta, _ = get_spherical_cell_indices(inputcols, modelmeta)
        ncoordgridr = len(modelmeta["vel_r_edges_kmps"]) - 1 if outputdimensions > 0 else 1
        ncoordgridz = len(modelmeta["costheta_edges"]) - 1 if outputdimensions == 2 else 1
        dfmodel_out = dfmodel_out.with_columns(
            out_n_r=(in_n_r if outputdimensions > 0 else pl.lit(0)).cast(pl.Int32),
            out_n_z=(in_n_costheta if outputdimensions == 2 else pl.lit(0)).cast(pl.Int32),
        )
        if outputdimensions == 1:
            modelmeta_out["ncoordgridr"] = ncoordgridr
    elif outputdimensions == 0:
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

    if not hassphericaledges:
        col_vel_r = pl.col("vel_rcyl_mid") if outputdimensions == 2 else pl.col("vel_r_mid")
        # the bins are closed on the left, because the centre cell of an odd grid has a mid-point velocity of zero.
        # A bin that is closed on the right puts that cell below the first bin, and the filter then drops its mass
        dfmodel_out = dfmodel_out.with_columns(get_bin_index_expr(col_vel_r, vel_r_bins).alias("out_n_r"))
        if outputdimensions == 2:
            dfmodel_out = dfmodel_out.with_columns(get_bin_index_expr(pl.col("vel_z_mid"), vel_z_bins).alias("out_n_z"))
    isinoutputgrid = pl.col("out_n_r").is_between(0, ncoordgridr - 1)
    if outputdimensions == 2:
        isinoutputgrid &= pl.col("out_n_z").is_between(0, ncoordgridz - 1)
    isinoutputgrid = isinoutputgrid.fill_null(value=False)

    # the radial bins end at vmax, and the corners of a Cartesian grid reach sqrt(3) * vmax
    mass_g_total, mass_g_outside = (
        dfmodel_out
        .select(total=pl.col("mass_g").sum(), outside=pl.col("mass_g").filter(~isinoutputgrid).sum())
        .collect()
        .row(0)
    )
    if mass_g_outside > 0:
        print_detail(
            f"The {outputdimensions}D grid ends at vmax = {vmax / km_to_cm:.0f} km/s. The cells of the input model"
            f" outside this grid hold {mass_g_outside / Msun_to_g:.3e} Msun, which is"
            f" {mass_g_outside / mass_g_total * 100:.3g}% of the mass. The output model leaves these cells out"
        )
    dfmodel_out = dfmodel_out.filter(isinoutputgrid)

    if outputdimensions == 2:
        dfmodel_out = dfmodel_out.with_columns(mgiout=pl.col("out_n_z") * ncoordgridr + pl.col("out_n_r"))
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
        # a spherical2d model has no position column, because its metadata holds the cell edges
        if not hassphericaledges:
            dfmodel_out = dfmodel_out.with_columns(
                pos_rcyl_mid=(pl.col("out_n_r") + 0.5) * (xmax / ncoordgridr),
                pos_z_mid=(pl.col("out_n_z") + 0.5) * (2 * xmax / ncoordgridz) - xmax,
            )
    elif hassphericaledges and outputdimensions == 1:
        shell_vel_r_max_kmps = pl.Series(modelmeta["vel_r_edges_kmps"][1:], dtype=pl.Float64)
        dfmodel_out = dfmodel_out.with_columns(vel_r_max_kmps=pl.lit(shell_vel_r_max_kmps).gather(pl.col("out_n_r")))
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

    dfoutcells_without_inputcells = dfmodel_out.filter(pl.col("inputcellid_list").list.len().fill_null(0) == 0).select(
        "inputcellid"
    )
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
            .join(dfoutcells_without_inputcells, on="inputcellid", how="full", coalesce=True)
            .with_columns((cs.starts_with("X_") | cs.by_name("mass_g")).fill_null(0.0))
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

    dfmodel_out, dfelabundances_collected, dfgridcontributions_collected = pl.collect_all((
        dfmodel_out,
        dfelabundances_out,
        dfgridcontributions_out,
    ))
    # an output is None for an input that is None
    dfelabundances_out = None if dfelabundances is None else dfelabundances_collected
    dfgridcontributions_out = None if dfgridcontributions is None else dfgridcontributions_collected

    if dfelabundances_out is not None:
        assert modelmeta_out["npts_model"] == dfelabundances_out.height

    if modelmeta_out["coordinates"] in SPHERICAL_EDGE_COORDINATES:
        # this model has format version 2 only, and that version has cell ids from 0
        dfmodel_out, dfelabundances_out, dfgridcontributions_out = with_cell_ids_from_zero(
            dfmodel_out, dfelabundances_out, dfgridcontributions_out
        )
        modelmeta_out["firstinputcellid"] = 0

    print(f"  Took {time.perf_counter() - timestart:.1f} seconds")

    return (dfmodel_out, dfelabundances_out, dfgridcontributions_out, modelmeta_out)


def scale_model_to_time(dfmodel: pl.DataFrame, targetmodeltime_days: float, t_model_days: float) -> pl.DataFrame:
    """Homologously expand model to targetmodeltime_days by reducing densities and adjusting cell positions.

    The velocities, the masses, and the kinetic energies do not change in a homologous expansion.
    """
    timefactor = targetmodeltime_days / t_model_days

    print(
        f"Adjusting t_model to {targetmodeltime_days} days (factor {timefactor}) "
        "using homologous expansion of positions and densities"
    )

    scale_exprs: list[pl.Expr] = [cs.starts_with("pos_") * timefactor]
    if "rho" in dfmodel.columns:
        scale_exprs.append(pl.col("rho") * timefactor**-3)
    if "logrho" in dfmodel.columns:
        # an empty cell keeps the marker -99, because a shrink by a factor below 0.46 moves it above -98
        scale_exprs.append(
            pl
            .when(pl.col("logrho") > -98)
            .then(pl.col("logrho") + math.log10(timefactor**-3))
            .otherwise(pl.col("logrho"))
        )
    if "volume" in dfmodel.columns:
        scale_exprs.append(pl.col("volume") * timefactor**3)
    return dfmodel.with_columns(scale_exprs)


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


def resolve_cell_direction_range(
    args: argparse.Namespace,
    modelpath: Path | str,
    quiet: bool = False,
    *,
    dimensions: int | None = None,
    mirrored: bool | None = None,
) -> CellDirectionRange | None:
    """Return the direction range of the options, as get_cell_direction_range does, for the cells of the model.

    A 2D model is symmetric around the z axis, thus the range of --dirbin then has no ϕ bin. The titles and the method
    lines then give the range of the selected cells. --average-over-theta-angle keeps only the ϕ bin, thus a 2D model
    refuses it. With quiet, the function does not log the range. Code on the host of a remote model gives dimensions
    and mirrored, because args.modelpath there still names the host.

    Each cell of a model with mirror symmetry (the key mirror_plane) fills both halves z > 0 and z < 0. A range of
    directions cannot select the half of a cell, thus the function stops for such a model.
    """
    directionrange = get_cell_direction_range(args, quiet=quiet)
    if directionrange is None:
        return None
    if dimensions is None or mirrored is None:
        modelmeta = scan_modeldata(modelpath, printwarningsonly=True)[1]
        dimensions = modelmeta["dimensions"]
        mirrored = has_mirror_plane(modelmeta)
    if mirrored:
        exit_with_error(
            f"{modelpath} has mirror symmetry in the plane z = 0, and each cell fills both halves. Thus a range of"
            " directions cannot select the part of a cell in that range",
            "Remove --dirbin, --costhetamin, --costhetamax, and the options --average-over-...-angle",
        )
    if directionrange.phibin is None:
        return directionrange
    if dimensions != 2:
        return directionrange
    if directionrange.costhetabin is None:
        exit_with_error(
            "A 2D model is symmetric around the z axis, thus --average-over-theta-angle would select each cell",
            "Remove --average-over-theta-angle, or give --average-over-phi-angle for a range of cos θ",
        )
    if not quiet:
        print_detail("A 2D model is symmetric around the z axis, thus --dirbin selects the cells at each ϕ")
    return directionrange._replace(phibin=None)


def get_cell_direction_selection(directionrange: CellDirectionRange, columns: Sequence[str]) -> pl.Expr:
    """Return a selection that is true for a cell whose centre has a direction in the range.

    columns are the columns of the cells, which must hold vel_r_mid_on_c and vel_z_mid_on_c. A ϕ bin also needs
    vel_x_mid_on_c and vel_y_mid_on_c, thus a 2D model needs the range of resolve_cell_direction_range. --dirbin
    selects a cell with the bins of ARTIS for a packet in the direction of the cell centre, thus each cell is in one
    bin. A 1D model has no direction of a cell, thus the function raises ValueError for it. See
    get_cell_selection for the edges of a range of --costhetamin and --costhetamax.
    """
    if "vel_z_mid_on_c" not in columns:
        msg = (
            "A 1D model has no direction of a cell, thus --dirbin, --costhetamin, and --costhetamax cannot select cells"
        )
        raise add_help_note(ValueError(msg), "Remove --dirbin, --costhetamin, and --costhetamax")
    if directionrange.costhetabin is None and directionrange.phibin is None:
        return get_cell_selection(
            thetamin=math.degrees(math.acos(directionrange.costhetamax)),
            thetamax=math.degrees(math.acos(directionrange.costhetamin)),
        )

    is3d = "vel_x_mid_on_c" in columns and "vel_y_mid_on_c" in columns
    vel_x = pl.col("vel_x_mid_on_c")
    vel_y = pl.col("vel_y_mid_on_c")
    vel_z = pl.col("vel_z_mid_on_c")
    # add_packet_directions_lazypolars takes |v| of the three parts. vel_r_mid_on_c can differ in the last bit, and
    # then a cell on the edge of a bin goes to a different bin
    vel_r = (vel_x**2 + vel_y**2 + vel_z**2).sqrt() if is3d else pl.col("vel_r_mid_on_c")
    costheta, phi = get_costheta_phi_exprs(vel_x, vel_y, vel_z)
    # the cell at the origin has no direction
    conditions = [vel_r > 0.0]
    if directionrange.costhetabin is not None:
        # the packet code casts cos θ and ϕ to Float32, thus this code does too
        costheta = costheta if is3d else vel_z / vel_r
        conditions.append(get_costhetabin_expr(costheta.cast(pl.Float32)) == directionrange.costhetabin)
    if directionrange.phibin is not None:
        if not is3d:
            msg = "A ϕ bin needs the columns vel_x_mid_on_c and vel_y_mid_on_c. For a 2D model, give phibin=None"
            raise ValueError(msg)
        conditions.append(get_phibin_expr(phi.cast(pl.Float32), vel_y) == directionrange.phibin)
    return pl.all_horizontal(conditions).fill_null(value=False)


# the names before the readers took the prefix scan_ or read_. Each alias keeps the old signature, thus a script
# of a user still runs, and it gives a DeprecationWarning that names the new function
@warnings.deprecated("get_modeldata is deprecated. Use scan_modeldata.")
def get_modeldata(
    modelpath: Path | str = ".", get_elemabundances: bool = False, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Return the result of scan_modeldata, the new name of this function."""
    return scan_modeldata(modelpath, get_elemabundances=get_elemabundances, printwarningsonly=printwarningsonly)


@warnings.deprecated("read_modeldata is deprecated. Use scan_local_modeldata.")
def read_modeldata(
    modelpath: Path, get_elemabundances: bool = False, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Return the result of scan_local_modeldata, the new name of this function."""
    return scan_local_modeldata(modelpath, get_elemabundances=get_elemabundances, printwarningsonly=printwarningsonly)


@warnings.deprecated("read_modelfile_text is deprecated. Use scan_modelfile_text.")
def read_modelfile_text(
    filename: Path | str, printwarningsonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Return the result of scan_modelfile_text, the new name of this function."""
    return scan_modelfile_text(filename, printwarningsonly=printwarningsonly)


@warnings.deprecated("get_initelemabundances is deprecated. Use scan_initelemabundances.")
def get_initelemabundances(modelpath: Path | str = ".", printwarningsonly: bool = False) -> pl.LazyFrame:
    """Return the result of scan_initelemabundances, the new name of this function."""
    return scan_initelemabundances(modelpath, printwarningsonly=printwarningsonly)
