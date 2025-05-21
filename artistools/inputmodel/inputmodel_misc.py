import calendar
import errno
import gc
import itertools
import math
import os
import tempfile
import time
import typing as t
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from warnings import deprecated

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

from artistools.configuration import get_config
from artistools.misc import firstexisting
from artistools.misc import get_atomic_number
from artistools.misc import get_elsymbol
from artistools.misc import get_z_a_nucname
from artistools.misc import stripallsuffixes
from artistools.misc import vec_len
from artistools.misc import zopen
from artistools.misc import zopenpl


# @lru_cache(maxsize=3)
def read_modelfile_text(
    filename: Path | str, printwarningsonly: bool = False, getheadersonly: bool = False
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides."""
    onelinepercellformat = None

    modelmeta: dict[str, t.Any] = {"headercommentlines": []}
    xmax_tmodel: float = 0.0
    ncoordgridx: int = 0
    ncoordgridy: int = 0
    ncoordgridz: int = 0

    if not printwarningsonly and not getheadersonly:
        print(f"Reading {filename}")

    numheaderrows = 0
    with zopen(filename) as fmodel:
        line = "#"
        while line.startswith("#"):
            line = fmodel.readline()
            if line.startswith("#"):
                modelmeta["headercommentlines"].append(line.removeprefix("#").removeprefix(" ").removesuffix("\n"))
                numheaderrows += 1

        if len(line.strip().split(" ")) == 2:
            modelmeta["dimensions"] = 2
            ncoordgridr, ncoordgridz = (int(n) for n in line.strip().split(" "))
            modelmeta["ncoordgridrcyl"] = ncoordgridr
            modelmeta["ncoordgridz"] = ncoordgridz
            npts_model = ncoordgridr * ncoordgridz
            if not printwarningsonly:
                print(f"  detected 2D model file with n_r*n_z={ncoordgridr}x{ncoordgridz}={npts_model} cells")
        else:
            npts_model = int(line)

        modelmeta["npts_model"] = npts_model
        modelmeta["t_model_init_days"] = float(fmodel.readline())
        numheaderrows += 2
        t_model_init_seconds = modelmeta["t_model_init_days"] * 24 * 60 * 60

        line = fmodel.readline()
        # if the next line is a single float then the model is 2D or 3D (vmax)
        try:
            modelmeta["vmax_cmps"] = float(line)  # velocity max in cm/s
            xmax_tmodel = modelmeta["vmax_cmps"] * t_model_init_seconds  # xmax = ymax = zmax
            numheaderrows += 1
            if "dimensions" not in modelmeta:  # not already detected as 2D
                modelmeta["dimensions"] = 3
                # number of grid cell steps along an axis (currently the same for xyz)
                ncoordgridx = round(npts_model ** (1.0 / 3.0))
                ncoordgridy = round(npts_model ** (1.0 / 3.0))
                ncoordgridz = round(npts_model ** (1.0 / 3.0))
                assert (ncoordgridx * ncoordgridy * ncoordgridz) == npts_model
                modelmeta["ncoordgridx"] = ncoordgridx
                modelmeta["ncoordgridy"] = ncoordgridy
                modelmeta["ncoordgridz"] = ncoordgridz
                if ncoordgridx == ncoordgridy == ncoordgridz:
                    modelmeta["ncoordgrid"] = ncoordgridx

                if not printwarningsonly:
                    print(f"  detected 3D model file with {ncoordgridx}x{ncoordgridy}x{ncoordgridz}={npts_model} cells")

            line = fmodel.readline()

        except ValueError:
            assert modelmeta.get("dimensions", -1) != 2, "2D model should have a vmax line here"
            if "dimensions" not in modelmeta:
                if not printwarningsonly:
                    print(f"  detected 1D model file with {npts_model} radial zones")
                modelmeta["dimensions"] = 1
                getheadersonly = False  # need to find vmax from the model data

        columns = None
        if line.startswith("#"):
            numheaderrows += 1
            columns = line.lstrip("#").split()
            line = fmodel.readline()

        data_line_even = line
        ncols_line_even = len(data_line_even.split())
        data_line_odd = fmodel.readline()
        ncols_line_odd = len(data_line_odd.split())

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
            if not printwarningsonly and not getheadersonly:
                print("  model file is one line per cell")
            ncols_line_odd = 0
            onelinepercellformat = True
        else:
            if not printwarningsonly:
                print("  model file format is two lines per cell")
            # columns split over two lines
            assert (ncols_line_even + ncols_line_odd) == len(columns)
            onelinepercellformat = False

    if onelinepercellformat and "  " not in data_line_even and "  " not in data_line_odd and not getheadersonly:
        if not printwarningsonly:
            print("  using fast method polars.read_csv (requires one line per cell and single space delimiters)")

        dfmodel = pl.read_csv(
            zopenpl(filename),
            separator=" ",
            comment_prefix="#",
            new_columns=columns,
            n_rows=npts_model,
            has_header=False,
            skip_rows=numheaderrows,
            schema={col: pl.Int32 if col == "inputcellid" else pl.Float32 for col in columns},
            truncate_ragged_lines=True,
        ).lazy()
    else:
        nrows_read = 1 if getheadersonly else npts_model
        skiprows: list[int] | int = (
            numheaderrows
            if onelinepercellformat
            else [
                x
                for x in range(numheaderrows + npts_model * 2)
                if x < numheaderrows or (x - numheaderrows - 1) % 2 == 0
            ]
        )

        dtypes: defaultdict[str, str] = defaultdict(lambda: "float32[pyarrow]")
        dtypes["inputcellid"] = "int32[pyarrow]"

        # each cell takes up two lines in the model file
        dfmodelpd = pd.read_csv(
            filename,
            sep=r"\s+",
            engine="c",
            header=None,
            skiprows=skiprows,
            names=columns[:ncols_line_even],
            usecols=columns[:ncols_line_even],
            nrows=nrows_read,
            dtype=dtypes,
            dtype_backend="pyarrow",
            memory_map=True,
        )

        if ncols_line_odd > 0 and not onelinepercellformat:
            # read in the odd rows and merge dataframes
            skipevenrows = [
                x
                for x in range(numheaderrows + npts_model * 2)
                if x < numheaderrows or (x - numheaderrows - 1) % 2 == 1
            ]
            dfmodeloddlines = pd.read_csv(
                filename,
                sep=r"\s+",
                engine="c",
                header=None,
                skiprows=skipevenrows,
                names=columns[ncols_line_even:],
                nrows=nrows_read,
                dtype=dtypes,
                dtype_backend="pyarrow",
            )
            assert len(dfmodelpd) == len(dfmodeloddlines)
            dfmodelpd = dfmodelpd.merge(dfmodeloddlines, left_index=True, right_index=True)
            del dfmodeloddlines

        if len(dfmodelpd) > npts_model:
            dfmodelpd = dfmodelpd.iloc[:npts_model]

        assert len(dfmodelpd) == npts_model or getheadersonly

        dfmodelpd.index.name = "cellid"

        dfmodel = pl.from_pandas(dfmodelpd).lazy()

    if "velocity_outer" in dfmodel.collect_schema().names():
        dfmodel = dfmodel.rename({"velocity_outer": "vel_r_max_kmps"}, strict=False)

    if "cellYe" in dfmodel.collect_schema().names():
        dfmodel = dfmodel.rename({"cellYe": "Ye"}, strict=False)

    if modelmeta["dimensions"] == 1:
        vmax_kmps = dfmodel.select(pl.col("vel_r_max_kmps").max()).collect().item()
        assert isinstance(vmax_kmps, float)
        modelmeta["vmax_cmps"] = vmax_kmps * 1.0e5

    elif modelmeta["dimensions"] == 2:
        wid_init_rcyl = modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridrcyl"]
        wid_init_z = 2 * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridz"]
        modelmeta["wid_init_rcyl"] = wid_init_rcyl
        modelmeta["wid_init_z"] = wid_init_z
        if not getheadersonly:
            # check pos_rcyl_mid and pos_z_mid are correct

            for inputcellid, cell_pos_rcyl_mid, cell_pos_z_mid in (
                dfmodel.select(["inputcellid", "pos_rcyl_mid", "pos_z_mid"]).collect().iter_rows()
            ):
                modelgridindex = inputcellid - 1
                n_r = modelgridindex % modelmeta["ncoordgridrcyl"]
                n_z = modelgridindex // modelmeta["ncoordgridrcyl"]
                pos_rcyl_min = modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridrcyl"] * n_r
                pos_z_min = (
                    -modelmeta["vmax_cmps"] * t_model_init_seconds
                    + 2.0 * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridz"] * n_z
                )
                pos_rcyl_mid = pos_rcyl_min + 0.5 * wid_init_rcyl
                pos_z_mid = pos_z_min + 0.5 * wid_init_z
                assert np.isclose(cell_pos_rcyl_mid, pos_rcyl_mid, atol=wid_init_rcyl / 2.0)
                assert np.isclose(cell_pos_z_mid, pos_z_mid, atol=wid_init_z / 2.0)

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
            if not getheadersonly:
                firstrow = dfmodel.select(cs.starts_with("pos_")).first().collect().row(index=0, named=True)
                expected_positions = (
                    ("pos_x_min", -xmax_tmodel),
                    ("pos_y_min", -xmax_tmodel),
                    ("pos_z_min", -xmax_tmodel),
                    ("pos_x_mid", -xmax_tmodel + wid_init_x / 2.0),
                    ("pos_y_mid", -xmax_tmodel + wid_init_y / 2.0),
                    ("pos_z_mid", -xmax_tmodel + wid_init_z / 2.0),
                )
                for col, pos in expected_positions:
                    if col in firstrow and not math.isclose(firstrow["pos_x_min"], pos, rel_tol=0.01):
                        print(
                            f"  WARNING: {col} does not match expected value. Check that vmax is consistent with the cell positions."
                        )

        elif not getheadersonly:

            def vectormatch(vec1: list[float], vec2: list[float]) -> bool:
                xclose = np.isclose(vec1[0], vec2[0], atol=wid_init_x * 0.05)
                yclose = np.isclose(vec1[1], vec2[1], atol=wid_init_y * 0.05)
                zclose = np.isclose(vec1[2], vec2[2], atol=wid_init_z * 0.05)

                return all([xclose, yclose, zclose])

            matched_pos_xyz_min = True
            matched_pos_zyx_min = True
            matched_pos_xyz_mid = True
            matched_pos_zyx_mid = True
            # important cell numbers to check for coordinate column order
            indexlist = [
                0,
                ncoordgridx - 1,
                ncoordgridx,
                (ncoordgridx - 1) * (ncoordgridy - 1),
                (ncoordgridx - 1) * ncoordgridy,
                (ncoordgridx - 1) * (ncoordgridy - 1) * (ncoordgridz - 1),
            ]

            for modelgridindex in indexlist:
                xindex = modelgridindex % ncoordgridx
                yindex = (modelgridindex // ncoordgridx) % ncoordgridy
                zindex = (modelgridindex // (ncoordgridx * ncoordgridy)) % ncoordgridz
                pos_x_min = -xmax_tmodel + xindex * wid_init_x
                pos_y_min = -xmax_tmodel + yindex * wid_init_y
                pos_z_min = -xmax_tmodel + zindex * wid_init_z
                pos_x_mid = -xmax_tmodel + (xindex + 0.5) * wid_init_x
                pos_y_mid = -xmax_tmodel + (yindex + 0.5) * wid_init_y
                pos_z_mid = -xmax_tmodel + (zindex + 0.5) * wid_init_z

                pos3_in = list(dfmodel.select(["inputpos_a", "inputpos_b", "inputpos_c"]).collect().row(modelgridindex))

                if not vectormatch(pos3_in, [pos_x_min, pos_y_min, pos_z_min]):
                    matched_pos_xyz_min = False

                if not vectormatch(pos3_in, [pos_z_min, pos_y_min, pos_x_min]):
                    matched_pos_zyx_min = False

                if not vectormatch(pos3_in, [pos_x_mid, pos_y_mid, pos_z_mid]):
                    matched_pos_xyz_mid = False

                if not vectormatch(pos3_in, [pos_z_mid, pos_y_mid, pos_x_mid]):
                    matched_pos_zyx_mid = False

            assert sum((matched_pos_xyz_min, matched_pos_zyx_min, matched_pos_xyz_mid, matched_pos_zyx_mid)) == 1, (
                "one option must match uniquely"
            )

            colrenames = {}
            if matched_pos_xyz_min:
                print("  model cell positions are consistent with x-y-z min corner columns")
                colrenames = {"inputpos_a": "pos_x_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_z_min"}

            if matched_pos_zyx_min:
                print("  cell positions are consistent with z-y-x min corner columns")
                colrenames = {"inputpos_a": "pos_z_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_x_min"}

            if matched_pos_xyz_mid:
                print("  model cell positions are consistent with x-y-z midpoint columns")
                colrenames = {"inputpos_a": "pos_x_mid", "inputpos_b": "pos_y_mid", "inputpos_c": "pos_z_mid"}

            if matched_pos_zyx_mid:
                print("  cell positions are consistent with z-y-x midpoint columns")
                colrenames = {"inputpos_a": "pos_z_mid", "inputpos_b": "pos_y_mid", "inputpos_c": "pos_x_mid"}

            dfmodel = dfmodel.rename(colrenames, strict=False)

            if matched_pos_xyz_mid or matched_pos_zyx_mid:
                dfmodel = dfmodel.with_columns(
                    pos_x_min=(pl.col("pos_x_mid") - modelmeta["wid_init_x"] / 2.0),
                    pos_y_min=(pl.col("pos_y_mid") - modelmeta["wid_init_y"] / 2.0),
                    pos_z_min=(pl.col("pos_z_mid") - modelmeta["wid_init_z"] / 2.0),
                )

    return dfmodel, modelmeta


def get_modeldata(
    modelpath: Path | str = Path(),
    get_elemabundances: bool = False,
    derived_cols: Sequence[str] | str | None = None,
    printwarningsonly: bool = False,
    getheadersonly: bool = False,
) -> tuple[pl.LazyFrame, dict[t.Any, t.Any]]:
    """Read an artis model.txt file containing cell velocities, densities, and mass fraction abundances of radioactive nuclides.

    Parameters
    ----------
        - modelpath: either a path to model.txt file, or a folder containing model.txt
        - get_elemabundances: also read elemental abundances (abundances.txt) and
            merge with the output DataFrame
        - derived_cols: list of derived columns to add to the model data, or "ALL" to add all possible derived columns

    return dfmodel, modelmeta
        - dfmodel: a pandas DataFrame with a row for each model grid cell
        - modelmeta: a dictionary of input model parameters, with keys such as t_model_init_days, vmax_cmps, dimensions, etc.

    """
    if isinstance(derived_cols, str):
        derived_cols = [derived_cols]

    inputpath = Path(modelpath)

    if inputpath.is_dir():
        modelpath = inputpath
        textfilepath = firstexisting("model.txt", folder=inputpath, tryzipped=True)
    elif inputpath.is_file():  # passed in a filename instead of the modelpath
        textfilepath = inputpath
        modelpath = Path(inputpath).parent
    elif not inputpath.exists() and inputpath.parts[0] == "codecomparison":
        modelpath = inputpath
        _, inputmodel, _ = modelpath.parts
        textfilepath = Path(get_config()["codecomparisonmodelartismodelpath"], inputmodel, "model.txt")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputpath)

    parquetfilepath = stripallsuffixes(Path(textfilepath)).with_suffix(".txt.parquet.tmp")

    if parquetfilepath.exists() and Path(textfilepath).stat().st_mtime > parquetfilepath.stat().st_mtime:
        print(f"{textfilepath} has been modified after {parquetfilepath}. Deleting out of date parquet file.")
        parquetfilepath.unlink()

    t_lastschemachange = calendar.timegm((2024, 1, 1, 9, 0, 0))

    if parquetfilepath.exists() and Path(parquetfilepath).stat().st_mtime < t_lastschemachange:
        print(f"{parquetfilepath} has been modified before the last schema change. Deleting out of date parquet file.")
        parquetfilepath.unlink()

    dfmodel: pl.LazyFrame | None = None
    if not getheadersonly and parquetfilepath.is_file():
        if not printwarningsonly:
            print(f"Reading model table from {parquetfilepath}")
        try:
            dfmodel = pl.scan_parquet(parquetfilepath)
        except pl.exceptions.ComputeError:
            print(f"Problem reading {parquetfilepath}. Will regenerate and overwrite from text source.")
            dfmodel = None

    if dfmodel is not None:
        # already read in the model data, just need to get the metadata
        getheadersonly = True

    dfmodel_textfile, modelmeta = read_modelfile_text(
        filename=textfilepath, printwarningsonly=printwarningsonly, getheadersonly=getheadersonly
    )

    if dfmodel is None:
        dfmodel = dfmodel_textfile
        assert dfmodel is not None

        mebibyte = 1024 * 1024
        if textfilepath.stat().st_size > 5 * mebibyte and not getheadersonly:
            print(f"Saving {parquetfilepath}")
            partialparquetfilepath = Path(
                tempfile.mkstemp(dir=modelpath, prefix=f"{parquetfilepath.name}.partial", suffix=".tmp")[1]
            )
            dfmodel.collect().write_parquet(partialparquetfilepath, compression="zstd", statistics=True)
            if parquetfilepath.exists():
                partialparquetfilepath.unlink()
            else:
                partialparquetfilepath.rename(parquetfilepath)
            print("  Done.")
            del dfmodel
            gc.collect()
            dfmodel = pl.scan_parquet(parquetfilepath)

    if not printwarningsonly:
        print(f"  model is {modelmeta['dimensions']}D with {modelmeta['npts_model']} cells")

    if get_elemabundances:
        abundancedata = get_initelemabundances_polars(modelpath, printwarningsonly=printwarningsonly)
        dfmodel = dfmodel.join(abundancedata, how="inner", on="inputcellid")

    dfmodel = dfmodel.with_columns(pl.col("inputcellid").sub(1).alias("modelgridindex"))

    if "cellYe" in dfmodel.collect_schema().names() and "Ye" not in dfmodel.collect_schema().names():
        dfmodel = dfmodel.rename({"cellYe": "Ye"}, strict=False)

    if derived_cols:
        dfmodel = add_derived_cols_to_modeldata(
            dfmodel=dfmodel, derived_cols=derived_cols, modelmeta=modelmeta, modelpath=modelpath
        )

    return dfmodel, modelmeta


@deprecated("Use get_modeldata() instead.")
def get_modeldata_tuple(*args: t.Any, **kwargs: t.Any) -> tuple[pd.DataFrame, float, float]:
    """Return tuple of (dfmodel, t_model_init_days, vmax_cmps) from model.txt file."""
    dfmodel, modelmeta = get_modeldata_pandas(*args, **kwargs)

    return dfmodel, modelmeta["t_model_init_days"], modelmeta["vmax_cmps"]


def get_empty_3d_model(
    ncoordgrid: int, vmax: float, t_model_init_days: float, includenico57: bool = False
) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
    xmax = vmax * t_model_init_days * 86400.0

    modelmeta = {
        "dimensions": 3,
        "t_model_init_days": t_model_init_days,
        "vmax_cmps": vmax,
        "npts_model": ncoordgrid**3,
        "wid_init": 2 * xmax / ncoordgrid,
        "wid_init_x": 2 * xmax / ncoordgrid,
        "wid_init_y": 2 * xmax / ncoordgrid,
        "wid_init_z": 2 * xmax / ncoordgrid,
        "ncoordgrid": ncoordgrid,
        "ncoordgridx": ncoordgrid,
        "ncoordgridy": ncoordgrid,
        "ncoordgridz": ncoordgrid,
        "headercommentlines": [],
    }

    fncoordgrid = float(ncoordgrid)  # fixes an issue with polars 0.20.23 https://github.com/pola-rs/polars/issues/15952

    dfmodel = (
        pl.DataFrame(
            {"modelgridindex": range(ncoordgrid**3), "inputcellid": range(1, 1 + ncoordgrid**3)},
            schema={"modelgridindex": pl.Int32, "inputcellid": pl.Int32},
        )
        .lazy()
        .with_columns([
            pl.col("modelgridindex").mod(ncoordgrid).alias("n_x"),
            (pl.col("modelgridindex") // ncoordgrid).mod(ncoordgrid).alias("n_y"),
            (pl.col("modelgridindex") // (ncoordgrid**2)).mod(ncoordgrid).alias("n_z"),
        ])
        .with_columns([
            (-xmax + 2.0 * pl.col("n_x") * xmax / fncoordgrid).cast(pl.Float32).alias("pos_x_min"),
            (-xmax + 2.0 * pl.col("n_y") * xmax / fncoordgrid).cast(pl.Float32).alias("pos_y_min"),
            (-xmax + 2.0 * pl.col("n_z") * xmax / fncoordgrid).cast(pl.Float32).alias("pos_z_min"),
        ])
    )

    standardcols = get_standard_columns(3, includenico57=includenico57)

    dfmodel = dfmodel.with_columns([
        pl.lit(0.0, dtype=pl.Float32).alias(colname)
        for colname in standardcols
        if colname not in dfmodel.collect_schema().names()
    ]).select([*standardcols, "modelgridindex"])

    return dfmodel, modelmeta


@deprecated("Use get_modeldata() instead.")
def get_modeldata_pandas(
    modelpath: Path | str = Path(),
    get_elemabundances: bool = False,
    derived_cols: Sequence[str] | str | None = None,
    printwarningsonly: bool = False,
    getheadersonly: bool = False,
) -> tuple[pd.DataFrame, dict[t.Any, t.Any]]:
    """Like get_modeldata() but convert polars DataFrame to pandas."""
    pldfmodel, modelmeta = get_modeldata(
        modelpath=modelpath,
        get_elemabundances=get_elemabundances,
        derived_cols=derived_cols,
        printwarningsonly=printwarningsonly,
        getheadersonly=getheadersonly,
    )
    dfmodel = pldfmodel.collect().to_pandas(use_pyarrow_extension_array=True)
    if modelmeta["npts_model"] > 100000 and not getheadersonly:
        # dfmodel.info(verbose=False, memory_usage="deep")
        print("WARNING: Using pandas DataFrame for large model data. Switch to using get_modeldata() instead.")

    return dfmodel, modelmeta


def add_derived_cols_to_modeldata(
    dfmodel: pl.DataFrame | pl.LazyFrame,
    derived_cols: Sequence[str],
    modelmeta: dict[str, t.Any],
    modelpath: Path | None = None,
) -> pl.LazyFrame:
    """Add columns to modeldata using e.g. derived_cols = ("velocity", "Ye")."""
    # with lazy mode, we can add every column and then drop the ones we don't need
    dfmodel = dfmodel.lazy()
    original_cols = dfmodel.collect_schema().names()
    derived_cols = list(derived_cols)

    t_model_init_seconds = modelmeta["t_model_init_days"] * 86400.0
    keep_all = "ALL" in derived_cols

    if "logrho" not in dfmodel.collect_schema().names():
        dfmodel = dfmodel.with_columns(logrho=pl.col("rho").log10())

    if "rho" not in dfmodel.collect_schema().names():
        dfmodel = dfmodel.with_columns(
            rho=(pl.when(pl.col("logrho") > -98).then(10 ** pl.col("logrho")).otherwise(0.0))
        )
    axes: list[str] = []
    dimensions = modelmeta["dimensions"]
    match dimensions:
        case 1:
            axes = ["r"]

            dfmodel = (
                dfmodel.with_columns(vel_r_min_kmps=pl.col("vel_r_max_kmps").shift(n=1, fill_value=0.0))
                .with_columns(vel_r_min=(pl.col("vel_r_min_kmps") * 1e5), vel_r_max=(pl.col("vel_r_max_kmps") * 1e5))
                .with_columns(vel_r_mid=((pl.col("vel_r_max") + pl.col("vel_r_min")) / 2))
                .with_columns(
                    volume=(
                        (4.0 / 3.0)
                        * math.pi
                        * (
                            pl.col("vel_r_max_kmps").cast(pl.Float64).pow(3)
                            - pl.col("vel_r_min_kmps").cast(pl.Float64).pow(3)
                        )
                        * (1e5 * t_model_init_seconds) ** 3
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
                pos_r_min=(
                    pl.col("pos_rcyl_min").pow(2)
                    + pl.min_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
                ).sqrt(),
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
                    / 8
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
                dfmodel.with_columns(
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
                    pl.min_horizontal(pl.col("pos_x_min").abs(), pl.col("pos_x_max").abs()).pow(2)
                    + pl.min_horizontal(pl.col("pos_y_min").abs(), pl.col("pos_y_max").abs()).pow(2)
                    + pl.min_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
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

    assert axes
    # get total kinetic energy from orthogonal components
    # all coord system also have a radial component calculated, so ignore this
    dfmodel = dfmodel.with_columns(
        kinetic_en_erg=(
            pl.sum_horizontal(pl.col(f"kinetic_en_erg_{ax}") for ax in axes if (ax != "r" or dimensions == 1))
        )
    )

    for col in dfmodel.collect_schema().names():
        if col.startswith("pos_"):
            dfmodel = dfmodel.with_columns((pl.col(col) / t_model_init_seconds).alias(col.replace("pos_", "vel_")))

    # add vel_*_on_c scaled velocities
    dfmodel = dfmodel.with_columns(mass_g=(pl.col("rho") * pl.col("volume"))).with_columns(
        (cs.starts_with("vel_") / 29979245800.0).name.suffix("_on_c")
    )

    if unknown_cols := [
        col
        for col in derived_cols
        if col not in dfmodel.collect_schema().names() and col not in {"pos_min", "pos_max", "ALL", "velocity"}
    ]:
        print(f"WARNING: Unknown derived columns: {unknown_cols}")

    if "pos_min" in derived_cols:
        derived_cols.extend(
            col for col in dfmodel.collect_schema().names() if col.startswith("pos_") and col.endswith("_min")
        )

    if "pos_max" in derived_cols:
        derived_cols.extend(
            col for col in dfmodel.collect_schema().names() if col.startswith("pos_") and col.endswith("_max")
        )

    if "velocity" in derived_cols:
        derived_cols.extend(col for col in dfmodel.collect_schema().names() if col.startswith("vel_"))

    if not keep_all:
        dfmodel = dfmodel.drop([
            col for col in dfmodel.collect_schema().names() if col not in original_cols and col not in derived_cols
        ])

    if "angle_bin" in derived_cols:
        assert modelpath is not None
        dfmodel = pl.from_pandas(get_cell_angle(dfmodel.collect().to_pandas(use_pyarrow_extension_array=True))).lazy()

    # if "Ye" in derived_cols and os.path.isfile(modelpath / "Ye.txt"):
    #     dfmodel["Ye"] = at.inputmodel.opacityinputfile.get_Ye_from_file(modelpath)
    # if "Q" in derived_cols and os.path.isfile(modelpath / "Q_energy.txt"):
    #     dfmodel["Q"] = at.inputmodel.energyinputfiles.get_Q_energy_from_file(modelpath)

    return dfmodel


def get_cell_angle(dfmodel: pd.DataFrame) -> pd.DataFrame:
    """Get angle between origin to cell midpoint and the syn_dir axis."""
    syn_dir = np.array([0.0, 0.0, 1.0])
    xhat = np.array([1.0, 0.0, 0.0])

    cos_theta = np.zeros(len(dfmodel))
    phi = np.zeros(len(dfmodel))
    for i, (_, cell) in enumerate(dfmodel.iterrows()):
        mid_point = [cell["pos_x_mid"], cell["pos_y_mid"], cell["pos_z_mid"]]
        cos_theta[i] = (np.dot(mid_point, syn_dir)) / (vec_len(mid_point) * vec_len(syn_dir))

        vec1 = np.cross(mid_point, syn_dir)
        vec2 = np.cross(xhat, syn_dir)
        cosphi = np.dot(vec1, vec2) / vec_len(vec1) / vec_len(vec2)

        vec3 = np.cross(vec2, syn_dir)
        testphi = np.dot(vec1, vec3)
        phi[i] = math.acos(cosphi) if testphi > 0 else (math.acos(-cosphi) + np.pi)

    dfmodel["cos_theta"] = cos_theta
    dfmodel["phi"] = phi
    cos_bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]  # including end bin
    labels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    # assert at.get_viewingdirection_costhetabincount() == 10
    # assert at.get_viewingdirection_phibincount() == 10
    dfmodel["cos_bin"] = pd.cut(dfmodel["cos_theta"], cos_bins, labels=labels)
    # dfmodel['cos_bin'] = np.searchsorted(cos_bins, dfmodel['cos_theta'].values) -1

    # phibins = ["0", "π/5", "2π/5", "3π/5", "4π/5", "π", "6π/5", "7π/5", "8π/5", "9π/5", "2π"]
    phibins = [
        0,
        np.pi / 5,
        2 * np.pi / 5,
        3 * np.pi / 5,
        4 * np.pi / 5,
        np.pi,
        6 * np.pi / 5,
        7 * np.pi / 5,
        8 * np.pi / 5,
        9 * np.pi / 5,
        2 * np.pi,
    ]
    # reorderphibins = {5: 9, 6: 8, 7: 7, 8: 6, 9: 5}
    labels = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5]
    dfmodel["phi_bin"] = pd.cut(dfmodel["phi"], phibins, labels=labels)

    return dfmodel


def get_mean_cell_properties_of_angle_bin(dfmodeldata: pd.DataFrame, vmax_cmps: float) -> dict[int, pd.DataFrame]:
    if "cos_bin" not in dfmodeldata:
        get_cell_angle(dfmodeldata)

    dfmodeldata["rho"][dfmodeldata["rho"] == 0] = None

    cell_velocities = np.unique(dfmodeldata["vel_x_min"].to_numpy())
    cell_velocities = cell_velocities[cell_velocities >= 0]
    velocity_bins = np.append(cell_velocities, vmax_cmps)

    mid_velocities = np.unique(dfmodeldata["vel_x_mid"].to_numpy())
    mid_velocities = mid_velocities[mid_velocities >= 0]

    mean_bin_properties = {
        bin_number: pd.DataFrame({
            "velocity": mid_velocities,
            "mean_rho": np.zeros_like(mid_velocities, dtype=float),
            "mean_Ye": np.zeros_like(mid_velocities, dtype=float),
            "mean_Q": np.zeros_like(mid_velocities, dtype=float),
        })
        for bin_number in range(10)
    }
    # cos_bin_number = 90
    for bin_number in range(10):
        cos_bin_number = bin_number * 10  # noqa: F841
        # get cells with bin number
        dfanglebin = dfmodeldata.query("cos_bin == @cos_bin_number", inplace=False)

        binned = pd.cut(x=dfanglebin["vel_r_mid"], bins=list(velocity_bins), labels=False, include_lowest=True)
        for binindex, mean_rho in dfanglebin.groupby(binned)["rho"].mean().iteritems():
            mean_bin_properties[bin_number]["mean_rho"][binindex] += mean_rho
        if "Ye" in dfmodeldata:
            for binindex, mean_Ye in dfanglebin.groupby(binned)["Ye"].mean().iteritems():
                mean_bin_properties[bin_number]["mean_Ye"][binindex] += mean_Ye
        if "Q" in dfmodeldata:
            for binindex, mean_Q in dfanglebin.groupby(binned)["Q"].mean().iteritems():
                mean_bin_properties[bin_number]["mean_Q"][binindex] += mean_Q

    return mean_bin_properties


def get_standard_columns(
    dimensions: int, includenico57: bool = False, includeabund: bool = True, pos_unknown: bool = False
) -> list[str]:
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

    if not includeabund:
        return cols

    cols += ["X_Fegroup", "X_Ni56", "X_Co56", "X_Fe52", "X_Cr48"]

    if includenico57:
        cols += ["X_Ni57", "X_Co57"]

    return cols


def save_modeldata(
    dfmodel: pd.DataFrame | pl.LazyFrame | pl.DataFrame,
    outpath: Path | str | None = None,
    vmax: float | None = None,
    headercommentlines: list[str] | None = None,
    modelmeta: dict[str, t.Any] | None = None,
    twolinespercell: bool = False,
    **kwargs: t.Any,
) -> None:
    """Save an artis model.txt (density and composition versus velocity) from a pandas DataFrame of cell properties and other metadata such as the time after explosion.

    1D
    -------
    dfmodel must contain columns inputcellid, vel_r_max_kmps, logrho, X_Fegroup, X_Ni56, X_Co56", X_Fe52, X_Cr48
    modelmeta is not required

    2D
    -------
    dfmodel must contain columns inputcellid, pos_rcyl_mid, pos_z_mid, rho, X_Fegroup, X_Ni56, X_Co56", X_Fe52, X_Cr48
    modelmeta must define: vmax, ncoordgridr and ncoordgridz

    3D
    -------
    dfmodel must contain columns: inputcellid, pos_x_min, pos_y_min, pos_z_min, rho, X_Fegroup, X_Ni56, X_Co56", X_Fe52, X_Cr48
    modelmeta must define: vmax, ncoordgridr and ncoordgridz
    """
    if isinstance(dfmodel, pd.DataFrame):
        dfmodel = pl.from_pandas(dfmodel)

    if "inputcellid" not in dfmodel.columns and "modelgridindex" in dfmodel.columns:
        dfmodel = dfmodel.with_columns(inputcellid=pl.col("modelgridindex") + 1)

    dfmodel = dfmodel.drop(("mass_g", "modelgridindex"), strict=False).lazy().collect()

    if modelmeta is None:
        modelmeta = {}

    assert all(
        key not in modelmeta or modelmeta[key] == kwargs[key] for key in kwargs
    )  # can't define the same thing twice unless the values are the same

    modelmeta |= kwargs  # add any extra keyword arguments to modelmeta

    if "headercommentlines" in modelmeta:
        assert headercommentlines is None
        headercommentlines = modelmeta["headercommentlines"]

    if "vmax_cmps" in modelmeta:
        assert vmax is None or vmax == modelmeta["vmax_cmps"]
        vmax = modelmeta["vmax_cmps"]

    if "npts_model" in modelmeta:
        assert len(dfmodel) == modelmeta["npts_model"]

    timestart = time.perf_counter()
    if modelmeta.get("dimensions") is None:
        modelmeta["dimensions"] = get_dfmodel_dimensions(dfmodel)

    if modelmeta["dimensions"] == 1:
        print(f" 1D grid radial bins: {len(dfmodel)}")

    elif modelmeta["dimensions"] == 2:
        print(f" 2D grid size: {len(dfmodel)} ({modelmeta['ncoordgridrcyl']} x {modelmeta['ncoordgridz']})")
        assert modelmeta["ncoordgridrcyl"] * modelmeta["ncoordgridz"] == len(dfmodel)

    elif modelmeta["dimensions"] == 3:
        dfmodel = dfmodel.rename({"gridindex": "inputcellid"}, strict=False)
        griddimension = round(len(dfmodel) ** (1.0 / 3.0))
        print(f" 3D grid size: {len(dfmodel)} ({griddimension}^3)")
        assert griddimension**3 == len(dfmodel)

    else:
        msg = f"dimensions must be 1, 2, or 3, not {modelmeta['dimensions']}"
        raise ValueError(msg)

    # the Ni57 and Co57 columns are optional, but position is important and they must appear before any other custom cols
    standardcols = get_standard_columns(
        modelmeta["dimensions"],
        includenico57=("X_Ni57" in dfmodel.collect_schema().names() or "X_Co57" in dfmodel.collect_schema().names()),
    )

    # set missing radioabundance columns to zero
    for col in standardcols:
        if col not in dfmodel.collect_schema().names() and col.startswith("X_"):
            dfmodel = dfmodel.with_columns(pl.lit(0.0).alias(col))

    dfmodel = dfmodel.with_columns(pl.col("inputcellid").cast(pl.Int32))
    customcols = [col for col in dfmodel.collect_schema().names() if col not in standardcols]
    customcols.sort(
        key=lambda col: get_z_a_nucname(col) if col.startswith("X_") else (math.inf, 0)
    )  # sort columns by atomic number, mass number

    if outpath is None:
        modelfilepath = Path("model.txt")
    elif Path(outpath).is_dir():
        modelfilepath = Path(outpath) / "model.txt"
    else:
        modelfilepath = Path(outpath)

    if modelfilepath.exists():
        oldfile = modelfilepath.rename(modelfilepath.with_suffix(".bak"))
        print(f"{modelfilepath} already exists. Renaming existing file to {oldfile}")

    with modelfilepath.open("w", encoding="utf-8") as fmodel:
        if headercommentlines:
            fmodel.write("\n".join([f"# {line}" for line in headercommentlines]) + "\n")

        fmodel.write(
            f"{len(dfmodel)}\n"
            if modelmeta["dimensions"] != 2
            else f"{modelmeta['ncoordgridrcyl']} {modelmeta['ncoordgridz']}\n"
        )

        fmodel.write(f"{modelmeta['t_model_init_days']}\n")

        if modelmeta["dimensions"] in {2, 3}:
            fmodel.write(f"{vmax:.4e}\n")

        if customcols:
            fmodel.write(f"#{' '.join(standardcols)} {' '.join(customcols)}\n")

        abundandcustomcols = [*[col for col in standardcols if col.startswith("X_")], *customcols]

        strzeroabund = " ".join(["0.0" if dfmodel.schema[col].is_float() else "0" for col in abundandcustomcols])
        if modelmeta["dimensions"] == 1:
            for inputcellid, vel_r_max_kmps, logrho, *abundandcustomcolvals in dfmodel.select([
                "inputcellid",
                "vel_r_max_kmps",
                "logrho",
                *abundandcustomcols,
            ]).iter_rows():
                fmodel.write(f"{inputcellid:d} {vel_r_max_kmps:9.2f} {logrho:10.8f} ")
                fmodel.write(
                    " ".join([(f"{colvalue:.4e}" if colvalue > 0.0 else "0.0") for colvalue in abundandcustomcolvals])
                    if logrho > -99.0
                    else strzeroabund
                )
                fmodel.write("\n")

        else:
            # startcols are the standard ones, but excluding any abundances
            startcols = [col for col in standardcols if not col.startswith("X_")]
            dfmodel = dfmodel.select([*startcols, *abundandcustomcols])
            if twolinespercell:
                # slow python writer. only needed to create models for classic ARTIS
                nstartcols = len(startcols)
                rhocolindex = len(startcols) - 1
                for colvals in dfmodel.iter_rows():
                    inputcellid = colvals[0]
                    fmodel.write(f"{inputcellid:d}" + "".join(f" {colvalue:.4e}" for colvalue in colvals[1:nstartcols]))
                    fmodel.write("\n")
                    fmodel.write(
                        " ".join((f"{colvalue:.4e}" if colvalue > 0.0 else "0.0") for colvalue in colvals[nstartcols:])
                        if colvals[rhocolindex] > 0.0
                        else strzeroabund
                    )
                    fmodel.write("\n")
            else:
                # fast polars writer
                # set abundances to null for cells with zero density (so that shorter form "0.0" can be written)
                dfmodel = dfmodel.with_columns(
                    pl.when(pl.col("rho") > 0).then(pl.col(col)).otherwise(pl.lit(None)).alias(col)
                    for col in dfmodel.columns
                    if not col.startswith("pos") and col != "inputcellid" and dfmodel.schema[col].is_float()
                )
                fmodel.flush()
                dfmodel.write_csv(
                    fmodel,
                    include_header=False,
                    separator=" ",
                    line_terminator="\n",
                    float_scientific=True,
                    float_precision=4,
                    null_value="0.0",
                )

    print(f"Saved {modelfilepath} (took {time.perf_counter() - timestart:.1f} seconds)")


def get_mgi_of_velocity_kms(modelpath: Path, velocity: float, mgilist: Sequence[int] | None = None) -> int | None:
    """Return the modelgridindex of the cell whose outer velocity is closest to velocity.

    If mgilist is given, then chose from these cells only.
    """
    modeldata, _, _ = get_modeldata_tuple(modelpath)

    if not mgilist:
        mgilist = list(modeldata.index)
        arr_vouter = modeldata["vel_r_max_kmps"].to_numpy()
    else:
        arr_vouter = np.array([modeldata["vel_r_max_kmps"][mgi] for mgi in mgilist])

    index_closestvouter = int(np.abs(arr_vouter - velocity).argmin())

    if velocity < arr_vouter[index_closestvouter] or index_closestvouter + 1 >= len(mgilist):
        return mgilist[index_closestvouter]
    if velocity < arr_vouter[index_closestvouter + 1]:
        return mgilist[index_closestvouter + 1]
    if np.isnan(velocity):
        return None

    print(f"Can't find cell with velocity of {velocity}. Velocity list: {arr_vouter}")
    raise AssertionError


def get_initelemabundances_pandas(modelpath: Path = Path(), printwarningsonly: bool = False) -> pd.DataFrame:
    """Return a table of elemental mass fractions by cell from abundances.

    Deprecated: Use get_initelemabundances_pandas() instead.
    """
    return (
        get_initelemabundances_polars(modelpath=modelpath, printwarningsonly=printwarningsonly)
        .with_columns(pl.col("inputcellid").sub(1).alias("modelgridindex"))
        .collect()
        .to_pandas(use_pyarrow_extension_array=True)
        .set_index("modelgridindex")
    )


def get_initelemabundances_polars(modelpath: Path = Path(), printwarningsonly: bool = False) -> pl.LazyFrame:
    """Return a table of elemental mass fractions by cell from abundances."""
    textfilepath = firstexisting("abundances.txt", folder=modelpath, tryzipped=True)
    parquetfilepath = stripallsuffixes(Path(textfilepath)).with_suffix(".txt.parquet.tmp")

    if parquetfilepath.exists() and Path(textfilepath).stat().st_mtime > parquetfilepath.stat().st_mtime:
        print(f"{textfilepath} has been modified after {parquetfilepath}. Deleting out of date parquet file.")
        parquetfilepath.unlink()

    if parquetfilepath.is_file():
        if not printwarningsonly:
            print(f"Reading {parquetfilepath}")

        abundancedata_lazy = pl.scan_parquet(parquetfilepath)
    else:
        if not printwarningsonly:
            print(f"Reading {textfilepath}")
        ncols = len(pd.read_csv(textfilepath, sep=r"\s+", header=None, comment="#", nrows=1).columns)
        colnames = ["inputcellid", *["X_" + get_elsymbol(x) for x in range(1, ncols)]]

        abundancedata = pl.read_csv(
            zopenpl(textfilepath), has_header=False, separator=" ", comment_prefix="#", infer_schema_length=0
        )

        # fix up multiple spaces at beginning of lines
        abundancedata = abundancedata.transpose()
        abundancedata = pl.DataFrame([
            abundancedata.to_series(idx).drop_nulls() for idx in range(len(abundancedata.columns))
        ]).transpose()
        abundancedata = abundancedata.rename({
            col: colnames[idx] for idx, col in enumerate(abundancedata.columns)
        }).cast({col: pl.Float32 if col.startswith("X_") else pl.Int32 for col in colnames})

        if textfilepath.stat().st_size > 5 * 1024 * 1024:
            print(f"Saving {parquetfilepath}")
            partialparquetfilepath = Path(
                tempfile.mkstemp(dir=modelpath, prefix=f"{parquetfilepath.name}.partial", suffix=".tmp")[1]
            )
            abundancedata.write_parquet(partialparquetfilepath, compression="zstd", statistics=True)
            if parquetfilepath.exists():
                partialparquetfilepath.unlink()
            else:
                partialparquetfilepath.rename(parquetfilepath)

            print("  Done.")
            del abundancedata
            gc.collect()
            abundancedata_lazy = pl.scan_parquet(parquetfilepath)
        else:
            abundancedata_lazy = abundancedata.lazy()

    return abundancedata_lazy


def save_initelemabundances(
    dfelabundances: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    outpath: Path | str | None = None,
    headercommentlines: Sequence[str] | None = None,
) -> None:
    """Save a DataFrame (same format as get_initelemabundances) to abundances.txt.

    columns must be:
        - inputcellid: integer index to match model.txt (starting from 1)
        - X_i: mass fraction of element with two-letter code 'i' (e.g., X_H, X_He, H_Li, ...).
    """
    timestart = time.perf_counter()

    if outpath is None:
        abundancefilename = Path("abundances.txt")
    elif Path(outpath).is_dir():
        abundancefilename = Path(outpath) / "abundances.txt"
    else:
        abundancefilename = Path(outpath)

    if isinstance(dfelabundances, pd.DataFrame):
        dfelabundances = pl.from_pandas(dfelabundances)

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

    # set missing elemental abundance columns to zero
    for col in elcolnames:
        if col not in dfelabundances.columns:
            dfelabundances[col] = 0.0

    dfelabundances = dfelabundances.select(["inputcellid", *elcolnames])

    if abundancefilename.exists():
        oldfile = abundancefilename.rename(abundancefilename.with_suffix(".bak"))
        print(f"{abundancefilename} already exists. Renaming existing file to {oldfile}")

    with Path(abundancefilename).open("w", encoding="utf-8") as fabund:
        if headercommentlines is not None:
            fabund.write("\n".join([f"# {line}" for line in headercommentlines]) + "\n")
        fabund.flush()
        dfelabundances.write_csv(
            fabund,
            include_header=False,
            separator=" ",
            line_terminator="\n",
            float_scientific=True,
            float_precision=4,
            null_value="0.0",
        )

    print(f"Saved {abundancefilename} (took {time.perf_counter() - timestart:.1f} seconds)")


def save_empty_abundance_file(npts_model: int, outputfilepath: str | Path = Path()) -> None:
    """Save dummy abundance file with only zeros."""
    if Path(outputfilepath).is_dir():
        outputfilepath = Path(outputfilepath) / "abundances.txt"

    save_initelemabundances(pl.DataFrame({"inputcellid": range(1, npts_model + 1)}), outpath=outputfilepath)


def get_dfmodel_dimensions(dfmodel: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> int:
    """Guess whether the model is 1D, 2D, or 3D based on which columns are present."""
    columns = dfmodel.columns if isinstance(dfmodel, pd.DataFrame) else dfmodel.collect_schema().names()
    if "pos_x_min" in columns:
        return 3

    return 2 if "pos_z_mid" in columns else 1


def dimension_reduce_model(
    dfmodel: pl.DataFrame | pl.LazyFrame,
    outputdimensions: int,
    dfelabundances: pl.DataFrame | pl.LazyFrame | None = None,
    dfgridcontributions: pl.DataFrame | None = None,
    ncoordgridr: int | None = None,
    ncoordgridz: int | None = None,
    modelmeta: dict[str, t.Any] | None = None,
    **kwargs: t.Any,
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame | None, dict[str, t.Any]]:
    """Convert 3D Cartesian grid model to 1D spherical or 2D cylindrical. Particle gridcontributions and an elemental abundance table can optionally be updated to match."""
    assert outputdimensions in {0, 1, 2}

    dfmodel = dfmodel.lazy().collect()

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

    ngridpoints = modelmeta.get("npts_model", len(dfmodel))
    assert isinstance(ngridpoints, int)
    assert ngridpoints > 0

    print(f"Resampling {ndim_in:d}D model with {ngridpoints} cells to {outputdimensions}D...")
    timestart = time.perf_counter()

    dfmodel = add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta, derived_cols=["velocity", "mass_g"])

    inputcellmass: dict[int, float] = {
        int(cellid): float(rho) for cellid, rho in dfmodel.select(["inputcellid", "mass_g"]).collect().iter_rows()
    }

    km_to_cm = 1e5

    if outputdimensions == 0:
        ncoordgridr = 1
        ncoordgridz = 1
    elif outputdimensions == 1:
        # make 1D model
        if ndim_in == 2:
            ncoordgridr = int(modelmeta.get("ncoordgridrcyl", round(math.sqrt(ngridpoints / 2.0))))
        elif ndim_in == 3:
            ncoordgridx = int(modelmeta.get("ncoordgridx", round(math.cbrt(ngridpoints))))
            ncoordgridr = int(ncoordgridx / 2.0)
        else:
            ncoordgridr = 1
        modelmeta_out["ncoordgridr"] = ncoordgridr
        ncoordgridz = 1
    elif outputdimensions == 2:
        dfmodel = dfmodel.with_columns([
            (pl.col("vel_x_mid") ** 2 + pl.col("vel_y_mid") ** 2).sqrt().alias("vel_rcyl_mid")
        ])
        if ncoordgridz is None:
            ncoordgridz = int(modelmeta.get("ncoordgridx", round(math.cbrt(ngridpoints))))
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
    vel_z_min_max = [
        (-vmax + 2 * vmax * n / ncoordgridz, -vmax + 2 * vmax * (n + 1) / ncoordgridz) for n in range(ncoordgridz)
    ]

    # "r" is the cylindrical radius in 2D, or the spherical radius in 1D
    vel_r_min_max = [(vmax * n / ncoordgridr, vmax * (n + 1) / ncoordgridr) for n in range(ncoordgridr)]

    if outputdimensions == 2:
        lazyframes_matched_cells = [
            dfmodel.filter(
                pl.col("vel_z_mid").is_between(cell_vel_z_min_max[0], cell_vel_z_min_max[1], closed="right")
                & pl.col("vel_rcyl_mid").is_between(cell_vel_r_min_max[0], cell_vel_r_min_max[1], closed="right")
            )
            for cell_vel_z_min_max, cell_vel_r_min_max in itertools.product(vel_z_min_max, vel_r_min_max)
        ]
    else:
        assert outputdimensions in {0, 1}
        lazyframes_matched_cells = [
            dfmodel.filter(pl.col("vel_r_mid").is_between(cell_vel_r_min_max[0], cell_vel_r_min_max[1], closed="right"))
            for cell_vel_r_min_max in vel_r_min_max
        ]

    dataframes_matchedcells = pl.collect_all(lazyframes_matched_cells)

    allmatchedcells = {}
    for mgiout, ((vel_z_min, vel_z_max), (vel_r_min, vel_r_max)) in enumerate(
        itertools.product(vel_z_min_max, vel_r_min_max)
    ):
        matchedcells = dataframes_matchedcells[mgiout]
        if len(matchedcells) == 0:
            rho_out = 0
        else:
            if outputdimensions in {0, 1}:
                shell_volume = (4 * math.pi / 3) * (
                    (vel_r_max * t_model_init_seconds) ** 3 - (vel_r_min * t_model_init_seconds) ** 3
                )
            else:
                # outputdimensions == 2
                shell_volume = (
                    math.pi * (vel_r_max**2 - vel_r_min**2) * (vel_z_max - vel_z_min) * t_model_init_seconds**3
                )
            rho_out = matchedcells["mass_g"].sum() / shell_volume

        cellout: dict[str, t.Any] = {"inputcellid": mgiout + 1, "mass_g": matchedcells["mass_g"].sum()}

        if outputdimensions in {0, 1}:
            cellout |= {
                "logrho": math.log10(max(1e-99, rho_out)) if rho_out > 0.0 else -99.0,
                "vel_r_max_kmps": vel_r_max / km_to_cm,
            }
        elif outputdimensions == 2:
            cellout |= {
                "rho": rho_out,
                "pos_rcyl_mid": (vel_r_min + vel_r_max) / 2 * t_model_init_seconds,
                "pos_z_mid": (vel_z_min + vel_z_max) / 2 * t_model_init_seconds,
            }

        allmatchedcells[mgiout] = (cellout, matchedcells)

    includemissingcolexists = (
        dfgridcontributions is not None and "frac_of_cellmass_includemissing" in dfgridcontributions.columns
    )

    outcells = []
    outcellabundances = []
    outgridcontributions = []

    for mgiout, (dictcell, matchedcells) in allmatchedcells.items():
        matchedcellmass = matchedcells["mass_g"].sum()
        nonempty = matchedcellmass > 0.0
        if matchedcellmass > 0.0 and dfgridcontributions is not None:
            dfcellcont = dfgridcontributions.filter(
                pl.col("cellindex").is_in(matchedcells.get_column("inputcellid").to_list())
            )

            for (particleid,), dfparticlecontribs in dfcellcont.group_by(["particleid"]):
                frac_of_cellmass_avg = (
                    sum(
                        row["frac_of_cellmass"] * inputcellmass[row["cellindex"]]
                        for row in dfparticlecontribs.iter_rows(named=True)
                    )
                    / matchedcellmass
                )

                contriboutrow = {
                    "particleid": particleid,
                    "cellindex": mgiout + 1,
                    "frac_of_cellmass": frac_of_cellmass_avg,
                }

                if includemissingcolexists:
                    frac_of_cellmass_includemissing_avg = (
                        sum(
                            row["frac_of_cellmass_includemissing"] * inputcellmass[row["cellindex"]]
                            for row in dfparticlecontribs.iter_rows(named=True)
                        )
                        / matchedcellmass
                    )
                    contriboutrow["frac_of_cellmass_includemissing"] = frac_of_cellmass_includemissing_avg

                outgridcontributions.append(contriboutrow)

        for column in matchedcells.columns:
            if column.startswith("X_") or column in {"Ye", "cellYe", "q"}:
                # take mass-weighted average mass fraction
                dotprod = matchedcells[column].dot(matchedcells["mass_g"])
                assert isinstance(dotprod, float)
                massfrac = dotprod / matchedcellmass if nonempty else 0.0
                dictcell[column] = massfrac
            elif column == "tracercount":
                dictcell[column] = matchedcells[column].sum()

        outcells.append(dictcell)

        if dfelabundances is not None:
            abund_matchedcells = (
                dfelabundances.filter(pl.col("inputcellid").is_in(matchedcells.get_column("inputcellid").to_list()))
                .lazy()
                .collect()
            )
            dictcellabundances: dict[str, int | float] = {"inputcellid": mgiout + 1}
            for column in dfelabundances.columns:
                if column.startswith("X_"):
                    dotprod = abund_matchedcells[column].dot(matchedcells["mass_g"]) if nonempty else 0.0
                    assert isinstance(dotprod, float)
                    massfrac = dotprod / matchedcellmass if nonempty else 0.0
                    dictcellabundances[column] = massfrac

            outcellabundances.append(dictcellabundances)

    dfmodel_out = pl.DataFrame(outcells)
    modelmeta_out["npts_model"] = len(dfmodel_out)

    dfabundances_out = pl.DataFrame(outcellabundances) if outcellabundances else None

    dfgridcontributions_out = pl.DataFrame(outgridcontributions) if outgridcontributions else None

    print(f"  took {time.perf_counter() - timestart:.1f} seconds")

    return (dfmodel_out, dfabundances_out, dfgridcontributions_out, modelmeta_out)


def scale_model_to_time(
    dfmodel: pd.DataFrame,
    targetmodeltime_days: float,
    t_model_days: float | None = None,
    modelmeta: dict[str, t.Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, t.Any]]:
    """Homologously expand model to targetmodeltime_days by reducing densities and adjusting cell positions."""
    if t_model_days is None:
        assert modelmeta is not None
        t_model_days = modelmeta["t_model_days"]

    assert t_model_days is not None

    timefactor = targetmodeltime_days / t_model_days

    print(
        f"Adjusting t_model to {targetmodeltime_days} days (factor {timefactor}) "
        "using homologous expansion of positions and densities"
    )

    for col in dfmodel.columns:
        if col.startswith("pos_"):
            dfmodel[col] *= timefactor
        elif col == "rho":
            dfmodel["rho"] *= timefactor**-3
        elif col == "logrho":
            dfmodel["logrho"] += math.log10(timefactor**-3)

    if modelmeta is None:
        modelmeta = {}

    modelmeta["t_model_days"] = targetmodeltime_days
    modelmeta.get("headercommentlines", []).append(
        f"scaled from {t_model_days} to {targetmodeltime_days} (no abund change from decays)"
    )

    return dfmodel, modelmeta


def savetologfile(outputfolderpath: Path, logfilename: str = "modellog.txt") -> Callable[..., None]:
    # save the printed output to a log file
    logfilepath = outputfolderpath / logfilename
    logfilepath.unlink(missing_ok=True)

    def logprint(*args: t.Any, **kwargs: t.Any) -> None:
        print(*args, **kwargs)
        with logfilepath.open("a", encoding="utf-8") as logfile:
            logfile.write(" ".join([str(x) for x in args]) + "\n")

    return logprint
