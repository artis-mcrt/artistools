import errno
import gc
import math
import os
import time
import typing as t
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import artistools as at


# @lru_cache(maxsize=3)
def read_modelfile_text(
    filename: Path | str,
    printwarningsonly: bool = False,
    getheadersonly: bool = False,
) -> tuple[pl.DataFrame, dict[t.Any, t.Any]]:
    """Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides."""
    onelinepercellformat = None

    modelmeta: dict[str, t.Any] = {"headercommentlines": []}

    if not printwarningsonly and not getheadersonly:
        print(f"Reading {filename}")

    numheaderrows = 0
    with at.zopen(filename) as fmodel:
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

        filepos = fmodel.tell()
        # if the next line is a single float then the model is 2D or 3D (vmax)
        try:
            modelmeta["vmax_cmps"] = float(fmodel.readline())  # velocity max in cm/s
            xmax_tmodel = modelmeta["vmax_cmps"] * t_model_init_seconds  # xmax = ymax = zmax
            numheaderrows += 1
            if "dimensions" not in modelmeta:  # not already detected as 2D
                modelmeta["dimensions"] = 3
                # number of grid cell steps along an axis (currently the same for xyz)
                ncoordgridx = int(round(npts_model ** (1.0 / 3.0)))
                ncoordgridy = int(round(npts_model ** (1.0 / 3.0)))
                ncoordgridz = int(round(npts_model ** (1.0 / 3.0)))
                assert (ncoordgridx * ncoordgridy * ncoordgridz) == npts_model
                modelmeta["ncoordgridx"] = ncoordgridx
                modelmeta["ncoordgridy"] = ncoordgridy
                modelmeta["ncoordgridz"] = ncoordgridz
                if ncoordgridx == ncoordgridy == ncoordgridz:
                    modelmeta["ncoordgrid"] = ncoordgridx

                if not printwarningsonly:
                    print(f"  detected 3D model file with {ncoordgridx}x{ncoordgridy}x{ncoordgridz}={npts_model} cells")

        except ValueError:
            assert modelmeta.get("dimensions", -1) != 2, "2D model should have a vmax line here"
            if "dimensions" not in modelmeta:
                if not printwarningsonly:
                    print(f"  detected 1D model file with {npts_model} radial zones")
                modelmeta["dimensions"] = 1
                getheadersonly = False  # need to find vmax from the model data

            fmodel.seek(filepos)  # undo the readline() and go back

        columns = None
        filepos = fmodel.tell()
        line = fmodel.readline()
        if line.startswith("#"):
            numheaderrows += 1
            columns = line.lstrip("#").split()
        else:
            fmodel.seek(filepos)  # undo the readline() and go back

        data_line_even = fmodel.readline()
        ncols_line_even = len(data_line_even.split())
        data_line_odd = fmodel.readline()
        ncols_line_odd = len(data_line_odd.split())

        if columns is None:
            columns = get_standard_columns(modelmeta["dimensions"], includenico57=True)
            # last two abundances are optional
            assert columns is not None
            if ncols_line_even == ncols_line_odd and (ncols_line_even + ncols_line_odd) > len(columns):
                # one line per cell format
                ncols_line_odd = 0

            assert len(columns) in {
                ncols_line_even + ncols_line_odd,
                ncols_line_even + ncols_line_odd + 2,
            }
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
        print("  using fast method polars.read_csv (requires one line per cell and single space delimiters)")

        pldtypes = {col: pl.Int32 if col == "inputcellid" else pl.Float32 for col in columns}

        dfmodel = pl.read_csv(
            at.zopenpl(filename),
            separator=" ",
            comment_prefix="#",
            new_columns=columns,
            n_rows=npts_model,
            has_header=False,
            skip_rows=numheaderrows,
            dtypes=pldtypes,
            truncate_ragged_lines=True,
        )
    else:
        nrows_read = 1 if getheadersonly else npts_model
        skiprows: list[int] | int | None = (
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

        dfmodel = pl.from_pandas(dfmodelpd)

    if "velocity_outer" in dfmodel.columns:
        dfmodel = dfmodel.rename({"velocity_outer": "vel_r_max_kmps"})

    if modelmeta["dimensions"] == 1:
        vmax_kmps = dfmodel["vel_r_max_kmps"].max()
        assert isinstance(vmax_kmps, float)
        modelmeta["vmax_cmps"] = vmax_kmps * 1.0e5

    elif modelmeta["dimensions"] == 2:
        wid_init_rcyl = modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridrcyl"]
        wid_init_z = 2 * modelmeta["vmax_cmps"] * t_model_init_seconds / modelmeta["ncoordgridz"]
        modelmeta["wid_init_rcyl"] = wid_init_rcyl
        modelmeta["wid_init_z"] = wid_init_z
        if not getheadersonly:
            # check pos_rcyl_mid and pos_z_mid are correct

            for inputcellid, cell_pos_rcyl_mid, cell_pos_z_mid in dfmodel.select(
                ["inputcellid", "pos_rcyl_mid", "pos_z_mid"]
            ).iter_rows():
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
        modelmeta["wid_init_x"] = wid_init_x
        modelmeta["wid_init_y"] = wid_init_x
        modelmeta["wid_init_z"] = wid_init_x
        modelmeta["wid_init"] = wid_init_x
        colrenames = {"pos_x": "pos_x_min", "pos_y": "pos_y_min", "pos_z": "pos_z_min"}
        dfmodel = dfmodel.rename({a: b for a, b in colrenames.items() if a in dfmodel.columns})
        if "pos_x_min" in dfmodel.columns and not printwarningsonly:
            print("  model cell positions are defined in the header")
        elif not getheadersonly:

            def vectormatch(vec1: list[float], vec2: list[float]) -> bool:
                xclose = np.isclose(vec1[0], vec2[0], atol=wid_init_x / 2.0)
                yclose = np.isclose(vec1[1], vec2[1], atol=wid_init_x / 2.0)
                zclose = np.isclose(vec1[2], vec2[2], atol=wid_init_x / 2.0)

                return all([xclose, yclose, zclose])

            posmatch_xyz = True
            posmatch_zyx = True
            # important cell numbers to check for coordinate column order
            indexlist = [
                0,
                ncoordgridx - 1,
                (ncoordgridx - 1) * (ncoordgridy - 1),
                (ncoordgridx - 1) * (ncoordgridy - 1) * (ncoordgridz - 1),
            ]
            for modelgridindex in indexlist:
                xindex = modelgridindex % ncoordgridx
                yindex = (modelgridindex // ncoordgridx) % ncoordgridy
                zindex = (modelgridindex // (ncoordgridx * ncoordgridy)) % ncoordgridz
                pos_x_min = -xmax_tmodel + 2 * xindex * xmax_tmodel / ncoordgridx
                pos_y_min = -xmax_tmodel + 2 * yindex * xmax_tmodel / ncoordgridy
                pos_z_min = -xmax_tmodel + 2 * zindex * xmax_tmodel / ncoordgridz

                pos3_in = list(dfmodel.select(["inputpos_a", "inputpos_b", "inputpos_c"]).row(modelgridindex))

                if not vectormatch(pos3_in, [pos_x_min, pos_y_min, pos_z_min]):
                    posmatch_xyz = False

                if not vectormatch(pos3_in, [pos_z_min, pos_y_min, pos_x_min]):
                    posmatch_zyx = False

            assert posmatch_xyz != posmatch_zyx  # one option must match
            colrenames = {}
            if posmatch_xyz:
                print("  model cell positions are consistent with x-y-z column order")
                colrenames = {"inputpos_a": "pos_x_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_z_min"}

            if posmatch_zyx:
                print("  cell positions are consistent with z-y-x column order")
                colrenames = {"inputpos_a": "pos_z_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_x_min"}
            dfmodel = dfmodel.rename({a: b for a, b in colrenames.items() if a in dfmodel.columns})

    return dfmodel, modelmeta


def get_modeldata_polars(
    modelpath: Path | str = Path(),
    get_elemabundances: bool = False,
    derived_cols: list[str] | str | None = None,
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
        filename = at.firstexisting("model.txt", folder=inputpath, tryzipped=True)
    elif inputpath.is_file():  # passed in a filename instead of the modelpath
        filename = inputpath
        modelpath = Path(inputpath).parent
    elif not inputpath.exists() and inputpath.parts[0] == "codecomparison":
        modelpath = inputpath
        _, inputmodel, _ = modelpath.parts
        filename = Path(at.get_config()["codecomparisonmodelartismodelpath"], inputmodel, "model.txt")
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputpath)

    filenameparquet = at.stripallsuffixes(Path(filename)).with_suffix(".txt.parquet")

    if filenameparquet.exists() and Path(filename).stat().st_mtime > filenameparquet.stat().st_mtime:
        print(f"{filename} has been modified after {filenameparquet}. Deleting out of date parquet file.")
        filenameparquet.unlink()

    dfmodel: pl.LazyFrame | None | pl.DataFrame = None
    if not getheadersonly and filenameparquet.is_file():
        if not printwarningsonly:
            print(f"Reading model table from {filenameparquet}")
        try:
            dfmodel = pl.scan_parquet(filenameparquet)
        except pl.exceptions.ComputeError:
            print(f"Problem reading {filenameparquet}. Will regenerate and overwite from text source.")
            dfmodel = None

    if dfmodel is not None:
        # already read in the model data, just need to get the metadata
        getheadersonly = True

    dfmodel_textfile, modelmeta = read_modelfile_text(
        filename=filename,
        printwarningsonly=printwarningsonly,
        getheadersonly=getheadersonly,
    )

    if dfmodel is None:
        dfmodel = dfmodel_textfile
    elif dfmodel.schema != dfmodel_textfile.schema:
        print(f"ERROR: parquet schema does not match model.txt. Remove {filenameparquet} and try again.")
        raise AssertionError

    mebibyte = 1024 * 1024
    if isinstance(dfmodel, pl.DataFrame) and filename.stat().st_size > 5 * mebibyte and not getheadersonly:
        print(f"Saving {filenameparquet}")
        dfmodel.write_parquet(filenameparquet, compression="zstd")
        print("  Done.")
        del dfmodel
        gc.collect()
        dfmodel = pl.scan_parquet(filenameparquet)

    if not printwarningsonly:
        print(f"  model is {modelmeta['dimensions']}D with {modelmeta['npts_model']} cells")
    dfmodel = dfmodel.lazy()

    if get_elemabundances:
        abundancedata = get_initelemabundances_polars(modelpath, printwarningsonly=printwarningsonly)
        dfmodel = dfmodel.join(abundancedata, how="inner", on="inputcellid")

    dfmodel = dfmodel.with_columns(pl.col("inputcellid").sub(1).alias("modelgridindex"))

    if derived_cols:
        dfmodel = add_derived_cols_to_modeldata(
            dfmodel=dfmodel,
            derived_cols=derived_cols,
            modelmeta=modelmeta,
            modelpath=modelpath,
        )

    return dfmodel, modelmeta


def get_modeldata_tuple(*args: t.Any, **kwargs: t.Any) -> tuple[pd.DataFrame, float, float]:
    """Get model from model.txt file.

    DEPRECATED: Use get_modeldata() instead.
    """
    dfmodel, modelmeta = get_modeldata(*args, **kwargs)

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

    dfmodel = pl.DataFrame(
        {"modelgridindex": range(ncoordgrid**3), "inputcellid": range(1, 1 + ncoordgrid**3)},
        schema={"modelgridindex": pl.Int32, "inputcellid": pl.Int32},
    ).lazy()

    dfmodel = dfmodel.with_columns(
        [
            pl.col("modelgridindex").mod(ncoordgrid).alias("n_x"),
            (pl.col("modelgridindex") // ncoordgrid).mod(ncoordgrid).alias("n_y"),
            (pl.col("modelgridindex") // (ncoordgrid**2)).mod(ncoordgrid).alias("n_z"),
        ]
    )

    dfmodel = dfmodel.with_columns(
        [
            (-xmax + 2 * pl.col("n_x") * xmax / ncoordgrid).cast(pl.Float32).alias("pos_x_min"),
            (-xmax + 2 * pl.col("n_y") * xmax / ncoordgrid).cast(pl.Float32).alias("pos_y_min"),
            (-xmax + 2 * pl.col("n_z") * xmax / ncoordgrid).cast(pl.Float32).alias("pos_z_min"),
        ]
    )

    standardcols = get_standard_columns(3, includenico57=includenico57)

    dfmodel = dfmodel.with_columns(
        [pl.lit(0.0, dtype=pl.Float32).alias(colname) for colname in standardcols if colname not in dfmodel.columns]
    )

    dfmodel = dfmodel.select([*standardcols, "modelgridindex"])

    return dfmodel, modelmeta


def get_modeldata(
    modelpath: Path | str = Path(),
    get_elemabundances: bool = False,
    derived_cols: list[str] | str | None = None,
    printwarningsonly: bool = False,
    getheadersonly: bool = False,
) -> tuple[pd.DataFrame, dict[t.Any, t.Any]]:
    """Call get_modeldata_polars() and convert to pandas DataFrame."""
    pldfmodel, modelmeta = get_modeldata_polars(
        modelpath=modelpath,
        get_elemabundances=get_elemabundances,
        derived_cols=derived_cols,
        printwarningsonly=printwarningsonly,
        getheadersonly=getheadersonly,
    )
    dfmodel = pldfmodel.collect().to_pandas(use_pyarrow_extension_array=True)
    if modelmeta["npts_model"] > 100000 and not getheadersonly:
        dfmodel.info(verbose=False, memory_usage="deep")

    return dfmodel, modelmeta


def add_derived_cols_to_modeldata(
    dfmodel: pl.DataFrame | pl.LazyFrame,
    derived_cols: list[str],
    modelmeta: dict[str, t.Any],
    modelpath: Path | None = None,
) -> pl.LazyFrame:
    """Add columns to modeldata using e.g. derived_cols = ("velocity", "Ye")."""
    # with lazy mode, we can add every column and then drop the ones we don't need
    dfmodel = dfmodel.lazy()
    original_cols = dfmodel.columns

    t_model_init_seconds = modelmeta["t_model_init_days"] * 86400.0
    keep_all = "ALL" in derived_cols

    dimensions = modelmeta["dimensions"]
    match dimensions:
        case 1:
            axes = ["r"]

            dfmodel = dfmodel.with_columns(pl.col("vel_r_max_kmps").shift(n=1, fill_value=0.0).alias("vel_r_min_kmps"))

            dfmodel = dfmodel.with_columns(
                [
                    (pl.col("vel_r_min_kmps") * 1e5).alias("vel_r_min"),
                    (pl.col("vel_r_max_kmps") * 1e5).alias("vel_r_max"),
                ]
            )

            dfmodel = dfmodel.with_columns(((pl.col("vel_r_max") + pl.col("vel_r_min")) / 2).alias("vel_r_mid"))

            dfmodel = dfmodel.with_columns(
                [
                    (
                        (4.0 / 3.0)
                        * math.pi
                        * (
                            pl.col("vel_r_max_kmps").cast(pl.Float64).pow(3)
                            - pl.col("vel_r_min_kmps").cast(pl.Float64).pow(3)
                        )
                        * pl.lit(1e5 * t_model_init_seconds).pow(3)
                    ).alias("volume")
                ]
            )

        case 2:
            axes = ["rcyl", "z"]

            assert t_model_init_seconds is not None
            # pos_mid is defined in the input file
            # TODO: get wid_init from modelmeta
            dfmodel = dfmodel.with_columns(
                [(pl.col(f"pos_{ax}_mid") - modelmeta[f"wid_init_{ax}"] / 2.0).alias(f"pos_{ax}_min") for ax in axes]
            )

            dfmodel = dfmodel.with_columns(
                [(pl.col(f"pos_{ax}_mid") + modelmeta[f"wid_init_{ax}"] / 2.0).alias(f"pos_{ax}_max") for ax in axes]
            )

            # add a 3D radius column
            axes.append("r")
            dfmodel = dfmodel.with_columns(
                [
                    (
                        pl.col("pos_rcyl_min").pow(2)
                        + pl.min_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
                    )
                    .sqrt()
                    .alias("pos_r_min")
                ]
            )

            dfmodel = dfmodel.with_columns(
                [(pl.col("pos_rcyl_mid").pow(2) + pl.col("pos_z_mid").pow(2)).sqrt().alias("pos_r_mid")]
            )

            dfmodel = dfmodel.with_columns(
                [
                    (
                        pl.col("pos_rcyl_max").pow(2)
                        + pl.max_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
                    )
                    .sqrt()
                    .alias("pos_r_max"),
                ]
            )

            dfmodel = dfmodel.with_columns(
                [
                    (
                        math.pi
                        * (
                            pl.col("pos_rcyl_max").cast(pl.Float64).pow(2)
                            - pl.col("pos_rcyl_min").cast(pl.Float64).pow(2)
                        )
                        * modelmeta["wid_init_z"]
                    ).alias("volume")
                ]
            )

        case 3:
            axes = ["x", "y", "z"]
            for ax in axes:
                if f"wid_init_{ax}" not in modelmeta:
                    modelmeta[f"wid_init_{ax}"] = modelmeta["wid_init"]

            dfmodel = dfmodel.with_columns(
                [pl.lit(modelmeta["wid_init_x"] * modelmeta["wid_init_y"] * modelmeta["wid_init_z"]).alias("volume")]
            )

            dfmodel = dfmodel.with_columns(
                [(pl.col(f"pos_{ax}_min") + 0.5 * modelmeta[f"wid_init_{ax}"]).alias(f"pos_{ax}_mid") for ax in axes]
            )

            dfmodel = dfmodel.with_columns(
                [(pl.col(f"pos_{ax}_min") + modelmeta[f"wid_init_{ax}"]).alias(f"pos_{ax}_max") for ax in axes]
            )

            # add a 3D radius column
            axes.append("r")

            # xyz positions can be negative, so the min xyz side of the cube can have a larger radius than the max side
            dfmodel = dfmodel.with_columns(
                [
                    (
                        pl.min_horizontal(pl.col("pos_x_min").abs(), pl.col("pos_x_max").abs()).pow(2)
                        + pl.min_horizontal(pl.col("pos_y_min").abs(), pl.col("pos_y_max").abs()).pow(2)
                        + pl.min_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
                    )
                    .sqrt()
                    .alias("pos_r_min")
                ]
            )

            dfmodel = dfmodel.with_columns(
                [
                    (pl.col("pos_x_mid").pow(2) + pl.col("pos_y_mid").pow(2) + pl.col("pos_z_mid").pow(2))
                    .sqrt()
                    .alias("pos_r_mid")
                ]
            )

            dfmodel = dfmodel.with_columns(
                [
                    (
                        pl.max_horizontal(pl.col("pos_x_min").abs(), pl.col("pos_x_max").abs()).pow(2)
                        + pl.max_horizontal(pl.col("pos_y_min").abs(), pl.col("pos_y_max").abs()).pow(2)
                        + pl.max_horizontal(pl.col("pos_z_min").abs(), pl.col("pos_z_max").abs()).pow(2)
                    )
                    .sqrt()
                    .alias("pos_r_max")
                ]
            )

    for col in dfmodel.columns:
        if col.startswith("pos_"):
            dfmodel = dfmodel.with_columns((pl.col(col) / t_model_init_seconds).alias(col.replace("pos_", "vel_")))

    if "logrho" not in dfmodel.columns:
        dfmodel = dfmodel.with_columns(pl.col("rho").log10().alias("logrho"))

    if "rho" not in dfmodel.columns:
        dfmodel = dfmodel.with_columns(
            (pl.when(pl.col("logrho") > -98).then(10 ** pl.col("logrho")).otherwise(0.0)).alias("rho")
        )

    dfmodel = dfmodel.with_columns([(pl.col("rho") * pl.col("volume")).alias("mass_g")])

    dfmodel = dfmodel.with_columns(
        [(pl.col(colname) / 29979245800.0).alias(f"{colname}_on_c") for colname in dfmodel.columns]
    )

    if unknown_cols := [
        col
        for col in derived_cols
        if col not in dfmodel.columns and col not in {"pos_min", "pos_max", "ALL", "velocity"}
    ]:
        print(f"WARNING: Unknown derived columns: {unknown_cols}")

    if "pos_min" in derived_cols:
        derived_cols.extend(col for col in dfmodel.columns if col.startswith("pos_") and col.endswith("_min"))

    if "pos_max" in derived_cols:
        derived_cols.extend(col for col in dfmodel.columns if col.startswith("pos_") and col.endswith("_max"))

    if "velocity" in derived_cols:
        derived_cols.extend(col for col in dfmodel.columns if col.startswith("vel_"))

    if not keep_all:
        dfmodel = dfmodel.drop([col for col in dfmodel.columns if col not in original_cols and col not in derived_cols])

    if "angle_bin" in derived_cols:
        assert modelpath is not None
        dfmodel = pl.from_pandas(
            get_cell_angle(dfmodel.collect().to_pandas(use_pyarrow_extension_array=True), modelpath)
        ).lazy()

    # if "Ye" in derived_cols and os.path.isfile(modelpath / "Ye.txt"):
    #     dfmodel["Ye"] = at.inputmodel.opacityinputfile.get_Ye_from_file(modelpath)
    # if "Q" in derived_cols and os.path.isfile(modelpath / "Q_energy.txt"):
    #     dfmodel["Q"] = at.inputmodel.energyinputfiles.get_Q_energy_from_file(modelpath)

    return dfmodel


def get_cell_angle(dfmodel: pd.DataFrame, modelpath: Path) -> pd.DataFrame:
    """Get angle between origin to cell midpoint and the syn_dir axis."""
    syn_dir = at.get_syn_dir(modelpath)
    xhat = np.array([1.0, 0.0, 0.0])

    cos_theta = np.zeros(len(dfmodel))
    phi = np.zeros(len(dfmodel))
    for i, (_, cell) in enumerate(dfmodel.iterrows()):
        mid_point = [cell["pos_x_mid"], cell["pos_y_mid"], cell["pos_z_mid"]]
        cos_theta[i] = (np.dot(mid_point, syn_dir)) / (at.vec_len(mid_point) * at.vec_len(syn_dir))

        vec1 = np.cross(mid_point, syn_dir)
        vec2 = np.cross(xhat, syn_dir)
        cosphi = np.dot(vec1, vec2) / at.vec_len(vec1) / at.vec_len(vec2)

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


def get_mean_cell_properties_of_angle_bin(
    dfmodeldata: pd.DataFrame, vmax_cmps: float, modelpath: Path
) -> dict[int, pd.DataFrame]:
    if "cos_bin" not in dfmodeldata:
        get_cell_angle(dfmodeldata, modelpath)

    dfmodeldata["rho"][dfmodeldata["rho"] == 0] = None

    cell_velocities = np.unique(dfmodeldata["vel_x_min"].values)
    cell_velocities = cell_velocities[cell_velocities >= 0]
    velocity_bins = np.append(cell_velocities, vmax_cmps)

    mid_velocities = np.unique(dfmodeldata["vel_x_mid"].values)
    mid_velocities = mid_velocities[mid_velocities >= 0]

    mean_bin_properties = {
        bin_number: pd.DataFrame(
            {
                "velocity": mid_velocities,
                "mean_rho": np.zeros_like(mid_velocities, dtype=float),
                "mean_Ye": np.zeros_like(mid_velocities, dtype=float),
                "mean_Q": np.zeros_like(mid_velocities, dtype=float),
            }
        )
        for bin_number in range(10)
    }
    # cos_bin_number = 90
    for bin_number in range(10):
        cos_bin_number = bin_number * 10  # noqa: F841
        # get cells with bin number
        dfanglebin = dfmodeldata.query("cos_bin == @cos_bin_number", inplace=False)

        binned = pd.cut(dfanglebin["vel_r_mid"], velocity_bins, labels=False, include_lowest=True)
        for binindex, mean_rho in dfanglebin.groupby(binned)["rho"].mean().iteritems():
            mean_bin_properties[bin_number]["mean_rho"][binindex] += mean_rho
        if "Ye" in dfmodeldata:
            for binindex, mean_Ye in dfanglebin.groupby(binned)["Ye"].mean().iteritems():
                mean_bin_properties[bin_number]["mean_Ye"][binindex] += mean_Ye
        if "Q" in dfmodeldata:
            for binindex, mean_Q in dfanglebin.groupby(binned)["Q"].mean().iteritems():
                mean_bin_properties[bin_number]["mean_Q"][binindex] += mean_Q

    return mean_bin_properties


def get_standard_columns(dimensions: int, includenico57: bool = False, includeabund: bool = True) -> list[str]:
    """Get standard (artis classic) columns for modeldata DataFrame."""
    match dimensions:
        case 1:
            cols = ["inputcellid", "vel_r_max_kmps", "logrho"]
        case 2:
            cols = ["inputcellid", "pos_rcyl_mid", "pos_z_mid", "rho"]
        case 3:
            cols = ["inputcellid", "pos_x_min", "pos_y_min", "pos_z_min", "rho"]

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

    dfmodel = dfmodel.drop("mass_g").lazy().collect()

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
        modelmeta["dimensions"] = at.get_dfmodel_dimensions(dfmodel)

    if modelmeta["dimensions"] == 1:
        print(f" 1D grid radial bins: {len(dfmodel)}")
    elif modelmeta["dimensions"] == 2:
        print(f" 2D grid size: {len(dfmodel)} ({modelmeta['ncoordgridrcyl']} x {modelmeta['ncoordgridz']})")
        assert modelmeta["ncoordgridrcyl"] * modelmeta["ncoordgridz"] == len(dfmodel)
    elif modelmeta["dimensions"] == 3:
        if "gridindex" in dfmodel.columns:
            dfmodel = dfmodel.rename({"gridindex": "inputcellid"})
        griddimension = int(round(len(dfmodel) ** (1.0 / 3.0)))
        print(f" 3D grid size: {len(dfmodel)} ({griddimension}^3)")
        assert griddimension**3 == len(dfmodel)
    else:
        msg = f"dimensions must be 1, 2, or 3, not {modelmeta['dimensions']}"
        raise ValueError(msg)

    # the Ni57 and Co57 columns are optional, but position is important and they must appear before any other custom cols
    standardcols = get_standard_columns(
        modelmeta["dimensions"], includenico57=("X_Ni57" in dfmodel.columns or "X_Co57" in dfmodel.columns)
    )

    # set missing radioabundance columns to zero
    for col in standardcols:
        if col not in dfmodel.columns and col.startswith("X_"):
            dfmodel = dfmodel.with_columns(pl.lit(0.0).alias(col))

    dfmodel = dfmodel.with_columns(pl.col("inputcellid").cast(pl.Int32))
    customcols = [col for col in dfmodel.columns if col not in standardcols]
    customcols.sort(
        key=lambda col: at.get_z_a_nucname(col) if col.startswith("X_") else (math.inf, 0)
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
            fmodel.write(f"{vmax}\n")

        if customcols:
            fmodel.write(f'#{" ".join(standardcols)} {" ".join(customcols)}\n')

        abundcols = [*[col for col in standardcols if col.startswith("X_")], *customcols]

        zeroabund = " ".join(["0.0" for _ in abundcols])
        if modelmeta["dimensions"] == 1:
            for inputcellid, vel_r_max_kmps, logrho, *othercolvals in dfmodel.select(
                ["inputcellid", "vel_r_max_kmps", "logrho", *abundcols]
            ).iter_rows():
                fmodel.write(f"{inputcellid:d} {vel_r_max_kmps:9.2f} {logrho:10.8f} ")
                fmodel.write(
                    " ".join([(f"{colvalue:.4e}" if colvalue > 0.0 else "0.0") for colvalue in othercolvals])
                    if logrho > -99.0
                    else zeroabund
                )
                fmodel.write("\n")

        else:
            lineend = "\n" if twolinespercell else " "
            startcols = get_standard_columns(modelmeta["dimensions"], includeabund=False)
            dfmodel = dfmodel.select([*startcols, *abundcols])
            nstartcols = len(startcols)
            for colvals in dfmodel.iter_rows():
                startvals = colvals[:nstartcols]
                fmodel.write(" ".join(str(val) for val in startvals))
                fmodel.write(lineend)
                fmodel.write(
                    " ".join((f"{colvalue:.4e}" if colvalue > 0.0 else "0.0") for colvalue in colvals[nstartcols:])
                    if startvals[-1] > 0.0
                    else zeroabund
                )
                fmodel.write("\n")

    print(f"Saved {modelfilepath} (took {time.perf_counter() - timestart:.1f} seconds)")


def get_mgi_of_velocity_kms(modelpath: Path, velocity: float, mgilist: t.Sequence[int] | None = None) -> int | float:
    """Return the modelgridindex of the cell whose outer velocity is closest to velocity.

    If mgilist is given, then chose from these cells only.
    """
    modeldata, _, _ = get_modeldata_tuple(modelpath)

    velocity = float(velocity)

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
        return math.nan

    print(f"Can't find cell with velocity of {velocity}. Velocity list: {arr_vouter}")
    raise AssertionError


def get_initelemabundances(
    modelpath: Path = Path(),
    printwarningsonly: bool = False,
) -> pd.DataFrame:
    return (
        get_initelemabundances_polars(modelpath=modelpath, printwarningsonly=printwarningsonly)
        .with_columns(pl.col("inputcellid").sub(1).alias("modelgridindex"))
        .collect()
        .to_pandas(use_pyarrow_extension_array=True)
        .set_index("modelgridindex")
    )


@lru_cache(maxsize=8)
def get_initelemabundances_polars(
    modelpath: Path = Path(),
    printwarningsonly: bool = False,
) -> pl.LazyFrame:
    """Return a table of elemental mass fractions by cell from abundances."""
    abundancefilepath = at.firstexisting("abundances.txt", folder=modelpath, tryzipped=True)

    filenameparquet = at.stripallsuffixes(Path(abundancefilepath)).with_suffix(".txt.parquet")
    if filenameparquet.exists() and Path(abundancefilepath).stat().st_mtime > filenameparquet.stat().st_mtime:
        print(f"{abundancefilepath} has been modified after {filenameparquet}. Deleting out of date parquet file.")
        filenameparquet.unlink()

    if filenameparquet.is_file():
        if not printwarningsonly:
            print(f"Reading {filenameparquet}")

        abundancedata_lazy = pl.scan_parquet(filenameparquet)
    else:
        if not printwarningsonly:
            print(f"Reading {abundancefilepath}")
        ncols = len(pd.read_csv(abundancefilepath, sep=r"\s+", header=None, comment="#", nrows=1).columns)
        colnames = ["inputcellid", *["X_" + at.get_elsymbol(x) for x in range(1, ncols)]]
        dtypes = {col: pl.Float32 if col.startswith("X_") else pl.Int32 for col in colnames}

        abundancedata = pl.read_csv(
            at.zopenpl(abundancefilepath),
            has_header=False,
            separator=" ",
            comment_prefix="#",
            infer_schema_length=0,
        )

        # fix up multiple spaces at beginning of lines
        abundancedata = abundancedata.transpose()
        abundancedata = pl.DataFrame(
            [abundancedata.to_series(idx).drop_nulls() for idx in range(len(abundancedata.columns))]
        ).transpose()
        abundancedata = abundancedata.rename(
            {col: colnames[idx] for idx, col in enumerate(abundancedata.columns)}
        ).cast(dtypes)  # type: ignore[arg-type]

        if abundancefilepath.stat().st_size > 5 * 1024 * 1024:
            print(f"Saving {filenameparquet}")
            abundancedata.write_parquet(filenameparquet, compression="zstd")
            print("  Done.")
            del abundancedata
            gc.collect()
            abundancedata_lazy = pl.scan_parquet(filenameparquet)
        else:
            abundancedata_lazy = abundancedata.lazy()

    return abundancedata_lazy


def save_initelemabundances(
    dfelabundances: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    outpath: Path | str | None = None,
    headercommentlines: t.Sequence[str] | None = None,
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

    dfelabundances = dfelabundances.clone().lazy().collect()

    dfelabundances = dfelabundances.with_columns([pl.col("inputcellid").cast(pl.Int32)])

    atomic_numbers = {
        at.get_atomic_number(colname[2:]) for colname in dfelabundances.select(pl.selectors.starts_with("X_")).columns
    }
    max_atomic_number = max([30, *atomic_numbers])
    elcolnames = [f"X_{at.get_elsymbol(Z)}" for Z in range(1, 1 + max_atomic_number)]
    for colname in elcolnames:
        if colname not in dfelabundances.columns:
            dfelabundances = dfelabundances.with_columns(pl.lit(0.0).alias(colname))

    # set missing elemental abundance columns to zero
    for col in elcolnames:
        if col not in dfelabundances.columns:
            dfelabundances[col] = 0.0

    if abundancefilename.exists():
        oldfile = abundancefilename.rename(abundancefilename.with_suffix(".bak"))
        print(f"{abundancefilename} already exists. Renaming existing file to {oldfile}")

    with Path(abundancefilename).open("w", encoding="utf-8") as fabund:
        if headercommentlines is not None:
            fabund.write("\n".join([f"# {line}" for line in headercommentlines]) + "\n")
        for inputcellid, *abundvals in dfelabundances.select(["inputcellid", *elcolnames]).iter_rows():
            fabund.write(f"{inputcellid:} ")
            fabund.write(" ".join([f"{abund:.6e}" for abund in abundvals]))
            fabund.write("\n")

    print(f"Saved {abundancefilename} (took {time.perf_counter() - timestart:.1f} seconds)")


def save_empty_abundance_file(npts_model: int, outputfilepath: str | Path = Path()) -> None:
    """Save dummy abundance file with only zeros."""
    if Path(outputfilepath).is_dir():
        outputfilepath = Path(outputfilepath) / "abundances.txt"

    save_initelemabundances(
        pl.DataFrame(
            {
                "inputcellid": range(1, npts_model + 1),
            }
        ),
        outpath=outputfilepath,
    )


def get_dfmodel_dimensions(dfmodel: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> int:
    """Guess whether the model is 1D, 2D, or 3D based on which columns are present."""
    if "pos_x_min" in dfmodel.columns:
        return 3

    return 2 if "pos_z_mid" in dfmodel.columns else 1


def dimension_reduce_3d_model(
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

    if outputdimensions == 0:
        outputdimensions = 1
        ncoordgridr = 1
        ncoordgridz = 0

    dfmodel = dfmodel.lazy().collect()

    if modelmeta is None:
        modelmeta = {}

    modelmeta_out = {k: v for k, v in modelmeta.items() if not k.startswith("ncoord") and not k.startswith("wid_init")}

    assert all(
        key not in modelmeta_out or modelmeta_out[key] == kwargs[key] for key in kwargs
    )  # can't define the same thing twice unless the values are the same

    modelmeta_out |= kwargs  # add any extra keyword arguments to modelmeta

    t_model_init_seconds = modelmeta["t_model_init_days"] * 24 * 60 * 60
    vmax = modelmeta["vmax_cmps"]
    xmax = vmax * t_model_init_seconds
    ngridpoints = modelmeta.get("npts_model", len(dfmodel))
    ncoordgridx = modelmeta.get("ncoordgridx", int(round(ngridpoints ** (1.0 / 3.0))))

    ndim_in = modelmeta["dimensions"]
    assert ndim_in > outputdimensions or (ndim_in == outputdimensions == 1 and ncoordgridr == 1)
    modelmeta_out["dimensions"] = outputdimensions

    print(f"Resampling {ndim_in:d}D model with {ngridpoints} cells to {outputdimensions}D...")
    timestart = time.perf_counter()

    dfmodel = at.inputmodel.add_derived_cols_to_modeldata(
        dfmodel, modelmeta=modelmeta, derived_cols=["velocity", "mass_g"]
    )

    inputcellmass: dict[int, float] = {
        int(cellid): float(rho) for cellid, rho in dfmodel.select(["inputcellid", "mass_g"]).collect().iter_rows()
    }

    km_to_cm = 1e5
    if ncoordgridr is None:
        ncoordgridr = int(ncoordgridx / 2.0)

    if ncoordgridz is None:
        ncoordgridz = int(ncoordgridx)

    if outputdimensions == 2:
        dfmodel = dfmodel.with_columns(
            [(pl.col("vel_x_mid") ** 2 + pl.col("vel_y_mid") ** 2).sqrt().alias("vel_rcyl_mid")]
        )
        modelmeta_out["ncoordgridz"] = ncoordgridz
        modelmeta_out["ncoordgridrcyl"] = ncoordgridr
        modelmeta_out["wid_init_z"] = 2 * xmax / ncoordgridz
        modelmeta_out["wid_init_rcyl"] = xmax / ncoordgridr
    else:
        # 1D
        modelmeta_out["ncoordgridr"] = ncoordgridr

    # velocities in cm/s
    velocity_bins_z_min = (
        [-vmax + 2 * vmax * n / ncoordgridz for n in range(ncoordgridz)] if outputdimensions == 2 else [-vmax]
    )
    velocity_bins_z_max = (
        [-vmax + 2 * vmax * n / ncoordgridz for n in range(1, ncoordgridz + 1)] if outputdimensions == 2 else [vmax]
    )

    velocity_bins_r_min = [vmax * n / ncoordgridr for n in range(ncoordgridr)]
    velocity_bins_r_max = [vmax * n / ncoordgridr for n in range(1, ncoordgridr + 1)]

    allmatchedcells = {}
    for n_z, (vel_z_min, vel_z_max) in enumerate(zip(velocity_bins_z_min, velocity_bins_z_max)):
        # "r" is the cylindrical radius in 2D, or the spherical radius in 1D
        for n_r, (vel_r_min, vel_r_max) in enumerate(zip(velocity_bins_r_min, velocity_bins_r_max)):
            assert vel_r_max > vel_r_min
            cellindexout = n_z * ncoordgridr + n_r + 1

            if outputdimensions == 1:
                matchedcells = dfmodel.filter(
                    pl.col("vel_r_mid").is_between(vel_r_min, vel_r_max, closed="right")
                ).collect()
            elif outputdimensions == 2:
                matchedcells = dfmodel.filter(
                    pl.col("vel_rcyl_mid").is_between(vel_r_min, vel_r_max, closed="right")
                    & pl.col("vel_z_mid").is_between(vel_z_min, vel_z_max, closed="right")
                ).collect()

            if len(matchedcells) == 0:
                rho_out = 0
            else:
                if outputdimensions == 1:
                    shell_volume = (4 * math.pi / 3) * (
                        (vel_r_max * t_model_init_seconds) ** 3 - (vel_r_min * t_model_init_seconds) ** 3
                    )
                elif outputdimensions == 2:
                    shell_volume = (
                        math.pi * (vel_r_max**2 - vel_r_min**2) * (vel_z_max - vel_z_min) * t_model_init_seconds**3
                    )
                rho_out = matchedcells["mass_g"].sum() / shell_volume

            cellout: dict[str, t.Any] = {"inputcellid": cellindexout, "mass_g": matchedcells["mass_g"].sum()}

            if outputdimensions == 1:
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

            allmatchedcells[cellindexout] = (
                cellout,
                matchedcells,
            )

    includemissingcolexists = (
        dfgridcontributions is not None and "frac_of_cellmass_includemissing" in dfgridcontributions.columns
    )

    outcells = []
    outcellabundances = []
    outgridcontributions = []

    for cellindexout, (dictcell, matchedcells) in allmatchedcells.items():
        matchedcellmass = matchedcells["mass_g"].sum()
        nonempty = matchedcellmass > 0.0
        if matchedcellmass > 0.0 and dfgridcontributions is not None:
            dfcellcont = dfgridcontributions.filter(pl.col("cellindex").is_in(matchedcells["inputcellid"]))

            for (particleid,), dfparticlecontribs in dfcellcont.group_by(["particleid"]):  # type: ignore[misc]
                frac_of_cellmass_avg = (
                    sum(
                        row["frac_of_cellmass"] * inputcellmass[row["cellindex"]]
                        for row in dfparticlecontribs.iter_rows(named=True)
                    )
                    / matchedcellmass
                )

                contriboutrow = {
                    "particleid": particleid,
                    "cellindex": cellindexout,
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
            if column.startswith("X_") or column in {"cellYe", "q"}:
                # take mass-weighted average mass fraction
                dotprod = matchedcells[column].dot(matchedcells["mass_g"])
                assert isinstance(dotprod, float)
                massfrac = dotprod / matchedcellmass if nonempty else 0.0
                dictcell[column] = massfrac
            elif column == "tracercount":
                dictcell["tracercount"] = matchedcells["tracercount"].sum()

        outcells.append(dictcell)

        if dfelabundances is not None:
            abund_matchedcells = (
                dfelabundances.filter(pl.col("inputcellid").is_in(matchedcells["inputcellid"])).lazy().collect()
            )
            dictcellabundances: dict[str, int | float] = {"inputcellid": cellindexout}
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

    return (
        dfmodel_out,
        dfabundances_out,
        dfgridcontributions_out,
        modelmeta_out,
    )


def scale_model_to_time(
    dfmodel: pd.DataFrame,
    targetmodeltime_days: float,
    t_model_days: float | None = None,
    modelmeta: dict[str, t.Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, t.Any] | None]:
    """Homologously expand model to targetmodeltime_days by reducing densities and adjusting cell positions."""
    if t_model_days is None:
        assert modelmeta is not None
        t_model_days = modelmeta["t_model_days"]

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

    if modelmeta is not None:
        modelmeta["t_model_days"] = targetmodeltime_days
        modelmeta.get("headercommentlines", []).append(
            "scaled from {t_model_days} to {targetmodeltime_days} (no abund change from decays)"
        )

    return dfmodel, modelmeta


def savetologfile(outputfolderpath, logfilename="modellog.txt"):
    # save the printed output to a log file
    logfilepath = outputfolderpath / logfilename
    logfilepath.unlink(missing_ok=True)

    def logprint(*args, **kwargs):
        print(*args, **kwargs)
        with logfilepath.open("a", encoding="utf-8") as logfile:
            logfile.write(" ".join([str(x) for x in args]) + "\n")

    return logprint
