from __future__ import annotations

import argparse
import errno
import gc
import math
import os.path
import pickle
import time
import typing as t
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from typeguard import typechecked

import artistools as at


@typechecked
def read_modelfile_text(
    filename: Path | str,
    printwarningsonly: bool = False,
    getheadersonly: bool = False,
    skipnuclidemassfraccolumns: bool = False,
    dtype_backend: t.Literal["pyarrow", "numpy_nullable"] = "numpy_nullable",
) -> tuple[pd.DataFrame, dict[t.Any, t.Any]]:
    """Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides."""
    onelinepercellformat = None

    modelmeta: dict[str, t.Any] = {"headercommentlines": []}

    modelpath = Path(filename).parent
    if not printwarningsonly:
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
            ncoordgrid_r, ncoordgrid_z = (int(n) for n in line.strip().split(" "))
            modelmeta["ncoordgrid_r"] = ncoordgrid_r
            modelmeta["ncoordgrid_z"] = ncoordgrid_z
            modelcellcount = ncoordgrid_r * ncoordgrid_z
            print(f"  detected 2D model file with {ncoordgrid_r}x{ncoordgrid_z}={modelcellcount} cells")
        else:
            modelcellcount = int(line)

        modelmeta["npts_model"] = modelcellcount
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
                ncoordgridx = int(round(modelcellcount ** (1.0 / 3.0)))
                ncoordgridy = int(round(modelcellcount ** (1.0 / 3.0)))
                ncoordgridz = int(round(modelcellcount ** (1.0 / 3.0)))
                assert (ncoordgridx * ncoordgridy * ncoordgridz) == modelcellcount
                modelmeta["ncoordgridx"] = ncoordgridx
                modelmeta["ncoordgridy"] = ncoordgridy
                modelmeta["ncoordgridz"] = ncoordgridz
                if ncoordgridx == ncoordgridy == ncoordgridz:
                    modelmeta["ncoordgrid"] = ncoordgridx

                if not printwarningsonly:
                    print(
                        "  detected 3D model file with"
                        f" {ncoordgridx}x{ncoordgridy}x{ncoordgridz}={modelcellcount} cells"
                    )

        except ValueError:
            assert modelmeta.get("dimensions", -1) != 2  # 2D model should have vmax line here
            if "dimensions" not in modelmeta:
                if not printwarningsonly:
                    print(f"  detected 1D model file with {modelcellcount} radial zones")
                modelmeta["dimensions"] = 1
                getheadersonly = False

            fmodel.seek(filepos)  # undo the readline() and go back

        columns = None
        filepos = fmodel.tell()
        line = fmodel.readline()
        if line.startswith("#"):
            numheaderrows += 1
            columns = line.lstrip("#").split()
        else:
            fmodel.seek(filepos)  # undo the readline() and go back

        data_line_even = fmodel.readline().split()
        ncols_line_even = len(data_line_even)
        ncols_line_odd = len(fmodel.readline().split())

        if columns is None:
            if modelmeta["dimensions"] == 1:
                columns = [
                    "inputcellid",
                    "velocity_outer",
                    "logrho",
                    "X_Fegroup",
                    "X_Ni56",
                    "X_Co56",
                    "X_Fe52",
                    "X_Cr48",
                    "X_Ni57",
                    "X_Co57",
                ]

            elif modelmeta["dimensions"] == 2:
                columns = [
                    "inputcellid",
                    "pos_r_mid",
                    "pos_z_mid",
                    "rho",
                    "X_Fegroup",
                    "X_Ni56",
                    "X_Co56",
                    "X_Fe52",
                    "X_Cr48",
                    "X_Ni57",
                    "X_Co57",
                ]

            elif modelmeta["dimensions"] == 3:
                columns = [
                    "inputcellid",
                    "inputpos_a",
                    "inputpos_b",
                    "inputpos_c",
                    "rho",
                    "X_Fegroup",
                    "X_Ni56",
                    "X_Co56",
                    "X_Fe52",
                    "X_Cr48",
                    "X_Ni57",
                    "X_Co57",
                ]
            # last two abundances are Optional
            assert columns is not None
            if ncols_line_even == ncols_line_odd and (ncols_line_even + ncols_line_odd) > len(columns):
                # one line per cell format
                ncols_line_odd = 0

            assert len(columns) in [
                ncols_line_even + ncols_line_odd,
                ncols_line_even + ncols_line_odd + 2,
            ]
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

    if skipnuclidemassfraccolumns:
        if not printwarningsonly:
            print("  skipping nuclide abundance columns in model")
        if modelmeta["dimensions"] == 1:
            ncols_line_even = 3
        elif modelmeta["dimensions"] == 2:
            ncols_line_even = 4
        elif modelmeta["dimensions"] == 3:
            ncols_line_even = 5
        ncols_line_odd = 0

    nrows_read = 1 if getheadersonly else modelcellcount

    skiprows: list[int] | int | None

    skiprows = (
        numheaderrows
        if onelinepercellformat
        else [
            x
            for x in range(numheaderrows + modelcellcount * 2)
            if x < numheaderrows or (x - numheaderrows - 1) % 2 == 0
        ]
    )

    dtypes: defaultdict[str, str]
    if dtype_backend == "pyarrow":
        dtypes = defaultdict(lambda: "float32[pyarrow]")
        dtypes["inputcellid"] = "int32[pyarrow]"
        dtypes["tracercount"] = "int32[pyarrow]"
    else:
        dtypes = defaultdict(lambda: "float32")
        dtypes["inputcellid"] = "int32"
        dtypes["tracercount"] = "int32"

    # each cell takes up two lines in the model file
    dfmodel = pd.read_csv(
        at.zopen(filename),
        sep=r"\s+",
        engine="c",
        header=None,
        skiprows=skiprows,
        names=columns[:ncols_line_even],
        usecols=columns[:ncols_line_even],
        nrows=nrows_read,
        dtype=dtypes,
        dtype_backend=dtype_backend,
    )

    if ncols_line_odd > 0 and not onelinepercellformat:
        # read in the odd rows and merge dataframes
        skipevenrows = [
            x
            for x in range(numheaderrows + modelcellcount * 2)
            if x < numheaderrows or (x - numheaderrows - 1) % 2 == 1
        ]
        dfmodeloddlines = pd.read_csv(
            at.zopen(filename),
            sep=r"\s+",
            engine="c",
            header=None,
            skiprows=skipevenrows,
            names=columns[ncols_line_even:],
            nrows=nrows_read,
            dtype=dtypes,
            dtype_backend=dtype_backend,
        )
        assert len(dfmodel) == len(dfmodeloddlines)
        dfmodel = dfmodel.merge(dfmodeloddlines, left_index=True, right_index=True)
        del dfmodeloddlines

    if len(dfmodel) > modelcellcount:
        dfmodel = dfmodel.iloc[:modelcellcount]

    assert len(dfmodel) == modelcellcount or getheadersonly

    dfmodel.index.name = "cellid"
    # dfmodel.drop('inputcellid', axis=1, inplace=True)

    if modelmeta["dimensions"] == 1:
        dfmodel["velocity_inner"] = np.concatenate([[0.0], dfmodel["velocity_outer"].to_numpy()[:-1]])
        dfmodel["cellmass_grams"] = (
            10 ** dfmodel["logrho"]
            * (4.0 / 3.0)
            * 3.14159265
            * (dfmodel["velocity_outer"] ** 3 - dfmodel["velocity_inner"] ** 3)
            * (1e5 * t_model_init_seconds) ** 3
        )
        modelmeta["vmax_cmps"] = dfmodel.velocity_outer.max() * 1e5

    elif modelmeta["dimensions"] == 3:
        wid_init = at.get_wid_init_at_tmodel(modelpath, modelcellcount, modelmeta["t_model_init_days"], xmax_tmodel)
        modelmeta["wid_init"] = wid_init
        dfmodel["cellmass_grams"] = dfmodel["rho"] * wid_init**3

        dfmodel = dfmodel.rename(columns={"pos_x": "pos_x_min", "pos_y": "pos_y_min", "pos_z": "pos_z_min"})
        if "pos_x_min" in dfmodel.columns and not printwarningsonly:
            print("  model cell positions are defined in the header")
        elif not getheadersonly:

            def vectormatch(vec1: list[float], vec2: list[float]) -> bool:
                xclose = np.isclose(vec1[0], vec2[0], atol=xmax_tmodel / ncoordgridx)
                yclose = np.isclose(vec1[1], vec2[1], atol=xmax_tmodel / ncoordgridy)
                zclose = np.isclose(vec1[2], vec2[2], atol=xmax_tmodel / ncoordgridz)

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

                cell = dfmodel.iloc[modelgridindex]
                if not vectormatch(
                    [cell.inputpos_a, cell.inputpos_b, cell.inputpos_c],
                    [pos_x_min, pos_y_min, pos_z_min],
                ):
                    posmatch_xyz = False
                if not vectormatch(
                    [cell.inputpos_a, cell.inputpos_b, cell.inputpos_c],
                    [pos_z_min, pos_y_min, pos_x_min],
                ):
                    posmatch_zyx = False

            assert posmatch_xyz != posmatch_zyx  # one option must match
            if posmatch_xyz:
                print("  model cell positions are consistent with x-y-z column order")
                dfmodel = dfmodel.rename(
                    columns={"inputpos_a": "pos_x_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_z_min"},
                )
            if posmatch_zyx:
                print("  cell positions are consistent with z-y-x column order")
                dfmodel = dfmodel.rename(
                    columns={"inputpos_a": "pos_z_min", "inputpos_b": "pos_y_min", "inputpos_c": "pos_x_min"},
                )

    modelmeta["modelcellcount"] = modelcellcount

    return dfmodel, modelmeta


@typechecked
def get_modeldata(
    modelpath: Path | str = Path(),
    get_elemabundances: bool = False,
    derived_cols: t.Sequence[str] | None = None,
    printwarningsonly: bool = False,
    getheadersonly: bool = False,
    skipnuclidemassfraccolumns: bool = False,
    dtype_backend: t.Literal["pyarrow", "numpy_nullable"] = "numpy_nullable",
    use_polars: bool = False,
) -> tuple[pd.DataFrame, dict[t.Any, t.Any]]:
    """Read an artis model.txt file containing cell velocities, densities, and mass fraction abundances of radioactive nuclides.

    Parameters
    ----------
        - inputpath: either a path to model.txt file, or a folder containing model.txt
        - get_elemabundances: also read elemental abundances (abundances.txt) and
            merge with the output DataFrame

    return dfmodel, modelmeta
        - dfmodel: a pandas DataFrame with a row for each model grid cell
        - modelmeta: a dictionary of input model parameters, with keys such as t_model_init_days, vmax_cmps, dimensions, etc.
    """
    inputpath = Path(modelpath)
    if use_polars:
        dtype_backend = "pyarrow"

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

    dfmodel = None
    filenameparquet = at.stripallsuffixes(Path(filename)).with_suffix(".txt.parquet")

    if filenameparquet.exists() and Path(filename).stat().st_mtime > filenameparquet.stat().st_mtime:
        print(f"{filename} has been modified after {filenameparquet}. Deleting out of date parquet file.")
        filenameparquet.unlink()

    source_textfile_details = {"st_size": filename.stat().st_size, "st_mtime": filename.stat().st_mtime}

    if filenameparquet.is_file() and not getheadersonly:
        if not printwarningsonly:
            print(f"  reading data table from {filenameparquet}")

        pqmetadata = pq.read_metadata(filenameparquet)
        if (
            b"artismodelmeta" not in pqmetadata.metadata
            or b"source_textfile_details" not in pqmetadata.metadata
            or pickle.dumps(source_textfile_details) != pqmetadata.metadata[b"source_textfile_details"]
        ):
            print(f"  text source {filename} doesn't match file header of {filenameparquet}. Removing parquet file")
            filenameparquet.unlink(missing_ok=True)
        else:
            modelmeta = pickle.loads(pqmetadata.metadata[b"artismodelmeta"])

            columns = (
                [col for col in pqmetadata.schema.names if not col.startswith("X_")]
                if skipnuclidemassfraccolumns
                else None
            )
            dfmodel = pd.read_parquet(
                filenameparquet,
                columns=columns,
                dtype_backend=dtype_backend,
            )
            print(f"  model is {modelmeta['dimensions']}D with {modelmeta['npts_model']} cells")

    if dfmodel is None:
        skipnuclidemassfraccolumns = False
        dfmodel, modelmeta = read_modelfile_text(
            filename=filename,
            printwarningsonly=printwarningsonly,
            getheadersonly=getheadersonly,
            skipnuclidemassfraccolumns=skipnuclidemassfraccolumns,
            dtype_backend=dtype_backend,
        )

        if len(dfmodel) > 1000 and not getheadersonly and not skipnuclidemassfraccolumns:
            print(f"Saving {filenameparquet}")
            patable = pa.Table.from_pandas(dfmodel)

            custom_metadata = {
                b"source_textfile_details": pickle.dumps(source_textfile_details),
                b"artismodelmeta": pickle.dumps(modelmeta),
            }
            merged_metadata = {**custom_metadata, **(patable.schema.metadata or {})}
            patable = patable.replace_schema_metadata(merged_metadata)
            pq.write_table(patable, filenameparquet, compression="ZSTD")
            # dfmodel.to_parquet(filenameparquet, compression="zstd")
            print("  Done.")

    dfmodel = pl.from_pandas(dfmodel).lazy()

    if get_elemabundances:
        abundancedata = pl.from_pandas(
            get_initelemabundances(modelpath, dtype_backend=dtype_backend, printwarningsonly=printwarningsonly)
        ).lazy()
        dfmodel = dfmodel.join(abundancedata, how="inner", on="inputcellid")

    if derived_cols:
        dfmodel = add_derived_cols_to_modeldata(
            dfmodel=dfmodel,
            derived_cols=derived_cols,
            dimensions=modelmeta["dimensions"],
            t_model_init_seconds=modelmeta["t_model_init_days"] * 86400.0,
            wid_init=modelmeta.get("wid_init", None),
            modelpath=modelpath,
        )

    if not use_polars:
        dfmodel = dfmodel.collect().to_pandas(use_pyarrow_extension_array=(dtype_backend == "pyarrow"))
        if modelmeta["npts_model"] > 100000 and not getheadersonly:
            dfmodel.info(verbose=False, memory_usage="deep")

    return dfmodel, modelmeta


@typechecked
def get_modeldata_tuple(*args: t.Any, **kwargs: t.Any) -> tuple[pd.DataFrame, float, float]:
    """Get model from model.txt file
    DEPRECATED: Use get_modeldata() instead.
    """
    dfmodel, modelmeta = get_modeldata(*args, **kwargs)

    return dfmodel, modelmeta["t_model_init_days"], modelmeta["vmax_cmps"]


@typechecked
def add_derived_cols_to_modeldata(
    dfmodel: pl.DataFrame | pl.LazyFrame,
    derived_cols: t.Sequence[str],
    dimensions: int | None = None,
    t_model_init_seconds: float | None = None,
    wid_init: float | None = None,
    modelpath: Path | None = None,
) -> pl.LazyFrame:
    """Add columns to modeldata using e.g. derived_cols = ('velocity', 'Ye')."""
    if dimensions is None:
        dimensions = get_dfmodel_dimensions(dfmodel)

    dfmodel = dfmodel.lazy()
    newcols = []

    if dimensions == 3:
        axes = ["x", "y", "z"]
        if "velocity" in derived_cols or "vel_min" in derived_cols:
            assert t_model_init_seconds is not None
            newcols += [(pl.col(f"pos_{ax}_min") / t_model_init_seconds).alias(f"vel_{ax}_min") for ax in axes]

        if "velocity" in derived_cols or "vel_max" in derived_cols:
            assert t_model_init_seconds is not None
            newcols += [
                ((pl.col(f"pos_{ax}_min") + wid_init) / t_model_init_seconds).alias(f"vel_{ax}_max") for ax in axes
            ]

        if any(col in derived_cols for col in ["velocity", "vel_mid", "vel_r_mid"]):
            assert wid_init is not None
            assert t_model_init_seconds is not None
            dfmodel = dfmodel.with_columns(
                [
                    ((pl.col(f"pos_{ax}_min") + (0.5 * wid_init)) / t_model_init_seconds).alias(f"vel_{ax}_mid")
                    for ax in axes
                ]
            )

            newcols.append(
                (pl.col("vel_x_mid").pow(2) + pl.col("vel_y_mid").pow(2) + pl.col("vel_z_mid").pow(2))
                .sqrt()
                .alias("vel_r_mid")
            )

    if dimensions == 3 and "pos_mid" in derived_cols or "angle_bin" in derived_cols:
        assert wid_init is not None
        newcols += [(pl.col(f"pos_{ax}_min") + 0.5 * wid_init).alias(f"pos_{ax}_mid") for ax in axes]

    if dimensions == 3 and "pos_max" in derived_cols:
        assert wid_init is not None
        newcols += [(pl.col(f"pos_{ax}_min") + wid_init).alias(f"pos_{ax}_max") for ax in axes]

    if "logrho" in derived_cols and "logrho" not in dfmodel.columns:
        newcols.append(pl.col("rho").log10().alias("logrho"))

    if "rho" in derived_cols and "rho" not in dfmodel.columns:
        newcols.append((10 ** pl.col("logrho")).alias("rho"))

    dfmodel = dfmodel.with_columns(newcols)

    if "angle_bin" in derived_cols:
        assert modelpath is not None
        dfmodel = pl.from_pandas(get_cell_angle(dfmodel.collect().to_pandas(), modelpath)).lazy()

    # if "Ye" in derived_cols and os.path.isfile(modelpath / "Ye.txt"):
    #     dfmodel["Ye"] = at.inputmodel.opacityinputfile.get_Ye_from_file(modelpath)
    # if "Q" in derived_cols and os.path.isfile(modelpath / "Q_energy.txt"):
    #     dfmodel["Q"] = at.inputmodel.energyinputfiles.get_Q_energy_from_file(modelpath)

    return dfmodel


def get_cell_angle(dfmodel: pd.DataFrame, modelpath: Path) -> pd.DataFrame:
    """Get angle between origin to cell midpoint and the syn_dir axis."""
    syn_dir = at.get_syn_dir(modelpath)

    cos_theta = np.zeros(len(dfmodel))
    for i, (_, cell) in enumerate(dfmodel.iterrows()):
        mid_point = [cell["pos_x_mid"], cell["pos_y_mid"], cell["pos_z_mid"]]
        cos_theta[i] = (np.dot(mid_point, syn_dir)) / (at.vec_len(mid_point) * at.vec_len(syn_dir))
    dfmodel["cos_theta"] = cos_theta
    cos_bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]  # including end bin
    labels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # to agree with escaping packet bin numbers
    assert at.get_viewingdirection_costhetabincount() == 10
    assert at.get_viewingdirection_phibincount() == 10
    dfmodel["cos_bin"] = pd.cut(dfmodel["cos_theta"], cos_bins, labels=labels)
    # dfmodel['cos_bin'] = np.searchsorted(cos_bins, dfmodel['cos_theta'].values) -1

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
        cos_bin_number = bin_number * 10
        # get cells with bin number
        dfanglebin = dfmodeldata.query("cos_bin == @cos_bin_number", inplace=False)

        binned = pd.cut(dfanglebin["vel_r_mid"], velocity_bins, labels=False, include_lowest=True)
        i = 0
        for binindex, mean_rho in dfanglebin.groupby(binned)["rho"].mean().iteritems():
            i += 1
            mean_bin_properties[bin_number]["mean_rho"][binindex] += mean_rho
        i = 0
        if "Ye" in dfmodeldata:
            for binindex, mean_Ye in dfanglebin.groupby(binned)["Ye"].mean().iteritems():
                i += 1
                mean_bin_properties[bin_number]["mean_Ye"][binindex] += mean_Ye
        if "Q" in dfmodeldata:
            for binindex, mean_Q in dfanglebin.groupby(binned)["Q"].mean().iteritems():
                i += 1
                mean_bin_properties[bin_number]["mean_Q"][binindex] += mean_Q

    return mean_bin_properties


def get_2d_modeldata(modelpath: str | Path) -> pd.DataFrame:
    filepath = Path(modelpath, "model.txt")
    with filepath.open() as f:
        num_lines = sum(1 for _ in f)
    skiprowlist = [0, 1, 2]
    skiprowlistodds = skiprowlist + [i for i in range(3, num_lines) if i % 2 == 1]
    skiprowlistevens = skiprowlist + [i for i in range(3, num_lines) if i % 2 == 0]

    model1stlines = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=skiprowlistevens)
    model2ndlines = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=skiprowlistodds)

    model = pd.concat([model1stlines, model2ndlines], axis=1)
    column_names = [
        "inputcellid",
        "cellpos_mid[r]",
        "cellpos_mid[z]",
        "rho_model",
        "X_Fegroup",
        "X_Ni56",
        "X_Co56",
        "X_Fe52",
        "X_Cr48",
    ]
    model.columns = column_names
    return model


def get_3d_model_data_merged_model_and_abundances_minimal(args: argparse.Namespace) -> pd.DataFrame:
    """Get 3D data without generating all the extra columns in standard routine.
    Needed for large (eg. 200^3) models.
    """
    model = get_3d_modeldata_minimal(args.modelpath)
    abundances = get_initelemabundances(args.modelpath[0])

    with Path(args.modelpath[0], "model.txt").open() as fmodelin:
        fmodelin.readline()  # npts_model3d
        args.t_model = float(fmodelin.readline())  # days
        args.vmax = float(fmodelin.readline())  # v_max in [cm/s]

    print(model.keys())

    merge_dfs = model.merge(abundances, how="inner", on="inputcellid")

    del model
    del abundances
    gc.collect()

    merge_dfs.info(verbose=False, memory_usage="deep")

    return merge_dfs


def get_3d_modeldata_minimal(modelpath: str | Path) -> pd.DataFrame:
    """Read 3D model without generating all the extra columns in standard routine.
    Needed for large (eg. 200^3) models.
    """
    model = pd.read_csv(Path(modelpath, "model.txt"), delim_whitespace=True, header=None, skiprows=3, dtype=np.float64)
    columns = [
        "inputcellid",
        "cellpos_in[z]",
        "cellpos_in[y]",
        "cellpos_in[x]",
        "rho_model",
        "X_Fegroup",
        "X_Ni56",
        "X_Co56",
        "X_Fe52",
        "X_Cr48",
    ]
    model = pd.DataFrame(model.to_numpy().reshape(-1, 10))
    model.columns = columns

    print("model.txt memory usage:")
    model.info(verbose=False, memory_usage="deep")
    return model


@typechecked
def save_modeldata(
    dfmodel: pd.DataFrame,
    t_model_init_days: float | None = None,
    filename: Path | str | None = None,
    modelpath: Path | str | None = None,
    vmax: float | None = None,
    dimensions: int | None = None,
    headercommentlines: list[str] | None = None,
    modelmeta: dict[str, t.Any] | None = None,
    twolinespercell: bool = False,
    float_format: str = ".4e",
) -> None:
    """Save a pandas DataFrame and snapshot time into ARTIS model.txt."""
    if modelmeta:
        if "headercommentlines" in modelmeta:
            assert headercommentlines is None
            headercommentlines = modelmeta["headercommentlines"]

        if "vmax_cmps" in modelmeta:
            assert vmax is None
            vmax = modelmeta["vmax_cmps"]

        if "dimensions" in modelmeta:
            assert dimensions is None
            dimensions = modelmeta["dimensions"]

        if "t_model_init_days" in modelmeta:
            assert t_model_init_days is None
            t_model_init_days = modelmeta["t_model_init_days"]

        if "modelcellcount" in modelmeta:
            assert len(dfmodel) == modelmeta["modelcellcount"]

    timestart = time.perf_counter()
    if dimensions is None:
        dimensions = at.get_dfmodel_dimensions(dfmodel)

    assert dimensions in [1, 3]
    if dimensions == 1:
        standardcols = ["inputcellid", "velocity_outer", "logrho", "X_Fegroup", "X_Ni56", "X_Co56", "X_Fe52", "X_Cr48"]
    elif dimensions == 3:
        dfmodel = dfmodel.rename(columns={"gridindex": "inputcellid"})
        griddimension = int(round(len(dfmodel) ** (1.0 / 3.0)))
        print(f" grid size: {len(dfmodel)} ({griddimension}^3)")
        assert griddimension**3 == len(dfmodel)

        standardcols = [
            "inputcellid",
            "pos_x_min",
            "pos_y_min",
            "pos_z_min",
            "rho",
            "X_Fegroup",
            "X_Ni56",
            "X_Co56",
            "X_Fe52",
            "X_Cr48",
        ]

    # these two columns are Optional, but position is important and they must appear before t.Any other custom cols
    if "X_Ni57" in dfmodel.columns:
        standardcols.append("X_Ni57")

    if "X_Co57" in dfmodel.columns:
        standardcols.append("X_Co57")

    # set missing radioabundance columns to zero
    for col in standardcols:
        if col not in dfmodel.columns and col.startswith("X_"):
            dfmodel[col] = 0.0

    dfmodel["inputcellid"] = dfmodel["inputcellid"].astype(int)
    customcols = [col for col in dfmodel.columns if col not in standardcols]
    customcols.sort(
        key=lambda col: at.get_z_a_nucname(col) if col.startswith("X_") else (float("inf"), 0)
    )  # sort columns by atomic number, mass number

    assert modelpath is not None or filename is not None
    if filename is None:
        filename = "model.txt"
    modelfilepath = Path(modelpath, filename) if modelpath is not None else Path(filename)

    with modelfilepath.open("w", encoding="utf-8") as fmodel:
        if headercommentlines is not None:
            fmodel.write("\n".join([f"# {line}" for line in headercommentlines]) + "\n")
        fmodel.write(f"{len(dfmodel)}\n")
        fmodel.write(f"{t_model_init_days}\n")
        if dimensions == 3:
            fmodel.write(f"{vmax}\n")

        if customcols:
            fmodel.write(f'#{" ".join(standardcols)} {" ".join(customcols)}\n')

        abundcols = [*[col for col in standardcols if col.startswith("X_")], *customcols]

        if dimensions == 1:
            for cell in dfmodel.itertuples(index=False):
                fmodel.write(f"{cell.inputcellid:d} {cell.velocity_outer:9.2f} {cell.logrho:10.8f} ")
                fmodel.write(" ".join([f"{getattr(cell, col)}" for col in abundcols]))
                fmodel.write("\n")

        elif dimensions == 3:
            # dfmodel.to_csv(fmodel, sep=" ", float_format=lambda x: "%.4e" if x > 0 else "0.0")
            zeroabund = " ".join(["0.0" for _ in abundcols])
            line_end = "\n" if twolinespercell else " "
            for inputcellid, posxmin, posymin, poszmin, rho, *othercolvals in dfmodel[
                ["inputcellid", "pos_x_min", "pos_y_min", "pos_z_min", "rho", *abundcols]
            ].itertuples(index=False, name=None):
                fmodel.write(f"{inputcellid:d} {posxmin} {posymin} {poszmin} {rho}{line_end}")
                fmodel.write(
                    " ".join(
                        [
                            (
                                (f"{colvalue:{float_format}}" if colvalue > 0.0 else "0.0")
                                if isinstance(colvalue, float)
                                else f"{colvalue}"
                            )
                            for colvalue in othercolvals
                        ]
                    )
                    if rho > 0.0
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
        arr_vouter = modeldata["velocity_outer"].to_numpy()
    else:
        arr_vouter = np.array([modeldata["velocity_outer"][mgi] for mgi in mgilist])

    index_closestvouter = int(np.abs(arr_vouter - velocity).argmin())

    if velocity < arr_vouter[index_closestvouter] or index_closestvouter + 1 >= len(mgilist):
        return mgilist[index_closestvouter]
    if velocity < arr_vouter[index_closestvouter + 1]:
        return mgilist[index_closestvouter + 1]
    if np.isnan(velocity):
        return float("nan")

    print(f"Can't find cell with velocity of {velocity}. Velocity list: {arr_vouter}")
    raise AssertionError


@lru_cache(maxsize=8)
def get_initelemabundances(
    modelpath: Path = Path(),
    printwarningsonly: bool = False,
    dtype_backend: t.Literal["pyarrow", "numpy_nullable"] = "numpy_nullable",
) -> pd.DataFrame:
    """Return a table of elemental mass fractions by cell from abundances."""
    abundancefilepath = at.firstexisting("abundances.txt", folder=modelpath, tryzipped=True)

    filenameparquet = at.stripallsuffixes(Path(abundancefilepath)).with_suffix(".txt.parquet")
    if filenameparquet.exists() and Path(abundancefilepath).stat().st_mtime > filenameparquet.stat().st_mtime:
        print(f"{abundancefilepath} has been modified after {filenameparquet}. Deleting out of date parquet file.")
        filenameparquet.unlink()

    if filenameparquet.is_file():
        if not printwarningsonly:
            print(f"Reading {filenameparquet}")

        abundancedata = pd.read_parquet(filenameparquet, dtype_backend=dtype_backend)
    else:
        if not printwarningsonly:
            print(f"Reading {abundancefilepath}")
        ncols = len(
            pd.read_csv(at.zopen(abundancefilepath), delim_whitespace=True, header=None, comment="#", nrows=1).columns
        )
        colnames = ["inputcellid", *["X_" + at.get_elsymbol(x) for x in range(1, ncols)]]
        dtypes = (
            {col: "float32[pyarrow]" if col.startswith("X_") else "int32[pyarrow]" for col in colnames}
            if dtype_backend == "pyarrow"
            else {col: "float32" if col.startswith("X_") else "int32" for col in colnames}
        )

        abundancedata = pd.read_csv(
            at.zopen(abundancefilepath),
            delim_whitespace=True,
            header=None,
            comment="#",
            names=colnames,
            dtype=dtypes,
            dtype_backend=dtype_backend,
        )

        if len(abundancedata) > 1000:
            print(f"Saving {filenameparquet}")
            abundancedata.to_parquet(filenameparquet, compression="zstd")
            print("  Done.")

    abundancedata.index.name = "modelgridindex"
    if dtype_backend == "pyarrow":
        abundancedata.index = abundancedata.index.astype("int32[pyarrow]")

    return abundancedata


@typechecked
def save_initelemabundances(
    dfelabundances: pd.DataFrame,
    abundancefilename: Path | str,
    headercommentlines: t.Sequence[str] | None = None,
) -> None:
    """Save a DataFrame (same format as get_initelemabundances) to abundances.txt.
    columns must be:
        - inputcellid: integer index to match model.txt (starting from 1)
        - X_El: mass fraction of element with two-letter code 'El' (e.g., X_H, X_He, H_Li, ...).
    """
    timestart = time.perf_counter()
    if Path(abundancefilename).is_dir():
        abundancefilename = Path(abundancefilename) / "abundances.txt"
    dfelabundances["inputcellid"] = dfelabundances["inputcellid"].astype(int)
    atomic_numbers = [
        at.get_atomic_number(colname[2:]) for colname in dfelabundances.columns if colname.startswith("X_")
    ]
    elcolnames = [f"X_{at.get_elsymbol(Z)}" for Z in range(1, 1 + max(atomic_numbers))]

    # set missing elemental abundance columns to zero
    for col in elcolnames:
        if col not in dfelabundances.columns:
            dfelabundances[col] = 0.0

    with Path(abundancefilename).open("w", encoding="utf-8") as fabund:
        if headercommentlines is not None:
            fabund.write("\n".join([f"# {line}" for line in headercommentlines]) + "\n")
        for row in dfelabundances.itertuples(index=False):
            fabund.write(f" {row.inputcellid:6d} ")
            fabund.write(" ".join([f"{getattr(row, colname, 0.):.6e}" for colname in elcolnames]))
            fabund.write("\n")

    print(f"Saved {abundancefilename} (took {time.perf_counter() - timestart:.1f} seconds)")


def save_empty_abundance_file(ngrid: int, outputfilepath: str | Path = Path()) -> None:
    """Save dummy abundance file with only zeros."""
    if Path(outputfilepath).is_dir():
        outputfilepath = Path(outputfilepath) / "abundances.txt"

    Z_atomic = np.arange(1, 31)

    abundancedata: dict[str, t.Any] = {"cellid": range(1, ngrid + 1)}
    for atomic_number in Z_atomic:
        abundancedata[f"Z={atomic_number}"] = np.zeros(ngrid)

    # abundancedata['Z=28'] = np.ones(ngrid)

    dfabundances = pd.DataFrame(data=abundancedata).round(decimals=5)
    dfabundances.to_csv(outputfilepath, header=False, sep=" ", index=False)


def get_dfmodel_dimensions(dfmodel: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> int:
    if "pos_x_min" in dfmodel.columns:
        return 3

    return 2 if "pos_z_mid" in dfmodel.columns else 1


def sphericalaverage(
    dfmodel: pd.DataFrame,
    t_model_init_days: float,
    vmax: float,
    dfelabundances: pd.DataFrame | None = None,
    dfgridcontributions: pd.DataFrame | None = None,
    nradialbins: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convert 3D Cartesian grid model to 1D spherical."""
    t_model_init_seconds = t_model_init_days * 24 * 60 * 60
    xmax = vmax * t_model_init_seconds
    ngridpoints = len(dfmodel)
    ncoordgridx = round(ngridpoints ** (1.0 / 3.0))
    wid_init = 2 * xmax / ncoordgridx

    print(f"Spherically averaging 3D model with {ngridpoints} cells...")
    timestart = time.perf_counter()

    # dfmodel = dfmodel.query('rho > 0.').copy()
    dfmodel = dfmodel.copy()
    celldensity = dict(dfmodel[["inputcellid", "rho"]].itertuples(index=False))

    for ax in ["x", "y", "z"]:
        dfmodel[f"vel_{ax}_mid"] = (dfmodel[f"pos_{ax}_min"] + (0.5 * wid_init)) / t_model_init_seconds

    dfmodel["vel_r_mid"] = np.sqrt(dfmodel["vel_x_mid"] ** 2 + dfmodel["vel_y_mid"] ** 2 + dfmodel["vel_z_mid"] ** 2)
    # print(dfmodel)
    # print(dfelabundances)
    km_to_cm = 1e5
    if nradialbins is None:
        nradialbins = int(ncoordgridx / 2.0)
    velocity_bins = [vmax * n / nradialbins for n in range(nradialbins + 1)]  # cm/s
    outcells = []
    outcellabundances = []
    outgridcontributions = []
    includemissingcolexists = (
        dfgridcontributions is not None and "frac_of_cellmass_includemissing" in dfgridcontributions.columns
    )

    # cellidmap_3d_to_1d = {}
    highest_active_radialcellid = -1
    for radialcellid, (velocity_inner, velocity_outer) in enumerate(zip(velocity_bins[:-1], velocity_bins[1:]), 1):
        assert velocity_outer > velocity_inner
        matchedcells = dfmodel.query("vel_r_mid > @velocity_inner and vel_r_mid <= @velocity_outer")
        matchedcellrhosum = matchedcells.rho.sum()
        # cellidmap_3d_to_1d.update({cellid_3d: radialcellid for cellid_3d in matchedcells.inputcellid})

        if len(matchedcells) == 0:
            rhomean = 0.0
        else:
            shell_volume = (4 * math.pi / 3) * (
                (velocity_outer * t_model_init_seconds) ** 3 - (velocity_inner * t_model_init_seconds) ** 3
            )
            rhomean = matchedcellrhosum * wid_init**3 / shell_volume
            # volumecorrection = len(matchedcells) * wid_init ** 3 / shell_volume
            # print(radialcellid, volumecorrection)

            if rhomean > 0.0 and dfgridcontributions is not None:
                dfcellcont = dfgridcontributions.query("cellindex in @matchedcells.inputcellid.to_numpy()")

                for particleid, dfparticlecontribs in dfcellcont.groupby("particleid"):
                    frac_of_cellmass_avg = (
                        sum(
                            row.frac_of_cellmass * celldensity[row.cellindex]
                            for row in dfparticlecontribs.itertuples(index=False)
                        )
                        / matchedcellrhosum
                    )

                    contriboutrow = {
                        "particleid": particleid,
                        "cellindex": radialcellid,
                        "frac_of_cellmass": frac_of_cellmass_avg,
                    }

                    if includemissingcolexists:
                        frac_of_cellmass_includemissing_avg = (
                            sum(
                                row.frac_of_cellmass_includemissing * celldensity[row.cellindex]
                                for row in dfparticlecontribs.itertuples(index=False)
                            )
                            / matchedcellrhosum
                        )
                        contriboutrow["frac_of_cellmass_includemissing"] = frac_of_cellmass_includemissing_avg

                    outgridcontributions.append(contriboutrow)

        if rhomean > 0.0:
            highest_active_radialcellid = radialcellid
        logrho = math.log10(max(1e-99, rhomean))

        dictcell = {
            "inputcellid": radialcellid,
            "velocity_outer": velocity_outer / km_to_cm,
            "logrho": logrho,
        }

        for column in matchedcells.columns:
            if column.startswith("X_") or column in ["cellYe", "q"]:
                massfrac = np.dot(matchedcells[column], matchedcells.rho) / matchedcellrhosum if rhomean > 0.0 else 0.0
                dictcell[column] = massfrac

        outcells.append(dictcell)

        if dfelabundances is not None:
            abund_matchedcells = dfelabundances.loc[matchedcells.index] if rhomean > 0.0 else None
            dictcellabundances = {"inputcellid": radialcellid}
            for column in dfelabundances.columns:
                if column.startswith("X_"):
                    massfrac = (
                        np.dot(abund_matchedcells[column], matchedcells.rho) / matchedcellrhosum
                        if rhomean > 0.0
                        else 0.0
                    )
                    dictcellabundances[column] = massfrac

            outcellabundances.append(dictcellabundances)

    dfmodel1d = pd.DataFrame(outcells[:highest_active_radialcellid])

    dfabundances1d = pd.DataFrame(outcellabundances[:highest_active_radialcellid]) if outcellabundances else None

    dfgridcontributions1d = pd.DataFrame(outgridcontributions) if outgridcontributions else None
    print(f"  took {time.perf_counter() - timestart:.1f} seconds")

    return dfmodel1d, dfabundances1d, dfgridcontributions1d


def scale_model_to_time(
    dfmodel: pd.DataFrame,
    targetmodeltime_days: float,
    t_model_days: float | None = None,
    modelmeta: dict[str, t.Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, t.Any] | None]:
    """Homologously expand model to targetmodeltime_days, reducing density and adjusting position columns to match."""
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
