import argparse
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import re
import shutil
import tarfile
import time
import typing as t
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import polars.selectors as cs
import polars.testing as pltest
import pytest
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at
from artistools.constants import day_to_s
from artistools.inputmodel.core import CACHEVERSION
from artistools.inputmodel.core import CREATED_COMMENT_PREFIX
from artistools.inputmodel.core import get_standard_columns

pytestmark = pytest.mark.usefixtures("close_figures")


def get_empty_3d_model(
    ncoordgrid: int, vmax: float, t_model_init_days: float, includenico57: bool = False
) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
    """Return a zero-density 3D model of ncoordgrid^3 cells, and its metadata, ready to be filled in."""
    xmax = vmax * t_model_init_days * day_to_s

    modelmeta: dict[str, t.Any] = {
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

    dfmodel = (
        pl
        .DataFrame(
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
            (-xmax + 2.0 * pl.col("n_x") * xmax / ncoordgrid).cast(pl.Float32).alias("pos_x_min"),
            (-xmax + 2.0 * pl.col("n_y") * xmax / ncoordgrid).cast(pl.Float32).alias("pos_y_min"),
            (-xmax + 2.0 * pl.col("n_z") * xmax / ncoordgrid).cast(pl.Float32).alias("pos_z_min"),
        ])
    )

    standardcols = get_standard_columns("cartesian3d", includenico57=includenico57)

    dfmodel = dfmodel.with_columns([
        pl.lit(0.0, dtype=pl.Float32).alias(colname)
        for colname in standardcols
        if colname not in dfmodel.collect_schema().names()
    ]).select([*standardcols, "modelgridindex"])

    return dfmodel, modelmeta


modelpath = at.get_path("testdata") / "testmodel"
modelpath_3d = at.get_path("testdata") / "testmodel_3d_10^3"
outputpath = at.get_path("testoutput")
testdatapath = at.get_path("testdata")
modelpath_classic_3d = testdatapath / "test-classicmode_3d"


def get_derived_modeldata(modelpath: Path, **kwargs: t.Any) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
    """Return the model with the derived columns, e.g. the mass of each cell, and the metadata."""
    lzdfmodel, modelmeta = at.inputmodel.scan_modeldata(modelpath, **kwargs)
    return at.inputmodel.add_derived_cols_to_modeldata(lzdfmodel, modelmeta=modelmeta), modelmeta


def test_trajectory_archive_formats_give_the_same_table(
    tmp_path: Path, write_trajectory_archive: Callable[[Path, int, dict[str, str], str], Path]
) -> None:
    """A .tar.zst, a .tar.xz, and a .tar give the same table, and a .tar.zst comes before a .tar.xz."""
    members = {
        "energy_thermo.dat": "#count time/s Qdot\n     1  1.000000E+00  1.500000E+21\n",
        "nz-plane00001": "  1  1.000000E+00   1.0E-06   2.0E-12   0.0E+00   6.3E-06\n  2   2 -1.27404E+00  0.0  0.0\n",
    }
    tables = {
        suffix: at.inputmodel.rprocess_from_trajectory.read_trajectory_network_data(
            write_trajectory_archive(tmp_path / suffix.lstrip("."), 3, members, suffix)
        )[0]
        for suffix in (".tar", ".tar.zst", ".tar.xz")
    }
    pltest.assert_frame_equal(tables[".tar.zst"], tables[".tar.xz"])
    pltest.assert_frame_equal(tables[".tar"], tables[".tar.xz"])

    write_trajectory_archive(tmp_path / "both", 3, members, ".tar.xz")
    write_trajectory_archive(tmp_path / "both", 3, members, ".tar.zst")
    archivepath = at.inputmodel.rprocess_from_trajectory.get_trajectory_archive_path(tmp_path / "both", 3)
    assert archivepath is not None
    assert archivepath.name == "3.tar.zst"


def test_trajectory_qdotintegral_is_zero_before_one_second(
    tmp_path: Path, write_trajectory_archive: Callable[[Path, int, dict[str, str], str], Path]
) -> None:
    """A last timestep before one second gives a zero integral, because the integration starts at one second.

    The filter on the last timestep came before the check for a time of one second or more, thus the check
    failed with an incorrect message.
    """
    write_trajectory_archive(
        tmp_path,
        6,
        {
            "energy_thermo.dat": (
                "#count time/s Qdot\n"
                "     1  5.000000E-01  1.000000E+20\n"
                "     2  2.000000E+00  2.000000E+10\n"
                "     3  4.000000E+00  1.000000E+10\n"
            )
        },
        ".tar.xz",
    )
    qdotintegral = at.inputmodel.rprocess_from_trajectory.get_trajectory_qdotintegral
    assert qdotintegral(6, tmp_path, nts_max=1, t_model_s=4.0) == 0
    # Qdot * t / t_model is 1e10 at both timesteps, thus the trapezoid over two seconds gives 2e10
    assert qdotintegral(6, tmp_path, nts_max=3, t_model_s=4.0) == pytest.approx(2.0e10)

    # a corrupt Qdot value makes energy_thermo.dat corrupt, and the cache leaves out the table, thus the particle
    # has no network data for a reader of Qdot
    write_trajectory_archive(
        tmp_path / "corrupt",
        6,
        {
            "energy_thermo.dat": (
                "#count time/s Qdot\n"
                "     1  2.000000E+00  2.000000E+10\n"
                "     2  3.000000E+00  **********\n"
                "     3  4.000000E+00  1.000000E+10\n"
            )
        },
        ".tar.xz",
    )
    with pytest.raises(FileNotFoundError, match="no column Qdot"):
        qdotintegral(6, tmp_path / "corrupt", nts_max=3, t_model_s=4.0)

    # a missing timestep in the range of the integral stops the integral, because a trapezoid across the gap
    # would give a wrong energy with no sign of the problem
    write_trajectory_archive(
        tmp_path / "gap",
        6,
        {
            "energy_thermo.dat": (
                "#count time/s Qdot\n     1  2.000000E+00  2.000000E+10\n     3  4.000000E+00  1.000000E+10\n"
            )
        },
        ".tar.xz",
    )
    with pytest.raises(ValueError, match="no Qdot value at each timestep up to 3"):
        qdotintegral(6, tmp_path / "gap", nts_max=3, t_model_s=4.0)
    # the missing timestep is the last timestep of the integral, thus the timesteps before it are consecutive
    with pytest.raises(ValueError, match="no Qdot value at each timestep up to 2"):
        qdotintegral(6, tmp_path / "gap", nts_max=2, t_model_s=4.0)
    assert qdotintegral(6, tmp_path / "gap", nts_max=1, t_model_s=4.0) == pytest.approx(0.0)

    # the last timestep has an abundance file and no row in energy_thermo.dat, thus the integral would stop early
    write_trajectory_archive(
        tmp_path / "short",
        6,
        {
            "energy_thermo.dat": (
                "#count time/s Qdot\n     1  2.000000E+00  2.000000E+10\n     2  3.000000E+00  2.000000E+10\n"
            ),
            "nz-plane00003": (
                "  3  4.000000E+00   1.0E-06   2.0E-12   0.0E+00   6.3E-06\n  2   2 -1.27404E+00  0.0  0.0\n"
            ),
        },
        ".tar.xz",
    )
    # the trajectory has no network data at that timestep, thus the particle is one with no network data
    with pytest.raises(FileNotFoundError, match="ends at timestep 2 before timestep 3"):
        qdotintegral(6, tmp_path / "short", nts_max=3, t_model_s=4.0)


def trajectory_step_count(traj_root: Path, _worker: int) -> int:
    """Return the number of rows that one process reads for a trajectory, after it builds the cache if necessary."""
    # a worker keeps the timesteps of the previous trial for the same archive, thus the test clears them
    at.inputmodel.rprocess_from_trajectory.collect_trajectory_timesteps.cache_clear()
    return at.inputmodel.rprocess_from_trajectory.get_trajectory_timesteps(traj_root, 114511).height


def test_trajectory_cache_write_is_atomic(tmp_path: Path) -> None:
    """Processes that build the cache of the same trajectory at the same time must all read a complete table.

    The write goes through a temporary file, thus a reader sees either no cache or a complete one. If a different
    process wrote a complete cache first, the function keeps that cache.
    """
    nworkers = 16
    traj_root = tmp_path / "trajectories"
    # a group that can write the trajectory root must be able to add a cache, thus the cache folder takes this mode
    traj_root.mkdir()
    traj_root.chmod(0o775)
    shutil.copy(testdatapath / "kilonova" / "trajectories" / "114511.tar.xz", traj_root)
    nrows = at.inputmodel.rprocess_from_trajectory.read_trajectory_network_data(traj_root / "114511.tar.xz")[0].height

    readrows = partial(trajectory_step_count, traj_root)
    # at.misc.parallel_map would give a thread pool on free-threaded builds, and only separate processes can race on
    # the filesystem. Spawn for the same reason parallel_map does: forking with polars/rayon threads live is unsafe.
    with ProcessPoolExecutor(max_workers=nworkers, mp_context=multiprocessing.get_context("spawn")) as executor:
        # start the workers before the trials, because then each trial tests the cache write and not the process start
        list(executor.map(time.sleep, [0.0] * nworkers))

        for _trial in range(20):
            shutil.rmtree(traj_root / "parquetcaches", ignore_errors=True)
            assert list(executor.map(readrows, range(nworkers), chunksize=1)) == [nrows] * nworkers

    assert (traj_root / "parquetcaches" / "114511.parquet").is_file()
    assert (traj_root / "parquetcaches").stat().st_mode & 0o777 == 0o775


def test_trajectory_cache_replaces_a_stale_cache_and_outlives_its_archive(tmp_path: Path) -> None:
    """The read rebuilds a cache from a different artistools version, and it uses a cache whose archive is absent.

    A user can delete the archives after artistools reads them one time, because the cache then holds all the data
    that artistools reads.
    """
    from artistools.inputmodel.rprocess_from_trajectory import scan_trajectory_network_data
    from artistools.inputmodel.rprocess_from_trajectory import TRAJECTORY_CACHEVERSION

    traj_root = tmp_path / "trajectories"
    traj_root.mkdir()
    with pytest.raises(FileNotFoundError, match="114511"):
        scan_trajectory_network_data(traj_root, 114511)

    archivepath = shutil.copy(testdatapath / "kilonova" / "trajectories" / "114511.tar.xz", traj_root)
    parquetpath = traj_root / "parquetcaches" / "114511.parquet"
    parquetpath.parent.mkdir()
    pl.DataFrame({"nstep": [1]}).write_parquet(
        parquetpath, metadata={"cacheversion": "0", "textsource_mtime": str(Path(archivepath).stat().st_mtime)}
    )

    dftrajectory = scan_trajectory_network_data(traj_root, 114511).collect()
    assert dftrajectory.height > 300
    pqmetadata = pl.read_parquet_metadata(parquetpath)
    assert pqmetadata["cacheversion"] == str(TRAJECTORY_CACHEVERSION)

    assert pqmetadata["textsource"] == "114511.tar.xz"

    # a copy with cp -p keeps the modification time, thus the size of the archive shows the change

    archivetime = Path(archivepath).stat().st_mtime
    shutil.copy(testdatapath / "kilonova" / "trajectories" / "109215.tar.xz", archivepath)
    os.utime(archivepath, (archivetime, archivetime))
    dfreplaced = scan_trajectory_network_data(traj_root, 114511).collect()
    assert dfreplaced.height != dftrajectory.height
    assert dfreplaced["abundancefile"].drop_nulls().to_list() == ["nz-plane00434"]

    Path(archivepath).unlink()
    pltest.assert_frame_equal(scan_trajectory_network_data(traj_root, 114511).collect(), dfreplaced)


def test_describeinputmodel(capsys: pytest.CaptureFixture[str]) -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=[modelpath], isotopes=True)

    output = capsys.readouterr().out
    assert "Model contains 1 1D spherical shells" in output
    # the mass of the one cell, 1.416963e33 g, is 0.71240 Msun
    assert "M_tot_rho           0.71240 MSun" in output


@pytest.mark.benchmark
def test_describeinputmodel_3d(capsys: pytest.CaptureFixture[str]) -> None:
    at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=modelpath_3d, isotopes=True)

    output = capsys.readouterr().out
    assert "Model contains 1000 grid cells (695 nonempty)" in output
    assert "M_tot_rho" in output


def test_scan_modeldata_1d() -> None:
    lzdfmodel, modelmeta = get_derived_modeldata(modelpath)
    assert math.isclose(modelmeta["t_model_init_days"], 0.00115740740741, rel_tol=0.0001)
    assert math.isclose(modelmeta["vmax_cmps"], 800000000.0)
    assert modelmeta["dimensions"] == 1
    assert modelmeta["npts_model"] == 1

    assert math.isclose(lzdfmodel.select(pl.col("mass_g").sum()).collect().item(), 1.416963e33, rel_tol=1e-05)


@pytest.mark.benchmark
def test_scan_modeldata_3d() -> None:
    lzdfmodel, modelmeta = get_derived_modeldata(modelpath_3d)
    assert math.isclose(modelmeta["vmax_cmps"], 2892020000.0)
    assert modelmeta["dimensions"] == 3
    assert modelmeta["npts_model"] == 1000
    assert modelmeta["ncoordgridx"] == 10

    assert math.isclose(lzdfmodel.select(pl.col("mass_g").sum()).collect().item(), 2.7861855e33, rel_tol=1e-05)


def test_get_spatial_scales() -> None:
    # the inner shell of this 1D model is 1259 km/s wide and the other 69 shells are 315 km/s wide
    smallest, largest, description = at.inputmodel.get_spatial_scales(testdatapath / "test-classicmode_1d")
    assert math.isclose(smallest, 314.84e5, rel_tol=1e-4)
    assert math.isclose(largest, 1259.375e5, rel_tol=1e-4)
    assert "the width of the narrowest shell, 315 km/s" in description
    # the grid has 10 cells on each axis from -vmax to vmax, thus the diagonal of a cell is sqrt(3) times its width
    smallest, largest, description = at.inputmodel.get_spatial_scales(modelpath_3d)
    assert math.isclose(smallest, 2 * 2892020000.0 / 10)
    assert math.isclose(largest, math.sqrt(3) * smallest)
    assert "the diagonal of a cell" in description


def test_get_spatial_scales_follows_the_working_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default model path "." gives the model of the current working folder.

    The cache held the relative path, thus the scales of the first folder stayed after a change of the working folder.
    """
    monkeypatch.chdir(testdatapath / "test-classicmode_1d")
    smallest_1d, _, _ = at.inputmodel.get_spatial_scales(Path())
    monkeypatch.chdir(modelpath_3d)
    smallest_3d, _, description_3d = at.inputmodel.get_spatial_scales(Path())

    assert math.isclose(smallest_1d, 314.84e5, rel_tol=1e-4)
    assert math.isclose(smallest_3d, 2 * 2892020000.0 / 10)
    assert "the diagonal of a cell" in description_3d


@pytest.mark.parametrize("cachecontents", [b"", b"not a parquet file", b"PAR1" + bytes(64)])
def test_scan_modeldata_replaces_unreadable_cache(tmp_path: Path, cachecontents: bytes) -> None:
    """A damaged parquet cache must be deleted and rebuilt, not raise from read_parquet_metadata."""
    shutil.copy(modelpath / "model.txt", tmp_path)
    cachefilepath = tmp_path / "model.txt.parquet.tmp"
    cachefilepath.write_bytes(cachecontents)

    lzdfmodel, modelmeta = at.scan_modeldata(modelpath=tmp_path)
    assert modelmeta["npts_model"] == 1

    # the rebuilt cache is written even though the text model is far under the 2 MiB threshold
    cachemetadata = pl.read_parquet_metadata(cachefilepath)
    assert cachemetadata["cacheversion"] == str(CACHEVERSION)
    assert cachemetadata["textsource_mtime"] == str((tmp_path / "model.txt").stat().st_mtime)
    pltest.assert_frame_equal(lzdfmodel.collect(), at.scan_modeldata(modelpath=tmp_path)[0].collect())


def test_scan_modeldata_rejects_cache_without_metadata(tmp_path: Path) -> None:
    """A readable parquet file that is missing the artistools metadata keys must not be trusted."""
    shutil.copy(modelpath / "model.txt", tmp_path)
    textfilepath = tmp_path / "model.txt"
    cachefilepath = tmp_path / "model.txt.parquet.tmp"
    # the version and the mtime match, thus the reader rejects the cache for the absent modelmeta_json alone
    at.misc.write_parquet_atomic(
        pl.DataFrame({"inputcellid": [1]}),
        cachefilepath,
        metadata={"cacheversion": str(CACHEVERSION), "textsource_mtime": str(textfilepath.stat().st_mtime)},
    )

    _, modelmeta = at.scan_modeldata(modelpath=tmp_path)
    assert modelmeta["npts_model"] == 1
    assert "modelmeta_json" in pl.read_parquet_metadata(cachefilepath)


def test_scan_modeldata_refreshes_stale_cache(tmp_path: Path) -> None:
    """A cache whose recorded text-source mtime no longer matches must be rewritten from the text source."""
    shutil.copy(modelpath / "model.txt", tmp_path)
    textfilepath = tmp_path / "model.txt"
    cachefilepath = tmp_path / "model.txt.parquet.tmp"
    lzdfmodel, modelmeta = at.scan_modeldata(modelpath=tmp_path)
    # the version matches, thus the reader rejects the cache for the stale mtime alone
    at.misc.write_parquet_atomic(
        lzdfmodel.collect(),
        cachefilepath,
        metadata={"cacheversion": str(CACHEVERSION), "textsource_mtime": "0", "modelmeta_json": json.dumps(modelmeta)},
    )

    assert at.scan_modeldata(modelpath=tmp_path)[1]["npts_model"] == 1
    assert pl.read_parquet_metadata(cachefilepath)["textsource_mtime"] == str(textfilepath.stat().st_mtime)


def test_downscale_3dmodel(tmp_path: Path) -> None:
    lzdfmodel, modelmeta = get_derived_modeldata(modelpath_3d, get_elemabundances=True)
    modelpath_3d_small = at.inputmodel.downscale3dgrid.make_downscaled_3d_grid(
        modelpath_3d, outputgridsize=2, outputfolder=tmp_path
    )
    dfmodel = lzdfmodel.collect()
    lzdfmodel_small, modelmeta_small = get_derived_modeldata(modelpath_3d_small, get_elemabundances=True)
    dfmodel_small = lzdfmodel_small.collect()
    assert math.isclose(dfmodel["mass_g"].sum(), dfmodel_small["mass_g"].sum(), rel_tol=1e-5)
    assert math.isclose(modelmeta["vmax_cmps"], modelmeta_small["vmax_cmps"], rel_tol=1e-5)
    assert math.isclose(modelmeta["t_model_init_days"], modelmeta_small["t_model_init_days"], rel_tol=1e-5)

    abundcols = (x for x in dfmodel.columns if x.startswith("X_"))
    for abundcol in abundcols:
        assert math.isclose(
            (dfmodel[abundcol] * dfmodel["mass_g"]).sum(),
            (dfmodel_small[abundcol] * dfmodel_small["mass_g"]).sum(),
            rel_tol=1e-5,
        )


def decompress_file(zstdpath: Path) -> Path:
    """Replace a zstd file with its uncompressed copy, e.g. model.txt for model.txt.zst, and return its path."""
    uncompressedpath = zstdpath.with_suffix("")
    with at.zopen(zstdpath, encoding="utf-8") as fzst:
        uncompressedpath.write_text(fzst.read(), encoding="utf-8")
    zstdpath.unlink()
    return uncompressedpath


def verify_file_checksums(
    checksums_expected: dict[t.Any, t.Any], digest: str = "sha256", folder: Path | str = "."
) -> None:
    checksums_actual: dict[Path, str] = {}

    createdprefix = f"# {CREATED_COMMENT_PREFIX}"
    for filename, checksum_expected in checksums_expected.items():
        fullpath = Path(folder) / filename
        m = hashlib.new(digest)
        # the checksum uses the decompressed text, because two zstd libraries can make different bytes from one text
        with at.zopen(fullpath, encoding="utf-8") as f:
            for line in f:
                # the creation time is different in each run
                if not line.startswith(createdprefix):
                    m.update(line.encode())

        checksums_actual[fullpath] = m.hexdigest()
        strpassfail = "pass" if checksums_actual[fullpath] == checksum_expected else "FAILED"
        print(f"{filename}: {strpassfail} expected {checksum_expected} actual {checksums_actual[fullpath]}")

    for filename, checksum_expected in checksums_expected.items():
        fullpath = Path(folder) / filename
        assert checksums_actual[fullpath] == checksum_expected, (
            f"{folder}/{filename} checksum mismatch. Expecting {checksum_expected}"
            f" but calculated {checksums_actual[fullpath]}"
        )


def test_makeartismodelfrom_sph_particles(tmp_path: Path, trajectory_copy: Path) -> None:
    gridfolderpath = tmp_path / "kilonova"

    config: dict[str, dict[str, t.Any]] = {
        "maptogridargs": {"ncoordgrid": 16},
        "maptogrid_sums": {
            "ejectapartanalysis.dat": "e8694a679515c54c2b4867122122263a375d9ffa144a77310873ea053bb5a8b4",
            "grid.dat": "2e4ff3708d8e0703fb50513386ead4b5549116895d29bf927a37de6a84a7760c",
            "gridcontributions.txt.zst": "dbe2427f88f5f8a8490b3a6523a24ef30cdb1def0b0bd5a1e5b8e69e8b227891",
        },
        # the model and abundance files carry eight significant figures, and the vmax header nine, so
        # that a model round-trips through the Float32 of the reader.
        "makeartismodel_sums": {
            "gridcontributions.txt.zst": "b3de0a54d97c45421d204d6b24e0bfd6400a00a9810df9c520f52a52d235c734",
            "abundances.txt.zst": "e931d575bfbe2b442fad45a09d5f2acf68306f36e93966c3ac797200f52cf7bb",
            "model.txt.zst": "4e6884143fbfcfe7e99731ca93a1e92b4270836c0db2d1012ce49dc0872b0bcf",
        },
    }

    shutil.copytree(
        testdatapath / "kilonova", gridfolderpath, dirs_exist_ok=True, ignore=shutil.ignore_patterns("trajectories")
    )

    at.inputmodel.maptogrid.main(
        argsraw=[], inputpath=gridfolderpath, outputpath=gridfolderpath, force=True, **config["maptogridargs"]
    )

    verify_file_checksums(config["maptogrid_sums"], digest="sha256", folder=gridfolderpath)

    dfcontribs = {}
    for dimensions in (3, 2, 1, 0):
        outpath_kn = tmp_path / f"kilonova_{dimensions:d}d"
        outpath_kn.mkdir(exist_ok=True, parents=True)

        at.inputmodel.modelfromhydro.main(
            argsraw=[],
            gridfolderpath=gridfolderpath,
            trajectoryroot=trajectory_copy,
            outputpath=outpath_kn,
            dimensions=dimensions,
            timedays=0.1,
        )

        dfcontribs[dimensions] = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(outpath_kn)

        if dimensions == 3:
            verify_file_checksums(config["makeartismodel_sums"], digest="sha256", folder=outpath_kn)
            dfcontrib_source = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(gridfolderpath)

            pltest.assert_frame_equal(
                dfcontrib_source,
                dfcontribs[3].drop("frac_of_cellmass").rename({"frac_of_cellmass_includemissing": "frac_of_cellmass"}),
                rel_tol=1e-4,
                abs_tol=1e-4,
            )
        else:
            dfmodel3lz, _ = get_derived_modeldata(tmp_path / f"kilonova_{3:d}d")
            dfmodel3 = dfmodel3lz.collect()
            dfmodel_lowerdlz, _ = get_derived_modeldata(tmp_path / f"kilonova_{dimensions:d}d")
            dfmodel_lowerd = dfmodel_lowerdlz.collect()

            # check that the total mass is conserved
            assert math.isclose(dfmodel_lowerd["mass_g"].sum(), dfmodel3["mass_g"].sum(), rel_tol=5e-2)
            assert math.isclose(dfmodel_lowerd["tracercount"].sum(), dfmodel3["tracercount"].sum(), rel_tol=1e-1)

            # check that the total mass of each species and total internal energy are conserved
            for col in dfmodel3.columns:
                if col.startswith("X_") or col == "q":
                    lowerd_mass = dfmodel_lowerd.select(pl.col("mass_g").dot(pl.col(col))).item()
                    model3_mass = dfmodel3.select(pl.col("mass_g").dot(pl.col(col))).item()
                    assert math.isclose(lowerd_mass, model3_mass, rel_tol=5e-2)


def test_lower_corner_sample_makes_the_model_of_an_older_version_again(tmp_path: Path, trajectory_copy: Path) -> None:
    """The option --sample_cell_lower_corner of maptogrid gives the grid files and the model of the older version.

    The expected checksums are those that the test of the SPH particles expected before the kernel took the
    centre of each cell. A user can thus make a historical model again from the same particles.
    """
    gridfolderpath = tmp_path / "kilonova"
    shutil.copytree(
        testdatapath / "kilonova", gridfolderpath, dirs_exist_ok=True, ignore=shutil.ignore_patterns("trajectories")
    )
    at.inputmodel.maptogrid.main(
        argsraw=[],
        inputpath=gridfolderpath,
        outputpath=gridfolderpath,
        ncoordgrid=16,
        sample_cell_lower_corner=True,
        force=True,
    )
    verify_file_checksums(
        {
            "grid.dat": "d7dbe63efe3544f5d6f77acc202e110e197b02dcfa953d8f2bf84b24d9b8e76d",
            "gridcontributions.txt.zst": "63e6331666c4928bdc6b7d0f59165e96d6555736243ea8998a779519052a425f",
        },
        digest="sha256",
        folder=gridfolderpath,
    )

    outpath = tmp_path / "kilonova_3d"
    outpath.mkdir()
    at.inputmodel.modelfromhydro.main(
        argsraw=[],
        gridfolderpath=gridfolderpath,
        trajectoryroot=trajectory_copy,
        outputpath=outpath,
        dimensions=3,
        timedays=0.1,
    )
    verify_file_checksums(
        {
            "gridcontributions.txt.zst": "f7ddda0c8789a642ad2399e2ae67acc15e2fac519bbddfcdaa65b93d32e3edeb",
            "abundances.txt.zst": "fb8b4f7c81e6b223ec9506d625cfc78cb778ad2056b8143078d7bfeb9451c1d2",
            "model.txt.zst": "e92e6f54d3e494df42c56213a9778a4594c65f370d6f1109975f4f6470627a12",
        },
        digest="sha256",
        folder=outpath,
    )


@pytest.mark.benchmark
def test_makeartismodelfrom_fortrangriddat(tmp_path: Path) -> None:
    gridfolderpath = testdatapath / "kilonova"
    outpath_kn = tmp_path / "kilonova_fromfortrangriddat"
    # CodSpeed runs the body of a benchmark more than once in the same tmp_path, thus the command overwrites its files
    at.inputmodel.modelfromhydro.main(
        argsraw=[], gridfolderpath=gridfolderpath, outputpath=outpath_kn, dimensions=3, timedays=0.1, force=True
    )

    # grid.dat gives 8 cells, and the command expands the grid to the time of the snapshot
    _, modelmeta = at.inputmodel.scan_modeldata(outpath_kn, printwarningsonly=True)
    assert modelmeta["dimensions"] == 3
    assert modelmeta["npts_model"] == 8
    assert np.isclose(modelmeta["t_model_init_days"], 0.1)
    # grid.dat names the electron fraction cellYe, and the command writes Ye.txt only for a column Ye
    assert (outpath_kn / "Ye.txt").is_file()


def test_make1dmodelfromcone(tmp_path: Path) -> None:
    at.inputmodel.slice1dfromconein3dmodel.main(
        argsraw=[],
        modelpath=[modelpath_3d],
        outputpath=tmp_path,
        axis="-z",
        coneshellspacingexponent=2.0,
        nshells=4,
        coneangle=60,
    )

    dfmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path / "model_1d.txt", printwarningsonly=True)
    assert modelmeta["dimensions"] == 1
    assert 1 <= modelmeta["npts_model"] <= 4
    assert dfmodel.select(pl.col("logrho").max()).collect().item() > -90.0
    assert (tmp_path / "abundances_1d.txt.zst").is_file()


def test_make1dmodelfromcone_of_a_version_2_model_gives_the_mass_fractions_of_version_1(tmp_path: Path) -> None:
    """A version 2 model holds each X_<el>_otherstable beside an element column that already includes it.

    The sum of the normalisation counted the otherstable columns twice, thus each mass fraction was too low.
    """
    version2folder = tmp_path / "version2"
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[modelpath_3d], format_version=2, outputfile=version2folder)
    massfractions = []
    for sourcefolder in (modelpath_3d, version2folder):
        conefolder = tmp_path / f"cone_{sourcefolder.name}"
        conefolder.mkdir()
        at.inputmodel.slice1dfromconein3dmodel.main(
            argsraw=[],
            modelpath=[sourcefolder],
            outputpath=conefolder,
            axis="-z",
            coneshellspacingexponent=2.0,
            nshells=4,
            coneangle=60,
        )
        dfmodel, _ = at.inputmodel.scan_modeldata(conefolder / "model_1d.txt", printwarningsonly=True)
        with at.zopen(conefolder / "abundances_1d.txt.zst") as abundfile:
            elementfractions = np.loadtxt(abundfile)[:, 1:]
        massfractions.append((dfmodel.select(cs.starts_with("X_")).collect(), elementfractions))

    (nuclides_v1, elements_v1), (nuclides_v2, elements_v2) = massfractions
    assert nuclides_v1.columns == nuclides_v2.columns
    for colname in nuclides_v1.columns:
        assert np.allclose(nuclides_v2[colname], nuclides_v1[colname], rtol=1e-6, atol=0.0), colname
    assert np.allclose(elements_v2, elements_v1, rtol=1e-6, atol=0.0)
    assert np.allclose(elements_v2.sum(axis=1), 1.0, rtol=1e-6, atol=0.0)


def test_empty_shell_warning_goes_to_the_standard_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--quiet hides the standard output alone, thus the warning for an empty shell takes the standard error.

    The command removes the empty shell and every shell outside it. A run with --quiet gave the new
    model and no warning, because the warning went to the standard output that --quiet hides.
    """
    dfmodel, modelmeta = get_derived_modeldata(modelpath_3d)
    dfmodel = dfmodel.collect()
    vmax = dfmodel["vel_r_mid"].max()
    assert isinstance(vmax, float)

    # empty the outer part of the model, thus every cell of an outer shell has a density of zero
    dfmodel = dfmodel.with_columns(
        pl.when(pl.col("vel_r_mid") > 0.4 * vmax).then(pl.lit(0.0)).otherwise(pl.col("rho")).alias("rho")
    )
    at.inputmodel.save_modeldata(dfmodel=dfmodel, outpath=tmp_path, modelmeta=modelmeta)
    shutil.copy(modelpath_3d / "abundances.txt.xz", tmp_path / "abundances.txt.xz")

    outputpath_cone = tmp_path / "out"
    outputpath_cone.mkdir()
    at.inputmodel.slice1dfromconein3dmodel.main(
        argsraw=[],
        modelpath=[tmp_path],
        outputpath=outputpath_cone,
        axis="-z",
        coneshellspacingexponent=2.0,
        nshells=4,
        coneangle=60,
    )

    captured = capsys.readouterr()
    assert "warning: Shell" in captured.err
    assert "warning: Shell" not in captured.out
    logtext = "".join(logfile.read_text(encoding="utf-8") for logfile in outputpath_cone.rglob("make1dmodellog.txt"))
    assert "warning: Shell" in logtext, "the log file must keep the warning"


def test_makemodel_energyfiles(tmp_path: Path) -> None:
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=modelpath, makeenergyinputfiles=True, outputpath=tmp_path)

    assert (tmp_path / "energydistribution.txt").is_file()
    assert (tmp_path / "energyrate.txt").is_file()


def test_makemodel_dimensionreduce_writes_the_gridcontributions(tmp_path: Path) -> None:
    """-dimensionreduce writes the particle contributions of the reduced grid as makeartismodelfromparticlegridmap does.

    The command discarded the remapped contributions, thus the reduced model folder held no gridcontributions.txt.
    """
    modelfolder = tmp_path / "model3d"
    modelfolder.mkdir()
    for filename in ("model.txt.xz", "abundances.txt.xz"):
        shutil.copy(modelpath_3d / filename, modelfolder)
    dfmodel = at.inputmodel.scan_modeldata(modelfolder, printwarningsonly=True)[0].collect()
    nonemptyids = dfmodel.filter(pl.col("rho") > 0)["inputcellid"].to_list()
    (modelfolder / "gridcontributions.txt").write_text(
        "particleid cellindex frac_of_cellmass\n" + "".join(f"7 {cellid} 1.0\n" for cellid in nonemptyids),
        encoding="utf-8",
    )

    outfolder = tmp_path / "out"
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[modelfolder], dimensionreduce=1, outputpath=outfolder)

    reducedfolder = outfolder / "model3d_dimreduce_1d"
    dfcontribs = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(reducedfolder)
    _, modelmeta = at.inputmodel.scan_modeldata(reducedfolder, printwarningsonly=True)
    assert set(dfcontribs["particleid"].to_list()) == {7}
    assert dfcontribs["cellindex"].max() <= modelmeta["npts_model"]


def test_opacity_by_ye_file(tmp_path: Path) -> None:
    griddata = pl.DataFrame({
        "Ye": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5],
        "rho": [0.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
        "inputcellid": range(1, 9),
    })
    at.inputmodel.opacityinputfile.opacity_by_ye(tmp_path, griddata=griddata)

    # Table 1 of Tanaka et al. (2020), and zero for the empty cell
    opacities = np.loadtxt(tmp_path / "opacity.txt", skiprows=1)[:, 1]
    assert np.allclose(opacities, [0.0, 19.5, 32.2, 22.3, 5.6, 5.36, 3.3, 0.96])


def test_opacity_by_integer_ye(tmp_path: Path) -> None:
    griddata = pl.DataFrame({"Ye": [0, 1], "rho": [1.0, 1.0], "inputcellid": [1, 2]})
    at.inputmodel.opacityinputfile.opacity_by_ye(tmp_path, griddata=griddata)
    assert np.allclose(np.loadtxt(tmp_path / "opacity.txt", skiprows=1)[:, 1], [19.5, 0.96])


def test_get_elemabund_from_nucabund() -> None:
    dfnucabund = pl.DataFrame({
        "Z": [26, 26, 28, 28, 1, 1],  # Fe, Fe, Ni, Ni, H, H
        "N": [30, 32, 30, 32, 0, 1],  # Corresponds to A = 56, 58, 58, 60, 1, 2
        "massfrac": [0.1, 0.2, 0.3, 0.4, 0.05, 0.05],
    })

    expected_abunds = {"X_H": 0.05 + 0.05, "X_Fe": 0.1 + 0.2, "X_Ni": 0.3 + 0.4}

    # Determine the maximum Z in the input to generate all element symbols up to it
    max_z = dfnucabund["Z"].max()
    assert isinstance(max_z, int)

    # Populate expected_abunds with all elements up to max_Z, defaulting to 0.0
    all_expected_abunds = {}
    for z_val in range(1, int(max_z) + 1):
        el_sym = at.get_elsymbol(z_val)
        all_expected_abunds[f"X_{el_sym}"] = expected_abunds.get(f"X_{el_sym}", 0.0)

    result_abunds = at.inputmodel.rprocess_from_trajectory.get_elemabund_from_nucabund(dfnucabund)

    assert result_abunds == all_expected_abunds


def test_trajectory_archive_joins_the_timestep_tables_and_the_named_abundance_file(
    tmp_path: Path, write_trajectory_archive: Callable[[Path, int, dict[str, str], str], Path]
) -> None:
    """A row holds energy_thermo.dat and heating.dat for its timestep, and a named abundance file is a row of its own.

    The named file tday_nz-plane has the timestep number of nz-plane00002 in its header, but its time differs. A
    full join on the timestep must keep both rows, and the duplicate time column of heating.dat must not replace
    the time of energy_thermo.dat.
    """
    nuclidelines = "  0   1 -1.11798E+01  0.00000E+00  0.00000E+00\n  2   2 -1.27404E+00  2.05770E+01  0.00000E+00\n"
    archivepath = write_trajectory_archive(
        tmp_path,
        7,
        {
            "energy_thermo.dat": (
                "#count       time/s            T9     rho(g/cc)            Ye          Qdot\n"
                "     1  1.000000E+00  1.000000E+01  2.000000E+07  4.600000E-01  1.500000E+21\n"
                "     2  9.000000E+04  1.000000E-06  2.000000E-12  4.600000E-01  2.000000E+10\n"
            ),
            # a restart of the network writes the header again and writes timestep 2 again, and it adds a timestep
            # that energy_thermo.dat does not have
            "heating.dat": (
                "#count       time/s            T9     rho(g/cc)         hbeta          htot\n"
                "     1  1.000000E+00  1.000000E+01  2.000000E+07  1.842623-281  3.000000E+14\n"
                "     2  9.000000E+04  1.000000E-06  2.000000E-12  1.000000E+00  1.000000E+00\n"
                "#count       time/s            T9     rho(g/cc)         hbeta          htot\n"
                "     2  9.000000E+04  1.000000E-06  2.000000E-12  4.000000E+09  5.000000E+09\n"
                "     3  1.000000E+05  1.000000E-06  2.000000E-12  6.000000E+09  7.000000E+09\n"
            ),
            "nz-plane00002": "  2  9.000000E+04   1.0E-06   2.0E-12   0.0E+00   6.3E-06\n" + nuclidelines,
            "tday_nz-plane": "  2  8.640000E+04   1.0E-06   2.0E-12   0.0E+00   6.3E-06\n" + nuclidelines,
            "._nz-plane00002": "resource fork\n",
        },
        ".tar.xz",
    )

    dftrajectory, leftoutfiles = at.inputmodel.rprocess_from_trajectory.read_trajectory_network_data(archivepath)

    assert leftoutfiles == []
    assert dftrajectory["nstep"].to_list() == [1, 2, 3, None]
    assert dftrajectory["abundancefile"].to_list() == [None, "nz-plane00002", None, "tday_nz-plane"]
    # timestep 3 takes its time from heating.dat, because energy_thermo.dat does not have timestep 3
    # the named file has no timestep, and its time comes from its header
    assert dftrajectory["time_s"].to_list() == [1.0, 9.0e4, 1.0e5, 8.64e4]
    assert dftrajectory["abundancetime_s"].to_list() == [None, 9.0e4, None, 8.64e4]
    # the Fortran exponent of hbeta has no "e", and the later row of timestep 2 replaces the earlier row
    assert dftrajectory["hbeta"].to_list()[:3] == pytest.approx([1.842623e-281, 4.0e9, 6.0e9])
    assert dftrajectory["Qdot"].to_list() == [pytest.approx(1.5e21), pytest.approx(2.0e10), None, None]
    # a timestep with no abundance file has null lists, and a file with no nuclide lines has empty lists
    assert dftrajectory["N"].to_list() == [None, [0, 2], None, [0, 2]]

    # a reader of heating.dat gets the rows of that file, and a reader of energy_thermo.dat gets its rows
    dfheating = at.inputmodel.rprocess_from_trajectory.get_network_timesteps_table(tmp_path, 7, ("hbeta",))
    assert dfheating["nstep"].to_list() == [1, 2, 3]
    dfthermo = at.inputmodel.rprocess_from_trajectory.get_network_timesteps_table(tmp_path, 7, ("Qdot",))
    assert dfthermo["nstep"].to_list() == [1, 2]
    # the search for the closest timestep sees timestep 2 alone, because the other timesteps have no abundance file
    assert at.inputmodel.rprocess_from_trajectory.get_traj_network_timesteps(tmp_path, 7)["nstep"].to_list() == [2]
    assert at.inputmodel.rprocess_from_trajectory.get_closest_network_timesteps(tmp_path, 7, [1.0, 1.0e5]) == [2, 2]
    with pytest.raises(FileNotFoundError, match="no column halpha"):
        at.inputmodel.rprocess_from_trajectory.get_network_timesteps_table(tmp_path, 7, ("halpha",))

    dfnucabund, timesecs = at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfiles_nuc_abund(
        tmp_path, 7, ["tday_nz-plane", "nz-plane00002"]
    )
    assert timesecs == [8.64e4, 9.0e4]
    assert dfnucabund["fileindex"].to_list() == [0, 0, 1, 1]
    assert dfnucabund["massfrac"].to_list() == pytest.approx([10**-11.1798, 4 * 10**-1.27404] * 2)

    with pytest.raises(FileNotFoundError, match="nz-plane00003"):
        at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfiles_nuc_abund(tmp_path, 7, ["nz-plane00003"])


def test_trajectory_archive_leaves_out_a_corrupt_timestep_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_trajectory_archive: Callable[[Path, int, dict[str, str], str], Path],
) -> None:
    """A corrupt time in heating.dat must not stop a read of the abundance files, which do not need that table."""
    nuclidelines = "  0   1 -1.11798E+01  0.00000E+00  0.00000E+00\n"
    write_trajectory_archive(
        tmp_path,
        5,
        {
            # the last line of energy_thermo.dat is incomplete, as a network run that stopped during a write leaves
            # it. The cut is in the Qdot column, which a reader uses
            "energy_thermo.dat": (
                "#count time/s Qdot Ye\n     1  1.000000E+00  1.500000E+21  **********\n     2  2.000000E+00\n"
            ),
            "heating.dat": "#count time/s hbeta\n     1  corrupt  2.000000E+09\n",
            "nz-plane00001": "  1  1.000000E+00   1.0E-06   2.0E-12   0.0E+00   6.3E-06\n" + nuclidelines,
        },
        ".tar.xz",
    )
    _, timesecs = at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfiles_nuc_abund(
        tmp_path, 5, ["nz-plane00001"]
    )
    assert timesecs == [1.0]
    warnings = capsys.readouterr().err
    assert "heating.dat in 5.tar.xz" in warnings
    assert "The last line of energy_thermo.dat in 5.tar.xz is incomplete" in warnings
    assert at.inputmodel.rprocess_from_trajectory.get_traj_network_timesteps(tmp_path, 5)["nstep"].to_list() == [1]
    # a Fortran overflow token in a column that no reader uses does no harm, because the cache holds the columns of
    # the readers alone
    dfthermo = at.inputmodel.rprocess_from_trajectory.get_network_timesteps_table(tmp_path, 5, ("Qdot",))
    assert dfthermo["Qdot"].to_list() == pytest.approx([1.5e21])
    with pytest.raises(FileNotFoundError, match="no column Ye"):
        at.inputmodel.rprocess_from_trajectory.get_network_timesteps_table(tmp_path, 5, ("Ye",))
    # the error of a later run names the file that the cache left out
    with pytest.raises(FileNotFoundError, match=r"no column hbeta.*left out.*heating\.dat"):
        at.inputmodel.rprocess_from_trajectory.get_network_timesteps_table(tmp_path, 5, ("hbeta",))


def test_trajectory_timestep_files_reject_a_blank_header_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write_trajectory_archive: Callable[[Path, int, dict[str, str], str], Path],
) -> None:
    """A timestep file whose first line is blank gives no time, thus the cache leaves it out and names that file.

    A blank line gave a null field count, which the check for six fields passed. The reader then returned None
    for the time. The check runs when artistools builds the cache. The other files of the archive stay readable,
    because a network run that stopped can leave one file with no header.
    """
    abundlines = "  0   1 -1.11798E+01  0.00000E+00  0.00000E+00\n  2   2 -1.27404E+00  2.05770E+01  0.00000E+00\n"
    goodfile = "  1  8.873645E+03   6.601614E-06   2.981756E-12   0.000000E+00   6.377000E-06\n" + abundlines
    reader = at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfiles_nuc_abund

    write_trajectory_archive(tmp_path / "good", 1, {"nz-plane00001": goodfile}, ".tar.xz")
    _, timesecs = reader(tmp_path / "good", 1, ["nz-plane00001"])
    assert timesecs == [8873.645]

    # a network run that stopped can leave a file with no header, and a file with a cut nuclide line
    write_trajectory_archive(
        tmp_path / "bad",
        1,
        {
            "nz-plane00001": goodfile,
            "nz-plane00002": "\n" + abundlines,
            "nz-plane00003": goodfile + "  3   4 -1.2E+",
            "nz-plane00004": "  4  8.87x645E+03   6.6E-06   2.9E-12   0.0E+00   6.3E-06\n" + abundlines,
            "nz-plane00005": "  5  8.873645E+03   6.601614E-06   2.981756E-12   0.000000E+00   6.377000E-06\n",
        },
        ".tar.xz",
    )
    _, timesecs = reader(tmp_path / "bad", 1, ["nz-plane00001"])
    assert timesecs == [8873.645]
    warnings = capsys.readouterr().err
    for filename in ("nz-plane00002", "nz-plane00003", "nz-plane00004", "nz-plane00005"):
        assert f"{filename} in 1.tar.xz" in warnings
    with pytest.raises(FileNotFoundError, match="nz-plane00002"):
        reader(tmp_path / "bad", 1, ["nz-plane00001", "nz-plane00002"])
    with pytest.raises(FileNotFoundError, match="nz-plane00003"):
        reader(tmp_path / "bad", 1, ["nz-plane00003"])


def test_get_trajectory_abund_q(trajectory_copy: Path) -> None:
    particleid = 109215

    abund_q = at.inputmodel.rprocess_from_trajectory.get_trajectory_abund_q(
        particleid=particleid, traj_root=trajectory_copy, t_model_s=0.1 * 86400
    )

    dfexpected = pl.read_csv(at.get_path("testdata") / "kilonova" / f"trajectory_{particleid}_massfractions.csv")
    nuclidekeys = list(zip(dfexpected["Z"], dfexpected["N"], strict=True))
    missingkeys = [key for key in nuclidekeys if key not in abund_q]
    assert not missingkeys, f"Keys {missingkeys} not found in abund_q"
    massfracs = np.array([abund_q[key] for key in nuclidekeys])
    assert np.allclose(massfracs, dfexpected["massfrac"].to_numpy(), rtol=1e-9, atol=0.0)

    # this value needs float64. An earlier reader gave time/s at float32, and the integration lost precision
    assert math.isclose(abund_q["q"], 5351333204182925.0)


def test_plotdensity_grid_with_one_cell_radius(tmp_path: Path) -> None:
    """A grid of 2 x 2 x 2 cells has one mid-point velocity, thus the bins have one step of zero.

    The bin width was the largest step of the mid-point velocities, thus the bin count divided by zero.
    """
    smallmodelpath = at.inputmodel.downscale3dgrid.make_downscaled_3d_grid(
        modelpath_3d, outputgridsize=2, outputfolder=tmp_path
    )

    at.inputmodel.plotdensity.main(argsraw=[], modelpath=[smallmodelpath], outputpath=tmp_path / "plot")

    assert (tmp_path / "plot" / "densityprofile.pdf").is_file()


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdensity_nbins_covers_the_model_below_xmax(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The fixed bins of -nbins must hold every cell, also the cells above -xmax.

    The bins ended at -xmax, thus get_binned_profile dropped each cell above it. The profile then gave
    no mass above -xmax, which a reader takes for an empty outer model.
    """
    _, modelmeta = at.inputmodel.scan_modeldata(modelpath_3d, printwarningsonly=True)
    vmax_on_c = modelmeta["vmax_cmps"] / at.constants.C_cm_per_s
    xmax = vmax_on_c / 2.0

    at.inputmodel.plotdensity.main(
        argsraw=[], modelpath=[modelpath_3d], outputpath=tmp_path, nbins=20, xmax=xmax, quiet=True
    )

    # main names the axes after it draws on them, thus the label of the axes gives the dM/dv profile
    massprofiles = [call for call in mockplot.call_args_list if call.args[0].get_ylabel().startswith(r"$\Delta$M")]
    assert len(massprofiles) == 1
    xvalues = np.asarray(massprofiles[0].args[1], dtype=float)
    massvalues = np.asarray(massprofiles[0].args[2], dtype=float)

    assert xvalues.max() > xmax, "the bins must reach past -xmax"
    assert massvalues[xvalues > xmax].max() > 0.0, "the outer cells must keep their mass"


@pytest.mark.benchmark
def test_plotinitialcomposition(tmp_path: Path) -> None:
    at.inputmodel.plotinitialcomposition.main(
        argsraw=["-modelpath", str(modelpath_3d), "-o", str(tmp_path), "rho", "Fe"]
    )

    assert (tmp_path / "plotcomposition_rho,fe.pdf").is_file()


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdensity_write_data_gives_the_plotted_profiles(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--write_data gives the points of each curve of each panel, and a model with no Ye has a null Ye.

    The enclosed mass and the dM/dv profile have different velocity grids, thus each one has its own table.
    """
    yemodelpath = tmp_path / "yemodel"
    dfyemodel = pl.DataFrame({
        "inputcellid": [1, 2, 3, 4],
        "vel_r_max_kmps": [5000.0, 10000.0, 15000.0, 20000.0],
        "logrho": [-10.0, -10.5, -11.0, -12.0],
        "X_Fegroup": [1.0, 1.0, 1.0, 1.0],
        "X_Ni56": [0.5, 0.4, 0.3, 0.2],
        "X_Co56": [0.0, 0.0, 0.0, 0.0],
        "X_Fe52": [0.0, 0.0, 0.0, 0.0],
        "X_Cr48": [0.0, 0.0, 0.0, 0.0],
        "Ye": [0.25, 0.3, 0.35, 0.45],
    })
    at.inputmodel.save_modeldata(dfyemodel, outpath=yemodelpath, modelmeta={"dimensions": 1, "t_model_init_days": 1.0})

    at.inputmodel.plotdensity.main(
        argsraw=[],
        modelpath=[yemodelpath, modelpath],
        label=["withye", "noye"],
        outputpath=tmp_path,
        plotye=True,
        write_data=True,
        quiet=True,
    )

    dfenclosed = read_data_file(tmp_path / "densityprofile.txt")
    dfprofile = read_data_file(tmp_path / "densityprofile_massprofile.txt")
    assert dfprofile.filter(pl.col("model") == "noye")["Ye"].is_null().all()

    # main names the axes after it draws on them, thus the label of the axes gives the panel of each call
    panels = {"Mass Enclosed": (dfenclosed, "mass_enclosed_msun"), r"$\Delta$M": (dfprofile, "dmass_dv_msun_per_c")}
    panels["Electron fraction"] = (dfprofile.drop_nulls("Ye"), "Ye")
    ncurves = 0
    for call in mockplot.call_args_list:
        ylabel = call.args[0].get_ylabel()
        dftable, ycolumn = next(value for key, value in panels.items() if ylabel.startswith(key))
        dfcurve = dftable.filter(pl.col("model") == call.kwargs["label"])
        np.testing.assert_allclose(dfcurve["vel_r_on_c"], np.asarray(call.args[1], dtype=float), rtol=1e-12)
        np.testing.assert_allclose(dfcurve[ycolumn], np.asarray(call.args[2], dtype=float), rtol=1e-12, atol=0.0)
        ncurves += 1
    # two models on each of the two mass panels, and one model with Ye on the Ye panel
    assert ncurves == 5


def test_plotinitialcomposition_write_data_gives_each_pixel_of_the_slice(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--write_data gives one row for each pixel, with the velocities of the cell and the value of each panel.

    The values are the values of the images, thus they have the log10 of the colour scale, and a hidden empty cell
    has a null value.
    """
    from artistools.inputmodel.plotinitialcomposition import get_2d_slice_through_3d_model

    with mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True) as mockimshow:
        at.inputmodel.plotinitialcomposition.main(
            argsraw=["rho", "Fe", str(modelpath_3d), "-o", str(tmp_path), "--logcolorscale", "--hideemptycells"],
            write_data=True,
        )

    dftable = read_data_file(tmp_path / "plotcomposition_rho,fe.txt")
    assert mockimshow.call_count == 2
    for call, column in zip(mockimshow.call_args_list, ["log10_rho_g_per_cm3", "log10_rho_Fe_g_per_cm3"], strict=True):
        valuegrid = call.args[1]
        isshown = ~np.ma.getmaskarray(valuegrid).ravel()
        assert not isshown.all(), "the 3D test model holds an empty cell to hide"
        np.testing.assert_array_equal(dftable[column].is_not_null().to_numpy(), isshown)
        np.testing.assert_allclose(dftable[column].to_numpy()[isshown], np.ma.getdata(valuegrid).ravel()[isshown])

    # the default slice is the layer above the origin along z, which shows x against y
    dfmodel, modelmeta = get_derived_modeldata(modelpath_3d)
    dfslice = get_2d_slice_through_3d_model(dfmodel.collect(), "z")
    t_model_s = modelmeta["t_model_init_days"] * day_to_s
    for axis in ("x", "y"):
        vel_mid_on_c = (
            (dfslice[f"pos_{axis}_min"] + dfslice[f"pos_{axis}_max"]) / 2.0 / t_model_s / at.constants.C_cm_per_s
        )
        np.testing.assert_allclose(dftable[f"vel_{axis}_on_c"], vel_mid_on_c, rtol=1e-5)


def test_a_cache_of_the_earlier_name_stays_in_use(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Use a cache named *.parquet.tmp, from before the caches took the name *.parquet, and do not convert again.

    If a rename of the caches had no such rule, artistools would convert the data of each model again. This takes a
    long time for a large model. A cache of the current name wins when both exist.
    """
    textfile = tmp_path / "model.txt"
    textfile.write_text("x", encoding="utf-8")
    legacycache = tmp_path / "model.txt.parquet.tmp"
    legacycache.write_bytes(b"")
    assert at.inputmodel.core.get_parquet_cache_path(textfile) == legacycache

    currentcache = tmp_path / "model.txt.parquet"
    currentcache.write_bytes(b"")
    assert at.inputmodel.core.get_parquet_cache_path(textfile) == currentcache

    at.inputmodel.core.remove_parquet_cache(textfile)
    assert not currentcache.exists()
    assert not legacycache.exists()
    capsys.readouterr()


@pytest.mark.benchmark
def test_save_load_3d_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    lzdfmodel, modelmeta = get_empty_3d_model(ncoordgrid=25, vmax=1000, t_model_init_days=1, includenico57=True)
    dfmodel = lzdfmodel.collect()

    # CodSpeed runs a benchmark test more than one time in one process, and each run writes to the
    # same folder. A seed gives the same model each time, thus a cache of an earlier run still matches
    rng = np.random.default_rng(seed=6021)

    # give a random rho to half of the cells
    dfmodel[rng.integers(0, dfmodel.height, dfmodel.height // 2), "rho"] = 10.0 * rng.random(
        dfmodel.height // 2, dtype=np.float32
    )

    dfelements = (
        at.atomic
        .get_elsymbols_df()
        .filter(
            pl.col("atomic_number").is_between(1, 50) | (pl.col("atomic_number") == 113)
        )  # Z=113 Uut has three chars, so test that that too
        .sort(by="atomic_number")
        .with_columns(
            elemcolname="X_" + pl.col("elsymbol"), mass_number_example=(pl.col("atomic_number") * 2).cast(pl.Int32)
        )
        .collect()
    )

    elcolnames = dfelements["elemcolname"].to_list()

    # this give us several isotopes for each element (doesn't matter if they are real or not)
    isocolnames = pl.concat(
        dfelements.select(
            (pl.col("elemcolname") + (pl.col("mass_number_example") + i).cast(pl.String)).alias("isocolname")
        )
        for i in range(2)
    ).get_column("isocolname")

    # give random abundances to the cells
    randommassfracs = rng.random(dfmodel.height, dtype=np.float32)
    assert isinstance(randommassfracs, np.ndarray)
    dfmodel = dfmodel.with_columns([
        pl.Series(name=isocol, values=randommassfracs, dtype=pl.Float32) for isocol in isocolnames
    ])

    # an abundance counts for nothing when rho is zero. Set each one to zero, so that the values
    # match the dataframe that the reader gives
    dfmodel = dfmodel.with_columns(
        pl.when(dfmodel["rho"] > 0).then(pl.col(col)).otherwise(0) for col in dfmodel.columns if col.startswith("X_")
    )

    # sum isotopic abundances to get elemental abundances
    dfelemabundances = dfmodel.select([
        "inputcellid",
        *[pl.sum_horizontal(cs.starts_with(elcol)).cast(pl.Float32).alias(elcol) for elcol in elcolnames],
    ])

    outpath = tmp_path
    at.inputmodel.save_modeldata(outpath=outpath, dfmodel=dfmodel, modelmeta=modelmeta)
    at.inputmodel.save_initelemabundances(outpath=outpath, dfelabundances=dfelemabundances)
    # the reader writes no cache for a file below 2 MiB, thus the test reads an uncompressed copy of the zstd file
    decompress_file(outpath / "model.txt.zst")

    # the first load reads the text file, and it writes a cache, because the file is larger than 2 MiB.
    # The second load reads that cache
    cachefilepath = outpath / "model.txt.parquet"
    for loadnumber in (0, 1):
        assert cachefilepath.is_file() == (loadnumber == 1)
        capsys.readouterr()
        dfmodel_loaded, modelmeta_loaded = at.inputmodel.scan_modeldata(modelpath=outpath)
        output = capsys.readouterr().out
        assert (f"Reading table from {cachefilepath}" in output) == (loadnumber == 1)
        assert (f"Saving {cachefilepath}" in output) == (loadnumber == 0)
        assert set(dfmodel_loaded.collect_schema().names()) == set(dfmodel.columns)
        pltest.assert_frame_equal(
            dfmodel, dfmodel_loaded.collect(), check_column_order=False, check_dtypes=False, rel_tol=1e-4, abs_tol=1e-4
        )
        assert modelmeta == modelmeta_loaded

        dfelemabundances_loaded = at.inputmodel.scan_initelemabundances(outpath)
        pltest.assert_frame_equal(
            dfelemabundances,
            dfelemabundances_loaded.select(
                # the file holds each atomic number from 1 to at least 30, thus the loaded dataframe has more
                # columns than dfelemabundances
                dfelemabundances.columns
            ).collect(),
            check_column_order=False,
            check_dtypes=False,
            rel_tol=1e-3,
            abs_tol=1e-3,
        )


@pytest.mark.parametrize("outputdimensions", [2, 1, 0])
def test_dimension_reduce(outputdimensions: int, benchmark: BenchmarkFixture) -> None:
    dfmodel3d_pl_lazy, modelmeta_3d = get_empty_3d_model(ncoordgrid=50, vmax=100000, t_model_init_days=1)
    dfmodel3d_pl = dfmodel3d_pl_lazy.collect()

    # it's important that we don't fill cells in the cube corners, as they will be lost when reducing dimensions
    mgi1 = 26 * 26 * 26 + 26 * 26 + 26
    dfmodel3d_pl[mgi1, "rho"] = 2
    dfmodel3d_pl[mgi1, "X_Ni56"] = 0.5
    mgi2 = 25 * 25 * 25 + 25 * 25 + 25
    dfmodel3d_pl[mgi2, "rho"] = 1
    dfmodel3d_pl[mgi2, "X_Ni56"] = 0.75

    dfmodel3d_derived = at.inputmodel.add_derived_cols_to_modeldata(dfmodel=dfmodel3d_pl, modelmeta=modelmeta_3d)
    mass_g_3d, ejecta_ke_erg = dfmodel3d_derived.select(pl.sum("mass_g"), pl.sum("kinetic_en_erg")).collect().row(0)
    dfmodel3d_pl = (
        dfmodel3d_derived
        .select(*dfmodel3d_pl.columns, "mass_g")
        .collect()
        .with_columns(tracercount=pl.lit(1, dtype=pl.Int32))
    )

    outpath = outputpath / f"test_dimension_reduce_3d_{outputdimensions:d}d"

    outpath.mkdir(exist_ok=True, parents=True)

    tracercountdtypes: list[pl.DataType] = []

    def run_dimension_reduce() -> None:
        (dfmodel_lowerd, _, _, modelmeta_lowerd) = (at.inputmodel.dimension_reduce_model)(
            dfmodel=dfmodel3d_pl, modelmeta=modelmeta_3d, outputdimensions=outputdimensions
        )
        tracercountdtypes.append(dfmodel_lowerd.schema["tracercount"])
        at.inputmodel.save_modeldata(outpath=outpath, dfmodel=dfmodel_lowerd, modelmeta=modelmeta_lowerd)

    benchmark(run_dimension_reduce)

    # tracercount counts the trajectories of a cell. A float fill of the empty output cells made the
    # column a float, and the writer then gave 0.0 in place of 0
    assert tracercountdtypes[0].is_integer()

    dfmodel_lowerd_lz, _ = get_derived_modeldata(outpath)
    dfmodel_lowerd = dfmodel_lowerd_lz.collect()

    # check that the total mass is conserved
    assert math.isclose(dfmodel_lowerd["mass_g"].sum(), mass_g_3d, rel_tol=1e-3)

    lowerd_ejecta_ke_erg: float | int = dfmodel_lowerd.select("kinetic_en_erg").sum().item()

    # check that kinetic energy very roughly matches (we conserved mass, not kinetic energy)
    assert math.isclose(lowerd_ejecta_ke_erg, ejecta_ke_erg, rel_tol=0.04), f"{lowerd_ejecta_ke_erg} {ejecta_ke_erg}"

    # check that the total mass of each species is conserved
    for col in dfmodel3d_pl.columns:
        if col.startswith("X_"):
            assert math.isclose(
                (dfmodel_lowerd["mass_g"] * dfmodel_lowerd[col]).sum(),
                (dfmodel3d_pl["mass_g"] * dfmodel3d_pl[col]).sum(),
                rel_tol=1e-5,
            )


def test_pos_r_min_straddling_cells() -> None:
    """A cell that straddles a coordinate plane reaches zero along that axis, so pos_r_min must account for it."""
    # an odd ncoordgrid puts a cell centred on the origin, and rings of cells straddling each coordinate plane
    ncoordgrid = 5
    dfmodel, modelmeta = get_empty_3d_model(ncoordgrid=ncoordgrid, vmax=1e9, t_model_init_days=1.0)

    dfmodel = at.inputmodel.add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta).collect()

    for ax in ("x", "y", "z"):
        straddles = (dfmodel[f"pos_{ax}_min"] < 0) & (dfmodel[f"pos_{ax}_max"] > 0)
        assert straddles.any(), f"expected some cells straddling the {ax} plane"

    straddles_all = (
        ((dfmodel["pos_x_min"] < 0) & (dfmodel["pos_x_max"] > 0))
        & ((dfmodel["pos_y_min"] < 0) & (dfmodel["pos_y_max"] > 0))
        & ((dfmodel["pos_z_min"] < 0) & (dfmodel["pos_z_max"] > 0))
    )
    # the cell containing the origin has zero closest approach
    assert dfmodel.filter(straddles_all)["pos_r_min"].to_list() == pytest.approx([0.0])

    # pos_r_min can never exceed the distance to any corner of its own cell, nor pos_r_mid
    assert (dfmodel["pos_r_min"] <= dfmodel["pos_r_mid"] * (1.0 + 1e-9)).all()
    assert (dfmodel["pos_r_min"] <= dfmodel["pos_r_max"]).all()


def test_min_abs_coordinate() -> None:
    dfpos = pl.DataFrame({"pos_x_min": [-3.0, 1.0, -5.0, 0.0], "pos_x_max": [-1.0, 4.0, 2.0, 3.0]})

    result = dfpos.select(at.inputmodel.core.min_abs_coordinate("x")).to_series().to_list()

    # entirely negative -> 1, entirely positive -> 1, straddling -> 0, touching the plane -> 0
    assert result == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_vel_on_c_excludes_kmps_columns() -> None:
    """The vel_*_kmps columns are in km/s, so they must not be divided by c as if they were cm/s."""
    dfmodel, _modelmeta = get_derived_modeldata(modelpath)
    dfmodel = dfmodel.collect()

    assert "vel_r_max_kmps" in dfmodel.columns
    assert "vel_r_max_kmps_on_c" not in dfmodel.columns
    assert "vel_r_max_on_c" in dfmodel.columns

    # vel_r_max is in cm/s, so its fraction of c must stay below 1 for a physical model
    assert dfmodel["vel_r_max_on_c"].to_numpy().max() < 1.0
    assert dfmodel["vel_r_max_on_c"].to_numpy() == pytest.approx(
        dfmodel["vel_r_max_kmps"].to_numpy() * 1e5 / at.constants.C_cm_per_s
    )


def test_scale_model_to_time_keeps_empty_cells_and_scales_volume() -> None:
    """A shrink keeps the empty-cell marker logrho = -99, and the volume follows the positions.

    The marker moved to -97.8 for a factor of 0.4, thus the empty cell got a density. The volume did not change.
    """
    dfmodel = pl.DataFrame({"logrho": [-99.0, 1.0], "volume": [1.0, 2.0]})

    dfscaled = at.inputmodel.scale_model_to_time(dfmodel=dfmodel, targetmodeltime_days=0.4, t_model_days=1.0)

    assert dfscaled["logrho"].to_list() == pytest.approx([-99.0, 1.0 - 3 * math.log10(0.4)])
    assert dfscaled["volume"].to_list() == pytest.approx([0.4**3, 2.0 * 0.4**3])


def test_scan_initelemabundances_reads_both_delimiter_formats(tmp_path: Path) -> None:
    """Read an abundances.txt that has one space between two values with no call of read_wsv.

    An older file with more spaces, and a file with a nan token, must give the same table as read_wsv gives.
    """
    rng = np.random.default_rng(seed=11)
    dfelabundances = pl.DataFrame(
        {"inputcellid": range(1, 6)}
        | {f"X_{at.get_elsymbol(z)}": rng.random(5, dtype=np.float32) for z in range(1, 31)}
    )
    onespacefolder = tmp_path / "onespace"
    at.inputmodel.save_initelemabundances(dfelabundances, outpath=onespacefolder)
    with mock.patch("artistools.inputmodel.core.read_wsv", side_effect=AssertionError("the slow reader ran")):
        dfonespace = at.inputmodel.scan_initelemabundances(onespacefolder).collect()

    morespacesfolder = tmp_path / "morespaces"
    morespacesfolder.mkdir()
    with at.zopen(onespacefolder / "abundances.txt.zst", encoding="utf-8") as fabund:
        lines = fabund.read().splitlines()
    (morespacesfolder / "abundances.txt").write_text(
        "\n".join(line if line.startswith("#") else "   " + line.replace(" ", "   ") for line in lines) + "\n",
        encoding="utf-8",
    )
    pltest.assert_frame_equal(dfonespace, at.inputmodel.scan_initelemabundances(morespacesfolder).collect())
    pltest.assert_frame_equal(dfonespace, dfelabundances.with_columns(pl.col("inputcellid").cast(pl.Int32)))

    # read_wsv gives null and not NaN for a nan token, and the fast reader must not change that
    nanfolder = tmp_path / "nan"
    nanfolder.mkdir()
    (nanfolder / "abundances.txt").write_text("1 nan 0.5\n2 0.25 0.75\n", encoding="utf-8")
    assert at.inputmodel.scan_initelemabundances(nanfolder).collect()["X_H"].to_list() == [None, 0.25]


def test_slice_abundance_file_renumbers_the_cells(tmp_path: Path) -> None:
    """Keep the cells of the slice, also the last cell of the input.

    The cells of the output must have the 1D cell numbers.
    """
    from artistools.inputmodel.make1dslicefrom3d import slice_abundance_file

    inputfolder = tmp_path / "in"
    outputfolder = tmp_path / "out"
    at.inputmodel.save_initelemabundances(
        pl.DataFrame({"inputcellid": [1, 2, 3], "X_H": [0.001, 0.002, 0.003]}), outpath=inputfolder
    )

    slice_abundance_file(inputfolder, outputfolder, {1: 1, 3: 2})

    dfelabundances = at.inputmodel.scan_initelemabundances(outputfolder).collect()
    assert dfelabundances["inputcellid"].to_list() == [1, 2]
    assert np.allclose(dfelabundances["X_H"], [0.001, 0.003], rtol=1e-6)

    # a cell of the slice that the input does not hold must stop the command
    with pytest.raises(ValueError, match="full line of mass fractions"):
        slice_abundance_file(inputfolder, tmp_path / "out2", {1: 1, 4: 2})


@pytest.mark.parametrize(("ncoordgrid", "cellids3d"), [(3, [14, 15]), (4, [43, 44])])
def test_slice_3dmodel_takes_the_cells_that_hold_the_axis(
    tmp_path: Path, ncoordgrid: int, cellids3d: list[int]
) -> None:
    """Select the cells that hold the positive x axis.

    A grid with an even cell count has a cell face on the axis. With an odd count, the axis goes through
    the middle of a cell, and the first shell reaches half a cell width.
    """
    from artistools.inputmodel.make1dslicefrom3d import slice_3dmodel

    vmax_cmps, t_model_days = 1.0e9, 1.0
    lzdfmodel, modelmeta = get_empty_3d_model(ncoordgrid=ncoordgrid, vmax=vmax_cmps, t_model_init_days=t_model_days)
    inputfolder = tmp_path / "in"
    outputfolder = tmp_path / "out"
    at.inputmodel.save_modeldata(lzdfmodel.with_columns(rho=pl.lit(1.0e-10)), outpath=inputfolder, modelmeta=modelmeta)

    dict3dcellidto1dcellid, xlist, _ylists = slice_3dmodel(inputfolder, outputfolder, "x")

    assert dict3dcellidto1dcellid == dict(zip(cellids3d, [1, 2], strict=True))
    wid_init_kmps = 2 * vmax_cmps / ncoordgrid / at.constants.km_to_cm
    vel_r_max_first = wid_init_kmps / 2 if ncoordgrid % 2 else wid_init_kmps
    assert xlist == pytest.approx([vel_r_max_first, vel_r_max_first + wid_init_kmps], rel=1e-5)
    dfmodel1d, modelmeta1d = at.inputmodel.scan_modeldata(outputfolder)
    assert modelmeta1d["dimensions"] == 1
    assert dfmodel1d.collect()["vel_r_max_kmps"].to_list() == pytest.approx(xlist, rel=1e-5)


def test_opacityfile_uniform_multicell(tmp_path: Path) -> None:
    """A uniform opacity.txt reads back with the same opacity in each of the 1000 cells of the model."""
    at.inputmodel.opacityinputfile.main(
        argsraw=[], action="uniform", modelpath=modelpath_classic_3d, outputpath=tmp_path, kappa=0.1
    )

    opacities = at.inputmodel.opacityinputfile.get_opacity_from_file(tmp_path)
    assert len(opacities) == 1000
    assert np.allclose(opacities, 0.1)


def test_energyfiles_written_then_described(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The energy files must read back with the totals and rate bounds they were written with.

    Each cell gets the energy per gram times its mass. A share in proportion to the density gave a shell
    of a 1D model 0.063 to 16.4 times the intended energy per gram.
    """
    cellmass_grams = np.array([1.0e32, 2.0e32, 7.0e32])
    mtot_grams = float(cellmass_grams.sum())
    at.inputmodel.energyinputfiles.make_energy_files(cellmass_grams, tmp_path)

    etot, energydistribution = at.inputmodel.energyinputfiles.get_etot_fromfile(tmp_path)
    assert len(energydistribution) == len(cellmass_grams)
    # the files hold six significant figures, thus the shares agree only to that precision
    assert np.allclose(
        energydistribution["cell_energy"].to_numpy() / etot, cellmass_grams / mtot_grams, rtol=1e-5, atol=0.0
    )

    # the analytic integral of 5e9 t^-1.3 erg/g/s over the seconds between 1e-4 days and 50 days.
    # An integration over the times in days gave a total that was 2.8 per cent too small
    analytic_etot_per_gram = 5e9 * at.constants.day_to_s / 0.3 * (0.0001**-0.3 - 50.0**-0.3)
    assert etot / mtot_grams == pytest.approx(analytic_etot_per_gram, rel=1e-3)

    dfrate = at.inputmodel.energyinputfiles.get_energy_rate_fromfile(tmp_path)
    assert dfrate["rate"].min() == pytest.approx(0.0)
    assert dfrate["rate"].max() == pytest.approx(1.0), "the cumulative energy fraction must reach one"

    at.inputmodel.energyinputfiles.main(argsraw=[], action="describe", modelpath=tmp_path)
    assert "energydistribution.txt" in capsys.readouterr().out


def test_plotrate_shows_the_heating_law_of_the_energy_files() -> None:
    """The plot of plotrate drew a different law from energyrate.txt, with approximately two times the rate."""
    import matplotlib.pyplot as plt

    from artistools.inputmodel import energyinputfiles

    fig, axis = plt.subplots()
    dfplotted = energyinputfiles.plot_energy_rate(modelpath, axis)
    plt.close(fig)
    lzmodel, modelmeta = at.inputmodel.scan_modeldata(modelpath)
    mass_g = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .select(pl.col("mass_g").sum())
        .collect()
        .item()
    )
    times_days = dfplotted["time_days"].to_numpy()
    heatingrate = dfplotted["nuclear_heating_power"].to_numpy() / mass_g
    assert np.allclose(heatingrate, 5e9 * times_days**-1.3, rtol=1e-12, atol=0.0)

    # the total energy of the files is the integral of the plotted rate
    dfcumulative, e_tot_per_gram = energyinputfiles.get_cumulative_heating_fraction()
    assert np.allclose(dfcumulative["times"].to_numpy(), times_days, rtol=1e-12, atol=0.0)
    assert np.isclose(e_tot_per_gram, np.trapezoid(heatingrate, times_days * day_to_s), rtol=1e-12)


def test_abundances_that_lack_a_cell_of_the_model_stop_the_read(tmp_path: Path) -> None:
    """The join with abundances.txt dropped a cell that the file does not give, and each mass left that cell out."""
    dfmodel = pl.DataFrame({
        "inputcellid": [1, 2],
        "vel_r_max_kmps": [1000.0, 2000.0],
        "logrho": [-10.0, -11.0],
        "X_Fegroup": [1.0, 1.0],
        "X_Ni56": [0.5, 0.5],
        "X_Co56": [0.0, 0.0],
        "X_Fe52": [0.0, 0.0],
        "X_Cr48": [0.0, 0.0],
    })
    at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta={"dimensions": 1, "t_model_init_days": 1.0})
    at.inputmodel.save_empty_abundance_file(npts_model=1, outputfilepath=tmp_path)

    assert at.inputmodel.scan_modeldata(tmp_path)[0].select(pl.len()).collect().item() == 2
    with pytest.raises(ValueError, match="no abundances for 1 of the 2 cells"):
        at.inputmodel.scan_modeldata(tmp_path, get_elemabundances=True)


def test_dimension_reduction_logs_the_mass_outside_the_output_grid(capsys: pytest.CaptureFixture[str]) -> None:
    """The log must give the mass that the output model leaves out.

    The radial bins end at vmax, thus the output model leaves out the corner cells of a 3D model.
    """
    lzdfmodel, modelmeta = at.inputmodel.scan_modeldata(modelpath_classic_3d)
    mass_g_in = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzdfmodel, modelmeta=modelmeta)
        .select(pl.col("mass_g").sum())
        .collect()
        .item()
    )
    capsys.readouterr()
    dfmodel1d, _, _, modelmeta1d = at.inputmodel.dimension_reduce_model(
        lzdfmodel, outputdimensions=1, modelmeta=modelmeta
    )
    mass_g_out = (
        at.inputmodel
        .add_derived_cols_to_modeldata(dfmodel1d.lazy(), modelmeta=modelmeta1d)
        .select(pl.col("mass_g").sum())
        .collect()
        .item()
    )
    match = re.search(r"outside this grid hold (\S+) Msun", capsys.readouterr().out)
    assert match is not None
    assert mass_g_in > mass_g_out
    assert np.isclose(float(match[1]) * at.constants.Msun_to_g, mass_g_in - mass_g_out, rtol=1e-3)


def test_snapshot_electron_fraction_leaves_out_the_free_neutrons() -> None:
    """A free neutron (X_n1 or X_N1) is not in the sum, and a nitrogen nuclide in lower case (X_n14) is."""
    from artistools.inputmodel.describeinputmodel import calculate_model_electron_frac

    for neutroncolumn in ("X_n1", "X_N1"):
        dfmodel = pl.LazyFrame({neutroncolumn: [0.5], "X_He4": [0.25], "X_Li7": [0.25], "mass_g": [1.0]})
        assert calculate_model_electron_frac(dfmodel) == pytest.approx((0.25 * 2 / 4 + 0.25 * 3 / 7) / 0.5)
        assert calculate_model_electron_frac(pl.LazyFrame({neutroncolumn: [1.0], "mass_g": [1.0]})) is None

    for nitrogencolumn in ("X_n14", "X_N14"):
        dfmodel = pl.LazyFrame({nitrogencolumn: [1.0], "mass_g": [1.0]})
        assert calculate_model_electron_frac(dfmodel) == pytest.approx(0.5)


def test_energyfiles_from_trajectory(trajectory_copy: Path) -> None:
    """Integrating a trajectory's heating rate must give a positive total energy."""
    thermofile = trajectory_copy / "energy_thermo.dat"
    with tarfile.open(trajectory_copy / "109215.tar.xz", "r:*") as tarfilehandle:
        memberfile = tarfilehandle.extractfile("./Run_rprocess/energy_thermo.dat")
        assert memberfile is not None
        thermofile.write_bytes(memberfile.read())
    dfthermo = at.inputmodel.energyinputfiles.read_trajectory_thermo(thermofile)
    assert dfthermo["time/s"].to_numpy().min() >= 1.0, "the unphysical sub-second Qdot values must be dropped"

    dftimes_and_rate, e_tot = at.inputmodel.energyinputfiles.energy_from_rprocess_calculation(dfthermo)
    assert e_tot > 0.0
    assert dftimes_and_rate["rate"].max() == pytest.approx(1.0)


def write_2d_model(
    modeldir: Path,
    ncoordgridrcyl: int,
    ncoordgridz: int,
    vmax_cmps: float,
    t_model_days: float,
    zshift: float = 0.0,
    x_ni56_upper: float = 0.4,
) -> Path:
    """Write a 2D cylindrical model.txt whose cell midpoints follow the ARTIS grid definition.

    The cells with a negative z take a Ni56 mass fraction of 0.4, and the cells with a positive z take
    x_ni56_upper. A test that gives a different x_ni56_upper can then find the half of the model that a
    selection kept.
    """
    t_model_s = t_model_days * at.constants.day_to_s
    wid_init_rcyl = vmax_cmps * t_model_s / ncoordgridrcyl
    wid_init_z = 2 * vmax_cmps * t_model_s / ncoordgridz

    x_ni56_lower = 0.4
    lines = [f"{ncoordgridrcyl} {ncoordgridz}", str(t_model_days), f"{vmax_cmps:.4e}"]
    for modelgridindex in range(ncoordgridrcyl * ncoordgridz):
        n_r = modelgridindex % ncoordgridrcyl
        n_z = modelgridindex // ncoordgridrcyl
        pos_rcyl_mid = wid_init_rcyl * n_r + 0.5 * wid_init_rcyl
        pos_z_mid = -vmax_cmps * t_model_s + wid_init_z * n_z + 0.5 * wid_init_z + zshift
        x_ni56 = x_ni56_upper if pos_z_mid > 0 else x_ni56_lower
        lines.append(f"{modelgridindex + 1} {pos_rcyl_mid:.6e} {pos_z_mid:.6e} 1.0e-10 0.5 {x_ni56} 0.05 0.03 0.02")

    modelfile = modeldir / "model.txt"
    modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return modelfile


def test_scan_modeldata_2d(tmp_path: Path) -> None:
    """A 2D cylindrical model must be read with the right grid metadata and cell positions."""
    ncoordgridrcyl, ncoordgridz = 4, 6
    vmax_cmps, t_model_days = 1.0e9, 1.0
    modelfile = write_2d_model(tmp_path, ncoordgridrcyl, ncoordgridz, vmax_cmps, t_model_days)

    dfmodel, modelmeta = at.inputmodel.scan_modeldata(modelfile)

    assert modelmeta["dimensions"] == 2
    assert modelmeta["ncoordgridrcyl"] == ncoordgridrcyl
    assert modelmeta["ncoordgridz"] == ncoordgridz
    assert modelmeta["npts_model"] == ncoordgridrcyl * ncoordgridz
    assert math.isclose(modelmeta["vmax_cmps"], vmax_cmps)

    t_model_s = t_model_days * at.constants.day_to_s
    assert math.isclose(modelmeta["wid_init_rcyl"], vmax_cmps * t_model_s / ncoordgridrcyl)
    assert math.isclose(modelmeta["wid_init_z"], 2 * vmax_cmps * t_model_s / ncoordgridz)

    dfcollect = dfmodel.collect()
    assert dfcollect.height == ncoordgridrcyl * ncoordgridz
    assert dfcollect["modelgridindex"].to_list() == list(range(ncoordgridrcyl * ncoordgridz))
    # innermost cell of the lowest z row sits half a cell width from the axis and from the -z boundary
    assert dfcollect.item(0, "pos_rcyl_mid") == pytest.approx(0.5 * modelmeta["wid_init_rcyl"], rel=1e-5)
    assert dfcollect.item(0, "pos_z_mid") == pytest.approx(
        -vmax_cmps * t_model_s + 0.5 * modelmeta["wid_init_z"], rel=1e-5
    )


def test_model_header_comments_round_trip(tmp_path: Path) -> None:
    """Keep a user comment that starts as a line of the writer does.

    The reader must drop the lines that the writer adds again.
    """
    ncoordgridrcyl, ncoordgridz = 4, 6
    sourcefolder = tmp_path / "source"
    sourcefolder.mkdir()
    modelfile = write_2d_model(sourcefolder, ncoordgridrcyl, ncoordgridz, vmax_cmps=1.0e9, t_model_days=1.0)
    dfmodel, modelmeta = at.inputmodel.scan_modeldata(modelfile)
    # a comment of a user can start with the same words as a line of the writer
    usercomments = ["created: by hand from a test model", "column units: calibrated by hand"]
    modelmeta["headercommentlines"] = usercomments.copy()

    at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta=modelmeta)

    _, modelmeta_saved = at.inputmodel.scan_modeldata(tmp_path)
    assert modelmeta_saved["headercommentlines"] == usercomments
    assert modelmeta_saved["ncoordgridrcyl"] == ncoordgridrcyl
    assert modelmeta_saved["ncoordgridz"] == ncoordgridz
    assert math.isclose(modelmeta_saved["t_model_init_days"], 1.0)
    assert math.isclose(modelmeta_saved["vmax_cmps"], 1.0e9)


def test_scan_modeldata_2d_rejects_misplaced_cells(tmp_path: Path) -> None:
    """Cell midpoints inconsistent with vmax and the grid size must be rejected rather than read silently."""
    ncoordgridrcyl, ncoordgridz = 4, 6
    vmax_cmps, t_model_days = 1.0e9, 1.0
    wid_init_z = 2 * vmax_cmps * t_model_days * at.constants.day_to_s / ncoordgridz
    # shift every cell by one and a half cell widths along z, which no cell centre can be
    modelfile = write_2d_model(tmp_path, ncoordgridrcyl, ncoordgridz, vmax_cmps, t_model_days, zshift=1.5 * wid_init_z)

    with pytest.raises(AssertionError, match="pos_z_mid"):
        at.inputmodel.scan_modeldata(modelfile)


def test_fromcmfgen_writes_named_columns(tmp_path: Path) -> None:
    """The CMFGEN converter must resolve isotopes by name, not by a snapshot-specific column index.

    CMFGEN's isotope table holds Fe52 and Cr48 at the indices the original writer labelled cr48frac and
    v48frac, so the values were right but the names were not. Writing through save_modeldata by column name
    means the reader's X_Fe52/X_Cr48 slots are filled from the Fe52/Cr48 isotopes by construction.
    """
    from artistools.inputmodel.fromcmfgen import convert_to_artis as cmfgen

    nd = 3
    specnames = ["OXY", "CHRO", "IRON", "COB", "NICK", "BAR"]
    isonames = ["CHRO", "IRON", "COB", "NICK"]
    massnumbers = [48, 52, 56, 56]

    snapshot = {
        "nd": nd,
        "time": 1.3,
        "spec": specnames,
        "iso": isonames,
        "aiso": np.array(massnumbers),
        "rad": np.array([1.0, 2.0, 3.0]),
        "dens": np.array([1e-10, 1e-11, 1e-12]),
        "dmass": np.ones(nd),
        # one distinguishable value per (cell, species) and (cell, isotope)
        "specfrac": np.array([[0.5, 0.1, 0.2, 0.05, 0.1, 0.05]] * nd),
        "isofrac": np.array([[0.011, 0.022, 0.033, 0.044]] * nd),
    }

    with mock.patch.object(cmfgen, "rd_sn_hydro_data", return_value=snapshot):
        cmfgen.main(argsraw=["-snapshot", "unused", "-o", str(tmp_path)])

    dfmodel, modelmeta = at.scan_modeldata(tmp_path)
    dfmodel = dfmodel.collect()

    assert modelmeta["npts_model"] == nd
    assert np.isclose(modelmeta["t_model_init_days"], 1.3)

    # each isotope must reach the column named after it, not the one that shares its table position
    assert np.allclose(dfmodel["X_Cr48"].to_numpy(), 0.011, rtol=1e-4)
    assert np.allclose(dfmodel["X_Fe52"].to_numpy(), 0.022, rtol=1e-4)
    assert np.allclose(dfmodel["X_Co56"].to_numpy(), 0.033, rtol=1e-4)
    assert np.allclose(dfmodel["X_Ni56"].to_numpy(), 0.044, rtol=1e-4)

    # Cr, Fe, Co, Ni and Ba are all Z > 20, so the iron-group fraction is their sum
    assert np.allclose(dfmodel["X_Fegroup"].to_numpy(), 0.1 + 0.2 + 0.05 + 0.1 + 0.05, rtol=1e-4)

    dfabund = at.inputmodel.scan_initelemabundances(tmp_path).collect()
    assert np.allclose(dfabund["X_O"].to_numpy(), 0.5, rtol=1e-4)
    assert np.allclose(dfabund["X_Fe"].to_numpy(), 0.2, rtol=1e-4)
    # Ba is folded into X_Fegroup and gets no element column
    assert "X_Ba" not in dfabund.columns


def test_energyfiles_plotrate_makes_the_output_folder(tmp_path: Path) -> None:
    """A -o path that has no file extension names a folder, which the command makes.

    The path went to save_or_show as it came, thus the command wrote a file that had the name of the
    folder and no extension.
    """
    outfolder = tmp_path / "plots"
    at.inputmodel.energyinputfiles.main(argsraw=[], action="plotrate", modelpath=modelpath, outputfile=str(outfolder))

    assert outfolder.is_dir(), "the command must make the folder that -o names"
    assert (outfolder / "energyfiles_plotrate.pdf").is_file(), f"no plot in {list(outfolder.iterdir())}"


def test_make1dmodelfromaxis(tmp_path: Path) -> None:
    """--no-makefromcone samples the cells along the axis, and both axis directions agree.

    The positive-axis branch renamed the inner cell edge to vel_r_max_kmps and dropped the cell
    with pos_min == 0. Thus the velocity of each shell was low by one cell width. The branch also
    read args.t_model, which only the cone branch set, and passed a list of paths to scan_modeldata.
    """
    _, modelmeta3d = at.inputmodel.scan_modeldata(modelpath_3d)
    outdirs = {}
    for axis in ("+z", "-z"):
        outdir = tmp_path / f"axis_{axis[1]}_{'pos' if axis[0] == '+' else 'neg'}"
        outdir.mkdir()
        at.inputmodel.slice1dfromconein3dmodel.main(
            argsraw=[], modelpath=[modelpath_3d], outputpath=outdir, axis=axis, makefromcone=False
        )
        outdirs[axis] = outdir

    dfpos = at.inputmodel.scan_modeldata(outdirs["+z"] / "model_1d.txt")[0].collect()
    dfneg = at.inputmodel.scan_modeldata(outdirs["-z"] / "model_1d.txt")[0].collect()

    ncells_along_axis = int(modelmeta3d["ncoordgrid"]) // 2
    assert dfpos.height == ncells_along_axis
    assert dfneg.height == ncells_along_axis

    # the outer velocity of the outermost shell is the maximum velocity of the model
    vmax_kmps = modelmeta3d["vmax_cmps"] / 1.0e5
    assert np.isclose(dfpos["vel_r_max_kmps"].item(-1), vmax_kmps, rtol=1e-4)

    assert np.allclose(dfpos["vel_r_max_kmps"], dfneg["vel_r_max_kmps"], rtol=1e-6)


def test_profile_along_axis_takes_the_middle_cell_by_index() -> None:
    """For an odd cell count, the profile takes the middle cell of each other axis, also after a float rounding.

    The function took the lower edge with the smallest magnitude. The rounding of -dx/2 and +dx/2 then
    selected the cells from +dx/2 to 3dx/2, which do not hold the axis.
    """
    from artistools.inputmodel.slice1dfromconein3dmodel import get_profile_along_axis

    loweredges = [-1.5, -0.5000001, 0.4999999]
    dfmodel = pl.DataFrame(
        [
            {"pos_x_min": x, "pos_y_min": y, "pos_z_min": z}
            for z, y, x in itertools.product(loweredges, loweredges, loweredges)
        ],
        schema=dict.fromkeys(("pos_x_min", "pos_y_min", "pos_z_min"), pl.Float32),
    )
    args = argparse.Namespace(sliceaxis="z", other_axis1="x", other_axis2="y", positive_axis=False)

    dfprofile = get_profile_along_axis(dfmodel, args)

    assert dfprofile.height == 2
    assert np.allclose(dfprofile["pos_x_min"], -0.5, rtol=1e-6)
    assert np.allclose(dfprofile["pos_y_min"], -0.5, rtol=1e-6)


def test_from_e2e_model_interpolation_weights_of_an_empty_cell() -> None:
    """A cell that both models leave empty gets a weight of zero for each model, and not a NaN.

    The weight was a plain quotient of the density of one model and the combined density. That gave
    0/0 for such a cell, and the NaN went into each isotope column and into model.txt.
    """
    from artistools.inputmodel.from_e2e_model import get_model_interpolation_weights

    dfweights = get_model_interpolation_weights(dens_3D=pl.Series([2.0, 0.0]), dens_2D=pl.Series([6.0, 0.0]))

    assert not any(dfweights.select(cs.float().is_nan().any()).row(0))
    assert dfweights["rho"].to_list() == [8.0, 0.0]
    assert dfweights["beta_3D"].to_list() == [0.25, 0.0]
    assert dfweights["beta_2D"].to_list() == [0.75, 0.0]


def test_from_e2e_model_2d_equatorial_symmetry_contributions_name_the_cells_with_mass(tmp_path: Path) -> None:
    """The particle contributions of a 2D model with equatorial symmetry must name the cells of both halves.

    The contributions numbered the rows of the half grid from the equator, but the model reflects that grid in z.
    Thus every contribution named a cell below the equator, which was not its mirror cell, and the upper half had none.
    """
    from artistools.inputmodel.from_e2e_model import get_grid
    from artistools.inputmodel.from_e2e_model import t_model_init_s
    from artistools.inputmodel.from_e2e_model import z_reflect

    rng = np.random.default_rng(seed=1)
    ntraj = 40
    ntimes = 8
    datpath = tmp_path / "e2emodel.npz"
    isopath = tmp_path / "iso_table.npy"
    # every polar angle is below pi / 2, thus the model takes equatorial symmetry
    np.savez(
        datpath,
        pos=np.column_stack([rng.uniform(0.05, 0.3, ntraj), rng.uniform(0.2, 1.4, ntraj)]),
        idx=np.arange(1, ntraj + 1, dtype=float),
        state=np.zeros(ntraj),
        mass=np.full(ntraj, 1e-3),
        qdot=np.full((ntraj, ntimes), 1e10),
        hnuloss=np.zeros((ntraj, ntimes)),
        time=np.linspace(0.0, 2.0 * t_model_init_s, ntimes),
        nz=rng.uniform(0.01, 0.2, size=(ntraj, 2)),
        t5out=np.column_stack([np.zeros((ntraj, 4)), np.full(ntraj, 0.3)]),
    )
    np.save(isopath, np.array([[2.0, 2.0], [30.0, 26.0]]))

    _rgrid, _zgrid, rhoint, _xint, _iso, _q, _ye, eqsymfac, dfcontribs = get_grid(
        datpath,
        isopath,
        0.4,
        model_dim=2,
        grid_dims=np.array([4, 8]),
        nodynej=False,
        nohmns=False,
        notorus=False,
        no_nu_trapping=False,
    )

    assert eqsymfac == 2
    # map_to_artis reflects the half grid in z, and the cell numbers follow that full grid in Fortran order
    rhofull = z_reflect(rhoint).flatten(order="F")
    cellswithmass = {int(index) + 1 for index in np.nonzero(rhofull > 0.0)[0]}
    assert set(dfcontribs["cellindex"].to_list()) == cellswithmass


def test_from_e2e_model_3d_grid_takes_the_arguments(tmp_path: Path) -> None:
    """The 3D mapping grid comes from -vmax_on_c and -ngridx/y/z.

    The 3D branch held vminr, vmaxr, nvr = -0.5, 0.5, 50, thus the arguments had no effect, and a
    grid size other than 50 stopped with a broadcast error.
    """
    from artistools.constants import C_cm_per_s
    from artistools.inputmodel.from_e2e_model import get_grid
    from artistools.inputmodel.from_e2e_model import t_model_init_s

    rng = np.random.default_rng(seed=1)
    ntraj = 4
    ntimes = 8
    datpath = tmp_path / "e2emodel.npz"
    isopath = tmp_path / "iso_table.npy"
    np.savez(
        datpath,
        pos=np.column_stack([np.array([0.10, 0.15, 0.20, 0.25]), np.array([0.5, 1.0, 2.0, 2.5])]),
        idx=np.arange(1, ntraj + 1, dtype=float),
        state=np.array([-1.0, 0.0, 0.0, 1.0]),
        mass=np.full(ntraj, 1e-3),
        qdot=np.full((ntraj, ntimes), 1e10),
        hnuloss=np.zeros((ntraj, ntimes)),
        time=np.linspace(0.0, 2.0 * t_model_init_s, ntimes),
        nz=rng.uniform(0.01, 0.2, size=(ntraj, 2)),
        t5out=np.column_stack([np.zeros((ntraj, 4)), np.full(ntraj, 0.3)]),
    )
    np.save(isopath, np.array([[2.0, 2.0], [30.0, 26.0]]))

    vmax_on_c = 0.4
    ncells = 6
    x3d_min, _y3d_min, z3d_min, rhoint, xint, _iso, _q, _ye, _bs, eqsymfac, _dfcontribs = get_grid(
        datpath,
        isopath,
        vmax_on_c,
        model_dim=3,
        grid_dims=np.array([ncells, ncells, ncells]),
        nodynej=False,
        nohmns=False,
        notorus=False,
        no_nu_trapping=False,
    )

    assert eqsymfac == 1
    assert rhoint.shape == (ncells, ncells, ncells)
    assert xint.shape == (2, ncells, ncells, ncells)

    # the cell edges span -vmax to +vmax, thus the smallest left edge is -vmax and the largest is
    # one cell width below +vmax
    xmax_cm = vmax_on_c * C_cm_per_s * t_model_init_s
    wid_cm = 2.0 * xmax_cm / ncells
    assert np.isclose(x3d_min.min(), -xmax_cm, rtol=1e-10)
    assert np.isclose(x3d_min.max(), xmax_cm - wid_cm, rtol=1e-10)
    assert np.isclose(z3d_min.min(), -xmax_cm, rtol=1e-10)
    assert np.amax(rhoint) > 0.0

    # an odd count in x or y puts a cell centre on the symmetry axis, where the mapping divides by
    # the cylindrical radius. The command must reject such a grid with a message
    with pytest.raises(ValueError, match="must be even"):
        get_grid(
            datpath,
            isopath,
            vmax_on_c,
            model_dim=3,
            grid_dims=np.array([5, 6, 6]),
            nodynej=False,
            nohmns=False,
            notorus=False,
            no_nu_trapping=False,
        )


def test_maketardismodel_maxatomicnumber_from_command_line(tmp_path: Path) -> None:
    """-maxatomicnumber reads an integer from the command line.

    The argument had no type, thus the command compared a string with an integer and stopped with
    TypeError before this test existed.
    """
    at.inputmodel.to_tardis.main(argsraw=["-inputpath", str(modelpath), "-o", str(tmp_path), "-maxatomicnumber", "26"])

    csvypath = tmp_path / f"{at.get_model_name(modelpath)}.csvy"
    assert csvypath.is_file()
    csvytext = csvypath.read_text(encoding="utf-8")
    assert csvytext.startswith("---\n")
    assert "velocity,density,t_rad,dilution_factor" in csvytext
    assert "Fe" in csvytext
    assert "Co" not in csvytext
    assert "Ni" not in csvytext


def test_an_unreadable_model_cache_is_not_deleted(tmp_path: Path) -> None:
    """A rejected cache must stay in place, because a rival process possibly replaced it.

    The reader deleted an unreadable file with a bare unlink, outside the identity rule of
    write_parquet_atomic. That unlink can remove the fresh cache that a rival process installed.
    """
    from artistools.inputmodel.core import scan_parquet_cache

    parquetfilepath = tmp_path / "model.txt.parquet.tmp"
    parquetfilepath.write_bytes(b"not parquet")

    assert scan_parquet_cache(parquetfilepath, textsource_mtime=1.0) is None
    assert parquetfilepath.is_file(), "the unreadable cache must stay for the identity-checked rewrite"


def test_downscale_mass_fractions_matches_cell_loop() -> None:
    """The reshape and sum must give the same result as a loop over the cells of each block."""
    from artistools.inputmodel.downscale3dgrid import downscale_cell_sums
    from artistools.inputmodel.downscale3dgrid import downscale_mass_fractions

    rng = np.random.default_rng(seed=3)
    grid, merge, nabund = 6, 3, 4
    smallgrid = grid // merge
    rho = rng.uniform(0.0, 1.0, size=(grid, grid, grid))
    rho[:merge, :merge, :merge] = 0.0
    massfracs = rng.uniform(0.0, 1.0, size=(grid, grid, grid, nabund))

    rho_small_expected = np.zeros((smallgrid, smallgrid, smallgrid))
    massfracs_small_expected = np.zeros((smallgrid, smallgrid, smallgrid, nabund))
    for x, y, z in itertools.product(range(smallgrid), repeat=3):
        for xx, yy, zz in itertools.product(range(merge), repeat=3):
            rho_cell = rho[x * merge + xx, y * merge + yy, z * merge + zz]
            rho_small_expected[x, y, z] += rho_cell
            massfracs_small_expected[x, y, z, :] += massfracs[x * merge + xx, y * merge + yy, z * merge + zz] * rho_cell
        if rho_small_expected[x, y, z] > 0:
            massfracs_small_expected[x, y, z, :] /= rho_small_expected[x, y, z]

    assert np.allclose(downscale_cell_sums(rho, merge), rho_small_expected, rtol=1e-12, atol=0.0)
    assert np.allclose(downscale_mass_fractions(massfracs, rho, merge), massfracs_small_expected, rtol=1e-12, atol=0.0)
    assert np.all(downscale_mass_fractions(massfracs, rho, merge)[0, 0, 0] == 0)


def test_remap_mass_weighted_quantity_matches_cell_loop() -> None:
    """The reshape to the merge axes must give the same result as the loop over the fine cells of each coarse cell."""
    from artistools.inputmodel.from_e2e_model import remap_mass_weighted_quantity

    rng = np.random.default_rng(seed=5)
    red_fact, n_r_new, n_z_new = 2, 3, 4
    n_r_old, n_z_old = red_fact * n_r_new, red_fact * n_z_new
    delta_r, delta_z = 1.5, 1.5
    dfmodel = pl.DataFrame({
        "mass_g": rng.uniform(1.0, 2.0, size=n_r_old * n_z_old),
        "Ye": rng.uniform(0.1, 0.5, size=n_r_old * n_z_old),
    })

    expected_rho = np.zeros(n_r_new * n_z_new)
    expected_ye = np.zeros(n_r_new * n_z_new)
    for new_z in range(1, n_z_new + 1):
        for new_r in range(1, n_r_new + 1):
            old_r_indices = np.arange(red_fact * (new_r - 1) + 1, red_fact * new_r + 1)
            old_z_indices = np.arange(red_fact * (new_z - 1), red_fact * new_z) * n_r_old
            old_idxs = np.add.outer(old_r_indices, old_z_indices).flatten() - 1
            masses = dfmodel["mass_g"].to_numpy()[old_idxs]
            new_cell_idx = new_r + n_r_new * (new_z - 1)
            r_i, r_o = (new_r - 1) * delta_r, new_r * delta_r
            expected_rho[new_cell_idx - 1] = masses.sum() / (np.pi * (r_o**2 - r_i**2) * delta_z)
            expected_ye[new_cell_idx - 1] = np.average(dfmodel["Ye"].to_numpy()[old_idxs], weights=masses)

    remap_args = (red_fact, n_r_new, n_z_new, n_r_old, delta_r, delta_z)
    assert np.allclose(remap_mass_weighted_quantity(dfmodel, "mass_g", *remap_args), expected_rho, rtol=1e-12, atol=0.0)
    assert np.allclose(remap_mass_weighted_quantity(dfmodel, "Ye", *remap_args), expected_ye, rtol=1e-12, atol=0.0)


def test_remap_mass_weighted_quantity_gives_zero_for_an_empty_coarse_cell() -> None:
    """A coarse cell in the empty corner of a model holds no mass, thus its average is zero and not an error."""
    from artistools.inputmodel.from_e2e_model import remap_mass_weighted_quantity

    red_fact, n_r_new, n_z_new = 2, 2, 1
    n_r_old = red_fact * n_r_new
    masses = np.array([0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0, 1.0])
    dfmodel = pl.DataFrame({"mass_g": masses, "Ye": np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])})

    ye = remap_mass_weighted_quantity(dfmodel, "Ye", red_fact, n_r_new, n_z_new, n_r_old, 1.0, 1.0)

    assert ye[0] == 0
    assert ye[1] == pytest.approx((0.3 * 1.0 + 0.4 * 3.0 + 0.3 * 1.0 + 0.4 * 1.0) / 6.0)


def test_scan_modeldata_regenerates_a_cache_with_malformed_metadata(tmp_path: Path) -> None:
    """A cache with a current stamp but an unreadable modelmeta_json must be regenerated from model.txt."""
    modelfile = tmp_path / "model.txt"
    shutil.copy(at.firstexisting("model.txt", folder=modelpath, tryzipped=True), modelfile)

    dfmodel_expected, modelmeta_expected = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)

    # a cache whose stamps are current, thus only the malformed json can reject it
    parquetfile = tmp_path / "model.txt.parquet.tmp"
    at.misc.write_parquet_atomic(
        dfmodel_expected.collect(),
        parquetfile,
        metadata={
            "cacheversion": str(at.inputmodel.core.CACHEVERSION),
            "textsource_mtime": str(modelfile.stat().st_mtime),
            "modelmeta_json": "{this is not json",
        },
    )
    assert parquetfile.is_file()

    dfmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)

    assert modelmeta == modelmeta_expected
    pltest.assert_frame_equal(dfmodel.collect(), dfmodel_expected.collect())


def test_describeinputmodel_prints_the_selected_cell(capsys: pytest.CaptureFixture[str]) -> None:
    """The -cell option must print the row of the cell, and not the plan of the query."""
    at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=[modelpath], cell=0)

    output = capsys.readouterr().out
    assert "Selected single cell mgi 0:" in output
    assert "naive plan" not in output
    # the table has a header row and one row for the one cell
    tablelines = output.split("Selected single cell mgi 0:\n", 1)[1].splitlines()
    assert tablelines[0].startswith("inputcellid ")
    assert tablelines[1].split()[0] == "1"


def test_plotinitialcomposition_reads_the_model_folder_last(tmp_path: Path) -> None:
    """The model folder is the last positional argument, and rho is the default variable."""
    at.inputmodel.plotinitialcomposition.main(argsraw=["-o", str(tmp_path), str(modelpath_3d)])

    assert (tmp_path / "plotcomposition_rho.pdf").is_file()


def test_plotinitialcomposition_takes_a_zero_floor_value(tmp_path: Path) -> None:
    """A floor value of zero is a value, thus the colour scale must start there and not at the default."""
    import matplotlib.axes as mplax

    with mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True) as mockimshow:
        at.inputmodel.plotinitialcomposition.main(
            argsraw=["-modelpath", str(modelpath_3d), "-o", str(tmp_path), "-floorval", "0", "rho"]
        )

    assert mockimshow.call_args.kwargs["vmin"] == 0


def test_plotinitialcomposition_log_scale_takes_a_linear_floor_value(tmp_path: Path) -> None:
    """With --logcolorscale, the floor is a linear density, and the colour scale starts at its logarithm."""
    import matplotlib.axes as mplax

    with mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True) as mockimshow:
        at.inputmodel.plotinitialcomposition.main(
            argsraw=[
                "-modelpath",
                str(modelpath_3d),
                "-o",
                str(tmp_path),
                "-floorval",
                "1e-12",
                "--logcolorscale",
                "rho",
            ]
        )

    assert mockimshow.call_args.kwargs["vmin"] == pytest.approx(-12.0)

    # a floor of zero has no logarithm, thus it is an error and not a colour scale with vmin above vmax
    with pytest.raises(ValueError, match="must be positive"):
        at.inputmodel.plotinitialcomposition.main(
            argsraw=["-modelpath", str(modelpath_3d), "-o", str(tmp_path), "-floorval", "0", "--logcolorscale", "rho"]
        )


def test_model_reader_renames_the_cellye_column_of_an_old_model(tmp_path: Path) -> None:
    """An old model.txt names the electron fraction cellYe, and scan_modeldata returns the column as Ye."""
    dfmodel = pl.DataFrame({
        "inputcellid": [1, 2],
        "vel_r_max_kmps": [1000.0, 2000.0],
        "logrho": [-10.0, -11.0],
        "X_Fegroup": [1.0, 1.0],
        "X_Ni56": [0.5, 0.4],
        "X_Co56": [0.0, 0.0],
        "X_Fe52": [0.0, 0.0],
        "X_Cr48": [0.0, 0.0],
        "Ye": [0.3, 0.4],
    })
    at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta={"dimensions": 1, "t_model_init_days": 1.0})
    modelfile = decompress_file(tmp_path / "model.txt.zst")
    modeltext = modelfile.read_text(encoding="utf-8")
    assert " Ye\n" in modeltext
    modelfile.write_text(modeltext.replace(" Ye\n", " cellYe\n"), encoding="utf-8")

    lzdfmodel, _modelmeta = at.inputmodel.scan_modeldata(tmp_path)

    columns = lzdfmodel.collect_schema().names()
    assert "Ye" in columns
    assert "cellYe" not in columns
    assert lzdfmodel.select("Ye").collect().to_series().to_list() == pytest.approx([0.3, 0.4])


@pytest.mark.parametrize("sourcemodelpath", [modelpath, modelpath_3d])
def test_scan_modeldata_gives_only_the_file_columns(sourcemodelpath: Path) -> None:
    """Give only the file columns. A change to a file column then keeps each derived column current."""
    lzdfmodel_file, _ = at.inputmodel.core.scan_modelfile_text(sourcemodelpath / "model.txt")
    lzdfmodel, _ = at.inputmodel.scan_modeldata(sourcemodelpath)
    assert lzdfmodel.collect_schema().names() == [*lzdfmodel_file.collect_schema().names(), "modelgridindex"]


@pytest.mark.parametrize("sourcemodelpath", [modelpath, modelpath_3d])
def test_add_derived_cols_calculates_each_column_again(sourcemodelpath: Path) -> None:
    """A second call after a change to the density gives current values, and it adds no column."""
    lzdfmodel, modelmeta = get_derived_modeldata(sourcemodelpath)
    lzdfmodel_changed = lzdfmodel.with_columns(pl.col("rho") * 10.0)
    lzdfmodel_derivedagain = at.inputmodel.add_derived_cols_to_modeldata(lzdfmodel_changed, modelmeta=modelmeta)

    assert lzdfmodel_derivedagain.collect_schema().names() == lzdfmodel.collect_schema().names()
    mass_g = lzdfmodel.select(pl.sum("mass_g")).collect().item()
    # the mass_g of the changed dataframe is out of date, and the second derivation gives the current value
    assert lzdfmodel_changed.select(pl.sum("mass_g")).collect().item() == pytest.approx(mass_g)
    assert lzdfmodel_derivedagain.select(pl.sum("mass_g")).collect().item() == pytest.approx(10.0 * mass_g, rel=1e-5)


def test_add_derived_cols_recalculates_the_inner_velocity_of_a_1d_shell() -> None:
    """A change to the outer velocities must also change the inner velocities, which come from the row before."""
    modelmeta = {"dimensions": 1, "t_model_init_days": 1.0}
    dfmodel = pl.DataFrame({
        "inputcellid": [1, 2, 3],
        "vel_r_max_kmps": [1000.0, 2000.0, 3000.0],
        "logrho": [-10.0] * 3,
    })
    lzdfmodel = at.inputmodel.add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)
    lzdfmodel_stretched = at.inputmodel.add_derived_cols_to_modeldata(
        lzdfmodel.with_columns(pl.col("vel_r_max_kmps") * 2.0), modelmeta=modelmeta
    )
    dfvelocities = lzdfmodel_stretched.select("vel_r_min_kmps", "vel_r_max_kmps").collect()
    assert dfvelocities["vel_r_min_kmps"].to_list() == [0.0, *dfvelocities["vel_r_max_kmps"].to_list()[:-1]]


def test_add_derived_cols_before_a_filter_keeps_the_inner_velocity_of_a_1d_shell() -> None:
    """A filter after the derivation keeps the inner velocity of a shell, which comes from the row before."""
    modelmeta = {"dimensions": 1, "t_model_init_days": 1.0}
    dfmodel = pl.DataFrame({
        "inputcellid": [1, 2, 3],
        "vel_r_max_kmps": [1000.0, 2000.0, 3000.0],
        "logrho": [-10.0, -99.0, -10.0],
    })
    dfshells = (
        at.inputmodel
        .add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)
        .filter(pl.col("logrho") > -98)
        .select("inputcellid", "vel_r_min_kmps")
        .collect()
    )
    assert dfshells["vel_r_min_kmps"].to_list() == [0.0, 2000.0]


def test_dimension_reduce_takes_a_model_with_a_volume_column() -> None:
    """A dataframe that already holds volume, e.g. from an earlier derivation, must reduce without an error."""
    lzdfmodel, modelmeta = get_derived_modeldata(modelpath_3d)
    dfmodel_out, _, _, _ = at.inputmodel.dimension_reduce_model(
        dfmodel=lzdfmodel.select(
            cs.by_name(at.inputmodel.scan_modeldata(modelpath_3d)[0].collect_schema().names()) | cs.by_name("volume")
        ),
        outputdimensions=1,
        modelmeta=modelmeta.copy(),
    )
    assert "volume" not in dfmodel_out.columns


@pytest.mark.parametrize("sourcemodelpath", [modelpath, modelpath_3d])
def test_save_modeldata_writes_the_same_columns_with_the_derived_columns(sourcemodelpath: Path, tmp_path: Path) -> None:
    """model.txt must be the same from the file columns alone and from a dataframe that holds every derived column.

    The dataframe also holds rho and logrho, and pos_ and vel_ columns of other dimensions.
    """
    lzdfmodel_file, _ = at.inputmodel.core.scan_modelfile_text(sourcemodelpath / "model.txt")
    lzdfmodel, modelmeta = get_derived_modeldata(sourcemodelpath)
    lzdfmodel_allderived = lzdfmodel.with_columns(
        pl.lit(1.0).alias(col) for col in ("pos_rcyl_mid", "pos_x_mid", "vel_x_mid_on_c", "kinetic_en_erg_x")
    )
    assert {"rho", "logrho"} <= set(lzdfmodel_allderived.collect_schema().names())

    dfmodel_written = {}
    for label, dfmodel in (("filecolumns", lzdfmodel_file), ("allderived", lzdfmodel_allderived)):
        outpath = tmp_path / label
        outpath.mkdir()
        at.inputmodel.save_modeldata(dfmodel.collect(), outpath=outpath, modelmeta=modelmeta.copy())
        dfmodel_written[label] = at.inputmodel.core.scan_modelfile_text(outpath / "model.txt.zst")[0].collect()

    pltest.assert_frame_equal(dfmodel_written["filecolumns"], dfmodel_written["allderived"])


@pytest.mark.parametrize(
    ("extracols", "expectedcustomcols"),
    [(None, ["Ye", "tracercount"]), (["mycolumn"], ["Ye", "tracercount", "mycolumn"])],
)
def test_save_modeldata_writes_the_extra_columns(
    extracols: list[str] | None, expectedcustomcols: list[str], tmp_path: Path
) -> None:
    """model.txt always gets Ye, q, and tracercount. It gets a different custom column only if a caller names it.

    The custom columns keep the order of the dataframe.
    """
    dfmodel = pl.DataFrame({
        "inputcellid": [1, 2],
        "vel_r_max_kmps": [1000.0, 2000.0],
        "logrho": [-10.0, -11.0],
        "X_Fegroup": [1.0, 1.0],
        "X_Ni56": [0.5, 0.4],
        "X_Sr89": [0.1, 0.2],
        "Ye": [0.3, 0.4],
        "tracercount": [3, 4],
        "mycolumn": [5.0, 6.0],
    })
    at.inputmodel.save_modeldata(
        dfmodel,
        outpath=tmp_path,
        modelmeta={"dimensions": 1, "t_model_init_days": 1.0},
        extracols=() if extracols is None else extracols,
    )

    lzdfmodel_written, _ = at.inputmodel.core.scan_modelfile_text(tmp_path / "model.txt.zst")
    standardcols = at.inputmodel.core.get_standard_columns("spherical1d")
    assert lzdfmodel_written.collect_schema().names() == [*standardcols, "X_Sr89", *expectedcustomcols]


@pytest.mark.parametrize(("sourcemodelpath", "outputdimensions"), [(modelpath_3d, 1), (modelpath, 0)])
def test_dimension_reduce_takes_the_other_density_column(sourcemodelpath: Path, outputdimensions: int) -> None:
    """A model can give rho in place of logrho, or logrho in place of rho, and the reduction makes the other one."""
    lzdfmodel, modelmeta = at.inputmodel.core.scan_modelfile_text(sourcemodelpath / "model.txt")
    dfmodel = lzdfmodel.collect()
    densitycol, otherdensitycol = ("rho", "logrho") if modelmeta["dimensions"] == 3 else ("logrho", "rho")
    dfmodel_otherdensity = at.inputmodel.add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta).select(
        *[col for col in dfmodel.columns if col != densitycol], otherdensitycol
    )

    dfmodel_out, _, _, _ = at.inputmodel.dimension_reduce_model(
        dfmodel=dfmodel, outputdimensions=outputdimensions, modelmeta=modelmeta.copy()
    )
    dfmodel_out_otherdensity, _, _, _ = at.inputmodel.dimension_reduce_model(
        dfmodel=dfmodel_otherdensity, outputdimensions=outputdimensions, modelmeta=modelmeta.copy()
    )

    pltest.assert_frame_equal(dfmodel_out, dfmodel_out_otherdensity, check_column_order=False, rel_tol=1e-6)


def test_model_files_with_dotted_names_keep_separate_caches(tmp_path: Path) -> None:
    """model_a.1.txt and model_a.2.txt must not share one parquet cache."""
    for variant, logrho in (("1", -10.0), ("2", -12.0)):
        dfmodel = pl.DataFrame({
            "inputcellid": [1, 2],
            "vel_r_max_kmps": [1000.0, 2000.0],
            "logrho": [logrho, logrho],
            "X_Fegroup": [1.0, 1.0],
            "X_Ni56": [0.5, 0.4],
            "X_Co56": [0.0, 0.0],
            "X_Fe52": [0.0, 0.0],
            "X_Cr48": [0.0, 0.0],
        })
        at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta={"dimensions": 1, "t_model_init_days": 1.0})
        (tmp_path / "model.txt.zst").rename(tmp_path / f"model_a.{variant}.txt.zst")
        # a cache file that exists makes the reader write a new cache, whatever the size of the model
        (tmp_path / f"model_a.{variant}.txt.parquet.tmp").touch()

    lzdfmodel1, _meta1 = at.inputmodel.scan_modeldata(tmp_path / "model_a.1.txt")
    lzdfmodel2, _meta2 = at.inputmodel.scan_modeldata(tmp_path / "model_a.2.txt")

    assert lzdfmodel1.select("logrho").collect().to_series().to_list() == pytest.approx([-10.0, -10.0])
    assert lzdfmodel2.select("logrho").collect().to_series().to_list() == pytest.approx([-12.0, -12.0])
    assert (tmp_path / "model_a.1.txt.parquet.tmp").stat().st_size > 0
    assert (tmp_path / "model_a.2.txt.parquet.tmp").stat().st_size > 0


def test_save_modeldata_deletes_an_old_uncompressed_file(tmp_path: Path) -> None:
    """If model.txt and model.txt.zst both exist, a reader uses model.txt. Thus an old uncompressed file must go."""
    for logrho in (-10.0, -12.0):
        dfmodel = pl.DataFrame({
            "inputcellid": [1],
            "vel_r_max_kmps": [1000.0],
            "logrho": [logrho],
            "X_Fegroup": [1.0],
            "X_Ni56": [0.5],
            "X_Co56": [0.0],
            "X_Fe52": [0.0],
            "X_Cr48": [0.0],
        })
        at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta={"dimensions": 1, "t_model_init_days": 1.0})
        at.inputmodel.save_empty_abundance_file(npts_model=1, outputfilepath=tmp_path)
        if logrho == -10:
            for filename in ("model.txt", "abundances.txt"):
                decompress_file(tmp_path / f"{filename}.zst")

    assert sorted(path.name for path in tmp_path.iterdir() if ".parquet" not in path.name) == [
        "abundances.txt.zst",
        "model.txt.zst",
    ]
    lzdfmodel, _ = at.inputmodel.scan_modeldata(tmp_path, get_elemabundances=True)
    assert lzdfmodel.select("logrho").collect().item() == pytest.approx(-12.0)

    # a name with a different compression extension gave model.txt.gz.zst, and the writer deleted model.txt.gz
    (tmp_path / "model.txt.gz").write_bytes(b"")
    at.inputmodel.save_modeldata(
        dfmodel, outpath=tmp_path / "model.txt.gz", modelmeta={"dimensions": 1, "t_model_init_days": 1.0}
    )
    assert sorted(path.name for path in tmp_path.iterdir() if ".parquet" not in path.name) == [
        "abundances.txt.zst",
        "model.txt.zst",
    ]


def test_command_asks_before_it_overwrites_a_compressed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An output name also finds its compressed copy, and the command stops before the work if nobody agrees."""
    (tmp_path / "abundances.txt.zst").write_bytes(b"")
    filenames = ("model.txt", "abundances.txt")

    # stdin of pytest is not a terminal, thus nobody can answer
    with pytest.raises(SystemExit):
        at.misc.confirm_overwrite([tmp_path], filenames, force=False)
    assert f"{tmp_path / 'abundances.txt.zst'} exists" in capsys.readouterr().err

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    for reply, agrees in (("y", True), ("n", False), ("", False)):
        with mock.patch("builtins.input", return_value=reply):
            if agrees:
                at.misc.confirm_overwrite([tmp_path], filenames, force=False)
            else:
                with pytest.raises(SystemExit):
                    at.misc.confirm_overwrite([tmp_path], filenames, force=False)

    at.misc.confirm_overwrite([tmp_path], filenames, force=True)

    from artistools.inputmodel import shen2018

    with mock.patch("artistools.inputmodel.shen2018.read_wsv", side_effect=AssertionError("the work started")):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        capsys.readouterr()
        with pytest.raises(SystemExit):
            shen2018.main(argsraw=["-o", str(tmp_path)])
        assert "Give -f or --force to overwrite them" in capsys.readouterr().err


def test_plotinitialcomposition_floor_value_keeps_the_hidden_empty_cells(tmp_path: Path) -> None:
    """--hideemptycells must still hide an empty cell when -floorval also applies.

    The clamp of the floor gave a plain array. Thus every cell that the mask hid came back on the
    plot at the floor value, and only this combination of the options showed it.
    """
    import matplotlib.axes as mplax

    with mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True) as mockimshow:
        at.inputmodel.plotinitialcomposition.main(
            argsraw=[
                "-modelpath",
                str(modelpath_3d),
                "-o",
                str(tmp_path),
                "-floorval",
                "1e-12",
                "--logcolorscale",
                "--hideemptycells",
                "rho",
            ]
        )

    colorscale = mockimshow.call_args.args[1]
    assert np.ma.isMaskedArray(colorscale), "the mask of --hideemptycells must survive the floor clamp"
    assert np.ma.getmaskarray(colorscale).any(), "the 3D test model holds an empty cell to hide"


def test_plotinitialabundances_takes_a_velocity_with_a_unit(tmp_path: Path) -> None:
    """-vmin and -vmax take a value with a unit, as in plotopacity, and a bare number is still a fraction of c.

    plotopacity takes "0.1c or 5000km/s", and plotinitialabundances took only a bare float in c. A bare number must
    give the same float as before, because a cell exactly at a limit can otherwise change its side.
    """
    from artistools.constants import C_cm_per_s
    from artistools.constants import km_to_cm
    from artistools.inputmodel import plotinitialabundances

    parser = argparse.ArgumentParser()
    plotinitialabundances.addargs(parser)
    assert np.isclose(parser.parse_args(["-vmin", "0.1"]).vmin, 0.1, rtol=0.0, atol=0.0)
    assert np.isclose(parser.parse_args(["-vmin", "0.1c"]).vmin, 0.1, rtol=0.0, atol=0.0)
    vmax_kmps = 0.02 * C_cm_per_s / km_to_cm
    assert np.isclose(parser.parse_args(["-vmax", f"{vmax_kmps!r}km/s"]).vmax, 0.02, rtol=1e-12, atol=0.0)
    with pytest.raises(SystemExit):
        parser.parse_args(["-vmin", "fast"])

    # a keyword of main gives the float directly, and the parser type converts only a text
    with mock.patch.object(
        plotinitialabundances, "get_nuclide_massfractions", wraps=plotinitialabundances.get_nuclide_massfractions
    ) as reader:
        plotinitialabundances.main(argsraw=["-vmax", "0.02c"], modelpath=[modelpath_classic_3d], outputfile=tmp_path)
        plotinitialabundances.main(argsraw=[], modelpath=[modelpath_classic_3d], vmax=0.02, outputfile=tmp_path)
    assert [call.kwargs["vmax"] for call in reader.call_args_list] == [0.02, 0.02]


def test_plotinitialabundances_filters_cells_by_velocity_and_polar_angle(tmp_path: Path) -> None:
    """A velocity range and a polar angle range select the expected cells of a 1D, a 2D, and a 3D model.

    A selection with no cell raises an error. A selection of empty cells raises a different error that
    names the mass. A 1D model rejects a polar angle range.
    """
    from artistools.inputmodel import get_cell_selection
    from artistools.inputmodel.plotinitialabundances import get_nuclide_massfractions

    def massfrac_ni56(dfcells: pl.DataFrame) -> float:
        return float((dfcells["X_Ni56"] * dfcells["mass_g"]).sum()) / float(dfcells["mass_g"].sum())

    def selected_massfrac_ni56(dfnuclides: pl.DataFrame) -> float:
        return float(dfnuclides.filter(pl.col("nuclide") == "X_Ni56")["massfraction"].item())

    dfmodel = get_derived_modeldata(modelpath_classic_3d)[0].collect()

    dfall = get_nuclide_massfractions(modelpath_classic_3d)
    assert np.isclose(selected_massfrac_ni56(dfall), massfrac_ni56(dfmodel))

    dfupper = get_nuclide_massfractions(modelpath_classic_3d, thetamax=90.0)
    cellsupper = dfmodel.filter(pl.col("vel_z_mid_on_c") >= 0.0)
    assert np.isclose(selected_massfrac_ni56(dfupper), massfrac_ni56(cellsupper))

    dfslow = get_nuclide_massfractions(modelpath_classic_3d, vmax=0.02, thetamin=90.0)
    cellsslow = dfmodel.filter((pl.col("vel_z_mid_on_c") <= 0.0) & (pl.col("vel_r_mid_on_c") <= 0.02))
    assert 0 < len(cellsslow) < len(dfmodel)
    assert np.isclose(selected_massfrac_ni56(dfslow), massfrac_ni56(cellsslow))

    with pytest.raises(ValueError, match=r"Every cell of .* is outside the selection \(vmin=0.5\)"):
        get_nuclide_massfractions(modelpath_classic_3d, vmin=0.5)

    # the corner cells beyond 0.15 c are empty, thus the message must name the mass and not the range
    assert len(dfmodel.filter(pl.col("vel_r_mid_on_c") >= 0.15)) > 0
    with pytest.raises(ValueError, match=r"The 8 selected cells of .* hold no mass \(vmin=0.15\)"):
        get_nuclide_massfractions(modelpath_classic_3d, vmin=0.15)

    with pytest.raises(ValueError, match="vmin must be less than or equal to vmax"):
        get_nuclide_massfractions(modelpath_classic_3d, vmin=0.1, vmax=0.01)
    with pytest.raises(ValueError, match="thetamax must be between 0 and 180 degrees"):
        get_nuclide_massfractions(modelpath_classic_3d, thetamax=190.0)

    modelpath_1d = testdatapath / "test-classicmode_1d"
    dfmodel_1d = get_derived_modeldata(modelpath_1d)[0].collect()
    dffast_1d = get_nuclide_massfractions(modelpath_1d, vmin=0.01)
    cellsfast_1d = dfmodel_1d.filter(pl.col("vel_r_mid_on_c") >= 0.01)
    assert 0 < len(cellsfast_1d) < len(dfmodel_1d)
    assert np.isclose(selected_massfrac_ni56(dffast_1d), massfrac_ni56(cellsfast_1d))
    with pytest.raises(ValueError, match=r"needs a 2D or 3D model, but .* is 1D"):
        get_nuclide_massfractions(modelpath_1d, thetamax=90.0)

    write_2d_model(tmp_path, ncoordgridrcyl=4, ncoordgridz=6, vmax_cmps=1.0e9, t_model_days=1.0, x_ni56_upper=0.1)
    dfmodel_2d = get_derived_modeldata(tmp_path)[0].collect()
    assert np.isclose(selected_massfrac_ni56(get_nuclide_massfractions(tmp_path, thetamax=90.0)), 0.1)
    assert np.isclose(selected_massfrac_ni56(get_nuclide_massfractions(tmp_path, thetamin=90.0)), 0.4)
    dfslow_2d = get_nuclide_massfractions(tmp_path, vmax=0.02, thetamin=45.0, thetamax=135.0)
    cellsslow_2d = dfmodel_2d.filter(
        (pl.col("vel_r_mid_on_c") <= 0.02) & (pl.col("vel_z_mid_on_c").abs() <= pl.col("vel_rcyl_mid_on_c"))
    )
    assert 0 < len(cellsslow_2d) < len(dfmodel_2d)
    assert np.isclose(selected_massfrac_ni56(dfslow_2d), massfrac_ni56(cellsslow_2d))

    # a grid with an odd cell count on each axis has a cell at the origin. No polar angle range keeps that cell
    dfmodel_origin, modelmeta_origin = get_empty_3d_model(ncoordgrid=3, vmax=1e9, t_model_init_days=1.0)
    dfcells_origin = at.inputmodel.add_derived_cols_to_modeldata(dfmodel_origin, modelmeta=modelmeta_origin).collect()
    assert len(dfcells_origin.filter(pl.col("vel_r_mid_on_c") == 0)) == 1
    assert len(dfcells_origin.filter(get_cell_selection())) == 27
    assert len(dfcells_origin.filter(get_cell_selection(thetamin=0.0))) == 26
    assert len(dfcells_origin.filter(get_cell_selection(thetamax=180.0))) == 26


def test_plotinitialabundances_bounds_keep_the_cells_on_the_bound(tmp_path: Path) -> None:
    """A cell whose midpoint lies on a bound stays inside the range, but the columns are Float32.

    A 2D grid of square cells has a diagonal of cells at 45 degrees. Their Float32 angles differ from 45
    degrees by approximately 1e-5 degrees. Polars casts a velocity bound down to Float32 for the
    comparison, thus each velocity probe below is one Float32 step off the bound.
    """
    from artistools.inputmodel import get_cell_selection

    write_2d_model(tmp_path, ncoordgridrcyl=4, ncoordgridz=8, vmax_cmps=1.0e9, t_model_days=1.0)
    dfcells = get_derived_modeldata(tmp_path)[0].collect()
    oncone = dfcells.filter(np.isclose(dfcells["vel_z_mid_on_c"].abs(), dfcells["vel_rcyl_mid_on_c"]))
    assert len(oncone) == 8
    assert len(oncone.filter(get_cell_selection(thetamax=45.0))) == 4
    assert len(oncone.filter(get_cell_selection(thetamin=45.0))) == 8
    assert len(oncone.filter(get_cell_selection(thetamin=135.0))) == 4
    assert len(oncone.filter(get_cell_selection(thetamin=45.0, thetamax=135.0))) == 8

    dfbound = pl.DataFrame({
        "vel_r_mid_on_c": pl.Series(
            [np.nextafter(np.float32(0.7), np.float32(0.0)), np.nextafter(np.float32(0.7), np.float32(1.0))],
            dtype=pl.Float32,
        ),
        "vel_z_mid_on_c": [0.0, 0.0],
    })
    assert len(dfbound.filter(get_cell_selection(vmin=0.7))) == 2
    assert len(dfbound.filter(get_cell_selection(vmax=0.7))) == 2


def test_dirbin_selects_the_cells_that_the_packet_binning_puts_in_the_bin(tmp_path: Path) -> None:
    """-dirbin selects the cells whose centre direction ARTIS puts in the bin. Thus it also applies the ϕ range.

    The selection took the cos θ range alone, thus -dirbin 93 gave the cells of each ϕ bin from 90 to 99. A 2D model
    is symmetric around the z axis, and its range has no ϕ range, thus the title gives the cells that it selects.
    """
    from artistools.inputmodel import get_cell_direction_selection
    from artistools.packets import add_packet_directions_lazypolars
    from artistools.packets import bin_packet_directions_polars

    lzmodel, modelmeta = at.scan_modeldata(modelpath_3d, printwarningsonly=True)
    dfcells = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("vel_r_mid_on_c") > 0.0)
        .collect()
    )
    dfbinned = (
        bin_packet_directions_polars(
            add_packet_directions_lazypolars(
                dfcells.select(
                    "modelgridindex",
                    dirx=pl.col("vel_x_mid_on_c"),
                    diry=pl.col("vel_y_mid_on_c"),
                    dirz=pl.col("vel_z_mid_on_c"),
                )
            )
        )
        .select("modelgridindex", "dirbin", "costhetabin")
        .collect()
    )
    for dirbin in (5, 34, 50, 59, 90, 93, 99):
        args = argparse.Namespace(dirbin=dirbin, costhetamin=None, costhetamax=None, average_over_phi_angle=False)
        directionrange = at.misc.get_cell_direction_range(args, quiet=True)
        assert directionrange is not None
        selected = dfcells.filter(get_cell_direction_selection(directionrange, dfcells.columns))["modelgridindex"]
        expected = dfbinned.filter(pl.col("dirbin") == dirbin)["modelgridindex"]
        assert len(expected) > 0
        assert sorted(selected) == sorted(expected)
        if dirbin == 93:
            assert at.misc.format_cell_direction_range(directionrange) == "0.8 ≤ cos θ ≤ 1.0, 6π/5 < ϕ ≤ 7π/5"

    args = argparse.Namespace(dirbin=90, costhetamin=None, costhetamax=None, average_over_phi_angle=True)
    directionrange = at.misc.get_cell_direction_range(args, quiet=True)
    assert directionrange is not None
    assert directionrange.phibin is None
    selected = dfcells.filter(get_cell_direction_selection(directionrange, dfcells.columns))["modelgridindex"]
    assert sorted(selected) == sorted(dfbinned.filter(pl.col("costhetabin") == 9)["modelgridindex"])

    write_2d_model(tmp_path, ncoordgridrcyl=4, ncoordgridz=6, vmax_cmps=1.0e9, t_model_days=1.0)
    args = argparse.Namespace(dirbin=93, costhetamin=None, costhetamax=None, average_over_phi_angle=False)
    directionrange = at.inputmodel.resolve_cell_direction_range(args, tmp_path, quiet=True)
    assert directionrange is not None
    assert directionrange.phibin is None
    assert at.misc.format_cell_direction_range(directionrange) == "0.8 ≤ cos θ ≤ 1.0"
    # a 2D model has no ϕ, thus a ϕ bin at each cos θ would select each cell
    args = argparse.Namespace(dirbin=3, costhetamin=None, costhetamax=None, average_over_theta_angle=True)
    with pytest.raises(SystemExit):
        at.inputmodel.resolve_cell_direction_range(args, tmp_path, quiet=True)
    directionrange = at.inputmodel.resolve_cell_direction_range(args, modelpath_3d, quiet=True)
    assert directionrange is not None
    assert at.misc.format_cell_direction_range(directionrange) == "6π/5 < ϕ ≤ 7π/5"
    directionrange = at.inputmodel.resolve_cell_direction_range(args, modelpath_3d, quiet=True)
    assert directionrange is not None
    assert directionrange.phibin is not None


def test_dirbin_puts_each_cell_of_an_odd_grid_in_one_bin() -> None:
    """Each cell of an odd grid is in the one bin that the packet binning gives, except the cell at the origin.

    An odd grid has cell centres on the edges of the bins, e.g. at ϕ = 0 and on the z axis. The selection included
    both edges, thus such a cell was in two bins, e.g. in bin 90 and in bin 95. On the 11 x 11 x 11 grid,
    vel_r_mid_on_c gave cell 148 the cos θ -0.79999995, and the packet code gave -0.80000001 from the three parts.
    """
    from artistools.inputmodel import get_cell_direction_selection
    from artistools.packets import add_packet_directions_lazypolars
    from artistools.packets import bin_packet_directions_polars

    centres = np.arange(-2.0, 3.0)
    vel_x, vel_y, vel_z = (axis.ravel() for axis in np.meshgrid(centres, centres, centres, indexing="ij"))
    dfcells_5 = pl.DataFrame({
        "modelgridindex": np.arange(len(vel_x)),
        "vel_x_mid_on_c": vel_x,
        "vel_y_mid_on_c": vel_y,
        "vel_z_mid_on_c": vel_z,
        "vel_r_mid_on_c": np.sqrt(vel_x**2 + vel_y**2 + vel_z**2),
    })
    lzmodel_11, modelmeta_11 = get_empty_3d_model(11, 1e9, 1.0)
    dfcells_11 = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel_11, modelmeta=modelmeta_11)
        .select("modelgridindex", "vel_x_mid_on_c", "vel_y_mid_on_c", "vel_z_mid_on_c", "vel_r_mid_on_c")
        .collect()
    )

    for dfcells_all in (dfcells_5, dfcells_11):
        dfcells = dfcells_all.filter(pl.col("vel_r_mid_on_c") > 0.0)
        dfbinned = (
            bin_packet_directions_polars(
                add_packet_directions_lazypolars(
                    dfcells.select(
                        "modelgridindex",
                        dirx=pl.col("vel_x_mid_on_c"),
                        diry=pl.col("vel_y_mid_on_c"),
                        dirz=pl.col("vel_z_mid_on_c"),
                    )
                )
            )
            .select("modelgridindex", "dirbin", "phibin")
            .collect()
        )
        expectedbins = dict(zip(dfbinned["modelgridindex"], dfbinned["dirbin"], strict=True))
        expectedphibins = dict(zip(dfbinned["modelgridindex"], dfbinned["phibin"], strict=True))

        selectedbins: dict[int, list[int]] = {}
        for dirbin in range(at.misc.get_viewingdirectionbincount()):
            args = argparse.Namespace(dirbin=dirbin, costhetamin=None, costhetamax=None, average_over_phi_angle=False)
            directionrange = at.misc.get_cell_direction_range(args, quiet=True)
            assert directionrange is not None
            for cell in dfcells.filter(get_cell_direction_selection(directionrange, dfcells.columns))["modelgridindex"]:
                selectedbins.setdefault(cell, []).append(dirbin)

        assert selectedbins == {cell: [dirbin] for cell, dirbin in expectedbins.items()}

        # --average-over-theta-angle selects the ϕ bin at each cos θ, e.g. the cells on the z axis are in ϕ bin 5
        selectedphibins: dict[int, list[int]] = {}
        for dirbin in range(at.misc.get_viewingdirection_phibincount()):
            args = argparse.Namespace(dirbin=dirbin, costhetamin=None, costhetamax=None, average_over_theta_angle=True)
            directionrange = at.misc.get_cell_direction_range(args, quiet=True)
            assert directionrange is not None
            for cell in dfcells.filter(get_cell_direction_selection(directionrange, dfcells.columns))["modelgridindex"]:
                selectedphibins.setdefault(cell, []).append(dirbin)

        assert selectedphibins == {cell: [phibin] for cell, phibin in expectedphibins.items()}


def test_plotinitialabundances_main_passes_the_selection(tmp_path: Path) -> None:
    """The command sends the four range arguments to the data function, and the file name records them."""
    import matplotlib.axes as mplax

    from artistools.inputmodel.plotinitialabundances import get_nuclide_massfractions

    with mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True) as mockplot:
        at.inputmodel.plotinitialabundances.main(
            argsraw=[str(modelpath_classic_3d), "-vmax", "0.02", "-thetamin", "90", "-o", str(tmp_path)]
        )

    expected = (
        get_nuclide_massfractions(modelpath_classic_3d, vmax=0.02, thetamin=90.0)
        .group_by("A")
        .agg(pl.col("massfraction").sum())
        .sort("A")
    )
    assert mockplot.call_count == 1
    plotted_x, plotted_y = mockplot.call_args.args[1:3]
    assert np.allclose(plotted_x, expected["A"])
    assert np.allclose(plotted_y, expected["massfraction"])
    assert (tmp_path / "plotinitialabundances_XvsA_vmax0.02_thetamin90.0.pdf").is_file()


def test_plotinitialabundances_write_data_gives_the_plotted_curves(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--write_data gives one table with the points of the curve of each model, which have different nuclides."""
    with mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True) as mockplot:
        at.inputmodel.plotinitialabundances.main(
            argsraw=[str(modelpath), str(modelpath_classic_3d), "-x", "Z", "-y", "abundance", "-o", str(tmp_path)],
            write_data=True,
        )

    dftable = read_data_file(tmp_path / "plotinitialabundances_abundancevsZ.txt")
    assert dftable.columns == ["model", "modelpath", "Z", "abundance"]
    assert mockplot.call_count == 2
    for call in mockplot.call_args_list:
        dfcurve = dftable.filter(pl.col("model") == call.kwargs["label"])
        dfdrawn = pl.DataFrame({"Z": np.asarray(call.args[1]), "abundance": np.asarray(call.args[2], dtype=float)})
        dfdrawn = dfdrawn.cast(dfcurve.select("Z", "abundance").schema)
        assert dfcurve.height == dfdrawn.height > 0
        assert dfcurve.join(dfdrawn, on=["Z", "abundance"], how="anti").is_empty()


def test_plotinitialabundances_table_keeps_two_models_of_one_name_apart(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Two models of one name gave one value of the column model, and the filter took them for one line."""
    modelcopy = tmp_path / "copy" / modelpath.name
    modelcopy.mkdir(parents=True)
    for filepath in modelpath.iterdir():
        if filepath.name != "plotlabel.txt":
            (modelcopy / filepath.name).symlink_to(filepath)
    at.inputmodel.plotinitialabundances.main(
        argsraw=[str(modelpath), str(modelcopy), "-x", "Z", "-y", "abundance", "-o", str(tmp_path)], write_data=True
    )

    dftable = read_data_file(tmp_path / "plotinitialabundances_abundancevsZ.txt")
    assert sorted(dftable["modelpath"].unique().to_list()) == sorted([str(modelpath), str(modelcopy)])
    curvesizes = dftable.group_by("modelpath").len()["len"].to_list()
    assert len(set(curvesizes)) == 1


def test_from_e2e_model_3d_equatorial_symmetry_with_nodyn(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A 3D grid of an equatorially symmetric model holds both halves, and --nodyn removes the dynamical ejecta.

    The 3D mapping scaled each isotope with the mass of every tracer, thus the excluded Zn56 came back. It
    scaled the grid to the mass of one half, and map_to_artis mirrored the 3D arrays, which stopped it.
    """
    from artistools.constants import C_cm_per_s
    from artistools.constants import Msun_to_g
    from artistools.inputmodel.from_e2e_model import get_grid
    from artistools.inputmodel.from_e2e_model import map_to_artis
    from artistools.inputmodel.from_e2e_model import t_model_init_s

    rng = np.random.default_rng(seed=2)
    ntraj = 40
    ntimes = 8
    state = np.where(np.arange(ntraj) < 20, -1.0, 0.0)
    # the dynamical tracers hold Zn56 alone, and the torus tracers hold He4 alone
    numberfractions = np.zeros((ntraj, 2))
    numberfractions[state == -1, 1] = 1.0 / 56
    numberfractions[state == 0, 0] = 1.0 / 4
    datpath = tmp_path / "e2emodel.npz"
    isopath = tmp_path / "iso_table.npy"
    np.savez(
        datpath,
        # every polar angle is below pi/2, thus the model has equatorial symmetry
        pos=np.column_stack([rng.uniform(0.05, 0.3, ntraj), rng.uniform(0.2, 1.4, ntraj)]),
        idx=np.arange(1, ntraj + 1, dtype=float),
        state=state,
        mass=np.full(ntraj, 1e-3),
        qdot=np.full((ntraj, ntimes), 1e10),
        hnuloss=np.zeros((ntraj, ntimes)),
        time=np.linspace(0.0, 2.0 * t_model_init_s, ntimes),
        nz=numberfractions,
        t5out=np.column_stack([np.zeros((ntraj, 4)), np.full(ntraj, 0.3)]),
    )
    np.save(isopath, np.array([[2, 2], [26, 30]]))

    vmax_on_c = 0.4
    dims = np.array([6, 6, 6])
    x3d, y3d, z3d, rhoint, xint, iso, q, ye, bs, eqsymfac, _ = get_grid(
        datpath, isopath, vmax_on_c, 3, dims, nodynej=True, nohmns=False, notorus=False, no_nu_trapping=False
    )
    assert eqsymfac == 2

    # the mapped data and the tracer data must print one total mass, or a correct mapping looks wrong
    printedmasses = [
        float(line.split(":")[1]) for line in capsys.readouterr().out.splitlines() if line.startswith("Total mass  ")
    ]
    assert printedmasses == pytest.approx([2 * 20 * 1e-3] * 2, rel=1e-6)

    cellvolume = (2 * vmax_on_c * C_cm_per_s * t_model_init_s / dims[0]) ** 3
    gridmass_msun = (rhoint * cellvolume).sum() / Msun_to_g
    # 20 torus tracers of 1e-3 Msun each, for both halves of the grid
    assert gridmass_msun == pytest.approx(2 * 20 * 1e-3, rel=1e-6)
    zn56mass_msun = (xint[1] * rhoint * cellvolume).sum() / Msun_to_g
    assert zn56mass_msun < 1e-20

    dfmodel, _, modelmeta = map_to_artis(
        3, dims, vmax_on_c, rhoint, xint, iso, q, ye, eqsymfac, x3d=x3d, y3d=y3d, z3d=z3d, bin_state=bs
    )
    assert dfmodel.height == dims.prod()
    assert "wid_init" in modelmeta


@pytest.mark.parametrize(("prefix", "suffix"), [(" ", ""), ("", " "), ("\t", "")])
def test_scan_modeldata_1d_reads_a_line_with_one_leading_space(tmp_path: Path, prefix: str, suffix: str) -> None:
    """A model.txt whose lines start or end with a space must give the cell ids and the velocities.

    The fast reader only tested for a double space. A single leading space gave polars an empty first
    field, thus each cell id became null and each value moved one column to the right.
    """
    (tmp_path / "model.txt").write_text(
        "3\n1.0\n"
        + "".join(f"{prefix}{cellid} {cellid}000.0 -5.0 1.0 0.5 0.0 0.0 0.0{suffix}\n" for cellid in (1, 2, 3)),
        encoding="utf-8",
    )

    dfmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)
    dfmodel = dfmodel.collect()

    assert dfmodel["inputcellid"].to_list() == [1, 2, 3]
    assert dfmodel["vel_r_max_kmps"].to_list() == pytest.approx([1000.0, 2000.0, 3000.0])
    assert dfmodel["logrho"].to_list() == pytest.approx([-5.0, -5.0, -5.0])
    assert np.isclose(modelmeta["vmax_cmps"], 3000.0 * 1e5)


def test_scan_modeldata_1d_with_one_cell_uses_the_fast_reader(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model.txt with one cell must use the fast reader.

    The second data line of such a file is empty, and the check of the separators rejected an empty line.
    """
    (tmp_path / "model.txt").write_text("1\n1.0\n1 1000.0 -5.0 1.0 0.5 0.0 0.0 0.0\n", encoding="utf-8")

    dfmodel, _ = at.inputmodel.scan_modeldata(tmp_path)

    assert "Using fast method" in capsys.readouterr().out
    assert dfmodel.collect()["vel_r_max_kmps"].to_list() == pytest.approx([1000.0])


def test_scan_modeldata_reads_an_extra_space_in_a_later_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A later line with a double space must give the same values as the other lines.

    The fast reader tests only the first two data lines. A double space later gave a null and moved the values.
    """
    (tmp_path / "model.txt").write_text(
        "3\n1.0\n1 1000.0 -5.0 1.0 0.5 0.0 0.0 0.0\n2 2000.0 -6.0 1.0 0.5 0.0 0.0 0.0\n"
        "3  3000.0 -7.0 1.0 0.5 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    dfmodel, _ = at.inputmodel.scan_modeldata(tmp_path)

    assert "Using fast method" not in capsys.readouterr().out
    assert dfmodel.collect()["logrho"].to_list() == pytest.approx([-5.0, -6.0, -7.0])


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("-1.735904-244", -1.735904e-244),
        ("-0.5", -0.5),
        ("1.735904-244", 1.735904e-244),
        ("-4.165372E-31", -4.165372e-31),
        ("1.5+100", 1.5e100),
    ],
)
def test_fix_fortran_exponents_keeps_a_negative_mantissa(token: str, expected: float) -> None:
    """Only the sign before a three-digit exponent gets the "e".

    The repair replaced each hyphen of a token whose fourth-last character was a hyphen. A negative
    mantissa then became e-1.735904e-244, and the strict cast of the heating column raised.
    """
    from artistools.inputmodel.rprocess_from_trajectory import fix_fortran_exponents

    value = pl.DataFrame({"htot": [token]}).select(fix_fortran_exponents(pl.Float64)).item()

    assert value == pytest.approx(expected, rel=1e-12)


def test_from_e2e_model_split_trajectory_pieces_give_the_same_heating(tmp_path: Path) -> None:
    """A dynamical trajectory that the e2e model splits into pieces gives the heating of the whole trajectory.

    The merge summed the specific heating rate of the pieces, which counted the heating one time for each
    piece. The mass fractions, Ye, and the velocity of the pieces were mass-averaged, and q now is too.
    """
    from artistools.inputmodel.from_e2e_model import get_grid
    from artistools.inputmodel.from_e2e_model import t_model_init_s

    rng = np.random.default_rng(seed=1)
    ntimes = 8
    isopath = tmp_path / "iso_table.npy"
    np.save(isopath, np.array([[2.0, 2.0], [30.0, 26.0]]))
    numberfractions = rng.uniform(0.01, 0.2, size=2)

    def write_dat(datpath: Path, idx: list[int], masses: list[float], qdots: list[float]) -> None:
        ntraj = len(idx)
        np.savez(
            datpath,
            pos=np.column_stack([np.full(ntraj, 0.2), np.full(ntraj, 1.0)]),
            idx=np.array(idx, dtype=float),
            state=np.full(ntraj, -1.0),
            mass=np.array(masses),
            qdot=np.tile(np.array(qdots)[:, np.newaxis], (1, ntimes)),
            hnuloss=np.zeros((ntraj, ntimes)),
            time=np.linspace(0.0, 2.0 * t_model_init_s, ntimes),
            nz=np.tile(numberfractions, (ntraj, 1)),
            t5out=np.column_stack([np.zeros((ntraj, 4)), np.full(ntraj, 0.3)]),
        )

    datpath_whole = tmp_path / "whole.npz"
    datpath_split = tmp_path / "split.npz"
    # the pieces have different masses and heating rates, thus a plain mean and a sum differ from the mass-weighted
    # mean of the whole trajectory, (1 * 1 + 2 * 2 + 3 * 4) / 6 = 17 / 6
    write_dat(datpath_whole, idx=[7], masses=[6e-3], qdots=[17.0 / 6.0 * 1e10])
    write_dat(datpath_split, idx=[7, 10007, 20007], masses=[1e-3, 2e-3, 3e-3], qdots=[1e10, 2e10, 4e10])

    results = [
        get_grid(
            datpath,
            isopath,
            0.4,
            model_dim=2,
            grid_dims=np.array([4, 8]),
            nodynej=False,
            nohmns=False,
            notorus=False,
            no_nu_trapping=False,
        )
        for datpath in (datpath_whole, datpath_split)
    ]
    (_, _, rho_whole, _, _, q_whole, _, _, _), (_, _, rho_split, _, _, q_split, _, _, _) = results

    assert np.amax(rho_whole) > 0.0
    np.testing.assert_allclose(rho_split, rho_whole, rtol=1e-10)
    np.testing.assert_allclose(q_split, q_whole, rtol=1e-10)
    for result in results:
        assert result[-1]["particleid"].unique().to_list() == [7]


def test_maptogrid_keeps_the_mirror_symmetry_of_the_particles(tmp_path: Path) -> None:
    """Two equal particles at x = -1000 and x = +1000 give a grid that is symmetric under x, y, z -> -x, -y, -z.

    The kernel sampled each cell at its lower corner, which grid.dat gives as pos_x_min. Thus the cell
    [-1000, -500] held the kernel peak, and its mirror cell [500, 1000] held the kernel value 500 away.
    """
    from artistools.inputmodel.maptogrid import maptogrid

    def snapshotline(particleid: int, x: float) -> str:
        values = [0.0] * 31
        values[0] = particleid
        values[1] = 300.0  # h
        values[2] = x
        for column in (14, 15, 16, 17):  # pmass, rho, p, rho_rst
            values[column] = 1.0
        values[20] = 0.3  # ye
        return " ".join(str(value) for value in values)

    snapshotpath = tmp_path / "ejectasnapshot.dat"
    snapshotpath.write_text(snapshotline(1, -1000.0) + "\n" + snapshotline(2, 1000.0) + "\n", encoding="utf-8")

    ncoordgrid = 4
    maptogrid(
        snapshotpath,
        tmp_path,
        ncoordgrid=ncoordgrid,
        dtextra_seconds=0.0,
        setgrid_fractionrmax=1.0,
        modifysmoothinglength="False",
    )

    dfgrid = at.misc.read_wsv(tmp_path / "grid.dat", skip_rows=3)
    rho = dfgrid["rho"].to_numpy().reshape((ncoordgrid, ncoordgrid, ncoordgrid), order="F")

    assert rho.max() > 0.0
    np.testing.assert_allclose(rho, rho[::-1, ::-1, ::-1], rtol=1e-10)
    # the lower edges of the cells span -rmax to rmax minus one cell width
    assert dfgrid["pos_x_min"].min() == pytest.approx(-1000.0)
    assert dfgrid["pos_x_min"].max() == pytest.approx(500.0)


@pytest.mark.parametrize(
    ("ncoordgrid", "positive_axis", "expectedlayer"), [(4, True, 2), (4, False, 1), (5, True, 2), (5, False, 2)]
)
def test_get_2d_slice_takes_the_side_of_the_axis(ncoordgrid: int, positive_axis: bool, expectedlayer: int) -> None:
    """-axis=-z takes the layer below the origin, and -axis=+z the layer above it.

    The parser read the sign of -axis, but the slice ignored it. Thus both signs gave the layer with pos_z_min == 0.
    The centre layer of an odd grid holds the origin, thus both signs give that layer.
    """
    from artistools.inputmodel.plotinitialcomposition import get_2d_slice_through_3d_model

    dfmodel, modelmeta = get_empty_3d_model(ncoordgrid=ncoordgrid, vmax=1e9, t_model_init_days=1.0)
    dfmodel = dfmodel.collect()
    wid = modelmeta["wid_init_z"]

    dfslice = get_2d_slice_through_3d_model(dfmodel, "z", modelmeta, "x", "y", positive_axis=positive_axis)

    assert dfslice.height == ncoordgrid**2
    assert dfslice["pos_z_min"].to_numpy() == pytest.approx(-modelmeta["vmax_cmps"] * day_to_s + expectedlayer * wid)


def test_plotinitialcomposition_negative_axis_plots_the_layer_below_the_origin(tmp_path: Path) -> None:
    """The command with -axis=-z draws the densities of the layer whose upper edge is the origin."""
    with mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True) as mockimshow:
        at.inputmodel.plotinitialcomposition.main(argsraw=["rho", str(modelpath_3d), "-axis=-z", "-o", str(tmp_path)])

    dfmodel, modelmeta = get_derived_modeldata(modelpath_3d)
    dfmodel = dfmodel.collect()
    # the cells of a layer go with x fastest, thus a reshape gives the rows of y and the columns of x
    shape = (modelmeta["ncoordgridy"], modelmeta["ncoordgridx"])
    layerbelow, layerabove = (
        dfmodel.filter(selection).sort("inputcellid")["rho"].to_numpy().reshape(shape)
        for selection in (
            pl.col("pos_z_max").abs() < 0.01 * modelmeta["wid_init_z"],
            pl.col("pos_z_min").abs() < 0.01 * modelmeta["wid_init_z"],
        )
    )
    # the two layers differ, thus the plot of the wrong layer gives other values
    assert not np.allclose(layerbelow, layerabove)
    np.testing.assert_allclose(np.asarray(mockimshow.call_args.args[1]), layerbelow, rtol=1e-6)


def test_save_modeldata_1d_keeps_a_negative_custom_value(tmp_path: Path) -> None:
    """A negative value of a custom column of a 1D model survives a write and a read.

    The writer gave 0.0 to each value that was not above zero, thus it lost a negative value. The 3D
    writer kept the sign. A NaN value still becomes zero. A negative mass fraction, e.g. from the noise of an
    interpolation, still becomes zero, because ARTIS needs a valid composition.
    """
    dfmodel = pl.DataFrame({
        "inputcellid": [1, 2, 3],
        "vel_r_max_kmps": [1000.0, 2000.0, 3000.0],
        "logrho": [-10.0, -11.0, -12.0],
        "X_Fegroup": [1.0, 1.0, 1.0],
        "X_Ni56": [0.5, -1.0e-12, 0.3],
        "mycolumn": [-3.2e4, 6.0, float("nan")],
    })
    at.inputmodel.save_modeldata(
        dfmodel, outpath=tmp_path, modelmeta={"dimensions": 1, "t_model_init_days": 1.0}, extracols=["mycolumn"]
    )

    dfwritten = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)[0].collect()

    assert dfwritten["mycolumn"].to_list() == pytest.approx([-3.2e4, 6.0, 0.0])
    assert dfwritten["X_Ni56"].to_list() == pytest.approx([0.5, 0.0, 0.3])


def test_add_mass_to_center_fills_a_sphere_symmetric_about_the_origin() -> None:
    """The hole takes each cell whose centre lies inside 0.1 c, and the filled set is mirror-symmetric.

    The lower corner of each cell set its radius, thus the hole reached one cell further on the positive
    side of each axis than on the negative side.
    """
    from artistools.constants import C_cm_per_s as CLIGHT
    from artistools.inputmodel.modelfromhydro import add_mass_to_center

    ncoordgrid = 8
    t_model_days = 1.0
    xmax = 0.2 * CLIGHT * t_model_days * day_to_s
    wid = 2 * xmax / ncoordgrid
    indices = np.arange(ncoordgrid)
    posmin = -xmax + wid * indices
    gridx, gridy, gridz = np.meshgrid(posmin, posmin, posmin, indexing="ij")
    griddata = pl.DataFrame({
        "inputcellid": np.arange(1, ncoordgrid**3 + 1),
        "pos_x_min": gridx.flatten(order="F"),
        "pos_y_min": gridy.flatten(order="F"),
        "pos_z_min": gridz.flatten(order="F"),
        "rho": np.zeros(ncoordgrid**3),
        "Ye": np.zeros(ncoordgrid**3),
    })

    modelmeta = {"dimensions": 3, "t_model_init_days": t_model_days, "wid_init": wid}
    dffilled = add_mass_to_center(griddata, modelmeta)

    filled = (dffilled["rho"].to_numpy() > 0.0).reshape((ncoordgrid, ncoordgrid, ncoordgrid), order="F")
    assert filled.any()
    np.testing.assert_array_equal(filled, filled[::-1, ::-1, ::-1])
    centreradius = np.sqrt((gridx + wid / 2) ** 2 + (gridy + wid / 2) ** 2 + (gridz + wid / 2) ** 2)
    np.testing.assert_array_equal(filled, centreradius < 0.1 * CLIGHT * t_model_days * day_to_s)


def test_get_coarse_velocity_bins_cover_the_outermost_cell() -> None:
    """The last coarse bin ends above the largest mid-point velocity, thus the outer cells keep their mass.

    The number of bins was the integer part of the velocity range over the bin width. Thus the last bin ended
    at or below the outermost cell, and get_binned_profile dropped that cell.
    """
    from artistools.inputmodel.plotdensity import get_binned_profile
    from artistools.inputmodel.plotdensity import get_coarse_velocity_bins

    dfmodel = pl.DataFrame({"vel_r_mid": [1.0e9, 2.0e9, 4.0e9, 5.0e9, 6.5e9], "mass_g": [1.0, 1.0, 1.0, 1.0, 1.0]})

    binedges = get_coarse_velocity_bins(dfmodel, nbins=None, vmax_cmps=6.5e9)
    assert binedges[-1] > 6.5e9

    _, binned_massvals, _ = get_binned_profile(dfmodel, binedges, plotye=False)
    binwidths_on_c = np.diff([0.0, *binedges]) / at.constants.C_cm_per_s
    assert np.sum(binned_massvals[::2] * binwidths_on_c) * at.constants.Msun_to_g == pytest.approx(5.0)


def test_apply_density_perturbations_keeps_the_cells_outside_vmax() -> None:
    """The sinusoidal perturbation changes a cell inside the sphere of radius vmax and no cell outside it.

    The mask compared the radius with 1, but the coordinates are in units of c and span only -vmax to
    vmax. Thus the mask had no effect, and the radius had no z component.
    """
    from artistools.inputmodel.from_e2e_model import apply_density_perturbations

    ncoordgrid = 4
    vmax = 0.5
    dfmodel = pl.DataFrame({"rho": np.ones(ncoordgrid**3)})

    dfperturbed = apply_density_perturbations(dfmodel, vmax, ("sinusoidal", 0.3, 0.3))

    rho = dfperturbed["rho"].to_numpy().reshape((ncoordgrid, ncoordgrid, ncoordgrid), order="F")
    posmid = -vmax + (np.arange(ncoordgrid) + 0.5) * (2 * vmax / ncoordgrid)
    gridx, gridy, gridz = np.meshgrid(posmid, posmid, posmid, indexing="ij")
    outside = np.sqrt(gridx**2 + gridy**2 + gridz**2) > vmax

    assert outside.any()
    assert (~outside).any()
    np.testing.assert_allclose(rho[outside], 1.0)
    assert np.all(np.abs(rho[~outside] - 1.0) > 0.1)


@pytest.mark.parametrize("outputdimensions", [0, 1])
def test_dimension_reduce_keeps_the_centre_cell_of_an_odd_grid(outputdimensions: int) -> None:
    """A 5^3 model has a centre cell with a mid-point velocity of zero, and the reduction keeps its mass.

    The bins were closed on the right, thus a velocity of exactly zero fell below the first bin and the
    filter dropped the cell.
    """
    ncoordgrid = 5
    dfmodel3d, modelmeta_3d = get_empty_3d_model(ncoordgrid=ncoordgrid, vmax=1e9, t_model_init_days=1.0)
    dfmodel3d = dfmodel3d.collect().with_columns(rho=pl.lit(1.0))
    dfmodel3d_derived = at.inputmodel.add_derived_cols_to_modeldata(dfmodel3d, modelmeta=modelmeta_3d).collect()
    assert dfmodel3d_derived.filter(pl.col("vel_r_mid") == 0).height == 1

    mass_inside_vmax = dfmodel3d_derived.filter(pl.col("vel_r_mid") <= modelmeta_3d["vmax_cmps"])["mass_g"].sum()
    dfmodel_lowerd, _, _, _ = at.inputmodel.dimension_reduce_model(
        dfmodel=dfmodel3d_derived.select([*dfmodel3d.columns, "mass_g"]),
        modelmeta=modelmeta_3d,
        outputdimensions=outputdimensions,
    )

    assert dfmodel_lowerd["mass_g"].sum() == pytest.approx(mass_inside_vmax, rel=1e-10)


@pytest.mark.parametrize("zero_density", [False, True])
def test_dimension_reduce_keeps_abundances_of_empty_shells(tmp_path: Path, zero_density: bool) -> None:
    """Keep abundance rows for shells with no source cells and shells with zero mass."""
    xmax = 1e9 * day_to_s
    cellindices = np.arange(8)
    dfmodel = pl.DataFrame({
        "inputcellid": cellindices + 1,
        "pos_rcyl_mid": (cellindices % 4 + 0.5) * xmax / 4,
        "pos_z_mid": (cellindices // 4 - 0.5) * xmax,
        "rho": [0.0] * 8 if zero_density else [1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 4.0],
    })
    modelmeta = {
        "dimensions": 2,
        "npts_model": 8,
        "ncoordgridrcyl": 4,
        "ncoordgridz": 2,
        "wid_init_rcyl": xmax / 4,
        "wid_init_z": xmax,
        "t_model_init_days": 1.0,
        "vmax_cmps": 1e9,
    }
    dfabundances = dfmodel.select("inputcellid").with_columns(
        X_Fe=pl.Series([0.0] * 4 + [1.0] * 4), X_Ni=pl.Series([1.0] * 4 + [0.0] * 4)
    )
    dfmodel_out, dfabundances_out, _, modelmeta_out = at.inputmodel.dimension_reduce_model(
        dfmodel, outputdimensions=1, dfelabundances=dfabundances, modelmeta=modelmeta
    )
    assert dfabundances_out is not None

    expected_abundances = pl.DataFrame({
        "inputcellid": [1, 2, 3, 4],
        "X_Fe": [0.0] * 4 if zero_density else [0.0, 0.0, 9 / 13, 1 / 3],
        "X_Ni": [0.0] * 4 if zero_density else [0.0, 0.0, 4 / 13, 2 / 3],
    })
    # The cylindrical cell volumes are proportional to 1, 3, 5, and 7 at each height.
    expected_masses = np.array([0.0] * 4 if zero_density else [0.0, 0.0, 13.0, 15.0]) * math.pi * xmax**3 / 16
    assert modelmeta_out["npts_model"] == 4
    pltest.assert_frame_equal(dfmodel_out.select("inputcellid"), expected_abundances.select("inputcellid"))
    pltest.assert_frame_equal(dfabundances_out.select(expected_abundances.columns), expected_abundances)
    np.testing.assert_allclose(dfabundances_out["mass_g"], expected_masses)
    np.testing.assert_allclose(dfmodel_out["mass_g"], expected_masses)

    modelfolder = tmp_path / "model2d"
    modelfolder.mkdir()
    at.inputmodel.save_modeldata(dfmodel, outpath=modelfolder, modelmeta=modelmeta)
    at.inputmodel.save_initelemabundances(dfabundances, outpath=modelfolder)
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=[modelfolder], dimensionreduce=1, outputpath=tmp_path / "output"
    )
    reducedfolder = tmp_path / "output" / "model2d_dimreduce_1d"
    dfmodel_read, modelmeta_read = at.inputmodel.scan_modeldata(reducedfolder)
    dfabundances_read = at.inputmodel.scan_initelemabundances(reducedfolder)
    assert modelmeta_read["npts_model"] == 4
    pltest.assert_frame_equal(
        dfmodel_read.select("inputcellid").collect(), expected_abundances.select("inputcellid"), check_dtypes=False
    )
    pltest.assert_frame_equal(
        dfabundances_read.select(expected_abundances.columns).collect(),
        expected_abundances,
        check_dtypes=False,
        rel_tol=1e-6,
    )

    # Cells 3 and 7 supply all the abundance data for the outer shell.
    with pytest.raises(AssertionError):
        at.inputmodel.dimension_reduce_model(
            dfmodel,
            outputdimensions=1,
            dfelabundances=dfabundances.filter(~pl.col("inputcellid").is_in([3, 7])),
            modelmeta=modelmeta,
        )


def test_scan_modeldata_reads_a_header_comment_with_a_quotation_mark(tmp_path: Path) -> None:
    """A header comment with one quotation mark must not hide the cells from the fast reader.

    The reader took the quotation mark as the start of a quoted field. Thus it skipped the cells after the
    header, and it gave an error for an empty file. save_modeldata writes such a comment again.
    """
    dfmodel, modelmeta = at.inputmodel.scan_modeldata(modelpath_3d, printwarningsonly=True)
    dfmodel = dfmodel.collect()
    at.inputmodel.save_modeldata(
        dfmodel, outpath=tmp_path, modelmeta=modelmeta | {"headercommentlines": ['DDT "N100 model']}
    )

    dfmodel_read, modelmeta_read = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)

    assert modelmeta_read["headercommentlines"] == ['DDT "N100 model']
    pltest.assert_frame_equal(dfmodel_read.collect(), dfmodel, rel_tol=1e-6)


@pytest.mark.parametrize(
    "modeltext",
    [
        "3\n1.0\n0 1000 -10 0.1 0.2 0 0 0\n1 2000 -11 0.3 0.4 0 0 0\n2 3000 -12 0.5 0.6 0 0 0\n",
        "3\n1.0\n0 1000 -10\n0.1 0.2 0 0 0\n1 2000 -11\n0.3 0.4 0 0 0\n2 3000 -12\n0.5 0.6 0 0 0\n",
    ],
    ids=["onelinepercell", "twolinespercell"],
)
def test_scan_modeldata_1d_cell_ids_can_start_at_zero(tmp_path: Path, modeltext: str) -> None:
    """sn3d takes the id of the first cell, 0 or 1, as the start of the cell index.

    The reader took each id as one more than the cell index. Thus the first cell got the index -1, and the
    second line of each cell of a file with two lines for each cell went to the next cell.
    """
    (tmp_path / "model.txt").write_text(modeltext, encoding="utf-8")

    dfmodel = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)[0].collect()

    assert dfmodel["inputcellid"].to_list() == [0, 1, 2]
    assert dfmodel["modelgridindex"].to_list() == [0, 1, 2]
    assert dfmodel["X_Ni56"].to_list() == pytest.approx([0.2, 0.4, 0.6])


def test_scan_modeldata_2d_cell_ids_can_start_at_zero(tmp_path: Path) -> None:
    """A 2D model with the ids 0 to 7 reads, and its cell positions agree with the cell index from 0."""
    ncoordgridrcyl, ncoordgridz = 2, 4
    vmax_cmps = 1e9
    xmax = vmax_cmps * day_to_s
    lines = [f"{ncoordgridrcyl} {ncoordgridz}", "1.0", f"{vmax_cmps}"]
    for mgi in range(ncoordgridrcyl * ncoordgridz):
        pos_rcyl_mid = (mgi % ncoordgridrcyl + 0.5) * xmax / ncoordgridrcyl
        pos_z_mid = -xmax + (mgi // ncoordgridrcyl + 0.5) * 2 * xmax / ncoordgridz
        lines.append(f"{mgi} {pos_rcyl_mid:.7e} {pos_z_mid:.7e} 1e-10 0 0 0 0 0")
    (tmp_path / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dfmodel = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)[0].collect()

    assert dfmodel["modelgridindex"].to_list() == list(range(8))


def test_scan_modeldata_3d_cell_ids_can_start_at_zero(tmp_path: Path) -> None:
    """A 3D model with the ids 0 to 7 gives the cell index 0 to 7."""
    lzdfmodel, modelmeta = get_empty_3d_model(ncoordgrid=2, vmax=1e9, t_model_init_days=1.0)
    dfmodel = lzdfmodel.collect().with_columns(pl.col("inputcellid") - 1, rho=pl.lit(1e-10))
    at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta=modelmeta)

    dfmodel_read = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)[0].collect()

    assert dfmodel_read["inputcellid"].to_list() == list(range(8))
    assert dfmodel_read["modelgridindex"].to_list() == list(range(8))


@pytest.mark.parametrize(
    ("modeltext", "match"),
    [
        ("5\n1.0\n1 1000 -10 0 0 0 0 0\n2 2000 -10 0 0 0 0 0\n3 3000 -10 0 0 0 0 0\n", "found only 3 cells"),
        ("5\n1.0\n1  1000 -10 0 0 0 0 0\n2  2000 -10 0 0 0 0 0\n", "found only 2 cells"),
        ("3\n1.0\n1 1000 -10\n0 0 0 0 0\n2 2000 -10\n0 0 0 0 0\n3 3000 -10\n", "found only 2 cells"),
        ("3\n1.0\n", "found only 0 cells"),
        ("3\n1.0\n1 1000 -10 0 0 0 0 0\n2 2000 -10 0 0 0 0 0\n4 3000 -10 0 0 0 0 0\n", "increase by one"),
    ],
    ids=["fastreader", "slowreader", "twolinespercell", "nocells", "missingid"],
)
def test_scan_modeldata_rejects_a_model_with_missing_cells(tmp_path: Path, modeltext: str, match: str) -> None:
    """A model.txt with fewer cells than its header gives, or with a gap in the ids, stops with a message, as in sn3d.

    The reader gave the cells that the file held, and the vmax of a 1D model then came from the last of them.
    """
    (tmp_path / "model.txt").write_text(modeltext, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)


@pytest.mark.parametrize("skippedline", ["# time of the snapshot [days]\n", "\n", "   # an indented comment\n"])
def test_scan_modeldata_skips_a_comment_before_the_time(tmp_path: Path, skippedline: str) -> None:
    """sn3d skips a comment line and an empty line before the time of the model, thus the reader must also skip it."""
    (tmp_path / "model.txt").write_text(
        f"3\n{skippedline}2.5\n1 1000 -10 0 0 0 0 0\n2 2000 -10 0 0 0 0 0\n3 3000 -10 0 0 0 0 0\n", encoding="utf-8"
    )

    dfmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)

    assert modelmeta["t_model_init_days"] == pytest.approx(2.5)
    assert dfmodel.collect()["vel_r_max_kmps"].to_list() == pytest.approx([1000.0, 2000.0, 3000.0])


def test_scan_modeldata_3d_header_with_mid_point_positions(tmp_path: Path) -> None:
    """A 3D header can name the positions pos_x_mid, pos_y_mid, and pos_z_mid, which sn3d accepts.

    Only the name pos_x_min selected the header names. Each other name went to the code for a file with no
    header, which looked for the columns inputpos_a, inputpos_b, and inputpos_c.
    """
    lzdfmodel, modelmeta = get_empty_3d_model(ncoordgrid=4, vmax=1e9, t_model_init_days=1.0)
    dfmodel = lzdfmodel.collect()
    wid = modelmeta["wid_init_x"]
    dfmid = dfmodel.select(
        "inputcellid", *[(pl.col(f"pos_{ax}_min") + wid / 2).alias(f"pos_{ax}_mid") for ax in "xyz"], rho=pl.lit(1e-10)
    )
    lines = [str(modelmeta["npts_model"]), "1.0", f"{modelmeta['vmax_cmps']}"]
    lines.append("#inputcellid pos_x_mid pos_y_mid pos_z_mid rho X_Fegroup X_Ni56 X_Co56 X_Fe52 X_Cr48")
    lines.extend(f"{cellid} {x:.7e} {y:.7e} {z:.7e} {rho} 0 0 0 0 0" for cellid, x, y, z, rho in dfmid.iter_rows())
    (tmp_path / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dfmodel_read = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)[0].collect()

    for ax in "xyz":
        np.testing.assert_allclose(
            dfmodel_read[f"pos_{ax}_min"].to_numpy(), dfmodel[f"pos_{ax}_min"].to_numpy(), atol=1e-4 * wid
        )


def test_save_modeldata_1d_writes_zero_for_a_null(tmp_path: Path) -> None:
    """The 1D writer writes 0.0 for a null value, as the writer of a 2D or a 3D model does.

    The writer called math.isnan on each value, which raised TypeError for a null.
    """
    dfmodel = pl.DataFrame({
        "inputcellid": [1, 2],
        "vel_r_max_kmps": [1000.0, 2000.0],
        "logrho": [-10.0, -11.0],
        "X_Ni56": [0.5, None],
    })
    at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta={"t_model_init_days": 1.0})

    dfwritten = at.inputmodel.scan_modeldata(tmp_path, printwarningsonly=True)[0].collect()

    assert dfwritten["X_Ni56"].to_list() == pytest.approx([0.5, 0.0])


def test_describeinputmodel_empty_cell_and_cell_outside_the_model(capsys: pytest.CaptureFixture[str]) -> None:
    """-modelgridindex of an empty cell gives a message, and a cell outside the model gives an error.

    The description divided by the mass of the selected cells, thus both raised ZeroDivisionError.
    """
    at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=[modelpath_3d], modelgridindex="0")
    assert "cell 0 is empty" in capsys.readouterr().out

    with pytest.raises(ValueError, match="names no cell"):
        at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=[modelpath], modelgridindex="5")


@pytest.mark.parametrize(
    ("loweredges", "expectedcellcount"),
    [
        # an odd count: the middle cell holds the origin, thus each side takes it
        ([-1.0, -0.6, -0.2, 0.2, 0.6], 3),
        # an even count with a rounding error at the origin, as a grid of -xmax + 2 xmax i / n can give
        ([-1.5, -1.0, -0.5, -1e-7, 0.5, 1.0], 3),
    ],
    ids=["odd", "evenwithrounding"],
)
def test_profile_along_axis_takes_half_of_the_cells_on_each_side(
    loweredges: list[float], expectedcellcount: int
) -> None:
    """Each side of the slice axis takes the same number of cells, and the middle cell of an odd grid on both sides.

    The positive side took the cells with pos_min >= 0, thus it lost the middle cell of an odd grid. An edge at
    the origin with a rounding error below zero moved a cell to the negative side.
    """
    from artistools.inputmodel.slice1dfromconein3dmodel import get_profile_along_axis

    middle = loweredges[len(loweredges) // 2]
    dfmodel = pl.DataFrame({"pos_x_min": middle, "pos_y_min": middle, "pos_z_min": z} for z in loweredges).cast(
        pl.Float32
    )

    for positive_axis in (True, False):
        args = argparse.Namespace(sliceaxis="z", other_axis1="x", other_axis2="y", positive_axis=positive_axis)
        dfprofile = get_profile_along_axis(dfmodel, args)
        assert dfprofile.height == expectedcellcount, positive_axis


def test_make1dmodelfromcone_keeps_ye_q_and_tracercount(tmp_path: Path) -> None:
    """The 1D model of a cone keeps Ye and q as mass-weighted means and tracercount as a sum, as the axis path does.

    The cone took only the X_ columns. A kilonova model then had no initial energy and no Ye.
    """
    dfmodel, modelmeta = at.inputmodel.scan_modeldata(modelpath_3d, printwarningsonly=True)
    dfmodel = dfmodel.collect().with_columns(Ye=pl.lit(0.25), q=pl.lit(1e12), tracercount=pl.lit(2, dtype=pl.Int32))
    modelfolder = tmp_path / "model3d"
    at.inputmodel.save_modeldata(dfmodel, outpath=modelfolder, modelmeta=modelmeta)
    shutil.copy(modelpath_3d / "abundances.txt.xz", modelfolder)

    at.inputmodel.slice1dfromconein3dmodel.main(
        argsraw=[], modelpath=[modelfolder], outputpath=tmp_path, axis="-z", nshells=4, coneangle=60
    )

    dfcone = at.inputmodel.scan_modeldata(tmp_path / "model_1d.txt", printwarningsonly=True)[0].collect()
    assert dfcone["Ye"].to_list() == pytest.approx([0.25] * dfcone.height)
    assert dfcone["q"].to_list() == pytest.approx([1e12] * dfcone.height, rel=1e-6)
    # each 3D cell gives 2 tracers, thus a shell has an even sum above zero
    assert all(count > 0 and count % 2 == 0 for count in dfcone["tracercount"].to_list())


def test_makeartismodelfromparticlegridmap_trajectory_q_replaces_the_grid_q(
    tmp_path: Path, trajectory_copy: Path
) -> None:
    """The model takes the q of the trajectories, and the Q column of an old grid.dat has no effect.

    The reader gave that column as q. A join then kept both columns, thus the model held the grid.dat value and
    the reduction to 1D stopped at the column q_right.
    """
    gridfolderpath = tmp_path / "grid"
    shutil.copytree(
        testdatapath / "kilonova", gridfolderpath, ignore=shutil.ignore_patterns("trajectories"), dirs_exist_ok=True
    )
    at.inputmodel.maptogrid.main(
        argsraw=[], inputpath=gridfolderpath, outputpath=gridfolderpath, ncoordgrid=4, force=True
    )
    gridlines = (gridfolderpath / "grid.dat").read_text(encoding="utf-8").splitlines()
    gridlines = [*gridlines[:3], f"{gridlines[3]} Q", *(f"{line} 1e10" for line in gridlines[4:])]
    (gridfolderpath / "grid.dat").write_text("\n".join(gridlines) + "\n", encoding="utf-8")

    for dimensions in (3, 1):
        outpath = tmp_path / f"model_{dimensions}d"
        at.inputmodel.modelfromhydro.main(
            argsraw=[],
            gridfolderpath=gridfolderpath,
            trajectoryroot=trajectory_copy,
            outputpath=outpath,
            dimensions=dimensions,
            timedays=0.1,
        )
        dfmodel = at.inputmodel.scan_modeldata(outpath, printwarningsonly=True)[0].collect()
        qvalues = dfmodel.filter(pl.col("q") > 0)["q"].to_numpy()
        assert qvalues.size > 0
        assert not np.allclose(qvalues, 1e10, rtol=1e-3), "the model holds the q of grid.dat"
        assert "Q" not in dfmodel.columns


def test_plotinitialcomposition_gives_each_panel_its_colour_bar(tmp_path: Path) -> None:
    """Each panel gets a colour bar with its own colour limits and a label of its own quantity.

    One colour bar came from the last panel, with a label that came from the list of the variables. Thus the
    bar of "rho Fe" gave the colours of Fe and the label of rho.
    """
    import matplotlib.colorbar as mplcolorbar

    with mock.patch.object(
        mplcolorbar.Colorbar, "set_label", side_effect=mplcolorbar.Colorbar.set_label, autospec=True
    ) as mocksetlabel:
        at.inputmodel.plotinitialcomposition.main(argsraw=["rho", "Fe", str(modelpath_3d), "-o", str(tmp_path)])

    # matplotlib makes each colour bar with an empty label first
    labelsandlimits = [
        (call.args[1], call.args[0].mappable.get_clim()) for call in mocksetlabel.call_args_list if call.args[1]
    ]
    assert len(labelsandlimits) == 2
    (rholabel, rholimits), (felabel, felimits) = labelsandlimits
    assert rholabel == r"$\rho$ [g/cm³]"
    assert rholimits == pytest.approx((1e-15, 1e-7))
    assert "Fe" in felabel
    assert felimits != pytest.approx((1e-15, 1e-7))


def test_plotinitialcomposition_2d_slice_reads_ye_from_the_ye_file(tmp_path: Path) -> None:
    """A model with no Ye column gets Ye from Ye.txt for the 2D slice, as for the 3D plot.

    The 2D slice looked for the column X_Ye, which does not exist.
    """
    modelfolder = tmp_path / "model"
    modelfolder.mkdir()
    for filename in ("model.txt.xz", "abundances.txt.xz"):
        shutil.copy(modelpath_3d / filename, modelfolder)
    dfmodel = get_derived_modeldata(modelfolder)[0].collect()
    dfmodel = dfmodel.with_columns(Ye=(0.1 + 0.4 * pl.col("modelgridindex") / pl.len()).cast(pl.Float32))
    at.inputmodel.opacityinputfile.write_ye_file(modelfolder, dfmodel)

    with mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True) as mockimshow:
        at.inputmodel.plotinitialcomposition.main(argsraw=["Ye", str(modelfolder), "-o", str(tmp_path)])

    # the default slice is the layer above the origin along z
    expected = dfmodel.filter(pl.col("pos_z_min").abs() < 1.0).sort("inputcellid")["Ye"].to_numpy()
    np.testing.assert_allclose(np.asarray(mockimshow.call_args.args[1]).ravel(), expected, rtol=1e-6)


def test_plotinitialcomposition_rejects_a_column_or_a_model_that_it_cannot_plot(tmp_path: Path) -> None:
    """An unknown element and a 1D model give a message, not a polars error or an internal check."""
    with pytest.raises(ValueError, match="holds no column X_Xx"):
        at.inputmodel.plotinitialcomposition.main(argsraw=["Xx", str(modelpath_3d), "-o", str(tmp_path)])

    with pytest.raises(ValueError, match="2D or a 3D model"):
        at.inputmodel.plotinitialcomposition.main(argsraw=["rho", str(modelpath), "-o", str(tmp_path)])


def test_shen2018_nitrogen_in_lower_case_is_no_neutron() -> None:
    """The names of a Shen et al. 2018 file are in lower case, thus n14 is nitrogen and only n and n1 are the neutron.

    A rule that read the case of the name gave the atomic number 0 to the nitrogen columns, and the written abundance
    of nitrogen was zero.
    """
    from artistools.inputmodel.shen2018 import get_nuclide_atomic_number

    names = ("n", "n1", "n14", "n15", "p", "ni56")
    assert [get_nuclide_atomic_number(name) for name in names] == [0, 0, 7, 7, 1, 28]


def test_makeartismodelfromshen2018(tmp_path: Path) -> None:
    """The command writes the model and the abundances, with the free nucleons n and p as neutrons and hydrogen.

    get_atomic_number reads "n" as nitrogen and "p" as phosphorus, thus the neutron mass went to nitrogen and the
    proton mass went to phosphorus. An element with no column stopped the command with KeyError.
    """
    from artistools.inputmodel import shen2018

    species = ["n", "p", "he4", "ni56", "co56", "fe52", "cr48", "ni57", "co57"]
    massfracs = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]
    lines = [" ".join(["m", "v", "rho", "temp", "ye", *species])]
    lines.extend(
        " ".join(str(value) for value in [enclosedmass, velocity, 1.0, 1.0, 0.5, *massfracs])
        for enclosedmass, velocity in ((0.5, 5e8), (1.0, 1e9))
    )
    inputpath = tmp_path / "shen.dat"
    inputpath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    shen2018.main(argsraw=["-i", str(inputpath), "-o", str(tmp_path / "out")])

    dfmodel = get_derived_modeldata(tmp_path / "out", get_elemabundances=True)[0].collect()
    assert dfmodel.height == 2
    assert float(dfmodel["mass_g"].sum()) / at.constants.Msun_to_g == pytest.approx(1.0, rel=1e-5)
    assert dfmodel["X_Ni56"].to_list() == pytest.approx([0.2, 0.2])
    assert dfmodel["X_H"].to_list() == pytest.approx([0.2, 0.2])
    assert dfmodel["X_He"].to_list() == pytest.approx([0.3, 0.3])
    assert dfmodel["X_N"].to_list() == pytest.approx([0.0, 0.0])
    assert dfmodel["X_P"].to_list() == pytest.approx([0.0, 0.0])
    assert dfmodel["X_Ni"].to_list() == pytest.approx([0.21, 0.21])


def test_make1dslicefrom3dmodel(tmp_path: Path) -> None:
    """The command writes a 1D model and its abundances from the cells on the positive x axis of a 3D model."""
    from artistools.inputmodel import make1dslicefrom3d

    outfolder = tmp_path / "slice"
    make1dslicefrom3d.main(argsraw=[str(modelpath_3d), "-o", str(outfolder), "-opdf", str(tmp_path / "slice.pdf")])

    dfmodel, modelmeta = at.inputmodel.scan_modeldata(outfolder, get_elemabundances=True, printwarningsonly=True)
    # the 10 x 10 x 10 grid has 5 cells on each side of the origin
    assert modelmeta["dimensions"] == 1
    assert dfmodel.collect().height == 5
    assert (tmp_path / "slice.pdf").is_file()


def test_makeartismodelfromsingletrajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_trajectory_archive: Callable[[Path, int, dict[str, str], str], Path],
) -> None:
    """The command takes the trajectory folder and the particle id as arguments, and it writes a one-cell model."""
    particleid = 42
    # N, Z, and the log10 of the number abundance: He4 and Ni56 have a mass fraction of 0.5 each
    nuclidelines = [f"{2:4d}{2:4d}{math.log10(0.5 / 4):13.5E}", f"{28:4d}{28:4d}{math.log10(0.5 / 56):13.5E}"]
    timeseconds = 86400.0
    write_trajectory_archive(
        tmp_path / "trajectories",
        particleid,
        {
            "tday_nz-plane": f"  1  {timeseconds:.6E}   6.6E-06   2.9E-12   0.0E+00   6.3E-06\n"
            + "\n".join(nuclidelines)
            + "\n"
        },
        ".tar.xz",
    )
    # the command reads a density profile from the working folder if one exists
    monkeypatch.chdir(tmp_path)

    at.inputmodel.rprocess_from_trajectory.main(
        argsraw=["-trajectoryroot", str(tmp_path / "trajectories"), "-particleid", str(particleid), "-o", "out"]
    )

    dfmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path / "out", get_elemabundances=True, printwarningsonly=True)
    dfmodel = dfmodel.collect()
    assert modelmeta["t_model_init_days"] == pytest.approx(1.0)
    assert dfmodel["X_Ni56"].to_list() == pytest.approx([0.5], rel=1e-4)
    assert dfmodel["X_Ni"].to_list() == pytest.approx([0.5], rel=1e-4)
    assert dfmodel["X_He"].to_list() == pytest.approx([0.5], rel=1e-4)
    with at.zopen(tmp_path / "out" / "gridcontributions.txt.zst", encoding="utf-8") as fcontribs:
        assert fcontribs.read().splitlines()[1] == "42 1 1.0"


def write_e2e_model(datpath: Path, isopath: Path) -> None:
    """Write a small end-to-end model of 40 tracers with dynamical and torus ejecta, and its nuclide table."""
    from artistools.inputmodel.from_e2e_model import t_model_init_s

    rng = np.random.default_rng(seed=3)
    ntraj = 40
    ntimes = 8
    np.savez(
        datpath,
        # the polar angles reach below the equator, thus the model has no equatorial symmetry
        pos=np.column_stack([rng.uniform(0.05, 0.3, ntraj), rng.uniform(0.2, 2.9, ntraj)]),
        idx=np.arange(1, ntraj + 1, dtype=float),
        state=np.where(np.arange(ntraj) < 20, -1.0, 0.0),
        mass=np.full(ntraj, 1e-3),
        qdot=np.full((ntraj, ntimes), 1e10),
        hnuloss=np.zeros((ntraj, ntimes)),
        time=np.linspace(0.0, 2.0 * t_model_init_s, ntimes),
        # number fractions whose mass fractions sum to less than 1, because sn3d stops for a larger sum
        nz=rng.uniform(0.01, 0.2, size=(ntraj, 2)) * np.array([0.5, 0.01]),
        t5out=np.column_stack([np.zeros((ntraj, 4)), np.full(ntraj, 0.3)]),
    )
    # the neutron number and the atomic number of He4 and Zn56
    np.save(isopath, np.array([[2, 2], [26, 30]]))


def test_from_e2e_model_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command writes a 2D model and a 3D model, and --interpolate writes its check files to the -o folder.

    The interpolation with the dynamical ejecta wrote dyn_abunds.txt and dyn_model_*.txt to the working folder.
    The overwrite check before the work covered only the model files, thus it did not protect the check files.
    """
    datpath = tmp_path / "e2emodel.npz"
    write_e2e_model(datpath, tmp_path / "iso_table.npy")
    workingfolder = tmp_path / "workingfolder"
    workingfolder.mkdir()
    monkeypatch.chdir(workingfolder)

    gridargs = ["-npz", str(datpath), "-vmax_on_c", "0.4", "-ngridrcyl", "4", "-ngridz", "8"]
    at.inputmodel.from_e2e_model.main(argsraw=[*gridargs, "-o", str(tmp_path / "model2d")])
    _, modelmeta2d = at.inputmodel.scan_modeldata(tmp_path / "model2d", printwarningsonly=True)
    assert modelmeta2d["dimensions"] == 2
    assert modelmeta2d["npts_model"] == 32

    gridargs3d = [*gridargs, "--mapto3D", "-ngridx", "4", "-ngridy", "4", "-ngridz", "4"]
    at.inputmodel.from_e2e_model.main(argsraw=[*gridargs3d, "-o", str(tmp_path / "dyn3d")])
    at.inputmodel.from_e2e_model.main(
        argsraw=[
            *gridargs3d,
            "-replacedyn",
            str(tmp_path / "dyn3d"),
            "--interpolate",
            "-o",
            str(tmp_path / "interpolated"),
        ]
    )

    _, modelmeta3d = at.inputmodel.scan_modeldata(tmp_path / "interpolated", printwarningsonly=True)
    assert modelmeta3d["npts_model"] == 64
    for filename in ("dyn_abunds.txt.zst", "dyn_model_notrescaled.txt.zst", "dyn_model_rescaled.txt.zst"):
        assert (tmp_path / "interpolated" / filename).is_file(), filename
    assert not list(workingfolder.iterdir())

    # the replacement with no --interpolate stopped at a duplicate column, and a Float32 cell volume then gave rho = NaN
    at.inputmodel.from_e2e_model.main(
        argsraw=[
            *gridargs3d,
            "-replacedyn",
            str(tmp_path / "dyn3d"),
            "--globaldynscale",
            "-o",
            str(tmp_path / "replaced"),
        ]
    )
    lzreplaced, _ = at.inputmodel.scan_modeldata(tmp_path / "replaced", printwarningsonly=True)
    assert lzreplaced.select(pl.col("rho").is_finite().all()).collect().item()

    # a model of format version 2 has the cell ids 0 to N - 1, and the joins then gave each cell the next cell
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=[tmp_path / "dyn3d"], format_version=2, outputfile=tmp_path / "dyn3d_v2"
    )
    for outfoldername, modeargs in (("replaced", ["--globaldynscale"]), ("interpolated", ["--interpolate"])):
        at.inputmodel.from_e2e_model.main(
            argsraw=[
                *gridargs3d,
                "-replacedyn",
                str(tmp_path / "dyn3d_v2"),
                *modeargs,
                "-o",
                str(tmp_path / f"{outfoldername}_v2"),
            ]
        )
        for reader in (at.inputmodel.scan_modeldata, at.inputmodel.scan_initelemabundances):
            dfresults = []
            for folder in (tmp_path / outfoldername, tmp_path / f"{outfoldername}_v2"):
                result = reader(folder, printwarningsonly=True)
                lzresult = result[0] if isinstance(result, tuple) else result
                dfresults.append(lzresult.select(cs.by_name("rho", require_all=False) | cs.starts_with("X_")).collect())
            dfresult_v1, dfresult_v2 = dfresults
            assert dfresult_v1.columns == dfresult_v2.columns
            for colname in dfresult_v1.columns:
                assert np.allclose(dfresult_v2[colname], dfresult_v1[colname], rtol=1e-5, atol=0.0), (
                    outfoldername,
                    colname,
                )

    for filename in at.inputmodel.core.MODEL_FILE_NAMES:
        (tmp_path / "interpolated" / f"{filename}.zst").unlink(missing_ok=True)
    capsys.readouterr()
    with pytest.raises(SystemExit):
        at.inputmodel.from_e2e_model.main(
            argsraw=[
                *gridargs3d,
                "-replacedyn",
                str(tmp_path / "dyn3d"),
                "--interpolate",
                "-o",
                str(tmp_path / "interpolated"),
            ]
        )
    assert f"{tmp_path / 'interpolated' / 'dyn_abunds.txt.zst'} exists" in capsys.readouterr().err
    assert not (tmp_path / "interpolated" / "model.txt.zst").exists()


def test_describeinputmodel_keeps_its_description_with_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    """The description is the product of the command, thus --quiet must keep it."""
    import artistools.__main__

    artistools.__main__.main(argsraw=["describeinputmodel", str(at.get_path("testdata") / "testmodel"), "--quiet"])

    assert "kinetic energy" in capsys.readouterr().out


def test_closest_network_timesteps_agree_with_a_filter_of_each_time(trajectory_copy: Path) -> None:
    """One binary search for all the times must give the steps that one filter for each time gives."""
    from artistools.inputmodel.rprocess_from_trajectory import get_closest_network_timesteps
    from artistools.inputmodel.rprocess_from_trajectory import get_traj_network_timesteps

    dfevol = get_traj_network_timesteps(trajectory_copy, 114511).unique(subset=["timesec"], keep="first")
    arrtimesec = np.sort(dfevol["timesec"].to_numpy())
    # the times of the steps, the mid-points between them, and a time on each side of the whole range
    times = [
        float(arrtimesec[0]) * 0.5,
        *arrtimesec[::7].tolist(),
        *((arrtimesec[:-1:5] + arrtimesec[1::5]) / 2).tolist(),
        float(arrtimesec[-1]) * 2.0,
    ]

    def get_reference_step(timesec: float, cond: str) -> int | None:
        dftime = dfevol.with_columns(pl.lit(timesec, dtype=pl.Float32).alias("t"))
        if cond == "lessorequal":
            expr = pl.col("nstep").filter(pl.col("timesec") <= pl.col("t")).max()
        elif cond == "greaterorequal":
            expr = pl.col("nstep").filter(pl.col("timesec") >= pl.col("t")).min()
        else:
            dftime = dftime.sort("timesec")
            expr = pl.col("nstep").get((pl.col("timesec") - pl.col("t")).abs().arg_min())
        step: int | None = dftime.select(expr).item()
        return step

    for cond in ("lessorequal", "greaterorequal", "nearest"):
        steps = get_closest_network_timesteps(trajectory_copy, 114511, times, cond=cond)
        assert steps == [get_reference_step(timesec, cond) for timesec in times], cond

    assert get_closest_network_timesteps(trajectory_copy, 114511, times[1:2], cond="lessorequal") == [
        get_reference_step(times[1], "lessorequal")
    ]
    assert get_closest_network_timesteps(trajectory_copy, 114511, times[:1], cond="lessorequal") == [None]
    assert get_closest_network_timesteps(trajectory_copy, 114511, times[-1:], cond="greaterorequal") == [None]


def test_plotdensity_table_keeps_two_models_of_one_label_apart(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Two models of one label gave one value of the column model, and the filter took them for one line."""
    modelcopy = tmp_path / "copy"
    modelcopy.mkdir()
    for filepath in modelpath.iterdir():
        (modelcopy / filepath.name).symlink_to(filepath)
    at.inputmodel.plotdensity.main(
        argsraw=[],
        modelpath=[modelpath, modelcopy],
        label=["same", "same"],
        outputpath=tmp_path,
        write_data=True,
        quiet=True,
    )
    for filename in ("densityprofile.txt", "densityprofile_massprofile.txt"):
        dftable = read_data_file(tmp_path / filename)
        assert sorted(dftable["modelpath"].unique().to_list()) == sorted([str(modelpath), str(modelcopy)]), filename
        assert len(set(dftable.group_by("modelpath").len()["len"].to_list())) == 1, filename


def test_plotdensity_write_data_gives_only_the_velocities_of_the_axis(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The axis shows the velocities from -xmin to -xmax, and the tables also gave the points far outside the axis.

    A point outside the axis stays only as the end of a segment that crosses an edge of the axis.
    """
    at.inputmodel.plotdensity.main(
        argsraw=[], modelpath=[modelpath], outputpath=tmp_path, xmin=0.01, xmax=0.05, write_data=True, quiet=True
    )
    for filename in ("densityprofile.txt", "densityprofile_massprofile.txt"):
        dftable = read_data_file(tmp_path / filename)
        isinside = pl.col("vel_r_on_c").is_between(0.01, 0.05)
        hasinsideneighbour = isinside.shift(1).fill_null(value=False) | isinside.shift(-1).fill_null(value=False)
        assert dftable.filter(isinside).height > 0, filename
        assert dftable.select((isinside | hasinsideneighbour).all()).item(), filename


def test_isosurface_table_leaves_out_a_level_with_no_surface() -> None:
    """A level of -surfaces3d outside the range of the data gives no surface, and the table gave it."""
    pv = pytest.importorskip("pyvista")
    from artistools.inputmodel.plotinitialcomposition import get_levels_with_surface

    x, y, z = np.meshgrid(*[np.arange(4.0)] * 3, indexing="ij")
    mesh = pv.StructuredGrid(x, y, z)
    mesh["value"] = (x + y + z).ravel(order="F")
    surf = mesh.contour([1.5, 4.0, 20.0], scalars="value")
    assert get_levels_with_surface(surf, "value", [1.5, 4.0, 20.0]).tolist() == [1.5, 4.0]


def test_plotinitialcomposition_3d_tables_follow_the_order_of_the_cells() -> None:
    """--plot3d gave no table. The cells of a model change x first, thus each row must keep the velocity of its cell."""
    from artistools.inputmodel.plotinitialcomposition import get_3d_tables

    xgrid = np.array([-0.2, 0.0])
    cellvalues = np.arange(8.0)
    tables = get_3d_tables(xgrid, cellvalues.reshape((2, 2, 2), order="F"), "rho", np.array([1.0, 5.0]))
    dfcells = tables[""]
    assert dfcells["rho"].to_list() == cellvalues.tolist()
    assert dfcells["vel_x_min_on_c"].to_list() == [-0.2, 0.0] * 4
    assert dfcells["vel_z_min_on_c"].to_list() == [-0.2] * 4 + [0.0] * 4
    assert tables["_isosurfaces"]["rho_of_isosurface"].to_list() == [1.0, 5.0]


def test_describeinputmodel_reads_the_grid_size_from_artisoptions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The mapped mass uses the grid size of artisoptions.h, because grid.out holds only the non-empty cells.

    Here only the bottom z layer of a 4^3 grid is non-empty. The size from the top cell index of grid.out was 3.
    """
    shutil.copy(modelpath / "model.txt", tmp_path)
    at.inputmodel.save_empty_abundance_file(npts_model=1, outputfilepath=tmp_path)
    Path(tmp_path, "grid.out").write_text("".join(f"{cellindex} 0\n" for cellindex in range(16)), encoding="utf-8")

    at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=[tmp_path])
    assert "gives CUBOID_NCOORDGRID_X" in capsys.readouterr().err

    Path(tmp_path, "artisoptions.h").write_text("constexpr int CUBOID_NCOORDGRID_X = 4;\n", encoding="utf-8")
    at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=[tmp_path])
    # 16 cells of width vmax t / 2 hold 16 / 8 / (4 pi / 3) = 0.4775 of the mass of the sphere
    assert "when mapped to 4^3 cubic grid, error -52.25%" in capsys.readouterr().out


def test_cone_shell_with_mass_and_no_elements_keeps_zero_mass_fractions(capsys: pytest.CaptureFixture[str]) -> None:
    """A shell with mass and only zero element mass fractions gets zero mass fractions and a warning.

    The normalisation divided by the zero sum, thus it raised ZeroDivisionError.
    """
    cone = pl.DataFrame({
        "pos_r_mid": [0.5, 1.5],
        "mass_g": [1.0, 2.0],
        "volume": [1.0, 1.0],
        "X_Fe": [0.5, 0.0],
        "X_Ni": [0.0, 0.0],
    })

    shellrows = at.inputmodel.slice1dfromconein3dmodel.get_cone_shells(
        cone, cone1d_bins=[0.0, 1.0, 2.0], speciescols=["X_Fe", "X_Ni"], logprint=print
    )

    assert shellrows[0]["X_Fe"] == pytest.approx(1.0)
    assert (shellrows[1]["X_Fe"], shellrows[1]["X_Ni"]) == (0.0, 0.0)
    assert shellrows[1]["rho"] == pytest.approx(2.0)
    assert "Shell 2 has mass but no element mass fractions" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("idx", "state"),
    [
        # two non-dynamical tracers with the same id modulo 10000 merged, and the merge lost the mass of one
        ([1, 10001], [0, 0]),
        # a non-dynamical id equal to a dynamical id took the mass of the dynamical pieces
        ([1, 1, 10001], [0, -1, -1]),
    ],
)
def test_from_e2e_model_refuses_tracer_ids_that_merge(tmp_path: Path, idx: list[int], state: list[int]) -> None:
    """get_grid stops with an error for tracer ids that id % 10000 merges, because the merge loses mass."""
    from artistools.inputmodel.from_e2e_model import get_grid
    from artistools.inputmodel.from_e2e_model import t_model_init_s

    ntraj = len(idx)
    ntimes = 8
    datpath = tmp_path / "e2emodel.npz"
    isopath = tmp_path / "iso_table.npy"
    np.savez(
        datpath,
        pos=np.column_stack([np.full(ntraj, 0.1), np.full(ntraj, 1.0)]),
        idx=np.array(idx, dtype=float),
        state=np.array(state, dtype=float),
        mass=np.full(ntraj, 1e-3),
        qdot=np.full((ntraj, ntimes), 1e10),
        hnuloss=np.zeros((ntraj, ntimes)),
        time=np.linspace(0.0, 2.0 * t_model_init_s, ntimes),
        nz=np.full((ntraj, 2), 0.1),
        t5out=np.column_stack([np.zeros((ntraj, 4)), np.full(ntraj, 0.3)]),
    )
    np.save(isopath, np.array([[2.0, 2.0], [30.0, 26.0]]))

    with pytest.raises(ValueError, match="unique id below 10000"):
        get_grid(
            datpath,
            isopath,
            0.4,
            model_dim=2,
            grid_dims=np.array([4, 8]),
            nodynej=False,
            nohmns=False,
            notorus=False,
            no_nu_trapping=False,
        )


def test_describeinputmodel_1d_model_without_grid_out_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A 1D model without grid.out gives a warning and no mapped mass, because ARTIS can map it to a 3D grid.

    describeinputmodel took a direct map when grid.out did not exist, and it gave no reason.
    """
    shutil.copy(modelpath / "model.txt", tmp_path)
    at.inputmodel.save_empty_abundance_file(npts_model=1, outputfilepath=tmp_path)

    at.inputmodel.describeinputmodel.main(argsraw=[], modelpath=[tmp_path])

    output = capsys.readouterr()
    assert "grid.out does not exist" in output.err
    assert "is direct" not in output.out
    assert "M_tot_rho_map" not in output.out
    assert "M_tot_rho " in output.out
    # abundances.txt holds zeros only. The list of the species masses was empty, and min() of it raised ValueError
    assert "no species has a mass above zero" in output.out


def test_add_mass_to_center_adds_the_mass_below_0p1c() -> None:
    """The cells of the hole get m(0) - m(0.1 c) = 3e-4 Msun, because m(v) of the source is the mass above v.

    The integral of m(v) over v/c gave 1.8e-5 Msun. The density came from the volume of the sphere of 0.1 c, but
    the 8 inner cubes of this 4^3 grid hold 0.81 of that volume, thus the cells got 0.81 of the mass.
    """
    from artistools.constants import C_cm_per_s as CLIGHT
    from artistools.constants import Msun_to_g
    from artistools.inputmodel.modelfromhydro import add_mass_to_center

    t_model_days = 1.0
    ncoordgrid = 4
    xmax = 0.15 * CLIGHT * t_model_days * day_to_s
    wid_init = 2 * xmax / ncoordgrid
    axisposmin = [-xmax + index * wid_init for index in range(ncoordgrid)]
    cellpositions = [(x, y, z) for z in axisposmin for y in axisposmin for x in axisposmin]
    griddata = pl.DataFrame({
        "inputcellid": range(1, ncoordgrid**3 + 1),
        "pos_x_min": [pos[0] for pos in cellpositions],
        "pos_y_min": [pos[1] for pos in cellpositions],
        "pos_z_min": [pos[2] for pos in cellpositions],
        "rho": [0.0] * ncoordgrid**3,
        "Ye": [0.0] * ncoordgrid**3,
    })
    modelmeta = {
        "dimensions": 3,
        "t_model_init_days": t_model_days,
        "vmax_cmps": xmax / (t_model_days * day_to_s),
        "ncoordgridx": ncoordgrid,
        "ncoordgridy": ncoordgrid,
        "ncoordgridz": ncoordgrid,
        "wid_init_x": wid_init,
        "wid_init_y": wid_init,
        "wid_init_z": wid_init,
    }

    dffilled = add_mass_to_center(griddata, modelmeta)

    assert (dffilled["rho"] > 0.0).sum() == 8
    added_mass_msun = float((dffilled["rho"] * wid_init**3).sum()) / Msun_to_g
    assert added_mass_msun == pytest.approx(3e-4 - 1e-9, rel=1e-9)


@pytest.mark.parametrize("model_dim", [2, 3])
def test_from_e2e_model_keeps_original_particle_ids(tmp_path: Path, model_dim: int) -> None:
    from artistools.inputmodel.from_e2e_model import get_grid

    datpath = tmp_path / "tracers.npz"
    isopath = tmp_path / "iso_table.npy"
    write_e2e_model(datpath, isopath)
    with np.load(datpath) as source:
        data = {name: source[name] for name in source.files}
    original_ids = np.arange(len(data["idx"])) * 13 + 17
    data["idx"] = original_ids
    np.savez(datpath, **data)
    shuffledpath = tmp_path / "shuffled.npz"
    order = np.random.default_rng(seed=123).permutation(len(original_ids))
    np.savez(shuffledpath, **{name: values if name == "time" else values[order] for name, values in data.items()})
    contributions = [
        get_grid(
            path,
            isopath,
            0.4,
            model_dim=model_dim,
            grid_dims=np.array([4, 4] if model_dim == 2 else [4, 4, 4]),
            nodynej=False,
            nohmns=False,
            notorus=False,
            no_nu_trapping=False,
        )[-1].sort("cellindex", "particleid")
        for path in (datpath, shuffledpath)
    ]
    assert len(contributions[0]) > 0
    assert set(contributions[0]["particleid"]) <= set(original_ids)
    pltest.assert_frame_equal(contributions[0], contributions[1])


@pytest.mark.parametrize("extension", ["", ".zst"])
@pytest.mark.parametrize("failure", ["validation", "write"])
def test_abundance_write_failure_keeps_the_previous_file(tmp_path: Path, extension: str, failure: str) -> None:
    output = tmp_path / f"abundances.txt{extension}"
    original = b"previous abundance data"
    output.write_bytes(original)
    abundance = pl.DataFrame({"inputcellid": [1], "X_H": [1.0]})
    if failure == "validation":
        abundance = abundance.with_columns(pl.lit(2).alias("inputcellid"))
        with pytest.raises(ValueError, match="must start at 0 or 1"):
            at.inputmodel.save_initelemabundances(abundance, outpath=output)
    else:
        msg = "Write failed"
        with (
            mock.patch("artistools.inputmodel.core.write_artis_csv", side_effect=OSError(msg)),
            pytest.raises(OSError, match=msg),
        ):
            at.inputmodel.save_initelemabundances(abundance, outpath=output)
    assert output.read_bytes() == original
    assert sorted(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("extension", ["", ".zst"])
def test_model_write_failure_keeps_the_previous_file(tmp_path: Path, extension: str) -> None:
    """A write of model.txt that stops early leaves the old model.txt, e.g. of a conversion in its model folder.

    The writer deleted the old file and then wrote the new file at its place, thus a stop left a part of the file.
    """
    output = tmp_path / f"model.txt{extension}"
    original = b"previous model data"
    output.write_bytes(original)
    dfmodel = pl.DataFrame({"inputcellid": [0], "rho": [1e-12]})
    modelmeta = {"coordinates": "spherical1d", "formatversion": 2, "t_model_init_days": 1.0}
    msg = "Write failed"
    with (
        mock.patch("artistools.inputmodel.core.write_artis_csv", side_effect=OSError(msg)),
        pytest.raises(OSError, match=msg),
    ):
        at.inputmodel.save_modeldata(
            dfmodel.with_columns(vel_r_max_kmps=pl.lit(1000.0), logrho=pl.lit(-12.0)),
            outpath=tmp_path,
            modelmeta=modelmeta,
        )
    assert output.read_bytes() == original
    assert sorted(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("empty_slice", [False, True])
def test_random_density_perturbations_preserve_each_slice_mass(empty_slice: bool) -> None:
    from artistools.inputmodel.from_e2e_model import apply_density_perturbations

    x, y = np.meshgrid(np.arange(4) - 1.5, np.arange(4) - 1.5)
    rho = np.tile((1.0 + x**2 + y**2).ravel(), 4).reshape((4, 16))
    if empty_slice:
        rho[-1] = 0.0
    model = pl.DataFrame({"rho": rho.ravel()})
    with mock.patch("numpy.random.default_rng", return_value=np.random.default_rng(seed=123)):
        perturbed = apply_density_perturbations(model, 0.5, ("random", 0.5))
    rho_perturbed = perturbed["rho"].to_numpy().reshape((4, 16))
    np.testing.assert_allclose(rho_perturbed.sum(axis=1), rho.sum(axis=1), rtol=1e-12)
    assert not np.allclose(rho_perturbed, rho)
    assert np.all(np.isfinite(rho_perturbed))


def test_from_e2e_model_rejects_unequal_3d_dimensions_before_a_write(tmp_path: Path) -> None:
    """The model writer needs a cubic grid, thus only a model that stays 3D needs equal dimensions.

    The command did the map and wrote abundances.txt before the model writer stopped at the unequal dimensions.
    """
    datpath = tmp_path / "e2emodel.npz"
    write_e2e_model(datpath, tmp_path / "iso_table.npy")
    gridargs = ["-npz", str(datpath), "-vmax_on_c", "0.4", "--mapto3D", "-ngridx", "4", "-ngridy", "6", "-ngridz", "8"]

    with pytest.raises(ValueError, match="equal --ngridx"):
        at.inputmodel.from_e2e_model.main(argsraw=[*gridargs, "-o", str(tmp_path / "model3d")])
    # parse_cli_args makes the -o folder of each command, thus the folder exists
    assert not list((tmp_path / "model3d").iterdir())

    at.inputmodel.from_e2e_model.main(argsraw=[*gridargs, "-dimensions", "1", "-o", str(tmp_path / "model1d")])
    _, modelmeta1d = at.inputmodel.scan_modeldata(tmp_path / "model1d", printwarningsonly=True)
    assert modelmeta1d["dimensions"] == 1


def get_random_spherical_model(coordinates: str) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return a model with spherical cell edges that are not uniform and random cell values, and its metadata."""
    rng = np.random.default_rng(seed=1)
    modelmeta: dict[str, t.Any] = {
        "coordinates": coordinates,
        "dimensions": int(coordinates[-2]),
        "t_model_init_days": 0.5,
        "vel_r_edges_kmps": [0.0, 2000.0, 4500.0, 9000.0, 30000.0],
        "theta_edges_radians": [*np.arccos([1.0, 0.4, -0.2]).tolist(), math.pi],
        "vmax_cmps": 30000.0 * at.constants.km_to_cm,
        "firstinputcellid": 0,
        "headercommentlines": ["a comment of the user"],
    }
    ncells = 4 * 3
    if coordinates == "spherical3d":
        modelmeta["phi_edges_radians"] = [0.0, 1.0, 2.5, 4.0, 5.5, 2.0 * math.pi]
        ncells *= 5
    dfmodel = pl.DataFrame({
        "inputcellid": pl.int_range(ncells, eager=True, dtype=pl.Int32),
        "rho": rng.random(ncells) * 1e-12,
        "X_Fegroup": np.ones(ncells),
        "X_Ni56": rng.random(ncells),
        "X_Co56": np.zeros(ncells),
        "X_Fe52": np.zeros(ncells),
        "X_Cr48": np.zeros(ncells),
        "Ye": rng.random(ncells),
        "tracercount": rng.integers(0, 5, ncells).astype(np.int32),
    })
    return dfmodel, modelmeta


@pytest.mark.parametrize("coordinates", ["spherical2d", "spherical3d"])
def test_spherical_model_round_trip(tmp_path: Path, coordinates: str) -> None:
    """A spherical2d or spherical3d model keeps its edges, its cells, and its volume through a write and a read.

    model.txt has no position column for these coordinates. Thus the edges of the header and the id of a cell give
    each position. An error in the cell order or in an edge shows in the volume and in the mid-points.
    """
    dfmodel, modelmeta = get_random_spherical_model(coordinates)
    # The reader gives Float32 values, and the nine significant figures of version 2 give each of them again. Eight
    # figures give a different Float32 for the density of the first cell
    dfmodel = dfmodel.with_columns(
        rho=pl.when(pl.col("inputcellid") == 0).then(1.04083036e-13).otherwise(pl.col("rho"))
    ).with_columns(cs.float().cast(pl.Float32))
    edgekeys = [key for key in ("vel_r_edges_kmps", "theta_edges_radians", "phi_edges_radians") if key in modelmeta]
    edges_written = {key: list(modelmeta[key]) for key in edgekeys}

    # a version 2 model has no opacity.txt. The file can hold the only copy of its values, thus the writer deletes it
    # only when the caller agrees
    (tmp_path / "opacity.txt").write_text("1\n1 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"opacity\.txt gives values of the cells"):
        at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta=dict(modelmeta))
    assert (tmp_path / "opacity.txt").is_file()
    assert not list(tmp_path.glob("model.txt*"))
    at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta=modelmeta, deletecellidfiles=True)
    assert not (tmp_path / "opacity.txt").exists()
    lzmodel_read, modelmeta_read = at.inputmodel.scan_modeldata(tmp_path)

    # the file has no composition column that holds zero in each cell, and the reader gives such a column again
    with at.zopen(tmp_path / "model.txt.zst") as fmodel:
        headerlines = fmodel.read().splitlines()[: -dfmodel.height]
    assert headerlines[-2:] == [
        f"cell_line_count {dfmodel.height}",
        "columns inputcellid rho X_Fegroup X_Ni56 Ye tracercount",
    ]
    # the key cells_per_axis gives the cell counts, thus the line of the coordinates has no comment
    coordinatesindex = headerlines.index(f"coordinates {coordinates}")
    gridcounts = " ".join(str(len(modelmeta[key]) - 1) for key in edgekeys)
    assert headerlines[coordinatesindex + 1] == f"cells_per_axis {gridcounts}"

    # the writer gives 17 significant digits, thus each edge comes back with no change
    assert {key: modelmeta_read[key] for key in edgekeys} == edges_written
    assert modelmeta_read["coordinates"] == coordinates
    assert modelmeta_read["headercommentlines"] == ["a comment of the user"]
    assert modelmeta_read["npts_model"] == dfmodel.height == at.misc.get_npts_model(tmp_path)
    pltest.assert_frame_equal(
        lzmodel_read.drop("modelgridindex").collect(), dfmodel, check_dtypes=False, check_exact=True
    )

    dfderived = at.inputmodel.add_derived_cols_to_modeldata(lzmodel_read, modelmeta_read).collect()
    radius_outer = modelmeta_read["vmax_cmps"] * modelmeta_read["t_model_init_days"] * day_to_s
    assert math.isclose(dfderived["volume"].sum(), 4.0 / 3.0 * math.pi * radius_outer**3, rel_tol=1e-12)
    assert np.allclose(
        np.hypot(dfderived["vel_rcyl_mid"].to_numpy(), dfderived["vel_z_mid"].to_numpy()),
        dfderived["vel_r_mid"].to_numpy(),
        rtol=1e-12,
        atol=0.0,
    )
    if coordinates == "spherical3d":
        assert np.allclose(
            np.hypot(dfderived["vel_x_mid"].to_numpy(), dfderived["vel_y_mid"].to_numpy()),
            dfderived["vel_rcyl_mid"].to_numpy(),
            rtol=1e-12,
            atol=0.0,
        )

    # the id of a cell gives its index in the grid. Thus a selection of the rows in a different order gives the
    # same values
    isselected = pl.col("inputcellid") % 3 == 0
    pltest.assert_frame_equal(
        at.inputmodel.add_derived_cols_to_modeldata(
            lzmodel_read.filter(isselected).sort("rho"), modelmeta_read
        ).collect(),
        dfderived.filter(isselected).sort("rho"),
    )


SPHERICAL2D_MODEL_LINES = (
    "format_version 2",
    "# a comment before the keys",
    "coordinates spherical2d",
    "# a comment between two keys",
    "t_model_init_days 1.0  # an inline comment",
    "cell_line_count 4",
    "vel_r_edges_kmps 0 1000 3000",
    f"theta_edges_radians 0 1.5 {math.pi!r}",
    "columns inputcellid rho X_Fegroup",
    "0 1e-12 1",
    "1 2e-12 1",
    "2 3e-12 1",
    "3 4e-12 1",
)


def test_spherical_model_reader_takes_cell_ids_from_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The first cell id of a file sets the index of each cell, also with no modelgridindex column.

    The reader also takes a comment line before each key line.
    """
    (tmp_path / "model.txt").write_text("\n".join(SPHERICAL2D_MODEL_LINES) + "\n", encoding="utf-8")
    lzmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path)

    assert modelmeta["firstinputcellid"] == 0
    assert modelmeta["headercommentlines"] == ["a comment before the keys", "a comment between two keys"]

    # the keys can come in any sequence before the key columns, and a key that artistools does not know gets a warning
    lines = [SPHERICAL2D_MODEL_LINES[0], "key_of_a_later_version 1", *reversed(SPHERICAL2D_MODEL_LINES[1:8])]
    # a comment line or an empty line can come before each cell line
    celllines = [SPHERICAL2D_MODEL_LINES[8], "# a comment before the first cell", "", SPHERICAL2D_MODEL_LINES[9]]
    (tmp_path / "model.txt").write_text(
        "\n".join([*lines, *celllines, "# a comment", "", *SPHERICAL2D_MODEL_LINES[10:]]) + "\n", encoding="utf-8"
    )
    dfreordered, metareordered = at.inputmodel.scan_modeldata(tmp_path)
    pltest.assert_frame_equal(dfreordered.collect(), lzmodel.collect())
    assert metareordered["theta_edges_radians"] == modelmeta["theta_edges_radians"]
    captured = capsys.readouterr()
    assert "does not know the key key_of_a_later_version" in captured.out + captured.err

    # a model with no density keeps one cell line, because a file with no cell line is not valid
    emptyfolder = tmp_path / "empty"
    emptyfolder.mkdir()
    # a version 2 model has no Ye.txt, thus the file of an earlier model goes, also for a model with no column Ye
    (emptyfolder / "Ye.txt").write_text("1\n1 0.3\n", encoding="utf-8")
    at.inputmodel.save_modeldata(
        lzmodel.with_columns(rho=pl.lit(0.0, dtype=pl.Float32)).collect(),
        outpath=emptyfolder,
        modelmeta=dict(modelmeta),
        deletecellidfiles=True,
    )
    assert not (emptyfolder / "Ye.txt").exists()
    # the count of the cell lines is 1, and the file has no composition column that is zero in each cell with density
    with at.zopen(emptyfolder / "model.txt.zst") as fmodel:
        assert fmodel.read().splitlines()[-3:-1] == ["cell_line_count 1", "columns inputcellid rho"]
    dfempty = at.inputmodel.scan_modeldata(emptyfolder)[0].collect()
    assert dfempty["inputcellid"].to_list() == [0, 1, 2, 3]
    assert dfempty["rho"].to_list() == [0.0, 0.0, 0.0, 0.0]

    # a cell with no line is empty in format version 2, and the count of the cell lines agrees with the file
    lines = [
        line.replace("cell_line_count 4", "cell_line_count 3")
        for line in SPHERICAL2D_MODEL_LINES
        if line != "2 3e-12 1"
    ]
    (tmp_path / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dfwithabsentcell = at.inputmodel.scan_modeldata(tmp_path)[0].collect()
    assert dfwithabsentcell["inputcellid"].to_list() == [0, 1, 2, 3]
    assert dfwithabsentcell["rho"].to_list() == pytest.approx([1e-12, 2e-12, 0.0, 4e-12])
    assert dfwithabsentcell["X_Fegroup"].to_list() == [1.0, 1.0, 0.0, 1.0]
    # a composition column that the file does not have holds zero, at its place in a version 1 file
    assert dfwithabsentcell.columns[:7] == ["inputcellid", "rho", "X_Fegroup", "X_Ni56", "X_Co56", "X_Fe52", "X_Cr48"]
    assert dfwithabsentcell["X_Ni56"].to_list() == [0.0, 0.0, 0.0, 0.0]
    dfderived = at.inputmodel.add_derived_cols_to_modeldata(lzmodel.drop("modelgridindex"), modelmeta).collect()
    # the radial index changes fastest, thus the cells 0 and 2 are in the inner shell
    assert dfderived["vel_r_max"].to_list() == pytest.approx([1e8, 3e8, 1e8, 3e8])
    assert dfderived["theta_min_radians"].to_list() == pytest.approx([0.0, 0.0, 1.5, 1.5])
    # metadata with no key firstinputcellid gave the first cell the edges of the last cell. The cell ids of format
    # version 2 start at 0
    metawithoutkey = {key: value for key, value in modelmeta.items() if key != "firstinputcellid"}
    pltest.assert_frame_equal(
        at.inputmodel.add_derived_cols_to_modeldata(lzmodel.drop("modelgridindex"), metawithoutkey).collect(), dfderived
    )


def get_mirrored_model_and_twin(
    coordinates: str,
) -> tuple[pl.DataFrame, dict[str, t.Any], pl.DataFrame, dict[str, t.Any], np.ndarray[t.Any, np.dtype[np.int64]]]:
    """Return a model with mirror symmetry in the plane z = 0, its full twin, and the half cell of each twin cell.

    The twin has the cells of both halves, and each pair of mirror cells has the values of the cell of the mirrored
    model. The cell index of each list is the model grid index, in which the first axis changes fastest.
    """
    rng = np.random.default_rng(seed=3)
    # the position columns of a cylindrical model. A spherical model has none
    mirrorpos: dict[str, np.ndarray[t.Any, np.dtype[np.floating]]]
    twinpos: dict[str, np.ndarray[t.Any, np.dtype[np.floating]]]
    if coordinates == "cylindrical2d":
        n0, nhalf, n2 = 3, 2, 1
        vmax_cmps = 3e9
        tmodel_s = 0.5 * day_to_s
        rcylwidth_cm = vmax_cmps * tmodel_s / n0
        zwidth_cm = vmax_cmps * tmodel_s / nhalf
        index0, index1 = np.arange(n0 * nhalf) % n0, np.arange(n0 * nhalf) // n0
        common = {"coordinates": coordinates, "t_model_init_days": 0.5, "vmax_cmps": vmax_cmps, "ncoordgridrcyl": n0}
        mirrormeta = common | {"ncoordgridz": nhalf, "mirror_plane": "z=0"}
        twinmeta = common | {"ncoordgridz": 2 * nhalf}
        mirrorpos = {"pos_rcyl_mid": (index0 + 0.5) * rcylwidth_cm, "pos_z_mid": (index1 + 0.5) * zwidth_cm}
        twinindex0, twinindex1 = np.arange(n0 * 2 * nhalf) % n0, np.arange(n0 * 2 * nhalf) // n0
        twinpos = {
            "pos_rcyl_mid": (twinindex0 + 0.5) * rcylwidth_cm,
            "pos_z_mid": (twinindex1 + 0.5) * zwidth_cm - vmax_cmps * tmodel_s,
        }
        # the z index of the half starts at the plane z = 0
        halfindex1 = np.where(twinindex1 >= nhalf, twinindex1 - nhalf, nhalf - 1 - twinindex1)
    else:
        n0, nhalf, n2 = 4, 2, 3
        halfthetaedges = [0.0, 0.6, math.pi / 2.0]
        common = {
            "coordinates": coordinates,
            "t_model_init_days": 0.5,
            "vel_r_edges_kmps": [0.0, 2000.0, 4500.0, 9000.0, 30000.0],
            "phi_edges_radians": [0.0, 1.0, 4.5, 2.0 * math.pi],
        }
        mirrormeta = common | {"theta_edges_radians": halfthetaedges, "mirror_plane": "z=0"}
        twinmeta = common | {"theta_edges_radians": [*halfthetaedges, math.pi - 0.6, math.pi]}
        mirrorpos = dict[str, np.ndarray[t.Any, np.dtype[np.floating]]]()
        twinpos = mirrorpos
        twinindex0 = np.arange(n0 * 2 * nhalf * n2) % n0
        twinindex1 = (np.arange(n0 * 2 * nhalf * n2) // n0) % (2 * nhalf)
        halfindex1 = np.where(twinindex1 < nhalf, twinindex1, 2 * nhalf - 1 - twinindex1)
    twinindex2 = np.arange(n0 * 2 * nhalf * n2) // (n0 * 2 * nhalf)
    halfcell = twinindex0 + n0 * (halfindex1 + nhalf * twinindex2)
    ncells = n0 * nhalf * n2
    mirrorcells = pl.DataFrame({
        "inputcellid": np.arange(ncells, dtype=np.int32),
        **mirrorpos,
        "rho": rng.random(ncells) * 1e-12,
        "X_Ni56": rng.random(ncells),
    })
    twincells = pl.DataFrame({
        "inputcellid": np.arange(2 * ncells, dtype=np.int32),
        **twinpos,
        "rho": mirrorcells["rho"].to_numpy()[halfcell],
        "X_Ni56": mirrorcells["X_Ni56"].to_numpy()[halfcell],
    })
    return mirrorcells, mirrormeta, twincells, twinmeta, halfcell


@pytest.mark.parametrize("coordinates", ["cylindrical2d", "spherical3d"])
def test_mirrored_model_has_the_masses_and_the_cells_of_its_full_twin(tmp_path: Path, coordinates: str) -> None:
    """A model with the key mirror_plane z=0 gives the masses and the cell search of the full model of both halves.

    Each cell of the file also fills its mirror cell. Thus its volume and its mass are two times the values of the
    half cell, and a velocity with z < 0 is in the cell of its mirror velocity.
    """
    mirrorcells, mirrormeta, twincells, twinmeta, halfcell = get_mirrored_model_and_twin(coordinates)
    readmodels: dict[str, tuple[pl.DataFrame, dict[str, t.Any]]] = {}
    for name, dfcells, meta in (("mirror", mirrorcells, mirrormeta), ("twin", twincells, twinmeta)):
        at.inputmodel.save_modeldata(dfcells, outpath=tmp_path / name, modelmeta=meta | {"formatversion": 2})
        lzmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path / name)
        readmodels[name] = (at.inputmodel.add_derived_cols_to_modeldata(lzmodel, modelmeta).collect(), modelmeta)
    dfmirror, metamirror = readmodels["mirror"]
    dftwin, metatwin = readmodels["twin"]

    with at.zopen(tmp_path / "mirror" / "model.txt.zst") as fmodel:
        assert "mirror_plane z=0" in fmodel.read().splitlines()
    assert at.misc.has_mirror_plane(metamirror)
    assert not at.misc.has_mirror_plane(metatwin)
    # the cells of a mirror pair together have the mass of their cell in the mirrored model
    twinpairmass = np.bincount(halfcell, weights=dftwin.sort("modelgridindex")["mass_g"].to_numpy())
    np.testing.assert_allclose(dfmirror.sort("modelgridindex")["mass_g"].to_numpy(), twinpairmass, rtol=1e-6)
    assert math.isclose(dfmirror["volume"].sum(), dftwin["volume"].sum(), rel_tol=1e-6)

    # a velocity with z < 0 is in the mirror cell, thus the twin cell of each velocity has the half cell of the file
    rng = np.random.default_rng(seed=4)
    velocities = rng.uniform(-2.9e9, 2.9e9, size=(3000, 3))
    if coordinates == "cylindrical2d":
        positions = {f"em_pos{axis}": velocities[:, i] * 1e5 for i, axis in enumerate("xyz")}
        dfpackets = pl.DataFrame(positions | {"em_time": np.full(len(velocities), 1e5)})
        cellsof = {
            name: dfpackets
            .select(at.packets.core.get_modelgridindex_expr("em", meta, pl.LazyFrame()).fill_null(-1))
            .to_series()
            .to_numpy()
            for name, meta in (("mirror", metamirror), ("twin", metatwin))
        }
    else:
        cellsof = {
            name: at.inputmodel.get_spherical_cell_index_of_velocities(
                velocities[:, 0], velocities[:, 1], velocities[:, 2], meta
            )
            for name, meta in (("mirror", metamirror), ("twin", metatwin))
        }
        dfvelocities = pl.DataFrame({"vel_x": velocities[:, 0], "vel_y": velocities[:, 1], "vel_z": velocities[:, 2]})
        vel_r = (pl.col("vel_x") ** 2 + pl.col("vel_y") ** 2 + pl.col("vel_z") ** 2).sqrt()
        packetcells = (
            dfvelocities
            .select(
                at.packets.get_spherical_modelgridindex_expr(
                    vel_r, pl.col("vel_x"), pl.col("vel_y"), pl.col("vel_z"), metamirror
                ).fill_null(-1)
            )
            .to_series()
            .to_numpy()
        )
        assert np.array_equal(packetcells, cellsof["mirror"])
    isingrid = cellsof["twin"] >= 0
    assert isingrid.sum() > 100
    assert np.array_equal(cellsof["mirror"][isingrid], halfcell[cellsof["twin"][isingrid]])

    # each cell fills both halves, thus a range of directions cannot select the part of a cell in the range
    args = argparse.Namespace(
        dirbin=0, costhetamin=None, costhetamax=None, average_over_phi_angle=True, average_over_theta_angle=False
    )
    assert at.inputmodel.resolve_cell_direction_range(args, tmp_path / "twin", quiet=True) is not None
    with pytest.raises(SystemExit):
        at.inputmodel.resolve_cell_direction_range(args, tmp_path / "mirror", quiet=True)

    if coordinates == "cylindrical2d":
        # the image shows both halves, thus the full grid of the mirrored model has the cells of the twin
        from artistools.inputmodel.plotinitialcomposition import get_full_grid_of_cylindrical_model

        dffull, metafull = get_full_grid_of_cylindrical_model(dfmirror, metamirror)
        assert metafull["ncoordgridz"] == metatwin["ncoordgridz"]
        np.testing.assert_allclose(dffull["rho"].to_numpy(), dftwin.sort("modelgridindex")["rho"].to_numpy())
        np.testing.assert_allclose(
            dffull["pos_z_mid"].to_numpy(), dftwin.sort("modelgridindex")["pos_z_mid"].to_numpy()
        )
        # format version 1 has no mirror symmetry
        with pytest.raises(ValueError, match="has no mirror symmetry"):
            at.inputmodel.save_modeldata(
                mirrorcells, outpath=tmp_path / "v1", modelmeta=mirrormeta | {"formatversion": 1}
            )
    else:
        # a reduction to 2D keeps the mirror plane, and a reduction to 1D gives the shells of both halves
        for outputdimensions, keepsmirror in ((2, True), (1, False)):
            dfreduced, _, _, metareduced = at.inputmodel.dimension_reduce_model(
                dfmirror.select(*mirrorcells.columns, "modelgridindex"), outputdimensions, modelmeta=dict(metamirror)
            )
            assert at.misc.has_mirror_plane(metareduced) == keepsmirror
            assert math.isclose(dfreduced["mass_g"].sum(), dftwin["mass_g"].sum(), rel_tol=1e-6)


def test_spherical_image_has_a_point_in_each_narrow_polar_cell(capsys: pytest.CaptureFixture[str]) -> None:
    """The image of a model with many narrow polar cells has a point in each cell of the outer shell.

    The image had 200 points from 0 to vmax for each model with 100 radial shells or fewer. A polar cell narrower than
    the step of the points then had no point, and its value was not in the image. A cell of an inner shell is
    narrower, thus the image gives a warning with the radius inside which such a cell can have no point.
    """
    modelmeta = {
        "coordinates": "spherical2d",
        "vmax_cmps": 1e9,
        "vel_r_edges_kmps": [0.0, 5000.0, 10000.0],
        "theta_edges_radians": np.linspace(0.0, math.pi, 1001).tolist(),
    }
    cellindices = at.inputmodel.get_spherical_image_cell_indices(modelmeta, sliceaxis=None)
    # the model grid index of a cell in the outer shell is 1 + 2 * i_theta
    outercells = {int(cellindex) for cellindex in cellindices.ravel() if cellindex >= 0 and cellindex % 2 == 1}
    assert outercells == set(range(1, 2 * 1000, 2))
    # the inner shell ends at 0.5 vmax, and two points across each of its polar cells need more than 1000 points
    captured = capsys.readouterr()
    assert (
        "Inside 0.637 vmax of the centre of the plane, a cell can have less than two points across it"
        in captured.out + captured.err
    )

    # each polar cell of a usual grid, uniform in cos(theta), has a point in each shell
    usualcells = at.inputmodel.get_spherical_image_cell_indices(
        modelmeta
        | {
            "vel_r_edges_kmps": np.linspace(0.0, 10000.0, 26).tolist(),
            "theta_edges_radians": np.arccos(np.linspace(1.0, -1.0, 17)).tolist(),
        },
        sliceaxis=None,
    )
    captured = capsys.readouterr()
    assert "less than two points across it" not in captured.out + captured.err
    assert set(usualcells.ravel().tolist()) == {-1, *range(25 * 16)}


def test_spherical_image_has_a_point_in_a_small_inner_shell_and_in_each_phi_cell_of_a_far_slice() -> None:
    """The image has a point in a small inner shell with one polar cell, and in each phi cell of a slice near vmax.

    The count of the points came from the narrowest angular cell at the outer edge of the innermost shell. A wide
    polar cell then gave too few points for the radius of that shell. A slice near vmax cuts the shells in a small
    circle, and its phi cells were narrower than the step of the points.
    """
    innershellmeta = {
        "coordinates": "spherical2d",
        "vmax_cmps": 1e9,
        "vel_r_edges_kmps": [0.0, 30.0, 10000.0],
        "theta_edges_radians": [0.0, math.pi],
    }
    cellindices = at.inputmodel.get_spherical_image_cell_indices(innershellmeta, sliceaxis=None)
    assert 0 in cellindices

    nphi = 360
    farslicemeta = {
        "coordinates": "spherical3d",
        "vmax_cmps": 1e9,
        "vel_r_edges_kmps": [0.0, 5000.0, 10000.0],
        "theta_edges_radians": np.arccos(np.linspace(1.0, -1.0, 5)).tolist(),
        "phi_edges_radians": np.linspace(0.0, 2.0 * math.pi, nphi + 1).tolist(),
    }
    cellindices = at.inputmodel.get_spherical_image_cell_indices(farslicemeta, "z", 0.99e9)
    # the model grid index is i_r + 2 * (i_theta + 4 * i_phi)
    assert {int(cellindex) // 8 for cellindex in cellindices.ravel() if cellindex >= 0} == set(range(nphi))


def test_spherical_image_has_a_point_in_a_narrow_shell_or_gives_a_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """The image has a point in a narrow shell, or it gives a warning for a cut shell with no point.

    The points of the image are a lattice, which has a discrete set of radii. A shell from 0.4546 to 0.4548 vmax lay
    between two radii of the 200 points, and the image did not show it with no warning. A shell from 0.30238 to
    0.30239 vmax lies between two radii of the maximum of 1000 points.
    """
    for shelledges_kmps, haspoint in (([4546.0, 4548.0], True), ([3023.8, 3023.9], False)):
        modelmeta = {
            "coordinates": "spherical2d",
            "vmax_cmps": 1e9,
            "vel_r_edges_kmps": [0.0, *shelledges_kmps, 10000.0],
            "theta_edges_radians": [0.0, math.pi],
        }
        cellindices = at.inputmodel.get_spherical_image_cell_indices(modelmeta, sliceaxis=None)
        assert (1 in cellindices) == haspoint
        captured = capsys.readouterr()
        haswarning = "1 radial shells that the plane cuts have no point" in captured.out + captured.err
        assert haswarning == (not haspoint)


def test_spherical_cell_search_of_an_image_agrees_with_the_cell_search_of_the_packets() -> None:
    """The cell of a velocity vector is the same for the points of an image and for the packets.

    The packets take the rules of sn3d. The points are random, and some are at the centre, on the z axis, in the
    plane z = 0, in the half-plane of phi = 0, and outside the grid.
    """
    _, modelmeta = get_random_spherical_model("spherical3d")
    rng = np.random.default_rng(seed=2)
    velocities = rng.uniform(-3.2e9, 3.2e9, size=(4000, 3))
    # the two inner shells are small, thus a part of the points is near the centre
    velocities[2000:] *= rng.uniform(0.0, 0.3, size=(2000, 1))
    velocities[:50, 2] = 0.0
    velocities[50:100, :2] = 0.0
    velocities[100:150, 1] = 0.0
    velocities[150] = 0.0
    cellindices = at.inputmodel.get_spherical_cell_index_of_velocities(
        velocities[:, 0], velocities[:, 1], velocities[:, 2], modelmeta
    )
    dfpackets = pl.DataFrame({"vel_x": velocities[:, 0], "vel_y": velocities[:, 1], "vel_z": velocities[:, 2]})
    vel_r = (pl.col("vel_x") ** 2 + pl.col("vel_y") ** 2 + pl.col("vel_z") ** 2).sqrt()
    packetcells = dfpackets.select(
        at.packets
        .get_spherical_modelgridindex_expr(vel_r, pl.col("vel_x"), pl.col("vel_y"), pl.col("vel_z"), modelmeta)
        .fill_null(-1)
        .alias("cellindex")
    )["cellindex"].to_numpy()
    assert np.array_equal(cellindices, packetcells)
    # a part of the points is outside the grid, and the points inside it find each cell
    assert (cellindices == -1).sum() > 100
    assert set(cellindices.tolist()) == {-1, *range(4 * 3 * 5)}


def test_plotinitialcomposition_of_a_spherical_model_gives_each_point_the_value_of_its_cell(tmp_path: Path) -> None:
    """The image of a spherical2d model is a grid of points, and each point has the density of the cell that holds it.

    The model has two shells with the outer velocities 1000 and 3000 km/s and two polar cells with the edge at 1.5
    rad. The index of a cell is i_r + 2 * i_theta, and its density is (index + 1) * 1e-12 g/cm3.
    """
    (tmp_path / "model.txt").write_text("\n".join(SPHERICAL2D_MODEL_LINES) + "\n", encoding="utf-8")
    with mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True) as mockimshow:
        at.inputmodel.plotinitialcomposition.main(argsraw=["rho", str(tmp_path), "-o", str(tmp_path / "plot.pdf")])

    valuegrid = mockimshow.call_args.args[1]
    assert valuegrid.shape == (400, 200)
    xmin, xmax, ymin, ymax = mockimshow.call_args.kwargs["extent"]
    vmax_on_c = 3000.0 * at.constants.km_to_cm / at.constants.C_cm_per_s
    assert (xmin, xmax, ymin, ymax) == pytest.approx((0.0, vmax_on_c, -vmax_on_c, vmax_on_c), rel=1e-6)

    def get_density(vel_rcyl_kmps: float, vel_z_kmps: float) -> float:
        return float(valuegrid[int((vel_z_kmps + 3000.0) / (6000.0 / 400)), int(vel_rcyl_kmps / (3000.0 / 200))])

    # +z is the first polar cell, and -z is the second one
    assert get_density(100.0, 500.0) == pytest.approx(1e-12, rel=1e-6)
    assert get_density(100.0, 2000.0) == pytest.approx(2e-12, rel=1e-6)
    assert get_density(100.0, -500.0) == pytest.approx(3e-12, rel=1e-6)
    assert get_density(2000.0, -500.0) == pytest.approx(4e-12, rel=1e-6)
    # a point outside the sphere has no cell
    assert np.ma.is_masked(valuegrid[399, 199])


@pytest.mark.parametrize(
    ("oldline", "newlines", "errortext"),
    [
        ("3 4e-12 1", ("3 4e-12 1", "4 5e-12 1"), "more than 4 cells"),
        ("3 4e-12 1", ("2 4e-12 1",), "must increase from each cell line to the next"),
        ("3 4e-12 1", ("4 4e-12 1",), "each inputcellid must be from 0 to 3"),
        ("3 4e-12 1", ("3 4e-12 1 0.5",), "a cell line has more values than the key columns"),
        ("3 4e-12 1", ("3 4e-12",), "a cell line has fewer values than the key columns"),
        # the writer does not write such a density, thus the reader also stops for it
        ("3 4e-12 1", ("3 nan 1",), "1 cells have a density that is not finite or is below 0"),
        ("3 4e-12 1", ("3 inf 1",), "1 cells have a density that is not finite or is below 0"),
        ("3 4e-12 1", ("3 -4e-12 1",), "1 cells have a density that is not finite or is below 0"),
        ("3 4e-12 1", ("3 4e-12 nan",), "the columns X_Fegroup hold a value that is not finite"),
        ("3 4e-12 1", ("3 4e-12 inf",), "the columns X_Fegroup hold a value that is not finite"),
        # a cell with no line is empty, thus only the count finds a file that lost its last line
        ("3 4e-12 1", (), "the key cell_line_count gives 4 cell lines, but the file has 3"),
        ("cell_line_count 4", ("cell_line_count 5",), "cell_line_count must have one integer from 1 to 4"),
        # the polar axis of a mirrored model ends at pi / 2
        (
            "cell_line_count 4",
            ("cell_line_count 4", "mirror_plane z=0"),
            r"the last value of theta_edges_radians must be 1\.5707963267948966 \(pi / 2\)",
        ),
        (
            "cell_line_count 4",
            ("cell_line_count 4", "mirror_plane z=1"),
            "the key mirror_plane must have the value z=0",
        ),
        # the key cells_per_axis is optional, and a different count stops the reader
        (
            "cell_line_count 4",
            ("cell_line_count 4", "cells_per_axis 2 3"),
            "cells_per_axis gives the cell counts '2 3', but the cell edges give '2 2'",
        ),
        (
            "columns inputcellid rho X_Fegroup",
            ("phi_edges_radians 0 6.283185307179586", "columns inputcellid rho X_Fegroup"),
            "takes no phi_edges_radians",
        ),
        ("columns inputcellid rho X_Fegroup", (), "must end with the key columns"),
        ("t_model_init_days 1.0  # an inline comment", (), "the header has no key t_model_init_days"),
        ("t_model_init_days 1.0  # an inline comment", ("t_model_init_days inf",), "one finite value above 0"),
        ("vel_r_edges_kmps 0 1000 3000", ("vel_r_edges_kmps 0 3000 1000",), "must increase"),
        (
            f"theta_edges_radians 0 1.5 {math.pi!r}",
            (f"theta_edges_radians 0 5e-5 1.5 {math.pi!r}",),
            "the polar edge 5e-05 rad is less than 0.0001 rad from the z axis",
        ),
        (
            f"theta_edges_radians 0 1.5 {math.pi!r}",
            (f"theta_edges_radians 0 1.5 {1.5 + 1e-10!r} {math.pi!r}",),
            "each cell of theta_edges_radians must be wider than 1e-08 rad",
        ),
        ("format_version 2", (), "needs the line 'format_version 2' as the first line"),
        ("format_version 2", ("# a comment", "format_version 2"), "must be the first line of the file"),
        ("format_version 2", ("format_version 3",), "the format version '3' is not known"),
        (
            f"theta_edges_radians 0 1.5 {math.pi!r}",
            ("theta_edges_radians 0 1.5 3.0",),
            "the last value of theta_edges_radians",
        ),
        # both edges become exactly pi / 2, as in sn3d, and the cell between them then has no volume
        (
            f"theta_edges_radians 0 1.5 {math.pi!r}",
            (f"theta_edges_radians 0 {math.pi / 2 - 4e-9!r} {math.pi / 2 + 4e-9!r} {math.pi!r}",),
            "theta_edges_radians must increase",
        ),
    ],
)
def test_spherical_model_reader_stops_for_a_wrong_file(
    tmp_path: Path, oldline: str, newlines: tuple[str, ...], errortext: str
) -> None:
    """The reader checks the cell lines against the cell edges and against the key cell_line_count of the header."""
    lines = [newline for line in SPHERICAL2D_MODEL_LINES for newline in (newlines if line == oldline else (line,))]
    (tmp_path / "model.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=errortext):
        at.inputmodel.scan_modeldata(tmp_path)


def test_spherical_model_reader_stops_for_a_cut_last_line(tmp_path: Path) -> None:
    """A cut inside the last value gives a valid number, e.g. 1.0 of 1.0e-12, thus only the missing line end shows it.

    The key cell_line_count finds a lost line, and the reader accepted a file that ended inside its last line.
    """
    (tmp_path / "model.txt").write_text("\n".join(SPHERICAL2D_MODEL_LINES), encoding="utf-8")

    with pytest.raises(ValueError, match="does not end with a line end"):
        at.inputmodel.scan_modeldata(tmp_path)


@pytest.mark.parametrize("coordinates", ["spherical1d", "cylindrical2d", "cartesian3d"])
def test_format_version_2_gives_the_cells_of_version_1(coordinates: str, tmp_path: Path) -> None:
    """A model in format version 2 gives the cells and the metadata of the same model in format version 1.

    A version 2 file has the cell edges in its header and no position column, thus the reader makes the position
    columns of version 1 from the edges. sn3d makes the grid of a 2D cylindrical or a 3D Cartesian model from vmax,
    thus that value must be exactly equal in the two versions.
    """
    lzmodel, modelmeta = at.inputmodel.scan_modeldata(
        at.get_path("testdata") / ("test-classicmode_1d" if coordinates == "spherical1d" else "test-classicmode_3d")
    )
    dfmodel = lzmodel.collect()
    if coordinates == "cylindrical2d":
        dfmodel, _, _, modelmeta = at.inputmodel.dimension_reduce_model(
            dfmodel, outputdimensions=2, modelmeta=modelmeta
        )

    cells: dict[int, pl.DataFrame] = {}
    metas: dict[int, dict[str, t.Any]] = {}
    for formatversion in (1, 2):
        folder = tmp_path / f"version{formatversion}"
        folder.mkdir()
        # the cell ids of the test model start at 1, and version 2 has cell ids from 0
        dfcells = at.inputmodel.with_cell_ids_from_zero(dfmodel)[0] if formatversion == 2 else dfmodel
        at.inputmodel.save_modeldata(dfcells, outpath=folder, modelmeta=modelmeta | {"formatversion": formatversion})
        lzcells, metas[formatversion] = at.inputmodel.scan_modeldata(folder)
        cells[formatversion] = lzcells.collect()

    with at.zopen(tmp_path / "version2" / "model.txt.zst") as fmodel:
        assert fmodel.readline() == "format_version 2\n"
    assert metas[1]["formatversion"] == 1
    assert metas[2]["formatversion"] == 2
    assert metas[2]["coordinates"] == coordinates

    positioncols = [col for col in cells[1].columns if col.startswith(("pos_", "vel_")) or col == "logrho"]
    assert positioncols
    othercols = [col for col in cells[1].columns if col not in {*positioncols, "inputcellid"}]
    pltest.assert_frame_equal(cells[1].select(othercols), cells[2].select(othercols))
    assert cells[1]["inputcellid"].to_list() == [inputcellid + 1 for inputcellid in cells[2]["inputcellid"]]
    if coordinates == "spherical1d":
        # version 1 gives vel_r_max_kmps with two decimal places and logrho with eight
        assert np.allclose(cells[1]["vel_r_max_kmps"], cells[2]["vel_r_max_kmps"], rtol=0.0, atol=0.006)
        assert np.allclose(cells[1]["logrho"], cells[2]["logrho"], rtol=0.0, atol=1e-6)
        assert metas[2]["vmax_cmps"] == pytest.approx(metas[1]["vmax_cmps"], rel=1e-6)
    else:
        xmax = metas[1]["vmax_cmps"] * metas[1]["t_model_init_days"] * 86400
        for col in positioncols:
            assert np.allclose(cells[1][col], cells[2][col], rtol=0.0, atol=1e-6 * xmax), col
        assert metas[2]["vmax_cmps"] == metas[1]["vmax_cmps"]

    edgekeys = at.misc.MODEL_EDGE_KEYS[coordinates]
    assert {key for key in metas[2] if key not in metas[1]} == set(edgekeys)
    for key, value in metas[1].items():
        if key not in {"formatversion", "vmax_cmps"}:
            assert metas[2][key] == value, key

    # version 2 keeps each value that it holds, thus a second write gives the same cells
    at.inputmodel.save_modeldata(cells[2], outpath=tmp_path, modelmeta=metas[2])
    pltest.assert_frame_equal(at.inputmodel.scan_modeldata(tmp_path)[0].collect(), cells[2])


def test_otherstable_columns_give_the_element_mass_fractions() -> None:
    """A version 2 model.txt has no abundances.txt, and sn3d sums the nuclide columns and the otherstable column.

    Thus the otherstable column of an element holds the mass fraction that its nuclide columns do not hold. An
    element gets no column if its nuclide columns hold its mass fraction to the Float32 rounding.
    """
    dfmodel = pl.DataFrame({
        "inputcellid": [0, 1],
        "rho": [1e-12, 2e-12],
        "X_Fegroup": [1.0, 1.0],
        "X_Ni56": [0.5, 0.25],
        "X_Ni57": [0.0, 0.25],
        "X_Co56": [0.25, 0.125],
    })
    dfelabundances = pl.DataFrame({
        "inputcellid": [0, 1],
        "X_Fe": [0.125, 0.0],
        "X_Co": [0.25, 0.125 * (1 + 1e-7)],
        "X_Ni": [0.625, 0.5],
    })
    dfwithotherstable = at.inputmodel.core.with_otherstable_columns(dfmodel, dfelabundances)
    assert [col for col in dfwithotherstable.columns if col.endswith("_otherstable")] == [
        "X_Fe_otherstable",
        "X_Ni_otherstable",
    ]
    assert dfwithotherstable["X_Fe_otherstable"].to_list() == [0.125, 0.0]
    assert dfwithotherstable["X_Ni_otherstable"].to_list() == [0.125, 0.0]
    dfelements = at.inputmodel.core.get_elemabundances_of_model_columns(dfwithotherstable.lazy()).collect()
    for elcol in ("X_Fe", "X_Co", "X_Ni"):
        assert dfelements[elcol].to_list() == pytest.approx(dfelabundances[elcol].to_list(), rel=1e-6)

    # sn3d reads a small negative nuclide mass fraction as 0, thus the otherstable value must not take it
    dfnegative = at.inputmodel.core.with_otherstable_columns(
        dfmodel.with_columns(X_Ni57=pl.Series([-5e-7, 0.25])), dfelabundances
    )
    assert dfnegative["X_Ni_otherstable"].to_list() == [0.125, 0.0]
    # the sum of the element also takes such a value as 0, as sn3d does
    dfnegativeelements = at.inputmodel.core.get_elemabundances_of_model_columns(dfnegative.lazy()).collect()
    assert dfnegativeelements["X_Ni"].to_list() == [0.625, 0.5]

    # a null or a NaN mass fraction is not known, and the otherstable column would give it as 0
    for unknownvalue in (None, float("nan")):
        with pytest.raises(ValueError, match="The element abundances have no finite value for 1 cells with density"):
            at.inputmodel.core.with_otherstable_columns(
                dfmodel, dfelabundances.with_columns(X_Fe=pl.Series([0.125, unknownvalue], dtype=pl.Float64))
            )

    # sn3d stops if the nuclide columns of an element hold more than the element
    with pytest.raises(ValueError, match=r"The nuclide columns of Ni hold up to 0\.125 more"):
        at.inputmodel.core.with_otherstable_columns(dfmodel, dfelabundances.with_columns(X_Ni=pl.lit(0.375)))

    # A cell with density and no row of element mass fractions got no otherstable value, thus it lost the elements
    # that its nuclide columns do not hold. An empty cell needs no row
    firstcellid = dfmodel["inputcellid"][0]
    with pytest.raises(ValueError, match="no row for 1 cells with density"):
        at.inputmodel.core.with_otherstable_columns(
            dfmodel, dfelabundances.filter(pl.col("inputcellid") != firstcellid)
        )
    isfirstcell = pl.col("inputcellid") == firstcellid
    at.inputmodel.core.with_otherstable_columns(
        dfmodel.with_columns(rho=pl.when(isfirstcell).then(0.0).otherwise(pl.col("rho"))),
        dfelabundances.filter(~isfirstcell),
    )

    # X_n gives the free neutrons, but the reader of an otherstable column takes the symbol n as nitrogen
    with_otherstable_columns = at.inputmodel.core.with_otherstable_columns
    dfneutrons = with_otherstable_columns(
        pl.DataFrame({"inputcellid": [0], "rho": [1e-12], "X_n1": [0.2], "X_He4": [0.8]}),
        pl.DataFrame({"inputcellid": [0], "X_n": [0.2], "X_He": [0.8]}),
    )
    assert not [col for col in dfneutrons.columns if col.endswith("_otherstable")]

    # The nuclide columns of nickel hold more than its element mass fraction, within the tolerance of sn3d. Thus the
    # otherstable value of nickel is 0, and the elements sum to 1.005, which sn3d stops for
    with pytest.raises(ValueError, match=r"sum to more than 1\.001, up to 1\.005"):
        with_otherstable_columns(
            pl.DataFrame({"inputcellid": [0], "rho": [1e-12], "X_Ni56": [0.505]}),
            pl.DataFrame({"inputcellid": [0], "X_Fe": [0.5], "X_Ni": [0.5]}),
        )

    # a 1D model of version 1 gives logrho, and a cell with logrho -99 is empty and needs no row
    df1d = with_otherstable_columns(
        pl.DataFrame({"inputcellid": [1, 2], "logrho": [-10.0, -99.0], "X_Ni56": [0.25, 0.0]}),
        pl.DataFrame({"inputcellid": [1], "X_Fe": [0.5], "X_Ni": [0.5]}),
    )
    assert df1d["X_Ni_otherstable"].to_list() == pytest.approx([0.25, 0.0])


def test_conversion_reads_the_folder_of_a_file_path_and_changes_no_file_before_its_checks_pass(tmp_path: Path) -> None:
    """A conversion takes the files of the model folder also for a path of model.txt, and it stops before a write.

    A path of model.txt gave the folder model.txt to the readers of abundances.txt and Ye.txt, thus the converted
    model lost its elements and its electron fractions. A conversion that stopped at the element check had already
    deleted the abundances.txt of the output folder. A Ye.txt of an earlier model in the output folder of a version
    1 model would give its values to the new model, because sn3d reads Ye.txt after model.txt.
    """
    sourcefolder = tmp_path / "source"
    sourcefolder.mkdir()
    lines1d = [
        "2",
        "1.0",
        "#inputcellid vel_r_max_kmps logrho X_Fegroup X_Ni56 X_Co56 X_Fe52 X_Cr48 Ye",
        "1 1000.00 -10.00000000 0.5 0.0 0.0 0.0 0.0 0.35",
        "2 2000.00 -11.00000000 0.5 0.0 0.0 0.0 0.0 0.35",
    ]
    (sourcefolder / "model.txt").write_text("\n".join(lines1d) + "\n", encoding="utf-8")
    (sourcefolder / "abundances.txt").write_text("1 0.8 0.2\n2 0.8 0.2\n", encoding="utf-8")
    (sourcefolder / "Ye.txt").write_text("2\n1 0.49\n2 0.49\n", encoding="utf-8")

    # A path of model.txt, a 1D model with logrho, and a Ye.txt for each cell. The path of model.txt also names its
    # compressed copy, as for the reader of model.txt
    outputfolder = tmp_path / "version2"
    for modelfilename in ("model.txt", "model.txt.zst"):
        if modelfilename == "model.txt.zst":
            with (sourcefolder / modelfilename).open("wb") as fmodel:
                at.misc.write_zstd_lines(fmodel, lines1d)
            (sourcefolder / "model.txt").unlink()
        at.inputmodel.makeartismodel.main(
            argsraw=[], modelpath=[sourcefolder / "model.txt"], format_version=2, outputfile=outputfolder, force=True
        )
        dfversion2 = at.inputmodel.scan_modeldata(outputfolder, get_elemabundances=True)[0].collect()
        assert dfversion2["X_H"].to_list() == pytest.approx([0.8, 0.8])
        assert dfversion2["Ye"].to_list() == pytest.approx([0.49, 0.49])

    def get_folder_text(folder: Path) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in sorted(folder.iterdir()) if path.is_file()}

    # a Ye.txt of an earlier model stops a conversion to version 1 into its folder, and the folder does not change
    (sourcefolder / "Ye.txt").unlink()
    version1folder = tmp_path / "version1"
    version1folder.mkdir()
    (version1folder / "Ye.txt").write_text("1\n1 0.9\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(
            argsraw=[], modelpath=[sourcefolder], format_version=1, outputfile=version1folder, force=True
        )
    assert get_folder_text(version1folder) == {"Ye.txt": b"1\n1 0.9\n"}

    # The nuclide column of nickel holds more than the element, within the tolerance of sn3d. Thus the otherstable
    # value of nickel is 0, and the elements of version 2 sum to 0.11 + 0.899, which is more than 1.001
    (sourcefolder / "model.txt.zst").unlink()
    (sourcefolder / "model.txt").write_text(
        "\n".join([*lines1d[:3], *(line.replace(" 0.5 0.0 ", " 0.5 0.899 ", 1) for line in lines1d[3:])]) + "\n",
        encoding="utf-8",
    )
    elementvalues = " ".join(["0.11", *["0.0"] * 26, "0.89"])
    (sourcefolder / "abundances.txt").write_text(f"1 {elementvalues}\n2 {elementvalues}\n", encoding="utf-8")
    folderbefore = get_folder_text(outputfolder)
    with pytest.raises(ValueError, match="sum to more than"):
        at.inputmodel.makeartismodel.main(
            argsraw=[], modelpath=[sourcefolder], format_version=2, outputfile=outputfolder, force=True
        )
    assert get_folder_text(outputfolder) == folderbefore


def test_conversion_reads_the_files_of_the_model_file_and_keeps_the_source_after_a_failed_write(tmp_path: Path) -> None:
    """A conversion takes the files beside the model.txt that the reader opens, and a failed write changes no file.

    The elements of a version 2 model came from a different copy of model.txt in the same folder, and a model in a
    subfolder lost the Ye.txt beside it. A conversion in the model folder that stopped at the write of abundances.txt
    or of gridcontributions.txt left a new model.txt without the other files of the source model.
    """
    sourcefolder = tmp_path / "source"
    modelfolder = sourcefolder / "run"
    modelfolder.mkdir(parents=True)
    lines1d = [
        "2",
        "1.0",
        "#inputcellid vel_r_max_kmps logrho X_Fegroup X_Ni56 X_Co56 X_Fe52 X_Cr48 Ye",
        "1 1000.00 -10.00000000 0.5 0.0 0.0 0.0 0.0 0.35",
        "2 2000.00 -11.00000000 0.5 0.0 0.0 0.0 0.0 0.35",
    ]
    (modelfolder / "model.txt").write_text("\n".join(lines1d) + "\n", encoding="utf-8")
    (modelfolder / "abundances.txt").write_text("1 0.8 0.2\n2 0.8 0.2\n", encoding="utf-8")
    (modelfolder / "Ye.txt").write_text("2\n1 0.49\n2 0.49\n", encoding="utf-8")
    (modelfolder / "gridcontributions.txt").write_text(
        "particleid cellindex frac_of_cellmass\n7 1 1.0\n8 2 1.0\n", encoding="utf-8"
    )

    # the reader finds the model in a subfolder, and the files beside that model.txt belong to it
    version2folder = tmp_path / "version2"
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[sourcefolder], format_version=2, outputfile=version2folder)
    dfversion2 = at.inputmodel.scan_modeldata(version2folder, get_elemabundances=True)[0].collect()
    assert dfversion2["X_H"].to_list() == pytest.approx([0.8, 0.8])
    assert dfversion2["Ye"].to_list() == pytest.approx([0.49, 0.49])

    # Two copies of model.txt can hold different values, and a reader of one copy can take the parquet cache of the
    # other copy. Thus a conversion stops for a folder with two copies, and sn3d reads the plain copy
    with at.zopen(version2folder / "model.txt.zst") as fmodel:
        version2text = fmodel.read()
    (version2folder / "model.txt").write_text(version2text.replace(" 8.0", " 2.0"), encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(
            argsraw=[], modelpath=[version2folder / "model.txt.zst"], format_version=1, outputfile=tmp_path / "selected"
        )

    # a selected model file of version 1 takes the abundances.txt beside it, also beside a model.txt of version 2
    (version2folder / "model.txt.zst").rename(version2folder / "model_v2.txt.zst")
    altfolder = tmp_path / "alt"
    altfolder.mkdir()
    (altfolder / "model_alt.txt").write_text("\n".join(lines1d) + "\n", encoding="utf-8")
    (altfolder / "abundances.txt").write_text("1 0.8 0.2\n2 0.8 0.2\n", encoding="utf-8")
    with at.zopen(version2folder / "model_v2.txt.zst") as fmodel:
        (altfolder / "model.txt").write_text(fmodel.read().replace(" 8.0", " 2.0"), encoding="utf-8")
    (version2folder / "model_v2.txt.zst").rename(version2folder / "model.txt.zst")
    altversion2folder = tmp_path / "altversion2"
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=[altfolder / "model_alt.txt"], format_version=2, outputfile=altversion2folder
    )
    dfalt = at.inputmodel.scan_modeldata(altversion2folder, get_elemabundances=True)[0].collect()
    assert dfalt["X_H"].to_list() == pytest.approx([0.8, 0.8])

    # sn3d reads the abundances.txt beside model.txt, and a file in a subfolder is of a different model. Thus a model
    # of version 1 with no abundances.txt beside its model.txt has no known element mass fractions
    nearfolder = tmp_path / "near"
    (nearfolder / "sub").mkdir(parents=True)
    (nearfolder / "model.txt").write_text("\n".join(lines1d) + "\n", encoding="utf-8")
    (nearfolder / "sub" / "abundances.txt").write_text("1 0.8 0.2\n2 0.8 0.2\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(
            argsraw=[], modelpath=[nearfolder], format_version=2, outputfile=tmp_path / "nearversion2"
        )

    # the particle contributions of an output folder in a subfolder of the source folder are of a different model
    (nearfolder / "sub" / "abundances.txt").rename(nearfolder / "abundances.txt")
    childfolder = nearfolder / "sub"
    (childfolder / "gridcontributions.txt").write_text(
        "particleid cellindex frac_of_cellmass\n42 1 1.0\n", encoding="utf-8"
    )
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=[nearfolder], format_version=2, outputfile=childfolder, force=True
    )
    assert not list(childfolder.glob("gridcontributions.txt*"))

    def get_folder_bytes(folder: Path) -> dict[str, bytes]:
        return {path.name: path.read_bytes() for path in sorted(folder.iterdir()) if path.is_file()}

    # the write of abundances.txt of a conversion to version 1 in the model folder stops
    (version2folder / "model.txt").unlink()
    version2files = get_folder_bytes(version2folder)
    msg = "Write failed"
    with (
        mock.patch("artistools.inputmodel.core.write_artis_csv", side_effect=OSError(msg)),
        pytest.raises(OSError, match=msg),
    ):
        at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[version2folder], format_version=1, force=True)
    assert get_folder_bytes(version2folder) == version2files
    assert not [path for path in version2folder.iterdir() if path.is_dir()]

    # the write of gridcontributions.txt with the cell ids from 0 of a conversion to version 2 stops
    sourcefiles = get_folder_bytes(modelfolder)
    with (
        mock.patch("artistools.inputmodel.makeartismodel.save_gridparticlecontributions", side_effect=OSError(msg)),
        pytest.raises(OSError, match=msg),
    ):
        at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[modelfolder], format_version=2, force=True)
    assert get_folder_bytes(modelfolder) == sourcefiles


def test_ye_merge_and_version_2_writer_follow_the_rules_of_sn3d(tmp_path: Path) -> None:
    """The values of Ye.txt and a NaN mass fraction of a version 2 file follow the rules of sn3d.

    The merge of Ye.txt ignored the count of its first line, thus a file that lost its last lines gave the old value
    of the column to the cells of those lines. sn3d stops for such a file, and it reads no value after the count.
    The writer of version 2 wrote a NaN mass fraction as NaN, which sn3d stops for.
    """
    dfmodel = pl.DataFrame({"inputcellid": [1, 2], "rho": [1e-12, 1e-12], "Ye": [0.3, 0.3]})
    yefile = tmp_path / "Ye.txt"
    yefile.write_text("3\n1 0.4\n2 0.4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ends after 2 of its 3 cell lines"):
        at.inputmodel.core.with_ye_of_yefile(dfmodel, yefile)
    yefile.write_text("1\n1 0.4\n2 0.45\n", encoding="utf-8")
    assert at.inputmodel.core.with_ye_of_yefile(dfmodel, yefile)["Ye"].to_list() == pytest.approx([0.4, 0.3])
    # a conversion to version 2 gives these values to the column Ye
    for invalidye in ("nan", "inf", "1.5", "-0.1"):
        yefile.write_text(f"1\n1 {invalidye}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Each electron fraction must be a finite number from 0 to 1"):
            at.inputmodel.core.with_ye_of_yefile(dfmodel, yefile)

    at.inputmodel.save_modeldata(
        pl.DataFrame({"inputcellid": [0, 1], "rho": [1e-12, 1e-12], "X_Ni56": [0.5, float("nan")]}),
        outpath=tmp_path,
        modelmeta={
            "coordinates": "spherical2d",
            "t_model_init_days": 1.0,
            "vel_r_edges_kmps": [0.0, 1000.0, 2000.0],
            "theta_edges_radians": [0.0, math.pi],
        },
        deletecellidfiles=True,
    )
    dfwritten = at.inputmodel.scan_modeldata(tmp_path)[0].collect()
    assert dfwritten["X_Ni56"].to_list() == [0.5, 0.0]

    # the column Ye of a version 2 file holds an electron fraction from 0 to 1, as Ye.txt does
    yemeta = {
        "coordinates": "spherical2d",
        "t_model_init_days": 1.0,
        "vel_r_edges_kmps": [0.0, 1000.0, 2000.0],
        "theta_edges_radians": [0.0, math.pi],
    }
    yefolder = tmp_path / "ye"
    yetext = "\n".join([
        "format_version 2",
        "coordinates spherical2d",
        "t_model_init_days 1.0",
        "vel_r_edges_kmps 0 1000 2000",
        f"theta_edges_radians 0 {math.pi!r}",
        "columns inputcellid rho Ye",
        "0 1e-12 0.4",
        "1 1e-12 {}",
    ])
    yefolder.mkdir()
    for invalidye in (float("nan"), float("inf"), 1.5, -0.1):
        with pytest.raises(ValueError, match="Each value of the column Ye must be a finite number from 0 to 1"):
            at.inputmodel.save_modeldata(
                pl.DataFrame({"inputcellid": [0, 1], "rho": [1e-12, 1e-12], "Ye": [0.4, invalidye]}),
                outpath=yefolder,
                modelmeta=dict(yemeta),
            )
        (yefolder / "model.txt").write_text(yetext.format(invalidye) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Each value of the column Ye must be a finite number from 0 to 1"):
            at.inputmodel.scan_modeldata(yefolder)
    # a null Ye is not known, and the writer gave it as 0.0
    with pytest.raises(ValueError, match="Each value of the column Ye must be a finite number from 0 to 1"):
        at.inputmodel.save_modeldata(
            pl.DataFrame({
                "inputcellid": [0, 1],
                "rho": [1e-12, 1e-12],
                "Ye": pl.Series([0.4, None], dtype=pl.Float64),
            }),
            outpath=yefolder,
            modelmeta=dict(yemeta),
        )
    # .all() takes no account of a null, and the writer gave a null cell id as 0.0
    with pytest.raises(ValueError, match="the cell ids are 0 to 1"):
        at.inputmodel.core.get_version2_headerlines(pl.Series([None, 1], dtype=pl.Int64), dict(yemeta))


def test_ye_file_to_merge_follows_the_search_of_sn3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversion to version 2 takes the Ye.txt that sn3d reads, and it stops if a write can lose a copy.

    A model.txt in data/ gave no Ye.txt of the model folder, thus the converted model had no column Ye. A write in place
    deleted a Ye.txt.zst behind Ye.txt. A data/Ye.txt or a data/abundances.txt of the output stayed, and sn3d stops for
    such a file.
    """
    from artistools.inputmodel.core import find_ye_file_to_merge

    modelfolder = tmp_path / "model"
    (modelfolder / "data").mkdir(parents=True)
    (modelfolder / "input.txt").write_text("", encoding="utf-8")
    (modelfolder / "Ye.txt").write_text("1\n1 0.49\n", encoding="utf-8")
    assert find_ye_file_to_merge(modelfolder / "data", tmp_path / "out", 2) == modelfolder / "Ye.txt"
    # the write into data/ does not delete the Ye.txt of the model folder, also for the relative path "."
    monkeypatch.chdir(modelfolder / "data")
    assert find_ye_file_to_merge(Path(), tmp_path / "out", 2) == modelfolder / "Ye.txt"
    with pytest.raises(SystemExit):
        find_ye_file_to_merge(Path(), Path(), 2)

    assert find_ye_file_to_merge(modelfolder, modelfolder, 2) == modelfolder / "Ye.txt"
    (modelfolder / "Ye.txt.zst").write_bytes(b"")
    # a write beside the Ye.txt.zst deletes it, and sn3d reads Ye.txt alone
    for sourcefolder in (modelfolder, modelfolder / "data"):
        with pytest.raises(SystemExit):
            find_ye_file_to_merge(sourcefolder, modelfolder, 2)

    outputfolder = tmp_path / "out"
    (outputfolder / "data").mkdir(parents=True)
    (outputfolder / "input.txt").write_text("", encoding="utf-8")
    (outputfolder / "data" / "Ye.txt").write_text("1\n1 0.3\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        find_ye_file_to_merge(modelfolder, outputfolder, 2)
    assert find_ye_file_to_merge(modelfolder, outputfolder, 1) == modelfolder / "Ye.txt"

    # sn3d also stops for the data/abundances.txt that a write from data/ into the model folder leaves
    (modelfolder / "Ye.txt.zst").unlink()
    (modelfolder / "data" / "abundances.txt.zst").write_bytes(b"")
    with pytest.raises(SystemExit):
        find_ye_file_to_merge(modelfolder / "data", modelfolder, 2)

    # a data/ that links to a shared folder is part of the model folder, as for sn3d
    linkedmodel, sharedfolder = tmp_path / "linkedmodel", tmp_path / "shared"
    sharedfolder.mkdir()
    linkedmodel.mkdir()
    (linkedmodel / "data").symlink_to(sharedfolder, target_is_directory=True)
    (linkedmodel / "input.txt").write_text("", encoding="utf-8")
    (linkedmodel / "Ye.txt").write_text("1\n1 0.49\n", encoding="utf-8")
    assert find_ye_file_to_merge(linkedmodel / "data", tmp_path / "linkedout", 2) == linkedmodel / "Ye.txt"


def test_abundances_and_old_ye_files_follow_the_search_of_sn3d(tmp_path: Path) -> None:
    """sn3d searches the model folder, data/, and artis/data/ for each input file, also for a model.txt in data/.

    The search for abundances.txt took only the folder of model.txt, thus a conversion of data/model.txt stopped.
    The check for the Ye.txt of an earlier model took only the output folder, and sn3d then read a data/Ye.txt.
    """
    from artistools.inputmodel.core import find_abundance_file
    from artistools.inputmodel.core import require_no_ye_file_of_earlier_model

    modelfolder = tmp_path / "model"
    (modelfolder / "data").mkdir(parents=True)
    (modelfolder / "input.txt").write_text("", encoding="utf-8")
    (modelfolder / "abundances.txt").write_text("1 1.0\n", encoding="utf-8")
    assert find_abundance_file(modelfolder / "data") == modelfolder / "abundances.txt"

    (modelfolder / "data" / "Ye.txt").write_text("1\n1 0.3\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        require_no_ye_file_of_earlier_model(modelfolder, tmp_path / "source")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"t_model_init_days": 0.0}, "t_model_init_days must be one finite value above 0"),
        ({"t_model_init_days": float("nan")}, "t_model_init_days must be one finite value above 0"),
        ({"rho": [1e-12, float("nan")]}, "1 cells have a density that is not finite or is below 0"),
        ({"rho": [1e-12, -1e-12]}, "1 cells have a density that is not finite or is below 0"),
        ({"rho": [1e-12, float("inf")]}, "1 cells have a density that is not finite or is below 0"),
        ({"X_Ni56": [0.5, float("inf")]}, "The columns X_Ni56 hold an infinite value in a cell with density"),
    ],
)
def test_version_2_writer_stops_for_a_value_that_the_reader_rejects(
    change: dict[str, t.Any], match: str, tmp_path: Path
) -> None:
    """The writer of version 2 wrote a model time that its reader rejects, and it lost a NaN or negative density.

    A cell with such a density got no line, thus the reader gave it rho = 0. The writer now stops before it writes.
    """
    modelmeta = {
        "coordinates": "spherical2d",
        "t_model_init_days": 1.0,
        "vel_r_edges_kmps": [0.0, 1000.0, 2000.0],
        "theta_edges_radians": [0.0, math.pi],
    }
    dfmodel = pl.DataFrame({"inputcellid": [0, 1], "rho": [1e-12, 1e-12]})
    if "rho" in change or "X_Ni56" in change:
        dfmodel = dfmodel.with_columns(itertools.starmap(pl.Series, change.items()))
    else:
        modelmeta |= change
    with pytest.raises(ValueError, match=match):
        at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path, modelmeta=modelmeta)
    assert not list(tmp_path.glob("model.txt*"))


def test_version_2_file_with_a_different_name_keeps_the_files_of_model_txt(tmp_path: Path) -> None:
    """A version 2 file with a different name keeps abundances.txt, Ye.txt, and opacity.txt beside model.txt.

    sn3d reads these files beside model.txt. The writer of model_v2.txt deleted them, thus the model.txt of version 1
    in the same folder had no abundances.txt.
    """
    oldfiles = dict.fromkeys(("abundances.txt", "Ye.txt", "opacity.txt"), "1\n1 0.5\n")
    for name, text in oldfiles.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    dfmodel, modelmeta = get_random_spherical_model("spherical2d")
    at.inputmodel.save_modeldata(dfmodel, outpath=tmp_path / "model_v2.txt", modelmeta=modelmeta)

    assert (tmp_path / "model_v2.txt.zst").is_file()
    assert {name: (tmp_path / name).read_text(encoding="utf-8") for name in oldfiles} == oldfiles


def test_reduction_and_downscale_take_the_elements_of_the_selected_model_file(tmp_path: Path) -> None:
    """A reduction and a downscale of a selected model file take its elements, also beside a different model.txt.

    The reader of the element mass fractions opened the model.txt of the folder again, thus the output got the
    elements of a different model.
    """
    version2folder = tmp_path / "version2"
    at.inputmodel.makeartismodel.main(
        argsraw=[],
        modelpath=[at.get_path("testdata") / "test-classicmode_3d"],
        format_version=2,
        outputfile=version2folder,
    )
    lzselected, metaselected = at.inputmodel.scan_modeldata(version2folder, get_elemabundances=True)
    selectedmodel = at.inputmodel.add_derived_cols_to_modeldata(lzselected, metaselected).collect()
    hydrogenmass = float((selectedmodel["X_H"] * selectedmodel["mass_g"]).sum())
    assert hydrogenmass > 0.0

    # the folder holds the selected model as model_alt.txt.zst and a model.txt with half of its hydrogen
    altfolder = tmp_path / "alt"
    otherfolder = tmp_path / "other"
    altfolder.mkdir()
    (version2folder / "model.txt.zst").rename(altfolder / "model_alt.txt.zst")
    dfother = at.inputmodel.scan_modeldata(altfolder / "model_alt.txt.zst")[0].drop("modelgridindex").collect()
    at.inputmodel.save_modeldata(
        dfother.with_columns(pl.col("X_H_otherstable") * 0.5), outpath=otherfolder, modelmeta=dict(metaselected)
    )
    with at.zopen(otherfolder / "model.txt.zst") as fmodel:
        (altfolder / "model.txt").write_text(fmodel.read(), encoding="utf-8")

    def get_hydrogen_mass(modelfolder: Path) -> float:
        lzmodel, modelmeta = at.inputmodel.scan_modeldata(modelfolder, get_elemabundances=True)
        dfmodel = at.inputmodel.add_derived_cols_to_modeldata(lzmodel, modelmeta).collect()
        return float((dfmodel["X_H"] * dfmodel["mass_g"]).sum())

    reducedfolder = tmp_path / "reduced"
    at.inputmodel.makeartismodel.main(
        argsraw=[],
        modelpath=[altfolder / "model_alt.txt.zst"],
        dimensionreduce=1,
        format_version=2,
        outputfile=reducedfolder,
    )
    assert math.isclose(get_hydrogen_mass(next(reducedfolder.iterdir())), hydrogenmass, rel_tol=1e-5)

    smallfolder = tmp_path / "small"
    at.inputmodel.makeartismodel.main(
        argsraw=[],
        modelpath=[altfolder / "model_alt.txt.zst"],
        downscale3dgrid=True,
        outputgridsize=5,
        format_version=2,
        outputfile=smallfolder,
    )
    assert math.isclose(get_hydrogen_mass(smallfolder), hydrogenmass, rel_tol=1e-5)
    # the model reader also takes the elements of the selected file
    assert math.isclose(get_hydrogen_mass(altfolder / "model_alt.txt.zst"), hydrogenmass, rel_tol=1e-6)

    # a conversion in the folder writes model.txt, thus it replaced the model.txt of the folder with the selected file
    modeltextbefore = (altfolder / "model.txt").read_bytes()
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(
            argsraw=[], modelpath=[altfolder / "model_alt.txt.zst"], format_version=1, force=True
        )
    assert (altfolder / "model.txt").read_bytes() == modeltextbefore

    # sn3d reads the abundances.txt beside model.txt, and a file in a subfolder is of a different model
    version1folder = tmp_path / "version1"
    (version1folder / "sub").mkdir(parents=True)
    for sourcefile in (at.get_path("testdata") / "test-classicmode_3d").glob("model.txt*"):
        shutil.copy(sourcefile, version1folder)
    for sourcefile in (at.get_path("testdata") / "test-classicmode_3d").glob("abundances.txt*"):
        shutil.copy(sourcefile, version1folder / "sub")
    with pytest.raises(FileNotFoundError):
        at.inputmodel.scan_initelemabundances(version1folder).collect()


def test_reduction_and_downscale_write_the_format_version_of_the_prompt(tmp_path: Path) -> None:
    """A reduction and a downscale write the format version that the (y/n) prompt took, and version 2 takes Ye.txt.

    A downscale with --format-version wrote the version of the source model, thus it deleted a Ye.txt that the
    prompt did not name. A reduction to version 2 ignored the Ye.txt of the source model, which sn3d uses.
    """
    sourcefolder = tmp_path / "source"
    shutil.copytree(at.get_path("testdata") / "test-classicmode_3d", sourcefolder)
    dfsource = at.inputmodel.scan_modeldata(sourcefolder)[0].collect()
    (sourcefolder / "Ye.txt").write_text(
        f"{dfsource.height}\n" + "".join(f"{cellid} 0.49\n" for cellid in dfsource["inputcellid"]), encoding="utf-8"
    )
    # The output folder of the reduction is a subfolder of the source folder, and it holds the particle contributions
    # of an earlier model. The source model has none, thus the reduced model gets none
    reducedfolder = sourcefolder / f"{sourcefolder.name}_dimreduce_1d"
    reducedfolder.mkdir()
    (reducedfolder / "gridcontributions.txt").write_text(
        "particleid cellindex frac_of_cellmass\n42 1 1.0\n", encoding="utf-8"
    )
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=[sourcefolder], dimensionreduce=1, format_version=2, outputfile=sourcefolder, force=True
    )
    dfreduced = at.inputmodel.scan_modeldata(reducedfolder)[0].collect()
    assert dfreduced.filter(pl.col("rho") > 0)["Ye"].to_list() == pytest.approx([0.49] * 5, rel=1e-6)
    assert not list(reducedfolder.glob("gridcontributions.txt*"))

    version2folder = tmp_path / "version2"
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[sourcefolder], format_version=2, outputfile=version2folder)
    downscaleargs: dict[str, t.Any] = {"argsraw": [], "downscale3dgrid": True, "outputgridsize": 5, "force": True}
    for formatversion in (1, 2):
        sourceversion = 2 if formatversion == 1 else 1
        smallfolder = tmp_path / f"small{formatversion}"
        smallfolder.mkdir()
        at.inputmodel.makeartismodel.main(
            **downscaleargs,
            modelpath=[version2folder if sourceversion == 2 else sourcefolder],
            format_version=formatversion,
            outputfile=smallfolder,
        )
        dfsmall, metasmall = at.inputmodel.scan_modeldata(smallfolder)
        assert metasmall["formatversion"] == formatversion
        # the small model of version 2 gets the values of the Ye.txt of the source model
        if formatversion == 2:
            yevalues = dfsmall.filter(pl.col("rho") > 0).collect()["Ye"].to_numpy()
            assert yevalues.size > 0
            assert np.allclose(yevalues, 0.49, rtol=1e-6, atol=0.0)

    # a Ye.txt in the output folder of a small model of version 1 is of a different model, thus the command stops
    (tmp_path / "small1" / "Ye.txt").write_text("1\n1 0.9\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(
            **downscaleargs, modelpath=[version2folder], format_version=1, outputfile=tmp_path / "small1"
        )


def test_version_2_edges_of_a_cartesian_grid_must_have_equal_steps() -> None:
    """A Cartesian grid has equal cells to one vmax in sn3d and in artistools, thus other edges are an error."""
    edgekeys = at.misc.MODEL_EDGE_KEYS["cartesian3d"]
    uniformedges = {key: [-3000.0, -1000.0, 1000.0, 3000.0] for key in edgekeys}
    assert at.misc.validate_model_edges("cartesian3d", uniformedges, "test") == uniformedges

    for wrongedges in ([-3000.0, -1500.0, 1000.0, 3000.0], [-2000.0, -666.0, 666.0, 2000.0]):
        with pytest.raises(ValueError, match="must have equal steps"):
            at.misc.validate_model_edges("cartesian3d", uniformedges | {"vel_z_edges_kmps": wrongedges}, "test")
    with pytest.raises(ValueError, match="same count of values"):
        at.misc.validate_model_edges("cartesian3d", uniformedges | {"vel_y_edges_kmps": [-3000.0, 0.0, 3000.0]}, "test")


def test_makeartismodel_converts_the_format_version_in_the_model_folder(tmp_path: Path) -> None:
    """--format-version alone writes model.txt again in its folder, and a spherical model has no version 1.

    The cell ids of version 2 start at 0. Thus a conversion gives the same ids to each file that holds cell ids, and
    a reduced model of version 2 also has them.
    """
    sourcefolder = tmp_path / "source"
    shutil.copytree(at.get_path("testdata") / "test-classicmode_3d", sourcefolder)
    dfsource = at.inputmodel.scan_modeldata(sourcefolder, get_elemabundances=True)[0]
    # the cell ids of this model start at 1, and sn3d reads the ids of Ye.txt as those of model.txt. The model.txt
    # has no column Ye, and Ye.txt holds two of the cells with density
    idswithdensity = dfsource.filter(pl.col("rho") > 0).select("inputcellid").collect().to_series()
    firstid, lastid = idswithdensity.item(0), idswithdensity.item(-1)
    (sourcefolder / "Ye.txt").write_text(f"2\n{firstid}\t0.25\n{lastid}\t0.45\n", encoding="utf-8")
    (sourcefolder / "opacity.txt").write_text("2\n1 0.1\n1000 3.3\n", encoding="utf-8")
    modelfolder = tmp_path / "model"
    # the output folder holds files of an earlier model, which the conversion deletes. The source model has no
    # particle contributions
    modelfolder.mkdir()
    (modelfolder / "opacity.txt").write_text("1\n1 0.5\n", encoding="utf-8")
    (modelfolder / "gridcontributions.txt").write_text("particleid cellindex frac_of_cellmass\n1 1 1.0\n")
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=[sourcefolder], format_version=2, outputfile=modelfolder, force=True
    )
    dfversion2, metaversion2 = at.inputmodel.scan_modeldata(modelfolder, get_elemabundances=True)
    assert metaversion2["formatversion"] == 2
    assert dfversion2.select(firstid=pl.col("inputcellid").min(), lastid=pl.col("inputcellid").max()).collect().row(
        0
    ) == (0, 999)
    # a version 2 model has no Ye.txt. The column Ye holds its values at the new cell ids, and 0 for each other cell
    assert not (modelfolder / "Ye.txt").exists()
    assert dfversion2.filter(pl.col("Ye") > 0).select("inputcellid", "Ye").collect().rows() == [
        (firstid - 1, pytest.approx(0.25)),
        (lastid - 1, pytest.approx(0.45)),
    ]
    # a version 2 model has no opacity.txt, which sn3d does not read
    assert not (modelfolder / "opacity.txt").exists()
    assert not (modelfolder / "gridcontributions.txt").exists()
    # a version 2 model has no abundances.txt, and the columns of model.txt give the same element mass fractions
    assert not list(modelfolder.glob("abundances.txt*"))
    pltest.assert_series_equal(
        dfversion2.select("X_Fe").collect().to_series(), dfsource.select("X_Fe").collect().to_series()
    )

    reducedfolder = tmp_path / "reduced"
    at.inputmodel.makeartismodel.main(
        argsraw=[], modelpath=[sourcefolder], dimensionreduce=1, format_version=2, outputfile=reducedfolder
    )
    dfreduced, metareduced = at.inputmodel.scan_modeldata(next(reducedfolder.iterdir()), get_elemabundances=True)
    assert metareduced["formatversion"] == 2
    assert dfreduced.select(
        firstid=pl.col("inputcellid").min(), hasnoabundance=pl.col("X_Fe").is_null().any()
    ).collect().row(0) == (0, False)

    # the values of opacity.txt have no column in model.txt, thus a conversion in the model folder stops for the file
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[sourcefolder], format_version=2, force=True)
    assert (sourcefolder / "opacity.txt").exists()
    # the conversion also finds a compressed copy of the file
    with (sourcefolder / "opacity.txt.zst").open("wb") as fopacity:
        at.misc.write_zstd_lines(fopacity, (sourcefolder / "opacity.txt").read_text(encoding="utf-8").splitlines())
    (sourcefolder / "opacity.txt").unlink()
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[sourcefolder], format_version=2, force=True)
    (sourcefolder / "opacity.txt.zst").unlink()

    # a conversion in the model folder deletes Ye.txt, because sn3d stops for a version 2 model with that file.
    # sn3d also reads a compressed Ye.txt, thus the conversion takes the values of such a file
    with (sourcefolder / "Ye.txt.zst").open("wb") as fye:
        at.misc.write_zstd_lines(fye, (sourcefolder / "Ye.txt").read_text(encoding="utf-8").splitlines())
    (sourcefolder / "Ye.txt").unlink()
    # a conversion of a version 1 model into a different folder copies the compressed Ye.txt
    version1folder = tmp_path / "version1"
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[sourcefolder], format_version=1, outputfile=version1folder)
    assert (version1folder / "Ye.txt.zst").read_bytes() == (sourcefolder / "Ye.txt.zst").read_bytes()
    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[sourcefolder], format_version=2, force=True)
    assert not list(sourcefolder.glob("Ye.txt*"))
    assert at.inputmodel.scan_modeldata(sourcefolder)[0].filter(pl.col("Ye") > 0).select(
        "inputcellid"
    ).collect().to_series().to_list() == [firstid - 1, lastid - 1]
    (sourcefolder / "Ye.txt").write_text("0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[sourcefolder], format_version=1, force=True)

    at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[modelfolder], format_version=1, force=True)
    dfmodel, modelmeta = at.inputmodel.scan_modeldata(modelfolder)
    assert modelmeta["formatversion"] == 1
    # the version 1 model has abundances.txt again, with the element mass fractions of the source
    pltest.assert_series_equal(
        at.inputmodel.scan_initelemabundances(modelfolder).select("X_Fe").collect().to_series(),
        dfsource.select("X_Fe").collect().to_series(),
        rel_tol=1e-6,
    )
    assert "pos_x_min" in dfmodel.collect_schema().names()

    (modelfolder / "model.txt.zst").unlink()
    (modelfolder / "model.txt").write_text("\n".join(SPHERICAL2D_MODEL_LINES) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.makeartismodel.main(argsraw=[], modelpath=[modelfolder], format_version=1, force=True)


def test_dimension_reduce_of_a_spherical_model_sums_complete_cells() -> None:
    """A spherical3d model gives a spherical2d model, a 1D model, and one zone, each as a sum of complete cells.

    Thus each output mass is exact. The contributions of the particles change with the masses of the cells.
    """
    dfmodel, modelmeta = get_random_spherical_model("spherical3d")
    gridshape = (4, 3, 5)
    cellmass = (
        at.inputmodel
        .add_derived_cols_to_modeldata(dfmodel, modelmeta)
        .select("mass_g")
        .collect()
        .to_series()
        .to_numpy()
        .reshape(gridshape, order="F")
    )
    cellni56 = dfmodel["X_Ni56"].to_numpy().reshape(gridshape, order="F")
    # the first particle gives 0.25 of the mass of each cell, and a second particle of each cell gives the other part
    dfgridcontributions = pl.DataFrame({
        "particleid": [*([1] * dfmodel.height), *range(2, dfmodel.height + 2)],
        "cellindex": [*dfmodel["inputcellid"], *dfmodel["inputcellid"]],
        "frac_of_cellmass": [*([0.25] * dfmodel.height), *([0.75] * dfmodel.height)],
    })

    expectedcoordinates = {2: "spherical2d", 1: "spherical1d", 0: "spherical1d"}
    summedaxes: dict[int, tuple[int, ...]] = {2: (2,), 1: (1, 2), 0: (0, 1, 2)}
    for outputdimensions, axes in summedaxes.items():
        dfmodel_out, _, dfcontributions_out, modelmeta_out = at.inputmodel.dimension_reduce_model(
            dfmodel, outputdimensions, dfgridcontributions=dfgridcontributions, modelmeta=dict(modelmeta)
        )
        assert dfcontributions_out is not None
        expectedmass = cellmass.sum(axis=axes).ravel(order="F")
        np.testing.assert_allclose(dfmodel_out["mass_g"].to_numpy(), expectedmass, rtol=1e-12)
        np.testing.assert_allclose(
            dfmodel_out["X_Ni56"].to_numpy(),
            (cellmass * cellni56).sum(axis=axes).ravel(order="F") / expectedmass,
            rtol=1e-12,
        )
        # the density of an output cell and the volume from the output metadata give the same mass again
        np.testing.assert_allclose(
            at.inputmodel
            .add_derived_cols_to_modeldata(dfmodel_out.drop("mass_g"), modelmeta_out)
            .select("mass_g")
            .collect()
            .to_series()
            .to_numpy(),
            expectedmass,
            rtol=1e-12,
        )
        assert modelmeta_out["coordinates"] == expectedcoordinates[outputdimensions]
        assert "phi_edges_radians" not in modelmeta_out
        if outputdimensions == 1:
            assert dfmodel_out["vel_r_max_kmps"].to_list() == modelmeta["vel_r_edges_kmps"][1:]

        np.testing.assert_allclose(
            dfcontributions_out.filter(pl.col("particleid") == 1).sort("cellindex")["frac_of_cellmass"].to_numpy(),
            0.25,
            rtol=1e-12,
        )
        # the second particle of the first input cell gives a part of the first output cell. The cell ids start at 0
        # for a spherical2d model (format version 2), and at 1 for the 1D model that artistools writes
        assert dfcontributions_out.filter(pl.col("particleid") == 2).row(0, named=True) == pytest.approx({
            "particleid": 2,
            "cellindex": 0 if outputdimensions == 2 else 1,
            "frac_of_cellmass": 0.75 * cellmass[0, 0, 0] / expectedmass[0],
        })


@pytest.mark.parametrize("normalisekernelweights", [False, True])
def test_spherical_deposit_agrees_with_a_sum_over_all_cells(normalisekernelweights: bool) -> None:
    """The deposit of the spherical map finds each cell mid-point in a kernel, for both deposit methods.

    The deposit searches a region of r, theta, and phi around a particle. The region must hold the kernel also at
    the polar axis, across phi = 0, and at the centre. The default method is the kernel sum of the Cartesian map.
    With normalised weights, the grid keeps the mass of each particle inside it, and a particle with no cell
    mid-point in its kernel gives its mass to the cell that holds it.
    """
    from artistools.inputmodel.maptogrid import get_spherical_cell_volumes
    from artistools.inputmodel.maptogrid import get_spherical_deposit
    from artistools.inputmodel.maptogrid import get_spherical_grid_edges
    from artistools.inputmodel.maptogrid import get_wij
    from artistools.inputmodel.maptogrid import kernelvals2

    # -setgrid_fractionrmax 0 gave edges that do not increase, and cells with no volume
    with pytest.raises(ValueError, match="must be a finite number above 0"):
        get_spherical_grid_edges(0.0, 6, 5, 8)
    r_outer = 1000.0
    edges = get_spherical_grid_edges(r_outer, 6, 5, 8)
    rng = np.random.default_rng(seed=3)
    position = np.vstack((
        [
            [5.0, 3.0, 600.0],  # near the +z axis
            [400.0, 2.0, 50.0],  # on one side of phi = 0
            [400.0, -2.0, 50.0],  # on the other side of phi = 0
            [3.0, -4.0, 5.0],  # near the centre
            [0.0, 990.0, 10.0],  # near the outer radius
            [0.0, 0.0, 1200.0],  # outside the grid
            [-300.0, 200.0, -100.0],  # a kernel that holds no cell mid-point
            [1.0, 0.0, -700.0],  # near the -z axis
            [0.0, 0.0, 0.0],  # at the centre, with the mid-points of the second shell at the edge of its kernel
        ],
        rng.uniform(-550.0, 550.0, (40, 3)),
    ))
    # the kernel radius 2h of the particle at the centre is the mid-point radius of the second shell
    h_centre = (edges[0][1] + edges[0][2]) / 4.0
    h = np.concatenate(([150.0, 80.0, 80.0, 100.0, 60.0, 400.0, 1e-3, 120.0, h_centre], rng.uniform(20.0, 300.0, 40)))
    mass = rng.uniform(1.0, 2.0, len(h))
    wij = get_wij()

    pairparticle, paircell, pairmass, weightsums = get_spherical_deposit(
        position, h, mass, edges, wij, normalisekernelweights
    )
    # A cell mid-point at the edge of a kernel has a weight of zero. Such a pair counted as a tracer of the cell, and
    # a cell with only such pairs got 0 / 0 as the fraction of its mass
    assert (pairmass > 0.0).all()

    r_mid, theta_mid, phi_mid = np.meshgrid(*((edge[:-1] + edge[1:]) / 2.0 for edge in edges), indexing="ij")
    cellposition = np.stack(
        (
            r_mid * np.sin(theta_mid) * np.cos(phi_mid),
            r_mid * np.sin(theta_mid) * np.sin(phi_mid),
            r_mid * np.cos(theta_mid),
        ),
        axis=-1,
    ).reshape((-1, 3), order="F")
    volumes = get_spherical_cell_volumes(*edges).ravel(order="F")
    expected = np.zeros((len(h), volumes.size))
    for n in range(len(h)):
        dis2 = np.sum((cellposition - position[n]) ** 2, axis=1)
        weights = np.where(dis2 < (2.0 * h[n]) ** 2, kernelvals2(np.minimum(dis2, 4.0 * h[n] ** 2), h[n], wij), 0.0)
        weights *= volumes
        if not normalisekernelweights:
            # the density of a cell is the sum of mass times kernel value, as in the Cartesian map
            expected[n] = mass[n] * weights
        elif np.linalg.norm(position[n]) > r_outer:
            continue
        elif weights.sum() > 0.0:
            expected[n] = mass[n] * weights / weights.sum()
        else:
            radius = float(np.linalg.norm(position[n]))
            index_r, index_theta, index_phi = (
                int(np.searchsorted(edge, value, side="right")) - 1
                for edge, value in zip(
                    edges,
                    (
                        radius,
                        math.acos(position[n, 2] / radius),
                        math.atan2(position[n, 1], position[n, 0]) % (2.0 * math.pi),
                    ),
                    strict=True,
                )
            )
            expected[n, index_r + 6 * (index_theta + 5 * index_phi)] = mass[n]

    actual = np.zeros_like(expected)
    np.add.at(actual, (pairparticle, paircell), pairmass)
    # a cell mid-point at the edge of a kernel has a weight near zero. The two distance formulas can put it on
    # different sides of the edge
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)

    isinside = np.linalg.norm(position, axis=1) <= r_outer
    assert weightsums[6] == pytest.approx(0.0, abs=0.0)
    if normalisekernelweights:
        assert np.isnan(weightsums[~isinside]).all()
        assert not actual[~isinside].any()
        np.testing.assert_allclose(actual.sum(axis=1)[isinside], mass[isinside], rtol=1e-12)
    else:
        # the particle outside the grid has cell mid-points in its kernel, and the particle with a very small
        # kernel has none
        assert actual[~isinside].any()
        assert not actual[6].any()
        np.testing.assert_allclose(actual.sum(axis=1), mass * weightsums, rtol=1e-12)


def test_spherical_map_gives_each_particle_the_cell_of_its_direction(tmp_path: Path) -> None:
    """The cell order and the angles of the spherical map agree with the reader of the model.

    A particle on the +z axis has theta = 0. A particle near the -y axis has phi = atan2(y, x) near 3 pi / 2. The
    kernels hold no cell mid-point. With normalised kernel weights, each particle then gives its mass to the cell
    that holds it.
    """
    from artistools.inputmodel.maptogrid import maptogrid

    def snapshotline(particleid: int, position: tuple[float, float, float]) -> str:
        values = [0.0] * 31
        values[0] = particleid
        values[1] = 1.0  # h
        values[2:5] = position
        for column in (14, 15, 16, 17):  # pmass, rho, p, rho_rst
            values[column] = 1.0
        values[20] = 0.3  # ye
        return " ".join(str(value) for value in values)

    (tmp_path / "ejectasnapshot.dat").write_text(
        snapshotline(1, (0.0, 0.0, 1000.0)) + "\n" + snapshotline(2, (100.0, -1000.0, 100.0)) + "\n", encoding="utf-8"
    )
    # the snapshot is 20000 time units after the merger, thus the velocities are below the speed of light
    (tmp_path / "test_info.dat").write_text("0 0 20000.0\n", encoding="utf-8")
    (tmp_path / "tmerger.txt").write_text("# time of merger in geom units\n0.0\n", encoding="utf-8")

    gridshape = (4, 4, 8)
    maptogrid(
        tmp_path / "ejectasnapshot.dat",
        tmp_path,
        dtextra_seconds=0.0,
        setgrid_fractionrmax=1.5,
        modifysmoothinglength="False",
        coordinates="spherical",
        sphericalgridshape=gridshape,
        normalisekernelweights=True,
    )
    at.inputmodel.modelfromhydro.main(
        argsraw=[], gridfolderpath=tmp_path, outputpath=tmp_path / "model", dimensions=3, timedays=0.1
    )

    lzmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path / "model")
    assert tuple(modelmeta[f"ncoordgrid{axis}"] for axis in ("r", "theta", "phi")) == gridshape
    dfcells = (
        at.inputmodel
        .add_derived_cols_to_modeldata(lzmodel, modelmeta)
        .filter(pl.col("rho") > 0.0)
        .sort("modelgridindex")
        .collect()
    )
    # both particles are in the third shell. The first one is in the first cell of theta and of phi. The second one
    # is in the cell with 60 <= theta < 90 degrees and 270 <= phi < 315 degrees
    assert dfcells["modelgridindex"].to_list() == [2, 2 + 4 * (1 + 4 * 6)]
    onaxis, nearminusy = dfcells.rows(named=True)
    assert onaxis["vel_z_mid"] > onaxis["vel_rcyl_mid"] > 0.0
    assert nearminusy["vel_y_mid"] < 0.0 < nearminusy["vel_x_mid"] < -nearminusy["vel_y_mid"]
    assert nearminusy["vel_z_mid"] > 0.0
    np.testing.assert_allclose(dfcells["mass_g"].to_numpy(), dfcells.item(0, "mass_g"), rtol=1e-6)


def test_spherical_map_and_cartesian_map_give_the_same_shell_masses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two maps of one snapshot estimate the same density field, thus their radial shells hold equal masses.

    The kernel of each particle is inside both grids and wider than a cell. The kernel sum of the Cartesian map then
    gives the mass pmass * rho / rho_rst of the particles to 1e-3. The spherical map has the same method by default,
    and its cells have different sizes, thus it gives that mass to 0.03. With normalised kernel weights, the
    spherical map keeps that mass exactly.

    The cumulative mass of the shells of each spherical map agrees with the Cartesian map to less than 0.05 of the
    total mass. The values are not equal, because the Cartesian cells do not follow the shells.
    """
    from artistools.inputmodel.maptogrid import get_spherical_cell_volumes
    from artistools.inputmodel.maptogrid import maptogrid

    rng = np.random.default_rng(seed=7)
    particlecount, ncoordgrid, r_outer = 400, 20, 1000.0
    direction = rng.normal(size=(particlecount, 3))
    position = direction / np.linalg.norm(direction, axis=1)[:, np.newaxis]
    position *= (600.0 * np.cbrt(rng.random(particlecount)))[:, np.newaxis]
    # the largest radius of the particles sets the size of the grids
    position[0] = (600.0, 0.0, 0.0)
    pmass = 1.0 + rng.random(particlecount)
    rho_on_rho_rst = 0.5 + 0.5 * rng.random(particlecount)
    snapshotlines = []
    for particleindex in range(particlecount):
        values = [0.0] * 31
        values[0] = particleindex + 1
        values[1] = 150.0  # h
        values[2:5] = position[particleindex].tolist()
        values[14] = pmass[particleindex]
        values[15] = rho_on_rho_rst[particleindex]  # rho
        values[17] = 1.0  # rho_rst
        values[20] = 0.3  # ye
        snapshotlines.append(" ".join(str(value) for value in values))

    for foldername, coordinates, normalisekernelweights in (
        ("cartesian", "cartesian", False),
        ("spherical", "spherical", False),
        ("spherical_normalised", "spherical", True),
    ):
        folder = tmp_path / foldername
        folder.mkdir()
        (folder / "ejectasnapshot.dat").write_text("\n".join(snapshotlines) + "\n", encoding="utf-8")
        maptogrid(
            folder / "ejectasnapshot.dat",
            folder,
            ncoordgrid=ncoordgrid,
            dtextra_seconds=0.0,
            setgrid_fractionrmax=r_outer / 600.0,
            modifysmoothinglength="False",
            coordinates=coordinates,
            sphericalgridshape=(ncoordgrid // 2, 10, 20),
            normalisekernelweights=normalisekernelweights,
        )

    dfcartesian = at.misc.read_wsv(tmp_path / "cartesian" / "grid.dat", skip_rows=3)
    cellwidth = 2.0 * r_outer / ncoordgrid
    cellradius = np.sqrt(
        sum((dfcartesian[f"pos_{axis}_min"].to_numpy() + cellwidth / 2.0) ** 2 for axis in ("x", "y", "z"))
    )
    cellmass_cartesian = dfcartesian["rho"].to_numpy() * cellwidth**3

    headerlines = (tmp_path / "spherical" / "grid_spherical.dat").read_text(encoding="utf-8").splitlines()[2:5]
    r_edges, theta_edges, phi_edges = (np.array(line.split()[1:], dtype=np.float64) for line in headerlines)
    cellvolumes = get_spherical_cell_volumes(r_edges, theta_edges, phi_edges).ravel(order="F")
    shellmass_cartesian, _ = np.histogram(cellradius, bins=r_edges, weights=cellmass_cartesian)

    particlemass = float(np.sum(pmass * rho_on_rho_rst))
    assert math.isclose(cellmass_cartesian.sum(), particlemass, rel_tol=1e-3)
    for foldername, mass_rtol in (("spherical", 0.03), ("spherical_normalised", 1e-12)):
        dfspherical = at.misc.read_wsv(tmp_path / foldername / "grid_spherical.dat", skip_rows=5)
        shellmass_spherical = (
            (dfspherical["rho"].to_numpy() * cellvolumes).reshape((len(r_edges) - 1, -1), order="F").sum(axis=1)
        )
        assert math.isclose(shellmass_spherical.sum(), particlemass, rel_tol=mass_rtol), foldername
        cumulativedifference = np.cumsum(shellmass_spherical - shellmass_cartesian) / particlemass
        assert np.abs(cumulativedifference).max() < 0.05, foldername

    # maptogrid writes a number for each cell, thus the reader stops for a damaged value and does not give it rho = 0
    damagedfolder = tmp_path / "damaged"
    shutil.copytree(tmp_path / "spherical", damagedfolder)
    (damagedfolder / "test_info.dat").write_text("0 0 20000.0\n", encoding="utf-8")
    (damagedfolder / "tmerger.txt").write_text("# time of merger in geom units\n0.0\n", encoding="utf-8")
    gridlines = (tmp_path / "spherical" / "grid_spherical.dat").read_text(encoding="utf-8").splitlines()
    gridindex, _, ye, tracercount = gridlines[-1].split()
    gridlines[-1] = f"{gridindex} nan {ye} {tracercount}"
    (damagedfolder / "grid_spherical.dat").write_text("\n".join(gridlines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.modelfromhydro.read_griddat_file(damagedfolder)
    captured = capsys.readouterr()
    assert "1 cells have no valid rho or Ye" in captured.out + captured.err

    # a grid index that occurs two times and one that is absent give the same count of rows
    gridlines[-1] = " ".join([gridlines[-2].split()[0], *gridlines[-1].split()[1:]]).replace("nan", "0.0")
    (damagedfolder / "grid_spherical.dat").write_text("\n".join(gridlines) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        at.inputmodel.modelfromhydro.read_griddat_file(damagedfolder)
    captured = capsys.readouterr()
    assert "the grid indices must be 0 to" in captured.out + captured.err


@pytest.mark.parametrize("mapto3d", [False, True])
def test_from_e2e_model_spherical_grid_holds_both_halves_of_a_symmetric_model(tmp_path: Path, mapto3d: bool) -> None:
    """A spherical grid holds both halves of a model with equatorial symmetry, with two times the tracer mass.

    The tracers of such a model are above the plane z = 0. The cylindrical grid maps one half and reflects it. A
    spherical grid has no reflection: the map reads each cell at |z|, and the normalisation takes two times the
    tracer mass. The contributions of the particles then name each cell of both halves.
    """
    from artistools.inputmodel.from_e2e_model import t_model_init_s

    rng = np.random.default_rng(seed=5)
    ntraj, ntimes = 60, 8
    datpath = tmp_path / "e2emodel.npz"
    np.savez(
        datpath,
        # each polar angle is below pi / 2, thus the model has equatorial symmetry
        pos=np.column_stack([rng.uniform(0.05, 0.3, ntraj), rng.uniform(0.2, 1.5, ntraj)]),
        idx=np.arange(1, ntraj + 1, dtype=float),
        state=np.zeros(ntraj),
        mass=np.full(ntraj, 1e-3),
        qdot=np.full((ntraj, ntimes), 1e10),
        hnuloss=np.zeros((ntraj, ntimes)),
        time=np.linspace(0.0, 2.0 * t_model_init_s, ntimes),
        # number fractions whose mass fractions sum to less than 1, because sn3d stops for a larger sum
        nz=rng.uniform(0.01, 0.2, size=(ntraj, 2)) * np.array([0.5, 0.01]),
        t5out=np.column_stack([np.zeros((ntraj, 4)), np.full(ntraj, 0.3)]),
    )
    np.save(tmp_path / "iso_table.npy", np.array([[2, 2], [26, 30]]))

    gridshape = (6, 8, 4) if mapto3d else (6, 8)
    argsraw = ["-npz", str(datpath), "-vmax_on_c", "0.4", "--coordinates", "spherical"]
    argsraw += ["--ngrid-r", "6", "--ngrid-theta", "8", *(["--mapto3D", "--ngrid-phi", "4"] if mapto3d else [])]
    at.inputmodel.from_e2e_model.main(argsraw=[*argsraw, "-o", str(tmp_path / "model")])

    lzmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path / "model", printwarningsonly=True)
    assert modelmeta["coordinates"] == ("spherical3d" if mapto3d else "spherical2d")
    dfmodel = at.inputmodel.add_derived_cols_to_modeldata(lzmodel, modelmeta).collect()
    cellmass = dfmodel["mass_g"].to_numpy().reshape(gridshape, order="F")

    tracermass_g = ntraj * 1e-3 * at.constants.Msun_to_g
    assert math.isclose(cellmass.sum(), 2.0 * tracermass_g, rel_tol=1e-6)
    # the polar cells are uniform in cos(theta), thus cell i_theta and cell n_theta - 1 - i_theta are mirror cells
    np.testing.assert_allclose(cellmass, cellmass[:, ::-1], rtol=1e-6)
    if mapto3d:
        # the tracers have rotational symmetry around the z axis
        np.testing.assert_allclose(cellmass, np.broadcast_to(cellmass[:, :, :1], cellmass.shape), rtol=1e-6)

    dfcontributions = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(tmp_path / "model")
    dfcellfractions = dfcontributions.group_by("cellindex").agg(pl.col("frac_of_cellmass").sum()).sort("cellindex")
    assert dfcellfractions["cellindex"].to_list() == dfmodel.filter(pl.col("rho") > 0)["inputcellid"].to_list()
    np.testing.assert_allclose(dfcellfractions["frac_of_cellmass"].to_numpy(), 1.0, rtol=1e-6)


def test_from_e2e_model_spherical_grid_agrees_with_the_cylindrical_grid(tmp_path: Path) -> None:
    """A 2D spherical map and a 2D cylindrical map of one model give the same mass in each radial shell.

    The spherical map has the method of the cylindrical map, and both hold the tracer mass after the normalisation.
    The cumulative mass of the shells agrees to less than 0.05 of the total mass. The values are not equal, because
    the cylindrical cells do not follow the shells.
    """
    datpath = tmp_path / "e2emodel.npz"
    write_e2e_model(datpath, tmp_path / "iso_table.npy")
    baseargs = ["-npz", str(datpath), "-vmax_on_c", "0.4"]
    gridargs = {
        "cylindrical": ["-ngridrcyl", "24", "-ngridz", "48"],
        "spherical": ["--coordinates", "spherical", "--ngrid-r", "24", "--ngrid-theta", "24"],
    }
    shelledges_on_c = np.linspace(0.0, 0.4, 9)
    shellmassfractions = {}
    for coordinates, coordinateargs in gridargs.items():
        at.inputmodel.from_e2e_model.main(argsraw=[*baseargs, *coordinateargs, "-o", str(tmp_path / coordinates)])
        lzmodel, modelmeta = at.inputmodel.scan_modeldata(tmp_path / coordinates, printwarningsonly=True)
        dfmodel = at.inputmodel.add_derived_cols_to_modeldata(lzmodel, modelmeta).collect()
        shellmass, _ = np.histogram(
            dfmodel["vel_r_mid_on_c"].to_numpy(), bins=shelledges_on_c, weights=dfmodel["mass_g"].to_numpy()
        )
        # 40 tracers of 1e-3 Msun
        assert math.isclose(dfmodel["mass_g"].sum(), 0.04 * at.constants.Msun_to_g, rel_tol=1e-6), coordinates
        shellmassfractions[coordinates] = shellmass / dfmodel["mass_g"].sum()

    cumulativedifference = np.cumsum(shellmassfractions["spherical"] - shellmassfractions["cylindrical"])
    assert np.abs(cumulativedifference).max() < 0.05


def test_scan_modeldata_of_a_model_file_reads_the_abundances_beside_it() -> None:
    """The path of a model file, e.g. model.txt, takes the abundances.txt of its folder.

    The reader searched the path of the file as a folder, thus it found no abundances.txt.
    """
    dffromfolder, _ = at.inputmodel.scan_modeldata(modelpath, get_elemabundances=True, printwarningsonly=True)
    dffromfile, _ = at.inputmodel.scan_modeldata(
        modelpath / "model.txt", get_elemabundances=True, printwarningsonly=True
    )
    pltest.assert_frame_equal(dffromfile.collect(), dffromfolder.collect())
