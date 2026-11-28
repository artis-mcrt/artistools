import math
import os
import stat
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"


def get_reference_dirbin(dirx: float, diry: float, dirz: float, nphibins: int, ncosthetabins: int) -> int:
    """Return the direction bin of one packet, computed in float64 as get_escapedirectionbin of ARTIS (vectors.h).

    The polars binning of the package works in Float32, thus this function checks it by a separate path. ARTIS takes
    cosphi = 1 for a direction along the z axis, thus such a direction is in the phi bin of the angle pi and not in
    the phi bin 0.
    """
    dirmag = math.sqrt(dirx**2 + diry**2 + dirz**2)
    unitx, unity, unitz = dirx / dirmag, diry / dirmag, dirz / dirmag
    costhetabin = min(max(int((unitz + 1.0) * ncosthetabins / 2.0), 0), ncosthetabins - 1)

    sintheta = math.sqrt(unitx**2 + unity**2)
    cosphi = min(max(unitx / sintheta, -1.0), 1.0) if sintheta > 1e-12 else 1.0
    phi = math.acos(cosphi) if unity < 0 else math.acos(cosphi) + math.pi
    return costhetabin * nphibins + min(max(int(phi / 2.0 / math.pi * nphibins), 0), nphibins - 1)


def test_directionbins() -> None:
    nphibins = 10
    ncosthetabins = 10
    costhetabinlowers, costhetabinuppers, _ = at.misc.get_costheta_bins(usedegrees=False)
    phibinlowers, phibinuppers, _ = at.misc.get_phi_bins(usedegrees=False)

    testdirections = pl.DataFrame({
        "phi_defined": np.linspace(0.1, 2 * math.pi, nphibins * 2, endpoint=False).tolist()
    }).join(
        pl.DataFrame({"costheta_defined": np.linspace(0.0, 1.0, ncosthetabins * 2, endpoint=True).tolist()}),
        how="cross",
    )

    testdirections = testdirections.with_columns(
        dirx=((1.0 - pl.col("costheta_defined").pow(2)).sqrt() * pl.col("phi_defined").cos()),
        diry=((1.0 - pl.col("costheta_defined").pow(2)).sqrt() * pl.col("phi_defined").sin()),
        dirz=pl.col("costheta_defined"),
    )

    testdirections = at.packets.add_packet_directions_lazypolars(testdirections).collect()
    testdirections = at.packets.bin_packet_directions_polars(testdirections).collect()

    for pkt in testdirections.iter_rows(named=True):
        assert np.isclose(pkt["dirx"] ** 2 + pkt["diry"] ** 2 + pkt["dirz"] ** 2, 1.0, rtol=0.001)

        assert np.isclose(pkt["costheta_defined"], pkt["costheta"], rtol=1e-4, atol=1e-4)
        pktdir_is_along_zaxis = np.isclose(pkt["dirz"], 1.0) or np.isclose(pkt["dirz"], -1.0)

        assert np.isclose(pkt["phi_defined"], pkt["phi"], rtol=1e-4, atol=1e-4) or pktdir_is_along_zaxis

        assert costhetabinlowers[pkt["costhetabin"]] <= pkt["costheta_defined"] * 1.01
        assert costhetabinuppers[pkt["costhetabin"]] > pkt["costheta_defined"] * 0.99

        assert pkt["dirbin"] == get_reference_dirbin(pkt["dirx"], pkt["diry"], pkt["dirz"], nphibins, ncosthetabins)
        assert pkt["costhetabin"] == pkt["dirbin"] // nphibins
        assert pkt["phibin"] == pkt["dirbin"] % nphibins

        assert phibinlowers[pkt["phibin"]] <= pkt["phi_defined"] or pktdir_is_along_zaxis
        assert phibinuppers[pkt["phibin"]] >= pkt["phi_defined"] or pktdir_is_along_zaxis


def test_directionbin_takes_the_sign_of_a_tiny_diry() -> None:
    """A tiny negative diry selects the phi range of ARTIS, which tests the sign of dir[1] in double precision.

    The old code tested the sign of -diry / |dir| after a cast to Float32, and 1e-300 became zero there.
    """
    dfpackets = pl.LazyFrame({"dirx": [0.96], "diry": [-1e-300], "dirz": [0.28]})
    dfpackets = at.packets.bin_packet_directions_polars(at.packets.add_packet_directions_lazypolars(dfpackets))

    assert dfpackets.select("dirbin").collect().item() == get_reference_dirbin(0.96, -1e-300, 0.28, 10, 10)


def test_directionbins_unequal_bincounts() -> None:
    """Check the dirbin layout when the phi and costheta bin counts differ.

    The default configuration uses 10 of each, which hides any confusion between the two counts.
    """
    nphibins = 8
    ncosthetabins = 4

    testdirections = pl.DataFrame({
        "phi_defined": np.linspace(0.05, 2 * math.pi, nphibins * 3, endpoint=False).tolist()
    }).join(
        pl.DataFrame({"costheta_defined": np.linspace(-0.99, 0.99, ncosthetabins * 3, endpoint=True).tolist()}),
        how="cross",
    )

    testdirections = testdirections.with_columns(
        dirx=((1.0 - pl.col("costheta_defined").pow(2)).sqrt() * pl.col("phi_defined").cos()),
        diry=((1.0 - pl.col("costheta_defined").pow(2)).sqrt() * pl.col("phi_defined").sin()),
        dirz=pl.col("costheta_defined"),
    )

    testdirections = at.packets.add_packet_directions_lazypolars(testdirections).collect()
    testdirections = at.packets.bin_packet_directions_polars(
        testdirections, nphibins=nphibins, ncosthetabins=ncosthetabins
    ).collect()

    for pkt in testdirections.iter_rows(named=True):
        assert 0 <= pkt["phibin"] < nphibins
        assert 0 <= pkt["costhetabin"] < ncosthetabins
        assert 0 <= pkt["dirbin"] < nphibins * ncosthetabins

        # dirbin packs the costheta index in the high part and the phi index in the low part
        assert pkt["dirbin"] == pkt["costhetabin"] * nphibins + pkt["phibin"]
        assert pkt["dirbin"] == get_reference_dirbin(pkt["dirx"], pkt["diry"], pkt["dirz"], nphibins, ncosthetabins)


@pytest.mark.parametrize(
    ("direction", "expecteddirbin"),
    [
        ((0.0, 0.0, 1.0), 95),
        ((0.0, 0.0, -1.0), 5),
        ((0.0, 1e-30, -1.0), 5),
        ((0.0, -1e-30, 1.0), 90),
        ((0.0, 1e-13, 1.0), 95),
        ((-1.0, 0.0, 0.0), 59),
    ],
)
def test_a_direction_on_an_edge_of_the_phi_bins_gets_the_bin_of_artis(
    direction: tuple[float, float, float], expecteddirbin: int
) -> None:
    """A direction along the z axis, or along -x, gets the direction bin of ARTIS (vectors.h).

    ARTIS takes cosphi = 1 along the z axis. The division 0/0 gave cosphi = NaN, thus the direction got phi bin 0.
    A small component across the axis gave the true phi, which ARTIS ignores below 1e-12. Along -x, phi is 2 pi,
    which must stay in the last phi bin.
    """
    dirx, diry, dirz = direction
    dfpackets = at.packets.add_packet_directions_lazypolars(
        pl.DataFrame({"dirx": [dirx], "diry": [diry], "dirz": [dirz]})
    )
    binned = at.packets.bin_packet_directions_polars(dfpackets, nphibins=10, ncosthetabins=10).collect()

    assert binned["dirbin"].item() == expecteddirbin
    assert get_reference_dirbin(dirx, diry, dirz, nphibins=10, ncosthetabins=10) == expecteddirbin


def test_scan_virtual_packets() -> None:
    nprocs_read, dfvpkt = at.packets.scan_virtual_packets(
        modelpath=at.get_path("testdata") / "vpktcontrib", maxpacketfiles=2
    )
    dfvpkt = dfvpkt.collect()

    assert nprocs_read == 2
    assert dfvpkt.height == 13783
    assert dfvpkt.columns == [
        "emissiontype",
        "trueemissiontype",
        "absorption_type",
        "absorption_freq",
        "dir0_t_arrive_d",
        "dir0_nu_rf",
        "dir0_e_rf_0",
        "dir0_e_rf_1",
        "dir0_e_rf_2",
        "dir1_t_arrive_d",
        "dir1_nu_rf",
        "dir1_e_rf_0",
        "dir1_e_rf_1",
        "dir1_e_rf_2",
        "dir2_t_arrive_d",
        "dir2_nu_rf",
        "dir2_e_rf_0",
        "dir2_e_rf_1",
        "dir2_e_rf_2",
        "mpirank",
        "type_id",
        "escape_type_id",
    ]
    assert dfvpkt.schema["emissiontype"] == pl.Int32
    assert dfvpkt.schema["dir0_t_arrive_d"] == pl.Float32
    assert dfvpkt.schema["dir0_e_rf_0"] == pl.Float64
    assert dfvpkt.schema["mpirank"] == pl.Int32
    assert dfvpkt["dir0_t_arrive_d"].is_sorted()
    assert dfvpkt["type_id"].unique().to_list() == [32]
    assert dfvpkt["escape_type_id"].unique().to_list() == [11]
    assert dfvpkt["dir0_t_arrive_d"].min() == pytest.approx(-1.0)
    assert dfvpkt["dir0_t_arrive_d"].max() == pytest.approx(145.587997)

    rank_summary = (
        dfvpkt
        .group_by("mpirank")
        .agg(pl.len().alias("packet_count"), pl.col("dir0_e_rf_0").sum().alias("energy_sum"))
        .sort("mpirank")
    )
    assert rank_summary["mpirank"].to_list() == [0, 1]
    assert rank_summary["packet_count"].to_list() == [9402, 4381]
    assert rank_summary["energy_sum"].to_list() == pytest.approx([5.56564454996292e44, 1.8265647804307455e44])


def test_sum_packets_by_dirbin_includes_both_outer_edges() -> None:
    """Every value between the first and last edge must land in a bin, including values on either outer edge."""
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    df = pl.LazyFrame({"x": values, "e": [1.0] * len(values)})

    sums, _, counts, solidanglefactor = at.packets.sum_packets_by_dirbin(df, [-1], "x", [0.0, 1.0, 2.0, 3.0], "e")[-1]

    # bins are [lower, upper), except the last which also includes its upper edge
    assert counts.tolist() == [2, 2, 3]
    assert sums.tolist() == pytest.approx([2.0, 2.0, 3.0])
    assert solidanglefactor == pytest.approx(1.0)


def test_readfile_text_drops_trailing_null_column(tmp_path: Path) -> None:
    """The all-null column produced by the trailing space on each packets line must not reach the DataFrame.

    The drop used to run after the mpirank column had been appended, so it tested mpirank (never null) and the
    null column survived into every cached parquet file.
    """
    columns = ["number", "where", "type_id", "posx", "posy", "posz"]
    # each line ends with a space, exactly as ARTIS writes them
    lines = ["1 58900 32 -2.2e16 2.7e15 -1.6e15 " for _ in range(3)]
    packetsfile = tmp_path / "packets00_0000.out"
    packetsfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from artistools.packets.core import readfile_text

    dfpackets = readfile_text(packetsfile, column_names=columns)

    assert dfpackets.columns == [*columns, "mpirank"]
    assert dfpackets["mpirank"].to_list() == [0, 0, 0]


def test_packets_cache_goes_stale_when_any_rank_file_changes(tmp_path: Path) -> None:
    """A change of the text file of a later rank makes the cache stale, also to a time before the stamp.

    The check read only the file of the first rank, and it compared in one direction. Thus a cache from a batch
    that a copy still wrote kept the partial data after the copy ended, e.g. a copy with cp -p.
    """
    import shutil

    from artistools.packets.core import get_packets_rankbatch_parquetfile

    sourcedir = at.get_path("testdata") / "test-classicmode_3d" / "packets"
    for rank in (0, 1):
        shutil.copy(sourcedir / f"packets00_{rank:04d}.out.zst", tmp_path)
    textfiles = [tmp_path / f"packets00_{rank:04d}.out.zst" for rank in (0, 1)]

    def get_stamp() -> float:
        parquetpath = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)
        return float(pl.read_parquet_metadata(parquetpath)["textsource_mtime"])

    assert np.isclose(get_stamp(), max(path.stat().st_mtime for path in textfiles), rtol=0.0, atol=1e-3)

    laterrankfile = textfiles[1]
    newtime = laterrankfile.stat().st_mtime + 100.0
    os.utime(laterrankfile, (newtime, newtime))
    assert np.isclose(get_stamp(), newtime, rtol=0.0, atol=1e-3)

    oldtime = newtime - 1000.0
    os.utime(laterrankfile, (oldtime, oldtime))
    assert np.isclose(get_stamp(), textfiles[0].stat().st_mtime, rtol=0.0, atol=1e-3)


def test_cut_rank_file_of_a_batch_gives_an_error_that_names_the_file(tmp_path: Path) -> None:
    """A cut .xz file gives a polars panic, which is not an Exception, and the error must name the rank file."""
    import lzma

    from artistools.packets.core import get_packets_rankbatch_parquetfile

    sourcedir = at.get_path("testdata") / "test-classicmode_3d" / "packets"
    for rank in (0, 1):
        with at.zopen(sourcedir / f"packets00_{rank:04d}.out.zst", mode="rb") as sourcefile:
            compressed = lzma.compress(sourcefile.read())
        (tmp_path / f"packets00_{rank:04d}.out.xz").write_bytes(compressed)
    cutfile = tmp_path / "packets00_0001.out.xz"
    cutfile.write_bytes(cutfile.read_bytes()[: cutfile.stat().st_size // 2])

    with pytest.raises(pl.exceptions.PanicException) as excinfo:
        get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)

    assert any(str(cutfile) in note for note in excinfo.value.__notes__ or [])


def test_virtual_packets_file_with_no_data_lines_gives_no_rows(tmp_path: Path) -> None:
    """ARTIS writes a line only for a virtual packet that escapes, thus the file of a rank can hold only the header.

    polars stopped with "empty CSV" for such a file, thus the conversion of the whole batch stopped.
    """
    import shutil

    from artistools.packets.core import get_packets_rankbatch_parquetfile

    sourcedir = at.get_path("testdata") / "vpktcontrib"
    shutil.copy(sourcedir / "vpackets_0000.out.zst", tmp_path)
    with at.zopen(sourcedir / "vpackets_0000.out.zst", mode="rt", encoding="utf-8") as sourcefile:
        headerline = sourcefile.readline()
    (tmp_path / "vpackets_0001.out").write_text(headerline, encoding="utf-8")

    parquetpath = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=True)
    dfvpkt = pl.read_parquet(parquetpath)

    assert dfvpkt.height == 9402
    assert dfvpkt["mpirank"].unique().to_list() == [0]
    assert dfvpkt.schema["dir0_t_arrive_d"] == pl.Float32
    assert dfvpkt.schema["emissiontype"] == pl.Int32


@pytest.mark.parametrize("virtual", [False, True])
def test_read_of_absent_packets_makes_no_cache_folder(tmp_path: Path, virtual: bool) -> None:
    """A read of the packets of a model with no packets files gives an error and leaves the model folder as it was.

    The reader made the folder packets or vpackets before it searched for the text files. The viewer then found the
    empty folder vpackets, and it enabled the observers of the virtual packets for a model that had none.
    """
    from artistools.packets.core import get_packets_rankbatch_parquetfile

    with pytest.raises(FileNotFoundError):
        get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0], batchindex=0, virtual=virtual)

    assert list(tmp_path.iterdir()) == []


def test_packets_cache_without_stamps_is_stale_when_the_text_files_exist(tmp_path: Path) -> None:
    """The reader replaces a cache with no stamp with a cache from the text files, as for a complete batch.

    The check of the first rank compares in one direction, and that rule also kept a cache with no stamp.
    """
    import shutil

    from artistools.packets.core import get_packets_rankbatch_parquetfile

    sourcedir = at.get_path("testdata") / "test-classicmode_3d" / "packets"
    for rank in (0, 1):
        shutil.copy(sourcedir / f"packets00_{rank:04d}.out.zst", tmp_path)
    (tmp_path / "packets").mkdir()
    cachepath = tmp_path / "packets" / "packetsbatch00_0000_0001.out.parquet.tmp"
    pl.DataFrame({"unstampedcolumn": [0]}).write_parquet(cachepath)

    parquetpath = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)

    assert parquetpath == cachepath
    assert "unstampedcolumn" not in pl.read_parquet_schema(parquetpath)


@pytest.mark.parametrize("virtual", [False, True])
def test_packets_cache_scans_each_folder_once(tmp_path: Path, virtual: bool) -> None:
    """Keep the number of directory scans independent of the number of ranks."""
    from artistools.packets.core import CACHEVERSION
    from artistools.packets.core import get_packets_rankbatch_parquetfile
    from artistools.packets.core import VIRTUAL_CACHEVERSION

    sourcefolder = tmp_path / "run1"
    sourcefolder.mkdir()
    prefix = "vpackets" if virtual else "packets00"
    sourcefiles = [sourcefolder / f"{prefix}_{rank:04d}.out" for rank in range(32)]
    for sourcefile in sourcefiles:
        sourcefile.touch()
    packetkind = "vpackets" if virtual else "packets"
    cachefolder = tmp_path / packetkind
    cachefolder.mkdir()
    cachepath = cachefolder / f"{packetkind}batch00_0000_0031.out.parquet.tmp"
    at.misc.write_parquet_atomic(
        pl.DataFrame({"number": [0]}),
        cachepath,
        metadata={
            "cacheversion": str(VIRTUAL_CACHEVERSION if virtual else CACHEVERSION),
            "textsource_mtime": str(max(path.stat().st_mtime for path in sourcefiles)),
        },
    )
    firstwrite = cachepath.stat().st_mtime_ns

    with mock.patch("os.scandir", wraps=os.scandir) as scandir:
        result = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=range(32), batchindex=0, virtual=virtual)

    assert result == cachepath
    assert result.stat().st_mtime_ns == firstwrite
    assert scandir.call_count <= 3


def test_packets_cache_check_stays_in_memory(tmp_path: Path) -> None:
    """A second read of the packets makes no scan of the folders, and a new cache gives a new check."""
    from artistools.packets.core import CACHEVERSION
    from artistools.packets.core import get_packets_batch_parquet_paths

    sourcefolder = tmp_path / "run1"
    sourcefolder.mkdir()
    sourcefiles = [sourcefolder / f"packets00_{rank:04d}.out" for rank in range(32)]
    for sourcefile in sourcefiles:
        sourcefile.touch()
    (tmp_path / "packets").mkdir()
    cachepath = tmp_path / "packets" / "packetsbatch00_0000_0031.out.parquet.tmp"
    metadata = {
        "cacheversion": str(CACHEVERSION),
        "textsource_mtime": str(max(path.stat().st_mtime for path in sourcefiles)),
    }
    at.misc.write_parquet_atomic(pl.DataFrame({"number": [0]}), cachepath, metadata=metadata)

    with (
        mock.patch("artistools.packets.core.get_nprocs", return_value=32),
        mock.patch("os.scandir", wraps=os.scandir) as scandir,
    ):
        assert get_packets_batch_parquet_paths(tmp_path) == (32, [cachepath])
        assert scandir.call_count > 0

        scandir.reset_mock()
        assert get_packets_batch_parquet_paths(tmp_path) == (32, [cachepath])
        assert scandir.call_count == 0

        # a new cache at the path can hold different data, thus the check runs again
        cachepath.unlink()
        at.misc.write_parquet_atomic(pl.DataFrame({"number": [1]}), cachepath, metadata=metadata)
        assert get_packets_batch_parquet_paths(tmp_path) == (32, [cachepath])
        assert scandir.call_count > 0

        # the reader searches the model folder before run1, thus a new text file there needs a new scan. The file is
        # older than the stamp, thus the cache stays current
        scandir.reset_mock()
        newtextfile = tmp_path / "packets00_0000.out"
        newtextfile.touch()
        os.utime(newtextfile, (1000.0, 1000.0))
        assert get_packets_batch_parquet_paths(tmp_path) == (32, [cachepath])
        assert scandir.call_count > 0


def test_packets_source_index_matches_the_reader(tmp_path: Path) -> None:
    """Use the source reader's order for folders and compressed files, and omit absent sources."""
    sourcefolder = tmp_path / "run1"
    sourcefolder.mkdir()
    filenames = [f"packets00_{rank:04d}.out" for rank in (0, 1, 10000)]
    paths = [
        tmp_path / f"{filenames[0]}.gz",
        sourcefolder / filenames[0],
        sourcefolder / f"{filenames[1]}.gz",
        sourcefolder / f"{filenames[1]}.zst",
        sourcefolder / f"{filenames[2]}.xz",
        sourcefolder / "packets00_note.out",
    ]
    for index, path in enumerate(paths):
        path.touch()
        os.utime(path, (1000 + index, 1000 + index))

    mtimes = at.packets.get_packets_textsource_mtimes(tmp_path, [*filenames, "packets00_0002.out"])
    expected = [at.firstexisting(filename, folder=tmp_path).stat().st_mtime for filename in filenames]
    assert sorted(mtimes) == pytest.approx(sorted(expected))


def test_add_derived_columns_gives_the_timestep_of_the_thermal_emission() -> None:
    """The timestep of the last thermal emission must use the same bins as the timestep of the last interaction."""
    modelpath = at.get_path("testdata") / "testmodel"
    tmids_s = [tmid * at.constants.day_to_s for tmid in at.get_timestep_times(modelpath, loc="mid")]
    dfpackets = pl.DataFrame({
        "em_posx": [0.0, 0.0, 0.0],
        "em_posy": [0.0, 0.0, 0.0],
        "em_posz": [0.0, 0.0, 0.0],
        "em_time": [tmids_s[52], tmids_s[52], tmids_s[3]],
        "trueem_time": [tmids_s[50], -1.0, tmids_s[3]],
        "true_emission_velocity": [1.0e8, math.nan, 1.0e8],
        "dirx": [0.0, 0.0, 0.0],
        "diry": [0.0, 0.0, 0.0],
        "dirz": [1.0, 1.0, 1.0],
    })

    dfderived = at.packets.add_derived_columns_lazy(dfpackets, modelpath).collect()
    assert dfderived["em_timestep"].to_list() == [52, 52, 3]
    assert dfderived["emtrue_timestep"].to_list() == [50, -1, 3]
    assert dfderived["emtrue_modelgridindex"].to_list() == [0, None, 0]

    # an old packets file has no trueem_time, thus it gets no timestep of the thermal emission
    dfold = at.packets.add_derived_columns_lazy(dfpackets.drop("trueem_time"), modelpath)
    assert "emtrue_timestep" not in dfold.collect_schema().names()


def test_a_3d_model_with_no_thermal_emission_position_gets_no_thermal_emission_timestep() -> None:
    """The timestep and the cell of the thermal emission select the estimators together, thus each needs the other.

    A 3D model with the thermal emission velocity and no position got the timestep and no cell. Thus the join of
    plotlinefluxes --plotemittingregions stopped with ColumnNotFoundError for emtrue_modelgridindex.
    """
    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    emtime_s = 5.0 * at.constants.day_to_s
    dfpackets = pl.DataFrame({
        "em_posx": [0.0],
        "em_posy": [0.0],
        "em_posz": [0.0],
        "em_time": [emtime_s],
        "true_emission_velocity": [1.0e9],
        "trueem_time": [emtime_s],
        "dirx": [0.0],
        "diry": [0.0],
        "dirz": [1.0],
    })

    derivedcolumns = at.packets.add_derived_columns_lazy(dfpackets, modelpath).collect_schema().names()

    assert "em_modelgridindex" in derivedcolumns
    assert "emtrue_modelgridindex" not in derivedcolumns
    assert "emtrue_timestep" not in derivedcolumns


def test_emission_expressions_give_no_value_for_a_packet_with_no_record() -> None:
    """ARTIS gives a time of 0 or -1 to a packet with no thermal emission record.

    An older ARTIS gives that packet a position of zero, and the current ARTIS gives it a position of NaN.
    """
    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    dfmodel, modelmeta = at.scan_modeldata(modelpath, printwarningsonly=True)
    emtime_s = 5.0 * at.constants.day_to_s
    dfpackets = pl.DataFrame({
        "trueem_posx": [1.0e9 * emtime_s, 0.0, 0.0, math.nan],
        "trueem_posy": [0.0, 0.0, 0.0, math.nan],
        "trueem_posz": [0.5e9 * emtime_s, 0.0, 0.0, math.nan],
        "trueem_time": [emtime_s, -1.0, 0.0, -1.0],
        "dirx": [0.0, 0.0, 0.0, 0.0],
        "diry": [0.0, 0.0, 0.0, 0.0],
        "dirz": [1.0, 1.0, 1.0, 1.0],
    })

    dfvalues = dfpackets.select(
        velocity=at.packets.get_emission_velocity_expr("trueem"),
        losvelocity=at.packets.get_emission_velocity_lineofsight_expr("trueem"),
        modelgridindex=at.packets.get_modelgridindex_expr("trueem", modelmeta, dfmodel),
    )

    assert np.isclose(dfvalues["velocity"][0], math.hypot(1.0e9, 0.5e9), rtol=1e-12, atol=0.0)
    assert np.isclose(dfvalues["losvelocity"][0], 0.5e9, rtol=1e-12, atol=0.0)
    assert dfvalues["modelgridindex"][0] is not None
    for norecordrow in (1, 2, 3):
        assert math.isnan(dfvalues["velocity"][norecordrow])
        assert math.isnan(dfvalues["losvelocity"][norecordrow])
        assert dfvalues["modelgridindex"][norecordrow] is None


def test_scan_packets_gives_nan_to_the_thermal_velocity_of_a_packet_with_no_record() -> None:
    """An old packets file holds a thermal emission velocity of zero for a packet with no thermal emission record."""
    _, lzdfpackets = at.packets.scan_packets(
        at.get_path("testdata") / "test-classicmode_3d", packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT"
    )
    dfpackets = lzdfpackets.select("trueem_time", "true_emission_velocity").collect()

    dfnorecord = dfpackets.filter(pl.col("trueem_time") <= 0)
    assert dfnorecord.height > 0
    # a null value passes both is_nan().all() and a comparison, thus the column must hold no null
    assert dfnorecord["true_emission_velocity"].null_count() == 0
    assert dfnorecord["true_emission_velocity"].is_nan().all()
    dfrecord = dfpackets.filter(pl.col("trueem_time") > 0)
    assert dfrecord.height > 0
    assert dfrecord["true_emission_velocity"].null_count() == 0
    assert (dfrecord["true_emission_velocity"] > 0).all()


@mock.patch("artistools.packets.plotlastpacketinteraction.save_figure")
@mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True)
def test_lastpacketinteraction_takes_the_start_of_the_timestep_and_ignores_a_packet_with_no_record(
    mockimshow: mock.MagicMock, mocksavefigure: mock.MagicMock
) -> None:
    """The plot takes a packet at the start of the timestep and not a packet at the end.

    The window of the arrival time included its right edge, thus it took a packet of the next timestep. A packet with
    no thermal emission record gave a large negative weight, which removed the innermost bin.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "testmodel"
    tdays = 300.0
    timestep = at.misc.get_timestep_of_timedays(modelpath, tdays)
    t_start = at.get_timestep_times(modelpath, loc="start")[timestep]
    t_end = at.get_timestep_times(modelpath, loc="end")[timestep]
    emtime_s = 250.0 * at.constants.day_to_s
    # a record at the start in the bin of 0.01 c, a packet at the start with no record in the same bin, and a
    # record at the end in the bin of 0.21 c
    betas = [0.01, 0.0, 0.21]
    dfpackets = pl.DataFrame({
        "t_arrive_d": [t_start, t_start, t_end],
        "e_rf": [1.0e40, 1.0e40, 1.0e40],
        "trueem_posx": [beta * at.constants.C_cm_per_s * emtime_s for beta in betas],
        "trueem_posy": [0.0, 0.0, 0.0],
        "trueem_posz": [beta * at.constants.C_cm_per_s * emtime_s for beta in betas],
        "trueem_time": [emtime_s, -1.0, emtime_s],
    })

    with mock.patch.object(
        plotlastpacketinteraction, "get_reduced_packet_set", return_value=(1, dfpackets.lazy(), 1.0)
    ):
        c_kmps = at.constants.C_cm_per_s / at.constants.km_to_cm
        plotlastpacketinteraction.packets_2d_hist_bin_and_ejecta_vel(
            modelpath,
            tdays,
            -1,
            25,
            0.5 * c_kmps,
            "c",
            ions=None,
            srIItriplet=False,
            logcolorscale=False,
            trueem=True,
            lambdarange=None,
            outputfile=None,
            args=None,
            absorption=False,
        )

    assert mocksavefigure.call_count == 1
    heatmap = mockimshow.call_args.args[1].T
    assert heatmap.count() == 1
    assert heatmap[0, 25] > 0.0


@pytest.mark.parametrize(("absorption", "velocityunit"), [(False, "kmps"), (True, "kmps"), (False, "c")])
def test_lastpacketinteraction_groupby_gives_the_luminosity_of_each_group_in_each_velocity_bin(
    tmp_path: Path, absorption: bool, velocityunit: str, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The group of a packet comes from its emission type, or with --absorption, from its absorption type.

    The groups after -maxseriescount and the codes with no line join Other, and a packet above the bins is in no bin.
    -velocityunit c gives -velocitymax, the bin edges, and the luminosity per unit velocity in units of c. A group with
    no packet in the bins also joins Other. NOT SET with no position took the only series and drew zero in each bin.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    tdays = 7.0
    timestep = at.misc.get_timestep_of_timedays(modelpath, tdays)
    t_start = at.get_timestep_times(modelpath, loc="start")[timestep]
    delta_t_s = (at.get_timestep_times(modelpath, loc="end")[timestep] - t_start) * at.constants.day_to_s
    emtime_s = 6.0 * at.constants.day_to_s
    kmps_per_unit = at.constants.C_cm_per_s / at.constants.km_to_cm if velocityunit == "c" else 1.0
    # rows 0 and 3 of the linelist are lines of Fe IV and Ni IV, and the next code is a free-free emission. The last
    # packet has no record of its position, and its NOT SET group has more energy than the Fe IV group
    typecodes = [0, 0, 0, 0, 3, -9999999, -9999000]
    velocities_kmps = [2000.0, 3000.0, 7000.0, 12000.0, 7000.0, 7000.0, 0.0]
    energies = [2e40, 2e40, 2e40, 2e40, 1e40, 0.5e40, 9e40]
    othertypecodes = [957] * len(typecodes)
    dfpackets = pl.DataFrame({
        "t_arrive_d": [t_start] * len(typecodes),
        "e_rf": energies,
        "emissiontype": othertypecodes if absorption else typecodes,
        "absorption_type": typecodes if absorption else othertypecodes,
        "em_posx": [velocity * at.constants.km_to_cm * emtime_s for velocity in velocities_kmps],
        "em_posy": [0.0] * len(typecodes),
        "em_posz": [0.0] * len(typecodes),
        "em_time": [emtime_s] * (len(typecodes) - 1) + [-1.0],
    })

    with mock.patch.object(
        plotlastpacketinteraction, "get_reduced_packet_set", return_value=(1, dfpackets.lazy(), 1.0)
    ):
        plotlastpacketinteraction.main(
            argsraw=[],
            plotinvalidpart=True,
            modelpath=modelpath,
            timedays=tdays,
            groupby="ion",
            absorption=absorption,
            maxseriescount=1,
            nvelocitybins=2,
            velocitymax_kmps=10000.0,
            velocityunit=velocityunit,
            outputfile=tmp_path / "profile.pdf",
            write_data=True,
        )

    dftable = read_data_file(tmp_path / "profile.txt")
    luminosities = {
        (label, round(xlower * kmps_per_unit)): (luminosity, perunitvelocity)
        for label, xlower, luminosity, perunitvelocity in dftable.select(
            "label",
            f"x_lower_{velocityunit}",
            "luminosity_erg_per_s",
            f"luminosity_per_velocity_erg_per_s_per_{velocityunit}",
        ).iter_rows()
    }
    # an absorption has no free-free label, thus its code joins Other as a code with no line
    expected = {("Fe IV", 0): 4e40, ("Fe IV", 5000): 2e40, ("Other", 0): 0.0, ("Other", 5000): 1.5e40}
    assert luminosities.keys() == expected.keys()
    binwidth = 5000.0 / kmps_per_unit
    for key, energy in expected.items():
        luminosity, perunitvelocity = luminosities[key]
        assert np.isclose(luminosity, energy / delta_t_s, rtol=1e-6, atol=0.0), key
        assert np.isclose(perunitvelocity, energy / delta_t_s / binwidth, rtol=1e-6, atol=0.0), key

    # the summary includes the Fe IV packet above the bins, and the median velocity takes the energy as the weight
    dfsummary = read_data_file(tmp_path / "profile_summary.txt")
    summaryrows = {row["label"]: row for row in dfsummary.iter_rows(named=True)}
    assert summaryrows.keys() == {"Fe IV", "Other"}
    assert summaryrows["Fe IV"]["packetcount"] == 4
    assert np.isclose(summaryrows["Fe IV"]["luminosity_erg_per_s"], 8e40 / delta_t_s, rtol=1e-6, atol=0.0)
    for label, median_kmps in (("Fe IV", 3000.0), ("Other", 7000.0)):
        median = summaryrows[label][f"velocity_luminosity_q50_{velocityunit}"]
        assert np.isclose(median * kmps_per_unit, median_kmps, rtol=1e-6, atol=0.0), label


def test_lastinteractionvelocity_gives_the_energy_weighted_quartiles_of_each_timestep(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """A packet takes the timestep of its arrival time, and the quartiles take the energy of each packet as the weight.

    A packet at the start time of a timestep is in that timestep, and a packet outside the wavelength range is in no
    timestep. A packet with no record of the position has a NaN velocity, and the strict cast of its bin stopped the
    command. It is in no bin now.
    """
    from artistools.packets import plotlastinteractionvelocity

    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    tstarts = at.get_timestep_times(modelpath, loc="start")
    emtime_s = 5.0 * at.constants.day_to_s
    # three packets are in timestep 25, one is at the start time of timestep 26, and one is outside the wavelength
    # range. The last packet is in timestep 25 and has no record of the position
    velocities_kmps = [3000.0, 5000.0, 9000.0, 7000.0, 1000.0, 0.0]
    lambdas_angstroms = [7500.0, 7500.0, 7500.0, 7500.0, 9000.0, 7500.0]
    dfpackets = pl.DataFrame({
        "t_arrive_d": [tstarts[25] + 0.01] * 3 + [tstarts[26]] + [tstarts[25] + 0.01] * 2,
        "e_rf": [1e40, 2e40, 1e40, 1e40, 5e40, 9e40],
        "nu_rf": [at.constants.c_ang_per_s / lambda_angstroms for lambda_angstroms in lambdas_angstroms],
        "em_posx": [velocity * at.constants.km_to_cm * emtime_s for velocity in velocities_kmps],
        "em_posy": [0.0] * 6,
        "em_posz": [0.0] * 6,
        "em_time": [emtime_s] * 5 + [-1.0],
        "costhetabin": [9] * 6,
        "dirbin": [90] * 6,
    })
    with mock.patch.object(plotlastinteractionvelocity, "scan_packets", return_value=(1, dfpackets.lazy())):
        plotlastinteractionvelocity.main(
            argsraw=[],
            plotinvalidpart=True,
            modelpath=[modelpath],
            xmin=7000.0,
            xmax=8000.0,
            velocityunit="kmps",
            outputfile=tmp_path / "velocity.pdf",
            write_data=True,
        )

    dftable = read_data_file(tmp_path / "velocity.txt")
    assert dftable["timestep"].to_list() == [25, 26]
    assert dftable["packetcount"].to_list() == [3, 1]
    # sqrt(sum(e^2)) / sum(e) of the energies 1, 2, and 1, and of one packet
    assert np.allclose(
        dftable["luminosity_relative_error"].to_numpy(), [math.sqrt(6.0) / 4.0, 1.0], rtol=1e-9, atol=0.0
    )
    # the histogram bins of the test model are 25 km/s wide, and each velocity takes the centre of its bin
    expected = {
        "velocity_luminosity_q25_kmps": [3000.0, 7000.0],
        "velocity_luminosity_q50_kmps": [5000.0, 7000.0],
        "velocity_luminosity_q75_kmps": [5000.0, 7000.0],
    }
    for column, values in expected.items():
        assert np.allclose(dftable[column].to_numpy(), values, rtol=0.0, atol=15.0), column


def test_lastpacketinteraction_groupby_reads_the_escaped_packets_of_the_window(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """With real packets, the groups hold the luminosity of the escaped r-packets of the wavelength range and timestep.

    The luminosity is the packet energy divided by the width of the timestep and the number of MPI ranks.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    timestep = at.misc.get_timestep_of_timedays(modelpath, 6.5)
    t_start = at.get_timestep_times(modelpath, loc="start")[timestep]
    t_end = at.get_timestep_times(modelpath, loc="end")[timestep]
    plotlastpacketinteraction.main(
        argsraw=[],
        modelpath=modelpath,
        timedays=6.5,
        groupby="ion",
        xmin=3500.0,
        xmax=4500.0,
        outputfile=tmp_path / "groups.pdf",
        write_data=True,
    )
    dfsummary = read_data_file(tmp_path / "groups_summary.txt")

    nprocs, lzpackets = at.packets.scan_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfwindow = (
        lzpackets
        .filter(
            pl.col("t_arrive_d").is_between(t_start, t_end, closed="left"),
            (at.constants.c_ang_per_s / pl.col("nu_rf")).is_between(3500.0, 4500.0),
        )
        .select("e_rf")
        .collect()
    )
    luminosity = float(dfwindow["e_rf"].sum()) / ((t_end - t_start) * at.constants.day_to_s * nprocs)
    assert dfsummary["packetcount"].sum() == dfwindow.height > 0
    assert np.isclose(float(dfsummary["luminosity_erg_per_s"].sum()), luminosity, rtol=1e-6, atol=0.0)


def test_lastpacketinteraction_groupby_bins_the_line_of_sight_velocity(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """-velocitykind lineofsight bins (em_pos . dir) / em_time from minus to plus the last edge.

    The plot had only the radial velocity, which cannot separate the blue side and the red side of a P Cygni profile.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    timestep = at.misc.get_timestep_of_timedays(modelpath, 6.5)
    t_start = at.get_timestep_times(modelpath, loc="start")[timestep]
    t_end = at.get_timestep_times(modelpath, loc="end")[timestep]
    plotlastpacketinteraction.main(
        argsraw=[],
        modelpath=modelpath,
        timedays=6.5,
        groupby="element",
        nvelocitybins=5,
        velocitykind="lineofsight",
        outputfile=tmp_path / "groups.pdf",
        write_data=True,
    )
    dftable = read_data_file(tmp_path / "groups.txt")
    dfbins = (
        dftable.group_by("x_lower_kmps", "x_upper_kmps").agg(pl.col("luminosity_erg_per_s").sum()).sort("x_lower_kmps")
    )
    edges = [*dfbins["x_lower_kmps"].to_list(), float(dfbins["x_upper_kmps"][-1])]
    assert len(edges) == 11
    assert np.isclose(edges[0], -edges[-1], rtol=1e-9, atol=0.0)

    nprocs, lzpackets = at.packets.scan_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfwindow = (
        lzpackets
        .filter(pl.col("t_arrive_d").is_between(t_start, t_end, closed="left"))
        .select(
            pl.col("e_rf").cast(pl.Float64),
            velocity_kmps=(
                pl.col("em_posx") * pl.col("dirx")
                + pl.col("em_posy") * pl.col("diry")
                + pl.col("em_posz") * pl.col("dirz")
            )
            / (pl.col("em_time") * at.constants.km_to_cm),
        )
        .collect()
    )
    velocities = dfwindow["velocity_kmps"].to_numpy()
    assert velocities.min() < 0.0 < velocities.max()
    luminosityfactor = 1.0 / ((t_end - t_start) * at.constants.day_to_s * nprocs)
    expected, _ = np.histogram(velocities, bins=edges, weights=dfwindow["e_rf"].to_numpy() * luminosityfactor)
    assert np.allclose(dfbins["luminosity_erg_per_s"].to_numpy(), expected, rtol=1e-6, atol=0.0)


def test_a_position_outside_the_grid_gets_no_cell() -> None:
    """A packet outside the 3D grid must get no cell, and a packet inside must get its own cell.

    The index truncated toward zero, and the sum of the axis indices had no range check. Thus x = vmax plus
    a tenth of a cell gave the cell at x = 0 of the next row, and the packet took that cell's Ye, Te and nne.
    """
    from artistools.packets.core import get_modelgridindex_expr

    ncoordgrid = 4
    vmax = 1e9
    t_model_days = 1.0
    vwidth = 2 * vmax / ncoordgrid
    modelmeta = {
        "dimensions": 3,
        "t_model_init_days": t_model_days,
        "vmax_cmps": vmax,
        "wid_init": vwidth * t_model_days * 86400.0,
        "ncoordgridx": ncoordgrid,
        "ncoordgridy": ncoordgrid,
        "ncoordgridz": ncoordgrid,
    }
    t_s = 86400.0
    # the first packet is inside cell (x=1, y=0, z=0), the other two are outside the grid on -x and +x
    velocities_x = [-vmax + 1.5 * vwidth, -vmax - 0.1 * vwidth, vmax + 0.1 * vwidth]
    dfpackets = pl.DataFrame({
        "em_posx": [vx * t_s for vx in velocities_x],
        "em_posy": [(-vmax + 0.5 * vwidth) * t_s] * 3,
        "em_posz": [(-vmax + 0.5 * vwidth) * t_s] * 3,
        "em_time": [t_s] * 3,
    })
    cells = dfpackets.select(get_modelgridindex_expr("em", modelmeta, pl.LazyFrame()).alias("mgi"))["mgi"].to_list()
    assert cells == [1, None, None]


def test_a_1d_velocity_at_the_edges_of_the_grid_gets_the_right_cell() -> None:
    """A velocity of zero is in the first cell, and a velocity at or above the outer edge gets no cell.

    A cut alone gave the index -1 to a velocity of zero. A velocity above the outer edge got the index of a cell
    that does not exist, and the 3D branch gives null for it.
    """
    from artistools.packets.core import get_modelgridindex_from_velocity_expr

    dfmodel = pl.LazyFrame({"vel_r_max_kmps": [1.0, 2.0, 3.0]})
    km_to_cm = 1e5
    velocities = [0.0, 0.5 * km_to_cm, 1.0 * km_to_cm, 2.5 * km_to_cm, 3.0 * km_to_cm, 3.5 * km_to_cm, float("nan")]
    cells = (
        pl
        .DataFrame({"v": velocities})
        .select(get_modelgridindex_from_velocity_expr(pl.col("v"), dfmodel).alias("mgi"))["mgi"]
        .to_list()
    )
    assert cells == [0, 0, 1, 2, None, None, None]


def test_scan_packets_gives_one_name_and_type_to_the_flag_of_the_particle_origin(tmp_path: Path) -> None:
    """An older ARTIS writes originated_from_positron, and the current ARTIS writes originated_from_particlenotgamma.

    The cache keeps the name and the type of its text file. The text reader gives the older flag the type Boolean,
    and the current flag stays an integer. The scan of a batch of each format stopped with an InvalidOperationError,
    because the conversion to Boolean came after the concatenation of the batches.
    """
    flagvalues_of_format: list[tuple[str, list[bool] | list[int]]] = [
        ("originated_from_positron", [True, False]),
        ("originated_from_particlenotgamma", [1, 0]),
    ]
    cachepaths = []
    for batchindex, (columnname, values) in enumerate(flagvalues_of_format):
        cachepaths.append(tmp_path / f"packetsbatch{batchindex:02d}_{batchindex:04d}_{batchindex:04d}.out.parquet")
        pl.DataFrame({
            "type_id": [32, 32],
            "dirx": [1.0, 0.0],
            "diry": [0.0, 1.0],
            "dirz": [0.0, 0.0],
            columnname: values,
        }).write_parquet(cachepaths[-1])

    with mock.patch("artistools.packets.core.get_packets_batch_parquet_paths", return_value=(2, cachepaths)):
        _, dfpackets = at.packets.scan_packets(tmp_path)

    dfflag = dfpackets.select("originated_from_particlenotgamma").collect()
    pltest.assert_frame_equal(dfflag, pl.DataFrame({"originated_from_particlenotgamma": [True, False, True, False]}))


def test_scan_packets_reads_a_batch_of_each_cache_version(tmp_path: Path) -> None:
    """A model can hold a batch of cache version 1 and a batch of version 2, and both give the same direction bins.

    Version 2 stores only dirbin_costheta10_phi10 and phi, and version 1 stores dirbin, the angles, and the bins. A
    scan of files with two schemas stops with a SchemaError, thus each batch gets the columns of version 2 before the
    concatenation.
    """
    import shutil

    from artistools.packets.core import get_packets_rankbatch_parquetfile

    sourcedir = at.get_path("testdata") / "test-classicmode_3d" / "packets"
    for rank in (0, 1):
        shutil.copy(sourcedir / f"packets00_{rank:04d}.out.zst", tmp_path)
    newcache = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)
    assert pl.read_parquet_metadata(newcache)["cacheversion"] == "2"
    newcolumns = list(pl.read_parquet_schema(newcache))
    assert {"dirbin_costheta10_phi10", "phi"} <= set(newcolumns)
    assert not {"dirbin", "costhetabin", "phibin", "costheta", "cosphi", "testphi"} & set(newcolumns)

    dfnew = pl.read_parquet(newcache)
    oldcache = tmp_path / "packetsbatch01_0002_0003.out.parquet.tmp"
    dfnew.rename({"dirbin_costheta10_phi10": "dirbin"}).with_columns(
        costhetabin=pl.col("dirbin") // 10,
        phibin=pl.col("dirbin") % 10,
        costheta=pl.lit(2.0, dtype=pl.Float32),
        cosphi=pl.lit(2.0, dtype=pl.Float32),
        testphi=pl.lit(0.0, dtype=pl.Float32),
    ).write_parquet(oldcache)

    with mock.patch("artistools.packets.core.get_packets_batch_parquet_paths", return_value=(4, [newcache, oldcache])):
        _, dfpackets = at.packets.scan_packets(tmp_path)
    directioncolumns = ["dirbin", "costhetabin", "phibin", "costheta", "phi"]
    dfdirections = dfpackets.select(directioncolumns).collect()

    nrows = dfnew.height
    pltest.assert_frame_equal(dfdirections.head(nrows), dfdirections.tail(nrows))
    expected = at.packets.add_packet_directions_lazypolars(
        at.packets.bin_packet_directions_polars(dfnew.drop("dirbin_costheta10_phi10", "phi"))
    )
    pltest.assert_frame_equal(dfdirections.head(nrows), expected.select(directioncolumns).collect())


def test_direction_bins_of_other_bin_counts_come_from_the_directions() -> None:
    """The cache column of 10 cos θ bins and 10 ϕ bins gives only those counts, and other counts use the directions."""
    dirx, diry, dirz = 0.3, -0.4, 0.5
    dfpackets = pl.LazyFrame({"dirx": [dirx], "diry": [diry], "dirz": [dirz], "dirbin_costheta10_phi10": [7]})

    def get_dirbin(ncosthetabins: int, nphibins: int) -> int:
        binned = at.packets.bin_packet_directions_polars(dfpackets, nphibins=nphibins, ncosthetabins=ncosthetabins)
        return int(binned.select("dirbin").collect().item())

    assert get_dirbin(10, 10) == 7
    assert get_dirbin(4, 8) == get_reference_dirbin(dirx, diry, dirz, nphibins=8, ncosthetabins=4)


def test_a_stored_phi_gives_the_phi_bins() -> None:
    """A frame that holds phi gives the phi bins from that column, because an atan2 of each packet is slow."""
    dfpackets = pl.LazyFrame(
        {"dirx": [1.0], "diry": [0.0], "dirz": [0.0], "phi": [4.0]}, schema_overrides={"phi": pl.Float32}
    )
    binned = at.packets.bin_packet_directions_polars(
        dfpackets, nphibins=8, ncosthetabins=4, phibintype="phibinmonotonicasc"
    ).collect()

    assert binned["phibinmonotonicasc"].item() == int(4.0 / (2 * math.pi) * 8)
    assert at.packets.add_packet_directions_lazypolars(dfpackets).collect()["phi"].item() == pytest.approx(4.0)


def test_lastpacketinteraction_main_writes_the_plot_to_the_output_folder(tmp_path: Path) -> None:
    """The command takes -timedays as the other commands do, and -o gives the folder or the file of the plot.

    The command took only its own -tdays, and it always wrote its file to the working folder.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "testmodel"
    plotlastpacketinteraction.main(argsraw=["-modelpath", str(modelpath), "-timedays", "300", "-o", str(tmp_path)])
    assert any(tmp_path.glob("testmodel_allions_t_arrive_d_*_ts54_into_dirbin-1.pdf"))

    # -timestep selects the same arrival time bin, as in the other commands
    timestepfolder = tmp_path / "timestep"
    plotlastpacketinteraction.main(argsraw=[str(modelpath), "-timestep", "54", "-o", str(timestepfolder)])
    assert any(timestepfolder.glob("testmodel_allions_t_arrive_d_*_ts54_into_dirbin-1.pdf"))

    # -tdays is the old spelling, and -o can also name the file
    outputfile = tmp_path / "lastinteraction.pdf"
    plotlastpacketinteraction.main(argsraw=["-modelpath", str(modelpath), "-tdays", "300", "-o", str(outputfile)])
    assert outputfile.is_file()


@mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True)
@pytest.mark.parametrize("logcolorscale", [False, True], ids=["linear", "log"])
def test_lastpacketinteraction_write_data_gives_the_values_of_the_image(
    mockimshow: mock.MagicMock, logcolorscale: bool, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Each row of the table gives the edges of one bin and the value that the image shows in that bin."""
    from artistools.packets import plotlastpacketinteraction

    outputfile = tmp_path / "lastinteraction.pdf"
    plotlastpacketinteraction.main(
        argsraw=[],
        modelpath=at.get_path("testdata") / "testmodel",
        timedays=300,
        outputfile=outputfile,
        logcolorscale=logcolorscale,
        velocityunit="c",
        velocitymax_kmps=0.5 * at.constants.C_cm_per_s / at.constants.km_to_cm,
        write_data=True,
    )

    image = mockimshow.call_args.args[1]
    dftable = read_data_file(outputfile.with_suffix(".txt"))
    valuecolumn = (
        "log10_escaped_luminosity_erg_per_s_per_cm3" if logcolorscale else "escaped_luminosity_erg_per_s_per_cm3"
    )
    assert dftable.height == image.size
    assert dftable[valuecolumn].null_count() == int(np.ma.count_masked(image))
    assert dftable[valuecolumn].null_count() < dftable.height

    # the image has one row for each v_z bin of 0.02 c from -0.5 c, and one column for each v_cyl bin of 0.02 c
    rowindices = np.rint((dftable["v_z_min_c"].to_numpy() + 0.5) / 0.02).astype(int)
    columnindices = np.rint(dftable["v_cyl_min_c"].to_numpy() / 0.02).astype(int)
    assert np.allclose(dftable["v_cyl_max_c"] - dftable["v_cyl_min_c"], 0.02, rtol=1e-9, atol=0.0)
    assert np.allclose(dftable["v_z_max_c"] - dftable["v_z_min_c"], 0.02, rtol=1e-9, atol=0.0)
    imagevalues = np.ma.filled(image[rowindices, columnindices].astype(float), np.nan)
    assert np.allclose(dftable[valuecolumn].to_numpy(), imagevalues, rtol=1e-12, atol=0.0, equal_nan=True)

    if logcolorscale:
        # the log scale took the natural logarithm, and the colour bar and the table named it log
        plotlastpacketinteraction.main(
            argsraw=[],
            modelpath=at.get_path("testdata") / "testmodel",
            timedays=300,
            outputfile=tmp_path / "linear.pdf",
            velocityunit="c",
            velocitymax_kmps=0.5 * at.constants.C_cm_per_s / at.constants.km_to_cm,
            write_data=True,
        )
        dflinear = read_data_file(tmp_path / "linear.txt")
        assert np.allclose(
            10 ** dftable[valuecolumn].to_numpy(),
            dflinear["escaped_luminosity_erg_per_s_per_cm3"].to_numpy(),
            rtol=1e-9,
            atol=0.0,
            equal_nan=True,
        )


def test_lastpacketinteraction_image_takes_each_escaped_packet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The 2D image takes the escaped r-packets that no line absorbed, too.

    The image took only the packets with a last absorption by a line.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    timedays = 6.5
    timestep = at.get_timestep_of_timedays(modelpath, timedays)
    tstart = at.get_timestep_times(modelpath, loc="start")[timestep]
    tend = at.get_timestep_times(modelpath, loc="end")[timestep]
    _, lzpackets = at.packets.scan_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfarrived = lzpackets.filter(pl.col("t_arrive_d").is_between(tstart, tend, closed="left")).collect()
    assert dfarrived.filter(pl.col("absorption_type") < 0).height > 0

    plotlastpacketinteraction.main(
        argsraw=[], modelpath=modelpath, timedays=timedays, outputfile=tmp_path / "lastinteraction.pdf"
    )
    assert (
        f"{dfarrived.height} escaped r-packets of the direction bin arrive in the timestep\n" in capsys.readouterr().out
    )


def test_packet_commands_check_the_valid_range_of_the_arrival_times(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Plotlastpacketinteraction stops for a timestep after the valid range, and plotlastinteractionvelocity skips it.

    The two commands took any timestep, and the light of a part of the ejecta did not reach the observer in some of
    them. --plotinvalidpart keeps them, as in plotspectra and plotlightcurves.
    """
    from artistools.packets import plotlastinteractionvelocity
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    _, _, validend_days = at.misc.get_escaped_arrivalrange(modelpath)
    assert validend_days is not None
    with pytest.raises(SystemExit):
        plotlastpacketinteraction.main(argsraw=[str(modelpath), "-t", "7", "-o", str(tmp_path / "image.pdf")])
    assert "after the end of the valid range" in capsys.readouterr().err
    plotlastpacketinteraction.main(
        argsraw=[str(modelpath), "-t", "7", "--plotinvalidpart", "-o", str(tmp_path / "image.pdf")]
    )

    for plotinvalidpart in (False, True):
        outputfile = tmp_path / f"velocity_{plotinvalidpart}.pdf"
        plotlastinteractionvelocity.main(
            argsraw=[str(modelpath), "-lambdamin", "3000", "-lambdamax", "8000", "-o", str(outputfile)],
            plotinvalidpart=plotinvalidpart,
            write_data=True,
        )
        dftable = read_data_file(outputfile.with_suffix(".txt"))
        timestepends = at.get_timestep_times(modelpath, loc="end")
        latestend_days = max(timestepends[timestep] for timestep in dftable["timestep"].to_list())
        assert (latestend_days > validend_days) == plotinvalidpart


def test_lastpacketinteraction_dirbin_takes_one_direction_bin_as_in_plotspectra(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """-dirbin N takes the one direction bin N, and --average_over_phi_angle takes all the phi bins of its costheta bin.

    The packet commands always took the costheta bin, thus -dirbin 10 selected other directions than in plotspectra.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    timestep = at.misc.get_timestep_of_timedays(modelpath, 6.5)
    t_start = at.get_timestep_times(modelpath, loc="start")[timestep]
    t_end = at.get_timestep_times(modelpath, loc="end")[timestep]
    _, lzpackets = at.packets.scan_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfarrived = (
        lzpackets
        .filter(pl.col("t_arrive_d").is_between(t_start, t_end, closed="left"))
        .select("dirbin", "costhetabin")
        .collect()
    )
    cases: list[tuple[list[str], int]] = [
        ([], dfarrived.filter(pl.col("dirbin") == 50).height),
        (["--average_over_phi_angle"], dfarrived.filter(pl.col("costhetabin") == 5).height),
    ]
    for extraargs, expectedcount in cases:
        outputfile = tmp_path / f"groups{len(extraargs)}.pdf"
        plotlastpacketinteraction.main(
            argsraw=[str(modelpath), "-t", "6.5", "-groupby", "element", "-dirbin", "50", *extraargs],
            outputfile=outputfile,
            write_data=True,
        )
        dfsummary = read_data_file(outputfile.with_name(f"{outputfile.stem}_summary.txt"))
        assert int(dfsummary["packetcount"].sum()) == expectedcount > 0


def make_filterpackets_model(tmp_path: Path) -> Path:
    """Copy the packets files of the 3D test model, with the file of rank 1 as an uncompressed file."""
    sourcemodel = at.get_path("testdata") / "test-classicmode_3d"
    modelpath = tmp_path / "model"
    (modelpath / "packets").mkdir(parents=True)
    (modelpath / "input.txt").write_bytes((sourcemodel / "input.txt").read_bytes())
    (modelpath / "packets" / "packets00_0000.out.zst").write_bytes(
        (sourcemodel / "packets" / "packets00_0000.out.zst").read_bytes()
    )
    with at.zopen(sourcemodel / "packets" / "packets00_0001.out.zst", mode="rb") as fsource:
        (modelpath / "packets" / "packets00_0001.out").write_bytes(fsource.read())
    return modelpath


def get_packet_types(packetsfile: Path, modelpath: Path) -> pl.DataFrame:
    columns = at.packets.core.get_packets_text_columns(packetsfile, modelpath)
    return at.packets.core.readfile_text(packetsfile, columns).select("type_id", "escape_type_id")


def test_filterpackets_keeps_only_the_escaped_rpkts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    modelpath = make_filterpackets_model(tmp_path)
    packetsfolder = modelpath / "packets"
    # a small frame size makes several zstd frames in each output file
    monkeypatch.setattr(at.packets.filterpackets, "LINES_PER_FRAME", 1000)
    parquetcache = at.packets.core.get_packets_rankbatch_parquetpath(modelpath, (0, 1), 0, virtual=False)
    parquetcache.write_bytes(b"")
    # the zstd file that replaces the uncompressed file must keep its mode
    (packetsfolder / "packets00_0001.out").chmod(0o640)
    # a stale copy of a rank still holds the removed packets
    (packetsfolder / "packets00_0001.out.gz").write_bytes(b"")
    escaped_rpkt = (pl.col("type_id") == at.packets.core.type_ids["TYPE_ESCAPE"]) & (
        pl.col("escape_type_id") == at.packets.core.type_ids["TYPE_RPKT"]
    )
    inputfiles = [packetsfolder / "packets00_0000.out.zst", packetsfolder / "packets00_0001.out"]
    expected_counts = [get_packet_types(path, modelpath).filter(escaped_rpkt).height for path in inputfiles]
    headerline = at.zopen(inputfiles[0]).readline()

    at.packets.filterpackets.main(argsraw=[str(modelpath), "-f"])

    assert sorted(path.name for path in packetsfolder.iterdir()) == ["packets00_0000.out.zst", "packets00_0001.out.zst"]
    assert stat.S_IMODE((packetsfolder / "packets00_0001.out.zst").stat().st_mode) == 0o640
    for rank, expected_count in enumerate(expected_counts):
        outfile = packetsfolder / f"packets00_{rank:04d}.out.zst"
        dfpackettypes = get_packet_types(outfile, modelpath)
        assert 0 < dfpackettypes.height == expected_count
        assert dfpackettypes.filter(~escaped_rpkt).is_empty()
        assert at.zopen(outfile).readline() == headerline


def test_filterpackets_dry_run_changes_no_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    modelpath = make_filterpackets_model(tmp_path)
    packetsfolder = modelpath / "packets"
    contents_before = {path.name: path.read_bytes() for path in packetsfolder.iterdir()}

    at.packets.filterpackets.main(argsraw=[str(modelpath), "--dry-run"])

    assert {path.name: path.read_bytes() for path in packetsfolder.iterdir()} == contents_before
    assert "In 2 of 2 packets files, the filter removes" in capsys.readouterr().out


def test_filterpackets_keeps_a_file_with_no_escaped_rpkt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    modelpath = make_filterpackets_model(tmp_path)
    packetsfile = modelpath / "packets" / "packets00_0001.out"
    columns = at.packets.core.get_packets_text_columns(packetsfile, modelpath)
    type_id_index = columns.index("type_id")
    escape_type_id_index = columns.index("escape_type_id")
    lines = packetsfile.read_text(encoding="utf-8").splitlines(keepends=True)
    packetsfile.write_text(
        "".join(
            line
            for line in lines
            if line.startswith("#")
            or (line.split()[type_id_index], line.split()[escape_type_id_index])
            != at.packets.filterpackets.ESCAPED_RPKT_TYPES
        ),
        encoding="utf-8",
    )
    contents_before = packetsfile.read_bytes()

    at.packets.filterpackets.main(argsraw=[str(modelpath), "-f"])

    assert packetsfile.read_bytes() == contents_before
    assert f"{packetsfile}: no escaped r-packets, thus the filter keeps the file" in capsys.readouterr().out


@pytest.mark.benchmark
def test_plotspherical(tmp_path: Path) -> None:
    at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, outputfile=tmp_path)

    assert [path.name for path in tmp_path.glob("plotspherical_*.pdf")] == ["plotspherical_261.55-333.82d.pdf"]


def test_plotspherical_gaussian_filter(tmp_path: Path) -> None:
    """-gaussian_sigma must reach smooth_direction_map, which takes the width in phi bins and not in degrees."""
    from artistools.packets.plotspherical import smooth_direction_map

    with mock.patch.object(
        at.packets.plotspherical, "smooth_direction_map", side_effect=smooth_direction_map
    ) as mocksmooth:
        at.packets.plotspherical.main(argsraw=[], modelpath=modelpath, gaussian_sigma=20, outputfile=tmp_path)

    # one map for each of the three default plot variables, on the default grid of 64 phi bins
    assert mocksmooth.call_count == 3
    for callargs in mocksmooth.call_args_list:
        _, sigma_bins, nphibins = callargs.args
        assert nphibins == 64
        assert np.isclose(sigma_bins, 20 / 360 * 64)


def test_plotspherical_one_pass_matches_one_pass_per_time_range() -> None:
    """The direction maps of several time ranges in one pass must equal the map of each range on its own."""
    from artistools.packets.plotspherical import bin_packets_by_direction

    nprocs_read, dfpackets = at.packets.scan_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfpackets = at.packets.add_derived_columns_lazy(dfpackets, modelpath=modelpath)
    tstarts = at.get_timestep_times(modelpath, loc="start")
    tends = at.get_timestep_times(modelpath, loc="end")
    # the last range repeats the first one, as every frame does when the observer never gets light from the full ejecta
    timeranges = [(tstarts[ts], tends[ts]) for ts in (60, 61, 62, 60)]
    plotvars = ["luminosity", "emvelocityoverc", "emlosvelocityoverc", "emvelocityoverc_sigma"]

    dfonepass, _ = bin_packets_by_direction(
        modelpath, dfpackets, nprocs_read, timeranges, 8, 4, plotvars, dfestimators=None
    )
    assert dfonepass.height == 4 * 8 * 4
    assert dfonepass["count"].sum() > 0

    for timebin, timerange in enumerate(timeranges):
        dfsingle, _ = bin_packets_by_direction(
            modelpath, dfpackets, nprocs_read, [timerange], 8, 4, plotvars, dfestimators=None
        )
        pltest.assert_frame_equal(
            dfonepass.filter(pl.col("timebin") == timebin).with_columns(timebin=pl.lit(0, dtype=pl.Int32)), dfsingle
        )


def test_plotspherical_smoothing_wraps_phi_and_crosses_each_pole() -> None:
    """The smoothing must blend no pole with the opposite pole.

    The filter wrapped the cos theta axis as it wraps phi. Thus the row at the south pole took the light of
    the row at the north pole.
    """
    from artistools.packets.plotspherical import smooth_direction_map

    ncosthetabins, nphibins = 8, 16
    data = np.zeros((ncosthetabins, nphibins))
    data[-1, 0] = 1.0  # the north polar cap, at phi bin 0

    smoothed = smooth_direction_map(data, 1.0, nphibins)

    assert np.allclose(smoothed[0], 0.0), "the south polar cap must stay dark"
    # phi wraps, thus the last phi bin is a neighbour of phi bin 0
    assert smoothed[-1, -1] > 0.0
    # the bin on the other side of the north pole is at phi + pi in the same row
    assert smoothed[-1, nphibins // 2] > smoothed[-1, nphibins // 4]


@mock.patch.object(mplax.Axes, "pcolormesh", side_effect=mplax.Axes.pcolormesh, autospec=True)
def test_plotspherical_puts_phi_zero_at_the_centre(mockpcolormesh: mock.MagicMock) -> None:
    """The direction phi = 0 (+X) must be at the longitude zero, which is the centre of the map.

    The longitude of the map started at -pi with phi bin 0, thus +X was at the edge and -X at the centre.
    """
    from artistools.packets.plotspherical import plot_spherical

    ncosthetabins, nphibins = 4, 8
    isphibinzero = [phibin == 0 for _ in range(ncosthetabins) for phibin in range(nphibins)]
    dirbins = pl.DataFrame({"count": [1] * len(isphibinzero), "luminosity": [float(x) for x in isphibinzero]})

    fig, _ = plot_spherical(dirbins, ["luminosity"], nphibins=nphibins, ncosthetabins=ncosthetabins)
    plt.close(fig)

    # the colour bar draws a mesh of its own after the map
    _, meshgrid_phi, _, data = mockpcolormesh.call_args_list[0].args
    (litcolumn,) = np.flatnonzero(np.asarray(data)[0])
    assert np.allclose(meshgrid_phi[0, litcolumn : litcolumn + 2], [0.0, 2 * np.pi / nphibins])


@pytest.mark.parametrize("nphibins", [1, 5, 8])
def test_plotspherical_map_columns_cover_each_longitude_with_the_correct_phi_bin(nphibins: int) -> None:
    """Each column of the map must show the phi bin that holds its longitude, from -pi to +pi.

    For an odd -nphibins, the map ran from -0.8 pi to 1.2 pi for five bins, thus a strip at -pi stayed blank.
    """
    from artistools.packets.plotspherical import get_map_columns

    longitude_edges, column_phibins = get_map_columns(nphibins)

    assert len(longitude_edges) == len(column_phibins) + 1
    assert np.isclose(longitude_edges[0], -np.pi)
    assert np.isclose(longitude_edges[-1], np.pi)
    assert np.all(np.diff(longitude_edges) > 0.0)
    assert np.any(np.isclose(longitude_edges, 0.0)), "phi = 0 must be an edge at the centre of the map"
    column_centre_phi = np.mod(0.5 * (longitude_edges[:-1] + longitude_edges[1:]), 2 * np.pi)
    assert np.array_equal(np.floor(column_centre_phi / (2 * np.pi) * nphibins).astype(int), column_phibins)


def test_plotspherical_smoothing_with_odd_phi_bins_crosses_the_pole_at_phi_plus_pi() -> None:
    """For an odd -nphibins, the row beyond a pole must take the two bins on each side of phi + pi."""
    from artistools.packets.plotspherical import get_rows_across_pole

    rows = np.arange(5, dtype=np.float64).reshape((1, 5))

    # the centre of bin 0 is at 0.2 pi, thus phi + pi is at 1.2 pi, which is the edge of bins 2 and 3
    assert np.allclose(get_rows_across_pole(rows, 5)[0], [2.5, 3.5, 2.0, 0.5, 1.5])


def test_plotspherical_gif(tmp_path: Path) -> None:
    """The gif holds one frame for each timestep in the time range, in the order of time.

    The frames and the gif go in a folder of this test alone. A pytest run clears the test output folder when it
    starts. Thus a second run deleted the frames of this test before the gif read them.
    """
    from PIL import Image

    # the timesteps 0, 1, and 2 end before 253 d. Three frames cover the code of the gif
    at.packets.plotspherical.main(
        argsraw=[], modelpath=modelpath, makegif=True, timemin=250, timemax=253, outputfile=tmp_path
    )

    framenames = sorted(path.name for path in tmp_path.glob("plotspherical_*.png"))
    assert framenames == [
        "plotspherical_250.00-250.84d.png",
        "plotspherical_250.84-251.69d.png",
        "plotspherical_251.69-252.54d.png",
    ]
    with Image.open(tmp_path / "sphericalplot.gif") as gif:
        assert getattr(gif, "n_frames", 1) == 3


@mock.patch.object(mplax.Axes, "pcolormesh", side_effect=mplax.Axes.pcolormesh, autospec=True)
def test_plotspherical_write_data_holds_the_drawn_maps(
    mockpcolormesh: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The table of --write_data must give each pixel of each map after the smoothing, at the drawn coordinates.

    The longitude of the map does not follow the ascending phi bins, and --phireverse changes its sign. Thus the
    table must give the edges that the axes received, and not the edges of the phi bins.
    """
    ncosthetabins, nphibins = 4, 5
    plotvars = ["luminosity", "emvelocityoverc"]
    at.packets.plotspherical.main(
        argsraw=[],
        modelpath=modelpath,
        outputfile=tmp_path / "map.pdf",
        nphibins=nphibins,
        ncosthetabins=ncosthetabins,
        plotvars=plotvars,
        gaussian_sigma=40,
        phireverse=True,
        write_data=True,
    )

    dftable = read_data_file(tmp_path / "map.txt")
    assert dftable.columns[-2:] == ["luminosity_erg_per_s", "emvelocityoverc"]
    # the colour bar draws a mesh of its own after each map
    mapcalls = [call for call in mockpcolormesh.call_args_list if call.args[0].name == "mollweide"]
    assert len(mapcalls) == len(plotvars)
    for call, colname in zip(mapcalls, ["luminosity_erg_per_s", "emvelocityoverc"], strict=True):
        drawn = np.asarray(call.args[3], dtype=np.float64)
        assert drawn.shape == (ncosthetabins, nphibins + 1)
        assert np.allclose(dftable[colname].to_numpy().reshape(drawn.shape), drawn, rtol=1e-6, atol=0, equal_nan=True)

    _, meshgrid_phi, meshgrid_theta, _ = mapcalls[-1].args
    longitude_edges = np.sort(meshgrid_phi[0])
    assert np.allclose(dftable["longitude_lower_rad"].to_numpy()[: nphibins + 1], longitude_edges[:-1][::-1])
    assert np.allclose(dftable["longitude_upper_rad"].to_numpy()[: nphibins + 1], longitude_edges[1:][::-1])
    assert np.allclose(
        dftable["costheta_lower"].to_numpy()[:: nphibins + 1], np.sin(meshgrid_theta[:-1, 0]), rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize(
    ("optionline", "expected"),
    [
        ("constexpr float RPKT_BOUNDBOUND_THERMALISATION_PROBABILITY = -1.;", "activates a macro atom"),
        ("constexpr std::optional<float> RPKT_BOUNDBOUND_THERMALISATION_PROBABILITY;", "activates a macro atom"),
        ("constexpr float RPKT_BOUNDBOUND_THERMALISATION_PROBABILITY = 0.5;", "not a macro atom"),
    ],
)
def test_last_interaction_convention_reads_a_negative_thermalisation_probability_as_off(
    optionline: str, expected: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ARTIS from 2024-05 to 2025-08 gave the option as a float, and the value -1 kept the macro atom.

    The log took each value of the option as a thermalisation probability, thus it gave the wrong convention.
    """
    (tmp_path / "artisoptions.h").write_text(f"{optionline}\n", encoding="utf-8")
    at.packets.core.print_last_interaction_convention(tmp_path)
    assert expected in capsys.readouterr().out


def test_a_position_in_a_spherical_grid_gets_the_cell_of_its_radius_and_angles() -> None:
    """A packet at the mid-point of each cell of a spherical3d grid gets that cell, and the edges follow sn3d.

    The cell index is i_r + n_r * (i_costheta + n_costheta * i_phi). The angle theta is from the +z axis, its cosine
    increases from -1 to 1, and phi = atan2(y, x) is from 0 to 2 pi. A position on the -z axis is in the first cell
    of cos(theta). A position at or outside the outer radius, and a position of NaN, get no cell.
    """
    from artistools.packets.core import get_modelgridindex_expr

    r_edges_kmps = [0.0, 2000.0, 4500.0, 9000.0]
    costheta_edges = [-1.0, -0.2, 0.6, 1.0]
    phi_edges = [0.0, 1.0, 2.5, 4.0, 2.0 * math.pi]
    modelmeta = {
        "coordinates": "spherical3d",
        "dimensions": 3,
        "t_model_init_days": 1.0,
        "vel_r_edges_kmps": r_edges_kmps,
        "costheta_edges": costheta_edges,
        "phi_edges_radians": phi_edges,
        "vmax_cmps": 9000.0 * at.constants.km_to_cm,
    }
    t_s = 5.0e4
    r_mid, costheta_mid, phi_mid = (
        mid.ravel(order="F")
        for mid in np.meshgrid(
            *(
                (np.array(edges[:-1]) + np.array(edges[1:])) / 2.0
                for edges in (r_edges_kmps, costheta_edges, phi_edges)
            ),
            indexing="ij",
        )
    )
    radius = r_mid * at.constants.km_to_cm * t_s
    extrapositions = [
        (0.0, 0.0, -3000.0 * at.constants.km_to_cm * t_s),  # the -z axis
        (3000.0 * at.constants.km_to_cm * t_s, -1e-9, 0.0),  # an angle below the x axis that rounds to phi = 2 pi
        # an angle of 0.1 rad below the x axis
        (3000.0 * at.constants.km_to_cm * t_s, -300.0 * at.constants.km_to_cm * t_s, 0.0),
        (0.0, 0.0, 0.0),  # the centre
        (9000.0 * at.constants.km_to_cm * t_s, 0.0, 0.0),  # the outer radius
        (math.nan, 0.0, 0.0),  # NaN
    ]
    extra_x, extra_y, extra_z = (list(component) for component in zip(*extrapositions, strict=True))
    dfpackets = pl.DataFrame({
        "em_posx": [*(radius * np.sqrt(1.0 - costheta_mid**2) * np.cos(phi_mid)), *extra_x],
        "em_posy": [*(radius * np.sqrt(1.0 - costheta_mid**2) * np.sin(phi_mid)), *extra_y],
        "em_posz": [*(radius * costheta_mid), *extra_z],
        "em_time": [t_s] * (len(radius) + 6),
    })

    cells = dfpackets.select(get_modelgridindex_expr("em", modelmeta, pl.LazyFrame()).alias("mgi"))["mgi"].to_list()

    ncells = len(radius)
    assert cells[:ncells] == list(range(ncells))
    # the shell of 3000 km/s has the radial index 1. cos(theta) = -1 is in the first cell of cos(theta), and
    # cos(theta) = 0 is in the cell with the index 1. phi = 2 pi is the angle 0, thus it is in the first cell of phi,
    # and an angle below it is in the last cell. The centre is in the first cell of each axis
    assert cells[ncells:] == [1 + 3 * (0 + 3 * 0), 1 + 3 * (1 + 3 * 0), 1 + 3 * (1 + 3 * 3), 0, None, None]
