import math
import os
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at


def get_reference_dirbin(dirx: float, diry: float, dirz: float, nphibins: int, ncosthetabins: int) -> int:
    """Return the direction bin of one packet, computed in float64 for a viewing direction along +z.

    The polars binning of the package works in Float32, thus this function checks it by a separate path. The function
    follows get_escapedirectionbin of ARTIS (vectors.h). ARTIS takes cosphi = 1 for a direction along the z axis,
    thus such a direction is in the phi bin of the angle pi and not in the phi bin 0.
    """
    syn_dir = np.array([0.0, 0.0, 1.0])
    pkt_dir = np.array([dirx, diry, dirz]) / math.sqrt(dirx**2 + diry**2 + dirz**2)
    costhetabin = min(int((float(pkt_dir @ syn_dir) + 1.0) / 2.0 * ncosthetabins), ncosthetabins - 1)

    vec1 = np.cross(pkt_dir, syn_dir)
    vec1len = float(np.linalg.norm(vec1))
    vec2 = np.cross(np.array([1.0, 0.0, 0.0]), syn_dir)
    cosphi = float(vec1 @ vec2) / vec1len / float(np.linalg.norm(vec2)) if vec1len > 1e-12 else 1.0
    phi = math.acos(cosphi) if float(vec1 @ np.cross(vec2, syn_dir)) > 0 else math.acos(cosphi) + math.pi
    return costhetabin * nphibins + min(int(phi / 2.0 / math.pi * nphibins), nphibins - 1)


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


@pytest.mark.parametrize("nphibins", [4, 10])
def test_directionbins_phibin_upper_edge(nphibins: int) -> None:
    """A direction with diry == 0 and dirx < 0 gives acos(cosphi) + pi == 2 pi, which must not overflow the ring."""
    ncosthetabins = 10
    dirx, diry, dirz = -1.0, 0.0, 0.0

    dfpackets = at.packets.add_packet_directions_lazypolars(
        pl.DataFrame({"dirx": [dirx], "diry": [diry], "dirz": [dirz]})
    )
    binned = at.packets.bin_packet_directions_polars(
        dfpackets, nphibins=nphibins, ncosthetabins=ncosthetabins
    ).collect()

    assert binned["phibin"].item() == nphibins - 1
    assert binned["dirbin"].item() == get_reference_dirbin(dirx, diry, dirz, nphibins, ncosthetabins)


@pytest.mark.parametrize(
    ("direction", "expecteddirbin"),
    [
        ((0.0, 0.0, 1.0), 95),
        ((0.0, 0.0, -1.0), 5),
        ((0.0, 1e-30, -1.0), 5),
        ((0.0, -1e-30, 1.0), 90),
        ((0.0, 1e-13, 1.0), 95),
    ],
)
def test_a_direction_along_the_z_axis_gets_the_bin_of_artis(
    direction: tuple[float, float, float], expecteddirbin: int
) -> None:
    """A direction along the z axis has no phi angle, and ARTIS takes cosphi = 1 for it (vectors.h).

    The division 0/0 gave cosphi = NaN, thus the direction got phi bin 0 and not the phi bin of ARTIS. A small
    component across the axis gave the true phi, which ARTIS ignores below 1e-12.
    """
    dirx, diry, dirz = direction
    dfpackets = at.packets.add_packet_directions_lazypolars(
        pl.DataFrame({"dirx": [dirx], "diry": [diry], "dirz": [dirz]})
    )
    binned = at.packets.bin_packet_directions_polars(dfpackets, nphibins=10, ncosthetabins=10).collect()

    assert binned["dirbin"].item() == expecteddirbin
    assert get_reference_dirbin(dirx, diry, dirz, nphibins=10, ncosthetabins=10) == expecteddirbin


def test_get_virtual_packets() -> None:
    nprocs_read, dfvpkt = at.packets.get_virtual_packets(
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


def test_a_packets_file_with_a_cut_line_stops_the_conversion(tmp_path: Path) -> None:
    """A file that ARTIS or a copy still writes ends in a cut line, and its cache must not keep the partial data.

    The reader gave the values of the cut line to the first columns and null to the others, with no error.
    """
    from artistools.packets.core import readfile_text

    columns = ["number", "where", "type_id", "e_cmf", "e_rf", "pellet_nucindex"]
    packetsfile = tmp_path / "packets00_0000.out"
    packetsfile.write_text("1 58900 32 1.2e45 1.00799e+45 5 \n2 58900 32 1.2e45 1.007", encoding="utf-8")

    with pytest.raises(ValueError, match="fewer values than the header"):
        readfile_text(packetsfile, column_names=columns)


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
            "cacheversion": str(CACHEVERSION),
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


def test_add_packet_directions_accepts_a_frame_that_holds_the_angles() -> None:
    """A frame that already carries phi must pass through, because the parquet cache stores it."""
    dfpackets = pl.LazyFrame({"dirx": [0.6, 0.0], "diry": [0.0, 0.8], "dirz": [0.8, 0.6]})

    dfonce = at.packets.add_packet_directions_lazypolars(dfpackets).collect()
    assert "phi" in dfonce.columns
    assert "vec1_x" not in dfonce.columns

    # the drop named the vec1 columns whatever the input held, thus this raised ColumnNotFoundError
    dftwice = at.packets.add_packet_directions_lazypolars(dfonce).collect()

    pltest.assert_frame_equal(dfonce, dftwice)


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
    dfmodel, modelmeta = at.get_modeldata(modelpath, printwarningsonly=True)
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


def test_get_packets_gives_nan_to_the_thermal_velocity_of_a_packet_with_no_record() -> None:
    """An old packets file holds a thermal emission velocity of zero for a packet with no thermal emission record."""
    _, lzdfpackets = at.packets.get_packets(
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
def test_lastpacketinteraction_ignores_a_packet_with_no_thermal_emission_record(
    mockimshow: mock.MagicMock, mocksavefigure: mock.MagicMock
) -> None:
    """A packet with no thermal emission record gave a large negative weight, which removed the innermost bin."""
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "testmodel"
    tdays = 300.0
    timestep = at.misc.get_timestep_of_timedays(modelpath, tdays)
    t_arrive_d = at.get_timestep_times(modelpath, loc="mid")[timestep]
    emtime_s = 250.0 * at.constants.day_to_s
    beta = 0.01
    # the two packets are in one bin of the histogram: a record at a velocity of 0.01 c, and no record
    dfpackets = pl.DataFrame({
        "t_arrive_d": [t_arrive_d, t_arrive_d],
        "e_rf": [1.0e40, 1.0e40],
        "trueem_posx": [beta * at.constants.C_cm_per_s * emtime_s, 0.0],
        "trueem_posy": [0.0, 0.0],
        "trueem_posz": [beta * at.constants.C_cm_per_s * emtime_s, 0.0],
        "trueem_time": [emtime_s, -1.0],
    })

    with mock.patch.object(
        plotlastpacketinteraction, "get_reduced_packet_set", return_value=(1, dfpackets.lazy(), 1.0)
    ):
        plotlastpacketinteraction.packets_2d_hist_bin_and_ejecta_vel(
            modelpath, tdays=tdays, srIItriplet=False, colorlogscale=False, dirbin=-1, trueem=True
        )

    assert mocksavefigure.call_count == 1
    heatmap = mockimshow.call_args.args[1].T
    assert heatmap.count() == 1
    assert heatmap[0, 25] > 0.0


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


@mock.patch("artistools.packets.plotlastpacketinteraction.save_figure")
@mock.patch.object(mplax.Axes, "imshow", side_effect=mplax.Axes.imshow, autospec=True)
def test_lastpacketinteraction_takes_the_start_of_the_timestep_and_not_the_end(
    mockimshow: mock.MagicMock, mocksavefigure: mock.MagicMock
) -> None:
    """A packet that arrives at the start of the timestep is in the plot, and a packet at the end is not.

    The window of the arrival time included its right edge, thus it took a packet of the next timestep instead.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "testmodel"
    tdays = 300.0
    timestep = at.misc.get_timestep_of_timedays(modelpath, tdays)
    t_start = at.get_timestep_times(modelpath, loc="start")[timestep]
    t_end = at.get_timestep_times(modelpath, loc="end")[timestep]
    emtime_s = 250.0 * at.constants.day_to_s
    # the packet at the start is in the bin of 0.01 c, and the packet at the end is in the bin of 0.21 c
    betas = [0.01, 0.21]
    dfpackets = pl.DataFrame({
        "t_arrive_d": [t_start, t_end],
        "e_rf": [1.0e40, 1.0e40],
        "trueem_posx": [beta * at.constants.C_cm_per_s * emtime_s for beta in betas],
        "trueem_posy": [0.0, 0.0],
        "trueem_posz": [beta * at.constants.C_cm_per_s * emtime_s for beta in betas],
        "trueem_time": [emtime_s, emtime_s],
    })

    with mock.patch.object(
        plotlastpacketinteraction, "get_reduced_packet_set", return_value=(1, dfpackets.lazy(), 1.0)
    ):
        plotlastpacketinteraction.packets_2d_hist_bin_and_ejecta_vel(
            modelpath, tdays=tdays, srIItriplet=False, colorlogscale=False, dirbin=-1, trueem=True
        )

    assert mocksavefigure.call_count == 1
    heatmap = mockimshow.call_args.args[1].T
    assert heatmap.count() == 1
    assert heatmap[0, 25] > 0.0


def test_get_packets_stops_for_a_run_with_no_escaped_gamma_packets(tmp_path: Path) -> None:
    """ARTIS removes the escaped gamma packets unless KEEP_ESCAPED_GAMMAS is true, which is not the default.

    The empty frame gave a gamma-ray light curve and a gamma-ray spectrum of zero with no message.
    """
    type_escape, type_rpkt = at.packets.core.type_ids["TYPE_ESCAPE"], at.packets.core.type_ids["TYPE_RPKT"]
    cachepath = tmp_path / "packetsbatch00_0000_0000.out.parquet.tmp"
    pl.DataFrame(
        {"type_id": [type_escape, type_escape], "escape_type_id": [type_rpkt, type_rpkt]},
        schema={"type_id": pl.Int32, "escape_type_id": pl.Int32},
    ).write_parquet(cachepath)

    with mock.patch("artistools.packets.core.get_packets_batch_parquet_paths", return_value=(1, [cachepath])):
        _, dfrpkt = at.packets.get_packets(tmp_path, escape_type="TYPE_RPKT")
        assert dfrpkt.collect().height == 2

        with pytest.raises(ValueError, match="KEEP_ESCAPED_GAMMAS"):
            at.packets.get_packets(tmp_path, escape_type="TYPE_GAMMA")


@pytest.mark.parametrize(
    ("columnname", "values"),
    [("originated_from_positron", [True, False]), ("originated_from_particlenotgamma", [1, 0])],
)
def test_get_packets_gives_one_name_and_type_to_the_flag_of_the_particle_origin(
    tmp_path: Path, columnname: str, values: list[bool] | list[int]
) -> None:
    """An older ARTIS writes originated_from_positron, and the current ARTIS writes originated_from_particlenotgamma.

    The cache keeps the name and the type of its text file, thus the two formats gave two names and two types.
    """
    cachepath = tmp_path / "packetsbatch00_0000_0000.out.parquet.tmp"
    pl.DataFrame({"type_id": [32, 32], columnname: values}).write_parquet(cachepath)

    with mock.patch("artistools.packets.core.get_packets_batch_parquet_paths", return_value=(1, [cachepath])):
        _, dfpackets = at.packets.get_packets(tmp_path)

    dfflag = dfpackets.select("originated_from_particlenotgamma").collect()
    pltest.assert_frame_equal(dfflag, pl.DataFrame({"originated_from_particlenotgamma": [True, False]}))


def test_lastpacketinteraction_main_writes_the_plot_to_the_output_folder(tmp_path: Path) -> None:
    """The command takes -timedays as the other commands do, and -o gives the folder or the file of the plot.

    The command took only its own -tdays, and it always wrote its file to the working folder.
    """
    from artistools.packets import plotlastpacketinteraction

    modelpath = at.get_path("testdata") / "testmodel"
    plotlastpacketinteraction.main(argsraw=["-modelpath", str(modelpath), "-timedays", "300", "-o", str(tmp_path)])
    assert (tmp_path / "testmodel_allelements_allions_t_arrive_d_300.0_ts54_into_dirbin-1.pdf").is_file()

    # -tdays is the old spelling, and -o can also name the file
    outputfile = tmp_path / "lastinteraction.pdf"
    plotlastpacketinteraction.main(argsraw=["-modelpath", str(modelpath), "-tdays", "300", "-o", str(outputfile)])
    assert outputfile.is_file()


@pytest.mark.parametrize(
    ("argsraw", "expectedmessage"),
    [
        (["-element", "Xx"], "-element Xx is no element symbol"),
        (["-dirbin", "100"], "-dirbin 100 is not the first direction bin"),
        (["-dirbin", "35"], "-dirbin 35 is not the first direction bin"),
        (["-dirbin", "-10"], "-dirbin -10 is not the first direction bin"),
        ([], "the time is missing"),
    ],
)
def test_lastpacketinteraction_refuses_a_selection_that_matches_no_packet(
    argsraw: list[str], expectedmessage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown element and a direction bin outside the costheta bins gave an empty plot with no error."""
    from artistools.packets import plotlastpacketinteraction

    timeargs = [] if not argsraw else ["-timedays", "300"]
    modelpath = at.get_path("testdata") / "testmodel"
    with (
        mock.patch.object(plotlastpacketinteraction, "packets_2d_hist_bin_and_ejecta_vel") as mockplot,
        pytest.raises(SystemExit),
    ):
        plotlastpacketinteraction.main(argsraw=["-modelpath", str(modelpath), *timeargs, *argsraw])

    assert mockplot.call_count == 0
    captured = capsys.readouterr()
    assert expectedmessage in captured.err + captured.out


def test_timestep_of_a_time_at_the_start_of_a_timestep() -> None:
    """An ARTIS timestep holds its start time.

    A time at a start went to the timestep before it, and the first gave -1.
    """
    from artistools.packets.core import get_timestep_expr

    dftimes = pl.DataFrame({"time": [0.5, 1.0, 1.5, 2.0, 3.0, 3.5]})
    timesteps = dftimes.select(get_timestep_expr(pl.col("time"), [1.0, 2.0, 3.0])).to_series().to_list()

    assert timesteps == [-1, 0, 0, 1, 2, 2]
