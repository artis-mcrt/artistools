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

    The polars binning of the package works in Float32, thus this function checks it by a separate path.
    """
    syn_dir = np.array([0.0, 0.0, 1.0])
    pkt_dir = np.array([dirx, diry, dirz]) / math.sqrt(dirx**2 + diry**2 + dirz**2)
    costhetabin = min(int((float(pkt_dir @ syn_dir) + 1.0) / 2.0 * ncosthetabins), ncosthetabins - 1)

    vec1 = np.cross(pkt_dir, syn_dir)
    if np.linalg.norm(vec1) == 0.0:
        return costhetabin * nphibins

    vec2 = np.cross(np.array([1.0, 0.0, 0.0]), syn_dir)
    cosphi = float(vec1 @ vec2) / float(np.linalg.norm(vec1)) / float(np.linalg.norm(vec2))
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


def test_bin_and_sum_includes_both_outer_edges() -> None:
    """Every value between the first and last edge must land in a bin, including values on either outer edge."""
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    df = pl.DataFrame({"x": values, "e": [1.0] * len(values)})

    binned = at.packets.bin_and_sum(df, bincol="x", bins=[0.0, 1.0, 2.0, 3.0], sumcols=["e"], getcounts=True).collect()

    assert binned["x_bin"].to_list() == [0, 1, 2]
    # bins are [lower, upper), except the last which also includes its upper edge
    assert binned["count"].to_list() == [2, 2, 3]
    assert binned["count"].sum() == len(values)
    assert binned["e_sum"].to_list() == pytest.approx([2.0, 2.0, 3.0])


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
    """A text file of the first rank that is newer than the cache makes the cache stale.

    ARTIS writes the files of all the ranks at the same time, thus the first rank gives the time of the batch.
    """
    import shutil

    from artistools.packets.core import get_packets_rankbatch_parquetfile

    sourcedir = at.get_path("testdata") / "test-classicmode_3d" / "packets"
    for rank in (0, 1):
        shutil.copy(sourcedir / f"packets00_{rank:04d}.out.zst", tmp_path)

    parquetpath = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)
    firstwrite = parquetpath.stat().st_mtime_ns

    firstrankfile = tmp_path / "packets00_0000.out.zst"
    newtime = firstrankfile.stat().st_mtime + 100.0
    os.utime(firstrankfile, (newtime, newtime))

    parquetpath = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)

    assert parquetpath.stat().st_mtime_ns > firstwrite


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


def test_emission_expressions_give_no_value_for_a_packet_with_no_record() -> None:
    """ARTIS gives a time of 0 or -1 and a position of zero to a packet with no thermal emission record."""
    modelpath = at.get_path("testdata") / "test-classicmode_3d"
    dfmodel, modelmeta = at.get_modeldata(modelpath, printwarningsonly=True)
    emtime_s = 5.0 * at.constants.day_to_s
    dfpackets = pl.DataFrame({
        "trueem_posx": [1.0e9 * emtime_s, 0.0, 0.0],
        "trueem_posy": [0.0, 0.0, 0.0],
        "trueem_posz": [0.5e9 * emtime_s, 0.0, 0.0],
        "trueem_time": [emtime_s, -1.0, 0.0],
        "dirx": [0.0, 0.0, 0.0],
        "diry": [0.0, 0.0, 0.0],
        "dirz": [1.0, 1.0, 1.0],
    })

    dfvalues = dfpackets.select(
        velocity=at.packets.get_emission_velocity_expr("trueem"),
        losvelocity=at.packets.get_emission_velocity_lineofsight_expr("trueem"),
        modelgridindex=at.packets.get_modelgridindex_expr("trueem", modelmeta, dfmodel),
    )

    assert np.isclose(dfvalues["velocity"][0], math.hypot(1.0e9, 0.5e9), rtol=1e-12, atol=0.0)
    assert np.isclose(dfvalues["losvelocity"][0], 0.5e9, rtol=1e-12, atol=0.0)
    assert dfvalues["modelgridindex"][0] is not None
    for norecordrow in (1, 2):
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
