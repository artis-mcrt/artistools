import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import artistools as at


def test_directionbins() -> None:
    nphibins = 10
    ncosthetabins = 10
    costhetabinlowers, costhetabinuppers, _ = at.get_costheta_bins(usedegrees=False)
    phibinlowers, phibinuppers, _ = at.get_phi_bins(usedegrees=False)

    testdirections = pl.DataFrame({
        "phi_defined": np.linspace(0.1, 2 * math.pi, nphibins * 2, endpoint=False).tolist()
    }).join(
        pl.DataFrame({"costheta_defined": np.linspace(0.0, 1.0, ncosthetabins * 2, endpoint=True).tolist()}),
        how="cross",
    )

    syn_dir = (0, 0, 1)
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

        dirbin2 = at.packets.get_directionbin(
            pkt["dirx"], pkt["diry"], pkt["dirz"], nphibins=nphibins, ncosthetabins=ncosthetabins, syn_dir=syn_dir
        )

        assert dirbin2 == pkt["dirbin"]

        assert costhetabinlowers[pkt["costhetabin"]] <= pkt["costheta_defined"] * 1.01
        assert costhetabinuppers[pkt["costhetabin"]] > pkt["costheta_defined"] * 0.99

        assert pkt["costhetabin"] == dirbin2 // nphibins
        assert pkt["phibin"] == dirbin2 % nphibins

        assert phibinlowers[pkt["phibin"]] <= pkt["phi_defined"] or pktdir_is_along_zaxis
        assert phibinuppers[pkt["phibin"]] >= pkt["phi_defined"] or pktdir_is_along_zaxis


def test_directionbins_unequal_bincounts() -> None:
    """Check the dirbin layout when the phi and costheta bin counts differ.

    The default configuration uses 10 of each, which hides any confusion between the two counts.
    """
    nphibins = 8
    ncosthetabins = 4
    syn_dir = (0, 0, 1)

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

        assert pkt["dirbin"] == at.packets.get_directionbin(
            pkt["dirx"], pkt["diry"], pkt["dirz"], nphibins=nphibins, ncosthetabins=ncosthetabins, syn_dir=syn_dir
        )


@pytest.mark.parametrize("nphibins", [4, 10])
def test_directionbins_phibin_upper_edge(nphibins: int) -> None:
    """A direction with diry == 0 and dirx < 0 gives acos(cosphi) + pi == 2 pi, which must not overflow the ring."""
    ncosthetabins = 10
    dirx, diry, dirz = -1.0, 0.0, 0.0

    dirbin = at.packets.get_directionbin(
        dirx, diry, dirz, nphibins=nphibins, ncosthetabins=ncosthetabins, syn_dir=(0, 0, 1)
    )
    assert dirbin % nphibins == nphibins - 1
    assert dirbin < nphibins * ncosthetabins

    dfpackets = at.packets.add_packet_directions_lazypolars(
        pl.DataFrame({"dirx": [dirx], "diry": [diry], "dirz": [dirz]})
    )
    binned = at.packets.bin_packet_directions_polars(
        dfpackets, nphibins=nphibins, ncosthetabins=ncosthetabins
    ).collect()

    assert binned["phibin"].item() == nphibins - 1
    assert binned["dirbin"].item() == dirbin


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

    from artistools.packets.packets import readfile_text

    dfpackets = readfile_text(packetsfile, column_names=columns)

    assert dfpackets.columns == [*columns, "mpirank"]
    assert dfpackets["mpirank"].to_list() == [0, 0, 0]


def test_packets_cache_goes_stale_when_any_rank_file_changes(tmp_path: Path) -> None:
    """Every rank of a batch decides the freshness of its cache, and not the last rank alone."""
    import os
    import shutil

    from artistools.packets.packets import get_packets_rankbatch_parquetfile

    sourcedir = at.get_path("testdata") / "test-classicmode_3d" / "packets"
    for rank in (0, 1):
        shutil.copy(sourcedir / f"packets00_{rank:04d}.out.zst", tmp_path)

    parquetpath = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)
    firstwrite = parquetpath.stat().st_mtime_ns

    # only the file of the first rank becomes newer, because a check of the last rank alone would miss it
    firstrankfile = tmp_path / "packets00_0000.out.zst"
    newtime = firstrankfile.stat().st_mtime + 100.0
    os.utime(firstrankfile, (newtime, newtime))

    parquetpath = get_packets_rankbatch_parquetfile(tmp_path, batch_mpiranks=[0, 1], batchindex=0, virtual=False)

    assert parquetpath.stat().st_mtime_ns > firstwrite
