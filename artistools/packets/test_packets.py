import math

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
