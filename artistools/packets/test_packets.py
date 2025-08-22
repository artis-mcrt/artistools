import math

import numpy as np
import polars as pl

import artistools as at


def test_directionbins() -> None:
    nphibins = 10
    ncosthetabins = 10
    costhetabinlowers, costhetabinuppers, _ = at.get_costheta_bins(usedegrees=False)
    phibinlowers, phibinuppers, _ = at.get_phi_bins(usedegrees=False)

    testdirections = pl.DataFrame({"phi_defined": np.linspace(0.1, 2 * math.pi, nphibins * 2, endpoint=False)}).join(
        pl.DataFrame({"costheta_defined": np.linspace(0.0, 1.0, ncosthetabins * 2, endpoint=True)}), how="cross"
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

        # print(dirx, diry, dirz, dirbin, costhetabin, phibin)

    testdirections_pandas = testdirections.to_pandas(use_pyarrow_extension_array=False)

    pddfpackets = at.packets.bin_packet_directions(dfpackets=testdirections_pandas)

    for row in pddfpackets.itertuples(index=True):
        assert isinstance(row.costheta_defined, float)
        assert isinstance(row.phi_defined, float)
        assert isinstance(row.costheta, float)
        assert isinstance(row.phi, float)
        assert isinstance(row.dirz, float)
        pktdir_is_along_zaxis = np.isclose(row.dirz, 1.0) or np.isclose(row.dirz, -1.0)
        assert math.isclose(row.costheta_defined, row.costheta, rel_tol=1e-4, abs_tol=1e-4)
        assert math.isclose(row.phi_defined, row.phi, rel_tol=1e-4, abs_tol=1e-4) or pktdir_is_along_zaxis

        assert isinstance(row.costhetabin, int)
        expected_costhetabin = testdirections.item(row[0], "costhetabin")
        assert expected_costhetabin == row.costhetabin, f"Expected {expected_costhetabin}, got {row.costhetabin}"

        assert isinstance(row.phibin, int)
        expected_phibin = testdirections.item(row[0], "phibin")
        assert expected_phibin == row.phibin or pktdir_is_along_zaxis, f"Expected {expected_phibin}, got {row.phibin}"

        assert isinstance(row.dirbin, int)
        expected_dirbin = testdirections.item(row[0], "dirbin")
        assert expected_dirbin == row.dirbin or pktdir_is_along_zaxis, f"Expected {expected_dirbin}, got {row.dirbin}"


def test_get_virtual_packets_pl() -> None:
    _, dfvpkt = at.packets.get_virtual_packets_pl(
        modelpath=at.get_config()["path_testdata"] / "vpktcontrib", maxpacketfiles=2
    )

    npkts_total = dfvpkt.select(pl.len()).collect().item()
    assert npkts_total == 13783
