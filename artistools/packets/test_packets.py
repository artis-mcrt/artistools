import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import artistools as at


def test_directionbins() -> None:
    nphibins = 10
    ncosthetabins = 10
    costhetabinlowers, costhetabinuppers, _ = at.get_costheta_bins(usedegrees=False)
    phibinlowers, phibinuppers, _ = at.get_phi_bins(usedegrees=False)

    testdirections = pl.DataFrame({"phi_defined": np.linspace(0.1, 2 * math.pi, nphibins * 2, endpoint=False)}).join(
        pl.DataFrame({"costheta_defined": np.linspace(0.0, 1.0, ncosthetabins * 2, endpoint=False)}), how="cross"
    )

    syn_dir = (0, 0, 1)
    testdirections = testdirections.with_columns(
        dirx=((1 - pl.col("costheta_defined") ** 2).sqrt() * pl.col("phi_defined").cos()),
        diry=((1 - pl.col("costheta_defined") ** 2).sqrt() * pl.col("phi_defined").sin()),
        dirz=pl.col("costheta_defined"),
    )

    testdirections = at.packets.add_packet_directions_lazypolars(testdirections, syn_dir=syn_dir).collect()
    testdirections = at.packets.bin_packet_directions_lazypolars(testdirections).collect()

    for pkt in testdirections.iter_rows(named=True):
        assert np.isclose(pkt["dirx"] ** 2 + pkt["diry"] ** 2 + pkt["dirz"] ** 2, 1.0, rtol=0.001)

        assert np.isclose(pkt["costheta_defined"], pkt["costheta"], rtol=1e-4, atol=1e-4)

        assert np.isclose(pkt["phi_defined"], pkt["phi"], rtol=1e-4, atol=1e-4)

        costhetabin = pkt["costhetabin"]

        dirbin2 = at.packets.get_directionbin(
            pkt["dirx"], pkt["diry"], pkt["dirz"], nphibins=nphibins, ncosthetabins=ncosthetabins, syn_dir=syn_dir
        )

        assert dirbin2 == pkt["dirbin"]

        costhetabin2 = dirbin2 // nphibins
        phibin2 = dirbin2 % nphibins
        assert costhetabin2 == pkt["costhetabin"]
        assert phibin2 == pkt["phibin"]

        pddfpackets = at.packets.bin_packet_directions(
            dfpackets=pd.DataFrame({"dirx": [pkt["dirx"]], "diry": [pkt["diry"]], "dirz": [pkt["dirz"]]}),
            modelpath=Path(),
            syn_dir=syn_dir,
        )

        assert pddfpackets["dirbin"][0] == pkt["dirbin"]
        assert pddfpackets["costhetabin"][0] == pkt["costhetabin"]
        assert pddfpackets["phibin"][0] == pkt["phibin"]

        assert costhetabinlowers[costhetabin] <= pkt["costheta_defined"] * 1.01
        assert costhetabinuppers[costhetabin] > pkt["costheta_defined"] * 0.99

        assert phibinlowers[pkt["phibin"]] <= pkt["phi_defined"]
        assert phibinuppers[pkt["phibin"]] >= pkt["phi_defined"]

        # print(dirx, diry, dirz, dirbin, costhetabin, phibin)
