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

    testdirections = pl.DataFrame({"phi": np.linspace(0.1, 2 * math.pi, nphibins * 2, endpoint=False)}).join(
        pl.DataFrame({"costheta": np.linspace(0.0, 1.0, ncosthetabins * 2, endpoint=False)}), how="cross"
    )

    syn_dir = (0, 0, 1)
    testdirections = testdirections.with_columns(
        dirx=((1 - testdirections["costheta"] ** 2).sqrt() * pl.col("phi").cos()),
        diry=((1 - testdirections["costheta"] ** 2).sqrt() * pl.col("phi").sin()),
        dirz=pl.col("costheta"),
    )

    for phi, costheta, dirx, diry, dirz in testdirections.select(
        ["phi", "costheta", "dirx", "diry", "dirz"]
    ).iter_rows():
        # dirx = math.sqrt(1.0 - costheta * costheta) * math.cos(phi)
        # diry = math.sqrt(1.0 - costheta * costheta) * math.sin(phi)
        # dirz = costheta

        assert np.isclose(dirx**2 + diry**2 + dirz**2, 1.0, rtol=0.001)

        packets = pl.DataFrame({"dirx": [dirx], "diry": [diry], "dirz": [dirz]})
        packets = at.packets.add_packet_directions_lazypolars(packets, syn_dir=syn_dir).collect()
        packets = at.packets.bin_packet_directions_lazypolars(packets).collect()

        assert np.isclose(
            costheta, float(packets.item(0, "costheta")), rtol=1e-4, atol=1e-4
        ), f"{costheta} != {packets['costheta'][0]}"

        assert np.isclose(phi, float(packets.item(0, "phi")), rtol=1e-4, atol=1e-4), f"{phi} != {packets['phi'][0]}"

        costhetabin = int(packets.item(0, "costhetabin"))
        phibin = packets.item(0, "phibin")
        dirbin = packets.item(0, "dirbin")

        dirbin2 = at.packets.get_directionbin(
            dirx, diry, dirz, nphibins=nphibins, ncosthetabins=ncosthetabins, syn_dir=syn_dir
        )
        assert dirbin2 == dirbin, f"{dirbin2} != {dirbin}, {phi}, {costheta}"
        costhetabin2 = dirbin2 // nphibins
        phibin2 = dirbin2 % nphibins
        assert costhetabin2 == costhetabin
        assert phibin2 == phibin

        pddfpackets = at.packets.bin_packet_directions(
            dfpackets=pd.DataFrame({"dirx": [dirx], "diry": [diry], "dirz": [dirz]}),
            modelpath=Path(),
            syn_dir=syn_dir,
        )

        assert pddfpackets["dirbin"][0] == dirbin, f"{pddfpackets['dirbin'][0]} != {dirbin}"
        assert pddfpackets["costhetabin"][0] == costhetabin, f"{pddfpackets['costhetabin'][0]} != {costhetabin}"
        assert pddfpackets["phibin"][0] == phibin, f"{pddfpackets['phibin'][0]} != {phibin}"

        assert costhetabinlowers[costhetabin] <= costheta * 1.01
        assert costhetabinuppers[costhetabin] > costheta * 0.99

        assert phibinlowers[phibin] <= phi, f"{phibinlowers[phibin]} <= {phi}"
        assert phibinuppers[phibin] >= phi

        # print(dirx, diry, dirz, dirbin, costhetabin, phibin)
