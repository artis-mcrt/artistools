#!/usr/bin/env python3
import argparse
import math
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import artistools as at


def min_dist(listin: Iterable[float], number: float) -> float:
    """Return the minimum distance between number and any item in listin."""
    min_dist_found = -1.0

    for x in listin:
        dist = abs(x - number)
        if dist < min_dist_found or min_dist_found < 0:
            min_dist_found = dist

    return min_dist_found


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create Botyanski et al. 2017 model."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    e_k = 1.2  # in units of 10^51 erg
    m_ej = 1.4  # in solar masses
    x_stb = 0.05  # mass fraction of stable Fe54 and Ni58 in Ni56 zone
    t_model_init_days = 0.0002
    t200 = t_model_init_days / 200  # time in units of 200 days

    delta = 0
    n = 10

    # density transition
    v_transition = 10943 * e_k**0.5 * m_ej**-0.5  # km/s
    rho_0 = 4.9e-17 * (e_k**-1.5) * (m_ej**2.5) * (t200**-3)  # g cm^-3

    print(f"v_transition = {v_transition:.3f}")

    # composition transition from Ni56-rich to IME-rich
    msun_g = 1.988409870698051e33
    mni56 = 0.6 * msun_g
    volni56 = mni56 / ((1 - x_stb) * rho_0)
    rni56 = (3 / 4 / math.pi * volni56) ** (1 / 3.0)
    vel_kmpersec_ni56 = rni56 / 1e5 / (200 * 86400 * t200)

    r = vel_kmpersec_ni56 * 1e5 * 200 * 86400 * t200
    m = (4 * math.pi / 3 * (r**3) * rho_0) / msun_g
    print(f"Ni56 region outer velocity = {vel_kmpersec_ni56:.3f} kms, M={m:.3f} Msun")

    dfmodel = pd.DataFrame(
        columns=["inputcellid", "vel_r_max_kmps", "logrho", "X_Fegroup", "X_Ni56", "X_Co56", "X_Fe52", "X_Cr48"]
    )
    dfmodel.index.name = "cellid"
    dfelabundances = pd.DataFrame(columns=["inputcellid", *["X_" + at.get_elsymbol(x) for x in range(1, 31)]])
    dfelabundances.index.name = "cellid"

    fixed_points = [v_transition, vel_kmpersec_ni56]
    regular_points = [v for v in np.arange(0, 14500, 1000)[1:] if min_dist(fixed_points, v) > 200]
    vlist = sorted([*fixed_points, *regular_points])

    v_inner = 0.0  # velocity at inner boundary of cell
    m_tot_msun = 0.0
    for cellid, v_outer in enumerate(vlist):  # km / s
        rho = rho_0 * (0.5 * (v_inner + v_outer) / v_transition) ** -(delta if v_outer <= v_transition else n)
        abundances = [0.0 for _ in range(31)]
        if v_outer <= vel_kmpersec_ni56:
            # Ni56-rich zone
            radioabundances = [1.0, 0.95, 0.0, 0.0, 0.0]
            abundances[26] = 0.025
            abundances[28] = 0.975
        else:
            # Intermediate-mass elements
            radioabundances = [0.0, 0.0, 0.0, 0.0, 0.0]
            abundances[14] = 0.7
            abundances[16] = 0.29
            abundances[20] = 0.01

        dfmodel.loc[cellid] = [cellid + 1, v_outer, math.log10(rho), *radioabundances]
        dfelabundances.loc[cellid] = [cellid + 1, *abundances[1:31]]
        r_inner, r_outer = (v * 1e5 * t200 * 200 * 86400 for v in (v_inner, v_outer))

        vol_shell = 4 * math.pi / 3 * (r_outer**3 - r_inner**3)
        m_tot_msun += rho * vol_shell / msun_g

        v_inner = v_outer
    print(f"M_tot = {m_tot_msun:.3f} solMass")

    at.inputmodel.save_modeldata(
        dfmodel=dfmodel, t_model_init_days=t_model_init_days, outpath=Path(args.outputpath, "model.txt")
    )
    at.inputmodel.save_initelemabundances(dfelabundances, outpath=Path(args.outputpath, "abundances.txt"))


if __name__ == "__main__":
    main()
