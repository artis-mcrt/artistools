#!/usr/bin/env python

# from rd_cmfgen import rd_nuc_decay_data
from math import exp

import numpy as np
from rd_cmfgen import rd_sn_hydro_data

# import math

msun = 1.989e33

model = "DDC25"
snapshot = "SN_HYDRO_DATA_1.300d"
# snapshot = 'SN_HYDRO_DATA_203.1d'

use_double_decay = (
    True  # should undo chains like Ni56 -> Co56 -> Fe56 instead of assuming all Fe56 and Co56 was initially Ni56
)


def undecay(a, indexofatomicnumber, indexofisotope, zparent, numnucleons):
    # e.g. parent=26, numnucleons=56 to reverse Ni56 -> Co56 decay
    daughterisofracin = a["isofrac"][:, indexofisotope[(zparent - 1, numnucleons)]]
    granddaughterisofracin = a["isofrac"][:, indexofisotope[(zparent - 2, numnucleons)]]

    a["specfrac"][:, indexofatomicnumber[zparent]] += daughterisofracin + granddaughterisofracin
    a["specfrac"][:, indexofatomicnumber[zparent - 1]] -= daughterisofracin
    a["specfrac"][:, indexofatomicnumber[zparent - 2]] -= granddaughterisofracin

    a["isofrac"][:, indexofisotope[(zparent, numnucleons)]] += daughterisofracin + granddaughterisofracin
    a["isofrac"][:, indexofisotope[(zparent - 1, numnucleons)]] -= daughterisofracin
    a["isofrac"][:, indexofisotope[(zparent - 2, numnucleons)]] -= granddaughterisofracin


def reverse_doubledecay(
    a, indexofatomicnumber, indexofisotope, zparent, numnucleons, tlate, meanlife1_days, meanlife2_days
):
    # get the abundances at time zero from the late time abundances
    # e.g. zparent=26, numnucleons=56 to reverse Ni56 -> Co56 -> Fe56 decay
    # meanlife1 is the mean lifetime of the parent (e.g. Ni56) and meanlife2 is the mean life of the daughter nucleus (e.g. Co56)
    assert tlate > 0
    iso1fraclate = a["isofrac"][:, indexofisotope[(zparent, numnucleons)]]
    iso2fraclate = a["isofrac"][:, indexofisotope[(zparent - 1, numnucleons)]]
    iso3fraclate = a["isofrac"][:, indexofisotope[(zparent - 2, numnucleons)]]

    lamb1 = 1 / meanlife1_days
    lamb2 = 1 / meanlife2_days

    iso1fract0 = np.zeros_like(iso1fraclate)
    iso2fract0 = np.zeros_like(iso1fract0)
    iso3fromdecay = np.zeros_like(iso1fract0)
    iso3fract0 = np.zeros_like(iso1fract0)
    for s in range(a["nd"]):
        iso1fract0[s] = iso1fraclate[s] * exp(lamb1 * tlate)  # larger abundance before decays

        iso2fract0[s] = (
            iso2fraclate[s] - iso1fract0[s] * lamb1 / (lamb1 - lamb2) * (exp(-lamb2 * tlate) - exp(-lamb1 * tlate))
        ) * exp(lamb2 * tlate)

        # print(iso1fract0[s], iso1fraclate[s], iso2fract0[s], iso2fraclate[s], iso1fract0[s] * lamb1 / (lamb1 - lamb2) * (exp(-lamb2 * tlate) - exp(-lamb1 * tlate)))
        # assert(iso2fract0[s] > 0)

        # print(s, (iso1fract0[s] - iso1fraclate[s]), (iso2fraclate[s] + iso3fraclate[s]))
        # print(s, (iso1fract0[s] - iso1fraclate[s]) >= (iso2fraclate[s] + iso3fraclate[s]), iso2fract0[s] < 0)

        if iso2fract0[s] < 0:
            iso2fract0[s] = iso2fraclate[s] + iso3fraclate[s] - (iso1fract0[s] - iso1fraclate[s])
            iso3fract0[s] = 0.0

            if iso2fract0[s] < 0.0:
                iso1fract0[s] += iso2fract0[s]
                iso2fract0[s] = 0.0
                print(
                    "shell",
                    s,
                    " goes fully to top isotope Z={} A={} of the chain at time zero".format(zparent, numnucleons),
                )
            else:
                print(
                    "shell",
                    s,
                    " has none of the last isotope Z={} A={} of the chain at time zero".format(
                        zparent - 2, numnucleons
                    ),
                )
        else:
            iso3fromdecay[s] = (
                (iso1fract0[s] + iso2fract0[s]) * (lamb1 - lamb2)
                - iso2fract0[s] * lamb1 * exp(-lamb2 * tlate)
                + iso2fract0[s] * lamb2 * exp(-lamb2 * tlate)
                - iso1fract0[s] * lamb1 * exp(-lamb2 * tlate)
                + iso1fract0[s] * lamb2 * exp(-lamb1 * tlate)
            ) / (lamb1 - lamb2)

            iso3fract0[s] = iso3fraclate[s] - iso3fromdecay[s]

        # print(iso2fract0[s] >= 0, iso3fract0[s] >= 0)
        sumt0 = iso1fract0[s] + iso2fract0[s] + iso3fract0[s]
        sumlate = iso1fraclate[s] + iso2fraclate[s] + iso3fraclate[s]
        if abs(sumlate - sumt0) > 1e-10:
            print(s, "t0", iso1fract0[s], iso2fract0[s], iso3fract0[s], sumt0)
            print(s, "tlate", iso1fraclate[s], iso2fraclate[s], iso3fraclate[s], sumlate, sumlate - sumt0)

        assert abs(sumt0 - sumlate) < 1e-10
        assert iso1fract0[s] >= 0.0
        assert iso2fract0[s] >= 0.0
        assert iso3fract0[s] >= 0.0

    a["isofrac"][:, indexofisotope[(zparent, numnucleons)]] = iso1fract0
    a["isofrac"][:, indexofisotope[(zparent - 1, numnucleons)]] = iso2fract0
    a["isofrac"][:, indexofisotope[(zparent - 2, numnucleons)]] = iso3fract0

    a["specfrac"][:, indexofatomicnumber[zparent]] += iso1fract0 - iso1fraclate
    a["specfrac"][:, indexofatomicnumber[zparent - 1]] += iso2fract0 - iso2fraclate
    a["specfrac"][:, indexofatomicnumber[zparent - 2]] += iso3fract0 - iso3fraclate


def forward_doubledecay(
    a, indexofatomicnumber, indexofisotope, zparent, numnucleons, tlate, meanlife1_days, meanlife2_days
):
    # get the abundances at a late time from the time zero abundances
    # e.g. zdaughter=27, numnucleons=56 for Ni56 -> Co56 -> Fe56 decay
    # meanlife1 is the mean lifetime of the parent (e.g. Ni56) and meanlife2 is the mean life of the daughter nucleus (e.g. Co56)
    assert tlate > 0
    iso1fract0 = a["isofrac"][:, indexofisotope[(zparent, numnucleons)]]
    iso2fract0 = a["isofrac"][:, indexofisotope[(zparent - 1, numnucleons)]]
    iso3fract0 = a["isofrac"][:, indexofisotope[(zparent - 2, numnucleons)]]

    lamb1 = 1 / meanlife1_days
    lamb2 = 1 / meanlife2_days

    iso1fraclate = iso1fract0 * exp(-lamb1 * tlate)  # larger abundance before decays

    iso2fraclate = iso2fract0 * exp(-lamb2 * tlate) + iso1fract0 * lamb1 / (lamb1 - lamb2) * (
        exp(-lamb2 * tlate) - exp(-lamb1 * tlate)
    )

    iso3fromdecay = (
        (iso1fract0 + iso2fract0) * (lamb1 - lamb2)
        - iso2fract0 * lamb1 * exp(-lamb2 * tlate)
        + iso2fract0 * lamb2 * exp(-lamb2 * tlate)
        - iso1fract0 * lamb1 * exp(-lamb2 * tlate)
        + iso1fract0 * lamb2 * exp(-lamb1 * tlate)
    ) / (lamb1 - lamb2)

    iso3fraclate = iso3fract0 + iso3fromdecay

    a["isofrac"][:, indexofisotope[(zparent, numnucleons)]] = iso1fraclate
    a["isofrac"][:, indexofisotope[(zparent - 1, numnucleons)]] = iso2fraclate
    a["isofrac"][:, indexofisotope[(zparent - 2, numnucleons)]] = iso3fraclate

    a["specfrac"][:, indexofatomicnumber[zparent]] += iso1fraclate - iso1fract0
    a["specfrac"][:, indexofatomicnumber[zparent - 1]] += iso2fraclate - iso2fract0
    a["specfrac"][:, indexofatomicnumber[zparent - 2]] += iso3fraclate - iso3fract0


def timeshift_double_decay(
    a, indexofatomicnumber, indexofisotope, zparent, numnucleons, timeold, timenew, meanlife1_days, meanlife2_days
):
    # take abundances back to time zero and then forward to the selected model time
    elfracsum_before = sum([a["specfrac"][:, indexofatomicnumber[zparent - i]] for i in range(3)])
    isofracsum_before = sum([a["isofrac"][:, indexofisotope[(zparent - i, numnucleons)]] for i in range(3)])

    reverse_doubledecay(
        a,
        indexofatomicnumber,
        indexofisotope,
        zparent=zparent,
        numnucleons=numnucleons,
        tlate=timeold,
        meanlife1_days=meanlife1_days,
        meanlife2_days=meanlife2_days,
    )

    forward_doubledecay(
        a,
        indexofatomicnumber,
        indexofisotope,
        zparent=zparent,
        numnucleons=numnucleons,
        tlate=timenew,
        meanlife1_days=meanlife1_days,
        meanlife2_days=meanlife2_days,
    )

    elfracsum_after = sum([a["specfrac"][:, indexofatomicnumber[zparent - i]] for i in range(3)])
    isofracsum_after = sum([a["isofrac"][:, indexofisotope[(zparent - i, numnucleons)]] for i in range(3)])
    assert np.all(abs(elfracsum_before - elfracsum_after) < 1e-10)
