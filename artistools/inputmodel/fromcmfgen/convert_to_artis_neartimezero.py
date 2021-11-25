#!/usr/bin/env python

import numpy as np

from rd_cmfgen import rd_sn_hydro_data
# from rd_cmfgen import rd_nuc_decay_data
from math import exp
import math

msun = 1.989e33

model = 'DDC25'
snapshot = 'SN_HYDRO_DATA_1.300d'
# snapshot = 'SN_HYDRO_DATA_203.1d'

use_double_decay = True  # should undo chains like Ni56 -> Co56 -> Fe56 instead of assuming all Fe56 and Co56 was initially Ni56


def undecay(a, indexofatomicnumber, indexofisotope, zparent, numnucleons):
    # e.g. parent=26, numnucleons=56 to reverse Ni56 -> Co56 decay
    daughterisofracin = a['isofrac'][:, indexofisotope[(zparent - 1, numnucleons)]]
    granddaughterisofracin = a['isofrac'][:, indexofisotope[(zparent - 2, numnucleons)]]

    a['specfrac'][:, indexofatomicnumber[zparent]] += daughterisofracin + granddaughterisofracin
    a['specfrac'][:, indexofatomicnumber[zparent - 1]] -= daughterisofracin
    a['specfrac'][:, indexofatomicnumber[zparent - 2]] -= granddaughterisofracin

    a['isofrac'][:, indexofisotope[(zparent, numnucleons)]] += daughterisofracin + granddaughterisofracin
    a['isofrac'][:, indexofisotope[(zparent - 1, numnucleons)]] -= daughterisofracin
    a['isofrac'][:, indexofisotope[(zparent - 2, numnucleons)]] -= granddaughterisofracin


def reverse_doubledecay(a, indexofatomicnumber, indexofisotope, zparent, numnucleons, tlate, meanlife1_days, meanlife2_days):
    # get the abundances at time zero from the late time abundances
    # e.g. zparent=26, numnucleons=56 to reverse Ni56 -> Co56 -> Fe56 decay
    # meanlife1 is the mean lifetime of the parent (e.g. Ni56) and meanlife2 is the mean life of the daughter nucleus (e.g. Co56)
    assert(tlate > 0)
    iso1fraclate = a['isofrac'][:, indexofisotope[(zparent, numnucleons)]]
    iso2fraclate = a['isofrac'][:, indexofisotope[(zparent - 1, numnucleons)]]
    iso3fraclate = a['isofrac'][:, indexofisotope[(zparent - 2, numnucleons)]]

    lamb1 = 1 / meanlife1_days
    lamb2 = 1 / meanlife2_days

    iso1fract0 = np.zeros_like(iso1fraclate)
    iso2fract0 = np.zeros_like(iso1fract0)
    iso3fromdecay = np.zeros_like(iso1fract0)
    iso3fract0 = np.zeros_like(iso1fract0)
    for s in range(a['nd']):
        iso1fract0[s] = iso1fraclate[s] * exp(lamb1 * tlate)  # larger abundance before decays

        iso2fract0[s] = (iso2fraclate[s] - iso1fract0[s] * lamb1 / (lamb1 - lamb2) * (exp(-lamb2 * tlate) - exp(-lamb1 * tlate))) * exp(lamb2 * tlate)

        # print(iso1fract0[s], iso1fraclate[s], iso2fract0[s], iso2fraclate[s], iso1fract0[s] * lamb1 / (lamb1 - lamb2) * (exp(-lamb2 * tlate) - exp(-lamb1 * tlate)))
        # assert(iso2fract0[s] > 0)

        # print(s, (iso1fract0[s] - iso1fraclate[s]), (iso2fraclate[s] + iso3fraclate[s]))
        # print(s, (iso1fract0[s] - iso1fraclate[s]) >= (iso2fraclate[s] + iso3fraclate[s]), iso2fract0[s] < 0)

        if iso2fract0[s] < 0:
            iso2fract0[s] = iso2fraclate[s] + iso3fraclate[s] - (iso1fract0[s] - iso1fraclate[s])
            iso3fract0[s] = 0.

            if (iso2fract0[s] < 0.):
                iso1fract0[s] += iso2fract0[s]
                iso2fract0[s] = 0.
                print("shell", s, " goes fully to top isotope Z={} A={} of the chain at time zero".format(zparent, numnucleons))
            else:
                print("shell", s, " has none of the last isotope Z={} A={} of the chain at time zero".format(zparent - 2, numnucleons))
        else:
            iso3fromdecay[s] = (
                (iso1fract0[s] + iso2fract0[s]) * (lamb1 - lamb2) -
                iso2fract0[s] * lamb1 * exp(-lamb2 * tlate) +
                iso2fract0[s] * lamb2 * exp(-lamb2 * tlate) -
                iso1fract0[s] * lamb1 * exp(-lamb2 * tlate) +
                iso1fract0[s] * lamb2 * exp(-lamb1 * tlate)) / (lamb1 - lamb2)

            iso3fract0[s] = iso3fraclate[s] - iso3fromdecay[s]

        # print(iso2fract0[s] >= 0, iso3fract0[s] >= 0)
        sumt0 = iso1fract0[s] + iso2fract0[s] + iso3fract0[s]
        sumlate = iso1fraclate[s] + iso2fraclate[s] + iso3fraclate[s]
        if abs(sumlate - sumt0) > 1e-10:
            print(s, "t0", iso1fract0[s], iso2fract0[s], iso3fract0[s], sumt0)
            print(s, "tlate", iso1fraclate[s], iso2fraclate[s], iso3fraclate[s], sumlate, sumlate - sumt0)

        assert(abs(sumt0 - sumlate) < 1e-10)
        assert(iso1fract0[s] >= 0.)
        assert(iso2fract0[s] >= 0.)
        assert(iso3fract0[s] >= 0.)

    a['isofrac'][:, indexofisotope[(zparent, numnucleons)]] = iso1fract0
    a['isofrac'][:, indexofisotope[(zparent - 1, numnucleons)]] = iso2fract0
    a['isofrac'][:, indexofisotope[(zparent - 2, numnucleons)]] = iso3fract0

    a['specfrac'][:, indexofatomicnumber[zparent]] += iso1fract0 - iso1fraclate
    a['specfrac'][:, indexofatomicnumber[zparent - 1]] += iso2fract0 - iso2fraclate
    a['specfrac'][:, indexofatomicnumber[zparent - 2]] += iso3fract0 - iso3fraclate


def forward_doubledecay(a, indexofatomicnumber, indexofisotope, zparent, numnucleons, tlate, meanlife1_days, meanlife2_days):
    # get the abundances at a late time from the time zero abundances
    # e.g. zdaughter=27, numnucleons=56 for Ni56 -> Co56 -> Fe56 decay
    # meanlife1 is the mean lifetime of the parent (e.g. Ni56) and meanlife2 is the mean life of the daughter nucleus (e.g. Co56)
    assert(tlate > 0)
    iso1fract0 = a['isofrac'][:, indexofisotope[(zparent, numnucleons)]]
    iso2fract0 = a['isofrac'][:, indexofisotope[(zparent - 1, numnucleons)]]
    iso3fract0 = a['isofrac'][:, indexofisotope[(zparent - 2, numnucleons)]]

    lamb1 = 1 / meanlife1_days
    lamb2 = 1 / meanlife2_days

    iso1fraclate = iso1fract0 * exp(-lamb1 * tlate)  # larger abundance before decays

    iso2fraclate = (
        iso2fract0 * exp(-lamb2 * tlate) +
        iso1fract0 * lamb1 / (lamb1 - lamb2) * (exp(-lamb2 * tlate) - exp(-lamb1 * tlate)))

    iso3fromdecay = (
        (iso1fract0 + iso2fract0) * (lamb1 - lamb2) -
        iso2fract0 * lamb1 * exp(-lamb2 * tlate) +
        iso2fract0 * lamb2 * exp(-lamb2 * tlate) -
        iso1fract0 * lamb1 * exp(-lamb2 * tlate) +
        iso1fract0 * lamb2 * exp(-lamb1 * tlate)) / (lamb1 - lamb2)

    iso3fraclate = iso3fract0 + iso3fromdecay

    a['isofrac'][:, indexofisotope[(zparent, numnucleons)]] = iso1fraclate
    a['isofrac'][:, indexofisotope[(zparent - 1, numnucleons)]] = iso2fraclate
    a['isofrac'][:, indexofisotope[(zparent - 2, numnucleons)]] = iso3fraclate

    a['specfrac'][:, indexofatomicnumber[zparent]] += iso1fraclate - iso1fract0
    a['specfrac'][:, indexofatomicnumber[zparent - 1]] += iso2fraclate - iso2fract0
    a['specfrac'][:, indexofatomicnumber[zparent - 2]] += iso3fraclate - iso3fract0


def timeshift_double_decay(a, indexofatomicnumber, indexofisotope, zparent, numnucleons, timeold, timenew, meanlife1_days, meanlife2_days):
    # take abundances back to time zero and then forward to the selected model time
    elfracsum_before = sum([a['specfrac'][:, indexofatomicnumber[zparent - i]] for i in range(3)])
    isofracsum_before = sum([a['isofrac'][:, indexofisotope[(zparent - i, numnucleons)]] for i in range(3)])

    reverse_doubledecay(a, indexofatomicnumber, indexofisotope, zparent=zparent, numnucleons=numnucleons, tlate=timeold, meanlife1_days=meanlife1_days, meanlife2_days=meanlife2_days)

    forward_doubledecay(a, indexofatomicnumber, indexofisotope, zparent=zparent, numnucleons=numnucleons, tlate=timenew, meanlife1_days=meanlife1_days, meanlife2_days=meanlife2_days)

    elfracsum_after = sum([a['specfrac'][:, indexofatomicnumber[zparent - i]] for i in range(3)])
    isofracsum_after = sum([a['isofrac'][:, indexofisotope[(zparent - i, numnucleons)]] for i in range(3)])
    assert(np.all(abs(elfracsum_before - elfracsum_after) < 1e-10))
    assert(np.all(abs(isofracsum_before - isofracsum_after) < 1e-10))


def adjust_model_to_time_nearzero(a, indexofatomicnumber, indexofisotope):
    input_time = a['time']
    a['time'] = 0.0001
    a['dens'][:] *= (input_time / a['time']) ** 3
    a['rad'][:] *= a['time'] / input_time

    if use_double_decay:
        # Ni56 -> Co56 -> Fe56
        timeshift_double_decay(a, indexofatomicnumber, indexofisotope, zparent=28, numnucleons=56, timeold=input_time, timenew=a['time'], meanlife1_days=8.76, meanlife2_days=111.427)

        # Fe52 -> Mn52 -> Cr52
        timeshift_double_decay(a, indexofatomicnumber, indexofisotope, zparent=26, numnucleons=52, timeold=input_time, timenew=a['time'], meanlife1_days=0.345/np.log(2.), meanlife2_days=0.015/np.log(2.))

        # Cr48 -> V48 -> Ti48
        timeshift_double_decay(a, indexofatomicnumber, indexofisotope, zparent=24, numnucleons=48, timeold=input_time, timenew=a['time'], meanlife1_days=1.29602, meanlife2_days=23.0442)

    else:
        undecay(a, indexofatomicnumber, indexofisotope, zparent=28, numnucleons=56)  # Ni56 -> Co56
        undecay(a, indexofatomicnumber, indexofisotope, zparent=26, numnucleons=52)  # Fe52 -> Mn52
        undecay(a, indexofatomicnumber, indexofisotope, zparent=24, numnucleons=48)  # Cr48 -> V48


# Mapping of the CMFGEN species to atomic numbers, and masking IGEs
# For now I include Ba in the IGE mass fraction, but do not include it as a chemical species
# I assume Ba is just a proxy for everything heavier than Ni in their simulations.
spectoz = [
    1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 56
]
ige_index = (np.array(spectoz) > 20)

def convert_model(filename):

    a = rd_sn_hydro_data(filename, reverse='true')

    atomicnumberoflabel = {}
    indexofatomicnumber = {}
    for i, z in enumerate(spectoz):
        atomicnumberoflabel[a['spec'][i]] = z
        indexofatomicnumber[z] = i

    indexofisotope = {}
    for i, (elementlabel, nucleons) in enumerate(zip(a['iso'], a['aiso'])):
        indexofisotope[(atomicnumberoflabel[elementlabel], nucleons)] = i

    s = 50
    print("shell ", s)
    print(a['time'], ": ni56 ", a['isofrac'][s, indexofisotope[(28, 56)]], " co56 ", a['isofrac'][s, indexofisotope[(27, 56)]], " fe56 ", a['isofrac'][s, indexofisotope[(26, 56)]] ," sum ", a['isofrac'][s, indexofisotope[(28, 56)]] + a['isofrac'][s, indexofisotope[(27, 56)]] + a['isofrac'][s, indexofisotope[(26, 56)]])
    # print("before: fe52 ", a['isofrac'][s, indexofisotope[(26, 52)]], " mn52 ", a['isofrac'][s, indexofisotope[(25, 52)]], " cr52 ", a['isofrac'][s, indexofisotope[(24, 52)]] ," sum ", a['isofrac'][s, indexofisotope[(26, 52)]] + a['isofrac'][s, indexofisotope[(25, 52)]] + a['isofrac'][s, indexofisotope[(24, 52)]])
    adjust_model_to_time_nearzero(a, indexofatomicnumber, indexofisotope)
    print("   t0 : ni56 ", a['isofrac'][s, indexofisotope[(28, 56)]], " co56 ", a['isofrac'][s, indexofisotope[(27, 56)]], " fe56 ", a['isofrac'][s, indexofisotope[(26, 56)]] ," sum ", a['isofrac'][s, indexofisotope[(28, 56)]] + a['isofrac'][s, indexofisotope[(27, 56)]] + a['isofrac'][s, indexofisotope[(26, 56)]])
    # print(" after: fe52 ", a['isofrac'][s, indexofisotope[(26, 52)]], " mn52 ", a['isofrac'][s, indexofisotope[(25, 52)]], " cr52 ", a['isofrac'][s, indexofisotope[(24, 52)]] ," sum ", a['isofrac'][s, indexofisotope[(26, 52)]] + a['isofrac'][s, indexofisotope[(25, 52)]] + a['isofrac'][s, indexofisotope[(24, 52)]])

    # The radii/velocity in the CMFGEN files are zone zentered, while in ARTIS they represent
    # the outer radius of a given zone. So we need to do a transformation
    r = a['rad'] * 1e10
    rmax = 0.5 * (r[:-1] + r[1:])
    rout = rmax
    rout = np.append(
        rout, r[-1]
    )  # cmfgen uses the radius of the outermost zone as the outer boundary
    rin = rmax
    rin = np.insert(
        rin, 0, 0
    )  # for artis we use 0 as inner radius for the innermost shell, cmfgen uses
    # the innermost radius r[0], this gives a slight discrepancy (<1%) in the total mass
    dm = 4 / 3 * np.pi * (rout**3 - rin**3) * a['dens'] / msun
    print(dm.sum(), dm.sum() / (a['dmass'].sum() / msun))  # Check total mass

    filetag = '_doubledecay' if use_double_decay else '_estimate'
    with open(model + '/model_t0' + filetag + '.txt', 'w') as f:
        f.write(str(a['nd']) + '\n')
        f.write(str(a['time']) + '\n')

        for i in range(a['nd']):
            vel = rout[i] / a['time'] / 3600 / 24 / 1e5  # a['vel'][i]
            rho = np.log10(a['dens'][i])
            igefrac = a['specfrac'][i, ige_index].sum()
            ni56frac = a['isofrac'][i, indexofisotope[(28, 56)]]
            co56frac = a['isofrac'][i, indexofisotope[(27, 56)]]
            fe52frac = a['isofrac'][i, indexofisotope[(26, 52)]]
            cr48frac = a['isofrac'][i, indexofisotope[(24, 48)]]
            strout = "{0:4d} {1:1.7e} {2:1.7e} {3:1.7e} {4:1.7e} {5:1.7e} {6:1.7e} {7:1.7e}\n".format(
                i + 1, vel, rho, igefrac, ni56frac, co56frac, fe52frac, cr48frac)
            f.write(strout)

    # Create an array of size n_radial_cells*31 (31=running index + 30 ARTIS species)
    abund = np.zeros((a['nd'], 31))
    # Fill the array with availble mass fractions and the running index
    for i in range(a['nspec'] - 1):
        abund[:, spectoz[i]] = a['specfrac'][:, i]
    for i in range(a['nd']):
        abund[i, 0] = i + 1
    # Write to file abundances.txt
    fmtstring = "%d " + "%1.7e " * 30
    np.savetxt(model + '/abundances_t0' + filetag + '.txt', abund[:, :], fmt=fmtstring)

    #
    #
    # M=0
    # for i in range(a['nd']):
    #     M+=dm[i]
    #     print(i+1,a['vel'][i],dm[i],M)


convert_model(model + '/' + snapshot)
