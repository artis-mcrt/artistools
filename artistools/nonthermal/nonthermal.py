#!/usr/bin/env python3
import math
import os

# import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import namedtuple
from scipy import linalg
from pathlib import Path
from math import atan
# import numba
# from numpy import arctan as atan

import artistools as at
import artistools.estimators
import artistools.nltepops
import artistools.nonthermal

# cgs units to match artis
EV = 1.6021772e-12  # in erg
H = 6.6260755e-27
ME = 9.1093897e-28
QE = 4.80325E-10
H_ionpot = 13.5979996 * EV
CLIGHT = 2.99792458e+10
PI = math.pi

use_collstrengths = False


def get_nntot(ions, ionpopdict):
    # total number density of all nuclei [cm^-3]
    nntot = 0.
    for Z, ionstage in ions:
        nntot += ionpopdict[(Z, ionstage)]
    return nntot


def get_nne(ions, ionpopdict):
    # number density of free electrons [cm-^3]
    nne = 0.
    for Z, ionstage in ions:
        charge = ionstage - 1
        assert(charge >= 0)
        nne += charge * ionpopdict[(Z, ionstage)]

    return nne


def get_Zbar(ions, ionpopdict):
    # number density-weighted average atomic number
    # i.e. protons per nucleus
    Zbar = 0.
    nntot = get_nntot(ions, ionpopdict)
    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        Zbar += Z * nnion / nntot

    return Zbar


def get_Zboundbar(ions, ionpopdict):
    # number density-weighted average number of bound electrons per nucleus
    Zboundbar = 0.
    nntot = get_nntot(ions, ionpopdict)
    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        Zboundbar += (Z - ionstage + 1) * nnion / nntot

    return Zboundbar


def get_nnetot(ions, ionpopdict):
    # total number density of electrons (free + bound) [cm-^3]
    # return get_Zbar(ions, ionpopdict) * get_nntot(ions, ionpopdict)
    nnetot = 0.
    for Z, ionstage in ions:
        nnetot += Z * ionpopdict[(Z, ionstage)]

    return nnetot


def read_binding_energies(modelpath='.'):
    collionfilename = at.firstexisting([os.path.join(modelpath, 'binding_energies.txt'),
                                        os.path.join(at.PYDIR, 'data', 'binding_energies.txt')])

    with open(collionfilename, "r") as f:
        nt_shells, n_z_binding = [int(x) for x in f.readline().split()]
        electron_binding = np.zeros((n_z_binding, nt_shells))

        for i in range(n_z_binding):
            electron_binding[i] = np.array([float(x) for x in f.readline().split()]) * EV

    return electron_binding


def get_electronoccupancy(atomic_number, ion_stage, nt_shells):
    # adapted from ARTIS code
    q = np.zeros(nt_shells)

    ioncharge = ion_stage - 1
    nbound = atomic_number - ioncharge  # number of bound electrons

    for electron_loop in range(nbound):
        if q[0] < 2:  # K 1s
            q[0] += 1
        elif(q[1] < 2):  # L1 2s
            q[1] += 1
        elif(q[2] < 2):  # L2 2p[1/2]
            q[2] += 1
        elif(q[3] < 4):  # L3 2p[3/2]
            q[3] += 1
        elif(q[4] < 2):  # M1 3s
            q[4] += 1
        elif(q[5] < 2):  # M2 3p[1/2]
            q[5] += 1
        elif(q[6] < 4):  # M3 3p[3/2]
            q[6] += 1
        elif ioncharge == 0:
            if q[9] < 2:  # N1 4s
                q[9] += 1
            elif q[7] < 4:  # M4 3d[3/2]
                q[7] += 1
            elif q[8] < 6:  # M5 3d[5/2]
                q[8] += 1
            else:
                print("Going beyond the 4s shell in NT calculation. Abort!\n")
        elif ioncharge == 1:
            if q[9] < 1:  # N1 4s
                q[9] += 1
            elif q[7] < 4:  # M4 3d[3/2]
                q[7] += 1
            elif q[8] < 6:  # M5 3d[5/2]
                q[8] += 1
            else:
                print("Going beyond the 4s shell in NT calculation. Abort!\n")
        elif(ioncharge > 1):
            if q[7] < 4:  # M4 3d[3/2]
                q[7] += 1
            elif q[8] < 6:  # M5 3d[5/2]
                q[8] += 1
            else:
                print("Going beyond the 4s shell in NT calculation. Abort!\n")
    return q


def get_mean_binding_energy(atomic_number, ion_stage, electron_binding, ionpot_ev):
    # LJS: this came from ARTIS and I'm not sure what this actually is - inverse binding energy? electrons per erg?
    n_z_binding, nt_shells = electron_binding.shape
    q = get_electronoccupancy(atomic_number, ion_stage, nt_shells)

    total = 0.0
    for electron_loop in range(nt_shells):
        electronsinshell = q[electron_loop]
        if ((electronsinshell) > 0):
            use2 = electron_binding[atomic_number - 1][electron_loop]
            use3 = ionpot_ev * EV
        if (use2 <= 0):
            use2 = electron_binding[atomic_number - 1][electron_loop - 1]
            # to get total += electronsinshell/electron_binding[get_element(element)-1][electron_loop-1];
            # set use3 = 0.
            if (electron_loop != 8):
                # For some reason in the Lotz data, this is no energy for the M5 shell before Ni. So if the complaint
                # is for 8 (corresponding to that shell) then just use the M4 value
                print("WARNING: I'm trying to use a binding energy when I have no data. "
                      f"element {atomic_number} ionstage {ion_stage}\n")
                assert(electron_loop == 8)
                # print("Z = %d, ion_stage = %d\n", get_element(element), get_ionstage(element, ion));
        if (use2 < use3):
            total += electronsinshell / use3
        else:
            total += electronsinshell / use2
        # print("total total)

    return total


def get_mean_binding_energy_alt(atomic_number, ion_stage, electron_binding, ionpot_ev):
    # LJS: this should be mean binding energy [erg] per electron
    n_z_binding, nt_shells = electron_binding.shape
    q = get_electronoccupancy(atomic_number, ion_stage, nt_shells)

    total = 0.0
    ecount = 0
    for electron_loop in range(nt_shells):
        electronsinshell = q[electron_loop]
        ecount += electronsinshell
        if ((electronsinshell) > 0):
            use2 = electron_binding[atomic_number - 1][electron_loop]
            use3 = ionpot_ev * EV
        if (use2 <= 0):
            use2 = electron_binding[atomic_number - 1][electron_loop - 1]
            # to get total += electronsinshell/electron_binding[get_element(element)-1][electron_loop-1];
            # set use3 = 0.
            if (electron_loop != 8):
                # For some reason in the Lotz data, this is no energy for the M5 shell before Ni. So if the complaint
                # is for 8 (corresponding to that shell) then just use the M4 value
                print("WARNING: I'm trying to use a binding energy when I have no data. "
                      f"element {atomic_number} ionstage {ion_stage}\n")
                assert(electron_loop == 8)
                # print("Z = %d, ion_stage = %d\n", get_element(element), get_ionstage(element, ion));
        if (use2 < use3):
            total += electronsinshell * use3
        else:
            total += electronsinshell * use2

    assert ecount == (atomic_number - ion_stage + 1)

    return total / ecount


def get_lotz_xs_ionisation(atomic_number, ion_stage, electron_binding, ionpot_ev, en_ev):
    # Axelrod 1980 Eq 3.38

    en_erg = en_ev * EV
    gamma = en_erg / (ME * CLIGHT ** 2) + 1
    beta = math.sqrt(1. - 1. / (gamma ** 2))
    # beta = 0.99
    # print(f'{gamma=} {beta=}')

    n_z_binding, nt_shells = electron_binding.shape
    q = get_electronoccupancy(atomic_number, ion_stage, nt_shells)

    part_sigma = 0.0
    for electron_loop in range(nt_shells):
        electronsinshell = q[electron_loop]
        if ((electronsinshell) > 0):
            use2 = electron_binding[atomic_number - 1][electron_loop]
            use3 = ionpot_ev * EV
        if (use2 <= 0):
            use2 = electron_binding[atomic_number - 1][electron_loop - 1]
            # to get total += electronsinshell/electron_binding[get_element(element)-1][electron_loop-1];
            # set use3 = 0.
            if (electron_loop != 8):
                # For some reason in the Lotz data, this is no energy for the M5 shell before Ni. So if the complaint
                # is for 8 (corresponding to that shell) then just use the M4 value
                print("WARNING: I'm trying to use a binding energy when I have no data. "
                      f"element {atomic_number} ionstage {ion_stage}\n")
                assert(electron_loop == 8)
                # print("Z = %d, ion_stage = %d\n", get_element(element), get_ionstage(element, ion));

        if (use2 < use3):
            p = use3
        else:
            p = use2

        if 0.5 * beta ** 2 * ME * CLIGHT ** 2 > p:
            part_sigma += electronsinshell / p * (
                (math.log(beta ** 2 * ME * CLIGHT ** 2 / 2. / p) - math.log10(1 - beta ** 2) - beta ** 2))

    Aconst = 1.33e-14 * EV * EV
    # me is electron mass
    sigma = 2 * Aconst / (beta ** 2) / ME / (CLIGHT ** 2) * part_sigma
    assert(sigma >= 0)
    return sigma


def lossfunction(energy_ev, nne_cgs):
    # free-electron plasma loss rate (as in Kozma & Fransson 1992)
    # - dE / dX [eV / cm]
    # returns a positive number

    # return math.log(energy_ev) / energy_ev
    if nne_cgs == 0:
        return 0.

    nne = nne_cgs
    energy = energy_ev * EV  # convert eV to erg

    # omegap = math.sqrt(4 * math.pi * nne_selected_cgs * pow(QE, 2) / ME)
    omegap = 5.64e4 * math.sqrt(nne_cgs)
    zetae = H * omegap / 2 / math.pi

    if energy_ev > 14:
        assert 2 * energy > zetae
        lossfunc = nne * 2 * math.pi * QE ** 4 / energy * math.log(2 * energy / zetae)
    else:
        v = math.sqrt(2 * energy / ME)  # velocity in m/s
        eulergamma = 0.577215664901532
        lossfunc = nne * 2 * math.pi * QE ** 4 / energy * math.log(ME * pow(v, 3) / (eulergamma * pow(QE, 2) * omegap))

    # lossfunc is now [erg / cm]
    return lossfunc / EV  # return as [eV / cm]


def Psecondary(e_p, ionpot_ev, J, e_s=-1, epsilon=-1):
    # probability distribution function for secondaries energy e_s [eV] (or equivalently the energy loss of
    # the primary electron epsilon [eV]) given a primary energy e_p [eV] for an impact ionisation event

    assert e_s >= 0 or epsilon >= 0
    # if e_p < I:
    #     return 0.
    #
    if e_s < 0:
        e_s = epsilon - ionpot_ev
    if epsilon < 0:
        epsilon = e_s + ionpot_ev

    #
    # if epsilon < I:
    #     return 0.
    # if e_s < 0:
    #     return 0.
    # if e_s > e_p - I:
    #     return 0.
    # if e_s > e_p:
    #     return 0.

    # test case: constant, always below ionisation
    # Psecondary_e_s_max = 1. / J / 2.
    # return 1. / Psecondary_e_s_max if (e_s < Psecondary_e_s_max) else 0.

    return 1. / J / atan((e_p - ionpot_ev) / 2. / J) / (1 + ((e_s / J) ** 2))


def get_J(Z, ionstage, ionpot_ev):
    # returns an energy in eV
    # values from Opal et al. 1971 as applied by Kozma & Fransson 1992
    if (ionstage == 1):
        if (Z == 2):  # He I
            return 15.8
        elif (Z == 10):  # Ne I
            return 24.2
        elif (Z == 18):  # Ar I
            return 10.0

    return 0.6 * ionpot_ev
    # return 2.0 * ionpot_ev
    # return 16.0 * ionpot_ev
    # return 500


def ar_xs(energy_ev, ionpot_ev, A, B, C, D):
    u = energy_ev / ionpot_ev
    if u <= 1:
        return 0

    return 1e-14 * (
        A * (1 - 1 / u) + B * pow((1 - 1 / u), 2) + C * math.log(u) + D * math.log(u) / u) / (u * pow(ionpot_ev, 2))


def get_arxs_array_shell(arr_enev, shell):
    ar_xs_array = np.array(
        [ar_xs(energy_ev, shell.ionpot_ev, shell.A, shell.B, shell.C, shell.D) for energy_ev in arr_enev])

    return ar_xs_array


def get_arxs_array_ion(arr_enev, dfcollion, Z, ionstage):
    ar_xs_array = np.zeros(len(arr_enev))
    dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage')
    for index, shell in dfcollion_thision.iterrows():
        ar_xs_array += get_arxs_array_shell(arr_enev, shell)

    return ar_xs_array


def get_xs_excitation(en_ev, row):
    A_naught_squared = 2.800285203e-17  # Bohr radius squared in cm^2

    coll_str = row.collstr
    epsilon_trans = row.epsilon_trans_ev * EV
    epsilon_trans_ev = row.epsilon_trans_ev

    if en_ev < epsilon_trans_ev:
        return 0.

    if coll_str >= 0 and use_collstrengths:
        # collision strength is available, so use it
        # Li et al. 2012 equation 11
        constantfactor = pow(H_ionpot, 2) / row.lower_g * coll_str * math.pi * A_naught_squared

        return constantfactor * (en_ev * EV) ** -2

    elif not row.forbidden:

        nu_trans = epsilon_trans / H
        g = row.upper_g / row.lower_g
        fij = g * ME * pow(CLIGHT, 3) / (8 * pow(QE * nu_trans * math.pi, 2)) * row.A
        # permitted E1 electric dipole transitions

        g_bar = 0.2

        A = 0.28
        B = 0.15

        prefactor = 45.585750051
        # Eq 4 of Mewe 1972, possibly from Seaton 1962?
        constantfactor = prefactor * A_naught_squared * pow(H_ionpot / epsilon_trans, 2) * fij

        U = en_ev / epsilon_trans_ev
        g_bar = A * np.log(U) + B

        return constantfactor * g_bar / U

    return 0.


def get_xs_excitation_vector(engrid, row):
    A_naught_squared = 2.800285203e-17  # Bohr radius squared in cm^2
    npts = len(engrid)
    xs_excitation_vec = np.empty(npts)

    coll_str = row.collstr
    epsilon_trans = row.epsilon_trans_ev * EV
    epsilon_trans_ev = row.epsilon_trans_ev

    startindex = get_energyindex_gteq(en_ev=epsilon_trans_ev, engrid=engrid)
    xs_excitation_vec[:startindex] = 0.

    if coll_str >= 0 and use_collstrengths:
        # collision strength is available, so use it
        # Li et al. 2012 equation 11
        constantfactor = pow(H_ionpot, 2) / row.lower_g * coll_str * math.pi * A_naught_squared

        xs_excitation_vec[startindex:] = constantfactor * (engrid[startindex:] * EV) ** -2

    elif not row.forbidden:

        nu_trans = epsilon_trans / H
        g = row.upper_g / row.lower_g
        fij = g * ME * pow(CLIGHT, 3) / (8 * pow(QE * nu_trans * math.pi, 2)) * row.A
        # permitted E1 electric dipole transitions

        g_bar = 0.2

        A = 0.28
        B = 0.15

        prefactor = 45.585750051
        # Eq 4 of Mewe 1972, possibly from Seaton 1962?
        constantfactor = prefactor * A_naught_squared * pow(H_ionpot / epsilon_trans, 2) * fij

        U = engrid[startindex:] / epsilon_trans_ev
        g_bar = A * np.log(U) + B

        xs_excitation_vec[startindex:] = constantfactor * g_bar / U
        for j, energy_ev in enumerate(engrid):
            energy = energy_ev * EV
            if (energy >= epsilon_trans):
                U = energy / epsilon_trans
                g_bar = A * math.log(U) + B
                xs_excitation_vec[j] = constantfactor * g_bar / U
    else:
        xs_excitation_vec[startindex:] = 0.

    return xs_excitation_vec


def read_colliondata(collionfilename='collion.txt', modelpath=Path(at.PYDIR, 'data')):

    collionrow = namedtuple('collionrow', ['Z', 'nelec', 'n', 'l', 'ionpot_ev', 'A', 'B', 'C', 'D'])

    nrows = -1
    with open(Path(modelpath, collionfilename), 'r') as collionfile:
        nrows = int(collionfile.readline().strip())
        # print(f'Collionfile: expecting {nrows} rows')
        dfcollion = pd.read_csv(
            collionfile, delim_whitespace=True, header=None, names=collionrow._fields)

    # assert len(dfcollion) == nrows  # artis enforces this, but last 10 rows were not inportant anyway (high ionized Ni)
    dfcollion.eval('ionstage = Z - nelec + 1', inplace=True)

    return dfcollion


def calculate_nt_frac_excitation(engrid, dftransitions, yvec, deposition_density_ev):
    # Kozma & Fransson equation 4, but summed over all transitions for given ion
    # integral in Kozma & Fransson equation 9
    deltaen = engrid[1] - engrid[0]
    npts = len(engrid)

    xs_excitation_vec_sum_alltrans = np.zeros(npts)

    for _, row in dftransitions.iterrows():
        nnlevel = row.lower_pop
        xs_excitation_vec_sum_alltrans += nnlevel * row.epsilon_trans_ev * get_xs_excitation_vector(engrid, row)

    return np.dot(xs_excitation_vec_sum_alltrans, yvec) * deltaen / deposition_density_ev


def get_energyindex_lteq(en_ev, engrid):
    # find energy bin lower boundary is less than or equal to search value
    # assert en_ev >= engrid[0]
    deltaen = engrid[1] - engrid[0]
    # assert en_ev < (engrid[-1] + deltaen)

    index = math.floor((en_ev - engrid[0]) / deltaen)

    if index < 0:
        return 0
    elif (index > len(engrid) - 1):
        return len(engrid) - 1
    else:
        return index


def get_energyindex_gteq(en_ev, engrid):
    # find energy bin lower boundary is greater than or equal to search value
    deltaen = engrid[1] - engrid[0]

    index = math.ceil((en_ev - engrid[0]) / deltaen)

    if index < 0:
        return 0
    elif (index > len(engrid) - 1):
        return len(engrid) - 1
    else:
        return index


def calculate_N_e(energy_ev, engrid, ions, ionpopdict, dfcollion, yvec, dftransitions, noexcitation):
    # Kozma & Fransson equation 6.
    # Something related to a number of electrons, needed to calculate the heating fraction in equation 3
    # not valid for energy > E_0
    if energy_ev == 0.:
        return 0.

    N_e = 0.

    deltaen = engrid[1] - engrid[0]

    for Z, ionstage in ions:
        N_e_ion = 0.
        nnion = ionpopdict[(Z, ionstage)]

        if not noexcitation:
            if (Z, ionstage) in dftransitions:
                for _, row in dftransitions[(Z, ionstage)].iterrows():
                    nnlevel = row.lower_pop
                    epsilon_trans_ev = row.epsilon_trans_ev
                    if energy_ev + epsilon_trans_ev >= engrid[0]:
                        i = get_energyindex_lteq(en_ev=energy_ev + epsilon_trans_ev, engrid=engrid)
                        N_e_ion += (nnlevel / nnion) * yvec[i] * get_xs_excitation(engrid[i], row)
                        # enbelow = engrid[i]
                        # enabove = engrid[i + 1]
                        # x = (energy_ev - enbelow) / (enabove - enbelow)
                        # yvecinterp = (1 - x) * yvec[i] + x * yvec[i + 1]
                        # N_e_ion += (nnlevel / nnion) * yvecinterp * get_xs_excitation(energy_ev + epsilon_trans_ev, row)

        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)

        for index, shell in dfcollion_thision.iterrows():
            ionpot_ev = shell.ionpot_ev

            enlambda = min(engrid[-1] - energy_ev, energy_ev + ionpot_ev)
            J = get_J(shell.Z, shell.ionstage, ionpot_ev)

            ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, shell)

            # integral from ionpot to enlambda
            integral1startindex = get_energyindex_lteq(en_ev=ionpot_ev, engrid=engrid)
            integral2stopindex = get_energyindex_lteq(en_ev=enlambda, engrid=engrid)

            for j in range(integral1startindex, integral2stopindex + 1):
                endash = engrid[j]
                k = get_energyindex_lteq(en_ev=energy_ev + endash, engrid=engrid)
                N_e_ion += deltaen * yvec[k] * ar_xs_array[k] * Psecondary(e_p=engrid[k], epsilon=endash, ionpot_ev=ionpot_ev, J=J)

                # interpolate the y value
                # xs = ar_xs(energy_ev + endash, shell.ionpot_ev, shell.A, shell.B, shell.C, shell.D)
                # enbelow = engrid[k]
                # enabove = engrid[k + 1]
                # x = (energy_ev - enbelow) / (enabove - enbelow)
                # yvecinterp = (1 - x) * yvec[k] + x * yvec[k + 1]
                # N_e_ion += deltaen * yvecinterp * xs * Psecondary(e_p=energy_ev + endash, epsilon=endash, ionpot_ev=ionpot_ev, J=J)

            # integral from 2E + I up to E_max
            integral2startindex = get_energyindex_lteq(en_ev=2 * energy_ev + ionpot_ev, engrid=engrid)
            N_e_ion += deltaen * sum([
                yvec[j] * ar_xs_array[j] *
                Psecondary(e_p=engrid[j], epsilon=energy_ev + ionpot_ev, ionpot_ev=ionpot_ev, J=J)
                for j in range(integral2startindex, len(engrid))])

        N_e += nnion * N_e_ion

    # source term not here because it should be zero at the low end anyway

    return N_e


def calculate_frac_heating(
        engrid, ions, ionpopdict, dfcollion, dftransitions, yvec, nne, nnetot, deposition_density_ev, noexcitation):
    # Kozma & Fransson equation 8

    frac_heating = 0.
    E_0 = engrid[0]

    deltaen = engrid[1] - engrid[0]
    frac_heating += deltaen / deposition_density_ev * sum([
        lossfunction(en_ev, nne, nnetot, ions=ions, ionpopdict=ionpopdict) * yvec[i]
        for i, en_ev in enumerate(engrid)])

    frac_heating_E_0_part = (
        E_0 * yvec[0] * lossfunction(E_0, nne, nnetot, ions=ions, ionpopdict=ionpopdict) / deposition_density_ev)

    frac_heating += frac_heating_E_0_part

    print(f"            frac_heating E_0 * y * l(E_0) part: {frac_heating_E_0_part:.5f}")

    frac_heating_N_e = 0.
    npts_integral = math.ceil(E_0 / deltaen) * 10
    print(f'N_e npts_integral: {npts_integral}')
    arr_en, deltaen2 = np.linspace(0., E_0, num=npts_integral, retstep=True, endpoint=True)
    arr_en_N_e = [en_ev * calculate_N_e(en_ev, engrid, ions, ionpopdict, dfcollion,
                                        yvec, dftransitions, noexcitation=noexcitation) for en_ev in arr_en]
    frac_heating_N_e += 1. / deposition_density_ev * sum(arr_en_N_e) * deltaen2

    print(f"            frac_heating N_e part: {frac_heating_N_e:.5f}")

    frac_heating += frac_heating_N_e

    return frac_heating


def sfmatrix_add_excitation(engrid, dftransitions_ion, nnion, sfmatrix):
    deltaen = engrid[1] - engrid[0]
    for _, row in dftransitions_ion.iterrows():
        nnlevel = row.lower_pop
        epsilon_trans_ev = row.epsilon_trans_ev
        if epsilon_trans_ev >= engrid[0]:
            vec_xs_excitation_nnlevel_deltae = nnlevel * deltaen * get_xs_excitation_vector(engrid, row)
            xsstartindex = get_energyindex_lteq(en_ev=epsilon_trans_ev, engrid=engrid)

            for i, en in enumerate(engrid):
                stopindex = get_energyindex_lteq(en_ev=en + epsilon_trans_ev, engrid=engrid)

                startindex = i if i > xsstartindex else xsstartindex
                # for j in range(startindex, stopindex):
                sfmatrix[i, startindex:stopindex] += vec_xs_excitation_nnlevel_deltae[startindex:stopindex]

                # do the last bit separately because we're not using the full deltaen interval

                delta_en_actual = (en + epsilon_trans_ev - engrid[stopindex])
                sfmatrix[i, stopindex] += vec_xs_excitation_nnlevel_deltae[stopindex] * delta_en_actual / deltaen


def sfmatrix_add_ionization_shell(engrid, nnion, shell, sfmatrix):
    # this code has been optimised and is now an almost unreadable form, but it contains the terms
    # related to ionisation cross sections
    deltaen = engrid[1] - engrid[0]
    ionpot_ev = shell.ionpot_ev
    J = get_J(shell.Z, shell.ionstage, ionpot_ev)
    npts = len(engrid)

    ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, shell)

    if ionpot_ev <= engrid[0]:
        xsstartindex = 0
    else:
        xsstartindex = npts + 1
        for i in range(npts):
            if ar_xs_array[i] > 0.:
                xsstartindex = i
                break
        xsstartindex = get_energyindex_gteq(en_ev=ionpot_ev, engrid=engrid)

    # J * atan[(epsilon - ionpot_ev) / J] is the indefinite integral of
    # 1/(1 + (epsilon - ionpot_ev)^2/ J^2) d_epsilon
    # in Kozma & Fransson 1992 equation 4

    prefactors = [nnion * ar_xs_array[j] / atan((engrid[j] - ionpot_ev) / 2. / J) * deltaen for j in range(npts)]
    epsilon_uppers = [min((engrid[j] + ionpot_ev) / 2, engrid[j]) for j in range(npts)]
    int_eps_uppers = [atan((epsilon_upper - ionpot_ev) / J) for epsilon_upper in epsilon_uppers]

    # for the resulting arrays, use index j - i corresponding to energy endash - en
    epsilon_lowers1 = [max(engrid[j] - engrid[0], ionpot_ev) for j in range(npts)]
    int_eps_lowers1 = [atan((epsilon_lower - ionpot_ev) / J) for epsilon_lower in epsilon_lowers1]

    for i, en in enumerate(engrid):
        # endash ranges from en to SF_EMAX, but skip over the zero-cross section points
        jstart = max(i, xsstartindex)

        # KF 92 limit
        # at each endash, the integral in epsilon ranges from
        # epsilon_lower = max(endash - en, ionpot_ev)
        # epsilon_upper = min((endash + ionpot_ev) / 2, endash)]

        # integral/J of 1/[1 + (epsilon - ionpot_ev) / J] for epsilon = en + ionpot_ev
        for j in range(jstart, npts):
            # j is the matrix column index which corresponds to the piece of the
            # integral at y(E') where E' >= E and E' = engrid[j]

            if epsilon_lowers1[j - i] <= epsilon_uppers[j]:
                sfmatrix[i, j] += prefactors[j] * (int_eps_uppers[j] - int_eps_lowers1[j - i])

        if 2 * en + ionpot_ev < engrid[-1] + (engrid[1] - engrid[0]):
            secondintegralstartindex = get_energyindex_lteq(2 * en + ionpot_ev, engrid)
        else:
            secondintegralstartindex = npts + 1

        # endash ranges from 2 * en + ionpot_ev to SF_EMAX
        # at each endash, the integral in epsilon ranges from
        # epsilon_lower = en + ionpot_ev
        # epsilon_upper = min((endash + ionpot_ev) / 2, endash)]
        epsilon_lower2 = en + ionpot_ev
        for j in range(secondintegralstartindex, npts):
            if epsilon_lower2 <= epsilon_uppers[j]:
                int_eps_lower2 = atan((epsilon_lower2 - ionpot_ev) / J)
                sfmatrix[i, j] -= prefactors[j] * (int_eps_uppers[j] - int_eps_lower2)


def differentialsfmatrix_add_ionization_shell(engrid, nnion, shell, sfmatrix):
    # this code has been optimised and is now an almost unreadable form, but it is the contains the terms
    # related to ionisation cross sections
    delta_en = engrid[1] - engrid[0]
    ionpot_ev = shell.ionpot_ev
    J = get_J(shell.Z, shell.ionstage, ionpot_ev)
    npts = len(engrid)

    ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, shell)

    if ionpot_ev <= engrid[0]:
        xsstartindex = 0
    else:
        xsstartindex = get_energyindex_lteq(en_ev=ionpot_ev, engrid=engrid)

    oneoveratangrid = 1. / np.arctan((engrid - ionpot_ev) / 2. / J)

    epsilon_lower_a = ionpot_ev
    int_eps_lower_a = atan((epsilon_lower_a - ionpot_ev) / J)
    for i in range(xsstartindex, npts):
        en = engrid[i]

        # integral of xs_ion(e_p=en, epsilon) with epsilon from I to (I + E) / 2

        epsilon_upper = (ionpot_ev + en) / 2.
        if (epsilon_lower_a < epsilon_upper):
            # P_int = 0.
            # eps_npts = 1000
            # delta_eps = (epsilon_upper - epsilon_lower) / eps_npts
            # for j in range(eps_npts):
            #     epsilon = epsilon_lower + j * delta_eps
            #     P_int += Psecondary(e_p=en, epsilon=epsilon, ionpot_ev=ionpot_ev, J=J) * delta_eps
            #
            # J * atan[(epsilon - ionpot_ev) / J] is the indefinite integral of
            # 1/(1 + (epsilon - ionpot_ev)^2/ J^2) d_epsilon

            int_eps_upper = atan((epsilon_upper - ionpot_ev) / J)
            P_int = 1. / atan((en - ionpot_ev) / 2. / J) * (int_eps_upper - int_eps_lower_a)
            # if int_eps_lower == int_eps_upper and epsilon_upper != epsilon_lower:
            # if (abs(P_int2 / P_int - 1.) > 0.2):
            #     print("warning eps low high int low high", epsilon_lower, epsilon_upper, int_eps_lower, int_eps_upper)
            #     print(f'{P_int=:.2e} {P_int2=:.2e} Ratio: {P_int2 / P_int:.2f}')

            sfmatrix[i, i] += nnion * ar_xs_array[i] * P_int

        enlambda = min(engrid[-1] - en, en + ionpot_ev)
        epsilon_lower = ionpot_ev
        epsilon_upper = enlambda
        if (epsilon_lower < epsilon_upper):
            eps_npts = 100
            delta_eps = (epsilon_upper - epsilon_lower) / eps_npts
            prefactor = nnion / J / atan((en - ionpot_ev) / 2. / J) * delta_eps
            for j in range(eps_npts):
                epsilon = epsilon_lower + j * delta_eps
                i_enpluseps = get_energyindex_lteq(en + epsilon, engrid=engrid)
                sfmatrix[i, i_enpluseps] -= prefactor * ar_xs_array[i_enpluseps] / (
                    1 + (((epsilon - ionpot_ev) / J) ** 2))

            # j_lower = get_energyindex_lteq(en + epsilon_lower, engrid=engrid)
            # j_upper = get_energyindex_lteq(en + epsilon_upper, engrid=engrid)
            # if (j_lower < j_upper):
            #     delta_eps = (epsilon_upper - epsilon_lower) / (j_upper - j_lower)
            #     prefactor = nnion / J / atan((en - ionpot_ev) / 2. / J) * delta_eps
            #     print(J, atan((en - ionpot_ev) / 2. / J))
            #     assert not math.isnan(prefactor)
            #     assert not math.isinf(prefactor)
            #     # for j in range(j_lower, j_upper):
            #     #     en_plus_epsilon = engrid[j]
            #     #     epsilon = en_plus_epsilon - en
            #     #     sfmatrix[i, j] -= prefactor * ar_xs_array[j] / (1 + (((epsilon - ionpot_ev) / J) ** 2))
            #     sfmatrix[i, j_lower:j_upper] -= prefactor * ar_xs_array[j_lower:j_upper] / (1 + ((((engrid[j_lower:j_upper] - en) - ionpot_ev) / J) ** 2))

        if (2 * en + ionpot_ev) < engrid[-1]:
            epsilon = en + ionpot_ev
            prefactor = nnion / J / (1 + (((epsilon - ionpot_ev) / J) ** 2)) * delta_en
            i_endash_lower = get_energyindex_lteq(2 * en + ionpot_ev, engrid)
            # for j in range(i_endash_lower, npts):
            #     sfmatrix[i, j] -= prefactor * ar_xs_array[j] * oneoveratangrid[j]
            sfmatrix[i, i_endash_lower:] -= prefactor * ar_xs_array[i_endash_lower:] * oneoveratangrid[i_endash_lower:]


def solve_spencerfano_differentialform(
        ions, ionpopdict, dfpops, nne, deposition_density_ev, engrid, sourcevec, dfcollion, args,
        adata=None, noexcitation=False):

    deltaen = engrid[1] - engrid[0]
    npts = len(engrid)
    nnetot = get_nnetot(ions, ionpopdict)

    print(f'\nSetting up differential-form Spencer-Fano equation with {npts} energy points'
          f' from {engrid[0]} to {engrid[-1]} eV...')

    E_init_ev = np.dot(engrid, sourcevec) * deltaen
    print(f'    E_init: {E_init_ev:7.2f} eV/s/cm3')

    constvec = np.zeros(npts)
    for i in range(npts):
        constvec[i] += sourcevec[i]

    lossfngrid = np.array([lossfunction(en, nne, nnetot, ions=ions, ionpopdict=ionpopdict) for en in engrid])

    sfmatrix = np.zeros((npts, npts))
    for i in range(npts):
        en = engrid[i]

        # - d/dE(y[E] * lossfn[E]) = - dy/dE(E) * lossfn(E) - y(E) * dlossfn/dE

        # - dy/dE(E) * lossfn(E) = - (y(E + deltaE) - y(E) ) / deltaen * lossfunction(E)

        # - ( - y(E) * lossfunction)
        sfmatrix[i, i] += lossfngrid[i] / deltaen

        if i + 1 < npts:
            # - y(E + deltaE) * lossfunction
            sfmatrix[i, i + 1] -= lossfngrid[i] / deltaen

        # - y(E) * dlossfn/dE
        sfmatrix[i, i] -= (lossfunction(en + deltaen, nne, nnetot, ions=ions, ionpopdict=ionpopdict) - lossfngrid[i]) / deltaen

    dftransitions = {}
    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        print(f'  including Z={Z} ion_stage {ionstage} ({at.get_ionstring(Z, ionstage)}) nnion={nnion:.2e} ionisation')
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)
        # print(dfcollion_thision)

        for index, shell in dfcollion_thision.iterrows():
            # assert shell.ionpot_ev >= engrid[0]
            if shell.ionpot_ev < engrid[0]:
                print(f'  WARNING: first energy point at {engrid[0]} eV is above shell ionpot {shell.ionpot_ev} eV')
            differentialsfmatrix_add_ionization_shell(engrid, nnion, shell, sfmatrix)

        assert noexcitation

    print()
    lu_and_piv = linalg.lu_factor(sfmatrix, overwrite_a=False)
    yvec_reference = linalg.lu_solve(lu_and_piv, constvec, trans=0)
    yvec = yvec_reference * deposition_density_ev / E_init_ev

    return yvec, dftransitions


def get_nne_nt(engrid, yvec):
    # oneovervelocity = np.sqrt(9.10938e-31 / 2 / engrid / 1.60218e-19) / 100  # in s/cm
    # enovervelocity = engrid * oneovervelocity
    # en_tot = np.dot(yvec, enovervelocity) * (engrid[1] - engrid[0])
    nne_nt = 0.
    deltaen = (engrid[1] - engrid[0])
    for i, en in enumerate(engrid):
        # oneovervelocity = np.sqrt(9.10938e-31 / 2 / en / 1.60218e-19) / 100.
        velocity = np.sqrt(2 * en * 1.60218e-19 / 9.10938e-31) * 100.  # cm/s
        nne_nt += yvec[i] / velocity * deltaen

    return nne_nt


def analyse_ntspectrum(
        engrid, yvec, ions, ionpopdict, nntot, nne, deposition_density_ev, dfcollion, dftransitions,
        noexcitation, modelpath):

    deltaen = engrid[1] - engrid[0]

    frac_ionization = 0.
    frac_excitation = 0.
    frac_ionization_ion = {}
    frac_excitation_ion = {}
    gamma_nt = {}

    electron_binding = read_binding_energies(modelpath)

    Zbar = get_Zbar(ions, ionpopdict)
    nnetot = get_nnetot(ions, ionpopdict)

    Aconst = 1.33e-14 * EV * EV

    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        if nnion == 0.:
            print(f'skipping Z={Z:2d} ion_stage {ionstage} due to nnion = {nnion}')
            continue
        X_ion = nnion / nntot
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)
        # if dfcollion.empty:
        #     continue
        ionpot_valence = dfcollion_thision.ionpot_ev.min()

        print(f'====> Z={Z:2d} ion_stage {ionstage} {at.get_ionstring(Z, ionstage)} (valence potential {ionpot_valence:.1f} eV)')

        print(f'               nnion: {nnion:.2e} [/cm3]')
        print(f'         nnion/nntot: {X_ion:.5f}')

        frac_ionization_ion[(Z, ionstage)] = 0.
        # integralgamma = 0.
        eta_over_ionpot_sum = 0.
        for index, shell in dfcollion_thision.iterrows():
            ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, shell)

            frac_ionization_shell = (
                nnion * shell.ionpot_ev * np.dot(yvec, ar_xs_array) * deltaen / deposition_density_ev)
            print(f'frac_ionization_shell(n {int(shell.n):d} l {int(shell.l):d}): '
                  f'{frac_ionization_shell:.4f} (ionpot {shell.ionpot_ev:.2f} eV)')

            # integralgamma += np.dot(yvec, ar_xs_array) * deltaen * shell.ionpot_ev / ionpot_valence

            if frac_ionization_shell > 1:
                frac_ionization_shell = 0.
                print(f'Ignoring frac_ionization_shell of {frac_ionization_shell}.')
                # for k in range(10):
                #     print(nnion * shell.ionpot_ev * yvec_reference[k] * ar_xs_array[k] * deltaen / E_init_ev)

            frac_ionization_ion[(Z, ionstage)] += frac_ionization_shell
            eta_over_ionpot_sum += frac_ionization_shell / shell.ionpot_ev

        frac_ionization += frac_ionization_ion[(Z, ionstage)]

        eff_ionpot_2 = X_ion / eta_over_ionpot_sum if eta_over_ionpot_sum else float('inf')

        eff_ionpot = (
            ionpot_valence * X_ion / frac_ionization_ion[(Z, ionstage)]
            if frac_ionization_ion[(Z, ionstage)] > 0. else float('inf'))

        print(f'     frac_ionization: {frac_ionization_ion[(Z, ionstage)]:.4f}')
        if not noexcitation:
            if nnion > 0.:
                frac_excitation_ion[(Z, ionstage)] = calculate_nt_frac_excitation(
                    engrid, dftransitions[(Z, ionstage)], yvec, deposition_density_ev)
            else:
                frac_excitation_ion[(Z, ionstage)] = 0.

            if frac_excitation_ion[(Z, ionstage)] > 1:
                frac_excitation_ion[(Z, ionstage)] = 0.
                print(f'Ignoring invalid frac_excitation_ion of {frac_excitation_ion[(Z, ionstage)]}.')
            frac_excitation += frac_excitation_ion[(Z, ionstage)]
            print(f'     frac_excitation: {frac_excitation_ion[(Z, ionstage)]:.4f}')
        else:
            frac_excitation_ion[(Z, ionstage)] = 0.

        print(f' eff_ionpot_shellpot: {eff_ionpot_2:.2f} [eV]')
        print(f'  eff_ionpot_valence: {eff_ionpot:.2f} [eV]')
        gamma_nt[(Z, ionstage)] = deposition_density_ev / nntot / eff_ionpot
        print(f'  Spencer-Fano Gamma: {gamma_nt[(Z, ionstage)]:.2e} [/s]')
        # print(f'Alternative Gamma: {integralgamma:.2e}')

        binding = get_mean_binding_energy(Z, ionstage, electron_binding, ionpot_ev=ionpot_valence)  # binding in erg
        oneoverW = Aconst * binding / Zbar / (2 * 3.14159 * pow(QE, 4))  # per erg
        deposition_density_erg = deposition_density_ev * EV

        # to get the non-thermal ionization rate we need to divide the energy deposited
        # per unit volume per unit time in the grid cell (sum of terms above)
        # by the total ion number density and the "work per ion pair"
        print(f"       work function: {1. / oneoverW / EV:.2f} [eV]")
        print(f"   work fn ratecoeff: {deposition_density_erg / nntot * oneoverW:.2e}")
        # Axelrod 1980 Eq 3.225 with E0 = E = E_max
        # xs = lossfunction(engrid[-1], nne, nnetot) * EV * oneoverW / nntot
        # print(f"         WFApprox xs: {xs:.2e} cm^2")

        print()

    # nne_nt = get_nne_nt(engrid, yvec)
    # print(f'               nne_nt: {nne_nt:.2e} /s/cm3')

    print(f'  frac_excitation_tot: {frac_excitation:.4f}')
    print(f'  frac_ionization_tot: {frac_ionization:.4f}')

    frac_heating = calculate_frac_heating(
        engrid, ions, ionpopdict, dfcollion, dftransitions, yvec, nne, nnetot, deposition_density_ev,
        noexcitation=noexcitation)

    print(f'         frac_heating: {frac_heating:.4f}')
    print(f'             frac_sum: {frac_excitation + frac_ionization + frac_heating:.4f}')

    return frac_excitation, frac_ionization, frac_heating, frac_excitation_ion, frac_ionization_ion, gamma_nt


def get_Latom_axelrod(Zboundbar, en_ev):
    # Axelrod 1980 Eq 3.21
    # Latom is 1/N * dE/dX where E is in erg
    # should be units of erg cm^2

    en_erg = en_ev * EV

    # relativistic
    gamma = en_erg / (ME * CLIGHT ** 2) + 1
    beta = math.sqrt(1. - 1. / (gamma ** 2))
    vel = beta * CLIGHT  # in cm/s

    # classical
    # vel = math.sqrt(2. * en_erg / ME)
    # beta = vel / CLIGHT

    # I = ionpot_ev * EV
    I = 280 * EV  # assumed in Axelrod thesis

    if 2 * ME * vel ** 2 < I:
        return 0.

    # if beta > 1.:
    #     print(vel, beta)
    #     beta = 0.9999

    return 4 * math.pi * QE ** 4 / (ME * vel ** 2) * Zboundbar * (
        math.log(2 * ME * vel ** 2 / I) + math.log(1. / (1. - beta ** 2)) - beta ** 2)


def get_Lelec_axelrod(en_ev, nne, nnetot, nntot):
    # - 1/N * dE / dX [erg cm^2]
    # returns a positive number

    # Axelrod Eq 3.36 (classical low energy limit)

    # return 1.95e-13 * math.log(3.2e4 * en_ev) / en_ev

    # Axelrod 1980 Eq 3.24

    HBAR = H / 2. / math.pi
    en_erg = en_ev * EV
    gamma = en_erg / (ME * CLIGHT ** 2) + 1
    beta = math.sqrt(1. - 1. / (gamma ** 2))
    vel = beta * CLIGHT  # in cm/s
    omegap = 5.64e4 * math.sqrt(nne)  # in per second
    return 4 * math.pi * QE ** 4 / (ME * vel ** 2) * nne / nntot * (
        math.log(2 * ME * vel ** 2 / (HBAR * omegap)) + 0.5 * math.log(1. / (1. - beta ** 2)) - 0.5 * beta ** 2)


def lossfunction_axelrod(en_ev, nne, nnetot):
    # - dE / dX [erg / cm]
    # returns a positive number

    return get_Lelec_axelrod(en_ev, nne=nne, nnetot=nnetot, nntot=1)


def get_fij_ln_en_ionisation(emax_ev, J, shell):
    npts = 1000
    e_p_lower = shell.ionpot_ev
    e_p_upper = emax_ev
    delta_e_p = (e_p_upper - e_p_lower) / npts
    sum = 0.
    for i in range(npts):
        e_p = e_p_lower + i * delta_e_p
        print(i, e_p)

        sigma = at.nonthermal.ar_xs(e_p, shell.ionpot_ev, shell.A, shell.B, shell.C, shell.D)
        eps_avg = get_epsilon_avg(e_p, J, shell.ionpot_ev)
        if eps_avg > 0:
            sum += ME * CLIGHT / math.pi / (QE ** 2) / H * sigma * math.log(eps_avg) * delta_e_p

    return sum


def e_s_test(ax, ionpot_ev, J, arr_en_ev, shellstr, color):
    e_p = arr_en_ev[-1]
    npts = 100000
    # ionpot_ev = 280

    J = ionpot_ev * 0.6
    # epsilon_lower = ionpot_ev
    # epsilon_upper = (ionpot_ev + e_p) / 2.
    # delta_eps = (epsilon_upper - epsilon_lower) / npts
    # print(f'{delta_eps=}')
    # prob_sum = 0.
    # eps_avg = 0.
    # prob = np.zeros(npts)
    # epsilon = np.zeros(npts)
    # for i in range(npts):
    #     epsilon[i] = epsilon_lower + i * delta_eps
    #     prob[i] = Psecondary(e_p, ionpot_ev, J, epsilon=epsilon[i]) * delta_eps
    #     prob_sum += prob[i]
    #     eps_avg = epsilon[i] * prob[i]
    # print(f'{prob_sum=:.1e}')
    # print(f'{eps_avg=:.1e}')
    # # ax.plot(epsilon, prob, label=shellstr + ' dP / d_epsilon')

    e_s_lower = 0
    e_s_upper = (e_p - ionpot_ev) / 2.
    delta_e_s = (e_s_upper - e_s_lower) / npts
    prob_sum = 0.
    e_s_avg = 0.
    prob = np.zeros(npts)
    e_s = np.zeros(npts)
    prob_e_s_can_ionise = 0.
    prob_e_s_cannot_ionise = 0.
    for i in range(npts):
        e_s[i] = e_s_lower + i * delta_e_s
        prob[i] = Psecondary(e_p, ionpot_ev, J, e_s=e_s[i]) * delta_e_s
        prob_sum += prob[i]
        e_s_avg = e_s[i] * prob[i]
        if e_s[i] >= ionpot_ev:
            prob_e_s_can_ionise += prob[i]
        else:
            prob_e_s_cannot_ionise += prob[i]

    print(f'prob_sum={prob_sum:.2f}')
    print(f'e_s_avg={e_s_avg:.1e}')
    print(f'prob_e_s_can_ionise={prob_e_s_can_ionise:.2f}')
    print(f'prob_e_s_cannot_ionise={prob_e_s_cannot_ionise:.2f}')
    # ax.plot(e_s, prob, label=shellstr + ' dP / d_e_s', color=color)
    # ax.vlines(ionpot_ev, ymin=0., ymax=max(prob), color=color)


def get_epsilon_avg(e_p, J, ionpot_ev, quiet=True):
    # average energy loss of the primary electron per ionisation in eV
    npts = 1000000

    epsilon_lower = ionpot_ev
    epsilon_upper = (ionpot_ev + e_p) / 2.
    if epsilon_upper <= epsilon_lower:
        return 0.
    delta_eps = (epsilon_upper - epsilon_lower) / npts
    # print(f'{delta_eps=}')
    # print(f'{epsilon_lower} {epsilon_upper} {delta_eps} {e_p}')
    prob_sum = 0.
    prob_e_s_can_ionise = 0.
    prob_e_s_cannot_ionise = 0.
    eps_avg = 0.
    e_s_avg = 0.
    for i in range(npts):
        epsilon = epsilon_lower + i * delta_eps
        prob = Psecondary(e_p, ionpot_ev, J, epsilon=epsilon) * delta_eps
        prob_sum += prob
        eps_avg += epsilon * prob
        e_s = epsilon - ionpot_ev
        e_s_avg += e_s * prob
        if e_s >= ionpot_ev:
            prob_e_s_can_ionise += prob
        else:
            prob_e_s_cannot_ionise += prob
    # print(f'{prob_sum=:.3f}')
    assert(abs(prob_sum - 1.) < 0.30)
    if not quiet:
        print(f'e_p: {e_p:.1e}')
        print(f'eps_avg: {eps_avg:.1e}')
        print(f'e_s_avg: {e_s_avg:.1e}')
        print(f'prob_e_s_can_ionise: {prob_e_s_can_ionise:.2f}')
        print(f'prob_e_s_cannot_ionise: {prob_e_s_cannot_ionise:.2f}')
    return eps_avg


def calculate_Latom_excitation(ions, ionpopdict, nntot, en_ev, adata, T_exc=5000):
    L_atom_sum = 0.
    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]

        ion = adata.query('Z == @Z and ion_stage == @ionstage').iloc[0]
        dftransitions_ion = ion.transitions

        maxnlevelslower = 5
        maxnlevelsupper = 250

        # filterquery = 'collstr >= 0 or forbidden == False'
        filterquery = 'collstr != -999'
        if maxnlevelslower is not None:
            filterquery += ' and lower < @maxnlevelslower'
        if maxnlevelsupper is not None:
            filterquery += ' and upper < @maxnlevelsupper'

        dftransitions_ion = dftransitions_ion.query(
            filterquery, inplace=False).copy()

        k_b = 8.617333262145179e-05  # eV / K
        energy_boltzfac_sum = ion.levels.eval(
            'energy_ev * g * exp(- energy_ev / @k_b / @T_exc)').sum()

        populations = ion.levels.eval('g * exp(- energy_ev / @k_b / @T_exc)').values / energy_boltzfac_sum

        dftransitions_ion.eval(
            'epsilon_trans_ev = '
            '@ion.levels.loc[upper].energy_ev.values - @ion.levels.loc[lower].energy_ev.values', inplace=True)

        dftransitions_ion.eval('upper_g = @ion.levels.loc[upper].g.values', inplace=True)
        dftransitions_ion.eval('lower_g = @ion.levels.loc[lower].g.values', inplace=True)

        for _, row in dftransitions_ion.iterrows():
            nnlevel = populations[row.lower] * nnion
            epsilon_trans_ev = row.epsilon_trans_ev
            sigma = get_xs_excitation(en_ev, row)
            L_atom_sum += sigma * epsilon_trans_ev * EV * nnlevel / nntot

    return L_atom_sum


def calculate_Latom_ionisation(ions, ionpopdict, dfcollion, nntot, nne, en_ev, nnetot=None, electron_binding=None):
    L_atom_sum = 0.
    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)
        # ionpot_valence_ev = dfcollion_thision.ionpot_ev.min()

        for _, shell in dfcollion_thision.iterrows():
            J = get_J(Z, ionstage, shell.ionpot_ev)

            eps_avg = get_epsilon_avg(en_ev, J, shell.ionpot_ev)
            # eps_avg = shell.ionpot_ev
            if eps_avg > 0:
                # sigma = get_lotz_xs_ionisation(Z, ionstage, electron_binding, ionpot_ev=ionpot_valence_ev, en_ev=en_ev)
                sigma = at.nonthermal.ar_xs(en_ev, shell.ionpot_ev, shell.A, shell.B, shell.C, shell.D)
                # L_atom_sum += ME * CLIGHT / math.pi / (QE ** 2) * sigma * eps_avg * nnion / nntot
                L_atom_sum += sigma * eps_avg * EV * nnion / nntot

        # J = get_J(Z, ionstage, ionpot_valence_ev)
        # eps_avg = get_epsilon_avg(en_ev, J, ionpot_valence_ev)
        # # test case: secondary electron has no energy
        # # eps_avg = ionpot_valence_ev
        # if eps_avg > 0:
        #     sigma = get_lotz_xs_ionisation(Z, ionstage, electron_binding, ionpot_ev=ionpot_valence_ev, en_ev=en_ev)
        #     # sigma = at.nonthermal.ar_xs(en_ev, shell.ionpot_ev, shell.A, shell.B, shell.C, shell.D)
        #     # L_atom_sum += ME * CLIGHT / math.pi / (QE ** 2) * sigma * eps_avg * nnion / nntot
        #     # nnebound = nnetot - nne
        #     # 1e-4 = EV * 2 * 3.14 * 1e7
        #     L_atom_sum += sigma * eps_avg * EV * nnion / nntot

        # print(f'ln I: {ln_I_sum:.2e} (after ionsiation)')
        # print(f'I: {math.exp(ln_I_sum):.2e} (after ionisation)')

    return L_atom_sum


def workfunction_tests(modelpath, args):
    electron_binding = read_binding_energies()
    dfcollion = at.nonthermal.read_colliondata()

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True,
                             figsize=(6*.7, 5*.7), tight_layout={"pad": 0.3, "w_pad": 0.0, "h_pad": 0.0})
    axes = [axes]

    ionpopdict = {
        # (16, 2): 3e5,
        (26, 2): 1e5,
    }

    ions = []
    for key in ionpopdict.keys():
        # keep only the ion populations, not element or total populations
        if isinstance(key, tuple) and len(key) == 2:
            ions.append(key)

    ions.sort()

    nntot = get_nntot(ions=ions, ionpopdict=ionpopdict)
    nnetot = get_nnetot(ions=ions, ionpopdict=ionpopdict)  # total electrons: free and bound included
    nne = get_nne(ions=ions, ionpopdict=ionpopdict)
    nnebound = nnetot - nne
    # x_e = nne / nntot
    Zbar = get_Zbar(ions=ions, ionpopdict=ionpopdict)  # average atomic number
    Zboundbar = get_Zboundbar(ions=ions, ionpopdict=ionpopdict)  # average bound electrons per nucleus
    print(f'nntot: {nntot:.1e}')
    print(f'nnetot:{nnetot:.1e}')
    print(f'nne: {nne:.1e}')
    print(f'nnebound: {nnebound:.1e}')
    print(f'Zbar: {Zbar:.1e}')
    print(f'Zboundbar: {Zboundbar:.1e}')

    hbar_ev_s = 6.58211951e-16  # in eV seconds
    omegap = 5.6e4 * math.sqrt(nne)  # in per second
    print(f'hbar * omegap = {hbar_ev_s * omegap:.2e} eV')

    en_min_ev = args.emin
    en_max_ev = args.emax
    print(f'en_min_ev: {en_min_ev:.1f} eV')
    print(f'en_max_ev: {en_max_ev:.1f} eV')
    # arr_en_ev = np.linspace(en_min_ev, en_max_ev, 2000)
    # delta_en_ev = arr_en_ev[1] - arr_en_ev[0]
    arr_en_ev = np.logspace(
        start=math.log10(en_min_ev), stop=math.log10(en_max_ev), base=10, num=args.npts, endpoint=True
    )

    global Psecondary_e_s_max
    Psecondary_e_s_max = arr_en_ev[2]
    print(f'Psecondary_e_s_max: {Psecondary_e_s_max}')

    delta_en_ev = arr_en_ev[1:] - arr_en_ev[:-1]
    arr_en_ev = arr_en_ev[:-1]  # remove the endpoint, now that we've used it to calculate detla_en_ev

    # Axelrod 1980 Eq 3.24
    Lelec_axelrod_nne = np.array([get_Lelec_axelrod(en_ev=en_ev, nne=nne, nnetot=nnetot, nntot=nntot) for en_ev in arr_en_ev])
    # Lelec_axelrod_nnetot = np.array([get_Lelec_axelrod(en_ev=en_ev, nne=nnetot, nnetot=nnetot, nntot=nntot) for en_ev in arr_en_ev])

    Lelec_kf92_nne = np.array([lossfunction(en_ev, nne, nnetot, use_nnetot=False) * EV / nntot for en_ev in arr_en_ev])

    Lelec_kf92_nnetot = np.array([lossfunction(en_ev, nne, nnetot, use_nnetot=True) * EV / nntot for en_ev in arr_en_ev])

    Lelec_kf92_nnebound = np.array([lossfunction(en_ev, nnebound, nnetot, use_nnetot=False) * EV / nntot for en_ev in arr_en_ev])

    # print(f'{Lelec_axelrod[-1]=}')
    # print(f'{Lelec_kf92[-1]=}')

    Latom_axelrod = np.array([get_Latom_axelrod(en_ev=en_ev, Zboundbar=Zboundbar) for en_ev in arr_en_ev])

    # adata = at.atomic.get_levels(modelpath, get_transitions=False, ionlist=tuple(ions))
    # adata = None
    arr_Latom_summed = np.zeros_like(arr_en_ev)
    for i, en_ev in enumerate(arr_en_ev):
        arr_Latom_summed[i] = calculate_Latom_ionisation(
            ions=ions, ionpopdict=ionpopdict, dfcollion=dfcollion, nntot=nntot, nne=nne,
            en_ev=en_ev, nnetot=nnetot, electron_binding=electron_binding)

        print(f'{en_ev:.2f} eV L_atom_summed: {arr_Latom_summed[i]:.3e} (ionisation only) '
              f'Latom_axelrod: {Latom_axelrod[i]:.3e} ratio(sum/axelrod): {arr_Latom_summed[i] / Latom_axelrod[i]:.2e} ratio(axelrod/sum) {Latom_axelrod[i] / arr_Latom_summed[i]:.2e} [erg cm2]')
        # print(f'{en_ev:.2f} eV L_elec_axelrod: {Lelec_axelrod_nne[i]:.3e} '
        #       f'Latom_axelrod: {Latom_axelrod[i]:.3e} ratio(elec/atom): {Lelec_axelrod_nne[i] / Latom_axelrod[i]:.2e} ratio(atom/elec) {Latom_axelrod[i] / Lelec_axelrod_nne[i]:.2e} bound/free {nnebound/nne:.2e}')

    for Z, ionstage in ions:
        # ionstr = at.get_ionstring(Z, ionstage, spectral=True, nospace=False)
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)

        # dfcollion_thision.query('n == 3 and l == 2', inplace=True)
        # dfcollion_thision.eval('A = 0', inplace=True)
        # dfcollion_thision.eval('B = 0', inplace=True)
        # # dfcollion_thision.eval('C = 0', inplace=True)
        # dfcollion_thision.eval('D = 0', inplace=True)

        # shellindex = 0
        # for _, shell in dfcollion_thision.iterrows():
        #     shellstr = f'\nn {shell.n} l {shell.l} ionpot {shell.ionpot_ev} eV'
        #     print(shellstr)
        #     e_p = arr_en_ev[-1]
        #     J = get_J(Z, ionstage, shell.ionpot_ev)
        #     e_s_test(axes[-1], shell.ionpot_ev, J, arr_en_ev, shellstr, color=f'C{shellindex}')
        #     shellindex += 1

        ionpot_valence_ev = dfcollion_thision.ionpot_ev.min()
        print(f'\n===> ion Z {Z} ionstage {ionstage} ionpot_valence_ev {ionpot_valence_ev}')
        # print(dfcollion_thision)

        binding_alt = get_mean_binding_energy_alt(Z, ionstage, electron_binding, ionpot_ev=ionpot_valence_ev)  # binding in erg
        print(f'mean binding energy: {binding_alt / EV:.2f} eV')
        Aconst = 1.33e-14 * EV * EV
        binding = get_mean_binding_energy(Z, ionstage, electron_binding, ionpot_ev=ionpot_valence_ev)  # binding in erg
        oneoverW_limit_sim = Aconst * binding / Zbar / (2 * 3.14159 * pow(QE, 4))  # per erg
        workfn_limit_ev_sim = 1. / oneoverW_limit_sim / EV
        print(f'\n workfn_limit_ev_sim {workfn_limit_ev_sim:.2f} eV')
        print(f'   eta_ion  {ionpot_valence_ev / workfn_limit_ev_sim:.3f}')
        print(f'   eta_heat {1 - ionpot_valence_ev / workfn_limit_ev_sim:.3f}')
        arr_workfn_limit_sim = np.array([workfn_limit_ev_sim for x in arr_en_ev])

        arr_xs_ar92 = at.nonthermal.get_arxs_array_ion(arr_en_ev, dfcollion_thision, Z, ionstage)
        Latom_ionisation_ar92 = arr_xs_ar92 * (ionpot_valence_ev * EV)

        arr_xs_lotz = np.array([
            get_lotz_xs_ionisation(Z, ionstage, electron_binding, ionpot_ev=ionpot_valence_ev, en_ev=en_ev)
            for en_ev in arr_en_ev])

        axes[-1].plot(arr_en_ev, Latom_axelrod, label='Latom_axelrod')

        # Axelrod 1980 Eq 3.20, (Latom part).
        # This assumes that the every bound electron cross section is included!
        Latom_ionisation_lotz = arr_xs_lotz * (ionpot_valence_ev * EV)

        L_over_sigma = (Lelec_axelrod_nne[-1] + Latom_axelrod[-1]) / arr_xs_lotz[-1]
        workfn_limit_axelrod = L_over_sigma / EV
        print(f'\n workfn_limit_axelrod: {workfn_limit_axelrod:.2f} eV')
        print(f'   eta_ion  {ionpot_valence_ev / workfn_limit_axelrod:.3f}')
        print(f'   eta_heat {1 - ionpot_valence_ev / workfn_limit_axelrod:.3f}')
        arr_workfn_limit_axelrod = np.array([workfn_limit_axelrod for x in arr_en_ev])

        # Approximation to Axelrod 1980 Eq 3.20 (left Latom part) where the transition energy
        # of every ionisation is just the valence potential
        # Latom_ionisation = arr_xs * (ionpot_valence_ev * EV)
        # print(Latom[-1], Latom2[-1], Latom3[-1])

        # arr_xs_latom = Latom1 / (ionpot_valence_ev * EV)

        # axes[-1].plot(arr_en_ev, arr_xs_lotz, label=r'$\sigma_{Lotz}$')
        # axes[-1].plot(arr_en_ev, arr_xs_ar92, label=r'$\sigma_{AR92}$')
        # axes[-1].plot(arr_en_ev, arr_xs_latom, label=r'$\sigma$=$L_{atom}/I$', linestyle='dashed')

        axes[-1].plot(arr_en_ev, arr_Latom_summed, label=r'Latom_summed')

        # Lelec = Lelec_kf92_nne
        # Lelec = Lelec_kf92_nnetot
        # Lelec = Lelec_kf92_nnebound
        Lelec = Lelec_axelrod_nne
        # Lelec = Lelec_axelrod_nnetot

        Latom = Latom_axelrod
        # Latom = Latom_ionisation_lotz
        # Latom = Latom_ionisation_ar92
        # axes[-1].plot(arr_en_ev, Lelec_kf92_nne / Latom_axelrod, label='Lelec_kf92 / Latom_axelrod')
        # axes[-1].plot(arr_en_ev, Lelec_axelrod / Latom_axelrod, label='Lelec_axelrod / Latom_axelrod')

        # arr_Lelec_over_Latom = [x_e / (Zbar - x_e) * (math.log(4 * en_ev / (hbar_ev_s * omegap)) / math.log(4 * en_ev / 300)) for en_ev in arr_en_ev]
        # axes[-1].plot(arr_en_ev, arr_Lelec_over_Latom, label='Lelec_axelrod / Latom_axelrod analytic')

        # axes[-1].plot(arr_en_ev, Lelec_axelrod / Lelec_kf92_nne, label='lossfunction_axelrod / lossfunction_kf92')

        # Latom = Latom_axelrod - Latom_ionisation_lotz + Latom_ionisation_ar92

        # arr_xs = arr_xs_lotz
        arr_xs = arr_xs_ar92
        # axes[-1].plot(arr_en_ev, arr_xs_ar92 * 1e18, label=ionstr + ' Arnaud & Rotherflug 1992', linewidth=2)
        # axes[-1].plot(arr_en_ev, arr_xs_lotz * 1e18, label=ionstr + ' Lotz (1967)', linewidth=2)

        L = Lelec + Latom

        with np.errstate(divide='ignore'):
            workfn_limit = L / EV / arr_xs

        print(f'\n workfn_limit at Emax: {workfn_limit[-1]:.2f} eV')
        print(f'   eta_ion  {ionpot_valence_ev / workfn_limit[-1]:.3f}')
        print(f'   eta_heat {1 - ionpot_valence_ev / workfn_limit[-1]:.3f}')

        arr_workfn_integrated = np.zeros_like(arr_en_ev)
        integrand = arr_xs / (L / EV)
        # arr_workfn_integrated[i] is the en_ev / (integral xs / L dE from EMIN to E[i])
        with np.errstate(divide='ignore'):
            arr_workfn_integrated = [arr_en_ev[i] / (sum((integrand * delta_en_ev)[:i])) for i in range(len(arr_en_ev))]

        print(f'\n workfn_integral_Emin_Emax: {arr_workfn_integrated[-1]:.2f} eV')
        print(f'   eta_ion  {ionpot_valence_ev / arr_workfn_integrated[-1]:.3f}')
        print(f'   eta_heat {1 - ionpot_valence_ev / arr_workfn_integrated[-1]:.3f}')

        # axes[-1].plot(arr_en_ev, arr_workfn_limit_axelrod, label='workfn limit E->inf (Axelrod)')
        # axes[-1].plot(arr_en_ev, arr_workfn_limit_sim, label='workfn limit E->inf (Sim)')
        # axes[-1].plot(arr_en_ev, workfn_limit, label='workfn integrated near Emax', color='C2')
        # axes[-1].plot(arr_en_ev, arr_workfn_integrated, label='workfn integrated Emin to Emax', color='C3')

        # arr = [lossfunction(en_ev, nne, nnetot) / arr_xs_ar92[i] for i, en_ev in enumerate(arr_en_ev)]
        # arr = [Latom_axelrod[i] / arr_xs_ar92[i] for i, en_ev in enumerate(arr_en_ev)]

        # print(arr)
        # print(min(arr), max(arr))
        # axes[-1].plot(arr_en_ev, arr, label='lossfunc / sigma', color='C3')

    axes[-1].set_xlabel(r'Energy [eV]')
    axes[-1].set_xscale('log')
    # axes[-1].set_xlim(0., 500)
    # axes[-1].set_ylim(0., 2000)
    # axes[-1].set_ylim(bottom=1e-3, top=1e-2)
    # axes[-1].set_ylim(bottom=min(arr) / 2., top=max(arr) * 2.)
    # axes[-1].set_ylabel(r'cross section [cm$^{2}$]')
    # axes[-1].set_ylabel(r'Cross Section $[10^{-18}$ cm$^{2}]$')
    # axes[-1].set_ylabel(r'Ionisation fraction of deposited energy')
    axes[-1].set_yscale('log')
    # axes[-1].set_ylabel(r'log y(E) [s$^{-1}$ cm$^{-2}$ eV$^{-1}$]', fontsize=fs)
    # axes[-1].yaxis.set_minor_locator(ticker.MultipleLocator(base=5))
    axes[-1].legend(frameon=False, loc='upper right')
    outputfilename = 'plot.pdf'
    print(f"Saving '{outputfilename}'")
    fig.savefig(str(outputfilename), format='pdf')
    plt.close()
