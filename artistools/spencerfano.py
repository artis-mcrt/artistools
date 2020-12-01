#!/usr/bin/env python3
import argparse
import math
import multiprocessing
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import constants as const
from functools import lru_cache
from scipy import linalg
from pathlib import Path

import artistools as at
import artistools.estimators
import artistools.nltepops
import artistools.nonthermal


minionfraction = 1.e-8  # minimum number fraction of the total population to include in SF solution

defaultoutputfile = 'spencerfano_cell{cell:03d}_ts{timestep:02d}_{time_days:.0f}d.pdf'


def get_lte_pops(adata, ions, ionpopdict, temperature):
    poplist = []

    K_B = const.k_B.to('eV / K').value

    for _, ion in adata.iterrows():
        ionid = (ion.Z, ion.ion_stage)
        if ionid in ions:
            Z = ion.Z
            ionstage = ion.ion_stage
            nnion = ionpopdict[(Z, ionstage)]

            ltepartfunc = ion.levels.eval('g * exp(-energy_ev / @K_B / @temperature)').sum()

            for levelindex, level in ion.levels.iterrows():
                nnlevel = nnion / ltepartfunc * math.exp(-level.energy_ev / K_B / temperature)

                poprow = (Z, ionstage, levelindex, nnlevel, nnlevel, nnlevel / nnion)
                poplist.append(poprow)

    dfpop = pd.DataFrame(poplist, columns=['Z', 'ion_stage', 'level', 'n_LTE', 'n_NLTE', 'ion_popfrac'])
    return dfpop


def lossfunction_ergs(energy, nne):
    H = 6.6260755e-27
    ME = 9.1093897e-28
    EV = 1.6021772e-12
    PI = 3.1415926535987
    QE = 4.80325E-10

    eulergamma = 0.577215664901532

    omegap = math.sqrt(4 * math.pi * nne * pow(QE, 2) / ME)
    zetae = H * omegap / 2 / PI
    v = math.sqrt(2 * energy / ME)
    # print(f'omegap {omegap:.2e} s^-1')
    # print(f'zetae {zetae:.2e} erg = {zetae / EV:.2e} eV')
    # print(f'v {v:.2e}  cm/s')
    if energy > 14 * EV:
        return nne * 2 * PI * pow(QE, 4) / energy * math.log(2 * energy / zetae)
    else:
        return nne * 2 * PI * pow(QE, 4) / energy * math.log(ME * pow(v, 3) / (eulergamma * pow(QE, 2) * omegap))


def lossfunction(energy_ev, nne_cgs):
    nne = nne_cgs * 1e6   # convert from cm^-3 to m^-3
    energy = energy_ev * 1.60218e-19  # convert eV to J
    qe = 1.609e-19  # elementary charge in Coulombs
    me = 9.10938e-31  # kg
    h = 6.62606e-34  # J s
    eps0 = 8.854187817e-12  # F / m

    # h = 4.1356e-15   # Planck's constant in eV * s
    # me = 0.510998e6  # mass of electron in eV/c^2
    # c = 29979245800.  # c in cm/s

    eulergamma = 0.577215664901532

    # zetae = h / (2 * math.pi) * math.sqrt((4 * math.pi * nne * (qe ** 2)) / me)
    # omegap = 2 * math.pi * zetae / h

    omegap = math.sqrt(nne * pow(qe, 2) / me / eps0)
    zetae = h * omegap / 2 / math.pi

    v = math.sqrt(2 * energy / me)  # velocity in m/s
    # print(f'omegap {omegap:.2e} s^-1')
    # print(f'zetae {zetae:.2e} J = {zetae / 1.602e-19:.2e} eV')
    # print(f'v {v:.2e} m/s')
    if energy > 14:
        lossfunc = nne * 2 * math.pi * (qe ** 4 / (4 * math.pi * eps0) ** 2) / energy * math.log(2 * energy / zetae)
    else:
        lossfunc = (nne * 2 * math.pi * (qe ** 4 / (4 * math.pi * eps0) ** 2) / energy * math.log(me * (v ** 3) /
                    (eulergamma * (qe ** 2) * omegap / (4 * math.pi * eps0))))

    # lossfunc is now in J / m
    return lossfunc / 1.60218e-19 / 100  # eV / cm


def Psecondary(e_p, I, J, e_s=-1, epsilon=-1):
    assert(e_s >= 0 or epsilon >= 0)
    # if e_p < I:
    #     return 0.
    #
    # if e_s < 0:
    #     e_s = epsilon - I
    # if epsilon < 0:
    #     epsilon = e_s + I
    #
    # if epsilon < I:
    #     return 0.
    # if e_s < 0:
    #     return 0.
    # if e_s > e_p - I:
    #     return 0.
    # if e_s > e_p:
    #     return 0.

    return 1. / J / np.arctan((e_p - I) / 2. / J) / (1 + ((e_s / J) ** 2))


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


def get_xs_excitation_vector(engrid, row):
    A_naught_squared = 2.800285203e-17  # Bohr radius squared in cm^2
    H = 6.6260755e-27
    ME = 9.1093897e-28
    EV = 1.6021772e-12
    QE = 4.80325E-10
    H_ionpot = 13.5979996 * EV
    CLIGHT = 2.99792458e+10

    deltaen = engrid[1] - engrid[0]
    npts = len(engrid)
    xs_excitation_vec = np.empty(npts)

    coll_str = row.collstr
    epsilon_trans = row.epsilon_trans_ev * EV
    epsilon_trans_ev = row.epsilon_trans_ev

    startindex = math.ceil((epsilon_trans_ev - engrid[0]) / deltaen)
    xs_excitation_vec[:startindex] = 0.

    if (coll_str >= 0):
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


def get_index(en_ev, engrid):
    if en_ev < engrid[0] or en_ev > engrid[-1] + (engrid[1] - engrid[0]):
        assert(False)

    for i, en in enumerate(engrid):
        if en < en_ev:
            index = i

    return index


def calculate_N_e(energy_ev, engrid, ions, ionpopdict, dfcollion, yvec, dftransitions, noexcitation):
    # Kozma & Fransson equation 6.
    # Something related to a number of electrons, needed to calculate the heating fraction in equation 3
    # not valid for energy > E_0
    if energy_ev == 0.:
        return 0.
    N_e = 0.

    E_0 = engrid[0]
    deltaen = engrid[1] - engrid[0]

    for Z, ionstage in ions:
        N_e_ion = 0.
        nnion = ionpopdict[(Z, ionstage)]
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)

        for index, shell in dfcollion_thision.iterrows():
            ionpot_ev = shell.ionpot_ev

            enlambda = min(engrid[-1] - energy_ev, energy_ev + ionpot_ev)
            J = get_J(shell.Z, shell.ionstage, ionpot_ev)

            ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, shell)

            # integral from ionpot to enlambda
            # delta_endash = engrid[1] - engrid[0]

            delta_endash = (enlambda - ionpot_ev) / 100.
            endashlist = np.arange(ionpot_ev, enlambda, delta_endash)
            weightlist = [1 if (i == 0 or i == len(endashlist) - 1) else 2 for i in range(len(endashlist))]
            for weight, endash in zip(weightlist, endashlist):
                i = get_index(en_ev=energy_ev + endash, engrid=engrid)
                N_e_ion += (
                    0.5 * weight * yvec[i] * ar_xs_array[i] *
                    Psecondary(e_p=energy_ev + endash, epsilon=endash, I=ionpot_ev, J=J) * delta_endash)

            # // integral from 2E + I up to E_max
            delta_endash = (engrid[-1] - (2 * energy_ev + ionpot_ev)) / 100.
            endashlist = np.arange(2 * energy_ev + ionpot_ev, engrid[-1], delta_endash)
            weightlist = [1 if (i == 0 or i == len(endashlist) - 1) else 2 for i in range(len(endashlist))]
            for weight, endash in zip(weightlist, endashlist):
                i = get_index(en_ev=endash, engrid=engrid)
                N_e_ion += (
                    0.5 * weight * yvec[i] * ar_xs_array[i] *
                    Psecondary(e_p=endash, epsilon=energy_ev + ionpot_ev, I=ionpot_ev, J=J) * delta_endash)

        N_e += nnion * N_e_ion

    if not noexcitation:
        for Z, ion_stage in ions:
            for _, row in dftransitions[(Z, ion_stage)].iterrows():
                nnlevel = row.lower_pop
                # nnlevel = nnion
                epsilon_trans_ev = row.epsilon_trans_ev
                if epsilon_trans_ev >= engrid[0]:
                    i = get_index(en_ev=energy_ev + epsilon_trans_ev, engrid=engrid)
                    xsvec = get_xs_excitation_vector(engrid, row)
                    N_e += nnlevel * epsilon_trans_ev * xsvec[i] * yvec[i]

    # // source term should be zero at the low end anyway

    return N_e


def calculate_frac_heating(
        engrid, ions, ionpopdict, dfcollion, dftransitions, yvec, nne, deposition_density_ev, noexcitation):
    # Kozma & Fransson equation 8

    frac_heating = 0.
    E_0 = engrid[0]

    deltaen = engrid[1] - engrid[0]
    npts = len(engrid)
    for i, en_ev in enumerate(engrid):
        weight = 1 if (i == 0 or i == npts - 1) else 2
        frac_heating += 0.5 * weight * lossfunction(en_ev, nne) * yvec[i] * deltaen / deposition_density_ev

    frac_heating += E_0 * yvec[0] * lossfunction(E_0, nne) / deposition_density_ev

    frac_heating_N_e = 0.
    delta_en = E_0 / 100.
    for en_ev in np.arange(0., E_0, delta_en):
        N_e = calculate_N_e(en_ev, engrid, ions, ionpopdict, dfcollion, yvec, dftransitions, noexcitation=noexcitation)
        frac_heating_N_e += N_e * en_ev * delta_en / deposition_density_ev

    print(f"            frac_heating N_e part: {frac_heating_N_e}")
    frac_heating += frac_heating_N_e

    return frac_heating


def sfmatrix_add_excitation(engrid, dftransitions_ion, nnion, sfmatrix):
    deltaen = engrid[1] - engrid[0]
    npts = len(engrid)
    for _, row in dftransitions_ion.iterrows():
        nnlevel = row.lower_pop
        epsilon_trans_ev = row.epsilon_trans_ev
        if epsilon_trans_ev >= engrid[0]:
            vec_xs_excitation_nnlevel_deltae = nnlevel * deltaen * get_xs_excitation_vector(engrid, row)
            for i, en in enumerate(engrid):
                stopindex = i + math.ceil(epsilon_trans_ev / deltaen)

                if (stopindex < npts - 1):
                    sfmatrix[i, i: stopindex - i + 1] += vec_xs_excitation_nnlevel_deltae[i: stopindex - i + 1]


def sfmatrix_add_ionization_shell(engrid, nnion, shell, sfmatrix):
    # this code has been optimised and is now an almost unreadable form, but it is the contains the terms
    # related to ionisation cross sections
    deltaen = engrid[1] - engrid[0]
    ionpot_ev = shell.ionpot_ev
    J = get_J(shell.Z, shell.ionstage, ionpot_ev)
    npts = len(engrid)

    ar_xs_array = at.nonthermal.get_arxs_array_shell(engrid, shell)

    if ionpot_ev <= engrid[0]:
        xsstartindex = 0
    else:
        xsstartindex = get_index(en_ev=ionpot_ev, engrid=engrid)

    for i, en in enumerate(engrid):
        # P_sum = 0.
        # if en >= ionpot_ev:
        #     e_s_min = 0.
        #     e_s_max = (en - ionpot_ev) / 2.
        #     delta_e_s = (e_s_max - e_s_min) / 1000.
        #     for e_s in np.arange(e_s_min, e_s_max, delta_e_s):
        #         P_sum += Psecondary(e_p=en, e_s=e_s, I=ionpot_ev, J=J) * delta_e_s
        #
        #     epsilon_upper = e_s_max + ionpot_ev
        #     epsilon_lower = e_s_min + ionpot_ev
        #     P_int = 1. / np.arctan((en - ionpot_ev) / 2. / J) * (np.arctan((epsilon_upper - ionpot_ev) / J) - np.arctan((epsilon_lower - ionpot_ev) / J))
        #
        #     print(f"E_p {en} eV, prob integral e_s from {e_s_min:5.2f} eV to {e_s_max:5.2f}: {P_sum:5.2f} analytical {P_int:5.2f}")

        # // endash ranges from en to SF_EMAX, but skip over the zero-cross section points
        jstart = i if i > xsstartindex else xsstartindex
        if 2 * en + ionpot_ev < engrid[-1] + (engrid[1] - engrid[0]):
            secondintegralstartindex = get_index(2 * en + ionpot_ev, engrid)
        else:
            secondintegralstartindex = npts + 1

        weight_a = 1
        weight_b = 1
        # integral/J of 1/[1 + (epsilon - ionpot_ev) / J] for epsilon = en + ionpot_ev
        int_eps_lower_b = np.arctan((en) / J)
        for j in range(jstart, npts):
            # j is the matrix column index which corresponds to the piece of the
            # integral at y(E') where E' >= E and E' = envec(j)
            endash = engrid[j]
            prefactor = nnion * ar_xs_array[j] / np.arctan((endash - ionpot_ev) / 2. / J) * deltaen
            assert(prefactor >= 0)
            epsilon_upper = (endash + ionpot_ev) / 2
            int_eps_upper = np.arctan((epsilon_upper - ionpot_ev) / J)

            # // atan[(epsilon - ionpot_ev) / J] is the indefinite integral of 1/[1 + (epsilon - ionpot_ev) / J]
            # // in Kozma & Fransson 1992 equation 4

            # epsilon_upper = (endash + ionpot_ev) / 2

            # J * atan[(epsilon - ionpot_ev) / J] is the indefinite integral of
            # 1/(1 + (epsilon - ionpot_ev)^2/ J^2) d_epsilon
            epsilon_lower = endash - en
            if (int_eps_upper - np.arctan((epsilon_lower - ionpot_ev) / J)) > 0.:
                if j == npts - 1:
                    weight_a = 1
                sfmatrix[i, j] += 0.5 * weight_a * prefactor * (int_eps_upper - np.arctan((epsilon_lower - ionpot_ev) / J))
                if weight_a == 1:
                    weight_a == 2

            # endash ranges from 2 * en + ionpot_ev to SF_EMAX
            if j >= secondintegralstartindex and int_eps_upper > int_eps_lower_b:
                # epsilon_lower = en + ionpot_ev

                if j == npts - 1:
                    weight_b = 1
                sfmatrix[i, j] -= 0.5 * weight_b * prefactor * (int_eps_upper - int_eps_lower_b)
                if weight_b == 1:
                    weight_b == 2


def get_d_etaexcitation_by_d_en_vec(engrid, yvec, ions, dftransitions, deposition_density_ev):
    npts = len(engrid)
    part_integrand = np.zeros(npts)

    for Z, ion_stage in ions:
        for _, row in dftransitions[(Z, ion_stage)].iterrows():
            nnlevel = row.lower_pop
            # nnlevel = nnion
            epsilon_trans_ev = row.epsilon_trans_ev
            if epsilon_trans_ev >= engrid[0]:
                xsvec = get_xs_excitation_vector(engrid, row)
                part_integrand += (nnlevel * epsilon_trans_ev * xsvec / deposition_density_ev)

    return yvec * part_integrand


def get_d_etaion_by_d_en_vec(engrid, yvec, ions, ionpopdict, dfcollion, deposition_density_ev):
    npts = len(engrid)
    part_integrand = np.zeros(npts)

    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)
        # print(dfcollion_thision)

        for index, shell in dfcollion_thision.iterrows():
            J = get_J(shell.Z, shell.ionstage, shell.ionpot_ev)
            xsvec = at.nonthermal.get_arxs_array_shell(engrid, shell)

            part_integrand += (nnion * shell.ionpot_ev * xsvec / deposition_density_ev)

    return yvec * part_integrand


def make_plot(
        engrid, yvec, ions, ionpopdict, dfcollion, dftransitions, nne,
        sourcevec, deposition_density_ev, outputfilename, noexcitation):

    fs = 13
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                             figsize=(6, 8), tight_layout={"pad": 0.3, "w_pad": 0.0, "h_pad": 0.0})

    npts = len(engrid)

    # E_init_ev = np.dot(engrid, sourcevec) * deltaen
    # d_etasource_by_d_en_vec = engrid * sourcevec / E_init_ev
    # axes[0].plot(engrid[1:], d_etasource_by_d_en_vec[1:], marker="None", lw=1.5, color='blue', label='Source')

    d_etaion_by_d_en_vec = get_d_etaion_by_d_en_vec(engrid, yvec, ions, ionpopdict, dfcollion, deposition_density_ev)
    axes[1].plot(engrid, d_etaion_by_d_en_vec, marker="None", lw=1.5, color='C0', label='Ionisation')

    if not noexcitation:
        d_etaexc_by_d_en_vec = get_d_etaexcitation_by_d_en_vec(engrid, yvec, ions, dftransitions, deposition_density_ev)
        axes[1].plot(engrid, d_etaexc_by_d_en_vec, marker="None", lw=1.5, color='C1', label='Excitation')
    else:
        d_etaexc_by_d_en_vec = np.zeros(npts)

    d_etaheat_by_d_en_vec = [
        lossfunction(engrid[i], nne) * yvec[i] for i in range(len(engrid))]
    axes[1].plot(engrid, d_etaheat_by_d_en_vec, marker="None", lw=1.5, color='C2', label='Heating')

    axes[1].set_ylabel(r'd $\eta$ / dE [eV$^{-1}$]', fontsize=fs)
    axes[1].set_ylim(bottom=0)

    axes[1].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})

    axes[-1].plot(engrid, np.log10(yvec), marker="None", lw=1.5, color='black')

    deltaen = engrid[1] - engrid[0]
    etaion_int = np.zeros(npts)
    etaexc_int = np.zeros(npts)
    etaheat_int = np.zeros(npts)
    for i in reversed(range(len(engrid) - 1)):
        etaion_int[i] = etaion_int[i + 1] + d_etaion_by_d_en_vec[i] * deltaen
        etaexc_int[i] = etaexc_int[i + 1] + d_etaexc_by_d_en_vec[i] * deltaen
        etaheat_int[i] = etaexc_int[i + 1] + d_etaheat_by_d_en_vec[i] * deltaen
    etatot_int = etaion_int + etaexc_int + etaheat_int
    axes[0].plot(engrid, etaion_int, marker="None", lw=1.5, color='C0', label='Ionisation')
    if not noexcitation:
        axes[0].plot(engrid, etaexc_int, marker="None", lw=1.5, color='C1', label='Excitation')
    axes[0].plot(engrid, etaheat_int, marker="None", lw=1.5, color='C2', label='Heating')
    axes[0].plot(engrid, etatot_int, marker="None", lw=1.5, color='black', label='Total')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
    axes[0].set_ylabel(r'Integral $\eta$ E to Emax', fontsize=fs)

    etatot_int = etaion_int + etaexc_int + etaheat_int

    #    plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    #    plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    #    for axis in ['top','bottom','left','right']:
    #        ax.spines[axis].set_linewidth(framewidth)
    #    ax.annotate(modellabel, xy=(0.97, 0.95), xycoords='axes fraction', horizontalalignment='right',
    #                verticalalignment='top', fontsize=fs)
    # ax.set_yscale('log')
    axes[-1].set_xlim(engrid[0], engrid[-1] * 1.)
    # ax.set_ylim(bottom=5, top=14)
    axes[-1].set_xlabel(r'Electron energy [eV]', fontsize=fs)
    axes[-1].set_ylabel(r'log y(E) [s$^{-1}$ cm$^{-2}$ eV$^{-1}$]', fontsize=fs)
    print(f"Saving '{outputfilename}'")
    fig.savefig(str(outputfilename), format='pdf')
    plt.close()


def solve_spencerfano(
        ions, ionpopdict, dfpops, nne, deposition_density_ev, engrid, sourcevec, dfcollion, args,
        adata=None, noexcitation=False):

    deltaen = engrid[1] - engrid[0]
    npts = len(engrid)

    print(f'\nSetting up Spencer-Fano equation with {npts} energy points from {engrid[0]} to {engrid[-1]} eV...')

    E_init_ev = np.dot(engrid, sourcevec) * deltaen
    # print(f'    E_init: {E_init_ev:7.2f} eV/s/cm3')

    constvec = np.zeros(npts)
    for i in range(npts):
        for j in range(i, npts):
            constvec[i] += sourcevec[j] * deltaen

    sfmatrix = np.zeros((npts, npts))
    for i in range(npts):
        en = engrid[i]
        sfmatrix[i, i] += lossfunction(en + deltaen, nne)
        # EV = 1.6021772e-12  # in erg
        # print(f"electron loss rate nne={nne:.3e} and {i:d} {en:.2e} eV is {lossfunction(en, nne):.2e} or '
        #       f'{lossfunction_ergs(en * EV, nne) / EV:.2e}")

    dftransitions = {}

    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        print(f'  including Z={Z} ion_stage {ionstage} ({at.get_ionstring(Z, ionstage)}). ionization', end='')
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)
        # print(dfcollion_thision)

        for index, shell in dfcollion_thision.iterrows():
            sfmatrix_add_ionization_shell(engrid, nnion, shell, sfmatrix)

        if not noexcitation:
            dfpops_thision = dfpops.query('Z==@Z & ion_stage==@ionstage')
            popdict = {x.level: x['n_NLTE'] for _, x in dfpops_thision.iterrows()}

            print(' and excitation ', end='')
            ion = adata.query('Z == @Z and ion_stage == @ionstage').iloc[0]
            groundlevelnoj = ion.levels.iloc[0].levelname.split('[')[0]
            topgmlevel = ion.levels[ion.levels.levelname.str.startswith(groundlevelnoj)].index.max()
            # topgmlevel = float('inf')
            topgmlevel = 4
            dftransitions[(Z, ionstage)] = ion.transitions.query('lower <= @topgmlevel', inplace=False).copy()

            print(f'with {len(dftransitions[(Z, ionstage)])} transitions from lower <= {topgmlevel}', end='')

            if not dftransitions[(Z, ionstage)].empty:
                dftransitions[(Z, ionstage)].query('collstr >= 0 or forbidden == False', inplace=True)
                dftransitions[(Z, ionstage)].eval(
                    'epsilon_trans_ev = '
                    '@ion.levels.loc[upper].energy_ev.values - @ion.levels.loc[lower].energy_ev.values',
                    inplace=True)
                dftransitions[(Z, ionstage)].query('epsilon_trans_ev >= @engrid[0]', inplace=True)
                dftransitions[(Z, ionstage)].eval('lower_g = @ion.levels.loc[lower].g.values', inplace=True)
                dftransitions[(Z, ionstage)].eval('upper_g = @ion.levels.loc[upper].g.values', inplace=True)
                dftransitions[(Z, ionstage)]['lower_pop'] = dftransitions[(Z, ionstage)].apply(
                    lambda x: popdict.get(x.lower, 0.), axis=1)

                sfmatrix_add_excitation(engrid, dftransitions[(Z, ionstage)], nnion, sfmatrix)

        print()

    print()
    lu_and_piv = linalg.lu_factor(sfmatrix, overwrite_a=False)
    yvec_reference = linalg.lu_solve(lu_and_piv, constvec, trans=0)
    yvec = yvec_reference * deposition_density_ev / E_init_ev

    return yvec, dftransitions


def get_nt_en_tot(engrid, yvec):
    # oneovervelocity = np.sqrt(9.10938e-31 / 2 / engrid / 1.60218e-19) / 100  # in s/cm
    # enovervelocity = engrid * oneovervelocity
    # en_tot = np.dot(yvec, enovervelocity) * (engrid[1] - engrid[0])
    en_tot = 0.
    deltaen = (engrid[1] - engrid[0])
    for i, en in enumerate(engrid):
        oneovervelocity = np.sqrt(9.10938e-31 / 2 / en / 1.60218e-19) / 100.
        en_tot += en * yvec[i] * oneovervelocity * deltaen

    return en_tot


def analyse_ntspectrum(
        engrid, yvec, ions, ionpopdict, nntot, nne, deposition_density_ev, dfcollion, dftransitions,
        noexcitation):

    deltaen = engrid[1] - engrid[0]

    frac_ionization = 0.
    frac_excitation = 0.
    frac_ionization_ion = {}
    frac_excitation_ion = {}
    gamma_nt = {}

    for Z, ionstage in ions:
        nnion = ionpopdict[(Z, ionstage)]
        X_ion = nnion / nntot
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage', inplace=False)
        # if dfcollion.empty:
        #     continue
        ionpot_valence = dfcollion_thision.ionpot_ev.min()

        print(f'====> Z={Z:2d} {at.get_ionstring(Z, ionstage)} (valence potential {ionpot_valence:.1f} eV)')

        print(f'               nnion: {nnion:.2e} /cm3')
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
                print('Ignoring frac_ionization_shell of {frac_ionization_shell}.')
                # for k in range(10):
                #     print(nnion * shell.ionpot_ev * yvec_reference[k] * ar_xs_array[k] * deltaen / E_init_ev)

            frac_ionization_ion[(Z, ionstage)] += frac_ionization_shell
            eta_over_ionpot_sum += frac_ionization_shell / shell.ionpot_ev

        frac_ionization += frac_ionization_ion[(Z, ionstage)]

        eff_ionpot_2 = X_ion / eta_over_ionpot_sum if eta_over_ionpot_sum else float('inf')

        try:
            eff_ionpot = ionpot_valence * X_ion / frac_ionization_ion[(Z, ionstage)]
        except ZeroDivisionError:
            eff_ionpot = float('inf')

        print(f'     frac_ionization: {frac_ionization_ion[(Z, ionstage)]:.4f}')
        if not noexcitation:
            frac_excitation_ion[(Z, ionstage)] = calculate_nt_frac_excitation(
                engrid, dftransitions[(Z, ionstage)], yvec, deposition_density_ev)
            if frac_excitation_ion[(Z, ionstage)] > 1:
                frac_excitation_ion[(Z, ionstage)] = 0.
                print('Ignoring frac_excitation_ion of {frac_excitation_ion[(Z, ionstage)]}.')
            frac_excitation += frac_excitation_ion[(Z, ionstage)]
            print(f'     frac_excitation: {frac_excitation_ion[(Z, ionstage)]:.4f}')
        else:
            frac_excitation_ion[(Z, ionstage)] = 0.

        print(f' eff_ionpot_shellpot: {eff_ionpot_2:.2f} eV')
        print(f'  eff_ionpot_valence: {eff_ionpot:.2f} eV')
        gamma_nt[(Z, ionstage)] = deposition_density_ev / nntot / eff_ionpot
        print(f'  Spencer-Fano Gamma: {gamma_nt[(Z, ionstage)]:.2e}')
        # print(f'Alternative Gamma: {integralgamma:.2e}')
        print()

    print(f'  frac_excitation_tot: {frac_excitation:.5f}')
    print(f'  frac_ionization_tot: {frac_ionization:.5f}')

    nt_en_tot = get_nt_en_tot(engrid, yvec)
    print(f' nt_en_tot: {nt_en_tot:.5f} eV/s/cm3')

    frac_heating = calculate_frac_heating(
        engrid, ions, ionpopdict, dfcollion, dftransitions, yvec, nne, deposition_density_ev,
        noexcitation=noexcitation)

    print(f'         frac_heating: {frac_heating:.5f}')
    print(f'             frac_sum: {frac_excitation + frac_ionization + frac_heating:.5f}')

    return frac_excitation, frac_ionization, frac_excitation_ion, frac_ionization_ion, gamma_nt


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', type=int,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('-npts', type=int, default=8192,
                        help='Number of points in the energy grid')

    parser.add_argument('-emin', type=float, default=0.1,
                        help='Minimum energy in eV of Spencer-Fano solution')

    parser.add_argument('-emax', type=float, default=16000,
                        help='Maximum energy in eV of Spencer-Fano solution (approx where energy is injected)')

    parser.add_argument('-vary', action='store', choices=['emin', 'emax', 'npts', 'emax,npts'],
                        help='Which parameter to vary')

    parser.add_argument('--makeplot', action='store_true',
                        help='Save a plot of the non-thermal spectrum')

    parser.add_argument('--noexcitation', action='store_true',
                        help='Do not include collisional excitation transitions')

    parser.add_argument('--ar1985', action='store_true',
                        help='Use Arnaud & Rothenflug (1985, A&AS, 60, 425) for Fe ionization cross sections')

    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='Path/filename for PDF file if --makeplot is enabled')

    parser.add_argument('-ostat', action='store',
                        help='Path/filename for stats output')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot estimated spectra from bound-bound transitions.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if Path(args.outputfile).is_dir():
        args.outputfile = Path(args.outputfile, defaultoutputfile)

    modelpath = Path(args.modelpath)

    # if args.timedays:
    #     args.timestep = at.get_timestep_of_timedays(modelpath, args.timedays)
    # elif args.timestep is None:
    #     print("A time or timestep must be specified.")
    #     sys.exit()

    # modeldata, _ = at.get_modeldata(modelpath)
    # estimators = at.estimators.read_estimators(modelpath, timestep=args.timestep, modelgridindex=args.modelgridindex)
    # estim = estimators[(args.timestep, args.modelgridindex)]
    #
    # dfpops = at.nltepops.read_files(modelpath, modelgridindex=args.modelgridindex, timestep=args.timestep)
    #
    # if dfpops is None or dfpops.empty:
    #     print(f'ERROR: no NLTE populations for cell {args.modelgridindex} at timestep {args.timestep}')
    #     return -1
    #
    # nntot = estim['populations']['total']
    # nne = estim['nne']
    # deposition_density_ev = estim['gamma_dep'] / 1.6021772e-12  # convert erg to eV
    # ionpopdict = estim['populations']

    # ionpopdict = {}
    # deposition_density_ev = 327
    # nne = 6.7e5
    #
    # ionpopdict[(26, 1)] = ionpopdict[26] * 1e-4
    # ionpopdict[(26, 2)] = ionpopdict[26] * 0.20
    # ionpopdict[(26, 3)] = ionpopdict[26] * 0.80
    # ionpopdict[(26, 4)] = ionpopdict[26] * 0.
    # ionpopdict[(26, 5)] = ionpopdict[26] * 0.
    # ionpopdict[(27, 2)] = ionpopdict[27] * 0.20
    # ionpopdict[(27, 3)] = ionpopdict[27] * 0.80
    # ionpopdict[(27, 4)] = 0.
    # # ionpopdict[(28, 1)] = ionpopdict[28] * 6e-3
    # ionpopdict[(28, 2)] = ionpopdict[28] * 0.18
    # ionpopdict[(28, 3)] = ionpopdict[28] * 0.82
    # ionpopdict[(28, 4)] = ionpopdict[28] * 0.
    # ionpopdict[(28, 5)] = ionpopdict[28] * 0.

    # x_e = 0.1
    deposition_density_ev = 1e-1
    nntot = 1.0
    ionpopdict = {}
    # nne = nntot * x_e
    nne = .1
    dfpops = {}

    # ionpopdict[(at.get_atomic_number('Fe'), 2)] = nntot * 0.5
    ionpopdict[(at.get_atomic_number('Fe'), 3)] = nntot

    # KF1992 D. The Oxygen-Carbon Zone
    # ionpopdict[(at.get_atomic_number('C'), 1)] = 0.16 * nntot
    # ionpopdict[(at.get_atomic_number('C'), 2)] = 0.16 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('O'), 1)] = 0.86 * nntot
    # ionpopdict[(at.get_atomic_number('O'), 2)] = 0.86 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('Ne'), 1)] = 0.016 * nntot

    # # KF1992 G. The Silicon-Calcium Zone
    # ionpopdict[(at.get_atomic_number('C'), 1)] = 0.38e-5 * nntot
    # ionpopdict[(at.get_atomic_number('O'), 1)] = 0.94e-4 * nntot
    # ionpopdict[(at.get_atomic_number('Si'), 1)] = 0.63 * nntot
    # ionpopdict[(at.get_atomic_number('Si'), 2)] = 0.63 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('S'), 1)] = 0.29 * nntot
    # ionpopdict[(at.get_atomic_number('S'), 2)] = 0.29 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('Ar'), 1)] = 0.041 * nntot
    # ionpopdict[(at.get_atomic_number('Ca'), 1)] = 0.026 * nntot
    # ionpopdict[(at.get_atomic_number('Fe'), 1)] = 0.012 * nntot

    # velocity = modeldata['velocity_outer'][args.modelgridindex]
    # args.time_days = float(at.get_timestep_time(modelpath, args.timestep))
    # print(f'timestep {args.timestep} cell {args.modelgridindex} (v={velocity} km/s at {args.time_days:.1f}d)')

    ions = []
    for key in ionpopdict.keys():
        # keep only the ion populations, not element or total populations
        if isinstance(key, tuple) and len(key) == 2 and ionpopdict[key] / nntot >= minionfraction:
            ions.append(key)

    ions.sort()

    if args.noexcitation:
        adata = None
        dfpops = None
    else:
        adata = at.get_levels(modelpath, get_transitions=True, ionlist=tuple(ions))
        dfpops = get_lte_pops(adata, ions, ionpopdict, temperature=6000)

    print(f'     nntot: {nntot:.2e} /cm3')
    print(f'       nne: {nne:.2e} /cm3')
    print(f'deposition: {deposition_density_ev:7.2f} eV/s/cm3')

    dfcollion = at.nonthermal.read_colliondata(
        collionfilename=('collion-AR1985.txt' if args.ar1985 else 'collion.txt'))

    # WARNING REMOVE
    dfcollion_thision = dfcollion.query('Z == 26 and ionstage == 3 and l == 2', inplace=True)

    if args.ostat:
        with open(args.ostat, 'w') as fstat:
            fstat.write('emin emax npts FeII_frac_ionization FeII_frac_excitation FeII_gamma_nt '
                        f'NiII_frac_ionization NiII_frac_excitation NiII_gamma_nt\n')

    stepcount = 20 if args.vary else 1
    for step in range(stepcount):
        emin = args.emin
        emax = args.emax
        npts = args.npts
        if args.vary == 'emin':
            emin *= 2 ** step
        elif args.vary == 'emax':
            emax *= 2 ** step
        elif args.vary == 'npts':
            npts *= 2 ** step
        if args.vary == 'emax,npts':
            npts *= 2 ** step
            emax *= 2 ** step
        engrid = np.linspace(emin, emax, num=npts, endpoint=True)
        deltaen = engrid[1] - engrid[0]

        sourcevec = np.zeros(engrid.shape)
        # source_spread_pts = math.ceil(npts / 10.)
        source_spread_pts = math.ceil(npts * 0.03333)
        for s in range(npts):
            # spread the source over some energy width
            if (s < npts - source_spread_pts):
                sourcevec[s] = 0.
            elif (s < npts):
                sourcevec[s] = 1. / (deltaen * source_spread_pts)
        # sourcevec[-1] = 1.
        # sourcevec[-3] = 1.

        yvec, dftransitions = solve_spencerfano(
            ions, ionpopdict, dfpops, nne, deposition_density_ev, engrid, sourcevec, dfcollion, args,
            adata=adata, noexcitation=args.noexcitation)

        if args.makeplot:
            # outputfilename = str(args.outputfile).format(
            #     cell=args.modelgridindex, timestep=args.timestep, time_days=args.time_days)
            outputfilename = 'spencerfano.pdf'
            make_plot(engrid, yvec, ions, ionpopdict, dfcollion, dftransitions, nne, sourcevec,
                      deposition_density_ev, outputfilename, noexcitation=args.noexcitation)

        (frac_excitation, frac_ionization, frac_excitation_ion, frac_ionization_ion, gamma_nt) = analyse_ntspectrum(
            engrid, yvec, ions, ionpopdict, nntot, nne, deposition_density_ev,
            dfcollion, dftransitions, noexcitation=args.noexcitation)

        if args.ostat:
            with open(args.ostat, 'a') as fstat:
                fstat.write(f'{emin} {emax} {npts} {frac_ionization_ion[(26, 2)]:.4f} '
                            f'{frac_excitation_ion[(26, 2)]:.4f} '
                            f'{gamma_nt[(26, 2)]:.4e} {frac_ionization_ion[(28, 2)]:.4f} '
                            f'{frac_excitation_ion[(28, 2)]:.4f} {gamma_nt[(28, 2)]:.4e}\n')


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
