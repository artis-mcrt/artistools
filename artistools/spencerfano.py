#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

import artistools as at
import artistools.nonthermalspec

def lossfunction(energy, nne):
    h = 4.1356e-15   # Planck's constant in eV * s
    e = 1.609e-19    # elementary charge in Coulombs
    me = 0.510998e6  # mass of electron in eV/c^2
    eulergamma = 0.577215664901532

    zetae = h / (2 * math.pi) * math.sqrt((4 * math.pi * nne * (e ** 2)) / me)
    omegap = 2 * math.pi * zetae / h
    v = math.sqrt(2 * energy / me)
    if energy > 14:
        return nne * 2 * math.pi * (e ** 4) / energy * math.log(2 * energy / zetae)
    else:
        return nne * 2 * math.pi * (e ** 4) / energy * math.log(me * (v ** 3) / (eulergamma * (e ** 2) * omegap))


def Psecondary(epsilon, e_p, I, J):
    e_s = epsilon - I
    return 1 / (J * math.atan((e_p - I) / (2 * J)) * (1 + ((e_s / J) ** 2)))


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


def main():
    # estimators = read_estimators(modelpath, modeldata)

    fs = 13

    npts = 2000
    engrid = np.linspace(1, 3000, num=npts, endpoint=False)
    source = np.zeros(engrid.shape)

    sfmatrix = np.zeros((npts, npts))
    constvec = np.zeros(npts)

    nntot = 1e10
    nne = 1e8   # density of electrons in number per cm^3

    enmax = engrid[-1]
    deltaen = engrid[1] - engrid[0]

    source_spread_pts = math.ceil(npts * 0.03333)
    for s in range(npts):
        # spread the source over some energy width
        if (s < npts - source_spread_pts):
            source[s] = 0.
        elif (s < npts):
            source[s] = 1. / (deltaen * source_spread_pts)

    E_init_ev = 0
    for j in range(npts):
        E_init_ev += source[j] * deltaen

    for i in range(npts):
        for j in range(i, npts):
            constvec[i] += source[j] * deltaen

    for i in range(npts):
        en = engrid[i]
        sfmatrix[i, i] += lossfunction(en, nne)

    dfcollion = at.nonthermalspec.read_colliondata()

    composition = [(26, 2, 0.2 * 0.8 * nntot),
                   (26, 2, 0.8 * 0.8 * nntot),
                   (28, 2, 0.2 * nntot)]

    # composition = [
    #     (8, 1, 0.99 * nntot),
    #     (8, 2, 0.01 * nntot)
    # ]

    for Z, ionstage, nnion in composition:
        dfcollion_thision = dfcollion.query('Z == @Z and ionstage == @ionstage')
        print(dfcollion_thision)
        for index, row in dfcollion_thision.iterrows():
            ar_xs_array = np.array([at.nonthermalspec.ar_xs(energy_ev, row.ionpot_ev, row.A, row.B, row.C, row.D) for energy_ev in engrid])

            ionpot_ev = row.ionpot_ev
            J = get_J(row.Z, row.ionstage, ionpot_ev)
            for i in range(npts):
                en = engrid[i]

                for j in range(i, npts):
                    endash = engrid[j]

                    prefactor = nnion * ar_xs_array[j] / (J * math.atan((endash - ionpot_ev) / (2 * J)))

                    epsilonb = (endash + ionpot_ev) / 2
                    epsilona = endash - en
                    ij_contribution = prefactor * J * (math.atan((epsilonb - ionpot_ev) / J) - math.atan((epsilona - ionpot_ev) / J)) * deltaen

                    if endash >= 2 * en + ionpot_ev:
                        epsilona = en + ionpot_ev
                        ij_contribution -= prefactor * J * (math.atan((epsilonb - ionpot_ev) / J) - math.atan((epsilona - ionpot_ev) / J)) * deltaen

                    sfmatrix[i, j] += ij_contribution


    lu_and_piv = linalg.lu_factor(sfmatrix, overwrite_a=False)
    yvec = linalg.lu_solve(lu_and_piv, constvec, trans=0)
    yvec *= 1e10

    # print("\n".join(["{:} {:11.5e}".format(i, y) for i, y in enumerate(yvec)]))

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 4), tight_layout={"pad": 0.3, "w_pad": 0.0, "h_pad": 0.0})

    ax.plot(engrid[10:], np.log10(yvec[10:]), marker="None", lw=1.5, color='black')

    #    plt.setp(plt.getp(ax, 'xticklabels'), fontsize=fsticklabel)
    #    plt.setp(plt.getp(ax, 'yticklabels'), fontsize=fsticklabel)
    #    for axis in ['top','bottom','left','right']:
    #        ax.spines[axis].set_linewidth(framewidth)
    #    ax.annotate(modellabel, xy=(0.97, 0.95), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top', fontsize=fs)
    # ax.set_yscale('log')
    ax.set_xlim(xmin=engrid[0], xmax=engrid[-1] * 1.0)
    ax.set_ylim(ymin=14, ymax=16.5)
    ax.set_xlabel(r'Electron energy [eV]', fontsize=fs)
    ax.set_ylabel(r'y(E)', fontsize=fs)

    fig.savefig(__file__ + '.pdf', format='pdf')
    plt.close()

if __name__ == "__main__":
    main()
