#!/usr/bin/env python3
import math
# import numpy as np
from astropy import units as u
# from astropy import constants as c
import artistools as at


def main():
    dfmodel, t_model_init = at.get_modeldata('model.txt')

    t_init = t_model_init * u.day

    meanlife_co56 = 113.7 * u.day
    # define TCOBALT (113.7*DAY)     // cobalt-56
    # T48CR = 1.29602 * u.day
    # T48V = 23.0442 * u.day
    # define T52FE   (0.497429*DAY)
    # define T52MN   (0.0211395*DAY)

    t_now = 200 * u.day
    print(f't_now = {t_now.to("d")}')
    print('The following assumes that all 56Ni has decayed to 56Co and all energy comes from emitted positrons')

    v_inner = 0 * u.km / u.s
    for i, row in dfmodel.iterrows():
        v_outer = row['velocity'] * u.km / u.s

        volume_init = ((4 * math.pi / 3) * ((v_outer * t_init) ** 3 - (v_inner * t_init) ** 3)).to('cm3')

        volume_now = ((4 * math.pi / 3) * ((v_outer * t_now) ** 3 - (v_inner * t_now) ** 3)).to('cm3')

        rho_init = (10 ** row['logrho']) * u.g / u.cm ** 3
        mco56_init = (row['f56ni'] + row['f56co']) * (volume_init * rho_init).to('solMass')
        mco56_now = mco56_init * math.exp(- t_now / meanlife_co56)

        co56_positron_dep = (0.19 * 0.610 * u.MeV * (mco56_now / (55.9398393 * u.u)) / meanlife_co56).to('erg/s')
        v48_positron_dep = 0
        # v48_positron_dep = (0.290 * 0.499 * u.MeV) * (math.exp(-t / T48V) - exp(-t / T48CR)) / (T48V - T48CR) * mcr48_now / MCR48;

        power_now = co56_positron_dep + v48_positron_dep

        epsilon = power_now / volume_now
        print(f'zone {i:3d}, velocity = {v_outer:8.2f}, epsilon = {epsilon:.3e}')
        # print(f'  epsilon = {epsilon.to("eV/(cm3s)"):.2f}')

        v_inner = v_outer


if __name__ == "__main__":
    main()
