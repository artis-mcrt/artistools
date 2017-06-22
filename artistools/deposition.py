#!/usr/bin/env python3
import math
# import numpy as np
from astropy import units as u
# from astropy import constants as c
import artistools as at


def main():
    dfmodel, t_model_init = at.get_modeldata('model.txt')

    t_init = t_model_init * u.day

    comeanlife = 111.3 * u.day
    mconucleus = 55.9398393 * u.u

    t_now = 200 * u.day
    print(f't_now = {t_now.to("d")}')
    print('The following assumes that all 56Ni has decayed to 56Co and all energy comes from emitted positrons')

    v_inner = 0 * u.km / u.s
    for i, row in dfmodel.iterrows():
        v_outer = row['velocity'] * u.km / u.s

        volume_now = (4 * math.pi / 3 * ((v_outer * t_now) ** 3 - (v_inner * t_now) ** 3)).to('cm3')

        rho_init = (10 ** row['logrho']) * u.g / u.cm ** 3
        volume_init = (4 * math.pi / 3 * (v_outer * t_init) ** 3).to('cm3')
        mco_init = (row['f56ni'] + row['f56co']) * (volume_init * rho_init).to('solMass')

        mco_now = mco_init * math.exp(- t_now / comeanlife)
        # print(f'  Mco(t_now) = {mco_now:.2f}')

        # power_now = (4.566 * u.MeV * (mco_now / mconucleus) / comeanlife).to('erg/s')
        power_now = (0.19 * 0.610 * u.MeV * (mco_now / mconucleus) / comeanlife).to('erg/s')

        epsilon = power_now / volume_now
        print(f'zone {i:3d}, velocity = {v_outer:8.2f}, epsilon = {epsilon:.3e}')
        # print(f'  epsilon = {epsilon.to("eV/(cm3s)"):.2f}')

        v_inner = v_outer


if __name__ == "__main__":
    main()
