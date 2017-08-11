#!/usr/bin/env python3

import math

import numpy as np
from astropy import units as u


def min_dist(listin, number):
    min_dist_found = -1

    for x in listin:
        dist = abs(x - number)
        if dist < min_dist_found or min_dist_found < 0:
            min_dist_found = dist

    return min_dist_found


def main():
    e_k = 1.2  # in units of 10^51 erg
    m_ej = 1.4  # in solar masses
    x_stb = 0.05  # mass fraction of stable Fe54 and Ni58 in Ni56 zone
    t200 = 0.0002 / 200  # time in units of 200 days

    delta = 0
    n = 10

    # density transition
    v_transition = 10943 * e_k ** 0.5 * m_ej ** -0.5  # km/s
    rho_0 = 4.9e-17 * (e_k ** -1.5) * (m_ej ** 2.5) * (t200 ** -3)  # g cm^-3

    print(f'v_transition = {v_transition:.3f}')

    # composition transition from Ni56-rich to IME-rich
    mni56 = 0.6 * u.solMass
    volni56 = (mni56 / ((1 - x_stb) * rho_0 * u.g * u.cm ** -3)).to('cm3')
    rni56 = (3 / 4 / math.pi * volni56) ** (1/3.)
    v_ni56 = (rni56 / (200 * t200 * u.day)).to('km/s').value

    r = (v_ni56 * (u.km / u.s) * 200 * t200 * u.day).to('cm')
    m = (4 * math.pi / 3 * (r ** 3) * (rho_0 * u.g * u.cm ** -3)).to('solMass')
    print(f'Ni56 region outer velocity = {v_ni56:.3f}, M={m:.3f}')

    with open('model.txt', 'w') as fmodel:
        with open('abundances.txt', 'w') as fabundances:

            fixed_points = [v_transition, v_ni56]
            regular_points = [v for v in np.arange(0, 14500, 1000)[1:] if min_dist(fixed_points, v) > 200]
            vlist = sorted(list([*fixed_points, *regular_points]))

            fmodel.write(f'{len(vlist)}\n')
            fmodel.write(f'{t200 * 200:f}\n')

            v_inner = 0.  # velocity at inner boundary of cell
            m_tot = 0.
            for index, v_outer in enumerate(vlist, 1):  # km / s
                rho = rho_0 * (0.5 * (v_inner + v_outer) / v_transition) ** -(delta if v_outer <= v_transition else n)
                if v_outer <= v_ni56:
                    # Ni56-rich zone
                    radioabundances = "1.0   0.95  0.0   0.0   0.0"
                    abundances = "0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.025   0.0   0.975   0.0   0.0"
                else:
                    # Intermediate-mass elements
                    radioabundances = "0.0   0.0   0.0   0.0   0.0"
                    abundances = "0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.7   0.0   0.29  0.0   0.0   0.0   0.01  0.0   0.0   0.0   0.0   0.0   0.000   0.0   0.000   0.0   0.0"

                fmodel.write(f'{index:6d}   {v_outer:9.2f}   {math.log10(rho):10.8f}   {radioabundances}\n')
                fabundances.write(f'{index:6d}   {abundances}\n')
                r_inner = (v_inner * u.km / u.s * t200 * 200 * u.day).to('cm').value
                r_outer = (v_outer * u.km / u.s * t200 * 200 * u.day).to('cm').value
                vol_shell = 4 * math.pi / 3 * (r_outer ** 3 - r_inner ** 3)
                m_shell = rho * vol_shell / u.solMass.to('g')
                m_tot += m_shell

                v_inner = v_outer
            print(f'M_tot = {m_tot:.3f} solMass')


if __name__ == "__main__":
    main()
