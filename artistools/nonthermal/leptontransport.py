#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt

CONST_EV_IN_J = 1.602176634e-19  # 1 eV [J]

CONST_RE = 2.8179403262e-15  # classical electron radius [m]
CONST_ME = 9.10938356e-31  # mass of electron [kg]
CONST_C = 299792458  # [m / s]
CONST_KB = 8.617333262145e-5  # Boltzmann constant [eV / K]


def calculate_dE_on_dx_plasma(energy, n_e_free):
    # Barnes et al. (2016) eq 4
    # electron loss rate to plasma
    # in [J / m] (will always be negative)

    assert energy > 0

    energy_ev = energy / CONST_EV_IN_J  # [eV]
    energy_mev = energy / CONST_EV_IN_J / 1e6  # [MeV]
    tau = energy / (CONST_ME * CONST_C ** 2)
    gamma = tau + 1
    beta = math.sqrt(1 - 1. / gamma ** 2)
    v = beta * CONST_C
    T_K = 11604
    T_mev = CONST_KB * T_K * 1e-6  # temperature in [MeV]

    de_on_dt = 1e6 * CONST_EV_IN_J * (
        7e-15 * (energy_mev ** -.5) * (n_e_free / 1e6) * 10 *
        (1. - 3.9 / 7.7 * T_mev / energy_mev)
    )

    # print(f'{energy_mev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s')
    # if energy_ev > 900 and energy_ev < 1100:
    #     print(f'{energy_mev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s')

    de_on_dx = de_on_dt / v
    if de_on_dx < 0.:
        print(f'plasma loss negative {energy_ev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s')
        de_on_dx = -de_on_dx  # weird minus sign shows up around energy = I = 240 eV

    return -de_on_dx


def calculate_dE_on_dx_ionexc(energy, n_e_bound):
    # Barnes et al. (2016) electron loss rate to ionisation and excitation
    # in [J / m] (will always be negative)

    assert energy > 0
    energy_ev = energy / CONST_EV_IN_J  # [eV]
    tau = energy / (CONST_ME * CONST_C ** 2)
    gamma = tau + 1
    beta = math.sqrt(1 - 1. / gamma ** 2)
    v = beta * CONST_C

    Z = 26

    I_ev = 9.1 * Z * (1 + 1.9 * Z ** (-2/3.))  # mean ionisation potention [eV]
    # I_ev = 287.8  # [eV]

    g = 1 + tau ** 2 / 8 - (2 * tau + 1) * math.log(2)

    de_on_dt = (
        2 * math.pi * CONST_RE ** 2 * CONST_ME * CONST_C ** 3 * n_e_bound / beta *
        # (2 * math.log(energy_ev / I_ev) + 1 - math.log(2)))
        (2 * math.log(energy_ev / I_ev) + math.log(1 + tau / 2.) + (1 - beta ** 2) * g))

    # print(f'{energy_ev=} {de_on_dt=} J/s {(de_on_dt / 1.602176634e-19)=} eV/s')
    de_on_dx = de_on_dt / v

    if de_on_dx < 0.:
        print(f'ion/exc loss negative {energy_ev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s')
        de_on_dx = -de_on_dx  # weird minus sign shows up around energy = I = 240 eV

    return -de_on_dx


def main():
    E_0_ev = 1e5  # initial energy [eV]
    E_0 = E_0_ev * CONST_EV_IN_J  # initial energy [J]
    n_e_bound_cgs = 1e5 * 26  # density of bound electrons in [cm-3]
    n_e_bound = n_e_bound_cgs * 1e6  # [m^-3]
    n_e_free_cgs = 1e5
    n_e_free = n_e_free_cgs * 1e6  # [m^-3]
    print(f'initial energy: {E_0 / CONST_EV_IN_J:.1e} [eV]')
    print(f'n_e_bound: {n_e_bound_cgs:.1e} [cm-3]')
    arr_energy_ev = []
    arr_dist = []
    arr_dE_on_dx_ionexc = []
    arr_dE_on_dx_plasma = []
    energy = E_0
    mean_free_path = 0.
    delta_energy = -E_0 / 1000000
    x = 0  # distance moved [m]
    steps = 0
    while True:
        energy_ev = energy / CONST_EV_IN_J
        arr_dist.append(x)
        arr_energy_ev.append(energy_ev)

        dE_on_dx_ionexc = calculate_dE_on_dx_ionexc(energy, n_e_bound)
        arr_dE_on_dx_ionexc.append(-dE_on_dx_ionexc / CONST_EV_IN_J)
        dE_on_dx_plasma = calculate_dE_on_dx_plasma(energy, n_e_free)
        arr_dE_on_dx_plasma.append(-dE_on_dx_plasma / CONST_EV_IN_J)
        dE_on_dx = dE_on_dx_ionexc
        if steps % 100000 == 0:
            print(f'E: {energy / CONST_EV_IN_J:.1f} eV x: {x:.1e} dE_on_dx_ionexc: '
                  f'{dE_on_dx_ionexc}, dE_on_dx_plasma: {dE_on_dx_plasma}')
        x += delta_energy / dE_on_dx
        mean_free_path += -x * delta_energy / E_0
        energy += delta_energy

        steps += 1
        if energy <= 0:
            break

    print(f'steps: {steps}')
    print(f'final energy: {energy / CONST_EV_IN_J:.1e} eV')
    print(f'distance travelled: {x:.1} m')
    print(f'mean free path: {mean_free_path:.1} m')

    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=False, figsize=(5, 8), tight_layout={"pad": 0.5, "w_pad": 0.0, "h_pad": 1.0})
    axes[0].plot(arr_dist, arr_energy_ev)
    axes[0].set_xlabel(r'Distance [m]')
    axes[0].set_ylabel(r'Energy [eV]')
    axes[0].set_yscale('log')

    axes[1].plot(arr_energy_ev, arr_dE_on_dx_ionexc, label='ion/exc')
    axes[1].plot(arr_energy_ev, arr_dE_on_dx_plasma, label='plasma')
    axes[1].set_xlabel(r'Energy [eV]')
    axes[1].set_ylabel(r'dE/dx [eV / m]')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].legend()
    # plt.show()
    fig.savefig('leptontransport.pdf', format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
