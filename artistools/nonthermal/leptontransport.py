import math

CONST_EV_IN_J = 1.602176634e-19  # 1 eV [J]


def calculate_dE_on_dx(energy, n_e_bound):
    # Barnes et al. (2016) electron loss rate to ionisation and excitation
    # in [J / m] (will always be negative)

    assert(energy > 0)
    # r_e = 2.8179403262e-13  # classical electron radius [cm]
    # # m_e = 9.10938356e-28  # mass of electron [g]
    # m_e = 511e3  # mass of electron [eV/c2]
    # c = 29979245800  # [cm / s]

    r_e = 2.8179403262e-15  # classical electron radius [m]
    m_e = 9.10938356e-31  # mass of electron [kg]
    c = 299792458  # [m / s]
    energy_ev = energy / CONST_EV_IN_J  # [eV]
    tau = energy / (m_e * c ** 2)
    gamma = tau + 1
    beta = math.sqrt(1 - 1. / gamma ** 2)
    v = beta * c

    Z = 26

    I_ev = 9.1 * Z * (1 + 1.9 * Z ** (-2/3.))  # mean ionisation potention [eV]
    # I_ev = 287.8  # [eV]

    g = 1 + tau ** 2 / 8 - (2 * tau + 1) * math.log(2)

    de_on_dt = (
        2 * math.pi * r_e ** 2 * m_e * c ** 3 * n_e_bound / beta *
        # (2 * math.log(energy_ev / I_ev) + 1 - math.log(2)))
        (2 * math.log(energy_ev / I_ev) + math.log(1 + tau / 2.) + (1 - beta ** 2) * g))

    # print(f'{energy_ev=} {de_on_dt=} J/s {(de_on_dt / 1.602176634e-19)=} eV/s')
    de_on_dx = de_on_dt / v

    return -de_on_dx


def main():
    E_0 = 1e6 * CONST_EV_IN_J  # initial energy [J]
    n_e_bound_cgs = 1e5 * 26  # density of bound electrons in [cm-3]
    n_e_bound = n_e_bound_cgs * 1e6  # [m^-3]
    print(f'initial energy: {E_0 / CONST_EV_IN_J:.1e} [eV]')
    print(f'n_e_bound: {n_e_bound_cgs:.1e} [cm-3]')
    energy = E_0
    mean_free_path = 0.
    x = 0  # distance moved [m]
    steps = 0
    while True:
        steps += 1
        delta_energy = -.5 * CONST_EV_IN_J
        dE_on_dx = calculate_dE_on_dx(energy, n_e_bound)
        x += delta_energy / dE_on_dx
        mean_free_path += -x * delta_energy / E_0
        if steps % 1000000 == 0:
            print(f'E: {energy / CONST_EV_IN_J:.1f} eV x: {x:.1e} dE_on_dx: {dE_on_dx}')
        energy += delta_energy
        if energy <= 0:
            break

    print(f'steps: {steps}')
    print(f'final energy: {energy / CONST_EV_IN_J:.1e} eV')
    print(f'distance travelled: {x:.1} m')
    print(f'mean free path: {mean_free_path:.1} m')


if __name__ == "__main__":
    main()
