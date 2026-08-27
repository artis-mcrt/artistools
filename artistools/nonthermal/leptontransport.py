"""Track the energy loss of a fast lepton to plasma, ionisation, and excitation, following Barnes et al. (2016)."""

import argparse
import math
import typing as t
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

import artistools as at
from artistools.constants import K_B_ev_per_K as CONST_KB  # Boltzmann constant [eV / K]
from artistools.plottools import save_figure

# CONST_KB above is shared with the rest of artistools. The constants below stay local because this module works in
# SI units (J, m, kg, s), which artistools.constants does not provide
CONST_EV_IN_J = 1.602176634e-19  # 1 eV [J]

CONST_RE = 2.8179403262e-15  # classical electron radius [m]
CONST_ME = 9.10938356e-31  # mass of electron [kg]
CONST_C = 299792458  # [m / s]

defaultoutputfile = "leptontransport.pdf"


def calculate_dE_on_dx_plasma(energy: float, n_e_free: float) -> float:
    """Return the electron energy loss rate to the plasma [J/m], from Barnes et al. (2016) eq 4.

    The result is always negative.
    """
    assert energy > 0

    energy_ev = energy / CONST_EV_IN_J  # [eV]
    energy_mev = energy / CONST_EV_IN_J / 1e6  # [MeV]
    tau = energy / (CONST_ME * CONST_C**2)
    gamma = tau + 1
    beta = math.sqrt(1 - 1.0 / gamma**2)
    v = beta * CONST_C
    T_K = 11604
    T_mev = CONST_KB * T_K * 1e-6  # temperature in [MeV]

    de_on_dt: float | int = (
        1e6
        * CONST_EV_IN_J
        * (7e-15 * (energy_mev**-0.5) * (n_e_free / 1e6) * 10 * (1.0 - 3.9 / 7.7 * T_mev / energy_mev))
    )

    # print(f'{energy_mev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s')
    # if energy_ev > 900 and energy_ev < 1100:
    #     print(f'{energy_mev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s')

    de_on_dx = de_on_dt / v
    if de_on_dx < 0.0:
        print(f"plasma loss negative {energy_ev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s")
        de_on_dx = -de_on_dx  # weird minus sign shows up around energy = I = 240 eV

    return -de_on_dx


def calculate_dE_on_dx_ionexc(energy: float, n_e_bound: float) -> float:
    """Return the electron energy loss rate to ionisation and excitation [J/m], from Barnes et al. (2016).

    The result is always negative.
    """
    assert energy > 0
    energy_ev = energy / CONST_EV_IN_J  # [eV]
    tau = energy / (CONST_ME * CONST_C**2)
    gamma = tau + 1
    beta = math.sqrt(1 - 1.0 / gamma**2)
    v = beta * CONST_C

    Z = 26

    I_ev = 9.1 * Z * (1 + 1.9 * Z ** (-2 / 3.0))  # mean ionisation potential [eV]
    # I_ev = 287.8  # [eV]

    g = 1 + tau**2 / 8 - (2 * tau + 1) * math.log(2)

    de_on_dt = (
        2
        * math.pi
        * CONST_RE**2
        * CONST_ME
        * CONST_C**3
        * n_e_bound
        / beta
        *
        # (2 * math.log(energy_ev / I_ev) + 1 - math.log(2)))
        (2 * math.log(energy_ev / I_ev) + math.log(1 + tau / 2.0) + (1 - beta**2) * g)
    )

    # print(f'{energy_ev=} {de_on_dt=} J/s {(de_on_dt / 1.602176634e-19)=} eV/s')
    de_on_dx = de_on_dt / v

    if de_on_dx < 0.0:
        print(f"ion/exc loss negative {energy_ev=} {de_on_dt=} J/s {(de_on_dt / CONST_EV_IN_J)=} eV/s")
        de_on_dx = -de_on_dx  # weird minus sign shows up around energy = I = 240 eV

    return -de_on_dx


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-energy", type=float, default=1e5, help="Initial lepton energy in eV")
    parser.add_argument("-nnebound", type=float, default=1e5 * 26, help="Number density of bound electrons in cm^-3")
    parser.add_argument("-nnefree", type=float, default=1e5, help="Number density of free electrons in cm^-3")
    parser.add_argument("-nsteps", type=int, default=1000000, help="Number of energy steps to integrate over")
    at.addarg_output(parser, kind="file", default=defaultoutputfile, astype=None, helptext="Filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Integrate a fast lepton's energy loss over distance and plot the result."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)

    E_0 = args.energy * CONST_EV_IN_J  # initial energy [J]
    n_e_bound_cgs = args.nnebound  # density of bound electrons in [cm-3]
    n_e_bound = n_e_bound_cgs * 1e6  # [m^-3]
    n_e_free_cgs = args.nnefree
    n_e_free = n_e_free_cgs * 1e6  # [m^-3]
    # both stopping-power helpers require energy > 0, and without their asserts (python -O) a zero energy
    # reaches a division by beta = 0 and log(0)
    if args.energy <= 0.0:
        msg = f"-energy must be positive, not {args.energy}"
        raise ValueError(msg)
    # both helpers return the magnitude of a loss rate, so a negative density would contribute as a positive one
    if n_e_bound < 0.0 or n_e_free < 0.0:
        msg = "-nnebound and -nnefree are number densities and cannot be negative"
        raise ValueError(msg)
    if n_e_bound == n_e_free == 0.0:
        msg = "at least one of -nnebound and -nnefree must be positive, otherwise the lepton never loses energy"
        raise ValueError(msg)
    if args.nsteps < 1:
        msg = f"-nsteps must be at least 1, not {args.nsteps}"
        raise ValueError(msg)
    print(f"initial energy: {E_0 / CONST_EV_IN_J:.1e} [eV]")
    print(f"n_e_bound: {n_e_bound_cgs:.1e} [cm-3]")
    arr_energy_ev = []
    arr_dist = []
    arr_dE_on_dx_ionexc = []
    arr_dE_on_dx_plasma = []
    energy = E_0
    mean_free_path = 0.0
    delta_energy = -E_0 / args.nsteps
    x = 0.0  # distance moved [m]
    steps = 0
    while True:
        energy_ev = energy / CONST_EV_IN_J
        arr_dist.append(x)
        arr_energy_ev.append(energy_ev)

        dE_on_dx_ionexc = calculate_dE_on_dx_ionexc(energy, n_e_bound)
        arr_dE_on_dx_ionexc.append(-dE_on_dx_ionexc / CONST_EV_IN_J)
        dE_on_dx_plasma = calculate_dE_on_dx_plasma(energy, n_e_free)
        arr_dE_on_dx_plasma.append(-dE_on_dx_plasma / CONST_EV_IN_J)
        # the lepton loses energy to both channels at once, so the trajectory follows the total stopping power.
        # Using the ion/exc term alone would make -nnefree affect only the plotted loss curve, and would divide
        # by zero for a fully ionised plasma
        dE_on_dx = dE_on_dx_ionexc + dE_on_dx_plasma
        if steps % 100000 == 0:
            print(
                f"E: {energy / CONST_EV_IN_J:.1f} eV x: {x:.1e} dE_on_dx_ionexc: {dE_on_dx}, dE_on_dx_plasma:"
                f" {dE_on_dx_plasma}"
            )
        x += delta_energy / dE_on_dx
        mean_free_path += -x * delta_energy / E_0
        energy += delta_energy

        steps += 1
        if energy <= 0:
            break

    print(f"steps: {steps}")
    print(f"final energy: {energy / CONST_EV_IN_J:.1e} eV")
    print(f"distance travelled: {x:.1} m")
    print(f"mean free path: {mean_free_path:.1} m")

    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=False, figsize=(5, 8), tight_layout={"pad": 0.5, "w_pad": 0.0, "h_pad": 1.0}
    )
    assert isinstance(axes, np.ndarray)
    axes[0].plot(arr_dist, arr_energy_ev)
    axes[0].set_xlabel(r"Distance [m]")
    axes[0].set_ylabel(r"Energy [eV]")
    axes[0].set_yscale("log")

    axes[1].plot(arr_energy_ev, arr_dE_on_dx_ionexc, label="ion/exc")
    axes[1].plot(arr_energy_ev, arr_dE_on_dx_plasma, label="plasma")
    axes[1].set_xlabel(r"Energy [eV]")
    axes[1].set_ylabel(r"dE/dx [eV / m]")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend()
    save_figure(fig, outputfile, format="pdf")


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
