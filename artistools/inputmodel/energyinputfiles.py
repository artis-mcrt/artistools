import argparse
import itertools
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at
from artistools.constants import day_to_s


def _cumulative_trapezoid(y: npt.ArrayLike, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Cumulatively integrate y over x with the trapezoidal rule, starting from zero."""
    yarr = np.asarray(y, dtype=np.float64)
    xarr = np.asarray(x, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(np.diff(xarr) * (yarr[:-1] + yarr[1:]) / 2.0)))


def _quad_adaptive(func: Callable[[float], float], a: float, b: float, *, rtol: float = 1.5e-8) -> float:
    """Integrate a smooth function over [a, b] with adaptive Simpson quadrature.

    rtol must stay above the rounding noise of the integrand (e.g. cancellation in
    1/2 - arctan(x)/pi at large x leaves ~1e-9 relative noise), or the recursion never converges.
    """

    def simpson(x0: float, x2: float, f0: float, f1: float, f2: float) -> float:
        return (x2 - x0) / 6.0 * (f0 + 4.0 * f1 + f2)

    def recurse(x0: float, x2: float, f0: float, f1: float, f2: float, whole: float, depth: int) -> float:
        x1 = 0.5 * (x0 + x2)
        fleft = func(0.5 * (x0 + x1))
        fright = func(0.5 * (x1 + x2))
        left = simpson(x0, x1, f0, fleft, f1)
        right = simpson(x1, x2, f1, fright, f2)
        if depth <= 0 or abs(left + right - whole) <= rtol * (abs(left) + abs(right)):
            # Richardson extrapolation of the two half-interval estimates
            return left + right + (left + right - whole) / 15.0
        return recurse(x0, x1, f0, fleft, f1, left, depth - 1) + recurse(x1, x2, f1, fright, f2, right, depth - 1)

    x1 = 0.5 * (a + b)
    fa, f1, fb = func(a), func(x1), func(b)
    return recurse(a, b, fa, f1, fb, simpson(a, b, fa, f1, fb), 30)


def get_cumulative_heating_fraction() -> tuple[pl.DataFrame, float]:

    tmin = 0.0001  # days
    tmax = 50

    times = np.logspace(np.log10(tmin), np.log10(tmax), num=300)  # days
    qdot = 5e9 * (times) ** (-1.3)  # define energy power law (5e9*t^-1.3)

    cumulative_energy = _cumulative_trapezoid(y=qdot, x=times)
    E_tot = float(cumulative_energy[-1])
    # print("Etot per gram", E_tot, E_tot*1.989e33*0.01)

    rate = cumulative_energy / E_tot

    times_and_rate = {"times": times, "rate": rate}
    dftimes_and_rate = pl.DataFrame(data=times_and_rate)

    dE = np.diff(dftimes_and_rate["rate"] * E_tot)
    dt = np.diff(times * 24 * 60 * 60)

    integrated_rate = dE / dt
    scale_factor_energy_diff = max(qdot[1:] / integrated_rate)
    print(np.mean(scale_factor_energy_diff))
    E_tot *= scale_factor_energy_diff
    # print(f"E_tot after integrated line scaled to match energy of power law: {E_tot}")

    dE = np.diff(dftimes_and_rate["rate"] * E_tot)
    dt = np.diff(times * 24 * 60 * 60)

    # check energy rate is on top of power law line
    # plt.plot(dftimes_and_rate["times"][1:], (dE / dt) * 0.01 * Msun_to_g)
    # plt.plot(dftimes_and_rate["times"], qdot * 0.01 * Msun_to_g)
    # plt.yscale("log")
    # plt.xscale("log")

    # plt.xlabel("Time [days]")
    # plt.ylabel("Q [erg/g/s]")
    # # plt.xlim(0.1, 20)
    # # plt.ylim(5e39, 2e41)
    # plt.show()

    return dftimes_and_rate, E_tot


def make_energydistribution_weightedbyrho(
    rho: npt.NDArray[np.floating], E_tot_per_gram: float, Mtot_grams: float
) -> pl.DataFrame:
    print(f"energy distribution weighted by rho (E_tot per gram {E_tot_per_gram})")
    Etot = E_tot_per_gram * Mtot_grams
    print("Etot", Etot)
    numberofcells = len(rho)

    cellenergy = np.array([Etot] * numberofcells)
    cellenergy *= rho / sum(rho)

    energydistdata = {"cellid": np.arange(1, len(rho) + 1), "cell_energy": cellenergy}

    print(f"sum energy cells {sum(energydistdata['cell_energy'])} should equal Etot")
    return pl.DataFrame(data=energydistdata)


def make_energy_files(rho: npt.NDArray[np.floating], Mtot_grams: float, outputpath: Path | str) -> None:
    print("Using power law for energy rate")
    times_and_rate, E_tot_per_gram = get_cumulative_heating_fraction()

    energydistributiondata = make_energydistribution_weightedbyrho(rho, E_tot_per_gram, Mtot_grams)

    print("Writing energydistribution.txt")
    with Path(outputpath, "energydistribution.txt").open("w", encoding="utf-8") as fmodel:
        fmodel.write(f"{len(energydistributiondata['cell_energy'])}\n")  # write number of points
        energydistributiondata.to_pandas().to_csv(fmodel, header=False, sep="\t", index=False, float_format="%g")

    print("Writing energyrate.txt")
    with Path(outputpath, "energyrate.txt").open("w", encoding="utf-8") as fmodel:
        fmodel.write(f"{len(times_and_rate['times'])}\n")  # write number of points
        times_and_rate.to_pandas().to_csv(fmodel, sep="\t", index=False, header=False, float_format="%.10f")


def rprocess_const_and_powerlaw() -> tuple[pl.DataFrame, float]:
    """Following eqn 4 Korobkin 2012."""
    tmin = 0.01 * day_to_s
    tmax = 50 * day_to_s
    t0 = 1.3  # seconds
    epsilon0 = 2e18
    sigma = 0.11
    alpha = 1.3
    thermalisation_factor = 0.5

    def integrand(t_sec: float) -> float:
        return float(epsilon0 * ((1 / 2) - (1 / np.pi * np.arctan((t_sec - t0) / sigma))) ** alpha) * (
            thermalisation_factor / 0.5
        )

    times = np.logspace(np.log10(tmin), np.log10(tmax), num=200)
    energy_per_gram_cumulative = [0.0]
    for tlow, thigh in itertools.pairwise(times):
        energy_per_gram_cumulative.append(energy_per_gram_cumulative[-1] + _quad_adaptive(integrand, tlow, thigh))

    E_tot = energy_per_gram_cumulative[-1]  # ergs/g
    print("Etot per gram", E_tot)

    rate = np.array(energy_per_gram_cumulative) / E_tot

    nuclear_heating_power = [integrand(time) for time in times]

    times_and_rate = {"times": times / day_to_s, "rate": rate, "nuclear_heating_power": nuclear_heating_power}
    dftimes_and_rate = pl.DataFrame(data=times_and_rate)

    return dftimes_and_rate, E_tot


def energy_from_rprocess_calculation(
    energy_thermo_data: pl.DataFrame, get_rate: bool = True
) -> float | tuple[pl.DataFrame, float]:

    energy_thermo_data = energy_thermo_data.filter(pl.col("time/s") <= 1e7)
    # print("Dropping times later than 116 days")

    skipfirstnrows = 0  # not sure first values look sensible -- check this
    times = energy_thermo_data["time/s"][skipfirstnrows:]
    qdot = energy_thermo_data["Qdot"][skipfirstnrows:]

    cumulative_energy = _cumulative_trapezoid(y=qdot, x=times)
    E_tot = float(cumulative_energy[-1])  # erg / g

    if get_rate:
        print(f"E_tot {E_tot} erg/g")
        rate = cumulative_energy / E_tot

        dftimes_and_rate = pl.DataFrame({"times": times / day_to_s, "rate": rate})

        return dftimes_and_rate, E_tot

    return E_tot


def plot_energy_rate(modelpath: str | Path, axis: mplax.Axes) -> None:
    times_and_rate, _ = at.inputmodel.energyinputfiles.rprocess_const_and_powerlaw()
    lzmodel, _ = at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g"])
    model = lzmodel.collect()
    Mtot_grams = model["mass_g"].sum()
    axis.plot(
        times_and_rate["times"], np.array(times_and_rate["nuclear_heating_power"]) * Mtot_grams, color="k", zorder=10
    )


def get_etot_fromfile(modelpath: str | Path) -> tuple[float, pl.DataFrame]:
    energydistribution_data = pl.from_pandas(
        pd.read_csv(
            Path(modelpath) / "energydistribution.txt",
            skiprows=1,
            sep=r"\s+",
            header=None,
            names=["cellid", "cell_energy"],
        )
    )
    etot = float(energydistribution_data["cell_energy"].sum())
    return etot, energydistribution_data


def get_energy_rate_fromfile(modelpath: str | Path) -> pl.DataFrame:
    return pl.from_pandas(
        pd.read_csv(Path(modelpath) / "energyrate.txt", skiprows=1, sep=r"\s+", header=None, names=["times", "rate"])
    )


def read_trajectory_thermo(trajthermofile: Path | str) -> pl.DataFrame:
    """Read the time/s and Qdot columns of a trajectory energy_thermo.dat.

    Times below one second are dropped, matching get_trajectory_qdotintegral: Qdot is negative there
    and integrating over it gives a negative total energy.
    """
    dfthermo = pl.from_pandas(
        pd.read_csv(trajthermofile, sep=r"\s+", usecols=["time/s", "Qdot"], engine="c", dtype_backend="pyarrow")
    )

    return dfthermo.filter(pl.col("time/s") >= 1.0)


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        choices=["plotrate", "describe", "fromtrajectory"],
        help=(
            "plotrate: plot the analytic nuclear heating power against time."
            " describe: report the total energy and rate from the written energy files."
            " fromtrajectory: integrate a trajectory energy_thermo.dat to get the total energy and rate."
        ),
    )
    at.add_modelpath_arg(parser, default=Path())
    at.add_outputfile_arg(parser, helptext="Path for the plot, or omit to show it interactively")
    parser.add_argument("-trajthermofile", type=Path, help="Trajectory energy_thermo.dat (fromtrajectory)")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot and inspect the ARTIS energy input files."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.action is None:
        print("ERROR: no action given. Run with --help to see the available actions.")
        raise SystemExit(1)

    modelpath = Path(args.modelpath)

    if args.action == "plotrate":
        fig, axis = plt.subplots()
        plot_energy_rate(modelpath, axis)
        axis.set_xlabel("time [days]")
        axis.set_ylabel("Nuclear heating power [erg/s]")
        axis.set_xscale("log")
        axis.set_yscale("log")
        if args.outputfile:
            fig.savefig(args.outputfile)
            at.print_saved(args.outputfile)
        else:
            plt.show()
        plt.close(fig)

    elif args.action == "describe":
        etot, energydistribution = get_etot_fromfile(modelpath)
        print(f"energydistribution.txt: {len(energydistribution)} cells, total energy {etot:.4e} erg")
        dfrate = get_energy_rate_fromfile(modelpath)
        times, rates = dfrate["times"].to_numpy(), dfrate["rate"].to_numpy()
        tmin, tmax = float(times.min()), float(times.max())
        ratemin, ratemax = float(rates.min()), float(rates.max())
        print(f"energyrate.txt: {len(dfrate)} times from {tmin:g} to {tmax:g} days")
        print(f"  cumulative energy fraction runs {ratemin:.4g} to {ratemax:.4g}")

    else:
        if args.trajthermofile is None:
            print("ERROR: fromtrajectory requires -trajthermofile")
            raise SystemExit(1)
        result = energy_from_rprocess_calculation(read_trajectory_thermo(args.trajthermofile), get_rate=True)
        assert isinstance(result, tuple)
        dftimes_and_rate, e_tot = result
        print(f"E_tot {e_tot:.4e} erg/g over {len(dftimes_and_rate)} times")


if __name__ == "__main__":
    main()
