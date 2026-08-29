"""Write and inspect the energydistribution.txt and energyrate.txt radioactive energy input files."""

import argparse
import itertools
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.constants import day_to_s
from artistools.misc import addarg_action
from artistools.misc import addarg_figscale
from artistools.misc import require_action
from artistools.misc import resolve_outputfile
from artistools.plottools import make_frame_figure
from artistools.plottools import save_or_show


def cumulative_trapezoid(y: npt.ArrayLike, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Cumulatively integrate y over x with the trapezoidal rule, starting from zero."""
    yarr = np.asarray(y, dtype=np.float64)
    xarr = np.asarray(x, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(np.diff(xarr) * (yarr[:-1] + yarr[1:]) / 2.0)))


def quad_adaptive(
    func: Callable[[float], float], a: float, b: float, *, rtol: float = 1.5e-8, maxdepth: int = 20
) -> float:
    """Integrate a smooth function over [a, b] with adaptive Simpson quadrature.

    rtol must stay above the rounding noise of the integrand (e.g. cancellation in
    1/2 - arctan(x)/pi at large x leaves ~1e-9 relative noise). Below that noise floor no
    subdivision can meet the tolerance, so the recursion raises rather than subdividing until
    maxdepth is exhausted, which would cost up to 2**maxdepth evaluations and silently return
    an unconverged result.
    """

    def simpson(x0: float, x2: float, f0: float, f1: float, f2: float) -> float:
        return (x2 - x0) / 6.0 * (f0 + 4.0 * f1 + f2)

    def recurse(x0: float, x2: float, f0: float, f1: float, f2: float, depth: int) -> float:
        x1 = 0.5 * (x0 + x2)
        fleft = func(0.5 * (x0 + x1))
        fright = func(0.5 * (x1 + x2))
        whole = simpson(x0, x2, f0, f1, f2)
        left = simpson(x0, x1, f0, fleft, f1)
        right = simpson(x1, x2, f1, fright, f2)
        if abs(left + right - whole) <= rtol * (abs(left) + abs(right)):
            # Richardson extrapolation of the two half-interval estimates
            return left + right + (left + right - whole) / 15.0
        if depth <= 0:
            msg = f"adaptive quadrature did not reach rtol={rtol:g} on [{x0:g}, {x2:g}] within {maxdepth} levels"
            raise RuntimeError(msg)
        return recurse(x0, x1, f0, fleft, f1, depth - 1) + recurse(x1, x2, f1, fright, f2, depth - 1)

    x1 = 0.5 * (a + b)
    return recurse(a, b, func(a), func(x1), func(b), maxdepth)


def get_cumulative_heating_fraction() -> tuple[pl.DataFrame, float]:
    """Return the cumulative fraction of energy released by each time, and the total energy [erg/g].

    The heating rate follows the 5e9 * t^-1.3 erg/g/s r-process power law.
    """
    tmin = 0.0001  # days
    tmax = 50

    times = np.logspace(np.log10(tmin), np.log10(tmax), num=300)  # days
    qdot = 5e9 * (times) ** (-1.3)  # define energy power law (5e9*t^-1.3)

    cumulative_energy = cumulative_trapezoid(y=qdot, x=times)
    E_tot = float(cumulative_energy[-1])

    rate = cumulative_energy / E_tot

    times_and_rate = {"times": times, "rate": rate}
    dftimes_and_rate = pl.DataFrame(data=times_and_rate)

    dE = np.diff(dftimes_and_rate["rate"] * E_tot)
    dt = np.diff(times * 24 * 60 * 60)

    integrated_rate = dE / dt
    scale_factor_energy_diff = max(qdot[1:] / integrated_rate)
    print(np.mean(scale_factor_energy_diff))
    E_tot *= scale_factor_energy_diff

    dE = np.diff(dftimes_and_rate["rate"] * E_tot)
    dt = np.diff(times * 24 * 60 * 60)

    return dftimes_and_rate, E_tot


def make_energydistribution_weightedbyrho(
    rho: npt.NDArray[np.floating], E_tot_per_gram: float, Mtot_grams: float
) -> pl.DataFrame:
    """Return the per-cell energy release, distributing the total energy in proportion to cell density."""
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
    """Write energydistribution.txt and energyrate.txt for the power-law heating rate."""
    print("Using power law for energy rate")
    times_and_rate, E_tot_per_gram = get_cumulative_heating_fraction()

    energydistributiondata = make_energydistribution_weightedbyrho(rho, E_tot_per_gram, Mtot_grams)

    print("Writing energydistribution.txt")
    with Path(outputpath, "energydistribution.txt").open("w", encoding="utf-8") as fmodel:
        fmodel.write(f"{len(energydistributiondata['cell_energy'])}\n")  # write number of points
        fmodel.writelines(f"{cellid}\t{cell_energy:g}\n" for cellid, cell_energy in energydistributiondata.iter_rows())

    print("Writing energyrate.txt")
    with Path(outputpath, "energyrate.txt").open("w", encoding="utf-8") as fmodel:
        fmodel.write(f"{len(times_and_rate['times'])}\n")  # write number of points
        times_and_rate.write_csv(fmodel, separator="\t", include_header=False, float_precision=10)


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
        energy_per_gram_cumulative.append(energy_per_gram_cumulative[-1] + quad_adaptive(integrand, tlow, thigh))

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
    """Integrate a trajectory Qdot to get the total energy [erg/g], and the cumulative rate when get_rate is set."""
    energy_thermo_data = energy_thermo_data.filter(pl.col("time/s") <= 1e7)

    skipfirstnrows = 0  # not sure first values look sensible -- check this
    times = energy_thermo_data["time/s"][skipfirstnrows:]
    qdot = energy_thermo_data["Qdot"][skipfirstnrows:]

    cumulative_energy = cumulative_trapezoid(y=qdot, x=times)
    E_tot = float(cumulative_energy[-1])  # erg / g

    if get_rate:
        print(f"E_tot {E_tot} erg/g")
        rate = cumulative_energy / E_tot

        dftimes_and_rate = pl.DataFrame({"times": times / day_to_s, "rate": rate})

        return dftimes_and_rate, E_tot

    return E_tot


def plot_energy_rate(modelpath: str | Path, axis: mplax.Axes) -> None:
    """Plot the analytic nuclear heating power of the whole model against time."""
    times_and_rate, _ = at.inputmodel.energyinputfiles.rprocess_const_and_powerlaw()
    lzmodel, _ = at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g"])
    model = lzmodel.collect()
    Mtot_grams = model["mass_g"].sum()
    axis.plot(
        times_and_rate["times"], np.array(times_and_rate["nuclear_heating_power"]) * Mtot_grams, color="k", zorder=10
    )


def get_etot_fromfile(modelpath: str | Path) -> tuple[float, pl.DataFrame]:
    """Return the total energy [erg] and the per-cell energies read from energydistribution.txt."""
    energydistribution_data = at.read_wsv(
        Path(modelpath) / "energydistribution.txt", has_header=False, skip_rows=1, new_columns=["cellid", "cell_energy"]
    )
    etot = float(energydistribution_data["cell_energy"].sum())
    return etot, energydistribution_data


def get_energy_rate_fromfile(modelpath: str | Path) -> pl.DataFrame:
    """Return the cumulative energy release fraction against time read from energyrate.txt."""
    return at.read_wsv(Path(modelpath) / "energyrate.txt", has_header=False, skip_rows=1, new_columns=["times", "rate"])


def read_trajectory_thermo(trajthermofile: Path | str) -> pl.DataFrame:
    """Read the time/s and Qdot columns of a trajectory energy_thermo.dat.

    Times below one second are dropped, matching get_trajectory_qdotintegral: Qdot is negative there
    and integrating over it gives a negative total energy.
    """
    dfthermo = at.read_wsv(trajthermofile).select("time/s", "Qdot")

    return dfthermo.filter(pl.col("time/s") >= 1.0)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_action(
        parser,
        choices=["plotrate", "describe", "fromtrajectory"],
        helptext=(
            "plotrate: plot the analytic nuclear heating power against time."
            " describe: report the total energy and rate from the written energy files."
            " fromtrajectory: integrate a trajectory energy_thermo.dat to get the total energy and rate"
        ),
    )
    at.addarg_modelpath(parser, default=Path())
    at.addarg_output(parser, kind="file", helptext="Path for the plot, or omit to show it interactively")

    addarg_figscale(parser)
    parser.add_argument("-trajthermofile", type=Path, help="Trajectory energy_thermo.dat (fromtrajectory)")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot and inspect the ARTIS energy input files."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    require_action(args)

    modelpath = Path(args.modelpath)

    if args.action == "plotrate":
        fig, axesgrid = make_frame_figure(args)
        axis = axesgrid[0][0]
        plot_energy_rate(modelpath, axis)
        axis.set_xlabel("time [days]")
        axis.set_ylabel("Nuclear heating power [erg/s]")
        axis.set_xscale("log")
        axis.set_yscale("log")
        # -o promises that a path with no file extension names a folder, and an empty -o shows the plot
        save_or_show(fig, resolve_outputfile(args.outputfile, "energyfiles_plotrate.pdf") if args.outputfile else None)

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
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
