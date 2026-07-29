from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at
from artistools.constants import day_to_s


def rprocess_const_and_powerlaw() -> tuple[pl.DataFrame, float]:
    """Following eqn 4 Korobkin 2012."""

    def integrand(
        t_days: float | int,
        t0: float | int,
        epsilon0: float | int,
        sigma: float | int,
        alpha: float | int,
        thermalisation_factor: float,
    ) -> float:
        return float(epsilon0 * ((1 / 2) - (1 / np.pi * np.arctan((t_days - t0) / sigma))) ** alpha) * (
            thermalisation_factor / 0.5
        )

    from scipy.integrate import quad

    tmin = 0.01 * day_to_s
    tmax = 50 * day_to_s
    t0 = 1.3  # seconds
    epsilon0 = 2e18
    sigma = 0.11
    alpha = 1.3
    thermalisation_factor = 0.5

    E_tot = quad(integrand, tmin, tmax, args=(t0, epsilon0, sigma, alpha, thermalisation_factor))  # ergs/s/g
    print("Etot per gram", E_tot[0])
    E_tot = E_tot[0]

    times = np.logspace(np.log10(tmin), np.log10(tmax), num=200)
    energy_per_gram_cumulative = [0.0]
    for time in times[1:]:
        cumulative_integral = quad(
            integrand, tmin, time, args=(t0, epsilon0, sigma, alpha, thermalisation_factor)
        )  # ergs/s/g
        energy_per_gram_cumulative.append(cumulative_integral[0])

    rate = np.array(energy_per_gram_cumulative) / E_tot

    nuclear_heating_power = [integrand(time, t0, epsilon0, sigma, alpha, thermalisation_factor) for time in times]

    times_and_rate = {"times": times / day_to_s, "rate": rate, "nuclear_heating_power": nuclear_heating_power}
    dftimes_and_rate = pl.DataFrame(data=times_and_rate)

    return dftimes_and_rate, E_tot


def get_cumulative_heating_fraction() -> tuple[pl.DataFrame, float]:

    tmin = 0.0001  # days
    tmax = 50

    times = np.logspace(np.log10(tmin), np.log10(tmax), num=300)  # days
    qdot = 5e9 * (times) ** (-1.3)  # define energy power law (5e9*t^-1.3)

    E_tot = np.trapezoid(y=qdot, x=times)
    assert isinstance(E_tot, float)
    # print("Etot per gram", E_tot, E_tot*1.989e33*0.01)

    from scipy import integrate

    cumulative_integrated_energy = integrate.cumulative_trapezoid(y=qdot, x=times)
    cumulative_integrated_energy = np.insert(cumulative_integrated_energy, 0, 0)

    rate = cumulative_integrated_energy / E_tot

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


def energy_from_rprocess_calculation(
    energy_thermo_data: pl.DataFrame, get_rate: bool = True
) -> float | tuple[pl.DataFrame, float]:

    energy_thermo_data = energy_thermo_data.filter(pl.col("time/s") <= 1e7)
    # print("Dropping times later than 116 days")

    skipfirstnrows = 0  # not sure first values look sensible -- check this
    times = energy_thermo_data["time/s"][skipfirstnrows:]
    qdot = energy_thermo_data["Qdot"][skipfirstnrows:]

    E_tot = float(np.trapezoid(y=qdot, x=times))  # erg / g

    if get_rate:
        print(f"E_tot {E_tot} erg/g")
        from scipy import integrate

        cumulative_integrated_energy = integrate.cumulative_trapezoid(y=qdot, x=times)
        cumulative_integrated_energy = np.insert(cumulative_integrated_energy, 0, 0)

        rate = cumulative_integrated_energy / E_tot

        dftimes_and_rate = pl.DataFrame({"times": times / day_to_s, "rate": rate})

        return dftimes_and_rate, E_tot

    return E_tot


def make_energydistribution_weightedbyrho(
    rho: npt.NDArray[np.floating], E_tot_per_gram: float | int, Mtot_grams: float
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


def make_energy_files(rho: npt.NDArray[np.floating], Mtot_grams: float | int, outputpath: Path | str) -> None:
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


def plot_energy_rate(modelpath: str | Path, axis: mplax.Axes) -> None:
    times_and_rate, _ = at.inputmodel.energyinputfiles.rprocess_const_and_powerlaw()
    lzmodel, _ = at.inputmodel.get_modeldata(modelpath)
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
