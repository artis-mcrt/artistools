from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl


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
