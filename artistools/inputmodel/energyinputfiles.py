"""Write and inspect the energydistribution.txt and energyrate.txt radioactive energy input files."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl

from artistools.constants import day_to_s
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import scan_modeldata
from artistools.misc import addarg_action
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_saved
from artistools.misc import read_wsv
from artistools.misc import require_action
from artistools.misc import resolve_outputfile
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure

ENERGY_FILE_NAMES = ("energydistribution.txt", "energyrate.txt")

# the r-process power law of the energy files: the heating rate [erg/g/s] at 1 day, and the exponent of the time
POWERLAW_HEATING_ERG_PER_G_PER_S_AT_1_DAY = 5e9
POWERLAW_HEATING_EXPONENT = -1.3


def cumulative_trapezoid(y: npt.ArrayLike, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Cumulatively integrate y over x with the trapezoidal rule, starting from zero."""
    yarr = np.asarray(y, dtype=np.float64)
    xarr = np.asarray(x, dtype=np.float64)
    return np.concatenate(([0.0], np.cumsum(np.diff(xarr) * (yarr[:-1] + yarr[1:]) / 2.0)))


def get_powerlaw_heating_rate() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the times [days] and the heating rate [erg/g/s] of the r-process power law of the energy files.

    The rate is 5e9 * t^-1.3 erg/g/s with t in days, from 1e-4 to 50 days. The code that writes the energy files and
    the plot of plotrate both call this function. Thus the plot shows the heating that the files give to ARTIS.
    """
    tmin_days, tmax_days = 0.0001, 50.0
    times_days = np.logspace(np.log10(tmin_days), np.log10(tmax_days), num=300)
    return times_days, POWERLAW_HEATING_ERG_PER_G_PER_S_AT_1_DAY * times_days**POWERLAW_HEATING_EXPONENT


def get_cumulative_heating_fraction() -> tuple[pl.DataFrame, float]:
    """Return the cumulative fraction of energy released by each time, and the total energy [erg/g].

    The heating rate is that of get_powerlaw_heating_rate.
    """
    times, qdot = get_powerlaw_heating_rate()

    # qdot is a rate for each second, thus the integration variable must also be a time in seconds
    cumulative_energy = cumulative_trapezoid(y=qdot, x=times * day_to_s)
    E_tot = float(cumulative_energy[-1])

    rate = cumulative_energy / E_tot

    return pl.DataFrame({"times": times, "rate": rate}), E_tot


def make_energydistribution_of_mass(cellmass_grams: npt.NDArray[np.floating], E_tot_per_gram: float) -> pl.DataFrame:
    """Return the energy release of each cell, which is the energy per gram times the mass of the cell.

    A share in proportion to the density gives the right energy per gram only when all cells have one
    volume. A 1D shell model then got 0.063 to 16.4 times the intended energy per gram.
    """
    print(f"Energy distribution in proportion to the cell mass (E_tot per gram {E_tot_per_gram})")
    cellenergy = E_tot_per_gram * cellmass_grams
    print("Etot", cellenergy.sum())

    energydistdata = {"cellid": np.arange(1, len(cellmass_grams) + 1), "cell_energy": cellenergy}

    print(f"Sum energy cells {sum(energydistdata['cell_energy'])} should equal Etot")
    return pl.DataFrame(data=energydistdata)


def make_energy_files(cellmass_grams: npt.NDArray[np.floating], outputpath: Path | str) -> None:
    """Write energydistribution.txt and energyrate.txt for the power-law heating rate."""
    print("Using power law for energy rate")
    times_and_rate, E_tot_per_gram = get_cumulative_heating_fraction()

    energydistributiondata = make_energydistribution_of_mass(cellmass_grams, E_tot_per_gram)

    with Path(outputpath, "energydistribution.txt").open("w", encoding="utf-8") as fmodel:
        fmodel.write(f"{len(energydistributiondata['cell_energy'])}\n")  # write number of points
        fmodel.writelines(f"{cellid}\t{cell_energy:g}\n" for cellid, cell_energy in energydistributiondata.iter_rows())

    print_saved(Path(outputpath, "energydistribution.txt"))
    with Path(outputpath, "energyrate.txt").open("w", encoding="utf-8") as fmodel:
        fmodel.write(f"{len(times_and_rate['times'])}\n")  # write number of points
        times_and_rate.write_csv(fmodel, separator="\t", include_header=False, float_precision=10)
    print_saved(Path(outputpath, "energyrate.txt"))


def energy_from_rprocess_calculation(energy_thermo_data: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Integrate a trajectory Qdot to get the cumulative rate and the total energy [erg/g]."""
    energy_thermo_data = energy_thermo_data.filter(pl.col("time/s") <= 1e7)

    times = energy_thermo_data["time/s"]
    qdot = energy_thermo_data["Qdot"]

    cumulative_energy = cumulative_trapezoid(y=qdot, x=times)
    E_tot = float(cumulative_energy[-1])  # erg / g

    print(f"E_tot {E_tot} erg/g")
    rate = cumulative_energy / E_tot

    return pl.DataFrame({"times": times / day_to_s, "rate": rate}), E_tot


def plot_energy_rate(modelpath: str | Path, axis: mplax.Axes) -> pl.DataFrame:
    """Plot the analytic power of the nuclear heating of the whole model against time, and return the plotted data.

    The power is the heating rate of get_powerlaw_heating_rate times the mass of the model.
    """
    times_days, heatingrate = get_powerlaw_heating_rate()
    lzmodel, modelmeta = scan_modeldata(modelpath)
    Mtot_grams = (
        add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta).select(pl.col("mass_g").sum()).collect().item()
    )
    print_detail(
        f"The heating power is {POWERLAW_HEATING_ERG_PER_G_PER_S_AT_1_DAY:g} * t^{POWERLAW_HEATING_EXPONENT:g} erg/g/s"
        f" (t in days) times the mass of the model, {Mtot_grams:.4e} g. This is the power law of the energy files"
        " that makeartismodel --makeenergyinputfiles writes"
    )
    nuclear_heating_power = heatingrate * Mtot_grams
    axis.plot(times_days, nuclear_heating_power, color="k", zorder=10)
    return pl.DataFrame({"time_days": times_days, "nuclear_heating_power": nuclear_heating_power})


def get_etot_fromfile(modelpath: str | Path) -> tuple[float, pl.DataFrame]:
    """Return the total energy [erg] and the per-cell energies read from energydistribution.txt."""
    energydistribution_data = read_wsv(
        Path(modelpath) / "energydistribution.txt", has_header=False, skip_rows=1, new_columns=["cellid", "cell_energy"]
    )
    etot = float(energydistribution_data["cell_energy"].sum())
    return etot, energydistribution_data


def get_energy_rate_fromfile(modelpath: str | Path) -> pl.DataFrame:
    """Return the cumulative energy release fraction against time read from energyrate.txt."""
    return read_wsv(Path(modelpath) / "energyrate.txt", has_header=False, skip_rows=1, new_columns=["times", "rate"])


def read_trajectory_thermo(trajthermofile: Path | str) -> pl.DataFrame:
    """Read the time/s and Qdot columns of a trajectory energy_thermo.dat.

    Times below one second are dropped, matching get_trajectory_qdotintegral: Qdot is negative there
    and integrating over it gives a negative total energy.
    """
    dfthermo = read_wsv(trajthermofile).select("time/s", "Qdot")

    return dfthermo.filter(pl.col("time/s") >= 1.0)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_action(
        parser,
        choices=["plotrate", "describe", "fromtrajectory"],
        helptext=(
            "plotrate: plot the nuclear heating power against time, with the power law of the energy files."
            " describe: report the total energy and rate from the written energy files."
            " fromtrajectory: integrate a trajectory energy_thermo.dat to get the total energy and rate"
        ),
    )
    addarg_modelpath(parser, positional=True, default=Path())
    addarg_output(parser, kind="file", helptext="Path for the plot of plotrate")
    addarg_force(parser)
    addarg_show(parser)
    addarg_data_output(parser)

    addarg_figscale(parser)
    addarg_darkmode(parser)
    parser.add_argument("-trajthermofile", type=Path, help="Trajectory energy_thermo.dat (fromtrajectory)")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot and inspect the ARTIS energy input files."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    require_action(args)

    modelpath = Path(args.modelpath)

    if args.action == "plotrate":
        plotfile = resolve_outputfile(args.outputfile, "energyfiles_plotrate.pdf")
        confirm_overwrite([plotfile.parent], [plotfile.name], force=args.force)
        fig, axesgrid = make_frame_figure(args)
        axis = axesgrid[0][0]
        plotteddata = plot_energy_rate(modelpath, axis)
        axis.set_xlabel("time [days]")
        axis.set_ylabel("Nuclear heating power [erg/s]")
        axis.set_xscale("log")
        axis.set_yscale("log")
        output_data(args, {"": plotteddata}, plotfile)
        save_figure(fig, plotfile, args=args)

    elif args.action == "describe":
        etot, energydistribution = get_etot_fromfile(modelpath)
        print(f"energydistribution.txt: {len(energydistribution)} cells, total energy {etot:.4e} erg")
        dfrate = get_energy_rate_fromfile(modelpath)
        times, rates = dfrate["times"].to_numpy(), dfrate["rate"].to_numpy()
        tmin, tmax = float(times.min()), float(times.max())
        ratemin, ratemax = float(rates.min()), float(rates.max())
        print(f"energyrate.txt: {len(dfrate)} times from {tmin:g} to {tmax:g} days")
        print(f"  Cumulative energy fraction runs {ratemin:.4g} to {ratemax:.4g}")

    else:
        if args.trajthermofile is None:
            exit_with_error("fromtrajectory needs --trajthermofile")
        dftimes_and_rate, e_tot = energy_from_rprocess_calculation(read_trajectory_thermo(args.trajthermofile))
        print(f"E_tot {e_tot:.4e} erg/g over {len(dftimes_and_rate)} times")
