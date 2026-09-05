"""Tools to get artis output in the required format for the code comparison workshop."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import polars as pl

import artistools as at
from artistools.constants import km_to_cm
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import print_warning


def write_spectra(modelpath: str | Path, selected_timesteps: Sequence[int], outfilepath: Path) -> None:
    """Write the spectra at the selected timesteps in code comparison workshop format."""
    with at.zopen(at.firstexisting("spec.out", folder=modelpath, tryzipped=True)) as specfile:
        spec_data = np.loadtxt(specfile)

    times = spec_data[0, 1:]
    freqs = spec_data[1:, 0]
    lambdas = at.constants.c_ang_per_s / freqs

    fluxes_nu = spec_data[1:, 1:]

    # area in cm^2 of a sphere of radius 1 Mpc
    area = 4.0 * math.pi * at.constants.megaparsec_to_cm**2

    # convert flux to power by multiplying by area
    lambdacolumn = lambdas[:, np.newaxis]
    lum_lambda = fluxes_nu * at.constants.c_ang_per_s / lambdacolumn / lambdacolumn * area

    with outfilepath.open("w", encoding="utf-8") as outfile:
        outfile.write(f"#NTIMES: {len(selected_timesteps)}\n")
        outfile.write(f"#NWAVE: {len(lambdas)}\n")
        outfile.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")
        outfile.write("#wavelength[Ang] flux_t0[erg/s/Ang] flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]\n")

        for n in reversed(range(len(lambdas))):
            outfile.write(
                f"{lambdas[n]:.2f} " + " ".join([f"{lum_lambda[n, ts]:.4e}" for ts in selected_timesteps]) + "\n"
            )


def write_ntimes_nvel(outfile: TextIOWrapper, selected_timesteps: Sequence[int], modelpath: str | Path) -> None:
    """Write the header lines giving the number of times, the number of cells, and the times themselves."""
    times = at.get_timestep_times(modelpath)
    _, modelmeta = at.inputmodel.get_modeldata(modelpath)
    outfile.write(f"#NTIMES: {len(selected_timesteps)}\n")
    outfile.write(f"#NVEL: {modelmeta['npts_model']}\n")
    outfile.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")


def get_nonempty_cells(
    modelpath: str | Path, allnonemptymgilist: Sequence[int]
) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return the model data of the cells that hold estimator data, with the mid-point velocity of each one."""
    lzmodeldata, modelmeta = at.inputmodel.get_modeldata(modelpath, derived_cols=["vel_r_mid"])
    return lzmodeldata.filter(pl.col("modelgridindex").is_in(allnonemptymgilist)).collect(), modelmeta


def write_single_estimator(
    modelpath: str | Path,
    selected_timesteps: Sequence[int],
    estimators: dict[tuple[int, int], dict[str, t.Any]],
    allnonemptymgilist: Sequence[int],
    outfile: Path,
    keyname: str,
) -> None:
    """Write one estimator's value in every cell at the selected timesteps, in code comparison workshop format."""
    modeldata, _ = get_nonempty_cells(modelpath, allnonemptymgilist)
    with Path(outfile).open("w", encoding="utf-8") as f:
        write_ntimes_nvel(f, selected_timesteps, modelpath)
        if keyname == "total_dep":
            f.write("#vel_mid[km/s] Edep_t0[erg/s/cm^3] Edep_t1[erg/s/cm^3] ... Edep_tn[erg/s/cm^3]\n")
        elif keyname == "nne":
            f.write("#vel_mid[km/s] ne_t0[/cm^3] ne_t1[/cm^3] … ne_tn[/cm^3]\n")
        elif keyname == "Te":
            f.write("#vel_mid[km/s] Tgas_t0[K] Tgas_t1[K] ... Tgas_tn[K]\n")
        for modelgridindex, vel_r_mid in modeldata.select("modelgridindex", "vel_r_mid").iter_rows():
            f.write(f"{vel_r_mid / km_to_cm:.2f}")
            for timestep in selected_timesteps:
                cellvalue = estimators[timestep, modelgridindex][keyname]
                f.write(f" {cellvalue:.4e}")
            f.write("\n")


def write_ionfracts(
    modelpath: Path | str,
    model_id: str,
    selected_timesteps: Sequence[int],
    estimators: dict[tuple[int, int], dict[str, t.Any]],
    allnonemptymgilist: Sequence[int],
    outputpath: Path,
) -> None:
    """Write the ion fractions of every element in code comparison workshop format, one file per element."""
    times = at.get_timestep_times(modelpath)
    modeldata, _ = get_nonempty_cells(modelpath, allnonemptymgilist)
    elementlist = at.get_composition_data(modelpath)
    nelements = len(elementlist)
    cellrows = modeldata.select("modelgridindex", "vel_r_mid").rows()
    for elementindex in range(nelements):
        atomic_number = elementlist["Z"].item(elementindex)
        elsymb = at.get_elsymbol(atomic_number)
        nions = elementlist["nions"].item(elementindex)
        pathfileout = Path(outputpath, f"ionfrac_{elsymb.lower()}_{model_id}_artisnebular.txt")
        fileisallzeros = True  # will be changed when a non-zero is encountered
        with pathfileout.open("w", encoding="utf-8") as f:
            f.write(f"#NTIMES: {len(selected_timesteps)}\n")
            f.write(f"#NSTAGES: {nions}\n")
            f.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")
            f.write("#\n")
            for timestep in selected_timesteps:
                f.write(f"#TIME: {times[timestep]:.2f}\n")
                f.write(f"#NVEL: {len(allnonemptymgilist)}\n")
                f.write(f"#vel_mid[km/s] {' '.join([f'{elsymb.lower()}{ion}' for ion in range(nions)])}\n")
                for modelgridindex, vel_r_mid in cellrows:
                    f.write(f"{vel_r_mid / km_to_cm:.2f}")
                    elabund = estimators[timestep, modelgridindex].get(f"nnelement_{elsymb}", 0)
                    for ion in range(nions):
                        ion_stage = ion + elementlist["lowermost_ion_stage"].item(elementindex)
                        ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
                        ionabund = estimators[timestep, modelgridindex].get(f"nnion_{ionstr}", 0)
                        ionfrac = ionabund / elabund if elabund > 0 else 0
                        if ionfrac > 0.0:
                            fileisallzeros = False
                        f.write(f" {ionfrac:.4e}")
                    f.write("\n")
        if fileisallzeros:
            print(f"Deleting {pathfileout} because it is all zeros")
            pathfileout.unlink()


def write_phys(
    modelpath: str | Path,
    model_id: str,
    selected_timesteps: Sequence[int],
    estimators: dict[tuple[int, int], dict[str, t.Any]],
    allnonemptymgilist: Sequence[int],
    outputpath: Path,
) -> None:
    """Write the physical conditions of every cell in code comparison workshop format."""
    times = at.get_timestep_times(modelpath)
    modeldata, modelmeta = get_nonempty_cells(modelpath, allnonemptymgilist)
    with Path(outputpath, f"phys_{model_id}_artisnebular.txt").open("w", encoding="utf-8") as f:
        f.write(f"#NTIMES: {len(selected_timesteps)}\n")
        f.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")
        f.write("#\n")
        for timestep in selected_timesteps:
            f.write(f"#TIME: {times[timestep]:.2f}\n")
            f.write(f"#NVEL: {len(modeldata)}\n")
            f.write("#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]\n")
            for cell in modeldata.iter_rows(named=True):
                modelgridindex = cell["modelgridindex"]

                estimators[timestep, modelgridindex]["rho"] = (
                    10 ** cell["logrho"] * (modelmeta["t_model_init_days"] / times[timestep]) ** 3
                )

                f.write(f"{cell['vel_r_mid'] / km_to_cm:.2f}")
                for keyname in ("Te", "rho", "nne", "nntot"):
                    estvalue = estimators[timestep, modelgridindex][keyname]
                    f.write(f" {estvalue:.4e}")
                f.write("\n")


def write_lbol_edep(modelpath: str | Path, selected_timesteps: Sequence[int], outputpath: Path) -> None:
    """Write the bolometric luminosity and energy deposition rate in code comparison workshop format."""
    # light_curve.out has one row per timestep in order, and deposition.out names its timesteps, so join on the
    # light curve's row index. The columns are time_days and luminosity_Lsun, not the time and lum this used to read
    dflightcurve = (
        at.lightcurve
        .readfile(at.lightcurve.find_lightcurve_file(modelpath))[-1]
        .with_row_index("timestep")
        .with_columns(pl.col("timestep").cast(pl.Int32))
        .join(at.get_deposition(modelpath), on="timestep", how="inner")
        .filter(pl.col("timestep").is_in(list(selected_timesteps)))
        .sort("timestep")
        .select("timestep", "time_days", "luminosity_Lsun", "total_dep_Lsun")
        .collect()
    )

    if missing := sorted(set(selected_timesteps) - set(dflightcurve["timestep"])):
        print_warning(f"no light curve or deposition data for timesteps {missing}. They are left out of the file")

    with outputpath.open("w", encoding="utf-8") as f:
        # the row count, not len(selected_timesteps): a selected timestep missing from either input is dropped by
        # the join above, and a header promising more times than the file contains would misalign every reader
        f.write(f"#NTIMES: {dflightcurve.height}\n")
        f.write("#time[d] Lbol[erg/s] Edep[erg/s] \n")

        for time_days, luminosity_Lsun, total_dep_Lsun in dflightcurve.drop("timestep").iter_rows():
            f.write(
                f"{time_days:.2f} {luminosity_Lsun * at.constants.Lsun_to_erg_per_s:.4e}"
                f" {total_dep_Lsun * at.constants.Lsun_to_erg_per_s:.4e}\n"
            )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, multiplepaths=True, default=[], helptext="Paths to ARTIS folders")

    parser.add_argument("-selected_timesteps", default=[], nargs="*", type=int, help="Selected ARTIS timesteps")

    addarg_output(parser, kind="folder", default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write ARTIS model data out in code comparison workshop format."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    args.modelpath = at.normalize_path_list(args.modelpath)

    modelpathlist = args.modelpath
    selected_timesteps = args.selected_timesteps
    if not selected_timesteps:
        # for an empty list, the loop below writes a valid file with no rows. Raise an error before the loop
        msg = "Give at least one timestep with -selected_timesteps"
        raise ValueError(msg)

    args.outputfile.mkdir(parents=True, exist_ok=True)

    for modelpath in modelpathlist:
        model_id = Path(modelpath).name.split("_")[0]
        print(f"{model_id=}")

        estimators = at.estimators.read_estimators(modelpath=modelpath, timestep=tuple(selected_timesteps))
        allnonemptymgilist = list({modelgridindex for ts, modelgridindex in estimators if ts == selected_timesteps[0]})

        try:
            write_lbol_edep(
                modelpath, selected_timesteps, Path(args.outputfile, f"lbol_edep_{model_id}_artisnebular.txt")
            )
        except FileNotFoundError:
            print("Can't write deposition because files are missing")

        write_spectra(modelpath, selected_timesteps, Path(args.outputfile, f"spectra_{model_id}_artisnebular.txt"))

        write_single_estimator(
            modelpath,
            selected_timesteps,
            estimators,
            allnonemptymgilist,
            Path(args.outputfile, f"edep_{model_id}_artisnebular.txt"),
            keyname="total_dep",
        )

        write_phys(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, args.outputfile)
        write_ionfracts(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, args.outputfile)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
