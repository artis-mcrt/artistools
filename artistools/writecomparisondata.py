#!/usr/bin/env python3
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


def write_spectra(modelpath: str | Path, selected_timesteps: Sequence[int], outfilepath: Path) -> None:
    spec_data = np.loadtxt(Path(modelpath, "spec.out"))

    times = spec_data[0, 1:]
    freqs = spec_data[1:, 0]
    lambdas = 2.99792458e18 / freqs

    # print("\n".join(["{0}, {1}".format(*x) for x in enumerate(times)]))

    fluxes_nu = spec_data[1:, 1:]

    # 1 parsec in cm is 3.086e18
    # area in cm^2 of a spherical of radius 1 Mpc is:
    area = 3.086e18 * 3.086e18 * 1e12 * 4.0 * math.pi

    lum_lambda = np.zeros((len(lambdas), len(times)))

    # convert flux to power by multiplying by area
    for n in range(1000):
        # 2.99792458e18 is c in Angstrom / second
        lum_lambda[n, :] = fluxes_nu[n, :] * 2.99792458e18 / lambdas[n] / lambdas[n] * area

    with outfilepath.open("w", encoding="utf-8") as outfile:
        outfile.write(f"#NTIMES: {len(selected_timesteps)}\n")
        outfile.write(f"#NWAVE: {len(lambdas)}\n")
        outfile.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")
        outfile.write("#wavelength[Ang] flux_t0[erg/s/Ang] flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]\n")

        for n in reversed(range(len(lambdas))):
            outfile.write(
                f"{lambdas[n]:.2f} " + " ".join([f"{lum_lambda[n, ts]:.2e}" for ts in selected_timesteps]) + "\n"
            )


def write_ntimes_nvel(outfile: TextIOWrapper, selected_timesteps: Sequence[int], modelpath: str | Path) -> None:
    times = at.get_timestep_times(modelpath)
    _, modelmeta = at.inputmodel.get_modeldata(modelpath, getheadersonly=True)
    outfile.write(f"#NTIMES: {len(selected_timesteps)}\n")
    outfile.write(f"#NVEL: {modelmeta['npts_model']}\n")
    outfile.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")


def write_single_estimator(
    modelpath: str | Path,
    selected_timesteps: Sequence[int],
    estimators: dict[tuple[int, int], dict[str, t.Any]],
    allnonemptymgilist: Sequence[int],
    outfile: Path,
    keyname: str,
) -> None:
    lzmodeldata, _modelmeta = at.inputmodel.get_modeldata(modelpath, derived_cols=["vel_r_mid"])
    lzmodeldata = lzmodeldata.filter(pl.col("modelgridindex").is_in(allnonemptymgilist))
    with Path(outfile).open("w", encoding="utf-8") as f:
        write_ntimes_nvel(f, selected_timesteps, modelpath)
        if keyname == "total_dep":
            f.write("#vel_mid[km/s] Edep_t0[erg/s/cm^3] Edep_t1[erg/s/cm^3] ... Edep_tn[erg/s/cm^3]\n")
        elif keyname == "nne":
            f.write("#vel_mid[km/s] ne_t0[/cm^3] ne_t1[/cm^3] … ne_tn[/cm^3]\n")
        elif keyname == "Te":
            f.write("#vel_mid[km/s] Tgas_t0[K] Tgas_t1[K] ... Tgas_tn[K]\n")
        for modelgridindex, vel_r_mid in lzmodeldata.select(["modelgridindex", "vel_r_mid"]).collect().iter_rows():
            f.write(f"{vel_r_mid / 1e5:.2f}")
            for timestep in selected_timesteps:
                cellvalue = estimators[timestep, modelgridindex][keyname]
                # try:
                #     cellvalue = estimators[(timestep, modelgridindex)][keyname]
                # except KeyError:
                #     cellvalue = (estimators[(timestep - 1, modelgridindex)][keyname]
                #                  + estimators[(timestep + 1, modelgridindex)][keyname]) / 2.
                f.write(f" {cellvalue:.3e}")
            f.write("\n")


def write_ionfracts(
    modelpath: Path | str,
    model_id: str,
    selected_timesteps: Sequence[int],
    estimators: dict[tuple[int, int], dict[str, t.Any]],
    allnonemptymgilist: Sequence[int],
    outputpath: Path,
) -> None:
    times = at.get_timestep_times(modelpath)
    lzmodeldata, _modelmeta = at.inputmodel.get_modeldata(modelpath, derived_cols=["vel_r_mid"])
    lzmodeldata = lzmodeldata.filter(pl.col("modelgridindex").is_in(allnonemptymgilist))
    elementlist = at.get_composition_data(modelpath)
    nelements = len(elementlist)
    for element in range(nelements):
        atomic_number = elementlist.Z[element]
        elsymb = at.get_elsymbol(atomic_number)
        nions = elementlist.nions[element]
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
                for modelgridindex, vel_r_mid in (
                    lzmodeldata.select(["modelgridindex", "vel_r_mid"]).collect().iter_rows()
                ):
                    f.write(f"{vel_r_mid / 1e5:.2f}")
                    elabund = estimators[timestep, modelgridindex].get(f"nnelement_{elsymb}", 0)
                    for ion in range(nions):
                        ion_stage = ion + elementlist.lowermost_ion_stage[element]
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
    times = at.get_timestep_times(modelpath)
    lzmodeldata, modelmeta = at.inputmodel.get_modeldata(modelpath, derived_cols=["vel_r_mid"])
    modeldata = lzmodeldata.filter(pl.col("modelgridindex").is_in(allnonemptymgilist)).collect()
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

                estimators[timestep, modelgridindex]["nntot"] = estimators[timestep, modelgridindex]["nntot"]

                f.write(f"{cell['vel_r_mid'] / 1e5:.2f}")
                for keyname in ("Te", "rho", "nne", "nntot"):
                    estvalue = estimators[timestep, modelgridindex][keyname]
                    f.write(f" {estvalue:.4e}")
                f.write("\n")


def write_lbol_edep(modelpath: str | Path, selected_timesteps: Sequence[int], outputpath: Path) -> None:
    # times = at.get_timestep_times(modelpath)
    dflightcurve = (
        at.lightcurve.readfile(Path(modelpath, "light_curve.out"))[-1]
        .to_pandas(use_pyarrow_extension_array=True)
        .merge(
            at.get_deposition(modelpath).to_pandas(use_pyarrow_extension_array=True),
            left_index=True,
            right_index=True,
            suffixes=("", "_dep"),
        )
    )

    with outputpath.open("w", encoding="utf-8") as f:
        f.write(f"#NTIMES: {len(selected_timesteps)}\n")
        f.write("#time[d] Lbol[erg/s] Edep[erg/s] \n")

        for timestep, row in dflightcurve.iterrows():
            if timestep not in selected_timesteps:
                continue
            f.write(f"{row.time:.2f} {row.lum * 3.826e33:.2e} {row.total_dep_Lsun * 3.826e33:.2e}\n")


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", default=[], nargs="*", action=at.AppendPath, help="Paths to ARTIS folders")

    parser.add_argument("-selected_timesteps", default=[], nargs="*", type=int, help="Selected ARTIS timesteps")

    parser.add_argument("-outputpath", "-o", action="store", type=Path, default=Path(), help="path for output files")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write ARTIS model data out in code comparison workshop format."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    if not args.modelpath:
        args.modelpath = [Path()]
    elif isinstance(args.modelpath, str | Path):
        args.modelpath = [args.modelpath]

    modelpathlist = args.modelpath
    selected_timesteps = args.selected_timesteps

    args.outputpath.mkdir(parents=True, exist_ok=True)

    for modelpath in modelpathlist:
        model_id = str(Path(modelpath).name).split("_")[0]
        print(f"{model_id=}")

        estimators = at.estimators.read_estimators(modelpath=modelpath)
        allnonemptymgilist = list({modelgridindex for ts, modelgridindex in estimators if ts == selected_timesteps[0]})

        try:
            write_lbol_edep(
                modelpath, selected_timesteps, Path(args.outputpath, f"lbol_edep_{model_id}_artisnebular.txt")
            )
        except FileNotFoundError:
            print("Can't write deposition because files are missing")

        write_spectra(modelpath, selected_timesteps, Path(args.outputpath, f"spectra_{model_id}_artisnebular.txt"))

        # write_single_estimator(modelpath, selected_timesteps, estimators, allnonemptymgilist,
        #                        Path(args.outputpath, "eden_" + model_id + "_artisnebular.txt"), keyname='nne')

        write_single_estimator(
            modelpath,
            selected_timesteps,
            estimators,
            allnonemptymgilist,
            Path(args.outputpath, f"edep_{model_id}_artisnebular.txt"),
            keyname="total_dep",
        )

        # write_single_estimator(modelpath, selected_timesteps, estimators, allnonemptymgilist,
        #                        Path(args.outputpath, "tgas_" + model_id + "_artisnebular.txt"), keyname='Te')

        write_phys(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, args.outputpath)
        write_ionfracts(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, args.outputpath)


if __name__ == "__main__":
    main()
