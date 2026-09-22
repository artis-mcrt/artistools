"""Tools to get artis output in the required format for the code comparison workshop."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import polars as pl

from artistools.atomic import get_composition_data
from artistools.atomic import get_elsymbol
from artistools.atomic import get_ionstring
from artistools.constants import c_ang_per_s
from artistools.constants import km_to_cm
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import megaparsec_to_cm
from artistools.estimators import scan_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.lightcurve import find_lightcurve_file
from artistools.lightcurve import scan_lightcurve
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import exit_with_error
from artistools.misc import firstexisting
from artistools.misc import get_deposition
from artistools.misc import get_timestep_times
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import zopen


def write_spectra(modelpath: str | Path, selected_timesteps: Sequence[int], outfilepath: Path) -> None:
    """Write the spectra at the selected timesteps in code comparison workshop format."""
    with zopen(firstexisting("spec.out", folder=modelpath, tryzipped=True)) as specfile:
        spec_data = np.loadtxt(specfile)

    times = spec_data[0, 1:]
    freqs = spec_data[1:, 0]
    lambdas = c_ang_per_s / freqs

    fluxes_nu = spec_data[1:, 1:]

    # area in cm^2 of a sphere of radius 1 Mpc
    area = 4.0 * math.pi * megaparsec_to_cm**2

    # convert flux to power by multiplying by area
    lambdacolumn = lambdas[:, np.newaxis]
    lum_lambda = fluxes_nu * c_ang_per_s / lambdacolumn / lambdacolumn * area

    with outfilepath.open("w", encoding="utf-8") as outfile:
        outfile.write(f"#NTIMES: {len(selected_timesteps)}\n")
        outfile.write(f"#NWAVE: {len(lambdas)}\n")
        outfile.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")
        outfile.write("#wavelength[Ang] flux_t0[erg/s/Ang] flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]\n")

        for n in reversed(range(len(lambdas))):
            outfile.write(
                f"{lambdas[n]:.2f} " + " ".join([f"{lum_lambda[n, ts]:.4e}" for ts in selected_timesteps]) + "\n"
            )


def write_ntimes_nvel(
    outfile: TextIOWrapper, selected_timesteps: Sequence[int], modelpath: str | Path, ncells: int
) -> None:
    """Write the header lines that give the number of times, the number of cells, and the times.

    ncells is the number of rows that the file holds. An empty cell has no estimators, thus the count
    of the model cells promised more rows than the file gives.
    """
    times = get_timestep_times(modelpath)
    outfile.write(f"#NTIMES: {len(selected_timesteps)}\n")
    outfile.write(f"#NVEL: {ncells}\n")
    outfile.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")


def get_nonempty_cells(
    modelpath: str | Path, allnonemptymgilist: Sequence[int]
) -> tuple[pl.DataFrame, dict[str, t.Any]]:
    """Return the model data of the cells that hold estimator data, with the mid-point velocity of each one."""
    # write_phys reads rho, which a 1D model.txt does not contain. The derivation calculates it from logrho
    lzmodeldata, modelmeta = get_modeldata(modelpath)
    return (
        add_derived_cols_to_modeldata(lzmodeldata, modelmeta=modelmeta)
        .filter(pl.col("modelgridindex").is_in(allnonemptymgilist))
        .select("modelgridindex", "vel_r_mid", "rho")
        .collect()
    ), modelmeta


def scan_cell_estimators(
    modelpath: str | Path, selected_timesteps: Sequence[int], modeldata: pl.DataFrame, modelmeta: dict[str, t.Any]
) -> pl.DataFrame:
    """Return the estimators of the selected timesteps, joined to the model data of each non-empty cell.

    The frame holds one row for each pair of cell and timestep. The rows of one timestep keep the order
    of the model. The density comes from the model snapshot, which the homologous flow expands with the
    cube of the time.
    """
    times = get_timestep_times(modelpath)
    dftimes = pl.LazyFrame(
        {"timestep": list(selected_timesteps), "time_days": [times[ts] for ts in selected_timesteps]},
        schema={"timestep": pl.Int32, "time_days": pl.Float64},
    )

    return (
        modeldata
        .lazy()
        .join(
            scan_estimators(modelpath=modelpath, timestep=tuple(selected_timesteps)),
            on="modelgridindex",
            how="inner",
            maintain_order="left",
        )
        .join(dftimes, on="timestep", how="inner", maintain_order="left")
        .with_columns(rho=pl.col("rho") * (modelmeta["t_model_init_days"] / pl.col("time_days")) ** 3)
        .collect()
    )


def write_edep(
    modelpath: str | Path, selected_timesteps: Sequence[int], dfestimators: pl.DataFrame, outfile: Path
) -> None:
    """Write the deposition of every cell at the selected timesteps, in the format of the workshop."""
    # the file gives one row for each cell and one column for each timestep, thus the values of one
    # cell come from several rows. The select puts the columns in the order that the header gives.
    # A cell that a timestep does not hold gives a null, and the format needs a number in every field
    dfdeposition = (
        dfestimators
        .with_columns(total_dep=get_column_or_zero(dfestimators, "total_dep"))
        .pivot(on="timestep", index=("modelgridindex", "vel_r_mid"), values="total_dep")
        .select("vel_r_mid", *(pl.col(str(timestep)).fill_null(0.0) for timestep in selected_timesteps))
    )

    with Path(outfile).open("w", encoding="utf-8") as f:
        write_ntimes_nvel(f, selected_timesteps, modelpath, dfdeposition.height)
        f.write("#vel_mid[km/s] Edep_t0[erg/s/cm^3] Edep_t1[erg/s/cm^3] ... Edep_tn[erg/s/cm^3]\n")
        for vel_r_mid, *celldepositions in dfdeposition.iter_rows():
            f.write(f"{vel_r_mid / km_to_cm:.2f}")
            f.write("".join(f" {celldeposition:.4e}" for celldeposition in celldepositions))
            f.write("\n")


def get_column_or_zero(dfestimators: pl.DataFrame, colname: str) -> pl.Expr:
    """Return the column of the estimators as Float64, or zero when this run holds no such column.

    ARTIS writes a population only for an ion that a cell holds, and a whole run can hold none. The
    estimator cache stores a population as Float32, and a ratio of two Float32 values loses precision
    below 1e-38.
    """
    return pl.col(colname).cast(pl.Float64).fill_null(0.0) if colname in dfestimators.columns else pl.lit(0.0)


def write_ionfracts(
    modelpath: Path | str,
    model_id: str,
    selected_timesteps: Sequence[int],
    dfestimators: pl.DataFrame,
    outputpath: Path,
) -> None:
    """Write the ion fractions of every element in code comparison workshop format, one file per element."""
    times = get_timestep_times(modelpath)
    elementlist = get_composition_data(modelpath)
    nelements = len(elementlist)
    for elementindex in range(nelements):
        atomic_number = elementlist["Z"].item(elementindex)
        elsymb = get_elsymbol(atomic_number)
        nions = elementlist["nions"].item(elementindex)
        lowermost_ion_stage = elementlist["lowermost_ion_stage"].item(elementindex)
        # the format labels the neutral stage 0 and needs a column for each stage up to the highest one. ARTIS
        # holds no population below lowermost_ion_stage, thus those columns hold zero
        nstages = lowermost_ion_stage + nions - 1
        nstagesbelowlowermost = lowermost_ion_stage - 1
        ionstrs = [
            get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
            for ion_stage in range(lowermost_ion_stage, nstages + 1)
        ]
        expr_elabund = get_column_or_zero(dfestimators, f"nnelement_{elsymb}")
        dfionfracs = dfestimators.select(
            "timestep",
            "vel_r_mid",
            *(
                pl
                .when(expr_elabund > 0.0)
                .then(get_column_or_zero(dfestimators, f"nnion_{ionstr}") / expr_elabund)
                .otherwise(0.0)
                .alias(ionstr)
                for ionstr in ionstrs
            ),
        )
        pathfileout = Path(outputpath, f"ionfrac_{elsymb.lower()}_{model_id}_artisnebular.txt")
        with pathfileout.open("w", encoding="utf-8") as f:
            f.write(f"#NTIMES: {len(selected_timesteps)}\n")
            f.write(f"#NSTAGES: {nstages}\n")
            f.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")
            f.write("#\n")
            for timestep in selected_timesteps:
                dftimestep = dfionfracs.filter(pl.col("timestep") == timestep).drop("timestep")
                f.write(f"#TIME: {times[timestep]:.2f}\n")
                # a cell that this timestep does not hold gives no row, thus each block counts its own rows
                f.write(f"#NVEL: {dftimestep.height}\n")
                f.write(f"#vel_mid[km/s] {' '.join([f'{elsymb.lower()}{stage}' for stage in range(nstages)])}\n")
                for vel_r_mid, *ionfracs in dftimestep.iter_rows():
                    f.write(f"{vel_r_mid / km_to_cm:.2f}")
                    f.write(f" {0.0:.4e}" * nstagesbelowlowermost)
                    f.write("".join(f" {ionfrac:.4e}" for ionfrac in ionfracs))
                    f.write("\n")

        # a file of zeros says nothing about the model, thus the writer takes it away again
        maxionfrac = float(dfionfracs.select(pl.max_horizontal(ionstrs).max().fill_null(0.0)).item())
        if maxionfrac <= 0.0:
            print(f"Deleting {pathfileout} because it is all zeros")
            pathfileout.unlink()


def write_phys(
    modelpath: str | Path,
    model_id: str,
    selected_timesteps: Sequence[int],
    dfestimators: pl.DataFrame,
    outputpath: Path,
) -> None:
    """Write the physical conditions of every cell in code comparison workshop format."""
    times = get_timestep_times(modelpath)
    dfphys = dfestimators.select("timestep", "vel_r_mid", "Te", "rho", "nne", "nntot")
    with Path(outputpath, f"phys_{model_id}_artisnebular.txt").open("w", encoding="utf-8") as f:
        f.write(f"#NTIMES: {len(selected_timesteps)}\n")
        f.write(f"#TIMES[d]: {' '.join([f'{times[ts]:.2f}' for ts in selected_timesteps])}\n")
        f.write("#\n")
        for timestep in selected_timesteps:
            dftimestep = dfphys.filter(pl.col("timestep") == timestep).drop("timestep")
            f.write(f"#TIME: {times[timestep]:.2f}\n")
            # a cell that this timestep does not hold gives no row, thus each block counts its own rows
            f.write(f"#NVEL: {dftimestep.height}\n")
            f.write("#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]\n")
            for vel_r_mid, *cellvalues in dftimestep.iter_rows():
                f.write(f"{vel_r_mid / km_to_cm:.2f}")
                f.write("".join(f" {cellvalue:.4e}" for cellvalue in cellvalues))
                f.write("\n")


def write_lbol_edep(modelpath: str | Path, selected_timesteps: Sequence[int], outputpath: Path) -> None:
    """Write the bolometric luminosity and energy deposition rate in code comparison workshop format."""
    # light_curve.out has one row per timestep in order, and deposition.out names its timesteps, so join on the
    # light curve's row index. The columns are time_days and luminosity_Lsun, not the time and lum this used to read
    dflightcurve = (
        scan_lightcurve(find_lightcurve_file(modelpath))[-1]
        .with_row_index("timestep")
        .with_columns(pl.col("timestep").cast(pl.Int32))
        .join(get_deposition(modelpath), on="timestep", how="inner")
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
                f"{time_days:.2f} {luminosity_Lsun * Lsun_to_erg_per_s:.4e} {total_dep_Lsun * Lsun_to_erg_per_s:.4e}\n"
            )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, multiplepaths=True, default=[], helptext="Paths to ARTIS folders")

    parser.add_argument("-selected_timesteps", default=[], nargs="*", type=int, help="Selected ARTIS timesteps")

    addarg_output(parser, kind="folder", default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write ARTIS model data out in code comparison workshop format."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    args.modelpath = normalize_path_list(args.modelpath)

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

        allnonemptymgilist = (
            scan_estimators(modelpath=modelpath, timestep=(selected_timesteps[0],))
            .select(pl.col("modelgridindex").unique())
            .collect()
            .get_column("modelgridindex")
            .to_list()
        )
        modeldata, modelmeta = get_nonempty_cells(modelpath, allnonemptymgilist)
        dfestimators = scan_cell_estimators(modelpath, selected_timesteps, modeldata, modelmeta)

        # a timestep that the run did not write gives an empty block, thus the file promises rows that it has not
        if missingtimesteps := sorted(set(selected_timesteps) - set(dfestimators["timestep"].to_list())):
            exit_with_error(
                f"{modelpath} holds no estimators for the timesteps"
                f" {', '.join(str(timestep) for timestep in missingtimesteps)}",
                "Give -selected_timesteps that the run wrote",
            )

        try:
            write_lbol_edep(
                modelpath, selected_timesteps, Path(args.outputfile, f"lbol_edep_{model_id}_artisnebular.txt")
            )
        except FileNotFoundError:
            print("Can't write deposition because files are missing")

        write_spectra(modelpath, selected_timesteps, Path(args.outputfile, f"spectra_{model_id}_artisnebular.txt"))

        write_edep(
            modelpath, selected_timesteps, dfestimators, Path(args.outputfile, f"edep_{model_id}_artisnebular.txt")
        )

        write_phys(modelpath, model_id, selected_timesteps, dfestimators, args.outputfile)
        write_ionfracts(modelpath, model_id, selected_timesteps, dfestimators, args.outputfile)
