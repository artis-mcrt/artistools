# PYTHON_ARGCOMPLETE_OK
"""Script for computing binned expansion opacities and Planck-mean opacities in postprocessing."""

import argparse
import math
import time
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import h_erg_s
from artistools.constants import K_B_erg_per_K
from artistools.constants import K_B_ev_per_K

HCLIGHTOVERFOURPI = h_erg_s * C_cm_per_s / 4 / math.pi


def get_binned_opacities_ion(
    dfcells: pl.LazyFrame,
    dflevels: pl.LazyFrame,
    dftransitions: pl.LazyFrame,
    ionstr: str,
    lambda_bin_edges: list[float],
    expopac_deltalambda: float,
    time_days: float,
) -> pl.LazyFrame:
    """Return one ion's Sobolev expansion opacity, summed into the given wavelength bins."""
    time_s = time_days * day_to_s
    dfcelllevelpops = dflevels.join(dfcells, how="cross").with_columns(
        nnlevel=pl.col("g")
        * (-pl.col("energy_ev") / K_B_ev_per_K / pl.col("Te")).exp()
        / ((pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / pl.col("Te")).exp()).sum().over("modelgridindex"))
        * pl.col(f"nnion_{ionstr}")
    )

    return (
        dftransitions
        .filter(pl.col("lambda_angstroms").is_between(lambda_bin_edges[0], lambda_bin_edges[-1]))
        .with_columns(nu_trans=1e8 * C_cm_per_s / (pl.col("lambda_angstroms")))
        .with_columns(B_ul=C_cm_per_s**2 / 2 / h_erg_s / pl.col("nu_trans").pow(3) * pl.col("A"))
        .with_columns(B_lu=pl.col("upper_g") / pl.col("lower_g") * pl.col("B_ul"))
        .with_columns(
            (pl.col("lambda_angstroms").cut(breaks=lambda_bin_edges).to_physical().cast(pl.Int32) - 1).alias(
                "lambda_angstroms_binindex"
            )
        )
        .join(dfcells.select("modelgridindex", "rho"), how="cross")
        .join(
            dfcelllevelpops.select("modelgridindex", lower=pl.col("levelindex"), nnlevel_lower=pl.col("nnlevel")),
            on=("modelgridindex", "lower"),
            how="left",
        )
        .join(
            dfcelllevelpops.select("modelgridindex", upper=pl.col("levelindex"), nnlevel_upper=pl.col("nnlevel")),
            on=("modelgridindex", "upper"),
            how="left",
        )
        .with_columns(
            tau_sobolev=(pl.col("nnlevel_lower") * pl.col("B_lu") - pl.col("nnlevel_upper") * pl.col("B_ul"))
            * HCLIGHTOVERFOURPI
            * time_s
        )
        .group_by("modelgridindex", "lambda_angstroms_binindex")
        .agg(
            (
                (
                    (1 - (-pl.col("tau_sobolev")).exp())
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (C_cm_per_s * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"exopac_contribution_{ionstr}"),
            (
                (
                    pl.min_horizontal(pl.col("tau_sobolev"), 1.0)
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (C_cm_per_s * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"linebinned_maxone_contribution_{ionstr}"),
            (
                (
                    pl.col("tau_sobolev")
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (C_cm_per_s * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"linebinned_contribution_{ionstr}"),
        )
    )


def get_expansion_opacities(
    adata: pl.DataFrame,
    time_days: float,
    dfestimators: pl.DataFrame,
    lambdamin: float,
    lambdamax: float,
    deltalambda: float,
) -> pl.LazyFrame:
    """Return the binned expansion opacity of every cell, summed over all ions in the atomic data."""
    numbins = int((lambdamax - lambdamin) / deltalambda)

    print("Summing opacities...")

    dfbinnedopacities = (
        pl
        .LazyFrame({"lambda_angstroms_binindex": range(numbins)})
        .set_sorted("lambda_angstroms_binindex")
        .with_columns(lambda_angstroms_binlower=lambdamin + pl.col("lambda_angstroms_binindex") * deltalambda)
        .with_columns(lambda_angstroms_bin_mid=pl.col("lambda_angstroms_binlower") + (deltalambda / 2))
        .join(dfestimators.select("modelgridindex", "Te", "mass_g").lazy(), how="cross")
    )

    lambda_bin_edges = [lambdamin + i * deltalambda for i in range(numbins + 1)]

    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        ionstr = at.get_ionstring(Z, ion_stage, sep="_")

        if f"nnion_{ionstr}" not in dfestimators.collect_schema().names():
            continue

        dfbinnedopacities = dfbinnedopacities.join(
            get_binned_opacities_ion(
                dfestimators.lazy(), dflevels.lazy(), dftransitions, ionstr, lambda_bin_edges, deltalambda, time_days
            ),
            on=("modelgridindex", "lambda_angstroms_binindex"),
            how="left",
        )

    return dfbinnedopacities.select(
        "modelgridindex",
        "lambda_angstroms_binindex",
        "lambda_angstroms_bin_mid",
        "Te",
        "mass_g",
        *[
            pl.sum_horizontal(cs.starts_with(prefix)).alias(prefix.removesuffix("_contribution_"))
            for prefix in ("exopac_contribution_", "linebinned_contribution_", "linebinned_maxone_contribution_")
        ],
    ).sort("modelgridindex", "lambda_angstroms_binindex")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    # mutex with time in days:
    timegroup = parser.add_argument_group("time selection (specify either timestep or time in days)")
    timegroup.add_argument("-timestep", "-ts", type=int, help="Timestep number to select")
    timegroup.add_argument("-timedays", "-time", "-t", help="Time in days to select.")

    at.addarg_modelpath(parser, default=Path(), helptext="Path of ARTIS model")
    parser.add_argument(
        "--show_binned_opacities",
        action="store_true",
        help="Show the binned opacities for each cell (can be very large).",
    )
    at.addarg_modelgridindex(parser, helptext="Model grid cell to select. If not specified, all cells are processed.")

    parser.add_argument(
        "-lambdamin", type=float, default=20.0, help="Minimum wavelength in Angstroms for binned opacities."
    )
    parser.add_argument(
        "-lambdamax", type=float, default=50000.0, help="Maximum wavelength in Angstroms for binned opacities."
    )
    parser.add_argument(
        "-deltalambda", type=float, default=10.0, help="Wavelength bin width in Angstroms for binned opacities."
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Compute binned expansion opacities and Planck-mean opacities in postprocessing."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.timedays is not None:
        if args.timestep is not None:
            at.exit_with_error("specify only one of -timestep and -timedays")
        timestep = at.misc.get_timestep_of_timedays(args.modelpath, args.timedays)
    else:
        timestep = args.timestep
        if timestep is None:
            at.exit_with_error("specify a time or a timestep, e.g. -timedays 250 or -timestep 30")

    dfestimators = (
        at.estimators
        .scan_estimators(args.modelpath, timestep=timestep, modelgridindex=args.modelgridindex, join_modeldata=True)
        .select("modelgridindex", "timestep", "Te", "rho", "mass_g", cs.starts_with("nnion_"))
        .collect()
    ).with_columns(batchindex=(pl.row_index() / 32).cast(pl.Int64))

    time_days = at.misc.get_timestep_time(args.modelpath, timestep)

    print()
    print(f"timestep {timestep} T_days = {time_days:.2f}")

    # get_binned_opacities_ion() needs the statistical weights as well as the wavelength, and
    # add_transition_columns() drops every derived column that is not requested here
    adata = at.atomic.get_levels(
        args.modelpath, get_transitions=True, derived_transitions_columns=["lambda_angstroms", "lower_g", "upper_g"]
    )

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(5000)
    # pl.Config.set_engine_affinity("streaming")
    cellcount = dfestimators.select(pl.len()).item()
    cells_processed = 0
    time_start = time.perf_counter()
    planckmeanopacity_times_mass = 0.0
    mass_g_sum = 0.0
    for dfcellbatch in dfestimators.partition_by("batchindex", maintain_order=True, include_key=False):
        dfbinnedopacities = get_expansion_opacities(
            adata=adata,
            time_days=time_days,
            dfestimators=dfcellbatch,
            lambdamin=args.lambdamin,
            lambdamax=args.lambdamax,
            deltalambda=args.deltalambda,
        )
        if args.show_binned_opacities:
            dfbinnedopacities = dfbinnedopacities.collect()
            print(dfbinnedopacities)

        dfplanckmean = (
            (
                dfbinnedopacities
                .lazy()
                .with_columns(lambda_cm_bin_mid=pl.col("lambda_angstroms_bin_mid") * 1e-8)
                .with_columns(
                    planckfactor=(
                        (pl.col("lambda_cm_bin_mid").pow(-5))
                        / (
                            (h_erg_s * C_cm_per_s / pl.col("lambda_cm_bin_mid") / pl.col("Te") / K_B_erg_per_K).exp()
                            - 1
                        )
                    )
                )
                .group_by("modelgridindex", "mass_g")
                .agg(
                    planckmean_opacity=(
                        (pl.col("planckfactor") * pl.col("exopac")).sum() / pl.col("planckfactor").sum()
                    )
                )
            )
            .sort("modelgridindex")
            .collect(engine="streaming")
        )

        print(dfplanckmean)
        planckmeanopacity_times_mass += (dfplanckmean.select(pl.col("planckmean_opacity").dot(pl.col("mass_g")))).item()
        mass_g_sum += dfplanckmean.select(pl.col("mass_g").sum()).item()

        cells_processed += dfcellbatch.select(pl.len()).item()
        elapsed = time.perf_counter() - time_start
        timepercell = elapsed / cells_processed
        print(
            f" average seconds per cell: {timepercell:.3f}. cells remaining: {cellcount - cells_processed}. time remaining: {timepercell * (cellcount - cells_processed):.1f}s"
        )

    print()
    globalplanckmeanopacity = planckmeanopacity_times_mass / mass_g_sum
    print(f"Global Planck mean opacity: {globalplanckmeanopacity:.2f} cm^2/g")


if __name__ == "__main__":
    main()
