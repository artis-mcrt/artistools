# PYTHON_ARGCOMPLETE_OK
"""Script for plotting the Planck mean opacity structure in postprocessing."""

import argparse
import math
import time
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
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
    time_s = time_days * 86400.0
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
            (
                pl.col("lambda_angstroms").cut(
                    breaks=lambda_bin_edges, labels=[str(x) for x in range(-1, len(lambda_bin_edges))]
                )
            )
            .cast(pl.String)
            .cast(pl.Int32)
            .alias("lambda_angstroms_binindex")
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


def get_expansion_opacities(adata: pl.DataFrame, time_days: float, dfestimators: pl.DataFrame) -> pl.LazyFrame:
    expopac_lambdamin = 20.0
    expopac_lambdamax = 50000.0
    expopac_deltalambda = 10.0
    expopac_nbins = int((expopac_lambdamax - expopac_lambdamin) / expopac_deltalambda)

    print("Summing opacities...")

    dfbinnedopacities = (
        pl
        .LazyFrame({"lambda_angstroms_binindex": range(expopac_nbins)})
        .set_sorted("lambda_angstroms_binindex")
        .with_columns(
            lambda_angstroms_binlower=expopac_lambdamin + pl.col("lambda_angstroms_binindex") * expopac_deltalambda
        )
        .with_columns(lambda_angstroms_bin_mid=pl.col("lambda_angstroms_binlower") + (expopac_deltalambda / 2))
        .join(dfestimators.select("modelgridindex", "Te", "mass_g").lazy(), how="cross")
    )

    lambda_bin_edges = [expopac_lambdamin + i * expopac_deltalambda for i in range(expopac_nbins + 1)]

    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        ionstr = at.get_ionstring(Z, ion_stage, sep="_")

        if f"nnion_{ionstr}" not in dfestimators.collect_schema().names():
            continue

        dfbinnedopacities = dfbinnedopacities.join(
            get_binned_opacities_ion(
                dfestimators.lazy(),
                dflevels.lazy(),
                dftransitions,
                ionstr,
                lambda_bin_edges,
                expopac_deltalambda,
                time_days,
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
    # mutex with time in days:
    timegroup = parser.add_argument_group("time selection (specify either timestep or time in days)")
    timegroup.add_argument("-timestep", "-ts", type=int, help="Timestep number to select")
    timegroup.add_argument("-timedays", "-time", "-t", type=float, help="Time in days to select.")

    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path of ARTIS model")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    if args.timedays is not None:
        assert args.timestep is None, "Cannot specify both timestep and timedays. Please specify only one of them."
        timestep = at.misc.get_timestep_of_timedays(args.modelpath, args.timedays)
    else:
        timestep = args.timestep
        assert timestep is not None, "Please specify either -timestep or -timedays."
    dfestimators = (
        at.estimators
        .scan_estimators(args.modelpath, timestep=timestep, join_modeldata=True)
        .select("modelgridindex", "timestep", "Te", "rho", "mass_g", cs.starts_with("nnion_"))
        .collect()
    ).with_columns(batchindex=(pl.row_index() / 32).cast(pl.Int64))

    time_days = at.misc.get_timestep_time(args.modelpath, timestep)

    print()
    print(f"timestep {timestep} T_days = {time_days:.2f}")

    adata = at.atomic.get_levels(args.modelpath, get_transitions=True, derived_transitions_columns=["lambda_angstroms"])

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(1200)
    # pl.Config.set_engine_affinity("streaming")
    cellcount = dfestimators.select(pl.len()).item()
    cells_processed = 0
    time_start = time.perf_counter()
    planckmeanopacity_times_mass = 0.0
    mass_g_sum = 0.0
    for dfcellbatch in dfestimators.partition_by("batchindex", maintain_order=True, include_key=False):
        dfbinnedopacities = get_expansion_opacities(adata, time_days, dfcellbatch)
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
    globalplanckmeanopacity = planckmeanopacity_times_mass / dfestimators.select(pl.col("mass_g").sum()).item()
    print(f"Global Planck mean opacity: {globalplanckmeanopacity:.2f} cm^2/g")


if __name__ == "__main__":
    main()
