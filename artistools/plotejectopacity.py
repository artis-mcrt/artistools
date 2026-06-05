# PYTHON_ARGCOMPLETE_OK
"""Script for plotting the Planck mean opacity structure in postprocessing."""

import argparse
import math
import time
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at
import artistools.constants as const

HCLIGHTOVERFOURPI = const.h_erg_s * const.C_cm_per_s / 4 / math.pi
c = const.C_cm_per_s
c_ang_per_s = const.c_ang_per_s
K_B = const.K_B_ev_per_K
H = const.h_erg_s


def get_ion_binned_opacities(
    dfcells: pl.LazyFrame,
    dflevels: pl.LazyFrame,
    dftransitions: pl.LazyFrame,
    ionstr: str,
    lambda_bin_edges: list[float],
    time_days: float,
) -> pl.LazyFrame:
    dfcelllevelpops = (
        dfcells
        .select("modelgridindex", "Te", f"nnion_{ionstr}")
        .join(dflevels, how="cross")
        .select(
            "levelindex",
            "modelgridindex",
            nnlevel=pl.col("g")
            * (-pl.col("energy_ev") / K_B / pl.col("Te")).exp()
            / ((pl.col("g") * (-pl.col("energy_ev") / K_B / pl.col("Te")).exp()).sum())
            * pl.col(f"nnion_{ionstr}"),
        )
    )

    expopac_deltalambda = lambda_bin_edges[1] - lambda_bin_edges[0]
    time_s = time_days * 86400.0

    return (
        dftransitions
        .join(dfcells, how="cross")
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
                    / (c * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"exopac_contribution_{ionstr}"),
            (
                (
                    pl.min_horizontal(pl.col("tau_sobolev"), 1.0)
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (c * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"linebinned_contribution_{ionstr}"),
            (
                (
                    pl.col("tau_sobolev")
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (c * time_s * pl.col("rho"))
                ).sum()
            ).alias(f"linebinned_maxone_contribution_{ionstr}"),
        )
    )


def get_expansion_opacities(
    ions_levels_transitions: dict[str, tuple[pl.LazyFrame, pl.LazyFrame]],
    time_days: float,
    dfmodelcells: pl.LazyFrame,
    dflambdabins: pl.LazyFrame,
    lambda_bin_edges: list[float],
) -> pl.LazyFrame:

    print("Summing opacities...")

    dfbinnedopacities = dflambdabins.join(dfmodelcells.select("modelgridindex", "Te").lazy(), how="cross")

    for ionstr, (dflevels, dftransitions) in ions_levels_transitions.items():
        dfbinnedopacities = dfbinnedopacities.join(
            get_ion_binned_opacities(dfmodelcells, dflevels, dftransitions, ionstr, lambda_bin_edges, time_days),
            on=("modelgridindex", "lambda_angstroms_binindex"),
            how="left",
        )

    return dfbinnedopacities.select(
        "modelgridindex",
        "Te",
        cs.starts_with("lambda_angstroms_"),
        *[
            pl.sum_horizontal(cs.starts_with(prefix)).alias(prefix.removesuffix("_contribution_"))
            for prefix in ("exopac_contribution_", "linebinned_contribution_", "linebinned_maxone_contribution_")
        ],
    ).sort("modelgridindex", "lambda_angstroms_binindex")


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-timestep", "-ts", type=int, required=True, help="Timestep number to select")
    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path of ARTIS model")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    timestep = args.timestep
    time_days = at.misc.get_timestep_time(args.modelpath, timestep)

    dfcells = (
        at.estimators
        .scan_estimators(args.modelpath, timestep=timestep, join_modeldata=True)
        .select("modelgridindex", "Te", "rho", "mass_g", cs.starts_with("nnion_"))
        .collect()
    ).with_columns(batchindex=(pl.row_index() / 32).cast(pl.Int64))

    print(f"timestep {timestep} T_days = {time_days:.2f}")

    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(1200)
    pl.Config.set_engine_affinity("streaming")

    cellcount = dfcells.select(pl.len()).item()
    cells_processed = 0
    time_start = time.perf_counter()

    expopac_lambdamin = 534.5
    expopac_lambdamax = 35000.0
    expopac_deltalambda = 35.5
    expopac_nbins = int((expopac_lambdamax - expopac_lambdamin) / expopac_deltalambda)
    dflambdabins = (
        pl
        .LazyFrame({"lambda_angstroms_binlower": expopac_lambdamin + np.arange(expopac_nbins) * expopac_deltalambda})
        .with_row_index("lambda_angstroms_binindex")
        .with_columns(lambda_angstroms_bin_mid=pl.col("lambda_angstroms_binlower") + (expopac_deltalambda / 2))
    )
    lambda_bin_edges = (
        dflambdabins
        .select(
            pl.concat([
                pl.col("lambda_angstroms_binlower"),
                pl.col("lambda_angstroms_binlower").last() + expopac_deltalambda,
            ])
        )
        .collect()
        .to_series()
        .to_list()
    )

    dflevels_dfbinnedtransitions_by_ionstr = {
        at.get_ionstring(Z, ion_stage, sep="_"): (
            dflevels.lazy(),
            dftransitions
            .filter(pl.col("lambda_angstroms").is_between(lambda_bin_edges[0], lambda_bin_edges[-1]))
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
            .with_columns(nu_trans=c / (pl.col("lambda_angstroms") * 1e-8))
            .with_columns(B_ul=c**2 / 2 / H / pl.col("nu_trans").pow(3) * pl.col("A"))
            .with_columns(B_lu=pl.col("upper_g") / pl.col("lower_g") * pl.col("B_ul"))
            .select("lambda_angstroms_binindex", "lower", "upper", "lambda_angstroms", "B_ul", "B_lu")
            .collect()
            .lazy(),
        )
        for Z, ion_stage, dflevels, dftransitions in at.atomic
        .get_levels(args.modelpath, get_transitions=True, derived_transitions_columns=["lambda_angstroms"])
        .select("Z", "ion_stage", "levels", "transitions")
        .iter_rows()
    }

    dfplanckmeanopacbatches = []
    for dfcellbatch in dfcells.partition_by("batchindex", maintain_order=True, include_key=False):
        dfbinnedopacities = get_expansion_opacities(
            dflevels_dfbinnedtransitions_by_ionstr, time_days, dfcellbatch.lazy(), dflambdabins, lambda_bin_edges
        )

        dfplanckmeanbatch = (
            (
                dfbinnedopacities
                .with_columns(nu_bin_mid=1e8 * c / pl.col("lambda_angstroms_bin_mid"))
                .join(dfcellbatch.select("modelgridindex", "Te").lazy(), how="cross")
                .with_columns(
                    planckfactor=pl.col("nu_bin_mid").pow(3)
                    / ((H * pl.col("nu_bin_mid") / pl.col("Te") / K_B).exp() - 1)
                )
                .group_by("modelgridindex")
                .agg(
                    planckmean_opacity=(
                        (pl.col("planckfactor") * pl.col("exopac")).sum() / pl.col("planckfactor").sum()
                    )
                )
            )
            .sort("modelgridindex")
            .collect(engine="streaming")
        )
        dfplanckmeanopacbatches.append(dfplanckmeanbatch)

        cells_processed += dfcellbatch.select(pl.len()).item()
        print(dfplanckmeanbatch)
        elapsed = time.perf_counter() - time_start
        timepercell = elapsed / cells_processed
        print(
            f"  average seconds per cell: {timepercell:.3f}. cells remaining: {cellcount - cells_processed}. time remaining: {timepercell * (cellcount - cells_processed):.1f}s"
        )

    globalplanckmean = (
        pl
        .concat(dfplanckmeanopacbatches)
        .sort("modelgridindex")
        .join(dfcells.select("modelgridindex", "mass_g"), on="modelgridindex")
        .select(globalplanckmean=pl.col("planckmean_opacity").dot(pl.col("mass_g")) / pl.col("mass_g").sum())
    ).item()
    print(f"Global Planck mean opacity: {globalplanckmean:.3f} cm^2/g")


if __name__ == "__main__":
    main()
