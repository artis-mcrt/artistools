# PYTHON_ARGCOMPLETE_OK
"""Script for plotting the Planck mean opacity structure in postprocessing."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at
import artistools.constants as const


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path of ARTIS model")
    parser.add_argument("-timestep", "-ts", type=int, help="Timestep number to select")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    K_B = const.K_B_ev_per_K
    HCLIGHTOVERFOURPI = const.h_erg_s * const.C_cm_per_s / 4 / math.pi
    c = const.C_cm_per_s
    H = const.h_erg_s

    expopac_lambdamin = 534.5
    expopac_lambdamax = 35000.0
    expopac_deltalambda = 35.5
    expopac_nbins = int((expopac_lambdamax - expopac_lambdamin) / expopac_deltalambda)
    print(f"expopac_nbins = {expopac_nbins}")
    lambda_lowers = expopac_lambdamin + np.arange(expopac_nbins) * expopac_deltalambda
    lambda_uppers = expopac_lambdamin + (np.arange(expopac_nbins) + 1) * expopac_deltalambda
    lambda_bin_edges = [*list(lambda_lowers), lambda_uppers[-1]]

    timestep = args.timestep
    dfestimators = (
        at.estimators
        .scan_estimators(args.modelpath, timestep=timestep, join_modeldata=True)
        .select("modelgridindex", "timestep", "Te", "tdays", "rho", "mass_g", cs.starts_with("nnion_"))
        .collect()
        .lazy()
    )

    time_days = dfestimators.select("tdays").first().collect().item()
    time_s = time_days * 86400.0
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(1200)

    adata = at.atomic.get_levels(args.modelpath, get_transitions=True, derived_transitions_columns=["lambda_angstroms"])
    print("Summing opacities...")
    dfbinnedopacities = (
        pl
        .LazyFrame({
            "lambda_angstroms_binindex": range(expopac_nbins),
            "lambda_angstroms_binlower": lambda_lowers,
            "lambda_angstroms_binupper": lambda_uppers,
        })
        .with_columns(
            lambda_angstroms_bin_mid=(pl.col("lambda_angstroms_binlower") + pl.col("lambda_angstroms_binupper")) / 2
        )
        .with_columns(nu_bin_mid=1e8 * c / pl.col("lambda_angstroms_bin_mid"))
        .join(dfestimators.select("modelgridindex", "Te"), how="cross")
        .with_columns(
            planckfactor=pl.col("nu_bin_mid").pow(3) / ((H * pl.col("nu_bin_mid") / pl.col("Te") / K_B).exp() - 1)
        )
    )

    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        ionstr = at.get_ionstring(Z, ion_stage, sep="_")

        dflevels = (
            dfestimators
            .select("modelgridindex", "Te", f"nnion_{ionstr}")
            .join(dflevels.lazy(), how="cross")
            .with_columns(
                nnlevel=pl.col("g")
                * (-pl.col("energy_ev") / K_B / pl.col("Te")).exp()
                / ((pl.col("g") * (-pl.col("energy_ev") / K_B / pl.col("Te")).exp()).sum())
                * pl.col(f"nnion_{ionstr}")
            )
        )

        dftransitions = (
            dfestimators
            .select("modelgridindex", "rho")
            .join(dftransitions.collect().lazy(), how="cross")
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
            .join(
                dflevels.select("modelgridindex", lower=pl.col("levelindex"), nnlevel_lower=pl.col("nnlevel")),
                on=("modelgridindex", "lower"),
                how="left",
            )
            .join(
                dflevels.select("modelgridindex", upper=pl.col("levelindex"), nnlevel_upper=pl.col("nnlevel")),
                on=("modelgridindex", "upper"),
                how="left",
            )
            .with_columns(nu_trans=1e8 * c / (pl.col("lambda_angstroms")))
            .with_columns(B_ul=c**2 / 2 / H / pl.col("nu_trans").pow(3) * pl.col("A"))
            .with_columns(B_lu=pl.col("upper_g") / pl.col("lower_g") * pl.col("B_ul"))
            .with_columns(
                tau_sobolev=(pl.col("nnlevel_lower") * pl.col("B_lu") - pl.col("nnlevel_upper") * pl.col("B_ul"))
                * HCLIGHTOVERFOURPI
                * time_s
            )
        )
        dftransitions = dftransitions.group_by("modelgridindex", "lambda_angstroms_binindex").agg(
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

        dfbinnedopacities = dfbinnedopacities.join(
            dftransitions, on=("modelgridindex", "lambda_angstroms_binindex"), how="left"
        )

    dfbinnedopacities = dfbinnedopacities.select(
        "modelgridindex",
        "Te",
        "planckfactor",
        cs.starts_with("lambda_angstroms_"),
        *[
            pl.sum_horizontal(cs.starts_with(prefix)).alias(prefix.removesuffix("_contribution_"))
            for prefix in ("exopac_contribution_", "linebinned_contribution_", "linebinned_maxone_contribution_")
        ],
    ).sort("modelgridindex", "lambda_angstroms_binindex")

    dfplanckmean = (
        dfbinnedopacities
        .lazy()
        .group_by("modelgridindex")
        .agg(planckmean_opacity=((pl.col("planckfactor") * pl.col("exopac")).sum() / pl.col("planckfactor").sum()))
    ).sort("modelgridindex")
    print()
    print(f"timestep {timestep} T_days = {time_days:.2e}")
    # print(lzdfresults.collect())
    print(dfplanckmean.collect())


if __name__ == "__main__":
    main()
