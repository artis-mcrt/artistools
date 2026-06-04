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

    parser.add_argument("-outputpath", type=Path, default=Path("expansionopacity.pdf"), help="Path to output PDF")


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

    timestep = 2
    modelgridindex = 0
    dfestimators = at.estimators.scan_estimators(
        args.modelpath, timestep=timestep, modelgridindex=modelgridindex, join_modeldata=True
    ).select("modelgridindex", "timestep", "Te", "tdays", "rho", cs.starts_with("nnion_"))

    print(dfestimators.head().collect())

    time_days = dfestimators.select("tdays").first().collect().item()
    time_s = time_days * 86400.0
    rho = dfestimators.select("rho").first().collect().item()

    pl.Config.set_tbl_cols(20)
    adata = at.atomic.get_levels(
        args.modelpath, get_transitions=True, derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms"]
    )
    print("Summing opacities...")
    lzdfresults = pl.LazyFrame({
        "lambda_angstroms_binindex": range(expopac_nbins),
        "lambda_angstroms_binlower": lambda_lowers,
        "lambda_angstroms_binupper": lambda_uppers,
    }).with_columns(lambda_bin_center=(pl.col("lambda_angstroms_binlower") + pl.col("lambda_angstroms_binupper")) / 2)
    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        temperature_exc = dfestimators.select("Te").collect().item()
        ionstr = at.get_ionstring(Z, ion_stage, sep="_")
        nnion = dfestimators.select(pl.col(f"nnion_{ionstr}")).collect().item()

        dflevels = dflevels.lazy().with_columns(
            nnlevel_on_nnion=pl.col("g")
            * (-pl.col("energy_ev") / K_B / temperature_exc).exp()
            / ((pl.col("g") * (-pl.col("energy_ev") / K_B / temperature_exc).exp()).sum())
        )

        dftransitions = (
            dftransitions
            .with_columns(nu_trans=pl.col("epsilon_trans_ev") / const.h_ev_s)
            .join(
                dflevels.select(lower=pl.col("levelindex"), nnlevel_lower_on_nnion=pl.col("nnlevel_on_nnion")),
                on="lower",
                how="left",
            )
            .join(
                dflevels.select(upper=pl.col("levelindex"), nnlevel_upper_on_nnion=pl.col("nnlevel_on_nnion")),
                on="upper",
                how="left",
            )
            .with_columns(B_ul=c**2 / 2 / H / pl.col("nu_trans").pow(3) * pl.col("A"))
            .with_columns(B_lu=pl.col("upper_g") / pl.col("lower_g") * pl.col("B_ul"))
            .with_columns(
                tau_sobolev_on_nnion=(
                    pl.col("nnlevel_lower_on_nnion") * pl.col("B_lu")
                    - pl.col("nnlevel_upper_on_nnion") * pl.col("B_ul")
                )
                * HCLIGHTOVERFOURPI
                * time_s
            )
            .with_columns(
                (
                    (1 - (-pl.col("tau_sobolev_on_nnion") * nnion).exp())
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / c
                    / time_s
                    / rho
                ).alias(f"exopac_contribution_{ionstr}"),
                (
                    pl.min_horizontal(pl.col("tau_sobolev_on_nnion") * nnion, 1.0)
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (c * time_s * rho)
                ).alias(f"linebinned_contribution_{ionstr}"),
                (
                    pl.col("tau_sobolev_on_nnion")
                    * nnion
                    * pl.col("lambda_angstroms")
                    / expopac_deltalambda
                    / (c * time_s * rho)
                ).alias(f"linebinned_maxone_contribution_{ionstr}"),
            )
        )
        dftransitions = (
            dftransitions
            .filter(pl.col("lambda_angstroms").is_between(lambda_bin_edges[0], lambda_bin_edges[-1], closed="both"))
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
            .group_by("lambda_angstroms_binindex")
            .agg([
                pl.col(col).sum()
                for col in (
                    f"exopac_contribution_{ionstr}",
                    f"linebinned_contribution_{ionstr}",
                    f"linebinned_maxone_contribution_{ionstr}",
                )
            ])
        )

        lzdfresults = lzdfresults.join(dftransitions, on="lambda_angstroms_binindex", how="left")

    lzdfresults = lzdfresults.select(
        cs.starts_with("lambda_angstroms_"),
        *[
            pl.sum_horizontal(cs.starts_with(prefix)).alias(prefix.removesuffix("_contribution_"))
            for prefix in ("exopac_contribution_", "linebinned_contribution_", "linebinned_maxone_contribution_")
        ],
    )
    dfresults = lzdfresults.collect()
    print()
    print(f"timestep {timestep} T_days = {time_days:.2e}")
    print(f"cell {modelgridindex} T_exc = {temperature_exc} K")
    print(dfresults)


if __name__ == "__main__":
    main()
