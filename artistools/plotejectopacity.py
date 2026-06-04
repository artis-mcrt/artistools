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

    selected_binindex = 0
    timestep = 2
    modelgridindex = 20
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
    lazy_dfs = []
    for Z, ion_stage, dflevels, dftransitions in adata.select("Z", "ion_stage", "levels", "transitions").iter_rows():
        temperature_exc = dfestimators.select("Te").collect().item()
        nnion = dfestimators.select(pl.col(f"nnion_{at.get_ionstring(Z, ion_stage, sep='_')}")).collect().item()

        dflevels = dflevels.lazy().with_columns(
            nnlevel_on_nnion=pl.col("g")
            * (-pl.col("energy_ev") / K_B / temperature_exc).exp()
            / ((pl.col("g") * (-pl.col("energy_ev") / K_B / temperature_exc).exp()).sum())
        )

        dftransitions = (
            dftransitions
            .with_columns(nu_trans=pl.col("epsilon_trans_ev") / const.h_ev_s)
            .filter(
                pl.col("lambda_angstroms").is_between(
                    lambda_lowers[selected_binindex], lambda_uppers[selected_binindex]
                )
            )
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
                exopac_contribution=(1 - (-pl.col("tau_sobolev_on_nnion") * nnion).exp())
                * pl.col("lambda_angstroms")
                / expopac_deltalambda
                / c
                / time_s
                / rho,
                linebinned_maxone_contribution=pl.min_horizontal(pl.col("tau_sobolev_on_nnion") * nnion, 1.0)
                * pl.col("lambda_angstroms")
                / expopac_deltalambda
                / (c * time_s * rho),
                linebinned_contribution=pl.col("tau_sobolev_on_nnion")
                * nnion
                * pl.col("lambda_angstroms")
                / expopac_deltalambda
                / (c * time_s * rho),
            )
        )
        lazy_dfs.append(
            dftransitions.select(
                expopac_expansion=pl.col("exopac_contribution").sum(),
                exopac_linebinned=pl.col("linebinned_contribution").sum(),
                linebinned_maxone_contribution=pl.col("linebinned_maxone_contribution").sum(),
            )
        )

    dfresults = (
        pl
        .concat(lazy_dfs)
        .select(
            expopac_expansion=pl.col("expopac_expansion").sum(),
            exopac_linebinned=pl.col("exopac_linebinned").sum(),
            linebinned_maxone_contribution=pl.col("linebinned_maxone_contribution").sum(),
        )
        .collect()
    )

    opac_expansion = dfresults["expopac_expansion"].item()
    opac_linebinned = dfresults["exopac_linebinned"].item()
    opac_linebinned_maxone = dfresults["linebinned_maxone_contribution"].item()

    print()
    print(f"timestep {timestep} T_days = {time_days:.2e}")
    print(f"cell {modelgridindex} T_exc = {temperature_exc} K")
    print(f"{'bin_lambda_lower':<20} {'kappa_expansion':<20} {'kappa_linebinned':<20} {'kappa_linebinned_maxone':<20}")
    print(
        f"{lambda_lowers[selected_binindex]:<20} {opac_expansion:<20.2e} {opac_linebinned:<20.2e} {opac_linebinned_maxone:<20.2e}"
    )


if __name__ == "__main__":
    main()
