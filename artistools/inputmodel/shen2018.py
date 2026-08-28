"""Convert the Shen et al. (2018) sub-Chandrasekhar detonation models to ARTIS format."""

import argparse
import math
import string
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

import artistools as at
from artistools.constants import km_to_cm
from artistools.constants import Msun_to_g


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-inputpath", "-i", default="1.00_5050.dat", help="Path of input file")
    at.addarg_output(parser, kind="folder", default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Convert Shen et al. 2018 models to ARTIS format."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    datain = at.read_wsv(args.inputpath).drop_nulls()

    isotopesofelem: dict[int, list[str]] = {}
    for species in datain.columns[5:]:
        atomic_number = at.get_atomic_number(species.rstrip(string.digits))
        isotopesofelem.setdefault(atomic_number, []).append(species)

    t_model_init_seconds = 10.0
    t_model_init_days = t_model_init_seconds / 24 / 60 / 60

    # the shell masses and radii come from the enclosed mass and velocity of each cell's outer boundary
    dfshells = (
        datain
        .with_row_index("cellid")
        .with_columns(
            m_enc_outer=pl.col("m") * Msun_to_g,  # convert Solar masses to grams
            v_outer=pl.col("v") * 1e-5,  # convert cm/s to km/s
        )
        .with_columns(
            m_shell_grams=pl.col("m_enc_outer") - pl.col("m_enc_outer").shift(1, fill_value=0.0),
            r_outer=pl.col("v_outer") * km_to_cm * t_model_init_seconds,
            r_inner=pl.col("v_outer").shift(1, fill_value=0.0) * km_to_cm * t_model_init_seconds,
        )
        .with_columns(
            rho=pl.col("m_shell_grams") / (4.0 / 3.0 * math.pi * (pl.col("r_outer") ** 3 - pl.col("r_inner") ** 3)),
            X_Fegroup=pl.sum_horizontal([
                pl.col(species)
                for atomic_number, specieslist in isotopesofelem.items()
                if 26 <= atomic_number <= 30
                for species in specieslist
            ]),
        )
    )

    m_enc_outer = float(dfshells["m_enc_outer"].item(-1))
    tot_ni56mass = float((dfshells["m_shell_grams"] * dfshells["ni56"]).sum())

    dfmodel = dfshells.select(
        inputcellid=pl.col("cellid"),
        vel_r_max_kmps=pl.col("v_outer"),
        logrho=pl.col("rho").log10(),
        X_Fegroup=pl.col("X_Fegroup"),
        X_Ni56=pl.col("ni56"),
        X_Co56=pl.col("co56"),
        X_Fe52=pl.col("fe52"),
        X_Cr48=pl.col("cr48"),
        X_Ni57=pl.col("ni57"),
        X_Co57=pl.col("co57"),
    )

    dfelabundances = dfshells.select(
        pl.col("cellid").alias("inputcellid"),
        *(
            pl.sum_horizontal([pl.col(species) for species in isotopesofelem[atomic_number]]).alias(
                f"X_{at.get_elsymbol(atomic_number)}"
            )
            for atomic_number in range(1, 31)
        ),
    )

    print(f"M_tot  = {m_enc_outer / Msun_to_g:.3f} solMass")
    print(f"M_Ni56 = {tot_ni56mass / Msun_to_g:.3f} solMass")

    at.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, outpath=args.outputfile)
    at.inputmodel.save_initelemabundances(dfelabundances=dfelabundances, outpath=args.outputfile)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
