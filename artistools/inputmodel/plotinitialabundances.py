# PYTHON_ARGCOMPLETE_OK
"""Plot the mass fractions of an ARTIS input model against atomic or mass number."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.plottools import save_figure


def make_plot(args: argparse.Namespace) -> None:
    """Plot the mass-weighted abundances of every model in args.modelpath and save the figure."""
    args.xaxis = {"Z": "atomicnumber", "A": "massnumber"}.get(args.xaxis, args.xaxis)

    at.plottools.set_mpl_style()
    fig, ax = plt.subplots(tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for model_path in args.modelpath:
        df, _ = at.inputmodel.get_modeldata(modelpath=Path(model_path), derived_cols=["mass_g"])
        df = (
            df
            .select((cs.matches(r"^X_[A-Z][a-z]?\d+$").dot(pl.col("mass_g"))) / pl.col("mass_g").sum())
            .unpivot(variable_name="nuclide", value_name="massfraction")
            # split X_Ni56 into its element symbol and mass number, then look Z up by joining the element table
            .with_columns(
                elsymbol=pl.col("nuclide").str.extract(r"^X_([A-Z][a-z]?)\d+$"),
                A=pl.col("nuclide").str.extract(r"^X_[A-Z][a-z]?(\d+)$").cast(pl.Int32),
            )
            .join(at.get_elsymbols_df(), on="elsymbol", how="left")
            .rename({"atomic_number": "Z"})
            .with_columns(abundance=pl.col("massfraction") / pl.col("A"))
            .collect()
        )

        # the join replaced get_atomic_number's assert, so an unrecognised symbol would otherwise leave a null Z
        # and be plotted as a stray bin instead of raising
        if unknown := df.filter(pl.col("Z").is_null())["elsymbol"].unique().to_list():
            msg = f"Unknown element symbols in {model_path}: {unknown}"
            raise ValueError(msg)

        massfracsum = df["massfraction"].sum()
        if not math.isclose(massfracsum, 1.0, abs_tol=1e-5):
            print(f"WARNING: mass fractions for model {model_path} sum to {massfracsum:.3f} instead of 1.0.")

        df = (
            df
            .select(
                xvalue="A" if args.xaxis == "massnumber" else "Z",
                yvalue="massfraction" if args.yaxis == "massfraction" else "abundance",
            )
            .group_by("xvalue")
            .agg(pl.col("yvalue").sum())
            .sort("xvalue")
        )

        ax.plot(df["xvalue"], df["yvalue"], label=at.get_model_name(model_path))

    ax.set_xlabel("Mass number" if args.xaxis == "massnumber" else "Atomic number")
    ax.set_ylabel("Mass fraction" if args.yaxis == "massfraction" else "Number abundance")

    ax.set_yscale("log")

    ax.set_ylim(*((1e-5, 1.0) if args.yaxis == "massfraction" else (1e-7, 0.1)))

    ax.legend()

    strxaxis = "A" if args.xaxis == "massnumber" else "Z"
    stryaxis = "X" if args.yaxis == "massfraction" else "abundance"
    outpath = Path(args.outputpath) / f"plotinitialabundances_{stryaxis}vs{strxaxis}.pdf"
    save_figure(fig, outpath, show=args.show, dpi=300)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    at.addarg_outputpath(parser, default=Path(), astype=Path)
    at.addarg_modelpath(
        parser,
        positional=True,
        multiplepaths=True,
        default=[Path()],
        helptext="Path(s) to ARTIS folders for which abundances / mass fractions shall be plotted",
    )

    parser.add_argument(
        "-xaxis",
        "-x",
        type=str,
        default="massnumber",
        choices=["massnumber", "atomicnumber", "Z", "A"],
        help="Horizontal axis quantity: mass number A or atomic number Z",
    )
    parser.add_argument(
        "-yaxis",
        "-y",
        type=str,
        default="massfraction",
        choices=["massfraction", "abundance"],
        help="Vertical axis quantity: mass fraction or number abundance",
    )
    at.addarg_show(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot initial abundances or mass fractions from one or more ARTIS models."""
    args = at.parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    make_plot(args)


if __name__ == "__main__":
    main()
