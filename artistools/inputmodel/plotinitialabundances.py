# PYTHON_ARGCOMPLETE_OK
import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs

import artistools as at


def make_plot(args: argparse.Namespace) -> None:
    args.xaxis = {"Z": "atomicnumber", "A": "massnumber"}.get(args.xaxis, args.xaxis)

    at.plottools.set_mpl_style()
    fig, ax = plt.subplots(tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for model_path in args.modelpath:
        df, _ = at.inputmodel.get_modeldata(modelpath=Path(model_path), derived_cols=["mass_g"])
        df = (
            df
            .select((cs.matches(r"^X_[A-Z][a-z]?\d+$") * pl.col("mass_g")).sum() / pl.col("mass_g").sum())
            .unpivot(variable_name="nuclide", value_name="massfraction")
            .with_columns(
                pl.col("nuclide").map_elements(
                    at.get_z_a_nucname, return_dtype=pl.Struct({"Z": pl.Int32, "A": pl.Int32})
                )  # convert X_Ni56 to {28, 56}
            )
            .unnest("nuclide")  # convert {28, 56} struct to columns Z and A
            .with_columns(abundance=pl.col("massfraction") / pl.col("A"))
        )
        assert math.isclose(df.select(pl.col("massfraction").sum()).collect().item(), 1.0, abs_tol=1e-5), (
            "Mass fractions do not sum to 1.0"
        )

        df = (
            df
            .select(
                xvalue="A" if args.xaxis == "massnumber" else "Z",
                yvalue="massfraction" if args.yaxis == "massfraction" else "abundance",
            )
            .group_by("xvalue")
            .agg(pl.col("yvalue").sum())
            .sort("xvalue")
            .collect()
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
    fig.savefig(outpath, dpi=300)
    print(f"open {outpath}")


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", type=Path, default=Path())
    parser.add_argument(
        "modelpath",
        default=[Path()],
        nargs="*",
        type=Path,
        help="Path(s) to ARTIS folders for which abundances / mass fractions shall be plotted",
    )

    parser.add_argument(
        "-xaxis", "-x", type=str, default="massnumber", choices=["massnumber", "atomicnumber", "Z", "A"]
    )
    parser.add_argument("-yaxis", "-y", type=str, default="massfraction", choices=["massfraction", "abundance"])


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot initial abundances or mass fractions from one or more ARTIS models."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=main.__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    make_plot(args)


if __name__ == "__main__":
    main()
