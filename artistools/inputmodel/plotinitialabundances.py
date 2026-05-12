import argparse
import re
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import polars as pl

import artistools as at


def make_plot(args: argparse.Namespace) -> None:
    for model_path in args.modelpath:
        df = at.inputmodel.get_modeldata(modelpath=Path(model_path), derived_cols=["volume", "rho"])[0].collect()

        df = df.with_columns((pl.col("rho") * pl.col("volume")).alias("m_cell"))

        nuclide_cols = df.select(pl.col("^X_.*$")).columns
        pattern = re.compile(r"X_([A-Z][a-z]?|n)(\d+)")

        meta = []
        for c in nuclide_cols:
            m = pattern.fullmatch(c)
            if not m:
                continue
            elem, A = m.group(1), int(m.group(2))
            Z = at.get_atomic_number(elem)
            meta.append((c, A, Z))

        meta_df = pl.DataFrame(meta, schema=["column", "A", "Z"])

        long = df.unpivot(on=nuclide_cols, index=["m_cell"], variable_name="column", value_name="X").join(
            meta_df, on="column"
        )

        if args.xaxis == "A":
            xcol = "A"

            if args.yaxis == "X":
                result = (
                    long
                    .group_by("A")
                    .agg([(pl.col("X") * pl.col("m_cell")).sum().alias("m_A"), pl.col("m_cell").sum().alias("m_total")])
                    .with_columns((pl.col("m_A") / pl.col("m_total")).alias("X_A"))
                    .select(["A", "X_A"])
                    .sort("A")
                )
            else:
                long = long.with_columns((pl.col("X") / pl.col("A")).alias("Y"))
                result = (
                    long
                    .group_by("A")
                    .agg([
                        (pl.col("Y") * pl.col("m_cell")).sum().alias("abund_weighted_mass"),
                        pl.col("m_cell").sum().alias("m_total"),
                    ])
                    .with_columns((pl.col("abund_weighted_mass") / pl.col("m_total")).alias("Y_A"))
                    .select(["A", "Y_A"])
                    .sort("A")
                )

        else:
            xcol = "Z"

            if args.yaxis == "X":
                result = (
                    long
                    .group_by("Z")
                    .agg([(pl.col("X") * pl.col("m_cell")).sum().alias("m_Z"), pl.col("m_cell").sum().alias("m_total")])
                    .with_columns((pl.col("m_Z") / pl.col("m_total")).alias("X_Z"))
                    .select(["Z", "X_Z"])
                    .sort("Z")
                )
            else:
                long = long.with_columns((pl.col("X") / pl.col("A")).alias("Y"))

                result = (
                    long
                    .group_by("Z")
                    .agg([
                        (pl.col("Y") * pl.col("m_cell")).sum().alias("abund_weighted_mass"),
                        pl.col("m_cell").sum().alias("m_total"),
                    ])
                    .with_columns((pl.col("abund_weighted_mass") / pl.col("m_total")).alias("Y_Z"))
                    .select(["Z", "Y_Z"])
                    .sort("Z")
                )

        df_plot = result.to_pandas()

        plt.figure()

        plt.plot(df_plot[xcol], df_plot[f"{args.yaxis}_{args.xaxis}"], label=at.get_model_name(model_path))

    plt.xlabel("mass number" if args.xaxis == "A" else "charge number")
    plt.ylabel("mass fraction" if args.yaxis == "X" else "abundance")

    plt.yscale("log")

    ylim_values = (1e-5, 1.0) if args.yaxis == "X" else (1e-7, 0.1)
    plt.ylim(ylim_values)
    plt.legend()

    pdf_name = f"plotinitialabundances_{args.yaxis}vs{args.xaxis}.pdf"
    plt.savefig(Path(args.outputpath) / pdf_name, dpi=300)
    plt.close()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", type=Path, default=Path())
    parser.add_argument(
        "modelpath",
        default=[Path()],
        nargs="*",
        type=Path,
        help="Path(s) to ARTIS folders for which abundances / mass fractions shall be plotted",
    )

    parser.add_argument("-xaxis", type=str, default="A", choices=["A", "Z"])
    parser.add_argument("-yaxis", type=str, default="X", choices=["X", "Y"])


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    make_plot(args)


if __name__ == "__main__":
    main()
