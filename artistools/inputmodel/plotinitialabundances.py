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
    at.plottools.set_mpl_style()
    fig, ax = plt.subplots()

    xaxis = "A" if args.xaxis == "massnumber" else "Z"
    yaxis = "X" if args.yaxis == "massfraction" else "Y"

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
            Z = 0 if elem == "n" else at.get_atomic_number(elem)
            if Z < 0:
                continue
            meta.append((c, A, Z))

        meta_df = pl.DataFrame(meta, schema=["column", "A", "Z"])

        long = df.unpivot(on=nuclide_cols, index=["m_cell"], variable_name="column", value_name="X").join(
            meta_df, on="column"
        )

        axis = "A" if xaxis == "A" else "Z"
        suffix = axis

        if yaxis == "X":
            y_expr = pl.col("X")
            y_name = f"X_{suffix}"
        else:
            y_expr = pl.col("X") / pl.col("A")
            y_name = f"Y_{suffix}"

        value_name = "m_total"
        weighted_name = f"m_{suffix}" if yaxis == "X" else "abund_weighted_mass"

        result = (
            long
            .with_columns(y_expr.alias("Y"))
            .group_by(axis)
            .agg([
                (pl.col("Y") * pl.col("m_cell")).sum().alias(weighted_name),
                pl.col("m_cell").sum().alias(value_name),
            ])
            .with_columns((pl.col(weighted_name) / pl.col(value_name)).alias(y_name))
            .select([axis, y_name])
            .sort(axis)
        )

        df_plot = result.to_pandas()

        ax.plot(df_plot[axis], df_plot[f"{yaxis}_{xaxis}"], label=at.get_model_name(model_path))

    ax.set_xlabel("mass number" if xaxis == "A" else "charge number")
    ax.set_ylabel("mass fraction" if yaxis == "X" else "abundance")

    ax.set_yscale("log")

    ylim_values = (1e-5, 1.0) if yaxis == "X" else (1e-7, 0.1)
    ax.set_ylim(*ylim_values)

    ax.legend()

    pdf_name = f"plotinitialabundances_{yaxis}vs{xaxis}.pdf"
    fig.savefig(Path(args.outputpath) / pdf_name, dpi=300)
    plt.close(fig)


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", type=Path, default=Path())
    parser.add_argument(
        "modelpath",
        default=[Path()],
        nargs="*",
        type=Path,
        help="Path(s) to ARTIS folders for which abundances / mass fractions shall be plotted",
    )

    parser.add_argument("-xaxis", "-x", type=str, default="massnumber", choices=["massnumber", "chargenumber"])
    parser.add_argument("-yaxis", "-y", type=str, default="massfraction", choices=["massfraction", "abundance"])


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot initial abundances or mass fractions from one or more ARTIS models."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=main.__doc__,
        )
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    make_plot(args)


if __name__ == "__main__":
    main()
