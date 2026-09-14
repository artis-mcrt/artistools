# PYTHON_ARGCOMPLETE_OK
"""Plot the mass fractions of an ARTIS input model against atomic or mass number."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.misc import addarg_figscale
from artistools.misc import print_warning
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend


def filter_model_cells(
    dfmodel: pl.LazyFrame,
    modelmeta: dict[str, t.Any],
    vmin: float | None = None,
    vmax: float | None = None,
    thetamin: float | None = None,
    thetamax: float | None = None,
) -> pl.LazyFrame:
    """Keep the cells whose mid-point velocity [c] and polar angle [degrees] are inside the given ranges.

    Call add_derived_cols_to_modeldata first, because this function reads the vel_*_on_c columns. The
    positive z axis gives a polar angle of zero. A cell at the origin has no polar angle, thus an angle
    range excludes it. A 1D model has no polar angle, thus an angle range on a 1D model raises an error.
    """
    if vmin is not None:
        dfmodel = dfmodel.filter(pl.col("vel_r_mid_on_c") >= vmin)
    if vmax is not None:
        dfmodel = dfmodel.filter(pl.col("vel_r_mid_on_c") <= vmax)

    if thetamin is thetamax is None:
        return dfmodel

    if modelmeta["dimensions"] == 1:
        msg = "-thetamin and -thetamax need a 2D or 3D model, but the model is 1D"
        raise ValueError(msg)

    # polars orders NaN above each finite value, thus the angle 0 / 0 of a cell at the origin needs its own test
    dfmodel = dfmodel.filter(pl.col("vel_r_mid_on_c") > 0.0)
    theta_deg = (pl.col("vel_z_mid_on_c") / pl.col("vel_r_mid_on_c")).arccos().degrees()
    if thetamin is not None:
        dfmodel = dfmodel.filter(theta_deg >= thetamin)
    if thetamax is not None:
        dfmodel = dfmodel.filter(theta_deg <= thetamax)

    return dfmodel


def get_nuclide_massfractions(
    modelpath: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    thetamin: float | None = None,
    thetamax: float | None = None,
) -> pl.DataFrame:
    """Return the mass-weighted mass fraction and number abundance of each nuclide in the selected cells."""
    dfmodel, modelmeta = at.inputmodel.get_modeldata(modelpath=modelpath)
    dfmodel = at.inputmodel.add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)
    dfmodel = filter_model_cells(dfmodel, modelmeta, vmin=vmin, vmax=vmax, thetamin=thetamin, thetamax=thetamax)

    # one collect gives the mass-weighted sums and the total mass, thus the model scan runs one time
    dfsums = dfmodel.select(
        cs.matches(r"^X_[A-Z][a-z]?\d+$").dot(pl.col("mass_g")), mass_g=pl.col("mass_g").sum()
    ).collect()

    # a selection with no cell would give 0 / 0 = NaN for every nuclide and a blank plot
    if dfsums["mass_g"].item() == 0.0:
        msg = f"No cell of {modelpath} is inside the velocity range and the polar angle range"
        raise ValueError(msg)

    return (
        dfsums
        .select(cs.exclude("mass_g") / pl.col("mass_g"))
        .unpivot(variable_name="nuclide", value_name="massfraction")
        # split X_Ni56 into its element symbol and mass number, then a join with the element table gives Z
        .with_columns(
            elsymbol=pl.col("nuclide").str.extract(r"^X_([A-Z][a-z]?)\d+$"),
            A=pl.col("nuclide").str.extract(r"^X_[A-Z][a-z]?(\d+)$").cast(pl.Int32),
        )
        .join(at.get_elsymbols_df().collect(), on="elsymbol", how="left", maintain_order="left")
        .rename({"atomic_number": "Z"})
        .with_columns(abundance=pl.col("massfraction") / pl.col("A"))
    )


def make_plot(args: argparse.Namespace) -> None:
    """Plot the mass-weighted abundances of every model in args.modelpath and save the figure."""
    args.xaxis = {"Z": "atomicnumber", "A": "massnumber"}.get(args.xaxis, args.xaxis)

    fig, axesgrid = make_frame_figure(args)
    ax = axesgrid[0][0]

    for model_path in args.modelpath:
        df = get_nuclide_massfractions(
            Path(model_path), vmin=args.vmin, vmax=args.vmax, thetamin=args.thetamin, thetamax=args.thetamax
        )

        # the join replaced get_atomic_number's assert, so an unrecognised symbol would otherwise leave a null Z
        # and be plotted as a stray bin instead of raising
        if unknown := df.filter(pl.col("Z").is_null())["elsymbol"].unique().to_list():
            msg = f"Unknown element symbols in {model_path}: {unknown}"
            raise ValueError(msg)

        massfracsum = df["massfraction"].sum()
        if not math.isclose(massfracsum, 1.0, abs_tol=1e-5):
            print_warning(f"mass fractions for model {model_path} sum to {massfracsum:.3f} instead of 1.0.")

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

    set_legend(ax, args)

    strxaxis = "A" if args.xaxis == "massnumber" else "Z"
    stryaxis = "X" if args.yaxis == "massfraction" else "abundance"
    outpath = at.resolve_outputfile(args.outputfile, f"plotinitialabundances_{stryaxis}vs{strxaxis}.pdf")
    save_figure(fig, outpath, args=args, dpi=300)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    at.addarg_output(parser, kind="file", default=Path())

    addarg_figscale(parser)
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
    parser.add_argument("-vmin", type=float, default=None, help="Minimum mid-point velocity of a cell [c]")
    parser.add_argument("-vmax", type=float, default=None, help="Maximum mid-point velocity of a cell [c]")
    parser.add_argument(
        "-thetamin", type=float, default=None, help="Minimum polar angle of a cell from the z axis [degrees]"
    )
    parser.add_argument(
        "-thetamax", type=float, default=None, help="Maximum polar angle of a cell from the z axis [degrees]"
    )

    at.addarg_show(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot initial abundances or mass fractions from one or more ARTIS models."""
    args = at.parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    make_plot(args)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
