"""Plot the mass fractions of an ARTIS input model against atomic or mass number."""

import argparse
import math
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import polars.selectors as cs

from artistools.atomic import get_elsymbols_df
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import get_cell_selection
from artistools.inputmodel.core import get_modeldata
from artistools.inputmodel.core import get_selection_labels
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import get_model_name
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend


def get_nuclide_massfractions(
    modelpath: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    thetamin: float | None = None,
    thetamax: float | None = None,
) -> pl.DataFrame:
    """Return the mass-weighted mass fraction and number abundance of each nuclide in the selected cells."""
    dfmodel, modelmeta = get_modeldata(modelpath=modelpath)
    if modelmeta["dimensions"] == 1 and (thetamin is not None or thetamax is not None):
        msg = f"A polar angle range needs a 2D or 3D model, but {modelpath} is 1D"
        raise ValueError(msg)

    dfmodel = add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)

    # A weight of zero in place of a row filter prevents a copy of every nuclide column. One collect gives
    # the mass-weighted sums, the total mass, and the cell count in one scan of the model.
    selected = get_cell_selection(vmin=vmin, vmax=vmax, thetamin=thetamin, thetamax=thetamax)
    weight = pl.when(selected).then(pl.col("mass_g")).otherwise(0.0)
    dfsums = dfmodel.select(
        cs.matches(r"^X_[A-Z][a-z]?\d+$").dot(weight), mass_g=weight.sum(), ncells=selected.sum()
    ).collect()

    selection = ", ".join(get_selection_labels(vmin=vmin, vmax=vmax, thetamin=thetamin, thetamax=thetamax))
    ncells = dfsums["ncells"].item()
    if ncells == 0:
        msg = f"Every cell of {modelpath} is outside the selection ({selection})"
        raise ValueError(msg)

    # a selection of empty cells would give 0 / 0 = NaN for every nuclide and a blank plot
    if dfsums["mass_g"].item() == 0.0:
        msg = f"The {ncells} selected cells of {modelpath} hold no mass ({selection or 'all cells'})"
        raise ValueError(msg)

    dfnuclides = (
        dfsums
        .select(cs.exclude("mass_g", "ncells") / pl.col("mass_g"))
        .unpivot(variable_name="nuclide", value_name="massfraction")
        # split X_Ni56 into its element symbol and mass number, then a join with the element table gives Z
        .with_columns(pl.col("nuclide").str.extract_groups(r"^X_(?<elsymbol>[A-Z][a-z]?)(?<A>\d+)$").struct.unnest())
        .with_columns(pl.col("A").cast(pl.Int32))
        .join(get_elsymbols_df().collect(), on="elsymbol", how="left", maintain_order="left")
        .rename({"atomic_number": "Z"})
        .with_columns(abundance=pl.col("massfraction") / pl.col("A"))
    )

    # The join replaced the assert in get_atomic_number. Without this test, an unknown symbol keeps a null Z,
    # and the plot shows a stray bin.
    if unknown := dfnuclides.filter(pl.col("Z").is_null())["elsymbol"].unique().to_list():
        msg = f"Unknown element symbols in {modelpath}: {unknown}"
        raise ValueError(msg)

    massfracsum = dfnuclides["massfraction"].sum()
    if not math.isclose(massfracsum, 1.0, abs_tol=1e-5):
        print_warning(f"mass fractions for model {modelpath} sum to {massfracsum:.3f} instead of 1.0.")

    return dfnuclides


def make_plot(args: argparse.Namespace) -> None:
    """Plot the mass-weighted abundances of every model in args.modelpath and save the figure."""
    args.xaxis = {"Z": "atomicnumber", "A": "massnumber"}.get(args.xaxis, args.xaxis)

    # Read every model before the code makes the figure. An error in the data then occurs while no figure is open.
    dfmodels = [
        get_nuclide_massfractions(
            Path(model_path), vmin=args.vmin, vmax=args.vmax, thetamin=args.thetamin, thetamax=args.thetamax
        )
        for model_path in args.modelpath
    ]

    fig, axesgrid = make_frame_figure(args)
    ax = axesgrid[0][0]

    for model_path, dfnuclides in zip(args.modelpath, dfmodels, strict=True):
        df = (
            dfnuclides
            .select(
                xvalue="A" if args.xaxis == "massnumber" else "Z",
                yvalue="massfraction" if args.yaxis == "massfraction" else "abundance",
            )
            .group_by("xvalue")
            .agg(pl.col("yvalue").sum())
            .sort("xvalue")
        )

        ax.plot(df["xvalue"], df["yvalue"], label=get_model_name(model_path))

    ax.set_xlabel("Mass number" if args.xaxis == "massnumber" else "Atomic number")
    ax.set_ylabel("Mass fraction" if args.yaxis == "massfraction" else "Number abundance")

    ax.set_yscale("log")

    ax.set_ylim(*((1e-5, 1.0) if args.yaxis == "massfraction" else (1e-7, 0.1)))

    selectionlabels = get_selection_labels(
        vmin=args.vmin, vmax=args.vmax, thetamin=args.thetamin, thetamax=args.thetamax
    )
    if selectionlabels:
        ax.set_title(", ".join(selectionlabels))

    set_legend(ax, args)

    strxaxis = "A" if args.xaxis == "massnumber" else "Z"
    stryaxis = "X" if args.yaxis == "massfraction" else "abundance"
    # the default file name records the selection, thus a second run with a range keeps the earlier figure
    namesuffix = "".join(f"_{label.replace('=', '')}" for label in selectionlabels)
    outpath = resolve_outputfile(args.outputfile, f"plotinitialabundances_{stryaxis}vs{strxaxis}{namesuffix}.pdf")
    save_figure(fig, outpath, args=args, dpi=300)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_output(parser, kind="file", default=Path())

    addarg_figscale(parser)
    addarg_modelpath(
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

    addarg_show(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot initial abundances or mass fractions from one or more ARTIS models."""
    args = parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    make_plot(args)
