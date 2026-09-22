"""Extract a 1D ARTIS model from the cells of a 3D model that lie along one coordinate axis."""

import argparse
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.inputmodel.core import get_initelemabundances
from artistools.inputmodel.core import get_modeldata
from artistools.inputmodel.core import LOGRHO_FROM_RHO
from artistools.inputmodel.core import save_initelemabundances
from artistools.inputmodel.core import save_modeldata
from artistools.misc import parse_cli_args
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-inputfolder", action="store", default=".", help="Path to folder with 3D files")

    parser.add_argument(
        "-axis", action="store", dest="chosenaxis", default="x", choices=["x", "y", "z"], help="Slice axis (x, y, or z)"
    )

    parser.add_argument(
        "-outputfolder", action="store", default="1dslice", help="Path to folder in which to store 1D output files"
    )

    parser.add_argument("-opdf", action="store", dest="pdfoutputfile", help="Path/filename for PDF plot")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Convert abundances.txt and model.txt from a 3D model to a one-dimensional slice."""
    args = parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    if not Path(args.outputfolder).exists():
        Path(args.outputfolder).mkdir(parents=True)
    elif Path(args.outputfolder, "model.txt").exists():
        print("ABORT: model.txt already exists")
        sys.exit()
    elif Path(args.outputfolder, "abundances.txt").exists():
        print("ABORT: abundances.txt already exists")
        sys.exit()

    dict3dcellidto1dcellid, xlist, ylists = slice_3dmodel(args.inputfolder, args.outputfolder, args.chosenaxis)

    slice_abundance_file(args.inputfolder, args.outputfolder, dict3dcellidto1dcellid)

    if args.pdfoutputfile:
        make_plot(xlist, ylists, args.pdfoutputfile)


def slice_3dmodel(
    inputfolder: Path | str, outputfolder: Path | str, chosenaxis: str
) -> tuple[dict[int, int], list[float], list[list[float]]]:
    """Write a 1D model.txt from the cells along chosenaxis, and return the 3D-to-1D cell id map and plot data."""
    dfmodel3d, modelmeta3d = get_modeldata(inputfolder)
    if modelmeta3d["dimensions"] != 3:
        msg = f"The model in {inputfolder} has {modelmeta3d['dimensions']} dimensions, but the slice needs a 3D model"
        raise ValueError(msg)
    t_model_s = modelmeta3d["t_model_init_days"] * day_to_s
    wid_init = modelmeta3d["wid_init"]

    # A grid with an even cell count has a cell face on the axis, and the slice takes the cell on the positive
    # side. With an odd count, the axis goes through the middle of a cell. The reader gives Float32
    # positions, thus a face counts as on the axis within a small part of the cell width
    postolerance = 1e-3 * wid_init
    dfslice = (
        dfmodel3d
        .filter(
            *(
                pl.col(f"pos_{ax}_min").is_between(-wid_init + postolerance, postolerance, closed="left")
                for ax in "xyz"
                if ax != chosenaxis
            ),
            pl.col(f"pos_{chosenaxis}_min") >= -wid_init + postolerance,
        )
        .sort("inputcellid")
        .with_columns(
            # pos_min is the inner face of a cell, but the 1D model gives the outer boundary of a shell.
            # The cell width makes that outer face, thus the first shell holds a volume
            vel_r_max_kmps=(pl.col(f"pos_{chosenaxis}_min") + wid_init) / t_model_s / km_to_cm,
            logrho=LOGRHO_FROM_RHO,
        )
        .collect()
    )
    dict3dcellidto1dcellid = {cellid3d: cellid1d for cellid1d, cellid3d in enumerate(dfslice["inputcellid"], start=1)}
    dfslice = dfslice.with_columns(inputcellid=pl.int_range(1, pl.len() + 1, dtype=pl.Int32))

    save_modeldata(
        dfslice,
        outpath=outputfolder,
        dimensions=1,
        t_model_init_days=modelmeta3d["t_model_init_days"],
        headercommentlines=[
            *modelmeta3d["headercommentlines"],
            f"slice of a 3D model along the positive {chosenaxis} axis",
        ],
    )

    ylists = [dfslice[col].to_list() for col in ("rho", "X_Ni56", "X_Co56")]
    return dict3dcellidto1dcellid, dfslice["vel_r_max_kmps"].to_list(), ylists


def slice_abundance_file(
    inputfolder: Path | str, outputfolder: Path | str, dict3dcellidto1dcellid: dict[int, int]
) -> None:
    """Write an abundances.txt holding only the cells kept by slice_3dmodel, renumbered to the 1D cell ids."""
    dfelabundances = (
        get_initelemabundances(inputfolder)
        .filter(pl.col("inputcellid").is_in(list(dict3dcellidto1dcellid)))
        .with_columns(pl.col("inputcellid").replace_strict(dict3dcellidto1dcellid, return_dtype=pl.Int32))
        .collect()
    )
    # save_initelemabundances writes 0.0 for a null value, thus a short line must stop the command here
    if dfelabundances.height != len(dict3dcellidto1dcellid) or dfelabundances.null_count().sum_horizontal().item() > 0:
        msg = (
            f"abundances.txt in {inputfolder} does not hold a full line of mass fractions for each of the "
            f"{len(dict3dcellidto1dcellid)} cells of the slice"
        )
        raise ValueError(msg)
    save_initelemabundances(dfelabundances, outpath=outputfolder)


def make_plot(xlist: list[float], ylists: list[list[float]], pdfoutputfile: str) -> None:
    """Plot density and the Ni56 and Co mass fractions of the slice against velocity, and save it as a PDF."""
    fig, axesgrid = make_frame_figure()
    axis = axesgrid[0][0]
    axis.set_xlabel(r"Velocity [km/s]")
    axis.set_ylabel(r"Density [g/cm$^3$] or mass fraction")
    ylabels = [r"$\rho$", "fNi56", "fCo"]
    for ylist, ylabel in zip(ylists, ylabels, strict=False):
        axis.plot(xlist, ylist, linewidth=1.5, label=ylabel)
    axis.set_yscale("log", nonpositive="clip")
    set_legend(axis)
    save_figure(fig, pdfoutputfile, format="pdf")
