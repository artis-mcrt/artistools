"""Plot the macro atom transition rates recorded in ARTIS macroatom_????.out files."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl

import artistools as at
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelgridindex
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import addarg_timestep
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure

defaultoutputfile = "plotmacroatom_cell{cell:05d}_ts{timestep:03d}-{timestep2:03d}.pdf"


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())

    addarg_figscale(parser)
    # deprecated double-dash spelling kept as a hidden alias
    parser.add_argument("--modelpath", dest="modelpath", type=Path, help=argparse.SUPPRESS)
    addarg_timestep(parser, default=10, helptext="Timestep number to plot, e.g. 40 or last")
    parser.add_argument("-timestepmax", type=int, default=-1, help="Make plots for all timesteps up to this timestep")
    addarg_modelgridindex(parser, default=0)
    parser.add_argument("element", nargs="?", default="Fe", help="Plotted element")
    addarg_axislimits(
        parser,
        xmindefault=1000,
        xmaxdefault=15000,
        xminhelp="Plot range: minimum wavelength in Angstroms",
        xmaxhelp="Plot range: maximum wavelength in Angstroms",
        include_y=False,
        wavelength_aliases=True,
    )
    addarg_output(parser, kind="file", defaultname=defaultoutputfile, helptext="Filename for PDF file")
    addarg_show(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the macroatom transitions."""
    args = at.parse_cli_args(addargs, "Plot ARTIS macroatom transitions.", args, argsraw, kwargs)

    atomic_number = at.get_atomic_number(args.element)
    if atomic_number < 1:
        at.exit_with_error(f"could not find element '{args.element}'")

    modelgridindex = at.get_single_modelgridindex(args.modelgridindex)
    timestepmin = at.get_single_timestep(args.timestep, args.modelpath)
    assert timestepmin is not None, "-timestep holds a default, thus it names a timestep"

    timestepmax = timestepmin if not args.timestepmax or args.timestepmax < 0 else args.timestepmax

    # the template took {0}, {1}, and {2} before it took names, thus a script holds those fields
    outputfile = str(args.outputfile).format(
        modelgridindex, timestepmin, timestepmax, cell=modelgridindex, timestep=timestepmin, timestep2=timestepmax
    )
    modelpath = args.modelpath
    xmin = args.xmin
    xmax = args.xmax
    time_days_min = at.get_timestep_time(modelpath, timestepmin)
    time_days_max = at.get_timestep_time(modelpath, timestepmax)

    dfmacroatom = read_files(modelpath, modelgridindex, timestepmin, timestepmax, atomic_number)
    print(f"Plotting {len(dfmacroatom)} transitions")

    fig, axesgrid = make_frame_figure(args, aspect=1.059)
    axis = axesgrid[0][0]

    axis.annotate(
        f"Timestep {timestepmin:d} to {timestepmax:d} (t={time_days_min} to {time_days_max})\nCell {modelgridindex:d}",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize="x-small",
    )

    with np.errstate(divide="ignore"):
        lambda_cmf_in = at.constants.c_ang_per_s / dfmacroatom["nu_cmf_in"].to_numpy()
        lambda_cmf_out = at.constants.c_ang_per_s / dfmacroatom["nu_cmf_out"].to_numpy()
    axis.plot(
        lambda_cmf_in,
        lambda_cmf_out,
        linestyle="none",
        marker="o",  # alpha=0.5,
        markersize=2,
        markerfacecolor="red",
        markeredgewidth=0,
    )
    axis.set_xlabel(r"Wavelength in ($\AA$)")
    axis.set_ylabel(r"Wavelength out ($\AA$)")
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(xmin, xmax)

    save_figure(fig, outputfile, args=args, format="pdf")


def read_files(
    modelpath: Path | str,
    modelgridindex: int | None = None,
    timestepmin: int | None = None,
    timestepmax: int | None = None,
    atomic_number: int | None = None,
) -> pl.DataFrame:
    """Return the macro atom transitions of a model, filtered by cell, timestep range, and element.

    read_rank_outputfiles reads one file for each rank, thus a run that holds a plain file and a
    compressed file of the same rank does not read that rank twice.
    """
    dfmacroatom = at.read_rank_outputfiles(modelpath, "macroatom_{mpirank:04d}.out", modelgridindex=modelgridindex)

    if timestepmin is not None:
        dfmacroatom = dfmacroatom.filter(pl.col("timestep") >= timestepmin)
    if timestepmax is not None:
        dfmacroatom = dfmacroatom.filter(pl.col("timestep") <= timestepmax)
    if atomic_number:
        dfmacroatom = dfmacroatom.filter(pl.col("Z") == atomic_number)

    if dfmacroatom.is_empty():
        msg = f"{modelpath} holds no macro atom transition for this cell, timestep range, and element"
        raise ValueError(msg)

    return dfmacroatom


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
