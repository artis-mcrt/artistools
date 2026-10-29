"""Plot per-rank stage durations from ARTIS log files."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from artistools.logs.core import get_logfiles
from artistools.logs.core import read_time_taken
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_legend
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_show
from artistools.misc import exit_with_error
from artistools.misc import format_frame_path
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import normalize_path_list
from artistools.misc import open_file
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import print_saved
from artistools.plottools import apply_darkmode
from artistools.plottools import make_frame_figure
from artistools.plottools import set_legend

defaultoutputfile = "plotlogfiles_{modelname}.pdf"


def get_time_taken_table(dfdrawnlines: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Return one row for each timestep and mpi rank, with one column of seconds for each stage.

    A stage that has no time for a rank or a timestep gives a null value.
    """
    if not dfdrawnlines:
        return pl.DataFrame()
    return (
        pl
        .concat(dfdrawnlines)
        .with_columns(stage=pl.col("stage") + "_s")
        .pivot(on="stage", index=["timestep", "mpirank"], values="time_s")
        .sort(["timestep", "mpirank"])
    )


def make_plot(
    logfiledict: dict[str, dict[int, dict[int, float]]],
    outputfile: Path | str,
    modelname: str = "",
    modellogname: str = "",
    *,
    args: argparse.Namespace,
) -> None:
    """Write one page per timestep of stage duration versus mpi rank to a multi-page PDF."""
    makestable = getattr(args, "print_data", False) or getattr(args, "write_data", False)
    dfdrawnlines: list[pl.DataFrame] = []

    # plot every timestep that has data for at least one stage, so a log format
    # missing one stage entirely still yields plots of the others
    timesteps = sorted(set().union(*(set(bytimestep) for bytimestep in logfiledict.values())))
    if not timesteps:
        print(f"No timing data found in the log files of {modellogname or modelname}")
        # the empty table replaces the file of an earlier run, thus that file cannot seem to hold the data of this run
        output_data(args, {"": get_time_taken_table([])}, outputfile)
        return

    # the PDF backend loads fontTools, which only this part of the command needs
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(outputfile) as pdf:
        for timestep in timesteps:
            fig, axesgrid = make_frame_figure()
            axis = axesgrid[0][0]
            for stage, bytimestep in logfiledict.items():
                if timestep not in bytimestep:
                    continue
                mpirank, timetaken = zip(*sorted(bytimestep[timestep].items()), strict=True)
                axis.plot(mpirank, timetaken, label=stage)
                if makestable:
                    dfdrawnlines.append(
                        pl.DataFrame(
                            {"mpirank": mpirank, "time_s": timetaken},
                            schema={"mpirank": pl.Int64, "time_s": pl.Float64},
                        ).select(
                            pl.lit(timestep, dtype=pl.Int64).alias("timestep"), pl.lit(stage).alias("stage"), pl.all()
                        )
                    )
            axis.set_xlabel("mpi rank")
            axis.set_ylabel("Time [s]")
            axis.set_title(f"{modelname} timestep {timestep}" if modelname else f"timestep {timestep}")
            set_legend(axis, args)
            # save_figure holds these rules for a figure of its own, and this pdf writes its own pages
            # without --darkmode, the transparency of the matplotlib configuration applies
            darkkwargs = {"transparent": apply_darkmode(fig, "pdf")} if getattr(args, "darkmode", False) else {}
            if getattr(args, "show", False):
                plt.show()
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02, **darkkwargs)
            plt.close(fig)

    print_saved(outputfile)
    if getattr(args, "open", False):
        open_file(outputfile)
    output_data(args, {"": get_time_taken_table(dfdrawnlines)}, outputfile)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_legend(parser)
    addarg_darkmode(parser)
    addarg_modelpath(
        parser, positional=True, multiplepaths=True, default=[], helptext="Paths of the ARTIS models with log files"
    )
    addarg_output(
        parser,
        kind="file",
        defaultname=defaultoutputfile,
        helptext="Filename for the PDF file. The field {modelname} gives the name of each model",
    )
    addarg_show(parser)
    addarg_data_output(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot durations from log files."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpaths = normalize_path_list(args.modelpath)
    outputfile = args.outputfile
    if len(modelpaths) > 1 and "{" not in str(outputfile):
        exit_with_error(
            f"The output file {outputfile} has no field, thus each model would replace the file of the model before it",
            "Give a name with the field {modelname}, e.g. -o plotlogfiles_{modelname}.pdf",
        )

    for modelpath in modelpaths:
        modelname = get_model_name(modelpath)
        logfiledict = read_time_taken(get_logfiles(Path(modelpath), None))
        make_plot(
            logfiledict,
            # {0} is the field of the earlier default name
            outputfile=format_frame_path(outputfile, modelname, modelname=modelname),
            modelname=modelname,
            modellogname=get_model_logname(modelpath),
            args=args,
        )
