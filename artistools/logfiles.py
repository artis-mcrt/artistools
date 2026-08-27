"""Plot per-rank stage durations from ARTIS log files."""

import argparse
import re
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

import artistools as at
from artistools.misc import print_warning

defaultoutputfile = "plotlogfiles_{0}.pdf"


def read_logfiles(modelpath: Path | str) -> list[Path]:
    """Return the per-rank ARTIS log files of a model, including compressed ones."""
    mpiranklist = at.get_mpiranklist(modelpath)
    # search_subfolders=False so a rank file missing from one run folder is skipped
    # rather than silently substituted by another folder's copy
    return [
        logfilepath
        for folderpath in at.get_runfolders(modelpath)
        for mpirank in mpiranklist
        if (
            logfilepath := at.firstexisting_or_none(
                [f"output_{mpirank}-0.txt"], folder=folderpath, search_subfolders=False
            )
        )
        is not None
    ]


# "timestep 3: time after update grid for all processes 1699974033 (rank 0 took 1s, waited 0s, total 1s)"
_re_stage_perrank = re.compile(
    r"timestep (?P<timestep>\d+): time after (?P<stage>update grid|update packets) for all processes \d+ "
    r"\(rank (?P<rank>\d+) took (?P<seconds>\d+)s"
)

# "timestep 3: time after estimators have been communicated 1699974033 (took 0 seconds)"
_re_estimators = re.compile(
    r"timestep (?P<timestep>\d+): time after estimators have been communicated \d+ \(took (?P<seconds>\d+) seconds\)"
)

_stagekey = {"update grid": "update_grid", "update packets": "update_packets"}


def read_time_taken(logfilepaths: Iterable[Path | str]) -> dict[str, dict[int, dict[int, int]]]:
    """Return {stage: {timestep: {mpi rank: seconds taken}}} parsed from ARTIS log files."""
    timetaken: dict[str, dict[int, dict[int, int]]] = {"update_grid": {}, "update_packets": {}, "write_estimators": {}}

    for logfilepath in logfilepaths:
        # the rank that wrote the file, e.g. output_12-0.txt -> 12
        filerank = int(Path(logfilepath).name.split("-")[0].split("_")[-1])
        with at.zopen(logfilepath, encoding="utf-8") as logfile:
            for line in logfile:
                if "took" not in line:
                    continue
                if match := _re_stage_perrank.search(line):
                    stage = _stagekey[match["stage"]]
                    rank = int(match["rank"])
                elif match := _re_estimators.search(line):
                    stage = "write_estimators"
                    rank = filerank
                else:
                    continue
                timetaken[stage].setdefault(int(match["timestep"]), {}).setdefault(rank, int(match["seconds"]))

    return timetaken


def make_plot(logfiledict: dict[str, dict[int, dict[int, int]]], outputfile: Path | str, modelname: str = "") -> None:
    """Write one page per timestep of stage duration versus mpi rank to a multi-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages

    # plot every timestep that has data for at least one stage, so a log format
    # missing one stage entirely still yields plots of the others
    timesteps = sorted(set().union(*(set(bytimestep) for bytimestep in logfiledict.values())))
    if not timesteps:
        print(f"No timing data found in the log files of {modelname}")
        return

    with PdfPages(outputfile) as pdf:
        for timestep in timesteps:
            fig, axis = plt.subplots()
            for stage, bytimestep in logfiledict.items():
                if timestep not in bytimestep:
                    continue
                mpirank, timetaken = zip(*sorted(bytimestep[timestep].items()), strict=True)
                axis.plot(mpirank, timetaken, label=stage)
            axis.set_xlabel("mpi rank")
            axis.set_ylabel("Time [s]")
            axis.set_title(f"{modelname} timestep {timestep}" if modelname else f"timestep {timestep}")
            axis.legend()
            pdf.savefig(fig)
            plt.close(fig)

    at.print_saved(outputfile)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    at.addarg_modelpath(
        parser, multiplepaths=True, default=[], helptext="Path to ARTIS model folders with model.txt and abundances.txt"
    )
    at.addarg_outputfile(parser, default=defaultoutputfile, astype=None, helptext="Filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot durations from log files."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpaths = at.normalize_path_list(args.modelpath)
    outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)
    if len(modelpaths) > 1 and "{" not in str(outputfile):
        print_warning(f"output filename {outputfile} has no {{0}} placeholder, so each model will overwrite it")

    for modelpath in modelpaths:
        modelname = at.get_model_name(modelpath)
        logfiledict = read_time_taken(read_logfiles(modelpath))
        make_plot(logfiledict, outputfile=str(outputfile).format(modelname), modelname=modelname)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
