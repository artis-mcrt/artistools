import argparse
import re
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

import artistools as at

defaultoutputfile = "plotlogfiles_{0}.pdf"


def read_logfiles(modelpath: Path | str) -> list[Path]:
    """Return the per-rank ARTIS log files of a model, including compressed ones."""
    return [
        logfilepath
        for folderpath in at.get_runfolders(modelpath)
        for mpirank in at.get_mpiranklist(modelpath)
        if (logfilepath := at.firstexisting_or_none([f"output_{mpirank}-0.txt"], folder=folderpath)) is not None
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

    # only timesteps recorded for every stage can be plotted together
    timesteps = sorted(set.intersection(*(set(bytimestep) for bytimestep in logfiledict.values())))
    if not timesteps:
        print(f"No timing data found in the log files of {modelname}")
        return

    with PdfPages(outputfile) as pdf:
        for timestep in timesteps:
            fig, axis = plt.subplots()
            for stage, bytimestep in logfiledict.items():
                mpirank, timetaken = zip(*sorted(bytimestep[timestep].items()), strict=True)
                axis.plot(mpirank, timetaken, label=stage)
            axis.set_xlabel("mpi rank")
            axis.set_ylabel("time (s)")
            axis.set_title(f"{modelname} timestep {timestep}" if modelname else f"timestep {timestep}")
            axis.legend()
            pdf.savefig(fig)
            plt.close(fig)

    at.print_saved(outputfile)


def addargs(parser: argparse.ArgumentParser) -> None:
    at.add_modelpath_arg(
        parser, multiplepaths=True, default=[], helptext="Path to ARTIS model folders with model.txt and abundances.txt"
    )
    at.add_outputfile_arg(parser, default=defaultoutputfile, astype=None, helptext="Filename for PDF file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot durations from log files."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    for modelpath in at.normalize_path_list(args.modelpath):
        modelname = at.get_model_name(modelpath)
        logfiledict = read_time_taken(read_logfiles(modelpath))
        outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)
        make_plot(logfiledict, outputfile=str(outputfile).format(modelname), modelname=modelname)


if __name__ == "__main__":
    main()
