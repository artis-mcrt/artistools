"""Sum the core hours of the runs of ARTIS models from the rank-0 logs of sn3d.

The last line of the log of a finished run gives its core hours. A run that stops early has no such line, thus the
command estimates its core hours from the first and the last timestamp of the log, and from the numbers of processes
and threads that the log gives. The command reads only the sn3d logs, and not the output of the job scripts.
"""

import argparse
import json
import typing as t
from collections.abc import Sequence
from datetime import datetime
from itertools import starmap
from pathlib import Path

from artistools.logsummary import get_unique_logfiles
from artistools.misc import addarg_modelpath
from artistools.misc import firstexisting_or_none
from artistools.misc import get_model_folder
from artistools.misc import get_run_subfolders
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_product
from artistools.misc import print_warning
from artistools.misc import zopen
from artistools.misc.fileio import get_decompress_errors


class RunCoreHours(t.NamedTuple):
    """The core hours of one run and the values of its log that the summary of the runs needs."""

    core_hours: float | None
    isestimate: bool
    wallclock_hours: float | None
    ntasks: int | None
    timestep_range: tuple[int, int] | None
    estimate_text: str | None
    first_line: str
    last_line: str


def get_job_procs_threads(loglines: Sequence[str]) -> tuple[int, int] | None:
    """Return the number of processes and the number of threads from the lines of an sn3d log."""
    nprocs: int | None = None
    nthreads = 1
    for line in loglines:
        if "with nprocs " in line:
            nprocs = int(line.split("with nprocs ")[1].split()[0].rstrip(")"))
        elif "OpenMP parallelisation is active" in line:
            nthreads = int(line.split("(max ")[1].split(")")[0])
        elif "stdpar" in line and " CPU threads" in line:
            nthreads = int(line.split(" CPU threads")[0].split()[-1])
    return None if nprocs is None else (nprocs, nthreads)


def parse_log_time(line: str) -> datetime | None:
    """Return the timestamp of a log line, or None for a line without one."""
    try:
        return datetime.strptime(line.split(maxsplit=1)[0], "%Y-%m-%dT%H:%M:%S%z")
    except (ValueError, IndexError):
        return None


def get_log_elapsed_hours(loglines: Sequence[str]) -> float | None:
    """Return the hours between the first and the last timestamp of an sn3d log."""
    # an assertion diagnostic has no timestamp, thus the search finds the nearest line with one
    time_start = next((time for line in loglines if (time := parse_log_time(line)) is not None), None)
    time_end = next((time for line in reversed(loglines) if (time := parse_log_time(line)) is not None), None)
    if time_start is None or time_end is None:
        return None
    return (time_end - time_start).total_seconds() / 3600.0


def get_timestep_range(last_line: str) -> tuple[int, int] | None:
    """Return the first and the last timestep of the run from the last line of its log."""
    if "pktprop ts " not in last_line:
        return None
    # example: "... (job: pktprop ts 0 to ts 9 grid-preprop, 45.909 wallclock hours ..."
    tokens = last_line.split("pktprop ts ")[1].split()
    return (int(tokens[0]), int(tokens[3]))


def get_log_timestep_range(loglines: Sequence[str]) -> tuple[int, int] | None:
    """Return the first and the last propagated timestep from the lines of an sn3d log."""
    ts_first: int | None = None
    ts_last: int | None = None
    for line in loglines:
        # example: "2026-08-20T21:31:02Z timestep 21: start update_packets"
        if ts_first is None and ": start update_packets" in line and " timestep " in line:
            ts_first = int(line.split(" timestep ")[1].split(":")[0])
        # example: "2026-08-22T03:26:29Z timestep 63: time after update grid on all processes (...)"
        if ": time after update grid" in line and " timestep " in line:
            ts_last = int(line.split(" timestep ")[1].split(":")[0])
    if ts_first is None or ts_last is None:
        return None
    return (ts_first, ts_last)


def get_run_core_hours(loglines: Sequence[str]) -> RunCoreHours:
    """Return the core hours of a run from the lines of its sn3d log, or an estimate for a run that stopped early."""
    last_line = loglines[-1].strip()
    core_hours: float | None = None
    wallclock_hours: float | None = None
    ntasks: int | None = None
    isestimate = False
    estimate_text: str | None = None
    log_timestep_range: tuple[int, int] | None = None
    if "CPU hours" in last_line:
        core_hours = float(last_line.split("CPU hours")[0].split()[-1])
        # the old last line gives no wallclock hours and no number of tasks, thus the log body gives the tasks
        procs_threads = get_job_procs_threads(loglines)
        if procs_threads is not None:
            ntasks = procs_threads[0] * procs_threads[1]
    elif "core hours" in last_line:
        core_hours = float(last_line.split("core hours")[0].split()[-1])
        if " wallclock hours" in last_line:
            wallclock_hours = float(last_line.split(" wallclock hours")[0].split()[-1])
        if " processes " in last_line and " threads " in last_line:
            ntasks = int(last_line.split(" processes")[0].split()[-1]) * int(last_line.split(" threads")[0].split()[-1])
    else:
        procs_threads = get_job_procs_threads(loglines)
        elapsed_hours = get_log_elapsed_hours(loglines)
        log_timestep_range = get_log_timestep_range(loglines)
        if procs_threads is not None and elapsed_hours is not None:
            nprocs, nthreads = procs_threads
            core_hours = elapsed_hours * nprocs * nthreads
            wallclock_hours = elapsed_hours
            ntasks = nprocs * nthreads
            isestimate = True
            if log_timestep_range is not None:
                estimate_text = (
                    f"WARNING: sn3d did not finish. Estimated: "
                    f"(job: pktprop ts {log_timestep_range[0]} to ts {log_timestep_range[1]} grid-preprop, "
                    f"{elapsed_hours:.3f} wallclock hours * {nprocs} processes * {nthreads} threads "
                    f"= {core_hours:.3f} core hours)"
                )

    return RunCoreHours(
        core_hours=core_hours,
        isestimate=isestimate,
        wallclock_hours=wallclock_hours,
        ntasks=ntasks,
        timestep_range=get_timestep_range(last_line) or log_timestep_range,
        estimate_text=estimate_text,
        first_line=loglines[0].strip(),
        last_line=last_line,
    )


def read_run_core_hours(logfile: Path) -> RunCoreHours | None:
    """Return the core hours of the run of an sn3d log, or None for an empty log or a damaged log."""
    try:
        with zopen(logfile) as flog:
            loglines = flog.readlines()
    except (OSError, EOFError, UnicodeDecodeError, *get_decompress_errors()) as exc:
        # the other runs keep their values, and the table shows the damaged log
        print_warning(f"{logfile} is unreadable: {exc}")
        return None
    return get_run_core_hours(loglines) if loglines else None


def get_rank0_logfiles(modelfolders: Sequence[Path]) -> list[tuple[Path, bool]]:
    """Return the rank-0 log of each run folder of the model folders in the order of the paths, with True for a copy.

    The model folder holds a link or a copy of the rank-0 log of its last run. get_unique_logfiles finds that copy, and
    the table shows it as a duplicate, thus a user sees that the sum counts that run one time.
    """
    logfiles_with_iscopy: list[tuple[Path, bool]] = []
    for modelfolder in modelfolders:
        logfiles = [
            logfile
            for folder in get_run_subfolders(modelfolder)
            if (logfile := firstexisting_or_none("output_0-0.txt", folder=folder, search_subfolders=False)) is not None
        ]
        uniquelogfiles = set(get_unique_logfiles(modelfolder, logfiles))
        logfiles_with_iscopy.extend((logfile, logfile not in uniquelogfiles) for logfile in logfiles)
    return sorted(logfiles_with_iscopy)


def format_run_lines(logfile: Path, col1width: int, iscopy: bool, run: RunCoreHours | None) -> list[str]:
    """Return the lines of the table for the log of one run."""
    logfilecolumn = f"{str(logfile) + ':':{col1width}s} "
    if iscopy:
        return [f"{logfilecolumn}\n  ignored duplicate log", ""]
    if run is None:
        return [f"{logfilecolumn}  WARNING: the log is empty or unreadable.", ""]
    lines: list[str] = []
    if run.core_hours is not None:
        lines.append(f"{logfilecolumn}{run.core_hours:8.1f} core-h")
        if run.isestimate and run.estimate_text is None:
            lines.append("  WARNING: sn3d did not finish cleanly. The value is an estimate from the log timestamps.")
    else:
        lines.append(f"{logfilecolumn}  WARNING: sn3d did not finish cleanly. Check the log to get the CPU time.")
    lines.extend((f"  {run.first_line}", f"  {run.last_line}"))
    if run.estimate_text is not None:
        lines.append(f"  {run.estimate_text}")
    lines.append("")
    return lines


def get_run_json(logfile: Path, iscopy: bool, run: RunCoreHours | None) -> dict[str, str | float | int | bool | None]:
    """Return the values of one log for the JSON output."""
    runjson: dict[str, str | float | int | bool | None] = {
        "logfile": str(logfile),
        "core_hours": None if run is None else run.core_hours,
        "estimate": run is not None and run.isestimate,
        "duplicate": iscopy,
    }
    if run is not None and run.timestep_range is not None:
        runjson["timestep_first"], runjson["timestep_last"] = run.timestep_range
    return runjson


def get_summary_lines(
    ntasks_seen: set[int],
    ntasks: int | None,
    wallclock_hours: float | None,
    total_core_hours: float,
    has_runs_without_core_hours: bool,
) -> list[str]:
    """Return the lines of the summary after the table."""
    lines: list[str] = []
    if has_runs_without_core_hours:
        lines.append("WARNING: some logs give no core hours. The CPU time omits their runs.")
    if len(ntasks_seen) > 1:
        lines.append("WARNING: the jobs use different task counts. The summary omits the task count.")
    if ntasks is not None:
        lines.append(f"{'Tasks:':15s} {ntasks:8d}")
    if wallclock_hours is not None:
        lines.append(f"{'Wallclock time:':15s} {wallclock_hours:12.3f}  hours")
    lines.extend((
        f"{'CPU time:':15s} {total_core_hours:12.3f}  core-h",
        f"{'CPU time:':15s} {total_core_hours / 1000:12.3f}  k core-h",
    ))
    return lines


def print_core_hours(args: argparse.Namespace, logfiles_with_iscopy: Sequence[tuple[Path, bool]]) -> None:
    """Print the core hours of each run and the sums, as a table or as JSON."""
    col1width = max((len(str(logfile)) for logfile, _ in logfiles_with_iscopy), default=0) + 1
    runs: list[tuple[Path, bool, RunCoreHours | None]] = []
    for logfile, iscopy in logfiles_with_iscopy:
        run = None if iscopy else read_run_core_hours(logfile)
        runs.append((logfile, iscopy, run))
        if not args.json:
            print_product(args, "\n".join(format_run_lines(logfile, col1width, iscopy, run)))

    countedruns = [run for _, iscopy, run in runs if not iscopy]
    # the plain addition gives the JSON values of the ARTIS script sumcorehourslogs.py, and sum() changes the last digit
    total_core_hours = 0.0
    wallclock_hours_sum = 0.0
    for run in countedruns:
        if run is not None and run.core_hours is not None:
            total_core_hours += run.core_hours
        if run is not None and run.wallclock_hours is not None:
            wallclock_hours_sum += run.wallclock_hours
    # an empty log or a run with no time value makes the sum of the wallclock hours incomplete
    iswallclockcomplete = all(run is not None and run.wallclock_hours is not None for run in countedruns)
    wallclock_hours = wallclock_hours_sum if iswallclockcomplete and total_core_hours > 0.0 else None
    has_runs_without_core_hours = any(run is None or run.core_hours is None for run in countedruns)
    ntasks_of_runs = [None if run is None else run.ntasks for run in countedruns]
    ntasks_seen = {ntasks for ntasks in ntasks_of_runs if ntasks is not None}
    # a run with no task count can have a different count, thus each run must give the same count
    ntasks = ntasks_of_runs[0] if None not in ntasks_of_runs and len(ntasks_seen) == 1 else None
    timestep_ranges = sorted(
        run.timestep_range for run in countedruns if run is not None and run.timestep_range is not None
    )

    if args.json:
        print_product(
            args,
            json.dumps(
                {
                    "jobs": list(starmap(get_run_json, runs)),
                    "tasks": ntasks,
                    "wallclock_hours": wallclock_hours,
                    "total_core_hours": None if has_runs_without_core_hours else total_core_hours,
                    "timestep_ranges": [list(timestep_range) for timestep_range in timestep_ranges],
                },
                indent=2,
            ),
        )
        return

    print_product(
        args,
        "\n".join(
            get_summary_lines(ntasks_seen, ntasks, wallclock_hours, total_core_hours, has_runs_without_core_hours)
        ),
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(
        parser,
        positional=True,
        multiplepaths=True,
        helptext="The model folders, or files in them. The default is the working folder",
    )
    parser.add_argument("--json", action="store_true", help="Write the results as JSON in place of a table")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Sum the core hours of the runs of ARTIS models from the rank-0 logs of sn3d."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelfolders = [get_model_folder(modelpath) for modelpath in normalize_path_list(args.modelpath)]
    print_core_hours(args, get_rank0_logfiles(modelfolders))
