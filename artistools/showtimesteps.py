# PYTHON_ARGCOMPLETE_OK
"""List the timesteps of an ARTIS model and the days that each one covers."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

from artistools.misc import addarg_modelpath
from artistools.misc import addarg_quiet
from artistools.misc import get_model_name
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import print_product


def get_timesteps_table(modelpath: Path | str) -> str:
    """Return the timesteps of a model as a table, one row for each timestep."""
    tstarts = get_timestep_times(modelpath, loc="start")
    tmids = get_timestep_times(modelpath, loc="mid")
    tends = get_timestep_times(modelpath, loc="end")

    lines = [
        f"{get_model_name(modelpath)}: {len(tmids)} timesteps from {tstarts[0]:.3f} to {tends[-1]:.3f} days",
        f"{'timestep':>8s} {'start_days':>11s} {'mid_days':>11s} {'end_days':>11s} {'width_days':>11s}",
    ]
    lines.extend(
        f"{timestep:8d} {tstart:11.3f} {tmid:11.3f} {tend:11.3f} {tend - tstart:11.3f}"
        for timestep, (tstart, tmid, tend) in enumerate(zip(tstarts, tmids, tends, strict=True))
    )
    lines.append(
        f'Select one with e.g. "-timestep {len(tmids) // 2}" or "-timedays {tmids[len(tmids) // 2]:.0f}", '
        f'a range with "-timestep {len(tmids) // 4}-{len(tmids) // 2}", and "last" names timestep {len(tmids) - 1}'
    )

    return "\n".join(lines)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())
    addarg_quiet(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """List the timesteps of an ARTIS model and the days that each one covers."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    print_product(args, get_timesteps_table(args.modelpath))


if __name__ == "__main__":
    from artistools.commands import run_subcommand

    run_subcommand("timesteps")
