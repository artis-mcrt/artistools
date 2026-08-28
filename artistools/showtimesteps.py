# PYTHON_ARGCOMPLETE_OK
"""List the timesteps of an ARTIS model and the days that each one covers."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

from artistools.misc import addarg_modelpath
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import get_model_name
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import parse_range_list
from artistools.misc import print_product
from artistools.misc.timesteps import get_bad_timestep_message


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


def get_timestep_days(modelpath: Path | str, timestep: int) -> str:
    """Return the days that one timestep covers, e.g. "299.812 to 300.823 days"."""
    tstarts = get_timestep_times(modelpath, loc="start")
    tends = get_timestep_times(modelpath, loc="end")

    return f"{tstarts[timestep]:.3f} to {tends[timestep]:.3f} days"


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())
    addarg_timedays(parser, kind="float", helptext="Name the timestep that covers this time in days")
    addarg_timestep(parser, kind="rangestr", helptext="Give the days that this timestep covers, e.g. 40 or last")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """List the timesteps of an ARTIS model and the days that each one covers."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.timedays is not None:
        timestep = get_timestep_of_timedays(args.modelpath, args.timedays)
        days = get_timestep_days(args.modelpath, timestep)
        print_product(args, f"{args.timedays:g} days falls in timestep {timestep}, which covers {days}")
    elif args.timestep is not None:
        lasttimestep = len(get_timestep_times(args.modelpath, loc="mid")) - 1
        # -timestep takes a range on every command, and a caller of the API can give a number
        timesteps = parse_range_list(str(args.timestep), dictvars={"last": lasttimestep})
        for timestep in timesteps:
            if not 0 <= timestep <= lasttimestep:
                msg = get_bad_timestep_message(args.modelpath, timestep)
                raise ValueError(msg)

            print_product(args, f"timestep {timestep} covers {get_timestep_days(args.modelpath, timestep)}")
    else:
        print_product(args, get_timesteps_table(args.modelpath))


if __name__ == "__main__":
    from artistools.commands import run_subcommand

    run_subcommand("timesteps")
