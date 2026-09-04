# PYTHON_ARGCOMPLETE_OK
"""Give the deposition rate of a model per unit volume, per ion, and per unit mass."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.constants import EV_to_erg
from artistools.estimators.estimators import scan_estimators
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import addarg_verbose
from artistools.misc import get_model_name
from artistools.misc import get_time_range
from artistools.misc import get_timestep_times
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_product
from artistools.misc import print_saved
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.misc.cliutils import CommaJoinAction

DEPOSITIONPREFIX = "deposition_"
DEFAULTOUTPUTNAME = "deposition.txt"


def get_deposition_channels(colnames: Sequence[str]) -> list[str]:
    """Return the deposition channels that the estimators of a model give, in alphabetical order.

    The reader of the estimator files keeps the columns in a hash map, thus the order of the columns
    changes between runs. The sort gives one order to the listing and to each message.
    """
    return sorted(name.removeprefix(DEPOSITIONPREFIX) for name in colnames if name.startswith(DEPOSITIONPREFIX))


def get_deposition_expression(colnames: Sequence[str], channels: Sequence[str] | None) -> pl.Expr:
    """Return the expression that gives the deposition rate of one cell in erg/s/cm3.

    The command needs the rate of each cell, which the deposition_ columns hold. An older run holds
    total_dep alone, and artistools derives that column from the heating rate. On the classic-mode
    test model that total is 1400 times below the tally of deposition.out, thus such a model gets an
    error and no number.
    """
    available = get_deposition_channels(colnames)
    if not available:
        msg = (
            "The estimators of this model hold no deposition_ column, thus they give no deposition "
            "rate of a cell. The file deposition.out holds the rate of the whole model, which "
            "artistools plotlightcurves --plotdeposition draws"
        )
        raise ValueError(msg)

    channels = [channel.removeprefix(DEPOSITIONPREFIX) for channel in channels] if channels else available
    if unknown := [channel for channel in channels if channel not in available]:
        msg = f"This model gives no deposition rate of {' and '.join(unknown)}. It holds {', '.join(available)}"
        raise ValueError(msg)

    # ignore_nulls=False keeps a null, thus a channel that one rank left out gives no partial total
    return pl.sum_horizontal((pl.col(f"{DEPOSITIONPREFIX}{channel}") for channel in channels), ignore_nulls=False)


def aggregate_deposition_rates(
    dfestim: pl.LazyFrame, timesteps: Sequence[int] | None = None, channels: Sequence[str] | None = None
) -> pl.DataFrame:
    """Return the deposition rate per unit volume, per ion, and per unit mass of each timestep.

    ARTIS writes the deposition rate of a timestep into the estimators of the next timestep. It
    divides that rate by the volume of the timestep that took the energy. Thus the rate of timestep n
    comes from the rows of timestep n + 1 and from the column volume_prevtimestep. The last timestep
    gets no row, because only a later timestep holds that rate.

    The join on the cell gives one row of one timestep to each cell. Thus the rate, the ion count,
    the volume, and the mass of a row cover one set of cells.
    """
    colnames = dfestim.collect_schema().names()
    if "nntot" not in colnames:
        msg = (
            "The estimators of this model hold no number density of the elements, "
            "thus the command cannot count the ions"
        )
        raise ValueError(msg)

    dfrates = dfestim.select(
        timestep=pl.col("timestep") - 1,
        modelgridindex=pl.col("modelgridindex"),
        dep_erg_per_s=get_deposition_expression(colnames, channels) * pl.col("volume_prevtimestep"),
    )

    # only a cell that holds matter has an ion. The filter also names is_finite, because a NaN
    # number density passes the comparison and gives no count
    dfcells = dfestim.filter(pl.col("nntot") > 0, pl.col("nntot").is_finite()).select(
        "timestep", "modelgridindex", "tmid_days", "mass_g", "volume", ioncount=pl.col("nntot") * pl.col("volume")
    )
    if timesteps is not None:
        dfcells = dfcells.filter(pl.col("timestep").is_in(list(timesteps)))

    return (
        dfcells
        .join(dfrates, on=("timestep", "modelgridindex"), how="inner")
        .group_by("timestep")
        .agg(
            tmid_days=pl.col("tmid_days").first(),
            dep_erg_per_s=pl.col("dep_erg_per_s").sum(),
            ioncount=pl.col("ioncount").sum(),
            volume=pl.col("volume").sum(),
            mass_g=pl.col("mass_g").sum(),
            # the sum leaves a null out, thus the count of the cells that gave no rate goes beside it
            cellswithnorate=pl.col("dep_erg_per_s").is_finite().fill_null(value=False).not_().sum(),
        )
        .with_columns(
            # a rate in erg divided by erg per eV gives the rate in eV
            dep_per_volume=pl.col("dep_erg_per_s") / EV_to_erg / pl.col("volume"),
            dep_per_ion=pl.col("dep_erg_per_s") / EV_to_erg / pl.col("ioncount"),
            dep_per_mass=pl.col("dep_erg_per_s") / EV_to_erg / pl.col("mass_g"),
        )
        .select(
            "timestep",
            # the table prints a number, thus a null arrives there as a NaN
            *(
                pl.col(name).fill_null(float("nan"))
                for name in ("tmid_days", "dep_per_volume", "dep_per_ion", "dep_per_mass")
            ),
            "cellswithnorate",
        )
        .sort("timestep")
        .collect(engine="streaming")
    )


def get_deposition_rates(
    modelpath: Path | str,
    timesteps: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    verbose: bool = False,
) -> pl.DataFrame:
    """Read the estimators of a model, and return the deposition rate of each timestep.

    A value of None for timesteps reads every timestep of the model.
    """
    readtimesteps = None if timesteps is None else tuple({*timesteps} | {timestep + 1 for timestep in timesteps})
    dfestim = scan_estimators(modelpath, timestep=readtimesteps, join_modeldata=True, verbose=verbose)

    return aggregate_deposition_rates(dfestim, timesteps, channels)


def get_selected_timesteps(modelpath: Path | str, timedays: str | None, timestep: str | None) -> list[int] | None:
    """Return the timesteps that -timedays or -timestep selects, or None if the caller gives no time.

    Each argument also takes a list, e.g. 1,5,30. get_time_range reads one item of that list. Thus a
    time, a range of times, a timestep, a range of timesteps, and "last" mean here what they mean on
    every other command.
    """
    if timedays is timestep is None:
        return None

    selected: set[int] = set()
    for token in str(timestep if timedays is None else timedays).split(","):
        timestepmin, timestepmax, _, _ = (
            get_time_range(modelpath, timestep_range_str=token)
            if timedays is None
            else get_time_range(modelpath, timedays_range_str=token)
        )
        selected.update(range(timestepmin, timestepmax + 1))

    return sorted(selected)


def format_channel_list(modelpath: Path | str, verbose: bool = False) -> str:
    """Return the deposition channels that the estimators of a model hold.

    The scan reads the files of one cell, thus the listing costs much less than the table.
    """
    channels = get_deposition_channels(
        scan_estimators(modelpath, modelgridindex=0, verbose=verbose).collect_schema().names()
    )

    return f"{get_model_name(modelpath)} holds {', '.join(channels) if channels else 'no deposition channel'}"


def format_deposition_table(modelpath: Path | str, dftable: pl.DataFrame, channels: Sequence[str] | None) -> str:
    """Return the deposition rates of one model as a table of one row for each timestep."""
    lines = [
        f"{get_model_name(modelpath)}: deposition of {', '.join(channels) if channels else 'every channel'}",
        f"{'timestep':>8s} {'t_days':>10s} {'dep_per_volume':>15s} {'dep_per_ion':>15s} {'dep_per_mass':>15s}",
        f"{'':>8s} {'[d]':>10s} {'[eV/s/cm^3]':>15s} {'[eV/s]':>15s} {'[eV/s/g]':>15s}",
    ]
    lines.extend(
        f"{timestep:8d} {tmid_days:10.3f} {dep_per_volume:15.3e} {dep_per_ion:15.3e} {dep_per_mass:15.3e}"
        for timestep, tmid_days, dep_per_volume, dep_per_ion, dep_per_mass in dftable.select(
            "timestep", "tmid_days", "dep_per_volume", "dep_per_ion", "dep_per_mass"
        ).iter_rows()
    )

    return "\n".join(lines)


def warn_about_gaps(
    modelpath: Path | str, dftable: pl.DataFrame, timesteps: Sequence[int] | None, lasttimestep: int
) -> None:
    """Print a warning for each timestep that got no row, and for the cells that gave no rate."""
    modelname = get_model_name(modelpath)
    gotrows = set(dftable["timestep"].to_list())
    # a run gives no rate for its last timestep, thus the default selection loses that one row
    wanted = set(timesteps) if timesteps is not None else {lasttimestep}
    if missing := sorted(wanted - gotrows):
        print_warning(
            f"{modelname}: no deposition rate for timestep {', '.join(str(timestep) for timestep in missing)}. "
            "The rate of a timestep arrives in the file of the next timestep, "
            "thus the last timestep of a run gives none"
        )

    if cellswithnorate := int(dftable["cellswithnorate"].sum()):
        print_warning(f"{modelname}: {cellswithnorate} cells gave no deposition rate, thus a row can hold a NaN")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[], helptext="Paths to ARTIS model folders")
    addarg_timedays(
        parser, kind="str", helptext="Times in days, e.g. 30, a range 10-30, or a list 1,5,30. Default: every timestep"
    )
    addarg_timestep(parser, helptext="Timesteps, e.g. 40, a range 45-65, a list 4,9, or last")
    parser.add_argument(
        "-channels",
        "-channel",
        dest="channels",
        action=CommaJoinAction,
        help=(
            "Deposition channels to add, e.g. gamma,electron,alpha. "
            "Default: every channel of the model. Give --listchannels to show the channels"
        ),
    )
    parser.add_argument(
        "--listchannels", action="store_true", help="Show the deposition channels of the model. The command then stops"
    )
    addarg_output(parser, kind="file", helptext="Path/filename for the output text file")
    addarg_verbose(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Give the deposition rate of a model per unit volume, per ion, and per unit mass."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    channels = str(args.channels).split(",") if args.channels else None
    tables: list[str] = []
    for modelpath in normalize_path_list(args.modelpath):
        if args.listchannels:
            tables.append(format_channel_list(modelpath, verbose=args.verbose))
        else:
            timesteps = get_selected_timesteps(modelpath, args.timedays, args.timestep)
            dftable = get_deposition_rates(modelpath, timesteps, channels, verbose=args.verbose)
            warn_about_gaps(modelpath, dftable, timesteps, len(get_timestep_times(modelpath, loc="mid")) - 1)
            tables.append(format_deposition_table(modelpath, dftable, channels))

        print_product(args, tables[-1])

    if args.outputfile:
        outputfile = resolve_outputfile(args.outputfile, DEFAULTOUTPUTNAME)
        outputfile.write_text("\n\n".join(tables) + "\n", encoding="utf-8")
        print_saved(outputfile)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
