"""Timestep definitions, time range selection, and deposition rates."""

import argparse
import contextlib
import math
import re
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from artistools.constants import C_cm_per_s
from artistools.misc.cliutils import exit_with_error
from artistools.misc.cliutils import parse_float_range
from artistools.misc.cliutils import print_warning
from artistools.misc.fileio import firstexisting
from artistools.misc.fileio import firstexisting_or_none
from artistools.misc.fileio import get_model_folder
from artistools.misc.fileio import path_is_artis_model
from artistools.misc.fileio import path_is_codecomparison
from artistools.misc.fileio import polars_source_open
from artistools.misc.fileio import read_wsv
from artistools.misc.modelinfo import get_inputparams
from artistools.misc.modelinfo import get_model_name


def match_closest_time(reftime: float, searchtimes: Iterable[t.Any]) -> float:
    """Return the time in searchtimes that is closest to reftime."""

    def offset_from_reftime(time: float) -> float:
        return abs(time - reftime)

    return min((float(x) for x in searchtimes), key=offset_from_reftime)


@lru_cache(maxsize=16)
def get_deposition(modelpath: Path | str = ".") -> pl.LazyFrame:
    """Return a polars LazyFrame containing the deposition data.

    The file is read and its times checked against the timesteps on every call, and a light curve plot asks
    for it once per model and again per escape type, so the parsed frame is cached. A LazyFrame has no
    in-place operations, so a caller cannot alter what the next one gets.
    """
    if Path(modelpath).is_file():
        depfilepath = Path(modelpath)
        modelpath = Path(modelpath).parent
    else:
        # read through firstexisting/zopen so that a compressed deposition.out is found like every other
        # ARTIS output file, instead of being reported as missing
        depfilepath = firstexisting("deposition.out", folder=modelpath, tryzipped=True, search_subfolders=False)

    ts_mids = get_timestep_times(modelpath, loc="mid")

    # read_wsv takes the column names from the header comment of the file, and keeps the names below
    # when the file has no such comment
    depdata = read_wsv(
        depfilepath,
        has_header=False,
        comment_prefix="#",
        header_from_comment=True,
        new_columns=["tmid_days", "gammadep_Lsun", "positrondep_Lsun", "total_dep_Lsun"],
    ).lazy()

    if "ts" in depdata.collect_schema().names():
        depdata = depdata.rename({"ts": "timestep"})

    if "timestep" not in depdata.collect_schema().names():
        depdata = depdata.with_row_index("timestep", offset=0)

    depdata = depdata.with_columns(timestep=pl.col("timestep").cast(pl.Int32))

    # no timesteps are given in the old format of deposition.out, so ensure that
    # the times in days match up with the times of our assumed timesteps
    t_mid_days = depdata.select("tmid_days").collect().to_series().to_numpy()
    # a file with more rows than the model has timesteps is a mismatch. The slice comparison
    # below raises a broadcast error for such a file instead of the message
    if len(t_mid_days) > len(ts_mids) or not np.allclose(t_mid_days, ts_mids[: len(t_mid_days)], rtol=0.01):
        msg = "Deposition times do not match the timesteps"
        raise AssertionError(msg)

    return depdata


def get_timesteps(modelpath: Path | str) -> pl.LazyFrame:
    """Return a LazyFrame containing the timestep indices, starts, mids, ends, deltas."""
    modelpath = Path(modelpath)
    # virtual path to code comparison workshop models
    if path_is_codecomparison(modelpath):
        from artistools.codecomparison import get_timestep_times as cc_get_times

        return (
            pl
            .LazyFrame({
                "tmid_days": cc_get_times(modelpath=modelpath, loc="mid"),
                "tstart_days": cc_get_times(modelpath=modelpath, loc="start"),
                "tend_days": cc_get_times(modelpath=modelpath, loc="end"),
                "twidth_days": cc_get_times(modelpath=modelpath, loc="delta"),
            })
            .with_row_index("timestep", offset=0)
            .with_columns(pl.col("timestep").cast(pl.Int32))
        )

    # use timesteps.out if possible (allowing arbitrary timestep lengths), compressed or not, so that a run
    # folder with compressed output does not silently fall back to reconstructing logarithmic timesteps
    tsfilepath = firstexisting_or_none("timesteps.out", folder=modelpath, tryzipped=True, search_subfolders=False)
    if tsfilepath is not None:
        with polars_source_open(tsfilepath) as source:
            return (
                pl
                .scan_csv(source, has_header=True, separator=" ")
                .rename(lambda column_name: column_name.removeprefix("#"))
                .with_columns(tend_days=pl.col("tstart_days") + pl.col("twidth_days"))
            )

    # older versions of Artis always used logarithmic timesteps and didn't produce a timesteps.out file
    inputparams = get_inputparams(modelpath)
    tmin = inputparams["tmin"]
    dlogt = (math.log(inputparams["tmax"]) - math.log(tmin)) / inputparams["ntstep"]
    timesteps = range(inputparams["ntstep"])

    return (
        pl
        .LazyFrame({"timestep": list(timesteps)}, schema={"timestep": pl.Int32})
        .with_columns(
            tmid_days=tmin * pl.lit(math.e).pow((pl.col("timestep") + 0.5) * dlogt),
            tstart_days=tmin * pl.lit(math.e).pow(pl.col("timestep") * dlogt),
            tend_days=tmin * pl.lit(math.e).pow((pl.col("timestep") + 1) * dlogt),
        )
        .with_columns(twidth_days=pl.col("tend_days") - pl.col("tstart_days"))
    )


@lru_cache(maxsize=16)
def get_timestep_times(modelpath: Path | str, loc: t.Literal["mid", "start", "end", "delta"] = "mid") -> list[float]:
    """Return a list of the times in days of each timestep."""
    colname_of_loc = {"mid": "tmid_days", "start": "tstart_days", "end": "tend_days", "delta": "twidth_days"}

    if colname := colname_of_loc.get(loc):
        return get_timesteps(modelpath).select(colname).collect().get_column(colname).to_list()

    msg = "loc must be one of 'mid', 'start', 'end', or 'delta'"
    raise ValueError(msg)


def get_timestep_of_timedays(modelpath: Path | str, timedays: str | float) -> int:
    """Return the timestep containing the given time in days."""
    if isinstance(timedays, str):
        # could be a string like '330d'
        timedays = timedays.rstrip("d")

    try:
        timedays_float = float(timedays)
    except ValueError as exc:
        msg = f"Cannot read {timedays!r} as a time in days"
        if isinstance(timedays, str) and timedays[:1].isalpha() and timedays.lstrip("s").replace(".", "").isdigit():
            # a value joins only a flag of one letter, thus -ts70 reads as -t with the value s70
            msg += (
                f". A joined value such as -ts{timedays.lstrip('s')} reads as -t {timedays}, thus put a space after -ts"
            )
        raise ValueError(msg) from exc

    arr_tstart = get_timestep_times(modelpath, loc="start")
    # to avoid roundoff errors, use the next timestep's tstart at each timestep's tend (t_width is not exact)
    # copy into a new list to avoid mutating the lru_cached list returned by get_timestep_times
    arr_tend = [*arr_tstart[1:], get_timestep_times(modelpath, loc="end")[-1]]

    for ts, (tstart, tend) in enumerate(zip(arr_tstart, arr_tend, strict=False)):
        if tstart <= timedays_float < tend:
            return ts

    # the message below says the model covers up to the last tend, thus that exact time must match
    if arr_tstart and timedays_float == arr_tend[-1]:
        return len(arr_tstart) - 1

    msg = (
        f"No timestep of this model covers {timedays_float:g} days. It has {len(arr_tstart)} timesteps, "
        f"which cover {arr_tstart[0]:.2f} to {arr_tend[-1]:.2f} days. Give -timedays in that range, or "
        "-timestep to name one directly"
    )
    raise ValueError(msg)


def parse_timedays_range(timedays_range_str: str | float) -> tuple[float, float] | None:
    """Return the two ends of a time range like 2.2-2.8, or None when the text names a single time.

    This function reads the text as one number first, thus a hyphen in front of a negative time
    splits no range.
    """
    text = str(timedays_range_str).strip()
    with contextlib.suppress(ValueError):
        float(text)
        return None

    return parse_float_range(text, "a time in days or as a range such as 2.2-2.8")


def get_bad_timestep_message(modelpath: Path | str, timestep: int) -> str:
    """Return the message that names the timesteps of a model, for a timestep that is not one of them."""
    tstarts = get_timestep_times(modelpath, loc="start")
    tends = get_timestep_times(modelpath, loc="end")

    return (
        f"Timestep {timestep} is not in this model. It has {len(tstarts)} timesteps, 0 to "
        f"{len(tstarts) - 1}, which cover {tstarts[0]:.2f} to {tends[-1]:.2f} days. "
        '"last" names the final timestep'
    )


def get_single_timestep(timestep: str | int | None, modelpath: Path | str) -> int | None:
    """Return the one timestep that -timestep names, or None when it names none.

    Every command reads the same grammar, thus "40" and "last" work on each one. A command that plots
    one timestep cannot take a range, thus a range earns a message that says so.
    """
    if timestep is None:
        return None

    from artistools.misc.cliutils import parse_range_list

    lasttimestep = len(get_timestep_times(modelpath, loc="mid")) - 1
    timesteps = parse_range_list(str(timestep), dictvars={"last": lasttimestep})
    if len(timesteps) > 1:
        msg = (
            f"-timestep '{timestep}' names {len(timesteps)} timesteps, and this command plots one. "
            "Give one timestep, e.g. -ts 40 or -ts last"
        )
        raise ValueError(msg)

    if not 0 <= timesteps[0] <= lasttimestep:
        raise ValueError(get_bad_timestep_message(modelpath, timesteps[0]))

    return timesteps[0]


def parse_timestep_token(token: str, dictvars: dict[str, int]) -> int:
    """Return the timestep that a token names, resolving a keyword such as "last"."""
    token = token.strip()

    return dictvars[token] if token in dictvars else int(token)


def apply_time_range_args(args: argparse.Namespace, modelpaths: Sequence[Path | str]) -> None:
    """Narrow the plotted time range from -timestep or from a -timedays range.

    -timemin and -timemax give the range directly. A single -timedays value names one time for
    --brightnessattime, thus only a value that holds a range takes part here.
    """
    dayrange = parse_timedays_range(args.timedays) if args.timedays is not None else None
    if args.timestep is dayrange is None:
        return

    # only a timestep needs the times of a model. A reference light curve holds no such data, thus a
    # command that plots reference data alone still takes a range in days
    # a path can name a light curve file of a run, and get_time_range reads the folder of the run
    artispaths = [get_model_folder(path) for path in modelpaths if path_is_artis_model(path)]
    if not artispaths:
        if dayrange is None:
            msg = "-timestep names a timestep of an ARTIS model, and no model path gives one. Give -timedays"
            raise ValueError(msg)
        rangemin, rangemax = dayrange
    else:
        _, _, rangemin, rangemax = get_time_range(
            artispaths[0], timestep_range_str=args.timestep, timedays_range_str=args.timedays
        )

        # the plot holds one time axis, thus one range in days must serve every model. A timestep
        # names different days on a different timestep grid. The days of the first model would
        # then show another timestep of the second model, and no message would say so
        if args.timestep is not None:
            for otherpath in artispaths[1:]:
                _, _, othermin, othermax = get_time_range(otherpath, timestep_range_str=args.timestep)
                if abs(othermin - rangemin) > 1e-4 or abs(othermax - rangemax) > 1e-4:
                    exit_with_error(
                        f"timestep {args.timestep} covers {rangemin:.2f} to {rangemax:.2f} days in "
                        f"{get_model_name(artispaths[0])} and {othermin:.2f} to {othermax:.2f} days in "
                        f"{get_model_name(otherpath)}, because their timestep grids differ. Give the "
                        "range in days with -timedays, which means the same for every model"
                    )

    if args.timemin is None:
        args.timemin = rangemin
    if args.timemax is None:
        args.timemax = rangemax


def get_time_range(
    modelpath: Path | str,
    timestep_range_str: str | int | None = None,
    timemin: float | str | None = None,
    timemax: float | str | None = None,
    timedays_range_str: str | float | None = None,
    clamp_to_timesteps: bool = True,
) -> tuple[int, int, float, float]:
    """Handle a time range specified in either days or timesteps."""
    # assertions make sure time is specified either by timesteps or times in days, but not both!
    tstarts = get_timestep_times(modelpath, loc="start")
    tmids = get_timestep_times(modelpath, loc="mid")
    tends = get_timestep_times(modelpath, loc="end")

    time_days_lower, time_days_upper = None, None

    # keep the bounds that the caller gave. The search below replaces a missing bound with a
    # sentinel (-1.0 or inf), and the function must not return a sentinel as a time
    user_timemin, user_timemax = timemin, timemax

    if timemin is not None and float(timemin) > tends[-1]:
        print_warning(f"{get_model_name(modelpath)}: timemin {timemin} is after the last timestep at {tends[-1]:.1f}")
        return -1, -1, -math.inf, -math.inf
    if timemax is not None and float(timemax) < tstarts[0]:
        print_warning(
            f"{get_model_name(modelpath)}: timemax {timemax} is before the first timestep at {tstarts[0]:.1f}"
        )
        return -1, -1, -math.inf, -math.inf

    if timestep_range_str is not None:
        # a keyword argument of the API gives an int, e.g. plot(timestep=11), and a command line gives
        # the string "11" or a range such as "10-20"
        timestep_range_str = str(timestep_range_str)
        # a silent precedence hid the argument that the user gave and did not get, thus this refuses the
        # combination. Only -timedays takes part. A caller assigns the timemin and timemax that this
        # function returns back onto its own arguments, thus a second call would see its own output.
        if timedays_range_str is not None:
            msg = "Specify only one of -timestep and -timedays"
            raise ValueError(msg)

        # "last" names the final timestep, so that a command needs no arithmetic to ask for it
        dictvars = {"last": len(tmids) - 1}
        rangeparts = re.split(r"(?<=[0-9a-zA-Z])-", timestep_range_str.strip())
        if len(rangeparts) == 2:
            timestepmin, timestepmax = (parse_timestep_token(nts, dictvars) for nts in rangeparts)
        elif len(rangeparts) > 2:
            msg = f"'{timestep_range_str}' names more than one range of timesteps"
            raise ValueError(msg)
        else:
            timestepmin = parse_timestep_token(timestep_range_str, dictvars)
            timestepmax = timestepmin

        # a range that overshoots the end still starts inside the run, thus only the start must be in it
        if timestepmin > dictvars["last"] or timestepmin < 0:
            msg = get_bad_timestep_message(modelpath, timestepmin)
            raise ValueError(msg)

        if timestepmax < timestepmin:
            msg = (
                f"'{timestep_range_str}' names the timestep range {timestepmin} to {timestepmax},"
                " which ends before it starts"
            )
            raise ValueError(msg)
    elif (timemin is not None or timemax is not None) or timedays_range_str is not None:
        if timemin is None and timemax is not None:
            timemin = -1.0
        elif timemax is None and timemin is not None:
            timemax = math.inf

        # time days range is specified
        timestepmin = None
        timestepmax = None
        if timedays_range_str is not None:
            if (timedaysrange := parse_timedays_range(timedays_range_str)) is not None:
                timemin, timemax = timedaysrange
                if not clamp_to_timesteps:
                    time_days_lower = timemin
                    time_days_upper = timemax
            else:
                timeavg = float(timedays_range_str)
                timestepmin = get_timestep_of_timedays(modelpath, timeavg)
                timestepmax = timestepmin
                timemin = tstarts[timestepmin]
                timemax = tends[timestepmax]

        assert timemin is not None

        for timestep, tmid in enumerate(tmids):
            if tmid >= float(timemin):
                timestepmin = timestep
                break

        if timestepmin is None:
            msg = f"Time min {timemin} is greater than all timesteps ({tstarts[0]} to {tends[-1]})"
            raise ValueError(msg)

        if timemax is None:
            timemax = tends[-1]
        assert timemax is not None

        for timestep, tmid in enumerate(tmids):
            if tmid <= float(timemax):
                timestepmax = timestep

        if timestepmax is None:
            msg = f"Time max {timemax} is less than all timesteps ({tstarts[0]} to {tends[-1]})"
            raise ValueError(msg)
        if timestepmax < timestepmin:
            if clamp_to_timesteps:
                msg = f"Specified time range does not include any full timesteps. {timestepmin=} {timestepmax=}"
                raise ValueError(msg)
            timestepmax = timestepmin
    else:
        msg = (
            "No time was given. Give one with -timedays (e.g. -t 300 or -t 290-320), with -timestep "
            "(e.g. -ts 40 or -ts last), or with -timemin and -timemax"
        )
        raise ValueError(msg)

    timesteplast = len(tmids) - 1
    if timestepmax > timesteplast:
        print_warning(f"timestepmax {timestepmax} > timesteplast {timesteplast}")
        timestepmax = timesteplast

    # when the range was given as timesteps there is no requested time in days, so the timestep bounds are the only
    # times available even if the caller asked not to clamp
    if time_days_lower is None:
        assert timestepmin is not None
        time_days_lower = tstarts[timestepmin] if (clamp_to_timesteps or user_timemin is None) else float(user_timemin)

    if time_days_upper is None:
        assert timestepmax is not None
        time_days_upper = tends[timestepmax] if (clamp_to_timesteps or user_timemax is None) else float(user_timemax)

    assert timestepmin is not None
    assert timestepmax is not None
    assert time_days_lower is not None
    assert time_days_upper is not None

    return timestepmin, timestepmax, time_days_lower, time_days_upper


def get_timestep_time(modelpath: Path | str, timestep: int) -> float:
    """Return the time in days of the midpoint of a timestep number."""
    timearray = get_timestep_times(modelpath, loc="mid")
    return timearray[timestep]


@lru_cache(maxsize=16)
def get_escaped_arrivalrange(modelpath: Path | str) -> tuple[int, float | int | None, float | int | None]:
    """Return the time range for which the entire model can send light signals the observer."""
    modelpath = Path(modelpath)
    from artistools.inputmodel import get_modeldata

    _, modelmeta = get_modeldata(modelpath, printwarningsonly=True)
    vmax = modelmeta["vmax_cmps"]  # max velocity component for a single axis [cm/s]

    # find the earliest possible escape time and add the largest possible travel time

    # for 2D and 3D models, the box corners are the maximum radius with (potentially) non-zero density
    dimensions = modelmeta["dimensions"]
    if dimensions not in {1, 2, 3}:
        msg = "Model dimensions must be 1, 2, or 3"
        raise ValueError(msg)
    cornervmax = vmax * math.sqrt(dimensions)

    # perfect initial conditions make t_arrive = tmin valid already. Light from the origin at tmin
    # escapes later, but the code subtracts that travel time, thus t_arrive stays tmin. The code
    # still waits until light from the origin reaches the corners
    validrange_start_days = get_timestep_times(modelpath, loc="start")[0] * (1 + cornervmax / C_cm_per_s)

    t_end = get_timestep_times(modelpath, loc="end")
    # find the last escape time, then subtract the longest travel time from the origin. This is the
    # correction of the observer time
    try:
        depdata = get_deposition(modelpath=modelpath)  # use this file to find the last computed timestep
        # get_deposition() always provides a timestep column, adding a row index if the file has no such column
        nts_last = depdata.select(pl.col("timestep").max()).collect().item()
    except FileNotFoundError:
        print_warning("No deposition.out file found. Assuming all timesteps have been computed")
        nts_last = len(t_end) - 1

    assert isinstance(nts_last, int)
    nts_last_tend = t_end[nts_last]

    # the last observer time is the escape at the end of the last computed timestep, minus the
    # longest travel time from the origin. The code assumes a 3D propagation grid, which is the safe
    # assumption. A 1D grid or a 2D grid can give a shorter travel time
    validrange_end_days: float | int = nts_last_tend * (1 - vmax * math.sqrt(3) / C_cm_per_s)

    if validrange_start_days > validrange_end_days:
        return nts_last, None, None

    return nts_last, validrange_start_days, validrange_end_days
