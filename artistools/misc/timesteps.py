"""Timestep definitions, time range selection, and deposition rates."""

import contextlib
import math
import re
import typing as t
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

from artistools.constants import C_cm_per_s
from artistools.misc.fileio import firstexisting
from artistools.misc.fileio import firstexisting_or_none
from artistools.misc.fileio import path_is_codecomparison
from artistools.misc.fileio import polars_source
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
    if not np.allclose(t_mid_days, ts_mids[: len(t_mid_days)], rtol=0.01):
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
        return (
            pl
            .scan_csv(polars_source(tsfilepath), has_header=True, separator=" ")
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

    msg = (
        f"No timestep of this model covers {timedays_float:g} days. It has {len(arr_tstart)} timesteps, "
        f"which cover {arr_tstart[0]:.2f} to {arr_tend[-1]:.2f} days. Give -timedays in that range, or "
        "-timestep to name one directly"
    )
    raise ValueError(msg)


def parse_timedays_range(timedays_range_str: str | float) -> tuple[float, float] | None:
    """Return the two ends of a time range like 2.2-2.8, or None when the text names a single time.

    A hyphen also appears inside an exponent, e.g. 1e-2, and in front of a negative time. The text is
    read as one number first, thus only a hyphen that separates two numbers splits the range.
    """
    text = str(timedays_range_str).strip()
    with contextlib.suppress(ValueError):
        float(text)
        return None

    # a digit or a decimal point in front of the hyphen means it separates two numbers
    parts = re.split(r"(?<=[0-9.])-", text)
    if len(parts) != 2:
        msg = f"Cannot read {text!r} as a time in days or as a range such as 2.2-2.8"
        raise ValueError(msg)

    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        msg = f"Cannot read {text!r} as a time in days or as a range such as 2.2-2.8"
        raise ValueError(msg) from exc


def parse_timestep_token(token: str, dictvars: dict[str, int]) -> int:
    """Return the timestep that a token names, resolving a keyword such as "last"."""
    token = token.strip()

    return dictvars[token] if token in dictvars else int(token)


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

    if timemin is not None and float(timemin) > tends[-1]:
        print(f"{get_model_name(modelpath)}: WARNING timemin {timemin} is after the last timestep at {tends[-1]:.1f}")
        return -1, -1, -math.inf, -math.inf
    if timemax is not None and float(timemax) < tstarts[0]:
        print(
            f"{get_model_name(modelpath)}: WARNING timemax {timemax} is before the first timestep at {tstarts[0]:.1f}"
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
        if "-" in timestep_range_str:
            timestepmin, timestepmax = (parse_timestep_token(nts, dictvars) for nts in timestep_range_str.split("-"))
        else:
            timestepmin = parse_timestep_token(timestep_range_str, dictvars)
            timestepmax = timestepmin

        # a range that overshoots the end still starts inside the run, thus only the start must be in it
        if timestepmin > dictvars["last"]:
            msg = (
                f"Timestep {timestepmin} is not in this model. It has {len(tmids)} timesteps, 0 to "
                f"{dictvars['last']}, which cover {tstarts[0]:.2f} to {tends[-1]:.2f} days. "
                '"last" names the final timestep'
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
                # timedelta = 10
                # timemin, timemax = timeavg - timedelta, timeavg + timedelta

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
        print(f"Warning timestepmax {timestepmax} > timesteplast {timesteplast}")
        timestepmax = timesteplast

    # when the range was given as timesteps there is no requested time in days, so the timestep bounds are the only
    # times available even if the caller asked not to clamp
    if time_days_lower is None:
        assert timestepmin is not None
        time_days_lower = tstarts[timestepmin] if (clamp_to_timesteps or timemin is None) else float(timemin)

    if time_days_upper is None:
        assert timestepmax is not None
        time_days_upper = tends[timestepmax] if (clamp_to_timesteps or timemax is None) else float(timemax)

    assert timestepmin is not None
    assert timestepmax is not None
    assert time_days_lower is not None
    assert time_days_upper is not None

    return timestepmin, timestepmax, time_days_lower, time_days_upper


def get_timestep_time(modelpath: Path | str, timestep: int) -> float:
    """Return the time in days of the midpoint of a timestep number."""
    timearray = get_timestep_times(modelpath, loc="mid")
    return timearray[timestep]


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

    # if the initial conditions were perfect, then t_arrive = tmin would be valid already
    # (with a free path, light from the origin at tmin would escape sometime later, but that travel time would be subtracted to get t_arrive = tmin),
    # but we should at least wait until light signals from the origin reach the corners
    validrange_start_days = get_timestep_times(modelpath, loc="start")[0] * (1 + cornervmax / C_cm_per_s)

    t_end = get_timestep_times(modelpath, loc="end")
    # find the last possible escape time and subtract the largest possible travel time (observer time correction)
    try:
        depdata = get_deposition(modelpath=modelpath)  # use this file to find the last computed timestep
        # get_deposition() always provides a timestep column, adding a row index if the file has no such column
        nts_last = depdata.select(pl.col("timestep").max()).collect().item()
    except FileNotFoundError:
        print("WARNING: No deposition.out file found. Assuming all timesteps have been computed")
        nts_last = len(t_end) - 1

    assert isinstance(nts_last, int)
    nts_last_tend = t_end[nts_last]

    # last valid observer time is escape at the end of the latest computed timestep minus the longest travel time relative to origin
    # assume we're on a 3D propagation grid for safety (1D or 2D could reduce the travel time somewhat)
    validrange_end_days: float | int = nts_last_tend * (1 - vmax * math.sqrt(3) / C_cm_per_s)

    if validrange_start_days > validrange_end_days:
        return nts_last, None, None

    return nts_last, validrange_start_days, validrange_end_days
