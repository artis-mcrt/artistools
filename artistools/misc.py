import argparse
import functools
import gzip
import io
import itertools
import lzma
import math
import multiprocessing
import multiprocessing.pool
import string
import typing as t
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
import zstandard as zstd

from artistools.configuration import get_config

roman_numerals = (
    "",
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
    "XV",
    "XVI",
    "XVII",
    "XVIII",
    "XIX",
    "XX",
)


class CustomArgHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom argparse formatter to show default values in help text, sorted with dashes last."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        kwargs["max_help_position"] = 39
        super().__init__(*args, **kwargs)

    def add_arguments(self, actions: Iterable[argparse.Action]) -> None:
        getinvocation = super()._format_action_invocation

        def my_sort(action: argparse.Action) -> str:
            return getinvocation(action).upper().replace("-", "z")  # push dash chars below alphabet

        actions = sorted(actions, key=my_sort)
        super().add_arguments(actions)


class AppendPath(argparse.Action):
    """Append a path to a list of paths."""

    def __call__(self, parser, args, values, option_string=None) -> None:  # type: ignore[no-untyped-def] # noqa: ARG002
        # if getattr(args, self.dest) is None:
        #     setattr(args, self.dest, [])
        if hasattr(values, "__iter__"):
            pathlist = getattr(args, self.dest)
            # not pathlist avoids repeated appending of the same items when called from Python
            # instead of from the command line
            if not pathlist:
                for pathstr in values:
                    # if Path(pathstr) not in pathlist:
                    pathlist.append(Path(pathstr))
        else:
            setattr(args, self.dest, Path(values))


def showtimesteptimes(modelpath: Path | None = None, numberofcolumns: int = 5) -> None:
    """Print a table showing the timesteps and their corresponding times."""
    if modelpath is None:
        modelpath = Path()

    print("Timesteps and midpoint times in days:\n")

    times = get_timestep_times(modelpath, loc="mid")
    indexendofcolumnone = math.ceil((len(times) - 1) / numberofcolumns)
    for rownum in range(indexendofcolumnone):
        strline = ""
        for colnum in range(numberofcolumns):
            if colnum > 0:
                strline += "\t"
            newindex = rownum + colnum * indexendofcolumnone
            if newindex + 1 < len(times):
                strline += f"{newindex:4d}: {times[newindex + 1]:.3f}d"
        print(strline)


@lru_cache(maxsize=8)
def get_composition_data(filename: Path | str) -> pd.DataFrame:
    """Return a pandas DataFrame containing details of included elements and ions."""
    filename = Path(filename, "compositiondata.txt") if Path(filename).is_dir() else Path(filename)

    columns = [
        "Z",
        "nions",
        "lowermost_ion_stage",
        "uppermost_ion_stage",
        "nlevelsmax_readin",
        "abundance",
        "mass",
        "startindex",
    ]

    rowdfs = []
    with filename.open(encoding="utf-8") as fcompdata:
        nelements = int(fcompdata.readline())
        fcompdata.readline()  # T_preset
        fcompdata.readline()  # homogeneous_abundances
        startindex = 0
        for _ in range(nelements):
            line = fcompdata.readline()
            linesplit = line.split()
            row_list = [int(x) for x in linesplit[:5]] + [float(x) for x in linesplit[5:]] + [startindex]

            rowdfs.append(pd.DataFrame([row_list], columns=columns))

            startindex += int(rowdfs[-1].iloc[0]["nions"])

    return pd.concat(rowdfs, ignore_index=True)


def get_composition_data_from_outputfile(modelpath: Path) -> pd.DataFrame:
    """Read ion list from output file."""
    atomic_composition = {}

    with (modelpath / "output_0-0.txt").open(encoding="utf-8") as foutput:
        Z: int | None = None
        ioncount = 0
        for row in foutput:
            if row.split()[0] == "[input.c]":
                split_row = row.split()
                if split_row[1] == "element":
                    Z = int(split_row[4])
                    ioncount = 0
                elif split_row[1] == "ion":
                    ioncount += 1
                    assert Z is not None
                    atomic_composition[Z] = ioncount

    composition_df = pd.DataFrame([(Z, atomic_composition[Z]) for Z in atomic_composition], columns=["Z", "nions"])
    composition_df["lowermost_ion_stage"] = [1] * composition_df.shape[0]
    composition_df["uppermost_ion_stage"] = composition_df["nions"]
    return composition_df


def split_multitable_dataframe(res_df: pl.DataFrame | pd.DataFrame) -> dict[int, pl.DataFrame]:
    """Res (angle-resolved) files include a table for each direction bin."""
    if isinstance(res_df, pd.DataFrame):
        res_df = pl.from_pandas(res_df)

    header_row_indices = pl.arg_where(res_df[:, 0] == res_df[0, 0], eager=True)

    res_data = {
        tableindex: (
            res_df[table_row_start : header_row_indices[tableindex + 1], :]
            if tableindex + 1 < len(header_row_indices)
            else res_df[table_row_start:, :]
        )
        for tableindex, table_row_start in enumerate(header_row_indices)
    }

    # the number of timesteps and frequency bins should match for each subtable
    assert all(df.columns == res_data[0].columns for df in res_data.values())
    assert all(df.get_column(df.columns[0]).equals(res_data[0].get_column(df.columns[0])) for df in res_data.values())

    return res_data


def average_direction_bins(
    dirbindataframes: dict[int, pl.DataFrame], overangle: t.Literal["phi", "theta"]
) -> dict[int, pl.DataFrame]:
    """Average dict of direction-binned polars DataFrames according to the phi or theta angle."""
    dirbincount = get_viewingdirectionbincount()
    nphibins = get_viewingdirection_phibincount()
    ncosthetabins = get_viewingdirection_costhetabincount()

    if overangle == "phi":
        start_bin_range = range(0, dirbincount, nphibins)
    elif overangle == "theta":
        start_bin_range = range(nphibins)
    else:
        msg = "overangle must be 'phi' or 'theta'"
        raise ValueError(msg)

    # we will make a copy to ensure that we don't cause side effects from altering the original DataFrames
    # that might be returned again later by an lru_cached function
    dirbindataframesout: dict[int, pl.DataFrame] = {}

    for start_bin in start_bin_range:
        dirbindataframesout[start_bin] = dirbindataframes[start_bin]

        contribbins = (
            range(start_bin + 1, start_bin + nphibins)
            if overangle == "phi"
            else range(start_bin + ncosthetabins, dirbincount, ncosthetabins)
        )

        for dirbin_contrib in contribbins:
            dirbindataframesout[start_bin] += dirbindataframes[dirbin_contrib]

        dirbindataframesout[start_bin] /= 1 + len(contribbins)  # every nth bin is the average of n bins
        print(f"bin number {start_bin} = the average of bins {[start_bin, *list(contribbins)]}")

    return dirbindataframesout


def match_closest_time(reftime: float, searchtimes: list[t.Any]) -> str:
    """Get time closest to reftime in list of times (searchtimes)."""
    return str(min((float(x) for x in searchtimes), key=lambda x: abs(x - reftime)))


def get_vpkt_config(modelpath: Path | str) -> dict[str, t.Any]:
    filename = Path(modelpath, "vpkt.txt")

    with filename.open(encoding="utf-8") as vpkt_txt:
        vpkt_config: dict[str, t.Any] = {
            "nobsdirections": int(vpkt_txt.readline()),
            "cos_theta": [float(x) for x in vpkt_txt.readline().split()],
            "phi": [float(x) for x in vpkt_txt.readline().split()],
        }
        assert vpkt_config["nobsdirections"] == len(vpkt_config["cos_theta"])
        assert len(vpkt_config["cos_theta"]) == len(vpkt_config["phi"])

        speclistline = vpkt_txt.readline().split()
        nspecflag = int(speclistline[0])

        if nspecflag == 1:
            vpkt_config["nspectraperobs"] = int(speclistline[1])
            vpkt_config["z_excludelist"] = [int(x) for x in speclistline[2:]]
        else:
            vpkt_config["nspectraperobs"] = 1
            vpkt_config["z_excludelist"] = [0]

        vpkt_config["time_limits_enabled"], vpkt_config["initial_time"], vpkt_config["final_time"] = (
            int(x) for x in vpkt_txt.readline().split()
        )

    return vpkt_config


def get_grid_mapping(modelpath: Path | str) -> tuple[dict[int, list[int]], dict[int, int]]:
    """Return dict with the associated propagation cells for each model grid cell and a dict with the associated model grid cell of each propagration cell."""
    modelpath = Path(modelpath)
    filename = firstexisting("grid.out", tryzipped=True, folder=modelpath)
    dfgrid = pl.read_csv(
        zopenpl(filename),
        separator=" ",
        has_header=False,
        comment_prefix="#",
        schema={"cellindex": pl.Int32, "modelgridindex": pl.Int32},
    )
    assoc_cells = dict(
        dfgrid.group_by("modelgridindex")
        .agg(pl.col("cellindex").implode())
        .select([pl.col("modelgridindex"), pl.col("cellindex")])
        .iter_rows()
    )

    mgi_of_propcells = dict(dfgrid.select([pl.col("cellindex"), pl.col("modelgridindex")]).iter_rows())

    return assoc_cells, mgi_of_propcells


def get_wid_init_at_tmodel(
    modelpath: Path | str | None = None,
    ngridpoints: int | None = None,
    t_model_days: float | None = None,
    xmax: float | None = None,
) -> float:
    """Return the Cartesian cell width [cm] at the model snapshot time."""
    if ngridpoints is None or t_model_days is None or xmax is None:
        # Luke: ngridpoint only equals the number of model cells if the model is 3D
        assert modelpath is not None
        from artistools.inputmodel import get_modeldata

        _, modelmeta = get_modeldata(modelpath, getheadersonly=True)
        assert modelmeta["dimensions"] == 3
        ngridpoints = modelmeta["npts_model"]
        xmax = modelmeta["vmax_cmps"] * modelmeta["t_model_init_days"] * 86400.0
    assert ngridpoints is not None
    ncoordgridx: int = round(ngridpoints ** (1.0 / 3.0))

    assert xmax is not None
    return 2.0 * xmax / ncoordgridx


def vec_len(vec: Sequence[float] | npt.NDArray[np.floating]) -> float:
    return float(np.sqrt(np.dot(vec, vec)))


@lru_cache(maxsize=16)
def get_nu_grid(modelpath: Path) -> npt.NDArray[np.floating]:
    """Return an array of frequencies at which the ARTIS spectra are binned by exspec."""
    specdata = pl.read_csv(
        firstexisting(["spec.out", "specpol.out"], folder=modelpath, tryzipped=True),
        separator=" ",
        has_header=False,
        skip_rows=1,
        columns=[0],
        new_columns=["nu"],
    )
    return specdata["nu"].to_numpy()


def get_deposition(modelpath: Path | str = ".") -> pl.DataFrame:
    """Return a polars DataFrame containing the deposition data."""
    if Path(modelpath).is_file():
        depfilepath = Path(modelpath)
        modelpath = Path(modelpath).parent
    else:
        depfilepath = Path(modelpath, "deposition.out")

    ts_mids = get_timestep_times(modelpath, loc="mid")

    with depfilepath.open(encoding="utf-8") as fdep:
        line = fdep.readline()
        if line.startswith("#"):
            skiprows = 1
            columns = line.lstrip("#").split()
        else:
            skiprows = 0
            columns = ["tmid_days", "gammadep_Lsun", "positrondep_Lsun", "total_dep_Lsun"]

    depdata = pl.read_csv(depfilepath, separator=" ", skip_rows=skiprows, has_header=False, new_columns=columns)

    if "ts" in depdata.columns:
        depdata = depdata.rename({"ts": "timestep"})

    if "timestep" not in depdata.columns:
        depdata = depdata.with_row_index("timestep", offset=0)

    depdata = depdata.with_columns(timestep=pl.col("timestep").cast(pl.Int32))

    # no timesteps are given in the old format of deposition.out, so ensure that
    # the times in days match up with the times of our assumed timesteps
    if not np.allclose(depdata["tmid_days"].to_numpy(), ts_mids[: len(depdata["tmid_days"])], rtol=0.01):
        msg = "Deposition times do not match the timesteps"
        raise AssertionError(msg)

    return depdata


@lru_cache(maxsize=16)
def get_timestep_times(modelpath: Path | str, loc: t.Literal["mid", "start", "end", "delta"] = "mid") -> list[float]:
    """Return a list of the times in days of each timestep."""
    modelpath = Path(modelpath)
    # virtual path to code comparison workshop models
    if not modelpath.exists() and modelpath.parts[0] == "codecomparison":
        import artistools.codecomparison

        return artistools.codecomparison.get_timestep_times(modelpath=modelpath, loc=loc)

    # use timestep.out if possible (allowing arbitrary timestep lengths)
    tsfilepath = Path(modelpath, "timesteps.out")
    if tsfilepath.exists():
        dftimesteps = (
            pl.read_csv(tsfilepath, has_header=True, separator=" ")
            .rename(lambda column_name: column_name.removeprefix("#"))
            .with_columns(tend_days=pl.col("tstart_days") + pl.col("twidth_days"))
        )

        if loc == "mid":
            return dftimesteps["tmid_days"].to_list()
        if loc == "start":
            return dftimesteps["tstart_days"].to_list()
        if loc == "end":
            return dftimesteps["tend_days"].to_list()
        if loc == "delta":
            return dftimesteps["twidth_days"].to_list()

        msg = "loc must be one of 'mid', 'start', 'end', or 'delta'"
        raise ValueError(msg)

    # older versions of Artis always used logarithmic timesteps and didn't produce a timesteps.out file
    inputparams = get_inputparams(modelpath)
    tmin = inputparams["tmin"]
    dlogt = (math.log(inputparams["tmax"]) - math.log(tmin)) / inputparams["ntstep"]
    timesteps = range(inputparams["ntstep"])
    if loc == "mid":
        return [tmin * math.exp((ts + 0.5) * dlogt) for ts in timesteps]
    if loc == "start":
        return [tmin * math.exp(ts * dlogt) for ts in timesteps]
    if loc == "end":
        return [tmin * math.exp((ts + 1) * dlogt) for ts in timesteps]
    if loc == "delta":
        return [tmin * (math.exp((ts + 1) * dlogt) - math.exp(ts * dlogt)) for ts in timesteps]

    msg = "loc must be one of 'mid', 'start', 'end', or 'delta'"
    raise ValueError(msg)


def get_timestep_of_timedays(modelpath: Path | str, timedays: str | float) -> int:
    """Return the timestep containing the given time in days."""
    if isinstance(timedays, str):
        # could be a string like '330d'
        timedays = timedays.rstrip("d")

    timedays_float = float(timedays)

    arr_tstart = get_timestep_times(modelpath, loc="start")
    arr_tend = get_timestep_times(modelpath, loc="end")
    # to avoid roundoff errors, use the next timestep's tstart at each timestep's tend (t_width is not exact)
    arr_tend[:-1] = arr_tstart[1:]

    for ts, (tstart, tend) in enumerate(zip(arr_tstart, arr_tend, strict=False)):
        if tstart <= timedays_float < tend:
            return ts

    msg = f"Could not find timestep bracketing time {timedays_float}"
    raise ValueError(msg)


def get_time_range(
    modelpath: Path,
    timestep_range_str: str | None = None,
    timemin: float | None = None,
    timemax: float | None = None,
    timedays_range_str: str | float | None = None,
    clamp_to_timesteps: bool = True,
) -> tuple[int, int, float | None, float | None]:
    """Handle a time range specified in either days or timesteps."""
    # assertions make sure time is specified either by timesteps or times in days, but not both!
    tstarts = get_timestep_times(modelpath, loc="start")
    tmids = get_timestep_times(modelpath, loc="mid")
    tends = get_timestep_times(modelpath, loc="end")

    time_days_lower, time_days_upper = None, None

    if timemin and timemin > tends[-1]:
        print(f"{get_model_name(modelpath)}: WARNING timemin {timemin} is after the last timestep at {tends[-1]:.1f}")
        return -1, -1, timemin, timemax
    if timemax and timemax < tstarts[0]:
        print(
            f"{get_model_name(modelpath)}: WARNING timemax {timemax} is before the first timestep at {tstarts[0]:.1f}"
        )
        return -1, -1, timemin, timemax

    if timestep_range_str is not None:
        if "-" in timestep_range_str:
            timestepmin, timestepmax = (int(nts) for nts in timestep_range_str.split("-"))
        else:
            timestepmin = int(timestep_range_str)
            timestepmax = timestepmin
    elif (timemin is not None and timemax is not None) or timedays_range_str is not None:
        # time days range is specified
        timestepmin = None
        timestepmax = None
        if timedays_range_str is not None:
            if isinstance(timedays_range_str, str) and "-" in timedays_range_str:
                timemin, timemax = (float(timedays) for timedays in timedays_range_str.split("-"))
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
            print(f"Time min {timemin} is greater than all timesteps ({tstarts[0]} to {tends[-1]})")
            raise ValueError

        if not timemax:
            timemax = tends[-1]
        assert timemax is not None

        for timestep, tmid in enumerate(tmids):
            if tmid <= float(timemax):
                timestepmax = timestep

        assert timestepmax is not None
        if timestepmax < timestepmin:
            if clamp_to_timesteps:
                msg = f"Specified time range does not include any full timesteps. {timestepmin=} {timestepmax=}"
                raise ValueError(msg)
            timestepmax = timestepmin
    else:
        msg = "Either time or timesteps must be specified."
        raise ValueError(msg)

    timesteplast = len(tmids) - 1
    if timestepmax is not None and timestepmax > timesteplast:
        print(f"Warning timestepmax {timestepmax} > timesteplast {timesteplast}")
        timestepmax = timesteplast
    if time_days_lower is None:
        assert timestepmin is not None
        time_days_lower = tstarts[timestepmin] if clamp_to_timesteps else timemin
    if time_days_upper is None:
        assert timestepmax is not None
        time_days_upper = tends[timestepmax] if clamp_to_timesteps else timemax
    assert timestepmin is not None
    assert timestepmax is not None

    return timestepmin, timestepmax, time_days_lower, time_days_upper


def get_timestep_time(modelpath: Path | str, timestep: int) -> float:
    """Return the time in days of the midpoint of a timestep number."""
    timearray = get_timestep_times(modelpath, loc="mid")
    return timearray[timestep]


def get_escaped_arrivalrange(modelpath: Path | str) -> tuple[int, float | None, float | None]:
    """Return the time range for which the entire model can send light signals the observer."""
    modelpath = Path(modelpath)
    from artistools.inputmodel import get_modeldata

    _, modelmeta = get_modeldata(modelpath, printwarningsonly=True, getheadersonly=True)
    vmax = modelmeta["vmax_cmps"]
    cornervmax = math.sqrt(3 * vmax**2)

    # find the earliest possible escape time and add the largest possible travel time

    # for 3D models, the box corners can have non-zero density (allowing packet escape from tmin)
    # for 1D and 2D, the largest escape radius at tmin is the box side radius (if the prop grid was also 1D or 2D)
    vmax_tmin = cornervmax if modelmeta["dimensions"] == 3 else vmax

    # earliest completely valid time is tmin plus maximum possible travel time from the origin to the corner
    validrange_start_days = get_timestep_times(modelpath, loc="start")[0] * (1 + vmax_tmin / 29979245800)

    t_end = get_timestep_times(modelpath, loc="end")
    # find the last possible escape time and subtract the largest possible travel time (observer time correction)
    try:
        depdata = get_deposition(modelpath=modelpath)  # use this file to find the last computed timestep
        nts_last = depdata["timestep"].max() if "timestep" in depdata.columns else len(depdata) - 1
    except FileNotFoundError:
        print("WARNING: No deposition.out file found. Assuming all timesteps have been computed")
        nts_last = len(t_end) - 1

    assert isinstance(nts_last, int)
    nts_last_tend = t_end[nts_last]

    # latest possible valid range is the end of the latest computed timestep plus the longest travel time
    validrange_end_days: float = nts_last_tend * (1 - cornervmax / 29979245800)

    if validrange_start_days > validrange_end_days:
        return nts_last, None, None

    return nts_last, validrange_start_days, validrange_end_days


@lru_cache(maxsize=8)
def get_model_name(path: Path | str) -> str:
    """Get the name of an ARTIS model from the path to any file inside it.

    Name will be either from a special plotlabel.txt file if it exists or the enclosing directory name
    """
    path = Path(path)
    if not path.exists() and path.parts[0] == "codecomparison":
        return str(path)

    abspath = path.resolve()

    modelpath = abspath if abspath.is_dir() else abspath.parent

    try:
        plotlabelfile = Path(modelpath, "plotlabel.txt")
        with plotlabelfile.open(encoding="utf-8") as f:
            return f.readline().strip()
    except FileNotFoundError:
        return Path(modelpath).name


def get_z_a_nucname(nucname: str) -> tuple[int, int]:
    """Return atomic number and mass number from a string like 'Pb208', 'X_Pb208', or "nniso_Pb208' (returns 92, 208)."""
    if "_" in nucname:
        nucname = nucname.split("_")[1]

    z = get_atomic_number(nucname.rstrip(string.digits))
    assert z > 0

    a = int(nucname.lower().lstrip(string.ascii_lowercase))

    return z, a


@lru_cache(maxsize=1)
def get_elsymbolslist() -> list[str]:
    """Return a list of element symbols.

    Example:
    -------
    elsymbolslist()[26] = 'Fe'.

    """
    return [
        "n",
        *list(pd.read_csv(get_config()["path_datadir"] / "elements.csv", usecols=["symbol"])["symbol"].to_numpy()),
    ]


def get_elsymbols_df() -> pl.DataFrame:
    """Return a polars DataFrame of atomic number and element symbols."""
    return (
        pl.read_csv(
            get_config()["path_datadir"] / "elements.csv",
            separator=",",
            has_header=True,
            schema_overrides={"Z": pl.Int32},
        )
        .drop("name")
        .rename({"symbol": "elsymbol", "Z": "atomic_number"})
    )


def get_atomic_number(elsymbol: str) -> int:
    """Return the atomic number of an element symbol."""
    assert elsymbol is not None
    elsymbol = elsymbol.removeprefix("X_")
    elsymbol = elsymbol.split("_")[0].split("-")[0].rstrip(string.digits)

    if elsymbol.title() in get_elsymbolslist():
        return get_elsymbolslist().index(elsymbol.title())

    return -1


def decode_roman_numeral(strin: str) -> int:
    """Return the integer corresponding to a Roman numeral."""
    if strin.upper() in roman_numerals:
        return roman_numerals.index(strin.upper())
    return -1


def get_ion_stage_roman_numeral_df() -> pl.DataFrame:
    """Return a polars DataFrame of ionisation stage and roman numerals."""
    return pl.DataFrame(
        {"ion_stage": list(range(1, len(roman_numerals))), "ion_stage_roman": roman_numerals[1:]},
        schema={"ion_stage": pl.Int32, "ion_stage_roman": pl.Utf8},
    )


def get_elsymbol(atomic_number: int | np.int64) -> str:
    """Return the element symbol of an atomic number."""
    return get_elsymbolslist()[atomic_number]


def get_ion_tuple(ionstr: str) -> tuple[int, int] | int:
    """Return a tuple of the atomic number and ionisation stage such as (26,2) for an ion string like 'FeII', 'Fe II', or '26_2'.

    Return the atomic number for a string like 'Fe' or '26'.
    """
    if "_" in ionstr:
        ionstr = ionstr.split("_", maxsplit=1)[1]

    if ionstr.isdigit():
        return int(ionstr)

    if ionstr in get_elsymbolslist():
        return get_atomic_number(ionstr)

    elem = "?"
    strion_stage = "?"
    if " " in ionstr:
        elem, strion_stage = ionstr.split(" ")
    elif "_" in ionstr:
        elem, strion_stage = ionstr.split("_")
    else:
        for elsym in get_elsymbolslist():
            if ionstr.startswith(elsym):
                elem = elsym
                strion_stage = ionstr.removeprefix(elsym)
                break

    if not elem:
        msg = f"Could not parse ionstr {ionstr}"
        raise ValueError(msg)

    atomic_number = int(elem) if elem.isdigit() else get_atomic_number(elem)
    ion_stage = int(strion_stage) if strion_stage.isdigit() else decode_roman_numeral(strion_stage)

    return (atomic_number, ion_stage)


@lru_cache(maxsize=16)
def get_ionstring(
    atomic_number: int | np.int64,
    ion_stage: int | np.int64 | t.Literal["ALL"] | None,
    style: t.Literal["spectral", "chargelatex", "charge"] = "spectral",
    sep: str = " ",
) -> str:
    """Return a string with the element symbol and ionisation stage."""
    if ion_stage is None or ion_stage == "ALL":
        return get_elsymbol(atomic_number)

    if isinstance(ion_stage, str) and ion_stage.startswith(get_elsymbol(atomic_number)):
        # nuclides like Sr89 get passed in as atomic_number=38, ion_stage='Sr89'
        return ion_stage

    assert not isinstance(ion_stage, str)

    if style == "spectral":
        return f"{get_elsymbol(atomic_number)}{sep}{roman_numerals[ion_stage]}"

    strcharge = ""
    if style == "chargelatex":
        # ion notion e.g. Co+, Fe2+
        if ion_stage > 2:
            strcharge = r"$^{" + str(ion_stage - 1) + r"{+}}$"
        elif ion_stage == 2:
            strcharge = r"$^{+}$"
    elif ion_stage > 2:
        strcharge = f"{ion_stage - 1}+"
    elif ion_stage == 2:
        strcharge = "+"

    return f"{get_elsymbol(atomic_number)}{strcharge}"


def set_args_from_dict(parser: argparse.ArgumentParser, kwargs: dict[str, t.Any]) -> None:
    """Set argparse defaults from a dictionary."""
    # set_defaults expects the dest of an argument. Here we allow the option strings to be used as keys
    for arg in parser._actions:  # noqa: SLF001
        for optstring in arg.option_strings:
            if optstring.lstrip("-") in kwargs and arg.dest not in kwargs:
                kwargs[arg.dest] = kwargs.pop(optstring.lstrip("-"))

    parser.set_defaults(**kwargs)
    if unknown := {k: v for k, v in kwargs.items() if k not in (arg.dest for arg in parser._actions)}:  # noqa: SLF001
        msg = f"Unknown argument names: {unknown}"
        raise ValueError(msg)


def parse_range(rng: str, dictvars: dict[str, int]) -> Iterable[t.Any]:
    """Parse a string with an integer range and return a list of numbers, replacing special variables in dictvars."""
    strparts = rng.split("-")

    if len(strparts) not in {1, 2}:
        msg = f"Bad range: '{rng}'"
        raise ValueError(msg)

    parts = [int(i) if i not in dictvars else dictvars[i] for i in strparts]
    start: int = parts[0]
    end: int = start if len(parts) == 1 else parts[1]

    if start > end:
        end, start = start, end

    return range(start, end + 1)


def parse_range_list(rngs: str | list[str] | int, dictvars: dict[str, int] | None = None) -> list[t.Any]:
    """Parse a string with comma-separated ranges or a list of range strings.

    Return a sorted list of integers in any of the ranges.
    """
    if isinstance(rngs, list):
        rngs = ",".join(rngs)
    elif not hasattr(rngs, "split"):
        return [rngs]

    assert isinstance(rngs, str)
    return sorted(set(chain.from_iterable([parse_range(rng, dictvars or {}) for rng in rngs.split(",")])))


def makelist(x: Sequence[t.Any] | str | Path | None) -> list[t.Any]:
    """If x is not a list (or is a string), make a list containing x."""
    if x is None:
        return []
    return list(x) if isinstance(x, Iterable) else [x]


def trim_or_pad(requiredlength: int, *listoflistin: t.Any) -> Sequence[Sequence[t.Any]]:
    """Make lists equal in length to requiredlength either by padding with None or truncating."""
    list_sequence = []
    for listin in listoflistin:
        listin_makelist = makelist(listin)

        listout = [listin_makelist[i] if i < len(listin_makelist) else None for i in range(requiredlength)]

        assert len(listout) == requiredlength
        list_sequence.append(listout)
    return list_sequence


def flatten_list(listin: list[t.Any]) -> list[t.Any]:
    """Flatten a list of lists."""
    listout = []
    for elem in listin:
        if isinstance(elem, list):
            listout.extend(elem)
        else:
            listout.append(elem)
    return listout


def zopen(filename: Path | str, mode: str = "rt", encoding: str | None = None) -> t.Any:
    """Open filename, filename.zst, filename.gz or filename.xz."""
    ext_fopen: dict[str, t.Any] = {".zst": zstd.open, ".gz": gzip.open, ".xz": lzma.open}

    for ext, fopen in ext_fopen.items():
        file_withext = str(filename) if str(filename).endswith(ext) else str(filename) + ext
        if Path(file_withext).exists():
            return fopen(file_withext, mode=mode, encoding=encoding)

    # open() can raise file not found if this file doesn't exist
    return Path(filename).open(mode=mode, encoding=encoding)


def zopenpl(filename: Path | str, mode: str = "rt", encoding: str | None = None) -> t.Any | Path:
    """Open filename, filename.zst, filename.gz or filename.xz. If polars.read_csv can read the file directly, return a Path object instead of a file object."""
    mode = "r"
    ext_fopen: dict[str, t.Any | None] = {".zst": None, ".gz": None, ".xz": lzma.open}

    for ext, fopen in ext_fopen.items():
        file_withext = str(filename) if str(filename).endswith(ext) else str(filename) + ext
        if Path(file_withext).exists():
            return Path(file_withext) if fopen is None else fopen(file_withext, mode=mode, encoding=encoding)

    return Path(filename)


def firstexisting(
    filelist: Sequence[str | Path] | str | Path,
    folder: Path | str = Path(),
    tryzipped: bool = True,
    search_subfolders: bool = True,
) -> Path:
    """Return the first existing file in file list. If none exist, raise exception."""
    if isinstance(filelist, str | Path):
        filelist = [Path(filelist)]
    else:
        assert isinstance(filelist, Iterable)
        filelist = [Path(x) for x in filelist]

    folder = Path(folder)
    thispath = Path(folder, filelist[0])

    if thispath.exists():
        return thispath

    fullpaths = []

    def search_folders() -> Generator[Path]:
        yield Path(folder)
        if search_subfolders:
            for filename in filelist:
                for p in Path(folder).glob(f"*/{filename}*"):
                    yield p.parent

    for searchfolder in search_folders():
        for filename in filelist:
            thispath = Path(searchfolder, filename)
            if thispath.exists():
                return thispath

            fullpaths.append(thispath)

            if tryzipped:
                for ext in (".zst", ".gz", ".xz"):
                    filename_withext = Path(str(filename) if str(filename).endswith(ext) else str(filename) + ext)
                    if filename_withext not in filelist:
                        thispath = Path(searchfolder, filename_withext)
                        if thispath.exists():
                            return thispath
                        fullpaths.append(thispath)

    strfilelist = "\n  ".join([str(x.relative_to(folder)) for x in fullpaths])
    orsub = " or subfolders" if search_subfolders else ""
    msg = f"None of these files exist in {folder}{orsub}: \n  {strfilelist}"
    raise FileNotFoundError(msg)


def anyexist(
    filelist: Sequence[str | Path], folder: Path | str = Path(), tryzipped: bool = True, search_subfolders: bool = True
) -> Path | None:
    """Return true if any files in file list exist."""
    try:
        filepath = firstexisting(
            filelist=filelist, folder=folder, tryzipped=tryzipped, search_subfolders=search_subfolders
        )
    except FileNotFoundError:
        return None

    return filepath


def batched(iterable: Iterable[t.Any], n: int) -> Generator[list[t.Any], t.Any]:
    """Batch data into iterators of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        msg = "n must be at least one"
        raise ValueError(msg)
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield list(itertools.chain((first_el,), chunk_it))


def stripallsuffixes(f: Path) -> Path:
    """Take a file path (e.g. packets00_0000.out.gz) and return the Path with no suffixes (e.g. packets)."""
    f_nosuffixes = Path(f)
    for _ in f.suffixes:
        f_nosuffixes = f_nosuffixes.with_suffix("")  # each call removes only one suffix

    return f_nosuffixes


def readnoncommentline(file: io.TextIOBase) -> str:
    """Read a line from the text file, skipping blank and comment lines that begin with #."""
    line = ""

    while not line.strip() or line.lstrip().startswith("#"):
        line = file.readline()

    return line


@lru_cache(maxsize=24)
def get_file_metadata(filepath: Path | str) -> dict[str, t.Any]:
    """Return a dict of metadata for a file, either from a metadata file or from the big combined metadata file."""
    filepath = Path(filepath)

    def add_derived_metadata(metadata: dict[str, t.Any]) -> dict[str, t.Any]:
        if "a_v" in metadata and "e_bminusv" in metadata and "r_v" not in metadata:
            metadata["r_v"] = metadata["a_v"] / metadata["e_bminusv"]
        elif "e_bminusv" in metadata and "r_v" in metadata and "a_v" not in metadata:
            metadata["a_v"] = metadata["e_bminusv"] * metadata["r_v"]
        elif "a_v" in metadata and "r_v" in metadata and "e_bminusv" not in metadata:
            metadata["e_bminusv"] = metadata["a_v"] / metadata["r_v"]

        return metadata

    import yaml

    filepath = Path(str(filepath).replace(".xz", "").replace(".gz", "").replace(".zst", ""))

    # check if the reference file (e.g. spectrum.txt) has an metadata file (spectrum.txt.meta.yml)
    individualmetafile = filepath.with_suffix(f"{filepath.suffix}.meta.yml")
    if individualmetafile.exists():
        with individualmetafile.open("r", encoding="utf-8") as yamlfile:
            metadata = yaml.safe_load(yamlfile)

        return add_derived_metadata(metadata)

    # check if the metadata is in the big combined metadata file (todo: eliminate this file)
    combinedmetafile = Path(filepath.parent.resolve(), "metadata.yml")
    if combinedmetafile.exists():
        with combinedmetafile.open("r", encoding="utf-8") as yamlfile:
            combined_metadata = yaml.safe_load(yamlfile)
        metadata = combined_metadata.get(str(filepath), {})

        return add_derived_metadata(metadata)

    print(f"No metadata found for: {filepath}")

    return {}


def get_filterfunc(args: argparse.Namespace, mode: str = "interp") -> Callable[[t.Any], t.Any] | None:
    """Use command line arguments to determine the appropriate filter function."""
    filterfunc = None
    dictargs = vars(args)

    if dictargs.get("filtermovingavg", False):

        def movavgfilterfunc(ylist: t.Any) -> t.Any:
            n = args.filtermovingavg
            arr_padded = np.pad(ylist, (n // 2, n - 1 - n // 2), mode="edge")
            return np.convolve(arr_padded, np.ones((n,)) / n, mode="valid")

        assert filterfunc is None
        filterfunc = movavgfilterfunc

    if dictargs.get("filtersavgol", False):
        import scipy.signal

        window_length, polyorder = (int(x) for x in args.filtersavgol)

        assert filterfunc is None
        filterfunc = functools.partial(
            scipy.signal.savgol_filter, window_length=window_length, polyorder=polyorder, mode=mode
        )

        print("Applying Savitzky-Golay filter")

    return filterfunc


def merge_pdf_files(pdf_files: list[str]) -> None:
    """Merge a list of PDF files into a single PDF file."""
    from pypdf import PdfWriter

    merger = PdfWriter()

    for pdfpath in pdf_files:
        with Path(pdfpath).open("rb") as pdffile:
            merger.append(pdffile)
        Path(pdfpath).unlink()

    resultfilename = f"{pdf_files[0].replace('.pdf', '')}-{pdf_files[-1].replace('.pdf', '')}"
    with Path(f"{resultfilename}.pdf").open("wb") as resultfile:
        merger.write(resultfile)

    print(f"Files merged and saved to {resultfilename}.pdf")


def get_nuclides(modelpath: Path | str) -> pl.LazyFrame:
    """Return LazyFrame with: pellet_nucindex atomic_number A nucname from nuclides.out file."""
    filepath = Path(modelpath, "nuclides.out")
    if not filepath.is_file():
        msg = f"File {filepath} not found"
        raise FileNotFoundError(msg)

    dfnuclides = (
        pl.scan_csv(filepath, separator=" ", has_header=True)
        .rename({"#nucindex": "pellet_nucindex", "Z": "atomic_number"})
        .join(get_elsymbols_df().lazy(), on="atomic_number", how="left")
        .with_columns(nucname=pl.col("elsymbol") + pl.col("A").cast(pl.String))
    ).with_columns(pl.col(pl.Int64).cast(pl.Int32))

    return pl.concat(
        [
            pl.LazyFrame(
                {
                    "pellet_nucindex": -1,
                    "atomic_number": -1,
                    "A": -1,
                    "elsymbol": "initial energy",
                    "nucname": "initial energy",
                },
                schema=dfnuclides.collect_schema(),
            ),
            dfnuclides,
        ],
        how="vertical",
    ).lazy()


def get_bflist(modelpath: Path | str, get_ion_str: bool = False) -> pl.LazyFrame:
    """Return a dict of bound-free transitions from bflist.out."""
    compositiondata = get_composition_data(modelpath)
    bflistpath = firstexisting(["bflist.out", "bflist.dat"], folder=modelpath, tryzipped=True)
    print(f"Reading {bflistpath}")
    schema = {
        "bfindex": pl.Int32,
        "elementindex": pl.Int32,
        "ionindex": pl.Int32,
        "lowerlevel": pl.Int32,
        "upperionlevel": pl.Int32,
    }
    try:
        dfboundfree = pl.read_csv(
            bflistpath,
            skip_rows=1,
            has_header=False,
            separator=" ",
            new_columns=["bfindex", "elementindex", "ionindex", "lowerlevel", "upperionlevel"],
            schema_overrides=schema,
        ).lazy()
    except pl.exceptions.NoDataError:
        dfboundfree = pl.DataFrame(schema=schema).lazy()

    dfboundfree = dfboundfree.with_columns(
        atomic_number=pl.col("elementindex").map_elements(
            lambda elementindex: compositiondata["Z"][elementindex], return_dtype=pl.Int32
        ),
        ion_stage=(
            pl.col("ionindex")
            + pl.col("elementindex").map_elements(
                lambda elementindex: compositiondata["lowermost_ion_stage"][elementindex], return_dtype=pl.Int32
            )
        ),
    )

    dfboundfree = dfboundfree.drop(["elementindex", "ionindex"])

    if get_ion_str:
        dfboundfree = (
            dfboundfree.join(get_ion_stage_roman_numeral_df().lazy(), on="ion_stage", how="left")
            .join(get_elsymbols_df().lazy(), on="atomic_number", how="left")
            .with_columns(ion_str=pl.col("elsymbol") + " " + pl.col("ion_stage_roman"))
        )

    return dfboundfree


class LineTuple(t.NamedTuple):
    """Named tuple for a line in linestat.out."""

    lambda_angstroms: float
    atomic_number: int
    ion_stage: int
    upperlevelindex: int
    lowerlevelindex: int


def read_linestatfile(filepath: Path | str) -> tuple[list[float], list[int], list[int], list[int], list[int]]:
    """Load linestat.out containing transitions wavelength, element, ion, upper and lower levels."""
    if Path(filepath).is_dir():
        filepath = firstexisting("linestat.out", folder=filepath, tryzipped=True)

    print(f"Reading {filepath}")

    data = np.loadtxt(zopen(filepath))
    lambda_angstroms = data[0] * 1e8
    nlines = len(lambda_angstroms)

    atomic_numbers = data[1].astype(int)
    assert len(atomic_numbers) == nlines

    ion_stages = data[2].astype(int)
    assert len(ion_stages) == nlines

    # the file adds one to the levelindex, i.e. lowest level is 1
    upper_levels = data[3].astype(int)
    assert len(upper_levels) == nlines

    lower_levels = data[4].astype(int)
    assert len(lower_levels) == nlines

    return lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels


def get_linelist_pldf(modelpath: Path | str, get_ion_str: bool = False) -> pl.LazyFrame:
    textfile = firstexisting("linestat.out", folder=modelpath)
    parquetfile = Path(modelpath, "linelist.out.parquet")
    if not parquetfile.is_file() or parquetfile.stat().st_mtime < textfile.stat().st_mtime:
        lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels = read_linestatfile(textfile)

        pldf = (
            pl.DataFrame({
                "lambda_angstroms": lambda_angstroms,
                "atomic_number": atomic_numbers,
                "ion_stage": ion_stages,
                "upper_level": upper_levels,
                "lower_level": lower_levels,
            })
            .with_row_count(name="lineindex")
            .with_columns(
                pl.col(pl.UInt32).cast(pl.Int32), pl.col(pl.Int64).cast(pl.Int32), pl.col(pl.Float64).cast(pl.Float32)
            )
        )
        pldf.write_parquet(parquetfile, compression="zstd")
        print(f"Saved {parquetfile}")
    else:
        print(f"Reading {parquetfile}")

    linelist_lazy = (
        pl.scan_parquet(parquetfile)
        .with_columns(
            pl.when(pl.col("lambda_angstroms").is_between(2000, 20000))
            .then(pl.col("lambda_angstroms") / 1.0003)
            .otherwise(pl.col("lambda_angstroms"))
            .alias("lambda_angstroms_air"),
            pl.col(pl.UInt32).cast(pl.Int32),
            pl.col(pl.Int64).cast(pl.Int32),
            pl.col(pl.Float64).cast(pl.Float32),
        )
        .with_columns(upperlevelindex=pl.col("upper_level") - 1, lowerlevelindex=pl.col("lower_level") - 1)
        .drop(["upper_level", "lower_level"])
        .with_columns(pl.col(pl.Int64).cast(pl.Int32))
    )

    if "ionstage" in linelist_lazy.collect_schema().names():
        linelist_lazy = linelist_lazy.rename({"ionstage": "ion_stage"})

    if get_ion_str:
        linelist_lazy = (
            linelist_lazy.join(get_ion_stage_roman_numeral_df().lazy(), on="ion_stage", how="left")
            .join(get_elsymbols_df().lazy(), on="atomic_number", how="left")
            .with_columns(ion_str=pl.col("elsymbol") + " " + pl.col("ion_stage_roman"))
        )

    return linelist_lazy


@lru_cache(maxsize=8)
def get_linelist_dataframe(modelpath: Path | str) -> pd.DataFrame:
    lambda_angstroms, atomic_numbers, ion_stages, upper_levels, lower_levels = read_linestatfile(
        Path(modelpath, "linestat.out")
    )

    return pd.DataFrame(
        {
            "lambda_angstroms": lambda_angstroms,
            "atomic_number": atomic_numbers,
            "ion_stage": ion_stages,
            "upperlevelindex": upper_levels,
            "lowerlevelindex": lower_levels,
        },
        dtype={
            "lambda_angstroms": float,
            "atomic_number": int,
            "ion_stage": int,
            "upperlevelindex": int,
            "lowerlevelindex": int,
        },
    )


@lru_cache(maxsize=8)
def get_npts_model(modelpath: Path) -> int:
    """Return the number of cell in the model.txt."""
    modelfilepath = (
        Path(modelpath) if Path(modelpath).is_file() else firstexisting("model.txt", folder=modelpath, tryzipped=True)
    )
    with zopen(modelfilepath) as modelfile:
        nptsline = readnoncommentline(modelfile).split(maxsplit=1)
        if len(nptsline) == 1:
            return int(nptsline[0])
        return int(nptsline[0]) * int(nptsline[1])


@lru_cache(maxsize=8)
def get_nprocs(modelpath: Path) -> int:
    """Return the number of MPI processes specified in input.txt."""
    return int(Path(modelpath, "input.txt").read_text(encoding="utf-8").split("\n")[21].split("#")[0])


@lru_cache(maxsize=8)
def get_inputparams(modelpath: Path) -> dict[str, t.Any]:
    """Return parameters specified in input.txt."""
    params: dict[str, t.Any] = {}
    with Path(modelpath, "input.txt").open("r", encoding="utf-8") as inputfile:
        params["pre_zseed"] = int(readnoncommentline(inputfile).split("#")[0])

        # number of time steps
        params["ntstep"] = int(readnoncommentline(inputfile).split("#")[0])

        # number of start and end time step
        params["itstep"], params["ftstep"] = (int(x) for x in readnoncommentline(inputfile).split("#")[0].split())

        params["tmin"], params["tmax"] = (float(x) for x in readnoncommentline(inputfile).split("#")[0].split())

        MeV_in_Hz = 2.417989242084918e20
        params["nusyn_min"], params["nusyn_max"] = (
            float(x) * MeV_in_Hz for x in readnoncommentline(inputfile).split("#")[0].split()
        )

        # number of times for synthesis
        params["nsyn_time"] = int(readnoncommentline(inputfile).split("#")[0])

        # start and end times for synthesis
        params["nsyn_time_start"], params["nsyn_time_end"] = (
            float(x) for x in readnoncommentline(inputfile).split("#")[0].split()
        )

        params["n_dimensions"] = int(readnoncommentline(inputfile).split("#")[0])

        # there are more parameters in the file that are not read yet...

    return params


@lru_cache(maxsize=16)
def get_runfolder_timesteps(folderpath: Path | str) -> tuple[int, ...]:
    """Get the set of timesteps covered by the output files in an ARTIS run folder."""
    estimfiles = sorted(Path(folderpath).glob("estimators_*.out*"))
    if not estimfiles:
        return ()

    with zopen(estimfiles[0]) as estfile:
        timesteps_contained = sorted({int(line.split()[1]) for line in estfile if line.startswith("timestep ")})
        # the first timestep of a restarted run is duplicate and should be ignored
        restart_timestep = None if 0 in timesteps_contained else timesteps_contained[0]
        return tuple(ts for ts in timesteps_contained if ts != restart_timestep)


def get_runfolders(
    modelpath: Path | str, timestep: int | None = None, timesteps: Sequence[int] | None = None
) -> Sequence[Path]:
    """Get a list of folders containing ARTIS output files from a modelpath, optionally with a timestep restriction.

    The folder list may include non-ARTIS folders if a timestep is not specified.
    """
    folderlist_all = (*sorted([child for child in Path(modelpath).iterdir() if child.is_dir()]), Path(modelpath))
    if (timestep is not None and timestep > -1) or (timesteps is not None and len(timesteps) > 0):
        folder_list_matching = []
        for folderpath in folderlist_all:
            folder_timesteps = get_runfolder_timesteps(folderpath)
            if timesteps is None and timestep is not None and timestep in folder_timesteps:
                return (folderpath,)
            if timesteps is not None and any(ts in folder_timesteps for ts in timesteps):
                folder_list_matching.append(folderpath)

        return tuple(folder_list_matching)

    return [folderpath for folderpath in folderlist_all if get_runfolder_timesteps(folderpath)]


def get_mpiranklist(
    modelpath: Path | str, modelgridindex: Iterable[int] | int | None = None, only_ranks_withgridcells: bool = False
) -> Sequence[int]:
    """Get a list of rank ids.

    - modelpath:
        pathlib.Path() to ARTIS model folder
    - modelgridindex:
        give a cell number to only return the rank number that updates this cell (and outputs its estimators)
    - only_ranks_withgridcells:
        set True to skip ranks that only update packets (i.e. that don't update any grid cells/output estimators).
    """
    if modelgridindex is None or modelgridindex == []:
        if only_ranks_withgridcells:
            return range(
                min(
                    get_nprocs(modelpath),
                    get_mpirankofcell(modelpath=modelpath, modelgridindex=get_npts_model(modelpath) - 1) + 1,
                )
            )
        return range(get_nprocs(modelpath))

    if isinstance(modelgridindex, Iterable):
        mpiranklist = set()
        for mgi in modelgridindex:
            if mgi < 0:
                if only_ranks_withgridcells:
                    return range(
                        min(
                            get_nprocs(modelpath),
                            get_mpirankofcell(modelpath=modelpath, modelgridindex=get_npts_model(modelpath) - 1) + 1,
                        )
                    )
                return range(get_nprocs(modelpath))

            mpiranklist.add(get_mpirankofcell(mgi, modelpath=modelpath))

        return sorted(mpiranklist)

    # in case modelgridindex is a single number rather than an iterable
    if modelgridindex < 0:
        return range(min(get_nprocs(modelpath), get_npts_model(modelpath)))

    return [get_mpirankofcell(modelgridindex, modelpath=modelpath)]


def get_cellsofmpirank(mpirank: int, modelpath: Path | str) -> Iterable[int]:
    """Return an iterable of the cell numbers processed by a given MPI rank."""
    npts_model = get_npts_model(modelpath)
    nprocs = get_nprocs(modelpath)

    assert mpirank < nprocs

    nblock = npts_model // nprocs
    n_leftover = npts_model % nprocs

    if mpirank < n_leftover:
        ndo = nblock + 1
        nstart = mpirank * (nblock + 1)
    else:
        ndo = nblock
        nstart = n_leftover + mpirank * nblock

    return list(range(nstart, nstart + ndo))


@lru_cache(maxsize=16)
def get_dfrankassignments(modelpath: Path | str) -> pl.DataFrame | None:
    filerankassignments = Path(modelpath, "modelgridrankassignments.out")
    if filerankassignments.is_file():
        return pl.read_csv(filerankassignments, has_header=True, separator=" ").rename(
            lambda column_name: column_name.removeprefix("#")
        )
    return None


def get_mpirankofcell(modelgridindex: int, modelpath: Path | str) -> int:
    """Return the rank number of the MPI process responsible for handling a specified cell's updating and output."""
    modelpath = Path(modelpath)
    npts_model = get_npts_model(modelpath)
    assert modelgridindex < npts_model

    dfrankassignments = get_dfrankassignments(modelpath)
    if dfrankassignments is not None:
        dfselected = dfrankassignments.filter(
            (pl.col("ndo") > 0)
            & (pl.col("nstart") <= modelgridindex)
            & ((pl.col("nstart") + pl.col("ndo") - 1) >= modelgridindex)
        )
        assert dfselected.height == 1
        return int(dfselected["rank"].item())

    nprocs = get_nprocs(modelpath)

    if nprocs > npts_model:
        mpirank = modelgridindex
    else:
        nblock = npts_model // nprocs
        n_leftover = npts_model % nprocs

        mpirank = (
            modelgridindex // (nblock + 1)
            if modelgridindex <= n_leftover * (nblock + 1)
            else n_leftover + (modelgridindex - n_leftover * (nblock + 1)) // nblock
        )

    assert modelgridindex in get_cellsofmpirank(mpirank, modelpath)

    return mpirank


def get_viewingdirectionbincount() -> int:
    return get_viewingdirection_phibincount() * get_viewingdirection_costhetabincount()


def get_viewingdirection_phibincount() -> int:
    return 10


def get_viewingdirection_costhetabincount() -> int:
    return 10


def print_theta_phi_definitions() -> None:
    print(
        "Spherical polar: x = r sinθ cosϕ, y = r sinθ sinϕ, z = r cosθ -> θ=0 is +Z and θ=π is -Z. At Z=0, ϕ=0 is +X and ϕ=π/2 is +Y"
    )


def get_phi_bins(
    usedegrees: bool,
) -> tuple[npt.NDArray[np.floating[t.Any]], npt.NDArray[np.floating[t.Any]], list[str]]:
    nphibins = get_viewingdirection_phibincount()
    # pi/2 must be an exact boundary because of the change in behaviour there
    assert nphibins % 2 == 0

    # for historical reasons, phi bins are descending and include a flip at half way
    # phisteps = [0, 1, 2, 3, 4, 9, 8, 7, 6, 5] for nphibins == 10
    phisteps = list(range(nphibins // 2)) + list(reversed(range(nphibins // 2, nphibins)))

    # set up monotonic descending phi bin boundaries
    phi_lower = np.array([2 * math.pi * (1 - (step + 1) / nphibins) for step in phisteps])
    phi_upper = np.array([2 * math.pi * (1 - step / nphibins) for step in phisteps])

    binlabels = ["" for _ in range(nphibins)]
    for phibin, phibinmonotonicdesc in enumerate(phisteps):
        if usedegrees:
            str_phi_lower = f"{phi_lower[phibinmonotonicdesc] / math.pi * 180:3.0f}°"
            str_phi_upper = f"{phi_upper[phibinmonotonicdesc] / math.pi * 180:3.0f}°"
        else:
            coeff_lower = phi_lower[phibinmonotonicdesc] / (2 * math.pi) * nphibins
            assert np.isclose(coeff_lower, round(coeff_lower), rtol=0.01), coeff_lower
            str_phi_lower = f"{round(coeff_lower)}π/{nphibins // 2}" if phi_lower[phibinmonotonicdesc] > 0.0 else "0"
            coeff_upper = phi_upper[phibinmonotonicdesc] / (2 * math.pi) * nphibins
            assert np.isclose(coeff_upper, round(coeff_upper), rtol=0.01)
            str_phi_upper = (
                f"{round(coeff_upper)}π/{nphibins // 2}" if phi_upper[phibinmonotonicdesc] < 2 * math.pi else "2π"
            )

        lower_compare = "≤" if phibin < (nphibins // 2) else "<"
        upper_compare = "≤" if phibin > (nphibins // 2) else "<"
        binlabels[phibinmonotonicdesc] = f"{str_phi_lower} {lower_compare} ϕ {upper_compare} {str_phi_upper}"

    # if nphibins == 10, then binlabels = [
    #     "9π/5 ≤ ϕ < 2π",
    #     "8π/5 ≤ ϕ < 9π/5",
    #     "7π/5 ≤ ϕ < 8π/5",
    #     "6π/5 ≤ ϕ < 7π/5",
    #     "5π/5 ≤ ϕ < 6π/5",
    #     "0 < ϕ ≤ 1π/5",
    #     "1π/5 < ϕ ≤ 2π/5",
    #     "2π/5 < ϕ ≤ 3π/5",
    #     "3π/5 < ϕ ≤ 4π/5",
    #     "4π/5 < ϕ < 5π/5",
    # ]

    return phi_lower, phi_upper, binlabels


def get_costheta_bins(
    usedegrees: bool, usepiminustheta: bool = False
) -> tuple[tuple[float, ...], tuple[float, ...], list[str]]:
    ncosthetabins = get_viewingdirection_costhetabincount()
    # the costheta bins are ordered by ascending cos θ from -1. to 1.,
    # which means that they are in descending order of theta from π to 0
    # i.e. costhetabins[0] is the θ=π or -Z axis direction
    costhetabins_lower = np.arange(-1.0, 1.0, 2.0 / ncosthetabins)
    costhetabins_upper = costhetabins_lower + 2.0 / ncosthetabins
    if usedegrees:
        if usepiminustheta:
            piminusthetabins_upper = (np.pi - np.arccos(costhetabins_upper)) / np.pi * 180
            piminusthetabins_lower = (np.pi - np.arccos(costhetabins_lower)) / np.pi * 180
            binlabels = [
                rf"{lower:.0f}° < π-θ < {upper:.0f}°"
                for lower, upper in zip(piminusthetabins_lower, piminusthetabins_upper, strict=False)
            ]
        else:
            thetabins_upper = np.arccos(costhetabins_lower) / np.pi * 180
            thetabins_lower = np.arccos(costhetabins_upper) / np.pi * 180

            binlabels = [
                f"{lower:.0f}° < θ < {upper:.0f}°"
                for lower, upper in zip(thetabins_lower, thetabins_upper, strict=False)
            ]
    else:
        binlabels = [
            f"{lower:.1f} ≤ cos θ < {upper:.1f}"
            for lower, upper in zip(costhetabins_lower, costhetabins_upper, strict=False)
        ]
    return tuple(float(x) for x in costhetabins_lower), tuple(costhetabins_upper), binlabels


def get_costhetabin_phibin_labels(usedegrees: bool) -> tuple[list[str], list[str]]:
    _, _, costhetabinlabels = get_costheta_bins(usedegrees=usedegrees)
    _, _, phibinlabels = get_phi_bins(usedegrees=usedegrees)
    return costhetabinlabels, phibinlabels


def get_opacity_condition_label(z_exclude: int) -> str:
    if z_exclude == 0:
        # normal case: all opacities sources included
        return ""
    if z_exclude == -1:
        return "no-bb"
    if z_exclude == -2:
        return "no-bf"
    return "no-es" if z_exclude == -3 else f"no-{get_elsymbol(z_exclude)}"


def get_vspec_dir_labels(modelpath: str | Path, usedegrees: bool = False) -> dict[int, str]:
    vpkt_config = get_vpkt_config(modelpath)
    dirlabels = {}
    for dirindex in range(vpkt_config["nobsdirections"]):
        phi_angle = round(vpkt_config["phi"][dirindex])
        for opacchoiceindex in range(vpkt_config["nspectraperobs"]):
            opacity_condition_label = get_opacity_condition_label(int(vpkt_config["z_excludelist"][opacchoiceindex]))
            ind_comb = vpkt_config["nspectraperobs"] * dirindex + opacchoiceindex
            cos_theta = vpkt_config["cos_theta"][dirindex]
            if usedegrees:
                theta_degrees = round(math.degrees(math.acos(cos_theta)))
                dirlabels[ind_comb] = rf"θ = {theta_degrees}°, ϕ = {phi_angle}° {opacity_condition_label}"
            else:
                dirlabels[ind_comb] = rf"cos θ = {cos_theta}, ϕ = {phi_angle}° {opacity_condition_label}"

    return dirlabels


def get_dirbin_labels(
    dirbins: npt.NDArray[np.int32] | Sequence[int] | None = None,
    modelpath: Path | str | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    usedegrees: bool = False,
    usepiminustheta: bool = False,
) -> dict[int, str]:
    """Return a dict of text labels for viewing direction bins."""
    if modelpath:
        modelpath = Path(modelpath)
        MABINS = get_viewingdirectionbincount()
        if list(modelpath.glob("*_res_00.out*")):
            # if the first direction bin file exists, check:
            # check last bin exists
            assert list(modelpath.glob(f"*_res_{MABINS - 1:02d}.out*"))
            # check one beyond does not exist
            assert not list(modelpath.glob(f"*_res_{MABINS:02d}.out*"))

    _, _, costhetabinlabels = get_costheta_bins(usedegrees=usedegrees, usepiminustheta=usepiminustheta)
    _, _, phibinlabels = get_phi_bins(usedegrees=usedegrees)

    nphibins = get_viewingdirection_phibincount()

    if dirbins is None:
        if average_over_phi:
            dirbins = np.arange(get_viewingdirection_costhetabincount()) * 10
        elif average_over_theta:
            dirbins = np.arange(nphibins)
        else:
            dirbins = np.arange(get_viewingdirectionbincount())

    angle_definitions: dict[int, str] = {}
    for dirbin in dirbins:
        dirbin_int = int(dirbin)
        if dirbin_int == -1:
            angle_definitions[dirbin_int] = ""
            continue

        costheta_index = dirbin_int // nphibins
        phi_index = dirbin_int % nphibins

        if average_over_phi:
            angle_definitions[dirbin_int] = costhetabinlabels[costheta_index]
            assert phi_index == 0
            assert not average_over_theta
        elif average_over_theta:
            angle_definitions[dirbin_int] = phibinlabels[phi_index]
            assert costheta_index == 0
        else:
            angle_definitions[dirbin_int] = f"{costhetabinlabels[costheta_index]}, {phibinlabels[phi_index]}"

    return angle_definitions


def get_multiprocessing_pool() -> multiprocessing.pool.Pool:
    """Return a multiprocessing pool that can be used to parallelize tasks."""
    # this is a workaround for to keep pytest-cov from crashing
    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()
    return multiprocessing.get_context("spawn").Pool(processes=get_config()["num_processes"])
