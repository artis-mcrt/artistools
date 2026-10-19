"""Check whether the results of an ARTIS model deserve trust, and give one row for each check.

The checks read the input files, the rank-0 logs, the estimators, deposition.out, and light_curve.out. --frompackets
also reads the escaped packets to find the Monte Carlo noise of the light curve. Each check runs alone, thus a check
that fails gives a row with the status "error", and the other checks still run. A status is "ok", "warning", "error",
or "not checked".
"""

import argparse
import itertools
import typing as t
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

from artistools.constants import Lsun_to_erg_per_s
from artistools.describemodel import get_run_versions
from artistools.estimators import get_temperature_limits
from artistools.estimators import scan_estimators
from artistools.lightcurve import find_lightcurve_file
from artistools.lightcurve import get_from_packets
from artistools.lightcurve import get_katz_integral
from artistools.lightcurve import scan_lightcurve
from artistools.logsummary import get_folder_log_ranks
from artistools.logsummary import get_log_summary
from artistools.logsummary import get_logfiles
from artistools.logsummary import get_logged_rank_count
from artistools.misc import addarg_modelpath
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import get_deposition
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_model_folder
from artistools.misc import get_run_subfolders
from artistools.misc import get_runfolders
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import path_is_dir
from artistools.misc import print_detail
from artistools.misc import print_product
from artistools.misc import print_warning
from artistools.misc.cliutils import format_range_list
from artistools.misc.remote import on_model_host

STATUSES: t.Final = ("ok", "warning", "error", "not checked")

# a fraction of the cells of a time step with Te at MINTEMP or MAXTEMP above this value gives a warning
TEMPERATURE_LIMIT_FRACTION_MAX = 0.1

# a ratio of the two Katz integrals outside this range gives a warning
KATZ_RATIO_RANGE: t.Final = (0.8, 1.2)

# a relative Monte Carlo noise of the luminosity of a time step above this value gives a warning
PACKET_RELNOISE_MAX = 0.1

# the input files that the readers need, and the readers of each one
INPUTFILE_READERS: t.Final[Mapping[str, str]] = MappingProxyType({
    "model.txt": "the readers of the cells",
    "abundances.txt": "the readers of the initial abundances",
    "compositiondata.txt": "the readers of the ions",
    "adata.txt": "the readers of the levels and the lines",
})

# the deposition column of the Katz test. Each gamma ray and particle channel adds to it, and it is the energy that
# heats the ejecta. The eps_ columns give the decay energy before the thermalisation, which is not the heating
DEPOSITIONCOLUMN = "total_dep_Lsun"


def get_runname(modelpath: Path, folder: Path) -> str:
    """Return the name of a run folder relative to the model folder, or "." for the model folder."""
    return "." if folder == modelpath else str(folder.relative_to(modelpath))


def get_names_text(names: Sequence[str]) -> str:
    """Return the names in one text, or the first and the last name of a long list, e.g. "job0 to job14"."""
    return f"{names[0]} to {names[-1]}" if len(names) > 3 else ", ".join(names)


def check_log_messages(modelpath: Path) -> tuple[str, str]:
    """Count the errors and the warnings in the rank-0 log of each run folder, and return the status and the detail."""
    logfiles = get_logfiles(modelpath, ranks=[0])
    if not logfiles:
        return "not checked", "the model folder and its subfolders hold no rank-0 log (output_0-0.txt)"

    dfkinds = get_log_summary(logfiles)
    errorcount = int(dfkinds.filter(pl.col("level") == "error")["count"].sum())
    warningcount = int(dfkinds.filter(pl.col("level") == "warning")["count"].sum())
    runnames = get_names_text([get_runname(modelpath, logfile.parent) for logfile in logfiles])
    detail = (
        f"the rank-0 logs of {len(logfiles)} run folders ({runnames}) hold {errorcount} errors and {warningcount}"
        f" warnings of {dfkinds.height} kinds"
    )
    if dfkinds.height > 0:
        # get_log_summary puts the errors and the warnings in one list, sorted by the count
        dfworst = dfkinds.sort(["level", "count"], descending=[False, True]).row(0, named=True)
        exampletext = str(dfworst["example"])
        if len(exampletext) > 100:
            exampletext = f"{exampletext[:100]}..."
        detail += f", and the most frequent {dfworst['level']} is {dfworst['count']} times '{exampletext}'"
    # almost each run of ARTIS logs warnings, e.g. of the solver of Te, thus only an error changes the status. The
    # check of the temperature limits gives the effect of the warnings of the solver of Te
    return ("error" if errorcount else "ok"), f"{detail} (artistools logsummary gives each kind)"


def check_temperature_limits(modelpath: Path) -> tuple[str, str]:
    """Find the fraction of the cells with Te at MINTEMP or MAXTEMP at each time step, and return the status and detail.

    ARTIS sets Te to a limit when the solution goes past it, thus a cell at a limit has no physical temperature.
    """
    limits = get_temperature_limits(modelpath)
    if not limits:
        return "not checked", "artisoptions.h gives no MINTEMP and no MAXTEMP"
    if not get_runfolders(modelpath):
        return "not checked", "the model has no estimator files"

    print_detail(
        "temperature limits: a cell is at a limit if its Te from the estimators is equal to "
        + " or ".join(f"{name} = {value:g} K" for name, value in limits.items())
        + " of artisoptions.h"
    )
    te = pl.col("Te").cast(pl.Float64)
    dftimesteps = (
        scan_estimators(modelpath)
        .select("timestep", "Te")
        .filter(te.is_not_null())
        .group_by("timestep")
        .agg(
            # a NaN or an infinite Te is no temperature, thus it does not count as a cell with Te
            nonfinitecount=(~te.is_finite()).sum(),
            cellcount=te.is_finite().sum(),
            # polars gives NaN == NaN as true, thus a limit that artisoptions.h does not give counts no cell
            atmintemp=(te == limits["MINTEMP"]).sum() if "MINTEMP" in limits else pl.lit(0, dtype=pl.UInt32),
            atmaxtemp=(te == limits["MAXTEMP"]).sum() if "MAXTEMP" in limits else pl.lit(0, dtype=pl.UInt32),
        )
        .with_columns(atlimitfraction=(pl.col("atmintemp") + pl.col("atmaxtemp")) / pl.col("cellcount"))
        .sort("timestep")
        .collect()
    )
    nonfinitecount = int(dftimesteps["nonfinitecount"].sum())
    dftimesteps = dftimesteps.filter(pl.col("cellcount") > 0)
    if dftimesteps.is_empty():
        if nonfinitecount:
            return "error", f"the estimator files hold {nonfinitecount} values of Te, and none of them is finite"
        return "not checked", "the estimator files hold no value of Te"

    worst = dftimesteps.sort(["atlimitfraction", "timestep"], descending=[True, False]).row(0, named=True)
    worsttimedays = get_timestep_times(modelpath, loc="mid")[worst["timestep"]]
    limittexts = [
        f"{worst[column]} ({100 * worst[column] / worst['cellcount']:.1f}%) are at {name} = {limits[name]:g} K"
        for name, column in (("MINTEMP", "atmintemp"), ("MAXTEMP", "atmaxtemp"))
        if name in limits
    ]
    overallfraction = float(
        dftimesteps.select((pl.sum("atmintemp") + pl.sum("atmaxtemp")) / pl.sum("cellcount")).item()
    )
    detail = (
        f"at the worst time step {worst['timestep']} ({worsttimedays:.2f} d), of the {worst['cellcount']} cells with"
        f" Te, {' and '.join(limittexts)}; over all {dftimesteps.height} time steps, {100 * overallfraction:.1f}% of"
        " the values of Te are at a limit, with an equal weight for each cell at each time step"
    )
    if nonfinitecount:
        # ARTIS writes a finite Te for each cell with matter, thus a NaN or an infinity shows a fault of the run
        detail += f"; {nonfinitecount} values of Te are NaN or infinite, and the fractions leave them out"
        return "error", detail
    # artisoptions.h gives the limits of the current build, and an earlier run of a different build can have other
    # limits. The options of an earlier build are not available, thus the check names the runs of other versions
    versions = {fields.get("git version") for fields in get_run_versions(modelpath).values()} - {None}
    if len(versions) > 1:
        detail += (
            f"; the runs used {len(versions)} ARTIS versions ({', '.join(sorted(v for v in versions if v))}), and"
            " artisoptions.h gives the limits of the current build alone, thus a run of an earlier build can have"
            " other limits"
        )
        return "warning", detail
    # one time step with each cell at a limit has no physical Te, also when the other time steps bring the mean down
    isbad = worst["atlimitfraction"] > TEMPERATURE_LIMIT_FRACTION_MAX
    detail += (
        f"; the status is a warning if more than {100 * TEMPERATURE_LIMIT_FRACTION_MAX:g}% of the cells of a time step"
        " are at a limit"
    )
    return ("warning" if isbad else "ok"), detail


def check_complete_timesteps(modelpath: Path) -> tuple[str, str]:
    """Find the complete time steps from deposition.out, and return the status and the detail.

    ARTIS writes a row of deposition.out at the end of each time step, thus the last row gives the last complete time
    step, as in get_escaped_arrivalrange.
    """
    try:
        dfdeposition = get_deposition(modelpath).select("timestep").collect()
    except FileNotFoundError:
        return (
            "not checked",
            "the model folder holds no deposition.out, thus a light curve plot takes each time step as complete",
        )

    tmids = get_timestep_times(modelpath, loc="mid")
    lastcomplete = dfdeposition["timestep"].max() if not dfdeposition.is_empty() else -1
    assert isinstance(lastcomplete, int)
    completetext = (
        f"deposition.out gives {lastcomplete + 1} of {len(tmids)} time steps as complete"
        f" (the last row of deposition.out is the last complete time step)"
    )
    # a deposition.out of a different model or of an earlier input.txt can hold more time steps than the model
    if lastcomplete >= len(tmids):
        return (
            "warning",
            (
                f"deposition.out has a row for time step {lastcomplete}, and the model has the time steps 0 to"
                f" {len(tmids) - 1}, thus deposition.out can belong to a different model or an earlier input.txt"
            ),
        )
    # a bad restart or a merge of two files can leave out a row, and the time step of that row is then not complete
    if missingtimesteps := sorted(set(range(lastcomplete + 1)) - set(dfdeposition["timestep"].to_list())):
        return (
            "warning",
            (
                f"{completetext}, but deposition.out has no row for the time steps"
                f" {format_range_list(missingtimesteps)}, thus their output can be partial"
            ),
        )
    if lastcomplete + 1 >= len(tmids):
        return "ok", f"{completetext}, to {tmids[-1]:.2f} d"
    return (
        "warning",
        (
            f"{completetext}; time steps {lastcomplete + 1} to {len(tmids) - 1} (from {tmids[lastcomplete + 1]:.2f} d)"
            " are not complete, thus their output is partial or empty"
        ),
    )


def check_restarts(modelpath: Path) -> tuple[str, str]:
    """Find the ARTIS version and the number of rank logs of each run, and return the status and the detail.

    A restart with a different build of ARTIS or a different number of ranks can change the results between two runs.
    """
    # a run that stops before it writes an estimator file has a log and no estimators, and it is still a restart. A
    # copy of a log, e.g. in the model folder, is not a run, thus get_logfiles gives the folders of the logs
    estimatorfolders = {folder.resolve() for folder in get_runfolders(modelpath)}
    logfolders = {logfile.parent.resolve() for logfile in get_logfiles(modelpath, None)}
    runfolders = [
        folder
        for folder in get_run_subfolders(modelpath)
        if folder.resolve() in estimatorfolders or folder.resolve() in logfolders
    ]
    if not runfolders:
        return "not checked", "the model has no run folder with estimator files or logs"

    versionofrun = {
        foldername: f"{fields.get('git version', 'unknown version')} (branch {fields.get('git branch', 'unknown')})"
        for foldername, fields in get_run_versions(modelpath).items()
        # a branch alone does not name the build, thus a run with no version gives no comparison
        if "git version" in fields
    }
    runnames = [get_runname(modelpath, folder) for folder in runfolders]
    # a run folder with no rank-0 log gives no version, and a run folder with no log gives no count
    versions = [
        (name, version)
        for name, folder in zip(runnames, runfolders, strict=True)
        if (version := versionofrun.get("." if folder == modelpath else folder.name)) is not None
    ]
    # the logs of the highest ranks can be missing, thus the rank-0 log gives the number of ranks if it can
    logcounts: list[tuple[str, int]] = []
    incompletelogs: list[str] = []
    # the names of the logs do not give the number of ranks, because the logs of the highest ranks can be missing
    unknowncounts: list[str] = []
    for name, folder in zip(runnames, runfolders, strict=True):
        logranks = get_folder_log_ranks(folder)
        loggedcount = get_logged_rank_count(folder)
        if loggedcount is not None:
            logcounts.append((name, loggedcount))
            # a log of a higher rank of an earlier run can stand in the place of a missing log, thus compare the ranks
            if missingranks := sorted(set(range(loggedcount)) - set(logranks)):
                incompletelogs.append(
                    f"{name} has no log of the ranks {format_range_list(missingranks)} of its {loggedcount} ranks"
                )
            if extraranks := sorted(set(logranks) - set(range(loggedcount))):
                incompletelogs.append(
                    f"{name} holds logs of the ranks {format_range_list(extraranks)}, above its {loggedcount} ranks,"
                    " e.g. from an earlier run with more ranks"
                )
        elif logranks:
            unknowncounts.append(name)
    changes = [
        f"the ARTIS version changes from {previousvalue} in {previousname} to {value} in {name}"
        for (previousname, previousvalue), (name, value) in itertools.pairwise(versions)
        if previousvalue != value
    ] + [
        f"the number of ranks changes from {previousvalue} in {previousname} to {value} in {name}"
        for (previousname, previousvalue), (name, value) in itertools.pairwise(logcounts)
        if previousvalue != value
    ]
    versiontext = (
        f"the logs give the ARTIS version {' or '.join(sorted({version for _, version in versions}))}"
        if versions
        else "no rank-0 log gives the ARTIS version"
    )
    logcounttext = (
        f"the rank-0 logs give {' or '.join(str(count) for count in sorted({c for _, c in logcounts}))} ranks (from"
        " the line of MPI_COMM_WORLD)"
        if logcounts
        else "no rank-0 log gives the number of ranks"
    )
    runcounttext = (
        f"{len(runfolders)} run folders with estimators or logs ({get_names_text(runnames)})"
        if len(runfolders) > 1
        else f"1 run folder with estimators or logs ({runnames[0]}), thus no restart"
    )

    # a run with no data for a comparison is not checked, thus the agreement of the other runs is not enough
    def get_some_names_text(names: Sequence[str]) -> str:
        # the names can skip runs, thus a range such as "job0 to job14" would name runs that have the data
        return ", ".join(names) if len(names) <= 3 else f"{len(names)} run folders, e.g. {', '.join(names[:3])},"

    gaps: list[str] = []
    if len(runfolders) > 1:
        versionnames = {name for name, _ in versions}
        logcountnames = {name for name, _ in logcounts} | set(unknowncounts)
        if noversion := [name for name in runnames if name not in versionnames]:
            gaps.append(
                f"{get_some_names_text(noversion)} {'gives' if len(noversion) == 1 else 'give'} no ARTIS version,"
                " thus the check cannot compare the versions"
            )
        if nologs := [name for name in runnames if name not in logcountnames]:
            gaps.append(
                f"{get_some_names_text(nologs)} {'holds' if len(nologs) == 1 else 'hold'} no rank log, thus the check"
                " cannot compare the number of ranks"
            )
        if unknowncounts:
            gaps.append(
                f"the rank-0 log of {get_some_names_text(unknowncounts)} gives no line of MPI_COMM_WORLD, thus the"
                " number of its ranks is not known, and the check cannot compare it"
            )
    gaps.extend(incompletelogs)
    summary = f"{runcounttext}: {versiontext}, and {logcounttext}"
    if changes or gaps:
        return "warning", f"{summary}; {'; '.join([*changes, *gaps])}"
    return "ok", summary


def check_energy(modelpath: Path) -> tuple[str, str]:
    """Compare the escaped luminosity with the deposition by the Katz integral test, and return the status and detail.

    Katz B., Kushnir D., Dong S. (2013), arXiv:1301.6766, give the exact relation
    t E(t) - t0 E(t0) = ∫ t Q dt - ∫ t L dt, with the thermal energy E of the ejecta. Thus the two integrals agree if
    t E(t) is small at the start and at the end.
    """
    try:
        lightcurvefile = find_lightcurve_file(modelpath)
    except FileNotFoundError:
        return "not checked", "the Katz test needs light_curve.out, and the model folder holds no light_curve.out"
    try:
        dfdeposition = get_deposition(modelpath).collect()
    except FileNotFoundError:
        return "not checked", "the Katz test needs deposition.out, and the model folder holds no deposition.out"
    if DEPOSITIONCOLUMN not in dfdeposition.columns:
        return "not checked", f"deposition.out has no column {DEPOSITIONCOLUMN}"
    if dfdeposition.height < 2:
        return "not checked", "deposition.out has fewer than two complete time steps"

    print_detail(
        f"energy: Q is {DEPOSITIONCOLUMN} of deposition.out, the energy that the gamma rays and the particles deposit"
        " in the ejecta. The eps_ columns give the decay energy before the thermalisation, thus Q does not use them"
    )
    print_detail(
        "energy: L is the angle average of light_curve.out, which holds the escaped r-packets of all wavelengths."
        " The Katz test (Katz B., Kushnir D., Dong S., 2013, arXiv:1301.6766) compares ∫ t L dt with ∫ t Q dt"
    )
    tmindays, tmaxdays = (
        float(value)
        for value in dfdeposition.select(pl.min("tmid_days").alias("tmin"), pl.max("tmid_days").alias("tmax")).row(0)
    )
    # light_curve.out and deposition.out give the mid times of the time steps with a different number of digits
    dflightcurve = (
        scan_lightcurve(lightcurvefile)[-1]
        .filter(pl.col("time_days").is_between(tmindays * (1 - 1e-4), tmaxdays * (1 + 1e-4)))
        .select("time_days", "luminosity_erg/s")
        .collect()
    )
    dffinitelc = dflightcurve.filter(pl.col("time_days").is_finite())
    if dffinitelc.height < 2:
        return "not checked", f"light_curve.out has fewer than two times from {tmindays:.2f} to {tmaxdays:.2f} d"
    lumintegral, lumomitted_days = get_katz_integral(dffinitelc["time_days"], dffinitelc["luminosity_erg/s"])
    # a time with no row of light_curve.out has no NaN either, thus the integral of L does not cover it
    lcfirstday, lclastday = (
        float(value) for value in dffinitelc.select(pl.min("time_days").alias("first"), pl.max("time_days")).row(0)
    )
    lumomitted_days += (lcfirstday - tmindays if lcfirstday > tmindays * (1 + 1e-4) else 0.0) + (
        tmaxdays - lclastday if lclastday < tmaxdays * (1 - 1e-4) else 0.0
    )
    # a mid time of deposition.out with no row of light_curve.out inside the range makes one trapezoid span the gap
    lctimes = dffinitelc["time_days"].to_numpy()
    bridgedtimes = [
        float(tmid)
        for tmid in dfdeposition["tmid_days"].to_numpy()
        if lcfirstday < tmid < lclastday and not np.isclose(lctimes, tmid, rtol=1e-4, atol=0.0).any()
    ]
    depintegral, depomitted_days = get_katz_integral(
        dfdeposition["tmid_days"], dfdeposition[DEPOSITIONCOLUMN] * Lsun_to_erg_per_s
    )
    ratio = lumintegral / depintegral if depintegral > 0.0 else np.nan
    detail = (
        f"∫ t L dt = {lumintegral:.3e} erg s from light_curve.out and ∫ t Q dt = {depintegral:.3e} erg s from"
        f" {DEPOSITIONCOLUMN} of deposition.out, with the trapezoid rule over the mid times of {dfdeposition.height}"
        f" complete time steps from {tmindays:.2f} to {tmaxdays:.2f} d, give the ratio L/Q = {ratio:.3f}"
    )
    if lumomitted_days or depomitted_days:
        # a missing rate is not a zero, thus the integrals leave out its trapezoids, and the ratio can then be wrong
        detail += (
            f". The integrals leave out {lumomitted_days:.3g} d of L and {depomitted_days:.3g} d of Q with no value"
            " or no row, thus the ratio covers different times"
        )
        return "warning", detail
    if bridgedtimes:
        detail += (
            f". light_curve.out has no row at {len(bridgedtimes)} mid times of deposition.out, e.g. at"
            f" {bridgedtimes[0]:.2f} d, thus the trapezoid rule takes a straight line of L across each gap"
        )
        return "warning", detail
    isinrange = KATZ_RATIO_RANGE[0] <= ratio <= KATZ_RATIO_RANGE[1]
    if not isinrange:
        # the ratio can be correct physics, thus the detail gives the condition of the relation
        detail += (
            ". The relation needs a small thermal energy t E at the start and at the end of the range, thus a range"
            " that starts with hot ejecta or ends while the ejecta are optically thick gives a ratio far from 1 with"
            " no fault in the model"
        )
    return ("ok" if isinrange else "warning"), detail


def check_packet_noise(modelpath: Path, frompackets: bool) -> tuple[str, str]:
    """Find the largest relative Monte Carlo noise of the bolometric light curve, and return the status and detail."""
    if not frompackets:
        return "not checked", "give --frompackets to read the packets"

    print_detail(
        "packet noise: the relative noise of a time step is sqrt(Σ e²) / Σ e over the energies e of its escaped"
        " r-packets, for the angle average"
    )
    dfalltimesteps = (
        get_from_packets(modelpath)[-1]
        .select("timestep", "time_days", "packetcount", "luminosity_Lsun", "relnoise")
        .collect()
    )
    dflightcurve = dfalltimesteps.filter(pl.col("relnoise").is_not_null())
    if dflightcurve.is_empty():
        return "warning", "no time step has an escaped r-packet"
    # a time step with no escaped packet has no sampled luminosity and no noise, thus the check names it
    nopackettimesteps = dfalltimesteps.filter(pl.col("relnoise").is_null())["timestep"].to_list()

    worst = dflightcurve.sort("relnoise", descending=True).row(0, named=True)
    peak = dflightcurve.sort("luminosity_Lsun", descending=True).row(0, named=True)
    noisycount = dflightcurve.filter(pl.col("relnoise") > PACKET_RELNOISE_MAX).height
    detail = (
        f"the largest relative noise is {100 * worst['relnoise']:.1f}% at {worst['time_days']:.2f} d"
        f" (with {worst['packetcount']} escaped packets), {noisycount} of {dflightcurve.height} time steps with"
        f" packets are above {100 * PACKET_RELNOISE_MAX:g}%, and the noise at the peak ({peak['time_days']:.2f} d) is"
        f" {100 * peak['relnoise']:.2f}%"
    )
    if nopackettimesteps:
        detail += (
            f"; the time steps {format_range_list(nopackettimesteps)} have no escaped r-packet, thus their luminosity"
            " and their noise are not known"
        )
        return "warning", detail
    return ("warning" if worst["relnoise"] > PACKET_RELNOISE_MAX else "ok"), detail


def check_input_files(modelpath: Path) -> tuple[str, str]:
    """Find the input files that the readers of artistools need, and return the status and the detail."""
    missing = [
        f"{filename} ({readers})"
        for filename, readers in INPUTFILE_READERS.items()
        if firstexisting_or_none(filename, folder=modelpath, search_subfolders=False) is None
    ]
    if not missing:
        return "ok", f"the model folder holds {', '.join(INPUTFILE_READERS)}"
    return "warning", f"the model folder holds no {'; no '.join(missing)}"


def check_caches(modelpath: Path) -> tuple[str, str]:
    """Count the parquet caches and the files of conversions that stopped, and return the status and the detail."""
    from artistools.cache import find_cache_files

    caches, leftovers = find_cache_files(modelpath)
    sizemb = sum(path.stat().st_size for path in caches) / 1e6
    detail = f"{len(caches)} parquet caches of {sizemb:.1f} MB (artistools cache list gives them)"
    if leftovers:
        return "warning", (
            f"{detail}, and {len(leftovers)} files of conversions that stopped, e.g. {leftovers[0].name}."
            " artistools cache delete deletes them"
        )
    return "ok", detail


def check_arrival_range(modelpath: Path) -> tuple[str, str]:
    """Find the range of the arrival times where the light of all the ejecta reaches the observer."""
    _, validstart_days, validend_days = get_escaped_arrivalrange(modelpath)
    if validstart_days is None or validend_days is None:
        return "warning", (
            "no arrival time has the light of all the ejecta, because the light travel time across the ejecta is"
            " longer than the time of the run"
        )
    return "ok", (
        f"the light of all the ejecta reaches the observer from {validstart_days:.2f} to {validend_days:.2f} d."
        " The commands that read packets keep this range, and --plotinvalidpart also takes the other times"
    )


def run_check(checkfunction: Callable[[], tuple[str, str]]) -> tuple[str, str]:
    """Run one check, and return its status and its detail. A check that fails gives the status "error"."""
    try:
        status, detail = checkfunction()
    # a reader stops with exit_with_error, which prints the message, and the other checks must still run
    except SystemExit as exc:
        return "error", f"the check stopped with the exit status {exc.code}, and the message above gives the reason"
    # the row gives the error of one check, and the other checks still run
    except Exception as exc:  # ruff:ignore[blind-except]
        return "error", " ".join(f"{type(exc).__name__}: {exc}".split())
    assert status in STATUSES
    # a message of an exception can hold line breaks, and a detail is one line of the table
    return status, " ".join(detail.split())


@on_model_host
def get_check_results(modelpath: Path, frompackets: bool = False) -> pl.DataFrame:
    """Return one row for each check, with the columns check, status, and detail.

    The host of a remote model runs the checks, because each check reads files of the model folder.
    """
    modelpath = get_model_folder(modelpath)
    if not modelpath.is_dir():
        msg = f"The model folder {modelpath} does not exist"
        raise FileNotFoundError(msg)
    checks: list[tuple[str, Callable[[], tuple[str, str]]]] = [
        ("input files", partial(check_input_files, modelpath)),
        ("caches", partial(check_caches, modelpath)),
        ("log messages", partial(check_log_messages, modelpath)),
        ("temperature limits", partial(check_temperature_limits, modelpath)),
        ("complete time steps", partial(check_complete_timesteps, modelpath)),
        ("restarts", partial(check_restarts, modelpath)),
        ("energy", partial(check_energy, modelpath)),
        ("arrival times", partial(check_arrival_range, modelpath)),
        ("packet noise", partial(check_packet_noise, modelpath, frompackets)),
    ]
    rows = [(checkname, *run_check(checkfunction)) for checkname, checkfunction in checks]
    return pl.DataFrame(rows, schema={"check": pl.String, "status": pl.String, "detail": pl.String}, orient="row")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())
    parser.add_argument(
        "--frompackets",
        action="store_true",
        help="Also read the escaped packets to find the Monte Carlo noise of the light curve. This read is slow",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Check whether the results of an ARTIS model deserve trust, and give one row for each check."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelfolder = get_model_folder(args.modelpath)
    if not path_is_dir(modelfolder):
        exit_with_error(f"The model folder {modelfolder} does not exist", "Give a model folder")
    dfresults = get_check_results(modelfolder, frompackets=args.frompackets)
    print_product(args, dfresults)
    for checkname, status, detail in dfresults.filter(pl.col("status").is_in(["warning", "error"])).iter_rows():
        print_warning(f"{checkname} ({status}): {detail}")
