#!/usr/bin/env python3
"""Functions for reading and processing estimator files.

Examples are temperatures, populations, and heating/cooling rates.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import multiprocessing
import sys
import typing as t
from collections import namedtuple
from functools import partial
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from typeguard import typechecked

import artistools as at


def get_variableunits(key: str | None = None) -> str | dict[str, str]:
    variableunits = {
        "time": "days",
        "gamma_NT": "/s",
        "gamma_R_bfest": "/s",
        "TR": "K",
        "Te": "K",
        "TJ": "K",
        "nne": "e-/cm3",
        "heating": "erg/s/cm3",
        "heating_dep/total_dep": "Ratio",
        "cooling": "erg/s/cm3",
        "velocity": "km/s",
        "velocity_outer": "km/s",
    }
    return variableunits[key] if key else variableunits


def get_variablelongunits(key: str | None = None) -> str | dict[str, str]:
    variablelongunits = {
        "heating_dep/total_dep": "",
        "TR": "Temperature [K]",
        "Te": "Temperature [K]",
        "TJ": "Temperature [K]",
    }
    return variablelongunits[key] if key else variablelongunits


def get_dictlabelreplacements(key: str | None = None) -> str | dict[str, str]:
    dictlabelreplacements = {
        "lognne": "Log nne",
        "Te": "T$_e$",
        "TR": "T$_R$",
        "gamma_NT": r"$\Gamma_{\rm non-thermal}$ [s$^{-1}$]",
        "gamma_R_bfest": r"$\Gamma_{\rm phot}$ [s$^{-1}$]",
        "heating_dep/total_dep": "Heating fraction",
    }
    return dictlabelreplacements[key] if key else dictlabelreplacements


def apply_filters(
    xlist: list[float] | np.ndarray, ylist: list[float] | np.ndarray, args: argparse.Namespace
) -> tuple[list[float] | np.ndarray, list[float] | np.ndarray]:
    filterfunc = at.get_filterfunc(args)

    if filterfunc is not None:
        ylist = filterfunc(ylist)

    return xlist, ylist


def get_ionrecombrates_fromfile(filename: Path | str) -> pd.DataFrame:
    """WARNING: copy pasted from artis-atomic! replace with a package import soon ionstage is the lower ion stage."""
    print(f"Reading {filename}")

    header_row = []
    with Path(filename).open() as filein:
        while True:
            line = filein.readline()
            if line.strip().startswith("TOTAL RECOMBINATION RATE"):
                line = filein.readline()
                line = filein.readline()
                header_row = filein.readline().strip().replace(" n)", "-n)").split()
                break

        if not header_row:
            print("ERROR: no header found")
            sys.exit()

        index_logt = header_row.index("log(T)")
        index_low_n = header_row.index("RRC(low-n)")
        index_tot = header_row.index("RRC(total)")

        recomb_tuple = namedtuple("recomb_tuple", ["logT", "RRC_low_n", "RRC_total"])
        records = []
        for line in filein:
            if row := line.split():
                if len(row) != len(header_row):
                    print("Row contains wrong number of items for header:")
                    print(header_row)
                    print(row)
                    sys.exit()
                records.append(recomb_tuple(*[float(row[index]) for index in [index_logt, index_low_n, index_tot]]))

    return pd.DataFrame.from_records(records, columns=recomb_tuple._fields)


def get_units_string(variable: str) -> str:
    if variable in get_variableunits():
        return f" [{get_variableunits(variable)}]"
    if variable.split("_")[0] in get_variableunits():
        return f' [{get_variableunits(variable.split("_")[0])}]'
    return ""


def parse_estimfile(
    estfilepath: Path,
    modelpath: Path,
    get_ion_values: bool = True,
    get_heatingcooling: bool = True,
    skip_emptycells: bool = False,
) -> t.Iterator[tuple[int, int, dict]]:  # pylint: disable=unused-argument
    """Generate timestep, modelgridindex, dict from estimator file."""
    with at.zopen(estfilepath) as estimfile:
        timestep: int = -1
        modelgridindex: int = -1
        estimblock: dict[t.Any, t.Any] = {}
        for line in estimfile:
            row: list[str] = line.split()
            if not row:
                continue

            if row[0] == "timestep":
                # yield the previous block before starting a new one
                if (
                    timestep >= 0
                    and modelgridindex >= 0
                    and (not skip_emptycells or not estimblock.get("emptycell", True))
                ):
                    yield timestep, modelgridindex, estimblock

                timestep = int(row[1])
                # if timestep > itstep:
                #     print(f"Dropping estimator data from timestep {timestep} and later (> itstep {itstep})")
                #     # itstep in input.txt is updated by ARTIS at every timestep, so the data beyond here
                #     # could be half-written to disk and cause parsing errors
                #     return

                modelgridindex = int(row[3])
                emptycell = row[4] == "EMPTYCELL"
                estimblock = {"emptycell": emptycell}
                if not emptycell:
                    # will be TR, Te, W, TJ, nne
                    for variablename, value in zip(row[4::2], row[5::2]):
                        estimblock[variablename] = float(value)
                    estimblock["lognne"] = math.log10(estimblock["nne"]) if estimblock["nne"] > 0 else float("-inf")

            elif row[1].startswith("Z=") and get_ion_values:
                variablename = row[0]
                if row[1].endswith("="):
                    atomic_number = int(row[2])
                    startindex = 3
                else:
                    atomic_number = int(row[1].split("=")[1])
                    startindex = 2

                estimblock.setdefault(variablename, {})

                for ion_stage_str, value in zip(row[startindex::2], row[startindex + 1 :: 2]):
                    if ion_stage_str.strip() == "(or":
                        continue

                    value_thision = float(value.rstrip(","))

                    if ion_stage_str.strip() == "SUM:":
                        estimblock[variablename][atomic_number] = value_thision
                        continue

                    try:
                        ion_stage = int(ion_stage_str.rstrip(":"))
                    except ValueError:
                        if variablename == "populations" and ion_stage_str.startswith(at.get_elsymbol(atomic_number)):
                            estimblock[variablename][ion_stage_str.rstrip(":")] = float(value)
                        else:
                            print(ion_stage_str, at.get_elsymbol(atomic_number))
                            print(f"Cannot parse row: {row}")
                        continue

                    estimblock[variablename][(atomic_number, ion_stage)] = value_thision

                    if variablename in ["Alpha_R*nne", "AlphaR*nne"]:
                        estimblock.setdefault("Alpha_R", {})
                        estimblock["Alpha_R"][(atomic_number, ion_stage)] = (
                            value_thision / estimblock["nne"] if estimblock["nne"] > 0.0 else float("inf")
                        )

                    else:  # variablename == 'populations':
                        # contribute the ion population to the element population
                        estimblock[variablename].setdefault(atomic_number, 0.0)
                        estimblock[variablename][atomic_number] += value_thision

                if variablename == "populations":
                    # contribute the element population to the total population
                    estimblock["populations"].setdefault("total", 0.0)
                    estimblock["populations"]["total"] += estimblock["populations"][atomic_number]
                    estimblock.setdefault("nntot", 0.0)
                    estimblock["nntot"] += estimblock["populations"][atomic_number]

            elif row[0] == "heating:" and get_heatingcooling:
                for heatingtype, value in zip(row[1::2], row[2::2]):
                    key = heatingtype if heatingtype.startswith("heating_") else "heating_" + heatingtype
                    estimblock[key] = float(value)

                if "heating_gamma/gamma_dep" in estimblock and estimblock["heating_gamma/gamma_dep"] > 0:
                    estimblock["gamma_dep"] = estimblock["heating_gamma"] / estimblock["heating_gamma/gamma_dep"]
                elif "heating_dep/total_dep" in estimblock and estimblock["heating_dep/total_dep"] > 0:
                    estimblock["total_dep"] = estimblock["heating_dep"] / estimblock["heating_dep/total_dep"]

            elif row[0] == "cooling:" and get_heatingcooling:
                for coolingtype, value in zip(row[1::2], row[2::2]):
                    estimblock["cooling_" + coolingtype] = float(value)

    # reached the end of file
    if timestep >= 0 and modelgridindex >= 0 and (not skip_emptycells or not estimblock.get("emptycell", True)):
        yield timestep, modelgridindex, estimblock


def read_estimators_from_file(
    folderpath: Path | str,
    modelpath: Path,
    arr_velocity_outer: t.Sequence[float] | None,
    mpirank: int,
    printfilename: bool = False,
    get_ion_values: bool = True,
    get_heatingcooling: bool = True,
    skip_emptycells: bool = False,
) -> dict[tuple[int, int], t.Any]:
    estimators_thisfile = {}
    estimfilename = f"estimators_{mpirank:04d}.out"
    try:
        estfilepath = at.firstexisting(estimfilename, folder=folderpath, tryzipped=True)
    except FileNotFoundError:
        # not worth printing an error, because ranks with no cells to update do not produce an estimator file
        # print(f'Warning: Could not find {estfilepath.relative_to(modelpath.parent)}')
        return {}

    if printfilename:
        filesize = Path(estfilepath).stat().st_size / 1024 / 1024
        print(f"Reading {estfilepath.relative_to(modelpath.parent)} ({filesize:.2f} MiB)")

    for fileblock_timestep, fileblock_modelgridindex, file_estimblock in parse_estimfile(
        estfilepath,
        modelpath,
        get_ion_values=get_ion_values,
        get_heatingcooling=get_heatingcooling,
        skip_emptycells=skip_emptycells,
    ):
        if arr_velocity_outer is not None:
            file_estimblock["velocity_outer"] = arr_velocity_outer[fileblock_modelgridindex]
            file_estimblock["velocity"] = file_estimblock["velocity_outer"]

        estimators_thisfile[(fileblock_timestep, fileblock_modelgridindex)] = file_estimblock

    return estimators_thisfile


@typechecked
def read_estimators(
    modelpath: Path | str = Path(),
    modelgridindex: None | int | t.Sequence[int] = None,
    timestep: None | int | t.Sequence[int] = None,
    mpirank: int | None = None,
    runfolder: None | str | Path = None,
    get_ion_values: bool = True,
    get_heatingcooling: bool = True,
    skip_emptycells: bool = False,
    add_velocity: bool = True,
) -> dict[tuple[int, int], dict]:
    """Read estimator files into a nested dictionary structure.

    Speed it up by only retrieving estimators for a particular timestep(s) or modelgrid cells.
    """
    modelpath = Path(modelpath)
    match_modelgridindex: t.Collection[int]
    if modelgridindex is None:
        match_modelgridindex = []
    elif isinstance(modelgridindex, int):
        match_modelgridindex = (modelgridindex,)
    else:
        match_modelgridindex = tuple(modelgridindex)

    if -1 in match_modelgridindex:
        match_modelgridindex = []

    match_timestep: t.Collection[int]
    if timestep is None:
        match_timestep = []
    elif isinstance(timestep, int):
        match_timestep = (timestep,)
    else:
        match_timestep = tuple(timestep)

    if not Path(modelpath).exists() and Path(modelpath).parts[0] == "codecomparison":
        return at.codecomparison.read_reference_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")

    arr_velocity_outer = None
    if add_velocity:
        modeldata, _ = at.inputmodel.get_modeldata(modelpath, getheadersonly=True)
        if "velocity_outer" in modeldata.columns:
            modeldata, _ = at.inputmodel.get_modeldata(modelpath)
            arr_velocity_outer = tuple(float(v) for v in modeldata["velocity_outer"].to_numpy())

    mpiranklist = (
        at.get_mpiranklist(modelpath, modelgridindex=match_modelgridindex, only_ranks_withgridcells=True)
        if mpirank is None
        else [mpirank]
    )

    runfolders = at.get_runfolders(modelpath, timesteps=match_timestep) if runfolder is None else [Path(runfolder)]

    printfilename = len(mpiranklist) < 10

    estimators: dict[tuple[int, int], dict] = {}
    for folderpath in runfolders:
        if not printfilename:
            print(
                f"Reading {len(list(mpiranklist))} estimator files in {folderpath.relative_to(Path(modelpath).parent)}"
            )

        processfile = partial(
            read_estimators_from_file,
            folderpath,
            modelpath,
            arr_velocity_outer,
            get_ion_values=get_ion_values,
            get_heatingcooling=get_heatingcooling,
            printfilename=printfilename,
            skip_emptycells=skip_emptycells,
        )

        if at.get_config()["num_processes"] > 1:
            with multiprocessing.get_context("spawn").Pool(processes=at.get_config()["num_processes"]) as pool:
                arr_rankestimators = pool.map(processfile, mpiranklist)
                pool.close()
                pool.join()
                pool.terminate()
        else:
            arr_rankestimators = [processfile(rank) for rank in mpiranklist]

        for mpirank, estimators_thisfile in zip(mpiranklist, arr_rankestimators):
            dupekeys = sorted([k for k in estimators_thisfile if k in estimators])
            for k in dupekeys:
                # dropping the lowest timestep is normal for restarts. Only warn about other cases
                if k[0] != dupekeys[0][0]:
                    filepath = Path(folderpath, f"estimators_{mpirank:04d}.out")
                    print(
                        f"WARNING: Duplicate estimator block for (timestep, mgi) key {k}. "
                        f"Dropping block from {filepath}"
                    )

                del estimators_thisfile[k]

            estimators |= estimators_thisfile

    return estimators


def get_averaged_estimators(
    modelpath: Path | str,
    estimators: dict[tuple[int, int], dict],
    timesteps: int | t.Sequence[int],
    modelgridindex: int,
    keys: str | list,
    avgadjcells: int = 0,
) -> t.Any | float:
    """Get the average of estimators[(timestep, modelgridindex)][keys[0]]...[keys[-1]] across timesteps."""
    if isinstance(keys, str):
        keys = [keys]

    # reduce(lambda d, k: d[k], keys, dictionary) returns dictionary[keys[0]][keys[1]]...[keys[-1]]
    # applying all keys in the keys list

    # if single timestep, no averaging needed
    if isinstance(timesteps, int):
        return reduce(lambda d, k: d[k], [(timesteps, modelgridindex), *keys], estimators)

    firsttimestepvalue = reduce(lambda d, k: d[k], [(timesteps[0], modelgridindex), *keys], estimators)
    if isinstance(firsttimestepvalue, dict):
        return {
            k: get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, [*keys, k])
            for k in firsttimestepvalue
        }

    tdeltas = at.get_timestep_times(modelpath, loc="delta")
    valuesum = 0
    tdeltasum = 0
    for timestep, tdelta in zip(timesteps, tdeltas):
        for mgi in range(modelgridindex - avgadjcells, modelgridindex + avgadjcells + 1):
            with contextlib.suppress(KeyError):
                valuesum += reduce(lambda d, k: d[k], [(timestep, mgi), *keys], estimators) * tdelta
                tdeltasum += tdelta
    return valuesum / tdeltasum

    # except KeyError:
    #     if (timestep, modelgridindex) in estimators:
    #         print(f'Unknown x variable: {xvariable} for timestep {timestep} in cell {modelgridindex}')
    #     else:
    #         print(f'No data for cell {modelgridindex} at timestep {timestep}')
    #     print(estimators[(timestep, modelgridindex)])
    #     sys.exit()


def get_averageionisation(populations: dict[t.Any, float], atomic_number: int) -> float:
    free_electron_weighted_pop_sum = 0.0
    found = False
    popsum = 0.0
    for key in populations:
        if isinstance(key, tuple) and key[0] == atomic_number:
            found = True
            ion_stage = key[1]
            free_electron_weighted_pop_sum += populations[key] * (ion_stage - 1)
            popsum += populations[key]

    if not found:
        return float("NaN")

    return free_electron_weighted_pop_sum / populations[atomic_number]


def get_averageexcitation(
    modelpath: Path, modelgridindex: int, timestep: int, atomic_number: int, ion_stage: int, T_exc: float
) -> float:
    dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)
    adata = at.atomic.get_levels(modelpath)
    ionlevels = adata.query("Z == @atomic_number and ion_stage == @ion_stage").iloc[0].levels

    energypopsum = 0
    ionpopsum = 0
    if dfnltepops.empty:
        return float("NaN")

    dfnltepops_ion = dfnltepops.query(
        "modelgridindex==@modelgridindex and timestep==@timestep and Z==@atomic_number & ion_stage==@ion_stage"
    )

    k_b = 8.617333262145179e-05  # eV / K

    ionpopsum = dfnltepops_ion.n_NLTE.sum()
    energypopsum = (
        dfnltepops_ion[dfnltepops_ion.level >= 0].eval("@ionlevels.iloc[level].energy_ev.values * n_NLTE").sum()
    )

    with contextlib.suppress(IndexError):  # no superlevel with cause IndexError
        superlevelrow = dfnltepops_ion[dfnltepops_ion.level < 0].iloc[0]
        levelnumber_sl = dfnltepops_ion.level.max() + 1

        energy_boltzfac_sum = (
            ionlevels.iloc[levelnumber_sl:].eval("energy_ev * g * exp(- energy_ev / @k_b / @T_exc)").sum()
        )

        boltzfac_sum = ionlevels.iloc[levelnumber_sl:].eval("g * exp(- energy_ev / @k_b / @T_exc)").sum()
        # adjust to the actual superlevel population from ARTIS
        energypopsum += energy_boltzfac_sum * superlevelrow.n_NLTE / boltzfac_sum
    return energypopsum / ionpopsum


def get_partiallycompletetimesteps(estimators: dict[tuple[int, int], dict]) -> list[int]:
    """During a simulation, some estimator files can contain information for some cells but not others
    for the current timestep.
    """
    timestepcells: dict[int, list[int]] = {}
    all_mgis = set()
    for nts, mgi in estimators:
        if nts not in timestepcells:
            timestepcells[nts] = []
        timestepcells[nts].append(mgi)
        all_mgis.add(mgi)

    return [nts for nts, mgilist in timestepcells.items() if len(mgilist) < len(all_mgis)]
