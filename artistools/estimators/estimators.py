#!/usr/bin/env python3
"""Functions for reading and processing estimator files.

Examples are temperatures, populations, and heating/cooling rates.
"""

import argparse
import contextlib
import itertools
import math
import multiprocessing
import sys
import typing as t
from collections import namedtuple
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs

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
        "beta": "v/c",
        "vel_r_max_kmps": "km/s",
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


def get_dictlabelreplacements() -> dict[str, str]:
    return {
        "lognne": r"Log n$_{\rm e}$",
        "Te": r"T$_{\rm e}$",
        "TR": r"T$_{\rm R}$",
        "TJ": r"T$_{\rm J}$",
        "gamma_NT": r"$\Gamma_{\rm non-thermal}$ [s$^{-1}$]",
        "gamma_R_bfest": r"$\Gamma_{\rm phot}$ [s$^{-1}$]",
        "heating_dep/total_dep": "Heating fraction",
    }


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


def read_estimators_from_file(
    estfilepath: Path | str,
    printfilename: bool = False,
    skip_emptycells: bool = True,
) -> pl.DataFrame:
    if printfilename:
        estfilepath = Path(estfilepath)
        filesize = estfilepath.stat().st_size / 1024 / 1024
        print(f"  Reading {estfilepath.relative_to(estfilepath.parent.parent)} ({filesize:.2f} MiB)")

    estimblocklist: list[dict[str, t.Any]] = []
    with at.zopen(estfilepath) as estimfile:
        timestep: int | None = None
        modelgridindex: int | None = None
        estimblock: dict[str, t.Any] = {}
        for line in estimfile:
            row: list[str] = line.split()
            if not row:
                continue

            if row[0] == "timestep":
                # yield the previous block before starting a new one
                if (
                    timestep is not None
                    and modelgridindex is not None
                    and (not skip_emptycells or not estimblock.get("emptycell", True))
                ):
                    estimblock["timestep"] = timestep
                    estimblock["modelgridindex"] = modelgridindex
                    estimblocklist.append(estimblock)

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

            elif row[1].startswith("Z="):
                variablename = row[0]
                if row[1].endswith("="):
                    atomic_number = int(row[2])
                    startindex = 3
                else:
                    atomic_number = int(row[1].split("=")[1])
                    startindex = 2
                elsymbol = at.get_elsymbol(atomic_number)

                for ionstage_str, value in zip(row[startindex::2], row[startindex + 1 :: 2]):
                    ionstage_str_strip = ionstage_str.strip()
                    if ionstage_str_strip == "(or":
                        continue

                    value_thision = float(value.rstrip(","))

                    if ionstage_str_strip == "SUM:":
                        estimblock[f"nnelement_{elsymbol}"] = value_thision
                        continue

                    try:
                        ionstage = int(ionstage_str.rstrip(":"))
                    except ValueError:
                        if variablename == "populations" and ionstage_str.startswith(elsymbol):
                            estimblock[f"nniso_{ionstage_str.rstrip(':')}"] = float(value)
                        else:
                            print(ionstage_str, elsymbol)
                            print(f"Cannot parse row: {row}")
                        continue

                    ionstr = at.get_ionstring(atomic_number, ionstage, sep="_", style="spectral")
                    estimblock[f"{'nnion' if variablename=='populations' else variablename}_{ionstr}"] = value_thision

                    if variablename in {"Alpha_R*nne", "AlphaR*nne"}:
                        estimblock[f"Alpha_R_{ionstr}"] = (
                            value_thision / estimblock["nne"] if estimblock["nne"] > 0.0 else float("inf")
                        )

                    elif variablename == "populations":
                        estimblock.setdefault(f"nnelement_{elsymbol}", 0.0)
                        estimblock[f"nnelement_{elsymbol}"] += value_thision

                if variablename == "populations":
                    # contribute the element population to the total population
                    estimblock.setdefault("nntot", 0.0)
                    estimblock["nntot"] += estimblock[f"nnelement_{elsymbol}"]

            elif row[0] == "heating:":
                for heatingtype, value in zip(row[1::2], row[2::2]):
                    key = heatingtype if heatingtype.startswith("heating_") else f"heating_{heatingtype}"
                    estimblock[key] = float(value)

                if "heating_gamma/gamma_dep" in estimblock and estimblock["heating_gamma/gamma_dep"] > 0:
                    estimblock["gamma_dep"] = estimblock["heating_gamma"] / estimblock["heating_gamma/gamma_dep"]
                elif "heating_dep/total_dep" in estimblock and estimblock["heating_dep/total_dep"] > 0:
                    estimblock["total_dep"] = estimblock["heating_dep"] / estimblock["heating_dep/total_dep"]

            elif row[0] == "cooling:":
                for coolingtype, value in zip(row[1::2], row[2::2]):
                    estimblock[f"cooling_{coolingtype}"] = float(value)

    # reached the end of file
    if (
        timestep is not None
        and modelgridindex is not None
        and (not skip_emptycells or not estimblock.get("emptycell", True))
    ):
        estimblock["timestep"] = timestep
        estimblock["modelgridindex"] = modelgridindex
        estimblocklist.append(estimblock)

    return pl.DataFrame(estimblocklist).with_columns(
        pl.col(pl.Int64).cast(pl.Int32), pl.col(pl.Float64).cast(pl.Float32)
    )


def batched_it(iterable, n):
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


def read_estimators_in_folder_polars(
    modelpath: Path,
    folderpath: Path,
    match_modelgridindex: None | t.Sequence[int],
) -> pl.LazyFrame:
    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=match_modelgridindex, only_ranks_withgridcells=True)

    mpirank_groups = list(batched_it(list(mpiranklist), 100))
    group_parquetfiles = [
        folderpath / f"estimators_{mpigroup[0]:04d}_{mpigroup[-1]:04d}.out.parquet.tmp" for mpigroup in mpirank_groups
    ]
    for mpigroup, parquetfilename in zip(mpirank_groups, group_parquetfiles):
        if not parquetfilename.exists():
            print(f"{parquetfilename.relative_to(modelpath.parent)} does not exist")
            estfilepaths = []
            for mpirank in mpigroup:
                # not worth printing an error, because ranks with no cells to update do not produce an estimator file
                with contextlib.suppress(FileNotFoundError):
                    estfilepath = at.firstexisting(f"estimators_{mpirank:04d}.out", folder=folderpath, tryzipped=True)
                    estfilepaths.append(estfilepath)

            print(
                f"Reading {len(list(estfilepaths))} estimator files from {folderpath.relative_to(Path(folderpath).parent)}"
            )

            processfile = partial(read_estimators_from_file)

            pldf_group = None
            with multiprocessing.get_context("spawn").Pool(processes=at.get_config()["num_processes"]) as pool:
                for pldf_file in pool.imap(processfile, estfilepaths):
                    if pldf_group is None:
                        pldf_group = pldf_file
                    else:
                        pldf_group = pl.concat([pldf_group, pldf_file], how="diagonal_relaxed")

                pool.close()
                pool.join()
                pool.terminate()
            assert pldf_group is not None
            print(f"Writing {parquetfilename.relative_to(modelpath.parent)}")
            pldf_group.write_parquet(parquetfilename, compression="zstd")

    for parquetfilename in group_parquetfiles:
        print(f"Reading {parquetfilename.relative_to(modelpath.parent)}")

    return pl.concat(
        [pl.scan_parquet(parquetfilename) for parquetfilename in group_parquetfiles], how="diagonal_relaxed"
    ).sort(["timestep", "modelgridindex"])


def read_estimators_polars(
    modelpath: Path | str = Path(),
    modelgridindex: None | int | t.Sequence[int] = None,
    timestep: None | int | t.Sequence[int] = None,
    runfolder: None | str | Path = None,
) -> pl.LazyFrame:
    """Read estimator files into a dictionary of (timestep, modelgridindex): estimators.

    Selecting particular timesteps or modelgrid cells will using speed this up by reducing the number of files that must be read.
    """
    modelpath = Path(modelpath)
    match_modelgridindex: None | t.Sequence[int]
    if modelgridindex is None:
        match_modelgridindex = None
    elif isinstance(modelgridindex, int):
        match_modelgridindex = (modelgridindex,)
    else:
        match_modelgridindex = tuple(modelgridindex)

    match_timestep: None | t.Sequence[int]
    if timestep is None:
        match_timestep = None
    elif isinstance(timestep, int):
        match_timestep = (timestep,)
    else:
        match_timestep = tuple(timestep)

    if not Path(modelpath).exists() and Path(modelpath).parts[0] == "codecomparison":
        estimators = at.codecomparison.read_reference_estimators(
            modelpath, timestep=timestep, modelgridindex=modelgridindex
        )
        return pl.DataFrame(
            [
                {
                    "timestep": ts,
                    "modelgridindex": mgi,
                    **estimvals,
                }
                for (ts, mgi), estimvals in estimators.items()
                if not estimvals.get("emptycell", True)
            ]
        ).lazy()

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")

    runfolders = at.get_runfolders(modelpath, timesteps=match_timestep) if runfolder is None else [Path(runfolder)]

    parquetfiles = [folderpath / "estimators.out.parquet.tmp" for folderpath in runfolders]

    for folderpath, parquetfile in zip(runfolders, parquetfiles):
        if not parquetfile.exists():
            pldflazy = read_estimators_in_folder_polars(
                modelpath,
                folderpath,
                match_modelgridindex=None,
            )

            print(f"Writing {parquetfile.relative_to(modelpath.parent)}")
            pldflazy.collect().write_parquet(parquetfile, compression="zstd")

    for folderpath, parquetfile in zip(runfolders, parquetfiles):
        print(f"Scanning {parquetfile.relative_to(modelpath.parent)}")

    pldflazy = pl.concat(
        [pl.scan_parquet(parquetfilename) for parquetfilename in parquetfiles], how="diagonal_relaxed"
    ).unique(["timestep", "modelgridindex"], maintain_order=True, keep="first")

    if match_modelgridindex is not None:
        pldflazy = pldflazy.filter(pl.col("modelgridindex").is_in(match_modelgridindex))

    if match_timestep is not None:
        pldflazy = pldflazy.filter(pl.col("timestep").is_in(match_timestep))

    return pldflazy


def read_estimators(
    modelpath: Path | str = Path(),
    modelgridindex: None | int | t.Sequence[int] = None,
    timestep: None | int | t.Sequence[int] = None,
    runfolder: None | str | Path = None,
    keys: t.Collection[str] | None = None,
) -> dict[tuple[int, int], dict[str, t.Any]]:
    if isinstance(keys, str):
        keys = {keys}
    pldflazy = read_estimators_polars(modelpath, modelgridindex, timestep, runfolder)
    estimators: dict[tuple[int, int], dict[str, t.Any]] = {}
    for estimtsmgi in pldflazy.collect().iter_rows(named=True):
        ts, mgi = estimtsmgi["timestep"], estimtsmgi["modelgridindex"]
        estimators[(ts, mgi)] = {
            k: v
            for k, v in estimtsmgi.items()
            if k not in {"timestep", "modelgridindex"} and (keys is None or k in keys) and v is not None
        }

    return estimators


def get_averaged_estimators(
    modelpath: Path | str,
    estimators: pl.LazyFrame | pl.DataFrame,
    timesteps: int | t.Sequence[int],
    modelgridindex: int,
    keys: str | list | None,
    avgadjcells: int = 0,
) -> dict[str, t.Any]:
    """Get the average of estimators[(timestep, modelgridindex)][keys[0]]...[keys[-1]] across timesteps."""
    modelgridindex = int(modelgridindex)
    if isinstance(timesteps, int):
        timesteps = [timesteps]

    if isinstance(keys, str):
        keys = [keys]
    elif keys is None or not keys:
        keys = [c for c in estimators.columns if c not in {"timestep", "modelgridindex"}]

    dictout = {}
    tdeltas = at.get_timestep_times(modelpath, loc="delta")
    mgilist = list(range(modelgridindex - avgadjcells, modelgridindex + avgadjcells + 1))
    estcollect = (
        estimators.lazy()
        .filter(pl.col("timestep").is_in(timesteps))
        .filter(pl.col("modelgridindex").is_in(mgilist))
        .select({*keys, "timestep", "modelgridindex"})
        .collect()
    )
    for k in keys:
        valuesum = 0
        tdeltasum = 0
        for timestep, tdelta in zip(timesteps, tdeltas):
            for mgi in mgilist:
                value = (
                    estcollect.filter(pl.col("timestep") == timestep).filter(pl.col("modelgridindex") == mgi)[k].item(0)
                )
                if value is None:
                    print(f"{k} not found for timestep {timestep} and modelgridindex {mgi}")
                    continue

                valuesum += value * tdelta
                tdeltasum += tdelta

        dictout[k] = valuesum / tdeltasum

    return dictout


def get_averageionisation(estimatorstsmgi: pl.LazyFrame, atomic_number: int) -> float:
    free_electron_weighted_pop_sum = 0.0
    elsymb = at.get_elsymbol(atomic_number)

    dfselected = estimatorstsmgi.select(
        cs.starts_with(f"nnion_{elsymb}_") | cs.by_name(f"nnelement_{elsymb}")
    ).collect()

    nnelement = dfselected[f"nnelement_{elsymb}"].item(0)
    if nnelement is None:
        return float("NaN")

    found = False
    popsum = 0.0
    for key in dfselected.columns:
        found = True
        nnion = dfselected[key].item(0)
        if nnion is None:
            continue

        ionstage = at.decode_roman_numeral(key.removeprefix(f"nnion_{elsymb}_"))
        free_electron_weighted_pop_sum += nnion * (ionstage - 1)
        popsum += nnion

    return free_electron_weighted_pop_sum / nnelement if found else float("NaN")


def get_averageexcitation(
    modelpath: Path, modelgridindex: int, timestep: int, atomic_number: int, ionstage: int, T_exc: float
) -> float:
    dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)
    adata = at.atomic.get_levels(modelpath)
    ionlevels = adata.query("Z == @atomic_number and ionstage == @ionstage").iloc[0].levels

    energypopsum = 0
    ionpopsum = 0
    if dfnltepops.empty:
        return float("NaN")

    dfnltepops_ion = dfnltepops.query(
        "modelgridindex==@modelgridindex and timestep==@timestep and Z==@atomic_number & ionstage==@ionstage"
    )

    k_b = 8.617333262145179e-05  # eV / K  # noqa: F841

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


def get_partiallycompletetimesteps(estimators: dict[tuple[int, int], dict[str, t.Any]]) -> list[int]:
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
