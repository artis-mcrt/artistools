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
import time
import typing as t
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import artistools as at


def get_variableunits(key: str) -> str | None:
    variableunits = {
        "time": "days",
        "gamma_NT": "s^-1",
        "gamma_R_bfest": "s^-1",
        "TR": "K",
        "Te": "K",
        "TJ": "K",
        "nne": "e^-/cm3",
        "nniso": "cm$^{-3}$",
        "nnion": "cm$^{-3}$",
        "nnelement": "cm$^{-3}$",
        "heating": "erg/s/cm3",
        "heating_dep/total_dep": "Ratio",
        "cooling": "erg/s/cm3",
        "rho": "g/cm3",
        "velocity": "km/s",
        "beta": "v/c",
        "vel_r_max_kmps": "km/s",
        **{f"vel_{ax}_mid": "cm/s" for ax in ["x", "y", "z", "r", "rcyl"]},
        **{f"vel_{ax}_mid_on_c": "c" for ax in ["x", "y", "z", "r", "rcyl"]},
    }

    return variableunits.get(key) or variableunits.get(key.split("_")[0])


def get_variablelongunits(key: str) -> str | None:
    variablelongunits = {
        "heating_dep/total_dep": "",
        "TR": "Temperature [K]",
        "Te": "Temperature [K]",
        "TJ": "Temperature [K]",
    }
    return variablelongunits.get(key)


def get_varname_formatted(varname: str) -> str:
    replacements = {
        "nne": r"n$_{\rm e}$",
        "lognne": r"Log n$_{\rm e}$",
        "rho": r"$\rho$",
        "Te": r"T$_{\rm e}$",
        "TR": r"T$_{\rm R}$",
        "TJ": r"T$_{\rm J}$",
        "gamma_NT": r"$\Gamma_{\rm non-thermal}$ [s$^{-1}$]",
        "gamma_R_bfest": r"$\Gamma_{\rm phot}$ [s$^{-1}$]",
        "heating_dep/total_dep": "Heating fraction",
        **{f"vel_{ax}_mid_on_c": f"$v_{{{ax}}}$" for ax in ["x", "y", "z", "r", "rcyl"]},
    }
    return replacements.get(varname, varname)


def apply_filters(
    xlist: t.Sequence[float] | np.ndarray, ylist: t.Sequence[float] | np.ndarray, args: argparse.Namespace
) -> tuple[t.Any, t.Any]:
    if (filterfunc := at.get_filterfunc(args)) is not None:
        ylist = filterfunc(ylist)

    return xlist, ylist


def get_ionrecombrates_fromfile(filename: Path | str) -> pd.DataFrame:
    """WARNING: copy pasted from artis-atomic! replace with a package import soon ion_stage is the lower ion stage."""
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
    return f" [{units}]" if (units := get_variableunits(variable)) else ""


def read_estimators_from_file(
    estfilepath: Path | str,
    printfilename: bool = False,
) -> pl.DataFrame:
    if printfilename:
        estfilepath = Path(estfilepath)
        filesize = estfilepath.stat().st_size / 1024 / 1024
        print(f"  Reading {estfilepath.relative_to(estfilepath.parent.parent)} ({filesize:.2f} MiB)")

    estimblocklist: list[dict[str, t.Any]] = []
    estimblock: dict[str, t.Any] = {}
    with at.zopen(estfilepath) as estimfile:
        for line in estimfile:
            row: list[str] = line.split()
            if not row:
                continue

            if row[0] == "timestep":
                # yield the previous block before starting a new one
                if estimblock:
                    estimblocklist.append(estimblock)

                emptycell = row[4] == "EMPTYCELL"
                if emptycell:
                    estimblock = {}
                else:
                    # will be TR, Te, W, TJ, nne
                    estimblock = {"timestep": int(row[1]), "modelgridindex": int(row[3])}
                    for variablename, value in zip(row[4::2], row[5::2]):
                        estimblock[variablename] = float(value)

            elif row[1].startswith("Z="):
                variablename = row[0]
                if row[1].endswith("="):
                    atomic_number = int(row[2])
                    startindex = 3
                else:
                    atomic_number = int(row[1].split("=")[1])
                    startindex = 2
                elsymbol = at.get_elsymbol(atomic_number)

                for ion_stage_str, value in zip(row[startindex::2], row[startindex + 1 :: 2]):
                    ion_stage_str_strip = ion_stage_str.strip()
                    if ion_stage_str_strip == "(or":
                        continue

                    value_thision = float(value.rstrip(","))

                    if ion_stage_str_strip == "SUM:":
                        estimblock[f"nnelement_{elsymbol}"] = value_thision
                        continue

                    try:
                        ion_stage = int(ion_stage_str.rstrip(":"))
                    except ValueError:
                        if variablename == "populations" and ion_stage_str.startswith(elsymbol):
                            estimblock[f"nniso_{ion_stage_str.rstrip(':')}"] = float(value)
                        else:
                            print(ion_stage_str, elsymbol)
                            print(f"Cannot parse row: {row}")
                        continue

                    ionstr = at.get_ionstring(atomic_number, ion_stage, sep="_", style="spectral")
                    estimblock[f"{'nnion' if variablename == 'populations' else variablename}_{ionstr}"] = value_thision

                    if variablename in {"Alpha_R*nne", "AlphaR*nne"}:
                        estimblock[f"Alpha_R_{ionstr}"] = (
                            value_thision / estimblock["nne"] if estimblock["nne"] > 0.0 else math.inf
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
    if estimblock:
        estimblocklist.append(estimblock)

    return pl.DataFrame(estimblocklist).with_columns(
        pl.col(pl.Int64).cast(pl.Int32), pl.col(pl.Float64).cast(pl.Float32)
    )


def batched(iterable, n):  # -> Generator[list, Any, None]:
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


def get_rankbatch_parquetfile(
    modelpath: Path,
    folderpath: Path,
    batch_mpiranks: t.Sequence[int],
    batchindex: int,
) -> Path:
    parquetfilepath = (
        folderpath / f"estimbatch{batchindex:02d}_{batch_mpiranks[0]:04d}_{batch_mpiranks[-1]:04d}.out.parquet.tmp"
    )

    if not parquetfilepath.exists():
        print(f"{parquetfilepath.relative_to(modelpath.parent)} does not exist")
        estfilepaths = []
        for mpirank in batch_mpiranks:
            # not worth printing an error, because ranks with no cells to update do not produce an estimator file
            with contextlib.suppress(FileNotFoundError):
                estfilepath = at.firstexisting(f"estimators_{mpirank:04d}.out", folder=folderpath, tryzipped=True)
                estfilepaths.append(estfilepath)

        print(
            f"  reading {len(list(estfilepaths))} estimator files from {folderpath.relative_to(Path(folderpath).parent)}"
        )

        time_start = time.perf_counter()

        pldf_group = None
        with multiprocessing.get_context("spawn").Pool(processes=at.get_config()["num_processes"]) as pool:
            for pldf_file in pool.imap(read_estimators_from_file, estfilepaths):
                if pldf_group is None:
                    pldf_group = pldf_file
                else:
                    pldf_group = pl.concat([pldf_group, pldf_file], how="diagonal_relaxed")

            pool.close()
            pool.join()
            pool.terminate()

        print(f"    took {time.perf_counter() - time_start:.1f} s")

        assert pldf_group is not None
        print(f"  writing {parquetfilepath.relative_to(modelpath.parent)}")
        pldf_group.write_parquet(parquetfilepath, compression="zstd", statistics=True, compression_level=8)

    filesize = parquetfilepath.stat().st_size / 1024 / 1024
    print(f"Scanning {parquetfilepath.relative_to(modelpath.parent)} ({filesize:.2f} MiB)")

    return parquetfilepath


def scan_estimators(
    modelpath: Path | str = Path(),
    modelgridindex: None | int | t.Sequence[int] = None,
    timestep: None | int | t.Sequence[int] = None,
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
            ]
        ).lazy()

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")
    mpiranklist = at.get_mpiranklist(modelpath, only_ranks_withgridcells=True)
    mpiranks_matched = (
        {at.get_mpirankofcell(modelpath=modelpath, modelgridindex=mgi) for mgi in match_modelgridindex}
        if match_modelgridindex
        else set(mpiranklist)
    )
    mpirank_groups = [
        (batchindex, mpiranks)
        for batchindex, mpiranks in enumerate(batched(mpiranklist, 100))
        if mpiranks_matched.intersection(mpiranks)
    ]

    runfolders = at.get_runfolders(modelpath, timesteps=match_timestep)

    parquetfiles = (
        get_rankbatch_parquetfile(modelpath, runfolder, mpiranks, batchindex=batchindex)
        for runfolder in runfolders
        for batchindex, mpiranks in mpirank_groups
    )
    assert bool(parquetfiles)

    pldflazy = pl.concat([pl.scan_parquet(pfile) for pfile in parquetfiles], how="diagonal_relaxed")
    pldflazy = pldflazy.unique(["timestep", "modelgridindex"], maintain_order=True, keep="first")

    if match_modelgridindex is not None:
        pldflazy = pldflazy.filter(pl.col("modelgridindex").is_in(match_modelgridindex))

    if match_timestep is not None:
        pldflazy = pldflazy.filter(pl.col("timestep").is_in(match_timestep))

    return pldflazy.fill_null(0)


def read_estimators(
    modelpath: Path | str = Path(),
    modelgridindex: None | int | t.Sequence[int] = None,
    timestep: None | int | t.Sequence[int] = None,
    keys: t.Collection[str] | None = None,
) -> dict[tuple[int, int], dict[str, t.Any]]:
    if isinstance(keys, str):
        keys = {keys}
    pldflazy = scan_estimators(modelpath, modelgridindex, timestep)
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
    modelgridindex: int | t.Sequence[int],
    keys: str | list | None,
) -> dict[str, t.Any]:
    """Get the average across timsteps for a cell."""
    assert isinstance(modelgridindex, int)
    if isinstance(timesteps, int):
        timesteps = [timesteps]

    if isinstance(keys, str):
        keys = [keys]
    elif keys is None or not keys:
        keys = [c for c in estimators.columns if c not in {"timestep", "modelgridindex"}]

    dictout = {}
    tdeltas = at.get_timestep_times(modelpath, loc="delta")

    estcollect = (
        estimators.lazy()
        .filter(pl.col("timestep").is_in(timesteps))
        .filter(pl.col("modelgridindex") == modelgridindex)
        .select({*keys, "timestep", "modelgridindex"})
        .collect()
    )
    for k in keys:
        valuesum = 0
        tdeltasum = 0
        for timestep, tdelta in zip(timesteps, tdeltas):
            value = (
                estcollect.filter(pl.col("timestep") == timestep)
                .filter(pl.col("modelgridindex") == modelgridindex)[k]
                .item(0)
            )
            if value is None:
                continue

            valuesum += value * tdelta
            tdeltasum += tdelta

        dictout[k] = valuesum / tdeltasum if tdeltasum > 0 else math.nan

    return dictout


def get_averageexcitation(
    modelpath: Path | str, modelgridindex: int, timestep: int, atomic_number: int, ion_stage: int, T_exc: float
) -> float | None:
    dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)
    if dfnltepops.empty:
        print(f"WARNING: NLTE pops not found for cell {modelgridindex} at timestep {timestep}")

    adata = at.atomic.get_levels(modelpath)
    ionlevels = adata.query("Z == @atomic_number and ion_stage == @ion_stage").iloc[0].levels

    energypopsum = 0
    ionpopsum = 0
    if dfnltepops.empty:
        return None

    dfnltepops_ion = dfnltepops.query(
        "modelgridindex==@modelgridindex and timestep==@timestep and Z==@atomic_number & ion_stage==@ion_stage"
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
