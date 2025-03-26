#!/usr/bin/env python3
"""Functions for reading and processing estimator files.

Examples are temperatures, populations, and heating/cooling rates.
"""

import argparse
import contextlib
import math
import sys
import tempfile
import time
import typing as t
import warnings
from collections import namedtuple
from collections.abc import Collection
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from polars import selectors as cs
from typing_extensions import deprecated

import artistools as at


def get_variableunits(key: str) -> str | None:
    variableunits = {
        "time": "days",
        "gamma_NT": "s^-1",
        "gamma_R_bfest": "s^-1",
        "TR": "K",
        "Te": "K",
        "TJ": "K",
        "nne": "e$^-$/cm$^3$",
        "nniso": "cm$^{-3}$",
        "nnion": "cm$^{-3}$",
        "nnelement": "cm$^{-3}$",
        "deposition": "erg/s/cm$^3$",
        "total_dep": "erg/s/cm$^3$",
        "heating": "erg/s/cm$^3$",
        "heating_dep/total_dep": "Ratio",
        "cooling": "erg/s/cm$^3$",
        "rho": "g/cm$^3$",
        "velocity": "km/s",
        "beta": "v/c",
        "vel_r_max_kmps": "km/s",
        **{f"vel_{ax}_mid": "cm/s" for ax in ["x", "y", "z", "r", "rcyl"]},
        **{f"vel_{ax}_mid_on_c": "c" for ax in ["x", "y", "z", "r", "rcyl"]},
    }

    return variableunits.get(key) or variableunits.get(key.split("_")[0])


def get_variablelongunits(key: str) -> str | None:
    return {"heating_dep/total_dep": "", "TR": "Temperature [K]", "Te": "Temperature [K]", "TJ": "Temperature [K]"}.get(
        key
    )


def get_varname_formatted(varname: str) -> str:
    return {
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
    }.get(varname, varname)


def apply_filters(
    xlist: Sequence[float] | npt.NDArray[np.floating],
    ylist: Sequence[float] | npt.NDArray[np.floating],
    args: argparse.Namespace,
) -> tuple[t.Any, t.Any]:
    if (filterfunc := at.get_filterfunc(args)) is not None:
        ylist = filterfunc(ylist)

    return xlist, ylist


def get_ionrecombrates_fromfile(filename: Path | str) -> pd.DataFrame:
    """WARNING: copy pasted from artis-atomic! replace with a package import soon ion_stage is the lower ion stage."""
    print(f"Reading {filename}")

    header_row = []
    with Path(filename).open(encoding="utf-8") as filein:
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
                records.append(recomb_tuple(*[float(row[index]) for index in (index_logt, index_low_n, index_tot)]))

    return pd.DataFrame.from_records(records, columns=recomb_tuple._fields)


def get_units_string(variable: str) -> str:
    return f" [{units}]" if (units := get_variableunits(variable)) else ""


def read_estimators_from_file(estfilepath: Path | str, printfilename: bool = False) -> pl.DataFrame:
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
                estimblock = {}
                if not emptycell:
                    # will be timestep, modelgridindex, TR, Te, W, TJ, nne, etc
                    for variablename, value in zip(row[::2], row[1::2], strict=True):
                        estimblock[variablename] = (
                            float(value)
                            if variablename not in {"timestep", "modelgridindex", "titeration", "thick"}
                            else int(value)
                        )

            elif row[1].startswith("Z="):
                variablename = row[0]
                if row[1].endswith("="):
                    atomic_number = int(row[2])
                    startindex = 3
                else:
                    atomic_number = int(row[1].split("=")[1])
                    startindex = 2
                elsymbol = at.get_elsymbol(atomic_number)

                for ion_stage_str, value in zip(row[startindex::2], row[startindex + 1 :: 2], strict=True):
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

            elif row[0].endswith(":"):
                # heating, cooling, deposition, etc
                variablename = row[0].removesuffix(":")
                for coolingtype, value in zip(row[1::2], row[2::2], strict=True):
                    estimblock[f"{variablename}_{coolingtype}"] = float(value)

    # reached the end of file
    if estimblock:
        estimblocklist.append(estimblock)

    return pl.DataFrame(estimblocklist).with_columns(
        pl.col(pl.Int64).cast(pl.Int32), pl.col(pl.Float64).cast(pl.Float32)
    )


def get_rankbatch_parquetfile(
    folderpath: Path | str,
    batch_mpiranks: Sequence[int],
    batchindex: int,
    modelpath: Path | str | None = None,
    use_rust_parser: bool | None = True,
) -> Path:
    modelpath = Path(folderpath).parent if modelpath is None else Path(modelpath)
    folderpath = Path(folderpath)
    parquetfilename = f"estimbatch{batchindex:02d}_{batch_mpiranks[0]:04d}_{batch_mpiranks[-1]:04d}.out.parquet.tmp"
    parquetfilepath = folderpath / parquetfilename

    if not parquetfilepath.exists():
        generate_parquet = True
    elif next(folderpath.glob("estimators_????.out*")).stat().st_mtime > parquetfilepath.stat().st_mtime:
        print(
            f"  {parquetfilepath.relative_to(modelpath.parent)} is older than the estimator text files. File will be deleted and regenerated..."
        )
        parquetfilepath.unlink()
        generate_parquet = True
    else:
        generate_parquet = False

    if generate_parquet:
        print(f"  generating {parquetfilepath.relative_to(modelpath.parent)}...")
        estfilepaths = []
        for mpirank in batch_mpiranks:
            # not worth printing an error, because ranks with no cells to update do not produce an estimator file
            with contextlib.suppress(FileNotFoundError):
                estfilepaths.append(
                    at.firstexisting(f"estimators_{mpirank:04d}.out", folder=folderpath, tryzipped=True)
                )

        time_start = time.perf_counter()

        if use_rust_parser is None or use_rust_parser:
            try:
                from artistools.rustext import estimparse as rustestimparse

                use_rust_parser = True

            except ImportError as err:
                warnings.warn(
                    "WARNING: Rust extension not available. Falling back to slow python reader.", stacklevel=2
                )
                if use_rust_parser:
                    msg = "Rust extension not available"
                    raise ImportError(msg) from err
                use_rust_parser = False

        print(
            f"    reading {len(estfilepaths)} estimator files in {folderpath.relative_to(Path(folderpath).parent)} with {'fast rust reader' if use_rust_parser else 'slow python reader'}...",
            end="",
            flush=True,
        )

        pldf_batch: pl.DataFrame
        if use_rust_parser:
            pldf_batch = rustestimparse(str(folderpath), min(batch_mpiranks), max(batch_mpiranks))
            pldf_batch = pldf_batch.with_columns(
                pl.col(c).cast(pl.Int32)
                for c in {"modelgridindex", "timestep", "titeration", "thick"}.intersection(pldf_batch.columns)
            )
        elif at.get_config()["num_processes"] > 1:
            with at.get_multiprocessing_pool() as pool:
                pldf_batch = pl.concat(pool.imap(read_estimators_from_file, estfilepaths), how="diagonal_relaxed")

                pool.close()
                pool.join()

        else:
            pldf_batch = pl.concat(map(read_estimators_from_file, estfilepaths), how="diagonal_relaxed")

        pldf_batch = pldf_batch.select(
            sorted(
                pldf_batch.columns,
                key=lambda col: f"-{col!r}" if col in {"timestep", "modelgridindex", "titer"} else str(col),
            )
        )
        print(f"took {time.perf_counter() - time_start:.1f} s. Writing parquet file...", end="", flush=True)
        time_start = time.perf_counter()

        assert pldf_batch is not None
        partialparquetfilepath = Path(
            tempfile.mkstemp(dir=folderpath, prefix=f"{parquetfilename}.partial", suffix=".partial")[1]
        )
        pldf_batch.write_parquet(partialparquetfilepath, compression="zstd", statistics=True, compression_level=8)
        if parquetfilepath.exists():
            partialparquetfilepath.unlink()
        else:
            partialparquetfilepath.rename(parquetfilepath)

        print(f"took {time.perf_counter() - time_start:.1f} s.")

    filesize = parquetfilepath.stat().st_size / 1024 / 1024
    try:
        print(f"  scanning {parquetfilepath.relative_to(modelpath.parent)} ({filesize:.2f} MiB)")
    except ValueError:
        print(f"  scanning {parquetfilepath} ({filesize:.2f} MiB)")

    return parquetfilepath


def scan_estimators(
    modelpath: Path | str = Path(),
    modelgridindex: int | Sequence[int] | None = None,
    timestep: int | Sequence[int] | None = None,
    use_rust_parser: bool | None = None,
) -> pl.LazyFrame:
    """Read estimator files into a dictionary of (timestep, modelgridindex): estimators.

    Selecting particular timesteps or modelgrid cells will using speed this up by reducing the number of files that must be read.
    """
    modelpath = Path(modelpath)
    match_modelgridindex: Sequence[int] | None
    if modelgridindex is None:
        match_modelgridindex = None
    elif isinstance(modelgridindex, int):
        match_modelgridindex = (modelgridindex,)
    else:
        match_modelgridindex = tuple(modelgridindex)

    match_timestep: Sequence[int] | None
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
        return pl.DataFrame([
            {"timestep": ts, "modelgridindex": mgi, **estimvals} for (ts, mgi), estimvals in estimators.items()
        ]).lazy()

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")
    mpiranklist = at.get_mpiranklist(modelpath, only_ranks_withgridcells=True)
    mpiranks_matched = (
        {at.get_mpirankofcell(modelpath=modelpath, modelgridindex=mgi) for mgi in match_modelgridindex}
        if match_modelgridindex
        else set(mpiranklist)
    )
    mpirank_groups = [
        (batchindex, mpiranks)
        for batchindex, mpiranks in enumerate(at.misc.batched(mpiranklist, 100))
        if mpiranks_matched.intersection(mpiranks)
    ]

    runfolders = at.get_runfolders(modelpath, timesteps=match_timestep)

    parquetfiles = (
        get_rankbatch_parquetfile(
            modelpath=modelpath,
            folderpath=runfolder,
            batch_mpiranks=mpiranks,
            batchindex=batchindex,
            use_rust_parser=use_rust_parser,
        )
        for runfolder in runfolders
        for batchindex, mpiranks in mpirank_groups
    )

    assert bool(parquetfiles)

    pldflazy = pl.concat([pl.scan_parquet(pfile) for pfile in parquetfiles], how="diagonal_relaxed").unique(
        ["timestep", "modelgridindex"], maintain_order=True, keep="first"
    )

    if match_modelgridindex is not None:
        pldflazy = pldflazy.filter(pl.col("modelgridindex").is_in(match_modelgridindex))

    if match_timestep is not None:
        pldflazy = pldflazy.filter(pl.col("timestep").is_in(match_timestep))

    colnames = pldflazy.collect_schema().names()
    # add some derived quantities
    if "heating_gamma/gamma_dep" in colnames:
        pldflazy = pldflazy.with_columns(gamma_dep=pl.col("heating_gamma") / pl.col("heating_gamma/gamma_dep"))

    if "deposition_gamma" in colnames:
        # sum up the gamma, elec, positron, alpha deposition contributions
        pldflazy = pldflazy.with_columns(total_dep=pl.sum_horizontal(cs.starts_with("deposition_")))
    elif "heating_heating_dep/total_dep" in colnames:
        # for older files with no deposition data, take heating part of deposition and heating fraction
        pldflazy = pldflazy.with_columns(total_dep=pl.col("heating_dep") / pl.col("heating_heating_dep/total_dep"))

    return pldflazy.with_columns(nntot=pl.sum_horizontal(cs.starts_with("nnelement_"))).fill_null(0)


@deprecated("Use scan_estimators instead.")
def read_estimators(
    modelpath: Path | str = Path(),
    modelgridindex: int | Sequence[int] | None = None,
    timestep: int | Sequence[int] | None = None,
    keys: Collection[str] | None = None,
) -> dict[tuple[int, int], dict[str, t.Any]]:
    """Read ARTIS estimator data into a dictionary keyed by (timestep, modelgridindex).

    DEPRECATED: This is very slow, and it's almost always better to use scan_estimators instead.
    """
    if isinstance(keys, str):
        keys = {keys}
    pldfestimators = scan_estimators(modelpath, modelgridindex, timestep).collect()
    estimators: dict[tuple[int, int], dict[str, t.Any]] = {}
    for estimtsmgi in pldfestimators.iter_rows(named=True):
        ts, mgi = estimtsmgi["timestep"], estimtsmgi["modelgridindex"]
        estimators[ts, mgi] = {
            k: v
            for k, v in estimtsmgi.items()
            if k not in {"timestep", "modelgridindex"} and (keys is None or k in keys) and v is not None
        }

    return estimators


def get_averageexcitation(
    modelpath: Path | str, modelgridindex: int, timestep: int, atomic_number: int, ion_stage: int, T_exc: float
) -> float | None:
    dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)
    if dfnltepops.empty:
        print(f"WARNING: NLTE pops not found for cell {modelgridindex} at timestep {timestep}")

    adata = at.atomic.get_levels_polars(modelpath)
    ionlevels = adata.filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage))["levels"].item()

    energypopsum = 0
    ionpopsum = 0
    if dfnltepops.empty:
        return None

    dfnltepops_ion = dfnltepops.query(
        "modelgridindex==@modelgridindex and timestep==@timestep and Z==@atomic_number & ion_stage==@ion_stage"
    )

    k_b = 8.617333262145179e-05  # eV / K

    ionpopsum = dfnltepops_ion.n_NLTE.sum()
    energypopsum = sum(
        ionlevels["energy_ev"].item(level) * n_NLTE
        for level, n_NLTE in dfnltepops_ion[dfnltepops_ion.level >= 0][["level", "n_NLTE"]].itertuples(index=False)
    )

    with contextlib.suppress(IndexError):  # no superlevel will cause IndexError
        superlevelrow = dfnltepops_ion[dfnltepops_ion.level < 0].iloc[0]
        levelnumber_sl = dfnltepops_ion.level.max() + 1

        energy_boltzfac_sum = (
            ionlevels[levelnumber_sl:]
            .select(pl.col("energy_ev") * pl.col("g") * (-pl.col("energy_ev") / k_b / T_exc).exp())
            .sum()
            .item()
        )

        boltzfac_sum = energy_boltzfac_sum = (
            ionlevels[levelnumber_sl:].select(pl.col("g") * (-pl.col("energy_ev") / k_b / T_exc).exp()).sum().item()
        )
        # adjust to the actual superlevel population from ARTIS
        energypopsum += energy_boltzfac_sum * superlevelrow.n_NLTE / boltzfac_sum

    return energypopsum / ionpopsum
