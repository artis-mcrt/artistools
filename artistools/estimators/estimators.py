"""Functions for reading and processing estimator files.

Examples are temperatures, populations, and heating/cooling rates.
"""

import contextlib
import datetime
import time
import typing as t
from collections.abc import Sequence
from itertools import batched
from pathlib import Path

import polars as pl
from polars import selectors as cs

import artistools as at
from artistools.constants import K_B_ev_per_K


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

    return variableunits.get(key) or variableunits.get(key.split("_", maxsplit=1)[0])


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
        **{f"vel_{ax}_mid": f"$v_{{{ax}}}$" for ax in ["x", "y", "z", "r", "rcyl"]},
        **{f"vel_{ax}_mid_on_c": f"$v_{{{ax}}}$" for ax in ["x", "y", "z", "r", "rcyl"]},
    }.get(varname, varname)


def get_units_string(variable: str) -> str:
    return f" [{units}]" if (units := get_variableunits(variable)) else ""


def get_estimators_rankbatch_parquetfile(
    folderpath: Path | str,
    batch_mpiranks: Sequence[int],
    batchindex: int,
    modelpath: Path | str | None = None,
    verbose: bool = False,
) -> Path:
    def printornot(msg: str) -> None:
        if verbose:
            print(msg)

    modelpath = Path(folderpath).parent if modelpath is None else Path(modelpath)
    folderpath = Path(folderpath)
    parquetfilename = f"estimbatch{batchindex:02d}_{batch_mpiranks[0]:04d}_{batch_mpiranks[-1]:04d}.out.parquet.tmp"
    parquetfilepath = folderpath / parquetfilename

    textsource_mtime: float | int | None = None
    with contextlib.suppress(StopIteration):
        textsource_mtime = next(folderpath.glob("estimators_????.out*")).stat().st_mtime

    assert len(batch_mpiranks) == max(batch_mpiranks) - min(batch_mpiranks) + 1, (
        "batch_mpiranks must be a contiguous range of ranks"
    )
    assert len(set(batch_mpiranks)) == len(batch_mpiranks), "batch_mpiranks must not contain duplicates"

    if not parquetfilepath.exists():
        generate_parquet = True
    elif textsource_mtime and textsource_mtime > parquetfilepath.stat().st_mtime:
        print(
            f"  {parquetfilepath.relative_to(modelpath.parent)} is older than the estimator text files. File will be deleted and regenerated..."
        )
        parquetfilepath.unlink()
        generate_parquet = True
    else:
        generate_parquet = False

    if generate_parquet:
        print(f"  generating {parquetfilepath.relative_to(modelpath.parent)}...")

        time_start = time.perf_counter()

        print(
            f"    reading {len(batch_mpiranks)} estimator files in {folderpath.relative_to(Path(folderpath).parent)}...",
            end="",
            flush=True,
        )

        pldf_batch = at.rustext.estimparse(folderpath, min(batch_mpiranks), max(batch_mpiranks))

        pldf_batch = pldf_batch.with_columns(
            cs.by_name("titeration", "timestep", "modelgridindex", require_all=False).cast(pl.Int32)
        )

        pldf_batch = pldf_batch.select(
            # sort columns with timestep, modelgridindex, and titeration first, then the rest alphabetically
            sorted(
                pldf_batch.columns,
                key=lambda col: f"-{col!r}" if col in {"timestep", "modelgridindex", "titeration"} else str(col),
            )
        )
        print(f"took {time.perf_counter() - time_start:.1f} s. Writing parquet file...", end="", flush=True)
        time_start = time.perf_counter()

        assert pldf_batch is not None
        at.write_parquet_atomic(
            pldf_batch,
            parquetfilepath,
            metadata={
                "creationtimeutc": str(datetime.datetime.now(datetime.UTC)),
                "textsource_mtime": str(textsource_mtime),
                "batch_rank_min": str(min(batch_mpiranks)),
                "batch_rank_max": str(max(batch_mpiranks)),
                "batchindex": str(batchindex),
            },
        )

        print(f"took {time.perf_counter() - time_start:.1f} s.")

    filesize = parquetfilepath.stat().st_size / 1024 / 1024
    try:
        printornot(f"  scanning {parquetfilepath.relative_to(modelpath.parent)} ({filesize:.2f} MiB)")
    except ValueError:
        printornot(f"  scanning {parquetfilepath} ({filesize:.2f} MiB)")

    return parquetfilepath


def join_cell_modeldata(
    estimators: pl.LazyFrame, modelpath: Path | str, verbose: bool = False
) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
    """Join the estimator data with data from model.txt and derived quantities, e.g. density, volume, etc."""
    assert estimators is not None
    estimators = estimators.join(
        at
        .get_timesteps(modelpath)
        .select("timestep", "tmid_days", "twidth_days")
        .with_columns(tmid_days_prevtimestep=pl.col("tmid_days").shift(1)),
        on="timestep",
        how="left",
        coalesce=True,
    )
    dfmodel, modelmeta = at.inputmodel.get_modeldata(
        modelpath, derived_cols=["ALL"], get_elemabundances=True, printwarningsonly=not verbose
    )

    dfmodel = dfmodel.rename({
        colname: f"init_{colname}"
        for colname in dfmodel.collect_schema().names()
        if not colname.startswith("vel_") and colname not in {"inputcellid", "modelgridindex", "mass_g"}
    })
    return estimators.join(dfmodel, on="modelgridindex", suffix="_initmodel").with_columns(
        rho=pl.col("init_rho") * (modelmeta["t_model_init_days"] / pl.col("tmid_days")) ** 3,
        volume=pl.col("init_volume") * (pl.col("tmid_days") / modelmeta["t_model_init_days"]) ** 3,
        volume_prevtimestep=pl.col("init_volume")
        * (pl.col("tmid_days_prevtimestep") / modelmeta["t_model_init_days"]) ** 3,
    ), modelmeta


def add_derived_estimator_columns(pldflazy: pl.LazyFrame) -> pl.LazyFrame:
    """Add quantities derived from the estimator columns that were read from file."""
    colnames = pldflazy.collect_schema().names()

    if "heating_gamma/gamma_dep" in colnames:
        pldflazy = pldflazy.with_columns(gamma_dep=pl.col("heating_gamma") / pl.col("heating_gamma/gamma_dep"))

    if "deposition_gamma" in colnames:
        # sum up the gamma, elec, positron, alpha deposition contributions
        pldflazy = pldflazy.with_columns(total_dep=pl.sum_horizontal(cs.starts_with("deposition_")))
    elif "heating_heating_dep/total_dep" in colnames:
        # for older files with no deposition data, take heating part of deposition and heating fraction
        pldflazy = pldflazy.with_columns(total_dep=pl.col("heating_dep") / pl.col("heating_heating_dep/total_dep"))

    # only fill the number density columns: ARTIS omits zero-abundance ions and isotopes from the estimator files,
    # so a missing number density means zero. The file reader already fills these with zero for cells that skip them
    # within one file, and this makes the columns that a whole rank omitted agree. Every other column (Te, TR, nne,
    # ...) must keep its nulls so that missing data isn't silently read as a real zero
    # a selector that matches nothing makes this a no-op, so no guard is needed here
    pldflazy = pldflazy.with_columns(cs.starts_with("nnelement_", "nnion_", "nniso_").fill_null(0))

    if any(col.startswith("nnelement_") for col in colnames):
        pldflazy = pldflazy.with_columns(nntot=pl.sum_horizontal(cs.starts_with("nnelement_")))

    return pldflazy


def scan_estimators(
    modelpath: Path | str = ".",
    modelgridindex: int | Sequence[int] | None = None,
    timestep: int | Sequence[int] | None = None,
    join_modeldata: bool = False,
    verbose: bool = False,
) -> pl.LazyFrame:
    """Read estimator files into a polars LazyFrame with columns for timestep, modelgridindex, and estimator values.

    Selecting particular timesteps or modelgrid cells will speed this up by reducing the number of files that must be read.
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
        return pl.LazyFrame(
            [{"timestep": ts, "modelgridindex": mgi, **estimvals} for (ts, mgi), estimvals in estimators.items()],
            orient="row",
        )

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")
    mpiranklist = at.get_mpiranklist(modelpath, only_ranks_withgridcells=True)
    mpiranks_matched = (
        {at.get_mpirankofcell(modelpath=modelpath, modelgridindex=mgi) for mgi in match_modelgridindex}
        if match_modelgridindex
        else set(mpiranklist)
    )
    mpirank_groups = [
        (batchindex, mpiranks)
        for batchindex, mpiranks in enumerate(batched(mpiranklist, 100, strict=False))
        if mpiranks_matched.intersection(mpiranks)
    ]

    runfolders = at.get_runfolders(modelpath, timesteps=match_timestep)
    if runfolders:
        parquetfiles = [
            get_estimators_rankbatch_parquetfile(
                modelpath=modelpath,
                folderpath=runfolder,
                batch_mpiranks=mpiranks,
                batchindex=batchindex,
                verbose=verbose,
            )
            for runfolder in runfolders
            for batchindex, mpiranks in mpirank_groups
        ]

        assert bool(parquetfiles)
        if not verbose:
            datasize_GB = sum(pfile.stat().st_size for pfile in parquetfiles) / 1024 / 1024 / 1024
            str_runfolders = ", ".join([Path(x).relative_to(modelpath).as_posix() for x in runfolders])
            print(
                f"  scanning {len(parquetfiles)} parquet estimator files ({datasize_GB:.1f} GB) from {str_runfolders}..."
            )
        pldflazy = (
            pl
            .concat([pl.scan_parquet(pfile) for pfile in parquetfiles], how="diagonal_relaxed")
            .unique(["timestep", "modelgridindex"], maintain_order=True, keep="first")
            .lazy()
        )
    else:
        print(
            f"WARNING: No run folders found in {modelpath}. Enabling fallback to cross join of all model data and timesteps."
        )
        pldflazy = (
            at
            .get_timesteps(modelpath)
            .select("timestep", "tmid_days", "twidth_days")
            .join(at.inputmodel.get_modeldata(modelpath)[0].select("modelgridindex"), how="cross")
        )

    if match_modelgridindex is not None:
        pldflazy = pldflazy.filter(pl.col("modelgridindex").is_in(match_modelgridindex))

    if match_timestep is not None:
        pldflazy = pldflazy.filter(pl.col("timestep").is_in(match_timestep))

    pldflazy = add_derived_estimator_columns(pldflazy)

    if join_modeldata:
        pldflazy, _ = join_cell_modeldata(estimators=pldflazy, modelpath=modelpath, verbose=verbose)

    return pldflazy


def read_estimators(
    modelpath: Path | str = ".",
    modelgridindex: int | Sequence[int] | None = None,
    timestep: int | Sequence[int] | None = None,
) -> dict[tuple[int, int], dict[str, t.Any]]:
    """Read ARTIS estimator data into a dictionary keyed by (timestep, modelgridindex).

    When collecting many cells and timesteps, this is very slow, and it's almost always better to use scan_estimators instead.
    """
    lzpldfestimators = scan_estimators(modelpath, modelgridindex, timestep)

    if isinstance(modelgridindex, int):
        lzpldfestimators = lzpldfestimators.filter(pl.col("modelgridindex") == modelgridindex)
    elif isinstance(modelgridindex, Sequence):
        lzpldfestimators = lzpldfestimators.filter(pl.col("modelgridindex").is_in(modelgridindex))
    if isinstance(timestep, int):
        lzpldfestimators = lzpldfestimators.filter(pl.col("timestep") == timestep)
    elif isinstance(timestep, Sequence):
        lzpldfestimators = lzpldfestimators.filter(pl.col("timestep").is_in(timestep))

    pldfestimators = lzpldfestimators.collect()

    estimators: dict[tuple[int, int], dict[str, t.Any]] = {}
    for estimtsmgi in pldfestimators.iter_rows(named=True):
        ts, mgi = estimtsmgi["timestep"], estimtsmgi["modelgridindex"]
        estimators[ts, mgi] = {
            k: v for k, v in estimtsmgi.items() if k not in {"timestep", "modelgridindex"} and v is not None
        }

    return estimators


def get_averageexcitation(
    modelpath: Path | str, atomic_number: int, ion_stage: int, dftexc: pl.LazyFrame
) -> pl.LazyFrame:
    """Return the population-weighted mean level excitation energy [eV] of an ion per timestep and cell.

    dftexc gives the excitation temperature (columns timestep, modelgridindex, T_exc) used to spread
    the superlevel population over the levels it stands in for.
    """
    dfpops = (
        at.nltepops
        .read_files(modelpath)
        .lazy()
        .filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage))
    )

    adata = at.atomic.get_levels(modelpath)
    dfionlevels = adata.filter((pl.col("Z") == atomic_number) & (pl.col("ion_stage") == ion_stage))["levels"].item()
    if dfionlevels is None:
        msg = f"No level data for Z={atomic_number} ion_stage={ion_stage}"
        raise ValueError(msg)
    dflevels = dfionlevels.lazy().select(
        level=pl.col("levelindex").cast(pl.Int64), energy_ev=pl.col("energy_ev"), g=pl.col("g")
    )

    groupcols = ["timestep", "modelgridindex"]

    # resolved levels contribute their own energy; the superlevel is added below
    dfresolved = (
        dfpops
        .filter(pl.col("level") >= 0)
        .join(dflevels, on="level", how="inner")
        .group_by(groupcols)
        .agg(energypopsum=(pl.col("energy_ev") * pl.col("n_NLTE")).sum(), levelnumber_sl=pl.col("level").max() + 1)
    )

    dfionpopsum = dfpops.group_by(groupcols).agg(ionpopsum=pl.col("n_NLTE").sum())

    dfsuperlevel = (
        dfpops
        .filter(pl.col("level") < 0)
        .group_by(groupcols)
        .agg(n_NLTE_sl=pl.col("n_NLTE").sum())
        .join(dfresolved.select([*groupcols, "levelnumber_sl"]), on=groupcols, how="inner")
        .join(dftexc, on=groupcols, how="inner")
        .collect()
    )

    dfsuperlevelenergy = _superlevel_energy(dfsuperlevel, dflevels.collect(), groupcols)

    # inner join on dftexc, so the result covers exactly the cells a temperature was given for and a
    # missing one cannot silently drop the superlevel term
    return (
        dfresolved
        .join(dftexc.select(groupcols), on=groupcols, how="inner")
        .join(dfionpopsum, on=groupcols, how="inner")
        .join(dfsuperlevelenergy.lazy(), on=groupcols, how="left")
        .with_columns(
            averageexcitation=(pl.col("energypopsum") + pl.col("energypopsum_sl").fill_null(0.0)) / pl.col("ionpopsum")
        )
        .select([*groupcols, "averageexcitation"])
    )


def _superlevel_energy(dfsuperlevel: pl.DataFrame, dflevels: pl.DataFrame, groupcols: list[str]) -> pl.DataFrame:
    """Return the energy the superlevel population contributes, Boltzmann-distributed at T_exc."""
    schema: dict[str, pl.DataType] = {
        **{col: dfsuperlevel.schema[col] for col in groupcols},
        "energypopsum_sl": pl.Float64(),
    }
    if dfsuperlevel.is_empty():
        return pl.DataFrame(schema=schema)

    # levelnumber_sl is the same for every cell in practice, so this loops once
    contributions = []
    for levelnumber_sl in dfsuperlevel["levelnumber_sl"].unique().sort():
        dflevels_above = dflevels.filter(pl.col("level") >= levelnumber_sl)
        if dflevels_above.is_empty():
            continue

        contributions.append(
            dfsuperlevel
            .filter(pl.col("levelnumber_sl") == levelnumber_sl)
            .join(dflevels_above, how="cross")
            .with_columns(boltzfac=pl.col("g") * (-pl.col("energy_ev") / K_B_ev_per_K / pl.col("T_exc")).exp())
            .group_by(groupcols)
            .agg(
                energypopsum_sl=pl.col("n_NLTE_sl").first()
                * (pl.col("energy_ev") * pl.col("boltzfac")).sum()
                / pl.col("boltzfac").sum()
            )
        )

    return pl.concat(contributions) if contributions else pl.DataFrame(schema=schema)
