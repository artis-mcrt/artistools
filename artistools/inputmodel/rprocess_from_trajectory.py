#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import contextlib
import gc
import math
import string
import tarfile
import time
import typing as t
from collections.abc import Sequence
from functools import lru_cache
from functools import partial
from itertools import chain
from pathlib import Path

import argcomplete
import numpy as np
import pandas as pd
import polars as pl

import artistools as at


def get_elemabund_from_nucabund(dfnucabund: pl.DataFrame) -> dict[str, float]:
    """Return a dictionary of elemental abundances from nuclear abundance DataFrame."""
    ZMAX = dfnucabund["Z"].max()
    assert isinstance(ZMAX, int)
    dictelemabund: dict[str, float] = {
        f"X_{at.get_elsymbol(atomic_number)}": dfnucabund.filter(pl.col("Z") == atomic_number)["massfrac"].sum()
        for atomic_number in range(1, ZMAX + 1)
    }
    return dictelemabund


def get_dfelemabund_from_dfmodel(dfmodel: pl.DataFrame, dfnucabundances: pl.DataFrame) -> pl.DataFrame:
    timestart = time.perf_counter()
    print("Adding up isotopes for elemental abundances and creating dfelabundances...", end="", flush=True)
    elemisotopes: dict[int, list[str]] = {}
    nuclidesincluded = 0

    for colname in sorted(dfnucabundances.columns):
        if not colname.startswith("X_"):
            continue
        nuclidesincluded += 1
        atomic_number = at.get_atomic_number(colname[2:].rstrip(string.digits))
        if atomic_number in elemisotopes:
            elemisotopes[atomic_number].append(colname)
        else:
            elemisotopes[atomic_number] = [colname]

    elementsincluded = len(elemisotopes)

    dfelabundances_partial = pl.DataFrame({
        "inputcellid": dfnucabundances["inputcellid"],
        **{
            f"X_{at.get_elsymbol(atomic_number)}": (
                dfnucabundances.select(elemisotopes[atomic_number]).sum_horizontal()
                if atomic_number in elemisotopes
                else np.zeros(len(dfnucabundances))
            )
            for atomic_number in range(1, max(elemisotopes.keys()) + 1)
        },
    })

    # ensure cells with no traj contributions are included
    dfelabundances = (
        pl.DataFrame(pl.Series(name="inputcellid", values=dfmodel["inputcellid"], dtype=pl.Int32))
        .join(dfelabundances_partial, how="left", on="inputcellid", coalesce=True)
        .fill_null(0.0)
    )

    print(f" took {time.perf_counter() - timestart:.1f} seconds")
    print(f" there are {nuclidesincluded} nuclides from {elementsincluded} elements included")

    return dfelabundances


def get_tar_member_extracted_path(traj_root: Path | str, particleid: int, memberfilename: str) -> Path:
    """Trajectory files are generally stored as {particleid}.tar.xz, but this is slow to access, so first check for extracted files, or decompressed .tar files, which are much faster to access.

    memberfilename: file path within the trajectory tarfile, eg. ./Run_rprocess/energy_thermo.dat
    """
    path_extracted_file = Path(traj_root, str(particleid), memberfilename)
    tarfilepaths = [
        Path(traj_root, filename)
        for filename in (
            f"{particleid}.tar",
            f"{particleid:05d}.tar",
            f"{particleid}.tar.xz",
            f"{particleid:05d}.tar.xz",
        )
    ]
    tarfilepath = next((tarfilepath for tarfilepath in tarfilepaths if tarfilepath.is_file()), None)

    if path_extracted_file.is_file():
        if path_extracted_file.stat().st_size > 0:
            with contextlib.suppress(OSError), path_extracted_file.open(encoding="utf-8") as f:
                if f.read(1):
                    return path_extracted_file

        # file is empty, so remove it
        path_extracted_file.unlink(missing_ok=True)

    # and memberfilename.endswith(".dat")
    if not path_extracted_file.is_file() and tarfilepath is not None:
        try:
            with tarfile.open(tarfilepath, "r:*") as tarfilehandle:
                tarfilehandle.extract(path=Path(traj_root, str(particleid)), member=memberfilename)
        except OSError:
            print(f"Problem extracting file {memberfilename} from {tarfilepath}")
            raise
        except KeyError:
            print(f"File {memberfilename} not found in {tarfilepath}")
            raise

    if path_extracted_file.is_file():
        return path_extracted_file

    if tarfilepath is None:
        print(f"  No network data found for particle {particleid} (so can't access {memberfilename})")
        raise FileNotFoundError

    print(f"Member {memberfilename} not found in {tarfilepath}")
    raise AssertionError


@lru_cache(maxsize=16)
def get_traj_network_timesteps(traj_root: Path, particleid: int) -> pl.DataFrame:
    with get_tar_member_extracted_path(
        traj_root=traj_root, particleid=particleid, memberfilename="./Run_rprocess/energy_thermo.dat"
    ).open(encoding="utf-8") as evolfile:
        return pl.from_pandas(
            pd.read_csv(
                evolfile,
                sep=r"\s+",
                comment="#",
                usecols=[0, 1],
                names=["nstep", "timesec"],
                engine="c",
                dtype={0: "int32[pyarrow]", 1: "float32[pyarrow]"},
                dtype_backend="pyarrow",
            )
        )


def get_closest_network_timestep(
    traj_root: Path, particleid: int, timesec: float, cond: t.Literal["lessthan", "greaterthan", "nearest"] = "nearest"
) -> int:
    """Find the closest network timestep to a given time in seconds.

    cond:
      - 'lessthan': find highest timestep less than time_sec
      - 'greaterthan': find lowest timestep greater than time_sec.
    """
    dfevol = get_traj_network_timesteps(traj_root, particleid)

    if cond == "nearest":
        idx = np.abs(dfevol["timesec"].to_numpy() - timesec).argmin()
        return int(dfevol["nstep"].to_numpy()[idx])

    if cond == "greaterthan":
        step = dfevol.filter(pl.col("timesec") > timesec).get_column("nstep").min()
        assert isinstance(step, int)
        return step

    if cond == "lessthan":
        step = dfevol.filter(pl.col("timesec") < timesec).get_column("nstep").max()
        assert isinstance(step, int)
        return step

    raise AssertionError


def get_trajectory_timestepfile_nuc_abund(
    traj_root: Path, particleid: int, memberfilename: str
) -> tuple[pl.DataFrame, float]:
    """Get the nuclear abundances for a particular trajectory id number and time memberfilename should be something like "./Run_rprocess/tday_nz-plane"."""
    with get_tar_member_extracted_path(traj_root=traj_root, particleid=particleid, memberfilename=memberfilename).open(
        encoding="utf-8"
    ) as trajfile:
        try:
            _, str_t_model_init_seconds, _, _, _, _ = trajfile.readline().split()
        except ValueError as exc:
            msg = f"Problem with {memberfilename} for traj {particleid}"
            print(msg)
            raise ValueError(msg) from exc

        t_model_init_seconds = float(str_t_model_init_seconds)
        trajfile.seek(0)

        trajfile.seek(0)
        dfnucabund = (
            pl.read_csv(trajfile, separator="\n", skip_rows=1, has_header=False, new_columns=["data"], n_threads=1)
            .lazy()
            .select(
                pl.col("data").str.slice(0, 4).str.strip_chars().cast(pl.Int32).alias("N"),
                pl.col("data").str.slice(4, 4).str.strip_chars().cast(pl.Int32).alias("Z"),
                pl.col("data").str.slice(8, 13).str.strip_chars().cast(pl.Float64).alias("log10abund"),
            )
            .with_columns(massfrac=(pl.col("N") + pl.col("Z")) * (10 ** pl.col("log10abund")))
            .drop("log10abund")
            .collect()
        )

    return dfnucabund, t_model_init_seconds


def get_trajectory_qdotintegral(particleid: int, traj_root: Path, nts_max: int, t_model_s: float) -> float:
    """Calculate initial cell energy [erg/g] from reactions t < t_model_s (reduced by work done)."""
    from scipy import integrate

    with get_tar_member_extracted_path(
        traj_root=traj_root, particleid=particleid, memberfilename="./Run_rprocess/energy_thermo.dat"
    ).open(encoding="utf-8") as enthermofile:
        try:
            dfthermo = pl.from_pandas(
                pd.read_csv(
                    enthermofile,
                    sep=r"\s+",
                    usecols=["time/s", "Qdot"],
                    engine="c",
                    dtype={0: "float32[pyarrow]", 1: "float32[pyarrow]"},
                    dtype_backend="pyarrow",
                )
            )
        except pd.errors.EmptyDataError:
            print(f"Problem with file {enthermofile}")
            raise

        dfthermo = dfthermo.rename({"time/s": "time_s"})
        startindex: int = int(np.argmax(dfthermo["time_s"] >= 1))  # start integrating at this number of seconds

        assert all(dfthermo["Qdot"][startindex : nts_max + 1] >= 0.0)

        dfthermo = dfthermo.with_columns(Qdot_expansionadjusted=pl.col("Qdot") * pl.col("time_s") / t_model_s)

        qdotintegral: float = integrate.trapezoid(
            y=dfthermo["Qdot_expansionadjusted"][startindex : nts_max + 1],
            x=dfthermo["time_s"][startindex : nts_max + 1],
        )
        assert qdotintegral >= 0.0

    return qdotintegral


def get_trajectory_abund_q(
    particleid: int,
    traj_root: Path,
    t_model_s: float | None = None,
    nts: int | None = None,  # GSI network timestep number
    getqdotintegral: bool = False,
) -> dict[tuple[int, int] | str, float]:
    """Get the nuclear mass fractions (and Qdotintegral) for a particle particle number as a given time."""
    assert t_model_s is not None or nts is not None
    try:
        if nts is not None:
            memberfilename = f"./Run_rprocess/nz-plane{nts:05d}"
        elif t_model_s is not None:
            # find the closest timestep to the required time
            nts = get_closest_network_timestep(traj_root, particleid, t_model_s)
            memberfilename = f"./Run_rprocess/nz-plane{nts:05d}"
        else:
            msg = "Either t_model_s or nts must be specified"
            raise ValueError(msg)

        dftrajnucabund, traj_time_s = get_trajectory_timestepfile_nuc_abund(traj_root, particleid, memberfilename)

        if t_model_s is None:
            t_model_s = traj_time_s

    except FileNotFoundError:
        # print(f" WARNING {particleid}.tar.xz file not found! ")
        return {}

    massfractotal = dftrajnucabund["massfrac"].sum()
    dftrajnucabund = dftrajnucabund.filter(pl.col("Z") >= 1)

    # print(f'trajectory particle id {particleid} massfrac sum: {massfractotal:.2f}')
    # print(f' grid snapshot: {t_model_s:.2e} s, network: {traj_time_s:.2e} s (timestep {nts})')
    assert np.isclose(massfractotal, 1.0, rtol=0.02)
    if not np.isclose(traj_time_s, t_model_s, rtol=0.2, atol=1.0):
        msg = f"ERROR: particle {particleid} step time of {traj_time_s} is not similar to target {t_model_s} seconds"
        raise AssertionError(msg)

    dict_traj_nuc_abund: dict[tuple[int, int] | str, float] = {
        (Z, N): massfrac / massfractotal for Z, N, massfrac in dftrajnucabund[["Z", "N", "massfrac"]].iter_rows()
    }

    if getqdotintegral:
        # set the cell energy at model time [erg/g]
        dict_traj_nuc_abund["q"] = get_trajectory_qdotintegral(
            particleid=particleid, traj_root=traj_root, nts_max=nts, t_model_s=t_model_s
        )

    return dict_traj_nuc_abund


def get_gridparticlecontributions(gridcontribpath: Path | str) -> pl.DataFrame:
    return pl.read_csv(
        at.firstexisting("gridcontributions.txt", folder=gridcontribpath, tryzipped=True),
        has_header=True,
        separator=" ",
        schema_overrides={
            "particleid": pl.Int32,
            "cellindex": pl.Int32,
            "frac_of_cellmass": pl.Float64,
            "frac_of_cellmass_includemissing": pl.Float64,
        },
    )


def particlenetworkdatafound(traj_root: Path, particleid: int) -> bool:
    tarfilepaths = [
        Path(traj_root, filename)
        for filename in (
            f"{particleid}.tar",
            f"{particleid:05d}.tar",
            f"{particleid}.tar.xz",
            f"{particleid:05d}.tar.xz",
        )
    ]
    return any(tarfilepath.is_file() for tarfilepath in tarfilepaths)


def filtermissinggridparticlecontributions(traj_root: Path, dfcontribs: pl.DataFrame) -> pl.DataFrame:
    missing_particleids = [
        particleid
        for particleid in sorted(dfcontribs["particleid"].unique())
        if not particlenetworkdatafound(traj_root, particleid)
    ]
    print(
        f"Adding gridcontributions column that excludes {len(missing_particleids)} "
        "particles without abundance data and renormalising...",
        end="",
    )
    # after filtering, frac_of_cellmass_includemissing will still include particles with rho but no abundance data
    # frac_of_cellmass will exclude particles with no abundances
    dfcontribs = dfcontribs.with_columns(frac_of_cellmass_includemissing=pl.col("frac_of_cellmass")).with_columns(
        frac_of_cellmass=pl.when(pl.col("particleid").is_in(missing_particleids))
        .then(0.0)
        .otherwise(pl.col("frac_of_cellmass"))
    )

    cell_frac_sum: dict[int, float] = {}
    cell_frac_includemissing_sum: dict[int, float] = {}
    for (cellindex,), dfparticlecontribs in dfcontribs.group_by(["cellindex"]):
        assert isinstance(cellindex, int)
        cell_frac_sum[cellindex] = dfparticlecontribs["frac_of_cellmass"].sum()
        cell_frac_includemissing_sum[cellindex] = dfparticlecontribs["frac_of_cellmass_includemissing"].sum()

    dfcontribs = (
        dfcontribs.lazy()
        .with_columns([
            pl.Series(
                (
                    row["frac_of_cellmass"] / cell_frac_sum[row["cellindex"]]
                    if cell_frac_sum[row["cellindex"]] > 0.0
                    else 0.0
                )
                for row in dfcontribs.iter_rows(named=True)
            ).alias("frac_of_cellmass"),
            pl.Series(
                (
                    row["frac_of_cellmass_includemissing"] / cell_frac_includemissing_sum[row["cellindex"]]
                    if cell_frac_includemissing_sum[row["cellindex"]] > 0.0
                    else 0.0
                )
                for row in dfcontribs.iter_rows(named=True)
            ).alias("frac_of_cellmass_includemissing"),
        ])
        .collect()
    )

    for _, dfparticlecontribs in dfcontribs.group_by(["cellindex"]):
        frac_sum: float = dfparticlecontribs["frac_of_cellmass"].sum()
        assert frac_sum == 0.0 or np.isclose(frac_sum, 1.0, rtol=0.02)

        cell_frac_includemissing_sum_thiscell: float = dfparticlecontribs["frac_of_cellmass_includemissing"].sum()
        assert cell_frac_includemissing_sum_thiscell == 0.0 or np.isclose(
            cell_frac_includemissing_sum_thiscell, 1.0, rtol=0.02
        )

    print("done")

    return dfcontribs


def save_gridparticlecontributions(dfcontribs: pl.DataFrame, gridcontribpath: Path | str) -> None:
    gridcontribpath = Path(gridcontribpath)
    if gridcontribpath.is_dir():
        gridcontribpath /= "gridcontributions.txt"
    if gridcontribpath.is_file():
        oldfile = gridcontribpath.rename(gridcontribpath.with_suffix(".bak"))
        print(f"{gridcontribpath} already exists. Renaming existing file to {oldfile}")

    dfcontribs.write_csv(gridcontribpath, separator=" ", float_scientific=True, float_precision=7)


def add_abundancecontributions(
    dfgridcontributions: pl.DataFrame,
    dfmodel: pl.LazyFrame | pl.DataFrame,
    t_model_days_incpremerger: float,
    traj_root: Path | str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Contribute trajectory network calculation abundances to model cell abundances and return dfmodel, dfelabundances, dfcontribs."""
    t_model_s = t_model_days_incpremerger * 86400
    dfcontribs = dfgridcontributions

    dfmodel = dfmodel.lazy().collect()
    if "X_Fegroup" not in dfmodel.columns:
        dfmodel = dfmodel.with_columns(pl.lit(1.0).alias("X_Fegroup"))

    traj_root = Path(traj_root)
    dfcontribs = filtermissinggridparticlecontributions(traj_root, dfcontribs).sort("particleid")
    active_inputcellcount = dfcontribs["cellindex"].unique().shape[0]

    particleids = dfcontribs["particleid"].unique()

    print(
        f"{active_inputcellcount} of {len(dfmodel)} model cells have >0 particles contributing "
        f"({len(dfcontribs)} cell contributions from {len(particleids)} particles)"
    )

    print("Reading trajectory abundances...")
    timestart = time.perf_counter()
    trajworker = partial(get_trajectory_abund_q, t_model_s=t_model_s, traj_root=traj_root, getqdotintegral=True)

    if at.get_config()["num_processes"] > 1:
        with at.get_multiprocessing_pool() as pool:
            list_traj_nuc_abund = pool.map(trajworker, particleids)
            pool.close()
            pool.join()
    else:
        list_traj_nuc_abund = [trajworker(particleid) for particleid in particleids]

    n_missing_particles = len([d for d in list_traj_nuc_abund if not d])
    print(f"  {n_missing_particles} particles are missing network abundance data out of {len(particleids)}")

    assert len(particleids) > n_missing_particles

    allkeys = list(set(chain.from_iterable(list_traj_nuc_abund)))

    dfnucabundances = pl.DataFrame({
        f"particle_{particleid}": [traj_nuc_abund.get(k, 0.0) for k in allkeys]
        for particleid, traj_nuc_abund in zip(particleids, list_traj_nuc_abund, strict=False)
    }).with_columns(pl.all().cast(pl.Float64))

    del list_traj_nuc_abund
    gc.collect()

    print(f"Reading trajectory abundances took {time.perf_counter() - timestart:.1f} seconds")

    timestart = time.perf_counter()
    print("Creating dfnucabundances...", end="", flush=True)

    groupbycellindex: t.Any = dfcontribs.group_by(["cellindex"])
    dfnucabundanceslz = dfnucabundances.lazy().with_columns([
        pl.sum_horizontal([
            pl.col(f"particle_{particleid}") * pl.lit(frac_of_cellmass)
            for particleid, frac_of_cellmass in dfthiscellcontribs[["particleid", "frac_of_cellmass"]].iter_rows()
        ]).alias(str(cellindex))
        for (cellindex,), dfthiscellcontribs in groupbycellindex
    ])

    colnames = [key if isinstance(key, str) else f"X_{at.get_elsymbol(key[0])}{key[0] + key[1]}" for key in allkeys]

    dfnucabundances = (
        dfnucabundanceslz.drop([col for col in dfnucabundances.columns if col.startswith("particle_")])
        .collect()
        .transpose(include_header=True, column_names=colnames, header_name="inputcellid")
        .with_columns(pl.col("inputcellid").cast(pl.Int32))
    )
    print(f" took {time.perf_counter() - timestart:.1f} seconds")

    dfelabundances = get_dfelemabund_from_dfmodel(dfmodel, dfnucabundances)

    timestart = time.perf_counter()
    print("Merging isotopic abundances into dfmodel...", end="", flush=True)
    dfmodel = dfmodel.join(dfnucabundances, how="left", on="inputcellid", coalesce=True).fill_null(0)
    print(f" took {time.perf_counter() - timestart:.1f} seconds")

    return dfmodel, dfelabundances, dfcontribs


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Create ARTIS model from single trajectory abundances."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    traj_root = Path(
        Path.home() / "Google Drive/Shared Drives/GSI NSM/Mergers/SFHo_long/Trajectory_SFHo_long-radius-entropy"
    )
    # particleid = 88969  # Ye = 0.0963284224
    particleid = 133371  # Ye = 0.403913230
    print(f"trajectory particle id {particleid}")
    dfnucabund, t_model_init_seconds = get_trajectory_timestepfile_nuc_abund(
        traj_root, particleid, "./Run_rprocess/tday_nz-plane"
    )
    dfnucabund = dfnucabund.filter(pl.col("Z") >= 1)
    dfnucabund["radioactive"] = True

    t_model_init_days = t_model_init_seconds / (24 * 60 * 60)

    wollaeger_profilename = "wollaeger_ejectaprofile_10bins.txt"
    if Path(wollaeger_profilename).exists():
        dfdensities = get_wollaeger_density_profile(wollaeger_profilename, t_model_init_seconds)
    else:
        rho = 1e-11
        print(f"{wollaeger_profilename} not found. Using rho {rho} g/cm³")
        dfdensities = pl.DataFrame({"mgi": 0, "rho": rho, "vel_r_max_kmps": 6.0e4})

    dfdensities["inputcellid"] = dfdensities["mgi"] + 1
    # print(dfdensities)

    # write abundances.txt
    dictelemabund = get_elemabund_from_nucabund(dfnucabund)

    dfelabundances = pl.DataFrame([dictelemabund | {"inputcellid": mgi + 1} for mgi in range(len(dfdensities))])
    at.inputmodel.save_initelemabundances(dfelabundances=dfelabundances, outpath=args.outputpath)

    # write model.txt

    rowdict = {
        "X_Fegroup": 1.0,
        "X_Ni56": 0.0,
        "X_Co56": 0.0,
        "X_Fe52": 0.0,
        "X_Cr48": 0.0,
        "X_Ni57": 0.0,
        "X_Co57": 0.0,
    }

    for row in dfnucabund.filter(pl.col("radioactive")).iter_rows(named=True):
        A = row["N"] + row["Z"]
        rowdict[f"X_{at.get_elsymbol(row['Z'])}{A}"] = row["massfrac"]

    dfmodel = pl.DataFrame([
        {
            "inputcellid": densityrow["inputcellid"],
            "vel_r_max_kmps": densityrow["vel_r_max_kmps"],
            "logrho": math.log10(densityrow["rho"]),
        }
        | rowdict
        for densityrow in dfdensities.iter_rows(named=True)
    ])
    at.inputmodel.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, filepath=Path(args.outputpath))

    with Path(args.outputpath, "gridcontributions.txt").open("w", encoding="utf-8") as fcontribs:
        fcontribs.write("particleid cellindex frac_of_cellmass\n")
        fcontribs.writelines(f"{particleid} {inputcellid} 1.0\n" for inputcellid in dfmodel["inputcellid"])


def get_wollaeger_density_profile(wollaeger_profilename: Path | str, t_model_init_seconds: float) -> pl.DataFrame:
    wollaeger_profilename = Path(wollaeger_profilename)
    print(f"{wollaeger_profilename} found")
    with Path(wollaeger_profilename).open("rt", encoding="utf-8") as f:
        t_model_init_days_in = float(f.readline().strip().removesuffix(" day"))
    t_model_init_seconds_in = t_model_init_days_in * 24 * 60 * 60

    return (
        pl.from_pandas(
            pd.read_csv(wollaeger_profilename, sep=r"\s+", skiprows=1, names=["cellid", "vel_r_max_kmps", "rho"])
        )
        .with_columns(pl.col("mgi").cast(pl.Int32))
        .with_columns(vel_r_min_kmps=pl.col("vel_r_max_kmps").shift(n=1, fill_value=0.0))
        .with_columns(
            mass_g=pl.col("rho")
            * 4.0
            / 3.0
            * math.pi
            * (pl.col("vel_r_max_kmps") ** 3 - pl.col("vel_r_min_kmps") ** 3)
            * (1e5 * t_model_init_seconds_in) ** 3
        )
        .with_columns(
            # now replace the density at the input time with the density at required time
            rho=pl.col("mass_g")
            / (
                4.0
                / 3.0
                * math.pi
                * (pl.col("vel_r_max_kmps") ** 3 - pl.col("vel_r_min_kmps") ** 3)
                * (1e5 * t_model_init_seconds) ** 3
            )
        )
    )


if __name__ == "__main__":
    main()
