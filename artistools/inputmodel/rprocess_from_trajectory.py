#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import gc
import io
import math
import multiprocessing
import tarfile
import time
import typing as t
from functools import lru_cache
from functools import partial
from pathlib import Path

import argcomplete
import numpy as np
import pandas as pd
import polars as pl

import artistools as at


def get_elemabund_from_nucabund(dfnucabund: pd.DataFrame) -> dict[str, float]:
    """Return a dictionary of elemental abundances from nuclear abundance DataFrame."""
    dictelemabund: dict[str, float] = {
        f"X_{at.get_elsymbol(atomic_number)}": dfnucabund[dfnucabund["Z"] == atomic_number]["massfrac"].sum()
        for atomic_number in range(1, dfnucabund.Z.max() + 1)
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
        atomic_number = at.get_atomic_number(colname[2:].rstrip("0123456789"))
        if atomic_number in elemisotopes:
            elemisotopes[atomic_number].append(colname)
        else:
            elemisotopes[atomic_number] = [colname]

    elementsincluded = len(elemisotopes)

    dfelabundances_partial = pl.DataFrame(
        {
            "inputcellid": dfnucabundances["inputcellid"],
            **{
                f"X_{at.get_elsymbol(atomic_number)}": (
                    dfnucabundances.select(elemisotopes[atomic_number]).sum_horizontal()
                    if atomic_number in elemisotopes
                    else np.zeros(len(dfnucabundances))
                )
                for atomic_number in range(1, max(elemisotopes.keys()) + 1)
            },
        },
    )

    # ensure cells with no traj contributions are included
    dfelabundances = pl.DataFrame(pl.Series(name="inputcellid", values=dfmodel["inputcellid"], dtype=pl.Int32))

    dfelabundances = dfelabundances.join(
        dfelabundances_partial, how="left", left_on="inputcellid", right_on="inputcellid"
    ).fill_null(0.0)

    print(f" took {time.perf_counter() - timestart:.1f} seconds")
    print(f" there are {nuclidesincluded} nuclides from {elementsincluded} elements included")

    return dfelabundances


def open_tar_file_or_extracted(traj_root: Path, particleid: int, memberfilename: str):
    """Trajectory files are generally stored as {particleid}.tar.xz, but this is slow
    to access, so first check for extracted files, or decompressed .tar files,
    which are much faster to access.

    memberfilename: file path within the trajectory tarfile, eg. ./Run_rprocess/evol.dat
    """
    path_extracted_file = Path(traj_root, str(particleid), memberfilename)
    tarfilepaths = [
        Path(traj_root, filename)
        for filename in [
            f"{particleid}.tar",
            f"{particleid:05d}.tar",
            f"{particleid}.tar.xz",
            f"{particleid:05d}.tar.xz",
        ]
    ]
    tarfilepath = next((tarfilepath for tarfilepath in tarfilepaths if tarfilepath.is_file()), None)

    # and memberfilename.endswith(".dat")
    if not path_extracted_file.is_file() and tarfilepath is not None:
        try:
            tarfile.open(tarfilepath, "r:*").extract(path=Path(traj_root, str(particleid)), member=memberfilename)
        except OSError:
            print(f"Problem extracting file {memberfilename} from {tarfilepath}")
            raise

    if path_extracted_file.is_file():
        return path_extracted_file.open(encoding="utf-8")

    if tarfilepath is None:
        print(f"  No network data found for particle {particleid} (so can't access {memberfilename})")
        raise FileNotFoundError

    # print(f"using {tarfilepath} for {memberfilename}")
    # return tarfile.open(tarfilepath, "r:*").extractfile(member=memberfilename)
    with tarfile.open(tarfilepath, "r|*") as tfile:
        for tarmember in tfile:
            if tarmember.name == memberfilename:
                extractedfile = tfile.extractfile(tarmember)
                if extractedfile is not None:
                    return io.StringIO(extractedfile.read().decode("utf-8"))

    print(f"Member {memberfilename} not found in {tarfilepath}")
    raise AssertionError


@lru_cache(maxsize=16)
def get_dfevol(traj_root: Path, particleid: int) -> pd.DataFrame:
    with open_tar_file_or_extracted(traj_root, particleid, "./Run_rprocess/evol.dat") as evolfile:
        return pd.read_csv(
            evolfile,
            sep=r"\s+",
            comment="#",
            usecols=[0, 1],
            names=["nstep", "timesec"],
            engine="c",
            dtype={0: "int32[pyarrow]", 1: "float32[pyarrow]"},
            dtype_backend="pyarrow",
        )


def get_closest_network_timestep(
    traj_root: Path, particleid: int, timesec: float, cond: t.Literal["lessthan", "greaterthan", "nearest"] = "nearest"
) -> int:
    """cond:
    'lessthan': find highest timestep less than time_sec
    'greaterthan': find lowest timestep greater than time_sec.
    """
    dfevol = get_dfevol(traj_root, particleid)

    if cond == "nearest":
        idx = np.abs(dfevol.timesec.to_numpy() - timesec).argmin()
        return int(dfevol["nstep"].to_numpy()[idx])

    if cond == "greaterthan":
        return dfevol[dfevol["timesec"] > timesec]["nstep"].min()

    if cond == "lessthan":
        return dfevol[dfevol["timesec"] < timesec]["nstep"].max()

    raise AssertionError


def get_trajectory_timestepfile_nuc_abund(
    traj_root: Path, particleid: int, memberfilename: str
) -> tuple[pd.DataFrame, float]:
    """Get the nuclear abundances for a particular trajectory id number and time
    memberfilename should be something like "./Run_rprocess/tday_nz-plane".
    """
    with open_tar_file_or_extracted(traj_root, particleid, memberfilename) as trajfile:
        try:
            _, str_t_model_init_seconds, _, rho, _, _ = trajfile.readline().split()
        except ValueError as exc:
            print(f"Problem with {memberfilename}")
            msg = f"Problem with {memberfilename}"
            raise ValueError(msg) from exc

        trajfile.seek(0)
        t_model_init_seconds = float(str_t_model_init_seconds)

        dfnucabund = pd.read_fwf(
            trajfile,
            skip_blank_lines=True,
            skiprows=1,
            colspecs=[(0, 4), (4, 8), (8, 21)],
            engine="c",
            names=["N", "Z", "log10abund"],
            dtype={0: "int32[pyarrow]", 1: "int32[pyarrow]", 2: "float32[pyarrow]"},
            dtype_backend="pyarrow",
        )

        # in case the files are inconsistent, switch to an adaptive reader
        # dfnucabund = pd.read_csv(
        #     trajfile,
        #     skip_blank_lines=True,
        #     skiprows=1,
        #     sep=r"\s+",
        #     engine='c',
        #     names=["N", "Z", "log10abund", "S1n", "S2n"],
        #     usecols=["N", "Z", "log10abund"],
        #     dtype={0: int, 1: int, 2: float},
        # )

    # dfnucabund.eval('abund = 10 ** log10abund', inplace=True)
    dfnucabund["massfrac"] = (dfnucabund["N"] + dfnucabund["Z"]) * (10 ** dfnucabund["log10abund"])
    # dfnucabund.eval('A = N + Z', inplace=True)
    # dfnucabund.query('abund > 0.', inplace=True)

    # abund is proportional to number abundance, but needs normalisation
    # normfactor = dfnucabund.abund.sum()
    # print(f'abund sum: {normfactor}')
    # dfnucabund.eval('numberfrac = abund / @normfactor', inplace=True)

    return dfnucabund, t_model_init_seconds


def get_trajectory_qdotintegral(particleid: int, traj_root: Path, nts_max: int, t_model_s: float) -> float:
    """Calculate initial cell energy [erg/g] from reactions t < t_model_s (reduced by work done)."""
    with open_tar_file_or_extracted(traj_root, particleid, "./Run_rprocess/energy_thermo.dat") as enthermofile:
        try:
            dfthermo: pd.DataFrame = pd.read_csv(
                enthermofile,
                sep=r"\s+",
                usecols=["time/s", "Qdot"],
                engine="c",
                dtype={0: "float32[pyarrow]", 1: "float32[pyarrow]"},
                dtype_backend="pyarrow",
            )
        except pd.errors.EmptyDataError:
            print(f"Problem with file {enthermofile}")
            raise

        dfthermo = dfthermo.rename(columns={"time/s": "time_s"})
        startindex: int = int(np.argmax(dfthermo["time_s"] >= 1))  # start integrating at this number of seconds

        assert all(dfthermo["Qdot"][startindex : nts_max + 1] > 0.0)
        dfthermo["Qdot_expansionadjusted"] = dfthermo["Qdot"] * dfthermo["time_s"] / t_model_s

        qdotintegral: float = np.trapz(
            y=dfthermo["Qdot_expansionadjusted"][startindex : nts_max + 1],
            x=dfthermo["time_s"][startindex : nts_max + 1],
        )
        assert qdotintegral >= 0.0

    return qdotintegral


def get_trajectory_abund_q(
    particleid: int,
    traj_root: Path,
    t_model_s: float | None = None,
    nts: int | None = None,
    getqdotintegral: bool = False,
) -> dict[tuple[int, int] | str, float]:
    """Get the nuclear mass fractions (and Qdotintegral) for a particle particle number as a given time
    nts: GSI network timestep number.
    """
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

    massfractotal = dftrajnucabund.massfrac.sum()
    dftrajnucabund = dftrajnucabund.loc[dftrajnucabund["Z"] >= 1]

    # print(f'trajectory particle id {particleid} massfrac sum: {massfractotal:.2f}')
    # print(f' grid snapshot: {t_model_s:.2e} s, network: {traj_time_s:.2e} s (timestep {nts})')
    assert np.isclose(massfractotal, 1.0, rtol=0.02)
    if t_model_s is not None:
        assert np.isclose(traj_time_s, t_model_s, rtol=0.2, atol=1.0)

    dict_traj_nuc_abund: dict[tuple[int, int] | str, float] = {
        (Z, N): massfrac / massfractotal
        for Z, N, massfrac in dftrajnucabund[["Z", "N", "massfrac"]].itertuples(index=False)
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
        dtypes={
            "particleid": pl.Int32,
            "cellindex": pl.Int32,
            "frac_of_cellmass": pl.Float32,
            "frac_of_cellmass_includemissing": pl.Float32,
        },
    )


def particlenetworkdatafound(traj_root: Path, particleid: int) -> bool:
    tarfilepaths = [
        Path(traj_root, filename)
        for filename in [
            f"{particleid}.tar",
            f"{particleid:05d}.tar",
            f"{particleid}.tar.xz",
            f"{particleid:05d}.tar.xz",
        ]
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
    dfcontribs = dfcontribs.with_columns(pl.col("frac_of_cellmass").alias("frac_of_cellmass_includemissing"))
    dfcontribs = dfcontribs.with_columns(
        pl.when(pl.col("particleid").is_in(missing_particleids))
        .then(0.0)
        .otherwise(pl.col("frac_of_cellmass"))
        .alias("frac_of_cellmass")
    )

    cell_frac_sum: dict[int, float] = {}
    cell_frac_includemissing_sum: dict[int, float] = {}
    for (cellindex,), dfparticlecontribs in dfcontribs.group_by(["cellindex"]):  # type: ignore[misc]
        assert isinstance(cellindex, int)
        cell_frac_sum[cellindex] = dfparticlecontribs["frac_of_cellmass"].sum()
        cell_frac_includemissing_sum[cellindex] = dfparticlecontribs["frac_of_cellmass_includemissing"].sum()

    dfcontribs = (
        dfcontribs.lazy()
        .with_columns(
            [
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
            ]
        )
        .collect()
    )

    for (cellindex,), dfparticlecontribs in dfcontribs.group_by(["cellindex"]):  # type: ignore[misc]
        frac_sum: float = dfparticlecontribs["frac_of_cellmass"].sum()
        assert frac_sum == 0.0 or np.isclose(frac_sum, 1.0, rtol=0.02)

        cell_frac_includemissing_sum_thiscell: float = dfparticlecontribs["frac_of_cellmass_includemissing"].sum()
        assert cell_frac_includemissing_sum_thiscell == 0.0 or np.isclose(
            cell_frac_includemissing_sum_thiscell, 1.0, rtol=0.02
        )

    print("done")

    return dfcontribs


def save_gridparticlecontributions(dfcontribs: pd.DataFrame | pl.DataFrame, gridcontribpath: Path | str) -> None:
    gridcontribpath = Path(gridcontribpath)
    if gridcontribpath.is_dir():
        gridcontribpath = gridcontribpath / "gridcontributions.txt"
    if gridcontribpath.is_file():
        oldfile = gridcontribpath.rename(gridcontribpath.with_suffix(".bak"))
        print(f"{gridcontribpath} already exists. Renaming existing file to {oldfile}")

    if isinstance(dfcontribs, pl.DataFrame):
        dfcontribs = dfcontribs.to_pandas(use_pyarrow_extension_array=True)

    dfcontribs.to_csv(gridcontribpath, sep=" ", index=False, float_format="%.7e")


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
        with multiprocessing.get_context("fork").Pool(processes=at.get_config()["num_processes"]) as pool:
            list_traj_nuc_abund = pool.map(trajworker, particleids)
            pool.close()
            pool.join()
    else:
        list_traj_nuc_abund = [trajworker(particleid) for particleid in particleids]

    n_missing_particles = len([d for d in list_traj_nuc_abund if not d])
    print(f"  {n_missing_particles} particles are missing network abundance data out of {len(particleids)}")

    assert len(particleids) > n_missing_particles

    allkeys = list({k for abund in list_traj_nuc_abund for k in abund})

    dfnucabundances = pl.DataFrame(
        {
            f"particle_{particleid}": [traj_nuc_abund.get(k, 0.0) for k in allkeys]
            for particleid, traj_nuc_abund in zip(particleids, list_traj_nuc_abund)
        }
    ).with_columns(pl.all().cast(pl.Float64))

    del list_traj_nuc_abund
    gc.collect()

    print(f"Reading trajectory abundances took {time.perf_counter() - timestart:.1f} seconds")

    timestart = time.perf_counter()
    print("Creating dfnucabundances...", end="", flush=True)

    dfnucabundanceslz = dfnucabundances.lazy().with_columns(
        [  # type: ignore[misc]
            pl.sum_horizontal(
                [
                    pl.col(f"particle_{particleid}") * pl.lit(frac_of_cellmass)
                    for particleid, frac_of_cellmass in dfthiscellcontribs[
                        ["particleid", "frac_of_cellmass"]
                    ].iter_rows()
                ]
            ).alias(f"{cellindex}")
            for (cellindex,), dfthiscellcontribs in dfcontribs.group_by(["cellindex"])
        ]
    )

    colnames = [
        key if isinstance(key, str) else f"X_{at.get_elsymbol(int(key[0]))}{int(key[0] + key[1])}" for key in allkeys
    ]

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
    dfmodel = dfmodel.join(dfnucabundances, how="left", left_on="inputcellid", right_on="inputcellid").fill_null(0)
    print(f" took {time.perf_counter() - timestart:.1f} seconds")

    return dfmodel, dfelabundances, dfcontribs


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs) -> None:
    """Create ARTIS model from single trajectory abundances."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=__doc__,
        )

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
    dfnucabund = dfnucabund.iloc[dfnucabund["Z"] >= 1]
    dfnucabund["radioactive"] = True

    t_model_init_days = t_model_init_seconds / (24 * 60 * 60)

    wollaeger_profilename = "wollaeger_ejectaprofile_10bins.txt"
    if Path(wollaeger_profilename).exists():
        dfdensities = get_wollaeger_density_profile(wollaeger_profilename)
    else:
        rho = 1e-11
        print(f"{wollaeger_profilename} not found. Using rho {rho} g/cm3")
        dfdensities = pd.DataFrame({"rho": rho, "vel_r_max_kmps": 6.0e4}, index=[0])

    # print(dfdensities)

    # write abundances.txt
    dictelemabund = get_elemabund_from_nucabund(dfnucabund)

    dfelabundances = pd.DataFrame([dict(inputcellid=mgi + 1, **dictelemabund) for mgi in range(len(dfdensities))])
    # print(dfelabundances)
    at.inputmodel.save_initelemabundances(dfelabundances=dfelabundances, outpath=args.outputpath)

    # write model.txt

    rowdict = {
        # 'inputcellid': 1,
        # 'vel_r_max_kmps': 6.e4,
        # 'logrho': -3.,
        "X_Fegroup": 1.0,
        "X_Ni56": 0.0,
        "X_Co56": 0.0,
        "X_Fe52": 0.0,
        "X_Cr48": 0.0,
        "X_Ni57": 0.0,
        "X_Co57": 0.0,
    }

    for _, row in dfnucabund.query("radioactive == True").iterrows():
        A = row.N + row.Z
        rowdict[f"X_{at.get_elsymbol(row.Z)}{A}"] = row.massfrac

    modeldata = [
        dict(
            inputcellid=mgi + 1,
            vel_r_max_kmps=densityrow["vel_r_max_kmps"],
            logrho=math.log10(densityrow["rho"]),
            **rowdict,
        )
        for mgi, densityrow in dfdensities.iterrows()
    ]
    # print(modeldata)

    dfmodel = pd.DataFrame(modeldata)
    # print(dfmodel)
    at.inputmodel.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, filepath=Path(args.outputpath))
    with Path(args.outputpath, "gridcontributions.txt").open("w") as fcontribs:
        fcontribs.write("particleid cellindex frac_of_cellmass\n")
        for cell in dfmodel.itertuples(index=False):
            fcontribs.write(f"{particleid} {cell.inputcellid} 1.0\n")


def get_wollaeger_density_profile(wollaeger_profilename):
    print(f"{wollaeger_profilename} found")
    with Path(wollaeger_profilename).open("rt") as f:
        t_model_init_days_in = float(f.readline().strip().removesuffix(" day"))
    result = pd.read_csv(
        wollaeger_profilename,
        sep=r"\s+",
        skiprows=1,
        names=["cellid", "vel_r_max_kmps", "rho"],
    )
    result["cellid"] = result["cellid"].astype(int)
    result["vel_r_min_kmps"] = np.concatenate(([0.0], result["vel_r_max_kmps"].to_numpy()[:-1]))

    t_model_init_seconds_in = t_model_init_days_in * 24 * 60 * 60  # noqa: F841
    result = result.eval(
        "mass_g = rho * 4. / 3. * @math.pi * (vel_r_max_kmps ** 3 - vel_r_min_kmps ** 3)"
        "* (1e5 * @t_model_init_seconds_in) ** 3"
    )

    # now replace the density at the input time with the density at required time

    return result.eval(
        "rho = mass_g / ("
        "4. / 3. * @math.pi * (vel_r_max_kmps ** 3 - vel_r_min_kmps ** 3)"
        " * (1e5 * @t_model_init_seconds) ** 3)"
    )


if __name__ == "__main__":
    main()
