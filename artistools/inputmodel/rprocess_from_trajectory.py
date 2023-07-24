#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
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

import artistools as at


def get_elemabund_from_nucabund(dfnucabund: pd.DataFrame) -> dict[str, float]:
    """Return a dictionary of elemental abundances from nuclear abundance DataFrame."""
    dictelemabund: dict[str, float] = {
        f"X_{at.get_elsymbol(atomic_number)}": dfnucabund.query("Z == @atomic_number", inplace=False).massfrac.sum()
        for atomic_number in range(1, dfnucabund.Z.max() + 1)
    }
    return dictelemabund


def open_tar_file_or_extracted(traj_root: Path, particleid: int, memberfilename: str):
    """Trajectory files are generally stored as {particleid}.tar.xz, but this is slow
    to access, so first check for extracted files, or decompressed .tar files,
    which are much faster to access.

    memberfilename: file path within the trajectory tarfile, eg. ./Run_rprocess/evol.dat
    """
    path_extracted_file = Path(traj_root, str(particleid), memberfilename)
    tarfilepath = Path(traj_root, f"{particleid}.tar")
    if not tarfilepath.is_file():
        tarfilepath = Path(traj_root, f"{particleid}.tar.xz")

    if memberfilename.endswith(".dat") and not path_extracted_file.is_file():
        tarfile.open(tarfilepath, "r:*").extract(path=Path(traj_root, str(particleid)), member=memberfilename)

    if path_extracted_file.is_file():
        return path_extracted_file.open(encoding="utf-8")

    if not tarfilepath.is_file():
        print(f"No network data found for particle {particleid} (so can't access {memberfilename})")
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
        return dfevol["nstep"].to_numpy()[idx]

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
) -> dict[str, float]:
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
        # print(f' WARNING {particleid}.tar.xz file not found! ')
        return {}

    massfractotal = dftrajnucabund.massfrac.sum()
    dftrajnucabund = dftrajnucabund.loc[dftrajnucabund["Z"] >= 1]

    dftrajnucabund["nucabundcolname"] = [
        f"X_{at.get_elsymbol(int(row.Z))}{int(row.N + row.Z)}" for row in dftrajnucabund.itertuples()
    ]

    colmassfracs = list(dftrajnucabund[["nucabundcolname", "massfrac"]].itertuples(index=False))
    colmassfracs.sort(key=lambda row: at.get_z_a_nucname(row[0]))

    # print(f'trajectory particle id {particleid} massfrac sum: {massfractotal:.2f}')
    # print(f' grid snapshot: {t_model_s:.2e} s, network: {traj_time_s:.2e} s (timestep {nts})')
    assert np.isclose(massfractotal, 1.0, rtol=0.02)
    if t_model_s is not None:
        assert np.isclose(traj_time_s, t_model_s, rtol=0.2, atol=1.0)

    dict_traj_nuc_abund = {nucabundcolname: massfrac / massfractotal for nucabundcolname, massfrac in colmassfracs}

    if getqdotintegral:
        # set the cell energy at model time [erg/g]
        dict_traj_nuc_abund["q"] = get_trajectory_qdotintegral(
            particleid=particleid, traj_root=traj_root, nts_max=nts, t_model_s=t_model_s
        )

    return dict_traj_nuc_abund


def get_modelcellabundance(
    dict_traj_nuc_abund: dict[int, pd.DataFrame], minparticlespercell: int, cellgroup: tuple[int, pd.DataFrame]
) -> dict[str, float] | None:
    cellindex: int
    dfthiscellcontribs: pd.DataFrame
    cellindex, dfthiscellcontribs = cellgroup

    if len(dfthiscellcontribs) < minparticlespercell:
        return None

    contribparticles = [
        (dict_traj_nuc_abund[particleid], frac_of_cellmass)
        for particleid, frac_of_cellmass in dfthiscellcontribs[["particleid", "frac_of_cellmass"]].itertuples(
            index=False
        )
        if particleid in dict_traj_nuc_abund
    ]

    # adjust frac_of_cellmass for missing particles
    cell_frac_sum = sum(frac_of_cellmass for _, frac_of_cellmass in contribparticles)

    nucabundcolnames = {
        col for particleid in dfthiscellcontribs.particleid for col in dict_traj_nuc_abund.get(particleid, {})
    }

    row = {
        nucabundcolname: sum(
            frac_of_cellmass * traj_nuc_abund.get(nucabundcolname, 0.0) / cell_frac_sum
            for traj_nuc_abund, frac_of_cellmass in contribparticles
        )
        for nucabundcolname in nucabundcolnames
    }

    row["inputcellid"] = cellindex

    # if n % 100 == 0:
    #     functime = time.perf_counter() - timefuncstart
    #     print(f'cell id {cellindex:6d} ('
    #           f'{n:4d} of {active_inputcellcount:4d}, {n / active_inputcellcount * 100:4.1f}%) '
    #           f' contributing {len(dfthiscellcontribs):4d} particles.'
    #           f' total func time {functime:.1f} s, {n / functime:.1f} cell/s,'
    #           f' expected time: {functime / n * active_inputcellcount:.1f}')
    return row


def get_gridparticlecontributions(gridcontribpath: Path | str) -> pd.DataFrame:
    return pd.read_csv(
        at.zopen(Path(gridcontribpath, "gridcontributions.txt")),
        delim_whitespace=True,
        dtype={
            0: "int32[pyarrow]",
            1: "int32[pyarrow]",
            2: "float32[pyarrow]",
            3: "float32[pyarrow]",
        },
        dtype_backend="pyarrow",
    )


def particlenetworkdatafound(traj_root: Path, particleid: int) -> bool:
    return (
        (traj_root / f"{particleid}.tar.xz").is_file()
        or (traj_root / f"{particleid}.tar").is_file()
        or (traj_root / str(particleid)).is_dir()
    )


def filtermissinggridparticlecontributions(traj_root: Path, dfcontribs: pd.DataFrame) -> pd.DataFrame:
    missing_particleids = [
        particleid
        for particleid in sorted(dfcontribs.particleid.unique())
        if not particlenetworkdatafound(traj_root, particleid)
    ]
    print(
        f"Adding gridcontributions column that excludes {len(missing_particleids)} "
        "particles without abundance data and renormalising...",
        end="",
    )
    # after filtering, frac_of_cellmass_includemissing will still include particles with rho but no abundance data
    # frac_of_cellmass will exclude particles with no abundances
    dfcontribs["frac_of_cellmass_includemissing"] = dfcontribs["frac_of_cellmass"]
    # dfcontribs.query('particleid not in @missing_particleids', inplace=True)
    dfcontribs.loc[dfcontribs.eval("particleid in @missing_particleids"), "frac_of_cellmass"] = 0.0

    dfcontribs["frac_of_cellmass"] = [
        row.frac_of_cellmass if row.particleid not in missing_particleids else 0.0 for row in dfcontribs.itertuples()
    ]

    cell_frac_sum: dict[int, float] = {}
    cell_frac_includemissing_sum: dict[int, float] = {}
    for cellindex, dfparticlecontribs in dfcontribs.groupby("cellindex"):
        cell_frac_sum[cellindex] = dfparticlecontribs.frac_of_cellmass.sum()
        cell_frac_includemissing_sum[cellindex] = dfparticlecontribs.frac_of_cellmass_includemissing.sum()

    dfcontribs["frac_of_cellmass"] = [
        row.frac_of_cellmass / cell_frac_sum[row.cellindex] if cell_frac_sum[row.cellindex] > 0.0 else 0.0
        for row in dfcontribs.itertuples()
    ]

    dfcontribs["frac_of_cellmass_includemissing"] = [
        (
            row.frac_of_cellmass_includemissing / cell_frac_includemissing_sum[row.cellindex]
            if cell_frac_includemissing_sum[row.cellindex] > 0.0
            else 0.0
        )
        for row in dfcontribs.itertuples()
    ]

    for cellindex, dfparticlecontribs in dfcontribs.groupby("cellindex"):
        frac_sum: float = dfparticlecontribs.frac_of_cellmass.sum()
        assert frac_sum == 0.0 or np.isclose(frac_sum, 1.0, rtol=0.02)

        cell_frac_includemissing_sum_thiscell: float = dfparticlecontribs.frac_of_cellmass_includemissing.sum()
        assert cell_frac_includemissing_sum_thiscell == 0.0 or np.isclose(
            cell_frac_includemissing_sum_thiscell, 1.0, rtol=0.02
        )

    print("done")

    return dfcontribs


def save_gridparticlecontributions(dfcontribs: pd.DataFrame, gridcontribpath) -> None:
    gridcontribpath = Path(gridcontribpath)
    if gridcontribpath.is_dir():
        gridcontribpath = Path(gridcontribpath, "gridcontributions.txt")
    dfcontribs.to_csv(gridcontribpath, sep=" ", index=False, float_format="%.7e")


def add_abundancecontributions(
    dfgridcontributions: pd.DataFrame,
    dfmodel: pd.DataFrame,
    t_model_days_incpremerger: float,
    traj_root: Path,
    minparticlespercell: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Contribute trajectory network calculation abundances to model cell abundances."""
    t_model_s = t_model_days_incpremerger * 86400
    dfcontribs = dfgridcontributions

    if "X_Fegroup" not in dfmodel.columns:
        dfmodel = pd.concat([dfmodel, pd.DataFrame({"X_Fegroup": np.ones(len(dfmodel))})], axis=1)

    active_inputcellids = [
        cellindex
        for cellindex, dfthiscellcontribs in dfcontribs.groupby("cellindex")
        if len(dfthiscellcontribs) >= minparticlespercell
    ]

    dfcontribs = dfcontribs.query("cellindex in @active_inputcellids")
    dfcontribs = filtermissinggridparticlecontributions(traj_root, dfcontribs)
    active_inputcellids = dfcontribs.cellindex.unique()
    active_inputcellcount = len(active_inputcellids)

    dfcontribs_particlegroups = dfcontribs.groupby("particleid")
    particle_count = len(dfcontribs_particlegroups)

    print(
        f"{active_inputcellcount} of {len(dfmodel)} model cells have >={minparticlespercell} particles contributing "
        f"({len(dfcontribs)} cell contributions from {particle_count} particles after filter)"
    )

    listcellnucabundances = []
    print("Reading trajectory abundances...")
    timestart = time.perf_counter()
    trajworker = partial(get_trajectory_abund_q, t_model_s=t_model_s, traj_root=traj_root, getqdotintegral=True)

    if at.get_config()["num_processes"] > 1:
        with multiprocessing.get_context("fork").Pool(processes=at.get_config()["num_processes"]) as pool:
            list_traj_nuc_abund = pool.map(trajworker, dfcontribs_particlegroups.groups)
            pool.close()
            pool.join()
    else:
        list_traj_nuc_abund = [trajworker(particleid) for particleid in dfcontribs_particlegroups.groups]

    n_missing_particles = len([d for d in list_traj_nuc_abund if not d])
    print(f"  {n_missing_particles} particles are missing network abundance data")

    assert particle_count > n_missing_particles

    dict_traj_nuc_abund = {
        particleid: dftrajnucabund
        for particleid, dftrajnucabund in zip(dfcontribs_particlegroups.groups, list_traj_nuc_abund)
        if dftrajnucabund
    }
    print(f"Reading trajectory abundances took {time.perf_counter() - timestart:.1f} seconds")

    print("Generating cell abundances...")
    timestart = time.perf_counter()
    dfcontribs_cellgroups = dfcontribs.groupby("cellindex")
    cellabundworker = partial(get_modelcellabundance, dict_traj_nuc_abund, minparticlespercell)

    if at.get_config()["num_processes"] > 1:
        chunksize = math.ceil(len(dfcontribs_cellgroups) / at.get_config()["num_processes"])
        with multiprocessing.get_context("fork").Pool(processes=at.get_config()["num_processes"]) as pool:
            listcellnucabundances = pool.map(cellabundworker, dfcontribs_cellgroups, chunksize=chunksize)
            pool.close()
            pool.join()
    else:
        listcellnucabundances = [cellabundworker(cellgroup) for cellgroup in dfcontribs_cellgroups]

    listcellnucabundances = [x for x in listcellnucabundances if x is not None]
    print(f"  took {time.perf_counter() - timestart:.1f} seconds")

    timestart = time.perf_counter()
    print("Creating dfnucabundances...", end="", flush=True)
    dfnucabundances = pd.DataFrame(listcellnucabundances)
    dfnucabundances = dfnucabundances.set_index("inputcellid", drop=False)
    dfnucabundances.index.name = None
    dfnucabundances = dfnucabundances.fillna(0.0)
    print(f" took {time.perf_counter() - timestart:.1f} seconds")

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

    dfelabundances_partial = pd.DataFrame(
        {
            "inputcellid": dfnucabundances.inputcellid,
            **{
                f"X_{at.get_elsymbol(atomic_number)}": (
                    dfnucabundances.eval(
                        f'{" + ".join(elemisotopes[atomic_number])}',
                        engine="python" if len(elemisotopes[atomic_number]) > 31 else None,
                    )
                    if atomic_number in elemisotopes
                    else np.zeros(len(dfnucabundances))
                )
                for atomic_number in range(1, max(elemisotopes.keys()) + 1)
            },
        },
        index=dfnucabundances.index,
    )

    # ensure cells with no traj contributions are included
    dfelabundances = dfmodel[["inputcellid"]].merge(
        dfelabundances_partial, how="left", left_on="inputcellid", right_on="inputcellid"
    )
    dfnucabundances = dfnucabundances.set_index("inputcellid", drop=False)
    dfnucabundances.index.name = None
    dfelabundances = dfelabundances.fillna(0.0)
    print(f" took {time.perf_counter() - timestart:.1f} seconds")
    print(f" there are {nuclidesincluded} nuclides from {elementsincluded} elements included")
    timestart = time.perf_counter()
    print("Merging isotopic abundances into dfmodel...", end="", flush=True)
    dfmodel = dfmodel.merge(dfnucabundances, how="left", left_on="inputcellid", right_on="inputcellid")
    dfmodel = dfmodel.fillna(0.0)
    print(f" took {time.perf_counter() - timestart:.1f} seconds")

    return dfmodel, dfelabundances, dfcontribs


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")


def main(args=None, argsraw=None, **kwargs) -> None:
    """Create ARTIS model from single trajectory abundances."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=__doc__,
        )

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

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
        dfdensities = pd.DataFrame({"rho": rho, "velocity_outer": 6.0e4}, index=[0])

    # print(dfdensities)

    # write abundances.txt
    dictelemabund = get_elemabund_from_nucabund(dfnucabund)

    dfelabundances = pd.DataFrame([dict(inputcellid=mgi + 1, **dictelemabund) for mgi in range(len(dfdensities))])
    # print(dfelabundances)
    at.inputmodel.save_initelemabundances(dfelabundances=dfelabundances, abundancefilename=args.outputpath)

    # write model.txt

    rowdict = {
        # 'inputcellid': 1,
        # 'velocity_outer': 6.e4,
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
            velocity_outer=densityrow["velocity_outer"],
            logrho=math.log10(densityrow["rho"]),
            **rowdict,
        )
        for mgi, densityrow in dfdensities.iterrows()
    ]
    # print(modeldata)

    dfmodel = pd.DataFrame(modeldata)
    # print(dfmodel)
    at.inputmodel.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, modelpath=Path(args.outputpath))
    with Path(args.outputpath, "gridcontributions.txt").open("w") as fcontribs:
        fcontribs.write("particleid cellindex frac_of_cellmass\n")
        for cell in dfmodel.itertuples(index=False):
            fcontribs.write(f"{particleid} {cell.inputcellid} 1.0\n")


def get_wollaeger_density_profile(wollaeger_profilename):
    print(f"{wollaeger_profilename} found")
    t_model_init_days_in = float(Path(wollaeger_profilename).open("rt").readline().strip().removesuffix(" day"))
    result = pd.read_csv(
        wollaeger_profilename,
        delim_whitespace=True,
        skiprows=1,
        names=["cellid", "velocity_outer", "rho"],
    )
    result["cellid"] = result["cellid"].astype(int)
    result["velocity_inner"] = np.concatenate(([0.0], result["velocity_outer"].to_numpy()[:-1]))

    t_model_init_seconds_in = t_model_init_days_in * 24 * 60 * 60
    result = result.eval(
        "cellmass_grams = rho * 4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)"
        "* (1e5 * @t_model_init_seconds_in) ** 3"
    )

    # now replace the density at the input time with the density at required time

    return result.eval(
        "rho = cellmass_grams / ("
        "4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)"
        " * (1e5 * @t_model_init_seconds) ** 3)"
    )


if __name__ == "__main__":
    main()
