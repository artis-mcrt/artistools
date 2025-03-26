# PYTHON_ARGCOMPLETE_OK

import argparse
import contextlib
import math
import string
import typing as t
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path


def strnuc_to_latex(strnuc: str) -> str:
    """Convert a string like sr89 to $^{89}$Sr."""
    elsym = strnuc.rstrip(string.digits)
    massnum = strnuc.removeprefix(elsym)

    return rf"$^{{{massnum}}}${elsym.title()}"


def plot_qdot(
    modelpath: Path,
    dfpartcontrib: pl.DataFrame,
    lzdfmodel: pl.LazyFrame,
    modelmeta: dict[str, t.Any],
    allparticledata: dict[int, dict[str, npt.NDArray[np.floating[t.Any]]]],
    arr_time_artis_days: Sequence[float],  # noqa: ARG001
    arr_time_gsi_days: Sequence[float],
    pdfoutpath: Path | str,
    xmax: float | None = None,
) -> None:
    try:
        depdata = at.get_deposition(modelpath=modelpath)

    except FileNotFoundError:
        print("Can't do qdot plot because no deposition.out file")
        return

    heatcols = ["hbeta", "halpha", "hbfis", "hspof", "Ye", "Qdot"]

    arr_heat = {col: np.zeros_like(arr_time_gsi_days) for col in heatcols}
    series_mass_g = lzdfmodel.select("mass_g").collect().get_column("mass_g")

    model_mass_grams = series_mass_g.sum()
    print(f"model mass: {model_mass_grams / 1.989e33:.3f} Msun")

    cell_mass_fracs = series_mass_g / model_mass_grams

    print("Calculating global heating rates from the individual particle heating rates...")
    dfpartcontrib_nomissing = dfpartcontrib.filter(pl.col("particleid").is_in(allparticledata.keys()))
    for cellindex, dfpartcontribthiscell in dfpartcontrib_nomissing.group_by("cellindex"):
        assert isinstance(cellindex, int)
        mgi = cellindex - 1
        if mgi >= modelmeta["npts_model"]:
            continue
        cell_mass_frac = cell_mass_fracs[mgi]

        if cell_mass_frac == 0.0:
            continue

        frac_of_cellmass_sum = dfpartcontribthiscell["frac_of_cellmass"].sum()

        for particleid, frac_of_cellmass in dfpartcontribthiscell.select([
            "particleid",
            "frac_of_cellmass",
        ]).iter_rows():
            thisparticledata = allparticledata[particleid]
            for col in heatcols:
                arr_heat[col] += thisparticledata[col] * cell_mass_frac * frac_of_cellmass / frac_of_cellmass_sum

    print("  done.")

    nrows = 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(6, 1 + 3 * nrows),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )
    if nrows == 1:
        axes = np.array([axes])

    assert isinstance(axes, np.ndarray)
    axis = axes[0]

    # axis.set_ylim(bottom=1e7, top=2e10)
    # axis.set_xlim(left=depdata["tmid_days"].min(), right=depdata["tmid_days"].max())
    xmin = min(arr_time_gsi_days) * 0.9
    xmax = xmax or max(arr_time_gsi_days) * 1.03
    axis.set_xlim(left=xmin, right=xmax)

    # axis.set_xscale('log')

    # axis.set_xlim(left=1., right=arr_time_artis[-1])
    axes[-1].set_xlabel("Time [days]")
    axis.set_yscale("log")
    # axis.set_ylabel(f'X({strnuc})')
    axis.set_ylabel("Qdot [erg/s/g]")
    # arr_time_days, arr_qdot = zip(
    #     *[(t, qdot) for t, qdot in zip(arr_time_days, arr_qdot)
    #       if depdata['tmid_days'].min() <= t and t <= depdata['tmid_days'].max()])

    # axis.plot(arr_time_gsi_days, arr_heat['Qdot'],
    #           # linestyle='None',
    #           linewidth=2, color='black',
    #           # marker='x', markersize=8,
    #           label='Qdot GSI Network')

    # axis.plot(depdata['tmid_days'], depdata['Qdot_ana_erg/s/g'],
    #           linewidth=2, color='red',
    #           # linestyle='None',
    #           # marker='+', markersize=15,
    #           label='Qdot ARTIS')

    axis.plot(
        arr_time_gsi_days,
        arr_heat["hbeta"],
        linewidth=2,
        color="black",
        linestyle="dashed",
        # marker='x', markersize=8,
        label=r"$\dot{Q}_\beta$ GSI Network",
    )

    axis.plot(
        depdata["tmid_days"],
        depdata["Qdot_betaminus_ana_erg/s/g"],
        linewidth=2,
        color="red",
        linestyle="dashed",
        # marker='+', markersize=15,
        label=r"$\dot{Q}_\beta$ ARTIS",
    )

    axis.plot(
        arr_time_gsi_days,
        arr_heat["halpha"],
        linewidth=2,
        color="black",
        linestyle="dotted",
        # marker='x', markersize=8,
        label=r"$\dot{Q}_\alpha$ GSI Network",
    )

    axis.plot(
        depdata["tmid_days"],
        depdata["Qdotalpha_ana_erg/s/g"],
        linewidth=2,
        color="red",
        linestyle="dotted",
        # marker='+', markersize=15,
        label=r"$\dot{Q}_\alpha$ ARTIS",
    )

    axis.plot(
        arr_time_gsi_days,
        arr_heat["hbfis"],
        linewidth=2,
        linestyle="dotted",
        # marker='x', markersize=8,
        # color='black',
        label=r"$\dot{Q}_{\beta fis}$ GSI Network",
    )

    axis.plot(
        arr_time_gsi_days,
        arr_heat["hspof"],
        linewidth=2,
        linestyle="dotted",
        # marker='x', markersize=8,
        # color='black',
        label=r"$\dot{Q}_{sponfis}$ GSI Network",
    )

    axis.legend(loc="best", frameon=False, handlelength=1, ncol=3, numpoints=1)

    # fig.suptitle(f'{at.get_model_name(modelpath)}', fontsize=10)
    at.plottools.autoscale(axis, margin=0.0)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"Saved {pdfoutpath}")


def plot_cell_abund_evolution(
    modelpath: Path,  # noqa: ARG001
    dfpartcontrib: pl.DataFrame,
    allparticledata: dict[int, dict[str, npt.NDArray[np.floating[t.Any]]]],
    arr_time_artis_days: Sequence[float],
    arr_time_gsi_days: Sequence[float],
    arr_strnuc: Sequence[str],
    arr_abund_artis: dict[str, list[float]],
    t_model_init_days: float,
    dfcell: pl.DataFrame,
    pdfoutpath: Path,
    mgi: int,
    hideinputmodelpoints: bool = True,
    xmax: float | None = None,
) -> None:
    dfpartcontrib_thiscell = dfpartcontrib.filter(
        (pl.col("cellindex") == (mgi + 1)) & (pl.col("particleid").is_in(allparticledata.keys()))
    )
    frac_of_cellmass_sum = dfpartcontrib_thiscell["frac_of_cellmass"].sum()
    print(f"frac_of_cellmass_sum: {frac_of_cellmass_sum} (can be < 1.0 because of missing particles)")
    # if arr_strnuc[0] != 'Ye':
    #     arr_strnuc.insert(0, 'Ye')

    arr_abund_gsi: dict[str, np.ndarray[t.Any, np.dtype[np.floating[t.Any]]]] = {
        strnuc: np.zeros_like(arr_time_gsi_days) for strnuc in arr_strnuc
    }

    # calculate the GSI values from the particles contributing to this cell
    for particleid, frac_of_cellmass in dfpartcontrib_thiscell.select(["particleid", "frac_of_cellmass"]).iter_rows():
        frac_of_cellmass = dfpartcontrib_thiscell.filter(pl.col("particleid") == particleid)["frac_of_cellmass"].sum()

        for strnuc in arr_strnuc:
            arr_abund_gsi[strnuc] += allparticledata[particleid][strnuc] * frac_of_cellmass / frac_of_cellmass_sum

    fig, axes = plt.subplots(
        nrows=len(arr_strnuc),
        ncols=1,
        sharex=False,
        sharey=False,
        figsize=(6, 1 + 2.0 * len(arr_strnuc)),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )
    fig.subplots_adjust(top=0.8)
    # axis.set_xscale('log')
    assert isinstance(axes, np.ndarray)
    axes[-1].set_xlabel("Time [days]")
    axis = axes[0]
    print("nuc gsi_abund artis_abund")
    for axis, strnuc in zip(axes, arr_strnuc, strict=False):
        # print(arr_time_artis_days)
        xmin = min(arr_time_gsi_days) * 0.9
        xmax = xmax or max(arr_time_gsi_days) * 1.03
        axis.set_xlim(left=xmin, right=xmax)
        # axis.set_yscale('log')
        # axis.set_ylabel(f'X({strnuc})')
        if strnuc == "Ye":
            axis.set_ylabel("Electron fraction")
        else:
            axis.set_ylabel("Mass fraction")

        strnuc_latex = strnuc_to_latex(strnuc)

        axis.plot(
            arr_time_gsi_days,
            arr_abund_gsi[strnuc],
            # linestyle='None',
            linewidth=2,
            marker="x",
            markersize=8,
            label=f"{strnuc_latex} Network",
            color="black",
        )

        if strnuc in arr_abund_artis:
            axis.plot(
                arr_time_artis_days,
                arr_abund_artis[strnuc],
                linewidth=2,
                # linestyle='None',
                # marker='+', markersize=15,
                label=f"{strnuc_latex} ARTIS",
                color="red",
            )

        print(f"{strnuc} {arr_abund_gsi[strnuc][0]:.2e} {arr_abund_artis[strnuc][0]:.2e}")
        if f"X_{strnuc}" in dfcell and not hideinputmodelpoints:
            axis.plot(
                t_model_init_days,
                dfcell[f"X_{strnuc}"],
                marker="+",
                markersize=15,
                markeredgewidth=2,
                label=f"{strnuc_latex} ARTIS inputmodel",
                color="blue",
            )

        axis.legend(loc="best", frameon=False, handlelength=1, ncol=1, numpoints=1)

        at.plottools.autoscale(ax=axis)

    # fig.suptitle(f"{at.get_model_name(modelpath)} cell {mgi}", y=0.995, fontsize=10)
    at.plottools.autoscale(axis, margin=0.05)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"Saved {pdfoutpath}")


def get_particledata(
    arr_time_s: Sequence[float] | npt.NDArray[np.floating[t.Any]],
    arr_strnuc_z_n: list[tuple[str, int, int]],
    traj_root: Path,
    particleid: int,
    verbose: bool = False,
) -> tuple[int, dict[str, npt.NDArray[np.floating]]]:
    """For an array of times (NSM time including time before merger), interpolate the heating rates of various decay channels and (if arr_strnuc is not empty) the nuclear mass fractions."""
    try:
        nts_min = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            traj_root, particleid, timesec=min(float(x) for x in arr_time_s), cond="lessthan"
        )
        nts_max = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            traj_root, particleid, timesec=max(float(x) for x in arr_time_s), cond="greaterthan"
        )

    except FileNotFoundError:
        print(f"No network calculation for particle {particleid}")
        # make sure we weren't requesting abundance data for this particle that has no network data
        if arr_strnuc_z_n:
            print("ERROR:", particleid, arr_strnuc_z_n)
        assert not arr_strnuc_z_n
        return -1, {}

    if verbose:
        print(
            "Reading network calculation heating.dat,"
            f" energy_thermo.dat{', and nz-plane abundances' if arr_strnuc_z_n else ''} for particle {particleid}..."
        )

    particledata = {}
    nstep_timesec = {}
    with get_tar_member_extracted_path(
        traj_root=traj_root, particleid=particleid, memberfilename="./Run_rprocess/heating.dat"
    ).open(encoding="utf-8") as f:
        dfheating = pd.read_csv(f, sep=r"\s+", usecols=["#count", "time/s", "hbeta", "halpha", "hbfis", "hspof"])
        heatcols = ["hbeta", "halpha", "hbfis", "hspof"]

        heatrates_in: dict[str, list[float]] = {col: [] for col in heatcols}
        arr_time_s_source = []
        for _, row in dfheating.iterrows():
            nstep_timesec[row["#count"]] = row["time/s"]
            arr_time_s_source.append(row["time/s"])
            for col in heatcols:
                try:
                    heatrates_in[col].append(float(row[col]))
                except ValueError:
                    heatrates_in[col].append(float(row[col].replace("-", "e-")))

        for col in heatcols:
            particledata[col] = np.array(np.interp(arr_time_s, arr_time_s_source, heatrates_in[col]))

    with get_tar_member_extracted_path(
        traj_root=traj_root, particleid=particleid, memberfilename="./Run_rprocess/energy_thermo.dat"
    ).open(encoding="utf-8") as f:
        storecols = ["Qdot", "Ye"]

        dfthermo = pd.read_csv(f, sep=r"\s+", usecols=["#count", "time/s", *storecols])

        data_in: dict[str, list[float]] = {col: [] for col in storecols}
        arr_time_s_source = []
        for _, row in dfthermo.iterrows():
            nstep_timesec[row["#count"]] = row["time/s"]
            arr_time_s_source.append(row["time/s"])
            for col in storecols:
                try:
                    data_in[col].append(float(row[col]))
                except ValueError:
                    data_in[col].append(float(row[col].replace("-", "e-")))

        for col in storecols:
            particledata[col] = np.array(np.interp(arr_time_s, arr_time_s_source, data_in[col]))

    if arr_strnuc_z_n:
        arr_traj_time_s = []
        arr_massfracs: dict[str, list[float]] = {strnuc: [] for strnuc, _, _ in arr_strnuc_z_n}
        for nts in range(nts_min, nts_max + 1):
            timesec = nstep_timesec[nts]
            arr_traj_time_s.append(timesec)
            # print(nts, timesec / 86400)
            traj_nuc_abund = at.inputmodel.rprocess_from_trajectory.get_trajectory_abund_q(
                particleid, traj_root=traj_root, nts=nts
            )
            for strnuc, Z, N in arr_strnuc_z_n:
                arr_massfracs[strnuc].append(traj_nuc_abund.get((Z, N), 0.0))

        for strnuc, _, _ in arr_strnuc_z_n:
            massfracs_interp = np.interp(arr_time_s, arr_traj_time_s, arr_massfracs[strnuc])
            particledata[strnuc] = np.array(massfracs_interp)

    return particleid, particledata


def plot_qdot_abund_modelcells(
    modelpath: Path,
    merger_root: Path,
    mgiplotlist: Sequence[int],
    arr_el_a: list[tuple[str, int]],
    xmax: float | None = None,
) -> None:
    # default values, because early model.txt didn't specify this
    griddatafolder: Path = Path("SFHo_snapshot")
    mergermodelfolder: Path = Path("SFHo_short")
    trajfolder: Path = Path("SFHo")
    with at.zopen(modelpath / "model.txt") as fmodel:
        while True:
            line = fmodel.readline()
            if not line.startswith("#"):
                break
            if line.startswith("# gridfolder:"):
                griddatafolder = Path(line.strip().removeprefix("# gridfolder: "))
                mergermodelfolder = Path(line.strip().removeprefix("# gridfolder: ").removesuffix("_snapshot"))
            elif line.startswith("# trajfolder:"):
                trajfolder = Path(line.strip().removeprefix("# trajfolder: ").replace("SFHO", "SFHo"))

    griddata_root = Path(merger_root, mergermodelfolder, griddatafolder)
    traj_root = Path(merger_root, mergermodelfolder, trajfolder)
    print(f"model.txt traj_root: {traj_root}")
    print(f"model.txt griddata_root: {griddata_root}")
    assert traj_root.is_dir()

    arr_el, arr_a = zip(*arr_el_a, strict=False)
    arr_strnuc: list[str] = [el + str(a) for el, a in arr_el_a]
    arr_z = [at.get_atomic_number(el) for el in arr_el]
    arr_n = [a - z for z, a in zip(arr_z, arr_a, strict=False)]
    arr_strnuc_z_n = list(zip(arr_strnuc, arr_z, arr_n, strict=True))

    # arr_z = [at.get_atomic_number(el) for el in arr_el]

    lzdfmodel, modelmeta = at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g", "rho", "logrho", "volume"])
    npts_model = modelmeta["npts_model"]

    # these factors correct for missing mass due to skipped shells, and volume error due to Cartesian grid map
    correction_factors = {}
    assoc_cells: dict[int, list[int]] = {}
    mgi_of_propcells: dict[int, int] = {}
    try:
        assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath)
        direct_model_propgrid_map = all(
            len(propcells) == 1 and mgi == propcells[0] for mgi, propcells in assoc_cells.items()
        )
        if direct_model_propgrid_map:
            print("  detected direct mapping of model cells to propagation grid")
    except FileNotFoundError:
        print("No grid mapping file found, assuming direct mapping of model cells to propagation grid")
        direct_model_propgrid_map = True

    if direct_model_propgrid_map:
        correction_factors = dict.fromkeys(arr_strnuc, 1.0)

        lzdfmodel = lzdfmodel.with_columns(n_assoc_cells=pl.lit(1.0))
    else:
        ncoordgridx = math.ceil(np.cbrt(max(mgi_of_propcells.keys())))
        propcellcount = int(math.ceil(ncoordgridx ** (1 / 3.0)) ** 3)
        assert propcellcount**3 == ncoordgridx
        xmax_tmodel = modelmeta["vmax_cmps"] * modelmeta["t_model_init_days"] * 86400
        wid_init = at.get_wid_init_at_tmodel(modelpath, propcellcount, modelmeta["t_model_init_days"], xmax_tmodel)

        lzdfmodel = lzdfmodel.with_columns(
            n_assoc_cells=pl.Series([
                len(assoc_cells.get(inputcellid - 1, []))
                for (inputcellid,) in lzdfmodel.select("inputcellid").collect().iter_rows()
            ])
        )

        # for spherical models, ARTIS mapping to a cubic grid introduces some errors in the cell volumes
        lzdfmodel = lzdfmodel.with_columns(mass_g_mapped=10 ** pl.col("logrho") * wid_init**3 * pl.col("n_assoc_cells"))
        for strnuc in arr_strnuc:
            corr = (
                lzdfmodel.select(pl.col(f"X_{strnuc}") * pl.col("mass_g_mapped")).sum().collect().item()
                / lzdfmodel.select(pl.col(f"X_{strnuc}") * pl.col("mass_g")).sum().collect().item()
            )
            # print(strnuc, corr)
            correction_factors[strnuc] = corr

    tmids = at.get_timestep_times(modelpath, loc="mid")
    MH = 1.67352e-24  # g

    arr_time_artis_days: list[float] = []
    arr_abund_artis: dict[int, dict[str, list[float]]] = {}

    with contextlib.suppress(FileNotFoundError):
        get_mgi_list = tuple(mgiplotlist)  # all cells if Ye is calculated
        estimators_lazy = at.estimators.scan_estimators(modelpath=modelpath, modelgridindex=get_mgi_list)
        assert estimators_lazy is not None
        estimators_lazy = estimators_lazy.filter(pl.col("timestep") > 0)

        first_mgi = None
        estimators_lazy = estimators_lazy.filter(pl.col("modelgridindex").is_in(mgiplotlist))

        estimators_lazy = estimators_lazy.select(
            "modelgridindex",
            "timestep",
            *[f"nniso_{strnuc}" for strnuc in arr_strnuc if f"nniso_{strnuc}" in estimators_lazy.columns],
        )

        estimators_lazy = (
            estimators_lazy.join(
                lzdfmodel.select(
                    "modelgridindex",
                    "rho",
                    *[f"X_{strnuc}" for strnuc in arr_strnuc if f"X_{strnuc}" in lzdfmodel.columns],
                ),
                on="modelgridindex",
            )
            .collect()
            .lazy()
        )

        estimators_lazy = estimators_lazy.join(
            pl.DataFrame({"timestep": range(len(tmids)), "time_mid": tmids})
            .with_columns(pl.col("timestep").cast(pl.Int32))
            .lazy(),
            on="timestep",
            how="left",
        )

        estimators_lazy = estimators_lazy.with_columns(
            rho_init=pl.col("rho"), rho=pl.col("rho") * (modelmeta["t_model_init_days"] / pl.col("time_mid")) ** 3
        )
        # assert False

        # estimators_lazy = estimators_lazy.with_columns(
        #     rho=pl.col("rho") * (modelmeta["t_model_init_days"] / pl.col("time_mid")) ** 3
        # )

        estimators_lazy = estimators_lazy.sort(by=["timestep", "modelgridindex"])
        estimators = estimators_lazy.collect()

        for (nts, mgi), estimtsmgsi in estimators.group_by(["timestep", "modelgridindex"], maintain_order=True):
            assert isinstance(nts, int)
            assert isinstance(mgi, int)

            if first_mgi is None:
                first_mgi = mgi
            time_days = estimtsmgsi["time_mid"].item()

            if mgi == first_mgi:
                arr_time_artis_days.append(time_days)

            for strnuc, a in zip(arr_strnuc, arr_a, strict=False):
                abund = estimtsmgsi[f"nniso_{strnuc}"].item()
                massfrac = abund * a * MH / estimtsmgsi["rho"].item()
                massfrac += estimtsmgsi[f"X_{strnuc}"].item() * (correction_factors[strnuc] - 1.0)

                if mgi not in arr_abund_artis:
                    arr_abund_artis[mgi] = {}

                if strnuc not in arr_abund_artis[mgi]:
                    arr_abund_artis[mgi][strnuc] = []

                arr_abund_artis[mgi][strnuc].append(massfrac)

    arr_time_artis_days_alltimesteps = at.get_timestep_times(modelpath)
    arr_time_artis_s_alltimesteps = np.array([t * 8.640000e04 for t in arr_time_artis_days_alltimesteps])
    # no completed timesteps yet, so display full set of timesteps that artis will compute
    if not arr_time_artis_days:
        arr_time_artis_days = arr_time_artis_days_alltimesteps.copy()

    arr_time_gsi_s = np.array([modelmeta["t_model_init_days"] * 86400, *arr_time_artis_s_alltimesteps], dtype=float)

    # times in artis are relative to merger, but NSM simulation time started earlier
    mergertime_geomunits = at.inputmodel.modelfromhydro.get_merger_time_geomunits(griddata_root)
    t_mergertime_s = mergertime_geomunits * 4.926e-6
    arr_time_gsi_s_incpremerger = np.array([
        modelmeta["t_model_init_days"] * 86400 + t_mergertime_s,
        *arr_time_artis_s_alltimesteps,
    ])
    arr_time_gsi_days = [float(x) / 86400.0 for x in arr_time_gsi_s]

    dfpartcontrib = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(modelpath).filter(
        (pl.col("cellindex") <= npts_model) & (pl.col("frac_of_cellmass") > 0)
    )

    mgiplotlistplus1 = [mgi + 1 for mgi in mgiplotlist]
    list_particleids_getabund = dfpartcontrib.filter(pl.col("cellindex").is_in(mgiplotlistplus1))["particleid"].unique()
    fworkerwithabund = partial(get_particledata, arr_time_gsi_s_incpremerger, arr_strnuc_z_n, traj_root, verbose=True)

    print(f"Reading trajectories from {traj_root}")
    print(f"Reading Qdot/thermo and abundance data for {len(list_particleids_getabund)} particles")

    if at.get_config()["num_processes"] > 1:
        with at.get_multiprocessing_pool() as pool:
            list_particledata_withabund = pool.map(fworkerwithabund, list_particleids_getabund)
            pool.close()
            pool.join()
    else:
        list_particledata_withabund = [fworkerwithabund(particleid) for particleid in list_particleids_getabund]

    list_particleids_noabund = [
        pid for pid in dfpartcontrib["particleid"].unique() if pid not in list_particleids_getabund
    ]
    fworkernoabund = partial(get_particledata, arr_time_gsi_s_incpremerger, [], traj_root)
    print(f"Reading for Qdot/thermo data (no abundances needed) for {len(list_particleids_noabund)} particles")

    if at.get_config()["num_processes"] > 1:
        with at.get_multiprocessing_pool() as pool:
            list_particledata_noabund = pool.map(fworkernoabund, list_particleids_noabund)
            pool.close()
            pool.join()
    else:
        list_particledata_noabund = [fworkernoabund(particleid) for particleid in list_particleids_noabund]

    allparticledata = dict(list_particledata_withabund + list_particledata_noabund)

    plot_qdot(
        modelpath,
        dfpartcontrib,
        lzdfmodel,
        modelmeta,
        allparticledata,
        arr_time_artis_days,
        arr_time_gsi_days,
        pdfoutpath=Path(modelpath, "gsinetwork_global-qdot.pdf"),
        xmax=xmax,
    )

    for mgi in mgiplotlist:
        plot_cell_abund_evolution(
            modelpath,
            dfpartcontrib,
            allparticledata,
            arr_time_artis_days,
            arr_time_gsi_days,
            arr_strnuc,
            arr_abund_artis.get(mgi, {}),
            modelmeta["t_model_init_days"],
            lzdfmodel.select(modelgridindex=mgi).collect(),
            mgi=mgi,
            pdfoutpath=Path(modelpath, f"gsinetwork_cell{mgi}-abundance.pdf"),
            xmax=xmax,
        )


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", default=".", help="Path for ARTIS files")

    parser.add_argument(
        "-mergerroot",
        default=Path(Path.home() / "Google Drive/Shared Drives/GSI NSM/Mergers"),
        help="Base path for merger snapshot and trajectory data specified in model.txt",
    )

    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")

    parser.add_argument("-xmax", default=None, type=int, help="Maximum time in days to plot")

    parser.add_argument(
        "-modelgridindex",
        "-cell",
        "-mgi",
        default=None,
        help="Modelgridindex (zero-indexed) to plot or list such as 4,5,6",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Compare the energy release and abundances from ARTIS to the GSI Network calculation."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    arr_el_a = [
        ("He", 4),
        # ("Ga", 72),
        ("Sr", 89),
        ("Sr", 91),
        ("Sr", 92),
        ("Y", 92),
        ("Y", 93),
        ("Zr", 93),
        ("Ba", 140),
        ("Ce", 141),
        ("Nd", 147),
        # ('Rn', 222),
        # ("Ra", 223),
        # ("Ra", 224),
        # ("Ra", 225),
        # ("Ac", 225),
        # ('Th', 234),
        # ('Pa', 233),
        # ('U', 235),
    ]

    # arr_el_a = [
    #     ("He", 4),
    #     ("Ga", 72),
    #     ("Sr", 91),
    #     ("Sr", 92),
    # ]

    # arr_el_a = [
    #     ("Y", 92),
    #     ("Zr", 93),
    #     ("Ce", 141),
    #     ("Nd", 147),
    # ]

    arr_el_a.sort(key=lambda x: (at.get_atomic_number(x[0]), x[1]))

    modelpath = Path(args.modelpath)
    if args.modelgridindex is None:
        mgiplotlist = []
    elif hasattr(args.modelgridindex, "split"):
        mgiplotlist = [int(mgi) for mgi in args.modelgridindex.split(",")]
    else:
        mgiplotlist = [int(args.modelgridindex)]

    plot_qdot_abund_modelcells(
        modelpath=modelpath, merger_root=args.mergerroot, mgiplotlist=mgiplotlist, arr_el_a=arr_el_a, xmax=args.xmax
    )


if __name__ == "__main__":
    main()
