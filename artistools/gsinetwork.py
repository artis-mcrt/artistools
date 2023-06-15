#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from __future__ import annotations

import argparse
import math
import multiprocessing
import typing as t
from functools import partial
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import artistools as at

if t.TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Sequence


def plot_qdot(
    modelpath: Path,
    dfpartcontrib: pd.DataFrame,
    dfmodel: pd.DataFrame,
    allparticledata: dict[int, dict[str, np.ndarray]],
    arr_time_artis_days: Collection[float],
    arr_artis_ye: Collection[float],
    arr_time_gsi_days: Collection[float],
    pdfoutpath: Path | str,
) -> None:
    try:
        depdata = at.get_deposition(modelpath=modelpath)

    except FileNotFoundError:
        print("Can't do qdot plot because no deposition.out file")
        return

    modelname = at.get_model_name(modelpath)

    tstart = depdata["tmid_days"].min()
    tend = depdata["tmid_days"].max()

    heatcols = ["hbeta", "halpha", "hbfis", "hspof", "Ye", "Qdot"]

    arr_heat = {col: np.zeros_like(arr_time_gsi_days) for col in heatcols}

    model_mass_grams = dfmodel.cellmass_grams.sum()
    print(f"model mass: {model_mass_grams / 1.989e33:.3f} Msun")

    print("Calculating global heating rates from the individual particle heating rates...")
    dfpartcontrib_nomissing = dfpartcontrib[dfpartcontrib["particleid"].isin(allparticledata.keys())]
    for cellindex, dfpartcontribthiscell in dfpartcontrib_nomissing.groupby("cellindex"):
        if cellindex >= len(dfmodel):
            continue

        cell_mass_frac = dfmodel.iloc[cellindex - 1].cellmass_grams / model_mass_grams

        if cell_mass_frac == 0.0:
            continue

        frac_of_cellmass_sum = dfpartcontribthiscell.frac_of_cellmass.sum()

        contribtuples = dfpartcontribthiscell[["particleid", "frac_of_cellmass"]].itertuples(index=False)

        for particleid, frac_of_cellmass in contribtuples:
            thisparticledata = allparticledata[particleid]
            for col in heatcols:
                arr_heat[col] += thisparticledata[col] * cell_mass_frac * frac_of_cellmass / frac_of_cellmass_sum

    print("  done.")

    show_ye = False
    nrows = 2 if show_ye else 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(6, 1 + 3 * nrows),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )
    if nrows == 1:
        axes = [axes]

    axis = axes[0]

    # axis.set_ylim(bottom=1e7, top=2e10)
    # axis.set_xlim(left=tstart, right=tend)
    xmin = min(arr_time_gsi_days) * 0.9
    xmax = max(arr_time_gsi_days) * 1.03
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
        label=r"$\dot{Q}_{hbfis}$ GSI Network",
    )

    axis.plot(
        arr_time_gsi_days,
        arr_heat["hspof"],
        linewidth=2,
        linestyle="dotted",
        # marker='x', markersize=8,
        # color='black',
        label=r"$\dot{Q}_{spof}$ GSI Network",
    )

    axis.legend(loc="best", frameon=False, handlelength=1, ncol=3, numpoints=1)

    if show_ye:
        axes[1].plot(
            arr_time_gsi_days,
            arr_heat["Ye"],
            linewidth=2,
            color="black",
            linestyle="dashed",
            # marker='x', markersize=8,
            label=r"Ye GSI Network",
        )

        axes[1].plot(
            arr_time_artis_days,
            arr_artis_ye,
            linewidth=2,
            # linestyle='None',
            # marker='+', markersize=15,
            label="Ye ARTIS",
            color="red",
        )

        axes[1].set_ylabel("Ye [e-/nucleon]")
        axes[1].legend(loc="best", frameon=False, handlelength=1, ncol=3, numpoints=1)

    # fig.suptitle(f'{modelname}', fontsize=10)
    at.plottools.autoscale(axis, margin=0.0)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"Saved {pdfoutpath}")


def plot_cell_abund_evolution(
    modelpath: Path,
    dfpartcontrib: pd.DataFrame,
    allparticledata,
    arr_time_artis_days: Sequence[float],
    arr_time_gsi_days: Sequence[float],
    arr_strnuc: Sequence[str],
    arr_abund_artis: dict[str, list[float]],
    t_model_init_days: float,
    dfcell: pd.DataFrame,
    pdfoutpath: Path,
    mgi: int,
) -> None:
    dfpartcontrib_thiscell = dfpartcontrib[
        (dfpartcontrib["cellindex"] == (mgi + 1)) & (dfpartcontrib["particleid"].isin(allparticledata.keys()))
    ]
    frac_of_cellmass_sum = dfpartcontrib_thiscell.frac_of_cellmass.sum()
    print(f"frac_of_cellmass_sum: {frac_of_cellmass_sum} (can be < 1.0 because of missing particles)")
    # if arr_strnuc[0] != 'Ye':
    #     arr_strnuc.insert(0, 'Ye')

    arr_abund_gsi: dict[str, np.ndarray[t.Any, np.dtype[np.float64]]] = {
        strnuc: np.zeros_like(arr_time_gsi_days) for strnuc in arr_strnuc
    }

    # calculate the GSI values from the particles contributing to this cell
    for particleid, frac_of_cellmass in dfpartcontrib_thiscell[["particleid", "frac_of_cellmass"]].itertuples(
        index=False
    ):
        frac_of_cellmass = dfpartcontrib_thiscell[dfpartcontrib_thiscell["particleid"] == particleid][
            "frac_of_cellmass"
        ].sum()

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

    modelname = at.get_model_name(modelpath)

    axes[-1].set_xlabel("Time [days]")
    axis = axes[0]
    print("nuc", "gsi_abund", "inputmodel_abund", "artis_abund")
    for axis, strnuc in zip(axes, arr_strnuc):
        # print(arr_time_artis_days)
        xmin = min(arr_time_gsi_days) * 0.9
        xmax = max(arr_time_gsi_days) * 1.03
        # xmax = 5  # TODO: remove
        axis.set_xlim(left=xmin, right=xmax)
        # axis.set_yscale('log')
        # axis.set_ylabel(f'X({strnuc})')
        if strnuc == "Ye":
            axis.set_ylabel("Electron fraction")
        else:
            axis.set_ylabel("Mass fraction")

        axis.plot(
            arr_time_gsi_days,
            arr_abund_gsi[strnuc],
            # linestyle='None',
            linewidth=2,
            marker="x",
            markersize=8,
            label=f"{strnuc} GSI Network",
            color="black",
        )

        if strnuc in arr_abund_artis:
            axis.plot(
                arr_time_artis_days,
                arr_abund_artis[strnuc],
                linewidth=2,
                # linestyle='None',
                # marker='+', markersize=15,
                label=f"{strnuc} ARTIS",
                color="red",
            )

        if f"X_{strnuc}" in dfcell:
            axis.plot(
                t_model_init_days,
                dfcell[f"X_{strnuc}"],
                marker="+",
                markersize=15,
                markeredgewidth=2,
                label=f"{strnuc} ARTIS inputmodel",
                color="blue",
            )
            print(
                f'{strnuc} {arr_abund_gsi[strnuc][0]:.2e} {dfcell[f"X_{strnuc}"]:.2e}, {arr_abund_artis[strnuc][0]:.2e}'
            )

        axis.legend(loc="best", frameon=False, handlelength=1, ncol=1, numpoints=1)

    # fig.suptitle(f"{modelname} cell {mgi}", y=0.995, fontsize=10)
    at.plottools.autoscale(axis, margin=0.05)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"Saved {pdfoutpath}")


def get_particledata(
    arr_time_s: Sequence[float],
    arr_strnuc: list[str],
    traj_root: Path,
    particleid: int,
    verbose: bool = False,
) -> tuple[int, dict[str, np.ndarray]]:
    """For an array of times (NSM time including time before merger), interpolate the heating rates of various decay channels
    and (if arr_strnuc is not empty) the nuclear mass fractions.
    """
    try:
        nts_min = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            traj_root, particleid, timesec=min(arr_time_s), cond="lessthan"
        )
        nts_max = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            traj_root, particleid, timesec=max(arr_time_s), cond="greaterthan"
        )

    except FileNotFoundError:
        print(f"No network calculation for particle {particleid}")
        # make sure we weren't requesting abundance data for this particle that has no network data
        if arr_strnuc:
            print("ERROR:", particleid, arr_strnuc)
        assert not arr_strnuc
        return -1, {}

    if verbose:
        print(
            "Reading network calculation heating.dat,"
            f" energy_thermo.dat{', and nz-plane abundances' if arr_strnuc else ''} for particle {particleid}..."
        )

    particledata = {}
    nstep_timesec = {}
    with at.inputmodel.rprocess_from_trajectory.open_tar_file_or_extracted(
        traj_root, particleid, "./Run_rprocess/heating.dat"
    ) as f:
        dfheating = pd.read_csv(
            f, delim_whitespace=True, usecols=["#count", "time/s", "hbeta", "halpha", "hbfis", "hspof"]
        )
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

    with at.inputmodel.rprocess_from_trajectory.open_tar_file_or_extracted(
        traj_root, particleid, "./Run_rprocess/energy_thermo.dat"
    ) as f:
        storecols = ["Qdot", "Ye"]

        dfthermo = pd.read_csv(f, delim_whitespace=True, usecols=["#count", "time/s", *storecols])

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

    if arr_strnuc:
        arr_traj_time_s = []
        arr_massfracs: dict[str, list[float]] = {strnuc: [] for strnuc in arr_strnuc}
        for nts in range(nts_min, nts_max + 1):
            timesec = nstep_timesec[nts]
            arr_traj_time_s.append(timesec)
            # print(nts, timesec / 86400)
            traj_nuc_abund = at.inputmodel.rprocess_from_trajectory.get_trajectory_abund_q(
                particleid, traj_root=traj_root, nts=nts
            )
            for strnuc in arr_strnuc:
                arr_massfracs[strnuc].append(traj_nuc_abund.get(f"X_{strnuc}", 0.0))

        for strnuc in arr_strnuc:
            massfracs_interp = np.interp(arr_time_s, arr_traj_time_s, arr_massfracs[strnuc])
            particledata[strnuc] = np.array(massfracs_interp)

    return particleid, particledata


def plot_qdot_abund_modelcells(modelpath: Path, mgiplotlist: Sequence[int], arr_el_a: list[tuple[str, int]]):
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

    merger_root = Path(Path.home() / "Google Drive/Shared Drives/GSI NSM/Mergers")
    # merger_root = Path("/Users/luke/Downloads/Mergers")
    griddata_root = Path(merger_root, mergermodelfolder, griddatafolder)
    traj_root = Path(merger_root, mergermodelfolder, trajfolder)
    print(f"model.txt traj_root: {traj_root}")
    print(f"model.txt griddata_root: {griddata_root}")
    assert traj_root.is_dir()

    arr_el, arr_a = zip(*arr_el_a)
    arr_strnuc: list[str] = [el + str(a) for el, a in arr_el_a]

    # arr_z = [at.get_atomic_number(el) for el in arr_el]

    dfmodel, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata_tuple(modelpath)
    if "logrho" not in dfmodel.columns:
        dfmodel = dfmodel.eval("logrho = log10(rho)")
    model_mass_grams = dfmodel.cellmass_grams.sum()
    npts_model = len(dfmodel)

    # these factors correct for missing mass due to skipped shells, and volume error due to Cartesian grid map
    correction_factors = {}
    assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath)
    # WARNING sketchy inference!
    propcellcount = math.ceil(max(mgi_of_propcells.keys()) ** (1 / 3.0)) ** 3
    xmax_tmodel = vmax_cmps * t_model_init_days * 86400
    wid_init = at.get_wid_init_at_tmodel(modelpath, propcellcount, t_model_init_days, xmax_tmodel)
    dfmodel["n_assoc_cells"] = [len(assoc_cells.get(inputcellid - 1, [])) for inputcellid in dfmodel["inputcellid"]]

    # for spherical models, ARTIS mapping to a cubic grid introduces some errors in the cell volumes
    dfmodel = dfmodel.eval("cellmass_grams_mapped = 10 ** logrho * @wid_init ** 3 * n_assoc_cells")
    for strnuc in arr_strnuc:
        corr = (
            dfmodel.eval(f"X_{strnuc} * cellmass_grams_mapped").sum()
            / dfmodel.eval(f"X_{strnuc} * cellmass_grams").sum()
        )
        # print(strnuc, corr)
        correction_factors[strnuc] = corr

    tmids = at.get_timestep_times_float(modelpath, loc="mid")
    MH = 1.67352e-24  # g

    arr_time_artis_days: list[float] = []
    arr_abund_artis: dict[int, dict[str, list[float]]] = {}
    get_global_Ye = False
    artis_ye_sum: dict[int, float] = {}
    artis_ye_norm: dict[int, float] = {}

    try:
        get_mgi_list = None if get_global_Ye else tuple(mgiplotlist)  # all cells if Ye is calculated
        estimators = at.estimators.read_estimators(modelpath, modelgridindex=get_mgi_list)

        first_mgi = None
        partiallycomplete_timesteps = at.estimators.get_partiallycompletetimesteps(estimators)
        for nts, mgi in sorted(estimators.keys()):
            if nts in partiallycomplete_timesteps:
                continue
            if mgi not in mgiplotlist and not get_global_Ye or estimators[(nts, mgi)]["emptycell"]:
                continue

            if first_mgi is None:
                first_mgi = mgi
            time_days = tmids[nts]

            if mgi == first_mgi:
                arr_time_artis_days.append(time_days)

            rho_init_cgs = 10 ** dfmodel.iloc[mgi].logrho
            rho_cgs = rho_init_cgs * (t_model_init_days / time_days) ** 3

            for strnuc, a in zip(arr_strnuc, arr_a):
                abund = estimators[(nts, mgi)]["populations"][strnuc]
                massfrac = abund * a * MH / rho_cgs
                massfrac = massfrac + dfmodel.iloc[mgi][f"X_{strnuc}"] * (correction_factors[strnuc] - 1.0)

                if mgi not in arr_abund_artis:
                    arr_abund_artis[mgi] = {}

                if strnuc not in arr_abund_artis[mgi]:
                    arr_abund_artis[mgi][strnuc] = []

                arr_abund_artis[mgi][strnuc].append(massfrac)

            if mgi not in arr_abund_artis:
                arr_abund_artis[mgi] = {}

            if "Ye" not in arr_abund_artis[mgi]:
                arr_abund_artis[mgi]["Ye"] = []

            abund = estimators[(nts, mgi)]["populations"].get(strnuc, 0.0)
            if "Ye" in estimators[(nts, mgi)]:
                cell_Ye = estimators[(nts, mgi)]["Ye"]
                arr_abund_artis[mgi]["Ye"].append(cell_Ye)
                artis_ye_sum[nts] = artis_ye_sum.get(nts, 0.0) + cell_Ye * dfmodel.iloc[mgi].cellmass_grams
                artis_ye_norm[nts] = artis_ye_norm.get(nts, 0.0) + dfmodel.iloc[mgi].cellmass_grams
            else:
                cell_protoncount = 0.0
                cell_nucleoncount = 0.0
                cellvolume = dfmodel.iloc[mgi].cellmass_grams / rho_cgs
                for popkey, abund in estimators[(nts, mgi)]["populations"].items():
                    if isinstance(popkey, str) and abund > 0.0:
                        if popkey.endswith("_otherstable"):
                            # TODO: use mean molecular weight, but this is not needed for kilonova input files anyway
                            pass
                        else:
                            try:
                                z, a = at.get_z_a_nucname(popkey)
                                cell_protoncount += z * abund * cellvolume
                                cell_nucleoncount += a * abund * cellvolume

                            except AssertionError:
                                pass
                cell_Ye = cell_protoncount / cell_nucleoncount

                arr_abund_artis[mgi]["Ye"].append(cell_Ye)
                artis_ye_sum[nts] = artis_ye_sum.get(nts, 0.0) + cell_protoncount
                artis_ye_norm[nts] = artis_ye_norm.get(nts, 0.0) + cell_nucleoncount

        arr_artis_ye = [artis_ye_sum[nts] / artis_ye_norm[nts] for nts in sorted(artis_ye_sum.keys())]

    except FileNotFoundError:
        pass

    arr_time_artis_days_alltimesteps = at.get_timestep_times_float(modelpath)
    arr_time_artis_s_alltimesteps = np.array([t * 8.640000e04 for t in arr_time_artis_days_alltimesteps])
    # no completed timesteps yet, so display full set of timesteps that artis will compute
    if len(arr_time_artis_days) == 0:
        arr_time_artis_days = list(arr_time_artis_days_alltimesteps)

    arr_time_gsi_s = np.array([t_model_init_days * 86400, *arr_time_artis_s_alltimesteps])

    # times in artis are relative to merger, but NSM simulation time started earlier
    mergertime_geomunits = at.inputmodel.modelfromhydro.get_merger_time_geomunits(griddata_root)
    t_mergertime_s = mergertime_geomunits * 4.926e-6
    arr_time_gsi_s_incpremerger = np.array([t_model_init_days * 86400 + t_mergertime_s, *arr_time_artis_s_alltimesteps])
    arr_time_gsi_days = list(arr_time_gsi_s / 86400)

    dfpartcontrib = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(modelpath)
    dfpartcontrib = dfpartcontrib[(dfpartcontrib["cellindex"] <= npts_model) & (dfpartcontrib["frac_of_cellmass"] > 0)]

    mgiplotlistplus1 = (mgi + 1 for mgi in mgiplotlist)
    list_particleids_getabund = dfpartcontrib[dfpartcontrib["cellindex"].isin(mgiplotlistplus1)].particleid.unique()
    fworkerwithabund = partial(get_particledata, arr_time_gsi_s_incpremerger, arr_strnuc, traj_root, verbose=True)

    print(f"Reading trajectories from {traj_root}")
    print(f"Reading Qdot/thermo and abundance data for {len(list_particleids_getabund)} particles")

    if at.get_config()["num_processes"] > 1:
        with multiprocessing.get_context("fork").Pool(processes=at.get_config()["num_processes"]) as pool:
            list_particledata_withabund = pool.map(fworkerwithabund, list_particleids_getabund)
            pool.close()
            pool.join()
    else:
        list_particledata_withabund = [fworkerwithabund(particleid) for particleid in list_particleids_getabund]

    list_particleids_noabund = [
        pid for pid in dfpartcontrib.particleid.unique() if pid not in list_particleids_getabund
    ]
    fworkernoabund = partial(get_particledata, arr_time_gsi_s_incpremerger, [], traj_root)
    print(f"Reading for Qdot/thermo data (no abundances needed) for {len(list_particleids_noabund)} particles")

    if at.get_config()["num_processes"] > 1:
        with multiprocessing.get_context("fork").Pool(processes=at.get_config()["num_processes"]) as pool:
            list_particledata_noabund = pool.map(fworkernoabund, list_particleids_noabund)
            pool.close()
            pool.join()
    else:
        list_particledata_noabund = [fworkernoabund(particleid) for particleid in list_particleids_noabund]

    allparticledata = dict(list_particledata_withabund + list_particledata_noabund)

    plot_qdot(
        modelpath,
        dfpartcontrib,
        dfmodel,
        allparticledata,
        arr_time_artis_days,
        arr_artis_ye,
        arr_time_gsi_days,
        pdfoutpath=Path(modelpath, "gsinetwork_global-qdot.pdf"),
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
            t_model_init_days,
            dfmodel.iloc[mgi],
            mgi=mgi,
            pdfoutpath=Path(modelpath, f"gsinetwork_cell{mgi}-abundance.pdf"),
        )


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", default=".", help="Path for ARTIS files")

    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")

    parser.add_argument(
        "-modelgridindex",
        "-cell",
        "-mgi",
        default=None,
        help="Modelgridindex (zero-indexed) to plot or list such as 4,5,6",
    )


def main(args=None, argsraw=None, **kwargs):
    """Compare the energy release and abundances from ARTIS to the GSI Network calculation."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

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
    arr_el_a.sort(key=lambda x: (at.get_atomic_number(x[0]), x[1]))

    modelpath = Path(args.modelpath)
    if args.modelgridindex is None:
        mgiplotlist = []
    elif hasattr(args.modelgridindex, "split"):
        mgiplotlist = [int(mgi) for mgi in args.modelgridindex.split(",")]
    else:
        mgiplotlist = [int(args.modelgridindex)]

    plot_qdot_abund_modelcells(modelpath, mgiplotlist, arr_el_a)


if __name__ == "__main__":
    main()
