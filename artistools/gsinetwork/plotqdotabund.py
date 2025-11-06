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
from polars import selectors as cs

import artistools as at
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path


def strnuc_to_latex(strnuc: str) -> str:
    """Convert a string like sr89 to $^{89}$Sr."""
    elsym = strnuc.rstrip(string.digits)
    massnum = strnuc.removeprefix(elsym)

    return rf"$^{{{massnum}}}${elsym.title()}"


def plot_qdot(
    modelpath: Path,
    dfcontribsparticledata: pl.LazyFrame,
    arr_time_artis_days: Sequence[float],  # noqa: ARG001
    arr_time_gsi_days: Sequence[float],
    pdfoutpath: Path | str,
    xmax: float | None = None,
) -> None:
    try:
        depdata = at.get_deposition(modelpath=modelpath).collect()

    except FileNotFoundError:
        print("Can't do qdot plot because no deposition.out file")
        return

    heatcols = ["hbeta", "halpha", "hspof"]

    print("Calculating global heating rates from the individual particle heating rates")
    dfgsiglobalheating = dfcontribsparticledata.select([
        pl.concat_arr(
            (pl.col(col) * pl.col("cellmass_on_mtot") * pl.col("frac_of_cellmass")).arr.get(n).sum()
            for n in range(len(arr_time_gsi_days))
        )
        .explode()
        .alias(col)
        for col in heatcols
    ]).collect()

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
    axis.set_ylabel("Qdot [erg/s/g]")

    axis.plot(
        arr_time_gsi_days,
        dfgsiglobalheating["hbeta"],
        linewidth=2,
        color="black",
        linestyle="solid",
        # marker='x', markersize=8,
        label=r"$\dot{Q}_\beta$ GSINET",
    )

    axis.plot(
        depdata["tmid_days"],
        depdata["Qdot_betaminus_ana_erg/s/g"],
        linewidth=2,
        color="red",
        linestyle="solid",
        # marker='+', markersize=15,
        label=r"$\dot{Q}_\beta$ ARTIS",
    )

    axis.plot(
        arr_time_gsi_days,
        dfgsiglobalheating["halpha"],
        linewidth=2,
        color="black",
        linestyle="dashed",
        # marker='x', markersize=8,
        label=r"$\dot{Q}_\alpha$ GSINET",
    )

    axis.plot(
        depdata["tmid_days"],
        depdata["Qdotalpha_ana_erg/s/g"],
        linewidth=2,
        color="red",
        linestyle="dashed",
        # marker='+', markersize=15,
        label=r"$\dot{Q}_\alpha$ ARTIS",
    )

    axis.plot(
        arr_time_gsi_days,
        dfgsiglobalheating["hspof"],
        linewidth=2,
        color="black",
        linestyle="dotted",
        # marker='x', markersize=8,
        label=r"$\dot{Q}_{sponfis}$ GSINET",
    )

    if "Qdotspfission_ana_erg/s/g" in depdata.columns:
        axis.plot(
            depdata["tmid_days"],
            depdata["Qdotspfission_ana_erg/s/g"],
            linewidth=2,
            color="red",
            linestyle="dotted",
            # marker='+', markersize=15,
            label=r"$\dot{Q}_{sponfis}$ ARTIS",
        )

    axis.legend(loc="best", frameon=False, handlelength=1, ncol=3, numpoints=1)

    # fig.suptitle(f'{at.get_model_name(modelpath)}', fontsize=10)
    at.plottools.autoscale(axis, margin=0.0)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"open {pdfoutpath}\n")


def plot_cell_abund_evolution(
    modelpath: Path,  # noqa: ARG001
    dfcontribsparticledata: pl.LazyFrame,
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
    print(f"Calculating abundances in model cell {mgi} from the individual particle abundances")
    dfpartcontrib_thiscell = dfcontribsparticledata.filter(pl.col("modelgridindex") == mgi)
    frac_of_cellmass_sum = dfpartcontrib_thiscell.select(pl.col("frac_of_cellmass").sum()).collect().item()
    print(f"frac_of_cellmass_sum: {frac_of_cellmass_sum} (can be < 1.0 because of missing particles)")

    # we didn't include all cells (maybe), so we need a normalization factor here
    normfactor = (
        dfpartcontrib_thiscell.group_by("modelgridindex")
        .agg(pl.col("cellmass_on_mtot").first())
        .drop("modelgridindex")
        .sum()
        .collect()
        .item()
    )

    df_gsi_abunds = dfpartcontrib_thiscell.select([
        pl.concat_arr(
            (pl.col(strnuc) * pl.col("frac_of_cellmass") * pl.col("cellmass_on_mtot") / normfactor).arr.get(n).sum()
            for n in range(len(arr_time_gsi_days))
        )
        .explode()
        .alias(strnuc)
        for strnuc in arr_strnuc
    ]).collect()

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
    print(f"{'nuc':7s}  gsi_abund artis_abund")
    for axis, strnuc in zip(axes, arr_strnuc, strict=False):
        # print(arr_time_artis_days)
        xmin = min(arr_time_gsi_days) * 0.9
        xmax = xmax or max(arr_time_gsi_days) * 1.03
        axis.set_xlim(left=xmin, right=xmax)
        # axis.set_yscale('log')
        # axis.set_ylabel(f'X({strnuc})')
        axis.set_ylabel("Mass fraction")

        strnuc_latex = strnuc_to_latex(strnuc)

        axis.plot(
            arr_time_gsi_days,
            df_gsi_abunds[strnuc],
            linewidth=2,
            marker="x",
            markersize=8,
            label=f"{strnuc_latex} GSINET",
            color="black",
        )

        print(f"{strnuc:7s}  {df_gsi_abunds[strnuc][1]:.2e}", end="")
        if strnuc in arr_abund_artis:
            print(f" {arr_abund_artis[strnuc][0]:.2e}")
            axis.plot(
                arr_time_artis_days, arr_abund_artis[strnuc], linewidth=2, label=f"{strnuc_latex} ARTIS", color="red"
            )
        else:
            print(" [no ARTIS data]")

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

        at.plottools.autoscale(ax=axis, margin=0.25)

    # fig.suptitle(f"{at.get_model_name(modelpath)} cell {mgi}", y=0.995, fontsize=10)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"open {pdfoutpath}")


def get_particledata(
    arr_time_s: Sequence[float] | npt.NDArray[np.floating[t.Any]],
    arr_strnuc_z_n: list[tuple[str, int, int | None]],
    traj_root: Path,
    particleid: int,
    verbose: bool = False,
) -> pl.LazyFrame:
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
        return pl.LazyFrame()

    if verbose:
        print(
            "Reading network calculation heating.dat,"
            f" energy_thermo.dat{', and nz-plane abundances' if arr_strnuc_z_n else ''} for particle {particleid}..."
        )

    particledata = pl.LazyFrame({"particleid": [particleid]}, schema={"particleid": pl.Int32})
    nstep_timesec = {}
    with get_tar_member_extracted_path(
        traj_root=traj_root, particleid=particleid, memberfilename="./Run_rprocess/heating.dat"
    ).open(encoding="utf-8") as f:
        dfheating = pd.read_csv(f, sep=r"\s+", usecols=["#count", "time/s", "hbeta", "halpha", "hspof"])
        heatcols = ["hbeta", "halpha", "hspof"]

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
            particledata = particledata.with_columns(
                pl.Series(
                    [np.interp(arr_time_s, arr_time_s_source, heatrates_in[col])],
                    dtype=pl.Array(pl.Float32, len(arr_time_s)),
                ).alias(col)
            )

    if arr_strnuc_z_n:
        nts_list = list(range(nts_min, nts_max + 1))
        nts_count = len(nts_list)
        arr_traj_time_s = [nstep_timesec[nts] for nts in nts_list]
        arr_massfracs = {strnuc: np.zeros(nts_count, dtype=np.float32) for strnuc, _, _ in arr_strnuc_z_n}
        for i, nts in enumerate(nts_list):
            dftrajnucabund, _traj_time_s = at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfile_nuc_abund(
                traj_root, particleid, f"./Run_rprocess/nz-plane{nts:05d}"
            )
            for strnuc, Z, N in arr_strnuc_z_n:
                if N is None:
                    # sum over all isotopes of this element
                    arr_massfracs[strnuc][i] = (
                        dftrajnucabund.filter(pl.col("Z") == Z).select(pl.col("massfrac").sum()).item()
                    )
                else:
                    arr_massfracs[strnuc][i] = (
                        dftrajnucabund.filter((pl.col("Z") == Z) & (pl.col("N") == N))
                        .select(pl.col("massfrac").sum())
                        .item()
                    )

        particledata = particledata.with_columns(
            pl.Series(
                [np.interp(arr_time_s, arr_traj_time_s, arr_massfracs[strnuc])],
                dtype=pl.Array(pl.Float32, len(arr_time_s)),
            ).alias(strnuc)
            for strnuc, _, _ in arr_strnuc_z_n
        )

    return particledata


def plot_qdot_abund_modelcells(
    modelpath: Path,
    merger_root: Path,
    mgiplotlist: Sequence[int],
    arr_el_a: list[tuple[str, int | None]],
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

    arr_el, arr_a = zip(*arr_el_a, strict=True)
    arr_strnuc: list[str] = [f"{el}{a}" if a is not None else el for el, a in arr_el_a]
    arr_z = [at.get_atomic_number(el) for el in arr_el]
    arr_n = [a - z if a is not None else None for z, a in zip(arr_z, arr_a, strict=True)]
    arr_strnuc_z_n = list(zip(arr_strnuc, arr_z, arr_n, strict=True))

    # arr_z = [at.get_atomic_number(el) for el in arr_el]

    lzdfmodel, modelmeta = at.inputmodel.get_modeldata(
        modelpath, derived_cols=["mass_g", "rho", "logrho", "volume"], get_elemabundances=True
    )
    lzdfmodel = lzdfmodel.with_columns(cellmass_on_mtot=pl.col("mass_g") / pl.col("mass_g").sum())

    model_mass_grams = lzdfmodel.select(pl.col("mass_g").sum()).collect().item()
    print(f"model mass: {model_mass_grams / 1.989e33:.3f} Msun")

    npts_model = modelmeta["npts_model"]

    # these factors correct for missing mass due to skipped shells, and volume error due to Cartesian grid map
    # it is important to follow the same method as artis to get the correct mass fractions
    correction_factors: dict[str, float] = {}
    assoc_cells: dict[int, list[int]] = {}
    mgi_of_propcells: dict[int, int] = {}
    try:
        assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath)
        for mgi in mgiplotlist:
            assert assoc_cells.get(mgi, []), (
                f"No propagation grid cells associated with model cell {mgi}, cannot plot abundances!"
            )
        direct_model_propgrid_map = all(
            len(propcells) == 1 and mgi == propcells[0] for mgi, propcells in assoc_cells.items()
        )
        if direct_model_propgrid_map:
            print("  detected direct mapping of model cells to propagation grid")
    except FileNotFoundError:
        print("No grid mapping file found, assuming direct mapping of model cells to propagation grid")
        direct_model_propgrid_map = True

    if direct_model_propgrid_map:
        lzdfmodel = lzdfmodel.with_columns(n_assoc_cells=pl.lit(1.0))
    else:
        ncoordgridx = math.ceil(np.cbrt(max(mgi_of_propcells.keys()) + 1))
        propcellcount = ncoordgridx**3
        print(f" inferring {propcellcount} propagation grid cells from grid mapping file")
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
            # could be a nuclide like "Sr89" or an element like "Sr"
            nucisocols = (
                [f"X_{strnuc}"]
                if strnuc[-1].isdigit()
                else [c for c in lzdfmodel.collect_schema().names() if c.startswith(f"X_{strnuc}")]
            )
            for nucisocol in nucisocols:
                if nucisocol not in lzdfmodel.collect_schema().names():
                    continue
                correction_factors[nucisocol.removeprefix("X_")] = (
                    lzdfmodel.select(
                        pl.col(nucisocol).dot(pl.col("mass_g_mapped")) / pl.col(nucisocol).dot(pl.col("mass_g"))
                    )
                    .collect()
                    .item()
                )

    MH = 1.67352e-24  # g

    arr_time_artis_days: list[float] = []
    arr_abund_artis: dict[int, dict[str, list[float]]] = {}

    with contextlib.suppress(FileNotFoundError):
        estimators_lazy = at.estimators.scan_estimators(
            modelpath=modelpath, modelgridindex=mgiplotlist, join_modeldata=True
        )

        estimators_lazy = estimators_lazy.filter(pl.col("modelgridindex").is_in(mgiplotlist))

        estimators_lazy = estimators_lazy.select(
            "modelgridindex",
            "timestep",
            "tmid_days",
            cs.starts_with(*[f"nniso_{strnuc}" for strnuc in arr_strnuc]),
            "rho",
            cs.starts_with(*[f"init_X_{strnuc}" for strnuc in arr_strnuc]),
        )

        estimators_lazy = estimators_lazy.sort(by=["timestep", "modelgridindex"])
        arr_time_artis_days = estimators_lazy.select(pl.col("tmid_days").unique()).collect().to_series().to_list()

        for mgi in mgiplotlist:
            assert isinstance(mgi, int)
            estim_thismgi = estimators_lazy.filter(pl.col("modelgridindex") == mgi).collect()

            for strnuc, a in zip(arr_strnuc, arr_a, strict=True):
                massfracs = np.zeros(estim_thismgi.height, dtype=float)
                if a is None:
                    for col in estim_thismgi.collect_schema().names():
                        if col.startswith(f"nniso_{strnuc}") and col.removeprefix(f"nniso_{strnuc}").isdigit():
                            a_iso = int(col.removeprefix(f"nniso_{strnuc}"))
                            offset = 0.0
                            if f"init_X_{strnuc}{a_iso}" in estim_thismgi.columns:
                                initmassfrac = pl.col(f"init_X_{strnuc}{a_iso}").first()
                                offset = initmassfrac * (correction_factors.get(f"{strnuc}{a_iso}", 1.0) - 1.0)
                            massfracs += (
                                estim_thismgi.select(pl.col(col) * a_iso * MH / estim_thismgi["rho"] + offset)
                                .to_series()
                                .to_numpy()
                            )

                elif f"nniso_{strnuc}" in estim_thismgi.columns:
                    offset = 0.0
                    if f"init_X_{strnuc}" in estim_thismgi.columns:
                        initmassfrac = pl.col(f"init_X_{strnuc}").first()
                        offset = initmassfrac * (correction_factors.get(strnuc, 1.0) - 1.0)
                    massfracs = (
                        estim_thismgi.select(pl.col(f"nniso_{strnuc}") * a * MH / estim_thismgi["rho"] + offset)
                        .to_series()
                        .to_list()
                    )
                else:
                    continue

                if mgi not in arr_abund_artis:
                    arr_abund_artis[mgi] = {}

                arr_abund_artis[mgi][strnuc] = list(massfracs)

    arr_time_artis_days_alltimesteps = (
        at.misc.df_filter_minmax_bounded(at.get_timesteps(modelpath).select("tmid_days"), "tmid_days", None, xmax)
        .collect()
        .get_column("tmid_days")
        .to_numpy()
    )
    arr_time_artis_s_alltimesteps = 86400.0 * arr_time_artis_days_alltimesteps

    # no completed timesteps yet, so display full set of timesteps that artis will compute
    if not arr_time_artis_days:
        arr_time_artis_days = list(arr_time_artis_days_alltimesteps)

    arr_time_gsi_s = np.array([modelmeta["t_model_init_days"] * 86400, *arr_time_artis_s_alltimesteps], dtype=float)

    # times in artis are relative to merger, but NSM simulation time started earlier
    mergertime_geomunits = at.inputmodel.modelfromhydro.get_merger_time_geomunits(griddata_root)
    t_mergertime_s = mergertime_geomunits * 4.926e-6
    arr_time_gsi_s_incpremerger = np.array([
        modelmeta["t_model_init_days"] * 86400 + t_mergertime_s,
        *arr_time_artis_s_alltimesteps,
    ])
    arr_time_gsi_days = list(arr_time_gsi_s / 86400.0)

    dfpartcontrib = (
        at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(modelpath)
        .lazy()
        .with_columns(modelgridindex=pl.col("cellindex") - 1)
        .filter(pl.col("modelgridindex") < npts_model)
        .filter(pl.col("frac_of_cellmass") > 0)
    ).join(lzdfmodel.select(["modelgridindex", "cellmass_on_mtot"]), on="modelgridindex", how="left")

    allcontribparticleids = dfpartcontrib.select(pl.col("particleid").unique()).collect().to_series().to_list()
    list_particleids_getabund = (
        dfpartcontrib.filter(pl.col("modelgridindex").is_in(mgiplotlist))
        .select(pl.col("particleid").unique())
        .collect()
        .to_series()
        .to_list()
    )
    fworkerwithabund = partial(get_particledata, arr_time_gsi_s_incpremerger, arr_strnuc_z_n, traj_root, verbose=False)

    print(f"Reading trajectories from {traj_root}")
    print(f"Reading Qdot/thermo and abundance data for {len(list_particleids_getabund)} particles")

    if at.get_config()["num_processes"] > 1:
        with at.get_multiprocessing_pool() as pool:
            list_particledata_withabund = pool.map(fworkerwithabund, list_particleids_getabund)
            pool.close()
            pool.join()
    else:
        list_particledata_withabund = [fworkerwithabund(particleid) for particleid in list_particleids_getabund]

    list_particleids_noabund = [pid for pid in allcontribparticleids if pid not in list_particleids_getabund]
    fworkernoabund = partial(get_particledata, arr_time_gsi_s_incpremerger, [], traj_root)
    print(f"Reading for Qdot/thermo data (no abundances needed) for {len(list_particleids_noabund)} particles")

    if at.get_config()["num_processes"] > 1:
        with at.get_multiprocessing_pool() as pool:
            list_particledata_noabund = pool.map(fworkernoabund, list_particleids_noabund)
            pool.close()
            pool.join()
    else:
        list_particledata_noabund = [fworkernoabund(particleid) for particleid in list_particleids_noabund]

    allparticledata = pl.concat(list_particledata_withabund + list_particledata_noabund, how="diagonal")

    dfcontribsparticledata = dfpartcontrib.join(allparticledata, on="particleid", how="inner")

    plot_qdot(
        modelpath,
        dfcontribsparticledata,
        arr_time_artis_days,
        arr_time_gsi_days,
        pdfoutpath=Path(modelpath, "gsinetwork_global-qdot.pdf"),
        xmax=xmax,
    )

    for mgi in mgiplotlist:
        plot_cell_abund_evolution(
            modelpath,
            dfcontribsparticledata,
            arr_time_artis_days,
            arr_time_gsi_days,
            arr_strnuc,
            arr_abund_artis.get(mgi, {}),
            modelmeta["t_model_init_days"],
            lzdfmodel.filter(modelgridindex=mgi).collect(),
            mgi=mgi,
            pdfoutpath=Path(modelpath, f"gsinetwork_cell{mgi}-abundance.pdf"),
            xmax=xmax,
        )


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path for ARTIS files")

    parser.add_argument(
        "-mergerroot",
        type=Path,
        default=Path(Path.home() / "Google Drive/Shared Drives/GSI NSM/Mergers"),
        help="Base path for merger snapshot and trajectory data specified in model.txt",
    )

    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")

    parser.add_argument("-xmax", default=None, type=float, help="Maximum time in days to plot")

    parser.add_argument(
        "-modelgridindex",
        "-cell",
        "-mgi",
        type=int,
        dest="mgilist",
        nargs="*",
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
        # ("He", 4),
        # ("Ga", 72),
        ("Sr", None),
        ("Y", None),
        ("Zr", None),
        ("Gd", None),
        ("La", None),
        ("Pr", None),
        ("Ce", None),
        ("Sr", 89),
        ("Sr", 91),
        ("Sr", 92),
        ("Sr", 104),
        ("Y", 92),
        ("Cf", 254),
        ("Rb", 88),
        ("I", 129),
        ("I", 132),
        ("Sb", 128),
        ("Cu", 66),
    ]

    arr_el_a.sort(key=lambda x: (at.get_atomic_number(x[0]), x[1] if x[1] is not None else -1))

    plot_qdot_abund_modelcells(
        modelpath=Path(args.modelpath),
        merger_root=Path(args.mergerroot),
        mgiplotlist=args.mgilist,
        arr_el_a=arr_el_a,
        xmax=args.xmax,
    )


if __name__ == "__main__":
    main()
