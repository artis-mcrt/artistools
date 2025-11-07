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


def get_abundance_correction_factors(
    lzdfmodel: pl.LazyFrame,
    mgiplotlist: Sequence[int],
    arr_strnuc: Sequence[str],
    modelpath: str | Path,
    modelmeta: dict[str, t.Any],
) -> dict[str, float]:
    """Get a dictionary of abundance multipliers that ARTIS will apply to correct for missing mass due to skipped shells, and volume error due to Cartesian grid mapping.

    It is important to follow the same method as artis to get the correct mass fractions.
    """
    correction_factors: dict[str, float] = {}
    assoc_cells: dict[int, list[int]] = {}
    mgi_of_propcells: dict[int, int] = {}
    try:
        assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath)
        for mgi in mgiplotlist:
            assert mgi < 0 or assoc_cells.get(mgi, []), (
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
    return correction_factors


def strnuc_to_latex(strnuc: str) -> str:
    """Convert a string like sr89 to $^{89}$Sr."""
    elsym = strnuc.rstrip(string.digits)
    massnum = strnuc.removeprefix(elsym)

    return rf"$^{{{massnum}}}${elsym.title()}"


def get_artis_abund_sequences(
    modelpath: str | Path,
    dftimesteps: pl.DataFrame,
    mgiplotlist: Sequence[int],
    arr_species: Sequence[str],
    arr_a: Sequence[int | None],
    correction_factors: dict[str, float],
) -> tuple[list[float], dict[int, dict[str, list[float]]]]:
    arr_time_artis_days: list[float] = []
    arr_abund_artis: dict[int, dict[str, list[float]]] = {}
    MH = 1.67352e-24  # g

    with contextlib.suppress(FileNotFoundError):
        estimators_lazy = at.estimators.scan_estimators(
            modelpath=modelpath,
            modelgridindex=None if any(mgi < 0 for mgi in mgiplotlist) else mgiplotlist,
            timestep=dftimesteps["timestep"].to_list(),
            join_modeldata=True,
            verbose=False,
        ).filter(pl.col("mass_g") > 0)

        if all(mgi >= 0 for mgi in mgiplotlist):
            estimators_lazy = estimators_lazy.filter(pl.col("modelgridindex").is_in(mgiplotlist))

        estimators_lazy = estimators_lazy.select(
            "modelgridindex",
            "timestep",
            "tmid_days",
            cs.starts_with(*[f"nniso_{strnuc}" for strnuc in arr_species]),
            "mass_g",
            "rho",
            cs.starts_with(*[f"init_X_{strnuc}" for strnuc in arr_species]),
        )

        estimators_lazy = estimators_lazy.sort(by=["timestep", "modelgridindex"])
        arr_time_artis_days = estimators_lazy.select(pl.col("tmid_days").unique()).collect().to_series().to_list()
        lazyresults = []
        for mgi in mgiplotlist:
            assert isinstance(mgi, int)
            estim_mgifiltered = estimators_lazy.filter((pl.col("modelgridindex") == mgi).or_(mgi < 0))
            mass_selected = (
                estim_mgifiltered.group_by("modelgridindex")
                .agg(pl.col("mass_g").first())
                .select("mass_g")
                .sum()
                .collect()
                .item()
            )
            estim_mgifiltered = estim_mgifiltered.with_columns(cellmass_on_mtot=pl.col("mass_g") / mass_selected)

            for strnuc, a in zip(arr_species, arr_a, strict=True):
                combinedlzdf = pl.LazyFrame()
                if a is None:
                    matched_cols = [
                        col
                        for col in estim_mgifiltered.collect_schema().names()
                        if col.startswith(f"nniso_{strnuc}") and col.removeprefix(f"nniso_{strnuc}").isdigit()
                    ]
                    list_a_iso = [int(col.removeprefix(f"nniso_{strnuc}")) for col in matched_cols]
                    list_str_nuc_iso = [f"{strnuc}{a_iso}" for a_iso in list_a_iso]
                elif f"nniso_{strnuc}" in estim_mgifiltered.collect_schema().names():
                    matched_cols = [f"nniso_{strnuc}"]
                    list_a_iso = [a]
                    list_str_nuc_iso = [strnuc]
                else:
                    continue

                for col, a_iso, strnuciso in zip(matched_cols, list_a_iso, list_str_nuc_iso, strict=True):
                    offset = 0.0
                    if f"init_X_{strnuciso}" in estim_mgifiltered.collect_schema().names():
                        initmassfrac = pl.col(f"init_X_{strnuciso}").first()
                        offset = initmassfrac * (correction_factors.get(f"{strnuciso}", 1.0) - 1.0)

                    combinedlzdf = pl.concat(
                        [
                            combinedlzdf,
                            (
                                estim_mgifiltered.group_by("modelgridindex", maintain_order=True)
                                .agg(
                                    ((pl.col(col) * a_iso * MH / pl.col("rho") + offset) * pl.col("cellmass_on_mtot"))
                                    .implode()
                                    .alias("cellmassfracs")
                                )
                                .select(
                                    pl.concat_arr([
                                        pl.col("cellmassfracs").list.get(n).sum()
                                        for n in range(len(arr_time_artis_days))
                                    ])
                                    .explode()
                                    .alias(f"{strnuciso}_massfracs")
                                )
                            ),
                        ],
                        how="horizontal",
                    )
                combinedlzdf = combinedlzdf.select(pl.sum_horizontal(cs.ends_with("_massfracs")).alias(strnuc))
                lazyresults.append((mgi, strnuc, combinedlzdf))

        print("Collecting ARTIS abundance sequences...")
        dfcollected = pl.collect_all([lzdf for _, _, lzdf in lazyresults])
        print("Finished collecting ARTIS abundance sequences.")
        for (mgi, strnuc, _lzdfs), df in zip(lazyresults, dfcollected, strict=True):
            if mgi not in arr_abund_artis:
                arr_abund_artis[mgi] = {}

            arr_abund_artis[mgi][strnuc] = df.to_series().to_list()
    return arr_time_artis_days, arr_abund_artis


def plot_qdot(
    modelpath: Path,
    dfcontribsparticledata: pl.LazyFrame | None,
    arr_time_gsi_days: Sequence[float] | None,
    pdfoutpath: Path | str,
    xmax: float | None = None,
) -> None:
    try:
        depdata = at.misc.df_filter_minmax_bounded(
            at.get_deposition(modelpath=modelpath), "tmid_days", None, xmax
        ).collect()

    except FileNotFoundError:
        print("Can't do qdot plot because no deposition.out file")
        return

    if dfcontribsparticledata is not None:
        heatcols = ["hbeta", "halpha", "hspof"]

        print("Calculating global heating rates from the individual particle heating rates")
        assert arr_time_gsi_days is not None
        dfgsiglobalheating = (
            dfcontribsparticledata.select([
                pl.concat_arr(
                    (pl.col(col).arr.get(n) * pl.col("cellmass_on_mtot") * pl.col("frac_of_cellmass")).sum()
                    for n in range(len(arr_time_gsi_days))
                )
                .explode()
                .alias(col)
                for col in heatcols
            ])
            .collect()
            .with_columns(time_days=pl.Series(arr_time_gsi_days))
        )
    else:
        dfgsiglobalheating = None

    fig, axis = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(6, 1 + 3),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )

    axis.set_xlabel("Time [days]")
    axis.set_yscale("log")
    axis.set_ylabel(r"$\dot{Q}$ [erg/s/g]")

    if dfgsiglobalheating is not None:
        assert arr_time_gsi_days is not None
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

    if dfgsiglobalheating is not None:
        axis.plot(
            dfgsiglobalheating["time_days"],
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

    if dfgsiglobalheating is not None:
        axis.plot(
            dfgsiglobalheating["time_days"],
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

    axis.legend(loc="best", frameon=False, handlelength=2, ncol=3, numpoints=1)

    # fig.suptitle(f'{at.get_model_name(modelpath)}', fontsize=10)
    axis.autoscale(enable=True, axis="both")
    axis.set_xmargin(0.02)
    axis.set_ymargin(0.02)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"open {pdfoutpath}")


def plot_cell_abund_evolution(
    modelpath: Path,
    dfcontribsparticledata: pl.LazyFrame | None,
    arr_time_artis_days: Sequence[float],
    arr_time_gsi_days: Sequence[float] | None,
    arr_species: Sequence[str],
    arr_abund_artis: dict[str, list[float]],
    t_model_init_days: float,
    dfcell: pl.DataFrame,
    pdfoutpath: Path,
    mgi: int,
    hideinputmodelpoints: bool = True,
) -> None:
    if dfcontribsparticledata is not None:
        print(f"Calculating abundances in model cell {mgi} from the individual particle abundances")
        dfpartcontrib_thiscell = (
            dfcontribsparticledata.filter(pl.col("modelgridindex") == mgi) if mgi >= 0 else dfcontribsparticledata
        )
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

        assert arr_time_gsi_days is not None
        df_gsi_abunds = dfpartcontrib_thiscell.select([
            pl.concat_arr(
                (pl.col(strnuc).arr.get(n) * pl.col("frac_of_cellmass") * pl.col("cellmass_on_mtot") / normfactor).sum()
                for n in range(len(arr_time_gsi_days))
            )
            .explode()
            .alias(strnuc)
            for strnuc in arr_species
        ]).collect()
    else:
        df_gsi_abunds = None

    fig, axes = plt.subplots(
        nrows=len(arr_species),
        ncols=1,
        sharex=False,
        sharey=False,
        figsize=(6, 1 + 2.0 * len(arr_species)),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )
    if len(arr_species) == 1:
        axes = np.array([axes])
    fig.subplots_adjust(top=0.8)
    assert isinstance(axes, np.ndarray)
    axes[-1].set_xlabel("Time [days]")
    axis = axes[0]
    print(f"{'nuc':7s}  gsi_abund artis_abund")

    for axis, strnuc in zip(axes, arr_species, strict=False):
        # axis.set_yscale('log')
        axis.set_ylabel("Mass fraction")

        strnuc_latex = strnuc_to_latex(strnuc)

        if df_gsi_abunds is not None:
            axis.plot(
                arr_time_gsi_days,
                df_gsi_abunds[strnuc],
                linewidth=2,
                marker="x",
                markersize=8,
                label=f"{strnuc_latex} GSINET",
                color="black",
            )

        print(f"{strnuc:7s}  ", end="")
        if df_gsi_abunds is not None:
            print(f"{df_gsi_abunds[strnuc][1]:.2e}", end="")
        else:
            print(" [no GSINET]", end="")

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

        axis.autoscale(enable=True, axis="both")
        axis.set_xmargin(0.02)
        axis.set_ymargin(0.05)

    fig.suptitle(f"{at.get_model_name(modelpath)} cell {mgi}", y=0.995, fontsize=10)
    fig.savefig(pdfoutpath, format="pdf")
    print(f"open {pdfoutpath}")


def get_particledata(
    arr_time_s_incpremerger: Sequence[float] | npt.NDArray[np.floating[t.Any]],
    arr_strnuc_z_n: list[tuple[str, int, int | None]],
    traj_root: Path,
    particleid: int,
    verbose: bool = False,
) -> pl.LazyFrame:
    """For an array of times (NSM time including time before merger), interpolate the heating rates of various decay channels and (if arr_strnuc is not empty) the nuclear mass fractions."""
    try:
        nts_min = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            traj_root, particleid, timesec=min(float(x) for x in arr_time_s_incpremerger), cond="lessthan"
        )
        nts_max = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            traj_root, particleid, timesec=max(float(x) for x in arr_time_s_incpremerger), cond="greaterthan"
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
                    [np.interp(arr_time_s_incpremerger, arr_time_s_source, heatrates_in[col])],
                    dtype=pl.Array(pl.Float32, len(arr_time_s_incpremerger)),
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
                [np.interp(arr_time_s_incpremerger, arr_traj_time_s, arr_massfracs[strnuc])],
                dtype=pl.Array(pl.Float32, len(arr_time_s_incpremerger)),
            ).alias(strnuc)
            for strnuc, _, _ in arr_strnuc_z_n
        )

    return particledata


def plot_qdot_abund_modelcells(
    modelpath: Path,
    merger_root: Path,
    mgiplotlist: Sequence[int],
    arr_species: list[str],
    timedaysmax: float | None = None,
    nogsinet: bool = False,
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
    gsinet_available = griddata_root.is_dir() and traj_root.is_dir() and not nogsinet
    if gsinet_available:
        print(f"model.txt traj_root: {traj_root}")
        print(f"model.txt griddata_root: {griddata_root}")
    else:
        if not griddata_root.is_dir():
            print(f"model.txt griddata_root {griddata_root} is not a directory!")
        if not traj_root.is_dir():
            print(f"model.txt traj_root {traj_root} is not a directory!")
        gsinet_available = False

    arr_species.sort(key=lambda x: (at.get_atomic_number(x), int(x.removeprefix(x.rstrip(string.digits)) or -1)))
    arr_z = [at.get_atomic_number(species) for species in arr_species]
    arr_a = [
        int(a) if a is not None else a
        for a in [species.removeprefix(species.rstrip(string.digits)) or None for species in arr_species]
    ]
    arr_n = [a - z if a is not None else None for z, a in zip(arr_z, arr_a, strict=True)]
    arr_strnuc_z_n = list(zip(arr_species, arr_z, arr_n, strict=True))

    lzdfmodel, modelmeta = at.inputmodel.get_modeldata(
        modelpath, derived_cols=["mass_g", "rho", "logrho", "volume"], get_elemabundances=True
    )
    lzdfmodel = lzdfmodel.with_columns(cellmass_on_mtot=pl.col("mass_g") / pl.col("mass_g").sum())

    model_mass_grams = lzdfmodel.select(pl.col("mass_g").sum()).collect().item()
    print(f"model mass: {model_mass_grams / 1.989e33:.3f} Msun")

    dftimesteps = at.misc.df_filter_minmax_bounded(
        at.get_timesteps(modelpath).select("timestep", "tmid_days"), "tmid_days", None, timedaysmax
    ).collect()

    arr_time_artis_s_alltimesteps = dftimesteps.select(pl.col("tmid_days") * 86400.0).to_series().to_numpy()
    arr_time_artis_days_alltimesteps = dftimesteps.select(pl.col("tmid_days")).to_series().to_numpy()

    if gsinet_available:
        # times in artis are relative to merger, but NSM simulation time started earlier
        mergertime_geomunits = at.inputmodel.modelfromhydro.get_merger_time_geomunits(griddata_root)
        t_mergertime_s = mergertime_geomunits * 4.926e-6
        arr_time_gsi_s_incpremerger = np.array([
            modelmeta["t_model_init_days"] * 86400.0 + t_mergertime_s,
            *arr_time_artis_s_alltimesteps,
        ])
        arr_time_gsi_days = [modelmeta["t_model_init_days"], *arr_time_artis_days_alltimesteps]

        dfpartcontrib = (
            at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(modelpath)
            .lazy()
            .with_columns(modelgridindex=pl.col("cellindex") - 1)
            .filter(pl.col("modelgridindex") < modelmeta["npts_model"])
            .filter(pl.col("frac_of_cellmass") > 0)
        ).join(lzdfmodel.select(["modelgridindex", "cellmass_on_mtot"]), on="modelgridindex", how="left")

        allcontribparticleids = dfpartcontrib.select(pl.col("particleid").unique()).collect().to_series().to_list()
        list_particleids_getabund = (
            dfpartcontrib.filter(pl.col("modelgridindex").is_in(mgiplotlist).or_(any(mgi < 0 for mgi in mgiplotlist)))
            .select(pl.col("particleid").unique())
            .collect()
            .to_series()
            .to_list()
        )
        fworkerwithabund = partial(
            get_particledata, arr_time_gsi_s_incpremerger, arr_strnuc_z_n, traj_root, verbose=False
        )

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
    else:
        dfcontribsparticledata = None
        arr_time_gsi_days = None

    plot_qdot(
        modelpath,
        dfcontribsparticledata,
        arr_time_gsi_days,
        pdfoutpath=Path(modelpath, "gsinetwork_global-qdot.pdf"),
        xmax=timedaysmax,
    )

    if mgiplotlist:
        correction_factors = get_abundance_correction_factors(lzdfmodel, mgiplotlist, arr_species, modelpath, modelmeta)
        arr_time_artis_days, arr_abund_artis = get_artis_abund_sequences(
            modelpath=modelpath,
            dftimesteps=dftimesteps,
            mgiplotlist=mgiplotlist,
            arr_species=arr_species,
            arr_a=arr_a,
            correction_factors=correction_factors,
        )

        for mgi in mgiplotlist:
            print()
            strmgi = f"mgi{mgi}" if mgi >= 0 else "global"
            plot_cell_abund_evolution(
                modelpath,
                dfcontribsparticledata,
                arr_time_artis_days,
                arr_time_gsi_days,
                arr_species,
                arr_abund_artis.get(mgi, {}),
                modelmeta["t_model_init_days"],
                lzdfmodel.filter(modelgridindex=mgi).collect(),
                mgi=mgi,
                pdfoutpath=Path(modelpath, f"gsinetwork_{strmgi}-abundance.pdf"),
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
        default=[],
        nargs="*",
        help="Modelgridindex (zero-indexed) to plot or list such as 4,5,6",
    )

    parser.add_argument(
        "--nogsinet", action="store_true", help="Do not attempt to read GSI Network data even if available"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Compare the energy release and abundances from ARTIS to the GSI Network calculation."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    arr_species = [
        "He4",
        # "Ga72",
        "Sr",
        # "Sr89",
        # "Sr91",
        # "I129",
        # "I132",
        # "Rb88",
        # "Y92",
        # "Sb128",
        # "Cu66",
        # "Cf254",
    ]

    plot_qdot_abund_modelcells(
        modelpath=Path(args.modelpath),
        merger_root=Path(args.mergerroot),
        mgiplotlist=args.mgilist,
        arr_species=arr_species,
        timedaysmax=args.xmax,
        nogsinet=args.nogsinet,
    )


if __name__ == "__main__":
    main()
