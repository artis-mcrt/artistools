# PYTHON_ARGCOMPLETE_OK
"""Compare ARTIS heating rates and abundances against GSI nuclear network trajectory calculations."""

import argparse
import contextlib
import math
import string
import typing as t
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from polars import selectors as cs

import artistools as at
from artistools.constants import day_to_s
from artistools.constants import MH_g
from artistools.constants import Msun_to_g
from artistools.inputmodel.rprocess_from_trajectory import fix_fortran_exponents
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend


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
        assoc_cells, mgi_of_propcells, direct_model_propgrid_map = at.get_grid_mapping(modelpath)
        for mgi in mgiplotlist:
            assert mgi < 0 or assoc_cells.get(mgi, []), (
                f"No propagation grid cells associated with model cell {mgi}, cannot plot abundances!"
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
        xmax_tmodel = modelmeta["vmax_cmps"] * modelmeta["t_model_init_days"] * day_to_s
        wid_init = at.get_wid_init_at_tmodel(modelpath, propcellcount, modelmeta["t_model_init_days"], xmax_tmodel)

        dfpropcellcounts = pl.LazyFrame(
            {
                "modelgridindex": list(assoc_cells.keys()),
                "n_assoc_cells": [len(cells) for cells in assoc_cells.values()],
            },
            schema={"modelgridindex": pl.Int32, "n_assoc_cells": pl.Int64},
        )
        lzdfmodel = (
            lzdfmodel
            .with_columns(pl.col("modelgridindex").cast(pl.Int32))
            .join(dfpropcellcounts, on="modelgridindex", how="left", maintain_order="left")
            .with_columns(pl.col("n_assoc_cells").fill_null(0))
        )

        # for spherical models, ARTIS mapping to a cubic grid introduces some errors in the cell volumes
        lzdfmodel = lzdfmodel.with_columns(mass_g_mapped=10 ** pl.col("logrho") * wid_init**3 * pl.col("n_assoc_cells"))
        modelcolumns = lzdfmodel.collect_schema().names()
        # an arr_strnuc entry is either a nuclide like "Sr89", naming one column, or an element like "Sr",
        # covering every isotope column of that element. Selecting out of modelcolumns keeps the list free of
        # duplicates when arr_strnuc holds both forms, and drops names the model does not carry
        nucisocols = [
            col
            for col in modelcolumns
            if any(
                col == f"X_{strnuc}" if strnuc[-1].isdigit() else col.startswith(f"X_{strnuc}") for strnuc in arr_strnuc
            )
        ]

        # one collect for every nuclide, rather than re-running the whole model scan once per isotope column
        if nucisocols:
            factors = (
                lzdfmodel
                .select(**{
                    nucisocol: pl.col(nucisocol).dot(pl.col("mass_g_mapped")) / pl.col(nucisocol).dot(pl.col("mass_g"))
                    for nucisocol in nucisocols
                })
                .collect()
                .row(0, named=True)
            )
            correction_factors |= {col.removeprefix("X_"): value for col, value in factors.items()}

    return correction_factors


def strnuc_to_latex(strnuc: str) -> str:
    """Convert a string like sr89 to $^{89}$Sr."""
    elsym = strnuc.rstrip(string.digits)
    massnum = strnuc.removeprefix(elsym)

    return rf"$^{{{massnum}}}${elsym.title()}" if massnum else elsym.title()


def get_artis_abund_sequences(
    modelpath: str | Path,
    dftimesteps: pl.DataFrame,
    mgiplotlist: Sequence[int],
    arr_species: Sequence[str],
    correction_factors: dict[str, float],
) -> dict[int, pl.DataFrame]:
    """Return the ARTIS abundance of each species against time, for each model cell in mgiplotlist."""
    arr_abund_artis: dict[int, pl.DataFrame] = {}

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
            cs.starts_with(*[f"nniso_{strspecies}" for strspecies in arr_species]),
            "mass_g",
            "rho",
            cs.starts_with(*[f"init_X_{strspecies}" for strspecies in arr_species]),
        )
        estimators_lazy = estimators_lazy.sort(by=["timestep", "modelgridindex"])
        allisotopes_in_df = [
            col.removeprefix("nniso_")
            for col in estimators_lazy.collect_schema().names()
            if col.startswith("nniso_") and col.removeprefix("nniso_").lstrip(string.ascii_letters).isdigit()
        ]
        estimators_lazy = estimators_lazy.with_columns(
            (pl.col(f"init_X_{striso}") * (correction_factors.get(striso, 1.0) - 1.0)).alias(f"offset_{striso}")
            for striso in allisotopes_in_df
        ).with_columns([
            (
                (pl.col(f"nniso_{striso}") * int(striso.lstrip(string.ascii_letters)) * MH_g / pl.col("rho"))
                + pl.col(f"offset_{striso}")
            ).alias(f"X_{striso}")
            for striso in allisotopes_in_df
        ])

        cellmassfrac_exprs = []
        for strspecies in arr_species:
            if strspecies[-1].isdigit() and f"X_{strspecies}" in estimators_lazy.collect_schema().names():
                cellmassfrac_exprs.append(pl.col(f"X_{strspecies}"))
            elif len(estimators_lazy.select(cs.matches(rf"^X_{strspecies}\d+")).collect_schema().names()) > 0:
                cellmassfrac_exprs.append(
                    pl.sum_horizontal(cs.matches(rf"^X_{strspecies}\d+"))
                )  # sum over all isotopes of this element
            else:
                cellmassfrac_exprs.append(pl.lit(float("-inf")))

        lazydfs = []
        for mgi in mgiplotlist:
            assert isinstance(mgi, int)
            combinedlzdf = (
                estimators_lazy
                .filter((pl.col("modelgridindex") == mgi).or_(mgi < 0))
                .group_by("timestep", maintain_order=True)
                .agg(
                    [
                        ((cellmassfracexpr * pl.col("mass_g")).sum() / pl.col("mass_g").sum()).alias(f"X_{strspecies}")
                        for strspecies, cellmassfracexpr in zip(arr_species, cellmassfrac_exprs, strict=True)
                    ]
                    + [pl.col("tmid_days").mean()]
                )
            )
            lazydfs.append((mgi, combinedlzdf))

        arr_abund_artis = dict(zip(mgiplotlist, pl.collect_all([lzdf for _, lzdf in lazydfs]), strict=True))

    return arr_abund_artis


def plot_qdot(
    modelpath: Path,
    dfcontribsparticledata: pl.LazyFrame | None,
    arr_time_gsi_days: Sequence[float] | None,
    pdfoutpath: Path | str,
    xmax: float | None = None,
) -> None:
    """Plot the ARTIS radioactive heating rate against the rate from the nuclear network trajectories."""
    try:
        depdata = at.misc.df_filter_minmax_bracketed(
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
            dfcontribsparticledata
            .select([
                pl
                .concat_arr(
                    (pl.col(col).arr.get(n) * pl.col("frac_of_cellmass") * pl.col("cellmass_on_mtot")).sum()
                    for n in range(len(arr_time_gsi_days))
                )
                .explode(empty_as_null=False)
                .alias(col)
                for col in heatcols
            ])
            .collect()
            .with_columns(time_days=pl.Series(arr_time_gsi_days))
        )
    else:
        dfgsiglobalheating = None

    fig, axesgrid = make_frame_figure()
    axis = axesgrid[0][0]

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

    set_legend(axis, ncol=3)

    axis.autoscale(enable=True, axis="both")
    axis.set_xmargin(0.02)
    axis.set_ymargin(0.02)
    save_figure(fig, pdfoutpath, format="pdf")


def plot_cell_abund_evolution(
    modelpath: Path,
    dfcontribsparticledata: pl.LazyFrame | None,
    arr_time_gsi_days: Sequence[float] | None,
    arr_species: Sequence[str],
    arr_abund_artis: pl.DataFrame | None,
    pdfoutpath: Path,
    mgi: int,
) -> None:
    """Plot the abundance evolution of one model cell, comparing ARTIS to the nuclear network trajectories."""
    if dfcontribsparticledata is not None:
        print(f"Calculating abundances in model cell {mgi} from the individual particle abundances")
        dfpartcontrib_thiscell = (
            dfcontribsparticledata.filter(pl.col("modelgridindex") == mgi) if mgi >= 0 else dfcontribsparticledata
        )
        frac_of_cellmass_sum = dfpartcontrib_thiscell.select(pl.col("frac_of_cellmass").sum()).collect().item()
        print(f"frac_of_cellmass_sum: {frac_of_cellmass_sum} (can be < 1.0 because of missing particles)")

        # we didn't include all cells (maybe), so we need a normalization factor here
        normfactor = (
            dfpartcontrib_thiscell
            .group_by("modelgridindex")
            .agg(pl.col("cellmass_on_mtot").first())
            .drop("modelgridindex")
            .sum()
            .collect()
            .item()
        )

        assert arr_time_gsi_days is not None
        df_gsi_abunds = dfpartcontrib_thiscell.select([
            pl
            .concat_arr(
                (pl.col(strnuc).arr.get(n) * pl.col("frac_of_cellmass") * pl.col("cellmass_on_mtot") / normfactor).sum()
                for n in range(len(arr_time_gsi_days))
            )
            .explode(empty_as_null=False)
            .alias(strnuc)
            for strnuc in arr_species
        ]).collect()
    else:
        df_gsi_abunds = None

    fig, axesgrid = make_frame_figure(rows=len(arr_species), aspect=0.383, sharex=False)
    axes = axesgrid[:, 0]
    axes[-1].set_xlabel("Time [days]")
    axis = axes[0]
    print(f"{'':7s}  gsi_abund artis_abund")

    for axis, strspecies in zip(axes, arr_species, strict=False):
        axis.set_ylabel("Mass fraction")

        strnuc_latex = strnuc_to_latex(strspecies)

        if df_gsi_abunds is not None:
            axis.plot(
                arr_time_gsi_days,
                df_gsi_abunds[strspecies],
                linewidth=2,
                marker="x",
                markersize=8,
                label=f"{strnuc_latex} GSINET",
                color="black",
            )

        print(f"{strspecies:7s}  ", end="")
        if df_gsi_abunds is not None:
            print(f"{df_gsi_abunds[strspecies][1]:.2e}", end="")
        else:
            print(" [no GSINET]", end="")

        if arr_abund_artis is not None and arr_abund_artis.select(pl.col(f"X_{strspecies}").is_finite().any()).item():
            print(f" {arr_abund_artis[f'X_{strspecies}'][0]:.2e}")
            axis.plot(
                arr_abund_artis["tmid_days"],
                arr_abund_artis[f"X_{strspecies}"],
                linewidth=2,
                label=f"{strnuc_latex} ARTIS",
                color="red",
            )
        else:
            print(" [no ARTIS data]")

        set_legend(axis, handlelength=1)

        axis.autoscale(enable=True, axis="both")
        axis.set_xmargin(0.02)
        axis.set_ymargin(0.05)

    strcell = f"cell {mgi}" if mgi >= 0 else "global"
    axes[0].set_title(f"{at.get_model_name(modelpath)} {strcell}")
    save_figure(fig, pdfoutpath, format="pdf")


def get_particledata(
    arr_time_s_incpremerger: Sequence[float] | npt.NDArray[np.floating],
    arr_strnuc_z_n: list[tuple[str, int, int | None]],
    traj_root: Path,
    particleid: int,
    verbose: bool = False,
) -> pl.LazyFrame:
    """For an array of times (NSM time including time before merger), interpolate the heating rates of various decay channels and (if arr_strnuc is not empty) the nuclear mass fractions."""
    try:  # ruff:ignore[too-many-statements-in-try-clause]
        if verbose:
            print(
                "Reading network calculation heating.dat,"
                f" energy_thermo.dat{', and nz-plane abundances' if arr_strnuc_z_n else ''} for particle {particleid}..."
            )

        particledata = pl.LazyFrame({"particleid": [particleid]}, schema={"particleid": pl.Int32})
        heatingfilepath = get_tar_member_extracted_path(
            traj_root=traj_root, particleid=particleid, memberfilename="./Run_rprocess/heating.dat"
        )
        heatcols = ["hbeta", "halpha", "hspof"]
        dfheating = at.read_wsv(heatingfilepath).select("#count", "time/s", *heatcols)
        dfheating = dfheating.with_columns(fix_fortran_exponents(pl.Float64))

        nstep_timesec: dict[int, float] = dict(dfheating.select("#count", "time/s").iter_rows())

        particledata = particledata.with_columns(
            pl.Series(
                # np.interp holds the end value outside the range. These curves are summed over the
                # trajectories, thus a NaN here would make the whole population sum NaN.
                [np.interp(arr_time_s_incpremerger, dfheating["time/s"], dfheating[col])],
                dtype=pl.Array(pl.Float32, len(arr_time_s_incpremerger)),
            ).alias(col)
            for col in heatcols
        )

        if arr_strnuc_z_n:
            ntslowers = at.inputmodel.rprocess_from_trajectory.get_closest_network_timesteps(
                traj_root, particleid, arr_time_s_incpremerger, cond="lessthan"
            )
            ntsuppers = at.inputmodel.rprocess_from_trajectory.get_closest_network_timesteps(
                traj_root, particleid, arr_time_s_incpremerger, cond="greaterthan"
            )
            nts_list = sorted(set(ntslowers + ntsuppers))
            nts_count = len(nts_list)
            arr_massfracs = {strnuc: np.zeros(nts_count, dtype=np.float32) for strnuc, _, _ in arr_strnuc_z_n}
            for i, nts in enumerate(nts_list):
                dftrajnucabund, traj_time_s = (
                    at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfile_nuc_abund(
                        traj_root, particleid, f"./Run_rprocess/nz-plane{nts:05d}"
                    )
                )
                # nts is the exact network step, thus these two times come from the same step and
                # agree to the precision of the file
                at.inputmodel.rprocess_from_trajectory.check_traj_time_matches(
                    particleid, traj_time_s, nstep_timesec[nts], rel_tol=1e-6, abs_tol=0.0
                )
                # one sum per element and one per nuclide, and then a lookup for each species
                massfrac_of_z = dict(dftrajnucabund.group_by("Z").agg(pl.col("massfrac").sum()).iter_rows())
                massfrac_of_z_n = {
                    (Z, N): massfrac
                    for Z, N, massfrac in dftrajnucabund.group_by("Z", "N").agg(pl.col("massfrac").sum()).iter_rows()
                }
                for strnuc, Z, N in arr_strnuc_z_n:
                    arr_massfracs[strnuc][i] = (
                        massfrac_of_z.get(Z, 0.0) if N is None else massfrac_of_z_n.get((Z, N), 0.0)
                    )

            particledata = particledata.with_columns(
                pl.Series(
                    [
                        np.interp(
                            arr_time_s_incpremerger, [nstep_timesec[nts] for nts in nts_list], arr_massfracs[strnuc]
                        )
                    ],
                    dtype=pl.Array(pl.Float32, len(arr_time_s_incpremerger)),
                ).alias(strnuc)
                for strnuc, _, _ in arr_strnuc_z_n
            )

    except FileNotFoundError:
        print(f"No network calculation for particle {particleid}")
        # make sure we weren't requesting abundance data for this particle that has no network data
        if arr_strnuc_z_n:
            print("ERROR:", particleid, arr_strnuc_z_n)
        assert not arr_strnuc_z_n
        return pl.LazyFrame()

    return particledata


def get_dfcontribsparticledata(
    modelpath: Path | str,
    mgiplotlist: Sequence[int],
    arr_strnuc_z_n: list[tuple[str, int, int | None]],
    traj_root: Path,
    griddata_root: Path,
    lzdfmodel: pl.LazyFrame,
    arr_time_gsi_days: list[float],
) -> pl.LazyFrame:
    """Get the grid particle contributions joined with their interpolated heating rates and abundances."""
    # times in artis are relative to merger, but NSM simulation time started earlier
    mergertime_geomunits = at.inputmodel.modelfromhydro.get_merger_time_geomunits(griddata_root)
    t_mergertime_s = mergertime_geomunits * 4.926e-6
    arr_time_gsi_s_incpremerger = np.array(arr_time_gsi_days) * day_to_s + t_mergertime_s

    dfpartcontrib = (
        at.inputmodel.rprocess_from_trajectory
        .get_gridparticlecontributions(modelpath)
        .lazy()
        .with_columns(modelgridindex=pl.col("cellindex") - 1)
        .filter(pl.col("frac_of_cellmass") > 0)
    ).join(
        lzdfmodel.select(["modelgridindex", "cellmass_on_mtot"]),
        on="modelgridindex",
        how="inner",
        maintain_order="left",
    )

    allcontribparticleids = dfpartcontrib.select(pl.col("particleid").unique()).collect().to_series().to_list()
    list_particleids_getabund = (
        dfpartcontrib
        .filter(pl.col("modelgridindex").is_in(mgiplotlist).or_(any(mgi < 0 for mgi in mgiplotlist)))
        .select(pl.col("particleid").unique())
        .collect()
        .to_series()
        .to_list()
    )
    fworkerwithabund = partial(get_particledata, arr_time_gsi_s_incpremerger, arr_strnuc_z_n, traj_root, verbose=False)

    print(f"Reading trajectories from {traj_root}")
    print(f"Reading Qdot/thermo and abundance data for {len(list_particleids_getabund)} particles")

    list_particledata_withabund = at.parallel_map(fworkerwithabund, list_particleids_getabund)
    print("  done")
    particleids_getabund = set(list_particleids_getabund)
    list_particleids_noabund = [pid for pid in allcontribparticleids if pid not in particleids_getabund]
    fworkernoabund = partial(get_particledata, arr_time_gsi_s_incpremerger, [], traj_root)
    print(f"Reading for Qdot/thermo data (no abundances needed) for {len(list_particleids_noabund)} particles")

    list_particledata_noabund = at.parallel_map(fworkernoabund, list_particleids_noabund, chunksize=16)
    print("  done")

    allparticledata = pl.concat(list_particledata_withabund + list_particledata_noabund, how="diagonal")

    return dfpartcontrib.join(allparticledata, on="particleid", how="inner", maintain_order="left")


def plot_qdot_abund_modelcells(
    modelpath: Path,
    merger_root: Path,
    mgiplotlist: Sequence[int],
    arr_species: list[str],
    timedaysmax: float | None = None,
    nogsinet: bool = False,
) -> None:
    """Plot the heating rate and the abundance evolution of each cell in mgiplotlist."""
    lzdfmodel, modelmeta = at.inputmodel.get_modeldata(
        modelpath, derived_cols=["mass_g", "rho", "logrho", "volume"], get_elemabundances=True
    )

    # default values, because early model.txt didn't specify this
    griddatafolder: Path = Path("SFHo_snapshot")
    mergermodelfolder: Path = Path("SFHo_short")
    trajfolder: Path = Path("SFHo")
    for line in modelmeta["headercommentlines"]:
        if line.startswith("gridfolder:"):
            griddatafolder = Path(line.strip().removeprefix("gridfolder: "))
            mergermodelfolder = Path(line.strip().removeprefix("gridfolder: ").removesuffix("_snapshot"))
        elif line.startswith("trajfolder:"):
            trajfolder = Path(line.strip().removeprefix("trajfolder: ").replace("SFHO", "SFHo"))

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

    arr_z = [at.get_atomic_number(species) for species in arr_species]
    arr_a = [
        int(a) if a is not None else a
        for a in [species.lstrip(string.ascii_letters) or None for species in arr_species]
    ]
    arr_n = [a - z if a is not None else None for z, a in zip(arr_z, arr_a, strict=True)]
    arr_strnuc_z_n = list(zip(arr_species, arr_z, arr_n, strict=True))

    lzdfmodel = lzdfmodel.with_columns(cellmass_on_mtot=pl.col("mass_g") / pl.col("mass_g").sum())

    model_mass_grams = lzdfmodel.select(pl.col("mass_g").sum()).collect().item()
    print(f"model mass: {model_mass_grams / Msun_to_g:.3f} Msun")

    dftimesteps = at.misc.df_filter_minmax_bracketed(
        at.get_timesteps(modelpath).select("timestep", "tmid_days"), "tmid_days", None, timedaysmax
    ).collect()

    arr_time_artis_days_alltimesteps = dftimesteps.select(pl.col("tmid_days")).to_series().to_numpy()

    if gsinet_available:
        arr_time_gsi_days = [modelmeta["t_model_init_days"], *arr_time_artis_days_alltimesteps]
        dfcontribsparticledata = get_dfcontribsparticledata(
            modelpath=modelpath,
            mgiplotlist=mgiplotlist,
            arr_strnuc_z_n=arr_strnuc_z_n,
            traj_root=traj_root,
            arr_time_gsi_days=arr_time_gsi_days,
            griddata_root=griddata_root,
            lzdfmodel=lzdfmodel,
        )

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
        arr_abund_artis = get_artis_abund_sequences(
            modelpath=modelpath,
            dftimesteps=dftimesteps,
            mgiplotlist=mgiplotlist,
            arr_species=arr_species,
            correction_factors=correction_factors,
        )

        for mgi in mgiplotlist:
            print()
            strmgi = f"mgi{mgi}" if mgi >= 0 else "global"
            plot_cell_abund_evolution(
                modelpath,
                dfcontribsparticledata,
                arr_time_gsi_days,
                arr_species,
                arr_abund_artis.get(mgi),
                mgi=mgi,
                pdfoutpath=Path(modelpath, f"gsinetwork_{strmgi}-abundance.pdf"),
            )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path(), helptext="Path for ARTIS files")

    parser.add_argument(
        "-mergerroot",
        type=Path,
        default=Path(Path.home() / "Google Drive/Shared Drives/GSI NSM/Mergers"),
        help="Base path for merger snapshot and trajectory data specified in model.txt",
    )

    addarg_output(parser, kind="folder", default=Path())

    parser.add_argument("-xmax", default=None, type=float, help="Maximum time in days to plot")

    # the help named a list such as 4,5,6, and nargs="*" with the type int took "4 5 6" and refused
    # that list. One builder gives every command the same text: a number, a range 3-7, or a list 4,5,6
    at.addarg_modelgridindex(parser, default=[], helptext="Model grid cell to plot, or a list such as 4,5,6")

    parser.add_argument(
        "--nogsinet", action="store_true", help="Do not attempt to read GSI Network data even if available"
    )

    parser.add_argument(
        "-species",
        type=str,
        default=[
            "He4",
            "Sr",
            "Y",
            "Zr",
            "Ga72",
            "Cu77",
            "Sr89",
            "Sr91",
            "I129",
            "I132",
            "Rb88",
            "Y92",
            "Sb128",
            "Cu66",
            "Cf254",
        ],
        nargs="*",
        help=("Element symbols or isotope names to plot abundances for, e.g., Sr Sr89 Y Zr"),
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Compare the energy release and abundances from ARTIS to the GSI Network calculation."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    print(f"Selected species: {' '.join(args.species)}")
    plot_qdot_abund_modelcells(
        modelpath=Path(args.modelpath),
        merger_root=Path(args.mergerroot),
        mgiplotlist=at.parse_range_list(args.modelgridindex) if args.modelgridindex else [],
        arr_species=args.species,
        timedaysmax=args.xmax,
        nogsinet=args.nogsinet,
    )


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
