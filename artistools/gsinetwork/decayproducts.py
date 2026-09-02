"""Script to load beta-decay (beta- and beta+) energy release data from nucleosynthesis trajectories. Optionally also writes output to parquet files."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import math
import multiprocessing as mp
import typing as t
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.commands import get_path
from artistools.constants import amu_g
from artistools.constants import day_to_s
from artistools.constants import MEV_to_erg
from artistools.constants import Msun_to_g
from artistools.inputmodel.rprocess_from_trajectory import fix_fortran_exponents
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path
from artistools.misc import addarg_figscale
from artistools.misc import print_warning
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend

ARTIS_colors = ["r", "g", "b", "m", "c", "orange"]  # reddish colors


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument(
        "-trajectoryroot", "-trajroot", required=True, type=Path, help="Path to nuclear network trajectory folder"
    )

    parser.add_argument(
        "-npz", default=None, type=Path, help="Path to npz file which specifies the ejecta type of each trajectory"
    )

    parser.add_argument("-timemin", "-tmin", dest="tmin", type=float, default=0.1, help="Minimum time in days")

    parser.add_argument("-timemax", "-tmax", dest="tmax", type=float, default=80.0, help="Maximum time in days")

    parser.add_argument("-nsteps", type=int, default=64, help="Number of timesteps")

    parser.add_argument(
        "-nucdata",
        default="hotokezaka",
        choices=["hotokezaka", "ensdf"],
        help='Nuclear dataset to use, either "hotokezaka" or "ensdf"',
    )

    parser.add_argument(
        "-yemax",
        type=float,
        default=0.52,
        help="Y_e,max of hydro model considered. Default 0.52 for e2e sym-n1a6 from Just+23",
    )

    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Prints output dictionaries of full Ye bins or ejecta components to parquet files",
    )

    parser.add_argument("--nuclides", action="store_true", help="Calculates contributions of individual nuclides")

    parser.add_argument(
        "--trajparquet", action="store_true", help="Writes individual parquet files for all trajectories"
    )

    at.addarg_output(parser, kind="folder", default=Path(), helptext="Path for output PDF and parquet files")

    addarg_figscale(parser)


def append_electroncapture_betaplus_nuclei(df: pl.DataFrame, nuc_dataset: str) -> pl.DataFrame:
    """Append the electron capture and beta-plus decays, which the beta-minus data files do not cover."""
    data = {
        "A": [48, 48, 52, 56, 56, 57, 57],
        "Z": [23, 24, 25, 27, 28, 27, 28],
        "Q[MeV]": [4.015, 1.657, 4.711, 4.566, 2.136, 0.836, 3.264],
        "Egamma[MeV]": [2.919, 0.416, 5.857, 3.566, 1.728, 0.120, 1.928],
        "Eelec[MeV]": [0.147, 0.001365, 0.071, 0.122, 0.0, 0.0, 0.154],
        "Eneutrino[MeV]": [0.949, 1.240, 1.040, 0.878, 0.408, 0.716, 1.182],
        "tau[s]": [
            1991140.7543850502,
            111976.2182936378,
            696911.7289199209,
            9627378.69698784,
            757989.666170996,
            33872078.915524825,
            185103.5445262176,
        ],
    }

    new_rows = pl.DataFrame(data) if nuc_dataset == "Hotokezaka" else pl.DataFrame(data | {"source": ["ENSDF"] * 7})

    return pl.concat([df, new_rows], how="vertical_relaxed")


def get_nuc_data(nuc_dataset: str) -> pl.DataFrame:
    """Return the decay energy per channel for every nuclide, from either the Hotokezaka or ENSDF dataset."""
    assert nuc_dataset in {"Hotokezaka", "ENSDF"}
    hotokezaka_betaminus = (
        pl
        .read_csv(
            get_path("datadir") / "betaminusdecays.txt",
            separator=" ",
            comment_prefix="#",
            has_header=False,
            new_columns=["A", "Z", "Q[MeV]", "Egamma[MeV]", "Eelec[MeV]", "Eneutrino[MeV]", "tau[s]"],
        )
        .filter(pl.col("Q[MeV]") > 0.0)
        .with_columns(pl.col(pl.Int32).cast(pl.Int64))
    )
    if nuc_dataset == "Hotokezaka":
        return append_electroncapture_betaplus_nuclei(hotokezaka_betaminus, nuc_dataset)
    csvpath = Path(get_path("datadir"), "betaminusdecays_ensdf.txt")
    if not csvpath.exists():
        print("Collecting ENSDF data...")
        import urllib.request

        rows = []
        for hrow in hotokezaka_betaminus.iter_rows(named=True):
            atomic_number = hrow["Z"]
            A = hrow["A"]
            elsymb = at.get_elsymbol(atomic_number)
            print(f"Element: Z={atomic_number} {elsymb} A={A}")
            isot_str = f"{A}{elsymb.lower()}"
            request = urllib.request.Request(
                f"https://nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides={isot_str}&rad_types=bm",
                headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0"},
            )
            with urllib.request.urlopen(request) as response:  # ruff:ignore[suspicious-url-open-usage]
                dfnuc = pl.read_csv(response.read(), infer_schema_length=None)
            if "mean_energy" in dfnuc.columns:
                dfnuc = dfnuc.drop_nulls(subset="mean_energy")
            if dfnuc.height > 0:
                dfnuc = dfnuc.filter(pl.col("p_energy") == 0)
                if dfnuc.is_empty():
                    print(f"No beta decay found for Z={atomic_number} A={A}")
                    continue
                # a missing value poisons the derived quantities to NaN (visible in the written file)
                # instead of crashing on None or being silently skipped by the sums
                dfnuc = dfnuc.with_columns(
                    pl
                    .col("half_life_sec", "q", "intensity_beta", "mean_energy", "anti_nu_mean_energy")
                    .cast(pl.Float64)
                    .fill_null(math.nan)
                )
                tau_s = dfnuc["half_life_sec"].item(0) / math.log(2)
                Q_MeV = dfnuc["q"].item(0) / 1000
                E_elec = (dfnuc["intensity_beta"] * dfnuc["mean_energy"]).sum() / 100 / 1000
                E_nu = (dfnuc["intensity_beta"] * dfnuc["anti_nu_mean_energy"]).sum() / 100 / 1000
                E_gamma = Q_MeV - E_elec - E_nu
                rows.append({
                    "A": A,
                    "Z": atomic_number,
                    "Q[MeV]": Q_MeV,
                    "Egamma[MeV]": E_gamma,
                    "Eelec[MeV]": E_elec,
                    "Eneutrino[MeV]": E_nu,
                    "tau[s]": tau_s,
                    "source": "ENSDF",
                })
            else:
                print(f"No ENSDF data found for Z={atomic_number} A={A}")
                rows.append(hrow | {"source": "Hotokezaka"})

        with csvpath.open("w", encoding="utf-8") as f:
            f.writelines(("# Data from ENSDF database\n", "#\n# "))
            pl.DataFrame(rows).write_csv(f, separator=" ", include_header=True)
        print("done!")

    ensdf_betaminus = pl.read_csv(
        csvpath,
        separator=" ",
        comment_prefix="#",
        has_header=False,
        new_columns=["A", "Z", "Q[MeV]", "Egamma[MeV]", "Eelec[MeV]", "Eneutrino[MeV]", "tau[s]", "source"],
    )
    return append_electroncapture_betaplus_nuclei(ensdf_betaminus, nuc_dataset)


def process_trajectory(
    nuc_data: pl.DataFrame,
    traj_root: Path | str,
    traj_masses_g: dict[int, float],
    arr_t_day: npt.NDArray[np.floating],
    nuclide_contrib: bool,
    traj_parquet_dir: Path | None,
    traj_ID: int,
) -> dict[str, npt.NDArray[np.floating]]:
    """Process a single trajectory to extract decay powers."""
    traj_mass_grams = traj_masses_g[traj_ID]
    traj_root = Path(traj_root)
    dfheatingthermo = (
        at
        .read_wsv(
            get_tar_member_extracted_path(
                traj_root=traj_root, particleid=traj_ID, memberfilename="./Run_rprocess/heating.dat"
            )
        )
        .select("#count", "hbeta", "htot")
        .with_columns(fix_fortran_exponents(pl.Float64))
        .join(
            at.read_wsv(
                get_tar_member_extracted_path(
                    traj_root=traj_root, particleid=traj_ID, memberfilename="./Run_rprocess/energy_thermo.dat"
                )
            ).select("#count", "time/s", "Qdot"),
            on="#count",
            how="left",
            coalesce=True,
            maintain_order="left",
        )
        .rename({"#count": "nstep", "time/s": "timesec"})
    )

    # get nearest network time to each plotted time
    arr_networktimedays = dfheatingthermo["timesec"].to_numpy() / day_to_s
    networktimestepindices = [
        int(dfheatingthermo["nstep"].item(int(np.abs(arr_networktimedays - plottimedays).argmin())))
        if plottimedays < arr_networktimedays[-1]
        else -1
        for plottimedays in arr_t_day
    ]

    decay_powers: dict[str, npt.NDArray[np.floating]]
    decay_powers = {
        key: np.zeros(len(arr_t_day))
        for key in (
            "abundweighted_nu",
            "abundweighted_elec",
            "abundweighted_gamma",
            "hbeta",
            "htot",
            "Qdot",
            "abundweighted_Qdot",
        )
    }
    decay_powers |= {
        col: (
            np.array([
                dfheatingthermo[col][networktimestepindex - 1] if networktimestepindex >= 1 else 0.0
                for networktimestepindex in networktimestepindices
            ])
            * traj_mass_grams
        )
        for col in ("hbeta", "htot", "Qdot")
    }
    decay_powers |= {"timedays": np.array(arr_t_day)}

    A_arr = nuc_data["A"].to_numpy()
    Z_arr = nuc_data["Z"].to_numpy()

    if nuclide_contrib:
        for AZ_tuple in zip(A_arr, Z_arr, strict=False):
            decay_powers[f"({int(AZ_tuple[0])},{int(AZ_tuple[1])})_elec"] = np.zeros(len(arr_t_day))
            decay_powers[f"({int(AZ_tuple[0])},{int(AZ_tuple[1])})_gam"] = np.zeros(len(arr_t_day))
            decay_powers[f"({int(AZ_tuple[0])},{int(AZ_tuple[1])})_nu"] = np.zeros(len(arr_t_day))

    # now get abundances from single timestep files
    for plottimestep, networktimestepindex in enumerate(networktimestepindices):
        if networktimestepindex < 1:
            continue

        dftrajnucabund, _networktime = at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfile_nuc_abund(
            traj_root=traj_root, particleid=traj_ID, memberfilename=f"./Run_rprocess/nz-plane{networktimestepindex:05d}"
        )

        assert dftrajnucabund.height > 100, dftrajnucabund.height

        pldf_all = (
            dftrajnucabund
            .lazy()
            .filter(pl.col("massfrac") > 0.0)
            .with_columns([
                pl.col(pl.Int32).cast(pl.Int64),
                pl.col(pl.Float32).cast(pl.Float64),
                (pl.col("Z") + pl.col("N")).alias("A"),
                (pl.col("massfrac") * traj_mass_grams / ((pl.col("Z") + pl.col("N")) * amu_g)).alias("num_nuc"),
            ])
            .join(nuc_data.lazy(), on=("Z", "A"), how="inner", maintain_order="left")
            .with_columns([(pl.col("num_nuc") / pl.col("tau[s]")).alias("N_dot")])
            .with_columns([
                (pl.col("N_dot") * pl.col("Eneutrino[MeV]") * MEV_to_erg).alias("eps_nu"),
                (pl.col("N_dot") * pl.col("Eelec[MeV]") * MEV_to_erg).alias("eps_elec"),
                (pl.col("N_dot") * pl.col("Egamma[MeV]") * MEV_to_erg).alias("eps_gamma"),
                (pl.col("N_dot") * pl.col("Q[MeV]") * MEV_to_erg).alias("eps_tot"),
            ])
            .collect()
        )

        global_sums = pldf_all.select(
            abundweighted_nu=pl.sum("eps_nu"),
            abundweighted_elec=pl.sum("eps_elec"),
            abundweighted_gamma=pl.sum("eps_gamma"),
            abundweighted_Qdot=pl.sum("eps_tot"),
        )
        for col in global_sums.columns:
            decay_powers[col][plottimestep] = float(global_sums.get_column(col).item())

        if nuclide_contrib:
            # store all nuclide contributions in detail
            grouped = pldf_all.group_by(["A", "Z"]).agg([
                pl.sum("eps_elec").alias("eps_elec"),
                pl.sum("eps_gamma").alias("eps_gamma"),
                pl.sum("eps_nu").alias("eps_nu"),
            ])
            for A, Z, Qe, Qg, Qn in grouped.select("A", "Z", "eps_elec", "eps_gamma", "eps_nu").iter_rows():
                decay_powers[f"({A},{Z})_elec"][plottimestep] = Qe
                decay_powers[f"({A},{Z})_gam"][plottimestep] = Qg
                decay_powers[f"({A},{Z})_nu"][plottimestep] = Qn

    # dump to parquet
    if traj_parquet_dir is not None:
        traj_df = pl.DataFrame(decay_powers)

        # perform time interpolation to prevent effects from low trajectory time grid resolution
        value_cols = [c for c in traj_df.columns if c != "timedays"]
        last_row = traj_df.tail(1)
        main = traj_df.slice(0, traj_df.height - 1).with_columns([
            pl
            .when(pl.col(c) == pl.col(c).shift())
            .then(None)
            .otherwise(pl.col(c))
            .interpolate_by("timedays")
            .forward_fill()
            .backward_fill()
            .alias(c)
            for c in value_cols
        ])
        traj_df = pl.concat([main, last_row])
        traj_df = traj_df.fill_null(0.0).fill_nan(0.0)

        # an output file overwrites whatever a previous run left at the path
        trajparquetpath = traj_parquet_dir / f"decay_powers_{traj_ID}.parquet"
        at.write_parquet_atomic(traj_df, trajparquetpath, replaces=at.get_file_identity(trajparquetpath))
    return decay_powers


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Comparison to constant beta decay splitup factors."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    nuc_dataset = "Hotokezaka" if args.nucdata == "hotokezaka" else "ENSDF"

    Ye_bins = [
        ("all", 0.0, float("inf")),
        ("low", 0.0, args.yemax / 3),
        ("mid", args.yemax / 3, args.yemax * 2 / 3),
        ("high", args.yemax * 2 / 3, args.yemax),
    ]

    if args.npz:
        npz_dict = np.load(args.npz)
        npz_idcs = np.asarray(npz_dict["idx"])
        npz_types = np.asarray(npz_dict["state"])

    # get beta decay data
    nuc_data = get_nuc_data(nuc_dataset)
    parquet_dir: Path | None = None
    if args.parquet or args.trajparquet:
        parquet_dir = Path(args.outputfile) / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing parquet files to '{parquet_dir}'.")
    assert nuc_data.height == nuc_data.unique(("Z", "A")).height

    # set timesteps logarithmically
    log_t_compar_min_s = np.log10(args.tmin)
    log_t_compar_max_s = np.log10(args.tmax)
    arr_t_day = 10 ** (np.linspace(log_t_compar_min_s, log_t_compar_max_s, args.nsteps, endpoint=True))

    # get masses of trajectories
    summarypath = Path(args.trajectoryroot, "summary-all.dat")
    with summarypath.open("r", encoding="utf-8") as f:
        if not f.readline().startswith("#"):
            msg = "ERROR: No header found in summary-all.dat. Please check the file format."
            raise ValueError(msg)

    traj_summ_data = at.read_wsv(summarypath, comment_prefix="#", header_from_comment=True).filter(
        pl.any_horizontal(pl.col("Ye").is_between(Ye_lower, Ye_upper) for _, Ye_lower, Ye_upper in Ye_bins)
    )

    print(traj_summ_data)

    traj_ids = traj_summ_data["Id"].to_list()

    traj_masses_g = {int(trajid): mass * Msun_to_g for trajid, mass in traj_summ_data[["Id", "Mass"]].to_numpy()}

    alltraj_decay_powers: list[dict[str, npt.NDArray[np.floating]]] = at.parallel_map(
        partial(
            process_trajectory,
            nuc_data,
            args.trajectoryroot,
            traj_masses_g,
            arr_t_day,
            args.nuclides,
            parquet_dir if args.trajparquet else None,
        ),
        traj_ids,
        chunksize=2,
        desc="Processing trajectories",
        unit="traj",
        smoothing=0.0,
    )

    print()

    ej_states = ["any", -1, 0, 1]
    ej_names = ["all", "dyn", "hmns", "torus"]
    for i in range(4):
        state = ej_states[i]
        if not args.npz:
            label, Ye_lower, Ye_upper = Ye_bins[i]
            labelfull = f"Ye [{Ye_lower}, {Ye_upper}]" if math.isfinite(Ye_upper) else "all Ye"
            print(f"Processing Ye bin {label}... Ye: [{Ye_lower}, {Ye_upper}]")
            selected_traj_ids = traj_summ_data.filter(pl.col("Ye").is_between(Ye_lower, Ye_upper))["Id"].to_list()

            print(f" {len(selected_traj_ids)} trajectories selected")
            if len(selected_traj_ids) == 0:
                print(f"No trajectories found for Ye [{Ye_lower}, {Ye_upper}]")
                continue
        else:
            # select by ejecta type
            selected_traj_ids = (
                traj_ids if state == "any" else list(set(traj_ids) & set(npz_idcs[npz_types == state].tolist()))
            )
            print(f" {len(selected_traj_ids)} trajectories selected")
            if len(selected_traj_ids) == 0:
                print_warning(f"No trajectories found for eject state {state}")
                continue
            labelfull = ej_names[i]
            label = ej_names[i]

        decay_powers = {
            k: sum(
                trajdata[k]
                for traj_id, trajdata in zip(traj_ids, alltraj_decay_powers, strict=True)
                if traj_id in selected_traj_ids
            )
            for k in alltraj_decay_powers[0]
            if k != "timedays"
        }
        decay_powers["timedays"] = np.array(arr_t_day)

        assert isinstance(decay_powers["abundweighted_gamma"], np.ndarray)
        assert isinstance(decay_powers["abundweighted_elec"], np.ndarray)
        assert isinstance(decay_powers["abundweighted_nu"], np.ndarray)
        decay_powers["abundweighted_gammanuelec"] = (
            decay_powers["abundweighted_gamma"] + decay_powers["abundweighted_nu"] + decay_powers["abundweighted_elec"]
        )

        if args.parquet:
            assert parquet_dir is not None
            traj_set_df = pl.DataFrame(decay_powers)
            # an output file overwrites whatever a previous run left at the path
            setparquetpath = parquet_dir / f"decay_powers_{labelfull}.parquet"
            at.write_parquet_atomic(traj_set_df, setparquetpath, replaces=at.get_file_identity(setparquetpath))

        fig, axesgrid = make_frame_figure(args, rows=2, aspect=0.913)
        axes = axesgrid[:, 0]
        ax0 = axes[0]
        ax0.axhline(y=0.45, color=ARTIS_colors[2], linestyle="dotted", label=r"Barnes+16 $\gamma$")
        ax0.axhline(y=0.20, color=ARTIS_colors[0], linestyle="dotted", label=r"Barnes+16 $e^{-}$")
        ax0.axhline(y=0.35, color=ARTIS_colors[1], linestyle="dotted", label=r"Barnes+16 $\nu$")
        ax0.plot(
            arr_t_day,
            decay_powers["abundweighted_gamma"] / decay_powers["abundweighted_gammanuelec"],
            color=ARTIS_colors[2],
            linestyle="-",
            label=f"Traj {labelfull} gamma",
        )
        ax0.plot(
            arr_t_day,
            decay_powers["abundweighted_elec"] / decay_powers["abundweighted_gammanuelec"],
            color=ARTIS_colors[0],
            linestyle="-",
            label=rf"Traj {labelfull} $e^{{-}}$",
        )
        ax0.plot(
            arr_t_day,
            decay_powers["abundweighted_nu"] / decay_powers["abundweighted_gammanuelec"],
            color=ARTIS_colors[1],
            linestyle="-",
            label=rf"Traj {labelfull} $\nu$",
        )
        ax0.set_ylim(0.15, 0.55)
        ax0.set_ylabel("energy release rate / Qdot")
        ax1 = axes[1]
        ax1.plot(arr_t_day, decay_powers["Qdot"], linestyle="-", linewidth=3, label=f"Traj {labelfull} Qdot")
        ax1.plot(
            arr_t_day,
            decay_powers["abundweighted_gammanuelec"],
            linestyle="-",
            linewidth=2,
            label=f"Traj {labelfull} abund -> beta + gamma + nu",
        )
        ax1.plot(
            arr_t_day, decay_powers["abundweighted_Qdot"], linestyle="-", label=f"Traj {labelfull} abund -> Qdot_beta"
        )
        ax1.set_ylabel("Energy release rate [erg/s]")
        ax1.set_yscale("log")
        for ax in axes:
            set_legend(ax, args)
            ax.set_xscale("log")
        axes[-1].set_xlabel("Time [days]")

        outfilepath = args.outputfile / f"beta_release_ratios_tot_{nuc_dataset}_Ye{label}.pdf"
        save_figure(fig, outfilepath)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    mp.freeze_support()
    run_module_as_subcommand(__spec__)
