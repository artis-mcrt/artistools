"""Script to load energy release data from nucleosynthesis trajectories. Optionally also writes output to a JSON."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import json
import math
import multiprocessing as mp
import typing as t
import warnings
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.configuration import get_config
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path

ARTIS_colors = ["r", "g", "b", "m", "c", "orange"]  # reddish colors
M_sol_cgs = 1.989e33
amu_g = 1.66e-24
MeV_to_erg = 1.60218e-6


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-trajectoryroot", "-trajroot", required=True, help="Path to nuclear network trajectory folder")

    parser.add_argument(
        "-npz", default=None, help="Path to npz file which specifies the ejecta type of each trajectory"
    )

    parser.add_argument("-tmin", type=float, default=0.1, help="Minimum time in days")

    parser.add_argument("-tmax", type=float, default=80.0, help="Maximum time in days")

    parser.add_argument("-nsteps", type=int, default=64, help="Number of timesteps")

    parser.add_argument("-nucdata", type=str, default="ensdf", help='Nuclear dataset to use, either "hoto" or "ensdf"')

    parser.add_argument(
        "-yemax",
        type=float,
        default=0.52,
        help="Y_e,max of hydro model considered. Default 0.52 for e2e sym-n1a6 from Just+23",
    )

    parser.add_argument(
        "--json", action="store_true", help="Prints output dictionaries of full Ye bins or ejecta componenets to a JSON"
    )

    parser.add_argument("--nuclides", action="store_true", help="Calculates contributions of individual nuclides")

    parser.add_argument("--trajjson", action="store_true", help="Writes individual JSONs for all trajectories.")


def get_nuc_data(nuc_dataset: str) -> pl.DataFrame:
    import pandas as pd

    assert nuc_dataset in {"Hotokezaka", "ENSDF"}
    hotokezaka_betaminus = (
        pl
        .read_csv(
            get_config()["path_datadir"] / "betaminusdecays.txt",
            separator=" ",
            comment_prefix="#",
            has_header=False,
            new_columns=["A", "Z", "Q[MeV]", "Egamma[MeV]", "Eelec[MeV]", "Eneutrino[MeV]", "tau[s]"],
        )
        .filter(pl.col("Q[MeV]") > 0.0)
        .with_columns(pl.col(pl.Int32).cast(pl.Int64))
    )
    if nuc_dataset == "Hotokezaka":
        return hotokezaka_betaminus
    csvpath = Path(get_config()["path_datadir"], "betaminusdecays_ensdf.txt")
    if not csvpath.exists():
        print("Collecting ENSDF data...")
        rows = []
        for hrow in hotokezaka_betaminus.iter_rows(named=True):
            atomic_number = hrow["Z"]
            A = hrow["A"]
            elsymb = at.get_elsymbol(atomic_number)
            print(f"Element: Z={atomic_number} {elsymb} A={A}")
            isot_str = f"{A}{elsymb.lower()}"
            dfnuc = pd.read_csv(
                f"https://nds.iaea.org/relnsd/v1/data?fields=decay_rads&nuclides={isot_str}&rad_types=bm",
                storage_options={
                    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0"
                },
            )
            if "mean_energy" in dfnuc.columns:
                dfnuc = dfnuc.dropna(subset=["mean_energy"])
            if len(dfnuc) > 0:
                dfnuc = dfnuc.loc[dfnuc["p_energy"] == 0]
                if dfnuc.empty:
                    print(f"No beta decay found for Z={atomic_number} A={A}")
                    continue
                tau_s = dfnuc.iloc[0]["half_life_sec"] / math.log(2)
                # tau_s = hrow["tau[s]"]
                Q_MeV = dfnuc.iloc[0]["q"] / 1000
                E_elec = (dfnuc["intensity_beta"] * dfnuc["mean_energy"]).sum() / 100 / 1000
                E_nu = (dfnuc["intensity_beta"] * dfnuc["anti_nu_mean_energy"]).sum() / 100 / 1000
                # dfnuc["E_gamma"] = (Q_MeV * 1000 - dfnuc["mean_energy"] - dfnuc["anti_nu_mean_energy"]) / 1000
                # E_gamma = (dfnuc["intensity_beta"] * dfnuc["E_gamma"]).sum() / 1000
                # E_gamma = max(0, E_gamma)
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

    return pl.read_csv(
        csvpath,
        separator=" ",
        comment_prefix="#",
        has_header=False,
        new_columns=["A", "Z", "Q[MeV]", "Egamma[MeV]", "Eelec[MeV]", "Eneutrino[MeV]", "tau[s]", "source"],
    )


def process_trajectory(
    nuc_data: pl.DataFrame,
    traj_root: Path | str,
    traj_masses_g: dict[int, float],
    arr_t_day: npt.NDArray[np.floating],
    nuclide_contrib: bool,
    traj_json: bool,
    traj_ID: int,
) -> dict[str, npt.NDArray[np.floating]]:
    """Process a single trajectory to extract decay powers."""
    traj_mass_grams = traj_masses_g[traj_ID]
    traj_root = Path(traj_root)
    import pandas as pd

    dfheatingthermo = (
        pl
        .from_pandas(
            pd.read_csv(
                get_tar_member_extracted_path(
                    traj_root=traj_root, particleid=traj_ID, memberfilename="./Run_rprocess/heating.dat"
                ),
                sep=r"\s+",
                usecols=["#count", "hbeta", "htot"],
            )
        )
        .with_columns(pl.col("htot").cast(pl.Float64, strict=False))
        .join(
            pl.from_pandas(
                pd.read_csv(
                    get_tar_member_extracted_path(
                        traj_root=traj_root, particleid=traj_ID, memberfilename="./Run_rprocess/energy_thermo.dat"
                    ),
                    sep=r"\s+",
                    usecols=["#count", "time/s", "Qdot"],
                )
            ),
            on="#count",
            how="left",
            coalesce=True,
        )
        .rename({"#count": "nstep", "time/s": "timesec"})
    )

    dfheatingthermo = dfheatingthermo.with_columns(
        htot=pl.when(pl.col("Qdot") <= 1e-20).then(pl.lit(0.0)).otherwise(pl.col("Qdot"))
    )
    dfheatingthermo = dfheatingthermo.with_columns(pl.col("htot").cast(pl.Float64))

    # get nearest network time to each plotted time
    arr_networktimedays = dfheatingthermo["timesec"].to_numpy() / 86400
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
            .join(nuc_data.lazy(), on=("Z", "A"), how="inner")
            .with_columns([(pl.col("num_nuc") / pl.col("tau[s]")).alias("N_dot")])
            .with_columns([
                (pl.col("N_dot") * pl.col("Eneutrino[MeV]") * MeV_to_erg).alias("Qnu"),
                (pl.col("N_dot") * pl.col("Eelec[MeV]") * MeV_to_erg).alias("Qelec"),
                (pl.col("N_dot") * pl.col("Egamma[MeV]") * MeV_to_erg).alias("Qgamma"),
                (pl.col("N_dot") * pl.col("Q[MeV]") * MeV_to_erg).alias("Qtot"),
            ])
            .collect()
        )
        global_sums = pldf_all.select([
            pl.sum("Qnu").alias("abundweighted_nu"),
            pl.sum("Qelec").alias("abundweighted_elec"),
            pl.sum("Qgamma").alias("abundweighted_gamma"),
            pl.sum("Qtot").alias("abundweighted_Qdot"),
        ])
        global_sums_row = global_sums.row(0)  # nur ein Row-Objekt, da Summe

        decay_powers["abundweighted_nu"][plottimestep] = global_sums_row[0]
        decay_powers["abundweighted_elec"][plottimestep] = global_sums_row[1]
        decay_powers["abundweighted_gamma"][plottimestep] = global_sums_row[2]
        decay_powers["abundweighted_Qdot"][plottimestep] = global_sums_row[3]

        grouped = pldf_all.group_by(["A", "Z"]).agg([
            pl.sum("Qelec").alias("Qelec"),
            pl.sum("Qgamma").alias("Qgamma"),
            pl.sum("Qnu").alias("Qnu"),
        ])
        if nuclide_contrib:
            A_vals = grouped["A"].to_numpy()
            Z_vals = grouped["Z"].to_numpy()
            Qelec_vals = grouped["Qelec"].to_numpy()
            Qgamma_vals = grouped["Qgamma"].to_numpy()
            Qnu_vals = grouped["Qnu"].to_numpy()

            for A, Z, Qe, Qg, Qn in zip(A_vals, Z_vals, Qelec_vals, Qgamma_vals, Qnu_vals, strict=True):
                decay_powers[f"({A},{Z})_elec"][plottimestep] = Qe
                decay_powers[f"({A},{Z})_gam"][plottimestep] = Qg
                decay_powers[f"({A},{Z})_nu"][plottimestep] = Qn
    # if not np.all(np.diff(decay_powers["abundweighted_Qdot"]) <= 0.01 * decay_powers["abundweighted_Qdot"][0]):
    #     print(f"\nTraj {traj_ID} has inconsistent Qdot values. delete {Path(traj_root, str(traj_ID))} and rerun")
    #     # import shutil

    # dump to JSON
    if traj_json:
        decay_powers_json_copy = decay_powers.copy()

        for key in decay_powers_json_copy:
            val = decay_powers_json_copy[key]
            if isinstance(val, np.ndarray):
                decay_powers_json_copy[key] = val.tolist()

        output_path = Path(f"json/decay_powers_{traj_ID}.json")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(decay_powers_json_copy, f)
        #     # shutil.rmtree(Path(traj_root, str(traj_ID)))
    return decay_powers


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Comparison to constant beta decay splitup factors."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)
    import pandas as pd

    nuc_dataset = "Hotokezaka" if args.nucdata == "hoto" else "ENSDF"

    if args.npz:
        npz_dict = np.load(args.npz)
        npz_idcs = npz_dict["idx"]
        npz_types = npz_dict["state"]
        Ye_bins = [("all", 0.0, float("inf"))]
    else:
        Ye_bins = [
            ("all", 0.0, float("inf")),
            ("low", 0.0, args.yemax / 3),
            ("mid", args.yemax / 3, args.yemax * 2 / 3),
            ("high", args.yemax * 2 / 3, args.yemax / 3),
        ]

    # get beta decay data
    nuc_data = get_nuc_data(nuc_dataset)
    if args.trajjson:
        traj_json_dir = f"json_{args.nucdata}"
        if not Path(traj_json_dir).exists():
            Path(traj_json_dir).mkdir(parents=True)
            print(f"Created directory '{traj_json_dir}'.")
        else:
            print(f"'{traj_json_dir}' already exists.")
    assert nuc_data.height == nuc_data.unique(("Z", "A")).height

    # set timesteps logarithmically
    log_t_compar_min_s = np.log10(args.tmin)
    log_t_compar_max_s = np.log10(args.tmax)
    arr_t_day = 10 ** (np.linspace(log_t_compar_min_s, log_t_compar_max_s, args.nsteps, endpoint=True))

    # get masses of trajectories
    colnames = None
    skiprows = 0
    with Path(args.trajectoryroot, "summary-all.dat").open("r", encoding="utf-8") as f:
        possible_header_line = f.readline()
        if possible_header_line.startswith("#"):
            colnames = possible_header_line[1:].split()
            skiprows = 1

    if colnames is None:
        msg = "ERROR: No header found in summary-all.dat. Please check the file format."
        raise ValueError(msg)

    traj_summ_data = pl.from_pandas(
        pd.read_csv(
            Path(args.trajectoryroot, "summary-all.dat"),
            delimiter=r"\s+",
            skiprows=skiprows,
            names=colnames,
            dtype_backend="pyarrow",
        )
    ).filter(pl.any_horizontal(pl.col("Ye").is_between(Ye_lower, Ye_upper) for _, Ye_lower, Ye_upper in Ye_bins))

    print(traj_summ_data)

    traj_ids = traj_summ_data["Id"].to_list()

    traj_masses_g = {trajid: mass * M_sol_cgs for trajid, mass in traj_summ_data[["Id", "Mass"]].to_numpy()}

    import tqdm.rich
    from tqdm import TqdmExperimentalWarning
    from tqdm.contrib.concurrent import process_map

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    alltraj_decay_powers: list[dict[str, npt.NDArray[np.floating]]] = process_map(
        partial(
            process_trajectory, nuc_data, args.trajectoryroot, traj_masses_g, arr_t_day, args.nuclides, args.trajjson
        ),
        traj_ids,
        chunksize=3,
        desc="Processing trajectories",
        unit="traj",
        smoothing=0.0,
        tqdm_class=tqdm.rich.tqdm,
        max_workers=1,
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
                traj_ids if state == "any" else list(set(traj_ids) & set(npz_idcs[np.where(npz_types == state)]))
            )
            print(f" {len(selected_traj_ids)} trajectories selected")
            if len(selected_traj_ids) == 0:
                print(f"Warning! No trajectories found for eject state {state}")
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
        }

        assert isinstance(decay_powers["abundweighted_gamma"], np.ndarray)
        assert isinstance(decay_powers["abundweighted_elec"], np.ndarray)
        assert isinstance(decay_powers["abundweighted_nu"], np.ndarray)
        decay_powers["abundweighted_gammanuelec"] = (
            decay_powers["abundweighted_gamma"] + decay_powers["abundweighted_nu"] + decay_powers["abundweighted_elec"]
        )

        if args.json:
            # dump to JSON
            decay_powers_json_copy = decay_powers.copy()

            for key in decay_powers_json_copy:
                val = decay_powers_json_copy[key]
                if isinstance(val, np.ndarray):
                    decay_powers_json_copy[key] = val.tolist()

            output_path = Path(f"decay_powers_{label}.json")
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(decay_powers_json_copy, f)

        fig, axes = plt.subplots(
            nrows=2, ncols=1, figsize=(6, 10), tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0}
        )
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
        # ax1.plot(arr_t_day, decay_powers["hbeta"], linestyle="-", label=f"Traj {labelfull} hbeta")
        # ax1.plot(arr_t_day, decay_powers["htot"], linestyle="-", label=f"Traj {labelfull} htot")
        ax1.plot(arr_t_day, decay_powers["Qdot"], linestyle="-", linewidth=3, label=f"Traj {labelfull} Qdot")
        # ax1.plot(arr_t_day, decay_powers["abundweighted_gamma"], linestyle="-", label=f"Traj {labelfull} abund -> gamma")
        # ax1.plot(arr_t_day, decay_powers["abundweighted_elec"], linestyle="-", label=f"Traj {labelfull} abund -> elec")
        # ax1.plot(arr_t_day, decay_powers["abundweighted_nu"], linestyle="-", label=f"Traj {labelfull} abund -> nu")
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
        ax1.set_ylabel("energy release rate (erg/s)")
        ax1.set_yscale("log")
        ax1.legend()

        for ax in axes:
            ax.legend()
            ax.set_xlabel("time (days)")
            ax.set_xscale("log")

        fig.savefig(f"beta_release_ratios_tot_{nuc_dataset}_Ye{label}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    mp.freeze_support()
    main()
