# PYTHON_ARGCOMPLETE_OK

# script to plot the total energy release rate from a set of trajectories (GSI network file format)
# can either plot single trajectories or the total set
# can also perform a comparison to common fitting formulae (e.g. Rosswog & Korobikin or Lippuner & Roberts)

import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path
import math
import os
import pdb
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.spatial import distance
from itertools import product
import warnings

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import tqdm.rich
from tqdm import TqdmExperimentalWarning
from tqdm.contrib.concurrent import process_map

import artistools as at
from artistools.configuration import get_config
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path

days_to_s = 86400
s_to_days = 1 / days_to_s
M_sol_cgs = 1.989e33
c_kms = 299792.458


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-trajectoryroot",
        "-trajroot",
        default=None,
        help="Path to nuclear network trajectory folder, if abundances are required",
    )

    parser.add_argument("-rosswogdata", default=None, help="Path to heating rate fitting data from Rosswog+2022 paper")

    parser.add_argument("-lippunerdata", default=None, help="Path to heating rate fitting data from Rosswog+2022 paper")

    parser.add_argument(
        "-singletrajectory",
        default=False,
        help="Compare the heating rate fitting formulae to single trajectories only for better comparison",
    )

    parser.add_argument(
        "-plotmodel", default=False, help="Plot the total energy release rate for a whole model / set of trajectories"
    )

    parser.add_argument(
        "-timemin", default=0.1, help="Minimum time for comparison between fit and network energy release rate"
    )

    parser.add_argument(
        "-timemax", default=10.0, help="Maximum time for comparison between fit and network energy release rate"
    )

    parser.add_argument("-timesteps", default=20, help="Number of time steps for the plots")

    parser.add_argument(
        "-scatterintegral",
        default=False,
        help="Make scatter plot for integrated energy release rate for all trajectories",
    )

    parser.add_argument(
        "-comparehomsphere", default=False, help="Compare the total model energy release to an equal-mass homogeneous sphere"
    )


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def integrate_arr(y, x):
    assert len(y) == len(x), "Arrays differ in length!"
    integral = y[0] * x[0]
    for i in range(1, len(x)):
        integral += y[i] * (x[i] - x[i - 1])
    return integral


def idx_of_nth_largest_val(data_list, n):
    if n == 0:
        return np.argmax(data_list)
    else:
        x = sorted(data_list)[-n]  # get nth largest element first
        return np.where(data_list == x)[0]


def process_traj_list(
    trajroot, model_type, t_days_min, t_days_max, numb_pts, rosswog_lib, lippuner_lib
) -> None:
    # compare fit heating rate to network data for Y_e of 0.05 up to 0.5 in 0.05 bins
    # creates a 5x2 plot, each subplot for the different Y_e values

    # step 1) read trajectory overview / summary-all.dat and Lippuner data
    if model_type == "sfho":
        colnames0 = ["Id", "Mass", "time", "t9", "Ye", "entropy", "n/seed", "tau", "radius", "velocity", "angle"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"), delimiter=r"\s+", skiprows=1, names=colnames0, dtype_backend="pyarrow"
        )
    elif model_type == "e2e":
        colnames0 = ["Id", "Mass", "Ye", "velocity"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"), delimiter=r"\s+", skiprows=1, names=colnames0, dtype_backend="pyarrow"
        )

    target_Y_e_values = np.linspace(0.05, 0.5, 10)  # values for the plots
    t_days_lists = []  # list of lists for plot time ranges (0.1 d up to 10 d roughly)
    ERR_network_lists = []  # list of lists for the plot energy release rates as read from the network
    ERR_Rosswog_lists = []  # list of lists for the plot energy release rates as read from the network
    ERR_Lippuner_lists = []  # list of lists for the plot energy release rates as read from the network

    # step 2) fill lists with data

    # time range
    log_t_min = math.log10(t_days_min)
    log_t_max = math.log10(t_days_max)
    log_t_arr = np.linspace(log_t_min, log_t_max, numb_pts)
    t_days_fixed_list = [10**log_t for log_t in log_t_arr]

    # for each target Y_e, read the trajectory which as Y_e closest to it
    for target_Y_e in target_Y_e_values:
        # 2.1) extract the network data

        # get trajectory with closest Y_e0 to target value
        red_traj_data = traj_summ_data.iloc[(traj_summ_data["Ye"] - target_Y_e).abs().argsort()[:1]]
        traj_ID = red_traj_data["Id"].item()
        print(f"Using trajectory {traj_ID} for Y_e = {target_Y_e}")

        # now read individual trajectory
        dfheatingthermo = (
            pl.from_pandas(
                pd.read_csv(
                    get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/heating.dat"),
                    sep=r"\s+",
                    usecols=["#count", "hbeta", "htot"],
                )
            )
            .join(
                pl.from_pandas(
                    pd.read_csv(
                        get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/energy_thermo.dat"),
                        sep=r"\s+",
                        usecols=["#count", "time/s", "Qdot", "Ye"],
                    )
                ),
                on="#count",
                how="left",
                coalesce=True,
            )
            .rename({"#count": "nstep", "time/s": "timesec"})
        ).to_pandas()

        # obtain reduced dataframe limited to the plotting time range
        red_heat_DF = dfheatingthermo.loc[
            (dfheatingthermo["timesec"] >= t_days_min * days_to_s)
            & (dfheatingthermo["timesec"] <= t_days_max * days_to_s)
        ]

        # obtain specific heating rate, includes now neutrinos
        t_days_lists.append([t / 86400 for t in red_heat_DF["timesec"].to_numpy()])
        q_dot_arr = red_heat_DF["Qdot"].to_numpy()

        Q_dot_arr = [float(qdot) for qdot in q_dot_arr]

        ERR_network_lists.append(Q_dot_arr)

        if rosswog_lib is not None:
            # 2.2) obtain Rosswog & Korobkin heating rate
            # need to get velocity and Y_e0 of the trajectory

            v_traj = traj_summ_data.loc[traj_summ_data["Id"] == traj_ID]["velocity"].item()  # in km per s
            Y_e_traj = traj_summ_data.loc[traj_summ_data["Id"] == traj_ID]["Ye"].item()

            # now round both velocity and Y_e to next steps from Rosswog's library
            v_Rosswog = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            Y_e_Rosswog = np.linspace(0.05, 0.5, 10)

            v_closest = min(v_Rosswog, key=lambda x: abs(x - v_traj / c_kms))
            Y_e_closest = min(Y_e_Rosswog, key=lambda x: abs(x - Y_e_traj))

            colnames = ["time_days", "qdot"]
            Rosswog_heating_df = pd.read_csv(
                f"{rosswog_lib}/Heating_M0.05_v{v_closest:,.2f}_Ye{Y_e_closest:,.2f}.dat",
                delimiter=r"\s+",
                names=colnames,
                skiprows=1,
            )

            Q_dot_Rosswog = np.zeros(len(t_days_fixed_list))

            for t_idx, t_d in enumerate(t_days_fixed_list):
                q_dot_interpol = interp1d(
                    Rosswog_heating_df.time_days.to_numpy(), Rosswog_heating_df.qdot.to_numpy(), kind="linear"
                )(t_d)
                Q_dot_Rosswog[t_idx] = 10**q_dot_interpol
            ERR_Rosswog_lists.append(Q_dot_Rosswog)

        if lippuner_lib is not None:
            # 2.3) obtain Lippuner & Roberts heating rate

            # ASSUME that Oli's e2e model used symmetric fission with zero free neutrons
            # L&R define their initial point in time by a temperature of 6 GK
            # ATTENTION: linearly interpolates between fit parameters as the expansion timescales
            # are far off the provided parameter range

            # read data
            colnames = [str(i) for i in range(1, 39)]
            lipp_data = pd.read_csv(
                Path(lippuner_lib, "hires_sym0_results"),
                delimiter=r"\s+",
                skiprows=41,
                names=colnames,
                dtype_backend="pyarrow",
            )

            dfevoldat = pd.read_csv(
                get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/evol.dat"),
                sep=r"\s+",
                usecols=["#count", "time/s", "T9", "rho(g/cc)", "Ye", "S[k_b]"],
            )
            dfevoldat = dfevoldat.drop(0,inplace=False)
            dfevoldat = dfevoldat.drop_duplicates(subset=['time/s'],keep='first')
            dfevoldat = dfevoldat.reset_index(drop=True)

            T_init = 6.0  # temperature of definition for the "initial" moment in the paper (in GK)
            # closest row
            NSE_data = dfevoldat.loc[dfevoldat["T9"] >= T_init]
            if len(NSE_data) > 1:
                closest_row = NSE_data.iloc[-1]
                n2_closest_row = NSE_data.iloc[-2]
            else:
                closest_row = dfevoldat.iloc[1]
                n2_closest_row = dfevoldat.iloc[0]
            # calculate expansion time scale in ms for the Lippuner data file
            tau_0 = (
                1000
                * closest_row["rho(g/cc)"].item()
                / (
                    (closest_row["rho(g/cc)"].item() - n2_closest_row["rho(g/cc)"].item())
                    / (n2_closest_row["time/s"].item() - closest_row["time/s"].item())
                )
            )

            Y_e_0 = closest_row["Ye"].item()
            s_0 = closest_row["S[k_b]"].item()
            rho_0 = closest_row["rho(g/cc)"].item()

            # now read the Lippuner data
            Y_e_0_fit_data = lipp_data["1"].unique()
            s_0_fit_data = lipp_data["2"].unique()
            tau_0_fit_data = lipp_data["3"].unique()
            rho_0_fit_data = lipp_data["4"].unique()

            # normalize the fit params
            Y_e_0_fit_data_max = max(Y_e_0_fit_data)
            s_0_fit_data_max = max(s_0_fit_data)
            tau_0_fit_data_max = max(tau_0_fit_data)
            rho_0_fit_data_max = max(rho_0_fit_data)

            all_fit_points = lipp_data[["1", "2", "3", "4"]].values.tolist()

            all_fit_points_norm = [
                [
                    quadruple[0] / Y_e_0_fit_data_max,
                    quadruple[1] / s_0_fit_data_max,
                    quadruple[2] / tau_0_fit_data_max,
                    quadruple[3] / rho_0_fit_data_max,
                ]
                for quadruple in lipp_data[["1", "2", "3", "4"]].values.tolist()
            ]
            query_data_point = [
                Y_e_0 / Y_e_0_fit_data_max,
                s_0 / s_0_fit_data_max,
                tau_0 / tau_0_fit_data_max,
                rho_0 / rho_0_fit_data_max,
            ]
            param_distances = -distance.cdist([query_data_point], np.array(all_fit_points_norm), metric="cityblock")
            # breakpoint()
            
            idx_in_tuple_list = idx_of_nth_largest_val(list(param_distances)[0], 0)
            fit_point = all_fit_points[int(idx_in_tuple_list)]

            # get absolute values again for calculating the actual Lippuner heating rate
            Y_e_0_closest = fit_point[0]
            s_0_closest = fit_point[1]
            tau_0_closest = fit_point[2]
            rho_0_closest = fit_point[3]

            row_to_take = lipp_data.loc[
                (lipp_data["1"] == Y_e_0_closest)
                & (lipp_data["2"] == s_0_closest)
                & (lipp_data["3"] == tau_0_closest)
                & (lipp_data["4"] == rho_0_closest)
            ]
            """
            print(f"Counter {counter} idx_in_tuple_list {idx_in_tuple_list}")
            print(f"Trajectory params: Y_e_0 = {Y_e_0}, s_0 = {s_0}, tau_0 = {tau_0}, rho_0 = {rho_0}")
            print(
                f"Fit data for point: Y_e_0 = {Y_e_0_closest}, s_0 = {s_0_closest}, tau_0 = {tau_0_closest}, rho_0 = {rho_0_closest}"
            )
            """
            A = row_to_take["6"].item()
            alpha = -row_to_take["7"].item()
            B_1 = row_to_take["8"].item()
            beta_1 = max(row_to_take["9"].item(), 1e-50)
            B_2 = row_to_take["10"].item()
            beta_2 = max(row_to_take["11"].item(), 1e-50)
            B_3 = row_to_take["12"].item()
            beta_3 = max(row_to_take["13"].item(), 1e-50)

            Q_dot_Lippuner = np.zeros(len(t_days_fixed_list))
            for t_idx, t_d in enumerate(t_days_fixed_list):
                q_dot = (
                    A * t_d**alpha
                    + B_1 * np.exp(-t_d / beta_1)
                    + B_2 * np.exp(-t_d / beta_2)
                    + B_3 * np.exp(-t_d / beta_3)
                )
                Q_dot_Lippuner[t_idx] = q_dot
            ERR_Lippuner_lists.append(Q_dot_Lippuner)

    # step 3) plot the results
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(6, 10))
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.6)

    for i in range(5):
        axs[i, 0].set(ylabel=r"$\dot{q}$ in erg / g s")
        for j in range(2):
            axs[i, j].plot(t_days_lists[2 * i + j], ERR_network_lists[2 * i + j], label="GSINet")
            if rosswog_lib is not None:
                axs[i, j].plot(t_days_fixed_list, ERR_Rosswog_lists[2 * i + j], label="RK24")
            if lippuner_lib is not None:
                axs[i, j].plot(t_days_fixed_list, ERR_Lippuner_lists[2 * i + j], label="LR15")
            axs[i, j].set_title(rf"$Y_e$ = {target_Y_e_values[2 * i + j]:.2f}")
            axs[i, j].grid(color="k", linestyle="--", linewidth=0.5)
            axs[i, j].set_xscale("log")
            axs[i, j].set_yscale("log")
            axs[i, j].yaxis.set_tick_params(labelleft=True)
    axs[4, 0].set(xlabel="time in days")
    axs[4, 1].set(xlabel="time in days")
    handles, labels = axs[4, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")
    plt.savefig("Q_dot_tot_single_trajs.pdf")
    plt.clf()


def check_single_trajectories(
    trajroot, model_type, t_days_min, t_days_max, numb_pts, rosswog_lib, lippuner_lib
) -> None:


def plot_model_release(
    trajroot, model_type, t_days_min, t_days_max, numb_pts, rosswog_lib, lippuner_lib, compare_hom_sphere
) -> None:
    # step 1) read trajectory overview / summary-all.dat and Lippuner data
    if model_type == "sfho":
        colnames0 = ["Id", "Mass", "time", "t9", "Ye", "entropy", "n/seed", "tau", "radius", "velocity", "angle"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"), delimiter=r"\s+", skiprows=1, names=colnames0, dtype_backend="pyarrow"
        )
    elif model_type == "e2e":
        colnames0 = ["Id", "Mass", "Ye", "velocity"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"), delimiter=r"\s+", skiprows=1, names=colnames0, dtype_backend="pyarrow"
        )

    ERR_network = np.zeros(numb_pts)
    ERR_Rosswog = np.zeros(numb_pts)
    ERR_Lippuner = np.zeros(numb_pts)

    traj_id_list = traj_summ_data["Id"].to_numpy()
    traj_ye0_list = traj_summ_data["Ye"].to_numpy()
    traj_M_list = traj_summ_data["Mass"].to_numpy()
    M_ej = sum(traj_M_list) * M_sol_cgs
    numb_trajs = len(traj_M_list)
    traj_s0_list = np.zeros(numb_trajs)
    traj_tau0_list = np.zeros(numb_trajs)
    traj_rho0_list = np.zeros(numb_trajs)
    
    # step 2) fill lists with data

    # time range
    log_t_min = math.log10(t_days_min)
    log_t_max = math.log10(t_days_max)
    log_t_arr = np.linspace(log_t_min, log_t_max, numb_pts)
    t_days_fixed_list = [10**log_t for log_t in log_t_arr]

    # for each target Y_e, read the trajectory which as Y_e closest to it
    for idx0, traj_ID in enumerate(traj_id_list):
        # 2.1) extract the network data
        M_traj = traj_M_list[idx0] * M_sol_cgs

        # now read individual trajectory
        dfheatingthermo = (
            pl.from_pandas(
                pd.read_csv(
                    get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/heating.dat"),
                    sep=r"\s+",
                    usecols=["#count", "hbeta", "htot"],
                )
            )
            .join(
                pl.from_pandas(
                    pd.read_csv(
                        get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/energy_thermo.dat"),
                        sep=r"\s+",
                        usecols=["#count", "time/s", "Qdot", "Ye"],
                    )
                ),
                on="#count",
                how="left",
                coalesce=True,
            )
            .rename({"#count": "nstep", "time/s": "timesec"})
        ).to_pandas()

        # obtain reduced dataframe limited to the plotting time range
        red_heat_DF = dfheatingthermo.loc[
            (dfheatingthermo["timesec"] >= 0.5 * t_days_min * days_to_s)
            & (dfheatingthermo["timesec"] <= 2 * t_days_max * days_to_s)
        ]

        # obtain specific heating rate, includes now neutrinos
        q_dot_arr = red_heat_DF["Qdot"].to_numpy()

        for t_idx, t_d in enumerate(t_days_fixed_list):
            q_dot_interpol = interp1d(
                (red_heat_DF["timesec"]).to_numpy(),
                (red_heat_DF["Qdot"]).to_numpy(),
                kind="linear",
            )(t_d * days_to_s)
            ERR_network[t_idx] += q_dot_interpol * M_traj

        if rosswog_lib is not None:
            # 2.2) obtain Rosswog & Korobkin heating rate
            # need to get velocity and Y_e0 of the trajectory

            v_traj = traj_summ_data.loc[traj_summ_data["Id"] == traj_ID]["velocity"].item()  # in km per s
            Y_e_traj = traj_summ_data.loc[traj_summ_data["Id"] == traj_ID]["Ye"].item()

            # now round both velocity and Y_e to next steps from Rosswog's library
            v_Rosswog = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            Y_e_Rosswog = np.linspace(0.05, 0.5, 10)

            v_closest = min(v_Rosswog, key=lambda x: abs(x - v_traj / c_kms))
            Y_e_closest = min(Y_e_Rosswog, key=lambda x: abs(x - Y_e_traj))

            colnames = ["time_days", "qdot"]
            Rosswog_heating_df = pd.read_csv(
                f"{rosswog_lib}/Heating_M0.05_v{v_closest:,.2f}_Ye{Y_e_closest:,.2f}.dat",
                delimiter=r"\s+",
                names=colnames,
                skiprows=1,
            )

            for t_idx, t_d in enumerate(t_days_fixed_list):
                q_dot_interpol = interp1d(
                    Rosswog_heating_df.time_days.to_numpy(), Rosswog_heating_df.qdot.to_numpy(), kind="linear"
                )(t_d)
                ERR_Rosswog[t_idx] += 10**q_dot_interpol * M_traj

        if lippuner_lib is not None:
            # 2.3) obtain Lippuner & Roberts heating rate

            # ASSUME that Oli's e2e model used symmetric fission with zero free neutrons
            # L&R define their initial point in time by a temperature of 6 GK
            # ATTENTION: linearly interpolates between fit parameters as the expansion timescales
            # are far off the provided parameter range

            # read data
            colnames = [str(i) for i in range(1, 39)]
            lipp_data = pd.read_csv(
                Path(lippuner_lib, "hires_sym0_results"),
                delimiter=r"\s+",
                skiprows=41,
                names=colnames,
                dtype_backend="pyarrow",
            )

            dfevoldat = pd.read_csv(
                get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/evol.dat"),
                sep=r"\s+",
                usecols=["#count", "time/s", "T9", "rho(g/cc)", "Ye", "S[k_b]"],
            )
            dfevoldat = dfevoldat.drop(0,inplace=False)
            dfevoldat = dfevoldat.drop_duplicates(subset=['time/s'],keep='first')
            dfevoldat = dfevoldat.reset_index(drop=True)

            T_init = 6.0  # temperature of definition for the "initial" moment in the paper (in GK)
            # closest row
            NSE_data = dfevoldat.loc[dfevoldat["T9"] >= T_init]
            if len(NSE_data) > 1:
                closest_row = NSE_data.iloc[-1]
                n2_closest_row = NSE_data.iloc[-2]
            else:
                closest_row = dfevoldat.iloc[1]
                n2_closest_row = dfevoldat.iloc[0]
            # calculate expansion time scale in ms for the Lippuner data file
            tau_0 = (
                1000
                * closest_row["rho(g/cc)"].item()
                / (
                    (closest_row["rho(g/cc)"].item() - n2_closest_row["rho(g/cc)"].item())
                    / (n2_closest_row["time/s"].item() - closest_row["time/s"].item())
                )
            )

            Y_e_0 = closest_row["Ye"].item()
            s_0 = closest_row["S[k_b]"].item()
            rho_0 = closest_row["rho(g/cc)"].item()
            
            traj_s0_list[idx0] = s_0
            traj_tau0_list[idx0] = tau_0
            traj_rho0_list[idx0] = rho_0

            # now read the Lippuner data
            Y_e_0_fit_data = lipp_data["1"].unique()
            s_0_fit_data = lipp_data["2"].unique()
            tau_0_fit_data = lipp_data["3"].unique()
            rho_0_fit_data = lipp_data["4"].unique()

            # normalize the fit params
            Y_e_0_fit_data_max = max(Y_e_0_fit_data)
            s_0_fit_data_max = max(s_0_fit_data)
            tau_0_fit_data_max = max(tau_0_fit_data)
            rho_0_fit_data_max = max(rho_0_fit_data)

            all_fit_points = lipp_data[["1", "2", "3", "4"]].values.tolist()

            all_fit_points_norm = [
                [
                    quadruple[0] / Y_e_0_fit_data_max,
                    quadruple[1] / s_0_fit_data_max,
                    quadruple[2] / tau_0_fit_data_max,
                    quadruple[3] / rho_0_fit_data_max,
                ]
                for quadruple in lipp_data[["1", "2", "3", "4"]].values.tolist()
            ]
            query_data_point = [
                Y_e_0 / Y_e_0_fit_data_max,
                s_0 / s_0_fit_data_max,
                tau_0 / tau_0_fit_data_max,
                rho_0 / rho_0_fit_data_max,
            ]
            param_distances = -distance.cdist([query_data_point], np.array(all_fit_points_norm), metric="cityblock")
            # breakpoint()
            fit_param_pt_found = False
            counter = 0
            while fit_param_pt_found == False:
                idx_in_tuple_list = idx_of_nth_largest_val(list(param_distances)[0], counter)
                try:
                    fit_point = all_fit_points[int(idx_in_tuple_list)]
                except TypeError:
                    breakpoint()

                # get absolute values again for calculating the actual Lippuner heating rate
                Y_e_0_closest = fit_point[0]
                s_0_closest = fit_point[1]
                tau_0_closest = fit_point[2]
                rho_0_closest = fit_point[3]

                row_to_take = lipp_data.loc[
                    (lipp_data["1"] == Y_e_0_closest)
                    & (lipp_data["2"] == s_0_closest)
                    & (lipp_data["3"] == tau_0_closest)
                    & (lipp_data["4"] == rho_0_closest)
                ]
                """
                print(f"Counter {counter} idx_in_tuple_list {idx_in_tuple_list}")
                print(f"Trajectory params: Y_e_0 = {Y_e_0}, s_0 = {s_0}, tau_0 = {tau_0}, rho_0 = {rho_0}")
                print(
                    f"Fit data for point: Y_e_0 = {Y_e_0_closest}, s_0 = {s_0_closest}, tau_0 = {tau_0_closest}, rho_0 = {rho_0_closest}"
                )
                """
                A = row_to_take["6"].item()
                alpha = row_to_take["7"].item()
                B_1 = row_to_take["8"].item()
                beta_1 = row_to_take["9"].item()
                B_2 = row_to_take["10"].item()
                beta_2 = row_to_take["11"].item()
                B_3 = row_to_take["12"].item()
                beta_3 = row_to_take["13"].item()

                if beta_1 == 0 or beta_2 == 0 or beta_3 == 0:
                    # print(f"beta_1: {beta_1} beta_2: {beta_2} beta_3: {beta_3}")
                    counter += 1
                else:
                    Q_dot_Lippuner = np.zeros(len(t_days_fixed_list))
                    for t_idx, t_d in enumerate(t_days_fixed_list):
                        q_dot = (
                            A * t_d**alpha
                            + B_1 * np.exp(-t_d / beta_1)
                            + B_2 * np.exp(-t_d / beta_2)
                            + B_3 * np.exp(-t_d / beta_3)
                        )
                        ERR_Lippuner[t_idx] += q_dot * M_traj
                    fit_param_pt_found = True

    # step 3) compare to homogeneous sphere

    ERR_Rosswog_HS = np.zeros(numb_pts)
    ERR_Lippuner_HS = np.zeros(numb_pts)

    if compare_hom_sphere:
        if rosswog_lib is not None:
            # 3.1) compare to Rosswog data. Need ejecta mean velocity and Y_e0 for that
            v_ej_mean = (sum((traj_summ_data['Mass']*traj_summ_data['velocity']).to_numpy())) / (sum(traj_summ_data['Mass'].to_numpy()))
            Y_e0_mean = (sum((traj_summ_data['Mass']*traj_summ_data['Ye']).to_numpy())) / (sum(traj_summ_data['Mass'].to_numpy()))
            v_Rosswog = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            Y_e_Rosswog = np.linspace(0.05, 0.5, 10)

            v_closest = min(v_Rosswog, key=lambda x: abs(x - v_ej_mean / c_kms))
            Y_e_closest = min(Y_e_Rosswog, key=lambda x: abs(x - Y_e0_mean))

            colnames = ["time_days", "qdot"]
            Rosswog_heating_df = pd.read_csv(
                f"{rosswog_lib}/Heating_M0.05_v{v_closest:,.2f}_Ye{Y_e_closest:,.2f}.dat",
                delimiter=r"\s+",
                names=colnames,
                skiprows=1,
            )

            for t_idx, t_d in enumerate(t_days_fixed_list):
                q_dot_interpol = interp1d(
                    Rosswog_heating_df.time_days.to_numpy(), Rosswog_heating_df.qdot.to_numpy(), kind="linear"
                )(t_d)
                ERR_Rosswog[t_idx] = 10**q_dot_interpol * M_ej
        if lippuner_lib is not None:
            s0_mean = (traj_s0_list * (traj_summ_data['Mass']*M_sol_cgs).to_numpy()) / M_ej
            Ye0_mean = (traj_ye0_list * (traj_summ_data['Mass']*M_sol_cgs).to_numpy()) / M_ej
            tau0_mean = (traj_tau0_list * (traj_summ_data['Mass']*M_sol_cgs).to_numpy()) / M_ej
            rho0_mean = (traj_rho0_list * (traj_summ_data['Mass']*M_sol_cgs).to_numpy()) / M_ej

            colnames = [str(i) for i in range(1, 39)]
            lipp_data = pd.read_csv(
                Path(lippuner_lib, "hires_sym0_results"),
                delimiter=r"\s+",
                skiprows=41,
                names=colnames,
                dtype_backend="pyarrow",
            )

            # now read the Lippuner data
            Y_e_0_fit_data = lipp_data["1"].unique()
            s_0_fit_data = lipp_data["2"].unique()
            tau_0_fit_data = lipp_data["3"].unique()
            rho_0_fit_data = lipp_data["4"].unique()

            # normalize the fit params
            Y_e_0_fit_data_max = max(Y_e_0_fit_data)
            s_0_fit_data_max = max(s_0_fit_data)
            tau_0_fit_data_max = max(tau_0_fit_data)
            rho_0_fit_data_max = max(rho_0_fit_data)

            all_fit_points = lipp_data[["1", "2", "3", "4"]].values.tolist()

            all_fit_points_norm = [
                [
                    quadruple[0] / Y_e_0_fit_data_max,
                    quadruple[1] / s_0_fit_data_max,
                    quadruple[2] / tau_0_fit_data_max,
                    quadruple[3] / rho_0_fit_data_max,
                ]
                for quadruple in lipp_data[["1", "2", "3", "4"]].values.tolist()
            ]
            query_data_point = [
                Ye0_mean / Y_e_0_fit_data_max,
                s0_mean / s_0_fit_data_max,
                tau0_mean / tau_0_fit_data_max,
                rho0_mean / rho_0_fit_data_max,
            ]
            param_distances = -distance.cdist([query_data_point], np.array(all_fit_points_norm), metric="cityblock")
            # breakpoint()
            fit_param_pt_found = False
            counter = 0
            while fit_param_pt_found == False:
                idx_in_tuple_list = idx_of_nth_largest_val(list(param_distances)[0], counter)
                fit_point = all_fit_points[int(idx_in_tuple_list)]

                # get absolute values again for calculating the actual Lippuner heating rate
                Y_e_0_closest = fit_point[0]
                s_0_closest = fit_point[1]
                tau_0_closest = fit_point[2]
                rho_0_closest = fit_point[3]

                row_to_take = lipp_data.loc[
                    (lipp_data["1"] == Y_e_0_closest)
                    & (lipp_data["2"] == s_0_closest)
                    & (lipp_data["3"] == tau_0_closest)
                    & (lipp_data["4"] == rho_0_closest)
                ]
                
                A = row_to_take["6"].item()
                alpha = row_to_take["7"].item()
                B_1 = row_to_take["8"].item()
                beta_1 = row_to_take["9"].item()
                B_2 = row_to_take["10"].item()
                beta_2 = row_to_take["11"].item()
                B_3 = row_to_take["12"].item()
                beta_3 = row_to_take["13"].item()

                if beta_1 == 0 or beta_2 == 0 or beta_3 == 0:
                    # print(f"beta_1: {beta_1} beta_2: {beta_2} beta_3: {beta_3}")
                    counter += 1
                else:
                    for t_idx, t_d in enumerate(t_days_fixed_list):
                        q_dot = (
                            A * t_d**alpha
                            + B_1 * np.exp(-t_d / beta_1)
                            + B_2 * np.exp(-t_d / beta_2)
                            + B_3 * np.exp(-t_d / beta_3)
                        )
                        ERR_Lippuner_HS[t_idx] = q_dot * M_traj
                    fit_param_pt_found = True
            
    # step 4) plot the results
    plt.plot(t_days_fixed_list, ERR_network, label="GSINet")
    if rosswog_lib is not None:
        plt.plot(t_days_fixed_list, ERR_Rosswog, label="RK24")
        if compare_hom_sphere:
            plt.plot(t_days_fixed_list, ERR_Rosswog_HS, label="RK24 HS")
    if lippuner_lib is not None:
        plt.plot(t_days_fixed_list, ERR_Lippuner, label="LR15")
        if compare_hom_sphere:
            plt.plot(t_days_fixed_list, ERR_Lippuner_HS, label="LR15 HS")
    plt.grid(color="k", linestyle="--", linewidth=0.5)
    plt.xlabel("time in days")
    plt.ylabel(r"$\dot{Q}_{tot}$ in erg / s")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("Q_dot_tot.pdf")
    plt.clf()


def scatter_integrated_energy_release(trajroot, model_type, t_days_min, t_days_max, RosswogLib_traj) -> None:
    if model_type == "sfho":
        colnames0 = ["Id", "Mass", "time", "t9", "Ye", "entropy", "n/seed", "tau", "radius", "velocity", "angle"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"), delimiter=r"\s+", skiprows=1, names=colnames0, dtype_backend="pyarrow"
        )
    elif model_type == "e2e":
        colnames0 = ["Id", "Mass", "Ye", "velocity"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"), delimiter=r"\s+", skiprows=1, names=colnames0, dtype_backend="pyarrow"
        )

    id_list = traj_summ_data["Id"].to_numpy()
    ye_list = traj_summ_data["Ye"].to_numpy()
    for idx, traj_id in enumerate(id_list):
        Y_e = ye_list[idx]

        # get integrated energy release
        integrated_network_err = 0
        integrated_rosswog_err = 0
        integrated_lippuner_err = 0

        dfheatingthermo = (
            pl.from_pandas(
                pd.read_csv(
                    get_tar_member_extracted_path(trajroot, traj_id, "./Run_rprocess/heating.dat"),
                    sep=r"\s+",
                    usecols=["#count", "hbeta", "htot"],
                )
            )
            .join(
                pl.from_pandas(
                    pd.read_csv(
                        get_tar_member_extracted_path(trajroot, traj_id, "./Run_rprocess/energy_thermo.dat"),
                        sep=r"\s+",
                        usecols=["#count", "time/s", "Qdot", "Ye"],
                    )
                ),
                on="#count",
                how="left",
                coalesce=True,
            )
            .rename({"#count": "nstep", "time/s": "timesec"})
        ).to_pandas()

        time_network = dfheatingthermo.loc[
            (dfheatingthermo["timesec"] >= t_days_min * days_to_s)
            & (dfheatingthermo["timesec"] <= t_days_max * days_to_s)
        ]["timesec"]
        err_network = dfheatingthermo.loc[
            (dfheatingthermo["timesec"] >= t_days_min * days_to_s)
            & (dfheatingthermo["timesec"] <= t_days_max * days_to_s)
        ]["htot"]

        err_rosswog = np.zeros(len(err_network))

        v_traj = traj_summ_data.loc[traj_summ_data["Id"] == traj_id]["velocity"].item()  # in km per s

        # now round both velocity and Y_e to next steps from Rosswog's library
        v_Rosswog = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        Y_e_Rosswog = np.linspace(0.05, 0.5, 10)

        # v_closest = min(v_Rosswog, key=lambda x: abs(x - v_traj / c_kms))
        v_closest = 0.1
        Y_e_closest = min(Y_e_Rosswog, key=lambda x: abs(x - Y_e))

        colnames = ["time_days", "qdot"]
        Rosswog_heating_df = pd.read_csv(
            f"{RosswogLib_traj}/Heating_M0.05_v{v_closest:,.2f}_Ye{Y_e_closest:,.2f}.dat",
            delimiter=r"\s+",
            names=colnames,
            skiprows=1,
        )
        for idx, t in enumerate(time_network.to_numpy()):
            q_dot_interpol = interp1d(
                Rosswog_heating_df.time_days.to_numpy(), Rosswog_heating_df.qdot.to_numpy(), kind="linear"
            )(t / 86400)
            err_rosswog[idx] = 10**q_dot_interpol
        err_network_arr = err_network.to_numpy()
        for idx, err_str in enumerate(err_network_arr):
            if isinstance(err_str, str):
                if "E" not in err_str:
                    err_network_arr[idx] = 0.0
                else:
                    err_network_arr[idx] = float(err_network_arr[idx])
                    if math.isnan(err_network_arr[idx]):
                        err_network_arr[idx] = 0.0
            elif math.isnan(err_network_arr[idx]):
                err_network_arr[idx] = 0.0
        integrated_network_err = integrate_arr(err_network_arr, x=time_network.to_numpy())
        integrated_rosswog_err = integrate_arr(err_rosswog, x=time_network.to_numpy())

        integral_ratio = integrated_rosswog_err / integrated_network_err

        plt.scatter(Y_e, integral_ratio, marker="x", color="b")
    plt.grid(color="k", linestyle="--", linewidth=0.5)
    plt.axhline(y=1.0, color="r", linestyle="-")
    plt.yscale("log")
    plt.xlabel("Y_e")
    plt.ylabel("Integrated Rosswog heating / network heating")
    plt.savefig(f"integrated_err_ratios_{t_days_min:.2f}d_{t_days_max:.2f}d.pdf")
    plt.clf()



def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs) -> None:
    """Comparison to constant beta decay splitup factors."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    model_type = (
        os.getcwd().split("/")[-1].split("_")[0]
    )  # assumes that trajectories are saved in a directory named e.g. "<MODEL INFO>_trajs"

    if not isinstance(args.timemin, float):
        args.timemin = float(args.timemin)

    if not isinstance(args.timemax, float):
        args.timemax = float(args.timemax)

    if args.singletrajectory:
        check_single_trajectories(
            args.trajectoryroot,
            model_type,
            args.timemin,
            args.timemax,
            args.timesteps,
            args.rosswogdata,
            args.lippunerdata,
        )

    if args.plotmodel:
        plot_model_release(args.trajectoryroot,
            model_type,
            args.timemin,
            args.timemax,
            args.timesteps,
            args.rosswogdata,
            args.lippunerdata,
            args.comparehomsphere)

    if args.scatterintegral:
        scatter_integrated_energy_release(
            args.trajectoryroot, model_type, float(args.timemin), float(args.timemax), args.rosswogdata
        )


if __name__ == "__main__":
    main()
