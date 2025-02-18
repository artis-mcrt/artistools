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
M_sol_cgs = 1.989e+33
c_kms = 299792.458

def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-trajectoryroot",
        "-trajroot",
        default=None,
        help="Path to nuclear network trajectory folder, if abundances are required",
    )

    parser.add_argument(
        "-rosswogdata",
        default=None,
        help="Path to heating rate fitting data from Rosswog+2022 paper",
    )

    parser.add_argument(
        "-lippunerdata",
        default=None,
        help="Path to heating rate fitting data from Rosswog+2022 paper",
    )

    parser.add_argument(
        "-singletrajectory",
        default=False,
        help="Compare the heating rate fitting formulae to single trajectories only for better comparison",
    )

    parser.add_argument(
        "-plotmodel",
        default=False,
        help="Plot the total energy release rate for a whole model / set of trajectories",
    )

    parser.add_argument(
        "-timemin",
        default=0.1,
        help="Minimum time for comparison between fit and network energy release rate",
    )

    parser.add_argument(
        "-timemax",
        default=10.0,
        help="Maximum time for comparison between fit and network energy release rate",
    )

    parser.add_argument(
        "-timesteps",
        default=20,
        help="Number of time steps for the plots",
    )

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



def check_single_trajectories(trajroot, model_type, t_days_min, t_days_max, numb_pts, RosswogLib_traj, Lippuner_traj) -> None:
    # compare fit heating rate to network data for Y_e of 0.05 up to 0.5 in 0.05 bins  
    # creates a 5x2 plot, each subplot for the different Y_e values

    # step 1) read trajectory overview / summary-all.dat and Lippuner data
    if model_type == "sfho":
        colnames0 = ["Id", "Mass", "time", "t9", "Ye", "entropy", "n/seed", "tau", "radius", "velocity", "angle"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"),
            delimiter=r"\s+",
            skiprows=1,
            names=colnames0,
            dtype_backend="pyarrow",
        )
    elif model_type == "e2e":
        colnames0 = ["Id", "Mass", "Ye", "velocity"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"),
            delimiter=r"\s+",
            skiprows=1,
            names=colnames0,
            dtype_backend="pyarrow",
        )

    colnames=[str(i) for i in range(1,39)]
    if Lippuner_traj is not None:
        lipp_data = pd.read_csv(
            Path(Lippuner_traj, "hires_sym0_results"),
            delimiter=r"\s+",
            skiprows=41,
            names=colnames,
            dtype_backend="pyarrow",
        )
    
    target_Y_e_values = np.linspace(0.05,0.5,10)

    t_days_lists = [] # list of lists for plot time ranges (0.1 d up to 10 d roughly)
    ERR_network_lists = [] # list of lists for the plot energy release rates as read from the network
    ERR_Rosswog_lists = [] # list of lists for the plot energy release rates as read from the network
    ERR_Lippuner_lists = [] # list of lists for the plot energy release rates as read from the network
    Y_e_plots = np.linspace(0.05,0.5,10) # trajectory Y_e values for the plots

    # step 2) fill lists with data

    # time range
    log_t_min = math.log10(t_days_min)
    log_t_max = math.log10(t_days_max)
    log_t_arr = np.linspace(log_t_min,log_t_max,numb_pts)
    t_days_fixed_list = [10**log_t for log_t in log_t_arr]

    # for each target Y_e, read the trajectory which as Y_e closest to it
    for target_Y_e in target_Y_e_values:
        # 2.1) extract the network data
        red_traj_data = traj_summ_data.iloc[(traj_summ_data['Ye']-target_Y_e).abs().argsort()[:1]]
        traj_ID = red_traj_data['Id'].item()
        
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
        red_heat_DF = dfheatingthermo.loc[(dfheatingthermo['timesec'] >= t_days_min * days_to_s) & (dfheatingthermo['timesec'] <= t_days_max * days_to_s)]

        # obtain specific heating rate, includes now neutrinos
        t_days_lists.append([t / 86400 for t in red_heat_DF['timesec'].to_numpy()])
        q_dot_arr = red_heat_DF['htot'].to_numpy()

        Q_dot_arr = [float(qdot) for qdot in q_dot_arr]

        ERR_network_lists.append(Q_dot_arr)

        # 2.2) obtain Rosswog & Korobkin heating rate
        # need to get velocity and Y_e0 of the trajectory

        v_traj = traj_summ_data.loc[traj_summ_data['Id'] == traj_ID]['velocity'].item() # in km per s
        Y_e_traj = traj_summ_data.loc[traj_summ_data['Id'] == traj_ID]['Ye'].item()

        # now round both velocity and Y_e to next steps from Rosswog's library
        v_Rosswog = [0.05,0.1,0.2,0.3,0.4,0.5]
        Y_e_Rosswog = np.linspace(0.05,0.5,10)

        v_closest = min(v_Rosswog, key=lambda x:abs(x-v_traj/c_kms))
        Y_e_closest = min(Y_e_Rosswog, key=lambda x:abs(x-Y_e_traj))

        colnames = ['time_days','qdot']
        Rosswog_heating_df = pd.read_csv(
            f"{RosswogLib_traj}/Heating_M0.05_v{v_closest:,.2f}_Ye{Y_e_closest:,.2f}.dat", delimiter=r"\s+", names=colnames, skiprows = 1
        )

        Q_dot_Rosswog = np.zeros(len(t_days_fixed_list))
        
        for t_idx, t_d in enumerate(t_days_fixed_list):
            q_dot_interpol = interp1d(Rosswog_heating_df.time_days.to_numpy(), Rosswog_heating_df.qdot.to_numpy(), kind='linear')(t_d)
            Q_dot_Rosswog[t_idx] = 10**q_dot_interpol
        ERR_Rosswog_lists.append(Q_dot_Rosswog)

        if Lippuner_traj is not None:
            # 2.3) obtain Lippuner & Roberts heating rate

            # ASSUME that Oli's e2e model used symmetric fission with zero free neutrons 
            # L&R define their initial point in time by a temperature of 6 GK

            dfevoldat = pd.read_csv(
                        get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/evol.dat"),
                        sep=r"\s+",
                        usecols=["#count", "time/s", "T9", "rho(g/cc)", "Ye", "S[k_b]"],
            )

            T_init = 6.0 # temperature of definition for the "initial" moment in the paper (in GK)
            # closest row
            closest_row = dfevoldat.iloc[(dfevoldat['T9']-T_init).abs().argsort()[:2][1]]
            t_exp_timescale = 1.0 # calculate expansion timescale w.r.t. time of 1 s
            other_row = dfevoldat.iloc[(dfevoldat['time/s']-t_exp_timescale).abs().argsort()[:2][1]]
            
            Y_e_0 = closest_row['Ye'].item()
            s_0 = closest_row['S[k_b]'].item()
            # expansion timescale in milliseconds!
            tau_0 = 1000 * closest_row['rho(g/cc)'].item() / ((closest_row['rho(g/cc)'].item()-other_row['rho(g/cc)'].item())/(other_row['time/s'].item()-closest_row['time/s'].item())) # exp timescale in s
            rho_0 = closest_row['rho(g/cc)'].item()

            # now read the Lippuner data
            Y_e_0_fit_data = lipp_data['1'].unique()
            s_0_fit_data = lipp_data['2'].unique()
            tau_0_fit_data = lipp_data['3'].unique()
            rho_0_fit_data = lipp_data['4'].unique()

            Y_e_0_closest = find_nearest(Y_e_0_fit_data, Y_e_0)
            s_0_closest = find_nearest(s_0_fit_data, s_0)
            tau_0_closest = find_nearest(tau_0_fit_data, tau_0)
            rho_0_closest = find_nearest(rho_0_fit_data, rho_0)

            row_to_take = lipp_data.loc[(lipp_data['1'] == Y_e_0_closest) & (lipp_data['2'] == s_0_closest) & (lipp_data['3'] == tau_0_closest) & (lipp_data['4'] == rho_0_closest)]
            
            assert len(row_to_take) > 0, f"No fit parameters found for point Y_e_0 = {Y_e_0}, s_0 = {s_0}, tau_0 = {tau_0}, rho_0 = {rho_0}"

            A = row_to_take['6'].item()
            alpha = row_to_take['7'].item()
            B_1 = row_to_take['8'].item()
            beta_1 = row_to_take['9'].item()
            B_2 = row_to_take['10'].item()
            beta_2 = row_to_take['11'].item()
            B_3 = row_to_take['12'].item()
            beta_3 = row_to_take['13'].item()

            Q_dot_Lippuner = np.zeros(len(t_days_fixed_list))
            for t_idx, t_d in enumerate(t_days_fixed_list): 
                q_dot = A*t_d**alpha + B_1 * np.exp(-t_d/beta_1) + B_2 * np.exp(-t_d/beta_2) + B_3 * np.exp(-t_d/beta_3)
                Q_dot_Lippuner[t_idx] = q_dot
            ERR_Lippuner_lists.append(Q_dot_Lippuner)

    # step 3) plot the results
    fig, axs = plt.subplots(
        nrows=5, ncols=2, figsize=(6, 10)
    )
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.6)
    
    for i in range(5):
        axs[i, 0].set(ylabel=r'$\dot{q}$ in erg / s')
        for j in range(2):
            axs[i, j].plot(t_days_lists[2*i+j], ERR_network_lists[2*i+j])
            axs[i, j].plot(t_days_fixed_list, ERR_Rosswog_lists[2*i+j])
            if Lippuner_traj is not None:
                axs[i, j].plot(t_days_fixed_list, ERR_Lippuner_lists[2*i+j])
            axs[i, j].set_title(rf'$Y_e$ = {Y_e_plots[2*i+j]:.2f}')
            axs[i, j].grid(color="k", linestyle="--", linewidth=0.5)
            axs[i, j].set_xscale('log')
            axs[i, j].set_yscale('log')
            axs[i, j].yaxis.set_tick_params(labelleft=True)
    axs[4, 0].set(xlabel='time in days')
    axs[4, 1].set(xlabel='time in days')


    #for ax in axs.flat:
        #ax.set(xlabel='time in days', ylabel=r'$\dot{Q}$ in erg / s')
    plt.savefig("Q_dot_tot_single_trajs.pdf")
    plt.clf()



def plot_set_of_trajectories(trajroot, model_type, t_days_min, t_days_max, numb_pts, RosswogLib_traj, Lippuner_traj) -> None:
    # loop over all trajectories

    # step 1) read trajectory overview / summary-all.dat and Lippuner data
    if model_type == "sfho":
        colnames0 = ["Id", "Mass", "time", "t9", "Ye", "entropy", "n/seed", "tau", "radius", "velocity", "angle"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"),
            delimiter=r"\s+",
            skiprows=1,
            names=colnames0,
            dtype_backend="pyarrow",
        )
    elif model_type == "e2e":
        colnames0 = ["Id", "Mass", "Ye", "velocity"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"),
            delimiter=r"\s+",
            skiprows=1,
            names=colnames0,
            dtype_backend="pyarrow",
        )

    colnames=[str(i) for i in range(1,39)]
    if Lippuner_traj is not None:
        lipp_data = pd.read_csv(
            Path(Lippuner_traj, "hires_sym0_results"),
            delimiter=r"\s+",
            skiprows=41,
            names=colnames,
            dtype_backend="pyarrow",
        )

    traj_id_list = traj_summ_data['Id'].to_numpy()

    # step 2) fill lists with data

    # time range
    log_t_min = math.log10(t_days_min)
    log_t_max = math.log10(t_days_max)
    log_t_arr = np.linspace(log_t_min,log_t_max,numb_pts)
    t_days_fixed_list = [10**log_t for log_t in log_t_arr]

    ERR_network_arr = np.zeros(len(t_days_fixed_list))
    ERR_Rosswog_arr = np.zeros(len(t_days_fixed_list))
    ERR_Lippuner_arr = np.zeros(len(t_days_fixed_list))

    # for each target Y_e, read the trajectory which as Y_e closest to it
    for traj_ID in traj_id_list:
        # 2.1) extract the network data
        
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
        red_heat_DF = dfheatingthermo.loc[(dfheatingthermo['timesec'] >= t_days_min * days_to_s) & (dfheatingthermo['timesec'] <= t_days_max * days_to_s)]

        # obtain specific heating rate, includes now neutrinos
        q_dot_arr = red_heat_DF['htot'].to_numpy()

        Q_dot_arr = [float(qdot) for qdot in q_dot_arr]

        ERR_network_arr = np.add(ERR_network_arr, Q_dot_arr)

        # 2.2) obtain Rosswog & Korobkin heating rate
        # need to get velocity and Y_e0 of the trajectory

        v_traj = traj_summ_data.loc[traj_summ_data['Id'] == traj_ID]['velocity'].item() # in km per s
        Y_e_traj = traj_summ_data.loc[traj_summ_data['Id'] == traj_ID]['Ye'].item()

        # now round both velocity and Y_e to next steps from Rosswog's library
        v_Rosswog = [0.05,0.1,0.2,0.3,0.4,0.5]
        Y_e_Rosswog = np.linspace(0.05,0.5,10)

        v_closest = min(v_Rosswog, key=lambda x:abs(x-v_traj/c_kms))
        Y_e_closest = min(Y_e_Rosswog, key=lambda x:abs(x-Y_e_traj))

        colnames = ['t[days]','qdot']
        Rosswog_heating_df = pd.read_csv(
            f"{RosswogLib_traj}/Heating_M0.05_v{v_closest:,.2f}_Ye{Y_e_closest:,.2f}.dat", delimiter=r"\s+", names=colnames, skiprows = 1
        )

        Q_dot_Rosswog = np.zeros(len(t_days_fixed_list))
        for t_idx, t_d in enumerate(t_days_fixed_list):
            q_dot_interpol = interp1d(Rosswog_heating_df.time_days.to_numpy(), Rosswog_heating_df.qdot.to_numpy(), kind='linear')(t_d)
            Q_dot_Rosswog[t_idx] = 10**q_dot_interpol

        ERR_Rosswog_arr = np.add(ERR_Rosswog_arr,Q_dot_Rosswog)

        if Lippuner_traj is not None:
            # 2.3) obtain Lippuner & Roberts heating rate

            # ASSUME that Oli's e2e model used symmetric fission with zero free neutrons 
            # L&R define their initial point in time by a temperature of 6 GK

            dfevoldat = pd.read_csv(
                        get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/evol.dat"),
                        sep=r"\s+",
                        usecols=["#count", "time/s", "T9", "rho(g/cc)", "Ye", "S[k_b]"],
            )

            T_init = 6.0 # temperature of definition for the "initial" moment in the paper
            # closest row
            closest_row = dfevoldat.iloc[(dfevoldat['T9']-T_init).abs().argsort()[:2][1]]
            t_exp_timescale = 1.0 # calculate expansion timescale w.r.t. time of 1 s
            other_row = dfevoldat.iloc[(dfevoldat['time/s']-t_exp_timescale).abs().argsort()[:2][1]]
            
            Y_e_0 = closest_row['Ye'].item()
            s_0 = closest_row['S[k_b]'].item()
            # expansion timescale in milliseconds!
            tau_0 = 1000 * closest_row['rho(g/cc)'].item() / ((closest_row['rho(g/cc)'].item()-other_row['rho(g/cc)'].item())/(other_row['time/s'].item()-closest_row['time/s'].item())) # exp timescale in s
            rho_0 = closest_row['rho(g/cc)'].item()

            # now read the Lippuner data
            Y_e_0_fit_data = lipp_data['1'].unique()
            s_0_fit_data = lipp_data['2'].unique()
            tau_0_fit_data = lipp_data['3'].unique()
            rho_0_fit_data = lipp_data['4'].unique()

            Y_e_0_closest = find_nearest(Y_e_0_fit_data, Y_e_0)
            s_0_closest = find_nearest(s_0_fit_data, s_0)
            tau_0_closest = find_nearest(tau_0_fit_data, tau_0)
            rho_0_closest = find_nearest(rho_0_fit_data, rho_0)

            row_to_take = lipp_data.loc[(lipp_data['1'] == Y_e_0_closest) & (lipp_data['2'] == s_0_closest) & (lipp_data['3'] == tau_0_closest) & (lipp_data['4'] == rho_0_closest)]

            assert len(row_to_take) > 0, f"No fit parameters found for point Y_e_0 = {Y_e_0}, s_0 = {s_0}, tau_0 = {tau_0}, rho_0 = {rho_0}"

            A = row_to_take['6'].item()
            alpha = row_to_take['7'].item()
            B_1 = row_to_take['8'].item()
            beta_1 = row_to_take['9'].item()
            B_2 = row_to_take['10'].item()
            beta_2 = row_to_take['11'].item()
            B_3 = row_to_take['12'].item()
            beta_3 = row_to_take['13'].item()

            Q_dot_Lippuner = np.zeros(len(t_days_fixed_list))
            for t_idx, t_d in enumerate(t_days_fixed_list): 
                q_dot = A*t_d**alpha + B_1 * np.exp(-t_d/beta_1) + B_2 * np.exp(-t_d/beta_2) + B_3 * np.exp(-t_d/beta_3)
                Q_dot_Lippuner[t_idx] = q_dot
            ERR_Lippuner_arr = np.add(ERR_Lippuner_arr,Q_dot_Lippuner)

    # step 3) plot the results
    plt.plot(t_days_lists,ERR_network_arr,color='r',label='GSI network')
    plt.plot(t_days_fixed_list,ERR_Rosswog_arr,color='g',label='Rosswog & Korobkin')
    if Lippuner_traj is not None:
        plt.plot(t_days_fixed_list,ERR_Lippuner_arr,color='b',label='Lippuner & Roberts')
    plt.grid(color="k", linestyle="--", linewidth=0.5)
    plt.xlabel("time in days")
    plt.ylabel(r"$\dot{Q}_{tot}$ in erg / s")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("Q_dot_tot.pdf")
    plt.clf()


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs) -> None:
    """Comparison to constant beta decay splitup factors."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    model_type = os.getcwd().split('/')[-1].split("_")[0] # assumes that trajectories are saved in a directory named e.g. "<MODEL INFO>_trajs"

    if args.singletrajectory:
        check_single_trajectories(args.trajectoryroot, model_type, args.timemin, args.timemax, args.timesteps, args.rosswogdata, args.lippunerdata)

    if args.plotmodel:
        plot_set_of_trajectories(args.trajectoryroot, model_type, args.timemin, args.timemax, args.timesteps, args.rosswogdata, args.lippunerdata)


if __name__ == "__main__":
    main()