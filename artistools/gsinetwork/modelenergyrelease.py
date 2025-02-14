# PYTHON_ARGCOMPLETE_OK
import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path

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
        "-singletrajectory",
        default=False,
        help="Compare the heating rate fitting formulae to single trajectories only for better comparison",
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


def check_single_trajectories(trajroot, model_type, t_days_min, t_days_max) -> None:
    # compare fit heating rate to network data for Y_e of 0.05 up to 0.5 in 0.05 bins  
    # creates a 5x2 plot, each subplot for the different Y_e values

    # step 1) read trajectory overview / summary-all.dat
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
        colnames0 = ["Id", "Mass", "Ye"]
        traj_summ_data = pd.read_csv(
            Path(trajroot, "summary-all.dat"),
            delimiter=r"\s+",
            skiprows=1,
            names=colnames0,
            dtype_backend="pyarrow",
        )
    
    target_Y_e_values = np.linspace(0.05,0.5,10)

    t_days_lists = [] # list of lists for plot time ranges (0.1 d up to 10 d roughly)
    ERR_network_lists = [] # list of lists for the plot energy release rates as read from the network
    ERR_Rosswog_lists = [] # list of lists for the plot energy release rates as read from the network
    ERR_Lippuner_lists = [] # list of lists for the plot energy release rates as read from the network
    Y_e_plots = np.zeros(10) # trajectory Y_e values for the plots

    # step 2) fill lists with data

    # for each target Y_e, read the trajectory which as Y_e closest to it
    for target_Y_e in target_Y_e_values:
        red_traj_data = traj_summ_data.iloc[(traj_summ_data['Ye']-target_Y_e).abs().argsort()[:1]]
        traj_ID = red_traj_data['Id']
        # now read individual trajectory
        dfheatingthermo = (
            pl.from_pandas(
                pd.read_csv(
                    get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/heating.dat"),
                    sep=r"\s+",
                    usecols=["#count", "hbeta", "htot", "Ye"],
                )
            )
            .join(
                pl.from_pandas(
                    pd.read_csv(
                        get_tar_member_extracted_path(trajroot, traj_ID, "./Run_rprocess/energy_thermo.dat"),
                        sep=r"\s+",
                        usecols=["#count", "time/s", "Qdot"],
                    )
                ),
                on="#count",
                how="left",
                coalesce=True,
            )
            .rename({"#count": "nstep", "time/s": "timesec"})
        ).to_pandas()

        red_heat_DF = dfheatingthermo.loc[(dfheatingthermo['time/s'] >= t_days_min * days_to_s) & (dfheatingthermo['time/s'] <= t_days_max * days_to_s)]


    # step 3) plot the results
    fig, axs = plt.subplots(5, 2)
    axs[0, 0].plot(t_days_lists[0], ERR_network_lists[0])
    axs[0, 0].plot(t_days_lists[0], ERR_Rosswog_lists[0])
    axs[0, 0].plot(t_days_lists[0], ERR_Lippuner_lists[0])
    axs[0, 0].set_title(rf'$Y_e$ = {Y_e_plots[0]}')
    axs[0, 1].plot(t_days_lists[1], ERR_network_lists[1])
    axs[0, 1].plot(t_days_lists[1], ERR_Rosswog_lists[1])
    axs[0, 1].plot(t_days_lists[1], ERR_Lippuner_lists[1])
    axs[0, 1].set_title(rf'$Y_e$ = {Y_e_plots[1]}')
    axs[1, 0].plot(t_days_lists[2], ERR_network_lists[2])
    axs[1, 0].plot(t_days_lists[2], ERR_Rosswog_lists[2])
    axs[1, 0].plot(t_days_lists[2], ERR_Lippuner_lists[2])
    axs[1, 0].set_title(rf'$Y_e$ = {Y_e_plots[2]}')
    axs[1, 1].plot(t_days_lists[3], ERR_network_lists[3])
    axs[1, 1].plot(t_days_lists[3], ERR_Rosswog_lists[3])
    axs[1, 1].plot(t_days_lists[3], ERR_Lippuner_lists[3])
    axs[1, 1].set_title(rf'$Y_e$ = {Y_e_plots[3]}')
    axs[2, 0].plot(t_days_lists[4], ERR_network_lists[4])
    axs[2, 0].plot(t_days_lists[4], ERR_Rosswog_lists[4])
    axs[2, 0].plot(t_days_lists[4], ERR_Lippuner_lists[4])
    axs[2, 0].set_title(rf'$Y_e$ = {Y_e_plots[4]}')
    axs[2, 1].plot(t_days_lists[5], ERR_network_lists[5])
    axs[2, 1].plot(t_days_lists[5], ERR_Rosswog_lists[5])
    axs[2, 1].plot(t_days_lists[5], ERR_Lippuner_lists[5])
    axs[2, 1].set_title(rf'$Y_e$ = {Y_e_plots[5]}')
    axs[3, 0].plot(t_days_lists[6], ERR_network_lists[6])
    axs[3, 0].plot(t_days_lists[6], ERR_Rosswog_lists[6])
    axs[3, 0].plot(t_days_lists[6], ERR_Lippuner_lists[6])
    axs[3, 0].set_title(rf'$Y_e$ = {Y_e_plots[6]}')
    axs[3, 1].plot(t_days_lists[7], ERR_network_lists[7])
    axs[3, 1].plot(t_days_lists[7], ERR_Rosswog_lists[7])
    axs[3, 1].plot(t_days_lists[7], ERR_Lippuner_lists[7])
    axs[3, 1].set_title(rf'$Y_e$ = {Y_e_plots[7]}')
    axs[4, 0].plot(t_days_lists[8], ERR_network_lists[8])
    axs[4, 0].plot(t_days_lists[8], ERR_Rosswog_lists[8])
    axs[4, 0].plot(t_days_lists[8], ERR_Lippuner_lists[8])
    axs[4, 0].set_title(rf'$Y_e$ = {Y_e_plots[8]}')
    axs[4, 1].plot(t_days_lists[9], ERR_network_lists[9])
    axs[4, 1].plot(t_days_lists[9], ERR_Rosswog_lists[9])
    axs[4, 1].plot(t_days_lists[9], ERR_Lippuner_lists[9])
    axs[4, 1].set_title(rf'$Y_e$ = {Y_e_plots[9]}')

    for ax in axs.flat:
        ax.set(xlabel='time in days', ylabel=r'$\dot{Q}$ in erg / s')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs) -> None:
    """Comparison to constant beta decay splitup factors."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    model_type = args.trajectoryroot.split("_")[0] # assumes that trajectories are saved in a directory named e.g. "<MODEL INFO>_trajs"

    if args.singletrajectory:
        check_single_trajectories(args.trajectoryroot, model_type, args.timemin, args.timemax)


if __name__ == "__main__":
    main()