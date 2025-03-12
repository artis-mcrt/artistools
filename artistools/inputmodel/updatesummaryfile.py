# short script to create summary-all.dat from trajectory set
# ASSUMES that tar files are extracted for now

import argparse
from collections.abc import Sequence
from functools import partial
from pathlib import Path
import math
import os
import pdb
from scipy.interpolate import interp1d
from scipy import integrate

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import tqdm.rich
from tqdm import TqdmExperimentalWarning
from tqdm.contrib.concurrent import process_map
from decimal import Decimal


import artistools as at
from artistools.configuration import get_config
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-trajectoryroot",
        "-trajroot",
        default=None,
        help="Path to nuclear network trajectory folder, if abundances are required",
    )

    parser.add_argument(
        "-updatetype",
        default="update",
        help="Specify if the summary file has to be newly created (option create) or just updated (default)",
    )

    parser.add_argument(
        "-addcolumns",
        default=None,
        help="Comma-separated string to give the table columns that shall be added. Options: id,mass,ye0,v0 (is also default right now)",
    )

    parser.add_argument(
        "-columns",
        default="id,mass,ye0,v0",
        help="Comma-separated string to give the table columns. Options: id,mass,ye0,v0 (is also default right now)",
    )


def create_summary_file():
    """
    ofs = open("summary-all.dat","w")
    # list all trajectories first
    traj_id_list = []
    for file in os.listdir(args.trajectoryroot):
        if file.endswith(".tar"):
            traj_id_list.append(int(file.split(".")[0]))

    # now get the data for each trajectory
    for traj_id in traj_id_list:
        traj_df = pd.read_csv(
            get_tar_member_extracted_path(args.trajroot, traj_id, "./Run_rprocess/heating.dat"),
            sep=r"\s+",
            usecols=["#count", "hbeta", "htot"],
        )
    """
    print("Not implemented yet! Sry.")


def update_summary_file(trajroot, columns_to_add_list, model_type):
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

    ifs = open("summary-all.dat", "r")
    la = ifs.read().splitlines()
    ofs = open("summary-all_update.dat", "w")
    if "v0" in columns_to_add_list:
        dd = np.load("kilonova_artis_input_138n1a6.npz")

        for line in la:
            traj_id = int(line.split()[0])
            dd_idx = np.where(dd["idx"] == float(traj_id))[0]
            assert dd_idx >= 0, f"trajectory {traj_id} not found in dict"
            v = dd["pos"][dd_idx][0]  # velocity in fraction of c
            v_kmps = v[0] * 299792.458
            ofs.write(line)
            ofs.write("   ")
            ofs.write(str(v_kmps))
            ofs.write("\n")

    if "ye0" in columns_to_add_list:
        header_str = "#  Id   Mass          Ye            velocity      T9\n"
        ofs.write(header_str)
        for line in la:
            ofs_line_str = ""
            traj_id = int(line.split()[0])
            traj_df = pd.read_csv(
                get_tar_member_extracted_path(trajroot, traj_id, "./Run_rprocess/energy_thermo.dat"),
                sep=r"\s+",
                usecols=["#count", "time/s", "Qdot", "Ye", "T9"],
            )
            if max(traj_df['T9']) < 5.0:
                last_row = traj_df.iloc[0]
            else:
                last_row = traj_df.loc[traj_df["T9"] > 5.0].iloc[-1]
            Y_e_0 = last_row["Ye"]
            T_0 = last_row['T9']
            ofs_line_str += (
                str(traj_id).rjust(6)
                + "  "
                + '%1.6e'%float(line.split()[1])
                + "  "
                + '%1.6e'%Y_e_0
                + "  "
                + '%1.6e'%float(line.split()[3])
                + "  "
                + '%1.6e'%T_0
                + "\n"
            )
            ofs.write(ofs_line_str)

    ifs.close()
    ofs.close()


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs) -> None:
    model_type = (
        os.getcwd().split("/")[-1].split("_")[0]
    )  # assumes that trajectories are saved in a directory named e.g. "<MODEL INFO>_trajs"

    if args.updatetype == "update":
        if args.addcolumns:
            columns_arr = args.addcolumns.split(",")
            update_summary_file(args.trajectoryroot, columns_arr, model_type)
        else:
            print("Error! No columns to be added specified. Abort.")
    elif args.updatetype == "create":
        create_summary_file()


if __name__ == "__main__":
    main()
