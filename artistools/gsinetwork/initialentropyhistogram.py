# PYTHON_ARGCOMPLETE_OK

import argparse
import math
import multiprocessing as mp
import urllib.request
from collections.abc import Sequence
from functools import partial
import os
import argcomplete
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import polars as pl
import pdb
from collections import Counter
import sys

import artistools as at
from artistools.configuration import get_config
from artistools.inputmodel.rprocess_from_trajectory import get_tar_member_extracted_path

def main(
    args: argparse.Namespace | None = None,
    argsraw: Sequence[str] | None = None,
    **kwargs,
) -> None:
    ejecta_model = os.getcwd().split("/")[-1]
    # get masses of trajectories
    if ejecta_model == "sfho_trajs":
        colnames0 = [
            "Id",
            "Mass",
            "time",
            "t9",
            "Ye",
            "entropy",
            "n/seed",
            "tau",
            "radius",
            "velocity",
            "angle",
        ]
    elif ejecta_model == "e2e_trajs":
        colnames0 = ["Id", "Mass", "Ye", "velocity"]
    else:
        print("Unknown ejecta model!")
        sys.exit()
    traj_summ_data = pd.read_csv(
        f"{args.trajectoryroot}/summary-all.dat",
        delimiter=r"\s+",
        skiprows=1,
        names=colnames0,
        dtype_backend="pyarrow",
    )
    ID_list = traj_summ_data['Id'].to_numpy()
    masses_list = traj_summ_data['Mass'].to_numpy()
    s_0_arr = np.zeros(len(ID_list))
    for idx, traj_id in enumerate(ID_list):
        dfevoldat = pd.read_csv(
                    get_tar_member_extracted_path(args.trajroot, traj_id, "./Run_rprocess/evol.dat"),
                    sep=r"\s+",
                    usecols=["#count", "time/s", "T9", "rho(g/cc)", "Ye", "S[k_b]"],
        )
        closest_row = dfevoldat.iloc[(dfevoldat['T9']-args.temperature).abs().argsort()[:2][1]]
        s_0_arr[idx] = closest_row['S[k_b]'].item()
    
    num_bins = args.numbbins
    s_0_width = max(s_0_arr) / num_bins
    assert num_bins - (1 / s_0_width) < 1e-20, "Bins not proper" 
    s_0_values = [(i+1/2)*s_0_width for i in range(num_bins)]
    masses = np.zeros(len(s_0_values))
    for idx in range(len(ID_list)):
        bin_idx = int(s_0_arr[idx] / s_0_width)
        masses[bin_idx] += masses_list[idx] 
    # pdb.set_trace()
    plt.bar(s_0_values, masses, width=s_0_width, zorder=2)
    plt.grid(color="k", linestyle="--", linewidth=0.5, zorder=1)
    plt.title(f'Model: {ejecta_model}')
    plt.xlabel("Ye")
    plt.ylabel('Mass in Msol')
    plt.savefig(
        f"s0histogram_{ejecta_model}_width{s_0_width:.2f}.pdf",
        bbox_inches="tight",
    )
    plt.clf()

def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-trajectoryroot",
        "-trajroot",
        default=None,
        help="Path to nuclear network trajectory folder, if abundances are required",
    )

    parser.add_argument(
        "-numbbins",
        default=20,
        help="Histogram bin number",
    )

    parser.add_argument(
        "-temperature",
        default=5.0,
        help="Temperature in GK at which the histogram is created",
    )

if __name__ == "__main__":
    main()