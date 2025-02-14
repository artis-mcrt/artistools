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
        colnames0 = ["Id", "Mass", "Ye"]
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
    y_e_width = float(args.width)
    num_bins = int(1 / y_e_width)
    assert num_bins - (1 / y_e_width) < 1e-20, "Bins not proper" 
    Y_e_values = [(i+1/2)*y_e_width for i in range(num_bins)]
    masses = np.zeros(len(Y_e_values))
    for idx, row in traj_summ_data.iterrows():
        Y_e = row['Ye']
        M = row['Mass']
        bin_idx = int(Y_e / y_e_width)
        masses[bin_idx] += M
    # pdb.set_trace()
    plt.bar(Y_e_values, masses, width=y_e_width, zorder=2)
    plt.grid(color="k", linestyle="--", linewidth=0.5, zorder=1)
    plt.title(f'Model: {ejecta_model}')
    plt.xlabel("Ye")
    plt.ylabel('Mass in Msol')
    plt.savefig(
        f"yehistogram_{ejecta_model}_width{y_e_width:.2f}.pdf",
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
        "-width",
        default=0.05,
        help="Histogram bin width",
    )

if __name__ == "__main__":
    main()