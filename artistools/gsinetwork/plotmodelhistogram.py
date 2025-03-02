# PYTHON_ARGCOMPLETE_OK

# script to plot various histograms for a given ejecta model

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

def plot_mass_histogram(trajectoryroot, x_values, masses, width, xlabel, ejecta_model):
    pass

def plot_initial_histogram(quantity, trajectoryroot):
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
        print("Unknown ejecta model type!")
        sys.exit()
    traj_summ_data = pd.read_csv(
        f"{trajectoryroot}/summary-all.dat",
        delimiter=r"\s+",
        skiprows=1,
        names=colnames0,
        dtype_backend="pyarrow",
    )

    # collect data
    match quantity:
        case "ye0":
            y_e_width = float(args.width)
            num_bins = int(1 / y_e_width)
            assert num_bins - (1 / y_e_width) < 1e-20, "Ye0 bins not proper" 
            Y_e_values = [(i+1/2)*y_e_width for i in range(num_bins)]
            masses = np.zeros(len(Y_e_values))
            for idx, row in traj_summ_data.iterrows():
                Y_e = row['Ye']
                M = row['Mass']
                bin_idx = int(Y_e / y_e_width)
                masses[bin_idx] += M

    # plot 
    plt.bar(x_values, masses, width=width, zorder=2)
    plt.grid(color="k", linestyle="--", linewidth=0.5, zorder=1)
    plt.title(f'Model: {ejecta_model}')
    plt.xlabel(xlabel)
    plt.ylabel(r'Mass in $M_{\odot}$')
    plt.savefig(
        f"yehistogram_{ejecta_model}_width{width:.2f}.pdf",
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
        "-initialye",
        default=False,
        help="Plot electron fraction distribution at last NSE timestep (5 GK)",
    )

    parser.add_argument(
        "-initialentropy",
        default=False,
        help="Plot specific entropy distribution at last NSE timestep (5 GK)",
    )

    parser.add_argument(
        "-initialexpansiontimescale",
        default=False,
        help="Plot expansion timescale at last NSE timestep (5 GK)",
    )

    parser.add_argument(
        "-numbbins",
        default=20,
        help="Number of histogram bins",
    )


def main(
    args: argparse.Namespace | None = None,
    argsraw: Sequence[str] | None = None,
    **kwargs,
) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    model_type = os.getcwd().split('/')[-1].split("_")[0]

    if args.initialye:
        plot_initial_histogram("ye0", args.trajectoryroot)

    if args.initialentropy:
        plot_initial_histogram("s0", args.trajectoryroot)

    if args.initialexpansiontimescale:
        plot_initial_histogram("tau0", args.trajectoryroot)


if __name__ == "__main__":
    main()

