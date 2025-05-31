"""Script to plot the average gamma ray opacity from an ARTIS run."""

# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import artistools as at

opac_thr = 0.1
base_dir = "/lustre/theory"


# function that reads output_0-0.txt which contains all the information
def read_gamma_opacity(path: Path) -> npt.NDArray[np.floating | np.integer]:
    p = Path(f"{path.absolute().as_posix()}/output_0-0.txt")
    with p.open(encoding="utf-8") as f:
        contents = f.read()
    lines_list = [line for line in contents.splitlines() if "kappa" in line]

    opac_data = np.zeros((len(lines_list), 11))
    for idx, line in enumerate(lines_list):
        opac_data[idx, 0] = float(line.split("|||")[0].split()[-2])  # t_d
        opac_data[idx, 1] = float(line.split("|||")[1].split()[2])  # kappa_dec_tot
        opac_data[idx, 2] = float(line.split("|||")[1].split()[4])  # kappa_dec_pp
        opac_data[idx, 3] = float(line.split("|||")[1].split()[6])  # kappa_dec_pe
        opac_data[idx, 4] = float(line.split("|||")[1].split()[8])  # kappa_dec_c
        opac_data[idx, 5] = float(line.split("|||")[2].split()[1])  # kappa_abs_tot
        opac_data[idx, 6] = float(line.split("|||")[2].split()[3])  # kappa_abs_pp
        opac_data[idx, 7] = float(line.split("|||")[2].split()[5])  # kappa_abs_pe
        opac_data[idx, 8] = float(line.split("|||")[2].split()[7])  # kappa_abs_c
        opac_data[idx, 9] = float(line.split("|||")[3].split()[4])  # E_avg_dec
        opac_data[idx, 10] = float(line.split("|||")[3].split()[6])  # E_avg_abs

    # truncate weird data points
    mask = np.all(opac_data[:, 1:8] < opac_thr, axis=1)
    return opac_data[mask]


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "modelpath", default=[], nargs="*", action=at.AppendPath, help="Path(s) to ARTIS folders with output_0-0.txt"
    )

    parser.add_argument("--dectot", action="store_true", help="Plot total opacity at packet decay")
    parser.add_argument("--decpp", action="store_true", help="Plot pair production opacity at packet decay")
    parser.add_argument("--decpe", action="store_true", help="Plot photoelectric opacity at packet decay")
    parser.add_argument("--decc", action="store_true", help="Plot Compton opacity at packet decay")
    parser.add_argument("--abstot", action="store_true", help="Plot total opacity at packet absorption")
    parser.add_argument("--abspp", action="store_true", help="Plot pair production opacity at packet absorption")
    parser.add_argument("--abspe", action="store_true", help="Plot photoelectric opacity at packet absorption")
    parser.add_argument("--absc", action="store_true", help="Plot Compton opacity at packet absorption")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # for each model obtain array consisting time,
    for mp_idx, mp in enumerate(args.modelpath):
        # label
        with (mp / "plotlabel.txt").open("r", encoding="utf-8") as fp:
            ls = fp.read().splitlines()
            plotlabel = ls[0]

        # obtain data
        opac_data = read_gamma_opacity(mp)
        # plot
        if args.abstot:
            plt.plot(opac_data[:, 0], opac_data[:, 5], linestyle="-", color=default_colours[mp_idx], label=plotlabel)

        if args.abspp:
            plt.plot(opac_data[:, 0], opac_data[:, 6], linestyle=":", color=default_colours[mp_idx])

        if args.abspe:
            plt.plot(opac_data[:, 0], opac_data[:, 7], linestyle="--", color=default_colours[mp_idx])

        if args.absc:
            plt.plot(opac_data[:, 0], opac_data[:, 8], linestyle="-.", color=default_colours[mp_idx])

    plt.grid(color="k", linestyle="--", linewidth=0.5)
    plt.xlabel("time in days")
    plt.ylabel(r"opacity in cm$^2$ g$^{-1}$")
    plt.xscale("log")
    plt.xlim((0.1, 10.0))
    plt.legend()
    plt.savefig("gammaopacity.pdf")
    plt.clf()


if __name__ == "__main__":
    main()
