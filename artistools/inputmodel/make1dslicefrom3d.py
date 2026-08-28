"""Extract a 1D ARTIS model from the cells of a 3D model that lie along one coordinate axis."""

import argparse
import math
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

import artistools as at
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.misc import print_warning
from artistools.plottools import save_figure


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-inputfolder", action="store", default=".", help="Path to folder with 3D files")

    parser.add_argument(
        "-axis", action="store", dest="chosenaxis", default="x", choices=["x", "y", "z"], help="Slice axis (x, y, or z)"
    )

    parser.add_argument(
        "-outputfolder", action="store", default="1dslice", help="Path to folder in which to store 1D output files"
    )

    parser.add_argument("-opdf", action="store", dest="pdfoutputfile", help="Path/filename for PDF plot")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Convert abundances.txt and model.txt from a 3D model to a one-dimensional slice."""
    args = at.parse_cli_args(addargs, main.__doc__, args, argsraw, kwargs)

    if not Path(args.outputfolder).exists():
        Path(args.outputfolder).mkdir(parents=True)
    elif Path(args.outputfolder, "model.txt").exists():
        print("ABORT: model.txt already exists")
        sys.exit()
    elif Path(args.outputfolder, "abundances.txt").exists():
        print("ABORT: abundances.txt already exists")
        sys.exit()

    dict3dcellidto1dcellid, xlist, ylists = slice_3dmodel(args.inputfolder, args.outputfolder, args.chosenaxis)

    slice_abundance_file(args.inputfolder, args.outputfolder, dict3dcellidto1dcellid)

    if args.pdfoutputfile:
        make_plot(xlist, ylists, args.pdfoutputfile)


def slice_3dmodel(
    inputfolder: Path | str, outputfolder: Path | str, chosenaxis: str
) -> tuple[dict[int, int], list[float], list[list[float]]]:
    """Write a 1D model.txt from the cells along chosenaxis, and return the 3D-to-1D cell id map and plot data."""
    xlist: list[float] = []
    ylists: list[list[float]] = [[], [], []]
    listout: list[str] = []
    dict3dcellidto1dcellid = {}
    outcellid = 0
    with Path(inputfolder, "model.txt").open(encoding="utf-8") as fmodelin:
        fmodelin.readline()  # npts_model3d
        t_model = fmodelin.readline()  # days
        fmodelin.readline()  # v_max in [cm/s]

        while True:
            # two lines making up a model grid cell
            block = fmodelin.readline(), fmodelin.readline()

            if not block[0] or not block[1]:
                break

            cell: dict[str, float | str] = {}
            blocksplit = block[0].split(), block[1].split()
            if len(blocksplit[0]) == 5:
                (cell["cellid"], cell["pos_x_min"], cell["pos_y_min"], cell["pos_z_min"], cell["rho"]) = blocksplit[0]
            else:
                print("Wrong line size")
                sys.exit()

            if len(blocksplit[1]) == 5:
                (cell["ffe"], cell["f56ni"], cell["fco"], cell["f52fe"], cell["f48cr"]) = map(float, blocksplit[1])
            else:
                print("Wrong line size")
                sys.exit()

            # a cell is on the chosen positive axis when its two other coordinates are zero. Compare the parsed
            # numbers, not their text: model.txt is written in several formats (e.g. "0.0000000" or "0.0000e0")
            positions = {ax: float(cell[f"pos_{ax}_min"]) for ax in ("x", "y", "z")}
            if all(pos == 0.0 or (chosenaxis == ax and pos >= 0.0) for ax, pos in positions.items()):
                outcellid += 1
                dict3dcellidto1dcellid[int(cell["cellid"])] = outcellid
                append_cell_to_output(cell, outcellid, t_model, listout, xlist, ylists)
                print(f"Cell {outcellid:4d} input1: {block[0].rstrip()}")
                print(f"Cell {outcellid:4d} input2: {block[1].rstrip()}")
                print(f"Cell {outcellid:4d} output: {listout[-1]}")

    with Path(outputfolder, "model.txt").open("w", encoding="utf-8") as fmodelout:
        fmodelout.write(f"{outcellid:7d}\n")
        fmodelout.write(t_model)
        fmodelout.writelines(line + "\n" for line in listout)

    return dict3dcellidto1dcellid, xlist, ylists


def slice_abundance_file(
    inputfolder: Path | str, outputfolder: Path | str, dict3dcellidto1dcellid: dict[int, int]
) -> None:
    """Write an abundances.txt holding only the cells kept by slice_3dmodel, renumbered to the 1D cell ids."""
    with (
        Path(inputfolder, "abundances.txt").open(encoding="utf-8") as fabundancesin,
        Path(outputfolder, "abundances.txt").open("w", encoding="utf-8") as fabundancesout,
    ):
        currentblock: list[str] = []
        keepcurrentblock = False
        blocklens: set[int] = set()
        for line in fabundancesin:
            linesplit = line.split()

            if len(currentblock) + len(linesplit) >= 30:
                if currentblock:  # record only completed blocks, not the empty state before the first one
                    blocklens.add(len(currentblock))
                if keepcurrentblock:
                    fabundancesout.write("  ".join(currentblock) + "\n")
                currentblock = []
                keepcurrentblock = False

            if not currentblock:
                currentblock = linesplit
                if int(linesplit[0]) in dict3dcellidto1dcellid:
                    outcellid = dict3dcellidto1dcellid[int(linesplit[0])]
                    currentblock[0] = f"{outcellid:6d}"
                    keepcurrentblock = True
            else:
                currentblock.extend(linesplit)

        # the loop only writes a block when the next one starts, so the last block still has to be flushed
        if currentblock:
            if blocklens and len(currentblock) < max(blocklens):
                print_warning(
                    f"the last block has {len(currentblock)} values, but earlier blocks have"
                    f" {max(blocklens)}. The input file looks truncated"
                )
            if keepcurrentblock:
                fabundancesout.write("  ".join(currentblock) + "\n")


def append_cell_to_output(
    cell: dict[str, float | str],
    outcellid: int,
    t_model: str | float,
    listout: list[str],
    xlist: list[float],
    ylists: list[list[float]],
) -> None:
    """Append one cell to the 1D model output lines and to the density and abundance plot series."""
    dist = math.sqrt(float(cell["pos_x_min"]) ** 2 + float(cell["pos_y_min"]) ** 2 + float(cell["pos_z_min"]) ** 2)
    velocity = dist / float(t_model) / day_to_s / km_to_cm

    listout.append(
        f"{outcellid:6d}  {velocity:8.2f}  {math.log10(max(float(cell['rho']), 1e-100)):8.5f}  "
        f"{cell['ffe']:.5f}  {cell['f56ni']:.5f}  {cell['fco']:.5f}  {cell['f52fe']:.5f}  {cell['f48cr']:.5f}"
    )

    xlist.append(velocity)
    ylists[0].append(float(cell["rho"]))
    ylists[1].append(float(cell["f56ni"]))
    ylists[2].append(float(cell["fco"]))


def make_plot(xlist: list[float], ylists: list[list[float]], pdfoutputfile: str) -> None:
    """Plot density and the Ni56 and Co mass fractions of the slice against velocity, and save it as a PDF."""
    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharey=True, figsize=(6, 4), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0}
    )
    axis.set_xlabel(r"Velocity [km/s]")
    axis.set_ylabel(r"Density [g/cm$^3$] or mass fraction")
    ylabels = [r"$\rho$", "fNi56", "fCo"]
    for ylist, ylabel in zip(ylists, ylabels, strict=False):
        axis.plot(xlist, ylist, linewidth=1.5, label=ylabel)
    axis.set_yscale("log", nonpositive="clip")
    axis.legend(loc="best", handlelength=2, frameon=False, numpoints=1, prop={"size": 10})
    save_figure(fig, pdfoutputfile, format="pdf")


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
