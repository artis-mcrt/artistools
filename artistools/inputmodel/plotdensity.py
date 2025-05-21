#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "modelpath",
        default=[],
        nargs="*",
        action=at.AppendPath,
        help="Path(s) to model.txt file(s) or folders containing model.txt)",
    )

    parser.add_argument("-label", default=[], nargs="*", help="List of series label overrides")

    parser.add_argument("-color", default=[f"C{i}" for i in range(10)], nargs="*", help="List of line colors")

    parser.add_argument("-xmax", type=float, default=None, help="Plot range: x-axis")

    parser.add_argument("-xmin", type=float, default=None, help="Plot range: x-axis")

    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radial density profile of an ARTIS model."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(6, 6),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )
    assert isinstance(axes, np.ndarray)

    if not args.modelpath:
        args.modelpath = ["."]

    args.color, args.label = at.trim_or_pad(len(args.modelpath), args.color, args.label)
    args.label = [
        at.get_model_name(modelpath) if label is None else label
        for modelpath, label in zip(args.modelpath, args.label, strict=True)
    ]

    max_vmax_on_c = float("-inf")
    for color, label, modelpath in zip(args.color, args.label, args.modelpath, strict=True):
        print(f"Plotting {label}")
        dfmodel, modelmeta = at.get_modeldata(modelpath, derived_cols=["vel_r_min", "vel_r_mid", "vel_r_max", "mass_g"])

        vmax_on_c = modelmeta["vmax_cmps"] / 29979245800
        max_vmax_on_c = max(vmax_on_c, max_vmax_on_c)

        # total_mass = dfmodel.mass_g.sum() / 1.989e33
        dfmodel = dfmodel.sort(by="vel_r_mid")

        cols = ["modelgridindex", "vel_r_min", "vel_r_mid", "vel_r_max", "mass_g"]
        if "Ye" in dfmodel.collect_schema().names():
            cols.append("Ye")

        dfmodelcollect = dfmodel.select(cols).collect()

        vuppers = dfmodelcollect["vel_r_max"].unique().sort()
        enclosed_xvals = [0.0, *(vuppers / 29979245800).to_list(), 29979245800]
        enclosed_yvals = [0.0] + [
            dfmodelcollect.filter(pl.col("vel_r_mid") <= vupper)["mass_g"].sum() / 1.989e33 for vupper in vuppers
        ]
        enclosed_yvals.append(dfmodelcollect["mass_g"].sum() / 1.989e33)
        axes[0].plot(enclosed_xvals, enclosed_yvals, label=label, color=color)

        if "vel_r_max_kmps" in dfmodel.collect_schema().names():
            vupperscoarse = vuppers.to_list()
        else:
            ncoarsevelbins = int(
                modelmeta["ncoordgridrcyl"] if "ncoordgridrcyl" in modelmeta else modelmeta["ncoordgridx"] / 2.0
            )
            vupperscoarse = [modelmeta["vmax_cmps"] / ncoarsevelbins * (i + 1) for i in range(ncoarsevelbins)]

        binned_xvals: list[float] = []
        binned_yvals: list[float] = []
        vlowerscoarse = [0.0, *vupperscoarse[:-1]]
        for vlower, vupper in zip(vlowerscoarse, vupperscoarse, strict=True):
            velbinmass = (
                dfmodelcollect.filter(pl.col("vel_r_mid").is_between(vlower, vupper, closed="left"))["mass_g"].sum()
                / 1.989e33
            )
            assert vlower < vupper
            binned_xvals.extend((vlower / 29979245800, vupper / 29979245800))
            delta_beta = (vupper - vlower) / 29979245800
            yval = velbinmass / delta_beta
            binned_yvals.extend((yval, yval))

        axes[1].plot(binned_xvals, binned_yvals, label=label, color=color)
        if "Ye" in dfmodelcollect.collect_schema().names():
            binned_xvals = []
            binned_yvals = []
            vlowerscoarse = [0.0, *vupperscoarse[:-1]]
            for vlower, vupper in zip(vlowerscoarse, vupperscoarse, strict=True):
                yval = (
                    dfmodelcollect.filter(pl.col("vel_r_mid").is_between(vlower, vupper, closed="left"))
                    .select(pl.col("Ye").dot(pl.col("mass_g")) / pl.col("mass_g").sum())
                    .item()
                )
                binned_xvals.extend((vlower / 29979245800, vupper / 29979245800))
                binned_yvals.extend((yval, yval))

            axes[2].plot(binned_xvals, binned_yvals, label=label, color=color)

    if args.xmin is not None:
        axes[0].set_xlim(left=args.xmin)
    else:
        axes[0].set_xlim(left=0.0)

    if args.xmax is not None:
        axes[0].set_xlim(right=args.xmax)
    else:
        axes[0].set_xlim(right=max_vmax_on_c)

    axes[-1].set_xlabel("Velocity [$c$]")
    axes[0].set_ylabel(r"Mass Enclosed [M$_\odot$]")
    axes[1].set_ylabel(r"$\Delta$M/$\Delta v$  [M$_\odot/c$]")
    axes[2].set_ylabel(r"Electron fraction Ye")
    axes[1].legend(frameon=False)

    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)

    outfilepath = Path(args.outputpath)
    if outfilepath.is_dir():
        outfilepath /= "densityprofile.pdf"

    fig.savefig(outfilepath)
    print(f"Saved {outfilepath}")


if __name__ == "__main__":
    main()
