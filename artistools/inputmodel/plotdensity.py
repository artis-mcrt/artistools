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
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(6, 6),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0},
    )
    assert isinstance(axes, np.ndarray)

    if not args.modelpath:
        args.modelpath = ["."]

    for modelpath in args.modelpath:
        dfmodel, modelmeta = at.get_modeldata(modelpath, derived_cols=["vel_r_min", "vel_r_mid", "vel_r_max", "mass_g"])
        label = at.get_model_name(modelpath)
        print(f"Plotting {label}")
        binned_xvals: list[float] = []
        binned_yvals: list[float] = []

        # total_mass = dfmodel.mass_g.sum() / 1.989e33
        dfmodel = dfmodel.sort(by="vel_r_mid")

        dfmodelcollect = dfmodel.select(["modelgridindex", "vel_r_min", "vel_r_mid", "vel_r_max", "mass_g"]).collect()

        vuppers = dfmodelcollect["vel_r_max"].unique().sort()
        enclosed_xvals = [0.0, *(vuppers / 29979245800).to_list()]
        enclosed_yvals = [0.0] + [
            dfmodelcollect.filter(pl.col("vel_r_mid") <= vupper)["mass_g"].sum() / 1.989e33 for vupper in vuppers
        ]
        axes[0].plot(enclosed_xvals, enclosed_yvals, label=label)

        if "vel_r_max_kmps" in dfmodel.collect_schema().names():
            vupperscoarse = vuppers.to_list()
        else:
            ncoarsevelbins = int(
                modelmeta["ncoordgridrcyl"] if "ncoordgridrcyl" in modelmeta else modelmeta["ncoordgridx"] / 2.0
            )
            vupperscoarse = [modelmeta["vmax_cmps"] / ncoarsevelbins * (i + 1) for i in range(ncoarsevelbins)]

        vlowerscoarse = [0.0] + vupperscoarse[:-1]
        for vlower, vupper in zip(vlowerscoarse, vupperscoarse, strict=False):
            velbinmass = (
                dfmodelcollect.filter(pl.col("vel_r_mid").is_between(vlower, vupper, closed="left"))["mass_g"].sum()
                / 1.989e33
            )
            assert vlower < vupper
            binned_xvals.extend((vlower / 29979245800, vupper / 29979245800))
            delta_beta = (vupper - vlower) / 29979245800
            yval = velbinmass / delta_beta
            binned_yvals.extend((yval, yval))

        axes[1].plot(binned_xvals, binned_yvals, label=label)
        vmax_on_c = modelmeta["vmax_cmps"] / 29979245800
        axes[0].set_xlim(right=vmax_on_c)

    axes[-1].set_xlabel("Velocity [$c$]")
    axes[0].set_ylabel(r"Mass Enclosed [M$_\odot$]")
    axes[1].set_ylabel(r"$\Delta$M/$\Delta v$  [M$_\odot/c$]")
    axes[1].legend(frameon=False)

    axes[-1].set_xlim(left=0.0)
    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)

    outfilepath = Path(args.outputpath)
    if outfilepath.is_dir():
        outfilepath /= "densityprofile.pdf"

    fig.savefig(outfilepath)
    print(f"Saved {outfilepath}")


if __name__ == "__main__":
    main()
