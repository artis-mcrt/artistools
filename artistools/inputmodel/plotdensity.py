# PYTHON_ARGCOMPLETE_OK
"""Plot mass density against velocity for one or more ARTIS input models."""

import argparse
import typing as t
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import Msun_to_g
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_outputpath
from artistools.misc import addarg_seriesstyle
from artistools.misc import addarg_show
from artistools.plottools import save_figure


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(
        parser,
        positional=True,
        multiplepaths=True,
        default=[],
        helptext="Path(s) to model.txt file(s) or folders containing model.txt)",
    )

    addarg_seriesstyle(
        parser, colordefault=[f"C{i}" for i in range(10)], include_linestyles=False, include_dashes=False
    )

    addarg_axislimits(parser, include_y=False)

    parser.add_argument(
        "-nbins", type=int, default=None, help="Use specified number of fixed velocity bins up to maximum plot velocity"
    )

    parser.add_argument("--plotye", action="store_true", help="Plot electron fraction versus velocity")

    addarg_outputpath(parser)
    addarg_show(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radial density profile of an ARTIS model."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    at.plottools.set_mpl_style()

    fig, axes = plt.subplots(
        nrows=3 if args.plotye else 2,
        ncols=1,
        sharex=True,
        sharey=False,
        figsize=(4, 4),
        tight_layout={"pad": 0.5, "w_pad": 0.5, "h_pad": 0.0},
    )
    assert isinstance(axes, np.ndarray)

    args.modelpath = at.normalize_path_list(args.modelpath)

    args.color, args.label = at.trim_or_pad(len(args.modelpath), args.color, args.label)
    args.label = [
        at.get_series_label(args.label, index, at.get_model_name(modelpath))
        for index, modelpath in enumerate(args.modelpath)
    ]

    max_vmax_on_c = float("-inf")
    for color, label, modelpath in zip(args.color, args.label, args.modelpath, strict=True):
        print(f"Plotting {label}")
        dfmodel, modelmeta = at.get_modeldata(modelpath, derived_cols=["vel_r_min", "vel_r_mid", "vel_r_max", "mass_g"])

        vmax_on_c = modelmeta["vmax_cmps"] / C_cm_per_s
        max_vmax_on_c = max(vmax_on_c, max_vmax_on_c)

        # total_mass = dfmodel.mass_g.sum() / Msun_to_g
        dfmodel = dfmodel.sort(by="vel_r_mid")

        cols = ["modelgridindex", "vel_r_min", "vel_r_mid", "vel_r_max", "mass_g"]
        if "Ye" in dfmodel.collect_schema().names():
            cols.append("Ye")

        dfmodelcollect = dfmodel.select(cols).collect()

        vuppers = dfmodelcollect.select(pl.col("vel_r_max").unique().sort()).to_series()
        enclosed_xvals = [0.0, *(vuppers / C_cm_per_s).to_list(), 1.0]
        enclosed_yvals = [0.0] + [
            float(dfmodelcollect.filter(pl.col("vel_r_mid") <= vupper)["mass_g"].sum()) / Msun_to_g
            for vupper in vuppers
        ]
        enclosed_yvals.append(float(dfmodelcollect["mass_g"].sum()) / Msun_to_g)
        axes[0].plot(enclosed_xvals, enclosed_yvals, label=label, color=color)

        if "vel_r_max_kmps" in dfmodel.collect_schema().names():
            # 1D spherical has a radial velocity specified
            vupperscoarse = vuppers.to_list()
        else:
            # 2D cylindrical or 3D Cartesian will have variable spacing in v_rad
            # so use the largest difference to set the bin size
            xmin = dfmodelcollect.select(pl.col("vel_r_mid").min()).item()
            # if we want to include the corners, then use this
            xmax = dfmodelcollect.select(pl.col("vel_r_mid").max()).item()
            # to exclude the corners:
            # xmax = modelmeta["vmax_cmps"]
            xdeltamax = dfmodelcollect.select(pl.col("vel_r_mid").sort().diff().max()).item()
            ncoarsevelbins = int((xmax - xmin) / xdeltamax)
            print(f"Using {ncoarsevelbins} velocity bins from {xmin} to {xmax} with max delta {xdeltamax}")
            vupperscoarse = [xmin + xdeltamax * (i + 1) for i in range(ncoarsevelbins)]

        if args.nbins:
            vupperscoarse = [(i + 1) * (C_cm_per_s / args.nbins) for i in range(args.nbins)]

        plotye = args.plotye and "Ye" in dfmodelcollect.columns
        binned_xvals: list[float] = []
        binned_massvals: list[float] = []
        binned_yevals: list[float] = []
        for vlower, vupper in zip([0.0, *vupperscoarse[:-1]], vupperscoarse, strict=True):
            assert vlower < vupper
            dfvelbin = dfmodelcollect.filter(pl.col("vel_r_mid").is_between(vlower, vupper, closed="left"))
            binned_xvals.extend((vlower / C_cm_per_s, vupper / C_cm_per_s))

            delta_beta = (vupper - vlower) / C_cm_per_s
            dmass_on_dbeta = float(dfvelbin["mass_g"].sum()) / Msun_to_g / delta_beta
            binned_massvals.extend((dmass_on_dbeta, dmass_on_dbeta))

            if plotye:
                ye = dfvelbin.select(pl.col("Ye").dot(pl.col("mass_g")) / pl.col("mass_g").sum()).item()
                binned_yevals.extend((ye, ye))

        # close the mass profile with a zero-valued step out to the edge of the plot
        axes[1].plot([*binned_xvals, binned_xvals[-1], 1.0], [*binned_massvals, 0.0, 0.0], label=label, color=color)
        if plotye:
            axes[2].plot(binned_xvals, binned_yevals, label=label, color=color)

    axes[0].set_xlim(left=0.0 if args.xmin is None else args.xmin)
    axes[0].set_xlim(right=max_vmax_on_c if args.xmax is None else args.xmax)

    axes[-1].set_xlabel(r"Velocity $\left[c\right]$")
    axes[0].set_ylabel(r"Mass Enclosed $\left[\mathrm{M}_\odot\right]$")
    axes[1].set_ylabel(r"$\Delta$M/$\Delta v$ $\left[\mathrm{M}_\odot/c\right]$")
    if args.plotye:
        axes[2].set_ylabel(r"Electron fraction Ye")
    axes[1].legend(frameon=False)

    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)

    outfilepath = at.resolve_outputfile(args.outputpath, "densityprofile.pdf")

    save_figure(fig, outfilepath, show=args.show, openfile=args.open)


if __name__ == "__main__":
    main()
