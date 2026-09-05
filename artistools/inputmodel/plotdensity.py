# PYTHON_ARGCOMPLETE_OK
"""Plot mass density against velocity for one or more ARTIS input models."""

import argparse
import typing as t
from collections.abc import Sequence

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import Msun_to_g
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_figscale
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_seriesstyle
from artistools.misc import addarg_show
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend


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

    addarg_output(parser, kind="file", defaultname="densityprofile.pdf")

    addarg_figscale(parser, helptext="Scale factor for plot area. 1.0 fills one column of a page")
    addarg_show(parser)


PROFILE_COLUMNS = ("modelgridindex", "vel_r_min", "vel_r_mid", "vel_r_max", "vel_r_max_kmps", "mass_g", "Ye")


def get_enclosed_mass(dfmodel: pl.DataFrame) -> tuple[list[float], list[float]]:
    """Return the velocity [c] and the enclosed mass [Msun] at each outer cell velocity, from zero to c."""
    dfsorted = dfmodel.sort("vel_r_mid").with_columns(enclosed_mass_g=pl.col("mass_g").cum_sum())
    vuppers = dfsorted.select(pl.col("vel_r_max").unique().sort()).to_series()

    # the enclosed mass at vupper is the cumulative sum at the last cell with vel_r_mid <= vupper
    cellcount_below = np.searchsorted(dfsorted["vel_r_mid"].to_numpy(), vuppers.to_numpy(), side="right")
    enclosed_mass = np.concatenate(([0.0], dfsorted["enclosed_mass_g"].to_numpy()))[cellcount_below] / Msun_to_g
    total_mass_msun = float(dfsorted["mass_g"].sum()) / Msun_to_g

    return [0.0, *(vuppers / C_cm_per_s).to_list(), 1.0], [0.0, *enclosed_mass.tolist(), total_mass_msun]


def get_binned_profile(
    dfmodel: pl.DataFrame, vupperscoarse: Sequence[float], plotye: bool
) -> tuple[list[float], list[float], list[float]]:
    """Return the step-plot velocities [c], dM/dv [Msun/c], and the mass-weighted Ye of each velocity bin."""
    binedges = np.array([0.0, *vupperscoarse])
    assert np.all(np.diff(binedges) > 0)
    nbins = len(vupperscoarse)

    binindex = np.searchsorted(binedges, dfmodel["vel_r_mid"].to_numpy(), side="right") - 1
    dfbinned = (
        dfmodel
        .with_columns(binindex=pl.Series(binindex, dtype=pl.Int64))
        .filter(pl.col("binindex").is_between(0, nbins - 1))
        .group_by("binindex")
        .agg(mass_g=pl.col("mass_g").sum(), ye_mass_g=(pl.col("Ye").dot(pl.col("mass_g")) if plotye else pl.lit(0.0)))
        .join(pl.DataFrame({"binindex": range(nbins)}, schema={"binindex": pl.Int64}), on="binindex", how="right")
        .sort("binindex")
        .fill_null(0.0)
        .with_columns(
            dmass_on_dbeta=pl.col("mass_g") / Msun_to_g / (pl.Series(np.diff(binedges)) / C_cm_per_s),
            ye=pl.col("ye_mass_g") / pl.col("mass_g"),
        )
    )

    # each bin gives two points, so that the profile is a step plot
    binned_xvals = np.repeat(binedges, 2)[1:-1] / C_cm_per_s
    binned_massvals = np.repeat(dfbinned["dmass_on_dbeta"].to_numpy(), 2)
    binned_yevals = np.repeat(dfbinned["ye"].to_numpy(), 2) if plotye else np.array([])

    return binned_xvals.tolist(), binned_massvals.tolist(), binned_yevals.tolist()


def get_coarse_velocity_bins(dfmodel: pl.DataFrame, nbins: int | None) -> list[float]:
    """Return the upper velocities [cm/s] of the bins for the dM/dv profile."""
    if nbins:
        return [(i + 1) * (C_cm_per_s / nbins) for i in range(nbins)]

    if "vel_r_max_kmps" in dfmodel.columns:
        # 1D spherical has a radial velocity specified
        return dfmodel.select(pl.col("vel_r_max").unique().sort()).to_series().to_list()

    # a 2D or 3D model has a variable spacing in the radial velocity, thus the largest step sets the bin size
    xmin, xmax, xdeltamax = dfmodel.select(
        pl.col("vel_r_mid").min().alias("xmin"),
        pl.col("vel_r_mid").max().alias("xmax"),
        pl.col("vel_r_mid").sort().diff().max().alias("xdeltamax"),
    ).row(0)
    ncoarsevelbins = int((xmax - xmin) / xdeltamax)
    print(f"Using {ncoarsevelbins} velocity bins from {xmin} to {xmax} with max delta {xdeltamax}")
    return [xmin + xdeltamax * (i + 1) for i in range(ncoarsevelbins)]


def plot_density_profiles(args: argparse.Namespace, axes: npt.NDArray[np.object_] | Sequence[mplax.Axes]) -> float:
    """Plot the enclosed mass, dM/dv, and Ye profile of each model, and return the largest vmax [c]."""
    max_vmax_on_c = float("-inf")
    for color, label, modelpath in zip(args.color, args.label, args.modelpath, strict=True):
        print(f"Plotting {label}")
        lzdfmodel, modelmeta = at.get_modeldata(
            modelpath, derived_cols=["vel_r_min", "vel_r_mid", "vel_r_max", "mass_g"]
        )

        vmax_on_c = modelmeta["vmax_cmps"] / C_cm_per_s
        max_vmax_on_c = max(vmax_on_c, max_vmax_on_c)

        dfmodel = lzdfmodel.select(cs.by_name(*PROFILE_COLUMNS, require_all=False)).collect()

        enclosed_xvals, enclosed_yvals = get_enclosed_mass(dfmodel)
        axes[0].plot(enclosed_xvals, enclosed_yvals, label=label, color=color)

        vupperscoarse = get_coarse_velocity_bins(dfmodel, args.nbins)
        plotye = args.plotye and "Ye" in dfmodel.columns
        binned_xvals, binned_massvals, binned_yevals = get_binned_profile(dfmodel, vupperscoarse, plotye)

        # close the mass profile with a zero-valued step out to the edge of the plot
        axes[1].plot([*binned_xvals, binned_xvals[-1], 1.0], [*binned_massvals, 0.0, 0.0], label=label, color=color)
        if plotye:
            axes[2].plot(binned_xvals, binned_yevals, label=label, color=color)

    return max_vmax_on_c


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radial density profile of an ARTIS model."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    fig, axesgrid = make_frame_figure(args, rows=3 if args.plotye else 2, aspect=0.45, fullwidth=False)
    axes = axesgrid[:, 0]

    args.modelpath = at.normalize_path_list(args.modelpath)

    args.color, args.label = at.trim_or_pad(len(args.modelpath), args.color, args.label)
    args.label = [
        at.get_series_label(args.label, index, at.get_model_name(modelpath))
        for index, modelpath in enumerate(args.modelpath)
    ]

    max_vmax_on_c = plot_density_profiles(args, axes)

    axes[0].set_xlim(left=0.0 if args.xmin is None else args.xmin)
    axes[0].set_xlim(right=max_vmax_on_c if args.xmax is None else args.xmax)

    axes[-1].set_xlabel(r"Velocity $\left[c\right]$")
    axes[0].set_ylabel(r"Mass Enclosed $\left[\mathrm{M}_\odot\right]$")
    axes[1].set_ylabel(r"$\Delta$M/$\Delta v$ $\left[\mathrm{M}_\odot/c\right]$")
    if args.plotye:
        axes[2].set_ylabel(r"Electron fraction Ye")
    set_legend(axes[1], args)

    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)

    save_figure(fig, args.outputfile, args=args)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
