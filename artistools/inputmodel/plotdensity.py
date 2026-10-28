"""Plot mass density against velocity for one or more ARTIS input models."""

import argparse
import math
import typing as t
from collections.abc import Sequence

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

from artistools.constants import C_cm_per_s
from artistools.constants import Msun_to_g
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import scan_modeldata
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_darkmode
from artistools.misc import addarg_data_output
from artistools.misc import addarg_figscale
from artistools.misc import addarg_legend
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import addarg_seriesstyle
from artistools.misc import addarg_show
from artistools.misc import exit_with_error
from artistools.misc import get_model_name
from artistools.misc import get_series_label
from artistools.misc import is_in_range
from artistools.misc import normalize_path_list
from artistools.misc import output_data
from artistools.misc import parse_cli_args
from artistools.misc import trim_or_pad
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_legend


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_legend(parser)
    addarg_modelpath(
        parser,
        positional=True,
        multiplepaths=True,
        default=[],
        helptext="Path(s) to model.txt file(s) or folders containing model.txt",
    )

    addarg_seriesstyle(
        parser, colordefault=[f"C{i}" for i in range(10)], include_linestyles=False, include_dashes=False
    )

    addarg_axislimits(parser, include_y=False)

    parser.add_argument(
        "-nbins",
        type=int,
        default=None,
        help="Use this number of fixed velocity bins from zero to the larger of the model vmax and --xmax",
    )

    parser.add_argument("--plotye", action="store_true", help="Plot electron fraction versus velocity")

    addarg_output(parser, kind="file", defaultname="densityprofile.pdf")

    addarg_figscale(parser, helptext="Scale factor for plot area. 1.0 fills one column of a page")
    addarg_data_output(parser)
    addarg_show(parser)
    addarg_darkmode(parser)


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


def get_coarse_velocity_bins(dfmodel: pl.DataFrame, nbins: int | None, vmax_cmps: float) -> list[float]:
    """Return the upper velocities [cm/s] of the bins for the dM/dv profile.

    vmax_cmps is the largest velocity of the plot. The fixed bins of --nbins fill that range. Bins that
    reach the speed of light would put the model into the first few bins.
    """
    if nbins:
        return [(i + 1) * (vmax_cmps / nbins) for i in range(nbins)]

    if "vel_r_max_kmps" in dfmodel.columns:
        # 1D spherical has a radial velocity specified
        return dfmodel.select(pl.col("vel_r_max").unique().sort()).to_series().to_list()

    # a 2D or 3D model has a variable spacing in the radial velocity, thus the largest step sets the bin size
    xmin, xmax, xdeltamax = dfmodel.select(
        pl.col("vel_r_mid").min().alias("xmin"),
        pl.col("vel_r_mid").max().alias("xmax"),
        pl.col("vel_r_mid").sort().diff().max().alias("xdeltamax"),
    ).row(0)
    if not xdeltamax:
        # every cell has the same mid-point velocity, e.g. a grid of 2 x 2 x 2 cells, or the grid has one cell.
        # One bin to the outer corner of the cells then holds the whole model
        return [dfmodel.select(pl.col("vel_r_max").max()).item()]

    # the bins are open at the upper edge, thus the last bin must end above the outermost cell
    ncoarsevelbins = math.floor((xmax - xmin) / xdeltamax) + 1
    print(f"Using {ncoarsevelbins} velocity bins from {xmin} to {xmax} with max delta {xdeltamax}")
    return [xmin + xdeltamax * (i + 1) for i in range(ncoarsevelbins)]


def plot_density_profiles(
    args: argparse.Namespace, axes: npt.NDArray[np.object_] | Sequence[mplax.Axes]
) -> tuple[float, dict[str, pl.DataFrame]]:
    """Plot the enclosed mass, dM/dv, and Ye profile of each model, and return the largest vmax [c] and the tables.

    The tables are empty if the command gives no data output. The enclosed mass and the binned profile have different
    velocity grids, thus each one has its own table.
    """
    makestable = getattr(args, "print_data", False) or getattr(args, "write_data", False)
    dfenclosedmass: list[pl.DataFrame] = []
    dfmassprofiles: list[pl.DataFrame] = []
    max_vmax_on_c = float("-inf")
    for color, label, modelpath in zip(args.color, args.label, args.modelpath, strict=True):
        print(f"Plotting {label}")
        lzdfmodel, modelmeta = scan_modeldata(modelpath)
        lzdfmodel = add_derived_cols_to_modeldata(lzdfmodel, modelmeta=modelmeta)

        vmax_on_c = modelmeta["vmax_cmps"] / C_cm_per_s
        max_vmax_on_c = max(vmax_on_c, max_vmax_on_c)

        dfmodel = lzdfmodel.select(cs.by_name(*PROFILE_COLUMNS, require_all=False)).collect()

        enclosed_xvals, enclosed_yvals = get_enclosed_mass(dfmodel)
        axes[0].plot(enclosed_xvals, enclosed_yvals, label=label, color=color)

        # the bins cover the whole model, thus -xmax changes the axis only. A -xmax above the model
        # vmax gives bins to the edge of the axis, and a lower one leaves the outer cells in their bins
        vmax_cmps = max(modelmeta["vmax_cmps"], (args.xmax or 0.0) * C_cm_per_s)
        vupperscoarse = get_coarse_velocity_bins(dfmodel, args.nbins, vmax_cmps)
        plotye = args.plotye and "Ye" in dfmodel.columns
        binned_xvals, binned_massvals, binned_yevals = get_binned_profile(dfmodel, vupperscoarse, plotye)

        # close the mass profile with a zero-valued step out to the edge of the plot
        massprofile_xvals = [*binned_xvals, binned_xvals[-1], 1.0]
        massprofile_yvals = [*binned_massvals, 0.0, 0.0]
        axes[1].plot(massprofile_xvals, massprofile_yvals, label=label, color=color)
        if plotye:
            axes[2].plot(binned_xvals, binned_yevals, label=label, color=color)

        if makestable:
            # two models can have one label, thus the path names the model of each row and keeps the lines apart
            pathcolumn = pl.lit(str(modelpath)).alias("modelpath")
            dfenclosedmass.append(
                pl.DataFrame(
                    {"model": label, "vel_r_on_c": enclosed_xvals, "mass_enclosed_msun": enclosed_yvals},
                    schema={"model": pl.String, "vel_r_on_c": pl.Float64, "mass_enclosed_msun": pl.Float64},
                ).select("model", pathcolumn, pl.exclude("model"))
            )
            dfmassprofiles.append(
                get_mass_profile_table(
                    label, massprofile_xvals, massprofile_yvals, binned_yevals if plotye else None
                ).select("model", pathcolumn, pl.exclude("model"))
            )

    tables = {
        "": pl.concat(dfenclosedmass) if dfenclosedmass else pl.DataFrame(),
        "_massprofile": pl.concat(dfmassprofiles, how="diagonal_relaxed") if dfmassprofiles else pl.DataFrame(),
    }
    return max_vmax_on_c, tables


def get_mass_profile_table(
    label: str, xvals: Sequence[float], massvals: Sequence[float], yevals: Sequence[float] | None
) -> pl.DataFrame:
    """Return the points of the dM/dv curve of one model, and the Ye of each point if the Ye curve exists.

    The Ye curve has no points for the closing step of the dM/dv curve, thus Ye is null at these two points.
    """
    dfprofile = pl.DataFrame(
        {"model": label, "vel_r_on_c": xvals, "dmass_dv_msun_per_c": massvals},
        schema={"model": pl.String, "vel_r_on_c": pl.Float64, "dmass_dv_msun_per_c": pl.Float64},
    )
    if yevals is None:
        return dfprofile
    return dfprofile.with_columns(Ye=pl.Series([*yevals, *[None] * (len(xvals) - len(yevals))], dtype=pl.Float64))


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot the radial density profile of an ARTIS model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    fig, axesgrid = make_frame_figure(args, rows=3 if args.plotye else 2, aspect=0.45, fullwidth=False)
    axes = axesgrid[:, 0]

    args.modelpath = normalize_path_list(args.modelpath)

    args.color, args.label = trim_or_pad(len(args.modelpath), args.color, args.label)
    args.label = [
        get_series_label(args.label, index, get_model_name(modelpath)) for index, modelpath in enumerate(args.modelpath)
    ]

    if args.xmax is not None and args.xmax <= 0.0:
        exit_with_error(f"--xmax {args.xmax} gives no range", "Give an --xmax above 0, in units of c")

    max_vmax_on_c, tables = plot_density_profiles(args, axes)

    xlimits = (0.0 if args.xmin is None else args.xmin, max_vmax_on_c if args.xmax is None else args.xmax)
    axes[0].set_xlim(*xlimits)

    axes[-1].set_xlabel(r"Velocity $\left[c\right]$")
    axes[0].set_ylabel(r"Mass Enclosed $\left[\mathrm{M}_\odot\right]$")
    axes[1].set_ylabel(r"$\Delta$M/$\Delta v$ $\left[\mathrm{M}_\odot/c\right]$")
    if args.plotye:
        axes[2].set_ylabel(r"Electron fraction Ye")
    set_legend(axes[1], args)

    axes[0].set_ylim(bottom=0.0)
    axes[1].set_ylim(bottom=0.0)

    # the bins cover the whole model, thus the tables take the velocities of -xmin and -xmax here
    output_data(
        args,
        {
            suffix: dftable.filter(is_in_range("vel_r_on_c", args.xmin, args.xmax))
            if not dftable.is_empty()
            else dftable
            for suffix, dftable in tables.items()
        },
        args.outputfile,
    )
    save_figure(fig, args.outputfile, args=args)
