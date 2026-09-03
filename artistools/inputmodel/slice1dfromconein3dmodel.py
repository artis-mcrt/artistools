"""Extract a 1D ARTIS model from a cone around one axis of a 3D model."""

import argparse
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output

CONE_DERIVED_COLS = [
    "volume",
    "pos_x_mid",
    "pos_y_mid",
    "pos_z_mid",
    "pos_x_min",
    "pos_y_min",
    "pos_z_min",
    "pos_r_mid",
    "mass_g",
    "pos_r_min",
]


def make_cone(args: argparse.Namespace, dfmodel: pl.LazyFrame, logprint: Callable[..., None]) -> pl.DataFrame:
    """Return the cells of the 3D model lying within args.coneangle of the chosen axis."""
    print("Making cone")

    angle_of_cone = args.coneangle  # in deg
    logprint(f"Using cone angle of {angle_of_cone} degrees")

    theta = np.radians([angle_of_cone / 2])  # angle between line of sight and edge is half angle of cone

    print(f"using {'positive' if args.positive_axis else 'negative'} axis")
    radial = (
        1.0
        / (np.tan(theta))
        * (pl.col(f"pos_{args.other_axis2}_mid") ** 2 + pl.col(f"pos_{args.other_axis1}_mid") ** 2).sqrt()
    )
    axiscol = pl.col(f"pos_{args.sliceaxis}_mid")
    cone = dfmodel.filter(axiscol >= radial if args.positive_axis else axiscol <= -radial)

    return cone.collect()


def get_profile_along_axis(dfmodel: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    """Return the cells of the 3D model running along the chosen axis, nearest to the other two axes' origin."""
    print("Getting profile along axis")

    argmin = dfmodel[f"pos_{args.other_axis2}_min"].abs().arg_min()
    assert argmin is not None
    position_closest_to_axis = dfmodel[f"pos_{args.other_axis2}_min"].item(argmin)

    # the innermost cell on the positive axis has pos_min == 0, thus the condition must keep it
    sliceaxis_cond = (
        (pl.col(f"pos_{args.sliceaxis}_min") >= 0) if args.positive_axis else (pl.col(f"pos_{args.sliceaxis}_min") < 0)
    )

    return dfmodel.filter(
        (pl.col(f"pos_{args.other_axis1}_min") == position_closest_to_axis)
        & (pl.col(f"pos_{args.other_axis2}_min") == position_closest_to_axis)
        & sliceaxis_cond
    )


def get_cone_shells(
    cone: pl.DataFrame, cone1d_bins: Sequence[float], speciescols: Sequence[str], logprint: Callable[..., None]
) -> list[dict[str, float]]:
    """Return the density and the normalised composition of each spherical shell of the cone.

    The shells end at the first empty shell, because the outer shells have no mass.
    """
    nshells = len(cone1d_bins) - 1
    # a cell belongs to the shell with cone1d_bins[i] <= pos_r_mid < cone1d_bins[i + 1]
    shellindex = np.searchsorted(np.asarray(cone1d_bins), cone["pos_r_mid"].to_numpy(), side="right") - 1
    dfshells = (
        cone
        .with_columns(shellindex=pl.Series(shellindex, dtype=pl.Int64))
        .filter(pl.col("shellindex").is_between(0, nshells - 1))
        .group_by("shellindex")
        .agg(
            # mass of each species in each 3D grid cell, summed over the cells
            *[(pl.col(species) * pl.col("mass_g")).sum().alias(species) for species in speciescols],
            cellcount=pl.len(),
            total_mass_g=pl.col("mass_g").sum(),
            total_volume=pl.col("volume").sum(),
        )
        .join(
            pl.DataFrame({"shellindex": range(nshells)}, schema={"shellindex": pl.Int64}),
            on="shellindex",
            how="right",
            maintain_order="right",
        )
        .with_columns(cs.float().fill_null(0.0), cellcount=pl.col("cellcount").fill_null(0))
    )

    shellrows: list[dict[str, float]] = []
    for i, shell in enumerate(dfshells.iter_rows(named=True)):
        total_mass_g = float(shell["total_mass_g"])
        total_volume = float(shell["total_volume"])

        if total_mass_g <= 0:
            assert total_volume > 0, (
                f"\nAssertion Error: No cell midpoints within cone limits for shell {i + 1}.\n"
                "The small volume contained within the cone for the innermost shell means this is quite likely to\n"
                "occur, especially for smaller cone angles and grid spacings where the inner shell radius is\n"
                "small. Also more likely to occur when ncoordgrid/2 is even, resulting in the slice axis being\n"
                "along cell minimums not cell midpoints in the 3D model. If this occurs you can either choose a \n"
                "different grid spacing (using -coneshellspacingexponent or -nshells) or increase -coneangle\n"
                "to ensure at least one cell midpoint is contained within the cone limits of the shell\n"
            )
            # Cells exist but all have density=0. The warning goes to the standard error, because
            # --quiet hides the standard output alone, and this warning names shells that go away.
            logprint(
                f"WARNING: Shell {i + 1} is empty (all 3D grid cells averaged in the shell must have density=0)."
                "This shell and all shells further out in the model will be removed from the model.\n"
                "This is safe provided this empty shell is far enough out in the model: check model file to \n"
                "confirm this is the case. If not there may be an issue with the model being read in.\n"
                "The outer regions of some models can have empty regions before there are more non-empty cells\n"
                "again at higher velocities. This should generally be in the very outer regions of models where\n"
                "the cells are too optically thin to impact the synthetic observables. However if you want cells\n"
                "in these outer regions to be included in the 1D cone can experiment with -coneangle,-nshells and\n"
                "-coneshellspacingexponent to ensure the shells for these outer regions include some non-empty 3D\n"
                "grid cell and thus the shells can be included in the 1D model.\n",
                file=sys.stderr,
            )
            break

        # the species mass summed over the cells, divided by the shell mass
        composition = {species: shell[species] / total_mass_g for species in speciescols}

        # Sum all composition values to ensure compositions are normalised to 1 in 3D model
        if i == 0:
            logprint(
                "\nSumming all mass weighted compositions in the shells. If these values significantly\n"
                "deviate from 1 there could be an issue with the input model. The compositions for each\n"
                "shell in the output 1D model are normalised here regardless of how close to 1 they are.\n"
                "Also printing how many 3D cells make up each 1D shell in the model generated.\n\n"
                "NOTE: the compositions do not always sum exactly to 1 in the 3D model grid cells.\n"
                "From limited testing this appears to be most pronounced in the outer cells of the 3D\n"
                "models where the composition sum can deviate by ~1% from 1 when averaging the 3D cells\n"
                "into the shells in the 1D model. The composition is normalised before writing out the \n"
                "1D model but worth checking the log file to ensure the normalisation of the cells in the 3D \n"
                "model used in the 1D model shells is close to 1 before this\n"
            )
        # Skipping first 5 columns which contain the radioisotopes utilised in SN models
        # the remaining columns contain the 30 elements in the composition file for SN models
        # which have the radioisotopes already included in the composition total for the
        # relevant elements
        sum_composition_check = sum(composition[species] for species in speciescols[5:])
        logprint(
            f"Shell {i + 1:<3}     3D cells averaged: {shell['cellcount']:<6} composition sum before norm: {sum_composition_check}"
        )
        composition = {species: massfrac / sum_composition_check for species, massfrac in composition.items()}

        shellrows.append(
            {"inputcellid": i + 1, "r_bin_max_boundary": cone1d_bins[i + 1], "rho": total_mass_g / total_volume}
            | composition
        )

    return shellrows


def make_1d_profile(args: argparse.Namespace, logprint: Callable[..., None]) -> pl.DataFrame:
    """Make 1D model from 3D model."""
    modelpath = at.normalize_path_list(args.modelpath)[0]
    logprint("Making 1D model from 3D model:", at.get_model_name(modelpath))
    pldfmodel, modelmeta = at.get_modeldata(
        modelpath=modelpath, get_elemabundances=True, derived_cols=CONE_DERIVED_COLS if args.makefromcone else None
    )
    args.t_model = modelmeta["t_model_init_days"]
    if args.makefromcone:
        logprint("from a cone")
        cone = make_cone(args, pldfmodel, logprint)
        N_shells = args.nshells
        # Max radius that still ensures a full shell as the cartesian grid means some
        # radius values will be greater than the max radius of the axis the cone is centred on
        r_max = modelmeta["wid_init"] * (modelmeta["ncoordgrid"] / 2)

        if args.coneshellsequalvolume:
            logprint("Spacing shells in 1D model so they have equal volume")
            V_total = (4 / 3) * np.pi * r_max**3
            cone1d_bins: list[float] = []
            for i in range(N_shells):
                r_inner = 0 if i == 0 else cone1d_bins[i - 1]
                r_outer = ((3 * V_total) / (4 * np.pi * N_shells) + r_inner**3) ** (1 / 3)
                cone1d_bins.append(r_outer)
            cone1d_bins.insert(0, 0.0)
        else:
            shell_spacing_power = args.coneshellspacingexponent  # Change this to get the desired velocity bin spacing
            logprint(f"Spacing shells in 1D model so they are equally spaced on a radius^{shell_spacing_power} grid")
            cone_radius_spacing = np.linspace(0, r_max**shell_spacing_power, N_shells + 1)
            cone1d_bins = list(np.power(cone_radius_spacing, (1 / shell_spacing_power)))
        speciescols = [colname for colname in cone.columns if colname.startswith("X_")]
        shellrows = get_cone_shells(cone, cone1d_bins, speciescols, logprint)

        # Combine all bin results into a single DataFrame
        slice1d = pl.DataFrame(shellrows)
        slice1d = slice1d.with_columns(pl.col("r_bin_max_boundary") / (args.t_model * day_to_s * km_to_cm)).rename({
            "r_bin_max_boundary": "vel_r_max_kmps"
        })

    else:  # make from along chosen axis
        logprint("from along the axis")
        slice1d = get_profile_along_axis(pldfmodel.collect(), args)
        # pos_min is the inner edge of a cell. On the positive axis, the outer edge is pos_min plus the
        # cell width of the slice axis. On the negative axis, pos_min is already the outer edge, and
        # the reverse and negate step below makes the velocities positive.
        pos_outer = (
            pl.col(f"pos_{args.sliceaxis}_min") + modelmeta[f"wid_init_{args.sliceaxis}"]
            if args.positive_axis
            else pl.col(f"pos_{args.sliceaxis}_min")
        )
        slice1d = (
            # Convert positions to velocities
            slice1d
            .with_columns((pos_outer / (args.t_model * day_to_s * km_to_cm)).alias(f"pos_{args.sliceaxis}_min"))
            .rename({f"pos_{args.sliceaxis}_min": "vel_r_max_kmps"})
            # Remove columns we don't need
            .drop("inputcellid", f"pos_{args.other_axis1}_min", f"pos_{args.other_axis2}_min")
        )
    logprint("using axis:", args.axis)

    if args.rhoscale:
        logprint("Scaling density by a factor of:", args.rhoscale)
        slice1d = slice1d.with_columns(pl.col("rho") * args.rhoscale)

    # slice1d = slice1d[slice1d['rho_model'] != -100]  # Remove empty cells
    # TODO: fix this, -100 probably breaks things if it's not one of the outer cells that gets chopped
    slice1d = slice1d.with_columns(
        pl.when(pl.col("rho") != 0).then(pl.col("rho").log10()).otherwise(-100).alias("rho")
    ).rename({"rho": "logrho"})

    if not args.positive_axis and not args.makefromcone:
        # Invert rows and *velocity by -1 to make velocities positive for slice on negative axis
        slice1d = slice1d.reverse().with_columns(pl.col("vel_r_max_kmps") * -1)

    return slice1d


def make_1d_model_files(args: argparse.Namespace, logprint: Callable[..., None]) -> None:
    """Write the 1D model.txt and abundances.txt for the extracted profile."""
    slice1d = make_1d_profile(args, logprint)

    abundancecolumns = [
        column
        for column in slice1d.columns
        if column.startswith("X_") and not any(i.isdigit() for i in column) and len(column) < 5
    ]

    npts_model = slice1d.height
    inputcellid = pl.Series("inputcellid", np.arange(1, npts_model + 1))

    model_df = slice1d.drop(abundancecolumns).with_columns(inputcellid)
    abundances_df = slice1d.select(abundancecolumns).with_columns(inputcellid)

    at.inputmodel.save_modeldata(
        dfmodel=model_df, t_model_init_days=args.t_model, outpath=Path(args.outputfile, "model_1d.txt")
    )

    at.inputmodel.save_initelemabundances(abundances_df, outpath=Path(args.outputfile, "abundances_1d.txt"))

    print("Saved abundances_1d.txt and model_1d.txt")


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(
        parser, multiplepaths=True, default=[], helptext="Path to ARTIS model folders with model.txt and abundances.txt"
    )

    parser.add_argument(
        "-axis",
        default="+x",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        help="Slice axis. Hint: for negative use e.g. -axis=-z",
    )

    parser.add_argument(
        "--makefromcone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make 1D model from a cone around the axis (--no-makefromcone samples points along the axis instead)",
    )

    parser.add_argument(
        "-coneangle", type=float, default=30.0, help="Cone angle in degrees, cone half angle given by coneangle/2"
    )

    parser.add_argument(
        "-nshells",
        type=int,
        default=100,
        help="Number of shells used when making 1D model from cone. Note the final number of shells may be lower as empty outer shells are removed from the output 1D model files",
    )

    parser.add_argument(
        "-coneshellspacingexponent",
        type=float,
        default=1.5,
        help="Vary the exponent used when selecting the radius dependence of the shell spacing when making 1D model from cone. By default the shells are spaced evenly in radius^(1.5)",
    )

    parser.add_argument(
        "--coneshellsequalvolume", action="store_true", help="Use equal volume shells when making 1D model from cone"
    )

    addarg_output(parser, kind="folder", default=Path())

    parser.add_argument("-rhoscale", "-v", default=None, type=float, help="Density scale factor")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Make 1D model from cone in 3D model."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    args.modelpath = at.normalize_path_list(args.modelpath)

    args.sliceaxis = args.axis[1]
    assert args.axis[0] in {"+", "-"}
    args.positive_axis = args.axis[0] == "+"

    print(f"making model from slice around {args.axis} axis")

    axes = ["x", "y", "z"]
    args.other_axis1 = next(ax for ax in axes if ax != args.sliceaxis)
    args.other_axis2 = next(ax for ax in axes if ax not in {args.sliceaxis, args.other_axis1})

    # remember: models before scaling down to artis input have x and z axis swapped compared to artis input files

    logprint = at.inputmodel.inputmodel_misc.savetologfile(
        outputfolderpath=Path(args.outputfile), logfilename="make1dmodellog.txt"
    )

    make_1d_model_files(args, logprint)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
