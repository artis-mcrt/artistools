"""Extract a 1D ARTIS model from a cone around one axis of a 3D model."""

import argparse
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import artistools as at
from artistools.constants import day_to_s
from artistools.constants import km_to_cm
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_outputpath

if t.TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


def make_cone(args: argparse.Namespace, logprint: Callable[..., None]) -> pl.DataFrame:
    """Return the cells of the 3D model lying within args.coneangle of the chosen axis."""
    print("Making cone")

    angle_of_cone = args.coneangle  # in deg
    logprint(f"Using cone angle of {angle_of_cone} degrees")

    theta = np.radians([angle_of_cone / 2])  # angle between line of sight and edge is half angle of cone

    pldfmodel, modelmeta = at.get_modeldata(
        modelpath=args.modelpath[0],
        get_elemabundances=True,
        derived_cols=[
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
        ],
    )
    dfmodel = pldfmodel
    args.t_model = modelmeta["t_model_init_days"]

    print(f"using {'positive' if args.positive_axis else 'negative'} axis")
    radial = (
        1.0
        / (np.tan(theta))
        * (pl.col(f"pos_{args.other_axis2}_mid") ** 2 + pl.col(f"pos_{args.other_axis1}_mid") ** 2).sqrt()
    )
    axiscol = pl.col(f"pos_{args.sliceaxis}_mid")
    cone = dfmodel.filter(axiscol >= radial if args.positive_axis else axiscol <= -radial)

    return cone.collect()


def get_profile_along_axis(
    args: argparse.Namespace, modeldata: pl.DataFrame | None = None, derived_cols: Sequence[str] | None = None
) -> pl.DataFrame:
    """Return the cells of the 3D model running along the chosen axis, nearest to the other two axes' origin."""
    print("Getting profile along axis")

    if modeldata is None:
        modeldata = at.inputmodel.get_modeldata(args.modelpath, get_elemabundances=True, derived_cols=derived_cols)[
            0
        ].collect()

    argmin = modeldata[f"pos_{args.other_axis2}_min"].abs().arg_min()
    assert argmin is not None
    position_closest_to_axis = modeldata[f"pos_{args.other_axis2}_min"].item(argmin)

    sliceaxis_cond = (
        (pl.col(f"pos_{args.sliceaxis}_min") > 0) if args.positive_axis else (pl.col(f"pos_{args.sliceaxis}_min") < 0)
    )

    return modeldata.filter(
        (pl.col(f"pos_{args.other_axis1}_min") == position_closest_to_axis)
        & (pl.col(f"pos_{args.other_axis2}_min") == position_closest_to_axis)
        & sliceaxis_cond
    )


def make_1d_profile(args: argparse.Namespace, logprint: Callable[..., None]) -> pl.DataFrame:
    """Make 1D model from 3D model."""
    logprint("Making 1D model from 3D model:", at.get_model_name(args.modelpath[0]))
    _, modelmeta = at.get_modeldata(modelpath=args.modelpath[0])
    if args.makefromcone:
        logprint("from a cone")
        cone = make_cone(args, logprint)
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
        shellrows: list[dict[str, float]] = []
        for i in range(len(cone1d_bins) - 1):
            # Filter cells within bin
            cells_within_bin = cone.filter(
                (pl.col("pos_r_mid") >= cone1d_bins[i]) & (pl.col("pos_r_mid") < cone1d_bins[i + 1])
            )
            # Sum mass and volume for each of the 3D cells that are being included in this 1D shell
            total_mass_g = float(cells_within_bin["mass_g"].sum())
            total_volume = float(cells_within_bin["volume"].sum())

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
                # Cells exist but all have density=0
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
                    "grid cell and thus the shells can be included in the 1D model.\n"
                )
                break

            # mass of each species in each 3D grid cell, summed over the cells and divided by the shell mass
            species_total_mass = cells_within_bin.select((pl.col(speciescols) * pl.col("mass_g")).sum())
            composition = {species: species_total_mass[species].item() / total_mass_g for species in speciescols}

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
                f"Shell {i + 1:<3}     3D cells averaged: {cells_within_bin.height:<6} composition sum before norm: {sum_composition_check}"
            )
            composition = {species: massfrac / sum_composition_check for species, massfrac in composition.items()}

            # Append results for this bin to the overall results
            shellrows.append(
                {"inputcellid": i + 1, "r_bin_max_boundary": cone1d_bins[i + 1], "rho": total_mass_g / total_volume}
                | composition
            )

        # Combine all bin results into a single DataFrame
        slice1d = pl.DataFrame(shellrows)
        slice1d = slice1d.with_columns(pl.col("r_bin_max_boundary") / (args.t_model * day_to_s * km_to_cm)).rename({
            "r_bin_max_boundary": "vel_r_max_kmps"
        })

    else:  # make from along chosen axis
        logprint("from along the axis")
        slice1d = get_profile_along_axis(args)
        slice1d = (
            # Convert positions to velocities
            slice1d
            .with_columns(pl.col(f"pos_{args.sliceaxis}_min") / (args.t_model * day_to_s * km_to_cm))
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
        dfmodel=model_df, t_model_init_days=args.t_model, outpath=Path(args.outputpath, "model_1d.txt")
    )

    at.inputmodel.save_initelemabundances(abundances_df, outpath=Path(args.outputpath, "abundances_1d.txt"))

    # with Path(args.modelpath[0], "model_1d.txt").open("r+") as f:  # add number of cells and tmodel to start of file
    #     content = f.read()
    #     f.seek(0, 0)
    #     f.write(f"{model_df.shape[0]}\n{args.t_model}".rstrip("\r\n") + "\n" + content)

    print("Saved abundances_1d.txt and model_1d.txt")


# with open(args.modelpath[0]/"model

# print(cone)

# cone = (merge_dfs.loc[merge_dfs[f'pos_{args.other_axis2}'] <= - (1/(np.tan(theta))
# * np.sqrt((merge_dfs[f'pos_{slice_on_axis}'])**2 + (merge_dfs[f'pos_{args.other_axis1}'])**2))])
# cone = merge_dfs
# cone = cone.loc[cone['rho_model'] > 0.0]


def make_plot(args: argparse.Namespace, logprint: Callable[..., None]) -> None:
    """Show a 3D scatter plot of the cone cells, coloured by density."""
    cone = make_cone(args, logprint)

    cone = cone.filter(pl.col("rho_model") > 0.0002)  # cut low densities (empty cells?) from plot
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(projection="3d")  # type: ignore[no-any-unimported]

    # print(cone['rho_model'])

    # set up for big model. For scaled down artis input model switch x and z
    x = cone["pos_z_min"] / km_to_cm / (args.t_model * day_to_s) / 1e3
    y = cone["pos_y_min"] / km_to_cm / (args.t_model * day_to_s) / 1e3
    z = cone["pos_x_min"] / km_to_cm / (args.t_model * day_to_s) / 1e3

    _surf = ax.scatter3D(x, y, z, c=-cone["fni"], cmap=plt.get_cmap("viridis"))

    # fig.colorbar(_surf, shrink=0.5, aspect=5)

    ax.set_xlabel(r"x [10$^3$ km/s]")
    ax.set_ylabel(r"y [10$^3$ km/s]")
    ax.set_zlabel(r"z [10$^3$ km/s]")

    plt.show()
    plt.close(fig)


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

    addarg_outputpath(parser)

    parser.add_argument("-rhoscale", default=None, type=float, help="Density scale factor")


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
        outputfolderpath=Path(args.outputpath), logfilename="make1dmodellog.txt"
    )

    make_1d_model_files(args, logprint)

    # make_plot(args, logprint) # Uncomment to make 3D plot todo: add command line option


if __name__ == "__main__":
    main()
