#!/usr/bin/env python3
import argparse
import gc
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

import artistools as at

if t.TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


def make_cone(args: argparse.Namespace, logprint: t.Callable[..., None]) -> pd.DataFrame:
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
    dfmodel = pldfmodel.collect().to_pandas(use_pyarrow_extension_array=True)
    args.t_model = modelmeta["t_model_init_days"]

    if args.positive_axis:
        print("using positive axis")
        cone = dfmodel.loc[
            dfmodel[f"pos_{args.sliceaxis}_mid"]
            >= 1
            / (np.tan(theta))
            * np.sqrt((dfmodel[f"pos_{args.other_axis2}_mid"]) ** 2 + (dfmodel[f"pos_{args.other_axis1}_mid"]) ** 2)
        ]  # positive axis
    else:
        print("using negative axis")
        cone = dfmodel.loc[
            dfmodel[f"pos_{args.sliceaxis}_mid"]
            <= -1
            / (np.tan(theta))
            * np.sqrt((dfmodel[f"pos_{args.other_axis2}_mid"]) ** 2 + (dfmodel[f"pos_{args.other_axis1}_mid"]) ** 2)
        ]  # negative axis
    # print(cone.loc[:, :[f'pos_{slice_on_axis}']])
    del dfmodel  # merge_dfs not needed anymore so free memory
    gc.collect()

    assert isinstance(cone, pd.DataFrame)
    return cone


def get_profile_along_axis(
    args: argparse.Namespace, modeldata: pd.DataFrame | None = None, derived_cols: Sequence[str] | None = None
) -> pd.DataFrame:
    print("Getting profile along axis")

    if modeldata is None:
        modeldata = (
            at.inputmodel
            .get_modeldata(args.modelpath, get_elemabundances=True, derived_cols=derived_cols)[0]
            .collect()
            .to_pandas(use_pyarrow_extension_array=True)
        )

    position_closest_to_axis = modeldata.iloc[(modeldata[f"pos_{args.other_axis2}_min"]).abs().argsort()][:1][
        f"pos_{args.other_axis2}_min"
    ].item()

    if args.positive_axis:
        profile1d = modeldata.loc[
            (modeldata[f"pos_{args.other_axis1}_min"] == position_closest_to_axis)
            & (modeldata[f"pos_{args.other_axis2}_min"] == position_closest_to_axis)
            & (modeldata[f"pos_{args.sliceaxis}_min"] > 0)
        ]
    else:
        profile1d = modeldata.loc[
            (modeldata[f"pos_{args.other_axis1}_min"] == position_closest_to_axis)
            & (modeldata[f"pos_{args.other_axis2}_min"] == position_closest_to_axis)
            & (modeldata[f"pos_{args.sliceaxis}_min"] < 0)
        ]

    profile1d = profile1d.reset_index(drop=True)

    assert isinstance(profile1d, pd.DataFrame)
    return profile1d


def make_1d_profile(args: argparse.Namespace, logprint: t.Callable[..., None]) -> pd.DataFrame:
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
            cone1d_bins = np.power(cone_radius_spacing, (1 / shell_spacing_power))
        cone1d_df = []
        for i in range(len(cone1d_bins) - 1):
            # Filter cells within bin
            cone1d_bin_mask = (cone["pos_r_mid"] >= cone1d_bins[i]) & (cone["pos_r_mid"] < cone1d_bins[i + 1])
            cells_within_bin = cone[cone1d_bin_mask]
            # Sum mass and volume for each of the 3D cells that are being included in this 1D shell
            total_mass_g = cells_within_bin["mass_g"].sum()
            total_volume = cells_within_bin["volume"].sum()

            if total_mass_g <= 0 and total_volume <= 0:
                assert total_mass_g > 0, (
                    f"\nAssertion Error: No cell midpoints within cone limits for shell {i + 1}.\n"
                    "The small volume contained within the cone for the innermost shell means this is quite likely to\n"
                    "occur, especially for smaller cone angles and grid spacings where the inner shell radius is\n"
                    "small. Also more likely to occur when ncoordgrid/2 is even, resulting in the slice axis being\n"
                    "along cell minimums not cell midpoints in the 3D model. If this occurs you can either chose a \n"
                    "different grid spacing (using -coneshellspacingexponent or -nshells) or increase -coneangle\n"
                    "to ensure at least one cell midpoint is contained within the cone limits of the shell\n"
                )

            elif total_mass_g == 0:
                logprint(
                    f"\nWARNING: Shell {i + 1} is empty (all 3D grid cells averaged in the shell must have density=0).\n"
                    "This shell and all shells further out in the model will be removed from the model.\n"
                    "This is safe provided this empty shell is far enough out in the model: check model file to \n"
                    "confirm this is the case. If not there may be an issue with the model being read in.\n"
                    "The outer regions of some models can have empty regions before there are more non-empty cells\n"
                    "again at higher velocities. This should generally be in the very outer regions of models where\n"
                    "the cells are too optically thin to impact the synthetic observables. However if you want cells\n"
                    "in these outer regions to be included in the 1D cone can experiment with -coneangle,-nshells and\n"
                    "-coneshellspacingexponent to ensure the shells for these outer regions include some non-empty 3D\n"
                    "grid cell and thus the shells can be included the 1D model.\n"
                )
                break

            species_mass = cells_within_bin.filter(regex=r"^X_", axis=1)
            mass_g_values = cells_within_bin["mass_g"].to_numpy().reshape(-1, 1)
            # Calculate mass of each species in each 3D grid cell
            species_mass *= mass_g_values
            # Sum masses of individual species
            species_total_mass = species_mass.sum(axis=0)

            # Calculate composition
            composition = species_total_mass / total_mass_g

            # Sum all composition values to ensure compositions are normalised to 1
            if i == 0:
                logprint(
                    "\nSumming all mass weighted compositions in the shells. If these values significantly\n"
                    "deviate from 1 there could be an issue with the input model. The compositions for each\n"
                    "shell in the output 1D model are normalised here regardless of how close to 1 they are.\n"
                    "Also printing how many 3D cells make up each 1D shell in the model generated.\n"
                )
            sum_composition_check = composition.iloc[5:].sum()
            logprint(
                f"Shell {i + 1:<3}     3D cells averaged: {len(cells_within_bin):<6} composition sum before norm: {sum_composition_check}"
            )
            composition /= sum_composition_check

            bin_cone1d_df_dict = {
                "inputcellid": [i + 1],
                "r_bin_max_boundary": [cone1d_bins[i + 1]],
                "rho": [total_mass_g / total_volume],
                **{species: [composition[species]] for species in composition.index},
            }

            # Create DataFrame from the dictionary
            bin_cone1d_df = pd.DataFrame(bin_cone1d_df_dict)

            # Append results for this bin to the overall results
            cone1d_df.append(bin_cone1d_df)

        # Concatenate all bin results into a single DataFrame
        slice1d = pd.concat(cone1d_df, ignore_index=True)
        slice1d["r_bin_max_boundary"] = slice1d["r_bin_max_boundary"].apply(lambda x: x / (args.t_model * 86400 * 1e5))
        slice1d = slice1d.rename(columns={"r_bin_max_boundary": "vel_r_max_kmps"})

    else:  # make from along chosen axis
        logprint("from along the axis")
        slice1d = get_profile_along_axis(args)
        slice1d.loc[:, f"pos_{args.sliceaxis}_min"] = slice1d[f"pos_{args.sliceaxis}_min"].apply(
            lambda x: x / (args.t_model * 86400 * 1e5)
        )  # Convert positions to velocities
        slice1d = slice1d.rename(columns={f"pos_{args.sliceaxis}_min": "vel_r_max_kmps"})
        # Convert position to velocity
        slice1d = slice1d.drop(
            ["inputcellid", f"pos_{args.other_axis1}_min", f"pos_{args.other_axis2}_min"], axis=1
        )  # Remove columns we don't need
    logprint("using axis:", args.axis)

    if args.rhoscale:
        logprint("Scaling density by a factor of:", args.rhoscale)
        slice1d.loc[:, "rho"] *= args.rhoscale

    slice1d.loc[:, "rho"] = slice1d["rho"].apply(lambda x: np.log10(x) if x != 0 else -100)
    # slice1d = slice1d[slice1d['rho_model'] != -100]  # Remove empty cells
    # TODO: fix this, -100 probably breaks things if it's not one of the outer cells that gets chopped
    slice1d = slice1d.rename(columns={"rho": "logrho"})

    slice1d.index += 1

    if not args.positive_axis:
        # Invert rows and *velocity by -1 to make velocities positive for slice on negative axis
        slice1d.iloc[:] = slice1d.iloc[::-1].to_numpy()
        slice1d.loc[:, "vel_r_max_kmps"] *= -1

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(slice1d)

    # print(slice1d.keys())
    return slice1d


def make_1d_model_files(args: argparse.Namespace, logprint: t.Callable[..., None]) -> None:
    slice1d = make_1d_profile(args, logprint)

    # query_abundances_positions = slice1d.columns.str.startswith("X_")
    query_abundances_positions = np.array([
        (column.startswith("X_") and not (any(i.isdigit() for i in column))) and (len(column) < 5)
        for column in slice1d.columns
    ])

    model_df = slice1d.loc[:, ~query_abundances_positions]
    abundances_df = slice1d.loc[:, query_abundances_positions]

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Print all rows in df
    #     print(model_df)

    # print(modelpath)
    # model_df = model_df.round(decimals=5)  # model files seem to be to 5 sf
    # model_df.to_csv(args.modelpath[0] / "model_1d.txt", sep=" ", header=False)  # write model.txt

    npts_model = len(model_df)
    inputcellid = np.arange(1, npts_model + 1)
    model_df.loc[:, ["inputcellid"]] = inputcellid
    abundances_df.loc[:, ["inputcellid"]] = inputcellid
    assert isinstance(model_df, pd.DataFrame)
    at.inputmodel.save_modeldata(
        dfmodel=pl.from_pandas(model_df), t_model_init_days=args.t_model, outpath=Path(args.outputpath, "model_1d.txt")
    )

    assert isinstance(abundances_df, pd.DataFrame)
    at.inputmodel.save_initelemabundances(
        pl.from_pandas(abundances_df), outpath=Path(args.outputpath, "abundances_1d.txt")
    )

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


def make_plot(args: argparse.Namespace, logprint: t.Callable[..., None]) -> None:
    cone = make_cone(args, logprint)

    cone = cone.loc[cone["rho_model"] > 0.0002]  # cut low densities (empty cells?) from plot
    ax: Axes3D = plt.figure().gca(projection="3d")  # type: ignore[call-arg,no-any-unimported] # pyright: ignore[reportCallIssue]

    # print(cone['rho_model'])

    # set up for big model. For scaled down artis input model switch x and z
    x = cone["pos_z_min"].apply(lambda x: x / 1e5 / (args.t_model * 86400)) / 1e3
    y = cone["pos_y_min"].apply(lambda x: x / 1e5 / (args.t_model * 86400)) / 1e3
    z = cone["pos_x_min"].apply(lambda x: x / 1e5 / (args.t_model * 86400)) / 1e3

    _surf = ax.scatter3D(x, y, z, c=-cone["fni"], cmap=plt.get_cmap("viridis"))  # pyright: ignore[reportArgumentType]

    # fig.colorbar(_surf, shrink=0.5, aspect=5)

    ax.set_xlabel(r"x [10$^3$ km/s]")
    ax.set_ylabel(r"y [10$^3$ km/s]")
    ax.set_zlabel(r"z [10$^3$ km/s]")

    # plt.scatter(cone[f'pos_x_min']/1e11, cone[f'pos_y_min']/1e11)
    plt.show()


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-modelpath",
        default=[],
        nargs="*",
        type=Path,
        help="Path to ARTIS model folders with model.txt and abundances.txt",
    )

    parser.add_argument(
        "-axis",
        default="+x",
        choices=["+x", "-x", "+y", "-y", "+z", "-z"],
        help="Choose an axis. USE INSTEAD OF DEPRECATED --POSITIVE_AXES AND -SLICEAXIS ARGS. Hint: for negative use e.g. -axis=-z",
    )

    parser.add_argument(
        "--makefromcone",
        action="store",
        default=True,
        help="Make 1D model from cone around axis. Default is True.If False uses points along axis.",
    )

    parser.add_argument(
        "-coneangle", type=float, default=30.0, help="Cone angle in degrees, cone half angle given by coneangle/2"
    )

    parser.add_argument(
        "-nshells",
        type=int,
        default=100,
        help="Number of shells used when making 1D model from cone. Note the final number of shell may be lower as empty outer shells are removed from the output 1D model files",
    )

    parser.add_argument(
        "-coneshellspacingexponent",
        type=float,
        default=1.5,
        help="Vary the exponent used when selecting the radius dependance of the shell spacing when making 1D model from cone. By default the shells are spaced evenly in radius^(1.5)",
    )

    parser.add_argument(
        "--coneshellsequalvolume", action="store_true", help="Use equal volume shells when making 1D model from cone"
    )

    parser.add_argument("-outputpath", "-o", default=".", help="Path for output files")

    parser.add_argument("-rhoscale", "-v", default=None, type=float, help="Density scale factor")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Make 1D model from cone in 3D model."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        args = parser.parse_args([] if kwargs else argsraw)

    if not args.modelpath:
        args.modelpath = [Path()]

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
