#!/usr/bin/env python3
import argparse
import gc
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import artistools as at

if t.TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


def make_cone(args: argparse.Namespace) -> pd.DataFrame:
    print("Making cone")

    angle_of_cone = 30  # in deg

    theta = np.radians([angle_of_cone / 2])  # angle between line of sight and edge is half angle of cone

    dfmodel, modelmeta = at.get_modeldata_pandas(modelpath=args.modelpath[0], get_elemabundances=True)
    args.t_model = modelmeta["t_model_init_days"]

    cone: pd.DataFrame
    if args.positive_axis:
        print("using positive axis")
        cone = dfmodel.loc[
            dfmodel[f"pos_{args.sliceaxis}_min"]
            >= 1
            / (np.tan(theta))
            * np.sqrt((dfmodel[f"pos_{args.other_axis2}_min"]) ** 2 + (dfmodel[f"pos_{args.other_axis1}_min"]) ** 2)
        ]  # positive axis
    else:
        print("using negative axis")
        cone = dfmodel.loc[
            dfmodel[f"pos_{args.sliceaxis}_min"]
            <= -1
            / (np.tan(theta))
            * np.sqrt((dfmodel[f"pos_{args.other_axis2}_min"]) ** 2 + (dfmodel[f"pos_{args.other_axis1}_min"]) ** 2)
        ]  # negative axis
    # print(cone.loc[:, :[f'pos_{slice_on_axis}']])

    del dfmodel  # merge_dfs not needed anymore so free memory
    gc.collect()

    return cone


def get_profile_along_axis(
    args: argparse.Namespace, modeldata: pd.DataFrame | None = None, derived_cols: Sequence[str] | None = None
) -> pd.DataFrame:
    print("Getting profile along axis")

    # merge_dfs, args.t_model, args.vmax = at.inputmodel.get_modeldata_tuple(args.modelpath, dimensions=3, get_elemabundances=True)
    if modeldata is None:
        modeldata, _ = at.inputmodel.get_modeldata_pandas(
            args.modelpath, get_elemabundances=True, derived_cols=derived_cols
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


def make_1d_profile(args: argparse.Namespace) -> pd.DataFrame:
    """Make 1D model from 3D model."""
    logprint = at.inputmodel.inputmodel_misc.savetologfile(
        outputfolderpath=Path(args.outputpath), logfilename="make1dmodellog.txt"
    )

    logprint("Making 1D model from 3D model:", at.get_model_name(args.modelpath[0]))
    if args.makefromcone:
        logprint("from a cone")
        cone = make_cone(args)

        slice1d = cone.groupby([f"pos_{args.sliceaxis}_min"], as_index=False).mean()
        # where more than 1 X value, average rows eg. (1,0,0) (1,1,0) (1,1,1)

    else:  # make from along chosen axis
        logprint("from along the axis")
        slice1d = get_profile_along_axis(args)
    logprint("using axis:", args.axis)

    slice1d[f"pos_{args.sliceaxis}_min"] = slice1d[f"pos_{args.sliceaxis}_min"].apply(
        lambda x: x / 1e5 / (args.t_model * 86400)
    )  # Convert positions to velocities
    slice1d = slice1d.rename(columns={f"pos_{args.sliceaxis}_min": "vel_r_max_kmps"})
    # Convert position to velocity

    slice1d = slice1d.drop(
        ["inputcellid", f"pos_{args.other_axis1}_min", f"pos_{args.other_axis2}_min"], axis=1
    )  # Remove columns we don't need

    if args.rhoscale:
        logprint("Scaling density by a factor of:", args.rhoscale)
        slice1d["rho"] *= args.rhoscale

    slice1d["rho"] = slice1d["rho"].apply(lambda x: np.log10(x) if x != 0 else -100)
    # slice1d = slice1d[slice1d['rho_model'] != -100]  # Remove empty cells
    # TODO: fix this, -100 probably breaks things if it's not one of the outer cells that gets chopped
    slice1d = slice1d.rename(columns={"rho": "logrho"})

    slice1d.index += 1

    if not args.positive_axis:
        # Invert rows and *velocity by -1 to make velocities positive for slice on negative axis
        slice1d.iloc[:] = slice1d.iloc[::-1].to_numpy()
        slice1d["vel_r_max_kmps"] *= -1

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(slice1d)

    # print(slice1d.keys())
    return slice1d


def make_1d_model_files(args: argparse.Namespace) -> None:
    slice1d = make_1d_profile(args)

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
        dfmodel=model_df, t_model_init_days=args.t_model, outpath=Path(args.outputpath, "model_1d.txt")
    )

    assert isinstance(abundances_df, pd.DataFrame)
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


def make_plot(args: argparse.Namespace) -> None:
    cone = make_cone(args)

    cone = cone.loc[cone["rho_model"] > 0.0002]  # cut low densities (empty cells?) from plot
    ax: Axes3D = plt.figure().gca(projection="3d")  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]

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
        action=at.AppendPath,
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

    make_1d_model_files(args)

    # make_plot(args) # Uncomment to make 3D plot todo: add command line option


if __name__ == "__main__":
    main()
