#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pandas as pd
import artistools as at
import matplotlib.pyplot as plt
from astropy import units as u
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import gc


def make_cone(args):
    print("Making cone")

    angle_of_cone = 30  # in deg

    theta = np.radians([angle_of_cone / 2])  # angle between line of sight and edge is half angle of cone

    # merge_dfs, args.t_model, args.vmax = at.inputmodel.get_modeldata(args.modelpath[0], dimensions=3, get_abundances=True)
    merge_dfs = at.inputmodel.get_3d_model_data_merged_model_and_abundances_minimal(args)

    if args.positive_axis:
        cone = (merge_dfs.loc[merge_dfs[f'cellpos_in[{args.sliceaxis}]'] >= 1 / (np.tan(theta))
                              * np.sqrt((merge_dfs[f'cellpos_in[{args.other_axis2}]']) ** 2
                                        + (merge_dfs[f'cellpos_in[{args.other_axis1}]']) ** 2)])  # positive axis
    else:
        cone = (merge_dfs.loc[merge_dfs[f'cellpos_in[{args.sliceaxis}]'] <= -1/(np.tan(theta))
                              * np.sqrt((merge_dfs[f'cellpos_in[{args.other_axis2}]'])**2
                                        + (merge_dfs[f'cellpos_in[{args.other_axis1}]'])**2)])  # negative axis
    # print(cone.loc[:, :[f'pos_{slice_on_axis}']])

    del merge_dfs  # merge_dfs not needed anymore so free memory
    gc.collect()

    return cone


def get_profile_along_axis(args=None):
    print("Getting profile along axis")

    merge_dfs, args.t_model, args.vmax = at.inputmodel.get_modeldata(args.modelpath, dimensions=3, get_abundances=True)

    position_closest_to_axis = merge_dfs.iloc[
        (merge_dfs[f'pos_{args.other_axis2}']).abs().argsort()][:1][f'pos_{args.other_axis2}'].item()

    if args.positive_axis:
        profile1D = merge_dfs.loc[(merge_dfs[f'pos_{args.other_axis1}'] == position_closest_to_axis)
                                  & (merge_dfs[f'pos_{args.other_axis2}'] == position_closest_to_axis)
                                  & (merge_dfs[f'pos_{args.sliceaxis}'] > 0)]
    else:
        profile1D = merge_dfs.loc[(merge_dfs[f'pos_{args.other_axis1}'] == position_closest_to_axis)
                                  & (merge_dfs[f'pos_{args.other_axis2}'] == position_closest_to_axis)
                                  & (merge_dfs[f'pos_{args.sliceaxis}'] < 0)]

    return profile1D.reset_index(drop=True)


def make_1D_profile(args):
    if args.makefromcone:
        cone = make_cone(args)

        slice1D = cone.groupby([f'cellpos_in[{args.sliceaxis}]'], as_index=False).mean()
        # where more than 1 X value, average rows eg. (1,0,0) (1,1,0) (1,1,1)

    else:  # make from along chosen axis
        slice1D = get_profile_along_axis(args)

    slice1D[f'cellpos_in[{args.sliceaxis}]'] = slice1D[f'cellpos_in[{args.sliceaxis}]'].apply(
        lambda x: x / args.t_model * (u.cm / u.day).to('km/s'))  # Convert positions to velocities
    slice1D = slice1D.rename(columns={f'cellpos_in[{args.sliceaxis}]': 'vout_kmps'})
    # Convert position to velocity

    slice1D = slice1D.drop(['inputcellid', f'cellpos_in[{args.other_axis1}]', f'cellpos_in[{args.other_axis2}]'],
                           axis=1)  # Remove columns we don't need

    slice1D['rho_model'] = slice1D['rho_model'].apply(lambda x: np.log10(x) if x != 0 else -100)
    # slice1D = slice1D[slice1D['rho_model'] != -100]  # Remove empty cells
    # TODO: fix this, -100 probably breaks things if it's not one of the outer cells that gets chopped
    slice1D = slice1D.rename(columns={'rho_model': 'log_rho'})

    slice1D.index += 1

    if not args.positive_axis:
        # Invert rows and *velocity by -1 to make velocities positive for slice on negative axis
        slice1D.iloc[:] = slice1D.iloc[::-1].values
        slice1D['vout_kmps'] = slice1D['vout_kmps'].apply(lambda x: x*-1)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(slice1D)

    # print(slice1D.keys())
    return slice1D


def make_1D_model_files(args):
    slice1D = make_1D_profile(args)
    query_abundances_positions = slice1D.columns.str.startswith('X_')
    model_df = slice1D.loc[:, ~query_abundances_positions]
    abundances_df = slice1D.loc[:, query_abundances_positions]

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Print all rows in df
    #     print(model_df)

    # print(modelpath)
    model_df = model_df.round(decimals=5)  # model files seem to be to 5 sf
    model_df.to_csv(args.modelpath[0]/"model_1D.txt", sep=' ', header=False)  # write model.txt

    abundances_df.to_csv(args.modelpath[0]/"abundances_1D.txt", sep=' ', header=False)  # write abundances.txt

    with open(args.modelpath[0]/"model_1D.txt", 'r+') as f:  # add number of cells and tmodel to start of file
        content = f.read()
        f.seek(0, 0)
        f.write(f"{model_df.shape[0]}\n{args.t_model}".rstrip('\r\n') + '\n' + content)

    print("Saved abundances_1D.txt and model_1D.txt")

# with open(args.modelpath[0]/"model

# print(cone)

# cone = (merge_dfs.loc[merge_dfs[f'pos_{args.other_axis2}'] <= - (1/(np.tan(theta))
# * np.sqrt((merge_dfs[f'pos_{slice_on_axis}'])**2 + (merge_dfs[f'pos_{args.other_axis1}'])**2))])
# cone = merge_dfs
# cone = cone.loc[cone['rho_model'] > 0.0]


def make_plot(args):
    cone = make_cone(args)

    cone = cone.loc[cone['rho_model'] > 0.0002]  # cut low densities (empty cells?) from plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # print(cone['rho_model'])

    # set up for big model. For scaled down artis input model switch x and z
    x = cone['pos_z'].apply(lambda x: x / args.t_model * (u.cm / u.day).to('km/s'))/1e3
    y = cone['pos_y'].apply(lambda x: x / args.t_model * (u.cm / u.day).to('km/s'))/1e3
    z = cone['pos_x'].apply(lambda x: x / args.t_model * (u.cm / u.day).to('km/s'))/1e3

    surf = ax.scatter3D(x, y, z, c=-cone['fni'], cmap=plt.get_cmap('viridis'))

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel(r'x [10$^3$ km/s]')
    ax.set_ylabel(r'y [10$^3$ km/s]')
    ax.set_zlabel(r'z [10$^3$ km/s]')

    # plt.scatter(cone[f'pos_x']/1e11, cone[f'pos_y']/1e11)
    plt.show()


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path to ARTIS model folders with model.txt and abundances.txt')

    parser.add_argument('-sliceaxis', type=str, default='x',
                        help='Choose x, y or z. The 1D model will be made based on this axis.'
                             'If cone then cone made around sliceaxis. Otherwise model along axis.')

    parser.add_argument('--positive_axis', action='store', default=True,
                        help='Make 1D model from positive axis. Default is True')

    parser.add_argument('--makefromcone', action='store', default=True,
                        help='Make 1D model from cone around axis. Default is True.'
                             'If False uses points along axis.')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Make 1D model from cone in 3D model. Call with "makeartismodel1dslicefromcone"')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = [Path('.')]

    args.other_axis1 = None
    args.other_axis2 = None
    axes = ['x', 'y', 'z']
    for ax in axes:
        if args.other_axis1 is None and ax != args.sliceaxis:
            args.other_axis1 = ax
        elif args.other_axis2 is None and ax != args.sliceaxis:
            args.other_axis2 = ax
    # remember: models before scaling down to artis input have x and z axis swapped compared to artis input files

    make_1D_model_files(args)

    # make_plot(args) # Uncomment to make 3D plot todo: add command line option


if __name__ == '__main__':
    main()
