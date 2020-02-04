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


def get_model_data(args):
    model = at.get_3d_modeldata(args.modelpath)
    abundances = at.get_initialabundances(args.modelpath[0])

    with open(os.path.join(args.modelpath[0], 'model.txt'), 'r') as fmodelin:
        fmodelin.readline()  # npts_model3d
        args.t_model = float(fmodelin.readline())  # days
        args.vmax = float(fmodelin.readline())  # v_max in [cm/s]

    print(model.keys())

    merge_dfs = model.merge(abundances, how='inner', on='inputcellid')
    return merge_dfs


def make_cone(args):
    args.slice_along_axis = 'x'
    args.other_axis1 = 'y'
    args.other_axis2 = 'z'

    args.positive_axis = True

    angle_of_cone = 30

    theta = np.radians([angle_of_cone / 2])

    merge_dfs = get_model_data(args)

    if args.positive_axis:
        cone = (merge_dfs.loc[merge_dfs[f'cellpos_in[{args.slice_along_axis}]'] >= 1 / (np.tan(theta))
                              * np.sqrt((merge_dfs[f'cellpos_in[{args.other_axis2}]']) ** 2
                                        + (merge_dfs[f'cellpos_in[{args.other_axis1}]']) ** 2)])  # positive axis
    else:
        cone = (merge_dfs.loc[merge_dfs[f'cellpos_in[{args.slice_along_axis}]'] <= -1/(np.tan(theta))
                              * np.sqrt((merge_dfs[f'cellpos_in[{args.other_axis2}]'])**2
                                        + (merge_dfs[f'cellpos_in[{args.other_axis1}]'])**2)])  # negative axis
    # print(cone.loc[:, :[f'cellpos_in[{slice_on_axis}]']])

    return cone


def make_1D_profile(args):

    cone = make_cone(args)

    slice1D = cone.groupby([f'cellpos_in[{args.slice_along_axis}]'], as_index=False).mean()
    # where more than 1 X value, average rows eg. (1,0,0) (1,1,0) (1,1,1)

    slice1D[f'cellpos_in[{args.slice_along_axis}]'] = slice1D[f'cellpos_in[{args.slice_along_axis}]'].apply(
        lambda x: x / args.t_model * (u.cm / u.day).to('km/s'))
    slice1D = slice1D.rename(columns={f'cellpos_in[{args.slice_along_axis}]': 'vout_kmps'})
    # Convert position to velocity

    slice1D = slice1D.drop(['inputcellid', f'cellpos_in[{args.other_axis1}]', f'cellpos_in[{args.other_axis2}]'],
                           axis=1)  # Remove columns we don't need

    slice1D['rho_model'] = slice1D['rho_model'].apply(lambda x: np.log10(x) if x != 0 else -100)
    # slice1D = slice1D[slice1D['rho_model'] != -100]  # Remove empty cells
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
    model_df = slice1D.loc[:,~query_abundances_positions]
    abundances_df = slice1D.loc[:,query_abundances_positions]

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # Print all rows in df
    #     print(model_df)

    # print(modelpath)
    model_df = model_df.round(decimals=5)  # model files seem to be to 5 df
    model_df.to_csv(args.modelpath[0]/"model_1D.txt", sep=' ', header=False)  # write model.txt

    abundances_df.to_csv(args.modelpath[0]/"abundances_1D.txt", sep=' ', header=False)  # write abundances.txt

    with open(args.modelpath[0]/"model_1D.txt", 'r+') as f:  # add number of cells and tmodel to start of file
        content = f.read()
        f.seek(0, 0)
        f.write(f"{model_df.shape[0]}\n{args.t_model}".rstrip('\r\n') + '\n' + content)

    print("Saved abundances_1D.txt and model_1D.txt")

# with open(args.modelpath[0]/"model

# print(cone)

# cone = (merge_dfs.loc[merge_dfs[f'cellpos_in[{args.other_axis2}]'] <= - (1/(np.tan(theta))
# * np.sqrt((merge_dfs[f'cellpos_in[{slice_on_axis}]'])**2 + (merge_dfs[f'cellpos_in[{args.other_axis1}]'])**2))])
# cone = merge_dfs
# cone = cone.loc[cone['rho_model'] > 0.0]


def make_plot(args):
    cone = make_cone(args)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.scatter3D(cone[f'cellpos_in[z]'], cone[f'cellpos_in[y]'], cone[f'cellpos_in[x]'], c=cone['rho_model'])

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # plt.scatter(cone[f'cellpos_in[x]']/1e11, cone[f'cellpos_in[y]']/1e11)
    plt.show()


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path to ARTIS model folders with model.txt and abundances.txt')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Make 1D model from cone in 3D model.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = [Path('.')]

    make_1D_model_files(args)


if __name__ == '__main__':
    main()
