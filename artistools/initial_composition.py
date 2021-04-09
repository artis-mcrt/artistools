#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
import matplotlib
import numpy as np

import artistools as at
import artistools.inputmodel.opacityinputfile

# import scipy.interpolate
# import artistools.inputmodel
# from mpl_toolkits.mplot3d import Axes3D


def plot_2d_initial_abundances(modelpath, args):
    model = at.inputmodel.get_2d_modeldata(modelpath[0])
    abundances = at.inputmodel.get_initialabundances(modelpath[0])

    abundances['inputcellid'] = abundances['inputcellid'].apply(lambda x: float(x))

    merge_dfs = model.merge(abundances, how='inner', on='inputcellid')

    with open(os.path.join(modelpath[0], 'model.txt'), 'r') as fmodelin:
        fmodelin.readline()  # npts r, npts z
        t_model = float(fmodelin.readline())  # days
        vmax = float(fmodelin.readline())  # v_max in [cm/s]

    r = merge_dfs['cellpos_mid[r]'] / t_model * (u.cm/u.day).to('km/s') / 10 ** 3
    z = merge_dfs['cellpos_mid[z]'] / t_model * (u.cm / u.day).to('km/s') / 10 ** 3

    ion = f'X_{args.ion}'
    font = {'weight': 'bold',
            'size': 18}

    f = plt.figure(figsize=(4, 5))
    ax = f.add_subplot(111)
    im = ax.scatter(r, z, c=merge_dfs[ion], marker="8")

    f.colorbar(im)
    plt.xlabel(fr"v$_x$ in 10$^3$ km/s", fontsize='x-large')#, fontweight='bold')
    plt.ylabel(fr"v$_z$ in 10$^3$ km/s", fontsize='x-large')#, fontweight='bold')
    plt.text(20, 25, args.ion, color='white', fontweight='bold', fontsize='x-large')
    plt.tight_layout()
    # ax.labelsize: 'large'
    # plt.title(f'At {sliceaxis} = {sliceposition}')

    outfilename = f'plotcomposition{args.ion}.pdf'
    plt.savefig(Path(modelpath[0]) / outfilename, format='pdf')
    print(f'Saved {outfilename}')


def plot_3d_initial_abundances(modelpath, args):

    # t_model is in days and vmax is in cm/s
    model, t_model, vmax = at.inputmodel.get_modeldata(modelpath[0], dimensions=3)

    abundances = at.inputmodel.get_initialabundances(modelpath[0])

    abundances['inputcellid'] = abundances['inputcellid'].apply(lambda x: float(x))

    merge_dfs = model.merge(abundances, how='inner', on='inputcellid')
    # merge_dfs = plot_most_abundant(modelpath, args)

    plotaxis1 = 'y'
    plotaxis2 = 'z'
    sliceaxis = 'x'
    sliceposition = merge_dfs.iloc[(merge_dfs['pos_x']).abs().argsort()][:1]['pos_x'].item()
    # Choose position to slice. This gets minimum absolute value as the closest to 0
    ion = f'X_{args.ion}'

    plotvals = (merge_dfs.loc[merge_dfs[f'pos_{sliceaxis}'] == sliceposition])
    print(plotvals.keys())

    font = {'weight': 'bold',
            'size': 18}

    matplotlib.rc('font', **font)
    x = plotvals[f'pos_{plotaxis1}'] / t_model * (u.cm/u.day).to('km/s') / 10 ** 3
    y = plotvals[f'pos_{plotaxis2}'] / t_model * (u.cm/u.day).to('km/s') / 10 ** 3
    # fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    im = ax.scatter(x, y, c=plotvals[ion], marker="8", rasterized=True)  # cmap=plt.get_cmap('PuOr')

    cbar = plt.colorbar(im)
    # cbar.set_label(label=ion, size='x-large') #, fontweight='bold')
    # cbar.ax.set_title(f'{args.ion}', size='small')
    # cbar.ax.tick_params(labelsize='x-large')
    plt.xlabel(fr"v$_{plotaxis1}$ in 10$^3$ km/s", fontsize='x-large')#, fontweight='bold')
    plt.ylabel(fr"v$_{plotaxis2}$ in 10$^3$ km/s", fontsize='x-large')#, fontweight='bold')

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    plt.text(xmax*0.6, ymax*0.7, args.ion, color='white', fontweight='bold', fontsize='x-large')
    plt.tight_layout()
    # ax.labelsize: 'large'
    # plt.title(f'At {sliceaxis} = {sliceposition}')

    outfilename = f'plotcomposition{args.ion}.pdf'
    plt.savefig(Path(modelpath[0]) / outfilename, format='pdf')
    print(f'Saved {outfilename}')


def get_model_abundances_Msun_1D(modelpath):
    filename = modelpath / 'model.txt'
    modeldata, t_model_init_days, _ = at.inputmodel.get_modeldata(filename)
    abundancedata = at.inputmodel.get_initialabundances(modelpath)

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60

    modeldata['volume_shell'] = 4 / 3 * math.pi * ((modeldata['velocity_outer'] * 1e5 * t_model_init_seconds) ** 3
                                                   - (modeldata['velocity_inner'] * 1e5 * t_model_init_seconds) ** 3)

    modeldata['mass_shell'] = (10 ** modeldata['logrho']) * modeldata['volume_shell']

    merge_dfs = modeldata.merge(abundancedata, how='inner', on='inputcellid')

    print("Total mass (Msun):")
    for key in merge_dfs.keys():
        if 'X_' in key:
            merge_dfs[f'mass_{key}'] = merge_dfs[key] * merge_dfs['mass_shell'] * u.g.to('solMass')
            # get mass of element in each cell
            print(key, merge_dfs[f'mass_{key}'].sum())  # print total mass of element in solmass

    return merge_dfs


def plot_most_abundant(modelpath, args):
    model, _, _ = at.inputmodel.get_modeldata(modelpath[0], dimensions=3)
    abundances = at.inputmodel.get_initialabundances(modelpath[0])

    merge_dfs = model.merge(abundances, how='inner', on='inputcellid')
    elements = [x for x in merge_dfs.keys() if 'X_' in x]

    merge_dfs['max'] = merge_dfs[elements].idxmax(axis=1)

    merge_dfs['max'] = merge_dfs['max'].apply(lambda x: at.get_atomic_number(x[2:]))
    merge_dfs = merge_dfs[merge_dfs['max'] != 1]

    return merge_dfs


def make_3d_plot(modelpath, args):
    import pyvista as pv
    pv.set_plot_theme("document")  # set white background

    model, t_model, vmax = at.inputmodel.get_modeldata(modelpath, dimensions=3, get_abundances=False)

    # choose what surface will be coloured by
    if args.rho:
        coloursurfaceby = 'rho'
    elif args.opacity:
        model['opacity'] = at.inputmodel.opacityinputfile.get_opacity_from_file(modelpath)
        coloursurfaceby = 'opacity'
    else:
        print(f"Colours set by X_{args.ion}")
        coloursurfaceby = f'X_{args.ion}'

    # generate grid from data
    grid = round(len(model['rho']) ** (1./3.))
    surfacecolorscale = np.zeros((grid, grid, grid))  # needs 3D array
    xgrid = np.zeros(grid)

    surfacearr = np.array(model[coloursurfaceby])

    i = 0
    for z in range(0, grid):
        for y in range(0, grid):
            for x in range(0, grid):
                surfacecolorscale[x, y, z] = surfacearr[i]
                xgrid[x] = -vmax + 2 * x * vmax / grid
                i += 1

    x,y,z = np.meshgrid(xgrid,xgrid,xgrid)

    mesh = pv.StructuredGrid(x,y,z)
    print(mesh) # tells you the properties of the mesh

    mesh[coloursurfaceby] = surfacecolorscale.ravel(order='F') # add data to the mesh
    # mesh.plot()

    print(f"{coloursurfaceby} minumin {min(mesh[coloursurfaceby])}, maximum {max(mesh[coloursurfaceby])}")

    if not args.surfaces3d:
        surfacepositions = np.linspace(min(mesh[coloursurfaceby]), max(mesh[coloursurfaceby]), num=10)
        print(f"Using default surfaces {surfacepositions} \n define these with -surfaces3d for better results")
    else:
        surfacepositions = args.surfaces3d
    # surfacepositions = [1, 50, 100, 300, 500, 800, 1000, 1100, 1200, 1300, 1400, 1450, 1500] # choose these

    surf = mesh.contour(surfacepositions,scalars=coloursurfaceby) # create isosurfaces

    surf.plot(opacity='linear', screenshot=modelpath / '3Dplot.png')    # plot surfaces and save screenshot


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path(s) to ARTIS folder'
                        ' (may include wildcards such as * and **)')

    parser.add_argument('-ion', type=str, default='Fe',
                        help='Choose ion to plot. Default is Fe')

    parser.add_argument('--rho', action='store_true',
                        help='Plot rho instead of ion')

    parser.add_argument('--opacity', action='store_true',
                        help='Plot opacity from opacity.txt (if available for model)')

    parser.add_argument('-modeldim', type=int, default=None,
                        help='Choose how many dimensions. 3 for 3D, 2 for 2D')

    parser.add_argument('--plot3d', action='store_true',
                        help='Make 3D plot')

    parser.add_argument('-surfaces3d', type=float, nargs='+',
                        help='define positions of surfaces for 3D plots')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS input model composition')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath:
        args.modelpath = ['.']

    args.modelpath = at.flatten_list(args.modelpath)

    if args.plot3d:
        make_3d_plot(Path(args.modelpath[0]), args)
        return

    if not args.modeldim:
        inputparams = at.get_inputparams(args.modelpath[0])
    else:
        inputparams = {'n_dimensions': args.modeldim}

    if inputparams['n_dimensions'] == 2:
        plot_2d_initial_abundances(args.modelpath, args)

    if inputparams['n_dimensions'] == 3:
        plot_3d_initial_abundances(args.modelpath, args)


if __name__ == '__main__':
    main()
