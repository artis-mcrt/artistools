#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
import pandas as pd
import artistools as at
import matplotlib.pyplot as plt
from astropy import units as u
import matplotlib


def get_3d_model_input(modelpath):
    model = pd.read_csv(os.path.join(modelpath[0], 'model.txt'), delim_whitespace=True, header=None, skiprows=3)
    columns = ['inputcellid', 'cellpos_in[z]', 'cellpos_in[y]', 'cellpos_in[x]', 'rho_model',
               'ffe', 'fni', 'fco', 'ffe52', 'fcr48']
    model = pd.DataFrame(model.values.reshape(-1, 10))
    model.columns = columns
    # print(model)
    return model

def plot_3d_initial_abundances(modelpath, args):
    model = get_3d_model_input(modelpath[0])
    abundances = at.get_initialabundances(modelpath[0])

    abundances['inputcellid'] = abundances['inputcellid'].apply(lambda x: float(x))

    merge_dfs = model.merge(abundances, how='inner', on='inputcellid')

    with open(os.path.join(modelpath[0], 'model.txt'), 'r') as fmodelin:
        fmodelin.readline()  # npts_model3d
        t_model = float(fmodelin.readline())  # days
        vmax = float(fmodelin.readline())  # v_max in [cm/s]

    plotaxis1 = 'z'
    plotaxis2 = 'y'
    sliceaxis = 'x'
    sliceposition = 0.0
    ion = f'X_{args.ion}'

    plotvals = (merge_dfs.loc[merge_dfs[f'cellpos_in[{sliceaxis}]'] == sliceposition])
    print(plotvals.keys())
    factor = 10 ** 3
    font = {'weight': 'bold',
            'size': 10}

    matplotlib.rc('font', **font)
    x = plotvals[f'cellpos_in[{plotaxis1}]'] / t_model * (u.cm/u.day).to('km/s') / factor
    y = plotvals[f'cellpos_in[{plotaxis2}]'] / t_model * (u.cm/u.day).to('km/s') / factor
    # fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    im = ax.scatter(x, y, c=plotvals[ion], marker="8")

    cbar = plt.colorbar(im)
    cbar.set_label(label=ion, size='x-large') #, fontweight='bold')
    # cbar.ax.tick_params(labelsize='x-large')
    plt.xlabel(fr"v$_{plotaxis1}$ in 10$^3$ km/s", fontsize='x-large')#, fontweight='bold')
    plt.ylabel(fr"v$_{plotaxis2}$ in 10$^3$ km/s", fontsize='x-large')#, fontweight='bold')
    plt.text(25, 25, args.ion, color='white', fontweight='bold', fontsize='x-large')
    ax.labelsize: 'large'
    # plt.title(f'At {sliceaxis} = {sliceposition}')

    outfilename = f'plot_composition{args.ion}.pdf'
    plt.savefig(Path(modelpath[0]) / outfilename, format='pdf')
    print(f'Saved {outfilename}')


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path(s) to ARTIS folder'
                        ' (may include wildcards such as * and **)')

    parser.add_argument('-ion', type=str, default='Fe',
                        help='Choose ion to plot. Default is Fe')


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

    plot_3d_initial_abundances(args.modelpath, args)

if __name__ == '__main__':
    main()
