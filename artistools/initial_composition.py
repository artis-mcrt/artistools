#!/usr/bin/env python3
import argparse
import math
import os
import pandas as pd
import artistools as at
import matplotlib.pyplot as plt


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

    plotaxis1 = 'z'
    plotaxis2 = 'y'
    sliceaxis = 'x'
    sliceposition = 0.0
    ion = f'X_{args.ion}'

    plotvals = (merge_dfs.loc[merge_dfs[f'cellpos_in[{sliceaxis}]'] == sliceposition])
    print(plotvals.keys())
    ax = plt.subplot(111)
    im = ax.scatter(plotvals[f'cellpos_in[{plotaxis1}]'], plotvals[f'cellpos_in[{plotaxis2}]'], c=plotvals[ion])
    plt.colorbar(im, label=ion)
    plt.xlabel(plotaxis1)
    plt.ylabel(plotaxis2)
    plt.title(f'At {sliceaxis} = {sliceposition}')

    plt.show()


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
