#!/usr/bin/env python3
import argparse
import glob
import itertools
import os.path

import artistools as at
import matplotlib.pyplot as plt


# import sys


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot ARTIS radiation field.')
    parser.add_argument('modelpath', default=[], nargs='*',
                        help='Paths to ARTIS folders with light_curve.out or packets files'
                        ' (may include wildcards such as * and **)')
    parser.add_argument('--frompackets', default=False, action='store_true',
                        help='Read packets files instead of light_curve.out')
    parser.add_argument('--gamma', default=False, action='store_true',
                        help='Make light curve from gamma rays instead of R-packets')
    parser.add_argument('-o', action='store', dest='outputfile',
                        help='Filename for PDF file')
    args = parser.parse_args()

    if len(args.modelpath) == 0:
        args.modelpath = ['.', '*']

    # combined the results of applying wildcards on each input
    modelpaths = list(itertools.chain.from_iterable([glob.glob(x) for x in args.modelpath if os.path.isdir(x)]))

    if not args.outputfile:
        args.outputfile = 'plotlightcurve_gamma.pdf' if args.gamma else 'plotlightcurve.pdf'

    make_lightcurve_plot(modelpaths, args.outputfile, args.frompackets, args.gamma)


def make_lightcurve_plot(modelpaths, filenameout, frompackets, gammalc):
    from astropy import units as u
    fig, axis = plt.subplots(1, 1, sharey=True, figsize=(8, 5), tight_layout={
        "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for index, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")
        lcpath = os.path.join(modelpath, 'gamma_light_curve.out' if gammalc else 'light_curve.out')
        if not os.path.exists(lcpath):
            print(f"Skipping {modelname} because {lcpath} does not exist")
            continue
        else:
            lcdata = at.lightcurves.readfile(lcpath)
            if frompackets:
                foundpacketsfiles = glob.glob(os.path.join(modelpath, 'packets00_????.out'))
                ranks = [int(os.path.basename(filename)[10:10 + 4]) for filename in foundpacketsfiles]
                nprocs = max(ranks) + 1
                print(f'Reading packets for {nprocs} processes')
                packetsfilepaths = [os.path.join(modelpath, f'packets00_{rank:04d}.out') for rank in range(nprocs)]

                timearray = lcdata['time'].values
                # timearray = np.arange(250, 350, 0.1)
                model, _ = at.get_modeldata(os.path.join(modelpath, 'model.txt'))
                vmax = model.iloc[-1].velocity * u.km / u.s
                lcdata = at.lightcurves.get_from_packets(packetsfilepaths, timearray, nprocs, vmax,
                                                         escape_type='TYPE_GAMMA' if gammalc else 'TYPE_RPKT')

        print("Plotting...")

        linestyle = ['-', '--'][int(index / 7)]

        axis.plot(lcdata.time, lcdata['lum'], linewidth=2, linestyle=linestyle, label=f'{modelname}')
        axis.plot(lcdata.time, lcdata['lum_cmf'], linewidth=2, linestyle=linestyle, label=f'{modelname} (cmf)')

    # axis.set_xlim(xmin=xminvalue,xmax=xmaxvalue)
    # axis.set_ylim(ymin=-0.1,ymax=1.3)

    axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')
    axis.set_ylabel(r'$\mathrm{L} ' + ('_\gamma' if gammalc else '') + r'/ \mathrm{L}_\odot$')

    fig.savefig(filenameout, format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


if __name__ == "__main__":
    main()
