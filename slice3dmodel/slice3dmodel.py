#!/usr/bin/env python3
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Covert abundances.txt and model.txt from 3D to a '
                'one dimensional slice.')
parser.add_argument('-inputfolder', action='store', default='3dmodel',
                    help='Path to folder with 3D files')
parser.add_argument('-axis', action='store', dest='chosenaxis', default='x',
                    choices=['x', 'y', 'z'],
                    help='Slice axis (x, y, or z)')
parser.add_argument('-outputfolder', action='store', default='1dslice',
                    help='Path to folder in which to store 1D output files')
parser.add_argument('-opdf', action='store', dest='pdfoutputfile',
                    default='plotmodel.pdf',
                    help='Path/filename for PDF plot.')
args = parser.parse_args()


def main():
    xlist = []
    ylist = []

    listout = []
    dict3dcellidto1dcellid = {}
    outcellid = 0
    with open(os.path.join(args.inputfolder, 'model.txt'), 'r') as fmodelin:
        fmodelin.readline()  # npts_model3d
        t_model = fmodelin.readline()  # days
        fmodelin.readline()  # v_max in [cm/s]

        while True:
            blockline1 = fmodelin.readline()
            blockline2 = fmodelin.readline()

            if not blockline1 or not blockline2:
                break

            line1split = blockline1.split()

            if len(line1split) == 5:
                (cellid, posx, posy, posz, rho) = line1split
            else:
                print("Wrong line size")
                sys.exit()

            line2split = blockline2.split()

            if len(line2split) == 5:
                (ffe, f56ni, fco, f52fe, f48cr) = map(float, line2split)
            else:
                print("Wrong line size")
                sys.exit()

            if posx != "0.0000000" and (
                    args.chosenaxis != 'x' or float(posx) < 0.):
                keep_cell = False
            elif posy != "0.0000000" and (
                    args.chosenaxis != 'y' or float(posy) < 0.):
                keep_cell = False
            elif posz != "0.0000000" and (
                    args.chosenaxis != 'z' or float(posz) < 0.):
                keep_cell = False
            else:
                keep_cell = True

            if keep_cell:
                outcellid += 1
                dict3dcellidto1dcellid[int(cellid)] = outcellid
                dist = math.sqrt(
                    float(posx) ** 2 + float(posy) ** 2 + float(posz) ** 2)
                velocity = float(dist) / float(t_model) / 86400. / 1.e5
                listout.append(
                    '{0:6d}  {1:8.2f}  {2:8.5f}  {3:.5f}  '
                    '{4:.5f}  {5:.5f}  {6:.5f}  {7:.5f}'
                    .format(outcellid, velocity,
                            math.log10(max(float(rho), 1e-100)),
                            ffe, f56ni, fco, f52fe, f48cr))
                print("Cell {0:4d} input1: {1}".format(
                    outcellid, blockline1.rstrip()))
                print("Cell {0:4d} input2: {1}".format(
                    outcellid, blockline2.rstrip()))
                print("Cell {0:4d} output: {1}".format(outcellid, listout[-1]))
                xlist.append(velocity)
                ylist.append(f56ni)

    with open(os.path.join(args.outputfolder, 'model.txt'), 'w') as fmodelout:
        fmodelout.write("{0:7d}\n".format(outcellid))
        fmodelout.write(t_model)
        for line in listout:
            fmodelout.write(line + "\n")

    convert_abundance_file(dict3dcellidto1dcellid)

    make_plot(xlist, ylist)


def convert_abundance_file(dict3dcellidto1dcellid):
    with open(os.path.join(
        args.inputfolder, 'abundances.txt'), 'r') as fabundancesin, open(
            os.path.join(args.outputfolder, 'abundances.txt'),
            'w') as fabundancesout:
        currentblock = []
        keepcurrentblock = False
        for line in fabundancesin:
            linesplit = line.split()

            if len(currentblock) + len(linesplit) >= 30:
                if keepcurrentblock:
                    fabundancesout.write("  ".join(currentblock) + "\n")
                currentblock = []
                keepcurrentblock = False

            if not currentblock:
                currentblock = linesplit
                if int(linesplit[0]) in dict3dcellidto1dcellid.keys():
                    outcellid = dict3dcellidto1dcellid[int(linesplit[0])]
                    currentblock[0] = "{0:6d}".format(outcellid)
                    keepcurrentblock = True
            else:
                currentblock.append(linesplit)

    if keepcurrentblock:
        print("WARNING: unfinished block")


def make_plot(xlist, ylist):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 4),
                           tight_layout={
                               "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    ax.set_xlabel(r'v (km/s)')
    ax.set_ylabel(r'Density (g/cm$^3$)')
    ax.plot(xlist, ylist, lw=1.5)
    fig.savefig(args.pdfoutputfile, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
