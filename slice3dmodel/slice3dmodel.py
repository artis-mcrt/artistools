#!/usr/bin/env python3
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Covert abundances.txt and model.txt from 3D to a '
                    'one dimensional slice.')
    parser.add_argument('-inputfolder', action='store', default='3dmodel',
                        help='Path to folder with 3D files')
    parser.add_argument('-axis', action='store', dest='chosenaxis',
                        default='x', choices=['x', 'y', 'z'],
                        help='Slice axis (x, y, or z)')
    parser.add_argument('-outputfolder', action='store', default='1dslice',
                        help='Path to folder in which to store 1D output'
                        ' files')
    parser.add_argument('-opdf', action='store', dest='pdfoutputfile',
                        default='plotmodel.pdf',
                        help='Path/filename for PDF plot.')
    args = parser.parse_args()

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
            # two lines making up a model grid cell
            block = fmodelin.readline(), fmodelin.readline()

            if not block[0] or not block[1]:
                break

            cell = {}
            blocksplit = block[0].split(), block[1].split()
            if len(blocksplit[0]) == 5:
                (cell['cellid'], cell['posx'], cell['posy'], cell['posz'],
                 cell['rho']) = blocksplit[0]
            else:
                print("Wrong line size")
                sys.exit()

            if len(blocksplit[1]) == 5:
                (cell['ffe'], cell['f56ni'], cell['fco'], cell['f52fe'],
                 cell['f48cr']) = map(float, blocksplit[1])
            else:
                print("Wrong line size")
                sys.exit()

            if cell['posx'] != "0.0000000" and (
                    args.chosenaxis != 'x' or float(cell['posx']) < 0.):
                pass
            elif cell['posy'] != "0.0000000" and (
                    args.chosenaxis != 'y' or float(cell['posy']) < 0.):
                pass
            elif cell['posz'] != "0.0000000" and (
                    args.chosenaxis != 'z' or float(cell['posz']) < 0.):
                pass
            else:
                outcellid += 1
                dict3dcellidto1dcellid[int(cell['cellid'])] = outcellid
                append_cell_to_output(cell, outcellid, t_model, listout,
                                      xlist, ylist)
                print("Cell {0:4d} input1: {1}".format(
                    outcellid, block[0].rstrip()))
                print("Cell {0:4d} input2: {1}".format(
                    outcellid, block[1].rstrip()))
                print("Cell {0:4d} output: {1}".format(outcellid, listout[-1]))

    with open(os.path.join(args.outputfolder, 'model.txt'), 'w') as fmodelout:
        fmodelout.write("{0:7d}\n".format(outcellid))
        fmodelout.write(t_model)
        for line in listout:
            fmodelout.write(line + "\n")

    convert_abundance_file(args.inputfolder, args.outputfolder,
                           dict3dcellidto1dcellid)

    make_plot(xlist, ylist, args.pdfoutputfile)


def convert_abundance_file(inputfolder, outputfolder, dict3dcellidto1dcellid):
    with open(os.path.join(
        inputfolder, 'abundances.txt'), 'r') as fabundancesin, open(
            os.path.join(outputfolder, 'abundances.txt'),
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


def make_plot(xlist, ylist, pdfoutputfile):
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(6, 4),
                           tight_layout={
                               "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    ax.set_xlabel(r'v (km/s)')
    ax.set_ylabel(r'Density (g/cm$^3$)')
    ax.plot(xlist, ylist, lw=1.5)
    fig.savefig(pdfoutputfile, format='pdf')
    plt.close()


def append_cell_to_output(cell, outcellid, t_model, listout, xlist, ylist):
    dist = math.sqrt(
        float(cell['posx']) ** 2 + float(cell['posy']) ** 2 +
        float(cell['posz']) ** 2)
    velocity = float(dist) / float(t_model) / 86400. / 1.e5

    listout.append(
        '{0:6d}  {1:8.2f}  {2:8.5f}  {3:.5f}  '
        '{4:.5f}  {5:.5f}  {6:.5f}  {7:.5f}'
        .format(outcellid, velocity,
                math.log10(max(float(cell['rho']), 1e-100)),
                cell['ffe'], cell['f56ni'], cell['fco'],
                cell['f52fe'], cell['f48cr']))

    xlist.append(velocity)
    ylist.append(cell['f56ni'])


if __name__ == "__main__":
    main()
