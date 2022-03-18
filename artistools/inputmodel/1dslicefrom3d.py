#!/usr/bin/env python3
import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
from artistools import CustomArgHelpFormatter


def addargs(parser):
    parser.add_argument('-inputfolder', action='store', default='.',
                        help='Path to folder with 3D files')

    parser.add_argument('-axis', action='store', dest='chosenaxis', default='x', choices=['x', 'y', 'z'],
                        help='Slice axis (x, y, or z)')

    parser.add_argument('-outputfolder', action='store', default='1dslice',
                        help='Path to folder in which to store 1D output files')

    parser.add_argument('-opdf', action='store', dest='pdfoutputfile',
                        help='Path/filename for PDF plot.')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=CustomArgHelpFormatter,
            description='Covert abundances.txt and model.txt from 3D to a one dimensional slice.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args()

    if not os.path.exists(args.outputfolder):
        os.makedirs(args.outputfolder)
    else:
        if os.path.exists(os.path.join(args.outputfolder, 'model.txt')):
            print('ABORT: model.txt already exists')
            sys.exit()
        elif os.path.exists(os.path.join(args.outputfolder, 'abundances.txt')):
            print('ABORT: abundances.txt already exists')
            sys.exit()

    dict3dcellidto1dcellid, xlist, ylists = slice_3dmodel(args.inputfolder, args.outputfolder, args.chosenaxis)

    slice_abundance_file(args.inputfolder, args.outputfolder, dict3dcellidto1dcellid)

    if args.pdfoutputfile:
        make_plot(xlist, ylists, args.pdfoutputfile)


def slice_3dmodel(inputfolder, outputfolder, chosenaxis):
    xlist = []
    ylists = [[], [], []]
    listout = []
    dict3dcellidto1dcellid = {}
    outcellid = 0
    with open(os.path.join(inputfolder, 'model.txt'), 'r') as fmodelin:
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
                (cell['cellid'], cell['pos_x_min'], cell['pos_y_min'], cell['pos_z_min'], cell['rho']) = blocksplit[0]
            else:
                print("Wrong line size")
                sys.exit()

            if len(blocksplit[1]) == 5:
                (cell['ffe'], cell['f56ni'], cell['fco'], cell['f52fe'], cell['f48cr']) = map(float, blocksplit[1])
            else:
                print("Wrong line size")
                sys.exit()

            if cell['pos_x_min'] != "0.0000000" and (chosenaxis != 'x' or float(cell['pos_x_min']) < 0.):
                pass
            elif cell['pos_y_min'] != "0.0000000" and (chosenaxis != 'y' or float(cell['pos_y_min']) < 0.):
                pass
            elif cell['pos_z_min'] != "0.0000000" and (chosenaxis != 'z' or float(cell['pos_z_min']) < 0.):
                pass
            else:
                outcellid += 1
                dict3dcellidto1dcellid[int(cell['cellid'])] = outcellid
                append_cell_to_output(cell, outcellid, t_model, listout, xlist, ylists)
                print(f"Cell {outcellid:4d} input1: {block[0].rstrip()}")
                print(f"Cell {outcellid:4d} input2: {block[1].rstrip()}")
                print(f"Cell {outcellid:4d} output: {listout[-1]}")

    with open(os.path.join(outputfolder, 'model.txt'), 'w') as fmodelout:
        fmodelout.write(f"{outcellid:7d}\n")
        fmodelout.write(t_model)
        for line in listout:
            fmodelout.write(line + "\n")

    return dict3dcellidto1dcellid, xlist, ylists


def slice_abundance_file(inputfolder, outputfolder, dict3dcellidto1dcellid):
    with open(os.path.join(inputfolder, 'abundances.txt'), 'r') as fabundancesin, \
            open(os.path.join(outputfolder, 'abundances.txt'), 'w') as fabundancesout:
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
                if int(linesplit[0]) in dict3dcellidto1dcellid:
                    outcellid = dict3dcellidto1dcellid[int(linesplit[0])]
                    currentblock[0] = f"{outcellid:6d}"
                    keepcurrentblock = True
            else:
                currentblock.append(linesplit)

    if keepcurrentblock:
        print("WARNING: unfinished block")


def append_cell_to_output(cell, outcellid, t_model, listout, xlist, ylists):
    dist = math.sqrt(float(cell['pos_x_min']) ** 2 + float(cell['pos_y_min']) ** 2 + float(cell['pos_z_min']) ** 2)
    velocity = float(dist) / float(t_model) / 86400. / 1.e5

    listout.append(
        f"{outcellid:6d}  {velocity:8.2f}  {math.log10(max(float(cell['rho']), 1e-100)):8.5f}  "
        f"{cell['ffe']:.5f}  {cell['f56ni']:.5f}  {cell['fco']:.5f}  {cell['f52fe']:.5f}  {cell['f48cr']:.5f}")

    xlist.append(velocity)
    ylists[0].append(cell['rho'])
    ylists[1].append(cell['f56ni'])
    ylists[2].append(cell['fco'])


def make_plot(xlist, ylists, pdfoutputfile):
    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharey=True, figsize=(6, 4), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    axis.set_xlabel(r'v (km/s)')
    axis.set_ylabel(r'Density (g/cm$^3$) or mass fraction')
    ylabels = [r'$\rho$', 'fNi56', 'fCo']
    for ylist, ylabel in zip(ylists, ylabels):
        axis.plot(xlist, ylist, linewidth=1.5, label=ylabel)
    axis.set_yscale("log", nonposy='clip')
    axis.legend(loc='best', handlelength=2, frameon=False,
                numpoints=1, prop={'size': 10})
    fig.savefig(pdfoutputfile, format='pdf')
    print(f'Saved {pdfoutputfile}')
    plt.close()


if __name__ == "__main__":
    main()
