#!/usr/bin/env python3
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import numpy as np
import readartisfiles as af

h = 6.62607004e-34  # m^2 kg / s
c = 299792458  # m / s

selectedtimestep = 22  # -1


def main():
    timesteptimes = []
    with open('light_curve.out', 'r') as lcfile:
        for line in lcfile:
            timesteptimes.append(line.split()[0])
    timesteptimes = timesteptimes[:len(timesteptimes) // 2]

    elementlist = af.getartiselementlist('compositiondata.txt')
    nions = elementlist[0].nions - 1

    list_ltepop = [[] for _ in range(nions)]
    list_nltepop = [[] for _ in range(nions)]
    list_levels = [[] for _ in range(nions)]
    skip_block = False
    with open('nlte_0000.out', 'r') as nltefile:
        currenttimestep = -1
        for line in nltefile:
            row = line.split()

            if row and row[0] == 'timestep':
                currenttimestep = int(row[1])
                if selectedtimestep in [-1, currenttimestep]:
                    # reset the lists, since they apply to a previous
                    # timestep
                    list_ltepop = [[] for _ in range(nions)]
                    list_nltepop = [[] for _ in range(nions)]
                    list_levels = [[] for _ in range(nions)]
                    skip_block = False
                else:
                    skip_block = True

            if (len(row) > 2 and row[0] == 'nlte_index' and
                    row[1] != '-' and not skip_block):

                ion = int(row[row.index('ion') + 1])
                level = int(row[row.index('level') + 1])
                nltepop = float(row[row.index('nnlevelnlte') + 1])
                ltepop = float(row[row.index('nnlevellte') + 1])
                list_ltepop[ion].append(ltepop)
                list_nltepop[ion].append(nltepop)
                list_levels[ion].append(level)

    fig, axes = plt.subplots(nions, 1, sharex=False, figsize=(
        8, 10), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    for ion, ax in enumerate(axes):
        ax.plot(list_levels[ion], list_ltepop[ion], lw=1.5,
                label='LTE', linestyle='None', marker='+')
        ax.plot(list_levels[ion], list_nltepop[ion], lw=1.5,
                label='NLTE', linestyle='None', marker='x')
        # list_departure_ratio = [
        #     nlte / lte for (nlte, lte) in zip(list_nltepop[ion],
        #                                       list_ltepop[ion])]
        # ax.plot(list_levels[ion], list_departure_ratio, lw=1.5,
        #         label='NLTE/LTE', linestyle='None', marker='x')
        # ax.set_ylabel(r'')
        plotlabel = "Fe {0}".format(af.roman_numerals[ion + 1])
        plotlabel += ' at t={0}d'.format(timesteptimes[selectedtimestep])
        ax.annotate(plotlabel, xy=(0.5, 0.96), xycoords='axes fraction',
                    horizontalalignment='center', verticalalignment='top',
                    fontsize=12)

    for ax in axes:
        # ax.set_xlim(xmin=270,xmax=300)
        # ax.set_ylim(ymin=-0.1,ymax=1.3)
        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1,
                  prop={'size': 9})
        ax.set_yscale('log')
    axes[-1].set_xlabel(r'Level number')
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    fig.savefig('plotnlte.pdf', format='pdf')
    plt.close()

if __name__ == "__main__":
    main()
