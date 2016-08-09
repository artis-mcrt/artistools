#!/usr/bin/env python3
import argparse
import math
import sys
import matplotlib.pyplot as plt
import readartisfiles as af
import matplotlib.ticker as ticker
from astropy import constants as const

T_exc = 6000.

def main():
    parser = argparse.ArgumentParser(
        description='Plot ARTIS non-LTE corrections.')
    parser.add_argument('-in', action='store', dest='nltefile',
                        default='nlte_0000.out',
                        help='Path to nlte_*.out file.')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=22,
                        help='Plotted timestep (-1 for last timestep)')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotnlte.pdf',
                        help='path/filename for PDF file')
    args = parser.parse_args()

    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        make_plot(args)


def make_plot(args):
    elementlist = af.get_composition_data('compositiondata.txt')
    nions = int(elementlist.iloc[0]['nions']) - 1  # top ion has 1 level, so not output
    all_levels = af.get_levels('adata.txt')

    list_ltepop = [[] for _ in range(nions)]
    list_nltepop = [[] for _ in range(nions)]
    list_levels = [[] for _ in range(nions)]
    skip_block = False
    selected_timestep = args.timestep
    print("Reading from timestep {:}".format(selected_timestep))
    with open(args.nltefile, 'r') as nltefile:
        currenttimestep = -1
        for line in nltefile:
            row = line.split()

            if row and row[0] == 'timestep':
                currenttimestep = int(row[1])
                if args.timestep in [-1, currenttimestep]:
                    selected_timestep = currenttimestep
                    # reset the lists, since they apply to a previous
                    # timestep
                    list_ltepop = [[] for _ in range(nions)]
                    list_nltepop = [[] for _ in range(nions)]
                    list_levels = [[] for _ in range(nions)]
                    list_odd_levels = [[] for _ in range(nions)]
                    list_ltepop_custom = [[] for _ in range(nions)]
                    skip_block = False
                else:
                    skip_block = True

            if (len(row) > 2 and row[0] == 'nlte_index' and
                    row[1] != '-' and not skip_block):

                ion = int(row[row.index('ion') + 1])
                if row[row.index('level') + 1] != 'SL':
                    levelnumber = int(row[row.index('level') + 1])
                else:
                    levelnumber = list_levels[ion][-1] + 3
                    print("Superlevel at level {:}".format(levelnumber))
                nltepop = float(row[row.index('nnlevel_NLTE') + 1])
                ltepop = float(row[row.index('nnlevel_LTE') + 1])
                if ion < len(list_ltepop):
                    list_levels[ion].append(levelnumber)
                    list_nltepop[ion].append(nltepop)
                    list_ltepop[ion].append(ltepop)

                    level = all_levels[ion].level_list[levelnumber]
                    gslevel = all_levels[ion].level_list[0]
                    k_B = const.k_B.to('eV / K').value
                    ltepop_custom = list_nltepop[ion][0] * level.g / gslevel.g * math.exp(-(level.energy_ev - gslevel.energy_ev) / k_B / T_exc)
                    list_ltepop_custom[ion].append(ltepop_custom)

                    hillier_name = level.hillier_name.split('[')[0]
                    parity = 1 if hillier_name[-1] == 'o' else 0
                    if parity == 1:
                        list_odd_levels[ion].append(levelnumber)
            elif len(row) > 1 and row[1] == '-' and not skip_block:
                ion = int(row[row.index('ion') + 1])
                if ion < len(list_ltepop):
                    levelnumber = int(row[row.index('level') + 1])
                    groundpop = float(row[row.index('nnlevel_LTE') + 1])

                    list_levels[ion].append(levelnumber)
                    list_nltepop[ion].append(groundpop)
                    list_ltepop[ion].append(groundpop)
                    list_ltepop_custom[ion].append(groundpop)

    fig, axes = plt.subplots(nions, 1, sharex=False, figsize=(8, 7),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if not list_levels[ion]:
        print("Error, no data for selected timestep")
        sys.exit()
    for ion, axis in enumerate(axes):
        axis.plot(list_levels[ion], list_ltepop[ion], lw=1.5,
                  label='LTE', linestyle='None', marker='+')

        axis.plot(list_levels[ion][:-1], list_ltepop_custom[ion][:-1], lw=1.5,
                  label='LTE {0:.0f} K'.format(T_exc), linestyle='None', marker='*')
        axis.plot(list_levels[ion], list_nltepop[ion], lw=1.5,
                  label='NLTE', linestyle='None', marker='x')

        list_odd_levels_no_sl = [l for l in list_odd_levels[ion] if l != list_levels[ion][-1]]
        list_nltepop_oddonly = [list_nltepop[ion][i] for i in range(len(list_levels[ion])) if list_levels[ion][i] in list_odd_levels_no_sl]

        axis.plot(list_odd_levels_no_sl, list_nltepop_oddonly, lw=2, label='Odd parity',
                  linestyle='None', marker='s', markersize=10, color='None')

        # list_departure_ratio = [
        #     nlte / lte for (nlte, lte) in zip(list_nltepop[ion],
        #                                       list_ltepop[ion])]
        # ax.plot(list_levels[ion], list_departure_ratio, lw=1.5,
        #         label='NLTE/LTE', linestyle='None', marker='x')
        # ax.set_ylabel(r'')
        plotlabel = "Fe {0}".format(af.roman_numerals[ion + 1])
        time_days = af.get_timestep_time('spec.out', selected_timestep)
        if time_days != -1:
            plotlabel += ' at t={0} days'.format(time_days)
        else:
            plotlabel += ' at timestep {0:d}'.format(selected_timestep)

        axis.annotate(plotlabel, xy=(0.5, 0.96), xycoords='axes fraction',
                      horizontalalignment='center', verticalalignment='top',
                      fontsize=12)
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

    for axis in axes:
        # ax.set_xlim(xmin=270,xmax=300)
        # ax.set_ylim(ymin=-0.1,ymax=1.3)
        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
        axis.set_yscale('log')
    axes[-1].set_xlabel(r'Level number')

    fig.savefig(args.outputfile, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
