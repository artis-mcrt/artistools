#!/usr/bin/env python3
import argparse
import math
import sys
import matplotlib.pyplot as plt
import readartisfiles as af
import matplotlib.ticker as ticker
from astropy import constants as const


def main():
    parser = argparse.ArgumentParser(
        description='Plot ARTIS non-LTE corrections.')
    parser.add_argument('-in', action='store', dest='nltefile',
                        default='nlte_0000.out',
                        help='Path to nlte_*.out file.')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=22,
                        help='Plotted timestep')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotnlte_{0:03d}.pdf',
                        help='path/filename for PDF file')
    args = parser.parse_args()

    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        make_plot(args)


def make_plot(args):
    T_exc = 6000.
    print("Reading from timestep {:}".format(args.timestep))
    dfpop = af.get_nlte_populations(args.nltefile, args.timestep, 26, T_exc)
    print(dfpop)

    ion_stage_list = dfpop.ion_stage.unique()
    fig, axes = plt.subplots(len(ion_stage_list), 1, sharex=False, figsize=(8, 7),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if len(dfpop) == 0:
        print("Error, no data for selected timestep")
        sys.exit()
    for ion, axis in enumerate(axes):
        ion_stage = ion_stage_list[ion]
        dfpopion = dfpop.query('ion_stage==@ion_stage')

        axis.plot(dfpopion.level.values, dfpopion.pop_lte.values, lw=1.5, label='LTE', linestyle='None', marker='+')

        axis.plot(dfpopion.level.values[:-1], dfpopion.pop_ltecustom.values[:-1], lw=1.5, label='LTE {0:.0f} K'.format(T_exc), linestyle='None', marker='*')
        axis.plot(dfpopion.level.values, dfpopion.pop_nlte.values, lw=1.5,
                  label='NLTE', linestyle='None', marker='x')

        dfpopionoddlevels = dfpopion.query('parity==1')

        axis.plot(dfpopionoddlevels.level.values, dfpopionoddlevels.pop_nlte.values, lw=2, label='Odd parity',
                  linestyle='None', marker='s', markersize=10, color='None')

        # list_departure_ratio = [
        #     nlte / lte for (nlte, lte) in zip(list_nltepop[ion],
        #                                       list_ltepop[ion])]
        # ax.plot(list_levels[ion], list_departure_ratio, lw=1.5,
        #         label='NLTE/LTE', linestyle='None', marker='x')
        # ax.set_ylabel(r'')
        plotlabel = "Fe {0}".format(af.roman_numerals[ion_stage])
        time_days = af.get_timestep_time('spec.out', args.timestep)
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
    axes[-1].set_xlabel(r'Level index')

    fig.savefig(args.outputfile.format(args.timestep), format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
