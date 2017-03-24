#!/usr/bin/env python3
import argparse
# import math
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import readartisfiles as af


# from astropy import constants as const


def main():
    parser = argparse.ArgumentParser(
        description='Plot ARTIS non-LTE corrections.')
    parser.add_argument('-in', action='store', dest='nltefile',
                        default='nlte_0000.out',
                        help='Path to nlte_*.out file.')
    parser.add_argument('-listtimesteps', action='store_true', default=False,
                        help='Show the times at each timestep')
    parser.add_argument('-timestep', type=int, default=70,
                        help='Plotted timestep')
    parser.add_argument('-modelgridindex', type=int, default=0,
                        help='Plotted modelgrid cell')
    parser.add_argument('element', nargs='?', default='Fe',
                        help='Plotted element')
    parser.add_argument('--oldformat', default=False, action='store_true',
                        help='Use the old file format')
    parser.add_argument('-o', action='store', dest='outputfile',
                        default='plotnlte_{0}_{1:03d}.pdf',
                        help='path/filename for PDF file .format(elsymbol, timestep)')
    args = parser.parse_args()

    if args.listtimesteps:
        af.showtimesteptimes('spec.out')
    else:
        exc_temperature = 6000.
        try:
            atomic_number = next(Z for Z, elsymb in enumerate(af.elsymbols) if elsymb.lower() == args.element.lower())
        except StopIteration:
            print(f"Could not find element '{args.element}'")
            return

        print(f'Getting level populations for modelgrid cell {args.modelgridindex} '
              f'timestep {args.timestep} element {args.element}')

        if not args.oldformat:
            dfpop = af.get_nlte_populations(args.nltefile, args.modelgridindex, args.timestep,
                                            atomic_number, exc_temperature)
        else:
            dfpop = af.get_nlte_populations_oldformat(args.nltefile, args.modelgridindex, args.timestep,
                                                      atomic_number, exc_temperature)

        if dfpop.empty:
            print(f'No data for modelgrid cell {args.modelgridindex} timestep {args.timestep}')
        else:
            make_plot(dfpop, atomic_number, exc_temperature, args)


def make_plot(dfpop, atomic_number, exc_temperature, args):
    top_ion = -1 if args.oldformat else -2
    ion_stage_list = dfpop.ion_stage.unique()[:top_ion]  # skip top ion, which is probably ground state only
    fig, axes = plt.subplots(len(ion_stage_list), 1, sharex=False, figsize=(8, 7),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if len(dfpop) == 0:
        print('Error, no data for selected timestep and element')
        sys.exit()
    for ion, axis in enumerate(axes):
        ion_stage = ion_stage_list[ion]
        dfpopion = dfpop.query('ion_stage==@ion_stage')

        axis.plot(dfpopion.level.values, dfpopion.n_LTE.values, lw=1.5, label='LTE', linestyle='None', marker='+')

        axis.plot(dfpopion.level.values[:-1], dfpopion.n_LTE_custom.values[:-1], lw=1.5,
                  label=f'LTE {exc_temperature:.0f} K', linestyle='None', marker='*')
        axis.plot(dfpopion.level.values, dfpopion.n_NLTE.values, lw=1.5,
                  label='NLTE', linestyle='None', marker='x')

        dfpopionoddlevels = dfpopion.query('parity==1')

        axis.plot(dfpopionoddlevels.level.values, dfpopionoddlevels.n_NLTE.values, lw=2, label='Odd parity',
                  linestyle='None', marker='s', markersize=10, markerfacecolor=(0, 0, 0, 0), markeredgecolor='black')

        # list_departure_ratio = [
        #     nlte / lte for (nlte, lte) in zip(list_nltepop[ion],
        #                                       list_ltepop[ion])]
        # axis.plot(list_levels[ion], list_departure_ratio, lw=1.5,
        #         label='NLTE/LTE', linestyle='None', marker='x')
        # axis.set_ylabel(r'')
        plotlabel = f'{af.elsymbols[atomic_number]} {af.roman_numerals[ion_stage]}'
        time_days = af.get_timestep_time('spec.out', args.timestep)
        if time_days != -1:
            plotlabel += f' at t={time_days} days'
        else:
            plotlabel += f' at timestep {args.timestep:d}'

        axis.annotate(plotlabel, xy=(0.5, 0.96), xycoords='axes fraction',
                      horizontalalignment='center', verticalalignment='top', fontsize=12)
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=1))

    for axis in axes:
        # axis.set_xlim(xmin=270,xmax=300)
        # axis.set_ylim(ymin=-0.1,ymax=1.3)
        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
        axis.set_yscale('log')
    axes[-1].set_xlabel(r'Level index')

    outputfilename = args.outputfile.format(af.elsymbols[atomic_number], args.timestep)
    print(f"Saving {outputfilename}")
    fig.savefig(outputfilename, format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
