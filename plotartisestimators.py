#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
# import numpy as np
import readartisfiles as af
# from astropy import constants as const

# h = const.h.to('J s').value
# c = const.c.to('m/s').value

selectedtimesteps = [-1]  # -1 means all time steps


def main():
    timesteptimes = []
    with open('light_curve.out', 'r') as lcfile:
        for line in lcfile:
            timesteptimes.append(line.split()[0])

    elementlist = af.getartiselementlist('compositiondata.txt')
    # modeldata = af.getmodeldata('model.txt')
    # initalabundances = af.getinitialabundances1d('abundances.txt')

    list_timestep = []
    list_modelgridindex = []
    list_grey_depth = []
    list_te = []
    list_w = []
    list_populations = []
    list_nne = []
    list_heating_bf = []
    list_cooling_bf = []
    list_cooling_ff = []
    list_cooling_coll = []
    skip_block = False
    with open('estimators_0000.out', 'r') as estfile:
        linenum = 0
        for line in estfile:
            if line.startswith('0 0 0 0 0 0 0 0'):
                continue
            linenum += 1
            row = line.split()
            if linenum < 7:
                print(linenum, row)
            if linenum % 7 == 1:
                timestep = int(row[1])
                if timestep in [-1, selectedtimesteps or -1]:
                    skip_block = False
                    list_timestep.append(timestep)
                    list_modelgridindex.append(int(row[3]))
                    list_grey_depth.append(row[13])
                    list_te.append(row[7])
                    list_w.append(math.log10(float(row[9])))
                    list_nne.append(float(row[15]))
                else:
                    skip_block = True
            if not skip_block:
                if linenum % 7 == 2:
                    if row[0] != 'populations':
                        print('not a populations row?')
                    else:
                        colnum = 2
                        list_populations.append([])
                        currentelementindex = 0
                        while colnum < len(row):
                            nions = elementlist[currentelementindex].nions
                            list_populations[-1].append(
                                list(map(float, row[colnum:colnum + nions])))
                            colnum += (1 + nions)
                            currentelementindex += 1
                        print(row)
                        print(list_populations[-1])
                if linenum % 7 == 5:
                    list_heating_bf.append(
                        math.log10(max(1e-20, float(row[4]))))
                if linenum % 7 == 6:
                    list_cooling_ff.append(
                        math.log10(max(1e-15, float(row[2]))))
                    list_cooling_bf.append(
                        math.log10(max(1e-15, float(row[4]))))
                    list_cooling_coll.append(
                        math.log10(max(1e-15, float(row[6]))))

    timesteptimes = timesteptimes[:len(timesteptimes) // 2]
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4),
                           tight_layout={
                               "pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    # list_velocity = [modeldata[mgi].velocity for mgi in list_modelgridindex]
    axes = [ax]
    # print(list_timestep,list_populations)
    for elindex, element in enumerate(elementlist):
        el_pop = [sum(x[elindex]) for x in list_populations]
        for ion in range(element.nions):
            ax.plot(list_timestep, [
                pop_ts[elindex][ion] / el_pop[tsindex]
                for tsindex, pop_ts in enumerate(list_populations)],
                    lw=1.5,
                    label="{0} {1}".format(
                        af.elsymbols[element.Z], af.roman_numerals[ion + 1]))

    # ax.plot(list_timestep, [x[0][0] for x in list_populations],
    #         lw=1.5, label="Fe I")
    # ax.plot(list_timestep, [x[0][1] for x in list_populations],
    #         lw=1.5, label="Fe II")
    # ax.plot(list_timestep, [x[0][2] for x in list_populations],
    #         lw=1.5, label="Fe III")
    # ax.plot(list_timestep, [x[0][3] for x in list_populations],
    #         lw=1.5, label="Fe IV")
    # ax.plot(list_timestep, [x[0][4] for x in list_populations],
    #         lw=1.5, label="Fe V")
    ax.set_ylabel(r'Population fraction')
    # plotlabel = 't={}d'.format(timesteptimes[selectedtimestep])
    # axes[1].annotate(plotlabel, xy=(0.1,0.96), xycoords='axes fraction',
    #                  horizontalalignment='left', verticalalignment='top',
    #                  fontsize=12)

    """
    list_abund_o = [initalabundances[mgi][8] for mgi in list_modelgridindex]
    axes[1].plot(list_velocity, list_abund_o, lw=1.5, label="O")
    list_abund_ni = [initalabundances[mgi][28] for mgi in list_modelgridindex]
    axes[1].plot(list_velocity, list_abund_ni, lw=1.5, label="Ni")
    axes[1].set_ylabel(r'Mass fraction')
    plotlabel = 'Initial abundances'
    axes[1].annotate(plotlabel, xy=(0.5,0.96), xycoords='axes fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=12)
    """

    for ax in axes:
        #      pass
        # ax.set_xlim(xmin=270,xmax=300)
        # ax.set_ylim(ymin=-0.1,ymax=1.3)
        ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1,
                  prop={'size': 9})

    axes[-1].set_xlabel(r'Timestep')
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    fig.savefig('plotestimators.pdf', format='pdf')
    plt.close()

if __name__ == "__main__":
    main()
