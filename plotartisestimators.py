#!/usr/bin/env python3
import math
import matplotlib.pyplot as plt
# import numpy as np
import readartisfiles as af
# from astropy import constants as const

# h = const.h.to('J s').value
# c = const.c.to('m/s').value

selectedtimestep = 70  # -1 means all time steps


def main():
    timesteptimes = []
    with open('light_curve.out', 'r') as lcfile:
        for line in lcfile:
            timesteptimes.append(line.split()[0])

    elementlist = af.get_composition_data('compositiondata.txt')
    modeldata = af.get_modeldata('model.txt')
    initalabundances = af.get_initialabundances1d('abundances.txt')

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
                if timestep == selectedtimestep:
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
                            nions = elementlist.nions[currentelementindex]
                            list_populations[-1].append(list(map(float, row[colnum:colnum + nions])))
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
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 6),
                             tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})
    list_velocity = [modeldata[mgi].velocity for mgi in sorted(list_modelgridindex)]

    for elindex, element in elementlist.iterrows():
        # if element['Z'] != 8:
        #     continue
        axis = axes[1 + elindex]
        axis.set_ylabel(r'Population fraction')
        for ion in range(elementlist['nions'][elindex]):
            if element['Z'] == 26 and ion == 0:
                continue
            ylist = []
            for mgi in sorted(list_modelgridindex):
                for index, thismgi in enumerate(list_modelgridindex):
                    total_pop = sum([ionpop for ions in list_populations[index] for ionpop in ions])
                    # el_pop = sum(list_populations[index][elindex])
                    if thismgi == mgi:
                        ylist.append(list_populations[index][elindex][ion] / total_pop)
            axis.plot(list_velocity, ylist, lw=1.5, label="{0} {1}".format(
                af.elsymbols[elementlist['Z'][elindex]], af.roman_numerals[ion + 1]))

    # axis.plot(list_timestep, [x[0][0] for x in list_populations],
    #           lw=1.5, label="Fe I")
    # axis.plot(list_timestep, [x[0][1] for x in list_populations],
    #           lw=1.5, label="Fe II")
    # axis.plot(list_timestep, [x[0][2] for x in list_populations],
    #           lw=1.5, label="Fe III")
    # axis.plot(list_timestep, [x[0][3] for x in list_populations],
    #           lw=1.5, label="Fe IV")
    # axis.plot(list_timestep, [x[0][4] for x in list_populations],
    #           lw=1.5, label="Fe V")
    plotlabel = 't={}d'.format(timesteptimes[selectedtimestep])
    axes[1].annotate(plotlabel, xy=(0.1, 0.96), xycoords='axes fraction',
                     horizontalalignment='left', verticalalignment='top',
                     fontsize=12)

    list_abund_o = [initalabundances[mgi][8] for mgi in list_modelgridindex]
    axes[0].plot(list_velocity, list_abund_o, lw=1.5, label="O")
    list_abund_ni = [initalabundances[mgi][28] for mgi in list_modelgridindex]
    axes[0].plot(list_velocity, list_abund_ni, lw=1.5, label="Ni")
    axes[0].set_ylabel(r'Mass fraction')
    plotlabel = 'Initial abundances'
    axes[0].annotate(plotlabel, xy=(0.5, 0.96), xycoords='axes fraction',
                     horizontalalignment='center', verticalalignment='top',
                     fontsize=12)

    for axis in axes:
        # axis.set_xlim(xmin=270,xmax=300)
        # axis.set_ylim(ymin=-0.1,ymax=1.3)
        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1,
                    prop={'size': 9})

    axes[-1].set_xlabel(r'Velocity [km/s]')
    # axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=5))

    fig.savefig('plotestimators.pdf', format='pdf')
    plt.close()


if __name__ == "__main__":
    main()
