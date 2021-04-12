#!/usr/bin/env python3

import argparse
# import glob
# import itertools
import math
import multiprocessing
import os
# import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import artistools as at
import artistools.spectra
import matplotlib.pyplot as plt
import matplotlib
from extinction import apply, ccm89
from astropy import constants as const

from matplotlib.legend_handler import HandlerTuple
from .lightcurve import *
import glob

color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))

define_colours_list = ['k', 'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold',
                       'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green', 'purple',
                       'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k',
                       'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold',
                       'tab:cyan',
                       'darkblue', 'bisque', 'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange',
                       'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k', 'tab:blue',
                       'tab:red',
                       'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue',
                       'bisque',
                       'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink',
                       'tab:gray',
                       'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k', 'tab:blue', 'tab:red', 'tab:green',
                       'purple',
                       'tab:orange', 'tab:pink', 'tab:gray', 'gold', 'tab:cyan', 'darkblue', 'bisque', 'yellow', 'k',
                       'tab:blue', 'tab:red', 'tab:green', 'purple', 'tab:orange', 'tab:pink', 'tab:gray', 'gold',
                       'tab:cyan',
                       'darkblue', 'bisque', 'yellow']

define_colours_list2 = ['gray', 'lightblue', 'pink', 'yellowgreen', 'mediumorchid', 'sandybrown', 'plum', 'lightgray',
                        'wheat', 'paleturquoise']


def make_lightcurve_plot_from_lightcurve_out_files(modelpaths, filenameout, frompackets=False,
                                                   escape_type=False, maxpacketfiles=None, args=None):
    """Use light_curve.out or light_curve_res.out files to plot light curve"""
    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if not frompackets and escape_type not in ['TYPE_RPKT', 'TYPE_GAMMA']:
        print(f'Escape_type of {escape_type} not one of TYPE_RPKT or TYPE_GAMMA, so frompackets must be enabled')
        assert False
    elif not frompackets and args.packet_type != 'TYPE_ESCAPE' and args.packet_type is not None:
        print(f'Looking for non-escaped packets, so frompackets must be enabled')
        assert False

    for seriesindex, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")
        lcname = 'gamma_light_curve.out' if (escape_type == 'TYPE_GAMMA' and not frompackets) else 'light_curve.out'
        if args.plotviewingangle and lcname == 'light_curve.out':
            lcname = 'light_curve_res.out'
        elif args.plotviewingangle:
            print("If you're trying to plot gamma_res - sorry haven't written that yet")
            quit()
        try:
            lcpath = at.firstexisting([lcname + '.xz', lcname + '.gz', lcname], path=modelpath)
        except FileNotFoundError:
            print(f"Skipping {modelname} because {lcname} does not exist")
            continue
        if not os.path.exists(str(lcpath)):
            print(f"Skipping {modelname} because {lcpath} does not exist")
            continue
        elif frompackets:
            lcdata = at.lightcurve.get_from_packets(
                modelpath, lcpath, packet_type=args.packet_type, escape_type=escape_type, maxpacketfiles=maxpacketfiles)
        else:
            lcdata = at.lightcurve.readfile(lcpath, args)

        plotkwargs = {}
        if args.label[seriesindex] is None:
            plotkwargs['label'] = modelname
        else:
            plotkwargs['label'] = args.label[seriesindex]

        plotkwargs['linestyle'] = args.linestyle[seriesindex]
        plotkwargs['color'] = args.color[seriesindex]
        if args.dashes[seriesindex]:
            plotkwargs['dashes'] = args.dashes[seriesindex]
        if args.linewidth[seriesindex]:
            plotkwargs['linewidth'] = args.linewidth[seriesindex]

        # check if doing viewing angle stuff, and if so define which data to use
        angles, viewing_angles, angle_definition = get_angle_stuff(modelpath, args)
        if args.plotviewingangle:
            lcdataframes = lcdata

            if args.colorbarcostheta or args.colorbarphi:
                costheta_viewing_angle_bins, phi_viewing_angle_bins = get_viewinganglebin_definitions()
                scaledmap = make_colorbar_viewingangles_colormap()

        for angleindex, angle in enumerate(angles):
            if args.plotviewingangle:
                lcdata = lcdataframes[angle]

                if args.colorbarcostheta or args.colorbarphi:
                    plotkwargs['alpha'] = 0.75
                    # Update plotkwargs with viewing angle colour
                    plotkwargs = get_viewinganglecolor_for_colorbar(angle_definition, angle,
                                                        costheta_viewing_angle_bins, phi_viewing_angle_bins,
                                                        scaledmap, plotkwargs, args)
                else:
                    plotkwargs['color'] = None
                    plotkwargs['label'] = f'{modelname}\n{angle_definition[angle]}'

            filterfunc = at.get_filterfunc(args)
            if filterfunc is not None:
                lcdata['lum'] = filterfunc(lcdata['lum'])

            if args.ergs or args.magnitude:
                lcdata['lum'] = lcdata['lum']*3.826e33  # Luminosity in erg/s

            if args.magnitude:
                # convert to bol magnitude
                lcdata['mag'] = 4.74 - (2.5 * np.log10(lcdata['lum'] / const.L_sun.to('erg/s').value))
                axis.plot(lcdata['time'], lcdata['mag'], **plotkwargs)
            else:
                axis.plot(lcdata['time'], lcdata['lum'], **plotkwargs)

                if args.print_data:
                    print(lcdata[['time', 'lum', 'lum_cmf']].to_string(index=False))
                if args.plotcmf:
                    plotkwargs['linewidth'] = 1
                    plotkwargs['label'] += ' (cmf)'
                    plotkwargs['color'] = 'tab:orange'
                    axis.plot(lcdata.time, lcdata['lum_cmf'], **plotkwargs)

    if args.reflightcurves:
        for bolreflightcurve in args.reflightcurves:
            if not args.ergs:
                print("Check units - trying to plot ref light curve in erg/s")
                quit()
            bollightcurve_data, metadata = read_bol_reflightcurve_data(bolreflightcurve)
            axis.scatter(bollightcurve_data['time_days'], bollightcurve_data['luminosity_erg/s'],
                         label=metadata['label'], color='k')

    if args.magnitude:
        plt.gca().invert_yaxis()

    if args.xmin is not None:
        axis.set_xlim(left=args.xmin)
    if args.xmax:
        axis.set_xlim(right=args.xmax)
    if args.ymin:
        axis.set_ylim(bottom=args.ymin)
    if args.ymax:
        axis.set_ylim(top=args.ymax)
    # axis.set_ylim(bottom=-0.1, top=1.3)

    if not args.nolegend:
        axis.legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 9})
    axis.set_xlabel(r'Time (days)')

    if args.magnitude:
        axis.set_ylabel('Absolute Bolometric Magnitude')

    elif args.ergs:
        axis.set_ylabel('erg/s')
    else:
        if escape_type == 'TYPE_GAMMA':
            lum_suffix = r'_\gamma'
        elif escape_type == 'TYPE_RPKT':
            lum_suffix = r'_{\mathrm{OVOIR}}'
        else:
            lum_suffix = r'_{\mathrm{' + escape_type.replace("_", r"\_") + '}}'
        axis.set_ylabel(r'$\mathrm{L} ' + lum_suffix + r'/ \mathrm{L}_\odot$')

    if args.colorbarcostheta or args.colorbarphi:
        make_colorbar_viewingangles(costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, args)

    if args.logscaley:
        axis.set_yscale('log')

    fig.savefig(str(filenameout), format='pdf')
    print(f'Saved {filenameout}')
    plt.close()


def create_axes(args):
    if 'labelfontsize' in args:
        font = {'size': args.labelfontsize}
        matplotlib.rc('font', **font)

    args.subplots = False  # todo: set as command line arg

    if (args.filter and len(args.filter) > 1) or args.subplots is True:
        args.subplots = True
        rows = 2
        cols = 3
    elif (args.colour_evolution and len(args.colour_evolution) > 1) or args.subplots is True:
        args.subplots = True
        rows = 1
        cols = 3
    else:
        args.subplots = False
        rows = 1
        cols = 1

    if 'figwidth' not in args:
        args.figwidth = at.figwidth * 1.6 * cols
    if 'figheight' not in args:
        args.figheight = at.figwidth * 1.1 * rows*1.5

    fig, ax = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True,
                           figsize=(args.figwidth, args.figheight),
                           tight_layout={"pad": 3.0, "w_pad": 0.6, "h_pad": 0.6})  # (6.2 * 3, 9.4 * 3)
    if args.subplots:
        ax = ax.flatten()

    return fig, ax


def set_axis_limit_args(args):
    if args.filter:
        plt.gca().invert_yaxis()
        if args.ymax is None:
            args.ymax = -20
        if args.ymin is None:
            args.ymin = -14

    if args.colour_evolution:
        if args.ymax is None:
            args.ymax = 1
        if args.ymin is None:
            args.ymin = -1

    if args.filter or args.colour_evolution:
        if args.xmax is None:
            args.xmax = 100
        if args.xmin is None:
            args.xmin = 0
        if args.timemax is None:
            args.timemax = args.xmax + 5
        if args.timemin is None:
            args.timemin = args.xmin - 5


def get_angle_stuff(modelpath, args):
    viewing_angles = None
    viewing_angle_data = False
    if len(glob.glob(str(Path(modelpath) / '*_res.out'))) > 1:
        viewing_angle_data = True

    if args.plotvspecpol and os.path.isfile(modelpath / 'vpkt.txt'):
        angles = args.plotvspecpol
    elif args.plotviewingangle and args.plotviewingangle[0] == -1 and viewing_angle_data:
        angles = np.arange(0, 100, 1, dtype=int)
    elif args.plotviewingangle and viewing_angle_data:
        angles = args.plotviewingangle
    elif args.calculate_costheta_phi_from_viewing_angle_numbers and \
            args.calculate_costheta_phi_from_viewing_angle_numbers[0] == -1:
        viewing_angles = np.arange(0, 100, 1, dtype=int)
        calculate_costheta_phi_for_viewing_angles(viewing_angles, modelpath)
    elif args.calculate_costheta_phi_from_viewing_angle_numbers:
        viewing_angles = args.calculate_costheta_phi_from_viewing_angle_numbers
        calculate_costheta_phi_for_viewing_angles(viewing_angles, modelpath)
    else:
        angles = [None]

    angle_definition = None
    if angles[0] is not None:
        angle_definition = calculate_costheta_phi_for_viewing_angles(angles, modelpath)
        if args.average_every_tenth_viewing_angle:
            for key in angle_definition.keys():
                costheta_label = angle_definition[key].split(',')[0]
                angle_definition[key] = costheta_label

    return angles, viewing_angles, angle_definition


def make_colorbar_viewingangles_colormap():
    norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
    scaledmap = matplotlib.cm.ScalarMappable(cmap='inferno', norm=norm)
    scaledmap.set_array([])
    return scaledmap


def get_viewinganglecolor_for_colorbar(angle_definition, angle, costheta_viewing_angle_bins, phi_viewing_angle_bins,
                                       scaledmap, plotkwargs, args):
    if args.colorbarcostheta:
        colorindex = costheta_viewing_angle_bins.index(angle_definition[angle].split(', ')[0])
        plotkwargs['color'] = scaledmap.to_rgba(colorindex)
    if args.colorbarphi:
        colorindex = phi_viewing_angle_bins.index(angle_definition[angle].split(', ')[1])
        plotkwargs['color'] = scaledmap.to_rgba(colorindex)
    return plotkwargs


def make_colorbar_viewingangles(costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, args):
    if args.colorbarcostheta:
        ticklabels = costheta_viewing_angle_bins
    if args.colorbarphi:
        ticklabels = phi_viewing_angle_bins
    cbar = plt.colorbar(scaledmap)
    ticklocs = np.arange(0, 10)
    cbar.locator = matplotlib.ticker.FixedLocator(ticklocs)
    cbar.formatter = matplotlib.ticker.FixedFormatter(ticklabels)
    cbar.update_ticks()


def get_linelabel(modelpath, modelname, modelnumber, angle, angle_definition, args):
    if args.plotvspecpol and angle is not None and os.path.isfile(modelpath / 'vpkt.txt'):
        vpkt_config = at.get_vpkt_config(modelpath)
        viewing_angle = round(math.degrees(math.acos(vpkt_config['cos_theta'][angle])))
        linelabel = fr"$\theta$ = {viewing_angle}"  # todo: update to be consistent with res definition
    elif args.plotviewingangle and angle is not None and os.path.isfile(modelpath / 'specpol_res.out'):
        linelabel = fr"{modelname} {angle_definition[angle]}"
        # linelabel = None
        # linelabel = fr"{modelname} $\theta$ = {angle_names[index]}$^\circ$"
        # plt.plot(time, magnitude, label=linelabel, linewidth=3)
    elif args.label:
        linelabel = fr'{args.label[modelnumber]}'
    else:
        linelabel = f'{modelname}'
        # linelabel = 'Angle averaged'

    if linelabel == 'None' or linelabel is None:
        linelabel = f'{modelname}'

    return linelabel


def set_axis_properties(ax, args):
    if args.subplots:
        for axis in ax:
            # axis.set_xscale('log')
            axis.minorticks_on()
            axis.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2,
                             labelsize=args.labelfontsize, direction='in')
            axis.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2,
                             labelsize=args.labelfontsize, direction='in')

    else:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2, labelsize=args.labelfontsize,
                       direction='in')
        ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2, labelsize=args.labelfontsize,
                       direction='in')

    plt.ylim(args.ymin, args.ymax)
    plt.xlim(args.xmin, args.xmax)

    plt.minorticks_on()
    return ax


def set_lightcurveplot_legend(ax, args):
    if not args.nolegend:
        if args.subplots:
            ax[0].legend(loc='lower left', frameon=True, fontsize='x-small', ncol=1)
        else:
            ax.legend(loc='best', frameon=False, fontsize='small', ncol=1, handlelength=0.7)
    return ax


def set_lightcurve_plot_labels(fig, ax, filternames_conversion_dict, args, band_name=None):
    ylabel = None
    if args.subplots:
        if args.filter:
            ylabel = 'Absolute Magnitude'
        if args.colour_evolution:
            ylabel = r'$\Delta$m'
        fig.text(0.5, 0.025, 'Time Since Explosion [days]', ha='center', va='center')
        fig.text(0.02, 0.5, ylabel , ha='center', va='center', rotation='vertical')
    else:
        if args.filter and band_name in filternames_conversion_dict:
            ylabel = f'{filternames_conversion_dict[band_name]} Magnitude'
        elif args.filter:
            ylabel = f'{band_name} Magnitude'
        elif args.colour_evolution:
            ylabel = r'$\Delta$m'
        ax.set_ylabel(ylabel, fontsize=args.labelfontsize)  # r'M$_{\mathrm{bol}}$'
        ax.set_xlabel('Time Since Explosion [days]', fontsize=args.labelfontsize)
    if ylabel is None:
        print("failed to set ylabel")
        quit()
    return fig, ax


def make_band_lightcurves_plot(modelpaths, filternames_conversion_dict, outputfolder, args):

    # determine if this will be a scatter plot or not
    calculate_peak_time_mag_deltam15_bool = False
    if (args.calculate_peakmag_risetime_delta_m15
            or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file
            or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
            or args.make_viewing_angle_peakmag_risetime_scatter_plot
            or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
        calculate_peak_time_mag_deltam15_bool = True
        if args.timemin is None or args.timemax is None:
            print("Trying to calculate peak time / dm15 / rise time with no time range. "
                  "This will give a stupid result. Specify args.timemin and args.timemax")
            quit()

    if calculate_peak_time_mag_deltam15_bool:  # If there's viewing angle scatter plot stuff define some arrays
        args.plotvalues = []  # a0 and p0 values for viewing angle scatter plots

        args.band_risetime_polyfit = []
        args.band_peakmag_polyfit = []
        args.band_deltam15_polyfit = []

        args.band_risetime_angle_averaged_polyfit = []
        args.band_peakmag_angle_averaged_polyfit = []
        args.band_delta_m15_angle_averaged_polyfit = []

    # angle_names = [0, 45, 90, 180]
    # plt.style.use('dark_background')

    modelnames = [] # save names of models
    args.labelfontsize = 22  #todo: make command line arg
    fig, ax = create_axes(args)
    set_axis_limit_args(args)

    for modelnumber, modelpath in enumerate(modelpaths):
        modelpath = Path(modelpath)  ## Make sure modelpath is defined as path. May not be necessary

        # check if doing viewing angle stuff, and if so define which data to use
        angles, viewing_angles, angle_definition = get_angle_stuff(modelpath, args)

        for index, angle in enumerate(angles):

            modelname = at.get_model_name(modelpath)
            modelnames.append(modelname)  # save for later
            print(f'Reading spectra: {modelname}')
            band_lightcurve_data = get_band_lightcurve_data(modelpath, args, angle, modelnumber=modelnumber)

            if modelnumber == 0 and args.plot_hesma_model:  # Todo: does this work?
                hesma_model = read_hesma_lightcurve(args)
                linelabel = str(args.plot_hesma_model).split('_')[:3]

            for plotnumber, band_name in enumerate(band_lightcurve_data):
                time, brightness_in_mag = get_band_lightcurve_data_to_plot(band_lightcurve_data, band_name, args)

                linelabel = get_linelabel(modelpath, modelname, modelnumber, angle, angle_definition, args)
                # linelabel = '\n'.join(wrap(linelabel, 40))  # todo: could be arg? wraps text in label

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    brightness_in_mag = filterfunc(brightness_in_mag)

                # Calculating band peak time, peak magnitude and delta m15
                if calculate_peak_time_mag_deltam15_bool:
                    calculate_peak_time_mag_deltam15(time, brightness_in_mag, modelname, angle, band_name,
                                                     filternames_conversion_dict, args)

                if args.plotviewingangle and args.plotviewingangles_lightcurves:
                    global define_colours_list
                    plt.plot(time, brightness_in_mag, label=modelname, color=define_colours_list[angle], linewidth=3)

                if modelnumber == 0 and args.plot_hesma_model and band_name in hesma_model.keys():  #todo: see if this works
                    ax.plot(hesma_model.t, hesma_model[band_name], color='black')

                # axarr[plotnumber].axis([0, 60, -16, -19.5])
                if band_name in filternames_conversion_dict:
                    text_key = filternames_conversion_dict[band_name]
                else:
                    text_key = band_name
                if args.subplots:
                    ax[plotnumber].text(args.xmax * 0.8, args.ymax * 0.97, text_key)
                # else:
                #     ax.text(args.xmax * 0.75, args.ymax * 0.95, text_key)

                if not calculate_peak_time_mag_deltam15_bool:  ##Finn does this still work??

                    if args.reflightcurves and modelnumber == 0:
                        if len(angles) > 1 and index > 0:
                            print('already plotted reflightcurve')
                        else:
                            define_colours_list = args.refspeccolors
                            markers = args.refspecmarkers
                            for i, reflightcurve in enumerate(args.reflightcurves):
                                plot_lightcurve_from_data(band_lightcurve_data.keys(), reflightcurve, define_colours_list[i], markers[i],
                                                          filternames_conversion_dict, ax, plotnumber)

                if args.color:
                    color = args.color[modelnumber]
                else:
                    color = define_colours_list[modelnumber]
                if args.linestyle:
                    linestyle = args.linestyle[modelnumber]

                if not (args.test_viewing_angle_fit or calculate_peak_time_mag_deltam15_bool):  ##Finn: does this still work?

                    if args.subplots:
                        # if linestyle == 'dashed':
                        #     alpha = 0.6
                        # else:
                        alpha = 1  # todo: set command line arg for this

                        if len(angles) > 1 or (args.plotviewingangle and os.path.isfile(modelpath / 'specpol_res.out')):
                            ax[plotnumber].plot(time, brightness_in_mag, label=linelabel, linewidth=4, linestyle=linestyle,
                                                alpha=alpha)
                        # I think this was just to have a different line style for viewing angles....
                        else:
                            ax[plotnumber].plot(time, brightness_in_mag, label=linelabel, linewidth=4, color=color,
                                                linestyle=linestyle, alpha=alpha)
                            # if key is not 'bol':
                            #     ax[plotnumber].plot(
                            #         cmfgen_mags['time[d]'], cmfgen_mags[key], label='CMFGEN', color='k', linewidth=3)
                    else:
                        # if 'FM3' in str(modelpath):
                        #     ax.plot(time, magnitude, label=linelabel, linewidth=3, color='darkblue')
                        # elif 'M2a' in str(modelpath):
                        #     ax.plot(time, magnitude, label=linelabel, linewidth=3, color='k')
                        # else:
                        ax.plot(time, brightness_in_mag, label=linelabel, linewidth=3.5)  # color=color, linestyle=linestyle)

        # Saving viewing angle data so it can be read in and plotted later on without re-running the script
        #    as it is quite time consuming
        if calculate_peak_time_mag_deltam15_bool:
            save_viewing_angle_data_for_plotting(band_name, modelname, args)

    # Saving all this viewing angle info for each model to a file so that it is available to plot if required again
    # as it takes relatively long to run this for all viewing angles
    if calculate_peak_time_mag_deltam15_bool:
        write_viewing_angle_data(band_name, modelname, modelnames, args)

    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, band_name, args)
        return

    elif args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        make_viewing_angle_peakmag_delta_m15_scatter_plot(modelnames, band_name, args)
        return

    ax = set_axis_properties(ax, args)
    fig, ax = set_lightcurve_plot_labels(fig, ax, filternames_conversion_dict, args, band_name=band_name)
    ax = set_lightcurveplot_legend(ax, args)

    if args.filter and len(band_lightcurve_data) == 1:
        args.outputfile = os.path.join(outputfolder, f'plot{band_name}lightcurves.pdf')
    if args.show:
        plt.show()
    plt.savefig(args.outputfile, format='pdf')
    print(f'Saved figure: {args.outputfile}')

## Incase this code is needed again...

# if 'redshifttoz' in args and args.redshifttoz[modelnumber] != 0:
#     # print('time before', time)
#     # print('z', args.redshifttoz[modelnumber])
#     time = np.array(time) * (1 + args.redshifttoz[modelnumber])
#     print(f'Correcting for time dilation at redshift {args.redshifttoz[modelnumber]}')
#     # print('time after', time)
#     linestyle = '--'
#     color = 'darkmagenta'
#     linelabel=args.label[1]
# else:
#     linestyle = '-'
#     color='k'
# plt.plot(time, magnitude, label=linelabel, linewidth=3)


    # if (args.magnitude or args.plotviewingangles_lightcurves) and not (
    #         args.calculate_peakmag_risetime_delta_m15 or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
    #         or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file or args.test_viewing_angle_fit
    #         or args.make_viewing_angle_peakmag_risetime_scatter_plot or
    #         args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
    #     if args.reflightcurves:
    #         colours = args.refspeccolors
    #         markers = args.refspecmarkers
    #         for i, reflightcurve in enumerate(args.reflightcurves):
    #             plot_lightcurve_from_data(filters_dict.keys(), reflightcurve, colours[i], markers[i],
    #                                       filternames_conversion_dict)


def save_viewing_angle_data_for_plotting(band_name, modelname, args):
    if args.save_viewing_angle_peakmag_risetime_delta_m15_to_file:
        np.savetxt(band_name + "band_" + f'{modelname}' + "_viewing_angle_data.txt",
                   np.c_[args.band_peakmag_polyfit, args.band_risetime_polyfit, args.band_deltam15_polyfit],
                   delimiter=' ', header='peak_mag_polyfit risetime_polyfit deltam15_polyfit', comments='')

    elif (args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
          or args.make_viewing_angle_peakmag_risetime_scatter_plot
          or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):

        args.band_risetime_angle_averaged_polyfit.append(args.band_risetime_polyfit)
        args.band_peakmag_angle_averaged_polyfit.append(args.band_peakmag_polyfit)
        args.band_delta_m15_angle_averaged_polyfit.append(args.band_deltam15_polyfit)

    args.band_risetime_polyfit = []
    args.band_peakmag_polyfit = []
    args.band_deltam15_polyfit = []

    # if args.magnitude and not (
    #         args.calculate_peakmag_risetime_delta_m15 or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
    #         or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file or args.test_viewing_angle_fit
    #         or args.make_viewing_angle_peakmag_risetime_scatter_plot or
    #         args.make_viewing_angle_peakmag_delta_m15_scatter_plot or args.plotviewingangle):
    #     plt.plot(time, magnitude, label=modelname, color=colours[modelnumber], linewidth=3)


def write_viewing_angle_data(band_name, modelname, modelnames, args):
    if (args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
            or args.make_viewing_angle_peakmag_risetime_scatter_plot
            or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
        np.savetxt(band_name + "band_" + f'{modelname}' + "_angle_averaged_all_models_data.txt",
                   np.c_[modelnames, args.band_risetime_angle_averaged_polyfit, args.band_peakmag_angle_averaged_polyfit,
                         args.band_delta_m15_angle_averaged_polyfit],
                   delimiter=' ', fmt='%s',
                   header="object " + str(band_name) + "_band_risetime " + str(band_name) + "_band_peakmag " + str(
                       band_name) + "_band_deltam15 ", comments='')


def calculate_peak_time_mag_deltam15(time, magnitude, modelname, angle, key, filternames_conversion_dict, args):
    """Calculating band peak time, peak magnitude and delta m15"""
    zfit = np.polyfit(x=time, y=magnitude, deg=10)
    xfit = np.linspace(args.timemin + 1, args.timemax - 1, num=1000)

    # Taking line_min and line_max from the limits set for the lightcurve being plotted
    fxfit = []
    for j in range(len(xfit)):
        fxfit.append(zfit[0] * (xfit[j] ** 10) + zfit[1] * (xfit[j] ** 9) + zfit[2] * (xfit[j] ** 8) +
                     zfit[3] * (xfit[j] ** 7) + zfit[4] * (xfit[j] ** 6) + zfit[5] * (xfit[j] ** 5) +
                     zfit[6] * (xfit[j] ** 4) + zfit[7] * (xfit[j] ** 3) + zfit[8] * (xfit[j] ** 2) +
                     zfit[9] * (xfit[j]) + zfit[10])
        # polynomial with 10 degrees of freedom used here but change as required if it improves the fit

    def match_closest_time_polyfit(reftime_polyfit):
        return str("{}".format(min([float(x) for x in xfit], key=lambda x: abs(x - reftime_polyfit))))

    index_min = np.argmin(fxfit)
    tmax_polyfit = xfit[index_min]
    time_after15days_polyfit = match_closest_time_polyfit(tmax_polyfit + 15)
    for ii, xfits in enumerate(xfit):
        if float(xfits) == float(time_after15days_polyfit):
            index_after_15_days = ii

    mag_after15days_polyfit = fxfit[index_after_15_days]
    print(f'{key}_max polyfit = {min(fxfit)} at time = {tmax_polyfit}')
    print(f'deltam15 polyfit = {min(fxfit) - mag_after15days_polyfit}')

    args.band_risetime_polyfit.append(tmax_polyfit)
    args.band_peakmag_polyfit.append(min(fxfit))
    args.band_deltam15_polyfit.append((min(fxfit) - mag_after15days_polyfit) * -1)

    # Plotting the lightcurves for all viewing angles specified in the command line along with the
    # polynomial fit and peak mag, risetime to peak and delta m15 marked on the plots to check the
    # fit is working correctly
    if args.test_viewing_angle_fit:
        plt.plot(time, magnitude)
        plt.plot(xfit, fxfit)

        if key in filternames_conversion_dict:
            plt.ylabel(f'{filternames_conversion_dict[key]} Magnitude')
        else:
            plt.ylabel(f'{key} Magnitude')

        plt.xlabel('Time Since Explosion [d]')
        plt.gca().invert_yaxis()
        plt.xlim(0, 40)
        plt.minorticks_on()
        plt.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=2, labelsize=12)
        plt.tick_params(axis='both', which='major', top=True, right=True, length=8, width=2, labelsize=12)
        plt.axhline(y=min(fxfit), color="black", linestyle="--")
        plt.axhline(y=mag_after15days_polyfit, color="black", linestyle="--")
        plt.axvline(x=tmax_polyfit, color="black", linestyle="--")
        plt.axvline(x=float(time_after15days_polyfit), color="black", linestyle="--")
        print("time after 15 days polyfit = ", time_after15days_polyfit)
        plt.tight_layout()
        plt.savefig(f'{key}' + "_band_" + f'{modelname}' + "_viewing_angle" + str(angle) + ".png")
        plt.close()


def make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, key, args):
    for ii, modelname in enumerate(modelnames):
        viewing_angle_plot_data = pd.read_csv(key + "band_" + f'{modelname}' + "_viewing_angle_data.txt",
                                              delimiter=" ")
        band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].values
        band_risetime_viewing_angles = viewing_angle_plot_data["risetime_polyfit"].values

        a0 = plt.scatter(band_risetime_viewing_angles, band_peak_mag_viewing_angles, marker='x', color=define_colours_list2[ii])
        p0 = plt.scatter(args.band_risetime_angle_averaged_polyfit[ii], args.band_peakmag_angle_averaged_polyfit[ii],
                         marker='o', color=define_colours_list[ii], s=40)
        args.plotvalues.append((a0, p0))
        plt.errorbar(args.band_risetime_angle_averaged_polyfit[ii], args.band_peakmag_angle_averaged_polyfit[ii],
                     xerr=np.std(band_risetime_viewing_angles),
                     yerr=np.std(band_peak_mag_viewing_angles), ecolor=define_colours_list[ii], capsize=2)

    plt.legend(args.plotvalues, modelnames, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='upper right', fontsize=8, ncol=2, columnspacing=1)
    plt.xlabel('Rise Time in Days', fontsize=14)
    plt.ylabel('Peak ' + key + ' Band Magnitude', fontsize=14)
    # plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', top=False, right=False, length=5, width=2, labelsize=12)
    plt.tick_params(axis='both', which='major', top=False, right=False, length=8, width=2, labelsize=12)
    plt.tight_layout()
    plt.savefig(key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_risetime_scatter_plot.pdf", format="pdf")
    print("saving " + key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_risetime_scatter_plot.pdf")
    plt.close()


def make_viewing_angle_peakmag_delta_m15_scatter_plot(modelnames, key, args):
    for ii, modelname in enumerate(modelnames):
        viewing_angle_plot_data = pd.read_csv(key + "band_" + f'{modelname}' + "_viewing_angle_data.txt",
                                              delimiter=" ")

        band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].values
        band_delta_m15_viewing_angles = viewing_angle_plot_data["deltam15_polyfit"].values

        a0 = plt.scatter(band_delta_m15_viewing_angles, band_peak_mag_viewing_angles, marker='x',
                         color=define_colours_list2[ii])
        p0 = plt.scatter(args.band_delta_m15_angle_averaged_polyfit[ii], args.band_peakmag_angle_averaged_polyfit[ii],
                         marker='o', color=define_colours_list[ii], s=40)
        args.plotvalues.append((a0, p0))
        plt.errorbar(args.band_delta_m15_angle_averaged_polyfit[ii], args.band_peakmag_angle_averaged_polyfit[ii],
                     xerr=np.std(band_delta_m15_viewing_angles),
                     yerr=np.std(band_peak_mag_viewing_angles), ecolor=define_colours_list[ii], capsize=2)

    # a0, label = at.lightcurve.get_sn_sample_bol()
    # a0, label = at.lightcurve.get_phillips_relation_data()
    # args.plotvalues.append((a0, a0))

    plt.legend(args.plotvalues, modelnames, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
               loc='upper right', fontsize=8, ncol=2, columnspacing=1)
    plt.xlabel(r'Decline Rate ($\Delta$m$_{15}$)', fontsize=14)
    plt.ylabel('Peak ' + key + ' Band Magnitude', fontsize=14)
    # plt.gca().invert_yaxis()
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', top=False, right=False, length=5, width=2, labelsize=12)
    plt.tick_params(axis='both', which='major', top=False, right=False, length=8, width=2, labelsize=12)
    plt.tight_layout()
    plt.savefig(key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_delta_m15_scatter_plot.pdf", format="pdf")
    print("saving " + key + "_band_" + f'{modelname}' + "_viewing_angle_peakmag_delta_m15_scatter_plot.pdf")
    plt.close()


def colour_evolution_plot(modelpaths, filternames_conversion_dict, outputfolder, args):
    args.labelfontsize = 24  #todo: make command line arg
    angle_counter = 0

    fig, ax = create_axes(args)
    set_axis_limit_args(args)

    for modelnumber, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f'Reading spectra: {modelname}')

        angles, viewing_angles, angle_definition = get_angle_stuff(modelpath, args)

        for index, angle in enumerate(angles):

            for plotnumber, filters in enumerate(args.colour_evolution):
                filter_names = filters.split('-')
                args.filter = filter_names
                band_lightcurve_data = get_band_lightcurve_data(modelpath, args, angle=angle, modelnumber=modelnumber)

                plot_times, colour_delta_mag = get_colour_delta_mag(band_lightcurve_data, filter_names)

                linelabel = get_linelabel(modelpath, modelname, modelnumber, angle, angle_definition, args)

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    colour_delta_mag = filterfunc(colour_delta_mag)

                if args.color and args.plotviewingangle:
                    print("WARNING: -color argument will not work with viewing angles for colour evolution plots,"
                          "colours are taken from color_list array instead")
                    color = color_list[angle_counter]
                    angle_counter += 1
                elif args.plotviewingangle and not args.color:
                    color = color_list[angle_counter]
                    angle_counter += 1
                elif args.color:
                    color = args.color[modelnumber]
                if args.linestyle:
                    linestyle = args.linestyle[modelnumber]

                if args.reflightcurves and modelnumber == 0:
                    if len(angles) > 1 and index > 0:
                        print('already plotted reflightcurve')
                    else:
                        for i, reflightcurve in enumerate(args.reflightcurves):
                            plot_color_evolution_from_data(
                                filter_names, reflightcurve, args.refspeccolors[i], args.refspecmarkers[i],
                                filternames_conversion_dict, ax, plotnumber, args)

                if args.subplots:
                    ax[plotnumber].plot(plot_times, colour_delta_mag, label=linelabel, linewidth=4, linestyle=linestyle,
                                        color=color)
                else:
                    ax.plot(plot_times, colour_delta_mag, label=linelabel, linewidth=3, linestyle=linestyle,
                            color=color)

                if args.subplots:
                    ax[plotnumber].text(10, args.ymax - 0.5, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')
                else:
                    ax.text(60, args.ymax * 0.8, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')
        # UNCOMMENT TO ESTIMATE COLOUR AT TIME B MAX
        # def match_closest_time(reftime):
        #     return ("{}".format(min([float(x) for x in plot_times], key=lambda x: abs(x - reftime))))
        #
        # tmax_B = 17.0  # CHANGE TO TIME OF B MAX
        # tmax_B = float(match_closest_time(tmax_B))
        # print(f'{filter_names[0]} - {filter_names[1]} at t_Bmax ({tmax_B}) = '
        #       f'{diff[plot_times.index(tmax_B)]}')

    fig, ax = set_lightcurve_plot_labels(fig, ax, filternames_conversion_dict, args)
    ax = set_axis_properties(ax, args)
    ax = set_lightcurveplot_legend(ax, args)

    args.outputfile = os.path.join(outputfolder, f'plotcolorevolution{filter_names[0]}-{filter_names[1]}.pdf')
    for i in range(2):
        if filter_names[i] in filternames_conversion_dict:
            filter_names[i] = filternames_conversion_dict[filter_names[i]]
    # plt.text(10, args.ymax - 0.5, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')

    if args.show:
        plt.show()
    plt.savefig(args.outputfile, format='pdf')

## Just incase it's needed...

# if 'redshifttoz' in args and args.redshifttoz[modelnumber] != 0:
#     plot_times = np.array(plot_times) * (1 + args.redshifttoz[modelnumber])
#     print(f'Correcting for time dilation at redshift {args.redshifttoz[modelnumber]}')
#     linestyle = '--'
#     color='darkmagenta'
#     linelabel = args.label[1]
# else:
#     linestyle = '-'
#     color='k'
#     color='k'


def plot_lightcurve_from_data(
        filter_names, lightcurvefilename, color, marker, filternames_conversion_dict, ax, plotnumber):

    lightcurve_data, metadata = read_reflightcurve_band_data(lightcurvefilename)
    linename = metadata['label'] if plotnumber == 0 else None
    filterdir = os.path.join(at.PYDIR, 'data/filters/')

    filter_data = {}
    for plotnumber, filter_name in enumerate(filter_names):
        if filter_name == 'bol':
            continue
        f = open(filterdir / Path(f'{filter_name}.txt'))
        lines = f.readlines()
        lambda0 = float(lines[2])

        if filter_name == 'bol':
            continue
        elif filter_name in filternames_conversion_dict:
            filter_name = filternames_conversion_dict[filter_name]
        filter_data[filter_name] = lightcurve_data.loc[lightcurve_data['band'] == filter_name]
        # plt.plot(limits_x, limits_y, 'v', label=None, color=color)
        # else:

        if 'a_v' in metadata or 'e_bminusv' in metadata:
            print('Correcting for reddening')

            clightinangstroms = 3e+18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.array([lambda0] * len(filter_data[filter_name]['magnitude']), dtype=float)

            filter_data[filter_name]['flux'] = clightinangstroms / (lambda0 ** 2) * 10 ** -(
                (filter_data[filter_name]['magnitude'] + 48.6) / 2.5)  # gs

            filter_data[filter_name]['dered'] = apply(
                ccm89(filters[:], a_v=-metadata['a_v'], r_v=metadata['r_v']), filter_data[filter_name]['flux'])

            filter_data[filter_name]['magnitude'] = 2.5 * np.log10(
                clightinangstroms / (filter_data[filter_name]['dered'] * lambda0 ** 2)) - 48.6
        else:
            print("WARNING: did not correct for reddening")
        if len(filter_names) > 1:
            ax[plotnumber].plot(filter_data[filter_name]['time'], filter_data[filter_name]['magnitude'], marker,
                                label=linename, color=color)
        else:
            ax.plot(filter_data[filter_name]['time'], filter_data[filter_name]['magnitude'], marker,
                    label=linename, color=color, linewidth=4)

        # if linename == 'SN 2018byg':
        #     x_values = []
        #     y_values = []
        #     limits_x = []
        #     limits_y = []
        #     for index, row in filter_data[filter_name].iterrows():
        #         if row['date'] == 58252:
        #             plt.plot(row['time'], row['magnitude'], '*', label=linename, color=color)
        #         elif row['e_magnitude'] != -1:
        #             x_values.append(row['time'])
        #             y_values.append(row['magnitude'])
        #         else:
        #             limits_x.append(row['time'])
        #             limits_y.append(row['magnitude'])
        #     print(x_values, y_values)
        #     plt.plot(x_values, y_values, 'o', label=linename, color=color)
        #     plt.plot(limits_x, limits_y, 's', label=linename, color=color)
    return linename


def plot_color_evolution_from_data(filter_names, lightcurvefilename, color, marker,
                                   filternames_conversion_dict, ax, plotnumber):
    lightcurve_from_data, metadata = read_reflightcurve_band_data(lightcurvefilename)
    filterdir = os.path.join(at.PYDIR, 'data/filters/')

    filter_data = []
    for i, filter_name in enumerate(filter_names):
        f = open(filterdir / Path(f'{filter_name}.txt'))
        lines = f.readlines()
        lambda0 = float(lines[2])

        if filter_name in filternames_conversion_dict:
            filter_name = filternames_conversion_dict[filter_name]
        filter_data.append(lightcurve_from_data.loc[lightcurve_from_data['band'] == filter_name])

        if 'a_v' in metadata or 'e_bminusv' in metadata:
            print('Correcting for reddening')
            if 'r_v' not in metadata:
                metadata['r_v'] = metadata['a_v'] / metadata['e_bminusv']
            elif 'a_v' not in metadata:
                metadata['a_v'] = metadata['e_bminusv'] * metadata['r_v']

            clightinangstroms = 3e+18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.array([lambda0] * filter_data[i].shape[0], dtype=float)

            filter_data[i]['flux'] = clightinangstroms / (lambda0 ** 2) * 10 ** -(
                (filter_data[i]['magnitude'] + 48.6) / 2.5)

            filter_data[i]['dered'] = apply(ccm89(filters[:], a_v=-metadata['a_v'], r_v=metadata['r_v']),
                                            filter_data[i]['flux'])

            filter_data[i]['magnitude'] = 2.5 * np.log10(
                clightinangstroms / (filter_data[i]['dered'] * lambda0 ** 2)) - 48.6

    # for i in range(2):
    #     # if metadata['label'] == 'SN 2018byg':
    #     #     filter_data[i] = filter_data[i][filter_data[i].e_magnitude != -99.00]
    #     if metadata['label'] in ['SN 2016jhr', 'SN 2018byg']:
    #         filter_data[i]['time'] = filter_data[i]['time'].apply(lambda x: round(float(x)))  # round to nearest day

    merge_dataframes = filter_data[0].merge(filter_data[1], how='inner', on=['time'])
    if args.subplots:
        ax[plotnumber].plot(merge_dataframes['time'], merge_dataframes['magnitude_x'] - merge_dataframes['magnitude_y'], marker,
                            label=metadata['label'], color=color, linewidth=4)
    else:
        ax.plot(merge_dataframes['time'], merge_dataframes['magnitude_x'] - merge_dataframes['magnitude_y'], marker,
                label=metadata['label'], color=color)


def addargs(parser):
    parser.add_argument('modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Path(s) to ARTIS folders with light_curve.out or packets files'
                        ' (may include wildcards such as * and **)')

    parser.add_argument('-label', default=[], nargs='*',
                        help='List of series label overrides')

    parser.add_argument('--nolegend', action='store_true',
                        help='Suppress the legend from the plot')

    parser.add_argument('-color', default=[f'C{i}' for i in range(10)], nargs='*',
                        help='List of line colors')

    parser.add_argument('-linestyle', default=[], nargs='*',
                        help='List of line styles')

    parser.add_argument('-linewidth', default=[], nargs='*',
                        help='List of line widths')

    parser.add_argument('-dashes', default=[], nargs='*',
                        help='Dashes property of lines')

    parser.add_argument('--frompackets', action='store_true',
                        help='Read packets files instead of light_curve.out')

    parser.add_argument('-maxpacketfiles', type=int, default=None,
                        help='Limit the number of packet files read')

    parser.add_argument('--gamma', action='store_true',
                        help='Make light curve from gamma rays instead of R-packets')

    parser.add_argument('-packet_type', default='TYPE_ESCAPE',
                        help='Type of escaping packets')

    parser.add_argument('-escape_type', default='TYPE_RPKT',
                        help='Type of escaping packets')

    parser.add_argument('-o', action='store', dest='outputfile', type=Path,
                        help='Filename for PDF file')

    parser.add_argument('--plotcmf', action='store_true',
                        help='Plot comoving frame light curve')

    parser.add_argument('--magnitude', action='store_true',
                        help='Plot light curves in magnitudes')

    parser.add_argument('--ergs', action='store_true',
                        help='Plot light curves in erg/s')

    parser.add_argument('-filter', '-band', dest='filter', type=str, nargs='+',
                        help='Choose filter eg. bol U B V R I. Default B. '
                        'WARNING: filter names are not case sensitive eg. sloan-r is not r, it is rs')

    parser.add_argument('-colour_evolution', nargs='*',
                        help='Plot of colour evolution. Give two filters eg. B-V')

    parser.add_argument('--print_data', action='store_true',
                        help='Print plotted data')

    parser.add_argument('-plot_hesma_model', action='store', type=Path, default=False,
                        help='Plot hesma model on top of lightcurve plot. '
                        'Enter model name saved in data/hesma directory')

    parser.add_argument('-plotvspecpol', type=int, nargs='+',
                        help='Plot vspecpol. Expects int for spec number in vspecpol files')

    parser.add_argument('-plotviewingangle', type=int, nargs='+',
                        help='Plot viewing angles. Expects int for angle number in specpol_res.out'
                        'use args = -1 to select all the viewing angles')

    parser.add_argument('-ymax', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-ymin', type=float, default=None,
                        help='Plot range: y-axis')

    parser.add_argument('-xmax', type=float, default=None,
                        help='Plot range: x-axis')

    parser.add_argument('-xmin', type=float, default=None,
                        help='Plot range: x-axis')

    parser.add_argument('-timemax', type=float, default=None,
                        help='Time max to plot')

    parser.add_argument('-timemin', type=float, default=None,
                        help='Time min to plot')

    parser.add_argument('--logscaley', action='store_true',
                        help='Use log scale for vertial axis')

    parser.add_argument('-reflightcurves', type=str, nargs='+', dest='reflightcurves',
                        help='Also plot reference lightcurves from these files')

    parser.add_argument('-refspeccolors', default=['0.0', '0.3', '0.5'], nargs='*',
                        help='Set a list of color for reference spectra')

    parser.add_argument('-refspecmarkers', default=['o', 's', 'h'], nargs='*',
                        help='Set a list of markers for reference spectra')

    parser.add_argument('-filtersavgol', nargs=2,
                        help='Savitzky–Golay filter. Specify the window_length and poly_order.'
                             'e.g. -filtersavgol 5 3')

    parser.add_argument('-redshifttoz', type=float, nargs='+',
                        help='Redshift to z = x. Expects array length of number modelpaths.'
                        'If not to be redshifted then = 0.')

    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plot before saving')

    parser.add_argument('--calculate_peakmag_risetime_delta_m15', action='store_true',
                        help='Calculate band risetime, peak mag and delta m15 values for '
                        'the models specified using a polynomial fitting method and '
                        'print to screen')

    parser.add_argument('--save_angle_averaged_peakmag_risetime_delta_m15_to_file', action='store_true',
                        help='Save the band risetime, peak mag and delta m15 values for '
                        'the angle averaged model lightcurves to file')

    parser.add_argument('--save_viewing_angle_peakmag_risetime_delta_m15_to_file', action='store_true',
                        help='Save the band risetime, peak mag and delta m15 values for '
                        'all viewing angles specified for plotting at a later time '
                        'as these values take a long time to calculate for all '
                        'viewing angles. Need to run this command first alongside '
                        '--plotviewingangles in order to save the data for the '
                        'viewing angles you want to use before making the scatter'
                        'plots')

    parser.add_argument('--test_viewing_angle_fit', action='store_true',
                        help='Plots the lightcurves for each  viewing angle along with'
                        'the polynomial fit for each viewing angle specified'
                        'to check the fit is working properly: use alongside'
                        '--plotviewingangle ')

    parser.add_argument('--make_viewing_angle_peakmag_risetime_scatter_plot', action='store_true',
                        help='Makes scatter plot of band peak mag with risetime with the '
                        'angle averaged values being the solid dot and the errors bars'
                        'representing the standard deviation of the viewing angle'
                        'distribution')

    parser.add_argument('--make_viewing_angle_peakmag_delta_m15_scatter_plot', action='store_true',
                        help='Makes scatter plot of band peak with delta m15 with the angle'
                        'averaged values being the solid dot and the error bars representing '
                        'the standard deviation of the viewing angle distribution')

    parser.add_argument('--plotviewingangles_lightcurves', action='store_true',
                        help='Make lightcurve plots for the viewing angles and models specified')

    parser.add_argument('--average_every_tenth_viewing_angle', action='store_true',
                        help='average every tenth viewing angle to reduce noise')

    parser.add_argument('-calculate_costheta_phi_from_viewing_angle_numbers', type=int, nargs='+',
                        help='calculate costheta and phi for each viewing angle given the number of the viewing angle'
                             'Expects ints for angle number supplied from the argument of plot viewing angle'
                             'use args = -1 to select all viewing angles'
                             'Note: this method will only work if the number of angle bins (MABINS) = 100'
                             'if this is not the case an error will be printed')

    parser.add_argument('--colorbarcostheta', action='store_true',
                        help='Colour viewing angles by cos theta and show color bar')

    parser.add_argument('--colorbarphi', action='store_true',
                        help='Colour viewing angles by phi and show color bar')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot ARTIS light curve.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath and not args.colour_evolution:
        args.modelpath = ['.']
    elif not args.modelpath and (args.filter or args.colour_evolution):
        args.modelpath = ['.']
    elif not isinstance(args.modelpath, Iterable):
        args.modelpath = [args.modelpath]

    args.modelpath = at.flatten_list(args.modelpath)
    # flatten the list
    modelpaths = []
    for elem in args.modelpath:
        if isinstance(elem, list):
            modelpaths.extend(elem)
        else:
            modelpaths.append(elem)

    args.color, args.label, args.linestyle, args.dashes, args.linewidth = at.trim_or_pad(
        len(args.modelpath), args.color, args.label, args.linestyle, args.dashes, args.linewidth)

    if args.gamma:
        args.escape_type = 'TYPE_GAMMA'

    if args.filter:
        defaultoutputfile = 'plotlightcurves.pdf'
    elif args.colour_evolution:
        defaultoutputfile = 'plot_colour_evolution.pdf'
    elif args.escape_type == 'TYPE_GAMMA':
        defaultoutputfile = 'plotlightcurve_gamma.pdf'
    elif args.escape_type == 'TYPE_RPKT':
        defaultoutputfile = 'plotlightcurve.pdf'
    else:
        defaultoutputfile = f'plotlightcurve_{args.escape_type}.pdf'

    if not args.outputfile:
        outputfolder = Path()
        args.outputfile = defaultoutputfile
    elif os.path.isdir(args.outputfile):
        outputfolder = Path(args.outputfile)
        args.outputfile = os.path.join(outputfolder, defaultoutputfile)

    filternames_conversion_dict = {'rs': 'r', 'gs': 'g', 'is': 'i'}
    if args.filter:
        make_band_lightcurves_plot(modelpaths, filternames_conversion_dict, outputfolder, args)

    elif args.colour_evolution:
        colour_evolution_plot(modelpaths, filternames_conversion_dict, outputfolder, args)
        print(f'Saved figure: {args.outputfile}')
    else:
        make_lightcurve_plot_from_lightcurve_out_files(args.modelpath, args.outputfile, args.frompackets,
                                                       args.escape_type, maxpacketfiles=args.maxpacketfiles, args=args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
