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
import artistools.plottools
import matplotlib.pyplot as plt
import matplotlib
from extinction import apply, ccm89
from astropy import constants as const

color_list = list(plt.get_cmap('tab20')(np.linspace(0, 1.0, 20)))


def make_lightcurve_plot_from_lightcurve_out_files(modelpaths, filenameout, frompackets=False,
                                                   escape_type=False, maxpacketfiles=None, args=None):
    """Use light_curve.out or light_curve_res.out files to plot light curve"""

    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharex=True,
        figsize=(args.figscale * at.config['figwidth'] * 1.6, args.figscale * at.config['figwidth']),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if args.plotthermalisation:
        figtherm, axistherm = plt.subplots(
            nrows=1, ncols=1, sharex=True,
            figsize=(args.figscale * at.config['figwidth'] * 1.4, args.figscale * at.config['figwidth']),
            tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0})

    if not frompackets and escape_type not in ['TYPE_RPKT', 'TYPE_GAMMA']:
        print(f'Escape_type of {escape_type} not one of TYPE_RPKT or TYPE_GAMMA, so frompackets must be enabled')
        assert False
    elif not frompackets and args.packet_type != 'TYPE_ESCAPE' and args.packet_type is not None:
        print('Looking for non-escaped packets, so frompackets must be enabled')
        assert False

    # take any assigned colours our of the cycle
    colors = [
        color for i, color in enumerate(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        if f'C{i}' not in args.color]
    axis.set_prop_cycle(color=colors)
    reflightcurveindex = 0

    for seriesindex, modelpath in enumerate(modelpaths):
        if not Path(modelpath).is_dir() and '.' in str(modelpath):
            bolreflightcurve = Path(modelpath)

            dflightcurve, metadata = at.lightcurve.read_bol_reflightcurve_data(bolreflightcurve)
            lightcurvelabel = metadata.get('label', bolreflightcurve)
            color = ['0.0', '0.5', '0.7'][reflightcurveindex]
            plotkwargs = dict(label=lightcurvelabel, color=color, zorder=0)
            if ('luminosity_errminus_erg/s' in dflightcurve.columns
                    and 'luminosity_errplus_erg/s' in dflightcurve.columns):
                axis.errorbar(
                    dflightcurve['time_days'], dflightcurve['luminosity_erg/s'],
                    yerr=[dflightcurve['luminosity_errminus_erg/s'], dflightcurve['luminosity_errplus_erg/s']],
                    fmt='o', capsize=3, **plotkwargs)
            else:
                axis.scatter(
                    dflightcurve['time_days'], dflightcurve['luminosity_erg/s'], **plotkwargs)
            print(f"====> {lightcurvelabel}")
            reflightcurveindex += 1
            continue

        modelname = at.get_model_name(modelpath)
        print(f"====> {modelname}")

        lcname = 'gamma_light_curve.out' if (escape_type == 'TYPE_GAMMA' and not frompackets) else 'light_curve.out'
        if args.plotviewingangle is not None and lcname == 'light_curve.out':
            lcname = 'light_curve_res.out'
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
            lcdata = at.lightcurve.readfile(lcpath, modelpath, args)

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

        if args.plotdeposition:
            dfmodel, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata(modelpath)
            model_mass_grams = dfmodel.cellmass_grams.sum()
            print(f"  model mass: {model_mass_grams / 1.989e33:.3f} Msun")
            depdata = at.get_deposition(modelpath)

            # color_total = next(axis._get_lines.prop_cycler)['color']

            # axis.plot(depdata['tmid_days'], depdata['eps_erg/s/g'] * model_mass_grams, **dict(
            #     plotkwargs, **{
            #         'label': plotkwargs['label'] + r' $\dot{\epsilon}_{\alpha\beta^\pm\gamma}$',
            #         'linestyle': 'dashed',
            #         'color': color_total,
            #     }))

            # axis.plot(depdata['tmid_days'], depdata['total_dep_Lsun'] * 3.826e33, **dict(
            #     plotkwargs, **{
            #         'label': plotkwargs['label'] + r' $\dot{E}_{dep,\alpha\beta^\pm\gamma}$',
            #         'linestyle': 'dotted',
            #         'color': color_total,
            #     }))
            # if args.plotthermalisation:
            #     # f = depdata['eps_erg/s/g'] / depdata['Qdot_ana_erg/s/g']
            #     f = depdata['total_dep_Lsun'] * 3.826e33 / (depdata['eps_erg/s/g'] * model_mass_grams)
            #     axistherm.plot(depdata['tmid_days'], f, **dict(
            #         plotkwargs, **{
            #             'label': plotkwargs['label'] + r' $\dot{E}_{dep}/\dot{E}_{rad}$',
            #             'linestyle': 'solid',
            #             'color': color_total,
            #         }))

            color_gamma = next(axis._get_lines.prop_cycler)['color']

            # axis.plot(depdata['tmid_days'], depdata['eps_gamma_Lsun'] * 3.826e33, **dict(
            #     plotkwargs, **{
            #         'label': plotkwargs['label'] + r' $\dot{E}_{rad,\gamma}$',
            #         'linestyle': 'dashed',
            #         'color': color_gamma,
            #     }))

            gammadep_lsun = depdata['gammadeppathint_Lsun'] if 'gammadeppathint_Lsun' in depdata else depdata['gammadep_Lsun']

            axis.plot(depdata['tmid_days'], gammadep_lsun * 3.826e33, **dict(
                plotkwargs, **{
                    'label': plotkwargs['label'] + r' $\dot{E}_{dep,\gamma}$',
                    'linestyle': 'dashed',
                    'color': color_gamma,
                }))

            color_beta = next(axis._get_lines.prop_cycler)['color']

            if 'eps_elec_Lsun' in depdata:
                axis.plot(depdata['tmid_days'], depdata['eps_elec_Lsun'] * 3.826e33, **dict(
                    plotkwargs, **{
                        'label': plotkwargs['label'] + r' $\dot{E}_{rad,\beta^-}$',
                        'linestyle': 'dotted',
                        'color': color_beta,
                    }))

            if 'elecdep_Lsun' in depdata:
                axis.plot(depdata['tmid_days'], depdata['elecdep_Lsun'] * 3.826e33, **dict(
                    plotkwargs, **{
                        'label': plotkwargs['label'] + r' $\dot{E}_{dep,\beta^-}$',
                        'linestyle': 'dashed',
                        'color': color_beta,
                    }))

            # color_alpha = next(axis._get_lines.prop_cycler)['color']
            color_alpha = 'C1'

            # if 'eps_alpha_ana_Lsun' in depdata:
            #     axis.plot(depdata['tmid_days'], depdata['eps_alpha_ana_Lsun'] * 3.826e33, **dict(
            #         plotkwargs, **{
            #             'label': plotkwargs['label'] + r' $\dot{E}_{rad,\alpha}$ analytical',
            #             'linestyle': 'solid',
            #             'color': color_alpha,
            #         }))

            # if 'eps_alpha_Lsun' in depdata:
            #     axis.plot(depdata['tmid_days'], depdata['eps_alpha_Lsun'] * 3.826e33, **dict(
            #         plotkwargs, **{
            #             'label': plotkwargs['label'] + r' $\dot{E}_{rad,\alpha}$',
            #             'linestyle': 'dashed',
            #             'color': color_alpha,
            #         }))

            # axis.plot(depdata['tmid_days'], depdata['alphadep_Lsun'] * 3.826e33, **dict(
            #     plotkwargs, **{
            #         'label': plotkwargs['label'] + r' $\dot{E}_{dep,\alpha}$',
            #         'linestyle': 'dotted',
            #         'color': color_alpha,
            #     }))

            if args.plotthermalisation:
                ejecta_ke = dfmodel.eval(
                    '0.5 * (cellmass_grams / 1000.) * (0.5 * 1000. * (velocity_inner + velocity_outer)) ** 2').sum()
                # velocity derived from ejecta kinetric energy to match Barnes et al. (2016) Section 2.1
                ejecta_v = np.sqrt(2 * ejecta_ke / (model_mass_grams * 1e-3))
                v2 = ejecta_v / (.2 * 299792458)
                m5 = model_mass_grams / (5e-3 * 1.989e+33)  # M / (5e-3 Msun)

                # v2 = 1.
                # m5 = 1.

                t_ineff_gamma = .5 * np.sqrt(m5) / v2
                barnes_f_gamma = [
                    1 - math.exp(- (t / t_ineff_gamma) ** -2)
                    for t in depdata['tmid_days'].values]

                axistherm.plot(depdata['tmid_days'], barnes_f_gamma, **dict(
                    plotkwargs, **{
                        'label': 'Barnes+16 f_gamma',
                        'linestyle': 'dashed', 'color': color_gamma}))

                e0_beta_mev = 0.5
                t_ineff_beta = 7.4 * (e0_beta_mev / 0.5) ** -0.5 * m5 ** 0.5 * (v2 ** (-3./2))
                barnes_f_beta = [
                    math.log(1 + 2 * (t / t_ineff_beta) ** 2) / (2 * (t / t_ineff_beta) ** 2)
                    for t in depdata['tmid_days'].values]

                axistherm.plot(depdata['tmid_days'], barnes_f_beta, **dict(
                    plotkwargs, **{
                        'label': 'Barnes+16 f_beta',
                        'linestyle': 'dashed', 'color': color_beta}))

                e0_alpha_mev = 6.
                t_ineff_alpha = 4.3 * 1.8 * (e0_alpha_mev / 6.) ** -0.5 * m5 ** 0.5 * (v2 ** (-3./2))
                barnes_f_alpha = [
                    math.log(1 + 2 * (t / t_ineff_alpha) ** 2) / (2 * (t / t_ineff_alpha) ** 2)
                    for t in depdata['tmid_days'].values]

                axistherm.plot(depdata['tmid_days'], barnes_f_alpha, **dict(
                    plotkwargs, **{
                        'label': 'Barnes+16 f_alpha',
                        'linestyle': 'dashed', 'color': color_alpha}))

                axistherm.plot(depdata['tmid_days'], depdata['gammadeppathint_Lsun'] / depdata['eps_gamma_Lsun'], **dict(
                    plotkwargs, **{
                        'label': 'ARTIS f_gamma',
                        'linestyle': 'solid', 'color': color_gamma}))

                axistherm.plot(depdata['tmid_days'], depdata['elecdep_Lsun'] / depdata['eps_elec_Lsun'], **dict(
                    plotkwargs, **{
                        'label': 'ARTIS f_beta',
                        'linestyle': 'solid', 'color': color_beta, }))

                f_alpha = depdata['alphadep_Lsun'] / depdata['eps_alpha_Lsun']
                kernel_size = 5
                if len(f_alpha) > kernel_size:
                    kernel = np.ones(kernel_size) / kernel_size
                    f_alpha = np.convolve(f_alpha, kernel, mode='same')
                axistherm.plot(depdata['tmid_days'], f_alpha, **dict(
                    plotkwargs, **{
                        'label': 'ARTIS f_alpha',
                        'linestyle': 'solid', 'color': color_alpha}))

        # check if doing viewing angle stuff, and if so define which data to use
        angles, viewing_angles, angle_definition = at.lightcurve.get_angle_stuff(modelpath, args)
        if args.plotviewingangle:
            lcdataframes = lcdata

            if args.colorbarcostheta or args.colorbarphi:
                costheta_viewing_angle_bins, phi_viewing_angle_bins = at.lightcurve.get_viewinganglebin_definitions()
                scaledmap = make_colorbar_viewingangles_colormap()

        print(f'  range of light curve: '
              f'{lcdata.time.min():.2f} to {lcdata.time.max():.2f} days')
        try:
            nts_last, validrange_start_days, validrange_end_days = at.get_escaped_arrivalrange(modelpath)
            if validrange_start_days is not None and validrange_end_days is not None:
                str_valid_range = f'{validrange_start_days:.2f} to {validrange_end_days:.2f} days' 
            else:
                str_valid_range = f'{validrange_start_days} to {validrange_end_days} days'
            print(f'  range of validity (last timestep {nts_last}): {str_valid_range}')
        except FileNotFoundError:
            print('  range of validity: could not determine due to missing files '
                  '(requires deposition.out, input.txt, model.txt)')
            nts_last, validrange_start_days, validrange_end_days = None, float('-inf'), float('inf')

        for angleindex, angle in enumerate(angles):
            if args.plotviewingangle:
                lcdata = lcdataframes[angle]

                if args.colorbarcostheta or args.colorbarphi:
                    plotkwargs['alpha'] = 0.75
                    plotkwargs['label'] = None
                    # Update plotkwargs with viewing angle colour
                    plotkwargs, _ = get_viewinganglecolor_for_colorbar(
                        angle_definition, angle,
                        costheta_viewing_angle_bins, phi_viewing_angle_bins,
                        scaledmap, plotkwargs, args)
                else:
                    plotkwargs['color'] = None
                    plotkwargs['label'] = f'{modelname}\n{angle_definition[angle]}'

            filterfunc = at.get_filterfunc(args)
            if filterfunc is not None:
                lcdata['lum'] = filterfunc(lcdata['lum'])

            if not args.Lsun or args.magnitude:
                # convert luminosity from Lsun to erg/s
                lcdata.eval('lum = lum * 3.826e33', inplace=True)
                lcdata.eval('lum_cmf = lum_cmf * 3.826e33', inplace=True)

            if args.magnitude:
                # convert to bol magnitude
                lcdata['mag'] = 4.74 - (2.5 * np.log10(lcdata['lum'] / const.L_sun.to('erg/s').value))
                axis.plot(lcdata['time'], lcdata['mag'], **plotkwargs)
            else:
                # show the parts of the light curve that are outside the valid arrival range partially transparent
                if validrange_start_days is None or validrange_end_days is None:
                    # entire range is invalid
                    lcdata_before_valid = lcdata
                    lcdata_after_valid = pd.DataFrame(data=None, columns=lcdata.columns)
                    lcdata_valid = pd.DataFrame(data=None, columns=lcdata.columns)
                else:
                    lcdata_valid = lcdata.query(
                        'time >= @validrange_start_days and time <= @validrange_end_days', inplace=False)
                    lcdata_before_valid = lcdata.query('time <= @lcdata_valid.time.min()')
                    lcdata_after_valid = lcdata.query('time >= @lcdata_valid.time.max()')

                axis.plot(lcdata_valid['time'], lcdata_valid['lum'], **plotkwargs)
                if args.plotinvalidpart:
                    plotkwargs_invalidrange = plotkwargs.copy()
                    plotkwargs_invalidrange.update({'label': None, 'alpha': 0.5})
                    axis.plot(lcdata_before_valid['time'], lcdata_before_valid['lum'], **plotkwargs_invalidrange)
                    axis.plot(lcdata_after_valid['time'], lcdata_after_valid['lum'], **plotkwargs_invalidrange)

                if args.print_data:
                    print(lcdata[['time', 'lum', 'lum_cmf']].to_string(index=False))
                if args.plotcmf:
                    plotkwargs['linewidth'] = 1
                    plotkwargs['label'] += ' (cmf)'
                    plotkwargs['linestyle'] = 'dashed'
                    # plotkwargs['color'] = 'tab:orange'
                    axis.plot(lcdata.time, lcdata['lum_cmf'], **plotkwargs)

    if args.reflightcurves:
        for bolreflightcurve in args.reflightcurves:
            if args.Lsun:
                print("Check units - trying to plot ref light curve in erg/s")
                quit()
            bollightcurve_data, metadata = at.lightcurve.read_bol_reflightcurve_data(bolreflightcurve)
            axis.scatter(bollightcurve_data['time_days'], bollightcurve_data['luminosity_erg/s'],
                         label=metadata.get('label', bolreflightcurve), color='k')

    if args.magnitude:
        plt.gca().invert_yaxis()

    if args.xmin is not None:
        axis.set_xlim(left=args.xmin)
    if args.xmax is not None:
        axis.set_xlim(right=args.xmax)
    if args.ymin is not None:
        axis.set_ylim(bottom=args.ymin)
    if args.ymax is not None:
        axis.set_ylim(top=args.ymax)
    # axis.set_ylim(bottom=-0.1, top=1.3)

    if not args.nolegend:
        axis.legend(loc='best', handlelength=2, frameon=args.legendframeon, numpoints=1, prop={'size': 9})
        if args.plotthermalisation:
            axistherm.legend(loc='best', handlelength=2, frameon=args.legendframeon, numpoints=1, prop={'size': 9})

    axis.set_xlabel(r'Time [days]')

    if args.magnitude:
        axis.set_ylabel('Absolute Bolometric Magnitude')
    else:
        if not args.Lsun:
            str_units = ' [erg/s]'
        else:
            str_units = r'$/ \mathrm{L}_\odot$'
        if args.plotdeposition:
            yvarname = ''
        elif escape_type == 'TYPE_GAMMA':
            yvarname = r'$\mathrm{L}_\gamma$'
        elif escape_type == 'TYPE_RPKT':
            yvarname = r'$\mathrm{L}_{\mathrm{UVOIR}}$'
        else:
            yvarname = r'$\mathrm{L}_{\mathrm{' + escape_type.replace("_", r"\_") + r'}}$'
        axis.set_ylabel(yvarname + str_units)

    if args.title:
        axis.set_title(modelname)

    if args.colorbarcostheta or args.colorbarphi:
        make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args)

    if args.logscalex:
        axis.set_xscale('log')

    if args.logscaley:
        axis.set_yscale('log')

    if args.show:
        plt.show()

    fig.savefig(str(filenameout), format='pdf')
    print(f'Saved {filenameout}')

    if args.plotthermalisation:
        # axistherm.set_xscale('log')
        axistherm.set_ylabel('Thermalisation ratio')
        axistherm.set_xlabel(r'Time [days]')
        # axistherm.set_xlim(left=0., args.xmax)
        if args.xmin is not None:
            axistherm.set_xlim(left=args.xmin)
        if args.xmax is not None:
            axistherm.set_xlim(right=args.xmax)
        axistherm.set_ylim(bottom=0.)
        # axistherm.set_ylim(top=1.05)

        filenameout2 = 'plotthermalisation.pdf'
        figtherm.savefig(str(filenameout2), format='pdf')
        print(f'Saved {filenameout2}')

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
        args.figwidth = at.config['figwidth'] * 1.6 * cols
    if 'figheight' not in args:
        args.figheight = at.config['figwidth'] * 1.1 * rows*1.5

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


def get_linelabel(modelpath, modelname, modelnumber, angle, angle_definition, args):
    if args.plotvspecpol and angle is not None and os.path.isfile(modelpath / 'vpkt.txt'):
        vpkt_config = at.get_vpkt_config(modelpath)
        viewing_angle = round(math.degrees(math.acos(vpkt_config['cos_theta'][angle])))
        linelabel = fr"$\theta$ = {viewing_angle}"  # todo: update to be consistent with res definition
    elif args.plotviewingangle and angle is not None and os.path.isfile(modelpath / 'specpol_res.out'):
        if args.nomodelname:
            linelabel = fr"{angle_definition[angle]}"
        else:
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


def set_lightcurveplot_legend(ax, args):
    if not args.nolegend:
        if args.subplots:
            ax[args.legendsubplotnumber].legend(loc=args.legendposition, frameon=args.legendframeon,
                                                fontsize='x-small', ncol=args.ncolslegend)
        else:
            ax.legend(loc=args.legendposition, frameon=args.legendframeon,
                      fontsize='small', ncol=args.ncolslegend, handlelength=0.7)


def set_lightcurve_plot_labels(fig, ax, filternames_conversion_dict, args, band_name=None):
    ylabel = None
    if args.subplots:
        if args.filter:
            ylabel = 'Absolute Magnitude'
        if args.colour_evolution:
            ylabel = r'$\Delta$m'
        fig.text(0.5, 0.025, 'Time Since Explosion [days]', ha='center', va='center')
        fig.text(0.02, 0.5, ylabel, ha='center', va='center', rotation='vertical')
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


def make_colorbar_viewingangles_colormap():
    norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
    scaledmap = matplotlib.cm.ScalarMappable(cmap='tab10', norm=norm)
    scaledmap.set_array([])
    return scaledmap


def get_viewinganglecolor_for_colorbar(angle_definition, angle, costheta_viewing_angle_bins, phi_viewing_angle_bins,
                                       scaledmap, plotkwargs, args):
    if args.colorbarcostheta:
        colorindex = costheta_viewing_angle_bins.index(angle_definition[angle].split(', ')[0])
        plotkwargs['color'] = scaledmap.to_rgba(colorindex)
    if args.colorbarphi:
        colorindex = phi_viewing_angle_bins.index(angle_definition[angle].split(', ')[1])
        reorderphibins = {5: 9, 6: 8, 7: 7, 8: 6, 9: 5}
        print("Reordering phi bins")
        if colorindex in reorderphibins.keys():
            colorindex = reorderphibins[colorindex]
        plotkwargs['color'] = scaledmap.to_rgba(colorindex)
    return plotkwargs, colorindex


def make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, fig=None, ax=None):
    if args.colorbarcostheta:
        # ticklabels = costheta_viewing_angle_bins
        ticklabels = [' -1', ' -0.8', ' -0.6', ' -0.4', ' -0.2', ' 0', ' 0.2', ' 0.4', ' 0.6', ' 0.8', ' 1']
        ticklocs = np.linspace(0, 9, num=11)
        label = 'cos(\u03B8)'
    if args.colorbarphi:
        print('reordered phi bins')
        phi_viewing_angle_bins_reordered = [
            '0', '\u03c0/5', '2\u03c0/5', '3\u03c0/5', '4\u03c0/5', '\u03c0',
            '6\u03c0/5', '7\u03c0/5', '8\u03c0/5', '9\u03c0/5', '2\u03c0']
        ticklabels = phi_viewing_angle_bins_reordered
        # ticklabels = phi_viewing_angle_bins
        ticklocs = np.linspace(0, 9, num=11)
        label = '\u03D5 bin'

    cbar = plt.colorbar(scaledmap)
    if label:
        cbar.set_label(label, rotation=90)
    cbar.locator = matplotlib.ticker.FixedLocator(ticklocs)
    cbar.formatter = matplotlib.ticker.FixedFormatter(ticklabels)
    cbar.update_ticks()


def make_band_lightcurves_plot(modelpaths, filternames_conversion_dict, outputfolder, args):
    # angle_names = [0, 45, 90, 180]
    # plt.style.use('dark_background')

    args.labelfontsize = 22  # todo: make command line arg
    fig, ax = create_axes(args)
    set_axis_limit_args(args)

    plotkwargs = {}

    if args.colorbarcostheta or args.colorbarphi:
        costheta_viewing_angle_bins, phi_viewing_angle_bins = at.lightcurve.get_viewinganglebin_definitions()
        scaledmap = make_colorbar_viewingangles_colormap()

    for modelnumber, modelpath in enumerate(modelpaths):
        modelpath = Path(modelpath)  # Make sure modelpath is defined as path. May not be necessary

        # check if doing viewing angle stuff, and if so define which data to use
        angles, viewing_angles, angle_definition = at.lightcurve.get_angle_stuff(modelpath, args)

        for index, angle in enumerate(angles):

            modelname = at.get_model_name(modelpath)
            print(f'Reading spectra: {modelname} (angle {angle})')
            band_lightcurve_data = at.lightcurve.generate_band_lightcurve_data(
                modelpath, args, angle, modelnumber=modelnumber)

            if modelnumber == 0 and args.plot_hesma_model:  # Todo: does this work?
                hesma_model = at.lightcurve.read_hesma_lightcurve(args)
                plotkwargs['label'] = str(args.plot_hesma_model).split('_')[:3]

            for plotnumber, band_name in enumerate(band_lightcurve_data):
                time, brightness_in_mag = at.lightcurve.get_band_lightcurve(band_lightcurve_data, band_name, args)

                if args.print_data or args.write_data:
                    txtlinesout = []
                    txtlinesout.append(f'# band: {band_name}')
                    txtlinesout.append(f'# model: {modelname}')
                    txtlinesout.append('# time_days magnitude')
                    for t, m in zip(time, brightness_in_mag):
                        txtlinesout.append(f'{t} {m}')
                    txtout = '\n'.join(txtlinesout)
                    if args.write_data:
                        bandoutfile = Path(f'band_{band_name}.txt')
                        with bandoutfile.open('w') as f:
                            f.write(txtout)
                        print(f'Saved {bandoutfile}')
                    if args.print_data:
                        print(txtout)

                plotkwargs['label'] = get_linelabel(modelpath, modelname, modelnumber, angle, angle_definition, args)
                # plotkwargs['label'] = '\n'.join(wrap(linelabel, 40))  # todo: could be arg? wraps text in label

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    brightness_in_mag = filterfunc(brightness_in_mag)

                # This does the same thing as below -- leaving code in case I'm wrong (CC)
                # if args.plotviewingangle and args.plotviewingangles_lightcurves:
                #     global define_colours_list
                #     plt.plot(time, brightness_in_mag, label=modelname, color=define_colours_list[angle], linewidth=3)

                if modelnumber == 0 and args.plot_hesma_model and band_name in hesma_model.keys():  # todo: see if this works
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

                # if not args.calculate_peak_time_mag_deltam15_bool:

                if args.reflightcurves and modelnumber == 0:
                    if len(angles) > 1 and index > 0:
                        print('already plotted reflightcurve')
                    else:
                        define_colours_list = args.refspeccolors
                        markers = args.refspecmarkers
                        for i, reflightcurve in enumerate(args.reflightcurves):
                            plot_lightcurve_from_data(
                                band_lightcurve_data.keys(), reflightcurve, define_colours_list[i], markers[i],
                                filternames_conversion_dict, ax, plotnumber)

                if args.color:
                    plotkwargs['color'] = args.color[modelnumber]
                else:
                    plotkwargs['color'] = define_colours_list[modelnumber]

                if args.colorbarcostheta or args.colorbarphi:
                    # Update plotkwargs with viewing angle colour
                    plotkwargs['label'] = None
                    plotkwargs, _ = get_viewinganglecolor_for_colorbar(
                        angle_definition, angle,
                        costheta_viewing_angle_bins, phi_viewing_angle_bins,
                        scaledmap, plotkwargs, args)

                if args.linestyle:
                    plotkwargs['linestyle'] = args.linestyle[modelnumber]

                # if not (args.test_viewing_angle_fit or args.calculate_peak_time_mag_deltam15_bool):

                if args.subplots:
                    if len(angles) > 1 or (args.plotviewingangle and os.path.isfile(modelpath / 'specpol_res.out')):
                        ax[plotnumber].plot(time, brightness_in_mag, linewidth=4, **plotkwargs)
                    # I think this was just to have a different line style for viewing angles....
                    else:
                        ax[plotnumber].plot(time, brightness_in_mag, linewidth=4, **plotkwargs)
                        # if key is not 'bol':
                        #     ax[plotnumber].plot(
                        #         cmfgen_mags['time[d]'], cmfgen_mags[key], label='CMFGEN', color='k', linewidth=3)
                else:
                    ax.plot(time, brightness_in_mag, linewidth=3.5, **plotkwargs)  # color=color, linestyle=linestyle)

    import artistools.plottools
    ax = at.plottools.set_axis_properties(ax, args)
    fig, ax = set_lightcurve_plot_labels(fig, ax, filternames_conversion_dict, args, band_name=band_name)
    ax = set_lightcurveplot_legend(ax, args)

    if args.colorbarcostheta or args.colorbarphi:
        make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, fig=fig, ax=ax)

    if args.filter and len(band_lightcurve_data) == 1:
        args.outputfile = os.path.join(outputfolder, f'plot{band_name}lightcurves.pdf')
    if args.show:
        plt.show()
    plt.savefig(args.outputfile, format='pdf')
    print(f'Saved figure: {args.outputfile}')

# In case this code is needed again...

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


def colour_evolution_plot(modelpaths, filternames_conversion_dict, outputfolder, args):
    args.labelfontsize = 24  # todo: make command line arg
    angle_counter = 0

    fig, ax = create_axes(args)
    set_axis_limit_args(args)

    plotkwargs = {}

    for modelnumber, modelpath in enumerate(modelpaths):
        modelpath = Path(modelpath)
        modelname = at.get_model_name(modelpath)
        print(f'Reading spectra: {modelname}')

        angles, viewing_angles, angle_definition = at.lightcurve.get_angle_stuff(modelpath, args)

        for index, angle in enumerate(angles):

            for plotnumber, filters in enumerate(args.colour_evolution):
                filter_names = filters.split('-')
                args.filter = filter_names
                band_lightcurve_data = at.lightcurve.generate_band_lightcurve_data(modelpath, args, angle=angle, modelnumber=modelnumber)

                plot_times, colour_delta_mag = at.lightcurve.get_colour_delta_mag(band_lightcurve_data, filter_names)

                plotkwargs['label'] = get_linelabel(modelpath, modelname, modelnumber, angle, angle_definition, args)

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    colour_delta_mag = filterfunc(colour_delta_mag)

                if args.color and args.plotviewingangle:
                    print("WARNING: -color argument will not work with viewing angles for colour evolution plots,"
                          "colours are taken from color_list array instead")
                    plotkwargs['color'] = color_list[angle_counter]  # index instaed of angle_counter??
                    angle_counter += 1
                elif args.plotviewingangle and not args.color:
                    plotkwargs['color'] = color_list[angle_counter]
                    angle_counter += 1
                elif args.color:
                    plotkwargs['color'] = args.color[modelnumber]
                if args.linestyle:
                    plotkwargs['linestyle'] = args.linestyle[modelnumber]

                if args.reflightcurves and modelnumber == 0:
                    if len(angles) > 1 and index > 0:
                        print('already plotted reflightcurve')
                    else:
                        for i, reflightcurve in enumerate(args.reflightcurves):
                            plot_color_evolution_from_data(
                                filter_names, reflightcurve, args.refspeccolors[i], args.refspecmarkers[i],
                                filternames_conversion_dict, ax, plotnumber, args)

                if args.subplots:
                    ax[plotnumber].plot(plot_times, colour_delta_mag, linewidth=4, **plotkwargs)
                else:
                    ax.plot(plot_times, colour_delta_mag, linewidth=3, **plotkwargs)

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
    ax = at.plottools.set_axis_properties(ax, args)
    ax = set_lightcurveplot_legend(ax, args)

    args.outputfile = os.path.join(outputfolder, f'plotcolorevolution{filter_names[0]}-{filter_names[1]}.pdf')
    for i in range(2):
        if filter_names[i] in filternames_conversion_dict:
            filter_names[i] = filternames_conversion_dict[filter_names[i]]
    # plt.text(10, args.ymax - 0.5, f'{filter_names[0]}-{filter_names[1]}', fontsize='x-large')

    if args.show:
        plt.show()
    plt.savefig(args.outputfile, format='pdf')

# Just in case it's needed...

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

    lightcurve_data, metadata = at.lightcurve.read_reflightcurve_band_data(lightcurvefilename)
    linename = metadata['label'] if plotnumber == 0 else None
    filterdir = os.path.join(at.config['path_artistools_dir'], 'data/filters/')

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
                                   filternames_conversion_dict, ax, plotnumber, args):
    lightcurve_from_data, metadata = at.lightcurve.read_reflightcurve_band_data(lightcurvefilename)
    filterdir = os.path.join(at.config['path_artistools_dir'], 'data/filters/')

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
        ax[plotnumber].plot(
            merge_dataframes['time'], merge_dataframes['magnitude_x'] - merge_dataframes['magnitude_y'],
            marker, label=metadata['label'], color=color, linewidth=4)
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

    parser.add_argument('--title', action='store_true',
                        help='Show title of plot')

    parser.add_argument('-color', default=[f'C{i}' for i in range(10)], nargs='*',
                        help='List of line colors')

    parser.add_argument('-linestyle', default=[], nargs='*',
                        help='List of line styles')

    parser.add_argument('-linewidth', default=[], nargs='*',
                        help='List of line widths')

    parser.add_argument('-dashes', default=[], nargs='*',
                        help='Dashes property of lines')

    parser.add_argument('-figscale', type=float, default=1.,
                        help='Scale factor for plot area. 1.0 is for single-column')

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

    parser.add_argument('-o', '-outputfile', action='store', dest='outputfile', type=Path,
                        help='Filename for PDF file')

    parser.add_argument('--plotcmf', '--plot_cmf', '--showcmf', '--show_cmf',
                        action='store_true',
                        help='Plot comoving frame light curve')

    parser.add_argument('--plotinvalidpart',
                        action='store_true',
                        help='Plot the entire light curve including partially accumulated parts (light travel time effects)')

    parser.add_argument('--plotdeposition',
                        action='store_true',
                        help='Plot model deposition rates')

    parser.add_argument('--plotthermalisation',
                        action='store_true',
                        help='Plot thermalisation rates')

    parser.add_argument('--magnitude', action='store_true',
                        help='Plot light curves in magnitudes')

    parser.add_argument('--Lsun', action='store_true',
                        help='Plot light curves in units of Lsun')

    parser.add_argument('-filter', '-band', dest='filter', type=str, nargs='+',
                        help='Choose filter eg. bol U B V R I. Default B. '
                        'WARNING: filter names are not case sensitive eg. sloan-r is not r, it is rs')

    parser.add_argument('-colour_evolution', nargs='*',
                        help='Plot of colour evolution. Give two filters eg. B-V')

    parser.add_argument('--print_data', action='store_true',
                        help='Print plotted data')

    parser.add_argument('--write_data', action='store_true',
                        help='Save data used to generate the plot in a text file')

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

    parser.add_argument('--logscalex', action='store_true',
                        help='Use log scale for horizontal axis')

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

    # parser.add_argument('--calculate_peakmag_risetime_delta_m15', action='store_true',
    #                     help='Calculate band risetime, peak mag and delta m15 values for '
    #                     'the models specified using a polynomial fitting method and '
    #                     'print to screen')

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

    parser.add_argument('--noerrorbars', action='store_true',
                        help="Don't plot error bars on viewing angle scatter plots")

    parser.add_argument('--noangleaveraged', action='store_true',
                        help="Don't plot angle averaged values on viewing angle scatter plots")

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

    parser.add_argument('--colouratpeak', action='store_true',
                        help='Make scatter plot of colour at peak for viewing angles')

    parser.add_argument('--brightnessattime', action='store_true',
                        help='Make scatter plot of light curve brightness at a given time (requires timedays)')

    parser.add_argument('-timedays', '-time', '-t', type=float,
                        help='Time in days to plot')

    parser.add_argument('--nomodelname', action='store_true',
                        help='Model name not added to linename in legend')

    parser.add_argument('-legendsubplotnumber', type=int, default=1,
                        help='Subplot number to place legend in. Default is subplot[1]')

    parser.add_argument('-legendposition', type=str, default='best',
                        help='Position of legend in plot. Default is best')

    parser.add_argument('-ncolslegend', type=int, default=1,
                        help='Number of columns in legend')

    parser.add_argument('--legendframeon', action='store_true',
                        help='Frame on in legend')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
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
    else:
        outputfolder = Path()

    filternames_conversion_dict = {'rs': 'r', 'gs': 'g', 'is': 'i'}

    # determine if this will be a scatter plot or not
    args.calculate_peak_time_mag_deltam15_bool = False
    if (    # args.calculate_peakmag_risetime_delta_m15 or
            args.save_viewing_angle_peakmag_risetime_delta_m15_to_file
            or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
            or args.make_viewing_angle_peakmag_risetime_scatter_plot
            or args.make_viewing_angle_peakmag_delta_m15_scatter_plot):
        args.calculate_peak_time_mag_deltam15_bool = True
        at.lightcurve.peakmag_risetime_declinerate_init(modelpaths, filternames_conversion_dict, args)
        return

    if args.colouratpeak:  # make scatter plot of colour at peak, eg. B-V at Bmax
        at.lightcurve.make_peak_colour_viewing_angle_plot(args)
        return

    if args.brightnessattime:
        if args.timedays is None:
            print('Specify timedays')
            quit()
        if not args.plotviewingangle:
            args.plotviewingangle = [-1]
        if not args.colorbarcostheta and not args.colorbarphi:
            args.colorbarphi = True
        at.lightcurve.plot_viewanglebrightness_at_fixed_time(Path(modelpaths[0]), args)
        return

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
