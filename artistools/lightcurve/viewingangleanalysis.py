import math
import os
from pathlib import Path
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.legend_handler import HandlerTuple
from astropy import constants as const

import artistools as at
from .lightcurve import *


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


def get_viewinganglebin_definitions():
    costheta_viewing_angle_bins = ['-1.0 \u2264 cos(\u03B8) < -0.8', '-0.8 \u2264 cos(\u03B8) < -0.6',
                                   '-0.6 \u2264 cos(\u03B8) < -0.4', '-0.4 \u2264 cos(\u03B8) < -0.2',
                                   '-0.2 \u2264 cos(\u03B8) < 0', '0 \u2264 cos(\u03B8) < 0.2',
                                   '0.2 \u2264 cos(\u03B8) < 0.4', '0.4 \u2264 cos(\u03B8) < 0.6',
                                   '0.6 \u2264 cos(\u03B8) < 0.8', '0.8 \u2264 cos(\u03B8) < 1']
    phi_viewing_angle_bins = ['0 \u2264 \u03D5 < \u03c0/5', '\u03c0/5 \u2264 \u03D5 < 2\u03c0/5',
                              '2\u03c0/5 \u2264 \u03D5 < 3\u03c0/5', '3\u03c0/5 \u2264 \u03D5 < 4\u03c0/5',
                              '4\u03c0/5 \u2264 \u03D5 < \u03c0', '9\u03c0/5 < \u03D5 < 2\u03c0',
                              '8\u03c0/5 < \u03D5 \u2264 9\u03c0/5', '7\u03c0/5 < \u03D5 \u2264 8\u03c0/5',
                              '6\u03c0/5 < \u03D5 \u2264 7\u03c0/5', '\u03c0 < \u03D5 \u2264 6\u03c0/5']
    return costheta_viewing_angle_bins, phi_viewing_angle_bins


def calculate_costheta_phi_for_viewing_angles(viewing_angles, modelpath):
    modelpath = Path(modelpath)
    if os.path.isfile(modelpath / 'absorptionpol_res_99.out') \
            and os.path.isfile(modelpath / 'absorptionpol_res_100.out'):
        print("Too many viewing angle bins (MABINS) for this method to work, it only works for MABINS = 100")
        exit()
    elif os.path.isfile(modelpath / 'light_curve_res.out'):
        angle_definition = {}

        costheta_viewing_angle_bins, phi_viewing_angle_bins = get_viewinganglebin_definitions()

        for angle in viewing_angles:
            MABINS = 100
            phibins = int(math.sqrt(MABINS))
            costheta_index = angle // phibins
            phi_index = angle % phibins

            angle_definition[angle] = f'{costheta_viewing_angle_bins[costheta_index]}, {phi_viewing_angle_bins[phi_index]}'
            print(f"{angle:4d}   {costheta_viewing_angle_bins[costheta_index]}   {phi_viewing_angle_bins[phi_index]}")

        return angle_definition
    else:
        print("Too few viewing angle bins (MABINS) for this method to work, it only works for MABINS = 100")
        exit()


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


def calculate_peak_time_mag_deltam15(time, magnitude, modelname, angle, key, args, filternames_conversion_dict=None):
    """Calculating band peak time, peak magnitude and delta m15"""
    if args.timemin is None or args.timemax is None:
        print("Trying to calculate peak time / dm15 / rise time with no time range. "
              "This will give a stupid result. Specify args.timemin and args.timemax")
        quit()

    zfit = np.polyfit(x=time, y=magnitude, deg=10)
    xfit = np.linspace(args.timemin, args.timemax, num=1000)

    # Taking line_min and line_max from the limits set for the lightcurve being plotted
    fxfit = []
    for j, _ in enumerate(xfit):
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
        make_plot_test_viewing_angle_fit(time, magnitude, xfit, fxfit, filternames_conversion_dict, key,
                                         mag_after15days_polyfit, tmax_polyfit, time_after15days_polyfit,
                                         modelname, angle)


def make_plot_test_viewing_angle_fit(time, magnitude, xfit, fxfit, filternames_conversion_dict, key,
                                     mag_after15days_polyfit, tmax_polyfit, time_after15days_polyfit,
                                     modelname, angle):
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


def peakmag_risetime_declinerate_init(modelpaths, filternames_conversion_dict, args):

    # if args.calculate_peak_time_mag_deltam15_bool:  # If there's viewing angle scatter plot stuff define some arrays
    args.plotvalues = []  # a0 and p0 values for viewing angle scatter plots

    args.band_risetime_polyfit = []
    args.band_peakmag_polyfit = []
    args.band_deltam15_polyfit = []

    args.band_risetime_angle_averaged_polyfit = []
    args.band_peakmag_angle_averaged_polyfit = []
    args.band_delta_m15_angle_averaged_polyfit = []

    modelnames = [] # save names of models

    for modelnumber, modelpath in enumerate(modelpaths):
        modelpath = Path(modelpath)

        if not args.filter:
            if args.plotviewingangle:
                lcname = 'light_curve_res.out'
            else:
                lcname = 'light_curve.out'
            lcpath = at.firstexisting([lcname + '.xz', lcname + '.gz', lcname], path=modelpath)
            print(f"Reading {lcname}")
            lightcurve_data = at.lightcurve.readfile(lcpath, args)

        # check if doing viewing angle stuff, and if so define which data to use
        angles, viewing_angles, angle_definition = get_angle_stuff(modelpath, args)
        if not args.filter and args.plotviewingangle:
            lcdataframes = lightcurve_data

        for index, angle in enumerate(angles):

            modelname = at.get_model_name(modelpath)
            modelnames.append(modelname)  # save for later
            print(f'Reading spectra: {modelname}')
            if args.filter:
                lightcurve_data = get_band_lightcurve_data(modelpath, args, angle, modelnumber=modelnumber)
                plottinglist = args.filter
            elif args.plotviewingangle:
                lightcurve_data = lcdataframes[angle]
            if not args.filter:
                plottinglist = ['lightcurve']

            for plotnumber, band_name in enumerate(plottinglist):
                if args.filter:
                    time, brightness = get_band_lightcurve_data_to_plot(lightcurve_data, band_name, args)
                else:
                    lightcurve_data = lightcurve_data.loc[(lightcurve_data['time'] > args.timemin) &
                                                          (lightcurve_data['time'] < args.timemax)]
                    time = lightcurve_data['time']
                    lightcurve_data['mag'] = 4.74 - (2.5 * np.log10((lightcurve_data['lum'] * 3.826e33)
                                                                    / const.L_sun.to('erg/s').value))

                    lightcurve_data = lightcurve_data.replace([np.inf, -np.inf], 0)
                    brightness = [mag for mag in lightcurve_data['mag'] if mag != 0]
                    time = [t for t, mag in zip(lightcurve_data['time'], lightcurve_data['mag']) if mag != 0]

                # Calculating band peak time, peak magnitude and delta m15
                if args.calculate_peak_time_mag_deltam15_bool:
                    calculate_peak_time_mag_deltam15(time, brightness, modelname, angle, band_name,
                                                     args, filternames_conversion_dict=filternames_conversion_dict)

        # Saving viewing angle data so it can be read in and plotted later on without re-running the script
        #    as it is quite time consuming
        if args.calculate_peak_time_mag_deltam15_bool:
            save_viewing_angle_data_for_plotting(band_name, modelname, args)

    # Saving all this viewing angle info for each model to a file so that it is available to plot if required again
    # as it takes relatively long to run this for all viewing angles
    if args.calculate_peak_time_mag_deltam15_bool:
        write_viewing_angle_data(band_name, modelname, modelnames, args)

    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, band_name, args)
        return

    elif args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        make_viewing_angle_peakmag_delta_m15_scatter_plot(modelnames, band_name, args)
        return
