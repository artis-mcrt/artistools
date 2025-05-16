import argparse
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

import artistools as at

define_colours_list = [
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "darkgreen",
    "maroon",
    "mediumvioletred",
    "saddlebrown",
    "darkslategrey",
    "bisque",
    "yellow",
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "bisque",
    "yellow",
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "bisque",
    "yellow",
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "bisque",
    "yellow",
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "bisque",
    "yellow",
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "bisque",
    "yellow",
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "bisque",
    "yellow",
    "k",
    "tab:blue",
    "tab:red",
    "tab:green",
    "purple",
    "tab:orange",
    "tab:pink",
    "tab:gray",
    "gold",
    "tab:cyan",
    "darkblue",
    "bisque",
    "yellow",
]

define_colours_list2 = [
    "gray",
    "lightblue",
    "pink",
    "yellowgreen",
    "mediumorchid",
    "sandybrown",
    "plum",
    "lightgray",
    "wheat",
    "paleturquoise",
    "royalblue",
    "springgreen",
    "r",
    "deeppink",
    "sandybrown",
    "teal",
]


def parse_directionbin_args(modelpath: Path | str, args: argparse.Namespace) -> tuple[Sequence[int], dict[int, str]]:
    modelpath = Path(modelpath)
    viewing_angle_data_exists = args.frompackets or bool(list(modelpath.glob("*_res.out*")))
    if isinstance(args.plotviewingangle, int):
        args.plotviewingangle = [args.plotviewingangle]
    dirbins = []
    if args.plotvspecpol and (modelpath / "vpkt.txt").is_file():
        dirbins = args.plotvspecpol
    elif args.plotviewingangle and args.plotviewingangle[0] == -2 and viewing_angle_data_exists:
        dirbins = list(np.arange(0, 100, 1, dtype=int))
        if args.average_over_phi_angle:
            dirbins = [d for d in dirbins if d % at.get_viewingdirection_phibincount() == 0]
        if args.average_over_theta_angle:
            dirbins = [d for d in dirbins if d < at.get_viewingdirection_costhetabincount()]
    elif args.plotviewingangle and viewing_angle_data_exists:
        dirbins = args.plotviewingangle
    elif (
        args.calculate_costheta_phi_from_viewing_angle_numbers
        and args.calculate_costheta_phi_from_viewing_angle_numbers[0] == -2
    ):
        dirbins = list(np.arange(0, 100, 1, dtype=int))
    elif args.calculate_costheta_phi_from_viewing_angle_numbers:
        dirbins = args.calculate_costheta_phi_from_viewing_angle_numbers
        assert dirbins is not None
    else:
        dirbins = [-1]

    dirbin_definition: dict[int, str] = {}

    if args.plotvspecpol:
        dirbin_definition = at.get_vspec_dir_labels(modelpath=modelpath, usedegrees=args.usedegrees)
    else:
        dirbin_definition = at.get_dirbin_labels(
            dirbins=dirbins,
            modelpath=modelpath,
            average_over_phi=args.average_over_phi_angle,
            average_over_theta=args.average_over_theta_angle,
            usedegrees=args.usedegrees,
        )

        if args.average_over_phi_angle:
            for dirbin in dirbin_definition:
                assert dirbin % at.get_viewingdirection_phibincount() == 0 or dirbin == -1

        if args.average_over_theta_angle:
            for dirbin in dirbin_definition:
                assert dirbin < at.get_viewingdirection_costhetabincount() or dirbin == -1

    return dirbins, dirbin_definition


def save_viewing_angle_data_for_plotting(band_name, modelname, args: argparse.Namespace) -> None:
    if args.save_viewing_angle_peakmag_risetime_delta_m15_to_file:
        outputfolder = Path(args.outputfile) if Path(args.outputfile).is_dir() else Path(args.outputfile).parent
        if args.include_delta_m40:
            np.savetxt(
                outputfolder / f"{band_name}band_{modelname}_viewing_angle_data.txt",
                np.c_[
                    args.band_peakmag_polyfit,
                    args.band_risetime_polyfit,
                    args.band_deltam15_polyfit,
                    args.band_deltam40_polyfit,
                ],
                delimiter=" ",
                header="peak_mag_polyfit risetime_polyfit deltam15_polyfit deltam40_polyfit",
                comments="",
            )
        else:
            np.savetxt(
                outputfolder / f"{band_name}band_{modelname}_viewing_angle_data.txt",
                np.c_[args.band_peakmag_polyfit, args.band_risetime_polyfit, args.band_deltam15_polyfit],
                delimiter=" ",
                header="peak_mag_polyfit risetime_polyfit deltam15_polyfit",
                comments="",
            )

    elif (
        args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
        or args.make_viewing_angle_peakmag_risetime_scatter_plot
        or args.make_viewing_angle_peakmag_delta_m15_scatter_plot
    ):
        args.band_risetime_angle_averaged_polyfit.append(args.band_risetime_polyfit)
        args.band_peakmag_angle_averaged_polyfit.append(args.band_peakmag_polyfit)
        args.band_delta_m15_angle_averaged_polyfit.append(args.band_deltam15_polyfit)

    args.band_risetime_polyfit = []
    args.band_peakmag_polyfit = []
    args.band_deltam15_polyfit = []
    if args.include_delta_m40:
        args.band_deltam40_polyfit = []

    # if args.magnitude and not (
    #         args.calculate_peakmag_risetime_delta_m15 or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
    #         or args.save_viewing_angle_peakmag_risetime_delta_m15_to_file or args.test_viewing_angle_fit
    #         or args.make_viewing_angle_peakmag_risetime_scatter_plot or
    #         args.make_viewing_angle_peakmag_delta_m15_scatter_plot or args.plotviewingangle):
    #     plt.plot(time, magnitude, label=modelname, color=colours[modelnumber], linewidth=3)


def write_viewing_angle_data(band_name, modelnames, args: argparse.Namespace) -> None:
    if (
        args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
        or args.make_viewing_angle_peakmag_risetime_scatter_plot
        or args.make_viewing_angle_peakmag_delta_m15_scatter_plot
    ):
        np.savetxt(
            f"{band_name}band_{modelnames[0]}_angle_averaged_all_models_data.txt",
            np.c_[
                modelnames,
                args.band_risetime_angle_averaged_polyfit,
                args.band_peakmag_angle_averaged_polyfit,
                args.band_delta_m15_angle_averaged_polyfit,
            ],
            delimiter=" ",
            fmt="%s",
            header="object "
            + str(band_name)
            + "_band_risetime "
            + str(band_name)
            + "_band_peakmag "
            + str(band_name)
            + "_band_deltam15 ",
            comments="",
        )


def calculate_peak_time_mag_deltam15(
    time, magnitude, modelname, angle, key, args, filternames_conversion_dict=None
) -> None:
    """Calculate band peak time, peak magnitude and delta m15."""
    if args.timemin is None or args.timemax is None:
        print(
            "Trying to calculate peak time / dm15 / rise time with no time range. "
            "This will give a stupid result. Specify args.timemin and args.timemax"
        )
        sys.exit(1)
    print(
        "WARNING: Both methods that can be used to fit model light curves to get  "
        "light curve parameters (rise, decline, peak) can be impacted by how much "
        "of the light curve is being fitted. It is safest to experiment with the  "
        "timemin and timemax args which set the region of the light curve fitted. "
        "The --test_viewing_angle_fit flag will allow you to check the fitting is "
        "behaving as expected. In general fitting over a smaller region of the    "
        "light curve tends to produce better fits."
    )
    fxfit, xfit = lightcurve_polyfit(time, magnitude, args)

    def match_closest_time_polyfit(reftime_polyfit):
        return str(min((float(x) for x in xfit), key=lambda x: abs(x - reftime_polyfit)))

    index_min = np.argmin(fxfit)
    tmax_polyfit = xfit[index_min]
    time_after15days_polyfit = match_closest_time_polyfit(tmax_polyfit + 15)
    if args.include_delta_m40:
        time_after40days_polyfit = match_closest_time_polyfit(tmax_polyfit + 40)
    index_after_40_days = None
    for ii, xfits in enumerate(xfit):
        if float(xfits) == float(time_after15days_polyfit):
            index_after_15_days = ii
        elif args.include_delta_m40 and float(xfits) == float(time_after40days_polyfit):
            index_after_40_days = ii

    mag_after15days_polyfit = fxfit[index_after_15_days]
    print(f"{key}_max polyfit = {min(fxfit)} at time = {tmax_polyfit}")
    print(f"deltam15 polyfit = {min(fxfit) - mag_after15days_polyfit}")

    args.band_risetime_polyfit.append(tmax_polyfit)
    args.band_peakmag_polyfit.append(min(fxfit))
    args.band_deltam15_polyfit.append((min(fxfit) - mag_after15days_polyfit) * -1)
    if args.include_delta_m40:
        mag_after40days_polyfit = fxfit[index_after_40_days]
        print(f"deltam40 polyfit = {min(fxfit) - mag_after40days_polyfit}")
        args.band_deltam40_polyfit.append((min(fxfit) - mag_after40days_polyfit) * -1)

    # Plotting the lightcurves for all viewing angles specified in the command line along with the
    # polynomial fit and peak mag, risetime to peak and delta m15 marked on the plots to check the
    # fit is working correctly
    if args.test_viewing_angle_fit:
        make_plot_test_viewing_angle_fit(
            time,
            magnitude,
            xfit,
            fxfit,
            filternames_conversion_dict,
            key,
            mag_after15days_polyfit,
            tmax_polyfit,
            time_after15days_polyfit,
            modelname,
            angle,
            args,
        )


def lightcurve_polyfit(time, magnitude, args, deg=10, kernel_scale=10, lc_error=0.01):
    try:
        import george
        import scipy.optimize as op
        from george import kernels

        kernel = np.var(magnitude) * kernels.Matern32Kernel(kernel_scale)
        gp = george.GP(kernel)

        # Define the objective function (negative log-likelihood in this case).
        def nll(p):
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(magnitude, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(magnitude, quiet=True)

        # You need to compute the GP once before starting the optimization.
        gp.compute(time, yerr=np.abs(magnitude) * lc_error)

        # Run the optimization routine.
        p0 = gp.get_parameter_vector()
        results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

        # Update the kernel and print the final log-likelihood.
        gp.set_parameter_vector(results.x)

        xfit = np.linspace(min(time), max(time), 1000)
        pred, _ = gp.predict(magnitude, xfit, return_var=True)

    except ModuleNotFoundError:
        print(
            "Could not find 'george' module, falling back to polynomial fit  "
            "WARNING: polynomial fit method is sensitive to the degrees of   "
            "freedom used in the polynomial fit. Therefore important to check"
            "which degree of freedom used in the polynomial provides the best"
            "fit using the --test_viewing_angle_fit flag                     "
        )
        zfit = np.polyfit(x=time, y=magnitude, deg=deg)
        xfit = np.linspace(args.timemin + 0.5, args.timemax - 0.5, num=1000)

        # Taking line_min and line_max from the limits set for the lightcurve being plotted
        # polynomial with 10 degrees of freedom used here but change as required if it improves the fit
        fxfit = np.poly1d(zfit)
        pred = fxfit(xfit)

    return pred, xfit


def make_plot_test_viewing_angle_fit(
    time,
    magnitude,
    xfit,
    fxfit,
    filternames_conversion_dict,
    key,
    mag_after15days_polyfit,
    tmax_polyfit,
    time_after15days_polyfit,
    modelname,
    angle,
    args,
) -> None:
    plt.plot(time, magnitude)
    plt.plot(xfit, fxfit)

    if key in filternames_conversion_dict:
        plt.ylabel(f"{filternames_conversion_dict[key]} Magnitude")
    else:
        plt.ylabel(f"{key} Magnitude")

    plt.xlabel("Time Since Explosion [d]")
    plt.gca().invert_yaxis()
    plt.xlim(args.timemin / 1.05, args.timemax * 1.05)
    plt.minorticks_on()
    plt.tick_params(axis="both", which="minor", top=True, right=True, length=5, width=2, labelsize=12)
    plt.tick_params(axis="both", which="major", top=True, right=True, length=8, width=2, labelsize=12)
    plt.axhline(y=min(fxfit), color="black", linestyle="--")
    plt.axhline(y=mag_after15days_polyfit, color="black", linestyle="--")
    plt.axvline(x=tmax_polyfit, color="black", linestyle="--")
    plt.axvline(x=float(time_after15days_polyfit), color="black", linestyle="--")
    print("time after 15 days polyfit = ", time_after15days_polyfit)
    plt.tight_layout()
    plt.savefig(f"{key}_band_{modelname}_viewing_angle{angle!s}.png")
    plt.close()


def set_scatterplot_plotkwargs(modelnumber: int, args: argparse.Namespace) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    plotkwargsviewingangles = {"marker": "x", "zorder": 0, "alpha": 0.8}
    if args.colorbarcostheta or args.colorbarphi:
        update_plotkwargs_for_viewingangle_colorbar(plotkwargsviewingangles, args)
    elif args.color:
        plotkwargsviewingangles["color"] = args.color[modelnumber]
    else:
        plotkwargsviewingangles["color"] = define_colours_list2[modelnumber]

    plotkwargsangleaveraged = {
        "marker": "o",
        "zorder": 10,
        "edgecolor": "k",
        "s": 120,
        "color": args.color[modelnumber] if args.color else define_colours_list[modelnumber],
    }
    if args.colorbarcostheta or args.colorbarphi:
        update_plotkwargs_for_viewingangle_colorbar(plotkwargsviewingangles, args)

    return plotkwargsviewingangles, plotkwargsangleaveraged


def update_plotkwargs_for_viewingangle_colorbar(
    plotkwargsviewingangles: dict[str, t.Any], args: argparse.Namespace
) -> dict[str, t.Any]:
    costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
    scaledmap = at.lightcurve.plotlightcurve.make_colorbar_viewingangles_colormap()

    angles = np.arange(0, at.get_viewingdirectionbincount(), dtype=int)
    colors = []
    for angle in angles:
        colorindex: t.Any
        _, colorindex = at.lightcurve.plotlightcurve.get_viewinganglecolor_for_colorbar(
            angle, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargsviewingangles, args
        )
        colors.append(scaledmap.to_rgba(colorindex))
    plotkwargsviewingangles["color"] = colors
    return plotkwargsviewingangles


def set_scatterplot_plot_params(args) -> None:
    if not args.colouratpeak:
        plt.gca().invert_yaxis()
    plt.xlim(args.xmin, args.xmax)
    plt.ylim(args.ymin, args.ymax)
    plt.minorticks_on()
    plt.tick_params(axis="both", which="minor", top=False, right=False, length=5, width=2, labelsize=12)
    plt.tick_params(axis="both", which="major", top=False, right=False, length=8, width=2, labelsize=12)
    plt.tight_layout()

    if args.colorbarcostheta or args.colorbarphi:
        _, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
        scaledmap = at.lightcurve.plotlightcurve.make_colorbar_viewingangles_colormap()
        at.lightcurve.plotlightcurve.make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args)


# COMBINED WITH DM15 plotting function now ###
# def make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, key, args: argparse.Namespace):
#     for ii, modelname in enumerate(modelnames):
#         viewing_angle_plot_data = pd.read_csv(key + "band_" + f'{modelname}' + "_viewing_angle_data.txt",
#                                               delimiter=" ")
#         band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].values
#         band_risetime_viewing_angles = viewing_angle_plot_data["risetime_polyfit"].values
#
#         plotkwargsviewingangles, plotkwargsangleaveraged = set_scatterplot_plotkwargs(ii, args)
#
#         a0 = plt.scatter(band_risetime_viewing_angles, band_peak_mag_viewing_angles, **plotkwargsviewingangles)
#         if not args.noangleaveraged:
#             p0 = plt.scatter(args.band_risetime_angle_averaged_polyfit[ii], args.band_peakmag_angle_averaged_polyfit[ii],
#                              **plotkwargsangleaveraged)
#             args.plotvalues.append((a0, p0))
#         else:
#             args.plotvalues.append((a0, a0))
#         if not args.noerrorbars:
#             plt.errorbar(args.band_risetime_angle_averaged_polyfit[ii], args.band_peakmag_angle_averaged_polyfit[ii],
#                          xerr=np.std(band_risetime_viewing_angles),
#                          yerr=np.std(band_peak_mag_viewing_angles), ecolor=define_colours_list[ii], capsize=2)
#
#     if not args.nolegend:
#         plt.legend(args.plotvalues, modelnames, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
#                    loc='upper left', fontsize=8, ncol=2, columnspacing=1, frameon=False)
#     plt.xlabel('Rise Time in Days', fontsize=14)
#     if args.filter:
#         ylabel = 'Peak ' + key + ' Band Magnitude'
#     else:
#         ylabel = 'Peak Magnitude'
#     plt.ylabel(ylabel, fontsize=14)
#     if args.title:
#         plt.title(f"{at.get_model_name(args.modelpath[0])}")
#     set_scatterplot_plot_params(args)
#     if args.show:
#         plt.show()
#     plt.savefig(key + "_band_" + f'{modelnames[0]}' + "_viewing_angle_peakmag_risetime_scatter_plot.pdf", format="pdf")
#     print("saving " + key + "_band_" + f'{modelnames[0]}' + "_viewing_angle_peakmag_risetime_scatter_plot.pdf")
#     plt.close()


def make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot(modelnames, key, args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=(8, 6), tight_layout={"pad": 0.5, "w_pad": 1.5, "h_pad": 0.3}
    )

    for ii, modelname in enumerate(modelnames):
        viewing_angle_plot_data = pd.read_csv(key + "band_" + str(modelname) + "_viewing_angle_data.txt", delimiter=" ")

        band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].to_numpy()
        band_delta_m15_viewing_angles = viewing_angle_plot_data["deltam15_polyfit"].to_numpy()
        band_risetime_viewing_angles = viewing_angle_plot_data["risetime_polyfit"].to_numpy()

        plotkwargsviewingangles, plotkwargsangleaveraged = set_scatterplot_plotkwargs(ii, args)

        if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
            xvalues_viewingangles = band_delta_m15_viewing_angles
        if args.make_viewing_angle_peakmag_risetime_scatter_plot:
            xvalues_viewingangles = band_risetime_viewing_angles

        a0 = ax.scatter(xvalues_viewingangles, band_peak_mag_viewing_angles, **plotkwargsviewingangles)

        if not args.noangleaveraged:
            if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
                xvalues_angleaveraged = args.band_delta_m15_angle_averaged_polyfit[ii]
            if args.make_viewing_angle_peakmag_risetime_scatter_plot:
                xvalues_angleaveraged = args.band_risetime_angle_averaged_polyfit[ii]

            p0 = ax.scatter(
                xvalues_angleaveraged, args.band_peakmag_angle_averaged_polyfit[ii], **plotkwargsangleaveraged
            )
            args.plotvalues.append((a0, p0))
        else:
            args.plotvalues.append((a0, a0))
        if not args.noerrorbars:
            ecolor = args.color or define_colours_list

            ax.errorbar(
                xvalues_angleaveraged,
                args.band_peakmag_angle_averaged_polyfit[ii],
                xerr=np.std(xvalues_viewingangles),
                yerr=np.std(band_peak_mag_viewing_angles),
                ecolor=ecolor[ii],
                capsize=2,
            )

    linelabels = args.label or modelnames

    # a0, datalabel = at.lightcurve.get_sn_sample_bol()
    # a0, datalabel = at.lightcurve.plot_phillips_relation_data()
    # args.plotvalues.append((a0, a0))
    # linelabels.append(datalabel)

    if not args.nolegend:
        ax.legend(
            args.plotvalues,
            linelabels,
            numpoints=1,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper right",
            fontsize="x-small",
            ncol=args.ncolslegend,
            columnspacing=1,
            frameon=False,
        )
    # ax.set_xlabel(r'Decline Rate ($\Delta$m$_{15}$)', fontsize=14)

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        xlabel = rf"$\Delta$m$_{{15}}$({key})"
    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        xlabel = "Rise Time [days]"

    ax.set_xlabel(xlabel, fontsize=14)
    # ax.set_ylabel('Peak ' + key + ' Band Magnitude', fontsize=14)
    ax.set_ylabel(rf"M$_{{\mathrm{{{key}}}}}$, max", fontsize=14)
    set_scatterplot_plot_params(args)

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        filename = rf"{key}_band_{modelnames[0]}_dm15_peakmag.pdf"
    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        filename = rf"{key}_band_{modelnames[0]}_risetime_peakmag.pdf"
    fig.savefig(filename, format="pdf")
    print(f"saving {filename}")


def make_peak_colour_viewing_angle_plot(args: argparse.Namespace) -> None:
    fig, ax = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=(8, 6), tight_layout={"pad": 0.5, "w_pad": 1.5, "h_pad": 0.3}
    )

    for modelnumber, modelpath in enumerate(args.modelpath):
        modelname = at.get_model_name(modelpath)

        bands = [args.filter[0], args.filter[1]]

        datafilename = bands[0] + "band_" + modelname + "_viewing_angle_data.txt"
        viewing_angle_plot_data = pd.read_csv(datafilename, delimiter=" ")
        data = {f"{bands[0]}max": viewing_angle_plot_data["peak_mag_polyfit"].to_numpy()}
        data[f"time_{bands[0]}max"] = viewing_angle_plot_data["risetime_polyfit"].to_numpy()

        # Get brightness in second band at time of peak in first band
        if len(data[f"time_{bands[0]}max"]) != 100:
            print(f"All 100 angles are not in file {datafilename}. Quitting")
            sys.exit(1)

        second_band_brightness: t.Any = second_band_brightness_at_peak_first_band(
            data, bands, modelpath, modelnumber, args
        )

        data[f"{bands[1]}at{bands[0]}max"] = second_band_brightness

        dfdata = pd.DataFrame(data)
        dfdata["peakcolour"] = dfdata[f"{bands[0]}max"] - dfdata[f"{bands[1]}at{bands[0]}max"]
        print(dfdata["peakcolour"], dfdata[f"{bands[0]}max"], dfdata[f"{bands[1]}at{bands[0]}max"])

        plotkwargsviewingangles, _ = set_scatterplot_plotkwargs(modelnumber, args)
        plotkwargsviewingangles["label"] = modelname
        ax.scatter(dfdata["peakcolour"], y=dfdata[f"{bands[0]}max"], **plotkwargsviewingangles)

    sn_data, label = at.lightcurve.get_phillips_relation_data()
    ax.errorbar(
        x=sn_data["(B-V)Bmax"],
        y=sn_data["MB"],
        xerr=sn_data["err_(B-V)Bmax"],
        yerr=sn_data["err_MB"],
        color="k",
        alpha=0.9,
        marker=".",
        capsize=2,
        label=label,
        ls="None",
        zorder=-1,
    )

    ax.legend(loc="upper right", fontsize=8, ncol=1, columnspacing=1, frameon=False)
    ax.set_xlabel(f"{bands[0]}-{bands[1]} at {bands[0]}max", fontsize=14)
    ax.set_ylabel(f"{bands[0]}max", fontsize=14)
    set_scatterplot_plot_params(args)
    plotname = f"plotviewinganglecolour{bands[0]}-{bands[1]}.pdf"
    fig.savefig(plotname, format="pdf")
    print(f"saving {plotname}")


def second_band_brightness_at_peak_first_band(data, bands, modelpath, modelnumber, args: argparse.Namespace):
    second_band_brightness = []
    for anglenumber, time in enumerate(data[f"time_{bands[0]}max"]):
        lightcurve_data = at.lightcurve.generate_band_lightcurve_data(
            modelpath, args, anglenumber, modelnumber=modelnumber
        )
        time, brightness_in_mag = at.lightcurve.get_band_lightcurve(lightcurve_data, bands[1], args)

        fxfit, xfit = lightcurve_polyfit(time, brightness_in_mag, args)

        closest_list_time_to_first_band_peak = at.match_closest_time(
            reftime=data[f"time_{bands[0]}max"][anglenumber],
            searchtimes=xfit,  # pyright: ignore[reportArgumentType]
        )

        for ii, xfits in enumerate(xfit):
            if float(xfits) == float(closest_list_time_to_first_band_peak):
                index_at_max = ii
                break

        brightness_in_second_band_at_first_band_peak = fxfit[index_at_max]
        print(brightness_in_second_band_at_first_band_peak)
        second_band_brightness.append(brightness_in_second_band_at_first_band_peak)

    return second_band_brightness


def peakmag_risetime_declinerate_init(
    modelpaths: list[str | Path], filternames_conversion_dict, args: argparse.Namespace
) -> None:
    # if args.calculate_peak_time_mag_deltam15_bool:  # If there's viewing angle scatter plot stuff define some arrays
    args.plotvalues = []  # a0 and p0 values for viewing angle scatter plots

    args.band_risetime_polyfit = []
    args.band_peakmag_polyfit = []
    args.band_deltam15_polyfit = []
    if args.include_delta_m40:
        args.band_deltam40_polyfit = []

    args.band_risetime_angle_averaged_polyfit = []
    args.band_peakmag_angle_averaged_polyfit = []
    args.band_delta_m15_angle_averaged_polyfit = []

    modelnames = []  # save names of models

    for modelnumber, modelpath in enumerate(modelpaths):
        modelpath = Path(modelpath)

        lightcurve_data: t.Any

        if not args.filter:
            lcname = "light_curve_res.out" if args.plotviewingangle else "light_curve.out"
            lcpath = at.firstexisting(lcname, folder=modelpath, tryzipped=True)
            lightcurve_data = at.lightcurve.readfile(lcpath)

        # check if doing viewing angle stuff, and if so define which data to use
        angles, _ = parse_directionbin_args(modelpath, args)
        if not args.filter and args.plotviewingangle:
            lcdataframes = lightcurve_data

        for angle in angles:
            modelname = at.get_model_name(modelpath)
            modelnames.append(modelname)  # save for later
            print(f"Reading spectra: {modelname}")
            if args.filter:
                lightcurve_data_filters = at.lightcurve.generate_band_lightcurve_data(
                    modelpath, args, angle, modelnumber=modelnumber
                )
                plottinglist = args.filter
            elif args.plotviewingangle:
                lightcurve_data = lcdataframes[angle]
            if not args.filter:
                plottinglist = ["lightcurve"]

            for band_name in plottinglist:
                brightness: npt.NDArray[np.floating[t.Any]] | list[float]
                if args.filter:
                    time, brightness = at.lightcurve.get_band_lightcurve(lightcurve_data_filters, band_name, args)
                else:
                    assert isinstance(lightcurve_data, pd.DataFrame)
                    lightcurve_data = lightcurve_data.loc[
                        (lightcurve_data["time"] > args.timemin) & (lightcurve_data["time"] < args.timemax)
                    ]

                    lightcurve_data["mag"] = 4.74 - (2.5 * np.log10(lightcurve_data["lum"]))

                    lightcurve_data = lightcurve_data.replace([np.inf, -np.inf], 0)
                    brightness = [mag for mag in lightcurve_data["mag"] if mag != 0]  # drop times with 0 brightness
                    time = [
                        t for t, mag in zip(lightcurve_data["time"], lightcurve_data["mag"], strict=False) if mag != 0
                    ]

                # Calculating band peak time, peak magnitude and delta m15
                if args.calculate_peak_time_mag_deltam15_bool:
                    calculate_peak_time_mag_deltam15(
                        time,
                        brightness,
                        modelname,
                        angle,
                        band_name,
                        args,
                        filternames_conversion_dict=filternames_conversion_dict,
                    )

        # Saving viewing angle data so it can be read in and plotted later on without re-running the script
        #    as it is quite time consuming
        if args.calculate_peak_time_mag_deltam15_bool:
            save_viewing_angle_data_for_plotting(plottinglist[0], modelname, args)

    # Saving all this viewing angle info for each model to a file so that it is available to plot if required again
    # as it takes relatively long to run this for all viewing angles
    if args.calculate_peak_time_mag_deltam15_bool:
        write_viewing_angle_data(plottinglist[0], modelnames, args)

    # if args.make_viewing_angle_peakmag_risetime_scatter_plot:
    #     make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, plottinglist[0], args)
    #     return

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot or args.make_viewing_angle_peakmag_risetime_scatter_plot:
        make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot(modelnames, plottinglist[0], args)
        return


def plot_viewanglebrightness_at_fixed_time(modelpath: Path, args: argparse.Namespace) -> None:
    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0}
    )

    costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
    scaledmap = at.lightcurve.plotlightcurve.make_colorbar_viewingangles_colormap()

    plotkwargs: dict[str, t.Any] = {}

    lcdataframes: t.Any = at.lightcurve.readfile(modelpath / "light_curve_res.out")

    timetoplot = at.match_closest_time(reftime=args.timedays, searchtimes=lcdataframes[0]["time"].to_list())
    print(timetoplot)

    for angleindex, lcdata in enumerate(lcdataframes.items()):
        angle = angleindex
        plotkwargs, _ = at.lightcurve.plotlightcurve.get_viewinganglecolor_for_colorbar(
            angle, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargs, args
        )

        rowattime = lcdata.loc[lcdata["time"] == float(timetoplot)]
        brightness = (rowattime["lum"].item()) * 3.826e33
        if args.colorbarphi:
            xvalues = int(angleindex / 10)
            xlabels = costheta_viewing_angle_bins
        if args.colorbarcostheta:
            xvalues = angleindex % 10
            xlabels = phi_viewing_angle_bins

        axis.scatter(xvalues, brightness, **plotkwargs)
        plt.xticks(ticks=np.arange(0, 10), labels=xlabels, rotation=30, ha="right")

    at.lightcurve.plotlightcurve.make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args)

    axis.set_xlabel("Angle bin")
    axis.set_ylabel("erg/s")
    axis.set_yscale("log")

    axis.set_title(f"time = {args.timedays} days")
    if args.show:
        plt.show()

    plotname = f"plotviewinganglebrightnessat{args.timedays}days.pdf"
    fig.savefig(plotname, format="pdf")
    print(f"Saved figure: {plotname}")
