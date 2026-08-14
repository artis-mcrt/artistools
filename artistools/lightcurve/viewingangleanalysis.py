"""Measure and plot how peak magnitude, rise time, and decline rate vary with viewing angle."""

import argparse
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.legend_handler import HandlerTuple

import artistools as at
from artistools.lightcurve.lightcurve import FILTERNAME_ALIASES
from artistools.plottools import save_figure

_base_colours = [
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

# the first cycle inserts five extra unique colours before bisque, then the 13-colour base repeats
define_colours_list = [
    *_base_colours[:11],
    "darkgreen",
    "maroon",
    "mediumvioletred",
    "saddlebrown",
    "darkslategrey",
    *_base_colours[11:],
    *(_base_colours * 7),
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
    """Return the direction bins selected by args, and a label for each of them."""
    modelpath = Path(modelpath)
    at.check_averaging_angles(args.average_over_phi_angle, args.average_over_theta_angle)

    viewing_angle_data_exists = args.frompackets or bool(list(modelpath.glob("*_res.out*")))
    if isinstance(args.plotviewingangle, int):
        args.plotviewingangle = [args.plotviewingangle]
    dirbins: list[int] = []
    if args.plotvspecpol and (modelpath / "vpkt.txt").is_file():
        dirbins = args.plotvspecpol
    elif args.plotviewingangle and args.plotviewingangle[0] == -2 and viewing_angle_data_exists:
        dirbins = at.get_dirbins(
            average_over_phi=args.average_over_phi_angle, average_over_theta=args.average_over_theta_angle
        )
    elif args.plotviewingangle and viewing_angle_data_exists:
        dirbins = args.plotviewingangle
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
            # averaging over theta leaves one bin per phi index
            for dirbin in dirbin_definition:
                assert dirbin < at.get_viewingdirection_phibincount() or dirbin == -1

    return dirbins, dirbin_definition


def save_viewing_angle_data_for_plotting(band_name: str, modelname: str, args: argparse.Namespace) -> None:
    """Write one model's per-direction-bin peak magnitude, rise time, and decline rate to a text file."""
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


def write_viewing_angle_data(band_name: str, modelnames: list[str], args: argparse.Namespace) -> None:
    """Write the angle-averaged peak magnitude, rise time, and decline rate of every model to a text file."""
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
            header=f"object {band_name}_band_risetime {band_name}_band_peakmag {band_name}_band_deltam15 ",
            comments="",
        )


def calculate_peak_time_mag_deltam15(
    time: Sequence[float],
    magnitude: npt.NDArray[np.floating],
    modelname: str,
    angle: int,
    key: str,
    args: argparse.Namespace,
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

    arr_xfit = np.asarray(xfit, dtype=float)
    tmax_polyfit = float(arr_xfit[np.argmin(fxfit)])
    peakmag_polyfit = float(np.min(fxfit))

    def index_of_days_after_peak(days: float) -> int:
        """Return the index of the fitted point closest to days after peak, warning if the fit stops short.

        Searching for an exact float match instead would leave the index unset whenever the requested time did
        not land precisely on a fitted point.
        """
        index = int(np.abs(arr_xfit - (tmax_polyfit + days)).argmin())
        if arr_xfit[-1] < tmax_polyfit + days:
            print(
                f"WARNING: the fitted range ends at {arr_xfit[-1]:.1f} d, which is before {days:.0f} d after the"
                f" peak at {tmax_polyfit:.1f} d. deltam{days:.0f} is really the decline to"
                f" {arr_xfit[-1] - tmax_polyfit:.1f} d after peak. Widen -timemax to cover the full range."
            )
        return index

    index_after_15_days = index_of_days_after_peak(15)
    time_after15days_polyfit = float(arr_xfit[index_after_15_days])
    mag_after15days_polyfit = fxfit[index_after_15_days]

    print(f"{key}_max polyfit = {peakmag_polyfit} at time = {tmax_polyfit}")
    print(f"deltam15 polyfit = {peakmag_polyfit - mag_after15days_polyfit}")

    args.band_risetime_polyfit.append(tmax_polyfit)
    args.band_peakmag_polyfit.append(peakmag_polyfit)
    args.band_deltam15_polyfit.append((peakmag_polyfit - mag_after15days_polyfit) * -1)
    if args.include_delta_m40:
        mag_after40days_polyfit = fxfit[index_of_days_after_peak(40)]
        print(f"deltam40 polyfit = {peakmag_polyfit - mag_after40days_polyfit}")
        args.band_deltam40_polyfit.append((peakmag_polyfit - mag_after40days_polyfit) * -1)

    # Plotting the lightcurves for all viewing angles specified in the command line along with the
    # polynomial fit and peak mag, risetime to peak and delta m15 marked on the plots to check the
    # fit is working correctly
    if args.test_viewing_angle_fit:
        make_plot_test_viewing_angle_fit(
            time,
            magnitude,
            xfit,
            fxfit,
            key,
            mag_after15days_polyfit,
            tmax_polyfit,
            time_after15days_polyfit,
            modelname,
            angle,
            args,
        )


def lightcurve_polyfit(
    time: Sequence[float],
    magnitude: npt.NDArray[np.floating],
    args: argparse.Namespace,
    deg: float = 10,
    kernel_scale: float = 10,
    lc_error: float = 0.01,
) -> tuple[t.Any, t.Any]:
    """Return a smooth fit to a band light curve, as (fitted magnitudes, times) in that order.

    Note the fitted values come first, not the times. The fit uses a george Gaussian process, falling back to a
    polynomial when george is unavailable.
    """
    try:
        import george

        # scipy is not a direct dependency of artistools, but george depends on it. Import it here
        # so a george install without scipy also falls back to the polynomial fit.
        import scipy.optimize as op

    except ModuleNotFoundError:
        print(
            "Could not find 'george' module, falling back to polynomial fit. WARNING: polynomial fit method is sensitive to the degrees of freedom used in the polynomial fit. "
            "Therefore, it is important to check which degree of freedom used in the polynomial provides the best fit using the --test_viewing_angle_fit flag"
        )
        zfit = np.polyfit(x=time, y=magnitude, deg=deg)
        xfit = np.linspace(args.timemin + 0.5, args.timemax - 0.5, num=1000)

        # Taking line_min and line_max from the limits set for the lightcurve being plotted
        # polynomial with 10 degrees of freedom used here but change as required if it improves the fit
        fxfit = np.poly1d(zfit)
        pred = fxfit(xfit)
    else:
        from george import kernels

        kernel = np.var(magnitude) * kernels.Matern32Kernel(kernel_scale)
        gp = george.GP(kernel)

        # Define the objective function (negative log-likelihood in this case).
        def nll(p: npt.NDArray[np.floating]) -> float:
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(magnitude, quiet=True)
            return -ll if np.isfinite(ll) else 1e25

        # And the gradient of the objective function.
        def grad_nll(p: npt.NDArray[np.floating]) -> t.Any:
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(magnitude, quiet=True)

        # You need to compute the GP once before starting the optimization.
        gp.compute(time, yerr=np.abs(magnitude) * lc_error)  # pyright: ignore[reportArgumentType]

        # Run the optimization routine.
        p0 = gp.get_parameter_vector()
        results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

        # Update the kernel and print the final log-likelihood.
        gp.set_parameter_vector(results.x)

        xfit = np.linspace(min(time), max(time), 1000)
        pred, _ = gp.predict(magnitude, xfit, return_var=True)

    return pred, xfit


def make_plot_test_viewing_angle_fit(
    time: Sequence[float],
    magnitude: npt.NDArray[np.floating],
    xfit: Sequence[float],
    fxfit: Sequence[float],
    key: str,
    mag_after15days_polyfit: float,
    tmax_polyfit: float,
    time_after15days_polyfit: float | str,
    modelname: str,
    angle: int,
    args: argparse.Namespace,
) -> None:
    """Plot a band light curve against its fit, so the quality of the fit can be checked by eye."""
    fig, axis = plt.subplots()
    axis.plot(time, magnitude)
    axis.plot(xfit, fxfit)

    if key in FILTERNAME_ALIASES:
        axis.set_ylabel(f"{FILTERNAME_ALIASES[key]} Magnitude")
    else:
        axis.set_ylabel(f"{key} Magnitude")

    axis.set_xlabel("Time Since Explosion [d]")
    axis.invert_yaxis()
    axis.set_xlim(args.timemin / 1.05, args.timemax * 1.05)
    axis.minorticks_on()
    axis.tick_params(axis="both", which="minor", top=True, right=True, length=5, width=2, labelsize=12)
    axis.tick_params(axis="both", which="major", top=True, right=True, length=8, width=2, labelsize=12)
    axis.axhline(y=min(fxfit), color="black", linestyle="--")
    axis.axhline(y=mag_after15days_polyfit, color="black", linestyle="--")
    axis.axvline(x=tmax_polyfit, color="black", linestyle="--")
    axis.axvline(x=float(time_after15days_polyfit), color="black", linestyle="--")
    print("time after 15 days polyfit = ", time_after15days_polyfit)
    fig.tight_layout()
    plotname = f"{key}_band_{modelname}_viewing_angle{angle!s}.png"
    save_figure(fig, plotname)


def set_scatterplot_plotkwargs(modelnumber: int, args: argparse.Namespace) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """Return the plot kwargs for one model's per-direction-bin points and for its angle-averaged point."""
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
    """Set one colour per direction bin in the plot kwargs, matching the viewing angle colorbar."""
    costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
    scaledmap = at.lightcurve.plotlightcurve.make_colorbar_viewingangles_colormap()

    angles = list(range(at.get_viewingdirectionbincount()))
    colors = []
    for angle in angles:
        colorindex: t.Any
        _, colorindex = at.lightcurve.plotlightcurve.get_viewinganglecolor_for_colorbar(
            angle, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargsviewingangles, args
        )
        colors.append(scaledmap.to_rgba(colorindex))
    plotkwargsviewingangles["color"] = colors
    return plotkwargsviewingangles


def set_scatterplot_plot_params(fig: mplfig.Figure, axis: mplax.Axes, args: argparse.Namespace) -> None:
    """Set the axis limits, labels, and legend shared by the viewing angle scatter plots."""
    if not args.colouratpeak:
        axis.invert_yaxis()
    axis.set_xlim(args.xmin, args.xmax)
    axis.set_ylim(args.ymin, args.ymax)
    axis.minorticks_on()
    axis.tick_params(axis="both", which="minor", top=False, right=False, length=5, width=2, labelsize=12)
    axis.tick_params(axis="both", which="major", top=False, right=False, length=8, width=2, labelsize=12)
    fig.tight_layout()

    if args.colorbarcostheta or args.colorbarphi:
        _, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
        scaledmap = at.lightcurve.plotlightcurve.make_colorbar_viewingangles_colormap()
        at.lightcurve.plotlightcurve.make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, ax=axis)


def make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot(
    modelnames: Sequence[str], key: str, args: argparse.Namespace
) -> None:
    """Scatter plot peak magnitude against rise time or decline rate, one point per direction bin per model."""
    fig, ax = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=(8, 6), tight_layout={"pad": 0.5, "w_pad": 1.5, "h_pad": 0.3}
    )

    for ii, modelname in enumerate(modelnames):
        viewing_angle_plot_data = at.read_wsv(f"{key}band_{modelname!s}_viewing_angle_data.txt")

        band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].cast(pl.Float64).to_numpy()
        band_delta_m15_viewing_angles = viewing_angle_plot_data["deltam15_polyfit"].cast(pl.Float64).to_numpy()
        band_risetime_viewing_angles = viewing_angle_plot_data["risetime_polyfit"].cast(pl.Float64).to_numpy()

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
    set_scatterplot_plot_params(fig, ax, args)

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        filename = rf"{key}_band_{modelnames[0]}_dm15_peakmag.pdf"
    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        filename = rf"{key}_band_{modelnames[0]}_risetime_peakmag.pdf"
    save_figure(fig, filename, format="pdf")


def make_peak_colour_viewing_angle_plot(args: argparse.Namespace) -> None:
    """Scatter plot the colour at peak against the peak magnitude, one point per direction bin per model."""
    fig, ax = plt.subplots(
        nrows=1, ncols=1, sharex=True, figsize=(8, 6), tight_layout={"pad": 0.5, "w_pad": 1.5, "h_pad": 0.3}
    )

    for modelnumber, modelpath in enumerate(args.modelpath):
        modelname = at.get_model_name(modelpath)

        bands = [args.filter[0], args.filter[1]]

        datafilename = f"{bands[0]}band_{modelname}_viewing_angle_data.txt"
        viewing_angle_plot_data = at.read_wsv(datafilename)
        data = {f"{bands[0]}max": viewing_angle_plot_data["peak_mag_polyfit"].cast(pl.Float64).to_numpy()}
        data[f"time_{bands[0]}max"] = viewing_angle_plot_data["risetime_polyfit"].cast(pl.Float64).to_numpy()

        # Get brightness in second band at time of peak in first band
        if len(data[f"time_{bands[0]}max"]) != 100:
            print(f"All 100 angles are not in file {datafilename}. Quitting")
            sys.exit(1)

        second_band_brightness: t.Any = second_band_brightness_at_peak_first_band(
            data, bands, modelpath, modelnumber, args
        )

        data[f"{bands[1]}at{bands[0]}max"] = second_band_brightness

        dfdata = pl.DataFrame(data).with_columns(
            (pl.col(f"{bands[0]}max") - pl.col(f"{bands[1]}at{bands[0]}max")).alias("peakcolour")
        )
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
    set_scatterplot_plot_params(fig, ax, args)
    plotname = f"plotviewinganglecolour{bands[0]}-{bands[1]}.pdf"
    save_figure(fig, plotname, format="pdf")


def second_band_brightness_at_peak_first_band(
    data: dict[str, npt.NDArray[np.float64]],
    bands: Sequence[str],
    modelpath: Path,
    modelnumber: int,
    args: argparse.Namespace,
) -> list[float]:
    """Return the second band's magnitude at the time the first band peaks, for each direction bin."""
    second_band_brightness: list[float] = []
    for anglenumber, _ in enumerate(data[f"time_{bands[0]}max"]):
        lightcurve_data = at.lightcurve.generate_band_lightcurve_data(
            modelpath, args, anglenumber, modelnumber=modelnumber
        )
        time, brightness_in_mag = at.lightcurve.get_band_lightcurve(lightcurve_data, bands[1], args)

        fxfit, xfit = lightcurve_polyfit(time, brightness_in_mag, args)

        closest_list_time_to_first_band_peak = at.match_closest_time(
            reftime=data[f"time_{bands[0]}max"][anglenumber], searchtimes=xfit
        )

        for ii, xfits in enumerate(xfit):
            if float(xfits) == closest_list_time_to_first_band_peak:
                index_at_max = ii
                break

        brightness_in_second_band_at_first_band_peak = fxfit[index_at_max]
        print(brightness_in_second_band_at_first_band_peak)
        second_band_brightness.append(brightness_in_second_band_at_first_band_peak)

    return second_band_brightness


def peakmag_risetime_declinerate_init(
    modelpaths: list[str | Path] | list[Path] | list[str], args: argparse.Namespace
) -> None:
    """Fit every model's band light curves and store the peak magnitudes, rise times, and decline rates on args."""
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
        lcdataframes: dict[int, pl.LazyFrame] = {}

        if not args.filter:
            lcname = "light_curve_res.out" if args.plotviewingangle else "light_curve.out"
            lcpath = at.firstexisting(lcname, folder=modelpath, tryzipped=True)
            lcdataframes = at.lightcurve.readfile(lcpath)

        # check if doing viewing angle stuff, and if so define which data to use
        dirbins, _ = parse_directionbin_args(modelpath, args)
        if not args.filter and args.plotviewingangle:
            dirbins = [-1]

        for dirbin in dirbins:
            modelname = at.get_model_name(modelpath)
            modelnames.append(modelname)  # save for later
            print(f"Reading spectra: {modelname}")
            if args.filter:
                lightcurve_data_filters = at.lightcurve.generate_band_lightcurve_data(
                    modelpath, args, dirbin, modelnumber=modelnumber
                )
                plottinglist = args.filter

            if not args.filter:
                plottinglist = ["lightcurve"]

            for band_name in plottinglist:
                if args.filter:
                    time, brightness = at.lightcurve.get_band_lightcurve(lightcurve_data_filters, band_name, args)
                else:
                    lightcurve_data = (
                        lcdataframes[dirbin]
                        .filter(pl.col("time_days").is_between(args.timemin, args.timemax))
                        .fill_nan(0.0)
                        .select("time_days", "mag")
                        .collect()
                    )
                    brightness = np.array(
                        [mag for mag in lightcurve_data["mag"] if mag != 0], dtype=np.float64
                    )  # drop times with 0 brightness
                    time = [
                        t
                        for t, mag in zip(lightcurve_data["time_days"], lightcurve_data["mag"], strict=False)
                        if mag != 0
                    ]

                # Calculating band peak time, peak magnitude and delta m15
                calculate_peak_time_mag_deltam15(time, brightness, modelname, dirbin, band_name, args)

        # Saving viewing angle data so it can be read in and plotted later on without re-running the script
        #    as it is quite time consuming
        save_viewing_angle_data_for_plotting(plottinglist[0], modelname, args)

    # Saving all this viewing angle info for each model to a file so that it is available to plot if required again
    # as it takes relatively long to run this for all viewing angles
    write_viewing_angle_data(plottinglist[0], modelnames, args)

    # if args.make_viewing_angle_peakmag_risetime_scatter_plot:
    #     make_viewing_angle_peakmag_risetime_scatter_plot(modelnames, plottinglist[0], args)
    #     return

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot or args.make_viewing_angle_peakmag_risetime_scatter_plot:
        make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot(modelnames, plottinglist[0], args)
        return


def plot_viewanglebrightness_at_fixed_time(modelpath: Path, args: argparse.Namespace) -> None:
    """Plot the luminosity of each direction bin at one time, to show the angular brightness variation."""
    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharey=True, figsize=(8, 5), tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0}
    )

    costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
    scaledmap = at.lightcurve.plotlightcurve.make_colorbar_viewingangles_colormap()

    plotkwargs: dict[str, t.Any] = {}

    lcdataframes = at.lightcurve.readfile(modelpath / "light_curve_res.out")

    timetoplot = at.match_closest_time(
        reftime=args.timedays, searchtimes=lcdataframes[0].collect()["time_days"].to_list()
    )
    print(timetoplot)

    for angleindex, lcdata in lcdataframes.items():
        angle = angleindex
        plotkwargs, _ = at.lightcurve.plotlightcurve.get_viewinganglecolor_for_colorbar(
            angle, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargs, args
        )

        lumattime = lcdata.filter(pl.col("time_days") == timetoplot).select("luminosity_Lsun").collect().item(0, 0)
        brightness = lumattime * at.constants.Lsun_to_erg_per_s
        if args.colorbarphi:
            xvalues = int(angleindex / 10)
            xlabels = costheta_viewing_angle_bins
        if args.colorbarcostheta:
            xvalues = angleindex % 10
            xlabels = phi_viewing_angle_bins

        axis.scatter(xvalues, brightness, **plotkwargs)
        axis.set_xticks(ticks=np.arange(0, 10), labels=xlabels, rotation=30, ha="right")

    at.lightcurve.plotlightcurve.make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, fig, axis)

    axis.set_xlabel("Angle bin")
    axis.set_ylabel("erg/s")
    axis.set_yscale("log")

    axis.set_title(f"time = {args.timedays} days")
    if args.show:
        plt.show()

    plotname = f"plotviewinganglebrightnessat{args.timedays}days.pdf"
    save_figure(fig, plotname, format="pdf")
