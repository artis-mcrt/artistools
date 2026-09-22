"""Measure and plot how peak magnitude, rise time, and decline rate vary with viewing angle."""

import argparse
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.legend_handler import HandlerTuple

from artistools.lightcurve.core import FILTERNAME_ALIASES
from artistools.lightcurve.core import find_lightcurve_file
from artistools.lightcurve.core import generate_band_lightcurve_data
from artistools.lightcurve.core import get_band_lightcurve
from artistools.lightcurve.core import get_phillips_relation_data
from artistools.lightcurve.core import scan_lightcurve
from artistools.misc import check_averaging_angles
from artistools.misc import exit_with_error
from artistools.misc import get_costhetabin_phibin_labels
from artistools.misc import get_dirbin_definitions
from artistools.misc import get_dirbins
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_series_label
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import match_closest_time
from artistools.misc import print_warning
from artistools.misc import read_wsv
from artistools.misc import resolve_outputfile
from artistools.plottools import get_viewinganglecolor_for_colorbar
from artistools.plottools import invert_magnitude_yaxis
from artistools.plottools import make_colorbar_viewingangles
from artistools.plottools import make_colorbar_viewingangles_colormap
from artistools.plottools import make_frame_figure
from artistools.plottools import save_figure
from artistools.plottools import set_axis_properties
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title


def parse_directionbin_args(modelpath: Path | str, args: argparse.Namespace) -> tuple[Sequence[int], dict[int, str]]:
    """Return the direction bins selected by args, and a label for each of them."""
    modelpath = Path(modelpath)
    check_averaging_angles(args.average_over_phi_angle, args.average_over_theta_angle)

    viewing_angle_data_exists = args.frompackets or bool(list(modelpath.glob("*_res.out*")))
    dirbins: list[int] = []
    if args.plotvspecpol and (modelpath / "vpkt.txt").is_file():
        dirbins = args.plotvspecpol
    elif args.plotviewingangle and args.plotviewingangle[0] == -2 and viewing_angle_data_exists:
        dirbins = get_dirbins(
            average_over_phi=args.average_over_phi_angle, average_over_theta=args.average_over_theta_angle
        )
    elif args.plotviewingangle and viewing_angle_data_exists:
        dirbins = args.plotviewingangle
    else:
        dirbins = [-1]

    dirbin_definition = get_dirbin_definitions(
        modelpath,
        dirbins,
        vpkt_observers=bool(args.plotvspecpol),
        average_over_phi=args.average_over_phi_angle,
        average_over_theta=args.average_over_theta_angle,
        usedegrees=args.usedegrees,
    )

    if not args.plotvspecpol:
        if args.average_over_phi_angle:
            for dirbin in dirbin_definition:
                assert dirbin % get_viewingdirection_phibincount() == 0 or dirbin == -1

        if args.average_over_theta_angle:
            # averaging over theta leaves one bin per phi index
            for dirbin in dirbin_definition:
                assert dirbin < get_viewingdirection_phibincount() or dirbin == -1

    return dirbins, dirbin_definition


def wants_angle_averaged_data(args: argparse.Namespace) -> bool:
    """Return whether the command-line arguments ask for the angle-averaged peak magnitude data."""
    return bool(
        args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
        or args.make_viewing_angle_peakmag_risetime_scatter_plot
        or args.make_viewing_angle_peakmag_delta_m15_scatter_plot
    )


def get_viewing_angle_data_folder(args: argparse.Namespace) -> Path:
    """Return the folder that holds the files of the viewing angle data, which -o names."""
    return resolve_outputfile(args.outputfile, "viewingangledata.txt").parent


def save_viewing_angle_data_for_plotting(
    band_name: str, modelname: str, dirbins: Sequence[int], args: argparse.Namespace
) -> None:
    """Write one model's per-direction-bin peak magnitude, rise time, and decline rate to a text file."""
    if args.save_viewing_angle_peakmag_risetime_delta_m15_to_file:
        # the first column names the direction bin of each row, thus a later plot needs no -plotviewingangle
        columns: list[Sequence[float]] = [
            list(dirbins),
            args.band_peakmag_polyfit,
            args.band_risetime_polyfit,
            args.band_deltam15_polyfit,
        ]
        header = "dirbin peak_mag_polyfit risetime_polyfit deltam15_polyfit"
        if args.include_delta_m40:
            columns.append(args.band_deltam40_polyfit)
            header += " deltam40_polyfit"
        np.savetxt(
            get_viewing_angle_data_folder(args) / f"{band_name}band_{modelname}_viewing_angle_data.txt",
            np.column_stack(columns),
            delimiter=" ",
            fmt=["%d", *["%.18e"] * (len(columns) - 1)],
            header=header,
            comments="",
        )

    elif wants_angle_averaged_data(args) and band_name == (args.filter[0] if args.filter else "lightcurve"):
        # the angle-averaged writers index these lists by model, thus the first band alone gives one entry
        args.band_risetime_angle_averaged_polyfit.append(args.band_risetime_polyfit)
        args.band_peakmag_angle_averaged_polyfit.append(args.band_peakmag_polyfit)
        args.band_delta_m15_angle_averaged_polyfit.append(args.band_deltam15_polyfit)

    args.band_risetime_polyfit = []
    args.band_peakmag_polyfit = []
    args.band_deltam15_polyfit = []
    if args.include_delta_m40:
        args.band_deltam40_polyfit = []


def write_viewing_angle_data(band_name: str, modelnames: list[str], args: argparse.Namespace) -> None:
    """Write the angle-averaged peak magnitude, rise time, and decline rate of every model to a text file."""
    if wants_angle_averaged_data(args):
        np.savetxt(
            get_viewing_angle_data_folder(args) / f"{band_name}band_{modelnames[0]}_angle_averaged_all_models_data.txt",
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
    print_warning(
        "Both methods that can be used to fit model light curves to get "
        "light curve parameters (rise, decline, peak) can be impacted by how much "
        "of the light curve is being fitted. It is safest to experiment with the  "
        "timemin and timemax args which set the region of the light curve fitted. "
        "The --test_viewing_angle_fit flag will allow you to check the fitting is "
        "behaving as expected. In general fitting over a smaller region of the    "
        "light curve tends to produce better fits."
    )
    fxfit, xfit = lightcurve_polyfit(time, magnitude)

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
            print_warning(
                f"the fitted range ends at {arr_xfit[-1]:.1f} d, which is before {days:.0f} d after the"
                f" peak at {tmax_polyfit:.1f} d. deltam{days:.0f} is really the decline to"
                f" {arr_xfit[-1] - tmax_polyfit:.1f} d after peak. Widen -timemax to cover the full range."
            )
        return index

    index_after_15_days = index_of_days_after_peak(15)
    time_after15days_polyfit = float(arr_xfit[index_after_15_days])
    mag_after15days_polyfit = fxfit[index_after_15_days]

    print(f"{key}_max polyfit = {peakmag_polyfit} at time = {tmax_polyfit}")
    # a decline rate is the magnitude after the peak minus the peak magnitude, thus it is positive for a fainter source
    deltam15_polyfit = mag_after15days_polyfit - peakmag_polyfit
    print(f"deltam15 polyfit = {deltam15_polyfit}")

    args.band_risetime_polyfit.append(tmax_polyfit)
    args.band_peakmag_polyfit.append(peakmag_polyfit)
    args.band_deltam15_polyfit.append(deltam15_polyfit)
    if args.include_delta_m40:
        deltam40_polyfit = fxfit[index_of_days_after_peak(40)] - peakmag_polyfit
        print(f"deltam40 polyfit = {deltam40_polyfit}")
        args.band_deltam40_polyfit.append(deltam40_polyfit)

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


def lightcurve_polyfit(time: Sequence[float], magnitude: npt.NDArray[np.floating]) -> tuple[t.Any, t.Any]:
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
        print_warning(
            "Could not find 'george' module, falling back to polynomial fit. WARNING: polynomial fit method is sensitive to the degrees of freedom used in the polynomial fit. "
            "Therefore, it is important to check which degree of freedom used in the polynomial provides the best fit using the --test_viewing_angle_fit flag"
        )
        zfit = np.polyfit(x=time, y=magnitude, deg=10)
        # the fit has no meaning outside the data, thus this range is the range of the data, as in the george branch
        xfit = np.linspace(min(time), max(time), num=1000)

        # Taking line_min and line_max from the limits set for the lightcurve being plotted
        # polynomial with 10 degrees of freedom used here but change as required if it improves the fit
        fxfit = np.poly1d(zfit)
        pred = fxfit(xfit)
    else:
        from george import kernels

        kernel = np.var(magnitude) * kernels.Matern32Kernel(10)
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
        gp.compute(time, yerr=np.abs(magnitude) * 0.01)  # pyright: ignore[reportArgumentType]

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
    fig, axesgrid = make_frame_figure(args)
    axis = axesgrid[0][0]
    axis.plot(time, magnitude)
    axis.plot(xfit, fxfit)

    axis.set_ylabel(f"{FILTERNAME_ALIASES.get(key, key)} Magnitude")

    axis.set_xlabel("Time Since Explosion [d]")
    # -ymin and -ymax can already give the limits in the order of a magnitude axis, and invert_yaxis() toggles them back
    set_axis_properties(axis, args, xlimits=(args.timemin / 1.05, args.timemax * 1.05, "-timemin"))
    invert_magnitude_yaxis(axis)
    axis.axhline(y=min(fxfit), color="black", linestyle="--")
    axis.axhline(y=mag_after15days_polyfit, color="black", linestyle="--")
    axis.axvline(x=tmax_polyfit, color="black", linestyle="--")
    axis.axvline(x=float(time_after15days_polyfit), color="black", linestyle="--")
    print("time after 15 days polyfit = ", time_after15days_polyfit)
    plotname = get_viewing_angle_data_folder(args) / f"{key}_band_{modelname}_viewing_angle{angle!s}.png"
    save_figure(fig, plotname, args=args)


def set_scatterplot_plotkwargs(
    modelnumber: int, dfdata: pl.DataFrame, datafilename: Path, args: argparse.Namespace
) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    """Return the plot kwargs for one model's per-direction-bin points and for its angle-averaged point."""
    plotkwargsviewingangles = {"marker": "x", "zorder": 0, "alpha": 0.8}
    if args.colorbarcostheta or args.colorbarphi:
        dirbins = get_datafile_dirbins(dfdata, datafilename, args)
        update_plotkwargs_for_viewingangle_colorbar(plotkwargsviewingangles, dirbins, args)
    else:
        plotkwargsviewingangles["color"] = args.color[modelnumber]

    plotkwargsangleaveraged = {
        "marker": "o",
        "zorder": 10,
        "edgecolor": "k",
        "s": 120,
        "color": args.color[modelnumber],
    }

    return plotkwargsviewingangles, plotkwargsangleaveraged


def get_datafile_dirbins(dfdata: pl.DataFrame, datafilename: Path, args: argparse.Namespace) -> list[int]:
    """Return the direction bin of each row of a viewing angle data file.

    A file of the current version holds a dirbin column, which names the bin of each row. An older file
    holds no such column. The rows of such a file follow the selection of the run that wrote it, thus the
    same selection gives the direction bins.
    """
    if "dirbin" in dfdata.columns:
        dirbins = dfdata["dirbin"].cast(pl.Int64).to_list()
    else:
        selection = args.plotvspecpol or args.plotviewingangle
        selection = [selection] if isinstance(selection, int) else selection
        if selection and selection[0] != -2:
            dirbins = list(selection)
        else:
            dirbins = get_dirbins(
                average_over_phi=args.average_over_phi_angle, average_over_theta=args.average_over_theta_angle
            )

    if len(dirbins) != dfdata.height or -1 in dirbins:
        msg = (
            f"The colour bar needs the direction bin of each row of {datafilename},"
            f" which has {dfdata.height} rows and gives {len(dirbins)} direction bins."
            " Write the file again with a -plotviewingangle selection, and give the same selection here."
        )
        raise ValueError(msg)

    return dirbins


def update_plotkwargs_for_viewingangle_colorbar(
    plotkwargsviewingangles: dict[str, t.Any], dirbins: Sequence[int], args: argparse.Namespace
) -> dict[str, t.Any]:
    """Set the colour of each direction bin in the plot kwargs, matching the viewing angle colorbar."""
    scaledmap = make_colorbar_viewingangles_colormap()

    colors = []
    for dirbin in dirbins:
        colorindex: t.Any
        _, colorindex = get_viewinganglecolor_for_colorbar(dirbin, scaledmap, plotkwargsviewingangles, args)
        colors.append(scaledmap.to_rgba(colorindex))
    plotkwargsviewingangles["color"] = colors
    return plotkwargsviewingangles


def set_scatterplot_plot_params(axis: mplax.Axes, args: argparse.Namespace) -> None:
    """Set the axis limits, labels, and legend shared by the viewing angle scatter plots."""
    # the x axis here is a rise time or a decline rate, not a time since explosion, so it takes no limit
    # from the command line: this parser spells -xmin/-xmax as aliases of the -timemin/-timemax time range
    set_axis_properties(axis, args, xlimits=(None, None, "-xmin"))
    if not args.colouratpeak:
        # after the limits: set_ylim re-sorts the pair it is given, so an inversion applied first is lost
        invert_magnitude_yaxis(axis)

    if args.colorbarcostheta or args.colorbarphi:
        scaledmap = make_colorbar_viewingangles_colormap()
        make_colorbar_viewingangles(scaledmap, args, ax=axis)


def make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot(
    modelnames: Sequence[str], key: str, args: argparse.Namespace
) -> None:
    """Scatter plot peak magnitude against rise time or decline rate, one point per direction bin per model."""
    fig, axesgrid = make_frame_figure(args)
    ax = axesgrid[0][0]

    datafolder = get_viewing_angle_data_folder(args)
    for ii, modelname in enumerate(modelnames):
        datafilename = datafolder / f"{key}band_{modelname!s}_viewing_angle_data.txt"
        viewing_angle_plot_data = read_wsv(datafilename)

        band_peak_mag_viewing_angles = viewing_angle_plot_data["peak_mag_polyfit"].cast(pl.Float64).to_numpy()
        band_delta_m15_viewing_angles = viewing_angle_plot_data["deltam15_polyfit"].cast(pl.Float64).to_numpy()
        band_risetime_viewing_angles = viewing_angle_plot_data["risetime_polyfit"].cast(pl.Float64).to_numpy()

        plotkwargsviewingangles, plotkwargsangleaveraged = set_scatterplot_plotkwargs(
            ii, viewing_angle_plot_data, datafilename, args
        )

        # the error bars below use the angle-averaged x value whether or not its point is drawn
        if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
            xvalues_viewingangles = band_delta_m15_viewing_angles
            xvalues_angleaveraged = args.band_delta_m15_angle_averaged_polyfit[ii]
        if args.make_viewing_angle_peakmag_risetime_scatter_plot:
            xvalues_viewingangles = band_risetime_viewing_angles
            xvalues_angleaveraged = args.band_risetime_angle_averaged_polyfit[ii]

        a0 = ax.scatter(xvalues_viewingangles, band_peak_mag_viewing_angles, **plotkwargsviewingangles)

        if not args.noangleaveraged:
            p0 = ax.scatter(
                xvalues_angleaveraged, args.band_peakmag_angle_averaged_polyfit[ii], **plotkwargsangleaveraged
            )
            args.plotvalues.append((a0, p0))
        else:
            args.plotvalues.append((a0, a0))
        if not args.noerrorbars:
            ax.errorbar(
                xvalues_angleaveraged,
                args.band_peakmag_angle_averaged_polyfit[ii],
                xerr=np.std(xvalues_viewingangles),
                yerr=np.std(band_peak_mag_viewing_angles),
                ecolor=args.color[ii],
                capsize=2,
            )

    linelabels = [get_series_label(args.label, ii, modelname) for ii, modelname in enumerate(modelnames)]

    set_legend(
        ax,
        args,
        handles=args.plotvalues,
        labels=linelabels,
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper right",
        fontsize="x-small",
        ncol=args.ncolslegend,
        columnspacing=1,
        frameon=False,
    )

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        xlabel = rf"$\Delta$m$_{{15}}$({key})"
    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        xlabel = "Rise Time [days]"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"M$_{{\mathrm{{{key}}}}}$, max")
    set_scatterplot_plot_params(ax, args)

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot:
        filename = rf"{key}_band_{modelnames[0]}_dm15_peakmag.pdf"
    if args.make_viewing_angle_peakmag_risetime_scatter_plot:
        filename = rf"{key}_band_{modelnames[0]}_risetime_peakmag.pdf"
    save_figure(fig, datafolder / filename, format="pdf", args=args)


def make_peak_colour_viewing_angle_plot(args: argparse.Namespace) -> None:
    """Scatter plot the colour at peak against the peak magnitude, one point per direction bin per model."""
    fig, axesgrid = make_frame_figure(args)
    ax = axesgrid[0][0]

    for modelnumber, modelpath in enumerate(args.modelpath):
        modelname = get_model_name(modelpath)

        bands = [args.filter[0], args.filter[1]]

        datafilename = get_viewing_angle_data_folder(args) / f"{bands[0]}band_{modelname}_viewing_angle_data.txt"
        viewing_angle_plot_data = read_wsv(datafilename)
        data = {f"{bands[0]}max": viewing_angle_plot_data["peak_mag_polyfit"].cast(pl.Float64).to_numpy()}
        data[f"time_{bands[0]}max"] = viewing_angle_plot_data["risetime_polyfit"].cast(pl.Float64).to_numpy()

        # Get brightness in second band at time of peak in first band
        if len(data[f"time_{bands[0]}max"]) != 100:
            print(f"All 100 angles are not in file {datafilename}. Quitting")
            sys.exit(1)

        second_band_brightness: t.Any = second_band_brightness_at_peak_first_band(data, bands, modelpath, args)

        data[f"{bands[1]}at{bands[0]}max"] = second_band_brightness

        dfdata = pl.DataFrame(data).with_columns(
            (pl.col(f"{bands[0]}max") - pl.col(f"{bands[1]}at{bands[0]}max")).alias("peakcolour")
        )
        print(dfdata["peakcolour"], dfdata[f"{bands[0]}max"], dfdata[f"{bands[1]}at{bands[0]}max"])

        plotkwargsviewingangles, _ = set_scatterplot_plotkwargs(
            modelnumber, viewing_angle_plot_data, datafilename, args
        )
        plotkwargsviewingangles["label"] = modelname
        ax.scatter(dfdata["peakcolour"], y=dfdata[f"{bands[0]}max"], **plotkwargsviewingangles)

    sn_data, label = get_phillips_relation_data()
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

    set_legend(ax, args, loc="upper right")
    ax.set_xlabel(f"{bands[0]}-{bands[1]} at {bands[0]}max")
    ax.set_ylabel(f"{bands[0]}max")
    set_scatterplot_plot_params(ax, args)
    plotname = get_viewing_angle_data_folder(args) / f"plotviewinganglecolour{bands[0]}-{bands[1]}.pdf"
    save_figure(fig, plotname, format="pdf", args=args)


def second_band_brightness_at_peak_first_band(
    data: dict[str, npt.NDArray[np.float64]], bands: Sequence[str], modelpath: Path, args: argparse.Namespace
) -> list[float]:
    """Return the second band's magnitude at the time the first band peaks, for each direction bin."""
    second_band_brightness: list[float] = []
    for anglenumber, _ in enumerate(data[f"time_{bands[0]}max"]):
        lightcurve_data = generate_band_lightcurve_data(modelpath, args, anglenumber)
        time, brightness_in_mag = get_band_lightcurve(lightcurve_data, bands[1], args)

        fxfit, xfit = lightcurve_polyfit(time, brightness_in_mag)

        index_at_max = int(np.abs(np.asarray(xfit, dtype=float) - data[f"time_{bands[0]}max"][anglenumber]).argmin())

        brightness_in_second_band_at_first_band_peak = fxfit[index_at_max]
        print(brightness_in_second_band_at_first_band_peak)
        second_band_brightness.append(brightness_in_second_band_at_first_band_peak)

    return second_band_brightness


def peakmag_risetime_declinerate_init(
    modelpaths: list[str | Path] | list[Path] | list[str], args: argparse.Namespace
) -> None:
    """Fit every model's band light curves and store the peak magnitudes, rise times, and decline rates on args."""
    if args.save_viewing_angle_peakmag_risetime_delta_m15_to_file and wants_angle_averaged_data(args):
        # writing the per-direction-bin files takes the branch that never measures the angle-averaged
        # values, so the two steps of the workflow cannot run at once. Say so before reading any spectra
        msg = (
            "The angle-averaged peak magnitudes are not measured while"
            " --save_viewing_angle_peakmag_risetime_delta_m15_to_file writes the per-direction-bin data."
            " Write the data in one run, then plot it in another."
        )
        raise ValueError(msg)

    if not args.filter and args.plotvspecpol is not None:
        # without a filter the fit reads light_curve_res.out, whose direction bins are not virtual packet observers
        exit_with_error(
            "-plotvspecpol names virtual packet observers, which light_curve_res.out does not hold",
            "Give -filter to make the light curve of each virtual packet observer from its spectra.",
        )

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

    # a band light curve comes from the spectra, and the bolometric light curve from light_curve.out
    plottinglist: list[str] = list(args.filter) if args.filter else ["lightcurve"]

    for modelpath in modelpaths:
        modelname = get_model_name(modelpath)
        # one entry per model, matching the per-model style lists and the one data file written per model
        modelnames.append(modelname)
        lcdataframes: dict[int, pl.LazyFrame] = {}

        # check if doing viewing angle stuff, and if so define which data to use
        dirbins, _ = parse_directionbin_args(modelpath, args)
        if args.plotviewingangle and wants_angle_averaged_data(args):
            # the angle-averaged modes fit the light curve of dirbin -1 alone. Thus a list of the bins
            # would give the scatter plot one angle-averaged point for each bin. The per-direction-bin
            # export keeps the parsed direction bins
            dirbins = [-1]

        dfbolo_of_dirbin: dict[int, pl.DataFrame] = {}
        if not args.filter:
            # dirbin -1 is the angle-averaged light curve, which only light_curve.out holds. The
            # direction-resolved bins come from light_curve_res.out
            directionresolved = list(dirbins) != [-1]
            lcpath = find_lightcurve_file(modelpath, directionresolved=directionresolved)
            # a mode that averages over the angles groups several direction bins, and dirbins then
            # names the first bin of each group. Thus the reader must average in the same way
            lcdataframes = (
                scan_lightcurve(
                    lcpath,
                    average_over_phi=args.average_over_phi_angle,
                    average_over_theta=args.average_over_theta_angle,
                )
                if directionresolved
                else scan_lightcurve(lcpath)
            )
            # scan_lightcurve slices one scan of the file. Thus one collect_all parses it one time for
            # every direction bin, in place of one parse for each bin
            lazyplans = [
                lcdataframes[dirbin]
                .filter(pl.col("time_days").is_between(args.timemin, args.timemax))
                .fill_nan(0.0)
                # a time with no luminosity has no magnitude, thus drop the zero and the
                # non-finite values before the fit
                .filter(pl.col("mag").is_finite() & (pl.col("mag") != 0.0))
                .select("time_days", "mag")
                for dirbin in dirbins
            ]
            dfbolo_of_dirbin = dict(zip(dirbins, pl.collect_all(lazyplans), strict=True))

        if args.verbose:
            print(f"Reading spectra: {get_model_logname(modelpath)}")
        # the spectra of a direction bin hold every band, thus read each direction bin one time before the band loop
        lightcurve_data_filters_of_dirbin = (
            {dirbin: generate_band_lightcurve_data(modelpath, args, dirbin) for dirbin in dirbins}
            if args.filter
            else {}
        )

        # the fit results of one band fill the shared lists, and the save function then empties them for the next band
        for band_name in plottinglist:
            for dirbin in dirbins:
                if args.filter:
                    time, brightness = get_band_lightcurve(lightcurve_data_filters_of_dirbin[dirbin], band_name, args)
                else:
                    lightcurve_data = dfbolo_of_dirbin[dirbin]
                    brightness = lightcurve_data["mag"].to_numpy()
                    time = lightcurve_data["time_days"].to_list()

                # Calculating band peak time, peak magnitude and delta m15
                calculate_peak_time_mag_deltam15(time, brightness, modelname, dirbin, band_name, args)

            # write the data of this band to a file. A later plot then reads the file in place of the slow fit
            save_viewing_angle_data_for_plotting(band_name, modelname, dirbins, args)

    # Saving all this viewing angle info for each model to a file so that it is available to plot if required again
    # as it takes relatively long to run this for all viewing angles
    write_viewing_angle_data(plottinglist[0], modelnames, args)

    if args.make_viewing_angle_peakmag_delta_m15_scatter_plot or args.make_viewing_angle_peakmag_risetime_scatter_plot:
        make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot(modelnames, plottinglist[0], args)
        return


def plot_viewanglebrightness_at_fixed_time(modelpath: Path, args: argparse.Namespace) -> None:
    """Plot the luminosity of each direction bin at one time, to show the angular brightness variation."""
    fig, axesgrid = make_frame_figure(args)
    axis = axesgrid[0][0]

    costheta_viewing_angle_bins, phi_viewing_angle_bins = get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
    nphibins = get_viewingdirection_phibincount()
    scaledmap = make_colorbar_viewingangles_colormap()

    plotkwargs: dict[str, t.Any] = {}

    lcdataframes_lazy = scan_lightcurve(find_lightcurve_file(modelpath, directionresolved=True))

    # one collect_all call parses light_curve_res.out one time for all the direction bins
    lcdataframes = dict(zip(lcdataframes_lazy.keys(), pl.collect_all(list(lcdataframes_lazy.values())), strict=True))

    timetoplot = match_closest_time(reftime=args.timedays, searchtimes=lcdataframes[0]["time_days"].to_list())
    print(timetoplot)

    # the colorbar shows one angle, thus the x axis shows the other one
    xlabels = phi_viewing_angle_bins if args.colorbarcostheta else costheta_viewing_angle_bins
    for angleindex, lcdata in lcdataframes.items():
        plotkwargs, _ = get_viewinganglecolor_for_colorbar(angleindex, scaledmap, plotkwargs, args)

        # scan_lightcurve derives the erg/s column, so it does not have to be converted here again
        brightness = lcdata.filter(pl.col("time_days") == timetoplot).select("luminosity_erg/s").item(0, 0)
        costhetaindex, phiindex = divmod(angleindex, nphibins)
        xvalues = phiindex if args.colorbarcostheta else costhetaindex

        axis.scatter(xvalues, brightness, **plotkwargs)

    axis.set_xticks(ticks=np.arange(len(xlabels)), labels=xlabels, rotation=30, ha="right")

    make_colorbar_viewingangles(scaledmap, args, fig, axis)

    axis.set_xlabel("Angle bin")
    axis.set_ylabel("erg/s")
    axis.set_yscale("log")

    set_plot_title(axis, f"time = {args.timedays} days", args)
    plotname = f"plotviewinganglebrightnessat{args.timedays}days.pdf"
    save_figure(fig, plotname, format="pdf", args=args)
