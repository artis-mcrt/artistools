"""Artistools - spectra plotting functions."""

import argparse
import contextlib
import math
import typing as t
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.patches as mpatches
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
from matplotlib import ticker
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

from artistools.commands import get_path
from artistools.constants import c_ang_per_s
from artistools.constants import C_cm_per_s
from artistools.constants import km_to_cm
from artistools.inputmodel import get_spatial_scales
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_dpi
from artistools.misc import addarg_figscale
from artistools.misc import addarg_filter
from artistools.misc import addarg_legend
from artistools.misc import addarg_maxpacketfiles
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_pathoption
from artistools.misc import addarg_residuals
from artistools.misc import addarg_seriesstyle
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timeminmax
from artistools.misc import addarg_timestep
from artistools.misc import addarg_verbose
from artistools.misc import addarg_viewingangle
from artistools.misc import addarg_yscale
from artistools.misc import apply_time_range_args
from artistools.misc import df_filter_minmax_bracketed
from artistools.misc import exit_with_error
from artistools.misc import find_reference_data_file
from artistools.misc import firstexisting_or_none
from artistools.misc import get_artis_run_folders
from artistools.misc import get_dirbin_definitions
from artistools.misc import get_dirbins
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_file_metadata
from artistools.misc import get_filterfunc
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_series_label
from artistools.misc import get_time_range
from artistools.misc import get_time_range_text
from artistools.misc import get_timestep_times
from artistools.misc import get_vpkt_config
from artistools.misc import KeepGivenPaths
from artistools.misc import make_output_folder
from artistools.misc import makelist
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import path_is_artis_model
from artistools.misc import path_is_codecomparison
from artistools.misc import path_is_dir
from artistools.misc import path_is_file
from artistools.misc import path_is_reference_data
from artistools.misc import print_detail
from artistools.misc import print_heading
from artistools.misc import print_saved
from artistools.misc import print_theta_phi_definitions
from artistools.misc import print_warning
from artistools.misc import read_wsv
from artistools.misc import resolve_outputfile
from artistools.misc import resolve_series_styles
from artistools.misc.fileio import modelpath_cache
from artistools.misc.fileio import resolve_modelpath
from artistools.misc.remote import is_remote_path
from artistools.misc.remote import model_path_from_text
from artistools.misc.timesteps import parse_timedays_range
from artistools.packets import get_packets
from artistools.packets import get_packets_column_names
from artistools.plottools import draw_residual_panel
from artistools.plottools import FRAMEHEIGHT_INCHES
from artistools.plottools import FRAMEWIDTH_INCHES
from artistools.plottools import label_dirbin_series
from artistools.plottools import make_frame_figure
from artistools.plottools import make_frame_figure_with_residuals
from artistools.plottools import plain_label
from artistools.plottools import print_dirbin_summary
from artistools.plottools import ResidualSeries
from artistools.plottools import save_figure
from artistools.plottools import set_auto_yscale
from artistools.plottools import set_axis_properties
from artistools.plottools import set_exponent_label
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title
from artistools.plottools import set_prop_cycle_unusedcolors
from artistools.plottools import write_residual_stats
from artistools.spectra.core import bin_spectrum
from artistools.spectra.core import check_time_range_overlaps_time_bins
from artistools.spectra.core import convert_angstroms_to_unit
from artistools.spectra.core import convert_xlimits_to_lambda_range
from artistools.spectra.core import convert_xunit_aliases_to_canonical
from artistools.spectra.core import DEFAULT_YE_SHELLS
from artistools.spectra.core import EmissionEvent
from artistools.spectra.core import FluxContributionTuple
from artistools.spectra.core import get_default_losvelocity_shells
from artistools.spectra.core import get_default_velocity_shells
from artistools.spectra.core import get_dfspectrum_x_y_with_units
from artistools.spectra.core import get_flux_contributions
from artistools.spectra.core import get_flux_contributions_from_packets
from artistools.spectra.core import get_from_packets
from artistools.spectra.core import get_lambda_bin_edges
from artistools.spectra.core import get_no_sampled_emission_message
from artistools.spectra.core import get_reference_spectrum
from artistools.spectra.core import get_shell_labels
from artistools.spectra.core import get_specpol_data
from artistools.spectra.core import get_spectra
from artistools.spectra.core import get_vspecpol_data
from artistools.spectra.core import get_vspecpol_spectrum
from artistools.spectra.core import get_xunit
from artistools.spectra.core import make_averaged_vspecfiles
from artistools.spectra.core import make_virtual_spectra_summed_file
from artistools.spectra.core import NO_SAMPLED_EMISSION_HELP
from artistools.spectra.core import parse_velocity_argument
from artistools.spectra.core import parse_xunit_argument
from artistools.spectra.core import print_integrated_flux
from artistools.spectra.core import SHELLCOLUMNS
from artistools.spectra.core import sort_and_reduce_flux_contribution_list
from artistools.spectra.core import timeshift_fluxscale_co56law
from artistools.spectra.writespectra import write_flambda_spectra

if t.TYPE_CHECKING:
    import matplotlib.typing as mplt


# the part of the height of the data that the axes of a spectrum add above the tallest peak, thus the peak does not
# touch the frame. set_legend gives the legend its own room
YMARGIN: t.Final = 0.05


def find_reference_spectrum_file_or_none(filename: Path | str) -> Path | None:
    """Return the reference spectrum path, or None when no such file exists.

    The file is either at the given path or in the bundled data/refspectra folder, and a compressed
    file with the same name is also accepted.
    """
    return find_reference_data_file(filename, "data/refspectra")


def find_reference_spectrum_file(filename: Path | str) -> Path:
    """Return the reference spectrum path, falling back to the bundled data/refspectra folder."""
    if (found := find_reference_spectrum_file_or_none(filename)) is None:
        msg = f"Reference spectrum {filename} was not found here or in the bundled data/refspectra folder"
        raise FileNotFoundError(msg)
    return found


def path_is_reference_spectrum(filepath: str | Path) -> bool:
    """Return whether the path is a reference spectrum file and not an ARTIS model.

    This mirrors path_is_reference_lightcurve, so that the two commands classify a path the same way.

    A name that ends in .out belongs to ARTIS, e.g. spec.out. A user can give reference data such a
    name as well, thus the folder decides: an ARTIS run holds input.txt beside its output files.
    """
    return path_is_reference_data(filepath, "data/refspectra")


def check_time_range_is_valid(modelpath: Path, timemin: float, timemax: float, allow_invalid: bool) -> None:
    """Warn, or raise unless allow_invalid, when the requested times fall outside the model's packet arrival range."""
    with contextlib.suppress(FileNotFoundError):
        timesteplast, validrange_start_days, validrange_end_days = get_escaped_arrivalrange(modelpath)
        problem_messages: list[str] = []
        if validrange_start_days is validrange_end_days is None:
            problem_messages.append("The model has no valid time range days")
        if validrange_start_days is not None and timemin < validrange_start_days:
            problem_messages.append(
                f"timemin {timemin} days is before the start of the valid range at {validrange_start_days:.2f} days"
            )
        if validrange_end_days is not None and timemax > validrange_end_days:
            problem_messages.append(
                f"timemax {timemax} days is after the end of the valid range at {validrange_end_days:.2f} days"
            )

        if problem_messages and not allow_invalid:
            lastend_days = get_timestep_times(modelpath, loc="end")[timesteplast]
            if timemin > lastend_days:
                # the run wrote no spectrum after its last timestep, thus --plotinvalidpart cannot give a plot
                problem_messages.append(
                    f"The run computed the timesteps up to {timesteplast}, which ends at {lastend_days:.2f} days,"
                    " thus it holds no spectrum of this time"
                )
            else:
                problem_messages.append("To override this error and plot anyway, run with --plotinvalidpart")
            raise ValueError("\n".join(problem_messages))

        for message in problem_messages:
            print_warning(message)


def get_axis_labels(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Get the x-axis and y-axis labels based on the arguments."""
    xunit = get_xunit(args.xunit)
    xtype = {"wavelength": "Wavelength", "frequency": "Frequency", "energy": "Energy"}[xunit.kind]
    str_xunit = xunit.label

    xlabel = None if args.hidexticklabels else f"{xtype} [{str_xunit}]"
    # a frequency axis draws f_nu, and an energy axis draws the flux per energy interval
    fluxsymbol = {"wavelength": r"F$_\lambda$", "frequency": r"F$_\nu$", "energy": "dF/dE"}[xunit.kind]

    ylabel = None
    if not args.hideyticklabels:
        if args.normalised:
            match args.yvariable:
                case "flux":
                    ylabel = f"Scaled {fluxsymbol}"
                case "luminosity":
                    ylabel = r"Scaled Luminosity"
                case "packetcount":
                    ylabel = r"Scaled Monte Carlo packets"
                case "photonflux":
                    ylabel = f"Scaled photons/{str_xunit}"
                case "photoncount":
                    ylabel = f"Scaled photons/{str_xunit}"
                case "eflux":
                    ylabel = "Scaled E$^2$ flux"
                case _:
                    msg = f"Unknown y-variable {args.yvariable}"
                    raise AssertionError(msg)

            if args.showemission or args.showabsorption:
                # plot_reference_spectra adds an offset to each normalised reference spectrum
                ylabel += " + offset"
        else:
            strdist = str(args.distmpc).removesuffix(".0") + " Mpc"
            match args.yvariable:
                case "flux":
                    ylabel = f"{fluxsymbol} at {strdist} [{{}}erg/s/cm$^2$/{str_xunit}]"
                case "luminosity":
                    ylabel = f"Luminosity [{{}}erg/s/{str_xunit}]"
                case "packetcount":
                    ylabel = r"{}Monte Carlo packets per bin"
                case "eflux" if xunit.kind == "wavelength":
                    # get_dfspectrum_x_y_with_units gives lambda F_lambda, which is E^2 dN/dE in erg
                    ylabel = rf"$\lambda$F$_\lambda$ at {strdist} [{{}}erg/s/cm$^2$]"
                case "eflux":
                    ylabel = f"E$^2$ flux at {strdist} [{{}}{str_xunit}/s/cm$^2$]"
                case "photoncount":
                    ylabel = f"Photon count [{{}}#/s/{str_xunit}]"
                case "photonflux":
                    ylabel = f"Photon flux at {strdist} [{{}}#/s/cm$^2$/{str_xunit}]"
                case _:
                    msg = f"Unknown y-variable {args.yvariable}"
                    raise AssertionError(msg)

        assert ylabel is not None
        if args.logscaley:
            # don't include the {} that will be replaced with the power of 10 by the custom formatter
            ylabel = ylabel.replace("{}", "")

    return xlabel, ylabel


def plot_polarisation(modelpath: Path, args: argparse.Namespace) -> None:
    """Plot the Stokes parameter selected by args.stokesparam against wavelength."""
    if args.plotvspecpol:
        angle = args.plotvspecpol[0]
        stokes_params = get_vspecpol_data(vspecindex=angle, modelpath=modelpath)
    else:
        angle = args.plotviewingangle[0] if args.plotviewingangle else -1
        stokes_params = get_specpol_data(dirbin=angle, modelpath=modelpath)

    dfspectrum = stokes_params[args.stokesparam].with_columns(lambda_angstroms=c_ang_per_s / pl.col("nu")).collect()

    timearray = dfspectrum.columns[1:-1]
    # locals, not a write-back onto args: this function runs once for each model, and a range that
    # one model resolved would then reach the next one. get_time_range refuses a timemin that sits
    # after the last timestep, thus it dropped a shorter model and printed one line
    (timestepmin, timestepmax, timemin, timemax) = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )
    if timestepmin == timestepmax == -1:
        exit_with_error(f"The time range selects no timestep of {modelpath}")
    assert timemin is not None
    assert timemax is not None

    timeavg_float = (timemin + timemax) / 2.0
    if args.plotvspecpol:
        # the time bins of the virtual packets differ from the timesteps, thus get_time_range does not test them
        check_time_range_overlaps_time_bins(
            [float(timestr) for timestr in timearray],
            timeavg_float,
            timeavg_float,
            f"vspecpol_total-{angle}.out of {modelpath}",
        )

    def timedistance(timestr: str) -> float:
        return abs(float(timestr) - timeavg_float)

    # select the column by the exact header string, because the file writes the times in its own format
    timecolname = min(timearray, key=timedistance)
    timeavg = f"{float(timecolname):.4f}"

    filterfunc = get_filterfunc(args)
    if filterfunc is not None:
        print("Applying filter to ARTIS spectrum")
        dfspectrum = dfspectrum.with_columns(pl.Series(timecolname, filterfunc(dfspectrum[timecolname])))

    if args.plotvspecpol:
        # the vpkt configuration is only necessary for the observer angle in the label
        vpkt_config = get_vpkt_config(modelpath)
        linelabel = (
            f"{timeavg} days, cos($\\theta$) = {vpkt_config['cos_theta'][angle // vpkt_config['nspectraperobs']]}"
        )
    else:
        linelabel = f"{timeavg} days"

    fig, axesgrid = make_frame_figure(args)
    axis = axesgrid[0][0]

    # main gives args.xmin and args.xmax in the unit of -xunit, thus the axis takes that same unit
    dfspectrum = dfspectrum.with_columns(
        x=pl.Series(convert_angstroms_to_unit(dfspectrum["lambda_angstroms"].to_numpy(), args.xunit))
    )

    if args.binflux:
        dfbinned = bin_spectrum(dfspectrum, 5, "x", timecolname)
        axis.plot(dfbinned["x"], dfbinned[timecolname])
    else:
        axis.plot(dfspectrum["x"], dfspectrum[timecolname], label=linelabel)

    if args.ymax is None:
        args.ymax = 0.5
    if args.ymin is None:
        args.ymin = -0.5
    if args.ymin >= args.ymax:
        exit_with_error(f"-ymin {args.ymin} must be less than -ymax {args.ymax}")

    axis.set_ylim(args.ymin, args.ymax)
    axis.set_xlim(args.xmin, args.xmax)

    xlabel, _ylabel = get_axis_labels(args)
    axis.set_ylabel(str(args.stokesparam))
    axis.set_xlabel(xlabel)
    figname = f"plotpol_{timeavg}_days_{args.stokesparam.split('/')[0]}_{args.stokesparam.split('/')[1]}.pdf"
    outpath = resolve_outputfile(args.outputfile, figname)
    save_figure(fig, outpath, args=args)


def plot_reference_spectrum(
    filename: Path | str,
    axis: mplax.Axes,
    xmin: float,
    xmax: float,
    fluxfilterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    scale_to_peak: float | None = None,
    offset: float = 0,
    scale_to_dist_mpc: float = 1,
    scaletoreftime: float | None = None,
    xunit: str = "angstrom",
    yvariable: str = "flux",
    residualseries: list[ResidualSeries] | None = None,
    **plotkwargs: t.Any,
) -> tuple[Line2D, str, float]:
    """Plot a single reference spectrum.

    The filename must be in space separated text formatted with the first two
    columns being wavelength in Angstroms, and F_lambda
    """
    filepath = find_reference_spectrum_file(filename)

    metadata = get_file_metadata(filepath)
    label = plotkwargs.get("label", metadata.get("label", str(filename)))
    assert isinstance(label, str)
    plotkwargs.pop("label", None)

    print_heading(f"Reference spectrum '{label}'")
    specdata = get_reference_spectrum(filepath)
    print_detail(f"file: {filepath}")

    # scale to 1 Mpc, and get_dfspectrum_x_y_with_units then scales to scale_to_dist_mpc
    print(f"Scaling to distance {scale_to_dist_mpc} Mpc")
    if "dist_mpc" in metadata:
        filedistance_mpc = float(metadata["dist_mpc"])
        if filedistance_mpc <= 0:
            msg = f"dist_mpc in the metadata of {filepath} must be above zero, not {filedistance_mpc}"
            raise ValueError(msg)
    else:
        # a plain file gives no distance, thus its flux is at the distance of the plot
        print_warning(f"no dist_mpc in the metadata of {filepath}, thus the flux is at {scale_to_dist_mpc} Mpc")
        filedistance_mpc = scale_to_dist_mpc
    specdata = specdata.with_columns(f_lambda=pl.col("f_lambda") * filedistance_mpc**2)

    if scaletoreftime is not None:
        timefactor = timeshift_fluxscale_co56law(scaletoreftime, float(metadata["t"]))
        print_detail(f"scaled from time {metadata['t']} to {scaletoreftime}, factor {timefactor} by the Co56 decay law")
        specdata = specdata.with_columns(f_lambda=pl.col("f_lambda") * timefactor)
        label += f" * {timefactor:.2f}"

    if "scale_factor" in metadata:
        specdata = specdata.with_columns(f_lambda=pl.col("f_lambda") * metadata["scale_factor"])

    if metadata.get("mask_telluric", False):
        print("Masking telluric regions")
        # a spectrum with no redshift in its metadata is in the observed frame, where the telluric bands are
        z = metadata.get("z", 0.0)
        bands = [(1.35e4, 1.44e4), (1.8e4, 1.94e4)]  # [Angstroms]
        bands_rest = [(band_low / (1 + z), band_high / (1 + z)) for band_low, band_high in bands]

        expr_masked = pl.when(
            pl.any_horizontal([
                pl.col("lambda_angstroms").is_between(band_low_rest, band_high_rest, closed="both")
                for band_low_rest, band_high_rest in bands_rest
            ])
        )
        specdata = specdata.with_columns(f_lambda=expr_masked.then(pl.lit(math.nan)).otherwise(pl.col("f_lambda")))

    print_detail(f"points: {len(specdata)}")

    print_detail(
        "metadata: " + ", ".join([f"{k}='{v}'" if hasattr(v, "lower") else f"{k}={v}" for k, v in metadata.items()])
    )

    lambda_min, lambda_max = convert_xlimits_to_lambda_range(xmin, xmax, xunit)

    # the reported flux covers the range that the user asked for, thus it takes the rows inside it
    inrange = specdata.filter(pl.col("lambda_angstroms").is_between(lambda_min, lambda_max))
    print_integrated_flux(inrange["f_lambda"], inrange["lambda_angstroms"])

    # the drawn line keeps the nearest row outside each bound, so that it reaches the edge of the axes
    # instead of stopping at the last point inside the range
    specdata = df_filter_minmax_bracketed(specdata, "lambda_angstroms", lambda_min, lambda_max).collect()

    if fluxfilterfunc:
        print_detail("applying the filter to the reference spectrum")
        specdata = specdata.with_columns(
            cs.starts_with("f_lambda").map_batches(fluxfilterfunc, return_dtype=pl.self_dtype())
        )

    specdata = get_dfspectrum_x_y_with_units(
        specdata, xunit=xunit, yvariable=yvariable, fluxdistance_mpc=scale_to_dist_mpc
    ).collect()

    if scale_to_peak:
        specdata = specdata.with_columns(y=pl.col("y") / pl.col("y").max() * scale_to_peak + offset)
    else:
        assert offset == 0
    ymax = specdata["y"].max()
    assert isinstance(ymax, float)
    (lineplot,) = axis.plot(specdata["x"], specdata["y"], label=label, **plotkwargs)

    if residualseries is not None:
        residualseries.append(
            ResidualSeries(
                label,
                np.asarray(specdata["x"].to_numpy(), dtype=np.float64),
                np.asarray(specdata["y"].to_numpy(), dtype=np.float64),
                lineplot.get_color(),
                isreference=True,
            )
        )

    return lineplot, label, ymax


def plot_reference_spectrum_for_args(
    filename: Path | str,
    axis: mplax.Axes,
    args: argparse.Namespace,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None,
    scale_to_peak: float | None,
    offset: float = 0.0,
    residualseries: list[ResidualSeries] | None = None,
    **plotkwargs: t.Any,
) -> tuple[Line2D, str, float]:
    """Plot a reference spectrum over the x range of the axes, in the units and at the distance that args give."""
    xmin, xmax = axis.get_xlim()
    return plot_reference_spectrum(
        filename=filename,
        axis=axis,
        xmin=xmin,
        xmax=xmax,
        fluxfilterfunc=filterfunc,
        scale_to_peak=scale_to_peak,
        offset=offset,
        scale_to_dist_mpc=args.distmpc,
        scaletoreftime=args.scaletoreftime,
        xunit=args.xunit,
        yvariable=args.yvariable,
        residualseries=residualseries,
        **plotkwargs,
    )


def plot_filter_functions(axis: mplax.Axes, xunit: str) -> None:
    """Plot the UBVI filter transmission curves on the flux axis, at the x values of the unit xunit.

    The transmission goes from 0 to 1, thus the curves agree with a spectrum of --normalised.
    """
    filter_names = ["U", "B", "V", "I"]
    colours = ["r", "b", "g", "c", "m"]

    filterdir = Path(get_path("artistools_dir"), "data/filters/")
    for index, filter_name in enumerate(filter_names):
        filter_data = read_wsv(
            filterdir / f"{filter_name}.txt",
            has_header=False,
            skip_rows=4,
            new_columns=["lambda_angstroms", "flux_normalised"],
        )
        # the files give the wavelength in angstroms, thus the curves take the unit of the x axis
        axis.plot(
            convert_angstroms_to_unit(filter_data["lambda_angstroms"].to_numpy(), xunit),
            filter_data["flux_normalised"],
            label=filter_name,
            color=colours[index],
            alpha=0.3,
        )


DELTALOGX_SCALES: t.Final = ("smallestscale", "largestscale")


def parse_deltalogx_argument(value: str) -> float | str:
    """Return the bin width factor of -deltalogx, or a keyword of DELTALOGX_SCALES."""
    if value.lower() in DELTALOGX_SCALES:
        return value.lower()
    try:
        return float(value)
    except ValueError:
        msg = f"give a number or one of {', '.join(DELTALOGX_SCALES)}, not {value}"
        raise argparse.ArgumentTypeError(msg) from None


def get_spatial_scale_deltalogx(
    runfolders: Sequence[Path], scale: t.Literal["smallest", "largest"]
) -> tuple[float, str, list[str]]:
    """Return the value of -deltalogx for the smallest or the largest spatial scale, a short note, and the method lines.

    In a homologous flow, each spatial scale of the model grid is a constant width Δv in velocity. A feature narrower
    than the Doppler shift across the smallest spatial scale comes from the grid. One bin width applies to all the
    models. The smallest spatial scale of all the models gives bins that lose no detail of the grid. The largest
    spatial scale of all the models gives bins that show no detail that is finer than the grid.
    """
    spatialscales = {runfolder: get_spatial_scales(runfolder) for runfolder in runfolders}
    smallestrun = min(spatialscales, key=lambda runfolder: spatialscales[runfolder][0])
    largestrun = max(spatialscales, key=lambda runfolder: spatialscales[runfolder][1])
    choices = {
        "smallest": (spatialscales[smallestrun][0], smallestrun, "lose no detail of the grid"),
        "largest": (
            spatialscales[largestrun][1],
            largestrun,
            (
                "are at least as wide as each spatial scale of all the models. Thus the bins show no detail that is"
                " finer than the grid"
            ),
        ),
    }
    velocity, run, _ = choices[scale]
    otherscale = "largest" if scale == "smallest" else "smallest"
    note = (
        f"-deltalogx {scale}scale = Δv/c = {velocity / C_cm_per_s:.3g}, where Δv = {velocity / km_to_cm:.0f} km/s is"
        f" the {scale} spatial scale in {get_model_name(run)}. The resolving power is λ/Δλ = c/Δv ="
        f" {C_cm_per_s / velocity:.0f}. -deltalogx {otherscale}scale gives {choices[otherscale][0] / C_cm_per_s:.3g}"
    )
    method = [
        (
            "Δv is a spatial scale of the model grid in velocity. In a homologous flow, Δv stays constant with time. A"
            " spectral feature with Δλ/λ less than the smallest Δv/c comes from the grid and not from the physics."
        ),
        *(
            f"The {name} spatial scale of the models: Δv = {width / km_to_cm:.0f} km/s in"
            f" {resolve_modelpath(widthrun)}. Δv/c = {width / C_cm_per_s:.3g}, and λ/Δλ = c/Δv ="
            f" {C_cm_per_s / width:.0f}. Bins of this width {effect}."
            for name, (width, widthrun, effect) in choices.items()
        ),
        f"-deltalogx {scale}scale selects the {scale} spatial scale.",
        *(
            f"{resolve_modelpath(runfolder)}: {description}."
            for runfolder, (_, _, description) in spatialscales.items()
        ),
        (
            "The Monte Carlo noise of the packets can make the resolution worse than this estimate. The estimate does"
            " not include this noise."
        ),
    ]
    return velocity / C_cm_per_s, note, method


def resolve_deltalogx_scale(args: argparse.Namespace) -> None:
    """Replace a keyword of -deltalogx with the bin width factor of the spatial scale of the models, and log it.

    args.deltalogxnote then holds a short note of the value for the spectrum viewer, or "" for a number.
    """
    args.deltalogxnote = ""
    if not isinstance(args.deltalogx, str):
        return
    runfolders = get_artis_run_folders(args.modelspecpaths)
    if not runfolders:
        exit_with_error(
            f"-deltalogx {args.deltalogx} reads the grid of each ARTIS model, and the command names no model",
            "Give an ARTIS model, or give -deltalogx a number",
        )
    scale: t.Literal["smallest", "largest"] = "smallest" if args.deltalogx == "smallestscale" else "largest"
    args.deltalogx, args.deltalogxnote, method = get_spatial_scale_deltalogx(runfolders, scale)
    print_heading("Bin width from the spatial scales of the models")
    for line in method:
        print_detail(line)


def get_packet_use_time(args: argparse.Namespace) -> t.Literal["escape", "emission", "arrival"]:
    """Return the time of each packet that selects it for the spectrum."""
    if args.use_escapetime:
        return "escape"
    if args.use_emissiontime:
        return "emission"
    return "arrival"


def get_packet_lambda_bin_edges(
    args: argparse.Namespace, xmin: float, xmax: float, modelpath: Path
) -> npt.NDArray[np.floating]:
    """Return the wavelength bin edges of a spectrum from packets, from the bin width that the arguments give."""
    return get_lambda_bin_edges(
        xmin,
        xmax,
        deltax=args.deltax,
        deltalogx=args.deltalogx,
        deltalambda=args.deltalambda,
        xunit=args.xunit,
        modelpath=modelpath,
        gamma=args.gamma,
    )


def get_bin_x_edges(
    dfspectrum: pl.DataFrame, xunit: str
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], str]:
    """Return the lower x edge and the upper x edge of each bin, and the source of the edges for the log.

    A packet spectrum has the column delta_lambda, thus its edges are exact. A different spectrum, e.g. spec.out or a
    spectrum after --binflux, has only the bin centres. Its inner edges are then the midpoints on the x axis between
    adjacent centres. Each outer edge is at the same distance from its centre as the inner edge of that bin.
    """
    if "delta_lambda" in dfspectrum.columns:
        dfbins = dfspectrum.select("lambda_angstroms", "delta_lambda").sort("lambda_angstroms")
        lambda_angstroms = dfbins["lambda_angstroms"].to_numpy()
        halfwidths = 0.5 * dfbins["delta_lambda"].to_numpy()
        # two adjacent bins take one value for their common edge. Edges from each side differed in the last bit, and a
        # readout of the step line then interpolated across a bin
        lambda_edges = np.append(lambda_angstroms - halfwidths, lambda_angstroms[-1] + halfwidths[-1])
        # a frequency or an energy decreases with the wavelength, thus the sort puts the x edges in the order of x
        x_edges = np.sort(convert_angstroms_to_unit(lambda_edges, xunit))
        return x_edges[:-1], x_edges[1:], f"the wavelength edges of the packet bins, in the unit {xunit}"

    x = dfspectrum["x"].to_numpy()
    if len(x) < 2:
        exit_with_error(
            "--histogram needs the width of a bin, and a spectrum with one bin centre gives no width",
            "Give a wider x range",
        )
    midpoints = 0.5 * (x[1:] + x[:-1])
    lower = np.concatenate([[x[0] - (midpoints[0] - x[0])], midpoints])
    upper = np.concatenate([midpoints, [x[-1] + (x[-1] - midpoints[-1])]])
    return (
        lower,
        upper,
        (
            "the midpoints on the x axis between adjacent bin centres, because the spectrum gives only the bin centres."
            " Each outer edge is at the same distance from its centre as the inner edge of that bin"
        ),
    )


def get_histogram_xy(
    lower: npt.NDArray[np.floating], upper: npt.NDArray[np.floating], values: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the points of a line that has the value of each bin from its lower edge to its upper edge."""
    return np.column_stack([lower, upper]).ravel(), np.repeat(values, 2)


def get_noise_method_text(yvariable: str) -> str:
    """Return the log line that tells how the noise band comes from the packets of each bin."""
    noisemethod = (
        "1 / sqrt(N), where N is the count of packets in the bin"
        if yvariable == "packetcount"
        else "sqrt(sum(e^2)) / sum(e), where e is the energy of each packet in the bin"
    )
    return (
        "noise band: the band goes from y * (1 - r) to y * (1 + r). r is the relative noise of the bin, which is the"
        f" estimate of one standard deviation of y divided by y. r = {noisemethod}"
    )


def plot_noise_band(
    axis: mplax.Axes,
    dfspectrum: pl.DataFrame,
    color: "mplt.ColorType",
    binedges: tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]] | None,
) -> None:
    """Draw the band of plus and minus one standard deviation of the Monte Carlo noise around a spectrum.

    The column yrelnoise of get_dfspectrum_x_y_with_units gives the relative noise r of y, and the band goes from
    y * (1 - r) to y * (1 + r). A bin with no packets has y = 0 and no estimate of r, thus its band has zero width. If
    the caller gives binedges, the band has the same steps as the line of --histogram.
    """
    dfband = dfspectrum.select("x", "y", relnoise=pl.col("yrelnoise").fill_null(np.nan))
    y = dfband["y"].to_numpy()
    relnoisevalues = dfband["relnoise"].to_numpy()
    hasnoise = ~np.isnan(relnoisevalues)
    # a gap in the band at an empty bin would also remove the band of a bin with packets between two empty bins
    ynoise = np.where(hasnoise, y * np.nan_to_num(relnoisevalues), 0.0)
    x = dfband["x"].to_numpy()
    if binedges is not None:
        x, y = get_histogram_xy(*binedges, y)
        ynoise = np.repeat(ynoise, 2)
    axis.fill_between(x, y - ynoise, y + ynoise, color=color, alpha=0.25, linewidth=0)

    if not hasnoise.any():
        print_warning("each drawn bin has y = 0, thus the plot shows no noise band")
        return
    percentnoise = 100.0 * relnoisevalues[hasnoise]
    print_detail(
        f"noise band: the relative noise r of the drawn bins is {percentnoise.min():.3g}% to"
        f" {percentnoise.max():.3g}%, and the median is {np.median(percentnoise):.3g}%. {(~hasnoise).sum()} of"
        f" {len(hasnoise)} bins have y = 0. These bins have no estimate of r, thus their band has zero width"
    )


def model_covers_timedays(modelpath: Path | str, timedays: str | float | None, *, quiet: bool = False) -> bool:
    """Return False if timedays names one time outside the timesteps of the model, and print a warning unless quiet.

    A -timemin and -timemax range outside a model skips that model with a warning, thus one time does the same. A
    range of -timedays and a text that is no number give True, because get_time_range gives their messages.
    """
    if timedays is None or parse_timedays_range(timedays) is not None:
        return True
    try:
        timedays_float = float(str(timedays).rstrip("d"))
    except ValueError:
        return True

    tstart_first = get_timestep_times(modelpath, loc="start")[0]
    tend_last = get_timestep_times(modelpath, loc="end")[-1]
    if tstart_first <= timedays_float <= tend_last:
        return True

    if not quiet:
        print_warning(
            f"{get_model_logname(modelpath)}: no timestep covers {timedays_float:g} days, because the timesteps cover"
            f" {tstart_first:.2f} to {tend_last:.2f} days. The plot skips this model at this time"
        )
    return False


def plot_artis_spectrum(
    axes: npt.NDArray[np.object_] | Sequence[mplax.Axes],
    modelpath: Path | str,
    args: argparse.Namespace,
    scale_to_peak: float | None = None,
    from_packets: bool = False,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    linelabel: str | None = None,
    yvariable: str = "flux",
    directionbins: list[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    usedegrees: bool = False,
    maxpacketfiles: int | None = None,
    xunit: str = "angstrom",
    residualseries: list[ResidualSeries] | None = None,
    **plotkwargs: t.Any,
) -> pl.DataFrame | None:
    """Plot an ARTIS output spectrum. The data plotted are also returned as a DataFrame."""
    modelpath = Path(modelpath)
    if path_is_file(modelpath):  # handle e.g. modelpath = 'modelpath/spec.out'
        print_warning(f"ignoring filename of {modelpath.name}")
        modelpath = modelpath.parent

    if not path_is_dir(modelpath):
        print_warning(f"Skipping because {modelpath} does not exist")
        return None

    # --write_data names one column for each drawn series, thus the loops below collect every one of
    # them. The suffix names the direction bin and the epoch of the panel
    drawnseries: list[tuple[str, pl.DataFrame]] = []
    # each direction bin and each panel draws a series, thus print_method_line prints each method line one time only
    printedmethodlines: set[str] = set()

    def print_method_line(line: str) -> None:
        if line not in printedmethodlines:
            printedmethodlines.add(line)
            print_detail(line)

    use_time = get_packet_use_time(args)
    assert from_packets or use_time == "arrival"

    if directionbins is None:
        directionbins = [-1]

    clamp_to_timesteps = not args.notimeclamp
    nprocs_read_dfpackets: tuple[int, pl.DataFrame] | None = None
    # every panel reads the packets, thus one read of the union of the time windows serves them all. Each
    # panel then applies its own arrival window to the frame in memory. The packets of a remote model stay on
    # its host, thus each panel asks the host for its own spectrum
    sharesreadofpackets = (
        from_packets
        and args.multispecplot
        and use_time == "arrival"
        and args.plotvspecpol is None
        and not is_remote_path(modelpath)
    )
    # the panel loop skips an epoch that the model does not hold, thus such an epoch needs no packets
    timeranges: list[tuple[int, int, float, float]] = (
        [
            get_time_range(modelpath, timedays_range_str=timedays, clamp_to_timesteps=clamp_to_timesteps)
            for timedays in args.timedayslist
            if model_covers_timedays(modelpath, timedays, quiet=True)
        ]
        if sharesreadofpackets
        else []
    )
    if timeranges:
        nprocs_read, dfpackets = get_packets(
            modelpath,
            maxpacketfiles=maxpacketfiles,
            packet_type="TYPE_ESCAPE",
            escape_type="TYPE_GAMMA" if args.gamma else "TYPE_RPKT",
        )
        timelow = min(timerange[2] for timerange in timeranges)
        timehigh = max(timerange[3] for timerange in timeranges)
        # get_from_packets reads only the time, the frequency, the energy, and the direction bin of each packet. The
        # packets files hold about 40 columns, thus a frame of every column needs about ten times the memory
        nprocs_read_dfpackets = (
            nprocs_read,
            dfpackets
            .filter(pl.col("t_arrive_d").is_between(timelow, timehigh))
            .select(cs.by_name("t_arrive_d", "nu_rf", "e_rf", "dirbin", "costhetabin", "phibin", require_all=False))
            .collect(),
        )

    for axindex, axis in enumerate(axes):
        assert isinstance(axis, mplax.Axes)
        # label_dirbin_series clears the colour of each direction bin after the first one, thus each panel takes a
        # copy. The later panels then keep the -color of the first bin, as the legend of the first panel shows
        panelplotkwargs = dict(plotkwargs)
        # locals, not a write-back onto args: this function runs once for each model and once for
        # each axis. A range that one of them resolved would reach the next one, and get_time_range
        # then drops a model whose last timestep ends before that range
        if args.multispecplot:
            if not model_covers_timedays(modelpath, args.timedayslist[axindex]):
                continue
            (timestepmin, timestepmax, timemin, timemax) = get_time_range(
                modelpath, timedays_range_str=args.timedayslist[axindex], clamp_to_timesteps=clamp_to_timesteps
            )
        else:
            if not model_covers_timedays(modelpath, args.timedays):
                return None
            (timestepmin, timestepmax, timemin, timemax) = get_time_range(
                modelpath,
                args.timestep,
                args.timemin,
                args.timemax,
                args.timedays,
                clamp_to_timesteps=clamp_to_timesteps,
            )

        if timestepmin == timestepmax == -1:
            return None

        assert timemin is not None
        assert timemax is not None
        timeavg = (timemin + timemax) / 2.0
        timedelta = (timemax - timemin) / 2
        linelabel_is_custom = linelabel is not None
        if linelabel is None:
            modelname = get_model_name(modelpath)
            linelabel = modelname if len(modelname) < 70 else f"...{modelname[-67:]}"

            if not args.hidemodeltime and not args.multispecplot:
                # TODO: fix this for multispecplot - use args.showtime for now
                linelabel += f" +{timeavg:.1f}d"
            if not args.hidemodeltimerange and not args.multispecplot and timedelta >= 0.1:
                linelabel += rf" ($\pm$ {timedelta:.1f}d)"

        # the label carries LaTeX for the figure, thus the log line shows the plain form
        print_heading(
            f"'{plain_label(linelabel)}': "
            + get_time_range_text(timestepmin, timestepmax, timemin, timemax, clamped=clamp_to_timesteps)
        )
        print_detail(f"modelpath: {modelpath}")

        check_time_range_is_valid(modelpath, timemin, timemax, args.plotinvalidpart)

        xmin, xmax = axis.get_xlim()
        if from_packets:
            lambda_bin_edges = get_packet_lambda_bin_edges(args, xmin, xmax, modelpath)

            viewinganglespectra = get_from_packets(
                modelpath,
                timelowdays=timemin,
                timehighdays=timemax,
                lambda_bin_edges=lambda_bin_edges,
                use_time=use_time,
                maxpacketfiles=maxpacketfiles,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                fluxfilterfunc=filterfunc,
                nprocs_read_dfpackets=nprocs_read_dfpackets,
                directionbins_are_vpkt_observers=args.plotvspecpol is not None,
                gamma=args.gamma,
                # the packets of each requested bin take a pass of the Rust kernel, thus the plot asks only for its bins
                directionbins=directionbins,
                # the squared energies of the packets give the noise, and only --shownoise needs them
                relnoise=args.shownoise,
            )

        elif args.plotvspecpol is not None:
            # read virtual packet files (after running plotartisspectrum --makevspecpol)
            viewinganglespectra = {
                dirbin: get_vspecpol_spectrum(
                    modelpath, timeavg, dirbin, args, fluxfilterfunc=filterfunc, timemin=timemin, timemax=timemax
                )
                for dirbin in directionbins
                if dirbin >= 0
            }
        else:
            viewinganglespectra = get_spectra(
                modelpath=modelpath,
                timestepmin=timestepmin,
                timestepmax=timestepmax,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                fluxfilterfunc=filterfunc,
                gamma=args.gamma,
                # bin -1 stays, because the plot shows it in place of a bin that the run does not have
                directionbins=[-1, *directionbins],
            )

        if args.plotvspecpol is None and (average_over_phi or average_over_theta):
            # an averaged bin is the first bin of its group, thus a different bin has no label and no packets
            validbins = get_dirbins(average_over_phi=average_over_phi, average_over_theta=average_over_theta)
            if invalidbins := [dirbin for dirbin in directionbins if dirbin >= 0 and dirbin not in validbins]:
                msg = f"Direction bin {invalidbins} is not the first bin of an average group. Valid bins: {validbins}"
                raise ValueError(msg)

        dirbin_definitions = get_dirbin_definitions(
            modelpath,
            directionbins,
            vpkt_observers=bool(args.plotvspecpol),
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
            usedegrees=usedegrees,
        )

        missingdirectionbins = [dirbin for dirbin in directionbins if dirbin not in viewinganglespectra]
        founddirectionbins = [dirbin for dirbin in directionbins if dirbin in viewinganglespectra]
        if missingdirectionbins:
            print(f"No data for direction bin(s): {missingdirectionbins}")
            if founddirectionbins:
                directionbins = founddirectionbins
            elif -1 in viewinganglespectra:
                directionbins = [-1]
                dirbin_definitions = get_dirbin_definitions(modelpath, directionbins, usedegrees=usedegrees)
                print("Showing spherically-averaged spectrum instead")
            else:
                print("No data to plot")
                return None

        if any(dirbin != -1 for dirbin in directionbins):
            print_theta_phi_definitions()

        dirbin_dfspec = zip(
            directionbins,
            pl.collect_all([
                df_filter_minmax_bracketed(
                    get_dfspectrum_x_y_with_units(
                        viewinganglespectra[dirbin], xunit=xunit, yvariable=yvariable, fluxdistance_mpc=args.distmpc
                    ),
                    colname="x",
                    minval=xmin,
                    maxval=xmax,
                )
                for dirbin in directionbins
            ]),
            strict=True,
        )
        for dirbin, dfspectrum_dirbin in dirbin_dfspec:
            dfspectrum = dfspectrum_dirbin
            print_dirbin_summary(dirbin, dirbin_definitions[dirbin], dfspectrum)
            linelabel_withdirbin = label_dirbin_series(
                dirbin, directionbins, dirbin_definitions, linelabel, linelabel_is_custom, panelplotkwargs
            )

            print_integrated_flux(dfspectrum["dflux_on_dx_onempc"], dfspectrum["x"])

            if scale_to_peak:
                dfspectrum = dfspectrum.with_columns(y=pl.col("y") / pl.col("y").max() * scale_to_peak)

            if args.binflux:
                if args.xunit.lower() != "angstrom":
                    exit_with_error(
                        f"--binflux averages over wavelength, and -xunit gives {args.xunit}", "Give -xunit angstrom"
                    )
                # bin f_lambda as well, because --write_data returns that column. The earlier
                # code gave it the value of y, which holds the selected y variable
                dfspectrum = (
                    bin_spectrum(dfspectrum, 5, "lambda_angstroms", ["y", "f_lambda"])
                    .rename({"lambda_angstroms": "x"})
                    .with_columns(lambda_angstroms=pl.col("x"))
                )

            seriessuffix = f"_dirbin{dirbin:02d}" if dirbin >= 0 else ""
            if args.multispecplot:
                seriessuffix += f"_{args.timedayslist[axindex]}d"
            drawnseries.append((seriessuffix, dfspectrum))

            xplot, yplot = dfspectrum["x"].to_numpy(), dfspectrum["y"].to_numpy()
            binedges = None
            if args.histogram:
                lower, upper, edgesource = get_bin_x_edges(dfspectrum, xunit)
                binedges = (lower, upper)
                print_method_line(
                    "histogram: each bin is a horizontal line from its lower edge to its upper edge."
                    f" The edges are {edgesource}"
                )
                xplot, yplot = get_histogram_xy(lower, upper, yplot)
            (modelline,) = axis.plot(
                xplot, yplot, label=linelabel_withdirbin if axindex == 0 else None, **panelplotkwargs
            )
            if args.shownoise:
                print_method_line(get_noise_method_text(yvariable))
                plot_noise_band(axis, dfspectrum, modelline.get_color(), binedges)
            if residualseries is not None and axindex == 0:
                residualseries.append(
                    ResidualSeries(
                        linelabel_withdirbin or "",
                        np.asarray(dfspectrum["x"].to_numpy(), dtype=np.float64),
                        np.asarray(dfspectrum["y"].to_numpy(), dtype=np.float64),
                        modelline.get_color(),
                        isreference=False,
                    )
                )

    if not drawnseries:
        return None

    if not args.write_data:
        # the caller counts a frame that is not None as a drawn series, and --write_data alone
        # reads the columns. One table needs one wavelength grid, which a plot does not need
        return pl.DataFrame()

    # f_lambda is per Angstrom at 1 Mpc. The plotted columns give the values of the plot, thus they follow
    # -distmpc, --normalised, -xunit, and -yvariable
    dfseriesdata = pl.DataFrame({
        "lambda_angstroms": drawnseries[0][1]["lambda_angstroms"],
        f"x_plotted_{args.xunit}": drawnseries[0][1]["x"],
    })
    for seriessuffix, dfspectrum in drawnseries:
        if not np.allclose(dfseriesdata["lambda_angstroms"], dfspectrum["lambda_angstroms"].to_numpy()):
            exit_with_error(
                "--write_data gives one table, and the drawn series have different wavelength grids",
                "Remove --write_data, or give one direction bin and one epoch",
            )
        dfseriesdata = dfseriesdata.with_columns(
            dfspectrum["f_lambda"].alias(f"f_lambda{seriessuffix}"),
            dfspectrum["y"].alias(f"{args.yvariable}_plotted{seriessuffix}"),
        )

    return dfseriesdata


def check_codecomparison_args(args: argparse.Namespace) -> None:
    """Stop if the plot of a code comparison spectrum cannot follow the arguments.

    The code comparison reader draws F_lambda per angstrom at 1 Mpc against the wavelength in angstroms, at the
    time nearest to the middle of the time range.
    """
    if args.timemin is None or args.timemax is None:
        exit_with_error(
            "a code comparison spectrum needs a time, and the command gives no time range",
            "Give -timedays (e.g. -t 300 or -t 290-320), or -timemin and -timemax",
        )
    unsupported = [
        option
        for option, given in (
            (f"-xunit {args.xunit}", args.xunit != "angstrom"),
            (f"-yvariable {args.yvariable}", args.yvariable != "flux"),
            (f"-distmpc {args.distmpc}", args.distmpc != 1.0),
            ("--normalised", args.normalised),
        )
        if given
    ]
    if unsupported:
        exit_with_error(
            "a code comparison spectrum is F_lambda per angstrom at 1 Mpc, and the command gives"
            f" {' and '.join(unsupported)}",
            f"Remove {' and '.join(unsupported)}, or remove the code comparison path",
        )


def make_spectrum_plot(
    speclist: Sequence[Path | str],
    axes: npt.NDArray[np.object_] | Sequence[mplax.Axes],
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None,
    args: argparse.Namespace,
    scale_to_peak: float | None = None,
    residualseries: list[ResidualSeries] | None = None,
) -> pl.DataFrame:
    """Plot reference spectra and ARTIS spectra.

    residualseries takes each drawn series for a residual panel.
    """
    dfalldata = pl.DataFrame()
    nseriesplotted = 0

    set_prop_cycle_unusedcolors(axes, args.color)
    for axis in axes:
        # matplotlib adds the y margin in the scale of the axis, thus a log axis also keeps room above the peak
        axis.margins(x=0.0, y=YMARGIN)

    for seriesindex, specpath in enumerate(speclist):
        plotkwargs: dict[str, t.Any] = {
            "alpha": args.linealpha[seriesindex],
            "linestyle": args.linestyle[seriesindex],
            "color": args.color[seriesindex],
        }

        if args.dashes[seriesindex]:
            plotkwargs["dashes"] = args.dashes[seriesindex]
        if args.linewidth[seriesindex]:
            plotkwargs["linewidth"] = args.linewidth[seriesindex]
        seriesname = "UNKNOWN"

        # only an ARTIS spectrum produces writable series data. Reset it every iteration, so that a reference
        # spectrum neither reads an unset variable nor re-writes the previous model's spectrum under its own name
        seriesdata: pl.DataFrame | None = None
        if path_is_reference_spectrum(specpath):
            # reference spectrum
            if "linewidth" not in plotkwargs:
                plotkwargs["linewidth"] = 1.1

            if args.multispecplot:
                # each panel shows one epoch of the models, and the reference spectrum goes on all of them
                plotkwargs["color"] = "k"
            if args.label[seriesindex]:
                plotkwargs["label"] = args.label[seriesindex]
            for axis in axes:
                plot_reference_spectrum_for_args(
                    specpath, axis, args, filterfunc, scale_to_peak, residualseries=residualseries, **plotkwargs
                )
            nseriesplotted += 1
        elif path_is_codecomparison(specpath):
            check_codecomparison_args(args)
            # resolve_plot_args gives both bounds. -timedays can hold a range such as 290-320, which is no number
            timeavg = (args.timemin + args.timemax) / 2.0
            from artistools.codecomparison import plot_spectrum

            plot_spectrum(specpath, timedays=timeavg, axis=axes[0], **plotkwargs)
            if residualseries is not None:
                print_warning("the residual panel does not include the code comparison series")
            nseriesplotted += 1
        else:
            # ARTIS model spectrum
            if "linewidth" not in plotkwargs:
                plotkwargs["linewidth"] = 1.3

            plotkwargs["linelabel"] = args.label[seriesindex]

            try:
                seriesdata = plot_artis_spectrum(
                    axes,
                    specpath,
                    args=args,
                    scale_to_peak=scale_to_peak,
                    from_packets=args.frompackets,
                    maxpacketfiles=args.maxpacketfiles,
                    filterfunc=filterfunc,
                    yvariable=args.yvariable,
                    directionbins=args.plotvspecpol or args.plotviewingangle,
                    average_over_phi=args.average_over_phi_angle,
                    average_over_theta=args.average_over_theta_angle,
                    usedegrees=args.usedegrees,
                    xunit=args.xunit,
                    residualseries=residualseries,
                    **plotkwargs,
                )
            except FileNotFoundError as e:
                # the folder can exist and still lack a file that the plot needs, thus the message names the file
                print_warning(f"Skipping {specpath}, because a file that the plot needs does not exist: {e}")
                continue

            if seriesdata is not None:
                seriesname = get_model_name(specpath)
                nseriesplotted += 1

        if args.write_data and seriesdata is not None:
            if dfalldata.is_empty():
                dfalldata = pl.DataFrame({"lambda_angstroms": seriesdata["lambda_angstroms"]})
            else:
                # one table holds one wavelength column, thus each model must give the same wavelengths
                lambda_first = dfalldata["lambda_angstroms"].to_numpy()
                lambda_this = seriesdata["lambda_angstroms"].to_numpy()
                if lambda_first.shape != lambda_this.shape or not np.allclose(lambda_first, lambda_this):
                    exit_with_error(
                        f"--write_data gives one table, and {get_model_name(specpath)} has a different wavelength grid"
                        " from the model before it",
                        "Remove --write_data, or give models with the same frequency bins",
                    )
            # the plotted x column is the same for every model, thus the table holds it once
            xcolumn = f"x_plotted_{args.xunit}"
            if xcolumn not in dfalldata.columns:
                dfalldata = dfalldata.with_columns(seriesdata[xcolumn])

            # one column for each direction bin and each panel, e.g. f_lambda.mymodel_dirbin05_300d
            yplotted = f"{args.yvariable}_plotted"
            dfalldata = dfalldata.with_columns(
                seriesdata[colname].alias(
                    f"f_lambda.{seriesname}{colname.removeprefix('f_lambda')}"
                    if colname.startswith("f_lambda")
                    else f"{yplotted}.{seriesname}{colname.removeprefix(yplotted)}"
                )
                for colname in seriesdata.columns
                if colname not in {"lambda_angstroms", xcolumn}
            )

    if nseriesplotted == 0:
        exit_with_error(
            "no spectra were plotted. Check that each given path holds an ARTIS run or a reference spectrum"
        )

    for axis in axes:
        if args.showfilterfunctions:
            if not args.normalised:
                print_warning("the filter functions plot normalised values, thus give --normalised as well")
            plot_filter_functions(axis, args.xunit)

        # a flux of stokes I is not negative, and the y margin puts the bottom below zero. make_plot applies -ymin
        # and -ymax after this function returns, and -ymax alone keeps this bottom
        if args.stokesparam == "I" and not args.logscaley and args.ymin is None:
            axis.set_ylim(bottom=0.0)

        set_plot_title(axis, args.title, args)

    return dfalldata


def get_xy_spectrum(
    flambda_array: npt.NDArray[np.floating], arraylambda_angstroms: npt.NDArray[np.floating], args: argparse.Namespace
) -> pl.LazyFrame:
    """Return the x series and the y series of one flux array, in the units that the arguments name."""
    return get_dfspectrum_x_y_with_units(
        pl.DataFrame({"f_lambda": flambda_array, "lambda_angstroms": arraylambda_angstroms}),
        xunit=args.xunit,
        yvariable=args.yvariable,
        fluxdistance_mpc=args.distmpc,
    )


def get_emission_contributions(
    modelpath: Path,
    args: argparse.Namespace,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None,
    xmin: float,
    xmax: float,
    timestepmin: int,
    timestepmax: int,
    timemin: float,
    timemax: float,
    dirbin: int | None,
) -> tuple[list[FluxContributionTuple], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the flux contribution of each series, the total emitted flux, and the wavelength grid.

    A run with --frompackets reads the packets files. A run without it reads the emission file and the
    absorption file that ARTIS writes for each timestep.
    """
    if not args.frompackets:
        assert not args.vpkt_match_emission_exclusion_to_opac
        # the emission files of exspec hold no sampled emission, thus resolve_frompackets selects the packets for it
        assert not args.use_sampledemissiontype
        lambda_min, lambda_max = convert_xlimits_to_lambda_range(xmin, xmax, args.xunit)

        return get_flux_contributions(
            modelpath,
            timestepmin,
            timestepmax,
            filterfunc=filterfunc,
            getemission=args.showemission,
            getabsorption=args.showabsorption,
            use_lastemissiontype=not args.use_thermalemissiontype,
            directionbin=dirbin,
            average_over_phi=args.average_over_phi_angle,
            average_over_theta=args.average_over_theta_angle,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            maxseriescount=args.maxseriescount,
            fixedionlist=args.fixedionlist,
        )

    use_time = get_packet_use_time(args)
    lambda_bin_edges = get_packet_lambda_bin_edges(args, xmin, xmax, modelpath)

    return get_flux_contributions_from_packets(
        modelpath,
        timelowdays=timemin,
        timehighdays=timemax,
        lambda_bin_edges=lambda_bin_edges,
        getemission=args.showemission,
        getabsorption=args.showabsorption,
        maxpacketfiles=args.maxpacketfiles,
        filterfunc=filterfunc,
        groupby=args.groupby,
        use_time=use_time,
        fixedionlist=args.fixedionlist,
        maxseriescount=args.maxseriescount + 20,
        gamma=args.gamma,
        emissionevent=get_emission_event(args),
        directionbin=dirbin,
        average_over_phi=args.average_over_phi_angle,
        average_over_theta=args.average_over_theta_angle,
        directionbins_are_vpkt_observers=args.plotvspecpol is not None,
        vpkt_match_emission_exclusion_to_opac=args.vpkt_match_emission_exclusion_to_opac,
        shelledges=args.shelledges,
        shellunit=args.shellunit,
        velocityranges=args.velocityranges_kmps,
    )


def order_and_color_shells(
    contributions: list[FluxContributionTuple],
    arraylambda_angstroms: npt.NDArray[np.floating],
    args: argparse.Namespace,
) -> list[FluxContributionTuple]:
    """Return the shells from the lowest edge to the highest one, with the colours of a sequential map.

    The order of the ions is the order of the flux. A reader expects the shells in the order of their
    edges, and a colour that goes from dark to light with the edge. -maxseriescount still applies: the
    shells with the least flux join the "Other" series, as the ions do.
    """
    import matplotlib.pyplot as plt

    fixedionlist = args.fixedionlist
    if fixedionlist is None:
        # a shell that holds no packet gives no series, and the name of such a shell gives a warning. The packet
        # reducer can already hold an Other series, which must not take the place of a shell. The NOT SET
        # series of the packets with no thermal emission record counts against the limit as a shell does
        named = [contribution for contribution in contributions if contribution.linelabel != "Other"]
        keptlabels = {
            contribution.linelabel
            for contribution in sorted(named, key=lambda c: -c.fluxcontrib)[: args.maxseriescount]
        }
        fixedionlist = [
            label for label in (*get_shell_labels(args.shelledges, args.shellunit), "NOT SET") if label in keptlabels
        ]

    contributions_sorted_reduced = sort_and_reduce_flux_contribution_list(
        contributions, args.maxseriescount, arraylambda_angstroms, fixedionlist=fixedionlist, hideother=args.hideother
    )

    shells = [
        contribution
        for contribution in contributions_sorted_reduced
        if contribution.linelabel not in {"Other", "NOT SET"}
    ]
    colormap = plt.get_cmap("viridis")
    shellcolors: dict[str, mplt.ColorType] = {
        contribution.linelabel: colormap(index / max(len(shells) - 1, 1)) for index, contribution in enumerate(shells)
    }
    shellcolors["NOT SET"] = "lightgrey"

    return [
        contribution._replace(color=shellcolors.get(contribution.linelabel, contribution.color))
        for contribution in contributions_sorted_reduced
    ]


def collect_emission_and_absorption(
    contributions: "Sequence[FluxContributionTuple]",
    arraylambda_angstroms: "npt.NDArray[np.floating]",
    args: argparse.Namespace,
) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    """Return the emission spectra and the absorption spectra of every contribution.

    One call runs every query together. A collect for each contribution runs them one after the other,
    which costs about four times as much for the default series count. The emission queries and the
    absorption queries are independent, thus one call takes both sets.
    """
    emissionqueries = (
        [
            get_xy_spectrum(contribution.array_flambda_emission, arraylambda_angstroms, args)
            for contribution in contributions
        ]
        if args.showemission
        else []
    )
    absorptionqueries = (
        [
            get_xy_spectrum(contribution.array_flambda_absorption, arraylambda_angstroms, args)
            for contribution in contributions
        ]
        if args.showabsorption
        else []
    )
    collected = pl.collect_all([*emissionqueries, *absorptionqueries])

    return collected[: len(emissionqueries)], collected[len(emissionqueries) :]


def plot_contributions_unstacked(
    axis: mplax.Axes,
    contributions: Sequence[FluxContributionTuple],
    arraylambda_angstroms: npt.NDArray[np.floating],
    args: argparse.Namespace,
    scalefactor: float,
    xmin: float,
    xmax: float,
) -> tuple[list[Artist], float]:
    """Draw one line for each contribution, and return the artists and the largest absorption.

    An absorption series goes below the axis, thus the caller reads the largest value to set the limit.
    """
    plotobjects: list[Artist] = []
    max_absorption = 0.0

    emissionspectra, absorptionspectra = collect_emission_and_absorption(contributions, arraylambda_angstroms, args)

    for index, contribution in enumerate(contributions):
        if args.showemission:
            dfspec = emissionspectra[index]
            (emissioncomponentplot,) = axis.plot(
                dfspec["x"], dfspec["y"] * scalefactor, linewidth=1, color=contribution.color
            )
            linecolor = emissioncomponentplot.get_color()
        else:
            linecolor = contribution.color

        if args.showabsorption:
            dfspec = absorptionspectra[index]
            (absorptioncomponentplot,) = axis.plot(
                dfspec["x"], -dfspec["y"] * scalefactor, color=linecolor, linewidth=1
            )
            if not args.showemission:
                linecolor = absorptioncomponentplot.get_color()

            # the drawn line reaches the nearest bin beyond each edge of the x range, thus the maximum reads those
            # bins too
            this_max_absorption = df_filter_minmax_bracketed(dfspec, "x", xmin, xmax).collect()["y"].max()
            if isinstance(this_max_absorption, float):
                max_absorption = max(max_absorption, this_max_absorption)

        plotobjects.append(mpatches.Patch(color=linecolor))

    return plotobjects, max_absorption


def plot_contributions_stacked(
    axis: mplax.Axes,
    contributions: Sequence[FluxContributionTuple],
    arraylambda_angstroms: npt.NDArray[np.floating],
    args: argparse.Namespace,
    scalefactor: float,
    xmin: float,
    xmax: float,
) -> tuple[list[Artist], float]:
    """Draw the contributions as one filled stack, and return the artists and the largest absorption."""
    plotobjects: list[Artist] = []
    max_absorption = 0.0

    contribcolors = [contribution.color for contribution in contributions]
    # if any contribution has no colour set, let matplotlib assign the whole stack from the Axes property cycle
    stackcolors: list[mplt.ColorType] | None = (
        None if any(c is None for c in contribcolors) else [c for c in contribcolors if c is not None]
    )

    # the collect comes before either stackplot, thus the draw order stays the same
    dfemissionspectra, dfabsorptionspectra = collect_emission_and_absorption(contributions, arraylambda_angstroms, args)

    facecolors: list[mplt.ColorType] | None
    if args.showemission:
        stackplot = axis.stackplot(
            dfemissionspectra[0]["x"],
            [dfspec["y"] * scalefactor for dfspec in dfemissionspectra],
            colors=stackcolors,
            linewidth=0,
        )
        plotobjects.extend(stackplot)
        # read back the drawn colours, which matplotlib assigned when stackcolors was None
        facecolors = [mplcolors.to_rgba(np.asarray(p.get_facecolor())[0]) for p in stackplot]
    else:
        facecolors = stackcolors

    if args.showabsorption:
        absstackplot = axis.stackplot(
            dfabsorptionspectra[0]["x"],
            [-dfspec["y"] * scalefactor for dfspec in dfabsorptionspectra],
            colors=facecolors,
            linewidth=0,
        )
        if not args.showemission:
            plotobjects.extend(absstackplot)

        max_absorption = (
            pl
            .DataFrame({
                f"y{i}": df_filter_minmax_bracketed(df, "x", xmin, xmax).collect().get_column("y")
                for i, df in enumerate(dfabsorptionspectra)
            })
            .select(pl.sum_horizontal(pl.all()).max())
            .item()
        )

    return plotobjects, max_absorption


def plot_reference_spectra(
    axis: mplax.Axes,
    args: argparse.Namespace,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None,
    scale_to_peak: float | None,
) -> tuple[list[Artist], list[str], float]:
    """Draw each reference spectrum of -specpath, and return the artists, the labels, and the maximum."""
    plotobjects: list[Artist] = []
    plotobjectlabels: list[str] = []
    ymaxrefall = 0.0
    plotkwargs: dict[str, t.Any] = {}

    for index, filepath in enumerate(args.specpath):
        # reference data can carry the .out suffix of ARTIS, thus the reference predicate decides.
        # A name that is neither falls through, and plot_reference_spectrum names the missing file
        if path_is_artis_model(filepath) and not path_is_reference_spectrum(filepath):
            continue

        if index < len(args.color):
            plotkwargs["color"] = args.color[index]
            # the loop shares one dict, thus a spectrum with no -label must not keep the label of
            # the spectrum before it. Its own metadata names it instead
            plotkwargs.pop("label", None)
            if args.label[index] is not None:
                plotkwargs["label"] = args.label[index]
            plotkwargs["alpha"] = args.linealpha[index]
            plotkwargs["linestyle"] = args.linestyle[index]
            for stylename in ("linewidth", "dashes"):
                plotkwargs.pop(stylename, None)
                if value := getattr(args, stylename)[index]:
                    plotkwargs[stylename] = value

        plotobj, serieslabel, ymaxref = plot_reference_spectrum_for_args(
            filepath, axis, args, filterfunc, scale_to_peak, offset=0.3 if scale_to_peak else 0.0, **plotkwargs
        )
        ymaxrefall = max(ymaxrefall, ymaxref)

        plotobjects.append(plotobj)
        plotobjectlabels.append(serieslabel)

    return plotobjects, plotobjectlabels, ymaxrefall


def get_emission_plot_label(
    modelpath: Path, args: argparse.Namespace, modelname: str, dirbin: int | None, timemin: float, timemax: float
) -> str:
    """Return the title of the plot, which names the model and each selection of the packets.

    The selections are the time range, the velocity ranges, and the direction bin.
    """
    if args.title:
        return str(args.title)

    plotlabel = f"{modelname} [{timemin:.2f}d to {timemax:.2f}d]"
    if args.velocityrangelabels:
        plotlabel += f", packets at {' and '.join(args.velocityrangelabels)}"
    if not (args.plotviewingangle or args.plotvspecpol):
        return plotlabel

    assert dirbin is not None
    # a label of the bin -1, the average over all the directions, comes only for a list of bins that names it
    dirbin_definitions = get_dirbin_definitions(
        modelpath,
        [dirbin],
        vpkt_observers=bool(args.plotvspecpol),
        average_over_phi=args.average_over_phi_angle,
        average_over_theta=args.average_over_theta_angle,
        usedegrees=args.usedegrees,
    )
    plotlabel += f", {dirbin_definitions[dirbin]}"

    if dirbin != -1:
        print_theta_phi_definitions()

    return plotlabel


def make_emissionabsorption_plot(
    modelpath: Path,
    axis: mplax.Axes,
    args: argparse.Namespace,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    scale_to_peak: float | None = None,
) -> tuple[list[Artist], list[str], pl.DataFrame]:
    """Plot the emission and absorption contribution spectra, grouped by ion/line/term for an ARTIS model."""
    modelname = get_series_label(args.label, 0, get_model_name(modelpath))

    print_heading(get_model_logname(modelpath, modelname))
    clamp_to_timesteps = not args.notimeclamp

    # locals, not a write-back onto args, for the reason given in plot_artis_spectrum
    (timestepmin, timestepmax, timemin, timemax) = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays, clamp_to_timesteps=clamp_to_timesteps
    )

    if timestepmin == timestepmax == -1:
        print(f"Can't plot {get_model_logname(modelpath, modelname)}...skipping")
        return [], [], pl.DataFrame()

    check_time_range_is_valid(modelpath, timemin, timemax, args.plotinvalidpart)

    assert timemin is not None
    assert timemax is not None

    print(
        f"'{modelname}': {get_time_range_text(timestepmin, timestepmax, timemin, timemax, clamped=clamp_to_timesteps)}"
    )

    xmin, xmax = axis.get_xlim()

    dirbin = args.plotviewingangle[0] if args.plotviewingangle else args.plotvspecpol[0] if args.plotvspecpol else None

    contribution_list, array_flambda_emission_total, arraylambda_angstroms = get_emission_contributions(
        modelpath, args, filterfunc, xmin, xmax, timestepmin, timestepmax, timemin, timemax, dirbin
    )
    if arraylambda_angstroms.size == 0:
        # every sum and every maximum below reads the bins of the x range. The limits use the unit of the x axis,
        # thus the test reads the bins that get_emission_contributions kept, which are in Angstroms
        exit_with_error(
            f"the x range {xmin:g} to {xmax:g} holds no bin of the spectrum", "Give a wider range with -xmin and -xmax"
        )

    print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    if args.groupby in SHELLCOLUMNS:
        contributions_sorted_reduced = order_and_color_shells(contribution_list, arraylambda_angstroms, args)
    else:
        contributions_sorted_reduced = sort_and_reduce_flux_contribution_list(
            contribution_list,
            args.maxseriescount,
            arraylambda_angstroms,
            fixedionlist=args.fixedionlist,
            hideother=args.hideother,
        )

    plotobjectlabels: list[str] = []
    plotobjects: list[Artist] = []

    dfspectotal = get_xy_spectrum(array_flambda_emission_total, arraylambda_angstroms, args).collect()

    # a narrow x range can hold no centre of a bin, and the drawn line then runs between the bins beyond its edges
    max_f_emission_total = df_filter_minmax_bracketed(dfspectotal, "x", xmin, xmax).collect()["y"].max()
    assert isinstance(max_f_emission_total, (float, np.floating))
    max_f_emission_total = float(max_f_emission_total)

    if scale_to_peak and max_f_emission_total <= 0.0:
        # the scale to the peak divides by this maximum
        exit_with_error(
            "--normalised needs a peak, and no packet of the selection emits inside the plotted range",
            "Widen the time range, the x range, or the selection of the packets",
        )

    scalefactor = scale_to_peak / max_f_emission_total if scale_to_peak else 1.0

    if not args.hidenetspectrum:
        plotobjectlabels.append("Spectrum")
        # the emission plot takes its model from the first series of -specpath
        (line,) = axis.plot(
            dfspectotal["x"],
            dfspectotal["y"] * scalefactor,
            linewidth=args.linewidth[0] or 1.5,
            color="black",
            zorder=100,
        )
        plotobjects.append(line)

    dfaxisdata = pl.DataFrame({"lambda_angstroms": arraylambda_angstroms})
    for contribution in contributions_sorted_reduced:
        dfaxisdata = dfaxisdata.with_columns(
            pl.Series(name=f"emission_flambda.{contribution.linelabel}", values=contribution.array_flambda_emission)
        )
        if args.showabsorption:
            dfaxisdata = dfaxisdata.with_columns(
                pl.Series(
                    name=f"absorption_flambda.{contribution.linelabel}", values=contribution.array_flambda_absorption
                )
            )

    max_absorption = 0.0
    if args.nostack:
        newobjects, max_absorption = plot_contributions_unstacked(
            axis, contributions_sorted_reduced, arraylambda_angstroms, args, scalefactor, xmin, xmax
        )
        plotobjects.extend(newobjects)
    elif contributions_sorted_reduced:
        newobjects, max_absorption = plot_contributions_stacked(
            axis, contributions_sorted_reduced, arraylambda_angstroms, args, scalefactor, xmin, xmax
        )
        plotobjects.extend(newobjects)

    plotobjectlabels.extend([contribution.linelabel for contribution in contributions_sorted_reduced])

    refobjects, reflabels, ymaxrefall = plot_reference_spectra(axis, args, filterfunc, scale_to_peak)
    plotobjects.extend(refobjects)
    plotobjectlabels.extend(reflabels)

    axis.axhline(color="black", linewidth=1)

    set_plot_title(axis, get_emission_plot_label(modelpath, args, modelname, dirbin, timemin, timemax), args)

    # set_legend gives the legend its room above the data
    if args.ymax is None:
        axis.set_ylim(top=max(ymaxrefall, scalefactor * max_f_emission_total) * (1.0 + YMARGIN))

    if args.ymin is None:
        axis.set_ylim(bottom=-scalefactor * max_absorption * (1.0 + YMARGIN))

    return plotobjects, plotobjectlabels, dfaxisdata


def check_residual_args(args: argparse.Namespace) -> None:
    """Stop the command when --residuals cannot apply to the plot that args selects."""
    if args.multispecplot or args.showemission or args.showabsorption or args.emissionabsorption or args.groupby:
        exit_with_error(
            "--residuals applies to a plot of one frame, thus not to -timedayslist, --showemission, or -groupby",
            "Give one time with -t, and no emission or absorption option",
        )
    if args.makevspecpol or args.averagevspecpolfiles or args.output_spectra or "/" in args.stokesparam:
        exit_with_error(
            "--residuals applies only to a plot of spectra, and the other options select a different action",
            "Remove --residuals, or remove --makevspecpol, --averagevspecpolfiles, --output_spectra, or the ratio",
        )
    nreferences = sum(path_is_reference_spectrum(path) for path in args.specpath)
    if nreferences in {0, len(args.specpath)}:
        exit_with_error(
            "--residuals compares a model with a reference spectrum, and the paths hold only one of the two",
            "Give both, e.g. plotspectra mymodel 2003du_20031213_3219_8822_00.txt",
        )
    if args.normalised:
        print_warning("--normalised scales each series to its own peak, thus the residual compares the shapes alone")
    if args.filtersavgol:
        print_warning("the residual and its statistics take the smoothed series of -filtersavgol")


def make_plot(args: argparse.Namespace) -> tuple[mplfig.Figure, npt.NDArray[np.object_], pl.DataFrame, pl.DataFrame]:
    """Plot the spectra that args selects.

    Return the figure, the axes, the plotted data, and the statistics of the residuals.
    """
    fig, axes, residualaxis = make_plot_figure(args)
    dfalldata, dfresidualstats = draw_plot(args, axes, residualaxis)

    defaultoutputfile = (
        "plotspectra_emission_{timemin:.2f}d-{timemax:.2f}d{directionbins}.pdf"
        if args.showemission or args.showabsorption
        else "plotspectra_{timemin:.2f}d-{timemax:.2f}d.pdf"
    )
    args.outputfile = resolve_outputfile(args.outputfile, defaultoutputfile)

    return fig, axes, dfalldata, dfresidualstats


def make_plot_figure(
    args: argparse.Namespace, *, fig: mplfig.Figure | None = None
) -> tuple[mplfig.Figure, npt.NDArray[t.Any], mplax.Axes | None]:
    """Return the figure, the axes of the spectra, and the residual panel if args asks for one.

    If the caller gives an empty figure as fig, the function adds the frames to it, e.g. the figure of the viewer.
    """
    nrows = len(args.timedayslist) if args.multispecplot else 1

    # an emission and absorption plot draws a taller frame
    aspect = FRAMEHEIGHT_INCHES / FRAMEWIDTH_INCHES * (1.56 if args.showabsorption else 1.0)
    residualaxis = None
    if args.residuals:
        fig, mainaxis, residualaxis = make_frame_figure_with_residuals(args, aspect=aspect, fig=fig)
        axesgrid = np.array([[mainaxis]], dtype=object)
    else:
        fig, axesgrid = make_frame_figure(args, rows=nrows, aspect=aspect, sharex=True, sharey=False, fig=fig)

    # the residual panel is not one of these axes, thus the code below treats the main frame as before
    axes = axesgrid[:, 0]
    assert isinstance(axes, np.ndarray)

    return fig, axes, residualaxis


def draw_plot(
    args: argparse.Namespace, axes: npt.NDArray[t.Any], residualaxis: mplax.Axes | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Draw the spectra that args selects, and return the plotted data and the statistics of the residuals.

    The axes must be empty. This function writes no file, thus the viewer can call it again for each change
    of a control.
    """
    dfresidualstats = pl.DataFrame()
    filterfunc = get_filterfunc(args)

    scale_to_peak = 1.0 if args.normalised else None

    # the plot functions read the x range back from the axes, thus the x scale and the x limits go on
    # before the data. The y properties wait until set_auto_yscale has read the drawn values
    set_axis_properties(axes, args, setyaxis=False)

    residualseries: list[ResidualSeries] | None = [] if residualaxis is not None else None
    if args.showemission or args.showabsorption:
        legendncol = 2
        plotobjects, plotobjectlabels, dfalldata = make_emissionabsorption_plot(
            modelpath=Path(args.modelspecpaths[0]),
            axis=axes[-1],
            filterfunc=filterfunc,
            args=args,
            scale_to_peak=scale_to_peak,
        )
    else:
        legendncol = 1

        # the legend comes from the first axis that a plot used, which is axes[0] for
        # --multispecplot and axes[-1] otherwise
        specaxes = list(axes) if args.multispecplot else [axes[-1]]
        dfalldata = make_spectrum_plot(
            args.specpath, specaxes, filterfunc, args, scale_to_peak=scale_to_peak, residualseries=residualseries
        )
        plotobjects, plotobjectlabels = specaxes[0].get_legend_handles_labels()

    # the absorption part of the plot is negative, and a log axis cannot show it. set_auto_yscale reads only the
    # lines. The stacked areas of the contributions are not lines, thus set_auto_yscale can choose a log axis here
    if args.showabsorption and args.yscale == "auto":
        args.yscale = "linear"

    # -yscale auto reads the drawn values, thus the scale of the y axis follows the data
    set_auto_yscale(list(axes), args)

    # the y limits, the locators and the labels all follow the scale that set_auto_yscale chose.
    # A y limit also goes on after the data, so that -ymin alone keeps the top that the data set
    xlabel, ylabel = get_axis_labels(args)
    set_axis_properties(axes, args)

    if args.showtime:
        for index, axis in enumerate(axes):
            if args.multispecplot:
                timetext = f"{args.timedayslist[index]} days"
            else:
                timeavg = (args.timemin + args.timemax) / 2.0
                timetext = f"{timeavg:.2f} days"
            # the position is a fraction of the axes, thus the text stays inside for each x unit and each x range
            axis.annotate(
                timetext,
                xy=(0.03, 0.97),
                xycoords="axes fraction",
                horizontalalignment="left",
                verticalalignment="top",
                fontsize="x-large",
            )

    for axis in axes:
        if not args.logscalex:
            axis.xaxis.set_major_locator(ticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="both"))
            axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        if args.hidexticklabels:
            axis.tick_params(axis="x", which="both", labelbottom=False)

        if args.hideyticklabels:
            axis.tick_params(axis="y", which="both", labelleft=False)
        else:
            axis.set_ylabel(ylabel)

        if not args.logscaley:
            set_exponent_label(axis)

        axis.set_xlabel("")  # remove xlabel (last axis xlabel optionally added later)

    if not args.hidexticklabels:
        axes[-1].set_xlabel(xlabel)

    # the panel shows a ratio below a log y axis, thus it follows the choice of -yscale auto
    if residualaxis is not None and residualseries is not None:
        dfresidualstats = draw_residual_panel(residualaxis, axes[-1], residualseries, args)

    if args.reverselegendorder:  # TODO: consider ax.legend(reverse=True)
        plotobjects, plotobjectlabels = plotobjects[::-1], plotobjectlabels[::-1]

    leg = set_legend(
        axes[-1],
        args,
        handles=plotobjects,
        labels=plotobjectlabels,
        loc="upper right",
        frameon=False,
        handlelength=1 if args.showemission or args.showabsorption else 2,
        # one column was the default, and a long legend of a plain spectrum then took more height than the frame.
        # An ncol of None makes set_legend fit the columns
        ncol=legendncol if legendncol > 1 else None,
        numpoints=1,
        columnspacing=1.0,
    )

    if leg is not None:
        leg.set_zorder(200)

        for line in leg.get_lines():
            line.set_linewidth(2.0)

    return dfalldata, dfresidualstats


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument(
        "specpath",
        default=[],
        nargs="*",
        type=model_path_from_text,
        action=KeepGivenPaths,
        help="Paths to ARTIS folders or reference spectra filenames",
    )
    for flag in ("-specpath", "-modelpath"):
        addarg_pathoption(parser, flag, "specpath", multiplepaths=True)

    addarg_seriesstyle(parser, include_linealpha=True)

    parser.add_argument(
        "--gamma", action="store_true", help="Plot the gamma-ray spectrum instead of the UVOIR spectrum"
    )

    parser.add_argument(
        "--frompackets", action="store_true", help="Read packets files directly instead of exspec results"
    )

    addarg_maxpacketfiles(parser)

    parser.add_argument(
        "--plotinvalidpart",
        action="store_true",
        help="Plot the spectra even if it falls outside the valid time range (due to light travel times)",
    )

    parser.add_argument("--emissionabsorption", action="store_true", help="Implies --showemission and --showabsorption")

    parser.add_argument("--showemission", action="store_true", help="Plot the emission spectra by ion/process")

    parser.add_argument("--showabsorption", action="store_true", help="Plot the absorption spectra by ion/process")

    yvariablechoices = ["flux", "packetcount", "photoncount", "photonflux", "eflux", "luminosity"]
    parser.add_argument(
        "-yvariable",
        "-y",
        type=str,
        default="flux",
        choices=yvariablechoices,
        help="Specify the y-axis variable for the plot",
    )
    # deprecated spelling kept as a hidden alias
    parser.add_argument("-yvar", dest="yvariable", type=str, choices=yvariablechoices, help=argparse.SUPPRESS)

    parser.add_argument(
        "--nostack",
        action="store_true",
        help="Plot each emission/absorption contribution separately instead of a stackplot",
    )

    parser.add_argument(
        "-fixedionlist",
        nargs="+",
        help="Specify a list of ions instead of using the auto-generated list in order of importance",
    )

    parser.add_argument(
        "-maxseriescount",
        type=int,
        default=None,
        help=(
            "The maximum number of series (ions or processes) in an emission or absorption plot. The plot shows only"
            " this number of -fixedionlist entries. The default is the length of -fixedionlist, or"
            f" {DEFAULT_MAXSERIESCOUNT} without that list"
        ),
    )

    addarg_filter(parser)

    addarg_timestep(parser)

    addarg_timedays(parser)

    addarg_timeminmax(
        parser,
        helptext_min="Lower time in days to integrate spectrum",
        helptext_max="Upper time in days to integrate spectrum",
    )

    parser.add_argument(
        "--notimeclamp",
        action="store_true",
        help="Do not extend the time range to the start and the end of its timesteps. This option reads the packets",
    )

    # no code reads this for plotspectra. It stays accepted, because a script holds the spelling,
    # and SUPPRESS keeps it out of the help text
    parser.add_argument("--classicartis", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "-xunit",
        dest="xunit",
        default=None,
        type=parse_xunit_argument,
        help="X (horizontal) axis unit, e.g. angstrom, nm, micron, Hz, keV, MeV",
    )
    # deprecated spellings kept as hidden aliases. -x names the axis variable on plotestimators, but
    # each parser reads its own arguments, and a script holds the -x of this command.
    parser.add_argument("-xunits", dest="xunit", type=parse_xunit_argument, help=argparse.SUPPRESS)
    parser.add_argument("-x", dest="xunit", type=parse_xunit_argument, help=argparse.SUPPRESS)

    addarg_axislimits(
        parser,
        include_y=False,
        wavelength_aliases=True,
        xminhelp="Plot range: minimum x range",
        xmaxhelp="Plot range: maximum x range",
    )

    xbinsizegroup = parser.add_mutually_exclusive_group()

    xbinsizegroup.add_argument(
        "-deltalambda", type=float, default=None, help="Lambda bin size in Angstroms (applies to from_packets only)"
    )

    xbinsizegroup.add_argument(
        "-deltax", "-dx", type=float, default=None, help="Horizontal bin size in x-unit (applies to from_packets only)"
    )

    xbinsizegroup.add_argument(
        "-deltalogx",
        "-dlogx",
        type=parse_deltalogx_argument,
        default=None,
        help=(
            "Horizontal bin size factor x[1] = x[0] * (1 + dlogx) (applies to from_packets only). The keyword"
            " smallestscale sets dlogx to Δv/c, where Δv is the smallest cell width or shell width of all the models."
            " The keyword largestscale sets dlogx to Δv/c, where Δv is the largest cell diagonal or shell width of all"
            " the models"
        ),
    )

    parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument(
        "--hidemodeltimerange", action="store_true", help='Hide the "at (+/- x.xd)" from the line labels'
    )

    parser.add_argument("--hidemodeltime", action="store_true", help="Hide the time from the line labels")

    parser.add_argument("--normalised", action="store_true", help="Normalise all spectra to their peak values")

    addarg_residuals(parser, "reference spectrum")

    timegroup = parser.add_mutually_exclusive_group()

    timegroup.add_argument(
        "--use_escapetime",
        action="store_true",
        help=(
            "Use the time of packet escape to the surface (instead of a plane toward the observer). Each packet then"
            " has the weight of its comoving-frame energy, as in the comoving-frame light curve of ARTIS, and the"
            " bin of its rest-frame frequency"
        ),
    )

    timegroup.add_argument("--use_emissiontime", action="store_true", help="Use the time of packet last emission")

    emissioneventgroup = parser.add_mutually_exclusive_group()
    emissioneventgroup.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Tag packets by their last thermal emission type rather than their last emission process",
    )

    emissioneventgroup.add_argument(
        "--use_sampledemissiontype",
        action="store_true",
        help=(
            "Use one emission of each packet for the emission type. ARTIS samples this emission with equal"
            " probability from the emissions that set the emission type. The absorption is the last absorption"
            " before that emission. On average, each emission of a packet thus gets an equal part of the energy."
            " ARTIS writes the necessary columns only with SAMPLE_RPKT_EMISSION (artis-mcrt/artis#661), and only"
            " -groupby element, ion, or line can use them. With --showemission or --showabsorption, this option"
            " also sets --frompackets"
        ),
    )

    parser.add_argument(
        "-groupby",
        default=None,
        choices=["element", "ion", "line", "nuc", "nucmass", "velocity", "losvelocity", "ye"],
        help=(
            "Use a different colour for each element, ion, line, or nuclide with --showemission, or for each shell"
            " of the last interaction: velocity bins the radial velocity, losvelocity the velocity along the line of"
            " sight, and ye the initial electron fraction of the cell. The element of a gamma packet is the element"
            " of its nuclide. Every choice but ion implies --frompackets"
        ),
    )

    parser.add_argument(
        "-velocityshells",
        type=parse_velocity_argument,
        nargs="+",
        default=None,
        metavar="velocity",
        help=(
            "Edges of the shells of -groupby velocity or losvelocity, in km/s, e.g. 0 5000 10000 20000, or as a"
            " fraction of c, e.g. 0c 0.1c 0.2c 0.3c. A value with a c suffix also puts the labels in units of c."
            " The default is ten shells of equal width up to vmax, and one more shell to the corner of a 2D or 3D"
            " grid, with the labels in units of c when vmax is at least 0.2 c"
        ),
    )

    parser.add_argument(
        "-emissionvelocityrange",
        type=parse_velocity_argument,
        nargs=2,
        default=None,
        metavar=("vmin", "vmax"),
        help=(
            "Keep only the emission and the absorption from a range of the radial velocity. Give vmin and vmax in"
            " km/s, e.g. 5000 10000, or as a fraction of c, e.g. 0.1c 0.2c. An emission takes the velocity of the last"
            " interaction, or of the last thermal emission with --use_thermalemissiontype. An absorption always takes"
            " the velocity of the last interaction. vmin is inside the range, and vmax is outside the range."
            " Implies --frompackets"
        ),
    )

    parser.add_argument(
        "-emissionlosvelocityrange",
        type=parse_velocity_argument,
        nargs=2,
        default=None,
        metavar=("vmin", "vmax"),
        help=(
            "Keep only the emission and the absorption from a range of the velocity along the line of sight. A"
            " positive velocity is motion toward the observer, and a value can be negative, e.g. -0.05c 0.05c. The"
            " other rules of -emissionvelocityrange apply. The two ranges together keep a packet that is inside"
            " both ranges"
        ),
    )

    parser.add_argument(
        "-yeshells",
        type=float,
        nargs="+",
        default=None,
        metavar="Ye",
        help=(
            "Edges of the shells of -groupby ye, e.g. 0 0.1 0.2 0.3 0.5. The default is shells of 0.05 up to 0.5,"
            " and one more shell to 1"
        ),
    )

    # the older spelling of a reference spectrum that a positional path now names
    parser.add_argument("-obsspec", "-refspecfiles", action="append", dest="refspecfiles", help=argparse.SUPPRESS)

    parser.add_argument(
        "-distmpc",
        type=float,
        default=None,
        help="Distance in megaparsec when calculating fluxes (default: first reference spec distance or 1 Mpc)",
    )
    # deprecated spellings kept as hidden aliases
    parser.add_argument("-dist_mpc", "-dist", "-fluxdistmpc", dest="distmpc", type=float, help=argparse.SUPPRESS)

    parser.add_argument(
        "-scaletoreftime", type=float, default=None, help="Scale reference spectra flux using Co56 decay timescale"
    )

    addarg_figscale(parser, include_figwidthscale=True)

    parser.add_argument("--logscalex", action="store_true", help="Use log scale for x values")

    addarg_yscale(parser)

    # the older spelling of "-yscale log"
    parser.add_argument("--logscaley", action="store_true", help="Use log scale for y values")

    parser.add_argument("--hidenetspectrum", action="store_true", help="Hide net spectrum")

    parser.add_argument("--hideother", action="store_true", help="Hide other contributions")

    addarg_notitle(parser)

    parser.add_argument("-title", type=str, default=None, help="Custom plot title text")

    parser.add_argument("--inset_title", action="store_true", help="Place title inside the plot")

    addarg_legend(parser)

    parser.add_argument("--reverselegendorder", action="store_true", help="Reverse the order of legend items")

    parser.add_argument("--hidexticklabels", action="store_true", help="Don't show numbers or a label on the x axis")

    parser.add_argument("--hideyticklabels", action="store_true", help="Don't show numbers or a label on the y axis")

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a CSV file")

    addarg_output(parser, kind="file", helptext="Path/filename for PDF file")

    addarg_dpi(parser)

    addarg_show(parser)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a window with controls for the time, the x range, and the emission plot, and show the command",
    )
    addarg_verbose(parser)

    parser.add_argument(
        "--output_spectra", "--write_spectra", action="store_true", help="Write out all timestep spectra to text files"
    )

    # Combines all vspecpol files into one file which can then be read by artistools
    parser.add_argument(
        "--makevspecpol", action="store_true", help="Make file summing the virtual packet spectra from all ranks"
    )

    # To get better statistics for polarisation use multiple runs of the same simulation. This will then average the
    # files produced by makevspecpol for all simulations.
    parser.add_argument(
        "--averagevspecpolfiles", action="store_true", help="Average the vspecpol-total files for multiple simulations"
    )

    addarg_viewingangle(parser)

    parser.add_argument(
        "-stokesparam",
        type=str,
        default="I",
        choices=["I", "Q", "U", "Q/I", "U/I"],
        help=(
            "The Stokes parameter to plot. Q and U read the virtual packet spectra of -plotvspecpol. The ratio Q/I"
            " or U/I draws the polarisation plot from specpol.out, specpol_res.out, or the virtual packet spectra"
        ),
    )

    parser.add_argument("--binflux", action="store_true", help="Bin flux over wavelength and average flux")

    parser.add_argument(
        "--histogram",
        action="store_true",
        help=(
            "Draw each ARTIS spectrum as a histogram, with a horizontal line across each bin. A packet spectrum"
            " (--frompackets) has its steps on the wavelength edges of the packet bins. A different spectrum, e.g."
            " spec.out or a spectrum with --binflux, has its steps at the midpoints on the x axis between adjacent bin"
            " centres. Each outer edge is at the same distance from its centre as the inner edge of that bin"
        ),
    )

    parser.add_argument(
        "--shownoise",
        action="store_true",
        help=(
            "Draw the Monte Carlo noise band around each ARTIS spectrum. The band goes from y * (1 - r) to"
            " y * (1 + r). r is the estimate of one standard deviation of y divided by y. r = sqrt(sum(e^2)) / sum(e)"
            " for the energies e of the packets in the bin. For -yvariable packetcount, r = 1 / sqrt(N). Implies"
            " --frompackets"
        ),
    )

    parser.add_argument(
        "--showfilterfunctions",
        action="store_true",
        help="Plot Bessell filter functions over spectrum. Also use --normalised",
    )

    parser.add_argument(
        "--multispecplot", action="store_true", help="Plot multiple spectra in subplots - expects timedayslist"
    )

    parser.add_argument("-timedayslist", nargs="+", help="List of times in days for time sequence subplots")

    parser.add_argument("--showtime", action="store_true", help="Write time on plot")

    parser.add_argument(
        "--vpkt_match_emission_exclusion_to_opac",
        action="store_true",
        help="Exclude packets with emission type no-bb/no-bf/no-(element) matching the vpkt opacity exclusion",
    )


def parse_velocity_values(
    values: Sequence[str | float | tuple[float, t.Literal["kmps", "c"]]],
) -> tuple[list[float], t.Literal["kmps", "c"]]:
    """Return the velocities [km/s] and the unit of their labels.

    The unit is c if one value has a c suffix. argparse gives a parsed pair, and a keyword argument
    of the API gives a text or a number.
    """
    parsedvalues = [parse_velocity_argument(str(value)) if not isinstance(value, tuple) else value for value in values]
    unit: t.Literal["kmps", "c"] = "c" if any(valueunit == "c" for _, valueunit in parsedvalues) else "kmps"
    return [velocity_kmps for velocity_kmps, _ in parsedvalues], unit


# the number of series of an emission or absorption plot without -maxseriescount or -fixedionlist
DEFAULT_MAXSERIESCOUNT: t.Final = 14

# the argument and the name in the title of the velocity range of each shell grouping
VELOCITYRANGEARGS: t.Final[Mapping[str, tuple[str, str]]] = MappingProxyType({
    "velocity": ("emissionvelocityrange", "radial velocity"),
    "losvelocity": ("emissionlosvelocityrange", "line-of-sight velocity"),
})


def get_emission_event(args: argparse.Namespace) -> EmissionEvent:
    """Return the emission that gives the emission type of a packet."""
    assert not (args.use_thermalemissiontype and args.use_sampledemissiontype)
    if args.use_sampledemissiontype:
        return "sampled"
    return "thermal" if args.use_thermalemissiontype else "last"


def exit_if_no_emission_position(args: argparse.Namespace) -> None:
    """Stop if a shell grouping or a velocity range needs an emission position that the packets do not hold.

    A virtual packet holds no emission position. A gamma packet holds the position of its decay, of its pair
    annihilation, or of its last Compton scattering, but it has no thermal emission. The packets hold no position of
    the sampled emission.
    """
    if args.groupby in SHELLCOLUMNS:
        option = f"-groupby {args.groupby}"
    elif args.velocityranges_kmps:
        option = " and ".join(f"-{VELOCITYRANGEARGS[rangegrouping][0]}" for rangegrouping in args.velocityranges_kmps)
    else:
        return

    # a virtual packet holds no emission position for any emission, thus this check comes before the checks of
    # --use_thermalemissiontype and --use_sampledemissiontype
    if args.plotvspecpol is not None:
        exit_with_error(
            f"a virtual packet holds no emission position, thus -plotvspecpol does not accept {option}",
            "Give -plotviewingangle for a direction bin of the real packets",
        )

    if args.gamma and args.use_thermalemissiontype:
        exit_with_error(
            f"a gamma packet has no thermal emission, thus {option} cannot use --use_thermalemissiontype",
            "Remove --use_thermalemissiontype. The gamma packets then use the position of the last interaction",
        )

    if args.use_sampledemissiontype:
        # a gamma packet has no thermal emission, and an absorption always takes the last interaction
        thermalhint = (
            ", or give --use_thermalemissiontype for the last thermal emission"
            if args.showemission and not args.gamma
            else ""
        )
        exit_with_error(
            f"the packets hold no position of the sampled emission, thus {option} cannot use --use_sampledemissiontype",
            f"Remove --use_sampledemissiontype{thermalhint}",
        )


def resolve_velocity_ranges(args: argparse.Namespace) -> None:
    """Set the velocity ranges in km/s and their labels for the title, from the two range arguments."""
    args.velocityranges_kmps = {}
    args.velocityrangelabels = []
    for rangegrouping, (argname, velocityname) in VELOCITYRANGEARGS.items():
        rangevalues = getattr(args, argname)
        if rangevalues is None:
            continue

        velocities_kmps, unit = parse_velocity_values(rangevalues)
        if len(velocities_kmps) != 2:
            exit_with_error(
                f"-{argname} takes two velocities, not {len(velocities_kmps)}",
                f"Give vmin and vmax, e.g. -{argname} 0.1c 0.2c",
            )
        args.velocityranges_kmps[rangegrouping] = (velocities_kmps[0], velocities_kmps[1])
        args.velocityrangelabels.append(f"{velocityname} {get_shell_labels(velocities_kmps, unit)[0]}")

    if not args.velocityranges_kmps:
        return

    if not (args.showemission or args.showabsorption):
        exit_with_error(
            "a velocity range selects the packets of the contributions, and the plot shows no contribution",
            "Give --showemission, --showabsorption, or --emissionabsorption",
        )


def resolve_shell_args(args: argparse.Namespace) -> None:
    """Set the shell edges and the unit of their labels for a shell grouping, from the arguments or the model."""
    args.shelledges = None
    args.shellunit = "kmps"
    if args.groupby == "ye":
        args.shelledges = list(args.yeshells) if args.yeshells is not None else list(DEFAULT_YE_SHELLS)
        args.shellunit = "ye"
    elif args.groupby in SHELLCOLUMNS and args.velocityshells is None:
        # the plot draws the first ARTIS model, thus the shells come from that model. Without a model,
        # check_emission_plot_args stops the command
        if args.modelspecpaths:
            getdefault = (
                get_default_losvelocity_shells if args.groupby == "losvelocity" else get_default_velocity_shells
            )
            args.shelledges, args.shellunit = getdefault(args.modelspecpaths[0])
    elif args.velocityshells is not None:
        args.shelledges, args.shellunit = parse_velocity_values(args.velocityshells)


@lru_cache(maxsize=64)
def has_gamma_spec_file(runfolder: Path) -> bool:
    """Return True if the run has gamma_spec.out.

    The spectrum viewer resolves the arguments at each change, and the search of the subfolders is slow on a network
    drive. Thus a file that exspec writes later stays unknown until Reload Data clears the cache, and until then the
    plot reads the packets.
    """
    return firstexisting_or_none("gamma_spec.out", folder=runfolder) is not None


@modelpath_cache(maxsize=64)
def get_no_sampled_emission_reason(runfolder: Path) -> str | None:
    """Return why the packets of the run hold no sampled emission, or None if they hold it.

    A run with no packets also gives None, and the packets reader gives its error later. The spectrum viewer resolves
    the arguments at each change. Thus the cache keeps the result until Reload Data clears it, also after a new run of
    ARTIS in the folder.
    """
    columnnames = get_packets_column_names(runfolder)
    return None if columnnames is None else get_no_sampled_emission_message(runfolder, columnnames)


def resolve_frompackets(args: argparse.Namespace) -> None:
    """Set args.frompackets and the default of -groupby, from the options that the exspec files cannot serve.

    Call this after main sets args.showemission and args.showabsorption. The default of -groupby
    applies to an emission plot alone, and that default selects the reader of the contributions.
    args.frompacketsreason names the first option that needs the packets, or it is None. The spectrum viewer
    shows it, also when the command gives --frompackets.
    """
    showcontributions = args.showemission or args.showabsorption
    if showcontributions and args.groupby is None:
        args.groupby = "nuc" if args.gamma else "ion"

    # each entry names an option in the message, and gives the condition under which it needs the packets
    packetreasons = {
        "-plotvspecpol and --showemission": showcontributions and bool(args.plotvspecpol),
        # the emission files of exspec give only the emission type of the last interaction and of the last thermal
        # emission
        "--use_sampledemissiontype": showcontributions and args.use_sampledemissiontype,
        "--gamma": args.gamma and (showcontributions or bool(args.plotviewingangle)),
        "--gamma with no gamma_spec.out": args.gamma
        and not all(
            has_gamma_spec_file(runfolder)
            for runfolder in get_artis_run_folders([
                Path(path) for path in args.specpath if not path_is_reference_spectrum(path)
            ])
        ),
        f"-groupby {args.groupby}": args.groupby in {"element", "line", "nuc", "nucmass", *SHELLCOLUMNS},
        "a velocity range": bool(args.velocityranges_kmps),
        "--use_emissiontime or --use_escapetime": args.use_emissiontime or args.use_escapetime,
        "a custom bin width": any(value is not None for value in (args.deltax, args.deltalogx, args.deltalambda)),
        # spec.out and the emission files hold whole timesteps, thus only the packets give a time range inside one
        # timestep
        "--notimeclamp": args.notimeclamp,
        # spec.out holds a flux and no count of packets
        "-yvariable packetcount": args.yvariable == "packetcount",
        # spec.out holds no energies of the packets in each bin, thus it gives no noise
        "--shownoise": args.shownoise,
    }
    args.frompacketsreason = next((option for option, needspackets in packetreasons.items() if needspackets), None)
    if args.frompacketsreason is not None and not args.frompackets:
        args.frompackets = True
        print(f"Enabling --frompackets, since {args.frompacketsreason} was specified")


def check_emission_plot_args(args: argparse.Namespace) -> None:
    """Stop the command when an emission plot cannot draw what the arguments name.

    Such a plot draws the contributions of one ARTIS model and of one direction bin. The name of the
    output file gives the model and the bins, thus more than one of either would not match the figure.
    """
    if args.vpkt_match_emission_exclusion_to_opac:
        missing = [
            option
            for option, given in (
                ("--showemission", args.showemission),
                ("--frompackets", args.frompackets),
                ("-plotvspecpol", args.plotvspecpol),
            )
            if not given
        ]
        if missing:
            exit_with_error(
                "--vpkt_match_emission_exclusion_to_opac reads the emission type of the virtual packets, and the"
                f" command gives no {' and no '.join(missing)}",
                "Give --showemission (or --emissionabsorption) and -plotvspecpol",
            )

    if not (args.showemission or args.showabsorption):
        return

    if not args.modelspecpaths:
        exit_with_error(
            "an emission plot draws the contributions of one ARTIS model, and no path names such a model",
            "Give the folder of an ARTIS run",
        )

    if len(args.modelspecpaths) > 1:
        exit_with_error(
            f"an emission plot draws one ARTIS model, and the command gives {len(args.modelspecpaths)} of them",
            "Give one model folder, and run the command again for each other model",
        )

    dirbins = args.plotviewingangle or args.plotvspecpol
    if dirbins and len(dirbins) > 1:
        exit_with_error(
            f"an emission plot draws one direction bin, and the command gives {len(dirbins)} of them",
            "Give one bin, e.g. -plotviewingangle 0",
        )

    if args.timedayslist and len(args.timedayslist) > 1:
        exit_with_error(
            "an emission plot draws one time, and -timedayslist gives several. The plot drew only the first one",
            "Give one time with -timedays. Run the command again for each other time",
        )

    # a gamma packet comes from the decay of a nuclide, and ARTIS records no ion or line for it
    if args.gamma and args.groupby in {"ion", "line"}:
        exit_with_error(
            f"a gamma packet has no emission by an ion or a line, thus a gamma-ray spectrum does not accept -groupby"
            f" {args.groupby}",
            "Give -groupby nuc or -groupby nucmass",
        )

    # ARTIS gives each escaped gamma packet the absorption type 0, which is the first line of the line list
    if args.gamma and args.showabsorption:
        exit_with_error(
            "a gamma packet holds no absorption record, thus a gamma-ray spectrum does not accept --showabsorption",
            "Remove --showabsorption",
        )

    # ARTIS samples only the emissions of the real r-packets
    if args.use_sampledemissiontype and args.gamma:
        exit_with_error(
            "a gamma packet has no sampled emission, thus a gamma-ray spectrum does not accept"
            " --use_sampledemissiontype",
            "Remove --use_sampledemissiontype",
        )

    if args.use_sampledemissiontype and args.plotvspecpol is not None:
        exit_with_error(
            "a virtual packet holds no sampled emission, thus -plotvspecpol does not accept --use_sampledemissiontype",
            "Give -plotviewingangle for a direction bin of the real packets, or remove --use_sampledemissiontype",
        )

    if args.use_sampledemissiontype and args.groupby in {"nuc", "nucmass"}:
        exit_with_error(
            f"-groupby {args.groupby} takes the nuclide of the pellet and not an emission, thus it does not accept"
            " --use_sampledemissiontype",
            "Give -groupby element, ion, or line, or remove --use_sampledemissiontype",
        )

    # most runs have no sampled emission, thus the command stops before it reads the packets
    if args.use_sampledemissiontype:
        for runfolder in get_artis_run_folders(args.modelspecpaths):
            if (reason := get_no_sampled_emission_reason(runfolder)) is not None:
                exit_with_error(reason, NO_SAMPLED_EMISSION_HELP)

    # get_flux_contributions_from_packets makes the same test, but only after it reads the packets
    if args.showabsorption and args.groupby == "nuc":
        exit_with_error(
            "a nuclide emits a packet, but a nuclide does not absorb a packet. Thus -groupby nuc cannot group the"
            " absorption contributions",
            "Give -groupby ion or line, or remove --showabsorption",
        )

    if args.yvariable == "packetcount":
        exit_with_error(
            "an emission plot draws a flux for each contribution, and it has no count of packets",
            "Remove -yvariable packetcount, or remove --showemission and --showabsorption",
        )


def check_yvariable_args(args: argparse.Namespace) -> None:
    """Stop the command when a series cannot give the quantity that -stokesparam or -yvariable names.

    Only the virtual packet spectra hold the Stokes parameters Q and U, and a reference spectrum holds
    no count of packets. Such a plot drew Stokes I, or it stopped with a missing column.
    """
    # a ratio such as Q/I selects the polarisation plot, which reads specpol_res.out. The packets give Stokes I
    # alone, and resolve_frompackets can select them, e.g. for --showemission
    if args.stokesparam != "I" and "/" not in args.stokesparam and (args.plotvspecpol is None or args.frompackets):
        exit_with_error(
            f"-stokesparam {args.stokesparam} reads the virtual packet spectra files, and the command gives no"
            " -plotvspecpol or it reads the packets",
            "Give -plotvspecpol, e.g. -plotvspecpol 0, and remove the options that need the packets",
        )

    if args.yvariable == "packetcount" and len(args.modelspecpaths) < len(args.specpath):
        exit_with_error(
            "a reference spectrum holds no count of packets, thus -yvariable packetcount needs ARTIS models alone",
            "Remove the reference spectra, or give another -yvariable",
        )


def check_bin_drawing_args(args: argparse.Namespace) -> None:
    """Stop the command if an option makes the histogram or the noise band incorrect.

    --binflux combines 5 bins, and a filter calculates each value from adjacent bins. Thus the noise of one bin does
    not apply. An emission plot and a Stokes ratio, e.g. -stokesparam Q/I, draw no band and no histogram.
    """
    if args.histogram:
        for option, given in (
            ("--showemission", args.showemission),
            ("--showabsorption", args.showabsorption),
            (f"-stokesparam {args.stokesparam}", "/" in args.stokesparam),
        ):
            if given:
                exit_with_error(
                    f"--histogram applies only to the line of each ARTIS spectrum, and {option} draws a different plot",
                    f"Remove {option}, or remove --histogram",
                )
    if not args.shownoise:
        return
    filterreason = "calculates each value from adjacent bins, thus the noise of one bin is incorrect"
    conflicts = [
        (option, reason)
        for option, given, reason in (
            ("--showemission", args.showemission, "draws an emission plot, which has no noise band"),
            ("--showabsorption", args.showabsorption, "draws an emission plot, which has no noise band"),
            ("--binflux", args.binflux, "combines the bins, thus the noise of one bin is incorrect"),
            ("-filtersavgol", args.filtersavgol, filterreason),
            ("-filtermovingavg", args.filtermovingavg, filterreason),
            (
                f"-stokesparam {args.stokesparam}",
                "/" in args.stokesparam,
                "draws a ratio from specpol_res.out, which holds no packets",
            ),
        )
        if given
    ]
    if conflicts:
        option, reason = conflicts[0]
        exit_with_error(
            f"--shownoise draws the noise of each bin, and {option} {reason}", f"Remove {option}, or remove --shownoise"
        )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot spectra from ARTIS and reference data."""
    # the dispatcher parses the command line and gives args alone, thus the viewer then reads sys.argv
    fromdispatcher = args is not None
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.interactive:
        from artistools.spectra.interactive import run_viewer
        from artistools.viewertools.core import get_command_tokens

        run_viewer(get_command_tokens(args, argsraw, kwargs, fromdispatcher=fromdispatcher))
        return

    resolve_plot_args(args)

    if args.makevspecpol:
        make_virtual_spectra_summed_file(args.specpath[0])
        return

    if args.averagevspecpolfiles:
        make_averaged_vspecfiles(args.specpath)
        return

    if "/" in args.stokesparam:
        plot_polarisation(args.specpath[0], args)
        return

    if args.output_spectra:
        # -o names the folder of the files, and no -o keeps them in the spectra folder of the model
        outputfile = Path(args.outputfile) if args.outputfile else None
        if outputfile is not None and outputfile.suffixes and not outputfile.is_dir():
            msg = f"--output_spectra writes a folder of files, thus -o must name a folder and not {outputfile}"
            raise ValueError(msg)
        outdirectory = make_output_folder(outputfile, "writes") if outputfile is not None else None
        for modelpath in args.specpath:
            # the file names hold no model name, thus several models get a subfolder each
            modeloutdirectory = (
                outdirectory / get_model_name(modelpath)
                if outdirectory is not None and len(args.specpath) > 1
                else outdirectory
            )
            write_flambda_spectra(modelpath, outdirectory=modeloutdirectory)

    else:
        fig, _axes, dfalldata, dfresidualstats = make_plot(args)

        # the observers of the virtual packets also name the file, thus the plots of two observers keep both files
        if args.plotviewingangle:
            strdirectionbins = "_direction" + "_".join([f"{angle:02d}" for angle in args.plotviewingangle])
        elif args.plotvspecpol:
            strdirectionbins = "_vspecpol" + "_".join([f"{observer:02d}" for observer in args.plotvspecpol])
        else:
            strdirectionbins = ""

        filenameout = str(args.outputfile)
        if args.timemin is not None and args.timemax is not None:
            filenameout = filenameout.format(timemin=args.timemin, timemax=args.timemax, directionbins=strdirectionbins)
        elif "{" in filenameout:
            # no model resolved both bounds, thus the time placeholders get no values
            filenameout = str(Path(filenameout).with_name("plotspectra.pdf"))

        if args.write_data and len(dfalldata.columns) > 0:
            datafilenameout = Path(filenameout).with_suffix(".txt")
            dfalldata.write_csv(datafilenameout, separator=" ")
            print_saved(datafilenameout)
        if args.write_data:
            write_residual_stats(dfresidualstats, filenameout)

        save_figure(fig, filenameout, args=args, dpi=args.dpi)


def get_default_xlimits(xunit: str, *, gamma: bool) -> tuple[float, float]:
    """Return the x limits in increasing order for a command that gives no -xmin and no -xmax."""
    lambdalimits = (0.2, 0.004) if gamma else (2500.0, 19000.0)
    xmin, xmax = sorted(convert_angstroms_to_unit(value, xunit) for value in lambdalimits)
    return xmin, xmax


def resolve_plot_args(args: argparse.Namespace) -> None:
    """Apply the defaults and the time range to args, and stop the command for a bad combination of arguments."""
    if args.xunit is None:
        args.xunit = "kev" if args.gamma else "angstrom"
    args.xunit = convert_xunit_aliases_to_canonical(args.xunit)

    defaultxmin, defaultxmax = get_default_xlimits(args.xunit, gamma=args.gamma)
    # a sort of a given limit and a default limit put the given limit on the other side, thus the plot showed a
    # different range. A pair that the user gave keeps the sort
    if args.xmin is not None and args.xmax is None and args.xmin >= defaultxmax:
        exit_with_error(
            f"-xmin {args.xmin:g} is not below the default upper limit {defaultxmax:g} {args.xunit}",
            "Give -xmax as well",
        )
    if args.xmax is not None and args.xmin is None and args.xmax <= defaultxmin:
        exit_with_error(
            f"-xmax {args.xmax:g} is not above the default lower limit {defaultxmin:g} {args.xunit}",
            "Give -xmin as well",
        )
    if args.xmin is None:
        args.xmin = defaultxmin
    if args.xmax is None:
        args.xmax = defaultxmax

    args.xmin, args.xmax = sorted([args.xmin, args.xmax])

    # -fixedionlist names each series, thus its length is the default count. A smaller -maxseriescount removes the
    # last entries
    if args.maxseriescount is None:
        args.maxseriescount = len(args.fixedionlist) if args.fixedionlist else DEFAULT_MAXSERIESCOUNT

    assert (
        not args.plotvspecpol or not args.plotviewingangle
    )  # choose either virtual packet directions or real packet direction bins

    # -obsspec named a reference spectrum before a positional path could name one. Nothing read that
    # list, thus the command drew the model alone and said nothing. The paths join the positional ones
    args.specpath = normalize_path_list([*makelist(args.specpath), *(args.refspecfiles or [])])

    if args.timedayslist:
        args.multispecplot = True
        args.timedays = args.timedayslist[0]

    if args.classicartis:
        print_warning(
            "plotspectra ignores --classicartis. A spectrum comes from spec.out or from the packets,"
            " and not from the estimators"
        )

    # one time axis serves every model, thus the range resolves one time and before any plot runs. A path
    # that is not a run gives no timesteps, and the plot loop skips it with a warning
    clamp_to_timesteps = not args.notimeclamp
    # the emission plot and the default shells read the first model, thus every reader takes the same list
    args.modelspecpaths = [path for path in args.specpath if not path_is_reference_spectrum(path)]
    modelspecpaths = args.modelspecpaths
    artispaths = get_artis_run_folders(modelspecpaths)
    codecomparisonpaths = [path for path in modelspecpaths if path_is_codecomparison(path)]
    if args.timestep is not None and args.timedays is None and not artispaths and not codecomparisonpaths:
        # a reference spectrum has no timesteps. The command plotted it before the range resolution moved to main
        print_warning(
            "-timestep names a timestep of a model, and no path is an ARTIS run or a code comparison model,"
            " thus it has no effect"
        )
    else:
        apply_time_range_args(args, modelspecpaths, clamp_to_timesteps=clamp_to_timesteps)

    gave_a_time = any(value is not None for value in (args.timestep, args.timedays, args.timemin, args.timemax))
    # a code comparison path carries timesteps of its own, thus it resolves the range when no ARTIS
    # model does. The plot code no longer writes the range back, thus nothing else would resolve it
    timesteppaths = artispaths or codecomparisonpaths
    if gave_a_time and timesteppaths and (args.timemin is None or args.timemax is None):
        # the output file name and the time annotation need both bounds. A single -timedays names one
        # time, and a -timemin or a -timemax on its own leaves the other side open.
        # -timedayslist names one epoch for each subplot, thus the range spans the whole list. The
        # first epoch alone would give one name to two lists that share it
        # read the range of every model, because a model can end before a -timemin that a later model reaches
        timedaysvalues = args.timedayslist or [args.timedays]
        # the plot skips a model that does not hold a time, as it skips a model outside a -timemin -timemax range
        resolvedranges = [
            get_time_range(
                timesteppath,
                timestep_range_str=args.timestep,
                timemin=args.timemin,
                timemax=args.timemax,
                timedays_range_str=timedays,
                clamp_to_timesteps=clamp_to_timesteps,
            )[2:]
            for timesteppath in timesteppaths
            for timedays in timedaysvalues
            if model_covers_timedays(timesteppath, timedays, quiet=True)
        ]
        finiteranges = [
            (rangemin, rangemax)
            for rangemin, rangemax in resolvedranges
            if math.isfinite(rangemin) and math.isfinite(rangemax)
        ]
        if finiteranges:
            args.timemin = min(rangemin for rangemin, _ in finiteranges)
            args.timemax = max(rangemax for _, rangemax in finiteranges)

    if args.residuals:
        check_residual_args(args)

    if args.multispecplot and not args.timedayslist:
        # every later step reads one epoch of the list for each subplot, thus an absent list gave a
        # TypeError on len(None). -timedayslist sets --multispecplot, thus the flag alone reaches here
        exit_with_error(
            "--multispecplot draws one subplot for each epoch of -timedayslist, and no such list was given",
            "Give the epochs with -timedayslist, e.g. -timedayslist 260 280 300",
        )

    if args.showtime and not args.multispecplot and (args.timemin is None or args.timemax is None):
        # the annotation writes the middle of the range, thus a missing bound gave a TypeError here.
        # A reference spectrum has no timesteps, thus no path can resolve a range for it
        exit_with_error(
            "--showtime writes the middle of the plotted time range, and no time range was given",
            "Give -timedays (e.g. -t 300 or -t 290-320), -timestep (e.g. -ts 40), or -timemin and -timemax",
        )

    # the reference spectra get black and greys, and the ARTIS models get the colours of the cycle
    args.color = resolve_series_styles(
        args,
        [path_is_reference_spectrum(filepath) for filepath in args.specpath],
        args.color,
        "label",
        "linestyle",
        "linealpha",
        "dashes",
        "linewidth",
    )

    if args.distmpc is None:
        for filepath in args.specpath:
            if path_is_reference_spectrum(filepath):
                fullfilepath = find_reference_spectrum_file(filepath)
                args.distmpc = get_file_metadata(fullfilepath).get("dist_mpc")
                if args.distmpc is not None:
                    print(f"Found distance {args.distmpc} Mpc in metadata of {filepath}")
                break
        if args.distmpc is None:
            args.distmpc = 1.0  # no reference spectra with distances, so default to 1 Mpc
    assert args.distmpc is not None
    if args.distmpc <= 0.0:
        msg = f"-distmpc gives the distance of the observer in Mpc, thus it must be above zero, not {args.distmpc}"
        raise ValueError(msg)

    if args.emissionabsorption:
        args.showemission = True
        args.showabsorption = True

    if args.groupby is not None:
        args.showemission = True

    # argparse rejects the two options together on the command line, but keyword arguments become defaults that
    # argparse does not check
    if args.use_thermalemissiontype and args.use_sampledemissiontype:
        exit_with_error(
            "--use_thermalemissiontype and --use_sampledemissiontype select two different emissions of a packet, and a"
            " plot uses only one",
            "Remove one of the two options",
        )

    resolve_velocity_ranges(args)
    resolve_shell_args(args)
    exit_if_no_emission_position(args)
    resolve_deltalogx_scale(args)
    resolve_frompackets(args)
    check_emission_plot_args(args)
    check_yvariable_args(args)
    check_bin_drawing_args(args)
