# PYTHON_ARGCOMPLETE_OK
"""Artistools - spectra plotting functions."""

import argparse
import contextlib
import math
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

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

import artistools.spectra.spectra as atspectra
from artistools.commands import get_path
from artistools.commands import run_subcommand
from artistools.constants import c_ang_per_s
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_dpi
from artistools.misc import addarg_figscale
from artistools.misc import addarg_filter
from artistools.misc import addarg_maxpacketfiles
from artistools.misc import addarg_nolegend
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_pathoption
from artistools.misc import addarg_seriesstyle
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timeminmax
from artistools.misc import addarg_timestep
from artistools.misc import addarg_verbose
from artistools.misc import addarg_viewingangle
from artistools.misc import addarg_yscale
from artistools.misc import df_filter_minmax_bracketed
from artistools.misc import exit_with_error
from artistools.misc import find_reference_data_file
from artistools.misc import get_dirbin_definitions
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_file_metadata
from artistools.misc import get_filterfunc
from artistools.misc import get_model_folder
from artistools.misc import get_model_name
from artistools.misc import get_series_label
from artistools.misc import get_time_range
from artistools.misc import get_vpkt_config
from artistools.misc import KeepGivenPaths
from artistools.misc import makelist
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import path_is_artis_model
from artistools.misc import path_is_codecomparison
from artistools.misc import path_is_reference_data
from artistools.misc import print_detail
from artistools.misc import print_heading
from artistools.misc import print_saved
from artistools.misc import print_theta_phi_definitions
from artistools.misc import print_warning
from artistools.misc import read_wsv
from artistools.misc import resolve_outputfile
from artistools.misc import resolve_series_styles
from artistools.plottools import FRAMEHEIGHT_INCHES
from artistools.plottools import FRAMEWIDTH_INCHES
from artistools.plottools import label_dirbin_series
from artistools.plottools import make_frame_figure
from artistools.plottools import plain_label
from artistools.plottools import print_dirbin_summary
from artistools.plottools import save_figure
from artistools.plottools import set_auto_yscale
from artistools.plottools import set_axis_properties
from artistools.plottools import set_exponent_label
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title
from artistools.plottools import set_prop_cycle_unusedcolors
from artistools.spectra.writespectra import write_flambda_spectra

if t.TYPE_CHECKING:
    import matplotlib.typing as mplt


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
        _, validrange_start_days, validrange_end_days = get_escaped_arrivalrange(modelpath)
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
            problem_messages.append("To override this error and plot anyway, run with --plotinvalidpart")
            raise ValueError("\n".join(problem_messages))

        for message in problem_messages:
            print_warning(message)


def get_axis_labels(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Get the x-axis and y-axis labels based on the arguments."""
    xunit = atspectra.get_xunit(args.xunit)
    xtype = {"wavelength": "Wavelength", "frequency": "Frequency", "energy": "Energy"}[xunit.kind]
    str_xunit = xunit.label

    xlabel = None if args.hidexticklabels else f"{xtype} [{str_xunit}]"

    ylabel = None
    if not args.hideyticklabels:
        if args.normalised:
            match args.yvariable:
                case "flux":
                    ylabel = r"Scaled F$_\lambda$"
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

            if args.groupby is not None:
                # emission plots add an offset to the reference spectra
                ylabel += " + offset"
        else:
            strdist = str(args.distmpc).removesuffix(".0") + " Mpc"
            match args.yvariable:
                case "flux":
                    if xunit.kind == "wavelength":
                        ylabel = rf"F$_\lambda$ at {strdist} [{{}}erg/s/cm$^2$/{str_xunit}]"
                    elif xunit.kind == "frequency":
                        ylabel = rf"F$_\nu$ at {strdist} [{{}}erg/s/cm$^2$/{str_xunit}]"
                    else:
                        ylabel = f"dF/dE at {strdist} [{{}}erg/s/cm$^2$/{str_xunit}]"
                case "luminosity":
                    ylabel = f"Luminosity [{{}}erg/s/{str_xunit}]"
                case "packetcount":
                    ylabel = r"{}Monte Carlo packets per bin"
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
        stokes_params = atspectra.get_vspecpol_data(vspecindex=angle, modelpath=modelpath)
    else:
        angle = args.plotviewingangle[0] if args.plotviewingangle else -1
        stokes_params = atspectra.get_specpol_data(dirbin=angle, modelpath=modelpath)

    dfspectrum = stokes_params[args.stokesparam].with_columns(lambda_angstroms=c_ang_per_s / pl.col("nu")).collect()

    timearray = dfspectrum.columns[1:-1]
    (_, _, args.timemin, args.timemax) = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )
    assert args.timemin is not None
    assert args.timemax is not None

    timeavg_float = (args.timemin + args.timemax) / 2.0

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

    if args.binflux:
        dfbinned = atspectra.bin_spectrum(dfspectrum, 5, "lambda_angstroms", timecolname)
        axis.plot(dfbinned["lambda_angstroms"], dfbinned[timecolname])
    else:
        axis.plot(dfspectrum["lambda_angstroms"], dfspectrum[timecolname], label=linelabel)

    if args.ymax is None:
        args.ymax = 0.5
    if args.ymin is None:
        args.ymin = -0.5
    if args.xmax is None:
        args.xmax = 10000
    if args.xmin is None:
        args.xmin = 0
    assert args.xmin < args.xmax
    assert args.ymin < args.ymax

    axis.set_ylim(args.ymin, args.ymax)
    axis.set_xlim(args.xmin, args.xmax)

    axis.set_ylabel(str(args.stokesparam))
    axis.set_xlabel(r"Wavelength ($\mathrm{{\AA}}$)")
    figname = f"plotpol_{timeavg}_days_{args.stokesparam.split('/')[0]}_{args.stokesparam.split('/')[1]}.pdf"
    outpath = resolve_outputfile(args.outputfile, figname)
    save_figure(fig, outpath, format="pdf", args=args)


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
    xunit: str = "angstroms",
    yvariable: str = "flux",
    **plotkwargs: t.Any,
) -> tuple[Line2D, str, float]:
    """Plot a single reference spectrum.

    The filename must be in space separated text formatted with the first two
    columns being wavelength in Angstroms, and F_lambda
    """
    filepath = find_reference_spectrum_file(filename)

    metadata = get_file_metadata(filepath)
    label = plotkwargs.get("label", metadata.get("label", filename))
    assert isinstance(label, str)
    plotkwargs.pop("label", None)

    print_heading(f"Reference spectrum '{label}'")
    specdata = atspectra.get_reference_spectrum(filepath)
    print_detail(f"file: {filepath}")

    # scale to flux at required distance
    if scale_to_dist_mpc:
        # scale to 1 Mpc and let get_dfspectrum_x_y_with_units scale to scale_to_dist_mpc later
        print(f"Scaling to distance {scale_to_dist_mpc} Mpc")
        assert metadata["dist_mpc"] > 0  # we must know the true distance in order to scale to some other distance
        specdata = specdata.with_columns(f_lambda=pl.col("f_lambda") * ((metadata["dist_mpc"]) ** 2))

    if scaletoreftime is not None:
        timefactor = atspectra.timeshift_fluxscale_co56law(scaletoreftime, float(metadata["t"]))
        print_detail(f"scaled from time {metadata['t']} to {scaletoreftime}, factor {timefactor} by the Co56 decay law")
        specdata = specdata.with_columns(f_lambda=pl.col("f_lambda") * timefactor)
        label += f" * {timefactor:.2f}"

    if "scale_factor" in metadata:
        specdata = specdata.with_columns(f_lambda=pl.col("f_lambda") * metadata["scale_factor"])

    if metadata.get("mask_telluric", False):
        print("Masking telluric regions")
        z = metadata["z"]
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

    lambda_min, lambda_max = atspectra.convert_xlimits_to_lambda_range(xmin, xmax, xunit)

    # the reported flux covers the range that the user asked for, thus it takes the rows inside it
    inrange = specdata.filter(pl.col("lambda_angstroms").is_between(lambda_min, lambda_max))
    atspectra.print_integrated_flux(inrange["f_lambda"], inrange["lambda_angstroms"])

    # the drawn line keeps the nearest row outside each bound, so that it reaches the edge of the axes
    # instead of stopping at the last point inside the range
    specdata = df_filter_minmax_bracketed(specdata, "lambda_angstroms", lambda_min, lambda_max).collect()

    if fluxfilterfunc:
        print_detail("applying the filter to the reference spectrum")
        specdata = specdata.with_columns(
            cs.starts_with("f_lambda").map_batches(fluxfilterfunc, return_dtype=pl.self_dtype())
        )

    specdata = atspectra.get_dfspectrum_x_y_with_units(
        specdata, xunit=xunit, yvariable=yvariable, fluxdistance_mpc=scale_to_dist_mpc
    ).collect()

    if scale_to_peak:
        specdata = specdata.with_columns(y=pl.col("y") / pl.col("y").max() * scale_to_peak + offset)
    else:
        assert offset == 0
    ymax = specdata["y"].max()
    assert isinstance(ymax, float)
    (lineplot,) = axis.plot(specdata["x"], specdata["y"], label=label, **plotkwargs)

    return lineplot, label, ymax


def plot_reference_spectrum_for_args(
    filename: Path | str,
    axis: mplax.Axes,
    args: argparse.Namespace,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None,
    scale_to_peak: float | None,
    offset: float = 0.0,
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
        **plotkwargs,
    )


def plot_filter_functions(axis: mplax.Axes) -> None:
    """Plot the UBVI filter transmission curves on a twinned y axis."""
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
        axis.plot(
            filter_data["lambda_angstroms"],
            filter_data["flux_normalised"],
            label=filter_name,
            color=colours[index],
            alpha=0.3,
        )


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
    xunit: str = "angstroms",
    **plotkwargs: t.Any,
) -> pl.DataFrame | None:
    """Plot an ARTIS output spectrum. The data plotted are also returned as a DataFrame."""
    modelpath = Path(modelpath)
    if modelpath.is_file():  # handle e.g. modelpath = 'modelpath/spec.out'
        print_warning(f"ignoring filename of {modelpath.name}")
        modelpath = get_model_folder(modelpath)

    if not modelpath.is_dir():
        print_warning(f"Skipping because {modelpath} does not exist")
        return None
    dfspectrum = None
    use_time: t.Literal["escape", "emission", "arrival"]
    if args.use_escapetime:
        use_time = "escape"
        assert from_packets
    elif args.use_emissiontime:
        use_time = "emission"
        assert from_packets
    else:
        use_time = "arrival"

    if directionbins is None:
        directionbins = [-1]

    if yvariable == "packetcount":
        from_packets = True

    for axindex, axis in enumerate(axes):
        assert isinstance(axis, mplax.Axes)
        clamp_to_timesteps = not args.notimeclamp
        if args.multispecplot:
            (timestepmin, timestepmax, args.timemin, args.timemax) = get_time_range(
                modelpath, timedays_range_str=args.timedayslist[axindex], clamp_to_timesteps=clamp_to_timesteps
            )
        else:
            (timestepmin, timestepmax, args.timemin, args.timemax) = get_time_range(
                modelpath,
                args.timestep,
                args.timemin,
                args.timemax,
                args.timedays,
                clamp_to_timesteps=clamp_to_timesteps,
            )

        if timestepmin == timestepmax == -1:
            return None

        assert args.timemin is not None
        assert args.timemax is not None
        timeavg = (args.timemin + args.timemax) / 2.0
        timedelta = (args.timemax - args.timemin) / 2
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
            f"'{plain_label(linelabel)}' timesteps {timestepmin} to {timestepmax} "
            f"({args.timemin:.3f} to {args.timemax:.3f}d"
            f"{'' if clamp_to_timesteps else ' not necessarily clamped to timestep start/end'})"
        )
        print_detail(f"modelpath: {modelpath}")

        check_time_range_is_valid(modelpath, args.timemin, args.timemax, args.plotinvalidpart)

        xmin, xmax = axis.get_xlim()
        if from_packets:
            lambda_bin_edges = atspectra.get_lambda_bin_edges(
                xmin,
                xmax,
                deltax=args.deltax,
                deltalogx=args.deltalogx,
                deltalambda=args.deltalambda,
                xunit=args.xunit,
                modelpath=modelpath,
                gamma=args.gamma,
            )

            viewinganglespectra = atspectra.get_from_packets(
                modelpath,
                timelowdays=args.timemin,
                timehighdays=args.timemax,
                lambda_bin_edges=lambda_bin_edges,
                use_time=use_time,
                maxpacketfiles=maxpacketfiles,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                fluxfilterfunc=filterfunc,
                directionbins_are_vpkt_observers=args.plotvspecpol is not None,
                gamma=args.gamma,
            )

        elif args.plotvspecpol is not None:
            # read virtual packet files (after running plotartisspectrum --makevspecpol)
            vpkt_config = get_vpkt_config(modelpath)
            if vpkt_config["time_limits_enabled"] and (
                args.timemin < vpkt_config["initial_time"] or args.timemax > vpkt_config["final_time"]
            ):
                print(
                    f"Timestep out of range of virtual packets: start time {vpkt_config['initial_time']} days "
                    f"end time {vpkt_config['final_time']} days"
                )
                sys.exit(1)

            viewinganglespectra = {
                dirbin: atspectra.get_vspecpol_spectrum(modelpath, timeavg, dirbin, args, fluxfilterfunc=filterfunc)
                for dirbin in directionbins
                if dirbin >= 0
            }
        else:
            viewinganglespectra = atspectra.get_spectra(
                modelpath=modelpath,
                timestepmin=timestepmin,
                timestepmax=timestepmax,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                fluxfilterfunc=filterfunc,
                gamma=args.gamma,
            )

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
                    atspectra.get_dfspectrum_x_y_with_units(
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
                dirbin, directionbins, dirbin_definitions, linelabel, linelabel_is_custom, plotkwargs
            )

            atspectra.print_integrated_flux(dfspectrum["dflux_on_dx_onempc"], dfspectrum["x"])

            if scale_to_peak:
                dfspectrum = dfspectrum.with_columns(y=pl.col("y") / pl.col("y").max() * scale_to_peak)

            if args.binflux:
                assert args.xunit.lower() == "angstroms"
                dfspectrum = (
                    atspectra
                    .bin_spectrum(dfspectrum, 5, "lambda_angstroms", "y")
                    .rename({"lambda_angstroms": "x"})
                    .with_columns(lambda_angstroms=pl.col("x"), f_lambda=pl.col("y"))
                )

            axis.plot(
                dfspectrum["x"], dfspectrum["y"], label=linelabel_withdirbin if axindex == 0 else None, **plotkwargs
            )

    return dfspectrum[["lambda_angstroms", "f_lambda"]] if dfspectrum is not None else None


def make_spectrum_plot(
    speclist: Sequence[Path | str],
    axes: npt.NDArray[np.object_] | Sequence[mplax.Axes],
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None,
    args: argparse.Namespace,
    scale_to_peak: float | None = None,
) -> pl.DataFrame:
    """Plot reference spectra and ARTIS spectra."""
    dfalldata = pl.DataFrame()
    artisindex = 0
    refspecindex = 0

    set_prop_cycle_unusedcolors(axes, args.color)
    for axis in axes:
        axis.margins(0.0, 0.0)

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
                plotkwargs["color"] = "k"
                plot_reference_spectrum_for_args(
                    specpath, axes[refspecindex], args, filterfunc, scale_to_peak, **plotkwargs
                )
            else:
                if args.label[seriesindex]:
                    plotkwargs["label"] = args.label[seriesindex]
                for axis in axes:
                    plot_reference_spectrum_for_args(specpath, axis, args, filterfunc, scale_to_peak, **plotkwargs)
            refspecindex += 1
        elif path_is_codecomparison(specpath):
            (_timestepmin, _timestepmax, args.timemin, args.timemax) = get_time_range(
                specpath, args.timestep, args.timemin, args.timemax, args.timedays
            )
            timeavg = args.timedays
            from artistools.codecomparison import plot_spectrum

            plot_spectrum(specpath, timedays=timeavg, axis=axes[0], **plotkwargs)
            refspecindex += 1
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
                    **plotkwargs,
                )
            except FileNotFoundError as e:
                print_warning(f"Skipping {specpath} because it does not exist ({e})")
                continue

            if seriesdata is not None:
                seriesname = get_model_name(specpath)
                artisindex += 1

        if args.write_data and seriesdata is not None:
            if dfalldata.is_empty():
                dfalldata = pl.DataFrame({"lambda_angstroms": seriesdata["lambda_angstroms"]})
            else:
                # make sure we can share the same set of wavelengths for this series
                assert np.allclose(dfalldata["lambda_angstroms"], seriesdata["lambda_angstroms"].to_numpy())
            dfalldata = dfalldata.with_columns(seriesdata["f_lambda"].alias(f"f_lambda.{seriesname}"))

    if artisindex == refspecindex == 0:
        exit_with_error(
            "no spectra were plotted. Check that each given path holds an ARTIS run or a reference spectrum"
        )

    for axis in axes:
        if args.showfilterfunctions:
            if not args.normalised:
                print_warning("the filter functions plot normalised values, thus give -normalised as well")
            plot_filter_functions(axis)

        # make_plot has already applied args.ymax, thus reading the top back would inflate the value
        # that the user asked for by five percent
        if args.stokesparam == "I" and not args.logscaley and args.ymax is None:
            # the axes carry no y margin, thus the top would sit on the tallest peak and clip it
            _, datatop = axis.get_ylim()
            axis.set_ylim(bottom=0.0, top=datatop * 1.05)

        set_plot_title(axis, args.title, args)

    return dfalldata


def get_xy_spectrum(
    flambda_array: npt.NDArray[np.floating], arraylambda_angstroms: npt.NDArray[np.floating], args: argparse.Namespace
) -> pl.LazyFrame:
    """Return the x series and the y series of one flux array, in the units that the arguments name."""
    return atspectra.get_dfspectrum_x_y_with_units(
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
    dirbin: int | None,
) -> tuple[list[atspectra.FluxContributionTuple], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return the flux contribution of each series, the total emitted flux, and the wavelength grid.

    A run with --frompackets reads the packets files. A run without it reads the emission file and the
    absorption file that ARTIS writes for each timestep.
    """
    if not args.frompackets:
        assert not args.vpkt_match_emission_exclusion_to_opac
        lambda_min, lambda_max = atspectra.convert_xlimits_to_lambda_range(xmin, xmax, args.xunit)

        return atspectra.get_flux_contributions(
            modelpath,
            filterfunc,
            timestepmin,
            timestepmax,
            getemission=args.showemission,
            getabsorption=args.showabsorption,
            use_lastemissiontype=not args.use_thermalemissiontype,
            directionbin=dirbin,
            average_over_phi=args.average_over_phi_angle,
            average_over_theta=args.average_over_theta_angle,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
        )

    use_time: t.Literal["escape", "emission", "arrival"]
    if args.use_escapetime:
        use_time = "escape"
    elif args.use_emissiontime:
        use_time = "emission"
    else:
        use_time = "arrival"

    if args.groupby in {"nuc", "nucmass"}:
        emtypecolumn = "pellet_nucindex"
    elif args.use_thermalemissiontype:
        emtypecolumn = "trueemissiontype"
    else:
        emtypecolumn = "emissiontype"

    lambda_bin_edges = atspectra.get_lambda_bin_edges(
        xmin,
        xmax,
        deltax=args.deltax,
        deltalogx=args.deltalogx,
        deltalambda=args.deltalambda,
        xunit=args.xunit,
        modelpath=modelpath,
        gamma=args.gamma,
    )

    return atspectra.get_flux_contributions_from_packets(
        modelpath,
        timelowdays=args.timemin,
        timehighdays=args.timemax,
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
        emtypecolumn=emtypecolumn,
        directionbin=dirbin,
        average_over_phi=args.average_over_phi_angle,
        average_over_theta=args.average_over_theta_angle,
        directionbins_are_vpkt_observers=args.plotvspecpol is not None,
        vpkt_match_emission_exclusion_to_opac=args.vpkt_match_emission_exclusion_to_opac,
    )


def collect_emission_and_absorption(
    contributions: "Sequence[atspectra.FluxContributionTuple]",
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
    contributions: Sequence[atspectra.FluxContributionTuple],
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

            # an x range that holds no bin gives None, thus the largest absorption stays where it was
            this_max_absorption = dfspec.filter(pl.col("x").is_between(xmin, xmax))["y"].max()
            if isinstance(this_max_absorption, float):
                max_absorption = max(max_absorption, this_max_absorption)

        plotobjects.append(mpatches.Patch(color=linecolor))

    return plotobjects, max_absorption


def plot_contributions_stacked(
    axis: mplax.Axes,
    contributions: Sequence[atspectra.FluxContributionTuple],
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
                f"y{i}": df.filter(pl.col("x").is_between(xmin, xmax)).get_column("y")
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
            if args.label[index] is not None:
                plotkwargs["label"] = args.label[index]
            plotkwargs["alpha"] = args.linealpha[index]

        plotobj, serieslabel, ymaxref = plot_reference_spectrum_for_args(
            filepath, axis, args, filterfunc, scale_to_peak, offset=0.3 if scale_to_peak else 0.0, **plotkwargs
        )
        ymaxrefall = max(ymaxrefall, ymaxref)

        plotobjects.append(plotobj)
        plotobjectlabels.append(serieslabel)

    return plotobjects, plotobjectlabels, ymaxrefall


def get_emission_plot_label(modelpath: Path, args: argparse.Namespace, modelname: str, dirbin: int | None) -> str:
    """Return the title of the plot, which names the model, the time range, and the direction bin."""
    if args.title:
        return str(args.title)

    plotlabel = f"{modelname} [{args.timemin:.2f}d to {args.timemax:.2f}d]"
    if not (args.plotviewingangle or args.plotvspecpol):
        return plotlabel

    assert dirbin is not None
    dirbin_definitions = get_dirbin_definitions(
        modelpath,
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

    print_heading(modelname)
    clamp_to_timesteps = not args.notimeclamp

    (timestepmin, timestepmax, args.timemin, args.timemax) = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays, clamp_to_timesteps=clamp_to_timesteps
    )

    if timestepmin == timestepmax == -1:
        print(f"Can't plot {modelname}...skipping")
        return [], [], pl.DataFrame()

    check_time_range_is_valid(modelpath, args.timemin, args.timemax, args.plotinvalidpart)

    if args.plotvspecpol and not args.frompackets:
        args.frompackets = True
        print("Enabling --frompackets, since --plotvspecpol was specified")

    if args.gamma and not args.frompackets:
        args.frompackets = True
        print("Enabling --frompackets, since --gamma and --showemission were specified")

    if args.groupby is None:
        args.groupby = "nuc" if args.gamma else "ion"

    assert args.timemin is not None
    assert args.timemax is not None

    print(
        f"Plotting {modelname} timesteps {timestepmin} to {timestepmax} ({args.timemin:.3f} to {args.timemax:.3f}d{'' if clamp_to_timesteps else ' not necessarily clamped to timestep start/end'})"
    )

    xmin, xmax = axis.get_xlim()

    dirbin = args.plotviewingangle[0] if args.plotviewingangle else args.plotvspecpol[0] if args.plotvspecpol else None

    contribution_list, array_flambda_emission_total, arraylambda_angstroms = get_emission_contributions(
        modelpath, args, filterfunc, xmin, xmax, timestepmin, timestepmax, dirbin
    )

    atspectra.print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    contributions_sorted_reduced = atspectra.sort_and_reduce_flux_contribution_list(
        contribution_list,
        args.maxseriescount,
        arraylambda_angstroms,
        fixedionlist=args.fixedionlist,
        hideother=args.hideother,
    )

    plotobjectlabels: list[str] = []
    plotobjects: list[Artist] = []

    dfspectotal = get_xy_spectrum(array_flambda_emission_total, arraylambda_angstroms, args).collect()

    max_f_emission_total = dfspectotal.filter(pl.col("x").is_between(xmin, xmax))["y"].max()
    assert isinstance(max_f_emission_total, (float, np.floating))
    max_f_emission_total = float(max_f_emission_total)

    scalefactor = scale_to_peak / max_f_emission_total if scale_to_peak else 1.0

    if not args.hidenetspectrum:
        plotobjectlabels.append("Spectrum")
        (line,) = axis.plot(dfspectotal["x"], dfspectotal["y"] * scalefactor, linewidth=1.5, color="black", zorder=100)
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

    set_plot_title(axis, get_emission_plot_label(modelpath, args, modelname, dirbin), args)

    if args.ymax is None:
        axis.set_ylim(top=max(ymaxrefall, scalefactor * max_f_emission_total * 1.2))

    if args.ymin is None:
        axis.set_ylim(bottom=-scalefactor * max_absorption * 1.2)

    return plotobjects, plotobjectlabels, dfaxisdata


def make_plot(args: argparse.Namespace) -> tuple[mplfig.Figure, npt.NDArray[np.object_], pl.DataFrame]:
    """Plot the spectra selected by args, and return the figure, the axes, and the plotted data."""
    nrows = len(args.timedayslist) if args.multispecplot else 1

    # an emission and absorption plot draws a taller frame
    aspect = FRAMEHEIGHT_INCHES / FRAMEWIDTH_INCHES * (1.56 if args.showabsorption else 1.0)
    fig, axesgrid = make_frame_figure(args, rows=nrows, aspect=aspect, sharex=True, sharey=False)

    axes = axesgrid[:, 0]
    assert isinstance(axes, np.ndarray)

    filterfunc = get_filterfunc(args)

    scale_to_peak = 1.0 if args.normalised else None

    xlabel, ylabel = get_axis_labels(args)

    if args.normalised and args.ymax is None:
        args.ymax = 1.10

    # make_emissionabsorption_plot reads the x range back from the axes, thus the scales and the
    # limits go on before the plot calls draw the data
    set_axis_properties(axes, args)
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

    if args.showemission or args.showabsorption:
        legendncol = 2
        defaultoutputfile = Path("plotspectra_emission_{timemin:.2f}d-{timemax:.2f}d{directionbins}.pdf")
        plotobjects, plotobjectlabels, dfalldata = make_emissionabsorption_plot(
            modelpath=Path(args.specpath[0]),
            axis=axes[-1],
            filterfunc=filterfunc,
            args=args,
            scale_to_peak=scale_to_peak,
        )
    else:
        legendncol = 1
        defaultoutputfile = Path("plotspectra_{timemin:.2f}d-{timemax:.2f}d.pdf")

        if args.multispecplot:
            dfalldata = make_spectrum_plot(args.specpath, axes, filterfunc, args, scale_to_peak=scale_to_peak)
            plotobjects, plotobjectlabels = axes[0].get_legend_handles_labels()
        else:
            dfalldata = make_spectrum_plot(args.specpath, [axes[-1]], filterfunc, args, scale_to_peak=scale_to_peak)
            plotobjects, plotobjectlabels = axes[-1].get_legend_handles_labels()

    # the annotation comes after the plot calls, because those calls resolve args.timemin and
    # args.timemax when the command line gave the time as -timedays or -timestep
    if args.showtime:
        for index, axis in enumerate(axes):
            if args.multispecplot:
                _ymin, ymax = axis.get_ylim()
                axis.text(5500, ymax * 0.9, f"{args.timedayslist[index]} days")  # multispecplot text
            else:
                timeavg = (args.timemin + args.timemax) / 2.0
                axis.annotate(
                    f"{timeavg:.2f} days",
                    xy=(0.03, 0.97),
                    xycoords="axes fraction",
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize="x-large",
                )

    # the loop above sets the scale before the data exists, because make_emissionabsorption_plot reads
    # the x range back from the axes. Thus -yscale auto reads the values here and sets the scale itself
    set_auto_yscale(list(axes), args)

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
        ncol=legendncol,
        numpoints=1,
        columnspacing=1.0,
    )

    if leg is not None:
        leg.set_zorder(200)

        # colour each legend label like the line or the patch that it names
        for artist, text in zip(leg.legend_handles, leg.get_texts(), strict=False):
            if artist is None:
                continue

            if hasattr(artist, "get_color") and hasattr(artist, "set_linewidth"):
                col = artist.get_color()  # ty:ignore[call-non-callable]
                artist.set_linewidth(2.0)  # ty:ignore[call-non-callable]
            elif hasattr(artist, "get_facecolor"):
                col = artist.get_facecolor()  # ty:ignore[call-non-callable]
            else:
                continue

            if isinstance(col, np.ndarray):
                col = col[0]
            text.set_color(col)

    args.outputfile = resolve_outputfile(args.outputfile, defaultoutputfile)

    return fig, axes, dfalldata


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument(
        "specpath",
        default=[],
        nargs="*",
        type=Path,
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
        default=14,
        help="Maximum number of plot series (ions/processes) for emission/absorption plot",
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
        "--notimeclamp", action="store_true", help="When plotting from packets, don't clamp to timestep start/end"
    )

    parser.add_argument(
        "-xunit",
        dest="xunit",
        default=None,
        type=atspectra.parse_xunit_argument,
        help="X (horizontal) axis unit, e.g. angstrom, nm, micron, Hz, keV, MeV",
    )
    # deprecated spellings kept as hidden aliases. -x names the axis variable on plotestimators, but
    # each parser reads its own arguments, and a script holds the -x of this command.
    parser.add_argument("-xunits", dest="xunit", type=atspectra.parse_xunit_argument, help=argparse.SUPPRESS)
    parser.add_argument("-x", dest="xunit", type=atspectra.parse_xunit_argument, help=argparse.SUPPRESS)

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
        type=float,
        default=None,
        help="Horizontal bin size factor x[1] = x[0] * (1 + dlogx) (applies to from_packets only)",
    )

    parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument(
        "--hidemodeltimerange", action="store_true", help='Hide the "at (+/- x.xd)" from the line labels'
    )

    parser.add_argument("--hidemodeltime", action="store_true", help="Hide the time from the line labels")

    parser.add_argument("--normalised", action="store_true", help="Normalise all spectra to their peak values")

    timegroup = parser.add_mutually_exclusive_group()

    timegroup.add_argument(
        "--use_escapetime",
        action="store_true",
        help="Use the time of packet escape to the surface (instead of a plane toward the observer)",
    )

    timegroup.add_argument("--use_emissiontime", action="store_true", help="Use the time of packet last emission")

    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Tag packets by their last thermal emission type rather than their last emission process",
    )

    parser.add_argument(
        "-groupby",
        default=None,
        choices=["ion", "line", "nuc", "nucmass"],
        help="Use a different color for each ion or line when using --showemission. groupby='line', 'nuc', 'nucmass' imply --frompackets",
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

    addarg_nolegend(parser)

    parser.add_argument("--reverselegendorder", action="store_true", help="Reverse the order of legend items")

    parser.add_argument("--hidexticklabels", action="store_true", help="Don't show numbers or a label on the x axis")

    parser.add_argument("--hideyticklabels", action="store_true", help="Don't show numbers or a label on the y axis")

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a CSV file")

    addarg_output(parser, kind="file", helptext="Path/filename for PDF file")

    addarg_dpi(parser)

    addarg_show(parser)
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
        "-stokesparam", type=str, default="I", help="Stokes param to plot. Default I. Expects I, Q or U"
    )

    parser.add_argument("--binflux", action="store_true", help="Bin flux over wavelength and average flux")

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
        "--classicartis", action="store_true", help="Flag to show using output from classic ARTIS branch"
    )

    parser.add_argument(
        "--vpkt_match_emission_exclusion_to_opac",
        action="store_true",
        help="Exclude packets with emission type no-bb/no-bf/no-(element) matching the vpkt opacity exclusion",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot spectra from ARTIS and reference data."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if getattr(args, "average_every_tenth_viewing_angle", False):
        print_warning("--average_every_tenth_viewing_angle is deprecated. use --average_over_phi_angle instead")
        args.average_over_phi_angle = True

    if args.xunit is None:
        args.xunit = "kev" if args.gamma else "angstroms"
    args.xunit = atspectra.convert_xunit_aliases_to_canonical(args.xunit)

    if args.xmin is None:
        args.xmin = atspectra.convert_angstroms_to_unit(0.2 if args.gamma else 2500.0, args.xunit)
    if args.xmax is None:
        args.xmax = atspectra.convert_angstroms_to_unit(0.004 if args.gamma else 19000.0, args.xunit)

    args.xmin, args.xmax = sorted([args.xmin, args.xmax])

    assert (
        not args.plotvspecpol or not args.plotviewingangle
    )  # choose either virtual packet directions or real packet direction bins

    # -obsspec named a reference spectrum before a positional path could name one. Nothing read that
    # list, thus the command drew the model alone and said nothing. The paths join the positional ones
    args.specpath = normalize_path_list([*makelist(args.specpath), *(args.refspecfiles or [])])

    if args.timedayslist:
        args.multispecplot = True
        args.timedays = args.timedayslist[0]

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

    if args.vpkt_match_emission_exclusion_to_opac:
        assert args.showemission
        assert args.frompackets
        assert args.plotvspecpol

    if args.groupby is not None:
        args.showemission = True

    if args.groupby in {"line", "nuc", "nucmass"}:
        args.frompackets = True

    if args.gamma and args.plotviewingangle:
        # exspec does not generate angle-resolved gamma spectra files,
        # so we need to use the packets instead
        args.frompackets = True

    if args.use_emissiontime or args.use_escapetime:
        # exspec spectra are binned by arrival time at the observer
        # so we need to use the packets instead
        args.frompackets = True

    if not args.frompackets and any(x is not None for x in (args.deltax, args.deltalogx, args.deltalambda)):
        args.frompackets = True
        print("Enabling --frompackets, since custom bin width was specified")

    if args.makevspecpol:
        atspectra.make_virtual_spectra_summed_file(args.specpath[0])
        return

    if args.averagevspecpolfiles:
        atspectra.make_averaged_vspecfiles(args.specpath)
        return

    if "/" in args.stokesparam:
        plot_polarisation(args.specpath[0], args)
        return

    if args.output_spectra:
        for modelpath in args.specpath:
            write_flambda_spectra(modelpath)

    else:
        if args.emissionabsorption:
            args.showemission = True
            args.showabsorption = True

        fig, _axes, dfalldata = make_plot(args)

        strdirectionbins = (
            "_direction" + "_".join([f"{angle:02d}" for angle in args.plotviewingangle])
            if args.plotviewingangle
            else ""
        )

        filenameout = str(args.outputfile)
        if args.timemin is not None:
            filenameout = filenameout.format(timemin=args.timemin, timemax=args.timemax, directionbins=strdirectionbins)
        elif "{" in filenameout:
            # no global time range was resolved (e.g. --multispecplot), so the time placeholders can't be filled
            filenameout = str(Path(filenameout).with_name("plotspectra.pdf"))

        if args.write_data and len(dfalldata.columns) > 0:
            datafilenameout = Path(filenameout).with_suffix(".txt")
            dfalldata.write_csv(datafilenameout, separator=" ")
            print_saved(datafilenameout)

        save_figure(fig, filenameout, args=args, dpi=args.dpi)


if __name__ == "__main__":
    run_subcommand("plotspectra")
