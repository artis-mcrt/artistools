#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Artistools - spectra plotting functions."""

import argparse
import math
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import matplotlib.artist as mplartist
import matplotlib.axes as mplax
import matplotlib.figure as mplfig
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
from matplotlib import ticker
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

import artistools.spectra as atspectra
from artistools.configuration import get_config
from artistools.inputmodel import get_modeldata_pandas
from artistools.misc import AppendPath
from artistools.misc import CustomArgHelpFormatter
from artistools.misc import flatten_list
from artistools.misc import get_dirbin_labels
from artistools.misc import get_filterfunc
from artistools.misc import get_model_name
from artistools.misc import get_nu_grid
from artistools.misc import get_time_range
from artistools.misc import get_vpkt_config
from artistools.misc import get_vspec_dir_labels
from artistools.misc import print_theta_phi_definitions
from artistools.misc import set_args_from_dict
from artistools.misc import trim_or_pad
from artistools.plottools import ExponentLabelFormatter
from artistools.plottools import set_mpl_style

hatchestypes = ["", "x", "-", "\\", "+", "O", ".", "", "x", "*", "\\", "+", "O", "."]  # ,


def path_is_artis_model(filepath: str | Path) -> bool:
    if Path(filepath).name.endswith(".out.zst"):
        return True
    return True if Path(filepath).suffix == ".out" else Path(filepath).is_dir()


def get_lambda_range_binsize(
    xmin: float, xmax: float, args: argparse.Namespace
) -> tuple[float, float, float | npt.NDArray[np.floating]]:
    lambda_min, lambda_max = sorted([
        atspectra.convert_unit_to_angstroms(xmin, args.xunit),
        atspectra.convert_unit_to_angstroms(xmax, args.xunit),
    ])
    if args.deltax is not None or args.deltalogx is not None:
        if args.deltalogx is not None:
            x = xmin
            list_x_bin_edges = [xmin]
            steps = 0
            while x <= xmax:
                x *= 1 + args.deltalogx
                list_x_bin_edges.append(x)
                steps += 1
            x_bin_edges = np.array(list_x_bin_edges)
        else:
            x_bin_edges = np.arange(xmin, xmax + args.deltax, args.deltax)
        lambda_bin_edges = np.array(
            sorted(atspectra.convert_unit_to_angstroms(float(x), args.xunit) for x in x_bin_edges)
        )
        delta_lambda = np.array([
            (lambda_bin_edges[i + 1] - lambda_bin_edges[i]) for i in range(len(lambda_bin_edges) - 1)
        ])
    else:
        delta_lambda = args.deltalambda
    return lambda_min, lambda_max, delta_lambda


def get_axis_labels(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Get the x-axis and y-axis labels based on the arguments."""
    match args.xunit.lower():
        case "angstroms":
            xtype = "Wavelength"
            str_xunit = "Å"
        case "nm":
            xtype = "Wavelength"
            str_xunit = "nm"
        case "micron":
            xtype = "Wavelength"
            str_xunit = "μm"
        case "hz":
            xtype = "Frequency"
            str_xunit = "Hz"
        case "ev":
            xtype = "Energy"
            str_xunit = "eV"
        case "kev":
            xtype = "Energy"
            str_xunit = "keV"
        case "mev":
            xtype = "Energy"
            str_xunit = "MeV"
        case _:
            msg = f"Unknown x-axis unit {args.xunit}"
            raise AssertionError(msg)

    xlabel = None if args.hidexticklabels else f"{xtype} " + r"$\left[\mathrm{{" + str_xunit + r"}}\right]$"

    ylabel = None
    if not args.hideyticklabels:
        if args.normalised:
            match args.yvariable:
                case "flux":
                    ylabel = r"Scaled F$_\lambda$"
                case "packetcount":
                    ylabel = r"Scaled Monte Carlo packets"
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
                    if xtype == "Wavelength":
                        ylabel = rf"F$_\lambda$ at {strdist} [{{}}erg/s/cm$^2$/{str_xunit}]"
                    elif xtype == "Frequency":
                        ylabel = rf"F$_\nu$ at {strdist} [{{}}erg/s/cm$^2$/{str_xunit}]"
                    elif xtype == "Energy":
                        ylabel = f"dF/dE at {strdist} [{{}}erg/s/cm$^2$/{str_xunit}]"
                case "packetcount":
                    ylabel = r"{}Monte Carlo packets per bin"
                case "eflux":
                    ylabel = f"E$^2$ flux at {strdist} [{{}}{str_xunit}/s/cm$^2$]"
                case "photoncount":
                    ylabel = f"Photon flux at {strdist} [{{}}/s/cm$^2$/{str_xunit}]"
                case _:
                    msg = f"Unknown y-variable {args.yvariable}"
                    raise AssertionError(msg)

        assert ylabel is not None
        if args.logscaley:
            # don't include the {} that will be replaced with the power of 10 by the custom formatter
            ylabel = ylabel.replace("{}", "")

    return xlabel, ylabel


def plot_polarisation(modelpath: Path, args: argparse.Namespace) -> None:
    angle = args.plotviewingangle[0]
    dfspectrum = (
        atspectra.get_specpol_data(angle=angle, modelpath=modelpath)[args.stokesparam]
        .with_columns(lambda_angstroms=2.99792458e18 / pl.col("nu"))
        .to_pandas(use_pyarrow_extension_array=True)
    )

    timearray = dfspectrum.keys()[1:-1]
    (_, _, args.timemin, args.timemax) = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )
    assert args.timemin is not None
    assert args.timemax is not None

    timeavg_float = (args.timemin + args.timemax) / 2.0
    timeavg = f"{min((float(x) for x in timearray), key=lambda x: abs(x - timeavg_float)):.4f}"

    filterfunc = get_filterfunc(args)
    if filterfunc is not None:
        print("Applying filter to ARTIS spectrum")
        dfspectrum[timeavg] = filterfunc(dfspectrum[timeavg])

    vpkt_config = get_vpkt_config(modelpath)

    linelabel = (
        f"{timeavg} days, cos($\\theta$) = {vpkt_config['cos_theta'][angle[0]]}"
        if args.plotvspecpol
        else f"{timeavg} days"
    )

    if args.binflux:
        new_lambda_angstroms = []
        binned_flux = []

        wavelengths = dfspectrum["lambda_angstroms"]
        fluxes = dfspectrum[timeavg]
        nbins = 5

        for i in np.arange(0, len(wavelengths - nbins), nbins, dtype=int):
            new_lambda_angstroms.append(wavelengths[i + nbins // 2])
            sum_flux = sum(fluxes[j] for j in range(i, i + nbins))
            binned_flux.append(sum_flux / nbins)

        plt.plot(new_lambda_angstroms, binned_flux)
    else:
        dfspectrum.plot(x="lambda_angstroms", y=timeavg, label=linelabel)

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

    plt.ylim(args.ymin, args.ymax)
    plt.xlim(args.xmin, args.xmax)

    plt.ylabel(str(args.stokesparam))
    plt.xlabel(r"Wavelength ($\mathrm{{\AA}}$)")
    figname = f"plotpol_{timeavg}_days_{args.stokesparam.split('/')[0]}_{args.stokesparam.split('/')[1]}.pdf"
    plt.savefig(modelpath / figname, format="pdf")
    print(f"Saved {figname}")


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
    specdata, metadata = atspectra.get_reference_spectrum(filename)
    label = plotkwargs.get("label", metadata.get("label", filename))
    assert isinstance(label, str)
    plotkwargs.pop("label", None)

    # scale to flux at required distance
    if scale_to_dist_mpc:
        # scale to 1 Mpc and let get_dfspectrum_x_y_with_units scale to scale_to_dist_mpc later
        print(f"Scale to {scale_to_dist_mpc} Mpc")
        assert metadata["dist_mpc"] > 0  # we must know the true distance in order to scale to some other distance
        specdata = specdata.with_columns(f_lambda=pl.col("f_lambda") * ((metadata["dist_mpc"]) ** 2))

    if scaletoreftime is not None:
        timefactor = atspectra.timeshift_fluxscale_co56law(scaletoreftime, float(metadata["t"]))
        print(f" Scale from time {metadata['t']} to {scaletoreftime}, factor {timefactor} using Co56 decay law")
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

    print(f"Reference spectrum '{label}' has {len(specdata)} points in the plot range")
    print(f" file: {filename}")

    print(" metadata: " + ", ".join([f"{k}='{v}'" if hasattr(v, "lower") else f"{k}={v}" for k, v in metadata.items()]))

    lambda_min, lambda_max = sorted([
        atspectra.convert_unit_to_angstroms(xmin, xunit),
        atspectra.convert_unit_to_angstroms(xmax, xunit),
    ])
    print(f" lambda_min {lambda_min:.1f} lambda_max {lambda_max:.1f}")

    specdata = specdata.filter(pl.col("lambda_angstroms").is_between(lambda_min, lambda_max))

    atspectra.print_integrated_flux(specdata["f_lambda"], specdata["lambda_angstroms"])

    if fluxfilterfunc:
        print(" applying filter to reference spectrum")
        specdata = specdata.with_columns(cs.starts_with("f_lambda").map_batches(fluxfilterfunc))

    specdata = atspectra.get_dfspectrum_x_y_with_units(
        specdata, xunit=xunit, yvariable=yvariable, fluxdistance_mpc=scale_to_dist_mpc
    )

    if scale_to_peak:
        specdata = specdata.with_columns(
            y_scaled=pl.col("y") / pl.col("y").max() * scale_to_peak + offset
        ).with_columns(y=pl.col("y_scaled"))
    else:
        assert offset == 0

    ymax = specdata["y"].max()
    assert isinstance(ymax, float)
    (lineplot,) = axis.plot(specdata["x"], specdata["y"], label=label, **plotkwargs)

    return lineplot, label, ymax


def plot_filter_functions(axis: mplax.Axes) -> None:
    import pandas as pd

    filter_names = ["U", "B", "V", "I"]
    colours = ["r", "b", "g", "c", "m"]

    filterdir = Path(get_config()["path_artistools_dir"], "data/filters/")
    for index, filter_name in enumerate(filter_names):
        filter_data = pd.read_csv(
            filterdir / f"{filter_name}.txt",
            sep=r"\s+",
            header=None,
            skiprows=4,
            names=["lamba_angstroms", "flux_normalised"],
        )
        filter_data.plot(
            x="lamba_angstroms", y="flux_normalised", ax=axis, label=filter_name, color=colours[index], alpha=0.3
        )


def plot_artis_spectrum(
    axes: npt.NDArray[t.Any] | Sequence[mplax.Axes],
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
    if Path(modelpath).is_file():  # handle e.g. modelpath = 'modelpath/spec.out'
        specfilename = Path(modelpath).parts[-1]
        print(f"WARNING: ignoring filename of {specfilename}")
        modelpath = Path(modelpath).parent

    if not modelpath.is_dir():
        print(f"\nWARNING: Skipping because {modelpath} does not exist\n")
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

        modelname = get_model_name(modelpath)
        if timestepmin == timestepmax == -1:
            return None

        assert args.timemin is not None
        assert args.timemax is not None
        timeavg = (args.timemin + args.timemax) / 2.0
        timedelta = (args.timemax - args.timemin) / 2
        linelabel_is_custom = linelabel is not None
        if linelabel is None:
            linelabel = modelname if len(modelname) < 70 else f"...{modelname[-67:]}"

            if not args.hidemodeltime and not args.multispecplot:
                # TODO: fix this for multispecplot - use args.showtime for now
                linelabel += f" +{timeavg:.1f}d"
            if not args.hidemodeltimerange and not args.multispecplot and timedelta >= 0.1:
                linelabel += rf" ($\pm$ {timedelta:.1f}d)"

        print(
            f"====> '{linelabel}' timesteps {timestepmin} to {timestepmax} ({args.timemin:.3f} to {args.timemax:.3f}d{'' if clamp_to_timesteps else ' not necessarily clamped to timestep start/end'})"
        )
        print(f" modelpath {modelpath}")

        viewinganglespectra = {}

        xmin, xmax = axis.get_xlim()
        if from_packets:
            lambda_min, lambda_max, delta_lambda = get_lambda_range_binsize(xmin, xmax, args)

            viewinganglespectra = atspectra.get_from_packets(
                modelpath,
                timelowdays=args.timemin,
                timehighdays=args.timemax,
                lambda_min=lambda_min * 0.9,
                lambda_max=lambda_max * 1.1,
                use_time=use_time,
                maxpacketfiles=maxpacketfiles,
                delta_lambda=delta_lambda,
                directionbins=directionbins,
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
            viewinganglespectra = atspectra.get_spectrum(
                modelpath=modelpath,
                directionbins=directionbins,
                timestepmin=timestepmin,
                timestepmax=timestepmax,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                fluxfilterfunc=filterfunc,
                gamma=args.gamma,
            )

        dirbin_definitions = (
            get_vspec_dir_labels(modelpath=modelpath, usedegrees=usedegrees)
            if args.plotvspecpol
            else get_dirbin_labels(
                dirbins=directionbins,
                modelpath=modelpath,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                usedegrees=usedegrees,
            )
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

        for dirbin in directionbins:
            if len(directionbins) > 1 and dirbin != directionbins[0]:
                # only one colour was specified, but we have multiple direction bins
                # to zero out all but the first one
                plotkwargs = plotkwargs.copy()
                plotkwargs["color"] = None

            dfspectrum = viewinganglespectra[dirbin].lazy().collect()

            linelabel_withdirbin = linelabel
            if dirbin != -1:
                print(f" direction {dirbin:4d}  {dirbin_definitions[dirbin]}")
                if len(directionbins) > 1 or not linelabel_is_custom:
                    linelabel_withdirbin = f"{linelabel} {dirbin_definitions[dirbin]}"

            dfspectrum = atspectra.get_dfspectrum_x_y_with_units(
                dfspectrum, xunit=xunit, yvariable=yvariable, fluxdistance_mpc=args.distmpc
            )

            dfspectrum = dfspectrum.filter(pl.col("x").is_between(xmin * 0.9, xmax * 1.1)).sort(
                "x", maintain_order=True
            )

            atspectra.print_integrated_flux(dfspectrum["yflux"], dfspectrum["x"])

            if scale_to_peak:
                dfspectrum = dfspectrum.with_columns(
                    y_scaled=pl.col("y") / pl.col("y").max() * scale_to_peak
                ).with_columns(y=pl.col("y_scaled"))

            if args.binflux:
                new_lambda_angstroms = []
                binned_flux = []
                assert args.xunit.lower() == "angstroms"

                wavelengths = dfspectrum["lambda_angstroms"]
                fluxes = dfspectrum["y"]
                nbins = 5

                for i in np.arange(0, len(wavelengths - nbins), nbins, dtype=int):
                    i_max = min(i + nbins, len(wavelengths))
                    ncontribs = i_max - i
                    sum_lambda = sum(wavelengths[j] for j in range(i, i_max))
                    new_lambda_angstroms.append(sum_lambda / ncontribs)
                    sum_flux = sum(fluxes[j] for j in range(i, i_max))
                    binned_flux.append(sum_flux / ncontribs)

                dfspectrum = pl.DataFrame({"x": new_lambda_angstroms, "y": binned_flux})

            axis.plot(
                dfspectrum["x"], dfspectrum["y"], label=linelabel_withdirbin if axindex == 0 else None, **plotkwargs
            )

    return dfspectrum[["lambda_angstroms", "f_lambda"]] if dfspectrum is not None else None


def make_spectrum_plot(
    speclist: Sequence[Path | str],
    axes: npt.NDArray[t.Any] | Sequence[mplax.Axes],
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None,
    args: argparse.Namespace,
    scale_to_peak: float | None = None,
) -> pl.DataFrame:
    """Plot reference spectra and ARTIS spectra."""
    dfalldata = pl.DataFrame()
    artisindex = 0
    refspecindex = 0
    seriesindex = 0

    # take any specified colours our of the cycle
    colors = [
        color for i, color in enumerate(plt.rcParams["axes.prop_cycle"].by_key()["color"]) if f"C{i}" not in args.color
    ]
    for axis in axes:
        axis.set_prop_cycle(color=colors)

    for seriesindex, specpath in enumerate(speclist):
        specpath = Path(specpath)
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

        seriesdata: pl.DataFrame | None
        if (
            Path(specpath).is_file()
            or Path(get_config()["path_artistools_dir"], "data", "refspectra", specpath).is_file()
            or Path(get_config()["path_artistools_dir"], "data", "refspectra", f"{specpath!s}.xz").is_file()
            or Path(get_config()["path_artistools_dir"], "data", "refspectra", f"{specpath!s}.zstd").is_file()
        ):
            # reference spectrum
            if "linewidth" not in plotkwargs:
                plotkwargs["linewidth"] = 1.1

            if args.multispecplot:
                plotkwargs["color"] = "k"
                supxmin, supxmax = axes[refspecindex].get_xlim()
                plot_reference_spectrum(
                    filename=specpath,
                    axis=axes[refspecindex],
                    xmin=supxmin,
                    xmax=supxmax,
                    xunit=args.xunit,
                    yvariable=args.yvariable,
                    fluxfilterfunc=filterfunc,
                    scale_to_peak=scale_to_peak,
                    scale_to_dist_mpc=args.distmpc,
                    scaletoreftime=args.scaletoreftime,
                    **plotkwargs,
                )
            else:
                if args.label[seriesindex]:
                    plotkwargs["label"] = args.label[seriesindex]
                for axis in axes:
                    supxmin, supxmax = axis.get_xlim()
                    plot_reference_spectrum(
                        filename=specpath,
                        axis=axis,
                        xmin=supxmin,
                        xmax=supxmax,
                        xunit=args.xunit,
                        yvariable=args.yvariable,
                        fluxfilterfunc=filterfunc,
                        scale_to_peak=scale_to_peak,
                        scale_to_dist_mpc=args.distmpc,
                        scaletoreftime=args.scaletoreftime,
                        **plotkwargs,
                    )
            refspecindex += 1
        elif not specpath.exists() and specpath.parts[0] == "codecomparison":
            # timeavg = (args.timemin + args.timemax) / 2.
            (_timestepmin, _timestepmax, args.timemin, args.timemax) = get_time_range(
                specpath, args.timestep, args.timemin, args.timemax, args.timedays
            )
            timeavg = args.timedays
            from artistools.codecomparison import plot_spectrum

            plot_spectrum(specpath, timedays=timeavg, axis=axes[0], **plotkwargs)
            refspecindex += 1
        else:
            # ARTIS model spectrum
            # plotkwargs['dash_capstyle'] = dash_capstyleList[artisindex]
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
                print(f"WARNING: Skipping {specpath} because it does not exist")
                print(e)
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

    plottedsomething = artisindex > 0 or refspecindex > 0
    assert plottedsomething

    for axis in axes:
        if args.showfilterfunctions:
            if not args.normalised:
                print("Use args.normalised")
            plot_filter_functions(axis)

        # H = 6.6260755e-27  # Planck constant [erg s]
        # KB = 1.38064852e-16  # Boltzmann constant [erg/K]

        # for temp in [2900]:
        #     bbspec_lambda = np.linspace(3000, 25000, num=1000)
        #     bbspec_nu_hz = 2.99792458e18 / bbspec_lambda
        #     bbspec_j_nu = np.array(
        #         [1.4745007e-47 * pow(nu_hz, 3) * 1.0 / (math.expm1(H * nu_hz / temp / KB)) for nu_hz in bbspec_nu_hz]
        #     )

        #     arr_j_lambda = bbspec_j_nu * bbspec_nu_hz / bbspec_lambda
        #     bbspec_y = arr_j_lambda * 6e-14 / arr_j_lambda.max()
        #     axis.plot(
        #         bbspec_lambda,
        #         bbspec_y,
        #         label=f"{temp}K Planck function (scaled)",
        #         color="black",
        #         alpha=0.5,
        #         zorder=-1,
        #     )

        if args.stokesparam == "I" and not args.logscaley:
            axis.set_ylim(bottom=0.0)

        if not args.notitle and args.title:
            if args.inset_title:
                axis.annotate(
                    args.title,
                    xy=(0.03, 0.97),
                    xycoords="axes fraction",
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize="x-large",
                )
            else:
                axis.set_title(args.title, fontsize=11)

    return dfalldata


def make_emissionabsorption_plot(
    modelpath: Path,
    axis: mplax.Axes,
    args: argparse.Namespace,
    filterfunc: Callable[[npt.NDArray[np.floating] | pl.Series], npt.NDArray[np.floating]] | None = None,
    scale_to_peak: float | None = None,
) -> tuple[list[Artist], list[str], pl.DataFrame | None]:
    """Plot the emission and absorption contribution spectra, grouped by ion/line/term for an ARTIS model."""
    modelname = get_model_name(modelpath)

    print(f"====> {modelname}")
    clamp_to_timesteps = not args.notimeclamp

    (timestepmin, timestepmax, args.timemin, args.timemax) = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays, clamp_to_timesteps=clamp_to_timesteps
    )

    if timestepmin == timestepmax == -1:
        print(f"Can't plot {modelname}...skipping")
        return [], [], None

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
    if args.frompackets:
        if args.groupby in {"nuc", "nucmass"}:
            emtypecolumn = "pellet_nucindex"
        elif args.use_thermalemissiontype:
            emtypecolumn = "trueemissiontype"
        else:
            emtypecolumn = "emissiontype"

        lambda_min, lambda_max, delta_lambda = get_lambda_range_binsize(xmin, xmax, args)

        (contribution_list, array_flambda_emission_total, arraylambda_angstroms) = (
            atspectra.get_flux_contributions_from_packets(
                modelpath,
                timelowdays=args.timemin,
                timehighdays=args.timemax,
                lambda_min=lambda_min,
                lambda_max=lambda_max,
                delta_lambda=delta_lambda,
                getemission=args.showemission,
                getabsorption=args.showabsorption,
                maxpacketfiles=args.maxpacketfiles,
                filterfunc=filterfunc,
                groupby=args.groupby,
                fixedionlist=args.fixedionlist,
                maxseriescount=args.maxseriescount + 20,
                gamma=args.gamma,
                emtypecolumn=emtypecolumn,
                emissionvelocitycut=args.emissionvelocitycut,
                directionbin=dirbin,
                average_over_phi=args.average_over_phi_angle,
                average_over_theta=args.average_over_theta_angle,
                directionbins_are_vpkt_observers=args.plotvspecpol is not None,
                vpkt_match_emission_exclusion_to_opac=args.vpkt_match_emission_exclusion_to_opac,
            )
        )
    else:
        assert not args.vpkt_match_emission_exclusion_to_opac
        arraylambda_angstroms = 2.99792458e18 / get_nu_grid(modelpath)
        contribution_list, array_flambda_emission_total = atspectra.get_flux_contributions(
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
        )

    atspectra.print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    contributions_sorted_reduced = atspectra.sort_and_reduce_flux_contribution_list(
        contribution_list,
        args.maxseriescount,
        arraylambda_angstroms,
        fixedionlist=args.fixedionlist,
        hideother=args.hideother,
        greyscale=args.greyscale,
    )

    plotobjectlabels: list[str] = []
    plotobjects: list[mplartist.Artist] = []

    dfspectotal = atspectra.get_dfspectrum_x_y_with_units(
        pl.DataFrame({"f_lambda": array_flambda_emission_total, "lambda_angstroms": arraylambda_angstroms}),
        xunit=args.xunit,
        yvariable=args.yvariable,
        fluxdistance_mpc=args.distmpc,
    )

    max_f_emission_total = dfspectotal.filter(pl.col("x").is_between(xmin, xmax))["y"].max()
    assert isinstance(max_f_emission_total, (float, np.floating))

    scalefactor = scale_to_peak / max_f_emission_total if scale_to_peak else 1.0

    if not args.hidenetspectrum:
        plotobjectlabels.append("Spectrum")
        (line,) = axis.plot(dfspectotal["x"], dfspectotal["y"] * scalefactor, linewidth=1.5, color="black", zorder=100)
        plotobjects.append(line)

    dfaxisdata = pl.DataFrame({"lambda_angstroms": arraylambda_angstroms})

    for x in contributions_sorted_reduced:
        dfaxisdata = dfaxisdata.with_columns(
            pl.Series(name=f"emission_flambda.{x.linelabel}", values=x.array_flambda_emission)
        )
        if args.showabsorption:
            dfaxisdata = dfaxisdata.with_columns(
                pl.Series(name=f"absorption_flambda.{x.linelabel}", values=x.array_flambda_absorption)
            )

    if args.nostack:
        for x in contributions_sorted_reduced:
            if args.showemission:
                dfspec = atspectra.get_dfspectrum_x_y_with_units(
                    pl.DataFrame({"f_lambda": x.array_flambda_emission, "lambda_angstroms": arraylambda_angstroms}),
                    xunit=args.xunit,
                    yvariable=args.yvariable,
                    fluxdistance_mpc=args.distmpc,
                )

                (emissioncomponentplot,) = axis.plot(dfspec["x"], dfspec["y"] * scalefactor, linewidth=1, color=x.color)

                linecolor = emissioncomponentplot.get_color()
            else:
                linecolor = None

            if args.showabsorption:
                dfspec = atspectra.get_dfspectrum_x_y_with_units(
                    pl.DataFrame({"f_lambda": x.array_flambda_absorption, "lambda_angstroms": arraylambda_angstroms}),
                    xunit=args.xunit,
                    yvariable=args.yvariable,
                    fluxdistance_mpc=args.distmpc,
                )
                (absorptioncomponentplot,) = axis.plot(
                    dfspec["x"], -dfspec["y"] * scalefactor, color=linecolor, linewidth=1
                )
                if not args.showemission:
                    linecolor = absorptioncomponentplot.get_color()

            plotobjects.append(mpatches.Patch(color=linecolor))

    elif contributions_sorted_reduced:
        if args.showemission:
            dfabsorptionspectra = [
                atspectra.get_dfspectrum_x_y_with_units(
                    pl.DataFrame({"f_lambda": x.array_flambda_emission, "lambda_angstroms": arraylambda_angstroms}),
                    xunit=args.xunit,
                    yvariable=args.yvariable,
                    fluxdistance_mpc=args.distmpc,
                )
                for x in contributions_sorted_reduced
            ]
            stackplot = axis.stackplot(
                dfabsorptionspectra[0]["x"],
                [dfspec["y"] * scalefactor for dfspec in dfabsorptionspectra],
                colors=[x.color for x in contributions_sorted_reduced],
                linewidth=0,
            )
            if args.greyscale:
                for i, stack in enumerate(stackplot):
                    selectedhatch = hatchestypes[i % len(hatchestypes)]
                    stack.set_hatch(selectedhatch * 7)
            plotobjects.extend(stackplot)
            facecolors = [p.get_facecolor()[0] for p in stackplot]
        else:
            facecolors = [x.color for x in contributions_sorted_reduced]

        if args.showabsorption:
            dfabsorptionspectra = [
                atspectra.get_dfspectrum_x_y_with_units(
                    pl.DataFrame({"f_lambda": x.array_flambda_absorption, "lambda_angstroms": arraylambda_angstroms}),
                    xunit=args.xunit,
                    yvariable=args.yvariable,
                    fluxdistance_mpc=args.distmpc,
                )
                for x in contributions_sorted_reduced
            ]
            absstackplot = axis.stackplot(
                dfabsorptionspectra[0]["x"],
                [-dfspec["y"] * scalefactor for dfspec in dfabsorptionspectra],
                colors=facecolors,  # type: ignore[arg-type] # pyright: ignore[reportArgumentType]
                linewidth=0,
            )
            if not args.showemission:
                plotobjects.extend(absstackplot)

    plotobjectlabels.extend([x.linelabel for x in contributions_sorted_reduced])

    ymaxrefall = 0.0
    plotkwargs = {}
    for index, filepath in enumerate(args.specpath):
        filepath = Path(filepath)
        if path_is_artis_model(filepath):
            continue

        if index < len(args.color):
            plotkwargs["color"] = args.color[index]
            if args.label[index] is not None:
                plotkwargs["label"] = args.label[index]
            plotkwargs["alpha"] = args.linealpha[index]

        supxmin, supxmax = axis.get_xlim()
        plotobj, serieslabel, ymaxref = plot_reference_spectrum(
            filepath,
            axis,
            xmin=supxmin,
            xmax=supxmax,
            fluxfilterfunc=filterfunc,
            scale_to_peak=scale_to_peak,
            scale_to_dist_mpc=args.distmpc,
            offset=0.3 if scale_to_peak else 0.0,
            scaletoreftime=args.scaletoreftime,
            xunit=args.xunit,
            yvariable=args.yvariable,
            **plotkwargs,
        )
        ymaxrefall = max(ymaxrefall, ymaxref)

        plotobjects.append(plotobj)
        plotobjectlabels.append(serieslabel)

    axis.axhline(color="black", linewidth=1)

    if args.title:
        plotlabel = args.title
    else:
        plotlabel = f"{modelname}\n{args.timemin:.2f}d to {args.timemax:.2f}d"
        if args.plotviewingangle or args.plotvspecpol:
            dirbin_definitions = (
                get_vspec_dir_labels(modelpath=modelpath, usedegrees=args.usedegrees)
                if args.plotvspecpol
                else get_dirbin_labels(
                    dirbins=args.plotviewingangle,
                    modelpath=modelpath,
                    average_over_phi=args.average_over_phi_angle,
                    average_over_theta=args.average_over_theta_angle,
                    usedegrees=args.usedegrees,
                )
            )
            assert dirbin is not None
            plotlabel += f", {dirbin_definitions[dirbin]}"

    if not args.notitle:
        if args.inset_title:
            axis.annotate(
                plotlabel,
                xy=(0.03, 0.96),
                xycoords="axes fraction",
                horizontalalignment="left",
                verticalalignment="top",
                fontsize="large",
            )
        else:
            axis.set_title(plotlabel, fontsize=11)

    # axis.annotate(plotlabel, xy=(0.97, 0.03), xycoords='axes fraction',
    #               horizontalalignment='right', verticalalignment='bottom', fontsize=7)

    ymax = max(ymaxrefall, scalefactor * max_f_emission_total * 1.2)
    if args.ymax is None:
        axis.set_ylim(top=ymax)

    if args.showbinedges:
        import artistools.radfield as atradfield

        radfielddata = atradfield.read_files(modelpath, timestep=timestepmax, modelgridindex=30)
        binedges = atradfield.get_binedges(radfielddata)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5, color="red", label="", zorder=-1, alpha=0.4)

    return plotobjects, plotobjectlabels, dfaxisdata


def make_contrib_plot(
    axes: Iterable[mplax.Axes], modelpath: Path, densityplotyvars: list[str], args: argparse.Namespace
) -> None:
    (_timestepmin, _timestepmax, args.timemin, args.timemax) = get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )
    import artistools.estimators as atestimators
    import artistools.packets as atpackets

    modeldata, _ = get_modeldata_pandas(modelpath)

    if args.classicartis:
        modeldata, _ = get_modeldata_pandas(modelpath)
        estimators = atestimators.estimators_classic.read_classic_estimators(modelpath, modeldata)
        allnonemptymgilist = list(modeldata.index)

    else:
        estimators = atestimators.read_estimators(modelpath=modelpath)
        allnonemptymgilist = list({modelgridindex for ts, modelgridindex in estimators})

    assert estimators is not None
    packetsfiles = atpackets.get_packets_text_paths(modelpath, args.maxpacketfiles)
    assert args.timemin is not None
    assert args.timemax is not None
    # tdays_min = float(args.timemin)
    # tdays_max = float(args.timemax)

    c_ang_s = 2.99792458e18
    # nu_min = c_ang_s / args.xmax
    # nu_max = c_ang_s / args.xmin

    list_lambda: dict[str, list[float]] = {}
    lists_y: dict[str, list[float]] = {}
    for packetsfile in packetsfiles:
        dfpackets = atpackets.readfile(packetsfile, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")

        dfpackets_selected = dfpackets.query(
            "@nu_min <= nu_rf < @nu_max and t_arrive_d >= @tdays_min and t_arrive_d <= @tdays_max", inplace=False
        ).copy()

        # TODO: optimize this to avoid calculating unused columns
        dfpackets_selected = atpackets.add_derived_columns(
            dfpackets_selected,
            modelpath,
            ["em_timestep", "emtrue_modelgridindex", "emission_velocity"],
            allnonemptymgilist=allnonemptymgilist,
        )

        # dfpackets.eval('xindex = floor((@c_ang_s / nu_rf - @lambda_min) / @delta_lambda)', inplace=True)
        # dfpackets.eval(
        #     "lambda_rf_binned = @lambda_min + @delta_lambda * floor((@c_ang_s / nu_rf - @lambda_min) / @delta_lambda)",
        #     inplace=True,
        # )

        for _, packet in dfpackets_selected.iterrows():
            for v in densityplotyvars:
                if v not in list_lambda:
                    list_lambda[v] = []
                if v not in lists_y:
                    lists_y[v] = []
                if v == "emission_velocity":
                    if not np.isnan(packet.emission_velocity) and not np.isinf(packet.emission_velocity):
                        list_lambda[v].append(c_ang_s / packet.nu_rf)
                        lists_y[v].append(packet.emission_velocity / 1e5)
                elif v == "true_emission_velocity":
                    if not np.isnan(packet.true_emission_velocity) and not np.isinf(packet.true_emission_velocity):
                        list_lambda[v].append(c_ang_s / packet.nu_rf)
                        lists_y[v].append(packet.true_emission_velocity / 1e5)
                else:
                    k: tuple[int, int] = (packet["em_timestep"], packet["emtrue_modelgridindex"])
                    if k in estimators:
                        list_lambda[v].append(c_ang_s / packet.nu_rf)
                        lists_y[v].append(estimators[k][v])

    for ax, yvar in zip(axes, densityplotyvars, strict=False):
        # ax.set_ylabel(r'velocity [{} km/s]')
        if not args.hideyticklabels:
            ax.set_ylabel(f"{yvar} {atestimators.get_units_string(yvar)}")
        # ax.plot(list_lambda, list_yvar, lw=0, marker='o', markersize=0.5)
        # ax.hexbin(list_lambda[yvar], lists_y[yvar], gridsize=100, cmap=plt.cm.BuGn_r)
        ax.hist2d(list_lambda[yvar], lists_y[yvar], bins=(50, 30), cmap="Greys")
        # plt.cm.Greys
        # x = np.array(list_lambda[yvar])
        # y = np.array(lists_y[yvar])
        # from scipy.stats import kde
        #
        # nbins = 30
        # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)


def make_plot(args: argparse.Namespace) -> tuple[mplfig.Figure, npt.NDArray[t.Any], pl.DataFrame]:
    # font = {'size': 16}
    # mpl.rc('font', **font)

    densityplotyvars: list[str] = []
    # densityplotyvars = ['emission_velocity', 'Te', 'nne']
    # densityplotyvars = ['true_emission_velocity', 'emission_velocity', 'Te', 'nne']

    nrows = len(args.timedayslist) if args.multispecplot else 1 + len(densityplotyvars)

    figwidth = args.figscale * get_config()["figwidth"] * args.figwidthscale
    figheight = args.figscale * get_config()["figwidth"] * (0.25 + nrows * 0.4)
    if args.showabsorption:
        figheight *= 1.56
    if args.hidexticklabels:
        figheight *= 0.87

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharey=False,
        sharex=True,
        squeeze=True,
        figsize=(figwidth, figheight),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    axes = np.array([axes]) if nrows == 1 else np.array(axes)
    assert isinstance(axes, np.ndarray)

    filterfunc = get_filterfunc(args)

    scale_to_peak = 1.0 if args.normalised else None

    dfalldata: pl.DataFrame | None = pl.DataFrame()

    xlabel, ylabel = get_axis_labels(args)

    if args.normalised and args.ymax is None:
        args.ymax = 1.10
    for index, axis in enumerate(axes):
        if args.xmin is not None:
            axis.set_xlim(left=args.xmin)
        if args.xmax is not None:
            axis.set_xlim(right=args.xmax)
        if args.ymin is not None:
            axis.set_ylim(bottom=args.ymin)
        if args.ymax is not None:
            axis.set_ylim(top=args.ymax)
        if args.logscalex:
            axis.set_xscale("log")
        if args.logscaley:
            axis.set_yscale("log")

        if not args.logscalex:
            axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        if args.hidexticklabels:
            axis.tick_params(axis="x", which="both", labelbottom=False)

        if args.hideyticklabels:
            axis.tick_params(axis="y", which="both", labelleft=False)
        else:
            axis.set_ylabel(ylabel)

        if "{" in axis.get_ylabel() and not args.logscaley:
            axis.yaxis.set_major_formatter(ExponentLabelFormatter(axis.get_ylabel(), decimalplaces=1))
            axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        axis.set_xlabel("")  # remove xlabel (last axis xlabel optionally added later)

        if args.showtime:
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

    if not args.hidexticklabels:
        axes[-1].set_xlabel(xlabel)

    if densityplotyvars:
        make_contrib_plot(axes[:-1], args.specpath[0], densityplotyvars, args)

    if args.showemission or args.showabsorption:
        legendncol = 2
        defaultoutputfile = Path("plotspecemission_{time_days_min:.1f}d_{time_days_max:.1f}d{directionbins}.pdf")
        plotobjects, plotobjectlabels, dfalldata = make_emissionabsorption_plot(
            modelpath=Path(args.specpath[0]),
            axis=axes[-1],
            filterfunc=filterfunc,
            args=args,
            scale_to_peak=scale_to_peak,
        )
    else:
        legendncol = 1
        defaultoutputfile = Path("plotspec_{time_days_min:.1f}d_{time_days_max:.1f}d.pdf")

        if args.multispecplot:
            dfalldata = make_spectrum_plot(args.specpath, axes, filterfunc, args, scale_to_peak=scale_to_peak)
            plotobjects, plotobjectlabels = axes[0].get_legend_handles_labels()
        else:
            dfalldata = make_spectrum_plot(args.specpath, [axes[-1]], filterfunc, args, scale_to_peak=scale_to_peak)
            plotobjects, plotobjectlabels = axes[-1].get_legend_handles_labels()

    if not args.nolegend:
        if args.reverselegendorder:  # TODO: consider ax.legend(reverse=True)
            plotobjects, plotobjectlabels = plotobjects[::-1], plotobjectlabels[::-1]

        leg = axes[-1].legend(
            plotobjects,
            plotobjectlabels,
            loc="upper right",
            frameon=False,
            handlelength=1 if args.showemission or args.showabsorption else 2,
            ncol=legendncol,
            numpoints=1,
            columnspacing=1.0,
        )
        leg.set_zorder(200)

        # Luke: I don't know what this code is for
        for artist, text in zip(leg.legend_handles, leg.get_texts(), strict=False):
            if hasattr(artist, "get_color"):
                col = artist.get_color()
                artist.set_linewidth(2.0)
                # artist.set_visible(False)  # hide line next to text
            elif hasattr(artist, "get_facecolor"):
                col = artist.get_facecolor()
            else:
                continue

            if isinstance(col, np.ndarray):
                col = col[0]
            text.set_color(col)

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile /= defaultoutputfile

    # plt.text(6000, (args.ymax * 0.9), f'{round(args.timemin) + 1} days', fontsize='large')

    assert dfalldata is not None
    return fig, axes, dfalldata


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "specpath",
        default=[],
        nargs="*",
        action=AppendPath,
        help="Paths to ARTIS folders or reference spectra filenames",
    )

    parser.add_argument("-label", default=[], nargs="*", help="List of series label overrides")

    parser.add_argument("-color", "-colors", dest="color", default=[], nargs="*", help="List of line colors")

    parser.add_argument("-linestyle", default=[], nargs="*", help="List of line styles")

    parser.add_argument("-linewidth", default=[], nargs="*", help="List of line widths")

    parser.add_argument("-linealpha", default=[], nargs="*", help="List of line alphas (opacities)")

    parser.add_argument("-dashes", default=[], nargs="*", help="Dashes property of lines")

    parser.add_argument("--gamma", action="store_true", help="Make light curve from gamma rays instead of R-packets")

    parser.add_argument("--greyscale", action="store_true", help="Plot in greyscale")

    parser.add_argument(
        "--frompackets", action="store_true", help="Read packets files directly instead of exspec results"
    )

    parser.add_argument("-maxpacketfiles", type=int, default=None, help="Limit the number of packet files read")

    parser.add_argument("--emissionabsorption", action="store_true", help="Implies --showemission and --showabsorption")

    parser.add_argument("--showemission", action="store_true", help="Plot the emission spectra by ion/process")

    parser.add_argument("--showabsorption", action="store_true", help="Plot the absorption spectra by ion/process")

    parser.add_argument(
        "-emissionvelocitycut",
        type=float,
        help=(
            "Only show contributions to emission plots where emission velocity "
            "is greater than some velocity (km/s) eg. --emissionvelocitycut 15000"
        ),
    )

    parser.add_argument(
        "-yvariable",
        "-yvar",
        "-y",
        type=str,
        default="flux",
        choices=["flux", "packetcount", "photoncount", "eflux"],
        help="Specify the y-axis variable for the plot",
    )

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

    parser.add_argument(
        "-filtersavgol",
        nargs=2,
        help="Savitzky-Golay filter. Specify the window_length and poly_order.e.g. -filtersavgol 5 3",
    )

    parser.add_argument("-filtermovingavg", type=int, default=0, help="Smoothing length (1 is same as none)")

    parser.add_argument("-timestep", "-ts", dest="timestep", nargs="?", help="First timestep or a range e.g. 45-65")

    parser.add_argument(
        "-timedays", "-time", "-t", dest="timedays", nargs="?", help="Range of times in days to plot (e.g. 50-100)"
    )

    parser.add_argument("-timemin", type=float, help="Lower time in days to integrate spectrum")

    parser.add_argument("-timemax", type=float, help="Upper time in days to integrate spectrum")

    parser.add_argument(
        "--notimeclamp", action="store_true", help="When plotting from packets, don't clamp to timestep start/end"
    )

    parser.add_argument(
        "-xunit",
        "-xunits",
        "-x",
        dest="xunit",
        default=None,
        type=str,
        help="x (horizontal) axis unit, e.g. angstrom, nm, micron, Hz, keV, MeV",
    )

    parser.add_argument(
        "-xmin", "-lambdamin", dest="xmin", type=float, default=None, help="Plot range: minimum x range"
    )

    parser.add_argument(
        "-xmax", "-lambdamax", dest="xmax", type=float, default=None, help="Plot range: maximum x range"
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
        help="Tag packets by their thermal emission type rather than last scattering process",
    )

    parser.add_argument(
        "-groupby",
        default=None,
        choices=["ion", "line", "nuc", "nucmass"],
        help="Use a different color for each ion or line when using --showemission. groupby='line', 'nuc', 'nucmass' imply --frompackets.",
    )

    parser.add_argument(
        "-obsspec",
        "-refspecfiles",
        action="append",
        dest="refspecfiles",
        help="Also plot reference spectrum from this file",
    )

    parser.add_argument(
        "-distmpc",
        "-dist_mpc",
        "-dist",
        "-fluxdistmpc",
        type=float,
        default=1.0,
        help="Distance in megaparsec when calculating fluxes (default: 1 Mpc)",
    )

    parser.add_argument(
        "-scaletoreftime", type=float, default=None, help="Scale reference spectra flux using Co56 decay timescale"
    )

    parser.add_argument("--showbinedges", action="store_true", help="Plot vertical lines at the bin edges")

    parser.add_argument(
        "-figscale", type=float, default=1.8, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("-figwidthscale", type=float, default=1.0, help="Scale factor for plot width")

    parser.add_argument("--logscalex", action="store_true", help="Use log scale for x values")

    parser.add_argument("--logscaley", action="store_true", help="Use log scale for y values")

    parser.add_argument("--hidenetspectrum", action="store_true", help="Hide net spectrum")

    parser.add_argument("--hideother", action="store_true", help="Hide other contributions")

    parser.add_argument("--notitle", action="store_true", help="Suppress the top title from the plot")

    parser.add_argument("-title", action="store_true", help="Set the plot title")

    parser.add_argument("--inset_title", action="store_true", help="Place title inside the plot")

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument("--reverselegendorder", action="store_true", help="Reverse the order of legend items")

    parser.add_argument("--hidexticklabels", action="store_true", help="Don't show numbers or a label on the x axis")

    parser.add_argument("--hideyticklabels", action="store_true", help="Don't show numbers or a label on the y axis")

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a CSV file")

    parser.add_argument(
        "-outputfile", "-o", action="store", dest="outputfile", type=Path, help="path/filename for PDF file"
    )

    parser.add_argument("-dpi", type=int, default=250, help="Dots Per Inch for output file")

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

    parser.add_argument(
        "-plotvspecpol",
        type=int,
        nargs="+",
        help="Plot viewing angles from vspecpol virtual packets. Expects int for angle = spec number in vspecpol files",
    )

    parser.add_argument(
        "-stokesparam", type=str, default="I", help="Stokes param to plot. Default I. Expects I, Q or U"
    )

    parser.add_argument(
        "-plotviewingangle",
        type=int,
        metavar="n",
        nargs="+",
        help="Plot viewing angles. Expects int for direction bin in specpol_res.out",
    )

    parser.add_argument(
        "--usedegrees",
        action="store_true",
        help="Use degrees instead of radians for viewing angles. Only works with -plotviewingangle",
    )

    parser.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # for backwards compatibility with above option
    parser.add_argument("--average_every_tenth_viewing_angle", action="store_true")

    parser.add_argument(
        "--average_over_theta_angle",
        action="store_true",
        help="Average over theta (polar) viewing angles to make direction bins into azimuthal angle bins",
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
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)
        if args.average_every_tenth_viewing_angle:
            print("WARNING: --average_every_tenth_viewing_angle is deprecated. use --average_over_phi_angle instead")
            args.average_over_phi_angle = True

    if args.xunit is None:
        args.xunit = "kev" if args.gamma else "angstroms"
    args.xunit = atspectra.convert_xunit_aliases_to_canonical(args.xunit)

    if args.xmin is None:
        args.xmin = atspectra.convert_angstroms_to_unit(0.2 if args.gamma else 2500.0, args.xunit)
    if args.xmax is None:
        args.xmax = atspectra.convert_angstroms_to_unit(0.004 if args.gamma else 19000.0, args.xunit)

    args.xmin, args.xmax = sorted([args.xmin, args.xmax])

    set_mpl_style()

    assert (
        not args.plotvspecpol or not args.plotviewingangle
    )  # choose either virtual packet directions or real packet direction bins

    if not args.specpath:
        args.specpath = [Path()]
    elif isinstance(args.specpath, str | Path):  # or not not isinstance(args.specpath, Iterable)
        args.specpath = [args.specpath]

    args.specpath = flatten_list(args.specpath)

    if args.timedayslist:
        args.multispecplot = True
        args.timedays = args.timedayslist[0]

    if not args.color:
        args.color = []
        artismodelcolors = [f"C{i}" for i in range(10)]
        refspeccolors = ["0.0", "0.4", "0.6", "0.7"]
        refspecnum = 0
        artismodelnum = 0
        for filepath in args.specpath:
            if path_is_artis_model(filepath):
                args.color.append(artismodelcolors[artismodelnum])
                artismodelnum += 1
            else:
                args.color.append(refspeccolors[refspecnum])
                refspecnum += 1

    args.color, args.label, args.linestyle, args.linealpha, args.dashes, args.linewidth = trim_or_pad(
        len(args.specpath), args.color, args.label, args.linestyle, args.linealpha, args.dashes, args.linewidth
    )

    if args.vpkt_match_emission_exclusion_to_opac:
        assert args.showemission
        assert args.frompackets
        assert args.plotvspecpol

    if args.groupby is not None:
        args.showemission = True

    if args.emissionvelocitycut or args.groupby in {"line", "nuc", "nucmass"}:
        args.frompackets = True

    if args.gamma and args.plotviewingangle:
        # exspec does not generate angle-resolved gamma spectra files,
        # so we need to use the packets instead
        args.frompackets = True

    if args.use_emissiontime or args.use_escapetime:
        # exspec spectra are binned by arrivine time at the observer
        # so we need to use the packets instead
        args.frompackets = True

    if not args.frompackets and any(x is not None for x in (args.deltax, args.deltalogx, args.deltalambda)):
        args.frompackets = True
        print("Enabling --frompackets, since custom bin width was specified")

    if args.makevspecpol:
        atspectra.make_virtual_spectra_summed_file(args.specpath[0])
        return

    if args.averagevspecpolfiles:
        atspectra.make_averaged_vspecfiles(args)
        return

    if "/" in args.stokesparam:
        plot_polarisation(args.specpath[0], args)
        return

    if args.output_spectra:
        for modelpath in args.specpath:
            atspectra.write_flambda_spectra(modelpath)

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

        filenameout = (
            str(args.outputfile).format(
                time_days_min=args.timemin, time_days_max=args.timemax, directionbins=strdirectionbins
            )
            if args.timemin is not None
            else "plotspec.pdf"
        )

        if args.write_data and len(dfalldata.columns) > 0:
            datafilenameout = Path(filenameout).with_suffix(".txt")
            dfalldata.write_csv(datafilenameout, separator=" ")
            print(f"Saved {datafilenameout}")

        # plt.minorticks_on()

        fig.savefig(filenameout, dpi=args.dpi)
        # plt.show()
        print(f"Saved {filenameout}")
        plt.close()


if __name__ == "__main__":
    main()
