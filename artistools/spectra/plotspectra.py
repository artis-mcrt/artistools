#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Artistools - spectra plotting functions."""
import argparse
import math
import os
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import argcomplete
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.artist import Artist

import artistools as at
import artistools.packets
from .spectra import get_from_packets
from .spectra import get_reference_spectrum
from .spectra import get_specpol_data
from .spectra import get_spectrum
from .spectra import get_vspecpol_spectrum
from .spectra import make_averaged_vspecfiles
from .spectra import make_virtual_spectra_summed_file
from .spectra import print_integrated_flux
from .spectra import timeshift_fluxscale_co56law

hatches = ["", "x", "-", "\\", "+", "O", ".", "", "x", "*", "\\", "+", "O", "."]  # ,


def plot_polarisation(modelpath: Path, args) -> None:
    angle = args.plotviewingangle[0]
    stokes_params = get_specpol_data(angle=angle, modelpath=modelpath)
    stokes_params[args.stokesparam] = stokes_params[args.stokesparam].eval("lambda_angstroms = 2.99792458e18 / nu")

    timearray = stokes_params[args.stokesparam].keys()[1:-1]
    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )
    assert args.timemin is not None
    assert args.timemax is not None
    timeavg = (args.timemin + args.timemax) / 2.0

    def match_closest_time(reftime):
        return str("{:.4f}".format(min([float(x) for x in timearray], key=lambda x: abs(x - reftime))))

    timeavg = match_closest_time(timeavg)

    filterfunc = at.get_filterfunc(args)
    if filterfunc is not None:
        print("Applying filter to ARTIS spectrum")
        stokes_params[args.stokesparam][timeavg] = filterfunc(stokes_params[args.stokesparam][timeavg])

    vpkt_config = at.get_vpkt_config(modelpath)

    linelabel = (
        f"{timeavg} days, cos($\\theta$) = {vpkt_config['cos_theta'][angle[0]]}"
        if args.plotvspecpol
        else f"{timeavg} days"
    )

    if args.binflux:
        new_lambda_angstroms = []
        binned_flux = []

        wavelengths = stokes_params[args.stokesparam]["lambda_angstroms"]
        fluxes = stokes_params[args.stokesparam][timeavg]
        nbins = 5

        for i in np.arange(0, len(wavelengths - nbins), nbins):
            new_lambda_angstroms.append(wavelengths[i + int(nbins / 2)])
            sum_flux = 0
            for j in range(i, i + nbins):
                sum_flux += fluxes[j]
            binned_flux.append(sum_flux / nbins)

        fig = plt.plot(new_lambda_angstroms, binned_flux)
    else:
        fig = stokes_params[args.stokesparam].plot(x="lambda_angstroms", y=timeavg, label=linelabel)

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

    plt.ylabel(f"{args.stokesparam}")
    plt.xlabel(r"Wavelength ($\mathrm{{\AA}}$)")
    figname = f"plotpol_{timeavg}_days_{args.stokesparam.split('/')[0]}_{args.stokesparam.split('/')[1]}.pdf"
    plt.savefig(modelpath / figname, format="pdf")
    print(f"Saved {figname}")


def plot_reference_spectrum(
    filename: Union[Path, str],
    axis: plt.Axes,
    xmin: float,
    xmax: float,
    flambdafilterfunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    scale_to_peak: Optional[float] = None,
    scale_to_dist_mpc: float = 1,
    scaletoreftime: Optional[float] = None,
    **plotkwargs,
):
    """Plot a single reference spectrum.

    The filename must be in space separated text formated with the first two
    columns being wavelength in Angstroms, and F_lambda
    """
    specdata, metadata = get_reference_spectrum(filename)

    # scale to flux at required distance
    if scale_to_dist_mpc:
        print(f"Scale to {scale_to_dist_mpc} Mpc")
        assert metadata["dist_mpc"] > 0  # we must know the true distance in order to scale to some other distance
        specdata["f_lambda"] = specdata["f_lambda"] * (metadata["dist_mpc"] / scale_to_dist_mpc) ** 2

    if "label" not in plotkwargs:
        plotkwargs["label"] = metadata["label"] if "label" in metadata else filename

    if scaletoreftime is not None:
        timefactor = timeshift_fluxscale_co56law(scaletoreftime, float(metadata["t"]))
        print(f" Scale from time {metadata['t']} to {scaletoreftime}, factor {timefactor} using Co56 decay law")
        specdata["f_lambda"] *= timefactor
        plotkwargs["label"] += f" * {timefactor:.2f}"
    if "scale_factor" in metadata:
        specdata["f_lambda"] *= metadata["scale_factor"]

    print(f"Reference spectrum '{plotkwargs['label']}' has {len(specdata)} points in the plot range")
    print(f" file: {filename}")

    print(" metadata: " + ", ".join([f"{k}='{v}'" if hasattr(v, "lower") else f"{k}={v}" for k, v in metadata.items()]))

    specdata = specdata.query("lambda_angstroms > @xmin and lambda_angstroms < @xmax")

    print_integrated_flux(specdata["f_lambda"], specdata["lambda_angstroms"], distance_megaparsec=metadata["dist_mpc"])

    if len(specdata) > 5000:
        # specdata = scipy.signal.resample(specdata, 10000)
        # specdata = specdata.iloc[::3, :].copy()
        print(f" downsampling to {len(specdata)} points")
        specdata = specdata.query("index % 3 == 0")

    # clamp negative values to zero
    # specdata['f_lambda'] = specdata['f_lambda'].apply(lambda x: max(0, x))

    if flambdafilterfunc:
        specdata["f_lambda"] = flambdafilterfunc(specdata["f_lambda"])

    if scale_to_peak:
        specdata["f_lambda_scaled"] = specdata["f_lambda"] / specdata["f_lambda"].max() * scale_to_peak
        ycolumnname = "f_lambda_scaled"
    else:
        ycolumnname = "f_lambda"

    ymax = max(specdata[ycolumnname])
    lineplot = specdata.plot(x="lambda_angstroms", y=ycolumnname, ax=axis, legend=None, **plotkwargs)

    return mpatches.Patch(color=lineplot.get_lines()[0].get_color()), plotkwargs["label"], ymax


def plot_filter_functions(axis: plt.Axes) -> None:
    filter_names = ["U", "B", "V", "I"]
    colours = ["r", "b", "g", "c", "m"]

    filterdir = os.path.join(at.get_config()["path_artistools_dir"], "data/filters/")
    for index, filter_name in enumerate(filter_names):
        filter_data = pd.read_csv(
            filterdir / Path(f"{filter_name}.txt"),
            delim_whitespace=True,
            header=None,
            skiprows=4,
            names=["lamba_angstroms", "flux_normalised"],
        )
        filter_data.plot(
            x="lamba_angstroms", y="flux_normalised", ax=axis, label=filter_name, color=colours[index], alpha=0.3
        )


def plot_artis_spectrum(
    axes: Collection[plt.Axes],
    modelpath: Union[Path, str],
    args,
    scale_to_peak: Optional[float] = None,
    from_packets: bool = False,
    filterfunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    linelabel: Optional[str] = None,
    plotpacketcount: bool = False,
    directionbins: Optional[list[int]] = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    maxpacketfiles: Optional[int] = None,
    **plotkwargs,
) -> Optional[pd.DataFrame]:
    """Plot an ARTIS output spectrum. The data plotted are also returned as a DataFrame"""
    modelpath_or_file = Path(modelpath)

    modelpath = Path(modelpath)
    if Path(modelpath).is_file():  # handle e.g. modelpath = 'modelpath/spec.out'
        specfilename = Path(modelpath).parts[-1]
        print("WARNING: ignoring filename of {specfilename}")
        modelpath = Path(modelpath).parent

    if not modelpath.is_dir():
        print(f"WARNING: Skipping because {modelpath} does not exist")
        return None

    if directionbins is None:
        directionbins = [-1]

    if plotpacketcount:
        from_packets = True

    for axindex, axis in enumerate(axes):
        if args.multispecplot:
            (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
                modelpath, timedays_range_str=args.timedayslist[axindex]
            )
        else:
            (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
                modelpath, args.timestep, args.timemin, args.timemax, args.timedays
            )

        modelname = at.get_model_name(modelpath)
        if timestepmin == timestepmax == -1:
            return None

        assert args.timemin is not None
        assert args.timemax is not None
        timeavg = (args.timemin + args.timemax) / 2.0
        timedelta = (args.timemax - args.timemin) / 2
        if linelabel is None:
            linelabel = f"{modelname}" if len(modelname) < 70 else f"...{modelname[-67:]}"

            if not args.hidemodeltime and not args.multispecplot:
                # todo: fix this for multispecplot - use args.showtime for now
                linelabel += f" +{timeavg:.1f}d"
            if not args.hidemodeltimerange and not args.multispecplot:
                linelabel += rf" ($\pm$ {timedelta:.1f}d)"
        # Luke: disabled below because line label has already been formatted with e.g. timeavg values
        # formatting for a second time makes it impossible to use curly braces in line labels (needed for LaTeX math)
        # else:
        #     linelabel = linelabel.format(**locals())
        print(
            f"====> '{linelabel}' timesteps {timestepmin} to {timestepmax} ({args.timemin:.3f} to {args.timemax:.3f}d)"
        )
        print(f" modelpath {modelpath}")

        viewinganglespectra: dict[int, pd.DataFrame] = {}

        # have to get the spherical average "bin" if directionbins is None
        dbins_get = list(directionbins).copy()
        if -1 not in dbins_get:
            dbins_get.append(-1)

        supxmin, supxmax = axis.get_xlim()
        if from_packets:
            viewinganglespectra = get_from_packets(
                modelpath,
                args.timemin,
                args.timemax,
                lambda_min=supxmin * 0.9,
                lambda_max=supxmax * 1.1,
                use_escapetime=args.use_escapetime,
                maxpacketfiles=maxpacketfiles,
                delta_lambda=args.deltalambda,
                useinternalpackets=args.internalpackets,
                getpacketcount=plotpacketcount,
                directionbins=dbins_get,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
                fnufilterfunc=filterfunc,
            )
        else:
            if args.plotvspecpol is not None:  # noqa: PLR5501
                # read virtual packet files (after running plotartisspectrum --makevspecpol)
                vpkt_config = at.get_vpkt_config(modelpath)
                if vpkt_config["time_limits_enabled"] and (
                    args.timemin < vpkt_config["initial_time"] or args.timemax > vpkt_config["final_time"]
                ):
                    print(
                        f"Timestep out of range of virtual packets: start time {vpkt_config['initial_time']} days "
                        f"end time {vpkt_config['final_time']} days"
                    )
                    sys.exit(1)

                viewinganglespectra = {
                    dirbin: get_vspecpol_spectrum(modelpath, timeavg, dirbin, args, fnufilterfunc=filterfunc)
                    for dirbin in dbins_get
                }
            else:
                viewinganglespectra = get_spectrum(
                    modelpath=modelpath,
                    directionbins=dbins_get,
                    timestepmin=timestepmin,
                    timestepmax=timestepmax,
                    average_over_phi=average_over_phi,
                    average_over_theta=average_over_theta,
                    fnufilterfunc=filterfunc,
                )

        dirbin_definitions = at.get_dirbin_labels(
            dirbins=directionbins,
            modelpath=modelpath,
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
        )

        for dirbin in directionbins:
            dfspectrum = (
                viewinganglespectra[dirbin]
                .query("@supxmin * 0.9 <= lambda_angstroms and lambda_angstroms <= @supxmax * 1.1")
                .copy()
            )

            if dirbin != -1:
                if args.plotvspecpol:
                    if args.viewinganglelabelunits == "deg":
                        viewing_angle = round(math.degrees(math.acos(vpkt_config["cos_theta"][dirbin])))
                        linelabel = rf"$\theta$ = {viewing_angle}$^\circ$" if axindex == 0 else None
                    elif args.viewinganglelabelunits == "rad":
                        linelabel = rf"cos $\theta$ = {vpkt_config['cos_theta'][dirbin]}" if axindex == 0 else None
                else:
                    linelabel = dirbin_definitions[dirbin]
                    print(f" directionbin {dirbin:4d}  {dirbin_definitions[dirbin]}")

            print_integrated_flux(dfspectrum["f_lambda"], dfspectrum["lambda_angstroms"])

            if scale_to_peak:
                dfspectrum["f_lambda_scaled"] = dfspectrum["f_lambda"] / dfspectrum["f_lambda"].max() * scale_to_peak
                if args.plotvspecpol is not None:
                    for angle in args.plotvspecpol:
                        viewinganglespectra[angle]["f_lambda_scaled"] = (
                            viewinganglespectra[angle]["f_lambda"]
                            / viewinganglespectra[angle]["f_lambda"].max()
                            * scale_to_peak
                        )

                ycolumnname = "f_lambda_scaled"
            else:
                ycolumnname = "f_lambda"

            if plotpacketcount:
                ycolumnname = "packetcount"

            if args.binflux:
                new_lambda_angstroms = []
                binned_flux = []

                wavelengths = dfspectrum["lambda_angstroms"]
                fluxes = dfspectrum[ycolumnname]
                nbins = 5

                for i in np.arange(0, len(wavelengths - nbins), nbins):
                    new_lambda_angstroms.append(wavelengths[i + int(nbins / 2)])
                    sum_flux = 0
                    for j in range(i, i + nbins):
                        sum_flux += fluxes[j]
                    binned_flux.append(sum_flux / nbins)

                dfspectrum = pd.DataFrame({"lambda_angstroms": new_lambda_angstroms, ycolumnname: binned_flux})

            dfspectrum.plot(
                x="lambda_angstroms",
                y=ycolumnname,
                ax=axis,
                legend=None,
                label=linelabel if axindex == 0 else None,
                **plotkwargs,
            )

    return dfspectrum[["lambda_angstroms", "f_lambda"]]


def make_spectrum_plot(
    speclist: Collection[Union[Path, str]],
    axes: plt.Axes,
    filterfunc: Optional[Callable[[np.ndarray], np.ndarray]],
    args,
    scale_to_peak: Optional[float] = None,
) -> pd.DataFrame:
    """Plot reference spectra and ARTIS spectra."""
    dfalldata = pd.DataFrame()
    artisindex = 0
    refspecindex = 0
    seriesindex = 0

    for seriesindex, specpath in enumerate(speclist):
        specpath = Path(specpath)
        plotkwargs: dict[str, Any] = {}
        plotkwargs["alpha"] = 0.95

        plotkwargs["linestyle"] = args.linestyle[seriesindex]
        if not args.plotviewingangle:
            plotkwargs["color"] = args.color[seriesindex]
        if args.dashes[seriesindex]:
            plotkwargs["dashes"] = args.dashes[seriesindex]
        if args.linewidth[seriesindex]:
            plotkwargs["linewidth"] = args.linewidth[seriesindex]

        seriesdata = pd.DataFrame()

        if not Path(specpath).is_dir() and not Path(specpath).exists() and "." in str(specpath):
            # reference spectrum
            if "linewidth" not in plotkwargs:
                plotkwargs["linewidth"] = 1.1

            if args.multispecplot:
                plotkwargs["color"] = "k"
                supxmin, supxmax = axes[refspecindex].get_xlim()
                plot_reference_spectrum(
                    specpath,
                    axes[refspecindex],
                    supxmin,
                    supxmax,
                    filterfunc,
                    scale_to_peak,
                    scaletoreftime=args.scaletoreftime,
                    **plotkwargs,
                )
            else:
                for axis in axes:
                    supxmin, supxmax = axis.get_xlim()
                    plot_reference_spectrum(
                        specpath,
                        axis,
                        supxmin,
                        supxmax,
                        filterfunc,
                        scale_to_peak,
                        scaletoreftime=args.scaletoreftime,
                        **plotkwargs,
                    )
            refspecindex += 1
        elif not specpath.exists() and specpath.parts[0] == "codecomparison":
            # timeavg = (args.timemin + args.timemax) / 2.
            (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
                specpath, args.timestep, args.timemin, args.timemax, args.timedays
            )
            timeavg = args.timedays
            artistools.codecomparison.plot_spectrum(specpath, timedays=timeavg, ax=axes[0], **plotkwargs)
            refspecindex += 1
        else:
            # ARTIS model spectrum
            # plotkwargs['dash_capstyle'] = dash_capstyleList[artisindex]
            if "linewidth" not in plotkwargs:
                plotkwargs["linewidth"] = 1.3

            plotkwargs["linelabel"] = args.label[seriesindex]

            seriesdata = plot_artis_spectrum(
                axes,
                specpath,
                args=args,
                scale_to_peak=scale_to_peak,
                from_packets=args.frompackets,
                maxpacketfiles=args.maxpacketfiles,
                filterfunc=filterfunc,
                plotpacketcount=args.plotpacketcount,
                directionbins=args.plotviewingangle,
                average_over_phi=args.average_over_phi_angle,
                average_over_theta=args.average_over_theta_angle,
                **plotkwargs,
            )
            if seriesdata is not None:
                seriesname = at.get_model_name(specpath)
                artisindex += 1

        if args.write_data and not seriesdata.empty:
            if dfalldata.empty:
                dfalldata = pd.DataFrame(index=seriesdata["lambda_angstroms"].to_numpy())
                dfalldata.index.name = "lambda_angstroms"
            else:
                # make sure we can share the same set of wavelengths for this series
                assert np.allclose(dfalldata.index.values, seriesdata["lambda_angstroms"].to_numpy())
            dfalldata[f"f_lambda.{seriesname}"] = seriesdata["f_lambda"].to_numpy()

        seriesindex += 1

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

        if args.stokesparam == "I":
            axis.set_ylim(bottom=0.0)
        if args.normalised:
            axis.set_ylim(top=1.25)
            axis.set_ylabel(r"Scaled F$_\lambda$")
        if args.plotpacketcount:
            axis.set_ylabel(r"Monte Carlo packets per bin")

    return dfalldata


def make_emissionabsorption_plot(
    modelpath: Path,
    axis: plt.Axes,
    filterfunc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    args=None,
    scale_to_peak: Optional[float] = None,
) -> tuple[list[Artist], list[str], Optional[pd.DataFrame]]:
    """Plot the emission and absorption contribution spectra, grouped by ion/line/term for an ARTIS model."""
    modelname = at.get_model_name(modelpath)

    print(f"====> {modelname}")
    arraynu = at.get_nu_grid(modelpath)

    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )

    if timestepmin == timestepmax == -1:
        print(f"Can't plot {modelname}...skipping")
        return [], [], None

    print(f"Plotting {modelname} timesteps {timestepmin} to {timestepmax} ({args.timemin:.3f} to {args.timemax:.3f}d)")

    xmin, xmax = axis.get_xlim()

    if args.frompackets:
        (
            contribution_list,
            array_flambda_emission_total,
            arraylambda_angstroms,
        ) = at.spectra.get_flux_contributions_from_packets(
            modelpath,
            args.timemin,
            args.timemax,
            xmin,
            xmax,
            getemission=args.showemission,
            getabsorption=args.showabsorption,
            maxpacketfiles=args.maxpacketfiles,
            filterfunc=filterfunc,
            groupby=args.groupby,
            delta_lambda=args.deltalambda,
            use_lastemissiontype=not args.use_thermalemissiontype,
            useinternalpackets=args.internalpackets,
            emissionvelocitycut=args.emissionvelocitycut,
        )
    else:
        arraylambda_angstroms = 2.99792458e18 / arraynu
        assert args.groupby in [None, "ion"]
        contribution_list, array_flambda_emission_total = at.spectra.get_flux_contributions(
            modelpath,
            filterfunc,
            timestepmin,
            timestepmax,
            getemission=args.showemission,
            getabsorption=args.showabsorption,
            use_lastemissiontype=not args.use_thermalemissiontype,
            directionbin=args.plotviewingangle[0] if args.plotviewingangle else None,
            averageoverphi=args.average_over_phi_angle,
            averageovertheta=args.average_over_theta_angle,
        )

    at.spectra.print_integrated_flux(array_flambda_emission_total, arraylambda_angstroms)

    # print("\n".join([f"{x[0]}, {x[1]}" for x in contribution_list]))

    contributions_sorted_reduced = at.spectra.sort_and_reduce_flux_contribution_list(
        contribution_list,
        args.maxseriescount,
        arraylambda_angstroms,
        fixedionlist=args.fixedionlist,
        hideother=args.hideother,
        greyscale=args.greyscale,
    )

    plotobjectlabels = []
    plotobjects = []

    max_flambda_emission_total = max(
        [
            flambda if (xmin < lambda_ang < xmax) else -99.0
            for lambda_ang, flambda in zip(arraylambda_angstroms, array_flambda_emission_total)
        ]
    )

    scalefactor = scale_to_peak / max_flambda_emission_total if scale_to_peak else 1.0

    if not args.hidenetspectrum:
        plotobjectlabels.append("Net spectrum")
        line = axis.plot(
            arraylambda_angstroms, array_flambda_emission_total * scalefactor, linewidth=1.5, color="black", zorder=100
        )
        linecolor = line[0].get_color()
        plotobjects.append(mpatches.Patch(color=linecolor))

    dfaxisdata = pd.DataFrame(index=arraylambda_angstroms)
    dfaxisdata.index.name = "lambda_angstroms"
    # dfaxisdata['nu_hz'] = arraynu
    for x in contributions_sorted_reduced:
        dfaxisdata["emission_flambda." + x.linelabel] = x.array_flambda_emission
        if args.showabsorption:
            dfaxisdata["absorption_flambda." + x.linelabel] = x.array_flambda_absorption

    if args.nostack:
        for x in contributions_sorted_reduced:
            if args.showemission:
                emissioncomponentplot = axis.plot(
                    arraylambda_angstroms, x.array_flambda_emission * scalefactor, linewidth=1, color=x.color
                )

                linecolor = emissioncomponentplot[0].get_color()
            else:
                linecolor = None
            plotobjects.append(mpatches.Patch(color=linecolor))

            if args.showabsorption:
                axis.plot(
                    arraylambda_angstroms,
                    -x.array_flambda_absorption * scalefactor,
                    color=linecolor,
                    linewidth=1,
                    alpha=0.6,
                )
    elif contributions_sorted_reduced:
        if args.showemission:
            stackplot = axis.stackplot(
                arraylambda_angstroms,
                [x.array_flambda_emission * scalefactor for x in contributions_sorted_reduced],
                colors=[x.color for x in contributions_sorted_reduced],
                linewidth=0,
            )
            if args.greyscale:
                for i, stack in enumerate(stackplot):
                    selectedhatch = hatches[i % len(hatches)]
                    stack.set_hatch(selectedhatch * 7)
            plotobjects.extend(stackplot)
            facecolors = [p.get_facecolor()[0] for p in stackplot]
        else:
            facecolors = [x.color for x in contributions_sorted_reduced]

        if args.showabsorption:
            absstackplot = axis.stackplot(
                arraylambda_angstroms,
                [-x.array_flambda_absorption * scalefactor for x in contributions_sorted_reduced],
                colors=facecolors,
                linewidth=0,
            )
            if not args.showemission:
                plotobjects.extend(absstackplot)

    plotobjectlabels.extend([x.linelabel for x in contributions_sorted_reduced])
    # print(plotobjectlabels)
    # print(len(plotobjectlabels), len(plotobjects))

    ymaxrefall = 0.0
    plotkwargs = {}
    for index, filepath in enumerate(args.specpath):
        if Path(filepath).is_dir() or Path(filepath).name.endswith(".out"):
            continue
        if index < len(args.color):
            plotkwargs["color"] = args.color[index]

        supxmin, supxmax = axis.get_xlim()
        plotobj, serieslabel, ymaxref = plot_reference_spectrum(
            filepath,
            axis,
            supxmin,
            supxmax,
            filterfunc,
            scale_to_peak,
            scaletoreftime=args.scaletoreftime,
            **plotkwargs,
        )
        ymaxrefall = max(ymaxrefall, ymaxref)

        plotobjects.append(plotobj)
        plotobjectlabels.append(serieslabel)

    axis.axhline(color="white", linewidth=0.5)

    plotlabel = f"{modelname}\n{args.timemin:.2f}d to {args.timemax:.2f}d"
    if args.plotviewingangle:
        dirbin_definitions = at.get_dirbin_labels(
            dirbins=args.plotviewingangle,
            modelpath=modelpath,
            average_over_phi=args.average_over_phi_angle,
            average_over_theta=args.average_over_theta_angle,
        )
        plotlabel += f", directionbin {dirbin_definitions[args.plotviewingangle[0]]}"

    if not args.notitle:
        axis.set_title(plotlabel, fontsize=11)
    # axis.annotate(plotlabel, xy=(0.97, 0.03), xycoords='axes fraction',
    #               horizontalalignment='right', verticalalignment='bottom', fontsize=7)

    ymax = max(ymaxrefall, scalefactor * max_flambda_emission_total * 1.2)
    axis.set_ylim(top=ymax)

    if scale_to_peak:
        axis.set_ylabel(r"Scaled F$_\lambda$")
    elif args.internalpackets:
        if args.logscale:
            # don't include the {} that will be replaced with the power of 10 by the custom formatter
            axis.set_ylabel(r"J$_\lambda$ [erg/s/cm$^2$/$\mathrm{{\AA}}$]")
        else:
            axis.set_ylabel(r"J$_\lambda$ [{}erg/s/cm$^2$/$\mathrm{{\AA}}$]")

    if args.showbinedges:
        radfielddata = at.radfield.read_files(modelpath, timestep=timestepmax, modelgridindex=30)
        binedges = at.radfield.get_binedges(radfielddata)
        axis.vlines(binedges, ymin=0.0, ymax=ymax, linewidth=0.5, color="red", label="", zorder=-1, alpha=0.4)

    return plotobjects, plotobjectlabels, dfaxisdata


def make_contrib_plot(axes: plt.Axes, modelpath: Path, densityplotyvars: list[str], args) -> None:
    (timestepmin, timestepmax, args.timemin, args.timemax) = at.get_time_range(
        modelpath, args.timestep, args.timemin, args.timemax, args.timedays
    )

    modeldata, _ = at.inputmodel.get_modeldata(modelpath)

    if args.classicartis:
        modeldata, _ = at.inputmodel.get_modeldata(modelpath)
        estimators = artistools.estimators.estimators_classic.read_classic_estimators(modelpath, modeldata)
        allnonemptymgilist = list(modeldata.index)

    else:
        estimators = at.estimators.read_estimators(modelpath=modelpath)
        allnonemptymgilist = [
            modelgridindex for modelgridindex in modeldata.index if not estimators[(0, modelgridindex)]["emptycell"]
        ]

    packetsfiles = at.packets.get_packetsfilepaths(modelpath, args.maxpacketfiles)
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
        dfpackets = at.packets.readfile(packetsfile, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")

        dfpackets_selected = dfpackets.query(
            "@nu_min <= nu_rf < @nu_max and t_arrive_d >= @tdays_min and t_arrive_d <= @tdays_max", inplace=False
        ).copy()

        # todo: optimize this to avoid calculating unused columns
        dfpackets_selected = at.packets.add_derived_columns(
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
                    k = (packet["em_timestep"], packet["emtrue_modelgridindex"])
                    if k in estimators:
                        list_lambda[v].append(c_ang_s / packet.nu_rf)
                        lists_y[v].append(estimators[k][v])

    for ax, yvar in zip(axes, densityplotyvars):
        # ax.set_ylabel(r'velocity [{} km/s]')
        ax.set_ylabel(yvar + " " + at.estimators.get_units_string(yvar))
        # ax.plot(list_lambda, list_yvar, lw=0, marker='o', markersize=0.5)
        # ax.hexbin(list_lambda[yvar], lists_y[yvar], gridsize=100, cmap=plt.cm.BuGn_r)
        ax.hist2d(list_lambda[yvar], lists_y[yvar], bins=(50, 30), cmap=plt.cm.Greys)  # pylint: disable=no-member
        # plt.cm.Greys
        # x = np.array(list_lambda[yvar])
        # y = np.array(lists_y[yvar])
        # from scipy.stats import kde
        #
        # nbins = 30
        # xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)


def make_plot(args) -> None:
    # font = {'size': 16}
    # mpl.rc('font', **font)

    densityplotyvars: list[str] = []
    # densityplotyvars = ['emission_velocity', 'Te', 'nne']
    # densityplotyvars = ['true_emission_velocity', 'emission_velocity', 'Te', 'nne']

    nrows = len(args.timedayslist) if args.multispecplot else 1 + len(densityplotyvars)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharey=False,
        sharex=True,
        squeeze=True,
        figsize=(
            args.figscale * at.get_config()["figwidth"],
            args.figscale * at.get_config()["figwidth"] * (0.25 + nrows * 0.4),
        ),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )

    if nrows == 1:
        axes = [axes]

    filterfunc = at.get_filterfunc(args)

    scale_to_peak = 1.0 if args.normalised else None

    dfalldata = pd.DataFrame()

    if args.multispecplot:
        for ax in axes:
            ax.set_ylabel(r"F$_\lambda$ at 1 Mpc [{}erg/s/cm$^2$/$\mathrm{{\AA}}$]")

    elif args.logscale:
        # don't include the {} that will be replaced with the power of 10 by the custom formatter
        axes[-1].set_ylabel(r"F$_\lambda$ at 1 Mpc [erg/s/cm$^2$/$\mathrm{{\AA}}$]")
    else:
        axes[-1].set_ylabel(r"F$_\lambda$ at 1 Mpc [{}erg/s/cm$^2$/$\mathrm{{\AA}}$]")

    for axis in axes:
        if args.logscale:
            axis.set_yscale("log")
        axis.set_xlim(left=args.xmin, right=args.xmax)

        if (args.xmax - args.xmin) < 2000:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(base=100))
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=10))
        elif (args.xmax - args.xmin) < 11000:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(base=1000))
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=100))
        elif (args.xmax - args.xmin) < 14000:
            axis.xaxis.set_major_locator(ticker.MultipleLocator(base=2000))
            axis.xaxis.set_minor_locator(ticker.MultipleLocator(base=500))

    if densityplotyvars:
        make_contrib_plot(axes[:-1], args.specpath[0], densityplotyvars, args)

    if args.showemission or args.showabsorption:
        legendncol = 2
        if args.internalpackets:
            defaultoutputfile = Path(
                "plotspecinternalemission_{time_days_min:.1f}d_{time_days_max:.1f}d{directionbins}.pdf"
            )
        else:
            defaultoutputfile = Path("plotspecemission_{time_days_min:.1f}d_{time_days_max:.1f}d{directionbins}.pdf")

        plotobjects, plotobjectlabels, dfalldata = make_emissionabsorption_plot(
            Path(args.specpath[0]), axes[-1], filterfunc, args=args, scale_to_peak=scale_to_peak
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
        if args.reverselegendorder:
            plotobjects, plotobjectlabels = plotobjects[::-1], plotobjectlabels[::-1]

        fs = 12 if (args.showemission or args.showabsorption) else None
        leg = axes[-1].legend(
            plotobjects,
            plotobjectlabels,
            loc="upper right",
            frameon=False,
            handlelength=2,
            ncol=legendncol,
            numpoints=1,
            fontsize=fs,
        )
        leg.set_zorder(200)

        for artist, text in zip(leg.legend_handles, leg.get_texts()):
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

    if args.ymin is not None:
        axes[-1].set_ylim(bottom=args.ymin)
    if args.ymax is not None:
        axes[-1].set_ylim(top=args.ymax)

    for index, ax in enumerate(axes):
        # ax.xaxis.set_major_formatter(plt.NullFormatter())

        if "{" in ax.get_ylabel() and not args.logscale:
            ax.yaxis.set_major_formatter(
                at.plottools.ExponentLabelFormatter(ax.get_ylabel(), useMathText=True, decimalplaces=1)
            )

        if args.hidexticklabels:
            ax.tick_params(
                axis="x",
                which="both",
                # bottom=True, top=True,
                labelbottom=False,
            )
        ax.set_xlabel("")

        if args.multispecplot and args.showtime:
            ymin, ymax = ax.get_ylim()
            ax.text(5500, ymax * 0.9, f"{args.timedayslist[index]} days")  # multispecplot text

    axes[-1].set_xlabel(args.xlabel)

    if not args.outputfile:
        args.outputfile = defaultoutputfile
    elif not Path(args.outputfile).suffixes:
        args.outputfile = args.outputfile / defaultoutputfile

    strdirectionbins = ""
    if args.plotviewingangle:
        strdirectionbins = "_direction" + "_".join([f"{angle:02d}" for angle in args.plotviewingangle])

    filenameout = str(args.outputfile).format(
        time_days_min=args.timemin, time_days_max=args.timemax, directionbins=strdirectionbins
    )

    # plt.text(6000, (args.ymax * 0.9), f'{round(args.timemin) + 1} days', fontsize='large')

    if args.showtime and not args.multispecplot:
        if not args.ymax:
            ymin, ymax = ax.get_ylim()
        else:
            ymax = args.ymax
        plt.text(5500, (ymax * 0.9), f"{int(round(args.timemin) + 1)} days", fontsize="large")

    if args.write_data and not dfalldata.empty:
        print(dfalldata)
        datafilenameout = Path(filenameout).with_suffix(".txt")
        dfalldata.to_csv(datafilenameout)
        print(f"Saved {datafilenameout}")

    # plt.minorticks_on()
    # plt.tick_params(axis='x', which='minor', length=5, width=2, labelsize=18)
    # plt.tick_params(axis='both', which='major', length=8, width=2, labelsize=18)

    fig.savefig(filenameout)
    # plt.show()
    print(f"Saved {filenameout}")
    plt.close()


def addargs(parser) -> None:
    parser.add_argument(
        "specpath",
        default=[],
        nargs="*",
        action=at.AppendPath,
        help="Paths to ARTIS folders or reference spectra filenames",
    )

    parser.add_argument("-label", default=[], nargs="*", help="List of series label overrides")

    parser.add_argument("-color", "-colors", dest="color", default=[], nargs="*", help="List of line colors")

    parser.add_argument("-linestyle", default=[], nargs="*", help="List of line styles")

    parser.add_argument("-linewidth", default=[], nargs="*", help="List of line widths")

    parser.add_argument("-dashes", default=[], nargs="*", help="Dashes property of lines")

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

    parser.add_argument("--internalpackets", action="store_true", help="Use non-escaped packets")

    parser.add_argument(
        "--plotpacketcount", action="store_true", help="Plot bin packet counts instead of specific intensity"
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

    parser.add_argument("-timestep", "-ts", dest="timestep", nargs="?", help="First timestep or a range e.g. 45-65")

    parser.add_argument(
        "-timedays", "-time", "-t", dest="timedays", nargs="?", help="Range of times in days to plot (e.g. 50-100)"
    )

    parser.add_argument("-timemin", type=float, help="Lower time in days to integrate spectrum")

    parser.add_argument("-timemax", type=float, help="Upper time in days to integrate spectrum")

    parser.add_argument(
        "-xmin", "-lambdamin", dest="xmin", type=int, default=2500, help="Plot range: minimum wavelength in Angstroms"
    )

    parser.add_argument(
        "-xmax", "-lambdamax", dest="xmax", type=int, default=11000, help="Plot range: maximum wavelength in Angstroms"
    )

    parser.add_argument(
        "-deltalambda", type=int, default=None, help="Lambda bin size in Angstroms (applies to from_packets only)"
    )

    parser.add_argument("-ymin", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument("-ymax", type=float, default=None, help="Plot range: y-axis")

    parser.add_argument(
        "--hidemodeltimerange", action="store_true", help='Hide the "at t=x to yd" from the line labels'
    )

    parser.add_argument("--hidemodeltime", action="store_true", help="Hide the time from the line labels")

    parser.add_argument("--normalised", action="store_true", help="Normalise all spectra to their peak values")

    parser.add_argument(
        "--use_escapetime",
        action="store_true",
        help="Use the time of packet escape to the surface (instead of a plane toward the observer)",
    )

    parser.add_argument(
        "--use_thermalemissiontype",
        action="store_true",
        help="Tag packets by their thermal emission type rather than last scattering process",
    )

    parser.add_argument(
        "-groupby",
        default="ion",
        choices=["ion", "line", "upperterm", "terms"],
        help="Use a different color for each ion or line. Requires showemission and frompackets.",
    )

    parser.add_argument(
        "-obsspec",
        "-refspecfiles",
        action="append",
        dest="refspecfiles",
        help="Also plot reference spectrum from this file",
    )

    parser.add_argument(
        "-fluxdistmpc",
        type=float,
        help=(
            "Plot flux at this distance in megaparsec. Default is the distance to "
            "first reference spectrum if this is known, or otherwise 1 Mpc"
        ),
    )

    parser.add_argument(
        "-scaletoreftime", type=float, default=None, help="Scale reference spectra flux using Co56 decay timescale"
    )

    parser.add_argument("--showbinedges", action="store_true", help="Plot vertical lines at the bin edges")

    parser.add_argument(
        "-figscale", type=float, default=1.8, help="Scale factor for plot area. 1.0 is for single-column"
    )

    parser.add_argument("--logscale", action="store_true", help="Use log scale")

    parser.add_argument("--hidenetspectrum", action="store_true", help="Hide net spectrum")

    parser.add_argument("--hideother", action="store_true", help="Hide other contributions")

    parser.add_argument("--notitle", action="store_true", help="Suppress the top title from the plot")

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument("--reverselegendorder", action="store_true", help="Reverse the order of legend items")

    parser.add_argument("--hidexticklabels", action="store_true", help="Dont show numbers on the x axis")

    parser.add_argument("-xlabel", default=r"Wavelength $\left[\mathrm{{\AA}}\right]$", help="Label for the x axis")

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a CSV file")

    parser.add_argument(
        "-outputfile", "-o", action="store", dest="outputfile", type=Path, help="path/filename for PDF file"
    )

    parser.add_argument(
        "--output_spectra", "--write_spectra", action="store_true", help="Write out all timestep spectra to text files"
    )

    # Combines all vspecpol files into one file which can then be read by artistools
    parser.add_argument(
        "--makevspecpol", action="store_true", help="Make file summing the virtual packet spectra from all ranks"
    )

    # To get better statistics for polarisation use multiple runs of the same simulation. This will then average the
    # files produced by makevspecpol for all simualtions.
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
        help="Plot viewing angles. Expects int for angle number in specpol_res.out",
    )

    parser.add_argument(
        "--average_over_phi_angle",
        action="store_true",
        help="Average over phi (azimuthal) viewing angles to make direction bins into polar angle bins",
    )

    # for backwards compatibility with above option
    parser.add_argument(
        "--average_every_tenth_viewing_angle",
        action="store_true",
    )

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
        "-viewinganglelabelunits", type=str, default="deg", help="Choose viewing angle label in deg or rad"
    )

    parser.add_argument(
        "--classicartis", action="store_true", help="Flag to show using output from classic ARTIS branch"
    )


def main(args=None, argsraw=None, **kwargs) -> None:
    """Plot spectra from ARTIS and reference data."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description=(
                "Plot ARTIS model spectra by finding spec.out files in the current directory or subdirectories."
            ),
        )
        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)
        if args.average_every_tenth_viewing_angle:
            print("WARNING: --average_every_tenth_viewing_angle is deprecated. use --average_over_phi_angle instead")
            args.average_over_phi_angle = True

    at.set_mpl_style()

    if not args.specpath:
        args.specpath = [Path(".")]
    elif isinstance(args.specpath, (str, Path)):  # or not not isinstance(args.specpath, Iterable)
        args.specpath = [args.specpath]

    args.specpath = at.flatten_list(args.specpath)

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
            if Path(filepath).is_dir() or Path(filepath).name.endswith(".out"):
                args.color.append(artismodelcolors[artismodelnum])
                artismodelnum += 1
            else:
                args.color.append(refspeccolors[refspecnum])
                refspecnum += 1

    args.color, args.label, args.linestyle, args.dashes, args.linewidth = at.trim_or_pad(
        len(args.specpath), args.color, args.label, args.linestyle, args.dashes, args.linewidth
    )

    if args.emissionvelocitycut:
        args.frompackets = True

    if args.makevspecpol:
        make_virtual_spectra_summed_file(args.specpath[0])
        return

    if args.averagevspecpolfiles:
        make_averaged_vspecfiles(args)
        return

    if "/" in args.stokesparam:
        plot_polarisation(args.specpath[0], args)
        return

    if args.output_spectra:
        for modelpath in args.specpath:
            at.spectra.write_flambda_spectra(modelpath, args)

    else:
        if args.emissionabsorption:
            args.showemission = True
            args.showabsorption = True

        make_plot(args)


if __name__ == "__main__":
    main()
