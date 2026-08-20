# PYTHON_ARGCOMPLETE_OK
"""Plot ARTIS bolometric and band light curves, colour evolution, and deposition curves."""

import argparse
import math
import sys
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.axes as mplax
import matplotlib.cm as mplcm
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy as np
import numpy.typing as npt
import polars as pl
from polars import selectors as cs

import artistools as at
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import Msun_to_g
from artistools.lightcurve.lightcurve import FILTERNAME_ALIASES
from artistools.lightcurve.lightcurve import lum_lsun_to_mag
from artistools.lightcurve.lightcurve import path_is_reference_lightcurve
from artistools.misc import add_axis_limit_args
from artistools.misc import add_figscale_args
from artistools.misc import add_filter_args
from artistools.misc import add_maxpacketfiles_arg
from artistools.misc import add_modelpath_arg
from artistools.misc import add_outputfile_arg
from artistools.misc import add_series_style_args
from artistools.misc import color_arg
from artistools.misc import get_series_label
from artistools.misc import makelist
from artistools.misc import print_theta_phi_definitions
from artistools.misc import trim_or_pad
from artistools.plottools import get_series_colors
from artistools.plottools import get_unused_colors
from artistools.plottools import iter_axes
from artistools.plottools import save_figure
from artistools.plottools import set_axis_labels
from artistools.plottools import set_prop_cycle_unusedcolors

type LumUnit = t.Literal["mag", "Lsun", "erg/s"]


def get_plot_lum_unit(args: argparse.Namespace) -> LumUnit:
    """Return the luminosity unit that the command-line arguments select for the y axis.

    This is the single source of truth for the unit choice: the light curve column to plot, the conversion
    applied to reference and deposition data, and the y axis label are all derived from it.
    """
    if args.magnitude:
        return "mag"

    return "Lsun" if args.Lsun else "erg/s"


def get_plot_lum_column(lumunit: LumUnit) -> str:
    """Return the light curve dataframe column holding the luminosity in the y axis units."""
    return "mag" if lumunit == "mag" else f"luminosity_{lumunit}"


def shows_deposition(args: argparse.Namespace) -> bool:
    """Return whether the command-line arguments ask for deposition rates on the light curve axis.

    Every deposition flag puts the gamma and beta curves on that axis. Asking is not drawing, though: the y
    axis label follows whether a model actually contributed curves, not this.
    """
    return bool(args.plotdeposition or args.plotalphadeposition or args.plotthermalisation)


def convert_lum_lsun_to_plotunits(
    lum_lsun: npt.NDArray[np.floating[t.Any]], lumunit: LumUnit
) -> npt.NDArray[np.floating[t.Any]]:
    """Convert a luminosity in solar luminosities to the y axis units: bolometric magnitude, Lsun, or erg/s."""
    if lumunit == "mag":
        return lum_lsun_to_mag(lum_lsun)

    return lum_lsun if lumunit == "Lsun" else lum_lsun * Lsun_to_erg_per_s


def convert_lum_ergs_to_plotunits(
    lum_erg_per_s: npt.NDArray[np.floating[t.Any]], lumunit: LumUnit
) -> npt.NDArray[np.floating[t.Any]]:
    """Convert a luminosity in erg/s to the y axis units: bolometric magnitude, Lsun, or erg/s."""
    if lumunit == "erg/s":
        return lum_erg_per_s

    return convert_lum_lsun_to_plotunits(lum_erg_per_s / Lsun_to_erg_per_s, lumunit)


def get_reflightcurve_yerr(
    lum_erg_per_s: npt.NDArray[np.floating[t.Any]],
    errminus_erg_per_s: npt.NDArray[np.floating[t.Any]],
    errplus_erg_per_s: npt.NDArray[np.floating[t.Any]],
    lumunit: LumUnit,
) -> tuple[list[npt.NDArray[np.floating[t.Any]]], npt.NDArray[np.bool_]]:
    """Return the [lower, upper] error bar sizes in the y axis units, and which faint sides are unbounded.

    A magnitude bar whose luminosity reaches zero has no faintest magnitude. Putting NaN there makes
    matplotlib drop the whole bar, including the finite bright half, so that side is given zero length and
    the point is flagged instead: the caller marks it with an arrow.
    """
    if lumunit != "mag":
        # the conversion is a simple scaling, so it applies to the error sizes directly
        return [
            convert_lum_ergs_to_plotunits(errminus_erg_per_s, lumunit),
            convert_lum_ergs_to_plotunits(errplus_erg_per_s, lumunit),
        ], np.zeros(len(lum_erg_per_s), dtype=np.bool_)

    # a magnitude gets smaller as the luminosity gets larger, so the error bars swap sides and become asymmetric
    mag = convert_lum_ergs_to_plotunits(lum_erg_per_s, lumunit)
    hasmag = np.isfinite(mag)
    unbounded = (errminus_erg_per_s >= lum_erg_per_s) & hasmag
    lum_faintest = np.where(unbounded, lum_erg_per_s, lum_erg_per_s - errminus_erg_per_s)

    with np.errstate(invalid="ignore"):
        yerr = [
            mag - convert_lum_ergs_to_plotunits(lum_erg_per_s + errplus_erg_per_s, lumunit),
            convert_lum_ergs_to_plotunits(lum_faintest, lumunit) - mag,
        ]

    # a non-positive luminosity has no magnitude to draw a bar around, and inf bounds make matplotlib warn
    return [np.where(hasmag, errside, 0.0) for errside in yerr], unbounded


def plot_bol_reflightcurve(
    axis: mplax.Axes, lightcurvefilename: str | Path, lumunit: LumUnit, color: str, label: str | None = None
) -> str:
    """Plot an observed bolometric light curve in the y axis units, with error bars if the data file has them.

    Return the label used in the plot legend, which comes from the file metadata unless label is given.
    """
    dflightcurve, metadata = at.lightcurve.read_bol_reflightcurve_data(lightcurvefilename)
    plotlabel = label or str(metadata.get("label", lightcurvefilename))
    lum_erg_per_s = dflightcurve["luminosity_erg/s"].to_numpy()
    time_days = dflightcurve["time_days"].to_numpy()
    yvalues = convert_lum_ergs_to_plotunits(lum_erg_per_s, lumunit)

    if {"luminosity_errminus_erg/s", "luminosity_errplus_erg/s"}.issubset(dflightcurve.columns):
        yerr, unbounded = get_reflightcurve_yerr(
            lum_erg_per_s,
            dflightcurve["luminosity_errminus_erg/s"].to_numpy(),
            dflightcurve["luminosity_errplus_erg/s"].to_numpy(),
            lumunit,
        )
        axis.errorbar(time_days, yvalues, yerr=yerr, fmt="o", capsize=3, label=plotlabel, color=color, zorder=0)
        if unbounded.any():
            # matplotlib draws only one side of a bar it is told is a limit, so the open faint side is a
            # second, zero-length bar whose arrow head sits on the point the bar above already reaches
            axis.errorbar(
                time_days[unbounded],
                yvalues[unbounded],
                yerr=np.zeros(int(unbounded.sum())),
                lolims=True,
                fmt="none",
                color=color,
                zorder=0,
            )
    else:
        axis.scatter(time_days, yvalues, label=plotlabel, color=color, zorder=0)

    return plotlabel


def get_model_folder(modelpath: str | Path) -> Path:
    """Return the model folder, whether modelpath names the folder itself or a file inside it."""
    path = Path(modelpath)

    return path.parent if path.is_file() else path


def plot_deposition_thermalisation(
    axis: mplax.Axes,
    axistherm: mplax.Axes | None,
    modelpath: str | Path,
    modelname: str,
    args: argparse.Namespace,
    linewidth: float | str | None = None,
) -> None:
    """Plot the gamma-ray and positron deposition rates, and the thermalisation efficiencies when axistherm is given.

    Every curve below appends its own suffix to modelname and picks its own linestyle and colour, so the line
    width is the only style the caller sets: passing the rest would reach matplotlib twice.
    """
    lumunit = get_plot_lum_unit(args)

    if args.plotthermalisation:
        dfmodel, _ = at.inputmodel.get_modeldata(modelpath, derived_cols=["mass_g", "vel_r_mid", "kinetic_en_erg"])

        # one collect for both sums: get_modeldata returns a plan, so a second one re-derives every column
        model_mass_grams, ejecta_ke_erg = dfmodel.select(pl.sum("mass_g"), pl.sum("kinetic_en_erg")).collect().row(0)
        print(f"  model mass: {model_mass_grams / Msun_to_g:.3f} Msun")

    depdata = at.get_deposition(modelpath).collect()

    at.plottools.get_next_color(axis)  # skip a colour so the deposition curves differ from the light curve
    color_gamma = at.plottools.get_next_color(axis)
    color_beta = at.plottools.get_next_color(axis)
    # the alpha curves are drawn only on request, so their colour is taken only then: consuming it anyway
    # would step the next model's deposition curves along the cycle for a curve that is never drawn
    plotsalpha = args.plotalphadeposition or args.plotthermalisation
    color_alpha: str | None = at.plottools.get_next_color(axis) if plotsalpha else None

    depositioncurves: list[tuple[str, str, str, str | None]] = [
        ("gammadep_Lsun", r" $\dot{E}_{dep,\gamma}$", "dashed", color_gamma),
        ("eps_elec_Lsun", r" $\dot{E}_{rad,\beta^-}$", "dotted", color_beta),
        ("elecdep_Lsun", r" $\dot{E}_{dep,\beta^-}$", "dashed", color_beta),
    ]

    if args.plotalphadeposition:
        depositioncurves += [
            ("eps_alpha_ana_Lsun", r" $\dot{E}_{rad,\alpha}$ analytical", "solid", color_alpha),
            ("eps_alpha_Lsun", r" $\dot{E}_{rad,\alpha}$", "dashed", color_alpha),
            ("alphadep_Lsun", r" $\dot{E}_{dep,\alpha}$", "dotted", color_alpha),
        ]

    # an older deposition.out has only the gamma columns, so skip any curve whose column is absent
    for depcol, labelsuffix, curvelinestyle, curvecolor in depositioncurves:
        if depcol not in depdata:
            continue
        axis.plot(
            depdata["tmid_days"],
            convert_lum_lsun_to_plotunits(depdata[depcol].to_numpy(), lumunit),
            linewidth=linewidth,
            label=modelname + labelsuffix,
            linestyle=curvelinestyle,
            color=curvecolor,
        )

    if args.plotthermalisation:
        assert axistherm is not None
        # an older deposition.out has only the gamma columns, so skip any ratio whose columns are absent
        thermalisation_ratios = [
            ("gammadep_Lsun", "eps_gamma_Lsun", r"\dot{E}_{dep,\gamma} \middle/ \dot{E}_{rad,\gamma}", color_gamma),
            ("elecdep_Lsun", "eps_elec_Lsun", r"\dot{E}_{dep,\beta^-} \middle/ \dot{E}_{rad,\beta^-}", color_beta),
            ("alphadep_Lsun", "eps_alpha_Lsun", r"\dot{E}_{dep,\alpha} \middle/ \dot{E}_{rad,\alpha}", color_alpha),
        ]
        for depcol, epscol, ratiolabel, ratiocolor in thermalisation_ratios:
            if depcol not in depdata or epscol not in depdata:
                continue
            axistherm.plot(
                depdata["tmid_days"],
                depdata[depcol] / depdata[epscol],
                linewidth=linewidth,
                label=modelname + rf" $\left({ratiolabel}\right)$",
                linestyle="solid",
                color=ratiocolor,
            )

        print(f"  ejecta kinetic energy: {ejecta_ke_erg / 1e7:.2e} [J] = {ejecta_ke_erg:.2e} [erg]")

        # velocity derived from ejecta kinetic energy to match Barnes et al. (2016) Section 2.1
        ejecta_v = np.sqrt(2 * ejecta_ke_erg / model_mass_grams)
        print(f"  Barnes average ejecta velocity: {ejecta_v / C_cm_per_s:.2f}c")
        m5 = model_mass_grams / (5e-3 * Msun_to_g)  # M / (5e-3 Msun)
        v2 = ejecta_v / (0.2 * C_cm_per_s)  # ejecta_v / (0.2c)

        # Barnes et al (2016) scaling form from equation 17, with fiducial t_ineff_gamma of 1.4 days
        t_ineff_gamma = 1.4 * np.sqrt(m5) / v2
        # Barnes et al (2016) equation 33
        barnes_f_gamma = [1 - math.exp(-((t / t_ineff_gamma) ** -2)) for t in depdata["tmid_days"]]

        axistherm.plot(
            depdata["tmid_days"],
            barnes_f_gamma,
            linewidth=linewidth,
            label=r"Barnes+2016 $f_\gamma$",
            linestyle="dashed",
            color=color_gamma,
        )

        e0_beta_mev = 0.5
        # Barnes et al (2016) equation 20
        t_ineff_beta = 7.4 * (e0_beta_mev / 0.5) ** -0.5 * m5**0.5 * (v2 ** (-3.0 / 2))
        # Barnes et al (2016) equation 32
        barnes_f_beta = [
            math.log(1 + 2 * (t / t_ineff_beta) ** 2) / (2 * (t / t_ineff_beta) ** 2) for t in depdata["tmid_days"]
        ]

        axistherm.plot(
            depdata["tmid_days"],
            barnes_f_beta,
            linewidth=linewidth,
            label=r"Barnes+2016 $f_\beta$",
            linestyle="dashed",
            color=color_beta,
        )

        e0_alpha_mev = 6.0
        # Barnes et al (2016) equation 25 times equation 16 for t_peak
        t_ineff_alpha = 4.3 * 1.8 * (e0_alpha_mev / 6.0) ** -0.5 * m5**0.5 * (v2 ** (-3.0 / 2))
        # Barnes et al (2016) equation 32
        barnes_f_alpha = [
            math.log(1 + 2 * (t / t_ineff_alpha) ** 2) / (2 * (t / t_ineff_alpha) ** 2) for t in depdata["tmid_days"]
        ]

        axistherm.plot(
            depdata["tmid_days"],
            barnes_f_alpha,
            linewidth=linewidth,
            label=r"Barnes+2016 $f_\alpha$",
            linestyle="dashed",
            color=color_alpha,
        )


def plot_artis_lightcurve(
    modelpath: str | Path,
    axis: mplax.Axes,
    lcindex: int = 0,
    linelabel: str | None = None,
    escape_type: str = "TYPE_RPKT",
    frompackets: bool = False,
    maxpacketfiles: int | None = None,
    directionbins: Sequence[int] | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    usedegrees: bool = False,
    args: argparse.Namespace | None = None,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
    **plotkwargs: t.Any,
) -> dict[int, pl.DataFrame] | None:
    """Plot one model's bolometric light curve, and return the plotted data per direction bin."""
    if args is None:
        args = argparse.Namespace()
    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)

    # handle e.g. modelpath = 'modelpath/light_curve.out'
    lcfilename = Path(modelpath).name if Path(modelpath).is_file() else None
    modelpath = get_model_folder(modelpath)

    if not modelpath.is_dir():
        print(f"\nWARNING: Skipping because {modelpath} does not exist\n")
        return None

    linelabel_is_custom = linelabel is not None
    assert "label" not in plotkwargs, "label is already set in plotkwargs"
    linelabel = linelabel or at.get_model_name(modelpath)
    assert linelabel is not None
    if escape_type == "TYPE_GAMMA":
        linelabel += r" $\gamma$"
    if pellet_nucname is not None:
        linelabel = rf"$\;$ {pellet_nucname}"

    print(f"====> {linelabel}")
    print(f" modelpath: {modelpath.resolve().parts[-1]}")

    if hasattr(args, "title") and args.title:
        axis.set_title(args.title if isinstance(args.title, str) else linelabel)

    if directionbins is None:
        directionbins = [-1]

    if frompackets:
        lcdataframes = at.lightcurve.get_from_packets(
            modelpath,
            escape_type=escape_type,
            maxpacketfiles=maxpacketfiles,
            directionbins=directionbins,
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
            directionbins_are_vpkt_observers=args.plotvspecpol is not None,
            pellet_nucname=pellet_nucname,
            use_pellet_decay_time=use_pellet_decay_time,
            timedaysmin=args.timemin,
            timedaysmax=args.timemax,
        )
    else:
        assert pellet_nucname is None, "pellet_nucname is only valid with frompackets=True"
        assert not use_pellet_decay_time, "use_pellet_decay_time is only valid with frompackets=True"
        if lcfilename is None:
            lcfilename = (
                "light_curve_res.out"
                if directionbins != [-1]
                else "gamma_light_curve.out"
                if escape_type == "TYPE_GAMMA"
                else "light_curve.out"
            )

        try:
            lcpath = at.firstexisting(lcfilename, folder=modelpath, tryzipped=True)
        except FileNotFoundError:
            print(f"WARNING: Skipping because {lcfilename} does not exist")
            return None

        lcdataframes = at.lightcurve.readfile(
            lcpath, average_over_phi=average_over_phi, average_over_theta=average_over_theta
        )

    lumunit = get_plot_lum_unit(args)
    ycolumn = get_plot_lum_column(lumunit)

    if args.dashes[lcindex]:
        plotkwargs["dashes"] = args.dashes[lcindex]
    if args.linewidth[lcindex]:
        plotkwargs["linewidth"] = args.linewidth[lcindex]

    # check if doing viewing angle stuff, and if so define which data to use
    dirbins, angle_definition = at.lightcurve.parse_directionbin_args(modelpath, args)

    if args.colorbarcostheta or args.colorbarphi:
        costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=usedegrees)
        scaledmap = make_colorbar_viewingangles_colormap()

    lcdataframes = dict(
        zip(
            dirbins,
            pl.collect_all(
                at.misc.df_filter_minmax_bounded(
                    lcdataframes[dirbin], colname="time_days", minval=args.timemin, maxval=args.timemax
                )
                for dirbin in dirbins
            ),
            strict=True,
        )
    )
    lctimemin = lcdataframes[dirbins[0]].select(pl.min("time_days")).item()
    lctimemax = lcdataframes[dirbins[0]].select(pl.max("time_days")).item()
    assert isinstance(lctimemin, float)
    assert isinstance(lctimemax, float)

    print(f" range of light curve: {lctimemin:.2f} to {lctimemax:.2f} days")
    try:
        nts_last, validrange_start_days, validrange_end_days = at.get_escaped_arrivalrange(modelpath)
    except FileNotFoundError:
        print(
            " range of validity: could not determine due to missing files "
            "(requires deposition.out, input.txt, model.txt)"
        )
        nts_last, validrange_start_days, validrange_end_days = None, -math.inf, math.inf
    else:
        if validrange_start_days is not None and validrange_end_days is not None:
            str_valid_range = f"{validrange_start_days:.2f} to {validrange_end_days:.2f} days"
        else:
            str_valid_range = f"{validrange_start_days} to {validrange_end_days} days"
        print(f" range of validity (last timestep {nts_last}): {str_valid_range}")

    if any(dirbin != -1 for dirbin in dirbins):
        print_theta_phi_definitions()

    colorindex: t.Any = None
    for dirbin in dirbins:
        lcdata = lcdataframes[dirbin]

        print(f" directionbin {dirbin:4d}  {angle_definition[dirbin]}", end="")

        if "packetcount" in lcdata.collect_schema().names():
            npkts_selected = lcdata.select(pl.col("packetcount").sum()).item()
            print(f"   \t{npkts_selected:.2e} packets")
        else:
            print()

        filterfunc = at.get_filterfunc(args)
        if filterfunc is not None:
            lcdata = lcdata.with_columns(
                (cs.starts_with("luminosity_") | cs.by_name("mag")).map_batches(
                    filterfunc, return_dtype=pl.self_dtype()
                )
            )

        label_with_tags: str | None = linelabel
        if dirbin != -1:
            if args.colorbarcostheta or args.colorbarphi:
                plotkwargs["alpha"] = 0.75
                if not linelabel_is_custom:
                    label_with_tags = None
                # Update plotkwargs with viewing angle colour
                plotkwargs, colorindex = get_viewinganglecolor_for_colorbar(
                    dirbin, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargs, args
                )
                if args.average_over_phi_angle:
                    plotkwargs["color"] = "lightgrey"
            else:
                # the first dirbin should use the color argument (which has been removed from the color cycle)
                if dirbin != dirbins[0]:
                    plotkwargs["color"] = None
                if len(dirbins) > 1 or not linelabel_is_custom:
                    assert label_with_tags is not None
                    label_with_tags += f" {angle_definition[dirbin]}"

        if pellet_nucname is not None:
            plotkwargs["color"] = None

        if (
            args.average_over_phi_angle
            and dirbin % at.get_viewingdirection_costhetabincount() == 0
            and (args.colorbarcostheta or args.colorbarphi)
        ):
            plotkwargs["color"] = scaledmap.to_rgba(colorindex)  # Update colours for light curves averaged over phi
            plotkwargs["zorder"] = 10

        lcdata_tmin = lcdata.select(pl.col("time_days").min()).item()
        lcdata_tmax = lcdata.select(pl.col("time_days").max()).item()
        lcdata = lcdata.with_columns(time_s=pl.col("time_days") * day_to_s)
        katz_integral = np.trapezoid(
            (
                lcdata
                .select(pl.col("luminosity_erg/s") * pl.col("time_s"))
                .fill_null(0.0)
                .fill_nan(0.0)
                .to_series()
                .to_numpy()
            ),
            x=lcdata["time_s"],
        )
        print(f" Katz integral L t dt ({lcdata_tmin:.2f} to {lcdata_tmax:.2f} days): {katz_integral:.3e} [erg s]")
        # show the parts of the light curve that are outside the valid arrival range as partially transparent
        if validrange_start_days is None or validrange_end_days is None:
            # entire range is invalid
            lcdata_before_valid = lcdata
            lcdata_after_valid = pl.DataFrame(schema=lcdata.schema)
            lcdata_valid = pl.DataFrame(schema=lcdata.schema)
        else:
            lcdata_valid = lcdata.filter(pl.col("time_days").is_between(validrange_start_days, validrange_end_days))
            if lcdata_valid.is_empty():
                # valid range doesn't contain any data points
                lcdata_before_valid = lcdata
                lcdata_after_valid = pl.DataFrame(schema=lcdata.schema)
            else:
                lcdata_before_valid = lcdata.filter(pl.col("time_days") <= lcdata_valid["time_days"].min())
                lcdata_after_valid = lcdata.filter(pl.col("time_days") >= lcdata_valid["time_days"].max())

        if args.plotinvalidpart:
            plotkwargs_invalidrange = plotkwargs.copy()
            plotkwargs_invalidrange.update({"label": None, "alpha": 0.5})
            axis.plot(lcdata_before_valid["time_days"], lcdata_before_valid[ycolumn], **plotkwargs_invalidrange)
            axis.plot(lcdata_after_valid["time_days"], lcdata_after_valid[ycolumn], **plotkwargs_invalidrange)
        elif lcdata_valid.is_empty():
            print(" WARNING: No data points in valid range")

        energy_released = abs(
            np.trapezoid(
                np.nan_to_num(lcdata_valid["luminosity_erg/s"], nan=0.0), x=lcdata_valid["time_days"] * day_to_s
            )
        )

        lcdata_valid_tmin = lcdata_valid.select(pl.col("time_days").min()).item()
        lcdata_valid_tmax = lcdata_valid.select(pl.col("time_days").max()).item()
        if lcdata_valid_tmin is not None and lcdata_valid_tmax is not None:
            print(
                f" Integrated luminosity ({lcdata_valid_tmin:.2f} to {lcdata_valid_tmax:.2f} days): {energy_released:.3e} [erg]"
            )

        axis.plot(lcdata_valid["time_days"], lcdata_valid[ycolumn], label=label_with_tags, **plotkwargs)
        if args.print_data:
            print(lcdata)

        if args.plotcmf:
            assert lumunit != "mag", "Cannot plot cmf luminosity if magnitude is selected"
            plotkwargs["linewidth"] = 1
            if not linelabel_is_custom:
                assert label_with_tags is not None
                label_with_tags += " (cmf)"
            plotkwargs["linestyle"] = "dashed"
            # plotkwargs['color'] = 'tab:orange'
            axis.plot(
                lcdata["time_days"],
                lcdata[ycolumn.replace("luminosity_", "luminosity_cmf_")],
                label=label_with_tags,
                **plotkwargs,
            )

    return lcdataframes


def invert_magnitude_yaxis(ax: mplax.Axes | npt.NDArray[t.Any]) -> None:
    """Point the magnitude axis downwards, so that a brighter series is drawn higher.

    invert_yaxis() toggles rather than sets, so calling it once per plotted series flips the axis back and
    forth instead of inverting it. Inverting a shared y axis already inverts the rest of the grid, which the
    check per axes skips over rather than undoing, so this does not rely on the subplots sharing one axis.
    """
    for axis in iter_axes(ax):
        ymin, ymax = axis.get_ylim()
        if ymin < ymax:
            axis.invert_yaxis()


def make_lightcurve_plot(
    modelpaths: Sequence[str | Path],
    filenameout: str | Path,
    frompackets: bool = False,
    showuvoir: bool = True,
    showgamma: bool = False,
    maxpacketfiles: int | None = None,
    args: argparse.Namespace | None = None,
) -> None:
    """Plot light curves from light_curve.out, gamma_light_curve.out or light_curve_res.out or packets files."""
    if args is None:
        args = argparse.Namespace()

    if "figwidthscale" not in args:
        args.figwidthscale = 1.0

    lumunit = get_plot_lum_unit(args)

    figwidth = args.figscale * 5.0 * args.figwidthscale
    figheight = args.figscale * 5.0 * (0.25 + 0.4)
    fig, axis = plt.subplots(
        nrows=1,
        ncols=1,
        sharex=True,
        figsize=(figwidth, figheight),
        tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
    )
    axis.margins(x=0.0)

    if args.plotthermalisation:
        figtherm, axistherm = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            figsize=(figwidth, figheight),
            tight_layout={"pad": 0.2, "w_pad": 0.0, "h_pad": 0.0},
        )

        axistherm.set_ylabel("Thermalisation ratio")
        axistherm.set_xlabel(r"Time [days]")
    else:
        axistherm = None

    set_prop_cycle_unusedcolors([axis], [*args.color, *args.refspeccolors])

    plottedsomething = False
    plotteddeposition = False
    for lcindex, modelpath in enumerate(modelpaths):
        if path_is_reference_lightcurve(modelpath):
            bolreflightcurve = Path(modelpath)

            lightcurvelabel = plot_bol_reflightcurve(
                axis, bolreflightcurve, lumunit, color=args.color[lcindex], label=args.label[lcindex]
            )
            print(f"====> {lightcurvelabel}")
            plottedsomething = True

        else:
            plottedthismodel = False
            dirbin = args.plotviewingangle or (args.plotvspecpol or [-1])
            escape_types: list[str] = ["TYPE_RPKT"] if showuvoir else []
            if showgamma:
                escape_types.append("TYPE_GAMMA")

            topnucs = args.topnucs
            for escape_type in escape_types:
                pellet_nucnames: list[str | None] = [None]
                if topnucs > 0:
                    try:
                        dfnuclides = at.get_nuclides(modelpath=modelpath)
                        _, dfpackets = at.packets.get_packets(
                            modelpath, maxpacketfiles, packet_type="TYPE_ESCAPE", escape_type=escape_type
                        )
                        top_nuclides = (
                            at.misc
                            .df_filter_minmax_bounded(
                                dfpackets.with_columns(tdecay_d=pl.col("tdecay") / day_to_s),
                                "tdecay_d" if args.use_pellet_decay_time else "t_arrive_d",
                                args.timemin,
                                args.timemax,
                            )
                            .group_by("pellet_nucindex")
                            .agg(pl.sum("e_rf").alias("e_rf_sum"))
                            .top_k(by="e_rf_sum", k=topnucs)
                            .join(dfnuclides, on="pellet_nucindex", how="left")
                            .select(["e_rf_sum", "nucname", "pellet_nucindex"])
                            .collect()
                        )
                        print(top_nuclides)
                        pellet_nucnames.extend(top_nuclides["nucname"])
                    except FileNotFoundError:
                        print("WARNING: no nuclides.out file found, skipping top nuclides")

                for pellet_nucname in pellet_nucnames:
                    lcdataframes = plot_artis_lightcurve(
                        modelpath=modelpath,
                        lcindex=lcindex,
                        axis=axis,
                        escape_type=escape_type,
                        frompackets=frompackets,
                        maxpacketfiles=maxpacketfiles,
                        directionbins=dirbin,
                        average_over_phi=args.average_over_phi_angle,
                        average_over_theta=args.average_over_theta_angle,
                        usedegrees=args.usedegrees,
                        args=args,
                        pellet_nucname=pellet_nucname,
                        use_pellet_decay_time=args.use_pellet_decay_time,
                        linestyle=args.linestyle[lcindex]
                        if (escape_type == "TYPE_RPKT" or len(escape_types) == 1)
                        else ":",
                        color=args.color[lcindex],
                        linelabel=args.label[lcindex],
                    )
                    plottedthismodel = plottedthismodel or (lcdataframes is not None)

            plottedsomething = plottedsomething or plottedthismodel

            if plottedthismodel and shows_deposition(args):
                # the rates belong to the model, not to one escape type or pellet nuclide, and the style
                # comes from the command line rather than from whatever a series left in its plot kwargs
                plot_deposition_thermalisation(
                    axis,
                    axistherm,
                    get_model_folder(modelpath),
                    modelname=get_series_label(args.label, lcindex, at.get_model_name(modelpath)),
                    args=args,
                    linewidth=args.linewidth[lcindex] or None,
                )
                plotteddeposition = True

        print()

    if args.reflightcurves:
        for refindex, bolreflightcurve in enumerate(args.reflightcurves):
            plot_bol_reflightcurve(axis, bolreflightcurve, lumunit, color=args.refspeccolors[refindex])
            plottedsomething = True

    assert plottedsomething, "No light curve was plotted"

    if not args.nolegend:
        axis.legend(loc="best", handlelength=2, frameon=args.legendframeon, numpoints=1, prop={"size": 9})
        if args.plotthermalisation:
            assert axistherm is not None
            axistherm.legend(
                loc="upper right", handlelength=2, frameon=args.legendframeon, numpoints=1, prop={"size": 9}
            )

    axis.set_xlabel(r"Time [days]")

    if lumunit == "mag":
        # the gamma-ray light curve and the deposition rates are converted onto this same magnitude axis, so
        # the label must name what is drawn rather than always claiming a bolometric luminosity
        ylabel = r"Absolute $\gamma$-ray Magnitude" if showgamma and not showuvoir else "Absolute Bolometric Magnitude"
        if plotteddeposition:
            ylabel += r" ($L$ or $\dot{E}$)"
        axis.set_ylabel(ylabel)
    else:
        str_units = r" [{}$\mathrm{{L}}_\odot$]" if lumunit == "Lsun" else " [{}erg/s]"
        if args.logscaley:
            str_units = str_units.replace("{}", "")
        if plotteddeposition:
            yvarname = r"$L$ or $\dot{{E}}$"
        elif showgamma and not showuvoir:
            yvarname = r"$\mathrm{{L}}_\gamma$"
        elif showuvoir and not showgamma:
            yvarname = r"$\mathrm{{L}}_{{\mathrm{{UVOIR}}}}$"
        else:
            yvarname = r"$\mathrm{{L}}$"

        axis.set_ylabel(yvarname + str_units)

        if "{" in axis.get_ylabel() and not args.logscaley:
            axis.yaxis.set_major_formatter(at.plottools.ExponentLabelFormatter(axis.get_ylabel()))
            axis.yaxis.set_major_locator(
                mplticker.MaxNLocator(nbins="auto", steps=[1, 2, 4, 5, 8, 10], integer=True, prune=None)
            )

    if args.colorbarcostheta or args.colorbarphi:
        _, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(usedegrees=args.usedegrees)
        scaledmap = make_colorbar_viewingangles_colormap()
        make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, ax=axis)

    # set the limits only now that the data is drawn: on an empty axes matplotlib turns autoscaling off, so a
    # one-sided limit would freeze the other side at the default 0-1 view instead of fitting the light curves
    at.plottools.set_axis_properties(axis, args)
    if lumunit == "mag":
        # invert last: set_ylim re-sorts the limits into the order of the pair it is given, so an inversion
        # applied before a one-sided limit is lost
        invert_magnitude_yaxis(axis)

    if args.plotthermalisation:
        assert axistherm is not None
        # the second figure covers the same times as the first, so take its range rather than re-deriving it
        axistherm.set_xlim(axis.get_xlim())
        # a thermalisation efficiency is a ratio, so keep the physical range rather than letting a
        # near-zero denominator at one timestep rescale every curve into the bottom of the panel
        axistherm.set_ylim(0.0, 1.0)

    if args.show:
        plt.show()

    save_figure(fig, filenameout, format="pdf")

    if args.plotthermalisation:
        assert figtherm is not None

        filenameout2 = str(filenameout).replace(".pdf", "_thermalisation.pdf")
        save_figure(figtherm, filenameout2, format="pdf")


def create_axes(args: argparse.Namespace) -> tuple[mplfig.Figure, npt.NDArray[t.Any] | mplax.Axes]:
    """Return the figure and axes, using a subplot grid when several filters or colours are plotted."""
    if "labelfontsize" in args:
        font = {"size": args.labelfontsize}
        mpl.rc("font", **font)

    args.subplots = False  # TODO: set as command line arg

    if (args.filter and len(args.filter) > 1) or args.subplots:
        args.subplots = True
        rows = 2
        cols = 3
    elif args.colour_evolution and len(args.colour_evolution) > 1:
        args.subplots = True
        rows = 1
        cols = 3
    else:
        args.subplots = False
        rows = 1
        cols = 1

    if "figwidthscale" not in args:
        args.figwidthscale = 1.0
    if "figwidth" not in args:
        args.figwidth = 5.0 * 1.6 * cols * args.figwidthscale
    if "figheight" not in args:
        args.figheight = 5.0 * 1.1 * rows * 1.5

    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        sharex=True,
        sharey=True,
        figsize=(args.figwidth, args.figheight),
        tight_layout={"pad": 3.0, "w_pad": 0.6, "h_pad": 0.6},
    )  # (6.2 * 3, 9.4 * 3)

    if args.subplots:
        assert isinstance(ax, np.ndarray)
        ax = ax.flatten()

    return fig, ax


def get_linelabel(
    modelname: str,
    modelnumber: int,
    angle: int | None,
    angle_definition: dict[int, str] | None,
    args: argparse.Namespace,
) -> str:
    """Return the legend label for one series, from the model name and viewing angle."""
    serieslabel = get_series_label(args.label, modelnumber, modelname)
    if angle is not None and angle != -1:
        assert angle_definition is not None
        return angle_definition[angle] if args.nomodelname else f"{serieslabel} {angle_definition[angle]}"
    return serieslabel


def set_lightcurveplot_legend(ax: mplax.Axes | npt.NDArray[t.Any], args: argparse.Namespace) -> None:
    """Add the legend, placing it on args.legendsubplotnumber when the figure has subplots."""
    if args.nolegend:
        return

    if args.subplots:
        axis = iter_axes(ax)[args.legendsubplotnumber]
        axis.legend(loc=args.legendposition, frameon=args.legendframeon, fontsize="x-small", ncol=args.ncolslegend)
    else:
        assert isinstance(ax, mplax.Axes)
        ax.legend(
            loc=args.legendposition,
            frameon=args.legendframeon,
            fontsize="small",
            ncol=args.ncolslegend,
            handlelength=0.7,
        )


def set_lightcurve_plot_labels(
    fig: mplfig.Figure,
    ax: mplax.Axes | npt.NDArray[t.Any],
    args: argparse.Namespace,
    band_name: str | None = None,
    colour_evolution: bool = False,
) -> tuple[mplfig.Figure, mplax.Axes | npt.NDArray[t.Any]]:
    """Set the axis labels and limits for a band magnitude or colour evolution plot.

    The caller states which kind of plot this is rather than it being read back off args, which cannot tell a
    colour plot from a band plot on its own.
    """
    if colour_evolution:
        ylabel = r"$\Delta$m"
    elif args.filter:
        if args.subplots:
            # the subplots layout shares one figure-level label, so it cannot name a particular band
            ylabel = "Absolute Magnitude"
        else:
            assert band_name is not None, "a single-axes band plot needs its band name for the y label"
            ylabel = f"{FILTERNAME_ALIASES.get(band_name, band_name)} Magnitude"
    else:
        msg = "No filter or colour evolution specified"
        raise AssertionError(msg)

    set_axis_labels(fig, ax, "Time Since Explosion [days]", ylabel, args.labelfontsize, args)

    return fig, ax


def make_colorbar_viewingangles_colormap() -> t.Any:
    """Return a tab10 scalar mappable covering the ten viewing angle bins."""
    norm = mplcolors.Normalize(vmin=0, vmax=9)
    scaledmap = mplcm.ScalarMappable(cmap="tab10", norm=norm)
    scaledmap.set_array([])
    return scaledmap


def get_viewinganglecolor_for_colorbar(
    angle: int,
    costheta_viewing_angle_bins: list[str],  # ruff:ignore[unused-function-argument]
    phi_viewing_angle_bins: list[str],  # ruff:ignore[unused-function-argument]
    scaledmap: t.Any,
    plotkwargs: dict[str, t.Any],
    args: argparse.Namespace,
) -> tuple[dict[str, t.Any], int]:
    """Set the series colour from the direction bin's cos(theta) or phi, and return the kwargs and the colour index."""
    nphibins = at.get_viewingdirection_phibincount()
    if args.colorbarcostheta:
        costheta_index = angle // nphibins
        colorindex = costheta_index
        plotkwargs["color"] = scaledmap.to_rgba(colorindex)
    if args.colorbarphi:
        phi_index = angle % nphibins
        assert nphibins == 10
        reorderphibins = {5: 9, 6: 8, 7: 7, 8: 6, 9: 5}
        print("Reordering phi bins")
        colorindex = reorderphibins.get(phi_index, phi_index)
        plotkwargs["color"] = scaledmap.to_rgba(colorindex)

    return plotkwargs, colorindex


def make_colorbar_viewingangles(
    phi_viewing_angle_bins: list[str],  # ruff:ignore[unused-function-argument]
    scaledmap: t.Any,
    args: argparse.Namespace,
    fig: mplfig.Figure | None = None,
    ax: mplax.Axes | Iterable[mplax.Axes] | None = None,
) -> None:
    """Add a colorbar labelled with the cos(theta) or phi viewing angle bin boundaries."""
    if args.colorbarcostheta:
        # ticklabels = costheta_viewing_angle_bins
        ticklabels = [" -1", " -0.8", " -0.6", " -0.4", " -0.2", " 0", " 0.2", " 0.4", " 0.6", " 0.8", " 1"]
        ticklocs = list(np.linspace(0, 9, num=11, dtype=float))
        label = "cos θ"
    if args.colorbarphi:
        print("reordered phi bins")
        phi_viewing_angle_bins_reordered = [
            "0",
            "π/5",
            "2π/5",
            "3π/5",
            "4π/5",
            "π",
            "6π/5",
            "7π/5",
            "8π/5",
            "9π/5",
            "2π",
        ]
        ticklabels = phi_viewing_angle_bins_reordered
        # ticklabels = phi_viewing_angle_bins
        ticklocs = list(np.linspace(0, 9, num=11, dtype=float))
        label = "ϕ bin"

    hidecolorbar = False
    if not hidecolorbar:
        if fig:
            cbar = fig.colorbar(scaledmap, orientation="horizontal", location="top", pad=0.10, ax=ax, shrink=0.95)
        else:
            assert ax is not None
            firstaxis = iter_axes(ax)[0]
            axisfigure = firstaxis.get_figure()
            assert axisfigure is not None
            cbar = axisfigure.colorbar(scaledmap, ax=ax)
        if label:
            cbar.set_label(label, rotation=0)
        cbar.locator = mplticker.FixedLocator(ticklocs)
        cbar.formatter = mplticker.FixedFormatter(ticklabels)
        cbar.update_ticks()


def make_band_lightcurves_plot(
    modelpaths: Sequence[str | Path], outputfolder: Path | str, args: argparse.Namespace
) -> None:
    """Plot band magnitude light curves for every model and save the figure."""
    if args.labelfontsize is None:
        args.labelfontsize = 22
    fig, ax = create_axes(args)
    axes = iter_axes(ax)

    # a model with several direction bins takes its line colours from the cycle, so keep the cycle clear of
    # the colours that other series were given
    set_prop_cycle_unusedcolors(axes, [*args.color, *args.refspeccolors])

    plotkwargs: dict[str, t.Any] = {}

    if args.colorbarcostheta or args.colorbarphi:
        costheta_viewing_angle_bins, phi_viewing_angle_bins = at.get_costhetabin_phibin_labels(
            usedegrees=args.usedegrees
        )
        scaledmap = make_colorbar_viewingangles_colormap()

    first_band_name = None
    bandnames: list[str] = []
    for modelnumber, modelpath in enumerate(Path(m) for m in modelpaths):
        # check if doing viewing angle stuff, and if so define which data to use
        dirbins, dirbin_definition = at.lightcurve.parse_directionbin_args(modelpath, args)

        for dirbin in dirbins:
            modelname = at.get_model_name(modelpath)
            print(f"Reading spectra: {modelname} (angle {dirbin})")
            band_lightcurve_data = at.lightcurve.generate_band_lightcurve_data(
                modelpath, args, dirbin, modelnumber=modelnumber
            )
            bandnames = list(band_lightcurve_data)

            if modelnumber == 0 and args.plot_hesma_model:  # TODO: does this work?
                hesma_model = at.lightcurve.read_hesma_lightcurve(args)
                plotkwargs["label"] = str(args.plot_hesma_model).split("_")[:3]

            for plotnumber, band_name in enumerate(band_lightcurve_data):
                axis = axes[plotnumber]
                if first_band_name is None:
                    first_band_name = band_name
                time, brightness_in_mag = at.lightcurve.get_band_lightcurve(band_lightcurve_data, band_name, args)

                if args.print_data or args.write_data:
                    txtlinesout = [f"# band: {band_name}", f"# model: {modelname}", "# time_days magnitude"]
                    txtlinesout.extend(f"{t_d} {m}" for t_d, m in zip(time, brightness_in_mag, strict=False))
                    txtout = "\n".join(txtlinesout)
                if args.write_data:
                    bandoutfile = (
                        Path(f"band_{band_name}_angle_{dirbin}.txt") if dirbin != -1 else Path(f"band_{band_name}.txt")
                    )
                    with bandoutfile.open("w", encoding="utf-8") as f:
                        f.write(txtout)
                    at.print_saved(bandoutfile)
                if args.print_data:
                    print(txtout)

                plotkwargs["label"] = get_linelabel(modelname, modelnumber, dirbin, dirbin_definition, args)

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    brightness_in_mag = filterfunc(brightness_in_mag)

                if (
                    modelnumber == 0 and args.plot_hesma_model and band_name in hesma_model.columns
                ):  # TODO: see if this works
                    assert isinstance(ax, mplax.Axes)
                    ax.plot(hesma_model["t"], hesma_model[band_name], color="black")

                text_key = FILTERNAME_ALIASES.get(band_name, band_name)

                if args.subplots:
                    assert isinstance(text_key, str)
                    axis.annotate(
                        text_key,
                        xy=(1.0, 1.0),
                        xycoords="axes fraction",
                        textcoords="offset points",
                        xytext=(-30, -30),
                        horizontalalignment="right",
                        verticalalignment="top",
                    )

                # a model with several direction bins has only one -color entry to share between them, so let
                # matplotlib pick a colour per line. plotkwargs is reused, so clear any previous model's colour
                plotkwargs["color"] = args.color[modelnumber] if len(dirbins) == 1 else None

                if args.colorbarcostheta or args.colorbarphi:
                    # Update plotkwargs with viewing angle colour
                    plotkwargs["label"] = None
                    plotkwargs, _ = get_viewinganglecolor_for_colorbar(
                        dirbin, costheta_viewing_angle_bins, phi_viewing_angle_bins, scaledmap, plotkwargs, args
                    )

                plotkwargs["linestyle"] = args.linestyle[modelnumber]

                axis.plot(time, brightness_in_mag, linewidth=4 if args.subplots else 3.5, **plotkwargs)

    # once for the whole figure: the helper draws every band of a reference file onto its own panel, so
    # calling it per band drew each reference curve once per band and re-read the file each time. It also
    # asserted a single axes, which a figure with more than one band never has
    for refindex, reflightcurve in enumerate(args.reflightcurves):
        plot_lightcurve_from_refdata(
            bandnames, reflightcurve, args.refspeccolors[refindex], args.refspecmarkers[refindex], ax
        )

    at.set_mpl_style()

    ax = at.plottools.set_axis_properties(ax, args)
    fig, ax = set_lightcurve_plot_labels(fig, ax, args, band_name=first_band_name)
    set_lightcurveplot_legend(ax, args)

    if args.colorbarcostheta or args.colorbarphi:
        make_colorbar_viewingangles(phi_viewing_angle_bins, scaledmap, args, fig=fig, ax=ax)

    if args.filter and len(band_lightcurve_data) == 1:
        args.outputfile = Path(outputfolder, f"plot{first_band_name}lightcurves.pdf")
    invert_magnitude_yaxis(ax)

    if args.show:
        plt.show()

    save_figure(fig, args.outputfile, format="pdf")


def colour_evolution_plot(modelpaths: Sequence[str | Path], outputfolder: str | Path, args: argparse.Namespace) -> None:
    """Plot the evolution of the colour between each pair of bands for every model, and save the figure."""
    if args.labelfontsize is None:
        args.labelfontsize = 24
    angle_counter = 0
    # a direction bin must not repeat the colour that a whole model was given, since both go on the same axes
    tab20colors = list(plt.get_cmap("tab20")(np.linspace(0, 1.0, 20)))
    color_list = get_unused_colors(tab20colors, [*args.color, *args.refspeccolors]) or tab20colors

    fig, ax = create_axes(args)
    axes = iter_axes(ax)

    # the filter pairs share bands, so integrate the bands of every pair once per direction bin
    bandnames = sorted({name for filters in args.colour_evolution for name in filters.split("-")})

    for modelnumber, modelpath in enumerate(modelpaths):
        modelname = at.get_model_name(modelpath)
        print(f"Reading spectra: {modelname}")

        dirbins, dirbin_definition = at.lightcurve.parse_directionbin_args(modelpath, args)

        for index, dirbin in enumerate(dirbins):
            if len(dirbins) > 1:
                # -color has one colour per model, which cannot distinguish a model's direction bins, so take a
                # colour per bin from the colour map instead, wrapping around when it runs out. The colour is
                # chosen once per bin, so a bin keeps its colour across the subplots of the filter pairs.
                dirbincolor = color_list[angle_counter % len(color_list)]
                angle_counter += 1
            else:
                dirbincolor = args.color[modelnumber]

            band_lightcurve_data = at.lightcurve.generate_band_lightcurve_data(
                modelpath, args, dirbin=dirbin, modelnumber=modelnumber, filternames=bandnames
            )

            for plotnumber, filters in enumerate(args.colour_evolution):
                filter_names = filters.split("-")
                plot_times, colour_delta_mag = at.lightcurve.get_colour_delta_mag(band_lightcurve_data, filter_names)

                filterfunc = at.get_filterfunc(args)
                if filterfunc is not None:
                    colour_delta_mag = filterfunc(colour_delta_mag)

                if args.reflightcurves and modelnumber == 0:
                    if len(dirbins) > 1 and index > 0:
                        print("already plotted reflightcurve")
                    else:
                        for i, reflightcurve in enumerate(args.reflightcurves):
                            plot_color_evolution_from_data(
                                filter_names,
                                reflightcurve,
                                args.refspeccolors[i],
                                args.refspecmarkers[i],
                                ax,
                                plotnumber,
                                args,
                            )

                axes[plotnumber].plot(
                    plot_times,
                    colour_delta_mag,
                    label=get_linelabel(modelname, modelnumber, dirbin, dirbin_definition, args),
                    color=dirbincolor,
                    linestyle=args.linestyle[modelnumber],
                    linewidth=4 if args.subplots else 3,
                )

    for plotnumber, filters in enumerate(args.colour_evolution):
        axes[plotnumber].annotate(
            filters,
            xy=(1.0, 1.0),
            xycoords="axes fraction",
            textcoords="offset points",
            xytext=(-30, -30),
            horizontalalignment="right",
            verticalalignment="top",
            fontsize="x-large",
        )

    fig, ax = set_lightcurve_plot_labels(fig, ax, args, colour_evolution=True)
    ax = at.plottools.set_axis_properties(ax, args)
    set_lightcurveplot_legend(ax, args)

    invert_magnitude_yaxis(ax)

    args.outputfile = Path(outputfolder, f"plotcolorevolution{'_'.join(args.colour_evolution)}.pdf")

    if args.show:
        plt.show()
    save_figure(fig, args.outputfile, format="pdf")


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


def plot_lightcurve_from_refdata(
    filter_names: Sequence[str],
    lightcurvefilename: Path | str,
    color: t.Any,
    marker: t.Any,
    ax: npt.NDArray[t.Any] | mplax.Axes,
) -> str | None:
    """Plot an observed band light curve, dereddened with CCM89, and return its legend label."""
    from extinction import apply
    from extinction import ccm89

    lightcurve_data, metadata = at.lightcurve.read_reflightcurve_band_data(lightcurvefilename)
    linename = metadata["label"]
    assert linename is None or isinstance(linename, str)
    filterdir = Path(at.get_path("artistools_dir"), "data/filters/")

    filter_data = {}
    axes = iter_axes(ax)
    for axnumber, filter_name_raw in enumerate(filter_names):
        axis = axes[axnumber]
        if filter_name_raw == "bol":
            continue
        with Path(filterdir / f"{filter_name_raw}.txt").open(encoding="utf-8") as f:
            lines = f.readlines()
        lambda0 = float(lines[2])

        if filter_name_raw == "bol":
            continue
        filter_name = FILTERNAME_ALIASES.get(filter_name_raw, filter_name_raw)
        filter_data[filter_name] = lightcurve_data.filter(pl.col("band") == filter_name)
        # plt.plot(limits_x, limits_y, 'v', label=None, color=color)
        # else:

        if "a_v" in metadata or "e_bminusv" in metadata:
            print("Correcting for reddening")

            clightinangstroms = 3e18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.full(filter_data[filter_name].height, lambda0, dtype=float)

            flux = (
                clightinangstroms
                / (lambda0**2)
                * 10 ** -((filter_data[filter_name]["magnitude"].to_numpy() + 48.6) / 2.5)
            )  # gs

            dered = apply(ccm89(filters, a_v=-metadata["a_v"], r_v=metadata["r_v"]), flux)

            filter_data[filter_name] = filter_data[filter_name].with_columns(
                pl.Series("magnitude", 2.5 * np.log10(clightinangstroms / (dered * lambda0**2)) - 48.6)
            )
        else:
            print("WARNING: did not correct for reddening")

        axis.plot(
            filter_data[filter_name]["time"],
            filter_data[filter_name]["magnitude"],
            marker,
            label=linename,
            color=color,
            linewidth=4 if len(filter_names) == 1 else None,
        )
    return linename


def plot_color_evolution_from_data(
    filter_names: Iterable[str],
    lightcurvefilename: Path | str,
    color: t.Any,
    marker: t.Any,
    ax: npt.NDArray[t.Any] | mplax.Axes,
    plotnumber: int,
    args: argparse.Namespace,
) -> None:
    """Plot the observed colour evolution between two bands, dereddened with CCM89."""
    from extinction import apply
    from extinction import ccm89

    lightcurve_from_data, metadata = at.lightcurve.read_reflightcurve_band_data(lightcurvefilename)
    filterdir = Path(at.get_path("artistools_dir"), "data/filters/")

    filter_data = []
    for i, filter_name_raw in enumerate(filter_names):
        with (filterdir / Path(f"{filter_name_raw}.txt")).open(encoding="utf-8") as f:
            lines = f.readlines()
        lambda0 = float(lines[2])

        filter_name = FILTERNAME_ALIASES.get(filter_name_raw, filter_name_raw)
        filter_data.append(lightcurve_from_data.filter(pl.col("band") == filter_name))

        if "a_v" in metadata or "e_bminusv" in metadata:
            print("Correcting for reddening")
            if "r_v" not in metadata:
                metadata["r_v"] = metadata["a_v"] / metadata["e_bminusv"]
            elif "a_v" not in metadata:
                metadata["a_v"] = metadata["e_bminusv"] * metadata["r_v"]

            clightinangstroms = 3e18
            # Convert to flux, deredden, then convert back to magnitudes
            filters = np.full(filter_data[i].height, lambda0, dtype=float)

            flux = clightinangstroms / (lambda0**2) * 10 ** -((filter_data[i]["magnitude"].to_numpy() + 48.6) / 2.5)

            dered = apply(ccm89(filters, a_v=-metadata["a_v"], r_v=metadata["r_v"]), flux)

            filter_data[i] = filter_data[i].with_columns(
                pl.Series("magnitude", 2.5 * np.log10(clightinangstroms / (dered * lambda0**2)) - 48.6)
            )

    # for i in range(2):
    #     # if metadata['label'] == 'SN 2018byg':
    #     #     filter_data[i] = filter_data[i][filter_data[i].e_magnitude != -99.00]
    #     if metadata['label'] in ['SN 2016jhr', 'SN 2018byg']:
    #         filter_data[i]['time'] = filter_data[i]['time'].apply(lambda x: round(float(x)))  # round to nearest day

    merge_dataframes = filter_data[0].join(
        filter_data[1], how="inner", on="time", suffix="_second", maintain_order="left"
    )
    axis = iter_axes(ax)[plotnumber]
    axis.plot(
        merge_dataframes["time"],
        merge_dataframes["magnitude"] - merge_dataframes["magnitude_second"],
        marker,
        label=metadata["label"],
        color=color,
        linewidth=4 if args.subplots else None,
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    add_modelpath_arg(
        parser,
        positional=True,
        multiplepaths=True,
        default=[],
        helptext="Path(s) to ARTIS folders with light_curve.out or packets files (may include wildcards such as * and **)",
    )

    add_series_style_args(parser)

    parser.add_argument("--nolegend", action="store_true", help="Suppress the legend from the plot")

    parser.add_argument(
        "-title",
        "--title",
        dest="title",
        nargs="?",
        const=True,
        default=None,
        help="Show a plot title: pass the title text, or use the bare flag for the model name",
    )

    add_figscale_args(parser, figscaledefault=1.8, include_figwidthscale=True)

    parser.add_argument("--frompackets", action="store_true", help="Read packets files instead of light_curve.out")

    add_maxpacketfiles_arg(parser)

    parser.add_argument("--gamma", action="store_true", help="Make light curve from gamma rays")

    parser.add_argument(
        "--rpkt",
        "--uvoir",
        dest="rpkt",
        action="store_true",
        help="Make light curve from R-packets (default unless --gamma is passed)",
    )

    parser.add_argument("-escape_type", default="TYPE_RPKT", help="Type of escaping packets")

    add_outputfile_arg(parser, helptext="Filename for PDF file")

    parser.add_argument(
        "--plotcmf",
        "--plot_cmf",
        "--showcmf",
        "--show_cmf",
        action="store_true",
        help="Plot comoving frame light curve",
    )

    parser.add_argument(
        "--plotinvalidpart",
        action="store_true",
        help="Plot the entire light curve including partially accumulated parts (light travel time effects)",
    )

    parser.add_argument("--plotdeposition", action="store_true", help="Plot model deposition rates")

    parser.add_argument(
        "--plotalphadeposition", action="store_true", help="Plot alpha decay energy release and deposition rates"
    )

    parser.add_argument(
        "--plotthermalisation", action="store_true", help="Plot thermalisation rates (in separate plot)"
    )

    parser.add_argument(
        "-topnucs", type=int, default=0, help="Show light curves from top n nuclides energy contributions."
    )

    parser.add_argument(
        "--use_pellet_decay_time", action="store_true", help="Use pellet decay time instead of observer arrival time"
    )

    parser.add_argument("--magnitude", action="store_true", help="Plot light curves in magnitudes")

    parser.add_argument("--Lsun", action="store_true", help="Plot light curves in units of Lsun")

    parser.add_argument(
        "-filter",
        "-band",
        dest="filter",
        type=str,
        nargs="+",
        help=(
            "Choose filter eg. bol U B V R I. Default B. "
            "WARNING: filter names are not case sensitive eg. sloan-r is not r, it is rs"
        ),
    )

    parser.add_argument("-colour_evolution", nargs="*", help="Plot of colour evolution. Give two filters eg. B-V")

    parser.add_argument("--print_data", action="store_true", help="Print plotted data")

    parser.add_argument("--write_data", action="store_true", help="Save data used to generate the plot in a text file")

    parser.add_argument(
        "-plot_hesma_model",
        action="store",
        type=Path,
        default=False,
        help="Plot hesma model on top of lightcurve plot. Enter model name saved in data/hesma directory",
    )

    at.add_viewingangle_args(parser, allow_select_all=True)

    add_axis_limit_args(parser, include_x=False)

    parser.add_argument(
        "-timemin", "-timedaysmin", "-xmin", type=float, default=None, help="Plot range: x-axis minimum"
    )

    parser.add_argument(
        "-timemax", "-timedaysmax", "-xmax", type=float, default=None, help="Plot range: x-axis maximum"
    )

    parser.add_argument("--logscalex", action="store_true", help="Use log scale for horizontal axis")

    parser.add_argument("--logscaley", action="store_true", help="Use log scale for vertical axis")

    parser.add_argument(
        "-reflightcurves",
        type=str,
        nargs="+",
        dest="reflightcurves",
        help="Also plot reference lightcurves from these files",
    )

    parser.add_argument(
        "-refspeccolors",
        type=color_arg,
        default=[],
        nargs="*",
        help="Set a list of colors for the reference light curves",
    )

    parser.add_argument(
        "-refspecmarkers", default=[], nargs="*", help="Set a list of markers for the reference light curves"
    )

    add_filter_args(parser)

    parser.add_argument(
        "-redshifttoz",
        type=float,
        nargs="+",
        help="Redshift to z = x. Expects array length of number modelpaths.If not to be redshifted then = 0.",
    )

    parser.add_argument("--show", action="store_true", default=False, help="Show plot before saving")

    # parser.add_argument('--calculate_peakmag_risetime_delta_m15', action='store_true',
    #                     help='Calculate band risetime, peak mag and delta m15 values for '
    #                     'the models specified using a polynomial fitting method and '
    #                     'print to screen')

    parser.add_argument(
        "--save_angle_averaged_peakmag_risetime_delta_m15_to_file",
        action="store_true",
        help="Save the band risetime, peak mag and delta m15 values for the angle averaged model lightcurves to file",
    )

    parser.add_argument(
        "--save_viewing_angle_peakmag_risetime_delta_m15_to_file",
        action="store_true",
        help=(
            "Save the band risetime, peak mag and delta m15 values for "
            "all viewing angles specified for plotting at a later time "
            "as these values take a long time to calculate for all "
            "viewing angles. Need to run this command first alongside "
            "-plotviewingangle in order to save the data for the "
            "viewing angles you want to use before making the scatter"
            "plots"
        ),
    )

    parser.add_argument(
        "--test_viewing_angle_fit",
        action="store_true",
        help=(
            "Plots the lightcurves for each  viewing angle along with"
            "the polynomial fit for each viewing angle specified"
            "to check the fit is working properly: use alongside"
            "-plotviewingangle "
        ),
    )

    parser.add_argument(
        "--make_viewing_angle_peakmag_risetime_scatter_plot",
        action="store_true",
        help=(
            "Makes scatter plot of band peak mag with risetime with the "
            "angle averaged values being the solid dot and the errors bars"
            "representing the standard deviation of the viewing angle"
            "distribution"
        ),
    )

    parser.add_argument(
        "--make_viewing_angle_peakmag_delta_m15_scatter_plot",
        action="store_true",
        help=(
            "Makes scatter plot of band peak with delta m15 with the angle"
            "averaged values being the solid dot and the error bars representing "
            "the standard deviation of the viewing angle distribution"
        ),
    )

    parser.add_argument(
        "--include_delta_m40",
        action="store_true",
        help="When calculating delta_m15, calculate delta_m40 as well.Only affects the saved viewing angle data.",
    )

    parser.add_argument(
        "--noerrorbars", action="store_true", help="Don't plot error bars on viewing angle scatter plots"
    )

    parser.add_argument(
        "--noangleaveraged", action="store_true", help="Don't plot angle averaged values on viewing angle scatter plots"
    )

    parser.add_argument(
        "--colorbarcostheta", action="store_true", help="Colour viewing angles by cos theta and show color bar"
    )

    parser.add_argument("--colorbarphi", action="store_true", help="Colour viewing angles by phi and show color bar")

    parser.add_argument(
        "--colouratpeak", action="store_true", help="Make scatter plot of colour at peak for viewing angles"
    )

    parser.add_argument(
        "--brightnessattime",
        action="store_true",
        help="Make scatter plot of light curve brightness at a given time (requires timedays)",
    )

    parser.add_argument("-timedays", "-time", "-t", type=float, help="Time in days to plot")

    parser.add_argument("--nomodelname", action="store_true", help="Model name not added to linename in legend")

    parser.add_argument(
        "-legendsubplotnumber", type=int, default=1, help="Subplot number to place legend in. Default is subplot[1]"
    )

    parser.add_argument("-legendposition", type=str, default="best", help="Position of legend in plot. Default is best")

    parser.add_argument("-ncolslegend", type=int, default=1, help="Number of columns in legend")

    parser.add_argument("--legendframeon", action="store_true", help="Frame on in legend")

    parser.add_argument(
        "-labelfontsize",
        type=float,
        default=None,
        help="Base font size for plot text (axis labels, tick labels, legend). Defaults to 22 for band light curves and 24 for colour evolution plots",
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS light curve."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if getattr(args, "average_every_tenth_viewing_angle", False):
        print("WARNING: --average_every_tenth_viewing_angle is deprecated. use --average_over_phi_angle instead")
        args.average_over_phi_angle = True

    at.set_mpl_style()

    args.modelpath = at.normalize_path_list(args.modelpath)

    modelpaths = args.modelpath

    args.color, args.label, args.linestyle, args.dashes, args.linewidth = trim_or_pad(
        len(args.modelpath), args.color, args.label, args.linestyle, args.dashes, args.linewidth
    )

    # the x axis is the time axis, so the shared limit helpers see the time range under the name they read
    args.xmin, args.xmax = args.timemin, args.timemax

    args.reflightcurves = makelist(args.reflightcurves)
    args.refspeccolors, args.refspecmarkers = trim_or_pad(
        len(args.reflightcurves), args.refspeccolors, args.refspecmarkers
    )

    # the reference data get black and greys, and the ARTIS models get the colours of the cycle
    isreference = [path_is_reference_lightcurve(path) for path in args.modelpath]
    seriescolors = get_series_colors(
        [*isreference, *([True] * len(args.reflightcurves))], [*args.color, *args.refspeccolors]
    )
    args.color = seriescolors[: len(args.modelpath)]
    args.refspeccolors = seriescolors[len(args.modelpath) :]

    defaultmarkers = ("o", "s", "h")
    args.refspecmarkers = [
        marker or defaultmarkers[i % len(defaultmarkers)] for i, marker in enumerate(args.refspecmarkers)
    ]

    if args.rpkt is False and not args.gamma:
        # if we're not plotting gamma, then we want to plot the r-packets by default
        args.rpkt = True
    if args.topnucs > 0:
        print("Enabling --frompackets because topnucs > 0")
        args.frompackets = True

    if args.filter:
        defaultoutputfile = "plotlightcurves.pdf"
    elif args.colour_evolution:
        defaultoutputfile = "plot_colour_evolution.pdf"
    else:
        defaultoutputfile = "plotlightcurve.pdf"

    args.outputfile = at.resolve_outputfile(args.outputfile, defaultoutputfile)
    outputfolder = args.outputfile.parent

    # determine if this will be a scatter plot or not
    if (  # args.calculate_peakmag_risetime_delta_m15 or
        args.save_viewing_angle_peakmag_risetime_delta_m15_to_file
        or args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
        or args.make_viewing_angle_peakmag_risetime_scatter_plot
        or args.make_viewing_angle_peakmag_delta_m15_scatter_plot
    ):
        at.lightcurve.peakmag_risetime_declinerate_init(modelpaths, args)
        return

    if args.colouratpeak:  # make scatter plot of colour at peak, eg. B-V at Bmax
        at.lightcurve.make_peak_colour_viewing_angle_plot(args)
        return

    if args.brightnessattime:
        if args.timedays is None:
            print("Specify timedays")
            sys.exit(1)
        if not args.plotviewingangle:
            args.plotviewingangle = [-1]
        if not args.colorbarcostheta and not args.colorbarphi:
            args.colorbarphi = True
        at.lightcurve.plot_viewanglebrightness_at_fixed_time(Path(modelpaths[0]), args)
        return

    if args.filter:
        make_band_lightcurves_plot(modelpaths, outputfolder, args)

    elif args.colour_evolution:
        colour_evolution_plot(modelpaths, outputfolder, args)
    else:
        make_lightcurve_plot(
            modelpaths=args.modelpath,
            filenameout=args.outputfile,
            frompackets=args.frompackets,
            showuvoir=args.rpkt,
            showgamma=args.gamma,
            maxpacketfiles=args.maxpacketfiles,
            args=args,
        )


if __name__ == "__main__":
    main()
