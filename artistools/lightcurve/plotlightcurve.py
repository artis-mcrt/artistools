"""Plot ARTIS bolometric and band light curves, colour evolution, and deposition curves."""

import argparse
import math
import typing as t
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import matplotlib.axes as mplax
import matplotlib.figure as mplfig
import matplotlib.lines as mpllines
import matplotlib.markers as mplmarkers
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from polars import selectors as cs

from artistools import misc
from artistools.commands import get_path
from artistools.constants import C_cm_per_s
from artistools.constants import day_to_s
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import Msun_to_g
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.lightcurve.core import FILTERNAME_ALIASES
from artistools.lightcurve.core import find_lightcurve_file
from artistools.lightcurve.core import generate_band_lightcurve_data
from artistools.lightcurve.core import get_band_lightcurve
from artistools.lightcurve.core import get_colour_delta_mag
from artistools.lightcurve.core import get_filename_part
from artistools.lightcurve.core import get_filter_data
from artistools.lightcurve.core import get_from_packets
from artistools.lightcurve.core import get_top_nuclides
from artistools.lightcurve.core import lum_lsun_to_mag
from artistools.lightcurve.core import path_is_reference_lightcurve
from artistools.lightcurve.core import read_bol_reflightcurve_data
from artistools.lightcurve.core import read_hesma_lightcurve
from artistools.lightcurve.core import read_reflightcurve_band_data
from artistools.lightcurve.core import scan_lightcurve
from artistools.lightcurve.viewingangleanalysis import make_peak_colour_viewing_angle_plot
from artistools.lightcurve.viewingangleanalysis import parse_directionbin_args
from artistools.lightcurve.viewingangleanalysis import peakmag_risetime_declinerate_init
from artistools.lightcurve.viewingangleanalysis import plot_viewanglebrightness_at_fixed_time
from artistools.misc import addarg_axislimits
from artistools.misc import addarg_dpi
from artistools.misc import addarg_figscale
from artistools.misc import addarg_filter
from artistools.misc import addarg_labelfontsize
from artistools.misc import addarg_legend
from artistools.misc import addarg_maxpacketfiles
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_notitle
from artistools.misc import addarg_output
from artistools.misc import addarg_residuals
from artistools.misc import addarg_seriesstyle
from artistools.misc import addarg_show
from artistools.misc import addarg_timedays
from artistools.misc import addarg_timestep
from artistools.misc import addarg_unsupported
from artistools.misc import addarg_verbose
from artistools.misc import addarg_viewingangle
from artistools.misc import addarg_yscale
from artistools.misc import apply_time_range_args
from artistools.misc import color_arg
from artistools.misc import df_filter_minmax_bracketed
from artistools.misc import exit_with_error
from artistools.misc import firstexisting
from artistools.misc import get_deposition
from artistools.misc import get_dirbins
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_filterfunc
from artistools.misc import get_model_folder
from artistools.misc import get_model_logname
from artistools.misc import get_model_name
from artistools.misc import get_series_label
from artistools.misc import makelist
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import path_is_dir
from artistools.misc import path_is_file
from artistools.misc import print_detail
from artistools.misc import print_heading
from artistools.misc import print_product
from artistools.misc import print_saved
from artistools.misc import print_theta_phi_definitions
from artistools.misc import print_warning
from artistools.misc import resolve_outputfile
from artistools.misc import resolve_series_styles
from artistools.misc import trim_or_pad
from artistools.misc.cliutils import positive_int_arg
from artistools.misc.remote import on_model_host
from artistools.plottools import AxesTree
from artistools.plottools import draw_residual_panel
from artistools.plottools import get_next_color
from artistools.plottools import get_unused_colors
from artistools.plottools import get_viewinganglecolor_for_colorbar
from artistools.plottools import invert_magnitude_yaxis
from artistools.plottools import iter_axes
from artistools.plottools import label_dirbin_series
from artistools.plottools import make_colorbar_viewingangles
from artistools.plottools import make_colorbar_viewingangles_colormap
from artistools.plottools import make_frame_figure
from artistools.plottools import make_frame_figure_with_residuals
from artistools.plottools import print_dirbin_summary
from artistools.plottools import RESIDUALROWHEIGHT
from artistools.plottools import ResidualSeries
from artistools.plottools import save_figure
from artistools.plottools import set_auto_yscale
from artistools.plottools import set_axis_labels
from artistools.plottools import set_axis_properties
from artistools.plottools import set_exponent_label
from artistools.plottools import set_legend
from artistools.plottools import set_plot_title
from artistools.plottools import set_prop_cycle_unusedcolors
from artistools.plottools import write_residual_stats

if t.TYPE_CHECKING:
    import matplotlib.typing as mplt

type LumUnit = t.Literal["mag", "Lsun", "erg/s"]

# the particles of the energy rates in deposition.out, in the order of the legend and of the controls of the viewer
ENERGYPARTICLES: t.Final = ("gamma", "betaminus", "betaplus", "alpha", "fission")

# -deposition also takes the sum of the deposition rates of all the particles
DEPOSITIONCHOICES: t.Final = (*ENERGYPARTICLES, "total")

# the symbol of each particle in the legend
PARTICLESYMBOLS: t.Final = MappingProxyType({
    "gamma": r"\gamma",
    "betaminus": r"\beta^-",
    "betaplus": r"\beta^+",
    "alpha": r"\alpha",
    "fission": r"\mathrm{fis}",
    "total": r"\mathrm{tot}",
})

# the column of deposition.out that holds the deposition rate of each particle. ARTIS deposits the energy of a
# fission where the pellet decays, thus fission has no rate from the packet trajectories
DEPOSITIONCOLUMNS: t.Final = MappingProxyType({
    "gamma": "gammadep_Lsun",
    "betaminus": "elecdep_Lsun",
    "betaplus": "positrondep_Lsun",
    "alpha": "alphadep_Lsun",
    "fission": "spfission_dep_discrete_Lsun",
    "total": "total_dep_Lsun",
})

# the column of the Monte Carlo emission rate, which ARTIS counts at the decays of the pellets. deposition.out holds no
# such rate of the positrons. The emission rate of fission is its deposition rate, because the energy deposits at the
# decay
EMISSIONCOLUMNS: t.Final = MappingProxyType({
    "gamma": "eps_gamma_Lsun",
    "betaminus": "eps_elec_Lsun",
    "alpha": "eps_alpha_Lsun",
    "fission": "spfission_dep_discrete_Lsun",
})

# the column of the analytical emission rate, which ARTIS calculates from the decay rates of the nuclides
ANALYTICEMISSIONCOLUMNS: t.Final = MappingProxyType({
    "betaminus": "eps_elec_ana_Lsun",
    "betaplus": "eps_positron_ana_Lsun",
    "alpha": "eps_alpha_ana_Lsun",
    "fission": "eps_spfission_ana_Lsun",
})

# the argument, the columns, the line style, and the legend label of each energy rate on the light curve axis
ENERGYRATEKINDS: t.Final = (
    ("deposition", DEPOSITIONCOLUMNS, "dashed", r"$\dot{{E}}_{{dep,{}}}$"),
    ("emission", EMISSIONCOLUMNS, "dotted", r"$\dot{{E}}_{{rad,{}}}$"),
    ("analyticemission", ANALYTICEMISSIONCOLUMNS, "dashdot", r"$\dot{{E}}_{{rad,{}}}$ analytical"),
)

# the arguments that name the particles of the energy rates
ENERGYRATEDESTS: t.Final = ("deposition", "emission", "analyticemission", "thermalisation")

# the height of the thermalisation panel as a part of the height of the light curve frame
THERMALISATIONROWHEIGHT: t.Final = 0.6

# the arguments that fit the peak magnitudes of the light curves of each direction bin
PEAKMAGDESTS: t.Final = (
    "save_angle_averaged_peakmag_risetime_delta_m15_to_file",
    "save_viewing_angle_peakmag_risetime_delta_m15_to_file",
    "make_viewing_angle_peakmag_risetime_scatter_plot",
    "make_viewing_angle_peakmag_delta_m15_scatter_plot",
)


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


def resolve_energy_rate_args(args: argparse.Namespace) -> None:
    """Give the older flags of the energy rates the particles that they drew, and sort each list of particles.

    --plotdeposition drew the deposition rates of the gamma rays and the electrons, and the emission rate of the
    electrons. --plotalphadeposition added the three rates of the alpha particles. --plotthermalisation added the
    thermalisation ratios of the gamma rays, the electrons, and the alpha particles, with the curves of Barnes et al.
    """
    named: dict[str, set[str]] = {dest: set(getattr(args, dest, None) or ()) for dest in ENERGYRATEDESTS}
    if args.plotdeposition or args.plotalphadeposition or args.plotthermalisation:
        named["deposition"] |= {"gamma", "betaminus"}
        named["emission"].add("betaminus")
    if args.plotalphadeposition:
        for dest in ("deposition", "emission", "analyticemission"):
            named[dest].add("alpha")
    if args.plotthermalisation:
        named["thermalisation"] |= {"gamma", "betaminus", "alpha"}
        args.showbarnes = True
    for dest, particles in named.items():
        setattr(args, dest, [particle for particle in DEPOSITIONCHOICES if particle in particles])
    args.plotdeposition = args.plotalphadeposition = args.plotthermalisation = False


def shows_energy_rates(args: argparse.Namespace) -> bool:
    """Return whether the arguments ask for an energy rate of deposition.out on the light curve axis.

    The y axis label depends on the curves that a model drew, and not on the value that this function returns.
    """
    return bool(args.deposition or args.emission or args.analyticemission)


def get_thermalisation_emission_column(particle: str) -> str:
    """Return the column of the emission rate that the thermalisation ratio of the particle divides by.

    The Monte Carlo rate comes first. deposition.out holds no such rate of the positrons, thus they take the
    analytical rate.
    """
    return EMISSIONCOLUMNS.get(particle) or ANALYTICEMISSIONCOLUMNS[particle]


def convert_lum_lsun_to_plotunits(lum_lsun: npt.NDArray[np.floating], lumunit: LumUnit) -> npt.NDArray[np.floating]:
    """Convert a luminosity in solar luminosities to the y axis units: bolometric magnitude, Lsun, or erg/s."""
    if lumunit == "mag":
        return lum_lsun_to_mag(lum_lsun)

    return lum_lsun if lumunit == "Lsun" else lum_lsun * Lsun_to_erg_per_s


def convert_lum_ergs_to_plotunits(
    lum_erg_per_s: npt.NDArray[np.floating], lumunit: LumUnit
) -> npt.NDArray[np.floating]:
    """Convert a luminosity in erg/s to the y axis units: bolometric magnitude, Lsun, or erg/s."""
    if lumunit == "erg/s":
        return lum_erg_per_s

    return convert_lum_lsun_to_plotunits(lum_erg_per_s / Lsun_to_erg_per_s, lumunit)


def get_reflightcurve_yerr(
    lum_erg_per_s: npt.NDArray[np.floating],
    errminus_erg_per_s: npt.NDArray[np.floating],
    errplus_erg_per_s: npt.NDArray[np.floating],
    lumunit: LumUnit,
) -> tuple[list[npt.NDArray[np.floating]], npt.NDArray[np.bool_]]:
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
    axis: mplax.Axes,
    lightcurvefilename: str | Path,
    lumunit: LumUnit,
    color: str,
    label: str | None = None,
    residualseries: list[ResidualSeries] | None = None,
    linewidth: float | None = None,
    alpha: float | None = None,
) -> str:
    """Plot an observed bolometric light curve in the y axis units, with error bars if the data file has them.

    Return the label used in the plot legend, which comes from the file metadata unless label is given.
    """
    dflightcurve, metadata = read_bol_reflightcurve_data(lightcurvefilename)
    # an empty label is a series deliberately left out of the legend, so only a missing one takes the metadata
    plotlabel = str(metadata.get("label", lightcurvefilename)) if label is None else label
    lum_erg_per_s = dflightcurve["luminosity_erg/s"].to_numpy()
    time_days = dflightcurve["time_days"].to_numpy()
    yvalues = convert_lum_ergs_to_plotunits(lum_erg_per_s, lumunit)

    residual_yerr = None
    errorbar_kwargs = None
    unbounded = None
    if {"luminosity_errminus_erg/s", "luminosity_errplus_erg/s"}.issubset(dflightcurve.columns):
        yerr, unbounded = get_reflightcurve_yerr(
            lum_erg_per_s,
            dflightcurve["luminosity_errminus_erg/s"].to_numpy(),
            dflightcurve["luminosity_errplus_erg/s"].to_numpy(),
            lumunit,
        )
        # the error bars are the only lines of this series, thus -linewidth sets their width
        errorbars = axis.errorbar(
            time_days,
            yvalues,
            yerr=yerr,
            fmt="o",
            capsize=3,
            label=plotlabel,
            color=color,
            elinewidth=linewidth,
            alpha=alpha,
            capthick=linewidth,
        )
        refline = errorbars.lines[0]
        residual_yerr = np.asarray(yerr, dtype=np.float64)
        errorbar_kwargs = {"color": color, "capsize": 3, "elinewidth": linewidth, "capthick": linewidth, "alpha": alpha}
        refartists = errorbars.get_children()
        if unbounded.any():
            # matplotlib draws only one side of a bar it is told is a limit, so the open faint side is a
            # second, zero-length bar whose arrow head sits on the point the bar above already reaches
            limitbars = axis.errorbar(
                time_days[unbounded],
                yvalues[unbounded],
                yerr=np.zeros(int(unbounded.sum())),
                lolims=True,
                fmt="none",
                color=color,
                alpha=alpha,
            )
            # matplotlib picks the direction of the arrow from the orientation of the axis as it is now, and
            # a magnitude axis is inverted only after every series is drawn, so point it at the faint side.
            # The cast is for the marker constant, which matplotlib types as an int that its own setter rejects
            caretdown = t.cast("t.Literal[11]", mplmarkers.CARETDOWNBASE)
            for capline in limitbars.lines[1]:
                capline.set_marker(caretdown)
            refartists += limitbars.get_children()
    else:
        (refline,) = axis.plot(
            time_days, yvalues, marker="o", linestyle="None", label=plotlabel, color=color, alpha=alpha
        )
        refartists = [refline]

    # a marker and a line have different default zorders. With an equal zorder, matplotlib draws
    # the series in the order of the command line
    for artist in refartists:
        artist.set_zorder(mpllines.Line2D.zorder)

    if residualseries is not None:
        residualseries.append(
            ResidualSeries(
                plotlabel,
                np.asarray(time_days, dtype=np.float64),
                np.asarray(yvalues, dtype=np.float64),
                color,
                line=refline,
                yerr=residual_yerr,
                errorbar_kwargs=errorbar_kwargs,
                unbounded=unbounded,
            )
        )

    return plotlabel


@on_model_host
def get_model_mass_and_kinetic_energy(modelpath: Path) -> tuple[float, float]:
    """Return the mass [g] and the kinetic energy [erg] of the ejecta. The host of a remote model reads the model."""
    dfmodel, modelmeta = get_modeldata(modelpath)
    dfmodel = add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta)

    # one collect for both sums: get_modeldata returns a plan, so a second one reads the model again
    model_mass_grams, ejecta_ke_erg = dfmodel.select(pl.sum("mass_g"), pl.sum("kinetic_en_erg")).collect().row(0)
    return float(model_mass_grams), float(ejecta_ke_erg)


def plot_energy_rates(
    axis: mplax.Axes,
    thermaxis: mplax.Axes | None,
    modelpath: str | Path,
    modelname: str,
    args: argparse.Namespace,
    linewidth: float | str | None = None,
    *,
    residualseries: list[ResidualSeries] | None = None,
) -> bool:
    """Plot the energy rates of deposition.out that args names, and return True if the function drew a curve on axis.

    axis takes the deposition rates and the emission rates. thermaxis takes the thermalisation ratios with the curves
    of Barnes, Kasen, Wu & Martínez-Pinedo (2016, ApJ, 829, 110, doi:10.3847/0004-637X/829/2/110). Each particle takes
    one colour, and each kind of rate takes one line style. Every curve appends its own suffix to modelname, thus the
    line width is the only style that the caller sets.
    """
    lumunit = get_plot_lum_unit(args)
    depdata = get_deposition(modelpath).collect()
    modellogname = get_model_logname(modelpath)

    get_next_color(axis)  # skip a colour, thus the energy rates have a different colour from the light curve
    # a colour for each particle that the plot names, thus a particle that no argument names takes no colour
    named = {*args.deposition, *args.emission, *args.analyticemission, *args.thermalisation}
    particlecolours = {particle: get_next_color(axis) for particle in DEPOSITIONCHOICES if particle in named}
    if "fission" in named:
        print_detail(
            "ARTIS deposits the energy of each fission where its pellet decays, thus the deposition rate and the"
            " emission rate of fission are one column of deposition.out"
        )

    drewrate = False
    for particle, colour in particlecolours.items():
        for dest, columns, linestyle, labelformat in ENERGYRATEKINDS:
            if particle not in getattr(args, dest):
                continue
            if (column := columns[particle]) not in depdata.columns:
                # an older deposition.out has only the gamma columns, and a run with no fission has no fission column
                print_warning(f"{modellogname} gives no {column} in deposition.out, thus the plot has no such curve")
                continue
            yvalues = convert_lum_lsun_to_plotunits(depdata[column].to_numpy(), lumunit)
            label = f"{modelname} {labelformat.format(PARTICLESYMBOLS[particle])}"
            (rateline,) = axis.plot(
                depdata["tmid_days"], yvalues, linewidth=linewidth, label=label, linestyle=linestyle, color=colour
            )
            if residualseries is not None:
                residualseries.append(
                    ResidualSeries(
                        label,
                        np.asarray(depdata["tmid_days"].to_numpy(), dtype=np.float64),
                        np.asarray(yvalues, dtype=np.float64),
                        rateline.get_color(),
                        line=rateline,
                    )
                )
            drewrate = True

    if args.thermalisation:
        assert thermaxis is not None
        plot_thermalisation(thermaxis, depdata, modelpath, modelname, args, particlecolours, linewidth)

    return drewrate


def plot_thermalisation(
    thermaxis: mplax.Axes,
    depdata: pl.DataFrame,
    modelpath: str | Path,
    modelname: str,
    args: argparse.Namespace,
    particlecolours: dict[str, str],
    linewidth: float | str | None,
) -> None:
    """Plot the deposition rate over the emission rate of each particle of -thermalisation.

    --showbarnes adds the curves of Barnes et al. (2016, ApJ, 829, 110) for the gamma rays, the electrons, and the alpha
    particles.
    """
    if "betaplus" in args.thermalisation:
        print_detail(
            "the thermalisation ratio of the positrons divides by the analytical emission rate, because"
            " deposition.out holds no Monte Carlo emission rate of the positrons"
        )
    for particle in args.thermalisation:
        depcolumn, emissioncolumn = DEPOSITIONCOLUMNS[particle], get_thermalisation_emission_column(particle)
        if missing := [
            column for column in dict.fromkeys((depcolumn, emissioncolumn)) if column not in depdata.columns
        ]:
            print_warning(
                f"{get_model_logname(modelpath)} gives no {' and no '.join(missing)} in deposition.out, thus the"
                f" plot has no thermalisation ratio of {particle}"
            )
            continue
        symbol = PARTICLESYMBOLS[particle]
        thermaxis.plot(
            depdata["tmid_days"],
            depdata[depcolumn] / depdata[emissioncolumn],
            linewidth=linewidth,
            label=rf"{modelname} $\left(\dot{{E}}_{{dep,{symbol}}} \middle/ \dot{{E}}_{{rad,{symbol}}}\right)$",
            linestyle="solid",
            color=particlecolours[particle],
        )

    # the curves of Barnes et al. (2016, ApJ, 829, 110) describe the gamma rays, the electrons, and the alpha particles
    barnesparticles = [particle for particle in ("gamma", "betaminus", "alpha") if particle in args.thermalisation]
    if not (args.showbarnes and barnesparticles):
        return

    model_mass_grams, ejecta_ke_erg = get_model_mass_and_kinetic_energy(Path(modelpath))
    print(f"  model mass: {model_mass_grams / Msun_to_g:.3f} Msun")
    print(f"  ejecta kinetic energy: {ejecta_ke_erg / 1e7:.2e} [J] = {ejecta_ke_erg:.2e} [erg]")

    # velocity derived from ejecta kinetic energy to match Barnes et al. (2016) Section 2.1
    ejecta_v = np.sqrt(2 * ejecta_ke_erg / model_mass_grams)
    print(f"  Barnes average ejecta velocity: {ejecta_v / C_cm_per_s:.2f}c")
    m5 = model_mass_grams / (5e-3 * Msun_to_g)  # M / (5e-3 Msun)
    v2 = ejecta_v / (0.2 * C_cm_per_s)  # ejecta_v / (0.2c)
    tmids = depdata["tmid_days"].to_list()

    def barnes_f_charged(t_ineff: float) -> list[float]:
        """Return the Barnes et al (2016) equation 32 thermalisation efficiency of a charged particle."""
        return [math.log(1 + 2 * (t / t_ineff) ** 2) / (2 * (t / t_ineff) ** 2) for t in tmids]

    # Barnes et al (2016) scaling form from equation 17, with fiducial t_ineff_gamma of 1.4 days
    t_ineff_gamma = 1.4 * np.sqrt(m5) / v2
    e0_beta_mev = 0.5
    # Barnes et al (2016) equation 20
    t_ineff_beta = 7.4 * (e0_beta_mev / 0.5) ** -0.5 * m5**0.5 * (v2 ** (-3.0 / 2))
    e0_alpha_mev = 6.0
    # Barnes et al (2016) equation 25 times equation 16 for t_peak
    t_ineff_alpha = 4.3 * 1.8 * (e0_alpha_mev / 6.0) ** -0.5 * m5**0.5 * (v2 ** (-3.0 / 2))

    barnes_curves = {
        # Barnes et al (2016) equation 33 for the gamma rays, equation 32 for the charged particles
        "gamma": ([1 - math.exp(-((t / t_ineff_gamma) ** -2)) for t in tmids], r"\gamma"),
        "betaminus": (barnes_f_charged(t_ineff_beta), r"\beta"),
        "alpha": (barnes_f_charged(t_ineff_alpha), r"\alpha"),
    }
    for particle in barnesparticles:
        barnes_f, symbol = barnes_curves[particle]
        thermaxis.plot(
            tmids,
            barnes_f,
            linewidth=linewidth,
            label=rf"Barnes+2016 $f_{symbol}$",
            linestyle="dashed",
            color=particlecolours[particle],
        )


def get_time_range_days(dflightcurve: pl.DataFrame) -> tuple[float | None, float | None]:
    """Return the first and the last time of a light curve in days, or None for a light curve with no rows."""
    tmin, tmax = dflightcurve.select(pl.col("time_days").min(), pl.col("time_days").max().alias("time_days_max")).row(0)
    return tmin, tmax


def plot_artis_lightcurve(
    modelpath: str | Path,
    axis: mplax.Axes,
    lcindex: int = 0,
    linelabel: str | None = None,
    escape_type: str = "TYPE_RPKT",
    frompackets: bool = False,
    maxpacketfiles: int | None = None,
    average_over_phi: bool = False,
    average_over_theta: bool = False,
    *,
    args: argparse.Namespace,
    pellet_nucname: str | None = None,
    use_pellet_decay_time: bool = False,
    residualseries: list[ResidualSeries] | None = None,
    **plotkwargs: t.Any,
) -> dict[int, pl.DataFrame] | None:
    """Plot one model's bolometric light curve, and return the plotted data per direction bin."""
    if escape_type not in {"TYPE_RPKT", "TYPE_GAMMA"}:
        msg = f"Unknown escape type {escape_type}"
        raise ValueError(msg)

    # handle e.g. modelpath = 'modelpath/light_curve.out'
    inputpath = Path(modelpath)
    lcfilename = inputpath.name if path_is_file(inputpath) else None
    modelpath = inputpath.parent if lcfilename else inputpath

    if not path_is_dir(modelpath):
        print_warning(f"Skipping because {modelpath} does not exist")
        return None

    linelabel_is_custom = linelabel is not None
    assert "label" not in plotkwargs, "label is already set in plotkwargs"
    # an empty label hides the series in the legend, thus only a label that the user did not give takes the name
    if linelabel is None:
        linelabel = get_model_name(modelpath)
    assert linelabel is not None
    if escape_type == "TYPE_GAMMA" and linelabel:
        linelabel += r" $\gamma$"
    if pellet_nucname is not None:
        # the model name and the gamma marker stay, because a second model or --gamma gives the same nuclide again
        linelabel = f"{linelabel} {pellet_nucname}" if linelabel else pellet_nucname

    print_heading(linelabel)
    print_detail(f"modelpath: {modelpath.resolve().parts[-1]}")

    if hasattr(args, "title") and args.title:
        set_plot_title(axis, args.title if isinstance(args.title, str) else linelabel, args)

    # resolve the direction bins first, because "-plotviewingangle -2" expands to every bin. The
    # packet-derived data must hold the same bin keys that the plot loop reads.
    dirbins, angle_definition = parse_directionbin_args(modelpath, args)
    # bin -1 is the angle average, which only light_curve.out holds
    directionresolved = list(dirbins) != [-1]

    if frompackets:
        lcdataframes = get_from_packets(
            modelpath,
            escape_type=escape_type,
            maxpacketfiles=maxpacketfiles,
            directionbins=dirbins,
            average_over_phi=average_over_phi,
            average_over_theta=average_over_theta,
            directionbins_are_vpkt_observers=args.plotvspecpol is not None,
            pellet_nucname=pellet_nucname,
            use_pellet_decay_time=use_pellet_decay_time,
            timedaysmin=args.timemin,
            timedaysmax=args.timemax,
        )
    else:
        if pellet_nucname is not None or use_pellet_decay_time:
            msg = "only the packets give the nuclide and the decay time of a pellet, thus give frompackets=True"
            raise ValueError(msg)
        if args.plotvspecpol is not None:
            exit_with_error(
                "-plotvspecpol names virtual packet observers, which light_curve_res.out does not hold",
                "Give --frompackets to make the light curve of each virtual observer from the packets.",
            )
        try:
            lcpath = (
                firstexisting(lcfilename, folder=modelpath, tryzipped=True)
                if lcfilename is not None
                else find_lightcurve_file(
                    modelpath, directionresolved=directionresolved, gamma=escape_type == "TYPE_GAMMA"
                )
            )
        except FileNotFoundError as exc:
            print_warning(f"Skipping {modelpath}: {exc}")
            return None

        try:
            lcdataframes = scan_lightcurve(
                lcpath,
                directionresolved=directionresolved,
                average_over_phi=average_over_phi,
                average_over_theta=average_over_theta,
            )
        except ValueError as exc:
            exit_with_error(
                str(exc),
                "Give -plotviewingangle with light_curve_res.out, give no -plotviewingangle with light_curve.out, and"
                " give a file that the run wrote in full",
            )
        # light_curve_res.out holds the bins 0 to 99, thus the angle average of bin -1 comes from light_curve.out
        if -1 in dirbins and -1 not in lcdataframes:
            lcdataframes[-1] = scan_lightcurve(
                find_lightcurve_file(modelpath, directionresolved=False, gamma=escape_type == "TYPE_GAMMA")
            )[-1]

    lumunit = get_plot_lum_unit(args)
    ycolumn = get_plot_lum_column(lumunit)

    if args.dashes[lcindex]:
        plotkwargs["dashes"] = args.dashes[lcindex]
    if args.linewidth[lcindex]:
        plotkwargs["linewidth"] = args.linewidth[lcindex]
    if args.linealpha[lcindex] is not None:
        plotkwargs["alpha"] = args.linealpha[lcindex]

    if args.colorbarcostheta or args.colorbarphi:
        scaledmap = make_colorbar_viewingangles_colormap()

    lcdataframes = dict(
        zip(
            dirbins,
            pl.collect_all(
                df_filter_minmax_bracketed(
                    lcdataframes[dirbin], colname="time_days", minval=args.timemin, maxval=args.timemax
                )
                for dirbin in dirbins
            ),
            strict=True,
        )
    )
    lctimemin, lctimemax = get_time_range_days(lcdataframes[dirbins[0]])
    assert isinstance(lctimemin, float)
    assert isinstance(lctimemax, float)

    print_detail(f"range of the light curve: {lctimemin:.2f} to {lctimemax:.2f} days")
    try:
        nts_last, validrange_start_days, validrange_end_days = get_escaped_arrivalrange(modelpath)
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
        print_detail(f"range of validity (last timestep {nts_last}): {str_valid_range}")

    if any(dirbin != -1 for dirbin in dirbins):
        print_theta_phi_definitions()

    filterfunc = get_filterfunc(args)
    for dirbin in dirbins:
        lcdata = lcdataframes[dirbin]
        print_dirbin_summary(dirbin, angle_definition[dirbin], lcdata)

        if filterfunc is not None:
            lcdata = lcdata.with_columns(
                (cs.starts_with("luminosity_") | cs.by_name("mag")).map_batches(
                    filterfunc, return_dtype=pl.self_dtype()
                )
            )

        label_with_tags: str | None = linelabel
        if dirbin != -1:
            if args.colorbarcostheta or args.colorbarphi:
                # the bins of a colour bar overlap, thus they are partly transparent unless -linealpha gives a value
                plotkwargs["alpha"] = 0.75 if args.linealpha[lcindex] is None else args.linealpha[lcindex]
                # the colour bar names the direction bin, thus the legend needs no entry for it. The
                # user gives -label to name the model, thus only the first bin keeps that label
                label_with_tags = linelabel if linelabel_is_custom and dirbin == dirbins[0] else None
                # Update plotkwargs with viewing angle colour
                plotkwargs, _ = get_viewinganglecolor_for_colorbar(dirbin, scaledmap, plotkwargs, args)
                if args.average_over_phi_angle:
                    # a curve that averages over phi holds one whole cos(theta) bin, thus draw it above the rest
                    plotkwargs["zorder"] = 10
            else:
                label_with_tags = label_dirbin_series(
                    dirbin, dirbins, angle_definition, linelabel, linelabel_is_custom, plotkwargs
                )

        if pellet_nucname is not None:
            plotkwargs["color"] = None

        lcdata_tmin, lcdata_tmax = get_time_range_days(lcdata)
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
        print_detail(f"Katz integral L t dt ({lcdata_tmin:.2f} to {lcdata_tmax:.2f} days): {katz_integral:.3e} [erg s]")
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
            print_warning("No data points in valid range")

        energy_released = abs(
            np.trapezoid(
                np.nan_to_num(lcdata_valid["luminosity_erg/s"], nan=0.0), x=lcdata_valid["time_days"] * day_to_s
            )
        )

        lcdata_valid_tmin, lcdata_valid_tmax = get_time_range_days(lcdata_valid)
        if lcdata_valid_tmin is not None and lcdata_valid_tmax is not None:
            print_detail(
                f"integrated luminosity ({lcdata_valid_tmin:.2f} to {lcdata_valid_tmax:.2f} days):"
                f" {energy_released:.3e} [erg]"
            )

        (modelline,) = axis.plot(lcdata_valid["time_days"], lcdata_valid[ycolumn], label=label_with_tags, **plotkwargs)
        if residualseries is not None:
            residualseries.append(
                ResidualSeries(
                    label_with_tags or f"direction bin {dirbin}",
                    np.asarray(lcdata_valid["time_days"].to_numpy(), dtype=np.float64),
                    np.asarray(lcdata_valid[ycolumn].to_numpy(), dtype=np.float64),
                    modelline.get_color(),
                    line=modelline,
                )
            )
        if args.print_data:
            print_product(args, lcdata)

        cmfcolumn = ycolumn.replace("luminosity_", "luminosity_cmf_")
        if args.plotcmf and cmfcolumn not in lcdata.columns:
            print_warning(
                f"the light curve of {linelabel} has no comoving frame luminosity, thus --plotcmf draws no curve"
            )
        elif args.plotcmf:
            # a copy, because the next direction bin keeps the rest-frame style and the -linewidth value
            plotkwargs_cmf: dict[str, t.Any] = plotkwargs | {"linewidth": 1, "linestyle": "dashed"}
            # a colour bar leaves the series with no label, thus there is no label to mark as comoving frame
            label_cmf = (
                f"{label_with_tags} (cmf)"
                if label_with_tags is not None and not linelabel_is_custom
                else label_with_tags
            )
            (cmfline,) = axis.plot(lcdata["time_days"], lcdata[cmfcolumn], label=label_cmf, **plotkwargs_cmf)
            if residualseries is not None:
                residualseries.append(
                    ResidualSeries(
                        label_cmf or f"direction bin {dirbin} (cmf)",
                        np.asarray(lcdata["time_days"].to_numpy(), dtype=np.float64),
                        np.asarray(lcdata[cmfcolumn].to_numpy(), dtype=np.float64),
                        cmfline.get_color(),
                        line=cmfline,
                    )
                )

    return lcdataframes


def make_plot_figure(
    args: argparse.Namespace, *, fig: mplfig.Figure | None = None
) -> tuple[mplfig.Figure, mplax.Axes, mplax.Axes | None, mplax.Axes | None]:
    """Return the figure, the axis of the light curves, the thermalisation panel, and the residual panel.

    The residual panel comes with -residual, and the thermalisation panel at the bottom comes with -thermalisation.
    If the caller gives an empty figure as fig, the function adds the frames to it, e.g. the figure of the viewer.
    """
    rowheights = [
        1.0,
        *([RESIDUALROWHEIGHT] if args.residualbaselineseries is not None else []),
        *([THERMALISATIONROWHEIGHT] if args.thermalisation else []),
    ]
    # each frame holds a size in inches, thus a grid of panels in a paper takes one room for each
    fig, axesgrid = make_frame_figure(args, rows=len(rowheights), sharex=True, rowheights=rowheights, fig=fig)
    axis, *panels = axesgrid[:, 0]
    residualaxis = panels.pop(0) if args.residualbaselineseries is not None else None
    thermaxis = panels.pop(0) if args.thermalisation else None
    return fig, axis, thermaxis, residualaxis


def draw_plot(
    args: argparse.Namespace, axis: mplax.Axes, thermaxis: mplax.Axes | None, residualaxis: mplax.Axes | None
) -> pl.DataFrame | None:
    """Draw the light curves and the energy rates that args selects, and return the statistics of the residuals.

    The light curves come from light_curve.out, gamma_light_curve.out, light_curve_res.out, or the packets files. The
    axes must be empty. This function writes no file, thus the viewer can call it again for each change of a control.
    The statistics are None without a residual panel.
    """
    modelpaths = args.modelpath
    showuvoir, showgamma = args.rpkt, args.gamma
    lumunit = get_plot_lum_unit(args)
    residualseries: list[ResidualSeries] | None = [] if residualaxis is not None else None
    axis.margins(x=0.0)
    if thermaxis is not None:
        # the panel shares the time axis, and its default margin widened the time range of both frames
        thermaxis.margins(x=0.0)
        thermaxis.set_ylabel("Thermalisation ratio")

    set_prop_cycle_unusedcolors([axis], [*args.color, *args.refspeccolors])

    plottedsomething = False
    plotteddeposition = False
    for lcindex, modelpath in enumerate(modelpaths):
        if path_is_reference_lightcurve(modelpath):
            bolreflightcurve = Path(modelpath)

            lightcurvelabel = plot_bol_reflightcurve(
                axis,
                bolreflightcurve,
                lumunit,
                color=args.color[lcindex],
                label=args.label[lcindex],
                residualseries=residualseries,
                linewidth=args.linewidth[lcindex] or None,
                alpha=args.linealpha[lcindex],
            )
            print_heading(lightcurvelabel)
            plottedsomething = True

        else:
            plottedthismodel = False
            escape_types: list[str] = ["TYPE_RPKT"] if showuvoir else []
            if showgamma:
                escape_types.append("TYPE_GAMMA")

            topnucs = args.topnucs
            for escape_type in escape_types:
                pellet_nucnames: list[str | None] = [None]
                if topnucs > 0:
                    try:
                        top_nuclides = get_top_nuclides(
                            Path(modelpath),
                            escape_type,
                            topnucs,
                            args.maxpacketfiles,
                            args.timemin,
                            args.timemax,
                            use_pellet_decay_time=args.use_pellet_decay_time,
                        )
                        print(f"Top nuclides by energy release: {top_nuclides['nucname'].to_list()}")
                        pellet_nucnames.extend(top_nuclides["nucname"])
                    except FileNotFoundError:
                        print_warning("no nuclides.out file found, skipping top nuclides")

                for pellet_nucname in pellet_nucnames:
                    lcdataframes = plot_artis_lightcurve(
                        modelpath=modelpath,
                        lcindex=lcindex,
                        axis=axis,
                        escape_type=escape_type,
                        frompackets=args.frompackets,
                        maxpacketfiles=args.maxpacketfiles,
                        average_over_phi=args.average_over_phi_angle,
                        average_over_theta=args.average_over_theta_angle,
                        args=args,
                        pellet_nucname=pellet_nucname,
                        use_pellet_decay_time=args.use_pellet_decay_time,
                        linestyle=args.linestyle[lcindex]
                        if (escape_type == "TYPE_RPKT" or len(escape_types) == 1)
                        else ":",
                        color=args.color[lcindex],
                        linelabel=args.label[lcindex],
                        residualseries=residualseries,
                    )
                    plottedthismodel = plottedthismodel or (lcdataframes is not None)

            plottedsomething = plottedsomething or plottedthismodel

            if plottedthismodel and (shows_energy_rates(args) or args.thermalisation):
                # the rates belong to the model, not to one escape type or pellet nuclide, and the style
                # comes from the command line rather than from whatever a series left in its plot kwargs
                drewrate = plot_energy_rates(
                    axis,
                    thermaxis,
                    get_model_folder(modelpath),
                    modelname=get_series_label(args.label, lcindex, get_model_name(modelpath)),
                    args=args,
                    linewidth=args.linewidth[lcindex] or None,
                    residualseries=residualseries,
                )
                plotteddeposition = plotteddeposition or drewrate

        print()

    if args.reflightcurves:
        for refindex, bolreflightcurve in enumerate(args.reflightcurves):
            plot_bol_reflightcurve(
                axis,
                bolreflightcurve,
                lumunit,
                color=args.refspeccolors[refindex],
                residualseries=residualseries,
                linewidth=args.linewidth[len(modelpaths) + refindex] or None,
                alpha=args.linealpha[len(modelpaths) + refindex],
            )
            plottedsomething = True

    if not plottedsomething:
        exit_with_error(
            "the plot holds no light curve, because the command skipped each path",
            "The warnings above give the reason for each path",
        )

    set_legend(axis, args, loc="best", handlelength=2, frameon=False, numpoints=1)
    if thermaxis is not None:
        # -ymin and -ymax give the range of the light curves, thus they do not fix the room of the panel legend
        thermargs = argparse.Namespace(**{**vars(args), "ymin": None, "ymax": None})
        set_legend(thermaxis, thermargs, loc="upper right", handlelength=2, frameon=False, numpoints=1)

    # a magnitude is a logarithm already, and its axis runs backwards, thus only a luminosity can
    # take a log scale. This follows the plot, because the drawn values give the answer
    if lumunit != "mag":
        set_auto_yscale(axis, args)

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

        if not args.logscaley:
            set_exponent_label(axis)

    if args.colorbarcostheta or args.colorbarphi:
        scaledmap = make_colorbar_viewingangles_colormap()
        make_colorbar_viewingangles(scaledmap, args, ax=axis)

    # set the limits only now that the data is drawn: on an empty axes matplotlib turns autoscaling off, so a
    # one-sided limit would freeze the other side at the default 0-1 view instead of fitting the light curves
    set_axis_properties(axis, args, xlimits=(args.timemin, args.timemax, "-timemin"))
    if lumunit == "mag":
        # invert last: set_ylim re-sorts the limits into the order of the pair it is given, so an inversion
        # applied before a one-sided limit is lost
        invert_magnitude_yaxis(axis)

    dfresidualstats = None
    if residualaxis is not None and residualseries is not None:
        dfresidualstats = draw_residual_panel(residualaxis, axis, residualseries, args, ismagnitude=lumunit == "mag")

    if thermaxis is not None:
        set_axis_properties(thermaxis, args, setyaxis=False)
        # a thermalisation efficiency is a ratio, so keep the physical range rather than letting a
        # near-zero denominator at one timestep rescale every curve into the bottom of the panel
        thermaxis.set_ylim(0.0, 1.0)
        # the panel is the lowest frame, thus it takes the label of the shared time axis
        labelaxis = residualaxis or axis
        thermaxis.set_xlabel(labelaxis.get_xlabel())
        labelaxis.set_xlabel("")

    return dfresidualstats


def create_axes(args: argparse.Namespace) -> tuple[mplfig.Figure, npt.NDArray[np.object_] | mplax.Axes]:
    """Return the figure and axes, using a subplot grid when several filters or colours are plotted."""
    panelcount = len(args.filter) if args.filter else len(args.colour_evolution or [])
    args.subplots = panelcount > 1
    if args.subplots:
        # the grid holds one panel for each band or for each pair of bands. A fixed grid of six panels
        # gave an IndexError for a seventh band, and left an empty panel for fewer than six
        cols = min(3, panelcount)
        rows = math.ceil(panelcount / cols)
    else:
        rows = 1
        cols = 1

    fig, axesgrid = make_frame_figure(args, rows=rows, cols=cols, sharey=True)
    ax = axesgrid.flatten() if args.subplots else axesgrid[0][0]

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


def set_lightcurveplot_legend(ax: AxesTree, args: argparse.Namespace) -> None:
    """Add the legend, and place it on args.legendsubplotnumber when the figure has subplots."""
    if args.nolegend:
        # set_legend would draw nothing, and the subplot index below must not run for it
        return

    if args.subplots:
        axis = iter_axes(ax)[args.legendsubplotnumber]
        set_legend(axis, args, loc=args.legendposition, frameon=False)
    else:
        assert isinstance(ax, mplax.Axes)
        set_legend(ax, args, loc=args.legendposition, frameon=False, handlelength=0.7)


def set_lightcurve_plot_labels(
    fig: mplfig.Figure,
    ax: AxesTree,
    args: argparse.Namespace,
    band_name: str | None = None,
    colour_evolution: bool = False,
) -> tuple[mplfig.Figure, AxesTree]:
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


def make_band_lightcurves_plot(
    modelpaths: Sequence[str | Path], outputfolder: Path | str, args: argparse.Namespace
) -> None:
    """Plot band magnitude light curves for every model and save the figure."""
    residualaxis = None
    residualseries: list[ResidualSeries] | None = None
    if args.residualbaselineseries is not None:
        args.subplots = False
        fig, ax, residualaxis = make_frame_figure_with_residuals(args)
        residualseries = []
    else:
        fig, ax = create_axes(args)
    axes = iter_axes(ax)

    # a model with several direction bins takes its line colours from the cycle, so keep the cycle clear of
    # the colours that other series were given
    set_prop_cycle_unusedcolors(axes, [*args.color, *args.refspeccolors])

    plotkwargs: dict[str, t.Any] = {}

    if args.colorbarcostheta or args.colorbarphi:
        scaledmap = make_colorbar_viewingangles_colormap()

    # every model is asked for the same bands, so one list serves the loader, the reference curves, the
    # y axis label and the output file name. main() dispatches here only when -filter has a value
    bandnames: list[str] = list(args.filter)
    filterfunc = get_filterfunc(args)
    # each model writes its own data files, thus the name of a file holds the model name if the plot holds several
    # models. The file name of one model stays the same
    namesmodel = len(modelpaths) > 1
    for modelnumber, modelpath in enumerate(Path(m) for m in modelpaths):
        # check if doing viewing angle stuff, and if so define which data to use
        dirbins, dirbin_definition = parse_directionbin_args(modelpath, args)

        for dirbin in dirbins:
            modelname = get_model_name(modelpath)
            linelabel_is_custom = get_series_label(args.label, modelnumber, modelname) != modelname
            if args.verbose:
                print(f"Reading spectra: {get_model_logname(modelpath)} (angle {dirbin})")
            band_lightcurve_data = generate_band_lightcurve_data(modelpath, args, dirbin, filternames=bandnames)

            for plotnumber, band_name in enumerate(band_lightcurve_data):
                axis = axes[plotnumber]
                time, brightness_in_mag = get_band_lightcurve(band_lightcurve_data, band_name, args)

                if args.print_data or args.write_data:
                    txtlinesout = [f"# band: {band_name}", f"# model: {modelname}", "# time_days magnitude"]
                    txtlinesout.extend(f"{t_d} {m}" for t_d, m in zip(time, brightness_in_mag, strict=False))
                    txtout = "\n".join(txtlinesout)
                if args.write_data:
                    filestem = f"band_{get_filename_part(band_name)}"
                    if namesmodel:
                        filestem += f"_{get_filename_part(modelname)}"
                    bandoutfile = Path(
                        outputfolder, f"{filestem}_angle_{dirbin}.txt" if dirbin != -1 else f"{filestem}.txt"
                    )
                    bandoutfile.write_text(txtout, encoding="utf-8")
                    print_saved(bandoutfile)
                if args.print_data:
                    print_product(args, txtout)

                plotkwargs["label"] = get_linelabel(modelname, modelnumber, dirbin, dirbin_definition, args)

                if filterfunc is not None:
                    brightness_in_mag = filterfunc(brightness_in_mag)

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

                # only the first direction bin can take the model's single -color entry, as on the bolometric
                # figure; the rest take a colour each from the cycle, which set_prop_cycle_unusedcolors has
                # kept clear of the assigned colours. plotkwargs is reused, so always assign it
                plotkwargs["color"] = args.color[modelnumber] if dirbin == dirbins[0] else None

                if dirbin != -1 and (args.colorbarcostheta or args.colorbarphi):
                    # the colour bar names the direction bin, thus the legend needs no entry for it. The
                    # user gives -label to name the model, thus only the first bin keeps that label
                    plotkwargs["label"] = (
                        get_series_label(args.label, modelnumber, modelname)
                        if linelabel_is_custom and dirbin == dirbins[0]
                        else None
                    )
                    # Update plotkwargs with viewing angle colour
                    plotkwargs, _ = get_viewinganglecolor_for_colorbar(dirbin, scaledmap, plotkwargs, args)

                plotkwargs["linestyle"] = args.linestyle[modelnumber]
                plotkwargs["linewidth"] = args.linewidth[modelnumber] or (4 if args.subplots else 3.5)
                # plotkwargs is reused for each model, thus a model with no -linealpha takes None, the default alpha
                plotkwargs["alpha"] = args.linealpha[modelnumber]

                (modelline,) = axis.plot(time, brightness_in_mag, **plotkwargs)
                if residualseries is not None:
                    residualseries.append(
                        ResidualSeries(
                            plotkwargs["label"] or f"direction bin {dirbin}",
                            np.asarray(time, dtype=np.float64),
                            np.asarray(brightness_in_mag, dtype=np.float64),
                            modelline.get_color(),
                            line=modelline,
                        )
                    )

    # once for the whole figure: the helper draws every band of a reference file onto its own panel, so
    # calling it per band drew each reference curve once per band and re-read the file each time. It also
    # asserted a single axes, which a figure with more than one band never has
    for refindex, reflightcurve in enumerate(args.reflightcurves):
        plot_lightcurve_from_refdata(
            bandnames,
            reflightcurve,
            args.refspeccolors[refindex],
            args.refspecmarkers[refindex],
            ax,
            linewidth=args.linewidth[len(modelpaths) + refindex] or None,
            residualseries=residualseries,
        )

    if args.plot_hesma_model:
        plot_hesma_lightcurve(axes, bandnames, args, residualseries)

    ax = set_axis_properties(ax, args, xlimits=(args.timemin, args.timemax, "-timemin"))
    fig, ax = set_lightcurve_plot_labels(fig, ax, args, band_name=bandnames[0] if bandnames else None)
    set_lightcurveplot_legend(ax, args)

    if args.colorbarcostheta or args.colorbarphi:
        make_colorbar_viewingangles(scaledmap, args, fig=fig, ax=ax)

    invert_magnitude_yaxis(ax)

    if residualaxis is not None and residualseries is not None:
        assert isinstance(ax, mplax.Axes)
        dfresidualstats = draw_residual_panel(residualaxis, ax, residualseries, args, ismagnitude=True)
        if args.write_data:
            write_residual_stats(dfresidualstats, args.outputfile)

    save_figure(fig, args.outputfile, args=args, dpi=args.dpi)


def plot_hesma_lightcurve(
    axes: Sequence[mplax.Axes],
    bandnames: Sequence[str],
    args: argparse.Namespace,
    residualseries: list[ResidualSeries] | None,
) -> None:
    """Draw each band of the HESMA model of -plot_hesma_model on the panel of that band.

    The first column of the file holds the time, and a column with the name of a band holds its magnitudes. The grid
    of panels can hold more panels than bands, e.g. six panels for four bands.
    """
    hesma_model = read_hesma_lightcurve(args)
    timecolumn = hesma_model.columns[0]
    label = Path(args.plot_hesma_model).stem
    for axis, band_name in zip(axes[: len(bandnames)], bandnames, strict=True):
        if band_name in hesma_model.columns:
            (hesmaline,) = axis.plot(hesma_model[timecolumn], hesma_model[band_name], color="black", label=label)
            if residualseries is not None:
                residualseries.append(
                    ResidualSeries(
                        label,
                        np.asarray(hesma_model[timecolumn].to_numpy(), dtype=np.float64),
                        np.asarray(hesma_model[band_name].to_numpy(), dtype=np.float64),
                        hesmaline.get_color(),
                        line=hesmaline,
                    )
                )


def get_dirbin_palette(seriescolors: Sequence[str | None]) -> list["mplt.ColorType"]:
    """Return the colours to hand out to the direction bins of a colour evolution plot.

    A direction bin must not repeat the colour that a whole model was given, since both go on the same axes.
    The whole map is the fallback for the case where every one of its colours was assigned to a series: a bin
    has to be drawn in some colour, and there is none left that no series holds.
    """
    # a tuple and not a numpy row: matplotlib gives no label colour to a series with a numpy row as its colour
    tab20colors: list[mplt.ColorType] = [
        (float(r), float(g), float(b)) for r, g, b, _a in plt.get_cmap("tab20")(np.linspace(0, 1.0, 20))
    ]

    return get_unused_colors(tab20colors, seriescolors) or tab20colors


def colour_evolution_plot(modelpaths: Sequence[str | Path], args: argparse.Namespace) -> None:
    """Plot the evolution of the colour between each pair of bands for every model, and save the figure."""
    angle_counter = 0
    color_list = get_dirbin_palette([*args.color, *args.refspeccolors])

    fig, ax = create_axes(args)
    axes = iter_axes(ax)

    # the filter pairs share bands, so integrate the bands of every pair once per direction bin
    bandnames = sorted({name for filters in args.colour_evolution for name in filters.split("-")})
    filterfunc = get_filterfunc(args)

    for modelnumber, modelpath in enumerate(modelpaths):
        modelname = get_model_name(modelpath)
        if args.verbose:
            print(f"Reading spectra: {get_model_logname(modelpath)}")

        dirbins, dirbin_definition = parse_directionbin_args(modelpath, args)

        for dirbin in dirbins:
            if len(dirbins) > 1:
                # -color has one colour per model, which cannot distinguish a model's direction bins, so take a
                # colour per bin from the colour map instead, wrapping around when it runs out. The colour is
                # chosen once per bin, so a bin keeps its colour across the subplots of the filter pairs.
                dirbincolor = color_list[angle_counter % len(color_list)]
                angle_counter += 1
            else:
                dirbincolor = args.color[modelnumber]

            band_lightcurve_data = generate_band_lightcurve_data(modelpath, args, dirbin=dirbin, filternames=bandnames)

            for plotnumber, filters in enumerate(args.colour_evolution):
                filter_names = filters.split("-")
                plot_times, colour_delta_mag = get_colour_delta_mag(band_lightcurve_data, filter_names)

                if filterfunc is not None:
                    colour_delta_mag = filterfunc(colour_delta_mag)

                axes[plotnumber].plot(
                    plot_times,
                    colour_delta_mag,
                    label=get_linelabel(modelname, modelnumber, dirbin, dirbin_definition, args),
                    color=dirbincolor,
                    linestyle=args.linestyle[modelnumber],
                    linewidth=args.linewidth[modelnumber] or (4 if args.subplots else 3),
                    alpha=args.linealpha[modelnumber],
                )

    # once for the whole figure, as on the band plot: the reference data does not depend on the models or
    # their direction bins, so drawing it inside those loops re-read each file once per model per pair
    for refindex, reflightcurve in enumerate(args.reflightcurves):
        for plotnumber, filters in enumerate(args.colour_evolution):
            plot_color_evolution_from_data(
                filters.split("-"),
                reflightcurve,
                args.refspeccolors[refindex],
                args.refspecmarkers[refindex],
                ax,
                plotnumber,
                linewidth=args.linewidth[len(modelpaths) + refindex] or None,
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
    ax = set_axis_properties(ax, args, xlimits=(args.timemin, args.timemax, "-timemin"))
    set_lightcurveplot_legend(ax, args)

    invert_magnitude_yaxis(ax)

    save_figure(fig, args.outputfile, args=args, dpi=args.dpi)


def deredden_band_magnitudes(dfband: pl.DataFrame, lambda0: float, a_v: float, r_v: float) -> pl.DataFrame:
    """Return the band data with the magnitudes corrected for reddening by the CCM89 extinction law.

    ccm89 gives the extinction in magnitudes at the reference wavelength of the band, and an extinction in
    magnitudes is additive, thus the correction is a subtraction. The earlier code turned the magnitudes
    into fluxes, applied the law, and turned them back. That round trip cancels: the flux zero point and
    the speed of light both divide out, which is why no such constant appears here.
    """
    from extinction import ccm89

    wavelengths = np.full(dfband.height, lambda0, dtype=float)
    extinction_mag = ccm89(wavelengths, a_v=a_v, r_v=r_v)

    return dfband.with_columns(pl.Series("magnitude", dfband["magnitude"].to_numpy() - extinction_mag))


def get_dereddened_band_data(
    lightcurve_data: pl.DataFrame, metadata: dict[str, t.Any], filter_name_raw: str, filterdir: Path
) -> pl.DataFrame:
    """Return the points of one band of a reference light curve, dereddened when the metadata gives the extinction."""
    _, lambda0, *_ = get_filter_data(filterdir, filter_name_raw)
    filter_name = FILTERNAME_ALIASES.get(filter_name_raw, filter_name_raw)
    dfband = lightcurve_data.filter(pl.col("band") == filter_name)

    # get_file_metadata derives the third extinction value when the metadata gives two of
    # a_v, r_v, and e_bminusv. One value alone is not enough for the correction.
    if "a_v" in metadata and "r_v" in metadata:
        print("Correcting for reddening")
        return deredden_band_magnitudes(dfband, lambda0, metadata["a_v"], metadata["r_v"])

    print_warning("did not correct for reddening")
    return dfband


def plot_lightcurve_from_refdata(
    filter_names: Sequence[str],
    lightcurvefilename: Path | str,
    color: t.Any,
    marker: t.Any,
    ax: npt.NDArray[np.object_] | mplax.Axes,
    linewidth: float | None = None,
    residualseries: list[ResidualSeries] | None = None,
) -> str | None:
    """Plot an observed band light curve, dereddened with CCM89, and return its legend label.

    The points have no line, thus -linewidth sets the size of each marker.
    """
    lightcurve_data, metadata = read_reflightcurve_band_data(lightcurvefilename)
    linename = metadata["label"]
    assert linename is None or isinstance(linename, str)
    filterdir = Path(get_path("artistools_dir"), "data/filters/")

    axes = iter_axes(ax)
    for axnumber, filter_name_raw in enumerate(filter_names):
        axis = axes[axnumber]
        if filter_name_raw == "bol":
            continue
        dfband = get_dereddened_band_data(lightcurve_data, metadata, filter_name_raw, filterdir)

        (refline,) = axis.plot(
            dfband["time"],
            dfband["magnitude"],
            marker=marker,
            linestyle="None",
            markersize=linewidth,
            label=linename,
            color=color,
        )
        if residualseries is not None:
            residualseries.append(
                ResidualSeries(
                    linename or str(lightcurvefilename),
                    np.asarray(dfband["time"].to_numpy(), dtype=np.float64),
                    np.asarray(dfband["magnitude"].to_numpy(), dtype=np.float64),
                    refline.get_color(),
                    line=refline,
                )
            )
    return linename


def plot_color_evolution_from_data(
    filter_names: Iterable[str],
    lightcurvefilename: Path | str,
    color: t.Any,
    marker: t.Any,
    ax: npt.NDArray[np.object_] | mplax.Axes,
    plotnumber: int,
    linewidth: float | None = None,
) -> None:
    """Plot the observed colour evolution between two bands, dereddened with CCM89.

    The points have no line, thus -linewidth sets the size of each marker.
    """
    lightcurve_from_data, metadata = read_reflightcurve_band_data(lightcurvefilename)
    filterdir = Path(get_path("artistools_dir"), "data/filters/")

    filter_data = [
        get_dereddened_band_data(lightcurve_from_data, metadata, filter_name_raw, filterdir)
        for filter_name_raw in filter_names
    ]

    merge_dataframes = filter_data[0].join(
        filter_data[1], how="inner", on="time", suffix="_second", maintain_order="left"
    )
    axis = iter_axes(ax)[plotnumber]
    axis.plot(
        merge_dataframes["time"],
        merge_dataframes["magnitude"] - merge_dataframes["magnitude_second"],
        marker=marker,
        linestyle="None",
        markersize=linewidth,
        label=metadata["label"],
        color=color,
    )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(
        parser,
        positional=True,
        multiplepaths=True,
        default=[],
        helptext=(
            "Path(s) to ARTIS folders with light_curve.out or packets files (may include wildcards such as * and **)"
        ),
    )

    addarg_seriesstyle(parser, include_linealpha=True)

    addarg_legend(parser)

    parser.add_argument(
        "-title",
        dest="title",
        nargs="?",
        const=True,
        default=None,
        help="Show a plot title: pass the title text, or use the bare flag for the model name",
    )
    # deprecated spelling kept as a hidden alias
    parser.add_argument("--title", dest="title", nargs="?", const=True, help=argparse.SUPPRESS)

    addarg_figscale(parser, include_figwidthscale=True)

    parser.add_argument("--frompackets", action="store_true", help="Read packets files instead of light_curve.out")

    addarg_maxpacketfiles(parser)

    parser.add_argument("--gamma", action="store_true", help="Make light curve from gamma rays")

    parser.add_argument(
        "--rpkt",
        "--uvoir",
        dest="rpkt",
        action="store_true",
        help="Make light curve from R-packets (default unless --gamma is passed)",
    )

    # the old spelling of the same choice, which a script can still hold
    parser.add_argument("-escape_type", choices=("TYPE_RPKT", "TYPE_GAMMA"), default=None, help=argparse.SUPPRESS)

    addarg_output(parser, kind="file", helptext="Filename for PDF file")

    parser.add_argument("--plotcmf", action="store_true", help="Plot comoving frame light curve")
    # deprecated spellings kept as hidden aliases
    parser.add_argument(
        "--plot_cmf", "--showcmf", "--show_cmf", dest="plotcmf", action="store_true", help=argparse.SUPPRESS
    )

    parser.add_argument(
        "--plotinvalidpart",
        action="store_true",
        help="Plot the entire light curve including partially accumulated parts (light travel time effects)",
    )

    parser.add_argument(
        "-deposition",
        nargs="+",
        choices=DEPOSITIONCHOICES,
        metavar="PARTICLE",
        help=(
            "Plot the deposition rate of each particle from deposition.out:"
            f" {', '.join(DEPOSITIONCHOICES)}. total is the sum of the rates of all the particles"
        ),
    )

    parser.add_argument(
        "-emission",
        nargs="+",
        choices=tuple(EMISSIONCOLUMNS),
        metavar="PARTICLE",
        help=(
            "Plot the Monte Carlo emission rate of each particle, which ARTIS counts at the decays of the pellets:"
            f" {', '.join(EMISSIONCOLUMNS)}. deposition.out holds no such rate of betaplus"
        ),
    )

    parser.add_argument(
        "-analyticemission",
        nargs="+",
        choices=tuple(ANALYTICEMISSIONCOLUMNS),
        metavar="PARTICLE",
        help=(
            "Plot the analytical emission rate of each particle, which ARTIS calculates from the decay rates:"
            f" {', '.join(ANALYTICEMISSIONCOLUMNS)}"
        ),
    )

    parser.add_argument(
        "-thermalisation",
        nargs="+",
        choices=ENERGYPARTICLES,
        metavar="PARTICLE",
        help=(
            "Plot the deposition rate over the emission rate of each particle in a panel below the light curves:"
            f" {', '.join(ENERGYPARTICLES)}. --showbarnes adds the published thermalisation curves"
        ),
    )

    parser.add_argument(
        "--showbarnes",
        action="store_true",
        help=(
            "With -thermalisation, also plot the thermalisation efficiencies for gamma, betaminus, and alpha of"
            " Barnes, Kasen, Wu & Martínez-Pinedo (2016), ApJ, 829, 110, doi:10.3847/0004-637X/829/2/110"
        ),
    )

    # the older flags of the energy rates. resolve_energy_rate_args gives each one the particles that it drew
    parser.add_argument("--plotdeposition", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plotalphadeposition", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plotthermalisation", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument(
        "-topnucs", type=int, default=0, help="Show light curves from top n nuclides energy contributions"
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
            "Plot the light curves of these bands, e.g. U B V R I, or bol for bolometric. With no -filter the "
            "command plots the bolometric light curve. The names are case sensitive, e.g. sloan-r is rs"
        ),
    )

    parser.add_argument(
        "-colour_evolution", nargs="*", help="Plot the colour evolution of each pair of bands, e.g. B-V U-B"
    )

    parser.add_argument("--print_data", action="store_true", help="Print plotted data")

    parser.add_argument(
        "--write_data",
        action="store_true",
        help="Save the band light curves of -filter and the statistics of -residual in text files",
    )

    addarg_residuals(parser)

    parser.add_argument(
        "-plot_hesma_model",
        action="store",
        type=Path,
        default=False,
        help="Plot hesma model on top of lightcurve plot. Enter model name saved in data/hesma directory",
    )

    addarg_viewingangle(parser, allow_select_all=True)

    addarg_axislimits(parser, include_x=False)

    parser.add_argument("-timemin", "-xmin", type=float, default=None, help="Plot range: x-axis minimum")

    parser.add_argument("-timemax", "-xmax", type=float, default=None, help="Plot range: x-axis maximum")
    # deprecated spellings kept as hidden aliases
    parser.add_argument("-timedaysmin", dest="timemin", type=float, help=argparse.SUPPRESS)
    parser.add_argument("-timedaysmax", dest="timemax", type=float, help=argparse.SUPPRESS)

    parser.add_argument("--logscalex", action="store_true", help="Use log scale for horizontal axis")

    addarg_yscale(parser)

    # the older spelling of "-yscale log"
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

    # every plot of this command draws the time in the frame of the model, thus a redshift has no use
    addarg_unsupported(parser, "-redshifttoz", instead="no argument")

    addarg_filter(parser)

    addarg_dpi(parser)

    addarg_show(parser)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open a window with controls for the light curves, the energy rates, and the axes, and show the command",
    )
    addarg_verbose(parser)

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
            "viewing angles you want to use before making the scatter "
            "plots"
        ),
    )

    parser.add_argument(
        "--test_viewing_angle_fit",
        action="store_true",
        help=(
            "Plot the light curve and its fit for each viewing angle, to check the fit. Use it with"
            " --save_viewing_angle_peakmag_risetime_delta_m15_to_file or with a scatter plot flag, which make the fits"
        ),
    )

    parser.add_argument(
        "--make_viewing_angle_peakmag_risetime_scatter_plot",
        action="store_true",
        help=(
            "Makes scatter plot of band peak mag with risetime with the "
            "angle averaged values being the solid dot and the error bars "
            "representing the standard deviation of the viewing angle "
            "distribution"
        ),
    )

    parser.add_argument(
        "--make_viewing_angle_peakmag_delta_m15_scatter_plot",
        action="store_true",
        help=(
            "Makes scatter plot of band peak with delta m15 with the angle "
            "averaged values being the solid dot and the error bars representing "
            "the standard deviation of the viewing angle distribution"
        ),
    )

    parser.add_argument(
        "--include_delta_m40",
        action="store_true",
        help="When calculating delta_m15, calculate delta_m40 as well. Only affects the saved viewing angle data",
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

    addarg_notitle(parser)

    addarg_timestep(parser, helptext="Timestep, or a range e.g. 20-30, to plot")

    addarg_timedays(parser, helptext="Time in days, or a range e.g. 2.2-2.8, to plot")

    parser.add_argument("--nomodelname", action="store_true", help="Model name not added to linename in legend")

    parser.add_argument("-legendsubplotnumber", type=int, default=1, help="Subplot number to place legend in")

    parser.add_argument("-legendposition", type=str, default="best", help="Position of legend in plot")

    # the old spelling of -legendcols, which addarg_legend adds
    parser.add_argument("-ncolslegend", dest="legendcols", type=positive_int_arg, help=argparse.SUPPRESS)

    # the old spelling of --legendframe, which addarg_legend adds
    parser.add_argument("--legendframeon", dest="legendframe", action="store_true", help=argparse.SUPPRESS)

    addarg_labelfontsize(parser)


def check_colour_evolution_args(args: argparse.Namespace) -> None:
    """Stop the command when a value of -colour_evolution does not name two bands."""
    for colour in args.colour_evolution or ():
        if len(colour.split("-")) != 2:
            exit_with_error(
                f"-colour_evolution takes two bands for each colour, and {colour} names {len(colour.split('-'))} bands",
                "Give each colour as two bands with a minus sign between them, e.g. -colour_evolution B-V",
            )


def check_residual_args(args: argparse.Namespace) -> None:
    """Stop the command when -residual cannot apply to the plot that args selects."""
    otherplotoptions = (
        args.colour_evolution,
        args.colouratpeak,
        args.brightnessattime,
        args.save_viewing_angle_peakmag_risetime_delta_m15_to_file,
        args.save_angle_averaged_peakmag_risetime_delta_m15_to_file,
        args.make_viewing_angle_peakmag_risetime_scatter_plot,
        args.make_viewing_angle_peakmag_delta_m15_scatter_plot,
    )
    if any(otherplotoptions):
        exit_with_error(
            "-residual applies only to a bolometric light curve plot and to a band light curve plot",
            "Remove -residual, or remove the option that selects a different plot",
        )
    if args.filter and (len(args.filter) != 1 or args.filter[0] == "bol"):
        exit_with_error(
            "-residual applies to a plot of one frame, and the band reference data hold no bolometric band",
            "Give one filter, e.g. -filter B. For a bolometric light curve, give no -filter",
        )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot ARTIS light curve."""
    # the dispatcher parses the command line and gives args alone, thus the viewer then reads sys.argv
    fromdispatcher = args is not None
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.interactive:
        from artistools.lightcurve.interactive import run_viewer
        from artistools.viewertools.core import get_command_tokens

        run_viewer(get_command_tokens(args, argsraw, kwargs, fromdispatcher=fromdispatcher))
        return

    resolve_plot_args(args)
    modelpaths = args.modelpath
    outputfolder = args.outputfile.parent

    # determine if this will be a scatter plot or not
    if any(getattr(args, dest) for dest in PEAKMAGDESTS):
        peakmag_risetime_declinerate_init(modelpaths, args)
        return

    if args.colouratpeak:  # make scatter plot of colour at peak, eg. B-V at Bmax
        make_peak_colour_viewing_angle_plot(args)
        return

    if args.brightnessattime:
        if args.timedays is None:
            misc.exit_with_error("specify a single time with -timedays")
        # this plot takes one time rather than a range
        try:
            args.timedays = float(args.timedays)
        except ValueError:
            exit_with_error(
                f"--brightnessattime takes one time, and -timedays gives {args.timedays}",
                "Give one time in days, e.g. -timedays 5",
            )
        if not args.plotviewingangle:
            args.plotviewingangle = [-1]
        if not args.colorbarcostheta and not args.colorbarphi:
            args.colorbarphi = True
        plot_viewanglebrightness_at_fixed_time(Path(modelpaths[0]), args)
        return

    if args.filter:
        make_band_lightcurves_plot(modelpaths, outputfolder, args)

    elif args.colour_evolution:
        colour_evolution_plot(modelpaths, args)
    else:
        fig, axis, thermaxis, residualaxis = make_plot_figure(args)
        dfresidualstats = draw_plot(args, axis, thermaxis, residualaxis)
        if args.write_data and dfresidualstats is not None:
            write_residual_stats(dfresidualstats, args.outputfile)
        save_figure(fig, args.outputfile, args=args, dpi=args.dpi)


class RefusedOption(t.NamedTuple):
    """An option that plotlightcurves refuses together with a different option of the command."""

    flag: str
    reason: str
    # the help text of the error if the command stops, or None if the command drops the option with a warning
    remedy: str | None


def get_other_plot_flag(args: argparse.Namespace) -> str | None:
    """Return the flag of the plot that is not the plot of the bolometric light curves, or None for that plot.

    Such a plot reads the spectra or the light curve files of ARTIS, and no packets files.
    """
    otherplotflags = (
        ("-filter", args.filter),
        ("-colour_evolution", args.colour_evolution),
        ("--colouratpeak", args.colouratpeak),
        ("--brightnessattime", args.brightnessattime),
        *((f"--{dest}", getattr(args, dest)) for dest in PEAKMAGDESTS),
    )
    return next((flag for flag, isgiven in otherplotflags if isgiven), None)


def get_refused_options(args: argparse.Namespace) -> list[RefusedOption]:
    """Return each option of args that plotlightcurves refuses together with a different option, and the reason.

    The command stops for an option with a remedy, and it drops a different option with a warning. The viewer drops
    each option. Thus the command and the viewer read the same rules.
    """
    refused: list[RefusedOption] = []
    isobserver = args.plotvspecpol is not None
    otherplotflag = get_other_plot_flag(args)
    # the comoving frame energy of the packets gives a luminosity, which has no magnitude. The virtual packets hold
    # the energy of their observer in the rest frame only
    if args.plotcmf and args.magnitude:
        refused.append(
            RefusedOption(
                "--plotcmf",
                "--plotcmf gives a comoving frame luminosity, which has no magnitude",
                "Remove --plotcmf or --magnitude",
            )
        )
    elif args.plotcmf and isobserver:
        refused.append(
            RefusedOption(
                "--plotcmf",
                "the virtual packets hold no comoving frame energy, thus --plotcmf draws no curve of an observer",
                None,
            )
        )
    elif args.plotcmf and args.use_pellet_decay_time:
        refused.append(
            RefusedOption(
                "--plotcmf",
                "--plotcmf bins the comoving frame light curve by the arrival time, and --use_pellet_decay_time bins by"
                " the decay time. Thus the two curves have different time axes",
                None,
            )
        )
    pelletremedy = "Remove -plotvspecpol, or remove -topnucs and --use_pellet_decay_time"
    for flag, isgiven in (("-topnucs", args.topnucs), ("--use_pellet_decay_time", args.use_pellet_decay_time)):
        if isgiven and isobserver:
            reason = f"the virtual packets hold no pellet, thus {flag} gives no light curve of an observer"
            refused.append(RefusedOption(flag, reason, pelletremedy))
    if args.gamma and isobserver:
        refused.append(
            RefusedOption(
                "--gamma",
                "ARTIS makes the virtual packets from the r-packets, thus an observer has no gamma-ray light curve",
                "Remove -plotvspecpol or --gamma",
            )
        )
    if otherplotflag is not None:
        refused.extend(
            RefusedOption(flag, f"{otherplotflag} reads no packets files, thus {flag} has no effect", None)
            for flag, isgiven in (
                ("--frompackets", args.frompackets),
                ("-topnucs", args.topnucs),
                ("--use_pellet_decay_time", args.use_pellet_decay_time),
            )
            if isgiven
        )
    elif args.use_pellet_decay_time and not isobserver and not (args.frompackets or args.topnucs):
        refused.append(
            RefusedOption(
                "--use_pellet_decay_time",
                "only the packets give the decay time of a pellet, and the command reads no packets files",
                "Give --frompackets, or remove --use_pellet_decay_time",
            )
        )
    refused.extend(get_refused_average_options(args))
    if args.test_viewing_angle_fit and not any(getattr(args, dest) for dest in PEAKMAGDESTS):
        refused.append(
            RefusedOption(
                "--test_viewing_angle_fit",
                "--test_viewing_angle_fit plots the fits of the peak magnitudes, and only a save flag or a scatter"
                " flag makes these fits",
                None,
            )
        )
    return refused


def get_refused_average_options(args: argparse.Namespace) -> list[RefusedOption]:
    """Return each average over an angle that plotlightcurves refuses for the direction bins of args."""
    averageflags = (
        ("--average_over_phi_angle", args.average_over_phi_angle, {"average_over_phi": True}),
        ("--average_over_theta_angle", args.average_over_theta_angle, {"average_over_theta": True}),
    )
    if not args.plotviewingangle:
        return [
            RefusedOption(
                flag, f"{flag} groups the direction bins of -plotviewingangle, and the command gives none", None
            )
            for flag, isgiven, _ in averageflags
            if isgiven
        ]
    refused: list[RefusedOption] = []
    # the virtual packet observers are no direction bins, thus an average does not group them
    if args.plotvspecpol is None:
        for flag, isgiven, averagekwargs in averageflags:
            groupstarts = get_dirbins(**averagekwargs)
            # -1 is the angle average, and -2 selects the first bin of each group
            if isgiven and (
                badbins := [dirbin for dirbin in args.plotviewingangle if dirbin not in {-1, -2, *groupstarts}]
            ):
                refused.append(
                    RefusedOption(
                        flag,
                        f"{flag} gives one curve for each group of direction bins, and the bins"
                        f" {' '.join(str(dirbin) for dirbin in badbins)} start no group",
                        "Give the first bin of each group, e.g. -plotviewingangle"
                        f" {' '.join(map(str, groupstarts[:3]))}, or give -plotviewingangle -2",
                    )
                )
    return refused


def drop_refused_options(
    args: argparse.Namespace, refused: Iterable[RefusedOption], warningformat: str = "{reason}"
) -> None:
    """Give each refused option the value of no selection, and print a warning.

    The value of no selection is 0 for -topnucs and False for a different flag. warningformat makes the text of the
    warning from the fields flag and reason.
    """
    for option in refused:
        print_warning(warningformat.format(flag=option.flag, reason=option.reason))
        setattr(args, option.flag.lstrip("-"), 0 if option.flag == "-topnucs" else False)


def check_refused_options(args: argparse.Namespace) -> None:
    """Stop the command for a refused option that has a remedy, and drop each other refused option with a warning."""
    refused = get_refused_options(args)
    for option in refused:
        if option.remedy is not None:
            exit_with_error(option.reason, option.remedy)
    drop_refused_options(args, refused)


def resolve_plot_args(args: argparse.Namespace) -> None:
    """Give args the values that the plot reads: the paths, the time range, the styles, and the output file.

    Each change here also applies to the plot of the viewer, which calls this function for each command.
    """
    args.modelpath = normalize_path_list(args.modelpath)
    apply_time_range_args(args, args.modelpath)
    resolve_energy_rate_args(args)
    if args.escape_type is not None:
        # -escape_type is the old spelling of --rpkt and --gamma. It reached no reader before, thus
        # -escape_type TYPE_GAMMA gave an R-packet light curve. It names one type, thus it sets one
        args.gamma = args.escape_type == "TYPE_GAMMA"
        args.rpkt = not args.gamma
    check_refused_options(args)
    check_colour_evolution_args(args)

    nmodels = len(args.modelpath)
    args.reflightcurves = makelist(args.reflightcurves)
    args.refspeccolors, args.refspecmarkers = trim_or_pad(
        len(args.reflightcurves), args.refspeccolors, args.refspecmarkers
    )

    # the reference data get black and greys, and the ARTIS models get the colours of the cycle. The
    # -reflightcurves series come after the model paths, thus their greys continue the sequence
    isreference = [path_is_reference_lightcurve(path) for path in args.modelpath]
    seriescolors = resolve_series_styles(
        args,
        [*isreference, *([True] * len(args.reflightcurves))],
        [*trim_or_pad(nmodels, args.color)[0], *args.refspeccolors],
        "label",
        "linestyle",
        "dashes",
        "linewidth",
        "linealpha",
    )
    args.color = seriescolors[:nmodels]
    args.refspeccolors = seriescolors[nmodels:]

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

    if args.residuals is not None and args.residualbaselineseries is None:
        args.residualbaselineseries = 0
    if args.residualbaselineseries is not None:
        check_residual_args(args)

    # the default name says what the figure holds. -o keeps the name that the user gave, thus the
    # plot functions below take the resolved name and do not make one of their own
    if args.filter:
        defaultoutputfile = (
            f"plot{get_filename_part(args.filter[0])}lightcurves.pdf"
            if len(args.filter) == 1
            else "plotlightcurves.pdf"
        )
    elif args.colour_evolution:
        defaultoutputfile = f"plotcolorevolution{get_filename_part('_'.join(args.colour_evolution))}.pdf"
    else:
        defaultoutputfile = "plotlightcurves.pdf"

    args.outputfile = resolve_outputfile(args.outputfile, defaultoutputfile)
