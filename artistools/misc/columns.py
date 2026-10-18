"""Give the meaning and the unit of each column of a table that --write_data writes.

The name of a column gives its unit in a short form, e.g. f_lambda. A sentence removes the doubt about the frame,
the distance, or the normalisation of the values.
"""

import re
import typing as t
from collections.abc import Sequence

# the plotted y variables of plotspectra (-yvariable), with the quantity and the unit of each one
SPECTRUM_YVARIABLES: t.Final = (
    ("flux", "flux per unit of x_plotted, at the distance of -distmpc", "erg/s/cm^2/(unit of x_plotted)"),
    ("luminosity", "luminosity per unit of x_plotted", "erg/s/(unit of x_plotted)"),
    ("photonflux", "photon flux per unit of x_plotted, at the distance of -distmpc", "1/s/cm^2/(unit of x_plotted)"),
    ("photoncount", "photon emission rate per unit of x_plotted", "1/s/(unit of x_plotted)"),
    ("packetcount", "number of Monte Carlo packets in the bin", ""),
    (
        "eflux",
        "energy flux E F_E, or lambda F_lambda on a wavelength axis, at the distance of -distmpc",
        "erg/s/cm^2, or (unit of x_plotted)/s/cm^2 on an axis of energy or frequency",
    ),
)

# One record (pattern, meaning, unit) for each family of columns, in the order of the search. The pattern must match
# the full name of the column. The meaning and the unit can hold {0} and {1}, which take the groups of the pattern. A
# unit of None takes the unit from the estimator table of artistools.estimators.core, thus the two cannot disagree.
# An empty unit gives no unit.
COLUMN_FAMILIES: t.Final[tuple[tuple[str, str, str | None], ...]] = (
    # the spectra of plotspectra and the other spectra
    ("lambda_angstroms", "the wavelength, in the rest frame of the source for an emergent spectrum", "Å"),
    ("x_plotted_(\\w+)", "the plotted horizontal value: the rest-frame wavelength, frequency, or photon energy", "{0}"),
    ("x_(lower|upper)_(\\w+)", "the {0} edge of the bin of the plotted horizontal value", "{1}"),
    ("f_lambda(\\..+)?", "the flux per unit wavelength at 1 Mpc", "erg/s/cm^2/Å"),
    (
        "emission_flambda\\..+",
        "the flux per unit wavelength at 1 Mpc of the packets with their last emission in the suffix group",
        "erg/s/cm^2/Å",
    ),
    (
        "absorption_flambda\\..+",
        "the flux per unit wavelength at 1 Mpc of the packets with their last absorption in the suffix group",
        "erg/s/cm^2/Å",
    ),
    *(
        record
        for yvariable, quantity, unit in SPECTRUM_YVARIABLES
        for record in (
            (f"{yvariable}_plotted(\\..+)?", f"the plotted {quantity}, or divided by the peak with --normalised", unit),
            (
                f"emission_{yvariable}_plotted\\..+",
                f"the plotted {quantity}, from the packets with their last emission in the suffix group",
                unit,
            ),
            (
                f"absorption_{yvariable}_plotted\\..+",
                f"the plotted {quantity}, from the packets with their last absorption in the suffix group, below 0",
                unit,
            ),
            (f"{yvariable}_relnoise(\\..+)?", "the relative Monte Carlo noise of the plotted value of the bin", ""),
            (f"peak_{yvariable}", f"the largest drawn {quantity}", unit),
        )
    ),
    ("x_(min|max)_(\\w+)", "the {0}imum plotted horizontal value of the drawn points", "{1}"),
    ("x_at_peak_(\\w+)", "the plotted horizontal value of the largest drawn point", "{0}"),
    ("lambda_(min|max)_angstroms", "the {0}imum wavelength of the integral of the luminosity", "Å"),
    (
        "gap_width_angstroms",
        "the width of the wavelength range with no value of f_lambda, which the integral omits",
        "Å",
    ),
    ("packet_count", "the number of packets in the drawn bins", ""),
    ("relnoise_luminosity", "the relative Monte Carlo noise of the luminosity of the drawn bins", ""),
    (
        "f_lambda_arbitrary",
        "the emission spectrum of the ion from a Gaussian profile for each line, in an arbitrary unit",
        "",
    ),
    (
        "f_lambda_normalised",
        "the emission spectrum of the ion from a Gaussian profile for each line, divided by its peak",
        "",
    ),
    ("packetcount", "the number of Monte Carlo packets of the point", ""),
    ("relnoise", "the relative Monte Carlo noise of the point, sqrt(sum(e^2)) / sum(e) for the packet energies e", ""),
    ("transmission", "the transmission of the filter curve", ""),
    ("(Q|U|V)/I", "the Stokes parameter {0} divided by the Stokes parameter I", ""),
    # the light curves
    (
        "luminosity_(erg/s|Lsun)",
        "the luminosity in the observer frame. A direction bin gives the isotropic-equivalent luminosity",
        "{0}",
    ),
    (
        "luminosity_cmf_(erg/s|Lsun)",
        "the luminosity from the energies of the escaped packets in the comoving frame of the ejecta",
        "{0}",
    ),
    ("mag", "the absolute bolometric magnitude, Mbol_sun - 2.5 log10(L / Lsun)", "mag"),
    ("luminosity_(erg/s|Lsun)_err(lower|upper)", "the {1} error bar of the luminosity of the reference data", "{0}"),
    ("mag_err(lower|upper)", "the {0} error bar of the magnitude of the reference data", "mag"),
    ("luminosity_err(minus|plus)_erg/s", "the {0} uncertainty of the luminosity in the reference file", "erg/s"),
    ("faint_side_unbounded", "true if the lower error bar of the reference luminosity reaches zero", ""),
    ("in_valid_range", "true if the light travel time leaves the light curve at this time complete", ""),
    ("escape_type", "the type of the escaped packets, e.g. TYPE_RPKT or TYPE_GAMMA, or reference", ""),
    ("pellet_nucname", "the nuclide of the pellets that give the light curve", ""),
    ("band", "the name of the filter band", ""),
    ("magnitude", "the absolute magnitude in the band", "mag"),
    ("colour", "the names of the two bands of the colour", ""),
    ("colour_mag", "the magnitude in the first band minus the magnitude in the second band", "mag"),
    ("peakmag", "the peak magnitude of the polynomial fit to the light curve of the band", "mag"),
    ("risetime_days", "the time of the peak of the polynomial fit, from the explosion", "days"),
    ("deltam15", "the magnitude 15 days after the peak minus the peak magnitude, from the polynomial fit", "mag"),
    (
        "risetime_days_err",
        "the half width of the error bar of the angle average, the standard deviation over the direction bins",
        "days",
    ),
    (
        "(peakmag|deltam15)_err",
        "the half width of the error bar of {0} of the angle average, the standard deviation over the direction bins",
        "mag",
    ),
    ("time_(min|max)_days", "the {0}imum time of the points in the time range, from the explosion", "days"),
    ("peak_time_days", "the time of the peak, from the explosion", "days"),
    ("peak_mag", "the smallest magnitude in the time range", "mag"),
    ("peak_luminosity_erg_per_s", "the largest luminosity in the time range", "erg/s"),
    ("peak_luminosity_Lsun", "the largest luminosity in the time range", "Lsun"),
    ("peak_packet_count", "the number of packets of the peak point", ""),
    ("peak_relnoise", "the relative Monte Carlo noise of the peak point", ""),
    ("peak_noise_mag", "the Monte Carlo noise of the magnitude of the peak point", "mag"),
    ("deltam15_mag", "the magnitude 15 days after the peak minus the peak magnitude", "mag"),
    ("quantity", "the name of the plotted energy rate or thermalisation ratio", ""),
    ("unit", "the unit of the column value", ""),
    # the viewing directions
    ("dirbin", "the index of the viewing direction bin, or -1 for the average of all directions", ""),
    ("costhetabin", "the index of the bin of cos(theta) of the viewing direction", ""),
    ("phibin", "the index of the bin of the azimuth phi of the viewing direction", ""),
    ("phibinmonotonicasc", "the index of the bin of the azimuth phi of the viewing direction, in the order of phi", ""),
    ("costheta_(lower|upper)", "the {0} edge of the pixel in cos(theta) of the viewing direction", ""),
    ("longitude_(lower|upper)_rad", "the {0} edge of the pixel in the longitude of the map, from -pi to pi", "rad"),
    ("timemin_days", "the start of the range of arrival times of the map", "days"),
    ("timemax_days", "the end of the range of arrival times of the map", "days"),
    (
        "luminosity_erg_per_s",
        "the isotropic-equivalent luminosity in the observer frame of a map pixel, the spectrum, or a velocity bin",
        "erg/s",
    ),
    (
        "luminosity_per_velocity_erg_per_s_per_kmps",
        "the isotropic-equivalent luminosity of the packets of the velocity bin, divided by the width of the bin",
        "erg/s/(km/s)",
    ),
    (
        "luminosity_per_velocity_erg_per_s_per_c",
        "the isotropic-equivalent luminosity of the packets of the velocity bin, divided by the width of the bin",
        "erg/s/c",
    ),
    ("emvelocityoverc", "the energy-weighted mean speed of the packets at their last interaction", "c"),
    ("emvelocityoverc_sigma", "the energy-weighted standard deviation of the speed at the last interaction", "c"),
    (
        "emlosvelocityoverc",
        "the energy-weighted mean line-of-sight velocity at the last interaction, positive toward the observer",
        "c",
    ),
    ("temperature_K", "the energy-weighted mean radiation temperature TR of the cells of the last interaction", "K"),
    ("temperature_sigma_K", "the energy-weighted standard deviation of TR of the cells of the last interaction", "K"),
    (
        "nnelement_(\\w+)_per_cm3",
        "the energy-weighted mean number density of {0} in the cells of the last interaction",
        "cm^-3",
    ),
    (
        "emissivity_erg_per_s_per_cm3",
        "the 4 pi equivalent emissivity of the escaped packets of the direction bin, at their last interaction",
        "erg/s/cm^3",
    ),
    ("v_r_(min|max)_c", "the {0} edge of the bin of the cylindrical radial velocity at the last interaction", "c"),
    ("v_z_(min|max)_c", "the {0} edge of the bin of the z velocity at the last interaction", "c"),
    ("pos_(x|y|z)_mid_cm", "the {0} coordinate of the centre of the cell at the time of the model snapshot", "cm"),
    ("(x|y|z)_cm", "the {0} coordinate of a point of the line of the direction bin", "cm"),
    # the model snapshot and the composition
    ("vel_(x|y|z|r|rcyl)_on_c", "the velocity coordinate {0} of the point (rcyl: the distance to the z axis)", "c"),
    ("vel_(x|y|z)_min_on_c", "the velocity coordinate {0} of the lower corner of the cell", "c"),
    ("mass_enclosed_msun", "the mass inside the radial velocity of the row", "Msun"),
    ("dmass_dv_msun_per_c", "the mass of the velocity bin divided by the width of the bin", "Msun/c"),
    ("Ye", "the electron fraction, as the mass-weighted mean for a bin", ""),
    ("rho_g_per_cm3", "the density of the cell at the time of the model snapshot", "g/cm^3"),
    ("rho_(\\w+)_g_per_cm3", "the density of {0} in the cell at the time of the model snapshot", "g/cm^3"),
    ("X_([A-Z][a-z]?\\d*)", "the mass fraction of the element or the nuclide {0}", ""),
    ("(.+)_of_isosurface", "the value of {0} at an isosurface of the figure", ""),
    ("Z", "the atomic number", ""),
    ("A", "the mass number", ""),
    ("massfraction", "the mass fraction of the species of the row", ""),
    ("abundance", "the number abundance, the sum of the mass fraction divided by the mass number of each nuclide", ""),
    # the nuclear network
    (
        "qdot_erg_per_s_per_g",
        "the radioactive energy release rate per unit mass of the ejecta, for the channel in the column series",
        "erg/s/g",
    ),
    ("source", "the code that gives the series: ARTIS, or the nuclear network GSINET", ""),
    (
        "specpath",
        "the path of the reference file as the command line gives it, because two files can have one label",
        "",
    ),
    # the opacities
    ("lambda_angstroms_bin_mid", "the wavelength at the middle of the bin", "Å"),
    ("lambda_angstroms_(lower|upper)", "the {0} edge of the wavelength bin", "Å"),
    (
        "exopac_cm2_per_g",
        "the expansion opacity of the lines in the wavelength bin, as the mass-weighted mean of the cells",
        "cm^2/g",
    ),
    (
        "linebinned_cm2_per_g",
        "the line-binned opacity, from the sum of the Sobolev optical depths, as the mass-weighted mean of the cells",
        "cm^2/g",
    ),
    (
        "linebinned_cap(.+)_cm2_per_g",
        "the line-binned opacity with each Sobolev optical depth capped at {0}, as the mass-weighted mean of the cells",
        "cm^2/g",
    ),
    ("(\\w+)_movingaverage_cm2_per_g", "the moving average over the wavelength of the opacity {0}", "cm^2/g"),
    (
        "planckmean_exopac_cm2_per_g",
        "the mass-weighted mean of the Planck mean of the expansion opacity at the excitation temperature of each cell",
        "cm^2/g",
    ),
    ("ratio_(\\w+)_to_exopac", "the opacity {0} divided by the expansion opacity", ""),
    ("linecount", "the number of lines in the wavelength bin", ""),
    # the radiation field
    (
        "J_lambda_erg_per_s_per_cm2_per_angstrom",
        "the mean intensity per unit wavelength of the radiation field of the cell, in the comoving frame",
        "erg/s/cm^2/sr/Å",
    ),
    # the line fluxes
    ("time", "the middle of the time bin, from the explosion", "days"),
    ("fratio", "the luminosity of the second feature divided by the luminosity of the first feature", ""),
    ("fratio_.+", "the luminosity of the second feature divided by the luminosity of the first feature", ""),
    ("feature", "the emission feature, or reference for a point of the reference data", ""),
    ("log10nne", "the base-10 logarithm of the electron density of the cell of the last emission", "log10(cm^-3)"),
    # the NLTE populations
    ("ion", "the ion, e.g. Fe II", ""),
    ("level", "the index of the level of the ion, from 0 for the ground level", ""),
    ("levelname", "the name of the level", ""),
    ("config", "the configuration and the term of the level, or superlevel", ""),
    ("odd_parity", "true if the level has odd parity", ""),
    ("n_NLTE_per_cm3", "the population of the level from the NLTE solution of ARTIS", "cm^-3"),
    (
        "n_LTE_T_(e|R)_per_cm3",
        "the population of the level in LTE at T{0}, scaled to the population of the ion",
        "cm^-3",
    ),
    ("n_Floers_per_cm3", "the population of the level in the reference NLTE data", "cm^-3"),
    ("departure_coeff", "the NLTE population divided by the LTE population at Te, scaled to the ground level", ""),
    ("departure_coeff_Floers", "the departure coefficient of the reference NLTE data", ""),
    # the lines of linecontributions
    ("lower", "the index of the lower level of the line in the ion, from 0 for the ground level", ""),
    ("upper", "the index of the upper level of the line in the ion", ""),
    ("A_per_s", "the Einstein coefficient of the spontaneous emission of the line", "s^-1"),
    ("T_exc", "the excitation temperature of the LTE populations and of the superlevel in the cell", "K"),
    ("n_(lower|upper)_per_cm3", "the population of the {0} level of the line in the cell", "cm^-3"),
    (
        "n_(lower|upper)_source",
        "the method of the population of the {0} level: NLTE, superlevel, Boltzmann from the ground level, or LTE",
        "",
    ),
    ("tau_sobolev", "the Sobolev optical depth of the line in the cell", ""),
    ("tau_sobolev_max", "the maximum Sobolev optical depth of the line in the cells", ""),
    ("velocity_kmps_at_tau_sobolev_max", "the radial velocity of the cell with the maximum tau_sobolev", "km/s"),
    ("cellcount_tau_sobolev_above_1", "the number of cells where the line has a tau_sobolev above 1", ""),
    ("escape_probability", "the Sobolev escape probability (1 - exp(-tau)) / tau of the line in the cell", ""),
    (
        "line_emissivity_erg_per_s_per_cm3",
        (
            "the energy that leaves a unit volume of the cell in the line each second, n_u A h nu times the escape"
            " probability"
        ),
        "erg/s/cm^3",
    ),
    (
        "emission_erg_per_s",
        "the energy that leaves the cell in the line each second, or the sum over all the cells",
        "erg/s",
    ),
    ("emission_fraction", "the emission of the line divided by the emission of all the lines in the range", ""),
    ("velocity_kmps_emission_mean", "the mean radial velocity of the cells, with the emission as the weight", "km/s"),
    ("Te_emission_mean", "the mean electron temperature of the cells, with the emission as the weight", "K"),
    (
        "emission_per_velocity_erg_per_s_per_kmps",
        "the emission of the cells in the velocity bin, divided by the width of the bin",
        "erg/s/(km/s)",
    ),
    (
        "emission_per_velocity_erg_per_s_per_c",
        "the emission of the cells in the velocity bin, divided by the width of the bin",
        "erg/s/c",
    ),
    (
        "emission_fraction_per_kmps",
        (
            "the emission of the line in the velocity bin, divided by the width of the bin and by the emission in all"
            " the cells"
        ),
        "(km/s)^-1",
    ),
    (
        "emission_fraction_per_c",
        (
            "the emission of the line in the velocity bin, divided by the width of the bin and by the emission in all"
            " the cells"
        ),
        "c^-1",
    ),
    (
        "tau_sobolev_volume_mean",
        "the mean tau_sobolev of the cells in the velocity bin, with the volume as the weight",
        "",
    ),
    (
        "tau_sobolev_volume_median",
        "the median tau_sobolev of the cells in the velocity bin, with the volume as the weight",
        "",
    ),
    (
        "volume_fraction_tau_sobolev_above_1",
        "the volume of the cells with a tau_sobolev above 1 in the velocity bin, divided by the volume of the bin",
        "",
    ),
    ("velocity_c_at_tau_sobolev_max", "the radial velocity of the cell with the maximum tau_sobolev", "c"),
    (
        "velocity_emission_q(25|50|75)_kmps",
        "the radial velocity at the {0}% quantile of the emission of the line in the cells",
        "km/s",
    ),
    (
        "velocity_emission_q(25|50|75)_c",
        "the radial velocity at the {0}% quantile of the emission of the line in the cells",
        "c",
    ),
    (
        "velocity_median_tau_above_1_(min|max)_kmps",
        "the {0}imum velocity of the velocity bins where the median tau_sobolev of the line is above 1",
        "km/s",
    ),
    (
        "velocity_median_tau_above_1_(min|max)_c",
        "the {0}imum velocity of the velocity bins where the median tau_sobolev of the line is above 1",
        "c",
    ),
    ("cellcount", "the number of cells of the row, e.g. of a velocity bin", ""),
    # the trough estimate of plotlinecontributions
    (
        "trough_depth",
        "the Sobolev estimate of the fraction of the flux of the photospheric disk that the line or the blend removes",
        "",
    ),
    # the groups of plotlastpacketinteraction -groupby and the series of plotlastinteractionvelocity
    ("luminosity_fraction", "the luminosity of the group divided by the luminosity of all the groups", ""),
    (
        "velocity_luminosity_q(25|50|75)_kmps",
        "the radial velocity of the last interaction at the {0}% quantile of the packet energy of the group",
        "km/s",
    ),
    (
        "velocity_luminosity_q(25|50|75)_c",
        "the radial velocity of the last interaction at the {0}% quantile of the packet energy of the group",
        "c",
    ),
    (
        "velocity_q(25|50|75)_kmps",
        "the radial velocity of the last interaction at the {0}% quantile of the packet energy of the timestep",
        "km/s",
    ),
    (
        "velocity_q(25|50|75)_c",
        "the radial velocity of the last interaction at the {0}% quantile of the packet energy of the timestep",
        "c",
    ),
    ("velocity_q50_(min|max)_kmps", "the {0}imum median velocity of the timesteps of the series", "km/s"),
    ("velocity_q50_(min|max)_c", "the {0}imum median velocity of the timesteps of the series", "c"),
    # the logs of ARTIS
    ("mpirank", "the index of the MPI rank", ""),
    ("update_grid_s", "the time that the MPI rank took to update the grid in the timestep", "s"),
    ("update_packets_s", "the time that the MPI rank took to move the packets in the timestep", "s"),
    ("communicate_estimators_s", "the time of the MPI reduction of the estimators in the timestep", "s"),
    # the estimators
    ("Te", "the electron temperature of the cell", None),
    ("TR", "the radiation temperature of the cell, from the mean frequency of the radiation field", None),
    ("TJ", "the temperature of a black body with the mean intensity J of the cell", None),
    ("W", "the dilution factor of the radiation field of the cell", None),
    ("nne", "the number density of the free electrons in the cell", None),
    ("rho", "the density of the cell", None),
    ("volume", "the volume of the cell at the mid time of the timestep", None),
    ("beta", "the radial velocity of the centre of the cell", None),
    ("velocity_kmps", "the radial velocity of the centre of the cell", "km/s"),
    ("vel_r_mid(_kmps|_on_c)?", "the radial velocity of the centre of the cell", None),
    ("nnelement_(\\w+)", "the number density of the element {0}", None),
    ("nnion_(\\w+)", "the number density of the ion {0}", None),
    ("nniso_(\\w+)", "the number density of the nuclide {0}", None),
    ("heating_(ff|bf|coll|dep)", "the heating rate per unit volume of the process {0}", None),
    ("heating_(ff|bf|coll)_(\\w+)", "the heating rate per unit volume of the process {0} for the ion {1}", None),
    ("cooling_(ff|fb|coll|adiabatic)", "the cooling rate per unit volume of the process {0}", None),
    ("cooling_(ff|fb|coll)_(\\w+)", "the cooling rate per unit volume of the process {0} for the ion {1}", None),
    ("deposition_(\\w+)", "the energy deposition rate per unit volume of the particle type {0}", None),
    ("gamma_NT_(\\w+)", "the non-thermal ionisation rate of the ion {0}", None),
    # the frames, the series, and the residuals of each plot command
    ("time_days", "the time from the explosion", "days"),
    ("timedays", "the time of the frame from the explosion, as the title of the figure gives it", ""),
    ("timestep", "the index of the timestep, or a range of timesteps", ""),
    ("cell", "the index of the cell in the model grid", ""),
    ("modelgridindex", "the index of the cell in the model grid", ""),
    ("model", "the name of the model, or the label of the reference data", ""),
    ("modelpath", "the path of the model as the command line gives it, because two models can have one name", ""),
    ("series", "the name of the plotted series", ""),
    ("label", "the label of the plotted series or panel", ""),
    ("subplot", "the index of the subplot, from 0", ""),
    ("panel", "the index of the panel, from 0", ""),
    ("kind", "the kind of the colour image: slice, projection, or cylindrical", ""),
    ("plane", "the plane of the colour image, e.g. xy or rz", ""),
    ("points", "line for a point of the line, or markers for a point of --markers", ""),
    ("bin_centre", "false for a point that repeats the first value or the last value at an end of the line", ""),
    ("value", "the plotted value of the quantity that the row names", ""),
    ("value_(min|max)", "the {0}imum value of the cells of the row, i.e. of a bin or of a series", ""),
    ("value_mean", "the mean value of the series, with the weight that the column method names", ""),
    ("velocity_kmps_(min|max)", "the {0}imum radial velocity of the points of the series", "km/s"),
    (
        "velocity_kmps_at_value_(min|max)",
        "the radial velocity of the point with the {0}imum value of the series",
        "km/s",
    ),
    ("time_days_(min|max)", "the {0}imum time of the points of the series, from the explosion", "days"),
    ("time_days_at_value_(min|max)", "the time of the point with the {0}imum value of the series", "days"),
    ("units", "the unit of the values of the series", ""),
    ("point_count", "the number of points of the series in the selected range", ""),
    ("value_count", "the number of points of the series in the selected range with a value", ""),
    ("empty_bin_count", "the number of bins of the horizontal axis with no value", ""),
    ("mintemp_point_count", "the number of points at the temperature limit MINTEMP of ARTIS", ""),
    ("maxtemp_point_count", "the number of points at the temperature limit MAXTEMP of ARTIS", ""),
    ("method", "the method that gives the values of the row", ""),
    ("deltam15_method", "the method that gives deltam15_mag", ""),
    ("noise_method", "the formula of the relative Monte Carlo noise for the packet energies e", ""),
    ("x", "the horizontal value of the point, in the unit of the horizontal axis", ""),
    ("reference", "the label of the reference series of the residual", ""),
    (
        "residual",
        "the series minus the reference series. With -residualtype relative or relativelog, the series divided by it",
        "",
    ),
    ("npoints", "the number of points of the residual", ""),
    ("rms", "the root mean square of the series minus the reference series, in the unit of the plotted values", ""),
    ("rms_relative", "the root mean square of the residual divided by the mean absolute value of the reference", ""),
)

COMPILED_COLUMN_FAMILIES: t.Final = tuple(
    (re.compile(pattern), meaning, unit) for pattern, meaning, unit in COLUMN_FAMILIES
)


def get_column_description(column: str) -> tuple[str, str] | None:
    """Return the meaning and the unit of a column, or None if no family holds the column.

    A column with the prefix log10_ takes the description of the column without the prefix.
    """
    for pattern, meaning, unit in COMPILED_COLUMN_FAMILIES:
        if match := pattern.fullmatch(column):
            if unit is None:
                from artistools.estimators import get_units

                return meaning.format(*match.groups()), get_units(column, latex=False) or ""
            return meaning.format(*match.groups()), unit.format(*match.groups())

    if column.startswith("log10_") and (description := get_column_description(column.removeprefix("log10_"))):
        meaning, unit = description
        return f"the base-10 logarithm of {meaning}", f"log10({unit})" if unit else ""

    return None


def get_column_description_lines(columns: Sequence[str]) -> list[str]:
    """Return a line with the meaning and the unit of each column that a family holds.

    A column that no family holds gets no line, because a guess can be wrong.
    """
    lines = []
    for column in columns:
        if description := get_column_description(column):
            meaning, unit = description
            lines.append(f"column {column}: {meaning}" + (f" [{unit}]" if unit else ""))
    return lines
