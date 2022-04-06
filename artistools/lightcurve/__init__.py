#!/usr/bin/env python3
"""Artistools - light curve functions."""

from artistools.lightcurve.lightcurve import (
    bolometric_magnitude,
    evaluate_magnitudes,
    generate_band_lightcurve_data,
    get_band_lightcurve,
    get_colour_delta_mag,
    get_filter_data,
    get_from_packets,
    get_phillips_relation_data,
    get_sn_sample_bol,
    get_spectrum_in_filter_range,
    plot_phillips_relation_data,
    read_3d_gammalightcurve,
    read_bol_reflightcurve_data,
    read_hesma_lightcurve,
    read_reflightcurve_band_data,
    readfile,
)

from artistools.lightcurve.viewingangleanalysis import (
    calculate_costheta_phi_for_viewing_angles,
    calculate_peak_time_mag_deltam15,
    get_angle_stuff,
    get_viewinganglebin_definitions,
    lightcurve_polyfit,
    make_peak_colour_viewing_angle_plot,
    make_plot_test_viewing_angle_fit,
    make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot,
    peakmag_risetime_declinerate_init,
    plot_viewanglebrightness_at_fixed_time,
    save_viewing_angle_data_for_plotting,
    second_band_brightness_at_peak_first_band,
    set_scatterplot_plot_params,
    set_scatterplot_plotkwargs,
    update_plotkwargs_for_viewingangle_colorbar,
    write_viewing_angle_data
)

from artistools.lightcurve.plotlightcurve import main, addargs
from artistools.lightcurve.plotlightcurve import main as plot
