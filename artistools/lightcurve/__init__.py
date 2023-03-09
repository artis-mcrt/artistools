#!/usr/bin/env python3
"""Artistools - light curve functions."""
from artistools.lightcurve.lightcurve import average_lightcurve_phi_bins
from artistools.lightcurve.lightcurve import bolometric_magnitude
from artistools.lightcurve.lightcurve import evaluate_magnitudes
from artistools.lightcurve.lightcurve import generate_band_lightcurve_data
from artistools.lightcurve.lightcurve import get_band_lightcurve
from artistools.lightcurve.lightcurve import get_colour_delta_mag
from artistools.lightcurve.lightcurve import get_filter_data
from artistools.lightcurve.lightcurve import get_from_packets
from artistools.lightcurve.lightcurve import get_phillips_relation_data
from artistools.lightcurve.lightcurve import get_sn_sample_bol
from artistools.lightcurve.lightcurve import get_spectrum_in_filter_range
from artistools.lightcurve.lightcurve import plot_phillips_relation_data
from artistools.lightcurve.lightcurve import read_3d_gammalightcurve
from artistools.lightcurve.lightcurve import read_bol_reflightcurve_data
from artistools.lightcurve.lightcurve import read_hesma_lightcurve
from artistools.lightcurve.lightcurve import read_reflightcurve_band_data
from artistools.lightcurve.lightcurve import readfile
from artistools.lightcurve.plotlightcurve import addargs
from artistools.lightcurve.plotlightcurve import main
from artistools.lightcurve.plotlightcurve import main as plot
from artistools.lightcurve.viewingangleanalysis import calculate_costheta_phi_for_viewing_angles
from artistools.lightcurve.viewingangleanalysis import calculate_peak_time_mag_deltam15
from artistools.lightcurve.viewingangleanalysis import get_angle_stuff
from artistools.lightcurve.viewingangleanalysis import lightcurve_polyfit
from artistools.lightcurve.viewingangleanalysis import make_peak_colour_viewing_angle_plot
from artistools.lightcurve.viewingangleanalysis import make_plot_test_viewing_angle_fit
from artistools.lightcurve.viewingangleanalysis import make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot
from artistools.lightcurve.viewingangleanalysis import peakmag_risetime_declinerate_init
from artistools.lightcurve.viewingangleanalysis import plot_viewanglebrightness_at_fixed_time
from artistools.lightcurve.viewingangleanalysis import save_viewing_angle_data_for_plotting
from artistools.lightcurve.viewingangleanalysis import second_band_brightness_at_peak_first_band
from artistools.lightcurve.viewingangleanalysis import set_scatterplot_plot_params
from artistools.lightcurve.viewingangleanalysis import set_scatterplot_plotkwargs
from artistools.lightcurve.viewingangleanalysis import update_plotkwargs_for_viewingangle_colorbar
from artistools.lightcurve.viewingangleanalysis import write_viewing_angle_data
