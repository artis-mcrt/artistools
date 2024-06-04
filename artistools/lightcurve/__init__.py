"""Artistools - light curve functions."""

__all__ = ["plot", "plotlightcurve"]

from artistools.lightcurve import plotlightcurve
from artistools.lightcurve.lightcurve import bolometric_magnitude as bolometric_magnitude
from artistools.lightcurve.lightcurve import evaluate_magnitudes as evaluate_magnitudes
from artistools.lightcurve.lightcurve import generate_band_lightcurve_data as generate_band_lightcurve_data
from artistools.lightcurve.lightcurve import get_band_lightcurve as get_band_lightcurve
from artistools.lightcurve.lightcurve import get_colour_delta_mag as get_colour_delta_mag
from artistools.lightcurve.lightcurve import get_filter_data as get_filter_data
from artistools.lightcurve.lightcurve import get_from_packets as get_from_packets
from artistools.lightcurve.lightcurve import get_phillips_relation_data as get_phillips_relation_data
from artistools.lightcurve.lightcurve import get_sn_sample_bol as get_sn_sample_bol
from artistools.lightcurve.lightcurve import get_spectrum_in_filter_range as get_spectrum_in_filter_range
from artistools.lightcurve.lightcurve import plot_phillips_relation_data as plot_phillips_relation_data
from artistools.lightcurve.lightcurve import read_3d_gammalightcurve as read_3d_gammalightcurve
from artistools.lightcurve.lightcurve import read_bol_reflightcurve_data as read_bol_reflightcurve_data
from artistools.lightcurve.lightcurve import read_hesma_lightcurve as read_hesma_lightcurve
from artistools.lightcurve.lightcurve import read_reflightcurve_band_data as read_reflightcurve_band_data
from artistools.lightcurve.lightcurve import readfile as readfile
from artistools.lightcurve.plotlightcurve import addargs as addargs
from artistools.lightcurve.plotlightcurve import main as plot
from artistools.lightcurve.viewingangleanalysis import (
    calculate_peak_time_mag_deltam15 as calculate_peak_time_mag_deltam15,
)
from artistools.lightcurve.viewingangleanalysis import lightcurve_polyfit as lightcurve_polyfit
from artistools.lightcurve.viewingangleanalysis import (
    make_peak_colour_viewing_angle_plot as make_peak_colour_viewing_angle_plot,
)
from artistools.lightcurve.viewingangleanalysis import (
    make_plot_test_viewing_angle_fit as make_plot_test_viewing_angle_fit,
)
from artistools.lightcurve.viewingangleanalysis import (
    make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot as make_viewing_angle_risetime_peakmag_delta_m15_scatter_plot,
)
from artistools.lightcurve.viewingangleanalysis import parse_directionbin_args as parse_directionbin_args
from artistools.lightcurve.viewingangleanalysis import (
    peakmag_risetime_declinerate_init as peakmag_risetime_declinerate_init,
)
from artistools.lightcurve.viewingangleanalysis import (
    plot_viewanglebrightness_at_fixed_time as plot_viewanglebrightness_at_fixed_time,
)
from artistools.lightcurve.viewingangleanalysis import (
    save_viewing_angle_data_for_plotting as save_viewing_angle_data_for_plotting,
)
from artistools.lightcurve.viewingangleanalysis import (
    second_band_brightness_at_peak_first_band as second_band_brightness_at_peak_first_band,
)
from artistools.lightcurve.viewingangleanalysis import set_scatterplot_plot_params as set_scatterplot_plot_params
from artistools.lightcurve.viewingangleanalysis import set_scatterplot_plotkwargs as set_scatterplot_plotkwargs
from artistools.lightcurve.viewingangleanalysis import (
    update_plotkwargs_for_viewingangle_colorbar as update_plotkwargs_for_viewingangle_colorbar,
)
from artistools.lightcurve.viewingangleanalysis import write_viewing_angle_data as write_viewing_angle_data
from artistools.lightcurve.writebollightcurvedata import get_bol_lc_from_lightcurveout as get_bol_lc_from_lightcurveout
from artistools.lightcurve.writebollightcurvedata import get_bol_lc_from_spec as get_bol_lc_from_spec
