"""Artistools - light curve functions."""

__all__ = ["plot", "plotlightcurve"]

# the core module comes first. A module of a different package imports these names from this package, and
# the plot modules below import such modules, thus a name must exist before a cycle comes back here
from artistools.lightcurve.core import bracket_spectrum_to_band as bracket_spectrum_to_band
from artistools.lightcurve.core import find_bol_reflightcurve_file as find_bol_reflightcurve_file
from artistools.lightcurve.core import find_lightcurve_file as find_lightcurve_file
from artistools.lightcurve.core import generate_band_lightcurve_data as generate_band_lightcurve_data
from artistools.lightcurve.core import get_band_lightcurve as get_band_lightcurve
from artistools.lightcurve.core import get_colour_delta_mag as get_colour_delta_mag
from artistools.lightcurve.core import get_filter_data as get_filter_data
from artistools.lightcurve.core import get_from_packets as get_from_packets
from artistools.lightcurve.core import luminosity_distance as luminosity_distance
from artistools.lightcurve.core import path_is_reference_lightcurve as path_is_reference_lightcurve
from artistools.lightcurve.core import read_bol_reflightcurve_data as read_bol_reflightcurve_data
from artistools.lightcurve.core import read_hesma_lightcurve_file as read_hesma_lightcurve_file
from artistools.lightcurve.core import scan_lightcurve as scan_lightcurve

# isort: split
from artistools.lightcurve import plotlightcurve
from artistools.lightcurve import viewingangleanalysis as viewingangleanalysis
from artistools.lightcurve.plotlightcurve import addargs as addargs
from artistools.lightcurve.plotlightcurve import main as plot
