"""Artistools - spectra related functions."""

__all__ = ["plot"]

from artistools.spectra import plotspectra as plotspectra
from artistools.spectra import writespectra as writespectra
from artistools.spectra.plotspectra import main as plot
from artistools.spectra.spectra import bin_spectrum as bin_spectrum
from artistools.spectra.spectra import convert_angstroms_to_unit as convert_angstroms_to_unit
from artistools.spectra.spectra import convert_unit_to_angstroms as convert_unit_to_angstroms
from artistools.spectra.spectra import get_default_losvelocity_shells as get_default_losvelocity_shells
from artistools.spectra.spectra import get_default_velocity_shells as get_default_velocity_shells
from artistools.spectra.spectra import get_flux_contributions as get_flux_contributions
from artistools.spectra.spectra import get_flux_contributions_from_packets as get_flux_contributions_from_packets
from artistools.spectra.spectra import get_from_packets as get_from_packets
from artistools.spectra.spectra import get_lambda_bin_edges as get_lambda_bin_edges
from artistools.spectra.spectra import get_shell_labels as get_shell_labels
from artistools.spectra.spectra import get_spectra as get_spectra
from artistools.spectra.spectra import make_virtual_spectra_summed_file as make_virtual_spectra_summed_file
from artistools.spectra.spectra import parse_velocity_argument as parse_velocity_argument
