"""Artistools - spectra related functions."""

__all__ = ["plot"]

# the core module comes first. A module of a different package imports these names from this package, and
# the plot modules below import such modules, thus a name must exist before a cycle comes back here
from artistools.spectra.core import convert_angstroms_to_unit as convert_angstroms_to_unit
from artistools.spectra.core import convert_unit_to_angstroms as convert_unit_to_angstroms
from artistools.spectra.core import get_escape_surface_gamma as get_escape_surface_gamma
from artistools.spectra.core import get_flux_contributions as get_flux_contributions
from artistools.spectra.core import get_flux_contributions_from_packets as get_flux_contributions_from_packets
from artistools.spectra.core import get_from_packets as get_from_packets
from artistools.spectra.core import get_specpol_data as get_specpol_data
from artistools.spectra.core import get_spectra as get_spectra
from artistools.spectra.core import get_spectrum_at_time as get_spectrum_at_time
from artistools.spectra.core import get_vspecpol_data as get_vspecpol_data
from artistools.spectra.core import make_virtual_spectra_summed_file as make_virtual_spectra_summed_file
from artistools.spectra.core import read_spec as read_spec
from artistools.spectra.core import read_spec_res as read_spec_res
from artistools.spectra.core import scan_spec as scan_spec
from artistools.spectra.core import scan_spec_res as scan_spec_res
from artistools.spectra.core import scan_specpol_data as scan_specpol_data
from artistools.spectra.core import scan_spectra as scan_spectra
from artistools.spectra.core import scan_spectra_from_packets as scan_spectra_from_packets
from artistools.spectra.core import scan_spectrum_at_time as scan_spectrum_at_time
from artistools.spectra.core import scan_vspecpol_data as scan_vspecpol_data

# isort: split
from artistools.spectra import plotspectra as plotspectra
from artistools.spectra import writespectra as writespectra
from artistools.spectra.plotspectra import main as plot
