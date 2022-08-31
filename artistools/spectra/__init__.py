#!/usr/bin/env python3
"""Artistools - spectra related functions."""
from artistools.spectra.plotspectra import addargs
from artistools.spectra.plotspectra import main
from artistools.spectra.plotspectra import main as plot
from artistools.spectra.spectra import average_angle_bins
from artistools.spectra.spectra import get_exspec_bins
from artistools.spectra.spectra import get_flux_contributions
from artistools.spectra.spectra import get_flux_contributions_from_packets
from artistools.spectra.spectra import get_line_flux
from artistools.spectra.spectra import get_reference_spectrum
from artistools.spectra.spectra import get_res_spectrum
from artistools.spectra.spectra import get_specpol_data
from artistools.spectra.spectra import get_spectrum
from artistools.spectra.spectra import get_spectrum_at_time
from artistools.spectra.spectra import get_spectrum_from_packets
from artistools.spectra.spectra import get_spectrum_from_packets_worker
from artistools.spectra.spectra import get_vspecpol_spectrum
from artistools.spectra.spectra import make_averaged_vspecfiles
from artistools.spectra.spectra import make_virtual_spectra_summed_file
from artistools.spectra.spectra import print_floers_line_ratio
from artistools.spectra.spectra import print_integrated_flux
from artistools.spectra.spectra import read_specpol_res
from artistools.spectra.spectra import sort_and_reduce_flux_contribution_list
from artistools.spectra.spectra import stackspectra
from artistools.spectra.spectra import timeshift_fluxscale_co56law
from artistools.spectra.spectra import write_flambda_spectra
