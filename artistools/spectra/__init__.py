#!/usr/bin/env python3
"""Artistools - spectra related functions."""

from artistools.spectra.spectra import (
    average_angle_bins,
    get_exspec_bins,
    get_flux_contributions,
    get_line_flux,
    get_reference_spectrum,
    get_res_spectrum,
    get_specpol_data,
    get_spectrum,
    get_spectrum_at_time,
    get_spectrum_from_packets,
    get_spectrum_from_packets_worker,
    get_vspecpol_spectrum,
    make_averaged_vspecfiles,
    make_virtual_spectra_summed_file,
    print_floers_line_ratio,
    print_integrated_flux,
    read_specpol_res,
    sort_and_reduce_flux_contribution_list,
    stackspectra,
    timeshift_fluxscale_co56law,
    write_flambda_spectra
)

from artistools.spectra.plotspectra import main, addargs
from artistools.spectra.plotspectra import main as plot
