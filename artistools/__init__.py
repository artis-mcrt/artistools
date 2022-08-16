#!/usr/bin/env python3
"""Artistools.

A collection of plotting, analysis, and file format conversion tools
for the ARTIS radiative transfer code.
"""

from artistools.configuration import config

from artistools.inputmodel import (
    add_derived_cols_to_modeldata,
    get_modeldata,
    get_2d_modeldata,
    get_cell_angle,
    get_dfmodel_dimensions,
    get_mean_cell_properties_of_angle_bin,
    get_mgi_of_velocity_kms,
    save_initialabundances,
    save_modeldata,
)

from artistools.misc import (
    AppendPath,
    CustomArgHelpFormatter,
    cross_prod,
    decode_roman_numeral,
    diskcache,
    dot,
    firstexisting,
    flatten_list,
    gather_res_data,
    get_artis_constants,
    get_atomic_number,
    get_cellsofmpirank,
    get_composition_data,
    get_composition_data_from_outputfile,
    get_deposition,
    get_escaped_arrivalrange,
    get_elsymbol,
    get_elsymbolslist,
    get_filterfunc,
    get_grid_mapping,
    get_model_name,
    get_inputparams,
    get_ionstring,
    get_linelist,
    get_mpiranklist,
    get_mpirankofcell,
    get_runfolders,
    get_syn_dir,
    get_time_range,
    get_timestep_of_timedays,
    get_timestep_time,
    get_timestep_times_float,
    get_vpkt_config,
    get_wid_init_at_tmin,
    get_wid_init_at_tmodel,
    get_z_a_nucname,
    join_pdf_files,
    make_namedtuple,
    makelist,
    match_closest_time,
    namedtuple,
    parse_cdefines,
    parse_range,
    parse_range_list,
    readnoncommentline,
    roman_numerals,
    showtimesteptimes,
    stripallsuffixes,
    trim_or_pad,
    vec_len,
    zopen,
)

import artistools.atomic
import artistools.codecomparison
import artistools.commands
import artistools.deposition
import artistools.estimators
import artistools.inputmodel
import artistools.lightcurve
import artistools.macroatom
import artistools.nltepops
import artistools.nonthermal
import artistools.packets
import artistools.radfield
import artistools.spectra
import artistools.transitions
# import artistools.plottools

from artistools.__main__ import main, addargs
