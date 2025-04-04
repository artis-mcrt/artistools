"""artistools.

A collection of plotting, analysis, and file format conversion tools
for the ARTIS radiative transfer code.
"""

import typing as t

from artistools import atomic as atomic
from artistools import codecomparison as codecomparison
from artistools import commands as commands
from artistools import estimators as estimators
from artistools import gsinetwork as gsinetwork
from artistools import inputmodel as inputmodel
from artistools import lightcurve as lightcurve
from artistools import macroatom as macroatom
from artistools import misc as misc
from artistools import nltepops as nltepops
from artistools import nonthermal as nonthermal
from artistools import packets as packets
from artistools import plotspherical as plotspherical
from artistools import plottools as plottools
from artistools import radfield as radfield
from artistools import spectra as spectra
from artistools import transitions as transitions
from artistools import writecomparisondata as writecomparisondata
from artistools.commands import addargs as addargs
from artistools.configuration import get_config as get_config
from artistools.configuration import set_config as set_config
from artistools.estimators import read_estimators as read_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata as add_derived_cols_to_modeldata
from artistools.inputmodel import get_cell_angle as get_cell_angle
from artistools.inputmodel import get_dfmodel_dimensions as get_dfmodel_dimensions
from artistools.inputmodel import get_mean_cell_properties_of_angle_bin as get_mean_cell_properties_of_angle_bin
from artistools.inputmodel import get_mgi_of_velocity_kms as get_mgi_of_velocity_kms
from artistools.inputmodel import get_modeldata as get_modeldata
from artistools.inputmodel import get_modeldata_pandas as get_modeldata_pandas
from artistools.inputmodel import get_modeldata_tuple as get_modeldata_tuple
from artistools.inputmodel import save_initelemabundances as save_initelemabundances
from artistools.inputmodel import save_modeldata as save_modeldata
from artistools.misc import anyexist as anyexist
from artistools.misc import AppendPath as AppendPath
from artistools.misc import average_direction_bins as average_direction_bins
from artistools.misc import CustomArgHelpFormatter as CustomArgHelpFormatter
from artistools.misc import decode_roman_numeral as decode_roman_numeral
from artistools.misc import firstexisting as firstexisting
from artistools.misc import flatten_list as flatten_list
from artistools.misc import get_atomic_number as get_atomic_number
from artistools.misc import get_bflist as get_bflist
from artistools.misc import get_cellsofmpirank as get_cellsofmpirank
from artistools.misc import get_composition_data as get_composition_data
from artistools.misc import get_composition_data_from_outputfile as get_composition_data_from_outputfile
from artistools.misc import get_costheta_bins as get_costheta_bins
from artistools.misc import get_costhetabin_phibin_labels as get_costhetabin_phibin_labels
from artistools.misc import get_deposition as get_deposition
from artistools.misc import get_dirbin_labels as get_dirbin_labels
from artistools.misc import get_elsymbol as get_elsymbol
from artistools.misc import get_elsymbols_df as get_elsymbols_df
from artistools.misc import get_elsymbolslist as get_elsymbolslist
from artistools.misc import get_escaped_arrivalrange as get_escaped_arrivalrange
from artistools.misc import get_file_metadata as get_file_metadata
from artistools.misc import get_filterfunc as get_filterfunc
from artistools.misc import get_grid_mapping as get_grid_mapping
from artistools.misc import get_inputparams as get_inputparams
from artistools.misc import get_ion_stage_roman_numeral_df as get_ion_stage_roman_numeral_df
from artistools.misc import get_ion_tuple as get_ion_tuple
from artistools.misc import get_ionstring as get_ionstring
from artistools.misc import get_linelist_dataframe as get_linelist_dataframe
from artistools.misc import get_linelist_pldf as get_linelist_pldf
from artistools.misc import get_model_name as get_model_name
from artistools.misc import get_mpiranklist as get_mpiranklist
from artistools.misc import get_mpirankofcell as get_mpirankofcell
from artistools.misc import get_multiprocessing_pool as get_multiprocessing_pool
from artistools.misc import get_nprocs as get_nprocs
from artistools.misc import get_nu_grid as get_nu_grid
from artistools.misc import get_nuclides as get_nuclides
from artistools.misc import get_phi_bins as get_phi_bins
from artistools.misc import get_runfolders as get_runfolders
from artistools.misc import get_time_range as get_time_range
from artistools.misc import get_timestep_of_timedays as get_timestep_of_timedays
from artistools.misc import get_timestep_time as get_timestep_time
from artistools.misc import get_timestep_times as get_timestep_times
from artistools.misc import get_viewingdirection_costhetabincount as get_viewingdirection_costhetabincount
from artistools.misc import get_viewingdirection_phibincount as get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount as get_viewingdirectionbincount
from artistools.misc import get_vpkt_config as get_vpkt_config
from artistools.misc import get_vspec_dir_labels as get_vspec_dir_labels
from artistools.misc import get_wid_init_at_tmodel as get_wid_init_at_tmodel
from artistools.misc import get_z_a_nucname as get_z_a_nucname
from artistools.misc import LineTuple as LineTuple
from artistools.misc import makelist as makelist
from artistools.misc import match_closest_time as match_closest_time
from artistools.misc import merge_pdf_files as merge_pdf_files
from artistools.misc import parse_range as parse_range
from artistools.misc import parse_range_list as parse_range_list
from artistools.misc import print_theta_phi_definitions as print_theta_phi_definitions
from artistools.misc import read_linestatfile as read_linestatfile
from artistools.misc import readnoncommentline as readnoncommentline
from artistools.misc import roman_numerals as roman_numerals
from artistools.misc import set_args_from_dict as set_args_from_dict
from artistools.misc import showtimesteptimes as showtimesteptimes
from artistools.misc import split_multitable_dataframe as split_multitable_dataframe
from artistools.misc import stripallsuffixes as stripallsuffixes
from artistools.misc import trim_or_pad as trim_or_pad
from artistools.misc import vec_len as vec_len
from artistools.misc import zopen as zopen
from artistools.misc import zopenpl as zopenpl
from artistools.plottools import set_mpl_style as set_mpl_style


def get_path(**kwargs: t.Any) -> None:  # noqa: ARG001
    print(get_config("path_artistools_dir"))


set_mpl_style()
