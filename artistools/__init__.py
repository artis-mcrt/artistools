"""artistools.

A collection of plotting, analysis, and file format conversion tools
for the ARTIS radiative transfer code.
"""
import typing as t

from artistools import atomic
from artistools import codecomparison
from artistools import commands
from artistools import deposition
from artistools import estimators
from artistools import gsinetwork
from artistools import inputmodel
from artistools import lightcurve
from artistools import macroatom
from artistools import nltepops
from artistools import nonthermal
from artistools import packets
from artistools import plotspherical
from artistools import plottools
from artistools import radfield
from artistools import spectra
from artistools import transitions
from artistools import writecomparisondata
from artistools.__main__ import addargs
from artistools.__main__ import main
from artistools.configuration import get_config
from artistools.configuration import set_config
from artistools.estimators import read_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_cell_angle
from artistools.inputmodel import get_dfmodel_dimensions
from artistools.inputmodel import get_mean_cell_properties_of_angle_bin
from artistools.inputmodel import get_mgi_of_velocity_kms
from artistools.inputmodel import get_modeldata
from artistools.inputmodel import get_modeldata_polars
from artistools.inputmodel import get_modeldata_tuple
from artistools.inputmodel import save_initelemabundances
from artistools.inputmodel import save_modeldata
from artistools.misc import anyexist
from artistools.misc import AppendPath
from artistools.misc import average_direction_bins
from artistools.misc import CustomArgHelpFormatter
from artistools.misc import decode_roman_numeral
from artistools.misc import firstexisting
from artistools.misc import flatten_list
from artistools.misc import get_atomic_number
from artistools.misc import get_bflist
from artistools.misc import get_cellsofmpirank
from artistools.misc import get_composition_data
from artistools.misc import get_composition_data_from_outputfile
from artistools.misc import get_costheta_bins
from artistools.misc import get_costhetabin_phibin_labels
from artistools.misc import get_deposition
from artistools.misc import get_dirbin_labels
from artistools.misc import get_elsymbol
from artistools.misc import get_elsymbols_df
from artistools.misc import get_elsymbolslist
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_file_metadata
from artistools.misc import get_filterfunc
from artistools.misc import get_grid_mapping
from artistools.misc import get_inputparams
from artistools.misc import get_ion_stage_roman_numeral_df
from artistools.misc import get_ion_tuple
from artistools.misc import get_ionstring
from artistools.misc import get_linelist_dataframe
from artistools.misc import get_linelist_pldf
from artistools.misc import get_model_name
from artistools.misc import get_mpiranklist
from artistools.misc import get_mpirankofcell
from artistools.misc import get_nprocs
from artistools.misc import get_nu_grid
from artistools.misc import get_phi_bins
from artistools.misc import get_runfolders
from artistools.misc import get_syn_dir
from artistools.misc import get_time_range
from artistools.misc import get_timestep_of_timedays
from artistools.misc import get_timestep_time
from artistools.misc import get_timestep_times
from artistools.misc import get_viewingdirection_costhetabincount
from artistools.misc import get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount
from artistools.misc import get_vpkt_config
from artistools.misc import get_vspec_dir_labels
from artistools.misc import get_wid_init_at_tmin
from artistools.misc import get_wid_init_at_tmodel
from artistools.misc import get_z_a_nucname
from artistools.misc import linetuple
from artistools.misc import makelist
from artistools.misc import match_closest_time
from artistools.misc import merge_pdf_files
from artistools.misc import namedtuple
from artistools.misc import parse_range
from artistools.misc import parse_range_list
from artistools.misc import read_linestatfile
from artistools.misc import readnoncommentline
from artistools.misc import roman_numerals
from artistools.misc import set_args_from_dict
from artistools.misc import showtimesteptimes
from artistools.misc import split_dataframe_dirbins
from artistools.misc import stripallsuffixes
from artistools.misc import trim_or_pad
from artistools.misc import vec_len
from artistools.misc import zopen
from artistools.misc import zopenpl
from artistools.plottools import set_mpl_style


def get_path(**kwargs: t.Any) -> None:
    print(get_config("path_artistools_dir"))


set_mpl_style()
