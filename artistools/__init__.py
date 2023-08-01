"""artistools.

A collection of plotting, analysis, and file format conversion tools
for the ARTIS radiative transfer code.
"""

import typing as t

from . import atomic
from . import codecomparison
from . import commands
from . import deposition
from . import estimators
from . import initial_composition
from . import inputmodel
from . import lightcurve
from . import macroatom
from . import nltepops
from . import nonthermal
from . import packets
from . import plotspherical
from . import radfield
from . import spectra
from . import transitions
from . import writecomparisondata
from .__main__ import addargs
from .__main__ import main
from .configuration import get_config
from .configuration import set_config
from .inputmodel import add_derived_cols_to_modeldata
from .inputmodel import get_2d_modeldata
from .inputmodel import get_cell_angle
from .inputmodel import get_dfmodel_dimensions
from .inputmodel import get_mean_cell_properties_of_angle_bin
from .inputmodel import get_mgi_of_velocity_kms
from .inputmodel import get_modeldata
from .inputmodel import get_modeldata_tuple
from .inputmodel import save_initelemabundances
from .inputmodel import save_modeldata
from .misc import anyexist
from .misc import AppendPath
from .misc import average_direction_bins
from .misc import CustomArgHelpFormatter
from .misc import decode_roman_numeral
from .misc import firstexisting
from .misc import flatten_list
from .misc import get_atomic_number
from .misc import get_bflist
from .misc import get_cellsofmpirank
from .misc import get_composition_data
from .misc import get_composition_data_from_outputfile
from .misc import get_costheta_bins
from .misc import get_costhetabin_phibin_labels
from .misc import get_deposition
from .misc import get_dirbin_labels
from .misc import get_elsymbol
from .misc import get_elsymbolslist
from .misc import get_escaped_arrivalrange
from .misc import get_file_metadata
from .misc import get_filterfunc
from .misc import get_grid_mapping
from .misc import get_inputparams
from .misc import get_ionstring
from .misc import get_linelist_dataframe
from .misc import get_linelist_dict
from .misc import get_linelist_pldf
from .misc import get_model_name
from .misc import get_mpiranklist
from .misc import get_mpirankofcell
from .misc import get_nprocs
from .misc import get_nu_grid
from .misc import get_phi_bins
from .misc import get_runfolders
from .misc import get_syn_dir
from .misc import get_time_range
from .misc import get_timestep_of_timedays
from .misc import get_timestep_time
from .misc import get_timestep_times
from .misc import get_viewingdirection_costhetabincount
from .misc import get_viewingdirection_phibincount
from .misc import get_viewingdirectionbincount
from .misc import get_vpkt_config
from .misc import get_vspec_dir_labels
from .misc import get_wid_init_at_tmin
from .misc import get_wid_init_at_tmodel
from .misc import get_z_a_nucname
from .misc import join_pdf_files
from .misc import linetuple
from .misc import makelist
from .misc import match_closest_time
from .misc import namedtuple
from .misc import parse_range
from .misc import parse_range_list
from .misc import read_linestatfile
from .misc import readnoncommentline
from .misc import roman_numerals
from .misc import showtimesteptimes
from .misc import split_dataframe_dirbins
from .misc import stripallsuffixes
from .misc import trim_or_pad
from .misc import vec_len
from .misc import zopen
from .plottools import set_mpl_style


def get_path(**kwargs: t.Any) -> None:
    print(get_config("path_artistools_dir"))


set_mpl_style()
