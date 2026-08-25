"""artistools.

A collection of plotting, analysis, and file format conversion tools
for the ARTIS radiative transfer code.
"""

# ruff:file-ignore[non-empty-init-module]
import sys

if sys.version_info >= (3, 15) and hasattr(sys, "set_lazy_imports_filter") and hasattr(sys, "set_lazy_imports"):
    sys.set_lazy_imports_filter(
        lambda _importing, imported, _fromlist: (
            not imported.startswith(("matplotlib.", "numpy", "polars", "polars.exceptions", "polars.selectors"))
        )
    )
    sys.set_lazy_imports("all")

    # numpy has to reach sys.modules before anything imports polars. polars substitutes its own lazy proxy for
    # numpy whenever numpy is absent at that moment, and on free-threaded 3.15 that proxy cannot resolve itself:
    # polars._dependencies captures globals() to publish the real module, which raises there with
    # "'module' object does not support item assignment".
    import numpy as np  # ruff:ignore[unused-import]

if sys.version_info >= (3, 15):
    from artistools._polarscompat import repair_series_expr_dispatch

    repair_series_expr_dispatch()

from artistools import atomic as atomic
from artistools import codecomparison as codecomparison
from artistools import commands as commands
from artistools import constants as constants
from artistools import ejectaopacity as ejectaopacity
from artistools import estimators as estimators
from artistools import gsinetwork as gsinetwork
from artistools import hesma_scripts as hesma_scripts
from artistools import inputmodel as inputmodel
from artistools import lightcurve as lightcurve
from artistools import linefluxes as linefluxes
from artistools import logfiles as logfiles
from artistools import macroatom as macroatom
from artistools import make_vpkt_input as make_vpkt_input
from artistools import misc as misc
from artistools import nltepops as nltepops
from artistools import nonthermal as nonthermal
from artistools import packets as packets
from artistools import plotspherical as plotspherical
from artistools import plottools as plottools
from artistools import radfield as radfield
from artistools import rustext as rustext
from artistools import spectra as spectra
from artistools import transitions as transitions
from artistools import viewing_angles_visualization as viewing_angles_visualization
from artistools import writecomparisondata as writecomparisondata
from artistools.atomic import decode_roman_numeral as decode_roman_numeral
from artistools.atomic import get_atomic_masses as get_atomic_masses
from artistools.atomic import get_atomic_number as get_atomic_number
from artistools.atomic import get_bflist as get_bflist
from artistools.atomic import get_composition_data as get_composition_data
from artistools.atomic import get_composition_data_from_outputfile as get_composition_data_from_outputfile
from artistools.atomic import get_elsymbol as get_elsymbol
from artistools.atomic import get_elsymbols_df as get_elsymbols_df
from artistools.atomic import get_elsymbolslist as get_elsymbolslist
from artistools.atomic import get_ion_tuple as get_ion_tuple
from artistools.atomic import get_ionstring as get_ionstring
from artistools.atomic import get_linelist_pldf as get_linelist_pldf
from artistools.atomic import get_nuclides as get_nuclides
from artistools.atomic import get_z_a_nucname as get_z_a_nucname
from artistools.atomic import roman_numerals as roman_numerals
from artistools.commands import addargs as addargs
from artistools.commands import CustomArgHelpFormatter as CustomArgHelpFormatter
from artistools.commands import get_path as get_path
from artistools.estimators import read_estimators as read_estimators
from artistools.estimators import scan_estimators as scan_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata as add_derived_cols_to_modeldata
from artistools.inputmodel import get_cell_angle as get_cell_angle
from artistools.inputmodel import get_mgi_of_velocity_kms as get_mgi_of_velocity_kms
from artistools.inputmodel import get_modeldata as get_modeldata
from artistools.inputmodel import save_initelemabundances as save_initelemabundances
from artistools.inputmodel import save_modeldata as save_modeldata
from artistools.misc import addarg_action as addarg_action
from artistools.misc import addarg_axislimits as addarg_axislimits
from artistools.misc import addarg_dpi as addarg_dpi
from artistools.misc import addarg_figscale as addarg_figscale
from artistools.misc import addarg_filter as addarg_filter
from artistools.misc import addarg_maxpacketfiles as addarg_maxpacketfiles
from artistools.misc import addarg_modelgridindex as addarg_modelgridindex
from artistools.misc import addarg_modelpath as addarg_modelpath
from artistools.misc import addarg_nolegend as addarg_nolegend
from artistools.misc import addarg_notitle as addarg_notitle
from artistools.misc import addarg_outputfile as addarg_outputfile
from artistools.misc import addarg_outputpath as addarg_outputpath
from artistools.misc import addarg_seriesstyle as addarg_seriesstyle
from artistools.misc import addarg_show as addarg_show
from artistools.misc import addarg_timedays as addarg_timedays
from artistools.misc import addarg_timeminmax as addarg_timeminmax
from artistools.misc import addarg_timestep as addarg_timestep
from artistools.misc import addarg_viewingangle as addarg_viewingangle
from artistools.misc import addarg_yscale as addarg_yscale
from artistools.misc import average_direction_bins as average_direction_bins
from artistools.misc import check_averaging_angles as check_averaging_angles
from artistools.misc import drop_trailing_null_column as drop_trailing_null_column
from artistools.misc import exit_with_error as exit_with_error
from artistools.misc import firstexisting as firstexisting
from artistools.misc import firstexisting_or_none as firstexisting_or_none
from artistools.misc import flatten_list as flatten_list
from artistools.misc import gaussian_filter_wrap as gaussian_filter_wrap
from artistools.misc import get_cellsofmpirank as get_cellsofmpirank
from artistools.misc import get_costheta_bins as get_costheta_bins
from artistools.misc import get_costhetabin_phibin_labels as get_costhetabin_phibin_labels
from artistools.misc import get_deposition as get_deposition
from artistools.misc import get_dirbin_labels as get_dirbin_labels
from artistools.misc import get_dirbins as get_dirbins
from artistools.misc import get_escaped_arrivalrange as get_escaped_arrivalrange
from artistools.misc import get_file_identity as get_file_identity
from artistools.misc import get_file_metadata as get_file_metadata
from artistools.misc import get_filterfunc as get_filterfunc
from artistools.misc import get_grid_mapping as get_grid_mapping
from artistools.misc import get_inputparams as get_inputparams
from artistools.misc import get_model_name as get_model_name
from artistools.misc import get_mpiranklist as get_mpiranklist
from artistools.misc import get_mpirankofcell as get_mpirankofcell
from artistools.misc import get_nprocs as get_nprocs
from artistools.misc import get_nu_grid as get_nu_grid
from artistools.misc import get_phi_bins as get_phi_bins
from artistools.misc import get_runfolders as get_runfolders
from artistools.misc import get_series_label as get_series_label
from artistools.misc import get_time_range as get_time_range
from artistools.misc import get_timestep_of_timedays as get_timestep_of_timedays
from artistools.misc import get_timestep_time as get_timestep_time
from artistools.misc import get_timestep_times as get_timestep_times
from artistools.misc import get_timesteps as get_timesteps
from artistools.misc import get_viewingdirection_costhetabincount as get_viewingdirection_costhetabincount
from artistools.misc import get_viewingdirection_phibincount as get_viewingdirection_phibincount
from artistools.misc import get_viewingdirectionbincount as get_viewingdirectionbincount
from artistools.misc import get_vpkt_config as get_vpkt_config
from artistools.misc import get_vspec_dir_labels as get_vspec_dir_labels
from artistools.misc import get_wid_init_at_tmodel as get_wid_init_at_tmodel
from artistools.misc import makelist as makelist
from artistools.misc import match_closest_time as match_closest_time
from artistools.misc import merge_pdf_files as merge_pdf_files
from artistools.misc import normalize_path_list as normalize_path_list
from artistools.misc import parallel_map as parallel_map
from artistools.misc import parse_cli_args as parse_cli_args
from artistools.misc import parse_range as parse_range
from artistools.misc import parse_range_list as parse_range_list
from artistools.misc import path_is_artis_model as path_is_artis_model
from artistools.misc import path_is_codecomparison as path_is_codecomparison
from artistools.misc import polars_source as polars_source
from artistools.misc import print_saved as print_saved
from artistools.misc import print_theta_phi_definitions as print_theta_phi_definitions
from artistools.misc import read_rank_outputfiles as read_rank_outputfiles
from artistools.misc import read_wsv as read_wsv
from artistools.misc import readnoncommentline as readnoncommentline
from artistools.misc import require_action as require_action
from artistools.misc import resolve_outputfile as resolve_outputfile
from artistools.misc import resolve_yscale as resolve_yscale
from artistools.misc import savgol_filter as savgol_filter
from artistools.misc import set_args_from_dict as set_args_from_dict
from artistools.misc import split_multitable_dataframe as split_multitable_dataframe
from artistools.misc import stripallsuffixes as stripallsuffixes
from artistools.misc import trim_or_pad as trim_or_pad
from artistools.misc import vec_len as vec_len
from artistools.misc import write_gif as write_gif
from artistools.misc import write_parquet_atomic as write_parquet_atomic
from artistools.misc import zopen as zopen
from artistools.misc import zopenpl as zopenpl
from artistools.plottools import set_mpl_style as set_mpl_style
