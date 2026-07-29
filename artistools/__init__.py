"""artistools.

A collection of plotting, analysis, and file format conversion tools
for the ARTIS radiative transfer code.
"""

# ruff:file-ignore[non-empty-init-module]
import sys
import typing as t

if sys.version_info >= (3, 15) and hasattr(sys, "set_lazy_imports_filter") and hasattr(sys, "set_lazy_imports"):
    sys.set_lazy_imports_filter(
        lambda _importing, imported, _fromlist: (
            not imported.startswith((
                "matplotlib.",
                "pandas._libs",
                "pandas.core",
                "polars",
                "polars.exceptions",
                "polars.selectors",
            ))
        )
    )
    sys.set_lazy_imports("all")

if t.TYPE_CHECKING:
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
    from artistools import version as version
    from artistools import viewing_angles_visualization as viewing_angles_visualization
    from artistools import writecomparisondata as writecomparisondata
    from artistools.atomic import decode_roman_numeral as decode_roman_numeral
    from artistools.atomic import get_atomic_number as get_atomic_number
    from artistools.atomic import get_bflist as get_bflist
    from artistools.atomic import get_composition_data as get_composition_data
    from artistools.atomic import get_composition_data_from_outputfile as get_composition_data_from_outputfile
    from artistools.atomic import get_elsymbol as get_elsymbol
    from artistools.atomic import get_elsymbols_df as get_elsymbols_df
    from artistools.atomic import get_elsymbolslist as get_elsymbolslist
    from artistools.atomic import get_ion_stage_roman_numeral_df as get_ion_stage_roman_numeral_df
    from artistools.atomic import get_ion_tuple as get_ion_tuple
    from artistools.atomic import get_ionstring as get_ionstring
    from artistools.atomic import get_linelist_pldf as get_linelist_pldf
    from artistools.atomic import get_nuclides as get_nuclides
    from artistools.atomic import get_z_a_nucname as get_z_a_nucname
    from artistools.atomic import LineTuple as LineTuple
    from artistools.atomic import read_linestatfile as read_linestatfile
    from artistools.atomic import roman_numerals as roman_numerals
    from artistools.commands import addargs as addargs
    from artistools.commands import CustomArgHelpFormatter as CustomArgHelpFormatter
    from artistools.commands import get_path as get_path
    from artistools.commands import show_version as show_version
    from artistools.estimators import read_estimators as read_estimators
    from artistools.estimators import scan_estimators as scan_estimators
    from artistools.inputmodel import add_derived_cols_to_modeldata as add_derived_cols_to_modeldata
    from artistools.inputmodel import get_cell_angle as get_cell_angle
    from artistools.inputmodel import get_dfmodel_dimensions as get_dfmodel_dimensions
    from artistools.inputmodel import get_mgi_of_velocity_kms as get_mgi_of_velocity_kms
    from artistools.inputmodel import get_modeldata as get_modeldata
    from artistools.inputmodel import save_initelemabundances as save_initelemabundances
    from artistools.inputmodel import save_modeldata as save_modeldata
    from artistools.misc import add_axis_limit_args as add_axis_limit_args
    from artistools.misc import add_figscale_args as add_figscale_args
    from artistools.misc import add_filter_args as add_filter_args
    from artistools.misc import add_maxpacketfiles_arg as add_maxpacketfiles_arg
    from artistools.misc import add_modelpath_arg as add_modelpath_arg
    from artistools.misc import add_outputfile_arg as add_outputfile_arg
    from artistools.misc import add_outputpath_arg as add_outputpath_arg
    from artistools.misc import add_series_style_args as add_series_style_args
    from artistools.misc import add_timedays_arg as add_timedays_arg
    from artistools.misc import add_timeminmax_args as add_timeminmax_args
    from artistools.misc import add_timestep_arg as add_timestep_arg
    from artistools.misc import add_viewingangle_args as add_viewingangle_args
    from artistools.misc import average_direction_bins as average_direction_bins
    from artistools.misc import firstexisting as firstexisting
    from artistools.misc import firstexisting_or_none as firstexisting_or_none
    from artistools.misc import flatten_list as flatten_list
    from artistools.misc import get_cellsofmpirank as get_cellsofmpirank
    from artistools.misc import get_costheta_bins as get_costheta_bins
    from artistools.misc import get_costhetabin_phibin_labels as get_costhetabin_phibin_labels
    from artistools.misc import get_deposition as get_deposition
    from artistools.misc import get_dirbin_labels as get_dirbin_labels
    from artistools.misc import get_dirbins as get_dirbins
    from artistools.misc import get_escaped_arrivalrange as get_escaped_arrivalrange
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
    from artistools.misc import print_theta_phi_definitions as print_theta_phi_definitions
    from artistools.misc import read_rank_outputfiles as read_rank_outputfiles
    from artistools.misc import readnoncommentline as readnoncommentline
    from artistools.misc import resolve_outputfile as resolve_outputfile
    from artistools.misc import set_args_from_dict as set_args_from_dict
    from artistools.misc import split_multitable_dataframe as split_multitable_dataframe
    from artistools.misc import stripallsuffixes as stripallsuffixes
    from artistools.misc import trim_or_pad as trim_or_pad
    from artistools.misc import vec_len as vec_len
    from artistools.misc import write_parquet_atomic as write_parquet_atomic
    from artistools.misc import zopen as zopen
    from artistools.misc import zopenpl as zopenpl
    from artistools.plottools import set_mpl_style as set_mpl_style
else:
    # Resolve re-exports on first attribute access (PEP 562) so that importing artistools does not
    # pull in matplotlib/polars/pandas. Python >= 3.15 gets this from the lazy import filter above,
    # but the CLI needs fast startup on 3.13/3.14 too. Keep these maps in sync with the
    # t.TYPE_CHECKING import block, which is what the type checkers see (test_package_attrs checks this).
    _submodules = frozenset({
        "atomic",
        "codecomparison",
        "commands",
        "constants",
        "ejectaopacity",
        "estimators",
        "gsinetwork",
        "hesma_scripts",
        "inputmodel",
        "lightcurve",
        "linefluxes",
        "logfiles",
        "macroatom",
        "misc",
        "nltepops",
        "nonthermal",
        "packets",
        "plotspherical",
        "plottools",
        "radfield",
        "rustext",
        "spectra",
        "transitions",
        "version",
        "viewing_angles_visualization",
        "writecomparisondata",
    })

    _functionmodules: dict[str, str] = {
        funcname: modulename
        for modulename, funcnames in {
            "artistools.atomic": (
                "LineTuple",
                "decode_roman_numeral",
                "get_atomic_number",
                "get_bflist",
                "get_composition_data",
                "get_composition_data_from_outputfile",
                "get_elsymbol",
                "get_elsymbols_df",
                "get_elsymbolslist",
                "get_ion_stage_roman_numeral_df",
                "get_ion_tuple",
                "get_ionstring",
                "get_linelist_pldf",
                "get_nuclides",
                "get_z_a_nucname",
                "read_linestatfile",
                "roman_numerals",
            ),
            "artistools.commands": ("CustomArgHelpFormatter", "addargs", "get_path", "show_version"),
            "artistools.estimators": ("read_estimators", "scan_estimators"),
            "artistools.inputmodel": (
                "add_derived_cols_to_modeldata",
                "get_cell_angle",
                "get_dfmodel_dimensions",
                "get_mgi_of_velocity_kms",
                "get_modeldata",
                "save_initelemabundances",
                "save_modeldata",
            ),
            "artistools.misc": (
                "add_axis_limit_args",
                "add_figscale_args",
                "add_filter_args",
                "add_maxpacketfiles_arg",
                "add_modelpath_arg",
                "add_outputfile_arg",
                "add_outputpath_arg",
                "add_series_style_args",
                "add_timedays_arg",
                "add_timeminmax_args",
                "add_timestep_arg",
                "add_viewingangle_args",
                "average_direction_bins",
                "firstexisting",
                "firstexisting_or_none",
                "flatten_list",
                "get_cellsofmpirank",
                "get_costheta_bins",
                "get_costhetabin_phibin_labels",
                "get_deposition",
                "get_dirbin_labels",
                "get_dirbins",
                "get_escaped_arrivalrange",
                "get_file_metadata",
                "get_filterfunc",
                "get_grid_mapping",
                "get_inputparams",
                "get_model_name",
                "get_mpiranklist",
                "get_mpirankofcell",
                "get_nprocs",
                "get_nu_grid",
                "get_phi_bins",
                "get_runfolders",
                "get_time_range",
                "get_timestep_of_timedays",
                "get_timestep_time",
                "get_timestep_times",
                "get_timesteps",
                "get_viewingdirection_costhetabincount",
                "get_viewingdirection_phibincount",
                "get_viewingdirectionbincount",
                "get_vpkt_config",
                "get_vspec_dir_labels",
                "get_wid_init_at_tmodel",
                "makelist",
                "match_closest_time",
                "merge_pdf_files",
                "normalize_path_list",
                "parallel_map",
                "parse_cli_args",
                "parse_range",
                "parse_range_list",
                "print_theta_phi_definitions",
                "read_rank_outputfiles",
                "readnoncommentline",
                "resolve_outputfile",
                "set_args_from_dict",
                "split_multitable_dataframe",
                "stripallsuffixes",
                "trim_or_pad",
                "vec_len",
                "write_parquet_atomic",
                "zopen",
                "zopenpl",
            ),
            "artistools.plottools": ("set_mpl_style",),
        }.items()
        for funcname in funcnames
    }

    def __getattr__(name: str) -> t.Any:
        import importlib

        if name in _submodules:
            return importlib.import_module(f"artistools.{name}")

        if (modulename := _functionmodules.get(name)) is not None:
            attr = getattr(importlib.import_module(modulename), name)
            # cache so later accesses skip __getattr__ (an idempotent write, so safe without the GIL)
            globals()[name] = attr
            return attr

        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    def __dir__() -> list[str]:
        return sorted(set(globals()) | _submodules | set(_functionmodules))
