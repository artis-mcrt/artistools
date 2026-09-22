"""artistools.

A collection of plotting, analysis, and file format conversion tools
for the ARTIS radiative transfer code.
"""

# ruff:file-ignore[non-empty-init-module]
import sys

if sys.version_info >= (3, 15) and hasattr(sys, "set_lazy_imports_filter") and hasattr(sys, "set_lazy_imports"):
    sys.set_lazy_imports_filter(
        # matplotlib registers docstring parts as a side effect of some imports, and later modules read them at
        # import time. Thus the imports inside matplotlib stay eager, but an import of matplotlib can be lazy
        lambda importing, imported, _fromlist: (
            importing.partition(".")[0] not in {"matplotlib", "mpl_toolkits"}
            and not imported.startswith(("numpy", "polars", "polars.exceptions", "polars.selectors"))
        )
    )
    sys.set_lazy_imports("all")

    # numpy has to reach sys.modules before anything imports polars. Otherwise polars makes its own proxy
    # for numpy, and on free-threaded 3.15 two threads that resolve that proxy at once raise with "'module'
    # object does not support item assignment". Every command loads numpy, thus this costs no start time
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
from artistools import make_vpkt_input as make_vpkt_input
from artistools import misc as misc
from artistools import nltepops as nltepops
from artistools import nonthermal as nonthermal
from artistools import packets as packets
from artistools import plotlinefluxes as plotlinefluxes
from artistools import plotlogfiles as plotlogfiles
from artistools import plotradfield as plotradfield
from artistools import plotspherical as plotspherical
from artistools import plottools as plottools
from artistools import plottransitions as plottransitions
from artistools import plotviewingangles as plotviewingangles
from artistools import rustext as rustext
from artistools import spectra as spectra
from artistools import timesteps as timesteps
from artistools import transitions as transitions
from artistools import writecomparisondata as writecomparisondata
from artistools.atomic import decode_roman_numeral as decode_roman_numeral
from artistools.atomic import get_atomic_number as get_atomic_number
from artistools.atomic import get_elsymbol as get_elsymbol
from artistools.atomic import get_ion_tuple as get_ion_tuple
from artistools.atomic import get_ionstring as get_ionstring
from artistools.atomic import get_z_a_nucname as get_z_a_nucname
from artistools.commands import get_path as get_path
from artistools.estimators import scan_estimators as scan_estimators
from artistools.inputmodel import add_derived_cols_to_modeldata as add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata as get_modeldata
from artistools.misc import firstexisting as firstexisting
from artistools.misc import get_deposition as get_deposition
from artistools.misc import get_inputparams as get_inputparams
from artistools.misc import get_model_name as get_model_name
from artistools.misc import get_nprocs as get_nprocs
from artistools.misc import get_timestep_of_timedays as get_timestep_of_timedays
from artistools.misc import get_timestep_times as get_timestep_times
from artistools.misc import zopen as zopen
