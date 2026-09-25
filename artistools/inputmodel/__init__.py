"""Read, write, convert, and plot ARTIS input models."""

# the core module comes first. A module of a different package imports these names from this package, and
# the plot modules below import such modules, thus a name must exist before a cycle comes back here
from artistools.inputmodel.core import add_derived_cols_to_modeldata as add_derived_cols_to_modeldata
from artistools.inputmodel.core import dimension_reduce_model as dimension_reduce_model
from artistools.inputmodel.core import get_cell_selection as get_cell_selection
from artistools.inputmodel.core import get_initelemabundances as get_initelemabundances
from artistools.inputmodel.core import get_mgi_of_velocity_kms as get_mgi_of_velocity_kms
from artistools.inputmodel.core import get_modeldata as get_modeldata
from artistools.inputmodel.core import get_selection_labels as get_selection_labels
from artistools.inputmodel.core import save_empty_abundance_file as save_empty_abundance_file
from artistools.inputmodel.core import save_initelemabundances as save_initelemabundances
from artistools.inputmodel.core import save_modeldata as save_modeldata
from artistools.inputmodel.core import scale_model_to_time as scale_model_to_time

# isort: split
from artistools.inputmodel import core as core
from artistools.inputmodel import describeinputmodel as describeinputmodel
from artistools.inputmodel import downscale3dgrid as downscale3dgrid
from artistools.inputmodel import energyinputfiles as energyinputfiles
from artistools.inputmodel import from_e2e_model as from_e2e_model
from artistools.inputmodel import makeartismodel as makeartismodel
from artistools.inputmodel import maptogrid as maptogrid
from artistools.inputmodel import modelfromhydro as modelfromhydro
from artistools.inputmodel import opacityinputfile as opacityinputfile
from artistools.inputmodel import plotdensity as plotdensity
from artistools.inputmodel import plotinitialabundances as plotinitialabundances
from artistools.inputmodel import plotinitialcomposition as plotinitialcomposition
from artistools.inputmodel import rprocess_from_trajectory as rprocess_from_trajectory
from artistools.inputmodel import slice1dfromconein3dmodel as slice1dfromconein3dmodel
from artistools.inputmodel import to_tardis as to_tardis
