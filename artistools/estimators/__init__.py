"""Artistools - functions for handling data in estimators_????.out files (e.g., temperatures, densities, abundances)."""

__all__ = ["plot"]

# the core module comes first. A module of a different package imports these names from this package, and
# the plot modules below import such modules, thus a name must exist before a cycle comes back here
from artistools.estimators.estimators import add_derived_estimator_columns as add_derived_estimator_columns
from artistools.estimators.estimators import CACHEVERSION as CACHEVERSION
from artistools.estimators.estimators import estimbatch_parquet_is_current as estimbatch_parquet_is_current
from artistools.estimators.estimators import get_averageexcitation as get_averageexcitation
from artistools.estimators.estimators import get_units as get_units
from artistools.estimators.estimators import read_estimators as read_estimators
from artistools.estimators.estimators import scan_estimators as scan_estimators

# isort: split
from artistools.estimators import deposition as deposition
from artistools.estimators import estimators_classic as estimators_classic
from artistools.estimators import exportmassfractions as exportmassfractions
from artistools.estimators import plotestimators as plotestimators
from artistools.estimators.plotestimators import addargs as addargs
from artistools.estimators.plotestimators import main as plot
