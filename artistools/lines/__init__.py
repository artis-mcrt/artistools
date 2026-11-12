"""Compute the opacities of the lines in the ejecta, and the contributions of the lines to the emission."""

from artistools.lines import core as core
from artistools.lines.core import addarg_excitationtemperature as addarg_excitationtemperature
from artistools.lines.core import get_excitation_temperature_column as get_excitation_temperature_column
from artistools.lines.core import get_opacity_atomic_data as get_opacity_atomic_data

# isort: split
from artistools.lines import ejectaopacity as ejectaopacity
from artistools.lines import linecontributions as linecontributions
from artistools.lines import plotlinecontributions as plotlinecontributions
from artistools.lines import plotopacity as plotopacity
