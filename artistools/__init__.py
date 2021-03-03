#!/usr/bin/env python3
"""Artistools.

A collection of plotting, analysis, and file format conversion tools for the ARTIS radiative transfer code.
"""

import sys
from .__main__ import main, addargs
from .misc import *

import artistools.atomic
import artistools.deposition
import artistools.estimators
import artistools.inputmodel
import artistools.lightcurve
import artistools.macroatom
import artistools.makemodel
import artistools.nltepops
import artistools.nonthermal
import artistools.packets
import artistools.radfield
import artistools.spectra
import artistools.transitions


if sys.version_info < (3,):
    print("Python 2 not supported")
