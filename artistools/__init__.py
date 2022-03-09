#!/usr/bin/env python3
"""Artistools.

A collection of plotting, analysis, and file format conversion tools for the ARTIS radiative transfer code.
"""

import sys
from .__main__ import main, addargs
from .commands import commandlist
from .config import config, num_processes, figwidth, enable_diskcache
from .misc import *
from .inputmodel import get_mgi_of_velocity_kms

import artistools.atomic
import artistools.codecomparison
import artistools.deposition
import artistools.estimators
import artistools.inputmodel
import artistools.lightcurve
import artistools.macroatom
import artistools.nltepops
import artistools.nonthermal
import artistools.packets
import artistools.radfield
import artistools.spectra
import artistools.transitions
import artistools.plottools


if sys.version_info < (3,):
    print("Python 2 not supported")
