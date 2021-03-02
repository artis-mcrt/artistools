#!/usr/bin/env python3

import math
import numpy as np
import os.path
import pandas as pd
from astropy import constants as const
from pathlib import Path

import artistools as at
import artistools.atomic
import artistools.deposition
import artistools.inputmodel
import artistools.lightcurve
import artistools.macroatom
import artistools.makemodel.botyanski2017
import artistools.nltepops
import artistools.nonthermal
import artistools.radfield
import artistools.spectra
import artistools.transitions

modelpath = Path(os.path.dirname(os.path.abspath(__file__)), 'data')
outputpath = Path(os.path.dirname(os.path.abspath(__file__)), 'output')
at.enable_diskcache = False


def test_timestep_times():
    timestartarray = at.get_timestep_times_float(modelpath, loc='start')
    timedeltarray = at.get_timestep_times_float(modelpath, loc='delta')
    timemidarray = at.get_timestep_times_float(modelpath, loc='mid')
    assert len(timestartarray) == 100
    assert math.isclose(float(timemidarray[0]), 250.421, abs_tol=1e-3)
    assert math.isclose(float(timemidarray[-1]), 349.412, abs_tol=1e-3)

    assert all([tstart < tmid < (tstart + tdelta)
                for tstart, tdelta, tmid in zip(timestartarray, timedeltarray, timemidarray)])


def test_deposition():
    at.deposition.main(modelpath=modelpath)


def test_estimator_snapshot():
    at.estimators.main(modelpath=modelpath, outputfile=outputpath, timedays=300)


def test_estimator_timeevolution():
    at.estimators.main(modelpath=modelpath, outputfile=outputpath, modelgridindex=0, x='time')


def test_get_modeldata():
    at.inputmodel.get_modeldata(modelpath)


def test_lightcurve():
    at.lightcurve.main(modelpath=modelpath, outputfile=outputpath)


def test_lightcurve_frompackets():
    at.lightcurve.main(modelpath=modelpath, frompackets=True,
                       outputfile=os.path.join(outputpath, 'lightcurve_from_packets.pdf'))


def test_lightcurve_magnitudes_plot():
    at.lightcurve.main(modelpath=modelpath, magnitude=True, outputfile=outputpath)


def test_macroatom():
    at.macroatom.main(modelpath=modelpath, outputfile=outputpath, timestep=10)


def test_makemodel():
    at.makemodel.botyanski2017.main(outputpath=outputpath)


def test_menu():
    at.main()
    at.showtimesteptimes(modelpath=modelpath)


def test_nltepops():
    # at.nltepops.main(modelpath=modelpath, outputfile=outputpath, timedays=300),
    #                    **benchargs)
    at.nltepops.main(modelpath=modelpath, outputfile=outputpath, timestep=40)


def test_nonthermal():
    at.nonthermal.main(modelpath=modelpath, outputfile=outputpath, timestep=70)


def test_radfield():
    at.radfield.main(modelpath=modelpath, modelgridindex=0, outputfile=outputpath)


def test_get_ionrecombratecalibration():
    at.atomic.get_ionrecombratecalibration(modelpath=modelpath)

def test_spencerfano():
    at.spencerfano.main(modelpath=modelpath, timedays=300, makeplot=True, npts=200, noexcitation=True, outputfile=outputpath)


def test_transitions():
    at.transitions.main(modelpath=modelpath, outputfile=outputpath, timedays=300)
