#!/usr/bin/env python3

import hashlib
import math
import numpy as np
import os
import os.path
import pandas as pd
from astropy import constants as const
from pathlib import Path

import artistools as at

modelpath = Path(os.path.dirname(os.path.abspath(__file__)), 'data')
outputpath = Path(os.path.dirname(os.path.abspath(__file__)), 'output')
at.enable_diskcache = False


def test_commands():
    import importlib
    # ensure that the commands are pointing to valid submodule.function() targets
    for command, (submodulename, funcname) in sorted(at.commandlist.items()):
        submodule = importlib.import_module(submodulename, package='artistools')
        assert hasattr(submodule, funcname)


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
    at.deposition.main(argsraw=[], modelpath=modelpath)


def test_estimator_snapshot():
    at.estimators.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)


def test_estimator_timeevolution():
    at.estimators.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, modelgridindex=0, x='time')


def test_get_modeldata():
    dfmodeldata, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata(modelpath, get_abundances=True)
    assert np.isclose(t_model_init_days, 0.00115740740741, rtol=0.0001)
    assert np.isclose(vmax_cmps, 800000000., rtol=0.0001)

    assert (
        hashlib.sha256(pd.util.hash_pandas_object(dfmodeldata, index=True).values).hexdigest() ==
        '40a02dfa933f6b28671d42f3cf69a182955a5a89dc93bbcd22c894192375fe9b')


def test_lightcurve():
    at.lightcurve.main(argsraw=[], modelpath=modelpath, outputfile=outputpath)


def test_lightcurve_frompackets():
    at.lightcurve.main(argsraw=[], modelpath=modelpath, frompackets=True,
                       outputfile=os.path.join(outputpath, 'lightcurve_from_packets.pdf'))


def test_lightcurve_magnitudes_plot():
    at.lightcurve.main(argsraw=[], modelpath=modelpath, magnitude=True, outputfile=outputpath)


def test_macroatom():
    at.macroatom.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=10)


def test_makemodel():
    at.makemodel.botyanski2017.main(argsraw=[], outputpath=outputpath)


def test_menu():
    at.main(argsraw=[])
    at.showtimesteptimes(modelpath=modelpath)


def test_nltepops():
    # at.nltepops.main(modelpath=modelpath, outputfile=outputpath, timedays=300),
    #                    **benchargs)
    at.nltepops.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40)


def test_nonthermal():
    at.nonthermal.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=70)


def test_radfield():
    at.radfield.main(argsraw=[], modelpath=modelpath, modelgridindex=0, outputfile=outputpath)


def test_get_ionrecombratecalibration():
    at.atomic.get_ionrecombratecalibration(modelpath=modelpath)


def test_spencerfano():
    at.spencerfano.main(argsraw=[], modelpath=modelpath, timedays=300, makeplot=True, npts=200,
                        noexcitation=True, outputfile=outputpath)


def test_transitions():
    at.transitions.main(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=300)
