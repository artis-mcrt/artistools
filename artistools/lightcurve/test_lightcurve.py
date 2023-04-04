#!/usr/bin/env python3
from pathlib import Path
from unittest import mock

import matplotlib.axes
import numpy as np

import artistools as at

modelpath = at.get_config()["path_testartismodel"]
outputpath = at.get_config()["path_testoutput"]
at.set_config("enable_diskcache", False)


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_lightcurve_plot(mockplot):
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], outputfile=outputpath, frompackets=False)

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 250.421, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 349.412, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 297.20121, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 28.83886, rtol=1e-4)

    integral = np.trapz(arr_lum, arr_time_d)
    assert np.isclose(integral, 2.7955e42, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 2.8993e40, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 1.1244e40, rtol=1e-4)


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_lightcurve_plot_frompackets(mockplot):
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath,
        frompackets=True,
        outputfile=Path(outputpath, "lightcurve_from_packets.pdf"),
    )

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 250.421, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 349.412, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 297.20121, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 28.83886, rtol=1e-4)

    integral = np.trapz(arr_lum, arr_time_d)

    assert np.isclose(integral, 1.0795078026708302e41, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 1.1176e39, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 4.4820e38, rtol=1e-4)


def test_band_lightcurve_plot():
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["B"], outputfile=outputpath)


def test_band_lightcurve_subplots():
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["bol", "B"], outputfile=outputpath)


def test_colour_evolution_plot():
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], outputfile=outputpath)


def test_colour_evolution_subplots():
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["U-B", "B-V"], outputfile=outputpath)
