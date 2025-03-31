#!/usr/bin/env python3
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
from scipy import integrate

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
outputpath = at.get_config()["path_testoutput"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot(mockplot, benchmark) -> None:
    benchmark(at.lightcurve.plot, argsraw=[], modelpath=[modelpath], outputfile=outputpath, frompackets=False)

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 257.253, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 333.334, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 293.67411, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 22.2348791, rtol=1e-4)

    integral = integrate.trapezoid(arr_lum, arr_time_d)
    assert np.isclose(integral, 2.4189054554e42, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 3.231155e40, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 7.2115e39, rtol=1e-4)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_frompackets(mockplot, benchmark) -> None:
    benchmark(
        at.lightcurve.plot,
        argsraw=[],
        modelpath=modelpath,
        frompackets=True,
        outputfile=Path(outputpath, "lightcurve_from_packets.pdf"),
    )

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 257.253, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 333.33389, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 293.67411, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 22.23483, rtol=1e-4)

    integral = integrate.trapezoid(arr_lum, arr_time_d)

    assert np.isclose(integral, 9.0323767e40, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 1.2033423e39, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 3.6121162e38, rtol=1e-4)


def test_band_lightcurve_plot() -> None:
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["B"], outputfile=outputpath)


def test_band_lightcurve_peakmag_risetime_plot() -> None:
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath,
        filter=["bol", "B"],
        include_delta_m40=True,
        plotviewingangle=-1,
        timemin=250,
        timemax=300,
        save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
        outputfile=outputpath,
    )


def test_band_lightcurve_subplots() -> None:
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["bol", "B"], outputfile=outputpath)


def test_colour_evolution_plot() -> None:
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], outputfile=outputpath)


def test_colour_evolution_subplots() -> None:
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["U-B", "B-V"], outputfile=outputpath)
