import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot(mockplot: t.Any, benchmark: BenchmarkFixture) -> None:
    benchmark(lambda: at.lightcurve.plot(argsraw=[], modelpath=[modelpath], outputfile=outputpath, frompackets=False))

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 257.253, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 333.334, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 293.67411, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 22.2348791, rtol=1e-4)

    integral = np.trapezoid(arr_lum, arr_time_d)
    assert np.isclose(integral, 2.4189054554e42, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 3.231155e40, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 7.2115e39, rtol=1e-4)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_frompackets(mockplot: t.Any, benchmark: BenchmarkFixture) -> None:
    benchmark(
        lambda: at.lightcurve.plot(
            argsraw=[],
            modelpath=modelpath,
            frompackets=True,
            outputfile=Path(outputpath, "lightcurve_from_packets.pdf"),
        )
    )

    arr_time_d = np.array(mockplot.call_args[0][1])
    arr_lum = np.array(mockplot.call_args[0][2])

    assert np.isclose(arr_time_d.min(), 257.253, rtol=1e-4)
    assert np.isclose(arr_time_d.max(), 333.33389, rtol=1e-4)

    assert np.isclose(arr_time_d.mean(), 293.67411, rtol=1e-4)
    assert np.isclose(arr_time_d.std(), 22.23483, rtol=1e-4)

    integral = np.trapezoid(arr_lum, arr_time_d)

    assert np.isclose(integral, 9.0323767e40, rtol=1e-2)

    assert np.isclose(arr_lum.mean(), 1.2039713396033405e39, rtol=1e-4)
    assert np.isclose(arr_lum.std(), 3.614004402353378e38, rtol=1e-4)


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


modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
modelpath_vspecpol = at.get_path("testdata") / "vspecpolmodel"


def get_plotted_lightcurves(mockplot: t.Any, **kwargs: t.Any) -> list[tuple[np.ndarray, np.ndarray]]:
    mockplot.reset_mock()
    at.lightcurve.plot(argsraw=[], outputfile=outputpath, **kwargs)
    return [(np.array(callargs[0][1]), np.array(callargs[0][2])) for callargs in mockplot.call_args_list]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_direction_tokens_frompackets(mockplot: t.Any) -> None:
    kwargs = {"modelpath": modelpath, "frompackets": True}

    costheta2 = get_plotted_lightcurves(mockplot, plotviewingangle=["costheta2"], **kwargs)
    legacy_costheta2 = get_plotted_lightcurves(mockplot, plotviewingangle=[20], average_over_phi_angle=True, **kwargs)
    assert len(costheta2) == len(legacy_costheta2) == 1
    assert np.allclose(costheta2[0][1], legacy_costheta2[0][1], equal_nan=True)

    phi3 = get_plotted_lightcurves(mockplot, plotviewingangle=["phi3"], **kwargs)
    legacy_phi3 = get_plotted_lightcurves(mockplot, plotviewingangle=[3], average_over_theta_angle=True, **kwargs)
    assert np.allclose(phi3[0][1], legacy_phi3[0][1], equal_nan=True)

    dirbin23 = get_plotted_lightcurves(mockplot, plotviewingangle=["t2p3"], **kwargs)
    legacy_dirbin23 = get_plotted_lightcurves(mockplot, plotviewingangle=[23], **kwargs)
    assert np.allclose(dirbin23[0][1], legacy_dirbin23[0][1], equal_nan=True)

    # phi-averaged, theta-averaged, individual, and spherically averaged series mixed in one plot
    mixed = get_plotted_lightcurves(mockplot, plotviewingangle=["costheta2", "phi3", "23", "all"], **kwargs)
    assert len(mixed) == 4
    assert np.allclose(mixed[0][1], costheta2[0][1], equal_nan=True)
    assert np.allclose(mixed[1][1], phi3[0][1], equal_nan=True)
    assert np.allclose(mixed[2][1], dirbin23[0][1], equal_nan=True)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_direction_tokens_from_files(mockplot: t.Any) -> None:
    costheta2 = get_plotted_lightcurves(mockplot, modelpath=modelpath_classic_3d, plotviewingangle=["costheta2"])
    legacy_costheta2 = get_plotted_lightcurves(
        mockplot, modelpath=modelpath_classic_3d, plotviewingangle=[20], average_over_phi_angle=True
    )
    assert len(costheta2) == len(legacy_costheta2) == 1
    assert np.allclose(costheta2[0][1], legacy_costheta2[0][1], equal_nan=True)

    # mix averaged bins, an individual bin, and the spherical average from light_curve.out in one plot
    mixed = get_plotted_lightcurves(
        mockplot, modelpath=modelpath_classic_3d, plotviewingangle=["costheta2", "23", "all"]
    )
    assert len(mixed) == 3
    assert np.allclose(mixed[0][1], costheta2[0][1], equal_nan=True)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_vspecpol_from_files(mockplot: t.Any) -> None:
    series = get_plotted_lightcurves(mockplot, modelpath=modelpath_vspecpol, plotviewingangle=["vpkt0", "v5"])
    assert len(series) == 2
    assert all(len(arr_time_d) > 0 and np.nanmax(arr_lum) > 0.0 for arr_time_d, arr_lum in series)
