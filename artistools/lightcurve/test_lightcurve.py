import typing as t
from operator import itemgetter
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import pytest
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


def test_filter_data_is_sorted_by_wavelength() -> None:
    """Filter curves must come back in ascending wavelength order with transmissions still paired.

    The files under data/filters/NOT are stored out of order. Callers interpolate on this grid and
    integrate over it with np.trapezoid, which silently cancels flux across negative-width segments,
    so the sort must happen here rather than at each use.
    """
    filterdir = Path(at.get_path("artistools_dir"), "data/filters/")

    rawlines = (filterdir / "NOT" / "B.txt").read_text(encoding="utf-8").splitlines()[4:]
    rawpairs = {float(row.split()[0]): float(row.split()[1]) for row in rawlines if row.split()}
    assert sorted(rawpairs) != list(rawpairs), "this fixture is only meaningful while the file is unsorted"

    _, wavefilter, transmission, wavefilter_min, wavefilter_max = at.lightcurve.get_filter_data(filterdir, "NOT/B")

    assert np.all(np.diff(wavefilter) > 0), "wavelengths must be strictly ascending after sorting"
    assert wavefilter_min == wavefilter[0]
    assert wavefilter_max == wavefilter[-1]

    # the sort must move transmissions with their wavelengths, not just reorder one of the two
    assert len(wavefilter) == len(rawpairs)
    for wavelength, transmit in zip(wavefilter, transmission, strict=True):
        assert transmit == rawpairs[wavelength]


def test_band_magnitude_calculations() -> None:
    band_magnitude_data = at.lightcurve.generate_band_lightcurve_data(
        modelpath,
        plotvspecpol=False,
        plotviewingangle=False,
        filter=["bol", "U", "B", "V", "I"],
        timemin=290.0,
        timemax=300.0,
        average_over_phi_angle=False,
        average_over_theta_angle=False,
    )

    expected_summary = {
        "bol": ((290.381, -12.522955565351443), (299.309, -12.290504747994545), -12.486325221679541),
        "U": ((290.381, -11.72755172004823), (299.309, -10.940395871435907), -11.552651445171437),
        "B": ((290.381, -12.80311303703452), (299.309, -12.468614058886018), -12.729462436319448),
        "V": ((290.381, -13.134615284653417), (299.309, -12.922527436514791), -13.018633619232805),
        "I": ((290.381, -12.353784741224969), (299.309, -12.099751514986119), -12.443177921324608),
    }
    expected_brightest = {
        "bol": (291.359, -12.572391488690043),
        "U": (298.303, -11.927722052704134),
        "B": (298.303, -12.885253294760465),
        "V": (293.327, -13.166532931259166),
        "I": (295.307, -12.701305015349229),
    }

    assert band_magnitude_data.keys() == expected_summary.keys()
    for band_name, (expected_first, expected_last, expected_mean) in expected_summary.items():
        magnitudes = band_magnitude_data[band_name]
        assert len(magnitudes) == 10
        assert magnitudes[0] == pytest.approx(expected_first)
        assert magnitudes[-1] == pytest.approx(expected_last)
        assert np.mean([magnitude for _, magnitude in magnitudes]) == pytest.approx(expected_mean)
        assert min(magnitudes, key=itemgetter(1)) == pytest.approx(expected_brightest[band_name])


def test_band_magnitude_selection_and_colour() -> None:
    band_magnitude_data = at.lightcurve.generate_band_lightcurve_data(
        modelpath,
        plotvspecpol=False,
        plotviewingangle=False,
        filter=["B", "V"],
        timemin=290.0,
        timemax=300.0,
        average_over_phi_angle=False,
        average_over_theta_angle=False,
    )

    times, b_magnitudes = at.lightcurve.get_band_lightcurve(band_magnitude_data, "B", timemin=293.0, timemax=296.0)

    assert times == pytest.approx([293.327, 294.315, 295.307])
    assert b_magnitudes == pytest.approx([-12.708765155582157, -12.656976514620492, -12.794116835974194])

    colour_times, b_minus_v = at.lightcurve.get_colour_delta_mag(band_magnitude_data, ["B", "V"])
    assert colour_times == pytest.approx([time for time, _ in band_magnitude_data["B"]])
    assert b_minus_v == pytest.approx([
        0.33150224761889824,
        0.2741971168171453,
        0.23314295407410945,
        0.4577677756770093,
        0.34487289523804776,
        0.0984198887313692,
        0.2854667714378305,
        0.38998118936833315,
        0.022447612542014994,
        0.4539133776287727,
    ])


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


def test_get_colour_delta_mag_unequal_sampling() -> None:
    """A band with no flux at some time has no point there, so the two bands can be sampled differently."""
    band_lightcurve_data: dict[str, list[tuple[float, float]]] = {
        "B": [(10.0, -18.0), (20.0, -17.0), (30.0, -16.0)],
        "V": [(10.0, -18.5), (30.0, -16.5), (40.0, -15.0)],
    }

    times, colours = at.lightcurve.get_colour_delta_mag(band_lightcurve_data, ["B", "V"])

    assert times == [10.0, 30.0]
    assert colours == pytest.approx([0.5, 0.5])
