import argparse
import typing as t
import warnings
from collections.abc import Sequence
from operator import itemgetter
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.markers as mplmarkers
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from matplotlib.container import ErrorbarContainer
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import Mbol_sun
from artistools.lightcurve import lightcurve
from artistools.lightcurve import viewingangleanalysis

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot(mockplot: mock.MagicMock, benchmark: BenchmarkFixture) -> None:
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
def test_lightcurve_plot_frompackets(mockplot: mock.MagicMock, benchmark: BenchmarkFixture) -> None:
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


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_lightcurve_plot_reflightcurves_keep_their_errorbars(mockerrorbar: mock.MagicMock) -> None:
    """Reference light curves keep their error bars by either route, the model path list or -reflightcurves."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=["AT2017gfo_waxmanetal2018.txt"],
        reflightcurves=["AT2017gfo_smarttetal2017.txt"],
        outputfile=Path(outputpath, "lightcurve_reflightcurves.pdf"),
    )

    # one call for the reference light curve given as a model path, one for the -reflightcurves file
    assert mockerrorbar.call_count == 2

    labels = [callitem[1]["label"] for callitem in mockerrorbar.call_args_list]
    assert labels == ["AT2017gfo (Waxman+2018)", "AT2017gfo (Smartt+2017)"]

    # the -reflightcurves file continues the grey sequence rather than starting it again
    assert [callitem[1]["color"] for callitem in mockerrorbar.call_args_list] == ["0.0", "0.4"]

    assert all(callitem[1]["zorder"] == 0 for callitem in mockerrorbar.call_args_list)

    for callitem, expected_time_d_min in zip(mockerrorbar.call_args_list, (0.5, 0.638), strict=True):
        arr_time_d = np.array(callitem[0][1])
        arr_lum = np.array(callitem[0][2])
        arr_errminus, arr_errplus = (np.array(err) for err in callitem[1]["yerr"])

        assert np.isclose(arr_time_d.min(), expected_time_d_min, rtol=1e-4)
        assert arr_errminus.shape == arr_lum.shape
        assert arr_errplus.shape == arr_lum.shape
        assert (arr_errminus > 0.0).all()
        assert (arr_errplus > 0.0).all()


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


@mock.patch("artistools.spectra.get_spectrum_at_time")
def test_spectrum_filter_range_includes_bracketing_points(mockgetspectrum: mock.MagicMock) -> None:
    """Band integration must retain the spectrum point on each side of the filter range."""
    mockgetspectrum.return_value = pl.DataFrame({
        "lambda_angstroms": [1000.0, 2000.0, 3000.0, 4000.0],
        "f_lambda": [1.0, 2.0, 4.0, 8.0],
    })

    wavelength, flux = lightcurve.get_spectrum_in_filter_range(
        modelpath=Path(), timestep=0, time=1.0, wavefilter_min=2200.0, wavefilter_max=2800.0
    )

    assert np.allclose(wavelength, np.array([2000.0, 3000.0]))
    assert np.allclose(flux, np.array([2.0, 4.0]))


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
        "B": ((290.381, -12.803113515779813), (299.309, -12.468614058886018), -12.72946419132831),
        "V": ((290.381, -13.134621676461588), (299.309, -12.922533596999637), -13.018642833106346),
        "I": ((290.381, -12.353786573875853), (299.309, -12.099768401014884), -12.443185093790348),
    }
    expected_brightest = {
        "bol": (291.359, -12.572391488690043),
        "U": (298.303, -11.927722052704134),
        "B": (298.303, -12.8852578600296),
        "V": (293.327, -13.16654427664928),
        "I": (295.307, -12.701314848818333),
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
    assert b_magnitudes == pytest.approx([-12.708769275316618, -12.656976514620492, -12.794120144812808])

    colour_times, b_minus_v = at.lightcurve.get_colour_delta_mag(band_magnitude_data, ["B", "V"])
    assert colour_times == pytest.approx([time for time, _ in band_magnitude_data["B"]])
    assert b_minus_v == pytest.approx([
        0.3315081606817749,
        0.2742048700430537,
        0.2331501773840028,
        0.4577750013326618,
        0.34488607340471944,
        0.09842377150842907,
        0.2854726927324762,
        0.38998827189773166,
        0.022457860681898367,
        0.4539195381136185,
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


def test_viewing_angle_peakmag_risetime_scatter_plot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The viewing angle scatter plot must not read -xmin/-xmax, which this parser does not define.

    Its time range is named -timemin/-timemax, so args has no xmin and reading it raised AttributeError
    before any scatter plot could be saved. The plot reads the data file that the first call writes.
    """
    monkeypatch.chdir(tmp_path)
    commonargs: dict[str, t.Any] = {
        "modelpath": [modelpath],
        "filter": ["B"],
        "timemin": 250,
        "timemax": 300,
        "outputfile": tmp_path,
    }
    at.lightcurve.plot(argsraw=[], save_viewing_angle_peakmag_risetime_delta_m15_to_file=True, **commonargs)
    at.lightcurve.plot(argsraw=[], make_viewing_angle_peakmag_risetime_scatter_plot=True, **commonargs)

    assert list(tmp_path.glob("*risetime_peakmag.pdf"))


def test_band_lightcurve_subplots() -> None:
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["bol", "B"], outputfile=outputpath)


def test_colour_evolution_plot() -> None:
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], outputfile=outputpath)


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_colour_evolution_plot_ylabel(mockylabel: mock.MagicMock) -> None:
    """A colour evolution plot must be labelled in delta magnitudes, not as a band magnitude.

    colour_evolution_plot assigns args.filter before asking for the labels, so reading the plot kind back off
    args labelled these axes "None Magnitude".
    """
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], outputfile=outputpath)

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert r"$\Delta$m" in ylabels, ylabels


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_linelabel_falls_back_to_the_model_name(mockplot: mock.MagicMock) -> None:
    """A series with no -label is named after its model, not after the None that pads the -label list."""
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["B"], outputfile=outputpath)

    assert [callargs.kwargs["label"] for callargs in mockplot.call_args_list] == ["TEST MODEL"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_linelabel_uses_the_label_arg(mockplot: mock.MagicMock) -> None:
    """A -label value names its series, in place of the model name."""
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, filter=["B"], label=["My model"], outputfile=outputpath)

    assert [callargs.kwargs["label"] for callargs in mockplot.call_args_list] == ["My model"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_linelabel_direction_bin_keeps_the_label_arg(mockplot: mock.MagicMock) -> None:
    """A direction bin is named after the -label value of its model, with the bin appended."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        filter=["B"],
        plotviewingangle=[0],
        label=["My model"],
        timemin=5,
        timemax=8,
        outputfile=outputpath,
    )

    labels = [callargs.kwargs["label"] for callargs in mockplot.call_args_list]
    assert len(labels) == 1
    assert labels[0].startswith("My model "), labels


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_color_arg(mockplot: mock.MagicMock) -> None:
    """A -color value must reach the plotted line."""
    at.lightcurve.plot(
        argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], color=["magenta"], outputfile=outputpath
    )

    assert [callargs.kwargs["color"] for callargs in mockplot.call_args_list] == ["magenta"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_viewingangle_colours(mockplot: mock.MagicMock) -> None:
    """Direction bins get one colour each from the tab20 colour map, whatever -color says.

    The -color list has one entry per model, so it cannot colour the direction bins. This used to warn about
    a -color argument that the user never gave, and then leave the colour unset. More direction bins than the
    colour map has colours must wrap around rather than run off the end of the list.
    """
    palette = ["#111111", "#222222"]
    ndirbins = len(palette) + 1  # one more than the palette holds, so the last bin wraps to the first colour

    # a short palette exercises the wrap with three direction bins rather than the twenty-one the tab20 map
    # would need, and the palette itself is covered by test_colour_evolution_plot_dirbin_colours_*
    with mock.patch.object(at.lightcurve.plotlightcurve, "get_dirbin_palette", return_value=palette):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=modelpath_classic_3d,
            colour_evolution=["B-V"],
            plotviewingangle=list(range(ndirbins)),
            timemin=5,
            timemax=8,
            color=["magenta"],
            outputfile=outputpath,
        )

    colors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list]
    assert colors == [*palette, palette[0]]


def test_dirbin_palette_avoids_the_assigned_colours() -> None:
    """The direction bin palette holds tab20 colours, minus any that a whole series was given."""
    tab20colors = [mplcolors.to_hex(color) for color in plt.get_cmap("tab20")(np.linspace(0, 1.0, 20))]

    palette = [mplcolors.to_hex(color) for color in at.lightcurve.plotlightcurve.get_dirbin_palette([tab20colors[0]])]
    assert palette == tab20colors[1:]

    # every colour of the map assigned leaves none free, and a bin still has to be drawn in something
    assert len(at.lightcurve.plotlightcurve.get_dirbin_palette(tab20colors)) == len(tab20colors)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_dirbin_colour_is_stable_across_subplots(mockplot: mock.MagicMock) -> None:
    """A direction bin keeps one colour across the subplots of the filter pairs.

    The legend is drawn on one subplot only, so a bin drawn in a different colour in each subplot
    contradicts the legend everywhere else.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        colour_evolution=["U-B", "B-V"],
        plotviewingangle=[0, 1],
        timemin=5,
        timemax=8,
        outputfile=outputpath,
    )

    # one call per (direction bin, filter pair), the filter pairs of a bin together
    colors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list]
    assert len(colors) == 4
    assert np.allclose(colors[0], colors[1]), "bin 0 changed colour between the subplots"
    assert np.allclose(colors[2], colors[3]), "bin 1 changed colour between the subplots"
    assert not np.allclose(colors[0], colors[2]), "the two direction bins share a colour"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_single_dirbin_colour(mockplot: mock.MagicMock) -> None:
    """Use the -color value when a model contributes a single line, even if that line is one direction bin."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        colour_evolution=["B-V"],
        plotviewingangle=[5],
        color=["magenta"],
        timemin=5,
        timemax=8,
        outputfile=outputpath,
    )

    assert [callargs.kwargs["color"] for callargs in mockplot.call_args_list] == ["magenta"]


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


@pytest.mark.parametrize(
    ("z", "dist_mpc"),
    [
        (0.0, 0.0),
        (0.005791, 24.912483443375777),  # SN 1991T
        (0.0133, 57.54493109140769),  # iPTF13ebh
        (0.01433, 62.04993050233093),  # SN 1999dq
        (0.1, 460.2999363904721),
        (0.5, 2832.9380939001253),
        (1.0, 6607.6576117749355),
        (3.0, 25422.741745189862),
    ],
)
def test_luminosity_distance(z: float, dist_mpc: float) -> None:
    """Reference values are astropy's FlatLambdaCDM(H0=70, Om0=0.3).luminosity_distance(z), which this replaced.

    astropy evaluates the same Baes et al. (2017) function through scipy's hyp2f1 where this integrates it,
    so the two agree to ~4e-14 and the tolerance is set well inside the 1e-10 that any use here needs.
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=z) == pytest.approx(dist_mpc, rel=1e-11, abs=1e-12)


def test_luminosity_distance_planck18_parameters() -> None:
    """The cosmology parameters must be used, not baked in (reference values from astropy)."""
    for z, dist_mpc in ((0.01433, 64.43316428422708), (0.5, 2927.080479237606), (2.0, 15936.22617736705)):
        assert at.lightcurve.luminosity_distance(H0=67.4, Om0=0.315, z=z) == pytest.approx(dist_mpc, rel=1e-11)


def test_luminosity_distance_negative_dark_energy() -> None:
    """Om0 > 1 leaves a flat universe with a negative dark energy density, which is still integrable.

    astropy's s = ((1 - Om0) / Om0)^(1/3) is a cube root of a negative number here, which it survives only
    by carrying a complex s through and discarding the imaginary part. The u form never forms s at all, so
    these come from adaptive quadrature rather than from astropy, which is itself ~7e-15 off at the first.
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1.5, z=0.1) == pytest.approx(425.1405279377727, rel=1e-12)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=2.0, z=1.0) == pytest.approx(4066.82076478827, rel=1e-12)


def test_luminosity_distance_dense_matter() -> None:
    """A negative dark energy density leaves an integrable pole in 1 / E at the turnaround.

    The larger Om0 is, the closer that turnaround crowds up to z = 0 and the further its pole reaches, so
    integrating over u alone ran 1.1% low for Om0 = 1e6 at z = 1 and ~1% over the last 0.01 in z above the
    turnaround of any Om0 > 1. The first two values agree with adaptive quadrature in ln(1 + z), the third
    with 32 to 1024 node rules in q, which converge where the adaptive routine warns and stops short.
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e6, z=1.0) == pytest.approx(8.56941262664432, rel=1e-11)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=100.0, z=1.0) == pytest.approx(803.7609074618974, rel=1e-11)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e6, z=1e-10) == pytest.approx(
        4.28242824230702e-07, rel=1e-11, abs=0
    )

    # just above the turnaround of Om0 = 2, where 1 / E is unbounded and u alone was 1.3% out
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=2.0, z=-0.2062994740159002) == pytest.approx(
        -1523.69803, rel=1e-6
    )

    # the q width has to come from the difference of cubes: taking qhi - qlo directly was 1.8e-7 out here
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (1e-10, 1e-8):
        expected = hubble_dist_mpc * z * (1.0 + z) * (1.0 - 0.75 * 5.0 * z)
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=5.0, z=z) == pytest.approx(expected, rel=1e-11, abs=0)


def test_luminosity_distance_past_turnaround_rejected() -> None:
    """A negative dark energy density halts the expansion, and nothing beyond that turnaround has a distance.

    E(z)^2 is negative there, which no domain check on z alone would catch. Below the turnaround the
    quadrature returns a NaN that would reach the magnitudes as one, and immediately below it something
    worse: the interior nodes still straddle positive radicands, so it returns a plausible finite number.
    """
    for Om0, z in ((2.0, -0.5), (2.0, -0.999), (1.5, -0.4), (1.5, -0.306638726649365)):
        with pytest.raises(ValueError, match="stops expanding"):
            at.lightcurve.luminosity_distance(H0=70.0, Om0=Om0, z=z)

    # just above its turnaround of z = -0.2063 the distance exists and must still be returned
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=2.0, z=-0.1) == pytest.approx(-462.8923430069639, rel=1e-11)


def test_luminosity_distance_matter_only() -> None:
    """For Om0 = 1 the integral is analytic: D_L = 2 (c / H0) (1 + z) (1 - 1 / sqrt(1 + z)).

    Its z integrand diverges as z' -> -1, so the blueshifts here are the ones quadrature cannot follow.
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (-0.999999, -0.99, -0.9, -0.01, 0.01, 0.5, 2.0, 10.0, 1e8):
        expected = 2 * hubble_dist_mpc * (1.0 + z) * (1.0 - 1.0 / np.sqrt(1.0 + z))
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1.0, z=z) == pytest.approx(expected, rel=1e-10, abs=0)


def test_luminosity_distance_matter_only_tiny_redshift() -> None:
    """Expanding the Om0 = 1 closed form gives D_L = (c / H0) z (1 + z) (1 - 3z/4 + O(z^2)) as z -> 0.

    The tolerance is tight enough to fail if that closed form is evaluated as 1 - 1 / sqrt(1 + z), which
    loses all but a few digits to cancellation here (8e-8 relative at z = 1e-10).
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (1e-10, 1e-8, 1e-6):
        expected = hubble_dist_mpc * z * (1.0 + z) * (1.0 - 0.75 * z)
        # abs=0 because these distances are ~1e-7 Mpc, so the default absolute tolerance of approx would
        # swallow the cancellation error entirely
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1.0, z=z) == pytest.approx(expected, rel=1e-11, abs=0)


def test_luminosity_distance_blueshift() -> None:
    """A blueshift is a valid redshift, and the closed forms hold for negative z as well.

    A blueshift lies below the crossover for any Om0 < 0.5, so these integrate over z rather than over u,
    which would stretch the range towards infinity as z -> -1.
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (-1e-6, -0.001, -0.01, -0.1, -0.5, -0.9, -0.999999):
        # Om0 = 0 has E(z) = 1, so the comoving distance is exactly (c / H0) z
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.0, z=z) == pytest.approx(
            hubble_dist_mpc * z * (1.0 + z), rel=1e-13, abs=0
        )
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=z) < 0.0

    # the mildly negative redshifts of nearby blueshifted galaxies are the only ones of practical interest
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=-0.001) == pytest.approx(-4.279429096775459)

    # at the edge of the domain a cosmology with no closed form has to come from the z quadrature: this
    # value agrees with both a 2048-node rule and adaptive quadrature, where the u substitution used for
    # redshifts would give a magnitude 41% too small
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=-0.999999) == pytest.approx(
        -0.0048851852579134521, rel=1e-12, abs=0
    )


def test_luminosity_distance_below_minus_one_rejected() -> None:
    """1 + z is a ratio of scale factors, so z <= -1 is not a redshift and must not silently give a NaN."""
    for z in (-1.0, -1.5, -10.0):
        with pytest.raises(ValueError, match="must be greater than -1"):
            at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=z)


def test_luminosity_distance_quadrature_extremes() -> None:
    """Pin the ends of the range that the quadrature, rather than a closed form, has to cover.

    A tiny z falls on whichever side of the crossover z = 0 sits: below it for Om0 = 0.3, whose width is
    z itself, and above it for Om0 = 0.9, whose width has to come from the difference of the roots.
    """
    hubble_dist_mpc = 299792.458 / 70.0

    # expanding the integrand gives D_L = (c / H0) z (1 + z) (1 - 3 Om0 z / 4 + O(z^2)) as z -> 0
    for Om0 in (0.3, 0.9):
        for z in (1e-10, 1e-8):
            expected = hubble_dist_mpc * z * (1.0 + z) * (1.0 - 0.75 * Om0 * z)
            assert at.lightcurve.luminosity_distance(H0=70.0, Om0=Om0, z=z) == pytest.approx(expected, rel=1e-11, abs=0)

    # adaptive quadrature in ln(1 + z), which resolves every regime of the integrand, gives this
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.3, z=1e8) == pytest.approx(1415324782383.9055, rel=1e-11)


def test_luminosity_distance_across_the_crossover() -> None:
    """1 / E has a plateau below Om0 (1 + z')^3 = 1 - Om0 and a (1 + z')^(-3/2) tail above it.

    Neither variable covers both once they differ by decades, which is why the range is split there. These
    are the two ways that goes wrong with a single variable: integrating a strongly blueshifted, nearly
    matter-only cosmology over z alone was 64% low, and taking a nearly matter-free one to high z over u
    alone was ~1e-4 out. Reference values are from adaptive quadrature in ln(1 + z).
    """
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1 - 1e-12, z=-0.999999) == pytest.approx(
        -1.1881950472843847, rel=1e-11
    )
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.999, z=-0.999999) == pytest.approx(
        -0.02942354607438868, rel=1e-11
    )
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e-8, z=1e8) == pytest.approx(556188062894733.94, rel=1e-11)
    assert at.lightcurve.luminosity_distance(H0=70.0, Om0=1e-4, z=1e5) == pytest.approx(25177127557.013027, rel=1e-11)


def test_luminosity_distance_matter_free() -> None:
    """A matter-free universe expands with E(z) = 1, so D_L = (c / H0) z (1 + z) at every redshift.

    The 2 / u^3 integrand of this limit defeats any fixed-node quadrature at high z, so it is a closed form.
    """
    hubble_dist_mpc = 299792.458 / 70.0
    for z in (0.0, 0.01, 1.0, 1e3, 1e8):
        expected = hubble_dist_mpc * z * (1.0 + z)
        assert at.lightcurve.luminosity_distance(H0=70.0, Om0=0.0, z=z) == pytest.approx(expected, rel=1e-13)


def test_read_hesma_lightcurve_file_header(tmp_path: Path) -> None:
    """Column names must come from splitting the comment header into words, not into characters."""
    hesmafile = tmp_path / "hesma_model.dat"
    hesmafile.write_text("# time bol B V\n1.0 2.0 3.0 4.0\n5.0 6.0 7.0 8.0\n", encoding="utf-8")

    dfhesma = at.lightcurve.read_hesma_lightcurve_file(hesmafile)

    assert list(dfhesma.columns) == ["time", "bol", "B", "V"]
    assert dfhesma["time"].to_list() == [1.0, 5.0]
    assert dfhesma["V"].to_list() == [4.0, 8.0]


def test_read_hesma_lightcurve_file_no_header(tmp_path: Path) -> None:
    """A file with no comment header uses its first line as the header."""
    hesmafile = tmp_path / "hesma_model_noheader.dat"
    hesmafile.write_text("time bol\n1.0 2.0\n3.0 4.0\n", encoding="utf-8")

    dfhesma = at.lightcurve.read_hesma_lightcurve_file(hesmafile)

    assert list(dfhesma.columns) == ["time", "bol"]
    assert dfhesma["bol"].to_list() == [2.0, 4.0]


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_reference_colors(mockplot: mock.MagicMock, mockerrorbar: mock.MagicMock) -> None:
    """The reference light curves get black and then grey, and the model keeps the first colour of the cycle."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=["AT2017gfo_smarttetal2017.txt", modelpath, "AT2017gfo_waxmanetal2018.txt"],
        outputfile=outputpath,
    )

    assert [callargs.kwargs["color"] for callargs in mockerrorbar.call_args_list] == ["0.0", "0.4"]

    modelcolors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list if "color" in callargs.kwargs]
    assert modelcolors == ["C0"]


@mock.patch.object(mplax.Axes, "scatter", side_effect=mplax.Axes.scatter, autospec=True)
@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_lightcurve_plot_reflightcurves_continue_the_greys(
    mockerrorbar: mock.MagicMock, mockscatter: mock.MagicMock
) -> None:
    """A -reflightcurves file follows the reference files of the model path list, thus no two series are black."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath, "AT2017gfo_smarttetal2017.txt"],
        reflightcurves=["AT2017gfo_waxmanetal2018.txt"],
        outputfile=outputpath,
    )

    assert [callargs.kwargs["color"] for callargs in mockerrorbar.call_args_list] == ["0.0", "0.4"]

    # both files have error columns, so neither route falls back to a plain scatter
    assert mockscatter.call_args_list == []


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_colors_survive_a_skipped_model(mockplot: mock.MagicMock, mockerrorbar: mock.MagicMock) -> None:
    """A model path that plots nothing must not shift the colour of every later series."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=["nonexistentmodelfolder", "AT2017gfo_smarttetal2017.txt", modelpath],
        outputfile=outputpath,
    )

    assert [callargs.kwargs["color"] for callargs in mockerrorbar.call_args_list] == ["0.0"]

    modelcolors = [callargs.kwargs["color"] for callargs in mockplot.call_args_list if "color" in callargs.kwargs]
    assert modelcolors == ["C1"]


def test_find_bol_reflightcurve_file_reads_a_compressed_file(tmp_path: Path) -> None:
    """A reference light curve that is compressed must be found under the name of the plain file."""
    import lzma

    with lzma.open(tmp_path / "myref.txt.xz", "wt", encoding="utf-8") as compressedfile:
        compressedfile.write("#time_days lum\n1.0 2.0\n")

    assert at.lightcurve.find_bol_reflightcurve_file(tmp_path / "myref.txt") == tmp_path / "myref.txt.xz"
    assert at.lightcurve.path_is_reference_lightcurve(tmp_path / "myref.txt")
    assert at.lightcurve.find_bol_reflightcurve_file(tmp_path / "notafile.txt") is None


REFLIGHTCURVE = "AT2017gfo_smarttetal2017.txt"


def get_reflightcurve_errorbar_call(
    mockerrorbar: mock.MagicMock,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the times and luminosities that the reference light curve was drawn with."""
    assert mockerrorbar.call_count == 1
    callargs = mockerrorbar.call_args_list[0]

    return np.array(callargs[0][1]), np.array(callargs[0][2])


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_erg_per_s(mockerrorbar: mock.MagicMock) -> None:
    """A bolometric reference light curve is plotted in erg/s by default."""
    at.lightcurve.plot(
        argsraw=[], modelpath=[REFLIGHTCURVE, modelpath], outputfile=outputpath / "lightcurve_reflightcurve_ergpers.pdf"
    )

    arr_time_d, arr_lum = get_reflightcurve_errorbar_call(mockerrorbar)

    assert np.isclose(arr_time_d[0], 0.638, rtol=1e-4)
    assert np.isclose(arr_lum[0], 1.1246049739669314e42, rtol=1e-4)

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    assert np.isclose(yerr[0][0], 2.7737755982632654e41, rtol=1e-4)
    assert np.isclose(yerr[1][0], 3.681894356120629e41, rtol=1e-4)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_lsun(mockerrorbar: mock.MagicMock) -> None:
    """A bolometric reference light curve must be converted to Lsun when the axis is in Lsun.

    Before the conversion was applied, the reference data stayed in erg/s and was drawn a factor of
    Lsun_to_erg_per_s above the ARTIS light curves on the same axis.
    """
    at.lightcurve.plot(
        argsraw=[], modelpath=[REFLIGHTCURVE, modelpath], Lsun=True, outputfile=outputpath / "lc_reflc_Lsun.pdf"
    )

    _arr_time_d, arr_lum = get_reflightcurve_errorbar_call(mockerrorbar)

    assert np.isclose(arr_lum[0], 1.1246049739669314e42 / Lsun_to_erg_per_s, rtol=1e-4)
    assert np.isclose(arr_lum[0], 2.9393752586e8, rtol=1e-4)

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    assert np.isclose(yerr[0][0], 2.7737755982632654e41 / Lsun_to_erg_per_s, rtol=1e-4)
    assert np.isclose(yerr[1][0], 3.681894356120629e41 / Lsun_to_erg_per_s, rtol=1e-4)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_magnitude(mockerrorbar: mock.MagicMock) -> None:
    """A bolometric reference light curve must be converted to magnitudes when the axis is in magnitudes."""
    at.lightcurve.plot(
        argsraw=[], modelpath=[REFLIGHTCURVE, modelpath], magnitude=True, outputfile=outputpath / "lc_reflc_mag.pdf"
    )

    _arr_time_d, arr_mag = get_reflightcurve_errorbar_call(mockerrorbar)

    dflightcurve, _metadata = at.lightcurve.read_bol_reflightcurve_data(REFLIGHTCURVE)
    lum_lsun = dflightcurve["luminosity_erg/s"].to_numpy() / Lsun_to_erg_per_s
    assert np.allclose(arr_mag, Mbol_sun - 2.5 * np.log10(lum_lsun), rtol=1e-10)
    assert np.isclose(arr_mag[0], -16.4306375858, rtol=1e-9)

    # the file gives a symmetric error in log10(luminosity), so both magnitude error bars are 2.5 times it
    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    expected_magerr = 2.5 * dflightcurve["log_lbol_err"].to_numpy()
    assert np.allclose(yerr[0], expected_magerr, rtol=1e-3)
    assert np.allclose(yerr[1], expected_magerr, rtol=1e-3)


def test_convert_lum_lsun_to_plotunits() -> None:
    """Deposition rates and reference data must use the same unit conversion as the ARTIS light curves."""
    lum_lsun = np.array([1.0, 1e8])

    assert np.allclose(at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, "Lsun"), lum_lsun)

    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, "erg/s"), lum_lsun * Lsun_to_erg_per_s
    )

    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, "mag"), [Mbol_sun, Mbol_sun - 20.0]
    )


def test_convert_lum_ergs_to_plotunits() -> None:
    """The erg/s axis must pass the reference data through untouched, with no round trip through Lsun."""
    lum_erg_per_s = np.array([1.1246049739669314e42, 3.0e41])

    assert (at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum_erg_per_s, "erg/s") == lum_erg_per_s).all()

    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum_erg_per_s, "Lsun"),
        lum_erg_per_s / Lsun_to_erg_per_s,
    )


def test_convert_lum_to_plotunits_nonpositive() -> None:
    """A zero or negative luminosity has no magnitude, and must not raise a numpy warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(np.array([0.0, -1.0, 1.0]), "mag")

    assert np.isinf(result[0])
    assert np.isnan(result[1])
    assert np.isclose(result[2], Mbol_sun)


def test_get_reflightcurve_yerr_magnitude() -> None:
    """Magnitude error bars must reach the magnitudes of the brightest and faintest luminosity bounds.

    matplotlib draws the bar from y - yerr[0] to y + yerr[1], and a brighter (larger) luminosity is a smaller
    magnitude, so the two rows are asymmetric and must not be swapped.
    """
    lum = np.array([1e42, 1e42, 1e42])
    errplus = np.array([1e42, 0.0, 1e42])
    errminus = np.array([0.9e42, 0.0, 2e42])  # the last error bar reaches past zero luminosity

    yerr, unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "mag")
    mag = at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum, "mag")

    assert np.isclose(yerr[0][0], 2.5 * np.log10(2.0))  # brighter by a factor of two
    assert np.isclose(yerr[1][0], 2.5)  # fainter by a factor of ten
    assert yerr[1][0] > yerr[0][0], "the fainter (upper) row must not be swapped with the brighter (lower) row"

    assert np.isclose(mag[0] - yerr[0][0], Mbol_sun - 2.5 * np.log10(2e42 / Lsun_to_erg_per_s))
    assert np.isclose(mag[0] + yerr[1][0], Mbol_sun - 2.5 * np.log10(0.1e42 / Lsun_to_erg_per_s))

    assert np.isclose(yerr[0][1], 0.0)  # a zero error stays zero on both sides
    assert np.isclose(yerr[1][1], 0.0)

    # a bar reaching zero luminosity has no faintest magnitude. A NaN there makes matplotlib drop the whole
    # bar, so the faint side has no length and the point is flagged for an arrow instead
    assert np.allclose(unbounded, [False, False, True])
    assert np.isfinite(yerr[1][2])
    assert np.isclose(yerr[1][2], 0.0)
    assert np.isclose(yerr[0][2], 2.5 * np.log10(2.0))


def test_get_reflightcurve_yerr_nonpositive_luminosity_draws_no_bar() -> None:
    """A luminosity at or below zero has no magnitude, so it must not reach matplotlib as an infinite bar.

    matplotlib adds the error to the value, and inf + -inf warns and yields NaN.
    """
    lum = np.array([0.0, -1e42, 1e42])
    errminus = np.array([1e41, 1e41, 1e41])
    errplus = np.array([1e41, 1e41, 1e41])

    yerr, unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "mag")

    assert np.all(np.isfinite(yerr[0])), yerr[0]
    assert np.all(np.isfinite(yerr[1])), yerr[1]
    assert np.allclose(yerr[0][:2], 0.0)
    assert np.allclose(yerr[1][:2], 0.0)
    # a point with no magnitude is not an open faint side, it is not drawn at all
    assert not unbounded[:2].any()
    assert yerr[0][2] > 0.0


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_unbounded_bar_keeps_the_bright_half_and_gets_an_arrow(
    mockerrorbar: mock.MagicMock, tmp_path: Path
) -> None:
    """An error bar reaching zero luminosity keeps its bright half, plus an arrow pointing at the faint side.

    Putting NaN in the faint row instead makes matplotlib collapse the segment to a single vertex, so the
    finite bright half disappears with it. The arrow direction is fixed at the errorbar call, and a
    magnitude axis is inverted only after every series is drawn, so the default direction is the wrong one.
    """
    reffile = tmp_path / "unbounded_reflightcurve.txt"
    reffile.write_text(
        "#time_days luminosity_erg/s luminosity_errminus_erg/s luminosity_errplus_erg/s\n"
        "2.0 1e42 5e41 5e41\n"
        "3.0 1e42 2e42 5e41\n",
        encoding="utf-8",
    )

    _fig, axis = plt.subplots()
    at.lightcurve.plotlightcurve.plot_bol_reflightcurve(axis, reffile, "mag", color="0.0")
    at.lightcurve.plotlightcurve.invert_magnitude_yaxis(axis)

    barcall, arrowcall = mockerrorbar.call_args_list
    yerr = barcall[1]["yerr"]
    assert np.all(np.isfinite(yerr[0]))
    assert np.all(np.isfinite(yerr[1]))
    assert yerr[0][1] > 0.0, "the bright half of the unbounded bar must still be drawn"
    assert np.isclose(yerr[1][1], 0.0), "the faint half has no length, so matplotlib keeps the bright one"

    assert arrowcall[1]["lolims"] is True
    assert np.allclose(arrowcall[0][1], [3.0]), "only the unbounded point gets an arrow"

    ymin, ymax = axis.get_ylim()
    assert ymin > ymax, "this test only means anything on an inverted magnitude axis"
    arrowcontainer = axis.containers[-1]
    assert isinstance(arrowcontainer, ErrorbarContainer)
    carets = [line.get_marker() for line in arrowcontainer.lines[1]]
    assert carets == [mplmarkers.CARETDOWNBASE], carets
    # a downward caret is drawn towards the lower screen positions, which on this axis are the fainter
    # (larger) magnitudes, so the arrow marks the side of the bar that has no end
    screen_y_of_fainter = axis.transData.transform((0.0, max(ymin, ymax)))[1]
    screen_y_of_brighter = axis.transData.transform((0.0, min(ymin, ymax)))[1]
    assert screen_y_of_fainter < screen_y_of_brighter


def test_bol_reflightcurve_empty_label_stays_out_of_the_legend() -> None:
    """An explicit -label "" keeps a positional reference curve out of the legend.

    Only a missing label takes the file metadata: matplotlib gives no legend entry to a series
    labelled "", which is how one is suppressed.
    """
    _fig, axis = plt.subplots()
    plotlabel = at.lightcurve.plotlightcurve.plot_bol_reflightcurve(
        axis, "AT2017gfo_smarttetal2017.txt", "erg/s", color="0.0", label=""
    )
    assert not plotlabel

    plotlabel = at.lightcurve.plotlightcurve.plot_bol_reflightcurve(
        axis, "AT2017gfo_smarttetal2017.txt", "erg/s", color="0.0", label=None
    )
    assert plotlabel
    assert plotlabel != "None"


def test_get_reflightcurve_yerr_scaling() -> None:
    """Without magnitudes the error bar sizes are scaled the same way as the luminosities."""
    lum = np.array([1e42, 2e42])
    errminus = np.array([1e41, 3e41])
    errplus = np.array([2e41, 4e41])

    yerr, unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "erg/s")
    assert np.allclose(yerr[0], errminus)
    assert np.allclose(yerr[1], errplus)
    assert not unbounded.any(), "a luminosity axis reaches zero, so no bar is open-ended"

    yerr, _unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, "Lsun")
    assert np.allclose(yerr[0], errminus / Lsun_to_erg_per_s)
    assert np.allclose(yerr[1], errplus / Lsun_to_erg_per_s)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_magnitude_asymmetric(mockerrorbar: mock.MagicMock) -> None:
    """A reference curve with asymmetric luminosity errors gets asymmetric magnitude error bars.

    AT2017gfo_smarttetal2017.txt has errors that are symmetric in log10(luminosity), so its two magnitude
    error bar rows are equal and cannot catch a swapped or symmetric-only implementation.
    """
    reflightcurve = "AT2017gfo_waxmanetal2018.txt"
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[reflightcurve, modelpath],
        magnitude=True,
        outputfile=outputpath / "lc_reflc_mag_asym.pdf",
    )

    dflightcurve, _metadata = at.lightcurve.read_bol_reflightcurve_data(reflightcurve)
    lum_erg_per_s = dflightcurve["luminosity_erg/s"].to_numpy()
    errminus = dflightcurve["luminosity_errminus_erg/s"].to_numpy()
    errplus = dflightcurve["luminosity_errplus_erg/s"].to_numpy()

    # computed from the luminosity ratios rather than from the function under test, so that swapping the two
    # rows of get_reflightcurve_yerr fails here instead of swapping the expectation with it
    expected = [
        2.5 * np.log10((lum_erg_per_s + errplus) / lum_erg_per_s),
        2.5 * np.log10(lum_erg_per_s / (lum_erg_per_s - errminus)),
    ]

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    assert np.allclose(yerr[0], expected[0])
    assert np.allclose(yerr[1], expected[1])
    assert not np.allclose(yerr[0], yerr[1], rtol=1e-2), "this file must exercise the asymmetric branch"


@pytest.mark.parametrize("lumunit", ["erg/s", "Lsun", "magnitude"])
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdeposition(mockplot: mock.MagicMock, lumunit: str) -> None:
    """Deposition curves are drawn in the y axis units, in every unit mode.

    plot_deposition_thermalisation() appends a suffix to the caller's label and picks its own linestyle and
    colour, so those keys must not also arrive in the **plotkwargs splat, which used to raise
    "got multiple values for keyword argument".
    """
    plotkwargs: dict[str, t.Any] = {} if lumunit == "erg/s" else {lumunit: True}

    with warnings.catch_warnings():
        # a zero deposition rate has no magnitude, but it must not warn
        warnings.simplefilter("error", RuntimeWarning)
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath_classic_3d],
            plotdeposition=True,
            outputfile=outputpath / f"lc_deposition_{lumunit.replace('/', '')}.pdf",
            **plotkwargs,
        )

    gammadep_lsun = at.get_deposition(modelpath_classic_3d).collect()["gammadep_Lsun"].to_numpy()
    with np.errstate(divide="ignore"):
        expected = {
            "erg/s": gammadep_lsun * Lsun_to_erg_per_s,
            "Lsun": gammadep_lsun,
            "magnitude": Mbol_sun - 2.5 * np.log10(gammadep_lsun),
        }[lumunit]

    gammacurves = [
        callargs
        for callargs in mockplot.call_args_list
        if str(callargs.kwargs.get("label", "")).endswith(r"$\dot{E}_{dep,\gamma}$")
    ]
    assert len(gammacurves) == 1
    assert np.allclose(np.asarray(gammacurves[0][0][2], dtype=float), expected, equal_nan=True)


def test_plotthermalisation() -> None:
    """The thermalisation curves share plot_deposition_thermalisation's kwargs handling."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotthermalisation=True,
        outputfile=outputpath / "lc_thermalisation.pdf",
    )


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_reflightcurves_arg_draws_error_bars(mockerrorbar: mock.MagicMock) -> None:
    """-reflightcurves must draw the same curve as a positional reference file, error bars included.

    This branch used to scatter the points with no uncertainties while the positional branch drew error bars
    from the same file, and it kept its own copy of the unit conversion.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        reflightcurves=[REFLIGHTCURVE],
        magnitude=True,
        outputfile=outputpath / "lc_reflightcurves_arg.pdf",
    )

    dflightcurve, _metadata = at.lightcurve.read_bol_reflightcurve_data(REFLIGHTCURVE)
    lum_erg_per_s = dflightcurve["luminosity_erg/s"].to_numpy()

    _arr_time_d, arr_mag = get_reflightcurve_errorbar_call(mockerrorbar)
    assert np.allclose(arr_mag, Mbol_sun - 2.5 * np.log10(lum_erg_per_s / Lsun_to_erg_per_s))

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    expected, _unbounded = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(
        lum_erg_per_s,
        dflightcurve["luminosity_errminus_erg/s"].to_numpy(),
        dflightcurve["luminosity_errplus_erg/s"].to_numpy(),
        "mag",
    )
    assert np.allclose(yerr[0], expected[0])
    assert np.allclose(yerr[1], expected[1])


def test_get_plot_lum_unit_and_column() -> None:
    """The unit choice and the light curve column to plot come from one place."""
    for magnitude, lsun, expected_unit, expected_col in (
        (True, False, "mag", "mag"),
        (True, True, "mag", "mag"),  # magnitude wins over Lsun
        (False, True, "Lsun", "luminosity_Lsun"),
        (False, False, "erg/s", "luminosity_erg/s"),
    ):
        args = argparse.Namespace(magnitude=magnitude, Lsun=lsun)
        lumunit = at.lightcurve.plotlightcurve.get_plot_lum_unit(args)
        assert lumunit == expected_unit
        assert at.lightcurve.plotlightcurve.get_plot_lum_column(lumunit) == expected_col


@pytest.mark.parametrize(
    ("dirbins", "colour_evolution"),
    [([0], ["B-V"]), ([0, 1], ["B-V"]), ([0], ["U-B", "B-V"]), ([0, 1], ["U-B", "B-V"])],
)
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_yaxis_is_inverted(
    mockplot: mock.MagicMock, dirbins: list[int], colour_evolution: list[str]
) -> None:
    """The colour axis points downwards however many direction bins and filter pairs are drawn.

    invert_yaxis() toggles rather than sets, and the subplots share one y axis, so inverting once per plotted
    line left the axis the right way up whenever an even number of lines was drawn.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        colour_evolution=colour_evolution,
        plotviewingangle=dirbins,
        timemin=5,
        timemax=8,
        outputfile=outputpath / "lc_colourevolution_inverted.pdf",
    )

    axes = {id(callargs[0][0]): callargs[0][0] for callargs in mockplot.call_args_list}
    assert axes
    for axis in axes.values():
        ymin, ymax = axis.get_ylim()
        assert ymin > ymax, f"the colour axis is not inverted for {len(dirbins)} bins and {colour_evolution}"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_colour_evolution_plot_dirbin_colours_avoid_the_model_colours(mockplot: mock.MagicMock) -> None:
    """A direction bin must not be given the colour that a whole model was assigned.

    The direction bin palette and the per-model colours are two independent sources drawn on one set of axes,
    and both used to start at the first colour of the matplotlib cycle.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath, modelpath_classic_3d],
        colour_evolution=["B-V"],
        plotviewingangle=[0, 1],
        timemin=5,
        timemax=8,
        outputfile=outputpath / "lc_colourevolution_colours.pdf",
    )

    colors = [mplcolors.to_hex(callargs.kwargs["color"]) for callargs in mockplot.call_args_list]
    assert len(colors) == 3
    assert len(set(colors)) == len(colors), colors


@pytest.mark.parametrize("plotalpha", [False, True])
@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotalphadeposition_draws_the_alpha_curves(mockplot: mock.MagicMock, plotalpha: bool) -> None:
    """The alpha decay curves are drawn only when -plotalphadeposition asks for them."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotdeposition=not plotalpha,
        plotalphadeposition=plotalpha,
        outputfile=outputpath / "lc_alphadep.pdf",
    )

    labels = [str(callargs.kwargs.get("label")) for callargs in mockplot.call_args_list]
    assert any(r"\alpha" in label for label in labels) == plotalpha, labels
    # the gamma and beta curves are drawn either way, so the alpha rates can be compared against them
    assert any(r"\gamma" in label for label in labels), labels


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_deposition_curves_do_not_reuse_a_model_colour(mockplot: mock.MagicMock) -> None:
    """The deposition curves take their colours from the axis cycle, which the model colours were removed from.

    A hardcoded colour would collide with whichever model get_series_colors happened to give that colour to.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d, modelpath_classic_3d],
        plotalphadeposition=True,
        timemin=3,
        timemax=8,
        outputfile=outputpath / "lc_deposition_colours.pdf",
    )

    lines = [
        (str(callargs.kwargs.get("label")), color)
        for callargs in mockplot.call_args_list
        if (color := callargs.kwargs.get("color")) is not None
    ]
    modelcolors = {mplcolors.to_hex(color) for label, color in lines if "dot" not in label}
    depositioncolors = {mplcolors.to_hex(color) for label, color in lines if "dot" in label}

    assert modelcolors
    assert depositioncolors
    assert not modelcolors & depositioncolors, f"{modelcolors} and {depositioncolors} share a colour"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdeposition_does_not_inherit_the_light_curve_style(mockplot: mock.MagicMock) -> None:
    """The deposition curves take their style from the command line, not from the last direction bin drawn.

    plot_artis_lightcurve mutates its plot kwargs per direction bin, and those used to ride into the
    deposition curves, drawing them semi-transparent and above the light curves for no stated reason.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotdeposition=True,
        plotviewingangle=[0, 1],
        colorbarphi=True,
        outputfile=outputpath / "lc_deposition_style.pdf",
    )

    depositionkwargs = [
        callargs.kwargs for callargs in mockplot.call_args_list if r"\dot{E}" in str(callargs.kwargs.get("label"))
    ]
    assert depositionkwargs
    for kwargs in depositionkwargs:
        assert "alpha" not in kwargs
        assert "zorder" not in kwargs


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_plotdeposition_is_drawn_once_per_model(mockplot: mock.MagicMock) -> None:
    """A model draws its deposition rates once, however many light curve series it contributes.

    plot_artis_lightcurve runs once per escape type and once per top nuclide, and the deposition rates were
    drawn on every run: the gamma-ray series repeated all three curves in new colours, labelled with the
    gamma-ray series name even though the deposition rates have nothing to do with the escape type.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        gamma=True,
        rpkt=True,
        plotdeposition=True,
        outputfile=outputpath / "lc_deposition_gamma_and_rpkt.pdf",
    )

    depositionlabels = [
        str(callargs.kwargs.get("label"))
        for callargs in mockplot.call_args_list
        if r"\dot{E}" in str(callargs.kwargs.get("label"))
    ]
    assert depositionlabels
    assert len(depositionlabels) == len(set(depositionlabels)), depositionlabels
    # the deposition rates are a model property, so they keep the model name rather than the gamma series name
    assert not any(r"$\gamma$ $\dot{E}" in label for label in depositionlabels), depositionlabels


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_lightcurve_plot_one_sided_time_limit_keeps_the_data_in_view(mockplot: mock.MagicMock) -> None:
    """A -timemin with no -timemax leaves the right hand side fitted to the light curve.

    Setting a one-sided limit before anything is drawn turns autoscaling off, freezing the other side at the
    default 0-1 view instead of the range of the data.
    """
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, timemin=250.0, outputfile=outputpath / "lc_timemin_only.pdf")

    axis = mockplot.call_args_list[0][0][0]
    xmin, xmax = axis.get_xlim()
    assert np.isclose(xmin, 250.0)
    assert xmax > 300.0, xmax


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_gamma_lightcurve_magnitude_ylabel(mockylabel: mock.MagicMock) -> None:
    """A gamma-ray light curve in magnitudes is not a bolometric magnitude, and must not be labelled as one."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        gamma=True,
        rpkt=False,
        magnitude=True,
        outputfile=outputpath / "lc_gamma_mag.pdf",
    )

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert ylabels == [r"Absolute $\gamma$-ray Magnitude"]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_averaged_direction_bin_magnitude_is_rebuilt(mockplot: mock.MagicMock) -> None:
    """Averaging direction bins averages the luminosity, so the magnitude must be rebuilt from the result.

    average_direction_bins takes the mean of every column, and a magnitude is logarithmic: the mean of the
    magnitudes is not the magnitude of the mean luminosity, and one dark bin sends the mean to inf.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        magnitude=True,
        plotviewingangle=[0],
        average_over_phi_angle=True,
        outputfile=outputpath / "lc_averagedphi_mag.pdf",
    )

    lcpath = at.firstexisting("light_curve_res.out", folder=modelpath_classic_3d, tryzipped=True)
    averaged = at.average_direction_bins(at.lightcurve.readfile(lcpath), overangle="phi")[0].collect()
    lum_lsun_by_time = dict(zip(averaged["time_days"], averaged["luminosity_Lsun"], strict=True))
    meanofmags_by_time = dict(zip(averaged["time_days"], averaged["mag"], strict=True))

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    arr_mag = np.array(mockplot.call_args_list[0][0][2])
    assert arr_time_d.size > 0

    with np.errstate(divide="ignore"):
        expected = Mbol_sun - 2.5 * np.log10(np.array([lum_lsun_by_time[t_d] for t_d in arr_time_d]))
    assert np.allclose(arr_mag, expected, equal_nan=True)

    meanofmags = np.array([meanofmags_by_time[t_d] for t_d in arr_time_d])
    assert not np.allclose(arr_mag, meanofmags, equal_nan=True), "the stale mean of the magnitudes was plotted"
    assert np.isfinite(arr_mag).sum() > np.isfinite(meanofmags).sum(), "a dark bin still poisons the average"


def test_colour_evolution_plot_leaves_the_filter_arg_alone() -> None:
    """The band selection must not be written back onto args.filter, which main() dispatches on.

    A caller that reuses one Namespace would otherwise take the band light curve branch on the second call
    and silently draw a different plot.
    """
    parser = argparse.ArgumentParser()
    at.lightcurve.addargs(parser)
    args = parser.parse_args([])
    args.modelpath = [modelpath]
    args.colour_evolution = ["U-B", "B-V"]
    args.outputfile = outputpath

    at.lightcurve.plot(args=args)

    assert not args.filter


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_lightcurve_ylabel_names_deposition_only_when_it_is_drawn(mockylabel: mock.MagicMock) -> None:
    """Asking for deposition rates is not drawing them, so a run with no ARTIS model must not claim them."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[REFLIGHTCURVE],
        plotdeposition=True,
        outputfile=outputpath / "lc_reflc_only_deposition.pdf",
    )

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert all(r"\dot{E}" not in ylabel for ylabel in ylabels), ylabels


def run_set_scatterplot_plot_params(
    magnitudes: Sequence[float], ymin: float | None, ymax: float | None
) -> tuple[float, float, float]:
    """Plot the magnitudes, apply the shared scatter plot setup, and return xmax, ymin, ymax."""
    fig, axis = plt.subplots()
    axis.plot([1.0, 10.0], list(magnitudes))
    args = argparse.Namespace(colouratpeak=False, ymin=ymin, ymax=ymax, colorbarcostheta=False, colorbarphi=False)

    viewingangleanalysis.set_scatterplot_plot_params(fig, axis, args)

    ymin, ymax = axis.get_ylim()
    return axis.get_xlim()[1], ymin, ymax


def test_set_scatterplot_plot_params_without_xmin() -> None:
    """The light curve parser names its range -timemin/-timemax, so the scatter setup must not read args.xmin.

    Reading args.xmin raised AttributeError and made every --make_viewing_angle_peakmag_* run fail.
    """
    xmax, ymin, ymax = run_set_scatterplot_plot_params([15.0, 19.0], ymin=None, ymax=None)

    assert ymin > ymax, "the magnitude axis must point downwards"
    # neither side falls back to the default 0-1 view when no limit was given
    assert xmax >= 10.0
    assert ymin >= 19.0


def test_readfile_rebuilds_the_magnitude_after_averaging() -> None:
    """The loader owns the averaging, so the magnitude cannot be left as the mean of the magnitudes.

    Averaging is linear and a magnitude is logarithmic, so a single dark bin sends the arithmetic mean of
    the magnitudes to inf while the magnitude of the averaged luminosity stays finite.
    """
    lcpath = at.firstexisting("light_curve_res.out", folder=modelpath_classic_3d, tryzipped=True)

    averaged = at.lightcurve.readfile(lcpath, average_over_phi=True)[0].collect()
    stalemean = at.average_direction_bins(at.lightcurve.readfile(lcpath), overangle="phi")[0].collect()

    with np.errstate(divide="ignore"):
        magofmeanlum = Mbol_sun - 2.5 * np.log10(averaged["luminosity_Lsun"].to_numpy())

    assert np.allclose(averaged["mag"].to_numpy(), magofmeanlum, equal_nan=True)
    assert not np.allclose(averaged["mag"].to_numpy(), stalemean["mag"].to_numpy(), equal_nan=True)


def test_color_arg_rejects_a_colour_matplotlib_cannot_parse() -> None:
    """A mistyped colour must be named by the argument that took it, not by a colour cycle helper."""
    parser = argparse.ArgumentParser()
    at.lightcurve.addargs(parser)

    assert parser.parse_args(["-color", "tab:blue", "#1F77B4"]).color == ["tab:blue", "#1F77B4"]

    for argname in ("-color", "-refspeccolors"):
        with pytest.raises(SystemExit):
            parser.parse_args([argname, "notacolour"])


def test_transparent_series_colour_leaves_the_cycle_alone() -> None:
    """A series drawn in "none" holds no colour, so it must not remove black from the palette."""
    assert not at.plottools.get_assigned_colors(["none"])
    assert mplcolors.to_hex("#000000") in {
        mplcolors.to_hex(color) for color in at.plottools.get_unused_colors(["#000000", "#ffffff"], ["none"])
    }


@pytest.mark.parametrize("plottype", ["bolometric", "band", "colour_evolution"])
@mock.patch.object(mplax.Axes, "set_xlim", side_effect=mplax.Axes.set_xlim, autospec=True)
def test_time_limits_reach_every_figure(mockxlim: mock.MagicMock, plottype: str) -> None:
    """-timemin/-timemax are this command's x limits, so every figure it draws must honour them."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath],
        timemin=260,
        timemax=300,
        outputfile=outputpath,
        filter=["B"] if plottype == "band" else [],
        colour_evolution=["B-V"] if plottype == "colour_evolution" else [],
    )

    limits = [callargs[0][1:3] for callargs in mockxlim.call_args_list if len(callargs[0]) > 2]
    assert (260, 300) in limits, limits


def test_nonpositive_limit_on_a_log_axis_is_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    """A log axis cannot show a limit at or below zero, so it is dropped with a message naming the argument."""
    _fig, axis = plt.subplots()
    axis.plot([1.0, 10.0], [1e40, 1e44])
    args = argparse.Namespace(logscaley=True, ymin=0.0, ymax=1e45, xmin=None, xmax=None)

    at.plottools.set_axis_properties(axis, args)

    # a warning goes to the standard error, thus --quiet keeps it
    assert "-ymin" in capsys.readouterr().err
    ymin, ymax = axis.get_ylim()
    assert ymin > 0.0, "the data must stay in view rather than the axis freezing at a rejected limit"
    assert np.isclose(ymax, 1e45)


def test_viewing_angle_scatter_needs_the_angle_averaged_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Writing the per-direction-bin data and plotting it are two runs, so asking for both must say so."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="angle-averaged"):
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath],
            filter=["B"],
            timemin=250,
            timemax=300,
            outputfile=tmp_path,
            save_viewing_angle_peakmag_risetime_delta_m15_to_file=True,
            make_viewing_angle_peakmag_risetime_scatter_plot=True,
        )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_band_reflightcurve_is_drawn_once_per_panel(mockplot: mock.MagicMock) -> None:
    """A band reference file is read and drawn once for the whole figure, not once per band.

    plot_lightcurve_from_refdata already draws every band of the file onto its own panel, so calling it
    from inside the band loop drew each reference curve once per band. It also asserted a single axes,
    which a figure with more than one band never has, so -filter B V -reflightcurves raised AssertionError.
    """
    refdata = pl.DataFrame({
        "band": ["B", "B", "V", "V"],
        "time": [260.0, 280.0, 260.0, 280.0],
        "magnitude": [-13.0, -12.5, -13.2, -12.7],
    })

    with mock.patch.object(
        at.lightcurve, "read_reflightcurve_band_data", return_value=(refdata, {"label": "refband"})
    ) as mockread:
        at.lightcurve.plot(
            argsraw=[],
            modelpath=[modelpath],
            filter=["B", "V"],
            reflightcurves=["fakeref.dat"],
            outputfile=outputpath / "lc_band_ref_once.pdf",
        )

    assert mockread.call_count == 1, "the reference file must be read once, not once per band"

    reflines = [callargs for callargs in mockplot.call_args_list if callargs[1].get("label") == "refband"]
    assert len(reflines) == 2, "one reference curve per band panel"
    assert {tuple(np.asarray(callargs[0][1])) for callargs in reflines} == {(260.0, 280.0)}


@mock.patch.object(at.plottools, "get_next_color", side_effect=at.plottools.get_next_color, autospec=True)
def test_alpha_deposition_colour_is_taken_only_when_it_is_drawn(mockcolor: mock.MagicMock) -> None:
    """A colour taken but not drawn steps every later series along the cycle for nothing."""
    for plotalphadeposition, expected_extra in ((False, 0), (True, 1)):
        _fig, axis = plt.subplots()
        args = argparse.Namespace(
            plotdeposition=True, plotalphadeposition=plotalphadeposition, plotthermalisation=False,
            magnitude=False, Lsun=False,
        )  # fmt: skip

        mockcolor.reset_mock()
        at.lightcurve.plotlightcurve.plot_deposition_thermalisation(axis, None, modelpath_classic_3d, "modelname", args)

        # one colour skipped plus gamma and beta, and the alpha colour only when its curves are drawn
        assert mockcolor.call_count == 3 + expected_extra


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_band_plot_first_dirbin_takes_the_color_arg(mockplot: mock.MagicMock) -> None:
    """A -color value is removed from the cycle for the model, so one of its lines has to use it.

    The band figure gave every line of a multi-bin model a cycle colour, so the requested colour was
    reserved and then drawn nowhere. The bolometric figure has always given it to the first bin.
    """
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        filter=["B"],
        plotviewingangle=[0, 1],
        color=["magenta"],
        timemin=5,
        timemax=8,
        outputfile=outputpath / "lc_band_dirbin_color.pdf",
    )

    colors = [
        mplcolors.to_hex(callargs[1]["color"]) for callargs in mockplot.call_args_list if callargs[1].get("color")
    ]
    assert mplcolors.to_hex("magenta") in colors, colors


def test_scatterplot_magnitude_axis_stays_inverted_with_limits() -> None:
    """set_ylim re-sorts the pair it is given, so an inversion applied before it is lost."""
    _xmax, ymin, ymax = run_set_scatterplot_plot_params([-19.0, -15.0], ymin=-20.0, ymax=-14.0)

    assert ymin > ymax, "the magnitude axis must point downwards even when -ymin/-ymax are given"
    assert np.isclose(ymin, -14.0)
    assert np.isclose(ymax, -20.0)


def test_lightcurve_print_data_survives_quiet(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """--print_data prints the plotted data, thus --quiet must keep it.

    addarg_quiet names the arguments whose product is the standard output, and run_command reads them.
    Only the dispatcher applies the redirection, thus the test gives a command line and not a keyword.
    """
    import artistools.__main__

    common = ["plotlightcurves", str(modelpath), "-o", str(tmp_path)]
    artistools.__main__.main(argsraw=[*common, "--print_data", "--quiet"])
    withdata = capsys.readouterr().out

    artistools.__main__.main(argsraw=[*common, "--quiet"])
    withoutdata = capsys.readouterr().out

    assert not withoutdata, "--quiet alone must hide the progress messages"
    assert withdata, "print_product writes the product, thus --quiet must keep it"

    # a script reads the product, thus no progress message may stand around it
    assert "====>" not in withdata
    assert "Reading " not in withdata
    assert withdata.lstrip().startswith("shape:")


def test_lightcurve_reference_data_takes_a_day_range(tmp_path: Path) -> None:
    """A range in days needs no timestep data, thus reference data alone still takes -timedays.

    apply_time_range_args gave the first path to get_time_range, which searches for the files of an
    ARTIS model. A reference light curve holds none, thus the command stopped before it drew anything.
    """
    at.lightcurve.plot(argsraw=["AT2017gfo_smarttetal2017.txt", "-timedays", "2-5", "-o", str(tmp_path / "ref.pdf")])

    assert (tmp_path / "ref.pdf").is_file()


def test_lightcurve_day_range_of_a_model_clamps_to_its_timesteps() -> None:
    """A model in the list gives the timestep times, thus the range still clamps to a timestep edge."""
    import argparse

    from artistools.lightcurve.plotlightcurve import apply_time_range_args

    withmodel = argparse.Namespace(timestep=None, timedays="260-300", timemin=None, timemax=None)
    apply_time_range_args(withmodel, [modelpath])

    reference = argparse.Namespace(timestep=None, timedays="260-300", timemin=None, timemax=None)
    apply_time_range_args(reference, ["AT2017gfo_smarttetal2017.txt"])

    # the model clamps to the start and the end of a timestep, and the reference data take the range
    assert withmodel.timemin > 260.0
    assert withmodel.timemax < 300.0
    assert np.isclose(reference.timemin, 260.0)
    assert np.isclose(reference.timemax, 300.0)


CLASSIC1DPATH = at.get_path("testdata") / "test-classicmode_1d"


@pytest.mark.skipif(not CLASSIC1DPATH.is_dir(), reason="run tests/data/setuptestdata.sh for the 1D classic model")
def test_lightcurve_timestep_must_mean_the_same_days_for_every_model() -> None:
    """A timestep names different days on a different timestep grid.

    The plot holds one time axis, thus the days of the first model filtered every model, and the second
    curve showed another timestep without a word. Two grids that disagree now stop the command, and
    -timedays serves both because a day means the same everywhere.
    """
    import argparse

    from artistools.lightcurve.plotlightcurve import apply_time_range_args

    def build(timestep: str | None, timedays: str | None) -> argparse.Namespace:
        return argparse.Namespace(timestep=timestep, timedays=timedays, timemin=None, timemax=None)

    # one grid, or the same grid twice, resolves the timestep as before
    sameargs = build("40", None)
    apply_time_range_args(sameargs, [modelpath, modelpath])
    assert sameargs.timemin is not None

    with pytest.raises(SystemExit) as excinfo:
        apply_time_range_args(build("40", None), [modelpath, CLASSIC1DPATH])
    assert excinfo.value.code == 1

    # a range in days means the same for every model, thus it needs no agreement between the grids
    daysargs = build(None, "260-300")
    apply_time_range_args(daysargs, [modelpath, CLASSIC1DPATH])
    assert daysargs.timemin is not None


def test_lightcurve_timestep_takes_a_path_that_names_a_file(tmp_path: Path) -> None:
    """A path can name the light curve file of a run, and -timestep must read the folder of that run.

    get_time_range reads timesteps.out or input.txt of a run. The file went to it as it came, thus
    "light_curve.out -timestep 20" asked for light_curve.out/input.txt and stopped.
    """
    outputfile = tmp_path / "lc.pdf"
    at.lightcurve.plot(argsraw=[str(modelpath / "light_curve.out"), "-timestep", "20", "-o", str(outputfile)])

    assert outputfile.is_file(), "the command must draw the light curve of the file"
