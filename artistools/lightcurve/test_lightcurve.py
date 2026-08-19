import argparse
import typing as t
import warnings
from operator import itemgetter
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import pytest
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at
from artistools.constants import Lsun_to_erg_per_s
from artistools.constants import Mbol_sun

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
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


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_lightcurve_plot_reflightcurves_keep_their_errorbars(mockerrorbar: t.Any) -> None:
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


@mock.patch.object(mplax.Axes, "set_ylabel", side_effect=mplax.Axes.set_ylabel, autospec=True)
def test_colour_evolution_plot_ylabel(mockylabel: t.Any) -> None:
    """A colour evolution plot must be labelled in delta magnitudes, not as a band magnitude.

    colour_evolution_plot assigns args.filter before asking for the labels, so reading the plot kind back off
    args labelled these axes "None Magnitude".
    """
    at.lightcurve.plot(argsraw=[], modelpath=modelpath, colour_evolution=["B-V"], outputfile=outputpath)

    ylabels = [callargs[0][1] for callargs in mockylabel.call_args_list]
    assert r"$\Delta$m" in ylabels, ylabels


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
def test_lightcurve_plot_reference_colors(mockplot: t.Any, mockerrorbar: t.Any) -> None:
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
def test_lightcurve_plot_reflightcurves_continue_the_greys(mockerrorbar: t.Any, mockscatter: t.Any) -> None:
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
def test_lightcurve_plot_colors_survive_a_skipped_model(mockplot: t.Any, mockerrorbar: t.Any) -> None:
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


def get_reflightcurve_errorbar_call(mockerrorbar: t.Any) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the times and luminosities that the reference light curve was drawn with."""
    assert mockerrorbar.call_count == 1
    callargs = mockerrorbar.call_args_list[0]

    return np.array(callargs[0][1]), np.array(callargs[0][2])


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_erg_per_s(mockerrorbar: t.Any) -> None:
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
def test_bol_reflightcurve_lsun(mockerrorbar: t.Any) -> None:
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
def test_bol_reflightcurve_magnitude(mockerrorbar: t.Any) -> None:
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

    args = argparse.Namespace(magnitude=False, Lsun=True)
    assert np.allclose(at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, args), lum_lsun)

    args = argparse.Namespace(magnitude=False, Lsun=False)
    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, args), lum_lsun * Lsun_to_erg_per_s
    )

    args = argparse.Namespace(magnitude=True, Lsun=False)
    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(lum_lsun, args), [Mbol_sun, Mbol_sun - 20.0]
    )


def test_convert_lum_ergs_to_plotunits() -> None:
    """The erg/s axis must pass the reference data through untouched, with no round trip through Lsun."""
    lum_erg_per_s = np.array([1.1246049739669314e42, 3.0e41])

    args = argparse.Namespace(magnitude=False, Lsun=False)
    assert (at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum_erg_per_s, args) == lum_erg_per_s).all()

    args = argparse.Namespace(magnitude=False, Lsun=True)
    assert np.allclose(
        at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum_erg_per_s, args),
        lum_erg_per_s / Lsun_to_erg_per_s,
    )


def test_convert_lum_to_plotunits_nonpositive() -> None:
    """A zero or negative luminosity has no magnitude, and must not raise a numpy warning."""
    args = argparse.Namespace(magnitude=True, Lsun=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = at.lightcurve.plotlightcurve.convert_lum_lsun_to_plotunits(np.array([0.0, -1.0, 1.0]), args)

    assert np.isinf(result[0])
    assert np.isnan(result[1])
    assert np.isclose(result[2], Mbol_sun)


def test_get_reflightcurve_yerr_magnitude() -> None:
    """Magnitude error bars must reach the magnitudes of the brightest and faintest luminosity bounds.

    matplotlib draws the bar from y - yerr[0] to y + yerr[1], and a brighter (larger) luminosity is a smaller
    magnitude, so the two rows are asymmetric and must not be swapped.
    """
    args = argparse.Namespace(magnitude=True, Lsun=False)
    lum = np.array([1e42, 1e42, 1e42])
    errplus = np.array([1e42, 0.0, 1e42])
    errminus = np.array([0.9e42, 0.0, 2e42])  # the last error bar reaches past zero luminosity

    yerr = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, args)
    mag = at.lightcurve.plotlightcurve.convert_lum_ergs_to_plotunits(lum, args)

    assert np.isclose(yerr[0][0], 2.5 * np.log10(2.0))  # brighter by a factor of two
    assert np.isclose(yerr[1][0], 2.5)  # fainter by a factor of ten
    assert yerr[1][0] > yerr[0][0], "the fainter (upper) row must not be swapped with the brighter (lower) row"

    assert np.isclose(mag[0] - yerr[0][0], Mbol_sun - 2.5 * np.log10(2e42 / Lsun_to_erg_per_s))
    assert np.isclose(mag[0] + yerr[1][0], Mbol_sun - 2.5 * np.log10(0.1e42 / Lsun_to_erg_per_s))

    assert yerr[0][1] == 0.0  # a zero error stays zero on both sides
    assert yerr[1][1] == 0.0

    # an error bar reaching zero luminosity has no faintest magnitude, but the brighter side is still defined
    assert np.isnan(yerr[1][2])
    assert not np.isnan(yerr[0][2])


def test_get_reflightcurve_yerr_scaling() -> None:
    """Without magnitudes the error bar sizes are scaled the same way as the luminosities."""
    lum = np.array([1e42, 2e42])
    errminus = np.array([1e41, 3e41])
    errplus = np.array([2e41, 4e41])

    args = argparse.Namespace(magnitude=False, Lsun=False)
    yerr = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, args)
    assert np.allclose(yerr[0], errminus)
    assert np.allclose(yerr[1], errplus)

    args = argparse.Namespace(magnitude=False, Lsun=True)
    yerr = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(lum, errminus, errplus, args)
    assert np.allclose(yerr[0], errminus / Lsun_to_erg_per_s)
    assert np.allclose(yerr[1], errplus / Lsun_to_erg_per_s)


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_bol_reflightcurve_magnitude_asymmetric(mockerrorbar: t.Any) -> None:
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
    expected = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(
        dflightcurve["luminosity_erg/s"].to_numpy(),
        dflightcurve["luminosity_errminus_erg/s"].to_numpy(),
        dflightcurve["luminosity_errplus_erg/s"].to_numpy(),
        argparse.Namespace(magnitude=True, Lsun=False),
    )

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    assert np.allclose(yerr[0], expected[0])
    assert np.allclose(yerr[1], expected[1])
    assert not np.allclose(yerr[0], yerr[1], rtol=1e-2), "this file must exercise the asymmetric branch"


@pytest.mark.parametrize("lumunit", ["erg/s", "Lsun", "magnitude"])
def test_plotdeposition(lumunit: str) -> None:
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


def test_plotthermalisation() -> None:
    """The thermalisation curves share plot_deposition_thermalisation's kwargs handling."""
    at.lightcurve.plot(
        argsraw=[],
        modelpath=[modelpath_classic_3d],
        plotthermalisation=True,
        outputfile=outputpath / "lc_thermalisation.pdf",
    )


@mock.patch.object(mplax.Axes, "errorbar", side_effect=mplax.Axes.errorbar, autospec=True)
def test_reflightcurves_arg_draws_error_bars(mockerrorbar: t.Any) -> None:
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
    args = argparse.Namespace(magnitude=True, Lsun=False)

    assert mockerrorbar.call_count == 1
    arr_mag = np.array(mockerrorbar.call_args_list[0][0][2])
    assert np.allclose(arr_mag, Mbol_sun - 2.5 * np.log10(lum_erg_per_s / Lsun_to_erg_per_s))

    yerr = mockerrorbar.call_args_list[0][1]["yerr"]
    expected = at.lightcurve.plotlightcurve.get_reflightcurve_yerr(
        lum_erg_per_s,
        dflightcurve["luminosity_errminus_erg/s"].to_numpy(),
        dflightcurve["luminosity_errplus_erg/s"].to_numpy(),
        args,
    )
    assert np.allclose(yerr[0], expected[0])
    assert np.allclose(yerr[1], expected[1])


def test_get_plot_lum_unit_and_column() -> None:
    """The unit choice and the light curve column to plot come from one place."""
    for magnitude, lsun, expected_unit, expected_col in [
        (True, False, "mag", "mag"),
        (True, True, "mag", "mag"),  # magnitude wins over Lsun
        (False, True, "Lsun", "luminosity_Lsun"),
        (False, False, "erg/s", "luminosity_erg/s"),
    ]:
        args = argparse.Namespace(magnitude=magnitude, Lsun=lsun)
        assert at.lightcurve.plotlightcurve.get_plot_lum_unit(args) == expected_unit
        assert at.lightcurve.plotlightcurve.get_plot_lum_column(args) == expected_col
