import argparse
import functools
import gzip
import lzma
import math
import re
import typing as t
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.testing as pltest
import pytest
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at
from artistools.spectra import core as atspectra
from artistools.spectra import plotspectra

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


@pytest.fixture
def patch_scan_packets(monkeypatch: pytest.MonkeyPatch) -> Callable[[pl.DataFrame], None]:
    """Return a function that makes the spectra code read the given packets in place of the packets of a model."""

    def patch_scan_packets(dfpackets: pl.DataFrame) -> None:
        def scan_packets(*_args: t.Any, **_kwargs: t.Any) -> tuple[int, pl.LazyFrame]:
            return 1, dfpackets.lazy()

        monkeypatch.setattr(atspectra, "scan_packets", scan_packets)

    return patch_scan_packets


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraplot(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, "sn2011fe_PTF11kly_20120822_norm.txt"],
        outputfile=tmp_path / "spectraplot.pdf",
        timemin=290,
        timemax=320,
        distmpc=1.0,
    )

    arr_lambda = np.array(mockplot.call_args[0][1])
    arr_f_lambda = np.array(mockplot.call_args[0][2])

    # the de-redshift of the reference spectrum at z = 0.0056 multiplies f_lambda by (1 + z)
    integral = np.trapezoid(y=arr_f_lambda, x=arr_lambda)
    assert np.isclose(integral, 5.903606996256828e-11, rtol=1e-6, atol=0.0)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_spectra_frompackets(mockplot: mock.MagicMock) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=Path(outputpath, "spectrum_from_packets.pdf"),
        timemin=290,
        timemax=320,
        frompackets=True,
    )

    arr_lambda = np.array(mockplot.call_args[0][1])
    arr_f_lambda = np.array(mockplot.call_args[0][2])

    integral = np.trapezoid(y=arr_f_lambda, x=arr_lambda)

    # the window is timesteps 44 to 72, and one more timestep at the low edge changes the integral by 2 per cent
    assert np.isclose(integral, 7.715075e-12, rtol=1e-3, atol=0.0)


def test_histogram_steps_are_on_the_packet_bin_edges() -> None:
    """The steps of --histogram lie on the edges of the packet bins, also for a frequency, which reverses the order."""
    lambda_bin_edges = np.geomspace(3000.0, 10000.0, 30)
    dfspectrum = atspectra.scan_spectra_from_packets(
        modelpath, 290.0, 330.0, lambda_bin_edges=lambda_bin_edges, directionbins=[-1]
    )[-1]
    for xunit in ("angstrom", "hz"):
        dfxy = atspectra.get_dfspectrum_x_y_with_units(
            dfspectrum, xunit=xunit, yvariable="flux", fluxdistance_mpc=1.0
        ).collect()
        lower, upper, _ = plotspectra.get_bin_x_edges(dfxy, xunit)
        expectededges = np.sort(atspectra.convert_angstroms_to_unit(lambda_bin_edges, xunit))
        assert np.allclose(lower, expectededges[:-1], rtol=1e-12, atol=0.0), xunit
        assert np.allclose(upper, expectededges[1:], rtol=1e-12, atol=0.0), xunit


def test_scan_spectra_from_packets_relnoise_is_the_noise_of_the_packet_energies() -> None:
    """The relative noise of a bin is sqrt(sum(e^2)) / sum(e) of the packets that arrive in it."""
    timelowdays, timehighdays = 290.0, 330.0
    lambda_bin_edges = np.geomspace(3000.0, 10000.0, 60)
    dfspectrum = atspectra.scan_spectra_from_packets(
        modelpath, timelowdays, timehighdays, lambda_bin_edges=lambda_bin_edges, directionbins=[-1], relnoise=True
    )[-1].collect()

    _, dfpackets = at.packets.scan_packets(modelpath, packet_type="TYPE_ESCAPE", escape_type="TYPE_RPKT")
    dfarrived = dfpackets.filter(pl.col("t_arrive_d").is_between(timelowdays, timehighdays)).collect()
    lambda_angstroms = at.constants.c_ang_per_s / dfarrived["nu_rf"].to_numpy()
    energies = dfarrived["e_rf"].cast(pl.Float64).to_numpy()
    energysums, _ = np.histogram(lambda_angstroms, bins=lambda_bin_edges, weights=energies)
    energysquaresums, _ = np.histogram(lambda_angstroms, bins=lambda_bin_edges, weights=energies**2)
    hasenergy = energysums > 0.0
    # the test needs empty bins, which have a null relnoise, and bins that hold packets
    assert 0 < hasenergy.sum() < len(energysums)

    relnoise = dfspectrum["relnoise"]
    assert relnoise.is_null().to_list() == (~hasenergy).tolist()
    assert np.allclose(
        relnoise.to_numpy()[hasenergy],
        np.sqrt(energysquaresums[hasenergy]) / energysums[hasenergy],
        rtol=1e-10,
        atol=0.0,
    )


def test_spectra_outputtext(tmp_path: Path) -> None:
    """-o names the folder of the spectrum files, and the command makes a folder that does not exist yet."""
    newfolder = tmp_path / "newfolder"
    at.spectra.plot(argsraw=[], specpath=modelpath, output_spectra=True, outputfile=newfolder)

    assert list(newfolder.glob("spectrum_ts*.txt"))


@mock.patch.object(mplax.Axes, "stackplot", side_effect=mplax.Axes.stackplot, autospec=True)
@pytest.mark.benchmark
def test_spectraemissionplot(mockstackplot: mock.MagicMock, tmp_path: Path) -> None:
    """The emission plot draws one stack above the axis and one absorption stack below it."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / "emission.pdf",
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        use_thermalemissiontype=True,
    )

    assert (tmp_path / "emission.pdf").is_file()
    emissionstack, absorptionstack = (call.args[2] for call in mockstackplot.call_args_list)
    assert len(emissionstack) == len(absorptionstack) > 1
    assert all(np.all(np.asarray(series) >= 0.0) for series in emissionstack)
    assert all(np.all(np.asarray(series) <= 0.0) for series in absorptionstack)
    assert any(np.any(np.asarray(series) > 0.0) for series in emissionstack)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_spectraemissionplot_nostack(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """--nostack draws the net spectrum in black, and one emission line and one absorption line for each series."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / "emission_nostack.pdf",
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        nostack=True,
        use_thermalemissiontype=True,
    )

    assert (tmp_path / "emission_nostack.pdf").is_file()
    netcalls = [call for call in mockplot.call_args_list if call.kwargs.get("color") == "black"]
    assert len(netcalls) == 1
    seriescalls = [call for call in mockplot.call_args_list if call.kwargs.get("color") != "black"]
    assert len(seriescalls) > 2
    assert len(seriescalls) % 2 == 0


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraabsorptionplot_draws_the_net_spectrum(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """--showabsorption alone draws the same net spectrum as --showemission.

    Before, a plot with no emission gave a net spectrum of zero.
    """
    netspectra = []
    for flag in ("--showemission", "--showabsorption"):
        mockplot.reset_mock()
        at.spectra.plot(
            argsraw=[flag],
            specpath=modelpath,
            outputfile=tmp_path / f"{flag}.pdf",
            timemin=290,
            timemax=320,
            use_thermalemissiontype=True,
        )
        (netcall,) = [call for call in mockplot.call_args_list if call.kwargs.get("color") == "black"]
        netspectra.append(np.asarray(netcall.args[2]))

    emissionnet, absorptionnet = netspectra
    assert np.any(emissionnet > 0.0)
    assert np.allclose(absorptionnet, emissionnet, rtol=1e-12, atol=0.0)


def test_spectra_get_spectrum() -> None:
    def check_spectrum(dfspectrumpkts: pl.DataFrame, expectedmax: float, expectedmean: float) -> None:
        assert math.isclose(max(dfspectrumpkts["f_lambda"]), expectedmax, rel_tol=1e-4)
        assert min(dfspectrumpkts["f_lambda"]) < 1e-9
        flambdamean = dfspectrumpkts["f_lambda"].mean()
        assert isinstance(flambdamean, float)
        assert math.isclose(flambdamean, expectedmean, rel_tol=1e-4)

    dfspectrum = at.spectra.scan_spectra(modelpath, 55, 65, fluxfilterfunc=None)[-1].collect()

    assert len(dfspectrum["lambda_angstroms"]) == 1000
    assert len(dfspectrum["f_lambda"]) == 1000
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[-1] - 29920.601421214415) < 1e-5
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[0] - 600.75759482509852) < 1e-5

    check_spectrum(dfspectrum, expectedmax=2.548532804918824e-13, expectedmean=1.0314682640070206e-14)

    timelowdays = at.get_timestep_times(modelpath)[55]
    timehighdays = at.get_timestep_times(modelpath)[65]

    dfspectrumpkts = at.spectra.scan_spectra_from_packets(
        modelpath, timelowdays=timelowdays, timehighdays=timehighdays
    )[-1].collect()

    # the packets file of the test model holds a part of the packets of the model, thus its flux is
    # below the flux of spec.out. test_spectra_get_flux_contributions compares spec.out with the sum
    # of the emission contributions
    check_spectrum(dfspectrumpkts, expectedmax=1.4601241778685615e-14, expectedmean=3.8221552332231758e-16)


@pytest.mark.benchmark
def test_spectra_get_spectrum_polar_angles() -> None:
    spectra = at.spectra.scan_spectra(
        modelpath=modelpath_classic_3d, average_over_phi=True, timestepmin=20, timestepmax=25
    )

    assert all(
        math.isclose(dirspec.select(pl.col("lambda_angstroms").mean()).collect().item(), 7510.074, rel_tol=1e-3)
        for dirspec in spectra.values()
    )
    assert all(
        math.isclose(dirspec.select(pl.col("lambda_angstroms").std()).collect().item(), 7647.317, rel_tol=1e-3)
        for dirspec in spectra.values()
    )

    results = {
        dirbin: (dfspecdir.select(mean=pl.col("f_lambda").mean(), std=pl.col("f_lambda").std()).collect().row(0))
        for dirbin, dfspecdir in spectra.items()
    }

    print(f"expected_results = {results!r}")

    expected_results = {
        0: (8.944885683622777e-12, 2.5390561316336613e-11),
        10: (7.192449910173842e-12, 2.0713405870496142e-11),
        20: (8.963182635824623e-12, 2.4720178744713477e-11),
        30: (8.06805028771611e-12, 2.2672897557383406e-11),
        40: (7.8306536944195e-12, 2.2812958326863807e-11),
        50: (8.259135507460651e-12, 2.2795973908331984e-11),
        60: (7.964029031817186e-12, 2.637892822134082e-11),
        70: (7.691392868658026e-12, 2.1262113332060223e-11),
        80: (8.450665096838155e-12, 2.352725654000879e-11),
        90: (8.828105146277665e-12, 2.534549767123003e-11),
    }

    for dirbin in expected_results:
        result_mean = results[dirbin][0]
        assert isinstance(result_mean, float)
        assert math.isclose(result_mean, expected_results[dirbin][0], rel_tol=1e-3)
        result_std = results[dirbin][1]
        assert isinstance(result_std, float)
        assert math.isclose(result_std, expected_results[dirbin][1], rel_tol=1e-3)


@pytest.mark.benchmark
def test_spectra_get_spectrum_polar_angles_frompackets(benchmark: BenchmarkFixture) -> None:
    timelowdays = at.get_timestep_times(modelpath_classic_3d, loc="start")[0]
    timehighdays = at.get_timestep_times(modelpath_classic_3d, loc="end")[25]

    lambda_bin_edges = np.arange(100.0, 50000.0 + 100.0, 100.0)

    spectrafrompkts = benchmark(
        lambda: at.spectra.scan_spectra_from_packets(
            modelpath=modelpath_classic_3d,
            average_over_phi=True,
            timelowdays=timelowdays,
            timehighdays=timehighdays,
            lambda_bin_edges=lambda_bin_edges,
        )
    )

    results_pkts = {
        dirbin: (
            dfspecdir.select(pl.col("f_lambda").mean()).collect().item(),
            dfspecdir.select(pl.col("f_lambda").std()).collect().item(),
        )
        for dirbin, dfspecdir in spectrafrompkts.items()
    }

    expected_results = {
        0: (1.797586165792736e-12, 5.590298324735384e-12),
        10: (1.5663080169205979e-12, 4.939024069693974e-12),
        20: (1.774102666296695e-12, 5.520228221791345e-12),
        30: (1.5995255247265518e-12, 4.9241711406614e-12),
        40: (1.639362478438235e-12, 4.9895610982735244e-12),
        50: (1.675756083487473e-12, 5.1182931882077005e-12),
        60: (1.5760353230761404e-12, 4.736330612172389e-12),
        70: (1.6263369485761792e-12, 4.913505318082798e-12),
        80: (1.7403350676338995e-12, 5.223319837133006e-12),
        90: (1.7311116087677568e-12, 5.333605673696894e-12),
    }
    actual_results = {
        dirbin: (float(results_pkts[dirbin][0]), float(results_pkts[dirbin][1])) for dirbin in expected_results
    }
    print(f"results: {actual_results}")

    for dirbin in expected_results:
        expected_mean = expected_results[dirbin][0]
        actual_mean = actual_results[dirbin][0]
        assert math.isclose(actual_mean, expected_mean, rel_tol=1e-3)
        expected_std = expected_results[dirbin][1]
        actual_std = actual_results[dirbin][1]
        assert math.isclose(actual_std, expected_std, rel_tol=1e-3)


def get_contributions_classic_3d(
    **kwargs: t.Any,
) -> tuple[list[atspectra.FluxContributionTuple], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Bin the packets of the classic 3d model. The time window and the wavelength window are constant.

    Thus the results of the different tests are comparable.
    """
    return atspectra.get_flux_contributions_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=3.0,
        timehighdays=8.0,
        lambda_bin_edges=np.linspace(3500.0, 8000.0, 101),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("groupby", "expected_contribs", "expected_top"),
    [
        (
            "ion",
            10,
            [
                ("Co II", 1.541452039348011e-08),
                ("free-free", 5.6865311711303595e-09),
                ("Co III", 6.196070067481124e-10),
            ],
        ),
        (
            "line",
            776,
            [
                ("free-free", 5.6865311711303595e-09),
                ("Co II λ3755 69-25", 4.2743896083241423e-10),
                ("Co II λ4161 69-31", 3.974052796767683e-10),
            ],
        ),
    ],
)
def test_spectra_flux_contribution_labels_from_packets(
    groupby: str, expected_contribs: int, expected_top: list[tuple[str, float]]
) -> None:
    """The emission group labels and the absorption group labels come from the linelist.

    This test makes sure that the labels and their fluxes do not change.
    """
    contributions, array_flambda_emission_total, array_lambda = get_contributions_classic_3d(groupby=groupby)

    assert len(contributions) == expected_contribs
    for contrib, (linelabel, fluxcontrib) in zip(contributions[: len(expected_top)], expected_top, strict=True):
        assert contrib.linelabel == linelabel
        assert np.isclose(contrib.fluxcontrib, fluxcontrib, rtol=1e-4, atol=0.0)

    assert np.isclose(
        np.trapezoid(array_flambda_emission_total, x=array_lambda), 2.019784984151385e-08, rtol=1e-4, atol=0.0
    )


# The "ion" group makes an ion label for the absorption. The "nucmass" group makes a line label.
# These two groups thus test the two conditions. The "line" group gives the same result as the "nucmass" group.
@pytest.mark.parametrize("groupby", ["ion", "nucmass"])
def test_spectra_absorption_contributions_from_packets(groupby: str) -> None:
    """The group of the contributions sets the shape of each absorption label.

    The ion group gives an ion label, e.g. "Co II". The nucmass group gives a line label, which also
    names the wavelength of the line, e.g. "Co II λ3755 69-25". No label names a free-free process or
    a bound-free process, because ARTIS records no absorption frequency for these processes.
    """
    contributions, _, _ = get_contributions_classic_3d(groupby=groupby, getemission=False)

    labels = [contrib.linelabel for contrib in contributions]
    assert labels
    assert all(labels)
    assert not [label for label in labels if "free" in label]
    if groupby == "nucmass":
        assert all("λ" in label for label in labels)
    else:
        assert not any("λ" in label for label in labels)


def test_spectra_element_contributions_sum_the_ions() -> None:
    """Each element series holds the emission and the absorption of all the ions of that element."""
    contributions, array_flambda_emission_total, _ = get_contributions_classic_3d(groupby="element")
    contributions_ion, array_flambda_emission_total_ion, _ = get_contributions_classic_3d(groupby="ion")

    assert np.allclose(array_flambda_emission_total, array_flambda_emission_total_ion, rtol=1e-6, atol=0.0)
    elements = {contribution.linelabel: contribution for contribution in contributions}
    assert set(elements) == {"Co", "Fe", "Ni", "free-free"}
    for label, contribution in elements.items():
        # "Fe II" belongs to "Fe", and "Fe II bound-free" belongs to "Fe bound-free"
        ions = [
            ion
            for ion in contributions_ion
            if ion.linelabel == label
            or (
                ion.linelabel.split(" ")[0] == label.split(" ")[0]
                and ("bound-free" in ion.linelabel) == ("bound-free" in label)
            )
        ]
        assert ions
        for arrayname in ("array_flambda_emission", "array_flambda_absorption"):
            iontotal = sum(getattr(ion, arrayname) for ion in ions)
            assert np.allclose(getattr(contribution, arrayname), iontotal, rtol=1e-6, atol=0.0)


def test_spectra_gamma_element_contributions(
    monkeypatch: pytest.MonkeyPatch, patch_scan_packets: Callable[[pl.DataFrame], None]
) -> None:
    """The element of a gamma packet is the element of its nuclide, thus Ni56 and Ni57 give one series."""
    dfpackets = pl.DataFrame({
        "e_rf": [1.0, 2.0, 4.0, 8.0],
        "t_arrive_d": [2.0] * 4,
        "pellet_nucindex": [0, 1, 2, -1],
        "nu_rf": [at.constants.c_ang_per_s / 5000.0] * 4,
    })

    def scan_nuclides(modelpath: Path | str) -> pl.LazyFrame:
        del modelpath
        return pl.LazyFrame({
            "pellet_nucindex": [-1, 0, 1, 2],
            "elsymbol": ["initial energy", "Ni", "Co", "Ni"],
            "nucname": ["initial energy", "Ni56", "Co56", "Ni57"],
        })

    patch_scan_packets(dfpackets)
    monkeypatch.setattr(atspectra, "scan_nuclides", scan_nuclides)
    contributions, array_flambda_emission_total, _ = atspectra.get_flux_contributions_from_packets(
        modelpath=Path(),
        timelowdays=1.5,
        timehighdays=2.5,
        lambda_bin_edges=np.array([4000.0, 5000.0, 6000.0]),
        getabsorption=False,
        groupby="element",
        gamma=True,
    )

    total = float(np.sum(array_flambda_emission_total))
    fractions = {
        contribution.linelabel: float(np.sum(contribution.array_flambda_emission)) / total
        for contribution in contributions
    }
    assert set(fractions) == {"Ni", "Co", "initial energy"}
    for label, energy in (("Ni", 5.0), ("Co", 2.0), ("initial energy", 8.0)):
        assert np.isclose(fractions[label], energy / 15.0, rtol=1e-12, atol=0.0)


def test_spectra_emission_text_files_fill_the_plot_range() -> None:
    """The contributions from emission.out cover the bins of the packets, which reach beyond each edge of the plot.

    The text files gave the bins with a centre inside the range only, thus each series stopped short of the edges,
    and the plot differed from the plot of the packets.
    """
    lambda_min, lambda_max = 4000.0, 6000.0
    _, _, arraylambda = atspectra.get_flux_contributions(
        modelpath_classic_3d, timestepmin=10, timestepmax=12, lambda_min=lambda_min, lambda_max=lambda_max
    )
    packetedges = atspectra.get_lambda_bin_edges(
        lambda_min, lambda_max, None, None, None, "angstrom", modelpath_classic_3d
    )
    assert arraylambda.min() < lambda_min
    assert arraylambda.max() > lambda_max
    # the text files give the centre of each bin, and the packets give the edges of the same bins
    centres = np.sort(arraylambda)
    assert len(centres) == len(packetedges) - 1
    assert np.all((packetedges[:-1] < centres) & (centres < packetedges[1:]))


def test_spectra_absorption_contributions_reject_nuclide_groupby() -> None:
    with pytest.raises(ValueError, match="cannot be grouped by nuclide"):
        get_contributions_classic_3d(groupby="nuc")


def test_spectra_velocity_shell_contributions() -> None:
    """A velocity shell holds the emission and the absorption of the packets whose last interaction lies inside it.

    The shells together hold every packet, thus their sums equal the totals of the ion groups.
    """
    # the corner of the 3D grid lies at sqrt(3) times vmax, which is 50 091 km/s
    shells = [0.0, 10000.0, 20000.0, 30000.0, 51000.0]
    contributions, array_flambda_emission_total, array_lambda = get_contributions_classic_3d(
        groupby="radialvelocity", shelledges=shells
    )
    contributions_ion, array_flambda_emission_total_ion, _ = get_contributions_classic_3d(groupby="ion")

    shelllabels = atspectra.get_shell_labels(shells)
    assert shelllabels == ["[0, 10000) km/s", "[10000, 20000) km/s", "[20000, 30000) km/s", "[30000, 51000) km/s"]
    assert [contrib.linelabel for contrib in contributions if contrib.linelabel in shelllabels] == [
        contrib.linelabel for contrib in contributions
    ]
    assert len(contributions) >= 2

    assert np.allclose(array_flambda_emission_total, array_flambda_emission_total_ion, rtol=1e-6, atol=0.0)

    absorption_shells = sum(contrib.array_flambda_absorption for contrib in contributions)
    absorption_ions = sum(contrib.array_flambda_absorption for contrib in contributions_ion)
    assert np.trapezoid(absorption_shells, x=array_lambda) > 0.0
    assert np.allclose(absorption_shells, absorption_ions, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize(
    ("rangegrouping", "rangeedges"),
    [("radialvelocity", (10000.0, 20000.0)), ("lineofsightvelocity", (-10000.0, 5000.0))],
)
def test_spectra_velocity_range_contributions(rangegrouping: str, rangeedges: tuple[float, float]) -> None:
    """The ion groups of a velocity range hold the emission and the absorption of the shell with the same edges."""
    shells = [-51000.0 if rangegrouping == "lineofsightvelocity" else 0.0, *rangeedges, 51000.0]
    contributions_shells, _, array_lambda = get_contributions_classic_3d(groupby=rangegrouping, shelledges=shells)
    (shelllabel,) = atspectra.get_shell_labels(rangeedges)
    (shell,) = (contrib for contrib in contributions_shells if contrib.linelabel == shelllabel)

    contributions, array_flambda_emission_total, _ = get_contributions_classic_3d(
        groupby="ion", velocityranges={rangegrouping: rangeedges}
    )

    assert len(contributions) >= 2
    assert np.trapezoid(shell.array_flambda_emission, x=array_lambda) > 0.0
    assert np.trapezoid(shell.array_flambda_absorption, x=array_lambda) > 0.0
    assert np.allclose(array_flambda_emission_total, shell.array_flambda_emission, rtol=1e-6, atol=0.0)
    absorption_ions = sum(contrib.array_flambda_absorption for contrib in contributions)
    assert np.allclose(absorption_ions, shell.array_flambda_absorption, rtol=1e-6, atol=0.0)


def test_spectra_two_velocity_ranges_keep_the_packets_inside_both() -> None:
    """A radial range and a line-of-sight range together hold the radial range of the line-of-sight shell."""
    losedges = (-10000.0, 5000.0)
    radialedges = (10000.0, 20000.0)
    contributions_shells, _, _ = get_contributions_classic_3d(
        groupby="lineofsightvelocity", shelledges=list(losedges), velocityranges={"radialvelocity": radialedges}
    )
    (shell,) = contributions_shells

    _, array_flambda_emission_total, _ = get_contributions_classic_3d(
        groupby="ion", velocityranges={"radialvelocity": radialedges, "lineofsightvelocity": losedges}
    )
    _, array_flambda_emission_total_radial, _ = get_contributions_classic_3d(
        groupby="ion", velocityranges={"radialvelocity": radialedges}
    )

    assert np.allclose(array_flambda_emission_total, shell.array_flambda_emission, rtol=1e-6, atol=0.0)
    assert 0.0 < array_flambda_emission_total.sum() < array_flambda_emission_total_radial.sum()


def test_spectra_no_thermal_emission_record_gives_no_thermal_velocity() -> None:
    """A packet with no thermal emission record is in the NOT SET series, and no velocity range from zero holds it.

    ARTIS gives such a packet a thermal emission velocity of zero. The thermal emission of this model
    lies above 11 000 km/s. Thus a packet below 10 000 km/s has no thermal emission record.
    """
    contributions, _, _ = get_contributions_classic_3d(
        groupby="radialvelocity", emissionevent="thermal", shelledges=[0.0, 10000.0, 51000.0], getabsorption=False
    )
    fluxes = {contrib.linelabel: contrib.fluxcontrib for contrib in contributions}
    assert "[0, 10000) km/s" not in fluxes
    assert fluxes["NOT SET"] > 0.0
    assert fluxes["[10000, 51000) km/s"] > 0.0

    contributions_range, _, _ = get_contributions_classic_3d(
        emissionevent="thermal", velocityranges={"radialvelocity": (0.0, 10000.0)}, getabsorption=False
    )
    assert not contributions_range


def test_spectra_sampled_emission_replaces_the_emission_type_and_the_absorption() -> None:
    """The sampled emission gives the emission type and the absorption, and the last interaction gives neither.

    The test model has no sampled emission. Thus the test copies the emission type of the last thermal emission and
    the last absorption to the sampled columns. It then sets the emission type of the last thermal emission and the
    columns of the last absorption to null. Thus only a plot that reads the sampled columns equals the plot of the last
    thermal emission.
    """
    scan_packets = atspectra.scan_packets

    def get_packets_with_sample(*args: t.Any, **kwargs: t.Any) -> tuple[int, pl.LazyFrame]:
        nprocs_read, lzdfpackets = scan_packets(*args, **kwargs)
        return nprocs_read, lzdfpackets.with_columns(
            sampled_emissiontype=pl.col("trueemissiontype"),
            sampled_absorption_type=pl.col("absorption_type"),
            sampled_absorption_freq=pl.col("absorption_freq"),
            trueemissiontype=pl.lit(None, dtype=pl.Int32),
            absorption_type=pl.lit(None, dtype=pl.Int32),
            absorption_freq=pl.lit(None, dtype=pl.Float32),
        )

    with mock.patch.object(atspectra, "scan_packets", side_effect=get_packets_with_sample):
        sampledcontributions, sampledtotal, _ = get_contributions_classic_3d(emissionevent="sampled")
    thermalcontributions, thermaltotal, _ = get_contributions_classic_3d(emissionevent="thermal")

    assert np.allclose(sampledtotal, thermaltotal, rtol=1e-10, atol=0.0)
    assert [contrib.linelabel for contrib in sampledcontributions] == [
        contrib.linelabel for contrib in thermalcontributions
    ]
    assert any(np.any(contrib.array_flambda_absorption > 0.0) for contrib in sampledcontributions)
    for sampledcontrib, thermalcontrib in zip(sampledcontributions, thermalcontributions, strict=True):
        for arrayname in ("array_flambda_emission", "array_flambda_absorption"):
            assert np.allclose(
                getattr(sampledcontrib, arrayname), getattr(thermalcontrib, arrayname), rtol=1e-10, atol=0.0
            )


def test_spectraemissionplot_of_a_model_with_no_sampled_emission_gives_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stop with a message before the command reads the packets of a model with no sampled emission.

    Each preset of ARTIS sets SAMPLE_RPKT_EMISSION to false.
    """
    with (
        mock.patch.object(atspectra, "get_flux_contributions_from_packets") as mockcontributions,
        pytest.raises(SystemExit),
    ):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath_classic_3d,
            timemin=4,
            timemax=6.5,
            emissionabsorption=True,
            use_sampledemissiontype=True,
            outputfile=tmp_path / "nosample.pdf",
        )

    mockcontributions.assert_not_called()
    errortext = capsys.readouterr().err
    assert "hold no sampled emission" in errortext
    assert "SAMPLE_RPKT_EMISSION" in errortext


def test_spectra_takes_the_earlier_names_of_the_velocity_groupings() -> None:
    """-groupby velocity and losvelocity, and the earlier names of the velocity ranges, stay valid in a script."""
    from artistools.spectra import plotspectra

    args = at.misc.parse_cli_args(
        plotspectra.addargs,
        None,
        None,
        [
            "-groupby",
            "losvelocity",
            "-emissionvelocityrange",
            "1000",
            "2000",
            "-emissionlosvelocityrange",
            "0c",
            "0.1c",
        ],
    )
    assert args.groupby == "lineofsightvelocity"
    assert args.emissionradialvelocityrange == [(1000.0, "kmps"), (2000.0, "kmps")]
    assert args.emissionlineofsightvelocityrange[1][1] == "c"
    assert at.misc.parse_cli_args(plotspectra.addargs, None, None, ["-groupby", "velocity"]).groupby == "radialvelocity"


def test_spectra_velocity_argument_takes_kmps_or_c() -> None:
    """A shell edge is a number in km/s, or a fraction of c with a c suffix, and the labels keep that unit."""
    assert at.misc.parse_velocity_argument("5000") == (5000.0, "kmps")
    assert at.misc.parse_velocity_argument("5000 km/s") == (5000.0, "kmps")
    assert at.misc.parse_velocity_argument("5000km/s", requireunit=True) == (5000.0, "kmps")
    with pytest.raises(argparse.ArgumentTypeError, match="has no unit"):
        at.misc.parse_velocity_argument("5000", requireunit=True)
    velocity_kmps, unit = at.misc.parse_velocity_argument("0.1C")
    assert unit == "c"
    assert np.isclose(velocity_kmps, 29979.2458)

    for badvalue in ("fast", "inf", "infc", "nan"):
        with pytest.raises(argparse.ArgumentTypeError, match="not a finite velocity"):
            at.misc.parse_velocity_argument(badvalue)

    shells = [0.0, 29979.2458, 59958.4916]
    assert atspectra.get_shell_labels(shells, "c") == ["[0, 0.1) c", "[0.1, 0.2) c"]
    # a default edge in units of c takes three significant digits
    assert atspectra.get_shell_labels([0.0, 14315.06, 28630.12], "c") == ["[0, 0.0477) c", "[0.0477, 0.0955) c"]
    assert atspectra.get_shell_labels(shells) == ["[0, 29979.2) km/s", "[29979.2, 59958.5) km/s"]

    # an edge with a fraction keeps its digits, because the label is the key of the group and names the bound
    assert atspectra.get_shell_labels([0.1, 0.2, 0.3]) == ["[0.1, 0.2) km/s", "[0.2, 0.3) km/s"]
    assert atspectra.get_shell_labels([0.4, 1.4, 2.0]) == ["[0.4, 1.4) km/s", "[1.4, 2) km/s"]
    assert atspectra.get_shell_labels([0.0, 0.15000000000000002, 0.5], "ye") == ["Ye [0, 0.15)", "Ye [0.15, 0.5)"]
    with pytest.raises(ValueError, match="same label"):
        atspectra.get_shell_labels([0.1, 0.1, 0.3])

    # two edges that six significant digits cannot separate keep every digit
    c_kmps = 2.99792458e5
    labels = atspectra.get_shell_labels([0.1234561 * c_kmps, 0.1234564 * c_kmps, 0.2 * c_kmps], "c")
    assert len(set(labels)) == 2
    assert labels[0].startswith("[0.1234561")
    labels = atspectra.get_shell_labels([0.1, 0.1 + 1e-12, 0.3])
    assert len(set(labels)) == 2


def test_spectra_velocity_shell_index_of_a_packet_with_no_thermal_emission() -> None:
    """A packet with a NaN velocity takes the index of NOT SET, and a packet outside every shell takes null."""
    shells = [0.0, 10000.0, 20000.0]
    dfpackets = pl.DataFrame({"v": [5.0e8, float("nan"), 5.0e10, 1.5e9, None, 0.0, 2.0e9]})
    indices = dfpackets.select(atspectra.get_shell_index_expr("v", shells)).to_series().to_list()
    labelofindex = atspectra.get_shell_label_of_index(shells, "kmps")
    labels = [None if index is None else labelofindex[index] for index in indices]
    # a shell holds its lower edge, and the last edge is outside every shell
    assert labels == ["[0, 10000) km/s", "NOT SET", None, "[10000, 20000) km/s", "NOT SET", "[0, 10000) km/s", None]


def test_spectra_velocity_shell_order_counts_not_set_against_the_limit() -> None:
    """The NOT SET series takes one place of -maxseriescount, thus the plot never keeps one series too many."""
    lambdas = np.array([4000.0, 5000.0])
    shells = [0.0, 10000.0, 20000.0, 30000.0]
    labels = [*atspectra.get_shell_labels(shells), "NOT SET"]
    contributions = [
        atspectra.FluxContributionTuple(flux, label, np.full(2, flux), np.zeros(2))
        for label, flux in zip(labels, [3.0, 1.0, 2.0, 4.0], strict=True)
    ]
    args = argparse.Namespace(fixedionlist=None, maxseriescount=2, shelledges=shells, shellunit="kmps", hideother=False)
    ordered = at.spectra.plotspectra.order_and_color_shells(contributions, lambdas, args)

    assert [contribution.linelabel for contribution in ordered] == ["[0, 10000) km/s", "NOT SET", "Other"]


def test_spectra_default_velocity_shells_take_units_of_c_for_a_fast_model() -> None:
    """The default shells take km/s below a vmax of 0.2 c, and units of c from 0.2 c."""
    edges, unit = atspectra.get_default_radialvelocity_shells(modelpath_classic_3d)
    assert unit == "kmps"
    assert len(edges) == 12
    assert np.isclose(edges[10], 28920.2, rtol=1e-4)

    fastmeta = {"vmax_cmps": 0.3 * 2.99792458e10, "dimensions": 1}
    with mock.patch("artistools.inputmodel.scan_modeldata", return_value=(pl.LazyFrame(), fastmeta)):
        edges, unit = atspectra.get_default_radialvelocity_shells("fastmodel", nshells=3)
    assert unit == "c"
    assert atspectra.get_shell_labels(edges, unit) == ["[0, 0.1) c", "[0.1, 0.2) c", "[0.2, 0.3) c"]


def test_spectra_losvelocity_shell_contributions() -> None:
    """A line-of-sight shell holds the packets by the signed velocity along the packet direction."""
    shells = [-51000.0, -20000.0, 0.0, 20000.0, 51000.0]
    contributions, array_flambda_emission_total, array_lambda = get_contributions_classic_3d(
        groupby="lineofsightvelocity", shelledges=shells
    )
    _, array_flambda_emission_total_ion, _ = get_contributions_classic_3d(groupby="ion")

    labels = [contrib.linelabel for contrib in contributions]
    assert set(labels) <= set(atspectra.get_shell_labels(shells))
    assert any(label.startswith("[-") for label in labels)
    assert np.allclose(array_flambda_emission_total, array_flambda_emission_total_ion, rtol=1e-6, atol=0.0)
    assert np.trapezoid(sum(contrib.array_flambda_absorption for contrib in contributions), x=array_lambda) > 0.0


def test_spectra_ye_shell_contributions() -> None:
    """A Ye shell holds the packets by the initial electron fraction of the cell of the last interaction."""
    import artistools.inputmodel

    realgetmodeldata = artistools.inputmodel.scan_modeldata

    def get_modeldata_with_ye(*args: t.Any, **kwargs: t.Any) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
        # the test model has no Ye column, thus the odd cells get 0.2 and the even cells get 0.4
        dfmodel, modelmeta = realgetmodeldata(*args, **kwargs)
        return dfmodel.with_columns(Ye=0.2 + 0.2 * (pl.col("inputcellid") % 2 == 0).cast(pl.Float32)), modelmeta

    with mock.patch("artistools.inputmodel.scan_modeldata", side_effect=get_modeldata_with_ye):
        contributions, array_flambda_emission_total, array_lambda = get_contributions_classic_3d(
            groupby="ye", shelledges=[0.0, 0.3, 0.6]
        )
    _, array_flambda_emission_total_ion, _ = get_contributions_classic_3d(groupby="ion")

    assert sorted(contrib.linelabel for contrib in contributions) == ["Ye [0, 0.3)", "Ye [0.3, 0.6)"]
    assert np.allclose(array_flambda_emission_total, array_flambda_emission_total_ion, rtol=1e-6, atol=0.0)
    assert np.trapezoid(sum(contrib.array_flambda_absorption for contrib in contributions), x=array_lambda) > 0.0

    with pytest.raises(ValueError, match="no Ye column"):
        get_contributions_classic_3d(groupby="ye", shelledges=[0.0, 0.3, 0.6])


def test_spectra_ye_of_a_model_whose_cell_ids_start_at_zero(tmp_path: Path) -> None:
    """A packet takes the Ye of its own cell when the ids of model.txt start at 0. The old code took the next cell."""
    (tmp_path / "model.txt").write_text(
        "3\n1.0\n#inputcellid vel_r_max_kmps logrho X_Fegroup X_Ni56 X_Co56 X_Fe52 X_Cr48 X_Ni57 X_Co57 Ye\n"
        + "".join(f"{cellid} {1000 * (cellid + 1)} -10 1 0 0 0 0 0 0 {0.1 * (cellid + 1)}\n" for cellid in range(3)),
        encoding="utf-8",
    )
    em_time = 86400.0
    dfpackets = pl.LazyFrame({
        "em_posx": [vel_kmps * 1e5 * em_time for vel_kmps in (500.0, 1500.0, 2500.0)],
        "em_posy": [0.0] * 3,
        "em_posz": [0.0] * 3,
        "em_time": [em_time] * 3,
    })

    yevalues = atspectra.add_ye_columns(dfpackets, tmp_path, [("ye", "em")], thermalfromvelocity=False).collect()["ye"]

    assert np.allclose(yevalues.to_numpy(), [0.1, 0.2, 0.3], rtol=1e-6)


def test_spectra_velocity_shell_contributions_need_shell_edges() -> None:
    with pytest.raises(ValueError, match="needs the shell edges"):
        get_contributions_classic_3d(groupby="radialvelocity")

    for badshells in ([0.0, 20000.0, 10000.0], [0.0, math.inf], [0.0, math.nan, 20000.0]):
        with pytest.raises(ValueError, match="must be finite, increase"):
            get_contributions_classic_3d(groupby="radialvelocity", shelledges=badshells)

    with pytest.raises(ValueError, match="edges of the lineofsightvelocity range must be finite, increase"):
        get_contributions_classic_3d(velocityranges={"lineofsightvelocity": (5000.0, -5000.0)})


@mock.patch.object(mplax.Axes, "stackplot", side_effect=mplax.Axes.stackplot, autospec=True)
def test_spectraemissionplot_velocity_shells(mockstackplot: mock.MagicMock, tmp_path: Path) -> None:
    """The emission plot stacks the shells from the inner one to the outer one, with the default shell edges."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath_classic_3d,
        outputfile=tmp_path / "velocityshells.pdf",
        timemin=4,
        timemax=6.5,
        emissionabsorption=True,
        groupby="radialvelocity",
    )

    assert mockstackplot.call_count == 2
    nseries = len(mockstackplot.call_args_list[0].args[2])
    assert 2 <= nseries <= 12


@mock.patch.object(mplax.Axes, "stackplot", side_effect=mplax.Axes.stackplot, autospec=True)
def test_spectraemissionplot_velocity_shells_in_units_of_c(mockstackplot: mock.MagicMock, tmp_path: Path) -> None:
    """A shell edge with a c suffix sets the edges and the labels in units of c."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath_classic_3d,
        outputfile=tmp_path / "velocityshells_c.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        groupby="radialvelocity",
        velocityshells=["0c", "0.04c", "0.06c", "0.1c"],
    )

    # the edges lie inside vmax of the model, thus each of the three shells holds packets
    assert mockstackplot.call_count == 1
    assert len(mockstackplot.call_args_list[0].args[2]) == 3


@mock.patch.object(mplax.Axes, "set_title", side_effect=mplax.Axes.set_title, autospec=True)
@mock.patch.object(mplax.Axes, "stackplot", side_effect=mplax.Axes.stackplot, autospec=True)
def test_spectraemissionplot_velocity_ranges(
    mockstackplot: mock.MagicMock, mocksettitle: mock.MagicMock, tmp_path: Path
) -> None:
    """A velocity range keeps the ion series, and the title gives each range in the unit of its values."""
    # argparse must read a negative value with a c suffix as a value and not as a flag
    at.spectra.plot(
        argsraw=[
            str(modelpath_classic_3d),
            "--showemission",
            "-emissionlineofsightvelocityrange",
            "-0.05c",
            "0.05c",
            "-timemin",
            "4",
            "-timemax",
            "6.5",
            "-outputfile",
            str(tmp_path / "losvelocityrange.pdf"),
        ]
    )

    assert mockstackplot.call_count == 1
    assert len(mockstackplot.call_args_list[0].args[2]) >= 2
    title = mocksettitle.call_args_list[-1].args[1]
    assert title.endswith(", packets at line-of-sight velocity [-0.05, 0.05) c")

    at.spectra.plot(
        argsraw=[],
        specpath=modelpath_classic_3d,
        outputfile=tmp_path / "velocityranges.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        emissionradialvelocityrange=["0.04c", "0.06c"],
        emissionlineofsightvelocityrange=[-15000, 15000],
    )

    assert mockstackplot.call_count == 2
    assert len(mockstackplot.call_args_list[1].args[2]) >= 2
    title = mocksettitle.call_args_list[-1].args[1]
    assert title.endswith(", packets at radial velocity [0.04, 0.06) c and line-of-sight velocity [-15000, 15000) km/s")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraemissionplot_linewidth_arg(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The -linewidth list sets the width of the net spectrum and of a reference spectrum on the emission plot."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d, "2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "emission_linewidth.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        groupby="radialvelocity",
        velocityshells=["0c", "0.04c", "0.06c", "0.1c"],
        linewidth=[0.5, 7.0],
        write_data=True,
    )

    # the net spectrum of the model is the first line, and the reference spectrum is the second line
    assert [callargs.kwargs.get("linewidth") for callargs in mockplot.call_args_list] == [0.5, 7.0]
    # the emission plot gave no table of the reference spectrum, because only make_spectrum_plot made one
    dfreferences = read_data_file(tmp_path / "emission_linewidth_references.txt")
    # the table holds the points that the axes show, and the reference has negative fluxes below the bottom of 0
    axis, drawnx, drawny = mockplot.call_args_list[1].args
    drawnx, drawny = np.asarray(drawnx, dtype=float), np.asarray(drawny, dtype=float)
    (xlow, xhigh), (ylow, yhigh) = sorted(axis.get_xlim()), sorted(axis.get_ylim())
    isinside = (drawnx >= xlow) & (drawnx <= xhigh) & (drawny >= ylow) & (drawny <= yhigh)
    assert 0 < dfreferences.height <= len(drawny)
    assert set(drawnx[isinside].round(4)) <= set(dfreferences["x_plotted_angstrom"].round(4).to_list())


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_emission_plot_data_gives_the_drawn_values(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The table gave f_lambda alone, and the figure scales it for --normalised and draws the absorption below zero."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d],
        outputfile=tmp_path / "emission.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        showabsorption=True,
        nostack=True,
        normalised=True,
        write_data=True,
    )
    dftable = read_data_file(tmp_path / "emission.txt")
    # the line can hold a point outside -xmin and -xmax, and the table takes the points inside them
    xlow, xhigh = dftable["x_plotted_angstrom"].min(), dftable["x_plotted_angstrom"].max()
    drawnlines = [
        np.asarray(call.args[2], dtype=np.float64)[
            (np.asarray(call.args[1], dtype=np.float64) >= xlow) & (np.asarray(call.args[1], dtype=np.float64) <= xhigh)
        ]
        for call in mockplot.call_args_list
    ]
    emissioncolumns = [col for col in dftable.columns if col.startswith("emission_flux_plotted.")]
    absorptioncolumns = [col for col in dftable.columns if col.startswith("absorption_flux_plotted.")]
    assert emissioncolumns
    assert len(absorptioncolumns) == len(emissioncolumns)
    for column in [*emissioncolumns, *absorptioncolumns, "flux_plotted.Spectrum"]:
        assert any(np.allclose(drawn, dftable[column].to_numpy(), rtol=1e-6, atol=0) for drawn in drawnlines), column
    assert (dftable[absorptioncolumns[0]] <= 0).all()


@mock.patch.object(mplax.Axes, "stackplot", side_effect=mplax.Axes.stackplot, autospec=True)
def test_spectraemissionplot_losvelocity_shells(mockstackplot: mock.MagicMock, tmp_path: Path) -> None:
    """The default line-of-sight shells run from -vmax to vmax, plus one shell to each corner."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath_classic_3d,
        outputfile=tmp_path / "losvelocity.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        groupby="lineofsightvelocity",
    )

    edges, _ = atspectra.get_default_lineofsightvelocity_shells(modelpath_classic_3d)
    assert len(edges) == 13
    assert edges[0] == -edges[-1]
    assert 2 <= len(mockstackplot.call_args_list[0].args[2]) <= 12


@mock.patch.object(mplax.Axes, "stackplot", side_effect=mplax.Axes.stackplot, autospec=True)
def test_spectraemissionplot_velocity_shells_keep_the_series_limit(
    mockstackplot: mock.MagicMock, tmp_path: Path
) -> None:
    """More shells than -maxseriescount give that many series plus Other, as the ions do."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath_classic_3d,
        outputfile=tmp_path / "velocityshells_limit.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        groupby="radialvelocity",
        velocityshells=list(np.linspace(8000.0, 28000.0, 21)),
        maxseriescount=3,
    )

    assert len(mockstackplot.call_args_list[0].args[2]) == 4


# the outputfile of each case is the same, because the command stops before it writes a file
@pytest.mark.parametrize(
    ("plotargs", "messages"),
    [
        pytest.param(
            {"specpath": [modelpath, modelpath_classic_3d], "timemin": 290, "timemax": 320, "emissionabsorption": True},
            ["one ARTIS model"],
            id="two_models",
        ),
        pytest.param(
            {
                "specpath": modelpath_classic_3d,
                "timemin": 4,
                "timemax": 6.5,
                "showemission": True,
                "plotviewingangle": [0, 5],
            },
            ["one direction bin"],
            id="two_direction_bins",
        ),
        # --vpkt_match_emission_exclusion_to_opac names the options that it needs, and does not fail an assert
        pytest.param(
            {"specpath": modelpath, "timemin": 290, "timemax": 320, "vpkt_match_emission_exclusion_to_opac": True},
            ["--showemission", "-plotvspecpol"],
            id="vpkt_exclusion_without_its_options",
        ),
        # a gamma packet has no ion. The spectrum viewer offered these choices in the mode of the gamma packets, and
        # the plot then failed
        *(
            pytest.param(
                {
                    "specpath": modelpath_classic_3d,
                    "timemin": 4,
                    "timemax": 6.5,
                    "showemission": True,
                    "gamma": True,
                    "groupby": groupby,
                },
                [f"does not accept --groupby {groupby}"],
                id=f"gamma_groupby_{groupby}",
            )
            for groupby in ("ion", "line")
        ),
        # a virtual packet holds no emission position, and a gamma packet holds no thermal emission position
        *(
            pytest.param(
                {
                    "specpath": modelpath_classic_3d,
                    "timemin": 4,
                    "timemax": 6.5,
                    "showemission": True,
                    **optionargs,
                    **packetargs,
                },
                [message],
                id=f"{packetid}_{optionid}",
            )
            for packetid, packetargs, message in (
                ("gamma_thermal", {"gamma": True, "use_thermalemissiontype": True}, "has no thermal emission"),
                ("vpkt", {"plotvspecpol": [0]}, "does not accept"),
            )
            for optionid, optionargs in (
                ("groupby_radialvelocity", {"groupby": "radialvelocity"}),
                ("radialvelocityrange", {"emissionradialvelocityrange": [5000, 10000]}),
                ("lineofsightvelocityrange", {"emissionlineofsightvelocityrange": [-5000, 5000]}),
            )
        ),
    ],
)
def test_emission_plot_refuses_a_request_that_it_cannot_draw(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], plotargs: dict[str, t.Any], messages: list[str]
) -> None:
    """An emission plot stops with a message for a request that it cannot draw, and not with a silent choice.

    An emission plot draws one ARTIS model and one direction bin. A shell grouping and a velocity range need the
    emission position of each packet.
    """
    with pytest.raises(SystemExit):
        at.spectra.plot(argsraw=[], outputfile=tmp_path / "refused.pdf", **plotargs)

    captured = capsys.readouterr().err
    for message in messages:
        assert message in captured


def test_spectra_gamma_packets_group_by_emission_velocity(patch_scan_packets: Callable[[pl.DataFrame], None]) -> None:
    """Each gamma packet goes to the shell of the velocity of its decay, and the shells sum to the whole spectrum.

    ARTIS records the decay position of a gamma packet since artis-mcrt/artis#637. Before that change, plotspectra
    refused a shell grouping for gamma packets.
    """
    emtime = 2.0 * at.constants.day_to_s
    speeds_kmps = [5000.0, 15000.0, 25000.0]
    dfpackets = pl.DataFrame({
        "e_rf": [1.0, 2.0, 4.0],
        "t_arrive_d": [2.0, 2.0, 2.0],
        "nu_rf": [at.constants.c_ang_per_s / 5000.0] * 3,
        "em_posx": [speed * at.constants.km_to_cm * emtime for speed in speeds_kmps],
        "em_posy": [0.0] * 3,
        "em_posz": [0.0] * 3,
        "em_time": [emtime] * 3,
        # the first packet moves away from the observer, thus its line-of-sight velocity is negative
        "dirx": [-1.0, 1.0, 1.0],
        "diry": [0.0] * 3,
        "dirz": [0.0] * 3,
    })

    patch_scan_packets(dfpackets)
    lambda_bin_edges = np.array([4000.0, 5000.0, 6000.0])
    expected = {
        "radialvelocity": {"[0, 10000) km/s": 1.0, "[10000, 20000) km/s": 2.0, "[20000, 30000) km/s": 4.0},
        "lineofsightvelocity": {"[-10000, 0) km/s": 1.0, "[10000, 20000) km/s": 2.0, "[20000, 30000) km/s": 4.0},
    }
    for groupby, expectedenergies in expected.items():
        contributions, array_flambda_emission_total, _ = atspectra.get_flux_contributions_from_packets(
            modelpath=Path(),
            timelowdays=1.5,
            timehighdays=2.5,
            lambda_bin_edges=lambda_bin_edges,
            getabsorption=False,
            groupby=groupby,
            gamma=True,
            shelledges=[-10000.0, 0.0, 10000.0, 20000.0, 30000.0],
        )
        energies = {
            contribution.linelabel: float(np.sum(contribution.array_flambda_emission)) for contribution in contributions
        }
        total = float(np.sum(array_flambda_emission_total))
        assert set(energies) == set(expectedenergies)
        for label, energy in expectedenergies.items():
            assert np.isclose(energies[label] / total, energy / 7.0, rtol=1e-12, atol=0.0)


def test_spectraemissionplot_gamma_shells_need_an_emission_position(tmp_path: Path) -> None:
    """A model from ARTIS before artis-mcrt/artis#637 stops with a message, because its gamma packets have no position.

    Without the message, each packet lay outside every shell and the plot had no series.
    """
    with pytest.raises(ValueError, match="hold no emission position"):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath,
            timedays=300,
            showemission=True,
            gamma=True,
            groupby="radialvelocity",
            outputfile=tmp_path / "gammashells.pdf",
        )


def test_spectraemissionplot_velocity_shells_reject_an_empty_selection(tmp_path: Path) -> None:
    """A shell selection that holds no packet stops with a message when the plot is normalised."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath_classic_3d,
            timemin=4,
            timemax=6.5,
            showemission=True,
            groupby="radialvelocity",
            velocityshells=["0.5c", "0.6c"],
            normalised=True,
            outputfile=tmp_path / "empty.pdf",
        )


def test_spectra_get_flux_contributions(benchmark: BenchmarkFixture) -> None:
    timestepmin = 40
    timestepmax = 80
    dfspectrum = at.spectra.scan_spectra(
        modelpath=modelpath, timestepmin=timestepmin, timestepmax=timestepmax, fluxfilterfunc=None
    )[-1].collect()

    integrated_flux_specout = np.trapezoid(dfspectrum["f_lambda"], x=dfspectrum["lambda_angstroms"])

    _contribution_list, array_flambda_emission_total, arraylambda_angstroms = benchmark(
        lambda: at.spectra.get_flux_contributions(
            modelpath, timestepmin=timestepmin, timestepmax=timestepmax, use_lastemissiontype=False
        )
    )

    integrated_flux_emission = -np.trapezoid(array_flambda_emission_total, x=arraylambda_angstroms)

    # total spectrum should be equal to the sum of all emission processes
    print(f"Integrated flux from spec.out:     {integrated_flux_specout}")
    print(f"Integrated flux from emission sum: {integrated_flux_emission}")
    assert math.isclose(integrated_flux_specout, integrated_flux_emission, rel_tol=4e-3)

    # check each bin is not out by a large fraction
    diff = [
        abs(x - y)
        for x, y in zip(reversed(array_flambda_emission_total), dfspectrum["f_lambda"].to_numpy(), strict=True)
    ]
    print(f"Max f_lambda difference {max(diff) / integrated_flux_specout}")
    assert max(diff) / integrated_flux_specout < 1e-9


def test_spectra_get_flux_contributions_reads_a_compressed_compositiondata(tmp_path: Path) -> None:
    """A model with only compositiondata.txt.xz gives the elements from that file and not from the log."""
    for filename in ("input.txt", "spec.out", "emissiontrue.out.xz", "absorption.out.xz"):
        (tmp_path / filename).symlink_to(modelpath / filename)
    (tmp_path / "compositiondata.txt.xz").write_bytes(lzma.compress((modelpath / "compositiondata.txt").read_bytes()))

    contributions, flambda_total, _ = at.spectra.get_flux_contributions(
        tmp_path, timestepmin=40, timestepmax=80, use_lastemissiontype=False
    )
    contributions_plain, flambda_total_plain, _ = at.spectra.get_flux_contributions(
        modelpath, timestepmin=40, timestepmax=80, use_lastemissiontype=False
    )

    assert [c.linelabel for c in contributions] == [c.linelabel for c in contributions_plain]
    assert np.allclose(flambda_total, flambda_total_plain, rtol=1e-12, atol=0.0)


def test_spectra_get_flux_contributions_wavelength_window() -> None:
    """A wavelength window restricts the spectra and the flux contributions used for ranking.

    The window holds the bins with a centre inside it and the nearest bin beyond each bound, as the packets do.
    """
    timestepmin = 40
    timestepmax = 80
    lambda_min = 3500.0
    lambda_max = 7000.0

    contributions_full, flambda_total_full, arraylambda_full = at.spectra.get_flux_contributions(
        modelpath, timestepmin=timestepmin, timestepmax=timestepmax, use_lastemissiontype=False
    )

    contributions_window, flambda_total_window, arraylambda_window = at.spectra.get_flux_contributions(
        modelpath,
        timestepmin=timestepmin,
        timestepmax=timestepmax,
        use_lastemissiontype=False,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )

    below = arraylambda_full[arraylambda_full <= lambda_min].max()
    above = arraylambda_full[arraylambda_full >= lambda_max].min()
    nu_select = (arraylambda_full >= below) & (arraylambda_full <= above)
    assert np.array_equal(arraylambda_window, arraylambda_full[nu_select])
    assert np.allclose(flambda_total_window, flambda_total_full[nu_select], rtol=1e-12, atol=0.0)

    contrib_full_bylabel = {c.linelabel: c for c in contributions_full}
    assert any(c.fluxcontrib < 0.999 * contrib_full_bylabel[c.linelabel].fluxcontrib for c in contributions_window), (
        "expected some flux contribution outside the window to be excluded from the ranking integral"
    )
    for contrib in contributions_window:
        full = contrib_full_bylabel[contrib.linelabel]
        assert contrib.fluxcontrib <= full.fluxcontrib * (1 + 1e-12)
        assert np.allclose(contrib.array_flambda_emission, full.array_flambda_emission[nu_select], rtol=1e-12, atol=0.0)
        assert np.allclose(
            contrib.array_flambda_absorption, full.array_flambda_absorption[nu_select], rtol=1e-12, atol=0.0
        )


def test_spectra_get_flux_contributions_from_packets(benchmark: BenchmarkFixture) -> None:
    lambdamin = 200.0
    lambdamax = 20000.0
    deltalambda = 100.0
    timelowdays = 4.0
    timehighdays = 7.0
    lambda_bin_edges = np.arange(lambdamin, lambdamax + deltalambda, deltalambda)
    dfspectrum = at.spectra.scan_spectra_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=timelowdays,
        timehighdays=timehighdays,
        lambda_bin_edges=lambda_bin_edges,
    )[-1].collect()

    integrated_flux_frompackets = np.trapezoid(dfspectrum["f_lambda"], x=dfspectrum["lambda_angstroms"])
    _contribution_list, array_flambda_emission_total, arraylambda_angstroms = benchmark(
        lambda: at.spectra.get_flux_contributions_from_packets(
            modelpath_classic_3d, timelowdays=timelowdays, timehighdays=timehighdays, lambda_bin_edges=lambda_bin_edges
        )
    )

    integrated_flux_emission = np.trapezoid(array_flambda_emission_total, x=arraylambda_angstroms)

    # total spectrum should be equal to the sum of all emission processes
    print(f"Integrated flux from the packets:  {integrated_flux_frompackets}")
    print(f"Integrated flux from emission sum: {integrated_flux_emission}")
    assert math.isclose(integrated_flux_frompackets, integrated_flux_emission, rel_tol=4e-3)

    # check each bin is not out by a large fraction
    diff = [abs(x - y) for x, y in zip(array_flambda_emission_total, dfspectrum["f_lambda"].to_numpy(), strict=True)]
    print(f"Max f_lambda difference {max(diff) / integrated_flux_frompackets}")
    assert max(diff) / integrated_flux_frompackets < 1e-10


@pytest.mark.parametrize(("directionbin", "average_over_phi"), [(12, False), (10, True)])
def test_spectra_packet_contributions_of_a_direction_bin(directionbin: int, average_over_phi: bool) -> None:
    """Check that the emission contributions of a direction bin have the same sum as the packet spectrum of that bin.

    get_flux_contributions_from_packets does not call scan_spectra_from_packets, thus it must apply the solid angle of
    the bin.
    """
    lambda_bin_edges = np.arange(3000.0, 9020.0, 20.0)
    dfspectrum = at.spectra.scan_spectra_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=4.0,
        timehighdays=7.0,
        lambda_bin_edges=lambda_bin_edges,
        directionbins=[directionbin],
        average_over_phi=average_over_phi,
    )[directionbin].collect()
    _, array_flambda_emission_total, _ = at.spectra.get_flux_contributions_from_packets(
        modelpath_classic_3d,
        timelowdays=4.0,
        timehighdays=7.0,
        lambda_bin_edges=lambda_bin_edges,
        directionbin=directionbin,
        average_over_phi=average_over_phi,
        getabsorption=False,
    )
    assert np.max(dfspectrum["f_lambda"].to_numpy()) > 0.0
    assert np.allclose(array_flambda_emission_total, dfspectrum["f_lambda"].to_numpy(), rtol=1e-9, atol=0.0)


@pytest.mark.parametrize(
    ("use_emissiontime", "use_escapetime", "expected_use_time"), [(True, False, "emission"), (False, True, "escape")]
)
@mock.patch("artistools.spectra.core.get_flux_contributions_from_packets")
def test_spectra_contribution_plot_forwards_packet_time(
    mockgetcontributions: mock.MagicMock,
    tmp_path: Path,
    use_emissiontime: bool,
    use_escapetime: bool,
    expected_use_time: str,
) -> None:
    """The contribution plot must use the packet time that the command selects."""
    contribution_list: list[atspectra.FluxContributionTuple] = []
    mockgetcontributions.return_value = (contribution_list, np.zeros(2), np.array([4000.0, 5000.0]))

    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / f"contributions_{expected_use_time}.pdf",
        timemin=290.0,
        timemax=320.0,
        emissionabsorption=True,
        use_emissiontime=use_emissiontime,
        use_escapetime=use_escapetime,
    )

    assert mockgetcontributions.call_args.kwargs["use_time"] == expected_use_time


def test_spectra_gamma_emission_time_uses_decay(patch_scan_packets: Callable[[pl.DataFrame], None]) -> None:
    """Gamma packet selection must use the decay time and not the radiation packet emission time."""
    dfpackets = pl.DataFrame({
        "e_rf": [1.0, 2.0],
        "t_arrive_d": [1.0, 3.0],
        "tdecay": [1.0 * at.constants.day_to_s, 3.0 * at.constants.day_to_s],
        "em_time": [3.0 * at.constants.day_to_s, 1.0 * at.constants.day_to_s],
        "nu_rf": [at.constants.c_ang_per_s / 5000.0] * 2,
    })

    patch_scan_packets(dfpackets)
    dfspectrum = atspectra.scan_spectra_from_packets(
        modelpath=Path(),
        timelowdays=0.5,
        timehighdays=1.5,
        lambda_bin_edges=np.array([4000.0, 5000.0, 6000.0]),
        use_time="emission",
        gamma=True,
        directionbins=[-1],
    )[-1].collect()

    integrated_flux = dfspectrum.select((pl.col("f_lambda") * pl.col("delta_lambda")).sum()).item()
    expected_flux = 1.0 / at.constants.day_to_s / (4 * math.pi * at.constants.megaparsec_to_cm**2)
    assert np.isclose(integrated_flux, expected_flux, rtol=1e-12, atol=0.0)


# The escape time of a packet is the value in the file times the Lorentz factor of the escape surface.
# The second packet enters the window of 1 to 2 days at beta 0.6, where 2.2 d * sqrt(1 - 0.6^2) = 1.76 d.
# The first packet leaves the window at beta 0.8, where 1.5 d * 0.6 = 0.9 d.
@pytest.mark.parametrize(("beta", "expected_e_cmf_sum"), [(0.0, 10.0), (0.6, 30.0), (0.8, 20.0)])
def test_spectra_contributions_use_escape_time(
    monkeypatch: pytest.MonkeyPatch,
    beta: float,
    expected_e_cmf_sum: float,
    patch_scan_packets: Callable[[pl.DataFrame], None],
) -> None:
    """Contribution spectra must use the same escape-time packet set and energy as the total spectrum."""
    nu_rf = at.constants.c_ang_per_s / 5000.0
    dfpackets = pl.DataFrame({
        "e_rf": [1.0, 2.0],
        "e_cmf": [10.0, 20.0],
        "t_arrive_d": [1.5, 3.0],
        "escape_time": [1.5 * at.constants.day_to_s, 2.2 * at.constants.day_to_s],
        "pellet_nucindex": [0, 0],
        "nu_rf": [nu_rf, nu_rf],
    })
    lambda_bin_edges = np.array([4000.0, 5000.0, 6000.0])

    def get_escape_surface_gamma(_modelpath: Path | str) -> float:
        return math.sqrt(1.0 - beta**2)

    def scan_nuclides(modelpath: Path | str) -> pl.LazyFrame:
        del modelpath
        return pl.LazyFrame({"pellet_nucindex": [0], "nucname": ["Ni56"]})

    patch_scan_packets(dfpackets)
    monkeypatch.setattr(atspectra, "get_escape_surface_gamma", get_escape_surface_gamma)
    monkeypatch.setattr(atspectra, "scan_nuclides", scan_nuclides)

    dfspectrum = atspectra.scan_spectra_from_packets(
        modelpath=Path(),
        timelowdays=1.0,
        timehighdays=2.0,
        lambda_bin_edges=lambda_bin_edges,
        use_time="escape",
        directionbins=[-1],
    )[-1].collect()
    contributions, array_flambda_emission_total, array_lambda = atspectra.get_flux_contributions_from_packets(
        modelpath=Path(),
        timelowdays=1.0,
        timehighdays=2.0,
        lambda_bin_edges=lambda_bin_edges,
        getabsorption=False,
        groupby="nuc",
        use_time="escape",
    )

    assert [contribution.linelabel for contribution in contributions] == ["Ni56"]
    assert np.array_equal(array_lambda, dfspectrum["lambda_angstroms"].to_numpy())
    assert np.allclose(array_flambda_emission_total, dfspectrum["f_lambda"].to_numpy(), rtol=1e-12, atol=0.0)

    integrated_flux = dfspectrum.select((pl.col("f_lambda") * pl.col("delta_lambda")).sum()).item()
    expected_flux = expected_e_cmf_sum / at.constants.day_to_s / (4 * math.pi * at.constants.megaparsec_to_cm**2)
    assert np.isclose(integrated_flux, expected_flux, rtol=1e-12, atol=0.0)


def test_spectra_escape_time_with_3d_model() -> None:
    """Escape-time spectra must get the outer velocity from metadata for a 3D model."""
    dfspectrum = atspectra.scan_spectra_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=3.0,
        timehighdays=8.0,
        lambda_bin_edges=np.linspace(3500.0, 8000.0, 101),
        use_time="escape",
        directionbins=[-1],
    )[-1].collect()

    assert dfspectrum["f_lambda"].is_finite().all()
    assert dfspectrum["f_lambda"].sum() > 0.0


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_spectra_timeseries_subplots(mockplot: mock.MagicMock) -> None:
    """-timedayslist draws one panel for each epoch, and each panel holds a different spectrum."""
    timedayslist = [295, 300]
    at.spectra.plot(
        argsraw=[], specpath=modelpath, outputfile=outputpath, timedayslist=timedayslist, multispecplot=True
    )

    axesofcalls = [call.args[0] for call in mockplot.call_args_list]
    assert len(axesofcalls) == len(timedayslist)
    assert axesofcalls[0] is not axesofcalls[1]
    fluxes = [np.asarray(call.args[2], dtype=np.float64) for call in mockplot.call_args_list]
    assert all(np.any(flux > 0.0) for flux in fluxes)
    assert not np.allclose(fluxes[0], fluxes[1], rtol=1e-3, atol=0.0)


def test_spectra_multispec_outputfile(tmp_path: Path) -> None:
    """A custom -outputfile must be honoured in multispecplot mode instead of falling back to plotspec.pdf in CWD."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / "custom_multispec.pdf",
        timedayslist=[295, 300],
        multispecplot=True,
    )
    assert (tmp_path / "custom_multispec.pdf").is_file()


def test_write_data_puts_reference_spectra_in_their_own_table(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--write_data must write one column for each ARTIS model, and the drawn points of a reference spectrum in a table.

    Only the ARTIS branch produces series data, so an unset variable used to raise UnboundLocalError for a
    reference spectrum alone, and to write the spectrum of the preceding model again as an incorrect extra column. A
    reference spectrum has its own wavelength grid, and the table then left it out.
    """
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, "2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "spec.pdf",
        timedays="300",
        write_data=True,
    )

    dfout = read_data_file(tmp_path / "spec.txt")
    assert [col for col in dfout.columns if col.startswith("f_lambda")] == ["f_lambda.TEST MODEL"]
    dfreferences = read_data_file(tmp_path / "spec_references.txt")
    assert dfreferences["series"].n_unique() == 1
    # two reference files can have one label, thus each row names its file
    assert dfreferences["specpath"].unique().to_list() == ["2003du_20031213_3219_8822_00.txt"]
    assert dfreferences.height > 100

    at.spectra.plot(
        argsraw=[],
        specpath=["2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "refonly.pdf",
        timedays="300",
        write_data=True,
    )
    assert (tmp_path / "refonly.pdf").is_file()
    assert not (tmp_path / "refonly.txt").exists()
    pltest.assert_frame_equal(read_data_file(tmp_path / "refonly_references.txt"), dfreferences)

    # the filter curves of --showfilterfunctions have their own wavelength grids, and no table held them
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath],
        outputfile=tmp_path / "filters.pdf",
        timedays="300",
        showfilterfunctions=True,
        normalised=True,
        write_data=True,
    )
    dffilters = read_data_file(tmp_path / "filters_references.txt")
    assert dffilters["series"].unique(maintain_order=True).to_list() == ["U", "B", "V", "I"]
    assert (dffilters["transmission"] <= 1.0).all()


@mock.patch.object(mplax.Axes, "set_title", side_effect=mplax.Axes.set_title, autospec=True)
def test_spectraplot_custom_title(mocksettitle: mock.MagicMock, tmp_path: Path) -> None:
    """-title text must be passed through to the axis title (previously store_true produced a title of 'True')."""
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / "customtitle.pdf",
        timemin=290,
        timemax=320,
        title="Custom title",
    )
    titles = [call[0][1] for call in mocksettitle.call_args_list]
    assert "Custom title" in titles
    assert all(isinstance(title, str) for title in titles)


def test_writespectra(tmp_path: Path) -> None:
    at.spectra.writespectra.main(argsraw=[], modelpath=modelpath, outputfile=tmp_path)

    assert list(tmp_path.glob("spectrum_ts*.txt"))


def test_writespectra_asks_for_all_the_models_before_the_first_model(tmp_path: Path) -> None:
    """A refusal for the second model leaves no files of the first model.

    Each model asked before its own files, thus a refusal for the second model came after the work of the first.
    """
    secondfolder = tmp_path / at.get_model_name(modelpath_classic_3d)
    secondfolder.mkdir()
    (secondfolder / "spectrum_ts05_5.00d.txt").write_text("x", encoding="utf-8")
    # pytest gives no terminal, thus nobody can answer, and the command stops without --force
    with pytest.raises(SystemExit):
        at.spectra.writespectra.main(argsraw=[str(modelpath), str(modelpath_classic_3d), "-o", str(tmp_path)])
    assert not list(tmp_path.glob(f"{at.get_model_name(modelpath)}/spectrum_ts*.txt"))


def test_writespectra_gives_two_models_of_one_name_two_folders(tmp_path: Path) -> None:
    """Two models of one name had one folder, and the second model replaced the files of the first with no prompt."""
    at.spectra.writespectra.main(argsraw=[str(modelpath), str(modelpath), "-o", str(tmp_path)])
    modelname = at.get_model_name(modelpath)
    assert list(tmp_path.glob(f"{modelname}/spectrum_ts*.txt"))
    assert list(tmp_path.glob(f"{modelname}_1/spectrum_ts*.txt"))


def test_writespectra_folder_names_stay_unique_when_a_model_has_a_name_with_an_index(tmp_path: Path) -> None:
    """Give the models foo, foo_2, and foo three folders. The third model took foo_2, the folder of the second."""
    names = {Path("a"): "foo", Path("b"): "foo_2", Path("c"): "foo"}
    with (
        mock.patch.object(at.spectra.writespectra, "get_model_name", side_effect=names.__getitem__),
        mock.patch.object(at.spectra.writespectra, "write_flambda_spectra") as mockwrite,
    ):
        at.spectra.writespectra.write_flambda_spectra_of_models(list(names), tmp_path, force=False)
    folders = [call.args[1] for call in mockwrite.call_args_list]
    assert len(set(folders)) == 3


def test_hiding_the_x_tick_labels_holds_the_width_and_the_frame(tmp_path: Path) -> None:
    """--hidexticklabels must not change the width of the file, nor the size of the frame.

    A paper puts several of these files in a grid that the author builds by hand. Each one goes in at
    one width, thus a file of a different width draws a frame of a different size beside its
    neighbour.
    """
    import pypdf

    import artistools.plottools as pt
    import artistools.spectra.plotspectra as ps

    sizes = {}
    for name, hide in (("shown", False), ("hidden", True)):
        frames: list[tuple[float, float]] = []
        realsave = pt.save_figure

        def spy(
            fig: t.Any,
            outpath: t.Any,
            frames: list[tuple[float, float]] = frames,
            realsave: t.Any = realsave,
            **kwargs: t.Any,
        ) -> None:
            fig.canvas.draw()
            figwidth, figheight = fig.get_size_inches()
            position = fig.axes[0].get_position()
            frames.append((position.width * figwidth, position.height * figheight))
            realsave(fig, outpath, **kwargs)

        outpath = tmp_path / f"{name}.pdf"
        with mock.patch.object(ps, "save_figure", spy):
            at.spectra.plot(argsraw=[], specpath=[modelpath], timedays=300, outputfile=outpath, hidexticklabels=hide)

        page = pypdf.PdfReader(outpath).pages[0].mediabox
        sizes[name] = (float(page.width) / 72.0, float(page.height) / 72.0, *frames[0])

    # the width of the file and the frame hold. The height falls by the labels that went, because a
    # crop takes the part of a margin that no label fills
    assert sizes["shown"][0] == pytest.approx(sizes["hidden"][0]), "the width of the file must not change"
    assert sizes["shown"][2:] == pytest.approx(sizes["hidden"][2:]), "the frame must not change"
    assert sizes["shown"][2:] == pytest.approx((pt.FRAMEWIDTH_INCHES, pt.FRAMEHEIGHT_INCHES))
    assert sizes["hidden"][1] < sizes["shown"][1], "the file loses the height of the labels"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_obsspec_draws_the_reference_spectrum(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """-obsspec names a reference spectrum, as a positional path does.

    Nothing read the list that -obsspec filled, thus the command took the file, drew the model alone,
    and said nothing about it.
    """
    refspec = "2003du_20031213_3219_8822_00.txt"
    at.spectra.plot(
        argsraw=[], specpath=[modelpath], refspecfiles=[refspec], timedays=300, outputfile=tmp_path / "obsspec.pdf"
    )

    labels = [str(call.kwargs.get("label")) for call in mockplot.call_args_list]
    assert len(labels) == 2, "the model and the reference spectrum must each give one series"
    assert any("SN2003du" in label for label in labels), "the reference spectrum must be drawn"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_empty_label_hides_a_reference_spectrum_in_the_legend(
    mockplot: mock.MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty -label gives a reference spectrum no legend entry, as it does for a model.

    The spectrum plot read an empty label as a missing label, thus the reference spectrum took the name in its
    metadata. The emission plot already kept the empty label. After the first correction, --scaletoreftime gave the
    line the legend entry " * 0.50", and the log and the summary named no series.
    """
    refspec = "2003du_20031213_3219_8822_00.txt"
    at.spectra.plot(
        argsraw=["-label", "default", ""],
        specpath=[modelpath, refspec],
        timedays=300,
        outputfile=tmp_path / "emptylabel.pdf",
    )

    labels = [call.kwargs.get("label") for call in mockplot.call_args_list]
    assert len(labels) == 2
    assert labels[1:] == [""]

    mockplot.reset_mock()
    capsys.readouterr()
    at.spectra.plot(
        argsraw=["-label", "default", ""],
        specpath=[modelpath, refspec],
        timedays=300,
        scaletoreftime=300,
        outputfile=tmp_path / "emptylabelscaled.pdf",
    )

    assert [call.kwargs.get("label") for call in mockplot.call_args_list][1:] == [""]
    assert "Reference spectrum ''" not in capsys.readouterr().out


def test_empty_label_keeps_the_name_of_a_reference_spectrum_in_the_tables(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """A reference spectrum with an empty -label keeps the label of its file in the tables and the residuals.

    The line had an empty label, thus the table of the reference spectra named it _child1, and the residual table
    named it "".
    """
    obsfile = write_fake_observed_spectrum(tmp_path)
    at.spectra.plot(
        argsraw=["-label", "default", ""],
        specpath=[modelpath, obsfile],
        timestep=54,
        residualbaselineseries=1,
        write_data=True,
        outputfile=tmp_path / "emptylabel.pdf",
    )

    assert read_data_file(tmp_path / "emptylabel_references.txt")["series"].unique().to_list() == ["fake observation"]
    assert read_data_file(tmp_path / "emptylabel_residuals.txt")["reference"].item() == "fake observation"


def test_emission_plot_leaves_a_reference_spectrum_with_an_empty_label_out_of_the_legend(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """An empty -label also gives a reference spectrum no entry in the legend of the emission plot.

    The emission plot gave the legend its labels as they were, thus the empty label gave an entry with no text. Its
    table of the reference spectra also had no file path, and the series name was _child15.
    """
    refspec = "2003du_20031213_3219_8822_00.txt"
    with mock.patch.object(plotspectra, "set_legend", side_effect=plotspectra.set_legend) as mocklegend:
        at.spectra.plot(
            argsraw=["-label", "default", "", "--use-thermalemissiontype"],
            specpath=[modelpath, refspec],
            timedays=300,
            showemission=True,
            write_data=True,
            outputfile=tmp_path / "emission.pdf",
        )

    assert "" not in mocklegend.call_args.kwargs["labels"]
    dfreferences = read_data_file(tmp_path / "emission_references.txt")
    assert dfreferences["specpath"].unique().to_list() == [refspec]
    assert not dfreferences["series"].str.starts_with("_").any()


@mock.patch.object(mplax.Axes, "set_yscale", side_effect=mplax.Axes.set_yscale, autospec=True)
def test_plotspectra_takes_the_yscale_argument(mockyscale: mock.MagicMock) -> None:
    """The command took --logscaley alone, thus -yscale worked on the light curves and not here."""
    at.spectra.plot(argsraw=[], specpath=[modelpath], yscale="log", timedays=300, outputfile=outputpath / "sp.pdf")
    assert [call.args[1] for call in mockyscale.call_args_list] == ["log"]

    mockyscale.reset_mock()
    at.spectra.plot(argsraw=[], specpath=[modelpath], yscale="linear", timedays=300, outputfile=outputpath / "sp.pdf")
    assert not mockyscale.call_args_list

    # more than half of the flux values of this model are zero, and a log axis hides each of them,
    # thus auto keeps the linear axis whatever range the values that remain cover
    mockyscale.reset_mock()
    at.spectra.plot(argsraw=[], specpath=[modelpath], yscale="auto", timedays=300, outputfile=outputpath / "sp.pdf")
    assert not mockyscale.call_args_list

    # the default is -yscale auto, thus the rule of the drawn values chooses the scale
    mockyscale.reset_mock()
    with mock.patch("artistools.plottools.wants_log_scale", return_value=True):
        at.spectra.plot(argsraw=[], specpath=[modelpath], timedays=300, outputfile=outputpath / "sp.pdf")
    assert {call.args[1] for call in mockyscale.call_args_list} == {"log"}
    # the linear axis had a bottom of zero, which put the data of the log axis in a thin band at the top
    assert all(call.args[0].get_ylim()[0] > 0.0 for call in mockyscale.call_args_list)

    # --logscaley replaces the default scale
    mockyscale.reset_mock()
    at.spectra.plot(argsraw=[], specpath=[modelpath], logscaley=True, timedays=300, outputfile=outputpath / "sp.pdf")
    assert [call.args[1] for call in mockyscale.call_args_list] == ["log"]


def test_a_unit_that_no_spectrum_takes_stops_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    """-xunit reads the name while argparse parses, thus a mistake stops the command before it reads a file.

    The command took every name and stopped later with "Unknown xunit", after it read the spectra.
    """
    import artistools.spectra.plotspectra

    parser = at.misc.argparser.SuggestingArgumentParser(prog="plotspectra")
    artistools.spectra.plotspectra.addargs(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([".", "-xunit", "micrometre"])

    assert "Did you mean micron?" in capsys.readouterr().err

    # a name of a unit is not case sensitive, and each spelling gives the canonical one
    assert parser.parse_args([".", "-xunit", "Angstrom"]).xunit == "angstrom"
    # the plural was the canonical name before, thus a script that gives it still works
    assert parser.parse_args([".", "-xunit", "angstroms"]).xunit == "angstrom"
    assert parser.parse_args([".", "-x", "MU"]).xunit == "micron"


@pytest.mark.parametrize("xunit", ["angstrom", "nm", "micron", "hz", "ev", "kev", "mev", "erg"])
def test_xunit_conversion_roundtrip(xunit: str) -> None:
    """Every unit accepted by convert_angstroms_to_unit must also be invertible by convert_unit_to_angstroms."""
    arr_lambda = np.array([1.0, 100.0, 5000.0, 1e5])

    arr_converted = at.spectra.convert_angstroms_to_unit(arr_lambda, xunit)
    arr_roundtrip = at.spectra.convert_unit_to_angstroms(arr_converted, xunit)

    assert arr_roundtrip == pytest.approx(arr_lambda, rel=1e-9)


def test_bin_spectrum_gives_the_mean_of_each_group() -> None:
    """bin_spectrum replaced a loop that took the mean x and the mean y of each group of nbins rows."""
    rng = np.random.default_rng(42)
    nrows, nbins = 23, 5
    wavelengths = np.sort(rng.uniform(3000.0, 9000.0, nrows))
    fluxes = rng.uniform(0.0, 1.0, nrows)

    expected_x = [np.mean(wavelengths[i : i + nbins]) for i in range(0, nrows, nbins)]
    expected_y = [np.mean(fluxes[i : i + nbins]) for i in range(0, nrows, nbins)]

    dfbinned = atspectra.bin_spectrum(pl.DataFrame({"x": wavelengths, "y": fluxes}), nbins, "x", "y")

    assert dfbinned.columns == ["x", "y"]
    assert dfbinned.height == 5, "the last group holds the three rows that remain"
    assert np.allclose(dfbinned["x"].to_numpy(), expected_x)
    assert np.allclose(dfbinned["y"].to_numpy(), expected_y)


def test_get_emabs_timeblock_count() -> None:
    """The row stride must come from each file's own size, not from shared state."""
    n_nu, n_timesteps = 5, 4

    # a normal file has one time block per timestep, a polarisation file has three (I, Q, U)
    dfplain = pl.DataFrame({"col": range(n_nu * n_timesteps)})
    dfpol = pl.DataFrame({"col": range(n_nu * n_timesteps * 3)})

    assert atspectra.get_emabs_timeblock_count(dfplain, n_nu, n_timesteps, "emission.out") == n_timesteps
    assert atspectra.get_emabs_timeblock_count(dfpol, n_nu, n_timesteps, "emissionpol.out") == 3 * n_timesteps

    # a row count that is not a whole number of frequency bins is rejected
    dfragged = pl.DataFrame({"col": range(n_nu * n_timesteps + 1)})
    with pytest.raises(AssertionError, match="not a multiple"):
        atspectra.get_emabs_timeblock_count(dfragged, n_nu, n_timesteps, "emission.out")

    # so is a whole-multiple count that matches neither the plain nor the polarisation layout
    dfwrong = pl.DataFrame({"col": range(n_nu * n_timesteps * 2)})
    with pytest.raises(AssertionError, match="time blocks per frequency bin"):
        atspectra.get_emabs_timeblock_count(dfwrong, n_nu, n_timesteps, "emission.out")


def test_scan_spectra_gamma_does_not_mix_in_uvoir_dirbins() -> None:
    """Gamma spectra must not pick up the UVOIR direction-resolved file.

    test-classicmode_3d has spec_res.out but no gamma_spec_res.out, so a gamma request must return only the
    spherically averaged bin rather than silently pairing gamma_spec.out with the UVOIR direction bins.
    """
    dirbin_spectra_uvoir = atspectra.scan_spectra(modelpath=modelpath_classic_3d, timestepmin=10, timestepmax=10)
    assert -1 in dirbin_spectra_uvoir
    assert len(dirbin_spectra_uvoir) > 1, "expected direction-resolved UVOIR spectra in this test model"

    dirbin_spectra_gamma = atspectra.scan_spectra(
        modelpath=modelpath_classic_3d, timestepmin=10, timestepmax=10, gamma=True
    )
    assert set(dirbin_spectra_gamma) == {-1}

    # the gamma spectrum must actually differ from the UVOIR one, i.e. it came from gamma_spec.out
    nu_gamma = dirbin_spectra_gamma[-1].select(pl.col("nu").max()).collect().item()
    nu_uvoir = dirbin_spectra_uvoir[-1].select(pl.col("nu").max()).collect().item()
    assert nu_gamma > nu_uvoir


def test_scan_spectra_rejects_averaging_over_both_angles() -> None:
    """Averaging over phi and theta at once leaves too few bins, so it must be rejected up front."""
    with pytest.raises(ValueError, match="both the phi and theta"):
        atspectra.scan_spectra(
            modelpath=modelpath_classic_3d,
            timestepmin=10,
            timestepmax=10,
            average_over_phi=True,
            average_over_theta=True,
        )


def test_reference_spectrum_named_out_is_not_read_as_artis(tmp_path: Path) -> None:
    """A user can name reference data with the .out suffix that ARTIS uses for its own output.

    path_is_artis_model reads that suffix, thus the predicate sent such a file to the ARTIS reader and
    the command drew nothing. The folder decides: an ARTIS model folder holds input.txt beside its output.
    """
    import shutil

    from artistools.spectra.plotspectra import path_is_reference_spectrum

    reference = at.get_path("artistools_dir") / "data" / "refspectra" / "2010lp_20110928_fors2.txt"
    shutil.copy(reference, tmp_path / "myref.out")
    shutil.copy(reference, tmp_path / "myref.txt")

    assert path_is_reference_spectrum(tmp_path / "myref.out")
    assert path_is_reference_spectrum(tmp_path / "myref.txt")

    # the output of a real ARTIS model keeps its own reading, whether the user names the folder or the file
    assert not path_is_reference_spectrum(modelpath)
    assert not path_is_reference_spectrum(modelpath / "spec.out")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraemissionplot_draws_a_reference_named_out(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The emission plot must draw a reference spectrum whose name ends in the .out suffix of ARTIS.

    The overlay loop read the suffix alone, thus it skipped such a file without a word.
    """
    import shutil

    source = at.get_path("artistools_dir") / "data" / "refspectra" / "2003du_20031213_3219_8822_00.txt"
    shutil.copy(source, tmp_path / "myref.out")
    shutil.copy(f"{source}.meta.yml", tmp_path / "myref.out.meta.yml")

    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, tmp_path / "myref.out"],
        outputfile=tmp_path,
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        use_thermalemissiontype=True,
    )

    labels = {callargs.kwargs.get("label") for callargs in mockplot.call_args_list}
    assert any(label and "2003du" in str(label) for label in labels), labels


def test_lambda_bin_edges_with_deltalambda_and_a_frequency_axis_stay_positive() -> None:
    """The unit conversion applies to a wavelength value, thus the code must not apply it to a wavelength interval."""
    lambda_bin_edges = atspectra.get_lambda_bin_edges(
        xmin_plot=atspectra.convert_angstroms_to_unit(19000.0, "hz"),
        xmax_plot=atspectra.convert_angstroms_to_unit(2500.0, "hz"),
        deltax=None,
        deltalogx=None,
        deltalambda=10.0,
        xunit="hz",
        modelpath=modelpath,
    )

    assert lambda_bin_edges[0] == pytest.approx(2495.0)
    assert lambda_bin_edges[-1] == pytest.approx(19005.0)
    assert np.all(np.diff(lambda_bin_edges) == pytest.approx(10.0))


def test_lambda_bin_edges_reject_a_zero_lower_limit_with_deltalogx() -> None:
    """A lower limit of zero made the multiplicative bin loop run without end."""
    with pytest.raises(ValueError, match="positive lower x limit"):
        atspectra.get_lambda_bin_edges(
            xmin_plot=0.0,
            xmax_plot=19000.0,
            deltax=None,
            deltalogx=0.05,
            deltalambda=None,
            xunit="angstrom",
            modelpath=modelpath,
        )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraplot_falls_back_to_the_angle_averaged_spectrum(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """A model with no spec_res.out shows direction bin -1 in place of the requested bin, and that bin needs a label."""
    at.spectra.plot(argsraw=[], specpath=[modelpath], plotviewingangle=[5], timedays=290, outputfile=tmp_path)

    assert list(tmp_path.glob("*.pdf"))
    (call,) = mockplot.call_args_list
    label = call.kwargs["label"]
    assert isinstance(label, str)
    assert label.startswith(at.get_model_name(modelpath))

    # the drawn series is the spectrum of bin -1 at that time
    timestep = at.get_timestep_of_timedays(modelpath, 290)
    dfexpected = at.spectra.scan_spectra(modelpath, timestep, timestep, directionbins=[-1])[-1].collect()
    drawnx = np.asarray(call.args[1], dtype=np.float64)
    drawny = np.asarray(call.args[2], dtype=np.float64)
    expectedy = np.interp(drawnx, dfexpected["lambda_angstroms"].to_numpy(), dfexpected["f_lambda"].to_numpy())
    assert np.allclose(drawny, expectedy, rtol=1e-6, atol=0.0)


def test_spectraplot_rejects_a_direction_bin_inside_an_average_group() -> None:
    """With --average_over_phi_angle, a bin that is not the first of its group has no label and no packets."""
    with pytest.raises(ValueError, match="first bin of an average group"):
        at.spectra.plot(
            argsraw=[], specpath=[modelpath], plotviewingangle=[5], average_over_phi_angle=True, timedays=290
        )


def test_output_spectra_rejects_a_file_name_for_the_output(tmp_path: Path) -> None:
    """--output_spectra writes a folder of files, thus -o with a file suffix is an error and not a fallback."""
    with pytest.raises(ValueError, match="must name a folder"):
        at.spectra.plot(argsraw=[], specpath=[modelpath], output_spectra=True, outputfile=tmp_path / "spectra.txt")


def test_plotspectra_refuses_two_models_whose_timestep_grids_disagree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse one -timestep that names different days on two timestep grids.

    The plot functions wrote the resolved range back onto args, thus the range of one model reached
    the next one. get_time_range refuses a timemin that sits after the last timestep of a model,
    thus it dropped the second model and printed one line. This is now an error, as it already is
    for the light curve command.
    """
    classic1dpath = at.get_path("testdata") / "test-classicmode_1d"

    with pytest.raises(SystemExit):
        at.spectra.plotspectra.main(
            argsraw=["-timestep", "30", str(modelpath), str(classic1dpath), "-outputfile", str(tmp_path)]
        )

    message = capsys.readouterr().err
    assert "timestep grids differ" in message
    assert "-timedays" in message, "the message must name the argument that works for both models"


def test_plotspectra_resolves_both_bounds_of_a_one_sided_time_range(tmp_path: Path) -> None:
    """-timemin alone left timemax as None, thus the output file name raised TypeError on the format."""
    at.spectra.plot(argsraw=["-timemin", "260", "--plotinvalidpart", str(modelpath), "-outputfile", str(tmp_path)])

    pdfnames = [path.name for path in tmp_path.glob("*.pdf")]
    assert len(pdfnames) == 1
    # both bounds reach the name, thus neither side formats a None
    assert pdfnames[0].startswith("plotspectra_260.")
    assert "None" not in pdfnames[0]


def test_plotspectra_emission_draws_the_bins_around_a_narrow_x_range(tmp_path: Path) -> None:
    """An x range that holds no centre of a bin draws the bin on each side of it, as a plot of the packets does.

    The text files gave no bin for such a range, and the command stopped.
    """
    outputfile = tmp_path / "narrow.pdf"
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=outputfile,
        timedays=300,
        emissionabsorption=True,
        use_thermalemissiontype=True,
        xmin=5000.0,
        xmax=5000.5,
    )
    assert outputfile.is_file()


@mock.patch("artistools.spectra.core.get_flux_contributions_from_packets")
def test_plotspectra_emission_takes_an_x_unit_other_than_angstroms(
    mockgetcontributions: mock.MagicMock, tmp_path: Path
) -> None:
    """The check for an empty x range must not compare the limits in keV with bins in Angstroms."""
    contribution_list: list[atspectra.FluxContributionTuple] = []
    mockgetcontributions.return_value = (contribution_list, np.ones(3), np.array([0.01, 0.02, 0.03]))

    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / "gamma.pdf",
        timemin=290.0,
        timemax=320.0,
        emissionabsorption=True,
        frompackets=True,
        xunit="kev",
        xmin=100.0,
        xmax=2000.0,
    )

    assert (tmp_path / "gamma.pdf").is_file()


def test_plotspectra_timestep_with_only_a_reference_spectrum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A reference spectrum has no timesteps, thus -timestep has no effect on it and the plot still runs.

    The range resolution in main asked for an ARTIS model and stopped before it drew the reference spectrum.
    """
    at.spectra.plot(argsraw=["-timestep", "30", "sn2011fe_PTF11kly_20120822_norm.txt", "-outputfile", str(tmp_path)])

    assert len(list(tmp_path.glob("*.pdf"))) == 1
    assert "-timestep names a timestep of a model" in capsys.readouterr().err


def test_read_emission_absorption_file_reads_a_new_version(tmp_path: Path) -> None:
    """The cache of an emission file gives the new data after a simulation writes the file again.

    The key of the cache held only the path, thus the viewer of plotspectra showed the old contributions.
    """
    emissionfile = tmp_path / "emission.out"
    emissionfile.write_text("1.0 2.0\n3.0 4.0\n")
    assert at.spectra.core.read_emission_absorption_file(emissionfile).row(0) == (1.0, 2.0)

    emissionfile.write_text("5.0 6.0 7.0\n8.0 9.0 10.0\n")
    assert at.spectra.core.read_emission_absorption_file(emissionfile).row(0) == (5.0, 6.0, 7.0)


def test_plotspectra_notimeclamp_keeps_a_one_sided_bound(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--notimeclamp keeps the -timemin that the user gave. The range resolution of main clamped it first.

    spec.out holds whole timesteps, thus --notimeclamp also makes plotspectra read the packets.
    """
    at.spectra.plot(
        argsraw=["--notimeclamp", "-timemin", "260", "--plotinvalidpart", str(modelpath), "-outputfile", str(tmp_path)]
    )

    pdfnames = [path.name for path in tmp_path.glob("*.pdf")]
    assert pdfnames == ["plotspectra_260.00d-350.00d.pdf"]
    assert "Enabling --frompackets, since --notimeclamp was specified" in capsys.readouterr().out


def test_plotspectra_skips_a_folder_that_is_not_a_model_folder(tmp_path: Path) -> None:
    """A folder without input.txt gives no timesteps, thus the plot of the other models must still run.

    The range resolution of main asked the first folder for its timesteps and stopped before any plot.
    """
    notamodel = tmp_path / "notamodel"
    notamodel.mkdir()
    outputfolder = tmp_path / "output"
    outputfolder.mkdir()

    at.spectra.plot(argsraw=["-timedays", "300", str(notamodel), str(modelpath), "-outputfile", str(outputfolder)])

    assert len(list(outputfolder.glob("*.pdf"))) == 1


def test_plotspectra_one_sided_bound_that_the_first_model_does_not_reach(tmp_path: Path) -> None:
    """A -timemin after the end of the first model must take its range from a later model.

    Only the first model resolved the range, thus timemax stayed None and the output file name raised TypeError.
    """
    at.spectra.plot(
        argsraw=[
            "-timemin",
            "260",
            "--plotinvalidpart",
            str(at.get_path("testdata") / "test-classicmode_1d"),
            str(modelpath),
            "-outputfile",
            str(tmp_path),
        ]
    )

    pdfnames = [path.name for path in tmp_path.glob("*.pdf")]
    assert len(pdfnames) == 1
    assert "None" not in pdfnames[0]


def test_plotspectra_takes_a_reference_named_out_before_a_model(tmp_path: Path) -> None:
    """A reference spectrum can carry the .out suffix of ARTIS, thus its folder holds no timesteps.

    The time range resolution asked such a folder for the timesteps of a model and raised
    FileNotFoundError before any plot ran.
    """
    import shutil

    source = at.get_path("artistools_dir") / "data" / "refspectra" / "2003du_20031213_3219_8822_00.txt"
    shutil.copy(source, tmp_path / "myref.out")
    shutil.copy(f"{source}.meta.yml", tmp_path / "myref.out.meta.yml")

    outputfolder = tmp_path / "out"
    outputfolder.mkdir()
    at.spectra.plot(
        argsraw=["-timedays", "260-300", str(tmp_path / "myref.out"), str(modelpath), "-outputfile", str(outputfolder)]
    )

    assert len(list(outputfolder.glob("*.pdf"))) == 1


def test_plotspectra_timedayslist_names_the_whole_range(tmp_path: Path) -> None:
    """-timedayslist names one epoch for each subplot, thus the file name must span the whole list.

    The resolution took args.timedays, which holds the first epoch alone, thus two lists that share
    their first epoch wrote one file name and the second plot overwrote the first.
    """
    names = []
    for lastday in ("300", "330"):
        outputfolder = tmp_path / lastday
        outputfolder.mkdir()
        at.spectra.plot(
            argsraw=[
                "-timedayslist",
                "260",
                lastday,
                "--plotinvalidpart",
                str(modelpath),
                "-outputfile",
                str(outputfolder),
            ]
        )
        pdfnames = [path.name for path in outputfolder.glob("*.pdf")]
        assert len(pdfnames) == 1
        names.append(pdfnames[0])

    assert names[0] != names[1], "two lists that differ in the last epoch must not share one file name"
    # the upper bound follows the last epoch of the list, not the first
    assert "300." in names[0]
    assert "330." in names[1]


def test_plotspectra_showtime_needs_a_time_range(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--showtime writes the middle of the time range, thus it needs both bounds.

    A reference spectrum has no timesteps, thus no path could resolve a range. The annotation then
    added two None values and raised TypeError.
    """
    with pytest.raises(SystemExit) as excinfo:
        at.spectra.plot(argsraw=["--showtime", "2003du_20031213_3219_8822_00.txt", "-outputfile", str(tmp_path)])

    assert excinfo.value.code == 1
    assert "--showtime" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.pdf"))


def test_plotspectra_multispecplot_needs_an_epoch_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--multispecplot draws one subplot for each epoch of -timedayslist, thus it needs that list.

    The row count read len(None) and raised TypeError. -timedayslist sets --multispecplot, thus only
    the flag on its own reaches this case.
    """
    with pytest.raises(SystemExit) as excinfo:
        at.spectra.plot(argsraw=["--multispecplot", "-timedays", "260", str(modelpath), "-outputfile", str(tmp_path)])

    assert excinfo.value.code == 1
    assert "-timedayslist" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.pdf"))


def write_fake_observed_spectrum(folder: Path) -> Path:
    """Write a reference spectrum of 1.1 times the model of timestep 54."""
    dfmodel = at.spectra.scan_spectra(modelpath, timestepmin=54, timestepmax=54)[-1].collect()
    obsfile = folder / "fakeobs.txt"
    dfmodel.select("lambda_angstroms", (pl.col("f_lambda") * 1.1).alias("flux")).sort("lambda_angstroms").write_csv(
        obsfile, separator=" ", include_header=False
    )
    obsfile.with_name(f"{obsfile.name}.meta.yml").write_text(
        "dist_mpc: 1\nlabel: fake observation\nt: 300\n", encoding="utf-8"
    )
    return obsfile


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectra_residual_panel_gives_model_minus_reference(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """An observed flux of 1.1 times the model gives a negative absolute residual and a ratio of 1 / 1.1."""
    obsfile = write_fake_observed_spectrum(tmp_path)
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, obsfile],
        timestep=54,
        residualbaselineseries=1,
        residualtype="absolute",
        write_data=True,
        outputfile=tmp_path / "residuals.pdf",
    )
    dfstats = read_data_file(tmp_path / "residuals_residuals.txt")
    assert dfstats.height == 1
    assert dfstats["reference"].item() == "fake observation"
    assert dfstats["npoints"].item() > 100
    assert dfstats["rms"].item() > 0.0
    assert 0.0 < dfstats["rms_relative"].item() < 0.1 / 1.1 * 3.0

    # the last plot call draws the residual line: the model is below the observed flux at each point
    residual = np.asarray(mockplot.call_args_list[-1].args[2])
    assert (residual <= 0.0).all()
    assert (residual < 0.0).any()
    # the table gave the statistics of the residual panel alone, and not its curve
    dfcurves = read_data_file(tmp_path / "residuals_residualcurves.txt")
    assert dfcurves["reference"].unique().to_list() == ["fake observation"]
    assert np.allclose(dfcurves["residual"].to_numpy(), residual, rtol=1e-6, atol=0, equal_nan=True)
    assert np.allclose(dfcurves["x"].to_numpy(), np.asarray(mockplot.call_args_list[-1].args[1]), rtol=1e-9, atol=0)

    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, obsfile],
        timestep=54,
        residualbaselineseries=1,
        residualtype="relative",
        logscaley=True,
        outputfile=tmp_path / "ratio.pdf",
    )
    # a reference flux of zero gives a gap
    ratio = np.asarray(mockplot.call_args_list[-1].args[2])
    assert np.isfinite(ratio).any()
    assert np.allclose(ratio[np.isfinite(ratio)], 1.0 / 1.1)


@pytest.mark.parametrize("referenceonly", [False, True])
def test_spectra_residual_panel_compares_series_of_the_same_type(
    tmp_path: Path, referenceonly: bool, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The default baseline is series 0, with two models or two reference spectra."""
    specpath = write_fake_observed_spectrum(tmp_path) if referenceonly else modelpath
    at.spectra.plot(
        argsraw=["-residuals"],
        specpath=[specpath, specpath],
        timestep=54,
        label=["baseline", "comparison"],
        write_data=True,
        outputfile=tmp_path / "same_type.pdf",
    )
    dfstats = read_data_file(tmp_path / "same_type_residuals.txt")
    assert dfstats["model"].item() == "comparison"
    assert dfstats["reference"].item() == "baseline"
    assert np.isclose(dfstats["rms"].item(), 0.0)


def test_residual_keywords_select_the_baseline_and_comparisons(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Keep the baseline keyword separate from the selected series."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath] * 3,
        timestep=54,
        label=["excluded", "baseline", "comparison"],
        residualbaselineseries=1,
        residuals=[2],
        write_data=True,
        outputfile=tmp_path / "selected.pdf",
    )
    stats = read_data_file(tmp_path / "selected_residuals.txt")
    assert stats["model"].to_list() == ["comparison"]
    assert stats["reference"].to_list() == ["baseline"]
    assert np.isclose(stats["rms"].item(), 0.0)


@pytest.mark.parametrize("baselineindex", [0, 1, 2])
def test_residual_baseline_counts_codecomparison_spectra(
    tmp_path: Path, baselineindex: int, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Check each baseline index with a code comparison spectrum between two other spectra."""
    wavelengths = np.array([3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 9000.0])
    data = pl.DataFrame({"lambda": wavelengths, "300.0": np.arange(1.0, 7.0) * 1e40})
    obsfile = write_fake_observed_spectrum(tmp_path)
    with (
        mock.patch.object(plotspectra, "read_reference_spectra", return_value=(data, np.array([300.0]))),
        mock.patch.object(plotspectra, "draw_residual_panel", wraps=plotspectra.draw_residual_panel) as mockdraw,
    ):
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath, "codecomparison/DDC10/testcode", obsfile],
            timemin=290,
            timemax=310,
            distmpc=1.0,
            color=["red", "blue", "green"],
            residualbaselineseries=baselineindex,
            write_data=True,
            outputfile=tmp_path / "codecomparison.pdf",
        )
    series = mockdraw.call_args.args[2]
    axis = mockdraw.call_args.args[1]
    assert len(series) == 3
    assert series[1].label == "codecomparison/DDC10/testcode 300.0d"
    assert np.allclose(series[1].x, wavelengths)
    for item, line in zip(series, axis.lines, strict=True):
        assert np.allclose(item.x, np.asarray(line.get_xdata(), dtype=np.float64))
        assert np.allclose(item.y, np.asarray(line.get_ydata(), dtype=np.float64), atol=0.0)
        assert item.color == line.get_color()
    stats = read_data_file(tmp_path / "codecomparison_residuals.txt")
    assert stats["reference"].to_list() == [series[baselineindex].label] * 2
    assert stats["model"].to_list() == [item.label for index, item in enumerate(series) if index != baselineindex]
    assert (stats["npoints"] > 0).all()


@pytest.mark.parametrize("baselineindex", range(5))
@pytest.mark.parametrize("xunit", ["angstrom", "nm"])
def test_residual_baseline_counts_filter_curves(
    tmp_path: Path, baselineindex: int, xunit: str, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Check each filter index and its data in both units of wavelength."""
    with mock.patch.object(plotspectra, "draw_residual_panel", wraps=plotspectra.draw_residual_panel) as mockdraw:
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath],
            timestep=54,
            normalised=True,
            showfilterfunctions=True,
            xunit=xunit,
            residualbaselineseries=baselineindex,
            write_data=True,
            outputfile=tmp_path / "filters.pdf",
        )
    series = mockdraw.call_args.args[2]
    axis = mockdraw.call_args.args[1]
    assert len(series) == 5
    assert [item.label for item in series[1:]] == ["U", "B", "V", "I"]
    for item, line in zip(series, axis.lines, strict=True):
        assert np.allclose(item.x, np.asarray(line.get_xdata(), dtype=np.float64))
        assert np.allclose(item.y, np.asarray(line.get_ydata(), dtype=np.float64))
        assert item.color == line.get_color()
    stats = read_data_file(tmp_path / "filters_residuals.txt")
    assert stats["reference"].to_list() == [series[baselineindex].label] * stats.height
    assert stats.height > 0
    if baselineindex == 0:
        assert stats["model"].to_list() == ["U", "B", "V", "I"]
    else:
        assert stats["model"][0] == series[0].label
        assert series[baselineindex].label not in stats["model"].to_list()


def test_spectra_residual_panel_refuses_a_plot_with_no_pair(tmp_path: Path) -> None:
    """-residual needs at least two series and one frame."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[], specpath=[modelpath], timestep=54, residualbaselineseries=0, outputfile=tmp_path / "a.pdf"
        )

    obsfile = write_fake_observed_spectrum(tmp_path)
    # --output_spectra draws no figure, thus it cannot hold a residual panel
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath, obsfile],
            output_spectra=True,
            residualbaselineseries=1,
            outputfile=tmp_path,
        )

    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath, obsfile],
            timedayslist=["280", "300"],
            residualbaselineseries=1,
            outputfile=tmp_path / "b.pdf",
        )


def get_saved_axes(**plotargs: t.Any) -> list[mplax.Axes]:
    """Return the axes of the figure that plotspectra saves, for a test that reads the axis properties."""
    savedaxes: list[mplax.Axes] = []

    def save_figure(fig: mplfig.Figure, *_args: t.Any, **_kwargs: t.Any) -> None:
        savedaxes.extend(fig.axes)

    with mock.patch.object(at.spectra.plotspectra, "save_figure", save_figure):
        at.spectra.plot(argsraw=[], **plotargs)

    return savedaxes


def test_plotspectra_ymin_alone_keeps_the_top_of_the_data(tmp_path: Path) -> None:
    """-ymin sets the bottom of the y range, and the top still follows the drawn flux.

    A limit stops the autoscale of both sides. The command applied the limits before it drew a line,
    thus -ymin alone froze the view at 0 to 1 and no spectrum was visible.
    """
    (axis,) = get_saved_axes(
        specpath=modelpath, outputfile=tmp_path / "ymin.pdf", timemin=290, timemax=320, ymin=0.0, yscale="linear"
    )

    bottom, top = axis.get_ylim()
    assert bottom == 0
    assert 0.0 < top < 1e-9

    # the axes carry no y margin, thus a top on the tallest peak clips it
    datatop = max(float(np.asarray(line.get_ydata(), dtype=np.float64).max()) for line in axis.get_lines())
    assert 1.02 * datatop < top < 1.1 * datatop


def test_plotspectra_multispecplot_prunes_the_ticks_of_a_log_axis(tmp_path: Path) -> None:
    """-yscale auto must choose the scale of the panels before the command sets their locators.

    The lowest tick of one panel meets the highest tick of the panel below it. Thus a stack of panels
    needs a locator that prunes both ends. The earlier code set the locators before it chose the scale.
    """
    import artistools.plottools as pt

    with mock.patch.object(pt, "wants_log_scale", return_value=True):
        axes = get_saved_axes(
            specpath=modelpath, outputfile=tmp_path / "multispec_log.pdf", timedayslist=["290", "300"], yscale="auto"
        )

    assert len(axes) == 2
    for axis in axes:
        assert axis.get_yscale() == "log"
        assert isinstance(axis.yaxis.get_major_locator(), pt.PrunedLogLocator)


def test_plotspectra_multispecplot_draws_a_reference_on_every_panel(tmp_path: Path) -> None:
    """--multispecplot gives each panel one epoch of the model, and every panel shows the reference spectrum.

    Each reference spectrum went on the panel of its own index. Thus one reference reached the first
    panel alone, and more references than epochs raised IndexError.
    """
    axes = get_saved_axes(
        specpath=[modelpath, "sn2011fe_PTF11kly_20120822_norm.txt"],
        outputfile=tmp_path / "multispec_ref.pdf",
        timedayslist=["290", "300"],
        distmpc=1.0,
    )

    assert len(axes) == 2
    for axis in axes:
        assert any(line.get_label() == "SN2011fe +364d" for line in axis.get_lines())


def test_plotspectra_write_data_names_each_direction_bin(tmp_path: Path) -> None:
    """--write_data writes one column for each direction bin, and not the last bin alone."""
    outputfile = tmp_path / "dirbins.pdf"
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath_classic_3d,
        outputfile=outputfile,
        timemin=4,
        timemax=6.5,
        plotviewingangle=[0, 5],
        write_data=True,
    )

    columns = pl.read_csv(outputfile.with_suffix(".txt"), separator=" ", comment_prefix="#").columns
    assert sum(column.startswith("f_lambda") and column.endswith("_dirbin00") for column in columns) == 1
    assert sum(column.startswith("f_lambda") and column.endswith("_dirbin05") for column in columns) == 1


def test_scan_specpol_data_reads_one_direction_bin() -> None:
    """ARTIS writes one file specpol_res.out that holds a table for each direction bin.

    The reader asked for specpol_res_<dirbin>.out, which no model folder holds, thus every direction bin
    raised FileNotFoundError.
    """
    dfdirbin = at.spectra.scan_specpol_data(dirbin=0, modelpath=modelpath_classic_3d)["I"].collect()
    dfaveraged = at.spectra.scan_specpol_data(dirbin=-1, modelpath=modelpath_classic_3d)["I"].collect()

    assert dfdirbin.columns == dfaveraged.columns
    assert dfdirbin.columns[0] == "nu"
    assert dfdirbin.height == dfaveraged.height
    assert not dfdirbin.equals(dfaveraged)


def test_vpkt_exclusion_removes_the_absorption_of_an_excluded_packet() -> None:
    """--vpkt_match_emission_exclusion_to_opac removes each excluded packet from the emission and the absorption.

    The code filtered only the emission after it binned both, thus the absorption kept the excluded packets.
    """
    c = at.constants.c_ang_per_s
    # packet 1: last emission in an Fe II line (line 0) at 5000 A, absorbed before in a Co II line (line 1) at 4000 A.
    # packet 2: last emission in the Co II line at 6000 A, absorbed before in the Fe II line at 4500 A
    packets = pl.LazyFrame(
        {
            "dir0_nu_rf": [c / 5000.0, c / 6000.0],
            "dir0_t_arrive_d": [5.0, 5.0],
            "dir0_e_rf_0": [1.0, 1.0],
            "dir0_e_rf_1": [1.0, 1.0],
            "emissiontype": [0, 1],
            "absorption_type": [1, 0],
            "absorption_freq": [c / 4000.0, c / 4500.0],
        },
        schema={
            "dir0_nu_rf": pl.Float32,
            "dir0_t_arrive_d": pl.Float32,
            "dir0_e_rf_0": pl.Float64,
            "dir0_e_rf_1": pl.Float64,
            "emissiontype": pl.Int32,
            "absorption_type": pl.Int32,
            "absorption_freq": pl.Float32,
        },
    )
    # the second spectrum of the direction excludes the emission of iron (Z = 26)
    vpktconfig: dict[str, t.Any] = {
        "nobsdirections": 1,
        "nspectraperobs": 2,
        "z_excludelist": [0, 26],
        "time_limits_enabled": 0,
        "custom_lambda_ranges": [],
    }
    bflist = pl.LazyFrame({"bfindex": [], "ion_str": []}, schema={"bfindex": pl.Int32, "ion_str": pl.String})
    with (
        mock.patch.object(atspectra, "scan_virtual_packets", return_value=(1, packets)),
        mock.patch.object(atspectra, "get_vpkt_config", return_value=vpktconfig),
        mock.patch.object(
            atspectra,
            "get_linelist_label_columns",
            return_value=pl.DataFrame({"atomic_number": [26, 27], "ion_stage": [2, 2]}),
        ),
        mock.patch.object(atspectra, "scan_bflist", return_value=bflist),
    ):
        contributions, _, array_lambda = atspectra.get_flux_contributions_from_packets(
            Path(),
            timelowdays=4.0,
            timehighdays=6.0,
            lambda_bin_edges=np.arange(3000.0, 8000.0, 100.0),
            groupby="ion",
            directionbin=1,
            directionbins_are_vpkt_observers=True,
            vpkt_match_emission_exclusion_to_opac=True,
        )

    def get_wavelengths(fluxes: npt.NDArray[np.floating]) -> list[float]:
        return [float(value) for value in array_lambda[fluxes > 0.0]]

    series = {
        row.linelabel: (get_wavelengths(row.array_flambda_emission), get_wavelengths(row.array_flambda_absorption))
        for row in contributions
    }
    assert series == {"Co II": ([6050.0], []), "Fe II": ([], [4550.0])}


def test_vpkt_contributions_leave_out_the_emission_outside_the_vpkt_ranges() -> None:
    """A virtual packet with nu_rf outside the ranges of vpkt.txt gives absorption, but no emission.

    ARTIS writes such a row when the absorption frequency is inside a range, and it keeps the row out of vspecpol.
    The code added its emission at nu_rf, thus the bins outside the ranges held a biased part of the flux.
    """
    c = at.constants.c_ang_per_s
    # packet 1: emission in the Fe II line at 5000 A, absorption in the Co II line at 4000 A. Packet 2: emission in
    # the Co II line at 6000 A, outside the range, and absorption in the Fe II line at 4500 A, inside the range
    packets = pl.LazyFrame(
        {
            "dir0_nu_rf": [c / 5000.0, c / 6000.0],
            "dir0_t_arrive_d": [5.0, 5.0],
            "dir0_e_rf_0": [1.0, 1.0],
            "emissiontype": [0, 1],
            "absorption_type": [1, 0],
            "absorption_freq": [c / 4000.0, c / 4500.0],
        },
        schema={
            "dir0_nu_rf": pl.Float64,
            "dir0_t_arrive_d": pl.Float32,
            "dir0_e_rf_0": pl.Float64,
            "emissiontype": pl.Int32,
            "absorption_type": pl.Int32,
            "absorption_freq": pl.Float64,
        },
    )
    vpktconfig: dict[str, t.Any] = {
        "nobsdirections": 1,
        "nspectraperobs": 1,
        "z_excludelist": [0],
        "time_limits_enabled": 0,
        "custom_lambda_ranges": [(3500.0, 5500.0)],
    }
    bflist = pl.LazyFrame({"bfindex": [], "ion_str": []}, schema={"bfindex": pl.Int32, "ion_str": pl.String})
    with (
        mock.patch.object(atspectra, "scan_virtual_packets", return_value=(1, packets)),
        mock.patch.object(atspectra, "get_vpkt_config", return_value=vpktconfig),
        mock.patch.object(
            atspectra,
            "get_linelist_label_columns",
            return_value=pl.DataFrame({"atomic_number": [26, 27], "ion_stage": [2, 2]}),
        ),
        mock.patch.object(atspectra, "scan_bflist", return_value=bflist),
    ):
        contributions, _, array_lambda = atspectra.get_flux_contributions_from_packets(
            Path(),
            timelowdays=4.0,
            timehighdays=6.0,
            lambda_bin_edges=np.arange(3000.0, 8000.0, 100.0),
            groupby="ion",
            directionbin=0,
            directionbins_are_vpkt_observers=True,
        )

    def get_wavelengths(fluxes: npt.NDArray[np.floating]) -> list[float]:
        return [float(value) for value in array_lambda[fluxes > 0.0]]

    series = {
        row.linelabel: (get_wavelengths(row.array_flambda_emission), get_wavelengths(row.array_flambda_absorption))
        for row in contributions
    }
    assert series == {"Fe II": ([5050.0], [4550.0]), "Co II": ([], [4050.0])}

    vpktconfig |= {"time_limits_enabled": 1, "initial_time": 4.5, "final_time": 10.0}
    with (
        mock.patch.object(atspectra, "scan_virtual_packets", return_value=(1, packets)),
        mock.patch.object(atspectra, "get_vpkt_config", return_value=vpktconfig),
        pytest.raises(ValueError, match=r"outside the virtual packets, which cover 4\.5 to 10\.0 days"),
    ):
        atspectra.get_flux_contributions_from_packets(
            Path(),
            timelowdays=4.0,
            timehighdays=6.0,
            lambda_bin_edges=np.arange(3000.0, 8000.0, 100.0),
            groupby="ion",
            directionbin=0,
            directionbins_are_vpkt_observers=True,
        )


def test_reference_spectrum_de_redshift_scales_the_flux(tmp_path: Path) -> None:
    """A de-redshift divides the wavelength by (1 + z) and multiplies f_lambda by (1 + z).

    The same energy falls in a narrower band in the rest frame, thus the flux density rises.
    """
    redshift = 0.5
    specfile = tmp_path / "redshifted.txt"
    specfile.write_text("4000 1.0\n5000 2.0\n6000 3.0\n", encoding="utf-8")
    specfile.with_suffix(".txt.meta.yml").write_text(f"---\nz: {redshift}\n", encoding="utf-8")

    specdata = atspectra.get_reference_spectrum(specfile, at.misc.get_file_metadata(specfile))

    lambda_obs = np.array([4000.0, 5000.0, 6000.0])
    assert np.allclose(specdata["lambda_angstroms"].to_numpy(), lambda_obs / (1 + redshift), atol=0.0)
    assert np.allclose(specdata["f_lambda"].to_numpy(), np.array([1.0, 2.0, 3.0]) * (1 + redshift), atol=0.0)


def test_plotspectra_normalised_log_axis_keeps_room_above_the_peak() -> None:
    """A normalised spectrum on a log axis keeps a margin above its peak of 1.

    The default -ymax 1.10 of --normalised gave the room, and a log axis had no margin after its removal.
    """
    args = at.misc.parse_cli_args(
        plotspectra.addargs, None, None, [str(modelpath), "-t", "300", "--normalised", "-yscale", "log"]
    )
    plotspectra.resolve_plot_args(args)
    fig, axes, residualaxis = plotspectra.make_plot_figure(args)
    plotspectra.draw_plot(args, axes, residualaxis)
    fig.canvas.draw()
    assert axes[0].get_yscale() == "log"
    assert axes[0].get_ylim()[1] > 1.02
    plt.close(fig)


@mock.patch("artistools.spectra.core.get_flux_contributions_from_packets")
def test_spectraemissionplot_forwards_the_velocity_ranges(mockgetcontributions: mock.MagicMock, tmp_path: Path) -> None:
    """The velocity range arguments must reach the packet reader that selects the contributions."""
    contribution_list: list[atspectra.FluxContributionTuple] = []
    mockgetcontributions.return_value = (contribution_list, np.zeros(2), np.array([4000.0, 5000.0]))

    at.spectra.plot(
        argsraw=[],
        specpath=modelpath_classic_3d,
        outputfile=tmp_path / "velocityranges_forwarded.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        emissionradialvelocityrange=["0.04c", "0.06c"],
        emissionlineofsightvelocityrange=[-15000, 15000],
    )

    velocityranges = mockgetcontributions.call_args.kwargs["velocityranges"]
    assert set(velocityranges) == {"radialvelocity", "lineofsightvelocity"}
    assert np.isclose(velocityranges["radialvelocity"][0], 0.04 * at.constants.C_cm_per_s / 1e5, rtol=1e-9, atol=0.0)
    assert np.isclose(velocityranges["radialvelocity"][1], 0.06 * at.constants.C_cm_per_s / 1e5, rtol=1e-9, atol=0.0)
    assert velocityranges["lineofsightvelocity"] == (-15000.0, 15000.0)


def test_scan_spec_follows_the_working_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The spectra of the default model path must change with the working folder.

    A cache held the relative Path("."). Thus a second model in the same process got the spectra of
    the first model.
    """
    for foldername, flux in (("modelA", 1.0), ("modelB", 2.0)):
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "spec.out").write_text(f"0 10.0\n1e15 {flux}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path / "modelA")
    assert atspectra.scan_spec(Path()).collect()["10.0"].item() == pytest.approx(1.0)

    monkeypatch.chdir(tmp_path / "modelB")
    assert atspectra.scan_spec(Path()).collect()["10.0"].item() == pytest.approx(2.0)


def test_scan_spec_of_a_cut_or_empty_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """sn3d writes spec.out again at each timestep. A run that stops during that write leaves a bad file.

    An empty file gave a polars NoDataError with no file name. A cut last line gave null values or its cut number as
    the value. A cut compressed file gave an EOFError or a polars panic with no file name.
    """
    specpath = tmp_path / "spec.out"
    specpath.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no spectrum"):
        atspectra.scan_spec(tmp_path)

    specpath.write_text("0 10.0 11.0\n1e15 1.0 2.0\n2e15 3.0 4.", encoding="utf-8")
    atspectra.scan_spec.cache_clear()
    assert atspectra.scan_spec(tmp_path).collect()["nu"].to_list() == [1e15]
    assert "ends with a cut line" in capsys.readouterr().err

    specpath.write_text("0 10.0 11.0\n1e15 1.0 2.0\n2e15 3.0\n", encoding="utf-8")
    atspectra.scan_spec.cache_clear()
    with pytest.raises(ValueError, match="a line with fewer values than the header"):
        atspectra.scan_spec(tmp_path)

    fulltext = (modelpath / "spec.out").read_bytes()
    specpath.unlink()
    for compressedname, compress in (("spec.out.gz", gzip.compress), ("spec.out.xz", lzma.compress)):
        compressedpath = tmp_path / compressedname
        compressedpath.write_bytes(compress(fulltext)[:3000])
        atspectra.scan_spec.cache_clear()
        with pytest.raises(OSError, match="ends before the end") as excinfo:
            atspectra.scan_spec(tmp_path)
        assert any(compressedname in note for note in [str(excinfo.value), *getattr(excinfo.value, "__notes__", [])])
        compressedpath.unlink()


@pytest.mark.parametrize(
    ("extraargs", "message"),
    [
        (["-stokesparam", "Q"], "reads the virtual packet spectra"),
        (["-stokesparam", "Q", "-plotvspecpol", "0", "--frompackets"], "reads the virtual packet spectra"),
        (["--showemission", "-timedayslist", "290", "320"], "draws one time"),
        (["--showemission", "-yvariable", "packetcount"], "has no count of packets"),
    ],
)
def test_plotspectra_refuses_a_quantity_that_the_series_lacks(
    extraargs: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each of these plots drew the wrong quantity or the wrong time, or it stopped with a missing column.

    -stokesparam Q drew Stokes I for spec.out, an emission plot of -timedayslist drew the first time alone,
    and -yvariable packetcount stopped an emission plot with ColumnNotFoundError.
    """
    import artistools.__main__

    with pytest.raises(SystemExit):
        artistools.__main__.main(argsraw=["plotspectra", str(modelpath), "-t", "300", *extraargs])
    assert message in capsys.readouterr().err


def test_polarisation_plot_refuses_a_y_range_that_is_not_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """-ymin must be below -ymax. The plot of a Stokes ratio stopped with an AssertionError for such a range."""
    import artistools.__main__

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        artistools.__main__.main(
            argsraw=[
                "plotspectra",
                str(modelpath_classic_3d),
                "-t",
                "5",
                "-stokesparam",
                "Q/I",
                "-ymin",
                "1",
                "-ymax",
                "0",
            ]
        )
    assert "must be less than --ymax" in capsys.readouterr().err


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_polarisation_plot_writes_the_drawn_ratio(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The plot of a Stokes ratio returned before output_data, thus --write_data gave no table."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d],
        timedays="5",
        stokesparam="Q/I",
        write_data=True,
        outputfile=tmp_path,
    )
    (tablepath,) = tmp_path.glob("plotpol_*.txt")
    dftable = read_data_file(tablepath)
    assert dftable.columns == ["x_plotted_angstrom", "Q/I"]
    _, drawnx, drawny = mockplot.call_args.args
    drawnx, drawny = np.asarray(drawnx, dtype=np.float64), np.asarray(drawny, dtype=np.float64)
    # the line can hold a point outside -xmin and -xmax, and the table takes the points inside them
    isinrange = (drawnx >= dftable["x_plotted_angstrom"].min()) & (drawnx <= dftable["x_plotted_angstrom"].max())
    assert set(drawnx[np.isfinite(drawny) & isinrange].round(6)) <= set(
        dftable["x_plotted_angstrom"].round(6).to_list()
    )

    # the table takes the wavelengths from -xmin to -xmax
    mockplot.reset_mock()
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d],
        timedays="5",
        stokesparam="Q/I",
        xmin=4000.0,
        xmax=6000.0,
        write_data=True,
        outputfile=tmp_path / "range",
    )
    (tablepath,) = (tmp_path / "range").glob("plotpol_*.txt")
    wavelengths = read_data_file(tablepath)["x_plotted_angstrom"]
    assert wavelengths.is_between(4000.0, 6000.0).all()
    assert wavelengths.len() > 0


def test_emission_plot_takes_a_list_of_one_time(tmp_path: Path) -> None:
    """A -timedayslist of one time names one time, thus an emission plot takes it as -timedays."""
    outputfile = tmp_path / "emission.pdf"
    at.spectra.plot(
        argsraw=[], specpath=[modelpath_classic_3d], timedayslist=["5"], showemission=True, outputfile=outputfile
    )
    assert outputfile.is_file()


def test_write_data_gives_the_plotted_values(tmp_path: Path) -> None:
    """--write_data must give the values of the plot, and not f_lambda at 1 Mpc alone.

    With --normalised the plot peaked at 1, and the file gave a peak of 2.6e-13.
    """
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath],
        timedays=300,
        normalised=True,
        write_data=True,
        outputfile=tmp_path / "normalised.pdf",
    )
    dfout = pl.read_csv(tmp_path / "normalised.txt", separator=" ", comment_prefix="#")
    plotted = [column for column in dfout.columns if column.startswith("flux_plotted")]
    assert len(plotted) == 1
    assert dfout[plotted[0]].max() == pytest.approx(1.0, rel=1e-6)


def test_xmin_alone_on_a_frequency_axis_keeps_the_given_value() -> None:
    """-xmin alone on a frequency axis must stay the lower limit.

    The default upper limit came from 19000 Å, which is a lower frequency than 2e14 Hz. The sort then made the
    given -xmin the upper limit.
    """
    args = at.misc.parse_cli_args(plotspectra.addargs, None, None, [str(modelpath), "-xunit", "hz", "-xmin", "2e14"])
    plotspectra.resolve_plot_args(args)
    assert args.xmin == pytest.approx(2e14)
    assert np.isclose(args.xmax, at.spectra.convert_angstroms_to_unit(2500.0, "hz"), rtol=1e-12, atol=0.0)


def test_single_x_limit_on_the_wrong_side_of_the_default_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """A single -xmin above the default upper limit became the upper limit, thus the plot showed a different range."""
    for argsraw, missing in (
        (["-xmin", "25000"], "--xmax"),
        (["-xmax", "2000"], "--xmin"),
        (["-xunit", "hz", "-xmax", "1e14"], "--xmin"),
        (["--gamma", "-xunit", "kev", "-xmin", "5000"], "--xmax"),
    ):
        args = at.misc.parse_cli_args(plotspectra.addargs, None, None, [str(modelpath), *argsraw])
        with pytest.raises(SystemExit):
            plotspectra.resolve_plot_args(args)
        assert f"Give {missing} as well" in capsys.readouterr().err

    # a pair of limits keeps its sort
    args = at.misc.parse_cli_args(plotspectra.addargs, None, None, [str(modelpath), "-xmin", "5000", "-xmax", "4000"])
    plotspectra.resolve_plot_args(args)
    assert (args.xmin, args.xmax) == (4000.0, 5000.0)


def make_model_with_a_new_spec_out(folder: Path, sourcemodel: Path, spectext: str) -> Path:
    """Return a copy of a model folder that links each file of the source, except spec.out, which holds spectext."""
    folder.mkdir()
    for sourcefile in sourcemodel.iterdir():
        if sourcefile.name not in {"spec.out", "packets"}:
            (folder / sourcefile.name).symlink_to(sourcefile)
    (folder / "spec.out").write_text(spectext, encoding="utf-8")
    return folder


def test_scan_spectra_names_the_last_timestep_of_a_run_that_stopped_early(tmp_path: Path) -> None:
    """ARTIS writes a column of spec.out for each timestep that ran, thus a later timestep needs a clear message.

    The column of a timestep came from its position, thus polars stopped with ColumnNotFoundError.
    """
    lines = (modelpath / "spec.out").read_text(encoding="utf-8").splitlines()
    # the frequency column and the timesteps 0 to 58
    cuttext = "".join(" ".join(line.split()[:60]) + "\n" for line in lines)
    earlymodel = make_model_with_a_new_spec_out(tmp_path / "early", modelpath, cuttext)

    dfspectrum = at.spectra.scan_spectra(earlymodel, 58, 58, directionbins=[-1])[-1].collect()
    assert dfspectrum.height == len(lines) - 1
    with pytest.raises(
        ValueError, match=r"spec\.out holds the timesteps 0 to 58, thus it has no spectrum of timestep 82"
    ):
        at.spectra.scan_spectra(earlymodel, 82, 82, directionbins=[-1])


def test_scan_spectra_reads_specpol_out_of_a_model_without_spec_out() -> None:
    """A model with POL_ON can hold specpol.out and no spec.out. The Stokes I part of specpol.out is the spectrum."""
    vspecpolmodel = at.get_path("testdata") / "vspecpolmodel"
    assert not (vspecpolmodel / "spec.out").exists()
    timestep = 120
    dfspectrum = at.spectra.scan_spectra(vspecpolmodel, timestep, timestep, directionbins=[-1])[-1].collect()

    stokes_i = at.spectra.scan_specpol_data(dirbin=-1, modelpath=vspecpolmodel)["I"].collect()
    expected = stokes_i.select("nu", pl.nth(timestep + 1).alias("f_nu")).sort("nu", descending=True)
    assert np.allclose(dfspectrum["nu"].to_numpy(), expected["nu"].to_numpy(), rtol=1e-12, atol=0.0)
    assert np.allclose(dfspectrum["f_nu"].to_numpy(), expected["f_nu"].to_numpy(), rtol=1e-12, atol=0.0)
    assert dfspectrum["f_nu"].sum() > 0.0


def test_vspecpol_spectrum_refuses_a_time_outside_the_time_bins() -> None:
    """The nearest time bin gave the spectrum of the first or the last bin, and the label gave the requested time."""
    vspecpolmodel = at.get_path("testdata") / "vspecpolmodel"
    # the first time bin of vspecpol_total-0.out has its middle at 1.06 days
    with pytest.raises(ValueError, match=r"thus no bin holds 0\.30 days"):
        atspectra.scan_vspecpol_spectrum(vspecpolmodel, 0.3, 0)
    with pytest.raises(ValueError, match="no bin holds"):
        atspectra.scan_vspecpol_spectrum(vspecpolmodel, 50.0, 0, timemin=40.0, timemax=60.0)

    dfspectrum = atspectra.scan_vspecpol_spectrum(vspecpolmodel, 11.5, 0).collect()
    assert dfspectrum["f_nu"].sum() > 0.0


def test_absorption_from_packets_skips_free_free_and_bound_free(
    monkeypatch: pytest.MonkeyPatch, patch_scan_packets: Callable[[pl.DataFrame], None]
) -> None:
    """ARTIS records no absorption frequency for a free-free or a bound-free absorption.

    The packet keeps the frequency of an earlier line absorption, thus such a series had the wavelength of an
    unrelated line. An absorption in an expansion opacity has a frequency but no line, thus it gets its own label.
    """
    c = at.constants.c_ang_per_s
    dfpackets = pl.DataFrame(
        {
            "e_rf": [1.0, 2.0, 4.0, 8.0],
            "t_arrive_d": [5.0] * 4,
            "nu_rf": [c / 9000.0] * 4,
            "emissiontype": [0] * 4,
            # a Fe II line, a free-free and a bound-free absorption after an earlier line absorption, and an
            # absorption in an expansion opacity
            "absorption_type": [0, -1, -2, -11],
            "absorption_freq": [c / 4050.0, c / 5050.0, c / 6050.0, c / 7050.0],
        },
        schema_overrides={"absorption_type": pl.Int32, "emissiontype": pl.Int32},
    )

    def get_linelist_label_columns(*_args: t.Any, **_kwargs: t.Any) -> pl.DataFrame:
        return pl.DataFrame({"atomic_number": [26], "ion_stage": [2]})

    patch_scan_packets(dfpackets)
    monkeypatch.setattr(atspectra, "get_linelist_label_columns", get_linelist_label_columns)
    contributions, _, array_lambda = atspectra.get_flux_contributions_from_packets(
        modelpath=Path(),
        timelowdays=4.0,
        timehighdays=6.0,
        lambda_bin_edges=np.arange(3000.0, 8000.0, 100.0),
        getemission=False,
        groupby="ion",
    )

    series = {
        row.linelabel: [float(value) for value in array_lambda[row.array_flambda_absorption > 0.0]]
        for row in contributions
    }
    assert series == {"Fe II": [4050.0], "bound-bound (expansion opacity)": [7050.0]}


def test_vpkt_exclusion_of_bound_bound_keeps_the_continuum_emission() -> None:
    """The exclusion of the bound-bound opacity (z = -1) removes only the bound-bound emission.

    The filter kept only the labels with bound-free, thus the free-free emission and the NOT SET emission went too.
    """
    c = at.constants.c_ang_per_s
    emissiontypes = [0, -9999999, -1, -9999000]
    packets = pl.LazyFrame(
        {
            "dir0_nu_rf": [c / 4050.0, c / 5050.0, c / 6050.0, c / 7050.0],
            "dir0_t_arrive_d": [5.0] * 4,
            "dir0_e_rf_0": [1.0] * 4,
            "dir0_e_rf_1": [1.0] * 4,
            "emissiontype": emissiontypes,
        },
        schema_overrides={"emissiontype": pl.Int32},
    )
    vpktconfig: dict[str, t.Any] = {
        "nobsdirections": 1,
        "nspectraperobs": 2,
        "z_excludelist": [0, -1],
        "time_limits_enabled": 0,
        "custom_lambda_ranges": [],
    }
    bflist = pl.LazyFrame({"bfindex": [0], "ion_str": ["Fe II"]}, schema={"bfindex": pl.Int32, "ion_str": pl.String})
    with (
        mock.patch.object(atspectra, "scan_virtual_packets", return_value=(1, packets)),
        mock.patch.object(atspectra, "get_vpkt_config", return_value=vpktconfig),
        mock.patch.object(
            atspectra,
            "get_linelist_label_columns",
            return_value=pl.DataFrame({"atomic_number": [26], "ion_stage": [2]}),
        ),
        mock.patch.object(atspectra, "scan_bflist", return_value=bflist),
    ):
        contributions, _, _ = atspectra.get_flux_contributions_from_packets(
            Path(),
            timelowdays=4.0,
            timehighdays=6.0,
            lambda_bin_edges=np.arange(3000.0, 8000.0, 100.0),
            getabsorption=False,
            groupby="ion",
            directionbin=1,
            directionbins_are_vpkt_observers=True,
            vpkt_match_emission_exclusion_to_opac=True,
        )

    assert {row.linelabel for row in contributions} == {"free-free", "Fe II bound-free", "NOT SET"}


def test_eflux_with_a_wavelength_unit_is_lambda_f_lambda() -> None:
    """-yvariable eflux on a wavelength axis gives lambda F_lambda in erg, whatever the unit of the wavelength.

    The code multiplied by the wavelength of a photon of one erg, which is no conversion of a unit.
    """
    dfspectrum = pl.DataFrame({"lambda_angstroms": [4000.0, 5000.0], "f_lambda": [2.0, 3.0]})
    expected = [8000.0, 15000.0]
    for xunit in ("angstrom", "nm", "micron"):
        dfxy = atspectra.get_dfspectrum_x_y_with_units(
            dfspectrum, xunit=xunit, yvariable="eflux", fluxdistance_mpc=1.0
        ).collect()
        assert np.allclose(dfxy["y"].to_numpy(), expected, rtol=1e-12, atol=0.0), xunit

    args = argparse.Namespace(
        xunit="nm", hidexticklabels=False, hideyticklabels=False, normalised=False, yvariable="eflux", distmpc=1.0
    )
    args.logscaley = False
    _, ylabel = plotspectra.get_axis_labels(args)
    assert ylabel is not None
    assert "lambda" in ylabel
    assert "erg/s/cm$^2$" in ylabel


def test_single_time_skips_a_model_that_does_not_hold_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A -timemin -timemax range outside a model skips that model, thus a single -timedays does the same.

    get_time_range raised ValueError for the model that did not hold the time, and the command stopped.
    """
    at.spectra.plot(
        argsraw=["-timedays", "300", str(modelpath), str(modelpath_classic_3d), "-outputfile", str(tmp_path)]
    )

    assert len(list(tmp_path.glob("*.pdf"))) == 1
    assert "no timestep covers 300 days" in capsys.readouterr().err


def test_multispecplot_keeps_the_colour_of_the_first_bin_on_each_panel(tmp_path: Path) -> None:
    """Each panel gives the first direction bin the -color, as the legend of the first panel shows.

    label_dirbin_series cleared the colour in the one dict of all the panels, thus the later panels lost it.
    """
    colours: dict[int, list[str]] = {}
    realplot = mplax.Axes.plot

    def spy(axis: mplax.Axes, *args: t.Any, **kwargs: t.Any) -> list[t.Any]:
        lines = realplot(axis, *args, **kwargs)
        colours.setdefault(id(axis), []).append(mplcolors.to_hex(lines[0].get_color()))
        return lines

    with mock.patch.object(mplax.Axes, "plot", spy):
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath_classic_3d],
            timedayslist=["5", "6"],
            plotviewingangle=[0, 5],
            color=["red"],
            outputfile=tmp_path / "colours.pdf",
        )

    panelcolours = list(colours.values())
    assert len(panelcolours) == 2
    assert panelcolours[0] == panelcolours[1]
    assert panelcolours[0][0] == "#ff0000"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_filter_functions_take_the_unit_of_the_x_axis(mockplot: mock.MagicMock) -> None:
    """The filter files give angstroms, thus a nanometre axis drew the curves ten times too far to the right."""
    fig = mplfig.Figure()
    axis = fig.add_subplot()
    plotspectra.plot_filter_functions(axis, "nm")

    xvalues = np.concatenate([np.asarray(call.args[1], dtype=np.float64) for call in mockplot.call_args_list])
    assert len(mockplot.call_args_list) == 4
    # the U, B, V, and I curves lie from about 300 nm to 950 nm
    assert xvalues.min() > 250.0
    assert xvalues.max() < 1000.0


def test_write_data_gives_a_table_for_each_wavelength_grid(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--write_data stopped for two models with different wavelength grids, and the figure can draw both.

    A wide table holds one grid, thus the second grid goes in its own table.
    """
    lines = (modelpath / "spec.out").read_text(encoding="utf-8").splitlines()
    shiftedlines = [lines[0]] + [
        " ".join([f"{float(line.split()[0]) * 1.02:g}", *line.split()[1:]]) for line in lines[1:]
    ]
    shiftedmodel = make_model_with_a_new_spec_out(tmp_path / "shifted", modelpath, "\n".join(shiftedlines) + "\n")

    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, shiftedmodel],
        timedays=300,
        write_data=True,
        outputfile=tmp_path / "twogrids.pdf",
    )
    dffirst = read_data_file(tmp_path / "twogrids.txt")
    dfsecond = read_data_file(tmp_path / "twogrids_grid2.txt")
    # the copy of the model has the same plotlabel.txt, thus the name of its columns adds the index of the model
    assert [col for col in dffirst.columns if col.startswith("f_lambda")] == ["f_lambda.TEST MODEL"]
    assert [col for col in dfsecond.columns if col.startswith("f_lambda")] == ["f_lambda.TEST MODEL_1"]
    assert not np.allclose(dffirst["lambda_angstroms"], dfsecond["lambda_angstroms"])

    # np.allclose took a grid that differs by 5 parts in a million as the same grid, and gave its values the
    # wavelengths of the first grid
    nearlines = [lines[0]] + [
        " ".join([repr(float(line.split()[0]) * 1.000005), *line.split()[1:]]) for line in lines[1:]
    ]
    nearmodel = make_model_with_a_new_spec_out(tmp_path / "near", modelpath, "\n".join(nearlines) + "\n")
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, nearmodel],
        timedays=300,
        write_data=True,
        outputfile=tmp_path / "neargrids.pdf",
    )
    assert (tmp_path / "neargrids_grid2.txt").is_file()


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_write_data_gives_the_edges_of_a_histogram(
    mockplot: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--histogram draws each bin from its lower edge to its upper edge, and the table gave the centre alone."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath],
        timedays="300",
        histogram=True,
        write_data=True,
        outputfile=tmp_path / "histogram.pdf",
    )
    dftable = read_data_file(tmp_path / "histogram.txt")
    drawnedges = np.asarray(mockplot.call_args_list[0].args[1], dtype=np.float64).reshape(-1, 2)
    edges = np.column_stack([dftable["x_lower_angstrom"].to_numpy(), dftable["x_upper_angstrom"].to_numpy()])
    # the plot can draw a bin outside -xmin and -xmax, and the table takes the bins inside them
    firstbin = int(np.argmin(np.abs(drawnedges[:, 0] - edges[0, 0])))
    assert np.allclose(drawnedges[firstbin : firstbin + len(edges)], edges, rtol=1e-9, atol=0)


@mock.patch.object(mplax.Axes, "fill_between", side_effect=mplax.Axes.fill_between, autospec=True)
def test_write_data_gives_the_noise_of_shownoise(
    mockfill: mock.MagicMock, tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """--shownoise draws the band y * (1 +- r), and the table gave y alone, thus it could not give the band."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d],
        timemin=4,
        timemax=6.5,
        frompackets=True,
        shownoise=True,
        write_data=True,
        outputfile=tmp_path / "noise.pdf",
    )
    dftable = read_data_file(tmp_path / "noise.txt")
    modelname = at.get_model_name(modelpath_classic_3d)
    y = dftable[f"flux_plotted.{modelname}"].to_numpy()
    relnoise = np.nan_to_num(dftable[f"flux_relnoise.{modelname}"].to_numpy())
    # the band can hold a point outside -xmin and -xmax, and the table takes the points inside them
    drawnx = np.asarray(mockfill.call_args.args[1], dtype=np.float64)
    isinrange = (drawnx >= dftable["x_plotted_angstrom"].min()) & (drawnx <= dftable["x_plotted_angstrom"].max())
    upper = np.asarray(mockfill.call_args.args[3], dtype=np.float64)[isinrange]
    assert np.allclose(upper, y * (1 + relnoise), rtol=1e-6, atol=0)


@pytest.mark.parametrize("zline", ["", "z: 0.1\n"])
def test_telluric_mask_of_a_reference_spectrum_uses_observed_wavelengths(tmp_path: Path, zline: str) -> None:
    """The telluric mask applies to the observed wavelengths, thus it removes the row at 14000 Å for each z.

    The mask moved before the de-redshift, and the result must stay the same. With z = 0.1, the row at 15000 Å moves
    to 13636 Å, which is inside the observed band from 13500 Å to 14400 Å. Thus the mask must not use the rest-frame
    wavelengths. With no z, the mask raised KeyError.
    """
    specfile = tmp_path / "telluric.txt"
    specfile.write_text("13000 1.0\n14000 2.0\n15000 3.0\n", encoding="utf-8")
    specfile.with_suffix(".txt.meta.yml").write_text(
        f"---\nmask_telluric: true\ndist_mpc: 1.0\n{zline}", encoding="utf-8"
    )

    fig = mplfig.Figure()
    axis = fig.add_subplot()
    line, _, _ = plotspectra.plot_reference_spectrum(specfile, axis, xmin=11000.0, xmax=16000.0)

    yvalues = np.asarray(line.get_ydata(), dtype=np.float64)
    assert np.isnan(yvalues).tolist() == [False, True, False]


@pytest.mark.parametrize("zvalue", ["-1", "-2", ".inf", ".nan", "true", "abc"])
def test_reference_spectrum_refuses_an_incorrect_redshift(tmp_path: Path, zvalue: str) -> None:
    """A z of -1 divided by zero, and a z that YAML reads as a string gave a TypeError with no file name."""
    specfile = tmp_path / "spec.txt"
    specfile.write_text("5000 1.0\n6000 2.0\n", encoding="utf-8")
    specfile.with_suffix(".txt.meta.yml").write_text(f"---\nz: {zvalue}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"spec\.txt\.meta\.yml must be a number above -1"):
        atspectra.get_reference_spectrum(specfile, at.misc.get_file_metadata(specfile))


def test_reference_spectrum_takes_a_redshift_with_an_exponent(tmp_path: Path) -> None:
    """YAML 1.1 read 1e-3 as a string, because the number has no dot. 1 + z then raised a TypeError."""
    specfile = tmp_path / "spec.txt"
    specfile.write_text("5000 1.0\n6000 2.0\n", encoding="utf-8")
    specfile.with_suffix(".txt.meta.yml").write_text("---\nz: 1e-3\n", encoding="utf-8")
    specdata = atspectra.get_reference_spectrum(specfile, at.misc.get_file_metadata(specfile))
    assert np.allclose(specdata["lambda_angstroms"].to_numpy(), [5000.0 / 1.001, 6000.0 / 1.001], rtol=1e-12, atol=0.0)
    assert np.allclose(specdata["f_lambda"].to_numpy(), [1.001, 2.002], rtol=1e-12, atol=0.0)


def test_stokesparam_takes_only_the_values_that_the_plot_reads() -> None:
    """A value such as q/i stopped with KeyError in plot_polarisation, thus argparse refuses it."""
    parser = argparse.ArgumentParser()
    plotspectra.addargs(parser)
    assert parser.parse_args(["-stokesparam", "Q/I"]).stokesparam == "Q/I"
    with pytest.raises(SystemExit):
        parser.parse_args(["-stokesparam", "q/i"])


def test_codecomparison_spectrum_refuses_the_options_that_it_cannot_draw(capsys: pytest.CaptureFixture[str]) -> None:
    """A code comparison spectrum is F_lambda per angstrom at 1 Mpc, and it needs a time.

    The reader took the raw -timedays text, which failed for a range, and it drew angstroms on an axis of another unit.
    """
    args = argparse.Namespace(timemin=290.0, timemax=320.0, xunit="nm", yvariable="flux", distmpc=1.0, normalised=False)
    with pytest.raises(SystemExit):
        plotspectra.check_codecomparison_args(args)
    assert "-xunit nm" in capsys.readouterr().err

    args.timemin = None
    with pytest.raises(SystemExit):
        plotspectra.check_codecomparison_args(args)
    assert "no time range" in capsys.readouterr().err

    args = argparse.Namespace(
        timemin=290.0, timemax=320.0, xunit="angstrom", yvariable="flux", distmpc=1.0, normalised=False
    )
    plotspectra.check_codecomparison_args(args)


def test_fixedionlist_warning_names_only_the_items_that_the_plot_shows(capsys: pytest.CaptureFixture[str]) -> None:
    """An item after -maxseriescount gives no warning, because the plot does not show it.

    The packet reader keeps a part of the list only, and the warning then named items that the packets hold.
    """
    arraylambda = np.array([4000.0, 5000.0])
    contributions = [
        atspectra.FluxContributionTuple(fluxcontrib, label, np.ones(2), np.zeros(2))
        for fluxcontrib, label in ((2.0, "Fe II"), (1.0, "Co II"))
    ]
    for maxseriescount, haswarning in ((2, False), (3, True)):
        plotspectra.sort_and_reduce_flux_contribution_list(
            contributions, maxseriescount, arraylambda, fixedionlist=["Fe II", "Co II", "Ni II"]
        )
        assert ("Did not find" in capsys.readouterr().err) is haswarning


def test_absorption_plot_keeps_a_linear_axis() -> None:
    """An absorption plot must keep a linear y axis with -yscale auto, because its absorption part is negative.

    The automatic rule read only the spectrum line, and it chose a log axis. The axis then showed no data.
    """
    args = at.misc.parse_cli_args(
        plotspectra.addargs,
        None,
        None,
        [str(modelpath_classic_3d), "-t", "5", "--showemission", "--showabsorption", "-yscale", "auto"],
    )
    plotspectra.resolve_plot_args(args)
    # the automatic rule then asks for a log axis for every set of values
    with mock.patch("artistools.plottools.wants_log_scale", return_value=True):
        fig, axes, residualaxis = plotspectra.make_plot_figure(args)
        plotspectra.draw_plot(args, axes, residualaxis)
    assert axes[0].get_yscale() == "linear"
    assert axes[0].get_ylim()[0] < 0.0
    plt.close(fig)


@pytest.mark.parametrize("fixedionlist", [None, ["ion 30", "ion 2", "ion 99", "ion 39"]])
def test_host_merge_of_flux_contributions_keeps_the_plot(
    fixedionlist: list[str] | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """The host merges the small series of emission.out into "Other", and the plot and its printed list must not change.

    print_contribution_table names the 3 largest other series, thus the host keeps them.
    """
    rng = np.random.default_rng(1)
    arraylambda = np.linspace(3000.0, 9000.0, 5)
    arraynu = at.constants.c_ang_per_s / arraylambda
    # only a bound-bound series has an absorbed f_nu
    series: list[atspectra.FluxSeries] = [
        (f"ion {index}", rng.random(5) * flux, rng.random(5) * flux if index % 3 else None)
        for index, flux in enumerate(rng.random(40))
    ]

    merged = atspectra.merge_other_flux_series(series, arraynu, 4, fixedionlist)
    # the host sends the kept series, the 3 named other series, and "Other"
    assert len(merged) <= 4 + 3 + 1

    reduced_of_source = []
    totals = []
    for source in (series, merged):
        contributions, total = atspectra.get_flux_contribution_tuples(source, arraynu, arraylambda)
        totals.append(total)
        reduced = plotspectra.sort_and_reduce_flux_contribution_list(contributions, 4, arraylambda, fixedionlist)
        reduced_of_source.append((reduced, capsys.readouterr().out))

    assert np.allclose(totals[1], totals[0], rtol=1e-12, atol=0.0)

    (reduced_full, printed_full), (reduced_merged, printed_merged) = reduced_of_source
    assert printed_merged == printed_full
    assert [row.linelabel for row in reduced_merged] == [row.linelabel for row in reduced_full]
    for row_merged, row_full in zip(reduced_merged, reduced_full, strict=True):
        assert np.isclose(row_merged.fluxcontrib, row_full.fluxcontrib, rtol=1e-12, atol=0.0)
        assert np.allclose(row_merged.array_flambda_emission, row_full.array_flambda_emission, rtol=1e-12, atol=0.0)
        assert np.allclose(row_merged.array_flambda_absorption, row_full.array_flambda_absorption, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("averaging", [[], ["--average_over_phi_angle"], ["--average_over_theta_angle"]])
@mock.patch.object(mplax.Axes, "set_title", side_effect=mplax.Axes.set_title, autospec=True)
def test_emission_plot_of_all_directions(mocktitle: mock.MagicMock, averaging: list[str], tmp_path: Path) -> None:
    """An emission plot takes bin -1, the average over all the directions, also with an average over one angle.

    The window of plotspectra gives bin -1 in the list of the direction bins. The title then had no label for the bin
    (KeyError), and an average over phi read bin -1 as the first bin of a group of phi bins (AssertionError).
    """
    at.spectra.plot(
        argsraw=[
            str(modelpath_classic_3d),
            "-t",
            "5",
            "--showemission",
            *averaging,
            "-plotviewingangle",
            "-1",
            "-o",
            str(tmp_path / "emission.pdf"),
        ]
    )
    assert any("all directions" in str(callargs.args[1]) for callargs in mocktitle.call_args_list)


def test_default_rpkt_bins_end_at_the_kilonova_nu_max_r_of_artis(tmp_path: Path) -> None:
    """Without spec.out, the packet bins go from 1e13 Hz to NU_MAX_R = 5e16 Hz.

    The kilonova artisoptions.h files of ARTIS give 5e16 Hz, and the others give 5e15 Hz. The default keeps
    the kilonova value.
    """
    lambda_bin_edges = atspectra.get_exspec_lambda_bin_edges(tmp_path)
    assert len(lambda_bin_edges) == 1001
    assert np.isclose(lambda_bin_edges[0], at.constants.c_ang_per_s / 5e16, rtol=1e-9, atol=0.0)
    assert np.isclose(lambda_bin_edges[-1], at.constants.c_ang_per_s / 1e13, rtol=1e-9, atol=0.0)


def test_plotspectra_draws_a_reference_spectrum_with_no_metadata(tmp_path: Path) -> None:
    """A file of two columns with no metadata gives no distance, thus its flux stays as the file gives it.

    The command read the distance from the metadata of each file, and it stopped with a KeyError for such a file.
    """
    reffile = tmp_path / "plainspectrum.txt"
    reffile.write_text("4000 1.0\n5000 2.0\n6000 3.0\n7000 2.0\n", encoding="utf-8")
    (axis,) = get_saved_axes(
        specpath=[modelpath, reffile], outputfile=tmp_path / "plain.pdf", timemin=290, timemax=320, distmpc=10.0
    )
    (refline,) = [line for line in axis.get_lines() if line.get_label() == str(reffile)]
    assert np.allclose(np.asarray(refline.get_ydata(), dtype=np.float64), [1.0, 2.0, 3.0, 2.0], rtol=1e-12, atol=0.0)


def test_flux_contributions_refuse_a_timestep_outside_the_model() -> None:
    """A timestep below zero is not in the model.

    polars takes a negative row index from the end, thus the default timestep of -1 gave the rows of other bins.
    """
    with pytest.raises(ValueError, match="outside the"):
        at.spectra.get_flux_contributions(modelpath, timestepmin=-1, timestepmax=-1)


def test_flux_contributions_cache_serves_each_filter() -> None:
    """The cache keeps the contributions without the filter, thus a new filter function reads no file again.

    get_filterfunc makes a new function for each plot, and a function in the key of the cache never gave a hit.
    """
    from artistools.misc import moving_average_filter

    at.spectra.core.get_flux_contributions_cached.cache_clear()
    unfiltered = at.spectra.get_flux_contributions(
        modelpath, timestepmin=40, timestepmax=42, use_lastemissiontype=False
    )
    filtered = [
        at.spectra.get_flux_contributions(
            modelpath,
            timestepmin=40,
            timestepmax=42,
            filterfunc=functools.partial(moving_average_filter, n=5),
            use_lastemissiontype=False,
        )
        for _ in range(2)
    ]
    cacheinfo = at.spectra.core.get_flux_contributions_cached.cache_info()
    assert (cacheinfo.misses, cacheinfo.hits) == (1, 2)
    assert np.allclose(filtered[0][1], filtered[1][1], rtol=1e-12, atol=0.0)
    # the filter is linear and it applies to f_nu, thus the filtered total is the filtered f_nu of the total
    arraylambda = unfiltered[2]
    flambda_per_fnu = at.constants.c_ang_per_s / arraylambda**2
    expectedtotal = moving_average_filter(unfiltered[1] / flambda_per_fnu, n=5) * flambda_per_fnu
    assert np.allclose(filtered[0][1], expectedtotal, rtol=1e-9, atol=1e-30)
    assert not np.allclose(filtered[0][1], unfiltered[1], rtol=1e-6, atol=0.0)


def test_plotspectra_showtime_of_multispecplot_stays_inside_each_panel(tmp_path: Path) -> None:
    """The epoch of each panel takes a position in the axes, thus it is visible for each x unit.

    The text took the position x = 5500 in the data, which is outside the axes for -xunit nm.
    """
    axes = get_saved_axes(
        specpath=modelpath, outputfile=tmp_path / "multi.pdf", timedayslist=["280", "300"], showtime=True, xunit="nm"
    )
    assert len(axes) == 2
    for axis, timedays in zip(axes, ["280", "300"], strict=True):
        (timetext,) = [text for text in axis.texts if text.get_text() == f"{timedays} days"]
        assert getattr(timetext, "xycoords", None) == "axes fraction"
        assert timetext.get_position() == (0.03, 0.97)


def test_deltax_refuses_a_bin_edge_at_or_below_zero() -> None:
    """A bin edge at or below zero gives a negative wavelength, thus one bin would hold all the packets below it."""
    with pytest.raises(ValueError, match="above half a bin width"):
        atspectra.get_lambda_bin_edges(0.0, 1e15, 1e14, None, None, "hz", modelpath)


def test_shell_groupby_with_no_model_gives_the_error_of_the_emission_plot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """-groupby radialvelocity with only a reference spectrum stops with the message of the emission plot.

    The default shells came from the first model before the check, thus the command raised IndexError. The test gives
    the earlier name velocity, which a script can hold.
    """
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=["2003du_20031213_3219_8822_00.txt", "-groupby", "velocity", "-outputfile", str(tmp_path)]
        )
    assert "no path names such a model" in capsys.readouterr().err


def test_emission_plot_file_name_holds_the_vspecpol_observer(tmp_path: Path) -> None:
    """The emission plots of two observers of the virtual packets get two file names.

    Only -plotviewingangle gave a part of the name, thus the plot of a second observer replaced the first.
    """
    savedfiles: list[str] = []

    def save_figure(_fig: mplfig.Figure, filename: str, *_args: t.Any, **_kwargs: t.Any) -> None:
        savedfiles.append(Path(filename).name)

    with (
        mock.patch.object(at.spectra.plotspectra, "save_figure", save_figure),
        mock.patch.object(at.spectra.plotspectra, "draw_plot", return_value={}),
    ):
        for observer in (0, 1):
            at.spectra.plot(
                argsraw=[],
                specpath=[at.get_path("testdata") / "vpktcontrib"],
                outputfile=tmp_path,
                plotvspecpol=[observer],
                frompackets=True,
                showemission=True,
                maxpacketfiles=2,
                timemin=130,
                timemax=135,
            )

    assert savedfiles == [
        "plotspectra_emission_130.00d-135.00d_vspecpol00.pdf",
        "plotspectra_emission_130.00d-135.00d_vspecpol01.pdf",
    ]


def test_plotspectra_ymax_alone_keeps_the_bottom_at_zero(tmp_path: Path) -> None:
    """A flux of Stokes I is not negative, thus -ymax alone keeps the bottom of the y range at zero.

    The y margin put the bottom below zero, and only a command with no -ymin and no -ymax moved it back.
    """
    (axis,) = get_saved_axes(
        specpath=modelpath, outputfile=tmp_path / "ymax.pdf", timemin=290, timemax=320, ymax=2e-12, yscale="linear"
    )
    assert axis.get_ylim() == (0.0, 2e-12)


def test_spectrum_table_keeps_two_models_of_the_same_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """Two models of the same name gave the same columns, and the second model replaced the first in the table."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, modelpath],
        timedays="300",
        outputfile=tmp_path / "samename.pdf",
        write_data=True,
    )
    assert "gives the columns of" in capsys.readouterr().err
    dftable = read_data_file(tmp_path / "samename.txt")
    fluxcolumns = [colname for colname in dftable.columns if colname.startswith("f_lambda.")]
    assert len(fluxcolumns) == 2, dftable.columns


def test_spectrum_summary_integrates_around_a_masked_band(capsys: pytest.CaptureFixture[str]) -> None:
    """A masked telluric band holds NaN, and the luminosity of the summary was NaN for a range across the band."""
    from artistools.spectra.plotspectra import print_spectrum_summary

    lambdas = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
    flambdas = [1.0, 1.0, float("nan"), 1.0, 1.0]
    dfspectrum = pl.DataFrame({"x": lambdas, "lambda_angstroms": lambdas, "f_lambda": flambdas, "y": flambdas})
    print_spectrum_summary("masked", dfspectrum, (1000.0, 5000.0), "angstrom", "flux")
    out = capsys.readouterr().out
    # the two drawn segments each have a width of 1000 Å, and the two segments next to the NaN are left out
    luminosity = 2000.0 * 4 * math.pi * at.constants.megaparsec_to_cm**2
    match = re.search(r"luminosity ([\d.e+-]+) erg/s", out)
    assert match is not None, out
    assert np.isclose(float(match.group(1)), luminosity, rtol=1e-3)
    assert "leaves out 2000 Å with no value of f_λ" in out
    assert "largest drawn flux 1 at" in out

    # a masked point at the end of the range is not drawn, and the summary counted it in the points and the x range
    dfendmasked = dfspectrum.with_columns(
        f_lambda=pl.Series([1.0, 1.0, 1.0, 1.0, float("nan")]), y=pl.Series([1.0, 1.0, 1.0, 1.0, float("nan")])
    )
    print_spectrum_summary("endmasked", dfendmasked, (1000.0, 5000.0), "angstrom", "flux")
    assert "endmasked: 4 points from 1000 to 4000 angstrom" in capsys.readouterr().out

    # a filter combines the packets of neighbouring bins, and the summary gave the noise of the unfiltered bins
    dfpackets = dfspectrum.with_columns(
        packetcount=pl.Series([10, 10, 10, 10, 10]), relnoise=0.3, delta_lambda=1000.0, f_lambda=1.0, y=1.0
    )
    print_spectrum_summary("filtered", dfpackets, (1000.0, 5000.0), "angstrom", "flux", isfiltered=True)
    out = capsys.readouterr().out
    assert "no noise estimate, because the filter combines the packets" in out
    assert "relative noise of the luminosity" not in out


def test_summary_table_agrees_with_the_summary_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The table _summary of --write_data gives the numbers of each summary line, thus an agent need not parse the line.

    The luminosity must also agree with 4π (1 Mpc)² times the trapezoid integral of f_λ in the table of the spectrum. A
    reference spectrum has no packets, thus its row has no noise.
    """
    plotspectra.main(
        argsraw=[],
        specpath=[modelpath, "sn2011fe_PTF11kly_20120822_norm.txt"],
        timedays="300",
        frompackets=True,
        outputfile=tmp_path / "spec.pdf",
        write_data=True,
    )
    output = capsys.readouterr().out
    dfsummary = read_data_file(tmp_path / "spec_summary.txt")
    assert dfsummary.height == 2
    for row in dfsummary.iter_rows(named=True):
        match = re.search(
            rf"{re.escape(row['series'])}: (\d+) points from (\S+) to (\S+) angstrom, largest drawn flux (\S+) at (\S+)"
            r" angstrom, luminosity (\S+) erg/s = 4π \(1 Mpc\)² ∫ f_λ dλ from (\S+) to (\S+) Å \((.+?)\)",
            output,
        )
        assert match is not None, output
        assert int(match[1]) == row["point_count"]
        assert np.allclose(
            [float(value) for value in match.group(2, 3, 4, 5, 6, 7, 8)],
            [
                row["x_min_angstrom"],
                row["x_max_angstrom"],
                row["peak_flux"],
                row["x_at_peak_angstrom"],
                row["luminosity_erg_per_s"],
                row["lambda_min_angstroms"],
                row["lambda_max_angstroms"],
            ],
            rtol=1e-3,
            atol=0.0,
        )
        assert match[9] == row["method"]

    modelrow, referencerow = dfsummary.iter_rows(named=True)
    noisematch = re.search(
        r"(\d+) packets in the drawn bins, relative noise of the luminosity (\S+)% = (.+) of these packets", output
    )
    assert noisematch is not None
    assert int(noisematch[1]) == modelrow["packet_count"]
    assert np.isclose(float(noisematch[2]) / 100.0, modelrow["relnoise_luminosity"], rtol=1e-3)
    assert noisematch[3] == modelrow["noise_method"]
    assert referencerow["relnoise_luminosity"] is None

    dfspectrum = read_data_file(tmp_path / "spec.txt").filter(
        pl.col("lambda_angstroms").is_between(modelrow["lambda_min_angstroms"], modelrow["lambda_max_angstroms"])
    )
    integral = np.trapezoid(dfspectrum["f_lambda.TEST MODEL"], dfspectrum["lambda_angstroms"])
    # the summary also takes the part of the end bins up to the axis limits, which this integral leaves out
    assert np.isclose(
        4 * math.pi * at.constants.megaparsec_to_cm**2 * integral, modelrow["luminosity_erg_per_s"], rtol=0.02
    )


def test_packet_spectrum_table_gives_the_noise_without_shownoise(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Give the relative Monte Carlo noise of each bin in the table of a spectrum from packets.

    The column came only with --shownoise, thus a reader of the table could not see that a feature was noise.
    """
    # the command saves its plot in the working folder
    monkeypatch.chdir(tmp_path)
    at.spectra.plotspectra.main(argsraw=[str(modelpath), "--frompackets", "-t", "300", "--print_data"])
    header = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("lambda_angstroms "))
    assert "flux_relnoise" in header


def test_packet_reason_gives_the_standard_spelling(capsys: pytest.CaptureFixture[str]) -> None:
    """The message that turns on --frompackets names the option as the help gives it, e.g. --use-sampledemissiontype.

    The message named --use_sampledemissiontype, which the help hides as an older spelling.
    """
    from artistools.spectra.plotspectra import resolve_frompackets

    args = at.misc.parse_cli_args(
        at.spectra.plotspectra.addargs, None, None, [str(modelpath), "--showemission", "--use-sampledemissiontype"]
    )
    # resolve_plot_args sets the velocity ranges before it resolves the packets, and the command gives none
    args.velocityranges_kmps = []
    resolve_frompackets(args)
    assert args.frompacketsreason == "--use-sampledemissiontype"
    assert "since --use-sampledemissiontype was specified" in capsys.readouterr().out


def test_plotspectra_x_still_gives_the_unit() -> None:
    """-x named the unit on plotspectra before -xunit, thus a script that holds it still runs.

    -x names the axis variable on plotestimators, but each parser reads its own arguments.
    """
    import artistools.spectra.plotspectra

    parser = argparse.ArgumentParser(prog="plotspectra")
    artistools.spectra.plotspectra.addargs(parser)

    # each spelling gives the canonical name of the unit, thus the plot code reads one name
    assert parser.parse_args([".", "-x", "nm"]).xunit == "nm"
    assert parser.parse_args([".", "-xunits", "micron"]).xunit == "micron"
    assert parser.parse_args([".", "-xunit", "Hz"]).xunit == "hz"


def test_emission_plot_names_a_direction_bin_that_the_average_leaves_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command refuses a bin that an average over phi leaves out, before it reads the data.

    The emission plot checked the bin with a bare assert, thus the command gave the message of a fault of artistools.
    With --frompackets, the command read each packet before the error.
    """
    packetargs: list[str]
    for packetargs in ([], ["--frompackets"]):
        with pytest.raises(ValueError, match=r"Direction bin 5 is not the first bin of an average group"):
            at.spectra.plot(
                argsraw=[
                    str(modelpath_classic_3d),
                    "--showemission",
                    *packetargs,
                    "-plotviewingangle",
                    "5",
                    "--average-over-phi-angle",
                    "-ts",
                    "10",
                    "-o",
                    str(tmp_path / "emission.pdf"),
                ]
            )
        assert "packets" not in capsys.readouterr().out


def test_output_template_stops_plotspectra_before_the_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A field of -o that plotspectra does not give stops the command before it reads the spectra.

    The command checked the template after the read and the draw. For the packets of a large model, these take minutes.
    """
    with pytest.raises(ValueError, match=r"holds the field 'timestep'"):
        at.spectra.plot(argsraw=[str(modelpath), "-t", "300", "-o", str(tmp_path / "spec_{timestep}.pdf")])
    # the draw logs the model path of each series, also when a cache holds the spectra
    assert "modelpath:" not in capsys.readouterr().out


def test_gamma_spectrum_check_follows_the_working_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check for gamma_spec.out keys its cache on the absolute path of the model folder, as each reader does.

    The check had a plain lru_cache, thus the relative path "." kept the answer of the first working folder.
    """
    for foldername in ("withgamma", "nogamma"):
        (tmp_path / foldername).mkdir()
    (tmp_path / "withgamma" / "gamma_spec.out").write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path / "withgamma")
    assert atspectra.has_gamma_spec_file(Path())
    monkeypatch.chdir(tmp_path / "nogamma")
    assert not atspectra.has_gamma_spec_file(Path())


def test_output_template_names_a_field_that_plotspectra_does_not_give(tmp_path: Path) -> None:
    """The command plotspectra refuses a field of -o that it does not give, as the other commands do.

    get_plot_outputfile filled the template with str.format, thus the command stopped with the traceback of a KeyError.
    """
    args = argparse.Namespace(
        outputfile=tmp_path / "spec_{foo}.pdf",
        showemission=False,
        showabsorption=False,
        plotviewingangle=None,
        plotvspecpol=None,
        timemin=300.0,
        timemax=301.0,
    )
    with pytest.raises(ValueError, match=r"holds the field 'foo'"):
        plotspectra.get_plot_outputfile(args)
