import argparse
import dataclasses as dc
import math
import shlex
import sys
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at
from artistools.spectra import core as atspectra
from artistools.spectra import interactive
from artistools.spectra import plotspectra

modelpath = at.get_path("testdata") / "testmodel"
outputpath = at.get_path("testoutput")
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


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


def test_spectra_outputtext(tmp_path: Path) -> None:
    """-o names the folder of the spectrum files, and the command makes a folder that does not exist yet."""
    newfolder = tmp_path / "newfolder"
    at.spectra.plot(argsraw=[], specpath=modelpath, output_spectra=True, outputfile=newfolder)

    assert list(newfolder.glob("spectrum_ts*.txt"))


@pytest.mark.benchmark
def test_spectraemissionplot(tmp_path: Path) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=tmp_path / "emission.pdf",
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        use_thermalemissiontype=True,
    )


@pytest.mark.benchmark
def test_spectraemissionplot_nostack(tmp_path: Path) -> None:
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


def test_spectra_get_spectrum() -> None:
    def check_spectrum(dfspectrumpkts: pl.DataFrame, expectedmax: float, expectedmean: float) -> None:
        assert math.isclose(max(dfspectrumpkts["f_lambda"]), expectedmax, rel_tol=1e-4)
        assert min(dfspectrumpkts["f_lambda"]) < 1e-9
        flambdamean = dfspectrumpkts["f_lambda"].mean()
        assert isinstance(flambdamean, float)
        assert math.isclose(flambdamean, expectedmean, rel_tol=1e-4)

    dfspectrum = at.spectra.get_spectra(modelpath, 55, 65, fluxfilterfunc=None)[-1].collect()

    assert len(dfspectrum["lambda_angstroms"]) == 1000
    assert len(dfspectrum["f_lambda"]) == 1000
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[-1] - 29920.601421214415) < 1e-5
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[0] - 600.75759482509852) < 1e-5

    check_spectrum(dfspectrum, expectedmax=2.548532804918824e-13, expectedmean=1.0314682640070206e-14)

    timelowdays = at.get_timestep_times(modelpath)[55]
    timehighdays = at.get_timestep_times(modelpath)[65]

    dfspectrumpkts = at.spectra.get_from_packets(modelpath, timelowdays=timelowdays, timehighdays=timehighdays)[
        -1
    ].collect()

    # the packets file of the test model holds a part of the packets of the run, thus its flux is
    # below the flux of spec.out. test_spectra_get_flux_contributions compares spec.out with the sum
    # of the emission contributions
    check_spectrum(dfspectrumpkts, expectedmax=1.4601241778685615e-14, expectedmean=3.8221552332231758e-16)


@pytest.mark.benchmark
def test_spectra_get_spectrum_polar_angles() -> None:
    spectra = at.spectra.get_spectra(
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
        lambda: at.spectra.get_from_packets(
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
                ("Co II λ3754 69-25", 4.2743896083241423e-10),
                ("Co II λ4160 69-31", 3.974052796767683e-10),
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
    names the wavelength of the line, e.g. "Co II λ3754 69-25". No label names a free-free process or
    a bound-free process, because a packet absorbs in a line alone.
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
        groupby="velocity", shelledges=shells
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
    ("rangegrouping", "rangeedges"), [("velocity", (10000.0, 20000.0)), ("losvelocity", (-10000.0, 5000.0))]
)
def test_spectra_velocity_range_contributions(rangegrouping: str, rangeedges: tuple[float, float]) -> None:
    """The ion groups of a velocity range hold the emission and the absorption of the shell with the same edges."""
    shells = [-51000.0 if rangegrouping == "losvelocity" else 0.0, *rangeedges, 51000.0]
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
        groupby="losvelocity", shelledges=list(losedges), velocityranges={"velocity": radialedges}
    )
    (shell,) = contributions_shells

    _, array_flambda_emission_total, _ = get_contributions_classic_3d(
        groupby="ion", velocityranges={"velocity": radialedges, "losvelocity": losedges}
    )
    _, array_flambda_emission_total_radial, _ = get_contributions_classic_3d(
        groupby="ion", velocityranges={"velocity": radialedges}
    )

    assert np.allclose(array_flambda_emission_total, shell.array_flambda_emission, rtol=1e-6, atol=0.0)
    assert 0.0 < array_flambda_emission_total.sum() < array_flambda_emission_total_radial.sum()


def test_spectra_no_thermal_emission_record_gives_no_thermal_velocity() -> None:
    """A packet with no thermal emission record is in the NOT SET series, and no velocity range from zero holds it.

    ARTIS gives such a packet a thermal emission velocity of zero. The thermal emission of this model
    lies above 11 000 km/s. Thus a packet below 10 000 km/s has no thermal emission record.
    """
    contributions, _, _ = get_contributions_classic_3d(
        groupby="velocity", usethermal=True, shelledges=[0.0, 10000.0, 51000.0], getabsorption=False
    )
    fluxes = {contrib.linelabel: contrib.fluxcontrib for contrib in contributions}
    assert "[0, 10000) km/s" not in fluxes
    assert fluxes["NOT SET"] > 0.0
    assert fluxes["[10000, 51000) km/s"] > 0.0

    contributions_range, _, _ = get_contributions_classic_3d(
        usethermal=True, velocityranges={"velocity": (0.0, 10000.0)}, getabsorption=False
    )
    assert not contributions_range


def test_spectra_velocity_argument_takes_kmps_or_c() -> None:
    """A shell edge is a number in km/s, or a fraction of c with a c suffix, and the labels keep that unit."""
    assert atspectra.parse_velocity_argument("5000") == (5000.0, "kmps")
    velocity_kmps, unit = atspectra.parse_velocity_argument("0.1C")
    assert unit == "c"
    assert np.isclose(velocity_kmps, 29979.2458)

    for badvalue in ("fast", "inf", "infc", "nan"):
        with pytest.raises(argparse.ArgumentTypeError, match="not a finite velocity"):
            atspectra.parse_velocity_argument(badvalue)

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


def test_spectra_velocity_shell_expr_labels_a_packet_with_no_thermal_emission() -> None:
    """A packet with a NaN velocity takes the label NOT SET, and a packet outside every shell takes null."""
    dfpackets = pl.DataFrame({"v": [5.0e8, float("nan"), 5.0e10, 1.5e9, None]})
    labels = dfpackets.select(atspectra.get_shell_expr("v", [0.0, 10000.0, 20000.0])).to_series().to_list()
    assert labels == ["[0, 10000) km/s", "NOT SET", None, "[10000, 20000) km/s", "NOT SET"]


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
    edges, unit = atspectra.get_default_velocity_shells(modelpath_classic_3d)
    assert unit == "kmps"
    assert len(edges) == 12
    assert np.isclose(edges[10], 28920.2, rtol=1e-4)

    fastmeta = {"vmax_cmps": 0.3 * 2.99792458e10, "dimensions": 1}
    with mock.patch("artistools.inputmodel.get_modeldata", return_value=(pl.LazyFrame(), fastmeta)):
        edges, unit = atspectra.get_default_velocity_shells("fastmodel", nshells=3)
    assert unit == "c"
    assert atspectra.get_shell_labels(edges, unit) == ["[0, 0.1) c", "[0.1, 0.2) c", "[0.2, 0.3) c"]


def test_spectra_losvelocity_shell_contributions() -> None:
    """A line-of-sight shell holds the packets by the signed velocity along the packet direction."""
    shells = [-51000.0, -20000.0, 0.0, 20000.0, 51000.0]
    contributions, array_flambda_emission_total, array_lambda = get_contributions_classic_3d(
        groupby="losvelocity", shelledges=shells
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

    realgetmodeldata = artistools.inputmodel.get_modeldata

    def get_modeldata_with_ye(*args: t.Any, **kwargs: t.Any) -> tuple[pl.LazyFrame, dict[str, t.Any]]:
        # the test model has no Ye column, thus the odd cells get 0.2 and the even cells get 0.4
        dfmodel, modelmeta = realgetmodeldata(*args, **kwargs)
        return dfmodel.with_columns(Ye=0.2 + 0.2 * (pl.col("inputcellid") % 2 == 0).cast(pl.Float32)), modelmeta

    with mock.patch("artistools.inputmodel.get_modeldata", side_effect=get_modeldata_with_ye):
        contributions, array_flambda_emission_total, array_lambda = get_contributions_classic_3d(
            groupby="ye", shelledges=[0.0, 0.3, 0.6]
        )
    _, array_flambda_emission_total_ion, _ = get_contributions_classic_3d(groupby="ion")

    assert sorted(contrib.linelabel for contrib in contributions) == ["Ye [0, 0.3)", "Ye [0.3, 0.6)"]
    assert np.allclose(array_flambda_emission_total, array_flambda_emission_total_ion, rtol=1e-6, atol=0.0)
    assert np.trapezoid(sum(contrib.array_flambda_absorption for contrib in contributions), x=array_lambda) > 0.0

    with pytest.raises(ValueError, match="no Ye column"):
        get_contributions_classic_3d(groupby="ye", shelledges=[0.0, 0.3, 0.6])


def test_spectra_velocity_shell_contributions_need_shell_edges() -> None:
    with pytest.raises(ValueError, match="needs the shell edges"):
        get_contributions_classic_3d(groupby="velocity")

    for badshells in ([0.0, 20000.0, 10000.0], [0.0, math.inf], [0.0, math.nan, 20000.0]):
        with pytest.raises(ValueError, match="must be finite, increase"):
            get_contributions_classic_3d(groupby="velocity", shelledges=badshells)

    with pytest.raises(ValueError, match="edges of the losvelocity range must be finite, increase"):
        get_contributions_classic_3d(velocityranges={"losvelocity": (5000.0, -5000.0)})


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
        groupby="velocity",
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
        groupby="velocity",
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
            "-emissionlosvelocityrange",
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
        emissionvelocityrange=["0.04c", "0.06c"],
        emissionlosvelocityrange=[-15000, 15000],
    )

    assert mockstackplot.call_count == 2
    assert len(mockstackplot.call_args_list[1].args[2]) >= 2
    title = mocksettitle.call_args_list[-1].args[1]
    assert title.endswith(", packets at radial velocity [0.04, 0.06) c and line-of-sight velocity [-15000, 15000) km/s")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectraemissionplot_linewidth_arg(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The -linewidth list sets the width of the net spectrum and of a reference spectrum on the emission plot."""
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath_classic_3d, "2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "emission_linewidth.pdf",
        timemin=4,
        timemax=6.5,
        showemission=True,
        groupby="velocity",
        velocityshells=["0c", "0.04c", "0.06c", "0.1c"],
        linewidth=[0.5, 7.0],
    )

    # the net spectrum of the model is the first line, and the reference spectrum is the second line
    assert [callargs.kwargs.get("linewidth") for callargs in mockplot.call_args_list] == [0.5, 7.0]


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
        groupby="losvelocity",
    )

    edges, _ = atspectra.get_default_losvelocity_shells(modelpath_classic_3d)
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
        groupby="velocity",
        velocityshells=list(np.linspace(8000.0, 28000.0, 21)),
        maxseriescount=3,
    )

    assert len(mockstackplot.call_args_list[0].args[2]) == 4


@pytest.mark.parametrize("packetargs", [{"gamma": True}, {"plotvspecpol": [0]}])
@pytest.mark.parametrize(
    "optionargs",
    [{"groupby": "velocity"}, {"emissionvelocityrange": [5000, 10000]}, {"emissionlosvelocityrange": [-5000, 5000]}],
)
def test_spectraemissionplot_refuses_packets_with_no_emission_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], optionargs: dict[str, t.Any], packetargs: dict[str, t.Any]
) -> None:
    """A shell grouping and a velocity range stop with a message for gamma packets and for virtual packets."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath_classic_3d,
            timemin=4,
            timemax=6.5,
            showemission=True,
            outputfile=tmp_path / "noposition.pdf",
            **optionargs,
            **packetargs,
        )
    assert "does not accept" in capsys.readouterr().err


def test_spectraemissionplot_velocity_shells_reject_an_empty_selection(tmp_path: Path) -> None:
    """A shell selection that holds no packet stops with a message when the plot is normalised."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath_classic_3d,
            timemin=4,
            timemax=6.5,
            showemission=True,
            groupby="velocity",
            velocityshells=["0.5c", "0.6c"],
            normalised=True,
            outputfile=tmp_path / "empty.pdf",
        )


def test_spectra_get_flux_contributions(benchmark: BenchmarkFixture) -> None:
    timestepmin = 40
    timestepmax = 80
    dfspectrum = at.spectra.get_spectra(
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


def test_spectra_get_flux_contributions_wavelength_window() -> None:
    """A wavelength window restricts the spectra and the flux contributions used for ranking."""
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

    nu_select = (arraylambda_full >= lambda_min) & (arraylambda_full <= lambda_max)
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
    dfspectrum = at.spectra.get_from_packets(
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

    get_flux_contributions_from_packets does not call get_from_packets, thus it must apply the solid angle of the bin.
    """
    lambda_bin_edges = np.arange(3000.0, 9020.0, 20.0)
    dfspectrum = at.spectra.get_from_packets(
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
@mock.patch("artistools.spectra.plotspectra.get_flux_contributions_from_packets")
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


def test_spectra_gamma_emission_time_uses_decay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gamma packet selection must use the decay time and not the radiation packet emission time."""
    dfpackets = pl.DataFrame({
        "e_rf": [1.0, 2.0],
        "t_arrive_d": [1.0, 3.0],
        "tdecay": [1.0 * at.constants.day_to_s, 3.0 * at.constants.day_to_s],
        "em_time": [3.0 * at.constants.day_to_s, 1.0 * at.constants.day_to_s],
        "nu_rf": [at.constants.c_ang_per_s / 5000.0] * 2,
    })

    def get_packets(*_args: t.Any, **_kwargs: t.Any) -> tuple[int, pl.LazyFrame]:
        return 1, dfpackets.lazy()

    monkeypatch.setattr(atspectra, "get_packets", get_packets)
    dfspectrum = atspectra.get_from_packets(
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
    monkeypatch: pytest.MonkeyPatch, beta: float, expected_e_cmf_sum: float
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

    def get_packets(*_args: t.Any, **_kwargs: t.Any) -> tuple[int, pl.LazyFrame]:
        return 1, dfpackets.lazy()

    def get_escape_surface_gamma(_modelpath: Path | str) -> float:
        return math.sqrt(1.0 - beta**2)

    def get_nuclides(modelpath: Path | str) -> pl.LazyFrame:
        del modelpath
        return pl.LazyFrame({"pellet_nucindex": [0], "nucname": ["Ni56"]})

    monkeypatch.setattr(atspectra, "get_packets", get_packets)
    monkeypatch.setattr(atspectra, "get_escape_surface_gamma", get_escape_surface_gamma)
    monkeypatch.setattr(atspectra, "get_nuclides", get_nuclides)

    dfspectrum = atspectra.get_from_packets(
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
    dfspectrum = atspectra.get_from_packets(
        modelpath=modelpath_classic_3d,
        timelowdays=3.0,
        timehighdays=8.0,
        lambda_bin_edges=np.linspace(3500.0, 8000.0, 101),
        use_time="escape",
        directionbins=[-1],
    )[-1].collect()

    assert dfspectrum["f_lambda"].is_finite().all()
    assert dfspectrum["f_lambda"].sum() > 0.0


@pytest.mark.benchmark
def test_spectra_timeseries_subplots() -> None:
    timedayslist = [295, 300]
    at.spectra.plot(
        argsraw=[], specpath=modelpath, outputfile=outputpath, timedayslist=timedayslist, multispecplot=True
    )


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


def test_write_data_skips_reference_spectra(tmp_path: Path) -> None:
    """--write_data must write one column per ARTIS model and nothing for a reference spectrum.

    Only the ARTIS branch produces series data, so an unset variable used to raise UnboundLocalError for a
    reference spectrum alone, and to re-write the preceding model's spectrum as a bogus extra column after it.
    """
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, "2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "spec.pdf",
        timedays="300",
        write_data=True,
    )

    dfout = pl.read_csv(tmp_path / "spec.txt", separator=" ")
    assert [col for col in dfout.columns if col.startswith("f_lambda")] == ["f_lambda.TEST MODEL"]

    # a reference spectrum on its own has nothing to write, but must still plot without error
    at.spectra.plot(
        argsraw=[],
        specpath=["2003du_20031213_3219_8822_00.txt"],
        outputfile=tmp_path / "refonly.pdf",
        timedays="300",
        write_data=True,
    )
    assert (tmp_path / "refonly.pdf").is_file()


def test_plotspectra_title_arg() -> None:
    parser = argparse.ArgumentParser()
    at.spectra.plotspectra.addargs(parser)
    assert parser.parse_args(["-title", "Custom title"]).title == "Custom title"
    assert parser.parse_args([]).title is None


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


def test_obsspec_draws_the_reference_spectrum(capsys: pytest.CaptureFixture[str]) -> None:
    """-obsspec names a reference spectrum, as a positional path does.

    Nothing read the list that -obsspec filled, thus the command took the file, drew the model alone,
    and said nothing about it.
    """
    refspec = "2003du_20031213_3219_8822_00.txt"
    at.spectra.plot(
        argsraw=[], specpath=[modelpath], refspecfiles=[refspec], timedays=300, outputfile=outputpath / "obsspec.pdf"
    )
    assert "SN2003du" in capsys.readouterr().out, "the reference spectrum must be drawn"


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

    # --logscaley replaces the default scale
    mockyscale.reset_mock()
    at.spectra.plot(argsraw=[], specpath=[modelpath], logscaley=True, timedays=300, outputfile=outputpath / "sp.pdf")
    assert [call.args[1] for call in mockyscale.call_args_list] == ["log"]


def test_explicit_linear_yscale_with_logscaley_stops_the_command() -> None:
    """An explicit -yscale linear gives a different scale from --logscaley, also when linear is the default.

    resolve_yscale compared -yscale with the default of the command, thus it did not see an explicit linear as a
    choice of the user.
    """
    with pytest.raises(SystemExit):
        at.misc.parse_cli_args(plotspectra.addargs, None, None, [str(modelpath), "-yscale", "linear", "--logscaley"])
    # a keyword becomes a default of the parser, and it is still a choice of the user
    with pytest.raises(SystemExit):
        at.misc.parse_cli_args(plotspectra.addargs, None, None, None, {"yscale": "linear", "logscaley": True})

    # the viewer has no box item for the alias "lin", thus the resolved scale has the spelling "linear"
    args = at.misc.parse_cli_args(plotspectra.addargs, None, None, [str(modelpath), "-yscale", "lin"])
    assert (args.yscale, args.logscaley) == ("linear", False)
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "-yscale", "lin", "--interactive"])
    assert viewer.values.yscale in viewer.yscalechoices


def test_a_unit_that_no_spectrum_takes_stops_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    """-xunit reads the name while argparse parses, thus a mistake stops the command before it reads a file.

    The command took every name and stopped later with "Unknown xunit", after it read the spectra.
    """
    import artistools.spectra.plotspectra

    parser = at.commands.SuggestingArgumentParser(prog="plotspectra")
    artistools.spectra.plotspectra.addargs(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([".", "-xunit", "micrometre"])

    assert "Did you mean micron?" in capsys.readouterr().err

    # a name of a unit is not case sensitive, and each spelling gives the canonical one
    assert parser.parse_args([".", "-xunit", "Angstrom"]).xunit == "angstroms"
    assert parser.parse_args([".", "-x", "MU"]).xunit == "micron"


@pytest.mark.parametrize("xunit", ["angstroms", "nm", "micron", "hz", "ev", "kev", "mev", "erg"])
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


def test_get_spectra_gamma_does_not_mix_in_uvoir_dirbins() -> None:
    """Gamma spectra must not pick up the UVOIR direction-resolved file.

    test-classicmode_3d has spec_res.out but no gamma_spec_res.out, so a gamma request must return only the
    spherically averaged bin rather than silently pairing gamma_spec.out with the UVOIR direction bins.
    """
    dirbin_spectra_uvoir = atspectra.get_spectra(modelpath=modelpath_classic_3d, timestepmin=10, timestepmax=10)
    assert -1 in dirbin_spectra_uvoir
    assert len(dirbin_spectra_uvoir) > 1, "expected direction-resolved UVOIR spectra in this test model"

    dirbin_spectra_gamma = atspectra.get_spectra(
        modelpath=modelpath_classic_3d, timestepmin=10, timestepmax=10, gamma=True
    )
    assert set(dirbin_spectra_gamma) == {-1}

    # the gamma spectrum must actually differ from the UVOIR one, i.e. it came from gamma_spec.out
    nu_gamma = dirbin_spectra_gamma[-1].select(pl.col("nu").max()).collect().item()
    nu_uvoir = dirbin_spectra_uvoir[-1].select(pl.col("nu").max()).collect().item()
    assert nu_gamma > nu_uvoir


def test_get_spectra_rejects_averaging_over_both_angles() -> None:
    """Averaging over phi and theta at once leaves too few bins, so it must be rejected up front."""
    with pytest.raises(ValueError, match="both the phi and theta"):
        atspectra.get_spectra(
            modelpath=modelpath_classic_3d,
            timestepmin=10,
            timestepmax=10,
            average_over_phi=True,
            average_over_theta=True,
        )


def test_reference_spectrum_named_out_is_not_read_as_artis(tmp_path: Path) -> None:
    """A user can name reference data with the .out suffix that ARTIS uses for its own output.

    path_is_artis_model reads that suffix, thus the predicate sent such a file to the ARTIS reader and
    the command drew nothing. The folder decides: an ARTIS run holds input.txt beside its output.
    """
    import shutil

    from artistools.spectra.plotspectra import path_is_reference_spectrum

    reference = at.get_path("artistools_dir") / "data" / "refspectra" / "2010lp_20110928_fors2.txt"
    shutil.copy(reference, tmp_path / "myref.out")
    shutil.copy(reference, tmp_path / "myref.txt")

    assert path_is_reference_spectrum(tmp_path / "myref.out")
    assert path_is_reference_spectrum(tmp_path / "myref.txt")

    # the output of a real run keeps its own reading, whether the user names the folder or the file
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
            xunit="angstroms",
            modelpath=modelpath,
        )


def test_spectraplot_falls_back_to_the_angle_averaged_spectrum(tmp_path: Path) -> None:
    """A model with no spec_res.out shows direction bin -1 in place of the requested bin, and that bin needs a label."""
    at.spectra.plot(argsraw=[], specpath=[modelpath], plotviewingangle=[5], timedays=290, outputfile=tmp_path)

    assert list(tmp_path.glob("*.pdf"))


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


def test_plotspectra_emission_refuses_an_x_range_without_a_bin(tmp_path: Path) -> None:
    """An x range that holds no wavelength bin must stop with a message, not with an error in the flux sums."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath,
            outputfile=tmp_path,
            timedays=300,
            emissionabsorption=True,
            use_thermalemissiontype=True,
            xmin=5000.0,
            xmax=5000.5,
        )


@mock.patch("artistools.spectra.plotspectra.get_flux_contributions_from_packets")
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


def test_plotspectra_skips_a_folder_that_is_not_a_run(tmp_path: Path) -> None:
    """A folder without input.txt gives no timesteps, thus the plot of the other models must still run.

    The range resolution of main asked the first folder for its timesteps and stopped before any plot.
    """
    notarun = tmp_path / "notarun"
    notarun.mkdir()
    outputfolder = tmp_path / "output"
    outputfolder.mkdir()

    at.spectra.plot(argsraw=["-timedays", "300", str(notarun), str(modelpath), "-outputfile", str(outputfolder)])

    assert len(list(outputfolder.glob("*.pdf"))) == 1


@pytest.mark.skipif(
    not (at.get_path("testdata") / "test-classicmode_1d").is_dir(),
    reason="run tests/data/setuptestdata.sh for the 1D classic model",
)
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

    The time range resolution asked such a folder for the timesteps of a run and raised
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


def test_plotspectra_accepts_classicartis_and_says_it_does_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A script holds the spelling, thus the argument stays accepted. No code reads it for this command."""
    at.spectra.plot(argsraw=["--classicartis", "-timedays", "260-300", str(modelpath), "-outputfile", str(tmp_path)])

    assert "ignores --classicartis" in capsys.readouterr().err


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
    dfmodel = at.spectra.get_spectra(modelpath, timestepmin=54, timestepmax=54)[-1].collect()
    obsfile = folder / "fakeobs.txt"
    dfmodel.select("lambda_angstroms", (pl.col("f_lambda") * 1.1).alias("flux")).sort("lambda_angstroms").write_csv(
        obsfile, separator=" ", include_header=False
    )
    obsfile.with_name(f"{obsfile.name}.meta.yml").write_text(
        "dist_mpc: 1\nlabel: fake observation\nt: 300\n", encoding="utf-8"
    )
    return obsfile


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_spectra_residual_panel_gives_model_minus_reference(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """An observed flux of 1.1 times the model gives a residual below zero, and a ratio of 1 / 1.1 with --logscaley."""
    obsfile = write_fake_observed_spectrum(tmp_path)
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, obsfile],
        timestep=54,
        residuals=True,
        write_data=True,
        outputfile=tmp_path / "residuals.pdf",
    )
    dfstats = pl.read_csv(tmp_path / "residuals_residuals.csv")
    assert dfstats.height == 1
    assert dfstats["reference"].item() == "fake observation"
    assert dfstats["npoints"].item() > 100
    assert dfstats["rms"].item() > 0.0
    assert 0.0 < dfstats["rms_relative"].item() < 0.1 / 1.1 * 3.0

    # the last plot call draws the residual line: the model is below the observed flux at each point
    residual = np.asarray(mockplot.call_args_list[-1].args[2])
    assert (residual <= 0.0).all()
    assert (residual < 0.0).any()

    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, obsfile],
        timestep=54,
        residuals=True,
        logscaley=True,
        outputfile=tmp_path / "ratio.pdf",
    )
    # a reference flux of zero gives a gap
    ratio = np.asarray(mockplot.call_args_list[-1].args[2])
    assert np.isfinite(ratio).any()
    assert np.allclose(ratio[np.isfinite(ratio)], 1.0 / 1.1)


def test_spectra_residual_panel_refuses_a_plot_with_no_pair(tmp_path: Path) -> None:
    """--residuals needs a model and an observed spectrum, and one frame."""
    with pytest.raises(SystemExit):
        at.spectra.plot(argsraw=[], specpath=[modelpath], timestep=54, residuals=True, outputfile=tmp_path / "a.pdf")

    obsfile = write_fake_observed_spectrum(tmp_path)
    # --output_spectra draws no figure, thus it cannot hold a residual panel
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[], specpath=[modelpath, obsfile], output_spectra=True, residuals=True, outputfile=tmp_path
        )

    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath, obsfile],
            timedayslist=["280", "300"],
            residuals=True,
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
    assert bottom == 0.0
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

    columns = pl.read_csv(outputfile.with_suffix(".txt"), separator=" ").columns
    assert sum(column.startswith("f_lambda") and column.endswith("_dirbin00") for column in columns) == 1
    assert sum(column.startswith("f_lambda") and column.endswith("_dirbin05") for column in columns) == 1


def test_get_specpol_data_reads_one_direction_bin() -> None:
    """ARTIS writes one file specpol_res.out that holds a table for each direction bin.

    The reader asked for specpol_res_<dirbin>.out, which no run holds, thus every direction bin
    raised FileNotFoundError.
    """
    dfdirbin = at.spectra.get_specpol_data(dirbin=0, modelpath=modelpath_classic_3d)["I"].collect()
    dfaveraged = at.spectra.get_specpol_data(dirbin=-1, modelpath=modelpath_classic_3d)["I"].collect()

    assert dfdirbin.columns == dfaveraged.columns
    assert dfdirbin.columns[0] == "nu"
    assert dfdirbin.height == dfaveraged.height
    assert not dfdirbin.equals(dfaveraged)


def test_plotspectra_emission_refuses_two_models(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An emission plot draws one ARTIS model, thus two of them give a message and not a silent choice."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath, modelpath_classic_3d],
            outputfile=tmp_path / "twomodels.pdf",
            timemin=290,
            timemax=320,
            emissionabsorption=True,
        )

    assert "one ARTIS model" in capsys.readouterr().err


def test_plotspectra_emission_refuses_two_direction_bins(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An emission plot draws one direction bin, and the name of its file gives every requested bin."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath_classic_3d,
            outputfile=tmp_path / "twobins.pdf",
            timemin=4,
            timemax=6.5,
            showemission=True,
            plotviewingangle=[0, 5],
        )

    assert "one direction bin" in capsys.readouterr().err


def test_plotspectra_vpkt_exclusion_names_the_missing_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--vpkt_match_emission_exclusion_to_opac names the options that it needs, and does not fail an assert."""
    with pytest.raises(SystemExit):
        at.spectra.plot(
            argsraw=[],
            specpath=modelpath,
            outputfile=tmp_path / "vpktexclusion.pdf",
            timemin=290,
            timemax=320,
            vpkt_match_emission_exclusion_to_opac=True,
        )

    captured = capsys.readouterr().err
    assert "--showemission" in captured
    assert "-plotvspecpol" in captured


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
    vpktconfig = {"nobsdirections": 1, "nspectraperobs": 2, "z_excludelist": [0, 26]}
    bflist = pl.LazyFrame({"bfindex": [], "ion_str": []}, schema={"bfindex": pl.Int32, "ion_str": pl.String})
    with (
        mock.patch.object(atspectra, "get_virtual_packets", return_value=(1, packets)),
        mock.patch.object(atspectra, "get_vpkt_config", return_value=vpktconfig),
        mock.patch.object(
            atspectra,
            "get_linelist_label_columns",
            return_value=pl.DataFrame({"atomic_number": [26, 27], "ion_stage": [2, 2]}),
        ),
        mock.patch.object(atspectra, "get_bflist", return_value=bflist),
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


def test_reference_spectrum_de_redshift_scales_the_flux(tmp_path: Path) -> None:
    """A de-redshift divides the wavelength by (1 + z) and multiplies f_lambda by (1 + z).

    The same energy falls in a narrower band in the rest frame, thus the flux density rises.
    """
    redshift = 0.5
    specfile = tmp_path / "redshifted.txt"
    specfile.write_text("4000 1.0\n5000 2.0\n6000 3.0\n", encoding="utf-8")
    specfile.with_suffix(".txt.meta.yml").write_text(f"---\nz: {redshift}\n", encoding="utf-8")

    specdata = atspectra.get_reference_spectrum(specfile)

    lambda_obs = np.array([4000.0, 5000.0, 6000.0])
    assert np.allclose(specdata["lambda_angstroms"].to_numpy(), lambda_obs / (1 + redshift), atol=0.0)
    assert np.allclose(specdata["f_lambda"].to_numpy(), np.array([1.0, 2.0, 3.0]) * (1 + redshift), atol=0.0)


@mock.patch("artistools.spectra.plotspectra.get_flux_contributions_from_packets")
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
        emissionvelocityrange=["0.04c", "0.06c"],
        emissionlosvelocityrange=[-15000, 15000],
    )

    velocityranges = mockgetcontributions.call_args.kwargs["velocityranges"]
    assert set(velocityranges) == {"velocity", "losvelocity"}
    assert np.isclose(velocityranges["velocity"][0], 0.04 * at.constants.C_cm_per_s / 1e5, rtol=1e-9, atol=0.0)
    assert np.isclose(velocityranges["velocity"][1], 0.06 * at.constants.C_cm_per_s / 1e5, rtol=1e-9, atol=0.0)
    assert velocityranges["losvelocity"] == (-15000.0, 15000.0)


def test_read_spec_follows_the_working_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The spectra of the default model path must change with the working folder.

    A cache held the relative Path("."). Thus a second model in the same process got the spectra of
    the first model.
    """
    for foldername, flux in (("modelA", 1.0), ("modelB", 2.0)):
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / "spec.out").write_text(f"0 10.0\n1e15 {flux}\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path / "modelA")
    assert atspectra.read_spec(Path()).collect()["10.0"].item() == pytest.approx(1.0)

    monkeypatch.chdir(tmp_path / "modelB")
    assert atspectra.read_spec(Path()).collect()["10.0"].item() == pytest.approx(2.0)


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
    dfout = pl.read_csv(tmp_path / "normalised.txt", separator=" ")
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
    assert args.xmin == 2e14
    assert np.isclose(args.xmax, at.spectra.convert_angstroms_to_unit(2500.0, "hz"), rtol=1e-12, atol=0.0)


def test_interactive_command_tokens() -> None:
    """The command of the viewer drops each form of an option that a control sets, and keeps the other options."""
    parser = interactive.make_parser()
    tokens = [
        "my model",
        "sn2011fe_PTF11kly_20120822_norm.txt",
        "-t300",
        "-xmin=3000",
        "-lambdamax",
        "9000",
        "-label",
        "foo",
        "bar",
        "--interactive",
        "--notimeclamp",
        "-timemin",
        "290",
        "-timema",
        "320",
        "-ts70",
        "-time",
        "300",
        "--emissionabsorption",
        "-maxseriescount",
        "20",
        "-groupby",
        "nuc",
        "-deltax=20",
        "--nostack",
        "-fixedionlist",
        "Fe II",
        "Co II",
        "-plotviewingangle",
        "-1",
        "--",
        "-folder",
    ]
    basetokens = interactive.remove_options(parser, tokens, interactive.CONTROLLED_DESTS)
    assert interactive.make_command_tokens(basetokens, ["-t", "306", "-xmin", "3000", "-xmax", "9000"]) == [
        "my model",
        "sn2011fe_PTF11kly_20120822_norm.txt",
        *("-t", "306", "-xmin", "3000", "-xmax", "9000"),
        # the viewer removes -plotviewingangle and also its negative value, which starts with "-"
        *("-label", "foo", "bar", "--", "-folder"),
    ]


def test_interactive_time_range_argument() -> None:
    """A snapped -timedays value must select its timesteps, and a continuous range must stay inside the valid times."""
    tmids, tstarts, tends = (at.get_timestep_times(modelpath, loc=loc) for loc in ("mid", "start", "end"))
    pairs = [(54, 58), (0, 3), (96, 99), (10, 11), *((timestep, timestep) for timestep in range(len(tmids)))]
    for first, last in pairs:
        timedays = interactive.get_snapped_timedays_argument(tmids, tstarts, tends, first, last)
        assert at.misc.get_time_range(modelpath, timedays_range_str=timedays)[:2] == (first, last), timedays
    assert interactive.get_snapped_timedays_argument(tmids, tstarts, tends, 54, 54) == "300"

    assert interactive.get_timedays_argument(306.4, 0.0, (250.0, 350.0)) == "306.4"
    assert interactive.get_timedays_argument(306.4, 5.0, (250.0, 350.0)) == "303.9-308.9"
    # the valid times of the test model start at 256.67 d, thus the range starts there
    assert interactive.get_timedays_argument(260.0, 10.0, (256.67, 333.82)) == "256.67-265"

    # format_days rounded a time at a bound to 6 decimals, and the time went outside the valid times. plotspectra
    # rejects such a time
    bounds = (0.1413524, 79.9999996)
    for centre, width in ((0.1413, 0.0), (0.1414, 0.001), (79.99999, 0.1), (40.0, 100.0)):
        for text in interactive.get_timedays_argument(centre, width, bounds).split("-"):
            assert bounds[0] <= float(text) <= bounds[1], (centre, width, text)


def test_interactive_continuous_range_keeps_its_bounds() -> None:
    """A --notimeclamp range of days keeps its bounds when the viewer opens, also inside one timestep.

    The viewer changed a range that holds the middle of one timestep or of no timestep into a single time, and a
    single time reads the whole timestep.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "299.5-301", "--notimeclamp", "--interactive"])
    tokens = shlex.split(viewer.get_command())
    assert tokens[tokens.index("-t") + 1] == "299.5-301"

    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--notimeclamp", "--interactive"])
    assert viewer.values.width == 0.0


def test_interactive_valid_timesteps() -> None:
    """The time controls stay inside the valid times, thus a step after the last valid timestep gives None."""
    viewer = make_headless_viewer([str(modelpath), "--interactive"])
    validstart, validend = viewer.timebounds
    assert viewer.tstarts[viewer.validtimesteps[0]] >= validstart > viewer.tstarts[0]
    assert viewer.tends[viewer.validtimesteps[-1]] <= validend < viewer.tends[-1]
    viewer.values = viewer.move_to_end(last=True)
    assert viewer.get_selection(viewer.values) == (viewer.validtimesteps[-1], viewer.validtimesteps[-1])
    assert viewer.step_time(1) is None
    assert viewer.draw() is None

    # a continuous time alone selects the whole timestep that holds it. At each end of the valid times, that
    # timestep is only in part valid, thus plotspectra rejected the command
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--notimeclamp", "--interactive"])
    for bound, timestep in ((validstart, viewer.validtimesteps[0]), (validend, viewer.validtimesteps[-1])):
        assert viewer.change(dc.replace(viewer.values, centre=bound, width=0.0)) is None
        assert viewer.values.centre == viewer.tmids[timestep]
        assert viewer.change(dc.replace(viewer.values, centre=bound, width=1.0)) is None


def make_headless_viewer(tokens: list[str]) -> interactive.SpectrumViewer:
    """Return a viewer that draws on a canvas with no window, after its first plot."""
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.SpectrumViewer(tokens, fig)
    assert viewer.draw() is None
    return viewer


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_interactive_command_reproduces_plot(mockplot: mock.MagicMock, tmp_path: Path) -> None:
    """The command that the viewer shows must draw the same data as the viewer."""
    viewer = make_headless_viewer([str(modelpath), "-t", "290", "--interactive"])
    # a continuous range gives --notimeclamp, and the snapped range below gives whole timesteps
    continuous = dc.replace(viewer.values, notimeclamp=True, centre=306.4, width=5.0, xmin="3000", xmax="9000")
    assert viewer.change(continuous) is None
    assert viewer.get_command().endswith(" -t 303.9-308.9 -xmin 3000 -xmax 9000 --notimeclamp")
    assert viewer.change(viewer.snap(viewer.values, 58, 62)) is None
    assert "timesteps 58 to 62" in viewer.get_timesteps_text()
    command = viewer.get_command()

    # each plot clears the frame first, thus the frame holds only the series of the model
    [viewerline] = viewer.axes[0].get_lines()
    viewerx, viewery = np.asarray(viewerline.get_xdata()), np.asarray(viewerline.get_ydata())

    mockplot.reset_mock()
    at.spectra.plot(argsraw=[*shlex.split(command)[2:], "-o", str(tmp_path / "spectrum.pdf")])
    assert mockplot.call_count == 1
    assert np.allclose(np.array(mockplot.call_args[0][1]), viewerx, rtol=1e-12, atol=0.0)
    assert np.allclose(np.array(mockplot.call_args[0][2]), viewery, rtol=1e-12, atol=0.0)


def test_interactive_emission_options() -> None:
    """A change that plotspectra rejects must keep the old values, and --showabsorption draws a taller frame."""
    # the test model has no emission.out, thus plotspectra cannot draw its emission plot
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    oldvalues, command = viewer.values, viewer.get_command()
    # the status line gives the first line of the error, which names the missing files
    message = viewer.change(dc.replace(viewer.values, showemission=True))
    assert message is not None
    assert message.startswith("None of these files exist")
    assert viewer.values == oldvalues
    assert viewer.get_command() == command

    # the viewer stores the default grouping as None. A stored "ion" was not equal to the None of the controls, thus a
    # change of an emission option removed the series lock
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "5", "-groupby", "ion", "--interactive"])
    assert viewer.values.groupby is None
    figheight = viewer.fig.get_figheight()
    assert viewer.change(dc.replace(viewer.values, showabsorption=True, maxseriescount=5, nostack=True)) is None
    assert viewer.get_command().endswith(" --showabsorption -maxseriescount 5 --nostack")
    assert viewer.fig.get_figheight() > figheight

    # the window disables a choice that plotspectra rejects, thus the test must find the rejection without a plot
    rejection = viewer.get_rejection(dc.replace(viewer.values, groupby="nuc"))
    assert rejection is not None
    assert "nuclide" in rejection


def test_interactive_xunit_and_references() -> None:
    """A new x unit converts and sorts the limits, and a reference spectrum goes after the model paths."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "-deltax", "20", "--interactive"])
    # a frequency increases where the wavelength decreases, thus the limit of 19000 Å becomes the minimum
    inhertz = interactive.convert_xunit(viewer.values, "hz", gamma=False)
    assert (inhertz.xmin, inhertz.xmax, inhertz.deltax) == ("1.578e+14", "1.199e+15", "")
    back = interactive.convert_xunit(inhertz, "angstroms", gamma=False)
    assert np.isclose(float(back.xmin), 2500.0, rtol=1e-3, atol=0.0)
    assert np.isclose(float(back.xmax), 19000.0, rtol=1e-3, atol=0.0)

    reference = "sn2011fe_PTF11kly_20120822_norm.txt"
    assert viewer.change(dc.replace(inhertz, references=(reference,))) is None
    tokens = shlex.split(viewer.get_command())[2:]
    assert tokens[:4] == [str(modelpath), reference, "-t", "300"]
    assert "-xunit" in tokens
    assert len(viewer.axes[0].get_lines()) == 2


def test_interactive_paths_keep_their_place() -> None:
    """Each path of the command stays a path and keeps its place, also a path after an option.

    The viewer took the paths only from the start of the command, and it put the references after the models. Thus a
    path after an option left the command, and -label named the wrong series.
    """
    reference = "sn2011fe_PTF11kly_20120822_norm.txt"
    viewer = make_headless_viewer([
        reference,
        str(modelpath),
        "-label",
        "Observed",
        "Model",
        "-t",
        "300",
        "--interactive",
    ])
    assert shlex.split(viewer.get_command())[2:4] == [reference, str(modelpath)]

    viewer = make_headless_viewer(["-t", "300", "--notitle", str(modelpath), "--interactive"])
    assert viewer.modelpathtokens == [str(modelpath)]
    # the command without the option starts with the path, and the viewer still reads the time of the command
    assert viewer.change(dc.replace(viewer.values, otheroptions=())) is None
    assert "timestep 54" in viewer.get_timesteps_text()

    # -fixedionlist reads each word that follows it, but a folder at the end of the command is a path
    viewer = make_headless_viewer([
        "--showemission",
        "-fixedionlist",
        "Fe II",
        "Co II",
        str(modelpath_classic_3d),
        "--interactive",
    ])
    assert viewer.modelpathtokens == [str(modelpath_classic_3d)]
    assert str(modelpath_classic_3d) in shlex.split(viewer.get_command())


def test_interactive_command_gives_the_series_count_of_the_box() -> None:
    """The command gives the count of the box, also when that count is the default without -fixedionlist.

    plotspectra gives a missing count the length of -fixedionlist, thus a count of 14 with a list needs the option.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--showemission", "--interactive"])
    series = viewer.get_drawn_series()
    for fixedionlist, maxseriescount in ((series[:3], plotspectra.DEFAULT_MAXSERIESCOUNT), (series[:3], 3), ((), 5)):
        values = dc.replace(viewer.values, fixedionlist=fixedionlist, maxseriescount=maxseriescount)
        args = at.misc.parse_cli_args(plotspectra.addargs, None, None, viewer.get_plot_tokens(values))
        plotspectra.resolve_plot_args(args)
        assert args.maxseriescount == maxseriescount


def test_interactive_fixed_y_axis() -> None:
    """A fixed y axis keeps a negative -ymin through the command, and the limits stay when the time changes."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    assert viewer.change(dc.replace(viewer.values, ymin="-1e-13", ymax="3e-13")) is None
    assert " -ymin -1e-13 -ymax 3e-13" in viewer.get_command()
    assert viewer.change(dc.replace(viewer.values, centre=305.0)) is None
    assert np.allclose(viewer.axes[0].get_ylim(), (-1e-13, 3e-13), rtol=1e-9, atol=0.0)


def test_interactive_figwidthscale_fills_the_plot_area() -> None:
    """The viewer sets -figwidthscale to give the figure the shape of the plot area, and the command shows it."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "-figwidthscale", "1.5", "--interactive"])
    for areawidth, areaheight in ((1600.0, 500.0), (600.0, 500.0)):
        figwidthscale = viewer.get_fitted_figwidthscale(areawidth, areaheight)
        assert viewer.change(dc.replace(viewer.values, figwidthscale=figwidthscale)) is None
        figwidth, figheight = viewer.fig.get_size_inches()
        # the scale has 2 decimals, thus the shape of the figure differs a little from the shape of the area
        assert np.isclose(figwidth / figheight, areawidth / areaheight, rtol=0.01)
        assert f" -figwidthscale {figwidthscale:g}" in viewer.get_command()


def test_maxseriescount_cuts_the_fixedionlist() -> None:
    """A smaller -maxseriescount removes the last entries of -fixedionlist, and a list alone keeps each entry.

    Before the correction, a -fixedionlist kept each entry, and -maxseriescount applied only to the names in the
    order of the flux.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--showemission", "--interactive"])
    series = viewer.get_drawn_series()
    assert len(series) > 2
    assert viewer.change(dc.replace(viewer.values, fixedionlist=series[::-1], maxseriescount=2)) is None
    assert viewer.get_drawn_series() == series[::-1][:2]

    for extraargs, maxseriescount in (("", 20), ("-maxseriescount 5", 5)):
        argsraw = [str(modelpath_classic_3d), "--showemission", "-fixedionlist", *(["Fe II"] * 20), *extraargs.split()]
        args = at.misc.parse_cli_args(plotspectra.addargs, None, None, argsraw)
        plotspectra.resolve_plot_args(args)
        assert args.maxseriescount == maxseriescount


def test_interactive_preview_reads_the_first_batch_of_ranks() -> None:
    """A preview of a plot of the packets reads the first batch of ranks, and the command keeps all the packets."""
    from artistools.packets.core import RANKS_PER_BATCH

    getcontributions = plotspectra.get_flux_contributions_from_packets
    # the test model has few ranks, thus a model of more than one batch comes from a patch of the rank count
    with (
        mock.patch.object(interactive, "get_nprocs", return_value=10 * RANKS_PER_BATCH),
        mock.patch.object(plotspectra, "get_flux_contributions_from_packets", wraps=getcontributions) as mockget,
    ):
        viewer = make_headless_viewer([
            str(modelpath_classic_3d),
            "-t",
            "4",
            "--showemission",
            "--frompackets",
            "--interactive",
        ])
        assert viewer.change(viewer.values, preview=True) is None
        assert viewer.drewpreview
        assert mockget.call_args.kwargs["maxpacketfiles"] == RANKS_PER_BATCH
        assert "-maxpacketfiles" not in viewer.get_command()

        assert viewer.change(viewer.values) is None
        assert not viewer.drewpreview
        assert mockget.call_args.kwargs["maxpacketfiles"] is None

        # a plot of the spectrum files reads no packets, thus it has no faster preview
        viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--interactive"])
        assert viewer.change(viewer.values, preview=True) is None
        assert not viewer.drewpreview

        # the reader divides the flux by the number of ranks, but not a count of packets
        viewer = make_headless_viewer([
            str(modelpath_classic_3d),
            "-t",
            "4",
            "--frompackets",
            "-yvariable",
            "packetcount",
            "--interactive",
        ])
        assert viewer.change(viewer.values, preview=True) is None
        assert not viewer.drewpreview


def test_interactive_assertion_of_plotspectra_is_a_rejection() -> None:
    """An AssertionError of plotspectra is a rejection, and the viewer keeps the old values.

    The viewer caught fewer errors than the CLI, thus the error left the window with the rejected values.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    oldvalues = viewer.values
    # resolve_plot_args asserts that the command does not give both kinds of viewing angle
    newvalues = dc.replace(oldvalues, otheroptions=(("-plotvspecpol", ("0",)), ("-plotviewingangle", ("0",))))
    assert viewer.get_rejection(newvalues) is not None
    assert viewer.change(newvalues) is not None
    assert viewer.values == oldvalues


def test_interactive_redraw_matches_a_new_plot() -> None:
    """A plot that the viewer draws again gives the same pixels as a new plot of the same command.

    The viewer keeps the tick objects and the title position from one plot to the next.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--interactive"])
    later = viewer.step_time(2)
    assert later is not None
    for values in (
        dc.replace(viewer.values, yscale="log", logscalex=True),
        dc.replace(viewer.values, showemission=True),
        dc.replace(later, showemission=True, maxseriescount=4),
    ):
        assert viewer.change(values) is None
    newviewer = make_headless_viewer([*shlex.split(viewer.get_command())[2:], "--interactive"])
    canvases = [viewer.fig.canvas, newviewer.fig.canvas]
    pixels = []
    for canvas in canvases:
        assert isinstance(canvas, FigureCanvasAgg)
        canvas.draw()
        pixels.append(np.asarray(canvas.buffer_rgba()).copy())
    assert np.array_equal(pixels[0], pixels[1])


def test_interactive_tick_labels_come_back_after_hidexticklabels() -> None:
    """The tick labels come back when the user removes --hidexticklabels from the table.

    cla() keeps the tick parameters, and the plot hides the labels only when the command gives the option.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    for otheroptions, labelsvisible in (((("--hidexticklabels", ()),), False), ((), True)):
        assert viewer.change(dc.replace(viewer.values, otheroptions=otheroptions)) is None
        viewer.fig.canvas.draw()
        assert viewer.axes[0].xaxis.get_major_ticks()[0].label1.get_visible() is labelsvisible


def test_interactive_status_line_gives_the_error() -> None:
    """The status line gives the error of argparse, and not the usage line that argparse prints before it."""
    stderr = "usage: artistools [options] [specpath ...]\nerror: argument -xmin: invalid float value: 'abc'\nhelp: -h"
    assert interactive.get_first_line(stderr) == "argument -xmin: invalid float value: 'abc'"
    assert interactive.get_first_line("A file is missing\nThe second line") == "A file is missing"

    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    rejection = viewer.get_rejection(dc.replace(viewer.values, deltax="20", otheroptions=(("-deltalambda", ("5",)),)))
    assert rejection is not None
    assert rejection.startswith("argument -deltalambda")


def test_interactive_zero_xmin_keeps_its_side_in_a_new_unit() -> None:
    """A minimum wavelength of 0 becomes a range of frequencies above the other limit, and not below it.

    The zero limit took the default limit at the same position, thus the range went to the other side of 5000 Å.
    """
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    values = dc.replace(viewer.values, xmin="0", xmax="5000")
    inhertz = interactive.convert_xunit(values, "hz", gamma=False)
    assert np.isclose(float(inhertz.xmin), at.constants.c_ang_per_s / 5000.0, rtol=1e-3, atol=0.0)
    # the range goes from 5000 Å to shorter wavelengths, which are higher frequencies
    assert float(inhertz.xmax) > float(inhertz.xmin)
    # a change between two units of wavelength keeps a minimum of 0
    inmicrons = interactive.convert_xunit(values, "micron", gamma=False)
    assert float(inmicrons.xmin) == 0.0


def test_interactive_single_time_ends_at_the_next_start() -> None:
    """A single time selects its timestep by the rule of get_timestep_of_timedays: a timestep ends at the next start.

    The end of a timestep can be a little after the start of the next one, and 2 then selected the next timestep.
    """
    tmids, tstarts, tends = [1.75, 2.25], [1.5, 2.0], [2.000008, 2.5]
    timedays = interactive.get_snapped_timedays_argument(tmids, tstarts, tends, 0, 0)
    assert tstarts[0] <= float(timedays) < tstarts[1]


def test_interactive_reference_name_finds_the_picked_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The command gives a bundled reference by its name only when plotspectra finds that same file by the name.

    plotspectra searches the working folder first, thus a file of the same name there took the place of the file
    that the user selected.
    """
    name = "sn2011fe_PTF11kly_20120822_norm.txt"
    bundledfile = plotspectra.find_reference_spectrum_file_or_none(name)
    assert bundledfile is not None
    monkeypatch.chdir(tmp_path)
    assert interactive.get_reference_token(str(bundledfile)) == name
    (tmp_path / name).write_text("1000 1\n2000 2\n")
    assert interactive.get_reference_token(str(bundledfile)) == str(bundledfile)


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
        atspectra.sort_and_reduce_flux_contribution_list(
            contributions, maxseriescount, arraylambda, fixedionlist=["Fe II", "Co II", "Ni II"]
        )
        assert ("did not find" in capsys.readouterr().err) is haswarning


def test_interactive_valid_times_of_each_run() -> None:
    """The time controls stay inside the times that are valid for every run of the command.

    plotspectra checks the times of each run, and the viewer took the valid times of the first run only.
    """
    # each call of the mock gives the valid range of the next run
    ranges = [(None, 260.0, 330.0), (None, 270.0, 320.0)]
    with mock.patch.object(interactive, "get_escaped_arrivalrange", side_effect=ranges):
        viewer = make_headless_viewer([str(modelpath), str(modelpath), "-t", "300", "--interactive"])
    assert viewer.timebounds == (270.0, 320.0)


def test_interactive_command_of_a_dispatcher_call() -> None:
    """A call of the dispatcher from Python code gives its own words to the viewer, and not the words of sys.argv."""
    from artistools.__main__ import main as dispatcher_main

    with (
        mock.patch.object(interactive, "run_viewer") as mockrunviewer,
        mock.patch.object(sys, "argv", ["myscript.py", "--flag", "x"]),
    ):
        dispatcher_main(argsraw=["plotspectra", str(modelpath), "-t", "300", "--interactive"])
    mockrunviewer.assert_called_once_with([str(modelpath), "-t", "300", "--interactive"])


def test_interactive_unlock_gives_the_default_count() -> None:
    """A count that came from the length of -fixedionlist becomes the default count when the lock ends."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "--interactive"])
    locked = dc.replace(viewer.values, fixedionlist=("Fe II", "Co II"), maxseriescount=2)
    assert interactive.remove_series_lock(locked).maxseriescount == plotspectra.DEFAULT_MAXSERIESCOUNT
    assert interactive.remove_series_lock(dc.replace(locked, maxseriescount=5)).maxseriescount == 5


def test_interactive_typed_centre_gives_back_the_range() -> None:
    """The centre that the time field shows gives back the same range of timesteps, also for an even count.

    The viewer took the timestep that holds the centre as the middle, and an even range then moved one timestep.
    """
    tmids = at.get_timestep_times(modelpath, loc="mid")
    for count in (1, 2, 3, 4):
        for start in range(len(tmids) - count + 1):
            centre = float(f"{(tmids[start] + tmids[start + count - 1]) / 2.0:.4g}")
            assert interactive.get_nearest_range_start(tmids, centre, count) == start, (count, start)


def test_interactive_frompackets_box() -> None:
    """The --frompackets box, and not the table of the other options, shows the --frompackets that the user gave."""
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--frompackets", "--interactive"])
    assert viewer.values.frompackets
    assert not viewer.values.otheroptions
    assert "--frompackets" in shlex.split(viewer.get_command())

    assert viewer.change(dc.replace(viewer.values, frompackets=False)) is None
    assert "--frompackets" not in shlex.split(viewer.get_command())


def test_interactive_direction_and_bin_controls() -> None:
    """The controls of the viewing direction and the bins take the options of the user, and give each option one time.

    The deprecated --average_every_tenth_viewing_angle becomes --average_over_phi_angle, and the table of the other
    options holds none of these options.
    """
    viewer = make_headless_viewer([
        str(modelpath_classic_3d),
        "-t",
        "4",
        "--average_every_tenth_viewing_angle",
        "-plotviewingangle",
        "10",
        "-dlogx",
        "0.002",
        "--normalised",
        "--interactive",
    ])
    values = viewer.values
    assert (values.directionkind, values.directionbins, values.deltalogx) == ("phi", (10,), "0.002")
    assert values.normalised
    assert not values.otheroptions
    command = shlex.split(viewer.get_command())
    for option in ("--average_over_phi_angle", "-plotviewingangle", "-deltalogx", "--normalised"):
        assert command.count(option) == 1
    assert "--average_every_tenth_viewing_angle" not in command
    assert command[command.index("-plotviewingangle") + 1] == "10"

    choices = interactive.get_direction_choices(modelpath_classic_3d, "theta", usedegrees=False)
    assert [dirbin for dirbin, _ in choices] == list(range(10))
    assert viewer.change(dc.replace(values, directionkind="theta", directionbins=(3,), deltalogx="")) is None
    command = shlex.split(viewer.get_command())
    assert "--average_over_theta_angle" in command
    assert not {"--average_over_phi_angle", "-deltalogx"} & set(command)

    # the legend gives the angles of the direction, and the command keeps --usedegrees only with a direction
    assert viewer.change(dc.replace(viewer.values, usedegrees=True)) is None
    assert "--usedegrees" in shlex.split(viewer.get_command())
    legend = viewer.axes[0].get_legend()
    assert legend is not None
    assert any("°" in text.get_text() for text in legend.get_texts())

    assert viewer.change(dc.replace(viewer.values, directionkind="", directionbins=())) is None
    command = set(shlex.split(viewer.get_command()))
    assert not {"-plotviewingangle", "--average_over_theta_angle", "--usedegrees"} & command


def test_interactive_option_rows() -> None:
    """The table of the window reads each form of an option that argparse accepts, and each row keeps its values."""
    parser = interactive.make_parser()
    tokens = ["-dx", "5", "-label", "a b", "c", "-filtersavgol", "5", "2", "--normalised", "-title=My plot", "-dpi300"]
    rows, othertokens = interactive.split_option_rows(parser, [*tokens, "--", "rest"])
    assert rows == (
        ("-deltax", ("5",)),
        ("-label", ("a b", "c")),
        ("-filtersavgol", ("5", "2")),
        ("--normalised", ()),
        ("-title", ("My plot",)),
        ("-dpi", ("300",)),
    )
    assert othertokens == ["--", "rest"]

    actions = {action.option_strings[0]: action for action in interactive.get_table_actions(parser)}
    # the other controls of the window set these options, and a list of times draws more than one plot
    assert (
        not {"-timedays", "-xmin", "-groupby", "-yvariable", "-plotviewingangle", "-timedayslist", "-h"}
        & actions.keys()
    )
    # no option of the table has choices now, but a new option with choices gets a list in the table
    allactions = {
        action.option_strings[0]: action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.option_strings
    }
    kinds = {flag: interactive.get_option_kind(allactions[flag]) for flag in ("--notitle", "-yvariable", "-dpi")}
    assert kinds == {"--notitle": "flag", "-yvariable": "choice", "-dpi": "int"}
    assert [interactive.get_option_kind(actions[flag]) for flag in ("-filtersavgol", "-label", "-title")] == [
        "values",
        "list",
        "text",
    ]
    # an option with no default needs a value from the user before the command can give it
    assert interactive.get_default_tokens(actions["-dpi"]) == ("250",)
    assert interactive.get_default_tokens(actions["-title"]) is None


def test_interactive_other_options_reach_the_command() -> None:
    """The command contains each row of the table, and a row that plotspectra rejects keeps the old options."""
    viewer = make_headless_viewer([str(modelpath), "-t", "300", "-filtermovingavg", "3", "--interactive"])
    assert viewer.values.otheroptions == (("-filtermovingavg", ("3",)),)

    otheroptions = (("-filtermovingavg", ("5",)), ("-title", ("A title",)))
    assert viewer.change(dc.replace(viewer.values, otheroptions=otheroptions)) is None
    command = viewer.get_command()
    assert " -filtermovingavg 5 -title 'A title'" in command
    assert viewer.axes[0].get_title() == "A title"

    # a ratio of Stokes parameters is a different plot from the one that the window shows
    message = viewer.change(dc.replace(viewer.values, otheroptions=(("-stokesparam", ("Q/I",)),)))
    assert message is not None
    assert "-stokesparam Q/I" in message
    assert viewer.values.otheroptions == otheroptions


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
        fig, axes, _, _ = plotspectra.make_plot(args)
    assert axes[0].get_yscale() == "linear"
    assert axes[0].get_ylim()[0] < 0.0
    plt.close(fig)


def test_interactive_lock_series() -> None:
    """A locked list of series keeps the colour of each series when the time changes."""
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "4", "--showemission", "--interactive"])

    def get_series_colours() -> dict[str, tuple[float, float, float, float]]:
        legend = viewer.axes[0].get_legend()
        assert legend is not None
        return {
            text.get_text(): mplcolors.to_rgba(handle.get_facecolor())
            for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
            if isinstance(handle, mpatches.Patch)
        }

    locked = viewer.get_drawn_series()
    assert len(locked) > 1
    assert viewer.change(dc.replace(viewer.values, fixedionlist=locked)) is None
    tokens = shlex.split(viewer.get_command())
    assert tokens[tokens.index("-fixedionlist") + 1 :] == list(locked)
    colours = get_series_colours()

    later = viewer.step_time(3)
    assert later is not None
    assert viewer.change(later) is None
    latercolours = get_series_colours()
    shared = set(colours) & set(latercolours)
    assert shared
    assert all(colours[name] == latercolours[name] for name in shared)
