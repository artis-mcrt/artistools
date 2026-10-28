import argparse
import math
import re
from pathlib import Path
from unittest import mock

import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import matplotlib.legend as mpllegend
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.container import ErrorbarContainer

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_residuals_take_the_reference_at_each_model_point() -> None:
    """Keep the model points inside the panel and reference ranges, with gaps for non-finite values."""
    model = at.plottools.ResidualSeries(
        "model", np.array([0.0, 5.0, 10.0, 12.0, 15.0, 20.0]), np.array([0.0, 10.0, 20.0, 24.0, 30.0, 40.0]), "C0"
    )
    reference = at.plottools.ResidualSeries(
        "obs",
        np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0]),
        np.array([1.0, 1.0, 11.0, 22.0, 33.0, 42.0, 50.0]),
        "k",
    )
    inrange, residual, yreference = at.plottools.get_residuals(reference, model, xmin=0.0, xmax=14.0)
    assert inrange.tolist() == [True, True, True, True, False, False]
    assert np.allclose(yreference, [1.0, 11.0, 22.0, 26.4])
    assert np.allclose(residual, [-1.0, -1.0, -2.0, -2.4])

    masked = reference._replace(y=np.array([1.0, 1.0, np.nan, 22.0, 33.0, 42.0, 50.0]))
    inrange, residual, _ = at.plottools.get_residuals(masked, model, xmin=0.0, xmax=20.0)
    assert inrange.all()
    assert np.isnan(residual[1])

    gapmodel = model._replace(y=np.array([0.0, 10.0, np.inf, 24.0, 30.0, 40.0]))
    _, residual, _ = at.plottools.get_residuals(reference, gapmodel, xmin=0.0, xmax=20.0)
    assert np.isnan(residual[2])
    assert np.isfinite(residual[[0, 1, 3, 4, 5]]).all()

    fig, axis = plt.subplots()
    dfstats = at.plottools.plot_residual_panel(axis, [masked, model], 0.0, 20.0)
    expected = np.array([-1.0, -2.0, -2.4, -3.0, -2.0])
    assert dfstats["npoints"].item() == 5
    assert np.isclose(dfstats["rms"].item(), np.sqrt(np.mean(expected**2)))
    assert np.isclose(dfstats["rms_relative"].item(), dfstats["rms"].item() / np.mean([1.0, 22.0, 26.4, 33.0, 42.0]))
    assert np.allclose(np.asarray(axis.lines[0].get_ydata())[[0, 2, 3, 4, 5]], expected)
    plt.close(fig)

    fig, ratioaxis = plt.subplots()
    at.plottools.plot_residual_panel(ratioaxis, [masked, model], 0.0, 20.0, residualtype="relative")
    assert np.allclose(
        np.asarray(ratioaxis.lines[0].get_ydata())[[0, 2, 3, 4, 5]],
        [0.0, 20.0 / 22.0, 24.0 / 26.4, 30.0 / 33.0, 40.0 / 42.0],
    )
    dfmagstats = at.plottools.plot_residual_panel(ratioaxis, [masked, model], 0.0, 20.0, ismagnitude=True)
    assert dfmagstats["rms_relative"].item() is None
    plt.close(fig)


@pytest.mark.parametrize("referencex", [[], [np.nan, np.inf, -np.inf], [2.0], [np.nan, 2.0, np.inf]])
@pytest.mark.parametrize("ratio", [False, True])
def test_residual_panel_accepts_exact_matches_to_a_single_baseline_point(referencex: list[float], ratio: bool) -> None:
    """Keep exact matches to one finite baseline point and exclude other x values."""
    x = np.array([1.0, np.nextafter(2.0, 0.0), 2.0, 2.0, np.nextafter(2.0, 3.0), 3.0, np.nan])
    model = at.plottools.ResidualSeries("model", x, np.array([1.0, 3.0, 6.0, 8.0, 7.0, 9.0, 10.0]), "C0")
    reference = at.plottools.ResidualSeries(
        "baseline", np.asarray(referencex, dtype=np.float64), np.full(len(referencex), 4.0), "k"
    )
    inrange, residual, yreference = at.plottools.get_residuals(reference, model, xmin=1.0, xmax=3.0)
    fig, axis = plt.subplots()
    stats = at.plottools.plot_residual_panel(
        axis, [reference, model], 1.0, 3.0, residualtype="relative" if ratio else "absolute"
    )
    if np.isfinite(reference.x).any():
        assert inrange.tolist() == [False, False, True, True, False, False, False]
        assert np.allclose(residual, [2.0, 4.0])
        assert np.allclose(yreference, [4.0, 4.0])
        assert stats["npoints"].item() == 2
        assert np.isclose(stats["rms"].item(), np.sqrt(10.0))
        assert np.isclose(stats["rms_relative"].item(), np.sqrt(10.0) / 4.0)
        assert np.allclose(axis.lines[0].get_xdata(), [2.0, 2.0])
        assert np.allclose(axis.lines[0].get_ydata(), [1.5, 2.0] if ratio else [2.0, 4.0])
        outside, outside_residual, _ = at.plottools.get_residuals(reference, model, xmin=2.1, xmax=3.0)
        assert not outside.any()
        assert outside_residual.size == 0
    else:
        assert not inrange.any()
        assert residual.size == yreference.size == 0
        assert stats.is_empty()
    plt.close(fig)


@pytest.mark.parametrize("ratio", [False, True])
def test_residual_panel_keeps_the_resolution_and_line_properties(ratio: bool) -> None:
    """A sparse reference keeps each comparison point and the line properties of the main frame."""
    x = np.linspace(-1.0, 9.0, 1001)
    yreference = np.interp(x, [0.0, 4.0, 8.0], [2.0, 6.0, 18.0])
    y = yreference + np.sin(x)
    fig, (mainaxis, residualaxis) = plt.subplots(2)
    (mainline,) = mainaxis.plot(
        x,
        y,
        color="purple",
        linestyle=(1, (2, 3)),
        linewidth=2.5,
        marker="s",
        markersize=7,
        markerfacecolor="none",
        markeredgewidth=2,
        alpha=0.4,
        drawstyle="steps-mid",
    )
    reference = at.plottools.ResidualSeries("baseline", np.array([8.0, 0.0, 4.0]), np.array([18.0, 2.0, 6.0]), "k")
    model = at.plottools.ResidualSeries("comparison", x, y, mainline.get_color(), line=mainline)
    stats = at.plottools.plot_residual_panel(
        residualaxis, [reference, model], 1.0, 7.0, residualtype="relative" if ratio else "absolute"
    )
    inrange = (x >= 1.0) & (x <= 7.0)
    line = residualaxis.lines[0]
    assert np.allclose(line.get_xdata(), x[inrange])
    assert np.allclose(line.get_ydata(), (y / yreference if ratio else y - yreference)[inrange])
    assert stats["npoints"].item() == int(inrange.sum())
    assert line.get_transform() == residualaxis.transData
    assert line.get_transform() != mainline.get_transform()
    assert np.allclose(line.get_clip_box().bounds, residualaxis.bbox.bounds)
    assert line.get_color() == mainline.get_color()
    assert line.get_linestyle() == mainline.get_linestyle()
    assert np.isclose(line.get_linewidth(), mainline.get_linewidth())
    assert line.get_marker() == mainline.get_marker()
    assert np.isclose(line.get_markersize(), mainline.get_markersize())
    assert line.get_markerfacecolor() == mainline.get_markerfacecolor()
    assert np.isclose(line.get_markeredgewidth(), mainline.get_markeredgewidth())
    assert np.isclose(line.get_alpha(), mainline.get_alpha())
    assert line.get_drawstyle() == mainline.get_drawstyle()
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.parametrize("baselineindex", [0, 1, 2])
@pytest.mark.parametrize("selectedindices", [None, [], [0, 2], [2, 2], [1], [2, 0]])
@pytest.mark.parametrize("ratio", [False, True])
def test_residual_panel_uses_the_selected_baseline(
    baselineindex: int, selectedindices: list[int] | None, ratio: bool
) -> None:
    """Each other series uses the selected baseline, its points, and its own colour."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 8.0])
    factors = [1.0, 2.0, 4.0]
    series = [
        at.plottools.ResidualSeries(f"series {index}", x, y * factor, f"C{index}")
        for index, factor in enumerate(factors)
    ]
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(argparse.Namespace(logscaley=ratio))
    mainaxis.set_xlim(1.0, 3.0)
    dfstats = at.plottools.draw_residual_panel(
        residualaxis,
        mainaxis,
        series,
        argparse.Namespace(
            residualbaselineseries=baselineindex,
            residuals=selectedindices,
            logscaley=ratio,
            residualtype="relative" if ratio else "absolute",
        ),
    )["_residuals"]
    otherindices = [
        index
        for index in range(len(series))
        if index != baselineindex and (not selectedindices or index in selectedindices)
    ]
    assert dfstats["model"].to_list() == [series[index].label for index in otherindices]
    assert dfstats["reference"].to_list() == [series[baselineindex].label] * len(otherindices)
    assert len(residualaxis.lines) == len(otherindices) + 1
    for line, index in zip(residualaxis.lines[:-1], otherindices, strict=True):
        expected = (
            np.full_like(y, factors[index] / factors[baselineindex])
            if ratio
            else y * (factors[index] - factors[baselineindex])
        )
        assert np.allclose(line.get_ydata(), expected)
        assert line.get_color() == series[index].color
    expectedrms = [
        float(np.sqrt(np.mean((y * (factors[index] - factors[baselineindex])) ** 2))) for index in otherindices
    ]
    assert np.allclose(dfstats["rms"].to_numpy(), expectedrms)
    plt.close(fig)


@pytest.mark.parametrize(("seriescount", "baselineindex"), [(0, 0), (1, 0), (2, -1), (2, 2)])
def test_residual_panel_rejects_an_invalid_baseline(seriescount: int, baselineindex: int) -> None:
    """The panel needs two series and an index inside their range."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(index), x, x, "k") for index in range(seriescount)]
    fig, axis = plt.subplots()
    with pytest.raises(SystemExit):
        at.plottools.plot_residual_panel(axis, series, 1.0, 2.0, baselineindex=baselineindex)
    plt.close(fig)


@pytest.mark.parametrize("index", [-1, 2])
def test_residual_panel_rejects_an_invalid_selection(index: int) -> None:
    """The panel rejects a selected index outside the range of the series."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(i), x, x, "k") for i in range(2)]
    fig, axis = plt.subplots()
    with pytest.raises(SystemExit):
        at.plottools.plot_residual_panel(axis, series, 1.0, 2.0, selectedindices=[index])
    assert not axis.lines
    plt.close(fig)


@pytest.mark.parametrize("modelfactor", [2.0, 100.0, 0.01])
@pytest.mark.parametrize("logscaley", [False, True])
@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_residual_type_sets_the_calculation_and_scale(modelfactor: float, logscaley: bool, residualtype: str) -> None:
    """The residual type sets the values and the y scale independently of the main frame."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    yreference = np.array([1.0, 2.0, 4.0, 8.0])
    factors = np.array([1.0, 1.0, modelfactor, modelfactor])
    series = [
        at.plottools.ResidualSeries("obs", x, yreference, "k"),
        at.plottools.ResidualSeries("model", x, yreference * factors, "C0"),
    ]
    args = argparse.Namespace(logscaley=logscaley, residualbaselineseries=0, residuals=None, residualtype=residualtype)
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(args)
    mainaxis.plot(x, series[0].y)
    mainaxis.set_yscale("log" if logscaley else "linear")
    mainaxis.set_ylabel("Flux [erg]")
    stats = at.plottools.draw_residual_panel(residualaxis, mainaxis, series, args)["_residuals"]
    expected = yreference * (factors - 1.0) if residualtype == "absolute" else factors
    assert residualaxis.get_yscale() == ("log" if residualtype == "relativelog" else "linear")
    assert residualaxis.get_ylabel() == (
        "series $-$ baseline\n[erg]" if residualtype == "absolute" else "series / baseline"
    )
    assert np.allclose(residualaxis.lines[0].get_ydata(), expected)
    assert np.allclose(residualaxis.lines[-1].get_ydata(), 0.0 if residualtype == "absolute" else 1.0)
    assert np.isclose(stats["rms"].item(), np.sqrt(np.mean((series[1].y - yreference) ** 2)))
    plt.close(fig)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
@pytest.mark.parametrize("ismagnitude", [False, True])
def test_residual_y_maximum_keeps_the_main_range(residualtype: str, ismagnitude: bool) -> None:
    """The residual maximum sets the numerical limit and keeps the main range and the magnitude direction."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(i), x, x + i, "k") for i in range(2)]
    args = argparse.Namespace(
        logscaley=False, residualbaselineseries=0, residuals=None, residualtype=residualtype, residualymax=3.5
    )
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(args)
    mainaxis.set_xlim(1.0, 2.0)
    mainaxis.set_ylim(0.0, 10.0)
    at.plottools.draw_residual_panel(residualaxis, mainaxis, series, args, ismagnitude=ismagnitude)
    assert np.isclose(max(residualaxis.get_ylim()), 3.5)
    assert residualaxis.yaxis_inverted() == ismagnitude
    assert np.allclose(mainaxis.get_ylim(), [0.0, 10.0])
    plt.close(fig)


def test_residual_log_axis_ignores_a_non_positive_maximum(capsys: pytest.CaptureFixture[str]) -> None:
    """A log residual axis gives a warning and keeps its data range for a non-positive maximum."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(i), x, x + i, "k") for i in range(2)]
    args = argparse.Namespace(
        logscaley=False, residualbaselineseries=0, residuals=None, residualtype="relativelog", residualymax=0.0
    )
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(args)
    mainaxis.set_xlim(1.0, 2.0)
    at.plottools.draw_residual_panel(residualaxis, mainaxis, series, args)
    assert min(residualaxis.get_ylim()) > 0.0
    assert "Ignoring --residualymax 0.0" in capsys.readouterr().err
    plt.close(fig)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
@pytest.mark.parametrize("zeropoint", [0.0, 30.0])
def test_magnitude_residuals_keep_differences(residualtype: str, zeropoint: float) -> None:
    """Magnitude residuals keep their differences, scale, and error bars when the zero point changes."""
    x = np.array([1.0, 2.0])
    ybaseline = np.array([-10.0, -8.0]) + zeropoint
    yerr = np.array([[0.3, 0.4], [0.5, 0.6]])
    series = [
        at.plottools.ResidualSeries("baseline", x, ybaseline, "k"),
        at.plottools.ResidualSeries("model", x, ybaseline + 2.0, "r", yerr=yerr),
    ]
    args = argparse.Namespace(logscaley=False, residualbaselineseries=0, residuals=None, residualtype=residualtype)
    fig, mainaxis, residualaxis = at.plottools.make_frame_figure_with_residuals(args)
    mainaxis.set_xlim(1.0, 2.0)
    mainaxis.set_ylabel("Magnitude [mag]")
    with mock.patch.object(residualaxis, "errorbar", wraps=residualaxis.errorbar) as mockerrorbar:
        stats = at.plottools.draw_residual_panel(residualaxis, mainaxis, series, args, ismagnitude=True)["_residuals"]
    assert np.allclose(residualaxis.lines[0].get_ydata(), 2.0)
    assert np.allclose(residualaxis.lines[-1].get_ydata(), 0.0)
    assert np.allclose(mockerrorbar.call_args.kwargs["yerr"], yerr)
    assert residualaxis.get_yscale() == "linear"
    assert residualaxis.yaxis_inverted()
    assert residualaxis.get_ylabel() == "series $-$ baseline\n[mag]"
    assert np.isclose(stats["rms"].item(), 2.0)
    plt.close(fig)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_residual_type_keeps_gaps_for_undefined_values(residualtype: str) -> None:
    """Zero baselines give gaps in ratios, and a logarithmic axis also excludes non-positive ratios."""
    x = np.arange(5, dtype=np.float64)
    baseline = at.plottools.ResidualSeries("baseline", x, np.array([1.0, -2.0, 0.0, 2.0, np.nan]), "k")
    model = at.plottools.ResidualSeries("model", x, np.array([2.0, 4.0, 3.0, 0.0, 2.0]), "C0")
    expected = {
        "absolute": [1.0, 6.0, 3.0, -2.0, np.nan],
        "relative": [2.0, -2.0, np.nan, 0.0, np.nan],
        "relativelog": [2.0, np.nan, np.nan, np.nan, np.nan],
    }
    fig, axis = plt.subplots()
    stats = at.plottools.plot_residual_panel(axis, [baseline, model], 0.0, 4.0, residualtype=residualtype)
    assert np.allclose(axis.lines[0].get_ydata(), expected[residualtype], equal_nan=True)
    assert stats["npoints"].item() == 4
    assert np.isclose(stats["rms"].item(), np.sqrt(np.mean(np.array([1.0, 6.0, 3.0, -2.0]) ** 2)))
    plt.close(fig)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_residual_type_scales_asymmetric_error_bars(residualtype: str) -> None:
    """Ratio error bars use the baseline magnitude and exchange their sides for a negative baseline."""
    x = np.array([1.0, 2.0])
    baseline = at.plottools.ResidualSeries("baseline", x, np.array([2.0, -4.0]), "k")
    model = at.plottools.ResidualSeries(
        "model", x, np.array([4.0, -8.0]), "C0", yerr=np.array([[0.4, 0.8], [0.8, 1.6]])
    )
    fig, axis = plt.subplots()
    at.plottools.plot_residual_panel(axis, [baseline, model], 1.0, 2.0, residualtype=residualtype)
    bars = axis.containers[0]
    assert isinstance(bars, ErrorbarContainer)
    segments = bars.lines[2][0].get_segments()
    expected = [[1.6, 2.8], [-4.8, -2.4]] if residualtype == "absolute" else [[1.8, 2.4], [1.6, 2.2]]
    assert np.allclose([segment[:, 1] for segment in segments], expected)
    plt.close(fig)


def test_residual_panel_rejects_an_unknown_type() -> None:
    """The panel rejects a residual type that it cannot calculate."""
    x = np.array([1.0, 2.0])
    series = [at.plottools.ResidualSeries(str(index), x, x, "k") for index in range(2)]
    fig, axis = plt.subplots()
    with pytest.raises(SystemExit):
        at.plottools.plot_residual_panel(axis, series, 1.0, 2.0, residualtype="unknown")
    assert not axis.lines
    plt.close(fig)


def test_frame_figure_takes_a_shorter_row() -> None:
    """A residual panel takes a part of the frame height, and the main frame keeps its size."""
    fig, axes = at.plottools.make_frame_figure(rows=2, rowheights=(1.0, 0.35))
    fig.canvas.draw()
    mainheight = axes[0][0].get_position().height
    residualheight = axes[1][0].get_position().height
    assert np.isclose(residualheight / mainheight, 0.35, rtol=1e-3)
    # row 0 is at the top
    assert axes[0][0].get_position().y0 > axes[1][0].get_position().y1
    plt.close(fig)

    figone, axesone = at.plottools.make_frame_figure()
    figone.canvas.draw()
    figtwo, axestwo = at.plottools.make_frame_figure(rows=2, rowheights=(1.0, 0.35))
    figtwo.canvas.draw()
    inchesone = axesone[0][0].get_position().height * figone.get_figheight()
    inchestwo = axestwo[0][0].get_position().height * figtwo.get_figheight()
    assert np.isclose(inchesone, inchestwo, rtol=1e-3)
    plt.close(figone)
    plt.close(figtwo)

    with pytest.raises(ValueError, match="rowheights gives"):
        at.plottools.make_frame_figure(rows=2, rowheights=(1.0,))


def test_save_figure_darkmode_gives_a_black_png_and_a_transparent_pdf(tmp_path: Path) -> None:
    """--darkmode gives white text, a black PNG background, and a transparent PDF background.

    A transparent PNG frame of a gif shows as white in many image viewers, thus only a vector file is transparent.
    """
    import matplotlib.colors as mcolors
    import matplotlib.image as mplimage

    args = argparse.Namespace(darkmode=True)
    for filename in ("x.png", "x.pdf"):
        fig = plt.figure()
        axis = fig.add_subplot()
        axis.plot([0, 1], [0, 1], color="black")
        axis.set_xlabel("velocity")
        with mock.patch.object(mplfig.Figure, "savefig", side_effect=mplfig.Figure.savefig, autospec=True) as spy:
            at.plottools.save_figure(fig, tmp_path / filename, args=args)
        assert mcolors.same_color(axis.xaxis.label.get_color(), "white")
        assert mcolors.same_color(axis.lines[0].get_color(), "white")
        assert spy.call_args.kwargs["transparent"] == (filename == "x.pdf")

    corner = mplimage.imread(tmp_path / "x.png")[0, 0]
    assert np.allclose(corner, [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=0.01)


def get_kilonova_lightcurve() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the times and the luminosities of a light curve that rises to a peak and then decays."""
    times = np.geomspace(0.11, 76.0, 56)
    return times, 1e42 * np.where(times < 0.5, (times / 0.5) ** 2.0, (times / 0.5) ** -1.3)


@pytest.mark.parametrize(
    ("name", "values", "wantslog"),
    [
        ("a flat series", np.linspace(1.0, 2.0, 50), False),
        ("a decay over four decades", np.geomspace(1e4, 1.0, 100), True),
        ("a decay that falls away", np.exp(-np.linspace(0.0, 10.0, 100)), True),
        # half of the values of a ramp are above half of the top
        ("a ramp over four decades", np.linspace(1.0, 1e4, 100), False),
        ("one point of noise near zero", np.concatenate([np.full(100, 1.0), [1e-30]]), False),
        ("a value of zero in every second place", np.concatenate([np.zeros(50), np.geomspace(1.0, 1e4, 50)]), False),
        ("a few values of zero", np.concatenate([np.zeros(3), np.geomspace(1.0, 1e4, 97)]), True),
        ("no value at all", np.empty(0), False),
        ("fewer values than a percentile needs", np.array([1.0, 10.0, 100.0]), False),
        ("every value zero", np.zeros(20), False),
    ],
)
def test_wants_log_scale_reads_the_range_of_the_values(
    name: str, values: npt.NDArray[np.float64], wantslog: bool
) -> None:
    """A log scale belongs to values that a linear axis draws on the line of zero."""
    assert at.plottools.wants_log_scale(values.astype(np.float64)) is wantslog, name


def test_auto_yscale_weights_each_value_by_its_part_of_the_x_axis() -> None:
    """A spectrum keeps a linear axis, and a light curve at geometric times takes a log axis.

    The rule of the quartiles chose a log axis for a kilonova spectrum at 3 to 5 days. The rule of commit 1598ad51
    chose a log axis for a hot spectrum at the first days. Each value covers its part of the x axis, thus the few late
    times of a light curve cover most of a linear time axis.
    """
    wavelengths = np.linspace(2500.0, 19000.0, 400)
    # 30% of the wavelengths hold the blue end, where the flux rises over three decades. Thus the quartiles are far
    # apart
    coolspectrum = np.concatenate([
        np.geomspace(1e-4, 0.1, 120),
        np.linspace(0.3, 1.0, 140),
        np.linspace(1.0, 0.3, 140),
    ])
    # a hot spectrum has its peak at the blue end, and the flux decreases to the red end
    hotspectrum = (wavelengths / 2500.0) ** -2.0
    for xvalues, yvalues, wantslog in (
        (wavelengths, coolspectrum, False),
        (wavelengths, hotspectrum, False),
        (*get_kilonova_lightcurve(), True),
    ):
        fig, axis = plt.subplots()
        axis.plot(xvalues, yvalues)
        args = argparse.Namespace(yscale="auto", logscaley=False)
        at.plottools.set_auto_yscale(axis, args)
        assert args.logscaley is wantslog
        plt.close(fig)


def test_auto_yscale_reads_the_drawn_values() -> None:
    """-yscale auto takes a log scale from the drawn values, and the other choices stand."""
    for ydata, wantslog in ((np.geomspace(1.0, 1e4, 50), True), (np.linspace(1.0, 2.0, 50), False)):
        fig, axis = plt.subplots()
        axis.plot(np.arange(ydata.size), ydata)

        args = argparse.Namespace(yscale="auto", logscaley=False)
        at.plottools.set_auto_yscale(axis, args)
        assert args.logscaley is wantslog

        # -yscale linear and -yscale log each give the answer already, thus the values change nothing
        for yscale, logscaley in (("linear", False), ("log", True)):
            args = argparse.Namespace(yscale=yscale, logscaley=logscaley)
            at.plottools.set_auto_yscale(axis, args)
            assert args.logscaley is logscaley

        plt.close(fig)


def test_set_legend_draws_no_legend_without_a_labelled_series() -> None:
    """An axes that has no series with a label gets no legend. matplotlib raised ValueError for the empty handles."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])

    assert at.plottools.set_legend(ax) is None
    assert ax.get_legend() is None
    plt.close(fig)


@pytest.mark.parametrize("yscale", ["linear", "log", "inverted"])
def test_set_legend_gives_the_legend_room_clear_of_the_data(yscale: str) -> None:
    """The legend of set_legend does not cover the data, on a linear, a log, and an inverted axis.

    Each command gave its legend room with a factor of its own, e.g. a top of 1.2 times the tallest peak.
    """
    fig = mplfig.Figure(figsize=(5.0, 3.5))
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    xvalues = np.linspace(0.0, 10.0, 50)
    ax.plot(xvalues, 10.0 ** (1.0 + xvalues / 10.0), label="rising")
    # a line of two points crosses the whole legend, and only the ends of its segment are data points
    ax.plot([0.0, 10.0], [90.0, 90.0], label="flat")
    if yscale == "log":
        ax.set_yscale("log")
    at.plottools.set_legend(ax, loc="upper right")
    # a limit that comes after set_legend still leaves the legend its room, because the room follows at the draw
    ax.set_ylim(5.0, 100.0)
    if yscale == "inverted":
        ax.invert_yaxis()

    canvas.draw()
    renderer = canvas.get_renderer()
    legend = ax.get_legend()
    assert legend is not None
    frame = at.plottools.get_legend_frame(legend, renderer)
    yrange = at.plottools.get_data_fraction_range(ax, frame.x0, frame.x1)
    assert yrange is not None
    # the fractions of the axes run upwards also on an inverted axis, thus the data stays below the legend
    assert yrange[1] < frame.y0
    ylimits = ax.get_ylim()
    canvas.draw()
    assert np.allclose(ax.get_ylim(), ylimits, rtol=1e-12, atol=0.0), "a second draw must keep the limits"


def test_set_legend_room_leaves_out_full_width_spans() -> None:
    """An axhspan covers each x, e.g. the span of a Shift-drag of a viewer, and it must not move the top.

    The top of the axis moved during a Shift-drag near the legend, thus the drag moved under the pointer.
    """
    fig = mplfig.Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot([0.0, 10.0], [0.0, 1.0], label="rising")
    ax.set_ylim(0.0, 2.0)
    at.plottools.set_legend(ax, loc="upper right")
    ax.axhspan(1.7, 1.9, color="0.5")
    ax.axhline(1.95)
    canvas.draw()
    assert np.allclose(ax.get_ylim(), (0.0, 2.0), rtol=1e-12, atol=0.0)


def test_set_legend_keeps_a_top_of_the_user() -> None:
    """A -ymax of the user stays, although the legend then covers the data."""
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 1.0], label="flat")
    ax.set_ylim(0.0, 1.02)
    at.plottools.set_legend(ax, argparse.Namespace(ymax=1.02), loc="upper right")
    fig.canvas.draw()
    assert np.isclose(ax.get_ylim()[1], 1.02, rtol=1e-12, atol=0.0)
    plt.close(fig)


@pytest.mark.parametrize(("labelwidth", "legendcols"), [(4, None), (60, None), (4, 1)])
def test_set_legend_takes_columns_for_a_long_legend(labelwidth: int, legendcols: int | None) -> None:
    """A long legend takes more columns, until it has half the frame height or it would be wider than the frame.

    Only one legend goes on the axes, thus a trial legend of the rule must stay off them. -legendcols overrides the
    rule.
    """
    fig = mplfig.Figure()
    canvas = FigureCanvasAgg(fig)
    _, axesgrid = at.plottools.make_frame_figure(fig=fig)
    ax = axesgrid[0][0]
    for index in range(16):
        ax.plot([0.0, 1.0], [index, index], label=f"{index}".ljust(labelwidth, "x"))
    # an ncol of None asks for the rule, as plotspectra and plotestimators give for a plot of one column
    legend = at.plottools.set_legend(ax, argparse.Namespace(legendcols=legendcols), loc="upper right", ncol=None)
    assert legend is not None
    assert ax.get_legend() is legend
    assert [child for child in ax.get_children() if isinstance(child, mpllegend.Legend)] == [legend]
    # the draw gives each text of the legend its position
    canvas.draw()
    renderer = canvas.get_renderer()
    frame = at.plottools.get_legend_frame(legend, renderer)
    # each column of the legend starts its labels at one x position
    ncols = len({round(text.get_window_extent(renderer).x0) for text in legend.get_texts()})
    if legendcols is not None:
        assert ncols == legendcols
    elif labelwidth > 10:
        # a wide label leaves no room for a second column
        assert frame.width <= 1.0
        assert ncols == 1
    else:
        assert frame.height <= at.plottools.MAX_LEGEND_HEIGHT_FRACTION
        assert ncols > 1


def test_set_legend_above_the_frame_keeps_the_y_limits() -> None:
    """--legend-above puts the legend on the top edge of the frame, and the y axis keeps its range.

    A legend inside the frame moves the top of the y axis to clear the data. A legend above the frame covers no data,
    thus the limits of the data stay.
    """
    fig = mplfig.Figure()
    canvas = FigureCanvasAgg(fig)
    _, axesgrid = at.plottools.make_frame_figure(fig=fig)
    ax = axesgrid[0][0]
    for index in range(6):
        ax.plot([0.0, 1.0], [index, index], label=f"series {index}")
    ax.set_ylim(0.0, 5.0)
    legend = at.plottools.set_legend(ax, argparse.Namespace(legend_above=True, legendcols=2), loc="upper right")
    assert legend is not None
    canvas.draw()
    frame = at.plottools.get_legend_frame(legend, canvas.get_renderer())
    assert frame.y0 > 1.0 or np.isclose(frame.y0, 1.0)
    assert np.allclose(ax.get_ylim(), (0.0, 5.0), rtol=1e-12, atol=0.0)
    plt.close(fig)


def test_get_series_colors_greys_then_cycle() -> None:
    """More reference series than greys must fall back to the colour cycle instead of an IndexError."""
    colors = at.plottools.get_series_colors([False, True, True, False, True, True, True, True])

    assert colors == ["C0", "0.0", "0.4", "C1", "0.6", "0.7", "C2", "C3"]


def test_get_series_colors_keeps_the_colours_of_the_user() -> None:
    """A colour of the user has priority, and no other series gets that colour."""
    assert at.plottools.get_series_colors([False, False, True], ["C1"]) == ["C1", "C0", "0.0"]
    assert at.plottools.get_series_colors([False, True], [None, "red"]) == ["C0", "red"]

    # a grey that the user asked for goes out of the sequence of the reference series
    assert at.plottools.get_series_colors([True, True], ["0.0"]) == ["0.0", "0.4"]
    assert at.plottools.get_series_colors([True, True], [None, "0.0"]) == ["0.4", "0.0"]
    assert at.plottools.get_series_colors([False, True], ["0.0"]) == ["0.0", "0.4"]

    # a colour that is not a grey does not take a grey of the sequence
    assert at.plottools.get_series_colors([True, True], ["red"]) == ["red", "0.0"]


def test_get_series_colors_knows_the_value_of_a_cycle_colour() -> None:
    """A colour value of the cycle that the user asked for must go out of the cycle, like the name CN."""
    at.plottools.set_mpl_style()
    cyclecolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    assert at.plottools.get_series_colors([False, False], [cyclecolors[0]]) == [cyclecolors[0], "C1"]
    assert at.plottools.get_series_colors([False, False], [cyclecolors[1]]) == [cyclecolors[1], "C0"]


def test_get_series_colors_matches_a_cycle_colour_by_any_spelling() -> None:
    """A cycle colour must leave the cycle whatever name the user gave it, not only its exact hex string."""
    at.plottools.set_mpl_style()
    firstcyclecolor = mplcolors.to_hex(plt.rcParams["axes.prop_cycle"].by_key()["color"][0])
    assert firstcyclecolor == mplcolors.to_hex("tab:blue")

    for spelling in ("tab:blue", firstcyclecolor.upper()):
        colors = at.plottools.get_series_colors([False, False], [spelling])
        assert colors[0] == spelling
        # the second series used to be handed C0, drawing both series in the same blue
        assert mplcolors.to_hex(colors[1]) != firstcyclecolor

    # a grey that the user asked for is also matched by value, so the next reference series steps over it
    assert at.plottools.get_series_colors([True, True], ["#000000"]) == ["#000000", "0.4"]


def test_prune_log_ticks_drops_only_the_ticks_against_each_end() -> None:
    """A tick at the very top or bottom of a log axis goes, and every decade inside it stays."""
    _fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1e-10, 1e3)
    at.plottools.prune_log_ticks(ax.yaxis)

    after = [loc for loc in ax.yaxis.get_majorticklocs() if 1e-10 <= loc <= 1e3]
    assert after == pytest.approx([10.0**exponent for exponent in range(-9, 3)])


def test_log_ticks_every_decade_on_a_short_axis() -> None:
    """A short log axis of many decades labels each power of ten, and it has minor ticks between them.

    The default locator counted the decades that fit the axis length, thus the opacity frame labelled
    every second decade and the ratio panel below it had no minor ticks.
    """
    fig, ax = plt.subplots(figsize=(3.0, 0.7))
    ax.set_yscale("log")
    ax.set_ylim(0.4, 2e8)
    fig.canvas.draw()
    assert len([loc for loc in ax.yaxis.get_majorticklocs() if 0.4 <= loc <= 2e8]) < 9

    at.plottools.set_log_ticks_every_decade(ax.yaxis)
    fig.canvas.draw()
    majors = [loc for loc in ax.yaxis.get_majorticklocs() if 0.4 <= loc <= 2e8]
    assert majors == pytest.approx([10.0**exponent for exponent in range(9)])
    minors = [loc for loc in ax.yaxis.get_minorticklocs() if 1.0 <= loc <= 10.0]
    assert minors == pytest.approx([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    # a range below one decade holds no power of ten, thus the minor ticks carry the labels
    ax.set_ylim(1.2, 8.0)
    fig.canvas.draw()
    assert [label.get_text() for label in ax.yaxis.get_minorticklabels() if label.get_text()]


def test_prune_log_ticks_keeps_a_sparse_axis_unchanged() -> None:
    """A log axis of few major ticks keeps them all, rather than end with too few to read."""
    from artistools.plottools import PrunedLogLocator

    locator = PrunedLogLocator(minticks=99, numticks=999)
    assert list(locator.tick_values(1e-30, 1e2)) == list(mplticker.LogLocator(numticks=999).tick_values(1e-30, 1e2))


def test_prune_log_ticks_follows_the_view() -> None:
    """The locator prunes when matplotlib draws, thus a zoom gives the ticks of the new view.

    A FixedLocator of one view would keep those ticks through a zoom, and save_figure shows the figure
    before it writes the file.
    """
    _fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e5)
    at.plottools.prune_log_ticks(ax.yaxis)
    first = list(ax.yaxis.get_majorticklocs())

    ax.set_ylim(1e2, 1e9)
    second = list(ax.yaxis.get_majorticklocs())

    assert second != first, "the ticks must follow the new view"
    assert max(second) > max(first), f"the zoomed view needs its own high ticks: {second}"


def test_set_axis_properties_log_scale_keeps_the_data_in_view() -> None:
    """A log scale must be set before the limits, so an unrequested limit does not freeze the linear view.

    set_ylim turns autoscaling off even when both sides are None, so applying it first left the log axis with
    the linearly padded limits, whose lower bound is negative and therefore off the axis entirely.
    """
    _fig, ax = plt.subplots()
    ax.plot([1.0, 10.0], [1.0, 1000.0])

    at.plottools.set_axis_properties(ax, argparse.Namespace(logscaley=True, ymin=None, ymax=None))

    ymin, ymax = ax.get_ylim()
    assert ymin > 0.0, ymin
    assert ymin < 1.0, ymin
    assert ymax > 1000.0, ymax


def test_set_axis_properties_applies_the_requested_limits() -> None:
    """A limit that the user did give must still be applied, on either side and on either axis."""
    _fig, ax = plt.subplots()
    ax.plot([1.0, 10.0], [1.0, 1000.0])

    at.plottools.set_axis_properties(ax, argparse.Namespace(ymin=None, ymax=500.0, xmin=2.0, xmax=None))

    assert np.isclose(ax.get_ylim()[1], 500.0)
    assert np.isclose(ax.get_xlim()[0], 2.0)
    # the side that was not given stays fitted to the data rather than falling back to the default view
    assert ax.get_ylim()[0] <= 1.0
    assert ax.get_xlim()[1] >= 10.0


def test_iter_axes_flattens_a_subplot_grid() -> None:
    """One axes and a grid of axes both become a flat list, so the callers need no isinstance dance."""
    _fig, singleax = plt.subplots()
    assert at.plottools.iter_axes(singleax) == [singleax]

    _fig, axes = plt.subplots(nrows=2, ncols=3)
    assert at.plottools.iter_axes(axes.flatten()) == list(axes.flatten())

    # iterating a 2D array yields its rows, which are arrays rather than axes
    assert at.plottools.iter_axes(axes) == list(axes.flatten())


def test_room_for_title_predicts_the_place_of_the_title_of_the_draw() -> None:
    """The height for the title comes from a prediction with no draw, and a draw must then put the title there.

    A draw took 83 ms of a plot of 250 ms. The draw moves a title that overlaps the offset text of the y axis, thus a
    title with and without an offset text must end 0.05 inches below the top of the figure. The text of the draw
    can end one pixel from the prediction.
    """
    for ymax in (2e-5, 20.0):
        fig, axes = at.plottools.make_frame_figure(argparse.Namespace(figwidthscale=0.3, figscale=1.0))
        axis = axes[0, 0]
        axis.plot([0.0, 1.0], [ymax / 2, ymax])
        axis.set_title("A model name\nTimestep 40 (10.00d-11.00d)")
        at.plottools.make_room_for_title(fig)
        fig.canvas.draw()
        assert bool(axis.yaxis.offsetText.get_text()) == (ymax < 1.0)
        gap = fig.get_figheight() - axis.title.get_window_extent().y1 / fig.dpi
        assert np.isclose(gap, 0.05, rtol=0.0, atol=0.01), (ymax, gap)
        plt.close(fig)


def test_room_for_title_keeps_the_axes_of_a_figure_with_no_frames() -> None:
    """Only a frame figure takes more height for its title, because its divider keeps the frames at the bottom.

    The axes of a different figure grew with the new height, and the title still went past the top.
    """
    fig, axis = plt.subplots(figsize=(4.0, 3.0))
    axis.set_title("line 1\nline 2\nline 3\nline 4")
    at.plottools.make_room_for_title(fig)
    assert np.isclose(fig.get_figheight(), 3.0, rtol=1e-12, atol=0.0)
    plt.close(fig)


def test_plain_label_and_saved_path_read_well_in_a_terminal() -> None:
    """A log line must carry no LaTeX, and it must give the shorter of the two forms of a path."""
    assert at.plottools.plain_label(r"TEST MODEL +300.3d ($\pm$ 0.5d)") == "TEST MODEL +300.3d (+/- 0.5d)"
    # the subscript mark goes and the underscore stays, thus the plain form reads as M_sun
    assert at.plottools.plain_label(r"M$_{\odot}$") == "M_sun"
    assert at.plottools.plain_label(r"T$_{\rm e}$ [K]") == "T_e [K]"
    assert at.plottools.plain_label("no mathematics here") == "no mathematics here"


def test_hidden_value_warning_names_a_horizontal_line_above_the_axes(capsys: pytest.CaptureFixture[str]) -> None:
    """An axhline gives x from 0 to 1 in the coordinates of the axes, and the x range of the data left it out."""
    axis = mplfig.Figure().add_subplot()
    axis.axhline(5.0, label="Planck mean")
    axis.set_xlim(1000.0, 2000.0)
    axis.set_ylim(0.0, 1.0)
    at.plottools.warn_hidden_values(axis)
    assert "The figure does not show these values of 'Planck mean'" in capsys.readouterr().err


def test_viewer_dark_colours_keep_the_colours_of_the_series() -> None:
    """In Dark Mode, a black line and a black text take the colour of the text, and a coloured series keeps its colour.

    A black line on the dark background of the window cannot show, and a series needs its colour for the legend. A
    dark colour of an element, e.g. of O, Si, or S, is a colour of a series too.
    """
    import matplotlib.colors as mcolors
    import matplotlib.figure as mplfig

    fig = mplfig.Figure()
    axis = fig.add_subplot()
    blackline = axis.plot([0, 1], [0, 1], color="black", label="model")[0]
    blueline = axis.plot([0, 1], [1, 0], color="tab:blue", label="reference")[0]
    # the colour of sulphur is dark, and it made the series grey
    sulphurline = axis.plot([0, 1], [0.5, 0.5], color="#7d0200", label="S")[0]
    # a black bar did not show on the dark background, because only the edge of a patch changed
    blackbar = axis.bar([0.5], [0.2], color="black")[0]
    # plotnltepops gives a hollow marker a black edge, and the marker did not show
    hollowmarkers = axis.plot([0.2], [0.8], "s", color="tab:blue", markeredgecolor="black", markerfacecolor="none")[0]
    # plotopacity draws the Planck mean in a grey of 0.3, and it stayed dark
    greyline = axis.plot([0, 1], [0.3, 0.3], color="0.3")[0]
    midgreyline = axis.plot([0, 1], [0.6, 0.6], color="0.5")[0]
    axis.set_xlabel("velocity")
    legend = axis.legend()

    at.plottools.apply_dark_colours(fig, "#1e1e1e", "#dddddd")

    assert mcolors.same_color(blackline.get_color(), "#dddddd")
    assert mcolors.same_color(blueline.get_color(), "tab:blue")
    assert mcolors.same_color(sulphurline.get_color(), "#7d0200")
    assert mcolors.same_color(axis.xaxis.label.get_color(), "#dddddd")
    assert mcolors.same_color(axis.get_facecolor(), "#1e1e1e")
    assert mcolors.same_color(blackbar.get_facecolor(), "#dddddd")
    assert mcolors.same_color(hollowmarkers.get_markeredgecolor(), "#dddddd")
    assert mcolors.same_color(hollowmarkers.get_color(), "tab:blue")
    assert min(mcolors.to_rgb(greyline.get_color())) > 0.6
    assert mcolors.same_color(midgreyline.get_color(), "0.5")
    # the frame of the legend keeps its transparency
    assert mcolors.to_hex(legend.get_frame().get_facecolor()) == "#1e1e1e"
    assert all(mcolors.same_color(text.get_color(), "#dddddd") for text in legend.get_texts())


def test_hidden_values_of_a_line_give_a_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """A y limit or a log axis can hide values of a series, thus the function gives a warning for these values."""
    from artistools.plottools import warn_hidden_values

    fig = mplfig.Figure()
    axis = fig.add_subplot()
    axis.plot([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 10.0, 100.0], label="series")
    # a line with no label is a part of a different series, e.g. a repeated end point
    axis.plot([1.0, 2.0], [1000.0, 1000.0])
    axis.set_yscale("log")
    axis.set_ylim(0.5, 50.0)
    warn_hidden_values(axis)
    err = capsys.readouterr().err
    assert "values above the top of the axis (50): 1" in err
    assert "values of zero or below on the log axis: 1" in err
    assert "below the bottom" not in err
    assert err.count("does not show") == 1


def test_packet_summaries_give_the_monte_carlo_noise(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """A number from few packets is noisy, thus the summary of a packet plot gives sqrt(Σ e²) / Σ e and a warning.

    ARTIS gives the packets of an r-packet spectrum almost the same energy, thus the noise is close to 1 / sqrt(N).
    """
    at.spectra.plot(argsraw=[], specpath=[modelpath], timedays="300", frompackets=True, outputfile=tmp_path / "s.pdf")
    captured = capsys.readouterr()
    match = re.search(
        r"(\d+) packets in the drawn bins, relative noise of the luminosity ([\d.]+)% = sqrt", captured.out
    )
    assert match is not None, captured.out
    assert np.isclose(float(match.group(2)) / 100, 1 / math.sqrt(int(match.group(1))), rtol=0.05)
    assert "the luminosity has a relative Monte Carlo noise of" in captured.err

    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], frompackets=True, outputfile=tmp_path)
    captured = capsys.readouterr()
    match = re.search(r"noise of the peak point ([\d.]+)% = sqrt\(Σ e²\) / Σ e of its (\d+) packets", captured.out)
    assert match is not None, captured.out
    assert np.isclose(float(match.group(1)) / 100, 1 / math.sqrt(int(match.group(2))), rtol=0.05)
    assert "the peak point has a relative Monte Carlo noise of" in captured.err

    # a filter combines the packets of neighbouring timesteps, and the line gave the noise of one unfiltered timestep
    at.lightcurve.plot(argsraw=["-filtermovingavg", "3"], modelpath=[modelpath], frompackets=True, outputfile=tmp_path)
    captured = capsys.readouterr()
    assert "noise of the peak point" not in captured.out
    assert "no noise estimate, because the filter combines the packets" in captured.out

    # a direction bin holds about 1/100 of the packets, and the files of the direction bins give no packet count
    at.lightcurve.plot(argsraw=["-plotviewingangle", "0"], modelpath=[modelpath_classic_3d], outputfile=tmp_path)
    assert "holds about 1/100 of the escaped packets" in capsys.readouterr().err
    at.spectra.plot(
        argsraw=["-plotviewingangle", "0", "--average_over_phi_angle"],
        specpath=[modelpath_classic_3d],
        timedays="5",
        outputfile=tmp_path / "s.pdf",
    )
    assert "Each direction bin of spec_res.out holds about 1/10 of the escaped packets" in capsys.readouterr().err

    # spec.out has no packet count, thus the line gives no noise
    at.spectra.plot(argsraw=[], specpath=[modelpath], timedays="300", outputfile=tmp_path / "s.pdf")
    assert "packets in the drawn bins" not in capsys.readouterr().out


def test_plot_summaries_give_the_statistics_of_each_series(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """An agent reads the summary in the log, thus each line gives the values that a question about the plot needs.

    In the 3D test model at 5 days, 258 cells are at MAXTEMP, which is a limit of ARTIS and not a physical value.
    """
    at.estimators.plot(argsraw=[], modelpath=modelpath_classic_3d, outputfile=tmp_path, timedays=5, plotlist=[["Te"]])
    captured = capsys.readouterr()
    assert re.search(r"  Te: subplot 0, 540 points, .*min 3500 K at x = .*max 1\.4e\+05 K", captured.out)
    assert "Te: 258 points are at MAXTEMP" in captured.err
    # 155 cells of the 695 with matter are in the corners of the cube, and the mean of the summary leaves them out
    assert "The cells whose centre has √(x_c² + y_c² + z_c²) ≤ v_max = 0.09647c" in captured.out
    assert "Te: 44 points are at MINTEMP" in captured.err

    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], filter=["B"], timemin=250, timemax=300, outputfile=tmp_path)
    assert re.search(
        r"B TEST MODEL: 54 points .*, peak -13\.1\d mag at 267\.9 d \(brightest point in the time range\),"
        r" Δm15 0\.2\d+ mag \(linear interpolation",
        capsys.readouterr().out,
    )

    # the frame keeps the nearest point outside -timemin, and the summary counted that point
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], timemin=250.5, timemax=299.0, outputfile=tmp_path)
    summarymatch = re.search(r"TEST MODEL: \d+ points from ([\d.]+) to", capsys.readouterr().out)
    assert summarymatch is not None
    assert float(summarymatch.group(1)) >= 250.5

    # a magnitude has no comoving-frame column, and the check read the faintest magnitude as the maximum
    at.lightcurve.plot(argsraw=[], modelpath=[modelpath], magnitude=True, outputfile=tmp_path)
    assert "comoving-frame" not in capsys.readouterr().err

    # a reference light curve gave no summary
    at.lightcurve.plot(argsraw=[], modelpath=["AT2017gfo_waxmanetal2018.txt"], outputfile=tmp_path)
    assert re.search(r"AT2017gfo \(Waxman\+2018\): \d+ points .*, peak ", capsys.readouterr().out)

    # the axis does not show the points before -timemin, and the summary gave the peak at 0.6 d
    at.lightcurve.plot(argsraw=[], modelpath=["AT2017gfo_waxmanetal2018.txt"], timemin=1.0, outputfile=tmp_path)
    assert re.search(
        r"AT2017gfo \(Waxman\+2018\): \d+ points from 1 to .*, peak 6\.15e\+41 erg/s at 1 d", capsys.readouterr().out
    )

    # the plot draws no point where the series has no value or an infinite value, and the x range counted such a point
    from artistools.estimators.plotestimators import get_series_summary_query

    dfseries = pl.LazyFrame({
        "xvalue": [1.0, 2.0, 3.0, 4.0, 5.0],
        "xvalue_binned": [1.0, 2.0, 3.0, 4.0, 5.0],
        "yvalue": [None, 5.0, 6.0, float("nan"), float("inf")],
        "celltsweight": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    summary = get_series_summary_query(dfseries, "cell volume", {}).collect().row(0, named=True)
    assert (summary["xmin"], summary["xmax"], summary["valuecount"], summary["emptybincount"]) == (2.0, 3.0, 2, 3)
    assert np.isclose(summary["ymax"], 6.0)

    # the luminosity is 4 pi (1 Mpc)^2 times the integrated flux at 1 Mpc of the drawn line, which reaches each axis
    # limit. The integral left out the segments between each axis limit and the nearest row inside it
    dfspec = at.spectra.scan_spectra(modelpath, timestepmin=54, timestepmax=54)[-1].collect().sort("lambda_angstroms")
    lambdas, flambdas = dfspec["lambda_angstroms"].to_numpy(), dfspec["f_lambda"].to_numpy()

    def get_expected_luminosity(lambdamin: float, lambdamax: float) -> float:
        isinside = (lambdas > lambdamin) & (lambdas < lambdamax)
        xpoints = np.concatenate([[lambdamin], lambdas[isinside], [lambdamax]])
        ypoints = np.interp(xpoints, lambdas, flambdas)
        return float(np.trapezoid(ypoints, x=xpoints)) * 4 * math.pi * at.constants.megaparsec_to_cm**2

    # a range between two rows has no row inside it, and the summary gave no luminosity for the drawn line
    for xlimits in [(2600.0, 18000.0), (4000.5, 4001.0)]:
        at.spectra.plot(
            argsraw=[],
            specpath=[modelpath],
            timedays="300",
            xmin=xlimits[0],
            xmax=xlimits[1],
            outputfile=tmp_path / "spec.pdf",
        )
        out = capsys.readouterr().out
        spectrummatch = re.search(
            r"TEST MODEL [^:]*: (?:\d+ points from .* angstrom, largest drawn flux [\d.e+-]+ at [\d.e+-]+ angstrom, )?"
            r"luminosity ([\d.e+-]+) erg/s = 4π \(1 Mpc\)² ∫ f_λ dλ from ([\d.e+-]+) to [\d.e+-]+ Å \(trapezoidal",
            out,
        )
        assert spectrummatch is not None, out
        assert np.isclose(float(spectrummatch.group(2)), xlimits[0], rtol=1e-3)
        assert np.isclose(float(spectrummatch.group(1)), get_expected_luminosity(*xlimits), rtol=1e-3)

    # --showemission gives one line for each drawn series with its share, and none for a series with no flux
    at.spectra.plot(
        argsraw=[], specpath=[modelpath_classic_3d], timedays="5", showemission=True, outputfile=tmp_path / "em.pdf"
    )
    out = capsys.readouterr().out
    shares = [float(share) for share in re.findall(r"^    \S.*? emission [\d.e+-]+ \( *([\d.]+)%\)", out, re.MULTILINE)]
    assert shares[:2] == [68.3, 21.6]
    assert 99.0 < sum(shares) <= 100.0
    assert "absorption 0.0" not in out
    assert "drawn series have no emission and no absorption" in out
