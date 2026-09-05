import os
import re
import shutil
import typing as t
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")

PLOTLIST_FULL: t.Final = (
    [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
    ["nne"],
    ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
    ["Te"],
    [["averageionisation", ["Fe", "Ni"]]],
    [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
    [["populations", ["Co II", "Co III", "Co IV"]]],
    [["gamma_NT", ["Fe I", "Fe II", "Fe III", "Fe IV"]]],
    ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
    ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
    [(pl.col("heating_coll") - pl.col("cooling_coll")).alias("collisional heating - cooling")],
)

PLOTLIST_IONS: t.Final = (
    [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
    ["nne"],
    ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
    ["Te"],
    [["averageionisation", ["Fe"]]],
    [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
    [["populations", ["Co II", "Co III", "Co IV"]]],
    ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
    ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_ymin_lets_the_other_side_follow_the_data(mockplot: mock.MagicMock) -> None:
    """A _ymin of the plot list must not freeze the top of the axis far above the data.

    set_ylim turns the autoscaling of the whole axis off, thus applying _ymin before the series were
    drawn left the top at the value it held with no data on the axes.
    """
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath,
        timedays=260,
        plotlist=[["rho", ["_yscale", "log"], ["_ymin", 1e-18]]],
    )

    ydata = np.concatenate([
        np.asarray(callargs[0][2], dtype=float) for callargs in mockplot.call_args_list if len(callargs[0]) > 2
    ])
    ydata = ydata[np.isfinite(ydata)]
    ax = mockplot.call_args_list[0][0][0]
    ylo, yhi = ax.get_ylim()

    assert ylo == pytest.approx(1e-18), "the requested floor must be applied"
    assert yhi < ydata.max() * 100, f"the top {yhi:.2e} is far above the data maximum {ydata.max():.2e}"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_ymin_does_not_hide_the_whole_series(
    mockplot: mock.MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fixed y limit of the plot list must give way when no data point would stay in view.

    The default plot list floors rho at 1e-16, and the density of the test model is below that, thus the
    panel drew nothing at all.
    """
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath,
        timedays=260,
        plotlist=[["rho", ["_yscale", "log"], ["_ymin", 1e-16]]],
    )

    ydata = np.concatenate([
        np.asarray(callargs[0][2], dtype=float) for callargs in mockplot.call_args_list if len(callargs[0]) > 2
    ])
    ydata = ydata[np.isfinite(ydata)]
    assert ydata.size > 0
    assert ydata.max() < 1e-16, "the test needs a model of a density below the floor of the plot list"

    assert "below the requested minimum" in capsys.readouterr().err

    # the floor must never be applied, because set_ylim accepts a bottom above the top and then turns
    # the axis upside down, and a later autoscale keeps that direction
    assert not mockplot.call_args_list[0][0][0].yaxis_inverted(), "the panel is drawn upside down"

    # the axes that the mock recorded must show the data, and not the empty range below the floor
    ax = mockplot.call_args_list[0][0][0]
    ylo, yhi = ax.get_ylim()
    assert ylo <= ydata.min(), f"the lowest point {ydata.min():.2e} is below the axis at {ylo:.2e}"
    assert ydata.max() <= yhi, f"the highest point {ydata.max():.2e} is above the axis at {yhi:.2e}"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_estimator_snapshot(mockplot: mock.MagicMock) -> None:
    plotlist = PLOTLIST_FULL

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot",
        timedays=300,
    )
    xarr = [0.0, 4000.0]
    for x in mockplot.call_args_list:
        assert np.allclose(xarr, x[0][1], rtol=1e-3, atol=1e-3)

    # order of keys is important
    expectedvals = {
        "init_fe": 0.10000000149011612,
        "init_nistable": 0.0,
        "init_ni56": 0.8999999761581421,
        "nne": 794211.0,
        "TR": 6932.45,
        "Te": 5776.620000000001,
        "averageionisation_Fe": 1.9453616269532485,
        "averageionisation_Ni": 1.970637712188408,
        "populations_FeI": 4.801001667392128e-05,
        "populations_FeII": 0.350781150587666,
        "populations_FeIII": 0.3951266859004141,
        "populations_FeIV": 0.21184950941623004,
        "populations_FeV": 0.042194644079016,
        "populations_CoII": 0.10471832570699871,
        "populations_CoIII": 0.476333358337709,
        "populations_CoIV": 0.41894831595529214,
        "gamma_NT_FeI": 7.571e-06,
        "gamma_NT_FeII": 3.711e-06,
        "gamma_NT_FeIII": 2.762e-06,
        "gamma_NT_FeIV": 1.702e-06,
        "heating_dep": 6.56117e-10,
        "heating_coll": 2.37823e-09,
        "heating_bf": 1.27067e-13,
        "heating_ff": 1.86474e-16,
        "cooling_adiabatic": 9.72392e-13,
        "cooling_coll": 3.02786e-09,
        "cooling_fb": 4.82714e-12,
        "cooling_ff": 1.62999e-13,
        "collisional heating - cooling": -6.4962990e-10,
    }
    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {
        varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list, strict=False)
    }

    print({key: yarr[1] for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose([expectedval, expectedval], yvals[varname], rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_estimator_averaging(mockplot: mock.MagicMock) -> None:
    plotlist = PLOTLIST_FULL

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_averaging",
        timestep="50-54",
    )

    xarr = [0.0, 4000.0]
    for x in mockplot.call_args_list:
        assert np.allclose(xarr, x[0][1], rtol=1e-3, atol=1e-3)

    # order of keys is important
    expectedvals = {
        "init_fe": 0.10000000149011612,
        "init_nistable": 0.0,
        "init_ni56": 0.8999999761581421,
        "nne": 811131.8125,
        "TR": 6932.65771484375,
        "Te": 5784.4521484375,
        "averageionisation_Fe": 1.9466091928476605,
        "averageionisation_Ni": 1.9673294753348698,
        "populations_FeI": 4.668364835386799e-05,
        "populations_FeII": 0.35026945954378863,
        "populations_FeIII": 0.39508678896764393,
        "populations_FeIV": 0.21220745115264195,
        "populations_FeV": 0.042389615364484115,
        "populations_CoII": 0.1044248111887582,
        "populations_CoIII": 0.4759472294613869,
        "populations_CoIV": 0.419627959349855,
        "gamma_NT_FeI": 7.741022037400234e-06,
        "gamma_NT_FeII": 3.7947153292832773e-06,
        "gamma_NT_FeIII": 2.824587987164586e-06,
        "gamma_NT_FeIV": 1.7406694591346083e-06,
        "heating_dep": 6.849705802558503e-10,
        "heating_coll": 2.4779998053503505e-09,
        "heating_bf": 1.2916119454357833e-13,
        "heating_ff": 2.1250019797070045e-16,
        "cooling_adiabatic": 1.000458830363593e-12,
        "cooling_coll": 3.1562059632506134e-09,
        "cooling_fb": 5.0357105638165756e-12,
        "cooling_ff": 1.7027620090835638e-13,
        "collisional heating - cooling": -6.782059913668093e-10,
    }
    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {
        varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list, strict=False)
    }

    print({key: yarr[1] for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose([expectedval, expectedval], yvals[varname], rtol=0.001, equal_nan=True)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_snapshot_classic_3d(mockplot: mock.MagicMock) -> None:
    plotlist = PLOTLIST_IONS

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        markers=True,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot_classic_3d.pdf",
        timedays=4,
    )

    # order of keys is important
    # the values changed when the x binning stopped the drop of the cell at the minimum x value
    expected_yvals_mean = {
        "init_fe": 0.015814311802387238,
        "init_nistable": 0.00967155396938324,
        "init_ni56": 0.051117077469825745,
        "nne": 14852512768.0,
        "TR": 19079.78515625,
        "Te": 71268.359375,
        "averageionisation_Fe": 3.055755138397217,
        "populations_FeI": 5.362183365449527e-16,
        "populations_FeII": 0.00019352462550159544,
        "populations_FeIII": 0.06814975291490555,
        "populations_FeIV": 0.8073020577430725,
        "populations_FeV": 0.12433895468711853,
        "populations_CoII": 0.16990603506565094,
        "populations_CoIII": 0.2491680532693863,
        "populations_CoIV": 0.5809222459793091,
        "heating_dep": 2.559114818723174e-06,
        "heating_coll": 0.0002118289703503251,
        "heating_bf": 2.1746404854638968e-06,
        "heating_ff": 5.587692530895083e-10,
        "cooling_adiabatic": 1.2879886046590627e-10,
        "cooling_coll": 4.351998359197751e-05,
        "cooling_fb": 9.605032857962215e-08,
        "cooling_ff": 6.669354513100245e-10,
    }

    expected_yvals_std = {
        "init_fe": 0.03864092379808426,
        "init_nistable": 0.02453182451426983,
        "init_ni56": 0.13668806850910187,
        "nne": 54105763840.0,
        "TR": 8786.4306640625,
        "Te": 53253.1875,
        "averageionisation_Fe": 0.36672845482826233,
        "populations_FeI": 1.0549461627071093e-14,
        "populations_FeII": 0.0039639282040297985,
        "populations_FeIII": 0.2204248011112213,
        "populations_FeIV": 0.31659460067749023,
        "populations_FeV": 0.261174738407135,
        "populations_CoII": 0.36840304732322693,
        "populations_CoIII": 0.38466954231262207,
        "populations_CoIV": 0.457830548286438,
        "heating_dep": 2.440772732370533e-05,
        "heating_coll": 0.004782144911587238,
        "heating_bf": 4.8423054977320135e-05,
        "heating_ff": 3.5524865271696626e-09,
        "cooling_adiabatic": 1.214427669538054e-09,
        "cooling_coll": 0.0009417609544470906,
        "cooling_fb": 2.126975687133381e-06,
        "cooling_ff": 7.287984438164585e-09,
    }

    plot_calls_markers = mockplot.call_args_list[1::2]
    assert len(expected_yvals_mean) == len(plot_calls_markers)

    yvals_mean = {
        varname: float(np.array(callargs[0][2]).mean())
        for varname, callargs in zip(expected_yvals_mean.keys(), plot_calls_markers, strict=True)
    }
    print(f"{yvals_mean=}")

    yvals_std = {
        varname: float(np.array(callargs[0][2]).std())
        for varname, callargs in zip(expected_yvals_std.keys(), plot_calls_markers, strict=True)
    }
    print(f"{yvals_std=}")

    for varname, expectedmean in expected_yvals_mean.items():
        assert np.isclose(expectedmean, yvals_mean[varname], rtol=0.01), (varname, expectedmean, yvals_mean[varname])
    for varname, expectedstd in expected_yvals_std.items():
        assert np.isclose(expectedstd, yvals_std[varname], rtol=0.01), (varname, expectedstd, yvals_std[varname])


def test_xbins_gives_the_number_of_bins() -> None:
    """-xbins N divides the x range into N bins. It gave N edges, thus N - 1 bins, before."""
    for xbins in (5, 10):
        xvalues, xlimits = get_binned_xvalues_and_limits(xbins)

        # the middle points of two bins beside each other lie one width apart, and N bins divide the
        # range of the axis. An empty bin gives a gap of two widths, and each end gives half of one
        expectedwidth = (xlimits[1] - xlimits[0]) / xbins
        widths = np.diff(xvalues)
        assert np.isclose(widths, expectedwidth).any(), xbins
        assert np.isclose(widths[-1], expectedwidth / 2.0), xbins


def get_binned_xvalues_and_limits(xbins: int) -> tuple[npt.NDArray[np.float64], tuple[float, float]]:
    """Return the x values that one plot of binned estimators draws, and the limits of its x axis."""
    drawn: list[npt.NDArray[np.float64]] = []
    limits: list[tuple[float, float]] = []
    realplot, realxlim = mplax.Axes.plot, mplax.Axes.set_xlim

    def spyplot(self: mplax.Axes, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if len(args) >= 2 and np.ndim(args[0]) > 0:
            drawn.append(np.asarray(args[0], dtype=np.float64))
        return realplot(self, *args, **kwargs)

    def spyxlim(self: mplax.Axes, *args: t.Any, **kwargs: t.Any) -> t.Any:
        limits.append((float(args[0]), float(args[1])))
        return realxlim(self, *args, **kwargs)

    with mock.patch.object(mplax.Axes, "plot", spyplot), mock.patch.object(mplax.Axes, "set_xlim", spyxlim):
        at.estimators.plot(
            argsraw=[],
            modelpath=modelpath_classic_3d,
            plotlist=[["Te"]],
            timedays=4,
            xbins=xbins,
            outputfile=outputpath / f"test_xbins_{xbins}.pdf",
        )

    return drawn[0], limits[0]


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_a_binned_line_reaches_the_edge_of_the_data(mockplot: mock.MagicMock) -> None:
    """A line of binned values reaches both edges of the data, and not the middle of the outer bins.

    The line ran through the middle of each bin, thus it stopped half a bin short of the highest x
    value that the plot shows. A model of many cells gave a gap of one part in fifty of the axis.
    """
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        plotlist=[["Te"]],
        timedays=4,
        xbins=10,
        outputfile=outputpath / "test_binned_line_reaches_the_edge.pdf",
    )

    xvalues = np.array(mockplot.call_args_list[0].args[1], dtype=float)
    assert xvalues.size > 2

    # the velocity of the x axis starts the line at zero, thus the low end reaches past the data
    assert xvalues[0] == 0.0

    # every bin has the same width, thus the last two points differ by half of it
    halfwidth = (xvalues[-2] - xvalues[-3]) / 2.0
    assert xvalues[-1] == pytest.approx(xvalues[-2] + halfwidth)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_snapshot_classic_3d_x_axis(mockplot: mock.MagicMock) -> None:
    plotlist = PLOTLIST_IONS

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot_classic_3d_x_axis.pdf",
        timedays=4,
        readonlymgi="alongaxis",
        axis="+x",
    )

    # order of keys is important
    # the values changed when select_cells_along_axis started to give the estimators the zero-based
    # modelgridindex. The one-based inputcellid selected the neighbour of each intended cell before.
    # They changed again when the axis profile started to keep the innermost cell, whose pos_min is
    # zero on the positive axis. The old filter pos_min > 0 dropped that cell from every mean.
    expectedvals = {
        "init_fe": 0.030691306495330833,
        "init_nistable": 0.03129141867854237,
        "init_ni56": 0.3625203213639654,
        "nne": 138785511355.11838,
        "TR": 32679.133463541668,
        "Te": 49283.68359375,
        "averageionisation_Fe": 3.5216503938039145,
        "populations_FeI": 2.5604762125407237e-24,
        "populations_FeII": 1.3273970472709221e-13,
        "populations_FeIII": 7.15164795263741e-05,
        "populations_FeIV": 0.47807180327557336,
        "populations_FeV": 0.5218229725433048,
        "populations_CoII": 0.166666736471993,
        "populations_CoIII": 0.015291374246013826,
        "populations_CoIV": 0.8180049657821655,
        "heating_dep": 2.2832464959235988e-14,
        "heating_coll": 0.0,
        "heating_bf": 7.490066713015075e-16,
        "heating_ff": 3.7438500580976916e-18,
        "cooling_adiabatic": 1.617221173361486e-14,
        "cooling_coll": 1.7812333777430962e-14,
        "cooling_fb": 2.8139667559446467e-17,
        "cooling_ff": 1.1622199729222756e-17,
    }

    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {
        varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list, strict=False)
    }

    print({key: float(np.array(yarr).mean()) for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose(expectedval, np.array(yvals[varname]).mean(), rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@pytest.mark.benchmark
def test_estimator_timeevolution() -> None:
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath / "test_estimator_timeevolution",
        plotlist=[["Te", "nne"]],
        modelgridindex=0,
        x="time",
    )


def test_estimparse_xz_high_preset(tmp_path: Path) -> None:
    """An estimator file compressed with xz -9 declares a 64 MiB dictionary and must still be readable."""
    import lzma

    with lzma.open(tmp_path / "estimators_0000.out.xz", "wt", preset=9) as f:
        f.write("timestep 0 modelgridindex 0 TR 2000 Te 3000 W 1 TJ 2000 nne 1.0e5\n")

    dfest = at.rustext.estimparse(tmp_path, 0, 0)
    assert dfest["Te"].to_list() == pytest.approx([3000.0])


def test_estimparse() -> None:
    pldf = at.rustext.estimparse(modelpath, 0, 0)
    assert pldf.height == 100
    assert {"timestep", "modelgridindex", "TR", "Te", "nne", "nnion_Fe_II", "heating_ff"} <= set(pldf.columns)

    firstcell = pldf.sort("timestep", "modelgridindex").row(0, named=True)
    assert firstcell["timestep"] == 0
    assert firstcell["modelgridindex"] == 0
    assert firstcell["nne"] == pytest.approx(71393.3)
    assert firstcell["nnion_Fe_II"] == pytest.approx(80.59)
    # nnelement is the sum over the ion stages of the element
    assert firstcell["nnelement_Fe"] == pytest.approx(6.226e05 + 8.059e01 + 3.940e-24 + 1.586e-27 + 1.010e-27)
    # quantities recorded as X*nne are also stored divided by the electron density
    assert firstcell["Alpha_R_Fe_II"] == pytest.approx(1.821e-07 / 71393.3)


def test_estimparse_missing_file() -> None:
    with pytest.raises(OSError, match="no estimator file found for rank 999"):
        at.rustext.estimparse(modelpath, 999, 999)


@pytest.mark.parametrize(
    ("badline", "errormessage"),
    [
        ("populations    Z=26  1: 6.226e+05  2: NOTANUMBER", 'could not parse "NOTANUMBER" as a number'),
        ("gamma_R        Z=26  99: 1.0", "no roman numeral for ion stage 99"),
        ("gamma_R        Z=999  1: 1.0", "no element symbol for atomic number 999"),
        ("populations    Z=26  1: 1.0  1: 2.0", "a column was given two values for one cell"),
    ],
)
def test_estimparse_malformed_line(tmp_path: Path, badline: str, errormessage: str) -> None:
    """Unparseable estimator data raises a Python exception naming the file and line, never a panic."""
    (tmp_path / "estimators_0000.out").write_text(
        f"timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 71393.3\n{badline}\n"
    )

    with pytest.raises(Exception, match=f"estimators_0000.out:2: {re.escape(errormessage)}"):
        at.rustext.estimparse(tmp_path, 0, 0)


def test_add_derived_estimator_columns_preserves_nulls() -> None:
    """Missing nnelement values count as zero, but a null in any other column must not become a real zero."""
    pldf = pl.LazyFrame({
        "timestep": [0, 1],
        "modelgridindex": [0, 0],
        "Te": [5000.0, None],
        "nne": [None, 1.0e8],
        "nnelement_Fe": [2.0, None],
        "nnelement_Ni": [3.0, 4.0],
    })

    dfout = at.estimators.add_derived_estimator_columns(pldf).collect()

    # nnelement nulls are filled with zero and summed into nntot
    assert dfout["nnelement_Fe"].to_list() == [2.0, 0.0]
    assert dfout["nntot"].to_list() == [5.0, 4.0]

    # every other column keeps its nulls
    assert dfout["Te"].to_list() == [5000.0, None]
    assert dfout["nne"].to_list() == [None, 1.0e8]


def test_add_derived_estimator_columns_total_dep() -> None:
    pldf = pl.LazyFrame({
        "timestep": [0],
        "deposition_gamma": [1.0],
        "deposition_positron": [2.0],
        "deposition_alpha": [4.0],
    })

    dfout = at.estimators.add_derived_estimator_columns(pldf).collect()

    assert dfout["total_dep"].to_list() == [7.0]
    assert "nntot" not in dfout.columns


def test_add_derived_estimator_columns_fills_ion_and_isotope_nulls() -> None:
    """A number density column that a whole MPI rank omitted arrives as null, and must be read as zero.

    Within a single estimator file the reader already fills these with zero, so leaving the cross-rank nulls
    alone would give the same missing ion two different values depending on which rank wrote the cell.
    """
    pldf = pl.LazyFrame({
        "timestep": [0, 1],
        "modelgridindex": [0, 1],
        "nnelement_Fe": [2.0, None],
        "nnion_Fe_II": [None, 1.5],
        "nniso_Ni56": [None, 3.0],
        "Te": [5000.0, None],
    })

    dfout = at.estimators.add_derived_estimator_columns(pldf).collect()

    assert dfout["nnion_Fe_II"].to_list() == [0.0, 1.5]
    assert dfout["nniso_Ni56"].to_list() == [0.0, 3.0]
    assert dfout["nnelement_Fe"].to_list() == [2.0, 0.0]

    # temperatures are not number densities, so a null must stay null
    assert dfout["Te"].to_list() == [5000.0, None]


def test_estimparse_rejects_missing_nne(tmp_path: Path) -> None:
    """A '*nne' estimator must not be divided by another cell's electron density.

    The nne column is only as long as the current row once this cell has set it, so a cell header without nne has
    to be an error rather than silently reusing the previous cell's value.
    """
    (tmp_path / "estimators_0000.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 1.0e5\n"
        "Alpha_R*nne    Z=26  2: 2.0e5\n"
        "timestep 0 modelgridindex 1 TR 2000 Te 2000 W 1 TJ 2000\n"
        "Alpha_R*nne    Z=26  2: 4.0e5\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="nne is not set for this cell"):
        at.rustext.estimparse(tmp_path, 0, 0)


def test_estimparse_divides_by_own_cell_nne(tmp_path: Path) -> None:
    """Each cell's '*nne' estimator is divided by that same cell's electron density."""
    (tmp_path / "estimators_0000.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 1.0e5\n"
        "Alpha_R*nne    Z=26  2: 2.0e5\n"
        "timestep 0 modelgridindex 1 TR 2000 Te 2000 W 1 TJ 2000 nne 4.0e5\n"
        "Alpha_R*nne    Z=26  2: 8.0e5\n",
        encoding="utf-8",
    )

    dfest = at.rustext.estimparse(tmp_path, 0, 0).sort("modelgridindex")

    assert dfest["nne"].to_list() == pytest.approx([1.0e5, 4.0e5])
    assert dfest["Alpha_R_Fe_II"].to_list() == pytest.approx([2.0, 2.0])


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_default_plotlist_skips_absent_elements(mockplot: mock.MagicMock) -> None:
    """The built-in plot list names particular elements, which most models do not contain."""
    funcoutpath = outputpath / "test_estimator_default_plotlist"
    funcoutpath.mkdir(exist_ok=True, parents=True)

    # testmodel has no Sr, so the default averageionisation/populations plots must be skipped rather than raising
    at.estimators.plot(argsraw=[], modelpath=modelpath, timedays=300, outputfile=funcoutpath)

    # the element-independent default subplots (rho and TR) are still drawn
    assert len(mockplot.call_args_list) >= 2
    for call in mockplot.call_args_list:
        yvalues = np.array(call[0][2], dtype=float)
        assert np.all(np.isfinite(yvalues))
        assert np.all(yvalues > 0.0)


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_levelpopulation_dn_on_dvel(mockplot: mock.MagicMock) -> None:
    """Plotting dN/dv needs the inner shell velocity, which is a derived model column."""
    funcoutpath = outputpath / "test_estimator_levelpopulation_dn_on_dvel"
    funcoutpath.mkdir(exist_ok=True, parents=True)

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        timedays=300,
        outputfile=funcoutpath,
        plotlist=[[["levelpopulation_dn_on_dvel", ["Fe II 5"]]]],
    )

    assert mockplot.call_args_list, "nothing was plotted"
    yvalues = np.array(mockplot.call_args_list[0][0][2], dtype=float)
    assert np.all(np.isfinite(yvalues))
    assert np.all(yvalues >= 0.0)


def test_get_averageexcitation() -> None:
    """The average excitation energy must be the population-weighted mean level energy of the ion."""
    dfpops = at.nltepops.read_files(modelpath).filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2))
    timestep = min(dfpops["timestep"].to_list())
    dftexc = pl.LazyFrame({"timestep": [timestep], "modelgridindex": [0], "T_exc": [6000.0]})

    dfavgexc = at.estimators.get_averageexcitation(modelpath, 26, 2, dftexc, dfnltepops=dfpops.lazy()).collect()
    assert len(dfavgexc) == 1

    avgexc = dfavgexc["averageexcitation"].item()
    ionlevels = (
        at.atomic.get_levels(modelpath).filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2))["levels"].item()
    )
    assert ionlevels is not None

    # the mean must lie between the lowest and highest level energies of the ion
    dfts = dfpops.filter((pl.col("timestep") == timestep) & (pl.col("modelgridindex") == 0))
    occupiedlevels = dfts.filter(pl.col("level") >= 0)["level"].to_list()
    energiesoccupied = ionlevels.filter(pl.col("levelindex").is_in(occupiedlevels))
    assert energiesoccupied["energy_ev"].min() <= avgexc <= energiesoccupied["energy_ev"].max()

    # weighting by hand over the resolved levels only must bracket the value, which also includes the superlevel
    dfresolved = dfts.filter(pl.col("level") >= 0).join(
        ionlevels.select(level=pl.col("levelindex").cast(pl.Int64), energy_ev="energy_ev"), on="level", how="inner"
    )
    avgexc_resolvedonly = float((dfresolved["energy_ev"] * dfresolved["n_NLTE"]).sum()) / float(dfts["n_NLTE"].sum())
    assert avgexc >= avgexc_resolvedonly, "adding the superlevel population can only raise the mean energy"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_averageexcitation_plot(mockplot: mock.MagicMock) -> None:
    """The averageexcitation plot item must draw a finite series for each requested ion."""
    funcoutpath = outputpath / "test_estimator_averageexcitation"
    funcoutpath.mkdir(exist_ok=True, parents=True)

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        plotlist=[[["averageexcitation", ["Fe II", "Fe III"]]]],
        timedays=300,
        outputfile=funcoutpath,
    )

    assert len(mockplot.call_args_list) >= 2
    for call in mockplot.call_args_list:
        yvalues = np.array(call[0][2], dtype=float)
        assert np.all(np.isfinite(yvalues))
        assert np.all(yvalues >= 0.0), "an excitation energy above the ground state cannot be negative"


def test_averageexcitation_plotitem_needs_nlte_files(tmp_path: Path) -> None:
    """The default plot list must drop averageexcitation for a model with no NLTE population files."""
    from artistools.estimators.plotestimators import default_plotitem_has_data

    plotitem = [["averageexcitation", ["Fe II"]]]
    estimatorcolumns = ["timestep", "modelgridindex", "Te", "nnelement_Fe"]

    assert default_plotitem_has_data(plotitem, estimatorcolumns, modelpath)
    assert not default_plotitem_has_data(plotitem, estimatorcolumns, tmp_path)


def test_get_elemcolor_is_stable_and_unbounded() -> None:
    """Every element must have a colour, fixed by the element and not by how many plots came before it.

    Colours used to be handed out in call order from a ten-entry list, so an element's colour depended on the
    plots that preceded it in the same process, and the eleventh element raised IndexError.
    """
    from artistools.estimators.plotestimators import colors_tab10
    from artistools.estimators.plotestimators import elementcolors
    from artistools.estimators.plotestimators import get_elemcolor

    assert get_elemcolor(elsymbol="Fe") == colors_tab10[0]
    assert get_elemcolor(elsymbol="Ni") == colors_tab10[1]
    assert get_elemcolor(elsymbol="Co") == colors_tab10[2]

    # far more elements than the reserved list can hold, in two different orders
    colors_ascending = [get_elemcolor(atomic_number=Z) for Z in range(1, 93)]
    colors_descending = [get_elemcolor(atomic_number=Z) for Z in reversed(range(1, 93))]
    assert colors_ascending == list(reversed(colors_descending))

    assert get_elemcolor(elsymbol="O") == get_elemcolor(atomic_number=8)

    # a reserved colour must stay unique to its element, so that Fe, Ni and Co remain identifiable
    reservedZ = {at.get_atomic_number(elsymbol) for elsymbol in elementcolors}
    unreserved = [get_elemcolor(atomic_number=Z) for Z in range(1, 93) if Z not in reservedZ]
    assert not set(elementcolors.values()).intersection(unreserved)


def test_estimparse_index_columns_are_integers() -> None:
    """The timestep and modelgridindex columns must be parsed as integers, not floats.

    Everything in the cell header used to be stored as f32, which cannot represent every cell number of a large
    3D grid exactly: 26999999 comes back as 27000000.
    """
    dfestim = at.rustext.estimparse(modelpath, 0, 0)

    for colname in ("timestep", "modelgridindex"):
        assert dfestim.schema[colname] == pl.Int32, f"{colname} must stay an exact integer"

    # the physical quantities are still f32
    assert dfestim.schema["Te"] == pl.Float32


def test_a_current_parquet_cache_starts_no_progress_bar(tmp_path: Path) -> None:
    """The bar counts a conversion of the estimator text files, thus a current cache starts none.

    The scan of the parquet files is lazy, thus a run whose caches were current showed a bar that
    came and went with no work behind it.
    """
    from artistools.estimators.estimators import CACHEVERSION
    from artistools.estimators.estimators import get_rankbatch_parquetpath
    from artistools.estimators.estimators import rankbatch_parquet_is_current
    from artistools.estimators.estimators import rankbatch_parquet_staleness

    parquetfilepath = get_rankbatch_parquetpath(tmp_path, [0, 1, 2], 0)
    assert parquetfilepath.name == "estimbatch00_0000_0002.out.parquet.tmp"

    # a cache that no run wrote yet needs the conversion
    assert rankbatch_parquet_staleness(parquetfilepath, None) == "the file does not exist"

    mtime = 1000.0
    at.write_parquet_atomic(
        pl.DataFrame({"timestep": [0]}),
        parquetfilepath,
        metadata={"cacheversion": str(CACHEVERSION), "textsource_mtime": str(mtime)},
    )

    # a folder with no text files keeps the cache, and a matching stamp also keeps it
    assert rankbatch_parquet_is_current(parquetfilepath, None)
    assert rankbatch_parquet_is_current(parquetfilepath, mtime)

    # a text source time other than the stamped one needs the conversion again
    assert not rankbatch_parquet_is_current(parquetfilepath, mtime + 10.0)


def test_a_cache_without_a_current_stamp_is_stale(tmp_path: Path) -> None:
    """A cache must hold the current cache version and the time of its text source, or a scan rejects it.

    The freshness rule compared only modification times. Thus a scan accepted a cache that an older
    artistools wrote with different columns, and the diagonal concat filled the difference with
    nulls.
    """
    from artistools.estimators.estimators import rankbatch_parquet_is_current

    mtime = 1000.0

    # no metadata stamp, e.g. a cache from before the stamp existed
    unstamped = tmp_path / "estimbatch00_0000_0002.out.parquet.tmp"
    at.write_parquet_atomic(pl.DataFrame({"timestep": [0]}), unstamped)
    assert not rankbatch_parquet_is_current(unstamped, mtime)

    # a matching text source time but a different cache version
    oldversion = tmp_path / "estimbatch01_0003_0005.out.parquet.tmp"
    at.write_parquet_atomic(
        pl.DataFrame({"timestep": [0]}), oldversion, metadata={"cacheversion": "0", "textsource_mtime": str(mtime)}
    )
    assert not rankbatch_parquet_is_current(oldversion, mtime)

    # a file that is not parquet
    unreadable = tmp_path / "estimbatch02_0006_0008.out.parquet.tmp"
    unreadable.write_bytes(b"not parquet")
    assert not rankbatch_parquet_is_current(unreadable, mtime)


def test_one_rewritten_rank_file_makes_its_batch_stale(tmp_path: Path) -> None:
    """The newest text file of the batch decides its freshness, and no other file hides it.

    One file, in the unspecified order of a glob, decided for the whole folder. Thus a restart that
    rewrote the file of a later rank kept the stale caches current.
    """
    from artistools.estimators.estimators import get_batch_textsource_mtime
    from artistools.estimators.estimators import get_textsource_mtimes

    for rank in range(3):
        (tmp_path / f"estimators_{rank:04d}.out").write_text("timestep 0\n")

    mtimes = get_textsource_mtimes(tmp_path)
    assert sorted(mtimes.keys()) == [0, 1, 2]

    newest = max(mtimes.values())
    os.utime(tmp_path / "estimators_0002.out", (newest + 100.0, newest + 100.0))
    mtimes = get_textsource_mtimes(tmp_path)

    assert get_batch_textsource_mtime(mtimes, 0, 2) == newest + 100.0
    # a batch that does not hold the rewritten rank keeps its own time
    assert get_batch_textsource_mtime(mtimes, 0, 1) == max(mtimes[0], mtimes[1])
    # a batch with no text file gives None
    assert get_batch_textsource_mtime(mtimes, 5, 9) is None


def test_the_stamp_reads_the_file_that_the_parser_reads(tmp_path: Path) -> None:
    """The mtime of a rank comes from the file that the Rust parser selects, and no sibling decides.

    The glob kept an arbitrary candidate of each rank. Thus a leftover sibling such as
    estimators_0000.out.bak could give the stamp while the parser read estimators_0000.out.
    """
    from artistools.estimators.estimators import get_textsource_mtimes

    outfile = tmp_path / "estimators_0000.out"
    outfile.write_text("timestep 0\n")
    bakfile = tmp_path / "estimators_0000.out.bak"
    bakfile.write_text("timestep 0\n")
    os.utime(bakfile, (outfile.stat().st_mtime + 100.0, outfile.stat().st_mtime + 100.0))

    assert get_textsource_mtimes(tmp_path) == {0: outfile.stat().st_mtime}

    # a compressed file counts only when the plain name is absent, in the order of the parser
    gzfile = tmp_path / "estimators_0001.out.gz"
    gzfile.write_bytes(b"")
    assert get_textsource_mtimes(tmp_path)[1] == gzfile.stat().st_mtime


def test_a_cached_scan_asks_for_no_progress_class() -> None:
    """A scan that converts no text file must not build the progress class, which takes a lock."""
    import artistools.misc.general

    with mock.patch.object(
        artistools.misc.general, "get_progress_class", side_effect=AssertionError("a cached scan made a bar")
    ) as mockprogress:
        at.estimators.scan_estimators(modelpath=modelpath).select(pl.len()).collect()

    mockprogress.assert_not_called()


def test_scan_estimators_filters_codecomparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The codecomparison branch must honour modelgridindex and timestep like the ARTIS branch does.

    scan_estimators returns early for a codecomparison/ path, so it has to apply the filters itself:
    read_reference_estimators ignores both arguments and always parses the whole phys file.
    """
    physdir = tmp_path / "toymodel"
    physdir.mkdir()
    (physdir / "phys_toymodel_toycode.txt").write_text(
        "#NTIMES: 2\n"
        "#TIMES[d]: 1.0 2.0\n"
        "#TIME: 1.0\n"
        "#NVEL: 3\n"
        "#vel_mid Te rho nne nntot\n"
        "1000.0 5000.0 1e-13 1e6 1e6\n"
        "2000.0 5100.0 2e-13 2e6 2e6\n"
        "3000.0 5200.0 3e-13 3e6 3e6\n"
        "#TIME: 2.0\n"
        "#NVEL: 3\n"
        "1000.0 6000.0 1e-14 1e5 1e5\n"
        "2000.0 6100.0 2e-14 2e5 2e5\n"
        "3000.0 6200.0 3e-14 3e5 3e5\n"
    )

    realgetpath = at.get_path

    def fake_get_path(key: str) -> Path:
        return tmp_path if key == "codecomparisondata1path" else realgetpath(key)

    # codecomparison.py calls at.get_path, i.e. the top-level re-export rather than commands.get_path
    monkeypatch.setattr(at, "get_path", fake_get_path)

    modelpath = "codecomparison/toymodel/toycode"

    dfall = at.estimators.scan_estimators(modelpath=modelpath).collect()
    assert dfall.height == 6, "the unfiltered scan should return every timestep and cell"

    dfone = at.estimators.scan_estimators(modelpath=modelpath, timestep=1, modelgridindex=2).collect()
    assert dfone.height == 1
    assert dfone["timestep"].item() == 1
    assert dfone["modelgridindex"].item() == 2
    assert np.isclose(dfone["Te"].item(), 6200.0)


def test_exportmassfractions(tmp_path: Path) -> None:
    """Every element the estimators carry should appear, weighted by its standard atomic mass."""
    outfile = tmp_path / "massfracs.txt"
    at.estimators.exportmassfractions.main(argsraw=[], modelpath=modelpath, modelgridindex="0", outputpath=outfile)

    lines = outfile.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("d shell 0")
    massfracs = {parts[1]: float(parts[2]) for parts in (line.split() for line in lines[1:])}

    # compositiondata.txt lists only Fe and Co, so an element mass taken from there would drop Ni
    assert set(massfracs) == {"Fe", "Co", "Ni"}
    assert np.isclose(sum(massfracs.values()), 1.0)
    assert np.isclose(massfracs["Fe"], 0.9030389, rtol=1e-5)
    assert np.isclose(massfracs["Co"], 0.0969611, rtol=1e-5)


def test_parse_ion_row_classic_keys_elements_by_symbol() -> None:
    """The classic reader must key an element number density on the element symbol, as every consumer does."""
    from artistools.estimators.estimators_classic import parse_ion_row_classic

    outdict: dict[str, t.Any] = {}
    # six leading values that the reader skips, then one population for each ion
    row = ["0", "1", "2", "3", "4", "5", "10.0", "20.0", "40.0"]
    parse_ion_row_classic(row, outdict, {26: 2, 28: 1})

    assert np.isclose(outdict["nnion_Fe_I"], 10.0)
    assert np.isclose(outdict["nnion_Fe_II"], 20.0)
    assert np.isclose(outdict["nnion_Ni_I"], 40.0)

    # the element key uses the symbol, so that plotestimators and exportmassfractions can find it
    assert np.isclose(outdict["nnelement_Fe"], 30.0)
    assert np.isclose(outdict["nnelement_Ni"], 40.0)
    assert not [key for key in outdict if key.startswith("nnelement_") and key[len("nnelement_") :].isdigit()]

    # add_derived_estimator_columns derives nntot, so the reader must not carry its own copy
    assert "nntot" not in outdict


@pytest.mark.parametrize(
    ("plotarg", "expected"),
    [("te", "Did you mean Te?"), ("notavariable", "--listvariables"), ("rhoo", "Did you mean rho?")],
)
def test_estimator_unknown_variable_suggests_a_name(
    plotarg: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown plot variable must name the closest estimator column, or list some of them."""
    with pytest.raises(SystemExit) as excinfo:
        at.estimators.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=260, plotlist=[[plotarg]])

    assert excinfo.value.code == 1
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize(("directive", "expected"), [("_foo=bar", "not a plot directive"), ("ymim=1", "ymin")])
def test_estimator_unknown_directive_names_the_valid_ones(
    directive: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown plot directive must say so and name the directives that exist."""
    with pytest.raises(SystemExit) as excinfo:
        at.estimators.plot(
            argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=260, plotlist=[["rho", directive]]
        )

    assert excinfo.value.code == 1
    captured = capsys.readouterr().err
    assert expected in captured
    assert "yscale=" in captured


def test_estimator_valid_ion_accepts_an_ion_and_rejects_a_variable_typo() -> None:
    """The ion test must accept the ion spellings that get_iontuple reads, and reject a variable typo."""
    for good in ("Fe", "Fe II", "FeII", "26", "Fe56"):
        assert at.estimators.plotestimators.is_valid_ion(good), good

    for bad in ("te", "notanion", "zz"):
        assert not at.estimators.plotestimators.is_valid_ion(bad), bad


def test_estimator_listvariables_collapses_the_species_families(capsys: pytest.CaptureFixture[str]) -> None:
    """--listvariables must name each per-species family one time, and not every column of it."""
    at.estimators.plot(argsraw=[], modelpath=modelpath, listvariables=True)

    out = capsys.readouterr().out
    assert "estimator variables:" in out
    assert "nnion_<ion>" in out
    assert "nnelement_<element>" in out

    # the listing names the family, thus no column of it appears in full
    assert "nnion_Fe_II" not in out
    assert "nnelement_Fe" not in out

    # it is far shorter than one line for each column
    assert len(out.splitlines()) < 60


@pytest.mark.parametrize("prefix", ["", "_"])
def test_estimator_directive_underscore_is_optional(prefix: str, capsys: pytest.CaptureFixture[str]) -> None:
    """A plot directive works with or without its underscore, and each subplot keeps its own scale."""
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath,
        timedays=260,
        plotlist=[["TR", [f"{prefix}yscale", "lin"]], ["rho", [f"{prefix}yscale", "log"]]],
    )

    # exit_with_error writes to the standard error and then raises SystemExit, thus a rejected directive
    # would already have ended this test. Read the scale that the directive asked for instead
    assert not capsys.readouterr().err


def test_estimator_xmin_is_a_figure_argument_and_not_a_directive(capsys: pytest.CaptureFixture[str]) -> None:
    """The subplots share one horizontal axis, thus xmin= names the argument that sets it for the figure."""
    with pytest.raises(SystemExit) as excinfo:
        at.estimators.plot(
            argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=260, plotlist=[["rho", "xmin=260"]]
        )

    assert excinfo.value.code == 1
    message = capsys.readouterr().err
    assert "-xmin" in message
    assert "share one horizontal axis" in message


@mock.patch.object(mplax.Axes, "set_xlim", side_effect=mplax.Axes.set_xlim, autospec=True)
def test_estimator_xmin_argument_sets_the_axis_of_every_subplot(mockxlim: mock.MagicMock) -> None:
    """-xmin and -xmax reach the whole figure, because one horizontal axis serves every subplot."""
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath,
        timedays=260,
        plotlist=[["TR"], ["rho"]],
        xmin=1000,
        xmax=4000,
    )

    # the last call on each axes decides the view, and both subplots must end at the requested range
    lastlimits: dict[int, tuple[float, float]] = {}
    for call in mockxlim.call_args_list:
        if len(call.args) >= 3 and call.args[1] is not None and call.args[2] is not None:
            lastlimits[id(call.args[0])] = (float(call.args[1]), float(call.args[2]))

    assert len(lastlimits) == 2, "each of the two subplots must get the limits"
    for limits in lastlimits.values():
        assert np.allclose(limits, (1000.0, 4000.0))


def test_estimator_xmin_as_a_plot_item_names_the_argument() -> None:
    """A bare list ["xmin", value] must give the same message as the "xmin=value" string.

    normalise_plotitems adds the underscore to the string form alone, thus the bare list reached the ion
    branch and stopped with a TypeError that named no argument.
    """
    with pytest.raises(SystemExit) as excinfo:
        at.estimators.plot(
            argsraw=[], modelpath=modelpath, outputfile=outputpath, timedays=260, plotlist=[["rho", ["xmin", 260.0]]]
        )

    assert excinfo.value.code == 1


def test_split_species_suffix_reads_a_symbol_that_is_also_a_roman_numeral() -> None:
    """C, V, I, X, and L are element symbols and Roman numerals, thus one reading of a name is not enough.

    In init_X_C the first reading takes X as the element and _C as the ion stage. That reading has to give
    way to the family init_X and the element C.
    """
    from artistools.estimators.estimators import split_species_suffix

    assert split_species_suffix("init_X_C") == ("init_X", "C")
    assert split_species_suffix("init_X_V") == ("init_X", "V")
    assert split_species_suffix("init_X_Al") == ("init_X", "Al")

    # a suffix that names no element stays out of a species family
    assert split_species_suffix("init_X_Fegroup") is None


def test_estimator_listvariables_describes_each_prefix_group(capsys: pytest.CaptureFixture[str]) -> None:
    """A group of columns that share a prefix appears one time, with a description of the group."""
    at.estimators.plot(argsraw=[], modelpath=modelpath, listvariables=True)

    out = capsys.readouterr().out
    # the test model has no deposition columns, thus this checks the groups that it does have
    for prefix, description in (
        ("cooling_<name>", "cooling rate"),
        ("heating_<name>", "heating rate"),
        ("init_<name>", "model snapshot"),
        ("vel_<name>", "velocity coordinates"),
    ):
        assert prefix in out, prefix
        assert description in out, description

    # the group names each member one time, thus no member appears with its prefix
    assert "cooling_adiabatic" not in out


def test_split_species_suffix_rebuilds_the_column_name() -> None:
    """Each family and species of the listing must join back into the column name that it came from."""
    from artistools.estimators.estimators import split_species_suffix

    for colname, expected in (
        ("nnion_Fe_II", ("nnion", "Fe II")),
        ("gamma_NT_Ar_III", ("gamma_NT", "Ar III")),
        ("nnelement_Ar", ("nnelement", "Ar")),
        ("nniso_Co56", ("nniso", "Co56")),
        # an underscore joins the element symbol to any suffix that is not an ion stage
        ("nniso_Fe_otherstable", ("nniso", "Fe_otherstable")),
    ):
        split = split_species_suffix(colname)
        assert split == expected, f"{colname} gave {split}"
        assert split is not None
        family, species = split
        assert f"{family}_{species.replace(' ', '_')}" == colname

    assert split_species_suffix("Te") is None
    assert split_species_suffix("cooling_ff") is None


def test_get_units_takes_a_column_name_or_a_prefix() -> None:
    """The lookup reads a whole column name or the prefix of a family, in LaTeX or in plain text."""
    # a family prefix, with or without its underscore, gives the same units as one of its columns
    for name in ("nniso", "nniso_", "nniso_Fe56", "nniso_Fe_otherstable"):
        assert at.estimators.get_units(name, latex=False) == "cm^-3", name

    assert at.estimators.get_units("nniso") == "cm$^{-3}$"
    assert at.estimators.get_units("nniso", latex=False) == "cm^-3"

    # a prefix of more than one part, and a name that carries the quantity at its end
    assert at.estimators.get_units("gamma_NT_Ar_I", latex=False) == "s^-1"
    assert at.estimators.get_units("init_volume", latex=False) == "cm^3"
    assert at.estimators.get_units("volume_prevtimestep", latex=False) == "cm^3"

    # a suffix names the units of a derived column
    assert at.estimators.get_units("vel_r_min_kmps", latex=False) == "km/s"
    assert at.estimators.get_units("vel_r_min_on_c", latex=False) == "c"
    assert at.estimators.get_units("vel_r_min", latex=False) == "cm/s"

    assert at.estimators.get_units("notavariable") is None


def test_every_estimator_column_has_units_explained() -> None:
    """Each estimator column must give units, or say why it carries none."""
    from artistools.estimators.estimators import get_variable

    columns = at.estimators.scan_estimators(modelpath).collect_schema().names()
    assert len(columns) > 50, "the test model must hold a representative set of columns"

    unexplained = [col for col in columns if not at.estimators.get_units(col) and not get_variable(col).note]
    assert not unexplained, f"no units for {sorted(unexplained)}"


def test_listvariables_names_the_units_of_a_group_whose_members_differ(capsys: pytest.CaptureFixture[str]) -> None:
    """A group of one unit names it one time. A group of several names the units of each member."""
    at.estimators.plot(argsraw=[], modelpath=modelpath, listvariables=True)
    # the listing wraps its long lines, thus a name and its units can fall on two lines
    out = " ".join(capsys.readouterr().out.split())

    # the cooling rates share one unit, thus the heading carries it
    assert "cooling_<name>" in out
    assert "[erg/s/cm^3]: cooling rate" in out

    # the test model holds vel_r_max_kmps and vel_r_min_kmps but no vel_r_mid_kmps, thus a heading of
    # three variants would name a column that does not exist. Every column takes its own units instead
    assert "vel_<name>, vel_<name>_kmps" not in out
    for member, units in (("r_max", "cm/s"), ("r_max_kmps", "km/s"), ("r_min_on_c", "c")):
        assert f"{member} [{units}]" in out, member

    # the model snapshot columns disagree, thus each of them carries its own units
    assert "kinetic_en_erg [erg]" in out
    assert "logrho [log10(g/cm^3)]" in out


def test_listvariables_names_the_variants_when_every_base_has_them() -> None:
    """A group whose bases all carry the same variants names them one time, with the units of each."""
    from artistools.estimators.estimators import summarise_columns

    # a complete grid: two bases, each with the plain form and the _on_c form
    complete = summarise_columns(["vel_r_mid", "vel_r_mid_on_c", "vel_x_mid", "vel_x_mid_on_c"])
    assert "vel_<name>, vel_<name>_on_c" in complete
    assert "[cm/s], [c]" in complete

    # one missing variant makes the grid describe a column that does not exist, thus the full listing
    incomplete = summarise_columns(["vel_r_mid", "vel_r_mid_on_c", "vel_x_mid"])
    assert "vel_<name>, vel_<name>_on_c" not in incomplete
    assert "r_mid_on_c [c]" in incomplete


def test_summarise_nuclides_replaces_a_long_family() -> None:
    """A family of more than MAXSPECIES_LISTED nuclides gives a summary, and names the flag for the rest."""
    from artistools.estimators.estimators import MAXSPECIES_LISTED
    from artistools.estimators.estimators import summarise_columns

    nuclides = [f"nniso_Fe{massnum}" for massnum in range(40, 40 + MAXSPECIES_LISTED + 5)]
    listing = summarise_columns(nuclides)
    assert "nuclides of 1 elements" in listing
    assert "--listnuclides" in listing
    assert "Fe56" not in listing

    # the same family with every nuclide named
    full = summarise_columns(nuclides, fullnuclides=True)
    assert "Fe56" in full
    assert "--listnuclides" not in full


def test_summarise_columns_keeps_a_family_of_elements_whole() -> None:
    """A family of bare element symbols is short, thus it stays whole however many elements it holds."""
    from artistools.estimators.estimators import MAXSPECIES_LISTED
    from artistools.estimators.estimators import summarise_columns

    elements = [f"nnelement_{sym}" for sym in at.get_elsymbolslist()[1 : MAXSPECIES_LISTED + 20]]
    listing = summarise_columns(elements)

    assert "nuclides" not in listing
    assert "--listnuclides" not in listing
    assert "Fe" in listing


def test_species_placeholder_names_what_a_family_takes() -> None:
    """The heading of a family must name what it takes, and a mixed family must name both kinds."""
    from artistools.estimators.estimators import species_placeholder

    assert species_placeholder(["Fe", "Ni"]) == "element"
    assert species_placeholder(["Fe II", "Ni III"]) == "ion"
    assert species_placeholder(["Fe56", "Ni_otherstable"]) == "nuclide"

    # init_X_ takes the mass fraction of an element such as init_X_Fe, and of a nuclide such as init_X_Fe52
    assert species_placeholder(["Fe", "Fe52", "Ni56"]) == "element or nuclide"


def test_listvariables_heading_names_the_kind_of_each_family(capsys: pytest.CaptureFixture[str]) -> None:
    """The listing must show which kind of species each family takes."""
    at.estimators.plot(argsraw=[], modelpath=modelpath, listvariables=True)
    out = capsys.readouterr().out

    assert "nnelement_<element>" in out
    assert "nnion_<ion>" in out
    assert "init_X_<element or nuclide>" in out

    # the test model holds no isotope columns, thus read that family from the summariser
    from artistools.estimators.estimators import summarise_columns

    assert "nniso_<nuclide>" in summarise_columns(["nniso_Fe56", "nniso_Ni_otherstable"])


def test_summarise_ions_breaks_a_range_at_a_gap() -> None:
    """A missing ion stage must break the range, so that the listing names no absent column."""
    from artistools.estimators.estimators import summarise_ions

    # Fe I and Fe III without Fe II: one range would name nnion_Fe_II, which the model does not hold
    assert summarise_ions(["Fe I", "Fe III"]) == "Fe I, Fe III"
    assert summarise_ions(["Fe I", "Fe II", "Fe IV", "Fe V"]) == "Fe I-II, Fe IV-V"

    # a run with no gap keeps its compact form
    assert summarise_ions(["Fe I", "Fe II", "Fe III"]) == "Fe I-III"
    assert summarise_ions(["Fe III"]) == "Fe III"


def test_estimator_lookup_tables_are_read_only() -> None:
    """AGENTS.md forbids mutable state at module level, thus each lookup table is a read-only view."""
    import artistools.commands
    from artistools.estimators import estimators

    for table in (
        estimators.VARIABLES,
        estimators.UNITS_BY_SUFFIX,
        estimators.PREFIX_GROUPS,
        artistools.commands.COMMANDGROUPS,
    ):
        with pytest.raises(TypeError):
            # the type forbids this too, which is the point: the table is read-only at run time as well
            table["newkey"] = "newvalue"  # ty:ignore[invalid-assignment]  # pyrefly: ignore[unsupported-operation]


def test_estimator_keyword_time_selection_refuses_a_conflict() -> None:
    """set_args_from_dict makes a keyword argument a default, thus the name has to count as given."""
    with pytest.raises(SystemExit) as withdays:
        at.estimators.plot(
            argsraw=[], modelpath=modelpath, outputfile=outputpath, plotlist=[["rho"]], timestep=11, timedays=260
        )
    assert withdays.value.code == 1

    with pytest.raises(SystemExit) as withbounds:
        at.estimators.plot(
            argsraw=[],
            modelpath=modelpath,
            outputfile=outputpath,
            plotlist=[["rho"]],
            timestep=11,
            timemin=100,
            timemax=200,
        )
    assert withbounds.value.code == 1


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_keyword_timestep_reads_an_int(mockplot: mock.MagicMock) -> None:
    """A keyword argument gives an int, and a command line gives a string. Both name one timestep.

    The test ran each of the two and read nothing back, thus it passed even where the two named a
    different timestep.
    """
    drawn = {}
    for timestep in (11, "11"):
        mockplot.reset_mock()
        at.estimators.plot(
            argsraw=[], modelpath=modelpath, outputfile=outputpath, plotlist=[["rho"]], timestep=timestep
        )
        drawn[timestep] = [np.asarray(call.args[2], dtype=float).tolist() for call in mockplot.call_args_list]

    assert drawn[11], "the plot must draw something"
    assert drawn[11] == drawn["11"], "the int and the text must name the same timestep"


def test_estimator_makegif_writes_one_frame_per_timestep(tmp_path: Path) -> None:
    """--makegif must write one frame per timestep and join them, and it must not need --multiplot.

    The frame list was made before --makegif set multiplot, thus --makegif alone wrote one frame and no gif.
    """
    at.estimators.plot(
        argsraw=[], modelpath=modelpath, outputfile=tmp_path, plotlist=[["rho"]], timestep="0-2", makegif=True
    )

    assert len(list(tmp_path.glob("*.png"))) == 3
    assert [giffile.name for giffile in tmp_path.glob("*.gif")] == ["plotestimators_evolution_ts000-ts002.gif"]


def test_classic_estimator_files_follow_zopen_precedence(tmp_path: Path) -> None:
    """The reader must skip a .bak sibling, and it must pick between two compressed forms as zopen does.

    A lexical sort put .bak in front of .gz, and .gz in front of .zst. Both orders differ from the one
    that zopen reads, thus the reader could take a stale file and draw obsolete data without a word.
    """
    import gzip

    from artistools.estimators.estimators_classic import get_classic_estimator_files
    from artistools.misc.fileio import get_decompress_open

    with gzip.open(tmp_path / "estimators_0000.out.gz", "wt") as gzfile:
        gzfile.write("stale\n")
    with get_decompress_open(".zst")(tmp_path / "estimators_0000.out.zst", "wt") as zstfile:
        zstfile.write("live\n")
    (tmp_path / "estimators_0000.out.bak").write_text("junk\n")

    allnames = sorted(path.name for path in tmp_path.glob("estimators_????.out*"))
    assert allnames[0] == "estimators_0000.out.bak", "the stale name must sort first, or this proves nothing"

    assert [path.name for path in get_classic_estimator_files(tmp_path)] == ["estimators_0000.out.zst"]

    # the plain file wins over every compressed form, as zopen reads it
    (tmp_path / "estimators_0000.out").write_text("newest\n")
    assert [path.name for path in get_classic_estimator_files(tmp_path)] == ["estimators_0000.out"]


def test_estimator_x_variable_names_the_choices(capsys: pytest.CaptureFixture[str]) -> None:
    """-x takes any variable of the model, thus a wrong name must say what the choices are."""
    with pytest.raises(SystemExit) as excinfo:
        at.estimators.plot(
            argsraw=[], modelpath=modelpath, outputfile=outputpath, timestep=40, plotlist=[["rho"]], x="Tee"
        )

    assert excinfo.value.code == 1
    message = capsys.readouterr().err
    assert "Did you mean Te?" in message
    assert "time, timestep, velocity, and beta" in message
    assert "--listvariables" in message


def build_classic_restart_model(tmp_path: Path, *, secondfolderfirsttimestep: int | None) -> Path:
    """Write a classic model of two run folders, as a restarted run leaves behind."""
    import shutil

    source = at.get_path("testdata") / "test-classicmode_3d"
    for name in ("model.txt", "abundances.txt", "input.txt", "compositiondata.txt"):
        shutil.copy(source / name, tmp_path / name)
    shutil.copy(source / "job0" / "output_0-0.txt", tmp_path / "output_0-0.txt")

    # each row gives a cell index, TR, Te, W, TJ, and then the nine rates that the reader takes from the end
    rows = "\n".join(" ".join([str(mgi), "5000", "4000", "0.5", "4500", *["0.0"] * 9]) for mgi in (0, 1))
    for folder in ("job0", "job1"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "estimators_0000.out").write_text(rows + "\n")

    if secondfolderfirsttimestep is not None:
        (tmp_path / "job1" / "output_0-0.txt").write_text(
            f"[debug] update_packets: updating packet 0 for timestep {secondfolderfirsttimestep}\n"
        )

    return tmp_path


def test_classic_restart_without_an_offset_is_refused(tmp_path: Path) -> None:
    """Two run folders that both start at timestep zero write the same keys.

    The later folder took the place of the earlier one in the dictionary, thus a plot showed the wrong
    data and said nothing.
    """
    from artistools.estimators.estimators_classic import read_classic_estimators

    modelpath = build_classic_restart_model(tmp_path, secondfolderfirsttimestep=None)

    with pytest.raises(ValueError, match="both give timestep 0 of cell 0"):
        read_classic_estimators(modelpath)


def test_classic_restart_with_an_offset_reads_both_folders(tmp_path: Path) -> None:
    """A folder whose log gives its first timestep keeps its own keys, thus the read succeeds."""
    from artistools.estimators.estimators_classic import read_classic_estimators

    modelpath = build_classic_restart_model(tmp_path, secondfolderfirsttimestep=2)

    estimators = read_classic_estimators(modelpath)
    assert estimators is not None
    assert sorted(estimators) == [(0, 0), (0, 1), (2, 0), (2, 1)]


CLASSIC1DPATH = at.get_path("testdata") / "test-classicmode_1d"


@pytest.mark.skipif(not CLASSIC1DPATH.is_dir(), reason="run tests/data/setuptestdata.sh for the 1D classic model")
def test_classic_estimators_read_a_real_run() -> None:
    """Read the estimators of a classic ARTIS run of one dimension.

    The 3D classic model of the test data does not parse with this reader, thus this run covers the
    classic code path: the ion counts of output_0-0.txt, the rows of the estimator files, and the run
    folder that holds them.
    """
    from artistools.estimators.estimators_classic import read_classic_estimators

    estimators = read_classic_estimators(CLASSIC1DPATH)
    assert estimators is not None

    timesteps = {timestep for timestep, _ in estimators}
    cells = {cell for _, cell in estimators}
    assert cells == {0, 1, 2}, "the run holds the three rank files that the archive keeps"
    assert min(timesteps) == 0
    assert max(timesteps) == 110

    firstcell = estimators[0, 0]
    assert np.isclose(firstcell["Te"], 91313.2, rtol=1e-5)
    assert np.isclose(firstcell["TR"], 91313.2, rtol=1e-5)
    assert np.isclose(firstcell["W"], 1.0, rtol=1e-5)

    # the ion counts of output_0-0.txt slice the row, thus a wrong count misaligns every later element
    assert firstcell["nnion_Al_I"] >= 0.0
    assert len([key for key in firstcell if key.startswith("nnion_")]) > 50


def test_classicartis_on_a_modern_model_names_the_difference() -> None:
    """A modern ARTIS run writes the modern estimator format, even with the options of the classic code.

    --classicartis reads the output of the classic ARTIS code, thus its name misleads. A user who gives
    it for a modern run met "invalid literal for int() with base 10: 'timestep'".
    """
    from artistools.estimators.estimators_classic import read_classic_estimators

    with pytest.raises(ValueError, match="format of a modern ARTIS run"):
        read_classic_estimators(modelpath_classic_3d)


@pytest.mark.skipif(not CLASSIC1DPATH.is_dir(), reason="run tests/data/setuptestdata.sh for the 1D classic model")
def test_classicartis_reads_a_classic_run_through_the_scanner() -> None:
    """The whole path from --classicartis to a dataframe must work for a classic ARTIS run."""
    estimators = at.estimators.scan_estimators(modelpath=CLASSIC1DPATH, classicartis=True).collect()

    assert estimators.height > 0
    assert {"timestep", "modelgridindex", "Te", "TR"} <= set(estimators.columns)
    # Series.max() gives a wide union, thus narrow it before the comparison
    maxtemperature = estimators["Te"].max()
    assert isinstance(maxtemperature, float)
    assert maxtemperature > 0.0


def test_makegif_opens_the_gif_and_no_frame(tmp_path: Path) -> None:
    """--open must open the product of the run and not each frame that the product holds.

    save_figure reads --open for each figure, thus a run of many timesteps opened a viewer for each
    frame, and it did not open the gif that holds them.
    """
    outfolder = tmp_path / "frames"
    with mock.patch("subprocess.run") as mockrun:
        at.estimators.plotestimators.main(
            argsraw=[],
            modelpath=modelpath,
            plotlist=[["Te"]],
            timestep="40-42",
            makegif=True,
            open=True,
            outputfile=str(outfolder),
        )

    opened = [call.args[0][1] for call in mockrun.call_args_list]
    assert len(opened) == 1, f"one file must open, not {len(opened)}"
    assert opened[0].endswith(".gif"), opened[0]
    assert len(list(outfolder.glob("*.png"))) == 3, "each timestep must still give a frame"


def test_makegif_takes_the_gif_name_from_o(tmp_path: Path) -> None:
    """-o names the gif that the run makes, and the frames go in the folder that holds it.

    The test for one file read a name that ends in .gif as one frame, thus it refused the name of the
    product that the run makes.
    """
    gifpath = tmp_path / "movie.gif"
    at.estimators.plotestimators.main(
        argsraw=[], modelpath=modelpath, plotlist=[["Te"]], timestep="40-42", makegif=True, outputfile=str(gifpath)
    )

    assert gifpath.is_file(), f"the gif must keep its name, but {list(tmp_path.iterdir())}"
    assert len(list(tmp_path.glob("*.png"))) == 3, "the frames go in the folder of the gif"

    # the folder of the gif can carry a suffix of its own, and the command makes it
    dottedgif = tmp_path / "results.v1" / "movie.gif"
    at.estimators.plotestimators.main(
        argsraw=[], modelpath=modelpath, plotlist=[["Te"]], timestep="40-41", makegif=True, outputfile=str(dottedgif)
    )
    assert dottedgif.is_file(), f"no gif in {list((tmp_path / 'results.v1').iterdir())}"


def test_a_compact_ion_name_reads_the_ion_stage_in_upper_case() -> None:
    """The roman numeral of an ion stage is always in upper case, and the element symbol is not.

    Thus the lower case "i" of "SiII" belongs to silicon, and "SIII" gives S III. The symbols came
    from a set, which holds no order, thus "SiII" took S and read "iII" as the ion stage 3. The order
    of a set of strings also changes with the hash seed, thus one run gave Ni II and another N III.
    """
    from artistools.estimators.plotestimators import get_iontuple
    from artistools.estimators.plotestimators import is_valid_ion

    for ionstr, elsymbol, ion_stage in (
        # the lower case letter of a symbol never reads as a roman numeral
        ("SiII", "Si", 2),
        ("SiIII", "Si", 3),
        ("NiII", "Ni", 2),
        ("CoII", "Co", 2),
        ("ClII", "Cl", 2),
        # the shorter symbol takes the longer run of roman numerals
        ("SII", "S", 2),
        ("SIII", "S", 3),
        ("NII", "N", 2),
        ("CII", "C", 2),
        ("CIII", "C", 3),
        # a symbol that is a roman numeral of its own still names the element
        ("VII", "V", 2),
        ("FeIII", "Fe", 3),
        # the symbol of the element is not case sensitive
        ("siII", "Si", 2),
        ("sIII", "S", 3),
        ("feII", "Fe", 2),
        ("FEII", "Fe", 2),
    ):
        atomic_number, stage = get_iontuple(ionstr)
        assert at.get_elsymbol(atomic_number) == elsymbol, f"{ionstr} gave {at.get_elsymbol(atomic_number)}"
        assert stage == ion_stage, f"{ionstr} gave the ion stage {stage}"

    # a roman numeral in lower case names no ion stage, thus these names hold no ion
    for ionstr in ("Feii", "FeIi", "feii", "Sii"):
        assert not is_valid_ion(ionstr), f"{ionstr} must name no ion"


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_snapshot_classic_3d_cone(mockplot: mock.MagicMock) -> None:
    """-readonlymgi cone selects the cells within -coneangle of the axis.

    The parser did not define -coneangle, which make_cone reads, thus the cone path stopped with
    AttributeError before this test existed.
    """
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        plotlist=[["Te"]],
        outputfile=outputpath / "test_estimator_snapshot_classic_3d_cone.pdf",
        timedays=4,
        readonlymgi="cone",
        axis="+z",
        coneangle=60.0,
    )

    xvalues = np.concatenate([np.array(callargs[0][1], dtype=float) for callargs in mockplot.call_args_list])
    assert len(xvalues) > 0


# the estimators of every test model of the repository hold no deposition_ column, thus a test that
# needs one writes this line into a copy of the test model
DEPOSITIONLINE = "deposition: gamma 2.0e-10 positron 1.0e-10 electron 5.0e-11 alpha 2.5e-11"


def make_model_with_deposition(tmp_path: Path, cellye: Sequence[float] | None = None) -> Path:
    """Return a copy of the test model whose estimators give the deposition rate of each channel.

    The copy holds the four small input files that the reader needs, thus it costs no time and no
    space. The atomic data of the test model stays where it is.

    The model of the repository holds one cell and no Ye column. Thus cellye gives one Ye to each
    cell, and it makes one copy of that cell for each value that it holds. A test of a selection of
    the cells then has a model that the selection can divide.
    """
    modeldir = tmp_path / "modelwithdeposition"
    modeldir.mkdir()
    for name in ("model.txt", "abundances.txt", "input.txt", "compositiondata.txt"):
        shutil.copy(modelpath / name, modeldir / name)

    ncells = 1 if cellye is None else len(cellye)
    if cellye is not None:
        dfmodel, modelmeta = at.inputmodel.get_modeldata(modelpath, printwarningsonly=True)
        cellnumber = pl.int_range(1, ncells + 1, dtype=pl.Int32)
        dfcells = pl.concat([dfmodel.collect()] * ncells).with_columns(
            inputcellid=cellnumber,
            # each shell must end above the one in front of it, thus the outer speed grows with the id
            vel_r_max_kmps=pl.col("vel_r_max_kmps") * cellnumber,
            # each cell holds a density of its own, thus a selection of the cells changes the mass
            logrho=pl.col("logrho") - (cellnumber - 1),
            Ye=pl.Series(cellye),
        )
        modelmeta["npts_model"] = ncells
        (modeldir / "model.txt").unlink()
        at.inputmodel.save_modeldata(dfcells, outpath=modeldir, modelmeta=modelmeta)

        abundfields = (modelpath / "abundances.txt").read_text(encoding="utf-8").split()
        (modeldir / "abundances.txt").write_text(
            "".join(" ".join([str(mgi + 1), *abundfields[1:]]) + "\n" for mgi in range(ncells)), encoding="utf-8"
        )

    # ARTIS writes one block for each cell of a timestep, thus each copy of the cell needs its block
    blocks: list[list[str]] = []
    for line in (modelpath / "estimators_0000.out").read_text(encoding="utf-8").splitlines():
        if line.startswith("timestep "):
            blocks.append([line, DEPOSITIONLINE])
        else:
            blocks[-1].append(line)

    (modeldir / "estimators_0000.out").write_text(
        "".join(
            line.replace("modelgridindex 0", f"modelgridindex {mgi}") + "\n"
            for block in blocks
            for mgi in range(ncells)
            for line in block
        ),
        encoding="utf-8",
    )

    return modeldir


def test_deposition_rates_read_the_next_timestep() -> None:
    """The rate of timestep n comes from the rows of n + 1, weighted by the volume of n.

    Every column of one row must cover one set of cells: a cell with no matter enters neither the
    rate nor the volume, the ion count, or the mass. A cell with no value for a channel received no
    energy in that channel. Its other channels enter the rate, and its volume stays in the denominators.
    """
    dfestim = pl.LazyFrame({
        "timestep": [0, 0, 0, 0, 1, 1, 1, 1],
        "modelgridindex": [0, 1, 2, 3, 0, 1, 2, 3],
        "tmid_days": [10.0] * 4 + [11.0] * 4,
        "nntot": [10.0, 20.0, 0.0, 5.0, 9.0, 19.0, 0.0, 4.0],
        "volume": [2.0, 4.0, 8.0, 1.0, 2.2, 4.4, 8.8, 1.1],
        "volume_prevtimestep": [None] * 4 + [2.0, 4.0, 8.0, 1.0],
        "mass_g": [3.0, 5.0, 7.0, 2.0, 3.0, 5.0, 7.0, 2.0],
        "deposition_gamma": [0.0] * 4 + [1.0e-9, 2.0e-9, 5.0e-9, None],
        "deposition_alpha": [0.0] * 4 + [1.0e-10, 0.0, 1.0e-9, 1.0e-9],
    })

    dftable = at.estimators.deposition.aggregate_deposition_rates(dfestim)

    # cell 2 holds no matter, and cell 3 has no gamma value, thus only its alpha channel enters the rate
    rate_erg_per_s = 1.1e-9 * 2.0 + 2.0e-9 * 4.0 + 1.0e-9 * 1.0
    assert dftable["timestep"].to_list() == [0]
    assert np.isclose(dftable["tmid_days"].item(), 10.0)
    assert np.isclose(dftable["dep_per_volume"].item(), rate_erg_per_s / at.constants.EV_to_erg / 7.0, rtol=1e-12)
    assert np.isclose(dftable["dep_per_ion"].item(), rate_erg_per_s / at.constants.EV_to_erg / 105.0, rtol=1e-12)
    assert np.isclose(dftable["dep_per_mass"].item(), rate_erg_per_s / at.constants.EV_to_erg / 10.0, rtol=1e-12)
    assert dftable["cellswithnorate"].item() == 0

    # one channel gives its own rate, and the ion count and the mass do not change
    dfgamma = at.estimators.deposition.aggregate_deposition_rates(dfestim, channels=["gamma"])
    assert np.isclose(
        dfgamma["dep_per_ion"].item(), (1.0e-9 * 2.0 + 2.0e-9 * 4.0) / at.constants.EV_to_erg / 105.0, rtol=1e-12
    )


def test_deposition_rates_from_the_estimator_files(tmp_path: Path) -> None:
    """The rates of a model must match a calculation that reads the estimators on its own.

    The volume of timestep 55 is 1.01 times the volume of 54, and the ion count of 55 is 0.99 times
    the count of 54. Thus this test fails if the code reads a quantity at the wrong timestep.
    """
    modeldir = make_model_with_deposition(tmp_path)
    dfestim = at.estimators.scan_estimators(modeldir, timestep=(54, 55), join_modeldata=True).collect()
    row54 = dfestim.filter(pl.col("timestep") == 54)
    # the four values of DEPOSITIONLINE, in erg/s/cm3
    rate_erg_per_s = (2.0e-10 + 1.0e-10 + 5.0e-11 + 2.5e-11) * dfestim.filter(pl.col("timestep") == 55)[
        "volume_prevtimestep"
    ].item()

    dftable = at.estimators.deposition.get_deposition_rates(modeldir, timesteps=[54])

    assert dftable["timestep"].to_list() == [54]
    assert np.isclose(
        dftable["dep_per_volume"].item(), rate_erg_per_s / at.constants.EV_to_erg / row54["volume"].item(), rtol=1e-6
    )
    assert np.isclose(
        dftable["dep_per_ion"].item(),
        rate_erg_per_s / at.constants.EV_to_erg / (row54["nntot"].item() * row54["volume"].item()),
        rtol=1e-6,
    )
    assert np.isclose(
        dftable["dep_per_mass"].item(), rate_erg_per_s / at.constants.EV_to_erg / row54["mass_g"].item(), rtol=1e-6
    )


def test_deposition_reads_a_ye_range() -> None:
    """-ye takes a range such as 0-0.2. A single number is no range, thus it gives an error."""
    assert at.estimators.deposition.parse_ye_range("0-0.2") == pytest.approx((0.0, 0.2))

    # a hyphen inside an exponent does not split the range, and a reversed range gives the same range
    assert at.estimators.deposition.parse_ye_range("0.35-1e-3") == pytest.approx((0.001, 0.35))

    for text in ("0.2", "0-0.1-0.2", "low-high"):
        with pytest.raises(ValueError, match="as a Ye range"):
            at.estimators.deposition.parse_ye_range(text)

    # a Ye is 0 to 1, thus a value outside that range, a nan, and an infinity each give an error
    for text in ("0-2", "0--1", "0-nan", "0.5-nan", "0-inf"):
        with pytest.raises(ValueError, match="names a Ye outside 0 to 1"):
            at.estimators.deposition.parse_ye_range(text)


def test_deposition_selects_the_cells_of_a_ye_range() -> None:
    """A Ye range keeps the cells inside it. The rate, the ions, the volume, and the mass follow it."""
    dfestim = pl.LazyFrame({
        "timestep": [0, 0, 1, 1],
        "modelgridindex": [0, 1, 0, 1],
        "tmid_days": [10.0, 10.0, 11.0, 11.0],
        "init_Ye": [0.15, 0.35, 0.15, 0.35],
        "nntot": [10.0, 20.0, 9.0, 19.0],
        "volume": [2.0, 4.0, 2.2, 4.4],
        "volume_prevtimestep": [None, None, 2.0, 4.0],
        "mass_g": [3.0, 5.0, 3.0, 5.0],
        "deposition_gamma": [0.0, 0.0, 1.0e-9, 2.0e-9],
    })

    dflowye = at.estimators.deposition.aggregate_deposition_rates(dfestim, yerange=(0.0, 0.2))
    rate_erg_per_s = 1.0e-9 * 2.0
    assert dflowye["timestep"].to_list() == [0]
    assert np.isclose(dflowye["dep_per_volume"].item(), rate_erg_per_s / at.constants.EV_to_erg / 2.0, rtol=1e-12)
    assert np.isclose(dflowye["dep_per_ion"].item(), rate_erg_per_s / at.constants.EV_to_erg / 20.0, rtol=1e-12)
    assert np.isclose(dflowye["dep_per_mass"].item(), rate_erg_per_s / at.constants.EV_to_erg / 3.0, rtol=1e-12)

    # a range that holds both cells gives the same rates as a selection of every cell
    dfbothcells = at.estimators.deposition.aggregate_deposition_rates(dfestim, yerange=(0.0, 0.4))
    pltest.assert_frame_equal(dfbothcells, at.estimators.deposition.aggregate_deposition_rates(dfestim))
    assert np.isclose(
        dfbothcells["dep_per_mass"].item(), (1.0e-9 * 2.0 + 2.0e-9 * 4.0) / at.constants.EV_to_erg / 8.0, rtol=1e-12
    )

    # a frame of a run that gives no Ye must give a message, and not the error of the query engine
    with pytest.raises(ValueError, match="gives no Ye of a cell"):
        at.estimators.deposition.aggregate_deposition_rates(dfestim.drop("init_Ye"), yerange=(0.0, 0.2))


def test_deposition_ye_range_needs_a_model_with_ye(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A Ye range that no cell holds must stop the command, and the message must name the Ye of the model."""
    with pytest.raises(ValueError, match="gives no Ye of a cell"):
        at.estimators.deposition.check_ye_range(modelpath, (0.0, 0.2))

    modeldir = make_model_with_deposition(tmp_path, cellye=[0.3])
    outfile = tmp_path / "deposition.txt"
    at.estimators.deposition.main(argsraw=[], modelpath=modeldir, timestep="54", ye="0.25-0.35", outputfile=outfile)

    # the one cell of the model is inside the range, thus the table gives the rate of that cell
    lines = outfile.read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith("in 1 of 1 cells with matter, with a Ye of 0.25 to 0.35")
    assert float(lines[3].split()[3]) == pytest.approx(
        at.estimators.deposition.get_deposition_rates(modeldir, [54])["dep_per_ion"].item(), rel=1e-3
    )
    assert "1 of 1 cells with matter (100.0%) have a Ye of 0.25 to 0.35" in capsys.readouterr().out

    # the one cell of the model holds matter and a Ye of 0.3, thus a range below that gets no cell
    with pytest.raises(
        ValueError, match=re.escape("has no cell with matter and a Ye of 0 to 0.2. Its cells with matter have a Ye")
    ):
        at.estimators.deposition.main(argsraw=[], modelpath=modeldir, timestep="54", ye="0-0.2")


def test_deposition_ye_range_drops_a_cell_of_a_model(tmp_path: Path) -> None:
    """A Ye range must divide the cells of a model that the reader of the estimators gives.

    The model holds two cells of a different Ye, thus a range that takes one of them must give a rate
    that is different from the rate of both cells.
    """
    modeldir = make_model_with_deposition(tmp_path, cellye=[0.15, 0.35])

    dfbothcells = at.estimators.deposition.get_deposition_rates(modeldir, [54])
    dflowye = at.estimators.deposition.get_deposition_rates(modeldir, [54], yerange=(0.1, 0.2))

    assert at.estimators.deposition.check_ye_range(modeldir, (0.1, 0.2)) == (1, 2)
    assert at.estimators.deposition.check_ye_range(modeldir, (0.1, 0.4)) == (2, 2)

    # the outer cell of a low density holds most of the volume and little of the mass. Thus the inner
    # cell alone gives a rate per unit mass that is far below the rate of both cells.
    assert dflowye["dep_per_mass"].item() < dfbothcells["dep_per_mass"].item() / 2.0

    # each cell gives the same rate per unit volume and holds the same ions, thus those two do not move
    for name in ("dep_per_volume", "dep_per_ion"):
        assert np.isclose(dflowye[name].item(), dfbothcells[name].item(), rtol=1e-6)

    # a range that holds every cell must give the rates of every cell
    pltest.assert_frame_equal(
        at.estimators.deposition.get_deposition_rates(modeldir, [54], yerange=(0.1, 0.4)), dfbothcells
    )


def test_deposition_needs_the_deposition_estimators() -> None:
    """A model that gives no rate of a cell must give an error and no number.

    artistools derives total_dep from the heating rate for such a model. On the classic-mode test
    model that total is 1400 times below the tally of deposition.out, thus it is no deposition rate.
    """
    with pytest.raises(ValueError, match="hold no deposition_ column"):
        at.estimators.deposition.main(argsraw=[], modelpath=modelpath, timestep="54")

    assert at.estimators.deposition.format_channel_list(modelpath).endswith("holds no deposition channel")


def test_deposition_names_the_channels_of_the_model() -> None:
    """An unknown channel must give a message that names the channels of the model."""
    colnames = ["total_dep", "deposition_gamma", "deposition_positron"]
    dfcell = pl.LazyFrame({"deposition_gamma": [1.0], "deposition_positron": [2.0]})

    # no channel of its own means every channel of the model
    expr = at.estimators.deposition.get_deposition_expression(colnames, None)
    assert np.isclose(dfcell.select(expr).collect().item(), 3.0)

    expr = at.estimators.deposition.get_deposition_expression(colnames, ["deposition_gamma"])
    assert np.isclose(dfcell.select(expr).collect().item(), 1.0)

    with pytest.raises(ValueError, match=re.escape("no deposition rate of alpha. It holds gamma, positron")):
        at.estimators.deposition.get_deposition_expression(colnames, ["alpha"])


def test_deposition_timeselection_takes_a_list_and_a_range() -> None:
    """-timedays and -timestep each take a list. Each item of the list means what it means elsewhere."""
    tmids = at.get_timestep_times(modelpath, loc="mid")
    timesteps = at.estimators.deposition.get_selected_timesteps(modelpath, "260,300-303", None)

    inrange = [timestep for timestep, tmid in enumerate(tmids) if 300 <= tmid <= 303]
    assert timesteps == sorted({at.get_timestep_of_timedays(modelpath, 260), *inrange})

    assert at.estimators.deposition.get_selected_timesteps(modelpath, None, "4,9-11") == [4, 9, 10, 11]
    assert at.estimators.deposition.get_selected_timesteps(modelpath, None, "last") == [len(tmids) - 1]
    assert at.estimators.deposition.get_selected_timesteps(modelpath, None, None) is None

    # a range that holds no timestep gives an error, and not a selection that quietly loses it
    with pytest.raises(ValueError, match="does not include any full timesteps"):
        at.estimators.deposition.get_selected_timesteps(modelpath, "303-300,260", None)


def test_deposition_writes_a_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The command writes the table that it prints. It also reports each timestep with no rate."""
    modeldir = make_model_with_deposition(tmp_path)
    outfile = tmp_path / "deposition.txt"
    at.estimators.deposition.main(argsraw=[], modelpath=modeldir, timestep="54,last", outputfile=outfile)

    lines = outfile.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith(at.get_model_name(modeldir))
    assert lines[1].split() == ["timestep", "t_days", "dep_per_volume", "dep_per_ion", "dep_per_mass"]
    assert lines[2].split() == ["[d]", "[eV/s/cm^3]", "[eV/s]", "[eV/s/g]"]

    values = [float(value) for value in lines[3].split()]
    assert len(lines) == 4, "the last timestep gets no rate, thus the table holds one row"
    assert values[0] == pytest.approx(54)
    # the table gives four digits, thus the value in the file has less precision
    assert np.isclose(
        values[3], at.estimators.deposition.get_deposition_rates(modeldir, [54])["dep_per_ion"].item(), rtol=1e-3
    )

    # the file of the next timestep holds the rate, thus the timestep that -timestep last names gives none
    assert "no deposition rate for timestep 99" in capsys.readouterr().err

    # a selection of every timestep always loses that one row, thus it must give no warning
    at.estimators.deposition.main(argsraw=[], modelpath=modeldir)
    assert "no deposition rate for" not in capsys.readouterr().err


def test_deposition_lists_the_channels(tmp_path: Path) -> None:
    """--listchannels must name the channels of the model, and write them when -o asks."""
    modeldir = make_model_with_deposition(tmp_path)
    outfile = tmp_path / "channels.txt"
    at.estimators.deposition.main(argsraw=[], modelpath=modeldir, listchannels=True, outputfile=outfile)

    assert outfile.read_text(encoding="utf-8").strip().endswith("holds alpha, electron, gamma, positron")


def test_line_points_leave_a_cell_with_no_value_out_of_the_average() -> None:
    """A null value shows that the cell reported no value, thus its weight must not decrease the average."""
    import argparse

    from artistools.estimators.plotestimators import get_line_points

    dfseries = pl.LazyFrame({
        "xvalue_binned": [1.0, 1.0, 1.0],
        "yvalue": [10.0, None, 10.0],
        "celltsweight": [1.0, 8.0, 1.0],
    })

    dflinepoints = get_line_points(dfseries, argparse.Namespace()).collect()

    assert np.isclose(dflinepoints["yvalue_binned"].item(), 10.0)


def test_add_derived_estimator_columns_fills_absent_channels_with_zero() -> None:
    """A deposition, heating, or cooling channel that a rank omitted is zero, and a heating ratio keeps its null."""
    pldf = pl.LazyFrame({
        "timestep": [0, 1],
        "modelgridindex": [0, 1],
        "deposition_gamma": [1.0, None],
        "heating_gamma": [1.0, 2.0],
        "heating_dep": [None, 2.0],
        "cooling_ff": [None, 1.0],
        "heating_gamma/gamma_dep": [0.5, None],
        "Te": [5000.0, None],
    })

    dfout = at.estimators.add_derived_estimator_columns(pldf).collect()

    assert dfout["deposition_gamma"].to_list() == [1.0, 0.0]
    assert dfout["heating_dep"].to_list() == [0.0, 2.0]
    assert dfout["cooling_ff"].to_list() == [0.0, 1.0]
    # a ratio of zero makes gamma_dep a division by zero, thus a ratio keeps its null
    assert dfout["heating_gamma/gamma_dep"].to_list() == [0.5, None]
    assert dfout["Te"].to_list() == [5000.0, None]
