import re
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
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
    expected_yvals_mean = {
        "init_fe": 0.015787530690431595,
        "init_nistable": 0.009560450911521912,
        "init_ni56": 0.04967936500906944,
        "nne": 14232720384.0,
        "TR": 19025.818359375,
        "Te": 71311.2109375,
        "averageionisation_Fe": 3.054003953933716,
        "populations_FeI": 5.372131767415029e-16,
        "populations_FeII": 0.0001938836503541097,
        "populations_FeIII": 0.06827619671821594,
        "populations_FeIV": 0.8087993860244751,
        "populations_FeV": 0.12271492183208466,
        "populations_CoII": 0.1702212691307068,
        "populations_CoIII": 0.24963033199310303,
        "populations_CoIV": 0.5801447629928589,
        "heating_dep": 2.5638628358137794e-06,
        "heating_coll": 0.0002122219739248976,
        "heating_bf": 2.178675231334637e-06,
        "heating_ff": 5.598059793499033e-10,
        "cooling_adiabatic": 1.2903782209416903e-10,
        "cooling_coll": 4.360072853160091e-05,
        "cooling_fb": 9.622852559232342e-08,
        "cooling_ff": 6.681727948709693e-10,
    }

    expected_yvals_std = {
        "init_fe": 0.03867174685001373,
        "init_nistable": 0.024418460205197334,
        "init_ni56": 0.13267292082309723,
        "nne": 52205641728.0,
        "TR": 8704.7080078125,
        "Te": 53293.2578125,
        "averageionisation_Fe": 0.3648064434528351,
        "populations_FeI": 1.0559215211458726e-14,
        "populations_FeII": 0.003967594355344772,
        "populations_FeIII": 0.2206096053123474,
        "populations_FeIV": 0.3149721026420593,
        "populations_FeV": 0.25867846608161926,
        "populations_CoII": 0.36867186427116394,
        "populations_CoIII": 0.3848763406276703,
        "populations_CoIV": 0.45789873600006104,
        "heating_dep": 2.4430109988315962e-05,
        "heating_coll": 0.0047865696251392365,
        "heating_bf": 4.846786396228708e-05,
        "heating_ff": 3.555698846469113e-09,
        "cooling_adiabatic": 1.2155411122094506e-09,
        "cooling_coll": 0.0009426323231309652,
        "cooling_fb": 2.1289438336680178e-06,
        "cooling_ff": 7.294685744341223e-09,
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
    expectedvals = {
        "init_fe": 0.011052947585195368,
        "init_nistable": 0.000944194626933764,
        "init_ni56": 0.002896747941237337,
        "nne": 382033722.1422282,
        "TR": 19732.04,
        "Te": 47127.520000000004,
        "averageionisation_Fe": 3.0271734010069435,
        "populations_FeI": 6.5617829754545176e-24,
        "populations_FeII": 3.161551652102325e-13,
        "populations_FeIII": 0.00010731048012085833,
        "populations_FeIV": 0.9728187853219049,
        "populations_FeV": 0.027125606020167697,
        "populations_CoII": 0.20777361030622207,
        "populations_CoIII": 0.22753057860431092,
        "populations_CoIV": 0.5646079825984672,
        "heating_dep": 5.879422739895874e-08,
        "heating_coll": 0.0,
        "heating_bf": 8.988080000000003e-16,
        "heating_ff": 4.492620000000028e-18,
        "cooling_adiabatic": 1.9406654213040002e-14,
        "cooling_coll": 2.1374800003106965e-14,
        "cooling_fb": 3.376760000131059e-17,
        "cooling_ff": 1.3946640000041897e-17,
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
        ("gamma_R        Z=26  12: 1.0", "no roman numeral for ion stage 12"),
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

    dfavgexc = at.estimators.get_averageexcitation(modelpath, 26, 2, dftexc).collect()
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
    parquetfilepath = at.estimators.estimators.get_rankbatch_parquetpath(tmp_path, [0, 1, 2], 0)
    assert parquetfilepath.name == "estimbatch00_0000_0002.out.parquet.tmp"

    # a cache that no run wrote yet needs the conversion
    assert not at.estimators.estimators.rankbatch_parquet_is_current(parquetfilepath, None)

    parquetfilepath.write_bytes(b"")
    mtime = parquetfilepath.stat().st_mtime
    assert at.estimators.estimators.rankbatch_parquet_is_current(parquetfilepath, None)
    assert at.estimators.estimators.rankbatch_parquet_is_current(parquetfilepath, mtime - 10.0)

    # an estimator text file that is newer than the cache needs the conversion again
    assert not at.estimators.estimators.rankbatch_parquet_is_current(parquetfilepath, mtime + 10.0)


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


@pytest.mark.parametrize("timestep", [11, "11"])
def test_estimator_keyword_timestep_reads_an_int(timestep: int | str) -> None:
    """A keyword argument gives an int, and a command line gives a string. Both name one timestep."""
    at.estimators.plot(argsraw=[], modelpath=modelpath, outputfile=outputpath, plotlist=[["rho"]], timestep=timestep)


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
