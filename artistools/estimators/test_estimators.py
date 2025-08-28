import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import pytest
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_classic_3d = at.get_config()["path_testdata"] / "test-classicmode_3d"
outputpath = Path(at.get_config()["path_testoutput"])


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_estimator_snapshot(mockplot: t.Any, benchmark: BenchmarkFixture) -> None:
    plotlist = [
        [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
        ["nne"],
        ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
        ["Te"],
        [["averageionisation", ["Fe", "Ni"]]],
        [["averageexcitation", ["Fe II"]]],
        [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
        [["populations", ["Co II", "Co III", "Co IV"]]],
        [["gamma_NT", ["Fe I", "Fe II", "Fe III", "Fe IV"]]],
        ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
        ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
        [(pl.col("heating_coll") - pl.col("cooling_coll")).alias("collisional heating - cooling")],
    ]

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
        "averageexcitation_FeII": 0.19701832980731157,
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
def test_estimator_averaging(mockplot: t.Any, benchmark: BenchmarkFixture) -> None:
    plotlist = [
        [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
        ["nne"],
        ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
        ["Te"],
        [["averageionisation", ["Fe", "Ni"]]],
        [["averageexcitation", ["Fe II"]]],
        [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
        [["populations", ["Co II", "Co III", "Co IV"]]],
        [["gamma_NT", ["Fe I", "Fe II", "Fe III", "Fe IV"]]],
        ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
        ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
        [(pl.col("heating_coll") - pl.col("cooling_coll")).alias("collisional heating - cooling")],
    ]

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
        "averageexcitation_FeII": 0.1975447074846265,
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
def test_estimator_snapshot_classic_3d(mockplot: t.Any) -> None:
    plotlist = [
        [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
        ["nne"],
        ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
        ["Te"],
        [["averageionisation", ["Fe"]]],
        [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
        [["populations", ["Co II", "Co III", "Co IV"]]],
        ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
        ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
    ]

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot_classic_3d",
        xbins=-1,
        timedays=4,
    )

    # order of keys is important
    expected_yvals_mean = {
        "init_fe": 0.0158143120149385,
        "init_nistable": 0.009671554499267782,
        "init_ni56": 0.05111707578637089,
        "nne": 14852513100.433994,
        "TR": 19079.784339735244,
        "Te": 71268.35864076967,
        "averageionisation_Fe": 3.055755031831655,
        "populations_FeI": 5.362183140852601e-16,
        "populations_FeII": 0.00019352461074175688,
        "populations_FeIII": 0.06814975391081575,
        "populations_FeIV": 0.8073020624301706,
        "populations_FeV": 0.12433895472009603,
        "populations_CoII": 0.16990604721796168,
        "populations_CoIII": 0.24916805418700844,
        "populations_CoIV": 0.5809222634054445,
        "heating_dep": 2.559115013945374e-06,
        "heating_coll": 0.00021182897897368226,
        "heating_bf": 2.1746407480448304e-06,
        "heating_ff": 5.58769247608382e-10,
        "cooling_adiabatic": 1.2879886096317952e-10,
        "cooling_coll": 4.3519983202307406e-05,
        "cooling_fb": 9.605032219254008e-08,
        "cooling_ff": 6.669354708218202e-10,
    }

    expected_yvals_std = {
        "init_fe": 0.03864092363074494,
        "init_nistable": 0.0245318237219884,
        "init_ni56": 0.13668807807395486,
        "nne": 54105764830.790764,
        "TR": 8786.430933187814,
        "Te": 53253.18563238727,
        "averageionisation_Fe": 0.36672845223208933,
        "populations_FeI": 1.0549459114516364e-14,
        "populations_FeII": 0.003963927962373676,
        "populations_FeIII": 0.22042479123325878,
        "populations_FeIV": 0.31659457250903567,
        "populations_FeV": 0.2611747334195758,
        "populations_CoII": 0.368403054302235,
        "populations_CoIII": 0.3846695608564221,
        "populations_CoIV": 0.45783056031765657,
        "heating_dep": 2.4407727400567926e-05,
        "heating_coll": 0.004782144675055231,
        "heating_bf": 4.842305588498566e-05,
        "heating_ff": 3.5524865990710362e-09,
        "cooling_adiabatic": 1.2144276845831258e-09,
        "cooling_coll": 0.0009417610247237612,
        "cooling_fb": 2.1269753993513636e-06,
        "cooling_ff": 7.287984018219307e-09,
    }

    assert len(expected_yvals_mean) == len(mockplot.call_args_list)

    yvals_mean = {
        varname: float(np.array(callargs[0][2]).mean())
        for varname, callargs in zip(expected_yvals_mean.keys(), mockplot.call_args_list, strict=False)
    }
    print(yvals_mean)

    yvals_std = {
        varname: float(np.array(callargs[0][2]).std())
        for varname, callargs in zip(expected_yvals_std.keys(), mockplot.call_args_list, strict=False)
    }
    print(yvals_std)

    for varname, expectedmean in expected_yvals_mean.items():
        assert np.allclose(expectedmean, yvals_mean[varname], rtol=0.001), (varname, expectedmean, yvals_mean[varname])
    for varname, expectedstd in expected_yvals_std.items():
        assert np.allclose(expectedstd, yvals_std[varname], rtol=0.001), (varname, expectedstd, yvals_std[varname])


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_snapshot_classic_3d_x_axis(mockplot: t.Any) -> None:
    plotlist = [
        [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
        ["nne"],
        ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
        ["Te"],
        [["averageionisation", ["Fe"]]],
        [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V"]]],
        [["populations", ["Co II", "Co III", "Co IV"]]],
        ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
        ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
    ]

    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath_classic_3d,
        plotlist=plotlist,
        outputfile=outputpath / "test_estimator_snapshot_classic_3d_x_axis",
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

    print({key: np.array(yarr).mean() for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose(expectedval, np.array(yvals[varname]).mean(), rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_estimator_timeevolution(mockplot: t.Any) -> None:
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath / "test_estimator_timeevolution",
        plotlist=[["Te", "nne"]],
        modelgridindex=0,
        x="time",
    )
