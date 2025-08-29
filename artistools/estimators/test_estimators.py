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
