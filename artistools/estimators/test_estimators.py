import shutil
from pathlib import Path
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_classic_3d = at.get_config()["path_testdata"] / "test-classicmode_3d"
outputpath = Path(at.get_config()["path_testoutput"])


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_estimator_snapshot(mockplot, benchmark) -> None:
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
        assert xarr == x[0][1]

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
def test_estimator_averaging(mockplot, benchmark) -> None:
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
        assert xarr == x[0][1]

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
def test_estimator_snapshot_classic_3d(mockplot) -> None:
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
        timedays=4,
    )

    # order of keys is important
    expectedvals = {
        "init_fe": 0.039081018094319186,
        "init_nistable": 0.026971354386497073,
        "init_ni56": 0.1741285660514124,
        "nne": 62629131723.8786,
        "TR": 24081.788509728773,
        "Te": 55260.50862323113,
        "averageionisation_Fe": 3.183501818759448,
        "populations_FeI": 2.6829624903107585e-16,
        "populations_FeII": 2.1309973094140065e-05,
        "populations_FeIII": 0.11156867875483492,
        "populations_FeIV": 0.5932461437623715,
        "populations_FeV": 0.29515117944382513,
        "populations_CoII": 0.19311608717270304,
        "populations_CoIII": 0.17864127106393107,
        "populations_CoIV": 0.6282359982199368,
        "heating_dep": 2.281243040310174e-06,
        "heating_coll": 0.000179854801522111,
        "heating_bf": 1.8349749858387718e-06,
        "heating_ff": 4.5324851021111127e-10,
        "cooling_adiabatic": 1.1138047075438075e-10,
        "cooling_coll": 3.687992746609425e-05,
        "cooling_fb": 8.13701682676968e-08,
        "cooling_ff": 5.604801913020247e-10,
    }

    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {
        varname: np.array(callargs[0][2]).mean()
        for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list, strict=False)
    }

    print(yvals)

    for varname, expectedval in expectedvals.items():
        assert np.allclose(expectedval, yvals[varname], rtol=0.001), (varname, expectedval, yvals[varname])


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
def test_estimator_snapshot_classic_3d_x_axis(mockplot) -> None:
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
def test_estimator_timeevolution(mockplot) -> None:
    at.estimators.plot(
        argsraw=[],
        modelpath=modelpath,
        outputfile=outputpath / "test_estimator_timeevolution",
        plotlist=[["Te", "nne"]],
        modelgridindex=0,
        x="time",
    )


@pytest.mark.benchmark
def test_rust_estimator_parser() -> None:
    test_outputpath = outputpath / "test_rust_estimator_parser"
    dfestimators: list[pl.DataFrame] = []
    for use_rust in (True, False):
        if test_outputpath.exists():
            shutil.rmtree(test_outputpath)
            test_outputpath.mkdir(exist_ok=True)
        shutil.copytree(modelpath, test_outputpath, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*parquet*"))
        dfestimators.append(
            at.estimators.scan_estimators(modelpath=test_outputpath, use_rust_parser=use_rust).collect()
        )

    pltest.assert_frame_equal(
        dfestimators[0],
        dfestimators[1],
        rtol=1e-4,
        atol=1e-4,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )
