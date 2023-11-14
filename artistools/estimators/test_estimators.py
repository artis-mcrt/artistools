from unittest import mock

import matplotlib.axes
import numpy as np

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_classic_3d = at.get_config()["path_testdata"] / "test-classicmode_3d"
outputpath = at.get_config()["path_testoutput"]


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_estimator_snapshot(mockplot) -> None:
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
    ]

    at.estimators.plot(argsraw=[], modelpath=modelpath, plotlist=plotlist, outputfile=outputpath, timedays=300)
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
    }
    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list)}

    print({key: yarr[1] for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose([expectedval, expectedval], yvals[varname], rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
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

    at.estimators.plot(argsraw=[], modelpath=modelpath_classic_3d, plotlist=plotlist, outputfile=outputpath, timedays=4)

    # order of keys is important
    expectedvals = {
        "init_fe": 0.015840992399058583,
        "init_nistable": 0.009782246630960749,
        "init_ni56": 0.05254947310106698,
        "nne": 15470013214.969236,
        "TR": 19133.550905730128,
        "Te": 71225.65831792976,
        "averageionisation_Fe": 3.057499505875313,
        "populations_FeI": 5.352271448040245e-16,
        "populations_FeII": 0.00019316689981412616,
        "populations_FeIII": 0.0680237830807816,
        "populations_FeIV": 0.8058102730120427,
        "populations_FeV": 0.12595698844445202,
        "populations_CoII": 0.16959198795238514,
        "populations_CoIII": 0.24870748781782487,
        "populations_CoIV": 0.58169689937825,
        "heating_dep": 2.554384679580454e-06,
        "heating_coll": 0.00021143743068391865,
        "heating_bf": 2.1706211346541803e-06,
        "heating_ff": 5.577364060934719e-10,
        "cooling_adiabatic": 1.2856078494705232e-10,
        "cooling_coll": 4.343953996625696e-05,
        "cooling_fb": 9.58727795462437e-08,
        "cooling_ff": 6.657026859049402e-10,
    }

    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list)}

    print({key: np.array(yarr).mean() for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose(expectedval, np.array(yvals[varname]).mean(), rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
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
        outputfile=outputpath,
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
    yvals = {varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list)}

    print({key: np.array(yarr).mean() for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose(expectedval, np.array(yvals[varname]).mean(), rtol=0.001), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_estimator_timeevolution(mockplot) -> None:
    at.estimators.plot(
        argsraw=[], modelpath=modelpath, outputfile=outputpath, plotlist=[["Te", "nne"]], modelgridindex=0, x="time"
    )
