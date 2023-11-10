from unittest import mock

import matplotlib.axes
import numpy as np

import artistools as at

modelpath = at.get_config()["path_testartismodel"]
outputpath = at.get_config()["path_testoutput"]

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


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_estimator_snapshot(mockplot) -> None:
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
def test_estimator_averaging(mockplot) -> None:
    at.estimators.plot(argsraw=[], modelpath=modelpath, plotlist=plotlist, outputfile=outputpath, timestep="50-54")
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
        "averageexcitation_FeII": float("nan"),
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
    }
    assert len(expectedvals) == len(mockplot.call_args_list)
    yvals = {varname: callargs[0][2] for varname, callargs in zip(expectedvals.keys(), mockplot.call_args_list)}

    print({key: yarr[1] for key, yarr in yvals.items()})

    for varname, expectedval in expectedvals.items():
        assert np.allclose([expectedval, expectedval], yvals[varname], rtol=0.001, equal_nan=True), (
            varname,
            expectedval,
            yvals[varname][1],
        )


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_estimator_timeevolution(mockplot) -> None:
    at.estimators.plot(
        argsraw=[], modelpath=modelpath, outputfile=outputpath, plotlist=[["Te", "nne"]], modelgridindex=0, x="time"
    )
