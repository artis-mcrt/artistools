from unittest import mock

import matplotlib.axes
import numpy as np

import artistools as at

modelpath = at.get_config()["path_testartismodel"]
outputpath = at.get_config()["path_testoutput"]

plotlist = [
    [["initabundances", ["Fe", "Ni_stable", "Ni_56"]]],
    ["heating_dep", "heating_coll", "heating_bf", "heating_ff", ["_yscale", "linear"]],
    ["cooling_adiabatic", "cooling_coll", "cooling_fb", "cooling_ff", ["_yscale", "linear"]],
    ["nne"],
    ["TR", ["_yscale", "linear"], ["_ymin", 1000], ["_ymax", 22000]],
    ["Te"],
    [["averageionisation", ["Fe", "Ni"]]],
    [["averageexcitation", ["Fe II", "Fe III"]]],
    [["populations", ["Fe I", "Fe II", "Fe III", "Fe IV", "Fe V", "Fe VI", "Fe VII", "Fe VIII"]]],
    [["populations", ["Co I", "Co II", "Co III", "Co IV", "Co V", "Co VI", "Co VII"]]],
    [["populations", ["Ni I", "Ni II", "Ni III", "Ni IV", "Ni V", "Ni VI", "Ni VII"]]],
    [["gamma_NT", ["Fe I", "Fe II", "Fe III", "Fe IV"]]],
]


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_estimator_snapshot(mockplot) -> None:
    at.estimators.plot(argsraw=[], modelpath=modelpath, plotlist=plotlist, outputfile=outputpath, timedays=300)
    xarr = [0.0, 4000.0]
    for x in mockplot.call_args_list:
        assert xarr == x[0][1]
    yvals = [callargs[0][2] for callargs in mockplot.call_args_list]
    init_fe = [0.10000000149011612, 0.1000000014901161]
    assert np.allclose(init_fe, yvals[0], rtol=0.001)
    init_nistable = [0, 0.0]
    assert np.allclose(init_nistable, yvals[1], rtol=0.001)
    init_ni56 = [0.8999999761581421, 0.8999999761581421]
    assert np.allclose(init_ni56, yvals[2], rtol=0.001)

    print("\n".join(str(x) for x in yvals))


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_estimator_timeevolution(mockplot) -> None:
    at.estimators.plot(
        argsraw=[], modelpath=modelpath, outputfile=outputpath, plotlist=[["Te", "nne"]], modelgridindex=0, x="time"
    )
