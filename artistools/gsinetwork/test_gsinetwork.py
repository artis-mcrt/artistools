import math
import typing as t
from unittest import mock

import matplotlib.axes as mplax
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_decayproducts(mockplot: t.Any) -> None:
    trajpath = at.get_path("testdata") / "kilonova" / "trajectories"
    at.gsinetwork.decayproducts.main(
        argsraw=[], trajectoryroot=trajpath, tmin=0.1, tmax=0.1, nsteps=1, outputpath=outputpath
    )

    expected_y_arrays = [
        [0.45123906],
        [0.20325522],
        [0.34550572],
        [6.40901247e39],
        [5.58169828e39],
        [5.76986032e39],
        [0.43677441],
        [0.21151689],
        [0.35170869],
        [6.03183992e39],
        [5.3560247e39],
        [5.54429711e39],
        [0.79453598],
        [0.00717668],
        [0.19828734],
        [3.77172547e38],
        [2.25673577e38],
        [2.25563208e38],
    ]
    for x, expected_y_arr in zip(mockplot.call_args_list, expected_y_arrays, strict=True):
        x_arr = x[0][1]
        assert len(x_arr) == 1
        assert math.isclose(x_arr[0], 0.1, rel_tol=1e-3)
        y_arr = x[0][2]
        assert len(y_arr) == 1
        assert math.isclose(y_arr[0], expected_y_arr[0], rel_tol=1e-3)
