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
        [0.4365851076441083],
        [0.2116200671053596],
        [0.3517948252505321],
        [6.409012468260871e39],
        [5.360947579017793e39],
        [5.549212674883347e39],
        [0.4367744110691814],
        [0.21151689440843421],
        [0.3517086945223843],
        [6.0318399209796e39],
        [5.3560247024277675e39],
        [5.54429711145833e39],
        [0.23062548172129188],
        [0.323870598525968],
        [0.4455039197527402],
        [3.7717254728127e38],
        [4.9228765900252654e36],
        [4.9155634250168247e36],
    ]
    for x, expected_y_arr in zip(mockplot.call_args_list, expected_y_arrays, strict=True):
        x_arr = x[0][1]
        assert len(x_arr) == 1
        assert math.isclose(x_arr[0], 0.1, rel_tol=1e-3)
        y_arr = x[0][2]
        assert len(y_arr) == 1
        assert math.isclose(y_arr[0], expected_y_arr[0], rel_tol=1e-3)
