import math
import typing as t
from unittest import mock

import matplotlib.axes as mplax
import pytest
from pytest_codspeed.plugin import BenchmarkFixture

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmodel_3d"
outputpath = at.get_path("testoutput")


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_decayproducts(mockplot: t.Any, benchmark: BenchmarkFixture) -> None:
    trajpath = at.get_path("testdata") / "kilonova" / "trajectories"
    at.gsinetwork.decayproducts.main(
        argsraw=[], trajectoryroot=trajpath, tmin=0.1, tmax=0.1, nsteps=1, outputpath=outputpath
    )

    expected_y_arrays = [
        [0.41982291048506115],
        [0.2175310498483443],
        [0.36264603966659464],
        [6.409012468260871e39],
        [5.588677239121782e39],
        [5.594888051331457e39],
        [0.4199871052754964],
        [0.2174383105917629],
        [0.36257458413274074],
        [6.0318399209796e39],
        [5.583699808922472e39],
        [5.589910621132147e39],
        [0.23562858062657116],
        [0.32156629464689707],
        [0.44280512472653166],
        [3.7717254728127e38],
        [4.9774301993101926e36],
        [4.9774301993101914e36],
    ]
    for x, expected_y_arr in zip(mockplot.call_args_list, expected_y_arrays, strict=True):
        x_arr = x[0][1]
        assert len(x_arr) == 1
        assert math.isclose(x_arr[0], 0.1, rel_tol=1e-3)
        y_arr = x[0][2]
        assert len(y_arr) == 1
        assert math.isclose(y_arr[0], expected_y_arr[0], rel_tol=1e-3)
