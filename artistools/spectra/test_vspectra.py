#!/usr/bin/env python3
from unittest import mock

import matplotlib.axes
import numpy as np

import artistools as at

modelpath = at.get_config()["path_testartismodel"].parent / "vspecpolmodel"
outputpath = at.get_config()["path_testoutput"]


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_vspectraplot(mockplot):
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, "sn2011fe_PTF11kly_20120822_norm.txt"],
        outputfile=outputpath,
        plotvspecpol=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        timemin=10,
        timemax=12,
    )

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    assert all(np.array_equal(arr_time_d, np.array(mockplot.call_args_list[vspecdir][0][1])) for vspecdir in range(10))

    arr_allvspec = np.vstack([np.array(mockplot.call_args_list[vspecdir][0][2]) for vspecdir in range(10)])
    assert np.allclose(
        arr_allvspec.std(axis=1),
        np.array(
            [
                2.01529689e-12,
                2.05807110e-12,
                2.01551623e-12,
                2.18216916e-12,
                2.85477069e-12,
                3.34384407e-12,
                2.94892344e-12,
                2.29084411e-12,
                2.05916843e-12,
                2.00515984e-12,
            ]
        ),
    )
