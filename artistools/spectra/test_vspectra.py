#!/usr/bin/env python3
from unittest import mock

import matplotlib.axes as mplax
import numpy as np
import pytest

import artistools as at


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_vspectraplot(mockplot, benchmark) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=[at.get_config()["path_testdata"] / "vspecpolmodel", "sn2011fe_PTF11kly_20120822_norm.txt"],
        outputfile=at.get_config()["path_testoutput"] / "test_vspectra.pdf",
        plotvspecpol=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        timemin=11,
        timemax=12,
    )

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    assert all(np.array_equal(arr_time_d, np.array(mockplot.call_args_list[vspecdir][0][1])) for vspecdir in range(10))

    arr_allvspec = np.vstack([np.array(mockplot.call_args_list[vspecdir][0][2]) for vspecdir in range(10)])

    assert np.allclose(
        arr_allvspec.std(axis=1),
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
        ],
        atol=1e-16,
    )

    assert np.allclose(
        arr_allvspec.mean(axis=1),
        [
            2.9864681492951925e-12,
            3.0063451037690416e-12,
            2.9785924608537284e-12,
            3.2028094816751935e-12,
            4.097482117229833e-12,
            4.663450168092402e-12,
            4.231106733071208e-12,
            3.350080172063692e-12,
            3.0234533505898177e-12,
            2.9721539798925583e-12,
        ],
        atol=1e-16,
    )


@mock.patch.object(mplax.Axes, "plot", side_effect=mplax.Axes.plot, autospec=True)
@pytest.mark.benchmark
def test_vpkt_frompackets_spectrum_plot(mockplot, benchmark) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=[at.get_config()["path_testdata"] / "vpktcontrib"],
        outputfile=at.get_config()["path_testoutput"] / "test_vpktscontrib_spectra.pdf",
        plotvspecpol=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        frompackets=True,
        maxpacketfiles=2,
        timemin=130,
        timemax=135,
    )

    arr_time_d = np.array(mockplot.call_args_list[0][0][1])
    assert all(np.array_equal(arr_time_d, np.array(mockplot.call_args_list[vspecdir][0][1])) for vspecdir in range(9))

    arr_allvspec = np.vstack([np.array(mockplot.call_args_list[vspecdir][0][2]) for vspecdir in range(9)])
    print(list(arr_allvspec.std(axis=1)))
    assert np.allclose(
        arr_allvspec.std(axis=1),
        [
            2.250169934569213e-15,
            1.5100666673940305e-14,
            4.869648743602329e-15,
            2.2088584226201254e-15,
            1.4772661475549563e-14,
            4.81122573011189e-15,
            2.3113250841019364e-15,
            1.45676358684038e-14,
            4.589891035117792e-15,
        ],
        atol=1e-18,
    )

    print(list(arr_allvspec.mean(axis=1)))
    assert np.allclose(
        arr_allvspec.mean(axis=1),
        [
            1.1359234428755927e-15,
            9.615508017438282e-15,
            2.959189512289016e-15,
            1.1078180792141745e-15,
            9.453979816323599e-15,
            2.981925804875244e-15,
            1.1359823034815934e-15,
            9.316807974058103e-15,
            2.9236313925953637e-15,
        ],
        atol=1e-18,
    )
