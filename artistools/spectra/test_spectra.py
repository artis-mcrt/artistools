#!/usr/bin/env python3
import math
from pathlib import Path
from unittest import mock

import matplotlib.axes
import numpy as np
import pandas as pd

import artistools as at

modelpath = at.get_config()["path_testartismodel"]
outputpath = at.get_config()["path_testoutput"]
modelpath_classic_3d = at.get_config()["path_testdata"] / "test-classicmode_3d"


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_spectraplot(mockplot) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=[modelpath, "sn2011fe_PTF11kly_20120822_norm.txt"],
        outputfile=outputpath,
        timemin=290,
        timemax=320,
    )

    arr_lambda = np.array(mockplot.call_args[0][1])
    arr_f_lambda = np.array(mockplot.call_args[0][2])

    integral = np.trapz(y=arr_f_lambda, x=arr_lambda)
    assert np.isclose(integral, 5.870730903198916e-11, atol=1e-14)


@mock.patch.object(matplotlib.axes.Axes, "plot", side_effect=matplotlib.axes.Axes.plot, autospec=True)
def test_spectra_frompackets(mockplot) -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=Path(outputpath, "spectrum_from_packets.pdf"),
        timemin=290,
        timemax=320,
        frompackets=True,
    )

    arr_lambda = np.array(mockplot.call_args[0][1])
    arr_f_lambda = np.array(mockplot.call_args[0][2])

    integral = np.trapz(y=arr_f_lambda, x=arr_lambda)

    assert np.isclose(integral, 7.7888e-12, rtol=1e-3)


def test_spectra_outputtext() -> None:
    at.spectra.plot(argsraw=[], specpath=modelpath, output_spectra=True)


def test_spectraemissionplot() -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=outputpath,
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        use_thermalemissiontype=True,
    )


def test_spectraemissionplot_nostack() -> None:
    at.spectra.plot(
        argsraw=[],
        specpath=modelpath,
        outputfile=outputpath,
        timemin=290,
        timemax=320,
        emissionabsorption=True,
        nostack=True,
        use_thermalemissiontype=True,
    )


def test_spectra_get_spectrum() -> None:
    def check_spectrum(dfspectrumpkts) -> None:
        assert math.isclose(max(dfspectrumpkts["f_lambda"]), 2.548532804918824e-13, abs_tol=1e-5)
        assert min(dfspectrumpkts["f_lambda"]) < 1e-9
        assert math.isclose(np.mean(dfspectrumpkts["f_lambda"]), 1.0314682640070206e-14, abs_tol=1e-5)

    dfspectrum = at.spectra.get_spectrum(modelpath, 55, 65, fnufilterfunc=None)[-1]
    assert len(dfspectrum["lambda_angstroms"]) == 1000
    assert len(dfspectrum["f_lambda"]) == 1000
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[-1] - 29920.601421214415) < 1e-5
    assert abs(dfspectrum["lambda_angstroms"].to_numpy()[0] - 600.75759482509852) < 1e-5

    check_spectrum(dfspectrum)

    lambda_min = dfspectrum["lambda_angstroms"].to_numpy()[0]
    lambda_max = dfspectrum["lambda_angstroms"].to_numpy()[-1]
    timelowdays = at.get_timestep_times(modelpath)[55]
    timehighdays = at.get_timestep_times(modelpath)[65]

    dfspectrumpkts = at.spectra.get_from_packets(
        modelpath, timelowdays=timelowdays, timehighdays=timehighdays, lambda_min=lambda_min, lambda_max=lambda_max
    )[-1]

    check_spectrum(dfspectrumpkts)


def test_spectra_get_spectrum_polar_angles() -> None:
    kwargs = {
        "modelpath": modelpath_classic_3d,
        "directionbins": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "average_over_phi": True,
    }

    spectra = at.spectra.get_spectrum(**kwargs, timestepmin=20, timestepmax=25)

    assert all(np.isclose(dirspec["lambda_angstroms"].mean(), 7510.074, rtol=1e-3) for dirspec in spectra.values())
    assert all(np.isclose(dirspec["lambda_angstroms"].std(), 7647.317, rtol=1e-3) for dirspec in spectra.values())

    expected_results = {
        0: (8.944885683622777e-12, 2.5390561316336613e-11),
        10: (7.192449910173842e-12, 2.0713405870496142e-11),
        20: (8.963182635824623e-12, 2.4720178744713477e-11),
        30: (8.06805028771611e-12, 2.2672897557383406e-11),
        40: (7.8306536944195e-12, 2.2812958326863807e-11),
        50: (8.259135507460651e-12, 2.2795973908331984e-11),
        60: (7.964029031817186e-12, 2.637892822134082e-11),
        70: (7.691392868658026e-12, 2.1262113332060223e-11),
        80: (8.450665096838155e-12, 2.352725654000879e-11),
        90: (8.828105146277665e-12, 2.534549767123003e-11),
    }

    results = {
        dirbin: (
            dfspecdir["f_lambda"].mean(),
            dfspecdir["f_lambda"].std(),
        )
        for dirbin, dfspecdir in spectra.items()
    }

    print(f"expected_results = {results!r}")

    for dirbin in spectra:
        assert results[dirbin] == expected_results[dirbin]

    # lambda_min = spectra[0]["lambda_angstroms"].to_numpy()[0]
    # lambda_max = spectra[0]["lambda_angstroms"].to_numpy()[-1]
    # timelowdays = at.get_timestep_times(modelpath, loc="start")[20]
    # timehighdays = at.get_timestep_times(modelpath, loc="end")[25]

    # spectrafrompkts = at.spectra.get_from_packets(
    #     **kwargs, timelowdays=timelowdays, timehighdays=timehighdays, lambda_min=lambda_min, lambda_max=lambda_max
    # )

    # results_pkts = {
    #     dirbin: (
    #         dfspecdir["f_lambda"].mean(),
    #         dfspecdir["f_lambda"].std(),
    #     )
    #     for dirbin, dfspecdir in spectrafrompkts.items()
    # }

    # print(f"results_pkts = {results_pkts!r}")

    # for dirbin in spectrafrompkts:
    #     assert (
    #         results_pkts[dirbin] == expected_results[dirbin]
    #     ), f"dirbin={dirbin} expected: {expected_results[dirbin]} actual: {results_pkts[dirbin]}"


def test_spectra_get_flux_contributions() -> None:
    timestepmin = 40
    timestepmax = 80
    dfspectrum = at.spectra.get_spectrum(
        modelpath=modelpath, timestepmin=timestepmin, timestepmax=timestepmax, fnufilterfunc=None
    )[-1]

    integrated_flux_specout = np.trapz(dfspectrum["f_lambda"], x=dfspectrum["lambda_angstroms"])

    specdata = pd.read_csv(modelpath / "spec.out", delim_whitespace=True)
    arraynu = specdata.loc[:, "0"].to_numpy()
    c_ang_per_s = 2.99792458e18
    arraylambda_angstroms = c_ang_per_s / arraynu

    contribution_list, array_flambda_emission_total = at.spectra.get_flux_contributions(
        modelpath,
        timestepmin=timestepmin,
        timestepmax=timestepmax,
        use_lastemissiontype=False,
    )

    integrated_flux_emission = -np.trapz(array_flambda_emission_total, x=arraylambda_angstroms)

    # total spectrum should be equal to the sum of all emission processes
    print(f"Integrated flux from spec.out:     {integrated_flux_specout}")
    print(f"Integrated flux from emission sum: {integrated_flux_emission}")
    assert math.isclose(integrated_flux_specout, integrated_flux_emission, rel_tol=4e-3)

    # check each bin is not out by a large fraction
    diff = [abs(x - y) for x, y in zip(array_flambda_emission_total, dfspectrum["f_lambda"].to_numpy())]
    print(f"Max f_lambda difference {max(diff) / integrated_flux_specout}")
    assert max(diff) / integrated_flux_specout < 2e-3


def test_spectra_timeseries_subplots() -> None:
    timedayslist = [295, 300]
    at.spectra.plot(
        argsraw=[], specpath=modelpath, outputfile=outputpath, timedayslist=timedayslist, multispecplot=True
    )


def test_writespectra() -> None:
    at.spectra.writespectra.main(argsraw=[], modelpath=modelpath)
