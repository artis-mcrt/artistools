#!/usr/bin/env python3

import math
import numpy as np
import os.path
import pandas as pd
from astropy import constants as const
from pathlib import Path

import artistools as at
import artistools.spectra
import artistools.transitions

modelpath = at.config['path_testartismodel']
outputpath = Path(at.PYDIR, '..', 'tests', 'output')
at.enable_diskcache = False


def test_spectraplot():
    at.spectra.main(argsraw=[], specpath=[modelpath, 'sn2011fe_PTF11kly_20120822_norm.txt'], outputfile=outputpath, timemin=290, timemax=320)


def test_spectra_frompackets():
    at.spectra.main(argsraw=[], specpath=modelpath, outputfile=os.path.join(outputpath, 'spectrum_from_packets.pdf'),
                    timemin=290, timemax=320, frompackets=True)


def test_spectra_outputtext():
    at.spectra.main(argsraw=[], specpath=modelpath, output_spectra=True)


def test_spectraemissionplot():
    at.spectra.main(argsraw=[], specpath=modelpath, outputfile=outputpath, timemin=290, timemax=320,
                    emissionabsorption=True)


def test_spectraemissionplot_nostack():
    at.spectra.main(argsraw=[], specpath=modelpath, outputfile=outputpath, timemin=290, timemax=320,
                    emissionabsorption=True, nostack=True)


def test_spectra_get_spectrum():
    def check_spectrum(dfspectrumpkts):
        assert math.isclose(max(dfspectrumpkts['f_lambda']), 2.548532804918824e-13, abs_tol=1e-5)
        assert min(dfspectrumpkts['f_lambda']) < 1e-9
        assert math.isclose(np.mean(dfspectrumpkts['f_lambda']), 1.0314682640070206e-14, abs_tol=1e-5)

    dfspectrum = at.spectra.get_spectrum(modelpath, 55, 65, fnufilterfunc=None)
    assert len(dfspectrum['lambda_angstroms']) == 1000
    assert len(dfspectrum['f_lambda']) == 1000
    assert abs(dfspectrum['lambda_angstroms'].values[-1] - 29920.601421214415) < 1e-5
    assert abs(dfspectrum['lambda_angstroms'].values[0] - 600.75759482509852) < 1e-5

    check_spectrum(dfspectrum)

    lambda_min = dfspectrum['lambda_angstroms'].values[0]
    lambda_max = dfspectrum['lambda_angstroms'].values[-1]
    timelowdays = at.get_timestep_times_float(modelpath)[55]
    timehighdays = at.get_timestep_times_float(modelpath)[65]

    dfspectrumpkts = at.spectra.get_spectrum_from_packets(
        modelpath, timelowdays=timelowdays, timehighdays=timehighdays, lambda_min=lambda_min, lambda_max=lambda_max)

    check_spectrum(dfspectrumpkts)


def test_spectra_get_flux_contributions():
    timestepmin = 40
    timestepmax = 80
    dfspectrum = at.spectra.get_spectrum(
        modelpath, timestepmin=timestepmin, timestepmax=timestepmax, fnufilterfunc=None)

    integrated_flux_specout = np.trapz(dfspectrum['f_lambda'], x=dfspectrum['lambda_angstroms'])

    specdata = pd.read_csv(modelpath / 'spec.out', delim_whitespace=True)
    arraynu = specdata.loc[:, '0'].values
    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu

    contribution_list, array_flambda_emission_total = at.spectra.get_flux_contributions(
        modelpath, timestepmin=timestepmin, timestepmax=timestepmax)

    integrated_flux_emission = -np.trapz(array_flambda_emission_total, x=arraylambda_angstroms)

    # total spectrum should be equal to the sum of all emission processes
    print(f'Integrated flux from spec.out:     {integrated_flux_specout}')
    print(f'Integrated flux from emission sum: {integrated_flux_emission}')
    assert math.isclose(integrated_flux_specout, integrated_flux_emission, rel_tol=4e-3)

    # check each bin is not out by a large fraction
    diff = [abs(x - y) for x, y in zip(array_flambda_emission_total, dfspectrum['f_lambda'].values)]
    print(f'Max f_lambda difference {max(diff) / integrated_flux_specout}')
    assert max(diff) / integrated_flux_specout < 2e-3


def test_spectra_timeseries_subplots():
    timedayslist = [295, 300]
    at.spectra.main(argsraw=[], specpath=modelpath, outputfile=outputpath,
                    timedayslist=timedayslist, multispecplot=True)
