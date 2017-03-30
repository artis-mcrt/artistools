#!/usr/bin/env python3

import artistools as at
import numpy as np

specfile = 'tests/data/spec.out'

def test_timestep_times():
    timearray = at.get_timestep_times(specfile)
    assert(len(timearray) == 100)
    assert(timearray[0] == '250.421')
    assert(timearray[-1] == '349.412')


def test_get_spectrum():
    dfspectrum = at.spectra.get_spectrum(specfile, 55, 65, fnufilterfunc=None)
    assert(len(dfspectrum['lambda_angstroms']) == 1000)
    assert(abs(dfspectrum['lambda_angstroms'].values[-1] - 29920.601421214415) < 1e-5)
    assert(abs(dfspectrum['lambda_angstroms'].values[0] - 600.75759482509852) < 1e-5)

    assert(len(dfspectrum['f_lambda']) == 1000)
    assert(abs(max(dfspectrum['f_lambda']) - 2.548532804918824e-13) < 1e-5)
    assert(min(dfspectrum['f_lambda']) < 1e-9)
    assert(abs(np.mean(dfspectrum['f_lambda']) - 1.0314682640070206e-14) < 1e-5)