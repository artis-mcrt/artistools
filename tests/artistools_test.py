#!/usr/bin/env python3

import numpy as np
import pandas as pd
from astropy import constants as const

import artistools as at
import artistools.lightcurve
import artistools.nltepops
import artistools.nonthermalspec
import artistools.radfield
import artistools.spectra

specfilename = 'tests/data/spec.out'
emissionfilename = 'tests/data/emissiontrue.out'
absorptionfilename = 'tests/data/absorption.out'


def test_timestep_times():
    timearray = at.get_timestep_times(specfilename)
    assert(len(timearray) == 100)
    assert(timearray[0] == '250.421')
    assert(timearray[-1] == '349.412')


def test_get_spectrum():
    dfspectrum = at.spectra.get_spectrum(specfilename, 55, 65, fnufilterfunc=None)
    assert(len(dfspectrum['lambda_angstroms']) == 1000)
    assert(abs(dfspectrum['lambda_angstroms'].values[-1] - 29920.601421214415) < 1e-5)
    assert(abs(dfspectrum['lambda_angstroms'].values[0] - 600.75759482509852) < 1e-5)

    assert(len(dfspectrum['f_lambda']) == 1000)
    assert(abs(max(dfspectrum['f_lambda']) - 2.548532804918824e-13) < 1e-5)
    assert(min(dfspectrum['f_lambda']) < 1e-9)
    assert(abs(np.mean(dfspectrum['f_lambda']) - 1.0314682640070206e-14) < 1e-5)


def test_get_flux_contributions():
    timestepmin = 40
    timestepmax = 80
    dfspectrum = at.spectra.get_spectrum(
        specfilename, timestepmin=timestepmin, timestepmax=timestepmax, fnufilterfunc=None)

    integrated_flux_specout = at.spectra.integrate_flux(dfspectrum['f_lambda'], dfspectrum['lambda_angstroms'])

    specdata = pd.read_csv(specfilename, delim_whitespace=True)
    timearray = specdata.columns.values[1:]
    arraynu = specdata.loc[:, '0'].values
    arraylambda_angstroms = const.c.to('angstrom/s').value / arraynu

    contribution_list, maxyvalueglobal, array_flambda_emission_total = at.spectra.get_flux_contributions(
        emissionfilename, absorptionfilename, 5, timearray, arraynu,
        xmin=-1, xmax=np.inf, timestepmin=timestepmin, timestepmax=timestepmax)

    integrated_flux_emission = at.spectra.integrate_flux(array_flambda_emission_total, arraylambda_angstroms)

    # total spectrum should be equal to the sum of all emission processes
    print(f'Integrated flux from spec.out:     {integrated_flux_specout}')
    print(f'Integrated flux from emission sum: {integrated_flux_emission}')
    assert(abs((integrated_flux_specout / integrated_flux_emission) - 1) < 4e-3)

    # check each bin is not out by a large fraction
    diff = [abs(x - y) for x, y in zip(array_flambda_emission_total, dfspectrum['f_lambda'].values)]
    print(f'Max f_lambda difference {max(diff) / integrated_flux_specout.value}')
    assert(max(diff) / integrated_flux_specout.value < 2e-3)


def test_plotters():
    arglist = ['tests/data/', '-o', 'tests/data/']
    at.nltepops.main(arglist)
    at.lightcurve.main(arglist)
    at.spectra.main(arglist)
    at.spectra.main(arglist + ['--emissionabsorption'])
    at.nonthermalspec.main(arglist)
    assert(at.radfield.main(arglist) == 0)
