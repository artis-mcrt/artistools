#!/usr/bin/env python3

import math
import numpy as np
import os.path
import pandas as pd
from astropy import constants as const
from pathlib import Path

import artistools as at
import artistools.deposition
import artistools.lightcurve
import artistools.macroatom
import artistools.makemodel.botyanski2017
import artistools.nltepops
import artistools.nonthermal
import artistools.radfield
import artistools.spectra
import artistools.transitions

modelpath = Path(os.path.dirname(os.path.abspath(__file__)), 'data')
outputpath = Path(os.path.dirname(os.path.abspath(__file__)), 'output')
at.enable_diskcache = False


def test_timestep_times():
    timestartarray = at.get_timestep_times_float(modelpath, loc='start')
    timedeltarray = at.get_timestep_times_float(modelpath, loc='delta')
    timemidarray = at.get_timestep_times_float(modelpath, loc='mid')
    assert len(timestartarray) == 100
    assert math.isclose(float(timemidarray[0]), 250.421, abs_tol=1e-3)
    assert math.isclose(float(timemidarray[-1]), 349.412, abs_tol=1e-3)

    assert all([tstart < tmid < (tstart + tdelta)
                for tstart, tdelta, tmid in zip(timestartarray, timedeltarray, timemidarray)])


def test_deposition():
    at.deposition.main(modelpath=modelpath)


def test_estimator_snapshot():
    at.estimators.main(modelpath=modelpath, outputfile=outputpath, timedays=300)


def test_estimator_timeevolution():
    at.estimators.main(modelpath=modelpath, outputfile=outputpath, modelgridindex=0, x='time')


def test_lightcurve():
    at.lightcurve.main(modelpath=modelpath, outputfile=outputpath)


def test_lightcurve_frompackets():
    at.lightcurve.main(modelpath=modelpath, frompackets=True,
                       outputfile=os.path.join(outputpath, 'lightcurve_from_packets.pdf'))


def test_lightcurve_magnitudes_plot():
    at.lightcurve.main(modelpath=modelpath, magnitude=True, outputfile=outputpath)


def test_macroatom():
    at.macroatom.main(modelpath=modelpath, outputfile=outputpath, timestep=10)


def test_makemodel():
    at.makemodel.botyanski2017.main(outputpath=outputpath)


def test_menu():
    at.main()
    at.showtimesteptimes(modelpath=modelpath)


def test_nltepops():
    # at.nltepops.main(modelpath=modelpath, outputfile=outputpath, timedays=300),
    #                    **benchargs)
    at.nltepops.main(modelpath=modelpath, outputfile=outputpath, timestep=40)


def test_nonthermal():
    at.nonthermal.main(modelpath=modelpath, outputfile=outputpath, timestep=70)


def test_radfield():
    at.radfield.main(modelpath=modelpath, modelgridindex=0, outputfile=outputpath)


def test_get_ionrecombratecalibration():
    at.get_ionrecombratecalibration(modelpath=modelpath)


def test_spectraplot():
    at.spectra.main(modelpath=modelpath, outputfile=outputpath, timemin=290, timemax=320)


def test_spectra_frompackets():
    at.spectra.main(modelpath=modelpath, outputfile=os.path.join(outputpath, 'spectrum_from_packets.pdf'),
                    timemin=290, timemax=320, frompackets=True)


def test_spectra_outputtext():
    at.spectra.main(modelpath=modelpath, output_spectra=True)


def test_spectraemissionplot():
    at.spectra.main(modelpath=modelpath, outputfile=outputpath, timemin=290, timemax=320,
                    emissionabsorption=True)


def test_spectraemissionplot_nostack():
    at.spectra.main(modelpath=modelpath, outputfile=outputpath, timemin=290, timemax=320,
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


def test_spencerfano():
    at.spencerfano.main(modelpath=modelpath, timedays=300, makeplot=True, npts=200, outputfile=outputpath)


def test_transitions():
    at.transitions.main(modelpath=modelpath, outputfile=outputpath, timedays=300)
