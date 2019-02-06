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

modelpath = Path('tests', 'data')
outputpath = Path('tests', 'output')
specfilename = modelpath / 'spec.out'

benchargs = dict(iterations=1, rounds=1)


def test_timestep_times():
    timestartarray = at.get_timestep_times_float(modelpath, loc='start')
    timedeltarray = at.get_timestep_times_float(modelpath, loc='delta')
    timearray = at.get_timestep_times_float(modelpath)
    strtimearray = at.get_timestep_times(modelpath)
    assert len(strtimearray) == 100
    assert math.isclose(float(strtimearray[0]), 250.421, abs_tol=1e-3)
    assert math.isclose(float(strtimearray[-1]), 349.412, abs_tol=1e-3)

    assert all([math.isclose(float(strtimearray[ts]), timearray[ts], abs_tol=1e-3)
                for ts in range(len(strtimearray))])

    assert all([math.isclose(tstart + (tdelta / 2.), tmid, abs_tol=1e-3)
                for tstart, tdelta, tmid in zip(timestartarray, timedeltarray, timearray)])


def test_deposition():
    at.deposition.main(modelpath=modelpath)


def test_estimator_snapshot(benchmark):
    benchmark.pedantic(at.estimators.main, kwargs=dict(modelpath=modelpath, outputfile=outputpath, timedays=300),
                       **benchargs)


def test_estimator_timeevolution(benchmark):
    benchmark.pedantic(at.estimators.main, kwargs=dict(modelpath=modelpath, outputfile=outputpath, modelgridindex=0,
                                                       x='time'),
                       **benchargs)


def test_lightcurve(benchmark):
    benchmark.pedantic(at.lightcurve.main, kwargs=dict(modelpath=modelpath, outputfile=outputpath), **benchargs)


def test_lightcurve_frompackets(benchmark):
    benchmark.pedantic(at.lightcurve.main, kwargs=dict(modelpath=modelpath, frompackets=True,
                       outputfile=os.path.join(outputpath, 'lightcurve_from_packets.pdf')), **benchargs)


def test_lightcurve_magnitudes_plot():
    at.lightcurve.main(modelpath=modelpath, magnitude=True, outputfile=outputpath)


def test_macroatom():
    at.macroatom.main(modelpath=modelpath, outputfile=outputpath, timestep=10)


def test_makemodel():
    at.makemodel.botyanski2017.main(outputpath=outputpath)


def test_menu():
    at.main()
    at.showtimesteptimes(modelpath=modelpath)


def test_nltepops(benchmark):
    # mybench(benchmark, at.nltepops.main, modelpath=modelpath, outputfile=outputpath, timedays=300)
    benchmark.pedantic(at.nltepops.main, kwargs=dict(modelpath=modelpath, outputfile=outputpath, timedays=300),
                       **benchargs)


def test_nltepops_departuremode(benchmark):
    benchmark.pedantic(at.nltepops.main, kwargs=dict(modelpath=modelpath, outputfile=outputpath, timedays=300,
                                                     departuremode=True), **benchargs)


def test_nonthermal(benchmark):
    benchmark.pedantic(at.nonthermal.main, kwargs=dict(modelpath=modelpath, outputfile=outputpath, timedays=300),
                       **benchargs)


def test_radfield(benchmark):
    benchmark.pedantic(at.radfield.main, kwargs=dict(modelpath=modelpath, modelgridindex=0, outputfile=outputpath), **benchargs)


def test_get_ionrecombratecalibration():
    at.get_ionrecombratecalibration(modelpath=modelpath)


def test_spectraplot(benchmark):
    benchmark.pedantic(at.spectra.main,
                       kwargs=dict(modelpath=modelpath, outputfile=outputpath, timemin=290, timemax=320),
                       **benchargs)


def test_spectra_frompackets(benchmark):
    benchmark.pedantic(at.spectra.main,
                       kwargs=dict(modelpath=modelpath,
                                   outputfile=os.path.join(outputpath, 'spectrum_from_packets.pdf'),
                                   timemin=290, timemax=320, frompackets=True), **benchargs)


def test_spectra_outputtext():
    at.spectra.main(modelpath=modelpath, output_spectra=True)


def test_spectraemissionplot(benchmark):
    benchmark.pedantic(at.spectra.main,
                       kwargs=dict(modelpath=modelpath, outputfile=outputpath, timemin=290, timemax=320,
                                   emissionabsorption=True), **benchargs)


def test_spectraemissionplot_nostack(benchmark):
    benchmark.pedantic(at.spectra.main,
                       kwargs=dict(modelpath=modelpath, outputfile=outputpath, timemin=290, timemax=320,
                                   emissionabsorption=True, nostack=True), **benchargs)


def test_spectra_get_spectrum():
    def check_spectrum(dfspectrumpkts):
        assert math.isclose(max(dfspectrumpkts['f_lambda']), 2.548532804918824e-13, abs_tol=1e-5)
        assert min(dfspectrumpkts['f_lambda']) < 1e-9
        assert math.isclose(np.mean(dfspectrumpkts['f_lambda']), 1.0314682640070206e-14, abs_tol=1e-5)

    dfspectrum = at.spectra.get_spectrum(specfilename, 55, 65, fnufilterfunc=None)
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
        specfilename, timestepmin=timestepmin, timestepmax=timestepmax, fnufilterfunc=None)

    integrated_flux_specout = np.trapz(dfspectrum['f_lambda'], x=dfspectrum['lambda_angstroms'])

    specdata = pd.read_csv(specfilename, delim_whitespace=True)
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
