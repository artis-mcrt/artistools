#!/usr/bin/env python3
import argparse
import multiprocessing
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
# import numba
# from numpy import arctan as atan

import pynonthermal as pynt

import artistools as at
import artistools.estimators
import artistools.nltepops
import artistools.nonthermal


minionfraction = 0.  # minimum number fraction of the total population to include in SF solution

defaultoutputfile = 'spencerfano_cell{cell:03d}_ts{timestep:02d}_{timedays:.0f}d.pdf'


def make_ntstats_plot(ntstatfile):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True,
                           figsize=(4, 3), tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3})

    dfstats = pd.read_csv(ntstatfile, delim_whitespace=True, escapechar='#')
    dfstats.fillna(0, inplace=True)

    norm_frac_sum = False
    if norm_frac_sum:
        # scale up (or down) ionisation, excitation, and heating to force frac_sum = 1.0
        dfstats.eval("frac_sum = frac_ionization + frac_excitation + frac_heating", inplace=True)
        norm_factors = 1. / dfstats['frac_sum']
    else:
        norm_factors = 1.0
    pd.options.display.max_rows = 50
    pd.options.display.width = 250
    pd.set_option("display.width", 250)
    print(dfstats.to_string())

    xarr = np.log10(dfstats.x_e)
    ax.plot(xarr, dfstats.frac_ionization * norm_factors, label='Ionisation')
    if not max(dfstats.frac_excitation) == 0.:
        ax.plot(xarr, dfstats.frac_excitation * norm_factors, label='Excitation')
    ax.plot(xarr, dfstats.frac_heating * norm_factors, label='Heating')
    ioncols = [col for col in dfstats.columns.values if col.startswith('frac_ionization_')]
    for ioncol in ioncols:
        ion = ioncol.replace('frac_ionization_', '')
        ax.plot(xarr, dfstats[ioncol] * norm_factors, label=f'{ion} ionisation')

    ax.set_ylabel(r'Energy fraction')
    ax.set_xlabel(r'log x$_e$')
    ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1)
    ax.autoscale(enable=True, axis='both', tight=True)
    outputfilename = Path(ntstatfile).with_suffix('.pdf')
    fig.savefig(outputfilename, format='pdf')
    print(f"Saved '{outputfilename}'")
    plt.close()


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-timedays', '-time', '-t',
                        help='Time in days to plot')

    parser.add_argument('-timestep', '-ts', type=int,
                        help='Timestep number to plot')

    parser.add_argument('-modelgridindex', '-cell', type=int, default=0,
                        help='Modelgridindex to plot')

    parser.add_argument('-velocity', '-v', type=float, default=-1,
                        help='Specify cell by velocity')

    parser.add_argument('-npts', type=int, default=4096,
                        help='Number of points in the energy grid')

    parser.add_argument('-emin', type=float, default=0.1,
                        help='Minimum energy in eV of Spencer-Fano solution')

    parser.add_argument('-emax', type=float, default=16000,
                        help='Maximum energy in eV of Spencer-Fano solution (approx where energy is injected)')

    parser.add_argument('-vary', action='store', choices=['emin', 'emax', 'npts', 'emax,npts', 'x_e'],
                        help='Which parameter to vary')

    parser.add_argument('-composition', action='store', default='artis', choices=['artis', *at.get_elsymbolslist()[1:]],
                        help='Composition comes from artis or specific an element to use')

    parser.add_argument('-x_e', type=float, default=2,
                        help='If not using artis composition, specify the electron fraction = N_e / N_ions')

    parser.add_argument('--workfn', action='store_true',
                        help='Testing related to work functions and high energy limits')

    parser.add_argument('--makeplot', action='store_true',
                        help='Save a plot of the non-thermal spectrum')

    parser.add_argument('--differentialform', action='store_true',
                        help=('Solve differential form (KF92 Equation 6) instead of'
                              'integral form (KF92 Equation 7)'))

    parser.add_argument('--noexcitation', action='store_true',
                        help='Do not include collisional excitation transitions')

    parser.add_argument('--atomlossrate', action='store_true',
                        help=('Use Axelrod/Bethe atomic loss rate instead of assuming included cross sections '
                              'are exhaustive'))

    parser.add_argument('--ar1985', action='store_true',
                        help='Use Arnaud & Rothenflug (1985, A&AS, 60, 425) for Fe ionization cross sections')

    parser.add_argument('-o', action='store', dest='outputfile',
                        default=defaultoutputfile,
                        help='Path/filename for PDF file if --makeplot is enabled')

    parser.add_argument('-ostat', action='store',
                        help='Path/filename for stats output')

    parser.add_argument('-plotstats', action='store', default=None,
                        help='Path/filename for NT stats input (no solution, only plotting stat file)')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Plot estimated spectra from bound-bound transitions.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if args.plotstats:
        make_ntstats_plot(args.plotstats)
        return

    # global at.nonthermal.experiment_use_Latom_in_spencerfano
    at.nonthermal.experiment_use_Latom_in_spencerfano = args.atomlossrate
    modelpath = Path(args.modelpath)

    if args.workfn:
        return at.nonthermal.workfunction_tests(modelpath, args)

    if Path(args.outputfile).is_dir():
        args.outputfile = Path(args.outputfile, defaultoutputfile)

    if args.composition == 'artis':
        if args.timedays:
            args.timestep = at.get_timestep_of_timedays(modelpath, args.timedays)
        elif args.timestep is None:
            print("A time or timestep must be specified.")
            sys.exit()

        modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
        if args.velocity >= 0.:
            args.modelgridindex = at.inputmodel.get_mgi_of_velocity_kms(modelpath, args.velocity)
        else:
            args.modelgridindex = args.modelgridindex
        estimators = at.estimators.read_estimators(
            modelpath, timestep=args.timestep, modelgridindex=args.modelgridindex)
        estim = estimators[(args.timestep, args.modelgridindex)]

        dfpops = at.nltepops.read_files(modelpath, modelgridindex=args.modelgridindex, timestep=args.timestep)

        if dfpops is None or dfpops.empty:
            print(f'ERROR: no NLTE populations for cell {args.modelgridindex} at timestep {args.timestep}')
            return -1

        nntot = estim['populations']['total']
        nne = estim['nne']
        T_e = estim['Te']
        print('WARNING: Use LTE pops at Te for now')
        deposition_density_ev = estim['heating_dep'] / 1.6021772e-12  # convert erg to eV
        ionpopdict = estim['populations']

        velocity = modeldata['velocity_outer'][args.modelgridindex]
        args.timedays = float(at.get_timestep_time(modelpath, args.timestep))
        print(f'timestep {args.timestep} cell {args.modelgridindex} (v={velocity} km/s at {args.timedays:.1f}d)')

    # ionpopdict = {}
    # deposition_density_ev = 327
    # nne = 6.7e5
    #
    # ionpopdict[(26, 1)] = ionpopdict[26] * 1e-4
    # ionpopdict[(26, 2)] = ionpopdict[26] * 0.20
    # ionpopdict[(26, 3)] = ionpopdict[26] * 0.80
    # ionpopdict[(26, 4)] = ionpopdict[26] * 0.
    # ionpopdict[(26, 5)] = ionpopdict[26] * 0.
    # ionpopdict[(27, 2)] = ionpopdict[27] * 0.20
    # ionpopdict[(27, 3)] = ionpopdict[27] * 0.80
    # ionpopdict[(27, 4)] = 0.
    # # ionpopdict[(28, 1)] = ionpopdict[28] * 6e-3
    # ionpopdict[(28, 2)] = ionpopdict[28] * 0.18
    # ionpopdict[(28, 3)] = ionpopdict[28] * 0.82
    # ionpopdict[(28, 4)] = ionpopdict[28] * 0.
    # ionpopdict[(28, 5)] = ionpopdict[28] * 0.

    # x_e = 1.e-2
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # # nne = nntot * x_e
    # # nne = .1
    # dfpops = {}

    # ionpopdict[(at.get_atomic_number('Fe'), 2)] = nntot * 1.
    # ionpopdict[(at.get_atomic_number('Fe'), 3)] = nntot * 0.5
    # ionpopdict[(at.get_atomic_number('Fe'), 4)] = nntot * 0.3

    # KF1992 Figure 2. Pure-Oxygen Plasma
    # x_e = 1.e-2
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # dfpops = {}
    # ionpopdict[(at.get_atomic_number('O'), 1)] = nntot * (1. - x_e)
    # ionpopdict[(at.get_atomic_number('O'), 2)] = nntot * x_e

    # KF1992 Figure 3. Pure-Helium Plasma
    # compelement = args.composition
    # compelement_atomicnumber = at.get_atomic_number(compelement)
    # x_e = args.x_e
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # dfpops = {}
    # ionpopdict[(at.get_atomic_number('He'), 1)] = nntot * (1. - x_e)
    # ionpopdict[(at.get_atomic_number('He'), 2)] = nntot * x_e

    # KF1992 Figure 5. Pure-Iron Plasma
    # x_e = 1.e-2
    # deposition_density_ev = 5.e3
    # nntot = 1.
    # ionpopdict = {}
    # dfpops = {}
    # ionpopdict[(at.get_atomic_number('Fe'), 1)] = nntot * (1. - x_e)
    # ionpopdict[(at.get_atomic_number('Fe'), 2)] = nntot * x_e

    # KF1992 D. The Oxygen-Carbon Zone
    # ionpopdict[(at.get_atomic_number('C'), 1)] = 0.16 * nntot
    # ionpopdict[(at.get_atomic_number('C'), 2)] = 0.16 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('O'), 1)] = 0.82 * nntot
    # ionpopdict[(at.get_atomic_number('O'), 2)] = 0.82 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('Ne'), 1)] = 0.016 * nntot

    # # KF1992 G. The Silicon-Calcium Zone
    # ionpopdict[(at.get_atomic_number('C'), 1)] = 0.38e-5 * nntot
    # ionpopdict[(at.get_atomic_number('O'), 1)] = 0.94e-4 * nntot
    # ionpopdict[(at.get_atomic_number('Si'), 1)] = 0.63 * nntot
    # ionpopdict[(at.get_atomic_number('Si'), 2)] = 0.63 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('S'), 1)] = 0.29 * nntot
    # ionpopdict[(at.get_atomic_number('S'), 2)] = 0.29 * nntot * x_e
    # ionpopdict[(at.get_atomic_number('Ar'), 1)] = 0.041 * nntot
    # ionpopdict[(at.get_atomic_number('Ca'), 1)] = 0.026 * nntot
    # ionpopdict[(at.get_atomic_number('Fe'), 1)] = 0.012 * nntot

    stepcount = 9 if args.vary else 1
    for step in range(stepcount):

        emin = args.emin
        emax = args.emax
        npts = args.npts
        if args.vary == 'emin':
            emin *= 2 ** step
        elif args.vary == 'emax':
            emax *= 2 ** step
        elif args.vary == 'npts':
            npts *= 2 ** step
        elif args.vary == 'x_e':
            assert args.composition != 'artis'
        if args.vary == 'emax,npts':
            npts *= 2 ** step
            emax *= 2 ** step

        if args.composition != 'artis':
            compelement = args.composition
            compelement_atomicnumber = at.get_atomic_number(compelement)
            deposition_density_ev = 5.e3
            nntot = 1.
            x_e = (args.x_e * 10 ** (0.5 * step)) if args.vary == 'x_e' else args.x_e
            ionpopdict = {}
            dfpops = {}
            T_e = 3000
            assert x_e <= 1.
            ionpopdict[(compelement_atomicnumber, 1)] = nntot * (1. - x_e)
            ionpopdict[(compelement_atomicnumber, 2)] = nntot * x_e

        ions = []
        for key in ionpopdict.keys():
            # keep only the ion populations, not element or total populations
            if isinstance(key, tuple) and len(key) == 2 and ionpopdict[key] / nntot >= minionfraction:
                ions.append(key)

        ions.sort()

        if args.noexcitation:
            adata = None
            dfpops = None
        else:
            adata = at.atomic.get_levels(modelpath, get_transitions=True, ionlist=tuple(ions))

        if step == 0 and args.ostat:
            with open(args.ostat, 'w') as fstat:
                strheader = '#emin emax npts x_e frac_sum frac_excitation frac_ionization frac_heating'
                for atomic_number, ion_stage in ions:
                    strheader += ' frac_ionization_' + at.get_ionstring(atomic_number, ion_stage, nospace=True)
                fstat.write(strheader + '\n')

        with pynt.SpencerFanoSolver(
                emin_ev=emin, emax_ev=emax, npts=npts, verbose=True) as sf:

            for Z, ionstage in ions:
                nnion = ionpopdict[(Z, ionstage)]
                if nnion == 0.:
                    print(f'   skipping Z={Z} ion_stage {ionstage} due to nnion={nnion:.1e}')
                    continue

                sf.add_ionisation(Z, ionstage, nnion)
                if not args.noexcitation:
                    sf.add_ion_ltepopexcitation(Z, ionstage, nnion, adata=adata, temperature=T_e)

            sf.solve(depositionratedensity_ev=deposition_density_ev)

            sf.analyse_ntspectrum()

            if args.makeplot:
                # outputfilename = str(args.outputfile).format(
                #     cell=args.modelgridindex, timestep=args.timestep, timedays=args.timedays)
                outputfilename = 'spencerfano.pdf'
                sf.plot_spec_channels(outputfilename=outputfilename)

            if args.ostat:
                with open(args.ostat, 'a') as fstat:
                    strlineout = (f'{emin} {emax} {npts} {x_e:7.2e} {sf.get_frac_sum():6.3f} '
                                  f'{sf.get_frac_excitation_tot():6.3f} {sf.get_frac_ionisation_tot():6.3f} '
                                  f' {sf.get_frac_heating():6.3f}')
                    for atomic_number, ionstage in ions:
                        nnion = ionpopdict[(atomic_number, ionstage)]
                        if nnion > 0.:
                            frac_ionis_ion = sf.get_frac_ionisation_ion(atomic_number, ionstage)
                        else:
                            frac_ionis_ion = 0.
                        strlineout += f' {frac_ionis_ion:.4f}'
                    fstat.write(strlineout + '\n')

    if args.ostat:
        make_ntstats_plot(args.ostat)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
