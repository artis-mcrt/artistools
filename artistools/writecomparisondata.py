#!/usr/bin/env python3

import argparse
import numpy as np
from pathlib import Path

import pandas as pd

import artistools as at
import artistools.estimators
import artistools.lightcurve


def write_spectra(modelpath, model_id, selected_timesteps, outfile):
    spec_data = np.loadtxt(Path(modelpath, "spec.out"))

    times = spec_data[0, 1:]
    freqs = spec_data[1:, 0]
    lambdas = 2.99792458e18 / freqs

    # print("\n".join(["{0}, {1}".format(*x) for x in enumerate(times)]))

    fluxes_nu = spec_data[1:, 1:]

    # 1 parsec in cm is 3.086e18
    # area in cm^2 of a spherical of radius 1 Mpc is:
    area = 3.086e18 * 3.086e18 * 1e12 * 4. * 3.141592654

    lum_lambda = np.zeros((len(lambdas), len(times)))

    # convert flux to power by multiplying by area
    for n in range(1000):
        # 2.99792458e18 is c in Angstrom / second
        lum_lambda[n, :] = fluxes_nu[n, :] * 2.99792458e18 / lambdas[n] / lambdas[n] * area

    with open(outfile, "w") as f:
        f.write("#NTIMES: {0}\n".format(len(selected_timesteps)))
        f.write("#NWAVE: {0}\n".format(len(lambdas)))
        f.write("#TIMES[d]: {0}\n".format(" ".join(["{0:.2f}".format(times[ts]) for ts in selected_timesteps])))
        f.write("#wavelength[Ang] flux_t0[erg/s/Ang] flux_t1[erg/s/Ang] ... flux_tn[erg/s/Ang]\n")

        for n in reversed(range(len(lambdas))):
            f.write("{0:.2f} ".format(lambdas[n]) + " ".join(
                ["{0:.2e}".format(lum_lambda[n, ts]) for ts in selected_timesteps]) + "\n")

        f.close()


def write_ntimes_nvel(f, selected_timesteps, modelpath):
    times = at.get_timestep_times_float(modelpath)
    modeldata, t_model_init_days, _ = at.inputmodel.get_modeldata(modelpath)
    f.write(f'#NTIMES: {len(selected_timesteps)}\n')
    f.write(f'#NVEL: {len(modeldata)}\n')
    f.write(f'#TIMES[d]: {" ".join([f"{times[ts]:.2f}" for ts in selected_timesteps])}\n')


def write_single_estimator(modelpath, selected_timesteps, estimators, allnonemptymgilist, outfile, keyname):
    modeldata, t_model_init_days, _ = at.inputmodel.get_modeldata(modelpath)
    with open(outfile, "w") as f:
        write_ntimes_nvel(f, selected_timesteps, modelpath)
        if keyname == 'total_dep':
            f.write('#vel_mid[km/s] Edep_t0[erg/s/cm^3] Edep_t1[erg/s/cm^3] ... Edep_tn[erg/s/cm^3]\n')
        elif keyname == 'nne':
            f.write('#vel_mid[km/s] ne_t0[/cm^3] ne_t1[/cm^3] … ne_tn[/cm^3]\n')
        elif keyname == 'Te':
            f.write('#vel_mid[km/s] Tgas_t0[K] Tgas_t1[K] ... Tgas_tn[K]\n')
        for modelgridindex, cell in modeldata.iterrows():
            if modelgridindex not in allnonemptymgilist:
                continue
            v_mid = (cell.velocity_inner + cell.velocity_outer) / 2.
            f.write(f'{v_mid:.2f}')
            for timestep in selected_timesteps:
                cellvalue = estimators[(timestep, modelgridindex)][keyname]
                # try:
                #     cellvalue = estimators[(timestep, modelgridindex)][keyname]
                # except KeyError:
                #     cellvalue = (estimators[(timestep - 1, modelgridindex)][keyname]
                #                  + estimators[(timestep + 1, modelgridindex)][keyname]) / 2.
                f.write(f' {cellvalue:.3e}')
            f.write('\n')


def write_ionfracts(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, outputpath):
    times = at.get_timestep_times_float(modelpath)
    modeldata, t_model_init_days, _ = at.inputmodel.get_modeldata(modelpath)
    elementlist = at.get_composition_data(modelpath)
    nelements = len(elementlist)
    for element in range(nelements):
        atomic_number = elementlist.Z[element]
        elsymb = at.get_elsymbol(atomic_number).lower()
        nions = elementlist.nions[element]
        pathfileout = Path(outputpath, f'ionfrac_{elsymb}_{model_id}_artisnebular.txt')
        fileisallzeros = True  # will be changed when a non-zero is encountered
        with open(pathfileout, 'w') as f:
            f.write(f'#NTIMES: {len(selected_timesteps)}\n')
            f.write(f'#NSTAGES: {nions}\n')
            f.write(f'#TIMES[d]: {" ".join([f"{times[ts]:.2f}" for ts in selected_timesteps])}\n')
            f.write('#\n')
            for timestep in selected_timesteps:
                f.write(f'#TIME: {times[timestep]:.2f}\n')
                f.write(f'#NVEL: {len(allnonemptymgilist)}\n')
                f.write(f'#vel_mid[km/s] {" ".join([f"{elsymb}{ion}" for ion in range(nions)])}\n')
                for modelgridindex, cell in modeldata.iterrows():
                    if modelgridindex not in allnonemptymgilist:
                        continue
                    v_mid = (cell.velocity_inner + cell.velocity_outer) / 2.
                    f.write(f'{v_mid:.2f}')
                    elabund = estimators[(timestep, modelgridindex)]['populations'].get(atomic_number, 0)
                    for ion in range(nions):
                        ion_stage = ion + elementlist.lowermost_ionstage[element]
                        ionabund = estimators[(timestep, modelgridindex)]['populations'].get(
                            (atomic_number, ion_stage), 0)
                        ionfrac = ionabund / elabund if elabund > 0 else 0
                        if ionfrac > 0.:
                            fileisallzeros = False
                        f.write(' {:.4e}'.format(ionfrac))
                    f.write('\n')
        if fileisallzeros:
            print(f'Deleting {pathfileout} because it is all zeros')
            pathfileout.unlink()


def write_phys(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, outputpath):
    times = at.get_timestep_times_float(modelpath)
    modeldata, t_model_init_days, _ = at.inputmodel.get_modeldata(modelpath)
    with open(Path(outputpath, f'phys_{model_id}_artisnebular.txt'), 'w') as f:
        f.write(f'#NTIMES: {len(selected_timesteps)}\n')
        f.write(f'#TIMES[d]: {" ".join([f"{times[ts]:.2f}" for ts in selected_timesteps])}\n')
        f.write('#\n')
        for timestep in selected_timesteps:
            f.write(f'#TIME: {times[timestep]:.2f}\n')
            f.write(f'#NVEL: {len(modeldata)}\n')
            f.write('#vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]\n')
            for modelgridindex, cell in modeldata.iterrows():
                if modelgridindex not in allnonemptymgilist:
                    continue

                estimators[(timestep, modelgridindex)]['rho'] = (
                    10 ** cell.logrho * (t_model_init_days / times[timestep]) ** 3)

                estimators[(timestep, modelgridindex)]['nntot'] = (
                    estimators[(timestep, modelgridindex)]['populations']['total'])

                v_mid = (cell.velocity_inner + cell.velocity_outer) / 2.
                f.write(f'{v_mid:.2f}')
                for keyname in ('Te', 'rho', 'nne', 'nntot'):
                    estvalue = estimators[(timestep, modelgridindex)][keyname]
                    f.write(' {:.4e}'.format(estvalue))
                f.write('\n')


def write_lbol_edep(modelpath, model_id, selected_timesteps, estimators, outputpath):
    # times = at.get_timestep_times_float(modelpath)
    dflightcurve = at.lightcurve.readfile(Path(modelpath, "light_curve.out"))
    dfdep = at.get_deposition(modelpath)

    df = pd.merge(dflightcurve, dfdep, left_index=True, right_index=True, suffixes=('', '_dep'))

    with open(outputpath, "w") as f:
        f.write(f'#NTIMES: {len(selected_timesteps)}\n')
        f.write("#time[d] Lbol[erg/s] Edep[erg/s] \n")

        for timestep, row in df.iterrows():
            if timestep not in selected_timesteps:
                continue
            f.write(f"{row.time:.2f} {row.lum * 3.826e33:.2e} {row.total_dep_Lsun * 3.826e33:.2e}\n")

    f.close()


def addargs(parser):
    parser.add_argument('-modelpath', default=[], nargs='*', action=at.AppendPath,
                        help='Paths to ARTIS folders')

    parser.add_argument('-selected_timesteps', default=[], nargs='*', action=at.AppendPath,
                        help='Selected ARTIS timesteps')

    parser.add_argument('-outputpath', '-o', action='store', type=Path, default=Path(),
                        help='path for output files')


def main(args=None, argsraw=None, **kwargs):
    """Plot spectra from ARTIS and reference data."""
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Write ARTIS model data out in code comparison workshop format.')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    if not args.modelpath and not args.specpath:
        args.modelpath = [Path('.')]
    elif isinstance(args.modelpath, (str, Path)):
        args.modelpath = [args.modelpath]

    modelpathlist = args.modelpath
    selected_timesteps = args.selected_timesteps

    args.outputpath.mkdir(parents=True, exist_ok=True)

    for modelpath in modelpathlist:
        model_id = str(modelpath.name).split('_')[0]

        modeldata, t_model_init_days, _ = at.inputmodel.get_modeldata(modelpath)
        estimators = at.estimators.read_estimators(modelpath=modelpath)
        allnonemptymgilist = [modelgridindex for modelgridindex in modeldata.index
                              if not estimators[(selected_timesteps[0], modelgridindex)]['emptycell']]

        write_lbol_edep(modelpath, model_id, selected_timesteps, estimators,
                        Path(args.outputpath, "lbol_edep_" + model_id + "_artisnebular.txt"))

        write_spectra(modelpath, model_id, selected_timesteps,
                      Path(args.outputpath, "spectra_" + model_id + "_artisnebular.txt"))

        # write_single_estimator(modelpath, selected_timesteps, estimators, allnonemptymgilist,
        #                        Path(args.outputpath, "eden_" + model_id + "_artisnebular.txt"), keyname='nne')

        write_single_estimator(modelpath, selected_timesteps, estimators, allnonemptymgilist,
                               Path(args.outputpath, "edep_" + model_id + "_artisnebular.txt"), keyname='total_dep')

        # write_single_estimator(modelpath, selected_timesteps, estimators, allnonemptymgilist,
        #                        Path(args.outputpath, "tgas_" + model_id + "_artisnebular.txt"), keyname='Te')

        write_phys(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, args.outputpath)
        write_ionfracts(modelpath, model_id, selected_timesteps, estimators, allnonemptymgilist, args.outputpath)


if __name__ == "__main__":
    main()
