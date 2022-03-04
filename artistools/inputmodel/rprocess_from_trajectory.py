#!/usr/bin/env python3

import argparse
import math
import tarfile
# import os.path

import numpy as np
import pandas as pd
from pathlib import Path

import artistools as at


def get_elemabund_from_nucabund(dfnucabund):
    dictelemabund = {}
    for atomic_number in range(1, dfnucabund.Z.max() + 1):
        dictelemabund[f'X_{at.elsymbols[atomic_number]}'] = (
            dfnucabund.query('Z == @atomic_number', inplace=False).massfrac.sum())
    return dictelemabund


def get_traj_tarpath(particleid):
    return Path(f'/Users/luke/Dropbox/Archive/Mergers/SFHo/{particleid}.tar.xz')


def get_trajectory_nuc_abund(particleid, memberfilename):
    trajpath = get_traj_tarpath(particleid)
    with tarfile.open(trajpath, mode='r:*') as tar:
        trajfile = tar.extractfile(member=memberfilename)

        # with open(trajfile) as ftraj:
        _, str_t_model_init_seconds, _, rho, _, _ = trajfile.readline().decode().split()
        t_model_init_seconds = float(str_t_model_init_seconds)

        dfnucabund = pd.read_csv(trajfile, delim_whitespace=True, comment='#',
                                 names=["N", "Z", "log10abund", "S1n", "S2n"], usecols=["N", "Z", "log10abund"],
                                 dtype={0: int, 1: int, 2: float})

    # dfnucabund.eval('abund = 10 ** log10abund', inplace=True)
    dfnucabund.eval('massfrac = (N + Z) * 10 ** log10abund', inplace=True)
    # dfnucabund.eval('A = N + Z', inplace=True)
    # dfnucabund.query('abund > 0.', inplace=True)

    # abund is proportional to number abundance, but needs normalisation
    # normfactor = dfnucabund.abund.sum()
    # print(f'abund sum: {normfactor}')
    # dfnucabund.eval('numberfrac = abund / @normfactor', inplace=True)

    return dfnucabund, t_model_init_seconds


def add_abundancecontributions(gridcontribpath, dfmodeldata, t_model_days):
    """ contribute trajectory network calculation abundances to model cell abundances """

    cellcount = len(dfmodeldata)

    t_model_s = t_model_days * 86400
    if 'X_Fegroup' not in dfmodeldata.columns:
        dfmodeldata = pd.concat([dfmodeldata, pd.DataFrame({'X_Fegroup': np.ones(cellcount)})], axis=1)

    dfcontribs = pd.read_csv(Path(gridcontribpath, 'gridcontributions.txt'), delim_whitespace=True,
                             dtype={0: int, 1: int, 2: float})

    dfelabundances = pd.DataFrame({'inputcellid': dfmodeldata.gridindex})
    # dfcontribs = dfcontribs[:10]  # TODO: REMOVE!
    particleids = dfcontribs.particleid.unique()
    particle_count = len(particleids)

    for n, particleid in enumerate(particleids):
        print(f'\ntrajectory particle id {particleid} ('
              f'{n + 1} of {particle_count}, {(n + 1) / particle_count * 100:.1f}%)')
        dfthisparticlecontribs = dfcontribs.query('particleid == @particleid', inplace=False)

        if not get_traj_tarpath(particleid).is_file():
            # print(dfthisparticlecontribs)
            print(f' WARNING {get_traj_tarpath(particleid)} not found! '
                  f'Contributes up to {dfthisparticlecontribs.frac_of_cellmass.max() * 100:.1f}% mass of some cells')
            continue

        # find the closest timestep to the require time
        with tarfile.open(get_traj_tarpath(particleid), mode='r:*') as tar:
            evolfile = tar.extractfile(member='./Run_rprocess/evol.dat')
            dfevol = pd.read_csv(
                evolfile, delim_whitespace=True, comment='#', usecols=[0, 1], names=['nstep', 'timesec'])
            idx = np.abs(dfevol.timesec.values - t_model_s).argmin()
            nts = dfevol.nstep.values[idx]

        dfnucabund, traj_time_s = get_trajectory_nuc_abund(particleid, f'./Run_rprocess/nz-plane{nts:05d}')
        dfnucabund.query('Z >= 1', inplace=True)
        dfnucabund['abundcolname'] = [f'X_{at.get_elsymbol(row.Z)}{row.N + row.Z}' for row in dfnucabund.itertuples()]

        massfracnormfactor = dfnucabund.massfrac.sum()
        print(f' massfrac sum: {massfracnormfactor}')

        print(f'    grid snapshot: {t_model_s:.2e} seconds')
        print(f' network timestep: {traj_time_s:.2e} seconds (timestep {nts})')
        assert np.isclose(t_model_s, traj_time_s, rtol=0.7, atol=1)
        print(f' contributing {len(dfnucabund)} abundances to {len(dfthisparticlecontribs)} cells')

        for contrib in dfthisparticlecontribs.itertuples():
            newmodelcols = [colname for colname in dfnucabund.abundcolname.values if colname not in dfmodeldata.columns]
            if newmodelcols:
                dfabundcols = pd.DataFrame({colname: np.zeros(cellcount) for colname in newmodelcols})
                dfmodeldata = pd.concat([dfmodeldata, dfabundcols], axis=1)

            elemabundcols = [f'X_{at.get_elsymbol(Z)}' for Z in dfnucabund.Z]
            newelemabundcols = [colname for colname in elemabundcols if colname not in dfelabundances.columns]
            if newelemabundcols:
                dfabundcols = pd.DataFrame({colname: np.zeros(cellcount) for colname in newelemabundcols})
                dfelabundances = pd.concat([dfelabundances, dfabundcols], axis=1)

            for _, Z, abundcolname, massfrac in dfnucabund[['Z', 'abundcolname', 'massfrac']].itertuples():
                dfmodeldata.at[contrib.cellindex - 1, abundcolname] += massfrac * contrib.frac_of_cellmass
                dfelabundances.at[contrib.cellindex - 1, f'X_{at.get_elsymbol(Z)}'] += massfrac * contrib.frac_of_cellmass

            # assert contrib.cellindex == dfmodeldata.at[contrib.cellindex - 1, 'gridindex']

    return dfmodeldata, dfelabundances


def addargs(parser):
    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Create solar r-process pattern in ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    particleid = 88969
    print(f'trajectory particle id {particleid}')
    dfnucabund, t_model_init_seconds = get_trajectory_nuc_abund(particleid, './Run_rprocess/tday_nz-plane')
    dfnucabund.query('Z >= 1', inplace=True)
    dfnucabund['radioactive'] = True

    t_model_init_days = t_model_init_seconds / (24 * 60 * 60)

    wollager_profilename = 'wollager_ejectaprofile_10bins.txt'
    if Path(wollager_profilename).exists():
        print(f'{wollager_profilename} found')
        t_model_init_days_in = float(Path(wollager_profilename).open('rt').readline().strip().removesuffix(' day'))
        dfdensities = pd.read_csv(wollager_profilename, delim_whitespace=True, skiprows=1,
                                  names=['cellid', 'velocity_outer', 'rho'])
        dfdensities['cellid'] = dfdensities['cellid'].astype(int)
        dfdensities['velocity_inner'] = np.concatenate(([0.], dfdensities['velocity_outer'].values[:-1]))

        t_model_init_seconds_in = t_model_init_days_in * 24 * 60 * 60
        dfdensities.eval('shellmass_grams = rho * 4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                         '* (1e5 * @t_model_init_seconds_in) ** 3', inplace=True)

        # now replace the density at the input time with the density at required time

        dfdensities.eval('rho = shellmass_grams / ('
                         '4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                         ' * (1e5 * @t_model_init_seconds) ** 3)', inplace=True)
    else:
        rho = 1e-11
        print(f'{wollager_profilename} not found. Using rho {rho} g/cm3')
        dfdensities = pd.DataFrame(dict(rho=rho, velocity_outer=6.e4), index=[0])

    # print(dfdensities)

    # write abundances.txt
    dictelemabund = get_elemabund_from_nucabund(dfnucabund)

    dfelabundances = pd.DataFrame([dict(inputcellid=mgi + 1, **dictelemabund) for mgi in range(len(dfdensities))])
    # print(dfelabundances)
    at.inputmodel.save_initialabundances(dfelabundances=dfelabundances, abundancefilename=args.outputpath)

    # write model.txt

    rowdict = {
        # 'inputcellid': 1,
        # 'velocity_outer': 6.e4,
        # 'logrho': -3.,
        'X_Fegroup': 1.,
        'X_Ni56': 0.,
        'X_Co56': 0.,
        'X_Fe52': 0.,
        'X_Cr48': 0.,
        'X_Ni57': 0.,
        'X_Co57': 0.,
    }

    for _, row in dfnucabund.query('radioactive == True').iterrows():
        A = row.N + row.Z
        rowdict[f'X_{at.elsymbols[row.Z]}{A}'] = row.massfrac

    modeldata = []
    for mgi, densityrow in dfdensities.iterrows():
        # print(mgi, densityrow)
        modeldata.append(dict(inputcellid=mgi + 1, velocity_outer=densityrow['velocity_outer'],
                         logrho=math.log10(densityrow['rho']), **rowdict))
    # print(modeldata)

    dfmodel = pd.DataFrame(modeldata)
    # print(dfmodel)
    at.inputmodel.save_modeldata(dfmodel, t_model_init_days, Path(args.outputpath, 'model.txt'))


if __name__ == "__main__":
    main()
