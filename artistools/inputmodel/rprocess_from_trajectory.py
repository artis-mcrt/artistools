#!/usr/bin/env python3

import argparse
import math
import tarfile
import time
from pathlib import Path

import artistools as at
import numpy as np
import pandas as pd

# import modin.pandas as pd


def get_elemabund_from_nucabund(dfnucabund):
    dictelemabund = {}
    for atomic_number in range(1, dfnucabund.Z.max() + 1):
        dictelemabund[f'X_{at.get_elsymbol(atomic_number)}'] = (
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


def add_abundancecontributions(gridcontribpath, dfmodel, t_model_days):
    """ contribute trajectory network calculation abundances to model cell abundances """

    timefuncstart = time.perf_counter()
    t_model_s = t_model_days * 86400
    if 'X_Fegroup' not in dfmodel.columns:
        dfmodel = pd.concat([dfmodel, pd.DataFrame({'X_Fegroup': np.ones(len(dfmodel))})], axis=1)

    dfcontribs = pd.read_csv(Path(gridcontribpath, 'gridcontributions.txt'), delim_whitespace=True,
                             dtype={0: int, 1: int, 2: float})

    active_inputcellids = dfcontribs.cellindex.unique()
    active_inputcellcount = len(active_inputcellids)
    print(f'{active_inputcellcount} of {len(dfmodel)} model cells have particle contributions')
    dfnucabundances = pd.DataFrame({'inputcellid': active_inputcellids}, index=active_inputcellids, dtype=int)

    dfcontribs_groups = dfcontribs.groupby('particleid')
    particle_count = len(dfcontribs_groups)

    for n, (particleid, dfthisparticlecontribs) in enumerate(dfcontribs_groups, 1):
        timestart = time.perf_counter()
        print(f'\ntrajectory particle id {particleid} ('
              f'{n} of {particle_count}, {n / particle_count * 100:.1f}%)')

        if not get_traj_tarpath(particleid).is_file():
            print(f' WARNING {get_traj_tarpath(particleid)} not found! '
                  f'Contributes up to {dfthisparticlecontribs.frac_of_cellmass.max() * 100:.1f}% mass of some cells')
            continue

        # find the closest timestep to the required time
        with tarfile.open(get_traj_tarpath(particleid), mode='r:*') as tar:
            evolfile = tar.extractfile(member='./Run_rprocess/evol.dat')
            dfevol = pd.read_csv(
                evolfile, delim_whitespace=True, comment='#', usecols=[0, 1], names=['nstep', 'timesec'])
            idx = np.abs(dfevol.timesec.values - t_model_s).argmin()
            nts = dfevol.nstep.values[idx]

        dftrajnucabund, traj_time_s = get_trajectory_nuc_abund(particleid, f'./Run_rprocess/nz-plane{nts:05d}')
        dftrajnucabund.query('Z >= 1', inplace=True)
        dftrajnucabund['nucabundcolname'] = [f'X_{at.get_elsymbol(row.Z)}{row.N + row.Z}'
                                             for row in dftrajnucabund.itertuples()]

        massfracnormfactor = dftrajnucabund.massfrac.sum()
        print(f' massfrac sum: {massfracnormfactor}')

        print(f'    grid snapshot: {t_model_s:.2e} seconds')
        print(f' network timestep: {traj_time_s:.2e} seconds (timestep {nts})')
        assert np.isclose(t_model_s, traj_time_s, rtol=0.7, atol=1)
        print(f' contributing {len(dftrajnucabund)} abundances to {len(dfthisparticlecontribs)} cells')

        newnucabundcols = [colname for colname in dftrajnucabund.nucabundcolname.values
                           if colname not in dfnucabundances.columns]
        if newnucabundcols:
            dfabundcols = pd.DataFrame(
                {colname: np.zeros(len(dfnucabundances)) for colname in newnucabundcols}, index=dfnucabundances.index)
            dfnucabundances = pd.concat([dfnucabundances, dfabundcols], axis=1, join='inner')

        for contrib in dfthisparticlecontribs.itertuples():
            for _, Z, nucabundcolname, massfrac in dftrajnucabund[['Z', 'nucabundcolname', 'massfrac']].itertuples():
                dfnucabundances.at[contrib.cellindex, nucabundcolname] += massfrac * contrib.frac_of_cellmass

        print(f' particle {n} contributions took {time.perf_counter() - timestart:.1f} seconds '
              f'(total func time {time.perf_counter() - timefuncstart:.1f} s)')

    timestart = time.perf_counter()
    print('Adding up elemental abundances...', end='')
    elemisotopes = {}
    for colname in dfnucabundances.columns:
        if not colname.startswith('X_'):
            continue
        atomic_number = at.get_atomic_number(colname[2:].rstrip('0123456789'))
        if atomic_number in elemisotopes:
            elemisotopes[atomic_number].append(colname)
        else:
            elemisotopes[atomic_number] = [colname]

    # elcolnames = [f'X_{artistools.get_elsymbol(Z)}' for Z in range(max(atomic_numbers))]
    dfelabundances = pd.DataFrame({
        'inputcellid': dfnucabundances.index,
        **{f'X_{at.get_elsymbol(atomic_number)}': dfnucabundances.eval(f'{" + ".join(elemisotopes[atomic_number])}')
            if atomic_number in elemisotopes else np.zeros(len(active_inputcellids))
            for atomic_number in range(1, max(elemisotopes.keys()) + 1)}}, index=dfnucabundances.index)
    print(f' took {time.perf_counter() - timestart:.1f} seconds')

    print('creating dfmodel')
    dfmodel = dfmodel.merge(dfnucabundances, how='left', left_on='inputcellid', right_on='inputcellid')
    dfmodel.fillna(0., inplace=True)
    # print(dfmodel.iloc[61615:61625])

    print('creating dfelabundances')
    # print(dfelabundances)
    dfelabundances = dfmodel[['inputcellid']].merge(dfelabundances, how='left')  # add cells with no traj contributions
    dfelabundances.fillna(0., inplace=True)
    # print(dfelabundances.iloc[61615:61625])
    return dfmodel, dfelabundances


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
        rowdict[f'X_{at.get_elsymbol(row.Z)}{A}'] = row.massfrac

    modeldata = []
    for mgi, densityrow in dfdensities.iterrows():
        # print(mgi, densityrow)
        modeldata.append(dict(inputcellid=mgi + 1, velocity_outer=densityrow['velocity_outer'],
                         logrho=math.log10(densityrow['rho']), **rowdict))
    # print(modeldata)

    dfmodel = pd.DataFrame(modeldata)
    # print(dfmodel)
    at.inputmodel.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, modelpath=Path(args.outputpath))


if __name__ == "__main__":
    main()
