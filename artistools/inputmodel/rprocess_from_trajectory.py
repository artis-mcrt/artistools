#!/usr/bin/env python3

import argparse
import math
import multiprocessing
import tarfile
import time
from pathlib import Path
import artistools as at
import numpy as np
import pandas as pd
from functools import partial

traj_root = Path('/Users/luke/Dropbox/Archive/Mergers/SFHo/')


def get_elemabund_from_nucabund(dfnucabund):
    dictelemabund = {}
    for atomic_number in range(1, dfnucabund.Z.max() + 1):
        dictelemabund[f'X_{at.get_elsymbol(atomic_number)}'] = (
            dfnucabund.query('Z == @atomic_number', inplace=False).massfrac.sum())
    return dictelemabund


def get_traj_tarpath(particleid):
    return Path(traj_root, f'{particleid}.tar.xz')


def open_tar_file_or_extracted(particleid, memberfilename):
    """ memberfilename is within the trajectory tarfile/folder, eg. ./Run_rprocess/evol.dat """
    path_extracted_file = Path(traj_root, str(particleid), memberfilename)
    if path_extracted_file.is_file():
        return open(path_extracted_file, mode='r')
    else:
        return tarfile.open(get_traj_tarpath(particleid), mode='r:*').extractfile(member=memberfilename)


def get_closest_network_timestep(particleid, t_model_s):
    try:
        with open_tar_file_or_extracted(particleid, './Run_rprocess/evol.dat') as evolfile:
            dfevol = pd.read_csv(
                evolfile, delim_whitespace=True, comment='#', usecols=[0, 1], names=['nstep', 'timesec'])
            idx = np.abs(dfevol.timesec.values - t_model_s).argmin()
            nts = dfevol.nstep.values[idx]

    except FileNotFoundError:
        return None

    return nts


def get_trajectory_nuc_abund(particleid, memberfilename):
    with open_tar_file_or_extracted(particleid, memberfilename) as trajfile:

        # with open(trajfile) as ftraj:
        _, str_t_model_init_seconds, _, rho, _, _ = trajfile.readline().split()
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


def get_trajectory_nuc_abund_group(t_model_s, particlegroup):
    particleid, dfthisparticlecontribs = particlegroup
    # find the closest timestep to the required time
    nts = get_closest_network_timestep(particleid, t_model_s)

    if nts is None:
        print(f' WARNING {get_traj_tarpath(particleid)} not found! '
              f'Contributes up to {dfthisparticlecontribs.frac_of_cellmass.max() * 100:.1f}% mass of some cells')
        return None

    dftrajnucabund, traj_time_s = get_trajectory_nuc_abund(
        particleid, f'./Run_rprocess/nz-plane{nts:05d}')

    massfractotal = dftrajnucabund.massfrac.sum()
    dftrajnucabund.query('Z >= 1', inplace=True)
    dftrajnucabund['nucabundcolname'] = [f'X_{at.get_elsymbol(int(row.Z))}{int(row.N + row.Z)}'
                                         for row in dftrajnucabund.itertuples()]

    colmassfracs = list(dftrajnucabund[['nucabundcolname', 'massfrac']].itertuples(index=False))
    colmassfracs.sort(key=lambda row: at.get_z_a_nucname(row[0]))

    # print(f'trajectory particle id {particleid} massfrac sum: {massfractotal:.2f}')
    # print(f' grid snapshot: {t_model_s:.2e} s, network: {traj_time_s:.2e} s (timestep {nts})')
    assert np.isclose(massfractotal, 1., rtol=0.02)
    assert np.isclose(traj_time_s, t_model_s, rtol=0.2, atol=1.)
    return {
        nucabundcolname: massfrac
        for nucabundcolname, massfrac in colmassfracs}


def get_cellmodelrow(dict_traj_nuc_abund, timefuncstart, active_inputcellcount, cellgroup):
    n, (cellindex, dfthiscellcontribs) = cellgroup
    # if len(listcellnucabundances) > 20:
    #     break
    if len(dfthiscellcontribs) < 10:
        return None
    contribparticles = [
        (dict_traj_nuc_abund[particleid], frac_of_cellmass)
        for particleid, frac_of_cellmass in dfthiscellcontribs[
            ['particleid', 'frac_of_cellmass']].itertuples(index=False)
        if particleid in dict_traj_nuc_abund]

    nucabundcolnames = set([col for trajnucabund, _ in contribparticles for col in trajnucabund.keys()])

    row = {'inputcellid': cellindex}
    for nucabundcolname in nucabundcolnames:
        abund = sum(
            [frac_of_cellmass * traj_nuc_abund.get(nucabundcolname, 0.)
             for traj_nuc_abund, frac_of_cellmass in contribparticles])
        row[nucabundcolname] = abund

    if n % 100 == 0:
        functime = time.perf_counter() - timefuncstart
        print(f'cell id {cellindex:6d} ('
              f'{n:4d} of {active_inputcellcount:4d}, {n / active_inputcellcount * 100:4.1f}%) '
              f' contributing {len(dfthiscellcontribs):4d} particles.'
              f' total func time {functime:.1f} s, {n / functime:.1f} cell/s,'
              f' expected time: {functime / n * active_inputcellcount:.1f}')
    return row


def add_abundancecontributions(gridcontribpath, dfmodel, t_model_days):
    """ contribute trajectory network calculation abundances to model cell abundances """
    timefuncstart = time.perf_counter()
    t_model_s = t_model_days * 86400
    if 'X_Fegroup' not in dfmodel.columns:
        dfmodel = pd.concat([dfmodel, pd.DataFrame({'X_Fegroup': np.ones(len(dfmodel))})], axis=1)

    dfcontribs = pd.read_csv(Path(gridcontribpath, 'gridcontributions.txt'), delim_whitespace=True,
                             dtype={0: int, 1: int, 2: float})

    minparticlesincell = 10
    active_inputcellids = [
        cellindex for cellindex, dfthiscellcontribs in dfcontribs.groupby('cellindex')
        if len(dfthiscellcontribs) >= minparticlesincell]
    dfcontribs.query('cellindex in @active_inputcellids', inplace=True)
    active_inputcellids = dfcontribs.cellindex.unique()
    active_inputcellcount = len(active_inputcellids)
    dfcontribs_particlegroups = dfcontribs.groupby('particleid')
    particle_count = len(dfcontribs_particlegroups)

    print(f'{active_inputcellcount} of {len(dfmodel)} model cells have >={minparticlesincell} particles contributing'
          f'({len(dfcontribs)} total contributions from {particle_count} particles after filter)')

    listcellnucabundances = []
    print('getting trajectory abundances...')
    trajnucabundworker = partial(get_trajectory_nuc_abund_group, t_model_s)
    if at.num_processes > 1:
        print(f"Reading packets files with {at.num_processes} processes")
        with multiprocessing.Pool(processes=at.num_processes) as pool:
            list_traj_nuc_abund = pool.map(trajnucabundworker, dfcontribs_particlegroups)
            pool.close()
            pool.join()
            # pool.terminate()
    else:
        list_traj_nuc_abund = [trajnucabundworker(particlegroup) for particlegroup in dfcontribs_particlegroups]

    dict_traj_nuc_abund = {
        particleid: dftrajnucabund
        for particleid, dftrajnucabund in zip(dfcontribs_particlegroups.groups, list_traj_nuc_abund)
        if dftrajnucabund is not None}

    dfcontribs_cellgroups = dfcontribs.groupby('cellindex')
    cellabundworker = partial(get_cellmodelrow, dict_traj_nuc_abund, timefuncstart, active_inputcellcount)
    if at.num_processes > 1:
        print(f"Generating cell abundances with {at.num_processes} processes")
        with multiprocessing.Pool(processes=at.num_processes) as pool:
            listcellnucabundances = pool.map(cellabundworker, enumerate(dfcontribs_cellgroups, 1))
            pool.close()
            pool.join()
            # pool.terminate()
    else:
        listcellnucabundances = [cellabundworker(n_cellgroup) for n_cellgroup in enumerate(dfcontribs_cellgroups, 1)]

    listcellnucabundances = [x for x in listcellnucabundances if x is not None]

    timestart = time.perf_counter()
    print('Creating dfnucabundances...', end='', flush=True)
    dfnucabundances = pd.DataFrame(listcellnucabundances)
    dfnucabundances.set_index('inputcellid', drop=False, inplace=True)
    dfnucabundances.index.name = None
    dfnucabundances.fillna(0., inplace=True)
    print(f' took {time.perf_counter() - timestart:.1f} seconds')

    timestart = time.perf_counter()
    print('Adding up elemental abundances...', end='', flush=True)
    elemisotopes = {}
    for colname in dfnucabundances.columns:
        if not colname.startswith('X_'):
            continue
        atomic_number = at.get_atomic_number(colname[2:].rstrip('0123456789'))
        if atomic_number in elemisotopes:
            elemisotopes[atomic_number].append(colname)
        else:
            elemisotopes[atomic_number] = [colname]

    dfelabundances_partial = pd.DataFrame({
        'inputcellid': dfnucabundances.inputcellid,
        **{f'X_{at.get_elsymbol(atomic_number)}': dfnucabundances.eval(f'{" + ".join(elemisotopes[atomic_number])}')
            if atomic_number in elemisotopes else np.zeros(len(dfnucabundances))
            for atomic_number in range(1, max(elemisotopes.keys()) + 1)}}, index=dfnucabundances.index)
    print(f' took {time.perf_counter() - timestart:.1f} seconds')

    timestart = time.perf_counter()
    print('creating dfmodel...', end='', flush=True)
    dfmodel = dfmodel.merge(dfnucabundances, how='left', left_on='inputcellid', right_on='inputcellid')
    dfmodel.fillna(0., inplace=True)
    print(f' took {time.perf_counter() - timestart:.1f} seconds')

    timestart = time.perf_counter()
    print('creating dfelabundances...', end='', flush=True)
    # add cells with no traj contributions
    dfelabundances = dfmodel[['inputcellid']].merge(
        dfelabundances_partial, how='left', left_on='inputcellid', right_on='inputcellid')
    dfnucabundances.set_index('inputcellid', drop=False, inplace=True)
    dfnucabundances.index.name = None
    dfelabundances.fillna(0., inplace=True)
    print(f' took {time.perf_counter() - timestart:.1f} seconds')

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

    # particleid = 88969  # Ye = 9.63284224E-02
    particleid = 133371  # Ye = 0.403913230
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
    with open(Path(args.outputpath, 'gridcontributions.txt'), 'w') as fcontribs:
        fcontribs.write('particleid cellindex frac_of_cellmass\n')
        for cell in dfmodel.itertuples(index=False):
            fcontribs.write(f'{particleid} {cell.inputcellid} {1.}\n')


if __name__ == "__main__":
    main()
