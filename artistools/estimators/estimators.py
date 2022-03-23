#!/usr/bin/env python3
"""Functions for reading and processing estimator files.

Examples are temperatures, populations, and heating/cooling rates.
"""
# import math
import math
import multiprocessing
import sys
from collections import namedtuple
from functools import lru_cache, partial, reduce
# from itertools import chain
from pathlib import Path

import pandas as pd

import artistools as at
import artistools.nltepops


variableunits = {
    'time': 'days',
    'gamma_NT': '/s',
    'gamma_R_bfest': '/s',
    'TR': 'K',
    'Te': 'K',
    'TJ': 'K',
    'nne': 'e-/cm3',
    'heating': 'erg/s/cm3',
    'heating_dep/total_dep': 'Ratio',
    'cooling': 'erg/s/cm3',
    'velocity': 'km/s',
    'velocity_outer': 'km/s',
}

variablelongunits = {
    'heating_dep/total_dep': '',
    'TR': 'Temperature [K]',
    'Te': 'Temperature [K]',
    'TJ': 'Temperature [K]',
}

dictlabelreplacements = {
    'lognne': 'Log nne',
    'Te': 'T$_e$',
    'TR': 'T$_R$',
    'gamma_NT': r'$\Gamma_{\rm non-thermal}$ [s$^{-1}$]',
    'gamma_R_bfest': r'$\Gamma_{\rm phot}$ [s$^{-1}$]',
    'heating_dep/total_dep': 'Heating fraction',
}


def apply_filters(xlist, ylist, args):
    filterfunc = at.get_filterfunc(args)

    if filterfunc is not None:
        ylist = filterfunc(ylist)

    return xlist, ylist


def get_ionrecombrates_fromfile(filename):
    """WARNING: copy pasted from artis-atomic! replace with a package import soon ionstage is the lower ion stage."""
    print(f'Reading {filename}')

    header_row = []
    with open(filename, 'r') as filein:
        while True:
            line = filein.readline()
            if line.strip().startswith('TOTAL RECOMBINATION RATE'):
                line = filein.readline()
                line = filein.readline()
                header_row = filein.readline().strip().replace(' n)', '-n)').split()
                break

        if not header_row:
            print("ERROR: no header found")
            sys.exit()

        index_logt = header_row.index('log(T)')
        index_low_n = header_row.index('RRC(low-n)')
        index_tot = header_row.index('RRC(total)')

        recomb_tuple = namedtuple("recomb_tuple", ['logT', 'RRC_low_n', 'RRC_total'])
        records = []
        for line in filein:
            row = line.split()
            if row:
                if len(row) != len(header_row):
                    print('Row contains wrong number of items for header:')
                    print(header_row)
                    print(row)
                    sys.exit()
                records.append(recomb_tuple(
                    *[float(row[index]) for index in [index_logt, index_low_n, index_tot]]))

    dfrecombrates = pd.DataFrame.from_records(records, columns=recomb_tuple._fields)
    return dfrecombrates


def get_units_string(variable):
    if variable in variableunits:
        return f' [{variableunits[variable]}]'
    elif variable.split('_')[0] in variableunits:
        return f' [{variableunits[variable.split("_")[0]]}]'
    return ''


def parse_estimfile(estfilepath, modelpath, get_ion_values=True, get_heatingcooling=True):
    """Generate timestep, modelgridindex, dict from estimator file."""
    # itstep = at.get_inputparams(modelpath)['itstep']

    with at.zopen(estfilepath, 'rt') as estimfile:
        timestep = -1
        modelgridindex = -1
        estimblock = {}
        for line in estimfile:
            row = line.split()
            if not row:
                continue

            if row[0] == 'timestep':
                # yield the previous block before starting a new one
                if timestep >= 0 and modelgridindex >= 0:
                    yield timestep, modelgridindex, estimblock

                timestep = int(row[1])
                # if timestep > itstep:
                #     print(f"Dropping estimator data from timestep {timestep} and later (> itstep {itstep})")
                #     # itstep in input.txt is updated by ARTIS at every timestep, so the data beyond here
                #     # could be half-written to disk and cause parsing errors
                #     return

                modelgridindex = int(row[3])
                # print(f'Timestep {timestep} cell {modelgridindex}')

                estimblock = {}
                emptycell = (row[4] == 'EMPTYCELL')
                estimblock['emptycell'] = emptycell
                if not emptycell:
                    # will be TR, Te, W, TJ, nne
                    for variablename, value in zip(row[4::2], row[5::2]):
                        estimblock[variablename] = float(value)
                    estimblock['lognne'] = math.log10(estimblock['nne']) if estimblock['nne'] > 0 else float('-inf')

            elif row[1].startswith('Z=') and get_ion_values:
                variablename = row[0]
                if row[1].endswith('='):
                    atomic_number = int(row[2])
                    startindex = 3
                else:
                    atomic_number = int(row[1].split('=')[1])
                    startindex = 2

                estimblock.setdefault(variablename,  {})

                for ion_stage_str, value in zip(row[startindex::2], row[startindex + 1::2]):
                    if ion_stage_str.strip() in ['SUM:', '(or']:
                        continue

                    try:
                        ion_stage = int(ion_stage_str.rstrip(':'))
                    except ValueError:
                        if variablename == 'populations' and ion_stage_str.startswith(at.get_elsymbol(atomic_number)):
                            estimblock[variablename][ion_stage_str.rstrip(':')] = float(value)
                        else:
                            print(ion_stage_str, at.get_elsymbol(atomic_number))
                            print(f'Cannot parse row: {row}')
                        continue

                    value_thision = float(value.rstrip(','))

                    estimblock[variablename][(atomic_number, ion_stage)] = value_thision

                    if variablename in ['Alpha_R*nne', 'AlphaR*nne']:
                        estimblock.setdefault('Alpha_R', {})
                        estimblock['Alpha_R'][(atomic_number, ion_stage)] = value_thision / estimblock['nne']

                    else:  # variablename == 'populations':

                        # contribute the ion population to the element population
                        estimblock[variablename].setdefault(atomic_number, 0.)
                        estimblock[variablename][atomic_number] += value_thision

                if variablename == 'populations':
                    # contribute the element population to the total population
                    estimblock['populations'].setdefault('total', 0.)
                    estimblock['populations']['total'] += estimblock['populations'][atomic_number]
                    estimblock['nntot'] = estimblock['populations']['total']

            elif row[0] == 'heating:' and get_heatingcooling:
                for heatingtype, value in zip(row[1::2], row[2::2]):
                    key = 'heating_' + heatingtype if not heatingtype.startswith('heating_') else heatingtype
                    estimblock[key] = float(value)

                if 'heating_gamma/gamma_dep' in estimblock and estimblock['heating_gamma/gamma_dep'] > 0:
                    estimblock['gamma_dep'] = (
                        estimblock['heating_gamma'] /
                        estimblock['heating_gamma/gamma_dep'])
                elif 'heating_dep/total_dep' in estimblock and estimblock['heating_dep/total_dep'] > 0:
                    estimblock['total_dep'] = (
                        estimblock['heating_dep'] /
                        estimblock['heating_dep/total_dep'])

            elif row[0] == 'cooling:' and get_heatingcooling:
                for coolingtype, value in zip(row[1::2], row[2::2]):
                    estimblock['cooling_' + coolingtype] = float(value)

    # reached the end of file
    if timestep >= 0 and modelgridindex >= 0:
        yield timestep, modelgridindex, estimblock


@at.diskcache(ignorekwargs=['printfilename'], quiet=False, funcdepends=parse_estimfile, savezipped=True)
def read_estimators_from_file(folderpath, modelpath, arr_velocity_outer, mpirank, printfilename=False,
                              get_ion_values=True, get_heatingcooling=True):

    estimators_thisfile = {}
    estimfilename = f'estimators_{mpirank:04d}.out'
    estfilepath = Path(folderpath, estimfilename)
    if not estfilepath.is_file():
        estfilepath = Path(folderpath, estimfilename + '.gz')
        if not estfilepath.is_file():
            estfilepath = Path(folderpath, estimfilename + '.xz')
            if not estfilepath.is_file():
                # not worth printing and error, because ranks with no cells to update do not produce an estimator file
                # print(f'Warning: Could not find {estfilepath.relative_to(modelpath.parent)}')
                return {}

    if printfilename:
        filesize = Path(estfilepath).stat().st_size / 1024 / 1024
        print(f'Reading {estfilepath.relative_to(modelpath.parent)} ({filesize:.2f} MiB)')

    for fileblock_timestep, fileblock_modelgridindex, file_estimblock in parse_estimfile(
            estfilepath, modelpath, get_ion_values=get_ion_values, get_heatingcooling=get_heatingcooling):

        if arr_velocity_outer is not None:
            file_estimblock['velocity_outer'] = arr_velocity_outer[fileblock_modelgridindex]
            file_estimblock['velocity'] = file_estimblock['velocity_outer']

        estimators_thisfile[(fileblock_timestep, fileblock_modelgridindex)] = file_estimblock

    return estimators_thisfile


@lru_cache(maxsize=16)
@at.diskcache(savezipped=True, funcdepends=[read_estimators_from_file, parse_estimfile])
def read_estimators(modelpath, modelgridindex=None, timestep=None, get_ion_values=True, get_heatingcooling=True):
    """Read estimator files into a nested dictionary structure.

    Speed it up by only retrieving estimators for a particular timestep(s) or modelgrid cells.
    """

    if modelgridindex is None:
        match_modelgridindex = []
    elif hasattr(modelgridindex, '__iter__'):
        match_modelgridindex = tuple(modelgridindex)
    else:
        match_modelgridindex = (modelgridindex,)

    if -1 in match_modelgridindex:
        match_modelgridindex = []

    if timestep is None:
        match_timestep = []
    else:
        match_timestep = tuple(timestep) if hasattr(timestep, '__iter__') else (timestep,)

    if not modelpath.exists() and modelpath.parts[0] == 'codecomparison':
        return artistools.codecomparison.read_reference_estimators(modelpath, timestep=timestep, modelgridindex=modelgridindex)

    # print(f" matching cells {match_modelgridindex} and timesteps {match_timestep}")

    modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
    if 'velocity_outer' in modeldata.columns:
        arr_velocity_outer = tuple(list([float(v) for v in modeldata['velocity_outer'].values]))
    else:
        arr_velocity_outer = None

    mpiranklist = at.get_mpiranklist(modelpath, modelgridindex=match_modelgridindex, only_ranks_withgridcells=True)

    printfilename = len(mpiranklist) < 10

    estimators = {}
    for folderpath in at.get_runfolders(modelpath, timesteps=match_timestep):
        print(f'Reading {len(list(mpiranklist))} estimator files in {folderpath.relative_to(Path(modelpath).parent)}')

        processfile = partial(read_estimators_from_file, folderpath, modelpath, arr_velocity_outer,
                              get_ion_values=get_ion_values, get_heatingcooling=get_heatingcooling,
                              printfilename=printfilename)

        if at.num_processes > 1:
            with multiprocessing.Pool(processes=at.num_processes) as pool:
                arr_rankestimators = pool.map(processfile, mpiranklist)
                pool.close()
                pool.join()
                pool.terminate()
        else:
            arr_rankestimators = [processfile(rank) for rank in mpiranklist]

        for mpirank, estimators_thisfile in zip(mpiranklist, arr_rankestimators):
            dupekeys = list(sorted([k for k in estimators_thisfile if k in estimators]))
            for k in dupekeys:
                # dropping the lowest timestep is normal for restarts. Only warn about other cases
                if k[0] != dupekeys[0][0]:
                    filepath = Path(folderpath, f'estimators_{mpirank:04d}.out')
                    print(f'WARNING: Duplicate estimator block for (timestep, mgi) key {k}. '
                          f'Dropping block from {filepath}')

                del estimators_thisfile[k]

            estimators.update(estimators_thisfile)

    return estimators


def get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, keys, avgadjcells=0):
    """Get the average of estimators[(timestep, modelgridindex)][keys[0]]...[keys[-1]] across timesteps."""
    if isinstance(keys, str):
        keys = [keys]

    # reduce(lambda d, k: d[k], keys, dictionary) returns dictionary[keys[0]][keys[1]]...[keys[-1]]
    # applying all keys in the keys list

    # if single timestep, no averaging needed
    if not hasattr(timesteps, '__iter__'):
        return reduce(lambda d, k: d[k], [(timesteps, modelgridindex)] + keys, estimators)

    firsttimestepvalue = reduce(lambda d, k: d[k], [(timesteps[0], modelgridindex)] + keys, estimators)
    if isinstance(firsttimestepvalue, dict):
        dictout = {k: get_averaged_estimators(modelpath, estimators, timesteps, modelgridindex, keys + [k])
                   for k in firsttimestepvalue.keys()}

        return dictout
    else:
        tdeltas = at.get_timestep_times_float(modelpath, loc='delta')
        valuesum = 0
        tdeltasum = 0
        for timestep, tdelta in zip(timesteps, tdeltas):
            for mgi in range(modelgridindex - avgadjcells, modelgridindex + avgadjcells + 1):
                try:
                    valuesum += reduce(lambda d, k: d[k], [(timestep, mgi)] + keys, estimators) * tdelta
                    tdeltasum += tdelta
                except KeyError:
                    pass
        return valuesum / tdeltasum

    # except KeyError:
    #     if (timestep, modelgridindex) in estimators:
    #         print(f'Unknown x variable: {xvariable} for timestep {timestep} in cell {modelgridindex}')
    #     else:
    #         print(f'No data for cell {modelgridindex} at timestep {timestep}')
    #     print(estimators[(timestep, modelgridindex)])
    #     sys.exit()


def get_averageionisation(populations, atomic_number):
    free_electron_weighted_pop_sum = 0.
    found = False
    popsum = 0
    for key in populations.keys():
        if isinstance(key, tuple) and key[0] == atomic_number:
            found = True
            ion_stage = key[1]
            free_electron_weighted_pop_sum += populations[key] * (ion_stage - 1)
            popsum += populations[key]

    if not found:
        return float('NaN')

    return free_electron_weighted_pop_sum / populations[atomic_number]


def get_averageexcitation(modelpath, modelgridindex, timestep, atomic_number, ion_stage, T_exc):
    import artistools.nltepops
    dfnltepops = at.nltepops.read_files(modelpath, modelgridindex=modelgridindex, timestep=timestep)
    adata = at.atomic.get_levels(modelpath)
    ionlevels = adata.query('Z == @atomic_number and ion_stage == @ion_stage').iloc[0].levels

    energypopsum = 0
    ionpopsum = 0
    if dfnltepops.empty:
        return float('NaN')
    else:
        dfnltepops_ion = dfnltepops.query(
            'modelgridindex==@modelgridindex and timestep==@timestep and Z==@atomic_number & ion_stage==@ion_stage')

        k_b = 8.617333262145179e-05  # eV / K

        ionpopsum = dfnltepops_ion.n_NLTE.sum()
        energypopsum = dfnltepops_ion[dfnltepops_ion.level >= 0].eval(
            '@ionlevels.iloc[level].energy_ev.values * n_NLTE').sum()

        try:
            superlevelrow = dfnltepops_ion[dfnltepops_ion.level < 0].iloc[0]
            levelnumber_sl = dfnltepops_ion.level.max() + 1

            energy_boltzfac_sum = ionlevels.iloc[levelnumber_sl:].eval(
                'energy_ev * g * exp(- energy_ev / @k_b / @T_exc)').sum()

            boltzfac_sum = ionlevels.iloc[levelnumber_sl:].eval('g * exp(- energy_ev / @k_b / @T_exc)').sum()
            # adjust to the actual superlevel population from ARTIS
            energypopsum += energy_boltzfac_sum * superlevelrow.n_NLTE / boltzfac_sum
        except IndexError:
            # no superlevel
            pass

    return energypopsum / ionpopsum
