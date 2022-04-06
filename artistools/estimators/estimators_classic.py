from pathlib import Path
import gzip
import glob
import os

import artistools as at


def get_atomic_composition(modelpath):
    """Read ion list from output file"""
    atomic_composition = {}

    output = open(modelpath / "output_0-0.txt", 'r').read().splitlines()
    ioncount = 0
    for row in output:
        if row.split()[0] == '[input.c]':
            split_row = row.split()
            if split_row[1] == 'element':
                Z = int(split_row[4])
                ioncount = 0
            elif split_row[1] == 'ion':
                ioncount += 1
                atomic_composition[Z] = ioncount
    return atomic_composition


def parse_ion_row_classic(row, outdict, atomic_composition):
    outdict['populations'] = {}

    elements = atomic_composition.keys()

    i = 6  # skip first 6 numbers in est file. These are n, TR, Te, W, TJ, grey_depth.
    # Numbers after these 6 are populations
    for atomic_number in elements:
        for ion_stage in range(1, atomic_composition[atomic_number] + 1):
            value_thision = float(row[i])
            outdict['populations'][(atomic_number, ion_stage)] = value_thision
            i += 1

            elpop = outdict['populations'].get(atomic_number, 0)
            outdict['populations'][atomic_number] = elpop + value_thision

            totalpop = outdict['populations'].get('total', 0)
            outdict['populations']['total'] = totalpop + value_thision


def get_estimator_files(modelpath):
    estimfiles = (glob.glob(os.path.join(modelpath, 'estimators_????.out'), recursive=True) +
                  glob.glob(os.path.join(modelpath, 'estimators_????.out.gz'), recursive=True) +
                  glob.glob(os.path.join(modelpath, '*/estimators_????.out'), recursive=True) +
                  glob.glob(os.path.join(modelpath, '*/estimators_????.out.gz'), recursive=True))

    return estimfiles


def get_first_ts_in_run_directory(modelpath):
    folderlist_all = tuple(sorted([child for child in Path(modelpath).iterdir() if child.is_dir()]) + [Path(modelpath)])

    first_timesteps_in_dir = {}

    for folder in folderlist_all:
        if os.path.isfile(folder/'output_0-0.txt'):
            with open(folder/'output_0-0.txt', 'r') as output_0:
                timesteps_in_dir = [line.strip('...\n').split(' ')[-1] for line in output_0
                                    if '[debug] update_packets: updating packet 0 for timestep' in line]
            first_ts = timesteps_in_dir[0]
            first_timesteps_in_dir[str(folder)] = int(first_ts)

    return first_timesteps_in_dir


def read_classic_estimators(modelpath, modeldata, readonly_mgi=False, readonly_timestep=False):
    estimfiles = get_estimator_files(modelpath)
    if not estimfiles:
        print("No estimator files found")
        return False
    print(f'Reading {len(estimfiles)} estimator files...')

    first_timesteps_in_dir = get_first_ts_in_run_directory(modelpath)
    atomic_composition = get_atomic_composition(modelpath)

    inputparams = at.get_inputparams(modelpath)
    ndimensions = inputparams['n_dimensions']

    estimators = {}
    for estfile in estimfiles:
        opener = gzip.open if estfile.endswith('.gz') else open

        # If classic plots break it's probably getting first timestep here
        # Try either of the next two lines
        timestep = first_timesteps_in_dir[str(estfile).split('/')[0]]  # get the starting timestep for the estfile
        # timestep = first_timesteps_in_dir[str(estfile[:-20])]
        # timestep = 0  # if the first timestep in the file is 0 then this is fine
        with opener(estfile, 'rt') as estfile:
            modelgridindex = -1
            for line in estfile:
                row = line.split()
                if int(row[0]) < int(modelgridindex):
                    timestep += 1
                elif int(row[0]) == int(modelgridindex):
                    timestep += 1
                modelgridindex = int(row[0])

                if readonly_mgi is False or modelgridindex in readonly_mgi:
                    if readonly_timestep is False or timestep in readonly_timestep:
                        estimators[(timestep, modelgridindex)] = {}

                        if ndimensions == 1:
                            estimators[(timestep, modelgridindex)]['velocity_outer'] = modeldata[
                                'velocity_outer'][modelgridindex]

                        estimators[(timestep, modelgridindex)]['TR'] = float(row[1])
                        estimators[(timestep, modelgridindex)]['Te'] = float(row[2])
                        estimators[(timestep, modelgridindex)]['W'] = float(row[3])
                        estimators[(timestep, modelgridindex)]['TJ'] = float(row[4])

                        parse_ion_row_classic(row, estimators[(timestep, modelgridindex)], atomic_composition)

                        # heatingrates[tid].ff, heatingrates[tid].bf, heatingrates[tid].collisional, heatingrates[tid].gamma,
                        # coolingrates[tid].ff, coolingrates[tid].fb, coolingrates[tid].collisional, coolingrates[tid].adiabatic)

                        estimators[(timestep, modelgridindex)]['heating_ff'] = float(row[-9])
                        estimators[(timestep, modelgridindex)]['heating_bf'] = float(row[-8])
                        estimators[(timestep, modelgridindex)]['heating_coll'] = float(row[-7])
                        estimators[(timestep, modelgridindex)]['heating_dep'] = float(row[-6])

                        estimators[(timestep, modelgridindex)]['cooling_ff'] = float(row[-5])
                        estimators[(timestep, modelgridindex)]['cooling_fb'] = float(row[-4])
                        estimators[(timestep, modelgridindex)]['cooling_coll'] = float(row[-3])
                        estimators[(timestep, modelgridindex)]['cooling_adiabatic'] = float(row[-2])

                        # estimators[(timestep, modelgridindex)]['cooling_coll - heating_coll'] = \
                        #     estimators[(timestep, modelgridindex)]['cooling_coll'] - estimators[(timestep, modelgridindex)]['heating_coll']
                        #
                        # estimators[(timestep, modelgridindex)]['cooling_fb - heating_bf'] = \
                        #     estimators[(timestep, modelgridindex)]['cooling_fb'] - estimators[(timestep, modelgridindex)]['heating_bf']
                        #
                        # estimators[(timestep, modelgridindex)]['cooling_ff - heating_ff'] = \
                        #     estimators[(timestep, modelgridindex)]['cooling_ff'] - estimators[(timestep, modelgridindex)]['heating_ff']
                        #
                        # estimators[(timestep, modelgridindex)]['cooling_adiabatic - heating_dep'] = \
                        #     estimators[(timestep, modelgridindex)]['cooling_adiabatic'] - estimators[(timestep, modelgridindex)]['heating_dep']

                        estimators[(timestep, modelgridindex)]['energy_deposition'] = float(row[-1])

    return estimators
