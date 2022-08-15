#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argcomplete
import argparse
import math
# import os.path

import numpy as np
import pandas as pd
from pathlib import Path
# import matplotlib
import matplotlib.pyplot as plt


import artistools as at
import artistools.inputmodel


def get_model_recombenergy(dfbinding, args):
        args.inputfile = Path('/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/artis_runs/kilonova_SFHo_1d_0p05day_ufix_betaminusthermdelay_fixinitialye_verylate_redo')

        print(f'Reading {args.inputfile}')
        dfmodel, t_model_init_days, vmax = at.inputmodel.get_modeldata(
            args.inputfile, get_abundances=False)

        t_model_init_seconds = t_model_init_days * 24 * 60 * 60
        print(f'Model is defined at {t_model_init_days} days ({t_model_init_seconds:.4f} seconds)')

        msun_g = 1.989e33
        amu_g = 1.66054e-24  # 1 atomic mass unit in grams
        ev_erg = 1.60218e-12  # 1 eV in erg
        mass_msun_rho = dfmodel['cellmass_grams'].sum() / msun_g

        mass_msun_accounted = 0.
        model_el_binding_en_ev = {}
        model_tot_binding_en_ev = 0.
        for column in dfmodel.columns:
            if column.startswith('X_'):
                species = column.replace('X_', '')
                speciesabund_g = np.dot(dfmodel[column], dfmodel['cellmass_grams'])

                species_mass_msun = speciesabund_g / msun_g

                if species[-1].isdigit():  # isotopic species
                    atomic_number = at.get_atomic_number(species)
                    elsymb = at.get_elsymbol(atomic_number)
                    massnumber = int(species.lstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                    matchrows = dfbinding.query('Z == @atomic_number')
                    if matchrows.empty:
                        binding_en_ev = 0.
                        # print(f'No binding energy for Z={atomic_number}')
                    else:
                        binding_en_ev = matchrows.iloc[0]['TotBEn']
                    # print(species, atomic_number, massnumber, el_binding_en_ev)
                    contrib_binding_en_ev = speciesabund_g / (massnumber * amu_g) * binding_en_ev

                    model_el_binding_en_ev[elsymb] = model_el_binding_en_ev.get(elsymb, 0.) + contrib_binding_en_ev
                    mass_msun_accounted += species_mass_msun
                    model_tot_binding_en_ev += contrib_binding_en_ev

                elif species.lower() != 'fegroup':  # ignore special group abundance
                    pass

                else:
                    pass

        def sortkey(item):
            return at.get_atomic_number(item[0])

        for elsymb, binding_en_ev in sorted(model_el_binding_en_ev.items(), key=sortkey):
            zstr = f'Z={atomic_number}'
            print(f'{zstr:>5} {binding_en_ev * ev_erg} erg')

        print(f'Total electron binding energy {model_tot_binding_en_ev * ev_erg:.3e} erg')
        print(f'Mass from density {mass_msun_rho:.3e}')
        print(f'Mass from sum of isotopes {mass_msun_accounted:.3e}')


def get_particle_elec_binding_energy_per_gram(dictbinding, particleid, time_s):
    # find the closest timestep to the required time
    nts = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(particleid, time_s)
    memberfilename = f'./Run_rprocess/nz-plane{nts:05d}'

    dftrajnucabund, traj_time_s = at.inputmodel.rprocess_from_trajectory.get_trajectory_timestepfile_nuc_abund(
        particleid=particleid, memberfilename=memberfilename)

    dftrajnucabund['Z_be_tot_ev'] = [dictbinding.get(Z, 0.) for Z in dftrajnucabund['Z']]

    amu_g = 1.66054e-24  # 1 atomic mass unit in grams

    # frac_unaccounted = dftrajnucabund[dftrajnucabund['Z_be_tot_ev'] == 0].massfrac.sum()
    # print(f'frac_unaccounted {frac_unaccounted}')
    # if frac_unaccounted > .3:
    #     print(dftrajnucabund)
    # assert frac_unaccounted < 0.3

    dftrajnucabund.eval('recombenergy_ev_per_gram = Z_be_tot_ev * massfrac / (Z + N) / @amu_g', inplace=True)

    # contrib_binding_en_ev = speciesabund_g / (massnumber * amu_g) * binding_en_ev

    # print(dftrajnucabund)

    return dftrajnucabund['recombenergy_ev_per_gram'].sum()


def get_particle_nucenergy_released(particleid, tmin_s, time_s_end):
    memberfilename = './Run_rprocess/energy_thermo.dat'
    erg_to_ev = 6.242e+11
    with at.inputmodel.rprocess_from_trajectory.open_tar_file_or_extracted(particleid, memberfilename) as fthermo:
        dfthermo = pd.read_csv(fthermo, delim_whitespace=True, usecols=['#count', 'time/s', 'Qdot', 'Ye'])
        dfthermo.rename(columns={'time/s': 'time_s'}, inplace=True)
        dfthermo.query("time_s >= @tmin_s", inplace=True)
        dfthermo.query("time_s <= @time_s_end", inplace=True)
        en_released_ev_per_gram = np.trapz(y=dfthermo['Qdot'], x=dfthermo['time_s']) * erg_to_ev
        # print(dfthermo)
    return en_released_ev_per_gram


def get_particles_recomb_nuc_energy(dfbinding):
    dfsnapshot = at.inputmodel.modelfromhydro.read_ejectasnapshot('/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/Archive/Astronomy/Mergers/SFHo_snapshot')
    dfsnapshot.sort_values('ye', inplace=True)

    dictbinding = {Z: be_tot_ev for Z, be_tot_ev in dfbinding[['Z', 'TotBEn']].itertuples(index=False)}

    tmin_s = 10
    time_s = 6 * 3600

    msun_g = 1.989e33

    ye_list = []
    elecbinding_en_list = []
    nuclear_released_en_list = []
    for particleid, ye, pmass in dfsnapshot[['id', 'ye', 'pmass']].itertuples(index=False):
        try:
            elecbinding_en = get_particle_elec_binding_energy_per_gram(dictbinding=dictbinding, particleid=particleid, time_s=time_s)
            nuc_en_released = get_particle_nucenergy_released(particleid=particleid, tmin_s=tmin_s, time_s_end=time_s)
            ye_list.append(ye)
            elecbinding_en_list.append(elecbinding_en)
            nuclear_released_en_list.append(nuc_en_released)
            # print(particleid, ye, elecbinding_en)
        except FileNotFoundError:
            pass
            # print(f' WARNING particle {particleid} not found! ')

    dfrecomb = pd.DataFrame(
        {'ye': ye_list, 'recombenergy_ev_per_gram': elecbinding_en_list, 'nuclear_released_ev_per_gram': nuclear_released_en_list})

    print(dfrecomb)

    fig, ax = plt.subplots(
        nrows=1, ncols=1, sharex=True, sharey=False, figsize=(6, 4),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0})

    ax.plot(dfrecomb['ye'], dfrecomb['recombenergy_ev_per_gram'], label=f'electron binding for composition at {time_s:.1e} s [eV/g]', lw=0, marker='.')
    ax.plot(dfrecomb['ye'], dfrecomb['nuclear_released_ev_per_gram'], label=f'nuclear release between {tmin_s:.1e} s and {time_s:.1e} s [eV/g]', lw=0, marker='.')
    ax.legend(loc='best', handlelength=2, frameon=False, numpoints=1)
    plt.xlabel('Ye')
    # plt.ylabel('eV / g')
    plt.yscale('log')
    ax.set_ylim(bottom=1e24, top=1e33)

    plt.savefig('recomb.pdf', format='pdf')


def addargs(parser):
    parser.add_argument('-inputfile', '-i', default='model.txt',
                        help='Path of input file or folder containing model.txt')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Get the recombination energy from fully ionised to fully neutral for an input model.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    with open(at.config['path_datadir'] / 'ElBiEn_2007.txt', 'r') as fbinding:
        for _ in range(11):
            header = fbinding.readline().lstrip(' #').split()
        # print(header)
        dfbinding = pd.read_csv(fbinding, delim_whitespace=True, names=header)
        # print(dfbinding)

    # get_model_recombenergy(dfbinding, args)
    get_particles_recomb_nuc_energy(dfbinding)


if __name__ == "__main__":
    main()
