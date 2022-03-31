#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argcomplete
import argparse

import numpy as np
# import io
import pandas as pd
# import math
import multiprocessing
from functools import partial
from pathlib import Path
import matplotlib.pyplot as plt

import artistools as at
# import artistools.estimators


def plot_qdot(modelpath, dfpartcontrib, dfmodel, allparticledata, arr_time_artis_days, arr_time_gsi_days, pdfoutpath):
    try:
        depdata = at.get_deposition(modelpath=modelpath)
    except FileNotFoundError:
        print("Can't do qdot plot because no deposition.out file")
        return

    modelname = at.get_model_name(modelpath)

    tstart = depdata['tmid_days'].min()
    tend = depdata['tmid_days'].max()

    arr_heat = {}

    heatcols = ['hbeta', 'halpha', 'hbfis', 'hspof']  # , 'Qdot'

    for col in heatcols:
        arr_heat[col] = np.zeros_like(arr_time_gsi_days)

    modelmass = dfmodel.cellmass_grams.sum()
    print("model mass: {modelmass} g")
    dfpartcontrib = dfpartcontrib.query('particleid in @allparticledata.keys()')
    for cellindex, dfpartcontrib in dfpartcontrib.groupby('cellindex'):
        if cellindex >= len(dfmodel):
            continue
        cell_mass_frac = dfmodel.iloc[cellindex - 1].cellmass_grams / modelmass
        if cell_mass_frac == 0.:
            continue
        frac_of_cellmass_sum = dfpartcontrib.frac_of_cellmass.sum()

        for particleid, frac_of_cellmass in dfpartcontrib[['particleid', 'frac_of_cellmass']].itertuples(index=False):
            thisparticledata = allparticledata[particleid]
            for col in heatcols:
                arr_heat[col] += (
                    thisparticledata[col] * cell_mass_frac * frac_of_cellmass / frac_of_cellmass_sum)

    fig, axis = plt.subplots(
        nrows=1, ncols=1, sharex=False, sharey=False, figsize=(6, 4),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0})

    axis.set_ylim(bottom=1e7, top=2e10)
    axis.set_xlim(left=tstart, right=tend)

    # axis.set_xscale('log')

    # axis.set_xlim(left=1., right=arr_time_artis[-1])
    axis.set_xlabel('Time [days]')
    axis.set_yscale('log')
    # axis.set_ylabel(f'X({strnuc})')
    axis.set_ylabel('Qdot [erg/g/s]')
    # arr_time_days, arr_qdot = zip(
    #     *[(t, qdot) for t, qdot in zip(arr_time_days, arr_qdot)
    #       if depdata['tmid_days'].min() <= t and t <= depdata['tmid_days'].max()])

    # axis.plot(arr_time_gsi_days, arr_heat['Qdot'],
    #           # linestyle='None',
    #           linewidth=2, color='black',
    #           # marker='x', markersize=8,
    #           label='Qdot GSI Network')
    #
    # axis.plot(depdata['tmid_days'], depdata['Qdot_ana_erg/g/s'],
    #           linewidth=2, color='red',
    #           # linestyle='None',
    #           # marker='+', markersize=15,
    #           label='Qdot ARTIS')

    axis.plot(arr_time_gsi_days, arr_heat['hbeta'],
              linewidth=2, color='black',
              linestyle='dashed',
              # marker='x', markersize=8,
              label=r'$\dot{Q}_\beta$ GSI Network')

    axis.plot(depdata['tmid_days'], depdata['Qdot_betaminus_ana_erg/g/s'],
              linewidth=2, color='red',
              linestyle='dashed',
              # marker='+', markersize=15,
              label=r'$\dot{Q}_\beta$ ARTIS')

    axis.plot(arr_time_gsi_days, arr_heat['halpha'],
              linewidth=2, color='black',
              linestyle='dotted',
              # marker='x', markersize=8,
              label=r'$\dot{Q}_\alpha$ GSI Network')

    axis.plot(depdata['tmid_days'], depdata['Qdotalpha_ana_erg/g/s'],
              linewidth=2, color='red',
              linestyle='dotted',
              # marker='+', markersize=15,
              label=r'$\dot{Q}_\alpha$ ARTIS')

    axis.plot(arr_time_gsi_days, arr_heat['hbfis'],
              linewidth=2,
              linestyle='dotted',
              # marker='x', markersize=8,
              # color='black',
              label=r'$\dot{Q}_{hbfis}$ GSI Network')

    axis.plot(arr_time_gsi_days, arr_heat['hspof'],
              linewidth=2,
              linestyle='dotted',
              # marker='x', markersize=8,
              # color='black',
              label=r'$\dot{Q}_{spof}$ GSI Network')

    axis.legend(loc='best', frameon=False, handlelength=1, ncol=3,
                numpoints=1)

    fig.suptitle(f'{modelname}', fontsize=10)
    plt.savefig(pdfoutpath, format='pdf')
    print(f'Saved {pdfoutpath}')


def plot_abund(
        modelpath, dfpartcontrib, allparticledata, arr_time_artis_days, arr_time_gsi_days,
        arr_strnuc, arr_abund_gsi, arr_abund_artis, t_model_init_days, dfcell, pdfoutpath, mgi):

    dfpartcontrib_thiscell = dfpartcontrib.query('cellindex == (@mgi + 1) and particleid in @allparticledata.keys()')
    frac_of_cellmass_sum = dfpartcontrib_thiscell.frac_of_cellmass.sum()
    print(f'frac_of_cellmass_sum: {frac_of_cellmass_sum} (can be < 1.0 because of missing particles)')

    for strnuc in arr_strnuc:
        arr_abund_gsi[strnuc] = np.zeros_like(arr_time_gsi_days)

    for particleid, frac_of_cellmass in dfpartcontrib_thiscell[
            ['particleid', 'frac_of_cellmass']].itertuples(index=False):
        frac_of_cellmass = dfpartcontrib_thiscell.query('particleid == @particleid').frac_of_cellmass.sum()

        for strnuc in arr_strnuc:
            arr_abund_gsi[strnuc] += (
                allparticledata[particleid][strnuc] * frac_of_cellmass / frac_of_cellmass_sum)

    fig, axes = plt.subplots(
        nrows=len(arr_strnuc), ncols=1, sharex=False, sharey=False, figsize=(6, 1 + 2. * len(arr_strnuc)),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0})
    fig.subplots_adjust(top=0.8)
    # axis.set_xscale('log')

    modelname = at.get_model_name(modelpath)

    axes[-1].set_xlabel('Time [days]')
    for axis, strnuc in zip(axes, arr_strnuc):
        # print(arr_time_artis_days)
        xmin = arr_time_gsi_days.min() * 0.9
        xmax = arr_time_gsi_days.max() * 1.03
        axis.set_xlim(left=xmin, right=xmax)
        # axis.set_yscale('log')
        # axis.set_ylabel(f'X({strnuc})')
        axis.set_ylabel('Mass fraction')

        axis.plot(arr_time_gsi_days, arr_abund_gsi[strnuc],
                  # linestyle='None',
                  linewidth=2,
                  marker='x', markersize=8,
                  label=f'{strnuc} GSI Network', color='black')

        if strnuc in arr_abund_artis:
            axis.plot(arr_time_artis_days, arr_abund_artis[strnuc],
                      linewidth=2,
                      # linestyle='None',
                      # marker='+', markersize=15,
                      label=f'{strnuc} ARTIS', color='red')

        if f'X_{strnuc}' in dfcell:
            axis.plot(t_model_init_days, dfcell[f'X_{strnuc}'],
                      marker='+', markersize=15, markeredgewidth=2,
                      label=f'{strnuc} ARTIS inputmodel', color='blue')
            print(strnuc, arr_abund_gsi[strnuc][0], dfcell[f'X_{strnuc}'])

        axis.legend(loc='best', frameon=False, handlelength=1, ncol=1, numpoints=1)

    fig.suptitle(f'{modelname} cell {mgi}', y=0.995, fontsize=10)
    plt.savefig(pdfoutpath, format='pdf')
    print(f'Saved {pdfoutpath}')


def get_particledata(arr_time_s, arr_strnuc, particleid):

    try:
        nts_min = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            particleid, timesec=min(arr_time_s), cond='lessthan')
        nts_max = at.inputmodel.rprocess_from_trajectory.get_closest_network_timestep(
            particleid, timesec=max(arr_time_s), cond='greaterthan')

    except FileNotFoundError:
        # print(f'WARNING: Particle data not found for id {particleid}')
        return 'NONE', None

    # print(f'Reading data for particle id {particleid}...')
    particledata = {
        'Qdot': {},
        'hbeta': {},
        'halpha': {},
        'hbfis': {},
        'hspof': {},
        **{strnuc: {} for strnuc in arr_strnuc}
    }
    nstep_timesec = {}
    with at.inputmodel.rprocess_from_trajectory.open_tar_file_or_extracted(
            particleid, './Run_rprocess/heating.dat') as f:

        dfheating = pd.read_csv(
            f, delim_whitespace=True, usecols=[
                '#count', 'time/s', 'hbeta', 'halpha', 'hbfis', 'hspof'])
        heatcols = ['hbeta', 'halpha', 'hbfis', 'hspof']

        heatrates_in = {col: [] for col in heatcols}
        arr_time_s_source = []
        for _, row in dfheating.iterrows():
            nstep_timesec[row['#count']] = row['time/s']
            arr_time_s_source.append(row['time/s'])
            for col in heatcols:
                try:
                    heatrates_in[col].append(float(row[col]))
                except ValueError:
                    heatrates_in[col].append(float(row[col].replace('-', 'e-')))

        for col in heatcols:
            particledata[col] = np.interp(arr_time_s, arr_time_s_source, heatrates_in[col])

    with at.inputmodel.rprocess_from_trajectory.open_tar_file_or_extracted(
            particleid, './Run_rprocess/energy_thermo.dat') as f:
        dfthermo = pd.read_csv(f, delim_whitespace=True, usecols=['#count', 'time/s', 'Qdot'])

        heatcols = ['Qdot']
        heatrates_in = {col: [] for col in heatcols}
        arr_time_s_source = []
        for _, row in dfthermo.iterrows():
            nstep_timesec[row['#count']] = row['time/s']
            arr_time_s_source.append(row['time/s'])
            for col in heatcols:
                try:
                    heatrates_in[col].append(float(row[col]))
                except ValueError:
                    heatrates_in[col].append(float(row[col].replace('-', 'e-')))

        for col in heatcols:
            particledata[col] = np.interp(arr_time_s, arr_time_s_source, heatrates_in[col])

    if arr_strnuc:
        arr_traj_time_s = []
        arr_massfracs = {strnuc: [] for strnuc in arr_strnuc}
        for nts in range(nts_min, nts_max + 1):
            timesec = nstep_timesec[nts]
            arr_traj_time_s.append(timesec)
            # print(nts, timesec / 86400)
            traj_nuc_abund = at.inputmodel.rprocess_from_trajectory.get_trajectory_nuc_abund(particleid, nts=nts)
            for strnuc in arr_strnuc:
                arr_massfracs[strnuc].append(traj_nuc_abund.get(f'X_{strnuc}', 0.))

        for strnuc in arr_strnuc:
            massfracs_interp = np.interp(arr_time_s, arr_traj_time_s, arr_massfracs[strnuc])
            particledata[strnuc] = massfracs_interp

    return particleid, particledata


def do_modelcells(modelpath, mgiplotlist, arr_el_a):
    arr_el, arr_a = zip(*arr_el_a)
    arr_strnuc = [z + str(a) for z, a in arr_el_a]

    # arr_z = [at.get_atomic_number(el) for el in arr_el]

    dfmodel, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata(modelpath)
    npts_model = len(dfmodel)
    MH = 1.67352e-24  # g

    arr_time_artis_days = []
    arr_abund_artis = {}

    try:
        estimators = at.estimators.read_estimators(modelpath, modelgridindex=tuple(mgiplotlist))
        for nts, mgi in estimators.keys():
            if mgi not in mgiplotlist:
                continue

            time_days = float(estimators[(nts, mgi)]['tdays'])
            if mgi == mgiplotlist[0]:
                arr_time_artis_days.append(time_days)

            rho_init_cgs = 10 ** dfmodel.iloc[mgi].logrho
            rho_cgs = rho_init_cgs * (t_model_init_days / time_days) ** 3
            for strnuc, a in zip(arr_strnuc, arr_a):
                abund = estimators[(nts, mgi)]['populations'].get(strnuc, 0.)
                massfrac = abund * a * MH / rho_cgs

                if mgi not in arr_abund_artis:
                    arr_abund_artis[mgi] = {}

                if strnuc not in arr_abund_artis[mgi]:
                    arr_abund_artis[mgi][strnuc] = []

                arr_abund_artis[mgi][strnuc].append(massfrac)

    except FileNotFoundError:
        pass

    arr_time_artis_s = np.array([t * 8.640000E+04 for t in arr_time_artis_days])

    arr_abund_gsi = {}

    arr_time_gsi_s = np.array([t_model_init_days * 86400, *arr_time_artis_s])
    arr_time_gsi_days = arr_time_gsi_s / 86400

    dfpartcontrib = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(modelpath)
    dfpartcontrib.query('cellindex <= @npts_model', inplace=True)

    list_particleids_getabund = dfpartcontrib.query('(cellindex - 1) in @mgiplotlist').particleid.unique()
    fworkerwithabund = partial(get_particledata, arr_time_gsi_s, arr_strnuc)

    print(f'Reading trajectory data for {len(list_particleids_getabund)} particles with abundances')

    if at.num_processes > 1:
        with multiprocessing.Pool(processes=at.num_processes) as pool:
            list_particledata_withabund = pool.map(fworkerwithabund, list_particleids_getabund)
            pool.close()
            pool.join()
    else:
        list_particledata_withabund = [fworkerwithabund(particleid) for particleid in list_particleids_getabund]

    list_particleids_noabund = [
        pid for pid in dfpartcontrib.particleid.unique() if pid not in list_particleids_getabund]
    fworkernoabund = partial(get_particledata, arr_time_gsi_s, [])
    print(f'Reading trajectory data for {len(list_particleids_noabund)} particles for Qdot/thermal data (no abundances)')

    if at.num_processes > 1:
        with multiprocessing.Pool(processes=at.num_processes) as pool:
            list_particledata_noabund = pool.map(fworkernoabund, list_particleids_noabund)
            pool.close()
            pool.join()
    else:
        list_particledata_noabund = [fworkernoabund(particleid) for particleid in list_particleids_noabund]

    allparticledata = {
        particleid: data for particleid, data in (list_particledata_withabund + list_particledata_noabund)}

    plot_qdot(
        modelpath, dfpartcontrib, dfmodel, allparticledata, arr_time_artis_days, arr_time_gsi_days,
        pdfoutpath=Path(modelpath, 'gsinetwork_global-qdot.pdf'))

    for mgi in mgiplotlist:
        plot_abund(
            modelpath, dfpartcontrib, allparticledata, arr_time_artis_days, arr_time_gsi_days, arr_strnuc,
            arr_abund_gsi, arr_abund_artis.get(mgi, []), t_model_init_days, dfmodel.iloc[mgi], mgi=mgi,
            pdfoutpath=Path(modelpath, f'gsinetwork_cell{mgi}-abundance.pdf'))


def addargs(parser):
    parser.add_argument('-modelpath',
                        default='.',
                        help='Path for ARTIS files')

    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')

    parser.add_argument('-modelgridindex', '-cell', '-mgi', default=0,
                        help='Modelgridindex (zero-indexed) to plot or list such as 4,5,6')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Create solar r-process pattern in ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    arr_el_a = [
        ('Ga', 72),
        ('Sr', 89),
        ('Ba', 140),
        # ('Ce', 141),
        ('Nd', 147),
        ('Rn', 222),
        ('Ra', 223),
        ('Ra', 224),
        ('Ra', 225),
        ('Ac', 225),
        ('Th', 234),
        ('Pa', 233),
        ('U', 235),
    ]
    arr_el_a.sort(key=lambda x: (at.elsymbols.index(x[0]), -x[1]))

    modelpath = Path(args.modelpath)
    if hasattr(args.modelgridindex, 'split'):
        mgiplotlist = [int(mgi) for mgi in args.modelgridindex.split(',')]
    else:
        mgiplotlist = [int(args.modelgridindex)]

    do_modelcells(modelpath, mgiplotlist, arr_el_a)


if __name__ == "__main__":
    main()
