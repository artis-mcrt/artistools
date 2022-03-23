#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import argcomplete
import argparse

import numpy as np
# import io
import pandas as pd
# import math
from pathlib import Path
import matplotlib.pyplot as plt

import tarfile

import artistools as at
# import artistools.estimators


def plot_qdot(modelpath, particledata, arr_time_s, pdfoutpath):
    depdata = at.get_deposition(modelpath=modelpath)

    tstart = depdata['tmid_days'].min()
    tend = depdata['tmid_days'].max()

    arr_heat = {}
    frac_of_cellmass_sum = sum([p['frac_of_cellmass'] for _, p in particledata.items()])
    heatcols = ['hbeta', 'halpha', 'hbfis', 'hspof']
    # heatcols = ['hbeta', 'halpha']
    for col in heatcols:
        arr_heat[col] = np.zeros_like(arr_time_s)

    for particleid, thisparticledata in particledata.items():
        for col in heatcols:
            arr_time_s_source = sorted(list(thisparticledata[col].keys()))
            arr_heat_source = np.array([thisparticledata[col][t] for t in arr_time_s_source])
            heatrates_interp = np.interp(arr_time_s, arr_time_s_source, arr_heat_source)
            arr_heat[col] += heatrates_interp * thisparticledata['frac_of_cellmass'] / frac_of_cellmass_sum

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
    arr_time_days = arr_time_s / 86400
    # arr_time_days, arr_qdot = zip(
    #     *[(t, qdot) for t, qdot in zip(arr_time_days, arr_qdot)
    #       if depdata['tmid_days'].min() <= t and t <= depdata['tmid_days'].max()])

    # axis.plot(df['time_d'], df['Qdot'],
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

    axis.plot(arr_time_days, arr_heat['hbeta'],
              linewidth=2, color='black',
              linestyle='dashed',
              # marker='x', markersize=8,
              label=r'$\dot{Q}_\beta$ GSI Network')

    axis.plot(depdata['tmid_days'], depdata['Qdot_betaminus_ana_erg/g/s'],
              linewidth=2, color='red',
              linestyle='dashed',
              # marker='+', markersize=15,
              label=r'$\dot{Q}_\beta$ ARTIS')

    axis.plot(arr_time_days, arr_heat['halpha'],
              linewidth=2, color='black',
              linestyle='dotted',
              # marker='x', markersize=8,
              label=r'$\dot{Q}_\alpha$ GSI Network')

    axis.plot(depdata['tmid_days'], depdata['Qdotalpha_ana_erg/g/s'],
              linewidth=2, color='red',
              linestyle='dotted',
              # marker='+', markersize=15,
              label=r'$\dot{Q}_\alpha$ ARTIS')

    axis.plot(arr_time_days, arr_heat['hbfis'],
              linewidth=2,
              linestyle='dotted',
              # marker='x', markersize=8,
              # color='black',
              label=r'$\dot{Q}_{hbfis}$ GSI Network')

    axis.plot(arr_time_days, arr_heat['hspof'],
              linewidth=2,
              linestyle='dotted',
              # marker='x', markersize=8,
              # color='black',
              label=r'$\dot{Q}_{spof}$ GSI Network')

    axis.legend(loc='best', frameon=False, handlelength=1, ncol=3,
                numpoints=1)

    plt.savefig(pdfoutpath, format='pdf')
    print(f'Saved {pdfoutpath}')


def plot_abund(arr_time_artis, arr_strnuc, arr_abund_gsi, arr_abund_artis, pdfoutpath):
    fig, axes = plt.subplots(
        nrows=len(arr_strnuc), ncols=1, sharex=False, sharey=False, figsize=(6, 1 + 2. * len(arr_strnuc)),
        tight_layout={"pad": 0.4, "w_pad": 0.0, "h_pad": 0.0})

    # axis.set_xscale('log')

    for axis, strnuc in zip(axes, arr_strnuc):
        axis.set_xlim(left=1., right=arr_time_artis[-1])
        axis.set_xlabel('Time [days]')
        # axis.set_yscale('log')
        # axis.set_ylabel(f'X({strnuc})')
        axis.set_ylabel('Mass fraction')

        axis.plot(arr_time_artis, arr_abund_gsi[strnuc],
                  # linestyle='None',
                  linewidth=2,
                  marker='x', markersize=8,
                  label=f'{strnuc} GSI Network', color='black')

        axis.plot(arr_time_artis, arr_abund_artis[strnuc],
                  linewidth=2,
                  # linestyle='None',
                  # marker='+', markersize=15,
                  label=f'{strnuc} ARTIS', color='red')

        axis.legend(loc='best', frameon=False, handlelength=1, ncol=1, numpoints=1)

    plt.savefig(pdfoutpath, format='pdf')
    print(f'Saved {pdfoutpath}')


def addargs(parser):
    parser.add_argument('-modelpath',
                        default='.',
                        help='Path for ARTIS files')

    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')

    parser.add_argument('-modelgridindex', '-cell', '-mgi', type=int, default=0,
                        help='Modelgridindex to plot')


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
        ('Ce', 141),
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
    arr_el_a.sort(key=lambda x: (-at.elsymbols.index(x[0]), -x[1]))
    arr_el, arr_a = zip(*arr_el_a)
    arr_strnuc = [z + str(a) for z, a in arr_el_a]

    arr_z = [at.elsymbols.index(el) for el in arr_el]

    modelpath = Path(args.modelpath)
    mgi = int(args.modelgridindex)

    dfmodel, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata(modelpath)
    rho_init_cgs = 10 ** dfmodel.iloc[0].logrho
    MH = 1.67352e-24  # g

    arr_time_artis = []
    arr_abund_artis = {}

    estimators = at.estimators.read_estimators(modelpath, modelgridindex=mgi)
    for key_ts, key_mgi in estimators.keys():
        if key_mgi != mgi:
            continue

        time_days = float(estimators[(key_ts, key_mgi)]['tdays'])
        arr_time_artis.append(time_days)

        rho_cgs = rho_init_cgs * (t_model_init_days / time_days) ** 3
        for strnuc, a in zip(arr_strnuc, arr_a):
            abund = estimators[(key_ts, key_mgi)]['populations'][strnuc]
            massfrac = abund * a * MH / rho_cgs
            if strnuc not in arr_abund_artis:
                arr_abund_artis[strnuc] = []
            arr_abund_artis[strnuc].append(massfrac)

    arr_time_artis_s = np.array([t * 8.640000E+04 for t in arr_time_artis])
    arr_abund_gsi = {}

    dfpartcontrib = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(modelpath)
    dfpartcontrib.query('cellindex == @mgi + 1', inplace=True)
    print(f'Reading trajectory data for {len(dfpartcontrib)} particles (mgi {mgi})')

    particledata = {}
    for particleid, frac_of_cellmass in dfpartcontrib[['particleid', 'frac_of_cellmass']].itertuples(index=False):
        trajtar = at.inputmodel.rprocess_from_trajectory.get_traj_tarpath(particleid)
        try:
            with tarfile.open(trajtar, mode='r:*') as tar:
                print(f'Reading data for particle id {particleid}...')
                particledata[particleid] = {
                    'frac_of_cellmass': frac_of_cellmass,
                    'Qdot': {},
                    'hbeta': {},
                    'halpha': {},
                    'hbfis': {},
                    'hspof': {},
                    **{strnuc: {} for strnuc in arr_strnuc}
                }

                # files in the compressed tarfile should be read in the natural order because seeking is extremely slow
                for member in tar.getmembers():
                    if member.isfile:
                        if member.name.startswith('./Run_rprocess/nz-plane'):
                            with tar.extractfile(member) as f:
                                header_str = f.readline().split()
                                header = int(header_str[0]), *[float(x) for x in header_str[1:]]
                                filecount, time, T9, rho, neutron_number_density, trajectory_mass = header
                                # if time < 0.5 * arr_time_artis_s.min() or time > arr_time_artis_s * 2.:
                                #     continue

                                dfabund = pd.read_csv(
                                    f, delim_whitespace=True,
                                    names=['N', 'Z', 'log10abund', 'S1n', 'S2n'], usecols=['N', 'Z', 'log10abund'])

                                dfabund.eval('massfrac = (N + Z) * (10 ** log10abund)', inplace=True)

                                for strnuc, a, z in zip(arr_strnuc, arr_a, arr_z):
                                    particledata[particleid][strnuc][time] = (
                                        dfabund.query('(N + Z) == @a and Z == @z')['massfrac'].sum())

                        # elif member.name == './Run_rprocess/energy_thermo.dat':
                        #     with tar.extractfile(member) as f:
                        #         dfthermo = pd.read_csv(f, delim_whitespace=True, usecols=['#count', 'time/s', 'Qdot'])
                        #         print(dfthermo)
                        elif member.name == './Run_rprocess/heating.dat':
                            with tar.extractfile(member) as f:
                                dfheating = pd.read_csv(
                                    f, delim_whitespace=True, usecols=[
                                        '#count', 'time/s', 'hbeta', 'halpha', 'hbfis', 'hspof'])
                                heatcols = ['hbeta', 'halpha', 'hbfis', 'hspof']
                                for _, row in dfheating.iterrows():
                                    time_s = row['time/s']
                                    for col in heatcols:
                                        try:
                                            particledata[particleid][col][time_s] = float(row[col])
                                        except ValueError:
                                            particledata[particleid][col][time_s] = float(row[col].replace('-', 'e-'))

        except FileNotFoundError:
            print(f'WARNING: Particle data not found for id {particleid}')
            continue

    frac_of_cellmass_sum = sum([p['frac_of_cellmass'] for _, p in particledata.items()])

    for strnuc in arr_strnuc:
        arr_abund_gsi[strnuc] = np.zeros_like(arr_time_artis_s)

    for particleid, thisparticledata in particledata.items():
        for strnuc, a, z in zip(arr_strnuc, arr_a, arr_z):
            arr_time_s = sorted(list(thisparticledata[strnuc].keys()))
            # arr_time_s_float = [float(t) for t in arr_time_s]
            arr_massfracs = np.array([thisparticledata[strnuc][t] for t in arr_time_s])
            massfracs_interp = np.interp(arr_time_artis_s, arr_time_s, arr_massfracs)
            arr_abund_gsi[strnuc] += massfracs_interp * thisparticledata['frac_of_cellmass'] / frac_of_cellmass_sum

    plot_qdot(
        modelpath, particledata, arr_time_artis_s,
        pdfoutpath=Path(modelpath, f'gsinetwork_cell{mgi}-qdot.pdf'))

    plot_abund(
        arr_time_artis, arr_strnuc, arr_abund_gsi, arr_abund_artis,
        pdfoutpath=Path(modelpath, f'gsinetwork_cell{mgi}-abundance.pdf'))


if __name__ == "__main__":
    main()
