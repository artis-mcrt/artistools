#!/usr/bin/env python3

# import numpy as np
# import io
import pandas as pd
# import math
from pathlib import Path
import matplotlib.pyplot as plt

import tarfile

import artistools as at
# import artistools.estimators


def plot_qdot(modelpath, dfthermo, dfheating):
    depdata = at.get_deposition(modelpath=modelpath)

    df = pd.merge(dfthermo, dfheating, how='inner', on=['#count', 'time/s', 'rho(g/cc)'])
    pd.options.display.max_rows = 50
    pd.options.display.width = 250
    pd.set_option("display.width", 250)
    tstart = depdata['tmid_days'].min()
    tend = depdata['tmid_days'].max()
    df.eval("time_d = `time/s`  / 86400", inplace=True)
    df.query("`time_d` >= @tstart - 0.3 and `time_d` <= @tend + 1.5", inplace=True)

    df['hspof'] = df['hspof'].astype(float)
    df['hbfis'] = df['hbfis'].astype(float)

    print(depdata.columns)
    print(df.columns)
    print(df)

    arr_time_s, arr_qdot = dfthermo['time/s'], dfthermo['Qdot']

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
    arr_time_days, arr_qdot = zip(
        *[(t, qdot) for t, qdot in zip(arr_time_days, arr_qdot)
          if depdata['tmid_days'].min() <= t and t <= depdata['tmid_days'].max()])

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

    axis.plot(df['time_d'], df['hbeta'],
              linewidth=2, color='black',
              linestyle='dashed',
              # marker='x', markersize=8,
              label=r'$\dot{Q}_\beta$ GSI Network')

    axis.plot(depdata['tmid_days'], depdata['Qdot_betaminus_ana_erg/g/s'],
              linewidth=2, color='red',
              linestyle='dashed',
              # marker='+', markersize=15,
              label=r'$\dot{Q}_\beta$ ARTIS')

    axis.plot(df['time_d'], df['halpha'],
              linewidth=2, color='black',
              linestyle='dotted',
              # marker='x', markersize=8,
              label=r'$\dot{Q}_\alpha$ GSI Network')

    axis.plot(depdata['tmid_days'], depdata['Qdotalpha_ana_erg/g/s'],
              linewidth=2, color='red',
              linestyle='dotted',
              # marker='+', markersize=15,
              label=r'$\dot{Q}_\alpha$ ARTIS')

    axis.plot(df['time_d'], df['hbfis'],
              linewidth=2,
              linestyle='dotted',
              # marker='x', markersize=8,
              # color='black',
              label=r'$\dot{Q}_{hbfis}$ GSI Network')

    axis.plot(df['time_d'], df['hspof'],
              linewidth=2,
              linestyle='dotted',
              # marker='x', markersize=8,
              # color='black',
              label=r'$\dot{Q}_{spof}$ GSI Network')

    axis.legend(loc='best', frameon=False, handlelength=1, ncol=3,
                numpoints=1)

    plt.savefig(Path(modelpath, 'gsinetwork-qdot.pdf'), format='pdf')


def main():
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

    rundir = Path('/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/artis_runs')
    # modelpath = Path(rundir, 'kilonova_rprocesstraj_testalphameantomweight')
    modelpath = Path(rundir, 'kilonova_rprocesstraj_testalphameanatomweight_redoqdot')

    dfmodel, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata(modelpath)
    rho_init_cgs = 10 ** dfmodel.iloc[0].logrho
    MH = 1.67352e-24  # g

    arr_time_artis = []
    arr_abund_artis = {}

    estimators = at.estimators.read_estimators(modelpath)
    for ts_mgi in estimators.keys():
        # print(estimators[ts_mgi])
        time_days = float(estimators[ts_mgi]['tdays'])
        arr_time_artis.append(time_days)

        rho_cgs = rho_init_cgs * (t_model_init_days / time_days) ** 3
        for strnuc, a in zip(arr_strnuc, arr_a):
            abund = estimators[ts_mgi]['populations'][strnuc]
            massfrac = abund * a * MH / rho_cgs
            if strnuc not in arr_abund_artis:
                arr_abund_artis[strnuc] = []
            arr_abund_artis[strnuc].append(massfrac)

    arr_time_gsi = []
    arr_abund_gsi = {}

    print('Reading trajectory data')
    trajtar = Path('/Volumes/GoogleDrive/My Drive/Archive/Astronomy Downloads/Mergers/SFHo/88969.tar.xz')
    tar = tarfile.open(trajtar, mode='r:*')

    # files in the compressed tarfile should be read in the natural order because seeking is extremely slow
    nzplanedata = {}
    for member in tar.getmembers():
        if member.isfile:
            if member.name.startswith('./Run_rprocess/nz-plane'):
                with tar.extractfile(member) as f:
                    header_str = f.readline().split()
                    df = pd.read_csv(f, delim_whitespace=True, names=['N', 'Z', 'log10abund', 'S1n', 'S2n'])
                    nzplanedata[member.name] = header_str, df
            elif member.name == './Run_rprocess/energy_thermo.dat':
                with tar.extractfile(member) as f:
                    dfthermo = pd.read_csv(f, delim_whitespace=True)
            elif member.name == './Run_rprocess/heating.dat':
                with tar.extractfile(member) as f:
                    dfheating = pd.read_csv(f, delim_whitespace=True)

    plot_qdot(modelpath, dfthermo, dfheating)

    for member in sorted(nzplanedata.keys()):
        header_str, df = nzplanedata[member]
        header = int(header_str[0]), *[float(x) for x in header_str[1:]]
        filecount, time, T9, rho, neutron_number_density, trajectory_mass = header
        # print(header)
        time_days = time / 8.640000E+04
        if not (arr_time_artis[0] <= time_days <= arr_time_artis[-1]):
            continue
        arr_time_gsi.append(time_days)

        df.eval('abund = 10 ** log10abund', inplace=True)
        df.eval('A = N + Z', inplace=True)
        df.eval('massfrac = A * abund', inplace=True)

        for strnuc, a, z in zip(arr_strnuc, arr_a, arr_z):
            if strnuc not in arr_abund_gsi:
                arr_abund_gsi[strnuc] = []

            massfrac = df.query('A == @a and Z == @z')['massfrac'].sum()
            arr_abund_gsi[strnuc].append(massfrac)

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

        axis.plot(arr_time_gsi, arr_abund_gsi[strnuc],
                  # linestyle='None',
                  linewidth=2,
                  marker='x', markersize=8,
                  label=f'{strnuc} GSI Network', color='black')

        axis.plot(arr_time_artis, arr_abund_artis[strnuc],
                  linewidth=2,
                  # linestyle='None',
                  # marker='+', markersize=15,
                  label=f'{strnuc} ARTIS', color='red')

        axis.legend(loc='best', frameon=False, handlelength=1, ncol=1,
                    numpoints=1)

    plt.savefig(Path(modelpath, 'gsinetwork-abundance.pdf'), format='pdf')


if __name__ == "__main__":
    main()
