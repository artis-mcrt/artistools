#!/usr/bin/env python3

# import numpy as np
# import io
import pandas as pd
# import math
from pathlib import Path
import matplotlib.pyplot as plt

import tarfile

import artistools as at
import artistools.estimators


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

    modelpath = Path(
        '/Users/luke/Library/Mobile Documents/com~apple~CloudDocs/artis_runs',
        # 'kilonova_rprocesstraj_testalpha')
        'kilonova_rprocesstraj_testalphameantomweight')

    dfmodeldata, t_model_init_days, vmax_cmps = at.inputmodel.get_modeldata(modelpath)
    rho_init_cgs = 10 ** dfmodeldata.iloc[0].logrho
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
        if member.isfile and member.name.startswith('./Run_rprocess/nz-plane'):
            with tar.extractfile(member) as f:
                header_str = f.readline().split()
                df = pd.read_csv(f, delim_whitespace=True, names=['N', 'Z', 'log10abund', 'S1n', 'S2n'])
                nzplanedata[member.name] = header_str, df

    for member in sorted(nzplanedata.keys()):
        header_str, df = nzplanedata[member]
        header = int(header_str[0]), *[float(x) for x in header_str[1:]]
        filecount, time, T9, rho, neutron_number_density, trajectory_mass = header
        # print(header)
        time_days = time / 8.640000E+04
        if not (1. <= time_days <= 10.):
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

    axis.legend()
    plt.savefig('gsinetwork.pdf', format='pdf')


if __name__ == "__main__":
    main()
