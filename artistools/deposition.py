#!/usr/bin/env python3
import argparse
import math
# import numpy as np
from astropy import units as u
# from astropy import constants as c
import artistools as at
import artistools.nltepops


def addargs(parser):
    parser.add_argument('-modelpath', default='.',
                        help='Path to ARTIS folder')

    parser.add_argument('-timedays', '-t', default=330, type=float,
                        help='Time in days')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Plot deposition rate of a model at time t (days).')
        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)
    dfmodel, t_model_init = at.get_modeldata(args.modelpath)

    t_init = t_model_init * u.day

    meanlife_co56 = 113.7 * u.day
    # define TCOBALT (113.7*DAY)     // cobalt-56
    # T48CR = 1.29602 * u.day
    # T48V = 23.0442 * u.day
    # define T52FE   (0.497429*DAY)
    # define T52MN   (0.0211395*DAY)

    t_now = args.timedays * u.day
    print(f't_now = {t_now.to("d")}')
    print('The following assumes that all 56Ni has decayed to 56Co and all energy comes from emitted positrons')

    adata = at.get_levels(args.modelpath, get_photoionisations=True)
    timestep = at.get_closest_timestep(args.modelpath, args.timedays)
    dfnltepops = at.nltepops.read_files(
        args.modelpath, adata, 26, 1000, 1000, timestep, -1, noprint=True)

    phixs = adata.query('Z==26 & ion_stage==1', inplace=False).iloc[0].levels.iloc[0].phixstable[0][1] * 1e-18

    v_inner = 0 * u.km / u.s
    for i, row in dfmodel.iterrows():
        v_outer = row['velocity'] * u.km / u.s

        volume_init = ((4 * math.pi / 3) * ((v_outer * t_init) ** 3 - (v_inner * t_init) ** 3)).to('cm3')

        volume_now = ((4 * math.pi / 3) * ((v_outer * t_now) ** 3 - (v_inner * t_now) ** 3)).to('cm3')

        # volume_now2 = (volume_init * (t_now / t_init) ** 3).to('cm3')

        rho_init = (10 ** row['logrho']) * u.g / u.cm ** 3
        mco56_init = (row['X_Ni56'] + row['X_Co56']) * (volume_init * rho_init).to('solMass')
        mco56_now = mco56_init * math.exp(- t_now / meanlife_co56)

        co56_positron_dep = (0.19 * 0.610 * u.MeV * (mco56_now / (55.9398393 * u.u)) / meanlife_co56).to('erg/s')
        v48_positron_dep = 0
        # v48_positron_dep = (0.290 * 0.499 * u.MeV) * (math.exp(-t / T48V) - exp(-t / T48CR)) / (T48V - T48CR) * mcr48_now / MCR48;

        power_now = co56_positron_dep + v48_positron_dep

        epsilon = power_now / volume_now
        print(f'zone {i:3d}, velocity = [{v_inner:8.2f}, {v_outer:8.2f}], epsilon = {epsilon:.3e}')
        # print(f'  epsilon = {epsilon.to("eV/(cm3s)"):.2f}')

        dfnltepops_cell = dfnltepops.query('modelgridindex == @i', inplace=False)
        if not dfnltepops_cell.empty:
            nnlevel = dfnltepops_cell.query('level == 0', inplace=False).iloc[0]['n_NLTE']
            width = ((v_outer - v_inner) * t_now).to('cm').value
            tau = width * phixs * nnlevel
            print(f'width: {width:.3e} cm, phixs: {phixs:.3e} cm^2, nnlevel: {nnlevel:.3e} cm^-3, tau: {tau:.3e}')

        v_inner = v_outer


if __name__ == "__main__":
    main(args)
