#!/usr/bin/env python3

import argparse
import math
import os.path

# import numpy as np
import pandas as pd
from astropy import units as u

import artistools as at


def addargs(parser):
    parser.add_argument('-inputpath', '-i',
                        default='1.00_5050.dat',
                        help='Path of input file')
    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')


def main(args=None, argsraw=None, **kwargs) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Convert Shen et al. 2018 models to ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    with open(args.inputpath) as infile:
        columns = infile.readline().split()

    atomicnumberofspecies = {}
    isotopesofelem = {}
    for species in columns[5:]:
        atomic_number = at.get_atomic_number(species.rstrip('0123456789'))
        atomicnumberofspecies[species] = atomic_number
        isotopesofelem.setdefault(atomic_number, list()).append(species)

    datain = pd.read_csv(args.inputpath, delim_whitespace=True, skiprows=0, header=[0]).dropna()

    dfmodel = pd.DataFrame(
        columns=[
            'inputcellid', 'velocity_outer', 'logrho', 'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48',
            'X_Ni57', 'X_Co57'])
    dfmodel.index.name = 'cellid'
    dfelabundances = pd.DataFrame(columns=['inputcellid', *['X_' + at.get_elsymbol(x) for x in range(1, 31)]])
    dfelabundances.index.name = 'cellid'

    t_model_init_seconds = 10.
    t_model_init_days = t_model_init_seconds / 24 / 60 / 60

    v_inner = 0.  # velocity at inner boundary of cell
    m_enc_inner = 0.  # mass enclosed at inner boundary
    tot_ni56mass = 0.
    for cellid, shell in datain.iterrows():
        m_enc_outer = float(shell['m']) * u.solMass.to('g')  # convert Solar masses to grams
        v_outer = float(shell['v']) * 1e-5  # convert cm/s to km/s

        m_shell_grams = (m_enc_outer - m_enc_inner)
        r_outer = v_outer * 1e5 * t_model_init_seconds
        r_inner = v_inner * 1e5 * t_model_init_seconds
        vol_shell = 4. / 3. * math.pi * (r_outer ** 3 - r_inner ** 3)
        rho = m_shell_grams / vol_shell

        tot_ni56mass += m_shell_grams * shell.ni56

        abundances = [0. for _ in range(31)]

        X_Fegroup = 0.
        for atomic_number in range(1, 31):
            abundances[atomic_number] = sum([float(shell[species]) for species in isotopesofelem[atomic_number]])
            if atomic_number >= 26:
                X_Fegroup += abundances[atomic_number]

        radioabundances = [X_Fegroup, shell.ni56, shell.co56, shell.fe52, shell.cr48, shell.ni57, shell.co57]

        dfmodel.loc[cellid] = [cellid, v_outer, math.log10(rho), *radioabundances]
        dfelabundances.loc[cellid] = [cellid, *abundances[1:31]]

        v_inner = v_outer
        m_enc_inner = m_enc_outer

    print(f'M_tot  = {m_enc_outer /  u.solMass.to("g"):.3f} solMass')
    print(f'M_Ni56 = {tot_ni56mass /  u.solMass.to("g"):.3f} solMass')

    at.save_modeldata(dfmodel, t_model_init_days, os.path.join(args.outputpath, 'model.txt'))
    at.inputmodel.save_initialabundances(dfelabundances, os.path.join(args.outputpath, 'abundances.txt'))


if __name__ == "__main__":
    main()
