#!/usr/bin/env python3

import argparse
import math
# import os.path

# import numpy as np
import pandas as pd
from astropy import units as u
from pathlib import Path

import artistools as at


def addargs(parser):
    parser.add_argument('-inputpath', '-i',
                        default='.',
                        help='Path of input file')
    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')


def main(args=None, argsraw=None, **kwargs) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Convert Lapuente model to ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    datain_structure = pd.read_csv(Path(args.inputpath, 'structureVMaveraged.txt'),
                                   engine='python', delimiter=r"\s\s+", index_col='i')
    datain_abund = pd.read_csv(Path(args.inputpath, 'abundancesVMaveraged.txt'),
                               delim_whitespace=True)

    datain = datain_structure.join(datain_abund, rsuffix='abund')
    if 'Fe52' not in datain.columns:
        datain.eval('Fe52 = 0.', inplace=True)
    if 'Cr48' not in datain.columns:
        datain.eval('Cr48 = 0.', inplace=True)

    dfmodel = pd.DataFrame(
        columns=['inputcellid', 'velocity_outer', 'logrho', 'X_Fegroup', 'X_Ni56', 'X_Co56',
                 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57'])
    dfmodel.index.name = 'cellid'
    dfelabundances = pd.DataFrame(columns=['inputcellid', *['X_' + at.get_elsymbol(x) for x in range(1, 31)]])
    dfelabundances.index.name = 'cellid'

    t_model_init_seconds = datain_structure['rad / cm'].iloc[0] / datain_structure['vel / cm/s'].iloc[0]
    t_model_init_days = t_model_init_seconds / 24 / 60 / 60

    tot_ni56mass = 0.
    tot_mass = 0.
    tot_mass2 = 0.
    for cellidin, shell in datain.iterrows():
        cellid = cellidin + 1
        if shell['rho / g/cm^3'] == 0.:
            continue

        v_outer = float(shell['vel / cm/s']) * 1e-5  # convert cm/s to km/s
        rho = shell['rho / g/cm^3']

        tot_mass += shell['shellmass / g']

        tot_ni56mass += shell['shellmass / g'] * shell.Ni56

        abundances = [0. for _ in range(31)]

        elsymbols = at.get_elsymbolslist()
        for col in datain.columns:
            atomic_number = -1
            if col in elsymbols:
                atomic_number = elsymbols.index(col)
                abundances[atomic_number] = shell[col]
            elif col.rstrip('0123456789') in elsymbols and col.rstrip('0123456789') not in datain.columns:
                # specific nuclide abundance
                atomic_number = elsymbols.index(col.rstrip('0123456789'))
                abundances[atomic_number] += shell[col]

        X_fegroup = sum(abundances[26:])
        tot_mass2 += sum(abundances[1:]) * shell['shellmass / g']

        radioabundances = [X_fegroup, shell.Ni56, shell.Co56, shell.Fe52, shell.Cr48, shell.Ni57, shell.Co57]

        dfmodel.loc[cellid] = [cellid, v_outer, math.log10(rho), *radioabundances]
        dfelabundances.loc[cellid] = [cellid, *abundances[1:31]]

    print(f'M_tot  = {tot_mass /  u.solMass.to("g"):.3f} solMass (from sum of specified shell masses)')
    print(f'M_tot  = {tot_mass2 /  u.solMass.to("g"):.3f} solMass (from sum of element densities)')
    print(f'M_Ni56 = {tot_ni56mass /  u.solMass.to("g"):.3f} solMass')

    at.save_modeldata(dfmodel, t_model_init_days, Path(args.outputpath, 'model.txt'))
    at.inputmodel.save_initialabundances(dfelabundances, Path(args.outputpath, 'abundances.txt'))


if __name__ == "__main__":
    main()
