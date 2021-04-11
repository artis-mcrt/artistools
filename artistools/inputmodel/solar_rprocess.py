#!/usr/bin/env python3

import argparse
import math
# import os.path

import numpy as np
import pandas as pd
from astropy import units as u
from pathlib import Path

import artistools as at


def addargs(parser):
    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')


def main(args=None, argsraw=None, **kwargs) -> None:
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Create solar r-process pattern in ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    dfsolarabund = pd.read_csv(at.config['path_datadir'] / 'solar_r_abundance_pattern.txt',
                               delim_whitespace=True, comment='#')

    dfbetaminus = pd.read_csv(at.config['path_datadir'] / 'betaminusdecays.txt',
                              delim_whitespace=True, comment='#',
                              names=['A', 'Z', 'Q[MeV]', 'Egamma[MeV]', 'Eelec[MeV]',
                                     'Eneutrino[MeV]', 'tau[s]'])
    normfactor = dfsolarabund.numberfrac.sum()  # convert number fractions in solar to fractions of r-process
    dfsolarabund.eval('numberfrac = numberfrac / @normfactor', inplace=True)

    dfsolarabund.eval('massfrac = numberfrac * A', inplace=True)
    massfracnormfactor = dfsolarabund.massfrac.sum()
    dfsolarabund.eval('massfrac = massfrac / @massfracnormfactor', inplace=True)

    print(dfsolarabund)

    def undecayed_z(row):
        dfmasschain = dfbetaminus.query('A == @row.A', inplace=False)
        if not dfmasschain.empty:
            return int(dfmasschain.Z.min())
        else:
            return int(row.Z)

    dfsolarabund_undecayed = dfsolarabund.copy()
    dfsolarabund_undecayed['Z'] = dfsolarabund_undecayed.apply(undecayed_z, axis=1)
    print(dfsolarabund_undecayed)

    # print(dfsolarabund.numberfrac.sum())
    # print(dfsolarabund.massfrac.sum())

    dictelemabund = {'inputcellid': 1}
    for atomic_number in range(1, dfsolarabund.Z.max() + 1):
        dictelemabund[f'X_{at.elsymbols[atomic_number]}'] = (
            dfsolarabund_undecayed.query('Z == @atomic_number', inplace=False).massfrac.sum())

    dfabundances = pd.DataFrame(dictelemabund, index=[0])
    print(dfabundances)
    at.inputmodel.save_initialabundances(dfabundances=dfabundances, abundancefilename=args.outputpath)

    modeldict = {
        'inputcellid': 1,
        'velocity_outer': 6.e4,
        'logrho': -2.,
        'X_Fegroup': 1.,
        'X_Ni56': 0.,
        'X_Co56': 0.,
        'X_Fe52': 0.,
        'X_Cr48': 0.,
        'X_Ni57': 0.,
        'X_Co57': 0.,
    }

    t_model_init_days = 0.000231481

    for _, row in dfsolarabund_undecayed.iterrows():
        modeldict[f'X_{at.elsymbols[int(row.Z)]}{int(row.A)}'] = row.massfrac

    dfmodel = pd.DataFrame(modeldict, index=[0])
    dfmodel.index.name = 'cellid'
    print(dfmodel)

    at.inputmodel.save_modeldata(dfmodel, t_model_init_days, Path(args.outputpath, 'model.txt'))


if __name__ == "__main__":
    main()
