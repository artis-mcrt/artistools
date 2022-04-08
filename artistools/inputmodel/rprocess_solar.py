#!/usr/bin/env python3

import argparse
import math
# import os.path

import numpy as np
import pandas as pd
from pathlib import Path

import artistools as at


def addargs(parser):
    parser.add_argument('-outputpath', '-o',
                        default='.',
                        help='Path for output files')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Create solar r-process pattern in ARTIS format.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        args = parser.parse_args(argsraw)

    dfsolarabund = pd.read_csv(at.config['path_datadir'] / 'solar_r_abundance_pattern.txt',
                               delim_whitespace=True, comment='#')

    dfsolarabund['radioactive'] = True

    # print(dfsolarabund)

    dfbetaminus = pd.read_csv(at.config['path_datadir'] / 'betaminusdecays.txt',
                              delim_whitespace=True, comment='#',
                              names=['A', 'Z', 'Q[MeV]', 'Egamma[MeV]', 'Eelec[MeV]',
                                     'Eneutrino[MeV]', 'tau[s]'])

    def undecayed_z(row):
        dfmasschain = dfbetaminus.query('A == @row.A', inplace=False)
        if not dfmasschain.empty:
            return int(dfmasschain.Z.min())  # decay to top of chain
        else:
            return int(row.Z)

    dfsolarabund_undecayed = dfsolarabund.copy()
    dfsolarabund_undecayed['Z'] = dfsolarabund_undecayed.apply(undecayed_z, axis=1)

    # Andreas uses 90% Fe and the rest solar
    dfsolarabund_undecayed = dfsolarabund_undecayed.append(
        {'Z': 26, 'A': 56, 'numberfrac': 0.005, 'radioactive': False}, ignore_index=True)
    dfsolarabund_undecayed = dfsolarabund_undecayed.append(
        {'Z': 27, 'A': 59, 'numberfrac': 0.005, 'radioactive': False}, ignore_index=True)
    dfsolarabund_undecayed = dfsolarabund_undecayed.append(
        {'Z': 28, 'A': 58, 'numberfrac': 0.005, 'radioactive': False}, ignore_index=True)

    normfactor = dfsolarabund_undecayed.numberfrac.sum()  # convert number fractions in solar to fractions of r-process
    dfsolarabund_undecayed.eval('numberfrac = numberfrac / @normfactor', inplace=True)

    dfsolarabund_undecayed.eval('massfrac = numberfrac * A', inplace=True)
    massfracnormfactor = dfsolarabund_undecayed.massfrac.sum()
    dfsolarabund_undecayed.eval('massfrac = massfrac / @massfracnormfactor', inplace=True)

    # print(dfsolarabund_undecayed)

    t_model_init_days = 0.000231481
    t_model_init_seconds = t_model_init_days * 24 * 60 * 60

    wollager_profilename = 'wollager_ejectaprofile_10bins.txt'
    if Path(wollager_profilename).exists():
        t_model_init_days_in = float(Path(wollager_profilename).open('rt').readline().strip().removesuffix(' day'))
        dfdensities = pd.read_csv(wollager_profilename, delim_whitespace=True, skiprows=1,
                                  names=['cellid', 'velocity_outer', 'rho'])
        dfdensities['cellid'] = dfdensities['cellid'].astype(int)
        dfdensities['velocity_inner'] = np.concatenate(([0.], dfdensities['velocity_outer'].values[:-1]))

        t_model_init_seconds_in = t_model_init_days_in * 24 * 60 * 60
        dfdensities.eval('cellmass_grams = rho * 4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                         '* (1e5 * @t_model_init_seconds_in) ** 3', inplace=True)

        # now replace the density at the input time with the density at required time

        dfdensities.eval('rho = cellmass_grams / ('
                         '4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                         ' * (1e5 * @t_model_init_seconds) ** 3)', inplace=True)
    else:
        dfdensities = pd.DataFrame(dict(rho=10 ** -3, velocity_outer=6.e4), index=[0])

    # print(dfdensities)
    cellcount = len(dfdensities)
    # write abundances.txt

    dictelemabund = {}
    for atomic_number in range(1, dfsolarabund_undecayed.Z.max() + 1):
        dictelemabund[f'X_{at.get_elsymbol(atomic_number)}'] = (
            dfsolarabund_undecayed.query('Z == @atomic_number', inplace=False).massfrac.sum())

    dfelabundances = pd.DataFrame([dict(inputcellid=mgi + 1, **dictelemabund) for mgi in range(cellcount)])
    # print(dfelabundances)
    at.inputmodel.save_initialabundances(dfelabundances=dfelabundances, abundancefilename=args.outputpath)

    # write model.txt

    rowdict = {
        # 'inputcellid': 1,
        # 'velocity_outer': 6.e4,
        # 'logrho': -3.,
        'X_Fegroup': 1.,
        'X_Ni56': 0.,
        'X_Co56': 0.,
        'X_Fe52': 0.,
        'X_Cr48': 0.,
        'X_Ni57': 0.,
        'X_Co57': 0.,
    }

    for _, row in dfsolarabund_undecayed.query('radioactive == True').iterrows():
        rowdict[f'X_{at.get_elsymbol(int(row.Z))}{int(row.A)}'] = row.massfrac

    modeldata = []
    for mgi, densityrow in dfdensities.iterrows():
        # print(mgi, densityrow)
        modeldata.append(dict(inputcellid=mgi + 1, velocity_outer=densityrow['velocity_outer'],
                         logrho=math.log10(densityrow['rho']), **rowdict))
    # print(modeldata)

    dfmodel = pd.DataFrame(modeldata)
    # print(dfmodel)
    at.inputmodel.save_modeldata(dfmodel=dfmodel, t_model_init_days=t_model_init_days, modelpath=Path(args.outputpath))


if __name__ == "__main__":
    main()
