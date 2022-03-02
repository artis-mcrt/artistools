#!/usr/bin/env python3

import argparse
import math
import tarfile
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

    trajtar = Path(
        '/Volumes/GoogleDrive/My Drive/Archive/Astronomy Downloads/Mergers/SFHo/88969.tar.xz')

    tar = tarfile.open(trajtar, mode='r:*')
    trajfile = tar.extractfile(member='./Run_rprocess/tday_nz-plane')

    # energy_thermo_data = pd.read_csv(energythermo_file, delim_whitespace=True)
    # with open(trajfile) as ftraj:
    _, str_t_model_init_seconds, _, rho, _, _ = trajfile.readline().decode().split()
    t_model_init_seconds = float(str_t_model_init_seconds)
    print('time', t_model_init_seconds)

    dfnucabund = pd.read_csv(trajfile, delim_whitespace=True, comment='#', names=["N", "Z", "log10abund", "S1n", "S2n"])
    dfnucabund.eval('abund = 10 ** log10abund', inplace=True)
    dfnucabund.eval('A = N + Z', inplace=True)
    dfnucabund.eval('massfrac = A * abund', inplace=True)
    dfnucabund.query('Z >= 1', inplace=True)
    dfnucabund.query('abund > 0.', inplace=True)
    print(dfnucabund)

    dfnucabund['radioactive'] = True

    normfactor = dfnucabund.abund.sum()  # convert number fractions in solar to fractions of r-process
    print(f'abund sum: {normfactor}')
    dfnucabund.eval('numberfrac = abund / @normfactor', inplace=True)

    massfracnormfactor = dfnucabund.massfrac.sum()
    print(f'massfrac sum: {massfracnormfactor}')
    dfnucabund.eval('massfrac = massfrac / @massfracnormfactor', inplace=True)

    # print(dfsolarabund_undecayed)

    t_model_init_days = t_model_init_seconds / (24 * 60 * 60)

    wollager_profilename = 'wollager_ejectaprofile_10bins.txt'
    if Path(wollager_profilename).exists():
        t_model_init_days_in = float(Path(wollager_profilename).open('rt').readline().strip().removesuffix(' day'))
        dfdensities = pd.read_csv(wollager_profilename, delim_whitespace=True, skiprows=1,
                                  names=['cellid', 'velocity_outer', 'rho'])
        dfdensities['cellid'] = dfdensities['cellid'].astype(int)
        dfdensities['velocity_inner'] = np.concatenate(([0.], dfdensities['velocity_outer'].values[:-1]))

        t_model_init_seconds_in = t_model_init_days_in * 24 * 60 * 60
        dfdensities.eval('shellmass_grams = rho * 4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                         '* (1e5 * @t_model_init_seconds_in) ** 3', inplace=True)

        # now replace the density at the input time with the density at required time

        dfdensities.eval('rho = shellmass_grams / ('
                         '4. / 3. * @math.pi * (velocity_outer ** 3 - velocity_inner ** 3)'
                         ' * (1e5 * @t_model_init_seconds) ** 3)', inplace=True)
    else:
        dfdensities = pd.DataFrame(dict(rho=10 ** -11, velocity_outer=6.e4), index=[0])

    # print(dfdensities)
    cellcount = len(dfdensities)
    # write abundances.txt

    dictelemabund = {}
    for atomic_number in range(1, dfnucabund.Z.max() + 1):
        dictelemabund[f'X_{at.elsymbols[atomic_number]}'] = (
            dfnucabund.query('Z == @atomic_number', inplace=False).massfrac.sum())

    dfabundances = pd.DataFrame([dict(inputcellid=mgi + 1, **dictelemabund) for mgi in range(cellcount)])
    # print(dfabundances)
    at.inputmodel.save_initialabundances(dfabundances=dfabundances, abundancefilename=args.outputpath)

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

    for _, row in dfnucabund.query('radioactive == True').iterrows():
        rowdict[f'X_{at.elsymbols[int(row.Z)]}{int(row.A)}'] = row.massfrac

    modeldata = []
    for mgi, densityrow in dfdensities.iterrows():
        # print(mgi, densityrow)
        modeldata.append(dict(inputcellid=mgi + 1, velocity_outer=densityrow['velocity_outer'],
                         logrho=math.log10(densityrow['rho']), **rowdict))
    # print(modeldata)

    dfmodel = pd.DataFrame(modeldata)
    # print(dfmodel)
    at.inputmodel.save_modeldata(dfmodel, t_model_init_days, Path(args.outputpath, 'model.txt'))


if __name__ == "__main__":
    main()
