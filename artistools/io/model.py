import os.path
from functools import lru_cache
import pandas as pd
from collections import namedtuple
import math
from pathlib import Path
import artistools


@lru_cache(maxsize=8)
def get_modeldata(filename=Path()):
    """
    Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides.

    Returns (dfmodeldata, t_model_init_days)
        - dfmodeldata: a pandas DataFrame with a row for each model grid cell
        - t_model_init_days: the time in days at which the snapshot is defined
    """

    if os.path.isdir(filename):
        filename = artistools.firstexisting(['model.txt.xz', 'model.txt.gz', 'model.txt'], path=filename)

    dfmodeldata = pd.DataFrame()

    gridcelltuple = None
    velocity_inner = 0.
    with open(filename, 'r') as fmodel:
        gridcellcount = int(fmodel.readline())
        t_model_init_days = float(fmodel.readline())
        for line in fmodel:
            row = line.split()

            if gridcelltuple is None:
                gridcelltuple = namedtuple('gridcell', [
                    'inputcellid', 'velocity_inner', 'velocity_outer', 'logrho',
                    'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57'][:len(row) + 1])

            celltuple = gridcelltuple(int(row[0]), velocity_inner, *(map(float, row[1:])))
            dfmodeldata = dfmodeldata.append([celltuple], ignore_index=True)

            # next inner is the current outer
            velocity_inner = celltuple.velocity_outer

            # the model.txt file may contain more shells, but we should ignore them
            # if we have already read in the specified number of shells
            if len(dfmodeldata) == gridcellcount:
                break

    assert len(dfmodeldata) <= gridcellcount
    dfmodeldata.index.name = 'cellid'

    t_model_init_seconds = t_model_init_days * 24 * 60 * 60
    piconst = math.pi
    dfmodeldata.eval('shellmass_grams = 10 ** logrho * 4. / 3. * @piconst * (velocity_outer ** 3 - velocity_inner ** 3)'
                     '* (1e5 * @t_model_init_seconds) ** 3', inplace=True)

    return dfmodeldata, t_model_init_days


def get_2d_modeldata(modelpath):
    filepath = os.path.join(modelpath, 'model.txt')
    num_lines = sum(1 for line in open(filepath))
    skiprowlist = [0, 1, 2]
    skiprowlistodds = skiprowlist + [i for i in range(3, num_lines) if i % 2 == 1]
    skiprowlistevens = skiprowlist + [i for i in range(3, num_lines) if i % 2 == 0]

    model1stlines = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=skiprowlistevens)
    model2ndlines = pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=skiprowlistodds)

    model = pd.concat([model1stlines, model2ndlines], axis=1)
    column_names = ['inputcellid', 'cellpos_mid[r]', 'cellpos_mid[z]', 'rho_model',
                    'ffe', 'fni', 'fco', 'ffe52', 'fcr48']
    model.columns = column_names
    return model


def get_3d_modeldata(modelpath):
    model = pd.read_csv(
        os.path.join(modelpath[0], 'model.txt'), delim_whitespace=True, header=None, skiprows=3, dtype=float)

    columns = ['inputcellid', 'cellpos_in[z]', 'cellpos_in[y]', 'cellpos_in[x]', 'rho_model',
               'ffe', 'fni', 'fco', 'ffe52', 'fcr48']
    model = pd.DataFrame(model.values.reshape(-1, 10))
    model.columns = columns
    return model


def save_modeldata(dfmodeldata, t_model_init_days, filename):
    """Save a pandas DataFrame into ARTIS model.txt"""
    with open(filename, 'w') as fmodel:
        fmodel.write(f'{len(dfmodeldata)}\n{t_model_init_days:f}\n')
        for _, cell in dfmodeldata.iterrows():
            fmodel.write(f'{cell.inputcellid:6.0f}   {cell.velocity_outer:9.2f}   {cell.logrho:10.8f} '
                         f'{cell.X_Fegroup:10.4e} {cell.X_Ni56:10.4e} {cell.X_Co56:10.4e} '
                         f'{cell.X_Fe52:10.4e} {cell.X_Cr48:10.4e}')
            if 'X_Ni57' in dfmodeldata.columns:
                fmodel.write(f' {cell.X_Ni57:10.4e}')
                if 'X_Co57' in dfmodeldata.columns:
                    fmodel.write(f' {cell.X_Co57:10.4e}')
            fmodel.write('\n')