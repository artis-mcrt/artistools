import os.path
from functools import lru_cache
import numpy as np
import pandas as pd
from collections import namedtuple
import math
import errno
import os
from pathlib import Path
import artistools
import gc


@lru_cache(maxsize=8)
def get_modeldata(inputpath=Path(), dimensions=None, get_abundances=False):
    """
    Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides.

    Arguments:
        - inputpath: either a path to model.txt file, or a folder containing model.txt
        - dimensions: number of dimensions in input file, or None for automatic
        - get_abundances: also read elemental abundances (abundances.txt) and
            merge with the output DataFrame

    Returns (dfmodeldata, t_model_init_days)
        - dfmodeldata: a pandas DataFrame with a row for each model grid cell
        - t_model_init_days: the time in days at which the snapshot is defined
    """

    assert dimensions in [1, 3, None]

    inputpath = Path(inputpath)

    if os.path.isdir(inputpath):
        modelpath = inputpath
        filename = artistools.firstexisting(['model.txt.xz', 'model.txt.gz', 'model.txt'], path=inputpath)
    elif os.path.isfile(inputpath):  # passed in a filename instead of the modelpath
        filename = inputpath
        modelpath = Path(inputpath).parent
    elif not inputpath.exists() and inputpath.parts[0] == 'codecomparison':
        modelpath = inputpath
        _, inputmodel, _ = modelpath.parts
        filename = Path(artistools.config['codecomparisonmodelartismodelpath'], inputmodel, 'model.txt')
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputpath)

    gridcelltuple = None
    velocity_inner = 0.
    with artistools.zopen(filename, 'rt') as fmodel:
        gridcellcount = int(artistools.readnoncommentline(fmodel))
        t_model_init_days = float(artistools.readnoncommentline(fmodel))
        t_model_init_seconds = t_model_init_days * 24 * 60 * 60

        filepos = fmodel.tell()
        # if the next line is a single float then the model is 3D
        try:

            vmax_cmps = float(artistools.readnoncommentline(fmodel))  # velocity max in cm/s
            xmax_tmodel = vmax_cmps * t_model_init_seconds  # xmax = ymax = zmax
            if dimensions is None:
                print("Detected 3D model file")
                dimensions = 3
            elif dimensions != 3:
                print(f" {dimensions} were specified but file appears to be 3D")
                assert False

        except ValueError:

            if dimensions is None:
                print("Detected 1D model file")
                dimensions = 1
            elif dimensions != 1:
                print(f" {dimensions} were specified but file appears to be 1D")
                assert False

            fmodel.seek(filepos)  # undo the readline() and go back

        if dimensions == 3:
            ncoordgridx = round(gridcellcount ** (1./3.))  # number of grid cell steps along an axis (same for xyz)
            ncoordgridy = round(gridcellcount ** (1./3.))
            ncoordgridz = round(gridcellcount ** (1./3.))

            assert (ncoordgridx * ncoordgridy * ncoordgridz) == gridcellcount
        # skiprows = 3 if dimensions == 3 else 2
        dfmodeldata = pd.read_csv(fmodel, delim_whitespace=True, header=None, dtype=np.float64, comment='#')
        if dimensions == 1:
            columns = ['inputcellid', 'velocity_outer', 'logrho', 'X_Fegroup', 'X_Ni56',
                       'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57']

        elif dimensions == 3:
            columns = ['inputcellid', 'inputpos_a', 'inputpos_b', 'inputpos_c', 'rho',
                       'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57']

            try:
                dfmodeldata = pd.DataFrame(dfmodeldata.values.reshape(-1, 12))
            except ValueError:
                dfmodeldata = pd.DataFrame(dfmodeldata.values.reshape(-1, 10))  # No Ni57 or Co57 columnss

        dfmodeldata.columns = columns[:len(dfmodeldata.columns)]

    dfmodeldata = dfmodeldata.iloc[:gridcellcount]

    assert len(dfmodeldata) == gridcellcount

    dfmodeldata.index.name = 'cellid'
    # dfmodeldata.drop('inputcellid', axis=1, inplace=True)

    if dimensions == 1:
        dfmodeldata['velocity_inner'] = np.concatenate([[0.], dfmodeldata['velocity_outer'].values[:-1]])
        piconst = math.pi
        dfmodeldata.eval(
            'shellmass_grams = 10 ** logrho * 4. / 3. * @piconst * (velocity_outer ** 3 - velocity_inner ** 3)'
            '* (1e5 * @t_model_init_seconds) ** 3', inplace=True)
        vmax_cmps = dfmodeldata.velocity_outer.max() * 1e5

    elif dimensions == 3:
        cellid = dfmodeldata.index.values
        xindex = cellid % ncoordgridx
        yindex = (cellid // ncoordgridx) % ncoordgridy
        zindex = (cellid // (ncoordgridx * ncoordgridy)) % ncoordgridz

        dfmodeldata['pos_x'] = -xmax_tmodel + 2 * xindex * xmax_tmodel / ncoordgridx
        dfmodeldata['pos_y'] = -xmax_tmodel + 2 * yindex * xmax_tmodel / ncoordgridy
        dfmodeldata['pos_z'] = -xmax_tmodel + 2 * zindex * xmax_tmodel / ncoordgridz

        wid_init = artistools.get_wid_init_at_tmodel(modelpath, gridcellcount, t_model_init_days, xmax_tmodel)
        dfmodeldata.eval('cellmass_grams = rho * @wid_init ** 3', inplace=True)

        def vectormatch(vec1, vec2):
            xclose = np.isclose(vec1[0], vec2[0], atol=xmax_tmodel / ncoordgridx)
            yclose = np.isclose(vec1[1], vec2[1], atol=xmax_tmodel / ncoordgridy)
            zclose = np.isclose(vec1[2], vec2[2], atol=xmax_tmodel / ncoordgridz)

            return all([xclose, yclose, zclose])

        posmatch_xyz = True
        posmatch_zyx = True
        # important cell numbers to check for coordinate column order
        indexlist = [0, ncoordgridx - 1, (ncoordgridx - 1) * (ncoordgridy - 1),
                     (ncoordgridx - 1) * (ncoordgridy - 1) * (ncoordgridz - 1)]
        for index in indexlist:
            cell = dfmodeldata.iloc[index]
            if not vectormatch([cell.inputpos_a, cell.inputpos_b, cell.inputpos_c],
                               [cell.pos_x, cell.pos_y, cell.pos_z]):
                posmatch_xyz = False
            if not vectormatch([cell.inputpos_a, cell.inputpos_b, cell.inputpos_c],
                               [cell.pos_z, cell.pos_y, cell.pos_x]):
                posmatch_zyx = False

        assert posmatch_xyz != posmatch_zyx  # one option must match
        if posmatch_xyz:
            print("Cell positions in model.txt are consistent with calculated values when x-y-z column order")
        if posmatch_zyx:
            print("Cell positions in model.txt are consistent with calculated values when z-y-x column order")

    if get_abundances:
        if dimensions == 3:
            print('Getting abundances')
        abundancedata = get_initialabundances(modelpath)
        dfmodeldata = dfmodeldata.merge(abundancedata, how='inner', on='inputcellid')

    return dfmodeldata, t_model_init_days, vmax_cmps


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


def get_3d_model_data_merged_model_and_abundances_minimal(args):
    """Get 3D data without generating all the extra columns in standard routine.
    Needed for large (eg. 200^3) models"""
    model = get_3d_modeldata_minimal(args.modelpath)
    abundances = get_initialabundances(args.modelpath[0])

    with open(os.path.join(args.modelpath[0], 'model.txt'), 'r') as fmodelin:
        fmodelin.readline()  # npts_model3d
        args.t_model = float(fmodelin.readline())  # days
        args.vmax = float(fmodelin.readline())  # v_max in [cm/s]

    print(model.keys())

    merge_dfs = model.merge(abundances, how='inner', on='inputcellid')

    del model
    del abundances
    gc.collect()

    merge_dfs.info(verbose=False, memory_usage="deep")

    return merge_dfs


def get_3d_modeldata_minimal(modelpath):
    """Read 3D model without generating all the extra columns in standard routine.
    Needed for large (eg. 200^3) models"""
    model = pd.read_csv(os.path.join(modelpath[0], 'model.txt'), delim_whitespace=True, header=None, skiprows=3, dtype=np.float64)
    columns = ['inputcellid', 'cellpos_in[z]', 'cellpos_in[y]', 'cellpos_in[x]', 'rho_model',
               'ffe', 'fni', 'fco', 'ffe52', 'fcr48']
    model = pd.DataFrame(model.values.reshape(-1, 10))
    model.columns = columns

    print('model.txt memory usage:')
    model.info(verbose=False, memory_usage="deep")
    return model


def save_modeldata(dfmodeldata, t_model_init_days, filename):
    """Save a pandas DataFrame and snapshot time into ARTIS model.txt"""
    standardcols = ['inputcellid',
                    'velocity_outer', 'logrho', 'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52',
                    'X_Cr48', 'X_Ni57', 'X_Co57']
    customcols = []
    for col in dfmodeldata.columns:
        if col not in standardcols:
            customcols.append(col)

    with open(filename, 'w') as fmodel:
        fmodel.write(f'{len(dfmodeldata)}\n{t_model_init_days:f}\n')
        fmodel.write('#' + "  ".join(standardcols))
        if customcols:
            fmodel.write("  " + "  ".join(customcols))
        fmodel.write('\n')
        for _, cell in dfmodeldata.iterrows():
            fmodel.write(f'{cell.inputcellid:6.0f}   {cell.velocity_outer:9.2f}   {cell.logrho:10.8f} '
                         f'{cell.X_Fegroup:10.4e} {cell.X_Ni56:10.4e} {cell.X_Co56:10.4e} '
                         f'{cell.X_Fe52:10.4e} {cell.X_Cr48:10.4e}')
            if 'X_Ni57' in dfmodeldata.columns or customcols:
                fmodel.write(f' {cell.X_Ni57:10.4e}')
                if 'X_Co57' in dfmodeldata.columns or customcols:
                    fmodel.write(f' {cell.X_Co57:10.4e}')
            if customcols:
                for col in customcols:
                    fmodel.write(f' {cell[col]:10.4e}')

            fmodel.write('\n')


def save_3d_modeldata(modelpath, griddata, t_model, vmax, radioactives=True):
    ngridpoints = len(griddata['gridindex'])  # xgrid * ygrid * zgrid
    gridsize = round(ngridpoints ** (1 / 3))
    print('grid size', gridsize)

    if not radioactives:
        ffe = 0.0
        fni = 0.0
        fco = 0.0
        ffe52 = 0.0
        fcr48 = 0.0

    with open(Path(modelpath) / 'model.txt', 'w') as fmodel:
        fmodel.write(f'{ngridpoints}\n')
        fmodel.write(f'{t_model}\n')
        fmodel.write(f'{vmax}\n')

        for i in griddata['gridindex']:
            line1 = [i, griddata['posx'][i - 1], griddata['posy'][i - 1], griddata['posz'][i - 1],
                     griddata['rho'][i - 1]]
            if not radioactives:
                line2 = [ffe, fni, fco, ffe52, fcr48]
            else:
                line2 = [griddata['ffe'][i - 1], griddata['fni'][i - 1], griddata['fco'][i - 1],
                         griddata['ffe52'][i - 1], griddata['fcr48'][i - 1]]

            fmodel.writelines("%s " % item for item in line1)
            fmodel.writelines("\n")
            fmodel.writelines("%s " % item for item in line2)
            fmodel.writelines("\n")


def get_mgi_of_velocity_kms(modelpath, velocity, mgilist=None):
    """Return the modelgridindex of the cell whose outer velocity is closest to velocity.
    If mgilist is given, then chose from these cells only"""
    modeldata, _, _ = get_modeldata(modelpath)

    if not mgilist:
        mgilist = [mgi for mgi in modeldata.index]
        arr_vouter = modeldata['velocity_outer'].values
    else:
        arr_vouter = np.array([modeldata['velocity_outer'][mgi] for mgi in mgilist])

    index_closestvouter = np.abs(arr_vouter - velocity).argmin()

    if velocity < arr_vouter[index_closestvouter] or index_closestvouter + 1 >= len(mgilist):
        return mgilist[index_closestvouter]
    elif velocity < arr_vouter[index_closestvouter + 1]:
        return mgilist[index_closestvouter + 1]
    elif np.isnan(velocity):
        return float('nan')
    else:
        print(f"Can't find cell with velocity of {velocity}. Velocity list: {arr_vouter}")
        assert(False)


@lru_cache(maxsize=8)
def get_initialabundances(modelpath):
    """Return a list of mass fractions."""
    abundancefilepath = artistools.firstexisting(
        ['abundances.txt.xz', 'abundances.txt.gz', 'abundances.txt'], path=modelpath)

    columns = ['inputcellid', *['X_' + artistools.elsymbols[x] for x in range(1, 31)]]
    abundancedata = pd.read_csv(abundancefilepath, delim_whitespace=True, header=None, names=columns)
    abundancedata.index.name = 'modelgridindex'
    if len(abundancedata) > 100000:
        print('abundancedata memory usage:')
        abundancedata.info(verbose=False, memory_usage="deep")
    return abundancedata


def save_initialabundances(dfabundances, abundancefilename):
    """Save a DataFrame (same format as get_initialabundances) to model.txt."""
    if Path(abundancefilename).is_dir():
        abundancefilename = Path(abundancefilename) / 'abundances.txt'
    dfabundances['inputcellid'] = dfabundances['inputcellid'].astype(int)
    dfabundances.to_csv(abundancefilename, header=False, sep=' ', index=False)


def save_empty_abundance_file(ngrid, outputfilepath='.'):
    """Dummy abundance file with only zeros"""
    Z_atomic = np.arange(1, 31)

    abundancedata = {'cellid': range(1, ngrid+1)}
    for atomic_number in Z_atomic:
        abundancedata[f'Z={atomic_number}'] = np.zeros(ngrid)

    # abundancedata['Z=28'] = np.ones(ngrid)

    abundancedata = pd.DataFrame(data=abundancedata)
    abundancedata = abundancedata.round(decimals=5)
    abundancedata.to_csv(Path(outputfilepath) / 'abundances.txt', header=False, sep='\t', index=False)
