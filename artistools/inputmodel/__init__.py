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
    elif not inputpath.exists() and inputpath.parts[0] == '_codecomparison':
        modelpath = inputpath
        _, inputmodel, _ = modelpath.parts
        filename = Path(artistools.config['codecomparisonmodelartismodelpath'], inputmodel, 'model.txt')
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), inputpath)

    gridcelltuple = None
    velocity_inner = 0.
    with open(filename, 'r') as fmodel:
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

        continuedfromrow = None
        lastinputcellid = 0
        recordlist = []
        for line in fmodel:
            if not line.strip() or line.lstrip().startswith('#'):
                continue

            row = line.split()

            if continuedfromrow is not None:
                row = continuedfromrow + row
                continuedfromrow = None

            if len(row) <= 5:  # rows are split across multiple lines
                continuedfromrow = row
                continue

            inputcellid = int(row[0])
            assert inputcellid == lastinputcellid + 1
            lastinputcellid = inputcellid

            if dimensions == 1:
                if gridcelltuple is None:
                    gridcelltuple = namedtuple('gridcell', [
                        'inputcellid', 'velocity_inner', 'velocity_outer', 'logrho',
                        'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57'][:len(row) + 1])

                assert len(row) >= 8
                celltuple = gridcelltuple(inputcellid, velocity_inner, *(map(float, row[1:])))
                recordlist.append(celltuple)

                # next inner is the current outer
                velocity_inner = celltuple.velocity_outer

                # the model.txt file may contain more shells, but we should ignore them
                # if we have already read in the specified number of shells
                if len(recordlist) == gridcellcount:
                    break

            elif dimensions == 3:
                assert len(row) >= 10  # can be 10 to 12 depending on presence of Ni57 and Co57 abundances

                if gridcelltuple is None:
                    # inputpos_a/b/c are used (instead of x/y/z) because these columns are used
                    # inconsistently between different scripts, and ignored by artis anyway
                    gridcelltuple = namedtuple('gridcell', [
                        'inputcellid', 'pos_x', 'pos_y', 'pos_z', 'inputpos_a', 'inputpos_b', 'inputpos_c', 'rho',
                        'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57'][:len(row) + 3])

                # increment x first, then y, then z
                # here inputcellid starts counting from one, so needs to be corrected
                cellid = inputcellid - 1
                xindex = cellid % ncoordgridx
                yindex = (cellid // ncoordgridx) % ncoordgridy
                zindex = (cellid // (ncoordgridx * ncoordgridy)) % ncoordgridz

                pos_x = -xmax_tmodel + 2 * xindex * xmax_tmodel / ncoordgridx
                pos_y = -xmax_tmodel + 2 * yindex * xmax_tmodel / ncoordgridy
                pos_z = -xmax_tmodel + 2 * zindex * xmax_tmodel / ncoordgridz

                celltuple = gridcelltuple(inputcellid, pos_x, pos_y, pos_z, *(map(float, row[1:])))
                recordlist.append(celltuple)

    dfmodeldata = pd.DataFrame.from_records(recordlist, columns=gridcelltuple._fields)

    assert len(dfmodeldata) == gridcellcount
    dfmodeldata.index.name = 'cellid'

    if get_abundances:
        abundancedata = get_initialabundances(modelpath)
        dfmodeldata = dfmodeldata.merge(abundancedata, how='inner', on='inputcellid')

    if dimensions == 1:
        piconst = math.pi
        dfmodeldata.eval(
            'shellmass_grams = 10 ** logrho * 4. / 3. * @piconst * (velocity_outer ** 3 - velocity_inner ** 3)'
            '* (1e5 * @t_model_init_seconds) ** 3', inplace=True)
        vmax_cmps = dfmodeldata.velocity_outer.max() * 1e5

    elif dimensions == 3:
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
    return abundancedata


def save_initialabundances(dfabundances, abundancefilename):
    """Save a DataFrame (same format as get_initialabundances) to model.txt."""
    dfabundances['inputcellid'] = dfabundances['inputcellid'].astype(int)
    dfabundances.to_csv(abundancefilename, header=False, sep=' ', index=False)
