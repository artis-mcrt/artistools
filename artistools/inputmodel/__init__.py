import errno
import gc
# from collections import namedtuple
import math
import os
import os.path
import time
from functools import lru_cache
from pathlib import Path

import artistools
import artistools.inputmodel.botyanski2017
import artistools.inputmodel.describeinputmodel
import artistools.inputmodel.makeartismodel
import artistools.inputmodel.rprocess_from_trajectory
import numpy as np
import pandas as pd


@lru_cache(maxsize=8)
def get_modeldata(inputpath=Path(), dimensions=None, get_abundances=False, derived_cols=False):
    """
    Read an artis model.txt file containing cell velocities, density, and abundances of radioactive nuclides.

    Arguments:
        - inputpath: either a path to model.txt file, or a folder containing model.txt
        - dimensions: number of dimensions in input file, or None for automatic
        - get_abundances: also read elemental abundances (abundances.txt) and
            merge with the output DataFrame

    Returns (dfmodel, t_model_init_days)
        - dfmodel: a pandas DataFrame with a row for each model grid cell
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

        columns = None
        filepos = fmodel.tell()
        line = fmodel.readline()
        if line.startswith('#'):
            columns = line.lstrip('#').split()
        else:
            fmodel.seek(filepos)  # undo the readline() and go back

        if dimensions == 3:
            ncoordgridx = int(round(gridcellcount ** (1. / 3.)))  # number of grid cell steps along an axis (same for xyz)
            ncoordgridy = int(round(gridcellcount ** (1. / 3.)))
            ncoordgridz = int(round(gridcellcount ** (1. / 3.)))

            assert (ncoordgridx * ncoordgridy * ncoordgridz) == gridcellcount

        # skiprows = 3 if dimensions == 3 else 2
        dfmodel = pd.read_csv(fmodel, delim_whitespace=True, header=None, dtype=np.float64, comment='#')

        if dimensions == 1 and columns is None:
            columns = ['inputcellid', 'velocity_outer', 'logrho', 'X_Fegroup', 'X_Ni56',
                       'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57']

        elif dimensions == 3 and columns is None:
            columns = ['inputcellid', 'inputpos_a', 'inputpos_b', 'inputpos_c', 'rho',
                       'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48', 'X_Ni57', 'X_Co57']

            try:
                dfmodel = pd.DataFrame(dfmodel.values.reshape(-1, 12))
            except ValueError:
                dfmodel = pd.DataFrame(dfmodel.values.reshape(-1, 10))  # No Ni57 or Co57 columnss

        dfmodel.columns = columns[:len(dfmodel.columns)]

    dfmodel = dfmodel.iloc[:gridcellcount]

    assert len(dfmodel) == gridcellcount

    dfmodel.index.name = 'cellid'
    # dfmodel.drop('inputcellid', axis=1, inplace=True)

    if dimensions == 1:
        dfmodel['velocity_inner'] = np.concatenate([[0.], dfmodel['velocity_outer'].values[:-1]])
        piconst = math.pi
        dfmodel.eval(
            'shellmass_grams = 10 ** logrho * 4. / 3. * @piconst * (velocity_outer ** 3 - velocity_inner ** 3)'
            '* (1e5 * @t_model_init_seconds) ** 3', inplace=True)
        vmax_cmps = dfmodel.velocity_outer.max() * 1e5

    elif dimensions == 3:
        cellid = dfmodel.index.values
        xindex = cellid % ncoordgridx
        yindex = (cellid // ncoordgridx) % ncoordgridy
        zindex = (cellid // (ncoordgridx * ncoordgridy)) % ncoordgridz

        dfmodel['pos_x'] = -xmax_tmodel + 2 * xindex * xmax_tmodel / ncoordgridx
        dfmodel['pos_y'] = -xmax_tmodel + 2 * yindex * xmax_tmodel / ncoordgridy
        dfmodel['pos_z'] = -xmax_tmodel + 2 * zindex * xmax_tmodel / ncoordgridz

        wid_init = artistools.get_wid_init_at_tmodel(modelpath, gridcellcount, t_model_init_days, xmax_tmodel)
        dfmodel.eval('cellmass_grams = rho * @wid_init ** 3', inplace=True)

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
            cell = dfmodel.iloc[index]
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
        dfmodel = dfmodel.merge(abundancedata, how='inner', on='inputcellid')

    if derived_cols:
        add_derived_cols_to_modeldata(dfmodel, derived_cols, dimensions, t_model_init_seconds, wid_init, modelpath)

    return dfmodel, t_model_init_days, vmax_cmps


def add_derived_cols_to_modeldata(dfmodel, derived_cols, dimensions=None, t_model_init_seconds=None, wid_init=None,
                                  modelpath=None):
    """add columns to modeldata using e.g. derived_cols = ('velocity', 'Ye')"""
    if dimensions == 3 and 'velocity' in derived_cols:
        dfmodel['vel_x_min'] = dfmodel['pos_x'] / t_model_init_seconds
        dfmodel['vel_y_min'] = dfmodel['pos_y'] / t_model_init_seconds
        dfmodel['vel_z_min'] = dfmodel['pos_z'] / t_model_init_seconds

        dfmodel['vel_x_max'] = (dfmodel['pos_x'] + wid_init) / t_model_init_seconds
        dfmodel['vel_y_max'] = (dfmodel['pos_y'] + wid_init) / t_model_init_seconds
        dfmodel['vel_z_max'] = (dfmodel['pos_z'] + wid_init) / t_model_init_seconds

        dfmodel['vel_x_mid'] = (dfmodel['pos_x'] + (0.5 * wid_init)) / t_model_init_seconds
        dfmodel['vel_y_mid'] = (dfmodel['pos_y'] + (0.5 * wid_init)) / t_model_init_seconds
        dfmodel['vel_z_mid'] = (dfmodel['pos_z'] + (0.5 * wid_init)) / t_model_init_seconds

        dfmodel['vel_angle_ave'] = np.sqrt(np.array(dfmodel['vel_x_mid']) ** 2
                                             + np.array(dfmodel['vel_y_mid']) ** 2
                                             + np.array(dfmodel['vel_z_mid']) ** 2)

    if dimensions == 3 and 'pos_mid' in derived_cols or 'angle_bin' in derived_cols:
        dfmodel['pos_x_mid'] = (dfmodel['pos_x'] + (0.5 * wid_init))
        dfmodel['pos_y_mid'] = (dfmodel['pos_y'] + (0.5 * wid_init))
        dfmodel['pos_z_mid'] = (dfmodel['pos_z'] + (0.5 * wid_init))

    if 'angle_bin' in derived_cols:
        get_cell_angle(dfmodel, modelpath)

    if 'Ye' in derived_cols and os.path.isfile(modelpath / 'Ye.txt'):
        dfmodel['Ye'] = artistools.inputmodel.opacityinputfile.get_Ye_from_file(modelpath)
    if 'Q' in derived_cols and os.path.isfile(modelpath / 'Q_energy.txt'):
        dfmodel['Q'] = artistools.inputmodel.energyinputfiles.get_Q_energy_from_file(modelpath)

    return dfmodel


def get_cell_angle(dfmodel, modelpath):
    """get angle between cell midpoint and axis"""
    syn_dir = artistools.get_syn_dir(modelpath)

    cos_theta = np.zeros(len(dfmodel))
    i = 0
    for _, cell in dfmodel.iterrows():
        mid_point = [cell['pos_x_mid'], cell['pos_y_mid'], cell['pos_z_mid']]
        cos_theta[i] = (
            artistools.dot(mid_point, syn_dir)) / (artistools.vec_len(mid_point) * artistools.vec_len(syn_dir))
        i += 1
    dfmodel['cos_theta'] = cos_theta
    cos_bins = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]  # including end bin
    labels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]  # to agree with escaping packet bin numbers
    dfmodel['cos_bin'] = pd.cut(dfmodel['cos_theta'], cos_bins, labels=labels)
    # dfmodel['cos_bin'] = np.searchsorted(cos_bins, dfmodel['cos_theta'].values) -1

    return dfmodel


def get_mean_cell_properties_of_angle_bin(dfmodeldata, vmax_cmps, modelpath=None):
    if 'cos_bin' not in dfmodeldata:
        get_cell_angle(dfmodeldata, modelpath)

    dfmodeldata['rho'][dfmodeldata['rho'] == 0] = None
    dfmodeldata['rho']

    cell_velocities = np.unique(dfmodeldata['vel_x_min'].values)
    cell_velocities = cell_velocities[cell_velocities >= 0]
    velocity_bins = np.append(cell_velocities, vmax_cmps)

    mid_velocities = np.unique(dfmodeldata['vel_x_mid'].values)
    mid_velocities = mid_velocities[mid_velocities >= 0]

    mean_bin_properties = {}
    for bin_number in range(10):
        mean_bin_properties[bin_number] = pd.DataFrame({'velocity': mid_velocities,
                                                        'mean_rho': np.zeros_like(mid_velocities, dtype=float),
                                                        'mean_Ye': np.zeros_like(mid_velocities, dtype=float),
                                                        'mean_Q': np.zeros_like(mid_velocities, dtype=float)})

    # cos_bin_number = 90
    for bin_number in range(10):
        cos_bin_number = bin_number * 10
        # get cells with bin number
        dfanglebin = dfmodeldata.query('cos_bin == @cos_bin_number', inplace=False)

        binned = pd.cut(dfanglebin['vel_angle_ave'], velocity_bins, labels=False, include_lowest=True)
        i=0
        for binindex, mean_rho in dfanglebin.groupby(binned)['rho'].mean().iteritems():
            i+=1
            mean_bin_properties[bin_number]['mean_rho'][binindex] += mean_rho
        i=0
        if 'Ye' in dfmodeldata.keys():
            for binindex, mean_Ye in dfanglebin.groupby(binned)['Ye'].mean().iteritems():
                i+=1
                mean_bin_properties[bin_number]['mean_Ye'][binindex] += mean_Ye
        if 'Q' in dfmodeldata.keys():
            for binindex, mean_Q in dfanglebin.groupby(binned)['Q'].mean().iteritems():
                i+=1
                mean_bin_properties[bin_number]['mean_Q'][binindex] += mean_Q

    return mean_bin_properties


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
                    'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48']
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
    model = pd.read_csv(os.path.join(modelpath[0], 'model.txt'),
                        delim_whitespace=True, header=None, skiprows=3, dtype=np.float64)
    columns = ['inputcellid', 'cellpos_in[z]', 'cellpos_in[y]', 'cellpos_in[x]', 'rho_model',
               'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52', 'X_Cr48']
    model = pd.DataFrame(model.values.reshape(-1, 10))
    model.columns = columns

    print('model.txt memory usage:')
    model.info(verbose=False, memory_usage="deep")
    return model


def save_modeldata(
        dfmodel, t_model_init_days, filename=None, modelpath=None, vmax=None, dimensions=1, radioactives=True):
    """Save a pandas DataFrame and snapshot time into ARTIS model.txt"""

    timestart = time.perf_counter()
    assert dimensions in [1, 3, None]
    if dimensions == 1:
        assert vmax is None
        standardcols = ['inputcellid', 'velocity_outer', 'logrho', 'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52',
                        'X_Cr48']
    elif dimensions == 3:
        dfmodel.rename(columns={'gridindex': 'inputcellid'}, inplace=True)
        griddimension = int(round(len(dfmodel) ** (1. / 3.)))
        print(f' grid size: {len(dfmodel)} ({griddimension}^3)')
        assert griddimension ** 3 == len(dfmodel)

        standardcols = ['inputcellid', 'posx', 'posy', 'posz', 'rho',  'X_Fegroup', 'X_Ni56', 'X_Co56', 'X_Fe52',
                        'X_Cr48']

    # these two columns are optional, but position is important and they must appear before any other custom cols
    if 'X_Ni57' in dfmodel.columns:
        standardcols.append('X_Ni57')

    if 'X_Co57' in dfmodel.columns:
        standardcols.append('X_Co57')

    dfmodel['inputcellid'] = dfmodel['inputcellid'].astype(int)
    customcols = [col for col in dfmodel.columns if col not in standardcols and col.startswith('X_')]

    # set missing radioabundance columns to zero
    for col in standardcols:
        if col not in dfmodel.columns and col.startswith('X_'):
            dfmodel[col] = 0.0

    assert modelpath is not None or filename is not None
    if filename is None:
        filename = 'model.txt'
    if modelpath is not None:
        modelfilepath = Path(modelpath, filename)
    else:
        modelfilepath = Path(filename)

    with open(modelfilepath, 'w') as fmodel:
        fmodel.write(f'{len(dfmodel)}\n')
        fmodel.write(f'{t_model_init_days}\n')
        if dimensions == 3:
            fmodel.write(f'{vmax}\n')

        if customcols:
            fmodel.write(f'#{"  ".join(standardcols)} {"  ".join(customcols)}\n')

        abundcols = [*[col for col in standardcols if col.startswith('X_')], *customcols]

        # for cell in dfmodel.itertuples():
        #     if dimensions == 1:
        #         fmodel.write(f'{cell.inputcellid:6d}   {cell.velocity_outer:9.2f}   {cell.logrho:10.8f} ')
        #     elif dimensions == 3:
        #         fmodel.write(f"{cell.inputcellid:6d} {cell.posx} {cell.posy} {cell.posz} {cell.rho}\n")
        #
        #     fmodel.write(" ".join([f'{getattr(cell, col)}' for col in abundcols]))
        #
        #     fmodel.write('\n')
        if dimensions == 1:

            for cell in dfmodel.itertuples():
                fmodel.write(f'{cell.inputcellid:6d}   {cell.velocity_outer:9.2f}   {cell.logrho:10.8f} ')
                fmodel.write(" ".join([f'{getattr(cell, col)}' for col in abundcols]))
                fmodel.write('\n')

        elif dimensions == 3:

            for inputcellid, posx, posy, posz, rho, *massfracs in dfmodel[
                    ['inputcellid', 'posx', 'posy', 'posz', 'rho', *abundcols]].itertuples(index=False, name=None):

                fmodel.write(f"{inputcellid:6d} {posx} {posy} {posz} {rho}\n")
                fmodel.write(" ".join([f'{abund}' for abund in massfracs]))
                fmodel.write('\n')

    print(f'Saved {filename} (took {time.perf_counter() - timestart:.1f} seconds)')


def get_mgi_of_velocity_kms(modelpath, velocity, mgilist=None):
    """Return the modelgridindex of the cell whose outer velocity is closest to velocity.
    If mgilist is given, then chose from these cells only"""
    modeldata, _, _ = get_modeldata(modelpath)

    velocity = float(velocity)

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

    abundancedata = pd.read_csv(abundancefilepath, delim_whitespace=True, header=None, dtype=np.float64)
    abundancedata.index.name = 'modelgridindex'
    abundancedata.columns = ['inputcellid', *['X_' + artistools.get_elsymbol(x) for x in range(1, len(abundancedata.columns))]]
    if len(abundancedata) > 100000:
        print('abundancedata memory usage:')
        abundancedata.info(verbose=False, memory_usage="deep")
    return abundancedata


def save_initialabundances(dfelabundances, abundancefilename):
    """Save a DataFrame (same format as get_initialabundances) to abundances.txt.
        columns must be:
            - inputcellid: integer index to match model.txt (starting from 1)
            - X_El: mass fraction of element with two-letter code 'El' (e.g., X_H, X_He, H_Li, ...)
    """
    timestart = time.perf_counter()
    if Path(abundancefilename).is_dir():
        abundancefilename = Path(abundancefilename) / 'abundances.txt'
    dfelabundances['inputcellid'] = dfelabundances['inputcellid'].astype(int)
    atomic_numbers = [artistools.get_atomic_number(colname[2:])
                      for colname in dfelabundances.columns if colname.startswith('X_')]
    elcolnames = [f'X_{artistools.get_elsymbol(Z)}' for Z in range(1, 1 + max(atomic_numbers))]

    # set missing elemental abundance columns to zero
    for col in elcolnames:
        if col not in dfelabundances.columns:
            dfelabundances[col] = 0.0

    with open(abundancefilename, 'w') as fabund:
        for row in dfelabundances.itertuples(index=False):
            fabund.write(f'{row.inputcellid}')
            fabund.write(" ".join([f' {getattr(row, colname, 0.)}' for colname in elcolnames]))
            fabund.write("\n")

    print(f'Saved {abundancefilename} (took {time.perf_counter() - timestart:.1f} seconds)')


def save_empty_abundance_file(ngrid, outputfilepath='.'):
    """Dummy abundance file with only zeros"""
    Z_atomic = np.arange(1, 31)

    abundancedata = {'cellid': range(1, ngrid + 1)}
    for atomic_number in Z_atomic:
        abundancedata[f'Z={atomic_number}'] = np.zeros(ngrid)

    # abundancedata['Z=28'] = np.ones(ngrid)

    abundancedata = pd.DataFrame(data=abundancedata)
    abundancedata = abundancedata.round(decimals=5)
    abundancedata.to_csv(Path(outputfilepath) / 'abundances.txt', header=False, sep='\t', index=False)
