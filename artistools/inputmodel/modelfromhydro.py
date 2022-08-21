#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import datetime
import math

import argparse
from pathlib import Path
import os.path

import argcomplete
import pandas as pd
from astropy import units as u
import numpy as np

import artistools as at

MSUN = 1.989e33
CLIGHT = 2.99792458e10


def read_ejectasnapshot(pathtosnapshot):

    column_names = ['id', 'h', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'vstx', 'vsty', 'vstz', 'u',
                    'psi', 'alpha', 'pmass', 'rho', 'p', 'rst', 'tau', 'av', 'ye', 'temp',
                    'prev_rho(i)', 'ynue(i)', 'yanue(i)', 'enuetrap(i)', 'eanuetrap(i)',
                    'enuxtrap(i)', 'iwasequil(i, 1)', 'iwasequil(i, 2)', 'iwasequil(i, 3)']

    types_dict = {'id': int}
    types_dict.update({col: float for col in column_names if col not in types_dict})

    ejectasnapshot = pd.read_csv(
        Path(pathtosnapshot) / "ejectasnapshot.dat", delim_whitespace=True, header=None,
        names=column_names, dtype=types_dict)

    # Everything is in geometric units here

    return ejectasnapshot


def get_snapshot_time_geomunits(pathtogriddata):
    import glob

    snapshotinfofile = glob.glob(str(Path(pathtogriddata) / "*_info.dat*"))
    if not snapshotinfofile:
        print("No info file found for dumpstep")
        quit()

    if len(snapshotinfofile) > 1:
        print('Too many sfho_info.dat files found')
        quit()
    snapshotinfofile = snapshotinfofile[0]

    if os.path.isfile(snapshotinfofile):
        with open(snapshotinfofile, "r") as fsnapshotinfo:
            line1 = fsnapshotinfo.readline()
            simulation_end_time_geomunits = float(line1.split()[2])
            print(f'Found simulation snapshot time to be {simulation_end_time_geomunits} '
                  f'({simulation_end_time_geomunits * 4.926e-6} s)')

    else:
        print("Could not find snapshot info file to get simulation time")
        quit()

    return simulation_end_time_geomunits


def read_griddat_file(pathtogriddata, targetmodeltime_days=None, minparticlespercell=0):
    griddatfilepath = Path(pathtogriddata) / "grid.dat"

    # Get simulation time for ejecta snapshot
    simulation_end_time_geomunits = get_snapshot_time_geomunits(pathtogriddata)

    griddata = pd.read_csv(griddatfilepath, delim_whitespace=True, comment='#', skiprows=3)
    griddata.rename(columns={
        'gridindex': 'inputcellid',
        'pos_x_min': 'pos_x_min',
        'pos_y_min': 'pos_y_min',
        'pos_z_min': 'pos_z_min',
    }, inplace=True)
    # griddata in geom units
    griddata['rho'] = np.nan_to_num(griddata['rho'], nan=0.)

    if 'cellYe' in griddata:
        griddata['cellYe'] = np.nan_to_num(griddata['cellYe'], nan=0.)
    if 'Q' in griddata:
        griddata['Q'] = np.nan_to_num(griddata['Q'], nan=0.)

    factor_position = 1.478  # in km
    km_to_cm = 1e5
    griddata.eval('pos_x_min = pos_x_min * @factor_position * @km_to_cm', inplace=True)
    griddata.eval('pos_y_min = pos_y_min * @factor_position * @km_to_cm', inplace=True)
    griddata.eval('pos_z_min = pos_z_min * @factor_position * @km_to_cm', inplace=True)

    griddata['rho'] = griddata['rho'] * 6.176e17  # convert to g/cm3

    with open(griddatfilepath, 'r', encoding='utf-8') as gridfile:
        ngrid = int(gridfile.readline().split()[0])
        if ngrid != len(griddata['inputcellid']):
            print("length of file and ngrid don't match")
            quit()
        extratime_geomunits = float(gridfile.readline().split()[0])
        xmax = abs(float(gridfile.readline().split()[0]))
        xmax = (xmax * factor_position) * (u.km).to(u.cm)

    t_model_sec = (simulation_end_time_geomunits + extratime_geomunits) * 4.926e-6  # in seconds
    vmax = xmax / t_model_sec  # cm/s

    t_model_days = t_model_sec / (24. * 3600)  # in days
    print(f"t_model in days {t_model_days} ({t_model_sec} s)")
    corner_vmax = math.sqrt(3 * vmax ** 2)
    print(f"vmax {vmax:.2e} cm/s ({vmax / 29979245800:.2f} * c) per component "
          f"real corner vmax {corner_vmax:.2e} cm/s ({corner_vmax / 29979245800:.2f} * c)")

    if targetmodeltime_days is not None:
        timefactor = targetmodeltime_days / t_model_days
        griddata.eval('pos_x_min = pos_x_min * @timefactor', inplace=True)
        griddata.eval('pos_y_min = pos_y_min * @timefactor', inplace=True)
        griddata.eval('pos_z_min = pos_z_min * @timefactor', inplace=True)
        griddata.eval('rho = rho * @timefactor ** -3', inplace=True)
        print(f"Adjusting t_model to {targetmodeltime_days} days (factor {timefactor}) "
              "using homologous expansion of positions and densities")
        t_model_days = targetmodeltime_days

    if minparticlespercell > 0:
        ncoordgridx = round(len(griddata) ** (1. / 3.))
        xmax = - griddata.pos_x_min.min()
        wid_init = 2 * xmax / ncoordgridx
        filter = np.logical_and(griddata.tracercount < minparticlespercell, griddata.rho > 0.)
        n_ignored = np.count_nonzero(filter)
        mass_ignored = griddata.loc[filter].rho.sum() * wid_init ** 3 / 1.989e33
        mass_orig = griddata.rho.sum() * wid_init ** 3 / 1.989e33

        print(f"Ignoring {n_ignored} nonempty cells ({mass_ignored:.2e} Msun, {mass_ignored / mass_orig * 100:.2f}% of mass) because they have < {minparticlespercell} tracer particles")
        griddata.loc[griddata.tracercount < minparticlespercell, ['rho', 'cellYe']] = 0., 0.

    print(f"Max tracers in a cell {max(griddata['tracercount'])}")

    return griddata, t_model_days, vmax


def read_mattia_grid_data_file(pathtogriddata):
    # griddatfilepath = Path(pathtogriddata) / "q90_m0.01_v0.1.txt"
    griddatfilepath = Path(pathtogriddata) / "1D_m0.01_v0.1.txt"

    griddata = pd.read_csv(griddatfilepath, delim_whitespace=True, comment='#', skiprows=1)
    with open(griddatfilepath, 'r') as gridfile:
        t_model = float(gridfile.readline())
        print(f't_model {t_model} seconds')
    xmax = max(griddata['posx'])
    vmax = xmax / t_model  # cm/s
    t_model = t_model / (24. * 3600)  # days
    ngrid = len(griddata['posx'])

    griddata['rho'][griddata['rho'] <= 1e-50] = 0.
    inputcellid = np.arange(1, ngrid+1)
    griddata['inputcellid'] = inputcellid

    return griddata, t_model, vmax


def mirror_model_in_axis(griddata):
    grid = round(len(griddata) ** (1. / 3.))

    rho = np.zeros((grid, grid, grid))
    cellYe = np.zeros((grid, grid, grid))
    tracercount = np.zeros((grid, grid, grid))
    Q = np.zeros((grid, grid, grid))

    i = 0
    for z in range(0, grid):
        for y in range(0, grid):
            for x in range(0, grid):
                rho[x, y, z] = griddata['rho'][i]
                cellYe[x, y, z] = griddata['cellYe'][i]
                tracercount[x, y, z] = griddata['tracercount'][i]
                Q[x, y, z] = griddata['Q'][i]
                i += 1

    for z in range(0, grid):
        z_mirror = grid-1 - z
        for y in range(0, grid):
            for x in range(0, grid):
                if z < 50:
                    rho[x, y, z] = rho[x, y, z]
                    cellYe[x, y, z] = cellYe[x, y, z]
                    tracercount[x, y, z] = tracercount[x, y, z]
                    Q[x, y, z] = Q[x, y, z]
                if z >= 50:
                    rho[x, y, z] = rho[x, y, z_mirror]
                    cellYe[x, y, z] = cellYe[x, y, z_mirror]
                    tracercount[x, y, z] = tracercount[x, y, z_mirror]
                    Q[x, y, z] = Q[x, y, z_mirror]

    rho_1d_array = np.zeros(len(griddata))
    cellYe_1d_array = np.zeros(len(griddata))
    tracercount_1d_array = np.zeros(len(griddata))
    Q_1d_array = np.zeros(len(griddata))
    i = 0
    for z in range(0, grid):
        for y in range(0, grid):
            for x in range(0, grid):
                rho_1d_array[i] = rho[x, y, z]
                cellYe_1d_array[i] = cellYe[x, y, z]
                tracercount_1d_array[i] = tracercount[x, y, z]
                Q_1d_array[i] = Q[x, y, z]
                i += 1

    griddata['rho'] = rho_1d_array
    griddata['cellYe'] = cellYe_1d_array
    griddata['tracercount'] = tracercount_1d_array
    griddata['Q'] = Q_1d_array

    return griddata


def add_mass_to_center(griddata, t_model_in_days, vmax, args):
    print(griddata)

    # Just (2021) Fig. 16 top left panel
    vel_hole = [0, 0.02, 0.05, 0.07, 0.09, 0.095, 0.1]
    mass_hole = [3e-4, 3e-4, 2e-4, 1e-4, 2e-5, 1e-5, 1e-9]
    mass_intergrated = np.trapz(y=mass_hole, x=vel_hole)  # Msun

    # # Just (2021) Fig. 16 4th down, left panel
    # vel_hole = [0, 0.02, 0.05, 0.1, 0.15, 0.16]
    # mass_hole = [4e-3, 2e-3, 1e-3, 1e-4, 6e-6, 1e-9]
    # mass_intergrated = np.trapz(y=mass_hole, x=vel_hole)  # Msun

    v_outer_hole = 0.1 * CLIGHT  # cm/s
    pos_outer_hole = v_outer_hole * t_model_in_days * (24. * 3600)  # cm
    vol_hole = 4 / 3 * np.pi * pos_outer_hole ** 3  # cm^3
    density_hole = (mass_intergrated * MSUN) / vol_hole  # g / cm^3
    print(density_hole)

    for i, cellid in enumerate(griddata['inputcellid']):
        # if pos < 0.1 c
        if ((np.sqrt(griddata['pos_x_min'][i] ** 2 + griddata['pos_y_min'][i] ** 2 + griddata['pos_z_min'][i] ** 2)) /
                (t_model_in_days * (24. * 3600)) / CLIGHT) < 0.1:
            # if griddata['rho'][i] == 0:
            print("Inner empty cells")
            print(cellid, griddata['pos_x_min'][i], griddata['pos_y_min'][i], griddata['pos_z_min'][i], griddata['rho'][i])
            griddata['rho'][i] += density_hole
            if griddata['cellYe'][i] < 0.4:
                griddata['cellYe'][i] = 0.4
            # print("Inner empty cells filled")
            print(cellid, griddata['pos_x_min'][i], griddata['pos_y_min'][i], griddata['pos_z_min'][i], griddata['rho'][i])

    return griddata


def makemodelfromgriddata(
        gridfolderpath=Path(), outputpath=Path(), targetmodeltime_days=None,
        minparticlespercell=0, traj_root=None, dimensions=3, args=None):

    assert dimensions in [1, 3]
    headerlines = [f'gridfolder: {Path(gridfolderpath).resolve().parts[-1]}']
    dfmodel, t_model_days, vmax = at.inputmodel.modelfromhydro.read_griddat_file(
        pathtogriddata=gridfolderpath, targetmodeltime_days=targetmodeltime_days,
        minparticlespercell=minparticlespercell)

    if getattr(args, 'fillcentralhole', False):
        dfmodel = at.inputmodel.modelfromhydro.add_mass_to_center(dfmodel, t_model_days, vmax, args)

    if getattr(args, 'getcellopacityfromYe', False):
        at.inputmodel.opacityinputfile.opacity_by_Ye(outputpath, dfmodel)

    if os.path.isfile(Path(gridfolderpath, 'gridcontributions.txt')):
        dfgridcontributions = at.inputmodel.rprocess_from_trajectory.get_gridparticlecontributions(gridfolderpath)
    else:
        dfgridcontributions = None

    if traj_root is not None:
        print(f'Nuclear network abundances from {traj_root} will be used')
        headerlines.append(f'trajfolder: {Path(traj_root).resolve().parts[-1]}')
        dfmodel, dfelabundances, dfgridcontributions = (
            at.inputmodel.rprocess_from_trajectory.add_abundancecontributions(
                dfgridcontributions=dfgridcontributions, dfmodel=dfmodel, t_model_days=t_model_days,
                minparticlespercell=minparticlespercell, traj_root=traj_root))
    else:
        print('WARNING: No abundances will be set because no nuclear network trajectories folder was specified')
        dfelabundances = None

    if dimensions == 1:
        dfmodel, dfelabundances, dfgridcontributions = at.inputmodel.sphericalaverage(
            dfmodel, t_model_days, vmax, dfelabundances, dfgridcontributions)

    if 'cellYe' in dfmodel:
        at.inputmodel.opacityinputfile.write_Ye_file(outputpath, dfmodel)

    if 'Q' in dfmodel and args.makeenergyinputfiles:
        at.inputmodel.energyinputfiles.write_Q_energy_file(outputpath, dfmodel)

    if dfgridcontributions is not None:
        at.inputmodel.rprocess_from_trajectory.save_gridparticlecontributions(
            dfgridcontributions, Path(outputpath, 'gridcontributions.txt'))

    headerlines.append(f'generated at (UTC): {datetime.datetime.utcnow()}')

    if traj_root is not None:
        print(f'Writing to {Path(outputpath) / "abundances.txt"}...')
        at.inputmodel.save_initialabundances(
            dfelabundances=dfelabundances, abundancefilename=outputpath, headerlines=headerlines)
    else:
        at.inputmodel.save_empty_abundance_file(len(dfmodel))

    print(f'Writing to {Path(outputpath) / "model.txt"}...')
    at.inputmodel.save_modeldata(
        modelpath=outputpath, dfmodel=dfmodel, t_model_init_days=t_model_days, dimensions=dimensions, vmax=vmax,
        headerlines=headerlines)


def addargs(parser):
    parser.add_argument('-gridfolderpath', '-i',
                        default='.',
                        help='Path to folder containing grid.dat and gridcontributions.dat')
    parser.add_argument('-minparticlespercell',
                        default=0, type=int,
                        help='Minimum number of SPH particles in each cell (otherwise set rho=0)')
    parser.add_argument('-trajectoryroot',
                        default=None,
                        help='Path to nuclear network trajectory folder, if abundances are requierd')
    parser.add_argument('-dimensions',
                        default=3, type=int,
                        help='Number of dimensions: 1 for spherically symmetric 1D, 3 for 3D Cartesian')
    parser.add_argument('-targetmodeltime_days', '-t', type=float,
                        default=1.,
                        help='Time in days for the output model snapshot')
    parser.add_argument('--noabundances', action='store_true',
                        help='Skip trajectory abundance mapping (get densities only)')
    parser.add_argument('-outputpath', '-o',
                        default=None,
                        help='Path for output model files')


def main(args=None, argsraw=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser(
            formatter_class=at.CustomArgHelpFormatter,
            description='Create ARTIS format model from grid.dat.')

        addargs(parser)
        parser.set_defaults(**kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args(argsraw)

    gridfolderpath = args.gridfolderpath
    if not Path(gridfolderpath, 'grid.dat').is_file() or not Path(gridfolderpath, 'gridcontributions.txt').is_file():
        print('grid.dat and gridcontributions.txt are required. Run artistools-maptogrid')
        return
        # at.inputmodel.maptogrid.main()

    if args.outputpath is None:
        outputpath = Path(f'artismodel_{args.dimensions}d')
    else:
        outputpath = Path(args.outputpath)

    outputpath.mkdir(parents=True, exist_ok=True)

    makemodelfromgriddata(
        gridfolderpath=gridfolderpath, outputpath=outputpath, minparticlespercell=args.minparticlespercell,
        targetmodeltime_days=args.targetmodeltime_days, traj_root=args.trajectoryroot,
        dimensions=args.dimensions)


if __name__ == "__main__":
    main()
