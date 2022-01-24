from pathlib import Path
import os.path
import pandas as pd
from astropy import units as u
import numpy as np

MSUN = 1.989e33
CLIGHT = 2.99792458e10


def read_sph_out(pathtosnapshot):
    arepo_out_file = Path(pathtosnapshot) / "ejectasnapshot.dat"

    column_names = ['h (i)', 'x (i)', 'y (i)', 'z (i)', 'vx (i)', 'vy (i)', 'vz (i)', 'vstx (i)',
                    'vsty (i)',
                    'vstz (i)', 'u (i)', 'psi(i)', 'alpha(i)', 'pmass (i)', 'rho (i)', 'p(i)', 'rst(i)',
                    'tau (i)', 'av (i)', 'ye(i)', 'temp(i)']
    # Everything is in geometric units here
    sph_out_dat = pd.read_csv(arepo_out_file, delim_whitespace=True, header=None, names=column_names)
    print("total mass", sum(sph_out_dat['pmass (i)']))

    return sph_out_dat


def get_snapshot_time(pathtogriddata):
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
            print(f"Found simulation snapshot time to be {simulation_end_time_geomunits}")

    else:
        print("Could not find snapshot info file to get simulation time")
        quit()

    return simulation_end_time_geomunits


def read_griddat_file(pathtogriddata):
    griddatfilepath = Path(pathtogriddata) / "grid.dat"

    # Get simulation time for ejecta snapshot
    simulation_end_time_geomunits = get_snapshot_time(pathtogriddata)

    griddata = pd.read_csv(griddatfilepath, delim_whitespace=True, comment='#', skiprows=3)

    griddata['cellYe'] = np.nan_to_num(griddata['cellYe'], nan=0.)
    griddata['rho'] = np.nan_to_num(griddata['rho'], nan=0.)

    factor_position = 1.478  # in km
    griddata['posx'] = (griddata['posx'] * factor_position) * (u.km).to(u.cm)
    griddata['posy'] = (griddata['posy'] * factor_position) * (u.km).to(u.cm)
    griddata['posz'] = (griddata['posz'] * factor_position) * (u.km).to(u.cm)

    griddata['rho'] = griddata['rho'] * 6.176e17  # g/cm3

    with open(griddatfilepath, 'r') as gridfile:
        ngrid = int(gridfile.readline().split()[0])
        if ngrid != len(griddata['gridindex']):
            print("length of file and ngrid don't match")
            quit()
        extratime_geomunits = float(gridfile.readline().split()[0])
        xmax = abs(float(gridfile.readline().split()[0]))
        xmax = (xmax * factor_position) * (u.km).to(u.cm)

    t_model = (simulation_end_time_geomunits + extratime_geomunits) * 4.926e-6  # in seconds
    vmax = xmax / t_model  # cm/s

    t_model = t_model / (24. * 3600)  # in days
    print("t_model in days", t_model)

    print("Ignoring cells with < 10 tracer particles")
    griddata.loc[griddata.tracercount < 10, ['rho', 'cellYe']] = 0, 0

    print(f"Max tracers in a cell {max(griddata['tracercount'])}")

    return griddata, t_model, vmax


def fill_central_hole(griddata, t_model):
    print(griddata)
def mirror_model_in_axis(griddata):
    grid = round(len(griddata) ** (1. / 3.))

    rho = np.zeros((grid, grid, grid))
    cellYe = np.zeros((grid, grid, grid))
    tracercount = np.zeros((grid, grid, grid))
    Q = np.zeros((grid, grid, grid))

    i=0
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
    i=0
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


    # Just (2021) Fig. 16 top left panel
    vel_hole = [0, 0.02, 0.05, 0.07, 0.09, 0.095, 0.1]
    mass_hole = [3e-4, 3e-4, 2e-4, 1e-4, 2e-5, 1e-5, 1e-9]
    mass_intergrated = np.trapz(y=mass_hole, x=vel_hole)  # Msun

    # # Just (2021) Fig. 16 4th down, left panel
    # vel_hole = [0, 0.02, 0.05, 0.1, 0.15, 0.16]
    # mass_hole = [4e-3, 2e-3, 1e-3, 1e-4, 6e-6, 1e-9]
    # mass_intergrated = np.trapz(y=mass_hole, x=vel_hole)  # Msun

    v_outer_hole = 0.1 * CLIGHT  # cm/s
    pos_outer_hole = v_outer_hole * t_model * (24. * 3600)  # cm
    vol_hole = 4 / 3 * np.pi * pos_outer_hole ** 3  # cm^3
    density_hole = (mass_intergrated * MSUN) / vol_hole  # g / cm^3
    print(density_hole)

    for i, cellid in enumerate(griddata['gridindex']):
        # if pos < 0.1 c
        if ((np.sqrt(griddata['posx'][i] ** 2 + griddata['posy'][i] ** 2 + griddata['posz'][i] ** 2)) /
                (t_model * (24. * 3600)) / CLIGHT) < 0.1:
            # if griddata['rho'][i] == 0:
            print("Inner empty cells")
            print(cellid, griddata['posx'][i], griddata['posy'][i], griddata['posz'][i], griddata['rho'][i])
            griddata['rho'][i] += density_hole
            if griddata['cellYe'][i] < 0.4:
                griddata['cellYe'][i] = 0.4
            # print("Inner empty cells filled")
            print(cellid, griddata['posx'][i], griddata['posy'][i], griddata['posz'][i], griddata['rho'][i])

    return griddata
