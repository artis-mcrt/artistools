from pathlib import Path
import pandas as pd
from astropy import units as u
import numpy as np

MSUN = 1.989e33
CLIGHT = 2.99792458e10


def read_arepo_out(pathtosnapshot):
    arepo_out_file = Path(pathtosnapshot) / "ejectasnapshot.dat"

    column_names = ['write(1744,*)h (i)', 'x (i)', 'y (i)', 'z (i)', 'vx (i)', 'vy (i)', 'vz (i)', 'vstx (i)',
                    'vsty (i)',
                    'vstz (i)', 'u (i)', 'psi(i)', 'alpha(i)', 'pmass (i)', 'rho (i)', 'p(i)', 'rst(i)',
                    'tau (i)', 'av (i)', 'ye(i)', 'temp(i)']
    # Everything is in geometric units here
    arepo_out_dat = pd.read_csv(arepo_out_file, delim_whitespace=True, header=None, names=column_names)
    print("total mass", sum(arepo_out_dat['pmass (i)']))

    return arepo_out_dat


def read_griddat_file(pathtogriddata):
    griddatfilepath = Path(pathtogriddata) / "grid.dat"
    simulation_end_time_geomunits = 5813.694  # todo: figure out how to reads this in
    print("WARNING: Simulation end time needs changed")

    gridindex, posx, posy, posz, rho, cellYe = np.loadtxt(griddatfilepath, skiprows=3, unpack=True)
    cellYe = np.nan_to_num(cellYe, nan=0.)

    griddata = {}

    griddata['gridindex'] = np.array([int(i) for i in gridindex])

    factor_position = 1.478  # in km
    griddata['posx'] = (posx * factor_position) * (u.km).to(u.cm)
    griddata['posy'] = (posy * factor_position) * (u.km).to(u.cm)
    griddata['posz'] = (posz * factor_position) * (u.km).to(u.cm)

    griddata['rho'] = rho * 6.176e17  # g/cm3

    griddata['cellYe'] = cellYe

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

    return griddata, t_model, vmax


def fill_central_hole(griddata, t_model):
    print(griddata)

    # Just (2021) Fig. 16 top left panel
    vel_hole = [0, 0.02, 0.05, 0.07, 0.09, 0.095, 0.1]
    mass_hole = [3e-4, 3e-4, 2e-4, 1e-4, 2e-5, 1e-5, 1e-9]
    mass_intergrated = np.trapz(y=mass_hole, x=vel_hole)  # Msun

    v_outer_hole = 0.1 * CLIGHT  # cm/s
    pos_outer_hole = v_outer_hole * t_model * (24. * 3600)  # cm
    vol_hole = 4 / 3 * np.pi * pos_outer_hole ** 3  # cm^3
    density_hole = (mass_intergrated * MSUN) / vol_hole  # g / cm^3
    print(density_hole)

    for i, cellid in enumerate(griddata['gridindex']):
        # if pos < 0.1 c
        if ((np.sqrt(griddata['posx'][i] ** 2 + griddata['posy'][i] ** 2 + griddata['posz'][i] ** 2)) /
                (t_model * (24. * 3600)) / CLIGHT) < 0.1:
            if griddata['rho'][i] == 0:
                print("Inner empty cells")
                print(cellid, griddata['posx'][i], griddata['posy'][i], griddata['posz'][i], griddata['rho'][i])
                griddata['rho'][i] = density_hole
                griddata['cellYe'] = 0.4
                print("Inner empty cells filled")
                print(cellid, griddata['posx'][i], griddata['posy'][i], griddata['posz'][i], griddata['rho'][i])

    return griddata
