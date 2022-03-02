import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pathlib import Path

import artistools as at
import artistools.estimators.estimators_classic
import artistools.inputmodel.slice1Dfromconein3dmodel
import artistools.initial_composition

from collections import namedtuple

CLIGHT = 2.99792458e10


def read_selected_mgi(modelpath, readonly_mgi, readonly_timestep=False):
    modeldata, _, _ = at.inputmodel.get_modeldata(modelpath)
    estimators = at.estimators.estimators_classic.read_classic_estimators(modelpath, modeldata,
                                                                          readonly_mgi=readonly_mgi,
                                                                          readonly_timestep=readonly_timestep)
    return estimators


def get_modelgridcells_along_axis(modelpath):

    ArgsTuple = namedtuple('args', 'modelpath sliceaxis other_axis1 other_axis2 positive_axis')
    args = ArgsTuple(
        modelpath=modelpath,
        sliceaxis='x',
        other_axis1='z',
        other_axis2='y',
        positive_axis=True
    )

    profile1d = at.inputmodel.slice1Dfromconein3dmodel.get_profile_along_axis(args=args)
    readonly_mgi = get_mgi_of_modeldata(profile1d, modelpath)

    return readonly_mgi


def get_modelgridcells_2D_slice(modeldata, modelpath):
    sliceaxis = 'z'

    slice = at.initial_composition.get_2D_slice_through_3d_model(modeldata, sliceaxis)
    readonly_mgi = get_mgi_of_modeldata(slice, modelpath)

    return readonly_mgi


def get_mgi_of_modeldata(modeldata, modelpath):
    assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
    readonly_mgi = []
    for index, row in modeldata.iterrows():
        if row['rho'] > 0:
            mgi = mgi_of_propcells[int(row['inputcellid'])-1]
            readonly_mgi.append(mgi)
    return readonly_mgi


def plot_Te_vs_time_lineofsight_3d_model(modelpath, modeldata, estimators, readonly_mgi):
    assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
    times = at.get_timestep_times_float(modelpath)

    for mgi in readonly_mgi:
        associated_modeldata_row_for_mgi = modeldata.loc[modeldata['inputcellid'] == assoc_cells[mgi][0]]

        Te = [estimators[(timestep, mgi)]['Te'] for timestep, _ in enumerate(times)]
        plt.scatter(times, Te, label=f'vel={associated_modeldata_row_for_mgi["vel_y_mid"].values[0] / CLIGHT}')

    plt.xlabel('time [days]')
    plt.ylabel('Te [K]')
    plt.xlim(0.15, 10)
    plt.xscale('log')
    plt.legend()
    plt.show()


def plot_Te_vs_velocity(modelpath, modeldata, estimators, readonly_mgi):
    assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
    times = at.get_timestep_times_float(modelpath)
    timesteps = [50, 55, 60, 65, 70, 75, 80, 90]

    for timestep in timesteps:
        Te = [estimators[(timestep, mgi)]['Te'] for mgi in readonly_mgi]

        associated_modeldata_rows = [modeldata.loc[modeldata['inputcellid'] == assoc_cells[mgi][0]] for mgi in readonly_mgi]
        velocity = [row["vel_y_mid"].values[0] / CLIGHT for row in associated_modeldata_rows]

        plt.plot(velocity, Te, label=f'{times[timestep]:.2f}', linestyle='-', marker='o')

    plt.xlabel('velocity/c')
    plt.ylabel('Te [K]')
    plt.yscale('log')
    plt.legend()
    plt.show()


def get_Te_vs_velocity_2D(modelpath, modeldata, vmax, estimators, readonly_mgi, timestep):
    assoc_cells, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
    times = at.get_timestep_times_float(modelpath)
    print([(ts, time) for ts, time in enumerate(times)])
    time = times[timestep]
    print(f'time {time} days')

    ngridcells = len(modeldata['inputcellid'])
    Te = np.zeros(ngridcells)

    for mgi in readonly_mgi:
        Te[assoc_cells[mgi][0]-1] = estimators[(timestep, mgi)]['Te']

    grid = round(len(modeldata['inputcellid']) ** (1./3.))
    grid_Te = np.zeros((grid, grid, grid))  # needs 3D array
    xgrid = np.zeros(grid)

    vmax = vmax / CLIGHT
    i = 0
    for z in range(0, grid):
        for y in range(0, grid):
            for x in range(0, grid):
                grid_Te[x, y, z] = Te[i]
                if modeldata['rho'][i] == 0.:
                    grid_Te[x, y, z] = None
                xgrid[x] = -vmax + 2 * x * vmax / grid
                i += 1

    return grid_Te, xgrid


def make_2d_plot(grid, grid_Te, vmax, modelpath, xgrid, time):
    # PYVISTA
    x, y, z = np.meshgrid(xgrid, xgrid, xgrid)
    mesh = pv.StructuredGrid(x, y, z)
    mesh['Te [K]'] = grid_Te.ravel(order='F')

    sargs = dict(height=0.75, vertical=True, position_x=0.02, position_y=0.1,
                 title_font_size=22, label_font_size=25)

    pv.set_plot_theme("document")  # set white background
    p = pv.Plotter()
    p.set_scale(1.5, 1.5, 1.5)
    single_slice = mesh.slice(normal='z')
    actor = p.add_mesh(single_slice, scalar_bar_args=sargs)  # , clim=[100, 60000]
    actor = p.show_bounds(grid=False, xlabel='vx / c', ylabel='vy / c', zlabel='vz / c',
                          ticks='inside', minor_ticks=False, use_2d=True, font_size=26, bold=False)

    p.camera_position = 'xy'
    p.add_title(f'{time:.1f} days')
    p.show(screenshot=modelpath / f'3Dplot_Te{time:.1f}days_disk.png')


def main():
    modelpath = Path(".")
    modeldata, _, vmax = at.inputmodel.get_modeldata(modelpath)

    # # Get mgi of grid cells along axis for 1D plot
    # # readonly_mgi = get_modelgridcells_along_axis(modelpath)
    readonly_mgi = get_modelgridcells_2D_slice(modeldata, modelpath)
    #
    timestep = 82
    times = at.get_timestep_times_float(modelpath)
    time = times[timestep]
    estimators = read_selected_mgi(modelpath, readonly_mgi=readonly_mgi, readonly_timestep=[timestep])
    grid_Te, xgrid = get_Te_vs_velocity_2D(modelpath, modeldata, vmax, estimators, readonly_mgi, timestep)
    grid = round(len(modeldata['inputcellid']) ** (1./3.))
    make_2d_plot(grid, grid_Te, vmax, modelpath, xgrid, time)


if __name__ == '__main__':
    main()
