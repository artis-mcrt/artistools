import argparse
import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import artistools as at

CLIGHT = 2.99792458e10


def read_selected_mgi(
    modelpath: Path | str, readonly_mgi: list[int] | None = None, readonly_timestep: list[int] | None = None
) -> dict[tuple[int, int], t.Any] | None:
    modeldata, _ = at.inputmodel.get_modeldata_pandas(modelpath)
    return at.estimators.estimators_classic.read_classic_estimators(
        Path(modelpath), modeldata, readonly_mgi=readonly_mgi, readonly_timestep=readonly_timestep
    )


def get_modelgridcells_along_axis(modelpath: Path | str, args: argparse.Namespace | None = None) -> list[int]:
    if args is None:
        args = argparse.Namespace(
            modelpath=modelpath, sliceaxis="x", other_axis1="z", other_axis2="y", positive_axis=True
        )
    else:
        axes = ["x", "y", "z"]
        axes.remove(args.sliceaxis)
        args.other_axis1 = axes[0]
        args.other_axis2 = axes[1]

    profile1d = at.inputmodel.slice1dfromconein3dmodel.get_profile_along_axis(args=args)
    return get_mgi_of_modeldata(profile1d, modelpath)


def get_modelgridcells_2D_slice(modeldata, modelpath) -> list[int]:
    sliceaxis: t.Literal["x", "y", "z"] = "x"

    slicedata = at.inputmodel.plotinitialcomposition.get_2D_slice_through_3d_model(modeldata, sliceaxis)
    return get_mgi_of_modeldata(slicedata, modelpath)


def get_mgi_of_modeldata(modeldata: pd.DataFrame, modelpath: Path | str) -> list[int]:
    _, mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)
    return [mgi_of_propcells[int(row["inputcellid"]) - 1] for _index, row in modeldata.iterrows() if row["rho"] > 0]


def plot_Te_vs_time_lineofsight_3d_model(modelpath, modeldata, estimators, readonly_mgi) -> None:
    assoc_cells, _ = at.get_grid_mapping(modelpath=modelpath)
    times = at.get_timestep_times(modelpath)

    for mgi in readonly_mgi:
        associated_modeldata_row_for_mgi = modeldata.loc[modeldata["inputcellid"] == assoc_cells[mgi][0]]

        Te = [estimators[timestep, mgi]["Te"] for timestep in range(len(times))]
        plt.scatter(times, Te, label=f"vel={associated_modeldata_row_for_mgi['vel_y_mid'].to_numpy()[0] / CLIGHT}")

    plt.xlabel("time [days]")
    plt.ylabel("Te [K]")
    plt.xlim(0.15, 10)
    plt.xscale("log")
    plt.legend()
    plt.show()


def plot_Te_vs_velocity(modelpath, modeldata, estimators, readonly_mgi) -> None:
    assoc_cells, _ = at.get_grid_mapping(modelpath=modelpath)
    times = at.get_timestep_times(modelpath)
    timesteps = [50, 55, 60, 65, 70, 75, 80, 90]

    for timestep in timesteps:
        Te = [estimators[timestep, mgi]["Te"] for mgi in readonly_mgi]

        associated_modeldata_rows = [
            modeldata.loc[modeldata["inputcellid"] == assoc_cells[mgi][0]] for mgi in readonly_mgi
        ]
        velocity = [row["vel_y_mid"].to_numpy()[0] / CLIGHT for row in associated_modeldata_rows]

        plt.plot(velocity, Te, label=f"{times[timestep]:.2f}", linestyle="-", marker="o")

    plt.xlabel("velocity/c")
    plt.ylabel("Te [K]")
    plt.yscale("log")
    plt.legend()
    plt.show()


def get_Te_vs_velocity_2D(modelpath, modeldata, vmax, estimators, readonly_mgi, timestep):
    assoc_cells, _ = at.get_grid_mapping(modelpath=modelpath)
    times = at.get_timestep_times(modelpath)
    print(list(enumerate(times)))
    time = times[timestep]
    print(f"time {time} days")

    ngridcells = len(modeldata["inputcellid"])
    Te = np.zeros(ngridcells)

    for mgi in readonly_mgi:
        Te[assoc_cells[mgi][0] - 1] = estimators[timestep, mgi]["Te"]

    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    grid_Te = np.zeros((grid, grid, grid))  # needs 3D array
    xgrid = np.zeros(grid)

    vmax /= CLIGHT
    i = 0
    for z in range(grid):
        for y in range(grid):
            for x in range(grid):
                grid_Te[x, y, z] = Te[i]
                if modeldata["rho"][i] == 0.0:
                    grid_Te[x, y, z] = np.nan
                xgrid[x] = -vmax + 2 * x * vmax / grid
                i += 1

    return grid_Te, xgrid


def make_2d_plot(grid, grid_Te, vmax, modelpath, xgrid, time) -> None:
    import pyvista as pv

    pyvista = False
    if pyvista:
        # PYVISTA
        arrx, arry, arrz = np.meshgrid(xgrid, xgrid, xgrid)
        mesh = pv.StructuredGrid(arrx, arry, arrz)
        mesh["Te [K]"] = grid_Te.ravel(order="F")

        sargs = {
            "height": 0.75,
            "vertical": True,
            "position_x": 0.02,
            "position_y": 0.1,
            "title_font_size": 22,
            "label_font_size": 25,
        }

        # set white background
        pv.set_plot_theme("document")
        p = pv.Plotter()
        p.set_scale(p, xscale=1.5, yscale=1.5, zscale=1.5)
        single_slice = mesh.slice(normal="z")
        p.add_mesh(single_slice, scalar_bar_args=sargs)  # , clim=[100, 60000]
        p.show_bounds(
            p,
            grid=False,
            xlabel="vx / c",
            ylabel="vy / c",
            zlabel="vz / c",
            ticks="inside",
            minor_ticks=False,
            use_2d=True,
            font_size=26,
            bold=False,
        )

        p.camera_position = "xy"
        p.add_title(f"{time:.1f} days")
        p.show(screenshot=modelpath / f"3Dplot_Te{time:.1f}days_disk.png")

    imshow = True
    if imshow:
        # imshow
        dextent = {"left": -vmax, "right": vmax, "bottom": vmax, "top": -vmax}
        extent = dextent["left"], dextent["right"], dextent["bottom"], dextent["top"]
        data = np.zeros((grid, grid))

        for z in range(grid):
            for y in range(grid):
                for x in range(grid):
                    # if z == round(grid/2)-1:
                    #     data[x, y] = grid_Te[x, y, z]
                    # if y == round(grid/2)-1:
                    #     data[z, x] = grid_Te[x, y, z]
                    if x == round(grid / 2) - 1:
                        data[z, y] = grid_Te[x, y, z]

        plt.imshow(data, extent=extent)
        cbar = plt.colorbar()
        cbar.set_label("Te [K]", rotation=90)
        # plt.xlabel('vx / c')
        # plt.ylabel('vy / c')
        plt.xlabel("vy / c")
        plt.ylabel("vz / c")
        plt.xlim(-vmax, vmax)
        plt.ylim(-vmax, vmax)
        outfilename = "plotestim.pdf"
        plt.savefig(Path(modelpath) / outfilename, format="pdf")
        print(f"Saved {outfilename}")


def main() -> None:
    modelpath = Path()
    modeldata, _, vmax = at.inputmodel.get_modeldata_tuple(modelpath)

    # # Get mgi of grid cells along axis for 1D plot
    # # readonly_mgi = get_modelgridcells_along_axis(modelpath)
    readonly_mgi = get_modelgridcells_2D_slice(modeldata, modelpath)
    timestep = 82
    times = at.get_timestep_times(modelpath)
    time = times[timestep]
    estimators = read_selected_mgi(modelpath, readonly_mgi=readonly_mgi, readonly_timestep=[timestep])
    grid_Te, xgrid = get_Te_vs_velocity_2D(modelpath, modeldata, vmax, estimators, readonly_mgi, timestep)
    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    make_2d_plot(grid, grid_Te, vmax, modelpath, xgrid, time)


if __name__ == "__main__":
    main()
