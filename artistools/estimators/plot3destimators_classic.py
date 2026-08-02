"""Plot a 2D slice of the electron temperature from a classic-mode 3D model."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import artistools as at
from artistools.constants import C_cm_per_s as CLIGHT


def read_selected_mgi(
    modelpath: Path | str, readonly_mgi: list[int] | None = None, readonly_timestep: list[int] | None = None
) -> dict[tuple[int, int], t.Any] | None:
    return at.estimators.estimators_classic.read_classic_estimators(
        Path(modelpath), readonly_mgi=readonly_mgi, readonly_timestep=readonly_timestep
    )


def get_modelgridcells_2D_slice(modeldata: pd.DataFrame, modelpath: Path | str) -> list[int]:
    sliceaxis: t.Literal["x", "y", "z"] = "x"

    slicedata = at.inputmodel.plotinitialcomposition.get_2D_slice_through_3d_model(
        dfmodel=modeldata, sliceaxis=sliceaxis
    )
    return get_mgi_of_modeldata(slicedata, modelpath)


def get_mgi_of_modeldata(modeldata: pd.DataFrame, modelpath: Path | str) -> list[int]:
    mgi_of_propcells = at.get_grid_mapping(modelpath=modelpath)[1]
    return [mgi_of_propcells[int(row["inputcellid"]) - 1] for _index, row in modeldata.iterrows() if row["rho"] > 0]


def get_Te_vs_velocity_2D(
    modelpath: Path | str,
    modeldata: pd.DataFrame,
    vmax: float,
    estimators: dict[tuple[int, int], t.Any],
    readonly_mgi: list[int],
    timestep: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    assoc_cells = at.get_grid_mapping(modelpath=modelpath)[0]
    times = at.get_timestep_times(modelpath)
    print(list(enumerate(times)))
    time = times[timestep]
    print(f"time {time} days")

    ngridcells = len(modeldata["inputcellid"])
    Te = np.zeros(ngridcells)

    for mgi in readonly_mgi:
        Te[assoc_cells[mgi][0] - 1] = estimators[timestep, mgi]["Te"]

    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    vmax /= CLIGHT
    # cells are ordered with x varying fastest, i.e. Fortran order on (x, y, z)
    rho = np.asarray(modeldata["rho"], dtype=float)
    grid_Te = np.where(rho == 0.0, np.nan, Te).reshape((grid, grid, grid), order="F")
    xgrid = -vmax + 2 * np.arange(grid) * vmax / grid

    return grid_Te, xgrid


def make_2d_plot(
    grid: int,
    grid_Te: npt.NDArray[np.floating],
    vmax: float,
    modelpath: Path | str,
    xgrid: npt.NDArray[np.floating],
    time: float,
) -> None:
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
        pv.set_plot_theme("document")  # type: ignore[no-untyped-call]
        p: t.Any = pv.Plotter()
        p.set_scale(p, xscale=1.5, yscale=1.5, zscale=1.5)
        single_slice = mesh.slice(normal="z")
        p.add_mesh(single_slice, scalar_bar_args=sargs)
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
        p.show(screenshot=Path(modelpath, f"3Dplot_Te{time:.1f}days_disk.png"))

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

        fig, axis = plt.subplots()
        im = axis.imshow(data, extent=extent)
        cbar = fig.colorbar(im)
        cbar.set_label("Te [K]", rotation=90)
        axis.set_xlabel("vy / c")
        axis.set_ylabel("vz / c")
        axis.set_xlim(-vmax, vmax)
        axis.set_ylim(-vmax, vmax)
        outfilename = "plotestim.pdf"
        fig.savefig(outfilename, format="pdf")
        at.print_saved(outfilename)
        plt.close(fig)


def addargs(parser: argparse.ArgumentParser) -> None:
    at.add_modelpath_arg(parser, default=Path())
    at.add_timestep_arg(parser, kind="int", default=82)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot a 2D slice of the electron temperature from a classic-mode 3D model."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    modelpath = Path(args.modelpath)
    plmodeldata, modelmeta = at.inputmodel.get_modeldata(modelpath)
    vmax = modelmeta["vmax_cmps"]
    modeldata = plmodeldata.collect().to_pandas(use_pyarrow_extension_array=True)

    readonly_mgi = get_modelgridcells_2D_slice(modeldata, modelpath)
    timestep = int(args.timestep)
    times = at.get_timestep_times(modelpath)
    if not -len(times) <= timestep < len(times):
        msg = f"timestep {timestep} is out of range: this model has {len(times)} timesteps (0-{len(times) - 1})"
        raise IndexError(msg)
    time = times[timestep]
    estimators = read_selected_mgi(modelpath, readonly_mgi=readonly_mgi, readonly_timestep=[timestep])
    assert estimators is not None
    grid_Te, xgrid = get_Te_vs_velocity_2D(modelpath, modeldata, vmax, estimators, readonly_mgi, timestep)
    grid = round(len(modeldata["inputcellid"]) ** (1.0 / 3.0))
    make_2d_plot(grid, grid_Te, vmax, modelpath, xgrid, time)


if __name__ == "__main__":
    main()
