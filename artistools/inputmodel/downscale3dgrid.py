"""Resample a 3D ARTIS model onto a coarser Cartesian grid."""

import argparse
import functools
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs

from artistools.constants import day_to_s
from artistools.inputmodel.core import get_model_input_folder
from artistools.inputmodel.core import require_no_ye_file_of_earlier_model
from artistools.inputmodel.core import save_modeldata
from artistools.inputmodel.core import scan_initelemabundances
from artistools.inputmodel.core import scan_modeldata
from artistools.inputmodel.core import with_ye_of_yefile
from artistools.misc import firstexisting_or_none
from artistools.misc import require_cartesian3d_model
from artistools.plottools import save_figure
from artistools.plottools import set_mpl_style


def downscale_cell_sums(values: npt.NDArray[np.float64], merge: int) -> npt.NDArray[np.float64]:
    """Sum an array indexed [x, y, z, ...] over each block of merge^3 cells."""
    nx, ny, nz = (n // merge for n in values.shape[:3])
    blocks = values.reshape((nx, merge, ny, merge, nz, merge, *values.shape[3:]))
    return blocks.sum(axis=(1, 3, 5))


def downscale_mass_fractions(
    massfracs: npt.NDArray[np.float64], rho: npt.NDArray[np.float64], merge: int
) -> npt.NDArray[np.float64]:
    """Return the density-weighted mean of the mass fractions [x, y, z, i] over each block of merge^3 cells."""
    mass_small = downscale_cell_sums(massfracs * rho[..., np.newaxis], merge)
    rho_small = downscale_cell_sums(rho, merge)[..., np.newaxis]
    return np.divide(mass_small, rho_small, out=np.zeros_like(mass_small), where=rho_small > 0)


def downscaled_columns(
    dfsource: pl.DataFrame,
    cols: Sequence[str],
    grid: int,
    reduce: t.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> list[pl.Series]:
    """Return one downscaled Series for each named column, with the dtype of the source column."""
    if not cols:
        return []

    # the flat cell lists vary x fastest, so a Fortran-order reshape gives arrays indexed [x, y, z, i]
    values = dfsource.select(cols).to_numpy().astype(np.float64).reshape((grid, grid, grid, -1), order="F")
    values_small = reduce(values)

    return [
        pl.Series(col, values_small[:, :, :, i].ravel(order="F")).cast(dfsource.schema[col])
        for i, col in enumerate(cols)
    ]


def get_downscale_outputfolder(modelpath: Path | str, outputgridsize: int, outputfolder: Path | str | None) -> Path:
    """Return the output folder of a downscaled model, which is a subfolder of the model if the caller gives none."""
    return Path(modelpath, f"downscale_{outputgridsize}^3") if outputfolder is None else Path(outputfolder)


def make_downscaled_3d_grid(
    modelpath: str | Path,
    outputgridsize: int = 50,
    plot: bool = False,
    outputfolder: Path | str | None = None,
    args: argparse.Namespace | None = None,
    formatversion: int | None = None,
) -> Path:
    """Get a 3D model with smallgrid^3 cells from a 3D model with grid^3 cells.

    Should be same as downscale_3d_grid.pro. args gives the options of the diagnostic plot, e.g. --darkmode.
    formatversion is the format version of the small model.txt, and None keeps the version of the model.

    A small model of format version 2 gets the electron fractions of the Ye.txt of the model in its column Ye,
    because sn3d takes them for the model. A small model of version 1 gets the values of the column Ye of model.txt,
    and the function stops for a Ye.txt in the output folder, because that file is of a different model.
    """
    modelpath = Path(modelpath)

    pldfmodel, modelmeta = scan_modeldata(modelpath)
    require_cartesian3d_model(modelmeta, "The merge of cells to a smaller grid needs", "Give a 3D Cartesian model")
    if formatversion is None:
        formatversion = int(modelmeta.get("formatversion", 1))
    outputfolder = get_downscale_outputfolder(modelpath, outputgridsize, outputfolder)
    if formatversion == 1:
        require_no_ye_file_of_earlier_model(outputfolder, modelpath)
    sourcefolder = get_model_input_folder(modelpath)
    yefilepath = firstexisting_or_none("Ye.txt", folder=sourcefolder, tryzipped=True, search_subfolders=False)
    if yefilepath is not None and formatversion == 2:
        print("  Version 2 has no Ye.txt.")
        pldfmodel = with_ye_of_yefile(pldfmodel.collect(), yefilepath).lazy()
    # save_modeldata also writes Ye, q, and tracercount, thus the downscaled model must keep them
    modelcolnames = pldfmodel.collect_schema().names()
    massweightedcols = [col for col in ("Ye", "q") if col in modelcolnames]
    summedcols = [col for col in ("tracercount",) if col in modelcolnames]
    dfmodel = pldfmodel.select("rho", cs.starts_with("X_"), *massweightedcols, *summedcols).collect()
    dfelemabund = scan_initelemabundances(modelpath=sourcefolder).collect()

    grid = int(modelmeta["ncoordgridx"])
    smallgrid = outputgridsize

    assert grid % smallgrid == 0
    merge = grid // smallgrid

    outputfolder.mkdir(exist_ok=True)
    smallmodelfile = outputfolder / "model.txt"

    abundcols = [x for x in dfmodel.columns if x.startswith("X_")]
    elemcolnames = [col for col in dfelemabund.columns if col.startswith("X_")]

    print("Reading model file")
    t_model_days = modelmeta["t_model_init_days"]
    vmax = modelmeta["vmax_cmps"]

    # the flat cell lists vary x fastest, so a Fortran-order reshape gives an array indexed [x, y, z]
    rho = dfmodel["rho"].to_numpy().astype(np.float64).reshape((grid, grid, grid), order="F")
    rho_small = downscale_cell_sums(rho, merge) / merge**3

    massweightedmean = functools.partial(downscale_mass_fractions, rho=rho, merge=merge)
    blocksum = functools.partial(downscale_cell_sums, merge=merge)

    # Ye and q are per unit mass, thus each block takes the mean over its mass. tracercount counts the
    # trajectories of a cell, thus each block takes the sum
    modelcols_small = [
        *downscaled_columns(dfmodel, abundcols, grid, massweightedmean),
        *downscaled_columns(dfmodel, massweightedcols, grid, massweightedmean),
        *downscaled_columns(dfmodel, summedcols, grid, blocksum),
    ]

    # the cell order of an ARTIS 3D file varies x fastest, which is the Fortran order of the arrays above
    xmax = vmax * t_model_days * day_to_s
    axispos = -xmax + 2 * xmax * np.arange(smallgrid) / smallgrid
    # the cell ids of format version 2 start at 0
    firstinputcellid = 0 if formatversion == 2 else 1
    inputcellid = pl.Series("inputcellid", range(firstinputcellid, smallgrid**3 + firstinputcellid), dtype=pl.Int32)

    dfmodel_small = pl.DataFrame({
        "inputcellid": inputcellid,
        "pos_x_min": np.tile(axispos, smallgrid**2),
        "pos_y_min": np.tile(np.repeat(axispos, smallgrid), smallgrid),
        "pos_z_min": np.repeat(axispos, smallgrid**2),
        "rho": rho_small.ravel(order="F"),
    }).with_columns(modelcols_small)

    dfelemabund_small = pl.DataFrame({"inputcellid": inputcellid}).with_columns(
        downscaled_columns(dfelemabund, elemcolnames, grid, massweightedmean)
    )

    modelmeta_small = modelmeta | {
        "formatversion": formatversion,
        "npts_model": smallgrid**3,
        "ncoordgridx": smallgrid,
        "ncoordgridy": smallgrid,
        "ncoordgridz": smallgrid,
        "vmax_cmps": vmax,
    }

    # a version 1 model also gets abundances.txt, and a version 2 model gets otherstable columns in model.txt
    print("Writing model file")
    save_modeldata(dfmodel_small, outpath=smallmodelfile, modelmeta=modelmeta_small, dfelabundances=dfelemabund_small)

    if plot:
        print("Making diagnostic plot")
        # no ModuleNotFoundError fallback here: artistools.plottools imports matplotlib at module scope, so
        # this module cannot be imported at all without it
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        set_mpl_style()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.8 * 1.5, 4.8))
        assert isinstance(axes, np.ndarray)
        (ax1, ax2) = axes

        middle_ind = int(rho.shape[0] / 2)
        im1 = ax1.imshow(rho[middle_ind, :, :])
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax1)
        ax1.set_xlabel("Cell index")
        ax1.set_ylabel("Cell index")
        ax1.set_title("Original resolution")
        cbar1.set_label(r"$\rho$ (g/cm$^3$)")

        middle_ind_small = int(rho_small.shape[0] / 2)
        im2 = ax2.imshow(rho_small[middle_ind_small, :, :])
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(im2, cax=cax2)
        ax2.set_xlabel("Cell index")
        ax2.set_ylabel("Cell index")
        ax2.set_title("Downscaled resolution")
        cbar2.set_label(r"$\rho$ (g/cm$^3$)")

        fig.tight_layout()

        diagnosticpath = outputfolder / "downscaled_density_diagnostic.png"
        save_figure(fig, diagnosticpath, dpi=300, args=args)

    return outputfolder
