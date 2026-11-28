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
from artistools.inputmodel.core import get_gridcontributions_folder
from artistools.inputmodel.core import remap_gridcontributions
from artistools.inputmodel.core import require_no_ye_file_of_earlier_model
from artistools.inputmodel.core import save_modeldata
from artistools.inputmodel.core import scan_initelemabundances
from artistools.inputmodel.core import scan_modeldata
from artistools.inputmodel.core import with_ye_that_sn3d_reads
from artistools.inputmodel.rprocess_from_trajectory import delete_gridparticlecontributions
from artistools.inputmodel.rprocess_from_trajectory import get_gridparticlecontributions_or_none
from artistools.inputmodel.rprocess_from_trajectory import save_gridparticlecontributions
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


def get_downscale_cell_map(
    dfmodel: pl.DataFrame, grid: int, merge: int, finecellvolume: float, firstinputcellid: int
) -> pl.DataFrame:
    """Return the inputcellid and mass_g of each fine cell, and the out_inputcellid and out_mass_g of its coarse cell.

    remap_gridcontributions reads this table. The rows of dfmodel are the fine cells, and x varies fastest. Each fine
    cell has one row, thus a join on inputcellid keeps the row count of the particle contributions.
    """
    smallgrid = grid // merge
    finecellindex = pl.int_range(pl.len(), dtype=pl.Int64)
    coarsecellindex = (
        (finecellindex % grid) // merge
        + smallgrid * ((finecellindex // grid) % grid // merge)
        + smallgrid**2 * (finecellindex // grid**2 // merge)
    )
    return dfmodel.select(
        pl.col("inputcellid").cast(pl.Int32),
        mass_g=pl.col("rho").cast(pl.Float64) * finecellvolume,
        out_inputcellid=(coarsecellindex + firstinputcellid).cast(pl.Int32),
    ).with_columns(out_mass_g=pl.col("mass_g").sum().over("out_inputcellid"))


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

    The small model gets the electron fractions of the Ye.txt of the model in its column Ye, because sn3d takes them
    for the model and Ye.txt has the cells of the model. For a small model of version 1, the function stops for a
    Ye.txt in the output folder, because that file is of a different model.

    The small model gets the gridcontributions.txt of the model with the coarse cells. A model with no
    gridcontributions.txt gives a small model with no such file. Thus the function deletes the file of an earlier
    model in the output folder.
    """
    modelpath = Path(modelpath)

    pldfmodel, modelmeta = scan_modeldata(modelpath)
    require_cartesian3d_model(modelmeta, "The merge of cells to a smaller grid needs", "Give a 3D Cartesian model")
    if formatversion is None:
        formatversion = int(modelmeta.get("formatversion", 1))
    outputfolder = get_downscale_outputfolder(modelpath, outputgridsize, outputfolder)
    if formatversion == 1:
        require_no_ye_file_of_earlier_model(outputfolder, modelpath)
    # Ye.txt has the cells of the source model, thus the small model gets its values in the column Ye of model.txt
    pldfmodel = with_ye_that_sn3d_reads(pldfmodel, modelpath, modelmeta)
    # save_modeldata also writes Ye, q, and tracercount, thus the downscaled model must keep them
    modelcolnames = pldfmodel.collect_schema().names()
    massweightedcols = [col for col in ("Ye", "q") if col in modelcolnames]
    summedcols = [col for col in ("tracercount",) if col in modelcolnames]
    dfmodel = pldfmodel.select("inputcellid", "rho", cs.starts_with("X_"), *massweightedcols, *summedcols).collect()
    dfelemabund = scan_initelemabundances(modelpath=modelpath).collect()
    dfgridcontributions = get_gridparticlecontributions_or_none(get_gridcontributions_folder(modelpath))

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

    dfgridcontributions_small = None
    if dfgridcontributions is not None:
        print(
            "  gridcontributions.txt: the mass fraction of a particle in a coarse cell is the mean of its mass"
            " fractions in the fine cells. The weight of each fine cell is its mass."
        )
        dfcellmap = get_downscale_cell_map(dfmodel, grid, merge, (2 * xmax / grid) ** 3, firstinputcellid)
        dfgridcontributions_small = remap_gridcontributions(dfgridcontributions, dfcellmap).collect()

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
    # the values of the Ye.txt of the source are in the Ye column, and the prompt names the files of an earlier model
    save_modeldata(
        dfmodel_small,
        outpath=smallmodelfile,
        modelmeta=modelmeta_small,
        dfelabundances=dfelemabund_small,
        deletecellidfiles=True,
    )
    if dfgridcontributions_small is not None:
        save_gridparticlecontributions(dfgridcontributions_small, outputfolder / "gridcontributions.txt")
    else:
        delete_gridparticlecontributions(outputfolder)

    if plot:
        print("Making diagnostic plot")
        # no ModuleNotFoundError fallback here: artistools.plottools imports matplotlib at module scope, so
        # this module cannot be imported at all without it
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        set_mpl_style()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.8 * 1.5, 4.8))
        assert isinstance(axes, np.ndarray)

        for ax, rhogrid, title in zip(
            axes, (rho, rho_small), ("Original resolution", "Downscaled resolution"), strict=True
        ):
            im = ax.imshow(rhogrid[int(rhogrid.shape[0] / 2), :, :])
            cbar = fig.colorbar(im, cax=make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05))
            ax.set_xlabel("Cell index")
            ax.set_ylabel("Cell index")
            ax.set_title(title)
            cbar.set_label(r"$\rho$ (g/cm$^3$)")

        fig.tight_layout()

        diagnosticpath = outputfolder / "downscaled_density_diagnostic.png"
        save_figure(fig, diagnosticpath, dpi=300, args=args)

    return outputfolder
