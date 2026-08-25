"""Resample a 3D ARTIS model onto a coarser Cartesian grid."""

import itertools
from pathlib import Path

import numpy as np
import polars as pl

import artistools as at
from artistools.constants import day_to_s
from artistools.inputmodel.inputmodel_misc import save_initelemabundances
from artistools.inputmodel.inputmodel_misc import save_modeldata
from artistools.plottools import save_figure


def make_downscaled_3d_grid(
    modelpath: str | Path, outputgridsize: int = 50, plot: bool = False, outputfolder: Path | str | None = None
) -> Path:
    """Get a 3D model with smallgrid^3 cells from a 3D model with grid^3 cells.

    Should be same as downscale_3d_grid.pro.
    """
    modelpath = Path(modelpath)

    pldfmodel, modelmeta = at.get_modeldata(modelpath)
    dfmodel = pldfmodel.collect()
    dfelemabund = at.inputmodel.get_initelemabundances(modelpath=modelpath).collect()

    grid = int(modelmeta["ncoordgridx"])
    smallgrid = outputgridsize

    assert grid % smallgrid == 0
    merge = grid // smallgrid

    outputfolder = Path(modelpath, f"downscale_{outputgridsize}^3") if outputfolder is None else Path(outputfolder)
    outputfolder.mkdir(exist_ok=True)
    smallmodelfile = outputfolder / "model.txt"
    smallabundancefile = outputfolder / "abundances.txt"

    abundcols = [x for x in dfmodel.columns if x.startswith("X_")]
    nabundcols = len(abundcols)
    elemcolnames = [col for col in dfelemabund.columns if col.startswith("X_")]
    max_atomic_number = len(elemcolnames)

    print("reading abundance file")

    # the flat cell lists vary x fastest, so a Fortran-order reshape gives arrays indexed [x, y, z]
    abund = dfelemabund.to_numpy().reshape((grid, grid, grid, max_atomic_number + 1), order="F")

    print("reading model file")
    t_model_days = modelmeta["t_model_init_days"]
    vmax = modelmeta["vmax_cmps"]

    rho = dfmodel["rho"].to_numpy().reshape((grid, grid, grid), order="F")
    radioabunds = dfmodel.select(abundcols).to_numpy().reshape((grid, grid, grid, nabundcols), order="F")

    rho_small = np.zeros((smallgrid, smallgrid, smallgrid))
    radioabunds_small = np.zeros((smallgrid, smallgrid, smallgrid, nabundcols))
    abund_small = np.zeros((smallgrid, smallgrid, smallgrid, max_atomic_number + 1))

    for z, y, x, zz, yy, xx in itertools.product(
        range(smallgrid), range(smallgrid), range(smallgrid), range(merge), range(merge), range(merge)
    ):
        rho_small[x, y, z] += rho[x * merge + xx, y * merge + yy, z * merge + zz]
        for i in range(nabundcols):
            radioabunds_small[x, y, z, i] += (
                radioabunds[x * merge + xx, y * merge + yy, z * merge + zz, i]
                * rho[x * merge + xx, y * merge + yy, z * merge + zz]
            )

        abund_small[x, y, z, :] += (
            abund[x * merge + xx, y * merge + yy, z * merge + zz] * rho[x * merge + xx, y * merge + yy, z * merge + zz]
        )

    for z, y, x in itertools.product(range(smallgrid), range(smallgrid), range(smallgrid)):
        if rho_small[x, y, z] > 0:
            radioabunds_small[x, y, z, :] /= rho_small[x, y, z]

            for i in range(1, max_atomic_number + 1):
                abund_small[x, y, z, i] /= rho_small[x, y, z]
            rho_small[x, y, z] /= merge**3

    # the cell order of an ARTIS 3D file varies x fastest, which is the Fortran order of the arrays above
    xmax = vmax * t_model_days * day_to_s
    axispos = -xmax + 2 * xmax * np.arange(smallgrid) / smallgrid
    inputcellid = pl.Series("inputcellid", range(1, smallgrid**3 + 1), dtype=pl.Int32)

    dfmodel_small = pl.DataFrame({
        "inputcellid": inputcellid,
        "pos_x_min": np.tile(axispos, smallgrid**2),
        "pos_y_min": np.tile(np.repeat(axispos, smallgrid), smallgrid),
        "pos_z_min": np.repeat(axispos, smallgrid**2),
        "rho": rho_small.ravel(order="F"),
    }).with_columns([
        pl.Series(abundcol, radioabunds_small[:, :, :, i].ravel(order="F")) for i, abundcol in enumerate(abundcols)
    ])

    dfelemabund_small = pl.DataFrame({"inputcellid": inputcellid}).with_columns([
        pl.Series(elemcol, abund_small[:, :, :, i + 1].ravel(order="F")) for i, elemcol in enumerate(elemcolnames)
    ])

    modelmeta_small = modelmeta | {
        "npts_model": smallgrid**3,
        "ncoordgridx": smallgrid,
        "ncoordgridy": smallgrid,
        "ncoordgridz": smallgrid,
        "vmax_cmps": vmax,
    }

    print("writing model file")
    save_modeldata(dfmodel_small, outpath=smallmodelfile, modelmeta=modelmeta_small)

    print("writing abundance file")
    save_initelemabundances(dfelemabund_small, outpath=smallabundancefile)

    if plot:
        print("making diagnostic plot")
        # no ModuleNotFoundError fallback here: artistools.plottools imports matplotlib at module scope, so
        # this module cannot be imported at all without it
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        save_figure(fig, diagnosticpath, dpi=300, bbox_inches="tight")

    return outputfolder
