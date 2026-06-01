import itertools
from pathlib import Path

import numpy as np
import polars as pl

import artistools as at


def make_downscaled_3d_grid(
    modelpath: str | Path, outputgridsize: int = 50, outputfolder: Path | str | None = None
) -> Path:
    """Get a 3D model with smallgrid^3 cells from a 3D model with grid^3 cells.

    Should be same as downscale_3d_grid.pro.
    """
    modelpath = Path(modelpath)

    pldfmodel, modelmeta = at.get_modeldata(modelpath)
    dfmodel = pldfmodel.collect().to_pandas(use_pyarrow_extension_array=True)
    dfelemabund = (
        at.inputmodel
        .get_initelemabundances(modelpath=modelpath)
        .with_columns(pl.col("inputcellid").sub(1).alias("modelgridindex"))
        .collect()
        .to_pandas(use_pyarrow_extension_array=True)
        .set_index("modelgridindex")
    )

    inputgridsize = modelmeta["ncoordgridx"]
    grid = int(inputgridsize)

    assert inputgridsize % outputgridsize == 0
    smallgrid = outputgridsize

    merge = grid / smallgrid
    merge = int(merge)

    outputfolder = Path(modelpath, f"downscale_{outputgridsize}^3") if outputfolder is None else Path(outputfolder)
    outputfolder.mkdir(exist_ok=True)
    smallmodelfile = outputfolder / "model.txt"
    smallabundancefile = outputfolder / "abundances.txt"

    abundcols = [x for x in dfmodel.columns if x.startswith("X_")]
    nabundcols = len(abundcols)
    rho = np.zeros((grid, grid, grid))
    radioabunds = np.zeros((grid, grid, grid, nabundcols))

    max_atomic_number = len([col for col in dfelemabund.columns if col.startswith("X_")])
    assert max_atomic_number == 30
    abund = np.zeros((grid, grid, grid, max_atomic_number + 1))

    print("reading abundance file")

    cellindex = 0
    for cellindex, (z, y, x) in enumerate(itertools.product(range(grid), range(grid), range(grid))):
        abund[x, y, z] = dfelemabund.iloc[cellindex].to_numpy()

    print("reading model file")
    t_model_days = modelmeta["t_model_init_days"]
    vmax = modelmeta["vmax_cmps"]

    for cellindex, (z, y, x) in enumerate(itertools.product(range(grid), range(grid), range(grid))):
        rho[x, y, z] = dfmodel.iloc[cellindex]["rho"]
        radioabunds[x, y, z, :] = dfmodel.iloc[cellindex][abundcols]

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

            for i in range(1, max_atomic_number + 1):  # check this
                abund_small[x, y, z, i] /= rho_small[x, y, z]
            rho_small[x, y, z] /= merge**3

    print("writing abundance file")
    i = 0
    with (modelpath / smallabundancefile).open("w", encoding="utf-8") as newabundancefile:
        for z, y, x in itertools.product(range(smallgrid), range(smallgrid), range(smallgrid)):
            line = abund_small[x, y, z, :][1:31]  # index 1:30 are abundances
            newabundancefile.writelines(f"{i + 1} ")
            newabundancefile.writelines(f"{item:g} " for item in line)
            newabundancefile.writelines("\n")
            i += 1

    print("writing model file")
    xmax = vmax * t_model_days * 3600 * 24
    cellindex = 0
    with (modelpath / smallmodelfile).open("w", encoding="utf-8") as newmodelfile:
        gridsize = smallgrid**3
        newmodelfile.write(f"{gridsize}\n")
        newmodelfile.write(f"{t_model_days}\n")
        newmodelfile.write(f"{vmax}\n")

        for z, y, x in itertools.product(range(smallgrid), range(smallgrid), range(smallgrid)):
            line1 = [
                cellindex + 1,
                -xmax + 2 * x * xmax / smallgrid,
                -xmax + 2 * y * xmax / smallgrid,
                -xmax + 2 * z * xmax / smallgrid,
                rho_small[x, y, z],
            ]
            line2 = radioabunds_small[x, y, z, :]
            newmodelfile.writelines(f"{item:g} " for item in line1)
            newmodelfile.writelines("\n")
            newmodelfile.writelines(f"{item:g} " for item in line2)
            newmodelfile.writelines("\n")
            cellindex += 1

    return outputfolder
