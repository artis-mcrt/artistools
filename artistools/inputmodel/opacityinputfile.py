from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def all_cells_same_opacity(modelpath, ngrid):
    cell_opacities = np.array([0.1] * ngrid)

    with Path(modelpath, "opacity.txt").open("w", encoding="utf-8") as fopacity:
        fopacity.write(f"{ngrid}\n")

        for cellid, opacity in enumerate(cell_opacities):
            fopacity.write(f"{cellid + 1}    {opacity}\n")


def opacity_by_Ye(outputfilepath, griddata):
    """Opacities from Table 1 Tanaka 2020."""
    griddata = pd.DataFrame(griddata)
    print("Getting opacity kappa from Ye")

    cell_opacities = np.zeros_like(griddata["cellYe"])

    for index, Ye in enumerate(griddata["cellYe"]):
        if Ye == 0.0 and griddata["rho"][index] == 0:
            cell_opacities[index] = 0.0
        elif Ye <= 0.1:
            cell_opacities[index] = 19.5
        elif Ye <= 0.15:
            cell_opacities[index] = 32.2
        elif Ye <= 0.2:
            cell_opacities[index] = 22.3
        elif Ye <= 0.25:
            cell_opacities[index] = 5.6
        elif Ye <= 0.3:
            cell_opacities[index] = 5.36
        elif Ye <= 0.35:
            cell_opacities[index] = 3.3
        else:
            cell_opacities[index] = 0.96

    griddata["opacity"] = cell_opacities

    with Path(outputfilepath, "opacity.txt").open("w", encoding="utf-8") as fopacity:
        fopacity.write(f'{len(griddata["inputcellid"])}\n')
        griddata[["inputcellid", "opacity"]].to_csv(fopacity, sep="\t", index=False, header=False, float_format="%.10f")


def get_opacity_from_file(modelpath):
    opacity_file_contents = np.loadtxt(Path(modelpath) / "opacity.txt", unpack=True, skiprows=1)
    return opacity_file_contents[1]


def write_Ye_file(outputfilepath: Path | str, griddata: pd.DataFrame | pl.DataFrame) -> None:
    if isinstance(griddata, pd.DataFrame):
        griddata = pl.from_pandas(griddata)

    assert griddata["inputcellid"].dtype in pl.INTEGER_DTYPES

    with Path(outputfilepath, "Ye.txt").open("w", encoding="utf-8") as fYe:
        fYe.write(f'{len(griddata["inputcellid"])}\n')
        griddata.to_pandas()[["inputcellid", "cellYe"]].to_csv(
            fYe, sep="\t", index=False, header=False, float_format="%.10f", na_rep="0.0"
        )

    print("Saved Ye.txt")
