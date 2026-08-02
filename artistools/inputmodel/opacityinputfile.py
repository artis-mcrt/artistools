from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def opacity_by_Ye(outputfilepath: Path | str, griddata: pd.DataFrame | pl.DataFrame) -> None:
    """Opacities from Table 1 Tanaka 2020."""
    if isinstance(griddata, pl.DataFrame):
        griddata = griddata.to_pandas()
    griddata = pd.DataFrame(griddata)
    print("Getting opacity kappa from Ye")

    cell_opacities = np.zeros(len(griddata), dtype=float)

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

    griddata.loc[:, "opacity"] = cell_opacities

    with Path(outputfilepath, "opacity.txt").open("w", encoding="utf-8") as fopacity:
        fopacity.write(f"{len(griddata['inputcellid'])}\n")
        griddata[["inputcellid", "opacity"]].to_csv(fopacity, sep="\t", index=False, header=False, float_format="%.10f")


def write_Ye_file(outputfilepath: Path | str, griddata: pl.DataFrame) -> None:
    assert griddata.schema["inputcellid"].is_integer()

    with Path(outputfilepath, "Ye.txt").open("w", encoding="utf-8") as fYe:
        fYe.write(f"{len(griddata['inputcellid'])}\n")
        griddata.to_pandas(use_pyarrow_extension_array=True)[["inputcellid", "cellYe"]].to_csv(
            fYe, sep="\t", index=False, header=False, float_format="%.10f", na_rep="0.0"
        )

    print("Saved Ye.txt")
