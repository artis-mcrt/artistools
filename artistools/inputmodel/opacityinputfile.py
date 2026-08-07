"""Write and inspect the grey opacity.txt and Ye.txt input files for an ARTIS model."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

import artistools as at


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
    """Write the per-cell electron fraction to Ye.txt in the given folder."""
    assert griddata.schema["inputcellid"].is_integer()

    with Path(outputfilepath, "Ye.txt").open("w", encoding="utf-8") as fYe:
        fYe.write(f"{len(griddata['inputcellid'])}\n")
        griddata.to_pandas(use_pyarrow_extension_array=True)[["inputcellid", "cellYe"]].to_csv(
            fYe, sep="\t", index=False, header=False, float_format="%.10f", na_rep="0.0"
        )

    print("Saved Ye.txt")


def all_cells_same_opacity(modelpath: str | Path, ngrid: int, kappa: float = 0.1) -> None:
    """Write an opacity.txt giving every cell the same grey opacity [cm2/g]."""
    with Path(modelpath, "opacity.txt").open("w", encoding="utf-8") as fopacity:
        fopacity.write(f"{ngrid}\n")
        fopacity.writelines(f"{cellid + 1}    {kappa}\n" for cellid in range(ngrid))

    at.print_saved(Path(modelpath, "opacity.txt"))


def get_opacity_from_file(modelpath: Path | str) -> npt.NDArray[np.float64]:
    """Return the per-cell grey opacities [cm2/g] from a model's opacity.txt."""
    # ndmin keeps the (cellid, opacity) columns separate even for a single-cell model
    opacity_file_contents = np.loadtxt(Path(modelpath) / "opacity.txt", unpack=True, skiprows=1, ndmin=2)

    return np.asarray(opacity_file_contents[1], dtype=np.float64)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument(
        "action",
        nargs="?",
        default=None,
        choices=["uniform", "describe"],
        help=(
            "uniform: write an opacity.txt with the same opacity in every cell."
            " describe: report the opacities in an existing opacity.txt."
        ),
    )
    at.add_modelpath_arg(parser, default=Path())
    at.add_outputpath_arg(parser, astype=Path, helptext="Folder to write opacity.txt into (uniform)")
    parser.add_argument("-kappa", type=float, default=0.1, help="Grey opacity for every cell [cm2/g] (uniform)")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write or inspect an ARTIS grey opacity.txt."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    if args.action is None:
        print("ERROR: no action given. Run with --help to see the available actions.")
        raise SystemExit(1)

    modelpath = Path(args.modelpath)

    if args.action == "uniform":
        _, modelmeta = at.inputmodel.get_modeldata(modelpath)
        all_cells_same_opacity(Path(args.outputpath), modelmeta["npts_model"], kappa=args.kappa)
    else:
        opacities = get_opacity_from_file(modelpath)
        print(f"opacity.txt: {len(opacities)} cells")
        print(f"  kappa min {opacities.min():.4g}, max {opacities.max():.4g}, mean {opacities.mean():.4g} cm2/g")


if __name__ == "__main__":
    main()
