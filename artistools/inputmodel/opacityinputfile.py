"""Write and inspect the grey opacity.txt and Ye.txt input files for an ARTIS model."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

import artistools as at
from artistools.misc import addarg_action
from artistools.misc import require_action


def opacity_by_Ye(outputfilepath: Path | str, griddata: pl.DataFrame) -> None:
    """Opacities from Table 1 Tanaka 2020."""
    print("Getting opacity kappa from Ye")

    Ye = pl.col("cellYe")
    griddata = griddata.with_columns(
        opacity=pl
        .when((Ye == 0.0) & (pl.col("rho") == 0))
        .then(0.0)
        .when(Ye <= 0.1)
        .then(19.5)
        .when(Ye <= 0.15)
        .then(32.2)
        .when(Ye <= 0.2)
        .then(22.3)
        .when(Ye <= 0.25)
        .then(5.6)
        .when(Ye <= 0.3)
        .then(5.36)
        .when(Ye <= 0.35)
        .then(3.3)
        .otherwise(0.96)
    )

    with Path(outputfilepath, "opacity.txt").open("w", encoding="utf-8") as fopacity:
        fopacity.write(f"{griddata.height}\n")
        griddata.select("inputcellid", "opacity").write_csv(
            fopacity, separator="\t", include_header=False, float_precision=10
        )


def write_Ye_file(outputfilepath: Path | str, griddata: pl.DataFrame) -> None:
    """Write the per-cell electron fraction to Ye.txt in the given folder."""
    assert griddata.schema["inputcellid"].is_integer()

    with Path(outputfilepath, "Ye.txt").open("w", encoding="utf-8") as fYe:
        fYe.write(f"{griddata.height}\n")
        # ARTIS needs a number in every field, thus write a missing and a NaN electron fraction as zero
        griddata.select("inputcellid", pl.col("cellYe").cast(pl.Float64).fill_null(0.0).fill_nan(0.0)).write_csv(
            fYe, separator="\t", include_header=False, float_precision=10
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
    addarg_action(
        parser,
        choices=["uniform", "describe"],
        helptext=(
            "uniform: write an opacity.txt with the same opacity in every cell."
            " describe: report the opacities in an existing opacity.txt"
        ),
    )
    at.addarg_modelpath(parser, default=Path())
    at.addarg_outputpath(parser, astype=Path, helptext="Folder to write opacity.txt into (uniform)")
    parser.add_argument("-kappa", type=float, default=0.1, help="Grey opacity for every cell [cm2/g] (uniform)")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write or inspect an ARTIS grey opacity.txt."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    require_action(args)

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
