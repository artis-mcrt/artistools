"""Write bolometric light curve data out as plain text files, one per model."""

import argparse
import os
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from artistools.lightcurve.core import find_lightcurve_file
from artistools.lightcurve.core import get_bolometric_luminosities
from artistools.lightcurve.core import scan_lightcurve
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import get_model_name
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_saved
from artistools.spectra import scan_spec_res


def get_bol_lc_from_spec(modelpath: Path) -> pl.DataFrame:
    """Return log10(bolometric luminosity) per direction bin at each timestep, integrated from the spectra."""
    res_specdata = scan_spec_res(modelpath)
    timearray = res_specdata[0].collect_schema().names()[1:]
    # the light_curve.out mode writes every time, thus this mode does too. A fixed range of 5 to 80 days
    # dropped every time of a kilonova below 5 days and gave an empty file for a nebular model
    lightcurvedata: dict[str, t.Any] = {"time": timearray}
    luminosities = get_bolometric_luminosities(modelpath, range(len(timearray)), dirbins=range(len(res_specdata)))
    for angle, angleluminosities in luminosities.items():
        lightcurvedata[f"angle={angle}"] = np.log10(angleluminosities)

    # a direction bin with no luminosity has no log10, and a zero there means one erg/s. Thus the
    # value of such a bin is a null, which the writer gives as nan
    lightcurvedataframe = pl.DataFrame(lightcurvedata).with_columns(cs.float().replace([np.inf, -np.inf], None))
    print(lightcurvedataframe)

    return lightcurvedataframe


def get_bol_lc_from_lightcurveout(modelpath: Path) -> pl.DataFrame:
    """Return the spherically averaged bolometric luminosity against time, read from light_curve.out."""
    # scan_lightcurve keys the spherically averaged light curve as dirbin -1, and light_curve.out has no other bins
    lcdata = scan_lightcurve(find_lightcurve_file(modelpath))[-1].collect()

    lightcurvedata = {"time": lcdata["time_days"], "lum (erg/s)": lcdata["luminosity_erg/s"]}

    return pl.DataFrame(lightcurvedata)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[])
    parser.add_argument(
        "--fromspectra",
        action="store_true",
        help="Integrate the direction-resolved spectra instead of reading light_curve.out",
    )
    addarg_output(parser, kind="folder", default=Path())
    addarg_force(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write bolometric light curve data out as a plain text file for each model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    header = (
        "# 1st col is time in days. Next columns are log10(luminosity) for each model viewing angle."
        " A viewing angle with no luminosity has no value, thus its column holds nan"
        if args.fromspectra
        else "# 1st col is time in days, 2nd col is the spherically averaged bolometric luminosity in erg/s"
    )

    modelpaths = normalize_path_list(args.modelpath)
    modelnames = [get_model_name(modelpath) for modelpath in modelpaths]
    # the two sources hold different columns, thus each one writes a file of its own
    suffix = "_fromspectra" if args.fromspectra else ""
    filenames = [f"bol_lightcurvedata_{modelname}{suffix}.txt" for modelname in modelnames]
    # two models with the same name have the same file, thus a later model replaces the file of an earlier one.
    # a name can hold a path, and some file systems ignore the letter case, thus "a//b" and "A/b" can be one file
    filekeys = [os.path.normpath(filename).casefold() for filename in filenames]
    samefilepaths = [
        str(modelpath) for modelpath, filekey in zip(modelpaths, filekeys, strict=True) if filekeys.count(filekey) > 1
    ]
    if samefilepaths:
        exit_with_error(
            f"These model paths have the output file name of a different model path: {', '.join(samefilepaths)}",
            "Give each model a different name in its plotlabel.txt, or write each model to a different output folder",
        )

    outputpath = Path(args.outputfile)
    outputpath.mkdir(parents=True, exist_ok=True)
    confirm_overwrite([outputpath], filenames, force=args.force)

    for modelpath, filename in zip(modelpaths, filenames, strict=True):
        lightcurvedataframe = (
            get_bol_lc_from_spec(modelpath) if args.fromspectra else get_bol_lc_from_lightcurveout(modelpath)
        )

        outfilepath = outputpath / filename
        with outfilepath.open("w", encoding="utf-8") as f:
            f.write(f"{header}\n")
            lightcurvedataframe.write_csv(f, separator=" ", include_header=False, null_value="nan")

        print_saved(outfilepath)
