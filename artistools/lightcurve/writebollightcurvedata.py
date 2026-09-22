"""Write bolometric light curve data out as plain text files, one per model."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from artistools.lightcurve.core import find_lightcurve_file
from artistools.lightcurve.core import get_bolometric_luminosities
from artistools.lightcurve.core import scan_lightcurve
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import get_model_name
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_saved
from artistools.spectra import read_spec_res


def get_bol_lc_from_spec(modelpath: Path) -> pl.DataFrame:
    """Return log10(bolometric luminosity) per direction bin between 5 and 80 days, integrated from the spectra."""
    res_specdata = read_spec_res(modelpath)
    timearray = res_specdata[0].collect_schema().names()[1:]
    # one pass gives both the time labels and the timesteps they came from, so the two cannot drift apart
    selected = [(ts, timestr) for ts, timestr in enumerate(timearray) if 5 < float(timestr) < 80]
    lightcurvedata: dict[str, t.Any] = {"time": [timestr for _, timestr in selected]}
    timesteps = [ts for ts, _ in selected]
    luminosities = get_bolometric_luminosities(modelpath, timesteps, dirbins=range(len(res_specdata)))
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
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[Path()])
    parser.add_argument(
        "--fromspectra",
        action="store_true",
        help="Integrate the direction-resolved spectra instead of reading light_curve.out",
    )
    addarg_output(parser, kind="folder", default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write bolometric light curve data out as a plain text file for each model."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    header = (
        "# 1st col is time in days. Next columns are log10(luminosity) for each model viewing angle."
        " A viewing angle with no luminosity has no value, thus its column holds nan"
        if args.fromspectra
        else "# 1st col is time in days, 2nd col is the spherically averaged bolometric luminosity in erg/s"
    )

    outputpath = Path(args.outputfile)
    outputpath.mkdir(parents=True, exist_ok=True)

    for modelpath in normalize_path_list(args.modelpath):
        modelname = get_model_name(modelpath)
        lightcurvedataframe = (
            get_bol_lc_from_spec(modelpath) if args.fromspectra else get_bol_lc_from_lightcurveout(modelpath)
        )

        # the two sources hold different columns, thus each one writes a file of its own
        suffix = "_fromspectra" if args.fromspectra else ""
        outfilepath = outputpath / f"bol_lightcurvedata_{modelname}{suffix}.txt"
        with outfilepath.open("w", encoding="utf-8") as f:
            f.write(f"{header}\n")
            lightcurvedataframe.write_csv(f, separator=" ", include_header=False, null_value="nan")

        print_saved(outfilepath)
