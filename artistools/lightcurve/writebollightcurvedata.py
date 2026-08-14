"""Write bolometric light curve data out as plain text files, one per model."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

import artistools as at


def get_bol_lc_from_spec(modelpath: Path) -> pl.DataFrame:
    """Return log10(bolometric luminosity) per direction bin between 5 and 80 days, integrated from the spectra."""
    res_specdata = at.spectra.read_spec_res(modelpath)
    timearray = res_specdata[0].columns[1:]
    times = [time for time in timearray if 5 < float(time) < 80]
    lightcurvedata: dict[str, t.Any] = {"time": times}
    Mpc_to_cm = at.constants.megaparsec_to_cm
    for angle in range(len(res_specdata)):
        bol_luminosity = []
        for timestep, timestr in enumerate(timearray):
            if 5 < float(timestr) < 80:
                spectrum = at.spectra.get_spectra(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep)[
                    angle
                ].collect()
                integrated_flux = np.trapezoid(spectrum["f_lambda"], spectrum["lambda_angstroms"])
                integrated_luminosity = integrated_flux * 4 * np.pi * Mpc_to_cm**2
                bol_luminosity.append(integrated_luminosity)

        lightcurvedata[f"angle={angle}"] = np.log10(bol_luminosity)

    lightcurvedataframe = pl.DataFrame(lightcurvedata).with_columns(cs.float().replace([np.inf, -np.inf], 0.0))
    print(lightcurvedataframe)

    return lightcurvedataframe


def get_bol_lc_from_lightcurveout(modelpath: Path) -> pl.DataFrame:
    """Return the spherically averaged bolometric luminosity against time, read from light_curve.out."""
    lcdataframes = {
        dirbin: df.collect() for dirbin, df in at.lightcurve.readfile(modelpath / "light_curve.out").items()
    }

    lightcurvedata = {
        "time": lcdataframes[next(iter(lcdataframes.keys()))]["time_days"],
        "lum (erg/s)": lcdataframes[-1]["luminosity_erg/s"],
    }

    return pl.DataFrame(lightcurvedata).with_columns(cs.float().replace([np.inf, -np.inf], 0.0))


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    at.add_modelpath_arg(parser, positional=True, multiplepaths=True, default=[Path()])
    parser.add_argument(
        "--fromspectra",
        action="store_true",
        help="Integrate the direction-resolved spectra instead of reading light_curve.out",
    )
    at.add_outputpath_arg(parser, default=Path(), astype=Path)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write bolometric light curve data out as a plain text file for each model."""
    args = at.parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    header = (
        "# 1st col is time in days. Next columns are log10(luminosity) for each model viewing angle"
        if args.fromspectra
        else "# 1st col is time in days, 2nd col is the spherically averaged bolometric luminosity in erg/s"
    )

    for modelpath in at.normalize_path_list(args.modelpath):
        modelname = at.get_model_name(modelpath)
        lightcurvedataframe = (
            get_bol_lc_from_spec(modelpath) if args.fromspectra else get_bol_lc_from_lightcurveout(modelpath)
        )

        outfilepath = Path(args.outputpath) / f"bol_lightcurvedata_{modelname}.txt"
        with outfilepath.open("w", encoding="utf-8") as f:
            f.write(f"{header}\n")
            lightcurvedataframe.write_csv(f, separator=" ", include_header=False)

        at.print_saved(outfilepath)


if __name__ == "__main__":
    main()
