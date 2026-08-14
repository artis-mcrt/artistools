"""Write bolometric light curve data out as plain text files, one per model."""

import typing as t
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


def main() -> None:
    """Write a bolometric light curve text file for each model in the hardcoded model list."""
    # modelnames = ['M08_03', 'M08_05', 'M08_10', 'M09_03', 'M09_05', 'M09_10',
    #               'M10_02_end55', 'M10_03', 'M10_05', 'M10_10', 'M11_05_1']
    modelnames = ["M2a"]

    for modelname in modelnames:
        # modelpath = Path("/Users/ccollins/harddrive4TB/parameterstudy") / Path(modelname)
        modelpath = Path("/Users/ccollins/harddrive4TB/Gronow2020") / Path(modelname)
        outfilepath = Path("/Users/ccollins/Desktop/bollightcurvedata")

        # lightcurvedataframe = get_bol_lc_from_spec(modelpath)
        lightcurvedataframe = get_bol_lc_from_lightcurveout(modelpath)

        lightcurvedataframe.write_csv(
            outfilepath / f"bol_lightcurvedata_{modelname}.txt", separator=" ", include_header=False
        )

        with (outfilepath / f"bol_lightcurvedata_{modelname}.txt").open("r+") as f:  # add comment to start of file
            content = f.read()
            f.seek(0, 0)
            f.write(
                "# 1st col is time in days. Next columns are log10(luminosity) for each model viewing angle".rstrip(
                    "\r\n"
                )
                + "\n"
                + content
            )

        print("done")


if __name__ == "__main__":
    main()
