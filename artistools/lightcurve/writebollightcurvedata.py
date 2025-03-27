import typing as t
from pathlib import Path

import numpy as np
import pandas as pd

import artistools as at


def get_bol_lc_from_spec(modelpath: Path) -> pd.DataFrame:
    from scipy import integrate

    res_specdata = at.spectra.read_spec_res(modelpath)
    timearray = res_specdata[0].columns[1:]
    times = [time for time in timearray if 5 < float(time) < 80]
    lightcurvedata: dict[str, t.Any] = {"time": times}
    Mpc_to_cm = 3.085677581491367e24
    for angle in range(len(res_specdata)):
        bol_luminosity = []
        for timestep, timestr in enumerate(timearray):
            if 5 < float(timestr) < 80:
                spectrum = at.spectra.get_spectrum(
                    modelpath=modelpath, directionbins=[angle], timestepmin=timestep, timestepmax=timestep
                )[angle]
                integrated_flux = integrate.trapezoid(spectrum["f_lambda"], spectrum["lambda_angstroms"])
                integrated_luminosity = integrated_flux * 4 * np.pi * Mpc_to_cm**2
                bol_luminosity.append(integrated_luminosity)

        lightcurvedata[f"angle={angle}"] = np.log10(bol_luminosity)

    lightcurvedataframe = pd.DataFrame(lightcurvedata)
    lightcurvedataframe = lightcurvedataframe.replace([np.inf, -np.inf], 0)
    print(lightcurvedataframe)

    return lightcurvedataframe


def get_bol_lc_from_lightcurveout(modelpath: Path, res: bool = False) -> pd.DataFrame:
    lcfilename = "light_curve_res.out" if res else "light_curve.out"
    lcdata = pd.read_csv(modelpath / lcfilename, sep=r"\s+", header=None, names=["time", "lum", "lum_cmf"])
    lcdataframes = {
        dirbin: pldf.to_pandas(use_pyarrow_extension_array=True)
        for dirbin, pldf in at.split_multitable_dataframe(lcdata).items()
    }

    lightcurvedata = {"time": np.array(lcdataframes[0]["time"])}

    nangles = len(lcdataframes) if res else 1
    for angle in range(nangles):
        lcdata = lcdataframes[angle]
        bol_luminosity = np.array(lcdata["lum"]) * 3.826e33  # Luminosity in erg/s

        # lightcurvedata[f'angle={angle}'] = np.log10(bol_luminosity)
        columnname = "lum (erg/s)"
        if res:
            columnname = f"angle={angle}"
        lightcurvedata[columnname] = bol_luminosity

    lightcurvedataframe = pd.DataFrame(lightcurvedata)
    return lightcurvedataframe.replace([np.inf, -np.inf], 0)


def main() -> None:
    # modelnames = ['M08_03', 'M08_05', 'M08_10', 'M09_03', 'M09_05', 'M09_10',
    #               'M10_02_end55', 'M10_03', 'M10_05', 'M10_10', 'M11_05_1']
    modelnames = ["M2a"]

    for modelname in modelnames:
        # modelpath = Path("/Users/ccollins/harddrive4TB/parameterstudy") / Path(modelname)
        modelpath = Path("/Users/ccollins/harddrive4TB/Gronow2020") / Path(modelname)
        outfilepath = Path("/Users/ccollins/Desktop/bollightcurvedata")

        # lightcurvedataframe = get_bol_lc_from_spec(modelpath)
        lightcurvedataframe = get_bol_lc_from_lightcurveout(modelpath)

        lightcurvedataframe.to_csv(
            outfilepath / f"bol_lightcurvedata_{modelname}.txt", sep=" ", index=False, header=False
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
