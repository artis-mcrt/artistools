"""Write out ARTIS spectra for each timestep to individual text files."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
import polars as pl

import artistools as at


def write_spectrum(dfspectrum: pl.DataFrame, outfilepath: Path) -> None:
    dfspectrum = dfspectrum.filter(pl.col("lambda_angstroms").is_between(1500, 60000))
    with outfilepath.open("w", encoding="utf-8") as spec_file:
        spec_file.write("#lambda f_lambda_1Mpc\n")
        spec_file.write("#[A] [erg/s/cm2/A]\n")

        dfspectrum.to_pandas(use_pyarrow_extension_array=True).to_csv(
            spec_file, header=False, sep=" ", index=False, columns=["lambda_angstroms", "f_lambda"]
        )

    print(f"Saved {outfilepath}")


def write_flambda_spectra(modelpath: Path) -> None:
    """Write out spectra to text files.

    Writes lambda_angstroms and f_lambda to .txt files for all timesteps and create
    a text file containing the time in days for each timestep.
    """
    outdirectory = Path(modelpath, "spectra")

    outdirectory.mkdir(parents=True, exist_ok=True)

    tmids = at.get_timestep_times(modelpath, loc="mid")

    tslast, tmin_d_valid, tmax_d_valid = at.get_escaped_arrivalrange(modelpath)

    assert tmin_d_valid is not None
    assert tmax_d_valid is not None
    timesteps = [ts for ts in range(tslast + 1) if tmids[ts] >= tmin_d_valid and tmids[ts] <= tmax_d_valid]

    for timestep in timesteps:
        dfspectrum = at.spectra.get_spectrum(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep)[-1]

        write_spectrum(dfspectrum, outfilepath=outdirectory / f"spectrum_ts{timestep:02.0f}_{tmids[timestep]:.2f}d.txt")

    for timestep in timesteps:
        dfspectra = at.spectra.get_spectrum(
            modelpath=modelpath, timestepmin=timestep, timestepmax=timestep, average_over_phi=True, directionbins=[0]
        )
        if 0 in dfspectra:
            write_spectrum(
                dfspectra[0],
                outfilepath=outdirectory / f"spectrum_polar00_ts{timestep:02.0f}_{tmids[timestep]:.2f}d.txt",
            )
        else:
            break


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-modelpath", type=Path, default=Path(), help="Path to ARTIS folder")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Plot spectra from ARTIS and reference data."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)
        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    write_flambda_spectra(args.modelpath)


if __name__ == "__main__":
    main()
