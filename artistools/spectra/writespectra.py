"""Write out ARTIS spectra for each timestep to individual text files."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.misc import addarg_modelpath
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import print_saved
from artistools.spectra.spectra import get_spectra


def write_spectrum(dfspectrum: pl.DataFrame, outfilepath: Path) -> None:
    """Write one spectrum between 1500 and 60000 Angstroms as a two-column text file."""
    dfspectrum = dfspectrum.filter(pl.col("lambda_angstroms").is_between(1500, 60000))
    with outfilepath.open("w", encoding="utf-8") as spec_file:
        spec_file.write("#lambda f_lambda_1Mpc\n")
        spec_file.write("#[A] [erg/s/cm2/A]\n")

        dfspectrum.select("lambda_angstroms", "f_lambda").write_csv(spec_file, separator=" ", include_header=False)

    print_saved(outfilepath)


def write_flambda_spectra(modelpath: Path) -> None:
    """Write out spectra to text files.

    Writes lambda_angstroms and f_lambda to .txt files for all timesteps and create
    a text file containing the time in days for each timestep.
    """
    outdirectory = Path(modelpath, "spectra")

    outdirectory.mkdir(parents=True, exist_ok=True)

    tmids = get_timestep_times(modelpath, loc="mid")

    tslast, tmin_d_valid, tmax_d_valid = get_escaped_arrivalrange(modelpath)

    assert tmin_d_valid is not None
    assert tmax_d_valid is not None
    timesteps = [ts for ts in range(tslast + 1) if tmids[ts] >= tmin_d_valid and tmids[ts] <= tmax_d_valid]

    # one collect_all call evaluates the queries of all the timesteps together
    dfspectra_alltimesteps = pl.collect_all([
        get_spectra(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep)[-1] for timestep in timesteps
    ])
    for timestep, dfspectrum in zip(timesteps, dfspectra_alltimesteps, strict=True):
        write_spectrum(dfspectrum, outfilepath=outdirectory / f"spectrum_ts{timestep:02.0f}_{tmids[timestep]:.2f}d.txt")

    lzspectra_polar = [
        get_spectra(modelpath=modelpath, timestepmin=timestep, timestepmax=timestep, average_over_phi=True)
        for timestep in timesteps
    ]
    if lzspectra_polar and 0 in lzspectra_polar[0]:
        dfspectra_polar = pl.collect_all([dirbin_spectra[0] for dirbin_spectra in lzspectra_polar])
        for timestep, dfspectrum in zip(timesteps, dfspectra_polar, strict=True):
            write_spectrum(
                dfspectrum, outfilepath=outdirectory / f"spectrum_polar00_ts{timestep:02.0f}_{tmids[timestep]:.2f}d.txt"
            )


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write ARTIS spectra for each timestep to individual text files."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    write_flambda_spectra(args.modelpath)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
