"""Write out ARTIS spectra for each timestep to individual text files."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import confirm_overwrite_matches
from artistools.misc import exit_with_error
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_model_name
from artistools.misc import get_timestep_times
from artistools.misc import normalize_path_list
from artistools.misc import parse_cli_args
from artistools.misc import print_saved
from artistools.misc import print_warning
from artistools.misc import split_remote_path
from artistools.spectra.core import scan_spectra


def write_spectrum(dfspectrum: pl.DataFrame, outfilepath: Path) -> None:
    """Write one spectrum between 1500 and 60000 Angstroms as a two-column text file."""
    dfspectrum = dfspectrum.filter(pl.col("lambda_angstroms").is_between(1500, 60000))
    with outfilepath.open("w", encoding="utf-8") as spec_file:
        spec_file.write("#lambda f_lambda_1Mpc\n")
        spec_file.write("#[A] [erg/s/cm2/A]\n")

        dfspectrum.select("lambda_angstroms", "f_lambda").write_csv(spec_file, separator=" ", include_header=False)

    print_saved(outfilepath)


def get_spectra_outdirectory(modelpath: Path, outdirectory: Path | None) -> Path:
    """Return outdirectory, or the spectra folder of the model when the caller gives none."""
    if outdirectory is not None:
        return outdirectory
    # a folder below a remote path would make a local folder of the same name, and that folder would then hide
    # the remote model. Thus the files of a remote model go to the working folder. The host and the whole path
    # give each model its own folder, e.g. spectra/vae26/home/scratch/mymodel
    if (remote := split_remote_path(modelpath)) is not None:
        host, hostpath = remote
        return Path("spectra", host, *("home" if part == "~" else part for part in hostpath.parts if part != "/"))
    return Path(modelpath, "spectra")


def write_flambda_spectra(modelpath: Path, outdirectory: Path) -> None:
    """Write the spectrum of each timestep of the valid time range to a text file in outdirectory.

    Each file holds lambda_angstroms and f_lambda, and its name gives the timestep and the time in days. A model
    with direction bins also gets one file of the first polar angle bin for each timestep.
    """
    outdirectory.mkdir(parents=True, exist_ok=True)

    tmids = get_timestep_times(modelpath, loc="mid")

    tslast, tmin_d_valid, tmax_d_valid = get_escaped_arrivalrange(modelpath)

    if tmin_d_valid is None or tmax_d_valid is None:
        exit_with_error(f"{modelpath} has no time range in which the spectra are valid, e.g. the run stopped early")
    timesteps = [ts for ts in range(tslast + 1) if tmids[ts] >= tmin_d_valid and tmids[ts] <= tmax_d_valid]

    # scan_spectra raises an error for a model with no angle-averaged spectrum. A model with no direction bins gives
    # no spectrum of the polar angle bin 0
    for fileprefix, average_over_phi, dirbin in (("spectrum", False, -1), ("spectrum_polar00", True, 0)):
        lzspectra_of_timestep = [
            scan_spectra(
                modelpath=modelpath,
                timestepmin=timestep,
                timestepmax=timestep,
                average_over_phi=average_over_phi,
                directionbins=[dirbin],
            )
            for timestep in timesteps
        ]
        if not lzspectra_of_timestep or dirbin not in lzspectra_of_timestep[0]:
            continue
        # one collect_all call evaluates the queries of all the timesteps together
        dfspectra = pl.collect_all([lzspectra[dirbin] for lzspectra in lzspectra_of_timestep])
        for timestep, dfspectrum in zip(timesteps, dfspectra, strict=True):
            write_spectrum(
                dfspectrum, outfilepath=outdirectory / f"{fileprefix}_ts{timestep:02.0f}_{tmids[timestep]:.2f}d.txt"
            )


def write_flambda_spectra_of_models(modelpaths: Sequence[Path], outdirectory: Path | None, *, force: bool) -> None:
    """Write the spectra of write_flambda_spectra for each model.

    The file names hold no model name, thus each of several models gets a subfolder of outdirectory with its name.
    Two models can have the same name, thus the second one gets its index after the name, e.g. mymodel_1. The default
    folder of a model is its spectra folder.
    """
    subfoldernames: list[str] = []
    for modelindex, modelpath in enumerate(modelpaths):
        modelname = get_model_name(modelpath)
        # the second model would replace the files of the first, and the check of the overwrite saw no file. A name
        # with the index can also be the name of a model, e.g. mymodel_2, thus the index grows until the name is new
        suffixindex = modelindex
        while modelname in subfoldernames:
            modelname = f"{get_model_name(modelpath)}_{suffixindex}"
            suffixindex += 1
        if modelname != get_model_name(modelpath) and outdirectory is not None and len(modelpaths) > 1:
            print_warning(f"{modelpath} has the name of an earlier model, thus its folder is {modelname}")
        subfoldernames.append(modelname)
    modeloutdirectories = [
        get_spectra_outdirectory(
            modelpath,
            outdirectory / subfoldername if outdirectory is not None and len(modelpaths) > 1 else outdirectory,
        )
        for modelpath, subfoldername in zip(modelpaths, subfoldernames, strict=True)
    ]
    # the check of all the folders comes before the first model, thus a refusal leaves no partial output. Each file
    # name holds a timestep, and the valid timesteps come from the data, thus the check takes patterns
    for modeloutdirectory in modeloutdirectories:
        confirm_overwrite_matches(modeloutdirectory, ("spectrum_ts*.txt", "spectrum_polar00_ts*.txt"), force=force)

    for modelpath, modeloutdirectory in zip(modelpaths, modeloutdirectories, strict=True):
        write_flambda_spectra(modelpath, modeloutdirectory)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, multiplepaths=True, default=[], helptext="Paths of the ARTIS models")
    addarg_output(
        parser, kind="folder", helptext="The folder for the spectrum files (default: the spectra folder of the model)"
    )
    addarg_force(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Write ARTIS spectra for each timestep to individual text files."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    write_flambda_spectra_of_models(
        normalize_path_list(args.modelpath), Path(args.outputfile) if args.outputfile else None, force=args.force
    )
