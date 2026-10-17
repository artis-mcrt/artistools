"""Summarise an ARTIS model: its folders, cells, timesteps, output files, ions, and main options.

The summary reads the model file and the names of the output files, and no other large file. Run it first to choose
the readers and the timesteps for a question about a model.
"""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.atomic import get_composition_data
from artistools.atomic import get_ionstring
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.misc import addarg_modelpath
from artistools.misc import firstexisting_or_none
from artistools.misc import get_artis_option
from artistools.misc import get_model_folder
from artistools.misc import get_runfolders
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import print_product
from artistools.misc.modelinfo import get_rank_outputfiles

# the output files that are not rank files, in the order of the summary. A reader also accepts each one with a
# compressed extension
SINGLEOUTPUTFILES = (
    "light_curve.out",
    "light_curve_res.out",
    "gamma_light_curve.out",
    "spec.out",
    "spec_res.out",
    "gamma_spec.out",
    "deposition.out",
)

# the options of artisoptions.h that change the meaning of an estimator value, see the artis-data skill
SUMMARYOPTIONS = ("MINTEMP", "MAXTEMP")


def get_model_lines(modelpath: Path) -> list[str]:
    """Return the lines that describe the cells of model.txt."""
    dfmodel, modelmeta = get_modeldata(modelpath, printwarningsonly=True)
    cellswithmatter = (
        add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta).select((pl.col("rho") > 0).sum()).collect().item()
    )
    return [
        f"model.txt: {modelmeta['dimensions']}D, {modelmeta['npts_model']} cells, {cellswithmatter} cells with matter",
        f"  t_model_init_days {modelmeta['t_model_init_days']:g}, vmax {modelmeta['vmax_cmps']:.4g} cm/s",
    ]


def get_timestep_lines(modelpath: Path) -> list[str]:
    """Return the line that gives the number of timesteps and the days that they cover."""
    tstarts = get_timestep_times(modelpath, loc="start")
    tends = get_timestep_times(modelpath, loc="end")
    return [
        f"timesteps: {len(tstarts)}, from {tstarts[0]:.3f} to {tends[-1]:.3f} days (artistools timesteps lists them)"
    ]


def get_output_lines(modelpath: Path) -> list[str]:
    """Return the lines that name the output files of the model and of its run folders."""
    runfolders = get_runfolders(modelpath)
    runfoldernames = [str(folder.relative_to(modelpath)) if folder != modelpath else "." for folder in runfolders]
    lines = [f"run folders with estimators: {', '.join(runfoldernames) or 'none'}"]
    for kind, filenameformat in (("estimators", "estimators_{mpirank:04d}.out"), ("nlte", "nlte_{mpirank:04d}.out")):
        rankcount = sum(len(get_rank_outputfiles(folder, filenameformat)) for folder in runfolders)
        lines.append(f"{kind}: {rankcount} rank files")

    packetranks = set(get_rank_outputfiles(modelpath, "packets00_{mpirank:04d}.out"))
    if (modelpath / "packets").is_dir():
        packetranks |= set(get_rank_outputfiles(modelpath / "packets", "packets00_{mpirank:04d}.out"))
    lines.append(f"packets: {len(packetranks)} rank files")

    for filename in SINGLEOUTPUTFILES:
        filepath = firstexisting_or_none(filename, folder=modelpath, search_subfolders=False)
        lines.append(f"{filename}: {filepath.name if filepath is not None else 'none'}")
    return lines


def get_ion_lines(modelpath: Path) -> list[str]:
    """Return one line for each element of compositiondata.txt, with its lowest and its top ion."""
    dfcomposition = get_composition_data(modelpath)
    lines = ["ions (ARTIS cannot ionise an element above its top ion):"]
    lines.extend(
        f"  {get_ionstring(z, lowest)} to {get_ionstring(z, top)}"
        for z, lowest, top in dfcomposition.select("Z", "lowermost_ion_stage", "uppermost_ion_stage").iter_rows()
    )
    return lines


def get_option_lines(modelpath: Path) -> list[str]:
    """Return one line for each main option of artisoptions.h, or a line that says that no file gives it."""
    return [
        f"artisoptions.h {name}: {get_artis_option(modelpath, name) or 'not found in artis/artisoptions.h'}"
        for name in SUMMARYOPTIONS
    ]


def get_model_summary(modelpath: Path) -> list[str]:
    """Return the lines of the summary. A part whose input file is missing gives one line that says so."""
    lines = [f"model folder: {modelpath.resolve()}"]
    parts = (get_model_lines, get_timestep_lines, get_output_lines, get_ion_lines, get_option_lines)
    for part in parts:
        try:
            lines.extend(part(modelpath))
        except FileNotFoundError as exc:
            lines.append(f"{part.__name__.removeprefix('get_').removesuffix('_lines')}: {exc}")
    return lines


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Summarise an ARTIS model: its folders, cells, timesteps, output files, ions, and main options."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelpath = get_model_folder(args.modelpath)
    print_product(args, "\n".join(get_model_summary(modelpath)))
