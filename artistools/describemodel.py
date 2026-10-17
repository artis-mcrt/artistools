"""Summarise an ARTIS model: its folders, cells, timesteps, output files, ions, ARTIS version, and physics options.

The summary reads the model file, the names of the output files, the start of the first log of each run, and
artisoptions.h. It reads no other large file. Run it first to choose the readers and the timesteps for a question about
a model, and to know which physics the run had on.
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
from artistools.misc import get_artis_options
from artistools.misc import get_model_folder
from artistools.misc import get_run_subfolders
from artistools.misc import get_runfolders
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import print_product
from artistools.misc import zopen
from artistools.misc.modelinfo import get_rank_outputfiles
from artistools.misc.remote import on_model_host
from artistools.packets import count_packets_textfiles
from artistools.packets import has_packets_files

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

# the options of artisoptions.h that change the physics of a run or the meaning of an output, by topic. ARTIS renamed
# some options, thus each entry gives the names of one option, the newest name first
PHYSICSOPTIONS: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    (
        "NLTE",
        (
            ("ION_NLEVELS_EXCITED_NLTE", "LEVEL_IS_NLTE"),
            ("FORCE_SAHA_ION_BALANCE",),
            ("NLTE_TIME_DEPENDENT_FIRST_TIMESTEP",),
            ("NLTE_TE_NNE_MAXITER", "NLTEITER"),
        ),
    ),
    ("non-thermal", (("NT_SCHEME", "NT_ON"), ("NT_SOLVE_SPENCERFANO",), ("NT_EXCITATION_ON",))),
    (
        "thermalisation",
        (("PARTICLE_THERMALISATION_SCHEME", "INSTANT_PARTICLE_DEPOSITION"), ("GAMMA_THERMALISATION_SCHEME",)),
    ),
    (
        "radiation field",
        (
            ("MULTIBIN_RADFIELD_MODEL_ON",),
            ("MULTIBIN_RADFIELD_FIRST_TIMESTEP", "FIRST_BINNED_RADFIELD_TIMESTEP", "FIRST_NLTE_RADFIELD_TIMESTEP"),
            ("LTEPOP_EXCITATION_USE_TJ",),
        ),
    ),
    (
        "estimators",
        (
            ("DETAILED_BF_ESTIMATORS_ON",),
            ("DETAILED_LINE_ESTIMATORS_ON",),
            ("USE_LUT_PHOTOION",),
            ("WRITE_ION_HEATING_COOLING_RATES",),
        ),
    ),
    ("temperature limits", (("MINTEMP",), ("MAXTEMP",))),
    ("packets", (("MPKTS",), ("POL_ON",), ("VPKT_ON",), ("KEEP_ESCAPED_GAMMAS",), ("SAMPLE_RPKT_EMISSION",))),
)


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

    packetcount = count_packets_textfiles(modelpath)
    if packetcount == 0 and has_packets_files(modelpath):
        lines.append("packets: no text files, and the parquet cache of the packets")
    else:
        lines.append(f"packets: {packetcount} rank files")

    for filename in SINGLEOUTPUTFILES:
        # the readers of the light curves and the spectra also search the run folders, and get_deposition does not
        filepath = firstexisting_or_none(filename, folder=modelpath, search_subfolders=filename != "deposition.out")
        lines.append(f"{filename}: {filepath.relative_to(modelpath) if filepath is not None else 'none'}")
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


def get_option_lines(modelpath: Path, alloptions: bool = False) -> list[str]:
    """Return the physics options of artisoptions.h by topic, or each option of the file with alloptions."""
    options = get_artis_options(modelpath)
    if not options:
        return ["artisoptions.h: none in the model folder or in its folder artis"]
    if alloptions:
        return ["artisoptions.h:", *(f"  {name} = {value}" for name, value in options.items())]

    lines = ["artisoptions.h (--alloptions gives each option):"]
    for topic, optionnames in PHYSICSOPTIONS:
        found = [
            f"{name} = {options[name]}"
            for names in optionnames
            if (name := next((name for name in names if name in options), None)) is not None
        ]
        if found:
            lines.append(f"  {topic}: {'; '.join(found)}")
    return lines


def get_version_lines(modelpath: Path) -> list[str]:
    """Return the ARTIS version and the git branch of each run, from the start of its first log.

    A restarted run can use a different build of ARTIS, thus each run folder gives its own version.
    """
    lines: list[str] = []
    for folder in get_run_subfolders(modelpath):
        logpath = firstexisting_or_none("output_0-0.txt", folder=folder, search_subfolders=False)
        if logpath is None:
            continue
        fields: dict[str, str] = {}
        with zopen(logpath) as flog:
            # ARTIS writes the git lines near the start of the log, thus the search reads only the first lines
            for _, line in zip(range(500), flog, strict=False):
                for key in ("git version", "git branch"):
                    if f"{key}" in line and key not in fields:
                        fields[key] = line.split(key, 1)[1].lstrip(": ").strip()
        if fields:
            foldername = "." if folder == modelpath else folder.name
            lines.append(
                f"ARTIS of run folder {foldername}: {fields.get('git version', 'unknown version')}"
                f" (branch {fields.get('git branch', 'unknown')})"
            )
    return lines


@on_model_host
def get_model_summary(modelpath: Path, alloptions: bool = False) -> list[str]:
    """Return the lines of the summary. A part whose input file is missing gives one line that says so.

    The host of a remote model makes the summary, because each part reads a file of the model folder.
    """
    lines = [f"model folder: {modelpath.resolve()}"]
    parts = (get_model_lines, get_timestep_lines, get_output_lines, get_ion_lines, get_version_lines)
    for part in parts:
        try:
            lines.extend(part(modelpath))
        except FileNotFoundError as exc:
            lines.append(f"{part.__name__.removeprefix('get_').removesuffix('_lines')}: {exc}")
    lines.extend(get_option_lines(modelpath, alloptions=alloptions))
    return lines


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path())
    parser.add_argument(
        "--alloptions", action="store_true", help="Give each option of artisoptions.h, and not only the physics options"
    )


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Summarise an ARTIS model: its folders, cells, timesteps, output files, ions, ARTIS version, and options."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    modelpath = get_model_folder(args.modelpath)
    print_product(args, "\n".join(get_model_summary(modelpath, alloptions=args.alloptions)))
