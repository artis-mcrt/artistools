"""Make a table of the ARTIS models below a folder, with one row for each model.

The table gives the cells, the time steps, the main physics options, the ARTIS version, and the peak of the
bolometric light curve of each model. The command reads model.txt, timesteps.out or input.txt, artisoptions.h, the
start of the first log of each run folder, and light_curve.out. It reads no packets, estimators, or spectra, thus a
folder of many models takes a short time.
"""

import argparse
import os
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from artistools.describemodel import get_run_versions
from artistools.describemodel import returns_only
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.lightcurve import find_lightcurve_file
from artistools.lightcurve import scan_lightcurve
from artistools.misc import addarg_table_output
from artistools.misc import exit_with_error
from artistools.misc import folder_is_artis_model
from artistools.misc import get_artis_options
from artistools.misc import get_model_name
from artistools.misc import get_nprocs
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import print_detail
from artistools.misc import print_or_write_table
from artistools.misc import print_product
from artistools.misc import print_warning
from artistools.misc import resolve_table_outputfile
from artistools.misc.fileio import natural_sort_key

CATALOGSCHEMA: t.Final = pl.Schema({
    "modelname": pl.String,
    "modelfolder": pl.String,
    "dimensions": pl.Int32,
    "cells": pl.Int64,
    "cells_with_matter": pl.Int64,
    "timesteps": pl.Int32,
    "tstart_days": pl.Float64,
    "tend_days": pl.Float64,
    "nlte": pl.Boolean,
    "nonthermal": pl.Boolean,
    "thermalisation": pl.String,
    "num_packets": pl.Int64,
    "artis_version": pl.String,
    "peak_lum_erg_per_s": pl.Float64,
    "peak_time_days": pl.Float64,
})

METHODLINES: t.Final = (
    "the method of each column:",
    "  modelfolder: the path of the model folder relative to the given folder. A model folder holds input.txt",
    "  cells_with_matter: the cells of model.txt with a density above zero",
    (
        "  timesteps, tstart_days, tend_days: timesteps.out, or the logarithmic time steps of input.txt with no such"
        " file. The run can stop before the last time step"
    ),
    (
        "  nlte: false if ION_NLEVELS_EXCITED_NLTE (LEVEL_IS_NLTE before June 2025) of artisoptions.h always returns 0"
        " or false, else true"
    ),
    "  nonthermal: false if NT_SCHEME of artisoptions.h is NT_OFF (NT_ON is false before August 2026), else true",
    (
        "  thermalisation: the value of PARTICLE_THERMALISATION_SCHEME of artisoptions.h, or of"
        " INSTANT_PARTICLE_DEPOSITION before June 2024"
    ),
    "  num_packets: NUM_PACKETS of artisoptions.h, or MPKTS times the MPI ranks of input.txt before September 2026",
    "  artis_version: the git version in the first log (output_0-0.txt) of the last run folder",
    (
        "  peak_lum_erg_per_s, peak_time_days: the largest point of light_curve.out, the average over all directions,"
        " in the frame of the observer. There is no interpolation between the time steps"
    ),
    "  nan: the file is missing, or a warning gives the reason",
)


def find_model_folders(folder: Path) -> list[Path]:
    """Return each model folder at or below a folder, in natural order.

    The search does not look below a model folder, because its subfolders are run folders. It follows a link to a
    folder, and it reads each real folder one time, thus a link that makes a loop cannot stop it. It does not read
    a hidden folder, e.g. .git.
    """
    modelfolders: list[Path] = []
    realfolders: set[Path] = set()
    for dirpath, dirnames, _ in os.walk(folder, followlinks=True):
        subfolder = Path(dirpath)
        realfolder = subfolder.resolve()
        if realfolder in realfolders:
            dirnames.clear()
            continue
        realfolders.add(realfolder)
        if folder_is_artis_model(subfolder):
            modelfolders.append(subfolder)
            dirnames.clear()
        else:
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
    return sorted(modelfolders, key=natural_sort_key)


def get_cell_columns(modelfolder: Path) -> dict[str, t.Any]:
    """Return the dimensions of model.txt, its number of cells, and the number of cells with matter."""
    dfmodel, modelmeta = get_modeldata(modelfolder, printwarningsonly=True)
    cellswithmatter = (
        add_derived_cols_to_modeldata(dfmodel, modelmeta=modelmeta).select((pl.col("rho") > 0).sum()).collect().item()
    )
    return {
        "dimensions": modelmeta["dimensions"],
        "cells": modelmeta["npts_model"],
        "cells_with_matter": cellswithmatter,
    }


def get_timestep_columns(modelfolder: Path) -> dict[str, t.Any]:
    """Return the number of time steps, the start of the first time step, and the end of the last time step."""
    tstarts = get_timestep_times(modelfolder, loc="start")
    tends = get_timestep_times(modelfolder, loc="end")
    return {"timesteps": len(tstarts), "tstart_days": tstarts[0], "tend_days": tends[-1]}


def get_option_columns(modelfolder: Path) -> dict[str, t.Any]:
    """Return the NLTE, non-thermal, thermalisation, and packet options of artisoptions.h, or None for each option.

    ARTIS renamed some options, thus each column also reads the older name.
    """
    options = get_artis_options(modelfolder)
    nlteoption = options.get("ION_NLEVELS_EXCITED_NLTE", options.get("LEVEL_IS_NLTE"))
    nlte = None if nlteoption is None else not (returns_only(nlteoption, "false") or returns_only(nlteoption, "0"))

    nonthermal: bool | None = None
    if "NT_SCHEME" in options:
        nonthermal = not options["NT_SCHEME"].endswith("NT_OFF")
    elif "NT_ON" in options:
        nonthermal = options["NT_ON"] == "true"

    thermalisation: str | None = None
    if "PARTICLE_THERMALISATION_SCHEME" in options:
        thermalisation = options["PARTICLE_THERMALISATION_SCHEME"].rpartition("::")[2]
    elif "INSTANT_PARTICLE_DEPOSITION" in options:
        thermalisation = f"INSTANT_PARTICLE_DEPOSITION={options['INSTANT_PARTICLE_DEPOSITION']}"

    numpackets: int | None = None
    if "NUM_PACKETS" in options:
        numpackets = parse_packet_count(options["NUM_PACKETS"])
    elif "MPKTS" in options:
        # MPKTS is the number of packets of each rank
        numpackets = parse_packet_count(options["MPKTS"]) * get_nprocs(modelfolder)

    return {"nlte": nlte, "nonthermal": nonthermal, "thermalisation": thermalisation, "num_packets": numpackets}


def parse_packet_count(valuetext: str) -> int:
    """Return the number of packets of an option value, e.g. 2e8 or 1'000'000."""
    return round(float(valuetext.replace("'", "")))


def get_version_columns(modelfolder: Path) -> dict[str, t.Any]:
    """Return the git version of ARTIS in the first log of the last run folder."""
    runversions = list(get_run_versions(modelfolder).values())
    return {"artis_version": runversions[-1].get("git version") if runversions else None}


def get_peak_columns(modelfolder: Path) -> dict[str, t.Any]:
    """Return the largest luminosity in erg/s of light_curve.out and its time in days."""
    dflightcurve = scan_lightcurve(find_lightcurve_file(modelfolder))[-1]
    peaktime, peaklum = (
        dflightcurve
        .filter(pl.col("time_days").is_finite() & pl.col("luminosity_erg/s").is_finite())
        .select(pl.col("time_days").get(pl.col("luminosity_erg/s").arg_max()), pl.col("luminosity_erg/s").max())
        .collect()
        .row(0)
    )
    return {"peak_lum_erg_per_s": peaklum, "peak_time_days": peaktime}


COLUMNPARTS: t.Final[tuple[tuple[str, Callable[[Path], dict[str, t.Any]]], ...]] = (
    ("the cells of model.txt", get_cell_columns),
    ("the time steps", get_timestep_columns),
    ("the options of artisoptions.h", get_option_columns),
    ("the ARTIS version", get_version_columns),
    ("the peak of light_curve.out", get_peak_columns),
)


def get_catalog_row(modelfolder: Path, folder: Path) -> dict[str, t.Any]:
    """Return the row of one model. A part that fails gives None values and a warning, and the other parts continue.

    A missing file gives None values and no warning, because a model with no output, or with no copy of
    artisoptions.h, is usual.
    """
    modelname = get_model_name(modelfolder)
    relativefolder = str(modelfolder.relative_to(folder))
    row: dict[str, t.Any] = dict.fromkeys(CATALOGSCHEMA.names()) | {
        "modelname": modelname,
        "modelfolder": relativefolder,
    }
    failures: list[str] = []
    for partname, getcolumns in COLUMNPARTS:
        try:
            row |= getcolumns(modelfolder)
        except FileNotFoundError:
            continue
        # a fault in one model must not stop the table of the other models
        except Exception as exc:  # ruff:ignore[blind-except]
            failures.append(f"{partname} ({type(exc).__name__}: {exc})")
    if failures:
        print_warning(f"Model {modelname} in {relativefolder}: the table gives nan for {'; '.join(failures)}")
    return row


def get_catalog(folder: Path) -> pl.DataFrame:
    """Return a table with one row for each model folder at or below a folder."""
    modelfolders = find_model_folders(folder)
    if not modelfolders:
        exit_with_error(f"{folder} holds no model folder", "A model folder holds input.txt")
    print_detail(f"{len(modelfolders)} model folders below {folder.resolve()}")
    return pl.DataFrame([get_catalog_row(modelfolder, folder) for modelfolder in modelfolders], schema=CATALOGSCHEMA)


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path(),
        help="The folder to search for model folders. The default is the working folder",
    )
    addarg_table_output(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Make a table of the ARTIS models below a folder, e.g. the cells, the options, and the peak luminosity."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    folder = Path(args.folder)
    if not folder.is_dir():
        exit_with_error(f"{folder} is not a folder", "Give the folder that holds the model folders")
    outputfile = resolve_table_outputfile(args, "catalog.txt")

    dfcatalog = get_catalog(folder)

    if outputfile is None:
        print_product(args, "\n".join(f"# {line}" for line in METHODLINES))
    print_or_write_table(args, dfcatalog, outputfile, METHODLINES)
