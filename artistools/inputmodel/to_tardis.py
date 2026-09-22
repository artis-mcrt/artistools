# PYTHON_ARGCOMPLETE_OK
"""Convert an ARTIS input model into a TARDIS model file."""

import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

from artistools.atomic import get_atomic_number
from artistools.inputmodel.core import add_derived_cols_to_modeldata
from artistools.inputmodel.core import get_modeldata
from artistools.misc import addarg_output
from artistools.misc import exit_with_error
from artistools.misc import get_model_name
from artistools.misc import parse_cli_args
from artistools.misc import print_saved


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    parser.add_argument("-inputpath", "-i", default=".", help="Path of input ARTIS model")

    parser.add_argument("-temperature", "-T", default=10000, help="Temperature to use in TARDIS file")

    parser.add_argument("-dilution_factor", "-W", default=1.0, help="Dilution factor to use in TARDIS file")

    parser.add_argument(
        "-abundtype",
        choices=["nuclear", "elemental"],
        default="elemental",
        help="Output nuclear or elemental abundances",
    )

    parser.add_argument("-maxatomicnumber", type=int, default=92, help="Maximum atomic number for elemental abundances")

    addarg_output(parser, kind="folder", helptext="Path of output TARDIS model file", default=Path())


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Convert an ARTIS format model to TARDIS format."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    temperature = args.temperature
    dilution_factor = args.dilution_factor

    modelpath = Path(args.inputpath)

    pldfmodel, modelmeta = get_modeldata(modelpath, get_elemabundances=(args.abundtype == "elemental"))
    if modelmeta["dimensions"] != 1:
        # a TARDIS model holds one radial velocity for each shell, which only a 1D model gives
        exit_with_error(
            f"the model is {modelmeta['dimensions']}D, and TARDIS takes a 1D model",
            "Reduce the model to 1D first, e.g. artistools makemodel -dimensionreduce 1",
        )

    t_model_init_days = modelmeta["t_model_init_days"]

    dfmodel = (
        add_derived_cols_to_modeldata(pldfmodel, modelmeta=modelmeta)
        .select("vel_r_max_kmps", "rho", cs.starts_with("X_"))
        .collect()
    )

    # a nuclide column ends with a mass number, e.g. X_Ni56. An elemental column does not
    wantsnuclides = args.abundtype == "nuclear"
    listspecies = [
        col[2:]
        for col in dfmodel.columns
        if col.startswith("X_") and col.upper() != "X_FEGROUP" and col[-1].isdigit() == wantsnuclides
    ]

    if args.maxatomicnumber and args.maxatomicnumber > 0:
        listspecies = [species for species in listspecies if get_atomic_number(species) <= args.maxatomicnumber]

    modelname = get_model_name(modelpath)
    outputfilepath = Path(args.outputfile, f"{modelname}.csvy")
    dictmeta = {
        "name": modelname,
        "description": "This model was converted from ARTIS format with artistools",
        "model_density_time_0": f"{t_model_init_days} day",
        "model_isotope_time_0": f"{t_model_init_days} day",
        "tardis_model_config_version": "v1.0",
        "datatype": {
            "fields": [
                {"name": "velocity", "unit": "km/s", "desc": "velocities of shell outer boundaries"},
                {"name": "density", "unit": "g/cm^3", "desc": "density of shell"},
                {"name": "t_rad", "unit": "K", "desc": "radiative temperature"},
                {"name": "dilution_factor", "desc": "dilution factor of shell"},
                *[{"name": strnuc, "desc": f"fractional {strnuc} abundance"} for strnuc in listspecies],
            ]
        },
    }
    from yaml import dump as yamldump

    def format_scientific(colname: str) -> pl.Series:
        """Format a column with four decimal digits in the scientific notation, e.g. 1.2345e-03."""
        return pl.Series(colname, np.char.mod("%.4e", dfmodel[colname].cast(pl.Float64).to_numpy()))

    dfout = pl.DataFrame([
        dfmodel["vel_r_max_kmps"].cast(pl.Float64).alias("velocity"),
        format_scientific("rho").alias("density"),
        pl.repeat(str(temperature), dfmodel.height, dtype=pl.String, eager=True).alias("t_rad"),
        pl.repeat(str(dilution_factor), dfmodel.height, dtype=pl.String, eager=True).alias("dilution_factor"),
        *[format_scientific(f"X_{strnuc}").alias(strnuc) for strnuc in listspecies],
    ])

    with outputfilepath.open("w", encoding="utf-8") as fileout:
        fileout.write("---\n")
        yamldump(dictmeta, fileout, sort_keys=False)
        fileout.write("---\n")
        fileout.flush()
        dfout.write_csv(fileout, separator=",", quote_style="never")

    print_saved(outputfilepath)


if __name__ == "__main__":
    from artistools.commands import run_module_as_subcommand

    run_module_as_subcommand(__spec__)
