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
from artistools.inputmodel.core import scan_modeldata
from artistools.misc import addarg_force
from artistools.misc import addarg_modelpath
from artistools.misc import addarg_output
from artistools.misc import confirm_overwrite
from artistools.misc import exit_with_error
from artistools.misc import get_model_name
from artistools.misc import model_path_from_text
from artistools.misc import parse_cli_args
from artistools.misc import print_saved


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add arguments to an argparse parser object."""
    addarg_modelpath(parser, positional=True, default=Path(), helptext="Path of the ARTIS model")
    # the earlier name of the model path
    parser.add_argument("-inputpath", "-i", dest="modelpath", type=model_path_from_text, help=argparse.SUPPRESS)

    parser.add_argument("-temperature", "-T", type=float, default=10000.0, help="Temperature [K] for the TARDIS file")

    parser.add_argument("-dilution_factor", "-W", type=float, default=1.0, help="Dilution factor for the TARDIS file")

    parser.add_argument(
        "-abundtype",
        choices=["nuclear", "elemental"],
        default="elemental",
        help="Output nuclear or elemental abundances",
    )

    parser.add_argument("-maxatomicnumber", type=int, default=92, help="Maximum atomic number for elemental abundances")

    addarg_output(
        parser,
        kind="folder",
        helptext="The folder for the TARDIS model file, which has the name of the model",
        default=Path(),
    )
    addarg_force(parser)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Convert an ARTIS format model to TARDIS format.

    TARDIS reads the velocity of the first row of the CSV part as the inner edge of the innermost shell. It reads the
    velocity of each other row as the outer edge of a shell, and it discards the cell values of the first row. Thus
    the first row holds the inner edge of the innermost ARTIS shell, which is 0 km/s for an ARTIS model. Its cell
    values are a copy of the values of the innermost shell. Each other row holds the outer edge and the values of one
    ARTIS shell.
    """
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)

    temperature = args.temperature
    dilution_factor = args.dilution_factor

    modelpath = Path(args.modelpath)
    # the file has the name of the model, thus the check needs no data and comes before the work
    confirm_overwrite([args.outputfile], (f"{get_model_name(modelpath)}.csvy",), force=args.force)

    pldfmodel, modelmeta = scan_modeldata(modelpath, get_elemabundances=(args.abundtype == "elemental"))
    if modelmeta["dimensions"] != 1:
        # a TARDIS model holds one radial velocity for each shell, which only a 1D model gives
        exit_with_error(
            f"The model is {modelmeta['dimensions']}D, and TARDIS takes a 1D model",
            "Reduce the model to 1D first, e.g. `artistools inputmodel makeartismodel --dimensionreduce 1`",
        )

    t_model_init_days = modelmeta["t_model_init_days"]

    dfmodel = (
        add_derived_cols_to_modeldata(pldfmodel, modelmeta=modelmeta)
        .select("vel_r_min_kmps", "vel_r_max_kmps", "rho", cs.starts_with("X_"))
        .collect()
    )
    inneredge_kmps = float(dfmodel["vel_r_min_kmps"].item(0))

    # a nuclide column ends with a mass number, e.g. X_Ni56. An elemental column does not
    wantsnuclides = args.abundtype == "nuclear"
    listspecies = [
        col[2:]
        for col in dfmodel.columns
        if col.startswith("X_")
        and col.upper() != "X_FEGROUP"
        and not col.endswith("_otherstable")
        and col[-1].isdigit() == wantsnuclides
    ]

    if args.maxatomicnumber and args.maxatomicnumber > 0:
        listspecies = [species for species in listspecies if get_atomic_number(f"X_{species}") <= args.maxatomicnumber]

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
                {
                    "name": "velocity",
                    "unit": "km/s",
                    "desc": (
                        "inner edge of the innermost shell in the first row,"
                        " and outer edge of a shell in each other row"
                    ),
                },
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

    dfshells = pl.DataFrame([
        dfmodel["vel_r_max_kmps"].cast(pl.Float64).alias("velocity"),
        format_scientific("rho").alias("density"),
        pl.repeat(str(temperature), dfmodel.height, dtype=pl.String, eager=True).alias("t_rad"),
        pl.repeat(str(dilution_factor), dfmodel.height, dtype=pl.String, eager=True).alias("dilution_factor"),
        *[format_scientific(f"X_{strnuc}").alias(strnuc) for strnuc in listspecies],
    ])
    # TARDIS discards the cell values of the first row, thus that row takes the valid values of the innermost shell
    dfout = pl.concat([dfshells.head(1).with_columns(velocity=pl.lit(inneredge_kmps, dtype=pl.Float64)), dfshells])
    print(
        f"The first row gives the inner edge of the innermost shell at {inneredge_kmps} km/s. TARDIS discards the cell"
        " values of that row, and the row holds a copy of the values of the innermost shell. Each other row gives the"
        " outer edge and the values of one ARTIS shell (TARDIS documentation: Edge-Defined and Cell-Defined Values)."
    )

    with outputfilepath.open("w", encoding="utf-8") as fileout:
        fileout.write("---\n")
        yamldump(dictmeta, fileout, sort_keys=False)
        fileout.write("---\n")
        fileout.flush()
        dfout.write_csv(fileout, separator=",", quote_style="never")

    print_saved(outputfilepath)
