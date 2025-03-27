# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence
from pathlib import Path

import argcomplete
from yaml import dump as yamldump

import artistools as at


def addargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-inputpath", "-i", default=".", help="Path of input ARTIS model")

    parser.add_argument("-temperature", "-T", default=10000, help="Temperature to use in TARDIS file")

    parser.add_argument("-dilution_factor", "-W", default=1.0, help="Dilution factor to use in TARDIS file")

    parser.add_argument(
        "-abundtype",
        choices=["nuclear", "elemental"],
        default="elemental",
        help="Output nuclear or elemental abundances",
    )

    parser.add_argument("-maxatomicnumber", default=92, help="Maximum atomic number for elemental abundances")

    parser.add_argument("-outputpath", "-o", default=".", help="Path of output TARDIS model file")


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Convert an ARTIS format model to TARDIS format."""
    if args is None:
        parser = argparse.ArgumentParser(formatter_class=at.CustomArgHelpFormatter, description=__doc__)

        addargs(parser)
        at.set_args_from_dict(parser, kwargs)
        argcomplete.autocomplete(parser)
        args = parser.parse_args([] if kwargs else argsraw)

    temperature = args.temperature
    dilution_factor = args.dilution_factor

    modelpath = Path(args.inputpath)

    dfmodel, t_model_init_days, _ = at.inputmodel.get_modeldata_tuple(
        modelpath, get_elemabundances=(args.abundtype == "elemental")
    )

    dfmodel["rho"] = 10 ** dfmodel["logrho"]

    if args.abundtype == "nuclear":
        # nuclide abundances
        listspecies = [
            col[2:]
            for col in dfmodel.columns
            if col.startswith("X_") and col.upper() != "X_FEGROUP" and col[-1].isdigit()
        ]
    else:
        # nuclide abundances
        listspecies = [
            col[2:]
            for col in dfmodel.columns
            if col.startswith("X_") and col.upper() != "X_FEGROUP" and not col[-1].isdigit()
        ]

    if args.maxatomicnumber and args.maxatomicnumber > 0:
        listspecies = [species for species in listspecies if at.get_atomic_number(species) <= args.maxatomicnumber]

    modelname = at.get_model_name(modelpath)
    outputfilepath = Path(args.outputpath, f"{modelname}.csvy")
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
    with outputfilepath.open("w", encoding="utf-8") as fileout:
        fileout.write("---\n")
        yamldump(dictmeta, fileout, sort_keys=False)
        fileout.write("---\n")
        fileout.write(",".join(["velocity", "density", "t_rad", "dilution_factor", *listspecies]))
        fileout.write("\n")

        # fileout.write(f'{0.},{0.:.4e},{0.},{0.},{",".join([f"{0.:.4e}" for _ in listspecies])}\n')

        for cell in dfmodel.itertuples(index=False):
            abundlist = [f"{getattr(cell, f'X_{strnuc}'):.4e}" for strnuc in listspecies]
            fileout.write(
                f"{cell.vel_r_max_kmps},{cell.rho:.4e},{temperature},{dilution_factor},{','.join(abundlist)}\n"
            )

    print(f"Saved {outputfilepath}")


if __name__ == "__main__":
    main()
