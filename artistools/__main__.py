# PYTHON_ARGCOMPLETE_OK
import argparse
import importlib

import argcomplete


def addargs(parser=None) -> None:
    pass


dictcommands = {
    "comparetogsinetwork": ("gsinetwork", "main"),
    "deposition": ("deposition", "main_analytical"),
    "describeinputmodel": ("inputmodel.describeinputmodel", "main"),
    "exportmassfractions": ("estimators.exportmassfractions", "main"),
    "listtimesteps": ("", "showtimesteptimes"),
    "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
    "maketardismodelfromartis": ("inputmodel.maketardismodelfromartis", "main"),
    "maptogrid": ("inputmodel.maptogrid", "main"),
    "plotestimators": ("estimators.plotestimators", "main"),
    "plotinitialcomposition": ("initial_composition", "main"),
    "plotlightcurves": ("lightcurve.plotlightcurve", "main"),
    "plotmodeldensity": ("inputmodel.plotdensity", "main"),
    "plotmodeldeposition": ("deposition", "main"),
    "plotlinefluxes": ("linefluxes", "main"),
    "plotmacroatom": ("macroatom", "main"),
    "plotnltepops": ("nltepops.plotnltepops", "main"),
    "plotnonthermal": ("nonthermal", "main"),
    "plotradfield": ("radfield", "main"),
    "plotspectra": ("spectra.plotspectra", "main"),
    "plotspherical": ("plotspherical", "main"),
    "plottransitions": ("transitions", "main"),
    "plotviewingangles": ("viewing_angles_visualization", "cli"),
    "setupcompletions": ("commands", "setup_completions"),
    "spencerfano": ("nonthermal.solvespencerfanocmd", "main"),
    "writecodecomparisondata": ("writecomparisondata", "main"),
    "inputmodel": {
        "describe": ("inputmodel.describeinputmodel", "main"),
        "maptogrid": ("inputmodel.maptogrid", "main"),
        "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
        "makeartismodel": ("inputmodel.makeartismodel", "main"),
        "artistools-make1dslicefrom3dmodel": ("inputmodel.1dslicefrom3d", "main"),
        "makeartismodel1dslicefromcone": ("inputmodel.slice1Dfromconein3dmodel", "main"),
        "makeartismodelbotyanski2017": ("inputmodel.botyanski2017", "main"),
        "makeartismodelfromshen2018": ("inputmodel.shen2018", "main"),
        "makeartismodelfromlapuente": ("inputmodel.lapuente", "main"),
        "makeartismodelscalevelocity": ("inputmodel.scalevelocity", "main"),
        "makeartismodelfullymixed": ("inputmodel.fullymixed", "main"),
        "makeartismodelsolar_rprocess": ("inputmodel.rprocess_solar", "main"),
        "makeartismodelfromsingletrajectory": ("inputmodel.rprocess_from_trajectory", "main"),
    },
    "spectra": {
        "plot": ("spectra.plotspectra", "main"),
    },
}


def addsubparsers(parser, parentcommand, dictcommands, depth: int = 1) -> None:
    subparsers = parser.add_subparsers(dest=f"{parentcommand} command", required=True)
    for subcommand, subcommands in dictcommands.items():
        subparser = subparsers.add_parser(subcommand, help=subcommand)
        if isinstance(subcommands, dict):
            addsubparsers(subparser, subcommand, subcommands, depth=depth + 1)
        else:
            submodulename, funcname = subcommands
            namestr = f"artistools.{submodulename.removeprefix('artistools.')}" if submodulename else "artistools"
            submodule = importlib.import_module(namestr, package="artistools")
            submodule.addargs(subparser)
            subparser.set_defaults(func=getattr(submodule, funcname))


def main(args=None, argsraw=None, **kwargs) -> None:
    """Parse and run artistools commands."""
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=None)

    addsubparsers(parser, "artistools", dictcommands)

    argcomplete.autocomplete(parser)
    args = parser.parse_args(argsraw)
    args.func(args=args)


if __name__ == "__main__":
    main()
