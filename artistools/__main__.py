# PYTHON_ARGCOMPLETE_OK
import argparse
import importlib

import argcomplete


def addargs(parser=None) -> None:
    pass


dictcommands = {
    "comparetogsinetwork": ("gsinetwork", "main"),
    "deposition": ("artistools.deposition", "main_analytical"),
    "describeinputmodel": ("inputmodel.describeinputmodel", "main"),
    "spencerfano": ("artistools.nonthermal.solvespencerfanocmd", "main"),
    "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
    "maptogrid": ("inputmodel.maptogrid", "main"),
    "plotestimators": ("estimators.plotestimators", "main"),
    "plotlightcurves": ("lightcurve.plotlightcurve", "main"),
    "plotspectra": ("spectra.plotspectra", "main"),
    "plotspherical": ("plotspherical", "main"),
    "listtimesteps": ("artistools", "showtimesteptimes"),
    "maketardismodelfromartis": ("artistools.inputmodel.maketardismodelfromartis", "main"),
    "plotmodeldensity": ("artistools.inputmodel.plotdensity", "main"),
    "plotmodeldeposition": ("artistools.deposition", "main"),
    "exportmassfractions": ("artistools.estimators.exportmassfractions", "main"),
    "plotlinefluxes": ("artistools.linefluxes", "main"),
    "plotmacroatom": ("artistools.macroatom", "main"),
    "plotnltepops": ("artistools.nltepops.plotnltepops", "main"),
    "plotnonthermal": ("artistools.nonthermal", "main"),
    "plotradfield": ("artistools.radfield", "main"),
    "plottransitions": ("artistools.transitions", "main"),
    "plotinitialcomposition": ("artistools.initial_composition", "main"),
    "setupcompletions": ("artistools.commands", "setup_completions"),
    "plotviewingangles": ("artistools.viewing_angles_visualization", "cli"),
    "writecodecomparisondata": ("artistools.writecomparisondata", "main"),
    "inputmodel": {
        "describe": ("inputmodel.describeinputmodel", "main"),
        "maptogrid": ("inputmodel.maptogrid", "main"),
        "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
        "makeartismodel": ("inputmodel.makeartismodel", "main"),
        "artistools-make1dslicefrom3dmodel": ("artistools.inputmodel.1dslicefrom3d", "main"),
        "makeartismodel1dslicefromcone": ("artistools.inputmodel.slice1Dfromconein3dmodel", "main"),
        "makeartismodelbotyanski2017": ("artistools.inputmodel.botyanski2017", "main"),
        "makeartismodelfromshen2018": ("artistools.inputmodel.shen2018", "main"),
        "makeartismodelfromlapuente": ("artistools.inputmodel.lapuente", "main"),
        "makeartismodelscalevelocity": ("artistools.inputmodel.scalevelocity", "main"),
        "makeartismodelfullymixed": ("artistools.inputmodel.fullymixed", "main"),
        "makeartismodelsolar_rprocess": ("artistools.inputmodel.rprocess_solar", "main"),
        "makeartismodelfromsingletrajectory": ("artistools.inputmodel.rprocess_from_trajectory", "main"),
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
            submodule = importlib.import_module(
                f"artistools.{submodulename.removeprefix('artistools.')}", package="artistools"
            )
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
