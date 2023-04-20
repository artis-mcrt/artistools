# PYTHON_ARGCOMPLETE_OK
import argparse
import importlib
import multiprocessing

import argcomplete


def addargs(parser=None) -> None:
    pass


dictcommands = {
    "comparetogsinetwork": ("gsinetwork", "main"),
    "describeinputmodel": ("inputmodel.describeinputmodel", "main"),
    "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
    "plotspherical": ("plotspherical", "main"),
    "estimators": {
        "plot": ("estimators.plotestimators", "main"),
    },
    "inputmodel": {
        "describe": ("inputmodel.describeinputmodel", "main"),
        "maptogrid": ("inputmodel.maptogrid", "main"),
        "makeartismodelfromparticlegridmap": ("inputmodel.modelfromhydro", "main"),
        "makeartismodel": ("inputmodel.makeartismodel", "main"),
    },
    "lightcurves": {
        "plot": ("lightcurve.plotlightcurve", "main"),
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
    multiprocessing.freeze_support()
    main()
