#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import importlib
import multiprocessing
from typing import Union

import argcomplete


def addargs(parser=None) -> None:
    pass


dictcommands = {
    "inputmodel": {
        "describe": ("inputmodel.describeinputmodel", "main"),
        "maptogrid": ("inputmodel.maptogrid", "main"),
    },
    "lightcurves": {
        "plot": ("lightcurve.plotlightcurve", "main"),
    },
    "spectra": {
        "plot": ("spectra.plotspectra", "main"),
    },
}


def addsubparsers(parser, parentcommand, dictcommands, depth: int = 1) -> None:
    subparsers = parser.add_subparsers(dest=f"{parentcommand} command", required=True, title="test")
    for subcommand, subcommands in dictcommands.items():
        if isinstance(subcommands, dict):
            subparser = subparsers.add_parser(subcommand)

            addsubparsers(subparser, subcommand, subcommands, depth=depth + 1)
        else:
            command = subcommand
            submodulename, funcname = subcommands
            submodule = importlib.import_module(f"artistools.{submodulename}", package="artistools")
            subparser = subparsers.add_parser(command)
            submodule.addargs(subparser)
            subparser.set_defaults(func=getattr(submodule, funcname))


def main(args=None, argsraw=None, **kwargs) -> None:
    """Parse and run artistools commands."""

    import artistools.commands

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=None)

    addsubparsers(parser, "artistools", dictcommands)

    argcomplete.autocomplete(parser)
    args = parser.parse_args(argsraw)
    args.func(args=args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
