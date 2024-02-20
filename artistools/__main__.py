# PYTHON_ARGCOMPLETE_OK


import argparse
import importlib
import typing as t

import argcomplete

from artistools.commands import CommandType
from artistools.commands import subcommandtree as atdictcommands
from artistools.misc import CustomArgHelpFormatter


def addsubparsers(
    parser: argparse.ArgumentParser, parentcommand: str, subcommandtree: CommandType, depth: int = 1
) -> None:
    def func(args: t.Any) -> None:
        parser.print_help()

    parser.set_defaults(func=func)
    subparsers = parser.add_subparsers(dest=f"{parentcommand} command", required=False)

    for subcommand, subcommands in subcommandtree.items():
        strhelp: str | None
        if isinstance(subcommands, dict):
            strhelp = "command group"
            submodule = None
        else:
            submodulename, funcname = subcommands
            namestr = f"artistools.{submodulename.removeprefix('artistools.')}" if submodulename else "artistools"
            submodule = importlib.import_module(namestr, package="artistools")
            func = getattr(submodule, funcname)
            strhelp = func.__doc__

        subparser = subparsers.add_parser(subcommand, help=strhelp, formatter_class=CustomArgHelpFormatter)

        if submodule:
            submodule.addargs(subparser)
            subparser.set_defaults(func=func)
        else:
            assert not isinstance(subcommands, tuple)
            addsubparsers(parser=subparser, parentcommand=subcommand, subcommandtree=subcommands, depth=depth + 1)


def addargs(parser: argparse.ArgumentParser) -> None:
    pass


def main(args: argparse.Namespace | None = None, argsraw: t.Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Parse and run artistools commands."""
    parser = argparse.ArgumentParser(
        formatter_class=CustomArgHelpFormatter,
        description="Artistools base command.",
    )
    parser.set_defaults(func=None)

    addsubparsers(parser, "artistools", atdictcommands)

    argcomplete.autocomplete(parser)
    if args is None:
        args = parser.parse_args([] if kwargs else argsraw)
        args.func(args=args)


if __name__ == "__main__":
    main()
