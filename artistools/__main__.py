# PYTHON_ARGCOMPLETE_OK


import argparse

import argcomplete

from .commands import addsubparsers
from .commands import dictcommands as atdictcommands
from .misc import CustomArgHelpFormatter


def addargs(parser: argparse.ArgumentParser) -> None:
    pass


def main(args=None, argsraw=None, **kwargs) -> None:
    """Parse and run artistools commands."""
    parser = argparse.ArgumentParser(
        formatter_class=CustomArgHelpFormatter,
        description="Artistools base command.",
    )
    parser.set_defaults(func=None)

    addsubparsers(parser, "artistools", atdictcommands)

    argcomplete.autocomplete(parser)
    args = parser.parse_args(argsraw)
    args.func(args=args)


if __name__ == "__main__":
    main()
