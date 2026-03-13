# PYTHON_ARGCOMPLETE_OK
import argparse
import typing as t
from collections.abc import Sequence

import argcomplete

from artistools.commands import addsubparsers
from artistools.commands import subcommandtree
from artistools.misc import CustomArgHelpFormatter


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Parse and run artistools commands."""
    parser = argparse.ArgumentParser(
        prog="artistools", formatter_class=CustomArgHelpFormatter, description="Artistools base command."
    )
    parser.set_defaults(func=None)

    addsubparsers(parser, "artistools", subcommandtree)

    argcomplete.autocomplete(parser)
    if args is None:
        args = parser.parse_args([] if kwargs else argsraw)
        args.func(args=args)


if __name__ == "__main__":
    main()
