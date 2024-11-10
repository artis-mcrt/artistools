# PYTHON_ARGCOMPLETE_OK
import argparse
import collections.abc
import typing as t

import argcomplete

from artistools.commands import addsubparsers
from artistools.commands import subcommandtree
from artistools.misc import CustomArgHelpFormatter


def main(
    args: argparse.Namespace | None = None, argsraw: collections.abc.Sequence[str] | None = None, **kwargs: t.Any
) -> None:
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
