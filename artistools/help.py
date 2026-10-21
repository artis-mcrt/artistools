"""Print the help of artistools, of a command group, or of one command.

"artistools help plotspectra" prints the same text as "artistools plotspectra --help", as "uv help run" and
"git help commit" do. The help is the product of the command, thus --quiet keeps it.
"""

import argparse
import typing as t
from collections.abc import Sequence

from artistools.misc import parse_cli_args
from artistools.misc import print_product


def addargs(parser: argparse.ArgumentParser) -> None:
    """Add the argument that names the command."""
    parser.add_argument(
        "command",
        nargs="*",
        default=[],
        help="Command to give the help of, e.g. plotspectra or inputmodel describe. With no command, the help of"
        " artistools",
    )


def get_command_parser(words: Sequence[str]) -> argparse.ArgumentParser:
    """Return the parser of the command that the words name. An unknown word stops with the usual error."""
    from artistools.__main__ import build_parser

    parser = build_parser()
    for index, word in enumerate(words):
        subparsers = next(
            (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),  # ruff:ignore[private-member-access]
            None,
        )
        # a word after a command with no subcommands is a bad argument, thus the error path of argparse gives the
        # exit status 2, as "artistools timesteps extra" does
        if subparsers is None:
            parser.error(f"the command {' '.join(words[:index])} has no subcommand {word}")
        else:
            # the check of argparse gives the error and the closest names for a word that names no command
            parser._check_value(subparsers, word)  # ruff:ignore[private-member-access]
            parser = subparsers.choices[word]

    return parser


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None, **kwargs: t.Any) -> None:
    """Print the help of artistools, of a command group, or of one command."""
    args = parse_cli_args(addargs, __doc__, args, argsraw, kwargs)
    print_product(args, get_command_parser(args.command).format_help().rstrip("\n"))
