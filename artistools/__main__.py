# PYTHON_ARGCOMPLETE_OK
import argparse
import os
import sys
import typing as t
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level artistools argument parser."""
    from artistools.commands import addsubparsers
    from artistools.commands import CustomArgHelpFormatter
    from artistools.commands import subcommandtree
    from artistools.version import version

    parserkwargs: dict[str, t.Any] = {
        "formatter_class": CustomArgHelpFormatter,
        "description": "Plotting and analysis tools for the ARTIS radiative transfer code.",
    }
    if sys.version_info >= (3, 14):
        parserkwargs["suggest_on_error"] = True  # suggest close matches for mistyped subcommands
    parser = argparse.ArgumentParser(**parserkwargs)
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {version}")

    addsubparsers(parser, "artistools", subcommandtree)

    return parser


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None) -> None:
    """Parse and run an artistools subcommand."""
    import argcomplete

    parser = build_parser()

    argcomplete.autocomplete(parser)

    if args is None:
        args = parser.parse_args(argsraw)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return

    try:
        func(args=args)
    except FileNotFoundError as exc:
        if os.environ.get("ARTISTOOLS_TRACEBACK"):
            raise
        # a missing input file is a user-environment problem, so report it without a traceback
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
