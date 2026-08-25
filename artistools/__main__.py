# PYTHON_ARGCOMPLETE_OK
"""Entry point for `python -m artistools` and the artistools command."""

import argparse
import os
import sys
import typing as t
from collections.abc import Sequence

if t.TYPE_CHECKING:
    from collections.abc import Callable


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level artistools argument parser."""
    from importlib.metadata import version

    from artistools.commands import addsubparsers
    from artistools.commands import CustomArgHelpFormatter
    from artistools.commands import subcommandtree

    parserkwargs: dict[str, t.Any] = {
        "formatter_class": CustomArgHelpFormatter,
        "description": "Plotting and analysis tools for the ARTIS radiative transfer code.",
    }
    if sys.version_info >= (3, 14):
        parserkwargs["suggest_on_error"] = True  # suggest close matches for mistyped subcommands
    parser = argparse.ArgumentParser(**parserkwargs)
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {version('artistools')}")

    addsubparsers(parser, "artistools", subcommandtree)

    return parser


def run_command(func: "Callable[..., None]", args: argparse.Namespace) -> None:
    """Run the subcommand. With --quiet, send its progress messages to the null device.

    An error message goes to the standard error, thus --quiet keeps it. No module holds a quiet flag,
    because the redirection covers the whole call.
    """
    if not getattr(args, "quiet", False):
        func(args=args)
        return

    import contextlib
    from pathlib import Path

    with Path(os.devnull).open("w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        func(args=args)


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
        run_command(func, args)
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        if os.environ.get("ARTISTOOLS_TRACEBACK"):
            raise
        # a bad argument or a missing input file is a user problem, thus report it without a traceback.
        # An assert gives no message, thus name the environment variable that shows where it happened
        detail = str(exc) or f"{type(exc).__name__} with no message. Set ARTISTOOLS_TRACEBACK=1 to see where"
        print(f"error: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
