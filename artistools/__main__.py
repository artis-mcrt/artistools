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


# An argument that asks for a listing makes the standard output the product of the command, thus --quiet
# must not hide it. Each name here is the dest of such an argument.
LISTING_ARGS = ("listvariables", "listnuclides")


def run_command(func: "Callable[..., None]", args: argparse.Namespace) -> None:
    """Run the subcommand. With --quiet, send its progress messages to the null device.

    An error message goes to the standard error, thus --quiet keeps it. No module holds a quiet flag,
    because the redirection covers the whole call.
    """
    wantslisting = any(getattr(args, name, False) for name in LISTING_ARGS)
    if wantslisting or not getattr(args, "quiet", False):
        func(args=args)
        return

    import contextlib
    from pathlib import Path

    with Path(os.devnull).open("w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        func(args=args)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None) -> None:
    """Parse and run an artistools subcommand."""
    import argcomplete

    from artistools.misc import check_time_selection

    parser = build_parser()

    argcomplete.autocomplete(parser)

    if args is None:
        args = parser.parse_args(argsraw)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return

    if (argparser := getattr(args, "argparser", None)) is not None:
        check_time_selection(argparser, args)

    try:
        run_command(func, args)
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        if os.environ.get("ARTISTOOLS_TRACEBACK"):
            raise
        # a bad argument or a missing input file is a user problem, thus report it without a traceback.
        # An assert that carries no message is an internal check, thus say so rather than let the user
        # read it as a mistake of their own, and name the variable that gives the full traceback
        if detail := str(exc):
            print(f"error: {detail}", file=sys.stderr)
        else:
            print(
                f"error: an internal check of artistools failed ({type(exc).__name__}). This is a fault in "
                "artistools and not in your arguments. Set ARTISTOOLS_TRACEBACK=1 to get the full traceback",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
