# PYTHON_ARGCOMPLETE_OK
"""Entry point for `python -m artistools` and the artistools command."""

import argparse
import os
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

if t.TYPE_CHECKING:
    from collections.abc import Callable


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level artistools argument parser."""
    from importlib.metadata import version

    from artistools.commands import addsubparsers
    from artistools.commands import CustomArgHelpFormatter
    from artistools.commands import get_epilog
    from artistools.commands import subcommandtree
    from artistools.commands import SuggestingArgumentParser

    parserkwargs: dict[str, t.Any] = {
        "formatter_class": CustomArgHelpFormatter,
        "description": "Plotting and analysis tools for the ARTIS radiative transfer code.",
        "epilog": get_epilog(),
    }
    # the subclass suggests a subcommand and a flag, on Python 3.13 and 3.14 alike
    parser = SuggestingArgumentParser(**parserkwargs)
    parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {version('artistools')}")

    addsubparsers(parser, subcommandtree)

    return parser


# a command that runs at least this long reports its wall time
SLOW_COMMAND_SECONDS = 15.0


def run_command(func: "Callable[..., None]", args: argparse.Namespace) -> None:
    """Run the subcommand. With --quiet, send its progress messages to the null device.

    An error message goes to the standard error, thus --quiet keeps it. A command writes its product
    with print_product, which reaches the standard output even with --quiet, thus a script reads that
    product with no progress message around it.
    """
    import time

    starttime = time.monotonic()
    if not getattr(args, "quiet", False):
        func(args=args)
    else:
        import contextlib

        with Path(os.devnull).open("w", encoding="utf-8") as devnull:
            args.productstream = sys.stdout
            with contextlib.redirect_stdout(devnull):
                func(args=args)

    # a long run says how long it took, thus a wait was the data and not a fault. A quick run says
    # nothing, and the line goes to the standard error beside the progress bars. --quiet takes it away,
    # because it reports the progress and not a fault
    elapsed = time.monotonic() - starttime
    if elapsed >= SLOW_COMMAND_SECONDS and not getattr(args, "quiet", False):
        print(f"The command took {elapsed:.1f} seconds", file=sys.stderr)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None) -> None:
    """Parse and run an artistools subcommand."""
    import argcomplete

    from artistools.commands import build_script_parser
    from artistools.misc import check_time_selection
    from artistools.misc import resolve_output_argument

    # a per-command console script such as plotartisestimators runs this same function. The name that
    # started it selects one subcommand, and that parser holds no other command. Every entry point then
    # reads --quiet, reports a bad argument without a traceback, and tests the time arguments the same way
    scriptparser = build_script_parser(Path(sys.argv[0]).stem) if args is argsraw is None else None
    parser = scriptparser or build_parser()

    argcomplete.autocomplete(parser)

    if args is None:
        args = parser.parse_args(argsraw)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return

    try:
        if (argparser := getattr(args, "argparser", None)) is not None:
            check_time_selection(argparser, args, argsraw)
            # the parser of the command recorded what it writes, thus -o takes its rule here
            resolve_output_argument(args)

        run_command(func, args)
    except (AssertionError, FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        if os.environ.get("ARTISTOOLS_TRACEBACK"):
            raise
        # a bad argument, a missing input file, or a missing optional package is a user problem, thus
        # report it without a traceback. import_optional names the command that installs the package.
        # An assert that carries no message is an internal check, thus say so rather than let the user
        # read it as a mistake of their own, and name the variable that gives the full traceback
        from artistools.misc import print_error

        if detail := str(exc):
            print_error(detail)
        else:
            print_error(
                f"an internal check of artistools failed ({type(exc).__name__}). This is a fault in "
                "artistools and not in your arguments. Set ARTISTOOLS_TRACEBACK=1 to get the full traceback"
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
