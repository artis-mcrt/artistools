# PYTHON_ARGCOMPLETE_OK
"""Entry point for `python -m artistools` and the artistools command."""

import argparse
import contextlib
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

    quiet = getattr(args, "quiet", False)
    starttime = time.monotonic()
    with contextlib.ExitStack() as stack:
        if quiet:
            devnull = stack.enter_context(Path(os.devnull).open("w", encoding="utf-8"))
            args.productstream = sys.stdout
            stack.enter_context(contextlib.redirect_stdout(devnull))

        func(args=args)

    # a reader of the output that stops early, e.g. head, makes the flush fail. The flush at the exit gives no
    # message that the dispatcher controls, thus the flush comes here
    sys.stdout.flush()

    # a long run says how long it took, thus a wait was the data and not a fault. A quick run says
    # nothing, and the line goes to the standard error beside the progress bars. --quiet takes it away,
    # because it reports the progress and not a fault. A window of --show or --interactive waits for the
    # user, thus the time of such a run is not the time of the data
    elapsed = time.monotonic() - starttime
    waitsforuser = getattr(args, "show", False) or getattr(args, "interactive", False)
    if elapsed >= SLOW_COMMAND_SECONDS and not quiet and not waitsforuser:
        print(f"The command took {elapsed:.1f} seconds", file=sys.stderr)


def main(args: argparse.Namespace | None = None, argsraw: Sequence[str] | None = None) -> None:
    """Parse and run an artistools subcommand."""
    import argcomplete

    from artistools.commands import build_script_parser
    from artistools.commands import get_hidden_commands
    from artistools.misc import check_time_selection
    from artistools.misc import resolve_output_argument
    from artistools.misc import resolve_yscale
    from artistools.misc import separate_trailing_folders

    # a per-command console script such as plotartisestimators runs this same function. The name that
    # started it selects one subcommand, and that parser holds no other command. Every entry point then
    # reads --quiet, reports a bad argument without a traceback, and tests the time arguments the same way
    scriptparser = build_script_parser(Path(sys.argv[0]).stem) if args is argsraw is None else None
    parser = scriptparser or build_parser()

    # the help hides a command such as server, thus tab completion does not offer it
    argcomplete.autocomplete(parser, exclude=get_hidden_commands())

    if args is None:
        args = parser.parse_args(separate_trailing_folders(argsraw))
        # During a call from Python code, sys.argv holds a different command. Thus a command that shows its own
        # command line, e.g. plotspectra --interactive, reads argsraw. The first word names the subcommand
        if argsraw is not None:
            args.dispatcherargsraw = list(argsraw)

    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return

    try:
        if (argparser := getattr(args, "argparser", None)) is not None:
            check_time_selection(argparser, args, argsraw)
            # the parser of the command recorded what it writes, thus -o takes its rule here
            resolve_output_argument(args)
            resolve_yscale(args)

        run_command(func, args)
    except KeyboardInterrupt:
        if os.environ.get("ARTISTOOLS_TRACEBACK"):
            raise
        print("artistools stopped at a keyboard interrupt", file=sys.stderr)
        # the shell gives 128 plus the number of the signal SIGINT
        raise SystemExit(130) from None
    except BrokenPipeError:
        # the reader of the output stopped. Python writes the rest of the buffer at the exit, thus the output goes
        # to the null device first, and the exit gives no second error
        nullfd = os.open(os.devnull, os.O_WRONLY)
        with contextlib.suppress(OSError, ValueError):
            os.dup2(nullfd, sys.stdout.fileno())
        os.close(nullfd)
        raise SystemExit(1) from None
    except (AssertionError, ImportError, OSError, ValueError) as exc:
        if os.environ.get("ARTISTOOLS_TRACEBACK"):
            raise
        # a bad argument, a missing or incomplete input file, a read-only model folder, a server that stopped, or an
        # optional package that does not import is a user problem, thus report it without a traceback.
        # import_optional names the install command. An assert that carries no message is an internal check,
        # thus say so rather than let the user read it as a mistake of their own, and name the variable that gives
        # the full traceback
        from artistools.misc import print_error

        if detail := str(exc):
            print_error("\n".join([detail, *get_context_notes(exc)]), get_error_help(exc))
        else:
            print_error(
                f"an internal check of artistools failed ({type(exc).__name__}). This is a fault in "
                "artistools and not in your arguments. Set ARTISTOOLS_TRACEBACK=1 to get the full traceback"
            )
        raise SystemExit(1) from exc
    except get_polars_panic() as exc:
        if not is_truncated_file_panic(exc) or os.environ.get("ARTISTOOLS_TRACEBACK"):
            raise
        from artistools.misc import print_error

        print_error(
            "\n".join(["an input file ends before the end of its data", *get_context_notes(exc)]), TRUNCATED_FILE_HELP
        )
        raise SystemExit(1) from exc


# polars gives no name of the file, but each reader prints the name of the file before it reads it
TRUNCATED_FILE_HELP = (
    "A compressed file can be incomplete, e.g. a partial copy or the output of a run that has not finished."
    " Without --quiet, the last line that starts with 'Reading' names the file"
)


def get_context_notes(exc: BaseException) -> list[str]:
    """Return the notes of one line of an exception, e.g. the file of a polars error or the host of a remote error.

    A note of more lines holds a traceback, e.g. the traceback of the server, which ARTISTOOLS_TRACEBACK=1 shows.
    """
    return [note for note in getattr(exc, "__notes__", []) if isinstance(note, str) and "\n" not in note]


def get_error_help(exc: BaseException) -> str:
    """Return the help line for an error that comes from an input file that ends too early, or an empty string."""
    return TRUNCATED_FILE_HELP if isinstance(exc, OSError) and "unexpected end of file" in str(exc) else ""


def get_polars_panic() -> type[BaseException]:
    """Return the class of a panic of polars, which is a BaseException.

    The dispatcher imports polars only when an error comes, thus a quick command such as version starts quickly.
    """
    import polars as pl

    return pl.exceptions.PanicException


def is_truncated_file_panic(exc: BaseException) -> bool:
    """Return whether a panic of polars comes from a source file that ends too early.

    polars panics when the reader of a compressed source raises EOFError, e.g. for an incomplete .xz file. A
    different panic is a fault of polars, thus it keeps its traceback.
    """
    return "EOFError" in str(exc)


if __name__ == "__main__":
    main()
