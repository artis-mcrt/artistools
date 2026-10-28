"""Print the messages of a command to the terminal, and ask the user a question."""

import contextlib
import sys
import typing as t
from collections.abc import Sequence
from pathlib import Path

if t.TYPE_CHECKING:
    from collections.abc import Collection


# a typing error in a flag name, e.g. --listvaraibles, gives a ratio of 0.76 or more. The name of a different flag
# gives less, e.g. 0.71 for -figscale against -lgoscaley, 0.56 for --listnuclides, and 0.50 for --verbose
FLAG_SUGGESTION_CUTOFF = 0.75


def suggest_names(name: str, candidates: "Collection[str]") -> str:
    """Return a sentence that names the closest candidates, or an empty string when none is close.

    The sentence goes on the help line of an error, thus it carries no leading space. A name that
    differs only in case comes first, because that mistake is common and difflib scores a short name
    such as "te" against "Te" below its own threshold.
    """
    import difflib

    names = list(candidates)
    if samecase := [other for other in names if other.lower() == name.lower() and other != name]:
        return f"Did you mean {samecase[0]}?"

    matches = difflib.get_close_matches(name, names, n=3, cutoff=0.6)

    return f"Did you mean {', '.join(matches)}?" if matches else ""


def is_abbreviation(short: str, name: str) -> bool:
    """Return whether name starts with the first letter of short and holds each letter of short in the same order."""
    remaining = iter(name)
    return short[:1] == name[:1] and len(short) < len(name) and all(letter in remaining for letter in short)


def suggest_flags(given: str, visibleflags: "Collection[str]") -> str:
    """Return a sentence that names the flags that the user possibly meant, or an empty string.

    The user can give the start of a longer name, e.g. --dim for --dimensionreduce, thus a flag that starts
    with the given text comes before the closest name of difflib, which gave "-d". The shortest names
    come first, and three of them, as suggest_names gives: "-ti" on plotspectra starts 13 flags.

    An abbreviation comes next: a flag with the same first letter that holds the letters of the given name in their
    order, e.g. --timemin for --tmin. difflib then compares the names without their dashes, because two dashes match
    in every pair of long flags. "--bogus" got the suggestion "--verbose" with the ratio 0.62, and the ratio of the
    names alone is 0.50.
    """
    import difflib

    starts = sorted(sorted(flag for flag in visibleflags if flag.startswith(given)), key=len)
    if starts:
        return f"Did you mean {', '.join(starts[:3])}?"

    givenname = given.lstrip("-")
    abbreviated = sorted(sorted(flag for flag in visibleflags if is_abbreviation(givenname, flag.lstrip("-"))), key=len)
    if abbreviated:
        return f"Did you mean {', '.join(abbreviated[:3])}?"

    flagofname = {flag.lstrip("-"): flag for flag in visibleflags}
    matches = difflib.get_close_matches(givenname, list(flagofname), n=3, cutoff=FLAG_SUGGESTION_CUTOFF)
    return f"Did you mean {', '.join(flagofname[name] for name in matches)}?" if matches else ""


def print_error(message: str, helptext: str = "") -> None:
    """Print an error to the standard error, then a help line that says what to do next.

    The error states the fault alone, thus a reader sees the remedy on its own line. rich colours the
    prefix in a terminal alone.
    """
    from rich.console import Console
    from rich.text import Text

    console = Console(stderr=True, highlight=False, soft_wrap=True)
    console.print(Text("error: ", style="bold red") + Text(message))
    if helptext:
        console.print(Text("help: ", style="bold cyan") + Text(helptext))


def print_warning(*values: object) -> None:
    """Print a warning to the standard error, thus --quiet keeps it and a script reads a clean product.

    rich colours the prefix in a terminal, and it writes plain text into a pipe or under NO_COLOR. The prefix is in
    lower case, as "error: " and "help: " are. The message after the prefix starts with a capital letter.
    """
    from rich.console import Console
    from rich.text import Text

    message = " ".join(str(value) for value in values)
    Console(stderr=True, highlight=False, soft_wrap=True).print(Text("warning: ", style="bold yellow") + Text(message))


def exit_with_error(message: str, helptext: str = "") -> t.NoReturn:
    """Print an error message and stop with a failing exit status.

    A mistake in the arguments earns a message rather than a traceback, and a script that runs the
    command sees that it failed. helptext names what the user can do next.
    """
    print_error(message, helptext)
    raise SystemExit(1)


def confirm_overwrite_of_files(existingfiles: Sequence[Path], force: bool) -> None:
    """Warn about each file that exists, and stop the command if the user does not agree to overwrite them."""
    if not existingfiles:
        return

    for path in existingfiles:
        print_warning(f"{path} exists")
    if not force and not user_agrees("Overwrite the files?"):
        exit_with_error(
            "The command stopped, and it did not overwrite the files", "Give -f or --force to overwrite them"
        )


def confirm_deletion(files: Sequence[Path], force: bool) -> None:
    """List the files, and stop the command if the user does not agree to delete them, as rm -i does."""
    for path in files:
        print_detail(path)
    if not force and not user_agrees(f"Delete the {len(files)} files?"):
        exit_with_error("The command stopped, and it did not delete the files", "Give -f or --force to delete them")


def user_agrees(question: str) -> bool:
    """Ask a question of the form (y/n) on the terminal, and return True for the answer yes.

    Nobody can answer when stdin is not a terminal, thus the answer is then no.
    """
    reply = ""
    if sys.stdin.isatty():
        # the end of the input, e.g. from Ctrl-D, is the answer "no"
        with contextlib.suppress(EOFError):
            reply = input(f"{question} (y/n) ")
    return reply.strip().lower() in {"y", "yes"}


def print_heading(text: str) -> None:
    """Print the name of the model or the series that the lines below it describe.

    Every command marks such a line, thus one style serves them all. rich gives the line its weight in
    a terminal alone, and a pipe takes the plain text. "====>" stood in front of it before.
    """
    from rich.console import Console
    from rich.text import Text

    Console(highlight=False, soft_wrap=True).print(Text(text, style="bold"))


def print_detail(*values: object) -> None:
    """Print one line of detail below a heading, with the indent that every command uses."""
    print(" ", *values)
