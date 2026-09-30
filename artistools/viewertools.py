"""Functions that each window of the --interactive option shares, e.g. plotspectra and plotestimators.

A viewer makes the command of the plot from its controls, then parses the command and draws the plot. The
functions here make and change the command tokens, and they build the parts of the Qt window that do not
depend on the command. Each function that uses Qt imports PySide6 when it runs, because the CLI must start
quickly and PySide6 is an optional dependency.
"""

import argparse
import contextlib
import io
import json
import math
import re
import shlex
import sys
import threading
import time
import traceback
import typing as t
import weakref
from functools import cache
from functools import partial
from functools import wraps
from pathlib import Path
from types import MappingProxyType

import numpy as np

from artistools.misc import addarg_quiet
from artistools.misc import exit_with_error
from artistools.misc import get_dirbin_definitions
from artistools.misc import get_dirbins
from artistools.misc import import_optional
from artistools.misc import path_is_file
from artistools.misc import print_error
from artistools.misc import separate_trailing_folders
from artistools.misc import write_gif
from artistools.misc.cliutils import dashes_arg
from artistools.misc.cliutils import SERIES_DEFAULT
from artistools.misc.fileio import resolve_modelpath
from artistools.misc.remote import is_remote_path
from artistools.misc.remote import on_model_host
from artistools.plottools import ExponentLabelFormatter
from artistools.plottools import get_series_colors
from artistools.plottools import make_room_for_title
from artistools.plottools import plain_label

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Generator
    from collections.abc import Mapping
    from collections.abc import Sequence
    from concurrent.futures import Future

    import matplotlib.axes as mplax
    import matplotlib.figure as mplfig
    import matplotlib.typing as mplt
    import numpy.typing as npt
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.font_manager import FontProperties
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    from artistools.commands import SuggestingArgumentParser

# each option of the table with its values, in the order of the command
type OptionRows = tuple[tuple[str, tuple[str, ...]], ...]

# a command raises these errors for a bad argument or input file. The dispatcher of the CLI reports the same errors
USER_ERRORS: t.Final = (AssertionError, FileNotFoundError, ModuleNotFoundError, PermissionError, ValueError)

REJECTED_MESSAGE: t.Final = "The command cannot draw this plot. The terminal shows the error"

# the process that runs from the application bundle of a viewer has this environment variable
MACOS_BUNDLE_VARIABLE: t.Final = "ARTISTOOLS_VIEWER_IN_BUNDLE"

# the limits of the -figwidthscale that a viewer gives the plot to fill the plot area
MIN_FIGWIDTHSCALE: t.Final[float] = 0.3
MAX_FIGWIDTHSCALE: t.Final[float] = 4.0

# the time after the last resize of the window, before the plot takes the new shape
FIT_MILLISECONDS: t.Final[int] = 200

# a new -figwidthscale that differs by less than this part from the old one draws no new plot, and fit_canvas scales
# the figure. A new plot reads the data again, thus a small change of the window does not start one
FIT_TOLERANCE: t.Final[float] = 0.05

# each continuous slider of a window has this number of positions
SLIDER_STEPS: t.Final = 1000

# the first width of the sidebar. The user can drag the handle between the plot and the sidebar
SIDEBAR_WIDTH: t.Final = 600

# Export Animation asks before it runs the command for more frames than this
MAX_ANIMATION_FRAMES: t.Final = 200

# changes closer together than this, e.g. the steps of a slider drag, give one step of Undo
UNDO_MERGE_SECONDS: t.Final = 0.8

# Undo keeps this number of steps
UNDO_LIMIT: t.Final = 200

# the first frame rate of the Play button, in frames per second
DEFAULT_PLAY_FPS: t.Final = 2.0


def get_command_tokens(
    argsraw: "Sequence[str] | None",
    kwargs: "Mapping[str, t.Any]",
    *,
    fromdispatcher: bool,
    dispatcherargsraw: "Sequence[str] | None",
) -> list[str]:
    """Return the arguments that the user gave to the command, without the name of the command.

    The dispatcher gives the parsed arguments alone. The tokens then come from the words of a call from Python code
    (dispatcherargsraw, which start with the subcommand), or else from sys.argv. A script for one command, e.g.
    plotartisspectrum, takes no word for the subcommand.
    """
    if kwargs:
        exit_with_error(
            "--interactive shows the command of the plot, and a call with keyword arguments has no command",
            "Give the arguments as a list, e.g. main(argsraw=['mymodel', '--interactive'])",
        )

    if argsraw is not None:
        return list(argsraw)

    if dispatcherargsraw is not None:
        return list(dispatcherargsraw[1:])

    if not fromdispatcher:
        return sys.argv[1:]

    from artistools.commands import get_subcommand_of_script

    return sys.argv[1:] if get_subcommand_of_script(Path(sys.argv[0]).stem) else sys.argv[2:]


def make_parser(addargs: "Callable[[argparse.ArgumentParser], None]") -> "SuggestingArgumentParser":
    """Return the parser of a command with the arguments of addargs, and --quiet, which the dispatcher adds."""
    from artistools.commands import SuggestingArgumentParser

    parser = SuggestingArgumentParser()
    addargs(parser)
    addarg_quiet(parser)
    return parser


def exit_for_other_actions(plotname: str, otheractions: "Mapping[str, bool]") -> None:
    """Stop when an argument selects an action that is not one plot. The window shows one plot only.

    otheractions gives each such argument and whether the command gives it.
    """
    if given := [name for name, isgiven in otheractions.items() if isgiven]:
        exit_with_error(
            f"--interactive shows one plot of {plotname}. A different action comes from: {', '.join(given)}",
            f"Remove {', '.join(given)}, or remove --interactive",
        )


def make_command_tokens(basetokens: "Sequence[str]", options: "Sequence[str]") -> list[str]:
    """Return the arguments of the command with the options of the controls after the paths at the start."""
    pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
    return [*basetokens[:pathcount], *options, *basetokens[pathcount:]]


def fix_title_position(axis: "mplax.Axes") -> None:
    """Keep the title at the top of the frame.

    matplotlib moves the title above the offset text of the y axis. For this, it measures the y axis each time that it
    draws the plot. ExponentLabelFormatter puts the offset in the label of the axis, thus the offset text is empty and
    the title stays at the top of the frame.
    """
    if axis.get_title() and isinstance(axis.yaxis.get_major_formatter(), ExponentLabelFormatter):
        axis.set_title(axis.get_title(), y=1.0)


def get_direction_kinds(runfolder: Path) -> list[str]:
    """Return the kinds of viewing direction of the run. A run with a configuration of virtual packets has observers."""
    return ["", "bin", "phi", "theta", *(["vpkt"] if path_is_file(runfolder / "vpkt.txt") else [])]


def get_direction_kind(args: argparse.Namespace) -> str:
    """Return the kind of viewing direction of the arguments, in the form of ControlValues.directionkind."""
    if args.plotvspecpol:
        return "vpkt"
    if not args.plotviewingangle:
        return ""
    if args.average_over_phi_angle:
        return "phi"
    return "theta" if args.average_over_theta_angle else "bin"


def get_direction_choices(runfolder: Path, directionkind: str, *, usedegrees: bool) -> list[tuple[int, str]]:
    """Return each bin of a kind of viewing direction with its label.

    An average over the phi angle or the theta angle takes the first bin of each group, as get_dirbins gives it.
    """
    averagephi, averagetheta = directionkind == "phi", directionkind == "theta"
    labels = get_dirbin_definitions(
        runfolder,
        get_dirbins(average_over_phi=averagephi, average_over_theta=averagetheta),
        vpkt_observers=directionkind == "vpkt",
        average_over_phi=averagephi,
        average_over_theta=averagetheta,
        usedegrees=usedegrees,
    )
    return list(labels.items())


class ThreadOutput(io.TextIOBase):
    """A standard stream that sends the text of each thread to the target of that thread.

    A worker thread draws a plot and hides its output, while the thread of the window still prints to the terminal.
    contextlib.redirect_stdout changes the stream of each thread, thus it cannot do this.
    """

    def __init__(self, stream: t.TextIO) -> None:
        """Send the text of each thread to stream, until send_output gives the thread a different target."""
        super().__init__()
        self.stream = stream
        self.local = threading.local()

    def get_target(self) -> t.TextIO:
        """Return the stream of the thread that calls this method."""
        target: t.TextIO = getattr(self.local, "target", self.stream)
        return target

    @t.override
    def write(self, text: str) -> int:
        return self.get_target().write(text)

    @t.override
    def flush(self) -> None:
        self.get_target().flush()

    @t.override
    def isatty(self) -> bool:
        return self.get_target().isatty()

    @t.override
    def fileno(self) -> int:
        return self.get_target().fileno()


@contextlib.contextmanager
def send_output(stdout: t.TextIO | None, stderr: t.TextIO) -> "Generator[None]":
    """Send the output of this thread to the streams until the block ends. A stdout of None keeps the terminal.

    A window installs ThreadOutput, and then the other threads keep their output. Without it, the streams of all the
    threads change.
    """
    if not (isinstance(sys.stdout, ThreadOutput) and isinstance(sys.stderr, ThreadOutput)):
        with contextlib.ExitStack() as stack:
            if stdout is not None:
                stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            yield
        return

    routes = [(stream, target) for stream, target in ((sys.stdout, stdout), (sys.stderr, stderr)) if target is not None]
    oldtargets = [getattr(stream.local, "target", None) for stream, _ in routes]
    for stream, target in routes:
        stream.local.target = target
    try:
        yield
    finally:
        for (stream, _), oldtarget in zip(routes, oldtargets, strict=True):
            if oldtarget is None:
                del stream.local.target
            else:
                stream.local.target = oldtarget


def run_command_step_with_warning(
    step: "Callable[[], str | None]", *, quiet: bool = True, echo: bool = True
) -> tuple[str | None, str]:
    """Run a step of a command, and return its message or the first line of its error, and its last warning.

    Each plot prints the same lines again, thus a quiet step discards the standard output. With echo, the terminal
    shows the whole error. A window stays open after a failed step, thus each type of error gives a message.
    """
    errors = io.StringIO()
    message: str | None
    try:
        with send_output(io.StringIO() if quiet else None, errors):
            message = step()
    except SystemExit:
        # exit_with_error and argparse print a line that starts with "error: " before they raise SystemExit
        message = get_first_line(errors.getvalue())
    except USER_ERRORS as exc:
        if echo:
            print_error(str(exc) or type(exc).__name__)
        message = get_first_line(str(exc))
    except Exception as exc:  # ruff:ignore[blind-except]
        errors.write(traceback.format_exc())
        message = f"{type(exc).__name__}: {get_first_line(str(exc))}"
    finally:
        if echo:
            sys.stderr.write(errors.getvalue())
    return message, get_last_warning(errors.getvalue())


def run_command_step(step: "Callable[[], str | None]", *, quiet: bool = True, echo: bool = True) -> str | None:
    """Run a step of a command, and return its message or the first line of its error for the status line."""
    message, _ = run_command_step_with_warning(step, quiet=quiet, echo=echo)
    return message


def remove_colour_codes(text: str) -> str:
    """Return the text without the colour codes of a terminal.

    rich colours the text under FORCE_COLOR or TTY_COMPATIBLE also when it writes into a capture.
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def get_last_warning(errors: str) -> str:
    """Return the last warning of the standard error of a step, without its prefix, or an empty text."""
    plainerrors = remove_colour_codes(errors)
    warnings = [line.strip() for line in plainerrors.splitlines() if line.strip().startswith("WARNING: ")]
    return warnings[-1].removeprefix("WARNING: ") if warnings else ""


def find_option_action(parser: argparse.ArgumentParser, argstring: str) -> tuple[argparse.Action | None, bool]:
    """Return the option that the argument names, and whether the argument also holds its value.

    argparse accepts these forms of a flag:

    - the full flag;
    - a flag with "=value";
    - a unique start of a flag;
    - a flag of one letter with its value joined, e.g. -t300.
    """
    if not argstring.startswith("-") or re.match(r"-\.?\d", argstring):
        return None, False

    optionactions = parser._option_string_actions  # ruff:ignore[private-member-access]
    name, equals, _ = argstring.partition("=")
    if name in optionactions:
        return optionactions[name], bool(equals)

    matches = {id(action): action for flag, action in optionactions.items() if flag.startswith(name)}
    if len(matches) == 1:
        return next(iter(matches.values())), bool(equals)

    if not argstring.startswith("--") and argstring[:2] in optionactions:
        return optionactions[argstring[:2]], True

    return None, False


def is_flag(argstring: str) -> bool:
    """Return True if the argument is a flag, and not a value such as a negative number."""
    return argstring.startswith("-") and not re.match(r"-\.?\d", argstring)


def split_argstrings(parser: "SuggestingArgumentParser", tokens: "Sequence[str]") -> list[str]:
    """Return the tokens with each joined flag and value apart, and each group of one-letter switches apart.

    argparse reads -qv as -q and -v. Each option has one row in the option table, thus each switch needs its own
    token. A group can end with a flag that takes a value, e.g. -qt300, and that value stays with its flag.
    """
    optionactions = parser._option_string_actions  # ruff:ignore[private-member-access]
    argstrings: list[str] = []
    for argstring in parser.split_joined_flags(tokens):
        firstaction = optionactions.get(argstring[:2])
        # each token after "--" is a positional argument, and a declared flag or a start of one is not a group
        isgroup = (
            "--" not in argstrings
            and not argstring.startswith("--")
            and len(argstring) > 2
            and firstaction is not None
            and firstaction.nargs == 0
            and not any(flag.startswith(argstring.partition("=")[0]) for flag in optionactions)
            and parser.is_switch_group(argstring)
        )
        if not isgroup:
            argstrings.append(argstring)
            continue
        for index, letter in enumerate(argstring[1:], start=2):
            argstrings.append(f"-{letter}")
            if optionactions[f"-{letter}"].nargs != 0:
                if rest := argstring[index:]:
                    argstrings.append(rest)
                break
    return argstrings


def remove_options(parser: "SuggestingArgumentParser", tokens: "Sequence[str]", dests: "Collection[str]") -> list[str]:
    """Return the tokens without the options of these dests and without the values of those options."""
    argstrings = split_argstrings(parser, tokens)
    kept: list[str] = []
    index = 0
    while index < len(argstrings):
        argstring = argstrings[index]
        index += 1
        if argstring == "--":
            kept.extend(argstrings[index - 1 :])
            break

        action, holdsvalue = find_option_action(parser, argstring)
        if action is None or action.dest not in dests:
            kept.append(argstring)
            continue

        if holdsvalue or action.nargs == 0:
            continue

        if action.nargs is None:
            index += 1
            continue

        # an option of nargs "?" takes one value, and an option of nargs "*" or "+" takes each value up to the next flag
        maxvalues = 1 if action.nargs == "?" else len(argstrings)
        while maxvalues and index < len(argstrings) and not is_flag(argstrings[index]):
            index += 1
            maxvalues -= 1

    return kept


def get_table_actions(parser: argparse.ArgumentParser, hiddendests: "Collection[str]") -> list[argparse.Action]:
    """Return the options that the table of the window offers, which are the options that are not in hiddendests.

    hiddendests holds the options that a different control of the window sets, and the options that give a
    different action from one plot.
    """
    return [
        action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.option_strings and action.help != argparse.SUPPRESS and action.dest not in hiddendests
    ]


def get_option_kind(action: argparse.Action) -> str:
    """Return the type of control that sets the value of an option in the table of the window.

    The types are these:

    - "flag": the option takes no value;
    - "choice": one value from a list;
    - "int": one integer that has a default, thus a spin box can show it;
    - "values": a fixed number of values, each in a separate field;
    - "list": a list of values that spaces separate;
    - "text": one value, or no value for an option with nargs "?".
    """
    if action.nargs == 0:
        return "flag"
    if isinstance(action.nargs, int):
        return "values"
    if action.nargs in {"*", "+"}:
        return "list"
    if action.choices:
        return "choice"
    if action.type is int and isinstance(action.default, int) and not isinstance(action.default, bool):
        return "int"
    return "text"


def get_default_tokens(action: argparse.Action) -> tuple[str, ...] | None:
    """Return the values of a new row of the table, or None if the option needs a value that has no default."""
    kind = get_option_kind(action)
    if kind == "flag" or action.nargs == "?":
        return ()
    if kind == "choice":
        choices = [str(choice) for choice in action.choices or ()]
        return (str(action.default) if str(action.default) in choices else choices[0],)
    if kind == "int":
        return (str(action.default),)
    return None


def split_option_rows(parser: "SuggestingArgumentParser", tokens: "Sequence[str]") -> tuple[OptionRows, list[str]]:
    """Return each option of the tokens with its values, and the tokens that are not part of an option.

    Each row gives the first flag of the option, thus an alias, e.g. -dx, becomes the full flag, e.g. -deltax.
    """
    rows: list[tuple[str, tuple[str, ...]]] = []
    othertokens: list[str] = []
    argstrings = split_argstrings(parser, tokens)
    index = 0
    while index < len(argstrings):
        argstring = argstrings[index]
        index += 1
        if argstring == "--":
            othertokens.extend(argstrings[index - 1 :])
            break

        action, holdsvalue = find_option_action(parser, argstring)
        if action is None:
            othertokens.append(argstring)
            continue

        values: list[str] = []
        if holdsvalue:
            _, equals, value = argstring.partition("=")
            values.append(value if equals else argstring[2:])
        elif action.nargs is None or isinstance(action.nargs, int):
            count = 1 if action.nargs is None else action.nargs
            values.extend(argstrings[index : index + count])
        else:
            # an option of nargs "?" takes one value, and an option of nargs "*" or "+" takes each value up to the
            # next flag
            maxvalues = 1 if action.nargs == "?" else len(argstrings)
            while len(values) < maxvalues and index + len(values) < len(argstrings):
                if is_flag(argstrings[index + len(values)]):
                    break
                values.append(argstrings[index + len(values)])
        index += 0 if holdsvalue else len(values)
        rows.append((action.option_strings[0], tuple(values)))

    return tuple(rows), othertokens


def get_option_tokens(flag: str, value: str) -> list[str]:
    """Return the tokens of an option with one value.

    Python 3.13 reads a value such as -1e-13 as an option, thus a value that starts with "-" joins its flag.
    """
    return [f"{flag}={value}"] if value.startswith("-") else [flag, value]


def get_row_values(rows: OptionRows, flag: str) -> tuple[str, ...] | None:
    """Return the values of the row of an option, or None if the rows have no such option."""
    return next((values for rowflag, values in rows if rowflag == flag), None)


def set_row_values(rows: OptionRows, changes: "Mapping[str, tuple[str, ...] | None]") -> OptionRows:
    """Return the rows with new values of some options, in their old places. None removes an option.

    A new option goes after the other rows.
    """
    changed = [
        (flag, changes.get(flag, values)) for flag, values in rows if not (flag in changes and changes[flag] is None)
    ]
    present = {flag for flag, _ in rows}
    added = [(flag, values) for flag, values in changes.items() if values is not None and flag not in present]
    return tuple((flag, values) for flag, values in (*changed, *added) if values is not None)


def get_option_row_tokens(rows: OptionRows) -> list[str]:
    """Return the command tokens of the rows of the option table."""
    return [
        token
        for flag, optionvalues in rows
        for token in (get_option_tokens(flag, *optionvalues) if len(optionvalues) == 1 else (flag, *optionvalues))
    ]


# the options that give one value for each series, e.g. each spectrum, in the order of the paths of the command
SERIES_STYLE_FLAGS: t.Final = ("-label", "-color", "-linestyle", "-linewidth", "-linealpha", "-dashes")


def get_series_value(values: "Sequence[str]", index: int) -> str | None:
    """Return the value of a series style option for the series at index, or None if the option gives none."""
    value = values[index] if index < len(values) else None
    return None if value == SERIES_DEFAULT else value


def get_series_tokens(values: "Sequence[str | None]") -> tuple[str, ...] | None:
    """Return the values of a series style option in the order of the series, or None for no option.

    A series with no value in front of a series with a value takes the token SERIES_DEFAULT.
    """
    values = list(values)
    while values and values[-1] is None:
        values.pop()
    return tuple(SERIES_DEFAULT if value is None else value for value in values) or None


def move_series_styles(rows: OptionRows, oldpaths: "Sequence[str]", newpaths: "Sequence[str]") -> OptionRows:
    """Return the option rows with the value of each series style option on the same path in the new list.

    Thus a -label stays on its series when the order changes. The values of a removed series go out of the rows.
    """
    changes: dict[str, tuple[str, ...] | None] = {}
    for flag in SERIES_STYLE_FLAGS:
        if not (values := get_row_values(rows, flag)):
            continue
        bypath = {path: get_series_value(values, index) for index, path in enumerate(oldpaths)}
        newvalues = get_series_tokens([bypath.get(path) for path in newpaths])
        if newvalues != values:
            changes[flag] = newvalues
    return set_row_values(rows, changes)


def set_series_rows(
    rows: OptionRows, paths: "Sequence[str]", path: str, changes: "Mapping[str, str | None]"
) -> OptionRows:
    """Return the option rows with the value of each series style option in changes for the series of one path.

    changes gives each option by its flag, e.g. -label. A value of None gives the series the default of the command,
    and the other series keep their values.
    """
    rowchanges: dict[str, tuple[str, ...] | None] = {}
    for flag, value in changes.items():
        oldvalues = get_row_values(rows, flag) or ()
        newvalues = [
            value if other == path else get_series_value(oldvalues, index) for index, other in enumerate(paths)
        ]
        rowchanges[flag] = get_series_tokens(newvalues)
    return set_row_values(rows, rowchanges)


def get_series_style(rows: OptionRows, paths: "Sequence[str]", path: str) -> dict[str, str | None]:
    """Return the value of each series style option of the series of one path, or None where it takes the default."""
    index = list(paths).index(path)
    return {flag: get_series_value(get_row_values(rows, flag) or (), index) for flag in SERIES_STYLE_FLAGS}


def get_path_colours(paths: "Sequence[str]", isreference: "Sequence[bool]", rows: OptionRows) -> dict[str, str]:
    """Return the colour of the series of each path, as the command gives it with the -color of the rows.

    A reference series takes black and greys, and a model takes the colours of the cycle.
    """
    usercolours = get_row_values(rows, "-color") or ()
    colours = get_series_colors(isreference, [get_series_value(usercolours, index) for index in range(len(paths))])
    return dict(zip(paths, colours, strict=True))


def get_helptexts(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Return the help text of each option by its dest. A tooltip gives it, thus the window and the CLI agree."""
    return {
        action.dest: str(action.help).replace("%(default)s", str(action.default)).replace("%%", "%")
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.help and action.help != argparse.SUPPRESS
    }


def get_actions_by_flag(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Return each option of the parser by its first flag, which is the flag of a row of the option table."""
    return {
        action.option_strings[0]: action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.option_strings
    }


def get_first_line(errortext: str) -> str:
    """Return the line of an error for the status line of the window, without the "error: " of print_error.

    argparse prints its usage line before the error, and a warning can come before an error. Thus the function
    returns the line that starts with "error: ". If no line has that start, it returns the first line. The function
    removes the colour codes of the terminal first, because a code in front of "error: " hides that start.
    """
    lines = [line.strip() for line in remove_colour_codes(errortext).splitlines() if line.strip()]
    errorline = next((line for line in lines if line.startswith("error: ")), lines[0] if lines else None)
    return errorline.removeprefix("error: ") if errorline is not None else REJECTED_MESSAGE


def get_nearest_range_start(tmids: "Sequence[float]", centre: float, count: int) -> int:
    """Return the first index of the range of count timesteps that has its centre nearest to the given time.

    The time field shows the centre of the range, thus a Return with no edit gives the same range. The centre of a
    range of an even count is near the boundary of two timesteps. Thus either timestep can hold the time.
    """

    def get_centre_offset(start: int) -> float:
        return abs((tmids[start] + tmids[start + count - 1]) / 2.0 - centre)

    return min(range(len(tmids) - count + 1), key=get_centre_offset)


def get_fitted_figwidthscale(
    figsize: tuple[float, float], figwidthscale: float, marginwidth: float, areawidth: float, areaheight: float
) -> float:
    """Return the -figwidthscale that gives the figure the shape of the plot area.

    The frame width is proportional to -figwidthscale, and the margins and the height stay the same.
    """
    figwidth, figheight = figsize
    widthperscale = (figwidth - marginwidth) / figwidthscale
    fitted = (figheight * areawidth / areaheight - marginwidth) / widthperscale
    # 2 decimals give a short command, and a small change of the window then keeps the frames
    return round(min(max(fitted, MIN_FIGWIDTHSCALE), MAX_FIGWIDTHSCALE), 2)


def make_icon_pixmap(size: int, curve: "npt.NDArray[np.float64]") -> "QtGui.QPixmap":
    """Return a pixmap of the icon of a viewer: a curve over a dark square.

    curve gives the height of the curve from the top of the icon, as a part of its size. The window gives the icon
    to the Dock.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtGui.QColor("#1b2a41"))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QtCore.QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    path = QtGui.QPainterPath()
    xvalues = np.linspace(0.1, 0.9, len(curve))
    path.moveTo(float(xvalues[0]) * size, float(curve[0]) * size)
    for xvalue, yvalue in zip(xvalues[1:].tolist(), curve[1:].tolist(), strict=True):
        path.lineTo(xvalue * size, yvalue * size)
    painter.setPen(QtGui.QPen(QtGui.QColor("#f5a623"), size * 0.05))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.end()
    return pixmap


# the tool of macOS that reads the Info.plist of an application bundle again
LSREGISTER: t.Final = Path(
    "/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A"
    "/Support/lsregister"
)


def get_macos_bundle_executable(applicationname: str, documenttypes: "Sequence[str]") -> Path:
    """Return the Python executable in the application bundle of the viewer.

    Make the bundle if it does not exist. The bundle holds a hard link to the Python executable, thus it uses almost
    no disk space. On a different volume, it holds a copy. If the Python executable changes, this function replaces
    the link. documenttypes gives the uniform type identifiers that the Dock icon accepts, e.g. "public.folder".
    """
    import os
    import plistlib
    import shutil
    import subprocess  # ruff:ignore[suspicious-subprocess-import]

    baseexecutable = Path(sys.executable).resolve()
    contents = Path.home() / "Library" / "Caches" / "artistools" / f"{applicationname}.app" / "Contents"
    executable = contents / "MacOS" / baseexecutable.name
    if not (executable.exists() and executable.samefile(baseexecutable)):
        executable.parent.mkdir(parents=True, exist_ok=True)
        # two viewers can make the bundle at the same time, thus each file receives its final name in one step
        tmpexecutable = executable.with_name(f"{executable.name}.{os.getpid()}.tmp")
        try:
            tmpexecutable.hardlink_to(baseexecutable)
        except OSError:
            # a hard link must be on the same volume as its target
            shutil.copy2(baseexecutable, tmpexecutable)
        tmpexecutable.replace(executable)

    info = {
        "CFBundleName": applicationname,
        "CFBundleDisplayName": applicationname,
        "CFBundleIdentifier": f"io.github.artis-mcrt.artistools.{applicationname.rsplit(maxsplit=1)[-1]}",
        "CFBundleExecutable": executable.name,
        "CFBundlePackageType": "APPL",
        "NSHighResolutionCapable": True,
        # a folder or a file that the user drops on the Dock icon comes to the viewer as a QFileOpenEvent
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "ARTIS data",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": list(documenttypes),
            }
        ],
    }
    infopath = contents / "Info.plist"
    infobytes = plistlib.dumps(info)
    if not infopath.is_file() or infopath.read_bytes() != infobytes:
        tmpinfo = contents / f"Info.plist.{os.getpid()}.tmp"
        tmpinfo.write_bytes(infobytes)
        tmpinfo.replace(infopath)
        # macOS keeps the old Info.plist of a bundle until lsregister reads the bundle again
        if LSREGISTER.is_file():
            subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
                [LSREGISTER, "-f", contents.parent], check=False, capture_output=True
            )
    return executable


def relaunch_in_macos_bundle(applicationname: str, documenttypes: "Sequence[str]") -> None:
    """Run the command again from an application bundle, which gives its name to the Dock and to the menu bar.

    The Dock gives a process outside a bundle the file name of its executable, e.g. "python3.14". A process cannot
    change that name after it starts. The new process finds the packages of the virtual environment through
    __PYVENV_LAUNCHER__.
    """
    import os
    import sysconfig

    # the new process runs sys.orig_argv again, thus a call from Python code, e.g. in a notebook, continues here.
    # A framework build starts Python.app, which names each process "Python", thus a bundle has no effect
    if (
        os.environ.get(MACOS_BUNDLE_VARIABLE)
        or "--interactive" not in sys.orig_argv
        or sysconfig.get_config_var("PYTHONFRAMEWORK")
    ):
        return

    try:
        executable = get_macos_bundle_executable(applicationname, documenttypes)
    except OSError:
        # the viewer can open without the bundle, and the Dock then gives the name of the executable
        return

    sys.stdout.flush()
    sys.stderr.flush()
    environment = os.environ | {MACOS_BUNDLE_VARIABLE: "1", "__PYVENV_LAUNCHER__": sys.executable}
    argv = [str(executable), *sys.orig_argv[1:]]
    os.execve(executable, argv, environment)  # ruff:ignore[start-process-with-no-shell]


def serialise_mathtext_parser() -> None:
    """Let one thread at a time parse mathtext, e.g. a tick label or a label of the legend.

    matplotlib keeps one parser for all threads, and pyparsing keeps a packrat cache in it. The worker thread lays out
    the text of a new plot while the window thread draws the old plot. A parse in both threads then corrupted the
    parser, and it rejected valid text, e.g. the tick label 0.8 of a log axis. A parse takes about 1 ms, thus
    the window thread waits for one parse and not for the whole plot.
    """
    import matplotlib.mathtext as mplmathtext

    parse = mplmathtext.MathTextParser.parse
    # wraps gives the new function __wrapped__, thus a second window does not wrap the parse again
    if hasattr(parse, "__wrapped__"):
        return
    lock = threading.Lock()

    @wraps(parse)
    def serialised_parse(
        self: mplmathtext.MathTextParser[t.Any],
        s: str,
        dpi: float = 72,
        prop: "FontProperties | None" = None,
        *,
        antialiased: bool | None = None,
    ) -> t.Any:
        with lock:
            return parse(self, s, dpi, prop, antialiased=antialiased)

    mplmathtext.MathTextParser.parse = serialised_parse  # ty:ignore[invalid-assignment]


def start_application(
    applicationname: str, iconcurve: "npt.NDArray[np.float64]", documenttypes: "Sequence[str]" = ("public.folder",)
) -> "QtWidgets.QApplication":
    """Return the Qt application of a viewer, with the name and the icon of the viewer.

    The window is a Qt window with native controls. The Qt canvas of matplotlib draws at the pixel ratio
    of the screen, thus the plot has the full resolution of a Retina display. documenttypes gives the types of the
    items that the Dock icon accepts on macOS.
    """
    import os

    if sys.platform == "darwin":
        relaunch_in_macos_bundle(applicationname, documenttypes)

    import_optional("PySide6.QtWidgets")
    import matplotlib.pyplot as plt
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # Qt stops the process with no Python error when it cannot find a display
    if sys.platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        exit_with_error(
            "--interactive needs a window, and this computer has no display",
            "Run the command on a computer with a display, e.g. with ssh -X",
        )

    # the Save command runs the command, which makes a pyplot figure. A pyplot window must not open beside the viewer
    plt.switch_backend("agg")
    serialise_mathtext_parser()
    # a worker thread draws each plot and hides its output, and the window thread still prints to the terminal
    if not isinstance(sys.stdout, ThreadOutput):
        sys.stdout = ThreadOutput(sys.stdout)
    if not isinstance(sys.stderr, ThreadOutput):
        sys.stderr = ThreadOutput(sys.stderr)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    app.setOrganizationName("artistools")
    app.setApplicationName("artistools")
    app.setApplicationDisplayName(applicationname)
    app.setWindowIcon(QtGui.QIcon(make_icon_pixmap(512, iconcurve)))

    arrowkeys = {
        QtCore.Qt.Key.Key_Left,
        QtCore.Qt.Key.Key_Right,
        QtCore.Qt.Key.Key_Up,
        QtCore.Qt.Key.Key_Down,
        QtCore.Qt.Key.Key_Home,
        QtCore.Qt.Key.Key_End,
        QtCore.Qt.Key.Key_PageUp,
        QtCore.Qt.Key.Key_PageDown,
    }

    class ApplicationFilter(QtCore.QObject):
        """Handle the events of the application that the windows need, in one filter.

        Qt calls an application filter for each event of each object. A worker thread holds the GIL during a plot,
        and each call of Python then waits for it. Thus the application has one filter and not one for each task:

        - give a key to the widget with the focus when that widget uses the key, and not to a window shortcut;
        - keep the text field that the user confirmed last, for the mark of a rejected change;
        - record the time of a quit;
        - give a path from the Dock icon to the handler of handle_file_open_events.
        """

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            eventtype = event.type()
            if eventtype == QtCore.QEvent.Type.ShortcutOverride:
                if isinstance(event, QtGui.QKeyEvent) and widget_uses_key(event):
                    event.accept()
                    return True
            elif eventtype in {QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.FocusOut}:
                keep_edited_field(watched, event)
            elif eventtype == QtCore.QEvent.Type.Quit:
                app.setProperty("quittime", time.monotonic())
            elif eventtype == QtCore.QEvent.Type.FileOpen:
                handler = app.property("fileopenhandler")
                if isinstance(event, QtGui.QFileOpenEvent) and callable(handler):
                    handler(event.file())
                    return True
            return super().eventFilter(watched, event)

    def widget_uses_key(event: QtGui.QKeyEvent) -> bool:
        """Return True if the widget with the focus uses the key, e.g. the Up key of a spin box.

        These widgets use the keys, but the shortcuts of the window took the keys from them:

        - a spin box;
        - a combo box;
        - a slider;
        - a list;
        - a popup, e.g. the list of names of a completer;
        - a button, which uses only the space key;
        - a text field or a text box with selected text, which uses the Copy key.

        For example, the Up key in -maxseriescount made the time range wider, and the Copy key in the Command box
        copied the figure.
        """
        focuswidget = QtWidgets.QApplication.focusWidget()
        # the field of a completer keeps the focus while its popup shows, and the popup takes the keys
        usesarrows = QtWidgets.QApplication.activePopupWidget() is not None or isinstance(
            focuswidget,
            QtWidgets.QAbstractSpinBox | QtWidgets.QComboBox | QtWidgets.QAbstractSlider | QtWidgets.QAbstractItemView,
        )
        usesspace = usesarrows or isinstance(focuswidget, QtWidgets.QAbstractButton)
        hastextselection = (
            isinstance(focuswidget, QtWidgets.QPlainTextEdit | QtWidgets.QTextEdit)
            and focuswidget.textCursor().hasSelection()
        ) or (isinstance(focuswidget, QtWidgets.QLineEdit) and focuswidget.hasSelectedText())
        key = event.key()
        keyowners = (
            (usesarrows, key in arrowkeys),
            (usesspace, key == QtCore.Qt.Key.Key_Space),
            (hastextselection, event.matches(QtGui.QKeySequence.StandardKey.Copy)),
        )
        return any(widgetuses and iskey for widgetuses, iskey in keyowners)

    def keep_edited_field(watched: QtCore.QObject, event: QtCore.QEvent) -> None:
        """Keep the text field that the user confirmed last, and the time, in two properties of its window.

        A plot that rejects the change of the field then marks the field, as a form of macOS does. The property holds
        a weak reference, because a QObject in a property is a raw pointer. A read of that pointer after Qt deleted
        the field crashed the process.
        """
        if not isinstance(watched, QtWidgets.QLineEdit):
            return
        isreturn = isinstance(event, QtGui.QKeyEvent) and event.key() in {
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        }
        confirms = isreturn if event.type() == QtCore.QEvent.Type.KeyPress else watched.isModified()
        if confirms:
            window = watched.window()
            window.setProperty("lasteditedfield", weakref.ref(watched))
            window.setProperty("lastedittime", time.monotonic())

    app.installEventFilter(ApplicationFilter(app))
    apply_appearance()
    return app


# a change that comes less than this time after the user confirmed a text field belongs to that field
EDIT_SECONDS: t.Final = 1.0


def get_edited_field(window: "QtCore.QObject") -> "QtWidgets.QLineEdit | None":
    """Return the text field that the user confirmed just now in the window, or None."""
    edittime = window.property("lastedittime")
    # each change of the user calls this function, thus a change with no recent edit returns before the import
    if not isinstance(edittime, float) or time.monotonic() - edittime >= EDIT_SECONDS:
        return None
    from PySide6 import QtWidgets

    fieldref = window.property("lasteditedfield")
    field = fieldref() if isinstance(fieldref, weakref.ref) else None
    return field if isinstance(field, QtWidgets.QLineEdit) and is_live(field) else None


def is_live(qobject: "QtCore.QObject") -> bool:
    """Return True if Qt did not delete the object, e.g. a field of a subplot card that show_subplots replaced."""
    import shiboken6

    return shiboken6.isValid(qobject)


def mark_field_error(field: "QtWidgets.QLineEdit", message: str) -> None:
    """Give the field a red border and the message beside it, as a form of macOS does."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    field.setStyleSheet("QLineEdit { border: 2px solid firebrick; border-radius: 3px; }")
    QtWidgets.QToolTip.showText(field.mapToGlobal(QtCore.QPoint(0, field.height())), message, field)


def clear_field_error(field: "QtWidgets.QLineEdit") -> None:
    """Remove the mark of mark_field_error, unless Qt deleted the field."""
    if is_live(field):
        field.setStyleSheet("")


def get_settings() -> "QtCore.QSettings":
    """Return the settings of the viewers, which keep the geometry of each window and the last model folder.

    The settings take the organisation and the name of the application, thus a test can send them to a folder with
    QSettings.setPath.
    """
    from PySide6 import QtCore

    return QtCore.QSettings()


def get_float_setting(key: str, default: float) -> float:
    """Return a number of the settings, or the default if the settings do not hold one."""
    value = get_settings().value(key, defaultValue=default, type=float)
    return value if isinstance(value, float) else default


def get_list_setting(key: str) -> list[str]:
    """Return a list of texts of the settings, or an empty list if the settings do not hold one."""
    value = get_settings().value(key, [])
    # the INI format of QSettings gives a list of one item as a string
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value] if isinstance(value, list) else []


def get_bool_setting(key: str, *, default: bool) -> bool:
    """Return a choice of the settings, or the default if the settings do not hold one."""
    value = get_settings().value(key, defaultValue=default, type=bool)
    return value if isinstance(value, bool) else default


def make_window(applicationname: str) -> "QtWidgets.QMainWindow":
    """Return the window of a viewer, which keeps its geometry and the sizes of its splitter when it closes.

    show_window restores them for the next window of the same viewer.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class ViewerWindow(QtWidgets.QMainWindow):
        """A window that writes its geometry and the state of its splitter to the settings when it closes.

        The window also takes the folders and the files that the user drops on it. It gives their paths to the
        function in its property "drophandler", which set_drop_handler sets.
        """

        def __init__(self, applicationname: str) -> None:
            super().__init__()
            # the name of the object gives the keys of the settings of the window
            self.setObjectName(applicationname)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self.setAcceptDrops(True)

        @t.override
        def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
            if get_dropped_paths(event.mimeData()) and callable(self.property("drophandler")):
                event.acceptProposedAction()

        @t.override
        def dropEvent(self, event: QtGui.QDropEvent) -> None:
            handler = self.property("drophandler")
            if (paths := get_dropped_paths(event.mimeData())) and callable(handler):
                event.acceptProposedAction()
                handler(paths)

        @t.override
        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            geometrykey, splitterkey = get_window_setting_keys(self)
            settings = get_settings()
            settings.setValue(geometrykey, self.saveGeometry())
            app = QtWidgets.QApplication.instance()
            isquitting = app is not None and app.property("quittime") is not None
            if isquitting and callable(get_tokens := self.property("sessiontokens")):
                add_session_window(get_tokens())
            splitter = self.centralWidget()
            if isinstance(splitter, QtWidgets.QSplitter):
                settings.setValue(splitterkey, splitter.saveState())
            super().closeEvent(event)

    return ViewerWindow(applicationname)


def get_dropped_paths(mimedata: "QtCore.QMimeData") -> list[str]:
    """Return the local paths of the folders and the files of a drop, e.g. from the Finder."""
    return [url.toLocalFile() for url in mimedata.urls() if url.isLocalFile()] if mimedata.hasUrls() else []


def set_drop_handler(window: "QtWidgets.QMainWindow", handler: "Callable[[list[str]], None]") -> None:
    """Give the window the function that receives the paths that the user drops on the window."""
    window.setProperty("drophandler", handler)


def handle_file_open_events(app: "QtWidgets.QApplication", open_folder: "Callable[[str], None]") -> None:
    """Open a folder from the Dock icon in a new window, and give a file to the active window.

    macOS gives such an item to the application as a QFileOpenEvent, and not to a window. At the start, macOS also
    gives each path of the command line as such an event, and the first window already shows those paths. Thus the
    filter ignores the first event of each of those paths. A later drop of the same path opens it as usual.
    """
    from PySide6 import QtWidgets

    launchpaths = {str(Path(argument).absolute()) for argument in sys.orig_argv}

    def open_path(path: str) -> None:
        if (absolutepath := str(Path(path).absolute())) in launchpaths:
            launchpaths.remove(absolutepath)
            return
        activewindow = QtWidgets.QApplication.activeWindow()
        handler = activewindow.property("drophandler") if activewindow is not None else None
        if Path(path).is_dir() or not callable(handler):
            open_folder(path)
        else:
            handler([path])

    # the filter of start_application receives the event and gives the path to this function
    app.setProperty("fileopenhandler", open_path)


# File > Open Recent shows this number of models
RECENT_LIMIT: t.Final = 10


def get_recent_setting_key() -> str:
    """Return the key of the settings that holds the recent models of this viewer."""
    from PySide6 import QtWidgets

    return f"{QtWidgets.QApplication.applicationDisplayName()}/recentmodels"


def get_recent_models() -> list[str]:
    """Return the folders of the models that the viewer opened last, the newest first."""
    return get_list_setting(get_recent_setting_key())


def add_recent_model(folder: Path | str) -> None:
    """Put the folder of a model at the start of the recent models of File > Open Recent."""
    # resolve_modelpath gives ".." the real name of its folder, thus two spellings of one folder give one entry in the
    # list. A model on a different host keeps its path
    path = str(resolve_modelpath(folder))
    recent = [path, *(other for other in get_recent_models() if other != path)][:RECENT_LIMIT]
    get_settings().setValue(get_recent_setting_key(), recent)


def get_session_setting_key() -> str:
    """Return the key of the settings that holds the commands of the windows that were open at the last quit."""
    from PySide6 import QtWidgets

    return f"{QtWidgets.QApplication.applicationDisplayName()}/session"


def add_session_window(tokens: "Sequence[str]") -> None:
    """Keep the command of a window that closes because the application quits.

    Each path of the command that exists in the working folder becomes an absolute path, because the next start can
    be in a different folder. A name of a reference file in the data of artistools stays a name.
    """
    get_settings().setValue(
        get_session_setting_key(),
        [*get_list_setting(get_session_setting_key()), json.dumps(get_absolute_tokens(tokens))],
    )


def get_absolute_tokens(tokens: "Sequence[str]") -> list[str]:
    """Return the tokens of a command with an absolute path for each path that exists in the working folder."""
    return [str(Path(word).absolute()) if not word.startswith("-") and Path(word).exists() else word for word in tokens]


def take_session_windows() -> list[list[str]]:
    """Return the commands of the windows that were open at the last quit, and remove them from the settings.

    If the setting "reopenwindows" of the Settings window is off, the list is empty.
    """
    saved = get_list_setting(get_session_setting_key())
    get_settings().remove(get_session_setting_key())
    if not get_bool_setting("reopenwindows", default=True):
        return []
    return [[str(token) for token in json.loads(item)] for item in saved]


def reopen_session_windows(
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    windows: "list[QtWidgets.QMainWindow]",
) -> None:
    """Open the windows of the last session, as the apps of macOS do, and keep the first window in front.

    A window with the same command as an open window does not open again. The comparison leaves out -figwidthscale,
    because each window fits it to its own size. An error of a window goes to the terminal. Each window opens after
    the event loop runs again, thus the first window can draw and take input while the others read their runs.
    """
    from PySide6 import QtCore

    shown = [
        remove_figwidthscale(get_absolute_tokens(window.property("sessiontokens")()))
        for window in windows
        if callable(window.property("sessiontokens"))
    ]
    pending = [tokens for tokens in take_session_windows() if remove_figwidthscale(tokens) not in shown]
    firstwindows = list(windows)

    def open_next_window() -> None:
        if not pending:
            for window in firstwindows:
                if is_live(window):
                    activate_window(window)
            return
        tokens = pending.pop(0)
        message = run_command_step(lambda: open_window(tokens, windows), quiet=False)
        if message is not None:
            print_error(f"The viewer cannot open the window of the last session: {message}")
        QtCore.QTimer.singleShot(0, open_next_window)

    if pending:
        QtCore.QTimer.singleShot(0, open_next_window)


def run_viewer_application(
    applicationname: str,
    iconcurve: "npt.NDArray[np.float64]",
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    tokens: "Sequence[str]",
    documenttypes: "Sequence[str]" = ("public.folder",),
) -> None:
    """Open a viewer window for the tokens and the windows of the last session, then run until the user quits.

    A folder from the Dock icon opens in a new window, and its errors go to the terminal.
    """
    app = start_application(applicationname, iconcurve, documenttypes)
    # the list holds a reference to each window, thus Python keeps the window while it is open
    windows: list[QtWidgets.QMainWindow] = []

    def open_dock_folder(folder: str) -> None:
        if (message := open_model_folder(folder, open_window, windows)) is not None:
            print_error(message)

    handle_file_open_events(app, open_dock_folder)
    open_window(tokens, windows)
    reopen_session_windows(open_window, windows)
    app.exec()


def remove_figwidthscale(tokens: "Sequence[str]") -> list[str]:
    """Return the tokens of a command without -figwidthscale and its value, which a window sets."""
    return [
        word
        for index, word in enumerate(tokens)
        if word != "-figwidthscale" and (index == 0 or tokens[index - 1] != "-figwidthscale")
    ]


def get_window_setting_keys(window: "QtWidgets.QMainWindow") -> tuple[str, str]:
    """Return the keys of the settings of the geometry and of the splitter of the window of a viewer."""
    return f"{window.objectName()}/geometry", f"{window.objectName()}/splitter"


def make_central_splitter(
    window: "QtWidgets.QMainWindow", plotarea: "QtWidgets.QWidget", sidebar: "QtWidgets.QWidget"
) -> "QtWidgets.QSplitter":
    """Put the plot area and the sidebar side by side at the centre of the window, with a handle between them.

    A drag of the handle changes the width of the sidebar, and a drag to the edge hides it. The plot area takes the
    extra width of the window.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    splitter.addWidget(plotarea)
    splitter.addWidget(sidebar)
    # the sidebar keeps its width and a drag to the edge hides it, which are the defaults of Qt for the sidebar
    splitter.setStretchFactor(0, 1)
    splitter.setCollapsible(0, False)  # ruff:ignore[boolean-positional-value-in-call]
    window.setCentralWidget(splitter)
    return splitter


@contextlib.contextmanager
def show_wait_cursor() -> "Generator[None]":
    """Show the wait cursor until the block ends, e.g. while a step that can take seconds runs in the window thread."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


# a section with one of these titles starts closed, because a user needs it less often than the plot controls
CLOSED_SECTIONS: t.Final = frozenset({"Other options", "Command", "Python"})


def add_section(
    panellayout: "QtWidgets.QVBoxLayout", title: str, key: str | None = None
) -> "tuple[QtWidgets.QToolButton, QtWidgets.QGridLayout]":
    """Add a section with a heading and a grid for its controls to the panel of the window.

    A click on the heading closes or opens the section, as a disclosure triangle does in the inspector of Keynote. The
    settings keep the state of each section by its key, which is the title if the caller gives no key.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    header = QtWidgets.QToolButton()
    header.setText(title)
    header.setCheckable(True)
    header.setAutoRaise(True)
    header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    header.setStyleSheet("QToolButton { border: none; }")
    # the macOS style gives a tool button a small font, and a heading takes the bold font of the application
    font = QtWidgets.QApplication.font()
    font.setBold(True)
    header.setFont(font)
    content = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(content)
    # the space under the content of a section is larger than the space between its rows, thus each section stays a
    # group of its own
    grid.setContentsMargins(12, 2, 4, 12)
    grid.setVerticalSpacing(ROW_SPACING)
    grid.setHorizontalSpacing(8)
    grid.setColumnStretch(1, 1)
    settingkey = f"{QtWidgets.QApplication.applicationDisplayName()}/sections/{key or title}"
    isopen = get_bool_setting(settingkey, default=title not in CLOSED_SECTIONS)

    def set_open(checked: bool) -> None:
        header.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
        header.setToolTip(f"{'Close' if checked else 'Open'} the section")
        content.setVisible(checked)
        get_settings().setValue(settingkey, checked)

    # the stretch at the end of the panel keeps each section, also Command and Python, below the last one
    index = panellayout.count() - 1
    panellayout.insertWidget(index, header)
    # a widget with no parent shows as a window of its own, thus the content goes into the panel before set_open
    panellayout.insertWidget(index + 1, content)
    header.toggled.connect(set_open)
    header.setChecked(isopen)
    set_open(isopen)
    return header, grid


# a button that shows one symbol, e.g. ✕ or ▲, has no frame, and a light background shows under the pointer, as the
# small buttons of the apps of macOS have. The grey with an alpha suits the light and the dark appearance
GLYPH_BUTTON_STYLE: t.Final = (
    "QToolButton { border: none; background: transparent; padding: 1px 4px; border-radius: 4px; }"
    " QToolButton:hover { background: rgba(128, 128, 128, 60); }"
    " QToolButton:pressed { background: rgba(128, 128, 128, 110); }"
)

# the space between the rows of a section, and the space in front of a group, e.g. a label, that follows a control
ROW_SPACING: t.Final = 6
LABEL_GAP: t.Final = 12


# the line styles of the style dialog: the value of -linestyle and the text of the box
LINESTYLE_CHOICES: t.Final = (("solid", "Solid"), ("dashed", "Dashed"), ("dotted", "Dotted"), ("dashdot", "Dash-dot"))

# the short names of the line styles of matplotlib, which a command can also give
LINESTYLE_ALIASES: t.Final = MappingProxyType({"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot"})


def get_dash_pattern(linestyle: str | None, dashes: str | None) -> list[float] | None:
    """Return the dash pattern of a line in units of its width, as matplotlib draws it, or None for a solid line.

    -dashes gives the pattern and replaces the line style. matplotlib scales a pattern by the width of the line, and a
    pattern of Qt has the same unit.
    """
    import matplotlib as mpl

    if dashes:
        with contextlib.suppress(ValueError, argparse.ArgumentTypeError):
            return list(dashes_arg(dashes))
    match LINESTYLE_ALIASES.get(linestyle or "", linestyle or "solid"):
        case "dashed":
            pattern = mpl.rcParams["lines.dashed_pattern"]
        case "dotted":
            pattern = mpl.rcParams["lines.dotted_pattern"]
        case "dashdot":
            pattern = mpl.rcParams["lines.dashdot_pattern"]
        case _:
            return None
    return [float(length) for length in pattern]


def make_line_swatch(
    colour: str, alpha: float, linewidth: float, dashpattern: "Sequence[float] | None"
) -> "QtGui.QPixmap":
    """Return a short image of a line with the colour, the opacity, the width, and the dash pattern of a series."""
    import matplotlib.colors as mplcolors
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    width, height = 36, 14
    ratio = QtWidgets.QApplication.primaryScreen().devicePixelRatio() if QtWidgets.QApplication.primaryScreen() else 1.0
    pixmap = QtGui.QPixmap(round(width * ratio), round(height * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    red, green, blue = (round(255 * part) for part in mplcolors.to_rgb(colour))
    pen = QtGui.QPen(QtGui.QColor(red, green, blue, round(255 * alpha)))
    # a very wide line fills the image, thus the width stays inside the height of the image
    pen.setWidthF(min(max(linewidth, 0.5), 5.0))
    # matplotlib ends each dash with no cap, and the square cap of Qt closes a short gap
    pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
    if dashpattern:
        pen.setDashPattern(list(dashpattern))
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(pen)
    painter.drawLine(QtCore.QPointF(2.0, height / 2.0), QtCore.QPointF(width - 2.0, height / 2.0))
    painter.end()
    return pixmap


def make_series_swatch(colour: str, style: "Mapping[str, str | None]") -> "QtGui.QPixmap":
    """Return the image of the line of a series from its colour and the values of its series style options."""
    import matplotlib as mpl

    return make_line_swatch(
        colour,
        float(style.get("-linealpha") or 1.0),
        float(style.get("-linewidth") or mpl.rcParams["lines.linewidth"]),
        get_dash_pattern(style.get("-linestyle"), style.get("-dashes")),
    )


# the options of the style dialog of a series, which each give one value for each series
SERIES_LINE_FLAGS: t.Final = ("-color", "-linestyle", "-dashes", "-linewidth", "-linealpha")


def edit_series_style(
    parent: "QtWidgets.QWidget",
    name: str,
    style: "Mapping[str, str | None]",
    defaultcolour: str,
    defaultlinewidth: float,
    flags: "Collection[str]" = SERIES_LINE_FLAGS,
) -> dict[str, str | None] | None:
    """Ask for the line style of one series, and return the value of each option of flags.

    style gives the current value of each option, or None for the default of the command. A field with the default
    gives None, thus the command then gives the series no value. A cancelled dialog gives None. The dialog shows only
    the fields of flags, e.g. a series of markers has no line style.
    """
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(f"Style of {name}")
    form = QtWidgets.QFormLayout(dialog)
    chosencolour: list[str | None] = [style.get("-color")]

    colourbutton = QtWidgets.QPushButton()
    colourbutton.setToolTip("Select the colour of the line (-color)")
    defaultcolourbutton = QtWidgets.QPushButton("Default")
    defaultcolourbutton.setToolTip("Give the line the colour of the command")
    colourrow = QtWidgets.QHBoxLayout()
    colourrow.addWidget(colourbutton, 1)
    colourrow.addWidget(defaultcolourbutton)
    form.addRow("Colour:", colourrow)

    linestylebox = QtWidgets.QComboBox()
    linestylebox.addItem("Default", None)
    for value, text in LINESTYLE_CHOICES:
        linestylebox.addItem(text, value)
    linestyle = style.get("-linestyle")
    linestylebox.setCurrentIndex(max(linestylebox.findData(LINESTYLE_ALIASES.get(linestyle or "", linestyle)), 0))
    linestylebox.setToolTip("The line style (-linestyle). A dash pattern replaces it")
    form.addRow("Line style:", linestylebox)

    dashesedit = QtWidgets.QLineEdit(style.get("-dashes") or "")
    dashesedit.setPlaceholderText("none, e.g. 5,2")
    dashesedit.setToolTip(
        "The lengths of each dash and each gap in units of the line width (-dashes), e.g. 5,2. The pattern replaces"
        " the line style"
    )
    form.addRow("Dash pattern:", dashesedit)

    def make_default_spinbox(high: float, step: float, value: str | None, tooltip: str) -> QtWidgets.QDoubleSpinBox:
        # the lowest value of the box shows "Default", which gives the series no value
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(0.0, high)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setSpecialValueText("Default")
        box.setValue(float(value) if value else 0.0)
        box.setToolTip(tooltip)
        return box

    widthbox = make_default_spinbox(10.0, 0.25, style.get("-linewidth"), "The width of the line in points (-linewidth)")
    form.addRow("Width:", widthbox)
    alphabox = make_default_spinbox(
        1.0, 0.05, style.get("-linealpha"), "The opacity of the line, from 0 (clear) to 1 (opaque) (-linealpha)"
    )
    form.addRow("Opacity:", alphabox)

    previewlabel = QtWidgets.QLabel()
    form.addRow("Preview:", previewlabel)
    errorlabel = QtWidgets.QLabel()
    errorlabel.setStyleSheet("color: red;")
    errorlabel.hide()
    form.addRow(errorlabel)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
    )
    form.addRow(buttons)

    def get_dashes() -> str | None:
        """Return the dash pattern of the field, or None for an empty field. Raise ValueError for a bad pattern."""
        text = dashesedit.text().strip()
        if not text:
            return None
        try:
            return ",".join(format(length, "g") for length in dashes_arg(text))
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc

    def show_preview() -> None:
        colour = chosencolour[0] or defaultcolour
        colourbutton.setIcon(QtGui.QIcon(make_line_swatch(colour, 1.0, 5.0, None)))
        colourbutton.setText(colour if chosencolour[0] else f"Default ({colour})")
        try:
            dashes = get_dashes()
        except ValueError as exc:
            errorlabel.setText(str(exc))
            errorlabel.show()
            dashes = None
        else:
            errorlabel.hide()
        previewlabel.setPixmap(
            make_line_swatch(
                colour,
                alphabox.value() or 1.0,
                widthbox.value() or defaultlinewidth,
                get_dash_pattern(linestylebox.currentData(), dashes) if "-linestyle" in flags else None,
            )
        )

    def on_colour() -> None:
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(chosencolour[0] or defaultcolour), dialog, "Line colour")
        if colour.isValid():
            chosencolour[0] = colour.name()
            show_preview()

    def on_default_colour() -> None:
        chosencolour[0] = None
        show_preview()

    def on_restore_defaults() -> None:
        chosencolour[0] = None
        linestylebox.setCurrentIndex(0)
        dashesedit.clear()
        widthbox.setValue(0.0)
        alphabox.setValue(0.0)
        show_preview()

    def on_accept() -> None:
        # a bad dash pattern keeps the dialog open, and the red text gives the reason
        try:
            get_dashes()
        except ValueError:
            dashesedit.setFocus()
            return
        dialog.accept()

    colourbutton.clicked.connect(on_colour)
    defaultcolourbutton.clicked.connect(on_default_colour)
    linestylebox.currentIndexChanged.connect(show_preview)
    dashesedit.textChanged.connect(show_preview)
    widthbox.valueChanged.connect(show_preview)
    alphabox.valueChanged.connect(show_preview)
    buttons.accepted.connect(on_accept)
    buttons.rejected.connect(dialog.reject)
    if (restorebutton := buttons.button(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults)) is not None:
        restorebutton.clicked.connect(on_restore_defaults)
    for flag, field in (
        ("-color", colourrow),
        ("-linestyle", linestylebox),
        ("-dashes", dashesedit),
        ("-linewidth", widthbox),
        ("-linealpha", alphabox),
    ):
        form.setRowVisible(field, flag in flags)
    show_preview()

    accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    # the parent keeps its children until it closes, thus the dialog goes when the event loop runs again
    dialog.deleteLater()
    if not accepted:
        return None
    values = {
        "-color": chosencolour[0],
        "-linestyle": linestylebox.currentData(),
        "-dashes": get_dashes(),
        "-linewidth": format(widthbox.value(), "g") if widthbox.value() else None,
        "-linealpha": format(alphabox.value(), "g") if alphabox.value() else None,
    }
    return {flag: value for flag, value in values.items() if flag in flags}


# each kind of viewing direction: the value of the kind, the text of its choice, and the option that gives its help
DIRECTION_KINDS: t.Final = (
    ("", "All directions", ""),
    ("bin", "-plotviewingangle", "plotviewingangle"),
    ("phi", "--average_over_phi_angle", "average_over_phi_angle"),
    ("theta", "--average_over_theta_angle", "average_over_theta_angle"),
    ("vpkt", "-plotvspecpol", "plotvspecpol"),
)


class DirectionChoice(t.NamedTuple):
    """The viewing direction of the values of a window: the kind, the bins, and the unit of the angles."""

    kind: str
    bins: tuple[int, ...]
    usedegrees: bool


def get_direction_bins(runfolder: Path | str, kind: str, *, usedegrees: bool) -> list[tuple[int, str]]:
    """Return each bin of a kind of viewing direction with its label, and first the average over all the directions.

    The average is bin -1. An observer of the virtual packets has no such average.
    """
    if not kind:
        return []
    averagebin = [] if kind == "vpkt" else [(-1, "All directions")]
    return [*averagebin, *get_direction_choices(Path(runfolder), kind, usedegrees=usedegrees)]


def add_direction_section(
    panellayout: "QtWidgets.QVBoxLayout",
    helptexts: "Mapping[str, str]",
    runfolder: Path | str,
    get_choice: "Callable[[], tuple[DirectionChoice, bool]]",
    on_change: "Callable[[DirectionChoice], None]",
    show_error: "Callable[[str], None]",
) -> "tuple[Callable[[], None], Callable[[Path | str], None]]":
    """Add the section of the viewing direction: the kind of direction, --usedegrees, and a list of the bins.

    get_choice gives the choice of the values of the window, and whether the plot draws one bin, e.g. an emission plot.
    Such a plot has a radio button for each bin, and a click on a bin replaces the bin. on_change receives each new
    choice. A new kind keeps each bin that the kind also has.

    Return the function that shows the choice of get_choice, and the function that reads the kinds of direction and
    the labels of the bins of a new first run.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    _, directiongrid = add_section(panellayout, "Viewing direction")
    kindbox = QtWidgets.QComboBox()
    usedegreescheck = QtWidgets.QCheckBox("--usedegrees")
    usedegreescheck.setToolTip(helptexts.get("usedegrees", ""))
    add_row(directiongrid, 0, [kindbox, usedegreescheck])
    # the plot can show several directions at once, thus each bin has a check box. The list scrolls, and the label of
    # a bin is long, thus the list takes the full width of the sidebar
    binbox = QtWidgets.QScrollArea()
    binbox.setWidgetResizable(True)
    binbox.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    binbox.setToolTip(
        "The direction bins of the plot, or the observers of the virtual packets. A plot that draws one bin, e.g. an"
        " emission plot, shows a radio button for each bin"
    )
    directiongrid.addWidget(binbox, 1, 0, 1, -1)
    binchecks: dict[int, QtWidgets.QAbstractButton] = {}
    # the labels of the bins come from the files of the run, thus the section reads them one time for each kind
    binlabels: dict[tuple[str, bool], list[tuple[int, str]]] = {}
    shownlist: tuple[str, bool, bool] | None = None
    run: list[Path | str] = [runfolder]

    def get_bins(kind: str, usedegrees: bool) -> list[tuple[int, str]]:
        if (kind, usedegrees) not in binlabels:
            binlabels[kind, usedegrees] = get_direction_bins(run[0], kind, usedegrees=usedegrees)
        return binlabels[kind, usedegrees]

    def set_run(newrunfolder: Path | str) -> None:
        nonlocal shownlist
        run[0], shownlist = newrunfolder, None
        binlabels.clear()
        kinds = get_direction_kinds(Path(newrunfolder))
        with QtCore.QSignalBlocker(kindbox):
            kindbox.clear()
            for kind, text, dest in DIRECTION_KINDS:
                if kind in kinds:
                    kindbox.addItem(text, kind)
                    kindbox.setItemData(
                        kindbox.count() - 1, helptexts.get(dest, ""), QtCore.Qt.ItemDataRole.ToolTipRole
                    )

    def show_bins(kind: str, usedegrees: bool, onebin: bool) -> bool:
        """Fill the list with a button for each bin of a kind of direction. Return whether the list is new."""
        nonlocal shownlist
        if (kind, usedegrees, onebin) == shownlist:
            return False
        checklist = QtWidgets.QWidget()
        checklayout = QtWidgets.QVBoxLayout(checklist)
        checklayout.setContentsMargins(6, 4, 6, 4)
        checklayout.setSpacing(2)
        binchecks.clear()
        for dirbin, label in get_bins(kind, usedegrees):
            text = f"{dirbin}: {label}"
            check = QtWidgets.QRadioButton(text) if onebin else QtWidgets.QCheckBox(text)
            # a click on a radio button also clears the previous button, thus the toggled signal calls the handler two times
            check.clicked.connect(on_direction)
            checklayout.addWidget(check)
            binchecks[dirbin] = check
        checklayout.addStretch(1)
        # the list shows up to 6 bins, and a longer list scrolls
        shownbins = min(max(len(binchecks), 1), 6)
        lineheight = max((check.sizeHint().height() for check in binchecks.values()), default=20)
        binbox.setFixedHeight(shownbins * (lineheight + 2) + 10)
        binbox.setWidget(checklist)
        shownlist = (kind, usedegrees, onebin)
        return True

    def show() -> None:
        choice, onebin = get_choice()
        with QtCore.QSignalBlocker(kindbox), QtCore.QSignalBlocker(usedegreescheck):
            kindbox.setCurrentIndex(max(kindbox.findData(choice.kind), 0))
            usedegreescheck.setChecked(choice.usedegrees)
        usedegreescheck.setEnabled(bool(choice.kind))
        isnewlist = show_bins(choice.kind, choice.usedegrees, onebin)
        for dirbin, check in binchecks.items():
            with QtCore.QSignalBlocker(check):
                check.setChecked(dirbin in choice.bins)
        # a new list scrolls to the first checked bin, which can be far down a list of 100 bins
        if isnewlist and choice.bins and (firstcheck := binchecks.get(choice.bins[0])):
            QtCore.QTimer.singleShot(0, binbox, partial(binbox.ensureWidgetVisible, firstcheck))
        # all the directions have no bin to select, thus the list shows only for a kind of direction
        binbox.setVisible(bool(choice.kind))

    def on_direction() -> None:
        kind: str = kindbox.currentData()
        usedegrees = usedegreescheck.isChecked()
        current, onebin = get_choice()
        if kind == current.kind:
            bins = tuple(dirbin for dirbin, check in binchecks.items() if check.isChecked())
            if kind and not bins:
                show_error("A kind of viewing direction needs one direction bin at least")
                return
            newbins = tuple(dirbin for dirbin in bins if dirbin not in current.bins)
            # a plot of one bin replaces the previous bin with the bin of the click
            if onebin and newbins:
                bins = newbins[:1]
        else:
            kindbins = [dirbin for dirbin, _ in get_bins(kind, usedegrees)]
            bins = tuple(dirbin for dirbin in current.bins if dirbin in kindbins) or tuple(kindbins[:1])
            if onebin:
                bins = bins[:1]
        on_change(DirectionChoice(kind=kind, bins=bins, usedegrees=usedegrees))

    kindbox.currentIndexChanged.connect(on_direction)
    usedegreescheck.toggled.connect(on_direction)
    set_run(runfolder)
    return show, set_run


def add_y_axis_actions(
    menu: "QtWidgets.QMenu",
    islog: bool | None,
    haslimits: bool,
    set_yscale: "Callable[[str], None]",
    clear_limits: "Callable[[], None]",
) -> None:
    """Add the actions on the y axis of a frame to a context menu: the scale, and the automatic y range.

    islog is None for an axis with no choice of scale, e.g. a magnitude. haslimits enables Auto Y Range, which removes
    the limits that the user gave.
    """
    if islog is not None:
        menu.addAction("Linear Scale" if islog else "Log Scale").triggered.connect(
            partial(set_yscale, "linear" if islog else "log")
        )
    resetaction = menu.addAction("Auto Y Range")
    resetaction.setEnabled(haslimits)
    resetaction.triggered.connect(clear_limits)
    menu.addSeparator()


def read_limit_fields(
    fields: "Sequence[tuple[QtWidgets.QLineEdit, str]]", show_error: "Callable[[str], None]"
) -> list[str] | None:
    """Return the text of two limit fields as numbers, or "" for an empty field, which gives the automatic limit.

    fields gives each field with its flag, e.g. -ymin. A field with text that is not a number, or a minimum that is not
    less than the maximum, gives the reason to show_error and None.
    """
    limits: list[str] = []
    for edit, flag in fields:
        edit.setModified(False)
        text = edit.text().strip()
        try:
            limits.append(format(float(text), ".10g") if text else "")
        except ValueError:
            show_error(f"Give a number for {flag}, or clear the field for the automatic limit")
            return None
    if limits[0] and limits[1] and not float(limits[0]) < float(limits[1]):
        show_error(f"Give a {fields[0][1]} that is less than {fields[1][1]}")
        return None
    return limits


def add_y_limits_row(
    grid: "QtWidgets.QGridLayout",
    row: int,
    helptexts: "Mapping[str, str]",
    set_limits: "Callable[[str, str], None]",
    get_drawn_limits: "Callable[[], tuple[float, float] | None]",
    show_error: "Callable[[str], None]",
) -> "Callable[[str, str], None]":
    """Add the fields of -ymin and -ymax and the button Set current y range to a row of a grid.

    set_limits receives the text of each limit, and "" gives the automatic limit. get_drawn_limits gives the y limits of
    the plot on the screen, or None while the plot does not show the values of the controls. Return the function that
    shows the limits of the values.
    """
    from PySide6 import QtWidgets

    yminedit, ymaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    for edit, dest in ((yminedit, "ymin"), (ymaxedit, "ymax")):
        edit.setFixedWidth(110)
        edit.setPlaceholderText("auto")
        edit.setToolTip(helptexts.get(dest, ""))
    setrangebutton = QtWidgets.QPushButton("Set current y range")
    setrangebutton.setToolTip(
        "Set y min and y max to the current range of the y axis. The axis then stays the same when a different option"
        " changes. Clear a field to get the automatic limit at that end again."
    )
    add_row(grid, row, [QtWidgets.QLabel("-ymin"), yminedit, QtWidgets.QLabel("-ymax"), ymaxedit, setrangebutton])

    def on_edit() -> None:
        if (limits := read_limit_fields([(yminedit, "-ymin"), (ymaxedit, "-ymax")], show_error)) is not None:
            set_limits(limits[0], limits[1])

    def on_set_range() -> None:
        if (drawnlimits := get_drawn_limits()) is None:
            show_error("The plot on the screen does not show the new values yet. Wait for the plot, then try again")
            return
        # the limits of the plot on the screen become the limits of the command, thus the plot does not change. An
        # inverted axis, e.g. of a magnitude, gives the lower number to -ymin
        low, high = (get_short_number(limit) for limit in sorted(drawnlimits))
        set_limits(low, high)

    def show_limits(ymin: str, ymax: str) -> None:
        set_edit_text(yminedit, ymin)
        set_edit_text(ymaxedit, ymax)

    yminedit.editingFinished.connect(on_edit)
    ymaxedit.editingFinished.connect(on_edit)
    setrangebutton.clicked.connect(on_set_range)
    return show_limits


def make_xscale_box(helptexts: "Mapping[str, str]") -> "QtWidgets.QComboBox":
    """Return a box with the choices Linear and Log for the x axis. The index 1 gives --logscalex.

    The commands have no -xscale, thus the box has no automatic scale as the y scale box has.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    xscalebox = QtWidgets.QComboBox()
    for text, tooltip in (("Linear", "A linear x axis"), ("Log", f"--logscalex: {helptexts.get('logscalex', '')}")):
        xscalebox.addItem(text)
        xscalebox.setItemData(xscalebox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    xscalebox.setToolTip("The scale of the x axis. Log gives --logscalex")
    return xscalebox


def make_glyph_button(glyph: str, tooltip: str, accessiblename: str) -> "QtWidgets.QToolButton":
    """Return a small button that shows one symbol, e.g. ✕ to remove an item, with no frame."""
    from PySide6 import QtWidgets

    button = QtWidgets.QToolButton()
    button.setText(glyph)
    button.setStyleSheet(GLYPH_BUTTON_STYLE)
    button.setToolTip(tooltip)
    button.setAccessibleName(accessiblename)
    return button


def make_row_layout(widgets: "Sequence[QtWidgets.QWidget]") -> "QtWidgets.QVBoxLayout":
    """Return a layout that puts the widgets side by side from the left, and wraps the row in a narrow sidebar.

    A label, a checkbox, or a push button that follows a control starts a new group, e.g. a pair of a label and a
    control. A wider space goes in front of it, thus the groups stay apart. A group stays on one line, and the next
    group goes to a new line if the line is full.
    """
    from PySide6 import QtWidgets

    groupstarts = QtWidgets.QLabel | QtWidgets.QCheckBox | QtWidgets.QPushButton
    groups: list[list[QtWidgets.QWidget]] = []
    for index, widget in enumerate(widgets):
        if index == 0 or (isinstance(widget, groupstarts) and not isinstance(widgets[index - 1], QtWidgets.QLabel)):
            groups.append([])
        groups[-1].append(widget)
    groupwidgets: list[QtWidgets.QWidget] = []
    for group in groups:
        if len(group) == 1:
            groupwidgets.append(group[0])
            continue
        groupbox = QtWidgets.QWidget()
        grouplayout = QtWidgets.QHBoxLayout(groupbox)
        grouplayout.setContentsMargins(0, 0, 0, 0)
        grouplayout.setSpacing(ROW_SPACING)
        for widget in group:
            grouplayout.addWidget(widget)
        groupwidgets.append(groupbox)
    rowlayout = QtWidgets.QVBoxLayout()
    rowlayout.setContentsMargins(0, 0, 0, 0)
    rowlayout.addWidget(get_wrap_row_class()(groupwidgets))
    return rowlayout


def get_wrapped_lines(widths: "Sequence[int]", available: int, gap: int) -> list[list[int]]:
    """Return the indices of the items on each line, for items of these widths on lines of the available width.

    An item goes on a new line if the line is full. An item of no width, e.g. a hidden widget, takes no place and no
    gap. A line holds one item at least, also when the item is wider than the line.
    """
    lines: list[list[int]] = [[]]
    used = 0
    for index, width in enumerate(widths):
        if width > 0 and used > 0 and used + gap + width > available:
            lines.append([])
            used = 0
        lines[-1].append(index)
        if width > 0:
            used += width if used == 0 else gap + width
    return lines


@cache
def get_wrap_row_class() -> "Callable[[list[QtWidgets.QWidget]], QtWidgets.QWidget]":
    """Return the class of the rows of make_row_layout, which takes the widget of each group.

    A layout class of Python ran each time that Qt arranged the sidebar, e.g. at each step of a drag. Each call
    waited for the worker thread, which held the GIL during a plot, thus a text change took 156 ms and not 7 ms. The
    row puts its groups in layouts of Qt, and Python runs only when the width of the row changes.
    """
    import shiboken6
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class WrapRow(QtWidgets.QWidget):
        """The groups of a row, on as many lines as the width of the row needs."""

        def __init__(self, groups: "list[QtWidgets.QWidget]") -> None:
            super().__init__()
            self.groups = groups
            self.lines: list[list[int]] = []
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(ROW_SPACING)
            # the width of the widest group is the minimum width of the row. A wider line then wraps and does not
            # widen the sidebar. A layout of Qt uses the minimum width of a widget if it has one, or its size hint
            self.setMinimumWidth(
                max((group.minimumWidth() or group.minimumSizeHint().width() for group in groups), default=1)
            )
            self.set_lines([list(range(len(groups)))])

        def set_lines(self, lines: list[list[int]]) -> None:
            layout = self.layout()
            assert layout is not None
            while (item := layout.takeAt(0)) is not None:
                if (line := item.layout()) is not None:
                    while line.takeAt(0) is not None:
                        pass
                    shiboken6.delete(line)
            for indices in lines:
                line = QtWidgets.QHBoxLayout()
                line.setSpacing(LABEL_GAP)
                for index in indices:
                    line.addWidget(self.groups[index])
                line.addStretch(1)
                assert isinstance(layout, QtWidgets.QBoxLayout)
                layout.addLayout(line)
            self.lines = lines

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent, /) -> None:
            super().resizeEvent(event)
            widths = [0 if group.isHidden() else group.sizeHint().width() for group in self.groups]
            lines = get_wrapped_lines(widths, event.size().width(), LABEL_GAP)
            if lines != self.lines:
                self.set_lines(lines)

    return WrapRow


def add_row(grid: "QtWidgets.QGridLayout", row: int, widgets: "Sequence[QtWidgets.QWidget]") -> None:
    """Put the widgets side by side in one row of the grid, from the left."""
    grid.addLayout(make_row_layout(widgets), row, 0, 1, -1)


def make_flow_layout() -> "QtWidgets.QLayout":
    """Return a layout that puts its widgets side by side from the left, and starts a new row when a row is full.

    Qt has no such layout. A row of chips, e.g. the series of a subplot, then wraps to the width of the sidebar.
    """
    return get_flow_layout_class()()


@cache
def get_flow_layout_class() -> "type[QtWidgets.QLayout]":
    """Return the class of the layouts of make_flow_layout.

    PySide keeps about 1.5 KB of memory for each class, thus the viewers make the class one time and not for each
    layout.
    """
    import shiboken6
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    def delete_items(items: "list[QtWidgets.QLayoutItem]") -> None:
        # a layout of Qt deletes its items when Qt deletes the layout, and a layout of Python must do the same
        for item in items:
            shiboken6.delete(item)
        items.clear()

    class FlowLayout(QtWidgets.QLayout):
        """A layout that fills rows from the left, and gives each widget the size that it asks for."""

        def __init__(self) -> None:
            super().__init__()
            self.layoutitems: list[QtWidgets.QLayoutItem] = []
            self.setSpacing(4)
            self.setContentsMargins(0, 0, 0, 0)
            # the signal comes after Python lost the layout, thus the function holds the list and not the layout
            items = self.layoutitems
            self.destroyed.connect(lambda: delete_items(items))

        @t.override
        def addItem(self, arg__1: QtWidgets.QLayoutItem, /) -> None:
            self.layoutitems.append(arg__1)

        @t.override
        def count(self, /) -> int:
            return len(self.layoutitems)

        @t.override
        def itemAt(self, index: int, /) -> QtWidgets.QLayoutItem | None:
            return self.layoutitems[index] if 0 <= index < len(self.layoutitems) else None

        @t.override
        def takeAt(self, index: int, /) -> QtWidgets.QLayoutItem | None:
            return self.layoutitems.pop(index) if 0 <= index < len(self.layoutitems) else None

        @t.override
        def expandingDirections(self, /) -> QtCore.Qt.Orientation:
            return QtCore.Qt.Orientation(0)

        @t.override
        def hasHeightForWidth(self, /) -> bool:
            return True

        @t.override
        def heightForWidth(self, arg__1: int, /) -> int:
            return self.arrange(QtCore.QRect(0, 0, arg__1, 0), move=False)

        @t.override
        def setGeometry(self, arg__1: QtCore.QRect, /) -> None:
            super().setGeometry(arg__1)
            self.arrange(arg__1, move=True)

        @t.override
        def sizeHint(self, /) -> QtCore.QSize:
            return self.minimumSize()

        @t.override
        def minimumSize(self, /) -> QtCore.QSize:
            size = QtCore.QSize()
            for item in self.layoutitems:
                size = size.expandedTo(item.minimumSize())
            return size

        def arrange(self, rect: QtCore.QRect, *, move: bool) -> int:
            """Put each item in its place inside rect if move is True, and return the height that the rows take.

            A hidden widget takes no place. The items of a row share a vertical centre, thus a label stays level
            with the text of the control beside it.
            """
            gap = self.spacing()
            rows: list[list[tuple[QtWidgets.QLayoutItem, QtCore.QSize, int]]] = [[]]
            x = rect.x()
            for item in self.layoutitems:
                hint = item.sizeHint()
                if item.isEmpty() or hint.isEmpty():
                    continue
                # an item wider than the row shrinks to the row, but not below its minimum width
                hint.setWidth(max(min(hint.width(), rect.width()), item.minimumSize().width()))
                if x + hint.width() > rect.right() + 1 and rows[-1]:
                    rows.append([])
                    x = rect.x()
                rows[-1].append((item, hint, x))
                x += hint.width() + gap
            y = rect.y()
            for row in rows:
                rowheight = max((hint.height() for _, hint, _ in row), default=0)
                if move:
                    for item, hint, itemx in row:
                        item.setGeometry(QtCore.QRect(QtCore.QPoint(itemx, y + (rowheight - hint.height()) // 2), hint))
                y += rowheight + self.spacing()
            return max(y - self.spacing() - rect.y(), 0)

    return FlowLayout


def make_elided_label(text: str) -> "QtWidgets.QLabel":
    """Return a label that shows the start and the end of its text, e.g. of a long path, in the width that it gets."""
    label = get_elided_label_class()()
    label.setProperty("fulltext", text)
    label.setToolTip(text)
    return label


@cache
def get_elided_label_class() -> "type[QtWidgets.QLabel]":
    """Return the class of the labels of make_elided_label.

    A QLabel shows its whole text or cuts its end. The class shortens the middle of the text at each change of its
    width, thus Python runs only at a resize.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class ElidedLabel(QtWidgets.QLabel):
        """A label with a text that ends in the middle with "…" if the label is too narrow for it."""

        def __init__(self) -> None:
            super().__init__()
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent, /) -> None:
            super().resizeEvent(event)
            fulltext = str(self.property("fulltext") or "")
            self.setText(self.fontMetrics().elidedText(fulltext, QtCore.Qt.TextElideMode.ElideMiddle, self.width()))

    return ElidedLabel


def make_drag_header(
    on_drag: "Callable[[QtCore.QPoint], None]",
    on_drop: "Callable[[QtCore.QPoint], None]",
    on_move: "Callable[[int], None]",
) -> "QtWidgets.QWidget":
    """Return a header that the user can drag, e.g. to move a card to a new place in a list.

    During a drag, on_drag receives each position of the pointer on the screen. on_drop receives the last position.
    A child control, e.g. a button, keeps its clicks, thus a drag starts only on the background or on a label. The
    header also takes the keyboard focus, and Alt-Up (Option-Up on macOS) or Alt-Down gives -1 or 1 to on_move. A
    user of the keyboard can then move the card too. The object name "dragheader" selects the header in a style sheet.
    """
    from PySide6 import QtCore

    header = get_drag_header_class()()
    header.setObjectName("dragheader")
    header.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    # a style sheet can draw the border of the focus only on a widget with a styled background
    header.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
    header.setProperty("on_drag", on_drag)
    header.setProperty("on_drop", on_drop)
    header.setProperty("on_move", on_move)
    return header


def make_reorder_list(on_move: "Callable[[int], None]") -> "QtWidgets.QListWidget":
    """Return a list whose rows the user can move with a drag or with the keyboard, as the subplot cards do.

    Alt-Up (Option-Up on macOS) or Alt-Down gives -1 or 1 to on_move for the selected row. A drag shows the drop
    position as a line of 2 pixels in the highlight colour, which is the line between two subplot cards.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    listwidget = get_reorder_list_class()()
    # a drag of a row moves it, and a drag needs the selection of the row
    listwidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    listwidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
    listwidget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
    listwidget.setProperty("on_move", on_move)
    return listwidget


@cache
def get_reorder_list_class() -> "type[QtWidgets.QListWidget]":
    """Return the class of the lists of make_reorder_list."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class DropLineStyle(QtWidgets.QProxyStyle):
        """A style that draws the drop position of a list as the drop line of the subplot cards."""

        @t.override
        def drawPrimitive(
            self,
            element: QtWidgets.QStyle.PrimitiveElement,
            option: QtWidgets.QStyleOption,
            painter: QtGui.QPainter,
            widget: QtWidgets.QWidget | None = None,
        ) -> None:
            if element != QtWidgets.QStyle.PrimitiveElement.PE_IndicatorItemViewItemDrop:
                super().drawPrimitive(element, option, painter, widget)
                return
            # the list gives a rectangle with no height at the top or the bottom of a row. The stubs of PySide give
            # the fields of an option with no type
            rect = t.cast("QtCore.QRect", option.rect)
            palette = t.cast("QtGui.QPalette", option.palette)
            painter.fillRect(QtCore.QRect(rect.left(), rect.top() - 1, rect.width(), 2), palette.highlight())

    class ReorderList(QtWidgets.QListWidget):
        """A list that gives Alt-Up and Alt-Down to its on_move callback."""

        def __init__(self) -> None:
            super().__init__()
            # a proxy of the style of the application, with the list as its parent, thus Qt deletes both together
            style = DropLineStyle()
            style.setParent(self)
            self.setStyle(style)

        @t.override
        def keyPressEvent(self, event: QtGui.QKeyEvent, /) -> None:
            steps = {QtCore.Qt.Key.Key_Up: -1, QtCore.Qt.Key.Key_Down: 1}
            holdsalt = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)
            if holdsalt and event.key() in steps and callable(callback := self.property("on_move")):
                callback(steps[QtCore.Qt.Key(event.key())])
                event.accept()
            else:
                super().keyPressEvent(event)

    return ReorderList


@cache
def get_drag_header_class() -> "type[QtWidgets.QWidget]":
    """Return the class of the headers of make_drag_header.

    PySide keeps about 1.5 KB of memory for each class, thus the viewers make the class one time and not for each
    header.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class DragHeader(QtWidgets.QWidget):
        """A widget that sends the positions of a drag with the left mouse button to its two callbacks."""

        def __init__(self) -> None:
            super().__init__()
            self.pressposition: QtCore.QPoint | None = None
            self.dragging = False

        def send(self, name: str, position: QtCore.QPoint) -> None:
            if callable(callback := self.property(name)):
                callback(position)

        @t.override
        def mousePressEvent(self, event: QtGui.QMouseEvent, /) -> None:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.pressposition = event.globalPosition().toPoint()
                event.accept()
            else:
                super().mousePressEvent(event)

        @t.override
        def mouseMoveEvent(self, event: QtGui.QMouseEvent, /) -> None:
            if self.pressposition is None:
                super().mouseMoveEvent(event)
                return
            position = event.globalPosition().toPoint()
            distance = (position - self.pressposition).manhattanLength()
            if not self.dragging and distance >= QtWidgets.QApplication.startDragDistance():
                self.dragging = True
                self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            if self.dragging:
                self.send("on_drag", position)

        @t.override
        def keyPressEvent(self, event: QtGui.QKeyEvent, /) -> None:
            steps = {QtCore.Qt.Key.Key_Up: -1, QtCore.Qt.Key.Key_Down: 1}
            holdsalt = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)
            if holdsalt and event.key() in steps and callable(callback := self.property("on_move")):
                callback(steps[QtCore.Qt.Key(event.key())])
                event.accept()
            else:
                super().keyPressEvent(event)

        @t.override
        def mouseReleaseEvent(self, event: QtGui.QMouseEvent, /) -> None:
            wasdragging = self.dragging
            self.pressposition, self.dragging = None, False
            self.unsetCursor()
            # on_drop can make the card again, thus the header resets its state first
            if wasdragging:
                self.send("on_drop", event.globalPosition().toPoint())
            else:
                super().mouseReleaseEvent(event)

    return DragHeader


def make_slider() -> "QtWidgets.QSlider":
    """Return a horizontal slider that does not take the keyboard focus."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    # the arrow keys move the time and change the width, thus a slider must not take them
    slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
    return slider


def make_range_slider(
    steps: int,
) -> tuple[
    "QtWidgets.QWidget",
    "Callable[[int, int], None]",
    "Callable[[Callable[[int, int], None]], None]",
    "Callable[[int], None]",
]:
    """Return a slider with two handles for a range of the positions 0 to steps, and three functions of the slider.

    The first function moves the handles. The second function connects a handler, which receives the index of the
    handle that the user moved and its new position. The third function gives the slider a new number of steps.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class RangeSlider(QtWidgets.QWidget):
        """A slider with two handles, which give the minimum and the maximum of a range.

        Qt has no slider with two handles. A drag moves the handle that is nearer to the pointer, and the minimum
        stays below the maximum. The signal gives the index of the handle that moved and its new position.
        """

        limitmoved = QtCore.Signal(int, int)
        handleradius: t.Final = 8.0

        def __init__(self) -> None:
            super().__init__()
            self.steps = steps
            self.positions = [0, steps]
            self.draghandle: int | None = None
            self.setMinimumHeight(round(3 * self.handleradius))
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            # the arrow keys move the time and change the width, thus the slider must not take them
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        def set_positions(self, low: int, high: int) -> None:
            self.positions = [low, high]
            self.update()

        def set_steps(self, steps: int) -> None:
            self.steps = steps
            self.positions = [min(position, steps) for position in self.positions]
            self.update()

        def get_pixel(self, position: int) -> float:
            return self.handleradius + (self.width() - 2.0 * self.handleradius) * position / self.steps

        def get_position(self, pixel: float) -> int:
            fraction = (pixel - self.handleradius) / max(self.width() - 2.0 * self.handleradius, 1.0)
            return round(min(max(fraction, 0.0), 1.0) * self.steps)

        @t.override
        def paintEvent(self, event: QtGui.QPaintEvent) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            palette = self.palette()
            middle = self.height() / 2.0
            lowpixel, highpixel = (self.get_pixel(position) for position in self.positions)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for left, right, colourrole in (
                (self.handleradius, self.width() - self.handleradius, QtGui.QPalette.ColorRole.Mid),
                (lowpixel, highpixel, QtGui.QPalette.ColorRole.Highlight),
            ):
                painter.setBrush(palette.color(colourrole))
                painter.drawRoundedRect(QtCore.QRectF(left, middle - 2.0, right - left, 4.0), 2.0, 2.0)
            painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.ColorRole.Mid)))
            painter.setBrush(palette.color(QtGui.QPalette.ColorRole.Light))
            for pixel in (lowpixel, highpixel):
                painter.drawEllipse(QtCore.QPointF(pixel, middle), self.handleradius - 1.0, self.handleradius - 1.0)
            painter.end()

        @t.override
        def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
            pixel = event.position().x()
            midpixel = sum(self.get_pixel(position) for position in self.positions) / 2.0
            self.draghandle = 0 if pixel < midpixel else 1
            self.move_handle(pixel)

        @t.override
        def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
            if self.draghandle is not None:
                self.move_handle(event.position().x())

        @t.override
        def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
            self.draghandle = None

        def move_handle(self, pixel: float) -> None:
            if self.draghandle is None:
                return
            position = self.get_position(pixel)
            if self.draghandle == 0:
                position = min(position, self.positions[1] - 1)
            else:
                position = max(position, self.positions[0] + 1)
            if position != self.positions[self.draghandle]:
                self.positions[self.draghandle] = position
                self.update()
                self.limitmoved.emit(self.draghandle, position)

    slider = RangeSlider()

    def connect_handler(handler: "Callable[[int, int], None]") -> None:
        slider.limitmoved.connect(handler)

    return slider, slider.set_positions, connect_handler, slider.set_steps


def make_sidebar() -> "tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]":
    """Return the sidebar of the window and the layout of its panel. The sections and the command scroll together."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    sidebar = QtWidgets.QWidget()
    # the handle of the splitter sets the width, and a narrower sidebar cuts the controls
    sidebar.setMinimumWidth(360)
    sidebarlayout = QtWidgets.QVBoxLayout(sidebar)
    sidebarlayout.setContentsMargins(0, 0, 0, 0)
    panel = QtWidgets.QWidget()
    panellayout = QtWidgets.QVBoxLayout(panel)
    panellayout.setSpacing(2)
    # add_section puts each section before this stretch, thus the empty space stays at the end
    panellayout.addStretch(1)
    panelscroll = QtWidgets.QScrollArea()
    panelscroll.setWidget(panel)
    panelscroll.setWidgetResizable(True)
    panelscroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    panelscroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sidebarlayout.addWidget(panelscroll, stretch=1)
    return sidebar, panellayout


def make_plot_area(canvas: "FigureCanvasQTAgg", on_resize: "Callable[[], None]") -> "QtWidgets.QWidget":
    """Return the area of the plot, which holds the canvas at its centre and calls on_resize at each resize."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # the instance holds on_resize, and the class holds no reference to it. PySide keeps each class, thus a class that
    # captured on_resize in a closure kept the viewer, the figure, and the canvas of each closed window
    class PlotArea(QtWidgets.QWidget):
        """The area of the plot, which scales the figure to its size."""

        def __init__(self, on_resize: "Callable[[], None]") -> None:
            super().__init__()
            self.on_resize = on_resize

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            place_plot_overlays(self)
            self.on_resize()

    plotarea = PlotArea(on_resize)
    plotarea.setMinimumSize(320, 240)
    plotlayout = QtWidgets.QVBoxLayout(plotarea)
    plotlayout.setContentsMargins(0, 0, 0, 0)
    plotlayout.addWidget(canvas, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
    # the banner shows the message of a rejected plot over the plot in the red of the status bar, and the spinner
    # shows a slow plot
    banner = QtWidgets.QLabel(plotarea)
    banner.setObjectName("plotbanner")
    banner.setWordWrap(True)
    banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    banner.setStyleSheet(
        "QLabel#plotbanner { background: rgba(178, 34, 34, 225); color: white; border-radius: 8px; padding: 8px 14px; }"
    )
    banner.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    banner.hide()
    hidetimer = QtCore.QTimer(banner)
    hidetimer.setSingleShot(True)
    hidetimer.setInterval(BANNER_MILLISECONDS)
    hidetimer.timeout.connect(banner.hide)
    spinner = get_spinner_class()(plotarea)
    spinner.setObjectName("plotspinner")
    spinner.hide()
    return plotarea


# the time that the banner of a rejected plot stays over the plot. The status bar keeps the message
BANNER_MILLISECONDS: t.Final = 5000


def place_plot_overlays(plotarea: "QtWidgets.QWidget") -> None:
    """Put the banner at the centre of the plot area, a third of the way down, and the spinner at its top right corner.

    The title of the figure is at the top, thus the banner goes below it.
    """
    from PySide6 import QtWidgets

    margin = 12
    if (banner := plotarea.findChild(QtWidgets.QLabel, "plotbanner")) is not None:
        # a label that wraps its text takes a narrow width from adjustSize, thus the width comes from the text
        padding = 2 * 14 + 2
        textwidth = banner.fontMetrics().horizontalAdvance(banner.text()) + padding
        width = max(min(textwidth, plotarea.width() - 4 * margin), 100)
        banner.resize(width, banner.heightForWidth(width))
        banner.move((plotarea.width() - width) // 2, (plotarea.height() - banner.height()) // 3)
        banner.raise_()
    if (spinner := plotarea.findChild(QtWidgets.QWidget, "plotspinner")) is not None:
        spinner.move(plotarea.width() - spinner.width() - margin, margin)
        spinner.raise_()


def show_plot_banner(window: "QtCore.QObject", message: str | None) -> None:
    """Show the message of a rejected plot over the plot for a few seconds, or hide the banner if message is None."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    banner = window.findChild(QtWidgets.QLabel, "plotbanner")
    if banner is None:
        return
    if message is None:
        banner.hide()
        return
    banner.setText(message)
    banner.show()
    if (plotarea := banner.parentWidget()) is not None:
        place_plot_overlays(plotarea)
    if (hidetimer := banner.findChild(QtCore.QTimer)) is not None:
        hidetimer.start()


def set_plot_busy(window: "QtCore.QObject", *, busy: bool) -> None:
    """Show or hide the spinner over the plot."""
    from PySide6 import QtWidgets

    spinner = window.findChild(QtWidgets.QWidget, "plotspinner")
    if spinner is not None:
        spinner.setVisible(busy)
        if busy and (plotarea := spinner.parentWidget()) is not None:
            place_plot_overlays(plotarea)


@cache
def get_spinner_class() -> "type[QtWidgets.QWidget]":
    """Return the class of the spinner over the plot, which turns while it shows.

    PySide keeps about 1.5 KB of memory for each class, thus the viewers make the class one time and not for each
    window.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class Spinner(QtWidgets.QWidget):
        """A circle of 12 spokes that turns, as the progress indicator of macOS."""

        spokes: t.Final = 12

        def __init__(self, parent: QtWidgets.QWidget) -> None:
            super().__init__(parent)
            self.setFixedSize(32, 32)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.step = 0
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(80)
            self.timer.timeout.connect(self.advance)

        def advance(self) -> None:
            self.step = (self.step + 1) % self.spokes
            self.update()

        @t.override
        def showEvent(self, event: QtGui.QShowEvent, /) -> None:
            super().showEvent(event)
            self.timer.start()

        @t.override
        def hideEvent(self, event: QtGui.QHideEvent, /) -> None:
            super().hideEvent(event)
            self.timer.stop()

        @t.override
        def paintEvent(self, event: QtGui.QPaintEvent, /) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            background = self.palette().color(QtGui.QPalette.ColorRole.Window)
            background.setAlpha(210)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(self.rect(), 8, 8)
            colour = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            painter.translate(self.width() / 2, self.height() / 2)
            for spoke in range(self.spokes):
                # the newest spoke is the darkest, and the others fade behind it
                colour.setAlphaF(0.15 + 0.85 * ((spoke - self.step) % self.spokes) / (self.spokes - 1))
                pen = QtGui.QPen(colour, 2.2)
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(0, 5), QtCore.QPointF(0, 10))
                painter.rotate(360 / self.spokes)
            painter.end()

    return Spinner


def fit_canvas(canvas: "FigureCanvasQTAgg", figsize: tuple[float, float], plotarea: "QtWidgets.QWidget") -> None:
    """Scale the figure to the plot area, and keep the shape of the frames.

    An embedded canvas has no figure manager, thus the figure cannot set the size of the widget. The resolution
    of the figure changes, thus the frames keep their size in inches and the whole figure fits in the area.
    """
    from PySide6 import QtCore

    fig = canvas.figure
    figwidth, figheight = figsize
    if figwidth <= 0.0 or figheight <= 0.0:
        return
    area = plotarea.contentsRect()
    logicaldpi = max(min(area.width() / figwidth, area.height() / figheight), 20.0)
    size = QtCore.QSize(math.ceil(figwidth * logicaldpi), math.ceil(figheight * logicaldpi))
    dpi = logicaldpi * canvas.device_pixel_ratio
    # matplotlib changes the size of the figure in inches when the pixel ratio of the screen changes
    sizeinches = tuple(fig.get_size_inches())
    if canvas.size() == size and math.isclose(fig.dpi, dpi, rel_tol=1e-6) and sizeinches == figsize:
        return
    fig.set_dpi(dpi)
    fig.set_size_inches(figwidth, figheight, forward=False)
    canvas.setFixedSize(size)
    canvas.draw_idle()


def make_option_table(
    window: "QtWidgets.QWidget",
    parser: argparse.ArgumentParser,
    hiddendests: "Collection[str]",
    rows: OptionRows,
    on_rows: "Callable[[OptionRows], None]",
) -> "tuple[QtWidgets.QTableWidget, Callable[[OptionRows], None]]":
    """Return a table of the options that no other control of the window sets, and a function that shows new rows.

    The table has a list of the options that the user can search, and a control that matches the type of each
    option. on_rows receives the rows that have all their values after each change. The function that shows new
    rows keeps a row that waits for a value, unless the complete rows changed.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    actionsbyflag = get_actions_by_flag(parser)
    helptexts = get_helptexts(parser)
    tableflags = [action.option_strings[0] for action in get_table_actions(parser, hiddendests)]
    optiontable = QtWidgets.QTableWidget(0, 2)
    optiontable.setHorizontalHeaderLabels(["Option", "Value"])
    optiontable.verticalHeader().setVisible(False)
    optiontable.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    optiontable.setToolTip("Give each option of the command that no other control sets. Type part of a name to search.")
    widestflag = max(tableflags, key=len, default="")
    optiontable.setColumnWidth(0, optiontable.fontMetrics().horizontalAdvance(widestflag) + 48)
    optiontable.horizontalHeader().setStretchLastSection(True)
    # a row that holds None needs a value from the user, and the command does not give it yet
    optionrows: list[tuple[str, tuple[str, ...] | None]] = list(rows)

    def get_complete_rows() -> OptionRows:
        return tuple((flag, optionvalues) for flag, optionvalues in optionrows if optionvalues is not None)

    def set_option(row: int, flag: str) -> None:
        """Put a different option in a row.

        An empty option removes the row. An option in the empty last row adds a new row.
        """
        oldflag = optionrows[row][0] if row < len(optionrows) else ""
        if flag == oldflag or (flag and flag not in actionsbyflag):
            return
        if not flag:
            del optionrows[row]
        elif row < len(optionrows):
            optionrows[row] = (flag, get_default_tokens(actionsbyflag[flag]))
        else:
            optionrows.append((flag, get_default_tokens(actionsbyflag[flag])))
        show_option_rows()
        on_rows(get_complete_rows())

    def set_option_values(row: int, flag: str, optionvalues: tuple[str, ...] | None) -> None:
        # a field that loses the focus when the table changes can send the values of a row that is not there now
        if row < len(optionrows) and optionrows[row][0] == flag:
            optionrows[row] = (flag, optionvalues)
            on_rows(get_complete_rows())

    def make_flag_box(row: int, flag: str) -> QtWidgets.QComboBox:
        """Return a list of the options that the user can search, with the option of the row."""
        box = QtWidgets.QComboBox()
        box.setEditable(True)
        box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        flags = ["", *tableflags]
        # an option that a hidden flag gave, e.g. an old spelling, stays in its row
        if flag not in flags:
            flags.append(flag)
        box.addItems(flags)
        for index, itemflag in enumerate(flags[1:], start=1):
            helptext = helptexts.get(actionsbyflag[itemflag].dest, "")
            box.setItemData(index, helptext, QtCore.Qt.ItemDataRole.ToolTipRole)
        if (completer := box.completer()) is not None:
            set_search_completion(completer)
        box.setCurrentText(flag)
        if (lineedit := box.lineEdit()) is not None:
            lineedit.setPlaceholderText("Add an option")

        def on_flag() -> None:
            # the new rows replace this box, thus the change waits until Qt finishes with the signal
            QtCore.QTimer.singleShot(0, window, lambda: set_option(row, box.currentText()))

        box.activated.connect(on_flag)
        return box

    def make_value_editor(row: int, flag: str, optionvalues: tuple[str, ...] | None) -> QtWidgets.QWidget:
        """Return the control for the value of an option, which matches the type of the option."""
        action = actionsbyflag[flag]
        kind = get_option_kind(action)
        editor = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(editor)
        layout.setContentsMargins(2, 0, 2, 0)
        if kind == "flag":
            label = QtWidgets.QLabel("no value")
            label.setEnabled(False)
            layout.addWidget(label, 1)
        elif kind == "choice":
            choicebox = QtWidgets.QComboBox()
            choicebox.addItems([str(choice) for choice in action.choices or ()])
            choicebox.setCurrentText(optionvalues[0] if optionvalues else "")

            def on_choice(text: str) -> None:
                set_option_values(row, flag, (text,))

            choicebox.currentTextChanged.connect(on_choice)
            layout.addWidget(choicebox, 1)
        elif kind == "int":
            spinbox = QtWidgets.QSpinBox()
            spinbox.setRange(-(2**31), 2**31 - 1)
            spinbox.setValue(int(optionvalues[0]) if optionvalues else action.default)
            spinbox.setKeyboardTracking(False)

            def on_spinbox(value: int) -> None:
                set_option_values(row, flag, (str(value),))

            spinbox.valueChanged.connect(on_spinbox)
            layout.addWidget(spinbox, 1)
        else:
            islist = kind == "list"
            fieldcount = action.nargs if isinstance(action.nargs, int) else 1
            fields = [QtWidgets.QLineEdit() for _ in range(fieldcount)]
            texts = [shlex.join(optionvalues or ())] if islist else list(optionvalues or ())
            for field, text in zip(fields, texts, strict=False):
                field.setText(text)
            for field in fields:
                if action.type in {int, float} and not islist:
                    validator = QtGui.QDoubleValidator() if action.type is float else QtGui.QIntValidator()
                    validator.setLocale(QtCore.QLocale.c())
                    field.setValidator(validator)
                if islist:
                    field.setPlaceholderText("values with spaces between them")
                elif action.default is not None:
                    field.setPlaceholderText(f"default {action.default}")
                layout.addWidget(field, 1)

            def on_fields() -> None:
                texts = [field.text().strip() for field in fields]
                newvalues: tuple[str, ...] | None
                if islist:
                    try:
                        newvalues = tuple(shlex.split(texts[0]))
                    except ValueError:
                        newvalues = None
                    # nargs "+" needs a value, and nargs "*" accepts none
                    if not newvalues and action.nargs == "+":
                        newvalues = None
                elif action.nargs == "?":
                    newvalues = (texts[0],) if texts[0] else ()
                else:
                    newvalues = tuple(texts) if all(texts) else None
                set_option_values(row, flag, newvalues)

            for field in fields:
                field.editingFinished.connect(on_fields)

        removebutton = make_glyph_button("✕", f"Remove {flag} from the command", f"Remove {flag}")
        removebutton.clicked.connect(lambda: QtCore.QTimer.singleShot(0, window, lambda: set_option(row, "")))
        layout.addWidget(removebutton)
        editor.setToolTip(helptexts.get(action.dest, ""))
        return editor

    def show_option_rows() -> None:
        """Make a row of the table for each option, and an empty row at the end that adds an option."""
        optiontable.setRowCount(len(optionrows) + 1)
        for row, (flag, optionvalues) in enumerate([*optionrows, ("", None)]):
            optiontable.setCellWidget(row, 0, make_flag_box(row, flag))
            if flag:
                optiontable.setCellWidget(row, 1, make_value_editor(row, flag, optionvalues))
            else:
                optiontable.removeCellWidget(row, 1)
        optiontable.resizeRowsToContents()
        # the sidebar scrolls, thus the table shows each row and does not scroll itself
        rowsheight = sum(optiontable.rowHeight(row) for row in range(optiontable.rowCount()))
        headerheight = optiontable.horizontalHeader().sizeHint().height()
        optiontable.setFixedHeight(rowsheight + headerheight + 2 * optiontable.frameWidth())

    def set_rows(newrows: OptionRows) -> None:
        # after the command rejects a change, the table shows the old options again without the rows with no value
        if get_complete_rows() != newrows:
            optionrows[:] = list(newrows)
            show_option_rows()

    show_option_rows()
    return optiontable, set_rows


def set_search_completion(completer: "QtWidgets.QCompleter") -> None:
    """Let a completer show each name that holds the typed text, e.g. "ion" shows averageionisation."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
    completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
    completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
    completer.setMaxVisibleItems(15)


def make_completer(names: "Sequence[str]", parent: "QtWidgets.QWidget") -> "QtWidgets.QCompleter":
    """Return a completer that finds each name that holds the typed text, e.g. "ion" finds averageionisation."""
    from PySide6 import QtWidgets

    completer = QtWidgets.QCompleter(list(names), parent)
    set_search_completion(completer)
    return completer


def add_command_section(
    panellayout: "QtWidgets.QVBoxLayout",
) -> "tuple[QtWidgets.QPlainTextEdit, QtWidgets.QPushButton]":
    """Add the command after the sections of the panel, and return its text box and its Copy button."""
    return add_copy_box(
        panellayout, "Command", f"Copy the command to the clipboard ({get_menu_shortcut_texts()['Copy Command']})"
    )


def add_copy_box(
    panellayout: "QtWidgets.QVBoxLayout", title: str, copytooltip: str, *, maxlines: int = 8, wraplines: bool = True
) -> "tuple[QtWidgets.QPlainTextEdit, QtWidgets.QPushButton]":
    """Add a section with a read-only box of text and a Copy button, and return the box and the button.

    The box shows up to maxlines lines, and a longer text scrolls. Code keeps its lines without a wrap.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    _, grid = add_section(panellayout, title)
    # the text takes the width, and the Copy button keeps its size at the right
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 0)

    # the instance of the box holds no reference to the window. PySide keeps each class, thus the class must hold none
    class CommandBox(QtWidgets.QPlainTextEdit):
        """The box of the text, which fits its height to the wrapped lines when its width changes."""

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            if event.size().width() != event.oldSize().width():
                fit_command_box(self)

    textbox = CommandBox()
    textbox.setProperty("maxlines", maxlines)
    if not wraplines:
        textbox.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
    textbox.setReadOnly(True)
    textbox.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    fit_command_box(textbox)
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip(copytooltip)
    grid.addWidget(textbox, 0, 0)
    grid.addWidget(copybutton, 0, 1, QtCore.Qt.AlignmentFlag.AlignTop)
    return textbox, copybutton


def parse_command_tokens(parser: argparse.ArgumentParser, tokens: "Sequence[str]") -> argparse.Namespace | None:
    """Return the arguments of the tokens of a command, or None if the parser rejects them.

    The plot of the same tokens fails too, and the status line then gives the reason.
    """
    try:
        return parser.parse_args(separate_trailing_folders(tokens))
    except SystemExit:
        return None


def get_changed_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace, skip: "Collection[str]" = ()
) -> dict[str, t.Any]:
    """Return each argument that differs from its default, in the order of the parser."""
    return {dest: value for dest, value in vars(args).items() if dest not in skip and value != parser.get_default(dest)}


# a list that is longer than this on one line gives one item on each line
PYTHON_LINE_LENGTH: t.Final = 100


def format_python_value(value: t.Any, indent: int) -> str:
    """Return the Python text of a value of an argument. A list that is too long for one line gives one item a line."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        # the JSON text of a string is also a Python string, and it has the double quotes of the usual Python style
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        opening, closing = ("(", ")") if isinstance(value, tuple) else ("[", "]")
        items = [format_python_value(item, indent + 4) for item in value]
        onetuple = "," if isinstance(value, tuple) and len(items) == 1 else ""
        oneline = f"{opening}{', '.join(items)}{onetuple}{closing}"
        if indent + len(oneline) <= PYTHON_LINE_LENGTH:
            return oneline
        itemlines = "".join(f"{' ' * (indent + 4)}{item},\n" for item in items)
        return f"{opening}\n{itemlines}{' ' * indent}{closing}"
    return repr(value)


def get_python_call(functionname: str, kwargs: "Mapping[str, t.Any]") -> str:
    """Return the Python code that calls the main function of a command with these keyword arguments."""
    arguments = "".join(f"    {name}={format_python_value(value, 4)},\n" for name, value in kwargs.items())
    call = f"{functionname}(\n{arguments})" if arguments else f"{functionname}()"
    return f"import artistools as at\n\n{call}"


# a box of text shows at least this number of lines
MIN_COMMAND_LINES: t.Final = 3


def set_command_text(commandtext: "QtWidgets.QPlainTextEdit", command: str) -> None:
    """Show the command in its box, and give the box the height of the lines of the command."""
    if commandtext.toPlainText() != command:
        commandtext.setPlainText(command)
        fit_command_box(commandtext)


def fit_command_box(commandtext: "QtWidgets.QPlainTextEdit") -> None:
    """Give the command box the height of the wrapped lines of its text, and keep that height inside the limits.

    The layout of the document gives the wrapped lines at the width of the box. The rectangle of each block is a few
    pixels taller than its lines, and the box needs those pixels, else it scrolls.
    """
    from PySide6 import QtWidgets

    document = commandtext.document()
    layout = document.documentLayout()
    textlines = math.ceil(layout.documentSize().height())
    textheight = sum(
        layout.blockBoundingRect(document.findBlockByNumber(i)).height() for i in range(document.blockCount())
    )
    shownlines = min(max(textlines, MIN_COMMAND_LINES), int(commandtext.property("maxlines")))
    boxheight = textheight + (shownlines - textlines) * commandtext.fontMetrics().lineSpacing()
    # a box with no wrap shows a horizontal scroll bar for a long line, and the bar must not cover the last line
    if commandtext.lineWrapMode() == QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap:
        boxheight += commandtext.horizontalScrollBar().sizeHint().height()
    commandtext.setFixedHeight(math.ceil(boxheight + 2 * document.documentMargin() + 2 * commandtext.frameWidth()))


def start_play_timer(playtimer: "QtCore.QTimer", plotseconds: float, fps: float) -> None:
    """Start the pause before the next step of Play, thus a step takes 1/fps seconds.

    The old plot stays in view while the worker draws the next one, thus the pause is the rest of the time of the step.
    A plot that takes longer than 1/fps seconds gives a lower rate.
    """
    playtimer.start(max(0, round(1000.0 / fps - plotseconds * 1000.0)))


def get_icon(symbol: str, themeicon: "QtGui.QIcon.ThemeIcon") -> "QtGui.QIcon":
    """Return the SF Symbol of the name on macOS, else the icon of the theme.

    Qt gives an SF Symbol for its name on macOS. The result can be an empty icon, and a tool button then shows its
    text.
    """
    from PySide6 import QtGui

    icon = QtGui.QIcon.fromTheme(symbol) if sys.platform == "darwin" else QtGui.QIcon()
    return QtGui.QIcon.fromTheme(themeicon) if icon.isNull() else icon


def make_segmented_control(labels: "Sequence[str]", tooltips: "Sequence[str]") -> "QtWidgets.QTabBar":
    """Return a control of side-by-side segments, in which the user selects one segment, e.g. one mode of the time.

    The macOS style of Qt draws a tab bar as a segmented control, which the apps of macOS use for a choice of modes.
    """
    from PySide6 import QtWidgets

    segments = QtWidgets.QTabBar()
    segments.setDrawBase(False)
    segments.setExpanding(False)
    for index, (label, tooltip) in enumerate(zip(labels, tooltips, strict=True)):
        segments.addTab(label)
        segments.setTabToolTip(index, tooltip)
    return segments


def make_step_button(*, forward: bool) -> "QtWidgets.QToolButton":
    """Return a button that moves the time to the next or the previous timestep, with the icon of the platform."""
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    button = QtWidgets.QToolButton()
    if forward:
        icon = get_icon("forward.end.fill", QtGui.QIcon.ThemeIcon.MediaSkipForward)
        button.setToolTip("Move the time to the next timestep (Right key)")
        button.setAccessibleName("Next Timestep")
    else:
        icon = get_icon("backward.end.fill", QtGui.QIcon.ThemeIcon.MediaSkipBackward)
        button.setToolTip("Move the time to the previous timestep (Left key)")
        button.setAccessibleName("Previous Timestep")
    button.setIcon(icon)
    # a platform with no icon shows the arrow as text
    if icon.isNull():
        button.setText("▶" if forward else "◀")
    return button


def make_play_button(tooltip: str) -> "QtWidgets.QToolButton":
    """Return the checkable Play button, which shows Pause and its icon while Play runs."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    playicon = get_icon("play.fill", QtGui.QIcon.ThemeIcon.MediaPlaybackStart)
    pauseicon = get_icon("pause.fill", QtGui.QIcon.ThemeIcon.MediaPlaybackPause)
    play = QtWidgets.QToolButton()
    play.setCheckable(True)
    play.setText("Play")
    play.setIcon(playicon)
    play.setToolTip(tooltip)
    play.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    def show_play_state(checked: bool) -> None:
        play.setIcon(pauseicon if checked else playicon)
        play.setText("Pause" if checked else "Play")

    play.toggled.connect(show_play_state)
    return play


def make_play_row(
    stepbuttons: "Sequence[QtWidgets.QWidget]",
    label: "QtWidgets.QLabel",
    fpsbox: "QtWidgets.QDoubleSpinBox",
    playbutton: "QtWidgets.QWidget",
) -> "QtWidgets.QHBoxLayout":
    """Return one row of the Time section: the step buttons, the label of the timesteps, the frame rate, and Play.

    One row for these items keeps more of the other controls in view. The step buttons touch, as a pair of arrows.
    """
    from PySide6 import QtWidgets

    steps = QtWidgets.QHBoxLayout()
    steps.setSpacing(0)
    for button in stepbuttons:
        steps.addWidget(button)
    row = QtWidgets.QHBoxLayout()
    row.addLayout(steps)
    # a narrow sidebar puts the text of the label on two lines
    label.setWordWrap(True)
    row.addWidget(label, 1)
    row.addWidget(QtWidgets.QLabel("FPS:"))
    row.addWidget(fpsbox)
    row.addWidget(playbutton)
    return row


def make_fps_box() -> "QtWidgets.QDoubleSpinBox":
    """Return the box of the frame rate of Play, with up and down buttons."""
    from PySide6 import QtWidgets

    fpsbox = QtWidgets.QDoubleSpinBox()
    # the minimum is one step, thus the up and down buttons keep each value on the steps of 0.5
    fpsbox.setRange(0.5, 60.0)
    fpsbox.setDecimals(1)
    fpsbox.setSingleStep(0.5)
    fpsbox.setValue(get_float_setting("playfps", DEFAULT_PLAY_FPS))
    fpsbox.setToolTip(
        "The frames per second of Play. A plot that takes longer than one frame gives a lower rate. After the last"
        " step, Play starts again at the first step."
    )
    return fpsbox


class StatusBar(t.NamedTuple):
    """The labels of the status bar of a window, and its help button."""

    message: "QtWidgets.QLabel"
    readout: "QtWidgets.QLabel"
    drawtime: "QtWidgets.QLabel"
    helpbutton: "QtWidgets.QToolButton"


def show_status_message(statusbar: StatusBar, message: str | None, warning: str) -> None:
    """Show the message of a rejected plot in red, or else the last warning of the plot in amber."""
    if message is not None:
        statusbar.message.setStyleSheet("color: firebrick")
        statusbar.message.setText(message)
    else:
        statusbar.message.setStyleSheet("color: darkorange")
        statusbar.message.setText(warning)


def show_status_note(statusbar: StatusBar, note: str) -> None:
    """Show a note about an action that succeeded, e.g. a saved file, in the colour of normal text."""
    statusbar.message.setStyleSheet("")
    statusbar.message.setText(note)


def make_status_bar(window: "QtWidgets.QMainWindow") -> StatusBar:
    """Return the labels of the status bar and its help button.

    The status bar gives the messages at the left, and the readout, the time of the plot, and the help at the right.
    """
    from PySide6 import QtWidgets

    statusbar = window.statusBar()
    messagelabel = QtWidgets.QLabel()
    messagelabel.setStyleSheet("color: firebrick")
    readoutlabel = QtWidgets.QLabel()
    drawtimelabel = QtWidgets.QLabel()
    helpbutton = QtWidgets.QToolButton()
    helpbutton.setText("?")
    helpbutton.setToolTip("Show the keys and the mouse actions of the window (?)")
    helpbutton.setAccessibleName("Keys and Mouse Actions")
    statusbar.addWidget(messagelabel, stretch=1)
    for widget in (readoutlabel, drawtimelabel, helpbutton):
        statusbar.addPermanentWidget(widget)
    return StatusBar(message=messagelabel, readout=readoutlabel, drawtime=drawtimelabel, helpbutton=helpbutton)


def get_menu_items() -> "list[tuple[str, str, QtGui.QKeySequence]]":
    """Return the menu, the text, and the shortcut of each menu item of a viewer.

    The texts use the capitals of a title, and an item that opens a dialog ends with an ellipsis, as in the apps of
    macOS. On macOS, Ctrl is the Command key and Meta is the Control key.
    """
    from PySide6 import QtGui

    standardkey = QtGui.QKeySequence.StandardKey
    return [
        ("File", "Open Model…", QtGui.QKeySequence(standardkey.Open)),
        ("File", "Reload Data", QtGui.QKeySequence(standardkey.Refresh)),
        ("File", "Save Figure…", QtGui.QKeySequence(standardkey.Save)),
        ("File", "Export Animation…", QtGui.QKeySequence("Ctrl+Shift+E")),
        ("File", "Close Window", QtGui.QKeySequence(standardkey.Close)),
        ("Edit", "Undo", QtGui.QKeySequence(standardkey.Undo)),
        ("Edit", "Redo", QtGui.QKeySequence(standardkey.Redo)),
        ("Edit", "Copy Figure", QtGui.QKeySequence(standardkey.Copy)),
        ("Edit", "Copy Command", QtGui.QKeySequence("Ctrl+Shift+C")),
        ("Edit", "Copy Python", QtGui.QKeySequence("Ctrl+Alt+C")),
        # macOS moves this item to the menu of the application
        ("Edit", "Settings…", QtGui.QKeySequence("Ctrl+,")),
        ("View", "Play", QtGui.QKeySequence("Space")),
        ("View", "Cancel Plot", QtGui.QKeySequence("Ctrl+.")),
        ("View", "Hide Sidebar", QtGui.QKeySequence("Ctrl+Meta+S")),
        ("View", "Enter Full Screen", QtGui.QKeySequence(standardkey.FullScreen)),
        ("Window", "Minimize", QtGui.QKeySequence("Ctrl+M")),
        ("Window", "Zoom", QtGui.QKeySequence()),
        ("Help", "Keys and Mouse Actions", QtGui.QKeySequence("?")),
        ("Help", "artistools Help", QtGui.QKeySequence()),
        # macOS moves this item to the menu of the application, with the name of the application
        ("Help", "About", QtGui.QKeySequence()),
    ]


# the help text of each menu item in the table of the keys
MENU_HELPTEXTS: t.Final = MappingProxyType({
    "Open Model…": "Open a model in a new window",
    "Reload Data": "Read the run again, e.g. while ARTIS writes more timesteps",
    "Save Figure…": "Save the figure in the format of the Figure section",
    "Export Animation…": "Save a GIF file of the steps of Play",
    "Undo": "Undo the last change",
    "Redo": "Redo the change that Undo removed",
    "Copy Figure": "Copy the figure in the format of the Figure section",
    "Copy Command": "Copy the command",
    "Copy Python": "Copy the Python code of the plot",
    "Play": "Play or pause",
    "Cancel Plot": "Stop the wait for a slow plot, and keep the plot on the screen",
    "Hide Sidebar": "Hide or show the sidebar",
})


def get_keyboard_help(keyrows: "Sequence[tuple[str, str]]", menuitems: "Collection[str]") -> str:
    """Return the table of the keys and the mouse actions of a viewer, with the shortcuts of the platform.

    keyrows gives the keys or the mouse action, and its help text, of each row of the viewer as HTML. menuitems gives
    the texts of the menu items of the viewer, as add_menus receives them, and the table gives the shortcut of each.
    """
    shortcuts = get_menu_shortcut_texts()
    # a viewer row replaces the menu row with the same keys, because it has more text, e.g. Space for Play
    viewerkeys = {keys for keys, _ in keyrows}
    rows = [
        *keyrows,
        # every window has the sidebar, and add_menus gives its item
        *(
            (f"<b>{shortcuts[text]}</b>", helptext)
            for text, helptext in MENU_HELPTEXTS.items()
            if (text in menuitems or text == "Hide Sidebar") and f"<b>{shortcuts[text]}</b>" not in viewerkeys
        ),
        (f"<b>{shortcuts['Keys and Mouse Actions']}</b>", "Show this list"),
    ]
    return (
        "<table>\n" + "".join(f"<tr><td>{keys}</td><td>{helptext}</td></tr>\n" for keys, helptext in rows) + "</table>"
    )


def get_short_number(value: float) -> str:
    """Return a number with 3 significant digits for a short command, e.g. 12300 or 1.23e-05."""
    return format(float(f"{value:.3g}"), ".10g")


def get_menu_shortcut_texts() -> dict[str, str]:
    """Return the shortcut of each menu item by its text, in the form of the platform, e.g. ⌘S or Ctrl+S."""
    from PySide6 import QtGui

    return {text: keys.toString(QtGui.QKeySequence.SequenceFormat.NativeText) for _menu, text, keys in get_menu_items()}


def add_menus(
    window: "QtWidgets.QMainWindow",
    callbacks: "Mapping[str, Callable[[], object]]",
    queue: "DrawQueue[t.Any]",
    playbutton: "QtWidgets.QAbstractButton | None",
    open_folder: "Callable[[str], object]",
) -> list[str]:
    """Add the menus File, Edit, View, Window, and Help, and return the text of each item.

    callbacks gives the function of each item by the text of get_menu_items. A viewer omits an item that it does not
    support, e.g. Reload Data. This function gives the items of the sidebar, the full screen, and the window. The
    queue gives Undo, Redo, and Cancel Plot, unless callbacks gives them, and playbutton gives Play. A viewer with no
    playbutton gets no Play item. File > Open Recent gives a recent model to open_folder.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    def cancel_plot() -> None:
        if playbutton is not None:
            playbutton.setChecked(False)
        queue.cancel()

    enabled: dict[str, Callable[[], bool]] = {
        "Undo": queue.can_undo,
        "Redo": queue.can_redo,
        "Cancel Plot": queue.is_busy,
    }
    titles: dict[str, Callable[[], str]] = {"Play": playbutton.text} if playbutton is not None else {}
    plotcallbacks: dict[str, Callable[[], object]] = {
        "Undo": queue.undo,
        "Redo": queue.redo,
        **({"Play": playbutton.toggle} if playbutton is not None else {}),
        "Cancel Plot": cancel_plot,
    }

    windowcallbacks: dict[str, Callable[[], object]] = {
        "Hide Sidebar": lambda: toggle_sidebar(window),
        "Enter Full Screen": lambda: window.showNormal() if window.isFullScreen() else window.showFullScreen(),
        "Minimize": window.showMinimized,
        "Zoom": lambda: window.showNormal() if window.isMaximized() else window.showMaximized(),
        "Settings…": show_settings_window,
        "artistools Help": lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(HELP_URL)),
        "About": lambda: show_about(window),
    }
    windowtitles: dict[str, Callable[[], str]] = {
        "Hide Sidebar": lambda: "Show Sidebar" if is_sidebar_hidden(window) else "Hide Sidebar",
        "Enter Full Screen": lambda: "Exit Full Screen" if window.isFullScreen() else "Enter Full Screen",
    }
    allcallbacks = {**windowcallbacks, **plotcallbacks, **callbacks}
    alltitles = {**windowtitles, **titles}
    menubar = window.menuBar()
    menus = {name: menubar.addMenu(name) for name in ("File", "Edit", "View", "Window", "Help")}
    actions: dict[str, QtGui.QAction] = {}
    for menuname, text, keys in get_menu_items():
        # macOS adds its own Enter Full Screen item to a menu with the title View
        if text not in allcallbacks or (text == "Enter Full Screen" and sys.platform == "darwin"):
            continue
        action = menus[menuname].addAction(text)
        action.setShortcut(keys)
        action.triggered.connect(allcallbacks[text])
        # the role of an application item puts Settings… in the menu of the application on macOS with this text.
        # The role of Preferences gave the old text "Preferences..."
        if text == "Settings…":
            action.setMenuRole(QtGui.QAction.MenuRole.ApplicationSpecificRole)
        elif text == "About":
            action.setMenuRole(QtGui.QAction.MenuRole.AboutRole)
        actions[text] = action
        if text == "Open Model…":
            add_recent_menu(menus["File"], open_folder)

    def update_items() -> None:
        for text, action in actions.items():
            if text in alltitles:
                action.setText(alltitles[text]())
            action.setEnabled(enabled[text]() if text in enabled else True)

    def enable_all() -> None:
        # a disabled item takes no shortcut, and the state of an item can change while the menu is closed
        for action in actions.values():
            action.setEnabled(True)

    windowmenu = menus["Window"]
    windowmenu.addSeparator()
    windowlistactions: list[QtGui.QAction] = []

    def update_window_list() -> None:
        """Show each window of the viewers at the end of the Window menu, with a mark at this window."""
        for action in windowlistactions:
            windowmenu.removeAction(action)
            action.deleteLater()
        windowlistactions.clear()
        for other in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(other, QtWidgets.QMainWindow) and other.isVisible():
                action = windowmenu.addAction(other.windowTitle())
                action.setCheckable(True)
                action.setChecked(other is window)
                action.triggered.connect(partial(activate_window, other))
                windowlistactions.append(action)

    for menu in menus.values():
        menu.aboutToShow.connect(update_items)
        menu.aboutToHide.connect(enable_all)
    windowmenu.aboutToShow.connect(update_window_list)
    return list(actions)


def add_recent_menu(filemenu: "QtWidgets.QMenu", open_folder: "Callable[[str], object]") -> None:
    """Add the submenu Open Recent, which shows the recent models when it opens, and Clear Menu at its end."""
    recentmenu = filemenu.addMenu("Open Recent")

    def show_recent_models() -> None:
        recentmenu.clear()
        for folder in get_recent_models():
            action = recentmenu.addAction(Path(folder).name)
            action.setToolTip(folder)
            # a test of a remote folder starts ssh, thus the menu enables a remote model with no test
            action.setEnabled(is_remote_path(folder) or Path(folder).is_dir())
            action.triggered.connect(partial(open_folder, folder))
        recentmenu.addSeparator()
        clearaction = recentmenu.addAction("Clear Menu")
        clearaction.setEnabled(bool(get_recent_models()))
        clearaction.triggered.connect(lambda: get_settings().remove(get_recent_setting_key()))

    recentmenu.aboutToShow.connect(show_recent_models)


# the documentation of artistools, which Help > artistools Help opens
HELP_URL: t.Final = "https://github.com/artis-mcrt/artistools#readme"

# the choices of the appearance in the Settings window, as the Settings of macOS and Xcode name them
APPEARANCES: t.Final = ("System", "Light", "Dark")


def apply_appearance() -> None:
    """Give the application the appearance of the Settings window: the appearance of the system, light, or dark.

    A change gives the signal colorSchemeChanged, and each window then draws its plot again.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    schemes = {"Light": QtCore.Qt.ColorScheme.Light, "Dark": QtCore.Qt.ColorScheme.Dark}
    appearance = str(get_settings().value("appearance", "System"))
    QtGui.QGuiApplication.styleHints().setColorScheme(schemes.get(appearance, QtCore.Qt.ColorScheme.Unknown))


def show_about(parent: "QtWidgets.QWidget") -> None:
    """Show the About panel of the viewer: the name, the version of artistools, and the link to the source."""
    import importlib.metadata

    from PySide6 import QtCore
    from PySide6 import QtWidgets

    name = QtWidgets.QApplication.applicationDisplayName()
    version = importlib.metadata.version("artistools")
    about = QtWidgets.QMessageBox(parent)
    about.setWindowTitle(f"About {name}")
    about.setIconPixmap(QtWidgets.QApplication.windowIcon().pixmap(64, 64))
    # macOS shows the text in bold and the informative text in a small regular font, as an About panel of a Mac app
    about.setText(name)
    about.setInformativeText(
        f"artistools {version}<br>Qt {QtCore.qVersion()}, Python {sys.version.split()[0]}<br><br>"
        '<a href="https://github.com/artis-mcrt/artistools">github.com/artis-mcrt/artistools</a>'
    )
    about.exec()


def set_window_document(window: "QtWidgets.QMainWindow", folder: Path, title: str) -> None:
    """Give the window the title of its model and the folder of the model as its file.

    macOS then shows the icon of the folder in the title bar. A Command-click on the title shows the path, and a drag
    of the icon gives the folder to a different app.
    """
    window.setWindowTitle(title)
    # a model on a different host has no local folder, thus its window shows no folder icon
    window.setWindowFilePath("" if is_remote_path(folder) else str(folder.absolute()))


def show_settings_window() -> None:
    """Show the Settings window of the viewers, or bring it to the front if it is open.

    A change applies at once and the settings keep it, as in the Settings windows of macOS. Each viewer window
    receives a change of the colours of the plot through its property "settingshandler". The Settings window belongs
    to all the viewer windows, thus it has no parent, and it stays open when a viewer window closes.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    for widget in QtWidgets.QApplication.topLevelWidgets():
        if widget.objectName() == "settings" and widget.isVisible():
            activate_window(widget)
            return
    settings = get_settings()
    dialog = QtWidgets.QDialog()
    # Python deletes a window with no parent and no reference, thus the application holds one until the window closes
    if (app := QtWidgets.QApplication.instance()) is not None:
        app.setProperty("settingswindow", lambda: dialog)
        dialog.destroyed.connect(lambda: app.setProperty("settingswindow", None))
    dialog.setObjectName("settings")
    dialog.setWindowTitle("Settings")
    dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    form = QtWidgets.QFormLayout(dialog)
    # a Settings window of macOS has the size of its content
    form.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

    appearancesegments = make_segmented_control(
        list(APPEARANCES),
        ["The appearance of macOS, light or dark", "A light window and plot", "A dark window and plot"],
    )
    appearance = str(settings.value("appearance", "System"))
    appearancesegments.setCurrentIndex(APPEARANCES.index(appearance) if appearance in APPEARANCES else 0)

    def on_appearance(index: int) -> None:
        settings.setValue("appearance", APPEARANCES[index])
        apply_appearance()

    appearancesegments.currentChanged.connect(on_appearance)
    form.addRow("Appearance:", appearancesegments)

    fpsbox = make_fps_box()
    fpsbox.setToolTip("The frames per second of Play in a new window")
    fpsbox.valueChanged.connect(partial(settings.setValue, "playfps"))
    form.addRow("Play in a new window [FPS]:", fpsbox)

    darkcheck = QtWidgets.QCheckBox("Show the plot in the colours of a dark appearance")
    darkcheck.setToolTip("A saved figure keeps its usual colours. Without this choice, the plot stays light")
    darkcheck.setChecked(get_bool_setting("darkplot", default=True))

    def on_dark(checked: bool) -> None:
        settings.setValue("darkplot", checked)
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if callable(handler := widget.property("settingshandler")):
                handler()

    darkcheck.toggled.connect(on_dark)
    form.addRow(darkcheck)

    reopencheck = QtWidgets.QCheckBox("Reopen the windows of the last session at the start")
    reopencheck.setToolTip("After Quit, the next start opens each window that was open, beside the new window")
    reopencheck.setChecked(get_bool_setting("reopenwindows", default=True))
    reopencheck.toggled.connect(partial(settings.setValue, "reopenwindows"))
    form.addRow(reopencheck)

    for flag, maximum, step in (("-figscale", 10.0, 0.1), ("-labelfontsize", 40.0, 1.0)):
        box = QtWidgets.QDoubleSpinBox()
        # the box shows the text "Default", which is wider than a number
        box.setMinimumWidth(100)
        box.setRange(0.0, maximum)
        box.setSingleStep(step)
        # the minimum of the box shows "Default", which gives no option
        box.setSpecialValueText("Default")
        box.setValue(get_float_setting(f"default{flag}", 0.0))
        box.setToolTip(f"A new window adds {flag} with this value if its command does not give {flag}")
        box.valueChanged.connect(partial(settings.setValue, f"default{flag}"))
        form.addRow(f"{flag} of a new window:", box)
    dialog.show()


def add_default_options(parser: "SuggestingArgumentParser", tokens: "Sequence[str]") -> list[str]:
    """Return the tokens with the options of the Settings window that the tokens do not give, e.g. -figscale.

    The options go before the other tokens, and an option that the parser of the command does not have gives nothing.
    """
    actions = get_actions_by_flag(parser)
    givendests = {action.dest for token in tokens if (action := find_option_action(parser, token)[0]) is not None}
    added: list[str] = []
    for flag in ("-figscale", "-labelfontsize"):
        value = get_float_setting(f"default{flag}", 0.0)
        if value > 0.0 and flag in actions and actions[flag].dest not in givendests:
            added += [flag, format(value, "g")]
    return [*added, *tokens]


def activate_window(window: "QtWidgets.QWidget") -> None:
    """Show the window in front of the other windows, and give it the keyboard."""
    if window.isMinimized():
        window.showNormal()
    window.raise_()
    window.activateWindow()


def get_sidebar(window: "QtWidgets.QMainWindow") -> "QtWidgets.QWidget | None":
    """Return the sidebar of the window, which make_central_splitter puts at the right of the plot."""
    from PySide6 import QtWidgets

    splitter = window.centralWidget()
    return splitter.widget(1) if isinstance(splitter, QtWidgets.QSplitter) else None


def is_sidebar_hidden(window: "QtWidgets.QMainWindow") -> bool:
    """Return whether the user hid the sidebar, with the menu or with a drag of the handle to the edge."""
    sidebar = get_sidebar(window)
    return sidebar is None or sidebar.isHidden() or sidebar.width() == 0


def toggle_sidebar(window: "QtWidgets.QMainWindow") -> None:
    """Hide the sidebar, or show it with its first width."""
    from PySide6 import QtWidgets

    sidebar = get_sidebar(window)
    splitter = window.centralWidget()
    if sidebar is None or not isinstance(splitter, QtWidgets.QSplitter):
        return
    if not is_sidebar_hidden(window):
        sidebar.hide()
        return
    sidebar.show()
    # a drag to the edge gives the sidebar a width of zero, thus the sidebar takes its first width again
    if sidebar.width() == 0:
        total = sum(splitter.sizes())
        splitter.setSizes([max(total - SIDEBAR_WIDTH, 0), SIDEBAR_WIDTH])


# the background and the text colour of a dark plot while the palette of the window is still light
DARK_PLOT_COLOURS: t.Final = ("#323232", "#dfdfdf")


def get_dark_plot_colours() -> tuple[str, str] | None:
    """Return the background colour and the text colour of the window in Dark Mode, or None for a light plot.

    The window thread calls this function, and a render in the worker thread receives the result. The setting
    "darkplot" of the Settings window can keep the plot light.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    if not get_bool_setting("darkplot", default=True):
        return None
    if QtGui.QGuiApplication.styleHints().colorScheme() != QtCore.Qt.ColorScheme.Dark:
        return None
    palette = QtGui.QGuiApplication.palette()
    background = palette.color(QtGui.QPalette.ColorRole.Window)
    # Qt gives the signal of a new colour scheme before it gives the new palette
    if background.lightness() > 128:
        return DARK_PLOT_COLOURS
    return background.name(), palette.color(QtGui.QPalette.ColorRole.WindowText).name()


def apply_dark_colours(fig: "mplfig.Figure", background: str, foreground: str) -> None:
    """Give a figure of the window the colours of Dark Mode.

    The frames, the ticks, and the text take the colours of the window. A black or dark grey line or text takes the
    colour of the text, because a dark line or text does not show on the dark background. The other colours stay,
    e.g. the colours of the series and of an image. A dark colour of a series, e.g. the dark red of sulphur, is not
    grey, thus it stays. The command saves a figure with its usual colours, because only the window calls this.
    """
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    def is_dark_grey(colour: "mplt.ColorType") -> bool:
        red, green, blue, alpha = mcolors.to_rgba(colour)
        isgrey = max(red, green, blue) - min(red, green, blue) < 0.1
        return alpha > 0.0 and isgrey and 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.25

    fig.patch.set_facecolor(background)
    for axis in fig.axes:
        axis.set_facecolor(background)
        for spine in axis.spines.values():
            spine.set_edgecolor(foreground)
        axis.tick_params(which="both", colors=foreground)
        if (legend := axis.get_legend()) is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(foreground)
    for text in fig.findobj(Text):
        if isinstance(text, Text) and is_dark_grey(text.get_color()):
            text.set_color(foreground)
    for line in fig.findobj(Line2D):
        if isinstance(line, Line2D) and is_dark_grey(line.get_color()):
            line.set_color(foreground)


def copy_figure_of_command(
    queue: "DrawQueue[t.Any]",
    statusbar: StatusBar,
    commandmain: "Callable[..., None]",
    parser: "SuggestingArgumentParser",
    plottokens: "Sequence[str]",
    choice: tuple[str, int],
) -> None:
    """Put the figure of the command on the clipboard, in the format and the resolution of choice.

    The command draws the figure, as for a saved file. Thus the image has the usual colours and no empty margin, also
    when the window shows the plot in Dark Mode. The worker thread runs the command, thus the window accepts input.
    """
    suffix, dpi = choice
    tokens = [*remove_options(parser, plottokens, {"dpi"}), "-dpi", str(dpi)]
    files: list[bytes] = []

    def draw_file() -> str | None:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            filepath = Path(folder) / f"figure.{suffix}"
            commandmain(argsraw=[*tokens, "-o", str(filepath)])
            if not filepath.is_file():
                return f"The command wrote no {suffix.upper()} file"
            files.append(filepath.read_bytes())
        return None

    def show_result(message: str | None) -> None:
        if message is None and files:
            message = put_file_on_clipboard(files[0], suffix)
        if message is not None:
            show_status_message(statusbar, f"The viewer did not copy the figure: {message}", "")
        else:
            pixelsize = get_pixel_size_text(files[0], suffix)
            resolution = f", with {dpi} dpi{pixelsize}" if suffix == "png" else ""
            show_status_note(statusbar, f"Copied the figure as {suffix.upper()}{resolution}")

    if not queue.run_task(lambda: run_command_step(draw_file), "Copy of the figure in progress...", show_result):
        show_status_message(statusbar, "A different task is in progress. Copy the figure after it", "")


def get_pixel_size_text(data: bytes, suffix: str) -> str:
    """Return the size of a PNG file in pixels for the status bar, or "" for a different format.

    The size of the plot in the window and the resolution give the size, and the command crops the empty part of a
    margin.
    """
    from PySide6 import QtGui

    image = QtGui.QImage.fromData(data) if suffix == "png" else QtGui.QImage()
    return "" if image.isNull() else f" ({image.width()} \N{MULTIPLICATION SIGN} {image.height()} px)"


# the type of each file format on the clipboard of macOS, and on the clipboard of Linux and Windows
PASTEBOARD_TYPES: t.Final = MappingProxyType({"png": "public.png", "pdf": "com.adobe.pdf", "svg": "public.svg-image"})
CLIPBOARD_MIME_TYPES: t.Final = MappingProxyType({"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"})


def put_file_on_clipboard(data: bytes, suffix: str) -> str | None:
    """Put the content of a PNG, a PDF, or an SVG file on the clipboard. Return an error message, or None.

    On macOS, Qt gives an image to the clipboard only as TIFF. Qt gives a PDF or an SVG file a type of Qt that no
    other application reads. Thus AppKit gives the file the type of macOS there. A PNG file also goes on the clipboard
    as TIFF, for an application that reads no PNG. An SVG file also goes on the clipboard as text, e.g. for an editor
    of SVG code.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    if sys.platform == "darwin":
        try:
            appkit = import_optional("AppKit")
        except ModuleNotFoundError as exc:
            if suffix != "png":
                return str(exc)
        else:
            pasteboard = appkit.NSPasteboard.generalPasteboard()
            pasteboard.clearContents()
            nsdata = appkit.NSData.dataWithBytes_length_(data, len(data))
            pasteboard.setData_forType_(nsdata, PASTEBOARD_TYPES[suffix])
            if suffix == "png":
                pasteboard.setData_forType_(
                    appkit.NSBitmapImageRep.imageRepWithData_(nsdata).TIFFRepresentation(), "public.tiff"
                )
            elif suffix == "svg":
                pasteboard.setString_forType_(data.decode(), "public.utf8-plain-text")
            return None
    mimedata = QtCore.QMimeData()
    mimedata.setData(CLIPBOARD_MIME_TYPES[suffix], QtCore.QByteArray(data))
    if suffix == "png":
        mimedata.setImageData(QtGui.QImage.fromData(data))
    elif suffix == "svg":
        mimedata.setText(data.decode())
    QtWidgets.QApplication.clipboard().setMimeData(mimedata)
    return None


def follow_colour_scheme(
    window: "QtWidgets.QMainWindow", viewer: "PlotViewer[t.Any]", queue: "DrawQueue[t.Any]"
) -> None:
    """Draw the plot again with the colours of each new appearance, e.g. Dark Mode, or of a change in Settings.

    Qt gives the new palette after the signal, thus the plot waits until Qt has no other events. The Settings window
    reaches each window through its property "settingshandler".
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    def draw_with_new_colours() -> None:
        viewer.darkcolours = get_dark_plot_colours()
        queue.redraw()

    def on_colour_scheme() -> None:
        QtCore.QTimer.singleShot(0, window, draw_with_new_colours)

    stylehints = QtGui.QGuiApplication.styleHints()
    window.setProperty("settingshandler", on_colour_scheme)
    stylehints.colorSchemeChanged.connect(on_colour_scheme)
    # the signal of the application stays after the window closes, thus the window removes its handler
    window.destroyed.connect(lambda: stylehints.colorSchemeChanged.disconnect(on_colour_scheme))


def show_figure_in_canvas(oldfig: "mplfig.Figure", newfig: "mplfig.Figure") -> tuple[float, float]:
    """Show a figure of the worker thread in the canvas of the window, and return the size of the figure in inches."""
    canvas = oldfig.canvas
    newfig.set_canvas(canvas)
    canvas.figure = newfig
    canvas.draw_idle()
    figwidth, figheight = newfig.get_size_inches()
    return float(figwidth), float(figheight)


def copy_text(text: str) -> None:
    """Print the text, e.g. the command, and put it on the clipboard."""
    from PySide6 import QtWidgets

    print(text)
    QtWidgets.QApplication.clipboard().setText(text)


# the Figure section selects PNG at the start, because each application can paste a PNG image
DEFAULT_FIGURE_FORMAT: t.Final = "png"


def get_figure_format() -> str:
    """Return the suffix of the format of the figure that the user selected last, e.g. "pdf"."""
    suffix = str(get_settings().value("figureformat", DEFAULT_FIGURE_FORMAT))
    return suffix if suffix in dict(EXPORT_FORMATS) else DEFAULT_FIGURE_FORMAT


def set_figure_format(suffix: str) -> None:
    """Keep the format of the figure, and show it in the Figure section of each window of the viewers."""
    from PySide6 import QtWidgets

    get_settings().setValue("figureformat", suffix)
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if callable(handler := widget.property("figureformathandler")):
            handler(suffix)


class FigureSection(t.NamedTuple):
    """The buttons of the Figure section and its Resolution box, which gives the -dpi of the command."""

    copybutton: "QtWidgets.QPushButton"
    savebutton: "QtWidgets.QPushButton"
    dpibox: "QtWidgets.QSpinBox"


def add_figure_section(
    window: "QtWidgets.QMainWindow", panellayout: "QtWidgets.QVBoxLayout", dpi: int
) -> FigureSection:
    """Add the Figure section: the format of a copied or saved figure, its resolution for PNG, and two buttons.

    dpi is the resolution of the command, and the window connects the Resolution box to the -dpi of its values. The
    box shows only for a PNG file. A PDF or an SVG file takes the resolution for its raster parts, e.g. a colour
    image, and its lines and its text have no resolution. The figure has the size of the plot in the window.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    _, grid = add_section(panellayout, "Figure")
    formatbox = QtWidgets.QComboBox()
    for index, (suffix, tooltip) in enumerate(EXPORT_FORMATS):
        formatbox.addItem(suffix.upper(), suffix)
        formatbox.setItemData(index, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    formatbox.setToolTip("The format of Copy Figure and Save Figure…")
    resolutionlabel = QtWidgets.QLabel("Resolution:")
    dpibox = QtWidgets.QSpinBox()
    dpibox.setRange(10, 2400)
    dpibox.setSingleStep(50)
    dpibox.setValue(dpi)
    dpibox.setSuffix(" dpi")
    dpibox.setKeyboardTracking(False)
    dpibox.setToolTip("The resolution of a PNG file (-dpi)")
    shortcuts = get_menu_shortcut_texts()
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip(f"Put the figure on the clipboard ({shortcuts['Copy Figure']})")
    # the Figure section tells a sighted user what the button copies, and a screen reader reads the whole name
    copybutton.setAccessibleName("Copy Figure")
    savebutton = QtWidgets.QPushButton("Save…")
    savebutton.setToolTip(f"Save the figure in a file ({shortcuts['Save Figure…']})")

    def show_format(suffix: str) -> None:
        # a format from a different window must not call set_figure_format again
        blocker = QtCore.QSignalBlocker(formatbox)
        formatbox.setCurrentIndex(max(formatbox.findData(suffix), 0))
        blocker.unblock()
        # make_row_layout puts the label and the box in one group widget, and a hidden group leaves no gap in the row
        if (resolutiongroup := dpibox.parentWidget()) is not None:
            resolutiongroup.setVisible(suffix == "png")
            # the row puts its groups on lines at a resize only, thus a narrow sidebar needs a new arrangement now
            if (row := resolutiongroup.parentWidget()) is not None:
                QtWidgets.QApplication.sendEvent(row, QtGui.QResizeEvent(row.size(), row.size()))

    def on_format(index: int) -> None:
        set_figure_format(str(formatbox.itemData(index)))

    add_row(grid, 0, [QtWidgets.QLabel("Format:"), formatbox, resolutionlabel, dpibox, copybutton, savebutton])
    show_format(get_figure_format())
    formatbox.currentIndexChanged.connect(on_format)
    window.setProperty("figureformathandler", show_format)
    return FigureSection(copybutton=copybutton, savebutton=savebutton, dpibox=dpibox)


# a GIF file shows on a screen, thus its frames take the resolution of a screen
ANIMATION_DPI: t.Final = 100

# the file formats of the Figure section, with the tooltip of each
EXPORT_FORMATS: t.Final = (
    ("pdf", "A vector file, for a paper. A colour image in it takes the resolution"),
    ("png", "An image with the resolution, e.g. for a slide"),
    ("svg", "A vector file, e.g. for a web page. A colour image in it takes the resolution"),
)


def export_animation(
    window: "QtWidgets.QWidget",
    queue: "DrawQueue[t.Any]",
    statusbar: StatusBar,
    commandmain: "Callable[..., None]",
    commandname: str,
    frames: "tuple[int, Callable[[int], list[str]]]",
    fps: float,
    parser: "SuggestingArgumentParser",
) -> None:
    """Save a GIF file of the steps of Play, with one run of the command for each frame.

    frames gives the count of the frames and a function that gives the command of a frame by its index. The worker
    thread makes the command of each frame, runs it, and joins the frames, thus the window accepts input. A plot
    against time can have a frame for each of 125 000 cells, and a list of their commands took seconds. The GIF shows
    each frame for 1/fps seconds, and each frame has the resolution of a screen.
    """
    import tempfile

    from PySide6 import QtWidgets

    framecount, get_frametokens = frames
    if framecount == 0:
        return
    if framecount > MAX_ANIMATION_FRAMES:
        answer = QtWidgets.QMessageBox.question(
            window,
            "Export Animation",
            f"The animation has {framecount} frames, and each frame runs the command. Continue?",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
    filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        window, "Export the animation", str(Path.cwd() / f"{commandname}.gif"), "GIF (*.gif)"
    )
    if not filename:
        return
    if not Path(filename).suffix:
        filename += ".gif"

    def export() -> str | None:
        with tempfile.TemporaryDirectory() as folder:
            framepaths: list[Path] = []
            for index in range(framecount):
                # the default -dpi suits a printed page, e.g. 600 dpi, and it gave frames larger than a screen
                tokens = [*remove_options(parser, get_frametokens(index), {"dpi"}), "-dpi", str(ANIMATION_DPI)]
                framepath = Path(folder) / f"frame{index:04d}.png"
                commandmain(argsraw=[*tokens, "-o", str(framepath)])
                if not framepath.is_file():
                    return f"The command wrote no frame for {shlex.join(tokens)}"
                framepaths.append(framepath)
            write_gif(filename, framepaths, duration=1000.0 / fps)
        return None

    def show_result(message: str | None) -> None:
        if message is not None:
            show_status_message(statusbar, f"The viewer did not export the animation: {message}", "")
        else:
            show_status_note(statusbar, f"Saved {filename}, with {framecount} frames")

    if not queue.run_task(
        lambda: run_command_step(export), f"Export of {framecount} frames in progress...", show_result
    ):
        show_status_message(statusbar, "A different task is in progress. Export the animation after it", "")


def save_figure_of_command(
    window: "QtWidgets.QWidget",
    statusbar: StatusBar,
    commandmain: "Callable[..., None]",
    commandname: str,
    plottokens: "Sequence[str]",
    parser: "SuggestingArgumentParser",
    choice: tuple[str, int],
) -> None:
    """Save the figure of the command in a file that the user selects, in the format and the resolution of choice.

    The figure comes from the command, thus the file is the same as the output of the command. The command reads a
    name with no suffix as a folder, thus the name takes the suffix of the format. The status bar shows the result.
    plottokens holds no -dpi. A PDF or an SVG file takes the resolution for its raster parts, e.g. a colour image.
    """
    from PySide6 import QtWidgets

    suffix, dpi = choice
    defaultdpi = parser.get_default("dpi")
    filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        window, "Save the figure", str(Path.cwd() / f"{commandname}.{suffix}"), f"{suffix.upper()} (*.{suffix})"
    )
    if not filename:
        return
    if not Path(filename).suffix:
        filename += f".{suffix}"
    savetokens = [*plottokens, *([] if dpi == defaultdpi else ["-dpi", str(dpi)]), "-o", filename]

    def save() -> str | None:
        commandmain(argsraw=savetokens)
        return None

    with show_wait_cursor():
        message = run_command_step(save)
    if message is not None:
        show_status_message(statusbar, f"The command did not save the figure: {message}", "")
    elif not Path(filename).is_file():
        show_status_message(statusbar, f"The command wrote no file at {filename}. The terminal shows its output", "")
    else:
        print(shlex.join(["artistools", commandname, *savetokens]))
        show_status_note(statusbar, f"Saved {filename}{get_pixel_size_text(Path(filename).read_bytes(), suffix)}")


def open_model_window(
    window: "QtWidgets.QWidget",
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    windows: "list[QtWidgets.QMainWindow]",
) -> str | None:
    """Ask for the folder of a run, and open a new window for it. Return an error message if no window opened.

    A SystemExit in a Qt slot ends the process, thus an error of the new window stays in this window.
    """
    from PySide6 import QtWidgets

    # the dialog starts beside the model that the user opened last, where the other runs of a project are
    settings = get_settings()
    startfolder = settings.value("modelfolder", str(Path.cwd()))
    folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Open the folder of an ARTIS run", str(startfolder))
    if not folder:
        return None
    return open_model_folder(folder, open_window, windows)


def open_model_folder(
    folder: str,
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    windows: "list[QtWidgets.QMainWindow]",
) -> str | None:
    """Open a new window for the folder of a run. Return an error message if no window opened.

    A SystemExit in a Qt slot ends the process, thus an error of the new window stays in this window.
    """
    message = run_command_step(lambda: open_window([folder], windows), quiet=False)
    if message is not None:
        return f"The viewer cannot open {folder}: {message}"
    get_settings().setValue("modelfolder", str(Path(folder).parent))
    return None


class ViewerCommand(t.NamedTuple):
    """The command of a viewer, and the functions that give the text of its plot."""

    # the name of the subcommand, e.g. "plotspectra"
    name: str
    main: "Callable[..., None]"
    parser: "SuggestingArgumentParser"
    # the arguments of the plot with no -dpi, because the Figure section gives the resolution
    get_figure_tokens: "Callable[[], list[str]]"
    get_command: "Callable[[], str]"
    get_python_code: "Callable[[], str]"


def add_window_actions(
    window: "QtWidgets.QMainWindow",
    windows: "list[QtWidgets.QMainWindow]",
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    queue: "DrawQueue[t.Any]",
    statusbar: StatusBar,
    command: ViewerCommand,
    figuresection: FigureSection,
    copybuttons: "tuple[QtWidgets.QPushButton, QtWidgets.QPushButton]",
    dpi: "tuple[Callable[[], int | None], Callable[[int | None], None]]",
    show_error: "Callable[[str], None]",
    keyrows: "Sequence[tuple[str, str]]",
    playbutton: "QtWidgets.QAbstractButton | None",
    extracallbacks: "Mapping[str, Callable[[], object]] | None" = None,
) -> "Callable[[QtWidgets.QMenu], None]":
    """Give a window the actions that each viewer has: copy, save, open, help, and the menus.

    copybuttons are the Copy buttons of the command and of the Python code. dpi gives the -dpi of the values, or None
    for the default, and the function that sets it. extracallbacks gives the menu items of one viewer, e.g. Reload
    Data. Return the function that adds the actions on the figure to a context menu of the plot.
    """
    from PySide6 import QtWidgets

    get_dpi, set_dpi = dpi
    defaultdpi: int = command.parser.get_default("dpi")
    callbacks = dict(extracallbacks or {})

    def get_figure_choice() -> tuple[str, int]:
        return get_figure_format(), get_dpi() or defaultdpi

    def on_copy_figure() -> None:
        tokens = command.get_figure_tokens()
        copy_figure_of_command(queue, statusbar, command.main, command.parser, tokens, get_figure_choice())

    def on_save() -> None:
        tokens = command.get_figure_tokens()
        save_figure_of_command(
            window, statusbar, command.main, command.name, tokens, command.parser, get_figure_choice()
        )

    def on_copy() -> None:
        copy_text(command.get_command())
        show_status_note(statusbar, "Copied the command")

    def on_copy_python() -> None:
        copy_text(command.get_python_code())
        show_status_note(statusbar, "Copied the Python code")

    def on_resolution(resolution: int) -> None:
        set_dpi(None if resolution == defaultdpi else resolution)

    def on_open_model() -> None:
        if (message := open_model_window(window, open_window, windows)) is not None:
            show_error(message)

    def on_open_recent(folder: str) -> None:
        if (message := open_model_folder(folder, open_window, windows)) is not None:
            show_error(message)

    def on_help() -> None:
        QtWidgets.QMessageBox.information(window, "Keys and mouse actions", get_keyboard_help(keyrows, menutexts))

    menucallbacks: dict[str, Callable[[], object]] = {
        "Open Model…": on_open_model,
        "Save Figure…": on_save,
        "Close Window": window.close,
        "Copy Figure": on_copy_figure,
        "Copy Command": on_copy,
        "Copy Python": on_copy_python,
        "Keys and Mouse Actions": on_help,
        **callbacks,
    }
    menutexts = add_menus(window, menucallbacks, queue, playbutton, open_folder=on_open_recent)
    figuresection.copybutton.clicked.connect(on_copy_figure)
    figuresection.savebutton.clicked.connect(on_save)
    figuresection.dpibox.valueChanged.connect(on_resolution)
    commandcopybutton, pythoncopybutton = copybuttons
    commandcopybutton.clicked.connect(on_copy)
    pythoncopybutton.clicked.connect(on_copy_python)
    statusbar.helpbutton.clicked.connect(on_help)

    def add_figure_actions(menu: QtWidgets.QMenu) -> None:
        """Add the actions on the figure to a context menu, as the context menu of a Mac app gives them."""
        menu.addAction("Copy Figure").triggered.connect(on_copy_figure)
        menu.addAction("Save Figure…").triggered.connect(on_save)
        if (exportanimation := callbacks.get("Export Animation…")) is not None:
            menu.addAction("Export Animation…").triggered.connect(exportanimation)

    return add_figure_actions


class SeriesRow(t.NamedTuple):
    """One row of the list of series of a viewer, which is an ARTIS model or a file of reference data."""

    path: str
    # the name of the series in the legend, and True if -label gives it
    name: str
    labelled: bool
    # the kind and the full path of the series, e.g. "Model: /data/mymodel"
    itemtext: str
    tooltip: str
    swatch: "QtGui.QPixmap"
    # the reason that the user cannot remove the row, e.g. for the last model, or None
    removereason: str | None
    # a glyph after the path, e.g. the mark of the model that gives the timesteps, and its tooltip
    mark: tuple[str, str] | None
    # the menu items of one viewer, with the text, whether the item is on, and the action
    extraactions: "tuple[tuple[str, bool, Callable[[], object]], ...]"


class ReferenceData(t.NamedTuple):
    """The reference data of the series of a viewer, e.g. the observed spectra."""

    # the kind of one series, e.g. "reference spectrum"
    kind: str
    names: "Sequence[str]"
    folder: Path
    # return the file of a name in the working folder or in the reference data, or None
    find: "Callable[[str], Path | None]"
    # return the token of the command for the path of a file, e.g. a name of the reference data
    get_token: "Callable[[str], str]"
    example: str


class SeriesListActions(t.NamedTuple):
    """The functions of a viewer that the list of series calls."""

    get_paths: "Callable[[], Sequence[str]]"
    # apply a new list of paths, e.g. a new order, an added path, or a removed path
    apply_paths: "Callable[[Sequence[str]], None]"
    edit_label: "Callable[[str], None]"
    edit_style: "Callable[[str], None]"
    # return the full path of a series, e.g. of ".", thus two spellings of one series give one row
    get_full_path: "Callable[[str], Path]"
    # return True for the folder of an ARTIS run. A plot needs one run at least
    is_run: "Callable[[str], bool]"
    show_error: "Callable[[str], None]"


def add_series_list(
    grid: "QtWidgets.QGridLayout",
    window: "QtWidgets.QMainWindow",
    show_note: "Callable[[str], None]",
    listtooltip: str,
    reference: ReferenceData,
    actions: SeriesListActions,
) -> "Callable[[object, Callable[[], Sequence[SeriesRow]]], None]":
    """Add the list of the series of a viewer, with the row that adds a model or a file of reference data.

    Each row shows the number, the image of the line, the name, and the path of a series, and buttons that move it
    and remove it. A double-click gives the series a -label, and the context menu of a row gives each action. A drag
    or Alt-Up and Alt-Down change the order. A folder or a file that the user drops on the window adds a series.

    Return the function that shows the rows. It takes a key of the parts of the values that the rows show, and it
    makes the rows again only for a new key, because each row reads the name of its series.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    def move_row(row: int, step: int) -> None:
        """Move the series of a row one row up or down, and select it."""
        paths = list(actions.get_paths())
        if row < 0 or not 0 <= row + step < len(paths):
            return
        paths.insert(row + step, paths.pop(row))
        actions.apply_paths(paths)
        serieslist.setCurrentRow(row + step)

    serieslist = make_reorder_list(lambda step: move_row(serieslist.currentRow(), step))
    # the widget of each row shows the text beside its buttons, thus the list draws no text of its own
    serieslist.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    serieslist.setToolTip(listtooltip)
    # a click opens the dialog, and the arrow opens the menu of the recent models
    addmodelbutton = QtWidgets.QToolButton()
    addmodelbutton.setText("Add Model…")
    addmodelbutton.setToolTip("Add the folder of an ARTIS model. The arrow shows the recent models")
    addmodelbutton.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
    recentmodelsmenu = QtWidgets.QMenu(addmodelbutton)
    addmodelbutton.setMenu(recentmodelsmenu)
    referenceedit = QtWidgets.QLineEdit()
    referenceedit.setPlaceholderText(f"Add a {reference.kind}, e.g. {reference.example}")
    referenceedit.setToolTip(
        f"Type part of the name of a {reference.kind} in the data of artistools, then press Return. A name of a file"
        " in the working folder also works."
    )
    referencecompleter = make_completer(reference.names, referenceedit)
    referenceedit.setCompleter(referencecompleter)
    openreferencebutton = QtWidgets.QPushButton("Open…")
    openreferencebutton.setToolTip(f"Add the file of a {reference.kind} from a folder")
    addrow = QtWidgets.QHBoxLayout()
    addrow.addWidget(referenceedit, 1)
    addrow.addWidget(openreferencebutton)
    addrow.addWidget(addmodelbutton)
    grid.addWidget(serieslist, 0, 0, 1, -1)
    grid.addLayout(addrow, 1, 0, 1, -1)

    # the key and the paths of the rows that show_rows made. A drag moves the rows, and the rows then show old numbers
    # and old buttons until show_rows makes them again
    shown: list[tuple[object, tuple[str, ...]] | None] = [None]

    def add_paths(paths: "Sequence[str]") -> None:
        """Add each series whose full path the list does not hold yet. A cancelled dialog gives no path."""
        if not paths:
            return
        current = list(actions.get_paths())
        fullpaths = {actions.get_full_path(path) for path in current}
        newpaths: list[str] = []
        for path in paths:
            if (fullpath := actions.get_full_path(path)) not in fullpaths:
                fullpaths.add(fullpath)
                newpaths.append(path)
        if not newpaths:
            actions.show_error("The list already holds each of these series")
            return
        actions.apply_paths((*current, *newpaths))

    def remove_path(path: str) -> None:
        paths = [other for other in actions.get_paths() if other != path]
        # the time controls read the timesteps of a run, thus the plot needs an ARTIS model
        if not any(actions.is_run(other) for other in paths):
            actions.show_error(
                "The plot needs one ARTIS model at least. Add a different model before you remove this one"
            )
            return
        actions.apply_paths(paths)

    def copy_path(path: str) -> None:
        copy_text(str(actions.get_full_path(path)))
        show_note("Copied the path")

    def open_folder(path: str) -> None:
        """Open the folder of a model, or the folder that holds a reference file, in the file manager."""
        fullpath = actions.get_full_path(path)
        folder = fullpath if fullpath.is_dir() else fullpath.parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def make_row(index: int, count: int, seriesrow: SeriesRow) -> QtWidgets.QWidget:
        path, name = seriesrow.path, seriesrow.name
        row = QtWidgets.QWidget()
        row.setToolTip(seriesrow.tooltip)
        rowlayout = QtWidgets.QHBoxLayout(row)
        rowlayout.setContentsMargins(4, 0, 2, 0)
        rowlayout.setSpacing(2)
        rowlayout.addWidget(QtWidgets.QLabel(f"<b>{index + 1}</b>"))
        swatch = QtWidgets.QToolButton()
        swatch.setAutoRaise(True)
        swatch.setIconSize(QtCore.QSize(36, 14))
        swatch.setIcon(QtGui.QIcon(seriesrow.swatch))
        swatch.setToolTip("The colour and the line style of the series in the plot. Click to change them")
        swatch.setAccessibleName(f"Set the style of {name}")
        # the new list replaces this row, thus each action of a button waits until the click ends
        swatch.clicked.connect(partial(QtCore.QTimer.singleShot, 0, window, partial(actions.edit_style, path)))
        rowlayout.addWidget(swatch)
        namelabel = QtWidgets.QLabel(name)
        namelabel.setToolTip(
            f"The -label of the series: {name}. Double-click the row to change it"
            if seriesrow.labelled
            else f"The name of the series in the legend: {name}. Double-click the row to give it a -label"
        )
        rowlayout.addWidget(namelabel)
        rowlayout.addSpacing(6)
        # a long path shows its start and its end, and the width of the box sets the length
        pathlabel = make_elided_label(seriesrow.itemtext)
        pathlabel.setEnabled(False)
        rowlayout.addWidget(pathlabel, 1)
        if seriesrow.mark is not None:
            marklabel = QtWidgets.QLabel(seriesrow.mark[0])
            marklabel.setToolTip(seriesrow.mark[1])
            rowlayout.addWidget(marklabel)
        for glyph, tooltip, enabled, action in (
            ("▲", f"Move {name} up (Alt-Up)", index > 0, partial(move_row, index, -1)),
            ("▼", f"Move {name} down (Alt-Down)", index < count - 1, partial(move_row, index, 1)),
            ("✕", f"Remove {name} from the plot", seriesrow.removereason is None, partial(remove_path, path)),
        ):
            button = make_glyph_button(glyph, tooltip, tooltip)
            button.setEnabled(enabled)
            if glyph == "✕" and seriesrow.removereason is not None:
                button.setToolTip(seriesrow.removereason)
            button.clicked.connect(partial(QtCore.QTimer.singleShot, 0, window, action))
            rowlayout.addWidget(button)
        grip = QtWidgets.QLabel("≡")
        grip.setEnabled(False)
        grip.setToolTip("Drag the row to move the series")
        grip.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        rowlayout.addWidget(grip)
        # the context menu gives each action, thus the keyboard and VoiceOver can also reach them
        row.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        for text, enabled, action in (
            ("Move to Top", index > 0, partial(move_row, index, -index)),
            ("Move to Bottom", index < count - 1, partial(move_row, index, count - 1 - index)),
            *seriesrow.extraactions,
            ("Set Label…", True, partial(actions.edit_label, path)),
            ("Set Style…", True, partial(actions.edit_style, path)),
            ("Copy Path", True, partial(copy_path, path)),
            ("Open Folder", not is_remote_path(path), partial(open_folder, path)),
            ("Remove", seriesrow.removereason is None, partial(remove_path, path)),
        ):
            rowaction = QtGui.QAction(text, row)
            rowaction.setEnabled(enabled)
            # the new list replaces this row, thus each action waits until the menu closes
            rowaction.triggered.connect(partial(QtCore.QTimer.singleShot, 0, window, action))
            row.addAction(rowaction)
        return row

    def show_rows(key: object, make_rows: "Callable[[], Sequence[SeriesRow]]") -> None:
        paths = tuple(actions.get_paths())
        order = tuple(
            serieslist.item(index).data(QtCore.Qt.ItemDataRole.UserRole) for index in range(serieslist.count())
        )
        if shown[0] == (key, paths) and order == paths:
            return
        rows = make_rows()
        serieslist.clear()
        rowheight = serieslist.fontMetrics().lineSpacing() + 4
        for index, seriesrow in enumerate(rows):
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, seriesrow.path)
            row = make_row(index, len(rows), seriesrow)
            rowheight = max(rowheight, row.sizeHint().height())
            serieslist.addItem(item)
            serieslist.setItemWidget(item, row)
        for index in range(serieslist.count()):
            if (item := serieslist.item(index)) is not None:
                item.setSizeHint(QtCore.QSize(0, rowheight))
        # the list has the height of its series, from 2 to 4 rows, and a longer list scrolls
        shownrows = min(max(serieslist.count(), 2), 4)
        serieslist.setFixedHeight(shownrows * rowheight + 2 * serieslist.frameWidth() + 4)
        shown[0] = (key, paths)

    def on_rows_dropped() -> None:
        """Apply the order of the rows after a drag.

        Qt can move a dropped row, or it can insert a copy and then remove the source row. Until the removal, the
        list holds one series two times, thus the function waits for the removal. A copy has no row widget, thus the
        next show_rows makes the rows again.
        """
        items = [serieslist.item(index) for index in range(serieslist.count())]
        order = [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in items]
        paths = list(actions.get_paths())
        if sorted(order) != sorted(paths):
            return
        if order == paths and all(serieslist.itemWidget(item) is not None for item in items):
            return
        # the rows show the numbers and the buttons of the old order, thus show_rows makes them again
        shown[0] = None
        actions.apply_paths(order)

    def get_start_folder() -> Path:
        # the dialog shows only local folders, thus it opens in the working folder for a first run on a different host
        runs = [path for path in actions.get_paths() if actions.is_run(path)]
        if not runs or is_remote_path(runs[0]):
            return Path.cwd()
        return actions.get_full_path(runs[0]).parent

    def add_model_folder(folder: str) -> None:
        if not actions.is_run(folder):
            actions.show_error(f"{folder} is not the folder of an ARTIS run, which holds input.txt")
            return
        add_recent_model(folder)
        add_paths([folder])

    def on_add_model() -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Add an ARTIS model", str(get_start_folder()))
        if folder:
            add_model_folder(folder)

    def show_recent_models() -> None:
        """Fill the menu of Add Model with the recent models that the list does not hold."""
        recentmodelsmenu.clear()
        fullpaths = {actions.get_full_path(path) for path in actions.get_paths()}
        folders = [folder for folder in get_recent_models() if actions.get_full_path(folder) not in fullpaths]
        for folder in folders:
            action = recentmodelsmenu.addAction(Path(folder).name)
            action.setToolTip(folder)
            # a test of a remote folder starts ssh, thus the menu enables each remote model and does not test its folder
            action.setEnabled(is_remote_path(folder) or Path(folder).is_dir())
            action.triggered.connect(partial(add_paths, [folder]))
        if not folders:
            recentmodelsmenu.addAction("No Recent Models").setEnabled(False)

    def on_open_reference() -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(window, f"Add a {reference.kind}", str(reference.folder))
        add_paths([reference.get_token(filename) for filename in filenames])

    def add_reference_name(name: str) -> None:
        name = name.strip()
        if not name:
            return
        if reference.find(name) is None:
            actions.show_error(f"No {reference.kind} {name} is in the working folder or in the reference data")
            return
        referenceedit.clear()
        add_paths([name])

    def on_complete_reference(name: str) -> None:
        # the completer puts the name in the field after this handler, thus clear the field after the event
        QtCore.QTimer.singleShot(0, referenceedit.clear)
        add_reference_name(name)

    def on_drop(paths: list[str]) -> None:
        """Add each dropped ARTIS run and each dropped file of reference data to the series of the plot."""
        folders = [path for path in paths if Path(path).is_dir()]
        runs = [folder for folder in folders if actions.is_run(folder)]
        if len(runs) < len(folders):
            actions.show_error("A dropped folder is not the folder of an ARTIS run, which holds input.txt")
        add_paths([*runs, *(reference.get_token(path) for path in paths if Path(path).is_file())])

    addmodelbutton.clicked.connect(on_add_model)
    recentmodelsmenu.aboutToShow.connect(show_recent_models)

    def on_double_click(item: QtWidgets.QListWidgetItem) -> None:
        actions.edit_label(item.data(QtCore.Qt.ItemDataRole.UserRole))

    serieslist.itemDoubleClicked.connect(on_double_click)
    # the list changes its rows at the end of the drop, thus the new order applies after the drop
    for rowsignal in (serieslist.model().rowsMoved, serieslist.model().rowsInserted, serieslist.model().rowsRemoved):
        rowsignal.connect(lambda: QtCore.QTimer.singleShot(0, window, on_rows_dropped))
    openreferencebutton.clicked.connect(on_open_reference)
    referencecompleter.activated.connect(on_complete_reference)
    referenceedit.returnPressed.connect(lambda: add_reference_name(referenceedit.text()))
    set_drop_handler(window, on_drop)
    return show_rows


def clear_output_caches() -> None:
    """Clear each cache of the files that ARTIS writes while it runs, e.g. spec.out, in this process.

    The files of the input of a run stay the same, e.g. input.txt and model.txt, thus their caches stay.
    """
    from artistools.misc.modelinfo import get_runfolder_timesteps_cached
    from artistools.misc.timesteps import get_deposition_cached
    from artistools.misc.timesteps import get_escaped_arrivalrange_cached
    from artistools.misc.timesteps import get_timestep_times_cached
    from artistools.packets.core import check_packets_batch_parquet_paths
    from artistools.packets.core import find_first_rank_textfiles
    from artistools.spectra.core import get_flux_contributions_cached
    from artistools.spectra.core import get_vspecpol_data_cached
    from artistools.spectra.core import read_spec_cached
    from artistools.spectra.core import read_spec_res_cached
    from artistools.spectra.core import read_specpol_res_cached

    for cachedfunction in (
        get_runfolder_timesteps_cached,
        get_deposition_cached,
        get_escaped_arrivalrange_cached,
        get_timestep_times_cached,
        check_packets_batch_parquet_paths,
        find_first_rank_textfiles,
        get_flux_contributions_cached,
        get_vspecpol_data_cached,
        read_spec_cached,
        read_spec_res_cached,
        read_specpol_res_cached,
    ):
        cachedfunction.cache_clear()


@on_model_host
def clear_output_caches_of_run(runfolder: Path) -> None:
    """Clear the caches of the output files on the host of a run. A local run clears the caches of this process."""
    del runfolder
    clear_output_caches()


def reload_runs(
    queue: "DrawQueue[t.Any]",
    runfolders: "Sequence[Path | str]",
    on_reloaded: "Callable[[], None]",
    show_error: "Callable[[str], None]",
) -> None:
    """Read the runs again in the worker thread, e.g. while ARTIS writes more timesteps, then call on_reloaded.

    The caches of this process and of the host of a remote run hold old data, thus the function clears both. The
    reload waits for the plot in progress, and a new plot waits for the reload.
    """

    def read() -> None:
        clear_output_caches()
        for runfolder in runfolders:
            clear_output_caches_of_run(Path(runfolder))

    def on_done(message: str | None) -> None:
        if message is not None:
            show_error(f"The viewer cannot reload the runs: {message}")
            return
        on_reloaded()

    if not queue.run_task(lambda: run_command_step(read, quiet=False), "Reload in progress...", on_done):
        show_error("A different task is in progress. Reload the runs after it")


def get_new_figwidthscale(
    plotarea: "QtWidgets.QWidget",
    figsize: tuple[float, float],
    figwidthscale: float,
    get_fitted: "Callable[[float, float], float]",
) -> float | None:
    """Return the -figwidthscale that fills the plot area, or None if the plot can keep its scale.

    get_fitted gives the fitted scale for the width and the height of the area. A change of less than FIT_TOLERANCE
    keeps the scale.
    """
    area = plotarea.contentsRect()
    if area.width() <= 0 or area.height() <= 0 or figsize[0] <= 0.0:
        return None
    fitted = get_fitted(area.width(), area.height())
    return fitted if abs(fitted - figwidthscale) > FIT_TOLERANCE * figwidthscale else None


class PlotViewer[ValuesT](t.Protocol):
    """A viewer with the values of its controls, the last warning of its plot, and the colours of Dark Mode."""

    values: ValuesT
    # the figure of the canvas, and its size in inches, which render_command sets for each plot
    fig: "mplfig.Figure"
    figsize: tuple[float, float]
    # the last warning of the last plot, which the status bar shows. A user of the application sees no terminal
    warning: str
    # the background and the foreground of the plot in Dark Mode, or None for the usual colours
    darkcolours: tuple[str, str] | None


def render_command[PlotT](
    viewer: "PlotViewer[t.Any]",
    draw: "Callable[[mplfig.Figure], PlotT | str]",
    keep: "Callable[[PlotT], None]",
    *,
    quiet: bool,
) -> "Callable[[], str | None]":
    """Draw a plot on a new figure, and return the function that shows it in the canvas of the viewer.

    draw parses the command and draws the plot on the empty figure that it receives. It returns the frames and the
    data that the window reads, or the reason that it rejects the values. keep gives these to the viewer. Each plot of
    each viewer gets the same last steps: the titles, the colours of Dark Mode, and the layout of the text.

    A worker thread can run this function, because it changes nothing that the window reads. The function that it
    returns must run in the thread of the window. That function returns the reason for the status line if the command
    rejects the values, and the old plot then stays.
    """
    import matplotlib.figure as mplfig
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    plots: list[tuple[mplfig.Figure, PlotT]] = []

    def make_plot() -> str | None:
        fig = mplfig.Figure()
        FigureCanvasAgg(fig)
        result = draw(fig)
        if isinstance(result, str):
            return result
        for axis in fig.axes:
            fix_title_position(axis)
        make_room_for_title(fig)
        if (darkcolours := viewer.darkcolours) is not None:
            apply_dark_colours(fig, *darkcolours)
        # the worker makes the ticks and the text layout, thus the first draw in the window is faster. On the test
        # model, the window draw of a spectrum took 33 ms in place of 44 ms, and of estimators 73 ms in place of 120 ms
        fig.draw_without_rendering()
        plots.append((fig, result))
        return None

    message, warning = run_command_step_with_warning(make_plot, quiet=quiet)

    def show_plot() -> str | None:
        viewer.warning = warning
        if message is not None:
            return message
        fig, result = plots[0]
        viewer.figsize = show_figure_in_canvas(viewer.fig, fig)
        viewer.fig = fig
        keep(result)
        return None

    return show_plot


def changes_values[ValuesT](
    values: ValuesT, current: ValuesT, keep_on_undo: "Callable[[ValuesT, ValuesT], ValuesT] | None"
) -> bool:
    """Return True if Undo or Redo to values gives values that differ from the current values.

    keep_on_undo keeps the parts that the window sets, e.g. the width of the figure. Thus an entry that differs only
    in those parts gives no step.
    """
    return (keep_on_undo(values, current) if keep_on_undo is not None else values) != current


# the time of a plot before the spinner shows over the plot
BUSY_MILLISECONDS: t.Final = 300


def get_plot_time_text(plotseconds: float) -> str:
    """Return the text of the status bar for the time of the last plot, or no text before the first plot."""
    return f"Plot time: {plotseconds:.2f} s" if plotseconds > 0.0 else ""


class DrawQueue[ValuesT]:
    """The plots of a window: a change shows its values at once, and the plot follows when Qt has no other events.

    A drag gives a new value for each movement of the mouse, and a plot can take seconds. The queue draws only the
    last values that the user gave. viewer.values holds the last values that the user gave, and drawnvalues holds the
    values of the plot. If the command rejects new values, the viewer keeps the values of the plot.

    A worker thread draws each plot. The window then shows each new value of a drag at once.

    The queue also keeps the values before each change of the user, thus Undo and Redo can return to them.
    """

    def __init__(
        self,
        window: "QtCore.QObject",
        viewer: PlotViewer[ValuesT],
        statusbar: StatusBar,
        show_values: "Callable[[], None]",
        after_draw: "Callable[[str | None], None]",
        render: "Callable[[ValuesT], Callable[[], str | None]]",
        keep_on_undo: "Callable[[ValuesT, ValuesT], ValuesT] | None" = None,
    ) -> None:
        """Make an empty queue. after_draw receives the message of each plot of the queue.

        render draws the plot of the values in a worker thread. It returns the function that shows that plot in the
        window and gives the message of a rejection.

        keep_on_undo receives the values that Undo or Redo restores and the current values. It returns the restored
        values with the parts that the window sets and the user does not, e.g. the width of the figure.
        """
        from concurrent.futures import ThreadPoolExecutor

        from PySide6 import QtCore

        self.window = window
        self.viewer = viewer
        self.statusbar = statusbar
        self.show_values = show_values
        self.after_draw = after_draw
        self.requestedvalues: ValuesT | None = None
        self.drawnvalues: ValuesT = viewer.values
        self.render = render
        self.keep_on_undo = keep_on_undo
        # the values before each change of the user, for Undo, and the values that Undo replaced, for Redo
        self.undovalues: list[ValuesT] = []
        self.redovalues: list[ValuesT] = []
        self.lastchangetime = -math.inf
        # the text field that gave the change of requestedvalues, the text field that gave the plot in progress, and
        # the text field that has the mark of a rejection. A rejection marks the field of its own plot
        self.editedfield: QtWidgets.QLineEdit | None = None
        self.renderedfield: QtWidgets.QLineEdit | None = None
        self.errorfield: QtWidgets.QLineEdit | None = None
        self.renderedvalues: ValuesT = viewer.values
        self.rendering: Future[Callable[[], str | None]] | None = None
        self.renderstart = 0.0
        # the time of the last plot, which sets the pause of Play
        self.plotseconds = 0.0
        # a task of the worker thread that is not a plot, e.g. Reload Data, from run_task to its end.
        # taskfuture is None until the plot in progress ends
        self.task: Callable[[], str | None] | None = None
        self.taskstatus = ""
        self.on_task_done: Callable[[str | None], None] | None = None
        self.taskfuture: Future[str | None] | None = None
        # True after Cancel Plot, until the plot in progress ends. The queue then discards that plot
        self.discardrendering = False
        # the values before each change that waits for its plot, and the undo and redo lists before that change
        self.pendinghistory: list[tuple[ValuesT, list[ValuesT], list[ValuesT]]] = []
        # one worker thread draws one plot at a time, and a drag during a plot waits for the end of that plot
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plot")
        # the window thread checks the worker at each tick, because a Qt call from the worker thread is not safe
        self.rendertimer = QtCore.QTimer(window)
        self.rendertimer.setInterval(10)
        self.rendertimer.timeout.connect(self.show_rendered)
        # a fast plot shows no spinner, thus the spinner does not flash at each step of a drag
        self.busytimer = QtCore.QTimer(window)
        self.busytimer.setSingleShot(True)
        self.busytimer.setInterval(BUSY_MILLISECONDS)
        self.busytimer.timeout.connect(partial(set_plot_busy, window, busy=True))

    def apply(self, values: ValuesT, *, undoable: bool = True) -> None:
        """Show the new values now, and draw them when Qt has no other events. Only a change of the values draws a plot.

        A change that the window makes, e.g. a step of Play or a fit to the size of the window, is not undoable.
        """
        if values == self.viewer.values:
            # a handler that clamps a control to the old values must still move the control back
            self.show_values()
            return
        # Cancel Plot returns to the history of the plot on the screen
        if self.viewer.values == self.drawnvalues:
            self.pendinghistory.clear()
        self.pendinghistory.append((self.viewer.values, [*self.undovalues], [*self.redovalues]))
        if undoable:
            now = time.monotonic()
            # a drag of a slider or a repeated key gives many changes, and one step of Undo reverts all of them
            if not self.undovalues or now - self.lastchangetime > UNDO_MERGE_SECONDS:
                self.undovalues = [*self.undovalues[-UNDO_LIMIT + 1 :], self.viewer.values]
            self.lastchangetime = now
            self.redovalues.clear()
        # the plot of the change comes later, and a rejection then marks the field that gave the change
        self.editedfield = get_edited_field(self.window)
        # each handler makes its values from viewer.values, thus a second change before the plot keeps the first
        self.viewer.values = values
        self.redraw()

    def can_undo(self) -> bool:
        """Return whether Undo has values that differ from the current values."""
        return any(changes_values(values, self.viewer.values, self.keep_on_undo) for values in self.undovalues)

    def can_redo(self) -> bool:
        """Return whether Redo has values that differ from the current values."""
        return any(changes_values(values, self.viewer.values, self.keep_on_undo) for values in self.redovalues)

    def undo(self) -> None:
        """Return to the values before the last change of the user."""
        self.step_history(self.undovalues, self.redovalues)

    def redo(self) -> None:
        """Return to the values that the last Undo replaced."""
        self.step_history(self.redovalues, self.undovalues)

    def step_history(self, source: list[ValuesT], target: list[ValuesT]) -> None:
        """Take the last values of source that differ from the current values, and keep the current values in target.

        The command can reject a change, and the viewer then keeps the old values. Such a change gives values in
        source that are the same as the current values, and a step of Undo skips them.
        """
        while source and not changes_values(source[-1], self.viewer.values, self.keep_on_undo):
            source.pop()
        if not source:
            return
        if self.viewer.values == self.drawnvalues:
            self.pendinghistory.clear()
        self.pendinghistory.append((self.viewer.values, [*self.undovalues], [*self.redovalues]))
        restored = source.pop()
        if self.keep_on_undo is not None:
            restored = self.keep_on_undo(restored, self.viewer.values)
        target.append(self.viewer.values)
        # the next change of the user starts a new step of Undo
        self.lastchangetime = -math.inf
        self.viewer.values = restored
        self.redraw()

    def redraw(self) -> None:
        """Show the values of the viewer, and draw them when Qt has no other events, e.g. after new data of the run."""
        from PySide6 import QtCore

        if self.requestedvalues is None:
            QtCore.QTimer.singleShot(0, self.window, self.draw_requested)
        self.requestedvalues = self.viewer.values
        self.show_values()

    def draw_requested(self) -> None:
        """Start the plot of the last values that the user gave, unless the worker thread is busy."""
        if self.rendering is None and self.task is None:
            self.start_render()

    def start_render(self) -> None:
        """Start a plot of the last values that the user gave in the worker thread."""
        values, self.requestedvalues = self.requestedvalues, None
        if values is None:
            return
        self.renderedvalues = values
        self.renderedfield, self.editedfield = self.editedfield, None
        # the status bar keeps the time of the last plot, and the spinner over the plot shows the plot in progress
        self.renderstart = time.perf_counter()
        self.rendering = self.executor.submit(self.render, values)
        self.rendertimer.start()
        if not self.busytimer.isActive():
            self.busytimer.start()

    def is_busy(self) -> bool:
        """Return True if a plot is in progress or waits, and Cancel Plot can stop the wait."""
        return (self.rendering is not None and not self.discardrendering) or self.requestedvalues is not None

    def cancel(self) -> None:
        """Stop the wait for the plot in progress and for the plots that wait, and keep the plot on the screen.

        A worker thread cannot stop, thus the plot in progress runs to its end, and the queue then discards it. The
        controls show the values of the plot on the screen again.
        """
        if not self.is_busy():
            return
        self.requestedvalues, self.editedfield = None, None
        self.discardrendering = self.rendering is not None
        self.viewer.values = self.drawnvalues
        # Cancel Plot also removes its changes from Undo and Redo
        for values, undovalues, redovalues in reversed(self.pendinghistory):
            if values == self.drawnvalues:
                self.undovalues, self.redovalues = undovalues, redovalues
                break
        self.pendinghistory.clear()
        self.lastchangetime = -math.inf
        self.busytimer.stop()
        set_plot_busy(self.window, busy=False)
        self.statusbar.drawtime.setText("Plot cancelled")
        self.show_values()

    def show_rendered(self) -> None:
        """Show the plot or the result of the task of the worker thread when it is complete.

        The task or the plot that waits then starts.
        """
        if self.rendering is not None:
            if not self.rendering.done():
                return
            self.show_rendered_plot(self.rendering)
        elif self.taskfuture is not None:
            if not self.taskfuture.done():
                return
            self.end_task(self.taskfuture)
        self.rendertimer.stop()
        # a task waits for the plot in progress, and the newer values of the user wait for the task
        if self.task is not None and self.taskfuture is None:
            self.start_task()
        elif self.task is None and self.requestedvalues is not None:
            self.start_render()
        # the spinner stays during a drag, which starts a new plot at the end of each plot
        if self.rendering is None and self.taskfuture is None:
            self.busytimer.stop()
            set_plot_busy(self.window, busy=False)

    def show_rendered_plot(self, rendering: "Future[Callable[[], str | None]]") -> None:
        """Show the complete plot of the worker thread, or the message of a rejection."""
        self.rendering = None
        if self.discardrendering:
            self.discardrendering, self.renderedfield = False, None
            return
        try:
            message = rendering.result()()
        except Exception as exc:  # ruff:ignore[blind-except]
            # the render wraps the command in run_command_step, thus only a defect of the viewer arrives here
            print_error(traceback.format_exc())
            message = f"{type(exc).__name__}: {get_first_line(str(exc))}"
        if message is None:
            self.drawnvalues = self.renderedvalues
        # newer values of the user stay, and the next plot draws them
        if self.requestedvalues is None:
            self.viewer.values = self.drawnvalues
        self.show_plot_status(message)
        self.after_draw(message)

    def run_task(
        self, task: "Callable[[], str | None]", statustext: str, on_done: "Callable[[str | None], None]"
    ) -> bool:
        """Run a task in the worker thread, e.g. Reload Data. Return False if a task is in progress.

        The task starts after the plot in progress, and a new plot waits for the end of the task. Thus a plot never
        reads data that the task replaces. The status bar shows statustext while the task runs. on_done receives the
        message of the task in the thread of the window.
        """
        if self.task is not None:
            return False
        self.task, self.taskstatus, self.on_task_done = task, statustext, on_done
        if self.rendering is None:
            self.start_task()
        return True

    def start_task(self) -> None:
        """Start the task of run_task in the worker thread."""
        if self.task is None:
            return
        self.statusbar.drawtime.setText(self.taskstatus)
        self.taskfuture = self.executor.submit(self.task)
        self.rendertimer.start()
        if not self.busytimer.isActive():
            self.busytimer.start()

    def end_task(self, taskfuture: "Future[str | None]") -> None:
        """Give the message of the complete task to the function of run_task."""
        on_done = self.on_task_done
        self.task, self.taskfuture, self.on_task_done = None, None, None
        # the status of the task replaced the time of the last plot, thus that time shows again
        self.statusbar.drawtime.setText(get_plot_time_text(self.plotseconds))
        try:
            message = taskfuture.result()
        except Exception as exc:  # ruff:ignore[blind-except]
            print_error(traceback.format_exc())
            message = f"{type(exc).__name__}: {get_first_line(str(exc))}"
        if on_done is not None:
            on_done(message)

    def close(self) -> None:
        """Cancel the plots that wait in the worker thread.

        A window calls this function when it closes. A plot or a task in progress runs to its end, and the process
        ends after it. Without this function, each plot that waits also runs before the process ends.
        """
        self.executor.shutdown(wait=False, cancel_futures=True)

    def show_plot_status(self, message: str | None) -> None:
        """Show the time of the plot that ended, its message or its warning, and the values."""
        self.plotseconds = time.perf_counter() - self.renderstart
        self.statusbar.drawtime.setText(get_plot_time_text(self.plotseconds))
        # the readout holds the values of the old plot until the mouse moves again
        self.statusbar.readout.setText("")
        hide_readout_tag(self.window)
        show_status_message(self.statusbar, message, self.viewer.warning)
        show_plot_banner(self.window, message)
        # show_values can replace the field of the plot, e.g. with a new card of a subplot, before the plot ends
        field, self.renderedfield = self.renderedfield, None
        rejectedfield = field if message is not None and field is not None and is_live(field) else None
        if self.errorfield is not None and (message is None or rejectedfield is not None):
            clear_field_error(self.errorfield)
            self.errorfield = None
        if rejectedfield is not None and message is not None:
            mark_field_error(rejectedfield, message)
            self.errorfield = rejectedfield
        self.show_values()


def set_edit_text(edit: "QtWidgets.QLineEdit", text: str) -> None:
    """Show the text in a field, unless the user types in that field.

    A plot or a Play step can end while the user types. Without this check, the text of the values replaces the
    text that the user typed. The handler of a field calls setModified(False), thus a field shows new values again
    after the user presses Return.
    """
    if not (edit.hasFocus() and edit.isModified()):
        edit.setText(text)


def set_spin_value(box: "QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox", value: float) -> None:
    """Show the value in a spin box, unless the user types in that box.

    A box with no keyboard tracking keeps the typed text until the user presses Return. setValue writes the text
    again also for the same value, thus the end of a plot erased the text that the user typed.
    """
    from PySide6 import QtWidgets

    if box.hasFocus() and math.isclose(box.value(), value):
        return
    if isinstance(box, QtWidgets.QSpinBox):
        box.setValue(round(value))
    else:
        box.setValue(value)


def connect_plot_mouse(
    canvas: "FigureCanvasBase",
    get_frames: "Callable[[], Sequence[mplax.Axes]]",
    get_readout: "Callable[[t.Any, mplax.Axes], str]",
    readoutlabel: "QtWidgets.QLabel",
    on_select: "Callable[[float, float], None]",
    on_reset: "Callable[[], None]",
    can_select: "Callable[[], bool]",
    on_select_y: "Callable[[int, float, float], None] | None" = None,
    on_menu: "Callable[[int, t.Any], None] | None" = None,
    show_tag: "Callable[[t.Any, str], None] | None" = None,
) -> "Callable[[], None]":
    """Give the plot a readout under the pointer, a drag across a frame that selects an x range, and a double-click.

    on_select receives the two x values of a drag, and on_reset receives a double-click on a frame. on_select_y
    receives the index of the frame and the two y values of a drag with the Shift key inside one frame. on_menu
    receives the index of the frame and the matplotlib event of a click with the right button. show_tag receives the
    matplotlib event and the readout, which is empty when the pointer leaves the frames. matplotlib keeps the
    connections in the figure. Call the returned function after the canvas receives a new figure.
    """
    # the data value and the pixel of the start of a drag, the span that shows it, and its frame
    dragstart: tuple[float, float] | None = None
    dragspan: t.Any = None
    dragframeindex = 0
    dragvertical = False

    def get_frame_index(event: t.Any) -> int | None:
        return next((index for index, axis in enumerate(get_frames()) if event.inaxes is axis), None)

    def on_press(event: t.Any) -> None:
        nonlocal dragstart, dragspan, dragframeindex, dragvertical
        frameindex = get_frame_index(event)
        if frameindex is None or event.xdata is None:
            return
        if event.button == 3:
            if on_menu is not None:
                on_menu(frameindex, event)
            return
        if event.button != 1:
            return
        if event.dblclick:
            on_reset()
            return
        dragframeindex = frameindex
        # the canvas has no keyboard focus, thus matplotlib gives no key, and the modifiers hold the Shift key
        dragvertical = "shift" in event.modifiers and on_select_y is not None
        if dragvertical:
            dragstart = (event.ydata, event.y)
            dragspan = event.inaxes.axhspan(event.ydata, event.ydata, color="0.5", alpha=0.3)
        elif can_select():
            dragstart = (event.xdata, event.x)
            dragspan = event.inaxes.axvspan(event.xdata, event.xdata, color="0.5", alpha=0.3)

    def on_motion(event: t.Any) -> None:
        frameindex = get_frame_index(event)
        readout = get_readout(event, event.inaxes) if frameindex is not None and event.xdata is not None else ""
        readoutlabel.setText(readout)
        if show_tag is not None:
            show_tag(event, readout)
        if dragstart is None or dragspan is None or event.xdata is None or frameindex is None:
            return
        # the frames share the x axis but not the y axis, thus a y value of a different frame does not apply
        if dragvertical and frameindex != dragframeindex:
            return
        if dragvertical:
            dragspan.set_y(min(dragstart[0], event.ydata))
            dragspan.set_height(abs(event.ydata - dragstart[0]))
        else:
            dragspan.set_x(min(dragstart[0], event.xdata))
            dragspan.set_width(abs(event.xdata - dragstart[0]))
        canvas.draw_idle()

    def on_release(event: t.Any) -> None:
        nonlocal dragstart, dragspan
        if dragstart is None or dragspan is None:
            return
        # a plot during the drag clears the figure, and the span then went with the old frames
        if dragspan.axes in canvas.figure.axes:
            dragspan.remove()
        start, dragstart, dragspan = dragstart, None, None
        canvas.draw_idle()
        if event.xdata is None or get_frame_index(event) is None:
            return
        # a movement of a few pixels is a click and not a selection
        if dragvertical:
            if on_select_y is not None and get_frame_index(event) == dragframeindex and abs(event.y - start[1]) > 5:
                on_select_y(dragframeindex, *sorted((start[0], event.ydata)))
        elif abs(event.x - start[1]) > 5:
            on_select(*sorted((start[0], event.xdata)))

    connectedfigure: object = None

    def connect_to_figure() -> None:
        nonlocal connectedfigure
        if canvas.figure is connectedfigure:
            return
        connectedfigure = canvas.figure
        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("motion_notify_event", on_motion)
        canvas.mpl_connect("button_release_event", on_release)
        if show_tag is not None:
            canvas.mpl_connect("figure_leave_event", lambda event: show_tag(event, ""))

    connect_to_figure()
    return connect_to_figure


def make_readout_tag(canvas: "FigureCanvasQTAgg") -> "Callable[[t.Any, str], None]":
    """Return a function that shows a readout in a small tag next to the pointer on the canvas.

    The function takes a matplotlib mouse event and the readout, and an empty readout hides the tag. Each part of the
    readout goes on a line of its own. The tag goes to the other side of the pointer at the edge of the canvas.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    tag = QtWidgets.QLabel(canvas)
    tag.setObjectName("readouttag")
    tag.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    tag.setStyleSheet(
        "QLabel#readouttag { background: palette(base); color: palette(text); border: 1px solid palette(mid);"
        " border-radius: 4px; padding: 2px 5px; }"
    )
    tag.hide()
    offset = 14

    def show_tag(event: t.Any, readout: str) -> None:
        if not readout or event.x is None:
            tag.hide()
            return
        tag.setText("\n".join(readout.split("   ")))
        tag.adjustSize()
        # matplotlib gives the pixels of the screen from the bottom, and Qt places a widget in points from the top
        ratio = canvas.device_pixel_ratio
        x, y = event.x / ratio, canvas.height() - event.y / ratio
        left = x + offset if x + offset + tag.width() <= canvas.width() else x - offset - tag.width()
        top = y + offset if y + offset + tag.height() <= canvas.height() else y - offset - tag.height()
        tag.move(max(0, round(left)), max(0, round(top)))
        tag.show()
        tag.raise_()

    return show_tag


def hide_readout_tag(window: "QtCore.QObject") -> None:
    """Hide the tag of make_readout_tag, which holds the values of the old plot until the mouse moves again."""
    from PySide6 import QtWidgets

    if (tag := window.findChild(QtWidgets.QLabel, "readouttag")) is not None:
        tag.hide()


def get_line_readouts(axis: "mplax.Axes", x: float) -> list[str]:
    """Return the value at x of each labelled line of the axes, as "label: value".

    A line with a label that starts with "_" is not a series of the legend. If no line has a label, the first line
    takes the label of the y axis. A subplot of one variable gives no label to its line.
    """
    lines = [line for line in axis.get_lines() if np.asarray(line.get_xdata()).size >= 2]
    labelledlines = [line for line in lines if not str(line.get_label()).startswith("_")]
    parts: list[str] = []
    for line in labelledlines or lines[:1]:
        label = plain_label(str(line.get_label()) if labelledlines else axis.get_ylabel())
        xdata, ydata = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        finite = np.isfinite(xdata) & np.isfinite(ydata)
        xdata, ydata = xdata[finite], ydata[finite]
        if xdata.size < 2:
            continue
        order = np.argsort(xdata)
        if xdata[order[0]] <= x <= xdata[order[-1]]:
            parts.append(f"{label}: {np.interp(x, xdata[order], ydata[order]):.4g}")
    return parts


# the readable text of the label or the checkbox of each flag. The tooltip then gives the flag
FLAG_LABELS: t.Final = MappingProxyType({
    "--colorbyion": "Colour by ion",
    "--frompackets": "ARTIS data source",
    "--hidenetspectrum": "Hide net spectrum",
    "--hideother": "Hide Other",
    "--hidexlabel": "Hide x label",
    "--legendframe": "Legend frame",
    "--logscalex": "Log x",
    "--markers": "Markers",
    "--nolegend": "Hide legend",
    "--normalised": "Normalise",
    "--nostack": "Unstacked",
    "--notitle": "Hide title",
    "--plotcmf": "Comoving frame",
    "--plotinvalidpart": "Partial times",
    "--showabsorption": "Show absorption",
    "--showbarnes": "Barnes et al. (2016)",
    "--showemission": "Show emission",
    "--use_pellet_decay_time": "Pellet decay time",
    "--use_thermalemissiontype": "Event",
    "--usedegrees": "Degrees",
    "-axis": "Axis",
    "-cell": "Cells",
    "-figscale": "Figure scale",
    "-groupby": "Group by",
    "-labelfontsize": "Label size",
    "-maxseriescount": "Max series",
    "-subplotsperrow": "Subplots per row",
    "-topnucs": "Top nuclides",
    "-x": "x variable",
    "-xbins": "x bins",
    "-xmax": "x max",
    "-xmin": "x min",
    "-xunit": "Unit",
    "-ymax": "y max",
    "-ymin": "y min",
    "-yscale": "y scale",
    "-yvariable": "Quantity",
})


def show_flag_labels(window: "QtWidgets.QWidget") -> None:
    """Give each label and each checkbox of a flag its readable text, and add the flag to its tooltip."""
    from PySide6 import QtWidgets

    widgets: list[QtWidgets.QLabel | QtWidgets.QCheckBox] = [
        *window.findChildren(QtWidgets.QLabel),
        *window.findChildren(QtWidgets.QCheckBox),
    ]
    for widget in widgets:
        flag = widget.text()
        if (text := FLAG_LABELS.get(flag, flag)) != flag:
            # a label names the control after it and ends with a colon, as in Keynote. A checkbox has no colon
            widget.setText(f"{text}:" if isinstance(widget, QtWidgets.QLabel) else text)
            tooltip = widget.toolTip()
            # a label whose text changes later, e.g. -xmin with a unit, already gives the flag in its tooltip
            if tooltip != flag and not tooltip.endswith(f"({flag})"):
                widget.setToolTip(f"{tooltip} ({flag})" if tooltip else flag)


def show_window(window: "QtWidgets.QMainWindow", figsize: tuple[float, float], on_screen: "Callable[[], None]") -> None:
    """Give the window the size of the last window of the viewer, or a first size, and show it.

    make_window and make_central_splitter give the window. on_screen runs after the window moves to a different
    screen.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    show_flag_labels(window)
    splitter = window.centralWidget()
    assert isinstance(splitter, QtWidgets.QSplitter)
    geometrykey, splitterkey = get_window_setting_keys(window)
    settings = get_settings()
    geometry = settings.value(geometrykey)
    splitterstate = settings.value(splitterkey)
    if isinstance(geometry, QtCore.QByteArray) and window.restoreGeometry(geometry):
        if isinstance(splitterstate, QtCore.QByteArray):
            splitter.restoreState(splitterstate)
    else:
        # the first size gives the plot 100 dpi, inside the screen
        screen = window.screen().availableGeometry()
        figwidth, figheight = figsize
        plotwidth = min(round(figwidth * 100) + 24, screen.width() - SIDEBAR_WIDTH - 80)
        plotheight = min(round(figheight * 100) + 24, screen.height() - 100)
        window.resize(plotwidth + SIDEBAR_WIDTH + 40, max(plotheight, 700))
        splitter.setSizes([plotwidth, SIDEBAR_WIDTH])
    window.show()
    if (windowhandle := window.windowHandle()) is not None:

        def on_screen_changed(_screen: QtGui.QScreen) -> None:
            # matplotlib handles the new pixel ratio first, thus the fit waits until Qt has no other events
            QtCore.QTimer.singleShot(0, window, on_screen)

        windowhandle.screenChanged.connect(on_screen_changed)


def make_timer(window: "QtWidgets.QWidget", milliseconds: int) -> "QtCore.QTimer":
    """Return a single-shot timer.

    The window is the parent of the timer, thus the timer stops when the window closes.
    """
    from PySide6 import QtCore

    timer = QtCore.QTimer(window)
    timer.setSingleShot(True)
    timer.setInterval(milliseconds)
    return timer
