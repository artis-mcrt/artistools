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
from functools import cache
from functools import partial
from pathlib import Path
from types import MappingProxyType

import numpy as np

from artistools.misc import addarg_quiet
from artistools.misc import exit_with_error
from artistools.misc import import_optional
from artistools.misc import print_error
from artistools.misc import separate_trailing_folders
from artistools.misc import write_gif
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
    import numpy.typing as npt
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
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

# the resolution of Copy Figure, in dots per inch
COPY_FIGURE_DPI: t.Final = 300

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


def get_option_row_tokens(rows: OptionRows) -> list[str]:
    """Return the command tokens of the rows of the option table."""
    return [
        token
        for flag, optionvalues in rows
        for token in (get_option_tokens(flag, *optionvalues) if len(optionvalues) == 1 else (flag, *optionvalues))
    ]


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

    class KeyOwnerFilter(QtCore.QObject):
        """Give a key to the widget with the focus when that widget uses the key, and not to a window shortcut.

        These widgets use the keys, but the shortcuts of the window took the keys from them:

        - a spin box;
        - a combo box;
        - a slider;
        - a list;
        - a popup, e.g. the list of names of a completer;
        - a button, which uses only the space key.

        For example, the Up key in -maxseriescount made the time range wider.
        """

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            if event.type() == QtCore.QEvent.Type.ShortcutOverride and isinstance(event, QtGui.QKeyEvent):
                focuswidget = QtWidgets.QApplication.focusWidget()
                # the field of a completer keeps the focus while its popup shows, and the popup takes the keys
                usesarrows = QtWidgets.QApplication.activePopupWidget() is not None or isinstance(
                    focuswidget,
                    QtWidgets.QAbstractSpinBox
                    | QtWidgets.QComboBox
                    | QtWidgets.QAbstractSlider
                    | QtWidgets.QAbstractItemView,
                )
                usesspace = usesarrows or isinstance(focuswidget, QtWidgets.QAbstractButton)
                key = event.key()
                if (usesarrows and key in arrowkeys) or (usesspace and key == QtCore.Qt.Key.Key_Space):
                    event.accept()
                    return True
            return super().eventFilter(watched, event)

    keyownerfilter = KeyOwnerFilter(app)
    app.installEventFilter(keyownerfilter)
    apply_appearance()

    class InspectorSizer(QtCore.QObject):
        """Give each control of the sidebar the small size of macOS, as the inspector of Keynote has.

        The cards of the subplots come and go, thus the filter gives the size to each widget when Qt polishes it.
        """

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            if event.type() == QtCore.QEvent.Type.Polish and isinstance(watched, QtWidgets.QWidget):
                parent = watched.parentWidget()
                while parent is not None and parent.objectName() != "inspector":
                    parent = parent.parentWidget()
                if parent is not None:
                    watched.setAttribute(QtCore.Qt.WidgetAttribute.WA_MacSmallSize)
            return super().eventFilter(watched, event)

    if sys.platform == "darwin":
        app.installEventFilter(InspectorSizer(app))

    class EditTracker(QtCore.QObject):
        """Keep the text field that the user confirmed last, and the time, in two properties of its window.

        A plot that rejects the change of the field then marks the field, as a form of macOS does.
        """

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            isreturn = isinstance(event, QtGui.QKeyEvent) and event.key() in {
                QtCore.Qt.Key.Key_Return,
                QtCore.Qt.Key.Key_Enter,
            }
            confirms = (event.type() == QtCore.QEvent.Type.KeyPress and isreturn) or (
                event.type() == QtCore.QEvent.Type.FocusOut
                and isinstance(watched, QtWidgets.QLineEdit)
                and watched.isModified()
            )
            if isinstance(watched, QtWidgets.QLineEdit) and confirms:
                window = watched.window()
                window.setProperty("lasteditedfield", watched)
                window.setProperty("lastedittime", time.monotonic())
            return super().eventFilter(watched, event)

    app.installEventFilter(EditTracker(app))

    class QuitTracker(QtCore.QObject):
        """Record the time of a quit before Qt closes the windows, e.g. after Quit in the menu of macOS.

        A window that closes during a quit keeps its command for the next start, and a window that the user closes
        does not.
        """

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            if event.type() == QtCore.QEvent.Type.Quit and (app := QtWidgets.QApplication.instance()) is not None:
                app.setProperty("quittime", time.monotonic())
            return super().eventFilter(watched, event)

    app.installEventFilter(QuitTracker(app))
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

    field = window.property("lasteditedfield")
    return field if isinstance(field, QtWidgets.QLineEdit) else None


def mark_field_error(window: "QtCore.QObject", field: "QtWidgets.QLineEdit", message: str) -> None:
    """Give the field a red border and the message beside it, as a form of macOS does."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    clear_field_error(window)
    field.setStyleSheet("QLineEdit { border: 2px solid firebrick; border-radius: 3px; }")
    QtWidgets.QToolTip.showText(field.mapToGlobal(QtCore.QPoint(0, field.height())), message, field)
    window.setProperty("errorfield", field)


def clear_field_error(window: "QtCore.QObject") -> None:
    """Remove the mark of mark_field_error, e.g. after a plot that the command accepts."""
    field = window.property("errorfield")
    # each plot calls this function, thus a window with no mark returns before the imports
    if field is None:
        return
    import shiboken6
    from PySide6 import QtWidgets

    # a new card of a subplot replaces its fields, and Qt then deletes the old field
    if isinstance(field, QtWidgets.QLineEdit) and shiboken6.isValid(field):
        field.setStyleSheet("")
    window.setProperty("errorfield", None)


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
            # the state of the window holds whether the toolbar shows
            settings.setValue(f"{self.objectName()}/state", self.saveState())
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
    filter ignores them.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # the instance holds open_folder, and the class holds no reference to it, as for PlotArea
    class FileOpenFilter(QtCore.QObject):
        """Receive the QFileOpenEvent of the application."""

        def __init__(self, parent: QtCore.QObject, open_folder: "Callable[[str], None]") -> None:
            super().__init__(parent)
            self.open_folder = open_folder
            self.launchpaths = {str(Path(argument).absolute()) for argument in sys.orig_argv}

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            if event.type() == QtCore.QEvent.Type.FileOpen and isinstance(event, QtGui.QFileOpenEvent):
                path = event.file()
                if str(Path(path).absolute()) in self.launchpaths:
                    return True
                activewindow = QtWidgets.QApplication.activeWindow()
                handler = activewindow.property("drophandler") if activewindow is not None else None
                if Path(path).is_dir() or not callable(handler):
                    self.open_folder(path)
                else:
                    handler([path])
                return True
            return super().eventFilter(watched, event)

    app.installEventFilter(FileOpenFilter(app, open_folder))


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
    path = str(Path(folder).absolute())
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
    because each window fits it to its own size. An error of a window goes to the terminal.
    """
    shown = [
        remove_figwidthscale(get_absolute_tokens(window.property("sessiontokens")()))
        for window in windows
        if callable(window.property("sessiontokens"))
    ]
    firstwindows = list(windows)
    for tokens in take_session_windows():
        if remove_figwidthscale(tokens) in shown:
            continue
        message = run_command_step(lambda tokens=tokens: open_window(tokens, windows), quiet=False)
        if message is not None:
            print_error(f"The viewer cannot open the window of the last session: {message}")
    for window in firstwindows:
        activate_window(window)


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
    # a small space between the rows and the sections keeps more of the controls in view
    grid.setContentsMargins(8, 2, 0, 6)
    grid.setVerticalSpacing(4)
    grid.setColumnStretch(1, 1)
    settingkey = f"{QtWidgets.QApplication.applicationDisplayName()}/sections/{key or title}"
    isopen = get_bool_setting(settingkey, default=title not in CLOSED_SECTIONS)

    def set_open(checked: bool) -> None:
        header.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
        header.setToolTip(f"{'Close' if checked else 'Open'} the section")
        content.setVisible(checked)
        get_settings().setValue(settingkey, checked)

    header.toggled.connect(set_open)
    header.setChecked(isopen)
    set_open(isopen)
    # the search field of the sidebar finds each section by its heading
    header.setProperty("sectioncontent", content)
    panellayout.addWidget(header)
    panellayout.addWidget(content)
    return header, grid


def set_section_shown(header: "QtWidgets.QToolButton", grid: "QtWidgets.QGridLayout", *, shown: bool) -> None:
    """Show or hide a section of add_section. A closed section that shows keeps its controls hidden."""
    # the search field of the sidebar shows only the sections that the window shows
    header.setProperty("sectionshown", shown)
    header.setVisible(shown)
    if (content := grid.parentWidget()) is not None:
        content.setVisible(shown and header.isChecked())


def make_row_layout(widgets: "Sequence[QtWidgets.QWidget]") -> "QtWidgets.QHBoxLayout":
    """Return a layout that puts the widgets side by side from the left."""
    from PySide6 import QtWidgets

    rowlayout = QtWidgets.QHBoxLayout()
    for widget in widgets:
        rowlayout.addWidget(widget)
    rowlayout.addStretch(1)
    return rowlayout


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
            """Put each item in its place inside rect if move is True, and return the height that the rows take."""
            x, y, rowheight = rect.x(), rect.y(), 0
            for item in self.layoutitems:
                hint = item.sizeHint()
                if x + hint.width() > rect.right() + 1 and rowheight > 0:
                    x, y, rowheight = rect.x(), y + rowheight + self.spacing(), 0
                if move:
                    item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
                x += hint.width() + self.spacing()
                rowheight = max(rowheight, hint.height())
            return y + rowheight - rect.y()

    return FlowLayout


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
            # a window reads this property, e.g. to draw a fast preview during a drag
            self.setProperty("dragging", self.draghandle is not None)
            self.move_handle(pixel)

        @t.override
        def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
            if self.draghandle is not None:
                self.move_handle(event.position().x())

        @t.override
        def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
            self.draghandle = None
            self.setProperty("dragging", self.draghandle is not None)

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
    # the controls in the sidebar take the small size of macOS, which start_application gives by this name
    sidebar.setObjectName("inspector")
    # the handle of the splitter sets the width, and a narrower sidebar cuts the controls
    sidebar.setMinimumWidth(360)
    sidebarlayout = QtWidgets.QVBoxLayout(sidebar)
    sidebarlayout.setContentsMargins(0, 0, 0, 0)
    panel = QtWidgets.QWidget()
    panellayout = QtWidgets.QVBoxLayout(panel)
    panellayout.setSpacing(2)
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
            self.on_resize()

    plotarea = PlotArea(on_resize)
    plotarea.setMinimumSize(320, 240)
    plotlayout = QtWidgets.QVBoxLayout(plotarea)
    plotlayout.setContentsMargins(0, 0, 0, 0)
    plotlayout.addWidget(canvas, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
    return plotarea


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
) -> "tuple[QtWidgets.QTableWidget, Callable[[OptionRows], None], Callable[[str], None]]":
    """Return a table of the other options, and the functions that show rows and add an option.

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

        removebutton = QtWidgets.QToolButton()
        removebutton.setText("✕")
        removebutton.setAutoRaise(True)
        removebutton.setToolTip(f"Remove {flag} from the command")
        removebutton.setAccessibleName(f"Remove {flag}")
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

    def add_option(flag: str) -> None:
        """Add a row of the option at the end of the table, e.g. from the search field of the sidebar."""
        set_option(len(optionrows), flag)

    show_option_rows()
    return optiontable, set_rows, add_option


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
    """Add the command at the bottom of the panel, and return its text box and its Copy button."""
    panellayout.addStretch(1)
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


def get_icon(
    symbol: str, themeicon: "QtGui.QIcon.ThemeIcon | None", fallback: "QtWidgets.QStyle.StandardPixmap | None"
) -> "QtGui.QIcon":
    """Return the SF Symbol of the name on macOS, else the icon of the theme, else the icon of the Qt style.

    Qt gives an SF Symbol for its name on macOS. The result can be an empty icon, and a tool button then shows its
    text.
    """
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    icon = QtGui.QIcon.fromTheme(symbol) if sys.platform == "darwin" else QtGui.QIcon()
    if icon.isNull() and themeicon is not None:
        icon = QtGui.QIcon.fromTheme(themeicon)
    if icon.isNull() and fallback is not None:
        icon = QtWidgets.QApplication.style().standardIcon(fallback)
    return icon


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


class ViewerToolbar(t.NamedTuple):
    """The toolbar of a window: the steps of the time, Play, the frame rate, the export actions, and the search."""

    previous: "QtGui.QAction"
    play: "QtGui.QAction"
    next: "QtGui.QAction"
    fpsbox: "QtWidgets.QDoubleSpinBox"
    save: "QtGui.QAction"
    exportanimation: "QtGui.QAction"
    copyfigure: "QtGui.QAction"
    copycommand: "QtGui.QAction"
    search: "QtWidgets.QLineEdit"


def make_toolbar(window: "QtWidgets.QMainWindow", playtooltip: str) -> ViewerToolbar:
    """Add the toolbar at the top of the window, as the apps of macOS have, and return its actions.

    The toolbar holds the actions that a user takes often, thus they stay in view while the sidebar scrolls. The
    Play action is checkable, and its icon shows Pause while Play runs. The search field and the button of the
    sidebar are at the trailing edge, as in Finder and Mail.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    themeicon, pixmap = QtGui.QIcon.ThemeIcon, QtWidgets.QStyle.StandardPixmap
    toolbar = window.addToolBar("Toolbar")
    # the state of the window names the toolbar, thus the next window keeps the choice of Hide Toolbar
    toolbar.setObjectName("toolbar")
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
    window.setUnifiedTitleAndToolBarOnMac(True)

    def add_action(text: str, icon: "QtGui.QIcon", tooltip: str) -> "QtGui.QAction":
        action = toolbar.addAction(icon, text)
        action.setToolTip(tooltip)
        return action

    previous = add_action(
        "Previous",
        get_icon("backward.end.fill", themeicon.MediaSkipBackward, pixmap.SP_MediaSkipBackward),
        "Move the time to the previous timestep (Left key)",
    )
    playicon = get_icon("play.fill", themeicon.MediaPlaybackStart, pixmap.SP_MediaPlay)
    pauseicon = get_icon("pause.fill", themeicon.MediaPlaybackPause, pixmap.SP_MediaPause)
    play = add_action("Play", playicon, playtooltip)
    play.setCheckable(True)

    def show_play_state(checked: bool) -> None:
        play.setIcon(pauseicon if checked else playicon)

    play.toggled.connect(show_play_state)
    nextaction = add_action(
        "Next",
        get_icon("forward.end.fill", themeicon.MediaSkipForward, pixmap.SP_MediaSkipForward),
        "Move the time to the next timestep (Right key)",
    )
    fpsbox = make_fps_box()
    toolbar.addWidget(QtWidgets.QLabel(" FPS: "))
    toolbar.addWidget(fpsbox)
    toolbar.addSeparator()
    save = add_action(
        "Save Figure…",
        get_icon("square.and.arrow.down", themeicon.DocumentSave, pixmap.SP_DialogSaveButton),
        "Save the figure of the command in a file",
    )
    exportanimation = add_action(
        "Export Animation…", get_icon("film", None, None), "Save a GIF file of the steps of Play"
    )
    copyfigure = add_action("Copy Figure", get_icon("doc.on.doc", themeicon.EditCopy, None), "Copy the figure")
    copycommand = add_action("Copy Command", get_icon("terminal", None, None), "Copy the command of the plot")

    spacer = QtWidgets.QWidget()
    spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
    toolbar.addWidget(spacer)
    search = QtWidgets.QLineEdit()
    search.setObjectName("sidebarsearch")
    search.setPlaceholderText("Search")
    search.setClearButtonEnabled(True)
    search.setFixedWidth(220)
    search.addAction(
        get_icon("magnifyingglass", themeicon.SystemSearch, None), QtWidgets.QLineEdit.ActionPosition.LeadingPosition
    )
    toolbar.addWidget(search)
    add_action(
        "Hide Sidebar", get_icon("sidebar.right", None, None), "Hide or show the sidebar of the controls"
    ).triggered.connect(lambda: toggle_sidebar(window))
    return ViewerToolbar(
        previous=previous,
        play=play,
        next=nextaction,
        fpsbox=fpsbox,
        save=save,
        exportanimation=exportanimation,
        copyfigure=copyfigure,
        copycommand=copycommand,
        search=search,
    )


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
        ("Edit", "Find", QtGui.QKeySequence(standardkey.Find)),
        # macOS moves this item to the menu of the application
        ("Edit", "Settings…", QtGui.QKeySequence("Ctrl+,")),
        ("View", "Play", QtGui.QKeySequence("Space")),
        ("View", "Hide Sidebar", QtGui.QKeySequence("Ctrl+Meta+S")),
        ("View", "Hide Toolbar", QtGui.QKeySequence("Ctrl+Alt+T")),
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
    "Save Figure…": "Run the command to save the figure",
    "Export Animation…": "Save a GIF file of the steps of Play",
    "Undo": "Undo the last change",
    "Redo": "Redo the change that Undo removed",
    "Copy Figure": "Copy the figure as an image",
    "Copy Command": "Copy the command",
    "Copy Python": "Copy the Python code of the plot",
    "Play": "Play or pause",
    "Find": "Search the controls and the options of the sidebar",
    "Hide Sidebar": "Hide or show the sidebar",
})


def get_keyboard_help(keyrows: "Sequence[tuple[str, str]]", menuitems: "Collection[str]") -> str:
    """Return the table of the keys and the mouse actions of a viewer, with the shortcuts of the platform.

    keyrows gives the keys or the mouse action, and its help text, of each row of the viewer as HTML. menuitems gives
    the texts of the menu items of the viewer, as add_menus receives them, and the table gives the shortcut of each.
    """
    shortcuts = get_menu_shortcut_texts()
    rows = [
        *keyrows,
        # every window has the sidebar, and add_menus gives its item
        *(
            (f"<b>{shortcuts[text]}</b>", helptext)
            for text, helptext in MENU_HELPTEXTS.items()
            if text in menuitems or text == "Hide Sidebar"
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
    enabled: "Mapping[str, Callable[[], bool]]",
    titles: "Mapping[str, Callable[[], str]]",
    open_folder: "Callable[[str], object]",
) -> None:
    """Add the menus File, Edit, View, Window, and Help.

    callbacks gives the function of each item by the text of get_menu_items. A viewer omits an item that it does not
    support, e.g. Reload Data. This function gives the items of the sidebar, the toolbar, the full screen, and the
    window. enabled tells whether an item can run, e.g. Undo, and titles gives the text of an item that changes, e.g.
    Pause for Play. File > Open Recent gives a recent model to open_folder.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    windowcallbacks: dict[str, Callable[[], object]] = {
        "Hide Sidebar": lambda: toggle_sidebar(window),
        "Hide Toolbar": lambda: toggle_toolbar(window),
        "Enter Full Screen": lambda: window.showNormal() if window.isFullScreen() else window.showFullScreen(),
        "Minimize": window.showMinimized,
        "Zoom": lambda: window.showNormal() if window.isMaximized() else window.showMaximized(),
        "Settings…": lambda: show_settings_window(window),
        "Find": lambda: focus_sidebar_search(window),
        "artistools Help": lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(HELP_URL)),
        "About": lambda: show_about(window),
    }
    windowtitles: dict[str, Callable[[], str]] = {
        "Hide Sidebar": lambda: "Show Sidebar" if is_sidebar_hidden(window) else "Hide Sidebar",
        "Hide Toolbar": lambda: (
            "Hide Toolbar" if any(bar.isVisible() for bar in get_toolbars(window)) else "Show Toolbar"
        ),
        "Enter Full Screen": lambda: "Exit Full Screen" if window.isFullScreen() else "Enter Full Screen",
    }
    allcallbacks = {**windowcallbacks, **callbacks}
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


def add_recent_menu(filemenu: "QtWidgets.QMenu", open_folder: "Callable[[str], object]") -> None:
    """Add the submenu Open Recent, which shows the recent models when it opens, and Clear Menu at its end."""
    recentmenu = filemenu.addMenu("Open Recent")

    def show_recent_models() -> None:
        recentmenu.clear()
        for folder in get_recent_models():
            action = recentmenu.addAction(Path(folder).name)
            action.setToolTip(folder)
            action.setEnabled(Path(folder).is_dir())
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
    window.setWindowFilePath(str(folder.absolute()))


def show_settings_window(parent: "QtWidgets.QWidget") -> None:
    """Show the Settings window of the viewers, or bring it to the front if it is open.

    A change applies at once and the settings keep it, as in the Settings windows of macOS. Each viewer window
    receives a change of the colours of the plot through its property "settingshandler".
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    for widget in QtWidgets.QApplication.topLevelWidgets():
        if widget.objectName() == "settings" and widget.isVisible():
            activate_window(widget)
            return
    settings = get_settings()
    dialog = QtWidgets.QDialog(parent)
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

    previewcheck = QtWidgets.QCheckBox("Draw a fast preview while a slider moves (plotspectra)")
    previewcheck.setToolTip("The preview reads the first batch of ranks only, and the full plot follows the release")
    previewcheck.setChecked(get_bool_setting("dragpreview", default=True))
    previewcheck.toggled.connect(partial(settings.setValue, "dragpreview"))
    form.addRow(previewcheck)

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


def focus_sidebar_search(window: "QtWidgets.QMainWindow") -> None:
    """Give the keyboard to the search field of the sidebar, and show the sidebar if it is hidden."""
    from PySide6 import QtWidgets

    if is_sidebar_hidden(window):
        toggle_sidebar(window)
    if (searchedit := window.findChild(QtWidgets.QLineEdit, "sidebarsearch")) is not None:
        searchedit.setFocus()
        searchedit.selectAll()


# the search field of the sidebar shows this number of options of the command at most
MAX_SEARCH_OPTIONS: t.Final = 8


def add_sidebar_search(
    searchedit: "QtWidgets.QLineEdit",
    sidebar: "QtWidgets.QWidget",
    panellayout: "QtWidgets.QVBoxLayout",
    parser: argparse.ArgumentParser,
    hiddendests: "Collection[str]",
    add_option: "Callable[[str], None]",
) -> None:
    """Let the search field of the toolbar filter the sections of the sidebar, as the Settings app of macOS does.

    The text shows only the sections with a control that holds it, e.g. in a label or a tooltip, and opens them. Each
    option of the command that holds the text shows as a button below the field. A click on the button adds the
    option to the table of the other options.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    searchedit.setToolTip(
        f"Show the controls and the options of the command that hold the text ({get_menu_shortcut_texts()['Find']})"
    )
    # the options of the command that match the text show above the sections
    resultsbox = QtWidgets.QWidget()
    resultslayout = make_flow_layout()
    resultsbox.setLayout(resultslayout)
    resultsbox.setContentsMargins(8, 6, 8, 2)
    resultsbox.hide()
    if (sidebarlayout := sidebar.layout()) is not None and isinstance(sidebarlayout, QtWidgets.QVBoxLayout):
        sidebarlayout.insertWidget(0, resultsbox)
    tableactions = get_table_actions(parser, hiddendests)
    helptexts = get_helptexts(parser)

    def get_sections() -> "list[tuple[QtWidgets.QToolButton, QtWidgets.QWidget]]":
        sections: list[tuple[QtWidgets.QToolButton, QtWidgets.QWidget]] = []
        for index in range(panellayout.count()):
            item = panellayout.itemAt(index)
            header = item.widget() if item is not None else None
            if isinstance(header, QtWidgets.QToolButton) and isinstance(
                content := header.property("sectioncontent"), QtWidgets.QWidget
            ):
                sections.append((header, content))
        return sections

    def get_texts(content: "QtWidgets.QWidget") -> list[str]:
        texts: list[str] = []
        for child in [content, *content.findChildren(QtWidgets.QWidget)]:
            texts.append(child.toolTip())
            if isinstance(child, QtWidgets.QLabel | QtWidgets.QAbstractButton):
                texts.append(child.text())
            elif isinstance(child, QtWidgets.QLineEdit):
                texts.append(child.placeholderText())
            elif isinstance(child, QtWidgets.QComboBox):
                texts.extend(child.itemText(index) for index in range(child.count()))
        return texts

    def on_add_option(flag: str) -> None:
        searchedit.clear()
        add_option(flag)
        for header, _ in get_sections():
            if header.text() == "Other options":
                header.setChecked(True)

    def on_search(text: str) -> None:
        query = text.strip().lower()
        # the results show in the sidebar, thus a search shows a hidden sidebar
        window = sidebar.window()
        if query and isinstance(window, QtWidgets.QMainWindow) and is_sidebar_hidden(window):
            toggle_sidebar(window)
        for header, content in get_sections():
            shown = header.property("sectionshown") is not False
            if not query:
                header.setVisible(shown)
                content.setVisible(shown and header.isChecked())
                continue
            matches = shown and (
                query in header.text().lower() or any(query in text.lower() for text in get_texts(content))
            )
            header.setVisible(matches)
            content.setVisible(matches)
        while (item := resultslayout.takeAt(0)) is not None:
            if (widget := item.widget()) is not None:
                # Qt deletes the button after the event, and the button must not show until then
                widget.hide()
                widget.deleteLater()
        matchingactions = [
            action
            for action in tableactions
            if query
            and (
                any(query in flag.lower() for flag in action.option_strings)
                or query in helptexts.get(action.dest, "").lower()
            )
        ][:MAX_SEARCH_OPTIONS]
        for action in matchingactions:
            flag = action.option_strings[0]
            button = QtWidgets.QToolButton()
            button.setText(f"+ {flag}")
            button.setToolTip(f"Add {flag} to the other options. {helptexts.get(action.dest, '')}")
            button.clicked.connect(partial(on_add_option, flag))
            resultslayout.addWidget(button)
        resultsbox.setVisible(bool(matchingactions))

    searchedit.textChanged.connect(on_search)
    QtGui.QShortcut(
        QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), searchedit, context=QtCore.Qt.ShortcutContext.WidgetShortcut
    ).activated.connect(searchedit.clear)


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


def get_toolbars(window: "QtWidgets.QMainWindow") -> "list[QtWidgets.QToolBar]":
    """Return the toolbars of the window."""
    from PySide6 import QtWidgets

    return window.findChildren(QtWidgets.QToolBar)


def toggle_toolbar(window: "QtWidgets.QMainWindow") -> None:
    """Hide the toolbars of the window, or show them."""
    toolbars = get_toolbars(window)
    shown = any(toolbar.isVisible() for toolbar in toolbars)
    for toolbar in toolbars:
        toolbar.setVisible(not shown)


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

    The frames, the ticks, and the text take the colours of the window. A black line or a black text takes the
    colour of the text, else it cannot show on the dark background. The other colours stay, e.g. the colours of the
    series and of an image. The command saves a figure with its usual colours, because only the window calls this.
    """
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    def is_dark(colour: t.Any) -> bool:
        red, green, blue, alpha = mcolors.to_rgba(colour)
        return alpha > 0.0 and 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.25

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
        if isinstance(text, Text) and is_dark(text.get_color()):
            text.set_color(foreground)
    for line in fig.findobj(Line2D):
        if isinstance(line, Line2D) and is_dark(line.get_color()):
            line.set_color(foreground)


def copy_figure(fig: "mplfig.Figure") -> None:
    """Put the figure on the clipboard as a PNG image, with the resolution of a printed page."""
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=COPY_FIGURE_DPI)
    QtWidgets.QApplication.clipboard().setImage(QtGui.QImage.fromData(buffer.getvalue()))


def copy_text(text: str) -> None:
    """Print the text, e.g. the command, and put it on the clipboard."""
    from PySide6 import QtWidgets

    print(text)
    QtWidgets.QApplication.clipboard().setText(text)


def split_dpi_row(rows: OptionRows, defaultdpi: int) -> tuple[OptionRows, int]:
    """Return the rows without -dpi, and the resolution that -dpi gives, or defaultdpi for rows with no -dpi."""
    dpivalues = next((values for flag, values in rows if flag == "-dpi"), None)
    dpi = int(dpivalues[0]) if dpivalues and dpivalues[0].isdecimal() else defaultdpi
    return tuple(row for row in rows if row[0] != "-dpi"), dpi


# the file types of Save Figure, with the tooltip of each
EXPORT_FORMATS: t.Final = (
    ("pdf", "A vector file, for a paper. A colour image in it takes the resolution"),
    ("png", "An image with the resolution, e.g. for a slide"),
    ("svg", "A vector file, e.g. for a web page. A colour image in it takes the resolution"),
)


def ask_export_options(window: "QtWidgets.QWidget", dpi: int) -> tuple[str, int] | None:
    """Ask for the file type and the resolution before the save panel, as the Export dialog of Keynote does.

    Return the suffix of the type and the resolution in dots per inch, or None if the user cancels. The settings keep
    the type for the next export.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    dialog = QtWidgets.QDialog(window)
    dialog.setWindowTitle("Save Figure")
    form = QtWidgets.QFormLayout(dialog)
    segments = make_segmented_control(
        [suffix.upper() for suffix, _ in EXPORT_FORMATS], [tooltip for _, tooltip in EXPORT_FORMATS]
    )
    suffixes = [suffix for suffix, _ in EXPORT_FORMATS]
    lastsuffix = str(get_settings().value("exportformat", "pdf"))
    segments.setCurrentIndex(suffixes.index(lastsuffix) if lastsuffix in suffixes else 0)
    form.addRow("Format:", segments)
    dpibox = QtWidgets.QSpinBox()
    dpibox.setRange(10, 2400)
    dpibox.setSingleStep(50)
    dpibox.setValue(dpi)
    dpibox.setSuffix(" dpi")
    dpibox.setToolTip("The resolution of a PNG file, and of a colour image in a PDF or an SVG file (-dpi)")
    form.addRow("Resolution:", dpibox)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    nextbutton = buttons.addButton("Next…", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
    nextbutton.setDefault(True)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    dialog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    suffix = suffixes[segments.currentIndex()]
    get_settings().setValue("exportformat", suffix)
    return suffix, dpibox.value()


def export_animation(
    window: "QtWidgets.QWidget",
    queue: "DrawQueue[t.Any]",
    statusbar: StatusBar,
    commandmain: "Callable[..., None]",
    commandname: str,
    frametokens: "Sequence[Sequence[str]]",
    fps: float,
) -> None:
    """Save a GIF file of the steps of Play, with one run of the command for each frame.

    Each frame comes from the command, as for Save Figure. The worker thread runs the command for each frame and
    joins the frames, thus the window stays responsive. The GIF shows each frame for 1/fps seconds.
    """
    import tempfile

    from PySide6 import QtWidgets

    if len(frametokens) > MAX_ANIMATION_FRAMES:
        answer = QtWidgets.QMessageBox.question(
            window,
            "Export Animation",
            f"The animation has {len(frametokens)} frames, and each frame runs the command. Continue?",
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
            for index, tokens in enumerate(frametokens):
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
            show_status_note(statusbar, f"Saved {filename}, with {len(frametokens)} frames")

    if not queue.run_task(
        lambda: run_command_step(export), f"Export of {len(frametokens)} frames in progress...", show_result
    ):
        show_status_message(statusbar, "A different task is in progress. Export the animation after it", "")


def save_figure_of_command(
    window: "QtWidgets.QWidget",
    statusbar: StatusBar,
    commandmain: "Callable[..., None]",
    commandname: str,
    plottokens: "Sequence[str]",
    dpi: int,
    defaultdpi: int,
) -> None:
    """Save the figure of the command in a file that the user selects.

    The figure comes from the command, thus the file is the same as the output of the command. The command reads a
    name with no suffix as a folder, thus the name takes the suffix of the selected type. The status bar shows the
    result.

    plottokens holds no -dpi. dpi is the resolution of the command, and the dialog for a PNG file proposes it. A PDF or
    an SVG file takes dpi for its raster parts, e.g. a colour image. defaultdpi is the default of the command.
    """
    from PySide6 import QtWidgets

    options = ask_export_options(window, dpi)
    if options is None:
        return
    suffix, dpi = options
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
        show_status_note(statusbar, f"Saved {filename}")


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
    """A viewer with the values of its controls, which draws the plot of new values or keeps the old values."""

    values: ValuesT
    # the last warning of the last plot, which the status bar shows. A user of the application sees no terminal
    warning: str

    def change(self, values: ValuesT) -> str | None:
        """Draw the plot of the values, or keep the old values and return the reason for the status line."""
        ...


class DrawQueue[ValuesT]:
    """The plots of a window: a change shows its values at once, and the plot follows when Qt has no other events.

    A drag gives a new value for each movement of the mouse, and a plot can take seconds. The queue draws only the
    last values that the user gave. viewer.values holds the last values that the user gave, and drawnvalues holds the
    values of the plot. If the command rejects new values, the viewer keeps the values of the plot.

    With render, a worker thread draws each plot. The window then shows each new value of a drag at once.

    The queue also keeps the values before each change of the user, thus Undo and Redo can return to them.
    """

    def __init__(
        self,
        window: "QtCore.QObject",
        viewer: PlotViewer[ValuesT],
        statusbar: StatusBar,
        show_values: "Callable[[], None]",
        after_draw: "Callable[[str | None], None]",
        change: "Callable[[ValuesT], str | None] | None" = None,
        get_drawkind: "Callable[[], str] | None" = None,
        render: "Callable[[ValuesT], Callable[[], str | None]] | None" = None,
        keep_on_undo: "Callable[[ValuesT, ValuesT], ValuesT] | None" = None,
    ) -> None:
        """Make an empty queue. after_draw receives the message of each plot of the queue.

        change draws the values of the queue in place of viewer.change, e.g. a preview. get_drawkind gives the name of
        the last plot for the status bar, e.g. "Preview". render draws the plot of the values in a worker thread. It
        returns the function that shows that plot in the window and gives the message of a rejection.

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
        self.change = change or viewer.change
        self.get_drawkind = get_drawkind
        self.requestedvalues: ValuesT | None = None
        self.drawnvalues: ValuesT = viewer.values
        self.render = render
        self.keep_on_undo = keep_on_undo
        # the values before each change of the user, for Undo, and the values that Undo replaced, for Redo
        self.undovalues: list[ValuesT] = []
        self.redovalues: list[ValuesT] = []
        self.lastchangetime = -math.inf
        # the text field that gave the last change, which a rejection marks
        self.editedfield: QtWidgets.QLineEdit | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.rendertimer: QtCore.QTimer | None = None
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
        if render is not None:
            # one worker thread draws one plot at a time, and a drag during a plot waits for the end of that plot
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plot")
            # the window thread checks the worker at each tick, because a Qt call from the worker thread is not safe
            self.rendertimer = QtCore.QTimer(window)
            self.rendertimer.setInterval(10)
            self.rendertimer.timeout.connect(self.show_rendered)

    def apply(self, values: ValuesT, *, undoable: bool = True) -> None:
        """Show the new values now, and draw them when Qt has no other events. Only a change of the values draws a plot.

        A change that the window makes, e.g. a step of Play or a fit to the size of the window, is not undoable.
        """
        if values == self.viewer.values:
            # a handler that clamps a control to the old values must still move the control back
            self.show_values()
            return
        if undoable:
            self.record_undo()
        # the plot of the change comes later, and a rejection then marks the field that gave the change
        self.editedfield = get_edited_field(self.window)
        # each handler makes its values from viewer.values, thus a second change before the plot keeps the first
        self.viewer.values = values
        self.redraw()

    def record_undo(self) -> None:
        """Keep the current values for Undo, before a change of the user."""
        now = time.monotonic()
        # a drag of a slider or a repeated key gives many changes, and one step of Undo reverts all of them
        if not self.undovalues or now - self.lastchangetime > UNDO_MERGE_SECONDS:
            self.undovalues = [*self.undovalues[-UNDO_LIMIT + 1 :], self.viewer.values]
        self.lastchangetime = now
        self.redovalues.clear()

    def can_undo(self) -> bool:
        """Return whether Undo has values that differ from the current values."""
        return any(values != self.viewer.values for values in self.undovalues)

    def can_redo(self) -> bool:
        """Return whether Redo has values that differ from the current values."""
        return any(values != self.viewer.values for values in self.redovalues)

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
        while source and source[-1] == self.viewer.values:
            source.pop()
        if not source:
            return
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
        """Draw the plot of the last values that the user gave."""
        if self.executor is not None:
            if self.rendering is None and self.task is None:
                self.start_render()
            return
        values, self.requestedvalues = self.requestedvalues, None
        if values is not None:
            self.after_draw(self.draw(values, self.change))

    def start_render(self) -> None:
        """Start a plot of the last values that the user gave in the worker thread."""
        values, self.requestedvalues = self.requestedvalues, None
        if values is None or self.executor is None or self.render is None or self.rendertimer is None:
            return
        self.renderedvalues = values
        self.renderstart = time.perf_counter()
        self.statusbar.drawtime.setText("Plot in progress...")
        self.rendering = self.executor.submit(self.render, values)
        self.rendertimer.start()

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
        if self.rendertimer is not None:
            self.rendertimer.stop()
        # a task waits for the plot in progress, and the newer values of the user wait for the task
        if self.task is not None and self.taskfuture is None:
            self.start_task()
        elif self.task is None and self.requestedvalues is not None:
            self.start_render()

    def show_rendered_plot(self, rendering: "Future[Callable[[], str | None]]") -> None:
        """Show the complete plot of the worker thread, or the message of a rejection."""
        self.rendering = None
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
        self.show_plot_status(message, self.renderstart)
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
        if self.executor is None:
            on_done(task())
            return True
        self.task, self.taskstatus, self.on_task_done = task, statustext, on_done
        if self.rendering is None:
            self.start_task()
        return True

    def start_task(self) -> None:
        """Start the task of run_task in the worker thread."""
        if self.task is None or self.executor is None or self.rendertimer is None:
            return
        self.statusbar.drawtime.setText(self.taskstatus)
        self.taskfuture = self.executor.submit(self.task)
        self.rendertimer.start()

    def end_task(self, taskfuture: "Future[str | None]") -> None:
        """Give the message of the complete task to the function of run_task."""
        on_done = self.on_task_done
        self.task, self.taskfuture, self.on_task_done = None, None, None
        self.statusbar.drawtime.setText("")
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
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)

    def draw(self, values: ValuesT, change: "Callable[[ValuesT], str | None]") -> str | None:
        """Draw the plot of the values with change, and return the message of a rejection."""
        from PySide6 import QtCore
        from PySide6 import QtWidgets

        # a plot can take seconds, thus the cursor and the status bar show the wait
        with show_wait_cursor():
            self.statusbar.drawtime.setText("Plot in progress...")
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            starttime = time.perf_counter()
            # change() keeps the values of the last plot when the command rejects the new values
            self.viewer.values = self.drawnvalues
            try:
                message = change(values)
            finally:
                self.drawnvalues = self.viewer.values
        self.show_plot_status(message, starttime)
        return message

    def show_plot_status(self, message: str | None, starttime: float) -> None:
        """Show the time of the plot that started at starttime, its message or its warning, and the values."""
        drawkind = self.get_drawkind() if self.get_drawkind is not None else "Plot"
        self.plotseconds = time.perf_counter() - starttime
        self.statusbar.drawtime.setText(f"{drawkind} time: {self.plotseconds:.2f} s")
        # the readout holds the values of the old plot until the mouse moves again
        self.statusbar.readout.setText("")
        show_status_message(self.statusbar, message, self.viewer.warning)
        if message is None:
            clear_field_error(self.window)
        elif self.editedfield is not None:
            mark_field_error(self.window, self.editedfield, message)
        self.editedfield = None
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
) -> "Callable[[], None]":
    """Give the plot a readout under the pointer, a drag across a frame that selects an x range, and a double-click.

    on_select receives the two x values of a drag, and on_reset receives a double-click on a frame. on_select_y
    receives the index of the frame and the two y values of a drag with the Shift key inside one frame. on_menu
    receives the index of the frame and the matplotlib event of a click with the right button. matplotlib keeps the
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
        readoutlabel.setText(
            get_readout(event, event.inaxes) if frameindex is not None and event.xdata is not None else ""
        )
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

    connect_to_figure()
    return connect_to_figure


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


def show_window(window: "QtWidgets.QMainWindow", figsize: tuple[float, float], on_screen: "Callable[[], None]") -> None:
    """Give the window the size of the last window of the viewer, or a first size, and show it.

    make_window and make_central_splitter give the window. on_screen runs after the window moves to a different
    screen.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    splitter = window.centralWidget()
    assert isinstance(splitter, QtWidgets.QSplitter)
    geometrykey, splitterkey = get_window_setting_keys(window)
    settings = get_settings()
    geometry = settings.value(geometrykey)
    splitterstate = settings.value(splitterkey)
    if isinstance(windowstate := settings.value(f"{window.objectName()}/state"), QtCore.QByteArray):
        window.restoreState(windowstate)
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
