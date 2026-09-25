"""Functions that each window of the --interactive option shares, e.g. plotspectra and plotestimators.

A viewer makes the command of the plot from its controls, then parses the command and draws the plot. The
functions here make and change the command tokens, and they build the parts of the Qt window that do not
depend on the command. Each function that uses Qt imports PySide6 when it runs, because the CLI must start
quickly and PySide6 is an optional dependency.
"""

import argparse
import contextlib
import io
import math
import re
import shlex
import sys
import threading
import time
import traceback
import typing as t
from pathlib import Path
from types import MappingProxyType

import numpy as np

from artistools.misc import addarg_quiet
from artistools.misc import exit_with_error
from artistools.misc import import_optional
from artistools.misc import print_error
from artistools.plottools import plain_label

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Generator
    from collections.abc import Mapping
    from collections.abc import Sequence
    from concurrent.futures import Future

    import matplotlib.axes as mplax
    import numpy.typing as npt
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

# the Play button waits for this time after each plot, thus the user can see each step
PLAY_MILLISECONDS: t.Final = 150


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


def get_last_warning(errors: str) -> str:
    """Return the last warning of the standard error of a step, without its prefix, or an empty text.

    rich colours the text under FORCE_COLOR or TTY_COMPATIBLE also when it writes into a capture, thus the colour
    codes go first.
    """
    plainerrors = re.sub(r"\x1b\[[0-9;]*m", "", errors)
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
    returns the line that starts with "error: ". If no line has that start, it returns the first line.
    """
    lines = [line.strip() for line in errortext.splitlines() if line.strip()]
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


def get_macos_bundle_executable(applicationname: str) -> Path:
    """Return the Python executable in the application bundle of the viewer.

    Make the bundle if it does not exist. The bundle holds a hard link to the Python executable, thus it uses almost
    no disk space. On a different volume, it holds a copy. If the Python executable changes, this function replaces
    the link.
    """
    import os
    import plistlib
    import shutil

    baseexecutable = Path(sys.executable).resolve()
    contents = Path.home() / "Library" / "Caches" / "artistools" / f"{applicationname}.app" / "Contents"
    executable = contents / "MacOS" / baseexecutable.name
    if executable.exists() and executable.samefile(baseexecutable):
        return executable

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
    }
    tmpinfo = contents / f"Info.plist.{os.getpid()}.tmp"
    tmpinfo.write_bytes(plistlib.dumps(info))
    tmpinfo.replace(contents / "Info.plist")
    return executable


def relaunch_in_macos_bundle(applicationname: str) -> None:
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
        executable = get_macos_bundle_executable(applicationname)
    except OSError:
        # the viewer can open without the bundle, and the Dock then gives the name of the executable
        return

    sys.stdout.flush()
    sys.stderr.flush()
    environment = os.environ | {MACOS_BUNDLE_VARIABLE: "1", "__PYVENV_LAUNCHER__": sys.executable}
    argv = [str(executable), *sys.orig_argv[1:]]
    os.execve(executable, argv, environment)  # ruff:ignore[start-process-with-no-shell]


def start_application(applicationname: str, iconcurve: "npt.NDArray[np.float64]") -> "QtWidgets.QApplication":
    """Return the Qt application of a viewer, with the name and the icon of the viewer.

    The window is a Qt window with native controls. The Qt canvas of matplotlib draws at the pixel ratio
    of the screen, thus the plot has the full resolution of a Retina display.
    """
    import os

    if sys.platform == "darwin":
        relaunch_in_macos_bundle(applicationname)

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
        - a button, which uses only the space key.

        For example, the Up key in -maxseriescount made the time range wider.
        """

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            if event.type() == QtCore.QEvent.Type.ShortcutOverride and isinstance(event, QtGui.QKeyEvent):
                focuswidget = QtWidgets.QApplication.focusWidget()
                usesarrows = isinstance(
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
    return app


def get_settings() -> "QtCore.QSettings":
    """Return the settings of the viewers, which keep the geometry of each window and the last model folder.

    The settings take the organisation and the name of the application, thus a test can send them to a folder with
    QSettings.setPath.
    """
    from PySide6 import QtCore

    return QtCore.QSettings()


def make_window(applicationname: str) -> "QtWidgets.QMainWindow":
    """Return the window of a viewer, which keeps its geometry and the sizes of its splitter when it closes.

    show_window restores them for the next window of the same viewer.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class ViewerWindow(QtWidgets.QMainWindow):
        """A window that writes its geometry and the state of its splitter to the settings when it closes."""

        def __init__(self, applicationname: str) -> None:
            super().__init__()
            # the name of the object gives the keys of the settings of the window
            self.setObjectName(applicationname)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        @t.override
        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            geometrykey, splitterkey = get_window_setting_keys(self)
            settings = get_settings()
            settings.setValue(geometrykey, self.saveGeometry())
            splitter = self.centralWidget()
            if isinstance(splitter, QtWidgets.QSplitter):
                settings.setValue(splitterkey, splitter.saveState())
            super().closeEvent(event)

    return ViewerWindow(applicationname)


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


def add_section(panellayout: "QtWidgets.QVBoxLayout", title: str) -> "tuple[QtWidgets.QLabel, QtWidgets.QGridLayout]":
    """Add a section with a heading and a grid for its controls to the panel of the window."""
    from PySide6 import QtWidgets

    header = QtWidgets.QLabel(title)
    font = header.font()
    font.setBold(True)
    header.setFont(font)
    content = QtWidgets.QWidget()
    grid = QtWidgets.QGridLayout(content)
    # a small space between the rows and the sections keeps more of the controls in view
    grid.setContentsMargins(8, 2, 0, 6)
    grid.setVerticalSpacing(4)
    grid.setColumnStretch(1, 1)
    panellayout.addWidget(header)
    panellayout.addWidget(content)
    return header, grid


def add_row(grid: "QtWidgets.QGridLayout", row: int, widgets: "Sequence[QtWidgets.QWidget]") -> None:
    """Put the widgets side by side in one row of the grid, from the left."""
    from PySide6 import QtWidgets

    rowlayout = QtWidgets.QHBoxLayout()
    for widget in widgets:
        rowlayout.addWidget(widget)
    rowlayout.addStretch(1)
    grid.addLayout(rowlayout, row, 0, 1, -1)


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
            completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
            completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
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


def add_command_section(
    panellayout: "QtWidgets.QVBoxLayout",
) -> "tuple[QtWidgets.QPlainTextEdit, QtWidgets.QPushButton]":
    """Add the command at the bottom of the panel, and return its text box and its Copy button."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    panellayout.addStretch(1)
    _, commandgrid = add_section(panellayout, "Command")
    # the command text takes the width, and the Copy button keeps its size at the right
    commandgrid.setColumnStretch(0, 1)
    commandgrid.setColumnStretch(1, 0)

    # the instance of the box holds no reference to the window. PySide keeps each class, thus the class must hold none
    class CommandBox(QtWidgets.QPlainTextEdit):
        """The box of the command, which fits its height to the wrapped lines when its width changes."""

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            if event.size().width() != event.oldSize().width():
                fit_command_box(self)

    commandtext = CommandBox()
    commandtext.setReadOnly(True)
    commandtext.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    fit_command_box(commandtext)
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip(f"Copy the command to the clipboard ({get_menu_shortcut_texts()['Copy Command']})")
    commandgrid.addWidget(commandtext, 0, 0)
    commandgrid.addWidget(copybutton, 0, 1, QtCore.Qt.AlignmentFlag.AlignTop)
    return commandtext, copybutton


# the command box shows between these numbers of lines, and a longer command scrolls inside the box
MIN_COMMAND_LINES: t.Final = 3
MAX_COMMAND_LINES: t.Final = 8


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
    document = commandtext.document()
    layout = document.documentLayout()
    textlines = math.ceil(layout.documentSize().height())
    textheight = sum(
        layout.blockBoundingRect(document.findBlockByNumber(i)).height() for i in range(document.blockCount())
    )
    shownlines = min(max(textlines, MIN_COMMAND_LINES), MAX_COMMAND_LINES)
    boxheight = textheight + (shownlines - textlines) * commandtext.fontMetrics().lineSpacing()
    commandtext.setFixedHeight(math.ceil(boxheight + 2 * document.documentMargin() + 2 * commandtext.frameWidth()))


def start_play_timer(playtimer: "QtCore.QTimer", plotseconds: float) -> None:
    """Start the pause before the next step of Play.

    The pause lets the user see each step. The old plot stays in view while the worker draws the next one, thus a
    plot that took longer than the pause needs no pause.
    """
    playtimer.start(max(0, PLAY_MILLISECONDS - round(plotseconds * 1000.0)))


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
    statusbar.addWidget(messagelabel, stretch=1)
    for widget in (readoutlabel, drawtimelabel, helpbutton):
        statusbar.addPermanentWidget(widget)
    return StatusBar(message=messagelabel, readout=readoutlabel, drawtime=drawtimelabel, helpbutton=helpbutton)


def get_menu_items() -> "list[tuple[str, str, QtGui.QKeySequence]]":
    """Return the menu, the text, and the shortcut of each menu item of a viewer."""
    from PySide6 import QtGui

    return [
        ("File", "Open Model...", QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open)),
        ("File", "Reload Data", QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Refresh)),
        ("File", "Save Figure...", QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save)),
        ("File", "Copy Command", QtGui.QKeySequence("Ctrl+Shift+C")),
        ("File", "Close Window", QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close)),
        ("Help", "Keys and Mouse Actions", QtGui.QKeySequence("?")),
    ]


# the help text of each menu item in the table of the keys
MENU_HELPTEXTS: t.Final = MappingProxyType({
    "Save Figure...": "Run the command to save the figure",
    "Copy Command": "Copy the command",
    "Open Model...": "Open a model in a new window",
    "Reload Data": "Read the run again, e.g. while ARTIS writes more timesteps",
})


def get_keyboard_help(keyrows: "Sequence[tuple[str, str]]", menuitems: "Collection[str]") -> str:
    """Return the table of the keys and the mouse actions of a viewer, with the shortcuts of the platform.

    keyrows gives the keys or the mouse action, and its help text, of each row of the viewer as HTML. menuitems gives
    the texts of the menu items of the viewer, as add_menus receives them, and the table gives the shortcut of each.
    """
    shortcuts = get_menu_shortcut_texts()
    rows = [
        *keyrows,
        *((f"<b>{shortcuts[text]}</b>", helptext) for text, helptext in MENU_HELPTEXTS.items() if text in menuitems),
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


def add_menus(window: "QtWidgets.QMainWindow", callbacks: "Mapping[str, Callable[[], object]]") -> None:
    """Add the File menu and the Help menu.

    callbacks gives the function of each item by the text of the item. A viewer omits an item that it does not
    support, e.g. Reload Data.
    """
    menubar = window.menuBar()
    menus = {name: menubar.addMenu(name) for name in ("File", "Help")}
    for menuname, text, keys in get_menu_items():
        if text not in callbacks:
            continue
        action = menus[menuname].addAction(text)
        action.setShortcut(keys)
        action.triggered.connect(callbacks[text])


def copy_command(command: str) -> None:
    """Print the command, and put it on the clipboard."""
    from PySide6 import QtWidgets

    print(command)
    QtWidgets.QApplication.clipboard().setText(command)


def save_figure_of_command(
    window: "QtWidgets.QWidget", commandmain: "Callable[..., None]", commandname: str, plottokens: "Sequence[str]"
) -> str | None:
    """Ask for a file name, and save the figure of the command there. Return the message for the status line.

    The figure comes from the command, thus the file is the same as the output of the command. The command reads a
    name with no suffix as a folder, thus the name takes the suffix of the selected type.
    """
    from PySide6 import QtWidgets

    filename, selectedfilter = QtWidgets.QFileDialog.getSaveFileName(
        window, "Save the figure", str(Path.cwd() / f"{commandname}.pdf"), "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)"
    )
    if not filename:
        return None
    if not Path(filename).suffix:
        # a filter such as "PNG (*.png)" names the suffix
        suffixmatch = re.search(r"\*(\.\w+)", selectedfilter)
        filename += suffixmatch.group(1) if suffixmatch else ".pdf"

    def save() -> str | None:
        commandmain(argsraw=[*plottokens, "-o", filename])
        return None

    with show_wait_cursor():
        message = run_command_step(save)
    if message is not None:
        return f"The command did not save the figure: {message}"
    if not Path(filename).is_file():
        return f"The command wrote no file at {filename}. The terminal shows its output"
    print(shlex.join(["artistools", commandname, *plottokens, "-o", filename]))
    return f"Saved {filename}"


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

    message = run_command_step(lambda: open_window([folder], windows), quiet=False)
    if message is not None:
        return f"The viewer cannot open {folder}: {message}"
    settings.setValue("modelfolder", str(Path(folder).parent))
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
    ) -> None:
        """Make an empty queue. after_draw receives the message of each plot of the queue.

        change draws the values of the queue in place of viewer.change, e.g. a preview. get_drawkind gives the name of
        the last plot for the status bar, e.g. "Preview". render draws the plot of the values in a worker thread. It
        returns the function that shows that plot in the window and gives the message of a rejection.
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
        self.executor: ThreadPoolExecutor | None = None
        self.rendertimer: QtCore.QTimer | None = None
        self.renderedvalues: ValuesT = viewer.values
        self.rendering: Future[Callable[[], str | None]] | None = None
        self.renderstart = 0.0
        # the time of the last plot, which sets the pause of Play
        self.plotseconds = 0.0
        if render is not None:
            # one worker thread draws one plot at a time, and a drag during a plot waits for the end of that plot
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plot")
            # the window thread checks the worker at each tick, because a Qt call from the worker thread is not safe
            self.rendertimer = QtCore.QTimer(window)
            self.rendertimer.setInterval(10)
            self.rendertimer.timeout.connect(self.show_rendered)

    def apply(self, values: ValuesT) -> None:
        """Show the new values now, and draw them when Qt has no other events. Values of no change draw no plot."""
        if values == self.viewer.values:
            # a handler that clamps a control to the old values must still move the control back
            self.show_values()
            return
        # each handler makes its values from viewer.values, thus a second change before the plot keeps the first
        self.viewer.values = values
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
            if self.rendering is None:
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
        """Show the plot of the worker thread when it is complete, then start a plot of newer values."""
        if self.rendering is None or not self.rendering.done():
            return
        if self.rendertimer is not None:
            self.rendertimer.stop()
        rendering, self.rendering = self.rendering, None
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
        if self.requestedvalues is not None:
            self.start_render()

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
        self.show_values()


def set_edit_text(edit: "QtWidgets.QLineEdit", text: str) -> None:
    """Show the text in a field, unless the user types in that field.

    A plot or a Play step can end while the user types. Without this check, the text of the values replaces the
    text that the user typed. The handler of a field calls setModified(False), thus a field shows new values again
    after the user presses Return.
    """
    if not (edit.hasFocus() and edit.isModified()):
        edit.setText(text)


def connect_plot_mouse(
    canvas: "FigureCanvasQTAgg",
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
