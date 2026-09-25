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
import typing as t
from pathlib import Path

import numpy as np

from artistools.misc import exit_with_error
from artistools.misc import import_optional

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.figure as mplfig
    import numpy.typing as npt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
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

# each continuous slider of a window has this number of positions
SLIDER_STEPS: t.Final = 1000

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


def make_room_for_title(fig: "mplfig.Figure") -> None:
    """Make the figure taller if the text above the frames goes past its top edge.

    A saved file takes the tight bounding box, thus a title of two lines fits in it. A window shows the full
    figure, without the tight bounding box. The divider of make_frame_figure puts the frames at the bottom edge,
    thus the new height goes above them.
    """
    overflow = fig.get_tightbbox().y1 - fig.get_figheight()
    if overflow > 0.0:
        # a small gap keeps the top of the letters whole
        fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() + overflow + 0.05, forward=False)


def make_icon_pixmap(size: int, curve: "npt.NDArray[np.float64]") -> t.Any:
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
    os.execve(executable, [str(executable), *sys.orig_argv[1:]], environment)  # ruff:ignore[start-process-with-no-shell]


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
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
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


def make_sidebar() -> "tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]":
    """Return the sidebar of the window and the layout of its panel. The sections and the command scroll together."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    sidebar = QtWidgets.QWidget()
    sidebar.setFixedWidth(600)
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

    class PlotArea(QtWidgets.QWidget):
        """The area of the plot, which scales the figure to its size."""

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            on_resize()

    plotarea = PlotArea()
    plotarea.setMinimumSize(320, 240)
    plotlayout = QtWidgets.QVBoxLayout(plotarea)
    plotlayout.setContentsMargins(0, 0, 0, 0)
    plotlayout.addWidget(canvas, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
    return plotarea


def fit_canvas(
    canvas: "FigureCanvasQTAgg", fig: "mplfig.Figure", figsize: tuple[float, float], plotarea: "QtWidgets.QWidget"
) -> None:
    """Scale the figure to the plot area, and keep the shape of the frames.

    An embedded canvas has no figure manager, thus the figure cannot set the size of the widget. The resolution
    of the figure changes, thus the frames keep their size in inches and the whole figure fits in the area.
    """
    from PySide6 import QtCore

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

    helptexts = get_helptexts(parser)
    actionsbyflag = get_actions_by_flag(parser)
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
    commandtext = QtWidgets.QPlainTextEdit()
    commandtext.setReadOnly(True)
    commandtext.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    commandtext.setFixedHeight(3 * commandtext.fontMetrics().lineSpacing() + 12)
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip("Copy the command to the clipboard (⇧⌘C)")
    commandgrid.addWidget(commandtext, 0, 0)
    commandgrid.addWidget(copybutton, 0, 1, QtCore.Qt.AlignmentFlag.AlignTop)
    return commandtext, copybutton


def make_status_bar(
    window: "QtWidgets.QMainWindow",
) -> "tuple[QtWidgets.QLabel, QtWidgets.QLabel, QtWidgets.QLabel, QtWidgets.QToolButton]":
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
    return messagelabel, readoutlabel, drawtimelabel, helpbutton


def add_menus(window: "QtWidgets.QMainWindow", callbacks: "Mapping[str, Callable[[], object]]") -> None:
    """Add the File menu and the Help menu. callbacks gives the function of each item by the text of the item."""
    from PySide6 import QtGui

    menubar = window.menuBar()
    filemenu = menubar.addMenu("File")
    helpmenu = menubar.addMenu("Help")
    for menu, text, keys in (
        (filemenu, "Open Model...", QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open)),
        (filemenu, "Save Figure...", QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Save)),
        (filemenu, "Copy Command", QtGui.QKeySequence("Ctrl+Shift+C")),
        (filemenu, "Close Window", QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Close)),
        (helpmenu, "Keys and Mouse Actions", QtGui.QKeySequence("?")),
    ):
        action = menu.addAction(text)
        action.setShortcut(keys)
        action.triggered.connect(callbacks[text])


def copy_command(command: str) -> None:
    """Print the command, and put it on the clipboard."""
    from PySide6 import QtWidgets

    print(command)
    QtWidgets.QApplication.clipboard().setText(command)


def save_figure_of_command(
    window: "QtWidgets.QWidget",
    commandmain: "Callable[..., None]",
    plottokens: "Sequence[str]",
    command: str,
    defaultname: str,
) -> str | None:
    """Ask for a file name, and save the figure of the command there. Return the message for the status line.

    The figure comes from the command, thus the file is the same as the output of the command.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        window, "Save the figure", str(Path.cwd() / defaultname), "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)"
    )
    if not filename:
        return None

    QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            commandmain(argsraw=[*plottokens, "-o", filename])
    except (SystemExit, *USER_ERRORS) as exc:
        return f"The command did not save the figure: {get_first_line(str(exc))}"
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()
    print(f"{command} -o {shlex.quote(filename)}")
    return f"Saved {filename}"


def open_model_window(
    window: "QtWidgets.QWidget", open_window: "Callable[[Sequence[str], list[t.Any]], bool]", windows: "list[t.Any]"
) -> str | None:
    """Ask for the folder of a run, and open a new window for it. Return an error message if no window opened."""
    from PySide6 import QtWidgets

    folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Open the folder of an ARTIS run", str(Path.cwd()))
    if not folder:
        return None
    # a SystemExit in a Qt slot ends the process, thus an error of the new window stays in this window
    errors = io.StringIO()
    try:
        with contextlib.redirect_stderr(errors):
            opened = open_window([folder], windows)
    except SystemExit:
        opened = False
    finally:
        sys.stderr.write(errors.getvalue())
    return None if opened else f"The viewer cannot open {folder}: {get_first_line(errors.getvalue())}"


def show_window(
    window: "QtWidgets.QMainWindow", figsize: tuple[float, float], sidebarwidth: int, on_screen: "Callable[[], None]"
) -> None:
    """Give the window its first size and show it. on_screen runs after the window moves to a different screen."""
    from PySide6 import QtCore
    from PySide6 import QtGui

    # the first size gives the plot 100 dpi, inside the screen
    screen = window.screen().availableGeometry()
    figwidth, figheight = figsize
    plotwidth = min(round(figwidth * 100) + 24, screen.width() - sidebarwidth - 80)
    plotheight = min(round(figheight * 100) + 24, screen.height() - 100)
    window.resize(plotwidth + sidebarwidth + 40, max(plotheight, 700))
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
