"""Show the plot of plotspectra in a window with controls for the time, the x range, and the emission plot."""

import argparse
import contextlib
import dataclasses as dc
import io
import math
import re
import shlex
import sys
import typing as t
from pathlib import Path

import matplotlib.figure as mplfig
import numpy as np

from artistools.commands import SuggestingArgumentParser
from artistools.misc import addarg_quiet
from artistools.misc import exit_with_error
from artistools.misc import get_time_range
from artistools.misc import get_timestep_times
from artistools.misc import import_optional
from artistools.misc import parse_cli_args
from artistools.misc import print_error
from artistools.spectra.core import get_xunit
from artistools.spectra.plotspectra import addargs
from artistools.spectra.plotspectra import draw_plot
from artistools.spectra.plotspectra import get_artis_run_folders
from artistools.spectra.plotspectra import get_default_xlimits
from artistools.spectra.plotspectra import make_plot_figure
from artistools.spectra.plotspectra import resolve_plot_args

if t.TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.axes as mplax
    import numpy.typing as npt

# the controls of the window give these arguments, thus the command drops the values that the user typed
CONTROLLED_DESTS: t.Final = frozenset({
    "timestep",
    "timedays",
    "timemin",
    "timemax",
    "xmin",
    "xmax",
    "showemission",
    "showabsorption",
    "emissionabsorption",
    "groupby",
    "maxseriescount",
    "deltax",
    "interactive",
})

REJECTED_MESSAGE: t.Final = "plotspectra cannot draw this plot. The terminal shows the error"

# each slider of the window has this number of positions
SLIDER_STEPS: t.Final = 1000


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class ControlValues:
    """The values of the controls of the viewer, which give the options of the plotspectra command.

    The x limits and the bin width keep the text of the command, and an empty deltax gives no option.
    """

    centre: float
    width: float
    xmin: str
    xmax: str
    showemission: bool
    showabsorption: bool
    groupby: str | None
    maxseriescount: int
    deltax: str


def get_command_tokens(
    argsraw: "Sequence[str] | None", kwargs: "Mapping[str, t.Any]", *, fromdispatcher: bool
) -> list[str]:
    """Return the arguments that the user gave to plotspectra, without the name of the command.

    The dispatcher gives the parsed arguments alone, thus the tokens then come from sys.argv. A script for one
    command, e.g. plotartisspectrum, takes no word for the subcommand.
    """
    if kwargs:
        exit_with_error(
            "--interactive shows the command of the plot, and a call with keyword arguments has no command",
            "Give the arguments as a list, e.g. main(argsraw=['mymodel', '--interactive'])",
        )

    if argsraw is not None:
        return list(argsraw)

    if not fromdispatcher:
        return sys.argv[1:]

    from artistools.commands import get_subcommand_of_script

    return sys.argv[1:] if get_subcommand_of_script(Path(sys.argv[0]).stem) else sys.argv[2:]


def make_parser() -> SuggestingArgumentParser:
    """Return the parser of plotspectra."""
    parser = SuggestingArgumentParser()
    addargs(parser)
    addarg_quiet(parser)
    return parser


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


def remove_options(parser: SuggestingArgumentParser, tokens: "Sequence[str]", dests: "Collection[str]") -> list[str]:
    """Return the tokens without the options of these dests and without the values of those options."""

    def isflag(argstring: str) -> bool:
        return argstring.startswith("-") and not re.match(r"-\.?\d", argstring)

    argstrings = parser.split_joined_flags(tokens)
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
        while maxvalues and index < len(argstrings) and not isflag(argstrings[index]):
            index += 1
            maxvalues -= 1

    return kept


def format_days(value: float) -> str:
    """Return a time in days in fixed-point notation. The option -timedays reads the "-" of 1e-05 as a range."""
    return np.format_float_positional(value, precision=6, trim="-")


def get_timedays_argument(runfolders: "Sequence[Path]", centre: float, width: float) -> str:
    """Return the -timedays value for a range of this width around the centre.

    A width of zero gives the centre alone, and plotspectra then reads the timestep that holds it. A range
    that holds the middle of no timestep of a run stops plotspectra, thus such a range also gives the centre
    alone. The range stays inside the timesteps of the first run.
    """
    if width > 0.0:
        tstart = get_timestep_times(runfolders[0], loc="start")[0]
        tend = get_timestep_times(runfolders[0], loc="end")[-1]
        lowtext = format_days(max(centre - width / 2.0, tstart))
        hightext = format_days(min(centre + width / 2.0, tend))
        low, high = float(lowtext), float(hightext)
        if all(any(low <= tmid <= high for tmid in get_timestep_times(folder, loc="mid")) for folder in runfolders):
            return f"{lowtext}-{hightext}"

    return format_days(centre)


def make_command_tokens(basetokens: "Sequence[str]", options: "Sequence[str]") -> list[str]:
    """Return the plotspectra arguments with the options of the controls after the paths at the start."""
    pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
    return [*basetokens[:pathcount], *options, *basetokens[pathcount:]]


def check_viewer_args(args: argparse.Namespace) -> None:
    """Stop when args selects an action that is not one plot of spectra. The window shows one plot only."""
    otheractions = {
        "-timedayslist": args.multispecplot,
        "--makevspecpol": args.makevspecpol,
        "--averagevspecpolfiles": args.averagevspecpolfiles,
        "--output_spectra": args.output_spectra,
        f"-stokesparam {args.stokesparam}": "/" in args.stokesparam,
    }
    if given := [name for name, isgiven in otheractions.items() if isgiven]:
        exit_with_error(
            f"--interactive shows one plot of spectra. A different action comes from: {', '.join(given)}",
            f"Remove {', '.join(given)}, or remove --interactive",
        )


class SpectrumViewer:
    """The plot of the viewer and the values of its controls.

    The window reads and changes the values, and a test can do the same without a display. Each call to draw
    makes the command from the values, then parses it and draws it with the code of plotspectra. Thus the plot
    always agrees with the command.
    """

    def __init__(self, tokens: "Sequence[str]", fig: mplfig.Figure) -> None:
        """Read the arguments of the user, and take the first values of the controls from them."""
        parser = make_parser()
        self.basetokens = remove_options(parser, tokens, CONTROLLED_DESTS)
        args = parse_cli_args(addargs, None, None, remove_options(parser, tokens, {"interactive"}))
        # resolve_frompackets gives an emission plot a default -groupby, thus the value comes from the arguments
        givengroupby: str | None = args.groupby
        resolve_plot_args(args)
        check_viewer_args(args)
        self.args = args

        self.runfolders = get_artis_run_folders(args.modelspecpaths)
        if not self.runfolders:
            exit_with_error(
                "--interactive takes the time range from the timesteps of an ARTIS run, and no path names a run",
                "Give the folder of an ARTIS run, e.g. plotspectra mymodel --interactive",
            )
        self.tmids = get_timestep_times(self.runfolders[0], loc="mid")
        self.tstart = get_timestep_times(self.runfolders[0], loc="start")[0]
        self.tend = get_timestep_times(self.runfolders[0], loc="end")[-1]

        # a range of one timestep is a single time, and a plot with no time starts in the middle of the run
        if args.timemin is not None and args.timemax is not None:
            centre = (args.timemin + args.timemax) / 2.0
            coversseveral = sum(args.timemin <= tmid <= args.timemax for tmid in self.tmids) > 1
            width = args.timemax - args.timemin if coversseveral else 0.0
        else:
            centre, width = self.tmids[len(self.tmids) // 2], 0.0

        groupbyaction = next(
            action
            for action in parser._actions  # ruff:ignore[private-member-access]
            if action.dest == "groupby"
        )
        self.groupbychoices: list[str] = [str(choice) for choice in groupbyaction.choices or ()]
        self.defaultmaxseriescount: int = parser.get_default("maxseriescount")
        self.defaultxlimits = get_default_xlimits(args.xunit, gamma=args.gamma)
        # 4 significant digits of the time and 3 of the width give a short command
        self.values = ControlValues(
            centre=float(f"{centre:.4g}"),
            width=float(f"{width:.3g}"),
            xmin=format(args.xmin, ".10g"),
            xmax=format(args.xmax, ".10g"),
            showemission=bool(args.showemission),
            showabsorption=bool(args.showabsorption),
            groupby=givengroupby,
            maxseriescount=args.maxseriescount,
            deltax="" if args.deltax is None else format(args.deltax, ".10g"),
        )

        self.fig = fig
        self.axes: npt.NDArray[t.Any] = np.empty(0, dtype=object)
        self.residualaxis: mplax.Axes | None = None
        self.framesforabsorption: bool | None = None
        # a window can change the size of the figure, thus the size of the frames stays here
        self.figsize: tuple[float, float] = (0.0, 0.0)

    def get_plot_tokens(self) -> list[str]:
        """Return the plotspectra arguments of the values."""
        values = self.values
        options = [
            *("-t", get_timedays_argument(self.runfolders, values.centre, values.width)),
            *("-xmin", values.xmin, "-xmax", values.xmax),
        ]
        if values.showemission:
            options.append("--showemission")
        if values.showabsorption:
            options.append("--showabsorption")
        if values.groupby is not None:
            options += ["-groupby", values.groupby]
        # the count applies to an emission plot alone, thus a different plot leaves it out
        if (values.showemission or values.showabsorption) and values.maxseriescount != self.defaultmaxseriescount:
            options += ["-maxseriescount", str(values.maxseriescount)]
        if values.deltax:
            options += ["-deltax", values.deltax]
        return make_command_tokens(self.basetokens, options)

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotspectra", *self.get_plot_tokens()])

    def get_timesteps_text(self) -> str:
        """Return the timesteps that the time range reads from spec.out, which holds whole timesteps."""
        timestepmin, timestepmax, *_ = get_time_range(
            self.runfolders[0],
            timedays_range_str=get_timedays_argument(self.runfolders, self.values.centre, self.values.width),
            clamp_to_timesteps=not self.args.notimeclamp,
        )
        if timestepmin == timestepmax:
            return f"timestep {timestepmin}"
        return f"timesteps {timestepmin} to {timestepmax}"

    def draw(self, *, quiet: bool = True) -> str | None:
        """Draw the plot of the command, and return a message for the status line if plotspectra rejects it.

        Each plot prints the same lines again, e.g. the list of the ions of an emission plot. Thus a quiet plot
        discards the standard output. The errors go to the standard error.
        """
        output = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        try:
            with output:
                return self.draw_command()
        except SystemExit:
            return REJECTED_MESSAGE
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc))
            return REJECTED_MESSAGE

    def draw_command(self) -> str | None:
        """Parse the command and draw its plot, or return a message if the plot differs from the values."""
        plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens())
        resolve_plot_args(plotargs)
        shown = (plotargs.showemission, plotargs.showabsorption)
        if shown != (self.values.showemission, self.values.showabsorption):
            return "A different option of the command keeps the emission plot on"
        self.draw_frames(plotargs)
        return None

    def draw_frames(self, plotargs: argparse.Namespace) -> None:
        """Draw the plot on empty frames.

        --showabsorption draws a taller frame, thus a change of it makes new frames.
        """
        if plotargs.showabsorption != self.framesforabsorption:
            self.fig.clear()
            _, self.axes, self.residualaxis = make_plot_figure(plotargs, fig=self.fig)
            self.framesforabsorption = plotargs.showabsorption
            figwidth, figheight = self.fig.get_size_inches()
            self.figsize = (float(figwidth), float(figheight))
        else:
            for axis in (*self.axes, self.residualaxis):
                if axis is not None:
                    axis.cla()

        draw_plot(plotargs, self.axes, self.residualaxis)
        self.fig.canvas.draw_idle()

    def change(self, values: ControlValues) -> str | None:
        """Draw the plot of the new values, and keep the old values if plotspectra rejects the new command."""
        oldvalues, self.values = self.values, values
        message = self.draw()
        if message is not None:
            self.values = oldvalues
            # the old values drew a plot before, thus they draw again
            self.draw()
        return message


def run_viewer(tokens: "Sequence[str]") -> None:
    """Open the window of the viewer, and print the command of the last plot when the window closes.

    The window is a Qt window with native controls. The Qt canvas of matplotlib draws at the pixel ratio
    of the screen, thus the plot has the full resolution of a Retina display.
    """
    import os

    import_optional("PySide6.QtWidgets")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # Qt stops the process with no Python error when it cannot find a display
    if sys.platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        exit_with_error(
            "--interactive needs a window, and this computer has no display",
            "Run the command on a computer with a display, e.g. with ssh -X",
        )

    viewer = SpectrumViewer(tokens, mplfig.Figure())
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    window.setWindowTitle("artistools plotspectra --interactive")
    canvas = FigureCanvasQTAgg(viewer.fig)
    if viewer.draw(quiet=False) is not None:
        # the arguments of the user give the error, and the terminal shows it
        raise SystemExit(1)

    # the frames keep their size in inches, thus a screen that is too small shows scroll bars
    scrollarea = QtWidgets.QScrollArea()
    scrollarea.setWidget(canvas)
    scrollarea.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    # the controls go beside the plot, thus the tall frame of --showabsorption keeps the full height of the screen
    layout = QtWidgets.QHBoxLayout(window)
    layout.addWidget(scrollarea, stretch=1)
    panel = QtWidgets.QWidget()
    panel.setFixedWidth(560)
    layout.addWidget(panel, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
    grid = QtWidgets.QGridLayout(panel)

    xunit = get_xunit(viewer.args.xunit)
    xname = f"{xunit.kind.capitalize()} [{xunit.label}]"
    # the x sliders act on log10(x), thus their range must be above zero
    xlow = min(value for value in (float(viewer.values.xmin), viewer.defaultxlimits[0]) if value > 0.0) / 2.0
    xhigh = max(float(viewer.values.xmax), viewer.defaultxlimits[1]) * 2.0
    widthmax = max((viewer.tend - viewer.tstart) / 4.0, viewer.values.width)

    def make_slider() -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        # the Left key and the Right key move the time, thus a slider must not take them
        slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        return slider

    # each slider maps its position from 0 to SLIDER_STEPS onto the range of its value
    def to_position(value: float, low: float, high: float) -> int:
        return round(SLIDER_STEPS * (min(max(value, low), high) - low) / (high - low))

    def from_position(position: int, low: float, high: float) -> float:
        return low + (high - low) * position / SLIDER_STEPS

    logtrange = (math.log10(viewer.tstart), math.log10(viewer.tend))
    logxrange = (math.log10(xlow), math.log10(xhigh))
    timeslider, widthslider, xminslider, xmaxslider = (make_slider() for _ in range(4))
    timetext, widthtext = QtWidgets.QLabel(), QtWidgets.QLabel()
    xminedit, xmaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    for widget in (timetext, widthtext, xminedit, xmaxedit):
        widget.setFixedWidth(160)
    rows = [
        ("Time [d]", timeslider, timetext),
        ("\u0394t [d]", widthslider, widthtext),
        (f"{xname} min", xminslider, xminedit),
        (f"{xname} max", xmaxslider, xmaxedit),
    ]
    for row, (label, slider, valuewidget) in enumerate(rows):
        grid.addWidget(QtWidgets.QLabel(label), row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(valuewidget, row, 2)

    checkboxes = QtWidgets.QHBoxLayout()
    grid.addLayout(checkboxes, 4, 0, 1, 3)
    emissioncheck = QtWidgets.QCheckBox("--showemission")
    absorptioncheck = QtWidgets.QCheckBox("--showabsorption")
    groupbybox = QtWidgets.QComboBox()
    groupbybox.addItems(["none", *viewer.groupbychoices])
    countlabel = QtWidgets.QLabel("-maxseriescount")
    countbox = QtWidgets.QSpinBox()
    countbox.setRange(1, 200)
    # an empty check box gives no -deltax, and plotspectra then takes its default bins. A step of approximately
    # 1/1000 of the x range is correct for each x unit, e.g. 10 Å for the default range of wavelengths
    deltaxcheck = QtWidgets.QCheckBox(f"-deltax [{xunit.label}]")
    deltaxbox = QtWidgets.QDoubleSpinBox()
    xspan = abs(float(viewer.values.xmax) - float(viewer.values.xmin))
    deltaxstep = 10.0 ** math.floor(math.log10(xspan / 1000.0))
    deltaxbox.setDecimals(max(0, -math.floor(math.log10(deltaxstep))))
    deltaxbox.setRange(deltaxstep, xspan)
    deltaxbox.setSingleStep(deltaxstep)
    deltaxbox.setValue(float(viewer.values.deltax) if viewer.values.deltax else 2.0 * deltaxstep)
    for box in (countbox, deltaxbox):
        # a typed number applies when the user presses Return or leaves the box, and not after each digit
        box.setKeyboardTracking(False)
    checkboxes.addWidget(emissioncheck)
    checkboxes.addWidget(absorptioncheck)
    checkboxes.addStretch(1)
    optionrows = [(QtWidgets.QLabel("-groupby"), groupbybox), (countlabel, countbox), (deltaxcheck, deltaxbox)]
    for row, (label, box) in enumerate(optionrows, start=5):
        grid.addWidget(label, row, 0)
        grid.addWidget(box, row, 1, QtCore.Qt.AlignmentFlag.AlignLeft)

    commandtext = QtWidgets.QPlainTextEdit()
    commandtext.setReadOnly(True)
    commandtext.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    commandtext.setFixedHeight(8 * commandtext.fontMetrics().lineSpacing() + 12)
    grid.addWidget(commandtext, 8, 0, 1, 3)
    copybutton = QtWidgets.QPushButton("Copy")
    grid.addWidget(copybutton, 9, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
    statuslabel = QtWidgets.QLabel()
    statuslabel.setStyleSheet("color: firebrick")
    statuslabel.setWordWrap(True)
    grid.addWidget(statuslabel, 10, 0, 1, 3)
    grid.setColumnStretch(1, 1)

    signalwidgets: list[QtWidgets.QWidget] = [
        timeslider,
        widthslider,
        xminslider,
        xmaxslider,
        emissioncheck,
        absorptioncheck,
        groupbybox,
        countbox,
        deltaxcheck,
        deltaxbox,
    ]

    def fit_canvas() -> None:
        """Give the canvas the size of the frames, and fit the window on the screen.

        An embedded canvas has no figure manager, thus the figure cannot set the size of the widget. A widget
        of a different size changes the size of the figure, and then the frames do not fit in the figure.
        """
        figwidth, figheight = viewer.figsize
        logicaldpi = viewer.fig.dpi / canvas.device_pixel_ratio
        size = QtCore.QSize(math.ceil(figwidth * logicaldpi), math.ceil(figheight * logicaldpi))
        if canvas.size() == size:
            return
        canvas.setFixedSize(size)
        screen = window.screen().availableGeometry()
        # the margins of the window and its title bar take approximately 60 points
        scrollarea.setMinimumSize(
            min(size.width() + 4, screen.width() - panel.width() - 60), min(size.height() + 4, screen.height() - 60)
        )
        window.adjustSize()

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        blockers = [QtCore.QSignalBlocker(widget) for widget in signalwidgets]
        values = viewer.values
        timeslider.setValue(to_position(math.log10(max(values.centre, viewer.tstart)), *logtrange))
        widthslider.setValue(to_position(values.width, 0.0, widthmax))
        xminslider.setValue(to_position(math.log10(max(float(values.xmin), xlow)), *logxrange))
        xmaxslider.setValue(to_position(math.log10(max(float(values.xmax), xlow)), *logxrange))
        timetext.setText(f"{values.centre:g} d")
        widthtext.setText(f"{values.width:g} d: {viewer.get_timesteps_text()}")
        xminedit.setText(values.xmin)
        xmaxedit.setText(values.xmax)
        emissioncheck.setChecked(values.showemission)
        absorptioncheck.setChecked(values.showabsorption)
        groupbybox.setCurrentText(values.groupby or "none")
        countbox.setValue(values.maxseriescount)
        deltaxcheck.setChecked(bool(values.deltax))
        # an empty check box keeps the last width in deltaxbox, thus that width applies when the user sets it again
        if values.deltax:
            deltaxbox.setValue(float(values.deltax))
        deltaxbox.setEnabled(bool(values.deltax))
        # -maxseriescount applies to an emission plot alone
        for widget in (countlabel, countbox):
            widget.setVisible(values.showemission or values.showabsorption)
        commandtext.setPlainText(viewer.get_command())
        for blocker in blockers:
            blocker.unblock()
        fit_canvas()

    requestedvalues: ControlValues | None = None

    def apply(values: ControlValues) -> None:
        """Show the new values now, and draw them when Qt has no other events to process.

        A drag gives a new value for each movement of the mouse, and a plot takes a maximum of 1 s.
        draw_requested reads only the last values that the user gave, thus the plot follows the drag.
        """
        nonlocal requestedvalues
        if requestedvalues is None:
            QtCore.QTimer.singleShot(0, draw_requested)
        requestedvalues = values
        # show_values reads viewer.values, thus the viewer holds the new values for this call only
        oldvalues, viewer.values = viewer.values, values
        show_values()
        viewer.values = oldvalues

    def draw_requested() -> None:
        """Draw the plot of the last values, or show a message and keep the old values if plotspectra rejects them."""
        nonlocal requestedvalues
        values, requestedvalues = requestedvalues, None
        if values is None:
            return
        message = viewer.change(values)
        statuslabel.setText(message or "")
        show_values()

    def on_time(position: int) -> None:
        centre = 10.0 ** from_position(position, *logtrange)
        apply(dc.replace(viewer.values, centre=float(f"{centre:.4g}")))

    def on_width(position: int) -> None:
        width = from_position(position, 0.0, widthmax)
        apply(dc.replace(viewer.values, width=float(f"{width:.3g}")))

    def on_xslider(_position: int) -> None:
        # a value of 3 significant digits gives a short command, and a text field gives an exact value
        low, high = (
            float(f"{10.0 ** from_position(slider.value(), *logxrange):.3g}") for slider in (xminslider, xmaxslider)
        )
        if low < high:
            apply(dc.replace(viewer.values, xmin=format(low, ".10g"), xmax=format(high, ".10g")))

    def on_xedit() -> None:
        try:
            low, high = float(xminedit.text()), float(xmaxedit.text())
        except ValueError:
            low, high = math.nan, math.nan
        if not 0.0 <= low < high:
            statuslabel.setText("Give two numbers of 0 or more, with the minimum less than the maximum")
            show_values()
            return
        values = dc.replace(viewer.values, xmin=format(low, ".10g"), xmax=format(high, ".10g"))
        if values != viewer.values:
            apply(values)

    def on_emission_options() -> None:
        groupby = None if groupbybox.currentText() == "none" else groupbybox.currentText()
        showemission = emissioncheck.isChecked()
        # -groupby colours the emission plot, thus a choice of -groupby also sets --showemission.
        # An empty --showemission check box removes the -groupby choice.
        if groupby != viewer.values.groupby and groupby is not None:
            showemission = True
        elif not showemission:
            groupby = None
        values = dc.replace(
            viewer.values,
            showemission=showemission,
            showabsorption=absorptioncheck.isChecked(),
            groupby=groupby,
            maxseriescount=countbox.value(),
            deltax=format(deltaxbox.value(), ".10g") if deltaxcheck.isChecked() else "",
        )
        if values != viewer.values:
            apply(values)

    def on_copy() -> None:
        command = viewer.get_command()
        print(command)
        QtWidgets.QApplication.clipboard().setText(command)
        statuslabel.setText("Copied")

    def on_arrow(step: int) -> None:
        # an arrow key moves the time to the middle of the adjacent timestep. The values keep a rounded time, thus
        # the move starts from the nearest middle and not from the first middle after the time
        nearest = int(np.argmin(np.abs(np.array(viewer.tmids) - viewer.values.centre)))
        tmid = viewer.tmids[min(max(nearest + step, 0), len(viewer.tmids) - 1)]
        apply(dc.replace(viewer.values, centre=float(f"{tmid:.4g}")))

    timeslider.valueChanged.connect(on_time)
    widthslider.valueChanged.connect(on_width)
    xminslider.valueChanged.connect(on_xslider)
    xmaxslider.valueChanged.connect(on_xslider)
    xminedit.editingFinished.connect(on_xedit)
    xmaxedit.editingFinished.connect(on_xedit)
    emissioncheck.toggled.connect(on_emission_options)
    absorptioncheck.toggled.connect(on_emission_options)
    groupbybox.currentTextChanged.connect(on_emission_options)
    countbox.valueChanged.connect(on_emission_options)
    deltaxcheck.toggled.connect(on_emission_options)
    deltaxbox.valueChanged.connect(on_emission_options)
    copybutton.clicked.connect(on_copy)
    # a text field takes the arrow keys while it has the focus, and the shortcuts apply otherwise
    QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), window).activated.connect(lambda: on_arrow(-1))
    QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), window).activated.connect(lambda: on_arrow(1))

    window.show()
    show_values()
    app.exec()
    print(viewer.get_command())
