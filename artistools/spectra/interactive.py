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
from artistools.spectra.core import convert_angstroms_to_unit
from artistools.spectra.core import convert_unit_to_angstroms
from artistools.spectra.core import get_xunit
from artistools.spectra.core import XUNITS
from artistools.spectra.plotspectra import addargs
from artistools.spectra.plotspectra import draw_plot
from artistools.spectra.plotspectra import get_artis_run_folders
from artistools.spectra.plotspectra import get_default_xlimits
from artistools.spectra.plotspectra import make_plot_figure
from artistools.spectra.plotspectra import path_is_reference_spectrum
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
    "xunit",
    "logscalex",
    "yscale",
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

# the Play button waits for this time after each plot, thus the user can see each timestep
PLAY_MILLISECONDS: t.Final = 150


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class ControlValues:
    """The values of the controls of the viewer, which give the options of the plotspectra command.

    The x limits and the bin width keep the text of the command, and an empty deltax gives no option.
    """

    centre: float
    width: float
    xmin: str
    xmax: str
    xunit: str
    logscalex: bool
    yscale: str
    showemission: bool
    showabsorption: bool
    groupby: str | None
    maxseriescount: int
    deltax: str
    references: tuple[str, ...]


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
    """Return the -timedays value for a range of this width around the middle time.

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


def get_first_line(errortext: str) -> str:
    """Return the first line of an error for the status line of the window, without the "error: " of print_error."""
    lines = [line.strip() for line in errortext.splitlines() if line.strip()]
    return lines[0].removeprefix("error: ") if lines else REJECTED_MESSAGE


def convert_xunit(values: ControlValues, xunit: str, *, gamma: bool) -> ControlValues:
    """Return the values with the x limits in a new unit of the x axis.

    A frequency or an energy increases where the wavelength decreases, thus the function sorts the limits again. A bin
    width has no linear conversion between a wavelength and a frequency, thus the new unit takes no -deltax. A
    limit of 0 or less has no conversion to a frequency, thus it takes the default limit of the new unit.
    """
    defaultlimits = get_default_xlimits(xunit, gamma=gamma)
    limits = sorted(
        convert_angstroms_to_unit(convert_unit_to_angstroms(float(text), values.xunit), xunit)
        if float(text) > 0.0
        else defaultlimit
        for text, defaultlimit in zip((values.xmin, values.xmax), defaultlimits, strict=True)
    )
    xmin, xmax = (format(float(f"{limit:.4g}"), ".10g") for limit in limits)
    return dc.replace(values, xunit=xunit, xmin=xmin, xmax=xmax, deltax="")


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
        basetokens = remove_options(parser, tokens, CONTROLLED_DESTS)
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
        self.twidths = get_timestep_times(self.runfolders[0], loc="delta")
        self.tstart = get_timestep_times(self.runfolders[0], loc="start")[0]
        self.tend = get_timestep_times(self.runfolders[0], loc="end")[-1]

        # the window adds and removes the reference spectra, thus it keeps them separate from the model paths
        pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
        startpaths = basetokens[:pathcount]
        self.modelpathtokens = [path for path in startpaths if not path_is_reference_spectrum(path)]
        self.othertokens = basetokens[pathcount:]

        # a range of one timestep is a single time, and a plot with no time starts in the middle of the run
        if args.timemin is not None and args.timemax is not None:
            centre = (args.timemin + args.timemax) / 2.0
            coversseveral = sum(args.timemin <= tmid <= args.timemax for tmid in self.tmids) > 1
            width = args.timemax - args.timemin if coversseveral else 0.0
        else:
            centre, width = self.tmids[len(self.tmids) // 2], 0.0

        actions = {action.dest: action for action in parser._actions}  # ruff:ignore[private-member-access]
        self.groupbychoices = [str(choice) for choice in actions["groupby"].choices or ()]
        self.yscalechoices = [str(choice) for choice in actions["yscale"].choices or () if choice != "lin"]
        # a tooltip gives the help text of the option, thus the window and the command line agree
        self.helptexts = {
            dest: str(action.help).replace("%(default)s", str(action.default)).replace("%%", "%")
            for dest, action in actions.items()
            if action.help and action.help != argparse.SUPPRESS
        }
        self.defaultmaxseriescount: int = parser.get_default("maxseriescount")
        self.defaultxunit = "kev" if args.gamma else "angstroms"
        # 4 significant digits of the time and 3 of the width give a short command
        self.values = ControlValues(
            centre=float(f"{centre:.4g}"),
            width=float(f"{width:.3g}"),
            xmin=format(args.xmin, ".10g"),
            xmax=format(args.xmax, ".10g"),
            xunit=args.xunit,
            logscalex=bool(args.logscalex),
            yscale=args.yscale,
            showemission=bool(args.showemission),
            showabsorption=bool(args.showabsorption),
            groupby=givengroupby,
            maxseriescount=args.maxseriescount,
            deltax="" if args.deltax is None else format(args.deltax, ".10g"),
            references=tuple(path for path in startpaths if path_is_reference_spectrum(path)),
        )

        self.fig = fig
        self.axes: npt.NDArray[t.Any] = np.empty(0, dtype=object)
        self.residualaxis: mplax.Axes | None = None
        self.framesforabsorption: bool | None = None
        # a window can change the size of the figure, thus the size of the frames stays here
        self.figsize: tuple[float, float] = (0.0, 0.0)

    def get_plot_tokens(self, values: ControlValues | None = None) -> list[str]:
        """Return the plotspectra arguments of the values, or of the current values if the caller gives none."""
        if values is None:
            values = self.values
        options = [
            *("-t", get_timedays_argument(self.runfolders, values.centre, values.width)),
            *("-xmin", values.xmin, "-xmax", values.xmax),
        ]
        if values.xunit != self.defaultxunit:
            options += ["-xunit", values.xunit]
        if values.logscalex:
            options.append("--logscalex")
        if values.yscale != "auto":
            options += ["-yscale", values.yscale]
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
        return make_command_tokens([*self.modelpathtokens, *values.references, *self.othertokens], options)

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotspectra", *self.get_plot_tokens()])

    def get_timesteps_text(self) -> str:
        """Return the timesteps and the days that the plot reads from spec.out, which holds complete timesteps."""
        timestepmin, timestepmax, daysmin, daysmax = get_time_range(
            self.runfolders[0],
            timedays_range_str=get_timedays_argument(self.runfolders, self.values.centre, self.values.width),
            clamp_to_timesteps=not self.args.notimeclamp,
        )
        timesteps = (
            f"timestep {timestepmin}" if timestepmin == timestepmax else f"timesteps {timestepmin} to {timestepmax}"
        )
        return f"The plot reads {timesteps}, from {daysmin:.4g} to {daysmax:.4g} d"

    def get_rejection(self, values: ControlValues) -> str | None:
        """Return why plotspectra rejects the arguments of the values, or None if it accepts them.

        This parses and checks the arguments and draws no plot, thus the window can test each choice of a control.
        A missing input file stops a plot later, and this test does not find it.
        """
        errors = io.StringIO()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
                plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens(values))
                resolve_plot_args(plotargs)
        except SystemExit:
            return get_first_line(errors.getvalue())
        except (FileNotFoundError, ValueError) as exc:
            return get_first_line(str(exc))
        if (plotargs.showemission, plotargs.showabsorption) != (values.showemission, values.showabsorption):
            return "A different option of the command keeps the emission plot on"
        return None

    def draw(self, *, quiet: bool = True) -> str | None:
        """Draw the plot of the command, and return the reason for the status line if plotspectra rejects it.

        Each plot prints the same lines again, e.g. the list of the ions of an emission plot. Thus a quiet plot
        discards the standard output. The terminal shows the whole error, and the status line shows its first line.
        """
        output = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        errors = io.StringIO()
        try:
            with output, contextlib.redirect_stderr(errors):
                return self.draw_command()
        except SystemExit:
            # exit_with_error printed a line that starts with "error: ", and a help line
            return get_first_line(errors.getvalue())
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc))
            return get_first_line(str(exc))
        finally:
            sys.stderr.write(errors.getvalue())

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

    def get_adjacent_centre(self, step: int) -> float | None:
        """Return the middle of the timestep beside the nearest timestep, or None at the end of the run.

        The values keep a rounded time, thus the step starts from the nearest middle and not from the first
        middle after the time.
        """
        nearest = int(np.argmin(np.abs(np.array(self.tmids) - self.values.centre)))
        if not 0 <= nearest + step < len(self.tmids):
            return None
        return float(f"{self.tmids[nearest + step]:.4g}")

    def get_nearest_timestep_width(self) -> float:
        """Return the width in days of the timestep that holds the middle of the time range."""
        return self.twidths[int(np.argmin(np.abs(np.array(self.tmids) - self.values.centre)))]


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

    from artistools.commands import get_path

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
    panellayout = QtWidgets.QVBoxLayout(panel)
    panelscroll = QtWidgets.QScrollArea()
    panelscroll.setWidget(panel)
    panelscroll.setWidgetResizable(True)
    panelscroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    panelscroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    panelscroll.setFixedWidth(600)
    layout.addWidget(panelscroll, alignment=QtCore.Qt.AlignmentFlag.AlignTop)

    def add_section(title: str) -> tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]:
        box = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(box)
        grid.setColumnStretch(1, 1)
        panellayout.addWidget(box)
        return box, grid

    def make_slider() -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        # the arrow keys move the time and change the width, thus a slider must not take them
        slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        return slider

    # each slider maps its position from 0 to SLIDER_STEPS onto the range of its value
    def to_position(value: float, low: float, high: float) -> int:
        return round(SLIDER_STEPS * (min(max(value, low), high) - low) / (high - low))

    def from_position(position: int, low: float, high: float) -> float:
        return low + (high - low) * position / SLIDER_STEPS

    helptexts = viewer.helptexts
    logtrange = (math.log10(viewer.tstart), math.log10(viewer.tend))
    widthmax = max((viewer.tend - viewer.tstart) / 4.0, viewer.values.width)

    _, timegrid = add_section("Time")
    timeslider, widthslider = make_slider(), make_slider()
    timeedit, widthedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    timestepslabel = QtWidgets.QLabel()
    playbutton = QtWidgets.QPushButton("Play")
    playbutton.setCheckable(True)
    timetip = (
        "The middle of the time range in days. The Left key and the Right key move it to the middle of the adjacent"
        " timestep."
    )
    widthtip = (
        "The width of the time range in days. A width of 0 gives one timestep. The Up key and the Down key change"
        " the width by one timestep."
    )
    for row, (label, slider, edit, tip) in enumerate([
        ("Time [d]", timeslider, timeedit, timetip),
        ("Δt [d]", widthslider, widthedit, widthtip),
    ]):
        edit.setFixedWidth(110)
        for widget in (slider, edit):
            widget.setToolTip(tip)
        timegrid.addWidget(QtWidgets.QLabel(label), row, 0)
        timegrid.addWidget(slider, row, 1)
        timegrid.addWidget(edit, row, 2)
    timegrid.addWidget(timestepslabel, 2, 0, 1, 2)
    timegrid.addWidget(playbutton, 2, 2)
    playbutton.setToolTip("Move the time through the timesteps of the run")

    xbox, xgrid = add_section("")
    xminslider, xmaxslider = make_slider(), make_slider()
    xminedit, xmaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    xminlabel, xmaxlabel = QtWidgets.QLabel(), QtWidgets.QLabel()
    zoomtip = " Drag across the plot to select a range. Double-click the plot to get the default range."
    for row, (label, slider, edit, dest) in enumerate([
        (xminlabel, xminslider, xminedit, "xmin"),
        (xmaxlabel, xmaxslider, xmaxedit, "xmax"),
    ]):
        edit.setFixedWidth(110)
        for widget in (slider, edit):
            widget.setToolTip(helptexts.get(dest, "") + zoomtip)
        xgrid.addWidget(label, row, 0)
        xgrid.addWidget(slider, row, 1)
        xgrid.addWidget(edit, row, 2)

    _, axesgrid = add_section("Axes")
    xunitbox, yscalebox = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    xunitbox.addItems(list(XUNITS))
    yscalebox.addItems(viewer.yscalechoices)
    logscalexcheck = QtWidgets.QCheckBox("--logscalex")
    for row, (label, widget, dest) in enumerate([
        (QtWidgets.QLabel("-xunit"), xunitbox, "xunit"),
        (QtWidgets.QLabel("-yscale"), yscalebox, "yscale"),
    ]):
        widget.setToolTip(helptexts.get(dest, ""))
        axesgrid.addWidget(label, row, 0)
        axesgrid.addWidget(widget, row, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
    logscalexcheck.setToolTip(helptexts.get("logscalex", ""))
    axesgrid.addWidget(logscalexcheck, 2, 0, 1, 2)

    _, emissiongrid = add_section("Emission and absorption")
    emissioncheck = QtWidgets.QCheckBox("--showemission")
    absorptioncheck = QtWidgets.QCheckBox("--showabsorption")
    groupbybox = QtWidgets.QComboBox()
    groupbybox.addItems(["none", *viewer.groupbychoices])
    countlabel = QtWidgets.QLabel("-maxseriescount")
    countbox = QtWidgets.QSpinBox()
    countbox.setRange(1, 200)
    for widget, dest in (
        (emissioncheck, "showemission"),
        (absorptioncheck, "showabsorption"),
        (groupbybox, "groupby"),
        (countbox, "maxseriescount"),
    ):
        widget.setToolTip(helptexts.get(dest, ""))
    emissiongrid.addWidget(emissioncheck, 0, 0)
    emissiongrid.addWidget(absorptioncheck, 0, 1)
    emissiongrid.addWidget(QtWidgets.QLabel("-groupby"), 1, 0)
    emissiongrid.addWidget(groupbybox, 1, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
    emissiongrid.addWidget(countlabel, 2, 0)
    emissiongrid.addWidget(countbox, 2, 1, QtCore.Qt.AlignmentFlag.AlignLeft)

    _, bingrid = add_section("Bins of the packet spectrum")
    # an empty check box gives no -deltax, and plotspectra then takes its default bins
    deltaxcheck = QtWidgets.QCheckBox()
    deltaxbox = QtWidgets.QDoubleSpinBox()
    for widget in (deltaxcheck, deltaxbox):
        widget.setToolTip(helptexts.get("deltax", ""))
    bingrid.addWidget(deltaxcheck, 0, 0)
    bingrid.addWidget(deltaxbox, 0, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
    for box in (countbox, deltaxbox):
        # a typed number applies when the user presses Return or leaves the box, and not after each digit
        box.setKeyboardTracking(False)

    _, referencegrid = add_section("Reference spectra")
    referencelist = QtWidgets.QListWidget()
    referencelist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    referencelist.setFixedHeight(4 * referencelist.fontMetrics().lineSpacing() + 12)
    referencelist.setToolTip(
        "The observed spectra of the plot. The command gives a file from the reference data of artistools by its name alone."
    )
    addbutton, removebutton = QtWidgets.QPushButton("Add..."), QtWidgets.QPushButton("Remove")
    referencegrid.addWidget(referencelist, 0, 0, 1, 2)
    referencegrid.addWidget(addbutton, 1, 0)
    referencegrid.addWidget(removebutton, 1, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
    referencefolder = get_path("artistools_dir") / "data" / "refspectra"

    _, commandgrid = add_section("Command")
    commandtext = QtWidgets.QPlainTextEdit()
    commandtext.setReadOnly(True)
    commandtext.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    commandtext.setFixedHeight(7 * commandtext.fontMetrics().lineSpacing() + 12)
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip("Copy the command to the clipboard")
    commandgrid.addWidget(commandtext, 0, 0, 1, 2)
    commandgrid.addWidget(copybutton, 1, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
    statuslabel = QtWidgets.QLabel()
    statuslabel.setStyleSheet("color: firebrick")
    statuslabel.setWordWrap(True)
    panellayout.addWidget(statuslabel)
    panellayout.addStretch(1)

    signalwidgets: list[QtWidgets.QWidget] = [
        timeslider,
        widthslider,
        xminslider,
        xmaxslider,
        xunitbox,
        yscalebox,
        logscalexcheck,
        emissioncheck,
        absorptioncheck,
        groupbybox,
        countbox,
        deltaxcheck,
        deltaxbox,
    ]

    # the x sliders and the step of -deltax follow the unit of the x axis, thus a new unit sets them again
    logxrange = (0.0, 1.0)
    rangesunit: str | None = None

    def set_xunit_ranges() -> None:
        nonlocal logxrange, rangesunit
        values = viewer.values
        defaultxmin, defaultxmax = get_default_xlimits(values.xunit, gamma=viewer.args.gamma)
        # the x sliders act on log10(x), thus their range must be above zero
        xlow = min(value for value in (float(values.xmin), defaultxmin) if value > 0.0) / 2.0
        xhigh = max(float(values.xmax), defaultxmax) * 2.0
        logxrange = (math.log10(xlow), math.log10(xhigh))
        # a step of approximately 1/1000 of the x range is correct for each x unit, e.g. 10 Å for wavelengths
        xspan = defaultxmax - defaultxmin
        deltaxstep = 10.0 ** math.floor(math.log10(xspan / 1000.0))
        deltaxbox.setDecimals(max(0, -math.floor(math.log10(deltaxstep))))
        deltaxbox.setRange(deltaxstep, xspan)
        deltaxbox.setSingleStep(deltaxstep)
        if not values.deltax:
            deltaxbox.setValue(2.0 * deltaxstep)
        xunit = get_xunit(values.xunit)
        xbox.setTitle(xunit.kind.capitalize())
        xminlabel.setText(f"{xunit.kind.capitalize()} min [{xunit.label}]")
        xmaxlabel.setText(f"{xunit.kind.capitalize()} max [{xunit.label}]")
        deltaxcheck.setText(f"-deltax [{xunit.label}]")
        rangesunit = values.xunit

    # each test of a choice parses the arguments again, thus the code keeps one result for each set of options
    rejections: dict[tuple[t.Any, ...], tuple[list[str | None], str | None, str | None]] = {}

    def get_rejections() -> tuple[list[str | None], str | None, str | None]:
        """Return why plotspectra rejects each -groupby choice, --showemission, and --showabsorption."""
        values = viewer.values
        key = (values.showemission, values.showabsorption, values.groupby, values.references, bool(values.deltax))
        if key not in rejections:
            groupbys = [
                None
                if choice in {"none", values.groupby}
                else viewer.get_rejection(dc.replace(values, groupby=choice, showemission=True))
                for choice in ("none", *viewer.groupbychoices)
            ]
            emission = None if values.showemission else viewer.get_rejection(dc.replace(values, showemission=True))
            absorption = (
                None if values.showabsorption else viewer.get_rejection(dc.replace(values, showabsorption=True))
            )
            rejections[key] = (groupbys, emission, absorption)
        return rejections[key]

    def show_rejections() -> None:
        """Disable each choice that plotspectra rejects, and give the reason in its tooltip."""
        groupbys, emission, absorption = get_rejections()
        model = groupbybox.model()
        if isinstance(model, QtGui.QStandardItemModel):
            for index, reason in enumerate(groupbys):
                if (item := model.item(index)) is not None:
                    item.setEnabled(reason is None)
                    item.setToolTip(reason or "")
        for checkbox, reason, dest in (
            (emissioncheck, emission, "showemission"),
            (absorptioncheck, absorption, "showabsorption"),
        ):
            checkbox.setEnabled(reason is None)
            checkbox.setToolTip(reason or helptexts.get(dest, ""))

    def fit_window() -> None:
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
        maxheight = screen.height() - 60
        scrollarea.setMinimumSize(
            min(size.width() + 4, screen.width() - panelscroll.width() - 60), min(size.height() + 4, maxheight)
        )
        panelscroll.setMinimumHeight(min(panel.sizeHint().height(), maxheight))
        window.adjustSize()

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        blockers = [QtCore.QSignalBlocker(widget) for widget in signalwidgets]
        values = viewer.values
        if values.xunit != rangesunit:
            set_xunit_ranges()
        timeslider.setValue(to_position(math.log10(max(values.centre, viewer.tstart)), *logtrange))
        widthslider.setValue(to_position(values.width, 0.0, widthmax))
        timeedit.setText(f"{values.centre:g}")
        widthedit.setText(f"{values.width:g}")
        timestepslabel.setText(viewer.get_timesteps_text())
        xminslider.setValue(to_position(math.log10(max(float(values.xmin), 10.0 ** logxrange[0])), *logxrange))
        xmaxslider.setValue(to_position(math.log10(max(float(values.xmax), 10.0 ** logxrange[0])), *logxrange))
        xminedit.setText(values.xmin)
        xmaxedit.setText(values.xmax)
        xunitbox.setCurrentText(values.xunit)
        yscalebox.setCurrentText(values.yscale)
        logscalexcheck.setChecked(values.logscalex)
        emissioncheck.setChecked(values.showemission)
        absorptioncheck.setChecked(values.showabsorption)
        groupbybox.setCurrentText(values.groupby or "none")
        countbox.setValue(values.maxseriescount)
        # -maxseriescount applies only to an emission or absorption plot, and a disabled box keeps its place
        for widget in (countlabel, countbox):
            widget.setEnabled(values.showemission or values.showabsorption)
        deltaxcheck.setChecked(bool(values.deltax))
        # an empty check box keeps the last width in deltaxbox, thus that width applies when the user sets it again
        if values.deltax:
            deltaxbox.setValue(float(values.deltax))
        deltaxbox.setEnabled(bool(values.deltax))
        if [referencelist.item(index).text() for index in range(referencelist.count())] != list(values.references):
            referencelist.clear()
            referencelist.addItems(list(values.references))
        commandtext.setPlainText(viewer.get_command())
        show_rejections()
        for blocker in blockers:
            blocker.unblock()
        fit_window()

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
        # a plot of an emission file or of the packets is slow, thus the cursor shows that the window is busy
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            message = viewer.change(values)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        statuslabel.setText(message or "")
        show_values()
        # a rejection occurs again at each step, thus a rejection stops the Play button
        if message is not None:
            playbutton.setChecked(False)
        elif playbutton.isChecked():
            QtCore.QTimer.singleShot(PLAY_MILLISECONDS, play_step)

    def show_error(message: str) -> None:
        statuslabel.setText(message)
        show_values()

    def on_time(position: int) -> None:
        centre = 10.0 ** from_position(position, *logtrange)
        apply(dc.replace(viewer.values, centre=float(f"{centre:.4g}")))

    def on_width(position: int) -> None:
        width = from_position(position, 0.0, widthmax)
        apply(dc.replace(viewer.values, width=float(f"{width:.3g}")))

    def on_timeedit() -> None:
        try:
            centre, width = float(timeedit.text()), float(widthedit.text())
        except ValueError:
            show_error("Give a number of days for the time and for Δt")
            return
        if not viewer.tstart <= centre <= viewer.tend or width < 0.0:
            show_error(f"Give a time from {viewer.tstart:g} to {viewer.tend:g} d, and a Δt of 0 or more")
            return
        values = dc.replace(viewer.values, centre=float(f"{centre:.4g}"), width=float(f"{width:.3g}"))
        if values != viewer.values:
            apply(values)

    def on_arrow(step: int) -> None:
        if (centre := viewer.get_adjacent_centre(step)) is not None:
            apply(dc.replace(viewer.values, centre=centre))

    def on_widthstep(step: int) -> None:
        width = max(0.0, viewer.values.width + step * viewer.get_nearest_timestep_width())
        apply(dc.replace(viewer.values, width=float(f"{width:.3g}")))

    def play_step() -> None:
        if not playbutton.isChecked():
            return
        if (centre := viewer.get_adjacent_centre(1)) is None:
            playbutton.setChecked(False)
            return
        apply(dc.replace(viewer.values, centre=centre))

    def on_play(checked: bool) -> None:
        playbutton.setText("Pause" if checked else "Play")
        if checked:
            play_step()

    def set_xlimits(low: float, high: float) -> None:
        # a value of 3 significant digits gives a short command, and a text field gives an exact value
        low, high = float(f"{low:.3g}"), float(f"{high:.3g}")
        if low < high:
            apply(dc.replace(viewer.values, xmin=format(low, ".10g"), xmax=format(high, ".10g")))

    def on_xslider(_position: int) -> None:
        set_xlimits(*(10.0 ** from_position(slider.value(), *logxrange) for slider in (xminslider, xmaxslider)))

    def on_xedit() -> None:
        try:
            low, high = float(xminedit.text()), float(xmaxedit.text())
        except ValueError:
            low, high = math.nan, math.nan
        if not 0.0 <= low < high:
            show_error("Give two numbers of 0 or more, with the minimum less than the maximum")
            return
        values = dc.replace(viewer.values, xmin=format(low, ".10g"), xmax=format(high, ".10g"))
        if values != viewer.values:
            apply(values)

    def on_axes() -> None:
        values = viewer.values
        if xunitbox.currentText() != values.xunit:
            values = convert_xunit(values, xunitbox.currentText(), gamma=viewer.args.gamma)
        values = dc.replace(values, yscale=yscalebox.currentText(), logscalex=logscalexcheck.isChecked())
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

    def on_add_reference() -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(window, "Add reference spectra", str(referencefolder))
        # plotspectra finds a file of the reference data by its name, thus the command stays short
        names = [
            Path(filename).name if Path(filename).parent.resolve() == referencefolder.resolve() else filename
            for filename in filenames
        ]
        references = (*viewer.values.references, *(name for name in names if name not in viewer.values.references))
        if references != viewer.values.references:
            apply(dc.replace(viewer.values, references=references))

    def on_remove_reference() -> None:
        selected = {item.text() for item in referencelist.selectedItems()}
        references = tuple(name for name in viewer.values.references if name not in selected)
        if references != viewer.values.references:
            apply(dc.replace(viewer.values, references=references))

    def on_copy() -> None:
        command = viewer.get_command()
        print(command)
        QtWidgets.QApplication.clipboard().setText(command)
        statuslabel.setText("Copied")

    # a drag across a frame selects the x range, and a double-click gives the default range
    dragstart: tuple[float, float] | None = None
    dragspan: t.Any = None

    def get_frame(event: t.Any) -> "mplax.Axes | None":
        frames = [axis for axis in (*viewer.axes, viewer.residualaxis) if axis is not None]
        return next((axis for axis in frames if event.inaxes is axis), None)

    def on_press(event: t.Any) -> None:
        nonlocal dragstart, dragspan
        frame = get_frame(event)
        if frame is None or event.button != 1 or event.xdata is None:
            return
        if event.dblclick:
            set_xlimits(*get_default_xlimits(viewer.values.xunit, gamma=viewer.args.gamma))
            return
        dragstart = (event.xdata, event.x)
        dragspan = frame.axvspan(event.xdata, event.xdata, color="0.5", alpha=0.3)

    def on_motion(event: t.Any) -> None:
        if dragstart is None or dragspan is None or event.xdata is None or get_frame(event) is None:
            return
        dragspan.set_x(min(dragstart[0], event.xdata))
        dragspan.set_width(abs(event.xdata - dragstart[0]))
        canvas.draw_idle()

    def on_release(event: t.Any) -> None:
        nonlocal dragstart, dragspan
        if dragstart is None or dragspan is None:
            return
        start, dragspan = dragstart, dragspan.remove()
        dragstart = None
        canvas.draw_idle()
        # a movement of a few pixels is a click and not a selection
        if event.xdata is not None and get_frame(event) is not None and abs(event.x - start[1]) > 5:
            set_xlimits(*sorted((start[0], event.xdata)))

    timeslider.valueChanged.connect(on_time)
    widthslider.valueChanged.connect(on_width)
    timeedit.editingFinished.connect(on_timeedit)
    widthedit.editingFinished.connect(on_timeedit)
    playbutton.toggled.connect(on_play)
    xminslider.valueChanged.connect(on_xslider)
    xmaxslider.valueChanged.connect(on_xslider)
    xminedit.editingFinished.connect(on_xedit)
    xmaxedit.editingFinished.connect(on_xedit)
    xunitbox.currentTextChanged.connect(on_axes)
    yscalebox.currentTextChanged.connect(on_axes)
    logscalexcheck.toggled.connect(on_axes)
    emissioncheck.toggled.connect(on_emission_options)
    absorptioncheck.toggled.connect(on_emission_options)
    groupbybox.currentTextChanged.connect(on_emission_options)
    countbox.valueChanged.connect(on_emission_options)
    deltaxcheck.toggled.connect(on_emission_options)
    deltaxbox.valueChanged.connect(on_emission_options)
    addbutton.clicked.connect(on_add_reference)
    removebutton.clicked.connect(on_remove_reference)
    copybutton.clicked.connect(on_copy)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)
    # a text field takes the arrow keys while it has the focus, and the shortcuts apply otherwise
    for key, callback in (
        (QtCore.Qt.Key.Key_Left, lambda: on_arrow(-1)),
        (QtCore.Qt.Key.Key_Right, lambda: on_arrow(1)),
        (QtCore.Qt.Key.Key_Up, lambda: on_widthstep(1)),
        (QtCore.Qt.Key.Key_Down, lambda: on_widthstep(-1)),
    ):
        QtGui.QShortcut(QtGui.QKeySequence(key), window).activated.connect(callback)

    window.show()
    show_values()
    app.exec()
    print(viewer.get_command())
