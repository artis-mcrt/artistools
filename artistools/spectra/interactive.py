"""Show the plot of plotspectra in a window with controls for the time, the x range, and the emission plot."""

import argparse
import contextlib
import dataclasses as dc
import io
import math
import re
import shlex
import sys
import time
import typing as t
from pathlib import Path

import matplotlib.figure as mplfig
import numpy as np
import polars as pl

from artistools.commands import SuggestingArgumentParser
from artistools.misc import addarg_quiet
from artistools.misc import exit_with_error
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_time_range
from artistools.misc import get_timestep_times
from artistools.misc import import_optional
from artistools.misc import parse_cli_args
from artistools.misc import print_error
from artistools.plottools import plain_label
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
    "notimeclamp",
    "xmin",
    "xmax",
    "xunit",
    "logscalex",
    "yscale",
    "ymin",
    "ymax",
    "showemission",
    "showabsorption",
    "emissionabsorption",
    "groupby",
    "maxseriescount",
    "deltax",
    "fixedionlist",
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
    notimeclamp: bool
    xmin: str
    xmax: str
    xunit: str
    logscalex: bool
    yscale: str
    ymin: str
    ymax: str
    showemission: bool
    showabsorption: bool
    groupby: str | None
    maxseriescount: int
    deltax: str
    fixedionlist: tuple[str, ...]
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


def get_timedays_argument(centre: float, width: float, bounds: tuple[float, float]) -> str:
    """Return the -timedays value of a continuous time range of this width around the middle time.

    A width of zero gives the middle time alone, and plotspectra then reads the timestep that holds it. The range
    stays inside the bounds, which are the valid times of the first run.
    """
    if width > 0.0:
        lowtext = format_days(max(centre - width / 2.0, bounds[0]))
        hightext = format_days(min(centre + width / 2.0, bounds[1]))
        if float(lowtext) < float(hightext):
            return f"{lowtext}-{hightext}"

    return format_days(centre)


def get_shortest_decimal(low: float, high: float, *, roundup: bool) -> str:
    """Return the decimal with the fewest digits after the point in the interval from low to high.

    A value that rounds down stays above low and can equal high. A value that rounds up can equal low and stays
    below high.
    """
    for decimals in range(8):
        scale = 10.0**decimals
        value = (math.ceil(low * scale) if roundup else math.floor(high * scale)) / scale
        text = f"{value:.{decimals}f}"
        if (low <= float(text) < high) if roundup else (low < float(text) <= high):
            return text
    return format_days(high if not roundup else low)


def get_snapped_timedays_argument(
    tmids: "Sequence[float]", tstarts: "Sequence[float]", tends: "Sequence[float]", first: int, last: int
) -> str:
    """Return the shortest -timedays value that selects the timesteps from first to last, as plotspectra clamps it.

    A clamped range selects each timestep with its middle inside the range, thus each bound can be any value
    between two middles. One timestep takes a single time, which selects the timestep that holds it.
    """
    if first == last:
        # a single time selects the timestep that holds it, from the start of the timestep up to its end
        for decimals in range(8):
            text = f"{round(tmids[first], decimals):.{decimals}f}"
            if tstarts[first] <= float(text) < tends[first]:
                return text
        return format_days(tmids[first])
    lowlimit = tmids[first - 1] if first > 0 else 0.0
    highlimit = tmids[last + 1] if last + 1 < len(tmids) else math.inf
    lowtext = get_shortest_decimal(lowlimit, tmids[first], roundup=False)
    hightext = get_shortest_decimal(tmids[last], highlimit, roundup=True)
    return f"{lowtext}-{hightext}"


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
        self.tstarts = get_timestep_times(self.runfolders[0], loc="start")
        self.tends = get_timestep_times(self.runfolders[0], loc="end")
        self.twidths = get_timestep_times(self.runfolders[0], loc="delta")

        # plotspectra rejects a time outside the arrival times of the escaped packets, thus the controls stay inside
        # them. With --plotinvalidpart, plotspectra accepts all times.
        validstart, validend = None, None
        if not args.plotinvalidpart:
            with contextlib.suppress(FileNotFoundError):
                _, validstart, validend = get_escaped_arrivalrange(self.runfolders[0])
        self.timebounds = (
            self.tstarts[0] if validstart is None else max(self.tstarts[0], float(validstart)),
            self.tends[-1] if validend is None else min(self.tends[-1], float(validend)),
        )
        self.validtimesteps = [
            timestep
            for timestep in range(len(self.tmids))
            if self.tstarts[timestep] >= self.timebounds[0] and self.tends[timestep] <= self.timebounds[1]
        ] or list(range(len(self.tmids)))

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
            centre, width = self.tmids[self.validtimesteps[len(self.validtimesteps) // 2]], 0.0

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
        values = ControlValues(
            centre=float(f"{centre:.4g}"),
            width=float(f"{width:.3g}"),
            notimeclamp=bool(args.notimeclamp),
            xmin=format(args.xmin, ".10g"),
            xmax=format(args.xmax, ".10g"),
            xunit=args.xunit,
            logscalex=bool(args.logscalex),
            yscale=args.yscale,
            ymin="" if args.ymin is None else format(args.ymin, ".10g"),
            ymax="" if args.ymax is None else format(args.ymax, ".10g"),
            showemission=bool(args.showemission),
            showabsorption=bool(args.showabsorption),
            groupby=givengroupby,
            maxseriescount=args.maxseriescount,
            deltax="" if args.deltax is None else format(args.deltax, ".10g"),
            fixedionlist=tuple(args.fixedionlist or ()),
            references=tuple(path for path in startpaths if path_is_reference_spectrum(path)),
        )
        self.values = values if values.notimeclamp else self.snap(values, *self.get_selection(values))

        self.fig = fig
        self.axes: npt.NDArray[t.Any] = np.empty(0, dtype=object)
        self.residualaxis: mplax.Axes | None = None
        self.framesforabsorption: bool | None = None
        # a window can change the size of the figure, thus the size of the frames stays here
        self.figsize: tuple[float, float] = (0.0, 0.0)
        # the readout of the window reads the contributions of an emission plot from this frame
        self.dfalldata = pl.DataFrame()

    def get_selection(self, values: ControlValues) -> tuple[int, int]:
        """Return the first and the last valid timestep with a middle in the time range of the values.

        This is the rule of get_time_range for a clamped range. A range that holds no middle gives the valid
        timestep with the nearest middle.
        """
        tolerance = 1e-9 * max(abs(values.centre), 1.0)
        low, high = values.centre - values.width / 2.0 - tolerance, values.centre + values.width / 2.0 + tolerance
        inside = [timestep for timestep in self.validtimesteps if low <= self.tmids[timestep] <= high]
        if inside:
            return inside[0], inside[-1]
        nearest = min(self.validtimesteps, key=lambda timestep: abs(self.tmids[timestep] - values.centre))
        return nearest, nearest

    def snap(self, values: ControlValues, first: int, last: int) -> ControlValues:
        """Return the values for a time range from the middle of the first timestep to the middle of the last."""
        return dc.replace(
            values,
            centre=(self.tmids[first] + self.tmids[last]) / 2.0,
            width=self.tmids[last] - self.tmids[first],
            notimeclamp=False,
        )

    def get_plot_tokens(self, values: ControlValues | None = None) -> list[str]:
        """Return the plotspectra arguments of the values, or of the current values if the caller gives none."""
        if values is None:
            values = self.values
        if values.notimeclamp:
            timedays = get_timedays_argument(values.centre, values.width, self.timebounds)
        else:
            timedays = get_snapped_timedays_argument(self.tmids, self.tstarts, self.tends, *self.get_selection(values))
        options = ["-t", timedays, "-xmin", values.xmin, "-xmax", values.xmax]
        if values.notimeclamp:
            options.append("--notimeclamp")
        if values.xunit != self.defaultxunit:
            options += ["-xunit", values.xunit]
        if values.logscalex:
            options.append("--logscalex")
        if values.yscale != "auto":
            options += ["-yscale", values.yscale]
        if values.ymin:
            options += ["-ymin", values.ymin]
        if values.ymax:
            options += ["-ymax", values.ymax]
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
        # a list option takes each word that follows it, thus it comes after every other option
        if values.fixedionlist and (values.showemission or values.showabsorption):
            options += ["-fixedionlist", *values.fixedionlist]
        return make_command_tokens([*self.modelpathtokens, *values.references, *self.othertokens], options)

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotspectra", *self.get_plot_tokens()])

    def get_timesteps_text(self) -> str:
        """Return the timesteps and the days that the plot reads from spec.out, which holds complete timesteps."""
        timedays = self.get_plot_tokens()[len(self.modelpathtokens) + len(self.values.references) + 1]
        timestepmin, timestepmax, daysmin, daysmax = get_time_range(
            self.runfolders[0], timedays_range_str=timedays, clamp_to_timesteps=not self.values.notimeclamp
        )
        timesteps = (
            f"timestep {timestepmin}" if timestepmin == timestepmax else f"timesteps {timestepmin} to {timestepmax}"
        )
        return f"The plot reads {timesteps}, from {daysmin:.4g} to {daysmax:.4g} d"

    def get_nearest_position(self) -> int:
        """Return the position in the valid timesteps of the timestep with the middle nearest to the time."""
        return min(
            range(len(self.validtimesteps)),
            key=lambda position: abs(self.tmids[self.validtimesteps[position]] - self.values.centre),
        )

    def get_selection_positions(self) -> tuple[int, int]:
        """Return the positions in the valid timesteps of the first and the last timestep of the time range."""
        first, last = self.get_selection(self.values)
        return self.validtimesteps.index(first), self.validtimesteps.index(last)

    def step_time(self, step: int) -> ControlValues | None:
        """Return the values with the time one timestep later or earlier, or None at the end of the valid range."""
        if self.values.notimeclamp:
            # a continuous range keeps its width and moves its middle to the middle of the adjacent timestep
            position = self.get_nearest_position() + step
            if not 0 <= position < len(self.validtimesteps):
                return None
            return dc.replace(self.values, centre=float(f"{self.tmids[self.validtimesteps[position]]:.4g}"))
        firstpos, lastpos = self.get_selection_positions()
        if firstpos + step < 0 or lastpos + step >= len(self.validtimesteps):
            return None
        return self.snap(self.values, self.validtimesteps[firstpos + step], self.validtimesteps[lastpos + step])

    def step_width(self, step: int) -> ControlValues:
        """Return the values with the time range one timestep wider or narrower."""
        if self.values.notimeclamp:
            width = self.values.width + step * self.twidths[self.validtimesteps[self.get_nearest_position()]]
            return dc.replace(self.values, width=float(f"{max(0.0, width):.3g}"))
        firstpos, lastpos = self.get_selection_positions()
        lastpos = min(max(lastpos + step, firstpos), len(self.validtimesteps) - 1)
        return self.snap(self.values, self.validtimesteps[firstpos], self.validtimesteps[lastpos])

    def move_to_end(self, *, last: bool) -> ControlValues:
        """Return the values with the time at the first or at the last valid timestep, and the same width."""
        if self.values.notimeclamp:
            timestep = self.validtimesteps[-1 if last else 0]
            return dc.replace(self.values, centre=float(f"{self.tmids[timestep]:.4g}"))
        firstpos, lastpos = self.get_selection_positions()
        count = lastpos - firstpos
        start = len(self.validtimesteps) - 1 - count if last else 0
        return self.snap(self.values, self.validtimesteps[start], self.validtimesteps[start + count])

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

        self.dfalldata, _ = draw_plot(plotargs, self.axes, self.residualaxis)
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

    def get_drawn_series(self) -> tuple[str, ...]:
        """Return the contribution series of the emission plot in the order of the plot, without "Other"."""
        names: list[str] = []
        for column in self.dfalldata.columns:
            for prefix in ("emission_flambda.", "absorption_flambda."):
                name = column.removeprefix(prefix)
                if column.startswith(prefix) and name != "Other" and name not in names:
                    names.append(name)
        return tuple(names)

    def get_readout(self, x: float) -> str:
        """Return the value of each drawn spectrum at x, and the strongest emission at x for an emission plot."""
        xunit = get_xunit(self.values.xunit)
        parts = [f"{x:.5g} {xunit.label}"]
        for line in self.axes[0].get_lines() if len(self.axes) else []:
            label = plain_label(str(line.get_label()))
            xdata, ydata = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
            if label.startswith("_") or xdata.size < 2:
                continue
            order = np.argsort(xdata)
            if xdata[order[0]] <= x <= xdata[order[-1]]:
                parts.append(f"{label}: {np.interp(x, xdata[order], ydata[order]):.3g}")
        emissioncolumns = [column for column in self.dfalldata.columns if column.startswith("emission_flambda.")]
        if emissioncolumns and "lambda_angstroms" in self.dfalldata.columns:
            lambda_angstroms = convert_unit_to_angstroms(x, self.values.xunit)
            row = self.dfalldata.row(
                int(np.argmin(np.abs(self.dfalldata["lambda_angstroms"].to_numpy() - lambda_angstroms))), named=True
            )
            emissions = {
                column.removeprefix("emission_flambda."): float(row[column] or 0.0) for column in emissioncolumns
            }
            total = sum(value for value in emissions.values() if value > 0.0)
            # "Other" holds all the small series, thus the readout names it only if no named series emits here
            named = [name for name in emissions if name != "Other" and emissions[name] > 0.0] or list(emissions)
            strongest = max(named, key=lambda name: emissions[name])
            if total > 0.0:
                parts.append(f"strongest emission: {strongest} ({100.0 * emissions[strongest] / total:.0f}%)")
        return "   ".join(parts)


def make_icon_pixmap(size: int) -> t.Any:
    """Return a pixmap of the icon of the viewer: a spectrum over a dark square.

    The window gives the icon to the Dock.
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
    xvalues = np.linspace(0.1, 0.9, 200)
    yvalues = 0.72 - 0.45 * np.exp(-(((xvalues - 0.42) / 0.06) ** 2)) - 0.25 * np.exp(-(((xvalues - 0.65) / 0.09) ** 2))
    path.moveTo(float(xvalues[0]) * size, float(yvalues[0]) * size)
    for xvalue, yvalue in zip(xvalues[1:].tolist(), yvalues[1:].tolist(), strict=True):
        path.lineTo(xvalue * size, yvalue * size)
    painter.setPen(QtGui.QPen(QtGui.QColor("#f5a623"), size * 0.05))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.end()
    return pixmap


KEYBOARD_HELP: t.Final = """<table>
<tr><td><b>Left</b>, <b>Right</b></td><td>Move the time to the adjacent timestep</td></tr>
<tr><td><b>Up</b>, <b>Down</b></td><td>Make the time range one timestep wider or narrower</td></tr>
<tr><td><b>Home</b>, <b>End</b></td><td>Move the time to the first or the last valid timestep</td></tr>
<tr><td><b>Space</b></td><td>Play or pause</td></tr>
<tr><td><b>Drag</b> across the plot</td><td>Select the x range</td></tr>
<tr><td><b>Double-click</b> the plot</td><td>Get the default x range</td></tr>
<tr><td><b>⌘S</b></td><td>Save the figure with the command</td></tr>
<tr><td><b>⇧⌘C</b></td><td>Copy the command</td></tr>
<tr><td><b>⌘O</b></td><td>Open a model in a new window</td></tr>
<tr><td><b>?</b></td><td>Show this list</td></tr>
</table>"""


def run_viewer(tokens: "Sequence[str]") -> None:
    """Open the window of the viewer, and print the command of the last plot when the window closes.

    The window is a Qt window with native controls. The Qt canvas of matplotlib draws at the pixel ratio
    of the screen, thus the plot has the full resolution of a Retina display.
    """
    import os

    import_optional("PySide6.QtWidgets")
    import matplotlib.pyplot as plt
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # Qt stops the process with no Python error when it cannot find a display
    if sys.platform == "linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        exit_with_error(
            "--interactive needs a window, and this computer has no display",
            "Run the command on a computer with a display, e.g. with ssh -X",
        )

    # the Save command runs plotspectra, which makes a pyplot figure. A pyplot window must not open beside the viewer
    plt.switch_backend("agg")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    app.setApplicationName("artistools")
    app.setApplicationDisplayName("artistools")
    app.setWindowIcon(QtGui.QIcon(make_icon_pixmap(512)))

    # each window holds a reference here, thus Python keeps it while it is open
    windows: list[QtWidgets.QMainWindow] = []
    open_window(tokens, windows)
    app.exec()


def open_window(tokens: "Sequence[str]", windows: "list[t.Any]") -> None:
    """Open a window of the viewer for the plotspectra arguments in tokens."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    from artistools.commands import get_path

    viewer = SpectrumViewer(tokens, mplfig.Figure())
    window = QtWidgets.QMainWindow()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    window.setWindowTitle(f"plotspectra {' '.join(Path(path).name for path in viewer.modelpathtokens)}")
    canvas = FigureCanvasQTAgg(viewer.fig)
    if viewer.draw(quiet=False) is not None:
        # the arguments of the user give the error, and the terminal shows it
        if not windows:
            raise SystemExit(1)
        return
    windows.append(window)

    class PlotArea(QtWidgets.QWidget):
        """The area of the plot, which scales the figure to its size."""

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            fit_canvas()

    plotarea = PlotArea()
    plotarea.setMinimumSize(320, 240)
    plotlayout = QtWidgets.QVBoxLayout(plotarea)
    plotlayout.setContentsMargins(0, 0, 0, 0)
    plotlayout.addWidget(canvas, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

    central = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(central)
    layout.addWidget(plotarea, stretch=1)
    window.setCentralWidget(central)

    # the sections scroll, and the command stays at the bottom of the panel
    sidebar = QtWidgets.QWidget()
    sidebar.setFixedWidth(600)
    sidebarlayout = QtWidgets.QVBoxLayout(sidebar)
    sidebarlayout.setContentsMargins(0, 0, 0, 0)
    panel = QtWidgets.QWidget()
    panellayout = QtWidgets.QVBoxLayout(panel)
    panelscroll = QtWidgets.QScrollArea()
    panelscroll.setWidget(panel)
    panelscroll.setWidgetResizable(True)
    panelscroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    panelscroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sidebarlayout.addWidget(panelscroll, stretch=1)
    layout.addWidget(sidebar)

    def add_section(title: str, *, expanded: bool = True) -> tuple[QtWidgets.QToolButton, QtWidgets.QGridLayout]:
        """Add a section with a header that shows or hides its controls."""
        header = QtWidgets.QToolButton()
        header.setText(title)
        header.setCheckable(True)
        header.setChecked(expanded)
        header.setAutoRaise(True)
        # a section header is a heading and not a button, thus it has no border in either state
        header.setStyleSheet("QToolButton { border: none; }")
        header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)
        font = header.font()
        font.setBold(True)
        header.setFont(font)
        content = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(content)
        grid.setColumnStretch(1, 1)
        content.setVisible(expanded)

        def on_toggled(checked: bool) -> None:
            header.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
            content.setVisible(checked)

        header.toggled.connect(on_toggled)
        panellayout.addWidget(header)
        panellayout.addWidget(content)
        return header, grid

    def make_slider() -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        # the arrow keys move the time and change the width, thus a slider must not take them
        slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        return slider

    # each continuous slider maps its position from 0 to SLIDER_STEPS onto the range of its value
    def to_position(value: float, low: float, high: float) -> int:
        return round(SLIDER_STEPS * (min(max(value, low), high) - low) / max(high - low, 1e-300))

    def from_position(position: int, low: float, high: float) -> float:
        return low + (high - low) * position / SLIDER_STEPS

    helptexts = viewer.helptexts
    logtrange = (math.log10(viewer.timebounds[0]), math.log10(viewer.timebounds[1]))
    widthmax = max((viewer.timebounds[1] - viewer.timebounds[0]) / 4.0, viewer.values.width)
    nvalid = len(viewer.validtimesteps)

    _, timegrid = add_section("Time")
    snapbutton = QtWidgets.QRadioButton("Snap to timesteps")
    continuousbutton = QtWidgets.QRadioButton("Continuous (--notimeclamp)")
    snapbutton.setToolTip("The time range holds whole timesteps, as plotspectra reads them by default")
    continuousbutton.setToolTip(helptexts.get("notimeclamp", ""))
    modebuttons = QtWidgets.QButtonGroup(window)
    for button in (snapbutton, continuousbutton):
        modebuttons.addButton(button)
    modelayout = QtWidgets.QHBoxLayout()
    modelayout.addWidget(snapbutton)
    modelayout.addWidget(continuousbutton)
    modelayout.addStretch(1)
    timegrid.addLayout(modelayout, 0, 0, 1, 3)
    timeslider, widthslider = make_slider(), make_slider()
    timeedit, widthedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    widthlabel = QtWidgets.QLabel()
    timestepslabel = QtWidgets.QLabel()
    playbutton = QtWidgets.QPushButton("Play")
    playbutton.setCheckable(True)
    playbutton.setToolTip("Move the time through the valid timesteps of the run (Space)")
    timetip = "The middle of the time range in days. The Left key and the Right key move it to the adjacent timestep."
    widthtip = (
        'The width of the time range. The Up key and the Down key change the width by one timestep. With "Snap to'
        ' timesteps", the width is a count of timesteps.'
    )
    for row, (label, slider, edit, tip) in enumerate(
        [(QtWidgets.QLabel("Time [d]"), timeslider, timeedit, timetip), (widthlabel, widthslider, widthedit, widthtip)],
        start=1,
    ):
        edit.setFixedWidth(110)
        for widget in (slider, edit):
            widget.setToolTip(tip)
        timegrid.addWidget(label, row, 0)
        timegrid.addWidget(slider, row, 1)
        timegrid.addWidget(edit, row, 2)
    timegrid.addWidget(timestepslabel, 3, 0, 1, 2)
    timegrid.addWidget(playbutton, 3, 2)

    xheader, xgrid = add_section("")
    xminslider, xmaxslider = make_slider(), make_slider()
    for slider in (xminslider, xmaxslider):
        slider.setRange(0, SLIDER_STEPS)
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

    _, axesgrid = add_section("Axes", expanded=False)
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
    fixycheck = QtWidgets.QCheckBox("Fix the y axis")
    fixycheck.setToolTip(
        "Keep the y limits of the plot when the time or a different option changes. The command gives the limits"
        " with -ymin and -ymax."
    )
    axesgrid.addWidget(fixycheck, 3, 0, 1, 2)
    yminedit, ymaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    ylimits = QtWidgets.QHBoxLayout()
    for label, edit, dest in (
        (QtWidgets.QLabel("-ymin"), yminedit, "ymin"),
        (QtWidgets.QLabel("-ymax"), ymaxedit, "ymax"),
    ):
        edit.setFixedWidth(110)
        edit.setToolTip(helptexts.get(dest, ""))
        ylimits.addWidget(label)
        ylimits.addWidget(edit)
    ylimits.addStretch(1)
    axesgrid.addLayout(ylimits, 4, 0, 1, 2)

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
    lockbutton = QtWidgets.QPushButton("Lock series")
    lockbutton.setCheckable(True)
    lockbutton.setToolTip(
        "Keep the series of the plot and their colours when the time or the x range changes. The command gives the"
        " series with -fixedionlist."
    )
    emissiongrid.addWidget(lockbutton, 3, 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignLeft)

    _, bingrid = add_section("Bins of the packet spectrum", expanded=bool(viewer.values.deltax))
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

    _, referencegrid = add_section("Reference spectra", expanded=bool(viewer.values.references))
    referencelist = QtWidgets.QListWidget()
    referencelist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    referencelist.setFixedHeight(4 * referencelist.fontMetrics().lineSpacing() + 12)
    referencelist.setToolTip(
        "The observed spectra of the plot. The command gives a file from the reference data of artistools by its"
        " name alone."
    )
    addbutton, removebutton = QtWidgets.QPushButton("Add..."), QtWidgets.QPushButton("Remove")
    referencegrid.addWidget(referencelist, 0, 0, 1, 2)
    referencegrid.addWidget(addbutton, 1, 0)
    referencegrid.addWidget(removebutton, 1, 1, QtCore.Qt.AlignmentFlag.AlignLeft)
    referencefolder = get_path("artistools_dir") / "data" / "refspectra"
    panellayout.addStretch(1)

    commandbox = QtWidgets.QGroupBox("Command")
    commandgrid = QtWidgets.QGridLayout(commandbox)
    commandtext = QtWidgets.QPlainTextEdit()
    commandtext.setReadOnly(True)
    commandtext.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    commandtext.setFixedHeight(5 * commandtext.fontMetrics().lineSpacing() + 12)
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip("Copy the command to the clipboard (⇧⌘C)")
    commandgrid.addWidget(commandtext, 0, 0, 1, 2)
    commandgrid.addWidget(copybutton, 1, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
    sidebarlayout.addWidget(commandbox)

    # the status bar gives the messages at the left, and the readout, the time of the plot, and the help at the right
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

    signalwidgets: list[QtWidgets.QWidget] = [
        snapbutton,
        continuousbutton,
        timeslider,
        widthslider,
        xminslider,
        xmaxslider,
        xunitbox,
        yscalebox,
        logscalexcheck,
        fixycheck,
        emissioncheck,
        absorptioncheck,
        groupbybox,
        countbox,
        lockbutton,
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
        xheader.setText(xunit.kind.capitalize())
        xminlabel.setText(f"{xunit.kind.capitalize()} min [{xunit.label}]")
        xmaxlabel.setText(f"{xunit.kind.capitalize()} max [{xunit.label}]")
        deltaxcheck.setText(f"-deltax [{xunit.label}]")
        rangesunit = values.xunit

    # the time sliders have one position for each valid timestep, or SLIDER_STEPS positions for a continuous time
    slidermode: bool | None = None

    def set_time_mode() -> None:
        nonlocal slidermode
        continuous = viewer.values.notimeclamp
        timeslider.setRange(0, SLIDER_STEPS if continuous else nvalid - 1)
        if continuous:
            widthslider.setRange(0, SLIDER_STEPS)
        else:
            widthslider.setRange(1, nvalid)
        widthlabel.setText("Δt [d]" if continuous else "Timesteps")
        slidermode = continuous

    # the test of each choice parses the arguments again, thus the code keeps one result for each set of options
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

    def fit_canvas() -> None:
        """Scale the figure to the plot area, and keep the shape of the frames.

        An embedded canvas has no figure manager, thus the figure cannot set the size of the widget. The resolution
        of the figure changes, thus the frames keep their size in inches and the whole figure fits in the area.
        """
        figwidth, figheight = viewer.figsize
        if figwidth <= 0.0 or figheight <= 0.0:
            return
        area = plotarea.contentsRect()
        logicaldpi = max(min(area.width() / figwidth, area.height() / figheight), 20.0)
        size = QtCore.QSize(math.ceil(figwidth * logicaldpi), math.ceil(figheight * logicaldpi))
        dpi = logicaldpi * canvas.device_pixel_ratio
        if canvas.size() == size and math.isclose(viewer.fig.dpi, dpi, rel_tol=1e-6):
            return
        viewer.fig.set_dpi(dpi)
        canvas.setFixedSize(size)
        canvas.draw_idle()

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        blockers = [QtCore.QSignalBlocker(widget) for widget in signalwidgets]
        values = viewer.values
        if values.xunit != rangesunit:
            set_xunit_ranges()
        if values.notimeclamp != slidermode:
            set_time_mode()
        (continuousbutton if values.notimeclamp else snapbutton).setChecked(True)
        if values.notimeclamp:
            timeslider.setValue(to_position(math.log10(max(values.centre, viewer.timebounds[0])), *logtrange))
            widthslider.setValue(to_position(values.width, 0.0, widthmax))
            widthedit.setText(f"{values.width:g}")
        else:
            first, last = (viewer.validtimesteps.index(timestep) for timestep in viewer.get_selection(values))
            timeslider.setValue((first + last) // 2)
            widthslider.setValue(last - first + 1)
            widthedit.setText(str(last - first + 1))
        timeedit.setText(f"{values.centre:.4g}")
        timestepslabel.setText(viewer.get_timesteps_text())
        xminslider.setValue(to_position(math.log10(max(float(values.xmin), 10.0 ** logxrange[0])), *logxrange))
        xmaxslider.setValue(to_position(math.log10(max(float(values.xmax), 10.0 ** logxrange[0])), *logxrange))
        xminedit.setText(values.xmin)
        xmaxedit.setText(values.xmax)
        xunitbox.setCurrentText(values.xunit)
        yscalebox.setCurrentText(values.yscale)
        logscalexcheck.setChecked(values.logscalex)
        isyfixed = bool(values.ymin or values.ymax)
        fixycheck.setChecked(isyfixed)
        yminedit.setText(values.ymin)
        ymaxedit.setText(values.ymax)
        for edit in (yminedit, ymaxedit):
            edit.setEnabled(isyfixed)
        emissioncheck.setChecked(values.showemission)
        absorptioncheck.setChecked(values.showabsorption)
        groupbybox.setCurrentText(values.groupby or "none")
        countbox.setValue(values.maxseriescount)
        # -maxseriescount applies only to an emission or absorption plot, and a disabled box keeps its place
        for widget in (countlabel, countbox, lockbutton):
            widget.setEnabled(values.showemission or values.showabsorption)
        lockbutton.setChecked(bool(values.fixedionlist))
        lockbutton.setText(f"Lock series ({len(values.fixedionlist)})" if values.fixedionlist else "Lock series")
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
        # a plot of an emission file or of the packets is slow, thus the cursor and the status bar show the wait
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        drawtimelabel.setText("Plot in progress...")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        starttime = time.perf_counter()
        try:
            message = viewer.change(values)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        drawtimelabel.setText(f"Plot time: {time.perf_counter() - starttime:.2f} s")
        # clear the readout, because it holds the values of the old plot until the mouse moves again
        readoutlabel.setText("")
        messagelabel.setText(message or "")
        show_values()
        # a rejection occurs again at each step, thus a rejection stops the Play button
        if message is not None:
            playbutton.setChecked(False)
        elif playbutton.isChecked():
            QtCore.QTimer.singleShot(PLAY_MILLISECONDS, play_step)

    def show_error(message: str) -> None:
        messagelabel.setText(message)
        show_values()

    def on_time_mode() -> None:
        values = viewer.values
        if continuousbutton.isChecked() and not values.notimeclamp:
            apply(
                dc.replace(
                    values, notimeclamp=True, centre=float(f"{values.centre:.4g}"), width=float(f"{values.width:.3g}")
                )
            )
        elif snapbutton.isChecked() and values.notimeclamp:
            apply(viewer.snap(values, *viewer.get_selection(values)))

    def on_time(position: int) -> None:
        values = viewer.values
        if values.notimeclamp:
            centre = 10.0 ** from_position(position, *logtrange)
            apply(dc.replace(values, centre=float(f"{centre:.4g}")))
            return
        first, last = (viewer.validtimesteps.index(timestep) for timestep in viewer.get_selection(values))
        count = last - first + 1
        start = min(max(position - (count - 1) // 2, 0), nvalid - count)
        apply(viewer.snap(values, viewer.validtimesteps[start], viewer.validtimesteps[start + count - 1]))

    def on_width(position: int) -> None:
        values = viewer.values
        if values.notimeclamp:
            width = from_position(position, 0.0, widthmax)
            apply(dc.replace(values, width=float(f"{width:.3g}")))
            return
        first = viewer.validtimesteps.index(viewer.get_selection(values)[0])
        last = min(first + position - 1, nvalid - 1)
        apply(viewer.snap(values, viewer.validtimesteps[first], viewer.validtimesteps[last]))

    def on_timeedit() -> None:
        values = viewer.values
        try:
            centre, width = float(timeedit.text()), float(widthedit.text())
        except ValueError:
            show_error("Give a number of days for the time, and a number for the width")
            return
        low, high = viewer.timebounds
        if not low <= centre <= high or width < 0.0:
            show_error(f"Give a time from {low:.4g} to {high:.4g} d, and a width of 0 or more")
            return
        if values.notimeclamp:
            newvalues = dc.replace(values, centre=float(f"{centre:.4g}"), width=float(f"{width:.3g}"))
        else:
            # a snapped time takes the valid timestep that holds the typed time, and the typed count of timesteps.
            # A time between the valid timesteps takes the valid timestep with the nearest middle
            count = max(1, round(width))
            holding = [
                position
                for position, timestep in enumerate(viewer.validtimesteps)
                if viewer.tstarts[timestep] <= centre < viewer.tends[timestep]
            ]
            middle = (
                holding[0]
                if holding
                else min(
                    range(nvalid), key=lambda position: abs(viewer.tmids[viewer.validtimesteps[position]] - centre)
                )
            )
            start = min(max(middle - (count - 1) // 2, 0), max(nvalid - count, 0))
            last = min(start + count - 1, nvalid - 1)
            newvalues = viewer.snap(values, viewer.validtimesteps[start], viewer.validtimesteps[last])
        if newvalues != values:
            apply(newvalues)

    def on_arrow(step: int) -> None:
        if (values := viewer.step_time(step)) is not None:
            apply(values)

    def on_widthstep(step: int) -> None:
        apply(viewer.step_width(step))

    def play_step() -> None:
        if not playbutton.isChecked():
            return
        if (values := viewer.step_time(1)) is None:
            playbutton.setChecked(False)
            return
        apply(values)

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

    def on_fixy(checked: bool) -> None:
        if not checked:
            apply(dc.replace(viewer.values, ymin="", ymax=""))
            return
        # the limits of the plot on the screen become the limits of the command, thus the plot does not change
        low, high = (format(float(f"{limit:.3g}"), ".10g") for limit in viewer.axes[0].get_ylim())
        apply(dc.replace(viewer.values, ymin=low, ymax=high))

    def on_yedit() -> None:
        try:
            low, high = float(yminedit.text()), float(ymaxedit.text())
        except ValueError:
            show_error("Give two numbers for -ymin and -ymax")
            return
        if not low < high:
            show_error("Give a -ymin that is less than -ymax")
            return
        values = dc.replace(viewer.values, ymin=format(low, ".10g"), ymax=format(high, ".10g"))
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
        # the labels of a locked list belong to one -groupby, thus a new -groupby removes the lock
        fixedionlist = viewer.values.fixedionlist if groupby == viewer.values.groupby else ()
        values = dc.replace(
            viewer.values,
            showemission=showemission,
            showabsorption=absorptioncheck.isChecked(),
            groupby=groupby,
            maxseriescount=countbox.value(),
            deltax=format(deltaxbox.value(), ".10g") if deltaxcheck.isChecked() else "",
            fixedionlist=fixedionlist,
        )
        if values != viewer.values:
            apply(values)

    def on_lock(checked: bool) -> None:
        if not checked:
            apply(dc.replace(viewer.values, fixedionlist=()))
            return
        if not (series := viewer.get_drawn_series()):
            show_error("The plot has no series of contributions to lock")
            return
        apply(dc.replace(viewer.values, fixedionlist=series))

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
        messagelabel.setText("Copied the command")

    def on_save() -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            window, "Save the figure", str(Path.cwd() / "plotspectra.pdf"), "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)"
        )
        if not filename:
            return
        from artistools.spectra.plotspectra import main as plotspectra_main

        # the figure comes from the command, thus the file is the same as the output of the command
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                plotspectra_main(argsraw=[*viewer.get_plot_tokens(), "-o", filename])
        except (SystemExit, FileNotFoundError, ValueError) as exc:
            messagelabel.setText(f"plotspectra did not save the figure: {get_first_line(str(exc))}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        print(f"{viewer.get_command()} -o {shlex.quote(filename)}")
        messagelabel.setText(f"Saved {filename}")

    def on_open_model() -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Open the folder of an ARTIS run", str(Path.cwd()))
        if folder:
            open_window([folder, *viewer.othertokens], windows)

    def on_help() -> None:
        QtWidgets.QMessageBox.information(window, "Keys and mouse actions", KEYBOARD_HELP)

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
        frame = get_frame(event)
        readoutlabel.setText(viewer.get_readout(event.xdata) if frame is not None and event.xdata is not None else "")
        if dragstart is None or dragspan is None or event.xdata is None or frame is None:
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

    def on_closed() -> None:
        print(viewer.get_command())

    menubar = window.menuBar()
    filemenu = menubar.addMenu("File")
    helpmenu = menubar.addMenu("Help")
    for menu, text, keys, callback in (
        (filemenu, "Open Model...", QtGui.QKeySequence.StandardKey.Open, on_open_model),
        (filemenu, "Save Figure...", QtGui.QKeySequence.StandardKey.Save, on_save),
        (filemenu, "Copy Command", QtGui.QKeySequence("Ctrl+Shift+C"), on_copy),
        (filemenu, "Close Window", QtGui.QKeySequence.StandardKey.Close, window.close),
        (helpmenu, "Keys and Mouse Actions", QtGui.QKeySequence("?"), on_help),
    ):
        action = menu.addAction(text)
        action.setShortcut(keys)
        action.triggered.connect(callback)

    modebuttons.buttonToggled.connect(on_time_mode)
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
    fixycheck.toggled.connect(on_fixy)
    yminedit.editingFinished.connect(on_yedit)
    ymaxedit.editingFinished.connect(on_yedit)
    emissioncheck.toggled.connect(on_emission_options)
    absorptioncheck.toggled.connect(on_emission_options)
    groupbybox.currentTextChanged.connect(on_emission_options)
    countbox.valueChanged.connect(on_emission_options)
    lockbutton.toggled.connect(on_lock)
    deltaxcheck.toggled.connect(on_emission_options)
    deltaxbox.valueChanged.connect(on_emission_options)
    addbutton.clicked.connect(on_add_reference)
    removebutton.clicked.connect(on_remove_reference)
    copybutton.clicked.connect(on_copy)
    helpbutton.clicked.connect(on_help)
    window.destroyed.connect(on_closed)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("motion_notify_event", on_motion)
    canvas.mpl_connect("button_release_event", on_release)
    # a text field takes these keys while it has the focus, and the shortcuts apply otherwise
    for key, callback in (
        (QtCore.Qt.Key.Key_Left, lambda: on_arrow(-1)),
        (QtCore.Qt.Key.Key_Right, lambda: on_arrow(1)),
        (QtCore.Qt.Key.Key_Up, lambda: on_widthstep(1)),
        (QtCore.Qt.Key.Key_Down, lambda: on_widthstep(-1)),
        (QtCore.Qt.Key.Key_Home, lambda: apply(viewer.move_to_end(last=False))),
        (QtCore.Qt.Key.Key_End, lambda: apply(viewer.move_to_end(last=True))),
        (QtCore.Qt.Key.Key_Space, playbutton.toggle),
    ):
        QtGui.QShortcut(QtGui.QKeySequence(key), window).activated.connect(callback)

    # the first size gives the plot 100 dpi, inside the screen
    screen = window.screen().availableGeometry()
    figwidth, figheight = viewer.figsize
    plotwidth = min(round(figwidth * 100) + 24, screen.width() - sidebar.width() - 80)
    plotheight = min(round(figheight * 100) + 24, screen.height() - 100)
    window.resize(plotwidth + sidebar.width() + 40, max(plotheight, 700))
    window.show()
    show_values()
