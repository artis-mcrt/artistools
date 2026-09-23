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
from artistools.misc import get_dirbin_definitions
from artistools.misc import get_dirbins
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_nprocs
from artistools.misc import get_time_range
from artistools.misc import get_timestep_times
from artistools.misc import import_optional
from artistools.misc import parse_cli_args
from artistools.misc import print_error
from artistools.misc import separate_trailing_folders
from artistools.packets.core import RANKS_PER_BATCH
from artistools.plottools import ExponentLabelFormatter
from artistools.plottools import LABELWIDTH_INCHES
from artistools.plottools import plain_label
from artistools.plottools import RIGHTMARGIN_INCHES
from artistools.spectra.core import convert_angstroms_to_unit
from artistools.spectra.core import convert_unit_to_angstroms
from artistools.spectra.core import get_xunit
from artistools.spectra.core import XUNITS
from artistools.spectra.plotspectra import addargs
from artistools.spectra.plotspectra import DEFAULT_MAXSERIESCOUNT
from artistools.spectra.plotspectra import draw_plot
from artistools.spectra.plotspectra import find_reference_spectrum_file_or_none
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
    "logscaley",
    "ymin",
    "ymax",
    "showemission",
    "showabsorption",
    "emissionabsorption",
    "frompackets",
    "groupby",
    "maxseriescount",
    "nostack",
    "deltax",
    "deltalogx",
    "yvariable",
    "normalised",
    "hidenetspectrum",
    "hideother",
    "use_thermalemissiontype",
    "plotviewingangle",
    "plotvspecpol",
    "average_over_phi_angle",
    "average_over_theta_angle",
    "average_every_tenth_viewing_angle",
    "fixedionlist",
    "figwidthscale",
    "interactive",
})

APPLICATION_NAME: t.Final = "artistools plotspectra"

# the number of decimals of a time in days in the command
DAYS_DECIMALS: t.Final = 6

# these options give a different action from one plot of spectra, thus the table of the window does not offer them
TABLE_EXCLUDED_DESTS: t.Final = frozenset({
    "help",
    "timedayslist",
    "multispecplot",
    "makevspecpol",
    "averagevspecpolfiles",
    "output_spectra",
})

# each option of the table with its values, in the order of the command
type OptionRows = tuple[tuple[str, tuple[str, ...]], ...]

# the process that runs from the application bundle of the viewer has this environment variable
MACOS_BUNDLE_VARIABLE: t.Final = "ARTISTOOLS_VIEWER_IN_BUNDLE"

# the limits of the -figwidthscale that the viewer gives the plot to fill the plot area
MIN_FIGWIDTHSCALE: t.Final[float] = 0.3
MAX_FIGWIDTHSCALE: t.Final[float] = 4.0

# the time after the last resize of the window, before the plot takes the new shape
FIT_MILLISECONDS: t.Final[int] = 200

# plotspectra raises these errors for a bad argument or input file. The dispatcher of the CLI reports the same errors
USER_ERRORS: t.Final = (AssertionError, FileNotFoundError, ModuleNotFoundError, PermissionError, ValueError)

REJECTED_MESSAGE: t.Final = "plotspectra cannot draw this plot. The terminal shows the error"

# each slider of the window has this number of positions
SLIDER_STEPS: t.Final = 1000

# the Play button waits for this time after each plot, thus the user can see each timestep
PLAY_MILLISECONDS: t.Final = 150

# the time after the last change of a control, before the full plot replaces the preview
FULL_DRAW_MILLISECONDS: t.Final = 250


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
    nostack: bool
    deltax: str
    deltalogx: str
    frompackets: bool
    yvariable: str
    normalised: bool
    hidenetspectrum: bool
    hideother: bool
    usethermalemissiontype: bool
    # the kind of viewing direction:
    # - "" for all directions;
    # - "bin" for -plotviewingangle;
    # - "phi" and "theta" for the averages;
    # - "vpkt" for -plotvspecpol.
    directionkind: str
    directionbins: tuple[int, ...]
    fixedionlist: tuple[str, ...]
    references: tuple[str, ...]
    figwidthscale: float
    otheroptions: OptionRows


def get_command_tokens(
    argsraw: "Sequence[str] | None",
    kwargs: "Mapping[str, t.Any]",
    *,
    fromdispatcher: bool,
    dispatcherargsraw: "Sequence[str] | None",
) -> list[str]:
    """Return the arguments that the user gave to plotspectra, without the name of the command.

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


def is_flag(argstring: str) -> bool:
    """Return True if the argument is a flag, and not a value such as a negative number."""
    return argstring.startswith("-") and not re.match(r"-\.?\d", argstring)


def remove_options(parser: SuggestingArgumentParser, tokens: "Sequence[str]", dests: "Collection[str]") -> list[str]:
    """Return the tokens without the options of these dests and without the values of those options."""
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
        while maxvalues and index < len(argstrings) and not is_flag(argstrings[index]):
            index += 1
            maxvalues -= 1

    return kept


def get_table_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    """Return the options that the table of the window offers, which are the options that no other control sets."""
    return [
        action
        for action in parser._actions  # ruff:ignore[private-member-access]
        if action.option_strings
        and action.help != argparse.SUPPRESS
        and action.dest not in CONTROLLED_DESTS | TABLE_EXCLUDED_DESTS
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


def split_option_rows(parser: SuggestingArgumentParser, tokens: "Sequence[str]") -> tuple[OptionRows, list[str]]:
    """Return each option of the tokens with its values, and the tokens that are not part of an option.

    Each row gives the first flag of the option, thus an alias, e.g. -dx, becomes the full flag, e.g. -deltax.
    """
    rows: list[tuple[str, tuple[str, ...]]] = []
    othertokens: list[str] = []
    argstrings = parser.split_joined_flags(tokens)
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


def format_days(value: float) -> str:
    """Return a time in days in fixed-point notation. The option -timedays reads the "-" of 1e-05 as a range."""
    return np.format_float_positional(value, precision=DAYS_DECIMALS, trim="-")


def format_days_inside(value: float, bounds: tuple[float, float]) -> str:
    """Return a time in days in fixed-point notation, with a value inside the bounds.

    format_days rounds to the nearest decimal, and a time near a bound can then go outside it. plotspectra rejects
    such a time, thus the nearest decimal inside the bound replaces it.
    """
    scale = 10.0**DAYS_DECIMALS
    text = format_days(min(max(value, bounds[0]), bounds[1]))
    if float(text) < bounds[0]:
        return format_days(math.ceil(bounds[0] * scale) / scale)
    if float(text) > bounds[1]:
        return format_days(math.floor(bounds[1] * scale) / scale)
    return text


def get_timedays_argument(centre: float, width: float, bounds: tuple[float, float]) -> str:
    """Return the -timedays value of a continuous time range of this width around the middle time.

    A width of zero gives the middle time alone, and plotspectra then reads the timestep that holds it. The range
    stays inside the bounds, which are the valid times of the first run.
    """
    if width > 0.0:
        lowtext = format_days_inside(centre - width / 2.0, bounds)
        hightext = format_days_inside(centre + width / 2.0, bounds)
        if float(lowtext) < float(hightext):
            return f"{lowtext}-{hightext}"

    return format_days_inside(centre, bounds)


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
        # a single time selects the timestep that holds it. get_timestep_of_timedays ends a timestep at the start of
        # the next one, because the end time of a timestep can be a little after that start
        end = tstarts[first + 1] if first + 1 < len(tstarts) else tends[first]
        for decimals in range(8):
            text = f"{round(tmids[first], decimals):.{decimals}f}"
            if tstarts[first] <= float(text) < end:
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


def clear_axes_keep_ticks(axis: "mplax.Axes") -> None:
    """Clear the axes, and keep the tick objects for the next plot.

    When cla clears the axes, it removes each tick. matplotlib then makes each tick again when it draws the plot. The
    plot code sets the properties of the ticks again, thus matplotlib draws the old ticks the same as new ticks.
    """
    ticklists = [(xyaxis, xyaxis.majorTicks, xyaxis.minorTicks) for xyaxis in (axis.xaxis, axis.yaxis)]
    axis.cla()
    for xyaxis, majorticks, minorticks in ticklists:
        # the tick lists are lazy descriptors that matplotlib replaces with a list in the instance dict
        vars(xyaxis).update(majorTicks=majorticks, minorTicks=minorticks)
        # cla makes a new patch for the axes, and the grid lines of the ticks must clip to the new patch
        xyaxis.set_clip_path(axis.patch)


def fix_title_position(axis: "mplax.Axes") -> None:
    """Keep the title at the top of the frame.

    matplotlib moves the title above the offset text of the y axis. For this, it measures the y axis each time that it
    draws the plot. ExponentLabelFormatter puts the offset in the label of the axis, thus the offset text is empty and
    the title stays at the top of the frame.
    """
    if axis.get_title() and isinstance(axis.yaxis.get_major_formatter(), ExponentLabelFormatter):
        axis.set_title(axis.get_title(), y=1.0)


def get_first_line(errortext: str) -> str:
    """Return the line of an error for the status line of the window, without the "error: " of print_error.

    argparse prints its usage line before the error, and a warning can come before an error. Thus the function
    returns the line that starts with "error: ". If no line has that start, it returns the first line.
    """
    lines = [line.strip() for line in errortext.splitlines() if line.strip()]
    errorline = next((line for line in lines if line.startswith("error: ")), lines[0] if lines else None)
    return errorline.removeprefix("error: ") if errorline is not None else REJECTED_MESSAGE


def convert_xunit(values: ControlValues, xunit: str, *, gamma: bool) -> ControlValues:
    """Return the values with the x limits in a new unit of the x axis.

    A frequency or an energy increases where the wavelength decreases, thus the function sorts the limits again. A bin
    width has no linear conversion between a wavelength and a frequency, thus the new unit takes no -deltax.

    A minimum of 0 has no conversion between a wavelength and a frequency, because the range then has no end on that
    side. The new range thus goes past the other limit to the default limit of the new unit, or further. A change
    between two units of wavelength, or between two units of frequency or energy, keeps a minimum of 0.
    """

    def convert(limit: float) -> float:
        return convert_angstroms_to_unit(convert_unit_to_angstroms(limit, values.xunit), xunit)

    xminold, xmaxold = float(values.xmin), float(values.xmax)
    if xminold > 0.0:
        limits = sorted((convert(xminold), convert(xmaxold)))
    elif convert(2.0) < convert(1.0):
        otherlimit = convert(xmaxold)
        limits = [otherlimit, max(get_default_xlimits(xunit, gamma=gamma)[1], 2.0 * otherlimit)]
    else:
        limits = [0.0, convert(xmaxold)]
    xmin, xmax = (format(float(f"{limit:.4g}"), ".10g") for limit in limits)
    return dc.replace(values, xunit=xunit, xmin=xmin, xmax=xmax, deltax="")


def get_nearest_range_start(tmids: "Sequence[float]", centre: float, count: int) -> int:
    """Return the first index of the range of count timesteps that has its centre nearest to the given time.

    The time field shows the centre of the range, thus a Return with no edit gives the same range. The centre of a
    range of an even count is near the boundary of two timesteps. Thus either timestep can hold the time.
    """

    def get_centre_offset(start: int) -> float:
        return abs((tmids[start] + tmids[start + count - 1]) / 2.0 - centre)

    return min(range(len(tmids) - count + 1), key=get_centre_offset)


def get_reference_token(filename: str) -> str:
    """Return the name of a reference file if plotspectra finds that same file by the name, and the path if not.

    plotspectra searches the working folder before the reference data of artistools. Thus a file of the same name in
    the working folder takes the place of a file from the reference data. The name alone gives a short command.
    """
    found = find_reference_spectrum_file_or_none(Path(filename).name)
    return Path(filename).name if found is not None and found.resolve() == Path(filename).resolve() else filename


def get_direction_kind(args: argparse.Namespace) -> str:
    """Return the kind of viewing direction of the arguments, in the form of ControlValues.directionkind."""
    if args.plotvspecpol:
        return "vpkt"
    if not args.plotviewingangle:
        return ""
    if args.average_over_phi_angle:
        return "phi"
    return "theta" if args.average_over_theta_angle else "bin"


def get_direction_choices(runfolder: Path, directionkind: str) -> list[tuple[int, str]]:
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
    )
    return list(labels.items())


def remove_series_lock(values: ControlValues) -> ControlValues:
    """Return the values with no -fixedionlist.

    plotspectra gives a missing -maxseriescount the length of -fixedionlist. A count equal to that length thus came
    from the list, and without the list the count is the default count.
    """
    fromlist = bool(values.fixedionlist) and values.maxseriescount == len(values.fixedionlist)
    return dc.replace(
        values, fixedionlist=(), maxseriescount=DEFAULT_MAXSERIESCOUNT if fromlist else values.maxseriescount
    )


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
        usertokens = remove_options(parser, tokens, {"interactive"})
        # parse_cli_args puts "--" in front of the ARTIS folders at the end, thus an option that reads a list, e.g.
        # -fixedionlist, does not take a folder. The removal of the controlled options must keep those folders
        basetokens = remove_options(parser, separate_trailing_folders(usertokens), CONTROLLED_DESTS)
        args = parse_cli_args(addargs, None, None, usertokens)
        # resolve_frompackets gives an emission plot a default -groupby, thus the value comes from the arguments
        givengroupby: str | None = args.groupby
        # -deltax also makes plotspectra read the packets, thus the box shows only a --frompackets that the user gave
        givesfrompackets = bool(args.frompackets)
        # with --notimeclamp, a range of days keeps its bounds, and a single time or a timestep reads a whole timestep
        givesdaysrange = args.timemin is not None or (args.timedays is not None and "-" in args.timedays)
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

        # plotspectra rejects a time outside the arrival times of the escaped packets of each run. Thus the controls
        # stay inside the times that are valid for all the runs. With --plotinvalidpart, plotspectra accepts all times
        timebounds = [self.tstarts[0], self.tends[-1]]
        if not args.plotinvalidpart:
            for runfolder in self.runfolders:
                with contextlib.suppress(FileNotFoundError):
                    _, validstart, validend = get_escaped_arrivalrange(runfolder)
                    if validstart is not None:
                        timebounds[0] = max(timebounds[0], float(validstart))
                    if validend is not None:
                        timebounds[1] = min(timebounds[1], float(validend))
        self.timebounds = (timebounds[0], timebounds[1])
        self.validtimesteps = [
            timestep
            for timestep in range(len(self.tmids))
            if self.tstarts[timestep] >= self.timebounds[0] and self.tends[timestep] <= self.timebounds[1]
        ] or list(range(len(self.tmids)))

        # the table of the window shows each option that no other control sets. The tokens that no option takes are
        # paths, e.g. the path of "--notitle mymodel", or the paths after "--"
        pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
        otheroptions, positionaltokens = split_option_rows(parser, basetokens[pathcount:])
        # the order of the paths gives the -label and the style of each series, thus the paths keep their order
        self.startpaths = [*basetokens[:pathcount], *(word for word in positionaltokens if word != "--")]
        self.modelpathtokens = [path for path in self.startpaths if not path_is_reference_spectrum(path)]
        self.tableflags = [action.option_strings[0] for action in get_table_actions(parser)]
        self.actionsbyflag = {
            action.option_strings[0]: action
            for action in parser._actions  # ruff:ignore[private-member-access]
            if action.option_strings
        }

        # a range of one timestep is a single time, and a plot with no time starts in the middle of the run
        if args.timemin is not None and args.timemax is not None:
            centre = (args.timemin + args.timemax) / 2.0
            coversseveral = sum(args.timemin <= tmid <= args.timemax for tmid in self.tmids) > 1
            width = args.timemax - args.timemin if coversseveral or (args.notimeclamp and givesdaysrange) else 0.0
        else:
            # 4 significant digits of the middle timestep give a short command
            centre, width = float(f"{self.tmids[self.validtimesteps[len(self.validtimesteps) // 2]]:.4g}"), 0.0

        actions = {action.dest: action for action in parser._actions}  # ruff:ignore[private-member-access]
        self.groupbychoices = [str(choice) for choice in actions["groupby"].choices or ()]
        self.yvariablechoices = [str(choice) for choice in actions["yvariable"].choices or ()]
        self.yscalechoices = [str(choice) for choice in actions["yscale"].choices or () if choice != "lin"]
        # a tooltip gives the help text of the option, thus the window and the command line agree
        self.helptexts = {
            dest: str(action.help).replace("%(default)s", str(action.default)).replace("%%", "%")
            for dest, action in actions.items()
            if action.help and action.help != argparse.SUPPRESS
        }
        self.defaultyscale: str = parser.get_default("defaultyscale")
        self.defaultxunit = "kev" if args.gamma else "angstroms"
        self.defaultgroupby = "nuc" if args.gamma else "ion"
        self.defaultyvariable: str = parser.get_default("yvariable")
        # a run with a configuration of virtual packets has observers for -plotvspecpol
        self.directionkinds = ["", "bin", "phi", "theta"]
        if (self.runfolders[0] / "vpkt.txt").is_file():
            self.directionkinds.append("vpkt")
        # the time of the command stays exact, because a rounded time can select a different timestep
        values = ControlValues(
            centre=centre,
            width=width,
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
            # None is the default grouping, thus a -groupby that names the default gives None as well
            groupby=None if givengroupby == self.defaultgroupby else givengroupby,
            maxseriescount=args.maxseriescount,
            nostack=bool(args.nostack),
            deltax="" if args.deltax is None else format(args.deltax, ".10g"),
            deltalogx="" if args.deltalogx is None else format(args.deltalogx, ".10g"),
            frompackets=givesfrompackets,
            yvariable=args.yvariable,
            normalised=bool(args.normalised),
            hidenetspectrum=bool(args.hidenetspectrum),
            hideother=bool(args.hideother),
            usethermalemissiontype=bool(args.use_thermalemissiontype),
            directionkind=get_direction_kind(args),
            directionbins=tuple(args.plotvspecpol or args.plotviewingangle or ()),
            fixedionlist=tuple(args.fixedionlist or ()),
            references=tuple(path for path in self.startpaths if path_is_reference_spectrum(path)),
            figwidthscale=args.figwidthscale,
            otheroptions=otheroptions,
        )
        self.values = self.clamp_time(values) if values.notimeclamp else self.snap(values, *self.get_selection(values))

        self.fig = fig
        self.axes: npt.NDArray[t.Any] = np.empty(0, dtype=object)
        self.residualaxis: mplax.Axes | None = None
        # the options in frameskey change the layout, the size, or the tick parameters of the frames
        self.frameskey: tuple[bool, bool, float, float, bool, bool, float | None] | None = None
        # a preview reads the packets of the first batch of ranks only. A run with one batch has no faster preview
        self.previewmaxpacketfiles = (
            RANKS_PER_BATCH if any(get_nprocs(runfolder) > RANKS_PER_BATCH for runfolder in self.runfolders) else None
        )
        self.drewpreview = False
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
        if values.yscale != self.defaultyscale:
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
        # the count applies only to an emission or absorption plot, thus the command of a different plot leaves it
        # out. plotspectra gives a missing count the length of -fixedionlist, or DEFAULT_MAXSERIESCOUNT without a list
        defaultcount = len(values.fixedionlist) if values.fixedionlist else DEFAULT_MAXSERIESCOUNT
        if (values.showemission or values.showabsorption) and values.maxseriescount != defaultcount:
            options += ["-maxseriescount", str(values.maxseriescount)]
        if (values.showemission or values.showabsorption) and values.nostack:
            options.append("--nostack")
        if values.deltax:
            options += ["-deltax", values.deltax]
        if values.deltalogx:
            options += ["-deltalogx", values.deltalogx]
        if values.frompackets:
            options.append("--frompackets")
        if values.yvariable != self.defaultyvariable:
            options += ["-yvariable", values.yvariable]
        for isgiven, flag in (
            (values.normalised, "--normalised"),
            (values.hidenetspectrum, "--hidenetspectrum"),
            (values.hideother, "--hideother"),
            (values.usethermalemissiontype, "--use_thermalemissiontype"),
            (values.directionkind == "phi", "--average_over_phi_angle"),
            (values.directionkind == "theta", "--average_over_theta_angle"),
        ):
            if isgiven:
                options.append(flag)
        if values.directionkind:
            directionflag = "-plotvspecpol" if values.directionkind == "vpkt" else "-plotviewingangle"
            options += [directionflag, *(str(dirbin) for dirbin in values.directionbins)]
        if values.figwidthscale != 1.0:
            options += ["-figwidthscale", format(values.figwidthscale, "g")]
        # a list option takes each word that follows it, thus it comes after every other option
        if values.fixedionlist and (values.showemission or values.showabsorption):
            options += ["-fixedionlist", *values.fixedionlist]
        # a reference that the user added goes after the paths of the command line
        paths = [
            *(path for path in self.startpaths if path in self.modelpathtokens or path in values.references),
            *(path for path in values.references if path not in self.startpaths),
        ]
        othertokens = [token for flag, optionvalues in values.otheroptions for token in (flag, *optionvalues)]
        return make_command_tokens([*paths, *othertokens], options)

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotspectra", *self.get_plot_tokens()])

    def get_timesteps_text(self) -> str:
        """Return the timesteps and the days that the plot reads from spec.out, which holds complete timesteps."""
        # a path does not start with "-", and the -t of the controls comes before each other option
        plottokens = self.get_plot_tokens()
        timedays = plottokens[plottokens.index("-t") + 1]
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
                check_viewer_args(plotargs)
        except SystemExit:
            return get_first_line(errors.getvalue())
        except USER_ERRORS as exc:
            return get_first_line(str(exc))
        if (plotargs.showemission, plotargs.showabsorption) != (values.showemission, values.showabsorption):
            return "A different option of the command keeps the emission plot on"
        return None

    def draw(self, *, quiet: bool = True, preview: bool = False) -> str | None:
        """Draw the plot of the command, and return the reason for the status line if plotspectra rejects it.

        Each plot prints the same lines again, e.g. the list of the ions of an emission plot. Thus a quiet plot
        discards the standard output. The terminal shows the whole error, and the status line shows its first line.
        """
        output = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
        errors = io.StringIO()
        try:
            with output, contextlib.redirect_stderr(errors):
                return self.draw_command(preview=preview)
        except SystemExit:
            # exit_with_error printed a line that starts with "error: ", and a help line
            return get_first_line(errors.getvalue())
        except USER_ERRORS as exc:
            print_error(str(exc) or type(exc).__name__)
            return get_first_line(str(exc))
        finally:
            sys.stderr.write(errors.getvalue())

    def draw_command(self, *, preview: bool = False) -> str | None:
        """Parse the command and draw its plot, or return a message if the plot differs from the values.

        A preview of a plot of the packets reads the first batch of ranks only. For the 20 batches of a kilonova run,
        a range of 8 days took 0.12 s in place of 1.1 s. The flux stays correct, because the reader divides by the
        number of ranks that it reads. The reader does not divide a count of packets, thus a plot of
        -yvariable packetcount has no preview. The command in the window has no -maxpacketfiles for the preview.
        """
        plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens())
        resolve_plot_args(plotargs)
        check_viewer_args(plotargs)
        self.drewpreview = bool(
            preview
            and plotargs.frompackets
            and plotargs.maxpacketfiles is None
            and plotargs.yvariable != "packetcount"
            and self.previewmaxpacketfiles
        )
        if self.drewpreview:
            plotargs.maxpacketfiles = self.previewmaxpacketfiles
        shown = (plotargs.showemission, plotargs.showabsorption)
        if shown != (self.values.showemission, self.values.showabsorption):
            return "A different option of the command keeps the emission plot on"
        self.draw_frames(plotargs)
        return None

    def draw_frames(self, plotargs: argparse.Namespace) -> None:
        """Draw the plot on empty frames."""
        # cla() keeps the tick parameters, and the plot sets them only for these options. Thus a change to one of
        # them makes new frames
        frameskey = (
            plotargs.showabsorption,
            plotargs.residuals,
            plotargs.figwidthscale,
            plotargs.figscale,
            plotargs.hidexticklabels,
            plotargs.hideyticklabels,
            getattr(plotargs, "labelfontsize", None),
        )
        if frameskey != self.frameskey:
            self.fig.clear()
            _, self.axes, self.residualaxis = make_plot_figure(plotargs, fig=self.fig)
            self.frameskey = frameskey
            figwidth, figheight = self.fig.get_size_inches()
            self.figsize = (float(figwidth), float(figheight))
        else:
            for axis in (*self.axes, self.residualaxis):
                if axis is not None:
                    clear_axes_keep_ticks(axis)

        self.dfalldata, _ = draw_plot(plotargs, self.axes, self.residualaxis)
        for axis in self.axes:
            fix_title_position(axis)
        self.fig.canvas.draw_idle()

    def get_fitted_figwidthscale(self, areawidth: float, areaheight: float) -> float:
        """Return the -figwidthscale that gives the figure the shape of the plot area.

        The frame width is proportional to -figwidthscale, and the margins and the height stay the same.
        """
        figwidth, figheight = self.figsize
        margins = LABELWIDTH_INCHES + RIGHTMARGIN_INCHES
        widthperscale = (figwidth - margins) / self.values.figwidthscale
        fitted = (figheight * areawidth / areaheight - margins) / widthperscale
        # 2 decimals give a short command, and a small change of the window then keeps the frames
        return round(min(max(fitted, MIN_FIGWIDTHSCALE), MAX_FIGWIDTHSCALE), 2)

    def clamp_time(self, values: ControlValues) -> ControlValues:
        """Return the values with a continuous time that gives a plot of valid times only.

        A range stays inside the valid times. A time alone selects the whole timestep that holds it. Thus such a
        time stays between the middles of the first and the last valid timestep.
        """
        if not values.notimeclamp:
            return values
        if values.width > 0.0:
            low, high = self.timebounds
        else:
            low, high = self.tmids[self.validtimesteps[0]], self.tmids[self.validtimesteps[-1]]
        centre = min(max(values.centre, low), high)
        return values if centre == values.centre else dc.replace(values, centre=centre)

    def change(self, values: ControlValues, *, preview: bool = False) -> str | None:
        """Draw the plot of the new values, and keep the old values if plotspectra rejects the new command."""
        oldvalues, self.values = self.values, self.clamp_time(values)
        message = self.draw(preview=preview)
        if message is not None:
            self.values = oldvalues
            # the old values drew a plot before, thus they draw again
            self.draw(preview=preview)
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


def get_macos_bundle_executable() -> Path:
    """Return the Python executable in the application bundle of the viewer.

    Make the bundle if it does not exist. The bundle holds a hard link to the Python executable, thus it uses almost
    no disk space. On a different volume, it holds a copy. If the Python executable changes, this function replaces
    the link.
    """
    import os
    import plistlib
    import shutil

    baseexecutable = Path(sys.executable).resolve()
    contents = Path.home() / "Library" / "Caches" / "artistools" / f"{APPLICATION_NAME}.app" / "Contents"
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
        "CFBundleName": APPLICATION_NAME,
        "CFBundleDisplayName": APPLICATION_NAME,
        "CFBundleIdentifier": "io.github.artis-mcrt.artistools.plotspectra",
        "CFBundleExecutable": executable.name,
        "CFBundlePackageType": "APPL",
        "NSHighResolutionCapable": True,
    }
    tmpinfo = contents / f"Info.plist.{os.getpid()}.tmp"
    tmpinfo.write_bytes(plistlib.dumps(info))
    tmpinfo.replace(contents / "Info.plist")
    return executable


def relaunch_in_macos_bundle() -> None:
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
        executable = get_macos_bundle_executable()
    except OSError:
        # the viewer can open without the bundle, and the Dock then gives the name of the executable
        return

    sys.stdout.flush()
    sys.stderr.flush()
    environment = os.environ | {MACOS_BUNDLE_VARIABLE: "1", "__PYVENV_LAUNCHER__": sys.executable}
    os.execve(executable, [str(executable), *sys.orig_argv[1:]], environment)  # ruff:ignore[start-process-with-no-shell]


def run_viewer(tokens: "Sequence[str]") -> None:
    """Open the window of the viewer, and print the command of the last plot when the window closes.

    The window is a Qt window with native controls. The Qt canvas of matplotlib draws at the pixel ratio
    of the screen, thus the plot has the full resolution of a Retina display.
    """
    import os

    if sys.platform == "darwin":
        relaunch_in_macos_bundle()

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

    # the Save command runs plotspectra, which makes a pyplot figure. A pyplot window must not open beside the viewer
    plt.switch_backend("agg")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    app.setApplicationName("artistools")
    app.setApplicationDisplayName(APPLICATION_NAME)
    app.setWindowIcon(QtGui.QIcon(make_icon_pixmap(512)))

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

    # each window holds a reference here, thus Python keeps it while it is open
    windows: list[QtWidgets.QMainWindow] = []
    open_window(tokens, windows)
    app.exec()


def open_window(tokens: "Sequence[str]", windows: "list[t.Any]") -> bool:
    """Open a window of the viewer for the plotspectra arguments in tokens, and return True if it opened."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    from artistools.commands import get_path

    viewer = SpectrumViewer(tokens, mplfig.Figure())
    window = QtWidgets.QMainWindow()
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    window.setWindowTitle(f"{APPLICATION_NAME} {' '.join(Path(path).name for path in viewer.modelpathtokens)}")
    canvas = FigureCanvasQTAgg(viewer.fig)
    if viewer.draw(quiet=False) is not None:
        # the arguments of the user give the error, and the terminal shows it
        if not windows:
            raise SystemExit(1)
        return False
    windows.append(window)

    # a timer with the window as parent stops when the window closes, thus it does not act on deleted widgets
    fulldrawtimer = QtCore.QTimer(window)
    fulldrawtimer.setSingleShot(True)
    fulldrawtimer.setInterval(FULL_DRAW_MILLISECONDS)
    fittimer = QtCore.QTimer(window)
    fittimer.setSingleShot(True)
    fittimer.setInterval(FIT_MILLISECONDS)

    class PlotArea(QtWidgets.QWidget):
        """The area of the plot, which scales the figure to its size."""

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            fit_canvas()
            # a new plot takes up to 1 s, thus the plot takes the new shape only when the resize stops
            fittimer.start()

    plotarea = PlotArea()
    plotarea.setMinimumSize(320, 240)
    plotlayout = QtWidgets.QVBoxLayout(plotarea)
    plotlayout.setContentsMargins(0, 0, 0, 0)
    plotlayout.addWidget(canvas, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

    central = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(central)
    layout.addWidget(plotarea, stretch=1)
    window.setCentralWidget(central)

    # the sections and the command scroll together
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
    layout.addWidget(sidebar)

    def add_section(title: str) -> tuple[QtWidgets.QLabel, QtWidgets.QGridLayout]:
        """Add a section with a heading and a grid for its controls."""
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

    def add_row(grid: QtWidgets.QGridLayout, row: int, widgets: "Sequence[QtWidgets.QWidget]") -> None:
        """Put the widgets side by side in one row of the grid, from the left."""
        rowlayout = QtWidgets.QHBoxLayout()
        for widget in widgets:
            rowlayout.addWidget(widget)
        rowlayout.addStretch(1)
        grid.addLayout(rowlayout, row, 0, 1, -1)

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

    class RangeSlider(QtWidgets.QWidget):
        """A slider with two handles, which give the minimum and the maximum of a range.

        Qt has no slider with two handles. A drag moves the handle that is nearer to the pointer, and the minimum
        stays below the maximum. The signal gives the index of the handle that moved and its new position.
        """

        limitmoved = QtCore.Signal(int, int)
        handleradius: t.Final = 8.0

        def __init__(self) -> None:
            super().__init__()
            self.positions = [0, SLIDER_STEPS]
            self.draghandle: int | None = None
            self.setMinimumHeight(round(3 * self.handleradius))
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            # the arrow keys move the time and change the width, thus the slider must not take them
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        def set_positions(self, low: int, high: int) -> None:
            self.positions = [low, high]
            self.update()

        def get_pixel(self, position: int) -> float:
            return self.handleradius + (self.width() - 2.0 * self.handleradius) * position / SLIDER_STEPS

        def get_position(self, pixel: float) -> int:
            fraction = (pixel - self.handleradius) / max(self.width() - 2.0 * self.handleradius, 1.0)
            return round(min(max(fraction, 0.0), 1.0) * SLIDER_STEPS)

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

    xheader, xgrid = add_section("")
    xrangeslider = RangeSlider()
    xminedit, xmaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    zoomtip = " Drag across the plot to select a range. Double-click the plot to get the default range."
    xrangeslider.setToolTip("The minimum and the maximum of the x axis." + zoomtip)
    for column, (widget, dest) in enumerate(((xminedit, "xmin"), (xrangeslider, ""), (xmaxedit, "xmax"))):
        if dest:
            widget.setFixedWidth(110)
            widget.setToolTip(helptexts.get(dest, "") + zoomtip)
        xgrid.addWidget(widget, 0, column)
    xgrid.setColumnStretch(1, 1)

    _, axesgrid = add_section("Axes")
    xunitbox, yscalebox = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    xunitbox.addItems(list(XUNITS))
    yscalebox.addItems(viewer.yscalechoices)
    logscalexcheck = QtWidgets.QCheckBox("--logscalex")
    fixycheck = QtWidgets.QCheckBox("Fix the y axis")
    fixycheck.setToolTip(
        "Keep the y limits of the plot when the time or a different option changes. The command gives the limits"
        " with -ymin and -ymax."
    )
    yminedit, ymaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    for widget, dest in (
        (xunitbox, "xunit"),
        (yscalebox, "yscale"),
        (logscalexcheck, "logscalex"),
        (yminedit, "ymin"),
        (ymaxedit, "ymax"),
    ):
        widget.setToolTip(helptexts.get(dest, ""))
    for edit in (yminedit, ymaxedit):
        edit.setFixedWidth(110)
    add_row(axesgrid, 0, [QtWidgets.QLabel("-xunit"), xunitbox, QtWidgets.QLabel("-yscale"), yscalebox, logscalexcheck])
    add_row(axesgrid, 1, [fixycheck, QtWidgets.QLabel("-ymin"), yminedit, QtWidgets.QLabel("-ymax"), ymaxedit])
    yvariablebox = QtWidgets.QComboBox()
    yvariablebox.addItems(viewer.yvariablechoices)
    normalisedcheck = QtWidgets.QCheckBox("--normalised")
    for widget, dest in ((yvariablebox, "yvariable"), (normalisedcheck, "normalised")):
        widget.setToolTip(helptexts.get(dest, ""))
    add_row(axesgrid, 2, [QtWidgets.QLabel("-yvariable"), yvariablebox, normalisedcheck])

    _, emissiongrid = add_section("Emission and absorption")
    emissioncheck = QtWidgets.QCheckBox("--showemission")
    absorptioncheck = QtWidgets.QCheckBox("--showabsorption")
    groupbybox = QtWidgets.QComboBox()
    groupbybox.addItems(viewer.groupbychoices)
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
    nostackcheck = QtWidgets.QCheckBox("--nostack")
    nostackcheck.setToolTip(helptexts.get("nostack", ""))
    lockbutton = QtWidgets.QPushButton("Lock series")
    lockbutton.setCheckable(True)
    lockbutton.setToolTip(
        "Keep the series of the plot and their colours when the time or the x range changes. The command gives the"
        " series with -fixedionlist."
    )
    add_row(emissiongrid, 0, [emissioncheck, absorptioncheck, nostackcheck])
    add_row(emissiongrid, 1, [QtWidgets.QLabel("-groupby"), groupbybox, countlabel, countbox, lockbutton])
    hidenetcheck = QtWidgets.QCheckBox("--hidenetspectrum")
    hideothercheck = QtWidgets.QCheckBox("--hideother")
    thermalcheck = QtWidgets.QCheckBox("--use_thermalemissiontype")
    for widget, dest in (
        (hidenetcheck, "hidenetspectrum"),
        (hideothercheck, "hideother"),
        (thermalcheck, "use_thermalemissiontype"),
    ):
        widget.setToolTip(helptexts.get(dest, ""))
    add_row(emissiongrid, 2, [hidenetcheck, hideothercheck, thermalcheck])

    _, bingrid = add_section("Bins of the packet spectrum")
    # the "Default bins" item gives no -deltax and no -deltalogx, thus plotspectra uses its own bins
    binmodebox = QtWidgets.QComboBox()
    for binmode, binmodetext in (("", "Default bins"), ("deltax", "-deltax"), ("deltalogx", "-deltalogx")):
        binmodebox.addItem(binmodetext, binmode)
        binmodebox.setItemData(binmodebox.count() - 1, helptexts.get(binmode, ""), QtCore.Qt.ItemDataRole.ToolTipRole)

    class BinWidthSpinBox(QtWidgets.QDoubleSpinBox):
        """A box for the bin width that shows the shortest text of its value.

        The box accepts more decimals than a bin width usually has. A fixed count of decimals then shows "20.000".
        """

        @t.override
        def textFromValue(self, v: float) -> str:
            return format(v, ".10g")

    binwidthbox = BinWidthSpinBox()
    # each arrow step is one power of ten below the value, thus the arrows reach each bin width
    binwidthbox.setStepType(QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
    frompacketscheck = QtWidgets.QCheckBox("--frompackets")
    frompacketscheck.setToolTip(helptexts.get("frompackets", ""))
    add_row(bingrid, 0, [frompacketscheck, binmodebox, binwidthbox])

    _, directiongrid = add_section("Viewing direction")
    directionkindbox, directionbox = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    for directionkind, directionkindtext, dest in (
        ("", "All directions", ""),
        ("bin", "-plotviewingangle", "plotviewingangle"),
        ("phi", "--average_over_phi_angle", "average_over_phi_angle"),
        ("theta", "--average_over_theta_angle", "average_over_theta_angle"),
        ("vpkt", "-plotvspecpol", "plotvspecpol"),
    ):
        if directionkind in viewer.directionkinds:
            directionkindbox.addItem(directionkindtext, directionkind)
            directionkindbox.setItemData(
                directionkindbox.count() - 1, helptexts.get(dest, ""), QtCore.Qt.ItemDataRole.ToolTipRole
            )
    directionbox.setToolTip("The direction bin of the plot, or the observer of the virtual packets")
    add_row(directiongrid, 0, [directionkindbox])
    # the label of a direction bin is long, thus the box of the direction bins takes the full width of the sidebar
    directiongrid.addWidget(directionbox, 1, 0, 1, -1)
    # the labels of the direction bins come from the files of the run, thus the window reads them one time for each kind
    directionchoices: dict[str, list[tuple[int, str]]] = {}
    shownkind: str | None = None
    for box in (countbox, binwidthbox):
        # a typed number applies when the user presses Return or leaves the box, and not after each digit
        box.setKeyboardTracking(False)

    _, referencegrid = add_section("Reference spectra")
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
    _, optiongrid = add_section("Other options")
    optiontable = QtWidgets.QTableWidget(0, 2)
    optiontable.setHorizontalHeaderLabels(["Option", "Value"])
    optiontable.verticalHeader().setVisible(False)
    optiontable.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    optiontable.setToolTip("Give each option of plotspectra that no other control sets. Type part of a name to search.")
    optiontable.setColumnWidth(0, optiontable.fontMetrics().horizontalAdvance("-emissionlosvelocityrange") + 48)
    optiontable.horizontalHeader().setStretchLastSection(True)
    optiongrid.addWidget(optiontable, 0, 0, 1, 2)
    # a row that holds None needs a value from the user, and the command does not give it yet
    optionrows: list[tuple[str, tuple[str, ...] | None]] = list(viewer.values.otheroptions)

    def get_complete_rows() -> OptionRows:
        return tuple((flag, optionvalues) for flag, optionvalues in optionrows if optionvalues is not None)

    def on_option_rows() -> None:
        rows = get_complete_rows()
        if rows != viewer.values.otheroptions:
            apply(dc.replace(viewer.values, otheroptions=rows))

    def set_option(row: int, flag: str) -> None:
        """Put a different option in a row.

        An empty option removes the row. An option in the empty last row adds a new row.
        """
        oldflag = optionrows[row][0] if row < len(optionrows) else ""
        if flag == oldflag or (flag and flag not in viewer.actionsbyflag):
            return
        if not flag:
            del optionrows[row]
        elif row < len(optionrows):
            optionrows[row] = (flag, get_default_tokens(viewer.actionsbyflag[flag]))
        else:
            optionrows.append((flag, get_default_tokens(viewer.actionsbyflag[flag])))
        show_option_rows()
        on_option_rows()

    def set_option_values(row: int, flag: str, optionvalues: tuple[str, ...] | None) -> None:
        # a field that loses the focus when the table changes can send the values of a row that is not there now
        if row < len(optionrows) and optionrows[row][0] == flag:
            optionrows[row] = (flag, optionvalues)
            on_option_rows()

    def make_flag_box(row: int, flag: str) -> QtWidgets.QComboBox:
        """Return a list of the options that the user can search, with the option of the row."""
        box = QtWidgets.QComboBox()
        box.setEditable(True)
        box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        flags = ["", *viewer.tableflags]
        # an option that a hidden flag gave, e.g. an old spelling, stays in its row
        if flag not in flags:
            flags.append(flag)
        box.addItems(flags)
        for index, itemflag in enumerate(flags[1:], start=1):
            helptext = helptexts.get(viewer.actionsbyflag[itemflag].dest, "")
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
        action = viewer.actionsbyflag[flag]
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

    show_option_rows()
    # the command is the last section, at the bottom of the panel
    panellayout.addStretch(1)
    _, commandgrid = add_section("Command")
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
        xrangeslider,
        xunitbox,
        yscalebox,
        logscalexcheck,
        fixycheck,
        emissioncheck,
        absorptioncheck,
        groupbybox,
        countbox,
        nostackcheck,
        lockbutton,
        binmodebox,
        binwidthbox,
        frompacketscheck,
        yvariablebox,
        normalisedcheck,
        hidenetcheck,
        hideothercheck,
        thermalcheck,
        directionkindbox,
        directionbox,
    ]

    # the x slider and the step of -deltax follow the unit of the x axis, thus a new unit sets them again
    logxrange = (0.0, 1.0)
    rangesunit: str | None = None

    def set_xunit_ranges() -> None:
        nonlocal logxrange, rangesunit
        values = viewer.values
        defaultxmin, defaultxmax = get_default_xlimits(values.xunit, gamma=viewer.args.gamma)
        # the x slider acts on log10(x), thus its range must be above zero
        xlow = min(value for value in (float(values.xmin), defaultxmin) if value > 0.0) / 2.0
        xhigh = max(float(values.xmax), defaultxmax) * 2.0
        logxrange = (math.log10(xlow), math.log10(xhigh))
        # a step of approximately 1/1000 of the x range is correct for each x unit, e.g. 10 Å for wavelengths
        xspan = defaultxmax - defaultxmin
        deltaxstep = 10.0 ** math.floor(math.log10(xspan / 1000.0))
        # 3 more decimals than the default step let the user give a bin width that is not a multiple of the step
        decimals = max(0, -math.floor(math.log10(deltaxstep))) + 3
        binwidthranges["deltax"] = (decimals, 10.0**-decimals, xspan, 2.0 * deltaxstep)
        # a -deltax of a different unit is not correct for the new unit
        lastbinwidths.pop("deltax", None)
        xunit = get_xunit(values.xunit)
        xheader.setText(f"{xunit.kind.capitalize()} [{xunit.label}]")
        binmodebox.setItemText(binmodebox.findData("deltax"), f"-deltax [{xunit.label}]")
        rangesunit = values.xunit

    # each bin mode keeps its decimals, its range, and its default width. The box keeps the last width of each mode
    binwidthranges: dict[str, tuple[int, float, float, float]] = {"deltalogx": (8, 1e-8, 1.0, 1e-3)}
    lastbinwidths: dict[str, str] = {}

    def set_binwidth_box(binmode: str) -> None:
        """Give the box of the bin width the range and the last width of a bin mode."""
        decimals, low, high, default = binwidthranges[binmode]
        binwidthbox.setDecimals(decimals)
        binwidthbox.setRange(low, high)
        binwidthbox.setValue(float(lastbinwidths.get(binmode, default)))

    def show_direction_choices(directionkind: str) -> None:
        """Fill the box of the direction bins with the bins of a kind of viewing direction."""
        nonlocal shownkind
        if directionkind == shownkind:
            return
        directionbox.clear()
        for dirbin, label in get_direction_choices_of_kind(directionkind):
            directionbox.addItem(f"{dirbin}: {label}", dirbin)
        shownkind = directionkind

    def get_direction_choices_of_kind(directionkind: str) -> list[tuple[int, str]]:
        if not directionkind:
            return []
        if directionkind not in directionchoices:
            directionchoices[directionkind] = get_direction_choices(viewer.runfolders[0], directionkind)
        return directionchoices[directionkind]

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
        key = (
            values.showemission,
            values.showabsorption,
            values.groupby,
            values.references,
            bool(values.deltax or values.deltalogx),
            values.yvariable,
            values.directionkind,
            values.otheroptions,
        )
        if key not in rejections:
            groupbys = [
                None
                if choice == (values.groupby or viewer.defaultgroupby)
                else viewer.get_rejection(dc.replace(values, groupby=choice, showemission=True))
                for choice in viewer.groupbychoices
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
        # matplotlib changes the size of the figure in inches when the pixel ratio of the screen changes
        sizeinches = tuple(viewer.fig.get_size_inches())
        if canvas.size() == size and math.isclose(viewer.fig.dpi, dpi, rel_tol=1e-6) and sizeinches == viewer.figsize:
            return
        viewer.fig.set_dpi(dpi)
        viewer.fig.set_size_inches(figwidth, figheight, forward=False)
        canvas.setFixedSize(size)
        canvas.draw_idle()

    def set_edit_text(edit: QtWidgets.QLineEdit, text: str) -> None:
        """Show the text in a field, unless the user types in that field.

        A plot or a Play step can end while the user types. Without this check, the text of the values replaces the
        text that the user typed.
        """
        if not (edit.hasFocus() and edit.isModified()):
            edit.setText(text)

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
            set_edit_text(widthedit, f"{values.width:g}")
        else:
            first, last = (viewer.validtimesteps.index(timestep) for timestep in viewer.get_selection(values))
            timeslider.setValue((first + last) // 2)
            widthslider.setValue(last - first + 1)
            set_edit_text(widthedit, str(last - first + 1))
        set_edit_text(timeedit, f"{values.centre:.4g}")
        timestepslabel.setText(viewer.get_timesteps_text())
        xrangeslider.set_positions(
            *(
                to_position(math.log10(max(float(limit), 10.0 ** logxrange[0])), *logxrange)
                for limit in (values.xmin, values.xmax)
            )
        )
        set_edit_text(xminedit, values.xmin)
        set_edit_text(xmaxedit, values.xmax)
        xunitbox.setCurrentText(values.xunit)
        yscalebox.setCurrentText(values.yscale)
        logscalexcheck.setChecked(values.logscalex)
        isyfixed = bool(values.ymin or values.ymax)
        fixycheck.setChecked(isyfixed)
        set_edit_text(yminedit, values.ymin)
        set_edit_text(ymaxedit, values.ymax)
        for edit in (yminedit, ymaxedit):
            edit.setEnabled(isyfixed)
        emissioncheck.setChecked(values.showemission)
        absorptioncheck.setChecked(values.showabsorption)
        groupbybox.setCurrentText(values.groupby or viewer.defaultgroupby)
        countbox.setValue(values.maxseriescount)
        # these options apply only to an emission or absorption plot, and a disabled control keeps its place
        for widget in (countlabel, countbox, nostackcheck, lockbutton):
            widget.setEnabled(values.showemission or values.showabsorption)
        nostackcheck.setChecked(values.nostack)
        lockbutton.setChecked(bool(values.fixedionlist))
        lockbutton.setText(f"Lock series ({len(values.fixedionlist)})" if values.fixedionlist else "Lock series")
        binmode = "deltax" if values.deltax else "deltalogx" if values.deltalogx else ""
        binmodebox.setCurrentIndex(binmodebox.findData(binmode))
        if binmode:
            lastbinwidths[binmode] = getattr(values, binmode)
        # the disabled box keeps the last -deltax, thus that width applies again when the user selects -deltax
        set_binwidth_box(binmode or "deltax")
        binwidthbox.setEnabled(bool(binmode))
        frompacketscheck.setChecked(values.frompackets)
        yvariablebox.setCurrentText(values.yvariable)
        normalisedcheck.setChecked(values.normalised)
        hidenetcheck.setChecked(values.hidenetspectrum)
        hideothercheck.setChecked(values.hideother)
        thermalcheck.setChecked(values.usethermalemissiontype)
        for widget in (hidenetcheck, hideothercheck, thermalcheck):
            widget.setEnabled(values.showemission or values.showabsorption)
        directionkindbox.setCurrentIndex(directionkindbox.findData(values.directionkind))
        show_direction_choices(values.directionkind)
        directionbox.setCurrentIndex(directionbox.findData(values.directionbins[0]) if values.directionbins else -1)
        directionbox.setEnabled(bool(values.directionkind))
        if [referencelist.item(index).text() for index in range(referencelist.count())] != list(values.references):
            referencelist.clear()
            referencelist.addItems(list(values.references))
        # after plotspectra rejects a command, the table shows the old options again without the rows with no value
        if get_complete_rows() != values.otheroptions:
            optionrows[:] = list(values.otheroptions)
            show_option_rows()
        commandtext.setPlainText(viewer.get_command())
        show_rejections()
        for blocker in blockers:
            blocker.unblock()
        fit_canvas()

    requestedvalues: ControlValues | None = None
    # viewer.values holds the last values that the user gave, and drawnvalues holds the values of the plot
    drawnvalues = viewer.values

    def fit_figwidthscale() -> None:
        """Give the plot the -figwidthscale that fills the plot area."""
        area = plotarea.contentsRect()
        values = viewer.values
        if area.width() <= 0 or area.height() <= 0 or viewer.figsize[0] <= 0.0:
            return
        figwidthscale = viewer.get_fitted_figwidthscale(area.width(), area.height())
        if figwidthscale != values.figwidthscale:
            apply(dc.replace(values, figwidthscale=figwidthscale))

    fittimer.timeout.connect(fit_figwidthscale)

    def apply(values: ControlValues) -> None:
        """Show the new values now, and draw them when Qt has no other events to process.

        A drag gives a new value for each movement of the mouse, and a plot takes a maximum of 1 s.
        draw_requested reads only the last values that the user gave, thus the plot follows the drag.
        """
        nonlocal requestedvalues
        if requestedvalues is None:
            QtCore.QTimer.singleShot(0, window, draw_requested)
        requestedvalues = values
        # each handler makes its values from viewer.values, thus a second change before the plot keeps the first
        viewer.values = values
        show_values()

    def draw_requested() -> None:
        """Draw the plot of the last values, or show a message and keep the old values if plotspectra rejects them."""
        nonlocal requestedvalues, drawnvalues
        values, requestedvalues = requestedvalues, None
        if values is None:
            return
        # a plot of an emission file or of the packets is slow, thus the cursor and the status bar show the wait
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        drawtimelabel.setText("Plot in progress...")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        starttime = time.perf_counter()
        # change() keeps the values of the last plot when plotspectra rejects the new values
        viewer.values = drawnvalues
        try:
            message = viewer.change(values, preview=True)
        finally:
            drawnvalues = viewer.values
            QtWidgets.QApplication.restoreOverrideCursor()
        drawkind = "Preview" if viewer.drewpreview else "Plot"
        drawtimelabel.setText(f"{drawkind} time: {time.perf_counter() - starttime:.2f} s")
        # clear the readout, because it holds the values of the old plot until the mouse moves again
        readoutlabel.setText("")
        messagelabel.setText(message or "")
        show_values()
        # --showabsorption changes the height of the frames, thus the plot can need a new -figwidthscale
        fittimer.start()
        # each change starts the timer again, thus the full plot follows after the last change
        if viewer.drewpreview:
            fulldrawtimer.start()
        # a rejection occurs again at each step, thus a rejection stops the Play button
        if message is not None:
            playbutton.setChecked(False)
        elif playbutton.isChecked():
            QtCore.QTimer.singleShot(PLAY_MILLISECONDS, window, play_step)

    def draw_full() -> None:
        """Replace the preview with the plot of all the packets."""
        nonlocal drawnvalues
        # a change in the queue, or the Play button, draws a new preview and starts this timer again
        if requestedvalues is not None or playbutton.isChecked() or not viewer.drewpreview:
            return
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        drawtimelabel.setText("Plot in progress...")
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        starttime = time.perf_counter()
        try:
            message = viewer.change(drawnvalues)
        finally:
            drawnvalues = viewer.values
            QtWidgets.QApplication.restoreOverrideCursor()
        drawtimelabel.setText(f"Plot time: {time.perf_counter() - starttime:.2f} s")
        readoutlabel.setText("")
        messagelabel.setText(message or "")
        show_values()

    fulldrawtimer.timeout.connect(draw_full)

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
            count = min(max(1, round(width)), nvalid)
            start = get_nearest_range_start(
                [viewer.tmids[timestep] for timestep in viewer.validtimesteps], centre, count
            )
            newvalues = viewer.snap(values, viewer.validtimesteps[start], viewer.validtimesteps[start + count - 1])
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
        elif viewer.drewpreview:
            fulldrawtimer.start()

    def set_xlimits(low: float, high: float) -> None:
        # a value of 3 significant digits gives a short command, and a text field gives an exact value
        low, high = float(f"{low:.3g}"), float(f"{high:.3g}")
        if low < high:
            apply(dc.replace(viewer.values, xmin=format(low, ".10g"), xmax=format(high, ".10g")))

    def on_xrange(handle: int, position: int) -> None:
        """Set the limit of the handle that moved, and keep the other limit as its text field gives it.

        A position of the slider is on a fixed range with a step, thus it cannot show each limit that the user types.
        """
        # a value of 3 significant digits gives a short command
        limit = float(f"{10.0 ** from_position(position, *logxrange):.3g}")
        values = viewer.values
        if handle == 0 and limit < float(values.xmax):
            apply(dc.replace(values, xmin=format(limit, ".10g")))
        elif handle == 1 and limit > float(values.xmin):
            apply(dc.replace(values, xmax=format(limit, ".10g")))

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
        values = dc.replace(
            values,
            yscale=yscalebox.currentText(),
            logscalex=logscalexcheck.isChecked(),
            yvariable=yvariablebox.currentText(),
            normalised=normalisedcheck.isChecked(),
        )
        if values != viewer.values:
            apply(values)

    def on_emission_options() -> None:
        groupby = groupbybox.currentText()
        showemission = emissioncheck.isChecked()
        # -groupby colours the emission plot, thus a choice of -groupby also sets --showemission.
        # An empty --showemission check box removes the -groupby choice.
        if groupby != (viewer.values.groupby or viewer.defaultgroupby):
            showemission = True
        elif not showemission:
            groupby = viewer.defaultgroupby
        # plotspectra takes the default -groupby when the command gives none, thus the command stays short
        if groupby == viewer.defaultgroupby:
            groupby = None
        values = dc.replace(
            viewer.values,
            showemission=showemission,
            showabsorption=absorptioncheck.isChecked(),
            groupby=groupby,
            maxseriescount=countbox.value(),
            nostack=nostackcheck.isChecked(),
            deltax=format(binwidthbox.value(), ".10g") if binmodebox.currentData() == "deltax" else "",
            deltalogx=format(binwidthbox.value(), ".10g") if binmodebox.currentData() == "deltalogx" else "",
            frompackets=frompacketscheck.isChecked(),
            hidenetspectrum=hidenetcheck.isChecked(),
            hideother=hideothercheck.isChecked(),
            usethermalemissiontype=thermalcheck.isChecked(),
        )
        # the labels of a locked list belong to one -groupby, thus a new -groupby removes the lock
        if groupby != viewer.values.groupby:
            values = remove_series_lock(values)
        if values != viewer.values:
            apply(values)

    def on_binmode() -> None:
        # the box holds the width of the previous mode, thus it takes the width of the new mode before the values change
        with QtCore.QSignalBlocker(binwidthbox):
            set_binwidth_box(binmodebox.currentData() or "deltax")
        on_emission_options()

    def on_direction() -> None:
        directionkind: str = directionkindbox.currentData()
        dirbins = [dirbin for dirbin, _ in get_direction_choices_of_kind(directionkind)]
        if directionkind == viewer.values.directionkind:
            directionbins = (directionbox.currentData(),) if directionkind else ()
        else:
            # a new kind keeps the direction bin if that kind has the same bin
            directionbins = tuple(dirbin for dirbin in viewer.values.directionbins[:1] if dirbin in dirbins) or tuple(
                dirbins[:1]
            )
        values = dc.replace(viewer.values, directionkind=directionkind, directionbins=directionbins)
        if values != viewer.values:
            apply(values)

    def on_lock(checked: bool) -> None:
        if not checked:
            apply(remove_series_lock(viewer.values))
            return
        if not (series := viewer.get_drawn_series()):
            show_error("The plot has no series of contributions to lock")
            return
        apply(dc.replace(viewer.values, fixedionlist=series))

    def on_add_reference() -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(window, "Add reference spectra", str(referencefolder))
        names = [get_reference_token(filename) for filename in filenames]
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
        if not folder:
            return
        # a SystemExit in a Qt slot ends the process, thus an error of the new window stays in this window
        errors = io.StringIO()
        try:
            with contextlib.redirect_stderr(errors):
                opened = open_window([folder], windows)
        except SystemExit:
            opened = False
        finally:
            sys.stderr.write(errors.getvalue())
        if not opened:
            show_error(f"The viewer cannot open {folder}: {get_first_line(errors.getvalue())}")

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
        # the list holds a reference to each open window, thus Python does not delete the window. A closed window
        # leaves the list
        windows.remove(window)

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
    xrangeslider.limitmoved.connect(on_xrange)
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
    nostackcheck.toggled.connect(on_emission_options)
    lockbutton.toggled.connect(on_lock)
    binmodebox.currentIndexChanged.connect(on_binmode)
    binwidthbox.valueChanged.connect(on_emission_options)
    frompacketscheck.toggled.connect(on_emission_options)
    for checkbox in (hidenetcheck, hideothercheck, thermalcheck):
        checkbox.toggled.connect(on_emission_options)
    yvariablebox.currentTextChanged.connect(on_axes)
    normalisedcheck.toggled.connect(on_axes)
    directionkindbox.currentIndexChanged.connect(on_direction)
    directionbox.currentIndexChanged.connect(on_direction)
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
    if (windowhandle := window.windowHandle()) is not None:

        def on_screen(_screen: QtGui.QScreen) -> None:
            # matplotlib handles the new pixel ratio first, thus the fit waits until Qt has no other events
            QtCore.QTimer.singleShot(0, window, fit_canvas)

        windowhandle.screenChanged.connect(on_screen)
    return True
