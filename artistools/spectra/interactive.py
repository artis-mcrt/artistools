"""Show the plot of plotspectra in a window with controls for the time, the x range, and the emission plot."""

import argparse
import contextlib
import dataclasses as dc
import math
import shlex
import typing as t
from functools import partial
from pathlib import Path

import matplotlib.figure as mplfig
import numpy as np
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg

from artistools.misc import exit_with_error
from artistools.misc import get_dirbin_definitions
from artistools.misc import get_dirbins
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_nprocs
from artistools.misc import get_time_range
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import separate_trailing_folders
from artistools.misc.fileio import COMPRESSED_EXTENSIONS
from artistools.packets.core import RANKS_PER_BATCH
from artistools.plottools import ExponentLabelFormatter
from artistools.plottools import LABELWIDTH_INCHES
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
from artistools.viewertools import add_command_section
from artistools.viewertools import add_copy_box
from artistools.viewertools import add_default_options
from artistools.viewertools import add_figure_section
from artistools.viewertools import add_menus
from artistools.viewertools import add_recent_model
from artistools.viewertools import add_row
from artistools.viewertools import add_section
from artistools.viewertools import apply_dark_colours
from artistools.viewertools import connect_plot_mouse
from artistools.viewertools import copy_figure_of_command
from artistools.viewertools import copy_text
from artistools.viewertools import DrawQueue
from artistools.viewertools import exit_for_other_actions
from artistools.viewertools import export_animation
from artistools.viewertools import fit_canvas
from artistools.viewertools import FIT_MILLISECONDS
from artistools.viewertools import follow_colour_scheme
from artistools.viewertools import get_bool_setting
from artistools.viewertools import get_changed_arguments
from artistools.viewertools import get_dark_plot_colours
from artistools.viewertools import get_fitted_figwidthscale
from artistools.viewertools import get_helptexts
from artistools.viewertools import get_keyboard_help
from artistools.viewertools import get_line_readouts
from artistools.viewertools import get_nearest_range_start
from artistools.viewertools import get_new_figwidthscale
from artistools.viewertools import get_option_row_tokens
from artistools.viewertools import get_option_tokens
from artistools.viewertools import get_python_call
from artistools.viewertools import get_short_number
from artistools.viewertools import make_central_splitter
from artistools.viewertools import make_completer
from artistools.viewertools import make_fps_box
from artistools.viewertools import make_option_table
from artistools.viewertools import make_parser
from artistools.viewertools import make_play_button
from artistools.viewertools import make_play_row
from artistools.viewertools import make_plot_area
from artistools.viewertools import make_range_slider
from artistools.viewertools import make_readout_tag
from artistools.viewertools import make_row_layout
from artistools.viewertools import make_segmented_control
from artistools.viewertools import make_sidebar
from artistools.viewertools import make_slider
from artistools.viewertools import make_status_bar
from artistools.viewertools import make_step_button
from artistools.viewertools import make_timer
from artistools.viewertools import make_window
from artistools.viewertools import open_model_folder
from artistools.viewertools import open_model_window
from artistools.viewertools import OptionRows
from artistools.viewertools import parse_command_tokens
from artistools.viewertools import remove_options
from artistools.viewertools import ROW_SPACING
from artistools.viewertools import run_command_step
from artistools.viewertools import run_command_step_with_warning
from artistools.viewertools import run_viewer_application
from artistools.viewertools import save_figure_of_command
from artistools.viewertools import set_command_text
from artistools.viewertools import set_drop_handler
from artistools.viewertools import set_edit_text
from artistools.viewertools import set_window_document
from artistools.viewertools import show_figure_in_canvas
from artistools.viewertools import show_status_message
from artistools.viewertools import show_status_note
from artistools.viewertools import show_window
from artistools.viewertools import SLIDER_STEPS
from artistools.viewertools import split_dpi_row
from artistools.viewertools import split_option_rows
from artistools.viewertools import start_play_timer

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import matplotlib.axes as mplax
    import numpy.typing as npt
    from PySide6 import QtWidgets

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
    "usedegrees",
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
    # the Figure section gives the resolution of a PNG file
    "dpi",
    "timedayslist",
    "multispecplot",
    "makevspecpol",
    "averagevspecpolfiles",
    "output_spectra",
})

# the time after the last preview, or after the release of the slider, before the full plot replaces the preview
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
    usedegrees: bool
    fixedionlist: tuple[str, ...]
    # the paths of the ARTIS models and the reference spectra, in the order of the command
    spectra: tuple[str, ...]
    figwidthscale: float
    otheroptions: OptionRows


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


class RenderedSpectrum(t.NamedTuple):
    """A figure that the worker thread drew, with the frames and the data that the window reads."""

    fig: mplfig.Figure
    axes: "npt.NDArray[t.Any]"
    residualaxis: "mplax.Axes | None"
    dfalldata: pl.DataFrame
    ispreview: bool


def fix_title_position(axis: "mplax.Axes") -> None:
    """Keep the title at the top of the frame.

    matplotlib moves the title above the offset text of the y axis. For this, it measures the y axis each time that it
    draws the plot. ExponentLabelFormatter puts the offset in the label of the axis, thus the offset text is empty and
    the title stays at the top of the frame.
    """
    if axis.get_title() and isinstance(axis.yaxis.get_major_formatter(), ExponentLabelFormatter):
        axis.set_title(axis.get_title(), y=1.0)


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


def get_reference_token(filename: str) -> str:
    """Return the name of a reference file if plotspectra finds that same file by the name, and the path if not.

    plotspectra searches the working folder before the reference data of artistools. Thus a file of the same name in
    the working folder takes the place of a file from the reference data. The name alone gives a short command.
    """
    found = find_reference_spectrum_file_or_none(Path(filename).name)
    return Path(filename).name if found is not None and found.resolve() == Path(filename).resolve() else filename


# plotspectra reads the model in the working folder when the command gives no path
DEFAULT_SPECTRA: t.Final = (".",)


def get_spectrum_path(path: str) -> Path:
    """Return the full path of the folder or the file of a spectrum, e.g. of "." or of a name of a reference spectrum.

    Two spellings of one spectrum, e.g. "." and the full path of the working folder, then give the same path.
    """
    if path_is_reference_spectrum(path):
        return (find_reference_spectrum_file_or_none(path) or Path(path)).resolve()
    return Path(path).resolve()


def get_spectrum_item_text(path: str) -> str:
    """Return the text of a path in the list of spectra: the kind and the full path.

    The list shortens a long path in the middle, thus the text keeps the start and the end of the path.
    """
    if path_is_reference_spectrum(path):
        return f"Reference: {(find_reference_spectrum_file_or_none(path) or Path(path)).absolute()}"
    return f"Model: {Path(path).absolute()}"


def get_reference_spectrum_names() -> list[str]:
    """Return the names of the reference spectra in the data of artistools, without the suffix of a compressed file.

    plotspectra finds a compressed file by the name without the suffix. A metadata file with no data file beside it
    gives no name, because plotspectra has no spectrum to read.
    """
    from artistools.commands import get_path

    folder = get_path("artistools_dir") / "data" / "refspectra"
    names = {
        path.name.removesuffix(path.suffix) if path.suffix in COMPRESSED_EXTENSIONS else path.name
        for path in folder.iterdir()
        if path.is_file() and not path.name.startswith(".") and not path.name.endswith(".meta.yml")
    }
    return sorted(names, key=str.lower)


def keep_figwidthscale(restored: ControlValues, current: ControlValues) -> ControlValues:
    """Return the values that Undo restores, with the current -figwidthscale, which the window sets."""
    return dc.replace(restored, figwidthscale=current.figwidthscale)


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
    exit_for_other_actions(
        "spectra",
        {
            "-timedayslist": args.multispecplot,
            "--makevspecpol": args.makevspecpol,
            "--averagevspecpolfiles": args.averagevspecpolfiles,
            "--output_spectra": args.output_spectra,
            f"-stokesparam {args.stokesparam}": "/" in args.stokesparam,
        },
    )


def get_python_code(parser: argparse.ArgumentParser, tokens: "Sequence[str]") -> str:
    """Return the Python code that draws the plot of the command, with each argument that differs from its default."""
    args = parse_command_tokens(parser, tokens)
    if args is None:
        return "# plotspectra rejects the command"
    return get_python_call("at.spectra.plotspectra.main", get_changed_arguments(parser, args))


class SpectrumViewer:
    """The plot of the viewer and the values of its controls.

    The window reads and changes the values, and a test can do the same without a display. Each call to draw
    makes the command from the values, then parses it and draws it with the code of plotspectra. Thus the plot
    always agrees with the command.
    """

    def __init__(self, tokens: "Sequence[str]", fig: mplfig.Figure) -> None:
        """Read the arguments of the user, and take the first values of the controls from them."""
        parser = make_parser(addargs)
        usertokens = remove_options(parser, tokens, {"interactive"})
        # parse_cli_args puts "--" in front of the ARTIS folders at the end, thus an option that reads a list, e.g.
        # -fixedionlist, does not take a folder. The removal of the controlled options must keep those folders
        basetokens = remove_options(parser, separate_trailing_folders(usertokens), CONTROLLED_DESTS)
        args = parse_cli_args(addargs, None, None, usertokens)
        # resolve_frompackets gives an emission plot a default -groupby, thus the value comes from the arguments
        givengroupby: str | None = args.groupby
        # -deltax and --notimeclamp also make plotspectra read the packets, thus the box shows only a --frompackets that
        # the user gave
        givesfrompackets = bool(args.frompackets)
        # with --notimeclamp, a range of days keeps its bounds, and a single time or a timestep reads a whole timestep
        givesdaysrange = args.timemin is not None or (args.timedays is not None and "-" in args.timedays)
        resolve_plot_args(args)
        check_viewer_args(args)
        self.args = args

        if not get_artis_run_folders(args.modelspecpaths):
            exit_with_error(
                "--interactive takes the time range from the timesteps of an ARTIS run, and no path names a run",
                "Give the folder of an ARTIS run, e.g. plotspectra mymodel --interactive",
            )
        self.load_runs(args.modelspecpaths)

        # the table of the window shows each option that no other control sets. The tokens that no option takes are
        # paths, e.g. the path of "--notitle mymodel", or the paths after "--"
        pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
        otheroptions, positionaltokens = split_option_rows(parser, basetokens[pathcount:])
        # the order of the paths gives the -label and the style of each series, thus the paths keep their order
        startpaths = [*basetokens[:pathcount], *(word for word in positionaltokens if word != "--")]
        self.modelpathtokens = [path for path in startpaths if not path_is_reference_spectrum(path)]
        self.parser = parser

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
        self.helptexts = get_helptexts(parser)
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
            usedegrees=bool(args.usedegrees),
            fixedionlist=tuple(args.fixedionlist or ()),
            spectra=tuple(startpaths) or DEFAULT_SPECTRA,
            figwidthscale=args.figwidthscale,
            otheroptions=otheroptions,
        )
        self.values = self.clamp_time(values) if values.notimeclamp else self.snap(values, *self.get_selection(values))
        # the list of spectra of the runs of load_runs. After a change of the list, e.g. Add Model or Undo, load_runs
        # reads the runs again. The paths of the command give the same runs as the list of the values
        self.runspectra = self.values.spectra

        self.fig = fig
        self.axes: npt.NDArray[t.Any] = np.empty(0, dtype=object)
        self.residualaxis: mplax.Axes | None = None
        self.drewpreview = False
        # the window sets this before each plot that a slider drag gives, and the worker thread reads it
        self.dragging = False
        # the colours of the window in Dark Mode, which the window sets and the worker thread reads
        self.darkcolours: tuple[str, str] | None = None
        # the last warning of the last plot, which the status bar shows
        self.warning = ""
        # a window can change the size of the figure, thus the size of the frames stays here
        self.figsize: tuple[float, float] = (0.0, 0.0)
        # the readout of the window reads the contributions of an emission plot from this frame
        self.dfalldata = pl.DataFrame()

    def load_runs(self, spectra: "Sequence[str | Path]") -> None:
        """Read the timesteps of the ARTIS runs of the spectra, and the times that are valid for all the runs.

        The time controls take the timesteps of the first run. plotspectra rejects a time outside the arrival times of
        the escaped packets of each run. Thus the controls stay inside the times that are valid for all the runs. With
        --plotinvalidpart, plotspectra accepts all times. A reference spectrum has no run.
        """
        runfolders = get_artis_run_folders([Path(path) for path in spectra])
        tmids = get_timestep_times(runfolders[0], loc="mid")
        tstarts = get_timestep_times(runfolders[0], loc="start")
        tends = get_timestep_times(runfolders[0], loc="end")
        timebounds = [tstarts[0], tends[-1]]
        if not self.args.plotinvalidpart:
            for runfolder in runfolders:
                with contextlib.suppress(FileNotFoundError):
                    _, validstart, validend = get_escaped_arrivalrange(runfolder)
                    if validstart is not None:
                        timebounds[0] = max(timebounds[0], float(validstart))
                    if validend is not None:
                        timebounds[1] = min(timebounds[1], float(validend))
        self.runfolders, self.tmids, self.tstarts, self.tends = runfolders, tmids, tstarts, tends
        self.twidths = get_timestep_times(runfolders[0], loc="delta")
        self.timebounds = (timebounds[0], timebounds[1])
        self.validtimesteps = [
            timestep
            for timestep in range(len(tmids))
            if tstarts[timestep] >= self.timebounds[0] and tends[timestep] <= self.timebounds[1]
        ] or list(range(len(tmids)))
        # a preview reads the packets of the first batch of ranks only. A run with one batch has no faster preview
        self.previewmaxpacketfiles = (
            RANKS_PER_BATCH if any(get_nprocs(runfolder) > RANKS_PER_BATCH for runfolder in runfolders) else None
        )
        self.runspectra = tuple(str(path) for path in spectra)

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
        options = ["-t", timedays, *get_option_tokens("-xmin", values.xmin), *get_option_tokens("-xmax", values.xmax)]
        if values.notimeclamp:
            options.append("--notimeclamp")
        if values.xunit != self.defaultxunit:
            options += ["-xunit", values.xunit]
        if values.logscalex:
            options.append("--logscalex")
        if values.yscale != self.defaultyscale:
            options += ["-yscale", values.yscale]
        if values.ymin:
            options += get_option_tokens("-ymin", values.ymin)
        if values.ymax:
            options += get_option_tokens("-ymax", values.ymax)
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
            # the angles of a direction go in the labels, thus the flag has no effect without a direction
            (values.usedegrees and bool(values.directionkind), "--usedegrees"),
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
        # a command with no path reads the model in the working folder, thus that model needs no path
        paths = [] if values.spectra == DEFAULT_SPECTRA else list(values.spectra)
        return make_command_tokens([*paths, *get_option_row_tokens(values.otheroptions)], options)

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

        def check() -> str | None:
            plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens(values))
            resolve_plot_args(plotargs)
            check_viewer_args(plotargs)
            if (plotargs.showemission, plotargs.showabsorption) != (values.showemission, values.showabsorption):
                return "A different option of the command keeps the emission plot on"
            return None

        return run_command_step(check, echo=False)

    def draw(self, *, quiet: bool = True) -> str | None:
        """Draw the plot of the values, and return the reason for the status line if plotspectra rejects it.

        The terminal shows the whole error, and the status line shows its first line.
        """
        return self.render(self.values, quiet=quiet)()

    def render(self, values: ControlValues, *, quiet: bool = True, preview: bool = False) -> "Callable[[], str | None]":
        """Draw the plot of the values on a new figure, and return the function that shows it in the canvas.

        The function returns the reason for the status line if plotspectra rejects the values, and the old plot then
        stays. A worker thread can run this method, because it changes nothing that the window reads. The function
        that it returns must run in the thread of the window.

        A preview of a plot of the packets reads the first batch of ranks only. For the 20 batches of a kilonova run,
        a range of 8 days took 0.12 s in place of 1.1 s. The flux stays correct, because the reader divides by the
        number of ranks that it reads. The reader does not divide a count of packets, thus a plot of
        -yvariable packetcount has no preview. The command in the window has no -maxpacketfiles for the preview.
        """
        plots: list[RenderedSpectrum] = []

        def make_plot() -> str | None:
            plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens(values))
            resolve_plot_args(plotargs)
            check_viewer_args(plotargs)
            ispreview = bool(
                preview
                and plotargs.frompackets
                and plotargs.maxpacketfiles is None
                and plotargs.yvariable != "packetcount"
                and self.previewmaxpacketfiles
            )
            if ispreview:
                plotargs.maxpacketfiles = self.previewmaxpacketfiles
            if (plotargs.showemission, plotargs.showabsorption) != (values.showemission, values.showabsorption):
                return "A different option of the command keeps the emission plot on"
            fig = mplfig.Figure()
            FigureCanvasAgg(fig)
            _, axes, residualaxis = make_plot_figure(plotargs, fig=fig)
            dfalldata, _ = draw_plot(plotargs, axes, residualaxis)
            for axis in axes:
                fix_title_position(axis)
            if (darkcolours := self.darkcolours) is not None:
                apply_dark_colours(fig, *darkcolours)
            plots.append(
                RenderedSpectrum(
                    fig=fig, axes=axes, residualaxis=residualaxis, dfalldata=dfalldata, ispreview=ispreview
                )
            )
            return None

        message, warning = run_command_step_with_warning(make_plot, quiet=quiet)

        def show_plot() -> str | None:
            self.warning = warning
            if message is not None:
                return message
            plot = plots[0]
            self.figsize = show_figure_in_canvas(self.fig, plot.fig)
            self.fig, self.axes, self.residualaxis = plot.fig, plot.axes, plot.residualaxis
            self.dfalldata, self.drewpreview = plot.dfalldata, plot.ispreview
            return None

        return show_plot

    def get_fitted_figwidthscale(self, areawidth: float, areaheight: float) -> float:
        """Return the -figwidthscale that gives the figure the shape of the plot area."""
        marginwidth = LABELWIDTH_INCHES + RIGHTMARGIN_INCHES
        return get_fitted_figwidthscale(self.figsize, self.values.figwidthscale, marginwidth, areawidth, areaheight)

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
        """Draw the plot of the new values, and keep the old values and the old plot if plotspectra rejects them."""
        values = self.clamp_time(values)
        message = self.render(values, preview=preview)()
        if message is None:
            self.values = values
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
        parts = [f"{x:.5g} {xunit.label}", *(get_line_readouts(self.axes[0], x) if len(self.axes) else [])]
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


def get_icon_curve() -> "npt.NDArray[np.float64]":
    """Return the curve of the icon of the viewer, which is a spectrum with two absorption lines."""
    xvalues = np.linspace(0.1, 0.9, 200)
    return 0.72 - 0.45 * np.exp(-(((xvalues - 0.42) / 0.06) ** 2)) - 0.25 * np.exp(-(((xvalues - 0.65) / 0.09) ** 2))


# the keys and the mouse actions of the window. get_keyboard_help adds the shortcuts of the menus
KEYBOARD_HELP_ROWS: t.Final = (
    ("<b>Left</b>, <b>Right</b>", "Move the time to the adjacent timestep"),
    ("<b>Up</b>, <b>Down</b>", "Make the time range one timestep wider or narrower"),
    ("<b>Home</b>, <b>End</b>", "Move the time to the first or the last valid timestep"),
    ("<b>Drag</b> across the plot", "Select the x range"),
    ("<b>Double-click</b> the plot", "Get the default x range"),
)


def run_viewer(tokens: "Sequence[str]") -> None:
    """Open the window of the viewer, and print the command of the last plot when the window closes.

    The Dock icon also takes a file, e.g. a reference spectrum, which the active window adds to its spectra.
    """
    run_viewer_application(APPLICATION_NAME, get_icon_curve(), open_window, tokens, ("public.folder", "public.data"))


def open_window(tokens: "Sequence[str]", windows: "list[QtWidgets.QMainWindow]") -> str | None:
    """Open a window of the viewer for the plotspectra arguments in tokens, or return the reason for no window."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    from artistools.commands import get_path

    # the Settings window can give a new window options, e.g. -figscale, that the command does not give
    viewer = SpectrumViewer(add_default_options(make_parser(addargs), tokens), mplfig.Figure())
    window = make_window(APPLICATION_NAME)
    # a command with no path reads the model of the working folder
    modelnames = [Path(path).resolve().name for path in viewer.modelpathtokens] or [viewer.runfolders[0].resolve().name]
    set_window_document(window, viewer.runfolders[0], ", ".join(modelnames))
    canvas = FigureCanvasQTAgg(viewer.fig)
    viewer.darkcolours = get_dark_plot_colours()
    if (message := viewer.draw(quiet=False)) is not None:
        # the arguments of the user give the error, and the terminal shows it
        if not windows:
            raise SystemExit(1)
        return message
    windows.append(window)
    add_recent_model(viewer.runfolders[0])

    fulldrawtimer = make_timer(window, FULL_DRAW_MILLISECONDS)
    playtimer = make_timer(window, 0)
    fittimer = make_timer(window, FIT_MILLISECONDS)

    def on_resize() -> None:
        fit_canvas(canvas, viewer.figsize, plotarea)
        # a new plot takes up to 1 s, thus the plot takes the new shape only when the resize stops
        fittimer.start()

    plotarea = make_plot_area(canvas, on_resize)
    sidebar, panellayout = make_sidebar()
    make_central_splitter(window, plotarea, sidebar)

    # each continuous slider maps its position from 0 to SLIDER_STEPS onto the range of its value
    def to_position(value: float, low: float, high: float) -> int:
        return round(SLIDER_STEPS * (min(max(value, low), high) - low) / max(high - low, 1e-300))

    def from_position(position: int, low: float, high: float) -> float:
        return low + (high - low) * position / SLIDER_STEPS

    helptexts = viewer.helptexts
    logtrange = (math.log10(viewer.timebounds[0]), math.log10(viewer.timebounds[1]))
    widthmax = max((viewer.timebounds[1] - viewer.timebounds[0]) / 4.0, viewer.values.width)
    nvalid = len(viewer.validtimesteps)

    _, timegrid = add_section(panellayout, "Time")
    # the index of a segment: 0 snaps the time range to whole timesteps, and 1 gives --notimeclamp
    modesegments = make_segmented_control(
        ["Snap to Timesteps", "Continuous"],
        [
            "The time range holds whole timesteps, as plotspectra reads them by default",
            f"--notimeclamp: {helptexts.get('notimeclamp', '')}",
        ],
    )
    timegrid.addWidget(modesegments, 0, 0, 1, 3, QtCore.Qt.AlignmentFlag.AlignLeft)
    timeslider, widthslider = make_slider(), make_slider()
    timeedit, widthedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    widthlabel = QtWidgets.QLabel()
    timestepslabel = QtWidgets.QLabel()
    playbutton = make_play_button(
        "Move the time through the valid timesteps of the run, and start again after the last timestep (Space)"
    )
    fpsbox = make_fps_box()
    previousbutton, nextbutton = make_step_button(forward=False), make_step_button(forward=True)
    timetip = "The middle of the time range in days. The Left key and the Right key move it to the adjacent timestep."
    widthtip = (
        'The width of the time range. The Up key and the Down key change the width by one timestep. With "Snap to'
        ' Timesteps", the width is a count of timesteps.'
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
    timegrid.addLayout(make_play_row([previousbutton, nextbutton], timestepslabel, fpsbox, playbutton), 3, 0, 1, -1)

    xheader, xgrid = add_section(panellayout, "", key="x axis")
    xrangeslider, set_xrange_positions, connect_xrange, _ = make_range_slider(SLIDER_STEPS)
    xminedit, xmaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    zoomtip = " Drag across the plot to select a range. Double-click the plot to get the default range."
    xrangeslider.setToolTip("The minimum and the maximum of the x axis." + zoomtip)
    for column, (widget, dest) in enumerate(((xminedit, "xmin"), (xrangeslider, ""), (xmaxedit, "xmax"))):
        if dest:
            widget.setFixedWidth(110)
            widget.setToolTip(helptexts.get(dest, "") + zoomtip)
        xgrid.addWidget(widget, 0, column)
    xgrid.setColumnStretch(1, 1)

    _, axesgrid = add_section(panellayout, "Axes")
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

    _, emissiongrid = add_section(panellayout, "Emission and absorption")
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
    lockbutton = QtWidgets.QPushButton("Lock Series")
    lockbutton.setCheckable(True)
    lockbutton.setToolTip(
        "Keep the series of the plot and their colours when the time or the x range changes. The command gives the"
        " series with -fixedionlist."
    )
    add_row(emissiongrid, 0, [emissioncheck, absorptioncheck, nostackcheck])
    hidenetcheck = QtWidgets.QCheckBox("--hidenetspectrum")
    hideothercheck = QtWidgets.QCheckBox("--hideother")
    thermalcheck = QtWidgets.QCheckBox("--use_thermalemissiontype")
    for widget, dest in (
        (hidenetcheck, "hidenetspectrum"),
        (hideothercheck, "hideother"),
        (thermalcheck, "use_thermalemissiontype"),
    ):
        widget.setToolTip(helptexts.get(dest, ""))
    # these rows apply only to an emission or absorption plot, thus they show only for such a plot
    emissionoptions = QtWidgets.QWidget()
    emissionoptionslayout = QtWidgets.QVBoxLayout(emissionoptions)
    emissionoptionslayout.setContentsMargins(0, 0, 0, 0)
    emissionoptionslayout.setSpacing(ROW_SPACING)
    emissionoptionslayout.addLayout(
        make_row_layout([QtWidgets.QLabel("-groupby"), groupbybox, countlabel, countbox, lockbutton])
    )
    emissionoptionslayout.addLayout(make_row_layout([hidenetcheck, hideothercheck, thermalcheck]))
    emissiongrid.addWidget(emissionoptions, 1, 0, 1, -1)

    _, bingrid = add_section(panellayout, "Bins of the packet spectrum")
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

    _, directiongrid = add_section(panellayout, "Viewing direction")
    directionkindbox = QtWidgets.QComboBox()
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
    usedegreescheck = QtWidgets.QCheckBox("--usedegrees")
    usedegreescheck.setToolTip(helptexts.get("usedegrees", ""))
    add_row(directiongrid, 0, [directionkindbox, usedegreescheck])
    # the plot can show several directions at once, thus each direction bin has a checkbox. The list scrolls, and the
    # label of a bin is long, thus the list takes the full width of the sidebar
    directionbox = QtWidgets.QScrollArea()
    directionbox.setWidgetResizable(True)
    directionbox.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    directionbox.setToolTip("The direction bins of the plot, or the observers of the virtual packets")
    directionchecks: dict[int, QtWidgets.QCheckBox] = {}
    directiongrid.addWidget(directionbox, 1, 0, 1, -1)
    # the labels of the direction bins come from the files of the run, thus the window reads them one time for each kind
    directionchoices: dict[tuple[str, bool], list[tuple[int, str]]] = {}
    shownchoices: tuple[str, bool] | None = None
    for box in (countbox, binwidthbox):
        # a typed number applies when the user presses Return or leaves the box, and not after each digit
        box.setKeyboardTracking(False)

    _, spectragrid = add_section(panellayout, "Spectra")
    spectralist = QtWidgets.QListWidget()
    spectralist.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
    # a long path shows its start and its end, and the width of the box sets the length
    spectralist.setTextElideMode(QtCore.Qt.TextElideMode.ElideMiddle)
    spectralist.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    spectralist.setToolTip(
        "The ARTIS models and the observed spectra of the plot, in the order of the command. The order sets the"
        " -label and the style of each series. The command gives a file from the reference data of artistools by"
        " its name alone."
    )
    addmodelbutton = QtWidgets.QPushButton("Add Model…")
    addmodelbutton.setToolTip("Add the folder of an ARTIS run")
    removebutton = QtWidgets.QPushButton("Remove")
    removebutton.setToolTip("Remove the selected spectra. The plot keeps one ARTIS model at least")
    referenceedit = QtWidgets.QLineEdit()
    referenceedit.setPlaceholderText("Add a reference spectrum, e.g. AT2017gfo")
    referenceedit.setToolTip(
        "Type part of the name of a reference spectrum in the data of artistools, then press Return. A name of a file"
        " in the working folder also works."
    )
    referencecompleter = make_completer(get_reference_spectrum_names(), referenceedit)
    referenceedit.setCompleter(referencecompleter)
    openreferencebutton = QtWidgets.QPushButton("Open…")
    openreferencebutton.setToolTip("Add the file of a reference spectrum from a folder")
    referencerow = QtWidgets.QHBoxLayout()
    referencerow.addWidget(referenceedit, 1)
    referencerow.addWidget(openreferencebutton)
    spectragrid.addWidget(spectralist, 0, 0, 1, -1)
    spectragrid.addLayout(make_row_layout([addmodelbutton, removebutton]), 1, 0, 1, -1)
    spectragrid.addLayout(referencerow, 2, 0, 1, -1)
    referencefolder = get_path("artistools_dir") / "data" / "refspectra"
    figuresection = add_figure_section(
        window, panellayout, split_dpi_row(viewer.values.otheroptions, viewer.parser.get_default("dpi"))[1]
    )
    _, optiongrid = add_section(panellayout, "Other options")

    def on_option_rows(rows: OptionRows) -> None:
        apply(dc.replace(viewer.values, otheroptions=rows))

    optiontable, set_option_rows = make_option_table(
        window, viewer.parser, CONTROLLED_DESTS | TABLE_EXCLUDED_DESTS, viewer.values.otheroptions, on_option_rows
    )
    optiongrid.addWidget(optiontable, 0, 0, 1, 2)
    commandtext, copybutton = add_command_section(panellayout)
    pythontext, pythoncopybutton = add_copy_box(
        panellayout, "Python", "Copy the Python code that draws the plot to the clipboard", maxlines=20, wraplines=False
    )
    statusbar = make_status_bar(window)
    # the first plot came before the status bar, and a user of the application sees no terminal
    show_status_message(statusbar, None, viewer.warning)

    signalwidgets: list[QtWidgets.QWidget] = [
        modesegments,
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
        usedegreescheck,
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

    def show_direction_choices(directionkind: str, usedegrees: bool) -> bool:
        """Fill the list of the direction bins with a checkbox for each bin of a kind of viewing direction.

        Return whether the list is new.
        """
        nonlocal shownchoices
        if (directionkind, usedegrees) == shownchoices:
            return False
        checklist = QtWidgets.QWidget()
        checklayout = QtWidgets.QVBoxLayout(checklist)
        checklayout.setContentsMargins(6, 4, 6, 4)
        checklayout.setSpacing(2)
        directionchecks.clear()
        for dirbin, label in get_direction_choices_of_kind(directionkind, usedegrees):
            check = QtWidgets.QCheckBox(f"{dirbin}: {label}")
            check.toggled.connect(on_direction)
            checklayout.addWidget(check)
            directionchecks[dirbin] = check
        checklayout.addStretch(1)
        # the list shows up to 6 bins, and a longer list scrolls
        shownbins = min(max(len(directionchecks), 1), 6)
        lineheight = max(check.sizeHint().height() for check in directionchecks.values()) if directionchecks else 20
        directionbox.setFixedHeight(shownbins * (lineheight + 2) + 10)
        directionbox.setWidget(checklist)
        shownchoices = (directionkind, usedegrees)
        return True

    def get_direction_choices_of_kind(directionkind: str, usedegrees: bool) -> list[tuple[int, str]]:
        if not directionkind:
            return []
        if (directionkind, usedegrees) not in directionchoices:
            directionchoices[directionkind, usedegrees] = get_direction_choices(
                viewer.runfolders[0], directionkind, usedegrees=usedegrees
            )
        return directionchoices[directionkind, usedegrees]

    # the time sliders have one position for each valid timestep, or SLIDER_STEPS positions for a continuous time
    slidermode: bool | None = None
    # the spectra of the runs that give the ranges of the time controls
    shownruns = viewer.runspectra

    def show_run_ranges() -> None:
        """Set the ranges of the time controls and the direction bins from the runs of the plot, e.g. after Add Model.

        The ranges of the sliders and the direction bins came from the models of the command only.
        """
        nonlocal logtrange, widthmax, nvalid, slidermode, shownchoices, shownruns
        logtrange = (math.log10(viewer.timebounds[0]), math.log10(viewer.timebounds[1]))
        widthmax = max((viewer.timebounds[1] - viewer.timebounds[0]) / 4.0, viewer.values.width)
        nvalid = len(viewer.validtimesteps)
        # show_values sets the ranges of the sliders again, and the direction bins come from the new first run
        slidermode, shownchoices = None, None
        directionchoices.clear()
        shownruns = viewer.runspectra

    def set_time_mode() -> None:
        nonlocal slidermode
        continuous = viewer.values.notimeclamp
        timeslider.setRange(0, SLIDER_STEPS if continuous else nvalid - 1)
        if continuous:
            widthslider.setRange(0, SLIDER_STEPS)
        else:
            widthslider.setRange(1, nvalid)
        widthlabel.setText("Δt [d]" if continuous else "Δ timesteps")
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
            values.spectra,
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

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        blockers = [QtCore.QSignalBlocker(widget) for widget in signalwidgets]
        values = viewer.values
        # Undo or a rejected change can give a different list of spectra, and its runs have different times
        if values.spectra != viewer.runspectra:
            viewer.load_runs(values.spectra)
        if viewer.runspectra != shownruns:
            show_run_ranges()
        if values.xunit != rangesunit:
            set_xunit_ranges()
        if values.notimeclamp != slidermode:
            set_time_mode()
        modesegments.setCurrentIndex(1 if values.notimeclamp else 0)
        previousbutton.setEnabled(viewer.step_time(-1) is not None)
        nextbutton.setEnabled(viewer.step_time(1) is not None)
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
        set_xrange_positions(
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
        emissionoptions.setVisible(values.showemission or values.showabsorption)
        nostackcheck.setEnabled(values.showemission or values.showabsorption)
        nostackcheck.setChecked(values.nostack)
        lockbutton.setChecked(bool(values.fixedionlist))
        lockbutton.setText(f"Lock Series ({len(values.fixedionlist)})" if values.fixedionlist else "Lock Series")
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
        directionkindbox.setCurrentIndex(directionkindbox.findData(values.directionkind))
        usedegreescheck.setChecked(values.usedegrees)
        usedegreescheck.setEnabled(bool(values.directionkind))
        isnewlist = show_direction_choices(values.directionkind, values.usedegrees)
        for dirbin, check in directionchecks.items():
            with QtCore.QSignalBlocker(check):
                check.setChecked(dirbin in values.directionbins)
        # a new list scrolls to the first checked bin, which can be far down a list of 100 bins
        if isnewlist and (firstcheck := directionchecks.get(values.directionbins[0] if values.directionbins else -1)):
            QtCore.QTimer.singleShot(0, window, partial(directionbox.ensureWidgetVisible, firstcheck))
        # all the directions have no bin to select, thus the list of the bins shows only for a kind of direction
        directionbox.setVisible(bool(values.directionkind))
        shownspectra = [
            spectralist.item(index).data(QtCore.Qt.ItemDataRole.UserRole) for index in range(spectralist.count())
        ]
        if shownspectra != list(values.spectra):
            spectralist.clear()
            for path in values.spectra:
                item = QtWidgets.QListWidgetItem(get_spectrum_item_text(path))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
                item.setToolTip(item.text())
                spectralist.addItem(item)
            # the list has the height of its spectra, from 2 to 4 lines, and a longer list scrolls
            shownlines = min(max(spectralist.count(), 2), 4)
            spectralist.setFixedHeight(shownlines * spectralist.fontMetrics().lineSpacing() + 12)
        set_option_rows(values.otheroptions)
        set_command_text(commandtext, viewer.get_command())
        set_command_text(pythontext, get_python_code(viewer.parser, viewer.get_plot_tokens()))
        show_rejections()
        for blocker in blockers:
            blocker.unblock()
        fit_canvas(canvas, viewer.figsize, plotarea)

    def after_draw(message: str | None) -> None:
        # matplotlib keeps the connections of the mouse in the figure, and each plot has a new figure
        connect_mouse_to_figure()
        # --showabsorption changes the height of the frames, thus the plot can need a new -figwidthscale
        fittimer.start()
        # each change starts the timer again, thus the full plot follows after the last change
        if viewer.drewpreview:
            fulldrawtimer.start()
        # a rejection occurs again at each step, thus a rejection stops the Play button
        if message is not None:
            playbutton.setChecked(False)
        elif playbutton.isChecked():
            # a draw that the Play button did not start also restarts the timer, thus one chain of steps stays
            start_play_timer(playtimer, queue.plotseconds, fpsbox.value())

    def is_slider_dragged() -> bool:
        """Return whether the user drags a slider. Only a drag draws a preview, because a drag gives many plots."""
        return timeslider.isSliderDown() or widthslider.isSliderDown() or bool(xrangeslider.property("dragging"))

    def render_plot(values: ControlValues) -> "Callable[[], str | None]":
        # the worker thread runs this function, thus it reads the drag state from the viewer and not from a widget
        return viewer.render(values, preview=viewer.dragging)

    def get_drawkind() -> str:
        return "Preview" if viewer.drewpreview else "Plot"

    queue = DrawQueue(
        window,
        viewer,
        statusbar,
        show_values,
        after_draw,
        get_drawkind=get_drawkind,
        render=render_plot,
        keep_on_undo=keep_figwidthscale,
    )

    def apply(values: ControlValues, *, undoable: bool = True) -> None:
        """Give the queue the new values, and draw a preview if a slider drag gives them."""
        viewer.dragging = is_slider_dragged() and get_bool_setting("dragpreview", default=True)
        queue.apply(viewer.clamp_time(values), undoable=undoable)

    def on_undo() -> None:
        viewer.dragging = False
        queue.undo()

    def on_redo() -> None:
        viewer.dragging = False
        queue.redo()

    def fit_figwidthscale() -> None:
        """Give the plot the -figwidthscale that fills the plot area."""
        figwidthscale = get_new_figwidthscale(
            plotarea, viewer.figsize, viewer.values.figwidthscale, viewer.get_fitted_figwidthscale
        )
        if figwidthscale is not None:
            # the window sets the width, thus Undo does not return to an old width
            apply(dc.replace(viewer.values, figwidthscale=figwidthscale), undoable=False)

    fittimer.timeout.connect(fit_figwidthscale)

    def draw_full() -> None:
        """Replace the preview with the plot of all the packets."""
        # a change in the queue, or a step of Play, draws a new plot in place of the preview
        if queue.requestedvalues is not None or playbutton.isChecked() or not viewer.drewpreview:
            return
        # the drag can give more previews, thus the full plot waits until the user releases the slider
        if is_slider_dragged():
            fulldrawtimer.start()
            return
        viewer.dragging = False
        queue.redraw()

    fulldrawtimer.timeout.connect(draw_full)

    def show_error(message: str) -> None:
        show_status_message(statusbar, message, "")
        show_values()

    def on_time_mode() -> None:
        values = viewer.values
        if modesegments.currentIndex() == 1 and not values.notimeclamp:
            apply(
                dc.replace(
                    values, notimeclamp=True, centre=float(f"{values.centre:.4g}"), width=float(f"{values.width:.3g}")
                )
            )
        elif modesegments.currentIndex() == 0 and values.notimeclamp:
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
        # a later plot can show new text in the fields only when they have no edit of the user
        timeedit.setModified(False)
        widthedit.setModified(False)
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
        apply(newvalues)

    def on_arrow(step: int) -> None:
        if (values := viewer.step_time(step)) is not None:
            apply(values)

    def on_widthstep(step: int) -> None:
        apply(viewer.step_width(step))

    def play_step() -> None:
        if not playbutton.isChecked():
            return
        # after the last timestep, Play starts again at the first timestep
        values = viewer.step_time(1) or viewer.move_to_end(last=False)
        # a time range that covers every valid timestep has no other step
        if values == viewer.values:
            playbutton.setChecked(False)
            return
        apply(values, undoable=False)

    def on_play(checked: bool) -> None:
        if checked:
            play_step()
        elif viewer.drewpreview:
            fulldrawtimer.start()

    def plot_shows_values() -> bool:
        """Return True if the plot on the screen has the values of the controls.

        DrawQueue gives the viewer the new values immediately, and the old plot stays until the worker draws the new
        one. A handler that reads the plot, e.g. the y limits or the series, must not put them into different values.
        """
        return queue.drawnvalues == viewer.values

    def plot_has_xunit() -> bool:
        """Return True if the x axis of the plot on the screen has the unit of the controls, e.g. during Play."""
        return queue.drawnvalues.xunit == viewer.values.xunit

    def set_xlimits(low: float, high: float) -> None:
        # a value of 3 significant digits gives a short command, and a text field gives an exact value
        low, high = float(f"{low:.3g}"), float(f"{high:.3g}")
        if low < high and plot_has_xunit():
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
        xminedit.setModified(False)
        xmaxedit.setModified(False)
        try:
            low, high = float(xminedit.text()), float(xmaxedit.text())
        except ValueError:
            low, high = math.nan, math.nan
        if not 0.0 <= low < high:
            show_error("Give two numbers of 0 or more, with the minimum less than the maximum")
            return
        values = dc.replace(viewer.values, xmin=format(low, ".10g"), xmax=format(high, ".10g"))
        apply(values)

    def on_fixy(checked: bool) -> None:
        if not checked:
            apply(dc.replace(viewer.values, ymin="", ymax=""))
            return
        if not plot_shows_values():
            show_error("The plot on the screen does not show the new values yet. Wait for the plot, then try again")
            return
        # the limits of the plot on the screen become the limits of the command, thus the plot does not change
        low, high = (get_short_number(limit) for limit in viewer.axes[0].get_ylim())
        apply(dc.replace(viewer.values, ymin=low, ymax=high))

    def on_yedit() -> None:
        yminedit.setModified(False)
        ymaxedit.setModified(False)
        try:
            low, high = float(yminedit.text()), float(ymaxedit.text())
        except ValueError:
            show_error("Give two numbers for -ymin and -ymax")
            return
        if not low < high:
            show_error("Give a -ymin that is less than -ymax")
            return
        values = dc.replace(viewer.values, ymin=format(low, ".10g"), ymax=format(high, ".10g"))
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
        apply(values)

    def on_binmode() -> None:
        # the box holds the width of the previous mode, thus it takes the width of the new mode before the values change
        with QtCore.QSignalBlocker(binwidthbox):
            set_binwidth_box(binmodebox.currentData() or "deltax")
        on_emission_options()

    def on_direction() -> None:
        directionkind: str = directionkindbox.currentData()
        usedegrees = usedegreescheck.isChecked()
        dirbins = [dirbin for dirbin, _ in get_direction_choices_of_kind(directionkind, usedegrees)]
        if directionkind == viewer.values.directionkind:
            directionbins = tuple(dirbin for dirbin, check in directionchecks.items() if check.isChecked())
            if directionkind and not directionbins:
                show_error("A kind of viewing direction needs one direction bin at least")
                return
        else:
            # a new kind keeps each direction bin that the kind also has
            directionbins = tuple(dirbin for dirbin in viewer.values.directionbins if dirbin in dirbins) or tuple(
                dirbins[:1]
            )
        values = dc.replace(
            viewer.values, directionkind=directionkind, directionbins=directionbins, usedegrees=usedegrees
        )
        apply(values)

    def on_lock(checked: bool) -> None:
        if not checked:
            apply(remove_series_lock(viewer.values))
            return
        if not plot_shows_values():
            show_error("The plot on the screen does not show the new values yet. Wait for the plot, then try again")
            return
        if not (series := viewer.get_drawn_series()):
            show_error("The plot has no series of contributions to lock")
            return
        apply(dc.replace(viewer.values, fixedionlist=series))

    def apply_spectra(spectra: "Sequence[str]") -> None:
        """Read the runs of a new list of spectra, and apply the list with a time that the runs have.

        The runs of the new list can cover a shorter time, e.g. after Add Model, thus the time moves inside it.
        """
        viewer.load_runs(spectra)
        values = dc.replace(viewer.values, spectra=tuple(spectra))
        apply(viewer.clamp_time(values) if values.notimeclamp else viewer.snap(values, *viewer.get_selection(values)))

    def add_spectra(paths: "Sequence[str]") -> None:
        """Add each spectrum whose full path the list does not hold yet, e.g. "." for the working folder."""
        spectra = viewer.values.spectra
        shown = {get_spectrum_path(path) for path in spectra}
        newpaths: list[str] = []
        for path in paths:
            if (fullpath := get_spectrum_path(path)) not in shown:
                shown.add(fullpath)
                newpaths.append(path)
        if not newpaths:
            show_error("The list of spectra already holds each of these spectra")
            return
        apply_spectra((*spectra, *newpaths))

    def on_add_model() -> None:
        startfolder = Path(viewer.runfolders[0]).absolute().parent
        folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Add an ARTIS model", str(startfolder))
        if not folder:
            return
        if not get_artis_run_folders([Path(folder)]):
            show_error(f"{folder} is not the folder of an ARTIS run, which holds input.txt and spec.out")
            return
        add_spectra([folder])

    def on_open_reference() -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(window, "Add reference spectra", str(referencefolder))
        add_spectra([get_reference_token(filename) for filename in filenames])

    def add_reference_name(name: str) -> None:
        name = name.strip()
        if not name:
            return
        if find_reference_spectrum_file_or_none(name) is None:
            show_error(f"No reference spectrum {name} is in the working folder or in the reference data of artistools")
            return
        referenceedit.clear()
        add_spectra([name])

    def on_complete_reference(name: str) -> None:
        # the completer puts the name in the field after this handler, thus clear the field after the event
        QtCore.QTimer.singleShot(0, referenceedit.clear)
        add_reference_name(name)

    def on_remove_spectra() -> None:
        selected = {item.data(QtCore.Qt.ItemDataRole.UserRole) for item in spectralist.selectedItems()}
        spectra = tuple(path for path in viewer.values.spectra if path not in selected)
        # the time controls read the timesteps of a run, thus the plot needs an ARTIS model
        if not get_artis_run_folders([Path(path) for path in spectra]):
            show_error("The plot needs one ARTIS model at least. Add a different model before you remove this one")
            return
        apply_spectra(spectra)

    def on_copy() -> None:
        copy_text(viewer.get_command())
        show_status_note(statusbar, "Copied the command")

    def get_figure_tokens() -> list[str]:
        """Return the command of the plot with no -dpi. The Figure section gives the resolution."""
        rows, _ = split_dpi_row(viewer.values.otheroptions, viewer.parser.get_default("dpi"))
        return viewer.get_plot_tokens(dc.replace(viewer.values, otheroptions=rows))

    def on_copy_figure() -> None:
        from artistools.spectra.plotspectra import main as plotspectra_main

        plottokens = get_figure_tokens()
        copy_figure_of_command(
            queue, statusbar, plotspectra_main, viewer.parser, plottokens, figuresection.get_choice()
        )

    def on_copy_python() -> None:
        copy_text(get_python_code(viewer.parser, viewer.get_plot_tokens()))
        show_status_note(statusbar, "Copied the Python code")

    def on_save() -> None:
        from artistools.spectra.plotspectra import main as plotspectra_main

        plottokens = get_figure_tokens()
        save_figure_of_command(
            window, statusbar, plotspectra_main, "plotspectra", plottokens, viewer.parser, figuresection.get_choice()
        )

    def on_open_model() -> None:
        if (message := open_model_window(window, open_window, windows)) is not None:
            show_error(message)

    def on_help() -> None:
        QtWidgets.QMessageBox.information(
            window, "Keys and mouse actions", get_keyboard_help(KEYBOARD_HELP_ROWS, menutexts)
        )

    def get_animation_frames() -> "tuple[int, Callable[[int], list[str]]]":
        """Return the count of the steps of Play, from the first valid timestep to the last, and the command of each."""
        values = viewer.values
        validtimesteps = list(viewer.validtimesteps)
        firstpos, lastpos = viewer.get_selection_positions()
        count = lastpos - firstpos

        def get_frame_tokens(index: int) -> list[str]:
            if values.notimeclamp:
                # a continuous range keeps its width and moves its middle to the middle of each timestep
                centre = float(f"{viewer.tmids[validtimesteps[index]]:.4g}")
                return viewer.get_plot_tokens(viewer.clamp_time(dc.replace(values, centre=centre)))
            return viewer.get_plot_tokens(viewer.snap(values, validtimesteps[index], validtimesteps[index + count]))

        return (len(validtimesteps) if values.notimeclamp else len(validtimesteps) - count), get_frame_tokens

    def on_export_animation() -> None:
        from artistools.spectra.plotspectra import main as plotspectra_main

        export_animation(
            window,
            queue,
            statusbar,
            plotspectra_main,
            "plotspectra",
            get_animation_frames(),
            fpsbox.value(),
            viewer.parser,
        )

    def on_plot_menu(_frameindex: int, _event: t.Any) -> None:
        """Show the actions on the figure under the pointer, as the context menu of a Mac app does."""
        menu = QtWidgets.QMenu(window)
        menu.addAction("Copy Figure").triggered.connect(on_copy_figure)
        menu.addAction("Save Figure…").triggered.connect(on_save)
        menu.addAction("Export Animation…").triggered.connect(on_export_animation)
        menu.exec(QtGui.QCursor.pos())
        # the window is the parent of the menu, thus without this the window keeps each menu until it closes
        menu.deleteLater()

    def on_open_recent(folder: str) -> None:
        if (message := open_model_folder(folder, open_window, windows)) is not None:
            show_error(message)

    def on_drop(paths: list[str]) -> None:
        """Add each dropped ARTIS run and each dropped reference file to the spectra of the plot."""
        folders = [path for path in paths if Path(path).is_dir()]
        runs = [folder for folder in folders if get_artis_run_folders([Path(folder)])]
        if len(runs) < len(folders):
            show_error("A dropped folder is not the folder of an ARTIS run, which holds input.txt and spec.out")
        add_spectra([*runs, *(get_reference_token(path) for path in paths if Path(path).is_file())])

    def on_closed() -> None:
        print(viewer.get_command())
        queue.close()
        # the list holds a reference to each open window, thus Python does not delete the window. A closed window
        # leaves the list
        windows.remove(window)

    menucallbacks = {
        "Open Model…": on_open_model,
        "Save Figure…": on_save,
        "Export Animation…": on_export_animation,
        "Close Window": window.close,
        "Undo": on_undo,
        "Redo": on_redo,
        "Copy Figure": on_copy_figure,
        "Copy Command": on_copy,
        "Copy Python": on_copy_python,
        "Keys and Mouse Actions": on_help,
    }
    menutexts = add_menus(window, menucallbacks, queue, playbutton, open_folder=on_open_recent)
    set_drop_handler(window, on_drop)
    follow_colour_scheme(window, viewer, queue)

    # the window keeps its command at a quit, and the next start opens the window again
    def get_session_tokens() -> list[str]:
        # a command with no path reads the working folder, and the next start can be in a different folder
        tokens = viewer.get_plot_tokens()
        return [str(Path.cwd()), *tokens] if viewer.values.spectra == DEFAULT_SPECTRA else tokens

    window.setProperty("sessiontokens", get_session_tokens)

    modesegments.currentChanged.connect(on_time_mode)
    previousbutton.clicked.connect(lambda: on_arrow(-1))
    nextbutton.clicked.connect(lambda: on_arrow(1))
    timeslider.valueChanged.connect(on_time)
    widthslider.valueChanged.connect(on_width)
    timeedit.editingFinished.connect(on_timeedit)
    widthedit.editingFinished.connect(on_timeedit)
    playbutton.toggled.connect(on_play)
    figuresection.copybutton.clicked.connect(on_copy_figure)
    figuresection.savebutton.clicked.connect(on_save)
    playtimer.timeout.connect(play_step)
    connect_xrange(on_xrange)
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
    usedegreescheck.toggled.connect(on_direction)
    addmodelbutton.clicked.connect(on_add_model)
    openreferencebutton.clicked.connect(on_open_reference)
    referencecompleter.activated.connect(on_complete_reference)
    referenceedit.returnPressed.connect(lambda: add_reference_name(referenceedit.text()))
    removebutton.clicked.connect(on_remove_spectra)
    copybutton.clicked.connect(on_copy)
    pythoncopybutton.clicked.connect(on_copy_python)
    statusbar.helpbutton.clicked.connect(on_help)
    window.destroyed.connect(on_closed)
    connect_mouse_to_figure = connect_plot_mouse(
        canvas,
        get_frames=lambda: [axis for axis in (*viewer.axes, viewer.residualaxis) if axis is not None],
        get_readout=lambda event, _frame: viewer.get_readout(event.xdata),
        readoutlabel=statusbar.readout,
        on_select=set_xlimits,
        on_reset=lambda: set_xlimits(*get_default_xlimits(viewer.values.xunit, gamma=viewer.args.gamma)),
        can_select=plot_has_xunit,
        on_menu=on_plot_menu,
        show_tag=make_readout_tag(canvas),
    )
    # a text field takes these keys while it has the focus, and the shortcuts apply otherwise
    for key, callback in (
        (QtCore.Qt.Key.Key_Left, lambda: on_arrow(-1)),
        (QtCore.Qt.Key.Key_Right, lambda: on_arrow(1)),
        (QtCore.Qt.Key.Key_Up, lambda: on_widthstep(1)),
        (QtCore.Qt.Key.Key_Down, lambda: on_widthstep(-1)),
        (QtCore.Qt.Key.Key_Home, lambda: apply(viewer.move_to_end(last=False))),
        (QtCore.Qt.Key.Key_End, lambda: apply(viewer.move_to_end(last=True))),
    ):
        QtGui.QShortcut(QtGui.QKeySequence(key), window).activated.connect(callback)

    show_window(window, viewer.figsize, lambda: fit_canvas(canvas, viewer.figsize, plotarea))
    show_values()
    return None
