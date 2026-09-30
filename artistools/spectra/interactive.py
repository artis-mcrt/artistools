"""Show the plot of plotspectra in a window with controls for the time, the x range, and the emission plot."""

import argparse
import contextlib
import dataclasses as dc
import math
import shlex
import typing as t
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import numpy as np
import polars as pl

from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
from artistools.misc import get_artis_run_folders
from artistools.misc import get_escaped_arrivalrange
from artistools.misc import get_file_metadata
from artistools.misc import get_model_name
from artistools.misc import get_time_range
from artistools.misc import get_time_range_text
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc.fileio import COMPRESSED_EXTENSIONS
from artistools.misc.fileio import resolve_modelpath
from artistools.misc.remote import is_remote_path
from artistools.packets.core import has_packets_files
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
from artistools.spectra.plotspectra import get_default_xlimits
from artistools.spectra.plotspectra import main as plotspectra_main
from artistools.spectra.plotspectra import make_plot_figure
from artistools.spectra.plotspectra import path_is_reference_spectrum
from artistools.spectra.plotspectra import resolve_plot_args
from artistools.viewertools import add_command_sections
from artistools.viewertools import add_default_options
from artistools.viewertools import add_direction_section
from artistools.viewertools import add_row
from artistools.viewertools import add_section
from artistools.viewertools import add_series_list
from artistools.viewertools import add_window_actions
from artistools.viewertools import add_y_axis_actions
from artistools.viewertools import add_y_limits_row
from artistools.viewertools import connect_plot_mouse
from artistools.viewertools import DirectionChoice
from artistools.viewertools import DrawQueue
from artistools.viewertools import edit_series_style
from artistools.viewertools import exit_for_other_actions
from artistools.viewertools import export_animation
from artistools.viewertools import finish_viewer_window
from artistools.viewertools import fit_canvas
from artistools.viewertools import get_changed_arguments
from artistools.viewertools import get_direction_kind
from artistools.viewertools import get_direction_kinds
from artistools.viewertools import get_fitted_figwidthscale
from artistools.viewertools import get_line_readouts
from artistools.viewertools import get_nearest_range_start
from artistools.viewertools import get_option_row_tokens
from artistools.viewertools import get_option_tokens
from artistools.viewertools import get_path_colours
from artistools.viewertools import get_python_call
from artistools.viewertools import get_row_values
from artistools.viewertools import get_series_style
from artistools.viewertools import get_series_value
from artistools.viewertools import get_short_number
from artistools.viewertools import make_command_tokens
from artistools.viewertools import make_figscale_box
from artistools.viewertools import make_fps_box
from artistools.viewertools import make_parser
from artistools.viewertools import make_play_button
from artistools.viewertools import make_play_row
from artistools.viewertools import make_range_slider
from artistools.viewertools import make_readout_tag
from artistools.viewertools import make_row_layout
from artistools.viewertools import make_segmented_control
from artistools.viewertools import make_series_swatch
from artistools.viewertools import make_slider
from artistools.viewertools import make_step_button
from artistools.viewertools import make_timer
from artistools.viewertools import make_xscale_box
from artistools.viewertools import move_series_styles
from artistools.viewertools import OptionRows
from artistools.viewertools import parse_command_tokens
from artistools.viewertools import parse_viewer_tokens
from artistools.viewertools import ReferenceData
from artistools.viewertools import reload_runs
from artistools.viewertools import render_command
from artistools.viewertools import ROW_SPACING
from artistools.viewertools import run_command_step
from artistools.viewertools import run_viewer_application
from artistools.viewertools import SERIES_STYLE_FLAGS
from artistools.viewertools import SeriesListActions
from artistools.viewertools import SeriesRow
from artistools.viewertools import set_command_text
from artistools.viewertools import set_edit_text
from artistools.viewertools import set_row_values
from artistools.viewertools import set_series_rows
from artistools.viewertools import set_spin_value
from artistools.viewertools import show_status_message
from artistools.viewertools import show_status_note
from artistools.viewertools import SLIDER_STEPS
from artistools.viewertools import start_play_timer
from artistools.viewertools import start_viewer_window
from artistools.viewertools import ViewerCommand

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.axes as mplax
    import numpy.typing as npt
    from PySide6 import QtWidgets

# the controls of the window give these arguments, thus the command drops the values that the user typed
CONTROLLED_DESTS: t.Final = frozenset({
    # the Resolution box of the Figure section gives -dpi
    "dpi",
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
    "gamma",
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


type DataSource = t.Literal["auto", "text", "packets"]

# the rule of the width of a continuous time range:
# - "dlogt" takes the width that gives ln(t_end / t_start) = dlogt, as the logarithmic timesteps of ARTIS do;
# - "days" keeps the width in days.
type WidthMode = t.Literal["dlogt", "days"]


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class ControlValues:
    """The values of the controls of the viewer, which give the options of the plotspectra command.

    The x limits and the bin width keep the text of the command, and an empty deltax gives no option.
    """

    centre: float
    # the width [d] of the time range. A continuous range has a width above 0, because a width of 0 selects the
    # whole timestep that holds the time, which is the clamped range
    width: float
    notimeclamp: bool
    # the rule of the width of a continuous range. The command gives the width that the rule gives, thus the mode
    # itself is not in the command
    widthmode: WidthMode
    # the Δ ln t of the width mode "dlogt". It starts with the Δ ln t of a logarithmic grid of the run
    dlogt: float
    # True for the gamma-ray spectrum of the gamma packets, and False for the UVOIR spectrum of the r-packets
    gamma: bool
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
    # "packets" gives --frompackets. "auto" and "text" give no flag, but "text" rejects an option that needs the
    # packets files
    datasource: DataSource
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
    # the path of the ARTIS model that gives the timesteps of the time controls, or "" for the first model. The
    # command gives days, thus the choice is not in the command
    timegrid: str
    figwidthscale: float
    # the resolution of a PNG file (-dpi), or None for the default of the command
    dpi: int | None
    otheroptions: OptionRows


def get_default_xunit(*, gamma: bool) -> str:
    """Return the x unit that plotspectra takes when the command gives no -xunit."""
    return "kev" if gamma else "angstroms"


def get_text_source_conflict(values: "ControlValues", plotargs: argparse.Namespace) -> str | None:
    """Return why the plot needs the packets files when the user selected the text files, or None."""
    if values.datasource == "text" and plotargs.frompacketsreason is not None:
        return (
            f"{plotargs.frompacketsreason} needs the packets files, and the data source is Text files. Select Auto"
            " or Packets files"
        )
    return None


def get_packets_reason(tokens: "Sequence[str]") -> str | None:
    """Return the option that makes plotspectra read the packets files, or None if the text files serve the plot.

    A command that plotspectra rejects gives None.
    """
    reasons: list[str | None] = []

    def check() -> None:
        plotargs = parse_cli_args(addargs, None, None, tokens)
        resolve_plot_args(plotargs)
        reasons.append(plotargs.frompacketsreason)

    run_command_step(check, echo=False)
    return reasons[0] if reasons else None


def get_thermal_emission_reason(values: "ControlValues") -> str | None:
    """Return why the choice of the last thermal emission does not change the plot, or None if it changes the plot."""
    if values.gamma:
        return "A gamma packet has no thermal emission"
    if (groupby := values.groupby or get_default_groupby(gamma=values.gamma)) in {"nuc", "nucmass"}:
        return f"-groupby {groupby} takes the nuclide of the pellet, and not an emission"
    if not values.showemission:
        return "An absorption always takes the last interaction, thus only an emission plot reads this choice"
    return None


def get_default_groupby(*, gamma: bool) -> str:
    """Return the -groupby that plotspectra takes for an emission plot when the command gives none."""
    return "nuc" if gamma else "ion"


def set_packet_type(values: ControlValues, *, gamma: bool) -> ControlValues:
    """Return the values for the spectrum of the r-packets, or of the gamma packets if gamma is True.

    The two spectra have different units, x ranges, and series, thus the x unit, the x range, the y range, the
    grouping, and the locked series return to the defaults of the new spectrum. The data source stays. For a run with
    no gamma_spec.out, plotspectra reads the packets of a gamma-ray spectrum.
    """
    xunit = get_default_xunit(gamma=gamma)
    xmin, xmax = get_default_xlimits(xunit, gamma=gamma)
    return dc.replace(
        values,
        gamma=gamma,
        usethermalemissiontype=values.usethermalemissiontype and not gamma,
        xunit=xunit,
        xmin=format(xmin, ".10g"),
        xmax=format(xmax, ".10g"),
        ymin="",
        ymax="",
        groupby=None,
        fixedionlist=(),
    )


def has_gamma_spectrum(runfolders: "Sequence[Path]") -> bool:
    """Return True if one type of file gives the gamma-ray spectrum of each run.

    plotspectra reads gamma_spec.out of each run, or the packets of each run when a run has no gamma_spec.out.
    Thus a run with gamma_spec.out alone and a run with packets alone give no plot. A run can keep only the parquet
    cache of its packets.
    """
    return all(
        firstexisting_or_none("gamma_spec.out", folder=runfolder) is not None for runfolder in runfolders
    ) or all(has_packets_files(runfolder) for runfolder in runfolders)


class RunTimes(t.NamedTuple):
    """The times of the timesteps of a run [d], and the times that plotspectra accepts for the run [d]."""

    tstart: float
    tend: float
    validstart: float
    validend: float


def get_run_times(runfolder: Path, *, plotinvalidpart: bool) -> RunTimes:
    """Return the times of the timesteps of a run, and the times inside them that plotspectra accepts.

    plotspectra accepts only the arrival times at which light from the whole model reaches the observer, unless the
    command gives --plotinvalidpart.
    """
    tstart = get_timestep_times(runfolder, loc="start")[0]
    tend = get_timestep_times(runfolder, loc="end")[-1]
    validstart, validend = tstart, tend
    if not plotinvalidpart:
        with contextlib.suppress(FileNotFoundError):
            _, arrivalstart, arrivalend = get_escaped_arrivalrange(runfolder)
            if arrivalstart is not None:
                validstart = max(validstart, float(arrivalstart))
            if arrivalend is not None:
                validend = min(validend, float(arrivalend))
    return RunTimes(tstart=tstart, tend=tend, validstart=validstart, validend=validend)


def fits_each_run(runfoldertimes: "Sequence[tuple[Path, RunTimes]]", timedays: str) -> bool:
    """Return True if each run clamps the -timedays value to timesteps inside the valid times of the run.

    plotspectra clamps the time to the timesteps of each run, and then it rejects days outside the valid times.
    """
    for runfolder, runtimes in runfoldertimes:
        try:
            _, _, daysmin, daysmax = get_time_range(runfolder, timedays_range_str=timedays, clamp_to_timesteps=True)
        except ValueError:
            return False
        if daysmin < runtimes.validstart or daysmax > runtimes.validend:
            return False
    return True


def get_run_times_text(runtimes: RunTimes) -> str:
    """Return the times of a run for the tooltip of its row in the list of spectra."""
    text = f"Timesteps from {runtimes.tstart:.4g} to {runtimes.tend:.4g} d."
    if (runtimes.validstart, runtimes.validend) != (runtimes.tstart, runtimes.tend):
        text += f" The escaped packets give valid times from {runtimes.validstart:.4g} to {runtimes.validend:.4g} d."
    return text


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


class RenderedSpectrum(t.NamedTuple):
    """The frames and the data of a plot that the worker thread drew, which the window reads."""

    axes: "npt.NDArray[t.Any]"
    residualaxis: "mplax.Axes | None"
    dfalldata: pl.DataFrame


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

    Two spellings of one spectrum, e.g. "." and the full path of the working folder, then give the same path. A model
    on a different host keeps its path.
    """
    if path_is_reference_spectrum(path):
        return (find_reference_spectrum_file_or_none(path) or Path(path)).resolve()
    return resolve_modelpath(path)


def get_spectrum_item_text(path: str) -> str:
    """Return the text of a path in the list of spectra: the kind and the full path.

    The list shortens a long path in the middle, thus the text keeps the start and the end of the path.
    """
    if path_is_reference_spectrum(path):
        return f"Reference: {(find_reference_spectrum_file_or_none(path) or Path(path)).absolute()}"
    return f"Model: {path if is_remote_path(path) else Path(path).absolute()}"


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
        parser, args, startpaths, otheroptions, self.helptexts = parse_viewer_tokens(addargs, tokens, CONTROLLED_DESTS)
        # resolve_frompackets gives an emission plot a default -groupby, thus the value comes from the arguments
        givengroupby: str | None = args.groupby
        # -deltax and --notimeclamp also make plotspectra read the packets, thus only a --frompackets that the user
        # gave selects the packets files, and the other commands start with the automatic choice
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
        self.defaultyscale: str = parser.get_default("defaultyscale")
        self.defaultyvariable: str = parser.get_default("yvariable")
        # the time of the command stays exact, because a rounded time can select a different timestep
        values = ControlValues(
            centre=centre,
            width=width,
            notimeclamp=bool(args.notimeclamp),
            # a continuous range of the command keeps its width in days. A continuous single time takes Δ ln t,
            # because a width of 0 reads the whole timestep, which is the clamped range
            widthmode="days" if width > 0.0 else "dlogt",
            dlogt=self.dlogt,
            gamma=bool(args.gamma),
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
            groupby=None if givengroupby == get_default_groupby(gamma=bool(args.gamma)) else givengroupby,
            maxseriescount=args.maxseriescount,
            nostack=bool(args.nostack),
            deltax="" if args.deltax is None else format(args.deltax, ".10g"),
            deltalogx="" if args.deltalogx is None else format(args.deltalogx, ".10g"),
            datasource="packets" if givesfrompackets else "auto",
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
            timegrid="",
            figwidthscale=args.figwidthscale,
            dpi=None if args.dpi == parser.get_default("dpi") else args.dpi,
            otheroptions=otheroptions,
        )
        self.values = self.clamp_time(values) if values.notimeclamp else self.snap(values, *self.get_selection(values))
        # the list of spectra and the time grid of the runs of load_runs. After a change of the list, e.g. Add Model
        # or Undo, load_runs reads the runs again. At the start, the runs of the command are the runs of the values,
        # thus load_runs does not read them again
        self.runkey = (self.values.spectra, self.values.timegrid)

        self.fig = fig
        self.axes: npt.NDArray[t.Any] = np.empty(0, dtype=object)
        self.residualaxis: mplax.Axes | None = None
        # the colours of the window in Dark Mode, which the window sets and the worker thread reads
        self.darkcolours: tuple[str, str] | None = None
        # the last warning of the last plot, which the status bar shows
        self.warning = ""
        # a window can change the size of the figure, thus the size of the frames stays here
        self.figsize: tuple[float, float] = (0.0, 0.0)
        # the readout of the window reads the contributions of an emission plot from this frame
        self.dfalldata = pl.DataFrame()

    def load_runs(self, spectra: "Sequence[str | Path]", timegrid: str = "") -> None:
        """Read the timesteps of the ARTIS runs of the spectra, and the times that are valid for all the runs.

        The time controls take the timesteps of the run of timegrid, or of the first run if timegrid names no model.
        plotspectra rejects a time outside the timesteps of a run, and a time outside the arrival times of the escaped
        packets of a run. Thus the controls stay inside the times that are valid for all the runs. With
        --plotinvalidpart, plotspectra accepts all the arrival times. A reference spectrum has no run.
        """
        runfolders = get_artis_run_folders([Path(path) for path in spectra])
        gridfolder = next(iter(get_artis_run_folders([Path(timegrid)] if timegrid else [])), runfolders[0])
        tmids = get_timestep_times(gridfolder, loc="mid")
        tstarts = get_timestep_times(gridfolder, loc="start")
        tends = get_timestep_times(gridfolder, loc="end")
        timebounds = [tstarts[0], tends[-1]]
        self.runtimes: dict[str, RunTimes] = {}
        runfoldertimes: list[tuple[Path, RunTimes]] = []
        for path in spectra:
            for runfolder in get_artis_run_folders([Path(path)]):
                runtimes = get_run_times(runfolder, plotinvalidpart=bool(self.args.plotinvalidpart))
                self.runtimes[str(path)] = runtimes
                runfoldertimes.append((runfolder, runtimes))
                timebounds = [max(timebounds[0], runtimes.validstart), min(timebounds[1], runtimes.validend)]
        self.runfolders, self.gridfolder = runfolders, gridfolder
        self.tmids, self.tstarts, self.tends = tmids, tstarts, tends
        self.twidths = get_timestep_times(gridfolder, loc="delta")
        # the Δ ln t of a logarithmic grid with the same start, end, and count of timesteps. A constant grid or a
        # hybrid grid of ARTIS has a different Δ ln t in each timestep, and the width mode "dlogt" starts with this one
        self.dlogt = float(f"{math.log(tends[-1] / tstarts[0]) / len(tmids):.4g}")
        self.timebounds = (timebounds[0], timebounds[1])
        # a different run clamps the time of a timestep to its own timestep, which can be longer and can end outside
        # the valid times of that run. Thus each run tests the time of each timestep
        self.validtimesteps = [
            timestep
            for timestep in range(len(tmids))
            if tstarts[timestep] >= self.timebounds[0]
            and tends[timestep] <= self.timebounds[1]
            and fits_each_run(runfoldertimes, get_snapped_timedays_argument(tmids, tstarts, tends, timestep, timestep))
        ] or list(range(len(tmids)))
        self.hasgammaspectrum = has_gamma_spectrum(runfolders)
        # the direction controls read the first run, e.g. for the observers of -plotvspecpol
        self.directionkinds = get_direction_kinds(runfolders[0])
        self.runkey = (tuple(str(path) for path in spectra), timegrid)

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
        if values.gamma:
            options.append("--gamma")
        if values.xunit != get_default_xunit(gamma=values.gamma):
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
        if values.datasource == "packets":
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
        if values.dpi is not None:
            options += ["-dpi", str(values.dpi)]
        # a list option takes each word that follows it, thus it comes after every other option
        if values.fixedionlist and (values.showemission or values.showabsorption):
            options += ["-fixedionlist", *values.fixedionlist]
        # a command with no path reads the model in the working folder, thus that model needs no path
        paths = [] if values.spectra == DEFAULT_SPECTRA else list(values.spectra)
        return make_command_tokens([*paths, *get_option_row_tokens(values.otheroptions)], options)

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotspectra", *self.get_plot_tokens()])

    def get_time_range_text(self) -> str:
        """Return the time range that the plot reads.

        A snapped range reads whole timesteps from spec.out. A continuous range reads the packets that arrive inside
        it, thus the text gives its days and its width and no timestep.
        """
        # a path does not start with "-", and the -t of the controls comes before each other option
        plottokens = self.get_plot_tokens()
        timedays = plottokens[plottokens.index("-t") + 1]
        timestepmin, timestepmax, daysmin, daysmax = get_time_range(
            self.gridfolder, timedays_range_str=timedays, clamp_to_timesteps=not self.values.notimeclamp
        )
        return get_time_range_text(timestepmin, timestepmax, daysmin, daysmax, clamped=not self.values.notimeclamp)

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
            # a step gives a width in days, thus the width mode becomes "days". A width of 0 or less stays out
            width = self.values.width + step * self.twidths[self.validtimesteps[self.get_nearest_position()]]
            if not float(f"{width:.3g}") > 0.0:
                return self.values
            return dc.replace(self.values, widthmode="days", width=float(f"{width:.3g}"))
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
            return get_text_source_conflict(values, plotargs)

        return run_command_step(check, echo=False)

    def draw(self, *, quiet: bool = True) -> str | None:
        """Draw the plot of the values, and return the reason for the status line if plotspectra rejects it.

        The terminal shows the whole error, and the status line shows its first line.
        """
        return self.render(self.values, quiet=quiet)()

    def render(self, values: ControlValues, *, quiet: bool = True) -> "Callable[[], str | None]":
        """Draw the plot of the values on a new figure, and return the function that shows it in the canvas.

        The function returns the reason for the status line if plotspectra rejects the values, and the old plot then
        stays. A worker thread can run this method, because it changes nothing that the window reads. The function
        that it returns must run in the thread of the window.
        """

        def draw(fig: mplfig.Figure) -> RenderedSpectrum | str:
            plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens(values))
            resolve_plot_args(plotargs)
            check_viewer_args(plotargs)
            if (conflict := get_text_source_conflict(values, plotargs)) is not None:
                return conflict
            if (plotargs.showemission, plotargs.showabsorption) != (values.showemission, values.showabsorption):
                return "A different option of the command keeps the emission plot on"
            _, axes, residualaxis = make_plot_figure(plotargs, fig=fig)
            dfalldata, _ = draw_plot(plotargs, axes, residualaxis)
            return RenderedSpectrum(axes=axes, residualaxis=residualaxis, dfalldata=dfalldata)

        def keep(plot: RenderedSpectrum) -> None:
            self.axes, self.residualaxis, self.dfalldata = plot

        return render_command(self, draw, keep, quiet=quiet)

    def get_fitted_figwidthscale(self, areawidth: float, areaheight: float) -> float:
        """Return the -figwidthscale that gives the figure the shape of the plot area."""
        marginwidth = LABELWIDTH_INCHES + RIGHTMARGIN_INCHES
        return get_fitted_figwidthscale(self.figsize, self.values.figwidthscale, marginwidth, areawidth, areaheight)

    def clamp_time(self, values: ControlValues) -> ControlValues:
        """Return the values with a continuous time inside the valid times, and the width that its width mode gives.

        Each change of the values passes here, thus the width follows the time, e.g. during a drag of the time
        slider or during Play. The width is never 0, because a width of 0 reads the whole timestep.
        """
        if not values.notimeclamp:
            return values
        low, high = self.timebounds
        centre = min(max(values.centre, low), high)
        widthmode = "dlogt" if values.widthmode == "days" and not values.width > 0.0 else values.widthmode
        width = values.width
        if widthmode == "dlogt":
            # this width gives (centre + width / 2) / (centre - width / 2) = exp(dlogt). The rounding never makes the
            # width larger, thus the start stays above 0
            exactwidth = 2.0 * centre * math.tanh(values.dlogt / 2.0)
            width = min(float(f"{exactwidth:.4g}"), exactwidth)
        if (centre, width, widthmode) == (values.centre, values.width, values.widthmode):
            return values
        return dc.replace(values, centre=centre, width=width, widthmode=widthmode)

    def change(self, values: ControlValues) -> str | None:
        """Draw the plot of the new values, and keep the old values and the old plot if plotspectra rejects them."""
        values = self.clamp_time(values)
        message = self.render(values)()
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


def get_series_name(path: str) -> str:
    """Return the legend name of a spectrum that has no -label, with no time in the name."""
    if path_is_reference_spectrum(path):
        filepath = find_reference_spectrum_file_or_none(path)
        metadata = get_file_metadata(filepath) if filepath is not None else {}
        return str(metadata.get("label", Path(path).name))
    return get_model_name(path)


def get_series_colours(spectra: "Sequence[str]", rows: OptionRows) -> dict[str, str]:
    """Return the colour of the plot of each spectrum, as plotspectra gives it."""
    return get_path_colours(spectra, [path_is_reference_spectrum(path) for path in spectra], rows)


def set_series_values(values: ControlValues, path: str, changes: "Mapping[str, str | None]") -> ControlValues:
    """Return the values with the value of each series style option in changes for one spectrum.

    changes gives each option by its flag, e.g. -label. A value of None gives the spectrum the default of plotspectra,
    and the other spectra keep their values.
    """
    return dc.replace(values, otheroptions=set_series_rows(values.otheroptions, values.spectra, path, changes))


def set_runs(viewer: SpectrumViewer, spectra: "Sequence[str]", timegrid: str) -> ControlValues:
    """Read the runs of a list of spectra, and return the values of the list with a time that the runs have.

    timegrid names the model that gives the timesteps of the time controls, or "" for the first model. A new time
    grid, e.g. after a change of the order, keeps the middle and the count of timesteps of a snapped range. A
    continuous range keeps its days. Each series style option, e.g. -label, stays on its spectrum.
    """
    first, last = viewer.get_selection(viewer.values)
    if timegrid not in spectra:
        timegrid = ""
    viewer.load_runs(spectra, timegrid)
    values = dc.replace(
        viewer.values,
        spectra=tuple(spectra),
        timegrid=timegrid,
        otheroptions=move_series_styles(viewer.values.otheroptions, viewer.values.spectra, spectra),
    )
    # a new first run can have no observers of the virtual packets, thus the kind of viewing direction can go
    if values.directionkind not in viewer.directionkinds:
        values = dc.replace(values, directionkind="", directionbins=())
    if values.notimeclamp:
        return viewer.clamp_time(values)
    count = min(last - first + 1, len(viewer.validtimesteps))
    start = get_nearest_range_start(
        [viewer.tmids[timestep] for timestep in viewer.validtimesteps], values.centre, count
    )
    return viewer.snap(values, viewer.validtimesteps[start], viewer.validtimesteps[start + count - 1])


def get_icon_curve() -> "npt.NDArray[np.float64]":
    """Return the curve of the icon of the viewer, which is a spectrum with two absorption lines."""
    xvalues = np.linspace(0.1, 0.9, 200)
    return 0.72 - 0.45 * np.exp(-(((xvalues - 0.42) / 0.06) ** 2)) - 0.25 * np.exp(-(((xvalues - 0.65) / 0.09) ** 2))


# the keys and the mouse actions of the window. get_keyboard_help adds the shortcuts of the menus
KEYBOARD_HELP_ROWS: t.Final = (
    ("<b>Left</b>, <b>Right</b>", "Move the time to the adjacent timestep"),
    ("<b>Up</b>, <b>Down</b>", "Make the time range one timestep wider or narrower"),
    ("<b>Home</b>, <b>End</b>", "Move the time to the first or the last valid timestep"),
    ("<b>Alt-Up</b>, <b>Alt-Down</b> in the list of spectra", "Move the spectrum up or down (Option on a Mac)"),
    ("<b>Double-click</b> a spectrum", "Give the spectrum a -label"),
    ("<b>Right-click</b> a spectrum", "Move it, use its timesteps, copy its path, or open its folder"),
    ("<b>Drag</b> across the plot", "Select the x range"),
    ("<b>Shift-drag</b> up or down the plot", "Select the y range (-ymin and -ymax)"),
    ("<b>Double-click</b> the plot", "Get the default x range"),
    ("<b>Right-click</b> the plot", "Change the y scale, get the automatic y range, or copy or save the figure"),
)


def run_viewer(tokens: "Sequence[str]") -> None:
    """Open the window of the viewer, and print the command of the last plot when the window closes.

    The Dock icon also takes a file, e.g. a reference spectrum, which the active window adds to its spectra.
    """
    run_viewer_application(APPLICATION_NAME, get_icon_curve(), open_window, tokens, ("public.folder", "public.data"))


def open_window(tokens: "Sequence[str]", windows: "list[QtWidgets.QMainWindow]") -> str | None:
    """Open a window of the viewer for the plotspectra arguments in tokens, or return the reason for no window."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    from artistools.commands import get_path

    # the Settings window can give a new window options, e.g. -figscale, that the command does not give
    viewer = SpectrumViewer(add_default_options(make_parser(addargs), tokens), mplfig.Figure())
    # a command with no path reads the model of the working folder
    modelnames = [resolve_modelpath(path).name for path in viewer.modelpathtokens] or [
        resolve_modelpath(viewer.runfolders[0]).name
    ]
    viewerwindow = start_viewer_window(
        APPLICATION_NAME, viewer, viewer.draw, windows, (viewer.runfolders[0], ", ".join(modelnames))
    )
    if isinstance(viewerwindow, str):
        return viewerwindow
    window, canvas, plotarea, panellayout, fittimer = viewerwindow
    playtimer = make_timer(window, 0)

    # each continuous slider maps its position from 0 to SLIDER_STEPS onto the range of its value
    def to_position(value: float, low: float, high: float) -> int:
        return round(SLIDER_STEPS * (min(max(value, low), high) - low) / max(high - low, 1e-300))

    def from_position(position: int, low: float, high: float) -> float:
        return low + (high - low) * position / SLIDER_STEPS

    helptexts = viewer.helptexts
    logtrange = (math.log10(viewer.timebounds[0]), math.log10(viewer.timebounds[1]))
    widthmax = max((viewer.timebounds[1] - viewer.timebounds[0]) / 4.0, viewer.values.width)
    dlogtmax = max(math.log(viewer.timebounds[1] / viewer.timebounds[0]) / 4.0, viewer.values.dlogt)
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
    widthlabel = QtWidgets.QLabel("Δ timesteps:")
    # a continuous range takes this box in place of the label of the width
    widthmodebox = QtWidgets.QComboBox()
    for widthmode, widthmodetext, widthmodetip in (
        (
            "dlogt",
            "Δ ln t",
            (
                "The range has ln(t_end / t_start) = Δ ln t, as the logarithmic timesteps of ARTIS do. The field gives"
                " Δ ln t, and it starts with the Δ ln t of a logarithmic grid with the timesteps of the run"
            ),
        ),
        ("days", "Δt", "Δt is a width in days. The field gives the width"),
    ):
        widthmodebox.addItem(widthmodetext, widthmode)
        widthmodebox.setItemData(widthmodebox.count() - 1, widthmodetip, QtCore.Qt.ItemDataRole.ToolTipRole)
    widthmodebox.setToolTip(
        "The rule of the width Δt of the continuous time range. Δt is never 0, because a width of 0 reads the whole"
        " timestep"
    )
    timestepslabel = QtWidgets.QLabel()
    playbutton = make_play_button(
        "Move the time through the valid timesteps of the run, and start again after the last timestep (Space)"
    )
    fpsbox = make_fps_box()
    previousbutton, nextbutton = make_step_button(forward=False), make_step_button(forward=True)
    timetip = "The middle of the time range in days. The Left key and the Right key move it to the adjacent timestep."
    widthtip = (
        'The width of the time range. With "Snap to Timesteps", the width is a count of timesteps. A continuous range'
        " takes the rule of the box on the left: a width Δ ln t in ln t, or a width Δt in days."
        " The Up key and the Down key change the width by one timestep."
    )
    for row, (label, slider, edit, tip) in enumerate(
        [
            (QtWidgets.QLabel("Time [d]:"), timeslider, timeedit, timetip),
            (widthlabel, widthslider, widthedit, widthtip),
        ],
        start=1,
    ):
        edit.setFixedWidth(110)
        for widget in (slider, edit):
            widget.setToolTip(tip)
        timegrid.addWidget(label, row, 0)
        timegrid.addWidget(slider, row, 1)
        timegrid.addWidget(edit, row, 2)
    timegrid.addWidget(widthmodebox, 2, 0)
    timegrid.addLayout(make_play_row([previousbutton, nextbutton], timestepslabel, fpsbox, playbutton), 3, 0, 1, -1)
    # the item data of each choice is the path of a model. The value "" of ControlValues.timegrid selects the first
    # model, thus the box then shows that model
    timegridbox = QtWidgets.QComboBox()
    timegridbox.setToolTip(
        "The ARTIS model that gives the timesteps of the time controls. The command gives the time in days, thus"
        " this choice does not change the command. The first ARTIS model in the list of spectra gives them until"
        " you select a different model"
    )
    timegridbox.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    timegridbox.setMinimumContentsLength(12)
    validtimeslabel = QtWidgets.QLabel()
    validtimeslabel.setToolTip(
        "The times that plotspectra accepts for all the models of the plot. Each time is inside the timesteps and"
        " inside the arrival times of the escaped packets of each model. The time controls stay inside these times"
    )
    gridrow = QtWidgets.QHBoxLayout()
    gridrow.addWidget(QtWidgets.QLabel("Timesteps of:"))
    gridrow.addWidget(timegridbox, 1)
    gridrow.addWidget(validtimeslabel)
    timegrid.addLayout(gridrow, 4, 0, 1, -1)

    # the settings keep the open state of the section by its key, which the older heading of the section gave
    _, xgrid = add_section(panellayout, "Horizontal axis", key="x axis")
    xunitbox = QtWidgets.QComboBox()
    xunitbox.addItems(list(XUNITS))
    xunitbox.setToolTip(helptexts.get("xunit", ""))
    xscalebox = make_xscale_box(helptexts)
    add_row(xgrid, 0, [QtWidgets.QLabel("-xunit"), xunitbox, QtWidgets.QLabel("x scale:"), xscalebox])
    xrangeslider, set_xrange_positions, connect_xrange, _ = make_range_slider(SLIDER_STEPS)
    xminedit, xmaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    # the label gives the quantity and the unit of the x axis, e.g. "Wavelength [Å]", and a new unit changes it
    xrangelabel = QtWidgets.QLabel()
    zoomtip = " Drag across the plot to select a range. Double-click the plot to get the default range."
    xrangeslider.setToolTip("The minimum and the maximum of the x axis." + zoomtip)
    for column, (widget, dest) in enumerate((
        (xrangelabel, ""),
        (xminedit, "xmin"),
        (xrangeslider, ""),
        (xmaxedit, "xmax"),
    )):
        if dest:
            widget.setFixedWidth(110)
            widget.setToolTip(helptexts.get(dest, "") + zoomtip)
        xgrid.addWidget(widget, 1, column)
    xgrid.setColumnStretch(2, 1)

    # the "Default bins" item gives no -deltax and no -deltalogx, thus plotspectra uses its own bins
    binmodebox = QtWidgets.QComboBox()
    for binmode, binmodetext in (("", "Default bins"), ("deltax", "-deltax"), ("deltalogx", "-deltalogx")):
        binmodebox.addItem(binmodetext, binmode)
        binmodebox.setItemData(binmodebox.count() - 1, helptexts.get(binmode, ""), QtCore.Qt.ItemDataRole.ToolTipRole)
    binmodebox.setToolTip("The bins of a spectrum from the packets files. The text files of exspec have their own bins")

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
    add_row(xgrid, 2, [QtWidgets.QLabel("Bins:"), binmodebox, binwidthbox])

    # the settings keep the open state of the section by its key, which the older heading of the section gave
    _, axesgrid = add_section(panellayout, "Vertical axis", key="y-axis")
    yscalebox = QtWidgets.QComboBox()
    # each item holds its -yscale choice, because the text of the "auto" item gives the scale of the drawn plot
    for yscale in viewer.yscalechoices:
        yscalebox.addItem(yscale.capitalize(), yscale)
    # the text of the "auto" item changes after each plot, and the box keeps a width for the longest text
    yscalebox.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
    yscalebox.setToolTip(helptexts.get("yscale", ""))
    # the handlers come later in this function, thus the lambdas read them at the time of a change
    show_y_limits = add_y_limits_row(
        axesgrid,
        1,
        helptexts,
        lambda ymin, ymax: apply(dc.replace(viewer.values, ymin=ymin, ymax=ymax)),
        lambda: viewer.axes[0].get_ylim() if plot_shows_values() and len(viewer.axes) else None,
        lambda message: show_error(message),  # ruff:ignore[unnecessary-lambda]
    )
    yvariablebox = QtWidgets.QComboBox()
    yvariablebox.addItems(viewer.yvariablechoices)
    normalisedcheck = QtWidgets.QCheckBox("--normalised")
    for widget, dest in ((yvariablebox, "yvariable"), (normalisedcheck, "normalised")):
        widget.setToolTip(helptexts.get(dest, ""))
    # the index of an item: 0 for the UVOIR spectrum of the r-packets, and 1 for the gamma packets (--gamma)
    packetbox = QtWidgets.QComboBox()
    gammatooltip = f"--gamma: {helptexts.get('gamma', '')}"
    for text, tooltip in (
        ("UVOIR", "The ultraviolet, optical, and infrared (UVOIR) spectrum of the radiation packets (r-packets)"),
        ("\N{GREEK SMALL LETTER GAMMA}-rays", gammatooltip),
    ):
        packetbox.addItem(text)
        packetbox.setItemData(packetbox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    packetbox.setToolTip(
        "The spectrum of the r-packets (UVOIR), or of the gamma packets (\N{GREEK SMALL LETTER GAMMA}-rays, --gamma)"
    )
    packetmodel = packetbox.model()
    assert isinstance(packetmodel, QtGui.QStandardItemModel)
    gammaitem = packetmodel.item(1)
    # the packets and the scale also describe the y axis, thus the three boxes share one label
    add_row(axesgrid, 0, [QtWidgets.QLabel("-yvariable"), yvariablebox, packetbox, yscalebox, normalisedcheck])

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
    for widget, dest in ((hidenetcheck, "hidenetspectrum"), (hideothercheck, "hideother")):
        widget.setToolTip(helptexts.get(dest, ""))
    # the index of an item: 0 for the last emission, and 1 for the last thermal emission (--use_thermalemissiontype)
    thermalbox = QtWidgets.QComboBox()
    thermaltooltip = f"--use_thermalemissiontype: {helptexts.get('use_thermalemissiontype', '')}"
    for text, tooltip in (
        ("Last emission", "The last emission or scattering of each packet"),
        ("Last thermal emission", thermaltooltip),
    ):
        thermalbox.addItem(text)
        thermalbox.setItemData(thermalbox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    thermalbox.setToolTip("The emission of each packet that gives its emission series and its shell")
    thermalmodel = thermalbox.model()
    assert isinstance(thermalmodel, QtGui.QStandardItemModel)
    thermalitem = thermalmodel.item(1)
    # these rows apply only to an emission or absorption plot, thus they show only for such a plot
    emissionoptions = QtWidgets.QWidget()
    emissionoptionslayout = QtWidgets.QVBoxLayout(emissionoptions)
    emissionoptionslayout.setContentsMargins(0, 0, 0, 0)
    emissionoptionslayout.setSpacing(ROW_SPACING)
    emissionoptionslayout.addLayout(
        make_row_layout([QtWidgets.QLabel("-groupby"), groupbybox, countlabel, countbox, lockbutton])
    )
    emissionoptionslayout.addLayout(
        make_row_layout([hidenetcheck, hideothercheck, QtWidgets.QLabel("--use_thermalemissiontype"), thermalbox])
    )
    emissiongrid.addWidget(emissionoptions, 1, 0, 1, -1)

    # the handlers come later in this function, thus the lambdas read them at the time of a change
    show_direction, set_direction_run = add_direction_section(
        panellayout,
        helptexts,
        viewer.runfolders[0],
        lambda: get_direction_choice(viewer.values),
        lambda choice: on_direction(choice),  # ruff:ignore[unnecessary-lambda]
        lambda message: show_error(message),  # ruff:ignore[unnecessary-lambda]
    )
    for box in (countbox, binwidthbox):
        # a typed number applies when the user presses Return or leaves the box, and not after each digit
        box.setKeyboardTracking(False)

    _, spectragrid = add_section(panellayout, "Spectra")
    datasourcebox = QtWidgets.QComboBox()
    for text, source, tooltip in (
        ("Auto", "auto", "Read the packets files only when an option needs them"),
        ("Text files", "text", "Read the spectra and the emission files of exspec, e.g. spec.out and emission.out"),
        ("Packets files", "packets", f"--frompackets: {helptexts.get('frompackets', '')}"),
    ):
        datasourcebox.addItem(text, source)
        datasourcebox.setItemData(datasourcebox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    datasourcebox.setToolTip(
        "The files of the plot. Auto shows in brackets the files that it selected for the current options"
    )
    datasourcemodel = datasourcebox.model()
    assert isinstance(datasourcemodel, QtGui.QStandardItemModel)
    autoitem, textitem = datasourcemodel.item(0), datasourcemodel.item(1)
    add_row(spectragrid, 2, [QtWidgets.QLabel("--frompackets"), datasourcebox])
    _, appearancegrid = add_section(panellayout, "Appearance")
    # the box edits the row of -figscale in the other options, as the box of the estimator viewer does
    figscalebox = make_figscale_box(helptexts)
    add_row(appearancegrid, 0, [QtWidgets.QLabel("-figscale"), figscalebox])
    defaultfigscale: float = viewer.parser.get_default("figscale")
    defaultdpi: int = viewer.parser.get_default("dpi")

    def on_option_rows(rows: OptionRows) -> None:
        apply(dc.replace(viewer.values, otheroptions=rows))

    def on_figscale(figscale: float) -> None:
        change = None if math.isclose(figscale, defaultfigscale) else (format(figscale, "g"),)
        on_option_rows(set_row_values(viewer.values.otheroptions, {"-figscale": change}))

    figuresection, set_option_rows, commandtext, pythontext, (copybutton, pythoncopybutton), statusbar = (
        add_command_sections(
            viewerwindow,
            viewer,
            viewer.parser,
            viewer.values.dpi,
            (CONTROLLED_DESTS | TABLE_EXCLUDED_DESTS, viewer.values.otheroptions, on_option_rows),
        )
    )

    signalwidgets: list[QtWidgets.QWidget] = [
        figscalebox,
        figuresection.dpibox,
        modesegments,
        packetbox,
        timeslider,
        widthslider,
        widthmodebox,
        timegridbox,
        xrangeslider,
        xunitbox,
        yscalebox,
        xscalebox,
        emissioncheck,
        absorptioncheck,
        groupbybox,
        countbox,
        nostackcheck,
        lockbutton,
        binmodebox,
        binwidthbox,
        datasourcebox,
        yvariablebox,
        normalisedcheck,
        hidenetcheck,
        hideothercheck,
        thermalbox,
    ]

    # the x slider and the step of -deltax follow the unit of the x axis, thus a new unit sets them again
    logxrange = (0.0, 1.0)
    # the x unit and the packet type of the ranges, because the default x range follows both
    rangesunit: tuple[str, bool] | None = None

    def set_xunit_ranges() -> None:
        nonlocal logxrange, rangesunit
        values = viewer.values
        defaultxmin, defaultxmax = get_default_xlimits(values.xunit, gamma=values.gamma)
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
        xrangelabel.setText(f"{xunit.kind.capitalize()} [{xunit.label}]:")
        binmodebox.setItemText(binmodebox.findData("deltax"), f"-deltax [{xunit.label}]")
        rangesunit = (values.xunit, values.gamma)

    # each bin mode keeps its decimals, its range, and its default width. The box keeps the last width of each mode
    binwidthranges: dict[str, tuple[int, float, float, float]] = {"deltalogx": (8, 1e-8, 1.0, 1e-3)}
    lastbinwidths: dict[str, str] = {}

    def set_binwidth_box(binmode: str) -> None:
        """Give the box of the bin width the range and the last width of a bin mode."""
        decimals, low, high, default = binwidthranges[binmode]
        binwidthbox.setDecimals(decimals)
        binwidthbox.setRange(low, high)
        binwidthbox.setValue(float(lastbinwidths.get(binmode, default)))

    # the time sliders have one position for each valid timestep, or SLIDER_STEPS positions for a continuous time
    slidermode: bool | None = None
    # the spectra and the time grid of the runs that give the ranges of the time controls
    shownruns: tuple[tuple[str, ...], str] | None = None

    def show_run_ranges() -> None:
        """Set the ranges of the time controls and the direction bins from the runs of the plot, e.g. after Add Model.

        The ranges of the sliders and the direction bins came from the models of the command only.
        """
        nonlocal logtrange, widthmax, dlogtmax, nvalid, slidermode, shownruns
        logtrange = (math.log10(viewer.timebounds[0]), math.log10(viewer.timebounds[1]))
        widthmax = max((viewer.timebounds[1] - viewer.timebounds[0]) / 4.0, viewer.values.width)
        dlogtmax = max(math.log(viewer.timebounds[1] / viewer.timebounds[0]) / 4.0, viewer.values.dlogt)
        nvalid = len(viewer.validtimesteps)
        # show_values sets the ranges of the sliders again, and the direction bins come from the new first run
        slidermode = None
        set_direction_run(viewer.runfolders[0])
        shownruns = viewer.runkey
        timegridbox.clear()
        # two models can have one name, thus each choice gives the place of the model in the list of spectra
        for path in viewer.runtimes:
            timegridbox.addItem(f"{viewer.values.spectra.index(path) + 1}. {get_model_name(path)}", path)
            timegridbox.setItemData(
                timegridbox.count() - 1, get_run_times_text(viewer.runtimes[path]), QtCore.Qt.ItemDataRole.ToolTipRole
            )
        low, high = viewer.timebounds
        validtimeslabel.setText(f"valid {low:.4g} to {high:.4g} d")

    def set_time_mode() -> None:
        nonlocal slidermode
        continuous = viewer.values.notimeclamp
        timeslider.setRange(0, SLIDER_STEPS if continuous else nvalid - 1)
        # the first position of the width slider is one step above 0, because a width of 0 reads the whole timestep
        widthslider.setRange(1, SLIDER_STEPS if continuous else nvalid)
        widthlabel.setVisible(not continuous)
        widthmodebox.setVisible(continuous)
        slidermode = continuous

    # the test of each choice parses the arguments again, thus the code keeps one result for each set of options
    rejections: dict[tuple[t.Any, ...], tuple[list[str | None], str | None, str | None]] = {}

    def get_rejections() -> tuple[list[str | None], str | None, str | None]:
        """Return why plotspectra rejects each -groupby choice, --showemission, and --showabsorption."""
        values = viewer.values
        key = (
            values.gamma,
            values.showemission,
            values.showabsorption,
            values.groupby,
            values.datasource,
            values.notimeclamp,
            values.spectra,
            bool(values.deltax or values.deltalogx),
            values.yvariable,
            values.directionkind,
            values.otheroptions,
        )
        if key not in rejections:
            groupbys = [
                None
                if choice == (values.groupby or get_default_groupby(gamma=values.gamma))
                else viewer.get_rejection(dc.replace(values, groupby=choice, showemission=True))
                for choice in viewer.groupbychoices
            ]
            emission = None if values.showemission else viewer.get_rejection(dc.replace(values, showemission=True))
            absorption = (
                None if values.showabsorption else viewer.get_rejection(dc.replace(values, showabsorption=True))
            )
            rejections[key] = (groupbys, emission, absorption)
        return rejections[key]

    # the time does not change the files that Auto selects, thus the key of a result holds no time
    autoreasons: dict[ControlValues, str | None] = {}

    def show_data_source(values: ControlValues) -> None:
        """Select the data source of the values, and show the files that Auto selects for the other options.

        Text files stays available while it is the source of the values, thus the user can change it in either
        direction.
        """
        key = dc.replace(values, datasource="auto", centre=0.0, width=0.0)
        if key not in autoreasons:
            autoreasons[key] = get_packets_reason(viewer.get_plot_tokens(dc.replace(values, datasource="auto")))
        reason = autoreasons[key]
        autoitem.setText(f"Auto ({'packets' if reason else 'text files'})")
        autoitem.setToolTip(
            f"Read the packets files, because {reason} needs them"
            if reason
            else "Read the text files of exspec. Auto reads the packets files when an option needs them"
        )
        textitem.setEnabled(reason is None or values.datasource == "text")
        textitem.setToolTip(
            f"{reason} needs the packets files"
            if reason
            else "Read the spectra and the emission files of exspec, e.g. spec.out and emission.out"
        )
        datasourcebox.setCurrentIndex(datasourcebox.findData(values.datasource))

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

    def get_spectra_key(values: ControlValues) -> tuple[t.Any, ...]:
        """Return the parts of the values that the rows of the list of spectra show."""
        styles = tuple(get_row_values(values.otheroptions, flag) for flag in SERIES_STYLE_FLAGS)
        return (values.spectra, values.timegrid, styles)

    def make_spectrum_rows(values: ControlValues) -> list[SeriesRow]:
        """Return a row for each spectrum. The model that gives the timesteps of the time controls has a mark."""
        models = list(viewer.runtimes)
        gridpath = get_timegrid_path(values)
        colours = get_series_colours(values.spectra, values.otheroptions)
        rows: list[SeriesRow] = []
        for path in values.spectra:
            style = get_series_style(values.otheroptions, values.spectra, path)
            itemtext = get_spectrum_item_text(path)
            rows.append(
                SeriesRow(
                    path=path,
                    name=style["-label"] or get_series_name(path),
                    labelled=style["-label"] is not None,
                    itemtext=itemtext,
                    tooltip=(
                        f"{itemtext}\n{get_run_times_text(viewer.runtimes[path])}"
                        if path in viewer.runtimes
                        else itemtext
                    ),
                    swatch=make_series_swatch(mplcolors.to_hex(colours[path]), style),
                    removereason=(
                        "The plot needs one ARTIS model at least. Add a different model first"
                        if models == [path]
                        else None
                    ),
                    mark=(
                        (
                            "⏱",
                            (
                                "The time controls read the timesteps of this model. To use a different model, select"
                                ' it in the "Timesteps of" box or in the context menu of its row'
                            ),
                        )
                        if path == gridpath
                        else None
                    ),
                    extraactions=(
                        (
                            "Use for the Time Controls",
                            path in viewer.runtimes and path != gridpath,
                            partial(on_use_timegrid, path),
                        ),
                    ),
                )
            )
        return rows

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        blockers = [QtCore.QSignalBlocker(widget) for widget in signalwidgets]
        values = viewer.values
        # Undo or a rejected change can give a different list of spectra, and its runs have different times
        if (values.spectra, values.timegrid) != viewer.runkey:
            viewer.load_runs(values.spectra, values.timegrid)
        if viewer.runkey != shownruns:
            show_run_ranges()
        if (values.xunit, values.gamma) != rangesunit:
            set_xunit_ranges()
        if values.notimeclamp != slidermode:
            set_time_mode()
        modesegments.setCurrentIndex(1 if values.notimeclamp else 0)
        timegridbox.setCurrentIndex(max(timegridbox.findData(get_timegrid_path(values)), 0))
        packetbox.setCurrentIndex(1 if values.gamma else 0)
        # the current mode stays available, thus the user can switch back from a plot that failed
        if gammaitem.isEnabled() != (gammaavailable := viewer.hasgammaspectrum or values.gamma):
            gammaitem.setEnabled(gammaavailable)
            gammaitem.setToolTip(gammatooltip if gammaavailable else "A run has no gamma_spec.out and no packet files")
        previousbutton.setEnabled(viewer.step_time(-1) is not None)
        nextbutton.setEnabled(viewer.step_time(1) is not None)
        if values.notimeclamp:
            timeslider.setValue(to_position(math.log10(max(values.centre, viewer.timebounds[0])), *logtrange))
            widthmodebox.setCurrentIndex(widthmodebox.findData(values.widthmode))
            # the field and the slider give the quantity of the width mode: Δ ln t, or a width in days
            if values.widthmode == "dlogt":
                widthslider.setValue(to_position(values.dlogt, 0.0, dlogtmax))
                set_edit_text(widthedit, f"{values.dlogt:g}")
            else:
                widthslider.setValue(to_position(values.width, 0.0, widthmax))
                set_edit_text(widthedit, f"{values.width:.2f}")
        else:
            first, last = (viewer.validtimesteps.index(timestep) for timestep in viewer.get_selection(values))
            timeslider.setValue((first + last) // 2)
            widthslider.setValue(last - first + 1)
            set_edit_text(widthedit, str(last - first + 1))
        set_edit_text(timeedit, f"{values.centre:.2f}")
        timestepslabel.setText(viewer.get_time_range_text())
        set_xrange_positions(
            *(
                to_position(math.log10(max(float(limit), 10.0 ** logxrange[0])), *logxrange)
                for limit in (values.xmin, values.xmax)
            )
        )
        set_edit_text(xminedit, values.xmin)
        set_edit_text(xmaxedit, values.xmax)
        xunitbox.setCurrentText(values.xunit)
        yscalebox.setCurrentIndex(yscalebox.findData(values.yscale))
        xscalebox.setCurrentIndex(1 if values.logscalex else 0)
        show_y_limits(values.ymin, values.ymax)
        emissioncheck.setChecked(values.showemission)
        absorptioncheck.setChecked(values.showabsorption)
        groupbybox.setCurrentText(values.groupby or get_default_groupby(gamma=values.gamma))
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
        show_data_source(values)
        yvariablebox.setCurrentText(values.yvariable)
        normalisedcheck.setChecked(values.normalised)
        hidenetcheck.setChecked(values.hidenetspectrum)
        hideothercheck.setChecked(values.hideother)
        thermalbox.setCurrentIndex(1 if values.usethermalemissiontype else 0)
        # the current choice stays available, thus the user can switch back from it
        thermalreason = get_thermal_emission_reason(values)
        if thermalitem.isEnabled() != (thermalavailable := thermalreason is None or values.usethermalemissiontype):
            thermalitem.setEnabled(thermalavailable)
        thermalitem.setToolTip(thermaltooltip if thermalreason is None else thermalreason)
        show_direction()
        show_series_rows(get_spectra_key(values), partial(make_spectrum_rows, values))
        set_option_rows(values.otheroptions)
        set_spin_value(figuresection.dpibox, values.dpi or defaultdpi)
        set_spin_value(
            figscalebox, float((get_row_values(values.otheroptions, "-figscale") or (str(defaultfigscale),))[0])
        )
        set_command_text(commandtext, viewer.get_command())
        set_command_text(pythontext, get_python_code(viewer.parser, viewer.get_plot_tokens()))
        show_rejections()
        for blocker in blockers:
            blocker.unblock()
        fit_canvas(canvas, viewer.figsize, plotarea)

    def after_draw(message: str | None) -> None:
        # matplotlib keeps the connections of the mouse in the figure, and each plot has a new figure
        connect_mouse_to_figure()
        # -yscale auto reads the drawn values, thus only the drawn plot gives the scale that it chose
        if message is None and viewer.values.yscale == "auto" and plot_shows_values():
            yscalebox.setItemText(yscalebox.findData("auto"), f"Auto ({viewer.axes[0].get_yscale()})")
        # --showabsorption changes the height of the frames, thus the plot can need a new -figwidthscale
        fittimer.start()
        # a rejection occurs again at each step, thus a rejection stops the Play button
        if message is not None:
            playbutton.setChecked(False)
        elif playbutton.isChecked():
            # a draw that the Play button did not start also restarts the timer, thus one chain of steps stays
            start_play_timer(playtimer, queue.plotseconds, fpsbox.value())

    queue = DrawQueue(
        window, viewer, statusbar, show_values, after_draw, render=viewer.render, keep_on_undo=keep_figwidthscale
    )

    def apply(values: ControlValues, *, undoable: bool = True) -> None:
        """Give the queue the new values, with a time that the runs have."""
        queue.apply(viewer.clamp_time(values), undoable=undoable)

    def show_error(message: str) -> None:
        show_status_message(statusbar, message, "")
        show_values()

    def on_packet_type(index: int) -> None:
        if (gamma := index == 1) != viewer.values.gamma:
            apply(set_packet_type(viewer.values, gamma=gamma))

    def on_time_mode() -> None:
        values = viewer.values
        if modesegments.currentIndex() == 1 and not values.notimeclamp:
            # a snapped range of several timesteps keeps its width in days. A snapped range of one timestep has a
            # width of 0, thus it takes Δ ln t
            width = float(f"{values.width:.3g}")
            apply(
                dc.replace(
                    values,
                    notimeclamp=True,
                    centre=float(f"{values.centre:.4g}"),
                    width=width,
                    widthmode="days" if width > 0.0 else "dlogt",
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

    def get_timegrid_path(values: ControlValues) -> str:
        """Return the path of the model that gives the timesteps, which is the first model for the value ""."""
        return values.timegrid or next(iter(viewer.runtimes), "")

    def on_timegrid() -> None:
        timegridpath: str = timegridbox.currentData()
        if timegridpath != get_timegrid_path(viewer.values):
            apply(set_runs(viewer, viewer.values.spectra, timegridpath))

    def on_widthmode() -> None:
        values = viewer.values
        widthmode: WidthMode = widthmodebox.currentData()
        # Δ ln t keeps its last value, which starts as the Δ ln t of a logarithmic grid of the run
        apply(dc.replace(values, widthmode=widthmode))

    def on_width(position: int) -> None:
        values = viewer.values
        if values.notimeclamp:
            if values.widthmode == "dlogt":
                dlogt = from_position(position, 0.0, dlogtmax)
                apply(dc.replace(values, dlogt=float(f"{dlogt:.4g}")))
            else:
                width = from_position(position, 0.0, widthmax)
                apply(dc.replace(values, width=float(f"{width:.3g}")))
            return
        first = viewer.validtimesteps.index(viewer.get_selection(values)[0])
        last = min(first + position - 1, nvalid - 1)
        apply(viewer.snap(values, viewer.validtimesteps[first], viewer.validtimesteps[last]))

    def on_timeedit() -> None:
        # a field shows a time with two decimal places, thus only a field that the user edited gives a new value, and
        # a Return in a field with no edit keeps the time range
        timeedited, widthedited = timeedit.isModified(), widthedit.isModified()
        # a later plot can show new text in the fields only when they have no edit of the user
        timeedit.setModified(False)
        widthedit.setModified(False)
        if not (timeedited or widthedited):
            return
        values = viewer.values
        try:
            centre = float(timeedit.text()) if timeedited else values.centre
            width = float(widthedit.text()) if widthedited else None
        except ValueError:
            show_error("Give a number of days for the time, and a number for the width")
            return
        low, high = viewer.timebounds
        if not low <= centre <= high:
            show_error(f"Give a time from {low:.2f} to {high:.2f} d")
            return
        if values.notimeclamp:
            if width is None:
                newvalues = dc.replace(values, centre=centre)
            elif not width > 0.0:
                show_error("Give a width above 0. A continuous range of width 0 reads the whole timestep")
                return
            elif values.widthmode == "dlogt" and not width <= (dlogtlimit := math.log(high / low)):
                # a larger Δ ln t covers more than all the valid times, and a very large one gives a start of 0
                show_error(f"Give a Δ ln t above 0 and up to {dlogtlimit:.4g}, which covers all the valid times")
                return
            elif values.widthmode == "dlogt":
                newvalues = dc.replace(values, centre=centre, dlogt=width)
            else:
                newvalues = dc.replace(values, centre=centre, width=width)
        else:
            firstpos, lastpos = (viewer.validtimesteps.index(timestep) for timestep in viewer.get_selection(values))
            count = lastpos - firstpos + 1 if width is None else min(max(1, round(width)), nvalid)
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

    def on_axes() -> None:
        values = viewer.values
        if xunitbox.currentText() != values.xunit:
            values = convert_xunit(values, xunitbox.currentText(), gamma=values.gamma)
        values = dc.replace(
            values,
            yscale=yscalebox.currentData(),
            logscalex=xscalebox.currentIndex() == 1,
            yvariable=yvariablebox.currentText(),
            normalised=normalisedcheck.isChecked(),
        )
        apply(values)

    def on_emission_options() -> None:
        groupby = groupbybox.currentText()
        showemission = emissioncheck.isChecked()
        # -groupby colours the emission plot, thus a choice of -groupby also sets --showemission.
        # An empty --showemission check box removes the -groupby choice.
        defaultgroupby = get_default_groupby(gamma=viewer.values.gamma)
        if groupby != (viewer.values.groupby or defaultgroupby):
            showemission = True
        elif not showemission:
            groupby = defaultgroupby
        # plotspectra takes the default -groupby when the command gives none, thus the command stays short
        if groupby == defaultgroupby:
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
            datasource=datasourcebox.currentData(),
            hidenetspectrum=hidenetcheck.isChecked(),
            hideother=hideothercheck.isChecked(),
            usethermalemissiontype=thermalbox.currentIndex() == 1,
        )
        # the labels of a locked list belong to one -groupby, thus a new -groupby removes the lock
        if groupby != viewer.values.groupby:
            values = remove_series_lock(values)
        # an emission plot draws one direction bin, thus it keeps the first selected bin
        if values.showemission or values.showabsorption:
            values = dc.replace(values, directionbins=values.directionbins[:1])
        apply(values)

    def on_binmode() -> None:
        # the box holds the width of the previous mode, thus it takes the width of the new mode before the values change
        with QtCore.QSignalBlocker(binwidthbox):
            set_binwidth_box(binmodebox.currentData() or "deltax")
        on_emission_options()

    def get_direction_choice(values: ControlValues) -> tuple[DirectionChoice, bool]:
        """Return the viewing direction of the values, and whether the plot draws one bin, i.e. an emission or absorption plot."""
        choice = DirectionChoice(kind=values.directionkind, bins=values.directionbins, usedegrees=values.usedegrees)
        return choice, values.showemission or values.showabsorption

    def on_direction(choice: DirectionChoice) -> None:
        apply(
            dc.replace(
                viewer.values, directionkind=choice.kind, directionbins=choice.bins, usedegrees=choice.usedegrees
            )
        )

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
        apply(set_runs(viewer, spectra, viewer.values.timegrid))

    def on_use_timegrid(path: str) -> None:
        apply(set_runs(viewer, viewer.values.spectra, path))

    def on_edit_label(path: str) -> None:
        """Ask for the -label of a spectrum. An empty label gives the automatic label of plotspectra."""
        labels = get_row_values(viewer.values.otheroptions, "-label") or ()
        index = viewer.values.spectra.index(path)
        label, accepted = QtWidgets.QInputDialog.getText(
            window,
            "Set Label",
            f"The label of {get_series_name(path)} in the legend.\nClear the field for the automatic label.",
            text=get_series_value(labels, index) or "",
        )
        if accepted:
            apply(set_series_values(viewer.values, path, {"-label": label.strip() or None}))

    def on_edit_style(path: str) -> None:
        """Ask for the colour, the line style, the width, and the opacity of the line of a spectrum."""
        values = viewer.values
        if path not in values.spectra:
            return
        # the colour of the dialog for "Default" is the colour that the spectrum has with no -color of its own
        defaultcolours = get_series_colours(
            values.spectra, set_series_values(values, path, {"-color": None}).otheroptions
        )
        style = get_series_style(values.otheroptions, values.spectra, path)
        name = style["-label"] or get_series_name(path)
        changes = edit_series_style(
            window, name, style, mplcolors.to_hex(defaultcolours[path]), float(mpl.rcParams["lines.linewidth"])
        )
        if changes is not None:
            apply(set_series_values(viewer.values, path, changes))

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

    def is_run_folder(path: str) -> bool:
        return bool(get_artis_run_folders([Path(path)]))

    show_series_rows = add_series_list(
        spectragrid,
        window,
        partial(show_status_note, statusbar),
        "The ARTIS models and the observed spectra of the plot, in the order of the command. The order sets the"
        " -label and the style of each series. The time controls read the timesteps of the first ARTIS model (⏱)."
        " Click ▲ or ▼, drag a row, or press Alt-Up or Alt-Down (Option on a Mac) to change the order. The command"
        " gives a file from the reference data of artistools by its name alone.",
        ReferenceData(
            kind="reference spectrum",
            names=get_reference_spectrum_names(),
            folder=get_path("artistools_dir") / "data" / "refspectra",
            find=find_reference_spectrum_file_or_none,
            get_token=get_reference_token,
            example="AT2017gfo",
        ),
        SeriesListActions(
            get_paths=lambda: viewer.values.spectra,
            apply_paths=apply_spectra,
            edit_label=on_edit_label,
            edit_style=on_edit_style,
            get_full_path=get_spectrum_path,
            is_run=is_run_folder,
            show_error=show_error,
        ),
    )

    def on_reload() -> None:
        """Read the runs again, e.g. while ARTIS writes more timesteps, and draw the plot again."""

        def show_reloaded_runs() -> None:
            viewer.load_runs(viewer.values.spectra, viewer.values.timegrid)
            show_run_ranges()
            queue.redraw()

        reload_runs(queue, viewer.runfolders, show_reloaded_runs, show_error)

    command = ViewerCommand(
        name="plotspectra",
        main=plotspectra_main,
        parser=viewer.parser,
        get_figure_tokens=lambda: viewer.get_plot_tokens(dc.replace(viewer.values, dpi=None)),
        get_command=viewer.get_command,
        get_python_code=lambda: get_python_code(viewer.parser, viewer.get_plot_tokens()),
    )
    add_figure_actions = add_window_actions(
        window,
        windows,
        open_window,
        queue,
        statusbar,
        command,
        figuresection,
        (copybutton, pythoncopybutton),
        (lambda: viewer.values.dpi, lambda dpi: apply(dc.replace(viewer.values, dpi=dpi))),
        show_error,
        KEYBOARD_HELP_ROWS,
        playbutton,
        extracallbacks={"Reload Data": on_reload, "Export Animation…": on_export_animation},
    )

    def on_select_y(frameindex: int, low: float, high: float) -> None:
        """Give the frame of the spectra the y range of a Shift-drag. A residual panel has its own range."""
        ymin, ymax = get_short_number(low), get_short_number(high)
        if frameindex == 0 and plot_shows_values() and float(ymin) < float(ymax):
            apply(dc.replace(viewer.values, ymin=ymin, ymax=ymax))

    def on_plot_menu(frameindex: int, _event: t.Any) -> None:
        """Show the actions on the frame and the figure under the pointer, as the context menu of a Mac app does."""
        menu = QtWidgets.QMenu(window)
        if frameindex == 0 and plot_shows_values() and len(viewer.axes):
            add_y_axis_actions(
                menu,
                viewer.axes[0].get_yscale() == "log",
                bool(viewer.values.ymin or viewer.values.ymax),
                lambda yscale: apply(dc.replace(viewer.values, yscale=yscale)),
                lambda: apply(dc.replace(viewer.values, ymin="", ymax="")),
            )
        add_figure_actions(menu)
        menu.exec(QtGui.QCursor.pos())
        # the window is the parent of the menu, thus without this the window keeps each menu until it closes
        menu.deleteLater()

    # the window keeps its command at a quit, and the next start opens the window again
    def get_session_tokens() -> list[str]:
        # a command with no path reads the working folder, and the next start can be in a different folder
        tokens = viewer.get_plot_tokens()
        return [str(Path.cwd()), *tokens] if viewer.values.spectra == DEFAULT_SPECTRA else tokens

    modesegments.currentChanged.connect(on_time_mode)
    packetbox.currentIndexChanged.connect(on_packet_type)
    previousbutton.clicked.connect(lambda: on_arrow(-1))
    nextbutton.clicked.connect(lambda: on_arrow(1))
    timeslider.valueChanged.connect(on_time)
    widthslider.valueChanged.connect(on_width)
    widthmodebox.currentIndexChanged.connect(on_widthmode)
    timeedit.editingFinished.connect(on_timeedit)
    widthedit.editingFinished.connect(on_timeedit)
    playbutton.toggled.connect(on_play)
    figscalebox.valueChanged.connect(on_figscale)
    playtimer.timeout.connect(play_step)
    connect_xrange(on_xrange)
    xminedit.editingFinished.connect(on_xedit)
    xmaxedit.editingFinished.connect(on_xedit)
    xunitbox.currentTextChanged.connect(on_axes)
    yscalebox.currentIndexChanged.connect(on_axes)
    xscalebox.currentIndexChanged.connect(on_axes)
    emissioncheck.toggled.connect(on_emission_options)
    absorptioncheck.toggled.connect(on_emission_options)
    groupbybox.currentTextChanged.connect(on_emission_options)
    countbox.valueChanged.connect(on_emission_options)
    nostackcheck.toggled.connect(on_emission_options)
    lockbutton.toggled.connect(on_lock)
    binmodebox.currentIndexChanged.connect(on_binmode)
    binwidthbox.valueChanged.connect(on_emission_options)
    datasourcebox.currentIndexChanged.connect(on_emission_options)
    for checkbox in (hidenetcheck, hideothercheck):
        checkbox.toggled.connect(on_emission_options)
    thermalbox.currentIndexChanged.connect(on_emission_options)
    yvariablebox.currentTextChanged.connect(on_axes)
    normalisedcheck.toggled.connect(on_axes)
    timegridbox.currentIndexChanged.connect(on_timegrid)
    connect_mouse_to_figure = connect_plot_mouse(
        canvas,
        get_frames=lambda: [axis for axis in (*viewer.axes, viewer.residualaxis) if axis is not None],
        get_readout=lambda event, _frame: viewer.get_readout(event.xdata),
        readoutlabel=statusbar.readout,
        on_select=set_xlimits,
        on_reset=lambda: set_xlimits(*get_default_xlimits(viewer.values.xunit, gamma=viewer.values.gamma)),
        can_select=plot_has_xunit,
        on_select_y=on_select_y,
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

    finish_viewer_window(
        viewerwindow,
        windows,
        viewer,
        queue,
        viewer.get_command,
        get_session_tokens,
        (
            viewer.get_fitted_figwidthscale,
            lambda: viewer.values.figwidthscale,
            lambda figwidthscale: apply(dc.replace(viewer.values, figwidthscale=figwidthscale), undoable=False),
        ),
    )
    show_values()
    return None
