"""Show the plot of plotlightcurves in a window with controls for the light curves, the energy rates, and the axes."""

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
from matplotlib.backends.backend_agg import FigureCanvasAgg

from artistools.lightcurve.core import find_bol_reflightcurve_file
from artistools.lightcurve.core import path_is_reference_lightcurve
from artistools.lightcurve.plotlightcurve import addargs
from artistools.lightcurve.plotlightcurve import ANALYTICEMISSIONCOLUMNS
from artistools.lightcurve.plotlightcurve import DEPOSITIONCHOICES
from artistools.lightcurve.plotlightcurve import DEPOSITIONCOLUMNS
from artistools.lightcurve.plotlightcurve import draw_plot
from artistools.lightcurve.plotlightcurve import EMISSIONCOLUMNS
from artistools.lightcurve.plotlightcurve import ENERGYPARTICLES
from artistools.lightcurve.plotlightcurve import ENERGYRATEDESTS
from artistools.lightcurve.plotlightcurve import get_plot_lum_unit
from artistools.lightcurve.plotlightcurve import get_thermalisation_emission_column
from artistools.lightcurve.plotlightcurve import LumUnit
from artistools.lightcurve.plotlightcurve import make_plot_figure
from artistools.lightcurve.plotlightcurve import resolve_plot_args
from artistools.misc import exit_with_error
from artistools.misc import get_artis_run_folders
from artistools.misc import get_deposition
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import print_warning
from artistools.misc import separate_trailing_folders
from artistools.misc.fileio import COMPRESSED_EXTENSIONS
from artistools.misc.fileio import resolve_modelpath
from artistools.misc.remote import is_remote_path
from artistools.plottools import LABELWIDTH_INCHES
from artistools.plottools import RIGHTMARGIN_INCHES
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
from artistools.viewertools import fit_canvas
from artistools.viewertools import FIT_MILLISECONDS
from artistools.viewertools import fix_title_position
from artistools.viewertools import follow_colour_scheme
from artistools.viewertools import get_changed_arguments
from artistools.viewertools import get_dark_plot_colours
from artistools.viewertools import get_direction_choices
from artistools.viewertools import get_direction_kind
from artistools.viewertools import get_direction_kinds
from artistools.viewertools import get_figure_format
from artistools.viewertools import get_fitted_figwidthscale
from artistools.viewertools import get_helptexts
from artistools.viewertools import get_keyboard_help
from artistools.viewertools import get_line_readouts
from artistools.viewertools import get_new_figwidthscale
from artistools.viewertools import get_option_row_tokens
from artistools.viewertools import get_option_tokens
from artistools.viewertools import get_python_call
from artistools.viewertools import get_row_values
from artistools.viewertools import get_short_number
from artistools.viewertools import make_central_splitter
from artistools.viewertools import make_command_tokens
from artistools.viewertools import make_completer
from artistools.viewertools import make_elided_label
from artistools.viewertools import make_glyph_button
from artistools.viewertools import make_option_table
from artistools.viewertools import make_parser
from artistools.viewertools import make_plot_area
from artistools.viewertools import make_range_slider
from artistools.viewertools import make_readout_tag
from artistools.viewertools import make_reorder_list
from artistools.viewertools import make_sidebar
from artistools.viewertools import make_status_bar
from artistools.viewertools import make_timer
from artistools.viewertools import make_window
from artistools.viewertools import open_model_folder
from artistools.viewertools import open_model_window
from artistools.viewertools import OptionRows
from artistools.viewertools import parse_command_tokens
from artistools.viewertools import remove_options
from artistools.viewertools import run_command_step_with_warning
from artistools.viewertools import run_viewer_application
from artistools.viewertools import save_figure_of_command
from artistools.viewertools import set_command_text
from artistools.viewertools import set_drop_handler
from artistools.viewertools import set_edit_text
from artistools.viewertools import set_row_values
from artistools.viewertools import set_spin_value
from artistools.viewertools import set_window_document
from artistools.viewertools import show_figure_in_canvas
from artistools.viewertools import show_status_message
from artistools.viewertools import show_status_note
from artistools.viewertools import show_window
from artistools.viewertools import SLIDER_STEPS
from artistools.viewertools import split_option_rows

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import matplotlib.axes as mplax
    import numpy.typing as npt
    from PySide6 import QtWidgets

# the controls of the window give these arguments, thus the command drops the values that the user typed
CONTROLLED_DESTS: t.Final = frozenset({
    # the Resolution box of the Figure section gives -dpi
    "dpi",
    # -timestep and -timedays give the time range, which the window shows as -timemin and -timemax
    "timestep",
    "timedays",
    "timemin",
    "timemax",
    "logscalex",
    "magnitude",
    "Lsun",
    "yscale",
    "logscaley",
    "ymin",
    "ymax",
    "rpkt",
    "gamma",
    "escape_type",
    "frompackets",
    "topnucs",
    "use_pellet_decay_time",
    "plotcmf",
    "plotinvalidpart",
    "deposition",
    "emission",
    "analyticemission",
    "thermalisation",
    # the older flags of the energy rates, which the lists of particles replace
    "plotdeposition",
    "plotalphadeposition",
    "plotthermalisation",
    "plotviewingangle",
    "plotvspecpol",
    "average_over_phi_angle",
    "average_over_theta_angle",
    "average_every_tenth_viewing_angle",
    "usedegrees",
    "figwidthscale",
    # the window shows the plot, thus the command opens no second window and no file
    "show",
    "interactive",
})

APPLICATION_NAME: t.Final = "artistools plotlightcurves"

# these options give a different action from one plot of the bolometric light curves, thus the table of the window
# does not offer them
TABLE_EXCLUDED_DESTS: t.Final = frozenset({
    "help",
    "filter",
    "colour_evolution",
    "colouratpeak",
    "brightnessattime",
    "save_angle_averaged_peakmag_risetime_delta_m15_to_file",
    "save_viewing_angle_peakmag_risetime_delta_m15_to_file",
    "make_viewing_angle_peakmag_risetime_scatter_plot",
    "make_viewing_angle_peakmag_delta_m15_scatter_plot",
    "test_viewing_angle_fit",
    "include_delta_m40",
    "noerrorbars",
    "noangleaveraged",
    "plot_hesma_model",
    "legendsubplotnumber",
})

# plotlightcurves reads the model in the working folder when the command gives no path
DEFAULT_LIGHTCURVES: t.Final = (".",)

# the text of each particle in the controls of the energy rates
PARTICLETEXTS: t.Final = {
    "gamma": "\N{GREEK SMALL LETTER GAMMA}",
    "betaminus": "\N{GREEK SMALL LETTER BETA}\N{SUPERSCRIPT MINUS}",
    "betaplus": "\N{GREEK SMALL LETTER BETA}\N{SUPERSCRIPT PLUS SIGN}",
    "alpha": "\N{GREEK SMALL LETTER ALPHA}",
    "fission": "Fission",
    "total": "Total",
}

# the text of each luminosity unit of the y axis, and its flag
LUMUNITS: t.Final[tuple[tuple[LumUnit, str, str], ...]] = (
    ("erg/s", "erg/s", ""),
    ("Lsun", "L\N{SUN}", "--Lsun"),
    ("mag", "Magnitude", "--magnitude"),
)


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class ControlValues:
    """The values of the controls of the viewer, which give the options of the plotlightcurves command.

    The limits of the axes keep the text of the command, and an empty limit gives no option.
    """

    timemin: str
    timemax: str
    logscalex: bool
    lumunit: LumUnit
    yscale: str
    ymin: str
    ymax: str
    # True for the light curve of the gamma packets, and False for the UVOIR light curve of the r-packets
    gamma: bool
    frompackets: bool
    topnucs: int
    usepelletdecaytime: bool
    plotcmf: bool
    plotinvalidpart: bool
    # the particles of each energy rate, in the order of DEPOSITIONCHOICES
    deposition: tuple[str, ...]
    emission: tuple[str, ...]
    analyticemission: tuple[str, ...]
    thermalisation: tuple[str, ...]
    # the kind of viewing direction:
    # - "" for all directions;
    # - "bin" for -plotviewingangle;
    # - "phi" and "theta" for the averages;
    # - "vpkt" for -plotvspecpol.
    directionkind: str
    directionbins: tuple[int, ...]
    usedegrees: bool
    # the paths of the ARTIS models and the reference light curves, in the order of the command
    lightcurves: tuple[str, ...]
    figwidthscale: float
    # the resolution of a PNG file (-dpi), or None for the default of the command
    dpi: int | None
    otheroptions: OptionRows


def get_energy_rate_columns(runfolders: "Sequence[Path]") -> frozenset[str]:
    """Return each column of deposition.out that at least one run of the plot gives.

    A run with no deposition.out, or with a deposition.out that does not match its timesteps, gives no column.
    plotlightcurves then stops only when the user selects an energy rate.
    """
    columns: set[str] = set()
    for runfolder in runfolders:
        with contextlib.suppress(FileNotFoundError, AssertionError):
            columns |= set(get_deposition(runfolder).collect_schema().names())
    return frozenset(columns)


def get_energy_rate_column_names(dest: str, particle: str) -> tuple[str, ...]:
    """Return the columns of deposition.out that the energy rate of the particle reads, or () for no such rate."""
    if dest == "thermalisation":
        # the emission rate of fission is its deposition rate, thus its ratio reads one column
        return (
            tuple(dict.fromkeys((DEPOSITIONCOLUMNS[particle], get_thermalisation_emission_column(particle))))
            if particle in ENERGYPARTICLES
            else ()
        )
    columns = {
        "deposition": DEPOSITIONCOLUMNS,
        "emission": EMISSIONCOLUMNS,
        "analyticemission": ANALYTICEMISSIONCOLUMNS,
    }
    return (columns[dest][particle],) if particle in columns[dest] else ()


def sort_particles(particles: "set[str] | Sequence[str]") -> tuple[str, ...]:
    """Return the particles in the order of the command and of the legend."""
    return tuple(particle for particle in DEPOSITIONCHOICES if particle in particles)


def check_viewer_args(args: argparse.Namespace) -> None:
    """Stop when args selects an action that is not one plot of light curves. The window shows one plot only."""
    exit_for_other_actions(
        "light curves",
        {
            "-filter": bool(args.filter),
            "-colour_evolution": args.colour_evolution is not None,
            "--colouratpeak": args.colouratpeak,
            "--brightnessattime": args.brightnessattime,
            "--save_angle_averaged_peakmag_risetime_delta_m15_to_file": (
                args.save_angle_averaged_peakmag_risetime_delta_m15_to_file
            ),
            "--save_viewing_angle_peakmag_risetime_delta_m15_to_file": (
                args.save_viewing_angle_peakmag_risetime_delta_m15_to_file
            ),
            "--make_viewing_angle_peakmag_risetime_scatter_plot": args.make_viewing_angle_peakmag_risetime_scatter_plot,
            "--make_viewing_angle_peakmag_delta_m15_scatter_plot": (
                args.make_viewing_angle_peakmag_delta_m15_scatter_plot
            ),
        },
    )


def get_python_code(parser: argparse.ArgumentParser, tokens: "Sequence[str]") -> str:
    """Return the Python code that draws the plot of the command, with each argument that differs from its default."""
    args = parse_command_tokens(parser, tokens)
    if args is None:
        return "# plotlightcurves rejects the command"
    return get_python_call("at.lightcurve.plot", get_changed_arguments(parser, args))


def keep_figwidthscale(restored: ControlValues, current: ControlValues) -> ControlValues:
    """Return the values that Undo restores, with the current -figwidthscale, which the window sets."""
    return dc.replace(restored, figwidthscale=current.figwidthscale)


def get_reference_token(filename: str) -> str:
    """Return the name of a reference file if plotlightcurves finds that same file by the name, and the path if not.

    plotlightcurves searches the working folder before the reference data of artistools. Thus a file of the same name
    in the working folder takes the place of a file from the reference data. The name alone gives a short command.
    """
    found = find_bol_reflightcurve_file(Path(filename).name)
    return Path(filename).name if found is not None and found.resolve() == Path(filename).resolve() else filename


def get_lightcurve_path(path: str) -> Path:
    """Return the full path of the folder or the file of a light curve, e.g. of "." or of a name of a reference file.

    Two spellings of one light curve, e.g. "." and the full path of the working folder, then give the same path. A
    model on a different host keeps its path.
    """
    if path_is_reference_lightcurve(path):
        return (find_bol_reflightcurve_file(path) or Path(path)).resolve()
    return resolve_modelpath(path)


def get_lightcurve_item_text(path: str) -> str:
    """Return the text of a path in the list of light curves: the kind and the full path.

    The list shortens a long path in the middle, thus the text keeps the start and the end of the path.
    """
    if path_is_reference_lightcurve(path):
        return f"Reference: {(find_bol_reflightcurve_file(path) or Path(path)).absolute()}"
    return f"Model: {path if is_remote_path(path) else Path(path).absolute()}"


def get_reference_lightcurve_names() -> list[str]:
    """Return the names of the bolometric reference light curves in the data of artistools.

    A name has no suffix of a compressed file, because plotlightcurves finds the compressed file by the name without
    the suffix. A metadata file with no data file beside it gives no name.
    """
    from artistools.commands import get_path

    folder = get_path("artistools_dir") / "data" / "lightcurves" / "bollightcurves"
    names = {
        path.name.removesuffix(path.suffix) if path.suffix in COMPRESSED_EXTENSIONS else path.name
        for path in folder.iterdir()
        if path.is_file() and not path.name.startswith(".") and not path.name.endswith(".meta.yml")
    }
    return sorted(names, key=str.lower)


class RenderedLightCurve(t.NamedTuple):
    """A figure that the worker thread drew, with the frames that the window reads."""

    fig: mplfig.Figure
    axis: "mplax.Axes"
    thermaxis: "mplax.Axes | None"
    residualaxis: "mplax.Axes | None"


class LightCurveViewer:
    """The plot of the viewer and the values of its controls.

    The window reads and changes the values, and a test can do the same without a display. Each call to draw
    makes the command from the values, then parses it and draws it with the code of plotlightcurves. Thus the plot
    always agrees with the command.
    """

    def __init__(self, tokens: "Sequence[str]", fig: mplfig.Figure) -> None:
        """Read the arguments of the user, and take the first values of the controls from them."""
        parser = make_parser(addargs)
        usertokens = remove_options(parser, tokens, {"interactive"})
        # parse_cli_args puts "--" in front of the ARTIS folders at the end, thus an option that reads a list, e.g.
        # -deposition, does not take a folder. The removal of the controlled options must keep those folders
        basetokens = remove_options(parser, separate_trailing_folders(usertokens), CONTROLLED_DESTS)
        args = parse_cli_args(addargs, None, None, usertokens)
        resolve_plot_args(args)
        check_viewer_args(args)
        if args.rpkt and args.gamma:
            print_warning(
                "The window shows the UVOIR light curve or the gamma-ray light curve, and not both. It starts with the"
                " UVOIR light curve"
            )
        self.args = args

        if not get_artis_run_folders(args.modelpath):
            exit_with_error(
                "--interactive takes the time range and the energy rates from an ARTIS run, and no path names a run",
                "Give the folder of an ARTIS run, e.g. plotlightcurves mymodel --interactive",
            )
        self.load_runs(args.modelpath)

        # the table of the window shows each option that no other control sets. The tokens that no option takes are
        # paths, e.g. the path of "--notitle mymodel", or the paths after "--"
        pathcount = next((index for index, token in enumerate(basetokens) if token.startswith("-")), len(basetokens))
        otheroptions, positionaltokens = split_option_rows(parser, basetokens[pathcount:])
        # the order of the paths gives the -label and the style of each series, thus the paths keep their order
        startpaths = [*basetokens[:pathcount], *(word for word in positionaltokens if word != "--")]
        self.parser = parser

        actions = {action.dest: action for action in parser._actions}  # ruff:ignore[private-member-access]
        self.yscalechoices = [str(choice) for choice in actions["yscale"].choices or () if choice != "lin"]
        self.helptexts = get_helptexts(parser)
        self.defaultyscale: str = parser.get_default("defaultyscale")
        directionkind = get_direction_kind(args)
        directionbins = tuple(args.plotvspecpol or args.plotviewingangle or ())
        if -2 in directionbins and directionkind:
            # -plotviewingangle -2 selects each direction bin, and the window shows a check box for each one
            directionbins = tuple(
                dirbin
                for dirbin, _ in get_direction_choices(self.runfolders[0], directionkind, usedegrees=args.usedegrees)
            )
        self.values = ControlValues(
            timemin="" if args.timemin is None else format(args.timemin, ".10g"),
            timemax="" if args.timemax is None else format(args.timemax, ".10g"),
            logscalex=bool(args.logscalex),
            lumunit=get_plot_lum_unit(args),
            yscale=args.yscale,
            ymin="" if args.ymin is None else format(args.ymin, ".10g"),
            ymax="" if args.ymax is None else format(args.ymax, ".10g"),
            gamma=bool(args.gamma) and not args.rpkt,
            frompackets=bool(args.frompackets) and not args.topnucs,
            topnucs=args.topnucs,
            usepelletdecaytime=bool(args.use_pellet_decay_time),
            plotcmf=bool(args.plotcmf),
            plotinvalidpart=bool(args.plotinvalidpart),
            deposition=tuple(args.deposition),
            emission=tuple(args.emission),
            analyticemission=tuple(args.analyticemission),
            thermalisation=tuple(args.thermalisation),
            directionkind=directionkind,
            directionbins=directionbins,
            usedegrees=bool(args.usedegrees),
            lightcurves=tuple(startpaths) or DEFAULT_LIGHTCURVES,
            figwidthscale=args.figwidthscale,
            dpi=None if args.dpi == parser.get_default("dpi") else args.dpi,
            otheroptions=otheroptions,
        )

        self.fig = fig
        self.axis: mplax.Axes | None = None
        self.thermaxis: mplax.Axes | None = None
        self.residualaxis: mplax.Axes | None = None
        # the colours of the window in Dark Mode, which the window sets and the worker thread reads
        self.darkcolours: tuple[str, str] | None = None
        # the last warning of the last plot, which the status bar shows
        self.warning = ""
        # a window can change the size of the figure, thus the size of the frames stays here
        self.figsize: tuple[float, float] = (0.0, 0.0)

    def load_runs(self, lightcurves: "Sequence[str | Path]") -> None:
        """Read the times and the columns of deposition.out of the ARTIS runs of the light curves.

        The time controls cover the timesteps of all the runs. A reference light curve has no run.
        """
        runfolders = get_artis_run_folders(lightcurves)
        self.runfolders = runfolders
        self.timebounds = (
            min(float(get_timestep_times(runfolder, loc="start")[0]) for runfolder in runfolders),
            max(float(get_timestep_times(runfolder, loc="end")[-1]) for runfolder in runfolders),
        )
        self.energyratecolumns = get_energy_rate_columns(runfolders)
        # the direction controls read the first run, e.g. for the observers of -plotvspecpol
        self.directionkinds = get_direction_kinds(runfolders[0])
        self.runlightcurves = tuple(str(path) for path in lightcurves)

    def get_plot_tokens(self, values: ControlValues | None = None) -> list[str]:
        """Return the plotlightcurves arguments of the values, or of the current values if the caller gives none."""
        if values is None:
            values = self.values
        options: list[str] = []
        for flag, limit in (
            ("-timemin", values.timemin),
            ("-timemax", values.timemax),
            ("-ymin", values.ymin),
            ("-ymax", values.ymax),
        ):
            if limit:
                options += get_option_tokens(flag, limit)
        options += [flag for unit, _, flag in LUMUNITS if unit == values.lumunit and flag]
        if values.yscale != self.defaultyscale:
            options += ["-yscale", values.yscale]
        # plotlightcurves draws the UVOIR light curve when the command gives no --gamma
        if values.gamma:
            options.append("--gamma")
        # -topnucs reads the packets without --frompackets
        if values.topnucs:
            options += ["-topnucs", str(values.topnucs)]
        for isgiven, flag in (
            (values.logscalex, "--logscalex"),
            (values.frompackets and not values.topnucs, "--frompackets"),
            (values.usepelletdecaytime, "--use_pellet_decay_time"),
            (values.plotcmf, "--plotcmf"),
            (values.plotinvalidpart, "--plotinvalidpart"),
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
        # a list option takes each word that follows it, thus the lists of particles come after every other option
        for dest in ENERGYRATEDESTS:
            if particles := getattr(values, dest):
                options += [f"-{dest}", *particles]
        # a command with no path reads the model in the working folder, thus that model needs no path
        paths = [] if values.lightcurves == DEFAULT_LIGHTCURVES else list(values.lightcurves)
        return make_command_tokens([*paths, *get_option_row_tokens(values.otheroptions)], options)

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotlightcurves", *self.get_plot_tokens()])

    def get_energy_rate_reason(self, dest: str, particle: str) -> str | None:
        """Return why no run of the plot gives an energy rate of a particle, or None if a run gives it."""
        columns = get_energy_rate_column_names(dest, particle)
        if not columns:
            return f"-{dest} takes no {particle}, because ARTIS writes no such rate in deposition.out"
        if missing := [column for column in columns if column not in self.energyratecolumns]:
            return f"No run of the plot gives {' or '.join(missing)} in deposition.out"
        return None

    def draw(self, *, quiet: bool = True) -> str | None:
        """Draw the plot of the values, and return the reason for the status line if plotlightcurves rejects it.

        The terminal shows the whole error, and the status line shows its first line.
        """
        return self.render(self.values, quiet=quiet)()

    def render(self, values: ControlValues, *, quiet: bool = True) -> "Callable[[], str | None]":
        """Draw the plot of the values on a new figure, and return the function that shows it in the canvas.

        The function returns the reason for the status line if plotlightcurves rejects the values, and the old plot
        then stays. A worker thread can run this method, because it changes nothing that the window reads. The
        function that it returns must run in the thread of the window.
        """
        plots: list[RenderedLightCurve] = []

        def make_plot() -> str | None:
            plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens(values))
            resolve_plot_args(plotargs)
            check_viewer_args(plotargs)
            fig = mplfig.Figure()
            FigureCanvasAgg(fig)
            _, axis, thermaxis, residualaxis = make_plot_figure(plotargs, fig=fig)
            draw_plot(plotargs, axis, thermaxis, residualaxis)
            fix_title_position(axis)
            if (darkcolours := self.darkcolours) is not None:
                apply_dark_colours(fig, *darkcolours)
            # the worker makes the ticks and the text layout, thus the first draw in the window is faster
            fig.draw_without_rendering()
            plots.append(RenderedLightCurve(fig=fig, axis=axis, thermaxis=thermaxis, residualaxis=residualaxis))
            return None

        message, warning = run_command_step_with_warning(make_plot, quiet=quiet)

        def show_plot() -> str | None:
            self.warning = warning
            if message is not None:
                return message
            plot = plots[0]
            self.figsize = show_figure_in_canvas(self.fig, plot.fig)
            self.fig, self.axis, self.thermaxis, self.residualaxis = plot
            return None

        return show_plot

    def get_frames(self) -> "list[mplax.Axes]":
        """Return the frames of the plot on the screen, from the top to the bottom."""
        return [axis for axis in (self.axis, self.residualaxis, self.thermaxis) if axis is not None]

    def get_fitted_figwidthscale(self, areawidth: float, areaheight: float) -> float:
        """Return the -figwidthscale that gives the figure the shape of the plot area."""
        marginwidth = LABELWIDTH_INCHES + RIGHTMARGIN_INCHES
        return get_fitted_figwidthscale(self.figsize, self.values.figwidthscale, marginwidth, areawidth, areaheight)

    def change(self, values: ControlValues) -> str | None:
        """Draw the plot of the new values, and keep the old values and the old plot if plotlightcurves rejects them."""
        message = self.render(values)()
        if message is None:
            self.values = values
        return message


def get_readout(x: float, frame: "mplax.Axes") -> str:
    """Return the time and the value of each drawn series of the frame at the time x."""
    return "   ".join([f"{x:.4g} d", *get_line_readouts(frame, x)])


def get_icon_curve() -> "npt.NDArray[np.float64]":
    """Return the curve of the icon of the viewer, which is a light curve with a fast rise and a slow decline."""
    xvalues = np.linspace(0.0, 1.0, 200)
    curve = (xvalues / 0.2) ** 2 * np.exp(2.0 * (1.0 - xvalues / 0.2) / 2.0)
    return np.asarray(0.15 + 0.7 * curve / float(curve.max()), dtype=np.float64)


# the keys and the mouse actions of the window. get_keyboard_help adds the shortcuts of the menus
KEYBOARD_HELP_ROWS: t.Final = (
    ("<b>Alt-Up</b>, <b>Alt-Down</b> in the list of light curves", "Move the light curve up or down (Option on a Mac)"),
    ("<b>Drag</b> across the plot", "Select the time range"),
    ("<b>Double-click</b> the plot", "Get the full time range"),
)


def run_viewer(tokens: "Sequence[str]") -> None:
    """Open the window of the viewer, and print the command of the last plot when the window closes.

    The Dock icon also takes a file, e.g. a reference light curve, which the active window adds to its light curves.
    """
    run_viewer_application(APPLICATION_NAME, get_icon_curve(), open_window, tokens, ("public.folder", "public.data"))


def open_window(tokens: "Sequence[str]", windows: "list[QtWidgets.QMainWindow]") -> str | None:
    """Open a window of the viewer for the plotlightcurves arguments in tokens, or return the reason for no window."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    from artistools.commands import get_path

    # the Settings window can give a new window options, e.g. -figscale, that the command does not give
    viewer = LightCurveViewer(add_default_options(make_parser(addargs), tokens), mplfig.Figure())
    window = make_window(APPLICATION_NAME)
    # a command with no path reads the model of the working folder
    modelnames = [
        resolve_modelpath(path).name for path in viewer.values.lightcurves if not path_is_reference_lightcurve(path)
    ] or [resolve_modelpath(viewer.runfolders[0]).name]
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

    fittimer = make_timer(window, FIT_MILLISECONDS)

    def on_resize() -> None:
        fit_canvas(canvas, viewer.figsize, plotarea)
        # a new plot takes up to 1 s, thus the plot takes the new shape only when the resize stops
        fittimer.start()

    plotarea = make_plot_area(canvas, on_resize)
    sidebar, panellayout = make_sidebar()
    make_central_splitter(window, plotarea, sidebar)
    helptexts = viewer.helptexts

    _, lightcurvegrid = add_section(panellayout, "Light curves")
    # this function defines the handler later, thus the lambda finds the handler when the user presses the key
    lightcurvelist = make_reorder_list(lambda step: on_move_lightcurve(step))  # ruff:ignore[unnecessary-lambda]
    # the widget of each row shows the text beside its ✕, thus the list draws no text of its own
    lightcurvelist.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    lightcurvelist.setToolTip(
        "The ARTIS models and the reference light curves of the plot, in the order of the command. The order sets the"
        " -label and the style of each series. Drag a row, or press Alt-Up or Alt-Down (Option on a Mac), to change"
        " the order. The command gives a file from the reference data of artistools by its name alone."
    )
    addmodelbutton = QtWidgets.QPushButton("Add Model…")
    addmodelbutton.setToolTip("Add the folder of an ARTIS run")
    referenceedit = QtWidgets.QLineEdit()
    referenceedit.setPlaceholderText("Add a reference light curve, e.g. AT2017gfo")
    referenceedit.setToolTip(
        "Type part of the name of a bolometric light curve in the data of artistools, then press Return. A name of a"
        " file in the working folder also works."
    )
    referencecompleter = make_completer(get_reference_lightcurve_names(), referenceedit)
    referenceedit.setCompleter(referencecompleter)
    openreferencebutton = QtWidgets.QPushButton("Open…")
    openreferencebutton.setToolTip("Add the file of a bolometric reference light curve from a folder")
    addrow = QtWidgets.QHBoxLayout()
    addrow.addWidget(referenceedit, 1)
    addrow.addWidget(openreferencebutton)
    addrow.addWidget(addmodelbutton)
    lightcurvegrid.addWidget(lightcurvelist, 0, 0, 1, -1)
    lightcurvegrid.addLayout(addrow, 1, 0, 1, -1)
    # the index of an item: 0 for the UVOIR light curve of the r-packets, and 1 for the gamma packets (--gamma)
    packetbox = QtWidgets.QComboBox()
    for text, tooltip in (
        ("UVOIR", "The ultraviolet, optical, and infrared (UVOIR) light curve of the radiation packets (r-packets)"),
        ("\N{GREEK SMALL LETTER GAMMA}-rays", f"--gamma: {helptexts.get('gamma', '')}"),
    ):
        packetbox.addItem(text)
        packetbox.setItemData(packetbox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    packetbox.setToolTip(
        "The light curve of the r-packets (UVOIR), or of the gamma packets (\N{GREEK SMALL LETTER GAMMA}-rays, --gamma)"
    )
    # the index of an item: 0 reads the light curve files of ARTIS, and 1 reads the packets files (--frompackets)
    datasourcebox = QtWidgets.QComboBox()
    for text, tooltip in (
        ("Text files", "Read light_curve.out, gamma_light_curve.out, and light_curve_res.out"),
        ("Packets files", f"--frompackets: {helptexts.get('frompackets', '')}"),
    ):
        datasourcebox.addItem(text)
        datasourcebox.setItemData(datasourcebox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    topnucsbox = QtWidgets.QSpinBox()
    topnucsbox.setRange(0, 50)
    topnucsbox.setSpecialValueText("none")
    topnucsbox.setToolTip(f"-topnucs: {helptexts.get('topnucs', '')}. The option reads the packets files")
    topnucsbox.setKeyboardTracking(False)
    pelletcheck = QtWidgets.QCheckBox("--use_pellet_decay_time")
    cmfcheck = QtWidgets.QCheckBox("--plotcmf")
    invalidcheck = QtWidgets.QCheckBox("--plotinvalidpart")
    for widget, dest in ((cmfcheck, "plotcmf"), (invalidcheck, "plotinvalidpart")):
        widget.setToolTip(helptexts.get(dest, ""))
    add_row(
        lightcurvegrid, 2, [QtWidgets.QLabel("Packets:"), packetbox, QtWidgets.QLabel("--frompackets"), datasourcebox]
    )
    add_row(lightcurvegrid, 3, [QtWidgets.QLabel("-topnucs"), topnucsbox, pelletcheck])
    add_row(lightcurvegrid, 4, [cmfcheck, invalidcheck])

    _, energygrid = add_section(panellayout, "Energy rates")
    # a check box for each particle and each energy rate. The columns of deposition.out decide which ones are on
    energychecks: dict[tuple[str, str], QtWidgets.QCheckBox] = {}
    energyheaders = {
        "deposition": ("Deposition", "The deposition rate: the energy that the particles give to the ejecta"),
        "emission": (
            "Monte Carlo",
            "The Monte Carlo emission rate, which ARTIS counts at the decays of the pellets in the run",
        ),
        "analyticemission": (
            "Analytical",
            "The analytical emission rate, which ARTIS calculates from the decay rates of the nuclides",
        ),
        "thermalisation": (
            "Thermalisation",
            (
                "The deposition rate over the emission rate, in a panel below the light curves. The positrons take"
                " the analytical emission rate"
            ),
        ),
    }
    # the two emission rates share one heading above their own headings, thus both columns are clearly emission rates
    emissionheader = QtWidgets.QLabel("Emission rate")
    emissionheader.setToolTip("The energy that the decays give to each particle (-emission and -analyticemission)")
    energygrid.addWidget(emissionheader, 0, 2, 1, 2, QtCore.Qt.AlignmentFlag.AlignHCenter)
    for column, dest in enumerate(ENERGYRATEDESTS, start=1):
        text, tooltip = energyheaders[dest]
        header = QtWidgets.QLabel(text)
        header.setToolTip(f"{tooltip} (-{dest})")
        energygrid.addWidget(header, 1, column, QtCore.Qt.AlignmentFlag.AlignHCenter)
        # the columns share the width of the section, thus each check box is under the middle of its heading
        energygrid.setColumnStretch(column, 1)
    for row, particle in enumerate(DEPOSITIONCHOICES, start=2):
        particlelabel = QtWidgets.QLabel(PARTICLETEXTS[particle])
        particlelabel.setToolTip(f"The word {particle} of the command")
        energygrid.addWidget(particlelabel, row, 0)
        for column, dest in enumerate(ENERGYRATEDESTS, start=1):
            if not get_energy_rate_column_names(dest, particle):
                continue
            check = QtWidgets.QCheckBox()
            check.setAccessibleName(f"-{dest} {particle}")
            # the grid has no text in a cell, thus the check box takes the middle of its column
            energygrid.addWidget(check, row, column, QtCore.Qt.AlignmentFlag.AlignHCenter)
            energychecks[dest, particle] = check

    _, timegrid = add_section(panellayout, "Time [d]")
    timerangeslider, set_timerange_positions, connect_timerange, _ = make_range_slider(SLIDER_STEPS)
    timeminedit, timemaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    zoomtip = " Drag across the plot to select a range. Double-click the plot to get the full range."
    timerangeslider.setToolTip("The minimum and the maximum of the time axis." + zoomtip)
    for column, (widget, dest) in enumerate((
        (timeminedit, "timemin"),
        (timerangeslider, ""),
        (timemaxedit, "timemax"),
    )):
        if dest:
            assert isinstance(widget, QtWidgets.QLineEdit)
            widget.setFixedWidth(110)
            widget.setPlaceholderText("auto")
            widget.setToolTip(helptexts.get(dest, "") + zoomtip)
        timegrid.addWidget(widget, 0, column)
    timegrid.setColumnStretch(1, 1)
    # the index of an item: 0 for a linear time axis, and 1 for a log time axis (--logscalex). The command has no
    # -xscale, thus the box gives no automatic scale as the y scale box does
    xscalebox = QtWidgets.QComboBox()
    for text, tooltip in (("Linear", "A linear time axis"), ("Log", f"--logscalex: {helptexts.get('logscalex', '')}")):
        xscalebox.addItem(text)
        xscalebox.setItemData(xscalebox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    xscalebox.setToolTip("The scale of the time axis. Log gives --logscalex")
    add_row(timegrid, 1, [QtWidgets.QLabel("x scale:"), xscalebox])

    _, ygrid = add_section(panellayout, "y axis")
    lumunitbox, yscalebox = QtWidgets.QComboBox(), QtWidgets.QComboBox()
    for unit, text, flag in LUMUNITS:
        lumunitbox.addItem(text, unit)
        lumunitbox.setItemData(
            lumunitbox.count() - 1,
            f"{flag}: {helptexts.get(flag.lstrip('-'), '')}" if flag else "The luminosity in erg/s",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
    lumunitbox.setToolTip("The unit of the luminosity and of the energy rates")
    # each item holds its -yscale choice, because the text of the "auto" item gives the scale of the drawn plot
    for yscale in viewer.yscalechoices:
        yscalebox.addItem(yscale.capitalize(), yscale)
    yscalebox.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
    yscalebox.setToolTip(helptexts.get("yscale", ""))
    setyrangebutton = QtWidgets.QPushButton("Set current y range")
    setyrangebutton.setToolTip(
        "Set y min and y max to the current range of the y axis. The axis then stays the same when a different option"
        " changes. Clear a field to get the automatic limit at that end again."
    )
    yminedit, ymaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    for edit, dest in ((yminedit, "ymin"), (ymaxedit, "ymax")):
        edit.setFixedWidth(110)
        edit.setPlaceholderText("auto")
        edit.setToolTip(helptexts.get(dest, ""))
    add_row(ygrid, 0, [QtWidgets.QLabel("Unit:"), lumunitbox, QtWidgets.QLabel("-yscale"), yscalebox])
    add_row(ygrid, 1, [QtWidgets.QLabel("-ymin"), yminedit, QtWidgets.QLabel("-ymax"), ymaxedit, setyrangebutton])

    _, directiongrid = add_section(panellayout, "Viewing direction")
    directionkindbox = QtWidgets.QComboBox()

    def show_direction_kinds() -> None:
        """Give the box the kinds of viewing direction of the first run, which Add Model or a new order can change."""
        with QtCore.QSignalBlocker(directionkindbox):
            directionkindbox.clear()
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

    show_direction_kinds()
    usedegreescheck = QtWidgets.QCheckBox("--usedegrees")
    usedegreescheck.setToolTip(helptexts.get("usedegrees", ""))
    add_row(directiongrid, 0, [directionkindbox, usedegreescheck])
    # the plot can show several directions at the same time, thus each direction bin has a check box. The list scrolls, and
    # the label of a bin is long, thus the list takes the full width of the sidebar
    directionbox = QtWidgets.QScrollArea()
    directionbox.setWidgetResizable(True)
    directionbox.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    directionbox.setToolTip(
        "The direction bins of the plot, or the observers of the virtual packets. A light curve from the text files"
        " reads light_curve_res.out, and the observers read the packets files"
    )
    directionchecks: dict[int, QtWidgets.QCheckBox] = {}
    directiongrid.addWidget(directionbox, 1, 0, 1, -1)
    # the labels of the direction bins come from the files of the run, thus the window reads them one time for each kind
    directionchoices: dict[tuple[str, bool], list[tuple[int, str]]] = {}
    shownchoices: tuple[str, bool] | None = None

    def get_direction_choices_of_kind(directionkind: str, usedegrees: bool) -> list[tuple[int, str]]:
        if not directionkind:
            return []
        if (directionkind, usedegrees) not in directionchoices:
            directionchoices[directionkind, usedegrees] = get_direction_choices(
                viewer.runfolders[0], directionkind, usedegrees=usedegrees
            )
        return directionchoices[directionkind, usedegrees]

    def show_direction_choices(directionkind: str, usedegrees: bool) -> bool:
        """Fill the list of the direction bins with a check box for each bin of a kind. Return whether it is new."""
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
            check.clicked.connect(on_direction)
            checklayout.addWidget(check)
            directionchecks[dirbin] = check
        checklayout.addStretch(1)
        # the list shows up to 6 bins, and a longer list scrolls
        shownbins = min(max(len(directionchecks), 1), 6)
        lineheight = max((check.sizeHint().height() for check in directionchecks.values()), default=20)
        directionbox.setFixedHeight(shownbins * (lineheight + 2) + 10)
        directionbox.setWidget(checklist)
        shownchoices = (directionkind, usedegrees)
        return True

    _, appearancegrid = add_section(panellayout, "Appearance")
    # the box edits the row of -figscale in the other options, as the box of the other viewers does
    figscalebox = QtWidgets.QDoubleSpinBox()
    figscalebox.setRange(0.1, 10.0)
    figscalebox.setSingleStep(0.1)
    figscalebox.setDecimals(2)
    figscalebox.setKeyboardTracking(False)
    figscalebox.setToolTip(helptexts.get("figscale", ""))
    add_row(appearancegrid, 0, [QtWidgets.QLabel("-figscale"), figscalebox])
    defaultfigscale: float = viewer.parser.get_default("figscale")
    defaultdpi: int = viewer.parser.get_default("dpi")
    figuresection = add_figure_section(window, panellayout, viewer.values.dpi or defaultdpi)
    _, optiongrid = add_section(panellayout, "Other options")

    def on_option_rows(rows: OptionRows) -> None:
        apply(dc.replace(viewer.values, otheroptions=rows))

    def on_figscale(figscale: float) -> None:
        change = None if math.isclose(figscale, defaultfigscale) else (format(figscale, "g"),)
        on_option_rows(set_row_values(viewer.values.otheroptions, {"-figscale": change}))

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
        figscalebox,
        figuresection.dpibox,
        packetbox,
        datasourcebox,
        topnucsbox,
        pelletcheck,
        cmfcheck,
        invalidcheck,
        *energychecks.values(),
        timerangeslider,
        xscalebox,
        lumunitbox,
        yscalebox,
        directionkindbox,
        usedegreescheck,
    ]

    # the time slider acts on log10(t) over the times of the runs
    logtrange = (0.0, 1.0)
    shownruns: tuple[str, ...] | None = None

    def show_run_ranges() -> None:
        """Set the range of the time slider and the direction bins from the runs of the plot, e.g. after Add Model."""
        nonlocal logtrange, shownchoices, shownruns
        logtrange = (math.log10(max(viewer.timebounds[0], 1e-3)), math.log10(viewer.timebounds[1]))
        # the direction bins come from the new first run
        shownchoices = None
        directionchoices.clear()
        show_direction_kinds()
        shownruns = viewer.runlightcurves

    def to_position(value: float) -> int:
        low, high = logtrange
        return round(SLIDER_STEPS * (min(max(math.log10(max(value, 1e-300)), low), high) - low) / max(high - low, 1e-9))

    def from_position(position: int) -> float:
        low, high = logtrange
        return float(10.0 ** (low + (high - low) * position / SLIDER_STEPS))

    def show_lightcurves(lightcurves: "Sequence[str]") -> None:
        """Show a row for each light curve, with a ✕ at the right end of the row that removes the light curve."""
        lightcurvelist.clear()
        models = [path for path in lightcurves if get_artis_run_folders([path])]
        rowheight = lightcurvelist.fontMetrics().lineSpacing() + 4
        for path in lightcurves:
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            name = Path(path).name or path
            removebutton = make_glyph_button("✕", f"Remove {name} from the plot", f"Remove {name}")
            if models == [path]:
                removebutton.setEnabled(False)
                removebutton.setToolTip("The plot needs one ARTIS model at least. Add a different model first")
            # the new list replaces this row, thus the removal waits until the click ends
            removebutton.clicked.connect(
                partial(QtCore.QTimer.singleShot, 0, window, partial(on_remove_lightcurve, path))
            )
            row = QtWidgets.QWidget()
            rowlayout = QtWidgets.QHBoxLayout(row)
            # a long path shows its start and its end, and the width of the box sets the length
            rowlayout.setContentsMargins(4, 0, 2, 0)
            rowlayout.addWidget(make_elided_label(get_lightcurve_item_text(path)), 1)
            rowlayout.addWidget(removebutton)
            rowheight = max(rowheight, row.sizeHint().height())
            lightcurvelist.addItem(item)
            lightcurvelist.setItemWidget(item, row)
        for index in range(lightcurvelist.count()):
            if (item := lightcurvelist.item(index)) is not None:
                item.setSizeHint(QtCore.QSize(0, rowheight))
        # the list has the height of its light curves, from 2 to 4 rows, and a longer list scrolls
        shownrows = min(max(lightcurvelist.count(), 2), 4)
        lightcurvelist.setFixedHeight(shownrows * rowheight + 2 * lightcurvelist.frameWidth() + 4)

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        blockers = [QtCore.QSignalBlocker(widget) for widget in signalwidgets]
        values = viewer.values
        # Undo or a rejected change can give a different list of light curves, and its runs have different times
        if values.lightcurves != viewer.runlightcurves:
            viewer.load_runs(values.lightcurves)
        if viewer.runlightcurves != shownruns:
            show_run_ranges()
        shownlightcurves = [
            lightcurvelist.item(index).data(QtCore.Qt.ItemDataRole.UserRole) for index in range(lightcurvelist.count())
        ]
        if shownlightcurves != list(values.lightcurves):
            show_lightcurves(values.lightcurves)
        packetbox.setCurrentIndex(1 if values.gamma else 0)
        # -topnucs reads the packets, thus the box shows that and takes no choice
        datasourcebox.setCurrentIndex(1 if values.frompackets or values.topnucs else 0)
        datasourcebox.setEnabled(not values.topnucs)
        datasourcebox.setToolTip(
            "-topnucs reads the packets files" if values.topnucs else "The files of the light curves of the ARTIS runs"
        )
        topnucsbox.setValue(values.topnucs)
        pelletcheck.setChecked(values.usepelletdecaytime)
        readspackets = values.frompackets or bool(values.topnucs)
        pelletcheck.setEnabled(readspackets or values.usepelletdecaytime)
        pelletcheck.setToolTip(
            helptexts.get("use_pellet_decay_time", "")
            if readspackets
            else "Only the packets give the decay time of a pellet. Select the packets files first"
        )
        cmfcheck.setChecked(values.plotcmf)
        cmfcheck.setEnabled(values.lumunit != "mag" or values.plotcmf)
        cmfcheck.setToolTip(
            helptexts.get("plotcmf", "") if values.lumunit != "mag" else "A magnitude has no comoving frame luminosity"
        )
        invalidcheck.setChecked(values.plotinvalidpart)
        for (dest, particle), check in energychecks.items():
            ischecked = particle in getattr(values, dest)
            check.setChecked(ischecked)
            reason = viewer.get_energy_rate_reason(dest, particle)
            # a rate that the plot has stays available, thus the user can remove it
            check.setEnabled(reason is None or ischecked)
            check.setToolTip(reason or f"-{dest} {particle}: {helptexts.get(dest, '')}")
        set_timerange_positions(
            to_position(float(values.timemin) if values.timemin else viewer.timebounds[0]),
            to_position(float(values.timemax) if values.timemax else viewer.timebounds[1]),
        )
        set_edit_text(timeminedit, values.timemin)
        set_edit_text(timemaxedit, values.timemax)
        xscalebox.setCurrentIndex(1 if values.logscalex else 0)
        lumunitbox.setCurrentIndex(lumunitbox.findData(values.lumunit))
        yscalebox.setCurrentIndex(yscalebox.findData(values.yscale))
        # a magnitude is a logarithm already, thus plotlightcurves gives it no log scale
        yscalebox.setEnabled(values.lumunit != "mag")
        set_edit_text(yminedit, values.ymin)
        set_edit_text(ymaxedit, values.ymax)
        directionkindbox.setCurrentIndex(max(directionkindbox.findData(values.directionkind), 0))
        usedegreescheck.setChecked(values.usedegrees)
        usedegreescheck.setEnabled(bool(values.directionkind))
        isnewlist = show_direction_choices(values.directionkind, values.usedegrees)
        for dirbin, check in directionchecks.items():
            check.setChecked(dirbin in values.directionbins)
        # a new list scrolls to the first checked bin, which can be far down a list of 100 bins
        if isnewlist and (firstcheck := directionchecks.get(values.directionbins[0] if values.directionbins else -3)):
            QtCore.QTimer.singleShot(0, window, partial(directionbox.ensureWidgetVisible, firstcheck))
        # "All directions" has no bins, thus the list of the bins shows only for a kind of direction
        directionbox.setVisible(bool(values.directionkind))
        set_option_rows(values.otheroptions)
        set_spin_value(figuresection.dpibox, values.dpi or defaultdpi)
        set_spin_value(
            figscalebox, float((get_row_values(values.otheroptions, "-figscale") or (str(defaultfigscale),))[0])
        )
        set_command_text(commandtext, viewer.get_command())
        set_command_text(pythontext, get_python_code(viewer.parser, viewer.get_plot_tokens()))
        for blocker in blockers:
            blocker.unblock()
        fit_canvas(canvas, viewer.figsize, plotarea)

    def after_draw(message: str | None) -> None:
        # matplotlib keeps the connections of the mouse in the figure, and each plot has a new figure
        connect_mouse_to_figure()
        # -yscale auto reads the drawn values, thus only the drawn plot gives the scale that it chose
        if message is None and viewer.values.yscale == "auto" and plot_shows_values() and viewer.axis is not None:
            yscalebox.setItemText(yscalebox.findData("auto"), f"Auto ({viewer.axis.get_yscale()})")
        # a new panel changes the height of the figure, thus the plot can need a new -figwidthscale
        fittimer.start()

    queue = DrawQueue(
        window, viewer, statusbar, show_values, after_draw, render=viewer.render, keep_on_undo=keep_figwidthscale
    )

    def apply(values: ControlValues, *, undoable: bool = True) -> None:
        queue.apply(values, undoable=undoable)

    def fit_figwidthscale() -> None:
        """Give the plot the -figwidthscale that fills the plot area."""
        figwidthscale = get_new_figwidthscale(
            plotarea, viewer.figsize, viewer.values.figwidthscale, viewer.get_fitted_figwidthscale
        )
        if figwidthscale is not None:
            # the window sets the width, thus Undo does not return to an old width
            apply(dc.replace(viewer.values, figwidthscale=figwidthscale), undoable=False)

    fittimer.timeout.connect(fit_figwidthscale)

    def show_error(message: str) -> None:
        show_status_message(statusbar, message, "")
        show_values()

    def plot_shows_values() -> bool:
        """Return True if the plot on the screen has the values of the controls.

        DrawQueue gives the viewer the new values immediately, and the old plot stays until the worker draws the new
        one. A handler that reads the plot, e.g. the y limits, must not put them into different values.
        """
        return queue.drawnvalues == viewer.values

    def on_series() -> None:
        topnucs = topnucsbox.value()
        frompackets = datasourcebox.currentIndex() == 1 and not topnucs
        readspackets = frompackets or bool(topnucs)
        values = dc.replace(
            viewer.values,
            gamma=packetbox.currentIndex() == 1,
            # a light curve of -topnucs reads the packets, thus the box keeps the choice of the user for later
            frompackets=frompackets or (viewer.values.frompackets and bool(topnucs)),
            topnucs=topnucs,
            # only the packets give the decay time of a pellet
            usepelletdecaytime=pelletcheck.isChecked() and readspackets,
            plotcmf=cmfcheck.isChecked(),
            plotinvalidpart=invalidcheck.isChecked(),
        )
        # the observers of the virtual packets need the packets files
        if values.directionkind == "vpkt" and not readspackets:
            values = dc.replace(values, directionkind="", directionbins=())
        apply(values)

    def get_checked_particles(dest: str) -> tuple[str, ...]:
        return sort_particles({
            particle for (checkdest, particle), check in energychecks.items() if checkdest == dest and check.isChecked()
        })

    def on_energy_rates() -> None:
        apply(
            dc.replace(
                viewer.values,
                deposition=get_checked_particles("deposition"),
                emission=get_checked_particles("emission"),
                analyticemission=get_checked_particles("analyticemission"),
                thermalisation=get_checked_particles("thermalisation"),
            )
        )

    def set_time_limits(low: float, high: float) -> None:
        # a value of 3 significant digits gives a short command, and a text field gives an exact value
        low, high = float(f"{low:.3g}"), float(f"{high:.3g}")
        if low < high:
            apply(dc.replace(viewer.values, timemin=format(low, ".10g"), timemax=format(high, ".10g")))

    def on_timerange(handle: int, position: int) -> None:
        """Set the limit of the handle that moved, and keep the other limit as its text field gives it."""
        limit = float(f"{from_position(position):.3g}")
        values = viewer.values
        high = float(values.timemax) if values.timemax else viewer.timebounds[1]
        low = float(values.timemin) if values.timemin else viewer.timebounds[0]
        if handle == 0 and limit < high:
            apply(dc.replace(values, timemin=format(limit, ".10g")))
        elif handle == 1 and limit > low:
            apply(dc.replace(values, timemax=format(limit, ".10g")))

    def read_limit_fields(fields: "Sequence[tuple[QtWidgets.QLineEdit, str]]") -> list[str] | None:
        """Return the text of each field as a number with no rounding, or "" for an empty field. Show an error for text."""
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

    def on_timeedit() -> None:
        if (limits := read_limit_fields([(timeminedit, "-timemin"), (timemaxedit, "-timemax")])) is not None:
            apply(dc.replace(viewer.values, timemin=limits[0], timemax=limits[1]))

    def on_yedit() -> None:
        if (limits := read_limit_fields([(yminedit, "-ymin"), (ymaxedit, "-ymax")])) is not None:
            apply(dc.replace(viewer.values, ymin=limits[0], ymax=limits[1]))

    def on_set_y_range() -> None:
        if not plot_shows_values() or viewer.axis is None:
            show_error("The plot on the screen does not show the new values yet. Wait for the plot, then try again")
            return
        # the limits of the plot on the screen become the limits of the command, thus the plot does not change. A
        # magnitude axis runs from the faint end to the bright end, and -ymin and -ymax give the low and high numbers
        low, high = (get_short_number(limit) for limit in sorted(viewer.axis.get_ylim()))
        apply(dc.replace(viewer.values, ymin=low, ymax=high))

    def on_axes() -> None:
        lumunit: LumUnit = lumunitbox.currentData()
        values = viewer.values
        # the limits of a luminosity do not apply to a magnitude, and the reverse
        if lumunit != values.lumunit:
            values = dc.replace(values, ymin="", ymax="", plotcmf=values.plotcmf and lumunit != "mag")
        # a magnitude is a logarithm already, and a log scale of its negative values gives no plot
        yscale = viewer.defaultyscale if lumunit == "mag" else yscalebox.currentData()
        apply(dc.replace(values, lumunit=lumunit, yscale=yscale, logscalex=xscalebox.currentIndex() == 1))

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
        # the observers of the virtual packets need the packets files
        if directionkind == "vpkt" and not values.topnucs:
            values = dc.replace(values, frompackets=True)
        apply(values)

    def apply_lightcurves(lightcurves: "Sequence[str]") -> None:
        """Read the runs of a new list of light curves, and apply the list."""
        viewer.load_runs(lightcurves)
        values = dc.replace(viewer.values, lightcurves=tuple(lightcurves))
        # a new first run can have no data for a kind of viewing direction, e.g. the observers of the virtual packets
        if values.directionkind not in viewer.directionkinds:
            values = dc.replace(values, directionkind="", directionbins=())
        apply(values)

    def add_lightcurves(paths: "Sequence[str]") -> None:
        """Add each light curve whose full path the list does not hold yet, e.g. "." for the working folder.

        A cancelled dialog gives no path, and the list then stays with no message.
        """
        if not paths:
            return
        lightcurves = viewer.values.lightcurves
        shown = {get_lightcurve_path(path) for path in lightcurves}
        newpaths: list[str] = []
        for path in paths:
            if (fullpath := get_lightcurve_path(path)) not in shown:
                shown.add(fullpath)
                newpaths.append(path)
        if not newpaths:
            show_error("The list of light curves already holds each of these light curves")
            return
        apply_lightcurves((*lightcurves, *newpaths))

    def on_add_model() -> None:
        # the dialog shows only local folders, thus it opens in the working folder if the first run is on a different host
        startfolder = (
            Path.cwd() if is_remote_path(viewer.runfolders[0]) else Path(viewer.runfolders[0]).absolute().parent
        )
        folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Add an ARTIS model", str(startfolder))
        if not folder:
            return
        if not get_artis_run_folders([folder]):
            show_error(f"{folder} is not the folder of an ARTIS run, which holds input.txt")
            return
        add_lightcurves([folder])

    referencefolder = get_path("artistools_dir") / "data" / "lightcurves" / "bollightcurves"

    def on_open_reference() -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            window, "Add reference light curves", str(referencefolder)
        )
        add_lightcurves([get_reference_token(filename) for filename in filenames])

    def add_reference_name(name: str) -> None:
        name = name.strip()
        if not name:
            return
        if find_bol_reflightcurve_file(name) is None:
            show_error(f"No reference light curve {name} is in the working folder or in the reference data")
            return
        referenceedit.clear()
        add_lightcurves([name])

    def on_complete_reference(name: str) -> None:
        # the completer puts the name in the field after this handler, thus clear the field after the event
        QtCore.QTimer.singleShot(0, referenceedit.clear)
        add_reference_name(name)

    def on_lightcurves_moved() -> None:
        """Apply the order of the rows after a drag in the list of light curves."""
        order = [
            lightcurvelist.item(index).data(QtCore.Qt.ItemDataRole.UserRole) for index in range(lightcurvelist.count())
        ]
        if order != list(viewer.values.lightcurves):
            apply_lightcurves(order)

    def on_move_lightcurve(step: int) -> None:
        """Move the selected light curve one row up or down, and keep the selection on it."""
        lightcurves = list(viewer.values.lightcurves)
        row = lightcurvelist.currentRow()
        if row < 0 or not 0 <= row + step < len(lightcurves):
            return
        lightcurves.insert(row + step, lightcurves.pop(row))
        apply_lightcurves(lightcurves)
        lightcurvelist.setCurrentRow(row + step)

    def on_remove_lightcurve(path: str) -> None:
        lightcurves = tuple(other for other in viewer.values.lightcurves if other != path)
        # the time controls and the energy rates read a run, thus the plot needs an ARTIS model
        if not get_artis_run_folders(lightcurves):
            show_error("The plot needs one ARTIS model at least. Add a different model before you remove this one")
            return
        apply_lightcurves(lightcurves)

    def on_copy() -> None:
        copy_text(viewer.get_command())
        show_status_note(statusbar, "Copied the command")

    def get_figure_tokens() -> list[str]:
        """Return the command of the plot with no -dpi. The Figure section gives the resolution."""
        return viewer.get_plot_tokens(dc.replace(viewer.values, dpi=None))

    def get_figure_choice() -> tuple[str, int]:
        """Return the format of the Figure section and the resolution of the command."""
        return get_figure_format(), viewer.values.dpi or defaultdpi

    def on_resolution(resolution: int) -> None:
        apply(dc.replace(viewer.values, dpi=None if resolution == defaultdpi else resolution))

    def on_copy_figure() -> None:
        from artistools.lightcurve.plotlightcurve import main as plotlightcurves_main

        plottokens = get_figure_tokens()
        copy_figure_of_command(queue, statusbar, plotlightcurves_main, viewer.parser, plottokens, get_figure_choice())

    def on_copy_python() -> None:
        copy_text(get_python_code(viewer.parser, viewer.get_plot_tokens()))
        show_status_note(statusbar, "Copied the Python code")

    def on_save() -> None:
        from artistools.lightcurve.plotlightcurve import main as plotlightcurves_main

        plottokens = get_figure_tokens()
        save_figure_of_command(
            window, statusbar, plotlightcurves_main, "plotlightcurves", plottokens, viewer.parser, get_figure_choice()
        )

    def on_open_model() -> None:
        if (message := open_model_window(window, open_window, windows)) is not None:
            show_error(message)

    def on_help() -> None:
        QtWidgets.QMessageBox.information(
            window, "Keys and mouse actions", get_keyboard_help(KEYBOARD_HELP_ROWS, menutexts)
        )

    def on_plot_menu(_frameindex: int, _event: t.Any) -> None:
        """Show the actions on the figure under the pointer, as the context menu of a Mac app does."""
        menu = QtWidgets.QMenu(window)
        menu.addAction("Copy Figure").triggered.connect(on_copy_figure)
        menu.addAction("Save Figure…").triggered.connect(on_save)
        menu.exec(QtGui.QCursor.pos())
        # the window is the parent of the menu, thus without this the window keeps each menu until it closes
        menu.deleteLater()

    def on_open_recent(folder: str) -> None:
        if (message := open_model_folder(folder, open_window, windows)) is not None:
            show_error(message)

    def on_drop(paths: list[str]) -> None:
        """Add each dropped ARTIS run and each dropped reference file to the light curves of the plot."""
        folders = [path for path in paths if Path(path).is_dir()]
        runs = [folder for folder in folders if get_artis_run_folders([folder])]
        if len(runs) < len(folders):
            show_error("A dropped folder is not the folder of an ARTIS run, which holds input.txt")
        add_lightcurves([*runs, *(get_reference_token(path) for path in paths if Path(path).is_file())])

    def on_closed() -> None:
        print(viewer.get_command())
        queue.close()
        # the list holds a reference to each open window, thus Python does not delete the window. A closed window
        # leaves the list
        windows.remove(window)

    menucallbacks = {
        "Open Model…": on_open_model,
        "Save Figure…": on_save,
        "Close Window": window.close,
        "Undo": queue.undo,
        "Redo": queue.redo,
        "Copy Figure": on_copy_figure,
        "Copy Command": on_copy,
        "Copy Python": on_copy_python,
        "Keys and Mouse Actions": on_help,
    }
    menutexts = add_menus(window, menucallbacks, queue, None, open_folder=on_open_recent)
    set_drop_handler(window, on_drop)
    follow_colour_scheme(window, viewer, queue)

    # the window keeps its command at a quit, and the next start opens the window again
    def get_session_tokens() -> list[str]:
        # a command with no path reads the working folder, and the next start can be in a different folder
        tokens = viewer.get_plot_tokens()
        return [str(Path.cwd()), *tokens] if viewer.values.lightcurves == DEFAULT_LIGHTCURVES else tokens

    window.setProperty("sessiontokens", get_session_tokens)

    for checkbox in (pelletcheck, cmfcheck, invalidcheck):
        checkbox.toggled.connect(on_series)
    packetbox.currentIndexChanged.connect(on_series)
    datasourcebox.currentIndexChanged.connect(on_series)
    topnucsbox.valueChanged.connect(on_series)
    for check in energychecks.values():
        check.toggled.connect(on_energy_rates)
    connect_timerange(on_timerange)
    timeminedit.editingFinished.connect(on_timeedit)
    timemaxedit.editingFinished.connect(on_timeedit)
    xscalebox.currentIndexChanged.connect(on_axes)
    lumunitbox.currentIndexChanged.connect(on_axes)
    yscalebox.currentIndexChanged.connect(on_axes)
    setyrangebutton.clicked.connect(on_set_y_range)
    yminedit.editingFinished.connect(on_yedit)
    ymaxedit.editingFinished.connect(on_yedit)
    directionkindbox.currentIndexChanged.connect(on_direction)
    usedegreescheck.toggled.connect(on_direction)
    figuresection.copybutton.clicked.connect(on_copy_figure)
    figuresection.dpibox.valueChanged.connect(on_resolution)
    figscalebox.valueChanged.connect(on_figscale)
    figuresection.savebutton.clicked.connect(on_save)
    addmodelbutton.clicked.connect(on_add_model)
    # the list moves the row at the end of the drop, thus the new order applies after the drop
    lightcurvelist.model().rowsMoved.connect(lambda: QtCore.QTimer.singleShot(0, window, on_lightcurves_moved))
    openreferencebutton.clicked.connect(on_open_reference)
    referencecompleter.activated.connect(on_complete_reference)
    referenceedit.returnPressed.connect(lambda: add_reference_name(referenceedit.text()))
    copybutton.clicked.connect(on_copy)
    pythoncopybutton.clicked.connect(on_copy_python)
    statusbar.helpbutton.clicked.connect(on_help)
    window.destroyed.connect(on_closed)
    connect_mouse_to_figure = connect_plot_mouse(
        canvas,
        get_frames=viewer.get_frames,
        get_readout=lambda event, frame: get_readout(event.xdata, frame),
        readoutlabel=statusbar.readout,
        on_select=set_time_limits,
        on_reset=lambda: apply(dc.replace(viewer.values, timemin="", timemax="")),
        can_select=plot_shows_values,
        on_menu=on_plot_menu,
        show_tag=make_readout_tag(canvas),
    )

    show_window(window, viewer.figsize, lambda: fit_canvas(canvas, viewer.figsize, plotarea))
    show_values()
    return None
