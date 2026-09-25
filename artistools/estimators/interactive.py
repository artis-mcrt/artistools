"""Show the plot of plotestimators in a window with controls for the time, the cell, and the subplots."""

import argparse
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

from artistools.constants import C_cm_per_s
from artistools.constants import km_to_cm
from artistools.estimators.core import convert_estimator_batch_caches
from artistools.estimators.core import get_estimator_batch_states
from artistools.estimators.core import get_units_string
from artistools.estimators.core import join_cell_modeldata
from artistools.estimators.core import scan_estimators
from artistools.estimators.core import scan_parquet_file
from artistools.estimators.plotestimators import add_plot_columns
from artistools.estimators.plotestimators import addargs
from artistools.estimators.plotestimators import default_plotitem_has_data
from artistools.estimators.plotestimators import DIRECTIVES
from artistools.estimators.plotestimators import draw_plot
from artistools.estimators.plotestimators import get_default_plotlist
from artistools.estimators.plotestimators import get_default_x
from artistools.estimators.plotestimators import require_artis_folder
from artistools.estimators.plotestimators import resolve_positional_args
from artistools.estimators.plotestimators import resolve_snapshot_arguments
from artistools.estimators.plotestimators import SERIESTYPES
from artistools.estimators.plotestimators import time_is_given
from artistools.estimators.plotestimators import TIME_XVARIABLES
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.misc import exit_with_error
from artistools.misc import get_runfolders
from artistools.misc import get_time_range
from artistools.misc import get_timestep_times
from artistools.misc import parse_cli_args
from artistools.misc import path_is_codecomparison
from artistools.misc import separate_trailing_folders
from artistools.misc.general import call_in_child_process
from artistools.misc.modelinfo import get_runfolder_timesteps
from artistools.misc.modelinfo import get_runfolder_timesteps_cached
from artistools.plottools import LABELWIDTH_INCHES
from artistools.plottools import make_room_for_title
from artistools.plottools import plain_label
from artistools.plottools import RIGHTMARGIN_INCHES
from artistools.viewertools import add_command_section
from artistools.viewertools import add_menus
from artistools.viewertools import add_row
from artistools.viewertools import add_section
from artistools.viewertools import connect_plot_mouse
from artistools.viewertools import copy_command
from artistools.viewertools import DrawQueue
from artistools.viewertools import exit_for_other_actions
from artistools.viewertools import fit_canvas
from artistools.viewertools import FIT_MILLISECONDS
from artistools.viewertools import get_fitted_figwidthscale
from artistools.viewertools import get_helptexts
from artistools.viewertools import get_last_warning
from artistools.viewertools import get_line_readouts
from artistools.viewertools import get_menu_shortcut_texts
from artistools.viewertools import get_nearest_range_start
from artistools.viewertools import get_new_figwidthscale
from artistools.viewertools import get_option_row_tokens
from artistools.viewertools import get_option_tokens
from artistools.viewertools import make_central_splitter
from artistools.viewertools import make_option_table
from artistools.viewertools import make_parser
from artistools.viewertools import make_plot_area
from artistools.viewertools import make_range_slider
from artistools.viewertools import make_sidebar
from artistools.viewertools import make_slider
from artistools.viewertools import make_status_bar
from artistools.viewertools import make_timer
from artistools.viewertools import make_window
from artistools.viewertools import open_model_window
from artistools.viewertools import OptionRows
from artistools.viewertools import PLAY_MILLISECONDS
from artistools.viewertools import remove_options
from artistools.viewertools import run_command_step
from artistools.viewertools import run_command_step_outcome
from artistools.viewertools import save_figure_of_command
from artistools.viewertools import set_command_text
from artistools.viewertools import set_edit_text
from artistools.viewertools import show_window
from artistools.viewertools import split_option_rows
from artistools.viewertools import start_application
from artistools.viewertools import start_play_timer

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.axes as mplax
    import numpy.typing as npt
    from PySide6 import QtWidgets

    from artistools.estimators.core import EstimatorBatchCache

# the controls of the window give these arguments, thus the command drops the values that the user typed
CONTROLLED_DESTS: t.Final = frozenset({
    "plotitems",
    "plotlist",
    "modelpath",
    "modelgridindex",
    "timestep",
    "timedays",
    "timemin",
    "timemax",
    "x",
    "xmin",
    "xmax",
    "xbins",
    "markers",
    "colorbyion",
    "figwidthscale",
    "interactive",
})

# these options change only the output file. Save Figure in the File menu gives the file, thus the command drops them
OUTPUT_DESTS: t.Final = frozenset({"outputfile", "format", "show", "open"})

# these options give a different action from one plot, thus the table of the window does not offer them
TABLE_EXCLUDED_DESTS: t.Final = frozenset({"help", "multiplot", "makegif", "listvariables", "listnuclides"})

# the -x choices that are not an estimator column
XVARIABLES: t.Final = ("velocity", "beta", "time", "timestep", "modelgridindex")

APPLICATION_NAME: t.Final = "artistools plotestimators"

# matplotlib gives this label to the axes of a colour bar
COLORBAR_LABEL: t.Final = "<colorbar>"


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class ControlValues:
    """The values of the controls of the viewer, which give the options of the plotestimators command.

    The x limits, the cells, and the bins keep the text of the command, and an empty text gives no option.
    """

    first: int
    last: int
    x: str
    xmin: str
    xmax: str
    cells: str
    # the items of each subplot, e.g. ("Te", "TR") or ("populations", "Fe II", "Fe III", "yscale=log")
    subplots: tuple[tuple[str, ...], ...]
    markers: bool
    xbins: str
    colorbyion: bool
    figwidthscale: float
    otheroptions: OptionRows


class RenderedPlot(t.NamedTuple):
    """A figure that the worker thread drew, with the properties of the plot that the window shows.

    plotestimators chooses the bins, the markers, and the colours when the command gives no option, thus the last
    three fields hold the values that the plot used.
    """

    fig: mplfig.Figure
    isimage: bool
    xlimitscale: float
    xbins: int | None
    markers: bool
    colorbyion: bool


def check_viewer_args(args: argparse.Namespace) -> None:
    """Stop when args selects an action that is not one plot of estimators. The window shows one plot only."""
    exit_for_other_actions(
        "estimators",
        {
            "--multiplot": args.multiplot,
            "--makegif": args.makegif,
            "--listvariables": args.listvariables,
            "--listnuclides": args.listnuclides,
        },
    )


def get_plotitem_tokens(plotitems: "Sequence[t.Any]") -> tuple[str, ...]:
    """Return the -plot tokens of the items of one subplot of the default plot list.

    A directive has the form ["_yscale", "log"], and it becomes "yscale=log". A type of series has the form
    ["populations", ["Fe II", "Fe III"]], and its names follow it.
    """
    tokens: list[str] = []
    for plotitem in plotitems:
        if isinstance(plotitem, str):
            tokens.append(plotitem)
        elif str(plotitem[0]).startswith("_"):
            tokens.append(f"{str(plotitem[0]).removeprefix('_')}={plotitem[1]}")
        else:
            tokens.extend([str(plotitem[0]), *(str(name) for name in plotitem[1])])
    return tuple(tokens)


def get_estimator_timesteps(modelpath: Path, estimators: pl.LazyFrame) -> list[int]:
    """Return the timesteps that the estimators of the run hold.

    The first parquet cache of each run folder gives its timesteps, thus a large run needs no scan of all its rows. A
    run with no such cache, e.g. a run of classic ARTIS, reads the timestep column of the estimators.
    """
    # a codecomparison path names a reference file and not a folder, thus it has no run folders
    runfolders: Sequence[Path] = () if path_is_codecomparison(modelpath) else get_runfolders(modelpath)
    timesteps = {timestep for runfolder in runfolders for timestep in get_runfolder_timesteps(runfolder)}
    if not timesteps:
        timesteps = set(estimators.select(pl.col("timestep").unique()).collect().to_series().to_list())
    return sorted(timesteps)


def get_default_xvariable(tokens: "Sequence[str]") -> str:
    """Return the x variable that plotestimators takes for command tokens with no -x.

    -slice and -dimensionreduce set the x variable of a snapshot. Otherwise a command with no time plots against
    time, and a command with a time plots a snapshot against the velocity. The plot of tokens that plotestimators
    rejects gives the message, thus this function then returns the velocity.
    """
    xvariables: list[str] = []

    def resolve() -> None:
        args = parse_cli_args(addargs, None, None, tokens)
        resolve_snapshot_arguments(args)
        xvariables.append(args.x or get_default_x(timegiven=time_is_given(args), makegif=args.makegif))

    run_command_step(resolve, echo=False)
    return xvariables[0] if xvariables else "velocity"


def is_evolution(values: ControlValues) -> bool:
    """Return True if the plot of the values is a plot against time, and not a snapshot."""
    return values.x in TIME_XVARIABLES


def get_time_text(tmids: "Sequence[float]", values: ControlValues) -> str:
    """Return the text of the time field, which is the centre of the middle times of the range.

    EstimatorViewer.select_centre reads the same centre, thus a Return in the field with no edit keeps the range.
    """
    return f"{(tmids[values.first] + tmids[values.last]) / 2.0:.4g}"


def get_single_cell(cells: str) -> int | None:
    """Return the cell of a -cell text that names one cell, or None for a list, a range, or no cell.

    str.isdigit accepts a superscript digit such as "²", which int rejects, thus the test reads ASCII digits alone.
    """
    return int(cells) if cells.isascii() and cells.isdecimal() else None


def load_run(viewer: "EstimatorViewer") -> None:
    """Read the caches, the columns, the valid timesteps, and the cells of the run of the viewer.

    The viewer checks and converts the estimator caches of the run one time. Each plot then reads the caches of its
    own timesteps and cells, as the command does.
    """
    args = viewer.userargs
    isartisrun = not args.classicartis and not path_is_codecomparison(viewer.modelpath)
    viewer.batchcaches = get_batch_caches(viewer.modelpath) if isartisrun else None
    estimators, modelmeta = join_cell_modeldata(
        estimators=scan_estimators(
            modelpath=viewer.modelpath, classicartis=args.classicartis, batchcaches=viewer.batchcaches
        ),
        modelpath=viewer.modelpath,
    )
    _, viewer.estimatorcolumns = add_plot_columns(args, estimators, modelmeta)
    # a run that stopped early has no estimators for the last timesteps, and a plot of those timesteps fails
    viewer.validtimesteps = get_estimator_timesteps(viewer.modelpath, estimators) or list(range(len(viewer.tmids)))
    # ARTIS writes the estimators of each cell that holds matter
    lzmodel, modelmeta = get_modeldata(viewer.modelpath)
    dfcells = (
        add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") > 0.0)
        .select("modelgridindex", "vel_r_mid")
        .sort("modelgridindex")
        .collect()
    )
    viewer.cells = dfcells["modelgridindex"].to_list()
    viewer.cellvelocities = dict(zip(viewer.cells, dfcells["vel_r_mid"].to_list(), strict=True))

    # the command omits -plot while the subplots are the default subplots of plotestimators
    viewer.defaultsubplots = tuple(
        get_plotitem_tokens(plotitems)
        for plotitems in get_default_plotlist()
        if default_plotitem_has_data(plotitems, viewer.estimatorcolumns, viewer.modelpath)
    )


def reload_run(viewer: "EstimatorViewer") -> None:
    """Read the run again, e.g. while ARTIS writes more timesteps, and keep the time range inside the valid timesteps.

    A plot against time of the whole run then covers the new timesteps too.
    """
    oldvalidtimesteps = viewer.validtimesteps
    # each kept scan and each list of timesteps of a run folder comes from the files of the last load
    scan_parquet_file.cache_clear()
    get_runfolder_timesteps_cached.cache_clear()
    load_run(viewer)
    values = viewer.values
    wholerun = (values.first, values.last) == (oldvalidtimesteps[0], oldvalidtimesteps[-1])
    if is_evolution(values) and wholerun:
        viewer.values = viewer.select_timesteps(values, 0, len(viewer.validtimesteps))
    else:
        firstpos, lastpos = viewer.get_selection_positions()
        viewer.values = viewer.select_timesteps(values, firstpos, lastpos - firstpos + 1)


def get_batch_caches(modelpath: Path) -> "list[EstimatorBatchCache]":
    """Return the current parquet caches of the batches of the run, and convert the stale batches first.

    A large run has more than 10 GB of estimators. The conversion of 40 batches of a 3D kilonova run kept 5.2 GB of
    freed memory in the viewer process. A child process gives that memory back to the system when it ends. The
    child process imports artistools again, which took 0.4 s, thus a run with no stale batch converts in this process.
    """
    states = get_estimator_batch_states(modelpath, None, None)
    convert = partial(convert_estimator_batch_caches, verbose=False)
    if any(state.rebuild for state in states):
        return call_in_child_process(convert, modelpath, states)
    return convert(modelpath, states)


def get_plot_frames(fig: mplfig.Figure) -> "list[mplax.Axes]":
    """Return the frames of the plot, which are the visible axes that are not a colour bar."""
    return [axis for axis in fig.axes if axis.get_visible() and axis.get_label() != COLORBAR_LABEL]


class EstimatorViewer:
    """The plot of the viewer and the values of its controls.

    The window reads and changes the values, and a test can do the same without a display. Each call to draw
    makes the command from the values, then parses it and draws it with the code of plotestimators. Thus the plot
    always agrees with the command.
    """

    # load_run reads these from the run, and reload_run reads them again
    batchcaches: "list[EstimatorBatchCache] | None"
    estimatorcolumns: list[str]
    validtimesteps: list[int]
    cells: list[int]
    cellvelocities: dict[int, float]
    defaultsubplots: tuple[tuple[str, ...], ...]

    def __init__(self, tokens: "Sequence[str]", fig: mplfig.Figure) -> None:
        """Read the arguments of the user, and take the first values of the controls from them."""
        parser = make_parser(addargs)
        usertokens = remove_options(parser, tokens, {"interactive"})
        # parse_cli_args also puts "--" in front of the ARTIS folder at the end. Then -plot does not take the folder
        basetokens = remove_options(parser, separate_trailing_folders(usertokens), CONTROLLED_DESTS | OUTPUT_DESTS)
        # the tokens that no option takes are the variables of the first subplot and the folder, which args holds
        otheroptions, _ = split_option_rows(parser, basetokens)
        args = parse_cli_args(addargs, None, None, usertokens)
        check_viewer_args(args)
        resolve_positional_args(args)
        self.parser = parser
        self.modelpath = Path(args.modelpath)
        require_artis_folder(self.modelpath)
        # the working folder needs no token in the command
        self.modeltoken = "" if self.modelpath == Path() else str(args.modelpath)
        self.helptexts = get_helptexts(parser)
        # the arguments of the user, which give the columns of the plot and the format of the run
        self.userargs = args

        self.tmids = get_timestep_times(self.modelpath, loc="mid")
        self.tstarts = get_timestep_times(self.modelpath, loc="start")
        self.tends = get_timestep_times(self.modelpath, loc="end")
        load_run(self)
        givensubplots = tuple(tuple(str(item) for item in plotitems) for plotitems in args.plotlist or ())
        # plotestimators stops when no default subplot applies to the model, thus the window then shows Te
        subplots = givensubplots or self.defaultsubplots or (("Te",),)

        # the default -x of a command depends on its time and its other options, thus each set has one result
        self.defaultxvariables: dict[tuple[bool, OptionRows], str] = {}
        timegiven = time_is_given(args)
        isimage = args.slice is not None or args.dimensionreduce == 2
        # plotestimators plots all the cells against time when the command gives no time. On a large model that plot
        # is slow, thus the window starts with a snapshot unless the command selects a cell
        xvariable: str = args.x or (
            "time"
            if not timegiven and args.modelgridindex is not None and not isimage
            else self.get_default_xvariable(otheroptions, timegiven=True)
        )
        if timegiven:
            first, last, _, _ = get_time_range(self.modelpath, args.timestep, args.timemin, args.timemax, args.timedays)
            if first < 0:
                exit_with_error(
                    f"the time of the command lies outside the run, which goes from {self.tstarts[0]:.1f} to"
                    f" {self.tends[-1]:.1f} days",
                    "Give a time inside that range",
                )
        elif xvariable in TIME_XVARIABLES:
            first, last = self.validtimesteps[0], self.validtimesteps[-1]
        else:
            first = last = self.validtimesteps[len(self.validtimesteps) // 2]

        self.values = ControlValues(
            first=first,
            last=last,
            x=xvariable,
            xmin="" if args.xmin is None else format(args.xmin, ".10g"),
            xmax="" if args.xmax is None else format(args.xmax, ".10g"),
            cells="" if args.modelgridindex is None else str(args.modelgridindex),
            subplots=subplots,
            markers=bool(args.markers),
            xbins="" if args.xbins is None else str(args.xbins),
            colorbyion=bool(args.colorbyion),
            figwidthscale=args.figwidthscale,
            otheroptions=otheroptions,
        )

        self.fig = fig
        # a window can change the size of the figure, thus the size of the plot stays here
        self.figsize: tuple[float, float] = (0.0, 0.0)
        self.isimage = False
        # the axis of a model faster than 0.3c shows v/c for -x velocity, and -xmin then takes km/s
        self.xlimitscale = 1.0
        # the bins, the markers, and the colours that the last plot used, which the window shows beside the controls
        self.plotxbins: int | None = None
        self.plotmarkers = False
        self.plotcolorbyion = False
        # the last warning of the last plot, which the status bar shows
        self.warning = ""

    def get_default_xvariable(self, otheroptions: OptionRows, *, timegiven: bool) -> str:
        """Return the x variable that plotestimators takes for a command with no -x, the time, and the options."""
        key = (timegiven, otheroptions)
        if key not in self.defaultxvariables:
            timetokens = ["-timestep", str(self.validtimesteps[0])] if timegiven else []
            self.defaultxvariables[key] = get_default_xvariable([*timetokens, *get_option_row_tokens(otheroptions)])
        return self.defaultxvariables[key]

    def get_time_tokens(self, values: ControlValues) -> list[str]:
        """Return the -timestep option of the values, or no option for a plot against time of the whole run."""
        if is_evolution(values) and (values.first, values.last) == (self.validtimesteps[0], self.validtimesteps[-1]):
            return []
        return ["-timestep", str(values.first) if values.first == values.last else f"{values.first}-{values.last}"]

    def get_plot_tokens(self, values: ControlValues | None = None) -> list[str]:
        """Return the plotestimators arguments of the values, or of the current values if the caller gives none.

        The first subplot comes before the folder, e.g. "Te TR mymodel", and each other subplot follows -plot at the
        end. A -plot takes each word up to the next flag, thus nothing can follow the last -plot.
        """
        if values is None:
            values = self.values
        subplots = () if values.subplots == self.defaultsubplots else values.subplots
        tokens = [*(subplots[0] if subplots else ()), *([self.modeltoken] if self.modeltoken else [])]
        timetokens = self.get_time_tokens(values)
        tokens += timetokens
        if values.x != self.get_default_xvariable(values.otheroptions, timegiven=bool(timetokens)):
            tokens += ["-x", values.x]
        if values.cells:
            tokens += get_option_tokens("-cell", values.cells)
        if values.xmin:
            tokens += get_option_tokens("-xmin", values.xmin)
        if values.xmax:
            tokens += get_option_tokens("-xmax", values.xmax)
        if values.xbins:
            tokens += get_option_tokens("-xbins", values.xbins)
        if values.markers:
            tokens.append("--markers")
        if values.colorbyion:
            tokens.append("--colorbyion")
        if values.figwidthscale != 1.0:
            tokens += ["-figwidthscale", format(values.figwidthscale, "g")]
        tokens += get_option_row_tokens(values.otheroptions)
        for subplot in subplots[1:]:
            tokens += ["-plot", *subplot]
        return tokens

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotestimators", *self.get_plot_tokens()])

    def get_timesteps_text(self) -> str:
        """Return the timesteps and the days that the plot reads."""
        first, last = self.values.first, self.values.last
        timesteps = f"timestep {first}" if first == last else f"timesteps {first} to {last}"
        return f"The plot reads {timesteps}, from {self.tstarts[first]:.4g} to {self.tends[last]:.4g} d"

    def select_centre(self, days: float) -> ControlValues:
        """Return the values with a time range of the same width that has its centre nearest to the time.

        get_time_text gives the same centre.
        """
        firstpos, lastpos = self.get_selection_positions()
        count = lastpos - firstpos + 1
        validtmids = [self.tmids[timestep] for timestep in self.validtimesteps]
        return self.select_timesteps(self.values, get_nearest_range_start(validtmids, days, count), count)

    def get_cell_text(self) -> str:
        """Return the cells of the plot, with the radial velocity of a single cell."""
        if not self.values.cells:
            return "The plot reads all the cells"
        cell = get_single_cell(self.values.cells)
        if cell is not None and (velocity := self.cellvelocities.get(cell)) is not None:
            return f"Cell {self.values.cells} at v_r = {velocity / C_cm_per_s:.3g}c ({velocity / km_to_cm:.4g} km/s)"
        return f"The plot reads the cells {self.values.cells}"

    def select_timesteps(self, values: ControlValues, firstpos: int, count: int) -> ControlValues:
        """Return the values with count valid timesteps from the position firstpos in the valid timesteps."""
        count = min(max(count, 1), len(self.validtimesteps))
        firstpos = min(max(firstpos, 0), len(self.validtimesteps) - count)
        return dc.replace(values, first=self.validtimesteps[firstpos], last=self.validtimesteps[firstpos + count - 1])

    def get_selection_positions(self, values: ControlValues | None = None) -> tuple[int, int]:
        """Return the positions in the valid timesteps of the first and the last timestep of the time range."""
        values = values or self.values

        def get_position(timestep: int) -> int:
            return min(
                range(len(self.validtimesteps)), key=lambda position: abs(self.validtimesteps[position] - timestep)
            )

        return get_position(values.first), get_position(values.last)

    def step_time(self, step: int) -> ControlValues | None:
        """Return the values with the time range one timestep later or earlier, or None at the end of the run."""
        firstpos, lastpos = self.get_selection_positions()
        if firstpos + step < 0 or lastpos + step >= len(self.validtimesteps):
            return None
        return self.select_timesteps(self.values, firstpos + step, lastpos - firstpos + 1)

    def step_width(self, step: int) -> ControlValues:
        """Return the values with the time range one timestep wider or narrower."""
        firstpos, lastpos = self.get_selection_positions()
        return self.select_timesteps(self.values, firstpos, lastpos - firstpos + 1 + step)

    def move_to_end(self, *, last: bool) -> ControlValues:
        """Return the values with the time range at the start or at the end of the run, and the same width."""
        firstpos, lastpos = self.get_selection_positions()
        count = lastpos - firstpos + 1
        return self.select_timesteps(self.values, len(self.validtimesteps) - count if last else 0, count)

    def step_cell(self, step: int) -> ControlValues | None:
        """Return the values with the next or the previous cell of the model, or None after the last cell.

        A plot of all the cells, or of a list of cells, moves to the first or the last cell.
        """
        if not self.cells:
            return None
        cell = get_single_cell(self.values.cells)
        if cell is not None and cell in self.cells:
            position = self.cells.index(cell) + step
            if not 0 <= position < len(self.cells):
                return None
        else:
            position = 0 if step > 0 else len(self.cells) - 1
        return dc.replace(self.values, cells=str(self.cells[position]))

    def set_xvariable(self, values: ControlValues, xvariable: str) -> ControlValues:
        """Return the values with a new -x variable.

        A plot against time reads the whole run, and a snapshot reads the valid timestep at the middle of the old time
        range. The x limits of one variable do not apply to a different variable. A snapshot reads all the cells,
        because the window hides the control of the cells for a snapshot.
        """
        if xvariable == values.x:
            return values
        values = dc.replace(values, x=xvariable, xmin="", xmax="")
        if is_evolution(values) and not is_evolution(self.values):
            return self.select_timesteps(values, 0, len(self.validtimesteps))
        if not is_evolution(values) and is_evolution(self.values):
            firstpos, lastpos = self.get_selection_positions(values)
            return self.select_timesteps(dc.replace(values, cells=""), (firstpos + lastpos) // 2, 1)
        return values

    def draw(self, *, quiet: bool = True) -> str | None:
        """Draw the plot of the values, and return the reason for the status line if plotestimators rejects it.

        The terminal shows the whole error, and the status line shows its first line.
        """
        return self.render(self.values, quiet=quiet)()

    def render(self, values: ControlValues, *, quiet: bool = True) -> "Callable[[], str | None]":
        """Draw the plot of the values on a new figure, and return the function that shows it in the canvas.

        The function returns the reason for the status line if plotestimators rejects the values, and the old plot
        then stays. A worker thread can run this method, because it changes nothing that the window reads. The
        function that it returns must run in the thread of the window.
        """
        plots: list[RenderedPlot] = []
        # the canvas of the window draws at this resolution. A draw at the same resolution in the worker thread
        # makes the ticks and the text of the figure, and the draw of the window then took 0.05 s in place of 0.22 s
        dpi = float(self.fig.dpi)

        def make_plot() -> None:
            plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens(values))
            check_viewer_args(plotargs)
            givenx = plotargs.x
            fig = mplfig.Figure()
            FigureCanvasAgg(fig)
            draw_plot(plotargs, fig, self.batchcaches)
            isimage = plotargs.dimensionreduce == 2
            # the constrained layout of a colour image keeps the title inside the figure
            if not isimage:
                make_room_for_title(fig)
            xlimitscale = C_cm_per_s / km_to_cm if plotargs.x == "beta" and givenx != "beta" else 1.0
            fig.set_dpi(dpi)
            fig.canvas.draw()
            plots.append(
                RenderedPlot(
                    fig=fig,
                    isimage=isimage,
                    xlimitscale=xlimitscale,
                    xbins=plotargs.xbins,
                    markers=bool(plotargs.markers),
                    colorbyion=bool(plotargs.colorbyion),
                )
            )

        outcome = run_command_step_outcome(make_plot, quiet=quiet)

        def show_plot() -> str | None:
            self.warning = get_last_warning(outcome.errors)
            if outcome.message is not None:
                return outcome.message
            plot = plots[0]
            fig, self.isimage, self.xlimitscale = plot.fig, plot.isimage, plot.xlimitscale
            self.plotxbins, self.plotmarkers, self.plotcolorbyion = plot.xbins, plot.markers, plot.colorbyion
            canvas = self.fig.canvas
            fig.set_canvas(canvas)
            canvas.figure = fig
            self.fig = fig
            figwidth, figheight = fig.get_size_inches()
            self.figsize = (float(figwidth), float(figheight))
            canvas.draw_idle()
            return None

        return show_plot

    def change(self, values: ControlValues) -> str | None:
        """Draw the plot of the new values, and keep the old values and the old plot if plotestimators rejects them."""
        message = self.render(values)()
        if message is None:
            self.values = values
        return message

    def get_fitted_figwidthscale(self, areawidth: float, areaheight: float) -> float:
        """Return the -figwidthscale that gives the figure the shape of the plot area.

        A colour image takes the constrained layout, thus the width of all the figure follows -figwidthscale.
        """
        marginwidth = 0.0 if self.isimage else LABELWIDTH_INCHES + RIGHTMARGIN_INCHES
        return get_fitted_figwidthscale(self.figsize, self.values.figwidthscale, marginwidth, areawidth, areaheight)

    def get_xlimit_text(self, xdata: float) -> str:
        """Return the -xmin or -xmax text of a position on the x axis, with 3 significant digits for a short command."""
        return get_short_number(xdata * self.xlimitscale)


def get_item_directive(item: str) -> str | None:
    """Return the name of the directive that an item of a subplot gives, e.g. "ymin" for "ymin=1e-16", or None."""
    name, equals, _ = item.partition("=")
    return name.removeprefix("_") if equals and name.removeprefix("_") in DIRECTIVES else None


def replace_directives(subplot: "Sequence[str]", directives: "Mapping[str, str | None]") -> tuple[str, ...]:
    """Return the items of a subplot with these directives in place of their old values. None removes a directive."""
    kept = [item for item in subplot if get_item_directive(item) not in directives]
    return (*kept, *(f"{name}={value}" for name, value in directives.items() if value is not None))


def get_short_number(value: float) -> str:
    """Return a number with 3 significant digits for a short command, e.g. 12300 or 1.23e-05."""
    return format(float(f"{value:.3g}"), ".10g")


def get_nearest_cell(viewer: EstimatorViewer, xdata: float) -> int | None:
    """Return the cell with the radial velocity nearest to a position on a velocity axis, or None for another axis.

    A 3D model has many cells at one radial velocity, and the function gives one of them.
    """
    if viewer.values.x not in {"velocity", "beta"} or not viewer.cellvelocities:
        return None
    axisisbeta = viewer.values.x == "beta" or viewer.xlimitscale != 1.0
    velocity = xdata * (C_cm_per_s if axisisbeta else km_to_cm)
    cells = np.fromiter(viewer.cellvelocities.keys(), dtype=np.int64, count=len(viewer.cellvelocities))
    velocities = np.fromiter(viewer.cellvelocities.values(), dtype=np.float64, count=len(viewer.cellvelocities))
    return int(cells[np.argmin(np.abs(velocities - velocity))])


def get_evolution_values(viewer: EstimatorViewer, cells: str) -> ControlValues:
    """Return the values that plot the cells against time over the whole run."""
    return viewer.set_xvariable(dc.replace(viewer.values, cells=cells), "time")


def get_snapshot_values(viewer: EstimatorViewer, xdata: float) -> ControlValues | None:
    """Return the values of a snapshot at a position on a time axis, or None for another axis."""
    if viewer.values.x == "time":
        validtmids = [viewer.tmids[timestep] for timestep in viewer.validtimesteps]
        position = get_nearest_range_start(validtmids, xdata, 1)
    elif viewer.values.x == "timestep":
        position = min(range(len(viewer.validtimesteps)), key=lambda pos: abs(viewer.validtimesteps[pos] - xdata))
    else:
        return None
    return viewer.select_timesteps(viewer.set_xvariable(viewer.values, "velocity"), position, 1)


def get_xunit_text(xlimitscale: float, xvariable: str) -> str:
    """Return the unit of -xmin and -xmax, e.g. " [km/s]", or an empty text for a variable with no unit.

    A scale of the x limits shows that the axis gives v/c and the options take km/s.
    """
    return plain_label(get_units_string("velocity" if xlimitscale != 1.0 else xvariable))


def replace_option_rows(viewer: EstimatorViewer, values: ControlValues, otheroptions: OptionRows) -> ControlValues:
    """Return the values with new rows of the option table.

    An option such as -slice changes the default x of a snapshot. An x that equals the old default follows the new
    default, because the command then gives no -x, as plotestimators does.
    """
    newvalues = dc.replace(values, otheroptions=otheroptions)
    if is_evolution(values) or values.x != viewer.get_default_xvariable(values.otheroptions, timegiven=True):
        return newvalues
    return viewer.set_xvariable(newvalues, viewer.get_default_xvariable(otheroptions, timegiven=True))


def get_readout(axis: "mplax.Axes", x: float) -> str:
    """Return the value of each series of a subplot at x.

    plotestimators gives no label to the line of a subplot of one variable. The label of the y axis then names it.
    """
    return "   ".join([f"x = {x:.5g}", *get_line_readouts(axis, x)])


def get_image_value(cursordata: t.Any) -> float | None:
    """Return the value of a colour image under the pointer, or None for an empty cell.

    QuadMesh.get_cursor_data gives a masked array of one element. NumPy 2.5 converts only an array with no dimensions
    to a float.
    """
    if cursordata is None:
        return None
    values = np.ma.masked_invalid(np.ma.asarray(cursordata, dtype=float)).compressed()
    return float(values[0]) if values.size else None


def get_icon_curve() -> "npt.NDArray[np.float64]":
    """Return the curve of the icon of the viewer, which is a temperature that decreases as the velocity increases."""
    xvalues = np.linspace(0.0, 1.0, 200)
    return 0.2 + 0.6 * (1.0 - np.exp(-4.0 * xvalues)) / (1.0 - math.exp(-4.0))


def get_keyboard_help() -> str:
    """Return the table of the keys and the mouse actions of the window, with the shortcuts of the platform."""
    shortcuts = get_menu_shortcut_texts()
    return f"""<table>
<tr><td><b>Left</b>, <b>Right</b></td><td>Move the time range to the adjacent timestep</td></tr>
<tr><td><b>Up</b>, <b>Down</b></td><td>Make the time range one timestep wider or narrower</td></tr>
<tr><td><b>Home</b>, <b>End</b></td><td>Move the time range to the start or the end of the run</td></tr>
<tr><td><b>Page Up</b>, <b>Page Down</b></td><td>Select the previous or the next cell</td></tr>
<tr><td><b>Space</b></td><td>Play or pause. A snapshot moves through the timesteps, and a plot against time moves
through the cells</td></tr>
<tr><td><b>Drag</b> across a plot</td><td>Select the x range</td></tr>
<tr><td><b>Shift-drag</b> up or down a subplot</td><td>Select the y range of the subplot (ymin= and ymax=)</td></tr>
<tr><td><b>Double-click</b> a subplot</td><td>Show the x range and the y range of the data</td></tr>
<tr><td><b>Right-click</b> a subplot</td><td>Show a menu: the y scale, the y range, a cell against time, or a
snapshot at a time</td></tr>
<tr><td><b>{shortcuts["Save Figure..."]}</b></td><td>Run the command to save the figure</td></tr>
<tr><td><b>{shortcuts["Copy Command"]}</b></td><td>Copy the command</td></tr>
<tr><td><b>{shortcuts["Open Model..."]}</b></td><td>Open a model in a new window</td></tr>
<tr><td><b>{shortcuts["Reload Data"]}</b></td><td>Read the run again, e.g. while ARTIS writes more timesteps</td></tr>
<tr><td><b>?</b></td><td>Show this list</td></tr>
</table>"""


def run_viewer(tokens: "Sequence[str]") -> None:
    """Open the window of the viewer, and print the command of the last plot when the window closes."""
    app = start_application(APPLICATION_NAME, get_icon_curve())
    # the list holds a reference to each window, thus Python keeps the window while it is open
    windows: list[QtWidgets.QMainWindow] = []
    open_window(tokens, windows)
    app.exec()


def open_window(tokens: "Sequence[str]", windows: "list[QtWidgets.QMainWindow]") -> str | None:
    """Open a window of the viewer for the plotestimators arguments in tokens, or return the reason for no window."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    viewer = EstimatorViewer(tokens, mplfig.Figure())
    window = make_window(APPLICATION_NAME)
    window.setWindowTitle(f"{APPLICATION_NAME} {viewer.modelpath.resolve().name}")
    canvas = FigureCanvasQTAgg(viewer.fig)
    if (message := viewer.draw(quiet=False)) is not None:
        # the arguments of the user give the error, and the terminal shows it
        if not windows:
            raise SystemExit(1)
        return message
    windows.append(window)

    fittimer = make_timer(window, FIT_MILLISECONDS)
    playtimer = make_timer(window, PLAY_MILLISECONDS)

    def on_resize() -> None:
        fit_canvas(canvas, viewer.figsize, plotarea)
        # a new plot can take a few seconds, thus the plot takes the new shape only when the resize stops
        fittimer.start()

    plotarea = make_plot_area(canvas, on_resize)
    sidebar, panellayout = make_sidebar()
    splitter = make_central_splitter(window, plotarea, sidebar)
    helptexts = viewer.helptexts

    _, timegrid = add_section(panellayout, "Time")
    timeslider, widthslider = make_slider(), make_slider()
    timeslider.setToolTip(
        "The middle of the time range. The Left key and the Right key move it to the adjacent timestep."
    )
    widthslider.setToolTip("The number of timesteps of the time range. The Up key and the Down key change it.")
    timeedit = QtWidgets.QLineEdit()
    timeedit.setFixedWidth(110)
    timeedit.setToolTip("A time in days. The time range moves to the timestep that holds it.")
    widthlabel = QtWidgets.QLabel()
    timestepslabel = QtWidgets.QLabel()
    playbutton = QtWidgets.QPushButton("Play")
    playbutton.setCheckable(True)
    playbutton.setToolTip(
        "Move a snapshot through the timesteps of the run, or move a plot against time through the cells (Space)"
    )
    # a plot against time takes a range of timesteps, as the x range of plotspectra. A snapshot takes a time and a width
    trangebox = QtWidgets.QWidget()
    trangelayout = QtWidgets.QHBoxLayout(trangebox)
    trangelayout.setContentsMargins(0, 0, 0, 0)
    tminedit, tmaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    trangeslider, set_trange_positions, connect_trange, set_trange_steps = make_range_slider(1)
    trangeslider.setToolTip("The first and the last timestep of the plot against time.")
    for edit, text in ((tminedit, "first"), (tmaxedit, "last")):
        edit.setFixedWidth(80)
        edit.setToolTip(
            f"The middle time of the {text} timestep of the range in days. A new time moves to the nearest."
        )
    for widget in (tminedit, trangeslider, tmaxedit):
        trangelayout.addWidget(widget)
    timegrid.addWidget(QtWidgets.QLabel("Time [d]"), 0, 0)
    timegrid.addWidget(timeslider, 0, 1)
    timegrid.addWidget(timeedit, 0, 2)
    timegrid.addWidget(trangebox, 0, 1, 1, 2)
    timegrid.addWidget(widthlabel, 1, 0)
    timegrid.addWidget(widthslider, 1, 1, 1, 2)
    timegrid.addWidget(timestepslabel, 2, 0, 1, 2)
    timegrid.addWidget(playbutton, 2, 2)

    _, cellgrid = add_section(panellayout, "Cells")
    cellslider = make_slider()
    cellslider.setToolTip("Select one cell. The Page Up key and the Page Down key select the adjacent cell.")
    celledit = QtWidgets.QLineEdit()
    celledit.setFixedWidth(110)
    celledit.setPlaceholderText("all cells")
    celledit.setToolTip(helptexts.get("modelgridindex", ""))
    allcellsbutton = QtWidgets.QPushButton("All cells")
    allcellsbutton.setToolTip("Remove -cell, thus the plot reads all the cells")
    celllabel = QtWidgets.QLabel()
    cellgrid.addWidget(QtWidgets.QLabel("-cell"), 0, 0)
    cellgrid.addWidget(cellslider, 0, 1)
    cellgrid.addWidget(celledit, 0, 2)
    cellgrid.addWidget(celllabel, 1, 0, 1, 2)
    cellgrid.addWidget(allcellsbutton, 1, 2)

    _, xgrid = add_section(panellayout, "Horizontal axis")
    xbox = QtWidgets.QComboBox()
    xbox.setEditable(True)
    xbox.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
    xbox.addItems([*XVARIABLES, *(column for column in viewer.estimatorcolumns if column not in XVARIABLES)])
    if (completer := xbox.completer()) is not None:
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
    xbox.setToolTip(helptexts.get("x", ""))
    xminedit, xmaxedit, xbinsedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    xminlabel, xmaxlabel = QtWidgets.QLabel("-xmin"), QtWidgets.QLabel("-xmax")
    zoomtip = " Drag across a plot to select a range. Double-click a plot to show the range of the data."
    for edit in (xminedit, xmaxedit):
        edit.setFixedWidth(100)
        edit.setPlaceholderText("auto")
    # a field takes each value of -xbins, e.g. a negative value for the automatic bins
    xbinsedit.setFixedWidth(80)
    xbinsedit.setPlaceholderText("default")
    # an empty field removes -xbins. QIntValidator gives the Intermediate state for an empty text, and the field
    # then sends no editingFinished signal
    xbinsedit.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"(-?\d+)?"), xbinsedit))
    xbinsedit.setToolTip(helptexts.get("xbins", ""))
    markerscheck = QtWidgets.QCheckBox("--markers")
    markerscheck.setToolTip(helptexts.get("markers", ""))
    colorbyioncheck = QtWidgets.QCheckBox("--colorbyion")
    colorbyioncheck.setToolTip(helptexts.get("colorbyion", ""))
    add_row(xgrid, 0, [QtWidgets.QLabel("-x"), xbox, QtWidgets.QLabel("-xbins"), xbinsedit])
    add_row(xgrid, 1, [xminlabel, xminedit, xmaxlabel, xmaxedit])
    add_row(xgrid, 2, [markerscheck, colorbyioncheck])

    _, subplotgrid = add_section(panellayout, "Subplots")
    subplotlist = QtWidgets.QListWidget()
    subplotlist.setFixedHeight(6 * subplotlist.fontMetrics().lineSpacing() + 12)
    subplotlist.setToolTip(
        "Each row gives the items of one subplot:\n"
        "- an estimator variable;\n"
        "- an ion;\n"
        "- a type of series and its names, e.g. populations 'Fe II' 'Fe III';\n"
        f"- a directive: {', '.join(f'{name}=' for name in DIRECTIVES)}.\n"
        "Double-click a row to change it. Right-click a subplot for the y scale, and Shift-drag it for the y range."
    )
    variablebox = QtWidgets.QComboBox()
    variablebox.setEditable(True)
    variablebox.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
    variablebox.addItems(["", *SERIESTYPES, *viewer.estimatorcolumns])
    if (completer := variablebox.completer()) is not None:
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
    if (lineedit := variablebox.lineEdit()) is not None:
        lineedit.setPlaceholderText("Add a variable to the selected subplot")
    variablebox.setToolTip(
        "Type part of a name to search, or type an ion or a directive."
        " The name goes at the end of the selected subplot."
    )
    addbutton, removebutton = QtWidgets.QPushButton("Add subplot"), QtWidgets.QPushButton("Remove")
    upbutton, downbutton = QtWidgets.QPushButton("Up"), QtWidgets.QPushButton("Down")
    defaultbutton = QtWidgets.QPushButton("Default")
    defaultbutton.setToolTip("Show the default subplots of plotestimators, which the command gives with no -plot")
    subplotgrid.addWidget(subplotlist, 0, 0, 1, -1)
    subplotgrid.addWidget(variablebox, 1, 0, 1, -1)
    add_row(subplotgrid, 2, [addbutton, removebutton, upbutton, downbutton, defaultbutton])

    _, optiongrid = add_section(panellayout, "Other options")

    def on_option_rows(rows: OptionRows) -> None:
        queue.apply(replace_option_rows(viewer, viewer.values, rows))

    optiontable, set_option_rows = make_option_table(
        window,
        viewer.parser,
        CONTROLLED_DESTS | OUTPUT_DESTS | TABLE_EXCLUDED_DESTS,
        viewer.values.otheroptions,
        on_option_rows,
    )
    optiongrid.addWidget(optiontable, 0, 0, 1, 2)
    commandtext, copybutton = add_command_section(panellayout)
    statusbar = make_status_bar(window)

    signalwidgets: list[QtWidgets.QWidget] = [
        timeslider,
        widthslider,
        cellslider,
        xbox,
        markerscheck,
        colorbyioncheck,
        subplotlist,
        variablebox,
    ]

    def set_ranges() -> None:
        """Give the sliders the number of valid timesteps and the number of cells of the run."""
        nvalid = len(viewer.validtimesteps)
        timeslider.setRange(0, nvalid - 1)
        widthslider.setRange(1, nvalid)
        set_trange_steps(max(nvalid - 1, 1))
        cellslider.setRange(0, max(len(viewer.cells) - 1, 0))

    set_ranges()

    def show_subplots(subplots: "Sequence[Sequence[str]]") -> None:
        """Show a row for each subplot, and keep the selected row."""
        texts = [shlex.join(subplot) for subplot in subplots]
        if [subplotlist.item(row).text() for row in range(subplotlist.count())] == texts:
            return
        selectedrow = subplotlist.currentRow()
        subplotlist.clear()
        for text in texts:
            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            subplotlist.addItem(item)
        subplotlist.setCurrentRow(min(selectedrow, subplotlist.count() - 1))

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        blockers = [QtCore.QSignalBlocker(widget) for widget in signalwidgets]
        try:
            show_blocked_values()
        finally:
            for blocker in blockers:
                blocker.unblock()
        fit_canvas(canvas, viewer.figsize, plotarea)

    def show_blocked_values() -> None:
        values = viewer.values
        firstpos, lastpos = viewer.get_selection_positions()
        evolution = is_evolution(values)
        for widget in (timeslider, timeedit, widthlabel, widthslider):
            widget.setVisible(not evolution)
        trangebox.setVisible(evolution)
        set_trange_positions(firstpos, lastpos)
        set_edit_text(tminedit, f"{viewer.tmids[values.first]:.4g}")
        set_edit_text(tmaxedit, f"{viewer.tmids[values.last]:.4g}")
        # the slider selects one cell, which suits a plot against time. A snapshot of one cell has one point, and
        # the field of a snapshot takes a list or a range of cells
        cellslider.setVisible(evolution)
        timeslider.setValue((firstpos + lastpos) // 2)
        widthslider.setValue(lastpos - firstpos + 1)
        widthlabel.setText(f"Timesteps: {lastpos - firstpos + 1}")
        set_edit_text(timeedit, get_time_text(viewer.tmids, values))
        timestepslabel.setText(viewer.get_timesteps_text())
        set_edit_text(celledit, values.cells)
        if (cell := get_single_cell(values.cells)) is not None and cell in viewer.cells:
            cellslider.setValue(viewer.cells.index(cell))
        celllabel.setText(viewer.get_cell_text())
        allcellsbutton.setEnabled(bool(values.cells))
        xbox.setCurrentText(values.x)
        xunit = get_xunit_text(viewer.xlimitscale, values.x)
        xminlabel.setText(f"-xmin{xunit}")
        xmaxlabel.setText(f"-xmax{xunit}")
        # the axis of a fast model shows v/c, and the option keeps km/s
        unittip = " The axis shows v/c, and the option takes km/s." if viewer.xlimitscale != 1.0 else ""
        for edit, dest in ((xminedit, "xmin"), (xmaxedit, "xmax")):
            edit.setToolTip(helptexts.get(dest, "") + zoomtip + unittip)
        set_edit_text(xminedit, values.xmin)
        set_edit_text(xmaxedit, values.xmax)
        set_edit_text(xbinsedit, values.xbins)
        # the window hides the output of the plot, thus the controls show the bins, the markers, and the colours
        # that the plot chose when the command gives no option
        if viewer.isimage:
            xbinsedit.setPlaceholderText("default")
        else:
            xbinsedit.setPlaceholderText("auto: no bins" if viewer.plotxbins is None else f"auto: {viewer.plotxbins}")
        markerscheck.setChecked(values.markers)
        markerscheck.setText(
            "--markers (on for -xbins 0)" if viewer.plotmarkers and not values.markers else "--markers"
        )
        colorbyioncheck.setChecked(values.colorbyion)
        colorbyioncheck.setText(
            "--colorbyion (on for automatic bins)"
            if viewer.plotcolorbyion and not values.colorbyion
            else "--colorbyion"
        )
        show_subplots(values.subplots)
        defaultbutton.setEnabled(values.subplots != viewer.defaultsubplots and bool(viewer.defaultsubplots))
        set_option_rows(values.otheroptions)
        set_command_text(commandtext, viewer.get_command())

    def after_draw(message: str | None) -> None:
        # matplotlib keeps the connections of the mouse in the figure, and each plot has a new figure
        connect_mouse_to_figure()
        # a new number of subplots changes the height of the figure, thus the plot can need a new -figwidthscale
        fittimer.start()
        # a rejection occurs again at each step, thus a rejection stops the Play button
        if message is not None:
            playbutton.setChecked(False)
        elif playbutton.isChecked():
            # a draw that the Play button did not start also restarts the timer, thus one chain of steps stays
            start_play_timer(playtimer, queue.plotseconds)

    queue = DrawQueue(window, viewer, statusbar, show_values, after_draw, render=viewer.render)
    apply = queue.apply

    def fit_figwidthscale() -> None:
        """Give the plot the -figwidthscale that fills the plot area."""
        figwidthscale = get_new_figwidthscale(
            plotarea, viewer.figsize, viewer.values.figwidthscale, viewer.get_fitted_figwidthscale
        )
        if figwidthscale is not None:
            apply(dc.replace(viewer.values, figwidthscale=figwidthscale))

    fittimer.timeout.connect(fit_figwidthscale)

    def show_error(message: str) -> None:
        statusbar.message.setText(message)
        show_values()

    def on_time(position: int) -> None:
        firstpos, lastpos = viewer.get_selection_positions()
        count = lastpos - firstpos + 1
        apply(viewer.select_timesteps(viewer.values, position - (count - 1) // 2, count))

    def on_width(count: int) -> None:
        firstpos, _ = viewer.get_selection_positions()
        apply(viewer.select_timesteps(viewer.values, firstpos, count))

    def on_timeedit() -> None:
        # a later plot can show new text in the field only when the field has no edit of the user
        timeedit.setModified(False)
        try:
            days = float(timeedit.text())
        except ValueError:
            show_error("Give a number of days for the time")
            return
        apply(viewer.select_centre(days))

    def on_trange(handle: int, position: int) -> None:
        firstpos, lastpos = viewer.get_selection_positions()
        if handle == 0:
            firstpos = position
        else:
            lastpos = position
        apply(viewer.select_timesteps(viewer.values, firstpos, lastpos - firstpos + 1))

    def on_trangeedit() -> None:
        tminedit.setModified(False)
        tmaxedit.setModified(False)
        try:
            firstdays, lastdays = float(tminedit.text()), float(tmaxedit.text())
        except ValueError:
            show_error("Give a number of days for the first and the last time")
            return
        validtmids = [viewer.tmids[timestep] for timestep in viewer.validtimesteps]
        firstpos, lastpos = (get_nearest_range_start(validtmids, days, 1) for days in (firstdays, lastdays))
        if firstpos > lastpos:
            show_error("Give a first time that is before the last time")
            return
        apply(viewer.select_timesteps(viewer.values, firstpos, lastpos - firstpos + 1))

    def on_step_time(step: int) -> None:
        if (values := viewer.step_time(step)) is not None:
            apply(values)

    def on_step_cell(step: int) -> None:
        if (values := viewer.step_cell(step)) is not None:
            apply(values)

    def play_step() -> None:
        if not playbutton.isChecked():
            return
        values = viewer.step_cell(1) if is_evolution(viewer.values) else viewer.step_time(1)
        if values is None:
            playbutton.setChecked(False)
            return
        apply(values)

    def on_play(checked: bool) -> None:
        playbutton.setText("Pause" if checked else "Play")
        if checked:
            play_step()

    def on_cell(position: int) -> None:
        if viewer.cells:
            apply(dc.replace(viewer.values, cells=str(viewer.cells[position])))

    def on_celledit() -> None:
        celledit.setModified(False)
        # -cell takes one word, thus a space between two cells becomes the comma of a list
        apply(dc.replace(viewer.values, cells=",".join(celledit.text().replace(",", " ").split())))

    def on_xvariable() -> None:
        if xvariable := xbox.currentText().strip():
            apply(viewer.set_xvariable(viewer.values, xvariable))

    def on_xedit() -> None:
        texts = []
        for edit in (xminedit, xmaxedit):
            edit.setModified(False)
            text = edit.text().strip()
            try:
                texts.append(format(float(text), ".10g") if text else "")
            except ValueError:
                show_error("Give a number for -xmin and -xmax, or leave a field empty for the range of the data")
                return
        if texts[0] and texts[1] and float(texts[0]) >= float(texts[1]):
            show_error("Give a -xmin that is less than -xmax")
            return
        apply(dc.replace(viewer.values, xmin=texts[0], xmax=texts[1]))

    def on_style() -> None:
        xbinsedit.setModified(False)
        apply(
            dc.replace(
                viewer.values,
                xbins=xbinsedit.text().strip(),
                markers=markerscheck.isChecked(),
                colorbyion=colorbyioncheck.isChecked(),
            )
        )

    def apply_subplots(subplots: "Sequence[tuple[str, ...]]") -> None:
        """Apply the subplots, or show the default subplots if the list is empty."""
        newsubplots = tuple(subplot for subplot in subplots if subplot) or viewer.defaultsubplots
        if not newsubplots:
            show_error("Give the items of at least one subplot")
            return
        if newsubplots == viewer.values.subplots:
            show_values()
            return
        apply(dc.replace(viewer.values, subplots=newsubplots))

    def on_subplot_edited() -> None:
        try:
            subplots = [tuple(shlex.split(subplotlist.item(row).text())) for row in range(subplotlist.count())]
        except ValueError:
            show_error("A subplot has a quote with no end")
            return
        apply_subplots(subplots)

    def on_add_variable() -> None:
        name = variablebox.currentText().strip()
        if not name:
            return
        with QtCore.QSignalBlocker(variablebox):
            variablebox.setCurrentText("")
        subplots = list(viewer.values.subplots)
        row = subplotlist.currentRow()
        if 0 <= row < len(subplots):
            subplots[row] = (*subplots[row], name)
            apply_subplots(subplots)
        else:
            subplots.append((name,))
            apply_subplots(subplots)
            # the new subplot takes the next names, thus it becomes the selected row
            subplotlist.setCurrentRow(subplotlist.count() - 1)

    def on_add_subplot() -> None:
        item = QtWidgets.QListWidgetItem("")
        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        with QtCore.QSignalBlocker(subplotlist):
            subplotlist.addItem(item)
            subplotlist.setCurrentItem(item)
        # the empty row takes no place in the command, thus the command changes when the user gives its items
        subplotlist.editItem(item)

    def on_remove_subplot() -> None:
        row = subplotlist.currentRow()
        subplots = list(viewer.values.subplots)
        if 0 <= row < len(subplots):
            del subplots[row]
            apply_subplots(subplots)

    def on_move_subplot(step: int) -> None:
        row = subplotlist.currentRow()
        subplots = list(viewer.values.subplots)
        if 0 <= row < len(subplots) and 0 <= row + step < len(subplots):
            subplots[row], subplots[row + step] = subplots[row + step], subplots[row]
            with QtCore.QSignalBlocker(subplotlist):
                subplotlist.setCurrentRow(row + step)
            apply_subplots(subplots)

    def on_copy() -> None:
        copy_command(viewer.get_command())
        statusbar.message.setText("Copied the command")

    def on_save() -> None:
        from artistools.estimators.plotestimators import main as plotestimators_main

        message = save_figure_of_command(window, plotestimators_main, "plotestimators", viewer.get_plot_tokens())
        if message is not None:
            statusbar.message.setText(message)

    def on_open_model() -> None:
        if (message := open_model_window(window, open_window, windows)) is not None:
            show_error(message)

    def on_reload() -> None:
        """Read the run again. A new batch of text files converts here, and the terminal shows its progress."""
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            message = run_command_step(partial(reload_run, viewer), quiet=False)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if message is not None:
            show_error(f"The viewer cannot reload the run: {message}")
            return
        set_ranges()
        queue.redraw()

    def on_help() -> None:
        QtWidgets.QMessageBox.information(window, "Keys and mouse actions", get_keyboard_help())

    def get_frame_readout(event: t.Any, frame: "mplax.Axes") -> str:
        if not viewer.isimage:
            return get_readout(frame, event.xdata)
        meshes = [collection for collection in frame.collections if collection.get_array() is not None]
        value = get_image_value(meshes[0].get_cursor_data(event)) if meshes else None
        valuetext = f"   value: {value:.4g}" if value is not None else ""
        return f"x = {event.xdata:.4g}c   y = {event.ydata:.4g}c{valuetext}"

    def on_select(low: float, high: float) -> None:
        xmin, xmax = viewer.get_xlimit_text(low), viewer.get_xlimit_text(high)
        if float(xmin) < float(xmax):
            apply(dc.replace(viewer.values, xmin=xmin, xmax=xmax))

    def get_subplot_row(frameindex: int) -> int | None:
        """Return the row of the subplot of a frame of the plot, or None for a colour image."""
        return frameindex if not viewer.isimage and frameindex < len(viewer.values.subplots) else None

    def set_directives(row: int, directives: "Mapping[str, str | None]", **changes: t.Any) -> None:
        subplots = list(viewer.values.subplots)
        subplots[row] = replace_directives(subplots[row], directives)
        apply(dc.replace(viewer.values, subplots=tuple(subplots), **changes))

    def on_select_y(frameindex: int, low: float, high: float) -> None:
        row = get_subplot_row(frameindex)
        ymin, ymax = get_short_number(low), get_short_number(high)
        if row is not None and float(ymin) < float(ymax):
            set_directives(row, {"ymin": ymin, "ymax": ymax})

    def on_reset(frameindex: int) -> None:
        row = get_subplot_row(frameindex)
        if row is None:
            apply(dc.replace(viewer.values, xmin="", xmax=""))
        else:
            set_directives(row, {"ymin": None, "ymax": None}, xmin="", xmax="")

    def on_menu(frameindex: int, event: t.Any) -> None:
        """Show the menu of a subplot: the y scale, the y range, and the plot of a cell or of a snapshot."""
        menu = QtWidgets.QMenu(window)
        row = get_subplot_row(frameindex)
        if row is not None:
            islog = get_plot_frames(viewer.fig)[frameindex].get_yscale() == "log"
            scaleaction = menu.addAction("Linear scale" if islog else "Log scale")
            scaleaction.triggered.connect(lambda: set_directives(row, {"yscale": "linear" if islog else "log"}))
            resetaction = menu.addAction("Show the y range of the data")
            resetaction.setEnabled(
                any(get_item_directive(item) in {"ymin", "ymax"} for item in viewer.values.subplots[row])
            )
            resetaction.triggered.connect(lambda: set_directives(row, {"ymin": None, "ymax": None}))
            menu.addSeparator()
        if is_evolution(viewer.values):
            snapshot = get_snapshot_values(viewer, event.xdata)
            if snapshot is not None:
                snapshotaction = menu.addAction(f"Plot a snapshot at {viewer.tmids[snapshot.first]:.4g} d")
                snapshotaction.triggered.connect(lambda: apply(snapshot))
        else:
            cell = get_nearest_cell(viewer, event.xdata)
            if cell is not None:
                cellaction = menu.addAction(f"Plot cell {cell} against time")
                cellaction.triggered.connect(lambda: apply(get_evolution_values(viewer, str(cell))))
            if viewer.values.cells:
                cellsaction = menu.addAction(f"Plot the cells {viewer.values.cells} against time")
                cellsaction.triggered.connect(lambda: apply(get_evolution_values(viewer, viewer.values.cells)))
        if menu.actions():
            menu.exec(QtGui.QCursor.pos())

    def on_closed() -> None:
        print(viewer.get_command())
        # each kept scan holds the metadata of its file, e.g. 7.6 MB for 3000 columns
        scan_parquet_file.cache_clear()
        # the list holds a reference to each open window, thus Python does not delete the window. A closed window
        # leaves the list
        windows.remove(window)

    add_menus(
        window,
        {
            "Open Model...": on_open_model,
            "Reload Data": on_reload,
            "Save Figure...": on_save,
            "Copy Command": on_copy,
            "Close Window": window.close,
            "Keys and Mouse Actions": on_help,
        },
    )

    timeslider.valueChanged.connect(on_time)
    widthslider.valueChanged.connect(on_width)
    timeedit.editingFinished.connect(on_timeedit)
    connect_trange(on_trange)
    tminedit.editingFinished.connect(on_trangeedit)
    tmaxedit.editingFinished.connect(on_trangeedit)
    playbutton.toggled.connect(on_play)
    playtimer.timeout.connect(play_step)
    cellslider.valueChanged.connect(on_cell)
    celledit.editingFinished.connect(on_celledit)
    allcellsbutton.clicked.connect(lambda: apply(dc.replace(viewer.values, cells="")))
    xbox.activated.connect(on_xvariable)
    if (xlineedit := xbox.lineEdit()) is not None:
        xlineedit.editingFinished.connect(on_xvariable)
    xminedit.editingFinished.connect(on_xedit)
    xmaxedit.editingFinished.connect(on_xedit)
    xbinsedit.editingFinished.connect(on_style)
    markerscheck.toggled.connect(on_style)
    colorbyioncheck.toggled.connect(on_style)
    subplotlist.itemChanged.connect(on_subplot_edited)
    variablebox.activated.connect(on_add_variable)
    # activated gives only a name of the list, and Return in the field also gives a typed ion or a directive
    if lineedit is not None:
        lineedit.returnPressed.connect(on_add_variable)
    addbutton.clicked.connect(on_add_subplot)
    removebutton.clicked.connect(on_remove_subplot)
    upbutton.clicked.connect(lambda: on_move_subplot(-1))
    downbutton.clicked.connect(lambda: on_move_subplot(1))
    defaultbutton.clicked.connect(lambda: apply_subplots(viewer.defaultsubplots))
    copybutton.clicked.connect(on_copy)
    statusbar.helpbutton.clicked.connect(on_help)
    window.destroyed.connect(on_closed)
    connect_mouse_to_figure = connect_plot_mouse(
        canvas,
        get_frames=lambda: get_plot_frames(viewer.fig),
        get_readout=get_frame_readout,
        readoutlabel=statusbar.readout,
        on_select=on_select,
        on_reset=on_reset,
        # a colour image has a velocity on each axis, and -xmin and -xmax take the velocity of the line plot
        can_select=lambda: not viewer.isimage,
        on_select_y=on_select_y,
        on_menu=on_menu,
    )
    # a text field takes these keys while it has the focus, and the shortcuts apply otherwise
    for key, callback in (
        (QtCore.Qt.Key.Key_Left, lambda: on_step_time(-1)),
        (QtCore.Qt.Key.Key_Right, lambda: on_step_time(1)),
        (QtCore.Qt.Key.Key_Up, lambda: apply(viewer.step_width(1))),
        (QtCore.Qt.Key.Key_Down, lambda: apply(viewer.step_width(-1))),
        (QtCore.Qt.Key.Key_Home, lambda: apply(viewer.move_to_end(last=False))),
        (QtCore.Qt.Key.Key_End, lambda: apply(viewer.move_to_end(last=True))),
        (QtCore.Qt.Key.Key_PageUp, lambda: on_step_cell(-1)),
        (QtCore.Qt.Key.Key_PageDown, lambda: on_step_cell(1)),
        (QtCore.Qt.Key.Key_Space, playbutton.toggle),
    ):
        QtGui.QShortcut(QtGui.QKeySequence(key), window).activated.connect(callback)

    show_window(
        window, splitter, viewer.figsize, APPLICATION_NAME, lambda: fit_canvas(canvas, viewer.figsize, plotarea)
    )
    show_values()
    return None
