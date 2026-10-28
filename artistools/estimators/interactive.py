"""Show the plot of plotestimators in a window with controls for the time, the cell, and the subplots."""

import argparse
import contextlib
import dataclasses as dc
import math
import shlex
import typing as t
from functools import partial
from pathlib import Path
from types import MappingProxyType

import matplotlib as mpl
import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import numpy as np
import polars as pl

from artistools.constants import C_cm_per_s
from artistools.constants import km_to_cm
from artistools.estimators.core import convert_estimator_batch_caches
from artistools.estimators.core import get_cell_selection_description
from artistools.estimators.core import get_estimator_batch_states
from artistools.estimators.core import get_estimator_timesteps
from artistools.estimators.core import get_units_string
from artistools.estimators.core import join_cell_modeldata
from artistools.estimators.core import scan_estimators
from artistools.estimators.core import scan_parquet_file
from artistools.estimators.plotestimators import add_plot_columns
from artistools.estimators.plotestimators import addargs
from artistools.estimators.plotestimators import draw_plot
from artistools.estimators.plotestimators import get_data_key
from artistools.estimators.plotestimators import get_default_colours
from artistools.estimators.plotestimators import get_default_x
from artistools.estimators.plotestimators import get_gid_series_name
from artistools.estimators.plotestimators import get_model_default_plotlist
from artistools.estimators.plotestimators import get_styled_series
from artistools.estimators.plotestimators import get_subplot_grid
from artistools.estimators.plotestimators import LineFigureData
from artistools.estimators.plotestimators import main as plotestimators_main
from artistools.estimators.plotestimators import require_artis_folder
from artistools.estimators.plotestimators import resolve_positional_args
from artistools.estimators.plotestimators import resolve_snapshot_arguments
from artistools.estimators.plotestimators import time_is_given
from artistools.estimators.plotestimators import TIME_XVARIABLES
from artistools.estimators.plotitems import get_item_directive
from artistools.estimators.plotitems import get_plotitem_tokens
from artistools.estimators.plotitems import get_python_plotitems
from artistools.estimators.plotitems import get_series_name
from artistools.estimators.plotitems import get_styles_of_subplots
from artistools.estimators.plotitems import resolve_item_alias
from artistools.estimators.subplotcards import add_subplot_section
from artistools.estimators.subplotcards import get_chip_items
from artistools.estimators.subplotcards import get_command_items
from artistools.estimators.subplotcards import get_first_common_variable
from artistools.estimators.subplotcards import get_linewidthscale
from artistools.estimators.subplotcards import move_poptype_to_subplots
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modelmeta
from artistools.inputmodel import scan_modeldata
from artistools.misc import call_in_child_process
from artistools.misc import exit_with_error
from artistools.misc import format_range_list
from artistools.misc import get_first_sentence
from artistools.misc import get_runfolder_timesteps
from artistools.misc import get_runfolders
from artistools.misc import get_time_range
from artistools.misc import get_time_range_text
from artistools.misc import get_timestep_times
from artistools.misc import is_remote_path
from artistools.misc import on_model_host
from artistools.misc import parse_cli_args
from artistools.misc import path_is_codecomparison
from artistools.misc import resolve_modelpath
from artistools.nltepops import has_nlte_populations
from artistools.plottools import DARKMODE_COLOURS
from artistools.plottools import get_legend_entries_in_draw_order
from artistools.plottools import get_light_grey
from artistools.plottools import LABELWIDTH_INCHES
from artistools.plottools import plain_label
from artistools.plottools import RIGHTMARGIN_INCHES
from artistools.viewertools.application import open_model_folder
from artistools.viewertools.application import set_drop_handler
from artistools.viewertools.core import exit_for_other_actions
from artistools.viewertools.core import get_actions_by_flag
from artistools.viewertools.core import get_centred_range
from artistools.viewertools.core import get_clamped_range
from artistools.viewertools.core import get_end_range
from artistools.viewertools.core import get_figure_option_tokens
from artistools.viewertools.core import get_fitted_figwidthscale
from artistools.viewertools.core import get_given_dpi
from artistools.viewertools.core import get_nearest_range_start
from artistools.viewertools.core import get_option_row_tokens
from artistools.viewertools.core import get_option_tokens
from artistools.viewertools.core import get_resized_range
from artistools.viewertools.core import get_row_values
from artistools.viewertools.core import get_short_limits
from artistools.viewertools.core import get_stepped_range
from artistools.viewertools.core import keep_figwidthscale
from artistools.viewertools.core import make_parser
from artistools.viewertools.core import OptionRows
from artistools.viewertools.core import parse_viewer_tokens
from artistools.viewertools.core import run_command_step
from artistools.viewertools.core import set_row_values
from artistools.viewertools.menus import add_default_options
from artistools.viewertools.menus import add_window_actions
from artistools.viewertools.menus import export_animation
from artistools.viewertools.menus import ViewerCommand
from artistools.viewertools.sections import add_time_controls
from artistools.viewertools.sections import add_y_axis_actions
from artistools.viewertools.sections import connect_play
from artistools.viewertools.sections import connect_time_keys
from artistools.viewertools.sections import get_next_play_time
from artistools.viewertools.sections import make_figscale_box
from artistools.viewertools.sections import read_limit_fields
from artistools.viewertools.sections import read_selected_range
from artistools.viewertools.sections import TIME_KEY_HELP_ROWS
from artistools.viewertools.sections import WAIT_FOR_PLOT_MESSAGE
from artistools.viewertools.series import get_dash_pattern
from artistools.viewertools.widgets import add_row
from artistools.viewertools.widgets import add_section
from artistools.viewertools.widgets import blocked_signals
from artistools.viewertools.widgets import fit_canvas
from artistools.viewertools.widgets import get_changed_arguments
from artistools.viewertools.widgets import get_python_call
from artistools.viewertools.widgets import make_note_label
from artistools.viewertools.widgets import make_range_slider
from artistools.viewertools.widgets import make_row_layout
from artistools.viewertools.widgets import make_slider
from artistools.viewertools.widgets import parse_command_tokens
from artistools.viewertools.widgets import set_edit_text
from artistools.viewertools.widgets import set_note_text
from artistools.viewertools.widgets import set_search_completion
from artistools.viewertools.widgets import set_spin_value
from artistools.viewertools.window import add_command_sections
from artistools.viewertools.window import connect_plot_mouse
from artistools.viewertools.window import DrawQueue
from artistools.viewertools.window import finish_viewer_window
from artistools.viewertools.window import FLAG_LABELS
from artistools.viewertools.window import get_line_readouts
from artistools.viewertools.window import make_readout_tag
from artistools.viewertools.window import reload_models
from artistools.viewertools.window import render_command
from artistools.viewertools.window import show_command_sections
from artistools.viewertools.window import start_viewer_window

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.axes as mplax
    import numpy.typing as npt
    from matplotlib.backend_bases import MouseEvent
    from PySide6 import QtWidgets

    from artistools.estimators.core import EstimatorBatchCache
    from artistools.estimators.plotestimators import FiguresData

# the controls of the window give these arguments, thus the command drops the values that the user typed
CONTROLLED_DESTS: t.Final = frozenset({
    # the Resolution box of the Figure section gives -dpi
    "dpi",
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
    "histogram",
    "colorbyion",
    "figwidthscale",
})

# these options change only the output file. Save Figure in the File menu gives the file, thus the command drops them
OUTPUT_DESTS: t.Final = frozenset({"outputfile", "format"})

# the window reads the model in the format of --classicartis when it opens, thus a change later has no effect. The
# option table neither shows nor offers this row, and the row stays in the command
MODELFORMAT_DESTS: t.Final = frozenset({"classicartis"})

# the ways to select the cells of the plot, by the key of the selector of the window
GEOMETRY_MODES: t.Final = MappingProxyType({
    "all": "The cells inside the sphere of radius v_max",
    "cells": "Selected cells (--cell)",
    "alongaxis": "A half line of cells from the centre along an axis (--readonlymgi alongaxis)",
    "cone": "The cells in a cone around a half-axis (--readonlymgi cone)",
    "plane": "A 2D plane slice of cells as an image (--slice)",
    "line": "A full line of cells through the grid along an axis (--slice)",
    "average": "The mean over rings around the z axis as an image (--dimensionreduce 2)",
    "projection": "The mean along an axis of each line of cells as an image (--projection)",
})

# the modes whose plot is a colour image of a snapshot
IMAGE_MODES: t.Final = frozenset({"plane", "average", "projection"})

# the rows of the option table that a mode of the geometry sets
GEOMETRY_FLAGS: t.Final = ("--slice", "--dimensionreduce", "--projection", "--readonlymgi", "--axis", "--coneangle")

# the planes of -slice through the origin, by the axis that is normal to the plane
PLANE_OF_NORMAL: t.Final = MappingProxyType({"z": "xy", "y": "xz", "x": "yz"})

# the ways to smooth each line, by the key of the selector of the window
SMOOTHING_MODES: t.Final = MappingProxyType({
    "none": "No smoothing",
    "movingavg": "Moving average (--filtermovingavg)",
    "savgol": "Savitzky-Golay filter (--filtersavgol)",
})


# the option table does not offer these options, but it shows their rows from the command. Some give a
# different action from one plot, and --verbose and --quiet change only the hidden output
TABLE_EXCLUDED_DESTS: t.Final = frozenset({
    "help",
    "multiplot",
    "makegif",
    "listvariables",
    "listnuclides",
    "quiet",
    "verbose",
})

# the -x choices that are not an estimator column
XVARIABLES: t.Final = ("velocity", "beta", "time", "timestep", "modelgridindex")

APPLICATION_NAME: t.Final = "artistools plotestimators"

# the items that the Dock icon takes: a model folder opens in a new window
DOCUMENT_TYPES: t.Final = ("public.folder",)

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
    histogram: bool
    xbins: str
    colorbyion: bool
    figwidthscale: float
    # the resolution of a PNG file (-dpi), or None for the default of the command
    dpi: int | None
    otheroptions: OptionRows


class RenderedPlot(t.NamedTuple):
    """The properties of a plot that the worker thread drew, which the window shows.

    plotestimators chooses the bins, the markers, and the colours when the command gives no option. xbins, markers, and
    colorbyion hold the values that the plot used.
    """

    isimage: bool
    xlimitscale: float
    xbins: int | None
    markers: bool
    colorbyion: bool
    swatches: "Mapping[tuple[int, str], SeriesSwatch]"
    # the rows of the subplots whose ions take the colours of their elements with --colorbyion
    elementcolourrows: frozenset[int]
    # the plot used the data of the last plot, thus it printed none of the warnings of the data
    reuseddata: bool


class SeriesSwatch(t.NamedTuple):
    """The line of one series in the plot, which a chip of its subplot shows as a short image.

    The popup of the line properties shows the default colour and the default width, which the series has with no
    style of its own.
    """

    colour: str
    alpha: float
    linewidth: float
    dashpattern: tuple[float, ...] | None
    defaultcolour: str
    defaultlinewidth: float


def get_series_swatches(
    figuredata: LineFigureData, args: argparse.Namespace, darkforeground: str | None
) -> dict[tuple[int, str], SeriesSwatch]:
    """Return the line of each series by the index of its subplot and its name, as the plot draws it.

    darkforeground is the foreground colour of Dark Mode, which takes the place of a dark grey, see get_light_grey.
    """

    def get_shown_rgba(colour: t.Any) -> tuple[float, float, float, float]:
        rgba = mplcolors.to_rgba(colour)
        if darkforeground is not None and (lightgrey := get_light_grey(rgba, darkforeground)) is not None:
            return lightgrey
        return rgba

    swatches: dict[tuple[int, str], SeriesSwatch] = {}
    stylesofsubplots = get_styles_of_subplots(args, len(figuredata.subplots))
    for index, ((series, _), styles) in enumerate(zip(figuredata.subplots, stylesofsubplots, strict=True)):
        for seriesdata, defaultcolour in zip(series, get_default_colours(series), strict=True):
            _, plotkwargs = get_styled_series(seriesdata, styles.get(seriesdata.seriesname, {}), args.linewidthscale)
            rgba = get_shown_rgba(plotkwargs.get("color", defaultcolour))
            dashes = plotkwargs.get("dashes") or get_dash_pattern(plotkwargs.get("linestyle"), None)
            swatches[index, seriesdata.seriesname] = SeriesSwatch(
                colour=mplcolors.to_hex(rgba),
                # an alpha of the line replaces the alpha of its colour, as matplotlib does
                alpha=float(plotkwargs.get("alpha", rgba[3])),
                linewidth=float(plotkwargs.get("linewidth", mpl.rcParams["lines.linewidth"])),
                dashpattern=tuple(dashes) if dashes else None,
                defaultcolour=mplcolors.to_hex(get_shown_rgba(defaultcolour)),
                defaultlinewidth=float(seriesdata.plotkwargs.get("linewidth", mpl.rcParams["lines.linewidth"])),
            )
    return swatches


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


def get_default_xvariable(tokens: "Sequence[str]") -> str:
    """Return the x variable that plotestimators takes for command tokens with no -x.

    --slice and --dimensionreduce set the x variable of a snapshot. Otherwise a command with no time plots against
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
    """Return the text of the time field: the centre of the middle times of the range, with two decimal places."""
    return f"{(tmids[values.first] + tmids[values.last]) / 2.0:.2f}"


def get_single_cell(cells: str) -> int | None:
    """Return the cell of a --cell text that names one cell, or None for a list, a range, or no cell.

    str.isdigit accepts a superscript digit such as "²", which int rejects, thus the test reads ASCII digits alone.
    """
    return int(cells) if cells.isascii() and cells.isdecimal() else None


class ModelData(t.NamedTuple):
    """The data of the model that the controls of the viewer need, which Reload Data reads again."""

    batchcaches: "list[EstimatorBatchCache] | None"
    estimatorcolumns: tuple[str, ...]
    validtimesteps: list[int]
    cells: list[int]
    cellvelocities: dict[int, float]
    defaultsubplots: tuple[tuple[str, ...], ...]
    skippeddefaults: tuple[str, ...]


@on_model_host
def read_remote_model_data(modelpath: Path, args: argparse.Namespace, ntimesteps: int) -> ModelData:
    """Read the data of a remote model on its host, which also converts the stale estimator caches there.

    The batch caches hold the paths on the host. The window gives them back to the host with each plot, thus the host
    checks no text file for a plot.
    """
    return read_model_data(modelpath, args, ntimesteps)


def read_model_data(modelpath: Path, args: argparse.Namespace, ntimesteps: int) -> ModelData:
    """Read the data of the model that the controls of the viewer need.

    The viewer checks and converts the estimator caches of the model one time. Each plot then reads the caches of its
    own timesteps and cells, as the command does. The function changes no viewer, thus a worker thread can run it.
    """
    if is_remote_path(modelpath):
        return read_remote_model_data(modelpath, args, ntimesteps)

    isartismodel = not args.classicartis and not path_is_codecomparison(modelpath)
    batchcaches = get_batch_caches(modelpath) if isartismodel else None
    estimators, modelmeta = join_cell_modeldata(
        estimators=scan_estimators(modelpath=modelpath, classicartis=args.classicartis, batchcaches=batchcaches),
        modelpath=modelpath,
    )
    _, estimatorcolumns = add_plot_columns(args, estimators, modelmeta)
    # a run that stopped early has no estimators for the last timesteps, and a plot of those timesteps fails
    validtimesteps = get_estimator_timesteps(modelpath, estimators) or list(range(ntimesteps))
    # ARTIS writes the estimators of each cell that holds matter
    lzmodel, modelmeta = scan_modeldata(modelpath)
    dfcells = (
        add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") > 0.0)
        .select("modelgridindex", "vel_r_mid")
        .sort("modelgridindex")
        .collect()
    )
    cells: list[int] = dfcells["modelgridindex"].to_list()
    defaultplotlist, skippedplotlist = get_model_default_plotlist(estimatorcolumns, modelpath)
    return ModelData(
        batchcaches=batchcaches,
        estimatorcolumns=tuple(estimatorcolumns),
        validtimesteps=validtimesteps,
        cells=cells,
        cellvelocities=dict(zip(cells, dfcells["vel_r_mid"].to_list(), strict=True)),
        # the subplots of plotestimators for a command with no -plot, which the window shows first
        defaultsubplots=tuple(get_plotitem_tokens(plotitems) for plotitems in defaultplotlist),
        # the window hides the output of the command, which names each default subplot that the model cannot show
        skippeddefaults=tuple(
            f"{' '.join(get_plotitem_tokens(plotitems))}: {reason}" for plotitems, reason in skippedplotlist
        ),
    )


def reload_model_data(viewer: "EstimatorViewer", modeldata: ModelData) -> None:
    """Give the viewer the data of the model that it read again, and keep the time range inside the valid timesteps.

    A plot against time over all the timesteps then covers the new timesteps too.
    """
    oldvalidtimesteps = viewer.modeldata.validtimesteps
    viewer.modeldata = modeldata
    # the model can hold new data, thus the next plot reads it
    viewer.lastdata = None
    values = viewer.values
    alltimesteps = (values.first, values.last) == (oldvalidtimesteps[0], oldvalidtimesteps[-1])
    if is_evolution(values) and alltimesteps:
        viewer.values = viewer.select_positions(values, (0, len(viewer.modeldata.validtimesteps) - 1))
    else:
        viewer.values = viewer.select_positions(values, viewer.get_selection_positions())


def get_geometry_choices(dimensions: int) -> list[str]:
    """Return the keys in GEOMETRY_MODES that a model can plot.

    A 1D or a 2D model has no axes, planes, lines, or projections, but it has the average around the z axis.
    """
    return [mode for mode in GEOMETRY_MODES if dimensions == 3 or mode in {"all", "cells", "average"}]


def get_geometry_mode(values: "ControlValues") -> str:
    """Return the key in GEOMETRY_MODES of the selection of the cells of the values."""
    rows = values.otheroptions
    if slicevalues := get_row_values(rows, "--slice"):
        return "line" if "," in slicevalues[0] else "plane"
    if get_row_values(rows, "--projection"):
        return "projection"
    if get_row_values(rows, "--dimensionreduce") == ("2",):
        return "average"
    if readonlymgi := get_row_values(rows, "--readonlymgi"):
        return readonlymgi[0]
    return "cells" if values.cells else "all"


def get_slice_parts(slicetext: str) -> tuple[str, str]:
    """Return the plane and the offset along its normal of a --slice plane, e.g. ("xy", "-0.2c") for "z=-0.2c".

    parse_slice_argument of plotestimators reads the same forms, e.g. "zx" for the plane xz and "z = -0.2c".
    """
    text = slicetext.strip().lower()
    if len(text) == 2 and text[0] != text[1] and set(text) <= set("xyz"):
        return "".join(sorted(text)), ""
    normal, equals, offset = (part.strip() for part in text.partition("="))
    if equals and normal in PLANE_OF_NORMAL:
        return PLANE_OF_NORMAL[normal], "" if offset in {"0", "0c", "0.0"} else offset
    return "xy", ""


def get_slice_text(plane: str, offset: str) -> str:
    """Return the --slice text of a plane with an offset along its normal. An empty offset gives the plane."""
    normal = next(axis for axis, planeaxes in PLANE_OF_NORMAL.items() if planeaxes == plane)
    return f"{normal}={offset.strip()}" if offset.strip() else plane


def get_line_axis(slicetext: str) -> str:
    """Return the axis along a --slice line, which is the axis that the line gives no condition for."""
    conditionaxes = {condition.partition("=")[0].strip() for condition in slicetext.lower().split(",")}
    return next((axis for axis in "xyz" if axis not in conditionaxes), "x")


def get_slice_conditions(slicetext: str) -> dict[str, str]:
    """Return the position on each axis of a --slice condition, e.g. {"z": "0.1c", "y": "0"} for "z=0.1c,y=0"."""
    conditions: dict[str, str] = {}
    for condition in slicetext.lower().split(","):
        name, equals, position = (part.strip() for part in condition.partition("="))
        if equals and name in {"x", "y", "z"}:
            conditions[name] = position
    return conditions


def get_line_label(axis: str, slicetext: str) -> str:
    """Return the text of a line along an axis in the line box, e.g. "x (y=z=0)".

    The line of slicetext shows its own positions, e.g. "x (y=0, z=0.1c)". A line along a different axis goes through
    the origin.
    """
    others = [other for other in "xyz" if other != axis]
    isthisline = "," in slicetext and get_line_axis(slicetext) == axis
    conditions = get_slice_conditions(slicetext if isthisline else "")
    first, second = (conditions.get(other) or "0" for other in others)
    positions = f"{others[0]}={others[1]}={first}" if first == second else f"{others[0]}={first}, {others[1]}={second}"
    return f"{axis} ({positions})"


def get_geometry_description(tokens: "Sequence[str]", modelmeta: "Mapping[str, t.Any]") -> str:
    """Return the text that plotestimators logs for the cells of the plot of the command tokens.

    The tokens that plotestimators rejects give an empty text, and the status line gives the reason.
    """
    descriptions: list[str] = []

    def describe() -> None:
        args = parse_cli_args(addargs, None, None, tokens)
        conditions = resolve_snapshot_arguments(args)
        descriptions.append(get_cell_selection_description(args, conditions, modelmeta))

    run_command_step(describe, echo=False)
    return descriptions[0] if descriptions else ""


def set_geometry_mode(viewer: "EstimatorViewer", values: "ControlValues", mode: str) -> "ControlValues":
    """Return the values with a new selection of the cells.

    A mode keeps the parameters that apply to it, e.g. the axis of the cells along an axis for a cone. A colour
    image is a snapshot with a velocity on each axis. Thus a plane or the average around the z axis takes the default x
    of a snapshot, and a plot against time becomes a snapshot.
    """
    rows = values.otheroptions
    keptaxis = get_row_values(rows, "--axis") if mode in {"alongaxis", "cone"} else None
    keptcone = get_row_values(rows, "--coneangle") if mode == "cone" else None
    oldslice = (get_row_values(rows, "--slice") or ("",))[0]
    oldprojection = get_row_values(rows, "--projection") or ("z",)
    changes: dict[str, tuple[str, ...] | None] = dict.fromkeys(GEOMETRY_FLAGS)
    changes |= {"--axis": keptaxis, "--coneangle": keptcone}
    if mode in {"alongaxis", "cone"}:
        changes["--readonlymgi"] = (mode,)
    elif mode == "plane":
        changes["--slice"] = (oldslice if oldslice and "," not in oldslice else "xy",)
    elif mode == "line":
        changes["--slice"] = (oldslice if "," in oldslice else "z=0,y=0",)
    elif mode == "average":
        changes["--dimensionreduce"] = ("2",)
    elif mode == "projection":
        changes["--projection"] = oldprojection
    cells = (
        (values.cells or (str(viewer.modeldata.cells[0]) if viewer.modeldata.cells else "")) if mode == "cells" else ""
    )
    newrows = set_row_values(rows, changes)
    newvalues = replace_option_rows(viewer, dc.replace(values, cells=cells), newrows)
    if mode in IMAGE_MODES:
        return viewer.set_xvariable(newvalues, viewer.get_default_xvariable(newrows, timegiven=True))
    return newvalues


def get_subplots_per_row(rows: OptionRows) -> int:
    """Return the number of subplots or image panels in each row, which --subplotsperrow gives, or 1."""
    values = get_row_values(rows, "--subplotsperrow")
    return max(int(values[0]), 1) if values and values[0].isdecimal() else 1


def get_smoothing(rows: OptionRows) -> tuple[str, tuple[int, ...]]:
    """Return the key in SMOOTHING_MODES of the smoothing of the rows, and its numbers."""
    if (savgol := get_row_values(rows, "--filtersavgol")) and all(value.lstrip("-").isdecimal() for value in savgol):
        return "savgol", tuple(int(value) for value in savgol)
    movingavg = get_row_values(rows, "--filtermovingavg")
    if movingavg and movingavg[0].isdecimal() and int(movingavg[0]) > 1:
        return "movingavg", (int(movingavg[0]),)
    return "none", ()


def set_smoothing(rows: OptionRows, mode: str, numbers: "Sequence[int]") -> OptionRows:
    """Return the rows with a new smoothing, e.g. "savgol" with the window length 5 and the order 2."""
    return set_row_values(
        rows,
        {
            "--filtermovingavg": (str(numbers[0]),) if mode == "movingavg" else None,
            "--filtersavgol": tuple(str(number) for number in numbers[:2]) if mode == "savgol" else None,
        },
    )


def cells_apply(values: ControlValues) -> bool:
    """Return True if --cell selects the cells of the plot of the values.

    --slice and --dimensionreduce 2 select the cells, thus plotestimators rejects --cell with them. --readonlymgi
    replaces --cell with the cells along an axis or in a cone.
    """
    return get_geometry_mode(values) in {"all", "cells"}


def get_batch_caches(modelpath: Path) -> "list[EstimatorBatchCache]":
    """Return the current parquet caches of the batches of the model, and convert the stale batches first.

    A large model has more than 10 GB of estimators. The conversion of 40 batches of a 3D kilonova model kept 5.2 GB of
    freed memory in the viewer process. A child process gives that memory back to the system when it ends. The
    child process imports artistools again, which took 0.4 s, thus the viewer process reads the caches itself when no
    batch is stale.
    """
    # a cache in a cloud folder that is not on the disk can stop the window for minutes with no other output
    runfoldernames = ", ".join(str(runfolder.relative_to(modelpath)) for runfolder in get_runfolders(modelpath))
    print(
        f"The viewer reads the metadata of the estimator caches in the run folders of {modelpath.resolve()}:"
        f" {runfoldernames}"
    )
    states = get_estimator_batch_states(modelpath, None, None)
    convert = partial(convert_estimator_batch_caches, verbose=False)
    if any(state.rebuild for state in states):
        batchcaches = call_in_child_process(convert, modelpath, states)
        # the conversion can drop an incomplete last timestep. The child process clears only its own copy of the
        # timesteps that get_runfolders() read from the text
        get_runfolder_timesteps.cache_clear()
        return batchcaches
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

    # the data of the model that the controls need, which read_model_data reads and Reload Data reads again
    modeldata: ModelData

    def __init__(self, tokens: "Sequence[str]", fig: mplfig.Figure) -> None:
        """Read the arguments of the user, and take the first values of the controls from them."""
        # the paths are the variables of the first subplot and the folder, which args holds
        parser, args, _, otheroptions, self.helptexts = parse_viewer_tokens(
            addargs, tokens, CONTROLLED_DESTS | OUTPUT_DESTS
        )
        check_viewer_args(args)
        resolve_positional_args(args)
        self.parser = parser
        self.modelpath = Path(args.modelpath)
        require_artis_folder(self.modelpath)
        # the working folder needs no token in the command
        self.modeltoken = "" if self.modelpath == Path() else str(args.modelpath)
        # the arguments of the user, which give the columns of the plot and the format of the model
        self.userargs = args

        self.tmids = get_timestep_times(self.modelpath, loc="mid")
        self.tstarts = get_timestep_times(self.modelpath, loc="start")
        self.tends = get_timestep_times(self.modelpath, loc="end")
        self.modeldata = read_model_data(self.modelpath, args, len(self.tmids))
        # the option table does not show the rows of these flags, see MODELFORMAT_DESTS
        self.modelformatflags = frozenset(
            flag for flag, action in get_actions_by_flag(parser).items() if action.dest in MODELFORMAT_DESTS
        )
        # the size of the grid, which gives the edges of the cells that a selection of the window reads
        self.modelmeta: dict[str, t.Any] = get_modelmeta(self.modelpath)
        self.dimensions = int(self.modelmeta["dimensions"])
        # the types of series that the model can plot from its NLTE populations. Only a 1D model gives the width in
        # velocity of each shell for the population for each unit of velocity
        self.nltetypes: tuple[str, ...] = (
            ()
            if path_is_codecomparison(self.modelpath) or not has_nlte_populations(self.modelpath)
            else (
                "averageexcitation",
                "levelpopulation",
                *(("levelpopulation_dn_on_dvel",) if self.dimensions == 1 else ()),
            )
        )
        givensubplots = tuple(
            tuple(resolve_item_alias(str(item)) for item in plotitems) for plotitems in args.plotlist or ()
        )
        # plotestimators stops when no default subplot applies to the model, thus the window shows a common variable
        subplots = (
            givensubplots
            or self.modeldata.defaultsubplots
            or ((get_first_common_variable(self.modeldata.estimatorcolumns),),)
        )

        # the default -x of a command depends on its time and its other options, thus each set has one result
        self.defaultxvariables: dict[tuple[bool, OptionRows], str] = {}
        timegiven = time_is_given(args)
        isimage = args.slice is not None or args.dimensionreduce == 2 or args.projection is not None
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
                    "The time of the command lies outside the time range of the model, which goes from"
                    f" {self.tstarts[0]:.1f} to {self.tends[-1]:.1f} days",
                    "Give a time inside that range",
                )
        elif xvariable in TIME_XVARIABLES:
            first, last = self.modeldata.validtimesteps[0], self.modeldata.validtimesteps[-1]
        else:
            first = last = self.modeldata.validtimesteps[len(self.modeldata.validtimesteps) // 2]

        subplots, otheroptions = move_poptype_to_subplots(subplots, otheroptions, self.modeldata.estimatorcolumns)
        self.values = ControlValues(
            first=first,
            last=last,
            x=xvariable,
            xmin="" if args.xmin is None else format(args.xmin, ".10g"),
            xmax="" if args.xmax is None else format(args.xmax, ".10g"),
            cells="" if args.modelgridindex is None else format_range_list(args.modelgridindex),
            subplots=subplots,
            markers=bool(args.markers),
            histogram=bool(args.histogram),
            xbins="" if args.xbins is None else str(args.xbins),
            colorbyion=bool(args.colorbyion),
            figwidthscale=args.figwidthscale,
            dpi=get_given_dpi(args, parser),
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
        # the colours of the window in Dark Mode, which the window sets and the worker thread reads
        self.darkcolours: tuple[str, str] | None = None
        # the line of each series in the last plot, by the index of its subplot and its name
        self.swatches: Mapping[tuple[int, str], SeriesSwatch] = {}
        # the data of the last plot and its key, see get_data_key. The worker thread sets and reads it, and Reload
        # Data clears it
        self.lastdata: tuple[str, FiguresData] | None = None
        # the last warning of the plot that read the data, which a plot with the same data shows again
        self.datawarning = ""
        # the rows of the subplots whose ions take the colours of their elements, see RenderedPlot
        self.elementcolourrows: frozenset[int] = frozenset()

    def get_default_xvariable(self, otheroptions: OptionRows, *, timegiven: bool) -> str:
        """Return the x variable that plotestimators takes for a command with no -x, the time, and the options."""
        key = (timegiven, otheroptions)
        if key not in self.defaultxvariables:
            timetokens = ["--timestep", str(self.modeldata.validtimesteps[0])] if timegiven else []
            self.defaultxvariables[key] = get_default_xvariable([*timetokens, *get_option_row_tokens(otheroptions)])
        return self.defaultxvariables[key]

    def get_time_tokens(self, values: ControlValues) -> list[str]:
        """Return the --timestep option of the values, or no option for a plot against time over all the timesteps."""
        if is_evolution(values) and (values.first, values.last) == (
            self.modeldata.validtimesteps[0],
            self.modeldata.validtimesteps[-1],
        ):
            return []
        return ["--timestep", str(values.first) if values.first == values.last else f"{values.first}-{values.last}"]

    def get_plot_tokens(self, values: ControlValues | None = None, modeltoken: str | None = None) -> list[str]:
        """Return the plotestimators arguments of the values, or of the current values if the caller gives none.

        The first subplot comes before the folder, e.g. "Te TR mymodel", and each other subplot follows --plot at the
        end. A --plot takes each word up to the next flag, thus nothing can follow the last --plot. The command gives
        each subplot also when the subplots are the default of plotestimators. The command then states the plot in
        full, and a later change of the default does not change it. modeltoken replaces the folder of the command,
        e.g. the full path of the working folder for the next start.
        """
        if values is None:
            values = self.values
        if modeltoken is None:
            modeltoken = self.modeltoken
        subplots = [get_command_items(subplot, self.modeldata.estimatorcolumns) for subplot in values.subplots]
        # a first subplot with no item cannot go before the folder, thus it follows -plot as the others do
        firstpositional = bool(subplots and subplots[0])
        tokens = [*(subplots[0] if firstpositional else ()), *([modeltoken] if modeltoken else [])]
        timetokens = self.get_time_tokens(values)
        tokens += timetokens
        if values.x != self.get_default_xvariable(values.otheroptions, timegiven=bool(timetokens)):
            tokens += ["-x", values.x]
        if values.cells:
            tokens += get_option_tokens("--cell", values.cells)
        if values.xmin:
            tokens += get_option_tokens("--xmin", values.xmin)
        if values.xmax:
            tokens += get_option_tokens("--xmax", values.xmax)
        if values.xbins:
            tokens += get_option_tokens("--xbins", values.xbins)
        if values.markers:
            tokens.append("--markers")
        if values.histogram:
            tokens.append("--histogram")
        if values.colorbyion:
            tokens.append("--colorbyion")
        tokens += get_figure_option_tokens(values)
        tokens += get_option_row_tokens(values.otheroptions)
        for subplot in subplots[1:] if firstpositional else subplots:
            tokens += ["--plot", *subplot]
        return tokens

    def get_command(self) -> str:
        """Return the command that draws the plot of the values."""
        return shlex.join(["artistools", "plotestimators", *self.get_plot_tokens()])

    def get_time_range_text(self) -> str:
        """Return the timesteps and the days that the plot reads."""
        first, last = self.values.first, self.values.last
        return get_time_range_text(first, last, self.tstarts[first], self.tends[last])

    def select_centre(self, days: float) -> ControlValues:
        """Return the values with a time range of the same width that has its centre nearest to the time.

        get_time_text gives the same centre.
        """
        firstpos, lastpos = self.get_selection_positions()
        count = lastpos - firstpos + 1
        validtmids = [self.tmids[timestep] for timestep in self.modeldata.validtimesteps]
        start = get_nearest_range_start(validtmids, days, count)
        return self.select_positions(self.values, (start, start + count - 1))

    def get_cell_text(self) -> str:
        """Return the cells of the plot, with the radial velocity of a single cell."""
        if not self.values.cells:
            return "All cells"
        cell = get_single_cell(self.values.cells)
        if cell is not None and (velocity := self.modeldata.cellvelocities.get(cell)) is not None:
            return f"Cell {self.values.cells} at v_r = {velocity / C_cm_per_s:.3g}c ({velocity / km_to_cm:.4g} km/s)"
        return f"Cells {self.values.cells}"

    def select_positions(self, values: ControlValues, positions: tuple[int, int]) -> ControlValues:
        """Return the values with the time range from the first to the last of two positions in the valid timesteps."""
        validtimesteps = self.modeldata.validtimesteps
        return dc.replace(values, first=validtimesteps[positions[0]], last=validtimesteps[positions[1]])

    def get_selection_positions(self, values: ControlValues | None = None) -> tuple[int, int]:
        """Return the positions in the valid timesteps of the first and the last timestep of the time range."""
        values = values or self.values
        validtimesteps = self.modeldata.validtimesteps
        return (
            get_nearest_range_start(validtimesteps, values.first, 1),
            get_nearest_range_start(validtimesteps, values.last, 1),
        )

    def step_time(self, step: int) -> ControlValues | None:
        """Return the values with the time range step timesteps later, or None past the valid timesteps."""
        positions = get_stepped_range(self.get_selection_positions(), step, len(self.modeldata.validtimesteps))
        return None if positions is None else self.select_positions(self.values, positions)

    def step_width(self, step: int) -> ControlValues:
        """Return the values with step more timesteps in the time range."""
        positions = get_resized_range(self.get_selection_positions(), step, len(self.modeldata.validtimesteps))
        return self.select_positions(self.values, positions)

    def move_to_end(self, *, last: bool) -> ControlValues:
        """Return the values with the time range at the first or the last valid timestep, and the same width."""
        positions = get_end_range(self.get_selection_positions(), len(self.modeldata.validtimesteps), last=last)
        return self.select_positions(self.values, positions)

    def step_cell(self, step: int) -> ControlValues | None:
        """Return the values with the next or the previous cell of the model, or None after the last cell.

        A plot of all the cells, or of a list of cells, moves to the first or the last cell. If --cell does not select
        the cells of the plot, e.g. for a plane, the result is None.
        """
        if not self.modeldata.cells or not cells_apply(self.values):
            return None
        cell = get_single_cell(self.values.cells)
        if cell is not None and cell in self.modeldata.cells:
            position = self.modeldata.cells.index(cell) + step
            if not 0 <= position < len(self.modeldata.cells):
                return None
        else:
            position = 0 if step > 0 else len(self.modeldata.cells) - 1
        return dc.replace(self.values, cells=str(self.modeldata.cells[position]))

    def set_xvariable(self, values: ControlValues, xvariable: str) -> ControlValues:
        """Return the values with a new -x variable.

        A plot against time reads all the timesteps, and a snapshot reads the valid timestep at the middle of the old
        time range. The x limits of one variable do not apply to a different variable. A snapshot reads all the cells,
        because a snapshot of the one cell of a plot against time has one point.
        """
        if xvariable == values.x:
            return values
        values = dc.replace(values, x=xvariable, xmin="", xmax="")
        if is_evolution(values) and not is_evolution(self.values):
            return self.select_positions(values, (0, len(self.modeldata.validtimesteps) - 1))
        if not is_evolution(values) and is_evolution(self.values):
            firstpos, lastpos = self.get_selection_positions(values)
            middle = (firstpos + lastpos) // 2
            return self.select_positions(dc.replace(values, cells=""), (middle, middle))
        return values

    def render(self, values: ControlValues, *, quiet: bool = True) -> "Callable[[], str | None]":
        """Draw the plot of the values on a new figure, and return the function that shows it in the canvas.

        The function returns the reason for the status line if plotestimators rejects the values, and the old plot
        then stays. A worker thread can run this method, because it changes nothing that the window reads. The
        function that it returns must run in the thread of the window.
        """

        def draw(fig: mplfig.Figure) -> RenderedPlot:
            plotargs = parse_cli_args(addargs, None, None, self.get_plot_tokens(values))
            check_viewer_args(plotargs)
            givenx = plotargs.x
            # draw_plot changes the arguments, thus the key comes first
            datakey = get_data_key(plotargs)
            lastdata = self.lastdata
            figuresdata = lastdata[1] if lastdata is not None and lastdata[0] == datakey else None
            reuseddata = figuresdata is not None
            figuresdata = draw_plot(plotargs, fig, self.modeldata.batchcaches, figuresdata)
            self.lastdata = (datakey, figuresdata)
            figuredata = figuresdata[0][0]
            # render_command gives the figure these colours after the draw, see apply_dark_colours
            darkcolours = DARKMODE_COLOURS if plotargs.darkmode else self.darkcolours
            islines = isinstance(figuredata, LineFigureData)
            return RenderedPlot(
                isimage=plotargs.dimensionreduce == 2,
                xlimitscale=C_cm_per_s / km_to_cm if plotargs.x == "beta" and givenx != "beta" else 1.0,
                xbins=plotargs.xbins,
                markers=bool(plotargs.markers),
                colorbyion=bool(plotargs.colorbyion),
                swatches=get_series_swatches(figuredata, plotargs, darkcolours[1] if darkcolours else None)
                if islines
                else {},
                elementcolourrows=frozenset(
                    index for index, (_, settings) in enumerate(figuredata.subplots) if settings.get("elementcolours")
                )
                if islines
                else frozenset(),
                reuseddata=reuseddata,
            )

        def keep(plot: RenderedPlot) -> None:
            self.isimage, self.xlimitscale, self.plotxbins, self.plotmarkers, self.plotcolorbyion = plot[:5]
            self.swatches, self.elementcolourrows = plot.swatches, plot.elementcolourrows
            # a plot with the data of the last plot printed none of its warnings, e.g. of an ion that the model lacks
            if plot.reuseddata:
                self.warning = self.warning or self.datawarning
            else:
                self.datawarning = self.warning

        return render_command(self, values, draw, keep, quiet=quiet)

    def get_fitted_figwidthscale(self, areawidth: float, areaheight: float) -> float:
        """Return the --figwidthscale that gives the figure the shape of the plot area.

        A colour image keeps its aspect, thus a wider figure only adds space at the sides of the images. The figure of
        the images has the shape of its content at --figwidthscale 1, and the window scales all the figure to the area.
        """
        if self.isimage:
            return 1.0
        # each column of subplots shows its own y labels, thus a column after the first adds the width of a label
        ncols = get_subplot_grid(len(self.values.subplots), get_subplots_per_row(self.values.otheroptions))[1]
        marginwidth = ncols * LABELWIDTH_INCHES + RIGHTMARGIN_INCHES
        return get_fitted_figwidthscale(self.figsize, self.values.figwidthscale, marginwidth, areawidth, areaheight)

    def get_xlimit_texts(self, low: float, high: float) -> tuple[str, str] | None:
        """Return the --xmin and --xmax texts of two positions on the x axis, or None if the positions have no width."""
        return get_short_limits(low * self.xlimitscale, high * self.xlimitscale)


def get_python_code(
    parser: argparse.ArgumentParser, tokens: "Sequence[str]", estimatorcolumns: "Collection[str]"
) -> str:
    """Return the Python code that draws the plot of the command, with the plot list in the form of main.

    The code gives each argument that differs from its default, as the command does.
    """
    args = parse_command_tokens(parser, tokens)
    if args is None:
        return "# plotestimators rejects the command"
    resolve_positional_args(args)
    changed = get_changed_arguments(parser, args, skip={"plotitems", "plotlist"})
    # -timestep reads text, because it also takes a range such as "10-12". main takes one timestep as an int
    if isinstance(timestep := changed.get("timestep"), str) and timestep.isascii() and timestep.isdecimal():
        changed["timestep"] = int(timestep)
    modelpath = {"modelpath": changed.pop("modelpath")} if "modelpath" in changed else {}
    plotlist = [get_python_plotitems(subplot, estimatorcolumns) for subplot in args.plotlist or []]
    return get_python_call(
        "at.estimators.plotestimators.main", {**modelpath, **changed, **({"plotlist": plotlist} if plotlist else {})}
    )


def get_nearest_cell(viewer: EstimatorViewer, xdata: float) -> int | None:
    """Return the cell with the radial velocity nearest to a position on a velocity axis, or None for another axis.

    A 3D model has many cells at one radial velocity, and the function gives one of them.
    """
    if viewer.values.x not in {"velocity", "beta"} or not viewer.modeldata.cellvelocities:
        return None
    axisisbeta = viewer.values.x == "beta" or viewer.xlimitscale != 1.0
    velocity = xdata * (C_cm_per_s if axisisbeta else km_to_cm)
    cells = np.fromiter(
        viewer.modeldata.cellvelocities.keys(), dtype=np.int64, count=len(viewer.modeldata.cellvelocities)
    )
    velocities = np.fromiter(
        viewer.modeldata.cellvelocities.values(), dtype=np.float64, count=len(viewer.modeldata.cellvelocities)
    )
    return int(cells[np.argmin(np.abs(velocities - velocity))])


def get_evolution_values(viewer: EstimatorViewer, cells: str) -> ControlValues:
    """Return the values that plot the cells against time over all the timesteps."""
    return viewer.set_xvariable(dc.replace(viewer.values, cells=cells), "time")


def get_snapshot_values(viewer: EstimatorViewer, xdata: float) -> ControlValues | None:
    """Return the values of a snapshot at a position on a time axis, or None for another axis."""
    if viewer.values.x == "time":
        validtmids = [viewer.tmids[timestep] for timestep in viewer.modeldata.validtimesteps]
        position = get_nearest_range_start(validtmids, xdata, 1)
    elif viewer.values.x == "timestep":
        position = get_nearest_range_start(viewer.modeldata.validtimesteps, xdata, 1)
    else:
        return None
    snapshotx = viewer.get_default_xvariable(viewer.values.otheroptions, timegiven=True)
    return viewer.select_positions(viewer.set_xvariable(viewer.values, snapshotx), (position, position))


def get_xunit_text(xlimitscale: float, xvariable: str) -> str:
    """Return the unit of --xmin and --xmax, e.g. " [km/s]", or an empty text for a variable with no unit.

    A scale of the x limits shows that the axis gives v/c and the options take km/s.
    """
    return plain_label(get_units_string("velocity" if xlimitscale != 1.0 else xvariable))


def replace_option_rows(viewer: EstimatorViewer, values: ControlValues, otheroptions: OptionRows) -> ControlValues:
    """Return the values with new rows of the option table.

    An option such as --slice changes the default x of a snapshot. An x that equals the old default follows the new
    default, because the command then gives no -x, as plotestimators does.
    """
    newvalues = dc.replace(values, otheroptions=otheroptions)
    if is_evolution(values) or values.x != viewer.get_default_xvariable(values.otheroptions, timegiven=True):
        return newvalues
    return viewer.set_xvariable(newvalues, viewer.get_default_xvariable(otheroptions, timegiven=True))


def get_series_at(axis: "mplax.Axes", event: "MouseEvent", *, legendonly: bool) -> str | None:
    """Return the name of the series of the legend entry or the line under the pointer, or None.

    plotestimators gives each line a gid from the name of its series, see get_series_gid. The legend holds copies of
    the lines with no gid, thus the order of the legend entries gives their series.
    """
    legend = axis.get_legend()
    # test the box of the legend first, because a mouse movement tests each entry only inside the legend
    if legend is not None and legend.get_visible() and legend.contains(event)[0]:
        handles, _ = get_legend_entries_in_draw_order(axis)
        for handle, legendhandle, text in zip(handles, legend.legend_handles, legend.get_texts(), strict=False):
            hit = text.contains(event)[0] or (legendhandle is not None and legendhandle.contains(event)[0])
            if hit and isinstance(gid := handle.get_gid(), str):
                return get_gid_series_name(gid)
    if legendonly:
        return None
    return next(
        (
            get_gid_series_name(gid)
            for line in axis.get_lines()
            if isinstance(gid := line.get_gid(), str) and line.contains(event)[0]
        ),
        None,
    )


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


# the keys and the mouse actions of the window. get_keyboard_help adds the shortcuts of the menus
KEYBOARD_HELP_ROWS: t.Final = (
    *TIME_KEY_HELP_ROWS,
    ("<b>Page Up</b>, <b>Page Down</b>", "Select the previous or the next cell"),
    (
        "<b>Space</b>",
        "Play or pause. A snapshot moves through the timesteps, and a plot against time moves through the cells",
    ),
    ("<b>Drag</b> across a plot", "Select the x range"),
    ("<b>Shift-drag</b> up or down a subplot", "Select the y range of the subplot (ymin= and ymax=)"),
    ("<b>Double-click</b> a plot", "Show the x range of the data"),
    ("<b>Right-click</b> a subplot", "Show the menu of the subplot, e.g. the y scale"),
    ("<b>Alt-Up</b>, <b>Alt-Down</b> on the header of a subplot", "Move the subplot up or down (Option on a Mac)"),
    ("<b>Click</b> a legend entry or a chip", "Change the line of the series, e.g. its colour"),
    ("<b>Drag</b> a chip", "Move the series to a different place or to a different subplot"),
    (
        "<b>Alt-Up</b>, <b>Alt-Down</b> on a chip",
        "Move the series one place to the front or the back (Option on a Mac)",
    ),
)


def open_window(
    tokens: "Sequence[str]", windows: "list[QtWidgets.QMainWindow]", *, newwindow: bool = True
) -> str | None:
    """Open a window of the viewer for the plotestimators arguments in tokens, or return the reason for no window."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # the Settings window can give a new window options, e.g. -figscale, that the command does not give
    # a window of the last session keeps its command, and a new window takes the options of the Settings window
    viewer = EstimatorViewer(
        add_default_options(make_parser(addargs), tokens) if newwindow else tokens, mplfig.Figure()
    )
    viewerwindow = start_viewer_window(
        APPLICATION_NAME, viewer, windows, (viewer.modelpath, resolve_modelpath(viewer.modelpath).name)
    )
    if isinstance(viewerwindow, str):
        return viewerwindow
    window, canvas, plotarea, panellayout, fittimer = viewerwindow
    helptexts = viewer.helptexts

    _, timegrid = add_section(panellayout, "Time")
    timecontrols = add_time_controls(
        timegrid,
        0,
        (
            (
                "The middle of the time range in days. The Left key and the Right key move it to the adjacent"
                " timestep, and a typed time moves it to the timestep that holds the time."
            ),
            "The number of timesteps of the time range. The Up key and the Down key change it.",
            (
                "Move a snapshot through the timesteps of the model, or move a plot against time through the cells."
                " After the last step, Play starts again at the first step (Space)"
            ),
        ),
    )
    timeslider, timeedit, widthlabel, widthslider, widthedit, timestepslabel, stepbuttons, fpsbox, playbutton = (
        timecontrols
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
    timegrid.addWidget(trangebox, 0, 1, 1, 2)

    # most plots need the time, the x axis, and the subplots, thus the other sections start closed
    _, cellgrid = add_section(panellayout, "Cells", closed=True)
    geometrybox = QtWidgets.QComboBox()
    # the box fills the width of the sidebar, and a narrow sidebar cuts its long texts
    geometrybox.setMinimumWidth(120)
    for mode in get_geometry_choices(viewer.dimensions):
        geometrybox.addItem(GEOMETRY_MODES[mode], mode)
    geometrybox.setToolTip(
        "The cells that the plot reads. x, y, and z are the velocity coordinates of the model grid. A 2D plane slice"
        " and the mean over rings give a colour image of a snapshot."
    )
    cellslider = make_slider()
    cellslider.setToolTip("Select one cell. The Page Up key and the Page Down key select the adjacent cell.")
    celledit = QtWidgets.QLineEdit()
    celledit.setFixedWidth(110)
    celledit.setPlaceholderText("e.g. 3 or 3-7")
    celledit.setToolTip(helptexts.get("modelgridindex", ""))
    celllabel = QtWidgets.QLabel()
    cellnamelabel = QtWidgets.QLabel("--cell")
    axisbox = QtWidgets.QComboBox()
    axisbox.addItems(["+x", "-x", "+y", "-y", "+z", "-z"])
    axisbox.setToolTip(helptexts.get("axis", ""))
    coneanglelabel = QtWidgets.QLabel("Full angle:")
    coneanglebox = QtWidgets.QDoubleSpinBox()
    coneanglebox.setRange(1.0, 180.0)
    coneanglebox.setSuffix("°")
    coneanglebox.setDecimals(1)
    coneanglebox.setKeyboardTracking(False)
    coneanglebox.setToolTip(helptexts.get("coneangle", ""))
    planebox = QtWidgets.QComboBox()
    planebox.addItems(list(PLANE_OF_NORMAL.values()))
    planebox.setToolTip("The plane of the image")
    offsetedit = QtWidgets.QLineEdit()
    offsetedit.setFixedWidth(110)
    offsetedit.setPlaceholderText("0")
    offsetedit.setToolTip(
        "The velocity of the plane along the axis that is normal to it, e.g. -0.2c, or 5000km/s. Empty gives 0."
    )
    lineaxisbox = QtWidgets.QComboBox()
    for axis in "xyz":
        lineaxisbox.addItem(get_line_label(axis, ""), axis)
    lineaxisbox.setToolTip("The axis of the line. The positions on the other two axes follow it")

    def make_parameter_row(widgets: "Sequence[QtWidgets.QWidget]") -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        rowlayout = make_row_layout(widgets)
        rowlayout.setContentsMargins(0, 0, 0, 0)
        row.setLayout(rowlayout)
        return row

    axisparameters = make_parameter_row([QtWidgets.QLabel("--axis"), axisbox, coneanglelabel, coneanglebox])
    # show_blocked_values names the axis that is normal to the plane, e.g. "at z ="
    planeatlabel = QtWidgets.QLabel()
    planeparameters = make_parameter_row([QtWidgets.QLabel("Plane:"), planebox, planeatlabel, offsetedit])
    lineparameters = make_parameter_row([QtWidgets.QLabel("Line along:"), lineaxisbox])
    projectionaxisbox = QtWidgets.QComboBox()
    projectionaxisbox.addItems(["x", "y", "z"])
    projectionaxisbox.setToolTip(helptexts.get("projection", ""))
    projectionparameters = make_parameter_row([QtWidgets.QLabel("Mean along:"), projectionaxisbox])
    add_row(cellgrid, 0, [geometrybox])
    cellgrid.addWidget(cellnamelabel, 1, 0)
    cellgrid.addWidget(cellslider, 1, 1)
    cellgrid.addWidget(celledit, 1, 2)
    cellgrid.addWidget(celllabel, 2, 0, 1, -1)
    # the set of cells in the terms of the model grid, which show_blocked_values writes
    geometrydescription = make_note_label()
    parameterrows = (axisparameters, planeparameters, lineparameters, projectionparameters, geometrydescription)
    for row, widget in enumerate(parameterrows, start=3):
        cellgrid.addWidget(widget, row, 0, 1, -1)

    _, xgrid = add_section(panellayout, "Horizontal axis")
    xbox = QtWidgets.QComboBox()
    xbox.setEditable(True)
    xbox.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
    xbox.addItems([*XVARIABLES, *(column for column in viewer.modeldata.estimatorcolumns if column not in XVARIABLES)])
    if (completer := xbox.completer()) is not None:
        set_search_completion(completer)
    xbox.setToolTip(helptexts.get("x", ""))
    xminedit, xmaxedit, xbinsedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    xminlabel, xmaxlabel = QtWidgets.QLabel("--xmin"), QtWidgets.QLabel("--xmax")
    # show_values adds the unit to the text, thus the tooltip gives the flag here
    xminlabel.setToolTip("--xmin")
    xmaxlabel.setToolTip("--xmax")
    zoomtip = " Drag across a plot to select a range. Double-click a plot to show the range of the data."
    for edit in (xminedit, xmaxedit):
        edit.setFixedWidth(100)
        edit.setPlaceholderText("auto")
    # a field takes each value of -xbins, e.g. a negative value for the automatic bins
    # the placeholder gives the bins of the plot, e.g. "auto: no bins", and the field shows it in full
    xbinsedit.setFixedWidth(110)
    xbinsedit.setPlaceholderText("default")
    # an empty field removes -xbins. QIntValidator gives the Intermediate state for an empty text, and the field
    # then sends no editingFinished signal
    xbinsedit.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression(r"(-?\d+)?"), xbinsedit))
    xbinsedit.setToolTip(helptexts.get("xbins", ""))
    markerscheck = QtWidgets.QCheckBox("--markers")
    colorbyioncheck = QtWidgets.QCheckBox("--colorbyion")
    # show_values adds a note to the text of these checkboxes, thus the tooltip gives the flag here
    markerscheck.setToolTip(f"{helptexts.get('markers', '')} (--markers)")
    colorbyioncheck.setToolTip(f"{helptexts.get('colorbyion', '')} (--colorbyion)")
    histogramcheck = QtWidgets.QCheckBox("--histogram")
    histogramcheck.setToolTip(helptexts.get("histogram", ""))
    add_row(xgrid, 0, [QtWidgets.QLabel("-x"), xbox, QtWidgets.QLabel("--xbins"), xbinsedit])
    add_row(xgrid, 1, [xminlabel, xminedit, xmaxlabel, xmaxedit])
    add_row(xgrid, 2, [markerscheck, colorbyioncheck, histogramcheck])
    smoothingbox = QtWidgets.QComboBox()
    smoothingbox.setMinimumWidth(120)
    for mode, modetext in SMOOTHING_MODES.items():
        smoothingbox.addItem(modetext, mode)
    smoothingbox.setToolTip("Smooth the line of each series")
    smoothinglengthlabel, smoothingorderlabel = QtWidgets.QLabel("Length:"), QtWidgets.QLabel("Order:")
    smoothinglengthbox, smoothingorderbox = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
    # show_blocked_values gives the length box its range and its step for the mode
    smoothinglengthbox.setRange(2, 999)
    smoothingorderbox.setRange(0, 998)
    for box in (smoothinglengthbox, smoothingorderbox):
        box.setKeyboardTracking(False)
    smoothinglengthbox.setToolTip("The number of points of the moving average, or the window length of the filter")
    smoothingorderbox.setToolTip("The order of the polynomial of the Savitzky-Golay filter, less than the length")
    add_row(
        xgrid,
        3,
        [
            QtWidgets.QLabel("Smoothing:"),
            smoothingbox,
            smoothinglengthlabel,
            smoothinglengthbox,
            smoothingorderlabel,
            smoothingorderbox,
        ],
    )

    def plot_shows_values() -> bool:
        """Return True if the plot on the screen has the subplots, the x variable, and the options of the controls.

        DrawQueue gives the viewer the new values at once, and the old plot stays until the worker draws the new one.
        A mouse action on an old frame then must not change a different subplot or read a different x variable.
        """
        drawn, values = queue.drawnvalues, viewer.values
        return (drawn.subplots, drawn.x, drawn.otheroptions) == (values.subplots, values.x, values.otheroptions)

    # the queue of the plots needs show_values, which shows the cards, thus the cards find the queue when they run
    show_subplot_section, edit_item_style, set_directives, on_add_series_key, on_add_subplot_key = add_subplot_section(
        panellayout, window, viewer, lambda: queue, plot_shows_values
    )

    _, appearancegrid = add_section(panellayout, "Appearance", closed=True)
    actionsbyflag = get_actions_by_flag(viewer.parser)
    # a flag that this version of plotestimators does not have gives no control
    appearancechecks = {
        flag: QtWidgets.QCheckBox(flag)
        for flag in ("--notitle", "--nolegend", "--legendframe", "--legend-above", "--hidexlabel")
        if flag in actionsbyflag
    }
    for flag, check in appearancechecks.items():
        check.setToolTip(helptexts.get(actionsbyflag[flag].dest, ""))
    # a wide range and two decimals keep the value of the command, because a box rounds and clamps its value
    fontsizebox = QtWidgets.QDoubleSpinBox()
    fontsizebox.setRange(0.0, 100.0)
    fontsizebox.setDecimals(2)
    fontsizebox.setSpecialValueText("default")
    fontsizebox.setToolTip(helptexts.get("labelfontsize", ""))
    figscalebox, show_figscale = make_figscale_box(
        viewer.parser,
        helptexts,
        lambda: viewer.values.otheroptions,
        lambda rows: apply(replace_option_rows(viewer, viewer.values, rows)),
    )
    subplotsperrowbox = QtWidgets.QSpinBox()
    subplotsperrowbox.setRange(1, 12)
    subplotsperrowbox.setToolTip(helptexts.get("subplotsperrow", ""))
    linewidthscalebox = QtWidgets.QDoubleSpinBox()
    linewidthscalebox.setRange(0.1, 10.0)
    linewidthscalebox.setSingleStep(0.25)
    linewidthscalebox.setDecimals(2)
    linewidthscalebox.setToolTip(helptexts.get("linewidthscale", ""))
    for box in (fontsizebox, figscalebox, subplotsperrowbox, linewidthscalebox):
        box.setKeyboardTracking(False)
    add_row(appearancegrid, 0, list(appearancechecks.values()))
    add_row(
        appearancegrid,
        1,
        [QtWidgets.QLabel("--labelfontsize"), fontsizebox, QtWidgets.QLabel("--figscale"), figscalebox],
    )
    add_row(
        appearancegrid,
        2,
        [
            QtWidgets.QLabel("--subplotsperrow"),
            subplotsperrowbox,
            QtWidgets.QLabel("--linewidthscale"),
            linewidthscalebox,
        ],
    )

    # the table offers each option that a section sets too, as the table of plotspectra does, thus the user can edit
    # each option of the command there. A section and the table show the same rows
    tablehiddendests = CONTROLLED_DESTS | OUTPUT_DESTS | TABLE_EXCLUDED_DESTS | MODELFORMAT_DESTS

    def get_table_rows(rows: OptionRows) -> OptionRows:
        """Return the rows that the option table shows, which are all the rows except those of MODELFORMAT_DESTS."""
        return tuple(row for row in rows if row[0] not in viewer.modelformatflags)

    def on_option_rows(rows: OptionRows) -> None:
        modelformatrows = tuple(row for row in viewer.values.otheroptions if row[0] in viewer.modelformatflags)
        queue.apply(replace_option_rows(viewer, viewer.values, (*rows, *modelformatrows)))

    commandsections = add_command_sections(
        viewerwindow,
        viewer,
        viewer.parser,
        (tablehiddendests, get_table_rows(viewer.values.otheroptions), on_option_rows),
    )
    figuresection, statusbar = commandsections.figuresection, commandsections.statusbar

    signalwidgets: list[QtWidgets.QWidget] = [
        figuresection.dpibox,
        timeslider,
        widthslider,
        cellslider,
        xbox,
        markerscheck,
        colorbyioncheck,
        histogramcheck,
        geometrybox,
        axisbox,
        coneanglebox,
        planebox,
        lineaxisbox,
        projectionaxisbox,
        smoothingbox,
        smoothinglengthbox,
        smoothingorderbox,
        fontsizebox,
        figscalebox,
        subplotsperrowbox,
        linewidthscalebox,
        *appearancechecks.values(),
    ]

    def set_ranges() -> None:
        """Give the sliders the number of valid timesteps and the number of cells of the model.

        A shorter range clamps the value of a slider. The handler of the slider must not change the values of the
        viewer then, and show_values gives each slider its value.
        """
        nvalid = len(viewer.modeldata.validtimesteps)
        with blocked_signals([timeslider, widthslider, cellslider]):
            timeslider.setRange(0, nvalid - 1)
            widthslider.setRange(1, nvalid)
            cellslider.setRange(0, max(len(viewer.modeldata.cells) - 1, 0))
        set_trange_steps(max(nvalid - 1, 1))

    set_ranges()

    def show_values() -> None:
        """Show the values of the viewer on each widget, and block the signals that change the values again."""
        with blocked_signals(signalwidgets):
            show_blocked_values()
        fit_canvas(canvas, viewer.figsize, plotarea)

    def show_blocked_values() -> None:
        values = viewer.values
        firstpos, lastpos = viewer.get_selection_positions()
        evolution = is_evolution(values)
        for widget in (timeslider, timeedit, widthlabel, widthslider, widthedit):
            widget.setVisible(not evolution)
        stepbuttons[0].setEnabled(firstpos > 0)
        stepbuttons[1].setEnabled(lastpos < len(viewer.modeldata.validtimesteps) - 1)
        trangebox.setVisible(evolution)
        set_trange_positions(firstpos, lastpos)
        set_edit_text(tminedit, f"{viewer.tmids[values.first]:.4g}")
        set_edit_text(tmaxedit, f"{viewer.tmids[values.last]:.4g}")
        rows = values.otheroptions
        geometrymode = get_geometry_mode(values)
        geometrybox.setCurrentIndex(max(geometrybox.findData(geometrymode), 0))
        for widget in (cellnamelabel, celledit):
            widget.setVisible(geometrymode == "cells")
        # the slider selects one cell, which suits a plot against time. A snapshot of one cell has one point, and
        # the field of a snapshot takes a list or a range of cells
        cellslider.setVisible(geometrymode == "cells" and evolution)
        # the text below the controls gives the cells of a 3D or a 2D model, which leaves out the corners
        celllabel.setVisible(geometrymode == "cells" or (geometrymode == "all" and viewer.dimensions == 1))
        axisparameters.setVisible(geometrymode in {"alongaxis", "cone"})
        for widget in (coneanglelabel, coneanglebox):
            widget.setVisible(geometrymode == "cone")
        axisbox.setCurrentText((get_row_values(rows, "--axis") or (viewer.parser.get_default("axis"),))[0])
        with contextlib.suppress(ValueError):
            coneangle = get_row_values(rows, "--coneangle") or (str(viewer.parser.get_default("coneangle")),)
            set_spin_value(coneanglebox, float(coneangle[0]))
        slicetext = (get_row_values(rows, "--slice") or ("",))[0]
        planeparameters.setVisible(geometrymode == "plane")
        lineparameters.setVisible(geometrymode == "line")
        projectionparameters.setVisible(geometrymode == "projection")
        projectionaxisbox.setCurrentText((get_row_values(rows, "--projection") or ("z",))[0])
        plane, offset = get_slice_parts(slicetext)
        planebox.setCurrentText(plane)
        description = get_geometry_description(viewer.get_plot_tokens(values), viewer.modelmeta)
        set_note_text(geometrydescription, get_first_sentence(description), description)
        normal = next(axis for axis, planeaxes in PLANE_OF_NORMAL.items() if planeaxes == plane)
        planeatlabel.setText(f"at {normal}:")
        set_edit_text(offsetedit, offset)
        for index, axis in enumerate("xyz"):
            lineaxisbox.setItemText(index, get_line_label(axis, slicetext))
        lineaxisbox.setCurrentIndex(lineaxisbox.findData(get_line_axis(slicetext)))
        smoothingmode, _ = get_smoothing(rows)
        smoothingbox.setCurrentIndex(max(smoothingbox.findData(smoothingmode), 0))
        for widget in (smoothinglengthlabel, smoothinglengthbox):
            widget.setVisible(smoothingmode != "none")
        for widget in (smoothingorderlabel, smoothingorderbox):
            widget.setVisible(smoothingmode == "savgol")
        # the Savitzky-Golay filter takes an odd window length of at least 3, thus the arrows of the length box skip
        # the even lengths
        issavgol = smoothingmode == "savgol"
        if smoothinglengthbox.minimum() != (3 if issavgol else 2):
            smoothinglengthbox.setRange(3 if issavgol else 2, 999)
            smoothinglengthbox.setSingleStep(2 if issavgol else 1)
        length, order = get_smoothing_numbers()
        set_spin_value(smoothinglengthbox, length)
        set_spin_value(smoothingorderbox, order)
        for flag, check in appearancechecks.items():
            check.setChecked(get_row_values(rows, flag) is not None)
        with contextlib.suppress(ValueError):
            set_spin_value(fontsizebox, float((get_row_values(rows, "--labelfontsize") or ("0",))[0]))
        show_figscale(rows)
        set_spin_value(subplotsperrowbox, get_subplots_per_row(rows))
        set_spin_value(linewidthscalebox, get_linewidthscale(viewer.parser, rows))
        timeslider.setValue((firstpos + lastpos) // 2)
        widthslider.setValue(lastpos - firstpos + 1)
        set_edit_text(widthedit, str(lastpos - firstpos + 1))
        set_edit_text(timeedit, get_time_text(viewer.tmids, values))
        timestepslabel.setText(viewer.get_time_range_text())
        set_edit_text(celledit, values.cells)
        if (cell := get_single_cell(values.cells)) is not None and cell in viewer.modeldata.cells:
            cellslider.setValue(viewer.modeldata.cells.index(cell))
        celllabel.setText(viewer.get_cell_text())
        xbox.setCurrentText(values.x)
        xunit = get_xunit_text(viewer.xlimitscale, values.x)
        xminlabel.setText(f"{FLAG_LABELS['--xmin']}{xunit}:")
        xmaxlabel.setText(f"{FLAG_LABELS['--xmax']}{xunit}:")
        axisisbeta = viewer.xlimitscale != 1.0
        unittip = " The axis shows v/c, and the option takes km/s." if axisisbeta else ""
        for edit, dest in ((xminedit, "xmin"), (xmaxedit, "xmax")):
            edit.setToolTip(helptexts.get(dest, "") + zoomtip + unittip)
        set_edit_text(xminedit, values.xmin)
        set_edit_text(xmaxedit, values.xmax)
        set_edit_text(xbinsedit, values.xbins)
        # the window hides the output of the plot. Thus the controls show the bins, the markers, and the colours
        # that the plot chose when the command gives no option
        if viewer.isimage:
            xbinsedit.setPlaceholderText("default")
        else:
            xbinsedit.setPlaceholderText("auto: no bins" if viewer.plotxbins is None else f"auto: {viewer.plotxbins}")
        markerscheck.setChecked(values.markers)
        markersnote = " (on for --xbins 0)" if viewer.plotmarkers and not values.markers else ""
        markerscheck.setText(FLAG_LABELS["--markers"] + markersnote)
        colorbyioncheck.setChecked(values.colorbyion)
        histogramcheck.setChecked(values.histogram)
        colorbyionnote = " (on for bins)" if viewer.plotcolorbyion and not values.colorbyion else ""
        colorbyioncheck.setText(FLAG_LABELS["--colorbyion"] + colorbyionnote)
        show_subplot_section()
        show_command_sections(
            commandsections,
            viewer,
            get_table_rows(values.otheroptions),
            get_python_code(viewer.parser, viewer.get_plot_tokens(), viewer.modeldata.estimatorcolumns),
        )

    def after_draw(message: str | None) -> None:
        # matplotlib keeps the connections of the mouse in the figure, and each plot has a new figure
        connect_mouse_to_figure()
        # a new number of subplots changes the height of the figure, thus the plot can need a new -figwidthscale
        fittimer.start()
        continue_play(message)

    queue = DrawQueue(
        window, viewer, statusbar, show_values, after_draw, render=viewer.render, keep_on_undo=keep_figwidthscale
    )
    apply, show_error = queue.apply, queue.show_error

    def on_time(position: int) -> None:
        nvalid = len(viewer.modeldata.validtimesteps)
        positions = get_centred_range(viewer.get_selection_positions(), position, nvalid)
        apply(viewer.select_positions(viewer.values, positions))

    def on_width(count: int) -> None:
        firstpos, _ = viewer.get_selection_positions()
        positions = get_clamped_range(firstpos, count, len(viewer.modeldata.validtimesteps))
        apply(viewer.select_positions(viewer.values, positions))

    def on_widthedit() -> None:
        # a later plot can show new text in the field only when the field has no edit of the user
        widthedit.setModified(False)
        text = widthedit.text().strip()
        if not (text.isascii() and text.isdecimal()) or int(text) < 1:
            show_error("The number of timesteps must be a whole number of 1 or more")
            return
        on_width(int(text))

    def on_timeedit() -> None:
        # the field shows a rounded time, thus a Return in the field with no edit keeps the range
        if not timeedit.isModified():
            return
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
        apply(viewer.select_positions(viewer.values, (firstpos, lastpos)))

    def on_trangeedit() -> None:
        tminedit.setModified(False)
        tmaxedit.setModified(False)
        try:
            firstdays, lastdays = float(tminedit.text()), float(tmaxedit.text())
        except ValueError:
            show_error("Give a number of days for the first and the last time")
            return
        validtmids = [viewer.tmids[timestep] for timestep in viewer.modeldata.validtimesteps]
        firstpos, lastpos = (get_nearest_range_start(validtmids, days, 1) for days in (firstdays, lastdays))
        if firstpos > lastpos:
            show_error("Give a first time that is before the last time")
            return
        apply(viewer.select_positions(viewer.values, (firstpos, lastpos)))

    def on_step_cell(step: int) -> None:
        if (values := viewer.step_cell(step)) is not None:
            apply(values)

    def get_play_values() -> ControlValues | None:
        """Return the values of the next step of Play, or None if Play has no step.

        After the last cell or the last timestep, Play starts again at the first one.
        """
        if not is_evolution(viewer.values):
            return get_next_play_time(viewer)
        # -cell does not select the cells of some plots, e.g. of a plane
        if not viewer.modeldata.cells or not cells_apply(viewer.values):
            return None
        return viewer.step_cell(1) or dc.replace(viewer.values, cells=str(viewer.modeldata.cells[0]))

    continue_play = connect_play(window, timecontrols, queue, get_play_values)

    def on_cell(position: int) -> None:
        if viewer.modeldata.cells:
            apply(dc.replace(viewer.values, cells=str(viewer.modeldata.cells[position])))

    def on_celledit() -> None:
        celledit.setModified(False)
        # -cell takes one word, thus a space between two cells becomes the comma of a list
        apply(dc.replace(viewer.values, cells=",".join(celledit.text().replace(",", " ").split())))

    def apply_rows(changes: "Mapping[str, tuple[str, ...] | None]") -> None:
        """Apply new values of some rows of the option table, e.g. of a section of the window."""
        apply(replace_option_rows(viewer, viewer.values, set_row_values(viewer.values.otheroptions, changes)))

    def on_geometry(index: int) -> None:
        apply(set_geometry_mode(viewer, viewer.values, str(geometrybox.itemData(index))))

    def on_axis(axis: str) -> None:
        apply_rows({"--axis": None if axis == viewer.parser.get_default("axis") else (axis,)})

    def on_coneangle(angle: float) -> None:
        isdefault = math.isclose(angle, viewer.parser.get_default("coneangle"))
        apply_rows({"--coneangle": None if isdefault else (format(angle, "g"),)})

    def on_plane() -> None:
        offsetedit.setModified(False)
        apply_rows({"--slice": (get_slice_text(planebox.currentText(), offsetedit.text()),)})

    def on_projectionaxis(axis: str) -> None:
        apply_rows({"--projection": (axis,)})

    def on_lineaxis(index: int) -> None:
        axis = str(lineaxisbox.itemData(index))
        # a pick of the current axis keeps the positions of the line
        if axis != get_line_axis((get_row_values(viewer.values.otheroptions, "--slice") or ("",))[0]):
            apply_rows({"--slice": (",".join(f"{other}=0" for other in "zyx" if other != axis),)})

    def get_smoothing_numbers() -> tuple[int, int]:
        """Return the length and the order of the smoothing of the values, or the first numbers of the boxes."""
        _, numbers = get_smoothing(viewer.values.otheroptions)
        return (numbers[0] if numbers else 5, numbers[1] if len(numbers) > 1 else 2)

    def set_smoothing_numbers(mode: str, length: int, order: int) -> None:
        # the order of the Savitzky-Golay filter must be less than its window length
        rows = set_smoothing(viewer.values.otheroptions, mode, (length, min(order, length - 1)))
        apply(dc.replace(viewer.values, otheroptions=rows))

    def on_smoothing_mode(index: int) -> None:
        mode = str(smoothingbox.itemData(index))
        length, order = get_smoothing_numbers()
        if mode == "savgol":
            # the filter takes an odd window length of at least 3
            length = max(length + 1 - length % 2, 3)
        set_smoothing_numbers(mode, length, order)

    def on_smoothing_length(length: int) -> None:
        mode, _ = get_smoothing(viewer.values.otheroptions)
        set_smoothing_numbers(mode, length, get_smoothing_numbers()[1])

    def on_smoothing_order(order: int) -> None:
        mode, _ = get_smoothing(viewer.values.otheroptions)
        set_smoothing_numbers(mode, get_smoothing_numbers()[0], order)

    # each control of the appearance changes only its own option, because a box rounds and clamps its value
    def on_appearance_check(flag: str, checked: bool) -> None:
        apply_rows({flag: () if checked else None})

    def on_fontsize(fontsize: float) -> None:
        apply_rows({"--labelfontsize": (format(fontsize, "g"),) if fontsize > 0.0 else None})

    def on_linewidthscale(scale: float) -> None:
        isdefault = math.isclose(scale, viewer.parser.get_default("linewidthscale"))
        apply_rows({"--linewidthscale": None if isdefault else (format(scale, "g"),)})

    def on_subplotsperrow(count: int) -> None:
        isdefault = count == viewer.parser.get_default("subplotsperrow")
        apply_rows({"--subplotsperrow": None if isdefault else (str(count),)})

    def on_xvariable() -> None:
        if xvariable := xbox.currentText().strip():
            apply(viewer.set_xvariable(viewer.values, xvariable))

    def on_xedit() -> None:
        if (texts := read_limit_fields([(xminedit, "--xmin"), (xmaxedit, "--xmax")], show_error)) is not None:
            apply(dc.replace(viewer.values, xmin=texts[0], xmax=texts[1]))

    def on_style() -> None:
        xbinsedit.setModified(False)
        apply(
            dc.replace(
                viewer.values,
                xbins=xbinsedit.text().strip(),
                markers=markerscheck.isChecked(),
                histogram=histogramcheck.isChecked(),
                colorbyion=colorbyioncheck.isChecked(),
            )
        )

    def on_reload() -> None:
        """Read the model again in the worker thread, and keep the window responsive.

        A conversion of new text files can take minutes, and the terminal shows the progress. The other viewers keep
        their caches.
        """
        modelpath, args, ntimesteps = viewer.modelpath, viewer.userargs, len(viewer.tmids)
        reloadedmodeldata: list[ModelData] = []

        def show_reloaded_model_data() -> None:
            reload_model_data(viewer, reloadedmodeldata[0])
            set_ranges()

        reload_models(
            queue,
            [modelpath],
            show_reloaded_model_data,
            show_error,
            estimatorsonly=True,
            read_models=lambda: reloadedmodeldata.append(read_model_data(modelpath, args, ntimesteps)),
        )

    def get_frame_readout(event: t.Any, frame: "mplax.Axes") -> str:
        if not viewer.isimage:
            return get_readout(frame, event.xdata)
        meshes = [collection for collection in frame.collections if collection.get_array() is not None]
        value = get_image_value(meshes[0].get_cursor_data(event)) if meshes else None
        valuetext = f"   value: {value:.4g}" if value is not None else ""
        return f"x = {event.xdata:.4g}c   y = {event.ydata:.4g}c{valuetext}"

    def on_select(low: float, high: float) -> None:
        if not plot_shows_values():
            show_error(WAIT_FOR_PLOT_MESSAGE)
        elif (limits := viewer.get_xlimit_texts(low, high)) is None:
            show_error("The selected x range has no width. Drag across a wider range")
        else:
            apply(dc.replace(viewer.values, xmin=limits[0], xmax=limits[1]))

    def get_subplot_row(frameindex: int) -> int | None:
        """Return the row of the subplot of a frame of the plot, or None for a colour image."""
        return frameindex if not viewer.isimage and frameindex < len(viewer.values.subplots) else None

    def on_select_y(frameindex: int, low: float, high: float) -> None:
        if (row := get_subplot_row(frameindex)) is None:
            return
        if not plot_shows_values():
            show_error(WAIT_FOR_PLOT_MESSAGE)
        elif (limits := read_selected_range(low, high, "y", show_error)) is not None:
            set_directives(row, {"ymin": limits[0], "ymax": limits[1]})

    def set_row_yscale(row: int, yscale: str) -> None:
        set_directives(row, {"yscale": yscale})

    def get_clicked_item(frameindex: int, event: "MouseEvent", *, legendonly: bool) -> tuple[int, int] | None:
        """Return the row of the subplot and the position of the series item under the pointer, or None.

        The pointer can be on a legend entry, or on a line if legendonly is False.
        """
        row = get_subplot_row(frameindex)
        if row is None or not plot_shows_values():
            return None
        seriesname = get_series_at(get_plot_frames(viewer.fig)[frameindex], event, legendonly=legendonly)
        subplot = viewer.values.subplots[row]
        position = next(
            (
                position
                for position, item in get_chip_items(subplot, viewer.modeldata.estimatorcolumns)
                if get_series_name(item) == seriesname
            ),
            None,
        )
        return (row, position) if position is not None else None

    def on_click(frameindex: int, event: "MouseEvent") -> None:
        if (clicked := get_clicked_item(frameindex, event, legendonly=True)) is not None:
            edit_item_style(*clicked)

    def get_click_hint(frameindex: int, event: "MouseEvent") -> str | None:
        if (clicked := get_clicked_item(frameindex, event, legendonly=True)) is None:
            return None
        return f"Click to change the line of {get_series_name(viewer.values.subplots[clicked[0]][clicked[1]])}"

    def on_menu(frameindex: int, event: "MouseEvent") -> None:
        """Show the menu of a subplot: the y scale, the y range, the plot of a cell or of a snapshot, and the figure."""
        if not plot_shows_values() or event.xdata is None:
            return
        menu = QtWidgets.QMenu(window)
        row = get_subplot_row(frameindex)
        # a line or a legend entry under the pointer gives the actions of its series first
        if (clicked := get_clicked_item(frameindex, event, legendonly=False)) is not None:
            name = get_series_name(viewer.values.subplots[clicked[0]][clicked[1]])
            styleaction = menu.addAction(f"Line Properties of {name}…")
            styleaction.triggered.connect(lambda: edit_item_style(*clicked))
            menu.addSeparator()
        if row is not None:
            add_y_axis_actions(
                menu,
                get_plot_frames(viewer.fig)[frameindex].get_yscale() == "log",
                any(get_item_directive(item) in {"ymin", "ymax"} for item in viewer.values.subplots[row]),
                partial(set_row_yscale, row),
                partial(set_directives, row, {"ymin": None, "ymax": None}),
            )
        if is_evolution(viewer.values):
            snapshot = get_snapshot_values(viewer, event.xdata)
            if snapshot is not None:
                snapshotaction = menu.addAction(f"Plot a Snapshot at {viewer.tmids[snapshot.first]:.4g} d")
                snapshotaction.triggered.connect(lambda: apply(snapshot))
        elif cells_apply(viewer.values):
            cell = get_nearest_cell(viewer, event.xdata)
            if cell is not None:
                cellaction = menu.addAction(f"Plot Cell {cell} Against Time")
                cellaction.triggered.connect(lambda: apply(get_evolution_values(viewer, str(cell))))
            if viewer.values.cells:
                cellsaction = menu.addAction(f"Plot Cells {viewer.values.cells} Against Time")
                cellsaction.triggered.connect(lambda: apply(get_evolution_values(viewer, viewer.values.cells)))
        # a context menu of a Mac app gives the actions on the object under the pointer, here the figure
        if menu.actions():
            menu.addSeparator()
        add_figure_actions(menu)
        if menu.actions():
            menu.exec(QtGui.QCursor.pos())
        # the window is the parent of the menu, thus without this the window keeps each menu until it closes
        menu.deleteLater()

    def get_animation_frames() -> "tuple[int, Callable[[int], list[str]]]":
        """Return the count of the steps of Play and the command of each step.

        For a snapshot, each step shows the next timestep. For a plot against time, each step shows the next cell.
        """
        values = viewer.values
        if is_evolution(values):
            cells = list(viewer.modeldata.cells)
            # -cell does not select the cells of some plots, e.g. of a plane
            if not cells or not cells_apply(values):
                return 1, lambda _index: viewer.get_plot_tokens(values)
            return len(cells), lambda index: viewer.get_plot_tokens(dc.replace(values, cells=str(cells[index])))
        firstpos, lastpos = viewer.get_selection_positions(values)
        width = lastpos - firstpos
        return (
            len(viewer.modeldata.validtimesteps) - width,
            lambda index: viewer.get_plot_tokens(viewer.select_positions(values, (index, index + width))),
        )

    command = ViewerCommand(
        name="plotestimators",
        main=plotestimators_main,
        parser=viewer.parser,
        get_python_code=lambda: get_python_code(
            viewer.parser, viewer.get_plot_tokens(), viewer.modeldata.estimatorcolumns
        ),
    )

    def on_export_animation() -> None:
        export_animation(window, queue, command, get_animation_frames(), fpsbox.value())

    def on_drop(paths: list[str]) -> None:
        """Open a new window for each dropped model folder."""
        for path in paths:
            if not Path(path).is_dir():
                show_error(f"plotestimators reads an ARTIS model folder, and {Path(path).name} is a file")
            elif (message := open_model_folder(path, open_window, windows)) is not None:
                show_error(message)

    def on_closed() -> None:
        if queue.task is not None:
            print("The reload of the model continues to its end, and then the process ends")
        # each kept scan holds the metadata of its file, e.g. 7.6 MB for 3000 columns
        scan_parquet_file.cache_clear()

    add_figure_actions = add_window_actions(
        window,
        windows,
        open_window,
        queue,
        command,
        figuresection,
        commandsections.copybuttons,
        KEYBOARD_HELP_ROWS,
        playbutton,
        extracallbacks={
            "Reload Data": on_reload,
            "Export Animation…": on_export_animation,
            "Add Series": on_add_series_key,
            "Add Subplot": on_add_subplot_key,
        },
    )
    set_drop_handler(window, on_drop)

    # the window keeps its command at a quit, and the next start opens the window again
    def get_session_tokens() -> list[str]:
        # a command with no folder reads the working folder, and the next start can be in a different folder. The
        # folder takes the place of the folder of the command, because a folder after an empty -plot removes that -plot
        return viewer.get_plot_tokens(modeltoken=viewer.modeltoken or str(Path.cwd()))

    timeslider.valueChanged.connect(on_time)
    widthslider.valueChanged.connect(on_width)
    widthedit.editingFinished.connect(on_widthedit)
    timeedit.editingFinished.connect(on_timeedit)
    connect_trange(on_trange)
    tminedit.editingFinished.connect(on_trangeedit)
    tmaxedit.editingFinished.connect(on_trangeedit)
    cellslider.valueChanged.connect(on_cell)
    celledit.editingFinished.connect(on_celledit)
    geometrybox.activated.connect(on_geometry)
    axisbox.textActivated.connect(on_axis)
    coneanglebox.valueChanged.connect(on_coneangle)
    planebox.textActivated.connect(on_plane)
    offsetedit.editingFinished.connect(on_plane)
    lineaxisbox.activated.connect(on_lineaxis)
    projectionaxisbox.textActivated.connect(on_projectionaxis)
    smoothingbox.activated.connect(on_smoothing_mode)
    smoothinglengthbox.valueChanged.connect(on_smoothing_length)
    smoothingorderbox.valueChanged.connect(on_smoothing_order)
    for flag, check in appearancechecks.items():
        check.toggled.connect(partial(on_appearance_check, flag))
    fontsizebox.valueChanged.connect(on_fontsize)
    subplotsperrowbox.valueChanged.connect(on_subplotsperrow)
    linewidthscalebox.valueChanged.connect(on_linewidthscale)
    xbox.activated.connect(on_xvariable)
    if (xlineedit := xbox.lineEdit()) is not None:
        xlineedit.editingFinished.connect(on_xvariable)
    xminedit.editingFinished.connect(on_xedit)
    xmaxedit.editingFinished.connect(on_xedit)
    xbinsedit.editingFinished.connect(on_style)
    markerscheck.toggled.connect(on_style)
    colorbyioncheck.toggled.connect(on_style)
    histogramcheck.toggled.connect(on_style)
    connect_mouse_to_figure = connect_plot_mouse(
        canvas,
        get_frames=lambda: get_plot_frames(viewer.fig),
        get_readout=get_frame_readout,
        readoutlabel=statusbar.readout,
        on_select=on_select,
        on_reset=lambda: apply(dc.replace(viewer.values, xmin="", xmax="")),
        # a colour image has a velocity on each axis, and -xmin and -xmax take the velocity of the line plot
        can_select=lambda: not viewer.isimage,
        on_select_y=on_select_y,
        on_menu=on_menu,
        show_tag=make_readout_tag(canvas),
        on_click=on_click,
        get_click_hint=get_click_hint,
    )
    connect_time_keys(
        window,
        stepbuttons,
        viewer,
        apply,
        extrakeys=(
            (QtCore.Qt.Key.Key_PageUp, partial(on_step_cell, -1)),
            (QtCore.Qt.Key.Key_PageDown, partial(on_step_cell, 1)),
        ),
    )

    finish_viewer_window(
        viewerwindow, windows, viewer, queue, get_session_tokens, on_closed, viewer.get_fitted_figwidthscale
    )
    show_values()
    return None
