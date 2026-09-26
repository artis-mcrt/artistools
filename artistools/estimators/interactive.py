"""Show the plot of plotestimators in a window with controls for the time, the cell, and the subplots."""

import argparse
import contextlib
import dataclasses as dc
import math
import shlex
import typing as t
from functools import lru_cache
from functools import partial
from pathlib import Path
from types import MappingProxyType

import matplotlib.figure as mplfig
import numpy as np
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg

from artistools.atomic import get_ionstring
from artistools.constants import C_cm_per_s
from artistools.constants import km_to_cm
from artistools.estimators.core import convert_estimator_batch_caches
from artistools.estimators.core import get_estimator_batch_states
from artistools.estimators.core import get_units_string
from artistools.estimators.core import join_cell_modeldata
from artistools.estimators.core import scan_estimators
from artistools.estimators.core import scan_parquet_file
from artistools.estimators.core import split_species_suffix
from artistools.estimators.estimators_classic import read_classic_estimators_cached
from artistools.estimators.plotestimators import add_plot_columns
from artistools.estimators.plotestimators import addargs
from artistools.estimators.plotestimators import default_plotitem_has_data
from artistools.estimators.plotestimators import DIRECTIVES
from artistools.estimators.plotestimators import draw_plot
from artistools.estimators.plotestimators import get_default_plotlist
from artistools.estimators.plotestimators import get_default_x
from artistools.estimators.plotestimators import get_iontuple
from artistools.estimators.plotestimators import get_iontuple_sortkey
from artistools.estimators.plotestimators import get_ylabel
from artistools.estimators.plotestimators import is_ionseriestype
from artistools.estimators.plotestimators import is_seriestype
from artistools.estimators.plotestimators import is_valid_ion
from artistools.estimators.plotestimators import POPTYPE_YLABELS
from artistools.estimators.plotestimators import require_artis_folder
from artistools.estimators.plotestimators import resolve_positional_args
from artistools.estimators.plotestimators import resolve_snapshot_arguments
from artistools.estimators.plotestimators import time_is_given
from artistools.estimators.plotestimators import TIME_XVARIABLES
from artistools.estimators.plotestimators import VARIABLE_ALIASES
from artistools.inputmodel import add_derived_cols_to_modeldata
from artistools.inputmodel import get_modeldata
from artistools.misc import exit_with_error
from artistools.misc import firstexisting_or_none
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
from artistools.viewertools import get_actions_by_flag
from artistools.viewertools import get_fitted_figwidthscale
from artistools.viewertools import get_helptexts
from artistools.viewertools import get_keyboard_help
from artistools.viewertools import get_line_readouts
from artistools.viewertools import get_nearest_range_start
from artistools.viewertools import get_new_figwidthscale
from artistools.viewertools import get_option_row_tokens
from artistools.viewertools import get_option_tokens
from artistools.viewertools import get_short_number
from artistools.viewertools import get_table_actions
from artistools.viewertools import make_central_splitter
from artistools.viewertools import make_flow_layout
from artistools.viewertools import make_option_table
from artistools.viewertools import make_parser
from artistools.viewertools import make_plot_area
from artistools.viewertools import make_range_slider
from artistools.viewertools import make_row_layout
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
from artistools.viewertools import run_command_step_with_warning
from artistools.viewertools import save_figure_of_command
from artistools.viewertools import set_command_text
from artistools.viewertools import set_edit_text
from artistools.viewertools import set_search_completion
from artistools.viewertools import set_spin_value
from artistools.viewertools import show_status_message
from artistools.viewertools import show_status_note
from artistools.viewertools import show_window
from artistools.viewertools import split_dpi_row
from artistools.viewertools import split_option_rows
from artistools.viewertools import start_application
from artistools.viewertools import start_play_timer

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
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

# a section of the window sets these options. Their rows stay in the command, but the option table does not show or
# offer them. The window reads the run in the format of --classicartis when it opens, thus that row also stays
SECTION_DESTS: t.Final = frozenset({
    "axis",
    "classicartis",
    "coneangle",
    "dimensionreduce",
    "figscale",
    "filtermovingavg",
    "filtersavgol",
    "hidexlabel",
    "labelfontsize",
    "legendframe",
    "nolegend",
    "notitle",
    "readonlymgi",
    "slice",
})

# the ways to select the cells of the plot, by the key of the selector of the window
GEOMETRY_MODES: t.Final = MappingProxyType({
    "all": "All the cells",
    "cells": "Selected cells (-cell)",
    "alongaxis": "Cells along an axis (-readonlymgi alongaxis)",
    "cone": "Cells in a cone around an axis (-readonlymgi cone)",
    "plane": "A plane of the model as an image (-slice)",
    "line": "A line through the model (-slice)",
    "average": "An image of the average around the z axis (-dimensionreduce 2)",
})

# the rows of the option table that a mode of the geometry sets
GEOMETRY_FLAGS: t.Final = ("-slice", "-dimensionreduce", "-readonlymgi", "-axis", "-coneangle")

# the planes of -slice through the origin, by the axis that is normal to the plane
PLANE_OF_NORMAL: t.Final = MappingProxyType({"z": "xy", "y": "xz", "x": "yz"})

# the ways to smooth each line, by the key of the selector of the window
SMOOTHING_MODES: t.Final = MappingProxyType({
    "none": "No smoothing",
    "movingavg": "Moving average (-filtermovingavg)",
    "savgol": "Savitzky-Golay filter (-filtersavgol)",
})

# the number of levels of each ion that the choices of a level population give, from the lowest NLTE level up
LEVEL_CHOICES_PER_ION: t.Final = 20

# the table of the window does not offer these options, but it shows their rows from the command. Some give a
# different action from one plot, and --verbose and --quiet change only the hidden output. Save Figure asks for the
# resolution of a PNG file (-dpi)
TABLE_EXCLUDED_DESTS: t.Final = frozenset({
    "help",
    "multiplot",
    "makegif",
    "listvariables",
    "listnuclides",
    "dpi",
    "quiet",
    "verbose",
})

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

    plotestimators chooses the bins, the markers, and the colours when the command gives no option. The last three
    fields hold the values that the plot used.
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


class RunData(t.NamedTuple):
    """The data of the run that the controls of the viewer need, which Reload Data reads again."""

    batchcaches: "list[EstimatorBatchCache] | None"
    estimatorcolumns: tuple[str, ...]
    validtimesteps: list[int]
    cells: list[int]
    cellvelocities: dict[int, float]
    defaultsubplots: tuple[tuple[str, ...], ...]


def read_run(modelpath: Path, args: argparse.Namespace, ntimesteps: int) -> RunData:
    """Read the data of the run that the controls of the viewer need.

    The viewer checks and converts the estimator caches of the run one time. Each plot then reads the caches of its
    own timesteps and cells, as the command does. The function changes no viewer, thus a worker thread can run it.
    """
    isartisrun = not args.classicartis and not path_is_codecomparison(modelpath)
    batchcaches = get_batch_caches(modelpath) if isartisrun else None
    estimators, modelmeta = join_cell_modeldata(
        estimators=scan_estimators(modelpath=modelpath, classicartis=args.classicartis, batchcaches=batchcaches),
        modelpath=modelpath,
    )
    _, estimatorcolumns = add_plot_columns(args, estimators, modelmeta)
    # a run that stopped early has no estimators for the last timesteps, and a plot of those timesteps fails
    validtimesteps = get_estimator_timesteps(modelpath, estimators) or list(range(ntimesteps))
    # ARTIS writes the estimators of each cell that holds matter
    lzmodel, modelmeta = get_modeldata(modelpath)
    dfcells = (
        add_derived_cols_to_modeldata(lzmodel, modelmeta=modelmeta)
        .filter(pl.col("rho") > 0.0)
        .select("modelgridindex", "vel_r_mid")
        .sort("modelgridindex")
        .collect()
    )
    cells: list[int] = dfcells["modelgridindex"].to_list()
    return RunData(
        batchcaches=batchcaches,
        estimatorcolumns=tuple(estimatorcolumns),
        validtimesteps=validtimesteps,
        cells=cells,
        cellvelocities=dict(zip(cells, dfcells["vel_r_mid"].to_list(), strict=True)),
        # the command omits -plot if the subplots are the default subplots of plotestimators
        defaultsubplots=tuple(
            get_plotitem_tokens(plotitems)
            for plotitems in get_default_plotlist()
            if default_plotitem_has_data(plotitems, estimatorcolumns, modelpath)
        ),
    )


def read_run_again(modelpath: Path, args: argparse.Namespace, ntimesteps: int) -> RunData:
    """Read the run again, e.g. while ARTIS writes more timesteps.

    These caches hold the files of the last read. A kept scan also holds the metadata of its file, e.g. 8 MB for a
    cache of 5335 columns, thus the scans of the replaced caches must go.
    """
    scan_parquet_file.cache_clear()
    get_runfolder_timesteps_cached.cache_clear()
    read_classic_estimators_cached.cache_clear()
    return read_run(modelpath, args, ntimesteps)


def set_run(viewer: "EstimatorViewer", run: RunData) -> None:
    """Give the viewer the data of the run."""
    viewer.batchcaches = run.batchcaches
    viewer.estimatorcolumns = run.estimatorcolumns
    viewer.validtimesteps = run.validtimesteps
    viewer.cells = run.cells
    viewer.cellvelocities = run.cellvelocities
    viewer.defaultsubplots = run.defaultsubplots


def reload_run(viewer: "EstimatorViewer", run: RunData) -> None:
    """Give the viewer the data of the run that it read again, and keep the time range inside the valid timesteps.

    A plot against time of the whole run then covers the new timesteps too.
    """
    oldvalidtimesteps = viewer.validtimesteps
    set_run(viewer, run)
    values = viewer.values
    wholerun = (values.first, values.last) == (oldvalidtimesteps[0], oldvalidtimesteps[-1])
    if is_evolution(values) and wholerun:
        viewer.values = viewer.select_timesteps(values, 0, len(viewer.validtimesteps))
    else:
        firstpos, lastpos = viewer.get_selection_positions()
        viewer.values = viewer.select_timesteps(values, firstpos, lastpos - firstpos + 1)


def get_row_values(rows: OptionRows, flag: str) -> tuple[str, ...] | None:
    """Return the values of the row of an option, or None if the rows have no such option."""
    return next((values for rowflag, values in rows if rowflag == flag), None)


def set_row_values(rows: OptionRows, changes: "Mapping[str, tuple[str, ...] | None]") -> OptionRows:
    """Return the rows with new values of some options, in their old places. None removes an option.

    A new option goes after the other rows.
    """
    changed = [
        (flag, changes.get(flag, values)) for flag, values in rows if not (flag in changes and changes[flag] is None)
    ]
    present = {flag for flag, _ in rows}
    added = [(flag, values) for flag, values in changes.items() if values is not None and flag not in present]
    return tuple((flag, values) for flag, values in (*changed, *added) if values is not None)


def get_geometry_choices(dimensions: int) -> list[str]:
    """Return the keys in GEOMETRY_MODES that a model can plot.

    A 1D or a 2D model has no axes, planes, or lines, but it has the average around the z axis.
    """
    return [mode for mode in GEOMETRY_MODES if dimensions == 3 or mode in {"all", "cells", "average"}]


def get_geometry_mode(values: "ControlValues") -> str:
    """Return the key in GEOMETRY_MODES of the selection of the cells of the values."""
    rows = values.otheroptions
    if slicevalues := get_row_values(rows, "-slice"):
        return "line" if "," in slicevalues[0] else "plane"
    if get_row_values(rows, "-dimensionreduce") == ("2",):
        return "average"
    if readonlymgi := get_row_values(rows, "-readonlymgi"):
        return readonlymgi[0]
    return "cells" if values.cells else "all"


def get_slice_parts(slicetext: str) -> tuple[str, str]:
    """Return the plane and the offset along its normal of a -slice plane, e.g. ("xy", "-0.2c") for "z=-0.2c".

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
    """Return the -slice text of a plane with an offset along its normal. An empty offset gives the plane."""
    normal = next(axis for axis, planeaxes in PLANE_OF_NORMAL.items() if planeaxes == plane)
    return f"{normal}={offset.strip()}" if offset.strip() else plane


def get_line_axis(slicetext: str) -> str:
    """Return the axis along a -slice line, which is the axis that the line gives no condition for."""
    conditionaxes = {condition.partition("=")[0].strip() for condition in slicetext.lower().split(",")}
    return next((axis for axis in "xyz" if axis not in conditionaxes), "x")


def set_geometry_mode(viewer: "EstimatorViewer", values: "ControlValues", mode: str) -> "ControlValues":
    """Return the values with a new selection of the cells.

    A mode keeps the parameters that apply to it, e.g. the axis of the cells along an axis for a cone. A colour
    image is a snapshot with a velocity on each axis. Thus a plane or the average around z takes the default x of a
    snapshot, and a plot against time becomes a snapshot.
    """
    rows = values.otheroptions
    keptaxis = get_row_values(rows, "-axis") if mode in {"alongaxis", "cone"} else None
    keptcone = get_row_values(rows, "-coneangle") if mode == "cone" else None
    oldslice = (get_row_values(rows, "-slice") or ("",))[0]
    changes: dict[str, tuple[str, ...] | None] = dict.fromkeys(GEOMETRY_FLAGS)
    changes |= {"-axis": keptaxis, "-coneangle": keptcone}
    if mode in {"alongaxis", "cone"}:
        changes["-readonlymgi"] = (mode,)
    elif mode == "plane":
        changes["-slice"] = (oldslice if oldslice and "," not in oldslice else "xy",)
    elif mode == "line":
        changes["-slice"] = (oldslice if "," in oldslice else "z=0,y=0",)
    elif mode == "average":
        changes["-dimensionreduce"] = ("2",)
    cells = (values.cells or (str(viewer.cells[0]) if viewer.cells else "")) if mode == "cells" else ""
    newrows = set_row_values(rows, changes)
    newvalues = replace_option_rows(viewer, dc.replace(values, cells=cells), newrows)
    if mode in {"plane", "average"}:
        return viewer.set_xvariable(newvalues, viewer.get_default_xvariable(newrows, timegiven=True))
    return newvalues


def get_smoothing(rows: OptionRows) -> tuple[str, tuple[int, ...]]:
    """Return the key in SMOOTHING_MODES of the smoothing of the rows, and its numbers."""
    if (savgol := get_row_values(rows, "-filtersavgol")) and all(value.lstrip("-").isdecimal() for value in savgol):
        return "savgol", tuple(int(value) for value in savgol)
    movingavg = get_row_values(rows, "-filtermovingavg")
    if movingavg and movingavg[0].isdecimal() and int(movingavg[0]) > 1:
        return "movingavg", (int(movingavg[0]),)
    return "none", ()


def set_smoothing(rows: OptionRows, mode: str, numbers: "Sequence[int]") -> OptionRows:
    """Return the rows with a new smoothing, e.g. "savgol" with the window length 5 and the order 2."""
    return set_row_values(
        rows,
        {
            "-filtermovingavg": (str(numbers[0]),) if mode == "movingavg" else None,
            "-filtersavgol": tuple(str(number) for number in numbers[:2]) if mode == "savgol" else None,
        },
    )


def has_nlte_populations(modelpath: Path) -> bool:
    """Return True if the run wrote NLTE populations, which a plot of level populations reads.

    The search also reads the run folders of the model, e.g. 12345.slurm.
    """
    return firstexisting_or_none("nlte_0000.out", folder=modelpath, tryzipped=True) is not None


def get_level_names(modelpath: Path, timestep: int, cell: int) -> list[str]:
    """Return the names of the lowest NLTE levels of each ion for a plot of level populations, e.g. "Fe II 0".

    Each cell has the same NLTE levels, thus one cell gives them. A timestep before the first NLTE timestep has no
    populations, and all the timesteps of the cell then give the levels.
    """
    from artistools.nltepops import read_nltepops

    dfpops = read_nltepops(modelpath, timestep=timestep, modelgridindex=cell)
    if dfpops.is_empty():
        dfpops = read_nltepops(modelpath, modelgridindex=cell)
    # a negative level is the superlevel of an ion, which is not one level
    dflevels = (
        dfpops
        .filter(pl.col("level") >= 0)
        .select("Z", "ion_stage", "level")
        .unique()
        .sort("Z", "ion_stage", "level")
        .group_by("Z", "ion_stage", maintain_order=True)
        .head(LEVEL_CHOICES_PER_ION)
    )
    return [
        f"{get_ionstring(atomic_number, ion_stage)} {level}"
        for atomic_number, ion_stage, level in zip(dflevels["Z"], dflevels["ion_stage"], dflevels["level"], strict=True)
    ]


def cells_apply(values: ControlValues) -> bool:
    """Return True if -cell selects the cells of the plot of the values.

    -slice and -dimensionreduce 2 select the cells, thus plotestimators rejects -cell with them. -readonlymgi replaces
    -cell with the cells along an axis or in a cone.
    """
    return get_geometry_mode(values) in {"all", "cells"}


def get_batch_caches(modelpath: Path) -> "list[EstimatorBatchCache]":
    """Return the current parquet caches of the batches of the run, and convert the stale batches first.

    A large run has more than 10 GB of estimators. The conversion of 40 batches of a 3D kilonova run kept 5.2 GB of
    freed memory in the viewer process. A child process gives that memory back to the system when it ends. The
    child process imports artistools again, which took 0.4 s, thus the viewer process reads the caches itself when no
    batch is stale.
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

    # read_run reads these from the run, and set_run gives them to the viewer
    batchcaches: "list[EstimatorBatchCache] | None"
    estimatorcolumns: tuple[str, ...]
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
        set_run(self, read_run(self.modelpath, args, len(self.tmids)))
        # a section of the window sets the rows of these flags, and the option table does not show them
        self.sectionflags = frozenset(
            flag for flag, action in get_actions_by_flag(parser).items() if action.dest in SECTION_DESTS
        )
        self.dimensions = int(get_modeldata(self.modelpath)[1]["dimensions"])
        # the types of series that the run can plot from its NLTE populations. Only a 1D model gives the width in
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
        givensubplots = tuple(resolve_aliases(str(item) for item in plotitems) for plotitems in args.plotlist or ())
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

        subplots, otheroptions = move_poptype_to_subplots(subplots, otheroptions, self.estimatorcolumns)
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

        A plot of all the cells, or of a list of cells, moves to the first or the last cell. A plot whose cells -cell
        does not select gives None.
        """
        if not self.cells or not cells_apply(self.values):
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
        because a snapshot of the one cell of a plot against time has one point.
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

        message, warning = run_command_step_with_warning(make_plot, quiet=quiet)

        def show_plot() -> str | None:
            self.warning = warning
            if message is not None:
                return message
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
    directive = name.removeprefix("_").lower()
    return directive if equals and directive in DIRECTIVES else None


def replace_directives(subplot: "Sequence[str]", directives: "Mapping[str, str | None]") -> tuple[str, ...]:
    """Return the items of a subplot with these directives in place of their old values. None removes a directive."""
    kept = [item for item in subplot if get_item_directive(item) not in directives]
    return (*kept, *(f"{name}={value}" for name, value in directives.items() if value is not None))


def resolve_aliases(items: "Iterable[str]") -> tuple[str, ...]:
    """Return the items of a subplot with the name of each variable that has an alias, e.g. nne for n_e.

    plotestimators reads the alias as the variable, and the controls of the window follow the variable.
    """
    return tuple(VARIABLE_ALIASES.get(item, item) for item in items)


# the variables that suit a new subplot, in the order of the suggestions
COMMON_VARIABLES: t.Final = ("Te", "TR", "nne", "rho", "heating_dep", "total_dep", "TJ", "W")

# the columns that give the grid, the time, or the size of a cell, and not the physics of the cell
BOOKKEEPING_COLUMNS: t.Final = frozenset({
    "deltavol_deltat",
    "inputcellid",
    "mass_g",
    "modelgridindex",
    "tdays",
    "thick",
    "timestep",
    "titeration",
    "tmid_days",
    "tmid_days_prevtimestep",
    "twidth_days",
    "volume_prevtimestep",
})

# the families of the estimator columns that give the species of each type of series
SPECIES_FAMILIES: t.Final = MappingProxyType({
    "populations": ("nnelement", "nnion", "nniso"),
    "averageionisation": ("nnelement",),
    "averageexcitation": ("nnion",),
    "initabundances": ("init_X",),
    "initmasses": ("init_X",),
})

# the types of series that read the NLTE populations of the run, and not the estimators alone
NLTE_SERIESTYPES: t.Final = frozenset({"averageexcitation", "levelpopulation", "levelpopulation_dn_on_dvel"})

# the number of suggestions of the card of a subplot, and of the new subplots
SERIES_SUGGESTION_COUNT: t.Final = 4
NEW_SUBPLOT_SUGGESTION_COUNT: t.Final = 5


def get_subplot_names(subplot: "Sequence[str]") -> list[str]:
    """Return the items of a subplot that are not a directive, e.g. the variables or the series type and its names."""
    return [item for item in subplot if get_item_directive(item) is None]


def get_subplot_seriestype(subplot: "Sequence[str]", estimatorcolumns: "Collection[str]") -> str | None:
    """Return the type of series of a subplot, e.g. "populations", or None for a subplot of variables.

    plotestimators reads a list of ions with no type as a plot of populations, thus this function does the same.
    """
    names = get_subplot_names(subplot)
    if not names:
        return None
    if is_seriestype(names[0], estimatorcolumns) or is_ionseriestype(names[0], estimatorcolumns, names[1:]):
        return names[0]
    if names[0] not in estimatorcolumns and all(is_valid_ion(name) for name in names):
        return "populations"
    return None


def get_species_sortkey(species: str) -> tuple[int, int, int, str]:
    """Return the key that sorts species by element, then the element before its ions, then the ions by stage."""
    return get_iontuple_sortkey(get_iontuple(species))


@lru_cache(maxsize=4)
def get_species_of_families(estimatorcolumns: tuple[str, ...]) -> "Mapping[str, tuple[str, ...]]":
    """Return the species of each family of estimator columns in sorted order, e.g. ("Fe II", "Fe III") for nnion.

    A large model has more than 5000 columns, and each show of the subplot cards reads the species of many types.
    """
    species: dict[str, set[str]] = {}
    for column in estimatorcolumns:
        if (split := split_species_suffix(column)) is not None:
            species.setdefault(split[0], set()).add(split[1])
    return MappingProxyType({
        family: tuple(sorted(names, key=get_species_sortkey)) for family, names in species.items()
    })


def get_species_choices(
    seriestype: str, estimatorcolumns: "Collection[str]", levelnames: "Sequence[str]" = ()
) -> list[str]:
    """Return each species that a type of series can plot for this model, e.g. "Fe II" for the populations.

    An ion series such as gamma_NT takes the species of its own columns, e.g. gamma_NT_Fe_II. A level population
    takes levelnames, e.g. "Fe II 0", which get_level_names gives.
    """
    if seriestype.startswith("levelpopulation"):
        return list(levelnames)
    speciesoffamilies = get_species_of_families(tuple(estimatorcolumns))
    families = SPECIES_FAMILIES.get(seriestype, (seriestype,))
    species = {name for family in families for name in speciesoffamilies.get(family, ())}
    return sorted(species, key=get_species_sortkey)


def get_first_choice(choices: "Sequence[str]") -> list[str]:
    """Return the first ion of the choices, or the first choice if no choice is an ion, e.g. Fe I before Fe."""
    ions = [choice for choice in choices if isinstance(get_iontuple(choice)[1], int)]
    return (ions or list(choices))[:1]


def is_suggested_variable(column: str) -> bool:
    """Return True if a column can be a series of its own, and not the grid, the time, or one species."""
    return (
        column not in BOOKKEEPING_COLUMNS
        and not column.startswith(("vel_", "init_pos_", "init_kinetic_"))
        and split_species_suffix(column) is None
    )


@lru_cache(maxsize=4)
def get_variable_choices(estimatorcolumns: tuple[str, ...]) -> tuple[str, ...]:
    """Return the columns that a subplot of variables can take, with the columns that suit a series first."""
    return tuple(sorted(estimatorcolumns, key=lambda column: not is_suggested_variable(column)))


@lru_cache(maxsize=4)
def get_variables_of_ylabels(estimatorcolumns: tuple[str, ...]) -> "Mapping[str, tuple[str, ...]]":
    """Return the columns that suit a series of their own, by the label of their y axis."""
    variables: dict[str, list[str]] = {}
    for column in estimatorcolumns:
        if is_suggested_variable(column):
            variables.setdefault(get_ylabel(column).strip(), []).append(column)
    return MappingProxyType({ylabel: tuple(columns) for ylabel, columns in variables.items()})


def get_series_suggestions(
    subplot: "Sequence[str]", estimatorcolumns: "Collection[str]", levelnames: "Sequence[str]" = ()
) -> list[str]:
    """Return the items that suit a subplot next, e.g. TR beside Te, or Fe IV beside Fe II and Fe III.

    A series type takes more of its species, first those of the elements that the subplot has. A level population
    takes the next levels, first those of the ions that the subplot has. A variable takes the other variables with
    the same quantity on the y axis.
    """
    names = get_subplot_names(subplot)
    if not names:
        return []
    seriestype = get_subplot_seriestype(subplot, estimatorcolumns)
    if seriestype is not None and seriestype.startswith("levelpopulation"):
        ions = {name.rpartition(" ")[0] for name in names}
        choices = [name for name in levelnames if name not in names]
        return sorted(choices, key=lambda name: name.rpartition(" ")[0] not in ions)[:SERIES_SUGGESTION_COUNT]
    if seriestype is not None:
        elements = {get_iontuple(name)[0] for name in names if is_valid_ion(name)}

        # the ions of the elements of the subplot come first, and the total of an element comes after its ions
        def get_choice_sortkey(species: str) -> tuple[bool, bool, tuple[int, int, int, str]]:
            atomic_number, ion_stage = get_iontuple(species)
            return atomic_number not in elements, not isinstance(ion_stage, int), get_species_sortkey(species)

        choices = [name for name in get_species_choices(seriestype, estimatorcolumns) if name not in names]
        suggestions = sorted(choices, key=get_choice_sortkey)
    else:
        ylabel = get_ylabel(names[0]).strip()
        variables = get_variables_of_ylabels(tuple(estimatorcolumns)).get(ylabel, ()) if ylabel else ()
        suggestions = [column for column in variables if column not in names]
    return suggestions[:SERIES_SUGGESTION_COUNT]


def get_new_subplot_suggestions(
    subplots: "Sequence[Sequence[str]]",
    defaultsubplots: "Sequence[tuple[str, ...]]",
    estimatorcolumns: "Collection[str]",
) -> list[tuple[str, ...]]:
    """Return the subplots that suit the plot next.

    The default subplots that the plot has not come first. Then come the first common variables that no subplot
    shows, a plot of the populations and of the average ionisation if the plot has none, and the other variables.
    """
    plotted = {name for subplot in subplots for name in get_subplot_names(subplot)}
    seriestypes = {get_subplot_seriestype(subplot, estimatorcolumns) for subplot in subplots}
    variables = [(name,) for name in COMMON_VARIABLES if name in estimatorcolumns and name not in plotted]
    series: list[tuple[str, ...]] = []
    for seriestype in ("populations", "averageionisation"):
        if seriestype not in seriestypes and (choices := get_species_choices(seriestype, estimatorcolumns)):
            # the ions of the first element, e.g. Fe II and Fe III, and not the element with its first ion
            firstelement = get_iontuple(choices[0])[0]
            names = [name for name in choices if get_iontuple(name)[0] == firstelement and name != choices[0]]
            series.append((seriestype, *(names or choices)[:2]))
    suggestions = [
        *(subplot for subplot in defaultsubplots if subplot not in subplots),
        *variables[:2],
        *series,
        *variables[2:],
    ]
    return list(dict.fromkeys(suggestions))[:NEW_SUBPLOT_SUGGESTION_COUNT]


def group_species_names(words: "Sequence[str]", choices: "Collection[str]") -> list[str]:
    """Return the names of species that the words give, e.g. "Fe II" and "Fe III" for the words Fe II Fe III.

    A name can have more than one word, e.g. "Fe II 0" for a level, thus the longest name in the choices goes first.
    A name that is not in the choices keeps an ion stage with its element, e.g. "Zn II".
    """
    names: list[str] = []
    position = 0
    while position < len(words):
        pairisanion = position + 1 < len(words) and is_valid_ion(" ".join(words[position : position + 2]))
        end = next(
            (end for end in range(len(words), position, -1) if " ".join(words[position:end]) in choices),
            position + (2 if pairisanion else 1),
        )
        names.append(" ".join(words[position:end]))
        position = end
    return names


def make_new_subplot(
    text: str, estimatorcolumns: "Collection[str]", levelnames: "Sequence[str]" = ()
) -> tuple[str, ...]:
    """Return the items of a new subplot that the user typed, e.g. the text of a suggestion.

    Quotes keep a name with spaces together, e.g. populations 'Fe II'. The names of a series type can also have
    spaces and no quotes, e.g. populations Fe II Fe III. A series type with no name takes its first species, because
    plotestimators needs at least one. A directive such as yscale=log goes after the names.
    """
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    words = resolve_aliases(token for token in tokens if get_item_directive(token) is None)
    directives = tuple(token for token in tokens if get_item_directive(token) is not None)
    if not words:
        return ()
    first = words[0]
    choices = get_species_choices(first, estimatorcolumns, levelnames)
    names = group_species_names(words[1:], set(choices))
    # a family of ion columns can also be a variable of its own, e.g. cooling_coll, and then its ions decide
    isvariable = first in estimatorcolumns
    if (not isvariable and (choices or is_seriestype(first, estimatorcolumns))) or (
        names and is_ionseriestype(first, estimatorcolumns, names)
    ):
        return (first, *(names or get_first_choice(choices)), *directives)
    ions = group_species_names(words, set(get_species_choices("populations", estimatorcolumns)))
    if first not in estimatorcolumns and all(is_valid_ion(ion) for ion in ions):
        return (*ions, *directives)
    return (*words, *directives)


# the name of the type of a subplot of variables in the type selector of a subplot, which gives no -plot token
VARIABLES_TYPE: t.Final = "variables"

# the help text of each type of subplot that the selector names
SUBPLOT_TYPE_HELPTEXTS: t.Final = MappingProxyType({
    VARIABLES_TYPE: "Estimator variables with the same quantity, e.g. Te and TR",
    "populations": "The population of each ion, element, or isotope",
    "averageionisation": "The mean ion charge of each element",
    "averageexcitation": "The mean excitation energy of each ion",
    "initabundances": "The initial mass fraction of each element or isotope",
    "initmasses": "The initial mass of each element or isotope",
    "levelpopulation": "The NLTE population of each level, e.g. Fe II 0",
    "levelpopulation_dn_on_dvel": "The NLTE population of each level for each unit of velocity",
})


def get_subplot_types(estimatorcolumns: "Collection[str]", nltetypes: "Collection[str]" = ()) -> list[str]:
    """Return the types of subplot that the model can plot: the variables, the series types, and the ion series.

    An ion series is a family of columns with one column for each ion, e.g. gamma_NT_Fe_II. The families of the
    other series types, e.g. nnion for the populations, are not a type of their own. nltetypes gives the types in
    NLTE_SERIESTYPES that the run can plot, because they need its NLTE populations.
    """
    columns = tuple(estimatorcolumns)
    speciestypes = [
        seriestype
        for seriestype in SPECIES_FAMILIES
        if (seriestype not in NLTE_SERIESTYPES or seriestype in nltetypes) and get_species_choices(seriestype, columns)
    ]
    leveltypes = [seriestype for seriestype in nltetypes if seriestype.startswith("levelpopulation")]
    otherfamilies = {family for families in SPECIES_FAMILIES.values() for family in families}
    ionfamilies = [family for family in get_species_of_families(columns) if family not in otherfamilies]
    return [VARIABLES_TYPE, *speciestypes, *leveltypes, *sorted(ionfamilies, key=str.lower)]


def change_subplot_type(
    subplot: "Sequence[str]", seriestype: str, estimatorcolumns: "Collection[str]", levelnames: "Sequence[str]" = ()
) -> tuple[str, ...]:
    """Return the subplot with a new type, and keep the names and the directives that still apply.

    A new type with no name that applies takes its first choice, e.g. the first ion of the populations, because
    plotestimators needs at least one name. The new type can plot a different quantity, thus ymin= and ymax= go.
    Only a plot of populations takes ionpoptype=. The same type keeps the subplot.
    """
    oldtype = get_subplot_seriestype(subplot, estimatorcolumns)
    if seriestype == (oldtype or VARIABLES_TYPE):
        return tuple(subplot)
    keptdirectives = {"yscale", "ionpoptype"} if seriestype == "populations" else {"yscale"}
    directives = [item for item in subplot if get_item_directive(item) in keptdirectives]
    names = get_subplot_names(subplot)
    if oldtype is not None and names and names[0] == oldtype:
        names = names[1:]
    if seriestype == VARIABLES_TYPE:
        variables = [name for name in names if oldtype is None and name in estimatorcolumns]
        common = [name for name in COMMON_VARIABLES if name in estimatorcolumns]
        return (*(variables or common[:1] or ["Te"]), *directives)
    choices = get_species_choices(seriestype, estimatorcolumns, levelnames)
    kept = [name for name in names if name in choices]
    return (seriestype, *(kept or get_first_choice(choices)), *directives)


def get_chip_items(subplot: "Sequence[str]", estimatorcolumns: "Collection[str]") -> list[tuple[int, str]]:
    """Return the position and the text of each item of a subplot that shows as a chip.

    The type selector shows the series type, and a control of the card sets each directive, thus they show no chip.
    """
    seriestype = get_subplot_seriestype(subplot, estimatorcolumns)
    typeposition = next((position for position, item in enumerate(subplot) if item == seriestype), None)
    return [
        (position, item)
        for position, item in enumerate(subplot)
        if position != typeposition and get_item_directive(item) is None
    ]


def get_directive_value(subplot: "Sequence[str]", directive: str) -> str | None:
    """Return the value of a directive of a subplot, e.g. "log" for yscale=log, or None if the subplot has none."""
    return next((item.partition("=")[2] for item in reversed(subplot) if get_item_directive(item) == directive), None)


# the quantity of the ions of a populations subplot with no ionpoptype=, which is the default of -ionpoptype
DEFAULT_POPTYPE: t.Final = "absolute"


def move_poptype_to_subplots(
    subplots: "Sequence[tuple[str, ...]]", otheroptions: OptionRows, estimatorcolumns: "Collection[str]"
) -> tuple[tuple[tuple[str, ...], ...], OptionRows]:
    """Return the subplots and the rows of the option table with -ionpoptype in each populations subplot.

    The window sets the quantity of the ions for each populations subplot (ionpoptype=), thus -ionpoptype of the
    command goes to each populations subplot that has no ionpoptype= of its own. With no populations subplot, the row
    stays, and a populations subplot that the user adds later takes it.
    """
    poptype = next((values[0] for flag, values in otheroptions if flag == "-ionpoptype" and values), None)
    ispopulations = [get_subplot_seriestype(subplot, estimatorcolumns) == "populations" for subplot in subplots]
    if poptype is None or not any(ispopulations):
        return tuple(subplots), otheroptions
    rows = tuple((flag, values) for flag, values in otheroptions if flag != "-ionpoptype")
    return tuple(
        (*subplot, f"ionpoptype={poptype}")
        if populations and poptype != DEFAULT_POPTYPE and get_directive_value(subplot, "ionpoptype") is None
        else subplot
        for subplot, populations in zip(subplots, ispopulations, strict=True)
    ), rows


def remove_subplot_item(
    subplots: "Sequence[tuple[str, ...]]", row: int, index: int, estimatorcolumns: "Collection[str]"
) -> tuple[tuple[str, ...], ...]:
    """Return the subplots without one item of a subplot.

    A subplot with no name left, or with its series type alone, has nothing to plot, thus it goes with its
    directives.
    """
    subplot = subplots[row][:index] + subplots[row][index + 1 :]
    names = get_subplot_names(subplot)
    seriestypeonly = (
        len(names) == 1
        and names[0] not in estimatorcolumns
        and (is_seriestype(names[0], estimatorcolumns) or bool(get_species_choices(names[0], estimatorcolumns)))
    )
    keep = bool(names) and not seriestypeonly
    return (
        tuple(item for position, item in enumerate(subplots) if position != row)
        if not keep
        else (*subplots[:row], subplot, *subplots[row + 1 :])
    )


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
    snapshotx = viewer.get_default_xvariable(viewer.values.otheroptions, timegiven=True)
    return viewer.select_timesteps(viewer.set_xvariable(viewer.values, snapshotx), position, 1)


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


def get_card_key(
    row: int, subplots: "Sequence[Sequence[str]]", estimatorcolumns: "Sequence[str]"
) -> tuple[object, ...]:
    """Return the parts of the plot that the widgets of the card of a subplot show.

    A directive, e.g. ymin=, changes only the text of a control of the card, thus the card stays. The keyboard focus
    then stays in the control, e.g. in the field of the y maximum after an edit of the y minimum. A chip removes the
    item at its position, thus the key holds the position of each name.
    """
    names = tuple((position, item) for position, item in enumerate(subplots[row]) if get_item_directive(item) is None)
    return (row, row == len(subplots) - 1, names, estimatorcolumns)


class SubplotCard(t.NamedTuple):
    """The card of a subplot in the window, with the controls that show the directives of the subplot."""

    key: tuple[object, ...]
    frame: "QtWidgets.QFrame"
    yscalebox: "QtWidgets.QComboBox"
    poptypebox: "QtWidgets.QComboBox | None"
    yminedit: "QtWidgets.QLineEdit"
    ymaxedit: "QtWidgets.QLineEdit"


# the style of the cards, the chips, and the suggestions. One style sheet for all the cards costs less time than a
# style sheet for each widget, which took about 10 ms for each show of the cards
SUBPLOT_STYLE_SHEET: t.Final = (
    "QFrame#subplotcard { border: 1px solid palette(mid); border-radius: 6px; }"
    " QFrame#chip { border: 1px solid palette(mid); border-radius: 10px; background: palette(base); }"
    " QToolButton#suggestion { border: 1px dashed palette(mid); border-radius: 10px; padding: 1px 8px; }"
    " QToolButton#suggestion:hover { border-style: solid; }"
)


def make_completer(names: "Sequence[str]", parent: "QtWidgets.QWidget") -> "QtWidgets.QCompleter":
    """Return a completer that finds each name that holds the typed text, e.g. "ion" finds averageionisation."""
    from PySide6 import QtWidgets

    completer = QtWidgets.QCompleter(list(names), parent)
    set_search_completion(completer)
    return completer


def make_chip(text: str, tooltip: str, on_remove: "Callable[[], None]") -> "QtWidgets.QFrame":
    """Return a chip that shows one item of a subplot, with a button that removes the item.

    SUBPLOT_STYLE_SHEET gives the chip its border.
    """
    from PySide6 import QtWidgets

    chip = QtWidgets.QFrame()
    chip.setObjectName("chip")
    layout = QtWidgets.QHBoxLayout(chip)
    layout.setContentsMargins(8, 0, 0, 0)
    layout.setSpacing(0)
    label = QtWidgets.QLabel(text)
    label.setToolTip(tooltip)
    removebutton = QtWidgets.QToolButton()
    removebutton.setText("✕")
    removebutton.setAutoRaise(True)
    removebutton.setToolTip(f"Remove {text} from the subplot")
    removebutton.clicked.connect(on_remove)
    layout.addWidget(label)
    layout.addWidget(removebutton)
    return chip


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
    ("<b>Left</b>, <b>Right</b>", "Move the time range to the adjacent timestep"),
    ("<b>Up</b>, <b>Down</b>", "Make the time range one timestep wider or narrower"),
    ("<b>Home</b>, <b>End</b>", "Move the time range to the start or the end of the run"),
    ("<b>Page Up</b>, <b>Page Down</b>", "Select the previous or the next cell"),
    (
        "<b>Space</b>",
        "Play or pause. A snapshot moves through the timesteps, and a plot against time moves through the cells",
    ),
    ("<b>Drag</b> across a plot", "Select the x range"),
    ("<b>Shift-drag</b> up or down a subplot", "Select the y range of the subplot (ymin= and ymax=)"),
    ("<b>Double-click</b> a plot", "Show the x range of the data"),
    ("<b>Right-click</b> a subplot", "Show the menu of the subplot, e.g. the y scale"),
)


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
    make_central_splitter(window, plotarea, sidebar)
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
    geometrybox = QtWidgets.QComboBox()
    for mode in get_geometry_choices(viewer.dimensions):
        geometrybox.addItem(GEOMETRY_MODES[mode], mode)
    geometrybox.setToolTip(
        "The cells that the plot reads. A plane and the average around the z axis give a colour image of a snapshot."
    )
    cellslider = make_slider()
    cellslider.setToolTip("Select one cell. The Page Up key and the Page Down key select the adjacent cell.")
    celledit = QtWidgets.QLineEdit()
    celledit.setFixedWidth(110)
    celledit.setPlaceholderText("e.g. 3 or 3-7")
    celledit.setToolTip(helptexts.get("modelgridindex", ""))
    celllabel = QtWidgets.QLabel()
    cellnamelabel = QtWidgets.QLabel("-cell")
    axisbox = QtWidgets.QComboBox()
    axisbox.addItems(["+x", "-x", "+y", "-y", "+z", "-z"])
    axisbox.setToolTip(helptexts.get("axis", ""))
    coneanglelabel = QtWidgets.QLabel("Angle")
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
    lineaxisbox.addItems(["x", "y", "z"])
    lineaxisbox.setToolTip("The axis of the line through the origin")

    def make_parameter_row(widgets: "Sequence[QtWidgets.QWidget]") -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        rowlayout = make_row_layout(widgets)
        rowlayout.setContentsMargins(0, 0, 0, 0)
        row.setLayout(rowlayout)
        return row

    axisparameters = make_parameter_row([QtWidgets.QLabel("-axis"), axisbox, coneanglelabel, coneanglebox])
    planeparameters = make_parameter_row([QtWidgets.QLabel("Plane"), planebox, QtWidgets.QLabel("at"), offsetedit])
    lineparameters = make_parameter_row([QtWidgets.QLabel("Line along"), lineaxisbox])
    add_row(cellgrid, 0, [geometrybox])
    cellgrid.addWidget(cellnamelabel, 1, 0)
    cellgrid.addWidget(cellslider, 1, 1)
    cellgrid.addWidget(celledit, 1, 2)
    cellgrid.addWidget(celllabel, 2, 0, 1, -1)
    for row, widget in enumerate((axisparameters, planeparameters, lineparameters), start=3):
        cellgrid.addWidget(widget, row, 0, 1, -1)

    _, xgrid = add_section(panellayout, "Horizontal axis")
    xbox = QtWidgets.QComboBox()
    xbox.setEditable(True)
    xbox.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
    xbox.addItems([*XVARIABLES, *(column for column in viewer.estimatorcolumns if column not in XVARIABLES)])
    if (completer := xbox.completer()) is not None:
        set_search_completion(completer)
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
    smoothingbox = QtWidgets.QComboBox()
    for mode, modetext in SMOOTHING_MODES.items():
        smoothingbox.addItem(modetext, mode)
    smoothingbox.setToolTip("Smooth the line of each series")
    smoothinglengthlabel, smoothingorderlabel = QtWidgets.QLabel("Length"), QtWidgets.QLabel("Order")
    smoothinglengthbox, smoothingorderbox = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
    # show_blocked_values gives the length its range and its step for the mode
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
            QtWidgets.QLabel("Smoothing"),
            smoothingbox,
            smoothinglengthlabel,
            smoothinglengthbox,
            smoothingorderlabel,
            smoothingorderbox,
        ],
    )

    _, subplotgrid = add_section(panellayout, "Subplots")
    # show_subplots makes the card of a subplot again when the names of the subplot change
    subplotsbox = QtWidgets.QWidget()
    subplotsbox.setStyleSheet(SUBPLOT_STYLE_SHEET)
    subplotslayout = QtWidgets.QVBoxLayout(subplotsbox)
    subplotslayout.setContentsMargins(0, 0, 0, 0)
    subplotslayout.setSpacing(6)
    newsubplotedit = QtWidgets.QLineEdit()
    newsubplotedit.setPlaceholderText("New subplot, e.g. nne, or populations Fe II")
    newsubplotedit.setToolTip(
        "Type a variable, a type of series and its names, or an ion. Press Return to add the subplot. Part of a name"
        " shows the names that hold it."
    )
    addsubplotbutton = QtWidgets.QPushButton("Add subplot")
    addsubplotbutton.setToolTip("Add a subplot of the text in the field")
    defaultbutton = QtWidgets.QPushButton("Default")
    defaultbutton.setToolTip("Show the default subplots of plotestimators, which the command gives with no -plot")
    newsubplotrow = QtWidgets.QHBoxLayout()
    newsubplotrow.addWidget(newsubplotedit, 1)
    newsubplotrow.addWidget(addsubplotbutton)
    newsubplotrow.addWidget(defaultbutton)
    newsuggestionsbox = QtWidgets.QWidget()
    newsuggestionsbox.setStyleSheet(SUBPLOT_STYLE_SHEET)
    newsuggestionslayout = make_flow_layout()
    newsuggestionsbox.setLayout(newsuggestionslayout)
    subplotgrid.addWidget(subplotsbox, 0, 0, 1, -1)
    subplotgrid.addLayout(newsubplotrow, 1, 0, 1, -1)
    subplotgrid.addWidget(newsuggestionsbox, 2, 0, 1, -1)

    _, appearancegrid = add_section(panellayout, "Appearance")
    actionsbyflag = get_actions_by_flag(viewer.parser)
    # a flag that this version of plotestimators does not have gives no control
    appearancechecks = {
        flag: QtWidgets.QCheckBox(flag)
        for flag in ("--notitle", "--nolegend", "--legendframe", "--hidexlabel")
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
    figscalebox = QtWidgets.QDoubleSpinBox()
    figscalebox.setRange(0.1, 10.0)
    figscalebox.setSingleStep(0.1)
    figscalebox.setDecimals(2)
    figscalebox.setToolTip(helptexts.get("figscale", ""))
    for box in (fontsizebox, figscalebox):
        box.setKeyboardTracking(False)
    add_row(appearancegrid, 0, list(appearancechecks.values()))
    add_row(
        appearancegrid, 1, [QtWidgets.QLabel("-labelfontsize"), fontsizebox, QtWidgets.QLabel("-figscale"), figscalebox]
    )

    optionheader, optiongrid = add_section(panellayout, "Other options")
    optioncontent = optiongrid.parentWidget()
    assert optioncontent is not None
    # the sections of the window set each option that the table offered, thus the table shows only the rows of
    # an option that the command gives and no section sets, e.g. of a new option of plotestimators
    tableoffers = bool(
        get_table_actions(viewer.parser, CONTROLLED_DESTS | OUTPUT_DESTS | TABLE_EXCLUDED_DESTS | SECTION_DESTS)
    )

    def get_table_rows(rows: OptionRows) -> OptionRows:
        """Return the rows that the option table shows, which are the rows that no section sets."""
        return tuple(row for row in rows if row[0] not in viewer.sectionflags)

    def on_option_rows(rows: OptionRows) -> None:
        sectionrows = tuple(row for row in viewer.values.otheroptions if row[0] in viewer.sectionflags)
        queue.apply(replace_option_rows(viewer, viewer.values, (*rows, *sectionrows)))

    optiontable, set_option_rows = make_option_table(
        window,
        viewer.parser,
        CONTROLLED_DESTS | OUTPUT_DESTS | TABLE_EXCLUDED_DESTS | SECTION_DESTS,
        get_table_rows(viewer.values.otheroptions),
        on_option_rows,
    )
    optiongrid.addWidget(optiontable, 0, 0, 1, 2)
    commandtext, copybutton = add_command_section(panellayout)
    statusbar = make_status_bar(window)
    # the first plot came before the status bar, and a user of the application sees no terminal
    show_status_message(statusbar, None, viewer.warning)

    signalwidgets: list[QtWidgets.QWidget] = [
        timeslider,
        widthslider,
        cellslider,
        xbox,
        markerscheck,
        colorbyioncheck,
        geometrybox,
        axisbox,
        coneanglebox,
        planebox,
        lineaxisbox,
        smoothingbox,
        smoothinglengthbox,
        smoothingorderbox,
        fontsizebox,
        figscalebox,
        *appearancechecks.values(),
    ]

    def set_ranges() -> None:
        """Give the sliders the number of valid timesteps and the number of cells of the run.

        A shorter range clamps the value of a slider. The handler of the slider must not change the values of the
        viewer then, and show_values gives each slider its value.
        """
        nvalid = len(viewer.validtimesteps)
        blockers = [QtCore.QSignalBlocker(slider) for slider in (timeslider, widthslider, cellslider)]
        try:
            timeslider.setRange(0, nvalid - 1)
            widthslider.setRange(1, nvalid)
            cellslider.setRange(0, max(len(viewer.cells) - 1, 0))
        finally:
            for blocker in blockers:
                blocker.unblock()
        set_trange_steps(max(nvalid - 1, 1))

    set_ranges()

    # the cards of the subplots on the screen, and the key of the suggestions of a new subplot
    cards: list[SubplotCard] = []
    shownnewkey: tuple[object, ...] = ()
    # the row of a card and the name of its control that takes the keyboard focus after the next show
    pendingfocus: tuple[int, str] | None = None

    def make_suggestion_button(text: str, tooltip: str, callback: "Callable[[], None]") -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setObjectName("suggestion")
        button.setText(f"+ {text}")
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def set_box_text(box: QtWidgets.QComboBox, text: str) -> None:
        # a directive of the command can give a value that is not a choice, e.g. yscale=symlog
        if box.findText(text) < 0:
            box.addItem(text)
        box.setCurrentText(text)

    def make_selector(
        name: str, choices: "Sequence[str]", current: str, tooltip: str, callback: "Callable[[str], None]"
    ) -> QtWidgets.QComboBox:
        box = QtWidgets.QComboBox()
        box.setObjectName(name)
        box.addItems(list(choices))
        set_box_text(box, current)
        box.setToolTip(tooltip)
        box.textActivated.connect(callback)
        return box

    def get_yscale_choice(subplot: "Sequence[str]") -> str:
        value = get_directive_value(subplot, "yscale") or "auto"
        return {"lin": "linear"}.get(value, value)

    def make_subplot_card(
        row: int, subplot: tuple[str, ...], subplottypes: "Sequence[str]", key: tuple[object, ...]
    ) -> SubplotCard:
        """Return the card of a subplot. The controls of the card follow the type of the subplot.

        The name of the object of a control gives its role. show_subplots gives the focus to the control with the same
        role in a new card.
        """
        columns = viewer.estimatorcolumns
        seriestype = get_subplot_seriestype(subplot, columns)
        currenttype = seriestype or VARIABLES_TYPE
        names = get_subplot_names(subplot)
        card = QtWidgets.QFrame()
        card.setObjectName("subplotcard")
        cardlayout = QtWidgets.QVBoxLayout(card)
        cardlayout.setContentsMargins(6, 4, 4, 6)
        cardlayout.setSpacing(4)

        typebox = QtWidgets.QComboBox()
        typebox.setObjectName("type")
        types = [*subplottypes, *([] if currenttype in subplottypes else [currenttype])]
        typebox.addItems(types)
        for index, name in enumerate(types):
            helptext = SUBPLOT_TYPE_HELPTEXTS.get(name, f"The {name} columns of each ion")
            typebox.setItemData(index, helptext, QtCore.Qt.ItemDataRole.ToolTipRole)
        typebox.setCurrentText(currenttype)
        typebox.setToolTip("The type of the subplot. A new type keeps the names that still apply.")
        typebox.textActivated.connect(partial(on_subplot_type, row))
        quantity = QtWidgets.QLabel(plain_label(get_ylabel(names[0])).strip() if seriestype is None and names else "")
        quantity.setEnabled(False)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel(f"<b>{row + 1}</b>"))
        header.addWidget(typebox)
        header.addWidget(quantity, 1)
        for text, tooltip, callback, enabled in (
            ("▲", "Move the subplot up", partial(on_move_subplot, row, -1), row > 0),
            ("▼", "Move the subplot down", partial(on_move_subplot, row, 1), row < len(viewer.values.subplots) - 1),
            ("✕", "Delete the subplot", partial(on_delete_subplot, row), True),
        ):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setAutoRaise(True)
            button.setToolTip(tooltip)
            button.setEnabled(enabled)
            button.clicked.connect(callback)
            header.addWidget(button)
        cardlayout.addLayout(header)

        chipsbox = QtWidgets.QWidget()
        chipslayout = make_flow_layout()
        chipsbox.setLayout(chipslayout)
        for position, item in get_chip_items(subplot, columns):
            if seriestype is None:
                tooltip = plain_label(get_ylabel(item)).strip() or item
            else:
                tooltip = f"The {currenttype} of {item}"
            chipslayout.addWidget(make_chip(item, tooltip, partial(on_remove_item, row, position)))
        cardlayout.addWidget(chipsbox)

        levelnames = get_levelnames(currenttype)
        choices = (
            get_species_choices(currenttype, columns, levelnames)
            if seriestype is not None
            else get_variable_choices(tuple(columns))
        )
        suggestions = get_series_suggestions(subplot, columns, levelnames)
        example = f", e.g. {suggestions[0]}" if suggestions else ""
        addedit = QtWidgets.QLineEdit()
        addedit.setObjectName("add")
        addcompleter = make_completer(choices, addedit)
        addedit.setCompleter(addcompleter)
        # the field takes the name first, and then this handler adds it
        addcompleter.activated.connect(partial(on_complete_item, row, addedit))
        if currenttype == VARIABLES_TYPE:
            addedit.setPlaceholderText(f"Add a variable{example}")
        elif currenttype.startswith("levelpopulation"):
            addedit.setPlaceholderText(f"Add a level: an ion and the index of its level{example}")
        elif currenttype == "populations":
            addedit.setPlaceholderText(f"Add an ion, an element, or an isotope{example}")
        else:
            addedit.setPlaceholderText(f"Add a species{example}")
        addedit.setToolTip(
            "Press Return to add the name. Part of a name shows the names that hold it. A directive such as"
            " ymin=1e-16 also goes here."
        )
        addedit.returnPressed.connect(partial(on_add_item, row, addedit))
        listbutton = QtWidgets.QToolButton()
        listbutton.setText("▾")
        listbutton.setToolTip("Show each name that the subplot can take")
        listbutton.clicked.connect(partial(show_all_choices, addedit))
        addrow = QtWidgets.QHBoxLayout()
        addrow.addWidget(addedit, 1)
        addrow.addWidget(listbutton)
        cardlayout.addLayout(addrow)

        if suggestions:
            suggestionsbox = QtWidgets.QWidget()
            suggestionslayout = make_flow_layout()
            suggestionsbox.setLayout(suggestionslayout)
            for suggestion in suggestions:
                suggestionslayout.addWidget(
                    make_suggestion_button(
                        suggestion, f"Add {suggestion} to the subplot", partial(add_item, row, suggestion)
                    )
                )
            cardlayout.addWidget(suggestionsbox)

        yscalebox = make_selector(
            "yscale",
            ("auto", "linear", "log"),
            get_yscale_choice(subplot),
            "The scale of the y axis (yscale=). Auto takes log for ions and linear for the other series.",
            partial(on_directive_selector, row, "yscale", "auto"),
        )
        selectorwidgets: list[QtWidgets.QWidget] = [QtWidgets.QLabel("y scale"), yscalebox]
        poptypebox = None
        if currenttype == "populations":
            poptypebox = make_selector(
                "poptype",
                tuple(POPTYPE_YLABELS),
                get_directive_value(subplot, "ionpoptype") or DEFAULT_POPTYPE,
                f"The quantity of each ion of this subplot (ionpoptype=). {DEFAULT_POPTYPE} needs no directive.",
                partial(on_directive_selector, row, "ionpoptype", DEFAULT_POPTYPE),
            )
            selectorwidgets += [QtWidgets.QLabel("Quantity"), poptypebox]
        cardlayout.addLayout(make_row_layout(selectorwidgets))

        # a Shift-drag on the subplot also sets these fields
        yminedit, ymaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for edit, directive in ((yminedit, "ymin"), (ymaxedit, "ymax")):
            edit.setObjectName(directive)
            edit.setFixedWidth(90)
            edit.setPlaceholderText("auto")
            edit.setText(get_directive_value(subplot, directive) or "")
            edit.setToolTip(f"The {directive[1:]}imum of the y axis ({directive}=). Shift-drag on the subplot sets it.")
            edit.editingFinished.connect(partial(on_yrange, row, yminedit, ymaxedit))
        cardlayout.addLayout(
            make_row_layout([QtWidgets.QLabel("y min"), yminedit, QtWidgets.QLabel("y max"), ymaxedit])
        )
        return SubplotCard(
            key=key, frame=card, yscalebox=yscalebox, poptypebox=poptypebox, yminedit=yminedit, ymaxedit=ymaxedit
        )

    # the names of the NLTE levels of the run, which get_levelnames reads when a card first needs them
    levelnamescache: list[list[str]] = []

    def get_levelnames(seriestype: str) -> list[str]:
        """Return the names of the levels for a level population, or no names for another type of series.

        The terminal shows the error of a failed read, and the level population then offers no level.
        """
        if not seriestype.startswith("levelpopulation") or not viewer.cells:
            return []
        if not levelnamescache:
            names: list[str] = []
            timestep, cell = viewer.values.last, viewer.cells[0]
            run_command_step(lambda: names.extend(get_level_names(viewer.modelpath, timestep, cell)))
            levelnamescache.append(names)
        return levelnamescache[0]

    def show_all_choices(edit: QtWidgets.QLineEdit) -> None:
        if (completer := edit.completer()) is not None:
            edit.setFocus()
            completer.setCompletionPrefix(edit.text())
            completer.complete()

    def remove_widget(layout: QtWidgets.QLayout, widget: QtWidgets.QWidget) -> None:
        """Remove a widget from the window.

        A field that loses the focus sends editingFinished, thus the controls of the widget stop their signals first.
        """
        for child in widget.findChildren(QtWidgets.QWidget):
            child.blockSignals(True)  # ruff:ignore[boolean-positional-value-in-call]
        layout.removeWidget(widget)
        widget.hide()
        widget.deleteLater()

    def show_card_directives(card: SubplotCard, subplot: "Sequence[str]") -> None:
        """Show the directives of a subplot on the controls of its card, e.g. the ymin= of a Shift-drag."""
        set_box_text(card.yscalebox, get_yscale_choice(subplot))
        if card.poptypebox is not None:
            set_box_text(card.poptypebox, get_directive_value(subplot, "ionpoptype") or DEFAULT_POPTYPE)
        set_edit_text(card.yminedit, get_directive_value(subplot, "ymin") or "")
        set_edit_text(card.ymaxedit, get_directive_value(subplot, "ymax") or "")

    def show_new_subplot_suggestions(subplottypes: "Sequence[str]", *, columnschanged: bool) -> None:
        """Show the suggestions of a new subplot, and give its field the names of the model."""
        if columnschanged:
            oldcompleter = newsubplotedit.completer()
            newsubplotedit.setCompleter(make_completer([*subplottypes[1:], *viewer.estimatorcolumns], newsubplotedit))
            if oldcompleter is not None:
                oldcompleter.deleteLater()
        oldbuttons = [
            item.widget()
            for index in range(newsuggestionslayout.count())
            if (item := newsuggestionslayout.itemAt(index)) is not None
        ]
        for button in oldbuttons:
            if button is not None:
                remove_widget(newsuggestionslayout, button)
        suggestions = get_new_subplot_suggestions(
            viewer.values.subplots, viewer.defaultsubplots, viewer.estimatorcolumns
        )
        for subplot in suggestions:
            text = shlex.join(subplot)
            newsuggestionslayout.addWidget(
                make_suggestion_button(text, f"Add a subplot of {text}", partial(add_new_subplot, subplot))
            )
        newsuggestionsbox.setVisible(bool(suggestions))

    def show_subplots() -> None:
        """Show a card for each subplot, and the suggestions of a new subplot.

        A card with the same key stays, and its controls show the new directives. A new card of the same row gives
        the keyboard focus to the control that had it in the old card.
        """
        nonlocal shownnewkey, pendingfocus
        subplots, columns = viewer.values.subplots, viewer.estimatorcolumns
        subplottypes = get_subplot_types(columns, viewer.nltetypes)
        focuswidget = QtWidgets.QApplication.focusWidget()
        focus, pendingfocus = pendingfocus, None
        for row, subplot in enumerate(subplots):
            key = get_card_key(row, subplots, columns)
            if row < len(cards) and cards[row].key == key:
                show_card_directives(cards[row], subplot)
                continue
            if row < len(cards):
                oldframe = cards[row].frame
                if focus is None and focuswidget is not None and oldframe.isAncestorOf(focuswidget):
                    focus = (row, focuswidget.objectName())
                remove_widget(subplotslayout, oldframe)
            card = make_subplot_card(row, subplot, subplottypes, key)
            subplotslayout.insertWidget(row, card.frame)
            if row < len(cards):
                cards[row] = card
            else:
                cards.append(card)
        for card in cards[len(subplots) :]:
            remove_widget(subplotslayout, card.frame)
        del cards[len(subplots) :]
        newkey = (subplots, viewer.defaultsubplots, columns)
        if newkey != shownnewkey:
            show_new_subplot_suggestions(subplottypes, columnschanged=newkey[2:] != shownnewkey[2:])
            shownnewkey = newkey
        if focus is not None and focus[1] and focus[0] < len(cards):
            target = cards[focus[0]].frame.findChild(QtWidgets.QWidget, focus[1])
            if target is not None:
                # a popup of a completer gives the focus back when it hides, thus the control takes it after the popup
                QtCore.QTimer.singleShot(0, target, target.setFocus)

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
        rows = values.otheroptions
        geometrymode = get_geometry_mode(values)
        geometrybox.setCurrentIndex(max(geometrybox.findData(geometrymode), 0))
        for widget in (cellnamelabel, celledit):
            widget.setVisible(geometrymode == "cells")
        # the slider selects one cell, which suits a plot against time. A snapshot of one cell has one point, and
        # the field of a snapshot takes a list or a range of cells
        cellslider.setVisible(geometrymode == "cells" and evolution)
        celllabel.setVisible(geometrymode in {"all", "cells"})
        axisparameters.setVisible(geometrymode in {"alongaxis", "cone"})
        for widget in (coneanglelabel, coneanglebox):
            widget.setVisible(geometrymode == "cone")
        axisbox.setCurrentText((get_row_values(rows, "-axis") or (viewer.parser.get_default("axis"),))[0])
        with contextlib.suppress(ValueError):
            coneangle = get_row_values(rows, "-coneangle") or (str(viewer.parser.get_default("coneangle")),)
            set_spin_value(coneanglebox, float(coneangle[0]))
        slicetext = (get_row_values(rows, "-slice") or ("",))[0]
        planeparameters.setVisible(geometrymode == "plane")
        lineparameters.setVisible(geometrymode == "line")
        plane, offset = get_slice_parts(slicetext)
        planebox.setCurrentText(plane)
        set_edit_text(offsetedit, offset)
        lineaxisbox.setCurrentText(get_line_axis(slicetext))
        smoothingmode, _ = get_smoothing(rows)
        smoothingbox.setCurrentIndex(max(smoothingbox.findData(smoothingmode), 0))
        for widget in (smoothinglengthlabel, smoothinglengthbox):
            widget.setVisible(smoothingmode != "none")
        for widget in (smoothingorderlabel, smoothingorderbox):
            widget.setVisible(smoothingmode == "savgol")
        # the Savitzky-Golay filter takes an odd window length of at least 3, thus its arrows skip the even lengths
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
            set_spin_value(fontsizebox, float((get_row_values(rows, "-labelfontsize") or ("0",))[0]))
        with contextlib.suppress(ValueError):
            figscale = get_row_values(rows, "-figscale") or (str(viewer.parser.get_default("figscale")),)
            set_spin_value(figscalebox, float(figscale[0]))
        timeslider.setValue((firstpos + lastpos) // 2)
        widthslider.setValue(lastpos - firstpos + 1)
        widthlabel.setText(f"Timesteps: {lastpos - firstpos + 1}")
        set_edit_text(timeedit, get_time_text(viewer.tmids, values))
        timestepslabel.setText(viewer.get_timesteps_text())
        set_edit_text(celledit, values.cells)
        if (cell := get_single_cell(values.cells)) is not None and cell in viewer.cells:
            cellslider.setValue(viewer.cells.index(cell))
        celllabel.setText(viewer.get_cell_text())
        xbox.setCurrentText(values.x)
        xunit = get_xunit_text(viewer.xlimitscale, values.x)
        xminlabel.setText(f"-xmin{xunit}")
        xmaxlabel.setText(f"-xmax{xunit}")
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
        markerscheck.setText(
            "--markers (on for -xbins 0)" if viewer.plotmarkers and not values.markers else "--markers"
        )
        colorbyioncheck.setChecked(values.colorbyion)
        colorbyioncheck.setText(
            "--colorbyion (on for automatic bins)"
            if viewer.plotcolorbyion and not values.colorbyion
            else "--colorbyion"
        )
        show_subplots()
        defaultbutton.setEnabled(values.subplots != viewer.defaultsubplots and bool(viewer.defaultsubplots))
        set_option_rows(get_table_rows(values.otheroptions))
        for widget in (optionheader, optioncontent):
            widget.setVisible(tableoffers or bool(get_table_rows(values.otheroptions)))
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
        show_status_message(statusbar, message, "")
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

    def apply_rows(changes: "Mapping[str, tuple[str, ...] | None]") -> None:
        """Apply new values of some rows of the option table, e.g. of a section of the window."""
        apply(replace_option_rows(viewer, viewer.values, set_row_values(viewer.values.otheroptions, changes)))

    def on_geometry(index: int) -> None:
        apply(set_geometry_mode(viewer, viewer.values, str(geometrybox.itemData(index))))

    def on_axis(axis: str) -> None:
        apply_rows({"-axis": None if axis == viewer.parser.get_default("axis") else (axis,)})

    def on_coneangle(angle: float) -> None:
        isdefault = math.isclose(angle, viewer.parser.get_default("coneangle"))
        apply_rows({"-coneangle": None if isdefault else (format(angle, "g"),)})

    def on_plane() -> None:
        offsetedit.setModified(False)
        apply_rows({"-slice": (get_slice_text(planebox.currentText(), offsetedit.text()),)})

    def on_lineaxis(axis: str) -> None:
        apply_rows({"-slice": (",".join(f"{other}=0" for other in "zyx" if other != axis),)})

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
        apply_rows({"-labelfontsize": (format(fontsize, "g"),) if fontsize > 0.0 else None})

    def on_figscale(figscale: float) -> None:
        isdefault = math.isclose(figscale, viewer.parser.get_default("figscale"))
        apply_rows({"-figscale": None if isdefault else (format(figscale, "g"),)})

    def on_xvariable() -> None:
        if xvariable := xbox.currentText().strip():
            apply(viewer.set_xvariable(viewer.values, xvariable))

    def get_limit_texts(edits: "Sequence[QtWidgets.QLineEdit]", low: str, high: str) -> list[str] | None:
        """Return the numbers of two fields in the form of the command, or None after an error message.

        An empty field gives an empty text, which takes the limit of the data.
        """
        texts: list[str] = []
        for edit in edits:
            # a later plot can show new text in the field only when the field has no edit of the user
            edit.setModified(False)
            text = edit.text().strip()
            try:
                texts.append(format(float(text), ".10g") if text else "")
            except ValueError:
                show_error(f"Give a number for {low} and {high}, or leave a field empty for the range of the data")
                return None
        if texts[0] and texts[1] and float(texts[0]) >= float(texts[1]):
            show_error(f"Give a {low} that is less than {high}")
            return None
        return texts

    def on_xedit() -> None:
        if (texts := get_limit_texts((xminedit, xmaxedit), "-xmin", "-xmax")) is not None:
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
        """Apply the subplots, or show the default subplots if the list is empty.

        -ionpoptype of the command waits for a populations subplot, and then it goes to each one.
        """
        newsubplots = tuple(subplot for subplot in subplots if subplot) or viewer.defaultsubplots
        if not newsubplots:
            show_error("Give the items of at least one subplot")
            return
        newsubplots, otheroptions = move_poptype_to_subplots(
            newsubplots, viewer.values.otheroptions, viewer.estimatorcolumns
        )
        apply(dc.replace(viewer.values, subplots=newsubplots, otheroptions=otheroptions))

    def set_subplot(row: int, subplot: tuple[str, ...]) -> None:
        subplots = list(viewer.values.subplots)
        subplots[row] = subplot
        apply_subplots(subplots)

    def is_shown_card(row: int, edit: QtWidgets.QLineEdit) -> bool:
        """Return True if a field belongs to the card of the row on the screen, and not to an old card."""
        return row < len(cards) and cards[row].frame.isAncestorOf(edit)

    def on_subplot_type(row: int, seriestype: str) -> None:
        subplot = viewer.values.subplots[row]
        levelnames = get_levelnames(seriestype)
        set_subplot(row, change_subplot_type(subplot, seriestype, viewer.estimatorcolumns, levelnames))

    def on_yrange(row: int, yminedit: QtWidgets.QLineEdit, ymaxedit: QtWidgets.QLineEdit) -> None:
        if not is_shown_card(row, yminedit):
            return
        if (texts := get_limit_texts((yminedit, ymaxedit), "y min", "y max")) is not None:
            subplot = viewer.values.subplots[row]
            set_subplot(row, replace_directives(subplot, {"ymin": texts[0] or None, "ymax": texts[1] or None}))

    def add_item(row: int, item: str) -> None:
        subplot = viewer.values.subplots[row]
        item = VARIABLE_ALIASES.get(item, item)
        directive = get_item_directive(item)
        if directive is not None:
            value = item.partition("=")[2].strip()
            if not value:
                show_error(f"Give a value after {item}, e.g. {directive}=1e-16")
                return
            set_subplot(row, replace_directives(subplot, {directive: value}))
        elif item in subplot:
            show_error(f"The subplot already shows {item}")
        else:
            names = get_subplot_names(subplot)
            # a name goes after the other names, thus the directives stay at the end
            set_subplot(
                row,
                (*names, item, *subplot[len(names) :]) if subplot[: len(names)] == tuple(names) else (*subplot, item),
            )

    def popup_has_pick(edit: QtWidgets.QLineEdit) -> bool:
        """Return True if the popup of the completer of a field shows a highlighted name.

        Return in the popup gives returnPressed first and activated second, thus only the handler of activated
        takes the name.
        """
        completer = edit.completer()
        popup = completer.popup() if completer is not None else None
        return popup is not None and popup.isVisible() and popup.currentIndex().isValid()

    def on_add_item(row: int, edit: QtWidgets.QLineEdit) -> None:
        if popup_has_pick(edit) or not is_shown_card(row, edit):
            return
        item = edit.text().strip()
        edit.clear()
        if item:
            add_item(row, item)

    def on_complete_item(row: int, edit: QtWidgets.QLineEdit, item: str) -> None:
        if not is_shown_card(row, edit):
            return
        edit.clear()
        add_item(row, item)

    def on_remove_item(row: int, position: int) -> None:
        apply_subplots(remove_subplot_item(viewer.values.subplots, row, position, viewer.estimatorcolumns))

    def on_delete_subplot(row: int) -> None:
        apply_subplots([subplot for position, subplot in enumerate(viewer.values.subplots) if position != row])

    def on_move_subplot(row: int, step: int) -> None:
        subplots = list(viewer.values.subplots)
        if 0 <= row + step < len(subplots):
            subplots[row], subplots[row + step] = subplots[row + step], subplots[row]
            apply_subplots(subplots)

    def on_directive_selector(row: int, directive: str, defaulttext: str, text: str) -> None:
        subplot = viewer.values.subplots[row]
        set_subplot(row, replace_directives(subplot, {directive: None if text == defaulttext else text}))

    def add_new_subplot(subplot: tuple[str, ...]) -> None:
        nonlocal pendingfocus
        if subplot:
            # the field of the new card takes the focus, thus the user can add more names
            pendingfocus = (len(viewer.values.subplots), "add")
            apply_subplots([*viewer.values.subplots, subplot])

    def on_new_subplot() -> None:
        # a name that the user picks in the popup goes into the field, and the next Return adds the subplot
        if popup_has_pick(newsubplotedit):
            return
        text = newsubplotedit.text()
        newsubplotedit.clear()
        words = text.split()
        add_new_subplot(make_new_subplot(text, viewer.estimatorcolumns, get_levelnames(words[0] if words else "")))

    def on_copy() -> None:
        copy_command(viewer.get_command())
        show_status_note(statusbar, "Copied the command")

    def on_save() -> None:
        from artistools.estimators.plotestimators import main as plotestimators_main

        defaultdpi = viewer.parser.get_default("dpi")
        rows, dpi = split_dpi_row(viewer.values.otheroptions, defaultdpi)
        plottokens = viewer.get_plot_tokens(dc.replace(viewer.values, otheroptions=rows))
        save_figure_of_command(window, statusbar, plotestimators_main, "plotestimators", plottokens, dpi, defaultdpi)

    def on_open_model() -> None:
        if (message := open_model_window(window, open_window, windows)) is not None:
            show_error(message)

    def on_reload() -> None:
        """Read the run again in the worker thread.

        A conversion of new text files can take minutes, thus the window stays responsive, and the terminal shows the
        progress. The reload waits for the plot in progress, and a new plot waits for the reload. Thus a plot never
        reads a cache that the reload replaces.
        """
        modelpath, args, ntimesteps = viewer.modelpath, viewer.userargs, len(viewer.tmids)
        reloadedruns: list[RunData] = []

        def read() -> None:
            reloadedruns.append(read_run_again(modelpath, args, ntimesteps))

        def show_reloaded_run(message: str | None) -> None:
            if message is not None or not reloadedruns:
                show_error(f"The viewer cannot reload the run: {message}")
                return
            reload_run(viewer, reloadedruns[0])
            set_ranges()
            queue.redraw()

        if not queue.run_task(lambda: run_command_step(read, quiet=False), "Reload in progress...", show_reloaded_run):
            show_error("A reload of the run is in progress")

    def on_help() -> None:
        QtWidgets.QMessageBox.information(
            window, "Keys and mouse actions", get_keyboard_help(KEYBOARD_HELP_ROWS, menucallbacks)
        )

    def get_frame_readout(event: t.Any, frame: "mplax.Axes") -> str:
        if not viewer.isimage:
            return get_readout(frame, event.xdata)
        meshes = [collection for collection in frame.collections if collection.get_array() is not None]
        value = get_image_value(meshes[0].get_cursor_data(event)) if meshes else None
        valuetext = f"   value: {value:.4g}" if value is not None else ""
        return f"x = {event.xdata:.4g}c   y = {event.ydata:.4g}c{valuetext}"

    def on_select(low: float, high: float) -> None:
        xmin, xmax = viewer.get_xlimit_text(low), viewer.get_xlimit_text(high)
        if plot_shows_values() and float(xmin) < float(xmax):
            apply(dc.replace(viewer.values, xmin=xmin, xmax=xmax))

    def plot_shows_values() -> bool:
        """Return True if the plot on the screen has the subplots, the x variable, and the options of the controls.

        DrawQueue gives the viewer the new values at once, and the old plot stays until the worker draws the new one.
        A mouse action on an old frame then must not change a different subplot or read a different x variable.
        """
        drawn, values = queue.drawnvalues, viewer.values
        return (drawn.subplots, drawn.x, drawn.otheroptions) == (values.subplots, values.x, values.otheroptions)

    def get_subplot_row(frameindex: int) -> int | None:
        """Return the row of the subplot of a frame of the plot, or None for a colour image."""
        return frameindex if not viewer.isimage and frameindex < len(viewer.values.subplots) else None

    def set_directives(row: int, directives: "Mapping[str, str | None]") -> None:
        subplots = list(viewer.values.subplots)
        subplots[row] = replace_directives(subplots[row], directives)
        apply(dc.replace(viewer.values, subplots=tuple(subplots)))

    def on_select_y(frameindex: int, low: float, high: float) -> None:
        row = get_subplot_row(frameindex)
        ymin, ymax = get_short_number(low), get_short_number(high)
        if row is not None and plot_shows_values() and float(ymin) < float(ymax):
            set_directives(row, {"ymin": ymin, "ymax": ymax})

    def on_menu(frameindex: int, event: t.Any) -> None:
        """Show the menu of a subplot: the y scale, the y range, and the plot of a cell or of a snapshot."""
        if not plot_shows_values():
            return
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
        elif cells_apply(viewer.values):
            cell = get_nearest_cell(viewer, event.xdata)
            if cell is not None:
                cellaction = menu.addAction(f"Plot cell {cell} against time")
                cellaction.triggered.connect(lambda: apply(get_evolution_values(viewer, str(cell))))
            if viewer.values.cells:
                cellsaction = menu.addAction(f"Plot the cells {viewer.values.cells} against time")
                cellsaction.triggered.connect(lambda: apply(get_evolution_values(viewer, viewer.values.cells)))
        if menu.actions():
            menu.exec(QtGui.QCursor.pos())
        # the window is the parent of the menu, thus without this the window keeps each menu until it closes
        menu.deleteLater()

    def on_closed() -> None:
        print(viewer.get_command())
        if queue.task is not None:
            print("The reload of the run continues to its end, and then the process ends")
        queue.close()
        # each kept scan holds the metadata of its file, e.g. 7.6 MB for 3000 columns
        scan_parquet_file.cache_clear()
        # the list holds a reference to each open window, thus Python does not delete the window. A closed window
        # leaves the list
        windows.remove(window)

    menucallbacks = {
        "Open Model...": on_open_model,
        "Reload Data": on_reload,
        "Save Figure...": on_save,
        "Copy Command": on_copy,
        "Close Window": window.close,
        "Keys and Mouse Actions": on_help,
    }
    add_menus(window, menucallbacks)

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
    geometrybox.activated.connect(on_geometry)
    axisbox.textActivated.connect(on_axis)
    coneanglebox.valueChanged.connect(on_coneangle)
    planebox.textActivated.connect(on_plane)
    offsetedit.editingFinished.connect(on_plane)
    lineaxisbox.textActivated.connect(on_lineaxis)
    smoothingbox.activated.connect(on_smoothing_mode)
    smoothinglengthbox.valueChanged.connect(on_smoothing_length)
    smoothingorderbox.valueChanged.connect(on_smoothing_order)
    for flag, check in appearancechecks.items():
        check.toggled.connect(partial(on_appearance_check, flag))
    fontsizebox.valueChanged.connect(on_fontsize)
    figscalebox.valueChanged.connect(on_figscale)
    xbox.activated.connect(on_xvariable)
    if (xlineedit := xbox.lineEdit()) is not None:
        xlineedit.editingFinished.connect(on_xvariable)
    xminedit.editingFinished.connect(on_xedit)
    xmaxedit.editingFinished.connect(on_xedit)
    xbinsedit.editingFinished.connect(on_style)
    markerscheck.toggled.connect(on_style)
    colorbyioncheck.toggled.connect(on_style)
    newsubplotedit.returnPressed.connect(on_new_subplot)
    addsubplotbutton.clicked.connect(on_new_subplot)
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
        on_reset=lambda: apply(dc.replace(viewer.values, xmin="", xmax="")),
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

    show_window(window, viewer.figsize, lambda: fit_canvas(canvas, viewer.figsize, plotarea))
    show_values()
    return None
