"""Build a viewer window around its figure, and draw the plot in a worker thread."""

import dataclasses as dc
import math
import time
import traceback
import typing as t
from functools import partial
from pathlib import Path
from types import MappingProxyType

import numpy as np

from artistools.misc import print_error
from artistools.misc.remote import on_model_host
from artistools.plottools import make_room_for_title
from artistools.plottools import plain_label
from artistools.viewertools.application import add_recent_model
from artistools.viewertools.application import clear_field_error
from artistools.viewertools.application import get_edited_field
from artistools.viewertools.application import get_settings
from artistools.viewertools.application import get_window_setting_keys
from artistools.viewertools.application import is_live
from artistools.viewertools.application import make_central_splitter
from artistools.viewertools.application import make_window
from artistools.viewertools.application import mark_field_error
from artistools.viewertools.core import FIT_MILLISECONDS
from artistools.viewertools.core import FIT_TOLERANCE
from artistools.viewertools.core import fix_title_position
from artistools.viewertools.core import get_first_line
from artistools.viewertools.core import run_command_step
from artistools.viewertools.core import run_command_step_with_warning
from artistools.viewertools.core import run_has_direction_data
from artistools.viewertools.core import SIDEBAR_WIDTH
from artistools.viewertools.core import UNDO_LIMIT
from artistools.viewertools.core import UNDO_MERGE_SECONDS
from artistools.viewertools.menus import add_figure_section
from artistools.viewertools.menus import apply_dark_colours
from artistools.viewertools.menus import FigureSection
from artistools.viewertools.menus import follow_colour_scheme
from artistools.viewertools.menus import get_dark_plot_colours
from artistools.viewertools.menus import set_window_document
from artistools.viewertools.menus import show_figure_in_canvas
from artistools.viewertools.widgets import add_command_section
from artistools.viewertools.widgets import add_copy_box
from artistools.viewertools.widgets import add_section
from artistools.viewertools.widgets import align_section_labels
from artistools.viewertools.widgets import fit_canvas
from artistools.viewertools.widgets import make_option_table
from artistools.viewertools.widgets import make_plot_area
from artistools.viewertools.widgets import make_sidebar
from artistools.viewertools.widgets import make_status_bar
from artistools.viewertools.widgets import set_plot_busy
from artistools.viewertools.widgets import show_plot_area_state
from artistools.viewertools.widgets import show_plot_banner
from artistools.viewertools.widgets import show_status_message
from artistools.viewertools.widgets import StatusBar

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Sequence
    from concurrent.futures import Future

    import matplotlib.axes as mplax
    import matplotlib.figure as mplfig
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    from artistools.commands import SuggestingArgumentParser
    from artistools.viewertools.core import OptionRows
    from artistools.viewertools.core import PlotValues


def clear_output_caches() -> None:
    """Clear each cache of the files that ARTIS writes while it runs, e.g. spec.out, in this process.

    The files of the input of a run stay the same, e.g. input.txt and model.txt, thus their caches stay.
    """
    from artistools.estimators.core import scan_parquet_file
    from artistools.estimators.estimators_classic import read_classic_estimators_cached
    from artistools.misc.modelinfo import get_nu_grid_cached
    from artistools.misc.modelinfo import get_runfolder_timesteps_cached
    from artistools.misc.timesteps import get_deposition_cached
    from artistools.misc.timesteps import get_escaped_arrivalrange_cached
    from artistools.misc.timesteps import get_timestep_times_cached
    from artistools.nltepops.core import read_nltepops_cached
    from artistools.packets.core import check_packets_batch_parquet_paths
    from artistools.packets.core import find_first_rank_textfiles
    from artistools.spectra.core import get_flux_contributions_cached
    from artistools.spectra.core import get_vspecpol_data_cached
    from artistools.spectra.core import read_spec_cached
    from artistools.spectra.core import read_spec_res_cached
    from artistools.spectra.core import read_specpol_res_cached
    from artistools.spectra.plotspectra import has_gamma_spec_file

    for cachedfunction in (
        # the file tests of the client also go, e.g. after exspec writes gamma_spec.out
        has_gamma_spec_file,
        run_has_direction_data,
        get_runfolder_timesteps_cached,
        get_nu_grid_cached,
        get_deposition_cached,
        get_escaped_arrivalrange_cached,
        get_timestep_times_cached,
        check_packets_batch_parquet_paths,
        find_first_rank_textfiles,
        get_flux_contributions_cached,
        get_vspecpol_data_cached,
        read_spec_cached,
        read_spec_res_cached,
        read_specpol_res_cached,
        # a kept scan of a parquet cache also holds the metadata of its file, e.g. 8 MB for 5335 columns
        scan_parquet_file,
        read_classic_estimators_cached,
        read_nltepops_cached,
    ):
        cachedfunction.cache_clear()


@on_model_host
def clear_output_caches_of_run(runfolder: Path) -> None:
    """Clear the caches of the output files on the host of a run. A local run clears the caches of this process."""
    del runfolder
    clear_output_caches()


def reload_runs(
    queue: "DrawQueue[t.Any]",
    runfolders: "Sequence[Path | str]",
    on_reloaded: "Callable[[], None]",
    show_error: "Callable[[str], None]",
) -> None:
    """Clear the caches of the runs in the worker thread, e.g. while ARTIS writes more timesteps, then call on_reloaded.

    The caches of this process and of the host of a remote run hold old data, thus the function clears both. The
    reload waits for the plot in progress, and a new plot waits for the reload. on_reloaded runs in the window thread,
    and it reads the runs again, e.g. their timesteps.
    """

    def read() -> None:
        clear_output_caches()
        for runfolder in runfolders:
            clear_output_caches_of_run(Path(runfolder))

    def on_done(message: str | None) -> None:
        if message is not None:
            show_error(f"The viewer cannot reload the runs: {message}")
            return
        on_reloaded()

    if not queue.run_task(lambda: run_command_step(read, quiet=False), "Reload in progress...", on_done):
        show_error("A different task is in progress. Reload the runs after it")


class ViewerWindow(t.NamedTuple):
    """The parts of a window of a viewer that start_viewer_window makes."""

    window: "QtWidgets.QMainWindow"
    canvas: "FigureCanvasQTAgg"
    plotarea: "QtWidgets.QWidget"
    # the layout of the sidebar, which takes the sections of the viewer
    panellayout: "QtWidgets.QVBoxLayout"
    # the timer that starts a new fit of the width of the figure when a resize stops
    fittimer: "QtCore.QTimer"


def start_viewer_window(
    applicationname: str,
    viewer: "PlotViewer[t.Any]",
    draw: "Callable[..., str | None]",
    windows: "list[QtWidgets.QMainWindow]",
    document: tuple[Path, str],
) -> ViewerWindow | str:
    """Make the window of a viewer with its first plot, the plot area, and the sidebar.

    draw is the draw method of the viewer. document gives the folder of the model and the title of the window. A first
    plot that the command rejects gives its reason and no window. The terminal shows the whole error, and the first
    window of a start then stops the process.
    """
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    window = make_window(applicationname)
    set_window_document(window, *document)
    canvas = FigureCanvasQTAgg(viewer.fig)
    viewer.darkcolours = get_dark_plot_colours()
    if (message := draw(quiet=False)) is not None:
        if not windows:
            raise SystemExit(1)
        return message
    windows.append(window)
    add_recent_model(document[0])
    fittimer = make_timer(window, FIT_MILLISECONDS)

    def on_resize() -> None:
        fit_canvas(canvas, viewer.figsize, plotarea)
        # a new plot can take a few seconds, thus the plot takes the new shape only when the resize stops
        fittimer.start()

    plotarea = make_plot_area(canvas, on_resize)
    sidebar, panellayout = make_sidebar()
    make_central_splitter(window, plotarea, sidebar)
    return ViewerWindow(window=window, canvas=canvas, plotarea=plotarea, panellayout=panellayout, fittimer=fittimer)


class CommandSections(t.NamedTuple):
    """The last sections of a window of a viewer: the figure, the other options, the command, and the Python code."""

    figuresection: FigureSection
    # the function that shows new rows in the table of the other options
    set_option_rows: "Callable[[OptionRows], None]"
    commandtext: "QtWidgets.QPlainTextEdit"
    pythontext: "QtWidgets.QPlainTextEdit"
    copybuttons: "tuple[QtWidgets.QPushButton, QtWidgets.QPushButton]"
    statusbar: StatusBar


def add_command_sections(
    viewerwindow: ViewerWindow,
    viewer: "PlotViewer[t.Any]",
    parser: "SuggestingArgumentParser",
    table: "tuple[Collection[str], OptionRows, Callable[[OptionRows], None]]",
) -> CommandSections:
    """Add the Figure section, the table of the other options, the command, the Python code, and the status bar.

    table gives the options that the table hides, the rows of the table, and the function that receives new rows.
    """
    window, panellayout = viewerwindow.window, viewerwindow.panellayout
    defaultdpi: int = parser.get_default("dpi")
    figuresection = add_figure_section(window, panellayout, viewer.values.dpi or defaultdpi)
    _, optiongrid = add_section(panellayout, "Other options")
    hiddendests, rows, on_rows = table
    optiontable, set_option_rows = make_option_table(window, parser, hiddendests, rows, on_rows)
    optiongrid.addWidget(optiontable, 0, 0, 1, 2)
    commandtext, copybutton = add_command_section(panellayout)
    pythontext, pythoncopybutton = add_copy_box(
        panellayout, "Python", "Copy the Python code that draws the plot to the clipboard", maxlines=20, wraplines=False
    )
    statusbar = make_status_bar(window)
    # the first plot came before the status bar, and a user of the application sees no terminal
    show_status_message(statusbar, None, viewer.warning)
    return CommandSections(
        figuresection=figuresection,
        set_option_rows=set_option_rows,
        commandtext=commandtext,
        pythontext=pythontext,
        copybuttons=(copybutton, pythoncopybutton),
        statusbar=statusbar,
    )


def finish_viewer_window[ValuesT: PlotValues](
    viewerwindow: ViewerWindow,
    windows: "list[QtWidgets.QMainWindow]",
    viewer: "PlotViewer[ValuesT]",
    queue: "DrawQueue[ValuesT]",
    get_session_tokens: "Callable[[], list[str]]",
    on_closed: "Callable[[], None] | None" = None,
) -> None:
    """Connect the parts that each window of a viewer has, then show the window.

    get_session_tokens gives the command that opens the window again at the next start. on_closed runs when the window
    closes, e.g. to clear the caches of a run.
    """
    window, canvas, plotarea = viewerwindow.window, viewerwindow.canvas, viewerwindow.plotarea

    def fit_figwidthscale() -> None:
        values = viewer.values
        figwidthscale = get_new_figwidthscale(
            plotarea, viewer.figsize, values.figwidthscale, viewer.get_fitted_figwidthscale
        )
        # the window sets the width, thus the change is not undoable, and Undo does not return to an old width
        if figwidthscale is not None:
            queue.apply(dc.replace(values, figwidthscale=figwidthscale), undoable=False)

    def on_window_closed() -> None:
        print(viewer.get_command())
        queue.close()
        if on_closed is not None:
            on_closed()
        # the list holds a reference to each open window, thus Python does not delete the window. A closed window
        # leaves the list
        windows.remove(window)

    viewerwindow.fittimer.timeout.connect(fit_figwidthscale)
    follow_colour_scheme(window, viewer, queue)
    show_plot_area_state(window, viewer.fig)
    # the window keeps its command at a quit, and the next start opens the window again
    window.setProperty("sessiontokens", get_session_tokens)
    window.destroyed.connect(on_window_closed)
    show_window(window, lambda: fit_canvas(canvas, viewer.figsize, plotarea))


def get_new_figwidthscale(
    plotarea: "QtWidgets.QWidget",
    figsize: tuple[float, float],
    figwidthscale: float,
    get_fitted: "Callable[[float, float], float]",
) -> float | None:
    """Return the -figwidthscale that fills the plot area, or None if the plot can keep its scale.

    get_fitted gives the fitted scale for the width and the height of the area. A change of less than FIT_TOLERANCE
    keeps the scale.
    """
    area = plotarea.contentsRect()
    if area.width() <= 0 or area.height() <= 0 or figsize[0] <= 0.0:
        return None
    fitted = get_fitted(area.width(), area.height())
    return fitted if abs(fitted - figwidthscale) > FIT_TOLERANCE * figwidthscale else None


class PlotViewer[ValuesT](t.Protocol):
    """A viewer with the values of its controls, the last warning of its plot, and the colours of Dark Mode."""

    values: ValuesT
    # the figure of the canvas, and its size in inches, which render_command sets for each plot
    fig: "mplfig.Figure"
    figsize: tuple[float, float]
    # the last warning of the last plot, which the status bar shows. A user of the application sees no terminal
    warning: str
    # the background and the foreground of the plot in Dark Mode, or None for the usual colours
    darkcolours: tuple[str, str] | None

    def get_plot_tokens(self, values: ValuesT | None = None) -> list[str]:
        """Return the arguments of the command for the values, or for the current values if values is None."""

    def get_command(self) -> str:
        """Return the command that draws the plot of the current values."""

    def get_fitted_figwidthscale(self, areawidth: float, areaheight: float) -> float:
        """Return the -figwidthscale that gives the figure the shape of a plot area of this width and height."""


def render_command[PlotT](
    viewer: "PlotViewer[t.Any]",
    draw: "Callable[[mplfig.Figure], PlotT | str]",
    keep: "Callable[[PlotT], None]",
    *,
    quiet: bool,
) -> "Callable[[], str | None]":
    """Draw a plot on a new figure, and return the function that shows it in the canvas of the viewer.

    draw parses the command and draws the plot on the empty figure that it receives. It returns the frames and the
    data that the window reads, or the reason that it rejects the values. keep gives these to the viewer. Each plot of
    each viewer gets the same last steps: the titles, the colours of Dark Mode, and the layout of the text.

    A worker thread can run this function, because it changes nothing that the window reads. The function that it
    returns must run in the thread of the window. That function returns the reason for the status line if the command
    rejects the values, and the old plot then stays.
    """
    import matplotlib.figure as mplfig
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    plots: list[tuple[mplfig.Figure, PlotT]] = []

    def make_plot() -> str | None:
        fig = mplfig.Figure()
        FigureCanvasAgg(fig)
        result = draw(fig)
        if isinstance(result, str):
            return result
        for axis in fig.axes:
            fix_title_position(axis)
        make_room_for_title(fig)
        if (darkcolours := viewer.darkcolours) is not None:
            apply_dark_colours(fig, *darkcolours)
        # the worker makes the ticks and the text layout, thus the first draw in the window is faster. On the test
        # model, the window draw of a spectrum took 33 ms in place of 44 ms, and of estimators 73 ms in place of 120 ms
        fig.draw_without_rendering()
        plots.append((fig, result))
        return None

    message, warning = run_command_step_with_warning(make_plot, quiet=quiet)

    def show_plot() -> str | None:
        viewer.warning = warning
        if message is not None:
            return message
        fig, result = plots[0]
        viewer.figsize = show_figure_in_canvas(viewer.fig, fig)
        viewer.fig = fig
        keep(result)
        return None

    return show_plot


def changes_values[ValuesT](
    values: ValuesT, current: ValuesT, keep_on_undo: "Callable[[ValuesT, ValuesT], ValuesT] | None"
) -> bool:
    """Return True if Undo or Redo to values gives values that differ from the current values.

    keep_on_undo keeps the parts that the window sets, e.g. the width of the figure. Thus an entry that differs only
    in those parts gives no step.
    """
    return (keep_on_undo(values, current) if keep_on_undo is not None else values) != current


# the time of a plot before the spinner shows over the plot
BUSY_MILLISECONDS: t.Final = 300


def get_plot_time_text(plotseconds: float) -> str:
    """Return the text of the status bar for the time of the last plot, or no text before the first plot."""
    return f"Plot time: {plotseconds:.2f} s" if plotseconds > 0.0 else ""


class DrawQueue[ValuesT]:
    """The plots of a window: a change shows its values at once, and the plot follows when Qt has no other events.

    A drag gives a new value for each movement of the mouse, and a plot can take seconds. The queue draws only the
    last values that the user gave. viewer.values holds the last values that the user gave, and drawnvalues holds the
    values of the plot. If the command rejects new values, the viewer keeps the values of the plot.

    A worker thread draws each plot. The window then shows each new value of a drag at once.

    The queue also keeps the values before each change of the user, thus Undo and Redo can return to them.
    """

    def __init__(
        self,
        window: "QtCore.QObject",
        viewer: PlotViewer[ValuesT],
        statusbar: StatusBar,
        show_values: "Callable[[], None]",
        after_draw: "Callable[[str | None], None]",
        render: "Callable[[ValuesT], Callable[[], str | None]]",
        keep_on_undo: "Callable[[ValuesT, ValuesT], ValuesT] | None" = None,
    ) -> None:
        """Make an empty queue. after_draw receives the message of each plot of the queue.

        render draws the plot of the values in a worker thread. It returns the function that shows that plot in the
        window and gives the message of a rejection.

        keep_on_undo receives the values that Undo or Redo restores and the current values. It returns the restored
        values with the parts that the window sets and the user does not, e.g. the width of the figure.
        """
        from concurrent.futures import ThreadPoolExecutor

        from PySide6 import QtCore

        self.window = window
        self.viewer = viewer
        self.statusbar = statusbar
        self.show_values = show_values
        self.after_draw = after_draw
        self.requestedvalues: ValuesT | None = None
        self.drawnvalues: ValuesT = viewer.values
        self.render = render
        self.keep_on_undo = keep_on_undo
        # the values before each change of the user, for Undo, and the values that Undo replaced, for Redo
        self.undovalues: list[ValuesT] = []
        self.redovalues: list[ValuesT] = []
        self.lastchangetime = -math.inf
        # the text field that gave the change of requestedvalues, the text field that gave the plot in progress, and
        # the text field that has the mark of a rejection. A rejection marks the field of its own plot
        self.editedfield: QtWidgets.QLineEdit | None = None
        self.renderedfield: QtWidgets.QLineEdit | None = None
        self.errorfield: QtWidgets.QLineEdit | None = None
        self.renderedvalues: ValuesT = viewer.values
        self.rendering: Future[Callable[[], str | None]] | None = None
        self.renderstart = 0.0
        # the time of the last plot, which sets the pause of Play
        self.plotseconds = 0.0
        # a task of the worker thread that is not a plot, e.g. Reload Data, from run_task to its end.
        # taskfuture is None until the plot in progress ends
        self.task: Callable[[], str | None] | None = None
        self.taskstatus = ""
        self.on_task_done: Callable[[str | None], None] | None = None
        self.taskfuture: Future[str | None] | None = None
        # True after Cancel Plot, until the plot in progress ends. The queue then discards that plot
        self.discardrendering = False
        # the values before each change that waits for its plot, and the undo and redo lists before that change
        self.pendinghistory: list[tuple[ValuesT, list[ValuesT], list[ValuesT]]] = []
        # one worker thread draws one plot at a time, and a drag during a plot waits for the end of that plot
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plot")
        # the window thread checks the worker at each tick, because a Qt call from the worker thread is not safe
        self.rendertimer = QtCore.QTimer(window)
        self.rendertimer.setInterval(10)
        self.rendertimer.timeout.connect(self.show_rendered)
        # a fast plot shows no spinner, thus the spinner does not flash at each step of a drag
        self.busytimer = QtCore.QTimer(window)
        self.busytimer.setSingleShot(True)
        self.busytimer.setInterval(BUSY_MILLISECONDS)
        self.busytimer.timeout.connect(partial(set_plot_busy, window, busy=True))

    def apply(self, values: ValuesT, *, undoable: bool = True) -> None:
        """Show the new values now, and draw them when Qt has no other events. Only a change of the values draws a plot.

        A change that the window makes, e.g. a step of Play or a fit to the size of the window, is not undoable.
        """
        if values == self.viewer.values:
            # a handler that clamps a control to the old values must still move the control back
            self.show_values()
            return
        # Cancel Plot returns to the history of the plot on the screen
        if self.viewer.values == self.drawnvalues:
            self.pendinghistory.clear()
        self.pendinghistory.append((self.viewer.values, [*self.undovalues], [*self.redovalues]))
        if undoable:
            now = time.monotonic()
            # a drag of a slider or a repeated key gives many changes, and one step of Undo reverts all of them
            if not self.undovalues or now - self.lastchangetime > UNDO_MERGE_SECONDS:
                self.undovalues = [*self.undovalues[-UNDO_LIMIT + 1 :], self.viewer.values]
            self.lastchangetime = now
            self.redovalues.clear()
        # the plot of the change comes later, and a rejection then marks the field that gave the change
        self.editedfield = get_edited_field(self.window)
        # each handler makes its values from viewer.values, thus a second change before the plot keeps the first
        self.viewer.values = values
        self.redraw()

    def show_error(self, message: str) -> None:
        """Show the reason that the viewer rejects a change of the user, and show the values of the viewer again."""
        show_status_message(self.statusbar, message, "")
        self.show_values()

    def can_undo(self) -> bool:
        """Return whether Undo has values that differ from the current values."""
        return any(changes_values(values, self.viewer.values, self.keep_on_undo) for values in self.undovalues)

    def can_redo(self) -> bool:
        """Return whether Redo has values that differ from the current values."""
        return any(changes_values(values, self.viewer.values, self.keep_on_undo) for values in self.redovalues)

    def undo(self) -> None:
        """Return to the values before the last change of the user."""
        self.step_history(self.undovalues, self.redovalues)

    def redo(self) -> None:
        """Return to the values that the last Undo replaced."""
        self.step_history(self.redovalues, self.undovalues)

    def step_history(self, source: list[ValuesT], target: list[ValuesT]) -> None:
        """Take the last values of source that differ from the current values, and keep the current values in target.

        The command can reject a change, and the viewer then keeps the old values. Such a change gives values in
        source that are the same as the current values, and a step of Undo skips them.
        """
        while source and not changes_values(source[-1], self.viewer.values, self.keep_on_undo):
            source.pop()
        if not source:
            return
        if self.viewer.values == self.drawnvalues:
            self.pendinghistory.clear()
        self.pendinghistory.append((self.viewer.values, [*self.undovalues], [*self.redovalues]))
        restored = source.pop()
        if self.keep_on_undo is not None:
            restored = self.keep_on_undo(restored, self.viewer.values)
        target.append(self.viewer.values)
        # the next change of the user starts a new step of Undo
        self.lastchangetime = -math.inf
        self.viewer.values = restored
        self.redraw()

    def redraw(self) -> None:
        """Show the values of the viewer, and draw them when Qt has no other events, e.g. after new data of the run."""
        from PySide6 import QtCore

        if self.requestedvalues is None:
            QtCore.QTimer.singleShot(0, self.window, self.draw_requested)
        self.requestedvalues = self.viewer.values
        self.show_values()

    def draw_requested(self) -> None:
        """Start the plot of the last values that the user gave, unless the worker thread is busy."""
        if self.rendering is None and self.task is None:
            self.start_render()

    def start_render(self) -> None:
        """Start a plot of the last values that the user gave in the worker thread."""
        values, self.requestedvalues = self.requestedvalues, None
        if values is None:
            return
        self.renderedvalues = values
        self.renderedfield, self.editedfield = self.editedfield, None
        # the status bar keeps the time of the last plot, and the spinner over the plot shows the plot in progress
        self.renderstart = time.perf_counter()
        self.rendering = self.executor.submit(self.render, values)
        self.rendertimer.start()
        if not self.busytimer.isActive():
            self.busytimer.start()

    def is_busy(self) -> bool:
        """Return True if a plot is in progress or waits, and Cancel Plot can stop the wait."""
        return (self.rendering is not None and not self.discardrendering) or self.requestedvalues is not None

    def cancel(self) -> None:
        """Stop the wait for the plot in progress and for the plots that wait, and keep the plot on the screen.

        A worker thread cannot stop, thus the plot in progress runs to its end, and the queue then discards it. The
        controls show the values of the plot on the screen again.
        """
        if not self.is_busy():
            return
        self.requestedvalues, self.editedfield = None, None
        self.discardrendering = self.rendering is not None
        self.viewer.values = self.drawnvalues
        # Cancel Plot also removes its changes from Undo and Redo
        for values, undovalues, redovalues in reversed(self.pendinghistory):
            if values == self.drawnvalues:
                self.undovalues, self.redovalues = undovalues, redovalues
                break
        self.pendinghistory.clear()
        self.lastchangetime = -math.inf
        # a task, e.g. Reload Data, continues in the worker thread, thus its spinner and its status text stay
        if self.task is None:
            self.busytimer.stop()
            set_plot_busy(self.window, busy=False)
            self.statusbar.drawtime.setText("Plot cancelled")
        self.show_values()

    def show_rendered(self) -> None:
        """Show the plot or the result of the task of the worker thread when it is complete.

        The task or the plot that waits then starts.
        """
        future = self.rendering if self.rendering is not None else self.taskfuture
        if future is not None and not future.done():
            return
        # the timer stops first, because after_draw or on_done can call run_task, which starts the timer again
        self.rendertimer.stop()
        if self.rendering is not None:
            self.show_rendered_plot(self.rendering)
        elif self.taskfuture is not None:
            self.end_task(self.taskfuture)
        # a task waits for the plot in progress, and the newer values of the user wait for the task
        if self.task is not None and self.taskfuture is None:
            self.start_task()
        elif self.task is None and self.requestedvalues is not None:
            self.start_render()
        # the spinner stays during a drag, which starts a new plot at the end of each plot
        if self.rendering is None and self.taskfuture is None:
            self.busytimer.stop()
            set_plot_busy(self.window, busy=False)

    def show_rendered_plot(self, rendering: "Future[Callable[[], str | None]]") -> None:
        """Show the complete plot of the worker thread, or the message of a rejection."""
        self.rendering = None
        if self.discardrendering:
            self.discardrendering, self.renderedfield = False, None
            return
        try:
            message = rendering.result()()
        except Exception as exc:  # ruff:ignore[blind-except]
            # the render wraps the command in run_command_step, thus only a defect of the viewer arrives here
            print_error(traceback.format_exc())
            message = f"{type(exc).__name__}: {get_first_line(str(exc))}"
        if message is None:
            self.drawnvalues = self.renderedvalues
        # newer values of the user stay, and the next plot draws them
        if self.requestedvalues is None:
            self.viewer.values = self.drawnvalues
        self.show_plot_status(message)
        self.after_draw(message)
        # after_draw can give a label a new text, e.g. a new unit, which changes the widths of the labels
        align_section_labels(self.window)

    def run_task(
        self, task: "Callable[[], str | None]", statustext: str, on_done: "Callable[[str | None], None]"
    ) -> bool:
        """Run a task in the worker thread, e.g. Reload Data. Return False if a task is in progress.

        The task starts after the plot in progress, and a new plot waits for the end of the task. Thus a plot never
        reads data that the task replaces. The status bar shows statustext while the task runs. on_done receives the
        message of the task in the thread of the window.
        """
        if self.task is not None:
            return False
        self.task, self.taskstatus, self.on_task_done = task, statustext, on_done
        if self.rendering is None:
            self.start_task()
        return True

    def start_task(self) -> None:
        """Start the task of run_task in the worker thread."""
        if self.task is None:
            return
        self.statusbar.drawtime.setText(self.taskstatus)
        self.taskfuture = self.executor.submit(self.task)
        self.rendertimer.start()
        if not self.busytimer.isActive():
            self.busytimer.start()

    def end_task(self, taskfuture: "Future[str | None]") -> None:
        """Give the message of the complete task to the function of run_task."""
        on_done = self.on_task_done
        self.task, self.taskfuture, self.on_task_done = None, None, None
        # the status of the task replaced the time of the last plot, thus that time shows again
        self.statusbar.drawtime.setText(get_plot_time_text(self.plotseconds))
        try:
            message = taskfuture.result()
        except Exception as exc:  # ruff:ignore[blind-except]
            print_error(traceback.format_exc())
            message = f"{type(exc).__name__}: {get_first_line(str(exc))}"
        if on_done is not None:
            on_done(message)

    def close(self) -> None:
        """Cancel the plots that wait in the worker thread.

        A window calls this function when it closes. A plot or a task in progress runs to its end, and the process
        ends after it. Without this function, each plot that waits also runs before the process ends.
        """
        self.executor.shutdown(wait=False, cancel_futures=True)

    def show_plot_status(self, message: str | None) -> None:
        """Show the time of the plot that ended, its message or its warning, and the values."""
        self.plotseconds = time.perf_counter() - self.renderstart
        self.statusbar.drawtime.setText(get_plot_time_text(self.plotseconds))
        # the readout holds the values of the old plot until the mouse moves again
        self.statusbar.readout.setText("")
        hide_readout_tag(self.window)
        show_status_message(self.statusbar, message, self.viewer.warning)
        show_plot_banner(self.window, message)
        show_plot_area_state(self.window, self.viewer.fig)
        # show_values can replace the field of the plot, e.g. with a new card of a subplot, before the plot ends
        field, self.renderedfield = self.renderedfield, None
        rejectedfield = field if message is not None and field is not None and is_live(field) else None
        if self.errorfield is not None and (message is None or rejectedfield is not None):
            clear_field_error(self.errorfield)
            self.errorfield = None
        if rejectedfield is not None and message is not None:
            mark_field_error(rejectedfield, message)
            self.errorfield = rejectedfield
        self.show_values()


def connect_plot_mouse(
    canvas: "FigureCanvasBase",
    get_frames: "Callable[[], Sequence[mplax.Axes]]",
    get_readout: "Callable[[t.Any, mplax.Axes], str]",
    readoutlabel: "QtWidgets.QLabel",
    on_select: "Callable[[float, float], None]",
    on_reset: "Callable[[], None]",
    can_select: "Callable[[], bool]",
    on_select_y: "Callable[[int, float, float], None] | None" = None,
    on_menu: "Callable[[int, t.Any], None] | None" = None,
    show_tag: "Callable[[t.Any, str], None] | None" = None,
) -> "Callable[[], None]":
    """Give the plot a readout under the pointer, a drag across a frame that selects an x range, and a double-click.

    on_select receives the two x values of a drag, and on_reset receives a double-click on a frame. on_select_y
    receives the index of the frame and the two y values of a drag with the Shift key inside one frame. on_menu
    receives the index of the frame and the matplotlib event of a click with the right button. show_tag receives the
    matplotlib event and the readout, which is empty when the pointer leaves the frames. matplotlib keeps the
    connections in the figure. Call the returned function after the canvas receives a new figure.
    """
    # the data value and the pixel of the start of a drag, the span that shows it, and its frame
    dragstart: tuple[float, float] | None = None
    dragspan: t.Any = None
    dragframeindex = 0
    dragvertical = False

    def get_frame_index(event: t.Any) -> int | None:
        return next((index for index, axis in enumerate(get_frames()) if event.inaxes is axis), None)

    def on_press(event: t.Any) -> None:
        nonlocal dragstart, dragspan, dragframeindex, dragvertical
        frameindex = get_frame_index(event)
        if frameindex is None or event.xdata is None:
            return
        if event.button == 3:
            if on_menu is not None:
                on_menu(frameindex, event)
            return
        if event.button != 1:
            return
        if event.dblclick:
            on_reset()
            return
        dragframeindex = frameindex
        # the canvas has no keyboard focus, thus matplotlib gives no key, and the modifiers hold the Shift key
        dragvertical = "shift" in event.modifiers and on_select_y is not None
        if dragvertical:
            dragstart = (event.ydata, event.y)
            dragspan = event.inaxes.axhspan(event.ydata, event.ydata, color="0.5", alpha=0.3)
        elif can_select():
            dragstart = (event.xdata, event.x)
            dragspan = event.inaxes.axvspan(event.xdata, event.xdata, color="0.5", alpha=0.3)

    def on_motion(event: t.Any) -> None:
        frameindex = get_frame_index(event)
        readout = get_readout(event, event.inaxes) if frameindex is not None and event.xdata is not None else ""
        readoutlabel.setText(readout)
        if show_tag is not None:
            show_tag(event, readout)
        if dragstart is None or dragspan is None or event.xdata is None or frameindex is None:
            return
        # the frames share the x axis but not the y axis, thus a y value of a different frame does not apply
        if dragvertical and frameindex != dragframeindex:
            return
        if dragvertical:
            dragspan.set_y(min(dragstart[0], event.ydata))
            dragspan.set_height(abs(event.ydata - dragstart[0]))
        else:
            dragspan.set_x(min(dragstart[0], event.xdata))
            dragspan.set_width(abs(event.xdata - dragstart[0]))
        canvas.draw_idle()

    def on_release(event: t.Any) -> None:
        nonlocal dragstart, dragspan
        if dragstart is None or dragspan is None:
            return
        # a plot during the drag clears the figure, and the span then went with the old frames
        if dragspan.axes in canvas.figure.axes:
            dragspan.remove()
        start, dragstart, dragspan = dragstart, None, None
        canvas.draw_idle()
        if event.xdata is None or get_frame_index(event) is None:
            return
        # a movement of a few pixels is a click and not a selection
        if dragvertical:
            if on_select_y is not None and get_frame_index(event) == dragframeindex and abs(event.y - start[1]) > 5:
                on_select_y(dragframeindex, *sorted((start[0], event.ydata)))
        elif abs(event.x - start[1]) > 5:
            on_select(*sorted((start[0], event.xdata)))

    connectedfigure: object = None

    def connect_to_figure() -> None:
        nonlocal connectedfigure
        if canvas.figure is connectedfigure:
            return
        connectedfigure = canvas.figure
        canvas.mpl_connect("button_press_event", on_press)
        canvas.mpl_connect("motion_notify_event", on_motion)
        canvas.mpl_connect("button_release_event", on_release)
        if show_tag is not None:
            canvas.mpl_connect("figure_leave_event", lambda event: show_tag(event, ""))

    connect_to_figure()
    return connect_to_figure


def make_readout_tag(canvas: "FigureCanvasQTAgg") -> "Callable[[t.Any, str], None]":
    """Return a function that shows a readout in a small tag next to the pointer on the canvas.

    The function takes a matplotlib mouse event and the readout, and an empty readout hides the tag. Each part of the
    readout goes on a line of its own. The tag goes to the other side of the pointer at the edge of the canvas.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    tag = QtWidgets.QLabel(canvas)
    tag.setObjectName("readouttag")
    tag.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    tag.setStyleSheet(
        "QLabel#readouttag { background: palette(base); color: palette(text); border: 1px solid palette(mid);"
        " border-radius: 4px; padding: 2px 5px; }"
    )
    tag.hide()
    offset = 14

    def show_tag(event: t.Any, readout: str) -> None:
        if not readout or event.x is None:
            tag.hide()
            return
        tag.setText("\n".join(readout.split("   ")))
        tag.adjustSize()
        # matplotlib gives the pixels of the screen from the bottom, and Qt places a widget in points from the top
        ratio = canvas.device_pixel_ratio
        x, y = event.x / ratio, canvas.height() - event.y / ratio
        left = x + offset if x + offset + tag.width() <= canvas.width() else x - offset - tag.width()
        top = y + offset if y + offset + tag.height() <= canvas.height() else y - offset - tag.height()
        tag.move(max(0, round(left)), max(0, round(top)))
        tag.show()
        tag.raise_()

    return show_tag


def hide_readout_tag(window: "QtCore.QObject") -> None:
    """Hide the tag of make_readout_tag, which holds the values of the old plot until the mouse moves again."""
    from PySide6 import QtWidgets

    if (tag := window.findChild(QtWidgets.QLabel, "readouttag")) is not None:
        tag.hide()


def get_line_readouts(axis: "mplax.Axes", x: float) -> list[str]:
    """Return the value at x of each labelled line of the axes, as "value label".

    A line with a label that starts with "_" is not a series of the legend. If no line has a label, the first line
    takes the label of the y axis. A subplot of one variable gives no label to its line.
    """
    lines = [line for line in axis.get_lines() if np.asarray(line.get_xdata()).size >= 2]
    labelledlines = [line for line in lines if not str(line.get_label()).startswith("_")]
    parts: list[str] = []
    for line in labelledlines or lines[:1]:
        label = plain_label(str(line.get_label()) if labelledlines else axis.get_ylabel())
        xdata, ydata = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
        finite = np.isfinite(xdata) & np.isfinite(ydata)
        xdata, ydata = xdata[finite], ydata[finite]
        if xdata.size < 2:
            continue
        # a step line has two points at each edge of a bin, and a stable sort keeps them in the order of the line
        order = np.argsort(xdata, kind="stable")
        if xdata[order[0]] <= x <= xdata[order[-1]]:
            parts.append(f"{np.interp(x, xdata[order], ydata[order]):.4g} {label}")
    return parts


# the readable text of the label or the checkbox of each flag. The tooltip then gives the flag
FLAG_LABELS: t.Final = MappingProxyType({
    "--average_over_phi_angle": "Average over φ",
    "--average_over_theta_angle": "Average over θ",
    "--colorbyion": "Colour by ion",
    "--frompackets": "Read ARTIS data from",
    "--hidenetspectrum": "Hide net spectrum",
    "--hideother": "Hide Other",
    "--hidexlabel": "Hide x label",
    "--histogram": "Histogram",
    "--legendframe": "Legend frame",
    "--markers": "Markers",
    "--nolegend": "Hide legend",
    "--normalised": "Normalise",
    "--nostack": "Unstacked",
    "--notitle": "Hide title",
    "--plotcmf": "Comoving frame",
    "--plotinvalidpart": "Partial times",
    "--showabsorption": "Show absorption",
    "--showbarnes": "Barnes et al. (2016)",
    "--showemission": "Show emission",
    "--shownoise": "Show noise",
    "--use_pellet_decay_time": "Pellet decay time",
    "--use_thermalemissiontype": "Event",
    "--usedegrees": "Degrees",
    "-axis": "Axis",
    "-cell": "Cells",
    "-figscale": "Figure scale",
    "-groupby": "Group by",
    "-labelfontsize": "Label size",
    "-maxseriescount": "Max series",
    "-subplotsperrow": "Subplots per row",
    "-topnucs": "Top nuclides",
    "-x": "x variable",
    "-xbins": "x bins",
    "-xmax": "x max",
    "-xmin": "x min",
    "-xunit": "Unit",
    "-ymax": "y max",
    "-ymin": "y min",
    "-yscale": "y scale",
    "-yvariable": "Quantity",
})


def show_flag_labels(window: "QtWidgets.QWidget") -> None:
    """Give each label and each checkbox of a flag its readable text, and add the flag to its tooltip."""
    from PySide6 import QtWidgets

    widgets: list[QtWidgets.QLabel | QtWidgets.QCheckBox] = [
        *window.findChildren(QtWidgets.QLabel),
        *window.findChildren(QtWidgets.QCheckBox),
    ]
    for widget in widgets:
        flag = widget.text()
        if (text := FLAG_LABELS.get(flag, flag)) != flag:
            # a label names the control after it and ends with a colon, as in Keynote. A checkbox has no colon
            widget.setText(f"{text}:" if isinstance(widget, QtWidgets.QLabel) else text)
            tooltip = widget.toolTip()
            # a label whose text changes later, e.g. -xmin with a unit, already gives the flag in its tooltip
            if tooltip != flag and not tooltip.endswith(f"({flag})"):
                widget.setToolTip(f"{tooltip} ({flag})" if tooltip else flag)


def show_window(window: "QtWidgets.QMainWindow", on_screen: "Callable[[], None]") -> None:
    """Give the window the size of the last window of the viewer, or a first size, and show it.

    make_window and make_central_splitter give the window. on_screen runs after the window moves to a different
    screen.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    show_flag_labels(window)
    # the readable text of each flag sets the width of its label, and the first values of the controls can give a label
    # a unit, thus the alignment waits until Qt has no other events
    QtCore.QTimer.singleShot(0, window, partial(align_section_labels, window))
    splitter = window.centralWidget()
    assert isinstance(splitter, QtWidgets.QSplitter)
    geometrykey, splitterkey = get_window_setting_keys(window)
    settings = get_settings()
    geometry = settings.value(geometrykey)
    splitterstate = settings.value(splitterkey)
    isfirstsize = not (isinstance(geometry, QtCore.QByteArray) and window.restoreGeometry(geometry))
    if not isfirstsize and isinstance(splitterstate, QtCore.QByteArray):
        splitter.restoreState(splitterstate)
    if isfirstsize:
        # a large first window gives the plot more space beside the sidebar. A figure at 100 dpi gave a window of only
        # 1399 x 700 on a screen of 2560 x 1440
        screen = window.screen().availableGeometry()
        # the minimum size is less important than the screen, thus a narrow screen still holds the whole window
        windowwidth = min(max(round(0.8 * screen.width()), SIDEBAR_WIDTH + 600), screen.width())
        window.resize(windowwidth, min(max(round(0.75 * screen.height()), 700), screen.height()))
        splitter.setSizes([windowwidth - SIDEBAR_WIDTH - 40, SIDEBAR_WIDTH])
        windowframe = window.frameGeometry()
        windowframe.moveCenter(screen.center())
        window.move(windowframe.topLeft())
    window.show()
    if isfirstsize:
        # the window system can change the size in window.show(), thus the first size is the size after that call
        window.setProperty("firstsize", window.size())
    if (windowhandle := window.windowHandle()) is not None:

        def on_screen_changed(_screen: QtGui.QScreen) -> None:
            # matplotlib handles the new pixel ratio first, thus the fit waits until Qt has no other events
            QtCore.QTimer.singleShot(0, window, on_screen)

        windowhandle.screenChanged.connect(on_screen_changed)


def make_timer(window: "QtWidgets.QWidget", milliseconds: int) -> "QtCore.QTimer":
    """Return a single-shot timer.

    The window is the parent of the timer, thus the timer stops when the window closes.
    """
    from PySide6 import QtCore

    timer = QtCore.QTimer(window)
    timer.setSingleShot(True)
    timer.setInterval(milliseconds)
    return timer
