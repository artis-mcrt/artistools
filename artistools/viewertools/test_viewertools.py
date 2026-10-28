import argparse
import dataclasses as dc
import importlib
import io
import json
import os
import shlex
import subprocess  # ruff:ignore[suspicious-subprocess-import]
import sys
import threading
import time
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import matplotlib.figure as mplfig
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

import artistools as at
from artistools.viewertools import application as viewerapplication
from artistools.viewertools import core as viewercore
from artistools.viewertools import menus as viewermenus
from artistools.viewertools import sections as viewersections
from artistools.viewertools import series as viewerseries
from artistools.viewertools import widgets as viewerwidgets
from artistools.viewertools import window as viewerwindow
from artistools.viewertools.window import apply_values
from artistools.viewertools.window import draw_values

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def make_viewer(kind: str, tokens: list[str]) -> t.Any:
    """Return the viewer of a command, with a canvas and no window, after its first plot."""
    module = importlib.import_module(f"artistools.{kind}.interactive")
    viewerclass = next(
        value for name, value in vars(module).items() if name.endswith("Viewer") and isinstance(value, type)
    )
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = viewerclass(tokens, fig)
    assert draw_values(viewer) is None
    return viewer


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
def test_viewer_gives_the_standard_spellings_for_older_ones(kind: str) -> None:
    """A command with the older spellings gives a viewer command with the standard spellings, as the help shows them.

    The viewers made their commands with the older spellings, e.g. -figwidthscale, thus a copied command disagreed
    with the help. A session that an older version saved holds such spellings too.
    """
    timetokens = ["-timedays", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath), *timetokens, "-figwidthscale", "2.25", "-label", "mine"])
    command = viewer.get_command()
    assert " --figwidthscale 2.25" in command
    assert " --label mine" in command
    assert " -figwidthscale" not in command
    assert " -label" not in command

    oldsession = ["mymodel", "-figwidthscale", "2", "-t", "300"]
    assert viewerapplication.remove_figwidthscale(oldsession) == ["mymodel", "-t", "300"]


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
def test_viewer_changes_the_residual_baseline(kind: str) -> None:
    """The option table gives the baseline index to the command and keeps the residual panel."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath), str(modelpath), *timetokens, "--residualbaselineseries"])
    assert viewer.values.otheroptions == (("--residualbaselineseries", ()),)
    assert viewer.residualaxis is not None
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=(("--residualbaselineseries", ("1",)),))) is None
    assert viewer.values.otheroptions == (("--residualbaselineseries", ("1",)),)
    assert viewer.residualaxis is not None
    tokens = viewer.get_plot_tokens()
    assert tokens[tokens.index("--residualbaselineseries") + 1] == "1"
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=())) is None
    assert viewer.residualaxis is None


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
@pytest.mark.parametrize("flag", ["-residualbaselineseries", "-residuals", "-residual", "--residuals"])
def test_viewer_keeps_paths_after_a_bare_residual_flag(kind: str, flag: str) -> None:
    """Both viewers keep the paths after a bare residual flag."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    paths = (str(modelpath), str(modelpath))
    viewer = make_viewer(kind, [flag, *paths, *timetokens])
    assert (viewer.values.spectra if kind == "spectra" else viewer.values.lightcurves) == paths
    assert viewer.residualaxis is not None
    values = ("0",) if flag in {"-residualbaselineseries", "-residuals"} else ()
    # the row gives the standard spelling, and -residuals keeps its spelling, because --residuals is a different flag
    shownflag = {"-residualbaselineseries": "--residualbaselineseries", "-residual": "--residual"}.get(flag, flag)
    assert viewer.values.otheroptions == ((shownflag, values),)


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
def test_viewer_selects_the_residual_series(kind: str) -> None:
    """Both viewers use the residual indices and the baseline index from the option table."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath)] * 3 + timetokens + ["-residual"])
    assert viewer.residualaxis is not None
    assert len(viewer.residualaxis.lines) == 3
    options = (("-residual", ("0",)), ("-residualbaselineseries", ("2",)))
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=options)) is None
    assert viewer.residualaxis is not None
    assert len(viewer.residualaxis.lines) == 2
    parser = viewercore.make_parser(
        at.spectra.plotspectra.addargs if kind == "spectra" else at.lightcurve.plotlightcurve.addargs
    )
    args = parser.parse_args(viewer.get_plot_tokens())
    assert args.residuals == [0]
    assert args.residualbaselineseries == 2
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=(("-residual", ()),))) is None
    assert len(viewer.residualaxis.lines) == 3
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=())) is None
    assert viewer.residualaxis is None


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_viewer_selects_the_residual_type(kind: str, residualtype: str) -> None:
    """Both viewers use the residual type from the option table."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath)] * 2 + timetokens + ["--residual"])
    assert viewer.residualaxis is not None
    assert viewer.residualaxis.get_yscale() == "linear"
    assert viewer.residualaxis.get_ylabel() == "series / baseline"
    initialvalues = np.asarray(viewer.residualaxis.lines[0].get_ydata())
    assert np.isfinite(initialvalues).any()
    assert np.allclose(initialvalues[np.isfinite(initialvalues)], 1.0)
    options = (("--residual", ()), ("--residualtype", (residualtype,)))
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=options)) is None
    assert viewer.residualaxis is not None
    assert viewer.residualaxis.get_yscale() == ("log" if residualtype == "relativelog" else "linear")
    line = viewer.residualaxis.lines[0]
    values = np.asarray(line.get_ydata())
    finite = np.isfinite(values)
    assert finite.any()
    assert np.allclose(values[finite], 0.0 if residualtype == "absolute" else 1.0)
    parser = viewercore.make_parser(
        at.spectra.plotspectra.addargs if kind == "spectra" else at.lightcurve.plotlightcurve.addargs
    )
    args = parser.parse_args(viewer.get_plot_tokens())
    assert args.residualtype == residualtype
    action = viewercore.get_actions_by_flag(parser)["--residualtype"]
    assert viewercore.get_option_kind(action) == "choice"
    assert viewercore.get_default_tokens(action) == ("relative",)


@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
def test_viewer_sets_the_residual_y_maximum(kind: str, residualtype: str) -> None:
    """Both viewers pass the residual maximum through the option table and the command."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath)] * 2 + timetokens + ["-residual", "-residualtype", residualtype])
    options = (("-residual", ()), ("-residualtype", (residualtype,)), ("-residualymax", ("2.5",)))
    assert apply_values(viewer, dc.replace(viewer.values, otheroptions=options)) is None
    assert viewer.residualaxis is not None
    assert np.isclose(viewer.residualaxis.get_ylim()[1], 2.5)
    parser = viewercore.make_parser(
        at.spectra.plotspectra.addargs if kind == "spectra" else at.lightcurve.plotlightcurve.addargs
    )
    assert np.isclose(parser.parse_args(viewer.get_plot_tokens()).residualymax, 2.5)


@pytest.mark.parametrize(("kind", "timetokens"), [("spectra", ["-t", "300"]), ("lightcurve", ["--plotcmf"])])
def test_viewer_takes_back_a_folder_of_a_list_option(
    kind: str, timetokens: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model folder that --label took must be a series of the window, as the command gives the folder back.

    The viewer split the tokens itself, thus the spectra window raised an IndexError and the light curve window kept
    the folder as a second label and read the working folder.
    """
    monkeypatch.chdir(tmp_path)
    viewer = make_viewer(kind, ["--label", "My model", str(modelpath), *timetokens, "--interactive"])
    seriespaths = viewer.values.spectra if kind == "spectra" else viewer.values.lightcurves
    assert seriespaths == (str(modelpath),)
    assert viewer.values.otheroptions == (("--label", ("My model",)),)
    # a list option that a control gives, e.g. -fixedionlist, leaves no row, and the parsed folder is the series
    parsed = viewercore.parse_viewer_tokens(
        at.spectra.plotspectra.addargs,
        ["--fixedionlist", "Fe II", str(modelpath), "-t", "300"],
        importlib.import_module("artistools.spectra.interactive").CONTROLLED_DESTS,
    )
    assert parsed.paths == [str(modelpath)]
    assert parsed.otheroptions == ()


@pytest.mark.parametrize(
    ("kind", "tokens"),
    [
        ("spectra", [str(modelpath), "-t", "300"]),
        ("lightcurve", [str(modelpath_classic_3d)]),
        ("estimators", ["Te", str(modelpath), "-timestep", "50"]),
    ],
)
def test_viewer_command_opens_no_file(kind: str, tokens: list[str]) -> None:
    """The command of a window must not hold --show or --open.

    Copy Figure and Export Animation run the command for a temporary file, and --open opened each such file.
    """
    viewer = make_viewer(kind, [*tokens, "--show", "--open", "--interactive"])
    assert {"--show", "--open"}.isdisjoint(viewer.get_plot_tokens())


def test_short_limits_keep_a_narrow_range() -> None:
    """A selected range takes more digits than 3 when it is narrow, and a range of no width gives None.

    3 digits gave the same limit at both ends of a drag of 6561.2 to 6563.9, and nothing happened.
    """
    assert viewercore.get_short_limits(6561.2, 6563.9) == ("6561.2", "6563.9")
    assert viewercore.get_short_limits(10750.0, 10900.0) == ("10750", "10900")
    assert viewercore.get_short_limits(1.0041e-13, 1.0049e-13) == ("1.0041e-13", "1.0049e-13")
    assert viewercore.get_short_limits(2512.3456, 18987.6) == ("2510", "19000")
    assert viewercore.get_short_limits(5.0, 5.0) is None
    assert viewercore.get_short_limits(6.0, 5.0) is None
    messages: list[str] = []
    assert viewersections.read_selected_range(5.0, 5.0, "x", messages.append) is None
    assert messages == ["The selected x range has no width. Drag across a wider range"]


def test_select_y_handler_gives_a_reason_for_no_change() -> None:
    """A Shift-drag that cannot change the y range must give the reason, and a panel below the first frame is quiet."""
    messages: list[str] = []
    limits: list[tuple[str, str]] = []
    showsvalues = [True]

    def set_limits(low: str, high: str) -> None:
        limits.append((low, high))

    on_select_y = viewersections.make_select_y_handler(lambda: showsvalues[0], set_limits, messages.append)
    on_select_y(0, 1.0041e-13, 1.0049e-13)
    assert limits == [("1.0041e-13", "1.0049e-13")]
    on_select_y(1, 1.0, 2.0)
    assert len(limits) == 1
    assert not messages
    showsvalues[0] = False
    on_select_y(0, 1.0, 2.0)
    assert messages == [viewersections.WAIT_FOR_PLOT_MESSAGE]


def test_session_keeps_a_label_that_names_a_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A --label or a --title with the name of a folder stays a text, and the path of the folder becomes absolute.

    The legend and the title of a restored window then showed the full path of the folder.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "model1").mkdir()
    tokens = ["model1", "-t", "300", "--label", "model1", "default", "--title", "model1", "--ymin=-1", "model1"]
    assert viewerapplication.get_absolute_tokens(tokens) == [
        str(tmp_path / "model1"),
        "-t",
        "300",
        "--label",
        "model1",
        "default",
        "--title",
        "model1",
        "--ymin=-1",
        str(tmp_path / "model1"),
    ]


def test_session_window_takes_no_option_of_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window of the last session must open with its saved command, without the -figscale of the Settings window.

    The user can have removed that option in the window, and the restored plot then differed.
    """
    qtcore = pytest.importorskip("PySide6.QtCore", exc_type=ImportError)
    monkeypatch.setattr(viewerapplication, "take_session_windows", lambda: [["mymodel", "-t", "300"]])

    def run_at_once(_milliseconds: int, function: "Callable[[], None]") -> None:
        function()

    monkeypatch.setattr(qtcore.QTimer, "singleShot", run_at_once)
    open_window = mock.Mock(return_value=None)
    viewerapplication.reopen_session_windows(open_window, [])
    open_window.assert_called_once_with(["mymodel", "-t", "300"], [], newwindow=False)


def test_series_label_that_the_command_cannot_give() -> None:
    """The popup of the line properties must refuse a label that the command reads as a flag or as the default.

    The command rejected "-ve" with a message about an ambiguous option, and it gave "default" the automatic label.
    """
    assert viewerseries.get_label_error("My model") is None
    assert viewerseries.get_label_error("") is None
    for label in ("-ve", "--", "default"):
        assert viewerseries.get_label_error(label) is not None


def test_dock_drop_of_a_launch_path_opens_after_the_launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The filter ignores a launch event of a path of the command line, but not a drop after the launch.

    macOS gives no launch event for some paths, e.g. the value of a flag. The filter then ignored the first drop of
    such a path on the Dock icon.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    monkeypatch.setattr(sys, "orig_argv", ["python3", "--modelpath", str(tmp_path)])
    for haslaunchevent in (True, False):
        app = mock.Mock()
        open_folder = mock.Mock()
        viewerapplication.handle_file_open_events(app, open_folder)
        open_path = app.setProperty.call_args.args[1]
        forget_launch_paths = app.applicationStateChanged.connect.call_args.args[0]
        with mock.patch("PySide6.QtWidgets.QApplication.activeWindow", return_value=None):
            if haslaunchevent:
                open_path(str(tmp_path))
            forget_launch_paths(QtCore.Qt.ApplicationState.ApplicationActive)
            open_path(str(tmp_path))
        open_folder.assert_called_once_with(str(tmp_path))


def test_viewer_save_waits_for_the_task_in_progress(tmp_path: Path) -> None:
    """Save Figure gives the command to the worker thread, as Copy Figure does.

    Save ran the command in the window thread while the worker thread drew a plot, and both changed the rcParams.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from artistools.misc.argparser import SuggestingArgumentParser
    from artistools.viewertools import menus as viewermenus

    parser = SuggestingArgumentParser()
    parser.add_argument("--dpi", type=int, default=250)
    commandmain = mock.Mock()
    statusbar = mock.Mock()
    queue = mock.Mock(task=None, run_task=mock.Mock(return_value=True))
    filename = str(tmp_path / "plot.png")
    with mock.patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(filename, "")) as mockdialog:
        viewermenus.save_figure_of_command(
            mock.Mock(), queue, statusbar, commandmain, "plotspectra", [], parser, ("png", 250)
        )
        commandmain.assert_not_called()
        queue.run_task.assert_called_once()

        # a task in progress, e.g. Reload Data, gives a message and no file dialog
        queue.task = mock.Mock()
        viewermenus.save_figure_of_command(
            mock.Mock(), queue, statusbar, commandmain, "plotspectra", [], parser, ("png", 250)
        )
    mockdialog.assert_called_once()
    statusbar.message.setText.assert_called_with("A different task is in progress. Save the figure after it")


# the code that opens a window of plotspectra on the offscreen platform of Qt, and prints the results as JSON
WINDOW_CHECK_CODE: t.Final = """
import json, sys, time
from PySide6 import QtCore, QtTest, QtWidgets
QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, sys.argv[2])
from artistools.spectra import interactive
from artistools.viewertools.application import start_application

app = start_application(interactive.APPLICATION_NAME, interactive.get_icon_curve())
windows = []
assert interactive.open_window([sys.argv[1], "-t", "300", "--frompackets", "-deltax", "12.34567"], windows) is None
window = windows[-1]

def get_command():
    return next(
        box.toPlainText()
        for box in window.findChildren(QtWidgets.QPlainTextEdit)
        if box.toPlainText().startswith("artistools ")
    )

def wait_for(condition):
    end = time.monotonic() + 60.0
    while not condition() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)

results = {}
noisecheck = next(check for check in window.findChildren(QtWidgets.QCheckBox) if check.text() == "Show noise")
noisecheck.setChecked(True)
wait_for(lambda: "--shownoise" in get_command())
results["noisecommand"] = get_command()
window.activateWindow()
xminedit = next(edit for edit in window.findChildren(QtWidgets.QLineEdit) if edit.text() == "2500")
xminedit.setFocus()
for key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_PageDown):
    QtTest.QTest.keyClick(xminedit, key)
deadline = time.monotonic() + 1.0
while time.monotonic() < deadline:
    app.processEvents()
results["keycommand"] = get_command()
results["untipped"] = [
    type(widget).__name__
    for widget in [*window.findChildren(QtWidgets.QAbstractSpinBox), *window.findChildren(QtWidgets.QComboBox)]
    if not widget.toolTip()
]
print(json.dumps(results))
window.close()
"""


def test_viewer_window_keeps_the_typed_bin_width_and_the_keys_of_a_field(tmp_path: Path) -> None:
    """A different control keeps -deltax, the Up key in a text field keeps the time, and each box has a tooltip.

    The bin width box rounded -deltax 12.34567 to 12.346 when the user checked Show noise. The Up key in the field of
    -xmin made the time range one timestep wider. The box of the bin width had no tooltip. A fresh interpreter opens
    the window, because a QCoreApplication of a different test in this process cannot show a widget.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    environment = os.environ | {"QT_QPA_PLATFORM": "offscreen", viewercore.MACOS_BUNDLE_VARIABLE: "1"}
    result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", WINDOW_CHECK_CODE, str(modelpath), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    # the window prints its command when it closes, thus the results are on the line of the JSON object
    results = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"noisecommand"')))
    assert "-deltax 12.34567" in results["noisecommand"]
    assert " -t 300 " in results["keycommand"]
    assert results["untipped"] == []


# the code that changes the palette of the application on the offscreen platform of Qt. It prints the new colour of a
# chip and of the symbol of a button
PALETTE_CHECK_CODE: t.Final = """
from functools import partial
from PySide6 import QtCore, QtGui, QtWidgets
from artistools.viewertools.application import make_window
from artistools.viewertools.menus import refresh_palette_style_sheets
from artistools.viewertools.widgets import make_glyph_button

app = QtWidgets.QApplication([])
window = make_window("palettecheck")
window.setProperty("palettehandler", partial(refresh_palette_style_sheets, window))
panel = QtWidgets.QWidget()
panel.setStyleSheet("QFrame#chip { background: palette(base); }")
chip = QtWidgets.QFrame(panel)
chip.setObjectName("chip")
chip.setFixedSize(20, 20)
button = make_glyph_button("xmark", "Remove", "Remove")
button.setParent(panel)
button.move(30, 0)
window.setCentralWidget(panel)
window.show()
app.processEvents()
palette = QtGui.QPalette(app.palette())
palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#202020"))
palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#ff0000"))
app.setPalette(palette)
app.processEvents()
symbol = button.icon().pixmap(QtCore.QSize(10, 10)).toImage()
pixels = [symbol.pixelColor(x, y) for x in range(symbol.width()) for y in range(symbol.height())]
print(chip.grab().toImage().pixelColor(10, 10).name(), max(pixels, key=QtGui.QColor.alpha).name())
"""


def test_window_takes_the_colours_of_a_new_palette() -> None:
    """A chip and a symbol of plotestimators kept the colours of a light window after a change to Dark Mode.

    Qt reads palette(base) in a style sheet one time only, and a symbol is an image in one colour. Thus the window must
    apply them again. Qt gives the change of the application palette to event() and not to changeEvent().
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    environment = os.environ | {"QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", PALETTE_CHECK_CODE],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().splitlines()[-1] == "#202020 #ff0000"


def test_viewer_status_line_gives_the_error() -> None:
    """The status line gives the error of argparse, and not the usage line that argparse prints before it."""
    stderr = "usage: artistools [options] [specpath ...]\nerror: argument -xmin: invalid float value: 'abc'\nhelp: -h"
    assert viewercore.get_first_line(stderr) == "argument -xmin: invalid float value: 'abc'"
    assert viewercore.get_first_line("A file is missing\nThe second line") == "A file is missing"


def test_viewer_queue_moves_a_clamped_control_back() -> None:
    """A handler that clamps a control to the values of the plot gives unchanged values, and the control must move back.

    The queue returned before it showed the values, thus a slider stayed at a position that the plot did not show.
    """
    # PySide6 is an optional dependency. CI does not install it for each Python version, and a CI machine with no
    # libEGL.so.1 gives an ImportError that is not a ModuleNotFoundError. The spinner and the banner of the queue
    # need QtWidgets, which needs libEGL.so.1, thus each queue test skips without QtWidgets
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    qtcore = pytest.importorskip("PySide6.QtCore", exc_type=ImportError)
    viewer = mock.Mock(values=5)
    showvalues = mock.Mock()
    queue = viewerwindow.DrawQueue(qtcore.QObject(), viewer, mock.Mock(), showvalues, mock.Mock(), render=mock.Mock())
    queue.apply(5)
    showvalues.assert_called_once_with()
    assert queue.requestedvalues is None, "unchanged values must draw no plot"
    queue.close()


def test_viewer_undo_reverts_a_drag_in_one_step_and_skips_a_rejected_change() -> None:
    """One step of Undo reverts all the changes of a drag, and Undo skips a change that the command rejected.

    A rejected change leaves the old values, thus its step of Undo holds the current values and changes nothing.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    qtcore = pytest.importorskip("PySide6.QtCore", exc_type=ImportError)
    viewer = mock.Mock(values=1)
    queue = viewerwindow.DrawQueue(qtcore.QObject(), viewer, mock.Mock(), mock.Mock(), mock.Mock(), render=mock.Mock())
    # the test needs no plot, and a timer of a plot that stays in the process crashed a later test of this worker
    with mock.patch.object(queue, "redraw"):
        # a drag gives a new value at each movement of the mouse
        for values in (2, 3, 4):
            queue.apply(values)
        # the user stops for a time that is longer than the merge time
        queue.lastchangetime -= 10.0 * viewercore.UNDO_MERGE_SECONDS
        queue.apply(5)
        # the command rejected 5, thus the viewer kept the values of the last plot
        viewer.values = 4

        queue.undo()
        assert viewer.values == 1
        queue.redo()
        assert viewer.values == 4
        queue.apply(7)
        assert not queue.can_redo(), "a new change must remove the steps of Redo"
        queue.apply(8, undoable=False)
        queue.undo()
        assert viewer.values == 4, "a change of the window, e.g. a step of Play, must give no step of Undo"
    queue.close()


def test_viewer_cancel_keeps_the_history_of_the_plot_on_the_screen() -> None:
    """Cancel Plot after Undo keeps the step of Undo, because the plot on the screen did not change.

    The cancel returned to the values of the plot but kept the lists of Undo and Redo of the cancelled step. Thus
    Undo lost the step back to the earlier values.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    viewer = mock.Mock(values=1, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), mock.Mock(), render=mock.Mock())
    with mock.patch.object(queue, "redraw"), mock.patch.object(viewerwindow, "set_plot_busy"):
        queue.apply(2)
        # the plot of 2 is on the screen, then Undo asks for a plot of 1, which waits
        queue.drawnvalues = 2
        queue.undo()
        assert viewer.values == 1
        queue.requestedvalues = 1
        queue.cancel()
        assert viewer.values == 2
        assert queue.can_undo()
        assert not queue.can_redo()
        queue.undo()
        assert viewer.values == 1
    queue.close()


def test_viewer_undo_ignores_the_parts_that_the_window_sets() -> None:
    """A step of Undo that differs only in a part that the window sets, e.g. the width of the figure, is no step.

    keep_on_undo gives such a part its current value. Undo compared the entries before it applied keep_on_undo.
    Thus a resize of the window after a rejected change enabled Undo, and Undo only drew the same plot again.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    @dc.dataclass(frozen=True, kw_only=True)
    class Values:
        time: int
        figwidthscale: float
        dpi: int | None = None

    viewer = mock.Mock(values=Values(time=1, figwidthscale=1.0))
    queue = viewerwindow.DrawQueue[Values](
        QtCore.QObject(),
        viewer,
        mock.Mock(),
        mock.Mock(),
        mock.Mock(),
        render=mock.Mock(),
        keep_on_undo=viewercore.keep_figwidthscale,
    )
    with mock.patch.object(queue, "redraw") as mockredraw:
        queue.apply(Values(time=2, figwidthscale=1.0))
        # the command rejected the change, thus the viewer kept the values of the plot
        viewer.values = Values(time=1, figwidthscale=1.0)
        queue.apply(Values(time=1, figwidthscale=1.3), undoable=False)
        assert not queue.can_undo()
        mockredraw.reset_mock()
        queue.undo()
        assert viewer.values == Values(time=1, figwidthscale=1.3)
        mockredraw.assert_not_called()
    queue.close()


def test_viewer_rejection_marks_the_field_of_its_own_plot() -> None:
    """A rejection marks the text field that gave its own change, and a field that Qt deleted gets no mark.

    The queue kept one field for all the plots, thus a second edit during a plot took the mark of the first plot. A
    new card of a subplot replaces its fields, and a mark of the deleted field raised an error, so the window then
    showed the rejected values.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def render(values: int) -> Callable[[], str | None]:
        time.sleep(0.2)
        return lambda: f"rejected {values}" if values >= 2 else None

    fields = [mock.Mock(name=f"field{index}") for index in range(4)]
    deletedfield = fields[3]
    viewer = mock.Mock(values=0, warning="")
    showvalues = mock.Mock()
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), showvalues, mock.Mock(), render=render)

    def wait_for_plots() -> None:
        deadline = time.perf_counter() + 10.0
        while (queue.rendering is not None or queue.requestedvalues is not None) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    def is_live(field: object) -> bool:
        return field is not deletedfield

    with (
        mock.patch.object(viewerwindow, "get_edited_field", side_effect=fields),
        mock.patch.object(viewerwindow, "is_live", side_effect=is_live),
        mock.patch.object(viewerwindow, "mark_field_error") as mockmark,
        mock.patch.object(viewerwindow, "clear_field_error"),
        mock.patch.object(viewerwindow, "set_plot_busy"),
        mock.patch.object(viewerwindow, "show_plot_banner"),
    ):
        queue.apply(1)
        app.processEvents()
        # the plot of 1 is in progress, and the command accepts it. The command rejects the next change
        queue.apply(2)
        wait_for_plots()
        mockmark.assert_called_once_with(fields[1], "rejected 2")
        queue.apply(3)
        wait_for_plots()
        assert mockmark.call_args == mock.call(fields[2], "rejected 3")
        showvalues.reset_mock()
        queue.apply(4)
        wait_for_plots()
        assert mockmark.call_count == 2, "a field that Qt deleted must get no mark"
        assert showvalues.called, "the window must show the values of the plot after a rejection"
    queue.close()


def test_viewer_default_options_fill_only_the_options_that_the_command_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Settings window gives a new window --figscale and --labelfontsize, and an option of the command has priority.

    plotspectra has no --labelfontsize, thus its window gets no such option.
    """
    settings = {"default-figscale": 1.5, "default-labelfontsize": 12.0}

    def get_float_setting(key: str, default: float) -> float:
        return settings.get(key, default)

    monkeypatch.setattr(viewermenus, "get_float_setting", get_float_setting)
    estimatorparser = viewercore.make_parser(at.estimators.addargs)
    assert viewermenus.add_default_options(estimatorparser, ["Te", "--figscale", "2"]) == [
        "--labelfontsize",
        "12",
        "Te",
        "--figscale",
        "2",
    ]
    spectraparser = viewercore.make_parser(at.spectra.plotspectra.addargs)
    assert viewermenus.add_default_options(spectraparser, ["mymodel"]) == ["--figscale", "1.5", "mymodel"]


def test_viewer_thread_output_keeps_the_output_of_each_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker thread hides the output of its plot, and the window thread still prints to the terminal.

    contextlib.redirect_stdout changed the stream of each thread, thus a print of the window went to the plot.
    """
    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stdout", viewercore.ThreadOutput(terminal))
    monkeypatch.setattr(sys, "stderr", viewercore.ThreadOutput(io.StringIO()))
    plotoutput = io.StringIO()
    inside, printed = threading.Event(), threading.Event()

    def plot() -> None:
        with viewercore.send_output(plotoutput, io.StringIO()):
            print("a line of the plot")
            inside.set()
            printed.wait(timeout=10)

    worker = threading.Thread(target=plot)
    worker.start()
    assert inside.wait(timeout=10)
    print("a line of the window")
    printed.set()
    worker.join()
    assert plotoutput.getvalue() == "a line of the plot\n"
    assert terminal.getvalue() == "a line of the window\n"


def test_viewer_cancel_discards_the_plot_in_progress() -> None:
    """Cancel Plot keeps the plot on the screen, and the plot in progress does not replace it when it ends.

    A change after Cancel Plot waits for the end of the discarded plot, and then the queue draws it.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    shown: list[int] = []

    def render(values: int) -> "Callable[[], str | None]":
        time.sleep(0.3)

        def show() -> str | None:
            shown.append(values)
            return None

        return show

    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), mock.Mock(), render=render)

    def wait_for_plots() -> None:
        deadline = time.perf_counter() + 10.0
        while (queue.rendering is not None or queue.requestedvalues is not None) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    queue.apply(1)
    app.processEvents()
    assert queue.is_busy()
    queue.cancel()
    assert viewer.values == 0, "the controls must show the values of the plot on the screen"
    assert not queue.is_busy()
    wait_for_plots()
    assert not shown, "the discarded plot must not replace the plot on the screen"
    queue.apply(2)
    app.processEvents()
    queue.cancel()
    queue.apply(3)
    wait_for_plots()
    assert shown == [3]
    assert queue.drawnvalues == viewer.values == 3
    queue.close()


def test_viewer_queue_draws_in_a_worker_thread() -> None:
    """The window stays free during a plot, and a drag during a plot gives a plot of the last values alone.

    The window thread drew each plot, thus a drag of the time slider stopped until the plot ended.
    """
    # the queue needs the timers of Qt alone, and a QCoreApplication loads no plugin of a display
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    renderthreads: list[str] = []
    showthreads: list[str] = []
    rendered: list[int] = []
    # the plot waits until the test releases it. A limit of the wall time failed on a busy machine
    releaseplot = threading.Event()

    def render(values: int) -> Callable[[], str | None]:
        renderthreads.append(threading.current_thread().name)
        rendered.append(values)
        assert releaseplot.wait(timeout=10.0), "the test did not release the plot"

        def show() -> str | None:
            showthreads.append(threading.current_thread().name)
            return "rejected" if values < 0 else None

        return show

    viewer = mock.Mock(values=0)
    afterdraw = mock.Mock()
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), afterdraw, render=render)

    def wait_until(condition: Callable[[], bool]) -> None:
        deadline = time.perf_counter() + 10.0
        while not condition() and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    def plots_ended() -> bool:
        return queue.rendering is None and queue.requestedvalues is None

    queue.apply(1)
    wait_until(lambda: rendered == [1])
    # the plot of 1 waits for the release, thus the window thread runs these changes while the plot is in progress
    queue.apply(2)
    queue.apply(3)
    app.processEvents()
    assert rendered == [1], "a change during a plot must wait for the end of that plot"
    assert viewer.values == 3, "the window must show the values of a change at once"
    releaseplot.set()
    wait_until(plots_ended)
    assert rendered == [1, 3]
    assert "MainThread" not in renderthreads
    assert showthreads == ["MainThread", "MainThread"]
    assert viewer.values == queue.drawnvalues == 3

    # a rejection keeps the values of the last plot
    queue.apply(-1)
    wait_until(plots_ended)
    assert viewer.values == queue.drawnvalues == 3
    afterdraw.assert_called_with("rejected")
    queue.close()


def test_viewer_open_model_gives_the_reason_of_the_new_window() -> None:
    """The status line of Open Model must give the error of the first plot of the new window.

    open_window returned only a bool, thus the message took the first line of the traceback on stderr.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

    def open_window(tokens: Sequence[str], windows: Sequence[object]) -> str:
        sys.stderr.write("Traceback (most recent call last):\n")
        return f"ComputeError: the query failed for {tokens[0]} and {len(windows)} window"

    with mock.patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="mymodel"):
        message = viewermenus.open_model_window(mock.Mock(), open_window, [mock.Mock()])
    assert message == "The viewer cannot open mymodel: ComputeError: the query failed for mymodel and 1 window"


def test_viewer_status_line_reads_text_in_colour() -> None:
    """The library rich colours its text under FORCE_COLOR also in a capture, and the status line must read it.

    A colour code in front of "error: " hid the error, thus the status line showed the usage line of argparse.
    """
    warning = "\x1b[1;33mwarning: Every Te value is below the requested minimum\x1b[0m\n"
    assert viewercore.get_last_warning(warning) == "Every Te value is below the requested minimum"
    error = "\x1b[1;34musage: \x1b[0martistools [options]\n\x1b[1;31merror: \x1b[0m'q=1' is not a plane or a line\n"
    assert viewercore.get_first_line(error) == "'q=1' is not a plane or a line"


def test_viewer_shift_drag_selects_a_y_range_in_one_frame() -> None:
    """The embedded canvas receives no key events, thus the Shift key comes from the modifiers of the mouse event.

    A release in a different frame gives a y value of a different scale, thus it selects nothing.
    """
    from matplotlib.backend_bases import MouseButton
    from matplotlib.backend_bases import MouseEvent

    fig = mplfig.Figure()
    canvas = FigureCanvasAgg(fig)
    frames = fig.subplots(2, 1)
    frames[0].set_ylim(0.0, 10.0)
    frames[1].set_ylim(1e5, 1e9)
    frames[1].set_yscale("log")
    canvas.draw()
    xselections: list[tuple[float, float]] = []
    yselections: list[tuple[int, float, float]] = []
    viewerwindow.connect_plot_mouse(
        canvas,
        get_frames=lambda: list(frames),
        get_readout=lambda _event, _frame: "",
        readoutlabel=mock.MagicMock(),
        on_select=lambda low, high: xselections.append((low, high)),
        on_reset=lambda: None,
        can_select=lambda: True,
        on_select_y=lambda index, low, high: yselections.append((index, low, high)),
    )

    def drag(startframe: int, starty: float, endframe: int, endy: float) -> None:
        x0, y0 = frames[startframe].transData.transform((0.5, starty))
        x1, y1 = frames[endframe].transData.transform((0.5, endy))
        for name, x, y in (
            ("button_press_event", x0, y0),
            ("motion_notify_event", x1, y1),
            ("button_release_event", x1, y1),
        ):
            canvas.callbacks.process(
                name, MouseEvent(name, canvas, x, y, button=MouseButton.LEFT, modifiers=frozenset({"shift"}))
            )

    drag(0, 2.0, 0, 8.0)
    assert len(yselections) == 1
    assert yselections[0][0] == 0
    assert np.allclose(yselections[0][1:], (2.0, 8.0), rtol=1e-6, atol=0.0)
    drag(0, 6.0, 1, 1e7)
    assert len(yselections) == 1
    assert not xselections


def test_viewer_save_gives_the_resolution_of_the_command(tmp_path: Path) -> None:
    """Each format of file takes the resolution of the Figure section.

    The estimator viewer dropped --dpi, thus a PDF file lost the resolution of its colour image.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    dpi = 300
    savedtokens: list[list[str]] = []

    def commandmain(argsraw: Sequence[str]) -> None:
        savedtokens.append(list(argsraw))
        Path(argsraw[-1]).write_text("figure", encoding="utf-8")

    from artistools.misc.argparser import SuggestingArgumentParser

    parser = SuggestingArgumentParser()
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--figscale", type=float, default=1.0)

    def run_task(task: Callable[[], str | None], _statustext: str, on_done: Callable[[str | None], None]) -> bool:
        on_done(task())
        return True

    queue = mock.Mock(task=None, run_task=run_task)
    statusbar = mock.Mock()
    for suffix in ("png", "pdf"):
        filename = str(tmp_path / "plot")

        with mock.patch("PySide6.QtWidgets.QFileDialog.getSaveFileName", return_value=(filename, "")):
            viewermenus.save_figure_of_command(
                mock.Mock(), queue, statusbar, commandmain, "plotspectra", ["--xmin", "5"], parser, (suffix, dpi)
            )
        # a name with no suffix takes the suffix of the selected format
        assert savedtokens[-1] == ["--xmin", "5", "--dpi", "300", "-o", f"{filename}.{suffix}"]
        statusbar.message.setText.assert_called_with(f"Saved {filename}.{suffix}")


def test_viewer_copy_gives_the_file_of_the_selected_format() -> None:
    """Copy Figure runs the command for a file of the selected format, and puts the file on the clipboard.

    Qt gave only a TIFF image to the clipboard of macOS, thus a copy could not give a PDF or an SVG file.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from artistools.misc.argparser import SuggestingArgumentParser

    parser = SuggestingArgumentParser()
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--figscale", type=float, default=1.0)
    commands: list[list[str]] = []

    def commandmain(argsraw: Sequence[str]) -> None:
        commands.append(list(argsraw))
        Path(argsraw[-1]).write_text("<svg/>", encoding="utf-8")

    def run_task(task: Callable[[], str | None], _statustext: str, on_done: Callable[[str | None], None]) -> bool:
        on_done(task())
        return True

    queue = mock.Mock(run_task=run_task)
    statusbar = mock.Mock()
    with mock.patch.object(viewermenus, "put_file_on_clipboard", return_value=None) as mockclipboard:
        viewermenus.copy_figure_of_command(
            queue, statusbar, commandmain, parser, ["--xmin", "5", "--dpi", "300"], ("svg", 150)
        )
    assert commands[0][:4] == ["--xmin", "5", "--dpi", "150"]
    assert commands[0][-1].endswith("figure.svg")
    mockclipboard.assert_called_once_with(b"<svg/>", "svg")
    statusbar.message.setText.assert_called_with("Copied the figure as SVG")


def test_viewer_row_wraps_its_groups() -> None:
    """A group of a row goes to a new line when the line is full, and a hidden group takes no place and no gap."""
    assert viewerwidgets.get_wrapped_lines([100, 100, 100], 250, 12) == [[0, 1], [2]]
    assert viewerwidgets.get_wrapped_lines([100, 0, 100], 212, 12) == [[0, 1, 2]]
    # a group wider than the line has a line of its own
    assert viewerwidgets.get_wrapped_lines([50, 400, 50], 300, 12) == [[0], [1], [2]]


def test_viewer_queue_runs_a_task_between_plots() -> None:
    """A task of the worker thread, e.g. Reload Data, waits for the plot in progress, and a new plot waits for it.

    A plot that the user asked for during a reload took the time of the reload as its plot time. Play then made no
    pause.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    events: list[str] = []

    def render(values: int) -> Callable[[], str | None]:
        events.append(f"plot {values}")
        time.sleep(0.05)
        return lambda: None

    def task() -> str | None:
        events.append("task")
        time.sleep(0.5)
        return None

    statusbar = mock.Mock()
    ondone = mock.Mock()
    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, statusbar, mock.Mock(), mock.Mock(), render=render)
    queue.apply(1)
    app.processEvents()
    assert queue.run_task(task, "Reload in progress...", ondone)
    assert not queue.run_task(task, "Reload in progress...", ondone), "a second task must wait for the first"
    queue.apply(2)
    deadline = time.perf_counter() + 10.0
    while (
        queue.rendering is not None or queue.task is not None or queue.requestedvalues is not None
    ) and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.005)
    assert events == ["plot 1", "task", "plot 2"]
    ondone.assert_called_once_with(None)
    assert queue.plotseconds < 0.4, "the time of the task is not the time of a plot"
    assert "Reload in progress..." in [call.args[0] for call in statusbar.drawtime.setText.call_args_list]
    queue.close()


def test_viewer_typed_centre_gives_back_the_range() -> None:
    """The centre that the time field shows gives back the same range of timesteps, also for an even count.

    The viewer took the timestep that holds the centre as the middle, and an even range then moved one timestep.
    """
    tmids = at.get_timestep_times(at.get_path("testdata") / "testmodel", loc="mid")
    for count in (1, 2, 3, 4):
        for start in range(len(tmids) - count + 1):
            centre = float(f"{(tmids[start] + tmids[start + count - 1]) / 2.0:.4g}")
            assert viewercore.get_nearest_range_start(tmids, centre, count) == start, (count, start)


def test_viewer_option_rows_split_a_group_of_switches() -> None:
    """Argparse reads -qv as -q and -v, and the option table must give each switch its own row.

    The table read -qv as the flag -q with the value "v", and the command then held a stray positional argument.
    """
    parser = viewercore.make_parser(at.estimators.addargs)
    rows, othertokens = viewercore.split_option_rows(parser, ["Te", "mymodel", "-qv", "-qt300", "--xmin", "5"])
    assert rows == (("--quiet", ()), ("--verbose", ()), ("--quiet", ()), ("--timedays", ("300",)), ("--xmin", ("5",)))
    assert othertokens == ["Te", "mymodel"]


def test_viewer_removes_an_option_of_two_values_with_its_values_alone() -> None:
    """An option of nargs 2 takes two values, thus the tokens after them stay in the command.

    remove_options took each token up to the next flag, as for nargs "*", thus the command lost a path after the values.
    """
    parser = viewercore.make_parser(at.spectra.plotspectra.addargs)
    tokens = ["--emissionradialvelocityrange", "1000", "2000", "mymodel", "--xmin", "5"]
    assert viewercore.remove_options(parser, tokens, {"emissionradialvelocityrange"}) == ["mymodel", "--xmin", "5"]
    rows, othertokens = viewercore.split_option_rows(parser, tokens)
    assert rows == (("--emissionradialvelocityrange", ("1000", "2000")), ("--xmin", ("5",)))
    assert othertokens == ["mymodel"]


# each viewer with the arguments of a plot that sets several of its controls
VIEWER_CASES: t.Final = (
    ("spectra", [str(modelpath), "-t", "300", "-xmin", "3000", "-label", "model", "--interactive"]),
    (
        "lightcurve",
        [
            str(modelpath_classic_3d),
            "-deposition",
            "gamma",
            "-thermalisation",
            "gamma",
            "--showbarnes",
            "--interactive",
        ],
    ),
    ("estimators", ["Te", str(modelpath), "-timestep", "50", "--interactive"]),
)


@pytest.mark.parametrize(("kind", "timetokens"), [("spectra", ["-t", "300"]), ("lightcurve", [])])
def test_viewer_takes_the_model_of_the_path_option(
    kind: str, timetokens: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A viewer must plot the model of -modelpath, and not read the working folder, which is not a model folder."""
    monkeypatch.chdir(tmp_path)
    viewer = make_viewer(kind, ["-modelpath", str(modelpath), *timetokens, "--interactive"])
    seriespaths = viewer.values.spectra if kind == "spectra" else viewer.values.lightcurves
    assert seriespaths == (str(modelpath),)
    assert "-modelpath" not in viewer.get_command()


@pytest.mark.parametrize(("kind", "tokens"), VIEWER_CASES)
def test_viewer_command_opens_the_same_plot(kind: str, tokens: list[str]) -> None:
    """The command that a viewer shows must open a viewer with the same command.

    A user copies the command of a window and runs it later, e.g. in a script. An option that the window gives but
    does not read back, or reads in a different way, then gives a different plot.
    """
    viewer = make_viewer(kind, tokens)
    command = viewer.get_command()
    again = make_viewer(kind, [*shlex.split(command)[2:], "--interactive"])
    assert again.get_command() == command


@pytest.mark.parametrize(("kind", "tokens"), VIEWER_CASES)
def test_viewer_keeps_its_plot_after_a_rejected_change(kind: str, tokens: list[str]) -> None:
    """A change that the command rejects must keep the values and the figure of the last plot, and give the reason."""
    viewer = make_viewer(kind, tokens)
    oldvalues, oldfig = viewer.values, viewer.fig
    badvalues = dc.replace(oldvalues, otheroptions=(*oldvalues.otheroptions, ("-figscale", ("abc",))))
    message = apply_values(viewer, badvalues)
    assert message is not None
    assert "-figscale" in message
    assert viewer.values == oldvalues
    assert viewer.fig is oldfig


def test_figure_shows_data_only_inside_the_axis_limits() -> None:
    """The window shows the note of an empty plot when no point of a line is inside the limits of its axes."""
    fig = mplfig.Figure()
    axis = fig.add_subplot()
    axis.plot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert viewerwidgets.figure_shows_data(fig)
    axis.set_xlim(10.0, 20.0)
    assert not viewerwidgets.figure_shows_data(fig)
    # an inverted axis has its limits in the other order, and a point inside them shows
    axis.set_xlim(3.5, 0.5)
    assert viewerwidgets.figure_shows_data(fig)
    axis.set_ylim(5.0, 6.0)
    assert not viewerwidgets.figure_shows_data(fig)
    axis.imshow([[1.0, 2.0]])
    assert viewerwidgets.figure_shows_data(fig)
    # a segment crosses the frame between two points, e.g. a time range between two timesteps
    crossfig = mplfig.Figure()
    crossaxis = crossfig.add_subplot()
    crossaxis.plot([0.0, 10.0], [0.0, 100.0])
    crossaxis.set_xlim(4.0, 6.0)
    assert viewerwidgets.figure_shows_data(crossfig)
    crossaxis.set_ylim(70.0, 90.0)
    assert not viewerwidgets.figure_shows_data(crossfig)


def test_direction_list_maps_all_directions_to_no_direction_option() -> None:
    """All the directions alone give no direction option, and the virtual packets never combine with the average."""
    choose = viewersections.get_new_direction_choice
    nooption = viewersections.DirectionChoice(kind="", bins=(), usedegrees=False)
    assert choose("bin", (-1,), (-1,), usedegrees=False, onebin=False) == nooption
    assert choose("phi", (-1, 3), (3,), usedegrees=False, onebin=False).bins == (-1, 3)
    # the virtual packets have no average, thus all the directions give the plot of the real packets
    assert choose("vpkt", (-1, 2), (-1,), usedegrees=False, onebin=False) == nooption
    assert choose("vpkt", (-1, 2), (2,), usedegrees=False, onebin=False).bins == (2,)
    # the average with no observer of the virtual packets is all the directions, and not a list with no bin
    assert choose("vpkt", (-1,), (), usedegrees=False, onebin=False) == nooption
    # a plot of one bin takes the bin of the click
    assert choose("bin", (0, 5), (5,), usedegrees=False, onebin=True).bins == (5,)
    assert choose("bin", (-1, 5), (-1,), usedegrees=False, onebin=True) == nooption


def test_mathtext_lock_wraps_the_parser_one_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lock of the mathtext parser must install on each Python version and keep the parse of matplotlib.

    The wrapper had annotations that Python 3.13 evaluates at once, and MathTextParser takes no subscript, thus no
    viewer opened on Python 3.13.
    """
    import matplotlib.mathtext as mplmathtext

    # monkeypatch restores the parse of matplotlib after the test
    monkeypatch.setattr(mplmathtext.MathTextParser, "parse", mplmathtext.MathTextParser.parse)
    viewerapplication.serialise_mathtext_parser()
    wrapped = mplmathtext.MathTextParser.parse
    viewerapplication.serialise_mathtext_parser()
    assert mplmathtext.MathTextParser.parse is wrapped
    width, height, *_ = mplmathtext.MathTextParser("path").parse(r"$10^{-3}$", dpi=72)
    assert width > 0
    assert height > 0


def test_direction_data_follows_the_files_and_the_data_source() -> None:
    """A direction bin needs a *_res.out file or the packets with --frompackets, and an observer needs vpkt.txt.

    The commands read the packets for a direction bin only with --frompackets. The viewer enabled the bins of a model
    with only the packets, and the plot then showed all the directions. It disabled the observers of a model with only
    the vspecpol files.
    """
    has_data = viewercore.model_has_direction_data
    datafolder = at.get_path("testdata")
    lightcurveres = ("light_curve_res.out",)
    # testmodel has the packets files and no light_curve_res.out
    assert not has_data(datafolder / "testmodel", "bin", lightcurveres, "", frompackets=False)
    assert has_data(datafolder / "testmodel", "bin", lightcurveres, "", frompackets=True)
    # vspecpolmodel has vpkt.txt and the vspecpol files, and no packets
    assert has_data(datafolder / "vspecpolmodel", "vpkt", ("spec_res.out",), "vspecpol*", frompackets=False)
    assert not has_data(datafolder / "vspecpolmodel", "vpkt", lightcurveres, "", frompackets=True)
    # vpktcontrib has the virtual packets
    assert has_data(datafolder / "vpktcontrib", "vpkt", lightcurveres, "", frompackets=True)
    assert not has_data(datafolder / "vpktcontrib", "vpkt", lightcurveres, "", frompackets=False)


def test_series_styles_keep_the_values_after_the_paths() -> None:
    """A style value of a series that the command adds after the paths, e.g. --obsspec, must stay on that series."""
    rows = (("--label", ("A", "B", "R")),)
    # a new order of the two models keeps R on the third series
    assert viewercore.move_series_styles(rows, ("a", "b"), ("b", "a")) == (("--label", ("B", "A", "R")),)
    # a new model goes before the series of the command
    assert viewercore.move_series_styles(rows, ("a", "b"), ("a", "b", "c")) == (
        ("--label", ("A", "B", "default", "R")),
    )
    assert viewercore.set_series_rows(rows, ("a", "b"), "a", {"--label": "N"}) == (("--label", ("N", "B", "R")),)


def test_viewer_queue_polls_a_task_that_after_draw_starts() -> None:
    """A task that after_draw starts must end, and a later plot must start after it.

    show_rendered stopped the timer after after_draw. A task that after_draw started then had no timer, and the queue
    refused each later plot.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    ondone = mock.Mock()
    rendered: list[int] = []

    def render(values: int) -> Callable[[], str | None]:
        rendered.append(values)
        return lambda: None

    def after_draw(_message: str | None) -> None:
        if not ondone.called and queue.task is None:
            queue.run_task(lambda: None, "Task in progress...", ondone)

    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, mock.Mock(), mock.Mock(), after_draw, render=render)

    def wait_for_queue() -> None:
        deadline = time.perf_counter() + 10.0
        while (
            queue.rendering is not None or queue.task is not None or queue.requestedvalues is not None
        ) and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.005)

    with mock.patch.object(viewerwindow, "set_plot_busy"), mock.patch.object(viewerwindow, "show_plot_banner"):
        queue.apply(1)
        wait_for_queue()
        ondone.assert_called_once_with(None)
        queue.apply(2)
        wait_for_queue()
    assert rendered == [1, 2]
    queue.close()


def test_viewer_cancel_keeps_the_status_of_a_task() -> None:
    """Cancel Plot during a task, e.g. Reload Data, keeps the spinner and the status text of the task.

    The cancel hid the spinner and showed "Plot cancelled" while the task still ran in the worker thread.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    from PySide6 import QtCore

    statusbar = mock.Mock()
    viewer = mock.Mock(values=0, warning="")
    queue = viewerwindow.DrawQueue(QtCore.QObject(), viewer, statusbar, mock.Mock(), mock.Mock(), render=mock.Mock())
    with mock.patch.object(viewerwindow, "set_plot_busy") as mockbusy:
        queue.task = lambda: None
        # a change of the user during the task waits for the end of the task
        queue.requestedvalues = 1
        queue.cancel()
        assert queue.requestedvalues is None
        mockbusy.assert_not_called()
        assert mock.call("Plot cancelled") not in statusbar.drawtime.setText.call_args_list
        queue.task = None
        queue.requestedvalues = 1
        queue.cancel()
        mockbusy.assert_called_once()
        statusbar.drawtime.setText.assert_called_with("Plot cancelled")
    queue.close()


def test_viewer_session_keeps_an_empty_token() -> None:
    """An empty value, e.g. of -label "", stays empty in the command of the next start.

    Path("") is the working folder, and it exists. Thus the empty label became the path of the working folder.
    """
    tokens = viewerapplication.get_absolute_tokens([str(modelpath), "-label", "", "-title", ""])
    assert tokens == [str(modelpath.absolute()), "-label", "", "-title", ""]


def test_viewer_starts_on_a_qt_platform_with_no_display() -> None:
    """A Qt platform that needs no display, e.g. offscreen or vnc, starts with no DISPLAY.

    The viewer stopped whenever DISPLAY and WAYLAND_DISPLAY were not set, and QT_QPA_PLATFORM=offscreen then failed.
    """
    needs_missing_display = viewerapplication.needs_missing_display
    assert needs_missing_display({})
    assert needs_missing_display({"QT_QPA_PLATFORM": "xcb"})
    assert needs_missing_display({"QT_QPA_PLATFORM": "wayland;xcb"})
    assert not needs_missing_display({"DISPLAY": ":0"})
    assert not needs_missing_display({"WAYLAND_DISPLAY": "wayland-0"})
    assert not needs_missing_display({"QT_QPA_PLATFORM": "offscreen"})
    assert not needs_missing_display({"QT_QPA_PLATFORM": "vnc:size=1280x800"})
    assert not needs_missing_display({"QT_QPA_PLATFORM": "wayland;eglfs"})


def test_viewer_bundle_keeps_a_copy_of_the_executable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A copy of the Python executable in the macOS bundle stays at the next start if the executable is the same.

    A copy on a different volume is not the same file as the executable. Thus each start copied the executable again.
    """
    baseexecutable = tmp_path / "python3"
    baseexecutable.write_bytes(b"executable")
    monkeypatch.setattr(sys, "executable", str(baseexecutable))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(viewerapplication, "LSREGISTER", tmp_path / "nolsregister")

    def refuse_hardlink(_path: Path, _target: Path) -> None:
        msg = "a hard link must be on the same volume"
        raise OSError(msg)

    monkeypatch.setattr(Path, "hardlink_to", refuse_hardlink)
    executable = viewerapplication.get_macos_bundle_executable("Viewer", ("public.folder",))
    assert executable.read_bytes() == b"executable"
    inode = executable.stat().st_ino
    assert viewerapplication.get_macos_bundle_executable("Viewer", ("public.folder",)) == executable
    assert executable.stat().st_ino == inode, "the same executable must not give a new copy"

    # a new Python executable replaces the copy
    baseexecutable.write_bytes(b"new executable")
    viewerapplication.get_macos_bundle_executable("Viewer", ("public.folder",))
    assert executable.read_bytes() == b"new executable"


def test_viewer_reload_clears_the_frequency_grid(tmp_path: Path) -> None:
    """Reload Data clears the cache of the frequency grid of spec.out, as it clears the cache of the spectra.

    exspec can write spec.out again with a different grid. The viewer then read the new spectra with the old grid.
    """

    def write_spec(nus: Sequence[float]) -> None:
        lines = ["0 1.0 2.0", *(f"{nu:g} 0.0 0.0" for nu in nus)]
        (tmp_path / "spec.out").write_text("\n".join(lines) + "\n")

    write_spec([1e14, 2e14])
    assert len(at.misc.get_nu_grid(tmp_path)) == 2
    write_spec([1e14, 2e14, 3e14, 4e14])
    viewerwindow.clear_output_caches_of_model(tmp_path, estimatorsonly=False)
    assert len(at.misc.get_nu_grid(tmp_path)) == 4


def test_viewer_command_tokens_read_the_words_of_the_dispatcher() -> None:
    """A call of the dispatcher from Python code gives the words of that call, without the name of the subcommand."""
    args = argparse.Namespace(dispatcherargsraw=["plotspectra", "mymodel", "--interactive"])
    assert viewercore.get_command_tokens(args, None, {}, fromdispatcher=True) == ["mymodel", "--interactive"]
    assert viewercore.get_command_tokens(args, ["other"], {}, fromdispatcher=True) == ["other"]


def test_viewer_finds_each_path_option() -> None:
    """Each option form of the paths of the positional argument gives a series of the viewer, e.g. --specpath."""
    for module, flags in (
        (at.spectra.plotspectra, {"--specpath", "--modelpath"}),
        (at.lightcurve.plotlightcurve, {"--modelpath"}),
    ):
        assert viewercore.get_path_option_flags(viewercore.make_parser(module.addargs)) == flags
    viewertokens = viewercore.parse_viewer_tokens(
        at.spectra.plotspectra.addargs, ["--specpath", str(modelpath), "-t", "300"], set()
    )
    assert viewertokens.paths == [str(modelpath)]


def test_viewer_python_code_gives_the_changed_arguments() -> None:
    """The Python code gives each argument that differs from its default, and a rejected command gives a comment."""
    parser = viewercore.make_parser(at.lightcurve.plotlightcurve.addargs)
    code = viewerwidgets.get_python_code(parser, ["mymodel", "--notitle"], "plotlightcurves", "at.lightcurve.plot")
    assert code.startswith("import artistools as at\n\nat.lightcurve.plot(\n")
    assert "notitle=True" in code
    rejected = viewerwidgets.get_python_code(parser, ["-xmin", "abc"], "plotlightcurves", "at.lightcurve.plot")
    assert rejected == "# plotlightcurves rejects the command"
