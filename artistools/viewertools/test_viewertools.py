import dataclasses as dc
import importlib
import json
import os
import subprocess  # ruff:ignore[suspicious-subprocess-import]
import sys
import typing as t
from pathlib import Path
from unittest import mock

import matplotlib.figure as mplfig
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

import artistools as at
from artistools.viewertools import application as viewerapplication
from artistools.viewertools import core as viewercore
from artistools.viewertools import sections as viewersections
from artistools.viewertools import series as viewerseries

if t.TYPE_CHECKING:
    from collections.abc import Callable

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
    assert viewer.draw() is None
    return viewer


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
def test_viewer_changes_the_residual_baseline(kind: str) -> None:
    """The option table gives the baseline index to the command and keeps the residual panel."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath), str(modelpath), *timetokens, "-residualbaselineseries"])
    assert viewer.values.otheroptions == (("-residualbaselineseries", ()),)
    assert viewer.residualaxis is not None
    assert viewer.change(dc.replace(viewer.values, otheroptions=(("-residualbaselineseries", ("1",)),))) is None
    assert viewer.values.otheroptions == (("-residualbaselineseries", ("1",)),)
    assert viewer.residualaxis is not None
    tokens = viewer.get_plot_tokens()
    assert tokens[tokens.index("-residualbaselineseries") + 1] == "1"
    assert viewer.change(dc.replace(viewer.values, otheroptions=())) is None
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
    assert viewer.values.otheroptions == ((flag, values),)


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
def test_viewer_selects_the_residual_series(kind: str) -> None:
    """Both viewers use the residual indices and the baseline index from the option table."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath)] * 3 + timetokens + ["-residual"])
    assert viewer.residualaxis is not None
    assert len(viewer.residualaxis.lines) == 3
    options = (("-residual", ("0",)), ("-residualbaselineseries", ("2",)))
    assert viewer.change(dc.replace(viewer.values, otheroptions=options)) is None
    assert viewer.residualaxis is not None
    assert len(viewer.residualaxis.lines) == 2
    parser = viewercore.make_parser(
        at.spectra.plotspectra.addargs if kind == "spectra" else at.lightcurve.plotlightcurve.addargs
    )
    args = parser.parse_args(viewer.get_plot_tokens())
    assert args.residuals == [0]
    assert args.residualbaselineseries == 2
    assert viewer.change(dc.replace(viewer.values, otheroptions=(("-residual", ()),))) is None
    assert len(viewer.residualaxis.lines) == 3
    assert viewer.change(dc.replace(viewer.values, otheroptions=())) is None
    assert viewer.residualaxis is None


@pytest.mark.parametrize("kind", ["spectra", "lightcurve"])
@pytest.mark.parametrize("residualtype", ["absolute", "relative", "relativelog"])
def test_viewer_selects_the_residual_type(kind: str, residualtype: str) -> None:
    """Both viewers use the residual type from the option table."""
    timetokens = ["-t", "300"] if kind == "spectra" else []
    viewer = make_viewer(kind, [str(modelpath)] * 2 + timetokens + ["-residual"])
    assert viewer.residualaxis is not None
    assert viewer.residualaxis.get_yscale() == "linear"
    assert viewer.residualaxis.get_ylabel() == "series / baseline"
    initialvalues = np.asarray(viewer.residualaxis.lines[0].get_ydata())
    assert np.isfinite(initialvalues).any()
    assert np.allclose(initialvalues[np.isfinite(initialvalues)], 1.0)
    options = (("-residual", ()), ("-residualtype", (residualtype,)))
    assert viewer.change(dc.replace(viewer.values, otheroptions=options)) is None
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
    action = viewercore.get_actions_by_flag(parser)["-residualtype"]
    assert viewercore.get_option_kind(action) == "choice"
    assert viewercore.get_default_tokens(action) == ("relative",)


@pytest.mark.parametrize(("kind", "timetokens"), [("spectra", ["-t", "300"]), ("lightcurve", ["--plotcmf"])])
def test_viewer_takes_back_a_folder_of_a_list_option(
    kind: str, timetokens: list[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model folder that -label took must be a series of the window, as the command gives the folder back.

    The viewer split the tokens itself, thus the spectra window raised an IndexError and the light curve window kept
    the folder as a second label and read the working folder.
    """
    monkeypatch.chdir(tmp_path)
    viewer = make_viewer(kind, ["-label", "My model", str(modelpath), *timetokens, "--interactive"])
    seriespaths = viewer.values.spectra if kind == "spectra" else viewer.values.lightcurves
    assert seriespaths == (str(modelpath),)
    assert viewer.values.otheroptions == (("-label", ("My model",)),)
    # a list option that a control gives, e.g. -fixedionlist, leaves no row, and the parsed folder is the series
    parsed = viewercore.parse_viewer_tokens(
        at.spectra.plotspectra.addargs,
        ["-fixedionlist", "Fe II", str(modelpath), "-t", "300"],
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
    """A -label or a -title with the name of a folder stays a text, and the path of the folder becomes absolute.

    The legend and the title of a restored window then showed the full path of the folder.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run1").mkdir()
    tokens = ["run1", "-t", "300", "-label", "run1", "default", "-title", "run1", "-ymin=-1", "run1"]
    assert viewerapplication.get_absolute_tokens(tokens) == [
        str(tmp_path / "run1"),
        "-t",
        "300",
        "-label",
        "run1",
        "default",
        "-title",
        "run1",
        "-ymin=-1",
        str(tmp_path / "run1"),
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
    """The dialog of the line properties must refuse a label that the command reads as a flag or as the default.

    The command rejected "-ve" with a message about an ambiguous option, and it gave "default" the automatic label.
    """
    assert viewerseries.get_label_error("My model") is None
    assert viewerseries.get_label_error("") is None
    for label in ("-ve", "--", "default"):
        assert viewerseries.get_label_error(label) is not None


def test_direction_bins_have_the_option_as_tooltip() -> None:
    """Each bin of the list of directions gives its option, as the other controls give the help of their option."""
    assert viewersections.get_bin_tooltip("bin", 3) == "-plotviewingangle 3"
    assert viewersections.get_bin_tooltip("phi", 3) == "-plotviewingangle 3 --average_over_phi_angle"
    assert viewersections.get_bin_tooltip("vpkt", 0) == "-plotvspecpol 0"
    assert "real packets" in viewersections.get_bin_tooltip("vpkt", viewersections.ALL_DIRECTIONS_BIN)


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
