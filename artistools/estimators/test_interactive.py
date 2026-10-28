import argparse
import copy
import dataclasses as dc
import json
import os
import shlex
import subprocess  # ruff:ignore[suspicious-subprocess-import]
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from unittest import mock

import matplotlib.colors as mplcolors
import matplotlib.figure as mplfig
import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg

import artistools as at
from artistools.estimators import interactive
from artistools.estimators import plotestimators
from artistools.estimators import plotitems
from artistools.estimators import subplotcards
from artistools.viewertools import application as viewerapplication
from artistools.viewertools import core as viewercore
from artistools.viewertools import widgets as viewerwidgets
from artistools.viewertools.window import apply_values
from artistools.viewertools.window import draw_values

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def make_headless_viewer(tokens: list[str]) -> interactive.EstimatorViewer:
    """Return a viewer that draws on a canvas with no window, after its first plot."""
    fig = mplfig.Figure()
    FigureCanvasAgg(fig)
    viewer = interactive.EstimatorViewer(tokens, fig)
    assert draw_values(viewer) is None
    return viewer


def get_drawn_lines(fig: mplfig.Figure) -> list[list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]]:
    """Return the x values and the y values of each line of each subplot of a figure."""
    return [
        [
            (np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float))
            for line in axis.get_lines()
        ]
        for axis in fig.axes
    ]


def assert_same_lines(fig: mplfig.Figure, otherfig: mplfig.Figure) -> None:
    """Assert that two figures draw the same lines in the same subplots."""
    lines, otherlines = get_drawn_lines(fig), get_drawn_lines(otherfig)
    assert [len(axislines) for axislines in lines] == [len(axislines) for axislines in otherlines]
    assert any(lines)
    for axislines, otheraxislines in zip(lines, otherlines, strict=True):
        for (xvalues, yvalues), (otherx, othery) in zip(axislines, otheraxislines, strict=True):
            assert np.allclose(xvalues, otherx, rtol=1e-12, atol=0.0, equal_nan=True)
            assert np.allclose(yvalues, othery, rtol=1e-12, atol=0.0, equal_nan=True)


def test_interactive_command_reproduces_plot(
    tmp_path: Path, get_plotestimators_figure: Callable[[Sequence[str], Path], mplfig.Figure]
) -> None:
    """The command that the viewer shows must draw the same data as the viewer.

    The first subplot goes before the folder and each other subplot follows --plot, thus a quoted ion and a directive
    must reach plotestimators unchanged.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "5", "--interactive"])
    subplots = (("Te", "TR", "yscale=linear"), ("populations", "Fe II", "Fe III", "Ni II"), ("nne",))
    newvalues = dc.replace(viewer.select_positions(viewer.values, (10, 12)), subplots=subplots, xbins="8")
    assert apply_values(viewer, newvalues) is None
    command = viewer.get_command()
    assert command.endswith("--timestep 10-12 --xbins 8 --plot populations 'Fe II' 'Fe III' 'Ni II' --plot nne")

    assert_same_lines(viewer.fig, get_plotestimators_figure(shlex.split(command)[2:], tmp_path / "estimators.pdf"))


def test_interactive_swatches_give_the_colours_of_the_lines() -> None:
    """A chip shows the colour of the line of its series, and the popup shows the colour with no style of its own.

    --markers draws the points after the line with a lighter colour. A series with a colour does not move the axes
    cycle, thus TR after the red Te takes the first colour of the cycle.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-timestep", "10", "--markers", "--interactive"])
    assert (
        apply_values(viewer, dc.replace(viewer.values, subplots=(("Te", "TR@linewidth=3"), ("Te", "TR@color=#d55e00"))))
        is None
    )
    # the line of TR comes first, and its points come after it
    trline, trpoints = (
        line
        for line in interactive.get_plot_frames(viewer.fig)[0].get_lines()
        if plotestimators.get_gid_series_name(str(line.get_gid())) == "TR"
    )
    assert viewer.swatches[0, "TR"].colour == mplcolors.to_hex(trline.get_color()) == mplcolors.to_hex("C0")
    assert mplcolors.to_hex(trpoints.get_color()) != mplcolors.to_hex("C0")
    assert viewer.swatches[1, "TR"].colour == "#d55e00"
    assert viewer.swatches[1, "TR"].defaultcolour == mplcolors.to_hex("C0")


def test_move_series_item_keeps_the_type_and_the_directives() -> None:
    """A dragged chip goes in front of the chip at the drop place, and the item keeps its style.

    The chips of a subplot do not show its type and its directives, thus the index of a chip is not the position of its
    item. In the same subplot, the chips on the screen still show the moved chip, and a later index moves to the front.
    """
    columns = ("Te", "TR", "nne", "nnion_Fe_II", "nnion_Fe_III", "nnion_Co_II", "nnelement_Fe", "nnelement_Co")
    subplots = (("Te", "TR@color=C3", "yscale=log"), ("populations", "Fe II", "Fe III", "ymin=1e-5"))
    assert subplotcards.move_series_item(subplots, (0, 1), 0, 0, columns) == (
        ("TR@color=C3", "Te", "yscale=log"),
        subplots[1],
    )
    # the place after the last chip of the same subplot
    assert subplotcards.move_series_item(subplots, (0, 0), 0, 2, columns) == (
        ("TR@color=C3", "Te", "yscale=log"),
        subplots[1],
    )
    assert subplotcards.move_series_item(subplots, (1, 1), 1, 2, columns) == (
        subplots[0],
        ("populations", "Fe III", "Fe II", "ymin=1e-5"),
    )
    # a subplot with no chip takes the item after its type and in front of its directives
    emptied = (("Te",), ("populations", "ymin=1e-5"))
    assert subplotcards.move_series_item((("Te",), ("populations", "Fe II", "ymin=1e-5")), (1, 1), 0, 0, columns) == (
        ("Fe II", "Te"),
        emptied[1],
    )
    assert subplotcards.move_series_item(((("Te", "TR"), ("TR",))), (0, 1), 1, 0, columns) is None


# the code that drives the keys, the mouse, and the line properties popup of an estimator window on the offscreen
# platform of Qt, and prints the results
ESTIMATOR_WINDOW_CHECK_CODE: t.Final = """
import json
import sys
import time

from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtTest
from PySide6 import QtWidgets

QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, sys.argv[2])
from artistools.estimators import interactive
from artistools.viewertools.application import start_application

app = start_application(interactive.APPLICATION_NAME, interactive.get_icon_curve())
windows = []
tokens = [sys.argv[1], "-timestep", "50", "-plot", "Te", "TR@linealpha=0", "nne@linewidth=2,color=red"]
assert interactive.open_window(tokens, windows) is None
window = windows[-1]
window.show()
LEFT = QtCore.Qt.MouseButton.LeftButton
NOMOD = QtCore.Qt.KeyboardModifier.NoModifier


def settle(seconds=0.6):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def get_command():
    return next(
        box.toPlainText()
        for box in window.findChildren(QtWidgets.QPlainTextEdit)
        if box.toPlainText().startswith("artistools ")
    )


def get_chip(name):
    return next(
        chip
        for chip in window.findChildren(QtWidgets.QWidget, "chip")
        if chip.isVisible() and chip.findChild(QtWidgets.QLabel, "chiplabel").text().split()[0] == name
    )


def get_focus_text():
    widget = QtWidgets.QApplication.focusWidget()
    if widget is not None and (label := widget.findChild(QtWidgets.QLabel, "chiplabel")) is not None:
        return label.text().split()[0]
    return widget.objectName() if widget is not None else None


def get_popup():
    return next(
        (widget for widget in QtWidgets.QApplication.topLevelWidgets() if widget.isVisible() and widget is not window),
        None,
    )


def get_action(text):
    return next(action for action in window.findChildren(QtGui.QAction) if action.text() == text)


settle(2.0)
results = {}
# the window keeps the state of a section only after a click on its heading
results["sectionsettings"] = [key for key in QtCore.QSettings().allKeys() if "section" in key]
# Qt shows the pointer of a disabled widget from its parent
grips = [label for label in window.findChildren(QtWidgets.QLabel) if label.property("symbol") == "line.3.horizontal"]
results["gripcursor"] = bool(grips) and all(
    grip.isEnabled() and grip.cursor().shape() == QtCore.Qt.CursorShape.OpenHandCursor for grip in grips
)
# Alt-Down twice moves the same chip to the end
get_chip("Te").setFocus()
settle(0.2)
for _ in range(2):
    QtTest.QTest.keyClick(
        QtWidgets.QApplication.focusWidget(), QtCore.Qt.Key.Key_Down, QtCore.Qt.KeyboardModifier.AltModifier
    )
    settle()
results["chipfocus"] = get_focus_text()
results["chiporder"] = get_command()

# a press on the remove button of a chip, a move off the button, and a release open no popup
remove = next(
    button for button in get_chip("Te").findChildren(QtWidgets.QToolButton) if button.objectName() != "swatch"
)
centre = remove.rect().center()
QtTest.QTest.mousePress(remove, LEFT, NOMOD, centre)
offbutton = QtCore.QPoint(remove.width() + 3, centre.y())
QtTest.QTest.mouseMove(remove, offbutton)
QtTest.QTest.mouseRelease(remove, LEFT, NOMOD, offbutton)
settle()
results["cancelledclickpopup"] = get_popup() is not None
if (popup := get_popup()) is not None:
    popup.close()
    settle()

# the popup shows an opacity of 0, and Reset then Return removes it
get_chip("TR").findChild(QtWidgets.QToolButton, "swatch").click()
settle()
popup = get_popup()
alphabox = popup.findChildren(QtWidgets.QDoubleSpinBox)[-1]
results["alphatext"] = alphabox.text()
next(button for button in popup.findChildren(QtWidgets.QPushButton) if button.text() == "Reset").click()
settle()
QtTest.QTest.keyClick(popup, QtCore.Qt.Key.Key_Return)
settle()
results["resetcommand"] = get_command()

# Return in the empty colour field closes the popup
get_chip("Te").findChild(QtWidgets.QToolButton, "swatch").click()
settle()
popup = get_popup()
colouredit = next(
    edit for edit in popup.findChildren(QtWidgets.QLineEdit) if edit.placeholderText().startswith("other")
)
colouredit.setFocus()
QtTest.QTest.keyClick(colouredit, QtCore.Qt.Key.Key_Return)
settle()
results["emptycolourclosed"] = get_popup() is None

# Escape gives the item its text from before the popup
get_chip("nne").findChild(QtWidgets.QToolButton, "swatch").click()
settle()
popup = get_popup()
next(
    button for button in popup.findChildren(QtWidgets.QToolButton) if button.accessibleName().startswith("Colour ")
).click()
settle(1.0)
results["previewcommand"] = get_command()
QtTest.QTest.keyClick(popup, QtCore.Qt.Key.Key_Escape)
settle()
results["escapecommand"] = get_command()

# a bad dash pattern keeps the preview of the width
get_chip("nne").findChild(QtWidgets.QToolButton, "swatch").click()
settle()
popup = get_popup()
next(edit for edit in popup.findChildren(QtWidgets.QLineEdit) if edit.placeholderText().startswith("none")).setText(
    "5,"
)
popup.findChildren(QtWidgets.QDoubleSpinBox)[0].setValue(4.0)
settle(1.0)
results["badfieldpreview"] = get_command()
QtTest.QTest.keyClick(popup, QtCore.Qt.Key.Key_Escape)
settle()

# Add Series on a collapsed card gives the keyboard to its field
window.activateWindow()
settle(0.3)
next(
    button
    for button in window.findChildren(QtWidgets.QToolButton)
    if button.isVisible() and button.accessibleName() == "Collapse"
).click()
settle()
next(box for box in window.findChildren(QtWidgets.QComboBox, "type") if box.isVisible()).setFocus()
get_action("Add Series").trigger()
settle()
results["addseriesfocus"] = get_focus_text()

# Add Subplot adds a subplot below the card with the focus, and the y axis button takes the keyboard
get_action("Add Subplot").trigger()
settle()
results["addsubplotcommand"] = get_command()
results["addsubplotfocus"] = get_focus_text()
results["yaxisfocus"] = all(
    button.focusPolicy() == QtCore.Qt.FocusPolicy.StrongFocus
    for button in window.findChildren(QtWidgets.QToolButton, "yaxis")
)
print(json.dumps(results))
window.close()
settle(0.3)
"""


def test_estimator_window_keeps_the_keyboard_and_the_popup_values(tmp_path: Path) -> None:
    """The keys, a cancelled click, and the line properties popup give the series and the focus that the user expects.

    The window had these faults:

    - after Alt-Down, the first chip took the focus, thus the next key moved a different series;
    - a press on the remove button of a chip and a release outside it opened the popup;
    - the popup showed an opacity of 0 as "Default", and Reset kept it;
    - Return in the empty colour field kept the popup open with an error;
    - Escape wrote the style keys of the item in a different order;
    - a bad dash pattern stopped the preview of each other field;
    - the window wrote the state of each section when it opened, thus a new default of closed did not apply;
    - a grip showed the arrow pointer, because it was a disabled button;
    - Add Series on a collapsed card gave the focus back to the header;
    - no menu item added a subplot, and Tab on macOS reaches no button that adds one.
    """
    pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
    environment = os.environ | {"QT_QPA_PLATFORM": "offscreen", viewercore.MACOS_BUNDLE_VARIABLE: "1"}
    result = subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", ESTIMATOR_WINDOW_CHECK_CODE, str(modelpath), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    # the window prints its command when it closes, thus the results are on the line of the JSON object
    results = json.loads(next(line for line in result.stdout.splitlines() if line.startswith('{"sectionsettings"')))
    assert results["chipfocus"] == "Te"
    assert " TR@linealpha=0 nne@linewidth=2,color=red Te " in results["chiporder"]
    assert not results["cancelledclickpopup"]
    assert results["alphatext"] != "Default"
    assert "linealpha" not in results["resetcommand"]
    assert results["emptycolourclosed"]
    assert "nne@color=" in results["previewcommand"]
    assert " nne@linewidth=2,color=red " in results["escapecommand"]
    assert " nne@color=red,linewidth=4 " in results["badfieldpreview"]
    assert results["sectionsettings"] == []
    assert results["gripcursor"]
    assert results["addseriesfocus"] == "add"
    assert results["addsubplotcommand"].count("--plot") == 1
    assert results["addsubplotfocus"] == "add"
    assert results["yaxisfocus"]


def test_colour_note_follows_the_colours_of_the_plot() -> None:
    """The note of the element colours shows only for a subplot whose plot gives its ions the element colours.

    The note counted the names of the subplot, but the plot leaves out an ion that the model does not hold. Thus the
    note showed for Fe II, Fe III, and Zn II, and the plot gave each ion its own colour.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-timestep", "10", "--colorbyion", "--interactive"])
    subplots = (("populations", "Fe II", "Fe III", "Zn II"), ("populations", "Fe II", "Co II"))
    assert apply_values(viewer, dc.replace(viewer.values, subplots=subplots)) is None
    assert viewer.elementcolourrows == frozenset({1})


def test_style_change_keeps_the_warning_of_the_data() -> None:
    """A plot that draws the data of the last plot shows the warning of that data again.

    The data cache read no data, thus the plot printed no warning, and the status bar removed a warning that applied.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-timestep", "10", "--interactive"])
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(("populations", "Fe II", "Fe VIII"),))) is None
    warning = viewer.warning
    assert "compositiondata.txt" in warning
    assert (
        apply_values(viewer, dc.replace(viewer.values, subplots=(("populations", "Fe II@color=C3", "Fe VIII"),)))
        is None
    )
    assert viewer.warning == warning


def test_style_colour_keeps_the_colours_of_the_other_series() -> None:
    """A colour of a style changes only its own series, and each other series keeps its colour of the axes cycle.

    A series with a style colour took no place in the cycle, thus each later series took the colour before its own.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-timestep", "10", "--interactive"])

    def get_line_colours(subplot: tuple[str, ...]) -> dict[str, str]:
        assert apply_values(viewer, dc.replace(viewer.values, subplots=(subplot,))) is None
        lines = interactive.get_plot_frames(viewer.fig)[0].get_lines()
        return {
            plotestimators.get_gid_series_name(str(line.get_gid())): mplcolors.to_hex(line.get_color())
            for line in lines
        }

    plain = get_line_colours(("TR", "TJ", "W"))
    styled = get_line_colours(("TR@color=black", "TJ", "W"))
    assert styled == plain | {"TR": "#000000"}
    assert viewer.swatches[0, "TJ"].colour == plain["TJ"]
    assert viewer.swatches[0, "TJ"].defaultcolour == plain["TJ"]


def test_swatches_show_the_drawn_colours() -> None:
    """The swatch of a chip shows the opacity of its colour and the colour of Dark Mode, as the plot draws the line."""
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-timestep", "10", "--interactive"])
    subplot = ("TR@color=#ff000080", "TJ@color=none", "W@color=#000000")
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(subplot,))) is None
    assert np.isclose(viewer.swatches[0, "TR"].alpha, 128 / 255)
    assert viewer.swatches[0, "TR"].colour == "#ff0000"
    assert viewer.swatches[0, "TJ"].alpha == 0
    viewer.darkcolours = at.plottools.DARKMODE_COLOURS
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(subplot[::-1],))) is None
    assert viewer.swatches[0, "W"].colour == "#ffffff"


def test_host_receives_the_plot_items_without_styles(tmp_path: Path) -> None:
    """The client draws the series styles and selects the cells, thus the host receives the names and the cells.

    A host that runs an older release read a styled item as an unknown directive, e.g. "'Te@color' is not a plot
    directive". That release also selected no cells, thus -readonlymgi plotted no rows. The data cache also selects
    the cells only when it reads the data.
    """
    plotlists: list[t.Any] = []
    selectedcells: list[t.Any] = []
    getfiguresdata = plotestimators.get_figures_data

    def record_plotlist(modelpath: Path, args: argparse.Namespace, *otherargs: t.Any) -> "plotestimators.FiguresData":
        plotlists.append(copy.deepcopy(args.plotlist))
        selectedcells.append(args.modelgridindex)
        return getfiguresdata(modelpath, args, *otherargs)

    with mock.patch.object(plotestimators, "get_figures_data", side_effect=record_plotlist):
        at.estimators.plot(
            argsraw=[],
            modelpath=modelpath_classic_3d,
            plotlist=[["Te@color=C3,label=T e", "TR"]],
            timestep=10,
            readonlymgi="alongaxis",
            axis="+x",
            outputfile=tmp_path / "styled.pdf",
        )
    assert plotlists == [[["Te", "TR"]]]
    assert len(selectedcells) == 1
    assert selectedcells[0]
    assert (tmp_path / "styled.pdf").is_file()

    viewer = make_headless_viewer([
        str(modelpath_classic_3d),
        "-timestep",
        "10",
        "-readonlymgi",
        "alongaxis",
        "-axis",
        "+x",
        "--interactive",
    ])
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(("Te",),))) is None
    with mock.patch.object(plotestimators, "get_cells_along_axis", wraps=plotestimators.get_cells_along_axis) as cells:
        assert apply_values(viewer, dc.replace(viewer.values, subplots=(("Te@color=C3",),))) is None
        assert cells.call_count == 0


def test_interactive_style_change_reuses_the_data() -> None:
    """A change of a series style or of -linewidthscale draws the data of the last plot again, and reads no data.

    A change of the time reads the data again. A click on a legend entry finds its series, also after a new style.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-timestep", "10", "--interactive"])
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(("populations", "Fe II", "Fe III"),))) is None
    styled = dc.replace(viewer.values, subplots=(("populations", "Fe II", "Fe III@color=#d55e00,label=Iron 2+"),))
    with mock.patch.object(plotestimators, "get_figures_data", wraps=plotestimators.get_figures_data) as getdata:
        assert apply_values(viewer, styled) is None
        widevalues = interactive.replace_option_rows(viewer, viewer.values, (("-linewidthscale", ("2",)),))
        assert apply_values(viewer, widevalues) is None
        assert getdata.call_count == 0
        assert apply_values(viewer, viewer.select_positions(viewer.values, (11, 11))) is None
        assert getdata.call_count == 1

    assert viewer.swatches[0, "Fe III"].colour == "#d55e00"
    axis = interactive.get_plot_frames(viewer.fig)[0]
    viewer.fig.canvas.draw()
    legend = axis.get_legend()
    assert legend is not None
    for text in legend.get_texts():
        centre = text.get_window_extent().get_points().mean(axis=0)
        event = MouseEvent("button_press_event", viewer.fig.canvas, *centre)
        expected = "Fe III" if text.get_text() == "Iron 2+" else "Fe II"
        assert interactive.get_series_at(axis, event, legendonly=True) == expected


def test_interactive_python_code_reproduces_plot(tmp_path: Path) -> None:
    """The Python code that the viewer shows must draw the same data as the viewer.

    The code groups the names of a type of series, also for a list of ions with no type. A directive becomes a pair with
    an underscore, and the value of ymin= becomes a number.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "5", "--interactive"])
    subplots = (("Te", "TR", "yscale=linear", "ymin=1000"), ("Fe II", "Fe III"), ("populations", "Ni II"))
    newvalues = dc.replace(viewer.select_positions(viewer.values, (10, 12)), subplots=subplots, xbins="8")
    assert apply_values(viewer, newvalues) is None
    code = interactive.get_plot_python_code(viewer)
    assert '["Te", "TR", ["_yscale", "linear"], ["_ymin", 1000]],' in code
    assert '[["populations", ["Fe II", "Fe III"]]],' in code
    assert 'timestep="10-12",' in code
    get_arguments = partial(interactive.get_python_arguments, viewer.parser, viewer.modeldata.estimatorcolumns)
    onetimestepcode = viewerwidgets.get_python_code(
        viewer.parser, ["Te", "-timestep", "11"], "plotestimators", "at.estimators.plotestimators.main", get_arguments
    )
    assert "timestep=11," in onetimestepcode

    savedfigures: list[mplfig.Figure] = []

    def keep_figure(fig: mplfig.Figure, *_: t.Any, **__: t.Any) -> None:
        savedfigures.append(fig)

    outputfile = tmp_path / "estimators.pdf"
    with mock.patch.object(plotestimators, "save_figure", side_effect=keep_figure):
        exec(code.replace("main(\n", f"main(\n    outputfile={str(outputfile)!r},\n", 1), {})  # ruff:ignore[exec-builtin]
    [savedfigure] = savedfigures
    assert_same_lines(viewer.fig, savedfigure)


def test_interactive_subplot_with_no_series_stays_as_an_empty_frame(
    tmp_path: Path, get_plotestimators_figure: Callable[[Sequence[str], Path], mplfig.Figure]
) -> None:
    """A subplot whose last series the user removed stays, and its frame is empty until the user adds a series.

    The viewer deleted such a subplot, thus the user lost its place and its type. A first subplot with no item follows
    --plot, and a series type with no names gives only its directives to the command.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "5", "--interactive"])
    subplots = ((), ("TR", "yscale=linear"), ("populations", "yscale=log"))
    assert apply_values(viewer, dc.replace(viewer.values, subplots=subplots)) is None
    assert viewer.values.subplots == subplots
    assert len(interactive.get_plot_frames(viewer.fig)) == 3
    tokens = viewer.get_plot_tokens()
    assert tokens[-6:] == ["--plot", "--plot", "TR", "yscale=linear", "--plot", "yscale=log"]

    assert_same_lines(viewer.fig, get_plotestimators_figure(tokens, tmp_path / "estimators.pdf"))


def test_interactive_session_keeps_an_empty_last_subplot() -> None:
    """The command of the next start keeps a last subplot with no item, also when the window read the working folder.

    The session put the working folder after the last -plot. plotestimators then took the folder out of that -plot and
    removed the -plot, which was empty, thus the next start lost the subplot.
    """
    viewer = make_headless_viewer([str(modelpath_classic_3d), "-t", "5", "--interactive"])
    subplots = (("TR",), ())
    assert apply_values(viewer, dc.replace(viewer.values, subplots=subplots)) is None
    tokens = viewer.get_plot_tokens(modeltoken=str(modelpath_classic_3d.absolute()))
    assert make_headless_viewer([*tokens, "--interactive"]).values.subplots == subplots


def test_interactive_command_gives_the_default_subplots(
    tmp_path: Path, get_plotestimators_figure: Callable[[Sequence[str], Path], mplfig.Figure]
) -> None:
    """The command gives each subplot also for the default subplots, and it draws the figure of no --plot.

    The command left out the default subplots, thus it depended on the default of the version of plotestimators. The
    viewer writes a default item such as ["_ymin", 1e-16] as the directive "ymin=1e-16".
    """
    viewer = make_headless_viewer([str(modelpath), "--timestep", "50", "--interactive"])
    assert viewer.values.subplots == viewer.modeldata.defaultsubplots
    assert "averageexcitation" in viewer.modeldata.defaultsubplots[-1]
    # the window hides the output of the command, thus the viewer keeps the reason for each left out subplot
    assert viewer.modeldata.skippeddefaults == (
        "averageionisation Sr: the estimators have no Sr",
        "populations Sr I Sr II Sr III Sr IV: the estimators have no Sr",
    )
    first, *others = viewer.modeldata.defaultsubplots
    assert viewer.get_plot_tokens()[: len(first)] == list(first)
    assert viewer.get_plot_tokens().count("--plot") == len(others)

    assert_same_lines(
        viewer.fig, get_plotestimators_figure([str(modelpath), "--timestep", "50"], tmp_path / "estimators.pdf")
    )


def test_interactive_default_x_follows_a_slice_line_of_the_table() -> None:
    """A -slice line in the option table changes the default x, and a command with no -x takes that default.

    The viewer kept the old default as a fixed -x, thus the command plotted the line against the radial velocity.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-timestep", "5", "--interactive"])
    sliceline = interactive.replace_option_rows(viewer, viewer.values, (("-slice", ("z=0,y=0",)),))
    assert "-x" not in viewer.get_plot_tokens(sliceline)
    assert apply_values(viewer, sliceline) is None
    assert "-x" not in viewer.get_plot_tokens(interactive.replace_option_rows(viewer, viewer.values, ()))

    # an x that the user chose stays
    chosen = viewer.set_xvariable(viewer.values, "vel_r_mid")
    assert interactive.replace_option_rows(viewer, chosen, (("-slice", ("z=0,y=0",)),)).x == "vel_r_mid"


def test_interactive_snapshot_reads_all_the_cells() -> None:
    """A change from a plot against time to a snapshot removes -cell.

    The window hides the control of the cells for a snapshot, thus a -cell of the plot against time stayed hidden.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "--interactive"])
    evolution = dc.replace(viewer.values, x="time", cells=str(viewer.modeldata.cells[5]))
    assert interactive.is_evolution(evolution)
    viewer.values = evolution
    snapshot = viewer.set_xvariable(evolution, "velocity")
    assert not snapshot.cells
    assert "-cell" not in viewer.get_plot_tokens(snapshot)


def test_interactive_snapshot_and_time_evolution() -> None:
    """A plot against time reads all the timesteps and names no time, and a snapshot reads one timestep.

    The Play button of a plot against time moves through the cells, and it stops after the last cell.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "--timestep", "20-22", "--interactive"])
    evolution = viewer.set_xvariable(viewer.values, "time")
    assert (evolution.first, evolution.last) == (
        viewer.modeldata.validtimesteps[0],
        viewer.modeldata.validtimesteps[-1],
    )
    assert apply_values(viewer, evolution) is None
    assert viewer.get_plot_tokens() == ["Te", str(modelpath_classic_3d)]

    viewer.values = dc.replace(viewer.values, cells=str(viewer.modeldata.cells[-2]))
    lastcell = viewer.step_cell(1)
    assert lastcell is not None
    assert apply_values(viewer, lastcell) is None
    assert viewer.get_plot_tokens() == ["Te", str(modelpath_classic_3d), "--cell", str(viewer.modeldata.cells[-1])]
    assert viewer.step_cell(1) is None

    # a part of the time range of the model needs -x time, because plotestimators plots a snapshot for a given time
    shortrange = viewer.select_positions(viewer.values, (5, 8))
    assert viewer.get_plot_tokens(shortrange)[2:6] == ["--timestep", "5-8", "-x", "time"]

    snapshot = viewer.set_xvariable(viewer.values, "velocity")
    # the middle of all the timesteps
    assert (
        snapshot.first
        == snapshot.last
        == viewer.modeldata.validtimesteps[(len(viewer.modeldata.validtimesteps) - 1) // 2]
    )


def test_interactive_command_of_a_dispatcher_call() -> None:
    """A call of the dispatcher from Python code gives its own words to the viewer, and not the words of sys.argv."""
    from artistools.__main__ import main as dispatcher_main

    with (
        mock.patch.object(viewerapplication, "run_viewer_application") as mockrunviewer,
        mock.patch.object(sys, "argv", ["myscript.py", "--flag", "x"]),
    ):
        dispatcher_main(argsraw=["plotestimators", "Te", str(modelpath), "--interactive"])
    mockrunviewer.assert_called_once()
    assert mockrunviewer.call_args.args[2:] == (
        interactive.open_window,
        ["Te", str(modelpath), "--interactive"],
        interactive.DOCUMENT_TYPES,
    )


def test_interactive_classicartis_row_reads_the_other_format() -> None:
    """A --classicartis row in the table must read the classic format, as the command does, and not the old query.

    The viewer read the model one time with the first --classicartis value. The window then drew the modern estimators
    for a command that plotestimators rejects.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    command = viewer.get_command()
    message = apply_values(viewer, dc.replace(viewer.values, otheroptions=(("--classicartis", ()),)))
    assert message is not None
    assert "modern ARTIS run" in message
    assert viewer.get_command() == command


def test_interactive_unexpected_error_keeps_the_old_plot() -> None:
    """An error that is not a user error, e.g. of polars, gives a message, and the old plot draws again.

    Such an error left draw and change, thus the figure stayed empty and the viewer kept the failed values.
    """
    viewer = make_headless_viewer(["rho", str(modelpath), "-ts", "5", "--interactive"])
    command = viewer.get_command()
    # a mock gives the unexpected error, thus the test does not depend on a fault of the code
    with mock.patch.object(
        plotestimators, "get_figures_data", side_effect=pl.exceptions.ColumnNotFoundError("mocked fault")
    ):
        message = apply_values(viewer, dc.replace(viewer.values, xbins="5"))
    assert message is not None
    assert message.startswith("ColumnNotFoundError")
    assert viewer.get_command() == command
    assert viewer.fig.axes[0].get_lines()


def test_interactive_codecomparison_model(toy_codecomparison_model: str) -> None:
    """The viewer opens a codecomparison model, which has no run folders, as the command does."""
    viewer = make_headless_viewer(["Te", toy_codecomparison_model, "-timestep", "0", "--interactive"])
    assert viewer.modeldata.validtimesteps == [0]
    assert viewer.fig.axes[0].get_lines()


def test_interactive_time_field_keeps_the_range() -> None:
    """A Return in the time field with no edit keeps the time range.

    The field showed the mean of the start and the end of the range, and the viewer read it as the middle of a
    timestep. A range of an even count then moved one timestep later.
    """
    viewer = make_headless_viewer(["Te", str(modelpath), "-timestep", "40", "--interactive"])
    nvalid = len(viewer.modeldata.validtimesteps)
    for count in range(1, 5):
        for firstpos in range(nvalid - count + 1):
            viewer.values = viewer.select_positions(viewer.values, (firstpos, firstpos + count - 1))
            text = interactive.get_time_text(viewer.tmids, viewer.values)
            assert viewer.select_centre(float(text)) == viewer.values, (count, firstpos, text)


def test_interactive_readout_names_each_series() -> None:
    """The readout gives the value of each series, also the one line with no label of a subplot of one variable.

    The readout skipped each line with a label that starts with "_", which is the only line of such a subplot. A
    colour image gives an array of one value under the pointer, and NumPy 2.5 does not convert it with float().
    """
    viewer = make_headless_viewer(["Te", "TR", str(modelpath_classic_3d), "-t", "5", "-plot", "nne", "--interactive"])
    firstaxis, nneaxis = viewer.fig.axes
    xmid = float(np.mean(firstaxis.get_xlim()))
    # the parts of a readout are the x value and then "value label" for each series
    firstparts = interactive.get_readout(firstaxis, xmid).split("   ")
    assert len(firstparts) == 3
    assert any(part.endswith(" T_e") for part in firstparts[1:])
    assert len(interactive.get_readout(nneaxis, xmid).split("   ")) == 2

    assert interactive.get_image_value(np.ma.masked_array([5.0])) == pytest.approx(5.0, rel=1e-12, abs=0.0)
    assert interactive.get_image_value(np.ma.masked_array([5.0], mask=[True])) is None


def test_interactive_converts_stale_batches_in_a_child_process() -> None:
    """The viewer process reads the caches of a model with no stale batch. A child process converts a stale batch."""
    at.estimators.scan_estimators(modelpath).head(1).collect()
    with mock.patch.object(interactive, "call_in_child_process") as mockchild:
        assert interactive.get_batch_caches(modelpath)
    mockchild.assert_not_called()

    states = interactive.get_estimator_batch_states(modelpath, None, None)
    stalestates = [states[0]._replace(rebuild=True), *states[1:]]
    with (
        mock.patch.object(interactive, "get_estimator_batch_states", return_value=stalestates),
        mock.patch.object(interactive, "call_in_child_process", return_value=[]) as mockchild,
    ):
        assert interactive.get_batch_caches(modelpath) == []
    mockchild.assert_called_once()


def test_interactive_shows_the_bins_that_the_plot_chose() -> None:
    """A snapshot of a 3D model has many cells at one velocity, thus the plot takes automatic bins and --colorbyion.

    -xbins N also draws a shaded area for each bin, and the area has no dash to separate the ions of an element. It
    turned --colorbyion on only for automatic bins. -xbins 0 draws the points alone, and it keeps the colours.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    assert viewer.plotxbins is not None
    assert viewer.plotxbins > 3
    assert viewer.plotcolorbyion
    assert not viewer.values.colorbyion
    assert apply_values(viewer, dc.replace(viewer.values, xbins="8")) is None
    assert viewer.plotxbins == 8
    assert viewer.plotcolorbyion
    assert apply_values(viewer, dc.replace(viewer.values, xbins="0")) is None
    assert not viewer.plotcolorbyion
    assert interactive.get_xunit_text(1.0, "velocity") == " [km/s]"
    assert interactive.get_xunit_text(1.0, "Te") == " [K]"
    assert not interactive.get_xunit_text(1.0, "timestep")
    assert interactive.get_xunit_text(1e5 / 299792.458, "beta") == " [km/s]"


def test_interactive_keeps_the_last_warning_of_a_plot() -> None:
    """A limit outside the data gives a warning, which the status bar of the window shows after the plot."""
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    assert not viewer.warning
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(("Te", "ymin=1e12"),))) is None
    assert "requested minimum" in viewer.warning
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(("Te",),))) is None
    assert not viewer.warning


def test_interactive_directives_of_a_subplot() -> None:
    """A directive replaces its old value, None removes it, and an ion or a variable is not a directive."""
    assert plotitems.get_item_directive("ymin=1e-16") == "ymin"
    assert plotitems.get_item_directive("_yscale=log") == "yscale"
    assert plotitems.get_item_directive("Fe II") is None
    assert plotitems.get_item_directive("Te") is None
    # plotestimators reads a directive in any case
    assert plotitems.get_item_directive("YMIN=100") == "ymin"
    assert plotitems.replace_directives(("TR", "YMIN=100"), {"ymin": None}) == ("TR",)
    subplot = ("rho", "yscale=log", "ymin=1e-16")
    assert plotitems.replace_directives(subplot, {"ymin": "2", "ymax": "3"}) == (
        "rho",
        "yscale=log",
        "ymin=2",
        "ymax=3",
    )
    assert plotitems.replace_directives(subplot, {"ymin": None, "ymax": None}) == ("rho", "yscale=log")
    assert plotitems.replace_directives(subplot, {"yscale": "linear"}) == ("rho", "ymin=1e-16", "yscale=linear")
    assert interactive.get_short_limits(12345.678, 23456.7) == ("12300", "23500")
    assert interactive.get_short_limits(1.23456e-5, 2.3456e-5) == ("1.23e-05", "2.35e-05")


def test_interactive_menu_plots_a_cell_against_time_and_a_snapshot_at_a_time() -> None:
    """The menu of a snapshot names the cell at the pointer, and the menu of a plot against time names the time."""
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    assert viewer.values.x == "velocity"
    assert interactive.get_snapshot_values(viewer, 1.0) is None
    cell = interactive.get_nearest_cell(viewer, 12000.0)
    assert cell is not None
    assert np.isclose(viewer.modeldata.cellvelocities[cell] / 1e5, 12000.0, rtol=0.2)
    evolution = interactive.get_evolution_values(viewer, str(cell))
    assert interactive.is_evolution(evolution)
    assert evolution.cells == str(cell)
    assert (evolution.first, evolution.last) == (
        viewer.modeldata.validtimesteps[0],
        viewer.modeldata.validtimesteps[-1],
    )
    assert apply_values(viewer, evolution) is None
    assert interactive.get_nearest_cell(viewer, 12000.0) is None
    snapshot = interactive.get_snapshot_values(viewer, viewer.tmids[7])
    assert snapshot is not None
    assert not interactive.is_evolution(snapshot)
    assert (snapshot.first, snapshot.last) == (7, 7)
    assert not snapshot.cells


def reread_model_data(viewer: interactive.EstimatorViewer) -> interactive.ModelData:
    """Read the model of a viewer again, as Reload Data does in the worker thread."""
    from artistools.viewertools.window import clear_output_caches_of_model

    clear_output_caches_of_model(viewer.modelpath, estimatorsonly=True)
    return interactive.read_model_data(viewer.modelpath, viewer.userargs, len(viewer.tmids))


def test_interactive_reload_keeps_the_time_range_inside_the_valid_timesteps() -> None:
    """A reload reads the valid timesteps again.

    A range of all the timesteps grows when ARTIS writes more timesteps. A different range stays inside the valid
    timesteps.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    validtimesteps = viewer.modeldata.validtimesteps
    assert len(validtimesteps) > 8
    lastpos = len(validtimesteps) - 1
    assert apply_values(viewer, viewer.select_positions(viewer.values, (lastpos, lastpos))) is None
    with mock.patch.object(interactive, "get_estimator_timesteps", return_value=validtimesteps[:-5]):
        interactive.reload_model_data(viewer, reread_model_data(viewer))
    assert viewer.modeldata.validtimesteps == validtimesteps[:-5]
    assert (viewer.values.first, viewer.values.last) == (validtimesteps[-6], validtimesteps[-6])
    assert draw_values(viewer) is None

    assert apply_values(viewer, interactive.get_evolution_values(viewer, str(viewer.modeldata.cells[0]))) is None
    assert (viewer.values.first, viewer.values.last) == (validtimesteps[0], validtimesteps[-6])
    with mock.patch.object(interactive, "get_estimator_timesteps", return_value=validtimesteps):
        interactive.reload_model_data(viewer, reread_model_data(viewer))
    assert (viewer.values.first, viewer.values.last) == (validtimesteps[0], validtimesteps[-1])
    assert draw_values(viewer) is None


def test_interactive_empty_selection_with_bins_gives_the_message() -> None:
    """An empty cell with -xbins gave a TypeError, because the bins read the minimum of no rows.

    The message gives the size of a long selection of cells and not each cell.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    emptycell = next(cell for cell in range(1000) if cell not in viewer.modeldata.cells)
    for xbins in ("8", "-1"):
        message = apply_values(viewer, dc.replace(viewer.values, cells=str(emptycell), xbins=xbins))
        assert message is not None
        assert message.startswith("The estimators hold no row"), message
        assert f"the cells {emptycell}" in message

    manycells = ",".join(str(cell) for cell in range(1000) if cell not in viewer.modeldata.cells)
    message = apply_values(viewer, dc.replace(viewer.values, cells=manycells, xmin="1e9"))
    assert message is not None
    assert message.startswith("The estimators hold no row"), message
    assert len(message) < 200


def test_interactive_cells_apply_only_where_cell_selects_the_cells() -> None:
    """-slice, -dimensionreduce 2, and -readonlymgi select the cells, thus the window offers no -cell with them.

    With -readonlymgi, plotestimators replaced -cell with the cells along the axis. The menu then plotted other cells,
    and Page Up, Page Down, and Play changed -cell with no effect on the plot.
    """
    viewer = make_headless_viewer([
        "Te",
        str(modelpath_classic_3d),
        "-t",
        "5",
        "-dimensionreduce",
        "2",
        "--interactive",
    ])
    assert viewer.isimage
    assert not interactive.cells_apply(viewer.values)
    for mode in ("alongaxis", "cone", "plane", "line"):
        viewer.values = interactive.set_geometry_mode(viewer, viewer.values, mode)
        assert not interactive.cells_apply(viewer.values), mode
        assert viewer.step_cell(1) is None, mode
    viewer.values = interactive.set_geometry_mode(viewer, viewer.values, "all")
    assert interactive.cells_apply(viewer.values)
    assert viewer.step_cell(1) is not None


def test_interactive_subplot_types_and_suggestions() -> None:
    """The type of a subplot sets its choices and its suggestions, and each result draws with plotestimators."""
    viewer = make_headless_viewer([str(modelpath), "-timestep", "50", "--interactive"])
    columns = viewer.modeldata.estimatorcolumns
    types = subplotcards.get_subplot_types(columns)
    assert types[:2] == [subplotcards.VARIABLES_TYPE, "populations"]
    assert "gamma_NT" in types
    # the families of the populations and of the initial abundances are no type of their own
    assert not {"nnion", "nnelement", "init_X"} & set(types)

    assert plotitems.get_subplot_seriestype(("Fe II", "Fe III"), columns) == "populations"
    assert plotitems.get_subplot_seriestype(("Te", "yscale=log"), columns) is None
    assert subplotcards.get_series_suggestions(("Te",), columns) == ["TJ", "TR"]
    # the other ions of the element come first, and the total of the element after its ions
    assert subplotcards.get_series_suggestions(("populations", "Fe II", "Fe III"), columns)[:2] == ["Fe I", "Fe IV"]
    # a control of the card sets each directive, thus only the names show as chips
    assert subplotcards.get_chip_items(("populations", "Fe II", "yscale=log", "ymin=1"), columns) == [(1, "Fe II")]

    newsubplots = subplotcards.get_new_subplot_suggestions(
        viewer.values.subplots, viewer.modeldata.defaultsubplots, columns
    )
    assert ("populations", "Fe I", "Fe II") in newsubplots
    # the + button of a card needs no text: it adds the first suggestion, or else a common variable
    assert (
        subplotcards.get_new_subplot(viewer.values.subplots, viewer.modeldata.defaultsubplots, columns)
        == newsubplots[0]
    )
    allsuggested = [*viewer.values.subplots, *newsubplots]
    remaining = subplotcards.get_new_subplot_suggestions(allsuggested, viewer.modeldata.defaultsubplots, columns)
    while remaining:
        allsuggested += remaining
        remaining = subplotcards.get_new_subplot_suggestions(allsuggested, viewer.modeldata.defaultsubplots, columns)
    assert subplotcards.get_new_subplot(allsuggested, viewer.modeldata.defaultsubplots, columns) == ("Te",)
    aliasviewer = make_headless_viewer(["n_e", str(modelpath), "-timestep", "50", "--interactive"])
    assert aliasviewer.values.subplots == (("nne",),)
    # a colour image gives the card the controls of a colour scale, thus the card of an image is a new card
    assert subplotcards.get_card_key(0, (("Te",),), columns, isimage=True) != subplotcards.get_card_key(
        0, (("Te",),), columns
    )
    # a directive changes only a control of the card, thus the card stays with the keyboard focus in it
    assert subplotcards.get_card_key(0, (("Te", "ymin=1000"),), columns) == subplotcards.get_card_key(
        0, (("Te",),), columns
    )
    # a chip removes the item at its position, thus a name at a new position needs a new card
    assert subplotcards.get_card_key(0, (("Te", "ymin=1", "TR"),), columns) != subplotcards.get_card_key(
        0, (("Te", "TR", "ymin=1"),), columns
    )

    populations = ("populations", "Fe II", "Fe III", "ionpoptype=elpop", "yscale=log", "ymin=1e-5")
    # only a plot of the populations takes ionpoptype=, and the names that apply to the new type stay
    assert subplotcards.change_subplot_type(populations, "gamma_NT", columns) == (
        "gamma_NT",
        "Fe II",
        "Fe III",
        "yscale=log",
    )
    assert subplotcards.change_subplot_type(populations, subplotcards.VARIABLES_TYPE, columns) == ("Te", "yscale=log")
    # the type selector removed ymin= and ymax= when the user selected the type that the subplot already had
    assert subplotcards.change_subplot_type(populations, "populations", columns) == populations
    # a subplot with no series left stays, and the command gives it with no item, which draws an empty frame
    assert subplotcards.remove_subplot_item((("Te",), ("gamma_NT", "Fe II")), 1, 1) == (("Te",), ("gamma_NT",))
    assert subplotcards.remove_subplot_item((("Te", "TR"),), 0, 0) == (("TR",),)
    assert subplotcards.get_command_items(("gamma_NT", "yscale=log"), columns) == ("yscale=log",)

    for subplot in (
        subplotcards.change_subplot_type(populations, "gamma_NT", columns),
        subplotcards.change_subplot_type(("Te",), "averageionisation", columns),
        *newsubplots,
    ):
        assert apply_values(viewer, dc.replace(viewer.values, subplots=(subplot,))) is None, subplot


def test_interactive_ionpoptype_belongs_to_each_populations_subplot() -> None:
    """The window sets the quantity of the ions for each populations subplot, thus --ionpoptype goes to them."""
    tokens = [
        "Te",
        str(modelpath),
        "--timestep",
        "50",
        "--plot",
        "populations",
        "Fe II",
        "Fe III",
        "--ionpoptype",
        "elpop",
    ]
    viewer = make_headless_viewer([*tokens, "--interactive"])
    assert viewer.values.subplots == (("Te",), ("populations", "Fe II", "Fe III", "ionpoptype=elpop"))
    assert "--ionpoptype" not in viewer.get_plot_tokens()
    # the default quantity needs no directive, and each subplot keeps its own
    assert viewer.parser.get_default("poptype") == subplotcards.DEFAULT_POPTYPE
    subplots = (("populations", "Fe II"), ("populations", "Fe III", "ionpoptype=totalpop"))
    assert subplotcards.move_poptype_to_subplots(
        subplots, (("--ionpoptype", ("elpop",)),), viewer.modeldata.estimatorcolumns
    ) == ((("populations", "Fe II", "ionpoptype=elpop"), ("populations", "Fe III", "ionpoptype=totalpop")), ())
    # with no populations subplot the option stays, thus a populations subplot that the user adds later takes it
    rows = (("--ionpoptype", ("elpop",)),)
    assert subplotcards.move_poptype_to_subplots((("Te",),), rows, viewer.modeldata.estimatorcolumns) == (
        (("Te",),),
        rows,
    )
    teviewer = make_headless_viewer([
        "Te",
        str(modelpath),
        "--timestep",
        "50",
        "--ionpoptype",
        "elpop",
        "--interactive",
    ])
    assert "--ionpoptype" in teviewer.get_plot_tokens()


def test_interactive_geometry_modes_draw() -> None:
    """Each way to select the cells of a 3D model gives a command that plotestimators draws."""
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    for mode in interactive.GEOMETRY_MODES:
        values = interactive.set_geometry_mode(viewer, viewer.values, mode)
        assert interactive.get_geometry_mode(values) == mode
        assert apply_values(viewer, values) is None, (mode, viewer.get_command())
    # a colour image is a snapshot, thus a plot against time becomes one
    assert apply_values(viewer, interactive.set_geometry_mode(viewer, viewer.values, "cells")) is None
    evolution = interactive.get_evolution_values(viewer, str(viewer.modeldata.cells[0]))
    assert apply_values(viewer, evolution) is None
    image = interactive.set_geometry_mode(viewer, viewer.values, "plane")
    assert not interactive.is_evolution(image)
    assert not image.cells
    assert apply_values(viewer, image) is None

    assert interactive.get_slice_parts("z=-0.2c") == ("xy", "-0.2c")
    assert interactive.get_slice_parts("xz") == ("xz", "")
    # plotestimators reads these forms, and the plane box showed xy for each of them
    assert interactive.get_slice_parts("zx") == ("xz", "")
    assert interactive.get_slice_parts("zy") == ("yz", "")
    assert interactive.get_slice_parts(" z = 0.1c ") == ("xy", "0.1c")
    assert interactive.get_slice_text("xz", "5000km/s") == "y=5000km/s"
    assert interactive.get_line_axis("z=0,y=0") == "x"
    assert interactive.get_line_label("x", "z=0,y=0") == "x (y=z=0)"
    assert interactive.get_line_label("x", "z=0.1c,y=0") == "x (y=0, z=0.1c)"
    assert interactive.get_line_label("z", "z=0.1c,y=0") == "z (x=y=0)"

    # the text below the controls gives the range of each coordinate that the selected cells cover. The test model
    # has 10 cells on each axis from -0.09647c to 0.09647c, thus each cell is 0.01929c wide
    def describe(mode: str, slicetext: str | None = None, dimensions: int = 3) -> str:
        values = interactive.set_geometry_mode(viewer, viewer.values, mode)
        rows = (("--slice", (slicetext,)),) if slicetext is not None else values.otheroptions
        values = dc.replace(values, otheroptions=viewercore.set_row_values(rows, {"--axis": ("-y",)}))
        modelmeta = viewer.modelmeta | {"dimensions": dimensions}
        return interactive.get_geometry_description(viewer.get_plot_tokens(values), modelmeta)

    assert describe("alongaxis") == (
        "The half line of cells from the centre along the -y axis, with 0 ≤ x < 0.01929c, 0 ≤ z < 0.01929c, and"
        " -0.09647c ≤ y < 0."
    )
    assert describe("cone") == (
        "The cells whose centre lies within 15° of the -y half-axis: -y_c ≥ √(x_c² + z_c²) / tan 15°."
    )
    # 1 / tan 90° is not 0 in floating point, thus the cone of 180° leaves out the central plane
    values180 = interactive.set_geometry_mode(viewer, viewer.values, "cone")
    values180 = dc.replace(values180, otheroptions=(*values180.otheroptions, ("--coneangle", ("180",))))
    assert interactive.get_geometry_description(viewer.get_plot_tokens(values180), viewer.modelmeta) == (
        "The cells on the side of the +z half-axis: z_c > 0, and the cell at the centre."
    )
    assert describe("plane") == "The 2D plane slice of cells with 0 ≤ z < 0.01929c, as an image in x and y."
    # a position between two edges selects the layer that holds it, in the unit of the position
    assert describe("plane", "z=0.02c").startswith("The 2D plane slice of cells with 0.01929c ≤ z < 0.03859c")
    assert describe("plane", "y=5000km/s").startswith("The 2D plane slice of cells with 0 ≤ y < 5784.04 km/s")
    # plotestimators rejects a position outside the grid. The text gave a layer inside the grid, or an IndexError
    for position in ("-0.2c", "0.2c", "-1e5km/s"):
        assert "outside the grid of |z| <" in describe("plane", f"z={position}"), position
    assert describe("line", "z=0.09c,y=-0.02c") == (
        "The full line of cells through the grid along the x axis with -0.03859c ≤ y < -0.01929c and"
        " 0.07717c ≤ z < 0.09647c, against v_x."
    )
    # a line keeps a -x of the user, e.g. time, and the text named v_x
    linevalues = dc.replace(viewer.values, x="time", otheroptions=(("--slice", ("z=0,y=0",)),))
    linetokens = viewer.get_plot_tokens(linevalues)
    assert interactive.get_geometry_description(linetokens, viewer.modelmeta).endswith("against -x time.")
    assert "Δr = 0.01929c" in describe("average")
    assert describe("projection").startswith(
        "Pixel (i, j) is the mean of the line of cells along z in layer i of x and layer j of y. Each layer is"
        " 0.01929c wide."
    )
    assert "n_element for an average ion charge" in describe("projection")
    assert "√(r² + z²)" in describe("average", dimensions=1)
    # plotestimators leaves out the corners of the grid, thus "all the cells" of a 3D model was wrong
    assert describe("all").startswith("The cells whose centre has √(x_c² + y_c² + z_c²) ≤ v_max = 0.09647c.")
    assert not describe("all", dimensions=1)
    # an odd number of cells has its middle cell edge at -Δ/2, which plotestimators takes for a half line
    oddgrid = {"ncoordgridx": 11, "ncoordgridy": 11, "ncoordgridz": 11, "vmax_cmps": 2297436514.4767036}
    oddvalues = interactive.set_geometry_mode(viewer, viewer.values, "alongaxis")
    oddtext = interactive.get_geometry_description(viewer.get_plot_tokens(oddvalues), viewer.modelmeta | oddgrid)
    assert "-0.006967c ≤ x < 0.006967c" in oddtext
    rows = (("--slice", ("xy",)), ("--coneangle", ("20",)))
    assert viewercore.set_row_values(rows, {"--slice": None, "--axis": ("-x",), "--coneangle": ("40",)}) == (
        ("--coneangle", ("40",)),
        ("--axis", ("-x",)),
    )
    # the axes of a colour image are velocities, and plotestimators rejected an image of a snapshot against rho
    rhovalues = viewer.set_xvariable(interactive.set_geometry_mode(viewer, viewer.values, "all"), "rho")
    assert apply_values(viewer, rhovalues) is None
    for mode in ("plane", "average"):
        assert apply_values(viewer, interactive.set_geometry_mode(viewer, rhovalues, mode)) is None, mode

    # a 1D model has the average around the z axis, and the selector of the window did not offer it
    assert interactive.get_geometry_choices(1) == ["all", "cells", "average"]
    viewer1d = make_headless_viewer(["Te", str(modelpath), "--timestep", "50", "--interactive"])
    assert apply_values(viewer1d, interactive.set_geometry_mode(viewer1d, viewer1d.values, "average")) is None

    # a projection averages each line of cells along its axis, thus it needs a 3D model and no -slice
    projection = dc.replace(viewer.values, otheroptions=(("--projection", ("y",)),))
    assert apply_values(viewer, projection) is None
    assert interactive.get_geometry_mode(viewer.values) == "projection"
    assert [frame.get_xlabel() for frame in interactive.get_plot_frames(viewer.fig)] == ["v$_x$ [$c$]"]
    rejected = apply_values(
        viewer, dc.replace(projection, otheroptions=(("--projection", ("y",)), ("--slice", ("xy",))))
    )
    assert rejected is not None
    assert "--slice" in rejected
    onedprojection = dc.replace(viewer1d.values, otheroptions=(("--projection", ("z",)),))
    message = apply_values(viewer1d, onedprojection)
    assert message is not None
    assert "needs a 3D model" in message


def test_interactive_smoothing_and_section_rows() -> None:
    """A smoothing draws, and the rows of the sections and of the output stay in the command.

    The window dropped --dpi, thus a PDF file lost the resolution of its raster parts. The table hid --verbose, thus
    the user could not remove it.
    """
    viewer = make_headless_viewer([
        "Te",
        str(modelpath_classic_3d),
        "-t",
        "5",
        "--dpi",
        "300",
        "--verbose",
        "--interactive",
    ])
    assert {"--dpi", "--verbose"} <= set(viewer.get_plot_tokens())
    assert "--verbose" not in viewer.modelformatflags
    for mode, numbers in (("movingavg", (3,)), ("savgol", (5, 2)), ("none", ())):
        rows = interactive.set_smoothing(viewer.values.otheroptions, mode, (*numbers, 2)[:2] if numbers else (5, 2))
        assert interactive.get_smoothing(rows) == (mode, numbers)
        assert apply_values(viewer, dc.replace(viewer.values, otheroptions=(*rows, ("--notitle", ())))) is None
    assert "--notitle" in viewer.get_plot_tokens()
    # the option table shows the rows that a section sets too, and only the row of --classicartis stays out
    assert "--notitle" not in viewer.modelformatflags
    assert viewer.modelformatflags == {"--classicartis"}


def test_interactive_level_populations() -> None:
    """A model with NLTE populations offers the level populations, with the lowest NLTE levels of each ion.

    The choices came from the atomic data, thus they gave levels with no NLTE population, e.g. Fe V 3 and each level
    of Ni. A model with no NLTE populations offered averageexcitation, which reads them.
    """
    viewer = make_headless_viewer([str(modelpath), "-timestep", "50", "--interactive"])
    assert viewer.nltetypes == ("averageexcitation", "levelpopulation", "levelpopulation_dn_on_dvel")
    columns = viewer.modeldata.estimatorcolumns
    assert "levelpopulation" in subplotcards.get_subplot_types(columns, viewer.nltetypes)
    levelnames = subplotcards.get_level_names(viewer.modelpath, 50, viewer.modeldata.cells[0])
    assert levelnames[:2] == ["Fe I 0", "Fe I 1"]
    # the NLTE populations hold only the ground level of the top ion, and no level of Ni
    assert "Fe V 0" in levelnames
    assert "Fe V 1" not in levelnames
    assert not any(name.startswith("Ni") for name in levelnames)
    # timestep 0 comes before the first NLTE timestep, and the other timesteps of the cell then give the levels
    assert subplotcards.get_level_names(viewer.modelpath, 0, viewer.modeldata.cells[0]) == levelnames
    subplot = subplotcards.change_subplot_type(("Te",), "levelpopulation", columns, levelnames)
    assert subplot == ("levelpopulation", "Fe I 0")
    assert subplotcards.get_series_suggestions(subplot, columns, levelnames)[:2] == ["Fe I 1", "Fe I 2"]
    for name in (subplot[1], "Fe V 0", levelnames[-1]):
        assert apply_values(viewer, dc.replace(viewer.values, subplots=(("levelpopulation", name),))) is None, name

    classicviewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    assert not classicviewer.nltetypes
    assert "averageexcitation" not in subplotcards.get_subplot_types(classicviewer.modeldata.estimatorcolumns)
    # each series type of plotestimators has a family of columns, or it reads the NLTE populations
    assert set(plotitems.SERIESTYPES) <= {*subplotcards.SPECIES_FAMILIES, *subplotcards.NLTE_SERIESTYPES}


def test_subplots_per_row_fill_each_row() -> None:
    """-subplotsperrow puts the subplots in rows from the left, and the lowest subplot of each column has x labels.

    A column whose last place is empty takes its x labels on the subplot above that place.
    """
    viewer = make_headless_viewer(["Te", str(modelpath), "-timestep", "50", "--interactive"])
    subplots = (("Te",), ("TR",), ("nne",))
    for count, expectedcolumns in ((None, 1), (2, 2), (5, 3)):
        rows = () if count is None else (("-subplotsperrow", (str(count),)),)
        assert apply_values(viewer, dc.replace(viewer.values, subplots=subplots, otheroptions=rows)) is None, count
        frames = interactive.get_plot_frames(viewer.fig)
        assert len(frames) == 3
        lefts = sorted({round(frame.get_position().x0, 6) for frame in frames})
        assert len(lefts) == expectedcolumns, count
        haslabel = [bool(frame.get_xlabel()) for frame in frames]
        assert haslabel == [index + expectedcolumns >= 3 for index in range(3)], count


def test_interactive_image_panels_name_their_subplot() -> None:
    """Each panel of a colour image names its subplot, thus Set current range finds the colour scale of each subplot.

    A subplot of two ions gives two panels, and a subplot that an image cannot show gives none.
    """
    viewer = make_headless_viewer([
        "Te",
        str(modelpath_classic_3d),
        "-t",
        "5",
        "-slice",
        "xy",
        "-plot",
        "populations",
        "Fe II",
        "Fe III",
        "--interactive",
    ])
    assert viewer.isimage
    labels = [ax.get_label() for ax in interactive.get_plot_frames(viewer.fig)]
    assert labels == [plotestimators.get_panel_axes_label(index) for index in (0, 1, 1)]


def test_image_shows_the_average_ionisation() -> None:
    """A colour image shows the average ion charge of an element. The image left out a subplot of averageionisation.

    The mean over the cells of a pixel takes the number density of the element as a weight, as the line does. Thus
    the pixel gives the charge of each nucleus of the pixel.
    """
    viewer = make_headless_viewer([
        "Te",
        str(modelpath_classic_3d),
        "-t",
        "5",
        "-slice",
        "xy",
        "-plot",
        "averageionisation",
        "Fe",
        "--interactive",
    ])
    frames = interactive.get_plot_frames(viewer.fig)
    assert [frame.get_label() for frame in frames] == [plotestimators.get_panel_axes_label(index) for index in (0, 1)]
    chargearray = frames[1].collections[0].get_array()
    assert chargearray is not None
    charges = np.ma.compressed(chargearray)
    assert charges.size
    assert np.all((charges >= 0.0) & (charges <= 26.0))

    # two cells in one pixel: 1e3 Fe nuclei at a charge of 1 and 1e5 Fe nuclei at a charge of 3
    dfcells = pl.DataFrame({
        "nnelement_Fe": [1e3, 1e5],
        "nnion_Fe_II": [1e3, 0.0],
        "nnion_Fe_IV": [0.0, 1e5],
        "deltavol_deltat": [1.0, 1.0],
    })
    panels = plotestimators.get_image_panels([[["averageionisation", ["Fe"]]]], dfcells.columns, "absolute")
    mean = dfcells.select(plotestimators.get_panel_means(panels)).item()
    assert np.isclose(mean, (1e3 * 1 + 1e5 * 3) / (1e3 + 1e5), rtol=1e-12)


def test_interactive_initial_abundance_of_an_isotope() -> None:
    """An isotope such as Fe52 has its own column of initial abundances, and not the column of its element.

    plotestimators removed the mass number and plotted the abundance of Fe with the label Fe52.
    """
    viewer = make_headless_viewer([str(modelpath), "-timestep", "50", "--interactive"])
    assert {"Fe52", "Fe"} <= set(subplotcards.get_species_choices("initabundances", viewer.modeldata.estimatorcolumns))
    assert apply_values(viewer, dc.replace(viewer.values, subplots=(("initabundances", "Fe52", "Fe"),))) is None
    ydata = {line.get_label(): np.asarray(line.get_ydata()) for ax in viewer.fig.axes for line in ax.get_lines()}
    fe52 = at.inputmodel.scan_modeldata(modelpath)[0].select("X_Fe52").collect().item()
    assert np.allclose(ydata["Fe52"], fe52, rtol=1e-6, atol=1e-30)
    assert not np.allclose(ydata["Fe52"], ydata["Fe"], rtol=1e-6, atol=1e-30)


def test_interactive_x_variable_with_no_value_names_the_variable() -> None:
    """A column with no value in the rows of the plot, e.g. tmid_days_prevtimestep at timestep 0, gives its name.

    The message said that the estimators held no row, although the rows were there.
    """
    viewer = make_headless_viewer(["Te", str(modelpath_classic_3d), "-t", "5", "--interactive"])
    first = viewer.modeldata.validtimesteps[0]
    message = apply_values(viewer, dc.replace(viewer.values, x="tmid_days_prevtimestep", first=first, last=first))
    assert message is not None
    assert "tmid_days_prevtimestep has no value" in message


def test_variable_menu_groups_the_columns() -> None:
    """The menu of the variables gives the temperatures first, then the other variables, then a submenu per family.

    A species column belongs to a subplot of its series type, and a column of the grid is not a variable.
    """
    columns = ("timestep", "modelgridindex", "Te", "TR", "nne", "rho", "heating_ff", "heating_bf", "nnion_Fe_II")
    groups = subplotcards.get_variable_menu_groups(columns)
    assert groups[:2] == (("", ("Te", "TR")), ("", ("nne", "rho")))
    assert len(groups) == 3
    assert groups[2][0].endswith("(heating_…)")
    assert groups[2][1] == ("heating_bf", "heating_ff")


def test_card_state_follows_its_subplot() -> None:
    """A collapsed card and a card with an open y axis stay with their subplot after each change of the subplots.

    The window moved the rows only for a move, an insert, or a delete of the window. After Undo of a delete, the
    window collapsed a different card.
    """
    before = (("Te",), ("TR",), ("nne",))
    # a delete of the first subplot, then Undo of the delete
    assert subplotcards.get_moved_rows(before, before[1:], {2}) == {1}
    assert subplotcards.get_moved_rows(before[1:], before, {1}) == {2}
    # a move of the last subplot to the top
    assert subplotcards.get_moved_rows(before, (("nne",), ("Te",), ("TR",)), {0, 2}) == {0, 1}
    # a new y scale changes the items of the subplot, and the subplot keeps its row
    assert subplotcards.get_moved_rows(before, (("Te",), ("TR", "yscale=log"), ("nne",)), {1}) == {1}
    # the change removes the subplot of the row
    assert not subplotcards.get_moved_rows(before, before[1:], {0})
