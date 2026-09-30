"""Make the list of the series and the style dialog of a series."""

import argparse
import contextlib
import typing as t
from functools import partial
from pathlib import Path
from types import MappingProxyType

from artistools.misc.cliutils import dashes_arg
from artistools.misc.remote import is_remote_path

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    from PySide6 import QtGui
    from PySide6 import QtWidgets


from artistools.viewertools.application import add_recent_model
from artistools.viewertools.application import get_recent_models
from artistools.viewertools.application import set_drop_handler
from artistools.viewertools.widgets import copy_text
from artistools.viewertools.widgets import make_completer
from artistools.viewertools.widgets import make_elided_label
from artistools.viewertools.widgets import make_glyph_button
from artistools.viewertools.widgets import make_reorder_list

# the line styles of the style dialog: the value of -linestyle and the text of the box
LINESTYLE_CHOICES: t.Final = (("solid", "Solid"), ("dashed", "Dashed"), ("dotted", "Dotted"), ("dashdot", "Dash-dot"))

# the short names of the line styles of matplotlib, which a command can also give
LINESTYLE_ALIASES: t.Final = MappingProxyType({"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot"})


def get_dash_pattern(linestyle: str | None, dashes: str | None) -> list[float] | None:
    """Return the dash pattern of a line in units of its width, as matplotlib draws it, or None for a solid line.

    -dashes gives the pattern and replaces the line style. matplotlib scales a pattern by the width of the line, and a
    pattern of Qt has the same unit.
    """
    import matplotlib as mpl

    if dashes:
        with contextlib.suppress(ValueError, argparse.ArgumentTypeError):
            return list(dashes_arg(dashes))
    match LINESTYLE_ALIASES.get(linestyle or "", linestyle or "solid"):
        case "dashed":
            pattern = mpl.rcParams["lines.dashed_pattern"]
        case "dotted":
            pattern = mpl.rcParams["lines.dotted_pattern"]
        case "dashdot":
            pattern = mpl.rcParams["lines.dashdot_pattern"]
        case _:
            return None
    return [float(length) for length in pattern]


def make_line_swatch(
    colour: str, alpha: float, linewidth: float, dashpattern: "Sequence[float] | None"
) -> "QtGui.QPixmap":
    """Return a short image of a line with the colour, the opacity, the width, and the dash pattern of a series."""
    import matplotlib.colors as mplcolors
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    width, height = 36, 14
    ratio = QtWidgets.QApplication.primaryScreen().devicePixelRatio() if QtWidgets.QApplication.primaryScreen() else 1.0
    pixmap = QtGui.QPixmap(round(width * ratio), round(height * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    red, green, blue = (round(255 * part) for part in mplcolors.to_rgb(colour))
    pen = QtGui.QPen(QtGui.QColor(red, green, blue, round(255 * alpha)))
    # a very wide line fills the image, thus the width stays inside the height of the image
    pen.setWidthF(min(max(linewidth, 0.5), 5.0))
    # matplotlib ends each dash with no cap, and the square cap of Qt closes a short gap
    pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
    if dashpattern:
        pen.setDashPattern(list(dashpattern))
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(pen)
    painter.drawLine(QtCore.QPointF(2.0, height / 2.0), QtCore.QPointF(width - 2.0, height / 2.0))
    painter.end()
    return pixmap


def make_series_swatch(colour: str, style: "Mapping[str, str | None]") -> "QtGui.QPixmap":
    """Return the image of the line of a series from its colour and the values of its series style options."""
    import matplotlib as mpl

    return make_line_swatch(
        colour,
        float(style.get("-linealpha") or 1.0),
        float(style.get("-linewidth") or mpl.rcParams["lines.linewidth"]),
        get_dash_pattern(style.get("-linestyle"), style.get("-dashes")),
    )


# the options of the style dialog of a series, which each give one value for each series
SERIES_LINE_FLAGS: t.Final = ("-color", "-linestyle", "-dashes", "-linewidth", "-linealpha")


def edit_series_style(
    parent: "QtWidgets.QWidget",
    name: str,
    style: "Mapping[str, str | None]",
    defaultcolour: str,
    defaultlinewidth: float,
    flags: "Collection[str]" = SERIES_LINE_FLAGS,
) -> dict[str, str | None] | None:
    """Ask for the line style of one series, and return the value of each option of flags.

    style gives the current value of each option, or None for the default of the command. A field with the default
    gives None, thus the command then gives the series no value. A cancelled dialog gives None. The dialog shows only
    the fields of flags, e.g. a series of markers has no line style.
    """
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(f"Style of {name}")
    form = QtWidgets.QFormLayout(dialog)
    chosencolour: list[str | None] = [style.get("-color")]

    colourbutton = QtWidgets.QPushButton()
    colourbutton.setToolTip("Select the colour of the line (-color)")
    defaultcolourbutton = QtWidgets.QPushButton("Default")
    defaultcolourbutton.setToolTip("Give the line the colour of the command")
    colourrow = QtWidgets.QHBoxLayout()
    colourrow.addWidget(colourbutton, 1)
    colourrow.addWidget(defaultcolourbutton)
    form.addRow("Colour:", colourrow)

    linestylebox = QtWidgets.QComboBox()
    linestylebox.addItem("Default", None)
    for value, text in LINESTYLE_CHOICES:
        linestylebox.addItem(text, value)
    linestyle = style.get("-linestyle")
    linestylebox.setCurrentIndex(max(linestylebox.findData(LINESTYLE_ALIASES.get(linestyle or "", linestyle)), 0))
    linestylebox.setToolTip("The line style (-linestyle). A dash pattern replaces it")
    form.addRow("Line style:", linestylebox)

    dashesedit = QtWidgets.QLineEdit(style.get("-dashes") or "")
    dashesedit.setPlaceholderText("none, e.g. 5,2")
    dashesedit.setToolTip(
        "The lengths of each dash and each gap in units of the line width (-dashes), e.g. 5,2. The pattern replaces"
        " the line style"
    )
    form.addRow("Dash pattern:", dashesedit)

    def make_default_spinbox(high: float, step: float, value: str | None, tooltip: str) -> QtWidgets.QDoubleSpinBox:
        # the lowest value of the box shows "Default", which gives the series no value
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(0.0, high)
        box.setSingleStep(step)
        box.setDecimals(2)
        box.setSpecialValueText("Default")
        box.setValue(float(value) if value else 0.0)
        box.setToolTip(tooltip)
        return box

    widthbox = make_default_spinbox(10.0, 0.25, style.get("-linewidth"), "The width of the line in points (-linewidth)")
    form.addRow("Width:", widthbox)
    alphabox = make_default_spinbox(
        1.0, 0.05, style.get("-linealpha"), "The opacity of the line, from 0 (clear) to 1 (opaque) (-linealpha)"
    )
    form.addRow("Opacity:", alphabox)

    previewlabel = QtWidgets.QLabel()
    form.addRow("Preview:", previewlabel)
    errorlabel = QtWidgets.QLabel()
    errorlabel.setStyleSheet("color: red;")
    errorlabel.hide()
    form.addRow(errorlabel)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok
        | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        | QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults
    )
    form.addRow(buttons)

    def get_dashes() -> str | None:
        """Return the dash pattern of the field, or None for an empty field. Raise ValueError for a bad pattern."""
        text = dashesedit.text().strip()
        if not text:
            return None
        try:
            return ",".join(format(length, "g") for length in dashes_arg(text))
        except argparse.ArgumentTypeError as exc:
            raise ValueError(str(exc)) from exc

    def show_preview() -> None:
        colour = chosencolour[0] or defaultcolour
        colourbutton.setIcon(QtGui.QIcon(make_line_swatch(colour, 1.0, 5.0, None)))
        colourbutton.setText(colour if chosencolour[0] else f"Default ({colour})")
        try:
            dashes = get_dashes()
        except ValueError as exc:
            errorlabel.setText(str(exc))
            errorlabel.show()
            dashes = None
        else:
            errorlabel.hide()
        previewlabel.setPixmap(
            make_line_swatch(
                colour,
                alphabox.value() or 1.0,
                widthbox.value() or defaultlinewidth,
                get_dash_pattern(linestylebox.currentData(), dashes) if "-linestyle" in flags else None,
            )
        )

    def on_colour() -> None:
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(chosencolour[0] or defaultcolour), dialog, "Line colour")
        if colour.isValid():
            chosencolour[0] = colour.name()
            show_preview()

    def on_default_colour() -> None:
        chosencolour[0] = None
        show_preview()

    def on_restore_defaults() -> None:
        chosencolour[0] = None
        linestylebox.setCurrentIndex(0)
        dashesedit.clear()
        widthbox.setValue(0.0)
        alphabox.setValue(0.0)
        show_preview()

    def on_accept() -> None:
        # a bad dash pattern keeps the dialog open, and the red text gives the reason
        try:
            get_dashes()
        except ValueError:
            dashesedit.setFocus()
            return
        dialog.accept()

    colourbutton.clicked.connect(on_colour)
    defaultcolourbutton.clicked.connect(on_default_colour)
    linestylebox.currentIndexChanged.connect(show_preview)
    dashesedit.textChanged.connect(show_preview)
    widthbox.valueChanged.connect(show_preview)
    alphabox.valueChanged.connect(show_preview)
    buttons.accepted.connect(on_accept)
    buttons.rejected.connect(dialog.reject)
    if (restorebutton := buttons.button(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults)) is not None:
        restorebutton.clicked.connect(on_restore_defaults)
    for flag, field in (
        ("-color", colourrow),
        ("-linestyle", linestylebox),
        ("-dashes", dashesedit),
        ("-linewidth", widthbox),
        ("-linealpha", alphabox),
    ):
        form.setRowVisible(field, flag in flags)
    show_preview()

    accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    # the parent keeps its children until it closes, thus the dialog goes when the event loop runs again
    dialog.deleteLater()
    if not accepted:
        return None
    values = {
        "-color": chosencolour[0],
        "-linestyle": linestylebox.currentData(),
        "-dashes": get_dashes(),
        "-linewidth": format(widthbox.value(), "g") if widthbox.value() else None,
        "-linealpha": format(alphabox.value(), "g") if alphabox.value() else None,
    }
    return {flag: value for flag, value in values.items() if flag in flags}


class SeriesRow(t.NamedTuple):
    """One row of the list of series of a viewer, which is an ARTIS model or a file of reference data."""

    path: str
    # the name of the series in the legend, and True if -label gives it
    name: str
    labelled: bool
    # the kind and the full path of the series, e.g. "Model: /data/mymodel"
    itemtext: str
    tooltip: str
    swatch: "QtGui.QPixmap"
    # the reason that the user cannot remove the row, e.g. for the last model, or None
    removereason: str | None
    # a glyph after the path, e.g. the mark of the model that gives the timesteps, and its tooltip
    mark: tuple[str, str] | None
    # the menu items of one viewer, with the text, whether the item is on, and the action
    extraactions: "tuple[tuple[str, bool, Callable[[], object]], ...]"


class ReferenceData(t.NamedTuple):
    """The reference data of the series of a viewer, e.g. the observed spectra."""

    # the kind of one series, e.g. "reference spectrum"
    kind: str
    names: "Sequence[str]"
    folder: Path
    # return the file of a name in the working folder or in the reference data, or None
    find: "Callable[[str], Path | None]"
    # return the token of the command for the path of a file, e.g. a name of the reference data
    get_token: "Callable[[str], str]"
    example: str


class SeriesListActions(t.NamedTuple):
    """The functions of a viewer that the list of series calls."""

    get_paths: "Callable[[], Sequence[str]]"
    # apply a new list of paths, e.g. a new order, an added path, or a removed path
    apply_paths: "Callable[[Sequence[str]], None]"
    edit_label: "Callable[[str], None]"
    edit_style: "Callable[[str], None]"
    # return the full path of a series, e.g. of ".", thus two spellings of one series give one row
    get_full_path: "Callable[[str], Path]"
    # return True for the folder of an ARTIS run. A plot needs one run at least
    is_run: "Callable[[str], bool]"
    show_error: "Callable[[str], None]"


def add_series_list(
    grid: "QtWidgets.QGridLayout",
    window: "QtWidgets.QMainWindow",
    show_note: "Callable[[str], None]",
    listtooltip: str,
    reference: ReferenceData,
    actions: SeriesListActions,
) -> "Callable[[object, Callable[[], Sequence[SeriesRow]]], None]":
    """Add the list of the series of a viewer, with the row that adds a model or a file of reference data.

    Each row shows the number, the image of the line, the name, and the path of a series, and buttons that move it
    and remove it. A double-click gives the series a -label, and the context menu of a row gives each action. A drag
    or Alt-Up and Alt-Down change the order. A folder or a file that the user drops on the window adds a series.

    Return the function that shows the rows. It takes a key of the parts of the values that the rows show, and it
    makes the rows again only for a new key, because each row reads the name of its series.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    def move_row(row: int, step: int) -> None:
        """Move the series of a row one row up or down, and select it."""
        paths = list(actions.get_paths())
        if row < 0 or not 0 <= row + step < len(paths):
            return
        paths.insert(row + step, paths.pop(row))
        actions.apply_paths(paths)
        serieslist.setCurrentRow(row + step)

    serieslist = make_reorder_list(lambda step: move_row(serieslist.currentRow(), step))
    # the widget of each row shows the text beside its buttons, thus the list draws no text of its own
    serieslist.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    serieslist.setToolTip(listtooltip)
    # a click opens the dialog, and the arrow opens the menu of the recent models
    addmodelbutton = QtWidgets.QToolButton()
    addmodelbutton.setText("Add Model…")
    addmodelbutton.setToolTip("Add the folder of an ARTIS model. The arrow shows the recent models")
    addmodelbutton.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
    recentmodelsmenu = QtWidgets.QMenu(addmodelbutton)
    addmodelbutton.setMenu(recentmodelsmenu)
    referenceedit = QtWidgets.QLineEdit()
    referenceedit.setPlaceholderText(f"Add a {reference.kind}, e.g. {reference.example}")
    referenceedit.setToolTip(
        f"Type part of the name of a {reference.kind} in the data of artistools, then press Return. A name of a file"
        " in the working folder also works."
    )
    referencecompleter = make_completer(reference.names, referenceedit)
    referenceedit.setCompleter(referencecompleter)
    openreferencebutton = QtWidgets.QPushButton("Open…")
    openreferencebutton.setToolTip(f"Add the file of a {reference.kind} from a folder")
    addrow = QtWidgets.QHBoxLayout()
    addrow.addWidget(referenceedit, 1)
    addrow.addWidget(openreferencebutton)
    addrow.addWidget(addmodelbutton)
    grid.addWidget(serieslist, 0, 0, 1, -1)
    grid.addLayout(addrow, 1, 0, 1, -1)

    # the key and the paths of the rows that show_rows made. A drag moves the rows, and the rows then show old numbers
    # and old buttons until show_rows makes them again
    shown: list[tuple[object, tuple[str, ...]] | None] = [None]

    def add_paths(paths: "Sequence[str]") -> None:
        """Add each series whose full path the list does not hold yet. A cancelled dialog gives no path."""
        if not paths:
            return
        current = list(actions.get_paths())
        fullpaths = {actions.get_full_path(path) for path in current}
        newpaths: list[str] = []
        for path in paths:
            if (fullpath := actions.get_full_path(path)) not in fullpaths:
                fullpaths.add(fullpath)
                newpaths.append(path)
        if not newpaths:
            actions.show_error("The list already holds each of these series")
            return
        actions.apply_paths((*current, *newpaths))

    def remove_path(path: str) -> None:
        paths = [other for other in actions.get_paths() if other != path]
        # the time controls read the timesteps of a run, thus the plot needs an ARTIS model
        if not any(actions.is_run(other) for other in paths):
            actions.show_error(
                "The plot needs one ARTIS model at least. Add a different model before you remove this one"
            )
            return
        actions.apply_paths(paths)

    def copy_path(path: str) -> None:
        copy_text(str(actions.get_full_path(path)))
        show_note("Copied the path")

    def open_folder(path: str) -> None:
        """Open the folder of a model, or the folder that holds a reference file, in the file manager."""
        fullpath = actions.get_full_path(path)
        folder = fullpath if fullpath.is_dir() else fullpath.parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def make_row(index: int, count: int, seriesrow: SeriesRow) -> QtWidgets.QWidget:
        path, name = seriesrow.path, seriesrow.name
        row = QtWidgets.QWidget()
        row.setToolTip(seriesrow.tooltip)
        rowlayout = QtWidgets.QHBoxLayout(row)
        rowlayout.setContentsMargins(4, 0, 2, 0)
        rowlayout.setSpacing(2)
        rowlayout.addWidget(QtWidgets.QLabel(f"<b>{index + 1}</b>"))
        swatch = QtWidgets.QToolButton()
        swatch.setAutoRaise(True)
        swatch.setIconSize(QtCore.QSize(36, 14))
        swatch.setIcon(QtGui.QIcon(seriesrow.swatch))
        swatch.setToolTip("The colour and the line style of the series in the plot. Click to change them")
        swatch.setAccessibleName(f"Set the style of {name}")
        # the new list replaces this row, thus each action of a button waits until the click ends
        swatch.clicked.connect(partial(QtCore.QTimer.singleShot, 0, window, partial(actions.edit_style, path)))
        rowlayout.addWidget(swatch)
        namelabel = QtWidgets.QLabel(name)
        namelabel.setToolTip(
            f"The -label of the series: {name}. Double-click the row to change it"
            if seriesrow.labelled
            else f"The name of the series in the legend: {name}. Double-click the row to give it a -label"
        )
        rowlayout.addWidget(namelabel)
        rowlayout.addSpacing(6)
        # a long path shows its start and its end, and the width of the box sets the length
        pathlabel = make_elided_label(seriesrow.itemtext)
        pathlabel.setEnabled(False)
        rowlayout.addWidget(pathlabel, 1)
        if seriesrow.mark is not None:
            marklabel = QtWidgets.QLabel(seriesrow.mark[0])
            marklabel.setToolTip(seriesrow.mark[1])
            rowlayout.addWidget(marklabel)
        for glyph, tooltip, enabled, action in (
            ("▲", f"Move {name} up (Alt-Up)", index > 0, partial(move_row, index, -1)),
            ("▼", f"Move {name} down (Alt-Down)", index < count - 1, partial(move_row, index, 1)),
            ("✕", f"Remove {name} from the plot", seriesrow.removereason is None, partial(remove_path, path)),
        ):
            button = make_glyph_button(glyph, tooltip, tooltip)
            button.setEnabled(enabled)
            if glyph == "✕" and seriesrow.removereason is not None:
                button.setToolTip(seriesrow.removereason)
            button.clicked.connect(partial(QtCore.QTimer.singleShot, 0, window, action))
            rowlayout.addWidget(button)
        grip = QtWidgets.QLabel("≡")
        grip.setEnabled(False)
        grip.setToolTip("Drag the row to move the series")
        grip.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        rowlayout.addWidget(grip)
        # the context menu gives each action, thus the keyboard and VoiceOver can also reach them
        row.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        for text, enabled, action in (
            ("Move to Top", index > 0, partial(move_row, index, -index)),
            ("Move to Bottom", index < count - 1, partial(move_row, index, count - 1 - index)),
            *seriesrow.extraactions,
            ("Set Label…", True, partial(actions.edit_label, path)),
            ("Set Style…", True, partial(actions.edit_style, path)),
            ("Copy Path", True, partial(copy_path, path)),
            ("Open Folder", not is_remote_path(path), partial(open_folder, path)),
            ("Remove", seriesrow.removereason is None, partial(remove_path, path)),
        ):
            rowaction = QtGui.QAction(text, row)
            rowaction.setEnabled(enabled)
            # the new list replaces this row, thus each action waits until the menu closes
            rowaction.triggered.connect(partial(QtCore.QTimer.singleShot, 0, window, action))
            row.addAction(rowaction)
        return row

    def show_rows(key: object, make_rows: "Callable[[], Sequence[SeriesRow]]") -> None:
        paths = tuple(actions.get_paths())
        order = tuple(
            serieslist.item(index).data(QtCore.Qt.ItemDataRole.UserRole) for index in range(serieslist.count())
        )
        if shown[0] == (key, paths) and order == paths:
            return
        rows = make_rows()
        serieslist.clear()
        rowheight = serieslist.fontMetrics().lineSpacing() + 4
        for index, seriesrow in enumerate(rows):
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, seriesrow.path)
            row = make_row(index, len(rows), seriesrow)
            rowheight = max(rowheight, row.sizeHint().height())
            serieslist.addItem(item)
            serieslist.setItemWidget(item, row)
        for index in range(serieslist.count()):
            if (item := serieslist.item(index)) is not None:
                item.setSizeHint(QtCore.QSize(0, rowheight))
        # the list has the height of its series, from 2 to 4 rows, and a longer list scrolls
        shownrows = min(max(serieslist.count(), 2), 4)
        serieslist.setFixedHeight(shownrows * rowheight + 2 * serieslist.frameWidth() + 4)
        shown[0] = (key, paths)

    def on_rows_dropped() -> None:
        """Apply the order of the rows after a drag.

        Qt can move a dropped row, or it can insert a copy and then remove the source row. Until the removal, the
        list holds one series two times, thus the function waits for the removal. A copy has no row widget, thus the
        next show_rows makes the rows again.
        """
        items = [serieslist.item(index) for index in range(serieslist.count())]
        order = [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in items]
        paths = list(actions.get_paths())
        if sorted(order) != sorted(paths):
            return
        if order == paths and all(serieslist.itemWidget(item) is not None for item in items):
            return
        # the rows show the numbers and the buttons of the old order, thus show_rows makes them again
        shown[0] = None
        actions.apply_paths(order)

    def get_start_folder() -> Path:
        # the dialog shows only local folders, thus it opens in the working folder for a first run on a different host
        runs = [path for path in actions.get_paths() if actions.is_run(path)]
        if not runs or is_remote_path(runs[0]):
            return Path.cwd()
        return actions.get_full_path(runs[0]).parent

    def add_model_folder(folder: str) -> None:
        if not actions.is_run(folder):
            actions.show_error(f"{folder} is not the folder of an ARTIS run, which holds input.txt")
            return
        add_recent_model(folder)
        add_paths([folder])

    def on_add_model() -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Add an ARTIS model", str(get_start_folder()))
        if folder:
            add_model_folder(folder)

    def show_recent_models() -> None:
        """Fill the menu of Add Model with the recent models that the list does not hold."""
        recentmodelsmenu.clear()
        fullpaths = {actions.get_full_path(path) for path in actions.get_paths()}
        folders = [folder for folder in get_recent_models() if actions.get_full_path(folder) not in fullpaths]
        for folder in folders:
            action = recentmodelsmenu.addAction(Path(folder).name)
            action.setToolTip(folder)
            # a test of a remote folder starts ssh, thus the menu enables each remote model and does not test its folder
            action.setEnabled(is_remote_path(folder) or Path(folder).is_dir())
            action.triggered.connect(partial(add_paths, [folder]))
        if not folders:
            recentmodelsmenu.addAction("No Recent Models").setEnabled(False)

    def on_open_reference() -> None:
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(window, f"Add a {reference.kind}", str(reference.folder))
        add_paths([reference.get_token(filename) for filename in filenames])

    def add_reference_name(name: str) -> None:
        name = name.strip()
        if not name:
            return
        if reference.find(name) is None:
            actions.show_error(f"No {reference.kind} {name} is in the working folder or in the reference data")
            return
        referenceedit.clear()
        add_paths([name])

    def on_complete_reference(name: str) -> None:
        # the completer puts the name in the field after this handler, thus clear the field after the event
        QtCore.QTimer.singleShot(0, referenceedit.clear)
        add_reference_name(name)

    def on_drop(paths: list[str]) -> None:
        """Add each dropped ARTIS run and each dropped file of reference data to the series of the plot."""
        folders = [path for path in paths if Path(path).is_dir()]
        runs = [folder for folder in folders if actions.is_run(folder)]
        if len(runs) < len(folders):
            actions.show_error("A dropped folder is not the folder of an ARTIS run, which holds input.txt")
        add_paths([*runs, *(reference.get_token(path) for path in paths if Path(path).is_file())])

    addmodelbutton.clicked.connect(on_add_model)
    recentmodelsmenu.aboutToShow.connect(show_recent_models)

    def on_double_click(item: QtWidgets.QListWidgetItem) -> None:
        actions.edit_label(item.data(QtCore.Qt.ItemDataRole.UserRole))

    serieslist.itemDoubleClicked.connect(on_double_click)
    # the list changes its rows at the end of the drop, thus the new order applies after the drop
    for rowsignal in (serieslist.model().rowsMoved, serieslist.model().rowsInserted, serieslist.model().rowsRemoved):
        rowsignal.connect(lambda: QtCore.QTimer.singleShot(0, window, on_rows_dropped))
    openreferencebutton.clicked.connect(on_open_reference)
    referencecompleter.activated.connect(on_complete_reference)
    referenceedit.returnPressed.connect(lambda: add_reference_name(referenceedit.text()))
    set_drop_handler(window, on_drop)
    return show_rows
