"""Make the list of the series and the dialog of the line properties of a series."""

import argparse
import contextlib
import getpass
import typing as t
from functools import cache
from functools import partial
from pathlib import Path
from types import MappingProxyType

from artistools.misc.cliutils import dashes_arg
from artistools.misc.remote import is_remote_path
from artistools.misc.remote import split_remote_path
from artistools.viewertools.application import add_recent_model
from artistools.viewertools.application import get_recent_models
from artistools.viewertools.application import set_drop_handler
from artistools.viewertools.widgets import copy_text
from artistools.viewertools.widgets import get_message_colours
from artistools.viewertools.widgets import make_completer
from artistools.viewertools.widgets import make_elided_label
from artistools.viewertools.widgets import make_glyph_button
from artistools.viewertools.widgets import make_reorder_list

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    from PySide6 import QtGui
    from PySide6 import QtWidgets


# the line styles of the dialog of the line properties: the value of -linestyle and the text of the box
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


# the pause after a change of the dialog of the line properties before the plot shows it
PREVIEW_MILLISECONDS: t.Final = 250

# the options of the dialog of the line properties, which each give one value for each series
SERIES_PROPERTY_FLAGS: t.Final = ("-label", "-color", "-linestyle", "-dashes", "-linewidth", "-linealpha")


def edit_series_properties(
    parent: "QtWidgets.QWidget",
    name: str,
    style: "Mapping[str, str | None]",
    defaultcolour: str,
    defaultlinewidth: float,
    show_changes: "Callable[[Mapping[str, str | None] | None, bool], None]",
    flags: "Collection[str]" = SERIES_PROPERTY_FLAGS,
) -> None:
    """Ask for the label and the line style of one series, and show each change in the plot at once.

    style gives the current value of each option, or None for the default of the command. A field with the default
    gives None, thus the command then gives the series no value. An empty label gives the automatic label of the
    command. The dialog shows only the fields of flags, e.g. a series of markers has no line style.

    show_changes receives the value of each option of flags, and whether the change is a new step of Undo. The first
    change is one step, thus Undo reverts all of the dialog. Cancel gives None, and the viewer then shows the values
    from before the dialog.
    """
    import matplotlib.colors as mplcolors
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(f"Line Properties of {name}")
    form = QtWidgets.QFormLayout(dialog)
    chosencolour: list[str | None] = [style.get("-color")]

    labeledit = QtWidgets.QLineEdit(style.get("-label") or "")
    labeledit.setPlaceholderText("automatic")
    labeledit.setToolTip("The label of the series in the legend (-label). Clear the field for the automatic label")
    labeledit.setMinimumWidth(240)
    form.addRow("Label:", labeledit)

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
        # a form of the macOS style keeps a field at its size hint, which leaves no space for the text "Default"
        box.setMinimumWidth(box.fontMetrics().horizontalAdvance("Default") + 56)
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
    errorlabel.setStyleSheet(f"color: {get_message_colours()[0]};")
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
        previewtimer.start()
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
        # Qt reads no colour name of matplotlib, e.g. "C1" or "tab:orange", thus the dialog takes the hexadecimal form
        try:
            startcolour = mplcolors.to_hex(chosencolour[0] or defaultcolour)
        except ValueError:
            startcolour = mplcolors.to_hex(defaultcolour)
        colour = QtWidgets.QColorDialog.getColor(QtGui.QColor(startcolour), dialog, "Line colour")
        if colour.isValid():
            chosencolour[0] = colour.name()
            show_preview()

    def on_default_colour() -> None:
        chosencolour[0] = None
        show_preview()

    def on_restore_defaults() -> None:
        labeledit.clear()
        chosencolour[0] = None
        linestylebox.setCurrentIndex(0)
        dashesedit.clear()
        widthbox.setValue(0.0)
        alphabox.setValue(0.0)
        show_preview()

    def get_widget_values() -> dict[str, str | None] | None:
        """Return the value of each option of flags that the fields show, or None for a bad dash pattern."""
        try:
            dashes = get_dashes()
        except ValueError:
            return None
        values = {
            "-label": labeledit.text().strip() or None,
            "-color": chosencolour[0],
            "-linestyle": linestylebox.currentData(),
            "-dashes": dashes,
            "-linewidth": format(widthbox.value(), "g") if widthbox.value() else None,
            "-linealpha": format(alphabox.value(), "g") if alphabox.value() else None,
        }
        return {flag: value for flag, value in values.items() if flag in flags}

    def get_changes() -> dict[str, str | None] | None:
        """Return the value of each option of flags, or None for a bad dash pattern.

        A field cannot show each value of the command, e.g. a width above the range of its box, an empty label that
        hides the series, or a line style that the box does not list. Thus an option keeps its value of the command
        until the user changes its field.
        """
        if (widgetvalues := get_widget_values()) is None:
            return None
        return {
            flag: value if value != openwidgetvalues.get(flag) else style.get(flag)
            for flag, value in widgetvalues.items()
        }

    # the values that the plot shows, and whether a change of the dialog made the step of Undo
    shownchanges: list[dict[str, str | None] | None] = [None]
    madestep = [False]

    def show_plot_changes() -> None:
        changes = get_changes()
        if changes is None or changes == shownchanges[0]:
            return
        show_changes(changes, not madestep[0])
        shownchanges[0], madestep[0] = changes, True

    # a typed label or a typed pattern gives a plot after a short pause, and not after each key
    previewtimer = QtCore.QTimer(dialog)
    previewtimer.setSingleShot(True)
    previewtimer.setInterval(PREVIEW_MILLISECONDS)
    previewtimer.timeout.connect(show_plot_changes)

    def on_accept() -> None:
        # a bad dash pattern keeps the dialog open, and the red text gives the reason
        if get_changes() is None:
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
        ("-label", labeledit),
        ("-color", colourrow),
        ("-linestyle", linestylebox),
        ("-dashes", dashesedit),
        ("-linewidth", widthbox),
        ("-linealpha", alphabox),
    ):
        form.setRowVisible(field, flag in flags)
    labeledit.textChanged.connect(previewtimer.start)
    show_preview()
    previewtimer.stop()
    openwidgetvalues = get_widget_values() or {}
    shownchanges[0] = get_changes()

    accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    previewtimer.stop()
    # the parent keeps its children until it closes, thus the dialog goes when the event loop runs again
    dialog.deleteLater()
    if accepted:
        show_plot_changes()
    elif madestep[0]:
        # the return to the values before the dialog reverts the step of the first change, thus it makes no step
        newstep = False
        show_changes(None, newstep)


def get_short_item_text(itemtext: str) -> str:
    """Return the kind and the name of the last folder or file of an item text, e.g. "Model: mymodel".

    A full path took most of the width of a row, and a long path showed only its start and its end.
    """
    kind, separator, path = itemtext.partition(": ")
    if not separator:
        return itemtext
    name = path.rstrip("/").rpartition("/")[2] or path
    return f"{kind}: {name}"


# the object name of a button of a row of the series list, which shows only under the pointer or when the list selects
# the row
HOVER_WIDGET_NAME: t.Final = "rowhoverwidget"


@cache
def get_hover_row_class() -> "type[QtWidgets.QWidget]":
    """Return the class of a row of the series list, which shows its buttons only under the pointer or if selected.

    The buttons of each row took the attention from the names. The context menu and the keys give the same actions,
    thus the keyboard and VoiceOver reach them while the buttons are hidden.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class HoverRow(QtWidgets.QWidget):
        """A row that shows its buttons under the pointer or while the list selects it."""

        @t.override
        def enterEvent(self, event: QtGui.QEnterEvent, /) -> None:
            super().enterEvent(event)
            show_row_buttons(self, visible=True)

        @t.override
        def leaveEvent(self, event: QtCore.QEvent, /) -> None:
            super().leaveEvent(event)
            show_row_buttons(self, visible=bool(self.property("rowselected")))

    return HoverRow


def make_hover_widget(widget: "QtWidgets.QWidget") -> None:
    """Make a widget of a row show only under the pointer or if the list selects the row. It keeps its place."""
    widget.setObjectName(HOVER_WIDGET_NAME)
    # a hidden button keeps its place, thus the name and the path do not move under the pointer
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)
    widget.hide()


def set_row_selected(row: "QtWidgets.QWidget", *, selected: bool) -> None:
    """Show the buttons of a row of the series list while the list selects it or the pointer is over it."""
    row.setProperty("rowselected", selected)
    show_row_buttons(row, visible=selected or row.underMouse())


def show_row_buttons(row: "QtWidgets.QWidget", *, visible: bool) -> None:
    """Show or hide the buttons of a row of the series list."""
    from PySide6 import QtWidgets

    for widget in row.findChildren(QtWidgets.QWidget, HOVER_WIDGET_NAME):
        widget.setVisible(visible)


def make_path_menu_action(menu: "QtWidgets.QMenu", folder: str, width: int) -> "QtGui.QAction":
    """Return a menu item with the name of a folder, and its parent folder in a very small font under the name.

    One item of a menu has one font, thus the item is a widget with two labels. The item is as wide as the width, e.g.
    the width of the list of series, or as the menu if the menu is wider. The parent folder fills the width of the item,
    and a longer path shows its start and its end, e.g. /Users/luke…/kilonova_runs. The tooltip gives the full path.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    action = QtWidgets.QWidgetAction(menu)
    item = get_path_menu_item_class()(action, menu)
    item.setObjectName("pathmenuitem")
    item.setToolTip(folder)
    layout = QtWidgets.QVBoxLayout(item)
    margin = 14
    layout.setContentsMargins(margin, 3, margin, 3)
    layout.setSpacing(0)
    item.setMinimumWidth(width)
    namelabel = QtWidgets.QLabel(Path(folder).name)
    # the label shortens the path to the width that the menu gives it, and a path that a fixed width shortened left a
    # space at the right of a wider menu
    pathlabel = make_elided_label(get_menu_parent_text(folder))
    pathlabel.setToolTip(folder)
    font = pathlabel.font()
    font.setPointSizeF(font.pointSizeF() * 0.75)
    pathlabel.setFont(font)
    pathlabel.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
    layout.addWidget(namelabel)
    layout.addWidget(pathlabel)
    item.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
    item.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    # the menu draws no highlight for a widget, thus the item takes the colours of a selected menu item. A style sheet
    # applies a state such as :hover only to the widget that it names and not to the labels in it, thus a property
    # gives the state. The palette greys the text of a disabled item
    item.setStyleSheet(
        '#pathmenuitem[highlighted="true"] { background: palette(highlight); }'
        ' #pathmenuitem[highlighted="true"] QLabel { color: palette(highlighted-text); }'
    )
    action.setDefaultWidget(item)
    return action


def get_menu_parent_text(folder: str) -> str:
    """Return the parent folder of a model for a menu, e.g. "vae26:~/short" for the model "vae26:~/short/mymodel".

    The first line of the menu item gives the name of the model folder, thus the second line leaves it out. A remote
    path gives its host and "~" for its home folder. The host gives its home folder only through ssh, thus the home
    folder of a remote user is /home/user or /Users/user. The user is the user of "user@host", or the local user.
    """
    if (remoteparts := split_remote_path(folder)) is None:
        return str(Path(folder).parent)
    host, hostpath = remoteparts
    user = host.partition("@")[0] if "@" in host else getpass.getuser()
    hostpathtext = str(hostpath.parent)
    for home in (f"/home/{user}", f"/Users/{user}"):
        hostpathtext = get_path_with_tilde(hostpathtext, home)
    return f"{host}:{hostpathtext}"


def get_path_with_tilde(path: str, home: str) -> str:
    """Return the path with "~" in place of the home folder at its start."""
    if path == home or path.startswith(f"{home}/"):
        return f"~{path.removeprefix(home)}"
    return path


@cache
def get_path_menu_item_class() -> "Callable[[QtGui.QAction, QtWidgets.QMenu], QtWidgets.QWidget]":
    """Return the class of the widget of a menu item of a path, which triggers its action at a click or at Return.

    A menu triggers no widget item, thus the widget triggers the action and closes the menu.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class PathMenuItem(QtWidgets.QWidget):
        """The widget of a menu item of a path."""

        def __init__(self, action: QtGui.QAction, menu: QtWidgets.QMenu) -> None:
            super().__init__()
            self.action = action
            self.menu = menu

        def choose(self) -> None:
            if self.isEnabled():
                self.menu.close()
                self.action.trigger()

        def set_highlighted(self, highlighted: bool) -> None:
            self.setProperty("highlighted", highlighted and self.isEnabled())
            # a new value of a property changes no style until the style polishes the widgets again
            for widget in (self, *self.findChildren(QtWidgets.QLabel)):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

        @t.override
        def enterEvent(self, event: QtGui.QEnterEvent, /) -> None:
            super().enterEvent(event)
            self.set_highlighted(True)

        @t.override
        def leaveEvent(self, event: QtCore.QEvent, /) -> None:
            super().leaveEvent(event)
            self.set_highlighted(self.hasFocus())

        @t.override
        def focusInEvent(self, event: QtGui.QFocusEvent, /) -> None:
            super().focusInEvent(event)
            self.set_highlighted(True)

        @t.override
        def focusOutEvent(self, event: QtGui.QFocusEvent, /) -> None:
            super().focusOutEvent(event)
            self.set_highlighted(self.underMouse())

        @t.override
        def mouseReleaseEvent(self, event: QtGui.QMouseEvent, /) -> None:
            super().mouseReleaseEvent(event)
            if self.rect().contains(event.position().toPoint()):
                self.choose()

        @t.override
        def keyPressEvent(self, event: QtGui.QKeyEvent, /) -> None:
            if event.key() in {QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter, QtCore.Qt.Key.Key_Space}:
                self.choose()
                return
            super().keyPressEvent(event)

    return PathMenuItem


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
    edit_properties: "Callable[[str], None]"
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
    and remove it. A double-click opens the line properties of the series, and the context menu of a row gives each
    action. A drag or Alt-Up and Alt-Down change the order. A folder or a file that the user drops on the window adds a
    series.

    Return the function that shows the rows. It takes a key of the parts of the values that the rows show, and it
    makes the rows again only for a new key, because each row reads the name of its series.
    """
    import shiboken6
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
    # a tool button with a menu arrow took the small font and the flat frame of the macOS style, thus two push buttons
    # give the dialog and the menu of the recent models
    addmodelbutton = QtWidgets.QPushButton("Add Model…")
    addmodelbutton.setToolTip("Add the folder of an ARTIS model")
    recentmodelsbutton = QtWidgets.QPushButton("Recent")
    recentmodelsbutton.setToolTip("Add one of the models that a window opened recently")
    recentmodelsmenu = QtWidgets.QMenu(recentmodelsbutton)
    recentmodelsbutton.setMenu(recentmodelsmenu)
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
    addrow.addWidget(recentmodelsbutton)
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
        row = get_hover_row_class()()
        row.setToolTip(seriesrow.tooltip)
        rowlayout = QtWidgets.QHBoxLayout(row)
        rowlayout.setContentsMargins(4, 0, 2, 0)
        rowlayout.setSpacing(2)
        rowlayout.addWidget(QtWidgets.QLabel(f"<b>{index + 1}</b>"))
        swatch = QtWidgets.QToolButton()
        swatch.setAutoRaise(True)
        swatch.setIconSize(QtCore.QSize(36, 14))
        swatch.setIcon(QtGui.QIcon(seriesrow.swatch))
        swatch.setToolTip(
            "The colour and the line style of the series in the plot. Click to change the line properties"
        )
        swatch.setAccessibleName(f"Set the line properties of {name}")
        # the new list replaces this row, thus each action of a button waits until the click ends
        swatch.clicked.connect(partial(QtCore.QTimer.singleShot, 0, window, partial(actions.edit_properties, path)))
        rowlayout.addWidget(swatch)
        namelabel = QtWidgets.QLabel(name)
        namelabel.setToolTip(
            f"The -label of the series: {name}. Double-click the row to change it in the line properties"
            if seriesrow.labelled
            else f"The name of the series in the legend: {name}. Double-click the row to give it a -label"
        )
        rowlayout.addWidget(namelabel)
        rowlayout.addSpacing(6)
        # the row gives the kind and the folder name, and the tooltip gives the full path
        pathlabel = make_elided_label(get_short_item_text(seriesrow.itemtext))
        pathlabel.setToolTip(seriesrow.itemtext)
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
            make_hover_widget(button)
        grip = QtWidgets.QLabel("≡")
        grip.setEnabled(False)
        grip.setToolTip("Drag the row to move the series")
        grip.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        rowlayout.addWidget(grip)
        make_hover_widget(grip)
        # the context menu gives each action, thus the keyboard and VoiceOver can also reach them
        row.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        row.customContextMenuRequested.connect(partial(on_row_menu, row))
        for text, enabled, action in (
            ("Move to Top", index > 0, partial(move_row, index, -index)),
            ("Move to Bottom", index < count - 1, partial(move_row, index, count - 1 - index)),
            *seriesrow.extraactions,
            ("Line Properties…", True, partial(actions.edit_properties, path)),
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

    def on_row_menu(row: QtWidgets.QWidget, position: QtCore.QPoint) -> None:
        """Show the menu of a row after the event of the row ends.

        A plot that ends while the menu is open can make the rows again, which deletes the row. A menu of the row ran
        in the event of the row, thus Qt deleted a row that was still in use and the process stopped. The menu of the
        window holds the actions of the row, and Qt removes the actions of a deleted row from the menu.
        """
        QtCore.QTimer.singleShot(0, window, partial(show_row_menu, list(row.actions()), row.mapToGlobal(position)))

    def show_row_menu(rowactions: "list[QtGui.QAction]", globalposition: QtCore.QPoint) -> None:
        menu = QtWidgets.QMenu(window)
        menu.addActions([action for action in rowactions if shiboken6.isValid(action)])
        menu.exec(globalposition)
        # the window is the parent of the menu, thus without this the window keeps each menu until it closes
        menu.deleteLater()

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
        # a test of a remote folder starts ssh, thus the menu keeps each remote model and leaves out a missing local
        # folder
        folders = [
            folder
            for folder in get_recent_models()
            if (is_remote_path(folder) or Path(folder).is_dir()) and actions.get_full_path(folder) not in fullpaths
        ]
        for folder in folders:
            action = make_path_menu_action(recentmodelsmenu, folder, serieslist.width())
            action.triggered.connect(partial(add_paths, [folder]))
            recentmodelsmenu.addAction(action)
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
        actions.edit_properties(item.data(QtCore.Qt.ItemDataRole.UserRole))

    serieslist.itemDoubleClicked.connect(on_double_click)

    def on_selection() -> None:
        for index in range(serieslist.count()):
            item = serieslist.item(index)
            if (row := serieslist.itemWidget(item)) is not None:
                set_row_selected(row, selected=item.isSelected())

    serieslist.itemSelectionChanged.connect(on_selection)
    # the list changes its rows at the end of the drop, thus the new order applies after the drop
    for rowsignal in (serieslist.model().rowsMoved, serieslist.model().rowsInserted, serieslist.model().rowsRemoved):
        rowsignal.connect(lambda: QtCore.QTimer.singleShot(0, window, on_rows_dropped))
    openreferencebutton.clicked.connect(on_open_reference)
    referencecompleter.activated.connect(on_complete_reference)
    referenceedit.returnPressed.connect(lambda: add_reference_name(referenceedit.text()))
    set_drop_handler(window, on_drop)
    return show_rows
