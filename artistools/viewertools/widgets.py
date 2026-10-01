"""Make the widgets of a viewer window, e.g. the sections, the option table, the sliders, and the status bar."""

import argparse
import html
import json
import math
import re
import shlex
import sys
import typing as t
from functools import cache
from pathlib import Path

import numpy as np

from artistools.misc import separate_trailing_folders
from artistools.viewertools.application import get_bool_setting
from artistools.viewertools.application import get_float_setting
from artistools.viewertools.application import get_settings
from artistools.viewertools.core import DEFAULT_PLAY_FPS
from artistools.viewertools.core import get_actions_by_flag
from artistools.viewertools.core import get_default_tokens
from artistools.viewertools.core import get_helptexts
from artistools.viewertools.core import get_option_kind
from artistools.viewertools.core import get_table_actions
from artistools.viewertools.core import OptionRows

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.figure as mplfig
    import numpy.typing as npt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets


# a section with one of these titles starts closed, because a user needs it less often than the plot controls
CLOSED_SECTIONS: t.Final = frozenset({"Other options", "Command", "Python"})


def add_section(
    panellayout: "QtWidgets.QVBoxLayout", title: str, key: str | None = None
) -> "tuple[QtWidgets.QToolButton, QtWidgets.QGridLayout]":
    """Add a section with a heading and a grid for its controls to the panel of the window.

    A click on the heading closes or opens the section, as a disclosure triangle does in the inspector of Keynote. The
    settings keep the state of each section by its key, which is the title if the caller gives no key.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    header = QtWidgets.QToolButton()
    header.setText(title)
    header.setCheckable(True)
    header.setAutoRaise(True)
    header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    # a faint band across the panel sets each heading apart from the controls. The grey with an alpha suits the light
    # and the dark appearance
    header.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
    header.setStyleSheet(SECTION_HEADER_STYLE)
    # the macOS style gives a tool button a small font, and a heading takes the bold font of the application
    font = QtWidgets.QApplication.font()
    font.setBold(True)
    header.setFont(font)
    content = QtWidgets.QWidget()
    content.setObjectName(SECTION_CONTENT_NAME)
    grid = QtWidgets.QGridLayout(content)
    # the space under the content of a section is larger than the space between its rows, thus each section stays a
    # group of its own
    grid.setContentsMargins(12, 6, 4, 14)
    grid.setVerticalSpacing(ROW_SPACING)
    # the columns of the grid have the gap of a row of make_row_layout, thus a control in the grid and a control in a
    # row start at one place
    grid.setHorizontalSpacing(ROW_SPACING)
    grid.setColumnStretch(1, 1)
    settingkey = f"{QtWidgets.QApplication.applicationDisplayName()}/sections/{key or title}"
    isopen = get_bool_setting(settingkey, default=title not in CLOSED_SECTIONS)

    def set_open(checked: bool) -> None:
        header.setArrowType(QtCore.Qt.ArrowType.DownArrow if checked else QtCore.Qt.ArrowType.RightArrow)
        header.setToolTip(f"{'Close' if checked else 'Open'} the section")
        content.setVisible(checked)
        get_settings().setValue(settingkey, checked)

    # the stretch at the end of the panel keeps each section, also Command and Python, below the last one
    index = panellayout.count() - 1
    panellayout.insertWidget(index, header)
    # a widget with no parent shows as a window of its own, thus the content goes into the panel before set_open
    panellayout.insertWidget(index + 1, content)
    header.toggled.connect(set_open)
    header.setChecked(isopen)
    set_open(isopen)
    return header, grid


# a button that shows one symbol, e.g. ✕ or ▲, has no frame, and a light background shows under the pointer, as the
# small buttons of the apps of macOS have. The grey with an alpha suits the light and the dark appearance
GLYPH_BUTTON_STYLE: t.Final = (
    "QToolButton { border: none; background: transparent; padding: 1px 4px; border-radius: 4px; }"
    " QToolButton:hover { background: rgba(128, 128, 128, 60); }"
    " QToolButton:pressed { background: rgba(128, 128, 128, 110); }"
)

# the space between the rows of a section, and the space in front of a group, e.g. a label, that follows a control
ROW_SPACING: t.Final = 6
LABEL_GAP: t.Final = 12
SECTION_CONTENT_NAME: t.Final = "sectioncontent"
SECTION_HEADER_STYLE: t.Final = (
    "QToolButton { border: none; border-radius: 5px; padding: 3px 6px; background: rgba(128, 128, 128, 34); }"
    " QToolButton:hover { background: rgba(128, 128, 128, 60); }"
)


def make_glyph_button(glyph: str, tooltip: str, accessiblename: str) -> "QtWidgets.QToolButton":
    """Return a small button that shows one symbol, e.g. ✕ to remove an item, with no frame."""
    from PySide6 import QtWidgets

    button = QtWidgets.QToolButton()
    button.setText(glyph)
    button.setStyleSheet(GLYPH_BUTTON_STYLE)
    button.setToolTip(tooltip)
    button.setAccessibleName(accessiblename)
    return button


def make_row_layout(widgets: "Sequence[QtWidgets.QWidget]") -> "QtWidgets.QVBoxLayout":
    """Return a layout that puts the widgets side by side from the left, and wraps the row in a narrow sidebar.

    A label, a checkbox, or a push button that follows a control starts a new group, e.g. a pair of a label and a
    control. A wider space goes in front of it, thus the groups stay apart. A group stays on one line, and the next
    group goes to a new line if the line is full.
    """
    from PySide6 import QtWidgets

    groupstarts = QtWidgets.QLabel | QtWidgets.QCheckBox | QtWidgets.QPushButton
    groups: list[list[QtWidgets.QWidget]] = []
    for index, widget in enumerate(widgets):
        if index == 0 or (isinstance(widget, groupstarts) and not isinstance(widgets[index - 1], QtWidgets.QLabel)):
            groups.append([])
        groups[-1].append(widget)
    groupwidgets: list[QtWidgets.QWidget] = []
    for group in groups:
        if len(group) == 1:
            groupwidgets.append(group[0])
            continue
        groupbox = QtWidgets.QWidget()
        grouplayout = QtWidgets.QHBoxLayout(groupbox)
        grouplayout.setContentsMargins(0, 0, 0, 0)
        grouplayout.setSpacing(ROW_SPACING)
        for widget in group:
            grouplayout.addWidget(widget)
        groupwidgets.append(groupbox)
    rowlayout = QtWidgets.QVBoxLayout()
    rowlayout.setContentsMargins(0, 0, 0, 0)
    rowlayout.addWidget(get_wrap_row_class()(groupwidgets))
    return rowlayout


def get_wrapped_lines(widths: "Sequence[int]", available: int, gap: int) -> list[list[int]]:
    """Return the indices of the items on each line, for items of these widths on lines of the available width.

    An item goes on a new line if the line is full. An item of no width, e.g. a hidden widget, takes no place and no
    gap. A line holds one item at least, also when the item is wider than the line.
    """
    lines: list[list[int]] = [[]]
    used = 0
    for index, width in enumerate(widths):
        if width > 0 and used > 0 and used + gap + width > available:
            lines.append([])
            used = 0
        lines[-1].append(index)
        if width > 0:
            used += width if used == 0 else gap + width
    return lines


@cache
def get_wrap_row_class() -> "Callable[[list[QtWidgets.QWidget]], QtWidgets.QWidget]":
    """Return the class of the rows of make_row_layout, which takes the widget of each group.

    A layout class of Python ran each time that Qt arranged the sidebar, e.g. at each step of a drag. Each call
    waited for the worker thread, which held the GIL during a plot, thus a text change took 156 ms and not 7 ms. The
    row puts its groups in layouts of Qt, and Python runs only when the width of the row changes.
    """
    import shiboken6
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class WrapRow(QtWidgets.QWidget):
        """The groups of a row, on as many lines as the width of the row needs."""

        def __init__(self, groups: "list[QtWidgets.QWidget]") -> None:
            super().__init__()
            self.groups = groups
            self.lines: list[list[int]] = []
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(ROW_SPACING)
            # the width of the widest group is the minimum width of the row. A wider line then wraps and does not
            # widen the sidebar. A layout of Qt uses the minimum width of a widget if it has one, or its size hint
            self.setMinimumWidth(
                max((group.minimumWidth() or group.minimumSizeHint().width() for group in groups), default=1)
            )
            self.set_lines([list(range(len(groups)))])

        def set_lines(self, lines: list[list[int]]) -> None:
            layout = self.layout()
            assert layout is not None
            while (item := layout.takeAt(0)) is not None:
                if (line := item.layout()) is not None:
                    while line.takeAt(0) is not None:
                        pass
                    shiboken6.delete(line)
            for indices in lines:
                line = QtWidgets.QHBoxLayout()
                line.setSpacing(LABEL_GAP)
                for index in indices:
                    line.addWidget(self.groups[index])
                line.addStretch(1)
                assert isinstance(layout, QtWidgets.QBoxLayout)
                layout.addLayout(line)
            self.lines = lines

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent, /) -> None:
            super().resizeEvent(event)
            widths = [0 if group.isHidden() else group.sizeHint().width() for group in self.groups]
            lines = get_wrapped_lines(widths, event.size().width(), LABEL_GAP)
            if lines != self.lines:
                self.set_lines(lines)

    return WrapRow


def add_row(grid: "QtWidgets.QGridLayout", row: int, widgets: "Sequence[QtWidgets.QWidget]") -> None:
    """Put the widgets side by side in one row of the grid, from the left."""
    grid.addLayout(make_row_layout(widgets), row, 0, 1, -1)


def make_flow_layout() -> "QtWidgets.QLayout":
    """Return a layout that puts its widgets side by side from the left, and starts a new row when a row is full.

    Qt has no such layout. A row of chips, e.g. the series of a subplot, then wraps to the width of the sidebar.
    """
    return get_flow_layout_class()()


@cache
def get_flow_layout_class() -> "type[QtWidgets.QLayout]":
    """Return the class of the layouts of make_flow_layout.

    PySide keeps about 1.5 KB of memory for each class, thus the viewers make the class one time and not for each
    layout.
    """
    import shiboken6
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    def delete_items(items: "list[QtWidgets.QLayoutItem]") -> None:
        # a layout of Qt deletes its items when Qt deletes the layout, and a layout of Python must do the same
        for item in items:
            shiboken6.delete(item)
        items.clear()

    class FlowLayout(QtWidgets.QLayout):
        """A layout that fills rows from the left, and gives each widget the size that it asks for."""

        def __init__(self) -> None:
            super().__init__()
            self.layoutitems: list[QtWidgets.QLayoutItem] = []
            self.setSpacing(4)
            self.setContentsMargins(0, 0, 0, 0)
            # the signal comes after Python lost the layout, thus the function holds the list and not the layout
            items = self.layoutitems
            self.destroyed.connect(lambda: delete_items(items))

        @t.override
        def addItem(self, arg__1: QtWidgets.QLayoutItem, /) -> None:
            self.layoutitems.append(arg__1)

        @t.override
        def count(self, /) -> int:
            return len(self.layoutitems)

        @t.override
        def itemAt(self, index: int, /) -> QtWidgets.QLayoutItem | None:
            return self.layoutitems[index] if 0 <= index < len(self.layoutitems) else None

        @t.override
        def takeAt(self, index: int, /) -> QtWidgets.QLayoutItem | None:
            return self.layoutitems.pop(index) if 0 <= index < len(self.layoutitems) else None

        @t.override
        def expandingDirections(self, /) -> QtCore.Qt.Orientation:
            return QtCore.Qt.Orientation(0)

        @t.override
        def hasHeightForWidth(self, /) -> bool:
            return True

        @t.override
        def heightForWidth(self, arg__1: int, /) -> int:
            return self.arrange(QtCore.QRect(0, 0, arg__1, 0), move=False)

        @t.override
        def setGeometry(self, arg__1: QtCore.QRect, /) -> None:
            super().setGeometry(arg__1)
            self.arrange(arg__1, move=True)

        @t.override
        def sizeHint(self, /) -> QtCore.QSize:
            return self.minimumSize()

        @t.override
        def minimumSize(self, /) -> QtCore.QSize:
            size = QtCore.QSize()
            for item in self.layoutitems:
                size = size.expandedTo(item.minimumSize())
            return size

        def arrange(self, rect: QtCore.QRect, *, move: bool) -> int:
            """Put each item in its place inside rect if move is True, and return the height that the rows take.

            A hidden widget takes no place. The items of a row share a vertical centre, thus a label stays level
            with the text of the control beside it.
            """
            gap = self.spacing()
            rows: list[list[tuple[QtWidgets.QLayoutItem, QtCore.QSize, int]]] = [[]]
            x = rect.x()
            for item in self.layoutitems:
                hint = item.sizeHint()
                if item.isEmpty() or hint.isEmpty():
                    continue
                # an item wider than the row shrinks to the row, but not below its minimum width
                hint.setWidth(max(min(hint.width(), rect.width()), item.minimumSize().width()))
                if x + hint.width() > rect.right() + 1 and rows[-1]:
                    rows.append([])
                    x = rect.x()
                rows[-1].append((item, hint, x))
                x += hint.width() + gap
            y = rect.y()
            for row in rows:
                rowheight = max((hint.height() for _, hint, _ in row), default=0)
                if move:
                    for item, hint, itemx in row:
                        item.setGeometry(QtCore.QRect(QtCore.QPoint(itemx, y + (rowheight - hint.height()) // 2), hint))
                y += rowheight + self.spacing()
            return max(y - self.spacing() - rect.y(), 0)

    return FlowLayout


def make_elided_label(text: str) -> "QtWidgets.QLabel":
    """Return a label that shows the start and the end of its text, e.g. of a long path, in the width that it gets."""
    label = get_elided_label_class()()
    label.setProperty("fulltext", text)
    label.setToolTip(text)
    return label


@cache
def get_elided_label_class() -> "type[QtWidgets.QLabel]":
    """Return the class of the labels of make_elided_label.

    A QLabel shows its whole text or cuts its end. The class shortens the middle of the text at each change of its
    width, thus Python runs only at a resize.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class ElidedLabel(QtWidgets.QLabel):
        """A label with a text that ends in the middle with "…" if the label is too narrow for it."""

        def __init__(self) -> None:
            super().__init__()
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent, /) -> None:
            super().resizeEvent(event)
            fulltext = str(self.property("fulltext") or "")
            self.setText(self.fontMetrics().elidedText(fulltext, QtCore.Qt.TextElideMode.ElideMiddle, self.width()))

    return ElidedLabel


def make_drag_header(
    on_drag: "Callable[[QtCore.QPoint], None]",
    on_drop: "Callable[[QtCore.QPoint], None]",
    on_move: "Callable[[int], None]",
) -> "QtWidgets.QWidget":
    """Return a header that the user can drag, e.g. to move a card to a new place in a list.

    During a drag, on_drag receives each position of the pointer on the screen. on_drop receives the last position.
    A child control, e.g. a button, keeps its clicks, thus a drag starts only on the background or on a label. The
    header also takes the keyboard focus, and Alt-Up (Option-Up on macOS) or Alt-Down gives -1 or 1 to on_move. A
    user of the keyboard can then move the card too. The object name "dragheader" selects the header in a style sheet.
    """
    from PySide6 import QtCore

    header = get_drag_header_class()()
    header.setObjectName("dragheader")
    header.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
    # a style sheet can draw the border of the focus only on a widget with a styled background
    header.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
    header.setProperty("on_drag", on_drag)
    header.setProperty("on_drop", on_drop)
    header.setProperty("on_move", on_move)
    return header


def make_reorder_list(on_move: "Callable[[int], None]") -> "QtWidgets.QListWidget":
    """Return a list whose rows the user can move with a drag or with the keyboard, as the subplot cards do.

    Alt-Up (Option-Up on macOS) or Alt-Down gives -1 or 1 to on_move for the selected row. A drag shows the drop
    position as a line of 2 pixels in the highlight colour, which is the line between two subplot cards.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    listwidget = get_reorder_list_class()()
    # a drag of a row moves it, and a drag needs the selection of the row
    listwidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
    listwidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
    listwidget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
    listwidget.setProperty("on_move", on_move)
    return listwidget


@cache
def get_reorder_list_class() -> "type[QtWidgets.QListWidget]":
    """Return the class of the lists of make_reorder_list."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class DropLineStyle(QtWidgets.QProxyStyle):
        """A style that draws the drop position of a list as the drop line of the subplot cards."""

        @t.override
        def drawPrimitive(
            self,
            element: QtWidgets.QStyle.PrimitiveElement,
            option: QtWidgets.QStyleOption,
            painter: QtGui.QPainter,
            widget: QtWidgets.QWidget | None = None,
        ) -> None:
            if element != QtWidgets.QStyle.PrimitiveElement.PE_IndicatorItemViewItemDrop:
                super().drawPrimitive(element, option, painter, widget)
                return
            # the list gives a rectangle with no height at the top or the bottom of a row. The stubs of PySide give
            # the fields of an option with no type
            rect = t.cast("QtCore.QRect", option.rect)
            palette = t.cast("QtGui.QPalette", option.palette)
            painter.fillRect(QtCore.QRect(rect.left(), rect.top() - 1, rect.width(), 2), palette.highlight())

    class ReorderList(QtWidgets.QListWidget):
        """A list that gives Alt-Up and Alt-Down to its on_move callback."""

        def __init__(self) -> None:
            super().__init__()
            # a proxy of the style of the application, with the list as its parent, thus Qt deletes both together
            style = DropLineStyle()
            style.setParent(self)
            self.setStyle(style)

        @t.override
        def keyPressEvent(self, event: QtGui.QKeyEvent, /) -> None:
            steps = {QtCore.Qt.Key.Key_Up: -1, QtCore.Qt.Key.Key_Down: 1}
            holdsalt = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)
            if holdsalt and event.key() in steps and callable(callback := self.property("on_move")):
                callback(steps[QtCore.Qt.Key(event.key())])
                event.accept()
            else:
                super().keyPressEvent(event)

    return ReorderList


@cache
def get_drag_header_class() -> "type[QtWidgets.QWidget]":
    """Return the class of the headers of make_drag_header.

    PySide keeps about 1.5 KB of memory for each class, thus the viewers make the class one time and not for each
    header.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class DragHeader(QtWidgets.QWidget):
        """A widget that sends the positions of a drag with the left mouse button to its two callbacks."""

        def __init__(self) -> None:
            super().__init__()
            self.pressposition: QtCore.QPoint | None = None
            self.dragging = False

        def send(self, name: str, position: QtCore.QPoint) -> None:
            if callable(callback := self.property(name)):
                callback(position)

        @t.override
        def mousePressEvent(self, event: QtGui.QMouseEvent, /) -> None:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self.pressposition = event.globalPosition().toPoint()
                event.accept()
            else:
                super().mousePressEvent(event)

        @t.override
        def mouseMoveEvent(self, event: QtGui.QMouseEvent, /) -> None:
            if self.pressposition is None:
                super().mouseMoveEvent(event)
                return
            position = event.globalPosition().toPoint()
            distance = (position - self.pressposition).manhattanLength()
            if not self.dragging and distance >= QtWidgets.QApplication.startDragDistance():
                self.dragging = True
                self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            if self.dragging:
                self.send("on_drag", position)

        @t.override
        def keyPressEvent(self, event: QtGui.QKeyEvent, /) -> None:
            steps = {QtCore.Qt.Key.Key_Up: -1, QtCore.Qt.Key.Key_Down: 1}
            holdsalt = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)
            if holdsalt and event.key() in steps and callable(callback := self.property("on_move")):
                callback(steps[QtCore.Qt.Key(event.key())])
                event.accept()
            else:
                super().keyPressEvent(event)

        @t.override
        def mouseReleaseEvent(self, event: QtGui.QMouseEvent, /) -> None:
            wasdragging = self.dragging
            self.pressposition, self.dragging = None, False
            self.unsetCursor()
            # on_drop can make the card again, thus the header resets its state first
            if wasdragging:
                self.send("on_drop", event.globalPosition().toPoint())
            else:
                super().mouseReleaseEvent(event)

    return DragHeader


def make_slider() -> "QtWidgets.QSlider":
    """Return a horizontal slider that does not take the keyboard focus."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    # the arrow keys move the time and change the width, thus a slider must not take them
    slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
    return slider


def make_range_slider(
    steps: int,
) -> tuple[
    "QtWidgets.QWidget",
    "Callable[[int, int], None]",
    "Callable[[Callable[[int, int], None]], None]",
    "Callable[[int], None]",
]:
    """Return a slider with two handles for a range of the positions 0 to steps, and three functions of the slider.

    The first function moves the handles. The second function connects a handler, which receives the index of the
    handle that the user moved and its new position. The third function gives the slider a new number of steps.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class RangeSlider(QtWidgets.QWidget):
        """A slider with two handles, which give the minimum and the maximum of a range.

        Qt has no slider with two handles. A drag moves the handle that is nearer to the pointer, and the minimum
        stays below the maximum. The signal gives the index of the handle that moved and its new position.
        """

        limitmoved = QtCore.Signal(int, int)
        handleradius: t.Final = 8.0

        def __init__(self) -> None:
            super().__init__()
            self.steps = steps
            self.positions = [0, steps]
            self.draghandle: int | None = None
            self.setMinimumHeight(round(3 * self.handleradius))
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            # the arrow keys move the time and change the width, thus the slider must not take them
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        def set_positions(self, low: int, high: int) -> None:
            self.positions = [low, high]
            self.update()

        def set_steps(self, steps: int) -> None:
            self.steps = steps
            self.positions = [min(position, steps) for position in self.positions]
            self.update()

        def get_pixel(self, position: int) -> float:
            return self.handleradius + (self.width() - 2.0 * self.handleradius) * position / self.steps

        def get_position(self, pixel: float) -> int:
            fraction = (pixel - self.handleradius) / max(self.width() - 2.0 * self.handleradius, 1.0)
            return round(min(max(fraction, 0.0), 1.0) * self.steps)

        @t.override
        def paintEvent(self, event: QtGui.QPaintEvent) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            palette = self.palette()
            middle = self.height() / 2.0
            lowpixel, highpixel = (self.get_pixel(position) for position in self.positions)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for left, right, colourrole in (
                (self.handleradius, self.width() - self.handleradius, QtGui.QPalette.ColorRole.Mid),
                (lowpixel, highpixel, QtGui.QPalette.ColorRole.Highlight),
            ):
                painter.setBrush(palette.color(colourrole))
                painter.drawRoundedRect(QtCore.QRectF(left, middle - 2.0, right - left, 4.0), 2.0, 2.0)
            painter.setPen(QtGui.QPen(palette.color(QtGui.QPalette.ColorRole.Mid)))
            painter.setBrush(palette.color(QtGui.QPalette.ColorRole.Light))
            for pixel in (lowpixel, highpixel):
                painter.drawEllipse(QtCore.QPointF(pixel, middle), self.handleradius - 1.0, self.handleradius - 1.0)
            painter.end()

        @t.override
        def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
            pixel = event.position().x()
            midpixel = sum(self.get_pixel(position) for position in self.positions) / 2.0
            self.draghandle = 0 if pixel < midpixel else 1
            self.move_handle(pixel)

        @t.override
        def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
            if self.draghandle is not None:
                self.move_handle(event.position().x())

        @t.override
        def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
            self.draghandle = None

        def move_handle(self, pixel: float) -> None:
            if self.draghandle is None:
                return
            position = self.get_position(pixel)
            if self.draghandle == 0:
                position = min(position, self.positions[1] - 1)
            else:
                position = max(position, self.positions[0] + 1)
            if position != self.positions[self.draghandle]:
                self.positions[self.draghandle] = position
                self.update()
                self.limitmoved.emit(self.draghandle, position)

    slider = RangeSlider()

    def connect_handler(handler: "Callable[[int, int], None]") -> None:
        slider.limitmoved.connect(handler)

    return slider, slider.set_positions, connect_handler, slider.set_steps


def make_sidebar() -> "tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]":
    """Return the sidebar of the window and the layout of its panel. The sections and the command scroll together."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    sidebar = QtWidgets.QWidget()
    # the handle of the splitter sets the width, and a narrower sidebar cuts the controls
    sidebar.setMinimumWidth(360)
    sidebarlayout = QtWidgets.QVBoxLayout(sidebar)
    sidebarlayout.setContentsMargins(0, 0, 0, 0)
    panel = QtWidgets.QWidget()
    panellayout = QtWidgets.QVBoxLayout(panel)
    panellayout.setSpacing(2)
    # add_section puts each section before this stretch, thus the empty space stays at the end
    panellayout.addStretch(1)
    panelscroll = QtWidgets.QScrollArea()
    panelscroll.setWidget(panel)
    panelscroll.setWidgetResizable(True)
    panelscroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    panelscroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sidebarlayout.addWidget(panelscroll, stretch=1)
    return sidebar, panellayout


def make_plot_area(canvas: "FigureCanvasQTAgg", on_resize: "Callable[[], None]") -> "QtWidgets.QWidget":
    """Return the area of the plot, which holds the canvas at its centre and calls on_resize at each resize."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # the instance holds on_resize, and the class holds no reference to it. PySide keeps each class, thus a class that
    # captured on_resize in a closure kept the viewer, the figure, and the canvas of each closed window
    class PlotArea(QtWidgets.QWidget):
        """The area of the plot, which scales the figure to its size."""

        def __init__(self, on_resize: "Callable[[], None]") -> None:
            super().__init__()
            self.on_resize = on_resize

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            place_plot_overlays(self)
            self.on_resize()

    plotarea = PlotArea(on_resize)
    plotarea.setObjectName(PLOT_AREA_NAME)
    # the area takes the background colour of the figure, which needs a styled background on a plain widget
    plotarea.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
    plotarea.setMinimumSize(320, 240)
    plotlayout = QtWidgets.QVBoxLayout(plotarea)
    plotlayout.setContentsMargins(0, 0, 0, 0)
    plotlayout.addWidget(canvas, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
    # the banner shows the message of a rejected plot over the plot in the red of the status bar, and the spinner
    # shows a slow plot
    banner = QtWidgets.QLabel(plotarea)
    banner.setObjectName("plotbanner")
    banner.setWordWrap(True)
    banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    banner.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    banner.hide()
    hidetimer = QtCore.QTimer(banner)
    hidetimer.setSingleShot(True)
    hidetimer.setInterval(BANNER_MILLISECONDS)
    hidetimer.timeout.connect(banner.hide)
    spinner = get_spinner_class()(plotarea)
    spinner.setObjectName("plotspinner")
    spinner.hide()
    emptynote = QtWidgets.QLabel(EMPTY_PLOT_TEXT, plotarea)
    emptynote.setObjectName("plotempty")
    emptynote.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    emptynote.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
    font = emptynote.font()
    font.setPointSizeF(font.pointSizeF() * 1.3)
    emptynote.setFont(font)
    emptynote.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    emptynote.hide()
    return plotarea


PLOT_AREA_NAME: t.Final = "plotarea"
EMPTY_PLOT_TEXT: t.Final = "The plot shows no data in the range of its axes.\nDouble-click the plot to reset the range."


def show_plot_area_state(window: "QtCore.QObject", fig: "mplfig.Figure") -> None:
    """Give the plot area the background colour of the figure, and show a note over a plot that shows no data.

    A figure narrower than the area left a band of the window colour at each side, which looked like a frame.
    """
    import matplotlib.colors as mcolors
    from PySide6 import QtWidgets

    plotarea = window.findChild(QtWidgets.QWidget, PLOT_AREA_NAME)
    if plotarea is None:
        return
    plotarea.setStyleSheet(f"#{PLOT_AREA_NAME} {{ background: {mcolors.to_hex(fig.get_facecolor())}; }}")
    if (emptynote := plotarea.findChild(QtWidgets.QLabel, "plotempty")) is not None:
        emptynote.setVisible(not figure_shows_data(fig))
        place_plot_overlays(plotarea)


def figure_shows_data(fig: "mplfig.Figure") -> bool:
    """Return True if an axes of the figure shows a part of a line inside its limits, an image, or a collection.

    A line can cross the frame between two of its points, e.g. a time range between two timesteps. Thus each segment
    between two points counts if it crosses the frame.
    """
    for axis in fig.axes:
        if axis.images or axis.collections or axis.patches:
            return True
        xlow, xhigh = sorted(axis.get_xlim())
        ylow, yhigh = sorted(axis.get_ylim())
        for line in axis.get_lines():
            points = np.asarray(line.get_xydata(), dtype=float)
            if points.size == 0:
                continue
            x, y = points[:, 0], points[:, 1]
            if np.any((x >= xlow) & (x <= xhigh) & (y >= ylow) & (y <= yhigh)):
                return True
            if np.any(get_segments_in_box(x, y, (xlow, xhigh, ylow, yhigh))):
                return True
    return False


def get_segments_in_box(
    x: "npt.NDArray[np.floating]", y: "npt.NDArray[np.floating]", box: tuple[float, float, float, float]
) -> "npt.NDArray[np.bool_]":
    """Return for each segment between two neighbouring points whether a part of it is inside the box.

    box gives xlow, xhigh, ylow, and yhigh. The test clips each segment to the box (the Liang-Barsky method). A NaN
    point breaks a line, thus a segment with a NaN point is not inside.
    """
    xlow, xhigh, ylow, yhigh = box
    xstart, ystart = x[:-1], y[:-1]
    dx, dy = x[1:] - xstart, y[1:] - ystart
    isinside = np.isfinite(xstart) & np.isfinite(ystart) & np.isfinite(dx) & np.isfinite(dy)
    tlow, thigh = np.zeros(len(dx)), np.ones(len(dx))
    for direction, distance in ((-dx, xstart - xlow), (dx, xhigh - xstart), (-dy, ystart - ylow), (dy, yhigh - ystart)):
        isparallel = direction == 0
        isinside &= ~(isparallel & (distance < 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = distance / direction
        tlow = np.where(~isparallel & (direction < 0), np.maximum(tlow, ratio), tlow)
        thigh = np.where(~isparallel & (direction > 0), np.minimum(thigh, ratio), thigh)
    return np.logical_and(isinside, tlow <= thigh)


# the time that the banner of a rejected plot stays over the plot. The status bar keeps the message
BANNER_MILLISECONDS: t.Final = 5000


def place_plot_overlays(plotarea: "QtWidgets.QWidget") -> None:
    """Put the banner at the centre of the plot area, a third of the way down, and the spinner at its top right corner.

    The title of the figure is at the top, thus the banner goes below it.
    """
    from PySide6 import QtWidgets

    margin = 12
    if (banner := plotarea.findChild(QtWidgets.QLabel, "plotbanner")) is not None:
        # a label that wraps its text takes a narrow width from adjustSize, thus the width comes from the text
        padding = 2 * 14 + 2
        textwidth = banner.fontMetrics().horizontalAdvance(banner.text()) + padding
        width = max(min(textwidth, plotarea.width() - 4 * margin), 100)
        banner.resize(width, banner.heightForWidth(width))
        banner.move((plotarea.width() - width) // 2, (plotarea.height() - banner.height()) // 3)
        banner.raise_()
    if (spinner := plotarea.findChild(QtWidgets.QWidget, "plotspinner")) is not None:
        spinner.move(plotarea.width() - spinner.width() - margin, margin)
        spinner.raise_()
    if (emptynote := plotarea.findChild(QtWidgets.QLabel, "plotempty")) is not None:
        emptynote.resize(plotarea.width(), emptynote.sizeHint().height())
        emptynote.move(0, (plotarea.height() - emptynote.height()) // 2)
        emptynote.raise_()


def show_plot_banner(window: "QtCore.QObject", message: str | None) -> None:
    """Show the message of a rejected plot over the plot for a few seconds, or hide the banner if message is None."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    banner = window.findChild(QtWidgets.QLabel, "plotbanner")
    if banner is None:
        return
    if message is None:
        banner.hide()
        return
    banner.setText(message)
    # the banner takes the red of the status bar, which changes with the appearance
    banner.setStyleSheet(
        f"QLabel#plotbanner {{ background: {get_message_colours()[0]}; color: palette(base); border-radius: 8px;"
        " padding: 8px 14px; }"
    )
    banner.show()
    if (plotarea := banner.parentWidget()) is not None:
        place_plot_overlays(plotarea)
    if (hidetimer := banner.findChild(QtCore.QTimer)) is not None:
        hidetimer.start()


def set_plot_busy(window: "QtCore.QObject", *, busy: bool) -> None:
    """Show or hide the spinner over the plot."""
    from PySide6 import QtWidgets

    spinner = window.findChild(QtWidgets.QWidget, "plotspinner")
    if spinner is not None:
        spinner.setVisible(busy)
        if busy and (plotarea := spinner.parentWidget()) is not None:
            place_plot_overlays(plotarea)


@cache
def get_spinner_class() -> "type[QtWidgets.QWidget]":
    """Return the class of the spinner over the plot, which turns while it shows.

    PySide keeps about 1.5 KB of memory for each class, thus the viewers make the class one time and not for each
    window.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class Spinner(QtWidgets.QWidget):
        """A circle of 12 spokes that turns, as the progress indicator of macOS."""

        spokes: t.Final = 12

        def __init__(self, parent: QtWidgets.QWidget) -> None:
            super().__init__(parent)
            self.setFixedSize(32, 32)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.step = 0
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(80)
            self.timer.timeout.connect(self.advance)

        def advance(self) -> None:
            self.step = (self.step + 1) % self.spokes
            self.update()

        @t.override
        def showEvent(self, event: QtGui.QShowEvent, /) -> None:
            super().showEvent(event)
            self.timer.start()

        @t.override
        def hideEvent(self, event: QtGui.QHideEvent, /) -> None:
            super().hideEvent(event)
            self.timer.stop()

        @t.override
        def paintEvent(self, event: QtGui.QPaintEvent, /) -> None:
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            background = self.palette().color(QtGui.QPalette.ColorRole.Window)
            background.setAlpha(210)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(self.rect(), 8, 8)
            colour = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            painter.translate(self.width() / 2, self.height() / 2)
            for spoke in range(self.spokes):
                # the newest spoke is the darkest, and the others fade behind it
                colour.setAlphaF(0.15 + 0.85 * ((spoke - self.step) % self.spokes) / (self.spokes - 1))
                pen = QtGui.QPen(colour, 2.2)
                pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QtCore.QPointF(0, 5), QtCore.QPointF(0, 10))
                painter.rotate(360 / self.spokes)
            painter.end()

    return Spinner


def fit_canvas(canvas: "FigureCanvasQTAgg", figsize: tuple[float, float], plotarea: "QtWidgets.QWidget") -> None:
    """Scale the figure to the plot area, and keep the shape of the frames.

    An embedded canvas has no figure manager, thus the figure cannot set the size of the widget. The resolution
    of the figure changes, thus the frames keep their size in inches and the whole figure fits in the area.
    """
    from PySide6 import QtCore

    fig = canvas.figure
    figwidth, figheight = figsize
    if figwidth <= 0.0 or figheight <= 0.0:
        return
    area = plotarea.contentsRect()
    logicaldpi = max(min(area.width() / figwidth, area.height() / figheight), 20.0)
    size = QtCore.QSize(math.ceil(figwidth * logicaldpi), math.ceil(figheight * logicaldpi))
    dpi = logicaldpi * canvas.device_pixel_ratio
    # matplotlib changes the size of the figure in inches when the pixel ratio of the screen changes
    sizeinches = tuple(fig.get_size_inches())
    if canvas.size() == size and math.isclose(fig.dpi, dpi, rel_tol=1e-6) and sizeinches == figsize:
        return
    fig.set_dpi(dpi)
    fig.set_size_inches(figwidth, figheight, forward=False)
    canvas.setFixedSize(size)
    canvas.draw_idle()


def make_option_table(
    window: "QtWidgets.QWidget",
    parser: argparse.ArgumentParser,
    hiddendests: "Collection[str]",
    rows: OptionRows,
    on_rows: "Callable[[OptionRows], None]",
) -> "tuple[QtWidgets.QTableWidget, Callable[[OptionRows], None]]":
    """Return a table of the options that no other control of the window sets, and a function that shows new rows.

    The table has a list of the options that the user can search, and a control that matches the type of each
    option. on_rows receives the rows that have all their values after each change. The function that shows new
    rows keeps a row that waits for a value, unless the complete rows changed.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    actionsbyflag = get_actions_by_flag(parser)
    helptexts = get_helptexts(parser)
    tableflags = [action.option_strings[0] for action in get_table_actions(parser, hiddendests)]
    optiontable = QtWidgets.QTableWidget(0, 2)
    optiontable.setHorizontalHeaderLabels(["Option", "Value"])
    optiontable.verticalHeader().setVisible(False)
    optiontable.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    optiontable.setToolTip("Give each option of the command that no other control sets. Type part of a name to search.")
    widestflag = max(tableflags, key=len, default="")
    optiontable.setColumnWidth(0, optiontable.fontMetrics().horizontalAdvance(widestflag) + 48)
    optiontable.horizontalHeader().setStretchLastSection(True)
    # a row that holds None needs a value from the user, and the command does not give it yet
    optionrows: list[tuple[str, tuple[str, ...] | None]] = list(rows)

    def get_complete_rows() -> OptionRows:
        return tuple((flag, optionvalues) for flag, optionvalues in optionrows if optionvalues is not None)

    def set_option(row: int, flag: str) -> None:
        """Put a different option in a row.

        An empty option removes the row. An option in the empty last row adds a new row.
        """
        oldflag = optionrows[row][0] if row < len(optionrows) else ""
        if flag == oldflag or (flag and flag not in actionsbyflag):
            return
        if not flag:
            del optionrows[row]
        elif row < len(optionrows):
            optionrows[row] = (flag, get_default_tokens(actionsbyflag[flag]))
        else:
            optionrows.append((flag, get_default_tokens(actionsbyflag[flag])))
        show_option_rows()
        on_rows(get_complete_rows())

    def set_option_values(row: int, flag: str, optionvalues: tuple[str, ...] | None) -> None:
        # a field that loses the focus when the table changes can send the values of a row that is not there now
        if row < len(optionrows) and optionrows[row][0] == flag:
            optionrows[row] = (flag, optionvalues)
            on_rows(get_complete_rows())

    def make_flag_box(row: int, flag: str) -> QtWidgets.QComboBox:
        """Return a list of the options that the user can search, with the option of the row."""
        box = QtWidgets.QComboBox()
        box.setEditable(True)
        box.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        flags = ["", *tableflags]
        # an option that a hidden flag gave, e.g. an old spelling, stays in its row
        if flag not in flags:
            flags.append(flag)
        box.addItems(flags)
        for index, itemflag in enumerate(flags[1:], start=1):
            helptext = helptexts.get(actionsbyflag[itemflag].dest, "")
            box.setItemData(index, helptext, QtCore.Qt.ItemDataRole.ToolTipRole)
        if (completer := box.completer()) is not None:
            set_search_completion(completer)
        box.setCurrentText(flag)
        if (lineedit := box.lineEdit()) is not None:
            lineedit.setPlaceholderText("Add an option")

        def on_flag() -> None:
            # the new rows replace this box, thus the change waits until Qt finishes with the signal
            QtCore.QTimer.singleShot(0, window, lambda: set_option(row, box.currentText()))

        box.activated.connect(on_flag)
        return box

    def make_value_editor(row: int, flag: str, optionvalues: tuple[str, ...] | None) -> QtWidgets.QWidget:
        """Return the control for the value of an option, which matches the type of the option."""
        action = actionsbyflag[flag]
        kind = get_option_kind(action)
        editor = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(editor)
        layout.setContentsMargins(2, 0, 2, 0)
        if kind == "flag":
            label = QtWidgets.QLabel("no value")
            label.setEnabled(False)
            layout.addWidget(label, 1)
        elif kind == "choice":
            choicebox = QtWidgets.QComboBox()
            choicebox.addItems([str(choice) for choice in action.choices or ()])
            choicebox.setCurrentText(optionvalues[0] if optionvalues else "")

            def on_choice(text: str) -> None:
                set_option_values(row, flag, (text,))

            choicebox.currentTextChanged.connect(on_choice)
            layout.addWidget(choicebox, 1)
        elif kind == "int":
            spinbox = QtWidgets.QSpinBox()
            spinbox.setRange(-(2**31), 2**31 - 1)
            spinbox.setValue(int(optionvalues[0]) if optionvalues else action.default)
            spinbox.setKeyboardTracking(False)

            def on_spinbox(value: int) -> None:
                set_option_values(row, flag, (str(value),))

            spinbox.valueChanged.connect(on_spinbox)
            layout.addWidget(spinbox, 1)
        else:
            islist = kind == "list"
            fieldcount = action.nargs if isinstance(action.nargs, int) else 1
            fields = [QtWidgets.QLineEdit() for _ in range(fieldcount)]
            texts = [shlex.join(optionvalues or ())] if islist else list(optionvalues or ())
            for field, text in zip(fields, texts, strict=False):
                field.setText(text)
            for field in fields:
                if action.type in {int, float} and not islist:
                    validator = QtGui.QDoubleValidator() if action.type is float else QtGui.QIntValidator()
                    validator.setLocale(QtCore.QLocale.c())
                    field.setValidator(validator)
                if islist:
                    field.setPlaceholderText("values with spaces between them")
                elif action.default is not None:
                    field.setPlaceholderText(f"default {action.default}")
                layout.addWidget(field, 1)

            def on_fields() -> None:
                texts = [field.text().strip() for field in fields]
                newvalues: tuple[str, ...] | None
                if islist:
                    try:
                        newvalues = tuple(shlex.split(texts[0]))
                    except ValueError:
                        newvalues = None
                    # nargs "+" needs a value, and nargs "*" accepts none
                    if not newvalues and action.nargs == "+":
                        newvalues = None
                elif action.nargs == "?":
                    newvalues = (texts[0],) if texts[0] else ()
                else:
                    newvalues = tuple(texts) if all(texts) else None
                set_option_values(row, flag, newvalues)

            for field in fields:
                field.editingFinished.connect(on_fields)

        removebutton = make_glyph_button("✕", f"Remove {flag} from the command", f"Remove {flag}")
        removebutton.clicked.connect(lambda: QtCore.QTimer.singleShot(0, window, lambda: set_option(row, "")))
        layout.addWidget(removebutton)
        editor.setToolTip(helptexts.get(action.dest, ""))
        return editor

    def show_option_rows() -> None:
        """Make a row of the table for each option, and an empty row at the end that adds an option."""
        optiontable.setRowCount(len(optionrows) + 1)
        for row, (flag, optionvalues) in enumerate([*optionrows, ("", None)]):
            optiontable.setCellWidget(row, 0, make_flag_box(row, flag))
            if flag:
                optiontable.setCellWidget(row, 1, make_value_editor(row, flag, optionvalues))
            else:
                optiontable.removeCellWidget(row, 1)
        optiontable.resizeRowsToContents()
        # the sidebar scrolls, thus the table shows each row and does not scroll itself
        rowsheight = sum(optiontable.rowHeight(row) for row in range(optiontable.rowCount()))
        headerheight = optiontable.horizontalHeader().sizeHint().height()
        optiontable.setFixedHeight(rowsheight + headerheight + 2 * optiontable.frameWidth())

    def set_rows(newrows: OptionRows) -> None:
        # after the command rejects a change, the table shows the old options again without the rows with no value
        if get_complete_rows() != newrows:
            optionrows[:] = list(newrows)
            show_option_rows()

    show_option_rows()
    return optiontable, set_rows


def set_search_completion(completer: "QtWidgets.QCompleter") -> None:
    """Let a completer show each name that holds the typed text, e.g. "ion" shows averageionisation."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
    completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
    completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
    completer.setMaxVisibleItems(15)


def make_completer(names: "Sequence[str]", parent: "QtWidgets.QWidget") -> "QtWidgets.QCompleter":
    """Return a completer that finds each name that holds the typed text, e.g. "ion" finds averageionisation."""
    from PySide6 import QtWidgets

    completer = QtWidgets.QCompleter(list(names), parent)
    set_search_completion(completer)
    return completer


def add_command_section(
    panellayout: "QtWidgets.QVBoxLayout",
) -> "tuple[QtWidgets.QPlainTextEdit, QtWidgets.QPushButton]":
    """Add the command after the sections of the panel, and return its text box and its Copy button."""
    return add_copy_box(
        panellayout, "Command", f"Copy the command to the clipboard ({get_menu_shortcut_texts()['Copy Command']})"
    )


def add_copy_box(
    panellayout: "QtWidgets.QVBoxLayout", title: str, copytooltip: str, *, maxlines: int = 8, wraplines: bool = True
) -> "tuple[QtWidgets.QPlainTextEdit, QtWidgets.QPushButton]":
    """Add a section with a read-only box of text and a Copy button, and return the box and the button.

    The box shows up to maxlines lines, and a longer text scrolls. Code keeps its lines without a wrap.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    _, grid = add_section(panellayout, title)
    # the text takes the width, and the Copy button keeps its size at the right
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 0)

    # the instance of the box holds no reference to the window. PySide keeps each class, thus the class must hold none
    class CommandBox(QtWidgets.QPlainTextEdit):
        """The box of the text, which fits its height to the wrapped lines when its width changes."""

        @t.override
        def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
            super().resizeEvent(event)
            if event.size().width() != event.oldSize().width():
                fit_command_box(self)

    textbox = CommandBox()
    textbox.setProperty("maxlines", maxlines)
    if not wraplines:
        textbox.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
    textbox.setReadOnly(True)
    textbox.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    fit_command_box(textbox)
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip(copytooltip)
    grid.addWidget(textbox, 0, 0)
    grid.addWidget(copybutton, 0, 1, QtCore.Qt.AlignmentFlag.AlignTop)
    return textbox, copybutton


def parse_command_tokens(parser: argparse.ArgumentParser, tokens: "Sequence[str]") -> argparse.Namespace | None:
    """Return the arguments of the tokens of a command, or None if the parser rejects them.

    The plot of the same tokens fails too, and the status line then gives the reason.
    """
    try:
        return parser.parse_args(separate_trailing_folders(tokens))
    except SystemExit:
        return None


def get_changed_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace, skip: "Collection[str]" = ()
) -> dict[str, t.Any]:
    """Return each argument that differs from its default, in the order of the parser."""
    return {dest: value for dest, value in vars(args).items() if dest not in skip and value != parser.get_default(dest)}


# a list that is longer than this on one line gives one item on each line
PYTHON_LINE_LENGTH: t.Final = 100


def format_python_value(value: t.Any, indent: int) -> str:
    """Return the Python text of a value of an argument. A list that is too long for one line gives one item a line."""
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        # the JSON text of a string is also a Python string, and it has the double quotes of the usual Python style
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        opening, closing = ("(", ")") if isinstance(value, tuple) else ("[", "]")
        items = [format_python_value(item, indent + 4) for item in value]
        onetuple = "," if isinstance(value, tuple) and len(items) == 1 else ""
        oneline = f"{opening}{', '.join(items)}{onetuple}{closing}"
        if indent + len(oneline) <= PYTHON_LINE_LENGTH:
            return oneline
        itemlines = "".join(f"{' ' * (indent + 4)}{item},\n" for item in items)
        return f"{opening}\n{itemlines}{' ' * indent}{closing}"
    return repr(value)


def get_python_call(functionname: str, kwargs: "Mapping[str, t.Any]") -> str:
    """Return the Python code that calls the main function of a command with these keyword arguments."""
    arguments = "".join(f"    {name}={format_python_value(value, 4)},\n" for name, value in kwargs.items())
    call = f"{functionname}(\n{arguments})" if arguments else f"{functionname}()"
    return f"import artistools as at\n\n{call}"


# a box of text shows at least this number of lines
MIN_COMMAND_LINES: t.Final = 3


def set_command_text(commandtext: "QtWidgets.QPlainTextEdit", command: str) -> None:
    """Show the command in its box, and give the box the height of the lines of the command."""
    if commandtext.toPlainText() != command:
        commandtext.setPlainText(command)
        fit_command_box(commandtext)


def fit_command_box(commandtext: "QtWidgets.QPlainTextEdit") -> None:
    """Give the command box the height of the wrapped lines of its text, and keep that height inside the limits.

    The layout of the document gives the wrapped lines at the width of the box. The rectangle of each block is a few
    pixels taller than its lines, and the box needs those pixels, else it scrolls.
    """
    from PySide6 import QtWidgets

    document = commandtext.document()
    layout = document.documentLayout()
    textlines = math.ceil(layout.documentSize().height())
    textheight = sum(
        layout.blockBoundingRect(document.findBlockByNumber(i)).height() for i in range(document.blockCount())
    )
    shownlines = min(max(textlines, MIN_COMMAND_LINES), int(commandtext.property("maxlines")))
    boxheight = textheight + (shownlines - textlines) * commandtext.fontMetrics().lineSpacing()
    # a box with no wrap shows a horizontal scroll bar for a long line, and the bar must not cover the last line
    if commandtext.lineWrapMode() == QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap:
        boxheight += commandtext.horizontalScrollBar().sizeHint().height()
    commandtext.setFixedHeight(math.ceil(boxheight + 2 * document.documentMargin() + 2 * commandtext.frameWidth()))


def start_play_timer(playtimer: "QtCore.QTimer", plotseconds: float, fps: float) -> None:
    """Start the pause before the next step of Play, thus a step takes 1/fps seconds.

    The old plot stays in view while the worker draws the next one, thus the pause is the rest of the time of the step.
    A plot that takes longer than 1/fps seconds gives a lower rate.
    """
    playtimer.start(max(0, round(1000.0 / fps - plotseconds * 1000.0)))


def get_icon(symbol: str, themeicon: "QtGui.QIcon.ThemeIcon") -> "QtGui.QIcon":
    """Return the SF Symbol of the name on macOS, else the icon of the theme.

    Qt gives an SF Symbol for its name on macOS. The result can be an empty icon, and a tool button then shows its
    text.
    """
    from PySide6 import QtGui

    icon = QtGui.QIcon.fromTheme(symbol) if sys.platform == "darwin" else QtGui.QIcon()
    return QtGui.QIcon.fromTheme(themeicon) if icon.isNull() else icon


def make_segmented_control(labels: "Sequence[str]", tooltips: "Sequence[str]") -> "QtWidgets.QTabBar":
    """Return a control of side-by-side segments, in which the user selects one segment, e.g. one mode of the time.

    The macOS style of Qt draws a tab bar as a segmented control, which the apps of macOS use for a choice of modes.
    """
    from PySide6 import QtWidgets

    segments = QtWidgets.QTabBar()
    segments.setDrawBase(False)
    segments.setExpanding(False)
    for index, (label, tooltip) in enumerate(zip(labels, tooltips, strict=True)):
        segments.addTab(label)
        segments.setTabToolTip(index, tooltip)
    return segments


def make_step_button(*, forward: bool) -> "QtWidgets.QToolButton":
    """Return a button that moves the time to the next or the previous timestep, with the icon of the platform."""
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    button = QtWidgets.QToolButton()
    if forward:
        icon = get_icon("forward.end.fill", QtGui.QIcon.ThemeIcon.MediaSkipForward)
        button.setToolTip("Move the time to the next timestep (Right key)")
        button.setAccessibleName("Next Timestep")
    else:
        icon = get_icon("backward.end.fill", QtGui.QIcon.ThemeIcon.MediaSkipBackward)
        button.setToolTip("Move the time to the previous timestep (Left key)")
        button.setAccessibleName("Previous Timestep")
    button.setIcon(icon)
    # a platform with no icon shows the arrow as text
    if icon.isNull():
        button.setText("▶" if forward else "◀")
    return button


def make_play_button(tooltip: str) -> "QtWidgets.QToolButton":
    """Return the checkable Play button, which shows Pause and its icon while Play runs."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    playicon = get_icon("play.fill", QtGui.QIcon.ThemeIcon.MediaPlaybackStart)
    pauseicon = get_icon("pause.fill", QtGui.QIcon.ThemeIcon.MediaPlaybackPause)
    play = QtWidgets.QToolButton()
    play.setCheckable(True)
    play.setText("Play")
    play.setIcon(playicon)
    play.setToolTip(tooltip)
    play.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

    def show_play_state(checked: bool) -> None:
        play.setIcon(pauseicon if checked else playicon)
        play.setText("Pause" if checked else "Play")

    play.toggled.connect(show_play_state)
    return play


def make_play_row(
    stepbuttons: "Sequence[QtWidgets.QWidget]",
    label: "QtWidgets.QLabel",
    fpsbox: "QtWidgets.QDoubleSpinBox",
    playbutton: "QtWidgets.QWidget",
) -> "QtWidgets.QHBoxLayout":
    """Return one row of the Time section: the step buttons, the label of the timesteps, the frame rate, and Play.

    One row for these items keeps more of the other controls in view. The step buttons touch, as a pair of arrows.
    """
    from PySide6 import QtWidgets

    steps = QtWidgets.QHBoxLayout()
    steps.setSpacing(0)
    for button in stepbuttons:
        steps.addWidget(button)
    row = QtWidgets.QHBoxLayout()
    row.addLayout(steps)
    # a narrow sidebar puts the text of the label on two lines
    label.setWordWrap(True)
    row.addWidget(label, 1)
    row.addWidget(QtWidgets.QLabel("FPS:"))
    row.addWidget(fpsbox)
    row.addWidget(playbutton)
    return row


def make_fps_box() -> "QtWidgets.QDoubleSpinBox":
    """Return the box of the frame rate of Play, with up and down buttons."""
    from PySide6 import QtWidgets

    fpsbox = QtWidgets.QDoubleSpinBox()
    # the minimum is one step, thus the up and down buttons keep each value on the steps of 0.5
    fpsbox.setRange(0.5, 60.0)
    fpsbox.setDecimals(1)
    fpsbox.setSingleStep(0.5)
    fpsbox.setValue(get_float_setting("playfps", DEFAULT_PLAY_FPS))
    fpsbox.setToolTip(
        "The frames per second of Play. A plot that takes longer than one frame gives a lower rate. After the last"
        " step, Play starts again at the first step."
    )
    return fpsbox


# the colour of an error and of a warning on a light window and on a dark window
MESSAGE_COLOURS_LIGHT: t.Final = ("#b3261e", "#9a5b00")
MESSAGE_COLOURS_DARK: t.Final = ("#ff8a80", "#ffc164")


class StatusBar(t.NamedTuple):
    """The labels of the status bar of a window, and its help button."""

    message: "QtWidgets.QLabel"
    readout: "QtWidgets.QLabel"
    drawtime: "QtWidgets.QLabel"
    helpbutton: "QtWidgets.QToolButton"


def get_message_colours() -> tuple[str, str]:
    """Return the colour of an error and of a warning for the current appearance.

    The dark red and the dark amber of a light window have too little contrast on a dark window, thus a dark window
    takes a light red and a light amber.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    # a test can run the queue with a QCoreApplication alone, which has no palette
    if not isinstance(application := QtCore.QCoreApplication.instance(), QtGui.QGuiApplication):
        return MESSAGE_COLOURS_LIGHT
    window = application.palette().color(QtGui.QPalette.ColorRole.Window)
    return MESSAGE_COLOURS_LIGHT if window.lightness() > 128 else MESSAGE_COLOURS_DARK


def show_status_message(statusbar: StatusBar, message: str | None, warning: str) -> None:
    """Show the message of a rejected plot in red, or else the last warning of the plot in amber."""
    errorcolour, warningcolour = get_message_colours()
    if message is not None:
        statusbar.message.setStyleSheet(f"color: {errorcolour}")
        statusbar.message.setText(message)
    else:
        statusbar.message.setStyleSheet(f"color: {warningcolour}")
        statusbar.message.setText(warning)


def show_status_note(statusbar: StatusBar, note: str) -> None:
    """Show a note about an action that succeeded, e.g. a saved file, in the colour of normal text."""
    statusbar.message.setStyleSheet("")
    statusbar.message.setText(note)


def make_status_bar(window: "QtWidgets.QMainWindow") -> StatusBar:
    """Return the labels of the status bar and its help button.

    The status bar gives the messages at the left, and the readout, the time of the plot, and the help at the right.
    """
    from PySide6 import QtWidgets

    statusbar = window.statusBar()
    messagelabel = QtWidgets.QLabel()
    messagelabel.setStyleSheet(f"color: {get_message_colours()[0]}")
    readoutlabel = QtWidgets.QLabel()
    drawtimelabel = QtWidgets.QLabel()
    helpbutton = QtWidgets.QToolButton()
    helpbutton.setText("?")
    helpbutton.setToolTip("Show the keys and the mouse actions of the window (?)")
    helpbutton.setAccessibleName("Keys and Mouse Actions")
    statusbar.addWidget(messagelabel, stretch=1)
    for widget in (readoutlabel, drawtimelabel, helpbutton):
        statusbar.addPermanentWidget(widget)
    return StatusBar(message=messagelabel, readout=readoutlabel, drawtime=drawtimelabel, helpbutton=helpbutton)


def get_menu_items() -> "list[tuple[str, str, QtGui.QKeySequence]]":
    """Return the menu, the text, and the shortcut of each menu item of a viewer.

    The texts use the capitals of a title, and an item that opens a dialog ends with an ellipsis, as in the apps of
    macOS. On macOS, Ctrl is the Command key and Meta is the Control key.
    """
    from PySide6 import QtGui

    standardkey = QtGui.QKeySequence.StandardKey
    return [
        ("File", "Open Model…", QtGui.QKeySequence(standardkey.Open)),
        ("File", "Reload Data", QtGui.QKeySequence(standardkey.Refresh)),
        ("File", "Save Figure…", QtGui.QKeySequence(standardkey.Save)),
        ("File", "Export Animation…", QtGui.QKeySequence("Ctrl+Shift+E")),
        ("File", "Close Window", QtGui.QKeySequence(standardkey.Close)),
        ("Edit", "Undo", QtGui.QKeySequence(standardkey.Undo)),
        ("Edit", "Redo", QtGui.QKeySequence(standardkey.Redo)),
        ("Edit", "Copy Figure", QtGui.QKeySequence(standardkey.Copy)),
        ("Edit", "Copy Command", QtGui.QKeySequence("Ctrl+Shift+C")),
        ("Edit", "Copy Python", QtGui.QKeySequence("Ctrl+Alt+C")),
        # macOS moves this item to the menu of the application
        ("Edit", "Settings…", QtGui.QKeySequence("Ctrl+,")),
        ("View", "Play", QtGui.QKeySequence("Space")),
        ("View", "Cancel Plot", QtGui.QKeySequence("Ctrl+.")),
        ("View", "Hide Sidebar", QtGui.QKeySequence("Ctrl+Meta+S")),
        ("View", "Enter Full Screen", QtGui.QKeySequence(standardkey.FullScreen)),
        ("Window", "Minimize", QtGui.QKeySequence("Ctrl+M")),
        ("Window", "Zoom", QtGui.QKeySequence()),
        ("Help", "Keys and Mouse Actions", QtGui.QKeySequence("?")),
        ("Help", "artistools Help", QtGui.QKeySequence()),
        # macOS moves this item to the menu of the application, with the name of the application
        ("Help", "About", QtGui.QKeySequence()),
    ]


def get_menu_shortcut_texts() -> dict[str, str]:
    """Return the shortcut of each menu item by its text, in the form of the platform, e.g. ⌘S or Ctrl+S."""
    from PySide6 import QtGui

    return {text: keys.toString(QtGui.QKeySequence.SequenceFormat.NativeText) for _menu, text, keys in get_menu_items()}


def copy_text(text: str) -> None:
    """Print the text, e.g. the command, and put it on the clipboard."""
    from PySide6 import QtWidgets

    print(text)
    QtWidgets.QApplication.clipboard().setText(text)


def set_edit_text(edit: "QtWidgets.QLineEdit", text: str) -> None:
    """Show the text in a field, unless the user types in that field.

    A plot or a Play step can end while the user types. Without this check, the text of the values replaces the
    text that the user typed. The handler of a field calls setModified(False), thus a field shows new values again
    after the user presses Return.
    """
    if not (edit.hasFocus() and edit.isModified()):
        edit.setText(text)


def set_spin_value(box: "QtWidgets.QSpinBox | QtWidgets.QDoubleSpinBox", value: float) -> None:
    """Show the value in a spin box, unless the user types in that box.

    A box with no keyboard tracking keeps the typed text until the user presses Return. setValue writes the text
    again also for the same value, thus the end of a plot erased the text that the user typed.
    """
    from PySide6 import QtWidgets

    if box.hasFocus() and math.isclose(box.value(), value):
        return
    if isinstance(box, QtWidgets.QSpinBox):
        box.setValue(round(value))
    else:
        box.setValue(value)


def make_note_label() -> "QtWidgets.QLabel":
    """Return a label for a note in a section, which shows a summary and a link to the full text.

    A long note pushed the controls below it down the panel. The note text takes the grey of a placeholder, and a
    disabled label would take no click on its link.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    label = QtWidgets.QLabel()
    label.setWordWrap(True)
    label.setTextFormat(QtCore.Qt.TextFormat.RichText)
    label.setForegroundRole(QtGui.QPalette.ColorRole.PlaceholderText)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse)

    def on_link(link: str) -> None:
        label.setProperty("noteexpanded", link == "more")
        show_note_text(label)

    label.linkActivated.connect(on_link)
    label.hide()
    return label


def set_note_text(label: "QtWidgets.QLabel", summary: str, detail: str = "") -> None:
    """Show a summary in a note label, with a More link to the detail if the detail gives more. No summary hides it."""
    label.setProperty("notesummary", summary)
    label.setProperty("notedetail", detail if detail != summary else "")
    label.setToolTip(detail or summary)
    show_note_text(label)
    label.setVisible(bool(summary))


def show_note_text(label: "QtWidgets.QLabel") -> None:
    """Show the summary or the detail of a note label, with the link that changes between them."""
    summary, detail = str(label.property("notesummary") or ""), str(label.property("notedetail") or "")
    if not detail:
        label.setText(html.escape(summary))
    elif label.property("noteexpanded"):
        label.setText(html.escape(detail).replace("\n", "<br>") + ' <a href="less">Less</a>')
    else:
        label.setText(html.escape(summary) + ' <a href="more">More…</a>')


def get_first_sentence(text: str) -> str:
    """Return the first sentence of a text, or all the text if it has one sentence."""
    return re.split(r"(?<=[.:])\s+(?=[A-Z])", text, maxsplit=1)[0]


def align_section_labels(window: "QtCore.QObject") -> None:
    """Give the labels at the start of the rows of each section one width, thus the controls start at one place.

    A row of make_row_layout puts its first label and its control in one group. A label in the first column of the grid
    of a section, or at the start of a row layout in that column, also takes the width. A label can change its text,
    e.g. for a new unit, thus the window calls the function again after each plot.
    """
    from PySide6 import QtWidgets

    if not isinstance(window, QtWidgets.QWidget):
        return
    for content in window.findChildren(QtWidgets.QWidget, SECTION_CONTENT_NAME):
        grid = content.layout()
        if not isinstance(grid, QtWidgets.QGridLayout):
            continue
        labels: list[QtWidgets.QLabel] = []
        for index in range(grid.count()):
            _, column, _, columnspan = t.cast("tuple[int, int, int, int]", grid.getItemPosition(index))
            item = grid.itemAt(index)
            if column != 0 or item is None:
                continue
            if isinstance(widget := item.widget(), QtWidgets.QLabel) and columnspan == 1:
                labels.append(widget)
            elif (rowlayout := item.layout()) is not None and (label := get_row_start_label(rowlayout)) is not None:
                labels.append(label)
        if len(labels) > 1:
            width = max(label.sizeHint().width() for label in labels)
            for label in labels:
                label.setMinimumWidth(width)


def get_row_start_label(rowlayout: "QtWidgets.QLayout") -> "QtWidgets.QLabel | None":
    """Return the label at the start of a row layout or of a row of make_row_layout if a control follows it, or None."""
    from PySide6 import QtWidgets

    if (rowitem := rowlayout.itemAt(0)) is None or (wraprow := rowitem.widget()) is None:
        return None
    if isinstance(wraprow, QtWidgets.QLabel):
        return wraprow if rowlayout.count() > 1 else None
    groups: list[QtWidgets.QWidget] = getattr(wraprow, "groups", [])
    firstgroup = groups[0] if groups else None
    if firstgroup is None or (grouplayout := firstgroup.layout()) is None or grouplayout.count() < 2:
        return None
    firstitem = grouplayout.itemAt(0)
    label = firstitem.widget() if firstitem is not None else None
    return label if isinstance(label, QtWidgets.QLabel) else None
