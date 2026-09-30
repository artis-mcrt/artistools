"""Make the sections of a viewer window for the viewing direction, the axes, and the time."""

import typing as t
from functools import partial
from pathlib import Path

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from PySide6 import QtCore
    from PySide6 import QtWidgets


from artistools.viewertools.core import get_direction_choices
from artistools.viewertools.core import get_direction_kinds
from artistools.viewertools.core import get_short_number
from artistools.viewertools.widgets import add_row
from artistools.viewertools.widgets import add_section
from artistools.viewertools.widgets import make_fps_box
from artistools.viewertools.widgets import make_play_button
from artistools.viewertools.widgets import make_play_row
from artistools.viewertools.widgets import make_slider
from artistools.viewertools.widgets import make_step_button
from artistools.viewertools.widgets import set_edit_text

# each kind of viewing direction: the value of the kind, the text of its choice, and the option that gives its help
DIRECTION_KINDS: t.Final = (
    ("", "All directions", ""),
    ("bin", "-plotviewingangle", "plotviewingangle"),
    ("phi", "--average_over_phi_angle", "average_over_phi_angle"),
    ("theta", "--average_over_theta_angle", "average_over_theta_angle"),
    ("vpkt", "-plotvspecpol", "plotvspecpol"),
)


class DirectionChoice(t.NamedTuple):
    """The viewing direction of the values of a window: the kind, the bins, and the unit of the angles."""

    kind: str
    bins: tuple[int, ...]
    usedegrees: bool


def get_direction_bins(runfolder: Path | str, kind: str, *, usedegrees: bool) -> list[tuple[int, str]]:
    """Return each bin of a kind of viewing direction with its label, and first the average over all the directions.

    The average is bin -1. An observer of the virtual packets has no such average.
    """
    if not kind:
        return []
    averagebin = [] if kind == "vpkt" else [(-1, "All directions")]
    return [*averagebin, *get_direction_choices(Path(runfolder), kind, usedegrees=usedegrees)]


def add_direction_section(
    panellayout: "QtWidgets.QVBoxLayout",
    helptexts: "Mapping[str, str]",
    runfolder: Path | str,
    get_choice: "Callable[[], tuple[DirectionChoice, bool]]",
    on_change: "Callable[[DirectionChoice], None]",
    show_error: "Callable[[str], None]",
) -> "tuple[Callable[[], None], Callable[[Path | str], None]]":
    """Add the section of the viewing direction: the kind of direction, --usedegrees, and a list of the bins.

    get_choice gives the choice of the values of the window, and whether the plot draws one bin, e.g. an emission plot.
    Such a plot has a radio button for each bin, and a click on a bin replaces the bin. on_change receives each new
    choice. A new kind keeps each bin that the kind also has.

    Return the function that shows the choice of get_choice, and the function that reads the kinds of direction and
    the labels of the bins of a new first run.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    _, directiongrid = add_section(panellayout, "Viewing direction")
    kindbox = QtWidgets.QComboBox()
    usedegreescheck = QtWidgets.QCheckBox("--usedegrees")
    usedegreescheck.setToolTip(helptexts.get("usedegrees", ""))
    add_row(directiongrid, 0, [kindbox, usedegreescheck])
    # the plot can show several directions at once, thus each bin has a check box. The list scrolls, and the label of
    # a bin is long, thus the list takes the full width of the sidebar
    binbox = QtWidgets.QScrollArea()
    binbox.setWidgetResizable(True)
    binbox.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    binbox.setToolTip(
        "The direction bins of the plot, or the observers of the virtual packets. A plot that draws one bin, e.g. an"
        " emission plot, shows a radio button for each bin"
    )
    directiongrid.addWidget(binbox, 1, 0, 1, -1)
    binchecks: dict[int, QtWidgets.QAbstractButton] = {}
    # the labels of the bins come from the files of the run, thus the section reads them one time for each kind
    binlabels: dict[tuple[str, bool], list[tuple[int, str]]] = {}
    shownlist: tuple[str, bool, bool] | None = None
    run: list[Path | str] = [runfolder]

    def get_bins(kind: str, usedegrees: bool) -> list[tuple[int, str]]:
        if (kind, usedegrees) not in binlabels:
            binlabels[kind, usedegrees] = get_direction_bins(run[0], kind, usedegrees=usedegrees)
        return binlabels[kind, usedegrees]

    def set_run(newrunfolder: Path | str) -> None:
        nonlocal shownlist
        run[0], shownlist = newrunfolder, None
        binlabels.clear()
        kinds = get_direction_kinds(Path(newrunfolder))
        with QtCore.QSignalBlocker(kindbox):
            kindbox.clear()
            for kind, text, dest in DIRECTION_KINDS:
                if kind in kinds:
                    kindbox.addItem(text, kind)
                    kindbox.setItemData(
                        kindbox.count() - 1, helptexts.get(dest, ""), QtCore.Qt.ItemDataRole.ToolTipRole
                    )

    def show_bins(kind: str, usedegrees: bool, onebin: bool) -> bool:
        """Fill the list with a button for each bin of a kind of direction. Return whether the list is new."""
        nonlocal shownlist
        if (kind, usedegrees, onebin) == shownlist:
            return False
        checklist = QtWidgets.QWidget()
        checklayout = QtWidgets.QVBoxLayout(checklist)
        checklayout.setContentsMargins(6, 4, 6, 4)
        checklayout.setSpacing(2)
        binchecks.clear()
        for dirbin, label in get_bins(kind, usedegrees):
            text = f"{dirbin}: {label}"
            check = QtWidgets.QRadioButton(text) if onebin else QtWidgets.QCheckBox(text)
            # a click on a radio button also clears the previous button, thus the toggled signal calls the handler two times
            check.clicked.connect(on_direction)
            checklayout.addWidget(check)
            binchecks[dirbin] = check
        checklayout.addStretch(1)
        # the list shows up to 6 bins, and a longer list scrolls
        shownbins = min(max(len(binchecks), 1), 6)
        lineheight = max((check.sizeHint().height() for check in binchecks.values()), default=20)
        binbox.setFixedHeight(shownbins * (lineheight + 2) + 10)
        binbox.setWidget(checklist)
        shownlist = (kind, usedegrees, onebin)
        return True

    def show() -> None:
        choice, onebin = get_choice()
        with QtCore.QSignalBlocker(kindbox), QtCore.QSignalBlocker(usedegreescheck):
            kindbox.setCurrentIndex(max(kindbox.findData(choice.kind), 0))
            usedegreescheck.setChecked(choice.usedegrees)
        usedegreescheck.setEnabled(bool(choice.kind))
        isnewlist = show_bins(choice.kind, choice.usedegrees, onebin)
        for dirbin, check in binchecks.items():
            with QtCore.QSignalBlocker(check):
                check.setChecked(dirbin in choice.bins)
        # a new list scrolls to the first checked bin, which can be far down a list of 100 bins
        if isnewlist and choice.bins and (firstcheck := binchecks.get(choice.bins[0])):
            QtCore.QTimer.singleShot(0, binbox, partial(binbox.ensureWidgetVisible, firstcheck))
        # all the directions have no bin to select, thus the list shows only for a kind of direction
        binbox.setVisible(bool(choice.kind))

    def on_direction() -> None:
        kind: str = kindbox.currentData()
        usedegrees = usedegreescheck.isChecked()
        current, onebin = get_choice()
        if kind == current.kind:
            bins = tuple(dirbin for dirbin, check in binchecks.items() if check.isChecked())
            if kind and not bins:
                show_error("A kind of viewing direction needs one direction bin at least")
                return
            newbins = tuple(dirbin for dirbin in bins if dirbin not in current.bins)
            # a plot of one bin replaces the previous bin with the bin of the click
            if onebin and newbins:
                bins = newbins[:1]
        else:
            kindbins = [dirbin for dirbin, _ in get_bins(kind, usedegrees)]
            bins = tuple(dirbin for dirbin in current.bins if dirbin in kindbins) or tuple(kindbins[:1])
            if onebin:
                bins = bins[:1]
        on_change(DirectionChoice(kind=kind, bins=bins, usedegrees=usedegrees))

    kindbox.currentIndexChanged.connect(on_direction)
    usedegreescheck.toggled.connect(on_direction)
    set_run(runfolder)
    return show, set_run


def add_y_axis_actions(
    menu: "QtWidgets.QMenu",
    islog: bool | None,
    haslimits: bool,
    set_yscale: "Callable[[str], None]",
    clear_limits: "Callable[[], None]",
) -> None:
    """Add the actions on the y axis of a frame to a context menu: the scale, and the automatic y range.

    islog is None for an axis with no choice of scale, e.g. a magnitude. haslimits enables Auto Y Range, which removes
    the limits that the user gave.
    """
    if islog is not None:
        menu.addAction("Linear Scale" if islog else "Log Scale").triggered.connect(
            partial(set_yscale, "linear" if islog else "log")
        )
    resetaction = menu.addAction("Auto Y Range")
    resetaction.setEnabled(haslimits)
    resetaction.triggered.connect(clear_limits)
    menu.addSeparator()


def read_limit_fields(
    fields: "Sequence[tuple[QtWidgets.QLineEdit, str]]", show_error: "Callable[[str], None]"
) -> list[str] | None:
    """Return the text of two limit fields as numbers, or "" for an empty field, which gives the automatic limit.

    fields gives each field with its flag, e.g. -ymin. A field with text that is not a number, or a minimum that is not
    less than the maximum, gives the reason to show_error and None.
    """
    limits: list[str] = []
    for edit, flag in fields:
        edit.setModified(False)
        text = edit.text().strip()
        try:
            limits.append(format(float(text), ".10g") if text else "")
        except ValueError:
            show_error(f"Give a number for {flag}, or clear the field for the automatic limit")
            return None
    if limits[0] and limits[1] and not float(limits[0]) < float(limits[1]):
        show_error(f"Give a {fields[0][1]} that is less than {fields[1][1]}")
        return None
    return limits


def add_y_limits_row(
    grid: "QtWidgets.QGridLayout",
    row: int,
    helptexts: "Mapping[str, str]",
    set_limits: "Callable[[str, str], None]",
    get_drawn_limits: "Callable[[], tuple[float, float] | None]",
    show_error: "Callable[[str], None]",
) -> "Callable[[str, str], None]":
    """Add the fields of -ymin and -ymax and the button Set current y range to a row of a grid.

    set_limits receives the text of each limit, and "" gives the automatic limit. get_drawn_limits gives the y limits of
    the plot on the screen, or None while the plot does not show the values of the controls. Return the function that
    shows the limits of the values.
    """
    from PySide6 import QtWidgets

    yminedit, ymaxedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    for edit, dest in ((yminedit, "ymin"), (ymaxedit, "ymax")):
        edit.setFixedWidth(110)
        edit.setPlaceholderText("auto")
        edit.setToolTip(helptexts.get(dest, ""))
    setrangebutton = QtWidgets.QPushButton("Set current y range")
    setrangebutton.setToolTip(
        "Set y min and y max to the current range of the y axis. The axis then stays the same when a different option"
        " changes. Clear a field to get the automatic limit at that end again."
    )
    add_row(grid, row, [QtWidgets.QLabel("-ymin"), yminedit, QtWidgets.QLabel("-ymax"), ymaxedit, setrangebutton])

    def on_edit() -> None:
        if (limits := read_limit_fields([(yminedit, "-ymin"), (ymaxedit, "-ymax")], show_error)) is not None:
            set_limits(limits[0], limits[1])

    def on_set_range() -> None:
        if (drawnlimits := get_drawn_limits()) is None:
            show_error("The plot on the screen does not show the new values yet. Wait for the plot, then try again")
            return
        # the limits of the plot on the screen become the limits of the command, thus the plot does not change. An
        # inverted axis, e.g. of a magnitude, gives the lower number to -ymin
        low, high = (get_short_number(limit) for limit in sorted(drawnlimits))
        set_limits(low, high)

    def show_limits(ymin: str, ymax: str) -> None:
        set_edit_text(yminedit, ymin)
        set_edit_text(ymaxedit, ymax)

    yminedit.editingFinished.connect(on_edit)
    ymaxedit.editingFinished.connect(on_edit)
    setrangebutton.clicked.connect(on_set_range)
    return show_limits


def make_figscale_box(helptexts: "Mapping[str, str]") -> "QtWidgets.QDoubleSpinBox":
    """Return the box of -figscale. A typed number applies when the user presses Return or leaves the box."""
    from PySide6 import QtWidgets

    figscalebox = QtWidgets.QDoubleSpinBox()
    figscalebox.setRange(0.1, 10.0)
    figscalebox.setSingleStep(0.1)
    figscalebox.setDecimals(2)
    figscalebox.setKeyboardTracking(False)
    figscalebox.setToolTip(helptexts.get("figscale", ""))
    return figscalebox


def make_xscale_box(helptexts: "Mapping[str, str]") -> "QtWidgets.QComboBox":
    """Return a box with the choices Linear and Log for the x axis. The index 1 gives --logscalex.

    The commands have no -xscale, thus the box has no automatic scale as the y scale box has.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    xscalebox = QtWidgets.QComboBox()
    for text, tooltip in (("Linear", "A linear x axis"), ("Log", f"--logscalex: {helptexts.get('logscalex', '')}")):
        xscalebox.addItem(text)
        xscalebox.setItemData(xscalebox.count() - 1, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    xscalebox.setToolTip("The scale of the x axis. Log gives --logscalex")
    return xscalebox


class TimeControls(t.NamedTuple):
    """The controls of the time section of a viewer: the time, the width of the time range, and Play."""

    timeslider: "QtWidgets.QSlider"
    timeedit: "QtWidgets.QLineEdit"
    # a viewer can put a different control in the place of the label, e.g. the rule of a continuous width
    widthlabel: "QtWidgets.QLabel"
    widthslider: "QtWidgets.QSlider"
    widthedit: "QtWidgets.QLineEdit"
    # the text of the timesteps of the time range, beside the step buttons
    timestepslabel: "QtWidgets.QLabel"
    stepbuttons: "tuple[QtWidgets.QToolButton, QtWidgets.QToolButton]"
    fpsbox: "QtWidgets.QDoubleSpinBox"
    playbutton: "QtWidgets.QToolButton"


def add_time_controls(grid: "QtWidgets.QGridLayout", firstrow: int, tips: tuple[str, str, str]) -> TimeControls:
    """Add the row of the time, the row of the width of the time range, and the row of Play to a grid.

    tips gives the tooltip of the time, of the width, and of the Play button. A viewer can put a control of its own on
    a row, e.g. the range of a plot against time on the row of the time.
    """
    from PySide6 import QtWidgets

    timetip, widthtip, playtip = tips
    timeslider, widthslider = make_slider(), make_slider()
    timeedit, widthedit = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    widthlabel = QtWidgets.QLabel("Δ timesteps:")
    for row, (label, slider, edit, tip) in enumerate(
        [
            (QtWidgets.QLabel("Time [d]:"), timeslider, timeedit, timetip),
            (widthlabel, widthslider, widthedit, widthtip),
        ],
        start=firstrow,
    ):
        edit.setFixedWidth(110)
        for widget in (slider, edit):
            widget.setToolTip(tip)
        grid.addWidget(label, row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(edit, row, 2)
    timestepslabel = QtWidgets.QLabel()
    stepbuttons = (make_step_button(forward=False), make_step_button(forward=True))
    fpsbox = make_fps_box()
    playbutton = make_play_button(playtip)
    grid.addLayout(make_play_row(list(stepbuttons), timestepslabel, fpsbox, playbutton), firstrow + 2, 0, 1, -1)
    return TimeControls(
        timeslider=timeslider,
        timeedit=timeedit,
        widthlabel=widthlabel,
        widthslider=widthslider,
        widthedit=widthedit,
        timestepslabel=timestepslabel,
        stepbuttons=stepbuttons,
        fpsbox=fpsbox,
        playbutton=playbutton,
    )


def connect_time_keys(
    window: "QtWidgets.QWidget",
    stepbuttons: "tuple[QtWidgets.QToolButton, QtWidgets.QToolButton]",
    step_time: "Callable[[int], None]",
    step_width: "Callable[[int], None]",
    move_to_ends: "tuple[Callable[[], None], Callable[[], None]]",
    extrakeys: "Sequence[tuple[QtCore.Qt.Key, Callable[[], None]]]" = (),
) -> None:
    """Connect the step buttons and the keys of the time section.

    Left and Right move the time by one timestep, Up and Down change the width by one timestep, and Home and End move
    the time range to the first or to the last timestep. extrakeys gives the other keys of one viewer. A text field
    takes these keys while it has the focus, and the shortcuts apply otherwise.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    previousbutton, nextbutton = stepbuttons
    previousbutton.clicked.connect(partial(step_time, -1))
    nextbutton.clicked.connect(partial(step_time, 1))
    move_to_start, move_to_end = move_to_ends
    for key, callback in (
        (QtCore.Qt.Key.Key_Left, partial(step_time, -1)),
        (QtCore.Qt.Key.Key_Right, partial(step_time, 1)),
        (QtCore.Qt.Key.Key_Up, partial(step_width, 1)),
        (QtCore.Qt.Key.Key_Down, partial(step_width, -1)),
        (QtCore.Qt.Key.Key_Home, move_to_start),
        (QtCore.Qt.Key.Key_End, move_to_end),
        *extrakeys,
    ):
        QtGui.QShortcut(QtGui.QKeySequence(key), window).activated.connect(callback)
