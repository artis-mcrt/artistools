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
# the kinds of the kind box: the direction bins, and the observers of the virtual packets. An average over the phi angle
# or the theta angle is a check box, because it changes the bins of the direction bins
DIRECTION_KINDS: t.Final = (
    ("bin", "Direction bins", "plotviewingangle"),
    ("vpkt", "Virtual packet observers", "plotvspecpol"),
)
# the kinds of direction that the kind "bin" of the kind box gives with the check boxes of the averages
BIN_KINDS: t.Final = frozenset({"bin", "phi", "theta"})
ALL_DIRECTIONS_BIN: t.Final = -1


class DirectionChoice(t.NamedTuple):
    """The viewing direction of the values of a window: the kind, the bins, and the unit of the angles."""

    kind: str
    bins: tuple[int, ...]
    usedegrees: bool


def get_direction_bins(runfolder: Path | str, kind: str, *, usedegrees: bool) -> list[tuple[int, str]]:
    """Return each bin of a kind of viewing direction with its label, and first the average over all the directions.

    The average is bin -1. All the directions alone give a plot with no direction option. A kind of the direction bins
    can also take the average together with its bins. The virtual packets have no average, thus all the directions
    give the plot of the real packets.
    """
    alltext = "All directions (real packets)" if kind == "vpkt" else "All directions"
    return [(ALL_DIRECTIONS_BIN, alltext), *get_direction_choices(Path(runfolder), kind, usedegrees=usedegrees)]


def get_direction_summary(choice: DirectionChoice, labels: "Mapping[int, str]") -> str:
    """Return the text of the button of the direction list, e.g. "All directions" or "3 directions: All, 0, 5"."""
    bins = choice.bins if choice.kind else (ALL_DIRECTIONS_BIN,)
    if len(bins) == 1:
        label = labels.get(bins[0], "")
        return label if bins[0] == ALL_DIRECTIONS_BIN else f"{bins[0]}: {label}"
    names = ["All" if dirbin == ALL_DIRECTIONS_BIN else str(dirbin) for dirbin in bins]
    shownnames = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
    return f"{len(bins)} directions: {shownnames}"


def get_new_direction_choice(
    kind: str, bins: tuple[int, ...], newbins: tuple[int, ...], *, usedegrees: bool, onebin: bool
) -> DirectionChoice:
    """Return the choice of a list of checked bins, which can hold the average over all the directions (bin -1).

    newbins gives the bins that the last click checked. The average alone gives no direction option. An observer of
    the virtual packets has no average, thus the average replaces the observers, and an observer replaces the
    average. A plot of one bin replaces the previous bin with the bin of the click.
    """
    if onebin and newbins:
        bins = newbins[:1]
    if kind == "vpkt" and ALL_DIRECTIONS_BIN in bins:
        bins = (
            (ALL_DIRECTIONS_BIN,)
            if ALL_DIRECTIONS_BIN in newbins
            else tuple(dirbin for dirbin in bins if dirbin != ALL_DIRECTIONS_BIN)
        )
    if bins == (ALL_DIRECTIONS_BIN,):
        return DirectionChoice(kind="", bins=(), usedegrees=usedegrees)
    return DirectionChoice(kind=kind, bins=bins, usedegrees=usedegrees)


def add_direction_section(
    panellayout: "QtWidgets.QVBoxLayout",
    helptexts: "Mapping[str, str]",
    runfolder: Path | str,
    get_choice: "Callable[[], tuple[DirectionChoice, bool]]",
    on_change: "Callable[[DirectionChoice], None]",
    show_error: "Callable[[str], None]",
    has_direction_data: "Callable[[Path | str], bool]",
) -> "tuple[Callable[[], None], Callable[[Path | str], None]]":
    """Add the section of the viewing direction: the kind, the averages, --usedegrees, and a drop-down list of the bins.

    The first item of the list is all the directions, and the check boxes of the bins follow it. get_choice gives the
    choice of the values of the window, and whether the plot draws one bin, e.g. an emission plot. Such a plot has a
    radio button for each bin, and a click on a bin replaces the bin. on_change receives each new choice. A new kind
    keeps each bin that the kind also has. has_direction_data tells whether a run gives the plot of a direction bin. A
    run with no such data has only all the directions, thus the averages and the bins are disabled.

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
    averagechecks = {
        "phi": QtWidgets.QCheckBox("--average_over_phi_angle"),
        "theta": QtWidgets.QCheckBox("--average_over_theta_angle"),
    }
    add_row(directiongrid, 1, list(averagechecks.values()))
    # the list of a run can hold 100 bins, thus a button opens it as a drop-down list, and the button gives the choice
    binbutton = QtWidgets.QPushButton()
    binbutton.setToolTip(
        "The directions of the plot: all the directions, the direction bins, or the observers of the virtual packets."
        " A plot that draws one bin, e.g. an emission plot, shows a radio button for each bin"
    )
    binmenu = QtWidgets.QMenu(binbutton)
    binbox = QtWidgets.QScrollArea()
    binbox.setWidgetResizable(True)
    binbox.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    # a check box in a widget of a menu takes its click, thus the menu stays open while the user checks several bins
    binaction = QtWidgets.QWidgetAction(binmenu)
    binaction.setDefaultWidget(binbox)
    binmenu.addAction(binaction)
    binbutton.setMenu(binmenu)
    directiongrid.addWidget(binbutton, 2, 0, 1, -1)
    binchecks: dict[int, QtWidgets.QAbstractButton] = {}
    # the labels of the bins come from the files of the run, thus the section reads them one time for each kind
    binlabels: dict[tuple[str, bool], list[tuple[int, str]]] = {}
    shownlist: tuple[str, bool, bool] | None = None
    run: list[Path | str] = [runfolder]
    hasdirections = [False]

    def get_bins(kind: str, usedegrees: bool) -> list[tuple[int, str]]:
        if (kind, usedegrees) not in binlabels:
            binlabels[kind, usedegrees] = get_direction_bins(run[0], kind, usedegrees=usedegrees)
        return binlabels[kind, usedegrees]

    def set_run(newrunfolder: Path | str) -> None:
        nonlocal shownlist
        run[0], shownlist = newrunfolder, None
        binlabels.clear()
        hasdirections[0] = has_direction_data(newrunfolder)
        for averagekind, check in averagechecks.items():
            check.setEnabled(hasdirections[0])
            check.setToolTip(
                helptexts.get(f"average_over_{averagekind}_angle", "")
                if hasdirections[0]
                else "The run gives no plot of a direction bin, thus it has no average over an angle"
            )
        kinds = get_direction_kinds(Path(newrunfolder))
        with QtCore.QSignalBlocker(kindbox):
            kindbox.clear()
            for kind, text, dest in DIRECTION_KINDS:
                if kind in kinds:
                    kindbox.addItem(text, kind)
                    kindbox.setItemData(
                        kindbox.count() - 1, f"{helptexts.get(dest, '')} (-{dest})", QtCore.Qt.ItemDataRole.ToolTipRole
                    )
        # a run with no virtual packets has one kind, thus the box gives no choice
        kindbox.setVisible(kindbox.count() > 1)

    def get_shown_kind() -> str:
        """Return the kind of the list: the kind of the kind box, with the average that a check box selects."""
        boxkind: str = kindbox.currentData() or "bin"
        if boxkind != "bin":
            return boxkind
        return next((kind for kind, check in averagechecks.items() if check.isChecked()), "bin")

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
            text = label if dirbin == ALL_DIRECTIONS_BIN else f"{dirbin}: {label}"
            check = QtWidgets.QRadioButton(text) if onebin else QtWidgets.QCheckBox(text)
            check.setEnabled(dirbin == ALL_DIRECTIONS_BIN or hasdirections[0])
            # a click on a radio button also clears the previous button, thus the toggled signal calls the handler two times
            check.clicked.connect(on_bin_click)
            checklayout.addWidget(check)
            binchecks[dirbin] = check
        checklayout.addStretch(1)
        # the list shows up to 12 bins, and a longer list scrolls
        shownbins = min(max(len(binchecks), 1), 12)
        lineheight = max((check.sizeHint().height() for check in binchecks.values()), default=20)
        binbox.setFixedHeight(shownbins * (lineheight + 2) + 10)
        binbox.setWidget(checklist)
        shownlist = (kind, usedegrees, onebin)
        return True

    def show() -> None:
        choice, onebin = get_choice()
        # all the directions alone keep the kind box and the averages of the last kind, thus a new bin returns to them
        if choice.kind:
            boxkind = "bin" if choice.kind in BIN_KINDS else choice.kind
            with QtCore.QSignalBlocker(kindbox):
                kindbox.setCurrentIndex(max(kindbox.findData(boxkind), 0))
            for averagekind, check in averagechecks.items():
                with QtCore.QSignalBlocker(check):
                    check.setChecked(choice.kind == averagekind)
        with QtCore.QSignalBlocker(usedegreescheck):
            usedegreescheck.setChecked(choice.usedegrees)
        listkind = get_shown_kind()
        usedegreescheck.setEnabled(bool(choice.kind))
        isnewlist = show_bins(listkind, choice.usedegrees, onebin)
        checkedbins = choice.bins if choice.kind else (ALL_DIRECTIONS_BIN,)
        for dirbin, check in binchecks.items():
            with QtCore.QSignalBlocker(check):
                check.setChecked(dirbin in checkedbins)
        labels = dict(get_bins(listkind, choice.usedegrees))
        summary = get_direction_summary(choice, labels)
        # a long label of a bin must not widen the sidebar, thus the button shows the start and the end of the text
        binbutton.setText(binbutton.fontMetrics().elidedText(summary, QtCore.Qt.TextElideMode.ElideMiddle, 320))
        binbutton.setToolTip(summary)
        # a new list scrolls to the first checked bin, which can be far down a list of 100 bins
        if isnewlist and (firstcheck := binchecks.get(checkedbins[0])):
            QtCore.QTimer.singleShot(0, binbox, partial(binbox.ensureWidgetVisible, firstcheck))

    def on_show_menu() -> None:
        binbox.setFixedWidth(max(binbutton.width(), 280))
        choice, _ = get_choice()
        checkedbins = choice.bins if choice.kind else (ALL_DIRECTIONS_BIN,)
        if firstcheck := binchecks.get(checkedbins[0]):
            QtCore.QTimer.singleShot(0, binbox, partial(binbox.ensureWidgetVisible, firstcheck))

    def on_bin_click() -> None:
        current, onebin = get_choice()
        bins = tuple(dirbin for dirbin, check in binchecks.items() if check.isChecked())
        if not bins:
            show_error("The plot needs one direction at least")
            show()
            return
        currentbins = current.bins if current.kind else (ALL_DIRECTIONS_BIN,)
        newbins = tuple(dirbin for dirbin in bins if dirbin not in currentbins)
        choice = get_new_direction_choice(
            get_shown_kind(), bins, newbins, usedegrees=usedegreescheck.isChecked(), onebin=onebin
        )
        # a radio button selects one bin, thus the list closes after the click
        if onebin:
            binmenu.close()
        on_change(choice)

    def on_kind() -> None:
        """Apply a new kind, a new average, or a new unit of the angles, and keep each bin that the new kind has."""
        current, onebin = get_choice()
        kind = get_shown_kind()
        usedegrees = usedegreescheck.isChecked()
        # all the directions alone give no direction option, thus a new kind or average changes only the list
        if not current.kind:
            show()
            return
        kindbins = [dirbin for dirbin, _ in get_bins(kind, usedegrees)]
        bins = tuple(dirbin for dirbin in current.bins if dirbin in kindbins) or tuple(kindbins[1:2] or kindbins[:1])
        on_change(get_new_direction_choice(kind, bins, (), usedegrees=usedegrees, onebin=onebin))

    def on_average(averagekind: str, checked: bool) -> None:
        # the averages over the two angles exclude each other, and an average applies to the direction bins
        if checked:
            for otherkind, check in averagechecks.items():
                if otherkind != averagekind:
                    with QtCore.QSignalBlocker(check):
                        check.setChecked(False)
            with QtCore.QSignalBlocker(kindbox):
                kindbox.setCurrentIndex(max(kindbox.findData("bin"), 0))
        on_kind()

    kindbox.currentIndexChanged.connect(on_kind)
    usedegreescheck.toggled.connect(on_kind)
    for averagekind, check in averagechecks.items():
        check.toggled.connect(partial(on_average, averagekind))
    binmenu.aboutToShow.connect(on_show_menu)
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
