"""Make the menus and the actions of a viewer window, e.g. the copy, the save, and the export of the figure."""

import dataclasses as dc
import shlex
import sys
import typing as t
from functools import partial
from pathlib import Path
from types import MappingProxyType

from artistools.misc import import_optional
from artistools.misc import write_gif
from artistools.misc.remote import is_remote_path
from artistools.viewertools.application import activate_window
from artistools.viewertools.application import APPEARANCES
from artistools.viewertools.application import apply_appearance
from artistools.viewertools.application import get_bool_setting
from artistools.viewertools.application import get_float_setting
from artistools.viewertools.application import get_recent_models
from artistools.viewertools.application import get_recent_setting_key
from artistools.viewertools.application import get_settings
from artistools.viewertools.application import open_model_folder
from artistools.viewertools.application import show_wait_cursor
from artistools.viewertools.core import find_option_action
from artistools.viewertools.core import get_actions_by_flag
from artistools.viewertools.core import MAX_ANIMATION_FRAMES
from artistools.viewertools.core import remove_options
from artistools.viewertools.core import run_command_step
from artistools.viewertools.core import SIDEBAR_WIDTH
from artistools.viewertools.widgets import add_row
from artistools.viewertools.widgets import add_section
from artistools.viewertools.widgets import copy_text
from artistools.viewertools.widgets import get_menu_items
from artistools.viewertools.widgets import get_menu_shortcut_texts
from artistools.viewertools.widgets import make_fps_box
from artistools.viewertools.widgets import make_segmented_control
from artistools.viewertools.widgets import show_status_message
from artistools.viewertools.widgets import show_status_note
from artistools.viewertools.widgets import StatusBar

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import matplotlib.figure as mplfig
    from PySide6 import QtWidgets

    from artistools.commands import SuggestingArgumentParser
    from artistools.viewertools.core import PlotValues
    from artistools.viewertools.window import DrawQueue
    from artistools.viewertools.window import PlotViewer


# the help text of each menu item in the table of the keys
MENU_HELPTEXTS: t.Final = MappingProxyType({
    "Open Model…": "Open a model in a new window",
    "Reload Data": "Read the run again, e.g. while ARTIS writes more timesteps",
    "Save Figure…": "Save the figure in the format of the Figure section",
    "Export Animation…": "Save a GIF file of the steps of Play",
    "Undo": "Undo the last change",
    "Redo": "Redo the change that Undo removed",
    "Copy Figure": "Copy the figure in the format of the Figure section",
    "Copy Command": "Copy the command",
    "Copy Python": "Copy the Python code of the plot",
    "Play": "Play or pause",
    "Cancel Plot": "Stop the wait for a slow plot, and keep the plot on the screen",
    "Hide Sidebar": "Hide or show the sidebar",
})


def get_keyboard_help(keyrows: "Sequence[tuple[str, str]]", menuitems: "Collection[str]") -> str:
    """Return the table of the keys and the mouse actions of a viewer, with the shortcuts of the platform.

    keyrows gives the keys or the mouse action, and its help text, of each row of the viewer as HTML. menuitems gives
    the texts of the menu items of the viewer, as add_menus receives them, and the table gives the shortcut of each.
    """
    shortcuts = get_menu_shortcut_texts()
    # a viewer row replaces the menu row with the same keys, because it has more text, e.g. Space for Play
    viewerkeys = {keys for keys, _ in keyrows}
    rows = [
        *keyrows,
        # every window has the sidebar, and add_menus gives its item
        *(
            (f"<b>{shortcuts[text]}</b>", helptext)
            for text, helptext in MENU_HELPTEXTS.items()
            if (text in menuitems or text == "Hide Sidebar") and f"<b>{shortcuts[text]}</b>" not in viewerkeys
        ),
        (f"<b>{shortcuts['Keys and Mouse Actions']}</b>", "Show this list"),
    ]
    return (
        "<table>\n" + "".join(f"<tr><td>{keys}</td><td>{helptext}</td></tr>\n" for keys, helptext in rows) + "</table>"
    )


def add_menus(
    window: "QtWidgets.QMainWindow",
    callbacks: "Mapping[str, Callable[[], object]]",
    queue: "DrawQueue[t.Any]",
    playbutton: "QtWidgets.QAbstractButton | None",
    open_folder: "Callable[[str], object]",
) -> list[str]:
    """Add the menus File, Edit, View, Window, and Help, and return the text of each item.

    callbacks gives the function of each item by the text of get_menu_items. A viewer omits an item that it does not
    support, e.g. Reload Data. This function gives the items of the sidebar, the full screen, and the window. The
    queue gives Undo, Redo, and Cancel Plot, unless callbacks gives them, and playbutton gives Play. A viewer with no
    playbutton gets no Play item. File > Open Recent gives a recent model to open_folder.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    def cancel_plot() -> None:
        if playbutton is not None:
            playbutton.setChecked(False)
        queue.cancel()

    enabled: dict[str, Callable[[], bool]] = {
        "Undo": queue.can_undo,
        "Redo": queue.can_redo,
        "Cancel Plot": queue.is_busy,
    }
    titles: dict[str, Callable[[], str]] = {"Play": playbutton.text} if playbutton is not None else {}
    plotcallbacks: dict[str, Callable[[], object]] = {
        "Undo": queue.undo,
        "Redo": queue.redo,
        **({"Play": playbutton.toggle} if playbutton is not None else {}),
        "Cancel Plot": cancel_plot,
    }

    windowcallbacks: dict[str, Callable[[], object]] = {
        "Hide Sidebar": lambda: toggle_sidebar(window),
        "Enter Full Screen": lambda: window.showNormal() if window.isFullScreen() else window.showFullScreen(),
        "Minimize": window.showMinimized,
        "Zoom": lambda: window.showNormal() if window.isMaximized() else window.showMaximized(),
        "Settings…": show_settings_window,
        "artistools Help": lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(HELP_URL)),
        "About": lambda: show_about(window),
    }
    windowtitles: dict[str, Callable[[], str]] = {
        "Hide Sidebar": lambda: "Show Sidebar" if is_sidebar_hidden(window) else "Hide Sidebar",
        "Enter Full Screen": lambda: "Exit Full Screen" if window.isFullScreen() else "Enter Full Screen",
    }
    allcallbacks = {**windowcallbacks, **plotcallbacks, **callbacks}
    alltitles = {**windowtitles, **titles}
    menubar = window.menuBar()
    menus = {name: menubar.addMenu(name) for name in ("File", "Edit", "View", "Window", "Help")}
    actions: dict[str, QtGui.QAction] = {}
    for menuname, text, keys in get_menu_items():
        # macOS adds its own Enter Full Screen item to a menu with the title View
        if text not in allcallbacks or (text == "Enter Full Screen" and sys.platform == "darwin"):
            continue
        action = menus[menuname].addAction(text)
        action.setShortcut(keys)
        action.triggered.connect(allcallbacks[text])
        # the role of an application item puts Settings… in the menu of the application on macOS with this text.
        # The role of Preferences gave the old text "Preferences..."
        if text == "Settings…":
            action.setMenuRole(QtGui.QAction.MenuRole.ApplicationSpecificRole)
        elif text == "About":
            action.setMenuRole(QtGui.QAction.MenuRole.AboutRole)
        actions[text] = action
        if text == "Open Model…":
            add_recent_menu(menus["File"], open_folder)

    def update_items() -> None:
        for text, action in actions.items():
            if text in alltitles:
                action.setText(alltitles[text]())
            action.setEnabled(enabled[text]() if text in enabled else True)

    def enable_all() -> None:
        # a disabled item takes no shortcut, and the state of an item can change while the menu is closed
        for action in actions.values():
            action.setEnabled(True)

    windowmenu = menus["Window"]
    windowmenu.addSeparator()
    windowlistactions: list[QtGui.QAction] = []

    def update_window_list() -> None:
        """Show each window of the viewers at the end of the Window menu, with a mark at this window."""
        for action in windowlistactions:
            windowmenu.removeAction(action)
            action.deleteLater()
        windowlistactions.clear()
        for other in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(other, QtWidgets.QMainWindow) and other.isVisible():
                action = windowmenu.addAction(other.windowTitle())
                action.setCheckable(True)
                action.setChecked(other is window)
                action.triggered.connect(partial(activate_window, other))
                windowlistactions.append(action)

    for menu in menus.values():
        menu.aboutToShow.connect(update_items)
        menu.aboutToHide.connect(enable_all)
    windowmenu.aboutToShow.connect(update_window_list)
    return list(actions)


def add_recent_menu(filemenu: "QtWidgets.QMenu", open_folder: "Callable[[str], object]") -> None:
    """Add the submenu Open Recent, which shows the recent models when it opens, and Clear Menu at its end."""
    recentmenu = filemenu.addMenu("Open Recent")

    def show_recent_models() -> None:
        recentmenu.clear()
        for folder in get_recent_models():
            action = recentmenu.addAction(Path(folder).name)
            action.setToolTip(folder)
            # a test of a remote folder starts ssh, thus the menu enables a remote model with no test
            action.setEnabled(is_remote_path(folder) or Path(folder).is_dir())
            action.triggered.connect(partial(open_folder, folder))
        recentmenu.addSeparator()
        clearaction = recentmenu.addAction("Clear Menu")
        clearaction.setEnabled(bool(get_recent_models()))
        clearaction.triggered.connect(lambda: get_settings().remove(get_recent_setting_key()))

    recentmenu.aboutToShow.connect(show_recent_models)


# the documentation of artistools, which Help > artistools Help opens
HELP_URL: t.Final = "https://github.com/artis-mcrt/artistools#readme"


def show_about(parent: "QtWidgets.QWidget") -> None:
    """Show the About panel of the viewer: the name, the version of artistools, and the link to the source."""
    import importlib.metadata

    from PySide6 import QtCore
    from PySide6 import QtWidgets

    name = QtWidgets.QApplication.applicationDisplayName()
    version = importlib.metadata.version("artistools")
    about = QtWidgets.QMessageBox(parent)
    about.setWindowTitle(f"About {name}")
    about.setIconPixmap(QtWidgets.QApplication.windowIcon().pixmap(64, 64))
    # macOS shows the text in bold and the informative text in a small regular font, as an About panel of a Mac app
    about.setText(name)
    about.setInformativeText(
        f"artistools {version}<br>Qt {QtCore.qVersion()}, Python {sys.version.split()[0]}<br><br>"
        '<a href="https://github.com/artis-mcrt/artistools">github.com/artis-mcrt/artistools</a>'
    )
    about.exec()


def set_window_document(window: "QtWidgets.QMainWindow", folder: Path, title: str) -> None:
    """Give the window the title of its model and the folder of the model as its file.

    macOS then shows the icon of the folder in the title bar. A Command-click on the title shows the path, and a drag
    of the icon gives the folder to a different app.
    """
    window.setWindowTitle(title)
    # a model on a different host has no local folder, thus its window shows no folder icon
    window.setWindowFilePath("" if is_remote_path(folder) else str(folder.absolute()))


def show_settings_window() -> None:
    """Show the Settings window of the viewers, or bring it to the front if it is open.

    A change applies at once and the settings keep it, as in the Settings windows of macOS. Each viewer window
    receives a change of the colours of the plot through its property "settingshandler". The Settings window belongs
    to all the viewer windows, thus it has no parent, and it stays open when a viewer window closes.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    for widget in QtWidgets.QApplication.topLevelWidgets():
        if widget.objectName() == "settings" and widget.isVisible():
            activate_window(widget)
            return
    settings = get_settings()
    dialog = QtWidgets.QDialog()
    # Python deletes a window with no parent and no reference, thus the application holds one until the window closes
    if (app := QtWidgets.QApplication.instance()) is not None:
        app.setProperty("settingswindow", lambda: dialog)
        dialog.destroyed.connect(lambda: app.setProperty("settingswindow", None))
    dialog.setObjectName("settings")
    dialog.setWindowTitle("Settings")
    dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    form = QtWidgets.QFormLayout(dialog)
    # a Settings window of macOS has the size of its content
    form.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)

    appearancesegments = make_segmented_control(
        list(APPEARANCES),
        ["The appearance of macOS, light or dark", "A light window and plot", "A dark window and plot"],
    )
    appearance = str(settings.value("appearance", "System"))
    appearancesegments.setCurrentIndex(APPEARANCES.index(appearance) if appearance in APPEARANCES else 0)

    def on_appearance(index: int) -> None:
        settings.setValue("appearance", APPEARANCES[index])
        apply_appearance()

    appearancesegments.currentChanged.connect(on_appearance)
    form.addRow("Appearance:", appearancesegments)

    fpsbox = make_fps_box()
    fpsbox.setToolTip("The frames per second of Play in a new window")
    fpsbox.valueChanged.connect(partial(settings.setValue, "playfps"))
    form.addRow("Play in a new window [FPS]:", fpsbox)

    darkcheck = QtWidgets.QCheckBox("Show the plot in the colours of a dark appearance")
    darkcheck.setToolTip("A saved figure keeps its usual colours. Without this choice, the plot stays light")
    darkcheck.setChecked(get_bool_setting("darkplot", default=True))

    def on_dark(checked: bool) -> None:
        settings.setValue("darkplot", checked)
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if callable(handler := widget.property("settingshandler")):
                handler()

    darkcheck.toggled.connect(on_dark)
    form.addRow(darkcheck)

    reopencheck = QtWidgets.QCheckBox("Reopen the windows of the last session at the start")
    reopencheck.setToolTip("After Quit, the next start opens each window that was open, beside the new window")
    reopencheck.setChecked(get_bool_setting("reopenwindows", default=True))
    reopencheck.toggled.connect(partial(settings.setValue, "reopenwindows"))
    form.addRow(reopencheck)

    for flag, maximum, step in (("-figscale", 10.0, 0.1), ("-labelfontsize", 40.0, 1.0)):
        box = QtWidgets.QDoubleSpinBox()
        # the box shows the text "Default", which is wider than a number
        box.setMinimumWidth(100)
        box.setRange(0.0, maximum)
        box.setSingleStep(step)
        # the minimum of the box shows "Default", which gives no option
        box.setSpecialValueText("Default")
        box.setValue(get_float_setting(f"default{flag}", 0.0))
        box.setToolTip(f"A new window adds {flag} with this value if its command does not give {flag}")
        box.valueChanged.connect(partial(settings.setValue, f"default{flag}"))
        form.addRow(f"{flag} of a new window:", box)
    dialog.show()


def add_default_options(parser: "SuggestingArgumentParser", tokens: "Sequence[str]") -> list[str]:
    """Return the tokens with the options of the Settings window that the tokens do not give, e.g. -figscale.

    The options go before the other tokens, and an option that the parser of the command does not have gives nothing.
    """
    actions = get_actions_by_flag(parser)
    givendests = {action.dest for token in tokens if (action := find_option_action(parser, token)[0]) is not None}
    added: list[str] = []
    for flag in ("-figscale", "-labelfontsize"):
        value = get_float_setting(f"default{flag}", 0.0)
        if value > 0.0 and flag in actions and actions[flag].dest not in givendests:
            added += [flag, format(value, "g")]
    return [*added, *tokens]


def get_sidebar(window: "QtWidgets.QMainWindow") -> "QtWidgets.QWidget | None":
    """Return the sidebar of the window, which make_central_splitter puts at the right of the plot."""
    from PySide6 import QtWidgets

    splitter = window.centralWidget()
    return splitter.widget(1) if isinstance(splitter, QtWidgets.QSplitter) else None


def is_sidebar_hidden(window: "QtWidgets.QMainWindow") -> bool:
    """Return whether the user hid the sidebar, with the menu or with a drag of the handle to the edge."""
    sidebar = get_sidebar(window)
    return sidebar is None or sidebar.isHidden() or sidebar.width() == 0


def toggle_sidebar(window: "QtWidgets.QMainWindow") -> None:
    """Hide the sidebar, or show it with its first width."""
    from PySide6 import QtWidgets

    sidebar = get_sidebar(window)
    splitter = window.centralWidget()
    if sidebar is None or not isinstance(splitter, QtWidgets.QSplitter):
        return
    if not is_sidebar_hidden(window):
        sidebar.hide()
        return
    sidebar.show()
    # a drag to the edge gives the sidebar a width of zero, thus the sidebar takes its first width again
    if sidebar.width() == 0:
        total = sum(splitter.sizes())
        splitter.setSizes([max(total - SIDEBAR_WIDTH, 0), SIDEBAR_WIDTH])


# the background and the text colour of a dark plot while the palette of the window is still light
DARK_PLOT_COLOURS: t.Final = ("#323232", "#dfdfdf")


def get_dark_plot_colours() -> tuple[str, str] | None:
    """Return the background colour and the text colour of the window in Dark Mode, or None for a light plot.

    The window thread calls this function, and a render in the worker thread receives the result. The setting
    "darkplot" of the Settings window can keep the plot light.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    if not get_bool_setting("darkplot", default=True):
        return None
    if QtGui.QGuiApplication.styleHints().colorScheme() != QtCore.Qt.ColorScheme.Dark:
        return None
    palette = QtGui.QGuiApplication.palette()
    background = palette.color(QtGui.QPalette.ColorRole.Window)
    # Qt gives the signal of a new colour scheme before it gives the new palette
    if background.lightness() > 128:
        return DARK_PLOT_COLOURS
    return background.name(), palette.color(QtGui.QPalette.ColorRole.WindowText).name()


def copy_figure_of_command(
    queue: "DrawQueue[t.Any]",
    statusbar: StatusBar,
    commandmain: "Callable[..., None]",
    parser: "SuggestingArgumentParser",
    plottokens: "Sequence[str]",
    choice: tuple[str, int],
) -> None:
    """Put the figure of the command on the clipboard, in the format and the resolution of choice.

    The command draws the figure, as for a saved file. Thus the image has no empty margin, and it has the colours of
    the command: the colours of --darkmode or the usual colours, also when the window shows the plot in Dark Mode. The
    worker thread runs the command, thus the window accepts input.
    """
    suffix, dpi = choice
    tokens = [*remove_options(parser, plottokens, {"dpi"}), "-dpi", str(dpi)]
    files: list[bytes] = []

    def draw_file() -> str | None:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            filepath = Path(folder) / f"figure.{suffix}"
            commandmain(argsraw=[*tokens, "-o", str(filepath)])
            if not filepath.is_file():
                return f"The command wrote no {suffix.upper()} file"
            files.append(filepath.read_bytes())
        return None

    def show_result(message: str | None) -> None:
        if message is None and files:
            message = put_file_on_clipboard(files[0], suffix)
        if message is not None:
            show_status_message(statusbar, f"The viewer did not copy the figure: {message}", "")
        else:
            pixelsize = get_pixel_size_text(files[0], suffix)
            resolution = f", with {dpi} dpi{pixelsize}" if suffix == "png" else ""
            show_status_note(statusbar, f"Copied the figure as {suffix.upper()}{resolution}")

    if not queue.run_task(lambda: run_command_step(draw_file), "Copy of the figure in progress...", show_result):
        show_status_message(statusbar, "A different task is in progress. Copy the figure after it", "")


def get_pixel_size_text(data: bytes, suffix: str) -> str:
    """Return the size of a PNG file in pixels for the status bar, or "" for a different format.

    The size of the plot in the window and the resolution give the size, and the command crops the empty part of a
    margin.
    """
    from PySide6 import QtGui

    image = QtGui.QImage.fromData(data) if suffix == "png" else QtGui.QImage()
    return "" if image.isNull() else f" ({image.width()} \N{MULTIPLICATION SIGN} {image.height()} px)"


# the type of each file format on the clipboard of macOS, and on the clipboard of Linux and Windows
PASTEBOARD_TYPES: t.Final = MappingProxyType({"png": "public.png", "pdf": "com.adobe.pdf", "svg": "public.svg-image"})
CLIPBOARD_MIME_TYPES: t.Final = MappingProxyType({"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"})


def put_file_on_clipboard(data: bytes, suffix: str) -> str | None:
    """Put the content of a PNG, a PDF, or an SVG file on the clipboard. Return an error message, or None.

    On macOS, Qt gives an image to the clipboard only as TIFF. Qt gives a PDF or an SVG file a type of Qt that no
    other application reads. Thus AppKit gives the file the type of macOS there. A PNG file also goes on the clipboard
    as TIFF, for an application that reads no PNG. An SVG file also goes on the clipboard as text, e.g. for an editor
    of SVG code.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    if sys.platform == "darwin":
        try:
            appkit = import_optional("AppKit")
        except ModuleNotFoundError as exc:
            if suffix != "png":
                return str(exc)
        else:
            pasteboard = appkit.NSPasteboard.generalPasteboard()
            pasteboard.clearContents()
            nsdata = appkit.NSData.dataWithBytes_length_(data, len(data))
            pasteboard.setData_forType_(nsdata, PASTEBOARD_TYPES[suffix])
            if suffix == "png":
                pasteboard.setData_forType_(
                    appkit.NSBitmapImageRep.imageRepWithData_(nsdata).TIFFRepresentation(), "public.tiff"
                )
            elif suffix == "svg":
                pasteboard.setString_forType_(data.decode(), "public.utf8-plain-text")
            return None
    mimedata = QtCore.QMimeData()
    mimedata.setData(CLIPBOARD_MIME_TYPES[suffix], QtCore.QByteArray(data))
    if suffix == "png":
        mimedata.setImageData(QtGui.QImage.fromData(data))
    elif suffix == "svg":
        mimedata.setText(data.decode())
    QtWidgets.QApplication.clipboard().setMimeData(mimedata)
    return None


def follow_colour_scheme(
    window: "QtWidgets.QMainWindow", viewer: "PlotViewer[t.Any]", queue: "DrawQueue[t.Any]"
) -> None:
    """Draw the plot again with the colours of each new appearance, e.g. Dark Mode, or of a change in Settings.

    Qt gives the new palette after the signal, thus the plot waits until Qt has no other events. The Settings window
    reaches each window through its property "settingshandler".
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    def draw_with_new_colours() -> None:
        refresh_palette_style_sheets(window)
        viewer.darkcolours = get_dark_plot_colours()
        queue.redraw()

    def on_colour_scheme() -> None:
        QtCore.QTimer.singleShot(0, window, draw_with_new_colours)

    stylehints = QtGui.QGuiApplication.styleHints()
    window.setProperty("settingshandler", on_colour_scheme)
    stylehints.colorSchemeChanged.connect(on_colour_scheme)
    # the signal of the application stays after the window closes, thus the window removes its handler
    window.destroyed.connect(lambda: stylehints.colorSchemeChanged.disconnect(on_colour_scheme))


def refresh_palette_style_sheets(window: "QtWidgets.QWidget") -> None:
    """Apply again each style sheet of the window that gives a colour of the palette, e.g. palette(base).

    Qt reads a colour of the palette in a style sheet one time only. Without this, a chip of plotestimators keeps the
    white background of a light window in Dark Mode.
    """
    from PySide6 import QtWidgets

    for widget in [window, *window.findChildren(QtWidgets.QWidget)]:
        if "palette(" in (stylesheet := widget.styleSheet()):
            widget.setStyleSheet("")
            widget.setStyleSheet(stylesheet)


def show_figure_in_canvas(oldfig: "mplfig.Figure", newfig: "mplfig.Figure") -> tuple[float, float]:
    """Show a figure of the worker thread in the canvas of the window, and return the size of the figure in inches."""
    canvas = oldfig.canvas
    newfig.set_canvas(canvas)
    canvas.figure = newfig
    canvas.draw_idle()
    figwidth, figheight = newfig.get_size_inches()
    return float(figwidth), float(figheight)


# the Figure section selects PNG at the start, because each application can paste a PNG image
DEFAULT_FIGURE_FORMAT: t.Final = "png"


def get_figure_format() -> str:
    """Return the suffix of the format of the figure that the user selected last, e.g. "pdf"."""
    suffix = str(get_settings().value("figureformat", DEFAULT_FIGURE_FORMAT))
    return suffix if suffix in dict(EXPORT_FORMATS) else DEFAULT_FIGURE_FORMAT


def set_figure_format(suffix: str) -> None:
    """Keep the format of the figure, and show it in the Figure section of each window of the viewers."""
    from PySide6 import QtWidgets

    get_settings().setValue("figureformat", suffix)
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if callable(handler := widget.property("figureformathandler")):
            handler(suffix)


class FigureSection(t.NamedTuple):
    """The buttons of the Figure section, its Resolution box, which gives the -dpi of the command, and its grid."""

    copybutton: "QtWidgets.QPushButton"
    savebutton: "QtWidgets.QPushButton"
    dpibox: "QtWidgets.QSpinBox"
    # a viewer can add a row of its own below the first row, e.g. the figure scale
    grid: "QtWidgets.QGridLayout"


def add_figure_section(
    window: "QtWidgets.QMainWindow", panellayout: "QtWidgets.QVBoxLayout", dpi: int
) -> FigureSection:
    """Add the Figure section: the format of a copied or saved figure, its resolution for PNG, and two buttons.

    dpi is the resolution of the command, and the window connects the Resolution box to the -dpi of its values. The
    box shows only for a PNG file. A PDF or an SVG file takes the resolution for its raster parts, e.g. a colour
    image, and its lines and its text have no resolution. The figure has the size of the plot in the window.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    _, grid = add_section(panellayout, "Figure")
    formatbox = QtWidgets.QComboBox()
    for index, (suffix, tooltip) in enumerate(EXPORT_FORMATS):
        formatbox.addItem(suffix.upper(), suffix)
        formatbox.setItemData(index, tooltip, QtCore.Qt.ItemDataRole.ToolTipRole)
    formatbox.setToolTip("The format of Copy Figure and Save Figure…")
    resolutionlabel = QtWidgets.QLabel("Resolution:")
    dpibox = QtWidgets.QSpinBox()
    dpibox.setRange(10, 2400)
    dpibox.setSingleStep(50)
    dpibox.setValue(dpi)
    dpibox.setSuffix(" dpi")
    dpibox.setKeyboardTracking(False)
    dpibox.setToolTip("The resolution of a PNG file (-dpi)")
    shortcuts = get_menu_shortcut_texts()
    copybutton = QtWidgets.QPushButton("Copy")
    copybutton.setToolTip(f"Put the figure on the clipboard ({shortcuts['Copy Figure']})")
    # the Figure section tells a sighted user what the button copies, and a screen reader reads the whole name
    copybutton.setAccessibleName("Copy Figure")
    savebutton = QtWidgets.QPushButton("Save…")
    savebutton.setToolTip(f"Save the figure in a file ({shortcuts['Save Figure…']})")

    def show_format(suffix: str) -> None:
        # a format from a different window must not call set_figure_format again
        blocker = QtCore.QSignalBlocker(formatbox)
        formatbox.setCurrentIndex(max(formatbox.findData(suffix), 0))
        blocker.unblock()
        # make_row_layout puts the label and the box in one group widget, and a hidden group leaves no gap in the row
        if (resolutiongroup := dpibox.parentWidget()) is not None:
            resolutiongroup.setVisible(suffix == "png")
            # the row puts its groups on lines at a resize only, thus a narrow sidebar needs a new arrangement now
            if (row := resolutiongroup.parentWidget()) is not None:
                QtWidgets.QApplication.sendEvent(row, QtGui.QResizeEvent(row.size(), row.size()))

    def on_format(index: int) -> None:
        set_figure_format(str(formatbox.itemData(index)))

    add_row(grid, 0, [QtWidgets.QLabel("Format:"), formatbox, resolutionlabel, dpibox, copybutton, savebutton])
    show_format(get_figure_format())
    formatbox.currentIndexChanged.connect(on_format)
    window.setProperty("figureformathandler", show_format)
    return FigureSection(copybutton=copybutton, savebutton=savebutton, dpibox=dpibox, grid=grid)


# a GIF file shows on a screen, thus its frames take the resolution of a screen
ANIMATION_DPI: t.Final = 100

# the file formats of the Figure section, with the tooltip of each
EXPORT_FORMATS: t.Final = (
    ("pdf", "A vector file, for a paper. A colour image in it takes the resolution"),
    ("png", "An image with the resolution, e.g. for a slide"),
    ("svg", "A vector file, e.g. for a web page. A colour image in it takes the resolution"),
)


class ViewerCommand(t.NamedTuple):
    """The command of a viewer, and the function that gives the Python code of its plot."""

    # the name of the subcommand, e.g. "plotspectra"
    name: str
    main: "Callable[..., None]"
    parser: "SuggestingArgumentParser"
    get_python_code: "Callable[[], str]"


def export_animation(
    window: "QtWidgets.QWidget",
    queue: "DrawQueue[t.Any]",
    command: ViewerCommand,
    frames: "tuple[int, Callable[[int], list[str]]]",
    fps: float,
) -> None:
    """Save a GIF file of the steps of Play, with one run of the command for each frame.

    frames gives the count of the frames and a function that gives the command of a frame by its index. The worker
    thread makes the command of each frame, runs it, and joins the frames, thus the window accepts input. A plot
    against time can have a frame for each of 125 000 cells, and a list of their commands took seconds. The GIF shows
    each frame for 1/fps seconds, and each frame has the resolution of a screen.
    """
    import tempfile

    from PySide6 import QtWidgets

    statusbar = queue.statusbar
    framecount, get_frametokens = frames
    if framecount == 0:
        return
    if framecount > MAX_ANIMATION_FRAMES:
        answer = QtWidgets.QMessageBox.question(
            window,
            "Export Animation",
            f"The animation has {framecount} frames, and each frame runs the command. Continue?",
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
    filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        window, "Export the animation", str(Path.cwd() / f"{command.name}.gif"), "GIF (*.gif)"
    )
    if not filename:
        return
    if not Path(filename).suffix:
        filename += ".gif"

    def export() -> str | None:
        with tempfile.TemporaryDirectory() as folder:
            framepaths: list[Path] = []
            for index in range(framecount):
                # the default -dpi suits a printed page, e.g. 600 dpi, and it gave frames larger than a screen
                tokens = [*remove_options(command.parser, get_frametokens(index), {"dpi"}), "-dpi", str(ANIMATION_DPI)]
                framepath = Path(folder) / f"frame{index:04d}.png"
                command.main(argsraw=[*tokens, "-o", str(framepath)])
                if not framepath.is_file():
                    return f"The command wrote no frame for {shlex.join(tokens)}"
                framepaths.append(framepath)
            write_gif(filename, framepaths, duration=1000.0 / fps)
        return None

    def show_result(message: str | None) -> None:
        if message is not None:
            show_status_message(statusbar, f"The viewer did not export the animation: {message}", "")
        else:
            show_status_note(statusbar, f"Saved {filename}, with {framecount} frames")

    if not queue.run_task(
        lambda: run_command_step(export), f"Export of {framecount} frames in progress...", show_result
    ):
        show_status_message(statusbar, "A different task is in progress. Export the animation after it", "")


def save_figure_of_command(
    window: "QtWidgets.QWidget",
    statusbar: StatusBar,
    commandmain: "Callable[..., None]",
    commandname: str,
    plottokens: "Sequence[str]",
    parser: "SuggestingArgumentParser",
    choice: tuple[str, int],
) -> None:
    """Save the figure of the command in a file that the user selects, in the format and the resolution of choice.

    The figure comes from the command, thus the file is the same as the output of the command. The command reads a
    name with no suffix as a folder, thus the name takes the suffix of the format. The status bar shows the result.
    plottokens holds no -dpi. A PDF or an SVG file takes the resolution for its raster parts, e.g. a colour image.
    """
    from PySide6 import QtWidgets

    suffix, dpi = choice
    defaultdpi = parser.get_default("dpi")
    filename, _ = QtWidgets.QFileDialog.getSaveFileName(
        window, "Save the figure", str(Path.cwd() / f"{commandname}.{suffix}"), f"{suffix.upper()} (*.{suffix})"
    )
    if not filename:
        return
    if not Path(filename).suffix:
        filename += f".{suffix}"
    savetokens = [*plottokens, *([] if dpi == defaultdpi else ["-dpi", str(dpi)]), "-o", filename]

    def save() -> str | None:
        commandmain(argsraw=savetokens)
        return None

    with show_wait_cursor():
        message = run_command_step(save)
    if message is not None:
        show_status_message(statusbar, f"The command did not save the figure: {message}", "")
    elif not Path(filename).is_file():
        show_status_message(statusbar, f"The command wrote no file at {filename}. The terminal shows its output", "")
    else:
        print(shlex.join(["artistools", commandname, *savetokens]))
        show_status_note(statusbar, f"Saved {filename}{get_pixel_size_text(Path(filename).read_bytes(), suffix)}")


def open_model_window(
    window: "QtWidgets.QWidget",
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    windows: "list[QtWidgets.QMainWindow]",
) -> str | None:
    """Ask for the folder of a run, and open a new window for it. Return an error message if no window opened.

    A SystemExit in a Qt slot ends the process, thus an error of the new window stays in this window.
    """
    from PySide6 import QtWidgets

    # the dialog starts beside the model that the user opened last, where the other runs of a project are
    settings = get_settings()
    startfolder = settings.value("modelfolder", str(Path.cwd()))
    folder = QtWidgets.QFileDialog.getExistingDirectory(window, "Open the folder of an ARTIS run", str(startfolder))
    if not folder:
        return None
    return open_model_folder(folder, open_window, windows)


def add_window_actions[ValuesT: PlotValues](
    window: "QtWidgets.QMainWindow",
    windows: "list[QtWidgets.QMainWindow]",
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    queue: "DrawQueue[ValuesT]",
    command: ViewerCommand,
    figuresection: FigureSection,
    copybuttons: "tuple[QtWidgets.QPushButton, QtWidgets.QPushButton]",
    keyrows: "Sequence[tuple[str, str]]",
    playbutton: "QtWidgets.QAbstractButton | None",
    extracallbacks: "Mapping[str, Callable[[], object]] | None" = None,
) -> "Callable[[QtWidgets.QMenu], None]":
    """Give a window the actions that each viewer has: copy, save, open, help, and the menus.

    copybuttons are the Copy buttons of the command and of the Python code. The -dpi box of the Figure section sets
    the -dpi of the values of the queue. extracallbacks gives the menu items of one viewer, e.g. Reload Data. Return
    the function that adds the actions on the figure to a context menu of the plot.
    """
    from PySide6 import QtWidgets

    viewer, statusbar = queue.viewer, queue.statusbar
    defaultdpi: int = command.parser.get_default("dpi")
    callbacks = dict(extracallbacks or {})

    def get_figure_choice() -> tuple[str, int]:
        return get_figure_format(), viewer.values.dpi or defaultdpi

    def get_figure_tokens() -> list[str]:
        # the Figure section gives the resolution, thus the arguments of the plot have no -dpi
        return viewer.get_plot_tokens(dc.replace(viewer.values, dpi=None))

    def on_copy_figure() -> None:
        tokens = get_figure_tokens()
        copy_figure_of_command(queue, statusbar, command.main, command.parser, tokens, get_figure_choice())

    def on_save() -> None:
        tokens = get_figure_tokens()
        save_figure_of_command(
            window, statusbar, command.main, command.name, tokens, command.parser, get_figure_choice()
        )

    def on_copy() -> None:
        copy_text(viewer.get_command())
        show_status_note(statusbar, "Copied the command")

    def on_copy_python() -> None:
        copy_text(command.get_python_code())
        show_status_note(statusbar, "Copied the Python code")

    def on_resolution(resolution: int) -> None:
        queue.apply(dc.replace(viewer.values, dpi=None if resolution == defaultdpi else resolution))

    def on_open_model() -> None:
        if (message := open_model_window(window, open_window, windows)) is not None:
            queue.show_error(message)

    def on_open_recent(folder: str) -> None:
        if (message := open_model_folder(folder, open_window, windows)) is not None:
            queue.show_error(message)

    def on_help() -> None:
        QtWidgets.QMessageBox.information(window, "Keys and mouse actions", get_keyboard_help(keyrows, menutexts))

    menucallbacks: dict[str, Callable[[], object]] = {
        "Open Model…": on_open_model,
        "Save Figure…": on_save,
        "Close Window": window.close,
        "Copy Figure": on_copy_figure,
        "Copy Command": on_copy,
        "Copy Python": on_copy_python,
        "Keys and Mouse Actions": on_help,
        **callbacks,
    }
    menutexts = add_menus(window, menucallbacks, queue, playbutton, open_folder=on_open_recent)
    figuresection.copybutton.clicked.connect(on_copy_figure)
    figuresection.savebutton.clicked.connect(on_save)
    figuresection.dpibox.valueChanged.connect(on_resolution)
    commandcopybutton, pythoncopybutton = copybuttons
    commandcopybutton.clicked.connect(on_copy)
    pythoncopybutton.clicked.connect(on_copy_python)
    statusbar.helpbutton.clicked.connect(on_help)

    def add_figure_actions(menu: QtWidgets.QMenu) -> None:
        """Add the actions on the figure to a context menu, as the context menu of a Mac app gives them."""
        menu.addAction("Copy Figure").triggered.connect(on_copy_figure)
        menu.addAction("Save Figure…").triggered.connect(on_save)
        if (exportanimation := callbacks.get("Export Animation…")) is not None:
            menu.addAction("Export Animation…").triggered.connect(exportanimation)

    return add_figure_actions
