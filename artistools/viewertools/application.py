"""Start the Qt application, and open, remember, and reopen the windows of the viewers."""

import contextlib
import json
import sys
import threading
import time
import typing as t
import weakref
from pathlib import Path

import numpy as np

from artistools.misc import exit_with_error
from artistools.misc import import_optional
from artistools.misc import print_error
from artistools.misc.fileio import resolve_modelpath
from artistools.viewertools.core import is_flag
from artistools.viewertools.core import MACOS_BUNDLE_VARIABLE
from artistools.viewertools.core import run_command_step
from artistools.viewertools.core import SERIES_STYLE_FLAGS
from artistools.viewertools.core import ThreadOutput

if t.TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Mapping
    from collections.abc import Sequence

    import numpy.typing as npt
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets


def make_icon_pixmap(size: int, curve: "npt.NDArray[np.float64]") -> "QtGui.QPixmap":
    """Return a pixmap of the icon of a viewer: a curve over a dark square.

    curve gives the height of the curve from the top of the icon, as a part of its size. The window gives the icon
    to the Dock.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setBrush(QtGui.QColor("#1b2a41"))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QtCore.QRectF(0, 0, size, size), size * 0.22, size * 0.22)
    path = QtGui.QPainterPath()
    xvalues = np.linspace(0.1, 0.9, len(curve))
    path.moveTo(float(xvalues[0]) * size, float(curve[0]) * size)
    for xvalue, yvalue in zip(xvalues[1:].tolist(), curve[1:].tolist(), strict=True):
        path.lineTo(xvalue * size, yvalue * size)
    painter.setPen(QtGui.QPen(QtGui.QColor("#f5a623"), size * 0.05))
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.end()
    return pixmap


# the tool of macOS that reads the Info.plist of an application bundle again
LSREGISTER: t.Final = Path(
    "/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A"
    "/Support/lsregister"
)


def get_macos_bundle_executable(applicationname: str, documenttypes: "Sequence[str]") -> Path:
    """Return the Python executable in the application bundle of the viewer.

    Make the bundle if it does not exist. The bundle holds a hard link to the Python executable, thus it uses almost
    no disk space. On a different volume, it holds a copy. If the Python executable changes, this function replaces
    the link. documenttypes gives the uniform type identifiers that the Dock icon accepts, e.g. "public.folder".
    """
    import filecmp
    import os
    import plistlib
    import shutil
    import subprocess  # ruff:ignore[suspicious-subprocess-import]

    baseexecutable = Path(sys.executable).resolve()
    contents = Path.home() / "Library" / "Caches" / "artistools" / f"{applicationname}.app" / "Contents"
    executable = contents / "MacOS" / baseexecutable.name
    # a copy on a different volume is not the same file, thus the test also accepts a copy with the same contents
    if not (executable.exists() and filecmp.cmp(executable, baseexecutable, shallow=True)):
        executable.parent.mkdir(parents=True, exist_ok=True)
        # two viewers can make the bundle at the same time, thus each file receives its final name in one step
        tmpexecutable = executable.with_name(f"{executable.name}.{os.getpid()}.tmp")
        try:
            tmpexecutable.hardlink_to(baseexecutable)
        except OSError:
            # a hard link must be on the same volume as its target
            shutil.copy2(baseexecutable, tmpexecutable)
        tmpexecutable.replace(executable)

    info = {
        "CFBundleName": applicationname,
        "CFBundleDisplayName": applicationname,
        "CFBundleIdentifier": f"io.github.artis-mcrt.artistools.{applicationname.rsplit(maxsplit=1)[-1]}",
        "CFBundleExecutable": executable.name,
        "CFBundlePackageType": "APPL",
        "NSHighResolutionCapable": True,
        # a folder or a file that the user drops on the Dock icon comes to the viewer as a QFileOpenEvent
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "ARTIS data",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": list(documenttypes),
            }
        ],
    }
    infopath = contents / "Info.plist"
    infobytes = plistlib.dumps(info)
    if not infopath.is_file() or infopath.read_bytes() != infobytes:
        tmpinfo = contents / f"Info.plist.{os.getpid()}.tmp"
        tmpinfo.write_bytes(infobytes)
        tmpinfo.replace(infopath)
        # macOS keeps the old Info.plist of a bundle until lsregister reads the bundle again
        if LSREGISTER.is_file():
            subprocess.run(  # ruff:ignore[subprocess-without-shell-equals-true]
                [LSREGISTER, "-f", contents.parent], check=False, capture_output=True
            )
    return executable


def relaunch_in_macos_bundle(applicationname: str, documenttypes: "Sequence[str]") -> None:
    """Run the command again from an application bundle, which gives its name to the Dock and to the menu bar.

    The Dock gives a process outside a bundle the file name of its executable, e.g. "python3.14". A process cannot
    change that name after it starts. The new process finds the packages of the virtual environment through
    __PYVENV_LAUNCHER__.
    """
    import os
    import sysconfig

    # the new process runs sys.orig_argv again, thus a call from Python code, e.g. in a notebook, continues here.
    # A framework build starts Python.app, which names each process "Python", thus a bundle has no effect
    if (
        os.environ.get(MACOS_BUNDLE_VARIABLE)
        or "--interactive" not in sys.orig_argv
        or sysconfig.get_config_var("PYTHONFRAMEWORK")
    ):
        return

    try:
        executable = get_macos_bundle_executable(applicationname, documenttypes)
    except OSError:
        # the viewer can open without the bundle, and the Dock then gives the name of the executable
        return

    sys.stdout.flush()
    sys.stderr.flush()
    environment = os.environ | {MACOS_BUNDLE_VARIABLE: "1", "__PYVENV_LAUNCHER__": sys.executable}
    argv = [str(executable), *sys.orig_argv[1:]]
    os.execve(executable, argv, environment)  # ruff:ignore[start-process-with-no-shell]


def serialise_mathtext_parser() -> None:
    """Let one thread at a time parse mathtext, e.g. a tick label or a label of the legend.

    matplotlib keeps one parser for all threads, and pyparsing keeps a packrat cache in it. The worker thread lays out
    the text of a new plot while the window thread draws the old plot. A parse in both threads then corrupted the
    parser, and it rejected valid text, e.g. the tick label 0.8 of a log axis. A parse takes about 1 ms, thus
    the window thread waits for one parse and not for the whole plot.
    """
    import matplotlib.mathtext as mplmathtext

    parse = mplmathtext.MathTextParser.parse
    # a second window must not wrap the parse again. matplotlib's own decorators also set __wrapped__, thus the test
    # reads the name of this wrapper
    if getattr(parse, "__name__", "") == "serialised_parse":
        return
    lock = threading.Lock()

    # the wrapper passes each argument on, thus a new parameter of a later matplotlib also works. Python 3.13 evaluates
    # the annotations of a nested function at once, and MathTextParser takes no subscript at run time
    def serialised_parse(*args: t.Any, **kwargs: t.Any) -> t.Any:
        with lock:
            return parse(*args, **kwargs)

    mplmathtext.MathTextParser.parse = serialised_parse


# the Qt platform plugins of Linux that show a window on an X11 display or on a Wayland display
DISPLAY_PLATFORMS: t.Final = ("xcb", "wayland")


def needs_missing_display(environment: "Mapping[str, str]") -> bool:
    """Return True if the Qt platform of the environment needs a display, and the environment gives no display.

    QT_QPA_PLATFORM can select a platform that needs no display, e.g. offscreen, vnc, or eglfs. Its value is a list
    of platforms with ";" between them, and Qt uses the first platform that it can load. Each platform can have
    options after a ":".
    """
    if environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY"):
        return False
    platforms = [entry.partition(":")[0].strip() for entry in environment.get("QT_QPA_PLATFORM", "").split(";")]
    return all(not platform or platform.startswith(DISPLAY_PLATFORMS) for platform in platforms)


def start_application(
    applicationname: str, iconcurve: "npt.NDArray[np.float64]", documenttypes: "Sequence[str]" = ("public.folder",)
) -> "QtWidgets.QApplication":
    """Return the Qt application of a viewer, with the name and the icon of the viewer.

    The window is a Qt window with native controls. The Qt canvas of matplotlib draws at the pixel ratio
    of the screen, thus the plot has the full resolution of a Retina display. documenttypes gives the types of the
    items that the Dock icon accepts on macOS.
    """
    import os

    if sys.platform == "darwin":
        relaunch_in_macos_bundle(applicationname, documenttypes)

    import_optional("PySide6.QtWidgets")
    import matplotlib.pyplot as plt
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    # Qt stops the process with no Python error when it cannot find a display
    if sys.platform == "linux" and needs_missing_display(os.environ):
        exit_with_error(
            "--interactive needs a window, and this computer has no display",
            "Run the command on a computer with a display, e.g. with ssh -X",
        )

    # the Save command runs the command, which makes a pyplot figure. A pyplot window must not open beside the viewer
    plt.switch_backend("agg")
    serialise_mathtext_parser()
    # a worker thread draws each plot and hides its output, and the window thread still prints to the terminal
    if not isinstance(sys.stdout, ThreadOutput):
        sys.stdout = ThreadOutput(sys.stdout)
    if not isinstance(sys.stderr, ThreadOutput):
        sys.stderr = ThreadOutput(sys.stderr)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert isinstance(app, QtWidgets.QApplication)
    app.setOrganizationName("artistools")
    app.setApplicationName("artistools")
    app.setApplicationDisplayName(applicationname)
    app.setWindowIcon(QtGui.QIcon(make_icon_pixmap(512, iconcurve)))

    arrowkeys = {
        QtCore.Qt.Key.Key_Left,
        QtCore.Qt.Key.Key_Right,
        QtCore.Qt.Key.Key_Up,
        QtCore.Qt.Key.Key_Down,
        QtCore.Qt.Key.Key_Home,
        QtCore.Qt.Key.Key_End,
        QtCore.Qt.Key.Key_PageUp,
        QtCore.Qt.Key.Key_PageDown,
    }

    class ApplicationFilter(QtCore.QObject):
        """Handle the events of the application that the windows need, in one filter.

        Qt calls an application filter for each event of each object. A worker thread holds the GIL during a plot,
        and each call of Python then waits for it. Thus the application has one filter and not one for each task:

        - give a key to the widget with the focus when that widget uses the key, and not to a window shortcut;
        - keep the text field that the user confirmed last, for the mark of a rejected change;
        - record the time of a quit;
        - give a path from the Dock icon to the handler of handle_file_open_events.
        """

        @t.override
        def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
            eventtype = event.type()
            if eventtype == QtCore.QEvent.Type.ShortcutOverride:
                if isinstance(event, QtGui.QKeyEvent) and widget_uses_key(event):
                    event.accept()
                    return True
            elif eventtype in {QtCore.QEvent.Type.KeyPress, QtCore.QEvent.Type.FocusOut}:
                keep_edited_field(watched, event)
            elif eventtype == QtCore.QEvent.Type.Quit:
                app.setProperty("quittime", time.monotonic())
            elif eventtype == QtCore.QEvent.Type.FileOpen:
                handler = app.property("fileopenhandler")
                if isinstance(event, QtGui.QFileOpenEvent) and callable(handler):
                    handler(event.file())
                    return True
            return super().eventFilter(watched, event)

    def widget_uses_key(event: QtGui.QKeyEvent) -> bool:
        """Return True if the widget with the focus uses the key, e.g. the Up key of a spin box.

        These widgets use the keys, but the shortcuts of the window took the keys from them:

        - a spin box;
        - a combo box;
        - a slider;
        - a list;
        - a popup, e.g. the list of names of a completer;
        - a text field or a text box, which uses the arrow keys, Home, End, and the page keys;
        - a button, which uses only the space key;
        - a text field or a text box with selected text, which uses the Copy key.

        For example, the Up key in -maxseriescount or in the field of -xmin made the time range wider, and the Copy
        key in the Command box copied the figure.
        """
        focuswidget = QtWidgets.QApplication.focusWidget()
        # the field of a completer keeps the focus while its popup shows, and the popup takes the keys
        popupshows = QtWidgets.QApplication.activePopupWidget() is not None
        isselector = isinstance(
            focuswidget,
            QtWidgets.QAbstractSpinBox | QtWidgets.QComboBox | QtWidgets.QAbstractSlider | QtWidgets.QAbstractItemView,
        )
        # a text field moves its cursor with the arrow keys, and a text box also scrolls with them
        istext = isinstance(focuswidget, QtWidgets.QLineEdit | QtWidgets.QPlainTextEdit | QtWidgets.QTextEdit)
        usesarrows = popupshows or isselector or istext
        usesspace = popupshows or isselector or isinstance(focuswidget, QtWidgets.QAbstractButton)
        hastextselection = (
            isinstance(focuswidget, QtWidgets.QPlainTextEdit | QtWidgets.QTextEdit)
            and focuswidget.textCursor().hasSelection()
        ) or (isinstance(focuswidget, QtWidgets.QLineEdit) and focuswidget.hasSelectedText())
        key = event.key()
        keyowners = (
            (usesarrows, key in arrowkeys),
            (usesspace, key == QtCore.Qt.Key.Key_Space),
            (hastextselection, event.matches(QtGui.QKeySequence.StandardKey.Copy)),
        )
        return any(widgetuses and iskey for widgetuses, iskey in keyowners)

    def keep_edited_field(watched: QtCore.QObject, event: QtCore.QEvent) -> None:
        """Keep the text field that the user confirmed last, and the time, in two properties of its window.

        A plot that rejects the change of the field then marks the field, as a form of macOS does. The property holds
        a weak reference, because a QObject in a property is a raw pointer. A read of that pointer after Qt deleted
        the field crashed the process.
        """
        if not isinstance(watched, QtWidgets.QLineEdit):
            return
        isreturn = isinstance(event, QtGui.QKeyEvent) and event.key() in {
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        }
        confirms = isreturn if event.type() == QtCore.QEvent.Type.KeyPress else watched.isModified()
        if confirms:
            window = watched.window()
            window.setProperty("lasteditedfield", weakref.ref(watched))
            window.setProperty("lastedittime", time.monotonic())

    app.installEventFilter(ApplicationFilter(app))
    apply_appearance()
    return app


# a change that comes less than this time after the user confirmed a text field belongs to that field
EDIT_SECONDS: t.Final = 1.0


def get_edited_field(window: "QtCore.QObject") -> "QtWidgets.QLineEdit | None":
    """Return the text field that the user confirmed just now in the window, or None."""
    edittime = window.property("lastedittime")
    # each change of the user calls this function, thus a change with no recent edit returns before the import
    if not isinstance(edittime, float) or time.monotonic() - edittime >= EDIT_SECONDS:
        return None
    from PySide6 import QtWidgets

    fieldref = window.property("lasteditedfield")
    field = fieldref() if isinstance(fieldref, weakref.ref) else None
    return field if isinstance(field, QtWidgets.QLineEdit) and is_live(field) else None


def is_live(qobject: "QtCore.QObject") -> bool:
    """Return True if Qt did not delete the object, e.g. a field of a subplot card that show_subplots replaced."""
    import shiboken6

    return shiboken6.isValid(qobject)


def mark_field_error(field: "QtWidgets.QLineEdit", message: str) -> None:
    """Give the field a red border and the message beside it, as a form of macOS does."""
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    field.setStyleSheet("QLineEdit { border: 2px solid firebrick; border-radius: 3px; }")
    QtWidgets.QToolTip.showText(field.mapToGlobal(QtCore.QPoint(0, field.height())), message, field)


def clear_field_error(field: "QtWidgets.QLineEdit") -> None:
    """Remove the mark of mark_field_error, unless Qt deleted the field."""
    if is_live(field):
        field.setStyleSheet("")


def get_settings() -> "QtCore.QSettings":
    """Return the settings of the viewers, which keep the geometry of each window and the last model folder.

    The settings take the organisation and the name of the application, thus a test can send them to a folder with
    QSettings.setPath.
    """
    from PySide6 import QtCore

    return QtCore.QSettings()


def get_float_setting(key: str, default: float) -> float:
    """Return a number of the settings, or the default if the settings do not hold one."""
    value = get_settings().value(key, defaultValue=default, type=float)
    return value if isinstance(value, float) else default


def get_list_setting(key: str) -> list[str]:
    """Return a list of texts of the settings, or an empty list if the settings do not hold one."""
    value = get_settings().value(key, [])
    # the INI format of QSettings gives a list of one item as a string
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value] if isinstance(value, list) else []


def get_bool_setting(key: str, *, default: bool) -> bool:
    """Return a choice of the settings, or the default if the settings do not hold one."""
    value = get_settings().value(key, defaultValue=default, type=bool)
    return value if isinstance(value, bool) else default


def make_window(applicationname: str) -> "QtWidgets.QMainWindow":
    """Return the window of a viewer, which keeps its geometry and the sizes of its splitter when it closes.

    show_window restores them for the next window of the same viewer.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    class ViewerWindow(QtWidgets.QMainWindow):
        """A window that writes its geometry and the state of its splitter to the settings when it closes.

        The window also takes the folders and the files that the user drops on it. It gives their paths to the
        function in its property "drophandler", which set_drop_handler sets.
        """

        def __init__(self, applicationname: str) -> None:
            super().__init__()
            # the name of the object gives the keys of the settings of the window
            self.setObjectName(applicationname)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
            self.setAcceptDrops(True)

        @t.override
        def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
            if get_dropped_paths(event.mimeData()) and callable(self.property("drophandler")):
                event.acceptProposedAction()

        @t.override
        def dropEvent(self, event: QtGui.QDropEvent) -> None:
            handler = self.property("drophandler")
            if (paths := get_dropped_paths(event.mimeData())) and callable(handler):
                event.acceptProposedAction()
                handler(paths)

        @t.override
        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            geometrykey, splitterkey = get_window_setting_keys(self)
            settings = get_settings()
            # a window that keeps its first size saves no size, thus a different first size of a new version applies
            if self.property("firstsize") == self.size():
                settings.remove(geometrykey)
            else:
                settings.setValue(geometrykey, self.saveGeometry())
            app = QtWidgets.QApplication.instance()
            isquitting = app is not None and app.property("quittime") is not None
            if isquitting and callable(get_tokens := self.property("sessiontokens")):
                add_session_window(get_tokens())
            splitter = self.centralWidget()
            if isinstance(splitter, QtWidgets.QSplitter):
                settings.setValue(splitterkey, splitter.saveState())
            super().closeEvent(event)

    return ViewerWindow(applicationname)


def get_dropped_paths(mimedata: "QtCore.QMimeData") -> list[str]:
    """Return the local paths of the folders and the files of a drop, e.g. from the Finder."""
    return [url.toLocalFile() for url in mimedata.urls() if url.isLocalFile()] if mimedata.hasUrls() else []


def set_drop_handler(window: "QtWidgets.QMainWindow", handler: "Callable[[list[str]], None]") -> None:
    """Give the window the function that receives the paths that the user drops on the window."""
    window.setProperty("drophandler", handler)


def handle_file_open_events(app: "QtWidgets.QApplication", open_folder: "Callable[[str], None]") -> None:
    """Open a folder from the Dock icon in a new window, and give a file to the active window.

    macOS gives such an item to the application as a QFileOpenEvent, and not to a window. At the start, macOS also
    gives each path of the command line as such an event, and the first window already shows those paths. Thus the
    filter ignores the first event of each of those paths. A later drop of the same path opens it as usual.
    """
    from PySide6 import QtWidgets

    launchpaths = {str(Path(argument).absolute()) for argument in sys.orig_argv}

    def open_path(path: str) -> None:
        if (absolutepath := str(Path(path).absolute())) in launchpaths:
            launchpaths.remove(absolutepath)
            return
        activewindow = QtWidgets.QApplication.activeWindow()
        handler = activewindow.property("drophandler") if activewindow is not None else None
        if Path(path).is_dir() or not callable(handler):
            open_folder(path)
        else:
            handler([path])

    # the filter of start_application receives the event and gives the path to this function
    app.setProperty("fileopenhandler", open_path)


# File > Open Recent shows this number of models
RECENT_LIMIT: t.Final = 10


def get_recent_setting_key() -> str:
    """Return the key of the settings that holds the recent models of this viewer."""
    from PySide6 import QtWidgets

    return f"{QtWidgets.QApplication.applicationDisplayName()}/recentmodels"


def get_recent_models() -> list[str]:
    """Return the folders of the models that the viewer opened last, the newest first."""
    return get_list_setting(get_recent_setting_key())


def add_recent_model(folder: Path | str) -> None:
    """Put the folder of a model at the start of the recent models of File > Open Recent."""
    # resolve_modelpath gives ".." the real name of its folder, thus two spellings of one folder give one entry in the
    # list. A model on a different host keeps its path
    path = str(resolve_modelpath(folder))
    recent = [path, *(other for other in get_recent_models() if other != path)][:RECENT_LIMIT]
    get_settings().setValue(get_recent_setting_key(), recent)


def get_session_setting_key() -> str:
    """Return the key of the settings that holds the commands of the windows that were open at the last quit."""
    from PySide6 import QtWidgets

    return f"{QtWidgets.QApplication.applicationDisplayName()}/session"


def add_session_window(tokens: "Sequence[str]") -> None:
    """Keep the command of a window that closes because the application quits.

    Each path of the command that exists in the working folder becomes an absolute path, because the next start can
    be in a different folder. A name of a reference file in the data of artistools stays a name.
    """
    get_settings().setValue(
        get_session_setting_key(),
        [*get_list_setting(get_session_setting_key()), json.dumps(get_absolute_tokens(tokens))],
    )


# the options that give a text or a style of the plot. Such a value is not a path, also when a folder of the working
# folder has the same name, e.g. the label run1 of the folder run1
TEXT_FLAGS: t.Final = frozenset({*SERIES_STYLE_FLAGS, "-title"})


def get_absolute_tokens(tokens: "Sequence[str]") -> list[str]:
    """Return the tokens of a command with an absolute path for each path that exists in the working folder.

    An empty token, e.g. the value of -label "", stays empty. Path("") is the working folder, and it exists. A value of
    an option of TEXT_FLAGS stays the same.
    """
    absolutetokens: list[str] = []
    # the flag of the option that takes the next value. A flag that holds its value, e.g. -ymin=-1, takes no more
    flag = ""
    for word in tokens:
        if is_flag(word):
            flag = "" if "=" in word else word
            absolutetokens.append(word)
        elif word and flag not in TEXT_FLAGS and Path(word).exists():
            absolutetokens.append(str(Path(word).absolute()))
        else:
            absolutetokens.append(word)
    return absolutetokens


def take_session_windows() -> list[list[str]]:
    """Return the commands of the windows that were open at the last quit, and remove them from the settings.

    If the setting "reopenwindows" of the Settings window is off, the list is empty.
    """
    saved = get_list_setting(get_session_setting_key())
    get_settings().remove(get_session_setting_key())
    if not get_bool_setting("reopenwindows", default=True):
        return []
    return [[str(token) for token in json.loads(item)] for item in saved]


class OpenWindow(t.Protocol):
    """The function of a viewer that opens a window for the arguments of a command, or returns the reason for none.

    A new window takes the options of the Settings window that the command does not give, e.g. -figscale. A window of
    the last session keeps its command, thus newwindow is then False.
    """

    def __call__(
        self, tokens: "Sequence[str]", windows: "list[QtWidgets.QMainWindow]", *, newwindow: bool = True
    ) -> str | None:
        """Open a window for the tokens, and return None, or the reason for no window."""


def reopen_session_windows(open_window: OpenWindow, windows: "list[QtWidgets.QMainWindow]") -> None:
    """Open the windows of the last session, as the apps of macOS do, and keep the first window in front.

    A window with the same command as an open window does not open again. The comparison leaves out -figwidthscale,
    because each window fits it to its own size. An error of a window goes to the terminal. Each window opens after
    the event loop runs again, thus the first window can draw and take input while the others read their runs. A
    window gets the command of the last session, and no option of the Settings window, because the user can have
    removed such an option, e.g. -figscale.
    """
    from PySide6 import QtCore

    shown = [
        remove_figwidthscale(get_absolute_tokens(window.property("sessiontokens")()))
        for window in windows
        if callable(window.property("sessiontokens"))
    ]
    pending = [tokens for tokens in take_session_windows() if remove_figwidthscale(tokens) not in shown]
    firstwindows = list(windows)

    def open_next_window() -> None:
        if not pending:
            for window in firstwindows:
                if is_live(window):
                    activate_window(window)
            return
        tokens = pending.pop(0)
        message = run_command_step(lambda: open_window(tokens, windows, newwindow=False), quiet=False)
        if message is not None:
            print_error(f"The viewer cannot open the window of the last session: {message}")
        QtCore.QTimer.singleShot(0, open_next_window)

    if pending:
        QtCore.QTimer.singleShot(0, open_next_window)


def run_viewer_application(
    applicationname: str,
    iconcurve: "npt.NDArray[np.float64]",
    open_window: OpenWindow,
    tokens: "Sequence[str]",
    documenttypes: "Sequence[str]" = ("public.folder",),
) -> None:
    """Open a viewer window for the tokens and the windows of the last session, then run until the user quits.

    A folder from the Dock icon opens in a new window, and its errors go to the terminal.
    """
    app = start_application(applicationname, iconcurve, documenttypes)
    # the list holds a reference to each window, thus Python keeps the window while it is open
    windows: list[QtWidgets.QMainWindow] = []

    def open_dock_folder(folder: str) -> None:
        if (message := open_model_folder(folder, open_window, windows)) is not None:
            print_error(message)

    handle_file_open_events(app, open_dock_folder)
    open_window(tokens, windows)
    reopen_session_windows(open_window, windows)
    app.exec()


def remove_figwidthscale(tokens: "Sequence[str]") -> list[str]:
    """Return the tokens of a command without -figwidthscale and its value, which a window sets."""
    return [
        word
        for index, word in enumerate(tokens)
        if word != "-figwidthscale" and (index == 0 or tokens[index - 1] != "-figwidthscale")
    ]


def get_window_setting_keys(window: "QtWidgets.QMainWindow") -> tuple[str, str]:
    """Return the keys of the settings of the geometry and of the splitter of the window of a viewer."""
    return f"{window.objectName()}/geometry", f"{window.objectName()}/splitter"


def make_central_splitter(
    window: "QtWidgets.QMainWindow", plotarea: "QtWidgets.QWidget", sidebar: "QtWidgets.QWidget"
) -> "QtWidgets.QSplitter":
    """Put the plot area and the sidebar side by side at the centre of the window, with a handle between them.

    A drag of the handle changes the width of the sidebar, and a drag to the edge hides it. The plot area takes the
    extra width of the window.
    """
    from PySide6 import QtCore
    from PySide6 import QtWidgets

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
    splitter.addWidget(plotarea)
    splitter.addWidget(sidebar)
    # the sidebar keeps its width and a drag to the edge hides it, which are the defaults of Qt for the sidebar
    splitter.setStretchFactor(0, 1)
    splitter.setCollapsible(0, False)  # ruff:ignore[boolean-positional-value-in-call]
    window.setCentralWidget(splitter)
    return splitter


@contextlib.contextmanager
def show_wait_cursor() -> "Generator[None]":
    """Show the wait cursor until the block ends, e.g. while a step that can take seconds runs in the window thread."""
    from PySide6 import QtCore
    from PySide6 import QtGui
    from PySide6 import QtWidgets

    QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
    try:
        yield
    finally:
        QtWidgets.QApplication.restoreOverrideCursor()


# the choices of the appearance in the Settings window, as the Settings of macOS and Xcode name them
APPEARANCES: t.Final = ("System", "Light", "Dark")


def apply_appearance() -> None:
    """Give the application the appearance of the Settings window: the appearance of the system, light, or dark.

    A change gives the signal colorSchemeChanged, and each window then draws its plot again.
    """
    from PySide6 import QtCore
    from PySide6 import QtGui

    schemes = {"Light": QtCore.Qt.ColorScheme.Light, "Dark": QtCore.Qt.ColorScheme.Dark}
    appearance = str(get_settings().value("appearance", "System"))
    QtGui.QGuiApplication.styleHints().setColorScheme(schemes.get(appearance, QtCore.Qt.ColorScheme.Unknown))


def activate_window(window: "QtWidgets.QWidget") -> None:
    """Show the window in front of the other windows, and give it the keyboard."""
    if window.isMinimized():
        window.showNormal()
    window.raise_()
    window.activateWindow()


def open_model_folder(
    folder: str,
    open_window: "Callable[[Sequence[str], list[QtWidgets.QMainWindow]], str | None]",
    windows: "list[QtWidgets.QMainWindow]",
) -> str | None:
    """Open a new window for the folder of a run. Return an error message if no window opened.

    A SystemExit in a Qt slot ends the process, thus an error of the new window stays in this window.
    """
    message = run_command_step(lambda: open_window([folder], windows), quiet=False)
    if message is not None:
        return f"The viewer cannot open {folder}: {message}"
    get_settings().setValue("modelfolder", str(Path(folder).parent))
    return None
