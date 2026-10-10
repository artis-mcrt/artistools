"""Keep the lazy imports of Python 3.15 away from the Qt bindings.

artistools makes each import lazy on Python 3.15. After PySide6 loads, the attribute of a module gives a pending
lazy import as it is. Examples are print_error of artistools and the version of kiwisolver that matplotlib reads.
A call of such an object fails. This module imports only the standard library, because artistools/__init__.py
imports it first.
"""

import contextlib
import sys
import types
import typing as t

if t.TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.machinery import ModuleSpec

# the top-level packages of the Qt bindings. shiboken runs its own import hook, and a lazy import in its support code
# imported itself again, thus Python stopped with "libshiboken: could not init enum"
QT_PACKAGES: t.Final = frozenset({"PySide6", "shiboken6", "shibokensupport"})


def qt_is_loaded() -> bool:
    """Return True if a module of the Qt bindings is in sys.modules."""
    return any(name.partition(".")[0] in QT_PACKAGES for name in sys.modules)


def refuse_lazy_import(importing: str | None, imported: str, fromlist: object) -> bool:
    """Return False, thus each import is eager. This includes an explicit lazy import of the standard library."""
    del importing, imported, fromlist
    return False


def make_imports_eager() -> None:
    """Make each later import eager, e.g. the explicit lazy import of concurrent.futures in the standard library."""
    if hasattr(sys, "set_lazy_imports_filter") and hasattr(sys, "set_lazy_imports"):
        sys.set_lazy_imports_filter(refuse_lazy_import)
        sys.set_lazy_imports("normal")


def resolve_lazy_imports() -> None:
    """Resolve each pending lazy import of the loaded modules, and make each later import eager.

    Run this function before PySide6 loads. After that, no read of a module resolves a pending lazy import.
    """
    lazyimporttype = getattr(types, "LazyImportType", None)
    if lazyimporttype is None:
        return
    # the imports become eager first, thus a module that a resolution imports has no pending import
    make_imports_eager()
    for module in list(sys.modules.values()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for name, value in list(namespace.items()):
            # an import that fails, e.g. of an optional package, stays lazy, and its first use gives the error
            if isinstance(value, lazyimporttype):
                with contextlib.suppress(Exception):
                    namespace[name] = getattr(module, name)


class QtImportGuard:
    """A finder of sys.meta_path that resolves the lazy imports before the first import of the Qt bindings.

    The finder leaves sys.meta_path at that import, and it finds no module itself. Thus each entry point is safe, e.g.
    a viewer, a test, or a script that imports PySide6 after artistools.
    """

    def find_spec(
        self, fullname: str, path: "Sequence[str] | None" = None, target: types.ModuleType | None = None
    ) -> "ModuleSpec | None":
        """Resolve the lazy imports before a module of the Qt bindings loads, and give the search to the next finder."""
        del path, target
        if fullname.partition(".")[0] in QT_PACKAGES:
            with contextlib.suppress(ValueError):
                sys.meta_path.remove(self)
            resolve_lazy_imports()
        return None
