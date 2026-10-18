"""Keep the lazy imports of Python 3.15 away from the Qt bindings.

artistools makes its own imports lazy on Python 3.15. After PySide6 loads, the attribute of a module gives a pending
lazy import as it is, e.g. print_error of artistools. A call of such an object fails. Thus the pending imports resolve
before PySide6 runs. This module imports only the standard library, because artistools/__init__.py imports it first.
"""

import contextlib
import importlib.abc
import importlib.machinery
import sys
import types
import typing as t

if t.TYPE_CHECKING:
    from collections.abc import Sequence

# the top-level packages of the Qt bindings. shiboken runs its own import hook, and a lazy import in its support code
# imported itself again, thus Python stopped with "libshiboken: could not init enum"
QT_PACKAGES: t.Final = frozenset({"PySide6", "shiboken6", "shibokensupport"})


def qt_is_loaded() -> bool:
    """Return True if a module of the Qt bindings is in sys.modules."""
    return any(name.partition(".")[0] in QT_PACKAGES for name in sys.modules)


def is_artistools_module(name: str | None) -> bool:
    """Return True for the name of artistools or of one of its modules."""
    return name is not None and (name == "artistools" or name.startswith("artistools."))


def can_import_lazily(importing: str | None, imported: str, fromlist: object) -> bool:
    """Return True for an import of an artistools module that can wait until the first use of its name.

    The imports of other code stay eager, e.g. of a script or of matplotlib. Then only artistools holds pending
    imports, and a script that imports PySide6 loads it at once. A lazy import inside polars stopped polars as it
    loaded, and one inside matplotlib broke its rcParams. numpy stays eager, see artistools/__init__.py. The Qt bindings
    also stay eager. polars itself is lazy, and a read of a polars name, e.g. an annotation of a decorated function,
    loads it.
    """
    del fromlist
    return (
        is_artistools_module(importing)
        and not imported.startswith("numpy")
        and imported.partition(".")[0] not in QT_PACKAGES
    )


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
    """Resolve each pending lazy import of the artistools modules, and make each later import eager.

    Run this function before PySide6 loads. After that, no read of a module resolves a pending lazy import.
    """
    lazyimporttype = getattr(types, "LazyImportType", None)
    if lazyimporttype is None:
        return
    # the imports become eager first, thus a module that a resolution imports has no pending import
    make_imports_eager()
    for name, module in list(sys.modules.items()):
        namespace = getattr(module, "__dict__", None)
        if not is_artistools_module(name) or not isinstance(namespace, dict):
            continue
        for attrname, value in list(namespace.items()):
            # an import that fails, e.g. of an optional package, stays lazy, and its first use gives the error
            if isinstance(value, lazyimporttype):
                with contextlib.suppress(Exception):
                    namespace[attrname] = getattr(module, attrname)


class ResolvingLoader(importlib.abc.Loader):
    """A loader of a Qt package that resolves the lazy imports of artistools, and then runs the loader of the package.

    The import system runs a loader without its global lock. A resolution can then wait for a module that a different
    thread imports, and no deadlock occurs.
    """

    def __init__(self, loader: importlib.abc.Loader) -> None:
        """Keep the loader of the package."""
        self.loader = loader

    @t.override
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType | None:
        return self.loader.create_module(spec)

    @t.override
    def exec_module(self, module: types.ModuleType) -> None:
        for finder in list(sys.meta_path):
            if isinstance(finder, QtImportGuard):
                with contextlib.suppress(ValueError):
                    sys.meta_path.remove(finder)
        resolve_lazy_imports()
        # the module keeps the loader of its package, e.g. for importlib.resources
        module.__loader__ = self.loader
        if module.__spec__ is not None:
            module.__spec__.loader = self.loader
        self.loader.exec_module(module)


class QtImportGuard:
    """A finder of sys.meta_path that gives a Qt package the loader that resolves the lazy imports first.

    The finder asks the next finders for the spec, thus a search alone resolves nothing, e.g. importlib.util.find_spec.
    The guard leaves sys.meta_path when a Qt package loads. Thus each entry point is safe, e.g. a viewer, a test, or a
    script that imports PySide6 after artistools.
    """

    def find_spec(
        self, fullname: str, path: "Sequence[str] | None" = None, target: types.ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        """Return the spec of the next finder for a Qt package, with a loader that resolves the lazy imports first."""
        if fullname.partition(".")[0] not in QT_PACKAGES:
            return None
        finders = list(sys.meta_path)
        position = next((index for index, finder in enumerate(finders) if finder is self), len(finders))
        for finder in finders[position + 1 :]:
            findspec = getattr(finder, "find_spec", None)
            spec = findspec(fullname, path, target) if callable(findspec) else None
            if not isinstance(spec, importlib.machinery.ModuleSpec):
                continue
            if isinstance(spec.loader, importlib.abc.Loader):
                spec.loader = ResolvingLoader(spec.loader)
            return spec
        return None
