"""Find the model folder of a model path, and test what a path names: a model, reference data, or comparison data."""

import functools
import inspect
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from functools import cache
from functools import lru_cache
from pathlib import Path

from artistools.misc.console import exit_with_error
from artistools.misc.fileio import COMPRESSED_EXTENSIONS
from artistools.misc.fileio import firstexisting_or_none
from artistools.misc.remote import is_remote_path
from artistools.misc.remote import on_model_host
from artistools.paths import get_path


@lru_cache(maxsize=256)
@on_model_host
def get_remote_path_kind(path: Path) -> tuple[bool, bool, bool]:
    """Return whether a remote path is a folder, whether it is a file, and whether it is a model folder.

    The host gives the three answers in one ssh round trip. The spectrum viewer asks about the same paths at each
    plot, and 10 round trips took 0.33 s of the 0.5 s of a plot. The kind of a model path does not change while the
    process runs, thus the cache keeps the answers until the process stops.
    """
    return path.is_dir(), path.is_file(), path.is_dir() and (path / "input.txt").is_file()


def path_is_file(path: Path) -> bool:
    """Return whether the path is a file. The host of a remote path gives the answer."""
    return get_remote_path_kind(path)[1] if is_remote_path(path) else path.is_file()


def path_is_dir(path: Path) -> bool:
    """Return whether the path is a folder. The host of a remote path gives the answer."""
    return get_remote_path_kind(path)[0] if is_remote_path(path) else path.is_dir()


@on_model_host
def folder_holds_match(folder: Path, pattern: str) -> bool:
    """Return whether a file in the folder matches the glob pattern. The host of a remote path gives the answer."""
    return any(folder.glob(pattern))


@cache
def find_bundled_data_file(filename: Path | str, bundledsubfolder: str) -> Path | None:
    """Return the path of a file in the named data folder of the package, or None when the folder holds none."""
    return firstexisting_or_none(
        filename, folder=Path(get_path("artistools_dir"), bundledsubfolder), tryzipped=True, search_subfolders=False
    )


def find_reference_data_file(filename: Path | str, bundledsubfolder: str) -> Path | None:
    """Return the path of a file of reference data, or None when no such file exists.

    The file is either at the given path or in the named folder of the data that the package holds,
    and a compressed file of the same name is also accepted. The light curves and the spectra each
    hold reference data of their own, thus the caller names the folder that holds its kind.

    The search of the working folder runs on every call, because the working folder can change while
    the process runs. A cache holds only the search of the package data, because that folder is
    constant.
    """
    if found := firstexisting_or_none(filename, folder=Path(), tryzipped=True, search_subfolders=False):
        return found

    return find_bundled_data_file(filename, bundledsubfolder)


def require_reference_data_file(filename: Path | str, bundledsubfolder: str, description: str) -> Path:
    """Return the path of a file of reference data, or stop with a message when no such file exists.

    The description names the kind of the file, e.g. "Kurucz line list". The message of the error
    gives that name to the user.
    """
    found = find_reference_data_file(filename, bundledsubfolder)
    if found is None:
        exit_with_error(
            f"Could not find the {description} file {filename}",
            f"Put {filename} in the working folder, or in the data folder of the artistools package",
        )

    return found


def path_is_reference_data(filepath: Path | str, bundledsubfolder: str) -> bool:
    """Return whether the path names a file of reference data and not the output of an ARTIS model.

    A name that ends in .out belongs to ARTIS, e.g. spec.out. A user can give reference data such a
    name as well, thus the folder decides: a model folder holds input.txt beside the output files. A reader opens a
    file of reference data on this host, thus a remote path names a model, and the test needs no connection to its
    host.
    """
    path = Path(filepath)
    if is_remote_path(path) or path_is_dir(path) or find_reference_data_file(path, bundledsubfolder) is None:
        return False

    if not path_is_artis_model(path):
        return True

    # a model folder holds the output of ARTIS, and a run on a cluster writes into a subfolder of it
    return not any(folder_is_artis_model(folder) for folder in (path.parent, path.parent.parent))


def get_model_folder(modelpath: str | Path) -> Path:
    """Return the model folder, whether modelpath names the folder itself or a file inside it."""
    path = Path(modelpath)

    return path.parent if path_is_file(path) else path


def path_is_artis_model(filepath: Path | str) -> bool:
    """Return whether the path is an ARTIS model and not a reference data file.

    An ARTIS model is a folder, or an output file of ARTIS that is possibly compressed.
    """
    filepath = Path(filepath)

    # the test of the name comes first, because the test of a remote folder asks its host
    return filepath.name.endswith((".out", *(f".out{ext}" for ext in COMPRESSED_EXTENSIONS))) or path_is_dir(filepath)


def folder_is_artis_model(folder: Path | str) -> bool:
    """Return whether this folder is a model folder.

    A model folder holds input.txt beside the output files. Every reader of a model needs that file, thus
    this function tests for it. The host of a remote path gives the answer.
    """
    folder = Path(folder)
    if is_remote_path(folder):
        return get_remote_path_kind(folder)[2]

    return folder.is_dir() and (folder / "input.txt").is_file()


def get_artis_model_folders(modelpaths: Sequence[Path | str]) -> list[Path]:
    """Return the model folder of each ARTIS model in modelpaths.

    A reference file or a code comparison file gives none.
    """
    return [
        get_model_folder(path)
        for path in modelpaths
        if path_is_artis_model(path) and folder_is_artis_model(get_model_folder(path))
    ]


def path_is_codecomparison(filepath: Path | str) -> bool:
    """Return whether the path is a virtual codecomparison path and not a real folder on disk.

    A codecomparison path has the form "codecomparison/<model>/<code>". It names a data set of the
    radiative transfer code comparison workshop, thus no such folder exists. The parts of the path
    come first, because that test reads no disk.
    """
    filepath = Path(filepath)

    return filepath.parts[:1] == ("codecomparison",) and not filepath.exists()


@lru_cache(maxsize=64)
def resolve_path_cached(pathstr: str, workingfolder: str) -> Path:
    """Return the absolute path of pathstr, which a relative path reads below workingfolder.

    A call of resolve reads the disk for each part of the path. Every cached reader of a model calls
    resolve_modelpath first, thus this cache keeps that call cheap.
    """
    return Path(workingfolder, pathstr).resolve() if workingfolder else Path(pathstr).resolve()


def resolve_modelpath(modelpath: Path | str) -> Path:
    """Return the absolute path of a model.

    A virtual codecomparison path and a remote path stay as they are. A cached function takes the absolute path. The
    default model path is the relative Path("."), thus the working folder belongs to the key of the
    cache. The answer would otherwise stay after the user changes that folder.
    """
    path = Path(modelpath)
    if path_is_codecomparison(path) or is_remote_path(path):
        return path

    return resolve_path_cached(str(path), "" if path.is_absolute() else str(Path.cwd()))


class ModelpathCache[**P, R]:
    """Call a reader with the absolute path of the model, and keep the results in an lru_cache.

    The default model path is the relative Path("."). A cache of a relative path keeps the first answer after the user
    changes the working folder, thus the key holds the path that resolve_modelpath gives. A call gives the path by
    position, by the keyword modelpath, or by the name of the first parameter of the reader, e.g. filename.
    """

    def __init__(self, function: Callable[t.Concatenate[Path, P], R], maxsize: int) -> None:
        """Put an lru_cache of maxsize entries on the reader."""
        cached = lru_cache(maxsize=maxsize)(function)
        # the types of lru_cache take each argument as Hashable, and the reader gives the types of its arguments
        self.callcached = t.cast("Callable[..., R]", cached)
        self.cache_clear = cached.cache_clear
        self.cache_info = cached.cache_info
        self.name = str(getattr(function, "__qualname__", function))
        self.signature = inspect.signature(function)
        self.pathkeywords = {"modelpath", next(iter(self.signature.parameters))}
        # the server of a remote model follows __wrapped__ to the reader below the cache
        functools.update_wrapper(self, cached)

    if t.TYPE_CHECKING:
        # the type checkers know only the keyword modelpath, because a ParamSpec cannot rename a parameter
        # pyrefly: ignore[empty-body]
        def __call__(self, modelpath: Path | str, *args: P.args, **kwargs: P.kwargs) -> R:
            """Return the result of the reader for the absolute path of the model."""
            ...

    else:

        def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
            """Return the result of the reader for the absolute path of the model.

            A path that a keyword gives goes to the first position. Each other argument takes its position in the
            signature, and a missing argument takes its default. Thus each form of a call shares one cache entry.
            """
            if not args and (pathkeyword := next((key for key in self.pathkeywords if key in kwargs), None)):
                args = (kwargs.pop(pathkeyword),)
            # bind raises the TypeError of a missing argument
            boundargs = self.signature.bind(*args, **kwargs)
            boundargs.apply_defaults()
            modelpath, *otherargs = boundargs.args
            return self.callcached(resolve_modelpath(modelpath), *otherargs, **boundargs.kwargs)

    @t.override
    def __reduce__(self) -> str:
        """Give pickle the name of the module attribute, because pickle cannot copy the cache."""
        return self.name


def modelpath_cache[**P, R](maxsize: int) -> Callable[[Callable[t.Concatenate[Path, P], R]], ModelpathCache[P, R]]:
    """Return a decorator that gives a reader of a model path a cache with the key of the absolute path."""

    def decorator(function: Callable[t.Concatenate[Path, P], R]) -> ModelpathCache[P, R]:
        return ModelpathCache(function, maxsize)

    return decorator


# the attribute of a cache that holds its group of clear_at_reload
RELOAD_GROUP_ATTRIBUTE: t.Final = "reloadgroup"

# a group of the caches that Reload Data clears. The estimators viewer clears only the group "estimators", thus the
# caches of the other viewers stay
type ReloadGroup = t.Literal["output", "estimators"]


def clear_at_reload[CacheT](group: ReloadGroup) -> Callable[[CacheT], CacheT]:
    """Return a decorator that marks the cache of a reader of a file that ARTIS writes while it runs, e.g. spec.out.

    Reload Data clears each cache with the mark of its groups, see clear_reload_caches. Put the decorator above the
    cache, e.g. above lru_cache. The files of the input of a model stay the same, e.g. model.txt, thus their caches
    get no mark.
    """

    def mark(cachedreader: CacheT) -> CacheT:
        setattr(cachedreader, RELOAD_GROUP_ATTRIBUTE, group)
        return cachedreader

    return mark


def clear_reload_caches(groups: Collection[ReloadGroup]) -> None:
    """Clear each cache with the mark of a group in groups, in each module of artistools that this process imported.

    A fixed list of the caches needs a change for each new reader. Thus the function searches the modules for the
    mark. A module that the process did not import holds no cache. getattr_static runs no code of an object, e.g. of a
    lazy import.
    """
    cachedreaders = {
        id(value): value
        for modulename, module in sys.modules.copy().items()
        if modulename.split(".", maxsplit=1)[0] == "artistools"
        for value in vars(module).copy().values()
        if inspect.getattr_static(value, RELOAD_GROUP_ATTRIBUTE, None) in groups
    }
    for cachedreader in cachedreaders.values():
        cachedreader.cache_clear()
