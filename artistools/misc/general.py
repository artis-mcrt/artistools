"""Small generic utilities."""

import contextlib
import functools
import sys
import typing as t

if t.TYPE_CHECKING:
    from types import ModuleType
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl


def df_filter_minmax_bracketed(
    df: pl.LazyFrame | pl.DataFrame, colname: str, minval: float | None, maxval: float | None
) -> pl.LazyFrame:
    """Filter rows by bounds and include the nearest exterior row at each bound for interpolation."""
    df = df.lazy()
    if minval is maxval is None:
        return df

    if minval is not None:
        df = df.filter(
            (pl.col(colname).min() >= minval)
            | (pl.col(colname) >= pl.col(colname).filter(pl.col(colname) <= minval).max())
        )

    if maxval is not None:
        df = df.filter(
            (pl.col(colname).max() <= maxval)
            | (pl.col(colname) <= pl.col(colname).filter(pl.col(colname) >= maxval).min())
        )

    return df


def vec_len(vec: Sequence[float] | npt.NDArray[np.floating]) -> float:
    """Return the Euclidean length of a vector."""
    return float(np.sqrt(np.dot(vec, vec)))


@functools.lru_cache
def savgol_coeffs(window_length: int, polyorder: int) -> npt.NDArray[np.float64]:
    """Return the Savitzky-Golay smoothing coefficients for a centred window."""
    halflen = window_length // 2
    xwindow = np.arange(-halflen, halflen + 1, dtype=np.float64)
    # the first pseudoinverse row evaluates the least-squares fitted polynomial at the window centre
    return np.asarray(np.linalg.pinv(np.vander(xwindow, polyorder + 1, increasing=True))[0], dtype=np.float64)


def savgol_filter(ylist: npt.ArrayLike, window_length: int, polyorder: int) -> npt.NDArray[np.float64]:
    """Apply Savitzky-Golay smoothing to a 1D array, fitting polynomials to the edge windows.

    Matches scipy.signal.savgol_filter with mode="interp" for an odd window_length. Unlike scipy,
    only odd windows and 1D input are accepted: an even window is centred half a sample off, which
    phase-shifts the output, and every caller here smooths a single series.
    """
    y = np.asarray(ylist, dtype=np.float64)
    if y.ndim != 1:
        msg = f"savgol_filter needs a 1D array, got {y.ndim} dimensions"
        raise ValueError(msg)
    if window_length % 2 == 0 or window_length < 3:
        msg = f"window_length {window_length} must be an odd number of at least 3"
        raise ValueError(msg)
    if not 0 <= polyorder < window_length:
        msg = f"polyorder {polyorder} must be at least zero and less than window_length {window_length}"
        raise ValueError(msg)
    if y.size < window_length:
        msg = f"window_length {window_length} exceeds the data length {y.size}"
        raise ValueError(msg)

    halflen = window_length // 2
    filtered = np.correlate(y, savgol_coeffs(window_length, polyorder), mode="same")

    # the outermost points have incomplete windows, so evaluate polynomial fits to the first and last full windows
    xedge = np.arange(window_length, dtype=np.float64)
    filtered[:halflen] = np.polynomial.Polynomial.fit(xedge, y[:window_length], polyorder)(xedge[:halflen])
    filtered[-halflen:] = np.polynomial.Polynomial.fit(xedge, y[-window_length:], polyorder)(xedge[-halflen:])

    return filtered


def gaussian_filter_wrap(data: npt.NDArray[np.floating], sigma: float) -> npt.NDArray[np.float64]:
    """Smooth a 2D array with a Gaussian kernel, wrapping at the array boundaries.

    Matches scipy.ndimage.gaussian_filter with mode="wrap" and the default truncation of four
    standard deviations, but only for a 2D array and a scalar sigma greater than zero.
    """
    out = np.asarray(data, dtype=np.float64)
    if out.ndim != 2:
        msg = f"gaussian_filter_wrap needs a 2D array, got {out.ndim} dimensions"
        raise ValueError(msg)
    if sigma <= 0.0:
        msg = f"sigma {sigma} must be greater than zero"
        raise ValueError(msg)

    radius = int(4.0 * sigma + 0.5)
    xkernel = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xkernel / sigma) ** 2)
    kernel /= kernel.sum()

    def convolve_valid(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.convolve(arr, kernel, mode="valid"), dtype=np.float64)

    for axis in (0, 1):
        padwidth = [(0, 0), (0, 0)]
        padwidth[axis] = (radius, radius)
        out = np.apply_along_axis(convolve_valid, axis, np.pad(out, padwidth, mode="wrap"))
    return out


def import_optional(modulename: str) -> "ModuleType":
    """Import a module of the optional dependencies, or say how to install them.

    A bare import of pyvista or pynonthermal stops with a traceback that names no fix. This raises one
    message that gives the install command.
    """
    import importlib

    try:
        return importlib.import_module(modulename)
    except ImportError as exc:
        packagename = modulename.partition(".")[0]
        msg = (
            f"This command needs {packagename}, which is not installed. Install the optional "
            "dependencies with: uv pip install 'artistools[extras]'"
        )
        raise ModuleNotFoundError(msg) from exc


def get_progress_class() -> "type[t.Any]":
    """Return the rich tqdm class, with a progress-bar lock that a spawn pool can take.

    tqdm builds its shared progress-bar lock from the default multiprocessing context at the first bar.
    On Linux the default was fork before Python 3.14, thus a bar made before parallel_map gave its
    workers a fork lock in a spawn pool: "A SemLock created in a fork context is being shared with a
    process in a spawn context". Every bar comes through here, thus the lock comes from a spawn context
    from the start. This sets no default start method, because a bar alone starts no process, and that
    default belongs to the caller.
    """
    import multiprocessing as mp
    import warnings

    import tqdm
    import tqdm.rich
    from tqdm import TqdmExperimentalWarning

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    spawnlock = mp.get_context("spawn").RLock()
    tqdm.tqdm.set_lock(spawnlock)
    tqdm.rich.tqdm.set_lock(spawnlock)

    return tqdm.rich.tqdm


def parallel_map[IterableType, ResultType](
    fn: Callable[[IterableType], ResultType], *iterables: Iterable[IterableType], **kwargs: t.Any
) -> list[ResultType]:
    """Execute a parallel map with a progress bar, with threads on a free-threaded build and processes otherwise."""
    progressclass = get_progress_class()

    use_multiprocessing = True
    with contextlib.suppress(AttributeError):
        if not sys._is_gil_enabled():  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
            # return a thread pool if we have no GIL (free threading)
            use_multiprocessing = False

    if use_multiprocessing:
        import multiprocessing as mp

        from tqdm.contrib.concurrent import process_map

        # the lock of the progress bar comes from a spawn context, thus the pool must live in one as
        # well. Spawn is also needed because forking a process that already has polars threads is
        # unsafe. A run that takes the thread pool changes no such default
        mp.set_start_method("spawn", force=True)
        results = process_map(fn, *iterables, tqdm_class=progressclass, **kwargs)  # type: ignore[arg-type]
    else:
        from tqdm.contrib.concurrent import thread_map

        results = thread_map(fn, *iterables, tqdm_class=progressclass, **kwargs)  # type: ignore[arg-type]

    assert isinstance(results, list)
    return results
