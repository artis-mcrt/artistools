"""Small generic utilities."""

import contextlib
import functools
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl


def df_filter_minmax_bounded(
    df: pl.LazyFrame | pl.DataFrame, colname: str, minval: float | None, maxval: float | None
) -> pl.LazyFrame:
    """Filter a DataFrame to selects rows where the value in colname is between minval and maxval, and also include the closest exterior rows if xmin/xmax are between two rows. This enables linear interpolation at xmin and xmax (if the surrounding values existed in the DataFrame)."""
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
    return float(np.sqrt(np.dot(vec, vec)))


@functools.lru_cache
def _savgol_coeffs(window_length: int, polyorder: int) -> npt.NDArray[np.float64]:
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
    if polyorder >= window_length:
        msg = f"polyorder {polyorder} must be less than window_length {window_length}"
        raise ValueError(msg)
    if y.size < window_length:
        msg = f"window_length {window_length} exceeds the data length {y.size}"
        raise ValueError(msg)

    halflen = window_length // 2
    filtered = np.correlate(y, _savgol_coeffs(window_length, polyorder), mode="same")

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


def parallel_map[IterableType, ResultType](
    fn: Callable[[IterableType], ResultType],
    *iterables: Iterable[IterableType],
    allow_multiprocessing: bool = True,
    **kwargs: t.Any,
) -> list[ResultType]:
    """Execute a parallel map with a progress bar using either multithreading (for free-threading python or allow_multiprocessing=False) or multiprocessing."""
    import multiprocessing as mp
    import warnings

    import tqdm.rich
    from tqdm import TqdmExperimentalWarning

    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

    use_multiprocessing = allow_multiprocessing
    if allow_multiprocessing:
        with contextlib.suppress(AttributeError):
            if not sys._is_gil_enabled():  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
                # return a thread pool if we have no GIL (free threading)
                use_multiprocessing = False

    if use_multiprocessing:
        mp.set_start_method("spawn", force=True)
        from tqdm.contrib.concurrent import process_map

        results = process_map(fn, *iterables, tqdm_class=tqdm.rich.tqdm, **kwargs)  # type: ignore[arg-type]
    else:
        from tqdm.contrib.concurrent import thread_map

        results = thread_map(fn, *iterables, tqdm_class=tqdm.rich.tqdm, **kwargs)  # type: ignore[arg-type]

    assert isinstance(results, list)
    return results
