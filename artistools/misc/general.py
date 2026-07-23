"""Small generic utilities."""

import contextlib
import sys
import typing as t
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl


def df_filter_minmax_bounded(
    df: pl.LazyFrame | pl.DataFrame, colname: str, minval: float | int | None, maxval: float | int | None
) -> pl.LazyFrame:
    """Filter a DataFrame to selects rows where the value in colname is between minval and maxval, and also include the closest exterior rows if xmin/xmax are between two rows. This enables linear interpolation at xmin and xmax (if the surrounding values existed in the DataFrame)."""
    df = df.lazy()
    if minval is None and maxval is None:
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

        results = process_map(fn, *iterables, tqdm_class=tqdm.rich.tqdm, **kwargs)  # type: ignore[arg-type] # zuban: ignore[no-untyped-call]
    else:
        from tqdm.contrib.concurrent import thread_map

        results = thread_map(fn, *iterables, tqdm_class=tqdm.rich.tqdm, **kwargs)  # type: ignore[arg-type] # zuban: ignore[no-untyped-call]

    assert isinstance(results, list)
    return results
