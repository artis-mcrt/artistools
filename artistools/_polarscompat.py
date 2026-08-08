"""Restore polars Series-to-Expr method dispatch on CPython 3.15.

polars leaves most Series methods as docstring-only stubs and, at import time, rebinds each one to the
matching Expr implementation. It picks the stubs to rebind by inspecting their code objects, requiring a
trailing None in co_consts. CPython 3.15 no longer stores that None for a docstring-only function, so
polars rebinds nothing and every stub silently returns None: Series.unique(), .abs(), .drop_nulls(), and
the whole .str/.dt/.list namespaces included.

Delete this module once polars ships a fix.
"""

import typing as t
from collections.abc import Callable
from types import FunctionType

import polars as pl
from polars.series import utils as plseriesutils
from polars.series.array import ArrayNameSpace
from polars.series.binary import BinaryNameSpace
from polars.series.categorical import CatNameSpace
from polars.series.datetime import DateTimeNameSpace
from polars.series.ext import ExtensionNameSpace
from polars.series.list import ListNameSpace
from polars.series.string import StringNameSpace
from polars.series.struct import StructNameSpace

_DISPATCH_CLASSES: tuple[type, ...] = (
    pl.Series,
    ArrayNameSpace,
    BinaryNameSpace,
    CatNameSpace,
    DateTimeNameSpace,
    ExtensionNameSpace,
    ListNameSpace,
    StringNameSpace,
    StructNameSpace,
)


def _is_empty_method(func: Callable[..., pl.Series]) -> bool:
    """Report whether a polars method is a stub whose body is nothing but a docstring."""
    # matching an empty function's bytecode already implies a body that only returns None, so unlike
    # polars' own check there is nothing left to confirm in the version-dependent co_consts
    return isinstance(func, FunctionType) and func.__code__.co_code in plseriesutils._EMPTY_BYTECODE  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]


def _dispatch_is_working() -> bool:
    """Report whether polars rebound its stubs, by calling a Series method it never implements itself."""
    # the type stubs promise a Series, so cast away the lie that an unrepaired stub returns one
    return t.cast("object", pl.Series("x", [1]).unique()) is not None


def repair_series_expr_dispatch() -> None:
    """Rebind polars' unimplemented Series methods to their Expr equivalents if polars itself failed to."""
    if _dispatch_is_working():
        return

    # expr_dispatch reads _is_empty_method as a module global, so swap it only for the rebinding pass
    original = plseriesutils._is_empty_method  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]
    plseriesutils._is_empty_method = _is_empty_method  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]  # ty:ignore[invalid-assignment]
    try:
        for cls in _DISPATCH_CLASSES:
            plseriesutils.expr_dispatch(cls)
    finally:
        plseriesutils._is_empty_method = original  # ruff:ignore[private-member-access]  # pyright: ignore[reportPrivateUsage]

    if not _dispatch_is_working():
        msg = "failed to restore the polars Series methods that polars leaves unimplemented on this Python version"
        raise RuntimeError(msg)
