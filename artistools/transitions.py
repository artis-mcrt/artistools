"""Deprecated alias that the package pynonthermal reads.

Delete this module after pynonthermal reads artistools.atomic.get_lte_partfunc. The command
plottransitions is in artistools/plottransitions.py.
"""

import typing as t
import warnings

from artistools.atomic import get_lte_partfunc as atomic_get_lte_partfunc

if t.TYPE_CHECKING:
    import polars as pl


def get_lte_partfunc(pldflevels: "pl.DataFrame", T_exc: float) -> float:
    """Return the LTE partition function of the ion. Deprecated: use artistools.atomic.get_lte_partfunc."""
    warnings.warn(
        "artistools.transitions.get_lte_partfunc is deprecated. Use artistools.atomic.get_lte_partfunc",
        DeprecationWarning,
        stacklevel=2,
    )
    return atomic_get_lte_partfunc(pldflevels, T_exc)
