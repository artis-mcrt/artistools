import collections.abc

import polars as pl

def estimparse(folderpath: str, rankmin: int, rankmax: int) -> pl.DataFrame: ...
def read_transitiondata(
    transitions_filename: str, ionlist: collections.abc.Collection[tuple[int, int]] | None
) -> dict[tuple[int, int], pl.DataFrame]: ...
