from collections.abc import Collection

import polars as pl

def estimparse(folderpath: str, rankmin: int, rankmax: int) -> pl.DataFrame: ...
def read_transitiondata(
    transitions_filename: str, ionlist: Collection[tuple[int, int]] | None
) -> dict[tuple[int, int], pl.DataFrame]: ...
