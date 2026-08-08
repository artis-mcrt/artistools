import os

import polars as pl

def estimparse(folderpath: str | os.PathLike[str], rankmin: int, rankmax: int) -> pl.DataFrame: ...
def read_transitiondata(
    transitions_filename: str | os.PathLike[str], ionlist: set[tuple[int, int]] | None = None
) -> dict[tuple[int, int], pl.DataFrame]: ...
def read_wsv_normalized(
    filepath: str | os.PathLike[str], skip_rows: int = 0, comment_prefix: str | None = None
) -> bytes: ...
