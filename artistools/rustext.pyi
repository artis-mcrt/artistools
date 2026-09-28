import os

import polars as pl

def estimparse(folderpath: str | os.PathLike[str], rankmin: int, rankmax: int) -> pl.DataFrame: ...
def read_transitiondata(
    transitions_filename: str | os.PathLike[str], ionlist: set[tuple[int, int]] | None = None
) -> dict[tuple[int, int], pl.DataFrame]: ...
def sum_binned_line_opacities(
    dflevels: pl.DataFrame,
    dflines: pl.DataFrame,
    dfcells: pl.DataFrame,
    nnioncolumns: list[str],
    numbins: int,
    k_b_ev_per_k: float,
) -> pl.DataFrame: ...
def get_bin_indices(df: pl.DataFrame, valuecolumn: str, edges: list[float]) -> pl.DataFrame: ...
def sum_weights_in_bins(
    df: pl.DataFrame,
    valuecolumn: str,
    weightcolumn: str,
    edges: list[float],
    groupcolumn: str | None = None,
    ngroups: int = 1,
) -> pl.DataFrame: ...
