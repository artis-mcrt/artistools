import os

import polars as pl

def estimparse(folderpath: str | os.PathLike[str], rankmin: int, rankmax: int) -> pl.DataFrame: ...
def estimparse_allranks(filepath: str | os.PathLike[str]) -> pl.DataFrame: ...
def estimtimesteps(filepath: str | os.PathLike[str]) -> list[int]: ...
def sum_binned_line_opacities(
    dflevels: pl.DataFrame,
    dflines: pl.DataFrame,
    dfcells: pl.DataFrame,
    nnioncolumns: list[str],
    taucaps: list[tuple[str, float]],
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
    sumsquares: bool = False,
) -> pl.DataFrame: ...
