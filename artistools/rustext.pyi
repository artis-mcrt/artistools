import polars as pl

def estimparse(folderpath: str, rankmin: int, rankmax: int) -> pl.DataFrame: ...
def read_transitiondata(transitions_filename: str) -> dict[tuple[int, int], pl.DataFrame]: ...
