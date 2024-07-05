from math import isclose
from pathlib import Path

import polars as pl
import pytest

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_classic_3d = at.get_config()["path_testdata"] / "test-classicmode_3d"
outputpath = Path(at.get_config()["path_testoutput"])


def test_get_levels_polars(benchmark) -> None:
    dflevels = benchmark(
        lambda: at.atomic.get_levels_polars(modelpath, get_transitions=True, get_photoionisations=True)
    )
    assert len(dflevels) == 12
    fe2_levels = dflevels.filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2)).row(0, named=True)["levels"]
    assert len(fe2_levels) == 2823
    assert isclose(fe2_levels.item(0, "energy_ev"), 0.0, abs_tol=1e-6)
    assert isclose(fe2_levels.item(2822, "energy_ev"), 23.048643, abs_tol=1e-6)


def test_get_levels(benchmark) -> None:
    dflevels = benchmark(lambda: at.atomic.get_levels(modelpath, get_transitions=True, get_photoionisations=True))
    assert len(dflevels) == 12
    fe2_levels = dflevels[(dflevels["Z"] == 26) & (dflevels["ion_stage"] == 2)].iloc[0]["levels"]
    assert len(fe2_levels) == 2823
    assert isclose(fe2_levels.iloc[0]["energy_ev"], 0.0, abs_tol=1e-6)
    assert isclose(fe2_levels.iloc[2822]["energy_ev"], 23.048643, abs_tol=1e-6)


@pytest.mark.benchmark
def test_get_ionrecombratecalibration() -> None:
    at.atomic.get_ionrecombratecalibration(modelpath=modelpath)
