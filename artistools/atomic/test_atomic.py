from math import isclose
from pathlib import Path

import polars as pl
import pytest

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_classic_3d = at.get_config()["path_testdata"] / "test-classicmode_3d"
outputpath = Path(at.get_config()["path_testoutput"])


@pytest.mark.benchmark()
def test_get_levels() -> None:
    dflevels = at.atomic.get_levels_polars(modelpath, get_transitions=True, get_photoionisations=True)
    # print(dflevels)
    assert len(dflevels) == 12
    fe2_levels = dflevels.filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2)).row(0, named=True)["levels"]
    # print(fe2_levels)
    assert len(fe2_levels) == 2823
    assert isclose(fe2_levels.item(0, "energy_ev"), 0.0, abs_tol=1e-6)
    assert isclose(fe2_levels.item(2822, "energy_ev"), 23.048643, abs_tol=1e-6)


@pytest.mark.benchmark()
def test_get_ionrecombratecalibration() -> None:
    at.atomic.get_ionrecombratecalibration(modelpath=modelpath)
