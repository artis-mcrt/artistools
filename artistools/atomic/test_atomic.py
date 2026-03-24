import math

import polars as pl
import pytest

import artistools as at

modelpath = at.get_path("testdata") / "testmodel"
modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"
outputpath = at.get_path("testoutput")


def test_get_levels() -> None:
    dflevels = at.atomic.get_levels(modelpath, get_transitions=True, get_photoionisations=True)
    assert len(dflevels) == 12
    fe2_levels = dflevels.filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 2)).row(0, named=True)["levels"]
    assert len(fe2_levels) == 2823
    assert math.isclose(fe2_levels.item(0, "energy_ev"), 0.0, abs_tol=1e-6)
    assert math.isclose(fe2_levels.item(2822, "energy_ev"), 23.048643, abs_tol=1e-6)


@pytest.mark.benchmark
def test_get_ionrecombratecalibration() -> None:
    at.atomic.get_ionrecombratecalibration(modelpath=modelpath)
