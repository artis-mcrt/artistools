from pathlib import Path

import pytest

import artistools as at

modelpath = at.get_config()["path_testdata"] / "testmodel"
modelpath_classic_3d = at.get_config()["path_testdata"] / "test-classicmode_3d"
outputpath = Path(at.get_config()["path_testoutput"])


@pytest.mark.benchmark()
def test_get_levels() -> None:
    dflevels = at.atomic.get_levels(modelpath, get_transitions=True, get_photoionisations=True)
    assert len(dflevels) == 12
    assert len(dflevels.iloc[0]) == 6


@pytest.mark.benchmark()
def test_get_ionrecombratecalibration() -> None:
    at.atomic.get_ionrecombratecalibration(modelpath=modelpath)
