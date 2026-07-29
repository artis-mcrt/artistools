import math

import numpy as np
import polars as pl
import polars.testing as pltest
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


def test_read_transitiondata() -> None:
    transitionsdict = at.rustext.read_transitiondata(modelpath / "transitiondata.txt")
    assert sorted(transitionsdict.keys()) == [
        (26, 1),
        (26, 2),
        (26, 3),
        (26, 4),
        (26, 5),
        (27, 2),
        (27, 3),
        (27, 4),
        (28, 2),
        (28, 3),
        (28, 4),
        (28, 5),
    ]

    dftransitions = transitionsdict[26, 2]
    assert dftransitions.columns == ["lower", "upper", "A", "collstr", "forbidden"]
    assert dftransitions.dtypes == [pl.Int32, pl.Int32, pl.Float32, pl.Float32, pl.Int32]
    assert dftransitions.shape == (539020, 5)
    # level indices are zero-based, unlike the one-based level numbers in the file
    assert dftransitions.row(0, named=True) == pytest.approx({
        "lower": 0,
        "upper": 1,
        "A": 0.00209,
        "collstr": 3.23,
        "forbidden": 1,
    })

    # an ionlist selects a subset of the ions without changing what is read for them
    selectedions = at.rustext.read_transitiondata(modelpath / "transitiondata.txt", ionlist={(26, 2)})
    assert selectedions.keys() == {(26, 2)}
    pltest.assert_frame_equal(selectedions[26, 2], dftransitions)


@pytest.mark.benchmark
def test_get_ionrecombratecalibration() -> None:
    recombination_rates = at.atomic.get_ionrecombratecalibration(modelpath=modelpath)

    assert len(recombination_rates) == 55
    assert {(26, 2), (26, 3), (26, 4), (26, 5)} <= recombination_rates.keys()
    assert all(
        dataframe.shape == (81, 4)
        and dataframe.columns == ["log10T_e", "rrc_low_n", "rrc_total", "T_e"]
        and dataframe["log10T_e"].is_sorted()
        for dataframe in recombination_rates.values()
    )

    fe2_rates = recombination_rates[26, 2]
    assert fe2_rates["log10T_e"].to_list() == pytest.approx(np.arange(1.0, 9.1, 0.1))
    assert fe2_rates["T_e"].to_list() == pytest.approx(10 ** fe2_rates["log10T_e"].to_numpy())
    assert fe2_rates.row(0, named=True) == pytest.approx({
        "log10T_e": 1.0,
        "rrc_low_n": 1.7009e-11,
        "rrc_total": 3.4763e-11,
        "T_e": 10.0,
    })
    assert fe2_rates.row(40, named=True) == pytest.approx({
        "log10T_e": 5.0,
        "rrc_low_n": 9.9265e-13,
        "rrc_total": 7.3507e-12,
        "T_e": 1.0e5,
    })
