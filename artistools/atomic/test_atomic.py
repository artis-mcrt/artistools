import math
import shutil
from pathlib import Path

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


def test_read_transitiondata_truncated(tmp_path: Path) -> None:
    """A truncated ion header raises a Python exception rather than panicking in the extension."""
    truncated = tmp_path / "transitiondata.txt"
    truncated.write_text("26 2\n")

    with pytest.raises(Exception, match="line ended where a transition count was expected"):
        at.rustext.read_transitiondata(truncated)


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


def test_parse_phixsdata_multiple_targets(tmp_path: Path) -> None:
    """A multi-target photoionisation entry must give one structured record per target, like the single-target case."""
    nphixspoints = 3
    lines = [
        str(nphixspoints),
        "0.1",
        # upper ion level -1 means the targets are listed on the following lines
        "26 3 -1 2 1 7.9",
        "2",
        "1 0.75",
        "3 0.25",
        *["1.0"] * nphixspoints,
    ]
    phixsfile = tmp_path / "phixsdata_v2.txt"
    phixsfile.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from artistools.atomic._atomic_core import parse_phixsdata

    phixsdict = parse_phixsdata(phixsfile)

    nptargetlist, phixstable = phixsdict[26, 2, 0]
    # one entry per target, not an (ntargets, 2) grid of duplicated tuples
    assert nptargetlist.shape == (2,)
    assert nptargetlist["level"].tolist() == [0, 2]
    assert nptargetlist["fraction"].tolist() == pytest.approx([0.75, 0.25])
    assert len(phixstable) == nphixspoints


@pytest.mark.parametrize("bflistcontents", ["0\n", "", "0"])
def test_get_bflist_with_no_transitions(tmp_path: Path, bflistcontents: str) -> None:
    """A run with no bound-free transitions must give an empty frame, not fail at a later collect().

    pl.scan_csv is lazy, so an empty bflist.out used to surface as NoDataError from whichever
    collect() consumed the frame, e.g. deep inside the packet query of a --frompackets spectrum.
    """
    shutil.copy(modelpath_classic_3d / "compositiondata.txt", tmp_path)
    (tmp_path / "bflist.out").write_text(bflistcontents, encoding="utf-8")

    dfbflist = at.get_bflist(tmp_path).collect()
    assert dfbflist.is_empty()
    assert set(dfbflist.columns) == {"bfindex", "lowerlevel", "upperionlevel", "atomic_number", "ion_stage"}

    dfbflist_ionstr = at.get_bflist(tmp_path, get_ion_str=True).collect()
    assert dfbflist_ionstr.is_empty()
    assert "ion_str" in dfbflist_ionstr.columns


def test_get_bflist_reads_transitions() -> None:
    """The populated case must be unaffected by the empty-file handling."""
    dfbflist = at.get_bflist(modelpath_classic_3d, get_ion_str=True).collect()

    assert len(dfbflist) == 10780
    assert dfbflist["atomic_number"].sum() == 289060
    assert dfbflist["ion_stage"].sum() == 23822
    assert dfbflist["ion_str"].to_list()[0] == "Fe I"
