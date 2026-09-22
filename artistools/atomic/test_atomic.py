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

    # a level name holds no space. An earlier reader cut the name at the fifth field, thus the name kept
    # the comment of artisatomic that follows the name
    assert fe2_levels.item(0, "levelname") == "3d6(5D)4s_a6De[9/2]"
    assert all(" " not in levelname for levelname in fe2_levels["levelname"])


def test_read_transitiondata_xz_high_preset(tmp_path: Path) -> None:
    """A transition data file compressed with xz -9 declares a 64 MiB dictionary and must still be readable."""
    import lzma

    (tmp_path / "transitiondata.txt.xz").write_bytes(
        lzma.compress((modelpath / "transitiondata.txt").read_bytes(), preset=9)
    )

    transitionsdict = at.rustext.read_transitiondata(tmp_path / "transitiondata.txt.xz")
    assert (26, 2) in transitionsdict


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

    from artistools.atomic.core import parse_phixsdata

    phixsdict = parse_phixsdata(phixsfile)

    nptargetlist, phixstable = phixsdict[26, 2, 0]
    # one entry per target, not an (ntargets, 2) grid of duplicated tuples
    assert nptargetlist.shape == (2,)
    assert nptargetlist["level"].tolist() == [0, 2]
    assert nptargetlist["fraction"].tolist() == pytest.approx([0.75, 0.25])
    assert len(phixstable) == nphixspoints


def test_get_levels_photoionisation_level_alignment() -> None:
    """Each level must carry its own photoionisation data.

    parse_phixsdata() keys on the zero-based level index while adata.txt numbers levels from one, so looking up
    the file's level number attached every level the cross-sections of the level above it, and left the highest
    level with none at all. Neither mismatch raises, so compare against the parsed file directly.
    """
    from artistools.atomic.core import parse_phixsdata

    phixsdict = parse_phixsdata(modelpath / "phixsdata_v2.txt", ionlist=[(26, 1)])
    dflevels = at.atomic.get_levels(modelpath, ionlist=[(26, 1)], get_photoionisations=True)
    levels = dflevels.filter((pl.col("Z") == 26) & (pl.col("ion_stage") == 1)).item(0, "levels")

    assert len(levels) > 1
    for levelindex in range(len(levels)):
        nptargetlist, phixstable = phixsdict[26, 1, levelindex]
        # an object column built from rows would hold a list of numpy scalars instead of the array itself
        assert isinstance(levels.item(levelindex, "phixstable"), np.ndarray)
        assert np.array_equal(levels.item(levelindex, "phixstable"), phixstable)
        assert np.array_equal(levels.item(levelindex, "phixstargetlist"), nptargetlist)


def write_atomic_files_of_two_ions(folder: Path, comments: bool) -> None:
    """Write small atomic data files, with or without the comment blocks that artisatomic writes before each ion."""

    def block(*commentlines: str) -> list[str]:
        return [f"# {line}" for line in commentlines] if comments else []

    nphixspoints = 3
    adatalines: list[str] = []
    transitionlines: list[str] = []
    phixslines = [str(nphixspoints), "0.1"]
    for ion_stage in (1, 2):
        adatalines += [
            *block(f"Z=26 Fe {ion_stage}", "handler: cmfgen", "Reading a file"),
            f"26 {ion_stage} 2 7.9",
            "1 0.0 9.0 1 groundlevel",
            "2 1.5 7.0 1 level_with_a_#_in_its_name",
            "",
        ]
        transitionlines += [
            *block("handler: cmfgen", "   # an indented comment"),
            f"26 {ion_stage} 1",
            "1 2 1.5e+06 0.5 0",
            "",
        ]
        phixslines += [
            *block("handler: cmfgen", "Writing 2 phixs tables"),
            f"26 {ion_stage + 1} 1 {ion_stage} 1 7.9",
            *["1.0"] * nphixspoints,
            # upper ion level -1 means the targets are listed on the following lines
            f"26 {ion_stage + 1} -1 {ion_stage} 2 6.4",
            "2",
            "1 0.75",
            "2 0.25",
            *["2.0"] * nphixspoints,
        ]

    for filename, lines in (
        ("adata.txt", adatalines),
        ("transitiondata.txt", transitionlines),
        ("phixsdata_v2.txt", phixslines),
    ):
        (folder / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("ionlist", [None, [(26, 2)]])
def test_atomic_files_with_comment_lines(tmp_path: Path, ionlist: list[tuple[int, int]] | None) -> None:
    """The comment lines that artisatomic writes before each ion must not change what the readers return.

    With an ionlist, the readers step over the first ion by its line count, so its comment block must not count.
    """
    from artistools.atomic.core import parse_phixsdata

    folders = {}
    for comments in (False, True):
        folders[comments] = tmp_path / f"comments_{comments}"
        folders[comments].mkdir()
        write_atomic_files_of_two_ions(folders[comments], comments=comments)

    dflevels_plain, dflevels_comments = (
        at.atomic.get_levels(folders[comments], ionlist=ionlist, get_transitions=True, get_photoionisations=True)
        for comments in (False, True)
    )
    expected_ions = ionlist or [(26, 1), (26, 2)]
    assert list(zip(dflevels_comments["Z"], dflevels_comments["ion_stage"], strict=True)) == expected_ions
    for row_plain, row_comments in zip(
        dflevels_plain.iter_rows(named=True), dflevels_comments.iter_rows(named=True), strict=True
    ):
        assert row_comments["levels"]["levelname"].to_list() == ["groundlevel", "level_with_a_#_in_its_name"]
        pltest.assert_frame_equal(
            row_comments["levels"].drop("phixstargetlist", "phixstable"),
            row_plain["levels"].drop("phixstargetlist", "phixstable"),
        )
        dftransitions = row_comments["transitions"].collect()
        pltest.assert_frame_equal(dftransitions, row_plain["transitions"].collect())
        assert dftransitions.height == 1

    phixs_plain, phixs_comments = (
        parse_phixsdata(folders[comments] / "phixsdata_v2.txt", ionlist=ionlist) for comments in (False, True)
    )
    assert (
        phixs_comments.keys()
        == phixs_plain.keys()
        == {(26, ion_stage, level) for _, ion_stage in expected_ions for level in (0, 1)}
    )
    for key, (nptargetlist, phixstable) in phixs_comments.items():
        assert np.array_equal(nptargetlist, phixs_plain[key][0])
        assert np.array_equal(phixstable, phixs_plain[key][1])


@pytest.mark.parametrize("bflistcontents", ["0\n", "", "0"])
def test_get_bflist_with_no_transitions(tmp_path: Path, bflistcontents: str) -> None:
    """A run with no bound-free transitions must give an empty frame, not fail at a later collect().

    pl.scan_csv is lazy, so an empty bflist.out used to surface as NoDataError from whichever
    collect() consumed the frame, e.g. deep inside the packet query of a --frompackets spectrum.
    """
    shutil.copy(modelpath_classic_3d / "compositiondata.txt", tmp_path)
    (tmp_path / "bflist.out").write_text(bflistcontents, encoding="utf-8")

    dfbflist = at.atomic.get_bflist(tmp_path).collect()
    assert dfbflist.is_empty()
    assert {"bfindex", "lowerlevel", "upperionlevel", "atomic_number", "ion_stage", "ion_str"} <= set(dfbflist.columns)


def test_get_bflist_reads_transitions() -> None:
    """The populated case must be unaffected by the empty-file handling."""
    dfbflist = at.atomic.get_bflist(modelpath_classic_3d).collect()

    assert len(dfbflist) == 10780
    assert dfbflist["atomic_number"].sum() == 289060
    assert dfbflist["ion_stage"].sum() == 23822
    assert dfbflist["ion_str"].to_list()[0] == "Fe I"


def test_the_cached_photoionisation_arrays_refuse_a_write() -> None:
    """A caller must not change the cross sections that the cache holds.

    A clone of a frame shares the numpy array of an object column. A copy of those arrays takes 10 ms
    for each call of this small model, against 0.05 ms for the call itself, thus the arrays carry the
    read-only mark instead and such a write raises.
    """
    import numpy as np

    dflevels = at.atomic.get_levels(modelpath, get_photoionisations=True)
    arrays = [
        value
        for dfions in dflevels["levels"]
        for colname in ("phixstargetlist", "phixstable")
        if colname in dfions.columns
        for value in dfions[colname]
        if isinstance(value, np.ndarray) and value.size > 0
    ]
    assert arrays, "the test model must hold photoionisation arrays"

    with pytest.raises(ValueError, match="read-only"):
        arrays[0][0] = 0.0
