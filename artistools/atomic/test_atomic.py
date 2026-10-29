import lzma
import math
import os
import shutil
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at
from artistools.atomic.core import read_transitiondata
from artistools.atomic.core import vacuum_to_air_angstroms
from artistools.constants import angstroms_per_cm
from artistools.misc import get_decompress_open
from artistools.misc import write_zstd_lines

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
    """A transition data file compressed with xz -9 declares a 64 MiB dictionary and must still be readable.

    A file of a few lines declares the same dictionary. A compression of the full 73 MB file took minutes.
    """
    import itertools
    import lzma

    # the first three transitions of Fe II, which start on line 141966 of the file of the test model
    with (modelpath / "transitiondata.txt").open(encoding="utf-8") as ftransitions:
        fe2lines = list(itertools.islice(ftransitions, 141965, 141968))
    (tmp_path / "transitiondata.txt.xz").write_bytes(lzma.compress(("26 2 3\n" + "".join(fe2lines)).encode(), preset=9))

    transitionsdict = read_transitiondata(tmp_path / "transitiondata.txt.xz")
    assert transitionsdict.keys() == {(26, 2)}
    assert transitionsdict[26, 2].shape == (3, 5)
    assert transitionsdict[26, 2].row(0, named=True) == pytest.approx({
        "lower": 0,
        "upper": 1,
        "A": 0.00209,
        "collstr": 3.23,
        "forbidden": 1,
    })


def test_read_transitiondata() -> None:
    transitionsdict = read_transitiondata(modelpath / "transitiondata.txt")
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
    selectedions = read_transitiondata(modelpath / "transitiondata.txt", ionlist={(26, 2)})
    assert selectedions.keys() == {(26, 2)}
    pltest.assert_frame_equal(selectedions[26, 2], dftransitions)


def test_read_transitiondata_truncated(tmp_path: Path) -> None:
    """A truncated ion header raises a Python exception rather than panicking in the extension."""
    truncated = tmp_path / "transitiondata.txt"
    truncated.write_text("26 2\n")

    with pytest.raises(Exception, match="line ended where a transition count was expected"):
        read_transitiondata(truncated)


@pytest.mark.parametrize("ionlist", [None, {(26, 2)}, {(27, 2)}])
def test_read_transitiondata_rejects_a_table_that_the_file_ends_inside(
    tmp_path: Path, ionlist: set[tuple[int, int]] | None
) -> None:
    """A transition table with fewer lines than its header gives must be an error, for a kept and for a skipped ion.

    A cut compressed file decodes with no error, thus the reader gave the short table in silence.
    """
    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text("26 2 5\n1 2 1.0 2.0 1\n1 3 1.0 2.0 1\n", encoding="utf-8")

    with pytest.raises(
        Exception, match=r"transitiondata\.txt: the file ends after 2 of the 5 transitions of Z=26 ion_stage=2"
    ):
        read_transitiondata(transitionsfile, ionlist=ionlist)

    # a complete table of the same ion is not an error
    transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 2.0 1\n", encoding="utf-8")
    transitionsdict = read_transitiondata(transitionsfile, ionlist=ionlist)
    assert [df.height for df in transitionsdict.values()] == ([] if ionlist == {(27, 2)} else [2])

    # ARTIS reads a complete file whose last line has no newline. The reader rejected such a file as cut
    transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 2.0 1", encoding="utf-8")
    transitionsdict = read_transitiondata(transitionsfile, ionlist=ionlist)
    assert [df.height for df in transitionsdict.values()] == ([] if ionlist == {(27, 2)} else [2])

    # a cut inside the last line of a kept table leaves too few numbers for the format of the table
    if ionlist != {(27, 2)}:
        transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 7.6", encoding="utf-8")
        with pytest.raises(Exception, match=r"transitiondata\.txt:3: line ended where a forbidden flag was expected"):
            read_transitiondata(transitionsfile, ionlist=ionlist)


def test_read_transitiondata_reads_the_legacy_format_of_four_columns(tmp_path: Path) -> None:
    """ARTIS takes the format of each table from its first line, and 4 numbers give "index lower upper A".

    The reader took such a table as "lower upper A collstr", thus it gave incorrect levels and A values with no error.
    ARTIS gives a transition of this format the collision strength -1 and the forbidden flag 0.
    """
    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text(
        "26 2 3\n1 1 2 1.5e8\n2 1 3 2.5e7\n3 2 3 4.0e6\n\n26 3 1\n1 2 2.0 0.5 1\n", encoding="utf-8"
    )
    transitionsdict = read_transitiondata(transitionsfile)
    pltest.assert_frame_equal(
        transitionsdict[26, 2],
        pl.DataFrame(
            {
                "lower": [0, 0, 1],
                "upper": [1, 2, 2],
                "A": [1.5e8, 2.5e7, 4.0e6],
                "collstr": [-1.0] * 3,
                "forbidden": [0] * 3,
            },
            schema={
                "lower": pl.Int32,
                "upper": pl.Int32,
                "A": pl.Float32,
                "collstr": pl.Float32,
                "forbidden": pl.Int32,
            },
        ),
    )
    assert transitionsdict[26, 3].row(0) == pytest.approx((0, 1, 2.0, 0.5, 1))

    # ARTIS reads no other count of numbers
    transitionsfile.write_text("26 2 1\n1 2 1.0\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:2: the first line of a table has 3 numbers"):
        read_transitiondata(transitionsfile)


def test_read_transitiondata_errors_name_the_file_and_the_line(tmp_path: Path) -> None:
    """A parse error names the file and the line. The test model file has 1.9 million lines."""
    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text("26 2 3\n1 2 1.0 2.0 1\n1 3 1.0e-3x 2.0 1\n2 3 1.0 2.0 1\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:3: could not parse \"1\.0e-3x\" as an A value"):
        read_transitiondata(transitionsfile)

    transitionsfile.write_text("26 2 3\n1 2 1.0 2.0 1\n\n2 3 1.0 2.0 1\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:3: line ended where a lower level was expected"):
        read_transitiondata(transitionsfile)

    transitionsfile.write_text("26 x 3\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:1: could not parse \"x\" as an ion stage"):
        read_transitiondata(transitionsfile)


def test_read_transitiondata_takes_the_format_from_the_leading_numbers_and_reports_the_first_bad_line(
    tmp_path: Path,
) -> None:
    """The format of a table comes from the leading numbers of its first line, and not from its count of words.

    The reader parses all the lines in one query. It must report the first bad line of the file, as ARTIS does.
    """
    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text(
        "26 2 2\n1 2 1.0 2.0 1 # a comment\n1 3 1.0 2.0 1\n\n26 3 1\n1 1 2 1.0e8 # a comment\n", encoding="utf-8"
    )
    transitionsdict = read_transitiondata(transitionsfile)
    assert transitionsdict[26, 2]["upper"].to_list() == [1, 2]
    assert transitionsdict[26, 3].row(0) == pytest.approx((0, 1, 1.0e8, -1.0, 0))

    transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 x 1\n\n26 x 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"transitiondata\.txt:3: could not parse \"x\" as a collision strength"):
        read_transitiondata(transitionsfile)


def test_level_numbers_that_start_at_zero(tmp_path: Path) -> None:
    """ARTIS takes the number of the first level, 0 or 1, from adata.txt, and it applies it to all three files.

    The readers took the numbering from 1. Thus a zero-based transition gave the lower level -1, a single target
    level 0 of a photoionisation table started a list of targets, and adata.txt gave a bare AssertionError.
    """
    from artistools.atomic.core import parse_adata
    from artistools.atomic.core import parse_phixsdata

    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text("26 1 2\n0 1 1.0 2.0 1\n0 2 1.0 2.0 1\n", encoding="utf-8")
    dftransitions = read_transitiondata(transitionsfile, firstlevelnumber=0)[26, 1]
    assert dftransitions["lower"].to_list() == [0, 0]
    assert dftransitions["upper"].to_list() == [1, 2]
    with pytest.raises(Exception, match="ARTIS numbers the levels from 0 or from 1, not from 2"):
        read_transitiondata(transitionsfile, firstlevelnumber=2)

    phixsfile = tmp_path / "phixsdata_v2.txt"
    phixsfile.write_text(
        "2\n0.1\n26 2 0 1 1 7.9\n1.0\n1.0\n26 2 -1 1 0 7.9\n2\n0 0.75\n1 0.25\n2.0\n2.0\n", encoding="utf-8"
    )
    phixsdict = parse_phixsdata(phixsfile, firstlevelnumber=0)
    assert phixsdict.keys() == {(26, 1, 0), (26, 1, 1)}
    assert phixsdict[26, 1, 1][0]["level"].tolist() == [0]
    assert phixsdict[26, 1, 0][0]["level"].tolist() == [0, 1]

    adatafile = tmp_path / "adata.txt"
    adatafile.write_text("26 1 3 7.9\n0 0.0 9.0 1 ground\n1 1.5 7.0 1 first\n2 2.5 5.0 1 second\n", encoding="utf-8")
    with adatafile.open(encoding="utf-8") as fadata:
        ions = list(parse_adata(fadata, {}, None, firstlevelnumber=0))
    assert ions[0][4]["levelindex"].to_list() == [0, 1, 2]
    with (
        adatafile.open(encoding="utf-8") as fadata,
        pytest.raises(ValueError, match="the first level of the ion has the number 0, but the numbering from 1"),
    ):
        list(parse_adata(fadata, {}, None))

    adatafile.write_text("26 1 3 7.9\n1 0.0 9.0 1 ground\n2 1.5 7.0 1 first\n4 2.5 5.0 1 second\n", encoding="utf-8")
    with (
        adatafile.open(encoding="utf-8") as fadata,
        pytest.raises(ValueError, match="level 3 of the ion has the number 4"),
    ):
        list(parse_adata(fadata, {}, None))


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
def test_scan_bflist_with_no_transitions(tmp_path: Path, bflistcontents: str) -> None:
    """A model with no bound-free transitions must give an empty frame, not fail at a later collect().

    pl.scan_csv is lazy, so an empty bflist.out used to surface as NoDataError from whichever
    collect() consumed the frame, e.g. deep inside the packet query of a --frompackets spectrum.
    """
    shutil.copy(modelpath_classic_3d / "compositiondata.txt", tmp_path)
    (tmp_path / "bflist.out").write_text(bflistcontents, encoding="utf-8")

    dfbflist = at.atomic.scan_bflist(tmp_path).collect()
    assert dfbflist.is_empty()
    assert {"bfindex", "lowerlevel", "upperionlevel", "atomic_number", "ion_stage", "ion_str"} <= set(dfbflist.columns)


def test_scan_bflist_reads_transitions() -> None:
    """The populated case must be unaffected by the empty-file handling."""
    dfbflist = at.atomic.scan_bflist(modelpath_classic_3d).collect()

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


def get_composition_z(folder: Path) -> int:
    """Return the atomic number of the one element of compositiondata.txt in the folder."""
    return int(at.get_composition_data(folder)["Z"].item())


def get_ground_level_energy(folder: Path) -> float:
    """Return the energy of the ground level of the one ion of adata.txt in the folder."""
    return float(at.atomic.get_levels(folder)["levels"].item()["energy_ev"].item())


@pytest.mark.parametrize(
    ("filename", "filetext", "readvalue"),
    [
        ("compositiondata.txt", "1\n0\n0\n{value} 2 1 2 300 1.0 56.0\n", get_composition_z),
        ("adata.txt", "26 1 1 7.9\n1 {value} 9.000 0 ground\n", get_ground_level_energy),
    ],
)
def test_atomic_data_follows_the_working_folder(
    filename: str, filetext: str, readvalue: Callable[[Path], float], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The atomic data of the default model path must change with the working folder.

    A cache held the relative Path("."). Thus a second model in the same process got the data of the first one.
    """
    for foldername, value in (("modelA", 26), ("modelB", 28)):
        (tmp_path / foldername).mkdir()
        (tmp_path / foldername / filename).write_text(filetext.format(value=value), encoding="utf-8")

    monkeypatch.chdir(tmp_path / "modelA")
    assert readvalue(Path()) == pytest.approx(26)
    monkeypatch.chdir(tmp_path / "modelB")
    assert readvalue(Path()) == pytest.approx(28)


def test_get_ion_levels_shares_the_cache_entry_of_get_levels(tmp_path: Path) -> None:
    """A repeated ion and the same ion through get_levels must use one cache entry, and parse adata.txt once.

    A copy of the atomic data has a path that no other test reads. Thus no other test of the same worker can put
    the entry in the cache first, and the count of misses is exact.
    """
    from artistools.atomic.core import get_levels_cached

    (tmp_path / "adata.txt.xz").symlink_to(modelpath / "adata.txt.xz")
    missesbefore = get_levels_cached.cache_info().misses
    dffe2 = at.atomic.get_ion_levels(tmp_path, 26, 2)
    assert dffe2 is not None
    assert at.atomic.get_ion_levels(tmp_path, 26, 2) is not None
    dflevels = at.atomic.get_levels(tmp_path, ionlist=[(26, 2)])
    assert get_levels_cached.cache_info().misses == missesbefore + 1
    assert dffe2.height == len(dflevels["levels"].item())
    assert at.atomic.get_ion_levels(tmp_path, 1, 1) is None


def test_get_levels_stops_at_an_adata_file_that_ends_inside_an_ion(tmp_path: Path) -> None:
    """A file that ends inside the levels of an ion must give an error, also for an ion that the caller skips.

    The parser read past the end of the file for a skipped ion, thus the ions after the cut went with no message.
    """
    (tmp_path / "adata.txt").write_text("26 1 2 7.9\n1 0.0 9.000 0 ground\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ends inside the levels of Z=26 ion_stage=1"):
        at.atomic.get_levels(tmp_path)
    with pytest.raises(ValueError, match="ends inside the levels of Z=26 ion_stage=1"):
        at.atomic.get_levels(tmp_path, ionlist=[(28, 2)])


def test_get_composition_data_ignores_comments(tmp_path: Path) -> None:
    """ARTIS ignores the text to the right of a # character, and a line that holds only a comment.

    The reader read fixed lines with int(), thus a comment stopped it with a ValueError.
    """
    (tmp_path / "compositiondata.txt").write_text(
        "# the elements of the model\n2  # number of elements\n0\n0\n\n26 5 1 5 500 0.0 55.845  # iron\n"
        "# cobalt follows\n27 3 2 4 -1 0.0 58.9332\n",
        encoding="utf-8",
    )

    dfcomposition = at.get_composition_data(tmp_path)
    assert dfcomposition["Z"].to_list() == [26, 27]
    assert dfcomposition["nlevelsmax_readin"].to_list() == [500, -1]
    assert dfcomposition["mass"].to_list() == pytest.approx([55.845, 58.9332])

    (tmp_path / "short").mkdir()
    (tmp_path / "short" / "compositiondata.txt").write_text("2\n0\n0\n26 5 1 5 500 0.0 55.845\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seven numbers for each element"):
        at.get_composition_data(tmp_path / "short")


@pytest.mark.parametrize(
    "loglines",
    [
        ["[input.c]   element Z = 26", "[input.c]     ion 1 with 500 levels", "[input.c]     ion 2 with 400 levels"],
        [
            "2023-11-14T15:01:58Z [input]  element 0 (Z=26 Fe)",
            "2023-11-14T15:01:58Z [input]    ionstage 1:  500 levels (1 in groundterm,  500 ionising)",
            "2023-11-14T15:01:58Z [input]    ionstage 2:  400 levels (1 in groundterm,  400 ionising)",
        ],
        [
            "2026-10-02T10:00:00Z [info]  element 0 (Z=26 Fe)",
            "2026-10-02T10:00:00Z [info]    ionstage 1:  500 levels ( 500 ionising)",
            "2026-10-02T10:00:00Z [info]    ionstage 2:  400 levels ( 400 ionising)",
        ],
    ],
)
def test_get_composition_data_from_outputfile_reads_each_log_format(loglines: list[str], tmp_path: Path) -> None:
    """The log of a run lists the elements and the ion stages, in the form of the version of ARTIS.

    The reader took only the classic form, thus it gave no element for the log of a later run.
    """
    (tmp_path / "output_0-0.txt").write_text(
        "\n".join(["start of the log", *loglines, "end of the list", "[info]  element 9 (Z=28 Ni)"]) + "\n",
        encoding="utf-8",
    )

    dfcomposition = at.atomic.get_composition_data_from_outputfile(tmp_path)
    assert dfcomposition.select("Z", "lowermost_ion_stage", "uppermost_ion_stage", "nions").rows() == [(26, 1, 2, 2)]


def test_get_composition_data_from_outputfile_of_the_test_models() -> None:
    """The log of each test model gives the elements and the ion stages of its compositiondata.txt."""
    for testmodelpath in (modelpath_classic_3d, at.get_path("testdata") / "test-classicmode_1d"):
        columns = ["Z", "lowermost_ion_stage", "uppermost_ion_stage", "nions"]
        pltest.assert_frame_equal(
            at.atomic.get_composition_data_from_outputfile(testmodelpath).select(columns),
            at.get_composition_data(testmodelpath).select(columns),
        )


def test_get_composition_data_from_outputfile_refuses_a_log_with_no_list(tmp_path: Path) -> None:
    """A log with no list of the elements gives a message, and not a frame of no elements."""
    (tmp_path / "output_0-0.txt").write_text("start of the log\n", encoding="utf-8")

    with pytest.raises(ValueError, match="holds no list of the elements"):
        at.atomic.get_composition_data_from_outputfile(tmp_path)


LINELIST_HEADER = "#lambda_angstroms atomic_number ion_stage upperlevelindex lowerlevelindex"
# each tuple is one line: the vacuum wavelength [Angstrom], the atomic number, the ion stage, the zero-based upper
# level index, and the zero-based lower level index, in the order of the columns of linelist.out
TEST_LINES = ((13898.2958, 57, 3, 1, 0), (4216.708, 38, 2, 2, 0), (6500.0, 26, 2, 4, 0))


def write_text_rows(filepath: Path, rows: Sequence[str]) -> None:
    """Write the rows to a text file, or to a zstd file if the name ends with .zst."""
    if filepath.suffix == ".zst":
        with filepath.open("wb") as fileobj:
            write_zstd_lines(fileobj, rows)
    else:
        filepath.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def write_linelistfile(filepath: Path, lines: Sequence[tuple[float, int, int, int, int]]) -> None:
    """Write the lines as ARTIS writes linelist.out: a header row, then one row for each line."""
    write_text_rows(filepath, [LINELIST_HEADER, *(" ".join(str(value) for value in line) for line in lines)])


def write_linestatfile(filepath: Path, lines: Sequence[tuple[float, int, int, int, int]]) -> None:
    """Write the lines as an older ARTIS writes linestat.out: one row for each column, with the wavelength in cm.

    linestat.out gives one-based level numbers, thus the function adds 1 to each zero-based level index.
    """
    columns = [
        [f"{line[0] / angstroms_per_cm:.8e}" for line in lines],
        *([str(line[column]) for line in lines] for column in (1, 2)),
        *([str(line[column] + 1) for line in lines] for column in (3, 4)),
    ]
    write_text_rows(filepath, [" ".join(column) for column in columns])


@pytest.mark.parametrize(("nlines", "suffix"), [(0, ""), (1, ""), (3, ""), (3, ".zst")])
# numpy warns for the five empty rows of linestat.out with no lines
@pytest.mark.filterwarnings("error::UserWarning")
def test_scan_linelist_reads_linelist_and_linestat(nlines: int, suffix: str, tmp_path: Path) -> None:
    """linelist.out and linestat.out with the same lines give the same frame.

    linestat.out holds one column for each line, thus a file of one line gave a one-dimensional array. The reader then
    stopped with a TypeError, and a file of no line stopped it with an IndexError.
    """
    lines = TEST_LINES[:nlines]
    (tmp_path / "linelist").mkdir()
    write_linelistfile(tmp_path / "linelist" / f"linelist.out{suffix}", lines)
    (tmp_path / "linestat").mkdir()
    write_linestatfile(tmp_path / "linestat" / f"linestat.out{suffix}", lines)

    dflinelist = at.atomic.scan_linelist(tmp_path / "linelist").collect()
    assert dflinelist.height == nlines
    # the float32 step is approximately 0.001 Angstroms at 14000 Angstroms. 6 significant digits gave errors of up to
    # 0.05 Angstroms
    assert np.allclose(dflinelist["lambda_angstroms"], [line[0] for line in lines], rtol=0.0, atol=1e-3)
    assert dflinelist["atomic_number"].to_list() == [line[1] for line in lines]
    assert dflinelist["ion_stage"].to_list() == [line[2] for line in lines]
    assert dflinelist["upperlevelindex"].to_list() == [line[3] for line in lines]
    assert dflinelist["lowerlevelindex"].to_list() == [line[4] for line in lines]
    pltest.assert_frame_equal(dflinelist, at.atomic.scan_linelist(tmp_path / "linestat").collect())


def test_scan_linelist_reads_the_columns_by_name(tmp_path: Path) -> None:
    """The header row gives the order of the columns, and the reader ignores a column that it does not know."""
    (tmp_path / "linelist.out").write_text(
        "#lambda_angstroms atomic_number ion_stage lowerlevelindex upperlevelindex einstein_a\n"
        "13898.2958 57 3 0 1 1e8\n",
        encoding="utf-8",
    )

    dflinelist = at.atomic.scan_linelist(tmp_path).collect()
    assert dflinelist.select("atomic_number", "upperlevelindex", "lowerlevelindex").row(0) == (57, 1, 0)


def test_scan_linelist_ignores_a_row_without_values(tmp_path: Path) -> None:
    """An edit can add an empty line. It holds no line, thus it must not give the error for a cut line."""
    write_text_rows(tmp_path / "linelist.out", [LINELIST_HEADER, "13898.2958 57 3 1 0", "", "4216.708 38 2 2 0", ""])

    assert at.atomic.scan_linelist(tmp_path).collect()["atomic_number"].to_list() == [57, 38]


@pytest.mark.parametrize("nlines", [0, 3])
# numpy warns for the empty rows of linestat.out with no lines
@pytest.mark.filterwarnings("error::UserWarning")
def test_scan_linelist_reads_linestat_with_rows_of_counts(nlines: int, tmp_path: Path) -> None:
    """An ARTIS before March 2026 with RECORD_LINESTAT adds two rows of counts at each timestep after the five rows."""
    write_linestatfile(tmp_path / "linestat.out", TEST_LINES[:nlines])
    with (tmp_path / "linestat.out").open("a", encoding="utf-8") as linestatfile:
        linestatfile.write(("".join(f"{count} " for count in range(nlines)) + "\n") * 4)

    dflinelist = at.atomic.scan_linelist(tmp_path).collect()
    assert dflinelist["upperlevelindex"].to_list() == [line[3] for line in TEST_LINES[:nlines]]


def test_scan_linelist_reads_a_wavelength_after_many_integer_wavelengths(tmp_path: Path) -> None:
    """ARTIS writes a wavelength such as 6500 Angstroms with no decimal point.

    polars infers the type of a column from the first rows, thus a column of integers stopped the read at 6500.5.
    """
    write_linelistfile(tmp_path / "linelist.out", [*([(6500, 26, 2, 4, 0)] * 200), (6500.5, 26, 2, 4, 0)])

    dflinelist = at.atomic.scan_linelist(tmp_path).collect()
    assert dflinelist.height == 201
    assert np.isclose(dflinelist["lambda_angstroms"][-1], 6500.5)


def test_scan_linelist_reads_only_the_model_folder(tmp_path: Path) -> None:
    """ARTIS writes linelist.out and linestat.out into the model folder only, thus a copy in a run folder is ignored."""
    (tmp_path / "123.slurm").mkdir()
    write_linelistfile(tmp_path / "123.slurm" / "linelist.out", TEST_LINES)
    with pytest.raises(FileNotFoundError, match="in the model folder"):
        at.atomic.scan_linelist(tmp_path)

    write_linestatfile(tmp_path / "linestat.out", TEST_LINES[:1])
    assert at.atomic.scan_linelist(tmp_path).collect().height == 1


def test_scan_linelist_refuses_the_cache_of_the_other_file(tmp_path: Path) -> None:
    """A cache of one text file is not current for a different file, also if the two files have the same time stamp."""
    write_linestatfile(tmp_path / "linestat.out", TEST_LINES[:1])
    assert at.atomic.scan_linelist(tmp_path).collect().height == 1
    firstmtime = (tmp_path / "linestat.out").stat().st_mtime
    (tmp_path / "linestat.out").unlink()
    write_linelistfile(tmp_path / "linelist.out", TEST_LINES)
    os.utime(tmp_path / "linelist.out", (firstmtime, firstmtime))

    assert at.atomic.scan_linelist(tmp_path).collect().height == 3


def test_scan_linelist_keeps_the_cache_after_a_compression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A compression with zstd keeps the modification time, thus the cache of the text must stay current."""
    write_linelistfile(tmp_path / "linelist.out", TEST_LINES)
    at.atomic.scan_linelist(tmp_path).collect()
    mtime = (tmp_path / "linelist.out").stat().st_mtime
    (tmp_path / "linelist.out").unlink()
    write_linelistfile(tmp_path / "linelist.out.zst", TEST_LINES)
    os.utime(tmp_path / "linelist.out.zst", (mtime, mtime))
    capsys.readouterr()

    assert at.atomic.scan_linelist(tmp_path).collect().height == 3
    assert "linelist.out.zst" not in capsys.readouterr().out


NO_LINE_END_MESSAGE = "is empty or does not end with a line end"


@pytest.mark.parametrize(
    ("filename", "content", "expectedmessage"),
    [
        ("linelist.out", f"{LINELIST_HEADER}\n13898.2958 57 3 1 0\n4216.708 38 2 11 1".encode(), NO_LINE_END_MESSAGE),
        (
            "linelist.out",
            f"{LINELIST_HEADER}\n13898.2958 57 3 1 0\n4216.708 38 2\n".encode(),
            "has a line with fewer values than the header",
        ),
        # zstd with worker threads writes no byte of a small file until the file closes
        ("linelist.out.zst", b"", NO_LINE_END_MESSAGE),
        ("linestat.out", b"", NO_LINE_END_MESSAGE),
        ("linestat.out", b"\n\n", "has 2 rows and not 5"),
        ("linestat.out", b"1.38982958e-04 4.21670800e-05\n57 38\n3 2\n", "has 3 rows and not 5"),
        ("linestat.out", b"1.38982958e-04 4.21670800e-05\n57 38\n3 2\n2 3\n1 1", NO_LINE_END_MESSAGE),
        # the wavelength row is not first, thus the atomic numbers have a fraction
        ("linestat.out", b"57 38\n1.38982958e-04 4.21670800e-05\n3 2\n2 3\n1 1\n", "is not an integer in the Int32"),
        # a level number of 0 gives the level index -1
        ("linestat.out", b"1.38982958e-04\n57\n3\n2\n0\n", "a level index below 0"),
        ("linelist.out", f"{LINELIST_HEADER}\n13898.2958 57 3 1 -1\n".encode(), "a level index below 0"),
        (
            "linelist.out",
            b"#lambda_angstroms atomic_number ion_stage upper_level lower_level\n13898.2958 57 3 2 1\n",
            "has no column upperlevelindex, lowerlevelindex",
        ),
        ("linestat.out", b"1.38982958e-04\n57\nnan\n2\n1\n", "is not an integer in the Int32 range"),
        ("linelist.out", f"# {LINELIST_HEADER[1:]}\n13898.2958 57 3 1 0\n".encode(), "has a column with no name"),
        ("linelist.out", f"{LINELIST_HEADER}\n13898.2958 57.0 3 1 0\n".encode(), "does not agree with the type"),
        ("linelist.out", f"{LINELIST_HEADER}\n13898.2958 57 3 1 0 \n".encode(), "has a row with more values"),
        (
            "linelist.out",
            f"{LINELIST_HEADER} einstein_a\n13898.2958 57 3 1 0 1e8\n4216.708 38 2 2 0\n".encode(),
            "has a line with fewer values than the header",
        ),
    ],
)
def test_scan_linelist_refuses_a_file_that_is_not_complete(
    filename: str, content: bytes, expectedmessage: str, tmp_path: Path
) -> None:
    """ARTIS or a copy can still write the file. A cache of the partial file stays current after the write ends."""
    (tmp_path / filename).write_bytes(content)

    with pytest.raises(ValueError, match=expectedmessage):
        at.atomic.scan_linelist(tmp_path)
    assert not (tmp_path / "linelist.out.parquet.tmp").exists()


def test_scan_linelist_refuses_a_compressed_file_that_ends_inside_a_row(tmp_path: Path) -> None:
    """A zstd file of two frames decompresses with no error, also if the second frame ends inside a row."""
    write_linelistfile(tmp_path / "linelist.out.zst", TEST_LINES[:1])
    with get_decompress_open(".zst")(tmp_path / "linelist.out.zst", mode="ab") as fout:
        fout.write(b"4216.708 38 2 2 0")

    with pytest.raises(ValueError, match=NO_LINE_END_MESSAGE):
        at.atomic.scan_linelist(tmp_path)


def test_polars_error_note_names_a_damaged_xz_file(tmp_path: Path) -> None:
    """The lzma module raises LZMAError, which is not an OSError, thus the error did not name the file."""
    (tmp_path / "linelist.out.xz").write_bytes(b"data that is not xz")

    with pytest.raises(lzma.LZMAError) as excinfo:
        at.atomic.scan_linelist(tmp_path)
    assert any("linelist.out.xz" in note for note in excinfo.value.__notes__)


def test_scan_linelist_reads_a_folder_name_with_glob_characters(tmp_path: Path) -> None:
    """Polars reads a path as a glob pattern by default, thus the folder model[1] gave the lines of model1."""
    (tmp_path / "model[1]").mkdir()
    write_linelistfile(tmp_path / "model[1]" / "linelist.out", TEST_LINES)
    (tmp_path / "model1").mkdir()
    write_linelistfile(tmp_path / "model1" / "linelist.out", TEST_LINES[:1])

    for _ in range(2):
        # the first call writes the cache, and the second call reads it
        assert at.atomic.scan_linelist(tmp_path / "model[1]").collect().height == 3


def make_zero_based_atomic_data(folder: Path, atomic_number: int, ion_stage: int) -> None:
    """Write a copy of the atomic data of one ion of the test model, with level numbers that start at 0."""
    import lzma

    folder.mkdir()
    adatalines = lzma.decompress((modelpath / "adata.txt.xz").read_bytes()).decode().splitlines()
    for linenumber, line in enumerate(adatalines):
        header = line.split()
        if len(header) == 4 and header[:2] == [str(atomic_number), str(ion_stage)]:
            levellines = adatalines[linenumber + 1 : linenumber + 1 + int(header[2])]
            break
    (folder / "adata.txt").write_text(
        "\n".join([
            "# a comment line in front of the first ion",
            line,
            *(
                f"{int(levelline.split(maxsplit=1)[0]) - 1} {levelline.split(maxsplit=1)[1]}"
                for levelline in levellines
            ),
        ])
        + "\n",
        encoding="utf-8",
    )

    transitionlines: list[str] = []
    with (modelpath / "transitiondata.txt").open(encoding="utf-8") as ftransitions:
        for line in ftransitions:
            header = line.split()
            if len(header) == 3 and header[:2] == [str(atomic_number), str(ion_stage)]:
                transitionlines.append(line)
                for _ in range(int(header[2])):
                    lower, upper, rest = next(ftransitions).split(maxsplit=2)
                    transitionlines.append(f"{int(lower) - 1} {int(upper) - 1} {rest.rstrip()}\n")
                break
    (folder / "transitiondata.txt").write_text("".join(transitionlines), encoding="utf-8")

    phixslines = iter(lzma.decompress((modelpath / "phixsdata_v2.txt.xz").read_bytes()).decode().splitlines())
    nphixspoints = int(next(phixslines))
    outlines = [str(nphixspoints), next(phixslines)]
    for line in phixslines:
        fields = line.split()
        ntargets = 0
        # a single target level and the lower level take the zero-based number. -1 marks a list of targets
        if int(fields[2]) < 0:
            ntargets = int(next(phixslines))
        targetlines = [next(phixslines) for _ in range(ntargets)]
        tablelines = [next(phixslines) for _ in range(nphixspoints)]
        if (int(fields[0]), int(fields[3])) == (atomic_number, ion_stage):
            upperlevel = int(fields[2]) - 1 if int(fields[2]) > 0 else -1
            outlines.append(f"{fields[0]} {fields[1]} {upperlevel} {fields[3]} {int(fields[4]) - 1} {fields[5]}")
            if ntargets:
                outlines.append(str(ntargets))
                outlines.extend(f"{int(target.split()[0]) - 1} {target.split()[1]}" for target in targetlines)
            outlines.extend(tablelines)
    (folder / "phixsdata_v2.txt").write_text("\n".join(outlines) + "\n", encoding="utf-8")


def test_get_levels_of_atomic_data_numbered_from_zero(tmp_path: Path) -> None:
    """ARTIS takes the number of the first level from adata.txt, and it applies it to the other two files.

    get_levels took the numbering from 1 for each file, thus a zero-based copy gave the wrong level indices.
    """
    ion = (27, 3)
    make_zero_based_atomic_data(tmp_path / "zerobased", *ion)

    assert at.atomic.core.get_first_level_number(tmp_path / "zerobased") == 0
    assert at.atomic.core.get_first_level_number(modelpath) == 1

    zerobased, onebased = (
        at.atomic.get_levels(folder, ionlist=[ion], get_transitions=True, get_photoionisations=True, quiet=True).row(
            0, named=True
        )
        for folder in (tmp_path / "zerobased", modelpath)
    )

    levelcolumns = ["levelindex", "energy_ev", "g", "transition_count", "levelname"]
    pltest.assert_frame_equal(zerobased["levels"].select(levelcolumns), onebased["levels"].select(levelcolumns))
    pltest.assert_frame_equal(zerobased["transitions"].collect(), onebased["transitions"].collect())
    assert zerobased["transitions"].collect()["lower"].min() == 0

    for column in ("phixstargetlist", "phixstable"):
        for zerovalue, onevalue in zip(zerobased["levels"][column], onebased["levels"][column], strict=True):
            assert (zerovalue is None) == (onevalue is None)
            if zerovalue is not None:
                assert zerovalue.tolist() == onevalue.tolist()


def test_vacuum_to_air_angstroms_agrees_with_nist() -> None:
    """The air wavelengths agree with the Ritz wavelengths of Sr II in the NIST Atomic Spectra Database.

    At and outside the limits of 2000 Å and 20000 Å, the function gives the vacuum wavelength, as NIST does.
    """
    nist_vacuum_and_air_angstroms = [
        (2153.528, 2152.850),
        (4078.8607, 4077.7094),
        (4216.7067, 4215.5193),
        (10039.405, 10036.653),
        (10917.877, 10914.887),
        (1500.0, 1500.0),
        (2000.0, 2000.0),
        (20000.0, 20000.0),
        (25000.0, 25000.0),
    ]
    dfwavelengths = pl.DataFrame(
        nist_vacuum_and_air_angstroms, schema=["lambda_vacuum_angstroms", "nist_lambda_air_angstroms"], orient="row"
    ).with_columns(lambda_air_angstroms=vacuum_to_air_angstroms(pl.col("lambda_vacuum_angstroms")))

    assert np.allclose(
        dfwavelengths["lambda_air_angstroms"], dfwavelengths["nist_lambda_air_angstroms"], rtol=0.0, atol=2e-3
    )


def test_get_z_a_nucname() -> None:
    assert at.get_z_a_nucname("Pb208") == (82, 208)
    assert at.get_z_a_nucname("X_Pb208") == (82, 208)
    assert at.get_z_a_nucname("nniso_Pb208") == (82, 208)
    assert at.get_z_a_nucname("Fe56") == (26, 56)
    assert at.get_z_a_nucname("Ni56") == (28, 56)
    assert at.get_z_a_nucname("Co56") == (27, 56)
    assert at.get_z_a_nucname("H1") == (1, 1)


def test_get_atomic_number_and_elsymbol() -> None:
    assert at.get_atomic_number("Fe") == 26
    assert at.get_atomic_number("Ni") == 28
    assert at.get_atomic_number("Co") == 27
    assert at.get_atomic_number("H") == 1
    assert at.get_atomic_number("He") == 2
    assert at.get_atomic_number("X_Fe") == 26
    assert at.get_atomic_number("UnknownXYZ") == -1

    # the free neutron is "n1", and a title-case lookup gave nitrogen for it
    assert at.get_atomic_number("X_n1") == 0
    assert at.get_z_a_nucname("X_n1") == (0, 1)
    assert at.get_z_a_nucname("n1") == (0, 1)
    # a symbol is not case sensitive, thus every other "n" is nitrogen
    assert at.get_atomic_number("n") == 7
    assert at.get_atomic_number("N") == 7
    assert at.get_atomic_number("X_N14") == 7
    assert at.get_z_a_nucname("X_N14") == (7, 14)
    assert at.get_z_a_nucname("n14") == (7, 14)
    assert at.get_ion_tuple("nII") == (7, 2)

    # a symbol that the caller gave in the wrong case still resolves
    assert at.get_atomic_number("fe") == 26

    assert at.get_elsymbol(26) == "Fe"
    assert at.get_elsymbol(28) == "Ni"
    assert at.get_elsymbol(1) == "H"
    assert at.get_elsymbol(2) == "He"


def test_decode_roman_numeral() -> None:
    assert at.decode_roman_numeral("I") == 1
    assert at.decode_roman_numeral("II") == 2
    assert at.decode_roman_numeral("III") == 3
    assert at.decode_roman_numeral("IV") == 4
    assert at.decode_roman_numeral("V") == 5
    assert at.decode_roman_numeral("X") == 10
    assert at.decode_roman_numeral("XX") == 20
    assert at.decode_roman_numeral("i") == 1  # case-insensitive
    assert at.decode_roman_numeral("INVALID") == -1


def test_get_ionstring() -> None:
    assert at.get_ionstring(26, 2) == "Fe II"
    assert at.get_ionstring(26, 1) == "Fe I"
    assert at.get_ionstring(28, 3) == "Ni III"
    assert at.get_ionstring(26, 2, sep="") == "FeII"
    assert at.get_ionstring(26, None) == "Fe"
    assert at.get_ionstring(26, "ALL") == "Fe"
    assert at.get_ionstring(26, 2, style="charge") == "Fe+"
    assert at.get_ionstring(26, 3, style="charge") == "Fe2+"
    assert at.get_ionstring(26, 1, style="charge") == "Fe0"


def test_get_ion_tuple() -> None:
    assert at.get_ion_tuple("nnelement_I") == 53
    assert at.get_ion_tuple("nnion_I_II") == (53, 2)
    assert at.get_ion_tuple("Fe_II") == (26, 2)
    assert at.get_ion_tuple("Fe II") == (26, 2)
    assert at.get_ion_tuple("Fe I") == (26, 1)
    assert at.get_ion_tuple("Ni III") == (28, 3)
    assert at.get_ion_tuple("Co II") == (27, 2)
    assert at.get_ion_tuple("Ni") == 28
    assert at.get_ion_tuple("26") == 26


def test_get_ion_tuple_no_separator() -> None:
    """Two-letter symbols must not be split on their first letter, e.g. 'FeII' is Fe II and not F + 'eII'."""
    assert at.get_ion_tuple("FeII") == (26, 2)
    assert at.get_ion_tuple("CoII") == (27, 2)
    assert at.get_ion_tuple("NiIII") == (28, 3)
    assert at.get_ion_tuple("HeII") == (2, 2)
    # single-letter symbols still work, and are not shadowed by a longer symbol that starts the same way
    assert at.get_ion_tuple("FII") == (9, 2)
    assert at.get_ion_tuple("CIV") == (6, 4)
    assert at.get_ion_tuple("OI") == (8, 1)


def test_get_ion_tuple_rejects_missing_element() -> None:
    """A string with a separator but no element symbol must raise rather than return atomic number -1."""
    for badionstr in (" II", "_II", "notanion", "Fe "):
        with pytest.raises(ValueError, match="Could not parse ionstr"):
            at.get_ion_tuple(badionstr)
