"""Tests of the Rust extension artistools.rustext."""

import re
from pathlib import Path

import numpy as np
import polars as pl
import polars.testing as pltest
import pytest

import artistools as at
from artistools.estimators.test_estimators import get_cell_texts

modelpath = at.get_path("testdata") / "testmodel"


def test_estimparse_drops_a_cell_that_a_cut_file_ends_inside(tmp_path: Path) -> None:
    """The file of a rank can end inside a cell when a job stops during the write.

    The reader of the file of a rank kept such a cell, with a zero for each value after the cut. The cache thus held
    zeros that looked like real values. The reader must give the rows of the complete cells only.
    """
    celltexts = get_cell_texts((modelpath / "estimators_0000.out").read_text(encoding="utf-8"))
    (tmp_path / "estimators_0000.out").write_text("".join(celltexts), encoding="utf-8")
    dfcomplete = at.rustext.estimparse(tmp_path, 0, 0)
    assert dfcomplete.height == len(celltexts) == 100

    lastcell = celltexts[-1]
    lastpopulations = lastcell.rindex("populations")
    cutcells = {
        "inside a value": lastcell[:lastpopulations] + "populations    Z=28                2: 2.6e-07  3:",
        # the last line ends with "adiabatic 5.00034e-13", and the cut leaves "5.00034e-1", which parses as a number
        "inside a mantissa": lastcell[: lastcell.rindex("e-13") + 3],
        "at the end of a line": lastcell[:lastpopulations],
        "after the cell header": lastcell[: lastcell.index("\n") + 1],
    }
    for cutname, cutcell in cutcells.items():
        (tmp_path / "estimators_0000.out").write_text("".join(celltexts[:-1]) + cutcell, encoding="utf-8")
        dfrank = at.rustext.estimparse(tmp_path, 0, 0)
        assert dfrank.height == 99, cutname
        pltest.assert_frame_equal(dfrank, dfcomplete.head(99).select(dfrank.columns))


def test_estimparse_gives_no_row_for_a_text_with_no_complete_cell(tmp_path: Path) -> None:
    """A text that ends inside the first cell gives a frame with no column, as the file of all ranks does."""
    (tmp_path / "estimators_0000.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 3000 W 1 TJ 2000 nne 1.0e5\n", encoding="utf-8"
    )
    dfrank = at.rustext.estimparse(tmp_path, 0, 0)
    assert dfrank.shape == (0, 0)


def test_estimparse_gives_the_index_columns_for_a_text_of_empty_cells(tmp_path: Path) -> None:
    """A text whose cells are all empty is complete, thus it gives the index columns and no row.

    Such a text gave a frame with no column, as a cut text does. The conversion then gave a warning about a cut text.
    """
    emptycells = "timestep 0 modelgridindex 0 EMPTYCELL\n\ntimestep 0 modelgridindex 1 EMPTYCELL\n\n"
    (tmp_path / "estimators_0000.out").write_text(emptycells, encoding="utf-8")
    (tmp_path / "estimators_allranks.out").write_text(emptycells, encoding="utf-8")
    dfrank = at.rustext.estimparse(tmp_path, 0, 0)
    dfallranks = at.rustext.estimparse_allranks(tmp_path / "estimators_allranks.out")
    for dfempty in (dfrank, dfallranks):
        assert dfempty.height == 0
        assert dict(dfempty.schema) == {"timestep": pl.Int32, "modelgridindex": pl.Int32}


def test_estimparse_rejects_a_rank_range_with_no_rank() -> None:
    """A rankmin above rankmax gave an empty concatenation, which panicked in polars with a BaseException."""
    with pytest.raises(Exception, match="the rank range 1 to 0 holds no rank"):
        at.rustext.estimparse(modelpath, 1, 0)


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
        at.rustext.read_transitiondata(transitionsfile, ionlist=ionlist)

    # a complete table of the same ion is not an error
    transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 2.0 1\n", encoding="utf-8")
    transitionsdict = at.rustext.read_transitiondata(transitionsfile, ionlist=ionlist)
    assert [df.height for df in transitionsdict.values()] == ([] if ionlist == {(27, 2)} else [2])

    # ARTIS reads a complete file whose last line has no newline. The reader rejected such a file as cut
    transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 2.0 1", encoding="utf-8")
    transitionsdict = at.rustext.read_transitiondata(transitionsfile, ionlist=ionlist)
    assert [df.height for df in transitionsdict.values()] == ([] if ionlist == {(27, 2)} else [2])

    # a cut inside the last line of a kept table leaves too few numbers for the format of the table
    if ionlist != {(27, 2)}:
        transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 7.6", encoding="utf-8")
        with pytest.raises(Exception, match=r"transitiondata\.txt:3: line ended where a forbidden flag was expected"):
            at.rustext.read_transitiondata(transitionsfile, ionlist=ionlist)


def test_read_transitiondata_reads_the_legacy_format_of_four_columns(tmp_path: Path) -> None:
    """ARTIS takes the format of each table from its first line, and 4 numbers give "index lower upper A".

    The reader took such a table as "lower upper A collstr", thus it gave wrong levels and A values with no error.
    ARTIS gives a transition of this format the collision strength -1 and the forbidden flag 0.
    """
    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text(
        "26 2 3\n1 1 2 1.5e8\n2 1 3 2.5e7\n3 2 3 4.0e6\n\n26 3 1\n1 2 2.0 0.5 1\n", encoding="utf-8"
    )
    transitionsdict = at.rustext.read_transitiondata(transitionsfile)
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

    # ARTIS reads no other count of columns
    transitionsfile.write_text("26 2 1\n1 2 1.0\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:2: the first line of a table has 3 numbers"):
        at.rustext.read_transitiondata(transitionsfile)


def test_read_transitiondata_errors_name_the_file_and_the_line(tmp_path: Path) -> None:
    """A parse error names the file and the line. The test model file has 1.9 million lines."""
    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text("26 2 3\n1 2 1.0 2.0 1\n1 3 1.0e-3x 2.0 1\n2 3 1.0 2.0 1\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:3: could not parse \"1\.0e-3x\" as an A value"):
        at.rustext.read_transitiondata(transitionsfile)

    transitionsfile.write_text("26 2 3\n1 2 1.0 2.0 1\n\n2 3 1.0 2.0 1\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:3: line ended where a lower level was expected"):
        at.rustext.read_transitiondata(transitionsfile)

    transitionsfile.write_text("26 x 3\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"transitiondata\.txt:1: could not parse \"x\" as an ion stage"):
        at.rustext.read_transitiondata(transitionsfile)


def test_level_numbers_that_start_at_zero(tmp_path: Path) -> None:
    """ARTIS takes the number of the first level, 0 or 1, from adata.txt, and it applies it to all three files.

    The readers took the numbering from 1. Thus a zero-based transition gave the lower level -1, a single target
    level 0 of a photoionisation table started a list of targets, and adata.txt gave a bare AssertionError.
    """
    from artistools.atomic.core import parse_adata
    from artistools.atomic.core import parse_phixsdata

    transitionsfile = tmp_path / "transitiondata.txt"
    transitionsfile.write_text("26 1 2\n0 1 1.0 2.0 1\n0 2 1.0 2.0 1\n", encoding="utf-8")
    dftransitions = at.rustext.read_transitiondata(transitionsfile, firstlevelnumber=0)[26, 1]
    assert dftransitions["lower"].to_list() == [0, 0]
    assert dftransitions["upper"].to_list() == [1, 2]
    with pytest.raises(Exception, match="ARTIS numbers the levels from 0 or from 1, not from 2"):
        at.rustext.read_transitiondata(transitionsfile, firstlevelnumber=2)

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
    with adatafile.open(encoding="utf-8") as fadata, pytest.raises(ValueError, match="numbers the levels from 0"):
        list(parse_adata(fadata, {}, None))

    adatafile.write_text("26 1 3 7.9\n1 0.0 9.0 1 ground\n2 1.5 7.0 1 first\n4 2.5 5.0 1 second\n", encoding="utf-8")
    with (
        adatafile.open(encoding="utf-8") as fadata,
        pytest.raises(ValueError, match="level 3 of the ion has the number 4"),
    ):
        list(parse_adata(fadata, {}, None))


def test_estimparse_roman_numerals_agree_with_the_python_table(tmp_path: Path) -> None:
    """The ion columns of the Rust reader must have the names that get_ionstring gives.

    The Rust table ended at XVI, and the Python table ended at XX. Thus an ion stage from 17 to 20 in an estimator file
    stopped the parse of the whole file.
    """
    stages = range(1, len(at.atomic.roman_numerals))
    assert stages.stop > 20
    (tmp_path / "estimators_0000.out").write_text(
        "timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 1.0\n"
        "populations    Z=26 " + " ".join(f"{stage}: {stage}.0" for stage in stages) + "\n\n",
        encoding="utf-8",
    )
    dfrank = at.rustext.estimparse(tmp_path, 0, 0)
    for stage in stages:
        assert dfrank[f"nnion_{at.get_ionstring(26, stage, sep='_')}"].item() == pytest.approx(float(stage))

    # a stage beyond both tables is an error in both
    (tmp_path / "estimators_0000.out").write_text(
        f"timestep 0 modelgridindex 0 TR 2000 Te 2000 W 1 TJ 2000 nne 1.0\npopulations    Z=26 {stages.stop}: 1.0\n\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match=f"no roman numeral for ion stage {stages.stop}"):
        at.rustext.estimparse(tmp_path, 0, 0)
    with pytest.raises(IndexError):
        at.get_ionstring(26, stages.stop)


def test_sum_binned_line_opacities_gives_no_opacity_for_a_temperature_of_zero_or_below() -> None:
    """A cell with a T_exc of zero or below has no LTE populations, thus each of its sums is zero and not NaN.

    A T_exc of zero gave 0 / 0 for the ground level, and the NaN then spread through the mass-weighted mean.
    """
    from artistools.rustext import sum_binned_line_opacities

    dflevels = pl.DataFrame(
        {"ionindex": [0, 0], "g": [1.0, 3.0], "energy_ev": [0.0, 1.0]}, schema_overrides={"ionindex": pl.UInt32}
    )
    dflines = pl.DataFrame(
        {
            "lambda_angstroms_binindex": [0],
            "lower": [0],
            "upper": [1],
            "sobolev_lower": [1e-3],
            "sobolev_upper": [1e-3],
            "lambda_angstroms": [1.0],
        },
        schema_overrides={"lambda_angstroms_binindex": pl.UInt32, "lower": pl.UInt32, "upper": pl.UInt32},
    )
    dfcells = pl.DataFrame({"T_exc": [0.0, -1.0, 5000.0], "nnion_0": [1.0, 1.0, 1.0]})
    dfsums = sum_binned_line_opacities(dflevels, dflines, dfcells, ["nnion_0"], 1, at.constants.K_B_ev_per_K)
    for column in at.ejectaopacity.OPACITYCOLUMNS:
        sums = dfsums[column].to_numpy()
        assert np.all(np.isfinite(sums))
        assert sums[0] == pytest.approx(0.0)
        assert sums[1] == pytest.approx(0.0)
        assert sums[2] > 0.0


def test_expansion_opacities_give_zero_for_a_cell_with_no_temperature() -> None:
    """A null T_exc and a T_exc of zero give a zero expansion opacity, and a positive T_exc gives a finite value."""
    timestep = 40
    time_days = at.get_timestep_times(modelpath)[timestep]
    dfcell = at.ejectaopacity.get_cell_estimators(modelpath, timestep, None, "Te")
    assert dfcell.height == 1
    edges = at.ejectaopacity.get_lambda_bin_edges(3000.0, 4000.0, 20.0)
    lines = at.ejectaopacity.get_opacity_lines(
        at.ejectaopacity.get_opacity_atomic_data(modelpath), dfcell.columns, edges, time_days
    )
    dfcells = pl.concat([dfcell, dfcell, dfcell]).with_columns(
        modelgridindex=pl.Series([0, 1, 2], dtype=dfcell["modelgridindex"].dtype),
        T_exc=pl.Series([None, 0.0, dfcell["T_exc"].item()], dtype=pl.Float64),
    )
    dfbins = at.ejectaopacity.get_expansion_opacities(lines, dfcells, edges, time_days)
    for column in at.ejectaopacity.OPACITYCOLUMNS:
        values = dfbins.group_by("modelgridindex", maintain_order=True).agg(pl.col(column).abs().sum())[column]
        assert values[0] == pytest.approx(0.0)
        assert values[1] == pytest.approx(0.0)
        assert np.isfinite(values[2])
        assert values[2] > 0.0


def test_estimtimesteps_ignores_a_header_line_that_a_cut_file_ends_inside(tmp_path: Path) -> None:
    """The scan of the timesteps takes a cell header only when a newline ends its line.

    A text cut right after "timestep " raised an error, and a cut of "timestep 11" to "timestep 1" named timestep 1,
    which a restarted run then took as the repeated timestep of the restart. A complete header line still names its
    timestep, because the job wrote that timestep.
    """
    celltexts = get_cell_texts((modelpath / "estimators_0000.out").read_text(encoding="utf-8"))
    estimfile = tmp_path / "estimators_0000.out"
    estimfile.write_text("".join(celltexts), encoding="utf-8")
    alltimesteps = at.rustext.estimtimesteps(estimfile)
    assert alltimesteps == sorted(set(at.rustext.estimparse(tmp_path, 0, 0)["timestep"].to_list()))
    lasttimestep = int(celltexts[-1].split()[1])
    assert lasttimestep not in {int(celltext.split()[1]) for celltext in celltexts[:-1]}

    for cut in ("timestep ", "timestep 1", celltexts[-1][: celltexts[-1].index("\n")]):
        estimfile.write_text("".join(celltexts[:-1]) + cut, encoding="utf-8")
        assert at.rustext.estimtimesteps(estimfile) == [ts for ts in alltimesteps if ts != lasttimestep], cut

    estimfile.write_text("timestep 7 modelgridindex 0\n", encoding="utf-8")
    assert at.rustext.estimtimesteps(estimfile) == [7]

    # a complete header line with a bad timestep is an error, as in the parser
    estimfile.write_text("timestep x modelgridindex 0 TR 1\n\n", encoding="utf-8")
    with pytest.raises(Exception, match=r"estimators_0000\.out:1: could not parse \"x\" as an integer"):
        at.rustext.estimtimesteps(estimfile)


def test_estimator_io_errors_name_the_file_and_the_line(tmp_path: Path) -> None:
    """An I/O error, e.g. a byte that is not UTF-8, names the file and the line.

    pyo3-polars gives Python only the text of the io::Error, thus the error named no file. A batch has many ranks.
    """
    estimfile = tmp_path / "estimators_0000.out"
    estimfile.write_bytes(b"timestep 0 modelgridindex 0 TR 2000 Te 3000\n\ntimestep 1 modelgridindex 0 TR 2\xff\n\n")
    for read in (
        lambda: at.rustext.estimparse(tmp_path, 0, 0),
        lambda: at.rustext.estimparse_allranks(estimfile),
        lambda: at.rustext.estimtimesteps(estimfile),
    ):
        with pytest.raises(OSError, match=r"estimators_0000\.out:3: stream did not contain valid UTF-8"):
            read()

    with pytest.raises(OSError, match=r"estimators_allranks\.out: "):
        at.rustext.estimparse_allranks(tmp_path / "estimators_allranks.out")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("timestep 3 TR 2000 Te 3000\n\n", "estimators_0000.out:1: the cell header gives no modelgridindex"),
        ("populations Z=26 1: 1.0\n\n", "estimators_0000.out:1: the line gives a value, but no cell header"),
    ],
)
def test_estimparse_rejects_a_cell_with_no_index(tmp_path: Path, text: str, message: str) -> None:
    """A cell header with no modelgridindex gave a ShapeError with no file, or the modelgridindex 0 with no message.

    A value before the first cell header gave the misleading error "a column was given two values for one cell".
    """
    (tmp_path / "estimators_0000.out").write_text(text, encoding="utf-8")
    with pytest.raises(Exception, match=re.escape(message)):
        at.rustext.estimparse(tmp_path, 0, 0)


def test_bin_kernels_take_a_null_as_no_bin() -> None:
    """A null value, weight, or group is in no bin, as a NaN value is.

    The kernels read the columns as one slice, and a null gave the error "chunked array is not contiguous". The last
    row of a cut file of virtual packets can hold a null energy.
    """
    from artistools.rustext import get_bin_indices
    from artistools.rustext import sum_weights_in_bins

    edges = [0.0, 1.0, 2.0]
    df = pl.DataFrame({"x": [0.5, None, 1.5, 1.5, 0.5], "e": [1.0, 2.0, None, 4.0, 8.0], "group": [0, 0, 0, None, 1]})
    for dtype in (pl.Float32, pl.Float64):
        dftyped = df.with_columns(pl.col("x").cast(dtype), pl.col("group").cast(pl.Int32))
        assert get_bin_indices(dftyped, "x", edges)["binindex"].to_list() == [0, -1, 1, 1, 0]
        dfsums = sum_weights_in_bins(dftyped, "x", "e", edges)
        assert dfsums["count"].to_list() == [2, 1]
        assert np.allclose(dfsums["sum"].to_numpy(), [9.0, 4.0])
        dfgroups = sum_weights_in_bins(dftyped, "x", "e", edges, "group", 2)
        assert dfgroups["count"].to_list() == [1, 0, 1, 0]
        assert np.allclose(dfgroups["sum"].to_numpy(), [1.0, 0.0, 8.0, 0.0])


def test_sum_weights_in_bins_with_more_bins_than_values_in_a_part() -> None:
    """The sums of many bins agree with numpy, also when the bins outnumber the values of a part.

    Each part held one sum for each bin, and 62 parts of 16500 bins of 100 groups needed 2.5 GB of memory. A part now
    holds at least one value for each sum, and the threads sum one wave of parts at a time.
    """
    from artistools.rustext import sum_weights_in_bins

    rng = np.random.default_rng(0)
    valuecount, ngroups, nbins = 300_000, 4, 30_000
    df = pl.DataFrame({
        "x": rng.uniform(0.0, 1.0, valuecount),
        "e": rng.uniform(0.0, 1.0, valuecount),
        "group": rng.integers(0, ngroups, valuecount, dtype=np.int32),
    })
    edges = np.linspace(0.0, 1.0, nbins + 1)
    dfsums = sum_weights_in_bins(df, "x", "e", edges.tolist(), "group", ngroups, sumsquares=True)
    bins = np.clip(np.searchsorted(edges, df["x"].to_numpy(), side="right") - 1, 0, nbins - 1)
    flatindex = df["group"].to_numpy() * nbins + bins
    weights = df["e"].to_numpy()
    assert np.array_equal(dfsums["count"].to_numpy(), np.bincount(flatindex, minlength=ngroups * nbins))
    assert np.allclose(dfsums["sum"].to_numpy(), np.bincount(flatindex, weights, minlength=ngroups * nbins))
    assert np.allclose(dfsums["sumsquares"].to_numpy(), np.bincount(flatindex, weights**2, minlength=ngroups * nbins))


def test_opacity_levels_are_the_levels_that_artis_keeps() -> None:
    """The opacities take only the first nlevelsmax levels of each ion from compositiondata.txt, as ARTIS does.

    The partition function summed all 1792 levels of Fe I in adata.txt. ARTIS keeps 500 of them in the classic test
    run, and the sum of all the levels is 1.97 times the sum of the kept levels at 20000 K.
    """
    classicmodelpath = at.get_path("testdata") / "test-classicmode_3d"
    adata = at.ejectaopacity.get_opacity_atomic_data(classicmodelpath)
    dfcomposition = at.get_composition_data(classicmodelpath)
    nlevelsmax = dict(zip(dfcomposition["Z"], dfcomposition["nlevelsmax_readin"], strict=True))
    alllevels = at.atomic.get_levels(classicmodelpath)
    for Z, ion_stage, dflevels in adata.select("Z", "ion_stage", "levels").iter_rows():
        levelcount = alllevels.filter(pl.col("Z") == Z, pl.col("ion_stage") == ion_stage)["levels"].item().height
        assert dflevels.height == min(nlevelsmax[Z], levelcount)
    assert adata.filter(pl.col("Z") == 26, pl.col("ion_stage") == 1)["levels"].item().height == 500

    # the test model keeps all the levels, because its compositiondata.txt gives -1
    assert at.get_composition_data(modelpath)["nlevelsmax_readin"].to_list() == [-1, -1]
    testmodeladata = at.ejectaopacity.get_opacity_atomic_data(modelpath)
    pltest.assert_series_equal(
        testmodeladata.select(pl.col("levels").map_elements(len, return_dtype=pl.Int64)).to_series(),
        at.atomic.get_levels(modelpath).select(pl.col("levels").map_elements(len, return_dtype=pl.Int64)).to_series(),
    )


def test_opacity_lines_name_an_ion_with_no_transition_table() -> None:
    """An ion with no table in transitiondata.txt gives a clear error and not ColumnNotFoundError."""
    adata = at.ejectaopacity.get_opacity_atomic_data(modelpath)
    adata = adata.with_columns(pl.Series("transitions", [pl.LazyFrame() for _ in range(adata.height)], dtype=pl.Object))
    with pytest.raises(ValueError, match=r"transitiondata\.txt holds no table for Fe_II"):
        at.ejectaopacity.get_opacity_lines(adata, ["nnion_Fe_II"], [3000.0, 4000.0], 10.0)
