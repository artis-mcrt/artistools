"""Tests of the Rust extension artistools.rustext."""

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

    # a cut inside the last line leaves a line that parses, e.g. a collision strength of 7.6 for 7.65e-01. ARTIS ends
    # each line with a newline, thus the missing newline shows the cut
    transitionsfile.write_text("26 2 2\n1 2 1.0 2.0 1\n1 3 1.0 7.6", encoding="utf-8")
    with pytest.raises(
        Exception, match=r"transitiondata\.txt: the file ends after 1 of the 2 transitions of Z=26 ion_stage=2"
    ):
        at.rustext.read_transitiondata(transitionsfile, ionlist=ionlist)


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


def test_sum_binned_line_opacities_gives_no_opacity_below_one_kelvin() -> None:
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
