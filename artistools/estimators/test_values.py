import io
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import pytest

import artistools as at

modelpath_classic_3d = at.get_path("testdata") / "test-classicmode_3d"


def test_values_of_one_cell_agree_with_the_estimators() -> None:
    """The values command must give the raw estimators of the cell, and not a mean over cells or bins."""
    from artistools.estimators.values import get_values_table

    dfvalues = get_values_table(modelpath_classic_3d, ["Te", "TR", "nne", "rho"], timesteps=[29], modelgridindex=[23])
    assert dfvalues.columns == [
        "timestep",
        "time_days",
        "modelgridindex",
        "vel_r_mid_kmps",
        "Te[K]",
        "TR[K]",
        "nne[e^-/cm^3]",
        "rho[g/cm^3]",
    ]
    assert dfvalues.height == 1

    expected = (
        at
        .scan_estimators(modelpath_classic_3d, modelgridindex=23, timestep=29, join_modeldata=True)
        .select("tmid_days", "vel_r_mid", "Te", "TR", "nne", "rho")
        .collect()
        .row(0, named=True)
    )
    row = dfvalues.row(0, named=True)
    assert row["timestep"] == 29
    assert row["modelgridindex"] == 23
    assert np.isclose(row["time_days"], at.get_timestep_times(modelpath_classic_3d, loc="mid")[29], rtol=1e-6)
    assert np.isclose(row["vel_r_mid_kmps"], expected["vel_r_mid"] / at.constants.km_to_cm, rtol=1e-6)
    for variable, column in (("Te", "Te[K]"), ("TR", "TR[K]"), ("nne", "nne[e^-/cm^3]"), ("rho", "rho[g/cm^3]")):
        assert np.isclose(row[column], expected[variable], rtol=1e-6, atol=0.0), variable


def test_values_of_a_range_of_cells_give_one_row_for_each_cell_and_timestep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A range of cells and timesteps gives one row for each pair, and a cell with no matter gives a warning.

    The cells 12, 13, and 16 of the 3D model hold no matter, thus ARTIS wrote no estimators for them.
    """
    from artistools.estimators.values import main

    outputfile = tmp_path / "values.txt"
    main(argsraw=["Te", "nne", "-ts", "28-29", "-cell", "12-16", "-o", str(outputfile), str(modelpath_classic_3d)])

    assert "modelgridindex 12-13,16 has no estimators" in capsys.readouterr().err

    dfwritten = pl.read_csv(outputfile, separator=" ", comment_prefix="#")
    assert dfwritten.select("timestep", "modelgridindex").rows() == [(28, 14), (28, 15), (29, 14), (29, 15)]

    dfexpected = (
        at
        .scan_estimators(modelpath_classic_3d, modelgridindex=[14, 15], timestep=[28, 29])
        .select("timestep", "modelgridindex", "Te", "nne")
        .sort("timestep", "modelgridindex")
        .collect()
    )
    assert np.allclose(dfwritten["Te[K]"].to_numpy(), dfexpected["Te"].to_numpy(), rtol=1e-6, atol=0.0)
    assert np.allclose(dfwritten["nne[e^-/cm^3]"].to_numpy(), dfexpected["nne"].to_numpy(), rtol=1e-6, atol=0.0)

    # the header of the file tells how the command made the table
    header = [line for line in outputfile.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert any("with no average" in line for line in header)
    assert any(line.startswith("# command:") for line in header)


def test_values_give_the_time_of_volume_prevtimestep(
    tmp_path: Path, read_data_file: Callable[[Path], pl.DataFrame]
) -> None:
    """The method lines gave the mid time of the row for volume_prevtimestep, which takes the timestep before."""
    from artistools.estimators.values import main

    outputfile = tmp_path / "values.txt"
    main(
        argsraw=[
            "volume",
            "volume_prevtimestep",
            "-ts",
            "0-1",
            "-cell",
            "14",
            "-o",
            str(outputfile),
            str(modelpath_classic_3d),
        ]
    )

    text = outputfile.read_text(encoding="utf-8")
    assert (
        "volume_prevtimestep = init_volume (t_prev / t_model)^3, where t_prev is the mid time of the timestep before"
        in text
    )
    dfwritten = read_data_file(outputfile).sort("timestep")
    volumecolumn = next(column for column in dfwritten.columns if column.startswith("volume["))
    prevcolumn = next(column for column in dfwritten.columns if column.startswith("volume_prevtimestep"))
    assert dfwritten[prevcolumn].to_list()[0] is None
    assert np.isclose(dfwritten[prevcolumn][1], dfwritten[volumecolumn][0], rtol=1e-6)


def test_values_ask_about_the_overwrite_before_they_read_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The command found the model folder before the question, e.g. through ssh for a remote model."""
    from artistools.estimators import values

    outputfile = tmp_path / "values.txt"
    outputfile.write_text("old\n", encoding="utf-8")
    with (
        mock.patch.object(values, "get_model_folder", side_effect=AssertionError("the model was read")),
        mock.patch("sys.stdin.isatty", return_value=False),
        pytest.raises(SystemExit),
    ):
        values.main(argsraw=["Te", "-ts", "28", "-o", str(outputfile), str(modelpath_classic_3d)])
    assert "did not overwrite the files" in capsys.readouterr().err


def test_values_take_a_file_of_the_model_as_the_model_path(tmp_path: Path) -> None:
    """A model path can name a file of the model, and the command stopped because the file holds no input.txt."""
    from artistools.estimators.values import main

    outputfile = tmp_path / "values.txt"
    main(argsraw=["Te", "-ts", "28", "-cell", "14", "-o", str(outputfile), str(modelpath_classic_3d / "input.txt")])
    dfwritten = pl.read_csv(outputfile, separator=" ", comment_prefix="#")
    assert dfwritten.select("timestep", "modelgridindex").rows() == [(28, 14)]


def test_values_warn_about_a_cell_with_no_estimators_at_one_timestep(capsys: pytest.CaptureFixture[str]) -> None:
    """A cell that has estimators at one selected timestep and none at a second timestep gives a warning.

    The check tested the cells and the timesteps apart, thus each cell and each timestep had a row and no warning came.
    """
    from artistools.estimators.values import warn_missing_rows

    dfvalues = pl.DataFrame({"timestep": [0, 1, 0], "modelgridindex": [0, 0, 1], "Te[K]": [1e4, 1e4, 1e4]})
    warn_missing_rows(dfvalues, [0, 1], modelgridindex=None)

    assert "modelgridindex 1 has no estimators at timestep 1" in capsys.readouterr().err


def test_values_warn_about_a_requested_cell_in_the_direction_range(capsys: pytest.CaptureFixture[str]) -> None:
    """-dirbin with -cell warns about a requested cell in the range that has no estimators, e.g. a cell with no matter.

    The warning took only the cells of the table, thus such a cell gave no warning. Cell 932 is near the pole and holds
    no matter. Cell 402 is near the equator, thus it is outside the range, and it gives no warning.
    """
    from artistools.estimators.values import main

    main(argsraw=["Te", "-ts", "25", "-dirbin", "90", "-cell", "402,644,932", str(modelpath_classic_3d)])

    stderr = capsys.readouterr().err
    assert "modelgridindex 932 has no estimators" in stderr
    assert "402" not in stderr


def test_values_with_quiet_print_only_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    """--quiet must leave only the table on the standard output, thus a script can read it directly."""
    import artistools.__main__

    artistools.__main__.main(argsraw=["values", "Te", "-t", "7.8", "-cell", "14", "--quiet", str(modelpath_classic_3d)])

    dfprinted = pl.read_csv(io.StringIO(capsys.readouterr().out), separator=" ")
    assert dfprinted.columns == ["timestep", "time_days", "modelgridindex", "vel_r_mid_kmps", "Te[K]"]
    assert dfprinted.select("timestep", "modelgridindex").rows() == [(29, 14)]


def test_values_of_an_unknown_variable_name_the_listing(capsys: pytest.CaptureFixture[str]) -> None:
    """A variable that the model does not hold stops the command, and the help names a near name and the listing."""
    from artistools.estimators.values import main

    with pytest.raises(SystemExit) as excinfo:
        main(argsraw=["Tee", "-t", "7.8", str(modelpath_classic_3d)])
    assert excinfo.value.code == 1

    stderr = capsys.readouterr().err
    assert "'Tee' is not a variable" in stderr
    assert "Did you mean Te?" in stderr
    assert "--listvariables" in stderr
